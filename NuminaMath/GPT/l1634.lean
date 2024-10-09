import Mathlib

namespace negative_only_option_B_l1634_163409

theorem negative_only_option_B :
  (0 > -3) ∧ 
  (|-3| = 3) ∧ 
  (0 < 3) ∧
  (0 < (1/3)) ∧
  ∀ x, x = -3 → x < 0 :=
by
  sorry

end negative_only_option_B_l1634_163409


namespace ratio_triangle_BFD_to_square_ABCE_l1634_163443

-- Defining necessary components for the mathematical problem
def square_ABCE (x : ℝ) : ℝ := 16 * x^2
def triangle_BFD_area (x : ℝ) : ℝ := 7 * x^2

-- The theorem that needs to be proven, stating the ratio of the areas
theorem ratio_triangle_BFD_to_square_ABCE (x : ℝ) (hx : x > 0) :
  (triangle_BFD_area x) / (square_ABCE x) = 7 / 16 :=
by
  sorry

end ratio_triangle_BFD_to_square_ABCE_l1634_163443


namespace order_of_a_b_c_l1634_163410

noncomputable def a : ℝ := (Real.log 5) / 5
noncomputable def b : ℝ := 1 / Real.exp 1
noncomputable def c : ℝ := (Real.log 4) / 4

theorem order_of_a_b_c : a < c ∧ c < b := by
  sorry

end order_of_a_b_c_l1634_163410


namespace total_litter_weight_l1634_163436

-- Definitions of the conditions
def gina_bags : ℕ := 2
def neighborhood_multiplier : ℕ := 82
def bag_weight : ℕ := 4

-- Representing the total calculation
def neighborhood_bags : ℕ := neighborhood_multiplier * gina_bags
def total_bags : ℕ := neighborhood_bags + gina_bags

def total_weight : ℕ := total_bags * bag_weight

-- Statement of the problem
theorem total_litter_weight : total_weight = 664 :=
by
  sorry

end total_litter_weight_l1634_163436


namespace quadratic_identity_l1634_163449

theorem quadratic_identity
  (a b c x : ℝ) (hab : a ≠ b) (hac : a ≠ c) (hbc : b ≠ c) :
  (a^2 * (x - b) * (x - c) / ((a - b) * (a - c))) +
  (b^2 * (x - a) * (x - c) / ((b - a) * (b - c))) +
  (c^2 * (x - a) * (x - b) / ((c - a) * (c - b))) =
  x^2 :=
sorry

end quadratic_identity_l1634_163449


namespace min_value_functions_l1634_163403

noncomputable def f_A (x : ℝ) : ℝ := x^2 + 1 / x^2
noncomputable def f_B (x : ℝ) : ℝ := 2 * x + 2 / x
noncomputable def f_C (x : ℝ) : ℝ := (x - 1) / (x + 1)
noncomputable def f_D (x : ℝ) : ℝ := Real.log (Real.sqrt x + 1)

theorem min_value_functions :
  (∃ x : ℝ, ∀ y : ℝ, f_A x ≤ f_A y) ∧
  (∃ x : ℝ, ∀ y : ℝ, f_D x ≤ f_D y) ∧
  ¬ (∃ x : ℝ, ∀ y : ℝ, f_B x ≤ f_B y) ∧
  ¬ (∃ x : ℝ, ∀ y : ℝ, f_C x ≤ f_C y) :=
by
  sorry

end min_value_functions_l1634_163403


namespace find_common_difference_find_max_sum_find_max_n_l1634_163481

-- Condition for the sequence
def is_arithmetic_sequence (a : ℕ → ℤ) (d : ℤ) : Prop :=
∀ n : ℕ, a (n + 1) = a n + d

-- Problem statement (1): Find the common difference
theorem find_common_difference (a : ℕ → ℤ) (d : ℤ) (h1 : a 1 = 23)
  (h2 : is_arithmetic_sequence a d)
  (h6 : a 6 > 0)
  (h7 : a 7 < 0) : d = -4 :=
sorry

-- Problem statement (2): Find the maximum value of the sum S₆
theorem find_max_sum (d : ℤ) (h : d = -4) : 6 * 23 + (6 * 5 / 2) * d = 78 :=
sorry

-- Problem statement (3): Find the maximum value of n when S_n > 0
theorem find_max_n (d : ℤ) (h : d = -4) : ∀ n : ℕ, (n > 0 ∧ (23 * n + (n * (n - 1) / 2) * d > 0)) → n ≤ 12 :=
sorry

end find_common_difference_find_max_sum_find_max_n_l1634_163481


namespace inequality_system_solution_l1634_163417

theorem inequality_system_solution (x : ℝ) :
  (3 * x > x + 6) ∧ ((1 / 2) * x < -x + 5) ↔ (3 < x) ∧ (x < 10 / 3) :=
by
  sorry

end inequality_system_solution_l1634_163417


namespace Scarlett_adds_correct_amount_l1634_163434

-- Define the problem with given conditions
def currentOilAmount : ℝ := 0.17
def desiredOilAmount : ℝ := 0.84

-- Prove that the amount of oil Scarlett needs to add is 0.67 cup
theorem Scarlett_adds_correct_amount : (desiredOilAmount - currentOilAmount) = 0.67 := by
  sorry

end Scarlett_adds_correct_amount_l1634_163434


namespace half_abs_diff_squares_l1634_163491

theorem half_abs_diff_squares (a b : ℝ) (h₁ : a = 25) (h₂ : b = 20) :
  (1 / 2) * |a^2 - b^2| = 112.5 :=
sorry

end half_abs_diff_squares_l1634_163491


namespace impossible_triangle_angle_sum_l1634_163442

theorem impossible_triangle_angle_sum (x y z : ℝ) (h : x + y + z = 180) : x + y + z ≠ 360 :=
by
sorry

end impossible_triangle_angle_sum_l1634_163442


namespace range_of_values_l1634_163448

variable (a : ℝ)

-- State the conditions
def prop.false (a : ℝ) : Prop := ¬ ∃ x : ℝ, a * x^2 + 4 * x + a ≤ 0

-- Prove that the range of values for a where the proposition is false is (2, +∞)
theorem range_of_values (ha : prop.false a) : 2 < a :=
sorry

end range_of_values_l1634_163448


namespace negate_exists_statement_l1634_163430

theorem negate_exists_statement : 
  (∃ x : ℝ, x^2 + x - 2 < 0) ↔ ¬ (∀ x : ℝ, x^2 + x - 2 ≥ 0) :=
by sorry

end negate_exists_statement_l1634_163430


namespace david_older_than_rosy_l1634_163493

theorem david_older_than_rosy
  (R D : ℕ) 
  (h1 : R = 12) 
  (h2 : D + 6 = 2 * (R + 6)) : 
  D - R = 18 := 
by
  sorry

end david_older_than_rosy_l1634_163493


namespace second_largest_between_28_and_31_l1634_163458

theorem second_largest_between_28_and_31 : 
  ∃ (n : ℕ), n > 28 ∧ n ≤ 31 ∧ (∀ m, (m > 28 ∧ m ≤ 31 ∧ m < 31) ->  m ≤ 30) :=
sorry

end second_largest_between_28_and_31_l1634_163458


namespace votes_cast_is_750_l1634_163427

-- Define the conditions as Lean statements
def initial_score : ℤ := 0
def score_increase (likes : ℕ) : ℤ := likes
def score_decrease (dislikes : ℕ) : ℤ := -dislikes
def observed_score : ℤ := 150
def percent_likes : ℚ := 0.60

-- Express the proof
theorem votes_cast_is_750 (total_votes : ℕ) (likes : ℕ) (dislikes : ℕ) 
  (h1 : total_votes = likes + dislikes) 
  (h2 : percent_likes * total_votes = likes) 
  (h3 : dislikes = (1 - percent_likes) * total_votes)
  (h4 : observed_score = score_increase likes + score_decrease dislikes) :
  total_votes = 750 := 
sorry

end votes_cast_is_750_l1634_163427


namespace log_ordering_l1634_163499

noncomputable def a : ℝ := Real.log 6 / Real.log 3
noncomputable def b : ℝ := Real.log 8 / Real.log 4
noncomputable def c : ℝ := Real.log 10 / Real.log 5

theorem log_ordering : a > b ∧ b > c :=
by {
  sorry
}

end log_ordering_l1634_163499


namespace value_of_sum_l1634_163437

theorem value_of_sum (a b : ℝ) (h : a^2 * b^2 + a^2 + b^2 + 1 - 2 * a * b = 2 * a * b) : a + b = 2 ∨ a + b = -2 :=
sorry

end value_of_sum_l1634_163437


namespace solve_equation_l1634_163464

theorem solve_equation (x : ℚ) (h : x ≠ 3) : (x + 5) / (x - 3) = 4 ↔ x = 17 / 3 :=
sorry

end solve_equation_l1634_163464


namespace triangle_square_side_ratio_l1634_163465

theorem triangle_square_side_ratio :
  (∀ (a : ℝ), (a * 3 = 60) → (∀ (b : ℝ), (b * 4 = 60) → (a / b = 4 / 3))) :=
by
  intros a h1 b h2
  sorry

end triangle_square_side_ratio_l1634_163465


namespace star_polygon_x_value_l1634_163424

theorem star_polygon_x_value
  (a b c d e p q r s t : ℝ)
  (h1 : p + q + r + s + t = 500)
  (h2 : a + b + c + d + e = x)
  :
  x = 140 :=
sorry

end star_polygon_x_value_l1634_163424


namespace find_b_l1634_163444

theorem find_b 
  (a b c d : ℚ) 
  (h1 : a = 2 * b + c) 
  (h2 : b = 2 * c + d) 
  (h3 : 2 * c = d + a - 1) 
  (h4 : d = a - c) : 
  b = 2 / 9 :=
by
  -- Proof is omitted (the proof steps would be inserted here)
  sorry

end find_b_l1634_163444


namespace thirty_two_not_sum_consecutive_natural_l1634_163494

theorem thirty_two_not_sum_consecutive_natural (n k : ℕ) : 
  (n > 0) → (32 ≠ (n * (2 * k + n - 1)) / 2) :=
by
  sorry

end thirty_two_not_sum_consecutive_natural_l1634_163494


namespace calculate_total_income_l1634_163433

/-- Total income calculation proof for a person with given distributions and remaining amount -/
theorem calculate_total_income
  (I : ℝ) -- total income
  (leftover : ℝ := 40000) -- leftover amount after distribution and donation
  (c1_percentage : ℝ := 3 * 0.15) -- percentage given to children
  (c2_percentage : ℝ := 0.30) -- percentage given to wife
  (c3_percentage : ℝ := 0.05) -- percentage donated to orphan house
  (remaining_percentage : ℝ := 1 - (c1_percentage + c2_percentage)) -- remaining percentage after children and wife
  (R : ℝ := remaining_percentage * I) -- remaining amount after children and wife
  (donation : ℝ := c3_percentage * R) -- amount donated to orphan house)
  (left_amount : ℝ := R - donation) -- final remaining amount
  (income : ℝ := (leftover / (1 - remaining_percentage * (1 - c3_percentage)))) -- calculation of the actual income
  : I = income := sorry

end calculate_total_income_l1634_163433


namespace intersection_of_A_and_B_l1634_163454

open Set

def A : Set ℝ := {-1, 0, 1}
def B : Set ℝ := {x | x < 0}

theorem intersection_of_A_and_B : A ∩ B = {-1} :=
  sorry

end intersection_of_A_and_B_l1634_163454


namespace a_b_c_at_least_one_not_less_than_one_third_l1634_163445

theorem a_b_c_at_least_one_not_less_than_one_third (a b c : ℝ) (h : a + b + c = 1) :
  ¬ (a < 1/3 ∧ b < 1/3 ∧ c < 1/3) :=
by
  sorry

end a_b_c_at_least_one_not_less_than_one_third_l1634_163445


namespace find_m_when_lines_parallel_l1634_163483

theorem find_m_when_lines_parallel (m : ℝ) :
  (∀ x y : ℝ, x + (1 + m) * y = 2 - m) ∧ (∀ x y : ℝ, 2 * m * x + 4 * y = -16) →
  ∃ m : ℝ, m = 1 :=
sorry

end find_m_when_lines_parallel_l1634_163483


namespace height_percentage_increase_l1634_163438

theorem height_percentage_increase (B A : ℝ) 
  (hA : A = B * 0.8) : ((B - A) / A) * 100 = 25 := by
--   Given the condition that A's height is 20% less than B's height
--   translate into A = B * 0.8
--   We need to show ((B - A) / A) * 100 = 25
sorry

end height_percentage_increase_l1634_163438


namespace product_of_two_numbers_l1634_163402

variable (x y : ℝ)

-- conditions
def condition1 : Prop := x + y = 23
def condition2 : Prop := x - y = 7

-- target
theorem product_of_two_numbers {x y : ℝ} 
  (h1 : condition1 x y) 
  (h2 : condition2 x y) : 
  x * y = 120 := 
sorry

end product_of_two_numbers_l1634_163402


namespace sum_A_k_div_k_l1634_163467

noncomputable def A (k : ℕ) : ℕ :=
  (Finset.filter (fun d => d % 2 = 1 ∧ d ≤ Nat.sqrt (2 * k - 1)) (Finset.range k)).card

noncomputable def sumExpression : ℝ :=
  ∑' k, (-1)^(k-1) * (A k / k : ℝ)

theorem sum_A_k_div_k : sumExpression = Real.pi^2 / 8 :=
  sorry

end sum_A_k_div_k_l1634_163467


namespace factorize_1_factorize_2_factorize_3_l1634_163496

-- Problem 1: Factorize 3a^3 - 6a^2 + 3a
theorem factorize_1 (a : ℝ) : 3 * a^3 - 6 * a^2 + 3 * a = 3 * a * (a - 1)^2 :=
sorry

-- Problem 2: Factorize a^2(x - y) + b^2(y - x)
theorem factorize_2 (a b x y : ℝ) : a^2 * (x - y) + b^2 * (y - x) = (x - y) * (a - b) * (a + b) :=
sorry

-- Problem 3: Factorize 16(a + b)^2 - 9(a - b)^2
theorem factorize_3 (a b : ℝ) : 16 * (a + b)^2 - 9 * (a - b)^2 = (a + 7 * b) * (7 * a + b) :=
sorry

end factorize_1_factorize_2_factorize_3_l1634_163496


namespace abs_fraction_eq_sqrt_seven_thirds_l1634_163469

open Real

theorem abs_fraction_eq_sqrt_seven_thirds {a b : ℝ} 
  (a_nonzero : a ≠ 0) (b_nonzero : b ≠ 0) (h : a^2 + b^2 = 5 * a * b) : 
  abs ((a + b) / (a - b)) = sqrt (7 / 3) :=
by
  sorry

end abs_fraction_eq_sqrt_seven_thirds_l1634_163469


namespace Diana_total_earnings_l1634_163475

def July : ℝ := 150
def August : ℝ := 3 * July
def September : ℝ := 2 * August
def October : ℝ := September + 0.1 * September
def November : ℝ := 0.95 * October
def Total_earnings : ℝ := July + August + September + October + November

theorem Diana_total_earnings : Total_earnings = 3430.50 := by
  sorry

end Diana_total_earnings_l1634_163475


namespace lara_cookies_l1634_163480

theorem lara_cookies (total_cookies trays rows_per_row : ℕ)
  (h_total : total_cookies = 120)
  (h_trays : trays = 4)
  (h_rows_per_row : rows_per_row = 6) :
  total_cookies / rows_per_row / trays = 5 :=
by
  sorry

end lara_cookies_l1634_163480


namespace sum_of_digits_Joey_age_twice_Max_next_l1634_163484

noncomputable def Joey_is_two_years_older (C : ℕ) : ℕ := C + 2

noncomputable def Max_age_today := 2

noncomputable def Eight_multiples_of_Max (C : ℕ) := 
  ∃ n : ℕ, C = 24 + n

noncomputable def Next_Joey_age_twice_Max (C J M n : ℕ): Prop := J + n = 2 * (M + n)

theorem sum_of_digits_Joey_age_twice_Max_next (C J M n : ℕ) 
  (h1: J = Joey_is_two_years_older C)
  (h2: M = Max_age_today)
  (h3: Eight_multiples_of_Max C)
  (h4: Next_Joey_age_twice_Max C J M n) 
  : ∃ s, s = 7 :=
sorry

end sum_of_digits_Joey_age_twice_Max_next_l1634_163484


namespace smallest_rel_prime_to_180_l1634_163421

theorem smallest_rel_prime_to_180 : ∃ x : ℕ, x > 1 ∧ Int.gcd x 180 = 1 ∧ ∀ y : ℕ, y > 1 ∧ Int.gcd y 180 = 1 → x ≤ y := 
sorry

end smallest_rel_prime_to_180_l1634_163421


namespace alpha_numeric_puzzle_l1634_163468

theorem alpha_numeric_puzzle : 
  ∀ (a b c d e f g h i : ℕ),
  (∀ x y : ℕ, x ≠ 0 → y ≠ 0 → x ≠ y) →
  100 * a + 10 * b + c + 100 * d + 10 * e + f + 100 * g + 10 * h + i = 1665 → 
  c + f + i = 15 →
  b + e + h = 15 :=
by
  intros a b c d e f g h i distinct nonzero_sum unit_digits_sum
  sorry

end alpha_numeric_puzzle_l1634_163468


namespace find_number_l1634_163431

theorem find_number (m : ℤ) (h1 : ∃ k1 : ℤ, k1 * k1 = m + 100) (h2 : ∃ k2 : ℤ, k2 * k2 = m + 168) : m = 156 :=
sorry

end find_number_l1634_163431


namespace mumu_identity_l1634_163460

def f (m u : ℕ) : ℕ := 
  -- Assume f is correctly defined to match the number of valid Mumu words 
  -- involving m M's and u U's according to the problem's definition.
  sorry 

theorem mumu_identity (u m : ℕ) (h₁ : u ≥ 2) (h₂ : 3 ≤ m) (h₃ : m ≤ 2 * u) :
  f m u = f (2 * u - m + 1) u ↔ f m (u - 1) = f (2 * u - m + 1) (u - 1) :=
by
  sorry

end mumu_identity_l1634_163460


namespace regression_equation_represents_real_relationship_maximized_l1634_163416

-- Definitions from the conditions
def regression_equation (y x : ℝ) := ∃ (a b : ℝ), y = a * x + b

def represents_real_relationship_maximized (y x : ℝ) := regression_equation y x

-- The proof problem statement
theorem regression_equation_represents_real_relationship_maximized 
: ∀ (y x : ℝ), regression_equation y x → represents_real_relationship_maximized y x :=
by
  sorry

end regression_equation_represents_real_relationship_maximized_l1634_163416


namespace circle_symmetric_line_l1634_163489
-- Importing the entire Math library

-- Define the statement
theorem circle_symmetric_line (a : ℝ) :
  (∀ (A B : ℝ × ℝ), 
    (A.1)^2 + (A.2)^2 = 2 * a * (A.1) 
    ∧ (B.1)^2 + (B.2)^2 = 2 * a * (B.1) 
    ∧ A.2 = 2 * A.1 + 1 
    ∧ B.2 = 2 * B.1 + 1 
    ∧ A.2 = B.2) 
  → a = -1/2 :=
by
  sorry

end circle_symmetric_line_l1634_163489


namespace general_term_sequence_l1634_163452

theorem general_term_sequence (a : ℕ → ℕ) (h₁ : a 1 = 1) (h₂ : ∀ n, a (n + 1) = a n + 3^n) :
  ∀ n, a n = (3^n - 1) / 2 := 
by
  sorry

end general_term_sequence_l1634_163452


namespace coordinates_of_C_l1634_163473

theorem coordinates_of_C (A B : ℝ × ℝ) (C : ℝ × ℝ) 
    (hA : A = (1, 3)) (hB : B = (9, -3)) (hBC_AB : dist B C = 1/2 * dist A B) : 
    C = (13, -6) :=
sorry

end coordinates_of_C_l1634_163473


namespace vasya_numbers_l1634_163488

theorem vasya_numbers : ∀ (x y : ℝ), 
  (x + y = x * y) ∧ (x * y = x / y) → (x = 1/2 ∧ y = -1) :=
by
  intros x y h
  sorry

end vasya_numbers_l1634_163488


namespace speed_of_stream_l1634_163485

theorem speed_of_stream (b s : ℝ) 
  (H1 : b + s = 10)
  (H2 : b - s = 4) : 
  s = 3 :=
sorry

end speed_of_stream_l1634_163485


namespace total_number_of_workers_l1634_163476

theorem total_number_of_workers 
  (W : ℕ) 
  (h_all_avg : W * 8000 = 10 * 12000 + (W - 10) * 6000) : 
  W = 30 := 
by
  sorry

end total_number_of_workers_l1634_163476


namespace sum_of_powers_of_i_l1634_163482

theorem sum_of_powers_of_i : 
  ∀ (i : ℂ), i^2 = -1 → 1 + i + i^2 + i^3 + i^4 + i^5 + i^6 = i :=
by
  intro i h
  sorry

end sum_of_powers_of_i_l1634_163482


namespace total_frames_l1634_163472

def frames_per_page : ℝ := 143.0

def pages : ℝ := 11.0

theorem total_frames : frames_per_page * pages = 1573.0 :=
by
  sorry

end total_frames_l1634_163472


namespace avg_weight_B_correct_l1634_163425

-- Definitions of the conditions
def students_A : ℕ := 24
def students_B : ℕ := 16
def avg_weight_A : ℝ := 40
def avg_weight_class : ℝ := 38

-- Definition of the total weight calculation for sections A and B
def total_weight_A : ℝ := students_A * avg_weight_A
def total_weight_class : ℝ := (students_A + students_B) * avg_weight_class

-- Defining the average weight of section B as the unknown to be proven
noncomputable def avg_weight_B : ℝ := 35

-- The theorem to prove that the average weight of section B is 35 kg
theorem avg_weight_B_correct : 
  total_weight_A + students_B * avg_weight_B = total_weight_class :=
by
  sorry

end avg_weight_B_correct_l1634_163425


namespace prevent_white_cube_n2_prevent_white_cube_n3_l1634_163463

def min_faces_to_paint (n : ℕ) : ℕ :=
  if n = 2 then 2 else if n = 3 then 12 else sorry

theorem prevent_white_cube_n2 : min_faces_to_paint 2 = 2 := by
  sorry

theorem prevent_white_cube_n3 : min_faces_to_paint 3 = 12 := by
  sorry

end prevent_white_cube_n2_prevent_white_cube_n3_l1634_163463


namespace probability_two_red_balls_randomly_picked_l1634_163415

theorem probability_two_red_balls_randomly_picked :
  (3/9) * (2/8) = 1/12 :=
by sorry

end probability_two_red_balls_randomly_picked_l1634_163415


namespace Nord_Stream_pipeline_payment_l1634_163497

/-- Suppose Russia, Germany, and France decided to build the "Nord Stream 2" pipeline,
     which is 1200 km long, agreeing to finance this project equally.
     Russia built 650 kilometers of the pipeline.
     Germany built 550 kilometers of the pipeline.
     France contributed its share in money and did not build any kilometers.
     Germany received 1.2 billion euros from France.
     Prove that Russia should receive 2 billion euros from France.
--/
theorem Nord_Stream_pipeline_payment
  (total_km : ℝ)
  (russia_km : ℝ)
  (germany_km : ℝ)
  (total_countries : ℝ)
  (payment_to_germany : ℝ)
  (germany_additional_payment : ℝ)
  (france_km : ℝ)
  (france_payment_ratio : ℝ)
  (russia_payment : ℝ) :
  total_km = 1200 ∧
  russia_km = 650 ∧
  germany_km = 550 ∧
  total_countries = 3 ∧
  payment_to_germany = 1.2 ∧
  france_km = 0 ∧
  germany_additional_payment = germany_km - (total_km / total_countries) ∧
  france_payment_ratio = 5 / 3 ∧
  russia_payment = payment_to_germany * (5 / 3) →
  russia_payment = 2 := by sorry

end Nord_Stream_pipeline_payment_l1634_163497


namespace problem_l1634_163450

   def f (n : ℕ) : ℕ := sorry

   theorem problem (f : ℕ → ℕ) (h1 : ∀ n, f (f n) + f n = 2 * n + 3) (h2 : f 0 = 1) :
     f 2013 = 2014 :=
   sorry
   
end problem_l1634_163450


namespace probability_interval_l1634_163432

theorem probability_interval (P_A P_B P_A_inter_P_B : ℝ) (h1 : P_A = 3 / 4) (h2 : P_B = 2 / 3) : 
  5/12 ≤ P_A_inter_P_B ∧ P_A_inter_P_B ≤ 2/3 :=
sorry

end probability_interval_l1634_163432


namespace center_of_tangent_circle_l1634_163405

theorem center_of_tangent_circle (x y : ℝ) 
  (h1 : 3*x - 4*y = 12) 
  (h2 : 3*x - 4*y = -24)
  (h3 : x - 2*y = 0) : 
  (x, y) = (-6, -3) :=
by
  sorry

end center_of_tangent_circle_l1634_163405


namespace find_coef_of_quadratic_l1634_163478

-- Define the problem conditions
def solutions_of_abs_eq : Set ℤ := {x | abs (x - 3) = 4}

-- Given that the solutions are 7 and -1
def paul_solutions : Set ℤ := {7, -1}

-- The problem translates to proving the equivalence of two sets
def equivalent_equation_solutions (d e : ℤ) : Prop :=
  ∀ x, x ∈ solutions_of_abs_eq ↔ x^2 + d * x + e = 0

theorem find_coef_of_quadratic :
  equivalent_equation_solutions (-6) (-7) :=
by
  sorry

end find_coef_of_quadratic_l1634_163478


namespace necessary_but_not_sufficient_condition_l1634_163477

theorem necessary_but_not_sufficient_condition (a c : ℝ) (h : c ≠ 0) : ¬ ((∀ (a : ℝ) (h : c ≠ 0), (ax^2 + y^2 = c) → ((ax^2 + y^2 = c) → ( (c ≠ 0) ))) ∧ ¬ ((∀ (a : ℝ), ¬ (ax^2 + y^2 ≠ c) → ( (ax^2 + y^2 = c) → ((c = 0) ))) )) :=
sorry

end necessary_but_not_sufficient_condition_l1634_163477


namespace roger_allowance_fraction_l1634_163407

noncomputable def allowance_fraction (A m s p : ℝ) : ℝ :=
  m + s + p

theorem roger_allowance_fraction (A : ℝ) (m s p : ℝ) 
  (h_movie : m = 0.25 * (A - s - p))
  (h_soda : s = 0.10 * (A - m - p))
  (h_popcorn : p = 0.05 * (A - m - s)) :
  allowance_fraction A m s p = 0.32 * A :=
by
  sorry

end roger_allowance_fraction_l1634_163407


namespace diameter_of_circular_field_l1634_163418

theorem diameter_of_circular_field :
  ∀ (π : ℝ) (cost_per_meter total_cost circumference diameter : ℝ),
    π = Real.pi → 
    cost_per_meter = 1.50 → 
    total_cost = 94.24777960769379 → 
    circumference = total_cost / cost_per_meter →
    circumference = π * diameter →
    diameter = 20 := 
by
  intros π cost_per_meter total_cost circumference diameter hπ hcp ht cutoff_circ hcirc
  sorry

end diameter_of_circular_field_l1634_163418


namespace airplane_total_luggage_weight_l1634_163412

def num_people := 6
def bags_per_person := 5
def weight_per_bag := 50
def additional_bags := 90

def total_weight_people := num_people * bags_per_person * weight_per_bag
def total_weight_additional_bags := additional_bags * weight_per_bag

def total_luggage_weight := total_weight_people + total_weight_additional_bags

theorem airplane_total_luggage_weight : total_luggage_weight = 6000 :=
by
  sorry

end airplane_total_luggage_weight_l1634_163412


namespace paint_rate_l1634_163441

theorem paint_rate (l b : ℝ) (cost : ℕ) (rate_per_sq_m : ℝ) 
  (h1 : l = 3 * b) 
  (h2 : cost = 300) 
  (h3 : l = 13.416407864998739) 
  (area : ℝ := l * b) : 
  rate_per_sq_m = 5 :=
by
  sorry

end paint_rate_l1634_163441


namespace mary_change_l1634_163420

/-- 
Calculate the change Mary will receive after buying tickets for herself and her 3 children 
at the circus, given the ticket prices and special group rate discount.
-/
theorem mary_change :
  let adult_ticket := 2
  let child_ticket := 1
  let discounted_child_ticket := 0.5 * child_ticket
  let total_cost_with_discount := adult_ticket + 2 * child_ticket + discounted_child_ticket
  let payment := 20
  payment - total_cost_with_discount = 15.50 :=
by
  sorry

end mary_change_l1634_163420


namespace smallest_n_power_mod_5_l1634_163479

theorem smallest_n_power_mod_5 :
  ∃ N : ℕ, 100 ≤ N ∧ N ≤ 999 ∧ (2^N + 1) % 5 = 0 ∧ ∀ M : ℕ, 100 ≤ M ∧ M ≤ 999 ∧ (2^M + 1) % 5 = 0 → N ≤ M := 
sorry

end smallest_n_power_mod_5_l1634_163479


namespace iced_coffee_days_per_week_l1634_163411

theorem iced_coffee_days_per_week (x : ℕ) (h1 : 5 * 4 = 20)
  (h2 : 20 * 52 = 1040)
  (h3 : 2 * x = 2 * x)
  (h4 : 52 * (2 * x) = 104 * x)
  (h5 : 1040 + 104 * x = 1040 + 104 * x)
  (h6 : 1040 + 104 * x - 338 = 1040 + 104 * x - 338)
  (h7 : (0.75 : ℝ) * (1040 + 104 * x) = 780 + 78 * x) :
  x = 3 :=
by
  sorry

end iced_coffee_days_per_week_l1634_163411


namespace cats_on_ship_l1634_163428

theorem cats_on_ship :
  ∃ (C S : ℕ), 
  (C + S + 1 + 1 = 16) ∧
  (4 * C + 2 * S + 2 * 1 + 1 * 1 = 41) ∧ 
  C = 5 :=
by
  sorry

end cats_on_ship_l1634_163428


namespace integers_with_abs_less_than_four_l1634_163455

theorem integers_with_abs_less_than_four :
  {x : ℤ | |x| < 4} = {-3, -2, -1, 0, 1, 2, 3} :=
sorry

end integers_with_abs_less_than_four_l1634_163455


namespace find_certain_number_l1634_163439

-- Define the conditions as constants
def n1 : ℕ := 9
def n2 : ℕ := 70
def n3 : ℕ := 25
def n4 : ℕ := 21
def smallest_given_number : ℕ := 3153
def certain_number : ℕ := 3147

-- Lean theorem statement
theorem find_certain_number (n1 n2 n3 n4 smallest_given_number certain_number: ℕ) :
  (∀ x, (∀ y ∈ [n1, n2, n3, n4], y ∣ x) → x ≥ smallest_given_number → x = smallest_given_number + certain_number) :=
sorry -- Skips the proof

end find_certain_number_l1634_163439


namespace four_units_away_l1634_163492

theorem four_units_away (x : ℤ) (h : abs (x + 2) = 4) : x = 2 ∨ x = -6 :=
by
  sorry

end four_units_away_l1634_163492


namespace expected_yield_correct_l1634_163404

/-- Define the problem variables and conditions -/
def steps_x : ℕ := 25
def steps_y : ℕ := 20
def step_length : ℝ := 2.5
def yield_per_sqft : ℝ := 0.75

/-- Calculate the dimensions in feet -/
def length_x := steps_x * step_length
def length_y := steps_y * step_length

/-- Calculate the area of the orchard -/
def area := length_x * length_y

/-- Calculate the expected yield of apples -/
def expected_yield := area * yield_per_sqft

/-- Prove the expected yield of apples is 2343.75 pounds -/
theorem expected_yield_correct : expected_yield = 2343.75 := sorry

end expected_yield_correct_l1634_163404


namespace inequality_proof_l1634_163495

noncomputable def a : ℝ := Real.log 0.3 / Real.log 0.2
noncomputable def b : ℝ := Real.log 0.3 / Real.log 2

theorem inequality_proof : (a * b < a + b ∧ a + b < 0) :=
by
  sorry

end inequality_proof_l1634_163495


namespace domain_of_log_l1634_163486

def log_domain := {x : ℝ | x > 1}

theorem domain_of_log : {x : ℝ | ∃ y, y = log_domain} = {x : ℝ | x > 1} :=
by
  sorry

end domain_of_log_l1634_163486


namespace amber_josh_departure_time_l1634_163414

def latest_departure_time (flight_time : ℕ) (check_in_time : ℕ) (drive_time : ℕ) (parking_time : ℕ) :=
  flight_time - check_in_time - drive_time - parking_time

theorem amber_josh_departure_time :
  latest_departure_time 20 2 (45 / 60) (15 / 60) = 17 :=
by
  -- Placeholder for actual proof
  sorry

end amber_josh_departure_time_l1634_163414


namespace discount_percentage_l1634_163400

theorem discount_percentage
  (MP CP SP : ℝ)
  (h1 : CP = 0.64 * MP)
  (h2 : 37.5 = (SP - CP) / CP * 100) :
  (MP - SP) / MP * 100 = 12 := by
  sorry

end discount_percentage_l1634_163400


namespace percentage_profits_revenues_previous_year_l1634_163446

noncomputable def companyProfits (R P R2009 P2009 : ℝ) : Prop :=
  (R2009 = 0.8 * R) ∧ (P2009 = 0.15 * R2009) ∧ (P2009 = 1.5 * P)

theorem percentage_profits_revenues_previous_year (R P : ℝ) (h : companyProfits R P (0.8 * R) (0.12 * R)) : 
  (P / R * 100) = 8 :=
by 
  sorry

end percentage_profits_revenues_previous_year_l1634_163446


namespace polynomial_not_divisible_by_x_minus_5_l1634_163490

theorem polynomial_not_divisible_by_x_minus_5 (m : ℝ) :
  (∀ x, x = 4 → (4 * x^3 - 16 * x^2 + m * x - 20) = 0) →
  ¬(∀ x, x = 5 → (4 * x^3 - 16 * x^2 + m * x - 20) = 0) :=
by
  sorry

end polynomial_not_divisible_by_x_minus_5_l1634_163490


namespace solve_z_squared_eq_l1634_163453

open Complex

theorem solve_z_squared_eq : 
  ∀ z : ℂ, z^2 = -100 - 64 * I → (z = 4 - 8 * I ∨ z = -4 + 8 * I) :=
by
  sorry

end solve_z_squared_eq_l1634_163453


namespace brownies_shared_l1634_163498

theorem brownies_shared
  (total_brownies : ℕ)
  (tina_brownies : ℕ)
  (husband_brownies : ℕ)
  (remaining_brownies : ℕ)
  (shared_brownies : ℕ)
  (h1 : total_brownies = 24)
  (h2 : tina_brownies = 10)
  (h3 : husband_brownies = 5)
  (h4 : remaining_brownies = 5) :
  shared_brownies = total_brownies - (tina_brownies + husband_brownies + remaining_brownies) → shared_brownies = 4 :=
by
  sorry

end brownies_shared_l1634_163498


namespace range_of_m_l1634_163429

theorem range_of_m (m : ℝ) (h : 1 < (8 - m) / (m - 5)) : 5 < m ∧ m < 13 / 2 :=
sorry

end range_of_m_l1634_163429


namespace min_value_of_T_l1634_163413

noncomputable def T_min_value (a b c : ℝ) : ℝ :=
  (5 + 2*a*b + 4*a*c) / (a*b + 1)

theorem min_value_of_T :
  ∀ (a b c : ℝ),
  a < 0 →
  b > 0 →
  b^2 ≤ (4 * c) / a →
  c ≤ (1/4) * a * b^2 →
  T_min_value a b c ≥ 4 ∧ (T_min_value a b c = 4 ↔ a * b = -3) :=
by
  intros
  sorry

end min_value_of_T_l1634_163413


namespace total_flour_needed_l1634_163451

-- Definitions of flour needed by Katie and Sheila
def katie_flour : ℕ := 3
def sheila_flour : ℕ := katie_flour + 2

-- Statement of the theorem
theorem total_flour_needed : katie_flour + sheila_flour = 8 := by
  -- The proof would go here
  sorry

end total_flour_needed_l1634_163451


namespace original_fraction_eq_two_thirds_l1634_163466

theorem original_fraction_eq_two_thirds (a b : ℕ) (h : (a^3 : ℚ) / (b + 3) = 2 * (a / b)) : a = 2 ∧ b = 3 :=
by {
  sorry
}

end original_fraction_eq_two_thirds_l1634_163466


namespace necessary_and_sufficient_condition_l1634_163457

-- Variables and conditions
variables (a : ℕ) (A B : ℝ)
variable (positive_a : 0 < a)

-- System of equations
def system_has_positive_integer_solutions (x y z : ℕ) : Prop :=
  (x^2 + y^2 + z^2 = (13 * a)^2) ∧ 
  (x^2 * (A * x^2 + B * y^2) + y^2 * (A * y^2 + B * z^2) + z^2 * (A * z^2 + B * x^2) = 
    (1 / 4) * (2 * A + B) * (13 * a)^4)

-- Statement of the theorem
theorem necessary_and_sufficient_condition:
  (∃ (x y z : ℕ), system_has_positive_integer_solutions a A B x y z) ↔ B = 2 * A :=
sorry

end necessary_and_sufficient_condition_l1634_163457


namespace choose_student_B_l1634_163461

-- Define the scores for students A and B
def scores_A : List ℕ := [72, 85, 86, 90, 92]
def scores_B : List ℕ := [76, 83, 85, 87, 94]

-- Function to calculate the average of scores
def average (scores : List ℕ) : ℚ :=
  scores.sum / scores.length

-- Function to calculate the variance of scores
def variance (scores : List ℕ) : ℚ :=
  let mean := average scores
  (scores.map (λ x => (x - mean) * (x - mean))).sum / scores.length

-- Calculate the average scores for A and B
def avg_A : ℚ := average scores_A
def avg_B : ℚ := average scores_B

-- Calculate the variances for A and B
def var_A : ℚ := variance scores_A
def var_B : ℚ := variance scores_B

-- The theorem to be proved
theorem choose_student_B : var_B < var_A :=
  by sorry

end choose_student_B_l1634_163461


namespace small_pump_fill_time_l1634_163459

noncomputable def small_pump_time (large_pump_time combined_time : ℝ) : ℝ :=
  let large_pump_rate := 1 / large_pump_time
  let combined_rate := 1 / combined_time
  let small_pump_rate := combined_rate - large_pump_rate
  1 / small_pump_rate

theorem small_pump_fill_time :
  small_pump_time (1 / 3) 0.2857142857142857 = 2 :=
by
  sorry

end small_pump_fill_time_l1634_163459


namespace people_and_carriages_condition_l1634_163470

-- Definitions corresponding to the conditions
def num_people_using_carriages (x : ℕ) : ℕ := 3 * (x - 2)
def num_people_sharing_carriages (x : ℕ) : ℕ := 2 * x + 9

-- The theorem statement we need to prove
theorem people_and_carriages_condition (x : ℕ) : 
  num_people_using_carriages x = num_people_sharing_carriages x ↔ 3 * (x - 2) = 2 * x + 9 :=
by sorry

end people_and_carriages_condition_l1634_163470


namespace max_quotient_l1634_163426

theorem max_quotient (a b : ℝ) (ha : 300 ≤ a ∧ a ≤ 500) (hb : 800 ≤ b ∧ b ≤ 1600) : ∃ q, q = b / a ∧ q ≤ 16 / 3 :=
by 
  sorry

end max_quotient_l1634_163426


namespace identity_proof_l1634_163435

theorem identity_proof : 
  ∀ x : ℝ, 
    (x^2 + 3*x + 2) * (x + 3) = (x + 1) * (x^2 + 5*x + 6) := 
by 
  sorry

end identity_proof_l1634_163435


namespace words_with_at_least_one_consonant_l1634_163422

-- Define the letters available and classify them as vowels and consonants
def letters : List Char := ['A', 'B', 'C', 'D', 'E', 'F']
def vowels : List Char := ['A', 'E']
def consonants : List Char := ['B', 'C', 'D', 'F']

-- Define the total number of 5-letter words using the given letters
def total_words : ℕ := 6^5

-- Define the total number of 5-letter words composed exclusively of vowels
def vowel_words : ℕ := 2^5

-- Define the number of 5-letter words that contain at least one consonant
noncomputable def words_with_consonant : ℕ := total_words - vowel_words

-- The theorem to prove
theorem words_with_at_least_one_consonant : words_with_consonant = 7744 := by
  sorry

end words_with_at_least_one_consonant_l1634_163422


namespace factorization_4x2_minus_144_l1634_163462

theorem factorization_4x2_minus_144 (x : ℝ) : 4 * x^2 - 144 = 4 * (x - 6) * (x + 6) := 
  sorry

end factorization_4x2_minus_144_l1634_163462


namespace trapezoid_area_l1634_163419

-- Geometry setup
variable (outer_area : ℝ) (inner_height_ratio : ℝ)

-- Conditions
def outer_triangle_area := outer_area = 36
def inner_height_to_outer_height := inner_height_ratio = 2 / 3

-- Conclusion: Area of one trapezoid
theorem trapezoid_area (outer_area inner_height_ratio : ℝ) 
  (h_outer : outer_triangle_area outer_area) 
  (h_inner : inner_height_to_outer_height inner_height_ratio) : 
  (outer_area - 16 * Real.sqrt 3) / 3 = (36 - 16 * Real.sqrt 3) / 3 := 
sorry

end trapezoid_area_l1634_163419


namespace min_value_x_plus_2y_l1634_163456

theorem min_value_x_plus_2y (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + y = x * y) : x + 2 * y ≥ 3 + 2 * Real.sqrt 2 :=
sorry

end min_value_x_plus_2y_l1634_163456


namespace train_length_calculation_l1634_163471

noncomputable def length_of_train (speed : ℝ) (time_in_sec : ℝ) : ℝ :=
  let time_in_hr := time_in_sec / 3600
  let distance_in_km := speed * time_in_hr
  distance_in_km * 1000

theorem train_length_calculation : 
  length_of_train 60 30 = 500 :=
by
  -- The proof would go here, but we provide a stub with sorry.
  sorry

end train_length_calculation_l1634_163471


namespace sin_angle_identity_l1634_163474

theorem sin_angle_identity : 
  (Real.sin (Real.pi / 4) * Real.sin (7 * Real.pi / 12) + Real.sin (Real.pi / 4) * Real.sin (Real.pi / 12)) = Real.sqrt 3 / 2 := 
by 
  sorry

end sin_angle_identity_l1634_163474


namespace sum_of_15th_set_l1634_163447

def first_element_of_set (n : ℕ) : ℕ :=
  3 + (n * (n - 1)) / 2

def sum_of_elements_in_set (n : ℕ) : ℕ :=
  let a_n := first_element_of_set n
  let l_n := a_n + n - 1
  n * (a_n + l_n) / 2

theorem sum_of_15th_set :
  sum_of_elements_in_set 15 = 1725 :=
by
  sorry

end sum_of_15th_set_l1634_163447


namespace speed_of_B_l1634_163408

theorem speed_of_B 
  (A_speed : ℝ)
  (t1 : ℝ)
  (t2 : ℝ)
  (d1 := A_speed * t1)
  (d2 := A_speed * t2)
  (total_distance := d1 + d2)
  (B_speed := total_distance / t2) :
  A_speed = 7 → 
  t1 = 0.5 → 
  t2 = 1.8 →
  B_speed = 8.944 :=
by
  intros h1 h2 h3
  simp [h1, h2, h3]
  exact sorry

end speed_of_B_l1634_163408


namespace distance_between_points_l1634_163487

theorem distance_between_points {A B : ℝ}
  (hA : abs A = 3)
  (hB : abs B = 9) :
  abs (A - B) = 6 ∨ abs (A - B) = 12 :=
sorry

end distance_between_points_l1634_163487


namespace intersection_of_A_and_B_l1634_163423

-- Define the set A
def A : Set ℝ := {-1, 0, 1}

-- Define the set B based on the given conditions
def B : Set ℝ := {y | ∃ x ∈ A, y = Real.cos (Real.pi * x)}

-- The main theorem to prove that A ∩ B is {-1, 1}
theorem intersection_of_A_and_B : A ∩ B = {-1, 1} := by
  sorry

end intersection_of_A_and_B_l1634_163423


namespace solve_inequality_system_l1634_163406

theorem solve_inequality_system (x : ℝ) (h1 : 3 * x - 2 < x) (h2 : (1 / 3) * x < -2) : x < -6 :=
sorry

end solve_inequality_system_l1634_163406


namespace initial_games_l1634_163401

-- Conditions
def games_given_away : ℕ := 7
def games_left : ℕ := 91

-- Theorem Statement
theorem initial_games (initial_games : ℕ) : 
  initial_games = games_left + games_given_away :=
by
  sorry

end initial_games_l1634_163401


namespace volume_of_regular_triangular_pyramid_l1634_163440

noncomputable def pyramid_volume (a b γ : ℝ) : ℝ :=
  (1 / 3) * (a^2 * Real.sqrt 3 / 4) * Real.sqrt (b^2 - (a * Real.sqrt 3 / (2 * Real.cos (γ / 2)))^2)

theorem volume_of_regular_triangular_pyramid (a b γ : ℝ) :
  pyramid_volume a b γ = (1 / 3) * (a^2 * Real.sqrt 3 / 4) * Real.sqrt (b^2 - (a * Real.sqrt 3 / (2 * Real.cos (γ / 2)))^2) :=
by
  sorry

end volume_of_regular_triangular_pyramid_l1634_163440
