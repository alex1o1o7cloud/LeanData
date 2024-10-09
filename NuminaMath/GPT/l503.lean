import Mathlib

namespace units_digit_17_mul_27_l503_50366

theorem units_digit_17_mul_27 : 
  let u17 := (17 % 10)
  let u27 := (27 % 10)
  let units_product := (u17 * u27) % 10
  units_product = 9 := by
  let u17 := (17 % 10)
  let u27 := (27 % 10)
  let units_product := (u17 * u27) % 10
  sorry

end units_digit_17_mul_27_l503_50366


namespace volume_of_63_ounces_l503_50386

variable {V W : ℝ}
variable (k : ℝ)

def directly_proportional (V W : ℝ) (k : ℝ) : Prop :=
  V = k * W

theorem volume_of_63_ounces (h1 : directly_proportional 48 112 k)
                            (h2 : directly_proportional V 63 k) :
  V = 27 := by
  sorry

end volume_of_63_ounces_l503_50386


namespace sufficient_necessary_condition_l503_50391

theorem sufficient_necessary_condition (a : ℝ) :
  (∃ x : ℝ, 2 * x + 1 = a ∧ x > 2) ↔ a > 5 :=
by
  sorry

end sufficient_necessary_condition_l503_50391


namespace inequality_solution_condition_necessary_but_not_sufficient_l503_50358

theorem inequality_solution_condition (a : ℝ) :
  (∃ x : ℝ, x^2 + 2 * a * x - a ≤ 0) ↔ (a ≥ 0 ∨ a ≤ -1) := sorry

theorem necessary_but_not_sufficient (a : ℝ) :
  (a > 0 ∨ a < -1) → (∃ x : ℝ, x^2 + 2 * a * x - a ≤ 0) ∧ ¬(∃ x : ℝ, x^2 + 2 * a * x - a ≤ 0 → (a > 0 ∨ a < -1)) := sorry

end inequality_solution_condition_necessary_but_not_sufficient_l503_50358


namespace gcd_largest_of_forms_l503_50378

theorem gcd_largest_of_forms (a b : ℕ) (h1 : a ≠ b) (h2 : a < 10) (h3 : b < 10) :
  Nat.gcd (100 * a + 11 * b) (101 * b + 10 * a) = 45 :=
by
  sorry

end gcd_largest_of_forms_l503_50378


namespace geometric_sequence_seventh_term_l503_50383

theorem geometric_sequence_seventh_term (a r : ℝ) 
  (h4 : a * r^3 = 16) 
  (h9 : a * r^8 = 2) : 
  a * r^6 = 8 := 
sorry

end geometric_sequence_seventh_term_l503_50383


namespace pq_eq_neg72_l503_50390

theorem pq_eq_neg72 {p q : ℝ} (h : ∀ x, (x - 7) * (3 * x + 11) = x ^ 2 - 20 * x + 63 →
(p = x ∨ q = x) ∧ p ≠ q) : 
(p + 2) * (q + 2) = -72 :=
sorry

end pq_eq_neg72_l503_50390


namespace A_or_B_not_A_or_C_A_and_C_A_and_B_or_C_l503_50302

-- Definitions of events
def A : Prop := sorry -- event that the part is of the first grade
def B : Prop := sorry -- event that the part is of the second grade
def C : Prop := sorry -- event that the part is of the third grade

-- Mathematically equivalent proof problems
theorem A_or_B : A ∨ B ↔ (A ∨ B) :=
by sorry

theorem not_A_or_C : ¬(A ∨ C) ↔ B :=
by sorry

theorem A_and_C : (A ∧ C) ↔ false :=
by sorry

theorem A_and_B_or_C : ((A ∧ B) ∨ C) ↔ C :=
by sorry

end A_or_B_not_A_or_C_A_and_C_A_and_B_or_C_l503_50302


namespace sum_common_elements_ap_gp_l503_50301

noncomputable def sum_of_first_10_common_elements : ℕ := 20 * (4^10 - 1) / (4 - 1)

theorem sum_common_elements_ap_gp :
  sum_of_first_10_common_elements = 6990500 :=
by
  unfold sum_of_first_10_common_elements
  sorry

end sum_common_elements_ap_gp_l503_50301


namespace sheila_weekly_earnings_l503_50319

-- Variables
variables {hours_mon_wed_fri hours_tue_thu rate_per_hour : ℕ}

-- Conditions
def sheila_works_mwf : hours_mon_wed_fri = 8 := by sorry
def sheila_works_tue_thu : hours_tue_thu = 6 := by sorry
def sheila_rate : rate_per_hour = 11 := by sorry

-- Main statement to prove
theorem sheila_weekly_earnings : 
  3 * hours_mon_wed_fri + 2 * hours_tue_thu = 36 →
  rate_per_hour = 11 →
  (3 * hours_mon_wed_fri + 2 * hours_tue_thu) * rate_per_hour = 396 :=
by
  intros h_hours h_rate
  sorry

end sheila_weekly_earnings_l503_50319


namespace problem_statement_l503_50399

variable (a b : ℝ)

theorem problem_statement (h1 : a > b) (h2 : b > 1/a) (h3 : 1/a > 0) :
  (a + b > 2) ∧ (a > 1) ∧ (a - 1/b > b - 1/a) :=
by 
  sorry

end problem_statement_l503_50399


namespace find_root_l503_50379

theorem find_root (y : ℝ) (h : y - 9 / (y - 4) = 2 - 9 / (y - 4)) : y = 2 :=
by
  sorry

end find_root_l503_50379


namespace cannot_be_sum_of_four_consecutive_even_integers_l503_50331

-- Define what it means to be the sum of four consecutive even integers
def sum_of_four_consecutive_even_integers (n : ℤ) : Prop :=
  ∃ m : ℤ, n = 4 * m + 12 ∧ m % 2 = 0

-- State the problem in Lean 4
theorem cannot_be_sum_of_four_consecutive_even_integers :
  ¬ sum_of_four_consecutive_even_integers 32 ∧
  ¬ sum_of_four_consecutive_even_integers 80 ∧
  ¬ sum_of_four_consecutive_even_integers 104 ∧
  ¬ sum_of_four_consecutive_even_integers 200 :=
by
  sorry

end cannot_be_sum_of_four_consecutive_even_integers_l503_50331


namespace card_game_impossible_l503_50394

theorem card_game_impossible : 
  ∀ (students : ℕ) (initial_cards : ℕ) (cards_distribution : ℕ → ℕ), 
  students = 2018 → 
  initial_cards = 2018 →
  (∀ n, n < students → (if n = 0 then cards_distribution n = initial_cards else cards_distribution n = 0)) →
  (¬ ∃ final_distribution : ℕ → ℕ, (∀ n, n < students → final_distribution n = 1)) :=
by
  intros students initial_cards cards_distribution stu_eq init_eq init_dist final_dist
  -- Sorry can be used here as the proof is not required
  sorry

end card_game_impossible_l503_50394


namespace min_value_f_l503_50350

noncomputable def f (x : ℝ) : ℝ := |x - 4| + |x - 6|

theorem min_value_f : ∃ x : ℝ, f x ≥ 2 :=
by
  sorry

end min_value_f_l503_50350


namespace average_first_two_numbers_l503_50335

theorem average_first_two_numbers (a1 a2 a3 a4 a5 a6 : ℝ)
  (h1 : (a1 + a2 + a3 + a4 + a5 + a6) / 6 = 3.95)
  (h2 : (a3 + a4) / 2 = 3.85)
  (h3 : (a5 + a6) / 2 = 4.200000000000001) :
  (a1 + a2) / 2 = 3.8 :=
by
  sorry

end average_first_two_numbers_l503_50335


namespace average_age_after_person_leaves_l503_50361

theorem average_age_after_person_leaves
  (average_age_seven : ℕ := 28)
  (num_people_initial : ℕ := 7)
  (person_leaves : ℕ := 20) :
  (average_age_seven * num_people_initial - person_leaves) / (num_people_initial - 1) = 29 := by
  sorry

end average_age_after_person_leaves_l503_50361


namespace SUVs_purchased_l503_50369

theorem SUVs_purchased (x : ℕ) (hToyota : ℕ) (hHonda : ℕ) (hNissan : ℕ) 
  (hRatioToyota : hToyota = 7 * x) 
  (hRatioHonda : hHonda = 5 * x) 
  (hRatioNissan : hNissan = 3 * x) 
  (hToyotaSUV : ℕ) (hHondaSUV : ℕ) (hNissanSUV : ℕ) 
  (hToyotaSUV_num : hToyotaSUV = (50 * hToyota) / 100) 
  (hHondaSUV_num : hHondaSUV = (40 * hHonda) / 100) 
  (hNissanSUV_num : hNissanSUV = (30 * hNissan) / 100) : 
  hToyotaSUV + hHondaSUV + hNissanSUV = 64 := 
by
  sorry

end SUVs_purchased_l503_50369


namespace geometric_series_common_ratio_l503_50363

theorem geometric_series_common_ratio
    (a : ℝ) (S : ℝ) (r : ℝ)
    (h_a : a = 512)
    (h_S : S = 3072)
    (h_sum : S = a / (1 - r)) : 
    r = 5 / 6 :=
by 
  rw [h_a] at h_sum
  rw [h_S] at h_sum
  sorry

end geometric_series_common_ratio_l503_50363


namespace Karlsson_eats_more_than_half_l503_50330

open Real

theorem Karlsson_eats_more_than_half
  (D : ℝ) (S : ℕ → ℝ)
  (a b : ℕ → ℝ)
  (cut_and_eat : ∀ n, S (n + 1) = S n - (S n * a n) / (a n + b n))
  (side_conditions : ∀ n, max (a n) (b n) ≤ D) :
  ∃ n, S n < (S 0) / 2 := sorry

end Karlsson_eats_more_than_half_l503_50330


namespace height_ratio_l503_50323

noncomputable def Anne_height := 80
noncomputable def Bella_height := 3 * Anne_height
noncomputable def Sister_height := Bella_height - 200

theorem height_ratio : Anne_height / Sister_height = 2 :=
by
  /-
  The proof here is omitted as requested.
  -/
  sorry

end height_ratio_l503_50323


namespace terminal_side_of_angle_y_eq_neg_one_l503_50374
/-
Given that the terminal side of angle θ lies on the line y = -x,
prove that y = -1 where y = sin θ / |sin θ| + |cos θ| / cos θ + tan θ / |tan θ|.
-/


noncomputable def y (θ : ℝ) : ℝ :=
  (Real.sin θ / |Real.sin θ|) + (|Real.cos θ| / Real.cos θ) + (Real.tan θ / |Real.tan θ|)

theorem terminal_side_of_angle_y_eq_neg_one (θ : ℝ) (k : ℤ) (h : θ = k * Real.pi - (Real.pi / 4)) :
  y θ = -1 :=
by
  sorry

end terminal_side_of_angle_y_eq_neg_one_l503_50374


namespace find_m_l503_50341

variables (a b m : ℝ)

def f (x : ℝ) : ℝ := x^3 + a*x^2 + b*x

def f' (x : ℝ) : ℝ := 3*x^2 + 2*a*x + b

theorem find_m (h1 : f m = 0) (h2 : f' m = 0) (h3 : m ≠ 0)
    (h4 : ∃ x, f' x = 0 ∧ ∀ y, x ≤ y → f x ≥ f y ∧ f x = 1/2) :
    m = 3/2 :=
sorry

end find_m_l503_50341


namespace stations_between_l503_50380

theorem stations_between (n : ℕ) (h : n * (n - 1) / 2 = 306) : n - 2 = 25 := 
by
  sorry

end stations_between_l503_50380


namespace poly_solution_l503_50313

-- Definitions for the conditions of the problem
def poly1 (d g : ℚ) := 5 * d ^ 2 - 4 * d + g
def poly2 (d h : ℚ) := 4 * d ^ 2 + h * d - 5
def product (d g h : ℚ) := 20 * d ^ 4 - 31 * d ^ 3 - 17 * d ^ 2 + 23 * d - 10

-- Statement of the problem: proving g + h = 7/2 given the conditions.
theorem poly_solution
  (g h : ℚ)
  (cond : ∀ d : ℚ, poly1 d g * poly2 d h = product d g h) :
  g + h = 7 / 2 :=
by
  sorry

end poly_solution_l503_50313


namespace rice_difference_l503_50381

theorem rice_difference:
  ∀ (total : ℕ) (kept_fraction : ℚ) (kept : ℕ) (given : ℕ),
  total = 50 →
  kept_fraction = (7 : ℚ) / 10 →
  kept = total * kept_fraction →
  given = total - kept →
  (kept - given = 20) :=
by
  intros total kept_fraction kept given
  intros h_total h_fraction h_kept h_given
  sorry

end rice_difference_l503_50381


namespace half_dollar_difference_l503_50310

theorem half_dollar_difference (n d h : ℕ) 
  (h1 : n + d + h = 150) 
  (h2 : 5 * n + 10 * d + 50 * h = 1500) : 
  ∃ h_max h_min, (h_max - h_min = 16) :=
by sorry

end half_dollar_difference_l503_50310


namespace min_value_parabola_l503_50332

theorem min_value_parabola : 
  ∃ (x : ℝ), 1 ≤ x ∧ x ≤ 4 ∧ (-x^2 + 4 * x - 2) = -2 :=
by
  sorry

end min_value_parabola_l503_50332


namespace determine_x_l503_50321

theorem determine_x (x y : ℝ) (h : x / (x - 2) = (y^2 + 3 * y - 2) / (y^2 + 3 * y + 1)) : 
  x = 2 * y^2 + 6 * y + 4 := 
by
  sorry

end determine_x_l503_50321


namespace num_factors_m_l503_50336

noncomputable def m : ℕ := 2^5 * 3^6 * 5^7 * 6^8

theorem num_factors_m : ∃ (k : ℕ), k = 1680 ∧ ∀ d : ℕ, d ∣ m ↔ ∃ (a b c : ℕ), 0 ≤ a ∧ a ≤ 13 ∧ 0 ≤ b ∧ b ≤ 14 ∧ 0 ≤ c ∧ c ≤ 7 ∧ d = 2^a * 3^b * 5^c :=
by 
sorry

end num_factors_m_l503_50336


namespace sqrt_addition_l503_50311

theorem sqrt_addition :
  (Real.sqrt (49 + 81) + Real.sqrt (36 - 9) = Real.sqrt 130 + 3 * Real.sqrt 3) := 
by sorry

end sqrt_addition_l503_50311


namespace find_missing_score_l503_50356

noncomputable def total_points (mean : ℝ) (games : ℕ) : ℝ :=
  mean * games

noncomputable def sum_of_scores (scores : List ℝ) : ℝ :=
  scores.sum

theorem find_missing_score
  (scores : List ℝ)
  (mean : ℝ)
  (games : ℕ)
  (total_points_value : ℝ)
  (sum_of_recorded_scores : ℝ)
  (missing_score : ℝ) :
  scores = [81, 73, 86, 73] →
  mean = 79.2 →
  games = 5 →
  total_points_value = total_points mean games →
  sum_of_recorded_scores = sum_of_scores scores →
  missing_score = total_points_value - sum_of_recorded_scores →
  missing_score = 83 :=
by
  intros
  exact sorry

end find_missing_score_l503_50356


namespace find_a_l503_50315

open Nat

-- Define the conditions and the proof goal
theorem find_a (a b : ℕ) (h1 : 2019 = a^2 - b^2) (h2 : a < 1000) : a = 338 :=
sorry

end find_a_l503_50315


namespace inequality_problem_l503_50349

variable (a b c d : ℝ)

theorem inequality_problem (h1 : b < 0) (h2 : 0 < a) (h3 : d < c) (h4 : c < 0) : 
  a + c > b + d := sorry

end inequality_problem_l503_50349


namespace quadrilateral_not_parallelogram_l503_50308

-- Definitions based on the given conditions
structure Quadrilateral :=
  (a b c d : ℝ) -- sides of the quadrilateral
  (parallel : Prop) -- one pair of parallel sides
  (equal_sides : Prop) -- another pair of equal sides

-- Problem statement
theorem quadrilateral_not_parallelogram (q : Quadrilateral) 
  (h1 : q.parallel) 
  (h2 : q.equal_sides) : 
  ¬ (∃ p : Quadrilateral, p = q) :=
sorry

end quadrilateral_not_parallelogram_l503_50308


namespace set_union_example_l503_50306

open Set

theorem set_union_example :
  let A := ({1, 3, 5, 6} : Set ℤ)
  let B := ({-1, 5, 7} : Set ℤ)
  A ∪ B = ({-1, 1, 3, 5, 6, 7} : Set ℤ) :=
by
  intros
  sorry

end set_union_example_l503_50306


namespace seminar_attendees_l503_50320

theorem seminar_attendees (a b c d attendees_not_from_companies : ℕ)
  (h1 : a = 30)
  (h2 : b = 2 * a)
  (h3 : c = a + 10)
  (h4 : d = c - 5)
  (h5 : attendees_not_from_companies = 20) :
  a + b + c + d + attendees_not_from_companies = 185 := by
  sorry

end seminar_attendees_l503_50320


namespace roots_greater_than_half_iff_l503_50322

noncomputable def quadratic_roots (a : ℝ) (x1 x2 : ℝ) : Prop :=
  (2 - a) * x1^2 - 3 * a * x1 + 2 * a = 0 ∧ 
  (2 - a) * x2^2 - 3 * a * x2 + 2 * a = 0 ∧
  x1 > 1/2 ∧ x2 > 1/2

theorem roots_greater_than_half_iff (a : ℝ) :
  (∃ x1 x2 : ℝ, quadratic_roots a x1 x2) ↔ (16 / 17 < a ∧ a < 2) :=
sorry

end roots_greater_than_half_iff_l503_50322


namespace expenditure_may_to_july_l503_50344

theorem expenditure_may_to_july (spent_by_may : ℝ) (spent_by_july : ℝ) (h_may : spent_by_may = 0.8) (h_july : spent_by_july = 3.5) :
  spent_by_july - spent_by_may = 2.7 :=
by
  sorry

end expenditure_may_to_july_l503_50344


namespace solution_set_of_inequality_l503_50346

theorem solution_set_of_inequality {x : ℝ} :
  {x : ℝ | |2 - 3 * x| ≥ 4} = {x : ℝ | x ≤ -2 / 3 ∨ 2 ≤ x} :=
by
  sorry

end solution_set_of_inequality_l503_50346


namespace cosine_double_angle_l503_50395

theorem cosine_double_angle (α : ℝ) (h : Real.sin α = 1 / 3) : Real.cos (2 * α) = 7 / 9 :=
by
  sorry

end cosine_double_angle_l503_50395


namespace least_pennies_l503_50334

theorem least_pennies : 
  ∃ (a : ℕ), a % 5 = 1 ∧ a % 3 = 2 ∧ a = 11 :=
by
  sorry

end least_pennies_l503_50334


namespace units_digit_powers_difference_l503_50314

theorem units_digit_powers_difference (p : ℕ) 
  (h1: p > 0) 
  (h2: p % 2 = 0) 
  (h3: (p % 10 + 2) % 10 = 8) : 
  ((p ^ 3) % 10 - (p ^ 2) % 10) % 10 = 0 :=
by
  sorry

end units_digit_powers_difference_l503_50314


namespace pointA_in_second_quadrant_l503_50387

def pointA : ℝ × ℝ := (-3, 4)

def isSecondQuadrant (p : ℝ × ℝ) : Prop :=
  p.1 < 0 ∧ p.2 > 0

theorem pointA_in_second_quadrant : isSecondQuadrant pointA :=
by
  sorry

end pointA_in_second_quadrant_l503_50387


namespace march_volume_expression_l503_50375

variable (x : ℝ) (y : ℝ)

def initial_volume : ℝ := 500
def growth_rate_volumes (x : ℝ) (initial_volume : ℝ) : ℝ := initial_volume * (1 + x)
def calculate_march_volume (x : ℝ) (initial_volume : ℝ) : ℝ := initial_volume * (1 + x)^2

theorem march_volume_expression :
  y = calculate_march_volume x initial_volume :=
sorry

end march_volume_expression_l503_50375


namespace xiao_ming_correctly_answered_question_count_l503_50325

-- Define the given conditions as constants and variables
def total_questions : ℕ := 20
def points_per_correct : ℕ := 8
def points_deducted_per_incorrect : ℕ := 5
def total_score : ℕ := 134

-- Prove that the number of correctly answered questions is 18
theorem xiao_ming_correctly_answered_question_count :
  ∃ (correct_count incorrect_count : ℕ), 
      correct_count + incorrect_count = total_questions ∧
      correct_count * points_per_correct - 
      incorrect_count * points_deducted_per_incorrect = total_score ∧
      correct_count = 18 :=
by
  sorry

end xiao_ming_correctly_answered_question_count_l503_50325


namespace correct_calculation_for_b_l503_50385

theorem correct_calculation_for_b (a : ℝ) : (a^2)^3 = a^6 := 
by sorry

end correct_calculation_for_b_l503_50385


namespace flowers_per_set_l503_50393

variable (totalFlowers : ℕ) (numSets : ℕ)

theorem flowers_per_set (h1 : totalFlowers = 270) (h2 : numSets = 3) : totalFlowers / numSets = 90 :=
by
  sorry

end flowers_per_set_l503_50393


namespace clinton_shoes_count_l503_50348

def num_hats : ℕ := 5
def num_belts : ℕ := num_hats + 2
def num_shoes : ℕ := 2 * num_belts

theorem clinton_shoes_count : num_shoes = 14 := by
  -- proof goes here
  sorry

end clinton_shoes_count_l503_50348


namespace geometric_sequence_product_l503_50392

theorem geometric_sequence_product
  (a : ℕ → ℝ)
  (a_geometric : ∃ r : ℝ, r > 0 ∧ ∀ n : ℕ, a (n + 1) = a n * r)
  (root_condition : ∃ x y : ℝ, x * y = 16 ∧ x + y = 10 ∧ a 1 = x ∧ a 19 = y) :
  a 8 * a 10 * a 12 = 64 :=
sorry

end geometric_sequence_product_l503_50392


namespace inequality_proof_l503_50342

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h_cond : x ≥ y + z) :
  (x + y) / z + (y + z) / x + (z + x) / y ≥ 7 :=
by
  sorry

end inequality_proof_l503_50342


namespace problem1_problem2_l503_50353

-- Problem 1
theorem problem1 : (Real.sqrt 8 - Real.sqrt 27 - (4 * Real.sqrt (1 / 2) + Real.sqrt 12)) = -5 * Real.sqrt 3 := by
  sorry

-- Problem 2
theorem problem2 : ((Real.sqrt 6 + Real.sqrt 12) * (2 * Real.sqrt 3 - Real.sqrt 6) - 3 * Real.sqrt 32 / (Real.sqrt 2 / 2)) = -18 := by
  sorry

end problem1_problem2_l503_50353


namespace quadratic_root_a_l503_50326

theorem quadratic_root_a (a : ℝ) : (∃ x : ℝ, x^2 + a * x + 4 = 0 ∧ x = 1) → a = -5 :=
by
  intro h
  have h1 : (1:ℝ)^2 + a * (1:ℝ) + 4 = 0 := sorry
  linarith

end quadratic_root_a_l503_50326


namespace greatest_number_remainder_l503_50305

theorem greatest_number_remainder (G R : ℕ) (h1 : 150 % G = 50) (h2 : 230 % G = 5) (h3 : 175 % G = R) (h4 : ∀ g, g ∣ 100 → g ∣ 225 → g ∣ (175 - R) → g ≤ G) : R = 0 :=
by {
  -- This is the statement only; the proof is omitted as per the instructions.
  sorry
}

end greatest_number_remainder_l503_50305


namespace inequality_half_l503_50327

-- The Lean 4 statement for the given proof problem
theorem inequality_half (a b : ℝ) (h : a > b) : (a / 2 > b / 2) :=
sorry


end inequality_half_l503_50327


namespace neither_biology_nor_chemistry_l503_50340

def science_club_total : ℕ := 80
def biology_members : ℕ := 50
def chemistry_members : ℕ := 40
def both_members : ℕ := 25

theorem neither_biology_nor_chemistry :
  (science_club_total -
  ((biology_members - both_members) +
  (chemistry_members - both_members) +
  both_members)) = 15 := by
  sorry

end neither_biology_nor_chemistry_l503_50340


namespace single_discount_eq_l503_50388

/--
A jacket is originally priced at $50. It is on sale for 25% off. After applying the sale discount, 
John uses a coupon that gives an additional 10% off of the discounted price. If there is a 5% sales 
tax on the final price, what single percent discount (before taxes) is equivalent to these series 
of discounts followed by the tax? --/
theorem single_discount_eq :
  let P0 := 50
  let discount1 := 0.25
  let discount2 := 0.10
  let tax := 0.05
  let discounted_price := P0 * (1 - discount1) * (1 - discount2)
  let after_tax_price := discounted_price * (1 + tax)
  let single_discount := (P0 - discounted_price) / P0
  single_discount * 100 = 32.5 :=
by
  sorry

end single_discount_eq_l503_50388


namespace sqrt_D_irrational_l503_50372

variable (k : ℤ)

def a := 3 * k
def b := 3 * k + 3
def c := a k + b k
def D := a k * a k + b k * b k + c k * c k

theorem sqrt_D_irrational : ¬ ∃ (r : ℚ), r * r = D k := 
by sorry

end sqrt_D_irrational_l503_50372


namespace area_enclosed_by_region_l503_50324

open Real

def condition (x y : ℝ) := abs (2 * x + 2 * y) + abs (2 * x - 2 * y) ≤ 8

theorem area_enclosed_by_region : 
  (∃ u v : ℝ, condition u v) → ∃ A : ℝ, A = 16 := 
sorry

end area_enclosed_by_region_l503_50324


namespace prime_divisor_greater_than_p_l503_50367

theorem prime_divisor_greater_than_p (p q : ℕ) (hp : Prime p) 
    (hq : Prime q) (hdiv : q ∣ 2^p - 1) : p < q := 
by
  sorry

end prime_divisor_greater_than_p_l503_50367


namespace sum_p_q_l503_50397

theorem sum_p_q (p q : ℚ) (g : ℚ → ℚ) (h : g = λ x => (x + 2) / (x^2 + p * x + q))
  (h_asymp1 : ∀ {x}, x = -1 → (x^2 + p * x + q) = 0)
  (h_asymp2 : ∀ {x}, x = 3 → (x^2 + p * x + q) = 0) :
  p + q = -5 := by
  sorry

end sum_p_q_l503_50397


namespace MNPQ_is_rectangle_l503_50377

variable {Point : Type}
variable {A B C D M N P Q : Point}

def is_parallelogram (A B C D : Point) : Prop := sorry
def altitude (X Y : Point) : Prop := sorry
def rectangle (M N P Q : Point) : Prop := sorry

theorem MNPQ_is_rectangle 
  (h_parallelogram : is_parallelogram A B C D)
  (h_alt1 : altitude B M)
  (h_alt2 : altitude B N)
  (h_alt3 : altitude D P)
  (h_alt4 : altitude D Q) :
  rectangle M N P Q :=
sorry

end MNPQ_is_rectangle_l503_50377


namespace max_tan_y_l503_50338

noncomputable def tan_y_upper_bound (x y : ℝ) (hx : 0 < x ∧ x < π / 2) (hy : 0 < y ∧ y < π / 2) 
    (h : Real.sin y = 2005 * Real.cos (x + y) * Real.sin x) : Real :=
  Real.tan y

theorem max_tan_y (x y : ℝ) (hx : 0 < x ∧ x < π / 2) (hy : 0 < y ∧ y < π / 2) 
    (h : Real.sin y = 2005 * Real.cos (x + y) * Real.sin x) : 
    tan_y_upper_bound x y hx hy h = 2005 * Real.sqrt 2006 / 4012 := 
by 
  sorry

end max_tan_y_l503_50338


namespace distance_light_in_50_years_l503_50352

/-- The distance light travels in one year, given in scientific notation -/
def distance_light_per_year : ℝ := 9.4608 * 10^12

/-- The distance light travels in 50 years is calculated -/
theorem distance_light_in_50_years :
  distance_light_per_year * 50 = 4.7304 * 10^14 :=
by
  -- the proof is not demanded, so we use sorry
  sorry

end distance_light_in_50_years_l503_50352


namespace elder_child_age_l503_50345

theorem elder_child_age (x : ℕ) (h : x + (x + 4) + (x + 8) + (x + 12) = 48) : (x + 12) = 18 :=
by
  sorry

end elder_child_age_l503_50345


namespace min_sum_of_factors_l503_50389

theorem min_sum_of_factors (a b c : ℕ) (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c) (h_prod : a * b * c = 1800) : 
  a + b + c = 64 :=
sorry

end min_sum_of_factors_l503_50389


namespace general_formula_a_n_general_formula_b_n_sum_c_n_T_n_l503_50355

open Classical

axiom S_n : ℕ → ℝ
axiom a_n : ℕ → ℝ
axiom b_n : ℕ → ℝ
axiom c_n : ℕ → ℝ
axiom T_n : ℕ → ℝ

noncomputable def general_a_n (n : ℕ) : ℝ :=
  sorry

axiom h1 : ∀ n, S_n n + a_n n = 2

theorem general_formula_a_n : ∀ n, a_n n = 1 / 2^(n-1) :=
  sorry

axiom h2 : b_n 1 = a_n 1
axiom h3 : ∀ n ≥ 2, b_n n = 3 * b_n (n-1) / (b_n (n-1) + 3)

theorem general_formula_b_n : ∀ n, b_n n = 3 / (n + 2) ∧
  (∀ n, 1 / b_n n = 1 + (n - 1) / 3) :=
  sorry

axiom h4 : ∀ n, c_n n = a_n n / b_n n

theorem sum_c_n_T_n : ∀ n, T_n n = 8 / 3 - (n + 4) / (3 * 2^(n-1)) :=
  sorry

end general_formula_a_n_general_formula_b_n_sum_c_n_T_n_l503_50355


namespace sixth_graders_bought_more_pencils_23_l503_50354

open Int

-- Conditions
def pencils_cost_whole_number_cents : Prop := ∃ n : ℕ, n > 0
def seventh_graders_total_cents := 165
def sixth_graders_total_cents := 234
def number_of_sixth_graders := 30

-- The number of sixth graders who bought more pencils than seventh graders
theorem sixth_graders_bought_more_pencils_23 :
  (seventh_graders_total_cents / 3 = 55) ∧
  (sixth_graders_total_cents / 3 = 78) →
  78 - 55 = 23 :=
by
  sorry

end sixth_graders_bought_more_pencils_23_l503_50354


namespace sqrt_addition_l503_50376

theorem sqrt_addition : Real.sqrt 8 + Real.sqrt 2 = 3 * Real.sqrt 2 :=
by
  sorry

end sqrt_addition_l503_50376


namespace soccer_minimum_wins_l503_50359

/-
Given that a soccer team has won 60% of 45 matches played so far, 
prove that the minimum number of matches that the team still needs to win to reach a winning percentage of 75% is 27.
-/
theorem soccer_minimum_wins 
  (initial_matches : ℕ)                 -- the initial number of matches
  (initial_win_rate : ℚ)                -- the initial win rate (as a percentage)
  (desired_win_rate : ℚ)                -- the desired win rate (as a percentage)
  (initial_wins : ℕ)                    -- the initial number of wins

  -- Given conditions
  (h1 : initial_matches = 45)
  (h2 : initial_win_rate = 0.60)
  (h3 : desired_win_rate = 0.75)
  (h4 : initial_wins = 27):
  
  -- To prove: the minimum number of additional matches that need to be won is 27
  ∃ (n : ℕ), (initial_wins + n) / (initial_matches + n) = desired_win_rate ∧ 
                  n = 27 :=
by 
  sorry

end soccer_minimum_wins_l503_50359


namespace value_subtracted_is_five_l503_50312

variable (N x : ℕ)

theorem value_subtracted_is_five
  (h1 : (N - x) / 7 = 7)
  (h2 : (N - 14) / 10 = 4) : x = 5 := by
  sorry

end value_subtracted_is_five_l503_50312


namespace find_A_l503_50337

theorem find_A (A B C : ℕ) (h1 : A ≠ B ∧ A ≠ C ∧ B ≠ C) 
  (h2 : 1 ≤ A ∧ A ≤ 9 ∧ 1 ≤ B ∧ B ≤ 9 ∧ 0 ≤ C ∧ C ≤ 9)
  (h3 : A * 10 + B + B * 10 + C = B * 100 + C * 10 + B) : 
  A = 9 :=
  sorry

end find_A_l503_50337


namespace maximum_reduced_price_l503_50329

theorem maximum_reduced_price (marked_price : ℝ) (cost_price : ℝ) (reduced_price : ℝ) 
    (h1 : marked_price = 240) 
    (h2 : marked_price = cost_price * 1.6) 
    (h3 : reduced_price - cost_price ≥ cost_price * 0.1) : 
    reduced_price ≤ 165 :=
sorry

end maximum_reduced_price_l503_50329


namespace soap_remaining_days_l503_50309

theorem soap_remaining_days 
  (a b c : ℝ) 
  (h1 : a > 0) 
  (h2 : b > 0) 
  (h3 : c > 0)
  (daily_consumption : ℝ)
  (h4 : daily_consumption = a * b * c / 8) 
  (h5 : ∀ t : ℝ, t > 0 → t ≤ 7 → daily_consumption = (a * b * c - (a * b * c) * (1 / 8))) :
  ∃ t : ℝ, t = 1 :=
by 
  sorry

end soap_remaining_days_l503_50309


namespace side_length_of_square_l503_50371

theorem side_length_of_square : 
  ∀ (L : ℝ), L = 28 → (L / 4) = 7 :=
by
  intro L h
  rw [h]
  norm_num

end side_length_of_square_l503_50371


namespace rectangle_midpoints_sum_l503_50343

theorem rectangle_midpoints_sum (A B C D M N O P : ℝ × ℝ)
  (hA : A = (0, 0))
  (hB : B = (4, 0))
  (hC : C = (4, 3))
  (hD : D = (0, 3))
  (hM : M = (2, 0))
  (hN : N = (4, 1.5))
  (hO : O = (2, 3))
  (hP : P = (0, 1.5)) :
  (Real.sqrt ((2 - 0) ^ 2 + (0 - 0) ^ 2) + 
  Real.sqrt ((4 - 0) ^ 2 + (1.5 - 0) ^ 2) + 
  Real.sqrt ((2 - 0) ^ 2 + (3 - 0) ^ 2) + 
  Real.sqrt ((0 - 0) ^ 2 + (1.5 - 0) ^ 2)) = 11.38 :=
by
  sorry

end rectangle_midpoints_sum_l503_50343


namespace smallest_winning_N_and_digit_sum_l503_50382

-- Definitions of operations
def B (x : ℕ) : ℕ := 3 * x
def S (x : ℕ) : ℕ := x + 100

/-- The main theorem confirming the smallest winning number and sum of its digits -/
theorem smallest_winning_N_and_digit_sum :
  ∃ (N : ℕ), 0 ≤ N ∧ N ≤ 999 ∧ (900 ≤ 9 * N + 400 ∧ 9 * N + 400 < 1000) ∧ (N = 56) ∧ (5 + 6 = 11) :=
by {
  -- Proof skipped
  sorry
}

end smallest_winning_N_and_digit_sum_l503_50382


namespace zero_point_condition_l503_50360

-- Define the function f(x) = ax + 3
def f (a x : ℝ) : ℝ := a * x + 3

-- Define that a > 2 is necessary but not sufficient condition
theorem zero_point_condition (a : ℝ) (h : a > 2) : 
  (∀ (x : ℝ), -1 ≤ x ∧ x ≤ 0 → f a x = 0) ↔ (a ≥ 3) := 
sorry

end zero_point_condition_l503_50360


namespace no_real_solution_x_squared_minus_2x_plus_3_eq_zero_l503_50317

theorem no_real_solution_x_squared_minus_2x_plus_3_eq_zero :
  ∀ x : ℝ, x^2 - 2 * x + 3 ≠ 0 :=
by
  sorry

end no_real_solution_x_squared_minus_2x_plus_3_eq_zero_l503_50317


namespace irwins_family_hike_total_distance_l503_50373

theorem irwins_family_hike_total_distance
    (d1 d2 d3 : ℝ)
    (h1 : d1 = 0.2)
    (h2 : d2 = 0.4)
    (h3 : d3 = 0.1)
    :
    d1 + d2 + d3 = 0.7 :=
by
  rw [h1, h2, h3]
  norm_num
  done

end irwins_family_hike_total_distance_l503_50373


namespace product_of_irwins_baskets_l503_50307

theorem product_of_irwins_baskets 
  (baskets_scored : Nat)
  (point_value : Nat)
  (total_baskets : baskets_scored = 2)
  (value_per_basket : point_value = 11) : 
  point_value * baskets_scored = 22 := 
by 
  sorry

end product_of_irwins_baskets_l503_50307


namespace symmetric_line_eq_l503_50384

theorem symmetric_line_eq (a b : ℝ) (ha : a ≠ 0) : 
  (∃ k m : ℝ, (∀ x: ℝ, ax + b = (k * ( -x)) + m ∧ (k = 1/a ∧ m = b/a )))  := 
sorry

end symmetric_line_eq_l503_50384


namespace point_in_fourth_quadrant_l503_50357

theorem point_in_fourth_quadrant (a b : ℝ) (h1 : -4 * a < 0) (h2 : 2 + b < 0) : 
  (a > 0) ∧ (b < -2) → (a > 0) ∧ (b < 0) := 
by
  sorry

end point_in_fourth_quadrant_l503_50357


namespace evaluate_f_3_minus_f_neg_3_l503_50316

def f (x : ℝ) : ℝ := x^4 + x^2 + 7 * x

theorem evaluate_f_3_minus_f_neg_3 : f 3 - f (-3) = 42 := by
  sorry

end evaluate_f_3_minus_f_neg_3_l503_50316


namespace markus_more_marbles_than_mara_l503_50398

variable (mara_bags : Nat) (mara_marbles_per_bag : Nat)
variable (markus_bags : Nat) (markus_marbles_per_bag : Nat)

theorem markus_more_marbles_than_mara :
  mara_bags = 12 →
  mara_marbles_per_bag = 2 →
  markus_bags = 2 →
  markus_marbles_per_bag = 13 →
  (markus_bags * markus_marbles_per_bag) - (mara_bags * mara_marbles_per_bag) = 2 :=
by
  intros
  sorry

end markus_more_marbles_than_mara_l503_50398


namespace find_number_l503_50364

theorem find_number (x : ℚ) (h : 15 + 3 * x = 6 * x - 10) : x = 25 / 3 :=
by
  sorry

end find_number_l503_50364


namespace sum_x_y_m_l503_50362

theorem sum_x_y_m (x y m : ℕ) (h1 : x >= 10 ∧ x < 100) (h2 : y >= 10 ∧ y < 100) 
  (h3 : ∃ a b : ℕ, x = 10 * a + b ∧ y = 10 * b + a ∧ a < 10 ∧ b < 10) 
  (h4 : x^2 - y^2 = 4 * m^2) : 
  x + y + m = 105 := 
sorry

end sum_x_y_m_l503_50362


namespace max_tan_A_minus_B_l503_50365

open Real

-- Given conditions
variables {A B C a b c : ℝ}

-- Assume the triangle ABC with sides a, b, c opposite to angles A, B, C respectively
-- and the equation a * cos B - b * cos A = (3 / 5) * c holds.
def condition (a b c A B C : ℝ) : Prop :=
  a * cos B - b * cos A = (3 / 5) * c

-- Prove that the maximum value of tan(A - B) is 3/4
theorem max_tan_A_minus_B (a b c A B C : ℝ) (h : condition a b c A B C) :
  ∃ t : ℝ, t = tan (A - B) ∧ 0 ≤ t ∧ t ≤ 3 / 4 :=
sorry

end max_tan_A_minus_B_l503_50365


namespace right_triangle_counterexample_l503_50339

def is_acute_angle (α : ℝ) : Prop := 0 < α ∧ α < 90

def is_right_angle (α : ℝ) : Prop := α = 90

def is_triangle (α β γ : ℝ) : Prop := α + β + γ = 180

def is_acute_triangle (α β γ : ℝ) : Prop := is_acute_angle α ∧ is_acute_angle β ∧ is_acute_angle γ

def is_right_triangle (α β γ : ℝ) : Prop := 
  (is_right_angle α ∧ is_acute_angle β ∧ is_acute_angle γ) ∨ 
  (is_acute_angle α ∧ is_right_angle β ∧ is_acute_angle γ) ∨ 
  (is_acute_angle α ∧ is_acute_angle β ∧ is_right_angle γ)

theorem right_triangle_counterexample (α β γ : ℝ) : 
  is_triangle α β γ → is_right_triangle α β γ → ¬ is_acute_triangle α β γ :=
by
  intro htri hrt hacute
  sorry

end right_triangle_counterexample_l503_50339


namespace complex_equation_l503_50370

theorem complex_equation (m n : ℝ) (i : ℂ)
  (hi : i^2 = -1)
  (h1 : m * (1 + i) = 1 + n * i) :
  ( (m + n * i) / (m - n * i) )^2 = -1 :=
sorry

end complex_equation_l503_50370


namespace rotated_point_coordinates_l503_50333

noncomputable def A : ℝ × ℝ := (1, 2)

def rotate_90_counterclockwise (p : ℝ × ℝ) : ℝ × ℝ :=
  (-p.snd, p.fst)

theorem rotated_point_coordinates :
  rotate_90_counterclockwise A = (-2, 1) :=
by
  -- Skipping the proof
  sorry

end rotated_point_coordinates_l503_50333


namespace train_speed_correct_l503_50303

def train_length : ℝ := 2500  -- Length of the train in meters.
def crossing_time : ℝ := 100  -- Time to cross the electric pole in seconds.
def expected_speed : ℝ := 25  -- Expected speed of the train in meters/second.

theorem train_speed_correct :
  (train_length / crossing_time) = expected_speed :=
by
  sorry

end train_speed_correct_l503_50303


namespace solve_for_z_l503_50300

variable {z : ℂ}
def complex_i := Complex.I

theorem solve_for_z (h : 1 - complex_i * z = -1 + complex_i * z) : z = -complex_i := by
  sorry

end solve_for_z_l503_50300


namespace determine_d_minus_r_l503_50318

theorem determine_d_minus_r :
  ∃ d r: ℕ, (∀ n ∈ [2023, 2459, 3571], n % d = r) ∧ (1 < d) ∧ (d - r = 1) :=
sorry

end determine_d_minus_r_l503_50318


namespace bus_distance_l503_50396

theorem bus_distance (w r : ℝ) (h1 : w = 0.17) (h2 : r = w + 3.67) : r = 3.84 :=
by
  sorry

end bus_distance_l503_50396


namespace determine_price_reduction_l503_50351

noncomputable def initial_cost_price : ℝ := 220
noncomputable def initial_selling_price : ℝ := 280
noncomputable def initial_daily_sales_volume : ℕ := 30
noncomputable def price_reduction_increase_rate : ℝ := 3

variable (x : ℝ)

noncomputable def daily_sales_volume (x : ℝ) : ℝ := initial_daily_sales_volume + price_reduction_increase_rate * x
noncomputable def profit_per_item (x : ℝ) : ℝ := (initial_selling_price - x) - initial_cost_price

theorem determine_price_reduction (x : ℝ) 
    (h1 : daily_sales_volume x = initial_daily_sales_volume + price_reduction_increase_rate * x)
    (h2 : profit_per_item x = 60 - x) : 
    (30 + 3 * x) * (60 - x) = 3600 → x = 30 :=
by 
  sorry

end determine_price_reduction_l503_50351


namespace wilsons_theorem_l503_50347

theorem wilsons_theorem (p : ℕ) (hp : p ≥ 2) : Nat.Prime p ↔ (Nat.factorial (p - 1) + 1) % p = 0 := 
sorry

end wilsons_theorem_l503_50347


namespace smallest_k_49_divides_binom_l503_50304

theorem smallest_k_49_divides_binom : 
  ∃ k : ℕ, 0 < k ∧ 49 ∣ Nat.choose (2 * k) k ∧ (∀ m : ℕ, 0 < m ∧ 49 ∣ Nat.choose (2 * m) m → k ≤ m) ∧ k = 25 :=
by
  sorry

end smallest_k_49_divides_binom_l503_50304


namespace employee_salary_percentage_l503_50328

theorem employee_salary_percentage (A B : ℝ)
    (h1 : A + B = 450)
    (h2 : B = 180) : (A / B) * 100 = 150 := by
  sorry

end employee_salary_percentage_l503_50328


namespace ineq_pos_xy_l503_50368

theorem ineq_pos_xy (x y : ℝ) (hx : 0 < x) (hy : 0 < y) :
  (x + y) / Real.sqrt (x * y) ≤ x / y + y / x := 
sorry

end ineq_pos_xy_l503_50368
