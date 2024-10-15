import Mathlib

namespace NUMINAMATH_GPT_alexander_has_more_pencils_l1546_154633

-- Definitions based on conditions
def asaf_age := 50
def total_age := 140
def total_pencils := 220

-- Auxiliary definitions based on conditions
def alexander_age := total_age - asaf_age
def age_difference := alexander_age - asaf_age
def asaf_pencils := 2 * age_difference
def alexander_pencils := total_pencils - asaf_pencils

-- Statement to prove
theorem alexander_has_more_pencils :
  (alexander_pencils - asaf_pencils) = 60 := sorry

end NUMINAMATH_GPT_alexander_has_more_pencils_l1546_154633


namespace NUMINAMATH_GPT_find_n_l1546_154660

theorem find_n (x k m n : ℤ) 
  (h1 : x = 82 * k + 5)
  (h2 : x + n = 41 * m + 18) :
  n = 5 :=
by
  sorry

end NUMINAMATH_GPT_find_n_l1546_154660


namespace NUMINAMATH_GPT_diagonal_rectangle_l1546_154600

theorem diagonal_rectangle (l w : ℝ) (hl : l = 20 * Real.sqrt 5) (hw : w = 10 * Real.sqrt 3) :
    Real.sqrt (l^2 + w^2) = 10 * Real.sqrt 23 :=
by
  sorry

end NUMINAMATH_GPT_diagonal_rectangle_l1546_154600


namespace NUMINAMATH_GPT_arith_seq_sum_ratio_l1546_154687

theorem arith_seq_sum_ratio 
  (S : ℕ → ℝ) 
  (a1 d : ℝ) 
  (h1 : S 1 = 1) 
  (h2 : (S 4) / (S 2) = 4) :
  (S 6) / (S 4) = 9 / 4 :=
sorry

end NUMINAMATH_GPT_arith_seq_sum_ratio_l1546_154687


namespace NUMINAMATH_GPT_sum_smallest_and_largest_prime_between_1_and_50_l1546_154651

noncomputable def smallest_prime_between_1_and_50 : ℕ := 2
noncomputable def largest_prime_between_1_and_50 : ℕ := 47

theorem sum_smallest_and_largest_prime_between_1_and_50 : 
  smallest_prime_between_1_and_50 + largest_prime_between_1_and_50 = 49 := 
by
  sorry

end NUMINAMATH_GPT_sum_smallest_and_largest_prime_between_1_and_50_l1546_154651


namespace NUMINAMATH_GPT_original_selling_price_l1546_154618

theorem original_selling_price:
  ∀ (P : ℝ), (1.17 * P - 1.10 * P = 56) → (P > 0) → 1.10 * P = 880 :=
by
  intro P h₁ h₂
  sorry

end NUMINAMATH_GPT_original_selling_price_l1546_154618


namespace NUMINAMATH_GPT_arithmetic_sequence_geometric_mean_l1546_154657

theorem arithmetic_sequence_geometric_mean (d : ℝ) (k : ℕ) (a : ℕ → ℝ)
  (h1 : d ≠ 0)
  (h2 : a 1 = 9 * d)
  (h3 : a (k + 1) = a 1 + k * d)
  (h4 : a (2 * k + 1) = a 1 + (2 * k) * d)
  (h_gm : (a k) ^ 2 = a 1 * a (2 * k)) :
  k = 4 :=
sorry

end NUMINAMATH_GPT_arithmetic_sequence_geometric_mean_l1546_154657


namespace NUMINAMATH_GPT_total_distance_covered_l1546_154639

def teams_data : List (String × Nat × Nat) :=
  [("Green Bay High", 5, 150), 
   ("Blue Ridge Middle", 7, 200),
   ("Sunset Valley Elementary", 4, 100),
   ("Riverbend Prep", 6, 250)]

theorem total_distance_covered (team : String) (members relays : Nat) :
  (team, members, relays) ∈ teams_data →
    (team = "Green Bay High" → members * relays = 750) ∧
    (team = "Blue Ridge Middle" → members * relays = 1400) ∧
    (team = "Sunset Valley Elementary" → members * relays = 400) ∧
    (team = "Riverbend Prep" → members * relays = 1500) :=
  by
    intros; sorry -- Proof omitted

end NUMINAMATH_GPT_total_distance_covered_l1546_154639


namespace NUMINAMATH_GPT_ratio_students_above_8_to_8_years_l1546_154631

-- Definitions of the problem's known conditions
def total_students : ℕ := 125
def students_below_8_years : ℕ := 25
def students_of_8_years : ℕ := 60

-- Main proof inquiry
theorem ratio_students_above_8_to_8_years :
  ∃ (A : ℕ), students_below_8_years + students_of_8_years + A = total_students ∧
             A * 3 = students_of_8_years * 2 := 
sorry

end NUMINAMATH_GPT_ratio_students_above_8_to_8_years_l1546_154631


namespace NUMINAMATH_GPT_composite_numbers_equal_l1546_154603

-- Define composite natural number
def is_composite (n : ℕ) : Prop :=
  ∃ m k : ℕ, 1 < m ∧ 1 < k ∧ n = m * k

-- Define principal divisors
def principal_divisors (n : ℕ) (principal1 principal2 : ℕ) : Prop :=
  is_composite n ∧ 
  (1 < principal1 ∧ principal1 < n) ∧ 
  (1 < principal2 ∧ principal2 < n) ∧
  principal1 * principal2 = n

-- Problem statement to prove
theorem composite_numbers_equal (a b p1 p2 : ℕ) :
  is_composite a → is_composite b →
  principal_divisors a p1 p2 → principal_divisors b p1 p2 →
  a = b :=
by
  sorry

end NUMINAMATH_GPT_composite_numbers_equal_l1546_154603


namespace NUMINAMATH_GPT_subtraction_from_double_result_l1546_154658

theorem subtraction_from_double_result (x : ℕ) (h : 2 * x = 18) : x - 4 = 5 :=
sorry

end NUMINAMATH_GPT_subtraction_from_double_result_l1546_154658


namespace NUMINAMATH_GPT_original_savings_l1546_154609

theorem original_savings (tv_cost : ℝ) (furniture_fraction : ℝ) (total_fraction : ℝ) (original_savings : ℝ) :
  tv_cost = 300 → furniture_fraction = 3 / 4 → total_fraction = 1 → 
  (total_fraction - furniture_fraction) * original_savings = tv_cost →
  original_savings = 1200 :=
by 
  intros htv hfurniture htotal hsavings_eq
  sorry

end NUMINAMATH_GPT_original_savings_l1546_154609


namespace NUMINAMATH_GPT_elsie_money_l1546_154676

noncomputable def compound_interest (P : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  P * (1 + r) ^ n

theorem elsie_money : 
  compound_interest 2500 0.04 20 = 5477.81 :=
by 
  sorry

end NUMINAMATH_GPT_elsie_money_l1546_154676


namespace NUMINAMATH_GPT_part1_part2_l1546_154661

noncomputable def f (a : ℝ) (x : ℝ) := (a - 1/2) * x^2 + Real.log x
noncomputable def g (a : ℝ) (x : ℝ) := f a x - 2 * a * x

theorem part1 (x : ℝ) (hxe : Real.exp (-1) ≤ x ∧ x ≤ Real.exp (1)) : 
    f (-1/2) x ≤ -1/2 - 1/2 * Real.log 2 ∧ f (-1/2) x ≥ 1 - Real.exp 2 := sorry

theorem part2 (h : ∀ x > 2, g a x < 0) : a ≤ 1/2 := sorry

end NUMINAMATH_GPT_part1_part2_l1546_154661


namespace NUMINAMATH_GPT_animal_products_sampled_l1546_154615

theorem animal_products_sampled
  (grains : ℕ)
  (oils : ℕ)
  (animal_products : ℕ)
  (fruits_vegetables : ℕ)
  (total_sample : ℕ)
  (total_food_types : grains + oils + animal_products + fruits_vegetables = 100)
  (sample_size : total_sample = 20)
  : (animal_products * total_sample / 100) = 6 := by
  sorry

end NUMINAMATH_GPT_animal_products_sampled_l1546_154615


namespace NUMINAMATH_GPT_arithmetic_sequence_sum_l1546_154693

theorem arithmetic_sequence_sum (a : ℕ → ℝ) (h_arith : ∃ d, ∀ n, a (n + 1) = a n + d)
  (h_roots : (a 3) * (a 10) - 3 * (a 3 + a 10) - 5 = 0) : a 5 + a 8 = 3 :=
sorry

end NUMINAMATH_GPT_arithmetic_sequence_sum_l1546_154693


namespace NUMINAMATH_GPT_geometric_sequence_min_n_l1546_154689

theorem geometric_sequence_min_n (n : ℕ) (h : 2^(n + 1) - 2 - n > 1020) : n ≥ 10 :=
sorry

end NUMINAMATH_GPT_geometric_sequence_min_n_l1546_154689


namespace NUMINAMATH_GPT_find_matrix_A_l1546_154653

theorem find_matrix_A (a b c d : ℝ) 
  (h1 : a - 3 * b = -1)
  (h2 : c - 3 * d = 3)
  (h3 : a + b = 3)
  (h4 : c + d = 3) :
  a = 2 ∧ b = 1 ∧ c = 3 ∧ d = 0 := by
  sorry

end NUMINAMATH_GPT_find_matrix_A_l1546_154653


namespace NUMINAMATH_GPT_jigsaw_puzzle_completion_l1546_154619

theorem jigsaw_puzzle_completion (p : ℝ) :
  let total_pieces := 1000
  let pieces_first_day := total_pieces * 0.10
  let remaining_after_first_day := total_pieces - pieces_first_day

  let pieces_second_day := remaining_after_first_day * (p / 100)
  let remaining_after_second_day := remaining_after_first_day - pieces_second_day

  let pieces_third_day := remaining_after_second_day * 0.30
  let remaining_after_third_day := remaining_after_second_day - pieces_third_day

  remaining_after_third_day = 504 ↔ p = 20 := 
by {
    sorry
}

end NUMINAMATH_GPT_jigsaw_puzzle_completion_l1546_154619


namespace NUMINAMATH_GPT_graphs_intersect_at_one_point_l1546_154613

theorem graphs_intersect_at_one_point (m : ℝ) (e := Real.exp 1) :
  (∀ f g : ℝ → ℝ,
    (∀ x, f x = x + Real.log x - 2 / e) ∧ (∀ x, g x = m / x) →
    ∃! x, f x = g x) ↔ (m ≥ 0 ∨ m = - (e + 1) / (e ^ 2)) :=
by sorry

end NUMINAMATH_GPT_graphs_intersect_at_one_point_l1546_154613


namespace NUMINAMATH_GPT_relationship_among_x_y_z_l1546_154650

variable (a b c d : ℝ)

-- Conditions
variables (h1 : a < b)
variables (h2 : b < c)
variables (h3 : c < d)

-- Definitions of x, y, z
def x : ℝ := (a + b) * (c + d)
def y : ℝ := (a + c) * (b + d)
def z : ℝ := (a + d) * (b + c)

-- Theorem: Prove the relationship among x, y, z
theorem relationship_among_x_y_z (h1 : a < b) (h2 : b < c) (h3 : c < d) : x a b c d < y a b c d ∧ y a b c d < z a b c d := by
  sorry

end NUMINAMATH_GPT_relationship_among_x_y_z_l1546_154650


namespace NUMINAMATH_GPT_maximum_xy_l1546_154685

variable {a b c x y : ℝ}

theorem maximum_xy 
  (h1 : a * x + b * y + 2 * c = 0)
  (h2 : c ≠ 0)
  (h3 : a * b - c^2 ≥ 0) :
  ∃ (m : ℝ), m = x * y ∧ m ≤ 1 :=
sorry

end NUMINAMATH_GPT_maximum_xy_l1546_154685


namespace NUMINAMATH_GPT_wage_increase_percentage_l1546_154648

theorem wage_increase_percentage (new_wage old_wage : ℝ) (h1 : new_wage = 35) (h2 : old_wage = 25) : 
  ((new_wage - old_wage) / old_wage) * 100 = 40 := 
by
  sorry

end NUMINAMATH_GPT_wage_increase_percentage_l1546_154648


namespace NUMINAMATH_GPT_find_abc_sum_l1546_154623

theorem find_abc_sum :
  ∀ (a b c : ℝ),
    2 * |a + 3| + 4 - b = 0 →
    c^2 + 4 * b - 4 * c - 12 = 0 →
    a + b + c = 5 :=
by
  intros a b c h1 h2
  sorry

end NUMINAMATH_GPT_find_abc_sum_l1546_154623


namespace NUMINAMATH_GPT_window_treatments_total_cost_l1546_154629

def sheers_cost_per_pair := 40
def drapes_cost_per_pair := 60
def number_of_windows := 3

theorem window_treatments_total_cost :
  (number_of_windows * sheers_cost_per_pair) + (number_of_windows * drapes_cost_per_pair) = 300 :=
by 
  -- calculations omitted
  sorry

end NUMINAMATH_GPT_window_treatments_total_cost_l1546_154629


namespace NUMINAMATH_GPT_average_marks_correct_l1546_154632

/-- Define the marks scored by Shekar in different subjects -/
def marks_math : ℕ := 76
def marks_science : ℕ := 65
def marks_social_studies : ℕ := 82
def marks_english : ℕ := 67
def marks_biology : ℕ := 55

/-- Define the total marks scored by Shekar -/
def total_marks : ℕ := marks_math + marks_science + marks_social_studies + marks_english + marks_biology

/-- Define the number of subjects -/
def num_subjects : ℕ := 5

/-- Define the average marks scored by Shekar -/
def average_marks : ℕ := total_marks / num_subjects

theorem average_marks_correct : average_marks = 69 := by
  -- We need to show that the average marks is 69
  sorry

end NUMINAMATH_GPT_average_marks_correct_l1546_154632


namespace NUMINAMATH_GPT_find_multiplier_l1546_154667

theorem find_multiplier (x y n : ℤ) (h1 : 3 * x + y = 40) (h2 : 2 * x - y = 20) (h3 : y^2 = 16) :
  n * y^2 = 48 :=
by 
  -- proof goes here
  sorry

end NUMINAMATH_GPT_find_multiplier_l1546_154667


namespace NUMINAMATH_GPT_product_less_by_nine_times_l1546_154638

theorem product_less_by_nine_times (x y : ℕ) (hx : 10 ≤ x ∧ x < 100) (hy : y < 10) : 
  (x * y) * 10 - x * y = 9 * (x * y) := 
by
  sorry

end NUMINAMATH_GPT_product_less_by_nine_times_l1546_154638


namespace NUMINAMATH_GPT_total_oranges_l1546_154688

theorem total_oranges (joan_oranges : ℕ) (sara_oranges : ℕ) 
                      (h1 : joan_oranges = 37) 
                      (h2 : sara_oranges = 10) :
  joan_oranges + sara_oranges = 47 := by
  sorry

end NUMINAMATH_GPT_total_oranges_l1546_154688


namespace NUMINAMATH_GPT_factorization_correct_l1546_154681

variable (a b : ℝ)

theorem factorization_correct :
  12 * a ^ 3 * b - 12 * a ^ 2 * b + 3 * a * b = 3 * a * b * (2 * a - 1) ^ 2 :=
by 
  sorry

end NUMINAMATH_GPT_factorization_correct_l1546_154681


namespace NUMINAMATH_GPT_isosceles_triangle_perimeter_l1546_154636

-- Define the given quadratic equation
def quadratic_eq (x : ℝ) : Prop := x^2 - 6 * x + 8 = 0

-- Define the roots based on factorization of the given equation
def root1 := 2
def root2 := 4

-- Define the perimeter of the isosceles triangle given the roots
def triangle_perimeter := root2 + root2 + root1

-- Prove that the perimeter of the isosceles triangle is 10
theorem isosceles_triangle_perimeter : triangle_perimeter = 10 :=
by
  -- We need to verify the solution without providing the steps explicitly
  sorry

end NUMINAMATH_GPT_isosceles_triangle_perimeter_l1546_154636


namespace NUMINAMATH_GPT_intersection_equiv_l1546_154640

-- Define the sets M and N based on the given conditions
def M : Set ℤ := {-1, 0, 1, 2, 3}
def N : Set ℤ := {x | x < 0 ∨ x > 2}

-- The main proof statement
theorem intersection_equiv : M ∩ N = {-1, 3} :=
by
  sorry -- proof goes here

end NUMINAMATH_GPT_intersection_equiv_l1546_154640


namespace NUMINAMATH_GPT_impossible_piles_of_three_l1546_154665

theorem impossible_piles_of_three (n : ℕ) (h1 : n = 1001)
  (h2 : ∀ p : ℕ, p > 1 → ∃ a b : ℕ, a + b = p - 1 ∧ a ≤ b) : 
  ¬ (∃ piles : List ℕ, ∀ pile ∈ piles, pile = 3 ∧ (piles.sum = n + piles.length)) :=
by
  sorry

end NUMINAMATH_GPT_impossible_piles_of_three_l1546_154665


namespace NUMINAMATH_GPT_next_birthday_monday_l1546_154625
open Nat

-- Define the basic structure and parameters of our problem
def is_leap_year (year : ℕ) : Prop := 
  (year % 4 = 0 ∧ year % 100 ≠ 0) ∨ (year % 400 = 0)

def day_of_week (start_day : ℕ) (year_diff : ℕ) (is_leap : ℕ → Prop) : ℕ :=
  (start_day + year_diff + (year_diff / 4) - (year_diff / 100) + (year_diff / 400)) % 7

-- Specify problem conditions
def initial_year := 2009
def initial_day := 5 -- 2009-06-18 is Friday, which is 5 if we start counting from Sunday as 0
def end_day := 1 -- target day is Monday, which is 1

-- Main theorem
theorem next_birthday_monday : ∃ year, year > initial_year ∧
  day_of_week initial_day (year - initial_year) is_leap_year = end_day := by
  use 2017
  -- The proof would go here, skipping with sorry
  sorry

end NUMINAMATH_GPT_next_birthday_monday_l1546_154625


namespace NUMINAMATH_GPT_total_students_surveyed_l1546_154601

variable (F E S FE FS ES FES N T : ℕ)

def only_one_language := 230
def exactly_two_languages := 190
def all_three_languages := 40
def no_language := 60

-- Summing up all categories
def total_students := only_one_language + exactly_two_languages + all_three_languages + no_language

theorem total_students_surveyed (h1 : F + E + S = only_one_language) 
    (h2 : FE + FS + ES = exactly_two_languages) 
    (h3 : FES = all_three_languages) 
    (h4 : N = no_language) 
    (h5 : T = F + E + S + FE + FS + ES + FES + N) : 
    T = total_students :=
by
  rw [total_students, only_one_language, exactly_two_languages, all_three_languages, no_language]
  sorry

end NUMINAMATH_GPT_total_students_surveyed_l1546_154601


namespace NUMINAMATH_GPT_card_draw_probability_l1546_154699

theorem card_draw_probability : 
  let P1 := (12 / 52 : ℚ) * (4 / 51 : ℚ) * (13 / 50 : ℚ)
  let P2 := (1 / 52 : ℚ) * (3 / 51 : ℚ) * (13 / 50 : ℚ)
  P1 + P2 = (63 / 107800 : ℚ) :=
by
  sorry

end NUMINAMATH_GPT_card_draw_probability_l1546_154699


namespace NUMINAMATH_GPT_find_m_values_l1546_154674

def has_unique_solution (m : ℝ) (A : Set ℝ) : Prop :=
  ∀ x1 x2, x1 ∈ A → x2 ∈ A → x1 = x2

theorem find_m_values :
  {m : ℝ | ∃ A : Set ℝ, has_unique_solution m A ∧ (A = {x | m * x^2 + 2 * x + 3 = 0})} = {0, 1/3} :=
by
  sorry

end NUMINAMATH_GPT_find_m_values_l1546_154674


namespace NUMINAMATH_GPT_smallest_x_l1546_154646

theorem smallest_x (x : ℝ) (h : |4 * x + 12| = 40) : x = -13 :=
sorry

end NUMINAMATH_GPT_smallest_x_l1546_154646


namespace NUMINAMATH_GPT_tunnel_length_l1546_154642

-- Definitions as per the conditions
def train_length : ℚ := 2  -- 2 miles
def train_speed : ℚ := 40  -- 40 miles per hour

def speed_in_miles_per_minute (speed_mph : ℚ) : ℚ :=
  speed_mph / 60  -- Convert speed from miles per hour to miles per minute

def time_travelled_in_minutes : ℚ := 5  -- 5 minutes

-- Theorem statement to prove the length of the tunnel
theorem tunnel_length (h1 : train_length = 2) (h2 : train_speed = 40) :
  (speed_in_miles_per_minute train_speed * time_travelled_in_minutes) - train_length = 4 / 3 :=
by
  sorry  -- Proof not included

end NUMINAMATH_GPT_tunnel_length_l1546_154642


namespace NUMINAMATH_GPT_problem_l1546_154670

def f (u : ℝ) : ℝ := u^2 - 2

theorem problem : f 3 = 7 := 
by sorry

end NUMINAMATH_GPT_problem_l1546_154670


namespace NUMINAMATH_GPT_art_club_artworks_l1546_154634

theorem art_club_artworks (students : ℕ) (artworks_per_student_per_quarter : ℕ)
  (quarters_per_year : ℕ) (years : ℕ) :
  students = 15 → artworks_per_student_per_quarter = 2 → 
  quarters_per_year = 4 → years = 2 → 
  (students * artworks_per_student_per_quarter * quarters_per_year * years) = 240 :=
by
  intros
  sorry

end NUMINAMATH_GPT_art_club_artworks_l1546_154634


namespace NUMINAMATH_GPT_medium_stores_to_select_l1546_154645

-- Definitions based on conditions in a)
def total_stores := 1500
def ratio_large := 1
def ratio_medium := 5
def ratio_small := 9
def sample_size := 30
def medium_proportion := ratio_medium / (ratio_large + ratio_medium + ratio_small)

-- Main theorem to prove
theorem medium_stores_to_select : (sample_size * medium_proportion) = 10 :=
by sorry

end NUMINAMATH_GPT_medium_stores_to_select_l1546_154645


namespace NUMINAMATH_GPT_total_money_spent_l1546_154664

/-- Erika, Elizabeth, Emma, and Elsa went shopping on Wednesday.
Emma spent $58.
Erika spent $20 more than Emma.
Elsa spent twice as much as Emma.
Elizabeth spent four times as much as Elsa.
Erika received a 10% discount on what she initially spent.
Elizabeth had to pay a 6% tax on her purchases.
Prove that the total amount of money they spent is $736.04.
-/
theorem total_money_spent :
  let emma_spent := 58
  let erika_initial_spent := emma_spent + 20
  let erika_discount := 0.10 * erika_initial_spent
  let erika_final_spent := erika_initial_spent - erika_discount
  let elsa_spent := 2 * emma_spent
  let elizabeth_initial_spent := 4 * elsa_spent
  let elizabeth_tax := 0.06 * elizabeth_initial_spent
  let elizabeth_final_spent := elizabeth_initial_spent + elizabeth_tax
  let total_spent := emma_spent + erika_final_spent + elsa_spent + elizabeth_final_spent
  total_spent = 736.04 := by
  sorry

end NUMINAMATH_GPT_total_money_spent_l1546_154664


namespace NUMINAMATH_GPT_arithmetic_sequence_problem_l1546_154697

theorem arithmetic_sequence_problem 
    (a : ℕ → ℝ)  -- Define the arithmetic sequence as a function from natural numbers to reals
    (a1 : ℝ)  -- Represent a₁ as a1
    (a8 : ℝ)  -- Represent a₈ as a8
    (a9 : ℝ)  -- Represent a₉ as a9
    (a10 : ℝ)  -- Represent a₁₀ as a10
    (a15 : ℝ)  -- Represent a₁₅ as a15
    (h1 : a 1 = a1)  -- Hypothesis that a(1) is represented by a1
    (h8 : a 8 = a8)  -- Hypothesis that a(8) is represented by a8
    (h9 : a 9 = a9)  -- Hypothesis that a(9) is represented by a9
    (h10 : a 10 = a10)  -- Hypothesis that a(10) is represented by a10
    (h15 : a 15 = a15)  -- Hypothesis that a(15) is represented by a15
    (h_condition : a1 + 2 * a8 + a15 = 96)  -- Condition of the problem
    : 2 * a9 - a10 = 24 := 
sorry

end NUMINAMATH_GPT_arithmetic_sequence_problem_l1546_154697


namespace NUMINAMATH_GPT_rug_area_correct_l1546_154672

def floor_length : ℕ := 10
def floor_width : ℕ := 8
def strip_width : ℕ := 2

def adjusted_length : ℕ := floor_length - 2 * strip_width
def adjusted_width : ℕ := floor_width - 2 * strip_width

def area_floor : ℕ := floor_length * floor_width
def area_rug : ℕ := adjusted_length * adjusted_width

theorem rug_area_correct : area_rug = 24 := by
  sorry

end NUMINAMATH_GPT_rug_area_correct_l1546_154672


namespace NUMINAMATH_GPT_part_one_part_two_l1546_154684

variable (α : Real) (h : Real.tan α = 2)

theorem part_one (h : Real.tan α = 2) : 
  (4 * Real.sin α - 2 * Real.cos α) / (5 * Real.cos α + 3 * Real.sin α) = 6 / 11 := 
by
  sorry

theorem part_two (h : Real.tan α = 2) : 
  (1 / 4 * Real.sin α ^ 2 + 1 / 3 * Real.sin α * Real.cos α + 1 / 2 * Real.cos α ^ 2 + 1) = 43 / 30 := 
by
  sorry

end NUMINAMATH_GPT_part_one_part_two_l1546_154684


namespace NUMINAMATH_GPT_trader_profit_l1546_154612

noncomputable def profit_percentage (P : ℝ) : ℝ :=
  let purchased_price := 0.72 * P
  let market_increase := 1.05 * purchased_price
  let expenses := 0.08 * market_increase
  let net_price := market_increase - expenses
  let first_sale_price := 1.50 * net_price
  let final_sale_price := 1.25 * first_sale_price
  let profit := final_sale_price - P
  (profit / P) * 100

theorem trader_profit
  (P : ℝ) 
  (hP : 0 < P) :
  profit_percentage P = 30.41 :=
by
  sorry

end NUMINAMATH_GPT_trader_profit_l1546_154612


namespace NUMINAMATH_GPT_brick_wall_l1546_154604

theorem brick_wall (x : ℕ) 
  (h1 : x / 9 * 9 = x)
  (h2 : x / 10 * 10 = x)
  (h3 : 5 * (x / 9 + x / 10 - 10) = x) :
  x = 900 := 
sorry

end NUMINAMATH_GPT_brick_wall_l1546_154604


namespace NUMINAMATH_GPT_denis_fourth_board_score_l1546_154698

theorem denis_fourth_board_score :
  ∀ (darts_per_board points_first_board points_second_board points_third_board points_total_boards : ℕ),
    darts_per_board = 3 →
    points_first_board = 30 →
    points_second_board = 38 →
    points_third_board = 41 →
    points_total_boards = (points_first_board + points_second_board + points_third_board) / 2 →
    points_total_boards = 34 :=
by
  intros darts_per_board points_first_board points_second_board points_third_board points_total_boards h1 h2 h3 h4 h5
  sorry

end NUMINAMATH_GPT_denis_fourth_board_score_l1546_154698


namespace NUMINAMATH_GPT_license_plates_possible_l1546_154694

open Function Nat

theorem license_plates_possible :
  let characters := ['B', 'C', 'D', '1', '2', '2', '5']
  let license_plate_length := 4
  let plate_count_with_two_twos := (choose 4 2) * (choose 5 2 * 2!)
  let plate_count_with_one_two := (choose 4 1) * (choose 5 3 * 3!)
  let plate_count_with_no_twos := (choose 5 4) * 4!
  let plate_count_with_three_twos := (choose 4 3) * (choose 4 1)
  plate_count_with_two_twos + plate_count_with_one_two + plate_count_with_no_twos + plate_count_with_three_twos = 496 := 
  sorry

end NUMINAMATH_GPT_license_plates_possible_l1546_154694


namespace NUMINAMATH_GPT_value_to_subtract_l1546_154647

theorem value_to_subtract (N x : ℕ) 
  (h1 : (N - x) / 7 = 7) 
  (h2 : (N - 2) / 13 = 4) : x = 5 :=
by 
  sorry

end NUMINAMATH_GPT_value_to_subtract_l1546_154647


namespace NUMINAMATH_GPT_increase_corrosion_with_more_active_metal_rivets_l1546_154690

-- Definitions representing conditions
def corrosion_inhibitor (P : Type) : Prop := true
def more_active_metal_rivets (P : Type) : Prop := true
def less_active_metal_rivets (P : Type) : Prop := true
def painted_parts (P : Type) : Prop := true

-- Main theorem statement
theorem increase_corrosion_with_more_active_metal_rivets (P : Type) 
  (h1 : corrosion_inhibitor P)
  (h2 : more_active_metal_rivets P)
  (h3 : less_active_metal_rivets P)
  (h4 : painted_parts P) : 
  more_active_metal_rivets P :=
by {
  -- proof goes here
  sorry
}

end NUMINAMATH_GPT_increase_corrosion_with_more_active_metal_rivets_l1546_154690


namespace NUMINAMATH_GPT_largest_multiple_5_6_lt_1000_is_990_l1546_154669

theorem largest_multiple_5_6_lt_1000_is_990 : ∃ n, (n < 1000) ∧ (n % 5 = 0) ∧ (n % 6 = 0) ∧ n = 990 :=
by 
  -- Needs to follow the procedures to prove it step-by-step
  sorry

end NUMINAMATH_GPT_largest_multiple_5_6_lt_1000_is_990_l1546_154669


namespace NUMINAMATH_GPT_solve_equation_1_solve_equation_2_l1546_154662

theorem solve_equation_1 (x : ℝ) (h : 0.5 * x + 1.1 = 6.5 - 1.3 * x) : x = 3 :=
  by sorry

theorem solve_equation_2 (x : ℝ) (h : (1 / 6) * (3 * x - 9) = (2 / 5) * x - 3) : x = -15 :=
  by sorry

end NUMINAMATH_GPT_solve_equation_1_solve_equation_2_l1546_154662


namespace NUMINAMATH_GPT_algebraic_expression_value_l1546_154655

theorem algebraic_expression_value
  (x : ℝ)
  (h : 2 * x^2 + 3 * x + 1 = 10) :
  4 * x^2 + 6 * x + 1 = 19 := 
by
  sorry

end NUMINAMATH_GPT_algebraic_expression_value_l1546_154655


namespace NUMINAMATH_GPT_train_time_to_pass_platform_l1546_154621

-- Definitions as per the conditions
def length_of_train : ℕ := 720 -- Length of train in meters
def speed_of_train_kmh : ℕ := 72 -- Speed of train in km/hr
def length_of_platform : ℕ := 280 -- Length of platform in meters

-- Conversion factor and utility functions
def kmh_to_ms (speed : ℕ) : ℕ :=
  speed * 1000 / 3600

def total_distance (train_len platform_len : ℕ) : ℕ :=
  train_len + platform_len

def time_to_pass (distance speed_ms : ℕ) : ℕ :=
  distance / speed_ms

-- Main statement to be proven
theorem train_time_to_pass_platform :
  time_to_pass (total_distance length_of_train length_of_platform) (kmh_to_ms speed_of_train_kmh) = 50 :=
by
  sorry

end NUMINAMATH_GPT_train_time_to_pass_platform_l1546_154621


namespace NUMINAMATH_GPT_mary_younger_than_albert_l1546_154677

variable (A M B : ℕ)

noncomputable def albert_age := 4 * B
noncomputable def mary_age := A / 2
noncomputable def betty_age := 4

theorem mary_younger_than_albert (h1 : A = 2 * M) (h2 : A = 4 * 4) (h3 : 4 = 4) :
  A - M = 8 :=
sorry

end NUMINAMATH_GPT_mary_younger_than_albert_l1546_154677


namespace NUMINAMATH_GPT_ratio_of_80_pencils_l1546_154610

theorem ratio_of_80_pencils (C S : ℝ)
  (CP : ℝ := 80 * C)
  (L : ℝ := 30 * S)
  (SP : ℝ := 80 * S)
  (h : CP = SP + L) :
  CP / SP = 11 / 8 :=
by
  -- Start the proof
  sorry

end NUMINAMATH_GPT_ratio_of_80_pencils_l1546_154610


namespace NUMINAMATH_GPT_Iggy_Tuesday_Run_l1546_154695

def IggyRunsOnTuesday (total_miles : ℕ) (monday_miles : ℕ) (wednesday_miles : ℕ) (thursday_miles : ℕ) (friday_miles : ℕ) : ℕ :=
  total_miles - (monday_miles + wednesday_miles + thursday_miles + friday_miles)

theorem Iggy_Tuesday_Run :
  let monday_miles := 3
  let wednesday_miles := 6
  let thursday_miles := 8
  let friday_miles := 3
  let total_miles := 240 / 10
  IggyRunsOnTuesday total_miles monday_miles wednesday_miles thursday_miles friday_miles = 4 :=
by
  sorry

end NUMINAMATH_GPT_Iggy_Tuesday_Run_l1546_154695


namespace NUMINAMATH_GPT_simplify_fraction_l1546_154682

noncomputable def simplify_expression (x : ℂ) : Prop :=
  (x^2 - 4*x + 3) / (x^2 - 6*x + 8) / ((x^2 - 6*x + 5) / (x^2 - 8*x + 15)) =
  (x - 3) / (x^2 - 6*x + 8)

theorem simplify_fraction (x : ℂ) : simplify_expression x :=
by
  sorry

end NUMINAMATH_GPT_simplify_fraction_l1546_154682


namespace NUMINAMATH_GPT_A_contribution_is_500_l1546_154686

-- Define the contributions
variables (A B C : ℕ)

-- Total amount spent
def total_contribution : ℕ := 820

-- Given ratios
def ratio_A_to_B : ℕ × ℕ := (5, 2)
def ratio_B_to_C : ℕ × ℕ := (5, 3)

-- Condition stating the sum of contributions
axiom sum_contribution : A + B + C = total_contribution

-- Conditions stating the ratios
axiom ratio_A_B : 5 * B = 2 * A
axiom ratio_B_C : 5 * C = 3 * B

-- The statement to prove
theorem A_contribution_is_500 : A = 500 :=
by
  sorry

end NUMINAMATH_GPT_A_contribution_is_500_l1546_154686


namespace NUMINAMATH_GPT_k1_k2_ratio_l1546_154620

theorem k1_k2_ratio (a b k k1 k2 : ℝ)
  (h1 : a^2 * k - (k - 1) * a + 5 = 0)
  (h2 : b^2 * k - (k - 1) * b + 5 = 0)
  (h3 : (a / b) + (b / a) = 4/5)
  (h4 : k1^2 - 16 * k1 + 1 = 0)
  (h5 : k2^2 - 16 * k2 + 1 = 0) :
  (k1 / k2) + (k2 / k1) = 254 := by
  sorry

end NUMINAMATH_GPT_k1_k2_ratio_l1546_154620


namespace NUMINAMATH_GPT_sequence_third_term_l1546_154675

theorem sequence_third_term (a m : ℤ) (h_a_neg : a < 0) (h_a1 : a + m = 2) (h_a2 : a^2 + m = 4) :
  (a^3 + m = 2) :=
by
  sorry

end NUMINAMATH_GPT_sequence_third_term_l1546_154675


namespace NUMINAMATH_GPT_no_solutions_l1546_154643

theorem no_solutions {x y : ℤ} :
  (x ≠ 1) → (y ≠ 1) →
  ((x^7 - 1) / (x - 1) = y^5 - 1) →
  false :=
by sorry

end NUMINAMATH_GPT_no_solutions_l1546_154643


namespace NUMINAMATH_GPT_total_distance_l1546_154627

theorem total_distance (D : ℕ) 
  (h1 : (1 / 2 * D : ℝ) + (1 / 4 * (1 / 2 * D : ℝ)) + 105 = D) : 
  D = 280 :=
by
  sorry

end NUMINAMATH_GPT_total_distance_l1546_154627


namespace NUMINAMATH_GPT_intersection_points_l1546_154616

theorem intersection_points (x : ℝ) (hx : 0 ≤ x ∧ x ≤ 2 * Real.pi) :
  (∃ x1 x2, 0 ≤ x1 ∧ x1 ≤ 2 * Real.pi ∧ 
   0 ≤ x2 ∧ x2 ≤ 2 * Real.pi ∧ 
   x1 ≠ x2 ∧ 
   1 + Real.sin x1 = 3 / 2 ∧ 
   1 + Real.sin x2 = 3 / 2 ) ∧ 
  (∀ x, (0 ≤ x ∧ x ≤ 2 * Real.pi ∧ 1 + Real.sin x = 3 / 2) → 
   (x = x1 ∨ x = x2)) :=
sorry

end NUMINAMATH_GPT_intersection_points_l1546_154616


namespace NUMINAMATH_GPT_average_after_discard_l1546_154652

theorem average_after_discard (avg : ℝ) (n : ℕ) (a b : ℝ) (new_avg : ℝ) :
  avg = 62 →
  n = 50 →
  a = 45 →
  b = 55 →
  new_avg = 62.5 →
  (avg * n - (a + b)) / (n - 2) = new_avg := 
by
  intros h_avg h_n h_a h_b h_new_avg
  rw [h_avg, h_n, h_a, h_b, h_new_avg]
  sorry

end NUMINAMATH_GPT_average_after_discard_l1546_154652


namespace NUMINAMATH_GPT_coefficients_sum_binomial_coefficients_sum_l1546_154630

theorem coefficients_sum (x : ℝ) (h : (x + 2 / x)^6 = coeff_sum) : coeff_sum = 729 := 
sorry

theorem binomial_coefficients_sum (x : ℝ) (h : (x + 2 / x)^6 = binom_coeff_sum) : binom_coeff_sum = 64 := 
sorry

end NUMINAMATH_GPT_coefficients_sum_binomial_coefficients_sum_l1546_154630


namespace NUMINAMATH_GPT_kolya_is_wrong_l1546_154668

def pencils_problem_statement (at_least_four_blue : Prop) 
                              (at_least_five_green : Prop) 
                              (at_least_three_blue_and_four_green : Prop) 
                              (at_least_four_blue_and_four_green : Prop) : 
                              Prop :=
  ∃ (B G : ℕ), -- B represents the number of blue pencils, G represents the number of green pencils
    ((B ≥ 4) ∧ (G ≥ 4)) ∧ -- Vasya's statement (at least 4 blue), Petya's and Misha's combined statement (at least 4 green)
    at_least_four_blue ∧ -- Vasya's statement (there are at least 4 blue pencils)
    (at_least_five_green ↔ G ≥ 5) ∧ -- Kolya's statement (there are at least 5 green pencils)
    at_least_three_blue_and_four_green ∧ -- Petya's statement (at least 3 blue and 4 green)
    at_least_four_blue_and_four_green -- Misha's statement (at least 4 blue and 4 green)

theorem kolya_is_wrong (at_least_four_blue : Prop) 
                        (at_least_five_green : Prop) 
                        (at_least_three_blue_and_four_green : Prop) 
                        (at_least_four_blue_and_four_green : Prop) : 
                        pencils_problem_statement at_least_four_blue 
                                                  at_least_five_green 
                                                  at_least_three_blue_and_four_green 
                                                  at_least_four_blue_and_four_green :=
sorry

end NUMINAMATH_GPT_kolya_is_wrong_l1546_154668


namespace NUMINAMATH_GPT_quadratic_value_at_point_l1546_154671

theorem quadratic_value_at_point :
  ∃ a b c, 
    (∃ y, y = a * 2^2 + b * 2 + c ∧ y = 7) ∧
    (∃ y, y = a * 0^2 + b * 0 + c ∧ y = -7) ∧
    (∃ y, y = a * 5^2 + b * 5 + c ∧ y = -24.5) := 
sorry

end NUMINAMATH_GPT_quadratic_value_at_point_l1546_154671


namespace NUMINAMATH_GPT_find_m_l1546_154679

theorem find_m (m : ℚ) : 
  (∃ m, (∀ x y z : ℚ, ((x, y) = (2, 9) ∨ (x, y) = (15, m) ∨ (x, y) = (35, 4)) ∧ 
  (∀ a b c d e f : ℚ, ((a, b) = (2, 9) ∨ (a, b) = (15, m) ∨ (a, b) = (35, 4)) → 
  ((b - d) / (a - c) = (f - d) / (e - c))) → m = 232 / 33)) :=
sorry

end NUMINAMATH_GPT_find_m_l1546_154679


namespace NUMINAMATH_GPT_percentage_problem_l1546_154611

theorem percentage_problem (P : ℕ) : (P / 100 * 400 = 20 / 100 * 700) → P = 35 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_percentage_problem_l1546_154611


namespace NUMINAMATH_GPT_vector_addition_example_l1546_154628

def vector_addition (a b : ℝ × ℝ) : ℝ × ℝ := (a.1 + b.1, a.2 + b.2)

theorem vector_addition_example : vector_addition (1, -1) (-1, 2) = (0, 1) := 
by 
  unfold vector_addition 
  simp
  sorry

end NUMINAMATH_GPT_vector_addition_example_l1546_154628


namespace NUMINAMATH_GPT_decreasing_interval_l1546_154683

noncomputable def f (x : ℝ) : ℝ := x / 2 + Real.cos x

theorem decreasing_interval : ∀ x ∈ Set.Ioo (Real.pi / 6) (5 * Real.pi / 6), 
  (1 / 2 - Real.sin x) < 0 := sorry

end NUMINAMATH_GPT_decreasing_interval_l1546_154683


namespace NUMINAMATH_GPT_C_paisa_for_A_rupee_l1546_154656

variable (A B C : ℝ)
variable (C_share : ℝ) (total_sum : ℝ)
variable (B_per_A : ℝ)

noncomputable def C_paisa_per_A_rupee (A B C C_share total_sum B_per_A : ℝ) : ℝ :=
  let C_paisa := C_share * 100
  C_paisa / A

theorem C_paisa_for_A_rupee : C_share = 32 ∧ total_sum = 164 ∧ B_per_A = 0.65 → 
  C_paisa_per_A_rupee A B C C_share total_sum B_per_A = 40 := by
  sorry

end NUMINAMATH_GPT_C_paisa_for_A_rupee_l1546_154656


namespace NUMINAMATH_GPT_part1_part2_part3_l1546_154649

-- Part 1
theorem part1 : (1 > -1) ∧ (1 < 2) ∧ (-(1/2) > -1) ∧ (-(1/2) < 2) := 
  by sorry

-- Part 2
theorem part2 (k : Real) : (3 < k) ∧ (k ≤ 4) := 
  by sorry

-- Part 3
theorem part3 (m : Real) : (2 < m) ∧ (m ≤ 3) := 
  by sorry

end NUMINAMATH_GPT_part1_part2_part3_l1546_154649


namespace NUMINAMATH_GPT_efficiency_ratio_l1546_154637

theorem efficiency_ratio (A B : ℝ) (h1 : A + B = 1 / 26) (h2 : B = 1 / 39) : A / B = 1 / 2 := 
by
  sorry

end NUMINAMATH_GPT_efficiency_ratio_l1546_154637


namespace NUMINAMATH_GPT_least_value_of_x_l1546_154607

theorem least_value_of_x (x p : ℕ) (h1 : x > 0) (h2 : Nat.Prime p) (h3 : x = 11 * p * 2) : x = 44 := 
by
  sorry

end NUMINAMATH_GPT_least_value_of_x_l1546_154607


namespace NUMINAMATH_GPT_combined_points_kjm_l1546_154605

theorem combined_points_kjm {P B K J M H C E: ℕ} 
  (total_points : P + B + K + J + M = 81)
  (paige_points : P = 21)
  (brian_points : B = 20)
  (karen_jennifer_michael_sum : K + J + M = 40)
  (karen_scores : ∀ p, K = 2 * p + 5 * (H - p))
  (jennifer_scores : ∀ p, J = 2 * p + 5 * (C - p))
  (michael_scores : ∀ p, M = 2 * p + 5 * (E - p)) :
  K + J + M = 40 :=
by sorry

end NUMINAMATH_GPT_combined_points_kjm_l1546_154605


namespace NUMINAMATH_GPT_playground_girls_count_l1546_154641

theorem playground_girls_count (boys : ℕ) (total_children : ℕ) 
  (h_boys : boys = 35) (h_total : total_children = 63) : 
  ∃ girls : ℕ, girls = 28 ∧ girls = total_children - boys := 
by 
  sorry

end NUMINAMATH_GPT_playground_girls_count_l1546_154641


namespace NUMINAMATH_GPT_solve_quadratic_l1546_154635

theorem solve_quadratic (x : ℝ) (h_pos : x > 0) (h_eq : 6 * x^2 + 9 * x - 24 = 0) : x = 4 / 3 :=
by
  sorry

end NUMINAMATH_GPT_solve_quadratic_l1546_154635


namespace NUMINAMATH_GPT_A_odot_B_correct_l1546_154696

open Set

def A : Set ℝ := { x | x ≥ 1 }
def B : Set ℝ := { x | x < 0 ∨ x > 2 }
def A_union_B : Set ℝ := A ∪ B
def A_inter_B : Set ℝ := A ∩ B
def A_odot_B : Set ℝ := { x | x ∈ A_union_B ∧ x ∉ A_inter_B }

theorem A_odot_B_correct : A_odot_B = (Iio 0) ∪ Icc 1 2 :=
by
  sorry

end NUMINAMATH_GPT_A_odot_B_correct_l1546_154696


namespace NUMINAMATH_GPT_find_minimum_x2_x1_l1546_154659

noncomputable def f (x : ℝ) : ℝ := Real.exp (2 * x)
noncomputable def g (x : ℝ) : ℝ := Real.log x + 1 / 2

theorem find_minimum_x2_x1 (x1 : ℝ) :
  ∃ x2 : {r : ℝ // 0 < r}, f x1 = g x2 → (x2 - x1) ≥ 1 + Real.log 2 / 2 :=
by
  -- Proof
  sorry

end NUMINAMATH_GPT_find_minimum_x2_x1_l1546_154659


namespace NUMINAMATH_GPT_nine_a_plus_a_plus_nine_l1546_154614

theorem nine_a_plus_a_plus_nine (A : Nat) (hA : 0 < A) : 
  10 * A + 9 = 9 * A + (A + 9) := 
by 
  sorry

end NUMINAMATH_GPT_nine_a_plus_a_plus_nine_l1546_154614


namespace NUMINAMATH_GPT_min_a_n_l1546_154608

def a_n (n : ℕ) : ℤ := n^2 - 8 * n + 5

theorem min_a_n : ∃ n : ℕ, ∀ m : ℕ, a_n n ≤ a_n m ∧ a_n n = -11 :=
by
  sorry

end NUMINAMATH_GPT_min_a_n_l1546_154608


namespace NUMINAMATH_GPT_borgnine_lizards_l1546_154692

theorem borgnine_lizards (chimps lions tarantulas total_legs : ℕ) (legs_per_chimp legs_per_lion legs_per_tarantula legs_per_lizard lizards : ℕ)
  (H_chimps : chimps = 12)
  (H_lions : lions = 8)
  (H_tarantulas : tarantulas = 125)
  (H_total_legs : total_legs = 1100)
  (H_legs_per_chimp : legs_per_chimp = 4)
  (H_legs_per_lion : legs_per_lion = 4)
  (H_legs_per_tarantula : legs_per_tarantula = 8)
  (H_legs_per_lizard : legs_per_lizard = 4)
  (H_seen_legs : total_legs = (chimps * legs_per_chimp) + (lions * legs_per_lion) + (tarantulas * legs_per_tarantula) + (lizards * legs_per_lizard)) :
  lizards = 5 := 
by
  sorry

end NUMINAMATH_GPT_borgnine_lizards_l1546_154692


namespace NUMINAMATH_GPT_f_0_eq_1_f_neg_1_ne_1_f_increasing_min_f_neg3_3_l1546_154602

open Real

noncomputable def f : ℝ → ℝ :=
sorry

axiom func_prop : ∀ x y : ℝ, f (x + y) = f x + f y - 1
axiom pos_x_gt_1 : ∀ x : ℝ, x > 0 → f x > 1
axiom f_1 : f 1 = 2

-- Prove that f(0) = 1
theorem f_0_eq_1 : f 0 = 1 :=
sorry

-- Prove that f(-1) ≠ 1 (and direct derivation showing f(-1) = 0)
theorem f_neg_1_ne_1 : f (-1) ≠ 1 ∧ f (-1) = 0 :=
sorry

-- Prove that f(x) is increasing
theorem f_increasing : ∀ x₁ x₂ : ℝ, x₁ < x₂ → f x₂ > f x₁ :=
sorry

-- Prove minimum value of f on [-3, 3] is -2
theorem min_f_neg3_3 : ∀ x : ℝ, -3 ≤ x ∧ x ≤ 3 → f x ≥ -2 :=
sorry

end NUMINAMATH_GPT_f_0_eq_1_f_neg_1_ne_1_f_increasing_min_f_neg3_3_l1546_154602


namespace NUMINAMATH_GPT_net_wealth_after_transactions_l1546_154654

-- Define initial values and transactions
def initial_cash_A : ℕ := 15000
def initial_cash_B : ℕ := 20000
def initial_house_value : ℕ := 15000
def first_transaction_price : ℕ := 20000
def depreciation_rate : ℝ := 0.15

-- Post-depreciation house value
def depreciated_house_value : ℝ := initial_house_value * (1 - depreciation_rate)

-- Final amounts after transactions
def final_cash_A : ℝ := (initial_cash_A + first_transaction_price) - depreciated_house_value
def final_cash_B : ℝ := depreciated_house_value

-- Net changes in wealth
def net_change_wealth_A : ℝ := final_cash_A + depreciated_house_value - (initial_cash_A + initial_house_value)
def net_change_wealth_B : ℝ := final_cash_B - initial_cash_B

-- Our proof goal
theorem net_wealth_after_transactions :
  net_change_wealth_A = 5000 ∧ net_change_wealth_B = -7250 :=
by
  sorry

end NUMINAMATH_GPT_net_wealth_after_transactions_l1546_154654


namespace NUMINAMATH_GPT_floor_plus_x_eq_17_over_4_l1546_154606

theorem floor_plus_x_eq_17_over_4
  (x : ℚ)
  (h : ⌊x⌋ + x = 17 / 4)
  : x = 9 / 4 :=
sorry

end NUMINAMATH_GPT_floor_plus_x_eq_17_over_4_l1546_154606


namespace NUMINAMATH_GPT_inequality_always_holds_l1546_154666

noncomputable def range_for_inequality (k : ℝ) : Prop :=
  0 < k ∧ k ≤ 2 * Real.sqrt (2 + Real.sqrt 5)

theorem inequality_always_holds (x y k : ℝ) (h₁ : 0 < x) (h₂ : 0 < y) (h₃ : x + y = k) :
  (x + 1/x) * (y + 1/y) ≥ (k/2 + 2/k)^2 ↔ range_for_inequality k :=
sorry

end NUMINAMATH_GPT_inequality_always_holds_l1546_154666


namespace NUMINAMATH_GPT_inequality_solution_l1546_154663

theorem inequality_solution (x : ℝ) (h : (x + 1) / 2 ≥ x / 3) : x ≥ -3 :=
by
  sorry

end NUMINAMATH_GPT_inequality_solution_l1546_154663


namespace NUMINAMATH_GPT_line_intersects_iff_sufficient_l1546_154691

noncomputable def sufficient_condition (b : ℝ) : Prop :=
b > 1

noncomputable def condition (b : ℝ) : Prop :=
b > 0

noncomputable def line_intersects_hyperbola (b : ℝ) : Prop :=
b > 2 / 3

theorem line_intersects_iff_sufficient (b : ℝ) (h : condition b) : 
  (sufficient_condition b) → (line_intersects_hyperbola b) ∧ ¬(line_intersects_hyperbola b) → (sufficient_condition b) :=
by {
  sorry
}

end NUMINAMATH_GPT_line_intersects_iff_sufficient_l1546_154691


namespace NUMINAMATH_GPT_no_nat_n_for_9_pow_n_minus_7_is_product_l1546_154617

theorem no_nat_n_for_9_pow_n_minus_7_is_product :
  ¬ ∃ (n k : ℕ), 9 ^ n - 7 = k * (k + 1) :=
by
  sorry

end NUMINAMATH_GPT_no_nat_n_for_9_pow_n_minus_7_is_product_l1546_154617


namespace NUMINAMATH_GPT_triangle_area_eq_l1546_154680

theorem triangle_area_eq :
  let y_intercept1 := 3
  let y_intercept2 := 15 / 2
  let base := y_intercept2 - y_intercept1
  let inter_x := 1
  let height := inter_x
  let area := (1 / 2) * base * height
  area = 9 / 4 :=
by
  let y_intercept1 := 3
  let y_intercept2 := 15 / 2
  let base := y_intercept2 - y_intercept1
  let inter_x := 1
  let height := inter_x
  let area := (1 / 2) * base * height
  sorry

end NUMINAMATH_GPT_triangle_area_eq_l1546_154680


namespace NUMINAMATH_GPT_school_spent_440_l1546_154626

-- Definition based on conditions listed in part a)
def cost_of_pencils (cartons_pencils : ℕ) (boxes_per_carton_pencils : ℕ) (cost_per_box_pencils : ℕ) : ℕ := 
  cartons_pencils * boxes_per_carton_pencils * cost_per_box_pencils

def cost_of_markers (cartons_markers : ℕ) (cost_per_carton_markers : ℕ) : ℕ := 
  cartons_markers * cost_per_carton_markers

noncomputable def total_cost (cartons_pencils cartons_markers boxes_per_carton_pencils cost_per_box_pencils cost_per_carton_markers : ℕ) : ℕ := 
  cost_of_pencils cartons_pencils boxes_per_carton_pencils cost_per_box_pencils + 
  cost_of_markers cartons_markers cost_per_carton_markers

-- Theorem statement to prove the total cost is $440 given the conditions
theorem school_spent_440 : total_cost 20 10 10 2 4 = 440 := by 
  sorry

end NUMINAMATH_GPT_school_spent_440_l1546_154626


namespace NUMINAMATH_GPT_part1_part2_l1546_154678

-- Definition of the function
def f (x : ℝ) (m : ℝ) : ℝ := x^2 + m * x - 1

-- Theorem for part (1)
theorem part1 
  (m n : ℝ)
  (h1 : ∀ x : ℝ, f x m < 0 ↔ -2 < x ∧ x < n) : 
  m = 3 / 2 ∧ n = 1 / 2 :=
sorry

-- Theorem for part (2)
theorem part2 
  (m : ℝ)
  (h2 : ∀ x : ℝ, m ≤ x ∧ x ≤ m + 1 → f x m < 0) : 
  -Real.sqrt 2 / 2 < m ∧ m < 0 :=
sorry

end NUMINAMATH_GPT_part1_part2_l1546_154678


namespace NUMINAMATH_GPT_repeating_decimal_to_fraction_l1546_154622

theorem repeating_decimal_to_fraction : ∀ (x : ℝ), x = 0.7 + 0.08 / (1-0.1) → x = 71 / 90 :=
by
  intros x hx
  sorry

end NUMINAMATH_GPT_repeating_decimal_to_fraction_l1546_154622


namespace NUMINAMATH_GPT_number_of_sandwiches_l1546_154624

-- Defining the conditions
def kinds_of_meat := 12
def kinds_of_cheese := 11
def kinds_of_bread := 5

-- Combinations calculation
def choose_one (n : Nat) := n
def choose_three (n : Nat) := Nat.choose n 3

-- Proof statement to show that the total number of sandwiches is 9900
theorem number_of_sandwiches : (choose_one kinds_of_meat) * (choose_three kinds_of_cheese) * (choose_one kinds_of_bread) = 9900 := by
  sorry

end NUMINAMATH_GPT_number_of_sandwiches_l1546_154624


namespace NUMINAMATH_GPT_new_person_weight_l1546_154644

theorem new_person_weight
  (avg_increase : ℝ) (original_person_weight : ℝ) (num_people : ℝ) (new_weight : ℝ)
  (h1 : avg_increase = 2.5)
  (h2 : original_person_weight = 85)
  (h3 : num_people = 8)
  (h4 : num_people * avg_increase = new_weight - original_person_weight):
    new_weight = 105 :=
by
  sorry

end NUMINAMATH_GPT_new_person_weight_l1546_154644


namespace NUMINAMATH_GPT_sugar_amount_l1546_154673

theorem sugar_amount (S F B : ℕ) (h1 : S = 5 * F / 4) (h2 : F = 10 * B) (h3 : F = 8 * (B + 60)) : S = 3000 := by
  sorry

end NUMINAMATH_GPT_sugar_amount_l1546_154673
