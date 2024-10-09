import Mathlib

namespace slope_angle_of_line_l158_15847

theorem slope_angle_of_line (m n : ℝ) (h1 : m ≠ 0) (h2 : n ≠ 0) (h3 : (m^2 + n^2) / m^2 = 4) :
  ∃ θ : ℝ, θ = π / 6 ∨ θ = 5 * π / 6 :=
by
  sorry

end slope_angle_of_line_l158_15847


namespace paul_total_vertical_distance_l158_15855

def total_vertical_distance
  (n_stories : ℕ)
  (trips_per_day : ℕ)
  (days_in_week : ℕ)
  (height_per_story : ℕ)
  : ℕ :=
  let trips_per_week := trips_per_day * days_in_week
  let distance_per_trip := n_stories * height_per_story
  trips_per_week * distance_per_trip

theorem paul_total_vertical_distance :
  total_vertical_distance 5 6 7 10 = 2100 :=
by
  -- Proof is omitted.
  sorry

end paul_total_vertical_distance_l158_15855


namespace bracelets_count_l158_15839

-- Define the conditions
def stones_total : Nat := 36
def stones_per_bracelet : Nat := 12

-- Define the theorem statement
theorem bracelets_count : stones_total / stones_per_bracelet = 3 := by
  sorry

end bracelets_count_l158_15839


namespace sum_prime_numbers_l158_15865

theorem sum_prime_numbers (a b c : ℕ) (h1 : Nat.Prime a) (h2 : Nat.Prime b) (h3 : Nat.Prime c) (hEqn : a * b * c + a = 851) : 
  a + b + c = 50 :=
sorry

end sum_prime_numbers_l158_15865


namespace degree_to_radian_conversion_l158_15817

theorem degree_to_radian_conversion : (1440 * (Real.pi / 180) = 8 * Real.pi) := 
by
  sorry

end degree_to_radian_conversion_l158_15817


namespace most_frequent_data_is_mode_l158_15821

-- Define the options
inductive Options where
  | Mean
  | Mode
  | Median
  | Frequency

-- Define the problem statement
def mostFrequentDataTerm (freqMost : String) : Options :=
  if freqMost == "Mode" then 
    Options.Mode
  else if freqMost == "Mean" then 
    Options.Mean
  else if freqMost == "Median" then 
    Options.Median
  else 
    Options.Frequency

-- Statement of the problem as a theorem
theorem most_frequent_data_is_mode (freqMost : String) :
  mostFrequentDataTerm freqMost = Options.Mode :=
by
  sorry

end most_frequent_data_is_mode_l158_15821


namespace correct_answer_statement_l158_15812

theorem correct_answer_statement
  (A := "In order to understand the situation of extracurricular reading among middle school students in China, a comprehensive survey should be conducted.")
  (B := "The median and mode of a set of data 1, 2, 5, 5, 5, 3, 3 are both 5.")
  (C := "When flipping a coin 200 times, there will definitely be 100 times when it lands 'heads up.'")
  (D := "If the variance of data set A is 0.03 and the variance of data set B is 0.1, then data set A is more stable than data set B.")
  (correct_answer := "D") : 
  correct_answer = "D" :=
  by sorry

end correct_answer_statement_l158_15812


namespace reaches_school_early_l158_15803

theorem reaches_school_early (R : ℝ) (T : ℝ) (F : ℝ) (T' : ℝ)
    (h₁ : F = (6/5) * R)
    (h₂ : T = 24)
    (h₃ : R * T = F * T')
    : T - T' = 4 := by
  -- All the given conditions are set; fill in the below placeholder with the proof.
  sorry

end reaches_school_early_l158_15803


namespace cricket_team_initial_games_l158_15863

theorem cricket_team_initial_games
  (initial_games : ℕ)
  (won_30_percent_initially : ℕ)
  (additional_wins : ℕ)
  (final_win_rate : ℚ) :
  won_30_percent_initially = initial_games * 30 / 100 →
  final_win_rate = (won_30_percent_initially + additional_wins) / (initial_games + additional_wins) →
  additional_wins = 55 →
  final_win_rate = 52 / 100 →
  initial_games = 120 := by sorry

end cricket_team_initial_games_l158_15863


namespace dwarf_heights_l158_15853

-- Define the heights of the dwarfs.
variables (F J M : ℕ)

-- Given conditions
def condition1 : Prop := J + F = M
def condition2 : Prop := M + F = J + 34
def condition3 : Prop := M + J = F + 72

-- Proof statement
theorem dwarf_heights
  (h1 : condition1 F J M)
  (h2 : condition2 F J M)
  (h3 : condition3 F J M) :
  F = 17 ∧ J = 36 ∧ M = 53 :=
by
  sorry

end dwarf_heights_l158_15853


namespace bounded_g_of_f_l158_15882

theorem bounded_g_of_f
  (f g : ℝ → ℝ)
  (h1 : ∀ x y, f (x + y) + f (x - y) = 2 * f x * g y)
  (h2 : ∃ x, f x ≠ 0)
  (h3 : ∀ x, |f x| ≤ 1) :
  ∀ y, |g y| ≤ 1 := 
sorry

end bounded_g_of_f_l158_15882


namespace sufficient_but_not_necessary_condition_l158_15826

-- Step d: Lean 4 statement
theorem sufficient_but_not_necessary_condition 
  (m n : ℕ) (e : ℚ) (h₁ : m = 5) (h₂ : n = 4) (h₃ : e = 3 / 5)
  (ellipse_eq : ∀ x y : ℝ, x^2 / m^2 + y^2 / n^2 = 1) :
  (m = 5 ∧ n = 4) → (e = 3 / 5) ∧ (¬(e = 3 / 5 → m = 5 ∧ n = 4)) :=
by
  sorry

end sufficient_but_not_necessary_condition_l158_15826


namespace initial_number_2008_l158_15852

theorem initial_number_2008 (x : ℕ) (h : x = 2008 ∨ (∃ y: ℕ, (x = 2*y + 1 ∨ (x = y / (y + 2))))): x = 2008 :=
by
  cases h with
  | inl h2008 => exact h2008
  | inr hexists => cases hexists with
    | intro y hy =>
        cases hy
        case inl h2y => sorry
        case inr hdiv => sorry

end initial_number_2008_l158_15852


namespace jake_later_than_austin_by_20_seconds_l158_15818

theorem jake_later_than_austin_by_20_seconds :
  (9 * 30) / 3 - 60 = 20 :=
by
  sorry

end jake_later_than_austin_by_20_seconds_l158_15818


namespace find_product_of_constants_l158_15893

theorem find_product_of_constants
  (M1 M2 : ℝ)
  (h : ∀ x : ℝ, (x - 1) * (x - 2) ≠ 0 → (45 * x - 31) / (x * x - 3 * x + 2) = M1 / (x - 1) + M2 / (x - 2)) :
  M1 * M2 = -826 :=
sorry

end find_product_of_constants_l158_15893


namespace time_for_one_paragraph_l158_15892

-- Definitions for the given conditions
def short_answer_time := 3 -- minutes
def essay_time := 60 -- minutes
def total_homework_time := 240 -- minutes
def essays_assigned := 2
def paragraphs_assigned := 5
def short_answers_assigned := 15

-- Function to calculate total time from given conditions
def total_time_for_essays (essays : ℕ) : ℕ :=
  essays * essay_time

def total_time_for_short_answers (short_answers : ℕ) : ℕ :=
  short_answers * short_answer_time

def total_time_for_paragraphs (paragraphs : ℕ) : ℕ :=
  total_homework_time - (total_time_for_essays essays_assigned + total_time_for_short_answers short_answers_assigned)

def time_per_paragraph (paragraphs : ℕ) : ℕ :=
  total_time_for_paragraphs paragraphs / paragraphs_assigned

-- Proving the question part
theorem time_for_one_paragraph : 
  time_per_paragraph paragraphs_assigned = 15 := by
  sorry

end time_for_one_paragraph_l158_15892


namespace pencils_placed_by_Joan_l158_15805

variable (initial_pencils : ℕ)
variable (total_pencils : ℕ)

theorem pencils_placed_by_Joan 
  (h1 : initial_pencils = 33) 
  (h2 : total_pencils = 60)
  : total_pencils - initial_pencils = 27 := 
by
  sorry

end pencils_placed_by_Joan_l158_15805


namespace man_son_age_ratio_is_two_to_one_l158_15866

-- Define the present age of the son
def son_present_age := 33

-- Define the present age of the man
def man_present_age := son_present_age + 35

-- Define the son's age in two years
def son_age_in_two_years := son_present_age + 2

-- Define the man's age in two years
def man_age_in_two_years := man_present_age + 2

-- Define the expected ratio of the man's age to son's age in two years
def ratio := man_age_in_two_years / son_age_in_two_years

-- Theorem statement verifying the ratio
theorem man_son_age_ratio_is_two_to_one : ratio = 2 := by
  -- Note: Proof not required, so we use sorry to denote the missing proof
  sorry

end man_son_age_ratio_is_two_to_one_l158_15866


namespace Markus_bags_count_l158_15809

-- Definitions of the conditions
def Mara_bags : ℕ := 12
def Mara_marbles_per_bag : ℕ := 2
def Markus_marbles_per_bag : ℕ := 13
def marbles_difference : ℕ := 2

-- Derived conditions
def Mara_total_marbles : ℕ := Mara_bags * Mara_marbles_per_bag
def Markus_total_marbles : ℕ := Mara_total_marbles + marbles_difference

-- Statement to prove
theorem Markus_bags_count : Markus_total_marbles / Markus_marbles_per_bag = 2 :=
by
  -- Skip the proof, leaving it as a task for the prover
  sorry

end Markus_bags_count_l158_15809


namespace cos_double_angle_of_parallel_vectors_l158_15857

theorem cos_double_angle_of_parallel_vectors (α : ℝ) 
  (h_parallel : (1 / 3, Real.tan α) = (Real.cos α, 1)) : 
  Real.cos (2 * α) = 7 / 9 :=
by
  sorry

end cos_double_angle_of_parallel_vectors_l158_15857


namespace problem_statement_l158_15886

open Real

theorem problem_statement (a b c A B C : ℝ) (h1 : a ≠ 0) (h2 : A ≠ 0)
    (h3 : ∀ x : ℝ, |a * x^2 + b * x + c| ≤ |A * x^2 + B * x + C|) : 
    |b^2 - 4 * a * c| ≤ |B^2 - 4 * A * C| := sorry

end problem_statement_l158_15886


namespace equally_spaced_markings_number_line_l158_15876

theorem equally_spaced_markings_number_line 
  (steps : ℕ) (distance : ℝ) (z_steps : ℕ) (z : ℝ)
  (h1 : steps = 4)
  (h2 : distance = 16)
  (h3 : z_steps = 2) :
  z = (distance / steps) * z_steps :=
by
  sorry

end equally_spaced_markings_number_line_l158_15876


namespace variation_of_powers_l158_15835

theorem variation_of_powers (x y z : ℝ) (k j : ℝ) (h1 : x = k * y^2) (h2 : y = j * z^(1/3)) :
  ∃ m : ℝ, x = m * z^(2/3) :=
by
  sorry

end variation_of_powers_l158_15835


namespace sugar_in_first_combination_l158_15842

def cost_per_pound : ℝ := 0.45
def cost_combination_1 (S : ℝ) : ℝ := cost_per_pound * S + cost_per_pound * 16
def cost_combination_2 : ℝ := cost_per_pound * 30 + cost_per_pound * 25
def total_weight_combination_2 : ℕ := 30 + 25
def total_weight_combination_1 (S : ℕ) : ℕ := S + 16

theorem sugar_in_first_combination :
  ∀ (S : ℕ), cost_combination_1 S = 26 ∧ cost_combination_2 = 26 → total_weight_combination_1 S = total_weight_combination_2 → S = 39 :=
by sorry

end sugar_in_first_combination_l158_15842


namespace mul_101_eq_10201_l158_15824

theorem mul_101_eq_10201 : 101 * 101 = 10201 := by
  sorry

end mul_101_eq_10201_l158_15824


namespace sandy_has_four_times_more_marbles_l158_15834

-- Definitions based on conditions
def jessica_red_marbles : ℕ := 3 * 12
def sandy_red_marbles : ℕ := 144

-- The theorem to prove
theorem sandy_has_four_times_more_marbles : sandy_red_marbles = 4 * jessica_red_marbles :=
by
  sorry

end sandy_has_four_times_more_marbles_l158_15834


namespace bubble_bath_amount_l158_15894

noncomputable def total_bubble_bath_needed 
  (couple_rooms : ℕ) (single_rooms : ℕ) (people_per_couple_room : ℕ) (people_per_single_room : ℕ) (ml_per_bath : ℕ) : ℕ :=
  couple_rooms * people_per_couple_room * ml_per_bath + single_rooms * people_per_single_room * ml_per_bath

theorem bubble_bath_amount :
  total_bubble_bath_needed 13 14 2 1 10 = 400 := by 
  sorry

end bubble_bath_amount_l158_15894


namespace total_distance_l158_15849

theorem total_distance (D : ℝ) (h_walk : ∀ d t, d = 4 * t) 
                       (h_run : ∀ d t, d = 8 * t) 
                       (h_time : ∀ t_walk t_run, t_walk + t_run = 0.75) 
                       (h_half : D / 2 = d_walk ∧ D / 2 = d_run) :
                       D = 8 := 
by
  sorry

end total_distance_l158_15849


namespace age_of_replaced_person_l158_15845

theorem age_of_replaced_person
    (T : ℕ) -- total age of the original group of 10 persons
    (age_person_replaced : ℕ) -- age of the person who was replaced
    (age_new_person : ℕ) -- age of the new person
    (h1 : age_new_person = 15)
    (h2 : (T / 10) - 3 = (T - age_person_replaced + age_new_person) / 10) :
    age_person_replaced = 45 :=
by
  sorry

end age_of_replaced_person_l158_15845


namespace probability_XOX_OXO_l158_15811

open Nat

/-- Setting up the math problem to be proved -/
def X : Finset ℕ := {1, 2, 3, 4}
def O : Finset ℕ := {5, 6, 7}

def totalArrangements : ℕ := choose 7 4

def favorableArrangements : ℕ := 1

theorem probability_XOX_OXO : (favorableArrangements : ℚ) / (totalArrangements : ℚ) = 1 / 35 := by
  have h_total : totalArrangements = 35 := by sorry
  have h_favorable : favorableArrangements = 1 := by sorry
  rw [h_total, h_favorable]
  norm_num

end probability_XOX_OXO_l158_15811


namespace arithmetic_sequence_sum_l158_15822

variable {a : ℕ → ℝ}
variable {S : ℕ → ℝ}
variable {d : ℝ}
variable {n : ℕ}

-- Conditions of the problem
def is_arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) := ∀ n : ℕ, a (n + 1) = a n + d
def sum_first_n_terms (a : ℕ → ℝ) (S : ℕ → ℝ) := ∀ n : ℕ, S n = n * (a 1 + a n) / 2
def S9_is_90 (S : ℕ → ℝ) := S 9 = 90

-- The proof goal
theorem arithmetic_sequence_sum (a : ℕ → ℝ) (S : ℕ → ℝ) (d : ℝ)
  (h1 : is_arithmetic_sequence a d)
  (h2 : sum_first_n_terms a S)
  (h3 : S9_is_90 S) :
  a 3 + a 5 + a 7 = 30 :=
by
  sorry

end arithmetic_sequence_sum_l158_15822


namespace wire_length_unique_l158_15860

noncomputable def distance_increment := (5 / 3)

theorem wire_length_unique (d L : ℝ) 
  (h1 : L = 25 * d) 
  (h2 : L = 24 * (d + distance_increment)) :
  L = 1000 := by
  sorry

end wire_length_unique_l158_15860


namespace percentage_increase_in_price_l158_15814

theorem percentage_increase_in_price (initial_price : ℝ) (total_cost : ℝ) (num_family_members : ℕ) 
  (pounds_per_person : ℝ) (new_price : ℝ) (percentage_increase : ℝ) :
  initial_price = 1.6 → 
  total_cost = 16 → 
  num_family_members = 4 → 
  pounds_per_person = 2 → 
  (total_cost / (num_family_members * pounds_per_person)) = new_price → 
  percentage_increase = ((new_price - initial_price) / initial_price) * 100 → 
  percentage_increase = 25 :=
by
  intros h_initial h_total h_members h_pounds h_new_price h_percentage
  sorry

end percentage_increase_in_price_l158_15814


namespace smallest_possible_value_other_integer_l158_15854

theorem smallest_possible_value_other_integer (x : ℕ) (n : ℕ) (h_pos : x > 0)
  (h_gcd : ∃ m, Nat.gcd m n = x + 3 ∧ m = 30) 
  (h_lcm : Nat.lcm 30 n = x * (x + 3)) :
  n = 162 := 
by sorry

end smallest_possible_value_other_integer_l158_15854


namespace segments_count_bound_l158_15861

-- Define the overall setup of the problem
variable (n : ℕ) (points : Finset ℕ)

-- The main hypothesis and goal
theorem segments_count_bound (hn : n ≥ 2) (hpoints : points.card = 3 * n) :
  ∃ A B : Finset (ℕ × ℕ), (∀ (i j : ℕ), i ∈ points → j ∈ points → i ≠ j → ((i, j) ∈ A ↔ (i, j) ∉ B)) ∧
  ∀ (X : Finset ℕ) (hX : X.card = n), ∃ C : Finset (ℕ × ℕ), (C ⊆ A) ∧ (X ⊆ points) ∧
  (∃ count : ℕ, count ≥ (n - 1) / 6 ∧ count = C.card ∧ ∀ (a b : ℕ), (a, b) ∈ C → a ∈ X ∧ b ∈ points \ X) := sorry

end segments_count_bound_l158_15861


namespace claire_needs_80_tiles_l158_15836

def room_length : ℕ := 14
def room_width : ℕ := 18
def border_width : ℕ := 2
def small_tile_side : ℕ := 1
def large_tile_side : ℕ := 3

def num_small_tiles : ℕ :=
  let perimeter_length := (2 * (room_width - 2 * border_width))
  let perimeter_width := (2 * (room_length - 2 * border_width))
  let corner_tiles := (2 * border_width) * 4
  perimeter_length + perimeter_width + corner_tiles

def num_large_tiles : ℕ :=
  let inner_length := room_length - 2 * border_width
  let inner_width := room_width - 2 * border_width
  let inner_area := inner_length * inner_width
  Nat.ceil (inner_area / (large_tile_side * large_tile_side))

theorem claire_needs_80_tiles : num_small_tiles + num_large_tiles = 80 :=
by sorry

end claire_needs_80_tiles_l158_15836


namespace arithmetic_sequence_15th_term_l158_15873

theorem arithmetic_sequence_15th_term :
  let first_term := 3
  let second_term := 8
  let third_term := 13
  let common_difference := second_term - first_term
  (first_term + (15 - 1) * common_difference) = 73 :=
by
  sorry

end arithmetic_sequence_15th_term_l158_15873


namespace no_positive_integer_satisfies_conditions_l158_15874

theorem no_positive_integer_satisfies_conditions :
  ¬∃ (n : ℕ), n > 1 ∧ (∃ (p1 : ℕ), Prime p1 ∧ n = p1^2) ∧ (∃ (p2 : ℕ), Prime p2 ∧ 3 * n + 16 = p2^2) :=
by
  sorry

end no_positive_integer_satisfies_conditions_l158_15874


namespace investment_time_period_l158_15889

variable (P : ℝ) (r15 r12 : ℝ) (T : ℝ)
variable (hP : P = 15000)
variable (hr15 : r15 = 0.15)
variable (hr12 : r12 = 0.12)
variable (diff : 2250 * T - 1800 * T = 900)

theorem investment_time_period :
  T = 2 := by
  sorry

end investment_time_period_l158_15889


namespace savings_per_month_l158_15851

noncomputable def annual_salary : ℝ := 48000
noncomputable def monthly_payments : ℝ := 12
noncomputable def savings_percentage : ℝ := 0.10

theorem savings_per_month :
  (annual_salary / monthly_payments) * savings_percentage = 400 :=
by
  sorry

end savings_per_month_l158_15851


namespace solve_pair_l158_15846

theorem solve_pair (x y : ℕ) (h₁ : x = 12785 ∧ y = 12768 ∨ x = 11888 ∧ y = 11893 ∨ x = 12784 ∧ y = 12770 ∨ x = 1947 ∧ y = 1945) :
  1983 = 1982 * 11888 - 1981 * 11893 :=
by {
  sorry
}

end solve_pair_l158_15846


namespace blue_pens_removed_l158_15828

def initial_blue_pens := 9
def initial_black_pens := 21
def initial_red_pens := 6
def removed_black_pens := 7
def pens_left := 25

theorem blue_pens_removed (x : ℕ) :
  initial_blue_pens - x + (initial_black_pens - removed_black_pens) + initial_red_pens = pens_left ↔ x = 4 := 
by 
  sorry

end blue_pens_removed_l158_15828


namespace largest_n_for_factoring_polynomial_l158_15899

theorem largest_n_for_factoring_polynomial :
  ∃ A B : ℤ, A * B = 120 ∧ (∀ n, (5 * 120 + 1 ≤ n → n ≤ 601)) := sorry

end largest_n_for_factoring_polynomial_l158_15899


namespace min_value_of_function_l158_15813

noncomputable def f (a x : ℝ) : ℝ := (a^x - a)^2 + (a^(-x) - a)^2

theorem min_value_of_function (a : ℝ) (h : a > 0) : ∃ x : ℝ, f a x = 2 :=
by
  sorry

end min_value_of_function_l158_15813


namespace simplify_and_evaluate_expression_l158_15895

-- Define the parameters for m and n.
def m : ℚ := -1 / 3
def n : ℚ := 1 / 2

-- Define the expression to simplify and evaluate.
def complex_expr (m n : ℚ) : ℚ :=
  -2 * (m * n - 3 * m^2) + 3 * (2 * m * n - 5 * m^2)

-- State the theorem that proves the expression equals -5/3.
theorem simplify_and_evaluate_expression :
  complex_expr m n = -5 / 3 :=
by
  sorry

end simplify_and_evaluate_expression_l158_15895


namespace find_a11_l158_15868

noncomputable def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem find_a11 (a : ℕ → ℝ)
  (h1 : geometric_sequence a)
  (h2 : a 1 * a 4 = 20)
  (h3 : a 0 + a 5 = 9) :
  a 10 = 25 / 4 :=
sorry

end find_a11_l158_15868


namespace new_car_distance_in_same_time_l158_15887

-- Define the given conditions and the distances
variable (older_car_distance : ℝ := 150)
variable (new_car_speed_factor : ℝ := 1.30)  -- Since the new car is 30% faster, its speed factor is 1.30
variable (time : ℝ)

-- Define the older car's distance as a function of time and speed
def older_car_distance_covered (t : ℝ) (distance : ℝ) : ℝ := distance

-- Define the new car's distance as a function of time and speed factor
def new_car_distance_covered (t : ℝ) (distance : ℝ) (speed_factor : ℝ) : ℝ := speed_factor * distance

theorem new_car_distance_in_same_time
  (older_car_distance : ℝ)
  (new_car_speed_factor : ℝ)
  (time : ℝ)
  (h1 : older_car_distance = 150)
  (h2 : new_car_speed_factor = 1.30) :
  new_car_distance_covered time older_car_distance new_car_speed_factor = 195 := by
  sorry

end new_car_distance_in_same_time_l158_15887


namespace compare_abc_l158_15840

noncomputable def a : ℝ := Real.exp 0.25
noncomputable def b : ℝ := 1
noncomputable def c : ℝ := -4 * Real.log 0.75

theorem compare_abc : b < c ∧ c < a := by
  -- Additional proof steps would follow here
  sorry

end compare_abc_l158_15840


namespace number_of_possible_routes_l158_15819

def f (x y : ℕ) : ℕ :=
  if y = 2 then sorry else sorry -- Here you need the exact definition of f(x, y)

theorem number_of_possible_routes (n : ℕ) (h : n > 0) : 
  f n 2 = (1 / 2 : ℚ) * (n^2 + 3 * n + 2) := 
by 
  sorry

end number_of_possible_routes_l158_15819


namespace number_of_real_solutions_l158_15891

noncomputable def system_of_equations_solutions_count (x : ℝ) : Prop :=
  3 * x^2 - 45 * (⌊x⌋:ℝ) + 60 = 0 ∧ 2 * x - 3 * (⌊x⌋:ℝ) + 1 = 0

theorem number_of_real_solutions : ∃ (x₁ x₂ x₃ : ℝ), system_of_equations_solutions_count x₁ ∧ system_of_equations_solutions_count x₂ ∧ system_of_equations_solutions_count x₃ ∧ x₁ ≠ x₂ ∧ x₁ ≠ x₃ ∧ x₂ ≠ x₃ :=
sorry

end number_of_real_solutions_l158_15891


namespace problem1_l158_15856

theorem problem1 : 3 * 403 + 5 * 403 + 2 * 403 + 401 = 4431 := by 
  sorry

end problem1_l158_15856


namespace find_value_l158_15871

variable (x y : ℝ)

def conditions (x y : ℝ) :=
  y > 2 * x ∧ 2 * x > 0 ∧ (x / y + y / x = 8)

theorem find_value (h : conditions x y) : (x + y) / (x - y) = -Real.sqrt (5 / 3) :=
sorry

end find_value_l158_15871


namespace fraction_representation_of_3_36_l158_15808

theorem fraction_representation_of_3_36 : (336 : ℚ) / 100 = 84 / 25 := 
by sorry

end fraction_representation_of_3_36_l158_15808


namespace simplify_expression_l158_15880

def expression1 (x : ℝ) : ℝ :=
  3 * x^3 + 4 * x^2 + 2 * x + 5 - (2 * x^3 - 5 * x^2 + x - 3) + (x^3 - 2 * x^2 - 4 * x + 6)

def expression2 (x : ℝ) : ℝ :=
  2 * x^3 + 7 * x^2 - 3 * x + 14

theorem simplify_expression (x : ℝ) : expression1 x = expression2 x :=
by 
  sorry

end simplify_expression_l158_15880


namespace range_of_a_l158_15896

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
if x ≤ 0 then (x - a)^2 else x + (1 / x) + a

theorem range_of_a (a : ℝ) (x : ℝ) (h : ∀ x : ℝ, f a x ≥ f a 0) : 0 ≤ a ∧ a ≤ 2 :=
sorry

end range_of_a_l158_15896


namespace distance_between_planes_l158_15883

def plane1 (x y z : ℝ) := 3 * x - y + z - 3 = 0
def plane2 (x y z : ℝ) := 6 * x - 2 * y + 2 * z + 4 = 0

theorem distance_between_planes :
  ∃ d : ℝ, d = (5 * Real.sqrt 11) / 11 ∧ 
            ∀ x y z : ℝ, plane1 x y z → plane2 x y z → d = (5 * Real.sqrt 11) / 11 :=
sorry

end distance_between_planes_l158_15883


namespace decreasing_function_l158_15807

def f (a x : ℝ) : ℝ := a * x^3 - x

theorem decreasing_function (a : ℝ) 
  (h : ∀ x y : ℝ, x < y → f a y ≤ f a x) : a ≤ 0 :=
by
  sorry

end decreasing_function_l158_15807


namespace banker_l158_15850

-- Define the given conditions
def present_worth : ℝ := 400
def interest_rate : ℝ := 0.10
def time_period : ℕ := 3

-- Define the amount due in the future
def amount_due (PW : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  PW * (1 + r) ^ n

-- Define the banker's gain
def bankers_gain (A PW : ℝ) : ℝ :=
  A - PW

-- State the theorem we need to prove
theorem banker's_gain_is_correct :
  bankers_gain (amount_due present_worth interest_rate time_period) present_worth = 132.4 :=
by sorry

end banker_l158_15850


namespace find_x_l158_15837

/-!
# Problem Statement
Given that the segment with endpoints (-8, 0) and (32, 0) is the diameter of a circle,
and the point (x, 20) lies on the circle, prove that x = 12.
-/

def point_on_circle (x y : ℝ) (center_x center_y radius : ℝ) : Prop :=
  (x - center_x)^2 + (y - center_y)^2 = radius^2

theorem find_x : 
  let center_x := (32 + (-8)) / 2
  let center_y := (0 + 0) / 2
  let radius := (32 - (-8)) / 2
  ∃ x : ℝ, point_on_circle x 20 center_x center_y radius → x = 12 :=
by
  sorry

end find_x_l158_15837


namespace arithmetic_series_sum_l158_15804

theorem arithmetic_series_sum :
  let a := 2
  let d := 3
  let l := 56
  let n := 19
  let pairs_sum := (n-1) / 2 * (-3)
  let single_term := 56
  2 - 5 + 8 - 11 + 14 - 17 + 20 - 23 + 26 - 29 + 32 - 35 + 38 - 41 + 44 - 47 + 50 - 53 + 56 = 29 :=
by
  sorry

end arithmetic_series_sum_l158_15804


namespace minimum_races_to_find_top3_l158_15898

-- Define a constant to represent the number of horses and maximum horses per race
def total_horses : ℕ := 25
def max_horses_per_race : ℕ := 5

-- Define the problem statement as a theorem
theorem minimum_races_to_find_top3 (total_horses : ℕ) (max_horses_per_race : ℕ) : ℕ :=
  if total_horses = 25 ∧ max_horses_per_race = 5 then 7 else sorry

end minimum_races_to_find_top3_l158_15898


namespace min_initial_seeds_l158_15816

/-- Given conditions:
  - The farmer needs to sell at least 10,000 watermelons each year.
  - Each watermelon produces 250 seeds when used for seeds but cannot be sold if used for seeds.
  - We need to find the minimum number of initial seeds S the farmer must buy to never buy seeds again.
-/
theorem min_initial_seeds : ∃ (S : ℕ), S = 10041 ∧ ∀ (yearly_sales : ℕ), yearly_sales = 10000 →
  ∀ (seed_yield : ℕ), seed_yield = 250 →
  ∃ (x : ℕ), S = yearly_sales + x ∧ x * seed_yield ≥ S :=
sorry

end min_initial_seeds_l158_15816


namespace time_spent_on_spelling_l158_15844

-- Define the given conditions
def total_time : Nat := 60
def math_time : Nat := 15
def reading_time : Nat := 27

-- Define the question as a Lean theorem statement
theorem time_spent_on_spelling : total_time - math_time - reading_time = 18 := sorry

end time_spent_on_spelling_l158_15844


namespace lake_circumference_ratio_l158_15890

theorem lake_circumference_ratio 
    (D C : ℝ) 
    (hD : D = 100) 
    (hC : C = 314) : 
    C / D = 3.14 := 
sorry

end lake_circumference_ratio_l158_15890


namespace count_even_three_digit_numbers_less_than_800_l158_15875

def even_three_digit_numbers_less_than_800 : Nat :=
  let hundreds_choices := 7
  let tens_choices := 8
  let units_choices := 4
  hundreds_choices * tens_choices * units_choices

theorem count_even_three_digit_numbers_less_than_800 :
  even_three_digit_numbers_less_than_800 = 224 := 
by 
  unfold even_three_digit_numbers_less_than_800
  rfl

end count_even_three_digit_numbers_less_than_800_l158_15875


namespace total_necklaces_made_l158_15829

-- Definitions based on conditions
def first_machine_necklaces : ℝ := 45
def second_machine_necklaces : ℝ := 2.4 * first_machine_necklaces

-- Proof statement
theorem total_necklaces_made : (first_machine_necklaces + second_machine_necklaces) = 153 := by
  sorry

end total_necklaces_made_l158_15829


namespace jimmy_max_loss_l158_15830

-- Definition of the conditions
def exam_points : ℕ := 20
def number_of_exams : ℕ := 3
def points_lost_for_behavior : ℕ := 5
def passing_score : ℕ := 50

-- Total points Jimmy has earned and lost
def total_points : ℕ := (number_of_exams * exam_points) - points_lost_for_behavior

-- The maximum points Jimmy can lose and still pass
def max_points_jimmy_can_lose : ℕ := total_points - passing_score

-- Statement to prove
theorem jimmy_max_loss : max_points_jimmy_can_lose = 5 := 
by
  sorry

end jimmy_max_loss_l158_15830


namespace bicycles_sold_saturday_l158_15878

variable (S : ℕ)

theorem bicycles_sold_saturday :
  let net_increase_friday := 15 - 10
  let net_increase_saturday := 8 - S
  let net_increase_sunday := 11 - 9
  (net_increase_friday + net_increase_saturday + net_increase_sunday = 3) → 
  S = 12 :=
by
  intros h
  sorry

end bicycles_sold_saturday_l158_15878


namespace smallest_integer_n_l158_15832

theorem smallest_integer_n (n : ℕ) (h : Nat.lcm 60 n / Nat.gcd 60 n = 75) : n = 500 :=
sorry

end smallest_integer_n_l158_15832


namespace find_x_l158_15848

-- Definitions of the conditions
def vector_a : ℝ × ℝ := (1, 2)
def vector_b (x : ℝ) : ℝ × ℝ := (x, -2)

-- Inner product definition
def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

-- Perpendicular condition
def is_perpendicular (v1 v2 : ℝ × ℝ) : Prop :=
  dot_product v1 v2 = 0

theorem find_x (x : ℝ) (h : is_perpendicular vector_a (vector_b x)) : x = 4 :=
  sorry

end find_x_l158_15848


namespace painting_frame_ratio_l158_15825

theorem painting_frame_ratio (x l : ℝ) (h1 : x > 0) (h2 : l > 0) 
  (h3 : (2 / 3) * x * x = (x + 2 * l) * ((3 / 2) * x + 2 * l) - x * (3 / 2) * x) :
  (x + 2 * l) / ((3 / 2) * x + 2 * l) = 3 / 4 :=
by
  sorry

end painting_frame_ratio_l158_15825


namespace total_corn_yield_l158_15810

/-- 
The total corn yield in centners, harvested from a certain field area, is expressed 
as a four-digit number composed of the digits 0, 2, 3, and 5. When the average 
yield per hectare was calculated, it was found to be the same number of centners 
as the number of hectares of the field area. 
This statement proves that the total corn yield is 3025. 
-/
theorem total_corn_yield : ∃ (Y A : ℕ), (Y = A^2) ∧ (A >= 10 ∧ A < 100) ∧ 
  (Y / 1000 != 0) ∧ (Y / 1000 != 1) ∧ (Y / 10 % 10 != 4) ∧ 
  (Y % 10 != 1) ∧ (Y % 10 = 0 ∨ Y % 10 = 5) ∧ 
  (Y / 100 % 10 == 0 ∨ Y / 100 % 10 == 2 ∨ Y / 100 % 10 == 3 ∨ Y / 100 % 10 == 5) ∧ 
  Y = 3025 := 
by 
  sorry

end total_corn_yield_l158_15810


namespace math_problem_l158_15838

theorem math_problem :
  (∃ n : ℕ, 28 = 4 * n) ∧
  ((∃ n1 : ℕ, 361 = 19 * n1) ∧ ¬(∃ n2 : ℕ, 63 = 19 * n2)) ∧
  (¬((∃ n3 : ℕ, 90 = 30 * n3) ∧ ¬(∃ n4 : ℕ, 65 = 30 * n4))) ∧
  ((∃ n5 : ℕ, 45 = 15 * n5) ∧ (∃ n6 : ℕ, 30 = 15 * n6)) ∧
  (∃ n7 : ℕ, 144 = 12 * n7) :=
by {
  -- We need to prove each condition to be true and then prove the statements A, B, D, E are true.
  sorry
}

end math_problem_l158_15838


namespace choose_athlete_B_l158_15864

variable (SA2 : ℝ) (SB2 : ℝ)
variable (num_shots : ℕ) (avg_rings : ℝ)

-- Conditions
def athlete_A_variance := SA2 = 3.5
def athlete_B_variance := SB2 = 2.8
def same_number_of_shots := true -- Implicit condition, doesn't need proof
def same_average_rings := true -- Implicit condition, doesn't need proof

-- Question: prove Athlete B should be chosen
theorem choose_athlete_B 
  (hA_var : athlete_A_variance SA2)
  (hB_var : athlete_B_variance SB2)
  (same_shots : same_number_of_shots)
  (same_avg : same_average_rings) :
  "B" = "B" :=
by 
  sorry

end choose_athlete_B_l158_15864


namespace units_digit_of_k3_plus_5k_l158_15831

def k : ℕ := 2024^2 + 3^2024

theorem units_digit_of_k3_plus_5k (k := 2024^2 + 3^2024) : 
  ((k^3 + 5^k) % 10) = 8 := 
by 
  sorry

end units_digit_of_k3_plus_5k_l158_15831


namespace melanie_trout_catch_l158_15858

def trout_caught_sara : ℕ := 5
def trout_caught_melanie (sara_trout : ℕ) : ℕ := 2 * sara_trout

theorem melanie_trout_catch :
  trout_caught_melanie trout_caught_sara = 10 :=
by
  sorry

end melanie_trout_catch_l158_15858


namespace ratio_is_five_over_twelve_l158_15815

theorem ratio_is_five_over_twelve (a b c d : ℚ) (h1 : b = 4 * a) (h2 : d = 2 * c) :
    (a + b) / (c + d) = 5 / 12 :=
sorry

end ratio_is_five_over_twelve_l158_15815


namespace trig_identity_l158_15897

open Real

theorem trig_identity : sin (20 * (π / 180)) * cos (10 * (π / 180)) - cos (200 * (π / 180)) * sin (10 * (π / 180)) = 1 / 2 := 
by
  sorry

end trig_identity_l158_15897


namespace find_number_of_persons_l158_15877

-- Definitions of the given conditions
def total_amount : ℕ := 42900
def amount_per_person : ℕ := 1950

-- The statement to prove
theorem find_number_of_persons (n : ℕ) (h : total_amount = n * amount_per_person) : n = 22 :=
sorry

end find_number_of_persons_l158_15877


namespace contrapositive_l158_15802

variable {α : Type} (M : α → Prop) (a b : α)

theorem contrapositive (h : (M a → ¬ M b)) : (M b → ¬ M a) := 
by
  sorry

end contrapositive_l158_15802


namespace planes_parallel_or_coincide_l158_15879

-- Define normal vectors
def normal_vector_u : ℝ × ℝ × ℝ := (1, 2, -2)
def normal_vector_v : ℝ × ℝ × ℝ := (-3, -6, 6)

-- The theorem states that planes defined by these normal vectors are either 
-- parallel or coincide if their normal vectors are collinear.
theorem planes_parallel_or_coincide (u v : ℝ × ℝ × ℝ) 
  (h_u : u = normal_vector_u) 
  (h_v : v = normal_vector_v) 
  (h_collinear : v = (-3) • u) : 
    ∃ k : ℝ, v = k • u := 
by
  sorry

end planes_parallel_or_coincide_l158_15879


namespace line_passes_through_fixed_point_l158_15884

theorem line_passes_through_fixed_point (k : ℝ) : ∃ (x y : ℝ), x = -2 ∧ y = 1 ∧ y = k * x + 2 * k + 1 :=
by
  sorry

end line_passes_through_fixed_point_l158_15884


namespace hyperbola_eccentricity_proof_l158_15862

noncomputable def hyperbola_eccentricity (a b k1 k2 : ℝ) (ha : 0 < a) (hb : 0 < b) 
    (C_on_hyperbola : ∃ (x y : ℝ), x^2 / a^2 - y^2 / b^2 = 1) 
    (slope_condition : k1 * k2 = b^2 / a^2)
    (minimized_expr : ∀ k1 k2: ℝ , (2 / (k1 * k2)) + Real.log k1 + Real.log k2 ≥ (2 / (b^2 / a^2)) + Real.log ((b^2 / a^2))) : 
    ℝ :=
  Real.sqrt (1 + (b^2 / a^2))

theorem hyperbola_eccentricity_proof (a b k1 k2 : ℝ) (ha : 0 < a) (hb : 0 < b) 
    (C_on_hyperbola : ∃ (x y : ℝ), x^2 / a^2 - y^2 / b^2 = 1) 
    (slope_condition : k1 * k2 = b^2 / a^2)
    (minimized_expr : ∀ k1 k2: ℝ , (2 / (k1 * k2)) + Real.log k1 + Real.log k2 ≥ (2 / (b^2 / a^2)) + Real.log ((b^2 / a^2))) :    
    hyperbola_eccentricity a b k1 k2 ha hb C_on_hyperbola slope_condition minimized_expr = Real.sqrt 3 :=
sorry

end hyperbola_eccentricity_proof_l158_15862


namespace Luke_trips_l158_15881

variable (carries : Nat) (table1 : Nat) (table2 : Nat)

theorem Luke_trips (h1 : carries = 4) (h2 : table1 = 20) (h3 : table2 = 16) : 
  (table1 / carries + table2 / carries) = 9 :=
by
  sorry

end Luke_trips_l158_15881


namespace perpendicular_lines_a_eq_3_l158_15843

theorem perpendicular_lines_a_eq_3 (a : ℝ) :
  let l₁ := (a + 1) * x + 2 * y + 6
  let l₂ := x + (a - 5) * y + a^2 - 1
  (a ≠ 5 → -((a + 1) / 2) * (1 / (5 - a)) = -1) → a = 3 := by
  intro l₁ l₂ h
  sorry

end perpendicular_lines_a_eq_3_l158_15843


namespace value_of_a_minus_b_l158_15833

theorem value_of_a_minus_b (a b : ℤ) (h1 : |a| = 2) (h2 : |b| = 5) (h3 : |a - b| = a - b) : a - b = 7 ∨ a - b = 3 :=
sorry

end value_of_a_minus_b_l158_15833


namespace slope_of_line_l158_15823

theorem slope_of_line (x1 y1 x2 y2 : ℝ)
  (h1 : 4 * y1 + 6 * x1 = 0)
  (h2 : 4 * y2 + 6 * x2 = 0)
  (h1x2 : x1 ≠ x2) :
  (y2 - y1) / (x2 - x1) = -3 / 2 :=
by sorry

end slope_of_line_l158_15823


namespace factorization_analysis_l158_15800

variable (a b c : ℝ)

theorem factorization_analysis : a^2 - 2 * a * b + b^2 - c^2 = (a - b + c) * (a - b - c) := 
sorry

end factorization_analysis_l158_15800


namespace decrease_percent_revenue_l158_15841

theorem decrease_percent_revenue (T C : ℝ) (hT : T > 0) (hC : C > 0) : 
  let original_revenue := T * C
  let new_tax := 0.80 * T
  let new_consumption := 1.10 * C
  let new_revenue := new_tax * new_consumption
  let decrease_in_revenue := original_revenue - new_revenue
  let decrease_percent := (decrease_in_revenue / original_revenue) * 100
  decrease_percent = 12 := by
  sorry

end decrease_percent_revenue_l158_15841


namespace evaluate_expression_l158_15820

theorem evaluate_expression (x : ℝ) : 
  x * (x * (x * (3 - x) - 5) + 12) + 2 = -x^4 + 3 * x^3 - 5 * x^2 + 12 * x + 2 := 
by 
  sorry

end evaluate_expression_l158_15820


namespace Zixuan_amount_l158_15870

noncomputable def amounts (X Y Z : ℕ) : Prop := 
  (X + Y + Z = 50) ∧
  (X = 3 * (Y + Z) / 2) ∧
  (Y = Z + 4)

theorem Zixuan_amount : ∃ Z : ℕ, ∃ X Y : ℕ, amounts X Y Z ∧ Z = 8 :=
by
  sorry

end Zixuan_amount_l158_15870


namespace range_of_independent_variable_l158_15801

theorem range_of_independent_variable (x : ℝ) : 
  (∃ y : ℝ, y = 2 * x / (x - 1)) ↔ x ≠ 1 :=
by sorry

end range_of_independent_variable_l158_15801


namespace robin_gum_pieces_l158_15827

-- Defining the conditions
def packages : ℕ := 9
def pieces_per_package : ℕ := 15
def total_pieces : ℕ := 135

-- Theorem statement
theorem robin_gum_pieces (h1 : packages = 9) (h2 : pieces_per_package = 15) : packages * pieces_per_package = total_pieces := by
  -- According to the problem, the correct answer is 135 pieces
  have h: 9 * 15 = 135 := by norm_num
  rw [h1, h2]
  exact h

end robin_gum_pieces_l158_15827


namespace marilyn_bananas_l158_15867

-- Defining the conditions
def boxes : ℕ := 8
def bananas_per_box : ℕ := 5

-- The statement that Marilyn has 40 bananas
theorem marilyn_bananas : boxes * bananas_per_box = 40 :=
by
  sorry

end marilyn_bananas_l158_15867


namespace swimming_pool_time_l158_15806

theorem swimming_pool_time
  (A B C : ℝ)
  (h1 : A + B = 1 / 3)
  (h2 : A + C = 1 / 6)
  (h3 : B + C = 1 / 4.5) :
  1 / (A + B + C) = 2.25 :=
by
  sorry

end swimming_pool_time_l158_15806


namespace ganesh_speed_x_to_y_l158_15872

-- Define the conditions
variables (D : ℝ) (V : ℝ)

-- Theorem statement: Prove that Ganesh's average speed from x to y is 44 km/hr
theorem ganesh_speed_x_to_y
  (H1 : 39.6 = 2 * D / (D / V + D / 36))
  (H2 : V = 44) :
  true :=
sorry

end ganesh_speed_x_to_y_l158_15872


namespace bill_steps_l158_15885

theorem bill_steps (step_length : ℝ) (total_distance : ℝ) (n_steps : ℕ) 
  (h_step_length : step_length = 1 / 2) 
  (h_total_distance : total_distance = 12) 
  (h_n_steps : n_steps = total_distance / step_length) : 
  n_steps = 24 :=
by sorry

end bill_steps_l158_15885


namespace present_population_l158_15859

theorem present_population (P : ℝ) (h1 : (P : ℝ) * (1 + 0.1) ^ 2 = 14520) : P = 12000 :=
sorry

end present_population_l158_15859


namespace prize_winners_l158_15888

variable (Elaine Frank George Hannah : Prop)

axiom ElaineImpliesFrank : Elaine → Frank
axiom FrankImpliesGeorge : Frank → George
axiom GeorgeImpliesHannah : George → Hannah
axiom OnlyTwoWinners : (Elaine ∧ Frank ∧ ¬George ∧ ¬Hannah) ∨ (Elaine ∧ ¬Frank ∧ George ∧ ¬Hannah) ∨ (Elaine ∧ ¬Frank ∧ ¬George ∧ Hannah) ∨ (¬Elaine ∧ Frank ∧ George ∧ ¬Hannah) ∨ (¬Elaine ∧ Frank ∧ ¬George ∧ Hannah) ∨ (¬Elaine ∧ ¬Frank ∧ George ∧ Hannah)

theorem prize_winners : (Elaine ∧ Frank ∧ ¬George ∧ ¬Hannah) ∨ (Elaine ∧ ¬Frank ∧ George ∧ ¬Hannah) ∨ (Elaine ∧ ¬Frank ∧ ¬George ∧ Hannah) ∨ (¬Elaine ∧ Frank ∧ George ∧ ¬Hannah) ∨ (¬Elaine ∧ Frank ∧ ¬George ∧ Hannah) ∨ (¬Elaine ∧ ¬Frank ∧ George ∧ Hannah) → (¬Elaine ∧ ¬Frank ∧ George ∧ Hannah) :=
by
  sorry

end prize_winners_l158_15888


namespace sharon_trip_distance_l158_15869

theorem sharon_trip_distance
  (h1 : ∀ (d : ℝ), (180 * d) = 1 ∨ (d = 0))  -- Any distance traveled in 180 minutes follows 180d=1 (usual speed)
  (h2 : ∀ (d : ℝ), (276 * (d - 20 / 60)) = 1 ∨ (d = 0))  -- With reduction in speed due to snowstorm too follows a similar relation
  (h3: ∀ (total_time : ℝ), total_time = 276 ∨ total_time = 0)  -- Total time is 276 minutes
  : ∃ (x : ℝ), x = 135 := sorry

end sharon_trip_distance_l158_15869
