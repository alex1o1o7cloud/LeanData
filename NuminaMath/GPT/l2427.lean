import Mathlib

namespace NUMINAMATH_GPT_ratio_of_average_speed_to_still_water_speed_l2427_242718

noncomputable def speed_of_current := 6
noncomputable def speed_in_still_water := 18
noncomputable def downstream_speed := speed_in_still_water + speed_of_current
noncomputable def upstream_speed := speed_in_still_water - speed_of_current
noncomputable def distance_each_way := 1
noncomputable def total_distance := 2 * distance_each_way
noncomputable def time_downstream := (distance_each_way : ℝ) / (downstream_speed : ℝ)
noncomputable def time_upstream := (distance_each_way : ℝ) / (upstream_speed : ℝ)
noncomputable def total_time := time_downstream + time_upstream
noncomputable def average_speed := (total_distance : ℝ) / (total_time : ℝ)
noncomputable def ratio_average_speed := (average_speed : ℝ) / (speed_in_still_water : ℝ)

theorem ratio_of_average_speed_to_still_water_speed :
  ratio_average_speed = (8 : ℝ) / (9 : ℝ) :=
sorry

end NUMINAMATH_GPT_ratio_of_average_speed_to_still_water_speed_l2427_242718


namespace NUMINAMATH_GPT_monomial_sum_mn_l2427_242726

theorem monomial_sum_mn (m n : ℤ) 
  (h1 : m + 6 = 1) 
  (h2 : 2 * n + 1 = 7) : 
  m * n = -15 := by
  sorry

end NUMINAMATH_GPT_monomial_sum_mn_l2427_242726


namespace NUMINAMATH_GPT_find_function_expression_l2427_242745

theorem find_function_expression (f : ℝ → ℝ) : 
  (∀ x : ℝ, f (x - 1) = x^2 - 3 * x) → 
  (∀ x : ℝ, f x = x^2 - x - 2) :=
by
  sorry

end NUMINAMATH_GPT_find_function_expression_l2427_242745


namespace NUMINAMATH_GPT_sequence_periodic_a_n_plus_2_eq_a_n_l2427_242778

-- Definition of the sequence and conditions
noncomputable def seq (a : ℕ → ℤ) :=
  ∀ n : ℕ, ∃ α k : ℕ, a n = Int.ofNat (2^α) * k ∧ Int.gcd (Int.ofNat k) 2 = 1 ∧ a (n+1) = Int.ofNat (2^α) - k

-- Definition of periodic sequence
def periodic (a : ℕ → ℤ) (d : ℕ) :=
  ∀ n : ℕ, a (n + d) = a n

-- Proving the desired property
theorem sequence_periodic_a_n_plus_2_eq_a_n (a : ℕ → ℤ) (d : ℕ) (h_seq : seq a) (h_periodic : periodic a d) :
  ∀ n : ℕ, a (n + 2) = a n :=
sorry

end NUMINAMATH_GPT_sequence_periodic_a_n_plus_2_eq_a_n_l2427_242778


namespace NUMINAMATH_GPT_power_calculation_l2427_242798

theorem power_calculation : 8^6 * 27^6 * 8^18 * 27^18 = 216^24 := by
  sorry

end NUMINAMATH_GPT_power_calculation_l2427_242798


namespace NUMINAMATH_GPT_box_width_l2427_242757

variable (l h vc : ℕ)
variable (nc : ℕ)
variable (v : ℕ)

-- Given
def length_box := 8
def height_box := 5
def volume_cube := 10
def num_cubes := 60
def volume_box := num_cubes * volume_cube

-- To Prove
theorem box_width : (volume_box = l * h * w) → w = 15 :=
by
  intro h1
  sorry

end NUMINAMATH_GPT_box_width_l2427_242757


namespace NUMINAMATH_GPT_son_age_is_eight_l2427_242716

theorem son_age_is_eight (F S : ℕ) (h1 : F + 6 + S + 6 = 68) (h2 : F = 6 * S) : S = 8 :=
by
  sorry

end NUMINAMATH_GPT_son_age_is_eight_l2427_242716


namespace NUMINAMATH_GPT_parabola_vertex_l2427_242733

theorem parabola_vertex :
  ∃ h k : ℝ, (∀ x : ℝ, y = 1 / 2 * (x + 1) ^ 2 - 1 / 2) →
    (h = -1 ∧ k = -1 / 2) :=
by
  sorry

end NUMINAMATH_GPT_parabola_vertex_l2427_242733


namespace NUMINAMATH_GPT_lines_intersect_at_same_point_l2427_242794

theorem lines_intersect_at_same_point (m k : ℝ) :
  (∃ x y : ℝ, y = 3 * x + 5 ∧ y = -4 * x + m ∧ y = 2 * x + k) ↔ k = (m + 30) / 7 :=
by {
  sorry -- proof not required, only statement.
}

end NUMINAMATH_GPT_lines_intersect_at_same_point_l2427_242794


namespace NUMINAMATH_GPT_cats_left_correct_l2427_242750

-- Define initial conditions
def siamese_cats : ℕ := 13
def house_cats : ℕ := 5
def sold_cats : ℕ := 10

-- Define the total number of cats initially
def total_cats_initial : ℕ := siamese_cats + house_cats

-- Define the number of cats left after the sale
def cats_left : ℕ := total_cats_initial - sold_cats

-- Prove the number of cats left is 8
theorem cats_left_correct : cats_left = 8 :=
by 
  sorry

end NUMINAMATH_GPT_cats_left_correct_l2427_242750


namespace NUMINAMATH_GPT_number_of_females_l2427_242705

theorem number_of_females (total_people : ℕ) (avg_age_total : ℕ) 
  (avg_age_males : ℕ) (avg_age_females : ℕ) (females : ℕ) :
  total_people = 140 → avg_age_total = 24 →
  avg_age_males = 21 → avg_age_females = 28 → 
  females = 60 :=
by
  intros h1 h2 h3 h4
  -- Using the given conditions
  sorry

end NUMINAMATH_GPT_number_of_females_l2427_242705


namespace NUMINAMATH_GPT_monotonic_intervals_range_of_k_l2427_242792

noncomputable def f (x k : ℝ) : ℝ := (Real.exp x) / (x ^ 2) - k * (2 / x + Real.log x)
noncomputable def f' (x k : ℝ) : ℝ := (x - 2) * (Real.exp x - k * x) / (x^3)

theorem monotonic_intervals (k : ℝ) (h : k ≤ 0) :
  (∀ x : ℝ, 0 < x ∧ x < 2 → f' x k < 0) ∧ (∀ x : ℝ, x > 2 → f' x k > 0) := sorry

theorem range_of_k (k : ℝ) (h : e < k ∧ k < (e^2)/2) :
  ∃ x1 x2 : ℝ, 0 < x1 ∧ x1 < 2 ∧ 0 < x2 ∧ x2 < 2 ∧ 
    (f' x1 k = 0 ∧ f' x2 k = 0 ∧ x1 ≠ x2) := sorry

end NUMINAMATH_GPT_monotonic_intervals_range_of_k_l2427_242792


namespace NUMINAMATH_GPT_ratio_of_women_working_in_retail_l2427_242780

-- Define the population of Los Angeles
def population_LA : ℕ := 6000000

-- Define the proportion of women in Los Angeles
def half_population : ℕ := population_LA / 2

-- Define the number of women working in retail
def women_retail : ℕ := 1000000

-- Define the total number of women in Los Angeles
def total_women : ℕ := half_population

-- The statement to be proven:
theorem ratio_of_women_working_in_retail :
  (women_retail / total_women : ℚ) = 1 / 3 :=
by {
  -- The proof goes here
  sorry
}

end NUMINAMATH_GPT_ratio_of_women_working_in_retail_l2427_242780


namespace NUMINAMATH_GPT_zengshan_suanfa_tongzong_l2427_242764

-- Definitions
variables (x y : ℝ)
variables (h1 : x = y + 5) (h2 : (1 / 2) * x = y - 5)

-- Theorem
theorem zengshan_suanfa_tongzong :
  x = y + 5 ∧ (1 / 2) * x = y - 5 :=
by
  -- Starting with the given hypotheses
  exact ⟨h1, h2⟩

end NUMINAMATH_GPT_zengshan_suanfa_tongzong_l2427_242764


namespace NUMINAMATH_GPT_parabola_eqn_min_distance_l2427_242773

theorem parabola_eqn (a b : ℝ) (h_a : 0 < a)
  (h_A : -a + b = 1)
  (h_B : 4 * a + 2 * b = 0) :
  (∀ x : ℝ,  y = a * x^2 + b * x) ↔ (∀ x : ℝ, y = (1/3) * x^2 - (2/3) * x) :=
by
  sorry

theorem min_distance (a b : ℝ) (h_a : 0 < a)
  (h_A : -a + b = 1)
  (h_B : 4 * a + 2 * b = 0)
  (line_eq : ∀ x, (y : ℝ) = x - 25/4) :
  (∀ P : ℝ × ℝ, ∃ P_min : ℝ × ℝ, P_min = (5/2, 5/12)) :=
by
  sorry

end NUMINAMATH_GPT_parabola_eqn_min_distance_l2427_242773


namespace NUMINAMATH_GPT_functional_equation_holds_l2427_242788

def f (p q : ℕ) : ℝ :=
  if p = 0 ∨ q = 0 then 0 else (p * q : ℝ)

theorem functional_equation_holds (p q : ℕ) : 
  f p q = 
    if p = 0 ∨ q = 0 then 0 
    else 1 + (1 / 2) * f (p + 1) (q - 1) + (1 / 2) * f (p - 1) (q + 1) :=
  by 
    sorry

end NUMINAMATH_GPT_functional_equation_holds_l2427_242788


namespace NUMINAMATH_GPT_kangaroo_arrangement_count_l2427_242797

theorem kangaroo_arrangement_count :
  let k := 8
  let tallest_at_ends := 2
  let middle := k - tallest_at_ends
  (tallest_at_ends * (middle.factorial)) = 1440 := by
  sorry

end NUMINAMATH_GPT_kangaroo_arrangement_count_l2427_242797


namespace NUMINAMATH_GPT_percentage_employees_six_years_or_more_l2427_242706

theorem percentage_employees_six_years_or_more:
  let marks : List ℕ := [6, 6, 7, 4, 3, 3, 3, 1, 1, 1]
  let total_employees (marks : List ℕ) (y : ℕ) := marks.foldl (λ acc m => acc + m * y) 0
  let employees_six_years_or_more (marks : List ℕ) (y : ℕ) := (marks.drop 6).foldl (λ acc m => acc + m * y) 0
  (employees_six_years_or_more marks 1 / total_employees marks 1 : ℚ) * 100 = 17.14 := by
  sorry

end NUMINAMATH_GPT_percentage_employees_six_years_or_more_l2427_242706


namespace NUMINAMATH_GPT_cubics_sum_div_abc_eq_three_l2427_242776

theorem cubics_sum_div_abc_eq_three {a b c : ℝ} (h₁ : a ≠ 0) (h₂ : b ≠ 0) (h₃ : c ≠ 0) (h₄ : a + b + c = 0) :
  (a^3 + b^3 + c^3) / (a * b * c) = 3 :=
by
  sorry

end NUMINAMATH_GPT_cubics_sum_div_abc_eq_three_l2427_242776


namespace NUMINAMATH_GPT_pears_sold_in_a_day_l2427_242765

-- Define the conditions
variable (morning_pears afternoon_pears : ℕ)
variable (h1 : afternoon_pears = 2 * morning_pears)
variable (h2 : afternoon_pears = 320)

-- Lean theorem statement to prove the question answer
theorem pears_sold_in_a_day :
  (morning_pears + afternoon_pears = 480) :=
by
  -- Insert proof here
  sorry

end NUMINAMATH_GPT_pears_sold_in_a_day_l2427_242765


namespace NUMINAMATH_GPT_women_more_than_men_l2427_242790

theorem women_more_than_men 
(M W : ℕ) 
(h_ratio : (M:ℚ) / W = 5 / 9) 
(h_total : M + W = 14) :
W - M = 4 := 
by 
  sorry

end NUMINAMATH_GPT_women_more_than_men_l2427_242790


namespace NUMINAMATH_GPT_age_of_25th_student_l2427_242729

-- Definitions derived from problem conditions
def averageAgeClass (totalAge : ℕ) (totalStudents : ℕ) : ℕ := totalAge / totalStudents
def totalAgeGivenAverage (numStudents : ℕ) (averageAge : ℕ) : ℕ := numStudents * averageAge

-- Given conditions
def totalAgeOfAllStudents := 25 * 24
def totalAgeOf8Students := totalAgeGivenAverage 8 22
def totalAgeOf10Students := totalAgeGivenAverage 10 20
def totalAgeOf6Students := totalAgeGivenAverage 6 28
def totalAgeOf24Students := totalAgeOf8Students + totalAgeOf10Students + totalAgeOf6Students

-- The proof that the age of the 25th student is 56 years
theorem age_of_25th_student : totalAgeOfAllStudents - totalAgeOf24Students = 56 := by
  sorry

end NUMINAMATH_GPT_age_of_25th_student_l2427_242729


namespace NUMINAMATH_GPT_budget_equality_year_l2427_242710

theorem budget_equality_year :
  ∀ Q R V W : ℕ → ℝ,
  Q 0 = 540000 ∧ R 0 = 660000 ∧ V 0 = 780000 ∧ W 0 = 900000 ∧
  (∀ n, Q (n+1) = Q n + 40000 ∧ 
         R (n+1) = R n + 30000 ∧ 
         V (n+1) = V n - 10000 ∧ 
         W (n+1) = W n - 20000) →
  ∃ n : ℕ, 1990 + n = 1995 ∧ 
  Q n + R n = V n + W n := 
by 
  sorry

end NUMINAMATH_GPT_budget_equality_year_l2427_242710


namespace NUMINAMATH_GPT_square_side_percentage_increase_l2427_242768

theorem square_side_percentage_increase (s : ℝ) (p : ℝ) :
  (s * (1 + p / 100)) ^ 2 = 1.44 * s ^ 2 → p = 20 :=
by
  sorry

end NUMINAMATH_GPT_square_side_percentage_increase_l2427_242768


namespace NUMINAMATH_GPT_exterior_angle_regular_octagon_l2427_242770

theorem exterior_angle_regular_octagon : 
  (∃ n : ℕ, n = 8 ∧ ∀ (i : ℕ), i < n → true) → 
  ∃ θ : ℝ, θ = 45 := by
  sorry

end NUMINAMATH_GPT_exterior_angle_regular_octagon_l2427_242770


namespace NUMINAMATH_GPT_largest_divisor_is_one_l2427_242753

theorem largest_divisor_is_one (p q : ℤ) (hpq : p > q) (hp : p % 2 = 1) (hq : q % 2 = 0) :
  ∀ d : ℤ, (∀ p q : ℤ, p > q → p % 2 = 1 → q % 2 = 0 → d ∣ (p^2 - q^2)) → d = 1 :=
sorry

end NUMINAMATH_GPT_largest_divisor_is_one_l2427_242753


namespace NUMINAMATH_GPT_correct_expression_l2427_242740

variables {a b c : ℝ}

theorem correct_expression :
  -2 * (3 * a - b) + 3 * (2 * a + b) = 5 * b :=
by
  sorry

end NUMINAMATH_GPT_correct_expression_l2427_242740


namespace NUMINAMATH_GPT_pants_price_l2427_242734

theorem pants_price (P B : ℝ) 
  (condition1 : P + B = 70.93)
  (condition2 : P = B - 2.93) : 
  P = 34.00 :=
by
  sorry

end NUMINAMATH_GPT_pants_price_l2427_242734


namespace NUMINAMATH_GPT_farmer_pigs_chickens_l2427_242717

-- Defining the problem in Lean 4

theorem farmer_pigs_chickens (p ch : ℕ) (h₁ : 30 * p + 24 * ch = 1200) (h₂ : p > 0) (h₃ : ch > 0) : 
  (p = 4) ∧ (ch = 45) :=
by sorry

end NUMINAMATH_GPT_farmer_pigs_chickens_l2427_242717


namespace NUMINAMATH_GPT_count_ways_to_choose_4_cards_l2427_242752

-- A standard deck has 4 suits
def suits : Finset ℕ := {1, 2, 3, 4}

-- Each suit has 6 even cards: 2, 4, 6, 8, 10, and Queen (12)
def even_cards_per_suit : Finset ℕ := {2, 4, 6, 8, 10, 12}

-- Define the problem in Lean: 
-- Total number of ways to choose 4 cards such that all cards are of different suits and each is an even card.
theorem count_ways_to_choose_4_cards : (suits.card = 4 ∧ even_cards_per_suit.card = 6) → (1 * 6^4 = 1296) :=
by
  intros h
  have suits_distinct : suits.card = 4 := h.1
  have even_cards_count : even_cards_per_suit.card = 6 := h.2
  sorry

end NUMINAMATH_GPT_count_ways_to_choose_4_cards_l2427_242752


namespace NUMINAMATH_GPT_sector_area_l2427_242762

theorem sector_area (r θ : ℝ) (h₁ : θ = 2) (h₂ : r * θ = 4) : (1 / 2) * r^2 * θ = 4 :=
by
  sorry

end NUMINAMATH_GPT_sector_area_l2427_242762


namespace NUMINAMATH_GPT_original_soldiers_eq_136_l2427_242777

-- Conditions
def original_soldiers (n : ℕ) : ℕ := 8 * n
def after_adding_120 (n : ℕ) : ℕ := original_soldiers n + 120
def after_removing_120 (n : ℕ) : ℕ := original_soldiers n - 120

-- Given that both after_adding_120 n and after_removing_120 n are perfect squares.
def is_square (m : ℕ) : Prop := ∃ k : ℕ, k * k = m

-- Theorem statement
theorem original_soldiers_eq_136 : ∃ n : ℕ, original_soldiers n = 136 ∧ 
                                   is_square (after_adding_120 n) ∧ 
                                   is_square (after_removing_120 n) :=
sorry

end NUMINAMATH_GPT_original_soldiers_eq_136_l2427_242777


namespace NUMINAMATH_GPT_interest_rate_is_5_percent_l2427_242724

noncomputable def interest_rate_1200_loan (R : ℝ) : Prop :=
  let time := 3.888888888888889
  let principal_1000 := 1000
  let principal_1200 := 1200
  let rate_1000 := 0.03
  let total_interest := 350
  principal_1000 * rate_1000 * time + principal_1200 * (R / 100) * time = total_interest

theorem interest_rate_is_5_percent :
  interest_rate_1200_loan 5 :=
by
  sorry

end NUMINAMATH_GPT_interest_rate_is_5_percent_l2427_242724


namespace NUMINAMATH_GPT_tank_length_l2427_242737

theorem tank_length (W D : ℝ) (cost_per_sq_m total_cost : ℝ) (L : ℝ):
  W = 12 →
  D = 6 →
  cost_per_sq_m = 0.70 →
  total_cost = 520.8 →
  total_cost = cost_per_sq_m * ((2 * (W * D)) + (2 * (L * D)) + (L * W)) →
  L = 25 :=
by
  intros hW hD hCostPerSqM hTotalCost hEquation
  sorry

end NUMINAMATH_GPT_tank_length_l2427_242737


namespace NUMINAMATH_GPT_sum_even_if_product_odd_l2427_242709

theorem sum_even_if_product_odd (a b : ℤ) (h : (a * b) % 2 = 1) : (a + b) % 2 = 0 := 
by
  sorry

end NUMINAMATH_GPT_sum_even_if_product_odd_l2427_242709


namespace NUMINAMATH_GPT_remainder_of_98_times_102_divided_by_9_l2427_242723

theorem remainder_of_98_times_102_divided_by_9 : (98 * 102) % 9 = 6 :=
by
  sorry

end NUMINAMATH_GPT_remainder_of_98_times_102_divided_by_9_l2427_242723


namespace NUMINAMATH_GPT_probability_of_purple_probability_of_blue_or_purple_l2427_242783

def total_jelly_beans : ℕ := 60
def purple_jelly_beans : ℕ := 5
def blue_jelly_beans : ℕ := 18

theorem probability_of_purple :
  (purple_jelly_beans : ℚ) / total_jelly_beans = 1 / 12 :=
by
  sorry
  
theorem probability_of_blue_or_purple :
  (blue_jelly_beans + purple_jelly_beans : ℚ) / total_jelly_beans = 23 / 60 :=
by
  sorry

end NUMINAMATH_GPT_probability_of_purple_probability_of_blue_or_purple_l2427_242783


namespace NUMINAMATH_GPT_angles_on_line_y_eq_x_l2427_242744

-- Define a predicate representing that an angle has its terminal side on the line y = x
def angle_on_line_y_eq_x (α : ℝ) : Prop :=
  ∃ k : ℤ, α = k * Real.pi + Real.pi / 4

-- The goal is to prove that the set of all such angles is as stated
theorem angles_on_line_y_eq_x :
  { α : ℝ | ∃ k : ℤ, α = k * Real.pi + Real.pi / 4 } = { α : ℝ | angle_on_line_y_eq_x α } :=
sorry

end NUMINAMATH_GPT_angles_on_line_y_eq_x_l2427_242744


namespace NUMINAMATH_GPT_original_board_length_before_final_cut_l2427_242772

-- Given conditions
def initial_length : ℕ := 143
def first_cut_length : ℕ := 25
def final_cut_length : ℕ := 7

def board_length_after_first_cut : ℕ := initial_length - first_cut_length
def board_length_after_final_cut : ℕ := board_length_after_first_cut - final_cut_length

-- The theorem to be proved
theorem original_board_length_before_final_cut : board_length_after_first_cut + final_cut_length = 125 :=
by
  sorry

end NUMINAMATH_GPT_original_board_length_before_final_cut_l2427_242772


namespace NUMINAMATH_GPT_complete_the_square_l2427_242793

theorem complete_the_square (a : ℝ) : a^2 + 4 * a - 5 = (a + 2)^2 - 9 :=
by sorry

end NUMINAMATH_GPT_complete_the_square_l2427_242793


namespace NUMINAMATH_GPT_square_root_combination_l2427_242766

theorem square_root_combination (a : ℝ) (h : 1 + a = 4 - 2 * a) : a = 1 :=
by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_square_root_combination_l2427_242766


namespace NUMINAMATH_GPT_jess_father_first_round_l2427_242707

theorem jess_father_first_round (initial_blocks : ℕ)
  (players : ℕ)
  (blocks_before_jess_turn : ℕ)
  (jess_falls_tower_round : ℕ)
  (h1 : initial_blocks = 54)
  (h2 : players = 5)
  (h3 : blocks_before_jess_turn = 28)
  (h4 : ∀ rounds : ℕ, rounds * players ≥ 26 → jess_falls_tower_round = rounds + 1) :
  jess_falls_tower_round = 6 := 
by
  sorry

end NUMINAMATH_GPT_jess_father_first_round_l2427_242707


namespace NUMINAMATH_GPT_simplify_radical_l2427_242756

theorem simplify_radical (x : ℝ) :
  2 * Real.sqrt (50 * x^3) * Real.sqrt (45 * x^5) * Real.sqrt (98 * x^7) = 420 * x^7 * Real.sqrt (5 * x) :=
by
  sorry

end NUMINAMATH_GPT_simplify_radical_l2427_242756


namespace NUMINAMATH_GPT_impossible_to_arrange_distinct_integers_in_grid_l2427_242787

theorem impossible_to_arrange_distinct_integers_in_grid :
  ¬ ∃ (f : Fin 25 × Fin 41 → ℤ),
    (∀ i j, abs (f i - f j) ≤ 16 → (i ≠ j) → (i.1 = j.1 ∨ i.2 = j.2)) ∧
    (∃ i j, i ≠ j ∧ f i = f j) := 
sorry

end NUMINAMATH_GPT_impossible_to_arrange_distinct_integers_in_grid_l2427_242787


namespace NUMINAMATH_GPT_total_value_of_coins_l2427_242774

variable (numCoins : ℕ) (coinsValue : ℕ) 

theorem total_value_of_coins : 
  numCoins = 15 → 
  (∀ n: ℕ, n = 5 → coinsValue = 12) → 
  ∃ totalValue : ℕ, totalValue = 36 :=
  by
    sorry

end NUMINAMATH_GPT_total_value_of_coins_l2427_242774


namespace NUMINAMATH_GPT_calculation_division_l2427_242749

theorem calculation_division :
  ((27 * 0.92 * 0.85) / (23 * 1.7 * 1.8)) = 0.3 :=
by
  sorry

end NUMINAMATH_GPT_calculation_division_l2427_242749


namespace NUMINAMATH_GPT_system_solution_y_greater_than_five_l2427_242758

theorem system_solution_y_greater_than_five (m x y : ℝ) :
  (y = (m + 1) * x + 2) → 
  (y = (3 * m - 2) * x + 5) → 
  y > 5 ↔ 
  m ≠ 3 / 2 := 
sorry

end NUMINAMATH_GPT_system_solution_y_greater_than_five_l2427_242758


namespace NUMINAMATH_GPT_air_conditioner_sales_l2427_242736

-- Definitions based on conditions
def ratio_air_conditioners_refrigerators : ℕ := 5
def ratio_refrigerators_air_conditioners : ℕ := 3
def difference_in_sales : ℕ := 54

-- The property to be proven: 
def number_of_air_conditioners : ℕ := 135

theorem air_conditioner_sales
  (r_ac : ℕ := ratio_air_conditioners_refrigerators) 
  (r_ref : ℕ := ratio_refrigerators_air_conditioners) 
  (diff : ℕ := difference_in_sales) 
  : number_of_air_conditioners = 135 := sorry

end NUMINAMATH_GPT_air_conditioner_sales_l2427_242736


namespace NUMINAMATH_GPT_larger_triangle_perimeter_is_65_l2427_242704

theorem larger_triangle_perimeter_is_65 (s1 s2 s3 t1 t2 t3 : ℝ)
  (h1 : s1 = 7) (h2 : s2 = 7) (h3 : s3 = 12)
  (h4 : t3 = 30)
  (similar : t1 / s1 = t2 / s2 ∧ t2 / s2 = t3 / s3) :
  t1 + t2 + t3 = 65 := by
  sorry

end NUMINAMATH_GPT_larger_triangle_perimeter_is_65_l2427_242704


namespace NUMINAMATH_GPT_expected_value_of_12_sided_die_is_6_5_l2427_242714

noncomputable def sum_arithmetic_series (n : ℕ) (a : ℕ) (l : ℕ) : ℕ :=
  n * (a + l) / 2

noncomputable def expected_value_12_sided_die : ℚ :=
  (sum_arithmetic_series 12 1 12 : ℚ) / 12

theorem expected_value_of_12_sided_die_is_6_5 :
  expected_value_12_sided_die = 6.5 :=
by
  sorry

end NUMINAMATH_GPT_expected_value_of_12_sided_die_is_6_5_l2427_242714


namespace NUMINAMATH_GPT_concentric_spheres_volume_l2427_242761

theorem concentric_spheres_volume :
  let r1 := 4
  let r2 := 7
  let r3 := 10
  let volume (r : ℝ) := (4/3) * Real.pi * r^3
  volume r3 - volume r2 = 876 * Real.pi := 
by
  let r1 := 4
  let r2 := 7
  let r3 := 10
  let volume (r : ℝ) := (4/3) * Real.pi * r^3
  show volume r3 - volume r2 = 876 * Real.pi
  sorry

end NUMINAMATH_GPT_concentric_spheres_volume_l2427_242761


namespace NUMINAMATH_GPT_hiking_rate_l2427_242702

theorem hiking_rate (rate_uphill: ℝ) (time_total: ℝ) (time_uphill: ℝ) (rate_downhill: ℝ) 
  (h1: rate_uphill = 4) (h2: time_total = 3) (h3: time_uphill = 1.2) : rate_downhill = 4.8 / (time_total - time_uphill) :=
by
  sorry

end NUMINAMATH_GPT_hiking_rate_l2427_242702


namespace NUMINAMATH_GPT_alcohol_percentage_new_mixture_l2427_242731

/--
Given:
1. The initial mixture has 15 liters.
2. The mixture contains 20% alcohol.
3. 5 liters of water is added to the mixture.

Prove:
The percentage of alcohol in the new mixture is 15%.
-/
theorem alcohol_percentage_new_mixture :
  let initial_mixture_volume := 15 -- in liters
  let initial_alcohol_percentage := 20 / 100
  let initial_alcohol_volume := initial_alcohol_percentage * initial_mixture_volume
  let added_water_volume := 5 -- in liters
  let new_total_volume := initial_mixture_volume + added_water_volume
  let new_alcohol_percentage := (initial_alcohol_volume / new_total_volume) * 100
  new_alcohol_percentage = 15 := 
by
  -- Proof steps go here
  sorry

end NUMINAMATH_GPT_alcohol_percentage_new_mixture_l2427_242731


namespace NUMINAMATH_GPT_inequality_example_l2427_242791

variable (f : ℝ → ℝ)
variable (f' : ℝ → ℝ)
variable (der : ∀ x, deriv f x = f' x)

theorem inequality_example (h : ∀ x : ℝ, f x > f' x) :
  2023 * f (Real.log 2022) > 2022 * f (Real.log 2023)
:= sorry

end NUMINAMATH_GPT_inequality_example_l2427_242791


namespace NUMINAMATH_GPT_correct_value_two_decimal_places_l2427_242746

theorem correct_value_two_decimal_places (x : ℝ) 
  (h1 : 8 * x + 8 = 56) : 
  (x / 8) + 7 = 7.75 :=
sorry

end NUMINAMATH_GPT_correct_value_two_decimal_places_l2427_242746


namespace NUMINAMATH_GPT_walmart_total_sales_l2427_242722

-- Define the constants for the prices
def thermometer_price : ℕ := 2
def hot_water_bottle_price : ℕ := 6

-- Define the quantities and relationships
def hot_water_bottles_sold : ℕ := 60
def thermometer_ratio : ℕ := 7
def thermometers_sold : ℕ := thermometer_ratio * hot_water_bottles_sold

-- Define the total sales for thermometers and hot-water bottles
def thermometer_sales : ℕ := thermometers_sold * thermometer_price
def hot_water_bottle_sales : ℕ := hot_water_bottles_sold * hot_water_bottle_price

-- Define the total sales amount
def total_sales : ℕ := thermometer_sales + hot_water_bottle_sales

-- Theorem statement
theorem walmart_total_sales : total_sales = 1200 := by
  sorry

end NUMINAMATH_GPT_walmart_total_sales_l2427_242722


namespace NUMINAMATH_GPT_convex_polygon_triangle_count_l2427_242751

theorem convex_polygon_triangle_count {n : ℕ} (h : n ≥ 5) :
  ∃ T : ℕ, T ≤ n * (2 * n - 5) / 3 :=
by
  sorry

end NUMINAMATH_GPT_convex_polygon_triangle_count_l2427_242751


namespace NUMINAMATH_GPT_aaronTotalOwed_l2427_242759

def monthlyPayment : ℝ := 100
def numberOfMonths : ℕ := 12
def interestRate : ℝ := 0.1

def totalCostWithoutInterest : ℝ := monthlyPayment * (numberOfMonths : ℝ)
def interestAmount : ℝ := totalCostWithoutInterest * interestRate
def totalAmountOwed : ℝ := totalCostWithoutInterest + interestAmount

theorem aaronTotalOwed : totalAmountOwed = 1320 := by
  sorry

end NUMINAMATH_GPT_aaronTotalOwed_l2427_242759


namespace NUMINAMATH_GPT_missing_digit_B_l2427_242719

theorem missing_digit_B :
  ∃ B : ℕ, 0 ≤ B ∧ B ≤ 9 ∧ (200 + 10 * B + 5) % 13 = 0 := 
sorry

end NUMINAMATH_GPT_missing_digit_B_l2427_242719


namespace NUMINAMATH_GPT_total_people_on_boats_l2427_242771

theorem total_people_on_boats (boats : ℕ) (people_per_boat : ℕ) (h_boats : boats = 5) (h_people : people_per_boat = 3) : boats * people_per_boat = 15 :=
by
  sorry

end NUMINAMATH_GPT_total_people_on_boats_l2427_242771


namespace NUMINAMATH_GPT_max_value_of_expr_l2427_242799

noncomputable def max_expr (a b : ℝ) (h : a + b = 5) : ℝ :=
  a^4 * b + a^3 * b + a^2 * b + a * b + a * b^2 + a * b^3 + a * b^4

theorem max_value_of_expr (a b : ℝ) (h : a + b = 5) : max_expr a b h ≤ 6084 / 17 :=
sorry

end NUMINAMATH_GPT_max_value_of_expr_l2427_242799


namespace NUMINAMATH_GPT_largest_angle_of_triangle_l2427_242782

theorem largest_angle_of_triangle (x : ℝ) 
  (h1 : 4 * x + 5 * x + 9 * x = 180) 
  (h2 : 4 * x > 40) : 
  9 * x = 90 := 
sorry

end NUMINAMATH_GPT_largest_angle_of_triangle_l2427_242782


namespace NUMINAMATH_GPT_fraction_eq_zero_iff_x_eq_2_l2427_242754

theorem fraction_eq_zero_iff_x_eq_2 (x : ℝ) : (x - 2) / (x + 2) = 0 ↔ x = 2 := by sorry

end NUMINAMATH_GPT_fraction_eq_zero_iff_x_eq_2_l2427_242754


namespace NUMINAMATH_GPT_value_of_a_8_l2427_242755

noncomputable def S (n : ℕ) : ℕ := n^2
noncomputable def a (n : ℕ) : ℕ := if n = 1 then S n else S n - S (n - 1)

theorem value_of_a_8 : a 8 = 15 := 
by
  sorry

end NUMINAMATH_GPT_value_of_a_8_l2427_242755


namespace NUMINAMATH_GPT_fuel_tank_capacity_l2427_242727

theorem fuel_tank_capacity
  (ethanol_A_fraction : ℝ)
  (ethanol_B_fraction : ℝ)
  (ethanol_total : ℝ)
  (fuel_A_volume : ℝ)
  (C : ℝ)
  (h1 : ethanol_A_fraction = 0.12)
  (h2 : ethanol_B_fraction = 0.16)
  (h3 : ethanol_total = 28)
  (h4 : fuel_A_volume = 99.99999999999999)
  (h5 : 0.12 * 99.99999999999999 + 0.16 * (C - 99.99999999999999) = 28) :
  C = 200 := 
sorry

end NUMINAMATH_GPT_fuel_tank_capacity_l2427_242727


namespace NUMINAMATH_GPT_black_car_overtakes_red_car_in_one_hour_l2427_242700

-- Define the speeds of the cars
def red_car_speed := 30 -- in miles per hour
def black_car_speed := 50 -- in miles per hour

-- Define the initial distance between the cars
def initial_distance := 20 -- in miles

-- Calculate the time required for the black car to overtake the red car
theorem black_car_overtakes_red_car_in_one_hour : initial_distance / (black_car_speed - red_car_speed) = 1 := by
  sorry

end NUMINAMATH_GPT_black_car_overtakes_red_car_in_one_hour_l2427_242700


namespace NUMINAMATH_GPT_solve_eq_1_solve_eq_2_l2427_242721

open Real

theorem solve_eq_1 :
  ∃ x : ℝ, x - 2 * (x - 4) = 3 * (1 - x) ∧ x = -2.5 :=
by
  sorry

theorem solve_eq_2 :
  ∃ x : ℝ, (2 * x + 1) / 3 - (5 * x - 1) / 60 = 1 ∧ x = 39 / 35 :=
by
  sorry

end NUMINAMATH_GPT_solve_eq_1_solve_eq_2_l2427_242721


namespace NUMINAMATH_GPT_problem_a_b_n_l2427_242701

theorem problem_a_b_n (a b n : ℕ) (h : ∀ k : ℕ, k ≠ 0 → (b - k) ∣ (a - k^n)) : a = b^n := 
sorry

end NUMINAMATH_GPT_problem_a_b_n_l2427_242701


namespace NUMINAMATH_GPT_total_roses_planted_l2427_242741

def roses_planted_two_days_ago := 50
def roses_planted_yesterday := roses_planted_two_days_ago + 20
def roses_planted_today := 2 * roses_planted_two_days_ago

theorem total_roses_planted :
  roses_planted_two_days_ago + roses_planted_yesterday + roses_planted_today = 220 := by
  sorry

end NUMINAMATH_GPT_total_roses_planted_l2427_242741


namespace NUMINAMATH_GPT_blueberry_jelly_amount_l2427_242795

-- Definition of the conditions
def total_jelly : ℕ := 6310
def strawberry_jelly : ℕ := 1792

-- Formal statement of the problem
theorem blueberry_jelly_amount : 
  total_jelly - strawberry_jelly = 4518 :=
by
  sorry

end NUMINAMATH_GPT_blueberry_jelly_amount_l2427_242795


namespace NUMINAMATH_GPT_simplify_and_evaluate_l2427_242789

theorem simplify_and_evaluate (m : ℝ) (h : m = Real.sqrt 2) :
  (1 + 1 / (m - 2)) / ((m^2 - m) / (m - 2)) = Real.sqrt 2 / 2 :=
by
  sorry

end NUMINAMATH_GPT_simplify_and_evaluate_l2427_242789


namespace NUMINAMATH_GPT_fourth_root_expression_l2427_242748

-- Define a positive real number y
variable (y : ℝ) (hy : 0 < y)

-- State the problem in Lean
theorem fourth_root_expression : 
  Real.sqrt (Real.sqrt (y^2 * Real.sqrt y)) = y^(5/8) := sorry

end NUMINAMATH_GPT_fourth_root_expression_l2427_242748


namespace NUMINAMATH_GPT_right_triangle_even_or_odd_l2427_242785

theorem right_triangle_even_or_odd (a b c : ℕ) (ha : Even a ∨ Odd a) (hb : Even b ∨ Odd b) (h : a^2 + b^2 = c^2) : 
  Even c ∨ (Even a ∧ Odd b) ∨ (Odd a ∧ Even b) :=
by
  sorry

end NUMINAMATH_GPT_right_triangle_even_or_odd_l2427_242785


namespace NUMINAMATH_GPT_ratio_AD_DC_in_ABC_l2427_242725

theorem ratio_AD_DC_in_ABC 
  (A B C D : Type)
  [MetricSpace A] [MetricSpace B] [MetricSpace C] [MetricSpace D]
  (AB BC AC : Real) 
  (hAB : AB = 6) (hBC : BC = 8) (hAC : AC = 10) 
  (BD : Real) 
  (hBD : BD = 8) 
  (AD DC : Real)
  (hAD : AD = 2 * Real.sqrt 7)
  (hDC : DC = 10 - 2 * Real.sqrt 7) :
  AD / DC = (10 * Real.sqrt 7 + 14) / 36 :=
sorry

end NUMINAMATH_GPT_ratio_AD_DC_in_ABC_l2427_242725


namespace NUMINAMATH_GPT_calculate_total_amount_l2427_242767

theorem calculate_total_amount
  (price1 discount1 price2 discount2 additional_discount : ℝ)
  (h1 : price1 = 76) (h2 : discount1 = 25)
  (h3 : price2 = 85) (h4 : discount2 = 15)
  (h5 : additional_discount = 10) :
  price1 - discount1 + price2 - discount2 - additional_discount = 111 :=
by {
  sorry
}

end NUMINAMATH_GPT_calculate_total_amount_l2427_242767


namespace NUMINAMATH_GPT_prove_m_value_l2427_242715

theorem prove_m_value (m : ℕ) : 8^4 = 4^m → m = 6 := by
  sorry

end NUMINAMATH_GPT_prove_m_value_l2427_242715


namespace NUMINAMATH_GPT_geometric_sequence_general_term_l2427_242747

theorem geometric_sequence_general_term (a : ℕ → ℝ) (h1 : a 1 = 3) (h4 : a 4 = 81) :
  ∃ q : ℝ, (a n = 3 * q ^ (n - 1)) := by
  sorry

end NUMINAMATH_GPT_geometric_sequence_general_term_l2427_242747


namespace NUMINAMATH_GPT_value_of_k_l2427_242763

theorem value_of_k (k : ℤ) (h : (∀ x : ℤ, (x^2 - k * x - 6) = (x - 2) * (x + 3))) : k = -1 := by
  sorry

end NUMINAMATH_GPT_value_of_k_l2427_242763


namespace NUMINAMATH_GPT_simplify_expression_l2427_242775

noncomputable def original_expression (x : ℝ) : ℝ :=
(x - 3 * x / (x + 1)) / ((x - 2) / (x^2 + 2 * x + 1))

theorem simplify_expression:
  ∀ (x : ℝ), -1 ≤ x ∧ x ≤ 2 ∧ x ≠ -1 ∧ x ≠ 2 → 
  (original_expression x = x^2 + x) ∧ 
  ((x = 1 → original_expression x = 2) ∧ (x = 0 → original_expression x = 0)) :=
by
  intros
  sorry

end NUMINAMATH_GPT_simplify_expression_l2427_242775


namespace NUMINAMATH_GPT_sequence_distinct_l2427_242708

theorem sequence_distinct (f : ℕ → ℕ) (h : ∀ n : ℕ, f (f n) = f (n + 1) + f n) :
  ∀ i j : ℕ, i ≠ j → f i ≠ f j :=
by
  sorry

end NUMINAMATH_GPT_sequence_distinct_l2427_242708


namespace NUMINAMATH_GPT_g_odd_find_a_f_increasing_l2427_242781

-- Problem (I): Prove that if g(x) = f(x) - a is an odd function, then a = 1, given f(x) = 1 - 2/x.
theorem g_odd_find_a (f : ℝ → ℝ) (g : ℝ → ℝ) (a : ℝ) : 
  (∀ x, f x = 1 - (2 / x)) → 
  (∀ x, g x = f x - a) → 
  (∀ x, g (-x) = - g x) → 
  a = 1 := 
  by
  intros h1 h2 h3
  sorry

-- Problem (II): Prove that f(x) is monotonically increasing on (0, +∞),
-- given f(x) = 1 - 2/x.

theorem f_increasing (f : ℝ → ℝ) : 
  (∀ x, f x = 1 - (2 / x)) → 
  ∀ (x1 x2 : ℝ), 0 < x1 → x1 < x2 → f x1 < f x2 := 
  by
  intros h1 x1 x2 hx1 hx12
  sorry

end NUMINAMATH_GPT_g_odd_find_a_f_increasing_l2427_242781


namespace NUMINAMATH_GPT_b_2_pow_100_value_l2427_242742

def seq (b : ℕ → ℕ) : Prop :=
  b 1 = 3 ∧ ∀ n > 0, b (2 * n) = 2 * n * b n

theorem b_2_pow_100_value
  (b : ℕ → ℕ)
  (h_seq : seq b) :
  b (2^100) = 2^5050 * 3 :=
by
  sorry

end NUMINAMATH_GPT_b_2_pow_100_value_l2427_242742


namespace NUMINAMATH_GPT_average_age_with_teacher_l2427_242779

theorem average_age_with_teacher (A : ℕ) (h : 21 * 16 = 20 * A + 36) : A = 15 := by
  sorry

end NUMINAMATH_GPT_average_age_with_teacher_l2427_242779


namespace NUMINAMATH_GPT_problem1_problem2_l2427_242712

-- Problem 1: Prove \( \sqrt{10} \times \sqrt{2} + \sqrt{15} \div \sqrt{3} = 3\sqrt{5} \)
theorem problem1 : Real.sqrt 10 * Real.sqrt 2 + Real.sqrt 15 / Real.sqrt 3 = 3 * Real.sqrt 5 := 
by sorry

-- Problem 2: Prove \( \sqrt{27} - (\sqrt{12} - \sqrt{\frac{1}{3}}) = \frac{4\sqrt{3}}{3} \)
theorem problem2 : Real.sqrt 27 - (Real.sqrt 12 - Real.sqrt (1 / 3)) = (4 * Real.sqrt 3) / 3 :=
by sorry

end NUMINAMATH_GPT_problem1_problem2_l2427_242712


namespace NUMINAMATH_GPT_family_boys_girls_l2427_242728

theorem family_boys_girls (B G : ℕ) 
  (h1 : B - 1 = G) 
  (h2 : B = 2 * (G - 1)) : 
  B = 4 ∧ G = 3 := 
by {
  sorry
}

end NUMINAMATH_GPT_family_boys_girls_l2427_242728


namespace NUMINAMATH_GPT_prob_neither_alive_l2427_242711

/-- Define the probability that a man will be alive for 10 more years -/
def prob_man_alive : ℚ := 1 / 4

/-- Define the probability that a wife will be alive for 10 more years -/
def prob_wife_alive : ℚ := 1 / 3

/-- Prove that the probability that neither the man nor his wife will be alive for 10 more years is 1/2 -/
theorem prob_neither_alive (p_man_alive p_wife_alive : ℚ)
    (h1 : p_man_alive = prob_man_alive) (h2 : p_wife_alive = prob_wife_alive) :
    (1 - p_man_alive) * (1 - p_wife_alive) = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_prob_neither_alive_l2427_242711


namespace NUMINAMATH_GPT_number_of_bookshelves_l2427_242786

def total_space : ℕ := 400
def reserved_space : ℕ := 160
def shelf_space : ℕ := 80

theorem number_of_bookshelves : (total_space - reserved_space) / shelf_space = 3 := by
  sorry

end NUMINAMATH_GPT_number_of_bookshelves_l2427_242786


namespace NUMINAMATH_GPT_sufficient_not_necessary_l2427_242730

theorem sufficient_not_necessary (a : ℝ) (h1 : a > 0) : (a^2 + a ≥ 0) ∧ ¬(a^2 + a ≥ 0 → a > 0) :=
by
  sorry

end NUMINAMATH_GPT_sufficient_not_necessary_l2427_242730


namespace NUMINAMATH_GPT_problem_l2427_242732

def f (n : ℕ) : ℤ := 3 ^ (2 * n) - 32 * n ^ 2 + 24 * n - 1

theorem problem (n : ℕ) (h : 0 < n) : 512 ∣ f n := sorry

end NUMINAMATH_GPT_problem_l2427_242732


namespace NUMINAMATH_GPT_total_fireworks_l2427_242713

-- Define the conditions
def fireworks_per_number := 6
def fireworks_per_letter := 5
def numbers_in_year := 4
def letters_in_phrase := 12
def number_of_boxes := 50
def fireworks_per_box := 8

-- Main statement: Prove the total number of fireworks lit during the display
theorem total_fireworks : fireworks_per_number * numbers_in_year + fireworks_per_letter * letters_in_phrase + number_of_boxes * fireworks_per_box = 484 :=
by
  sorry

end NUMINAMATH_GPT_total_fireworks_l2427_242713


namespace NUMINAMATH_GPT_average_minutes_run_l2427_242784

theorem average_minutes_run (t : ℕ) (t_pos : 0 < t) 
  (average_first_graders : ℕ := 8) 
  (average_second_graders : ℕ := 12) 
  (average_third_graders : ℕ := 16)
  (num_first_graders : ℕ := 9 * t)
  (num_second_graders : ℕ := 3 * t)
  (num_third_graders : ℕ := t) :
  (8 * 9 * t + 12 * 3 * t + 16 * t) / (9 * t + 3 * t + t) = 10 := 
by
  sorry

end NUMINAMATH_GPT_average_minutes_run_l2427_242784


namespace NUMINAMATH_GPT_garden_breadth_l2427_242739

-- Problem statement conditions
def perimeter : ℝ := 600
def length : ℝ := 205

-- Translate the problem into Lean:
theorem garden_breadth (breadth : ℝ) (h1 : 2 * (length + breadth) = perimeter) : breadth = 95 := 
by sorry

end NUMINAMATH_GPT_garden_breadth_l2427_242739


namespace NUMINAMATH_GPT_page_sum_incorrect_l2427_242703

theorem page_sum_incorrect (sheets : List (Nat × Nat)) (h_sheets_len : sheets.length = 25)
  (h_consecutive : ∀ (a b : Nat), (a, b) ∈ sheets → (b = a + 1 ∨ a = b + 1))
  (h_sum_eq_2020 : (sheets.map (λ p => p.1 + p.2)).sum = 2020) : False :=
by
  sorry

end NUMINAMATH_GPT_page_sum_incorrect_l2427_242703


namespace NUMINAMATH_GPT_gcd_g102_g103_l2427_242735

def g (x : ℕ) : ℕ := x^2 - x + 2007

theorem gcd_g102_g103 : 
  Nat.gcd (g 102) (g 103) = 3 :=
by
  sorry

end NUMINAMATH_GPT_gcd_g102_g103_l2427_242735


namespace NUMINAMATH_GPT_find_min_value_l2427_242760

noncomputable def min_value (a b : ℝ) (h_a : a > 0) (h_b : b > 0) (h_slope : 2 * a + b = 1) :=
  (8 * a + b) / (a * b)

theorem find_min_value (a b : ℝ) (h_a : a > 0) (h_b : b > 0) (h_slope : 2 * a + b = 1) :
  min_value a b h_a h_b h_slope = 18 :=
sorry

end NUMINAMATH_GPT_find_min_value_l2427_242760


namespace NUMINAMATH_GPT_find_other_solution_l2427_242738

theorem find_other_solution (x : ℚ) :
  (72 * x ^ 2 + 43 = 113 * x - 12) → (x = 3 / 8) → (x = 43 / 36 ∨ x = 3 / 8) :=
by
  sorry

end NUMINAMATH_GPT_find_other_solution_l2427_242738


namespace NUMINAMATH_GPT_victoria_worked_weeks_l2427_242769

-- Definitions for given conditions
def hours_worked_per_day : ℕ := 9
def total_hours_worked : ℕ := 315
def days_in_week : ℕ := 7

-- Main theorem to prove
theorem victoria_worked_weeks : total_hours_worked / hours_worked_per_day / days_in_week = 5 :=
by
  sorry

end NUMINAMATH_GPT_victoria_worked_weeks_l2427_242769


namespace NUMINAMATH_GPT_intersecting_line_l2427_242796

theorem intersecting_line {x y : ℝ} (h1 : x^2 + y^2 = 10) (h2 : (x - 1)^2 + (y - 3)^2 = 10) :
  x + 3 * y - 5 = 0 :=
sorry

end NUMINAMATH_GPT_intersecting_line_l2427_242796


namespace NUMINAMATH_GPT_circles_intersect_l2427_242720

-- Definition of the first circle
def circle1 (x y r : ℝ) : Prop := x^2 + y^2 = r^2

-- Definition of the second circle
def circle2 (x y : ℝ) (r : ℝ) : Prop := (x - 3)^2 + (y + 4)^2 = 49

-- Statement proving the range of r for which the circles intersect
theorem circles_intersect (r : ℝ) (h : r > 0) : (∃ x y : ℝ, circle1 x y r ∧ circle2 x y r) → (2 ≤ r ∧ r ≤ 12) :=
by
  -- Definition of the distance between centers and conditions for intersection
  sorry

end NUMINAMATH_GPT_circles_intersect_l2427_242720


namespace NUMINAMATH_GPT_repeating_decimal_sum_l2427_242743

theorem repeating_decimal_sum :
  (0.12121212 + 0.003003003 + 0.0000500005 : ℚ) = 124215 / 999999 :=
by 
  have h1 : (0.12121212 : ℚ) = (0.12 + 0.0012) := sorry
  have h2 : (0.003003003 : ℚ) = (0.003 + 0.000003) := sorry
  have h3 : (0.0000500005 : ℚ) = (0.00005 + 0.0000000005) := sorry
  sorry


end NUMINAMATH_GPT_repeating_decimal_sum_l2427_242743
