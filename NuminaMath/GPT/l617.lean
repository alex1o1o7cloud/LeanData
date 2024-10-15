import Mathlib

namespace NUMINAMATH_GPT_exists_fg_pairs_l617_61708

theorem exists_fg_pairs (a b : ℤ) :
  (∃ (f g : ℤ → ℤ), (∀ x : ℤ, f (g x) = x + a) ∧ (∀ x : ℤ, g (f x) = x + b)) ↔ (a = b ∨ a = -b) := 
sorry

end NUMINAMATH_GPT_exists_fg_pairs_l617_61708


namespace NUMINAMATH_GPT_factorization_sum_l617_61788

theorem factorization_sum (a b c : ℤ) 
  (h1 : ∀ x : ℤ, x ^ 2 + 9 * x + 18 = (x + a) * (x + b))
  (h2 : ∀ x : ℤ, x ^ 2 + 19 * x + 90 = (x + b) * (x + c)) :
  a + b + c = 22 := by
sorry

end NUMINAMATH_GPT_factorization_sum_l617_61788


namespace NUMINAMATH_GPT_trout_split_equally_l617_61702

-- Conditions: Nancy and Joan caught 18 trout and split them equally
def total_trout : ℕ := 18
def equal_split (n : ℕ) : ℕ := n / 2

-- Theorem: Prove that if they equally split the trout, each person will get 9 trout.
theorem trout_split_equally : equal_split total_trout = 9 :=
by 
  -- Placeholder for the actual proof
  sorry

end NUMINAMATH_GPT_trout_split_equally_l617_61702


namespace NUMINAMATH_GPT_parallel_lines_slope_l617_61766

theorem parallel_lines_slope (n : ℝ) :
  (∀ x y : ℝ, 2 * x + 2 * y - 5 = 0 → 4 * x + n * y + 1 = 0 → -1 = - (4 / n)) →
  n = 4 :=
by sorry

end NUMINAMATH_GPT_parallel_lines_slope_l617_61766


namespace NUMINAMATH_GPT_find_ratio_l617_61714

variable {R : Type} [LinearOrderedField R]

def f (x a b : R) : R := x^3 + a*x^2 + b*x - a^2 - 7*a

def condition1 (a b : R) : Prop := f 1 a b = 10

def condition2 (a b : R) : Prop :=
  let f' := fun x => 3*x^2 + 2*a*x + b
  f' 1 = 0

theorem find_ratio (a b : R) (h1 : condition1 a b) (h2 : condition2 a b) :
  a / b = -2 / 3 :=
  sorry

end NUMINAMATH_GPT_find_ratio_l617_61714


namespace NUMINAMATH_GPT_product_multiplication_rule_l617_61784

theorem product_multiplication_rule (a : ℝ) : (a * a^3)^2 = a^8 := 
by  
  -- The proof will apply the rule of product multiplication here
  sorry

end NUMINAMATH_GPT_product_multiplication_rule_l617_61784


namespace NUMINAMATH_GPT_students_suggested_pasta_l617_61713

-- Define the conditions as variables in Lean
variable (total_students : ℕ := 470)
variable (suggested_mashed_potatoes : ℕ := 230)
variable (suggested_bacon : ℕ := 140)

-- The problem statement to prove
theorem students_suggested_pasta : 
  total_students - (suggested_mashed_potatoes + suggested_bacon) = 100 := by
  sorry

end NUMINAMATH_GPT_students_suggested_pasta_l617_61713


namespace NUMINAMATH_GPT_fraction_zero_iff_numerator_zero_l617_61746

-- Define the conditions and the result in Lean 4.
theorem fraction_zero_iff_numerator_zero (x : ℝ) (h : x ≠ 0) : (x - 3) / x = 0 ↔ x = 3 :=
by
  sorry

end NUMINAMATH_GPT_fraction_zero_iff_numerator_zero_l617_61746


namespace NUMINAMATH_GPT_bicycle_wheels_l617_61716

theorem bicycle_wheels :
  ∀ (b : ℕ),
  let bicycles := 24
  let tricycles := 14
  let wheels_per_tricycle := 3
  let total_wheels := 90
  ((bicycles * b) + (tricycles * wheels_per_tricycle) = total_wheels) → b = 2 :=
by {
  sorry
}

end NUMINAMATH_GPT_bicycle_wheels_l617_61716


namespace NUMINAMATH_GPT_sum_of_interior_angles_l617_61768

theorem sum_of_interior_angles (n : ℕ) (h : 180 * (n - 2) = 1980) :
    180 * ((n + 3) - 2) = 2520 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_interior_angles_l617_61768


namespace NUMINAMATH_GPT_at_least_one_composite_l617_61752

theorem at_least_one_composite (a b c : ℕ) (h_odd_a : a % 2 = 1) (h_odd_b : b % 2 = 1) (h_odd_c : c % 2 = 1) 
    (h_not_perfect_square : ∀ m : ℕ, m * m ≠ a) : 
    a ^ 2 + a + 1 = 3 * (b ^ 2 + b + 1) * (c ^ 2 + c + 1) →
    (∃ p, p > 1 ∧ p ∣ (b ^ 2 + b + 1)) ∨ (∃ q, q > 1 ∧ q ∣ (c ^ 2 + c + 1)) :=
by sorry

end NUMINAMATH_GPT_at_least_one_composite_l617_61752


namespace NUMINAMATH_GPT_seq_proof_l617_61709
noncomputable def seq1_arithmetic (a1 a2 : ℝ) : Prop :=
  ∃ d : ℝ, a1 = -2 + d ∧ a2 = a1 + d ∧ -8 = a2 + d

noncomputable def seq2_geometric (b1 b2 b3 : ℝ) : Prop :=
  ∃ r : ℝ, b1 = -2 * r ∧ b2 = b1 * r ∧ b3 = b2 * r ∧ -8 = b3 * r

theorem seq_proof (a1 a2 b1 b2 b3: ℝ) (h1 : seq1_arithmetic a1 a2) (h2 : seq2_geometric b1 b2 b3) :
  (a2 - a1) / b2 = 1 / 2 :=
sorry

end NUMINAMATH_GPT_seq_proof_l617_61709


namespace NUMINAMATH_GPT_choose_three_of_nine_l617_61775

def combination (n k : ℕ) := Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem choose_three_of_nine : combination 9 3 = 84 :=
by 
  sorry

end NUMINAMATH_GPT_choose_three_of_nine_l617_61775


namespace NUMINAMATH_GPT_stream_speed_l617_61711

/-- The speed of the stream problem -/
theorem stream_speed 
    (b s : ℝ) 
    (downstream_time : ℝ := 3)
    (upstream_time : ℝ := 3)
    (downstream_distance : ℝ := 60)
    (upstream_distance : ℝ := 30)
    (h1 : downstream_distance = (b + s) * downstream_time)
    (h2 : upstream_distance = (b - s) * upstream_time) : 
    s = 5 := 
by {
  -- The proof can be filled here
  sorry
}

end NUMINAMATH_GPT_stream_speed_l617_61711


namespace NUMINAMATH_GPT_triangleProblem_correct_l617_61725

noncomputable def triangleProblem : Prop :=
  ∃ (a b c A B C : ℝ),
    A = 60 * Real.pi / 180 ∧
    b = 1 ∧
    (1 / 2) * b * c * Real.sin A = Real.sqrt 3 ∧
    Real.cos A = 1 / 2 ∧
    a ^ 2 = b ^ 2 + c ^ 2 - 2 * b * c * Real.cos A ∧
    (a / Real.sin A) = (b / Real.sin B) ∧ (b / Real.sin B) = (c / Real.sin C) ∧
    (a + b + c) / (Real.sin A + Real.sin B + Real.sin C) = 2 * Real.sqrt 39 / 3

theorem triangleProblem_correct : triangleProblem :=
  sorry

end NUMINAMATH_GPT_triangleProblem_correct_l617_61725


namespace NUMINAMATH_GPT_calculate_principal_l617_61791

theorem calculate_principal
  (I : ℝ) (R : ℝ) (T : ℝ) (P : ℝ)
  (hI : I = 8625)
  (hR : R = 50 / 3)
  (hT : T = 3 / 4)
  (hInterest : I = (P * (R / 100) * T)) :
  P = 6900000 := by
  sorry

end NUMINAMATH_GPT_calculate_principal_l617_61791


namespace NUMINAMATH_GPT_sum_of_ages_l617_61765

theorem sum_of_ages (a b : ℕ) :
  let c1 := a
  let c2 := a + 2
  let c3 := a + 4
  let c4 := a + 6
  let coach1 := b
  let coach2 := b + 2
  c1^2 + c2^2 + c3^2 + c4^2 + coach1^2 + coach2^2 = 2796 →
  c1 + c2 + c3 + c4 + coach1 + coach2 = 106 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_sum_of_ages_l617_61765


namespace NUMINAMATH_GPT_average_math_score_of_class_l617_61717

theorem average_math_score_of_class (n : ℕ) (jimin_score jung_score avg_others : ℕ) 
  (h1 : n = 40) 
  (h2 : jimin_score = 98) 
  (h3 : jung_score = 100) 
  (h4 : avg_others = 79) : 
  (38 * avg_others + jimin_score + jung_score) / n = 80 :=
by sorry

end NUMINAMATH_GPT_average_math_score_of_class_l617_61717


namespace NUMINAMATH_GPT_sara_total_cents_l617_61721

def number_of_quarters : ℕ := 11
def value_per_quarter : ℕ := 25

theorem sara_total_cents : number_of_quarters * value_per_quarter = 275 := by
  sorry

end NUMINAMATH_GPT_sara_total_cents_l617_61721


namespace NUMINAMATH_GPT_quadratic_increasing_l617_61730

theorem quadratic_increasing (x : ℝ) (hx : x > 1) : ∃ y : ℝ, y = (x-1)^2 + 1 ∧ ∀ (x₁ x₂ : ℝ), x₁ > x ∧ x₂ > x₁ → (x₁ - 1)^2 + 1 < (x₂ - 1)^2 + 1 := by
  sorry

end NUMINAMATH_GPT_quadratic_increasing_l617_61730


namespace NUMINAMATH_GPT_power_calc_l617_61755

noncomputable def n := 2 ^ 0.3
noncomputable def b := 13.333333333333332

theorem power_calc : n ^ b = 16 := by
  sorry

end NUMINAMATH_GPT_power_calc_l617_61755


namespace NUMINAMATH_GPT_rhombus_area_three_times_diagonals_l617_61763

theorem rhombus_area_three_times_diagonals :
  let d1 := 6
  let d2 := 4
  let new_d1 := 3 * d1
  let new_d2 := 3 * d2
  (new_d1 * new_d2) / 2 = 108 :=
by
  let d1 := 6
  let d2 := 4
  let new_d1 := 3 * d1
  let new_d2 := 3 * d2
  have h : (new_d1 * new_d2) / 2 = 108 := sorry
  exact h

end NUMINAMATH_GPT_rhombus_area_three_times_diagonals_l617_61763


namespace NUMINAMATH_GPT_part_a_part_b_l617_61798

def fake_coin_min_weighings_9 (n : ℕ) : ℕ :=
  if n = 9 then 2 else 0

def fake_coin_min_weighings_27 (n : ℕ) : ℕ :=
  if n = 27 then 3 else 0

theorem part_a : fake_coin_min_weighings_9 9 = 2 := by
  sorry

theorem part_b : fake_coin_min_weighings_27 27 = 3 := by
  sorry

end NUMINAMATH_GPT_part_a_part_b_l617_61798


namespace NUMINAMATH_GPT_sqrt_product_is_four_l617_61745

theorem sqrt_product_is_four : (Real.sqrt 2 * Real.sqrt 8) = 4 := 
by
  sorry

end NUMINAMATH_GPT_sqrt_product_is_four_l617_61745


namespace NUMINAMATH_GPT_glove_selection_correct_l617_61799

-- Define the total number of different pairs of gloves
def num_pairs : Nat := 6

-- Define the required number of gloves to select
def num_gloves_to_select : Nat := 4

-- Define the function to calculate the number of ways to select 4 gloves with exactly one matching pair
noncomputable def count_ways_to_select_gloves (num_pairs : Nat) : Nat :=
  let select_pair := Nat.choose num_pairs 1
  let remaining_gloves := 2 * (num_pairs - 1)
  let select_two_from_remaining := Nat.choose remaining_gloves 2
  let subtract_unwanted_pairs := num_pairs - 1
  select_pair * (select_two_from_remaining - subtract_unwanted_pairs)

-- The correct answer we need to prove
def expected_result : Nat := 240

-- The theorem to prove the number of ways to select the gloves
theorem glove_selection_correct : count_ways_to_select_gloves num_pairs = expected_result :=
  by
    sorry

end NUMINAMATH_GPT_glove_selection_correct_l617_61799


namespace NUMINAMATH_GPT_prime_p_range_l617_61741

open Classical

variable {p : ℤ} (hp_prime : Prime p)

def is_integer_root (a b c : ℤ) := 
  ∃ x y : ℤ, x * y = c ∧ x + y = -b

theorem prime_p_range (hp_roots : is_integer_root 1 p (-500 * p)) : 1 < p ∧ p ≤ 10 :=
by
  sorry

end NUMINAMATH_GPT_prime_p_range_l617_61741


namespace NUMINAMATH_GPT_Mike_monthly_time_is_200_l617_61793

def tv_time (days : Nat) (hours_per_day : Nat) : Nat := days * hours_per_day

def video_game_time (total_tv_time_per_week : Nat) (num_days_playing : Nat) : Nat :=
  (total_tv_time_per_week / 7 / 2) * num_days_playing

def piano_time (weekday_hours : Nat) (weekend_hours : Nat) : Nat :=
  weekday_hours * 5 + weekend_hours * 2

def weekly_time (tv_time : Nat) (video_game_time : Nat) (piano_time : Nat) : Nat :=
  tv_time + video_game_time + piano_time

def monthly_time (weekly_time : Nat) (weeks : Nat) : Nat :=
  weekly_time * weeks

theorem Mike_monthly_time_is_200 : monthly_time
  (weekly_time 
     (tv_time 3 4 + tv_time 2 3 + tv_time 2 5) 
     (video_game_time 28 3) 
     (piano_time 2 3))
  4 = 200 :=
  by
  sorry

end NUMINAMATH_GPT_Mike_monthly_time_is_200_l617_61793


namespace NUMINAMATH_GPT_prime_number_five_greater_than_perfect_square_l617_61760

theorem prime_number_five_greater_than_perfect_square 
(p x : ℤ) (h1 : p - 5 = x^2) (h2 : p + 9 = (x + 1)^2) : 
  p = 41 :=
sorry

end NUMINAMATH_GPT_prime_number_five_greater_than_perfect_square_l617_61760


namespace NUMINAMATH_GPT_value_of_expression_l617_61743

theorem value_of_expression (x : ℝ) (h : x^2 + 3*x + 5 = 7) : x^2 + 3*x - 2 = 0 := 
by {
  -- proof logic will be here
  sorry
}

end NUMINAMATH_GPT_value_of_expression_l617_61743


namespace NUMINAMATH_GPT_ants_square_paths_l617_61783

theorem ants_square_paths (a : ℝ) :
  (∃ a, a = 4 ∧ a + 2 = 6 ∧ a + 4 = 8) →
  (∀ (Mu Ra Vey : ℝ), 
    (Mu = (a + 4) / 2) ∧ 
    (Ra = (a + 2) / 2 + 1) ∧ 
    (Vey = 6) →
    (Mu + Ra + Vey = 2 * (a + 4) + 2)) :=
sorry

end NUMINAMATH_GPT_ants_square_paths_l617_61783


namespace NUMINAMATH_GPT_find_x_l617_61751

-- Define the conditions
def atomic_weight_C : ℝ := 12.01
def atomic_weight_Cl : ℝ := 35.45
def molecular_weight : ℝ := 152

-- State the theorem
theorem find_x : ∃ x : ℕ, molecular_weight = atomic_weight_C + atomic_weight_Cl * x ∧ x = 4 := by
  sorry

end NUMINAMATH_GPT_find_x_l617_61751


namespace NUMINAMATH_GPT_polynomial_simplification_simplify_expression_evaluate_expression_l617_61781

-- Prove that the correct simplification of 6mn - 2m - 3(m + 2mn) results in -5m.
theorem polynomial_simplification (m n : ℤ) :
  6 * m * n - 2 * m - 3 * (m + 2 * m * n) = -5 * m :=
by {
  sorry
}

-- Prove that simplifying a^2b^3 - 1/2(4ab + 6a^2b^3 - 1) + 2(ab - a^2b^3) results in -4a^2b^3 + 1/2.
theorem simplify_expression (a b : ℝ) :
  a^2 * b^3 - 1/2 * (4 * a * b + 6 * a^2 * b^3 - 1) + 2 * (a * b - a^2 * b^3) = -4 * a^2 * b^3 + 1/2 :=
by {
  sorry
}

-- Prove that evaluating the expression -4a^2b^3 + 1/2 at a = 1/2 and b = 3 results in -26.5
theorem evaluate_expression :
  -4 * (1/2) ^ 2 * 3 ^ 3 + 1/2 = -26.5 :=
by {
  sorry
}

end NUMINAMATH_GPT_polynomial_simplification_simplify_expression_evaluate_expression_l617_61781


namespace NUMINAMATH_GPT_prob_at_least_two_diamonds_or_aces_in_three_draws_l617_61778

noncomputable def prob_at_least_two_diamonds_or_aces: ℚ :=
  580 / 2197

def cards_drawn (draws: ℕ) : Prop :=
  draws = 3

def cards_either_diamonds_or_aces: ℕ :=
  16

theorem prob_at_least_two_diamonds_or_aces_in_three_draws:
  cards_drawn 3 →
  cards_either_diamonds_or_aces = 16 →
  prob_at_least_two_diamonds_or_aces = 580 / 2197 :=
  by
  intros
  sorry

end NUMINAMATH_GPT_prob_at_least_two_diamonds_or_aces_in_three_draws_l617_61778


namespace NUMINAMATH_GPT_range_of_phi_l617_61742

theorem range_of_phi (f : ℝ → ℝ) (ω : ℝ) (φ : ℝ) 
  (h1 : ω > 0)
  (h2 : |φ| < (Real.pi / 2))
  (h3 : ∀ x, f x = Real.sin (ω * x + φ))
  (h4 : ∀ x, f (x + (Real.pi / ω)) = f x)
  (h5 : ∀ x y, (x ∈ Set.Ioo (Real.pi / 3) (4 * Real.pi / 5)) ∧
                  (y ∈ Set.Ioo (Real.pi / 3) (4 * Real.pi / 5)) → 
                  (x < y → f x ≤ f y)) :
  (φ ∈ Set.Icc (- Real.pi / 6) (- Real.pi / 10)) :=
by
  sorry

end NUMINAMATH_GPT_range_of_phi_l617_61742


namespace NUMINAMATH_GPT_solve_equation_1_solve_quadratic_equation_2_l617_61753

theorem solve_equation_1 (x : ℝ) : 2 * (x - 1)^2 = 1 - x ↔ x = 1 ∨ x = 1/2 := sorry

theorem solve_quadratic_equation_2 (x : ℝ) :
  4 * x^2 - 2 * (Real.sqrt 3) * x - 1 = 0 ↔
    x = (Real.sqrt 3 + Real.sqrt 7) / 4 ∨ x = (Real.sqrt 3 - Real.sqrt 7) / 4 := sorry

end NUMINAMATH_GPT_solve_equation_1_solve_quadratic_equation_2_l617_61753


namespace NUMINAMATH_GPT_white_patches_count_l617_61787

-- Definitions based on the provided conditions
def total_patches : ℕ := 32
def white_borders_black (x : ℕ) : ℕ := 3 * x
def black_borders_white (x : ℕ) : ℕ := 5 * (total_patches - x)

-- The theorem we need to prove
theorem white_patches_count :
  ∃ x : ℕ, white_borders_black x = black_borders_white x ∧ x = 20 :=
by 
  sorry

end NUMINAMATH_GPT_white_patches_count_l617_61787


namespace NUMINAMATH_GPT_paul_packed_total_toys_l617_61744

def toys_in_box : ℕ := 8
def number_of_boxes : ℕ := 4
def total_toys_packed (toys_in_box number_of_boxes : ℕ) : ℕ := toys_in_box * number_of_boxes

theorem paul_packed_total_toys :
  total_toys_packed toys_in_box number_of_boxes = 32 :=
by
  sorry

end NUMINAMATH_GPT_paul_packed_total_toys_l617_61744


namespace NUMINAMATH_GPT_wendy_made_money_l617_61757

-- Given conditions
def price_per_bar : ℕ := 3
def total_bars : ℕ := 9
def bars_sold : ℕ := total_bars - 3

-- Statement to prove: Wendy made $18
theorem wendy_made_money : bars_sold * price_per_bar = 18 := by
  sorry

end NUMINAMATH_GPT_wendy_made_money_l617_61757


namespace NUMINAMATH_GPT_billy_sleep_total_l617_61764

theorem billy_sleep_total :
  let day1 := 6
  let day2 := day1 + 2
  let day3 := day2 / 2
  let day4 := day3 * 3
  day1 + day2 + day3 + day4 = 30 :=
by
  -- Definitions
  let day1 := 6
  let day2 := day1 + 2
  let day3 := day2 / 2
  let day4 := day3 * 3
  -- Assertion
  have h : day1 + day2 + day3 + day4 = 30 := sorry
  exact h

end NUMINAMATH_GPT_billy_sleep_total_l617_61764


namespace NUMINAMATH_GPT_range_of_t_l617_61762

open Real

noncomputable def f (x : ℝ) : ℝ := x / log x
noncomputable def g (x : ℝ) : ℝ := x / (x^2 - exp 1 * x + exp 1 ^ 2)

theorem range_of_t :
  (∀ x > 1, ∀ t > 0, (t + 1) * g x ≤ t * f x)
  ↔ (∀ t > 0, t ≥ 1 / (exp 1 ^ 2 - 1)) :=
by
  sorry

end NUMINAMATH_GPT_range_of_t_l617_61762


namespace NUMINAMATH_GPT_instantaneous_velocity_at_3_l617_61739

noncomputable def s (t : ℝ) : ℝ := t^2 + 10

theorem instantaneous_velocity_at_3 :
  deriv s 3 = 6 :=
by {
  -- proof goes here
  sorry
}

end NUMINAMATH_GPT_instantaneous_velocity_at_3_l617_61739


namespace NUMINAMATH_GPT_rocco_total_usd_l617_61774

def us_quarters := 4 * 8 * 0.25
def canadian_dimes := 6 * 12 * 0.10 * 0.8
def us_nickels := 9 * 10 * 0.05
def euro_cents := 5 * 15 * 0.01 * 1.18
def british_pence := 3 * 20 * 0.01 * 1.4
def japanese_yen := 2 * 10 * 1 * 0.0091
def mexican_pesos := 4 * 5 * 1 * 0.05

def total_usd := us_quarters + canadian_dimes + us_nickels + euro_cents + british_pence + japanese_yen + mexican_pesos

theorem rocco_total_usd : total_usd = 21.167 := by
  simp [us_quarters, canadian_dimes, us_nickels, euro_cents, british_pence, japanese_yen, mexican_pesos]
  sorry

end NUMINAMATH_GPT_rocco_total_usd_l617_61774


namespace NUMINAMATH_GPT_multiple_of_a_l617_61727

theorem multiple_of_a's_share (A B : ℝ) (x : ℝ) (h₁ : A + B + 260 = 585) (h₂ : x * A = 780) (h₃ : 6 * B = 780) : x = 4 :=
sorry

end NUMINAMATH_GPT_multiple_of_a_l617_61727


namespace NUMINAMATH_GPT_crazy_silly_school_diff_books_movies_l617_61786

theorem crazy_silly_school_diff_books_movies 
    (total_books : ℕ) (total_movies : ℕ)
    (hb : total_books = 36)
    (hm : total_movies = 25) :
    total_books - total_movies = 11 :=
by {
  sorry
}

end NUMINAMATH_GPT_crazy_silly_school_diff_books_movies_l617_61786


namespace NUMINAMATH_GPT_jenny_collects_20_cans_l617_61748

theorem jenny_collects_20_cans (b c : ℕ) (h1 : 6 * b + 2 * c = 100) (h2 : 10 * b + 3 * c = 160) : c = 20 := 
by sorry

end NUMINAMATH_GPT_jenny_collects_20_cans_l617_61748


namespace NUMINAMATH_GPT_baker_cakes_total_l617_61738

-- Define the variables corresponding to the conditions
def cakes_sold : ℕ := 145
def cakes_left : ℕ := 72

-- State the theorem to prove that the total number of cakes made is 217
theorem baker_cakes_total : cakes_sold + cakes_left = 217 := 
by 
-- The proof is omitted according to the instructions
sorry

end NUMINAMATH_GPT_baker_cakes_total_l617_61738


namespace NUMINAMATH_GPT_volume_of_regular_triangular_pyramid_l617_61796

noncomputable def regular_triangular_pyramid_volume (h : ℝ) : ℝ :=
  (h^3 * Real.sqrt 3) / 2

theorem volume_of_regular_triangular_pyramid (h : ℝ) :
  regular_triangular_pyramid_volume h = (h^3 * Real.sqrt 3) / 2 :=
by
  sorry

end NUMINAMATH_GPT_volume_of_regular_triangular_pyramid_l617_61796


namespace NUMINAMATH_GPT_equation_1_solution_set_equation_2_solution_set_l617_61723

open Real

theorem equation_1_solution_set (x : ℝ) : x^2 - 4 * x - 8 = 0 ↔ (x = 2 * sqrt 3 + 2 ∨ x = -2 * sqrt 3 + 2) :=
by sorry

theorem equation_2_solution_set (x : ℝ) : 3 * x - 6 = x * (x - 2) ↔ (x = 2 ∨ x = 3) :=
by sorry

end NUMINAMATH_GPT_equation_1_solution_set_equation_2_solution_set_l617_61723


namespace NUMINAMATH_GPT_Kath_takes_3_friends_l617_61794

theorem Kath_takes_3_friends
  (total_paid: Int)
  (price_before_6: Int)
  (price_reduction: Int)
  (num_family_members: Int)
  (start_time: Int)
  (start_time_condition: start_time < 18)
  (total_payment_condition: total_paid = 30)
  (admission_cost_before_6: price_before_6 = 8 - price_reduction)
  (num_family_members_condition: num_family_members = 3):
  (total_paid / price_before_6 - num_family_members = 3) := 
by
  -- Since no proof is required, simply add sorry to skip the proof
  sorry

end NUMINAMATH_GPT_Kath_takes_3_friends_l617_61794


namespace NUMINAMATH_GPT_page_number_counted_twice_l617_61722

theorem page_number_counted_twice {n x : ℕ} (h₁ : n = 70) (h₂ : x > 0) (h₃ : x ≤ n) (h₄ : 2550 = n * (n + 1) / 2 + x) : x = 65 :=
by {
  sorry
}

end NUMINAMATH_GPT_page_number_counted_twice_l617_61722


namespace NUMINAMATH_GPT_percentage_difference_correct_l617_61754

noncomputable def percentage_difference (initial_price : ℝ) (increase_2012_percent : ℝ) (decrease_2013_percent : ℝ) : ℝ :=
  let price_end_2012 := initial_price * (1 + increase_2012_percent / 100)
  let price_end_2013 := price_end_2012 * (1 - decrease_2013_percent / 100)
  ((price_end_2013 - initial_price) / initial_price) * 100

theorem percentage_difference_correct :
  ∀ (initial_price : ℝ),
  percentage_difference initial_price 25 12 = 10 := 
by
  intros
  sorry

end NUMINAMATH_GPT_percentage_difference_correct_l617_61754


namespace NUMINAMATH_GPT_number_of_monsters_l617_61759

theorem number_of_monsters
    (M S : ℕ)
    (h1 : 4 * M + 3 = S)
    (h2 : 5 * M = S - 6) :
  M = 9 :=
sorry

end NUMINAMATH_GPT_number_of_monsters_l617_61759


namespace NUMINAMATH_GPT_inequality_b_does_not_hold_l617_61732

theorem inequality_b_does_not_hold (a b c d : ℝ) (h1 : a > b) (h2 : c > d) : ¬(a + d > b + c) ↔ a + d ≤ b + c :=
by
  -- We only need the statement, so we add sorry at the end
  sorry

end NUMINAMATH_GPT_inequality_b_does_not_hold_l617_61732


namespace NUMINAMATH_GPT_circle_tangent_y_eq_2_center_on_y_axis_radius_1_l617_61740

theorem circle_tangent_y_eq_2_center_on_y_axis_radius_1 :
  ∃ (y0 : ℝ), (∀ x y : ℝ, (x - 0)^2 + (y - y0)^2 = 1 ↔ y = y0 + 1 ∨ y = y0 - 1) := by
  sorry

end NUMINAMATH_GPT_circle_tangent_y_eq_2_center_on_y_axis_radius_1_l617_61740


namespace NUMINAMATH_GPT_divides_expression_l617_61776

theorem divides_expression (n : ℕ) (h1 : n ≥ 3) 
  (h2 : Prime (4 * n + 1)) : (4 * n + 1) ∣ (n^(2 * n) - 1) :=
by
  sorry

end NUMINAMATH_GPT_divides_expression_l617_61776


namespace NUMINAMATH_GPT_coaches_together_next_l617_61780

theorem coaches_together_next (a b c d : ℕ) (h_a : a = 5) (h_b : b = 9) (h_c : c = 8) (h_d : d = 11) :
  Nat.lcm a (Nat.lcm b (Nat.lcm c d)) = 3960 :=
by 
  rw [h_a, h_b, h_c, h_d]
  sorry

end NUMINAMATH_GPT_coaches_together_next_l617_61780


namespace NUMINAMATH_GPT_value_of_expression_l617_61707

theorem value_of_expression (a b : ℝ) (h : a + b = 3) : 2 * a^2 + 4 * a * b + 2 * b^2 - 6 = 12 :=
by
  sorry

end NUMINAMATH_GPT_value_of_expression_l617_61707


namespace NUMINAMATH_GPT_axis_of_symmetry_values_ge_one_range_m_l617_61718

open Real

-- Definitions for vectors and the function f(x)
noncomputable def a (x : ℝ) : ℝ × ℝ := (sin x, cos x)
noncomputable def b (x : ℝ) : ℝ × ℝ := (sin x, sin x)
noncomputable def f (x : ℝ) : ℝ := (a x).fst * (b x).fst + (a x).snd * (b x).snd

-- Part I: Prove the equation of the axis of symmetry of f(x)
theorem axis_of_symmetry {k : ℤ} : f x = (sqrt 2 / 2) * sin (2 * x - π / 4) + 1 / 2 → 
                                    x = k * π / 2 + 3 * π / 8 := 
sorry

-- Part II: Prove the set of values x for which f(x) ≥ 1
theorem values_ge_one : (f x ≥ 1) ↔ (∃ (k : ℤ), π / 4 + k * π ≤ x ∧ x ≤ π / 2 + k * π) := 
sorry

-- Part III: Prove the range of m given the inequality
theorem range_m (m : ℝ) : (∀ x, π / 6 ≤ x ∧ x ≤ π / 3 → f x - m < 2) → 
                            m > (sqrt 3 - 5) / 4 := 
sorry

end NUMINAMATH_GPT_axis_of_symmetry_values_ge_one_range_m_l617_61718


namespace NUMINAMATH_GPT_solve_for_x_l617_61724

theorem solve_for_x (x : ℝ) (h : 1 / (x + 5) + 1 / (x - 5) = 1 / (x - 5)) : x = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_solve_for_x_l617_61724


namespace NUMINAMATH_GPT_train_crossing_time_l617_61777

noncomputable def train_length : ℕ := 150
noncomputable def bridge_length : ℕ := 150
noncomputable def train_speed_kmph : ℕ := 36

noncomputable def kmph_to_mps (speed_kmph : ℕ) : ℕ :=
  (speed_kmph * 1000) / 3600

noncomputable def train_speed_mps : ℕ := kmph_to_mps train_speed_kmph

noncomputable def total_distance : ℕ := train_length + bridge_length

noncomputable def crossing_time_in_seconds (distance : ℕ) (speed : ℕ) : ℕ :=
  distance / speed

theorem train_crossing_time :
  crossing_time_in_seconds total_distance train_speed_mps = 30 :=
by
  sorry

end NUMINAMATH_GPT_train_crossing_time_l617_61777


namespace NUMINAMATH_GPT_original_number_is_16_l617_61735

theorem original_number_is_16 (x : ℕ) : 213 * x = 3408 → x = 16 :=
by
  sorry

end NUMINAMATH_GPT_original_number_is_16_l617_61735


namespace NUMINAMATH_GPT_karen_start_time_late_l617_61729

theorem karen_start_time_late
  (karen_speed : ℝ := 60) -- Karen drives at 60 mph
  (tom_speed : ℝ := 45) -- Tom drives at 45 mph
  (tom_distance : ℝ := 24) -- Tom drives 24 miles before Karen wins
  (karen_lead : ℝ := 4) -- Karen needs to beat Tom by 4 miles
  : (60 * (24 / 45) - 60 * (28 / 60)) * 60 = 4 := by
  sorry

end NUMINAMATH_GPT_karen_start_time_late_l617_61729


namespace NUMINAMATH_GPT_exactly_two_toads_l617_61767

universe u

structure Amphibian where
  brian : Bool
  julia : Bool
  sean : Bool
  victor : Bool

def are_same_species (x y : Bool) : Bool := x = y

-- Definitions of statements by each amphibian
def Brian_statement (a : Amphibian) : Bool :=
  are_same_species a.brian a.sean

def Julia_statement (a : Amphibian) : Bool :=
  a.victor

def Sean_statement (a : Amphibian) : Bool :=
  ¬ a.julia

def Victor_statement (a : Amphibian) : Bool :=
  (if a.brian then 1 else 0) +
  (if a.julia then 1 else 0) +
  (if a.sean then 1 else 0) +
  (if a.victor then 1 else 0) = 2

-- Conditions translated to Lean definition
def valid_statements (a : Amphibian) : Prop :=
  (a.brian → Brian_statement a) ∧
  (¬ a.brian → ¬ Brian_statement a) ∧
  (a.julia → Julia_statement a) ∧
  (¬ a.julia → ¬ Julia_statement a) ∧
  (a.sean → Sean_statement a) ∧
  (¬ a.sean → ¬ Sean_statement a) ∧
  (a.victor → Victor_statement a) ∧
  (¬ a.victor → ¬ Victor_statement a)

theorem exactly_two_toads (a : Amphibian) (h : valid_statements a) : 
( (if a.brian then 1 else 0) +
  (if a.julia then 1 else 0) +
  (if a.sean then 1 else 0) +
  (if a.victor then 1 else 0) = 2 ) :=
sorry

end NUMINAMATH_GPT_exactly_two_toads_l617_61767


namespace NUMINAMATH_GPT_complex_multiplication_l617_61733

-- Define i such that i^2 = -1
def i : ℂ := Complex.I

theorem complex_multiplication : (3 - 4 * i) * (-7 + 6 * i) = 3 + 46 * i := by
  sorry

end NUMINAMATH_GPT_complex_multiplication_l617_61733


namespace NUMINAMATH_GPT_find_number_l617_61749

theorem find_number (Number : ℝ) (h : Number / 5 = 30 / 600) : Number = 1 / 4 :=
by sorry

end NUMINAMATH_GPT_find_number_l617_61749


namespace NUMINAMATH_GPT_taxi_fare_calculation_l617_61770

def fare_per_km : ℝ := 1.8
def starting_fare : ℝ := 8
def starting_distance : ℝ := 2
def total_distance : ℝ := 12

theorem taxi_fare_calculation : 
  (if total_distance <= starting_distance then starting_fare
   else starting_fare + (total_distance - starting_distance) * fare_per_km) = 26 := by
  sorry

end NUMINAMATH_GPT_taxi_fare_calculation_l617_61770


namespace NUMINAMATH_GPT_necklace_cost_l617_61734

theorem necklace_cost (total_savings earrings_cost remaining_savings: ℕ) 
                      (h1: total_savings = 80) 
                      (h2: earrings_cost = 23) 
                      (h3: remaining_savings = 9) : 
   total_savings - earrings_cost - remaining_savings = 48 :=
by
  sorry

end NUMINAMATH_GPT_necklace_cost_l617_61734


namespace NUMINAMATH_GPT_number_of_squares_l617_61705

open Int

theorem number_of_squares (n : ℕ) (h : n < 10^7) : 
  (∃ n, 36 ∣ n ∧ n^2 < 10^07) ↔ (n = 87) :=
by sorry

end NUMINAMATH_GPT_number_of_squares_l617_61705


namespace NUMINAMATH_GPT_rectangular_solid_surface_area_l617_61790

theorem rectangular_solid_surface_area 
  (a b c : ℝ) 
  (h1 : a + b + c = 14) 
  (h2 : a^2 + b^2 + c^2 = 121) : 
  2 * (a * b + b * c + a * c) = 75 := 
by
  sorry

end NUMINAMATH_GPT_rectangular_solid_surface_area_l617_61790


namespace NUMINAMATH_GPT_greatest_value_of_sum_l617_61736

variable (a b c : ℕ)

theorem greatest_value_of_sum
    (h1 : 2022 < a)
    (h2 : 2022 < b)
    (h3 : 2022 < c)
    (h4 : ∃ k1 : ℕ, a + b = k1 * (c - 2022))
    (h5 : ∃ k2 : ℕ, a + c = k2 * (b - 2022))
    (h6 : ∃ k3 : ℕ, b + c = k3 * (a - 2022)) :
    a + b + c = 2022 * 85 := 
  sorry

end NUMINAMATH_GPT_greatest_value_of_sum_l617_61736


namespace NUMINAMATH_GPT_time_per_room_l617_61761

theorem time_per_room (R P T: ℕ) (h: ℕ) (h₁ : R = 11) (h₂ : P = 2) (h₃ : T = 63) (h₄ : h = T / (R - P)) : h = 7 :=
by
  sorry

end NUMINAMATH_GPT_time_per_room_l617_61761


namespace NUMINAMATH_GPT_speed_down_l617_61731

theorem speed_down {u avg_speed d v : ℝ} (hu : u = 18) (havg : avg_speed = 20.571428571428573) (hv : 2 * d / ((d / u) + (d / v)) = avg_speed) : v = 24 :=
by
  have h1 : 20.571428571428573 = 20.571428571428573 := rfl
  have h2 : 18 = 18 := rfl
  sorry

end NUMINAMATH_GPT_speed_down_l617_61731


namespace NUMINAMATH_GPT_arithmetic_sequence_eighth_term_l617_61703

theorem arithmetic_sequence_eighth_term (a d : ℚ) 
  (h1 : 6 * a + 15 * d = 21) 
  (h2 : a + 6 * d = 8) : 
  a + 7 * d = 9 + 2/7 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_eighth_term_l617_61703


namespace NUMINAMATH_GPT_total_surface_area_correct_l617_61715

-- Defining the dimensions of the rectangular solid
def length := 10
def width := 9
def depth := 6

-- Definition of the total surface area of a rectangular solid
def surface_area (l w d : ℕ) := 2 * (l * w + w * d + l * d)

-- Proposition that the total surface area for the given dimensions is 408 square meters
theorem total_surface_area_correct : surface_area length width depth = 408 := 
by
  sorry

end NUMINAMATH_GPT_total_surface_area_correct_l617_61715


namespace NUMINAMATH_GPT_Paul_sold_350_pencils_l617_61769

-- Variables representing conditions
def pencils_per_day : ℕ := 100
def days_in_week : ℕ := 5
def starting_stock : ℕ := 80
def ending_stock : ℕ := 230

-- The total pencils Paul made in a week
def total_pencils_made : ℕ := pencils_per_day * days_in_week

-- The total pencils before selling any
def total_pencils_before_selling : ℕ := total_pencils_made + starting_stock

-- The number of pencils sold is the difference between total pencils before selling and ending stock
def pencils_sold : ℕ := total_pencils_before_selling - ending_stock

theorem Paul_sold_350_pencils :
  pencils_sold = 350 :=
by {
  -- The proof body is replaced with sorry to indicate a placeholder for the proof.
  sorry
}

end NUMINAMATH_GPT_Paul_sold_350_pencils_l617_61769


namespace NUMINAMATH_GPT_roberta_has_11_3_left_l617_61750

noncomputable def roberta_leftover_money (initial: ℝ) (shoes: ℝ) (bag: ℝ) (lunch: ℝ) (dress: ℝ) (accessory: ℝ) : ℝ :=
  initial - (shoes + bag + lunch + dress + accessory)

theorem roberta_has_11_3_left :
  roberta_leftover_money 158 45 28 (28 / 4) (62 - 0.15 * 62) (2 * (28 / 4)) = 11.3 :=
by
  sorry

end NUMINAMATH_GPT_roberta_has_11_3_left_l617_61750


namespace NUMINAMATH_GPT_fencing_required_l617_61782

def width : ℝ := 25
def area : ℝ := 260
def height_difference : ℝ := 15
def extra_fencing_per_5ft_height : ℝ := 2

noncomputable def length : ℝ := area / width

noncomputable def expected_fencing : ℝ := 2 * length + width + (height_difference / 5) * extra_fencing_per_5ft_height

-- Theorem stating the problem's conclusion
theorem fencing_required : expected_fencing = 51.8 := by
  sorry -- Proof will go here

end NUMINAMATH_GPT_fencing_required_l617_61782


namespace NUMINAMATH_GPT_animal_stickers_l617_61719

theorem animal_stickers {flower stickers total_stickers animal_stickers : ℕ} 
  (h_flower_stickers : flower = 8) 
  (h_total_stickers : total_stickers = 14)
  (h_total_eq : total_stickers = flower + animal_stickers) : 
  animal_stickers = 6 :=
by
  sorry

end NUMINAMATH_GPT_animal_stickers_l617_61719


namespace NUMINAMATH_GPT_no_minus_three_in_range_l617_61720

theorem no_minus_three_in_range (b : ℝ) :
  (∀ x : ℝ, x^2 + b * x + 3 ≠ -3) ↔ b^2 < 24 :=
by
  sorry

end NUMINAMATH_GPT_no_minus_three_in_range_l617_61720


namespace NUMINAMATH_GPT_max_value_of_sum_l617_61747

theorem max_value_of_sum (a c d : ℤ) (b : ℕ) (h1 : a + b = c) (h2 : b + c = d) (h3 : c + d = a) :
  a + b + c + d ≤ -5 := 
sorry

end NUMINAMATH_GPT_max_value_of_sum_l617_61747


namespace NUMINAMATH_GPT_new_average_weight_calculation_l617_61726

noncomputable def new_average_weight (total_weight : ℝ) (number_of_people : ℝ) : ℝ :=
  total_weight / number_of_people

theorem new_average_weight_calculation :
  let initial_people := 6
  let initial_avg_weight := 156
  let new_person_weight := 121
  (initial_people * initial_avg_weight + new_person_weight) / (initial_people + 1) = 151 := by
  sorry

end NUMINAMATH_GPT_new_average_weight_calculation_l617_61726


namespace NUMINAMATH_GPT_leap_year_1996_l617_61789

def divisible_by (n m : ℕ) : Prop := m % n = 0

def is_leap_year (y : ℕ) : Prop :=
  (divisible_by 4 y ∧ ¬divisible_by 100 y) ∨ divisible_by 400 y

theorem leap_year_1996 : is_leap_year 1996 :=
by
  sorry

end NUMINAMATH_GPT_leap_year_1996_l617_61789


namespace NUMINAMATH_GPT_convert_4512_base8_to_base10_l617_61772

-- Definitions based on conditions
def base8_to_base10 (n : Nat) : Nat :=
  let d3 := 4 * 8^3
  let d2 := 5 * 8^2
  let d1 := 1 * 8^1
  let d0 := 2 * 8^0
  d3 + d2 + d1 + d0

-- The proof statement
theorem convert_4512_base8_to_base10 :
  base8_to_base10 4512 = 2378 :=
by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_convert_4512_base8_to_base10_l617_61772


namespace NUMINAMATH_GPT_jake_earnings_per_hour_l617_61779

-- Definitions for conditions
def initialDebt : ℕ := 100
def payment : ℕ := 40
def hoursWorked : ℕ := 4
def remainingDebt : ℕ := initialDebt - payment

-- Theorem stating Jake's earnings per hour
theorem jake_earnings_per_hour : remainingDebt / hoursWorked = 15 := by
  sorry

end NUMINAMATH_GPT_jake_earnings_per_hour_l617_61779


namespace NUMINAMATH_GPT_tan_alpha_neg_four_thirds_cos2alpha_plus_cos_alpha_add_pi_over_2_l617_61704

variable (α : ℝ)
variable (h1 : π / 2 < α)
variable (h2 : α < π)
variable (h3 : Real.sin α = 4 / 5)

theorem tan_alpha_neg_four_thirds (h1 : π / 2 < α) (h2 : α < π) (h3 : Real.sin α = 4 / 5) : Real.tan α = -4 / 3 := 
by sorry

theorem cos2alpha_plus_cos_alpha_add_pi_over_2 (h1 : π / 2 < α) (h2 : α < π) (h3 : Real.sin α = 4 / 5) : 
  Real.cos (2 * α) + Real.cos (α + π / 2) = -27 / 25 := 
by sorry

end NUMINAMATH_GPT_tan_alpha_neg_four_thirds_cos2alpha_plus_cos_alpha_add_pi_over_2_l617_61704


namespace NUMINAMATH_GPT_solution_exists_l617_61706

def operation (a b : ℚ) : ℚ :=
if a ≥ b then a^2 * b else a * b^2

theorem solution_exists (m : ℚ) (h : operation 3 m = 48) : m = 4 := by
  sorry

end NUMINAMATH_GPT_solution_exists_l617_61706


namespace NUMINAMATH_GPT_triangle_third_side_length_l617_61795

theorem triangle_third_side_length
  (A B C : ℝ)
  (a b c : ℝ)
  (h1 : b = 10)
  (h2 : c = 7)
  (h3 : A = 2 * B) :
  a = (50 + 5 * Real.sqrt 2) / 7 ∨ a = (50 - 5 * Real.sqrt 2) / 7 :=
sorry

end NUMINAMATH_GPT_triangle_third_side_length_l617_61795


namespace NUMINAMATH_GPT_jasmine_gives_lola_marbles_l617_61758

theorem jasmine_gives_lola_marbles :
  ∃ (y : ℕ), ∀ (j l : ℕ), 
    j = 120 ∧ l = 15 ∧ 120 - y = 3 * (15 + y) → y = 19 := 
sorry

end NUMINAMATH_GPT_jasmine_gives_lola_marbles_l617_61758


namespace NUMINAMATH_GPT_chessboard_ratio_sum_l617_61785

theorem chessboard_ratio_sum :
  let m := 19
  let n := 135
  m + n = 154 :=
by
  sorry

end NUMINAMATH_GPT_chessboard_ratio_sum_l617_61785


namespace NUMINAMATH_GPT_problem_statement_l617_61771

variable (a b c : ℝ)

theorem problem_statement 
  (h1 : ab / (a + b) = 1 / 3)
  (h2 : bc / (b + c) = 1 / 4)
  (h3 : ca / (c + a) = 1 / 5) :
  abc / (ab + bc + ca) = 1 / 6 := 
sorry

end NUMINAMATH_GPT_problem_statement_l617_61771


namespace NUMINAMATH_GPT_number_of_solutions_l617_61773

theorem number_of_solutions (p : ℕ) (hp : Nat.Prime p) : (∃ n : ℕ, 
  (p % 4 = 1 → n = 11) ∧
  (p = 2 → n = 5) ∧
  (p % 4 = 3 → n = 3)) :=
sorry

end NUMINAMATH_GPT_number_of_solutions_l617_61773


namespace NUMINAMATH_GPT_height_of_pyramid_l617_61700

-- Define the volumes
def volume_cube (s : ℕ) : ℕ := s^3
def volume_pyramid (b : ℕ) (h : ℕ) : ℕ := (b^2 * h) / 3

-- Given constants
def s := 6
def b := 12

-- Given volume equality
def volumes_equal (s : ℕ) (b : ℕ) (h : ℕ) : Prop :=
  volume_cube s = volume_pyramid b h

-- The statement to prove
theorem height_of_pyramid (h : ℕ) (h_eq : volumes_equal s b h) :
  h = 9 := sorry

end NUMINAMATH_GPT_height_of_pyramid_l617_61700


namespace NUMINAMATH_GPT_tammy_investment_change_l617_61737

theorem tammy_investment_change :
  ∀ (initial_investment : ℝ) (loss_percent : ℝ) (gain_percent : ℝ),
    initial_investment = 200 → 
    loss_percent = 0.2 → 
    gain_percent = 0.25 →
    ((initial_investment * (1 - loss_percent)) * (1 + gain_percent)) = initial_investment :=
by
  intros initial_investment loss_percent gain_percent
  sorry

end NUMINAMATH_GPT_tammy_investment_change_l617_61737


namespace NUMINAMATH_GPT_Mary_regular_hourly_rate_l617_61797

theorem Mary_regular_hourly_rate (R : ℝ) (h1 : ∃ max_hours : ℝ, max_hours = 70)
  (h2 : ∀ hours: ℝ, hours ≤ 70 → (hours ≤ 20 → earnings = hours * R) ∧ (hours > 20 → earnings = 20 * R + (hours - 20) * 1.25 * R))
  (h3 : ∀ max_earning: ℝ, max_earning = 660)
  : R = 8 := 
sorry

end NUMINAMATH_GPT_Mary_regular_hourly_rate_l617_61797


namespace NUMINAMATH_GPT_suzanna_history_book_pages_l617_61712

theorem suzanna_history_book_pages (H G M S : ℕ) 
  (h_geography : G = H + 70)
  (h_math : M = (1 / 2) * (H + H + 70))
  (h_science : S = 2 * H)
  (h_total : H + G + M + S = 905) : 
  H = 160 := 
by
  sorry

end NUMINAMATH_GPT_suzanna_history_book_pages_l617_61712


namespace NUMINAMATH_GPT_N_subset_M_values_l617_61728

def M : Set ℝ := { x | 2 * x^2 - 3 * x - 2 = 0 }
def N (a : ℝ) : Set ℝ := { x | a * x = 1 }

theorem N_subset_M_values (a : ℝ) (h : N a ⊆ M) : a = 0 ∨ a = -2 ∨ a = 1/2 := 
by
  sorry

end NUMINAMATH_GPT_N_subset_M_values_l617_61728


namespace NUMINAMATH_GPT_expression_value_l617_61792

theorem expression_value : 23^2 - 21^2 + 19^2 - 17^2 + 15^2 - 13^2 + 11^2 - 9^2 + 7^2 - 5^2 + 3^2 - 1^2 = 288 :=
by
  sorry

end NUMINAMATH_GPT_expression_value_l617_61792


namespace NUMINAMATH_GPT_number_of_other_numbers_l617_61756

-- Definitions of the conditions
def avg_five_numbers (S : ℕ) : Prop := S / 5 = 20
def sum_three_numbers (S2 : ℕ) : Prop := 100 = S2 + 48
def avg_other_numbers (N S2 : ℕ) : Prop := S2 / N = 26

-- Theorem statement
theorem number_of_other_numbers (S S2 N : ℕ) 
  (h1 : avg_five_numbers S) 
  (h2 : sum_three_numbers S2) 
  (h3 : avg_other_numbers N S2) : 
  N = 2 := 
  sorry

end NUMINAMATH_GPT_number_of_other_numbers_l617_61756


namespace NUMINAMATH_GPT_tractors_moved_l617_61710

-- Define initial conditions
def total_area (tractors: ℕ) (days: ℕ) (hectares_per_day: ℕ) := tractors * days * hectares_per_day

theorem tractors_moved (original_tractors remaining_tractors: ℕ)
  (days_original: ℕ) (hectares_per_day_original: ℕ)
  (days_remaining: ℕ) (hectares_per_day_remaining: ℕ)
  (total_area_original: ℕ) 
  (h1: total_area original_tractors days_original hectares_per_day_original = total_area_original)
  (h2: total_area remaining_tractors days_remaining hectares_per_day_remaining = total_area_original) :
  original_tractors - remaining_tractors = 2 :=
by
  sorry

end NUMINAMATH_GPT_tractors_moved_l617_61710


namespace NUMINAMATH_GPT_min_value_inequality_l617_61701

theorem min_value_inequality (a b c : ℝ) 
  (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : a + b + c = 9)
  : (a^2 + b^2 + c^2)/(a + b + c) + (b^2 + c^2)/(b + c) + (c^2 + a^2)/(c + a) + (a^2 + b^2)/(a + b) ≥ 12 :=
by
  sorry

end NUMINAMATH_GPT_min_value_inequality_l617_61701
