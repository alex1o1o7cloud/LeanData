import Mathlib

namespace digit_difference_l333_33372

variable (X Y : ℕ)

theorem digit_difference (h : 10 * X + Y - (10 * Y + X) = 27) : X - Y = 3 :=
by
  sorry

end digit_difference_l333_33372


namespace recurring_to_fraction_l333_33315

theorem recurring_to_fraction : ∀ (x : ℚ), x = 0.5656 ∧ 100 * x = 56.5656 → x = 56 / 99 :=
by
  intros x hx
  sorry

end recurring_to_fraction_l333_33315


namespace eval_power_l333_33316

theorem eval_power (h : 81 = 3^4) : 81^(5/4) = 243 := by
  sorry

end eval_power_l333_33316


namespace number_of_zeros_of_f_l333_33374

noncomputable def f (x : ℝ) : ℝ := Real.log x + 2 * x - 6

theorem number_of_zeros_of_f :
  ∃! x : ℝ, x > 0 ∧ f x = 0 :=
sorry

end number_of_zeros_of_f_l333_33374


namespace grasshopper_catched_in_finite_time_l333_33343

theorem grasshopper_catched_in_finite_time :
  ∀ (x0 y0 x1 y1 : ℤ),
  ∃ (T : ℕ), ∃ (x y : ℤ), 
  ((x = x0 + x1 * T) ∧ (y = y0 + y1 * T)) ∧ -- The hunter will catch the grasshopper at this point
  ((∀ t : ℕ, t ≤ T → (x ≠ x0 + x1 * t ∨ y ≠ y0 + y1 * t) → (x = x0 + x1 * t ∧ y = y0 + y1 * t))) :=
sorry

end grasshopper_catched_in_finite_time_l333_33343


namespace count_valid_b_values_l333_33388

-- Definitions of the inequalities and the condition
def inequality1 (x : ℤ) : Prop := 3 * x > 4 * x - 4
def inequality2 (x b: ℤ) : Prop := 4 * x - b > -8

-- The main statement proving that the count of valid b values is 4
theorem count_valid_b_values (x b : ℤ) (h1 : inequality1 x) (h2 : inequality2 x b) :
  ∃ (b_values : Finset ℤ), 
    ((∀ b' ∈ b_values, ∀ x' : ℤ, inequality2 x' b' → x' ≠ 3) ∧ 
     (∀ b' ∈ b_values, 16 ≤ b' ∧ b' < 20) ∧ 
     b_values.card = 4) := by
  sorry

end count_valid_b_values_l333_33388


namespace expand_array_l333_33393

theorem expand_array (n : ℕ) (h₁ : n ≥ 3) 
  (matrix : Fin (n-2) → Fin n → Fin n)
  (h₂ : ∀ i : Fin (n-2), ∀ j: Fin n, ∀ k: Fin n, j ≠ k → matrix i j ≠ matrix i k)
  (h₃ : ∀ j : Fin n, ∀ k: Fin (n-2), ∀ l: Fin (n-2), k ≠ l → matrix k j ≠ matrix l j) :
  ∃ (expanded_matrix : Fin n → Fin n → Fin n), 
    (∀ i : Fin n, ∀ j: Fin n, ∀ k: Fin n, j ≠ k → expanded_matrix i j ≠ expanded_matrix i k) ∧
    (∀ j : Fin n, ∀ k: Fin n, ∀ l: Fin n, k ≠ l → expanded_matrix k j ≠ expanded_matrix l j) :=
sorry

end expand_array_l333_33393


namespace find_b_l333_33336

theorem find_b (b : ℝ) (h_floor : b + ⌊b⌋ = 22.6) : b = 11.6 :=
sorry

end find_b_l333_33336


namespace set_intersection_correct_l333_33345

def set_A := {x : ℝ | x + 1 > 0}
def set_B := {x : ℝ | x - 3 < 0}
def set_intersection := {x : ℝ | -1 < x ∧ x < 3}

theorem set_intersection_correct : (set_A ∩ set_B) = set_intersection :=
by
  sorry

end set_intersection_correct_l333_33345


namespace marks_of_A_l333_33306

variable (a b c d e : ℕ)

theorem marks_of_A:
  (a + b + c = 144) →
  (a + b + c + d = 188) →
  (e = d + 3) →
  (b + c + d + e = 192) →
  a = 43 := 
by 
  intros h1 h2 h3 h4
  sorry

end marks_of_A_l333_33306


namespace diagonals_in_polygon_of_150_sides_l333_33373

-- Definition of the number of diagonals formula
def number_of_diagonals (n : ℕ) : ℕ :=
  (n * (n - 3)) / 2

-- Given condition: the polygon has 150 sides
def n : ℕ := 150

-- Statement to prove
theorem diagonals_in_polygon_of_150_sides : number_of_diagonals n = 11025 :=
by
  sorry

end diagonals_in_polygon_of_150_sides_l333_33373


namespace robbers_can_divide_loot_equally_l333_33359

theorem robbers_can_divide_loot_equally (coins : List ℕ) (h1 : (coins.sum % 2 = 0)) 
    (h2 : ∀ k, (k % 2 = 1 ∧ 1 ≤ k ∧ k ≤ 2017) → k ∈ coins) :
  ∃ (subset1 subset2 : List ℕ), subset1 ∪ subset2 = coins ∧ subset1.sum = subset2.sum :=
by
  sorry

end robbers_can_divide_loot_equally_l333_33359


namespace coefficient_of_term_free_of_x_l333_33390

theorem coefficient_of_term_free_of_x 
  (n : ℕ) 
  (h1 : ∀ k : ℕ, k ≤ n → n = 10) 
  (h2 : (n.choose 4 / n.choose 2) = 14 / 3) : 
  ∃ (c : ℚ), c = 5 :=
by
  sorry

end coefficient_of_term_free_of_x_l333_33390


namespace gnuff_tutor_minutes_l333_33377

/-- Definitions of the given conditions -/
def flat_rate : ℕ := 20
def per_minute_charge : ℕ := 7
def total_paid : ℕ := 146

/-- The proof statement -/
theorem gnuff_tutor_minutes :
  (total_paid - flat_rate) / per_minute_charge = 18 :=
by
  sorry

end gnuff_tutor_minutes_l333_33377


namespace problem_l333_33353

theorem problem (a b : ℝ) (h1 : a - b = 3) (h2 : a * b = 10) : a^2 + b^2 = 29 :=
by
  sorry

end problem_l333_33353


namespace M_eq_N_l333_33398

def M : Set ℝ := {x | ∃ k : ℤ, x = (7/6) * Real.pi + 2 * k * Real.pi ∨ x = (5/6) * Real.pi + 2 * k * Real.pi}
def N : Set ℝ := {x | ∃ k : ℤ, x = (7/6) * Real.pi + 2 * k * Real.pi ∨ x = -(7/6) * Real.pi + 2 * k * Real.pi}

theorem M_eq_N : M = N := 
by {
  sorry
}

end M_eq_N_l333_33398


namespace four_digit_palindromic_squares_with_different_middle_digits_are_zero_l333_33384

theorem four_digit_palindromic_squares_with_different_middle_digits_are_zero :
  ∀ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ (∃ k, k * k = n) ∧ (∃ a b, n = 1001 * a + 110 * b) → a ≠ b → false :=
by sorry

end four_digit_palindromic_squares_with_different_middle_digits_are_zero_l333_33384


namespace pqr_problem_l333_33367

noncomputable def pqr_abs (p q r : ℝ) (h1 : p ≠ 0) (h2 : q ≠ 0) (h3 : r ≠ 0) (h4 : p ≠ q) (h5 : q ≠ r) (h6 : r ≠ p)
  (h7 : p + (2 / q) = q + (2 / r)) 
  (h8 : q + (2 / r) = r + (2 / p)) : ℝ :=
|p * q * r|

theorem pqr_problem (p q r : ℝ) (h1 : p ≠ 0) (h2 : q ≠ 0) (h3 : r ≠ 0) (h4 : p ≠ q) (h5 : q ≠ r) (h6 : r ≠ p)
  (h7 : p + (2 / q) = q + (2 / r)) 
  (h8 : q + (2 / r) = r + (2 / p)) : pqr_abs p q r h1 h2 h3 h4 h5 h6 h7 h8 = 2 := 
sorry

end pqr_problem_l333_33367


namespace trig_identity_l333_33389

theorem trig_identity (x : ℝ) (h : 3 * Real.sin x + Real.cos x = 0) :
  Real.sin x ^ 2 + 2 * Real.sin x * Real.cos x + Real.cos x ^ 2 = 2 / 5 :=
sorry

end trig_identity_l333_33389


namespace first_group_persons_l333_33368

-- Define the conditions as formal variables
variables (P : ℕ) (hours_per_day_1 days_1 hours_per_day_2 days_2 num_persons_2 : ℕ)

-- Define the conditions from the problem
def first_group_work := P * days_1 * hours_per_day_1
def second_group_work := num_persons_2 * days_2 * hours_per_day_2

-- Set the conditions based on the problem statement
axiom conditions : 
  hours_per_day_1 = 5 ∧ 
  days_1 = 12 ∧ 
  hours_per_day_2 = 6 ∧
  days_2 = 26 ∧
  num_persons_2 = 30 ∧
  first_group_work = second_group_work

-- Statement to prove
theorem first_group_persons : P = 78 :=
by
  -- The proof goes here
  sorry

end first_group_persons_l333_33368


namespace num_impossible_events_l333_33361

def water_boils_at_90C := false
def iron_melts_at_room_temp := false
def coin_flip_results_heads := true
def abs_value_not_less_than_zero := true

theorem num_impossible_events :
  water_boils_at_90C = false ∧ iron_melts_at_room_temp = false ∧ 
  coin_flip_results_heads = true ∧ abs_value_not_less_than_zero = true →
  (if ¬water_boils_at_90C then 1 else 0) + (if ¬iron_melts_at_room_temp then 1 else 0) +
  (if ¬coin_flip_results_heads then 1 else 0) + (if ¬abs_value_not_less_than_zero then 1 else 0) = 2
:= by
  intro h
  have : 
    water_boils_at_90C = false ∧ iron_melts_at_room_temp = false ∧ 
    coin_flip_results_heads = true ∧ abs_value_not_less_than_zero = true := h
  sorry

end num_impossible_events_l333_33361


namespace Grant_made_total_l333_33350

-- Definitions based on the given conditions
def price_cards : ℕ := 25
def price_bat : ℕ := 10
def price_glove_before_discount : ℕ := 30
def glove_discount_rate : ℚ := 0.20
def price_cleats_each : ℕ := 10
def cleats_pairs : ℕ := 2

-- Calculations
def price_glove_after_discount : ℚ := price_glove_before_discount * (1 - glove_discount_rate)
def total_price_cleats : ℕ := price_cleats_each * cleats_pairs
def total_price : ℚ :=
  price_cards + price_bat + total_price_cleats + price_glove_after_discount

-- The statement we need to prove
theorem Grant_made_total :
  total_price = 79 := by sorry

end Grant_made_total_l333_33350


namespace log_function_domain_l333_33339

theorem log_function_domain :
  { x : ℝ | x^2 - 2 * x - 3 > 0 } = { x | x > 3 } ∪ { x | x < -1 } :=
by {
  sorry
}

end log_function_domain_l333_33339


namespace apple_slices_count_l333_33331

theorem apple_slices_count :
  let boxes := 7
  let apples_per_box := 7
  let slices_per_apple := 8
  let total_apples := boxes * apples_per_box
  let total_slices := total_apples * slices_per_apple
  total_slices = 392 :=
by
  sorry

end apple_slices_count_l333_33331


namespace find_constants_exist_l333_33396

theorem find_constants_exist :
  ∃ A B C, (∀ x, 4 * x / ((x - 5) * (x - 3)^2) = A / (x - 5) + B / (x - 3) + C / (x - 3)^2)
  ∧ (A = 5) ∧ (B = -5) ∧ (C = -6) := 
sorry

end find_constants_exist_l333_33396


namespace cycle_selling_price_l333_33386

theorem cycle_selling_price
(C : ℝ := 1900)  -- Cost price of the cycle
(Lp : ℝ := 18)  -- Loss percentage
(S : ℝ := 1558) -- Expected selling price
: (S = C - (Lp / 100) * C) :=
by 
  sorry

end cycle_selling_price_l333_33386


namespace days_to_complete_work_l333_33304

theorem days_to_complete_work :
  ∀ (M B: ℝ) (D: ℝ),
    (M = 2 * B)
    → (13 * M + 24 * B) * 4 = (12 * M + 16 * B) * D
    → D = 5 :=
by
  intros M B D h1 h2
  sorry

end days_to_complete_work_l333_33304


namespace correct_option_is_A_l333_33385

-- Define the options as terms
def optionA (x : ℝ) := (1/2) * x - 5 * x = 18
def optionB (x : ℝ) := (1/2) * x > 5 * x - 1
def optionC (y : ℝ) := 8 * y - 4
def optionD := 5 - 2 = 3

-- Define a function to check if an option is an equation
def is_equation (option : Prop) : Prop :=
  ∃ (x : ℝ), option = ((1/2) * x - 5 * x = 18)

-- Prove that optionA is the equation
theorem correct_option_is_A : is_equation (optionA x) :=
by
  sorry

end correct_option_is_A_l333_33385


namespace advertisement_revenue_l333_33375

theorem advertisement_revenue
  (cost_per_program : ℝ)
  (num_programs : ℕ)
  (selling_price_per_program : ℝ)
  (desired_profit : ℝ)
  (total_cost_production : ℝ)
  (total_revenue_sales : ℝ)
  (total_revenue_needed : ℝ)
  (revenue_from_advertisements : ℝ) :
  cost_per_program = 0.70 →
  num_programs = 35000 →
  selling_price_per_program = 0.50 →
  desired_profit = 8000 →
  total_cost_production = cost_per_program * num_programs →
  total_revenue_sales = selling_price_per_program * num_programs →
  total_revenue_needed = total_cost_production + desired_profit →
  revenue_from_advertisements = total_revenue_needed - total_revenue_sales →
  revenue_from_advertisements = 15000 :=
by
  intros h1 h2 h3 h4 h5 h6 h7 h8
  sorry

end advertisement_revenue_l333_33375


namespace stickers_on_fifth_page_l333_33358

theorem stickers_on_fifth_page :
  ∀ (stickers : ℕ → ℕ),
    stickers 1 = 8 →
    stickers 2 = 16 →
    stickers 3 = 24 →
    stickers 4 = 32 →
    (∀ n, stickers (n + 1) = stickers n + 8) →
    stickers 5 = 40 :=
by
  intros stickers h1 h2 h3 h4 pattern
  apply sorry

end stickers_on_fifth_page_l333_33358


namespace circle_intersection_l333_33370

theorem circle_intersection : 
  ∀ (O : ℝ × ℝ), ∃ (m n : ℤ), (dist (O.1, O.2) (m, n) ≤ 100 + 1/14) := 
sorry

end circle_intersection_l333_33370


namespace quadruple_application_of_h_l333_33351

-- Define the function as specified in the condition
def h (N : ℝ) : ℝ := 0.6 * N + 2

-- State the theorem
theorem quadruple_application_of_h : h (h (h (h 40))) = 9.536 :=
  by
    sorry

end quadruple_application_of_h_l333_33351


namespace probability_of_third_round_expected_value_of_X_variance_of_X_l333_33399

-- Define the probabilities for passing each round
def P_A : ℚ := 2 / 3
def P_B : ℚ := 3 / 4
def P_C : ℚ := 4 / 5

-- Prove the probability of reaching the third round
theorem probability_of_third_round :
  P_A * P_B = 1 / 2 := sorry

-- Define the probability distribution
def P_X (x : ℕ) : ℚ :=
  if x = 1 then 1 / 3 
  else if x = 2 then 1 / 6
  else if x = 3 then 1 / 2
  else 0

-- Expected value
def EX : ℚ := 1 * (1 / 3) + 2 * (1 / 6) + 3 * (1 / 2)

theorem expected_value_of_X :
  EX = 13 / 6 := sorry

-- E(X^2) computation
def EX2 : ℚ := 1^2 * (1 / 3) + 2^2 * (1 / 6) + 3^2 * (1 / 2)

-- Variance
def variance_X : ℚ := EX2 - EX^2

theorem variance_of_X :
  variance_X = 41 / 36 := sorry

end probability_of_third_round_expected_value_of_X_variance_of_X_l333_33399


namespace find_value_of_f2_plus_g3_l333_33338

def f (x : ℝ) : ℝ := 3 * x - 2
def g (x : ℝ) : ℝ := x^2 + 2

theorem find_value_of_f2_plus_g3 : f (2 + g 3) = 37 :=
by
  simp [f, g]
  norm_num
  done

end find_value_of_f2_plus_g3_l333_33338


namespace no_two_perfect_cubes_between_two_perfect_squares_l333_33364

theorem no_two_perfect_cubes_between_two_perfect_squares :
  ∀ n a b : ℤ, n^2 < a^3 ∧ a^3 < b^3 ∧ b^3 < (n + 1)^2 → False :=
by 
  sorry

end no_two_perfect_cubes_between_two_perfect_squares_l333_33364


namespace total_books_l333_33333

theorem total_books (hbooks : ℕ) (fbooks : ℕ) (gbooks : ℕ)
  (Harry_books : hbooks = 50)
  (Flora_books : fbooks = 2 * hbooks)
  (Gary_books : gbooks = hbooks / 2) :
  hbooks + fbooks + gbooks = 175 := by
  sorry

end total_books_l333_33333


namespace number_of_valid_sequences_l333_33308

-- Define the sequence and conditions
def is_valid_sequence (a : Fin 9 → ℝ) : Prop :=
  a 0 = 1 ∧ a 8 = 1 ∧
  ∀ i : Fin 8, (a (i + 1) / a i) ∈ ({2, 1, -1/2} : Set ℝ)

-- The main problem statement
theorem number_of_valid_sequences : ∃ n, n = 491 ∧ ∀ a : Fin 9 → ℝ, is_valid_sequence a ↔ n = 491 := 
sorry

end number_of_valid_sequences_l333_33308


namespace ceil_floor_arith_l333_33320

theorem ceil_floor_arith :
  (Int.ceil (((15: ℚ) / 8)^2 * (-34 / 4)) - Int.floor ((15 / 8) * Int.floor (-34 / 4))) = -12 :=
by sorry

end ceil_floor_arith_l333_33320


namespace no_t_for_xyz_equal_l333_33352

theorem no_t_for_xyz_equal (t : ℝ) (x y z : ℝ) : 
  (x = 1 - 3 * t) → 
  (y = 2 * t - 3) → 
  (z = 4 * t^2 - 5 * t + 1) → 
  ¬ (x = y ∧ y = z) := 
by
  intro h1 h2 h3 h4
  have h5 : t = 4 / 5 := 
    by linarith [h1, h2, h4]
  rw [h5] at h3
  sorry

end no_t_for_xyz_equal_l333_33352


namespace set_intersection_l333_33347

noncomputable def SetA : Set ℝ := {x | Real.sqrt (x - 1) < Real.sqrt 2}
noncomputable def SetB : Set ℝ := {x | x^2 - 6 * x + 8 < 0}

theorem set_intersection :
  SetA ∩ SetB = {x | 2 < x ∧ x < 3} := by
  sorry

end set_intersection_l333_33347


namespace total_books_of_gwen_l333_33337

theorem total_books_of_gwen 
  (mystery_shelves : ℕ) (picture_shelves : ℕ) (books_per_shelf : ℕ)
  (h1 : mystery_shelves = 3) (h2 : picture_shelves = 5) (h3 : books_per_shelf = 9) : 
  mystery_shelves * books_per_shelf + picture_shelves * books_per_shelf = 72 :=
by
  -- Given:
  -- 1. Gwen had 3 shelves of mystery books.
  -- 2. Each shelf had 9 books.
  -- 3. Gwen had 5 shelves of picture books.
  -- 4. Each shelf had 9 books.
  -- Prove:
  -- The total number of books Gwen had is 72.
  sorry

end total_books_of_gwen_l333_33337


namespace probability_of_intersecting_diagonals_l333_33318

def num_vertices := 8
def total_diagonals := (num_vertices * (num_vertices - 3)) / 2
def total_ways_to_choose_two_diagonals := Nat.choose total_diagonals 2
def ways_to_choose_4_vertices := Nat.choose num_vertices 4
def number_of_intersecting_pairs := ways_to_choose_4_vertices
def probability_intersecting_diagonals := (number_of_intersecting_pairs : ℚ) / (total_ways_to_choose_two_diagonals : ℚ)

theorem probability_of_intersecting_diagonals :
  probability_intersecting_diagonals = 7 / 19 := by
  sorry

end probability_of_intersecting_diagonals_l333_33318


namespace incorrect_statement_d_l333_33355

-- Definitions based on the problem's conditions
def is_acute (θ : ℝ) := 0 < θ ∧ θ < 90

def is_complementary (θ₁ θ₂ : ℝ) := θ₁ + θ₂ = 90

def is_supplementary (θ₁ θ₂ : ℝ) := θ₁ + θ₂ = 180

-- Statement D from the problem
def statement_d (θ : ℝ) := is_acute θ → ∀ θc, is_complementary θ θc → θ > θc

-- The theorem we want to prove
theorem incorrect_statement_d : ¬(∀ θ : ℝ, statement_d θ) := 
by sorry

end incorrect_statement_d_l333_33355


namespace rhombus_area_l333_33310

theorem rhombus_area (d1 d2 : ℝ) (h_d1 : d1 = 25) (h_d2 : d2 = 50) :
  (d1 * d2) / 2 = 625 := 
by
  sorry

end rhombus_area_l333_33310


namespace minimum_value_of_a_plus_5b_l333_33334

theorem minimum_value_of_a_plus_5b :
  ∀ (a b : ℝ), a > 0 → b > 0 → (1 / a + 5 / b = 1) → a + 5 * b ≥ 36 :=
by
  sorry

end minimum_value_of_a_plus_5b_l333_33334


namespace shaded_areas_I_and_III_equal_l333_33362

def area_shaded_square_I : ℚ := 1 / 4
def area_shaded_square_II : ℚ := 1 / 2
def area_shaded_square_III : ℚ := 1 / 4

theorem shaded_areas_I_and_III_equal :
  area_shaded_square_I = area_shaded_square_III ∧
   area_shaded_square_I ≠ area_shaded_square_II ∧
   area_shaded_square_III ≠ area_shaded_square_II :=
by {
  sorry
}

end shaded_areas_I_and_III_equal_l333_33362


namespace test_question_count_l333_33394

def total_test_questions 
  (total_points : ℕ) 
  (points_per_2pt : ℕ) 
  (points_per_4pt : ℕ) 
  (num_2pt_questions : ℕ) 
  (num_4pt_questions : ℕ) : Prop :=
  total_points = points_per_2pt * num_2pt_questions + points_per_4pt * num_4pt_questions 

theorem test_question_count 
  (total_points : ℕ) 
  (points_per_2pt : ℕ) 
  (points_per_4pt : ℕ) 
  (num_2pt_questions : ℕ) 
  (correct_total_questions : ℕ) :
  total_test_questions total_points points_per_2pt points_per_4pt num_2pt_questions (correct_total_questions - num_2pt_questions) → correct_total_questions = 40 :=
by
  intros h
  sorry

end test_question_count_l333_33394


namespace eggs_in_each_group_l333_33321

theorem eggs_in_each_group (eggs marbles groups : ℕ) 
  (h_eggs : eggs = 15)
  (h_groups : groups = 3) 
  (h_marbles : marbles = 4) :
  eggs / groups = 5 :=
by sorry

end eggs_in_each_group_l333_33321


namespace min_workers_needed_to_make_profit_l333_33397

def wage_per_worker_per_hour := 20
def fixed_cost := 800
def units_per_worker_per_hour := 6
def price_per_unit := 4.5
def hours_per_workday := 9

theorem min_workers_needed_to_make_profit : ∃ (n : ℕ), 243 * n > 800 + 180 * n ∧ n ≥ 13 :=
by
  sorry

end min_workers_needed_to_make_profit_l333_33397


namespace derek_books_ratio_l333_33349

theorem derek_books_ratio :
  ∃ (T : ℝ), 960 - T - (1/4) * (960 - T) = 360 ∧ T / 960 = 1 / 2 :=
by
  sorry

end derek_books_ratio_l333_33349


namespace inverse_function_value_l333_33369

theorem inverse_function_value (f : ℝ → ℝ) (h : ∀ x, f x = 2 * x^3 + 4) :
  f⁻¹ 58 = 3 :=
by sorry

end inverse_function_value_l333_33369


namespace find_horses_l333_33324

theorem find_horses {x : ℕ} :
  (841 : ℝ) = 8 * (x : ℝ) + 16 * 9 + 18 * 6 → 348 = 16 * 9 →
  x = 73 :=
by
  intros h₁ h₂
  sorry

end find_horses_l333_33324


namespace tim_words_per_day_l333_33366

variable (original_words : ℕ)
variable (years : ℕ)
variable (increase_percent : ℚ)

noncomputable def words_per_day (original_words : ℕ) (years : ℕ) (increase_percent : ℚ) : ℚ :=
  let increase_words := original_words * increase_percent
  let total_days := years * 365
  increase_words / total_days

theorem tim_words_per_day :
    words_per_day 14600 2 (50 / 100) = 10 := by
  sorry

end tim_words_per_day_l333_33366


namespace min_total_cost_of_tank_l333_33378

theorem min_total_cost_of_tank (V D c₁ c₂ : ℝ) (hV : V = 0.18) (hD : D = 0.5)
  (hc₁ : c₁ = 400) (hc₂ : c₂ = 100) : 
  ∃ x : ℝ, x > 0 ∧ (y = c₂*D*(2*x + 0.72/x) + c₁*0.36) ∧ y = 264 := 
sorry

end min_total_cost_of_tank_l333_33378


namespace first_grade_children_count_l333_33381

theorem first_grade_children_count (a : ℕ) (R L : ℕ) :
  200 ≤ a ∧ a ≤ 300 ∧ a = 25 * R + 10 ∧ a = 30 * L - 15 ∧ (R > 0 ∧ L > 0) → a = 285 :=
by
  sorry

end first_grade_children_count_l333_33381


namespace supplements_delivered_l333_33323

-- Define the conditions as given in the problem
def total_medicine_boxes : ℕ := 760
def vitamin_boxes : ℕ := 472

-- Define the number of supplement boxes
def supplement_boxes : ℕ := total_medicine_boxes - vitamin_boxes

-- State the theorem to be proved
theorem supplements_delivered : supplement_boxes = 288 :=
by
  -- The actual proof is not required, so we use "sorry"
  sorry

end supplements_delivered_l333_33323


namespace krishan_money_l333_33311

theorem krishan_money (R G K : ℕ) 
  (h_ratio1 : R * 17 = G * 7) 
  (h_ratio2 : G * 17 = K * 7) 
  (h_R : R = 735) : 
  K = 4335 := 
sorry

end krishan_money_l333_33311


namespace probability_green_light_is_8_over_15_l333_33357

def total_cycle_duration (red yellow green : ℕ) : ℕ :=
  red + yellow + green

def probability_green_light (red yellow green : ℕ) : ℚ :=
  green / (total_cycle_duration red yellow green : ℚ)

theorem probability_green_light_is_8_over_15 :
  probability_green_light 30 5 40 = 8 / 15 := by
  sorry

end probability_green_light_is_8_over_15_l333_33357


namespace condition_is_sufficient_but_not_necessary_l333_33365

variable (P Q : Prop)

theorem condition_is_sufficient_but_not_necessary :
    (P → Q) ∧ ¬(Q → P) :=
sorry

end condition_is_sufficient_but_not_necessary_l333_33365


namespace total_value_of_item_l333_33300

theorem total_value_of_item (V : ℝ) 
  (h1 : 0.07 * (V - 1000) = 109.20) : 
  V = 2560 :=
sorry

end total_value_of_item_l333_33300


namespace solve_missing_figure_l333_33307

theorem solve_missing_figure (x : ℝ) (h : 0.25/100 * x = 0.04) : x = 16 :=
by
  sorry

end solve_missing_figure_l333_33307


namespace paul_sandwiches_in_6_days_l333_33392

def sandwiches_eaten_in_n_days (n : ℕ) : ℕ :=
  let day1 := 2
  let day2 := 2 * day1
  let day3 := 2 * day2
  let three_day_total := day1 + day2 + day3
  three_day_total * (n / 3)

theorem paul_sandwiches_in_6_days : sandwiches_eaten_in_n_days 6 = 28 :=
by
  sorry

end paul_sandwiches_in_6_days_l333_33392


namespace seq_a5_eq_one_ninth_l333_33383

theorem seq_a5_eq_one_ninth (a : ℕ → ℚ) (h1 : a 1 = 1) (h_rec : ∀ n, a (n + 1) = a n / (2 * a n + 1)) :
  a 5 = 1 / 9 :=
sorry

end seq_a5_eq_one_ninth_l333_33383


namespace cost_of_notebooks_and_markers_l333_33314

theorem cost_of_notebooks_and_markers 
  (x y : ℝ) 
  (h1 : 3 * x + 2 * y = 7.30) 
  (h2 : 5 * x + 3 * y = 11.65) : 
  2 * x + y = 4.35 :=
by
  sorry

end cost_of_notebooks_and_markers_l333_33314


namespace find_second_sum_l333_33302

def total_sum : ℝ := 2691
def interest_rate_first : ℝ := 0.03
def interest_rate_second : ℝ := 0.05
def time_first : ℝ := 8
def time_second : ℝ := 3

theorem find_second_sum (x second_sum : ℝ) 
  (H : x + second_sum = total_sum)
  (H_interest : x * interest_rate_first * time_first = second_sum * interest_rate_second * time_second) :
  second_sum = 1656 :=
sorry

end find_second_sum_l333_33302


namespace parabola_properties_and_intersection_l333_33387

-- Definition of the parabola C: y^2 = -4x
def parabola_C (x y : ℝ) : Prop := y^2 = -4 * x

-- Focus of the parabola
def focus_C : ℝ × ℝ := (-1, 0)

-- Equation of the directrix
def directrix_C (x: ℝ): Prop := x = 1

-- Distance from the focus to the directrix
def distance_focus_to_directrix : ℝ := 2

-- Line l passing through P(1, 2) with slope k
def line_l (k x y : ℝ) : Prop := y = k * x - k + 2

-- Main theorem statement
theorem parabola_properties_and_intersection (k: ℝ) :
  (focus_C = (-1, 0)) ∧
  (∀ x, directrix_C x ↔ x = 1) ∧
  (distance_focus_to_directrix = 2) ∧
  ((k = 1 + Real.sqrt 2 ∨ k = 1 - Real.sqrt 2) →
    ∃ x y, parabola_C x y ∧ line_l k x y ∧
    (∀ x' y', parabola_C x' y' ∧ line_l k x' y' → x = x' ∧ y = y')) ∧
  ((1 - Real.sqrt 2 < k ∧ k < 1 + Real.sqrt 2) →
    ∃ x y x' y', x ≠ x' ∧ y ≠ y' ∧
    parabola_C x y ∧ line_l k x y ∧
    parabola_C x' y' ∧ line_l k x' y') ∧
  ((k > 1 + Real.sqrt 2 ∨ k < 1 - Real.sqrt 2) →
    ∀ x y, ¬(parabola_C x y ∧ line_l k x y)) :=
by sorry

end parabola_properties_and_intersection_l333_33387


namespace tripodasaurus_flock_l333_33330

theorem tripodasaurus_flock (num_tripodasauruses : ℕ) (total_head_legs : ℕ) 
  (H1 : ∀ T, total_head_legs = 4 * T)
  (H2 : total_head_legs = 20) :
  num_tripodasauruses = 5 :=
by
  sorry

end tripodasaurus_flock_l333_33330


namespace figure_representation_l333_33328

theorem figure_representation (x y : ℝ) : 
  |x| + |y| ≤ 2 * Real.sqrt (x^2 + y^2) ∧ 2 * Real.sqrt (x^2 + y^2) ≤ 3 * max (|x|) (|y|) → 
  Figure2 :=
sorry

end figure_representation_l333_33328


namespace walkway_time_l333_33342

theorem walkway_time {v_p v_w : ℝ} 
  (cond1 : 60 = (v_p + v_w) * 30) 
  (cond2 : 60 = (v_p - v_w) * 120) 
  : 60 / v_p = 48 := 
by
  sorry

end walkway_time_l333_33342


namespace mark_eggs_supply_l333_33348

theorem mark_eggs_supply 
  (dozens_per_day : ℕ)
  (eggs_per_dozen : ℕ)
  (extra_eggs : ℕ)
  (days_in_week : ℕ)
  (H1 : dozens_per_day = 5)
  (H2 : eggs_per_dozen = 12)
  (H3 : extra_eggs = 30)
  (H4 : days_in_week = 7) :
  (dozens_per_day * eggs_per_dozen + extra_eggs) * days_in_week = 630 :=
by
  sorry

end mark_eggs_supply_l333_33348


namespace surface_area_of_sphere_l333_33346

theorem surface_area_of_sphere (a b c : ℝ) (h1 : a = 1) (h2 : b = 2) (h3 : c = 2)
  (h4 : ∀ d, d = Real.sqrt (a^2 + b^2 + c^2)) : 
  4 * Real.pi * (d / 2)^2 = 9 * Real.pi :=
by
  sorry

end surface_area_of_sphere_l333_33346


namespace sum_xy_sum_inv_squared_geq_nine_four_l333_33326

variable {x y z : ℝ}

theorem sum_xy_sum_inv_squared_geq_nine_four (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (x * y + y * z + z * x) * (1 / (x + y)^2 + 1 / (y + z)^2 + 1 / (z + x)^2) ≥ 9 / 4 :=
by sorry

end sum_xy_sum_inv_squared_geq_nine_four_l333_33326


namespace totalBirdsOnFence_l333_33305

/-
Statement: Given initial birds and additional birds joining, the total number
           of birds sitting on the fence is 10.
Conditions:
1. Initially, there are 4 birds.
2. 6 more birds join them.
3. There are 46 storks on the fence, but they do not affect the number of birds.
-/

def initialBirds : Nat := 4
def additionalBirds : Nat := 6

theorem totalBirdsOnFence : initialBirds + additionalBirds = 10 := by
  sorry

end totalBirdsOnFence_l333_33305


namespace num_factors_of_2_pow_20_minus_1_l333_33332

/-- 
Prove that the number of positive two-digit integers 
that are factors of \(2^{20} - 1\) is 5.
-/
theorem num_factors_of_2_pow_20_minus_1 :
  ∃ (n : ℕ), n = 5 ∧ (∀ (k : ℕ), k ∣ (2^20 - 1) → 10 ≤ k ∧ k < 100 → k = 33 ∨ k = 15 ∨ k = 27 ∨ k = 41 ∨ k = 45) 
  :=
sorry

end num_factors_of_2_pow_20_minus_1_l333_33332


namespace prob_two_sunny_days_l333_33309

-- Define the probability of rain and sunny
def probRain : ℚ := 3 / 4
def probSunny : ℚ := 1 / 4

-- Define the problem statement
theorem prob_two_sunny_days : (10 * (probSunny^2) * (probRain^3)) = 135 / 512 := 
by
  sorry

end prob_two_sunny_days_l333_33309


namespace prime_gt_p_l333_33344

theorem prime_gt_p (p q : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) (hgt : q > 5) (hdiv : q ∣ 2^p + 3^p) : q > p := 
sorry

end prime_gt_p_l333_33344


namespace gcd_4830_3289_l333_33380

theorem gcd_4830_3289 : Nat.gcd 4830 3289 = 23 :=
by sorry

end gcd_4830_3289_l333_33380


namespace some_number_is_105_l333_33325

def find_some_number (a : ℕ) (num : ℕ) : Prop :=
  a ^ 3 = 21 * 25 * num * 7

theorem some_number_is_105 (a : ℕ) (num : ℕ) (h : a = 105) (h_eq : find_some_number a num) : num = 105 :=
by
  sorry

end some_number_is_105_l333_33325


namespace find_value_of_M_l333_33303

theorem find_value_of_M (M : ℝ) (h : 0.2 * M = 0.6 * 1230) : M = 3690 :=
by {
  sorry
}

end find_value_of_M_l333_33303


namespace sum_S10_equals_10_div_21_l333_33376

def a (n : ℕ) : ℚ := 1 / (4 * n^2 - 1)
def S (n : ℕ) : ℚ := (Finset.range n).sum (λ k => a (k + 1))

theorem sum_S10_equals_10_div_21 : S 10 = 10 / 21 :=
by
  sorry

end sum_S10_equals_10_div_21_l333_33376


namespace find_A_l333_33391

theorem find_A (A M C : Nat) (h1 : (10 * A^2 + 10 * M + C) * (A + M^2 + C^2) = 1050) (h2 : A < 10) (h3 : M < 10) (h4 : C < 10) : A = 2 := by
  sorry

end find_A_l333_33391


namespace cakes_count_l333_33319

theorem cakes_count (x y : ℕ) 
  (price_fruit price_chocolate total_cost : ℝ) 
  (avg_price : ℝ) 
  (H1 : price_fruit = 4.8)
  (H2 : price_chocolate = 6.6)
  (H3 : total_cost = 167.4)
  (H4 : avg_price = 6.2)
  (H5 : price_fruit * x + price_chocolate * y = total_cost)
  (H6 : total_cost / (x + y) = avg_price) : 
  x = 6 ∧ y = 21 := 
by
  sorry

end cakes_count_l333_33319


namespace g_expression_l333_33335

def f (x : ℝ) : ℝ := 2 * x + 3

def g (x : ℝ) : ℝ := sorry

theorem g_expression :
  (∀ x : ℝ, g (x + 2) = f x) → ∀ x : ℝ, g x = 2 * x - 1 :=
by
  sorry

end g_expression_l333_33335


namespace solve_equation_l333_33360

theorem solve_equation (x : ℝ) (h_eq : 1 / (x - 2) = 3 / (x - 5)) : 
  x = 1 / 2 :=
  sorry

end solve_equation_l333_33360


namespace correct_reaction_for_phosphoric_acid_l333_33340

-- Define the reactions
def reaction_A := "H₂ + 2OH⁻ - 2e⁻ = 2H₂O"
def reaction_B := "H₂ - 2e⁻ = 2H⁺"
def reaction_C := "O₂ + 4H⁺ + 4e⁻ = 2H₂O"
def reaction_D := "O₂ + 2H₂O + 4e⁻ = 4OH⁻"

-- Define the condition that the electrolyte used is phosphoric acid
def electrolyte := "phosphoric acid"

-- Define the correct reaction
def correct_negative_electrode_reaction := reaction_B

-- Theorem to state that given the conditions above, the correct reaction is B
theorem correct_reaction_for_phosphoric_acid :
  (∃ r, r = reaction_B ∧ electrolyte = "phosphoric acid") :=
by
  sorry

end correct_reaction_for_phosphoric_acid_l333_33340


namespace elena_savings_l333_33354

theorem elena_savings :
  let original_cost := 7 * 3
  let discount_rate := 0.25
  let rebate := 5
  let disc_amount := original_cost * discount_rate
  let price_after_discount := original_cost - disc_amount
  let final_price := price_after_discount - rebate
  original_cost - final_price = 10.25 :=
by
  sorry

end elena_savings_l333_33354


namespace jerry_more_votes_l333_33382

-- Definitions based on conditions
def total_votes : ℕ := 196554
def jerry_votes : ℕ := 108375
def john_votes : ℕ := total_votes - jerry_votes

-- Theorem to prove the number of more votes Jerry received than John Pavich
theorem jerry_more_votes : jerry_votes - john_votes = 20196 :=
by
  -- Definitions and proof can be filled out here as required.
  sorry

end jerry_more_votes_l333_33382


namespace trigonometric_identity_l333_33371

theorem trigonometric_identity (α : ℝ) (h1 : 0 < α) (h2 : α < π) (h3 : Real.tan α = -2) :
  2 * (Real.sin α)^2 - (Real.sin α) * (Real.cos α) + (Real.cos α)^2 = 11 / 5 := by
  sorry

end trigonometric_identity_l333_33371


namespace least_positive_value_tan_inv_k_l333_33313

theorem least_positive_value_tan_inv_k 
  (a b : ℝ) 
  (x : ℝ) 
  (h1 : Real.tan x = a / b) 
  (h2 : Real.tan (2 * x) = 2 * b / (a + 2 * b)) 
  : x = Real.arctan 1 := 
sorry

end least_positive_value_tan_inv_k_l333_33313


namespace grassy_area_percentage_l333_33341

noncomputable def percentage_grassy_area (park_area path1_area path2_area intersection_area : ℝ) : ℝ :=
  let covered_by_paths := path1_area + path2_area - intersection_area
  let grassy_area := park_area - covered_by_paths
  (grassy_area / park_area) * 100

theorem grassy_area_percentage (park_area : ℝ) (path1_area : ℝ) (path2_area : ℝ) (intersection_area : ℝ) 
  (h1 : park_area = 4000) (h2 : path1_area = 400) (h3 : path2_area = 250) (h4 : intersection_area = 25) : 
  percentage_grassy_area park_area path1_area path2_area intersection_area = 84.375 :=
by
  rw [percentage_grassy_area, h1, h2, h3, h4]
  simp
  sorry

end grassy_area_percentage_l333_33341


namespace drawing_probability_consecutive_order_l333_33363

theorem drawing_probability_consecutive_order :
  let total_ways := Nat.factorial 12
  let desired_ways := (1 * Nat.factorial 3 * Nat.factorial 5)
  let probability := desired_ways / total_ways
  probability = 1 / 665280 :=
by
  let total_ways := Nat.factorial 12
  let desired_ways := (1 * Nat.factorial 3 * Nat.factorial 5)
  let probability := desired_ways / total_ways
  sorry

end drawing_probability_consecutive_order_l333_33363


namespace data_plan_comparison_l333_33395

theorem data_plan_comparison : ∃ (m : ℕ), 500 < m :=
by
  let cost_plan_x (m : ℕ) : ℕ := 15 * m
  let cost_plan_y (m : ℕ) : ℕ := 2500 + 10 * m
  use 501
  have h : 500 < 501 := by norm_num
  exact h

end data_plan_comparison_l333_33395


namespace combined_sum_is_115_over_3_l333_33327

def geometric_series_sum (a : ℚ) (r : ℚ) : ℚ :=
  if h : abs r < 1 then a / (1 - r) else 0

def arithmetic_series_sum (a : ℚ) (d : ℚ) (n : ℕ) : ℚ :=
  n * (2 * a + (n - 1) * d) / 2

noncomputable def combined_series_sum : ℚ :=
  let geo_sum := geometric_series_sum 5 (-1/2)
  let arith_sum := arithmetic_series_sum 3 2 5
  geo_sum + arith_sum

theorem combined_sum_is_115_over_3 : combined_series_sum = 115 / 3 := 
  sorry

end combined_sum_is_115_over_3_l333_33327


namespace hotel_room_mistake_l333_33312

theorem hotel_room_mistake (a b c : ℕ) (ha : 1 ≤ a ∧ a ≤ 9) (hb : 0 ≤ b ∧ b ≤ 9) (hc : 0 ≤ c ∧ c ≤ 9) :
  100 * a + 10 * b + c = (a + 1) * (b + 1) * c → false := by sorry

end hotel_room_mistake_l333_33312


namespace scientific_notation_of_0_000000023_l333_33301

theorem scientific_notation_of_0_000000023 : 
  0.000000023 = 2.3 * 10^(-8) :=
sorry

end scientific_notation_of_0_000000023_l333_33301


namespace statement_II_and_IV_true_l333_33317

-- Definitions based on the problem's conditions
def AllNewEditions (P : Type) (books : P → Prop) := ∀ x, books x

-- Condition that the statement "All books in the library are new editions." is false
def NotAllNewEditions (P : Type) (books : P → Prop) := ¬ (AllNewEditions P books)

-- Statements to analyze
def SomeBookNotNewEdition (P : Type) (books : P → Prop) := ∃ x, ¬ books x
def NotAllBooksNewEditions (P : Type) (books : P → Prop) := ∃ x, ¬ books x

-- The theorem to prove
theorem statement_II_and_IV_true 
  (P : Type) 
  (books : P → Prop) 
  (h : NotAllNewEditions P books) : 
  SomeBookNotNewEdition P books ∧ NotAllBooksNewEditions P books :=
  by
    sorry

end statement_II_and_IV_true_l333_33317


namespace Alexei_finished_ahead_of_Sergei_by_1_9_km_l333_33329

noncomputable def race_distance : ℝ := 10
noncomputable def v_A : ℝ := 1  -- speed of Alexei
noncomputable def v_V : ℝ := 0.9 * v_A  -- speed of Vitaly
noncomputable def v_S : ℝ := 0.81 * v_A  -- speed of Sergei

noncomputable def distance_Alexei_finished_Ahead_of_Sergei : ℝ :=
race_distance - (0.81 * race_distance)

theorem Alexei_finished_ahead_of_Sergei_by_1_9_km :
  distance_Alexei_finished_Ahead_of_Sergei = 1.9 :=
by
  simp [race_distance, v_A, v_V, v_S, distance_Alexei_finished_Ahead_of_Sergei]
  sorry

end Alexei_finished_ahead_of_Sergei_by_1_9_km_l333_33329


namespace closest_perfect_square_to_325_is_324_l333_33322

theorem closest_perfect_square_to_325_is_324 :
  ∃ n : ℕ, n^2 = 324 ∧ (∀ m : ℕ, m * m ≠ 325) ∧
    (n = 18 ∧ (∀ k : ℕ, (k*k < 325 ∧ (325 - k*k) > 325 - 324) ∨ 
               (k*k > 325 ∧ (k*k - 325) > 361 - 325))) :=
by
  sorry

end closest_perfect_square_to_325_is_324_l333_33322


namespace circumscribed_sphere_radius_l333_33379

noncomputable def radius_of_circumscribed_sphere (a : ℝ) : ℝ :=
  a * (Real.sqrt (6 + Real.sqrt 20)) / 8

theorem circumscribed_sphere_radius (a : ℝ) :
  radius_of_circumscribed_sphere a = a * (Real.sqrt (6 + Real.sqrt 20)) / 8 :=
by
  sorry

end circumscribed_sphere_radius_l333_33379


namespace magnitude_diff_is_correct_l333_33356

def vector_a : ℝ × ℝ × ℝ := (2, -3, 1)
def vector_b : ℝ × ℝ × ℝ := (-1, 1, -4)

theorem magnitude_diff_is_correct : 
  ‖(2, -3, 1) - (-1, 1, -4)‖ = 5 * Real.sqrt 2 := 
by
  sorry

end magnitude_diff_is_correct_l333_33356
