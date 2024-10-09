import Mathlib

namespace pumpkin_patch_pie_filling_l1394_139414

def pumpkin_cans (small_pumpkins : ℕ) (large_pumpkins : ℕ) (sales : ℕ) (small_price : ℕ) (large_price : ℕ) : ℕ :=
  let remaining_small_pumpkins := small_pumpkins
  let remaining_large_pumpkins := large_pumpkins
  let small_cans := remaining_small_pumpkins / 2
  let large_cans := remaining_large_pumpkins
  small_cans + large_cans

#eval pumpkin_cans 50 33 120 3 5 -- This evaluates the function with the given data to ensure the logic matches the question

theorem pumpkin_patch_pie_filling : pumpkin_cans 50 33 120 3 5 = 58 := by sorry

end pumpkin_patch_pie_filling_l1394_139414


namespace geometric_power_inequality_l1394_139415

theorem geometric_power_inequality {a : ℝ} {n k : ℕ} (h₀ : 1 < a) (h₁ : 0 < n) (h₂ : n < k) :
  (a^n - 1) / n < (a^k - 1) / k :=
sorry

end geometric_power_inequality_l1394_139415


namespace sqrt_expression_eq_neg_one_l1394_139465

theorem sqrt_expression_eq_neg_one : 
  Real.sqrt ((-2)^2) + (Real.sqrt 3)^2 - (Real.sqrt 12 * Real.sqrt 3) = -1 :=
sorry

end sqrt_expression_eq_neg_one_l1394_139465


namespace unique_involution_l1394_139423

noncomputable def f (x : ℤ) : ℤ := sorry

theorem unique_involution (f : ℤ → ℤ) :
  (∀ x : ℤ, f (f x) = x) →
  (∀ x y : ℤ, (x + y) % 2 = 1 → f x + f y ≥ x + y) →
  (∀ x : ℤ, f x = x) :=
sorry

end unique_involution_l1394_139423


namespace shelley_weight_l1394_139476

theorem shelley_weight (p s r : ℕ) (h1 : p + s = 151) (h2 : s + r = 132) (h3 : p + r = 115) : s = 84 := 
  sorry

end shelley_weight_l1394_139476


namespace equality_of_x_and_y_l1394_139432

theorem equality_of_x_and_y (x y : ℕ) (hx : x > 0) (hy : y > 0) (h : x^(y^x) = y^(x^y)) : x = y :=
sorry

end equality_of_x_and_y_l1394_139432


namespace eval_f_function_l1394_139427

noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then x + 1 else if x = 0 then Real.pi else 0

theorem eval_f_function : f (f (f (-1))) = Real.pi + 1 :=
  sorry

end eval_f_function_l1394_139427


namespace not_divisible_by_97_l1394_139497

theorem not_divisible_by_97 (k : ℤ) (h : k ∣ (99^3 - 99)) : k ≠ 97 :=
sorry

end not_divisible_by_97_l1394_139497


namespace proof_solution_l1394_139445

open Real

noncomputable def proof_problem (x : ℝ) : Prop :=
  0 < x ∧ 7.61 * log x / log 2 + 2 * log x / log 4 = x ^ (log 16 / log 3 / log x / log 9)

theorem proof_solution : proof_problem (16 / 3) :=
by
  sorry

end proof_solution_l1394_139445


namespace rectangle_length_l1394_139454

theorem rectangle_length
  (side_length_square : ℝ)
  (width_rectangle : ℝ)
  (area_equiv : side_length_square ^ 2 = width_rectangle * l)
  : l = 24 := by
  sorry

end rectangle_length_l1394_139454


namespace nh4i_required_l1394_139421

theorem nh4i_required (KOH NH4I NH3 KI H2O : ℕ) (h_eq : 1 * NH4I + 1 * KOH = 1 * NH3 + 1 * KI + 1 * H2O)
  (h_KOH : KOH = 3) : NH4I = 3 := 
by
  sorry

end nh4i_required_l1394_139421


namespace no_common_elements_in_sequences_l1394_139411

theorem no_common_elements_in_sequences :
  ∀ (k : ℕ), (∃ n : ℕ, k = n^2 - 1) ∧ (∃ m : ℕ, k = m^2 + 1) → False :=
by sorry

end no_common_elements_in_sequences_l1394_139411


namespace inequality_proof_l1394_139449

theorem inequality_proof (a b x : ℝ) (h : a > b) : a * 2 ^ x > b * 2 ^ x :=
sorry

end inequality_proof_l1394_139449


namespace sugar_inventory_l1394_139402

theorem sugar_inventory :
  ∀ (initial : ℕ) (day2_use : ℕ) (day2_borrow : ℕ) (day3_buy : ℕ) (day4_buy : ℕ) (day5_use : ℕ) (day5_return : ℕ),
  initial = 65 →
  day2_use = 18 →
  day2_borrow = 5 →
  day3_buy = 30 →
  day4_buy = 20 →
  day5_use = 10 →
  day5_return = 3 →
  initial - day2_use - day2_borrow + day3_buy + day4_buy - day5_use + day5_return = 85 :=
by
  intros initial day2_use day2_borrow day3_buy day4_buy day5_use day5_return
  intro h_initial
  intro h_day2_use
  intro h_day2_borrow
  intro h_day3_buy
  intro h_day4_buy
  intro h_day5_use
  intro h_day5_return
  subst h_initial
  subst h_day2_use
  subst h_day2_borrow
  subst h_day3_buy
  subst h_day4_buy
  subst h_day5_use
  subst h_day5_return
  sorry

end sugar_inventory_l1394_139402


namespace road_construction_problem_l1394_139438

theorem road_construction_problem (x : ℝ) (h₁ : x > 0) :
    1200 / x - 1200 / (1.20 * x) = 2 :=
by
  sorry

end road_construction_problem_l1394_139438


namespace french_students_l1394_139467

theorem french_students 
  (T : ℕ) (G : ℕ) (B : ℕ) (N : ℕ) (F : ℕ)
  (hT : T = 78) (hG : G = 22) (hB : B = 9) (hN : N = 24)
  (h_eq : F + G - B = T - N) :
  F = 41 :=
by
  sorry

end french_students_l1394_139467


namespace inequality_transformation_incorrect_l1394_139471

theorem inequality_transformation_incorrect (a b : ℝ) (h : a > b) : (3 - a > 3 - b) -> false :=
by
  intros h1
  simp at h1
  sorry

end inequality_transformation_incorrect_l1394_139471


namespace initial_price_of_iphone_l1394_139478

variable (P : ℝ)

def initial_price_conditions : Prop :=
  (P > 0) ∧ (0.72 * P = 720)

theorem initial_price_of_iphone (h : initial_price_conditions P) : P = 1000 :=
by
  sorry

end initial_price_of_iphone_l1394_139478


namespace shelter_cats_l1394_139409

theorem shelter_cats (initial_dogs initial_cats additional_cats : ℕ) 
  (h1 : initial_dogs = 75)
  (h2 : initial_dogs * 7 = initial_cats * 15)
  (h3 : initial_dogs * 11 = 15 * (initial_cats + additional_cats)) : 
  additional_cats = 20 :=
by
  sorry

end shelter_cats_l1394_139409


namespace total_people_l1394_139486

-- Given definitions
def students : ℕ := 37500
def ratio_students_professors : ℕ := 15
def professors : ℕ := students / ratio_students_professors

-- The statement to prove
theorem total_people : students + professors = 40000 := by
  sorry

end total_people_l1394_139486


namespace track_length_l1394_139447

theorem track_length (x : ℕ) 
  (diametrically_opposite : ∃ a b : ℕ, a + b = x)
  (first_meeting : ∃ b : ℕ, b = 100)
  (second_meeting : ∃ s s' : ℕ, s = 150 ∧ s' = (x / 2 - 100 + s))
  (constant_speed : ∀ t₁ t₂ : ℕ, t₁ / t₂ = 100 / (x / 2 - 100)) :
  x = 400 := 
by sorry

end track_length_l1394_139447


namespace max_students_gave_away_balls_more_l1394_139458

theorem max_students_gave_away_balls_more (N : ℕ) (hN : N ≤ 13) : 
  ∃(students : ℕ), students = 27 ∧ (students = 27 ∧ N ≤ students - N) :=
by
  sorry

end max_students_gave_away_balls_more_l1394_139458


namespace monotonicity_of_f_solve_inequality_range_of_m_l1394_139482

variable {f : ℝ → ℝ}
variable {a b m : ℝ}

-- Given conditions
def is_odd_function (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x
def in_interval (x : ℝ) : Prop := -1 ≤ x ∧ x ≤ 1
def f_at_one (f : ℝ → ℝ) : Prop := f 1 = 1
def positivity_condition (f : ℝ → ℝ) (a b : ℝ) : Prop := (a + b ≠ 0) → ((f a + f b) / (a + b) > 0)

-- Proof problems
theorem monotonicity_of_f 
  (h_odd : is_odd_function f) 
  (h_interval : ∀ x, in_interval x → in_interval (f x)) 
  (h_f_one : f_at_one f) 
  (h_pos : ∀ a b, in_interval a → in_interval b → positivity_condition f a b) :
  ∀ x1 x2, in_interval x1 → in_interval x2 → x1 < x2 → f x1 < f x2 :=
sorry

theorem solve_inequality 
  (h_odd : is_odd_function f) 
  (h_interval : ∀ x, in_interval x → in_interval (f x)) 
  (h_f_increasing : ∀ x1 x2, in_interval x1 → in_interval x2 → x1 < x2 → f x1 < f x2) 
  (h_f_one : f_at_one f) 
  (h_pos : ∀ a b, in_interval a → in_interval b → positivity_condition f a b) :
  ∀ x, in_interval (x + 1/2) → in_interval (1 / (x - 1)) → f (x + 1/2) < f (1 / (x - 1)) → -3/2 ≤ x ∧ x < -1 :=
sorry

theorem range_of_m 
  (h_odd : is_odd_function f) 
  (h_interval : ∀ x, in_interval x → in_interval (f x)) 
  (h_f_one : f_at_one f) 
  (h_pos : ∀ a b, in_interval a → in_interval b → positivity_condition f a b) 
  (h_f_increasing : ∀ x1 x2, in_interval x1 → in_interval x2 → x1 < x2 → f x1 < f x2) :
  (∀ a, in_interval a → f a ≤ m^2 - 2 * a * m + 1) → (m = 0 ∨ m ≤ -2 ∨ m ≥ 2) :=
sorry

end monotonicity_of_f_solve_inequality_range_of_m_l1394_139482


namespace compute_value_l1394_139487

def diamond_op (x y : ℕ) : ℕ := 3 * x + 5 * y
def heart_op (z x : ℕ) : ℕ := 4 * z + 2 * x

theorem compute_value : heart_op (diamond_op 4 3) 8 = 124 := by
  sorry

end compute_value_l1394_139487


namespace triangle_side_length_l1394_139473

theorem triangle_side_length (a b c : ℝ)
  (h1 : 1/2 * a * c * (Real.sin (60 * Real.pi / 180)) = Real.sqrt 3)
  (h2 : a^2 + c^2 = 3 * a * c) :
  b = 2 * Real.sqrt 2 :=
by
  sorry

end triangle_side_length_l1394_139473


namespace max_blocks_fit_l1394_139494

-- Define the dimensions of the block
def block_length : ℕ := 3
def block_width : ℕ := 1
def block_height : ℕ := 1

-- Define the dimensions of the box
def box_length : ℕ := 5
def box_width : ℕ := 3
def box_height : ℕ := 2

-- Theorem stating the maximum number of blocks that can fit in the box
theorem max_blocks_fit :
  (box_length * box_width * box_height) / (block_length * block_width * block_height) = 15 := sorry

end max_blocks_fit_l1394_139494


namespace james_spends_90_dollars_per_week_l1394_139495

structure PistachioPurchasing where
  can_cost : ℕ  -- cost in dollars per can
  can_weight : ℕ -- weight in ounces per can
  consumption_oz_per_5days : ℕ -- consumption in ounces per 5 days

def cost_per_week (p : PistachioPurchasing) : ℕ :=
  let daily_consumption := p.consumption_oz_per_5days / 5
  let weekly_consumption := daily_consumption * 7
  let cans_needed := (weekly_consumption + p.can_weight - 1) / p.can_weight -- round up
  cans_needed * p.can_cost

theorem james_spends_90_dollars_per_week :
  cost_per_week ⟨10, 5, 30⟩ = 90 :=
by
  sorry

end james_spends_90_dollars_per_week_l1394_139495


namespace rectangles_on_8x8_chessboard_rectangles_on_nxn_chessboard_l1394_139416

theorem rectangles_on_8x8_chessboard : 
  (Nat.choose 9 2) * (Nat.choose 9 2) = 1296 := by
  sorry

theorem rectangles_on_nxn_chessboard (n : ℕ) : 
  (Nat.choose (n + 1) 2) * (Nat.choose (n + 1) 2) = (n * (n + 1) / 2) * (n * (n + 1) / 2) := by 
  sorry

end rectangles_on_8x8_chessboard_rectangles_on_nxn_chessboard_l1394_139416


namespace find_number_l1394_139418

theorem find_number (N : ℝ) (h : 0.015 * N = 90) : N = 6000 :=
  sorry

end find_number_l1394_139418


namespace lines_intersect_and_find_point_l1394_139446

theorem lines_intersect_and_find_point (n : ℝ)
  (h₁ : ∀ t : ℝ, ∃ (x y z : ℝ), x / 2 = t ∧ y / -3 = t ∧ z / n = t)
  (h₂ : ∀ t : ℝ, ∃ (x y z : ℝ), (x + 1) / 3 = t ∧ (y + 5) / 2 = t ∧ z / 1 = t) :
  n = 1 ∧ (∃ (x y z : ℝ), x = 2 ∧ y = -3 ∧ z = 1) :=
sorry

end lines_intersect_and_find_point_l1394_139446


namespace dan_spent_more_on_chocolates_l1394_139450

def price_candy_bar : ℝ := 4
def number_of_candy_bars : ℕ := 5
def candy_discount : ℝ := 0.20
def discount_threshold : ℕ := 3
def price_chocolate : ℝ := 6
def number_of_chocolates : ℕ := 4
def chocolate_tax_rate : ℝ := 0.05

def candy_cost_total : ℝ :=
  let cost_without_discount := number_of_candy_bars * price_candy_bar
  if number_of_candy_bars >= discount_threshold
  then cost_without_discount * (1 - candy_discount)
  else cost_without_discount

def chocolate_cost_total : ℝ :=
  let cost_without_tax := number_of_chocolates * price_chocolate
  cost_without_tax * (1 + chocolate_tax_rate)

def difference_in_spending : ℝ :=
  chocolate_cost_total - candy_cost_total

theorem dan_spent_more_on_chocolates :
  difference_in_spending = 9.20 :=
by
  sorry

end dan_spent_more_on_chocolates_l1394_139450


namespace fractions_zero_condition_l1394_139425

variable {a b c : ℝ}

theorem fractions_zero_condition 
  (h : (a - b) / (1 + a * b) + (b - c) / (1 + b * c) + (c - a) / (1 + c * a) = 0) :
  (a - b) / (1 + a * b) = 0 ∨ (b - c) / (1 + b * c) = 0 ∨ (c - a) / (1 + c * a) = 0 := 
sorry

end fractions_zero_condition_l1394_139425


namespace lynne_total_spent_l1394_139431

theorem lynne_total_spent :
  let books_about_cats := 7
  let books_about_solar := 2
  let magazines := 3
  let cost_per_book := 7
  let cost_per_magazine := 4
  let total_books := books_about_cats + books_about_solar
  let total_cost_books := total_books * cost_per_book
  let total_cost_magazines := magazines * cost_per_magazine
  let total_spent := total_cost_books + total_cost_magazines
  total_spent = 75 :=
by
  -- Definitions
  let books_about_cats := 7
  let books_about_solar := 2
  let magazines := 3
  let cost_per_book := 7
  let cost_per_magazine := 4
  let total_books := books_about_cats + books_about_solar
  let total_cost_books := total_books * cost_per_book
  let total_cost_magazines := magazines * cost_per_magazine
  let total_spent := total_cost_books + total_cost_magazines
  -- Conclusion
  have h1 : total_books = 9 := by sorry
  have h2 : total_cost_books = 63 := by sorry
  have h3 : total_cost_magazines = 12 := by sorry
  have h4 : total_spent = 75 := by sorry
  exact h4


end lynne_total_spent_l1394_139431


namespace time_fraction_l1394_139479

variable (t₅ t₁₅ : ℝ)

def total_distance (t₅ t₁₅ : ℝ) : ℝ :=
  5 * t₅ + 15 * t₁₅

def total_time (t₅ t₁₅ : ℝ) : ℝ :=
  t₅ + t₁₅

def average_speed_eq (t₅ t₁₅ : ℝ) : Prop :=
  10 * (t₅ + t₁₅) = 5 * t₅ + 15 * t₁₅

theorem time_fraction (t₅ t₁₅ : ℝ) (h : average_speed_eq t₅ t₁₅) :
  (t₁₅ / (t₅ + t₁₅)) = 1 / 2 := by
  sorry

end time_fraction_l1394_139479


namespace value_of_q_l1394_139448

theorem value_of_q (p q : ℝ) (h1 : 1 < p) (h2 : p < q) (h3 : 1/p + 1/q = 1) (h4 : p * q = 12) : q = 6 + 2 * Real.sqrt 6 :=
by
  sorry

end value_of_q_l1394_139448


namespace value_of_expression_l1394_139417

theorem value_of_expression :
  (43 + 15)^2 - (43^2 + 15^2) = 2 * 43 * 15 :=
by
  sorry

end value_of_expression_l1394_139417


namespace find_side_length_of_cube_l1394_139408

theorem find_side_length_of_cube (n : ℕ) :
  (4 * n^2 = (1/3) * 6 * n^3) -> n = 2 :=
by
  sorry

end find_side_length_of_cube_l1394_139408


namespace range_of_b2_plus_c2_l1394_139442

theorem range_of_b2_plus_c2 (A B C : ℝ) (a b c : ℝ) 
  (h1 : (a - b) * (Real.sin A + Real.sin B) = (c - b) * Real.sin C)
  (ha : a = Real.sqrt 3)
  (hAcute : 0 < A ∧ A < π / 2 ∧ 0 < B ∧ B < π / 2 ∧ 0 < C ∧ C < π / 2 ∧ A + B + C = π) :
  (∃ x, 5 < x ∧ x ≤ 6 ∧ x = b^2 + c^2) :=
sorry

end range_of_b2_plus_c2_l1394_139442


namespace calculate_expression_l1394_139498

theorem calculate_expression :
  (121^2 - 110^2 + 11) / 10 = 255.2 := 
sorry

end calculate_expression_l1394_139498


namespace time_after_increment_l1394_139433

-- Define the current time in minutes
def current_time_minutes : ℕ := 15 * 60  -- 3:00 p.m. in minutes

-- Define the time increment in minutes
def time_increment : ℕ := 1567

-- Calculate the total time in minutes after the increment
def total_time_minutes : ℕ := current_time_minutes + time_increment

-- Convert total time back to hours and minutes
def calculated_hours : ℕ := total_time_minutes / 60
def calculated_minutes : ℕ := total_time_minutes % 60

-- The expected hours and minutes after the increment
def expected_hours : ℕ := 17 -- 17:00 hours which is 5:00 p.m.
def expected_minutes : ℕ := 7 -- 7 minutes

theorem time_after_increment :
  (calculated_hours - 24 * (calculated_hours / 24) = expected_hours) ∧ (calculated_minutes = expected_minutes) :=
by
  sorry

end time_after_increment_l1394_139433


namespace general_term_formula_l1394_139437

def seq (n : ℕ) : ℤ :=
  match n with
  | 1     => 2
  | 2     => -6
  | 3     => 12
  | 4     => -20
  | 5     => 30
  | 6     => -42
  | _     => 0 -- We match only the first few elements as given

theorem general_term_formula (n : ℕ) :
  seq n = (-1)^(n+1) * n * (n + 1) := by
  sorry

end general_term_formula_l1394_139437


namespace inequality_solution_l1394_139439

theorem inequality_solution (x : ℝ) : x + 1 < (4 + 3 * x) / 2 → x > -2 :=
by
  intros h
  sorry

end inequality_solution_l1394_139439


namespace rectangular_prism_sum_of_dimensions_l1394_139499

theorem rectangular_prism_sum_of_dimensions (a b c : ℕ) (h_volume : a * b * c = 21) 
(h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c) (h_positive : 0 < a ∧ 0 < b ∧ 0 < c) : 
a + b + c = 11 :=
sorry

end rectangular_prism_sum_of_dimensions_l1394_139499


namespace cube_root_of_neg8_l1394_139426

-- Define the condition
def is_cube_root (x : ℝ) : Prop := x^3 = -8

-- State the problem to be proved.
theorem cube_root_of_neg8 : is_cube_root (-2) :=
by 
  sorry

end cube_root_of_neg8_l1394_139426


namespace terminating_decimal_contains_digit_3_l1394_139435

theorem terminating_decimal_contains_digit_3 :
  ∃ n : ℕ, n > 0 ∧ (∃ a b : ℕ, n = 2 ^ a * 5 ^ b) ∧ (∃ d, n = d * 10 ^ 0 + 3) ∧ n = 32 :=
by sorry

end terminating_decimal_contains_digit_3_l1394_139435


namespace initial_coloring_books_l1394_139475

theorem initial_coloring_books
  (x : ℝ)
  (h1 : x - 20 = 80 / 4) :
  x = 40 :=
by
  sorry

end initial_coloring_books_l1394_139475


namespace remainder_when_divided_by_29_l1394_139477

theorem remainder_when_divided_by_29 (k N : ℤ) (h : N = 761 * k + 173) : N % 29 = 28 :=
by
  sorry

end remainder_when_divided_by_29_l1394_139477


namespace tan_alpha_fraction_eq_five_sevenths_l1394_139413

theorem tan_alpha_fraction_eq_five_sevenths (α : ℝ) (h : Real.tan α = 3) : 
  (4 * Real.sin α - 2 * Real.cos α) / (5 * Real.cos α + 3 * Real.sin α) = 5 / 7 :=
sorry

end tan_alpha_fraction_eq_five_sevenths_l1394_139413


namespace batsman_average_after_17th_l1394_139461

theorem batsman_average_after_17th (A : ℤ) (h1 : 86 + 16 * A = 17 * (A + 3)) : A + 3 = 38 :=
by
  sorry

end batsman_average_after_17th_l1394_139461


namespace barry_sotter_length_increase_l1394_139451

theorem barry_sotter_length_increase (n : ℕ) : (n + 3) / 3 = 50 → n = 147 :=
by
  intro h
  sorry

end barry_sotter_length_increase_l1394_139451


namespace max_value_of_expression_l1394_139404

theorem max_value_of_expression 
  (x y : ℝ)
  (h : x^2 + y^2 = 20 * x + 9 * y + 9) :
  ∃ x y : ℝ, 4 * x + 3 * y = 83 := sorry

end max_value_of_expression_l1394_139404


namespace perpendicular_vectors_l1394_139403

-- Define the vectors a and b.
def vector_a : ℝ × ℝ := (1, -2)
def vector_b (x : ℝ) : ℝ × ℝ := (-2, x)

-- Define the dot product function.
def dot_product (v1 v2 : ℝ × ℝ) : ℝ :=
  v1.1 * v2.1 + v1.2 * v2.2

-- The condition that a is perpendicular to b.
def perp_condition (x : ℝ) : Prop :=
  dot_product vector_a (vector_b x) = 0

-- Main theorem stating that if a is perpendicular to b, then x = -1.
theorem perpendicular_vectors (x : ℝ) (h : perp_condition x) : x = -1 :=
by sorry

end perpendicular_vectors_l1394_139403


namespace profit_divided_equally_l1394_139469

noncomputable def Mary_investment : ℝ := 800
noncomputable def Mike_investment : ℝ := 200
noncomputable def total_profit : ℝ := 2999.9999999999995
noncomputable def Mary_extra : ℝ := 1200

theorem profit_divided_equally (E : ℝ) : 
  (E / 2 + 4 / 5 * (total_profit - E)) - (E / 2 + 1 / 5 * (total_profit - E)) = Mary_extra →
  E = 1000 :=
  by sorry

end profit_divided_equally_l1394_139469


namespace teachers_quit_before_lunch_percentage_l1394_139480

variables (n_initial n_after_one_hour n_after_lunch n_quit_before_lunch : ℕ)

def initial_teachers : ℕ := 60
def teachers_after_one_hour (n_initial : ℕ) : ℕ := n_initial / 2
def teachers_after_lunch : ℕ := 21
def quit_before_lunch (n_after_one_hour n_after_lunch : ℕ) : ℕ := n_after_one_hour - n_after_lunch
def percentage_quit (n_quit_before_lunch n_after_one_hour : ℕ) : ℕ := (n_quit_before_lunch * 100) / n_after_one_hour

theorem teachers_quit_before_lunch_percentage :
  ∀ n_initial n_after_one_hour n_after_lunch n_quit_before_lunch,
  n_initial = initial_teachers →
  n_after_one_hour = teachers_after_one_hour n_initial →
  n_after_lunch = teachers_after_lunch →
  n_quit_before_lunch = quit_before_lunch n_after_one_hour n_after_lunch →
  percentage_quit n_quit_before_lunch n_after_one_hour = 30 := by 
    sorry

end teachers_quit_before_lunch_percentage_l1394_139480


namespace no_solution_for_vectors_l1394_139463

theorem no_solution_for_vectors {t s k : ℝ} :
  (∃ t s : ℝ, (1 + 6 * t = -1 + 3 * s) ∧ (3 + 1 * t = 4 + k * s)) ↔ k ≠ 0.5 :=
sorry

end no_solution_for_vectors_l1394_139463


namespace rectangular_box_inscribed_in_sphere_l1394_139466

noncomputable def problem_statement : Prop :=
  ∃ (a b c s : ℝ), (4 * (a + b + c) = 72) ∧ (2 * (a * b + b * c + c * a) = 216) ∧
  (a^2 + b^2 + c^2 = 108) ∧ (4 * s^2 = 108) ∧ (s = 3 * Real.sqrt 3)

theorem rectangular_box_inscribed_in_sphere : problem_statement := 
  sorry

end rectangular_box_inscribed_in_sphere_l1394_139466


namespace count_arrangements_california_l1394_139429

-- Defining the counts of letters in "CALIFORNIA"
def word_length : ℕ := 10
def count_A : ℕ := 3
def count_I : ℕ := 2
def count_C : ℕ := 1
def count_L : ℕ := 1
def count_F : ℕ := 1
def count_O : ℕ := 1
def count_R : ℕ := 1
def count_N : ℕ := 1

-- The final proof statement to show the number of unique arrangements
theorem count_arrangements_california : 
  (Nat.factorial word_length) / 
  ((Nat.factorial count_A) * (Nat.factorial count_I)) = 302400 := by
  -- Placeholder for the proof, can be filled in later by providing the actual steps
  sorry

end count_arrangements_california_l1394_139429


namespace birds_landed_l1394_139452

theorem birds_landed (original_birds total_birds : ℕ) (h : original_birds = 12) (h2 : total_birds = 20) :
  total_birds - original_birds = 8 :=
by {
  sorry
}

end birds_landed_l1394_139452


namespace maximum_value_expression_l1394_139462

theorem maximum_value_expression (a b c : ℝ) 
  (h1 : 0 ≤ a) (h2 : 0 ≤ b) (h3 : 0 ≤ c) (h_sum : a + b + c = 3) : 
  (a^2 - a * b + b^2) * (a^2 - a * c + c^2) * (b^2 - b * c + c^2) ≤ 1 :=
sorry

end maximum_value_expression_l1394_139462


namespace ratio_of_height_to_width_l1394_139436

-- Define variables
variable (W H L V : ℕ)
variable (x : ℝ)

-- Given conditions
def condition_1 := W = 3
def condition_2 := H = x * W
def condition_3 := L = 7 * H
def condition_4 := V = 6804

-- Prove that the ratio of height to width is 6√3
theorem ratio_of_height_to_width : (W = 3 ∧ H = x * W ∧ L = 7 * H ∧ V = 6804 ∧ V = W * H * L) → x = 6 * Real.sqrt 3 :=
by
  sorry

end ratio_of_height_to_width_l1394_139436


namespace johnnys_hourly_wage_l1394_139470

def totalEarnings : ℝ := 26
def totalHours : ℝ := 8
def hourlyWage : ℝ := 3.25

theorem johnnys_hourly_wage : totalEarnings / totalHours = hourlyWage :=
by
  sorry

end johnnys_hourly_wage_l1394_139470


namespace problem1_problem2_l1394_139410

namespace MathProblem

-- Problem 1
theorem problem1 : (π - 2)^0 + (-1)^3 = 0 := by
  sorry

-- Problem 2
variable (m n : ℤ)

theorem problem2 : (3 * m + n) * (m - 2 * n) = 3 * m ^ 2 - 5 * m * n - 2 * n ^ 2 := by
  sorry

end MathProblem

end problem1_problem2_l1394_139410


namespace arithmetic_geometric_sequence_l1394_139455

theorem arithmetic_geometric_sequence (d : ℤ) (a_1 a_2 a_5 : ℤ)
  (h1 : d ≠ 0)
  (h2 : a_2 = a_1 + d)
  (h3 : a_5 = a_1 + 4 * d)
  (h4 : a_2 ^ 2 = a_1 * a_5) :
  a_5 = 9 * a_1 := 
sorry

end arithmetic_geometric_sequence_l1394_139455


namespace largest_angle_in_triangle_l1394_139405

theorem largest_angle_in_triangle (x : ℝ) (h1 : 40 + 60 + x = 180) (h2 : max 40 60 ≤ x) : x = 80 :=
by
  -- Proof skipped
  sorry

end largest_angle_in_triangle_l1394_139405


namespace cream_ratio_l1394_139457

def joe_ends_with_cream (start_coffee : ℕ) (drank_coffee : ℕ) (added_cream : ℕ) : ℕ :=
  added_cream

def joann_cream_left (start_coffee : ℕ) (added_cream : ℕ) (drank_mix : ℕ) : ℚ :=
  added_cream - drank_mix * (added_cream / (start_coffee + added_cream))

theorem cream_ratio (start_coffee : ℕ) (joe_drinks : ℕ) (joe_adds : ℕ)
                    (joann_adds : ℕ) (joann_drinks : ℕ) :
  joe_ends_with_cream start_coffee joe_drinks joe_adds / 
  joann_cream_left start_coffee joann_adds joann_drinks = (9 : ℚ) / (7 : ℚ) :=
by
  sorry

end cream_ratio_l1394_139457


namespace largest_integer_of_five_with_product_12_l1394_139412

theorem largest_integer_of_five_with_product_12 (a b c d e : ℤ) (h : a * b * c * d * e = 12) (h_diff : a ≠ b ∧ b ≠ c ∧ c ≠ d ∧ d ≠ e ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ b ≠ d ∧ b ≠ e ∧ c ≠ e) : 
  max a (max b (max c (max d e))) = 3 :=
sorry

end largest_integer_of_five_with_product_12_l1394_139412


namespace find_f_neg_2_l1394_139453

theorem find_f_neg_2 (f : ℝ → ℝ) (b x : ℝ) (h1 : ∀ x, f (-x) = -f x)
  (h2 : ∀ x, x ≥ 0 → f x = x^2 - 3*x + b) (h3 : f 0 = 0) : f (-2) = 2 := by
sorry

end find_f_neg_2_l1394_139453


namespace find_largest_number_l1394_139428

theorem find_largest_number
  (a b c d e : ℝ) 
  (distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ c ≠ d ∧ c ≠ e ∧ d ≠ e)
  (h₁ : a + b = 32)
  (h₂ : a + c = 36)
  (h₃ : b + c = 37)
  (h₄ : c + e = 48)
  (h₅ : d + e = 51) :
  (max a (max b (max c (max d e)))) = 27.5 :=
sorry

end find_largest_number_l1394_139428


namespace number_of_quarters_l1394_139460

-- Defining constants for the problem
def value_dime : ℝ := 0.10
def value_nickel : ℝ := 0.05
def value_penny : ℝ := 0.01
def value_quarter : ℝ := 0.25

-- Given conditions
def total_dimes : ℝ := 3
def total_nickels : ℝ := 4
def total_pennies : ℝ := 200
def total_amount : ℝ := 5.00

-- Theorem stating the number of quarters found
theorem number_of_quarters :
  (total_amount - (total_dimes * value_dime + total_nickels * value_nickel + total_pennies * value_penny)) / value_quarter = 10 :=
by
  sorry

end number_of_quarters_l1394_139460


namespace ticket_cost_correct_l1394_139496

noncomputable def calculate_ticket_cost : ℝ :=
  let x : ℝ := 5  -- price of an adult ticket
  let child_price := x / 2  -- price of a child ticket
  let senior_price := 0.75 * x  -- price of a senior ticket
  10 * x + 8 * child_price + 5 * senior_price

theorem ticket_cost_correct :
  let x : ℝ := 5  -- price of an adult ticket
  let child_price := x / 2  -- price of a child ticket
  let senior_price := 0.75 * x  -- price of a senior ticket
  (4 * x + 3 * child_price + 2 * senior_price = 35) →
  (10 * x + 8 * child_price + 5 * senior_price = 88.75) :=
by
  intros
  sorry

end ticket_cost_correct_l1394_139496


namespace distinct_license_plates_l1394_139444

theorem distinct_license_plates :
  let digit_choices := 10
  let letter_choices := 26
  let positions := 7
  let total := positions * (digit_choices ^ 6) * (letter_choices ^ 3)
  total = 122504000 :=
by
  -- Definitions from the conditions
  let digit_choices := 10
  let letter_choices := 26
  let positions := 7
  -- Calculation
  let total := positions * (digit_choices ^ 6) * (letter_choices ^ 3)
  -- Assertion
  have h : total = 122504000 := sorry
  exact h

end distinct_license_plates_l1394_139444


namespace truck_capacity_solution_l1394_139492

variable (x y : ℝ)

theorem truck_capacity_solution (h1 : 3 * x + 4 * y = 22) (h2 : 2 * x + 6 * y = 23) :
  x + y = 6.5 := sorry

end truck_capacity_solution_l1394_139492


namespace total_cost_is_correct_l1394_139420

def cost_shirt (S : ℝ) : Prop := S = 12
def cost_shoes (Sh S : ℝ) : Prop := Sh = S + 5
def cost_dress (D : ℝ) : Prop := D = 25
def discount_shoes (Sh Sh' : ℝ) : Prop := Sh' = Sh - 0.10 * Sh
def discount_dress (D D' : ℝ) : Prop := D' = D - 0.05 * D
def cost_bag (B twoS Sh' D' : ℝ) : Prop := B = (twoS + Sh' + D') / 2
def total_cost_before_tax (T_before twoS Sh' D' B : ℝ) : Prop := T_before = twoS + Sh' + D' + B
def sales_tax (tax T_before : ℝ) : Prop := tax = 0.07 * T_before
def total_cost_including_tax (T_total T_before tax : ℝ) : Prop := T_total = T_before + tax
def convert_to_usd (T_usd T_total : ℝ) : Prop := T_usd = T_total * 1.18

theorem total_cost_is_correct (S Sh D Sh' D' twoS B T_before tax T_total T_usd : ℝ) :
  cost_shirt S →
  cost_shoes Sh S →
  cost_dress D →
  discount_shoes Sh Sh' →
  discount_dress D D' →
  twoS = 2 * S →
  cost_bag B twoS Sh' D' →
  total_cost_before_tax T_before twoS Sh' D' B →
  sales_tax tax T_before →
  total_cost_including_tax T_total T_before tax →
  convert_to_usd T_usd T_total →
  T_usd = 119.42 :=
by
  sorry

end total_cost_is_correct_l1394_139420


namespace point_on_y_axis_l1394_139489

theorem point_on_y_axis (x y : ℝ) (h : x = 0 ∧ y = -1) : y = -1 := by
  -- Using the conditions directly
  cases h with
  | intro hx hy =>
    -- The proof would typically follow, but we include sorry to complete the statement
    sorry

end point_on_y_axis_l1394_139489


namespace parking_space_area_l1394_139468

theorem parking_space_area (L W : ℕ) (h1 : L = 9) (h2 : L + 2 * W = 37) : L * W = 126 := by
  sorry

end parking_space_area_l1394_139468


namespace fraction_numerator_l1394_139440

theorem fraction_numerator (x : ℚ) :
  (∃ n : ℚ, 4 * n - 4 = x ∧ x / (4 * n - 4) = 3 / 7) → x = 12 / 5 :=
by
  sorry

end fraction_numerator_l1394_139440


namespace greater_than_neg2_by_1_l1394_139424

theorem greater_than_neg2_by_1 : -2 + 1 = -1 := by
  sorry

end greater_than_neg2_by_1_l1394_139424


namespace incorrect_expression_l1394_139485

theorem incorrect_expression : 
  ∀ (x y : ℚ), (x / y = 2 / 5) → (x + 3 * y) / x ≠ 17 / 2 :=
by
  intros x y h
  sorry

end incorrect_expression_l1394_139485


namespace water_bottle_capacity_l1394_139484

theorem water_bottle_capacity :
  (20 * 250 + 13 * 600) / 1000 = 12.8 := 
by
  sorry

end water_bottle_capacity_l1394_139484


namespace second_oldest_brother_age_l1394_139422

theorem second_oldest_brother_age
  (y s o : ℕ)
  (h1 : y + s + o = 34)
  (h2 : o = 3 * y)
  (h3 : s = 2 * y - 2) :
  s = 10 := by
  sorry

end second_oldest_brother_age_l1394_139422


namespace most_frequent_third_number_l1394_139441

def is_lottery_condition (e1 e2 e3 e4 e5 : ℕ) : Prop :=
  1 ≤ e1 ∧ e1 < e2 ∧ e2 < e3 ∧ e3 < e4 ∧ e4 < e5 ∧ e5 ≤ 90 ∧ (e1 + e2 = e3)

theorem most_frequent_third_number :
  ∃ h : ℕ, 3 ≤ h ∧ h ≤ 88 ∧ (∀ h', (h' = 31 → ¬ (31 < h')) ∧ 
        ∀ e1 e2 e3 e4 e5, is_lottery_condition e1 e2 e3 e4 e5 → e3 = h) :=
sorry

end most_frequent_third_number_l1394_139441


namespace proof_problem_l1394_139459

-- Definitions of the conditions
def odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) + f x = 0

def decreasing_on (f : ℝ → ℝ) (I : Set ℝ) : Prop :=
  ∀ x y : ℝ, x ∈ I → y ∈ I → x < y → f y < f x

def satisfies_neq_point (f : ℝ → ℝ) (a : ℝ) : Prop :=
  f a = 0

-- Main problem statement to prove (with conditions)
theorem proof_problem (f : ℝ → ℝ)
  (Hodd : odd_function f)
  (Hdec : decreasing_on f {y | 0 < y})
  (Hpt : satisfies_neq_point f (-2)) :
  {x : ℝ | (x - 1) * f x > 0} = {x : ℝ | -2 < x ∧ x < 0} ∪ {x : ℝ | 1 < x ∧ x < 2} :=
sorry

end proof_problem_l1394_139459


namespace speed_of_ferry_P_l1394_139481

variable (v_P v_Q : ℝ)

noncomputable def condition1 : Prop := v_Q = v_P + 4
noncomputable def condition2 : Prop := (6 * v_P) / v_Q = 4
noncomputable def condition3 : Prop := 2 + 2 = 4

theorem speed_of_ferry_P
    (h1 : condition1 v_P v_Q)
    (h2 : condition2 v_P v_Q)
    (h3 : condition3) :
    v_P = 8 := 
by 
    sorry

end speed_of_ferry_P_l1394_139481


namespace predicted_customers_on_Saturday_l1394_139483

theorem predicted_customers_on_Saturday 
  (breakfast_customers : ℕ)
  (lunch_customers : ℕ)
  (dinner_customers : ℕ)
  (prediction_factor : ℕ)
  (h1 : breakfast_customers = 73)
  (h2 : lunch_customers = 127)
  (h3 : dinner_customers = 87)
  (h4 : prediction_factor = 2) :
  prediction_factor * (breakfast_customers + lunch_customers + dinner_customers) = 574 :=  
by 
  sorry 

end predicted_customers_on_Saturday_l1394_139483


namespace Susan_total_peaches_l1394_139472

-- Define the number of peaches in the knapsack
def peaches_in_knapsack : ℕ := 12

-- Define the condition that the number of peaches in the knapsack is half the number of peaches in each cloth bag
def peaches_per_cloth_bag (x : ℕ) : Prop := peaches_in_knapsack * 2 = x

-- Define the total number of peaches Susan bought
def total_peaches (x : ℕ) : ℕ := x + 2 * x

-- Theorem statement: Prove that the total number of peaches Susan bought is 60
theorem Susan_total_peaches (x : ℕ) (h : peaches_per_cloth_bag x) : total_peaches peaches_in_knapsack = 60 := by
  sorry

end Susan_total_peaches_l1394_139472


namespace a_values_condition_l1394_139488

def is_subset (A B : Set ℝ) : Prop := ∀ x, x ∈ A → x ∈ B

theorem a_values_condition (a : ℝ) : 
  (2 * a + 1 ≤ 3 ∧ 3 * a - 5 ≤ 22 ∧ 2 * a + 1 ≤ 3 * a - 5) 
  ↔ (6 ≤ a ∧ a ≤ 9) :=
by 
  sorry

end a_values_condition_l1394_139488


namespace number_of_ways_to_choose_roles_l1394_139464

-- Define the problem setup
def friends := Fin 6
def cooks (maria : Fin 1) := {f : Fin 6 | f ≠ maria}
def cleaners (cooks : Fin 6 → Prop) := {f : Fin 6 | ¬cooks f}

-- The number of ways to select one additional cook from the remaining friends
def chooseSecondCook : ℕ := Nat.choose 5 1  -- 5 ways

-- The number of ways to select two cleaners from the remaining friends
def chooseCleaners : ℕ := Nat.choose 4 2  -- 6 ways

-- The final number of ways to choose roles
theorem number_of_ways_to_choose_roles (maria : Fin 1) : 
  let total_ways : ℕ := chooseSecondCook * chooseCleaners
  total_ways = 30 := sorry

end number_of_ways_to_choose_roles_l1394_139464


namespace largest_number_of_cakes_l1394_139434

theorem largest_number_of_cakes : ∃ (c : ℕ), c = 65 :=
by
  sorry

end largest_number_of_cakes_l1394_139434


namespace speed_limit_correct_l1394_139491

def speed_limit (distance : ℕ) (time : ℕ) (over_limit : ℕ) : ℕ :=
  let speed := distance / time
  speed - over_limit

theorem speed_limit_correct :
  speed_limit 60 1 10 = 50 :=
by
  sorry

end speed_limit_correct_l1394_139491


namespace feasible_measures_l1394_139474

-- Conditions for the problem
def condition1 := "Replace iron filings with iron pieces"
def condition2 := "Use excess zinc pieces instead of iron pieces"
def condition3 := "Add a small amount of CuSO₄ solution to the dilute hydrochloric acid"
def condition4 := "Add CH₃COONa solid to the dilute hydrochloric acid"
def condition5 := "Add sulfuric acid of the same molar concentration to the dilute hydrochloric acid"
def condition6 := "Add potassium sulfate solution to the dilute hydrochloric acid"
def condition7 := "Slightly heat (without considering the volatilization of HCl)"
def condition8 := "Add NaNO₃ solid to the dilute hydrochloric acid"

-- The criteria for the problem
def isFeasible (cond : String) : Prop :=
  cond = condition1 ∨ cond = condition2 ∨ cond = condition3 ∨ cond = condition7

theorem feasible_measures :
  ∀ cond, 
  cond ≠ condition4 →
  cond ≠ condition5 →
  cond ≠ condition6 →
  cond ≠ condition8 →
  isFeasible cond :=
by
  intros
  sorry

end feasible_measures_l1394_139474


namespace January_to_November_ratio_l1394_139430

variable (N D J : ℝ)

-- Condition 1: November revenue is 3/5 of December revenue
axiom revenue_Nov : N = (3 / 5) * D

-- Condition 2: December revenue is 2.5 times the average of November and January revenues
axiom revenue_Dec : D = 2.5 * (N + J) / 2

-- Goal: Prove the ratio of January revenue to November revenue is 1/3
theorem January_to_November_ratio : J / N = 1 / 3 :=
by
  -- We will use the given axioms to derive the proof
  sorry

end January_to_November_ratio_l1394_139430


namespace probability_of_x_gt_8y_l1394_139419

noncomputable def probability_x_gt_8y : ℚ :=
  let rect_area := 2020 * 2030
  let tri_area := (2020 * (2020 / 8)) / 2
  tri_area / rect_area

theorem probability_of_x_gt_8y :
  probability_x_gt_8y = 255025 / 4100600 := by
  sorry

end probability_of_x_gt_8y_l1394_139419


namespace exists_c_d_rel_prime_l1394_139406

theorem exists_c_d_rel_prime (a b : ℤ) :
  ∃ c d : ℤ, ∀ n : ℤ, gcd (a * n + c) (b * n + d) = 1 :=
sorry

end exists_c_d_rel_prime_l1394_139406


namespace part1_simplified_part2_value_part3_independent_l1394_139401

-- Definitions of A and B
def A (x y : ℝ) : ℝ := 3 * x^2 - x + 2 * y - 4 * x * y
def B (x y : ℝ) : ℝ := 2 * x^2 - 3 * x - y + x * y

-- Proof statement for part 1
theorem part1_simplified (x y : ℝ) :
  2 * A x y - 3 * B x y = 7*x + 7*y - 11*x*y :=
by sorry

-- Proof statement for part 2
theorem part2_value (x y : ℝ) (hxy : x + y = 6/7) (hprod : x * y = -1) :
  2 * A x y - 3 * B x y = 17 :=
by sorry

-- Proof statement for part 3
theorem part3_independent (y : ℝ) :
  2 * A (7/11) y - 3 * B (7/11) y = 49/11 :=
by sorry

end part1_simplified_part2_value_part3_independent_l1394_139401


namespace corrected_mean_l1394_139400

theorem corrected_mean :
  let original_mean := 45
  let num_observations := 100
  let observations_wrong := [32, 12, 25]
  let observations_correct := [67, 52, 85]
  let original_total_sum := original_mean * num_observations
  let incorrect_sum := observations_wrong.sum
  let correct_sum := observations_correct.sum
  let adjustment := correct_sum - incorrect_sum
  let corrected_total_sum := original_total_sum + adjustment
  let corrected_new_mean := corrected_total_sum / num_observations
  corrected_new_mean = 46.35 := 
by
  sorry

end corrected_mean_l1394_139400


namespace manufacturers_price_l1394_139493

theorem manufacturers_price (M : ℝ) 
  (h1 : 0.1 ≤ 0.3) 
  (h2 : 0.2 = 0.2) 
  (h3 : 0.56 * M = 25.2) : 
  M = 45 := 
sorry

end manufacturers_price_l1394_139493


namespace cone_to_sphere_surface_area_ratio_l1394_139456

noncomputable def sphere_radius (r : ℝ) := r
noncomputable def cone_height (r : ℝ) := 3 * r
noncomputable def side_length_of_triangle (r : ℝ) := 2 * Real.sqrt 3 * r
noncomputable def surface_area_of_sphere (r : ℝ) := 4 * Real.pi * r^2
noncomputable def surface_area_of_cone (r : ℝ) := 9 * Real.pi * r^2
noncomputable def ratio_of_areas (cone_surface : ℝ) (sphere_surface : ℝ) := cone_surface / sphere_surface

theorem cone_to_sphere_surface_area_ratio (r : ℝ) :
    ratio_of_areas (surface_area_of_cone r) (surface_area_of_sphere r) = 9 / 4 := sorry

end cone_to_sphere_surface_area_ratio_l1394_139456


namespace train_length_is_correct_l1394_139490

variable (speed_km_hr : ℕ) (time_sec : ℕ)
def convert_speed (speed_km_hr : ℕ) : ℚ :=
  (speed_km_hr * 1000 : ℚ) / 3600

noncomputable def length_of_train (speed_km_hr time_sec : ℕ) : ℚ :=
  convert_speed speed_km_hr * time_sec

theorem train_length_is_correct (speed_km_hr : ℕ) (time_sec : ℕ) (h₁ : speed_km_hr = 300) (h₂ : time_sec = 33) :
  length_of_train speed_km_hr time_sec = 2750 := by
  sorry

end train_length_is_correct_l1394_139490


namespace point_G_six_l1394_139443

theorem point_G_six : 
  ∃ (A B C D E F G : ℕ), 
    1 ≤ A ∧ A ≤ 10 ∧
    1 ≤ B ∧ B ≤ 10 ∧
    1 ≤ C ∧ C ≤ 10 ∧
    1 ≤ D ∧ D ≤ 10 ∧
    1 ≤ E ∧ E ≤ 10 ∧
    1 ≤ F ∧ F ≤ 10 ∧
    1 ≤ G ∧ G ≤ 10 ∧
    (A + B = A + C + D) ∧ 
    (A + B = B + E + F) ∧
    (A + B = C + F + G) ∧
    (A + B = D + E + G) ∧ 
    (A + B = 12) →
    G = 6 := 
by
  sorry

end point_G_six_l1394_139443


namespace binary_arithmetic_l1394_139407

theorem binary_arithmetic :
    let a := 0b1011101
    let b := 0b1101
    let c := 0b101010
    let d := 0b110
    ((a + b) * c) / d = 0b1110111100 :=
by
  sorry

end binary_arithmetic_l1394_139407
