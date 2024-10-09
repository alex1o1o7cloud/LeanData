import Mathlib

namespace wire_cut_min_area_l598_59878

theorem wire_cut_min_area :
  ∃ x : ℝ, 0 < x ∧ x < 100 ∧ S = π * (x / (2 * π))^2 + ((100 - x) / 4)^2 ∧ 
  (∀ y : ℝ, 0 < y ∧ y < 100 → (π * (y / (2 * π))^2 + ((100 - y) / 4)^2 ≥ S)) ∧
  x = 100 * π / (16 + π) :=
sorry

end wire_cut_min_area_l598_59878


namespace no_sol_for_eq_xn_minus_yn_eq_2k_l598_59894

theorem no_sol_for_eq_xn_minus_yn_eq_2k (k n : ℕ) (h_pos_k : k > 0) (h_pos_n : n > 0) (h_n : n > 2) :
  ¬ ∃ x y : ℕ, x > 0 ∧ y > 0 ∧ x^n - y^n = 2^k := 
sorry

end no_sol_for_eq_xn_minus_yn_eq_2k_l598_59894


namespace amount_each_person_needs_to_raise_l598_59832

theorem amount_each_person_needs_to_raise (Total_goal Already_collected Number_of_people : ℝ) 
(h1 : Total_goal = 2400) (h2 : Already_collected = 300) (h3 : Number_of_people = 8) : 
    (Total_goal - Already_collected) / Number_of_people = 262.5 := 
by
  sorry

end amount_each_person_needs_to_raise_l598_59832


namespace candles_on_rituprts_cake_l598_59884

theorem candles_on_rituprts_cake (peter_candles : ℕ) (rupert_factor : ℝ) 
  (h_peter : peter_candles = 10) (h_rupert : rupert_factor = 3.5) : 
  ∃ rupert_candles : ℕ, rupert_candles = 35 :=
by
  sorry

end candles_on_rituprts_cake_l598_59884


namespace box_volume_l598_59850

-- Definitions for the dimensions of the box: Length (L), Width (W), and Height (H)
variables (L W H : ℝ)

-- Condition 1: Area of the front face is half the area of the top face
def condition1 := L * W = 0.5 * (L * H)

-- Condition 2: Area of the top face is 1.5 times the area of the side face
def condition2 := L * H = 1.5 * (W * H)

-- Condition 3: Area of the side face is 200
def condition3 := W * H = 200

-- Theorem stating the volume of the box is 3000 given the above conditions
theorem box_volume : condition1 L W H ∧ condition2 L W H ∧ condition3 W H → L * W * H = 3000 :=
by sorry

end box_volume_l598_59850


namespace solution_to_axb_eq_0_l598_59809

theorem solution_to_axb_eq_0 (a b x : ℝ) (h₀ : a ≠ 0) (h₁ : (0, 4) ∈ {p : ℝ × ℝ | p.snd = a * p.fst + b}) (h₂ : (-3, 0) ∈ {p : ℝ × ℝ | p.snd = a * p.fst + b}) :
  x = -3 :=
by
  sorry

end solution_to_axb_eq_0_l598_59809


namespace at_least_one_genuine_product_l598_59868

-- Definitions of the problem conditions
structure Products :=
  (total : ℕ)
  (genuine : ℕ)
  (defective : ℕ)

def products : Products := { total := 12, genuine := 10, defective := 2 }

-- Definition of the event
def certain_event (p : Products) (selected : ℕ) : Prop :=
  selected > p.defective

-- The theorem stating that there is at least one genuine product among the selected ones
theorem at_least_one_genuine_product : certain_event products 3 :=
by
  sorry

end at_least_one_genuine_product_l598_59868


namespace fourth_root_of_expression_l598_59839

theorem fourth_root_of_expression (x : ℝ) (h : 0 < x) : Real.sqrt (x^3 * Real.sqrt (x^2)) ^ (1 / 4) = x := sorry

end fourth_root_of_expression_l598_59839


namespace jake_and_luke_items_l598_59897

theorem jake_and_luke_items :
  ∃ (p j : ℕ), 6 * p + 2 * j ≤ 50 ∧ (∀ (p' : ℕ), 6 * p' + 2 * j ≤ 50 → p' ≤ p) ∧ p + j = 9 :=
by
  sorry

end jake_and_luke_items_l598_59897


namespace sum_of_non_solutions_l598_59812

theorem sum_of_non_solutions (A B C x: ℝ) 
  (h1 : A = 2) 
  (h2 : B = C / 2) 
  (h3 : C = 28) 
  (eq_inf_solutions : ∀ x, (x ≠ -C ∧ x ≠ -14) → 
  (x + B) * (A * x + 56) = 2 * ((x + C) * (x + 14))) : 
  (-14 + -28) = -42 :=
by
  sorry

end sum_of_non_solutions_l598_59812


namespace number_line_steps_l598_59804

theorem number_line_steps (n : ℕ) (total_distance : ℕ) (steps_to_x : ℕ) (x : ℕ)
  (h1 : total_distance = 32)
  (h2 : n = 8)
  (h3 : steps_to_x = 6)
  (h4 : x = (total_distance / n) * steps_to_x) :
  x = 24 := 
sorry

end number_line_steps_l598_59804


namespace shaded_square_percentage_l598_59881

theorem shaded_square_percentage (total_squares shaded_squares : ℕ) (h_total: total_squares = 25) (h_shaded: shaded_squares = 13) : 
(shaded_squares * 100) / total_squares = 52 := 
by
  sorry

end shaded_square_percentage_l598_59881


namespace mark_sold_9_boxes_less_than_n_l598_59834

theorem mark_sold_9_boxes_less_than_n :
  ∀ (n M A : ℕ),
  n = 10 →
  M < n →
  A = n - 2 →
  M + A < n →
  M ≥ 1 →
  A ≥ 1 →
  M = 1 ∧ n - M = 9 :=
by
  intros n M A h_n h_M_lt_n h_A h_MA_lt_n h_M_ge_1 h_A_ge_1
  rw [h_n, h_A] at *
  sorry

end mark_sold_9_boxes_less_than_n_l598_59834


namespace simplify_expression_l598_59845

theorem simplify_expression (a b : ℚ) (ha : a = -1) (hb : b = 1/4) : 
  (a + 2 * b)^2 + (a + 2 * b) * (a - 2 * b) = 1 := 
by 
  sorry

end simplify_expression_l598_59845


namespace apples_per_pie_l598_59817

theorem apples_per_pie
  (total_apples : ℕ) (apples_handed_out : ℕ) (remaining_apples : ℕ) (number_of_pies : ℕ)
  (h1 : total_apples = 96)
  (h2 : apples_handed_out = 42)
  (h3 : remaining_apples = total_apples - apples_handed_out)
  (h4 : remaining_apples = 54)
  (h5 : number_of_pies = 9) :
  remaining_apples / number_of_pies = 6 := by
  sorry

end apples_per_pie_l598_59817


namespace difference_between_extrema_l598_59820

noncomputable def f (x a b : ℝ) : ℝ := x^3 + 3 * a * x^2 + 3 * b * x

theorem difference_between_extrema (a b : ℝ)
  (h1 : 3 * (2 : ℝ)^2 + 6 * a * (2 : ℝ) + 3 * b = 0)
  (h2 : 3 * (1 : ℝ)^2 + 6 * a * (1 : ℝ) + 3 * b = -3) :
  f 0 a b - f 2 a b = 4 :=
by
  sorry

end difference_between_extrema_l598_59820


namespace differential_savings_l598_59883

theorem differential_savings (income : ℕ) (tax_rate_before : ℝ) (tax_rate_after : ℝ) : 
  income = 36000 → tax_rate_before = 0.46 → tax_rate_after = 0.32 →
  ((income * tax_rate_before) - (income * tax_rate_after)) = 5040 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end differential_savings_l598_59883


namespace find_initial_number_l598_59826

-- Define the initial equation
def initial_equation (x : ℤ) : Prop := x - 12 * 3 * 2 = 9938

-- Prove that the initial number x is equal to 10010 given initial_equation
theorem find_initial_number (x : ℤ) (h : initial_equation x) : x = 10010 :=
sorry

end find_initial_number_l598_59826


namespace total_simple_interest_l598_59862

theorem total_simple_interest (P R T : ℝ) (hP : P = 6178.846153846154) (hR : R = 0.13) (hT : T = 5) :
    P * R * T = 4011.245192307691 := by
  rw [hP, hR, hT]
  norm_num
  sorry

end total_simple_interest_l598_59862


namespace rainfall_mondays_l598_59893

theorem rainfall_mondays
  (M : ℕ)
  (rain_monday : ℝ)
  (rain_tuesday : ℝ)
  (num_tuesdays : ℕ)
  (extra_rain_tuesdays : ℝ)
  (h1 : rain_monday = 1.5)
  (h2 : rain_tuesday = 2.5)
  (h3 : num_tuesdays = 9)
  (h4 : num_tuesdays * rain_tuesday = rain_monday * M + extra_rain_tuesdays)
  (h5 : extra_rain_tuesdays = 12) :
  M = 7 := 
sorry

end rainfall_mondays_l598_59893


namespace cylinder_volume_transformation_l598_59802

theorem cylinder_volume_transformation (π : ℝ) (r h : ℝ) (V : ℝ) (V_new : ℝ)
  (hV : V = π * r^2 * h) (hV_initial : V = 20) : V_new = π * (3 * r)^2 * (4 * h) :=
by
sorry

end cylinder_volume_transformation_l598_59802


namespace average_age_combined_l598_59853

theorem average_age_combined (fifth_graders_count : ℕ) (fifth_graders_avg_age : ℚ)
                             (parents_count : ℕ) (parents_avg_age : ℚ)
                             (grandparents_count : ℕ) (grandparents_avg_age : ℚ) :
  fifth_graders_count = 40 →
  fifth_graders_avg_age = 10 →
  parents_count = 60 →
  parents_avg_age = 35 →
  grandparents_count = 20 →
  grandparents_avg_age = 65 →
  (fifth_graders_count * fifth_graders_avg_age + 
   parents_count * parents_avg_age + 
   grandparents_count * grandparents_avg_age) / 
  (fifth_graders_count + parents_count + grandparents_count) = 95 / 3 := sorry

end average_age_combined_l598_59853


namespace solve_for_x_l598_59872

-- Defining the given conditions
def y : ℕ := 6
def lhs (x : ℕ) : ℕ := Nat.pow x y
def rhs : ℕ := Nat.pow 3 12

-- Theorem statement to prove
theorem solve_for_x (x : ℕ) (hypothesis : lhs x = rhs) : x = 9 :=
by sorry

end solve_for_x_l598_59872


namespace prime_iff_totient_divisor_sum_l598_59808

def is_prime (n : ℕ) : Prop := 2 ≤ n ∧ ∀ m, m ∣ n → m = 1 ∨ m = n

def euler_totient (n : ℕ) : ℕ := sorry  -- we assume implementation of Euler's Totient function
def divisor_sum (n : ℕ) : ℕ := sorry  -- we assume implementation of Divisor sum function

theorem prime_iff_totient_divisor_sum (n : ℕ) :
  (2 ≤ n) → (euler_totient n ∣ (n - 1)) → (n + 1 ∣ divisor_sum n) → is_prime n :=
  sorry

end prime_iff_totient_divisor_sum_l598_59808


namespace amount_each_student_should_pay_l598_59831

noncomputable def total_rental_fee_per_book_per_half_hour : ℕ := 4000 
noncomputable def total_books : ℕ := 4
noncomputable def total_students : ℕ := 6
noncomputable def total_hours : ℕ := 3
noncomputable def total_half_hours : ℕ := total_hours * 2

noncomputable def total_fee_one_book : ℕ := total_rental_fee_per_book_per_half_hour * total_half_hours
noncomputable def total_fee_all_books : ℕ := total_fee_one_book * total_books

theorem amount_each_student_should_pay : total_fee_all_books / total_students = 16000 := by
  sorry

end amount_each_student_should_pay_l598_59831


namespace value_of_x_l598_59886

theorem value_of_x (x : ℝ) (h : 3 * x + 15 = (1/3) * (7 * x + 45)) : x = 0 :=
by
  sorry

end value_of_x_l598_59886


namespace slope_of_line_l598_59866

theorem slope_of_line (x y : ℝ) (h : 2 * y = -3 * x + 6) : (∃ m b : ℝ, y = m * x + b) ∧  (m = -3 / 2) :=
by 
  sorry

end slope_of_line_l598_59866


namespace loads_of_laundry_l598_59860

theorem loads_of_laundry (families : ℕ) (days : ℕ) (adults_per_family : ℕ) (children_per_family : ℕ)
  (adult_towels_per_day : ℕ) (child_towels_per_day : ℕ) (initial_capacity : ℕ) (reduced_capacity : ℕ)
  (initial_days : ℕ) (remaining_days : ℕ) : 
  families = 7 → days = 12 → adults_per_family = 2 → children_per_family = 4 → 
  adult_towels_per_day = 2 → child_towels_per_day = 1 → initial_capacity = 8 → 
  reduced_capacity = 6 → initial_days = 6 → remaining_days = 6 → 
  (families * (adults_per_family * adult_towels_per_day + children_per_family * child_towels_per_day) * initial_days / initial_capacity) +
  (families * (adults_per_family * adult_towels_per_day + children_per_family * child_towels_per_day) * remaining_days / reduced_capacity) = 98 :=
by 
  intros _ _ _ _ _ _ _ _ _ _
  sorry

end loads_of_laundry_l598_59860


namespace find_denomination_of_oliver_bills_l598_59813

-- Definitions based on conditions
def denomination (x : ℕ) : Prop :=
  let oliver_total := 10 * x + 3 * 5
  let william_total := 15 * 10 + 4 * 5
  oliver_total = william_total + 45

-- The Lean theorem statement
theorem find_denomination_of_oliver_bills (x : ℕ) : denomination x → x = 20 := by
  sorry

end find_denomination_of_oliver_bills_l598_59813


namespace tom_hockey_games_l598_59840

def tom_hockey_games_last_year (games_this_year missed_this_year total_games : Nat) : Nat :=
  total_games - games_this_year

theorem tom_hockey_games :
  ∀ (games_this_year missed_this_year total_games : Nat),
    games_this_year = 4 →
    missed_this_year = 7 →
    total_games = 13 →
    tom_hockey_games_last_year games_this_year total_games = 9 := by
  intros games_this_year missed_this_year total_games h1 h2 h3
  -- The proof steps would go here
  sorry

end tom_hockey_games_l598_59840


namespace find_abc_l598_59869

theorem find_abc (a b c : ℕ) (h1 : 1 < a) (h2 : a < b) (h3 : b < c) (h4 : (a-1) * (b-1) * (c-1) ∣ a * b * c - 1) :
  (a = 3 ∧ b = 5 ∧ c = 15) ∨ (a = 2 ∧ b = 4 ∧ c = 8) :=
by 
  sorry

end find_abc_l598_59869


namespace regression_eq_change_in_y_l598_59882

-- Define the regression equation
def regression_eq (x : ℝ) : ℝ := 2 - 1.5 * x

-- Define the statement to be proved
theorem regression_eq_change_in_y (x : ℝ) :
  regression_eq (x + 1) = regression_eq x - 1.5 :=
by sorry

end regression_eq_change_in_y_l598_59882


namespace donuts_percentage_missing_l598_59852

noncomputable def missing_donuts_percentage (initial_donuts : ℕ) (remaining_donuts : ℕ) : ℝ :=
  ((initial_donuts - remaining_donuts : ℕ) : ℝ) / initial_donuts * 100

theorem donuts_percentage_missing
  (h_initial : ℕ := 30)
  (h_remaining : ℕ := 9) :
  missing_donuts_percentage h_initial h_remaining = 70 :=
by
  sorry

end donuts_percentage_missing_l598_59852


namespace sally_lost_orange_balloons_l598_59819

theorem sally_lost_orange_balloons :
  ∀ (initial_orange_balloons lost_orange_balloons current_orange_balloons : ℕ),
  initial_orange_balloons = 9 →
  current_orange_balloons = 7 →
  lost_orange_balloons = initial_orange_balloons - current_orange_balloons →
  lost_orange_balloons = 2 :=
by
  intros initial_orange_balloons lost_orange_balloons current_orange_balloons
  intros h_init h_current h_lost
  rw [h_init, h_current] at h_lost
  exact h_lost

end sally_lost_orange_balloons_l598_59819


namespace total_marks_scored_l598_59815

theorem total_marks_scored :
  let Keith_score := 3.5
  let Larry_score := Keith_score * 3.2
  let Danny_score := Larry_score + 5.7
  let Emma_score := (Danny_score * 2) - 1.2
  let Fiona_score := (Keith_score + Larry_score + Danny_score + Emma_score) / 4
  Keith_score + Larry_score + Danny_score + Emma_score + Fiona_score = 80.25 :=
by
  sorry

end total_marks_scored_l598_59815


namespace range_of_m_l598_59816

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := 2 * m * x^2 - 2 * (4 - m) * x + 1
noncomputable def g (m : ℝ) (x : ℝ) : ℝ := m * x

theorem range_of_m (m : ℝ) :
  (∀ x : ℝ, f m x > 0 ∨ g m x > 0) ↔ 0 < m ∧ m < 8 :=
sorry

end range_of_m_l598_59816


namespace simplify_expression_l598_59847

theorem simplify_expression : 4 * (15 / 5) * (24 / -60) = - (24 / 5) := 
by
  sorry

end simplify_expression_l598_59847


namespace total_sales_l598_59898

theorem total_sales (T : ℝ) (h1 : (2 / 5) * T = (2 / 5) * T) (h2 : (3 / 5) * T = 48) : T = 80 :=
by
  -- added sorry to skip proofs as per the requirement
  sorry

end total_sales_l598_59898


namespace geometric_sequence_condition_l598_59858

-- Define the condition ac = b^2
def condition (a b c : ℝ) : Prop := a * c = b ^ 2

-- Define what it means for a, b, c to form a geometric sequence
def geometric_sequence (a b c : ℝ) : Prop := 
  (b ≠ 0 → a / b = b / c) ∧ (a = 0 → b = 0 ∧ c = 0)

-- The goal is to prove the necessary but not sufficient condition
theorem geometric_sequence_condition (a b c : ℝ) :
  condition a b c ↔ (geometric_sequence a b c → condition a b c) ∧ (¬ (geometric_sequence a b c) → condition a b c ∧ ¬ (geometric_sequence (2 : ℝ) (0 : ℝ) (0 : ℝ))) :=
by
  sorry

end geometric_sequence_condition_l598_59858


namespace remainder_problem_l598_59803

theorem remainder_problem :
  ((98 * 103 + 7) % 12) = 1 :=
by
  sorry

end remainder_problem_l598_59803


namespace value_of_n_l598_59811

-- Define required conditions
variables (n : ℕ) (f : ℕ → ℕ → ℕ)

-- Conditions
axiom cond1 : n > 7
axiom cond2 : ∀ m k : ℕ, f m k = 2^(n - m) * Nat.choose m k

-- Given condition
axiom after_seventh_round : f 7 5 = 42

-- Theorem to prove
theorem value_of_n : n = 8 :=
by
  -- Proof goes here
  sorry

end value_of_n_l598_59811


namespace inequality_holds_l598_59891

theorem inequality_holds (x y : ℝ) (hx : x ≠ -1) (hy : y ≠ -1) (hxy : x * y = 1) : 
  ((2 + x)/(1 + x))^2 + ((2 + y)/(1 + y))^2 ≥ 9/2 := 
sorry

end inequality_holds_l598_59891


namespace area_of_region_l598_59854

theorem area_of_region : 
    ∃ (area : ℝ), 
    (∀ (x y : ℝ), (x^2 + y^2 + 6 * x - 10 * y + 5 = 0) → 
    area = 29 * Real.pi) := 
by
  use 29 * Real.pi
  intros x y h
  sorry

end area_of_region_l598_59854


namespace poly_remainder_l598_59818

theorem poly_remainder (x : ℤ) :
  (x^1012) % (x^3 - x^2 + x - 1) = 1 := by
  sorry

end poly_remainder_l598_59818


namespace yield_percentage_is_correct_l598_59836

-- Defining the conditions and question
def market_value := 70
def face_value := 100
def dividend_percentage := 7
def annual_dividend := (dividend_percentage * face_value) / 100

-- Lean statement to prove the yield percentage
theorem yield_percentage_is_correct (market_value: ℕ) (annual_dividend: ℝ) : 
  ((annual_dividend / market_value) * 100) = 10 := 
by
  -- conditions from a)
  have market_value := 70
  have face_value := 100
  have dividend_percentage := 7
  have annual_dividend := (dividend_percentage * face_value) / 100
  
  -- proof will go here
  sorry

end yield_percentage_is_correct_l598_59836


namespace find_point_coordinates_l598_59873

open Real

-- Define circles C1 and C2
def circle_C1 (x y : ℝ) : Prop := (x + 4)^2 + (y - 2)^2 = 9
def circle_C2 (x y : ℝ) : Prop := (x - 5)^2 + (y - 6)^2 = 9

-- Define mutually perpendicular lines passing through point P
def line_l1 (P : ℝ × ℝ) (k : ℝ) (x y : ℝ) : Prop := y - P.2 = k * (x - P.1)
def line_l2 (P : ℝ × ℝ) (k : ℝ) (x y : ℝ) : Prop := y - P.2 = -1/k * (x - P.1)

-- Define the condition that chord lengths intercepted by lines on respective circles are equal
def equal_chord_lengths (P : ℝ × ℝ) (k : ℝ) : Prop :=
  abs (-4 * k - 2 + P.2 - k * P.1) / sqrt ((k^2) + 1) = abs (5 + 6 * k - k * P.2 - P.1) / sqrt ((k^2) + 1)

-- Main statement to be proved
theorem find_point_coordinates :
  ∃ (P : ℝ × ℝ), 
  circle_C1 (P.1) (P.2) ∧
  circle_C2 (P.1) (P.2) ∧
  (∀ k : ℝ, k ≠ 0 → equal_chord_lengths P k) ∧
  (P = (-3/2, 17/2) ∨ P = (5/2, -1/2)) :=
sorry

end find_point_coordinates_l598_59873


namespace total_cows_l598_59887

theorem total_cows (n : ℕ) 
  (h₁ : n / 3 + n / 6 + n / 9 + 8 = n) : n = 144 :=
by sorry

end total_cows_l598_59887


namespace solve_equation_solve_inequality_l598_59874

-- Defining the first problem
theorem solve_equation (x : ℝ) : 3 * (x - 2) - (1 - 2 * x) = 3 ↔ x = 2 := 
by
  sorry

-- Defining the second problem
theorem solve_inequality (x : ℝ) : (2 * x - 1 < 4 * x + 3) ↔ (x > -2) :=
by
  sorry

end solve_equation_solve_inequality_l598_59874


namespace tan_neg_five_pi_over_four_l598_59859

theorem tan_neg_five_pi_over_four : Real.tan (-5 * Real.pi / 4) = -1 :=
  sorry

end tan_neg_five_pi_over_four_l598_59859


namespace negation_exists_or_l598_59800

theorem negation_exists_or (x : ℝ) :
  ¬ (∃ x : ℝ, x ≤ -1 ∨ x ≥ 2) ↔ ∀ x : ℝ, -1 < x ∧ x < 2 :=
by sorry

end negation_exists_or_l598_59800


namespace max_value_of_quadratic_l598_59825

theorem max_value_of_quadratic :
  ∃ y : ℚ, ∀ x : ℚ, -x^2 - 3 * x + 4 ≤ y :=
sorry

end max_value_of_quadratic_l598_59825


namespace perpendicular_lines_condition_l598_59876

theorem perpendicular_lines_condition (m : ℝ) :
    (m = 1 → (∀ (x y : ℝ), (∀ (c d : ℝ), c * (m * x + y - 1) = 0 → d * (x - m * y - 1) = 0 → (c * m + d / m) ^ 2 = 1))) ∧ (∀ (m' : ℝ), m' ≠ 1 → ¬ (∀ (x y : ℝ), (∀ (c d : ℝ), c * (m' * x + y - 1) = 0 → d * (x - m' * y - 1) = 0 → (c * m' + d / m') ^ 2 = 1))) :=
by
  sorry

end perpendicular_lines_condition_l598_59876


namespace find_pairs_l598_59856

theorem find_pairs (m n : ℕ) : 
  (20^m - 10 * m^2 + 1 = 19^n ↔ (m = 0 ∧ n = 0) ∨ (m = 2 ∧ n = 2)) :=
by
  sorry

end find_pairs_l598_59856


namespace Isabella_paint_area_l598_59864

def bedroom1_length : ℕ := 14
def bedroom1_width : ℕ := 11
def bedroom1_height : ℕ := 9

def bedroom2_length : ℕ := 13
def bedroom2_width : ℕ := 12
def bedroom2_height : ℕ := 9

def unpaintable_area_per_bedroom : ℕ := 70

theorem Isabella_paint_area :
  let wall_area (length width height : ℕ) := 2 * (length * height) + 2 * (width * height)
  let paintable_area (length width height : ℕ) := wall_area length width height - unpaintable_area_per_bedroom
  paintable_area bedroom1_length bedroom1_width bedroom1_height +
  paintable_area bedroom1_length bedroom1_width bedroom1_height +
  paintable_area bedroom2_length bedroom2_width bedroom2_height +
  paintable_area bedroom2_length bedroom2_width bedroom2_height =
  1520 := 
by
  sorry

end Isabella_paint_area_l598_59864


namespace isosceles_triangle_perimeter_l598_59837

theorem isosceles_triangle_perimeter (a b : ℝ) (h1 : a = 4) (h2 : b = 8) (h3 : ∃ p q r, p = b ∧ q = b ∧ r = a ∧ p + q > r) : 
  a + b + b = 20 := 
by 
  sorry

end isosceles_triangle_perimeter_l598_59837


namespace difference_of_results_l598_59822

theorem difference_of_results (a b : ℕ) (ha : 0 ≤ a ∧ a ≤ 9) (hb : 0 ≤ b ∧ b ≤ 9) (h_diff: a ≠ b) :
  (70 * a - 7 * a) - (70 * b - 7 * b) = 0 :=
by
  sorry

end difference_of_results_l598_59822


namespace solve_quadratic_inequality_l598_59867

theorem solve_quadratic_inequality (x : ℝ) : (-x^2 - 2 * x + 3 < 0) ↔ (x < -3 ∨ x > 1) := 
sorry

end solve_quadratic_inequality_l598_59867


namespace profit_percentage_l598_59865

theorem profit_percentage (cost_price selling_price marked_price : ℝ)
  (h1 : cost_price = 47.50)
  (h2 : selling_price = 0.90 * marked_price)
  (h3 : selling_price = 65.97) :
  ((selling_price - cost_price) / cost_price) * 100 = 38.88 := 
by
  sorry

end profit_percentage_l598_59865


namespace euler_totient_inequality_l598_59888

open Int

def is_power_of_prime (m : ℕ) : Prop :=
  ∃ p k : ℕ, (Nat.Prime p) ∧ (k ≥ 1) ∧ (m = p^k)

def φ (n m : ℕ) (h : m ≠ 1) : ℕ := -- This is a placeholder, you would need an actual implementation for φ
  sorry

theorem euler_totient_inequality (m : ℕ) (h : m ≠ 1) :
  (is_power_of_prime m) ↔ (∀ n > 0, (φ n m h) / n ≥ (φ m m h) / m) :=
sorry

end euler_totient_inequality_l598_59888


namespace apr_sales_is_75_l598_59805

-- Definitions based on conditions
def sales_jan : ℕ := 90
def sales_feb : ℕ := 50
def sales_mar : ℕ := 70
def avg_sales : ℕ := 72

-- Total sales of first three months
def total_sales_jan_to_mar : ℕ := sales_jan + sales_feb + sales_mar

-- Total sales considering average sales over 5 months
def total_sales : ℕ := avg_sales * 5

-- Defining April sales
def sales_apr (sales_may : ℕ) : ℕ := total_sales - total_sales_jan_to_mar - sales_may

theorem apr_sales_is_75 (sales_may : ℕ) : sales_apr sales_may = 75 :=
by
  unfold sales_apr total_sales total_sales_jan_to_mar avg_sales sales_jan sales_feb sales_mar
  -- Here we could insert more steps if needed to directly connect to the proof
  sorry


end apr_sales_is_75_l598_59805


namespace stones_on_one_side_l598_59889

theorem stones_on_one_side (total_perimeter_stones : ℕ) (h : total_perimeter_stones = 84) :
  ∃ s : ℕ, 4 * s - 4 = total_perimeter_stones ∧ s = 22 :=
by
  use 22
  sorry

end stones_on_one_side_l598_59889


namespace remainder_of_3056_div_78_l598_59849

-- Define the necessary conditions and the statement
theorem remainder_of_3056_div_78 : (3056 % 78) = 14 :=
by
  sorry

end remainder_of_3056_div_78_l598_59849


namespace average_score_of_male_students_l598_59824

theorem average_score_of_male_students
  (female_students : ℕ) (male_students : ℕ) (female_avg_score : ℕ) (class_avg_score : ℕ)
  (h_female_students : female_students = 20)
  (h_male_students : male_students = 30)
  (h_female_avg_score : female_avg_score = 75)
  (h_class_avg_score : class_avg_score = 72) :
  (30 * (((class_avg_score * (female_students + male_students)) - (female_avg_score * female_students)) / male_students) = 70) :=
by
  -- Sorry for the proof
  sorry

end average_score_of_male_students_l598_59824


namespace triangle_angle_contradiction_l598_59880

theorem triangle_angle_contradiction :
  ∀ (α β γ : ℝ), (α + β + γ = 180) →
  (α > 60) ∧ (β > 60) ∧ (γ > 60) →
  false :=
by
  intros α β γ h_sum h_angles
  sorry

end triangle_angle_contradiction_l598_59880


namespace turtle_distance_during_rabbit_rest_l598_59843

theorem turtle_distance_during_rabbit_rest
  (D : ℕ)
  (vr vt : ℕ)
  (rabbit_speed_multiple : vr = 15 * vt)
  (rabbit_remaining_distance : D - 100 = 900)
  (turtle_finish_time : true)
  (rabbit_to_be_break : true)
  (turtle_finish_during_rabbit_rest : true) :
  (D - (900 / 15) = 940) :=
by
  sorry

end turtle_distance_during_rabbit_rest_l598_59843


namespace factor_expression_l598_59828

theorem factor_expression (x : ℝ) : 75 * x + 45 = 15 * (5 * x + 3) :=
  sorry

end factor_expression_l598_59828


namespace ratio_Smax_Smin_l598_59842

-- Define the area of a cube's diagonal cross-section through BD1
def cross_section_area (a : ℝ) : ℝ := sorry

theorem ratio_Smax_Smin (a : ℝ) (S S_min S_max : ℝ) :
  cross_section_area a = S →
  S_min = (a^2 * Real.sqrt 6) / 2 →
  S_max = a^2 * Real.sqrt 6 →
  S_max / S_min = 2 * Real.sqrt 3 / 3 :=
by
  sorry

end ratio_Smax_Smin_l598_59842


namespace necessary_and_sufficient_l598_59827

variable (α β : ℝ)
variable (p : Prop := α > β)
variable (q : Prop := α + Real.sin α * Real.cos β > β + Real.sin β * Real.cos α)

theorem necessary_and_sufficient : (p ↔ q) :=
by
  sorry

end necessary_and_sufficient_l598_59827


namespace unique_solution_3_pow_x_minus_2_pow_y_eq_7_l598_59861

theorem unique_solution_3_pow_x_minus_2_pow_y_eq_7 :
  ∀ x y : ℕ, (1 ≤ x) → (1 ≤ y) → (3 ^ x - 2 ^ y = 7) → (x = 2 ∧ y = 1) :=
by
  intros x y hx hy hxy
  sorry

end unique_solution_3_pow_x_minus_2_pow_y_eq_7_l598_59861


namespace unique_digit_10D4_count_unique_digit_10D4_l598_59835

theorem unique_digit_10D4 (D : ℕ) (hD : D < 10) : 
  (5 + D) % 3 = 0 ∧ (10 * D + 4) % 4 = 0 ↔ D = 4 :=
by
  sorry

theorem count_unique_digit_10D4 :
  ∃! D, (D < 10 ∧ (5 + D) % 3 = 0 ∧ (10 * D + 4) % 4 = 0) :=
by
  use 4
  simp [unique_digit_10D4]
  sorry

end unique_digit_10D4_count_unique_digit_10D4_l598_59835


namespace yi_reads_more_than_jia_by_9_pages_l598_59890

-- Define the number of pages in the book
def total_pages : ℕ := 120

-- Define number of pages read per day by Jia and Yi
def pages_per_day_jia : ℕ := 8
def pages_per_day_yi : ℕ := 13

-- Define the number of days in the period
def total_days : ℕ := 7

-- Calculate total pages read by Jia in the given period
def pages_read_by_jia : ℕ := total_days * pages_per_day_jia

-- Calculate the number of reading days by Yi in the given period
def reading_days_yi : ℕ := (total_days / 3) * 2 + (total_days % 3).min 2

-- Calculate total pages read by Yi in the given period
def pages_read_by_yi : ℕ := reading_days_yi * pages_per_day_yi

-- Given all conditions, prove that Yi reads 9 pages more than Jia over the 7-day period
theorem yi_reads_more_than_jia_by_9_pages :
  pages_read_by_yi - pages_read_by_jia = 9 :=
by
  sorry

end yi_reads_more_than_jia_by_9_pages_l598_59890


namespace solution1_solution2_l598_59801

noncomputable def Problem1 : ℝ :=
  4 + (-2)^3 * 5 - (-0.28) / 4

theorem solution1 : Problem1 = -35.93 := by
  sorry

noncomputable def Problem2 : ℚ :=
  -1^4 - (1/6) * (2 - (-3)^2)

theorem solution2 : Problem2 = 1/6 := by
  sorry

end solution1_solution2_l598_59801


namespace television_combinations_l598_59814

def combination (n k : ℕ) : ℕ := Nat.choose n k

theorem television_combinations :
  ∃ (combinations : ℕ), 
  ∀ (A B total : ℕ), A = 4 → B = 5 → total = 3 →
  combinations = (combination 4 2 * combination 5 1 + combination 4 1 * combination 5 2) →
  combinations = 70 :=
sorry

end television_combinations_l598_59814


namespace avg_salary_increase_l598_59851

theorem avg_salary_increase (A1 : ℝ) (M : ℝ) (n : ℕ) (N : ℕ) 
  (h1 : n = 20) (h2 : A1 = 1500) (h3 : M = 4650) (h4 : N = n + 1) :
  (20 * A1 + M) / N - A1 = 150 :=
by
  -- proof goes here
  sorry

end avg_salary_increase_l598_59851


namespace part1_part2_l598_59848

theorem part1 : (π - 3)^0 + (-1)^(2023) - Real.sqrt 8 = -2 * Real.sqrt 2 := sorry

theorem part2 (x : ℝ) : (4 * x - 3 > 9) ∧ (2 + x ≥ 0) ↔ x > 3 := sorry

end part1_part2_l598_59848


namespace quadratic_condition_l598_59829

theorem quadratic_condition (a : ℝ) : 
  (∃ x : ℝ, a * x^2 - 2 * x + 3 = 0) → a ≠ 0 :=
by 
  intro h
  -- Proof will be here
  sorry

end quadratic_condition_l598_59829


namespace polygon_with_150_degree_interior_angles_has_12_sides_l598_59870

-- Define the conditions as Lean definitions
def regular_polygon (n : ℕ) : Prop :=
  n ≥ 3 ∧ ∃ (a : ℝ), a = 150 ∧ 180 * (n - 2) = 150 * n

-- Theorem statement
theorem polygon_with_150_degree_interior_angles_has_12_sides :
  ∃ (n : ℕ), regular_polygon n ∧ n = 12 := 
by
  sorry

end polygon_with_150_degree_interior_angles_has_12_sides_l598_59870


namespace geometric_sequence_properties_l598_59806

theorem geometric_sequence_properties (a : ℕ → ℝ) (q : ℝ) :
  a 1 = 1 / 2 ∧ a 4 = -4 → q = -2 ∧ (∀ n, a n = 1 / 2 * q ^ (n - 1)) :=
by
  intro h
  sorry

end geometric_sequence_properties_l598_59806


namespace man_total_pay_l598_59863

def regular_rate : ℕ := 3
def regular_hours : ℕ := 40
def overtime_hours : ℕ := 13

def regular_pay : ℕ := regular_rate * regular_hours
def overtime_rate : ℕ := 2 * regular_rate
def overtime_pay : ℕ := overtime_rate * overtime_hours

def total_pay : ℕ := regular_pay + overtime_pay

theorem man_total_pay : total_pay = 198 := by
  sorry

end man_total_pay_l598_59863


namespace initial_sale_price_percent_l598_59875

theorem initial_sale_price_percent (P S : ℝ) (h1 : S * 0.90 = 0.63 * P) :
  S = 0.70 * P :=
by
  sorry

end initial_sale_price_percent_l598_59875


namespace ski_helmet_final_price_l598_59838

variables (original_price : ℝ) (first_discount : ℝ) (second_discount : ℝ)
def final_price_after_discounts (initial_price : ℝ) (discount1 : ℝ) (discount2 : ℝ) : ℝ :=
  let after_first_discount := initial_price * (1 - discount1)
  let after_second_discount := after_first_discount * (1 - discount2)
  after_second_discount

theorem ski_helmet_final_price :
  final_price_after_discounts 120 0.40 0.20 = 57.60 := 
  sorry

end ski_helmet_final_price_l598_59838


namespace intersection_of_M_and_N_is_0_and_2_l598_59879

open Set

def M : Set ℕ := {0, 1, 2}

def N : Set ℕ := {x | ∃ a ∈ M, x = 2 * a}

theorem intersection_of_M_and_N_is_0_and_2 : M ∩ N = {0, 2} :=
by
  sorry

end intersection_of_M_and_N_is_0_and_2_l598_59879


namespace total_shaded_area_of_square_carpet_l598_59877

theorem total_shaded_area_of_square_carpet :
  ∀ (S T : ℝ),
    (9 / S = 3) →
    (S / T = 3) →
    (8 * T^2 + S^2 = 17) :=
by
  intros S T h1 h2
  sorry

end total_shaded_area_of_square_carpet_l598_59877


namespace nat_implies_int_incorrect_reasoning_due_to_minor_premise_l598_59857

-- Definitions for conditions
def is_integer (x : ℚ) : Prop := ∃ (n : ℤ), x = n
def is_natural (x : ℚ) : Prop := ∃ (n : ℕ), x = n

-- Major premise: Natural numbers are integers
theorem nat_implies_int (n : ℕ) : is_integer n := 
  ⟨n, rfl⟩

-- Minor premise: 1 / 3 is a natural number
def one_div_three_is_natural : Prop := is_natural (1 / 3)

-- Conclusion: 1 / 3 is an integer
def one_div_three_is_integer : Prop := is_integer (1 / 3)

-- The proof problem
theorem incorrect_reasoning_due_to_minor_premise :
  ¬one_div_three_is_natural :=
sorry

end nat_implies_int_incorrect_reasoning_due_to_minor_premise_l598_59857


namespace correct_calculation_l598_59892

theorem correct_calculation (a b : ℝ) : (a * b) ^ 2 = a ^ 2 * b ^ 2 := by
  sorry

end correct_calculation_l598_59892


namespace selling_prices_for_10_percent_profit_l598_59807

theorem selling_prices_for_10_percent_profit
    (cost1 cost2 cost3 : ℝ)
    (cost1_eq : cost1 = 200)
    (cost2_eq : cost2 = 300)
    (cost3_eq : cost3 = 500)
    (profit_percent : ℝ)
    (profit_percent_eq : profit_percent = 0.10):
    ∃ s1 s2 s3 : ℝ,
      s1 = cost1 + 33.33 ∧
      s2 = cost2 + 33.33 ∧
      s3 = cost3 + 33.33 ∧
      s1 + s2 + s3 = 1100 :=
by
  sorry

end selling_prices_for_10_percent_profit_l598_59807


namespace exam_total_boys_l598_59871

theorem exam_total_boys (T F : ℕ) (avg_total avg_passed avg_failed : ℕ) 
    (H1 : avg_total = 40) (H2 : avg_passed = 39) (H3 : avg_failed = 15) (H4 : 125 > 0) (H5 : 125 * avg_passed + (T - 125) * avg_failed = T * avg_total) : T = 120 :=
by
  sorry

end exam_total_boys_l598_59871


namespace floor_square_of_sqrt_50_eq_49_l598_59833

theorem floor_square_of_sqrt_50_eq_49 : (Int.floor (Real.sqrt 50))^2 = 49 := 
by
  sorry

end floor_square_of_sqrt_50_eq_49_l598_59833


namespace remainder_of_sum_of_squares_mod_n_l598_59885

theorem remainder_of_sum_of_squares_mod_n (a b n : ℤ) (hn : n > 1) 
  (ha : a * a ≡ 1 [ZMOD n]) (hb : b * b ≡ 1 [ZMOD n]) : 
  (a^2 + b^2) % n = 2 := 
by 
  sorry

end remainder_of_sum_of_squares_mod_n_l598_59885


namespace opposite_of_neg_three_l598_59830

-- Define the concept of negation and opposite of a number
def opposite (x : ℤ) : ℤ := -x

-- State the problem: Prove that the opposite of -3 is 3
theorem opposite_of_neg_three : opposite (-3) = 3 :=
by
  -- Proof
  sorry

end opposite_of_neg_three_l598_59830


namespace Joan_balloons_l598_59844

variable (J : ℕ) -- Joan's blue balloons

theorem Joan_balloons (h : J + 41 = 81) : J = 40 :=
by
  sorry

end Joan_balloons_l598_59844


namespace total_female_officers_l598_59841

theorem total_female_officers
  (percent_female_on_duty : ℝ)
  (total_on_duty : ℝ)
  (half_of_total_on_duty : ℝ)
  (num_females_on_duty : ℝ) :
  percent_female_on_duty = 0.10 →
  total_on_duty = 200 →
  half_of_total_on_duty = total_on_duty / 2 →
  num_females_on_duty = half_of_total_on_duty →
  num_females_on_duty = percent_female_on_duty * (1000 : ℝ) :=
by
  intros h1 h2 h3 h4
  sorry

end total_female_officers_l598_59841


namespace point_in_fourth_quadrant_l598_59896

theorem point_in_fourth_quadrant (x y : Real) (hx : x = 2) (hy : y = Real.tan 300) : 
  (0 < x) → (y < 0) → (x = 2 ∧ y = -Real.sqrt 3) :=
by
  intro hx_trans hy_trans
  -- Here you will provide statements or tactics to assist the proof if you were completing it
  sorry

end point_in_fourth_quadrant_l598_59896


namespace length_of_segment_AC_l598_59810

theorem length_of_segment_AC :
  ∀ (a b h: ℝ),
    (a = b) →
    (h = a * Real.sqrt 2) →
    (4 = (a + b - h) / 2) →
    a = 4 * Real.sqrt 2 + 8 :=
by
  sorry

end length_of_segment_AC_l598_59810


namespace find_cos_alpha_l598_59899

variable (α β : ℝ)

-- Conditions
def acute_angles (α β : ℝ) : Prop := 0 < α ∧ α < (Real.pi / 2) ∧ 0 < β ∧ β < (Real.pi / 2)
def cos_alpha_beta : Prop := Real.cos (α + β) = 12 / 13
def cos_2alpha_beta : Prop := Real.cos (2 * α + β) = 3 / 5

-- Main theorem
theorem find_cos_alpha (h1 : acute_angles α β) (h2 : cos_alpha_beta α β) (h3 : cos_2alpha_beta α β) : 
  Real.cos α = 56 / 65 :=
sorry

end find_cos_alpha_l598_59899


namespace total_cantaloupes_l598_59846

theorem total_cantaloupes (fred_cantaloupes : ℕ) (tim_cantaloupes : ℕ) (h1 : fred_cantaloupes = 38) (h2 : tim_cantaloupes = 44) : fred_cantaloupes + tim_cantaloupes = 82 :=
by
  sorry

end total_cantaloupes_l598_59846


namespace carrots_not_used_l598_59895

theorem carrots_not_used :
  let total_carrots := 300
  let carrots_before_lunch := (2 / 5) * total_carrots
  let remaining_after_lunch := total_carrots - carrots_before_lunch
  let carrots_by_end_of_day := (3 / 5) * remaining_after_lunch
  remaining_after_lunch - carrots_by_end_of_day = 72
:= by
  sorry

end carrots_not_used_l598_59895


namespace cubes_difference_l598_59821

theorem cubes_difference 
  (a b : ℝ) 
  (h1 : a - b = 7) 
  (h2 : a^2 + b^2 = 53) : 
  a^3 - b^3 = 385 := by
  sorry

end cubes_difference_l598_59821


namespace team_C_has_most_uniform_height_l598_59823

theorem team_C_has_most_uniform_height
  (S_A S_B S_C S_D : ℝ)
  (h_A : S_A = 0.13)
  (h_B : S_B = 0.11)
  (h_C : S_C = 0.09)
  (h_D : S_D = 0.15)
  (h_same_num_members : ∀ (a b c d : ℕ), a = b ∧ b = c ∧ c = d) 
  : S_C = min S_A (min S_B (min S_C S_D)) :=
by
  sorry

end team_C_has_most_uniform_height_l598_59823


namespace convert_base_10_to_base_6_l598_59855

theorem convert_base_10_to_base_6 : 
  ∃ (digits : List ℕ), (digits.length = 4 ∧
    List.foldr (λ (x : ℕ) (acc : ℕ) => acc * 6 + x) 0 digits = 314 ∧
    digits = [1, 2, 4, 2]) := by
  sorry

end convert_base_10_to_base_6_l598_59855
