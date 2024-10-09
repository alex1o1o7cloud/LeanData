import Mathlib

namespace solve_for_x_l2234_223404

theorem solve_for_x (x : ℤ) : 3 * (5 - x) = 9 → x = 2 :=
by {
  sorry
}

end solve_for_x_l2234_223404


namespace object_speed_conversion_l2234_223426

theorem object_speed_conversion 
  (distance : ℝ)
  (velocity : ℝ) 
  (conversion_factor : ℝ) 
  (distance_in_km : ℝ)
  (time_in_seconds : ℝ) 
  (time_in_minutes : ℝ) 
  (speed_in_kmh : ℝ) :
  distance = 200 ∧ 
  velocity = 1/3 ∧ 
  time_in_seconds = distance / velocity ∧ 
  time_in_minutes = time_in_seconds / 60 ∧ 
  conversion_factor = 3600 * 0.001 ∧ 
  speed_in_kmh = velocity * conversion_factor ↔ 
  speed_in_kmh = 0.4 :=
by sorry

end object_speed_conversion_l2234_223426


namespace books_per_author_l2234_223431

theorem books_per_author (total_books : ℕ) (authors : ℕ) (h1 : total_books = 198) (h2 : authors = 6) : total_books / authors = 33 :=
by sorry

end books_per_author_l2234_223431


namespace shelves_used_l2234_223480

def initial_books : ℕ := 86
def books_sold : ℕ := 37
def books_per_shelf : ℕ := 7
def remaining_books : ℕ := initial_books - books_sold
def shelves : ℕ := remaining_books / books_per_shelf

theorem shelves_used : shelves = 7 := by
  -- proof will go here
  sorry

end shelves_used_l2234_223480


namespace eval_sum_and_subtract_l2234_223478

theorem eval_sum_and_subtract : (2345 + 3452 + 4523 + 5234) - 1234 = 14320 := by {
  -- The rest of the proof should go here, but we'll use sorry to skip it.
  sorry
}

end eval_sum_and_subtract_l2234_223478


namespace quarters_number_l2234_223430

theorem quarters_number (total_value : ℝ)
    (bills1 : ℝ := 2)
    (bill5 : ℝ := 5)
    (dimes : ℝ := 20 * 0.1)
    (nickels : ℝ := 8 * 0.05)
    (pennies : ℝ := 35 * 0.01) :
    total_value = 13 → (total_value - (bills1 + bill5 + dimes + nickels + pennies)) / 0.25 = 13 :=
by
  intro h
  have h_total := h
  sorry

end quarters_number_l2234_223430


namespace number_that_multiplies_b_l2234_223401

variable (a b x : ℝ)

theorem number_that_multiplies_b (h1 : 7 * a = x * b) (h2 : a * b ≠ 0) (h3 : (a / 8) / (b / 7) = 1) : x = 8 := 
sorry

end number_that_multiplies_b_l2234_223401


namespace fisher_catch_l2234_223472

theorem fisher_catch (x y : ℕ) (h1 : x + y = 80)
  (h2 : ∃ a : ℕ, x = 9 * a)
  (h3 : ∃ b : ℕ, y = 11 * b) :
  x = 36 ∧ y = 44 :=
by
  sorry

end fisher_catch_l2234_223472


namespace strawberry_harvest_l2234_223422

theorem strawberry_harvest
  (length : ℕ) (width : ℕ)
  (plants_per_sqft : ℕ) (yield_per_plant : ℕ)
  (garden_area : ℕ := length * width) 
  (total_plants : ℕ := plants_per_sqft * garden_area) 
  (expected_strawberries : ℕ := yield_per_plant * total_plants) :
  length = 10 ∧ width = 12 ∧ plants_per_sqft = 5 ∧ yield_per_plant = 8 → 
  expected_strawberries = 4800 := by
  sorry

end strawberry_harvest_l2234_223422


namespace divisible_by_5_last_digit_l2234_223448

theorem divisible_by_5_last_digit (B : ℕ) (h : B < 10) : (∃ k : ℕ, 5270 + B = 5 * k) ↔ B = 0 ∨ B = 5 :=
by sorry

end divisible_by_5_last_digit_l2234_223448


namespace business_ownership_l2234_223415

variable (x : ℝ) (total_value : ℝ)
variable (fraction_sold : ℝ)
variable (sale_amount : ℝ)

-- Conditions
axiom total_value_condition : total_value = 10000
axiom fraction_sold_condition : fraction_sold = 3 / 5
axiom sale_amount_condition : sale_amount = 2000
axiom equation_condition : (fraction_sold * x * total_value = sale_amount)

theorem business_ownership : x = 1 / 3 := by 
  have hv := total_value_condition
  have hf := fraction_sold_condition
  have hs := sale_amount_condition
  have he := equation_condition
  sorry

end business_ownership_l2234_223415


namespace subset_A_B_l2234_223449

def A := {x : ℝ | 1 ≤ x ∧ x ≤ 2} -- Definition of set A
def B (a : ℝ) := {x : ℝ | x > a} -- Definition of set B

theorem subset_A_B (a : ℝ) : a < 1 → A ⊆ B a :=
by
  sorry

end subset_A_B_l2234_223449


namespace men_entered_room_l2234_223475

theorem men_entered_room (M W x : ℕ) 
  (h1 : M / W = 4 / 5) 
  (h2 : M + x = 14) 
  (h3 : 2 * (W - 3) = 24) 
  (h4 : 14 = 14) 
  (h5 : 24 = 24) : x = 2 := 
by 
  sorry

end men_entered_room_l2234_223475


namespace remainder_of_k_l2234_223412

theorem remainder_of_k {k : ℕ} (h1 : k % 5 = 2) (h2 : k % 6 = 5) (h3 : k % 8 = 7) (h4 : k % 11 = 3) (h5 : k < 168) :
  k % 13 = 8 := 
sorry

end remainder_of_k_l2234_223412


namespace find_price_of_stock_A_l2234_223473

-- Define conditions
def stock_investment_A (price_A : ℝ) : Prop := 
  ∃ (income_A: ℝ), income_A = 0.10 * 100

def stock_investment_B (price_B : ℝ) (investment_B : ℝ) : Prop := 
  price_B = 115.2 ∧ investment_B = 10 / 0.12

-- The main goal statement
theorem find_price_of_stock_A 
  (price_A : ℝ) (investment_B : ℝ) 
  (hA : stock_investment_A price_A) 
  (hB : stock_investment_B price_A investment_B) :
  price_A = 138.24 := 
sorry

end find_price_of_stock_A_l2234_223473


namespace rectangle_diagonals_equal_l2234_223420

-- Define the properties of a rectangle
def is_rectangle (AB CD AD BC : ℝ) (diagonal1 diagonal2 : ℝ) : Prop :=
  AB = CD ∧ AD = BC ∧ diagonal1 = diagonal2

-- State the theorem to prove that the diagonals of a rectangle are equal
theorem rectangle_diagonals_equal (AB CD AD BC diagonal1 diagonal2 : ℝ) (h : is_rectangle AB CD AD BC diagonal1 diagonal2) :
  diagonal1 = diagonal2 :=
by
  sorry

end rectangle_diagonals_equal_l2234_223420


namespace dividend_from_tonys_stock_l2234_223421

theorem dividend_from_tonys_stock (investment price_per_share total_income : ℝ) 
  (h1 : investment = 3200) (h2 : price_per_share = 85) (h3 : total_income = 250) : 
  (total_income / (investment / price_per_share)) = 6.76 :=
by 
  sorry

end dividend_from_tonys_stock_l2234_223421


namespace factorization_problem1_factorization_problem2_l2234_223483

-- Mathematical statements
theorem factorization_problem1 (x y : ℝ) : 2 * x^2 * y - 8 * x * y + 8 * y = 2 * y * (x - 2)^2 := by
  sorry

theorem factorization_problem2 (a : ℝ) : 18 * a^2 - 50 = 2 * (3 * a + 5) * (3 * a - 5) := by
  sorry

end factorization_problem1_factorization_problem2_l2234_223483


namespace no_solution_for_parallel_lines_values_of_a_for_perpendicular_lines_l2234_223492

-- Problem 1: There is no value of m that makes the lines parallel.
theorem no_solution_for_parallel_lines (m : ℝ) :
  ¬ ∃ m, (2 * m^2 + m - 3) / (m^2 - m) = 1 := sorry

-- Problem 2: The values of a that make the lines perpendicular.
theorem values_of_a_for_perpendicular_lines (a : ℝ) :
  (a = 1 ∨ a = -3) ↔ (a * (a - 1) + (1 - a) * (2 * a + 3) = 0) := sorry

end no_solution_for_parallel_lines_values_of_a_for_perpendicular_lines_l2234_223492


namespace circle_inequality_l2234_223424

-- Given a circle of 100 pairwise distinct numbers a : ℕ → ℝ for 1 ≤ i ≤ 100
variables {a : ℕ → ℝ}
-- Hypothesis 1: distinct numbers
def distinct_numbers (a : ℕ → ℝ) := ∀ i j : ℕ, (1 ≤ i ∧ i ≤ 100) ∧ (1 ≤ j ∧ j ≤ 100) ∧ (i ≠ j) → a i ≠ a j

-- Theorem: Prove that there exist four consecutive numbers such that the sum of the first and the last number is strictly greater than the sum of the two middle numbers
theorem circle_inequality (h_distinct : distinct_numbers a) : 
  ∃ i : ℕ, (1 ≤ i ∧ i ≤ 100) ∧ (a i + a ((i + 3) % 100) > a ((i + 1) % 100) + a ((i + 2) % 100)) :=
sorry

end circle_inequality_l2234_223424


namespace phi_range_l2234_223427

noncomputable def f (x : ℝ) (φ : ℝ) : ℝ := 2 * Real.sin (2 * x + φ) + 1

theorem phi_range (φ : ℝ) : 
  (|φ| ≤ Real.pi / 2) ∧ 
  (∀ x ∈ Set.Ioo (Real.pi / 24) (Real.pi / 3), f x φ > 2) →
  (Real.pi / 12 ≤ φ ∧ φ ≤ Real.pi / 6) :=
by
  sorry

end phi_range_l2234_223427


namespace num_primes_with_squares_in_range_l2234_223434

/-- There are exactly 6 prime numbers whose squares are between 2500 and 5500. -/
theorem num_primes_with_squares_in_range : 
  ∃ primes : Finset ℕ, 
    (∀ p ∈ primes, Prime p) ∧
    (∀ p ∈ primes, 2500 < p^2 ∧ p^2 < 5500) ∧
    primes.card = 6 :=
by
  sorry

end num_primes_with_squares_in_range_l2234_223434


namespace time_spent_answering_questions_l2234_223447

theorem time_spent_answering_questions (total_questions answered_per_question_minutes unanswered_questions : ℕ) (minutes_per_hour : ℕ) :
  total_questions = 100 → unanswered_questions = 40 → answered_per_question_minutes = 2 → minutes_per_hour = 60 → 
  ((total_questions - unanswered_questions) * answered_per_question_minutes) / minutes_per_hour = 2 :=
by
  intros h1 h2 h3 h4
  sorry

end time_spent_answering_questions_l2234_223447


namespace exists_root_between_roots_l2234_223470

theorem exists_root_between_roots 
  (a b c : ℝ) 
  (h_a : a ≠ 0) 
  (x₁ x₂ : ℝ) 
  (h₁ : a * x₁^2 + b * x₁ + c = 0) 
  (h₂ : -a * x₂^2 + b * x₂ + c = 0) 
  (hx : x₁ < x₂) :
  ∃ x₃ : ℝ, x₁ < x₃ ∧ x₃ < x₂ ∧ (a / 2) * x₃^2 + b * x₃ + c = 0 :=
by 
  sorry

end exists_root_between_roots_l2234_223470


namespace limit_does_not_exist_l2234_223499

noncomputable def does_not_exist_limit : Prop := 
  ¬ ∃ l : ℝ, ∀ ε > 0, ∃ δ > 0, ∀ x y : ℝ,
    (0 < |x| ∧ 0 < |y| ∧ |x| < δ ∧ |y| < δ) →
    |(x^2 - y^2) / (x^2 + y^2) - l| < ε

theorem limit_does_not_exist :
  does_not_exist_limit :=
sorry

end limit_does_not_exist_l2234_223499


namespace saved_percent_correct_l2234_223441

noncomputable def price_kit : ℝ := 144.20
noncomputable def price1 : ℝ := 21.75
noncomputable def price2 : ℝ := 18.60
noncomputable def price3 : ℝ := 23.80
noncomputable def price4 : ℝ := 29.35

noncomputable def total_price_individual : ℝ := 2 * price1 + 2 * price2 + price3 + 2 * price4
noncomputable def amount_saved : ℝ := total_price_individual - price_kit
noncomputable def percent_saved : ℝ := 100 * (amount_saved / total_price_individual)

theorem saved_percent_correct : percent_saved = 11.64 := by
  sorry

end saved_percent_correct_l2234_223441


namespace percentage_of_students_chose_spring_is_10_l2234_223417

-- Define the constants given in the problem
def total_students : ℕ := 10
def students_spring : ℕ := 1

-- Define the percentage calculation formula
def percentage (part total : ℕ) : ℕ := (part * 100) / total

-- State the theorem
theorem percentage_of_students_chose_spring_is_10 :
  percentage students_spring total_students = 10 :=
by
  -- We don't need to provide a proof here, just state it.
  sorry

end percentage_of_students_chose_spring_is_10_l2234_223417


namespace sequence_a_n_general_formula_and_value_sequence_b_n_general_formula_l2234_223490

theorem sequence_a_n_general_formula_and_value (a : ℕ → ℕ) 
  (h1 : a 1 = 3) 
  (h10 : a 10 = 21) 
  (h_linear : ∃ (k b : ℕ), ∀ n, a n = k * n + b) :
  (∀ n, a n = 2 * n + 1) ∧ a 2005 = 4011 :=
by 
  sorry

theorem sequence_b_n_general_formula (a b : ℕ → ℕ)
  (h_seq_a : ∀ n, a n = 2 * n + 1) 
  (h_b_formed : ∀ n, b n = a (2 * n)) : 
  ∀ n, b n = 4 * n + 1 :=
by 
  sorry

end sequence_a_n_general_formula_and_value_sequence_b_n_general_formula_l2234_223490


namespace count_and_largest_special_numbers_l2234_223465

def is_prime (n : ℕ) : Prop := 
  (n > 1) ∧ (∀ m : ℕ, m ∣ n → m = 1 ∨ m = n)

def is_four_digit_number (n : ℕ) : Prop := 
  1000 ≤ n ∧ n < 10000

theorem count_and_largest_special_numbers :
  ∃ (nums : List ℕ), 
    (∀ n ∈ nums, ∃ x y : ℕ, is_prime x ∧ is_prime y ∧ 
      55 * x * y = n ∧ is_four_digit_number (n * 5))
    ∧ nums.length = 3
    ∧ nums.maximum = some 4785 :=
sorry

end count_and_largest_special_numbers_l2234_223465


namespace intersection_of_A_and_B_l2234_223435

def A := {x : ℝ | x^2 - 5 * x + 6 > 0}
def B := {x : ℝ | x / (x - 1) < 0}

theorem intersection_of_A_and_B : A ∩ B = {x : ℝ | 0 < x ∧ x < 1} :=
by sorry

end intersection_of_A_and_B_l2234_223435


namespace side_length_square_l2234_223477

-- Define the length and width of the rectangle
def length_rect := 10 -- cm
def width_rect := 8 -- cm

-- Define the perimeter of the rectangle
def perimeter_rect := 2 * (length_rect + width_rect)

-- Define the perimeter of the square
def perimeter_square (s : ℕ) := 4 * s

-- The theorem to prove
theorem side_length_square : ∃ s : ℕ, perimeter_rect = perimeter_square s ∧ s = 9 :=
by
  sorry

end side_length_square_l2234_223477


namespace max_k_value_l2234_223496

noncomputable def f (x : ℝ) := x + x * Real.log x

theorem max_k_value : ∃ k : ℤ, (∀ x > 2, k * (x - 2) < f x) ∧ k = 4 :=
by
  sorry

end max_k_value_l2234_223496


namespace total_stamps_is_38_l2234_223432

-- Definitions based directly on conditions
def snowflake_stamps := 11
def truck_stamps := snowflake_stamps + 9
def rose_stamps := truck_stamps - 13
def total_stamps := snowflake_stamps + truck_stamps + rose_stamps

-- Statement to be proved
theorem total_stamps_is_38 : total_stamps = 38 := 
by 
  sorry

end total_stamps_is_38_l2234_223432


namespace tyrone_gave_15_marbles_l2234_223400

variables (x : ℕ)

-- Define initial conditions for Tyrone and Eric
def initial_tyrone := 120
def initial_eric := 20

-- Define the condition after giving marbles
def condition_after_giving (x : ℕ) := 120 - x = 3 * (20 + x)

theorem tyrone_gave_15_marbles (x : ℕ) : condition_after_giving x → x = 15 :=
by
  intro h
  sorry

end tyrone_gave_15_marbles_l2234_223400


namespace find_speed_of_stream_l2234_223423

def boat_speeds (V_b V_s : ℝ) : Prop :=
  V_b + V_s = 10 ∧ V_b - V_s = 8

theorem find_speed_of_stream (V_b V_s : ℝ) (h : boat_speeds V_b V_s) : V_s = 1 :=
by
  sorry

end find_speed_of_stream_l2234_223423


namespace midpoint_on_hyperbola_l2234_223402

theorem midpoint_on_hyperbola (A B : ℝ × ℝ) 
  (hA : A.1^2 - (A.2^2 / 9) = 1) 
  (hB : B.1^2 - (B.2^2 / 9) = 1) 
  (M : ℝ × ℝ) 
  (hM : M = (1/2 * (A.1 + B.1), 1/2 * (A.2 + B.2)))
  (M_options : M = (1,1) ∨ M = (-1,2) ∨ M = (1,3) ∨ M = (-1,-4)) :
  M = (-1, -4) :=
by
  sorry

end midpoint_on_hyperbola_l2234_223402


namespace tan_alpha_l2234_223487

variable (α : ℝ)

theorem tan_alpha (h₁ : Real.sin α = -5/13) (h₂ : 0 < α ∧ α < 2 * Real.pi ∧ α > 3 * Real.pi / 2) :
  Real.tan α = -5/12 :=
sorry

end tan_alpha_l2234_223487


namespace single_elimination_games_l2234_223463

theorem single_elimination_games (n : ℕ) (h : n = 512) : 
  ∃ g : ℕ, g = n - 1 ∧ g = 511 := 
by
  use n - 1
  sorry

end single_elimination_games_l2234_223463


namespace total_paintable_area_is_2006_l2234_223408

-- Define the dimensions of the bedrooms and the hallway
def bedroom_length := 14
def bedroom_width := 11
def bedroom_height := 9

def hallway_length := 20
def hallway_width := 7
def hallway_height := 9

def num_bedrooms := 4
def doorway_window_area := 70

-- Compute the areas of the bedroom walls and the hallway walls
def bedroom_wall_area : ℕ :=
  2 * (bedroom_length * bedroom_height) +
  2 * (bedroom_width * bedroom_height)

def paintable_bedroom_wall_area : ℕ :=
  bedroom_wall_area - doorway_window_area

def total_paintable_bedroom_area : ℕ :=
  num_bedrooms * paintable_bedroom_wall_area

def hallway_wall_area : ℕ :=
  2 * (hallway_length * hallway_height) +
  2 * (hallway_width * hallway_height)

-- Compute the total paintable area
def total_paintable_area : ℕ :=
  total_paintable_bedroom_area + hallway_wall_area

-- Theorem stating the total paintable area is 2006 sq ft
theorem total_paintable_area_is_2006 : total_paintable_area = 2006 := 
  by
    unfold total_paintable_area
    rw [total_paintable_bedroom_area, paintable_bedroom_wall_area, bedroom_wall_area]
    rw [hallway_wall_area]
    norm_num
    sorry -- Proof omitted

end total_paintable_area_is_2006_l2234_223408


namespace water_inflow_rate_in_tank_A_l2234_223405

-- Definitions from the conditions
def capacity := 20
def inflow_rate_B := 4
def extra_time_A := 5

-- Target variable
noncomputable def inflow_rate_A : ℕ :=
  let time_B := capacity / inflow_rate_B
  let time_A := time_B + extra_time_A
  capacity / time_A

-- Hypotheses
def tank_capacity : capacity = 20 := rfl
def tank_B_inflow : inflow_rate_B = 4 := rfl
def tank_A_extra_time : extra_time_A = 5 := rfl

-- Theorem statement
theorem water_inflow_rate_in_tank_A : inflow_rate_A = 2 := by
  -- Proof would go here
  sorry

end water_inflow_rate_in_tank_A_l2234_223405


namespace combined_experience_is_correct_l2234_223469

-- Define the conditions as given in the problem
def james_experience : ℕ := 40
def partner_less_years : ℕ := 10
def partner_experience : ℕ := james_experience - partner_less_years

-- The combined experience of James and his partner
def combined_experience : ℕ := james_experience + partner_experience

-- Lean statement to prove the combined experience is 70 years
theorem combined_experience_is_correct : combined_experience = 70 := by sorry

end combined_experience_is_correct_l2234_223469


namespace total_experiments_non_adjacent_l2234_223484

theorem total_experiments_non_adjacent (n_org n_inorg n_add : ℕ) 
  (h_org : n_org = 3) (h_inorg : n_inorg = 2) (h_add : n_add = 2) 
  (no_adjacent : True) : 
  (n_org + n_inorg + n_add).factorial / (n_inorg + n_add).factorial * 
  (n_inorg + n_add + 1).choose n_org = 1440 :=
by
  -- The actual proof will go here.
  sorry

end total_experiments_non_adjacent_l2234_223484


namespace total_rectangles_l2234_223425

-- Definitions
def horizontal_lines : ℕ := 5
def vertical_lines : ℕ := 4
def exclude_line_pair: ℕ := 1
def total_combinations (n m : ℕ) : ℕ := Nat.choose n m

-- Statement
theorem total_rectangles (h_lines : ℕ) (v_lines : ℕ) 
  (exclude_pair : ℕ) (valid_h_comb : ℕ) (valid_v_comb : ℕ) :
  h_lines = horizontal_lines →
  v_lines = vertical_lines →
  exclude_pair = exclude_line_pair →
  valid_h_comb = total_combinations 5 2 - exclude_pair →
  valid_v_comb = total_combinations 4 2 →
  valid_h_comb * valid_v_comb = 54 :=
by intros; sorry

end total_rectangles_l2234_223425


namespace point_distance_from_origin_l2234_223497

theorem point_distance_from_origin (x y m : ℝ) (h1 : |y| = 15) (h2 : (x - 2)^2 + (y - 7)^2 = 169) (h3 : x > 2) :
  m = Real.sqrt (334 + 4 * Real.sqrt 105) :=
sorry

end point_distance_from_origin_l2234_223497


namespace winter_sales_l2234_223452

theorem winter_sales (spring_sales summer_sales fall_sales : ℕ) (fall_sales_pct : ℝ) (total_sales winter_sales : ℕ) :
  spring_sales = 6 →
  summer_sales = 7 →
  fall_sales = 5 →
  fall_sales_pct = 0.20 →
  fall_sales = ⌊fall_sales_pct * total_sales⌋ →
  total_sales = spring_sales + summer_sales + fall_sales + winter_sales →
  winter_sales = 7 :=
by
  sorry

end winter_sales_l2234_223452


namespace right_triangle_longer_leg_l2234_223406

theorem right_triangle_longer_leg (a b c : ℕ) (h₀ : a^2 + b^2 = c^2) (h₁ : c = 65) (h₂ : a < b) : b = 60 :=
sorry

end right_triangle_longer_leg_l2234_223406


namespace time_after_2500_minutes_l2234_223459

/-- 
To prove that adding 2500 minutes to midnight on January 1, 2011 results in 
January 2 at 5:40 PM.
-/
theorem time_after_2500_minutes :
  let minutes_in_a_day := 1440 -- 24 hours * 60 minutes
  let minutes_in_an_hour := 60
  let start_time_minutes := 0 -- Midnight January 1, 2011 as zero minutes
  let total_minutes := 2500
  let resulting_minutes := start_time_minutes + total_minutes
  let days_passed := resulting_minutes / minutes_in_a_day
  let remaining_minutes := resulting_minutes % minutes_in_a_day
  let hours := remaining_minutes / minutes_in_an_hour
  let minutes := remaining_minutes % minutes_in_an_hour
  days_passed = 1 ∧ hours = 17 ∧ minutes = 40 :=
by
  -- Proof to be filled in
  sorry

end time_after_2500_minutes_l2234_223459


namespace sin_and_tan_sin_add_pi_over_4_and_tan_2alpha_l2234_223460

variable {α : ℝ} (h_cos : Real.cos α = -4/5) (h_quadrant : π < α ∧ α < 3 * π / 2)

theorem sin_and_tan (h_cos : Real.cos α = -4/5) (h_quadrant : π < α ∧ α < 3 * π / 2) :
  Real.sin α = -3/5 ∧ Real.tan α = 3/4 :=
sorry

theorem sin_add_pi_over_4_and_tan_2alpha (h_cos : Real.cos α = -4/5) (h_quadrant : π < α ∧ α < 3 * π / 2)
  (h_sin : Real.sin α = -3/5) (h_tan : Real.tan α = 3/4) :
  Real.sin (α + π/4) = -7 * Real.sqrt 2 / 10 ∧ Real.tan (2 * α) = 24/7 :=
sorry

end sin_and_tan_sin_add_pi_over_4_and_tan_2alpha_l2234_223460


namespace union_of_A_and_B_l2234_223476

open Set

variable {x : ℝ}

-- Define sets A and B based on the given conditions
def A : Set ℝ := { x | 0 < 3 - x ∧ 3 - x ≤ 2 }
def B : Set ℝ := { x | 0 ≤ x ∧ x ≤ 2 }

-- The theorem to prove
theorem union_of_A_and_B : A ∪ B = { x | 0 ≤ x ∧ x < 3 } := 
by 
  sorry

end union_of_A_and_B_l2234_223476


namespace maize_storage_l2234_223445

theorem maize_storage (x : ℝ)
  (h1 : 24 * x - 5 + 8 = 27) : x = 1 :=
  sorry

end maize_storage_l2234_223445


namespace remainder_when_xy_div_by_22_l2234_223433

theorem remainder_when_xy_div_by_22
  (x y : ℤ)
  (h1 : x % 126 = 37)
  (h2 : y % 176 = 46) : 
  (x + y) % 22 = 21 := by
  sorry

end remainder_when_xy_div_by_22_l2234_223433


namespace avocados_per_serving_l2234_223458

-- Definitions for the conditions
def original_avocados : ℕ := 5
def additional_avocados : ℕ := 4
def total_avocados : ℕ := original_avocados + additional_avocados
def servings : ℕ := 3

-- Theorem stating the result
theorem avocados_per_serving : (total_avocados / servings) = 3 :=
by
  sorry

end avocados_per_serving_l2234_223458


namespace equal_values_on_plane_l2234_223410

theorem equal_values_on_plane (f : ℤ × ℤ → ℕ)
    (h_avg : ∀ (i j : ℤ), f (i, j) = (f (i+1, j) + f (i-1, j) + f (i, j+1) + f (i, j-1)) / 4) :
  ∃ c : ℕ, ∀ (i j : ℤ), f (i, j) = c :=
by
  sorry

end equal_values_on_plane_l2234_223410


namespace peggy_stamps_l2234_223464

-- Defining the number of stamps Peggy, Ernie, and Bert have
variables (P : ℕ) (E : ℕ) (B : ℕ)

-- Given conditions
def bert_has_four_times_ernie (B : ℕ) (E : ℕ) : Prop := B = 4 * E
def ernie_has_three_times_peggy (E : ℕ) (P : ℕ) : Prop := E = 3 * P
def peggy_needs_stamps (P : ℕ) (B : ℕ) : Prop := B = P + 825

-- Question to Answer / Theorem Statement
theorem peggy_stamps (P : ℕ) (E : ℕ) (B : ℕ)
  (h1 : bert_has_four_times_ernie B E)
  (h2 : ernie_has_three_times_peggy E P)
  (h3 : peggy_needs_stamps P B) :
  P = 75 :=
sorry

end peggy_stamps_l2234_223464


namespace range_of_a_l2234_223436

theorem range_of_a (a : ℝ) : (∀ x : ℝ, x ≠ 0 → |x + 1/x| > |a - 2| + 1) ↔ 1 < a ∧ a < 3 :=
by
  sorry

end range_of_a_l2234_223436


namespace ratio_is_five_ninths_l2234_223442

-- Define the conditions
def total_profit : ℕ := 48000
def total_income : ℕ := 108000

-- Define the total spending based on conditions
def total_spending : ℕ := total_income - total_profit

-- Define the ratio of spending to income
def ratio_spending_to_income : ℚ := total_spending / total_income

-- The theorem we need to prove
theorem ratio_is_five_ninths : ratio_spending_to_income = 5 / 9 := 
  sorry

end ratio_is_five_ninths_l2234_223442


namespace quadratic_equation_with_given_roots_l2234_223407

theorem quadratic_equation_with_given_roots :
  (∃ (x : ℝ), (x - 3) * (x + 4) = 0 ↔ x = 3 ∨ x = -4) :=
by
  sorry

end quadratic_equation_with_given_roots_l2234_223407


namespace isosceles_triangle_perimeter_l2234_223438

theorem isosceles_triangle_perimeter (a b : ℝ) (h1 : a^2 - 9 * a + 18 = 0) (h2 : b^2 - 9 * b + 18 = 0) (h3 : a ≠ b) :
  a + 2 * b = 15 :=
by
  -- Proof is omitted.
  sorry

end isosceles_triangle_perimeter_l2234_223438


namespace sum_ab_l2234_223466

theorem sum_ab (a b : ℕ) (h1 : 1 < b) (h2 : a ^ b < 500) (h3 : ∀ x y : ℕ, (1 < y ∧ x ^ y < 500 ∧ (x + y) % 2 = 0) → a ^ b ≥ x ^ y) (h4 : (a + b) % 2 = 0) : a + b = 24 :=
  sorry

end sum_ab_l2234_223466


namespace Yoongi_has_fewest_apples_l2234_223416

def Jungkook_apples : Nat := 6 * 3
def Yoongi_apples : Nat := 4
def Yuna_apples : Nat := 5

theorem Yoongi_has_fewest_apples :
  Yoongi_apples < Jungkook_apples ∧ Yoongi_apples < Yuna_apples :=
by
  sorry

end Yoongi_has_fewest_apples_l2234_223416


namespace greatest_multiple_of_4_l2234_223454

/-- 
Given x is a positive multiple of 4 and x^3 < 2000, 
prove that x is at most 12 and 
x = 12 is the greatest value that satisfies these conditions. 
-/
theorem greatest_multiple_of_4 (x : ℕ) (hx1 : x % 4 = 0) (hx2 : x^3 < 2000) : x ≤ 12 ∧ x = 12 :=
by
  sorry

end greatest_multiple_of_4_l2234_223454


namespace sequence_term_500_l2234_223467

theorem sequence_term_500 :
  ∃ (a : ℕ → ℤ), 
  a 1 = 1001 ∧
  a 2 = 1005 ∧
  (∀ n, 1 ≤ n → (a n + a (n+1) + a (n+2)) = 2 * n) → 
  a 500 = 1334 := 
sorry

end sequence_term_500_l2234_223467


namespace cylinder_volume_ratio_l2234_223489

theorem cylinder_volume_ratio (r1 r2 V1 V2 : ℝ) (h1 : 2 * Real.pi * r1 = 6) (h2 : 2 * Real.pi * r2 = 10) (hV1 : V1 = Real.pi * r1^2 * 10) (hV2 : V2 = Real.pi * r2^2 * 6) :
  V1 < V2 → (V2 / V1) = 5 / 3 :=
by
  sorry

end cylinder_volume_ratio_l2234_223489


namespace figure_perimeter_l2234_223462

theorem figure_perimeter (h_segments v_segments : ℕ) (side_length : ℕ) 
  (h_count : h_segments = 16) (v_count : v_segments = 10) (side_len : side_length = 1) :
  2 * (h_segments + v_segments) * side_length = 26 :=
by
  sorry

end figure_perimeter_l2234_223462


namespace divisibility_by_37_l2234_223474

def sum_of_segments (n : ℕ) : ℕ :=
  let rec split_and_sum (num : ℕ) (acc : ℕ) : ℕ :=
    if num < 1000 then acc + num
    else split_and_sum (num / 1000) (acc + num % 1000)
  split_and_sum n 0

theorem divisibility_by_37 (A : ℕ) : 
  (37 ∣ A) ↔ (37 ∣ sum_of_segments A) :=
sorry

end divisibility_by_37_l2234_223474


namespace average_speed_l2234_223491

-- Define the conditions given in the problem
def distance_first_hour : ℕ := 50 -- distance traveled in the first hour
def distance_second_hour : ℕ := 60 -- distance traveled in the second hour
def total_distance : ℕ := distance_first_hour + distance_second_hour -- total distance traveled

-- Define the total time
def total_time : ℕ := 2 -- total time in hours

-- The problem statement: proving the average speed
theorem average_speed : total_distance / total_time = 55 := by
  unfold total_distance total_time
  sorry

end average_speed_l2234_223491


namespace smaller_third_angle_l2234_223485

theorem smaller_third_angle (x y : ℕ) (h₁ : x = 64) 
  (h₂ : 2 * x + (x - y) = 180) : y = 12 :=
by
  sorry

end smaller_third_angle_l2234_223485


namespace tan_periodic_example_l2234_223486

theorem tan_periodic_example : Real.tan (13 * Real.pi / 4) = 1 := 
by 
  sorry

end tan_periodic_example_l2234_223486


namespace initial_percentage_water_l2234_223414

theorem initial_percentage_water (P : ℝ) (H1 : 150 * P / 100 + 10 = 40) : P = 20 :=
by
  sorry

end initial_percentage_water_l2234_223414


namespace price_reduction_equation_l2234_223488

variable (x : ℝ)

theorem price_reduction_equation (h : 25 * (1 - x) ^ 2 = 16) : 25 * (1 - x) ^ 2 = 16 :=
by
  assumption

end price_reduction_equation_l2234_223488


namespace trigonometric_operation_l2234_223429

theorem trigonometric_operation :
  let m := Real.cos (Real.pi / 6)
  let n := Real.sin (Real.pi / 6)
  let op (m n : ℝ) := m^2 - m * n - n^2
  op m n = (1 / 2 : ℝ) - (Real.sqrt 3 / 4) :=
by
  sorry

end trigonometric_operation_l2234_223429


namespace coefficients_sum_l2234_223494

theorem coefficients_sum :
  let A := 3
  let B := 14
  let C := 18
  let D := 19
  let E := 30
  A + B + C + D + E = 84 := by
  sorry

end coefficients_sum_l2234_223494


namespace blocks_per_friend_l2234_223456

theorem blocks_per_friend (total_blocks : ℕ) (friends : ℕ) (h1 : total_blocks = 28) (h2 : friends = 4) :
  total_blocks / friends = 7 :=
by
  sorry

end blocks_per_friend_l2234_223456


namespace find_arrays_l2234_223498

-- Defines a condition where positive integers satisfy the given properties
def satisfies_conditions (a b c d : ℕ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ 
  a ∣ b * c * d - 1 ∧ 
  b ∣ a * c * d - 1 ∧ 
  c ∣ a * b * d - 1 ∧ 
  d ∣ a * b * c - 1

-- The theorem that any four positive integers satisfying the conditions are either (2, 3, 7, 11) or (2, 3, 11, 13)
theorem find_arrays :
  ∀ a b c d : ℕ, satisfies_conditions a b c d → 
    (a = 2 ∧ b = 3 ∧ c = 7 ∧ d = 11) ∨ 
    (a = 2 ∧ b = 3 ∧ c = 11 ∧ d = 13) :=
by
  intro a b c d h
  sorry

end find_arrays_l2234_223498


namespace a6_add_b6_geq_ab_a4_add_b4_l2234_223446

theorem a6_add_b6_geq_ab_a4_add_b4 (a b : ℝ) (ha : a ≥ 0) (hb : b ≥ 0) : 
  a^6 + b^6 ≥ ab * (a^4 + b^4) :=
sorry

end a6_add_b6_geq_ab_a4_add_b4_l2234_223446


namespace identified_rectangle_perimeter_l2234_223451

-- Define the side length of the square
def side_length_mm : ℕ := 75

-- Define the heights of the rectangles
variables (x y z : ℕ)

-- Define conditions
def rectangles_cut_condition (x y z : ℕ) : Prop := x + y + z = side_length_mm
def perimeter_relation_condition (x y z : ℕ) : Prop := 2 * (x + side_length_mm) = (y + side_length_mm) + (z + side_length_mm)

-- Define the perimeter of the identified rectangle
def identified_perimeter_mm (x : ℕ) := 2 * (x + side_length_mm)

-- Define conversion from mm to cm
def mm_to_cm (mm : ℕ) : ℕ := mm / 10

-- Final proof statement
theorem identified_rectangle_perimeter :
  ∃ x y z : ℕ, rectangles_cut_condition x y z ∧ perimeter_relation_condition x y z ∧ mm_to_cm (identified_perimeter_mm x) = 20 := 
sorry

end identified_rectangle_perimeter_l2234_223451


namespace lambda_sum_ellipse_l2234_223437

noncomputable def ellipse (x y : ℝ) : Prop :=
  x^2 / 25 + y^2 / 9 = 1

noncomputable def line_through_focus (k : ℝ) (x y : ℝ) : Prop :=
  y = k * (x - 4)

noncomputable def intersects_y_axis (k : ℝ) : ℝ × ℝ :=
  (0, -4 * k)

noncomputable def lambda1 (x1 : ℝ) : ℝ :=
  x1 / (4 - x1)

noncomputable def lambda2 (x2 : ℝ) : ℝ :=
  x2 / (4 - x2)

theorem lambda_sum_ellipse {k x1 x2 : ℝ}
  (h1 : ellipse x1 (k * (x1 - 4)))
  (h2 : ellipse x2 (k * (x2 - 4)))
  (h3 : line_through_focus k x1 (k * (x1 - 4)))
  (h4 : line_through_focus k x2 (k * (x2 - 4))) :
  lambda1 x1 + lambda2 x2 = -50 / 9 := 
sorry

end lambda_sum_ellipse_l2234_223437


namespace circle_radius_l2234_223457

-- Given the equation of a circle, we want to prove its radius
theorem circle_radius : ∀ (x y : ℝ), x^2 + y^2 - 6*y - 16 = 0 → (∃ r, r = 5) :=
  by
    sorry

end circle_radius_l2234_223457


namespace minimum_value_of_expression_l2234_223443

theorem minimum_value_of_expression :
  ∃ x y : ℝ, ∀ x y : ℝ, 3 * x^2 + 4 * x * y + 2 * y^2 - 6 * x + 4 * y + 7 ≥ 28 := by
  sorry

end minimum_value_of_expression_l2234_223443


namespace g_extreme_points_product_inequality_l2234_223495

noncomputable def f (a x : ℝ) : ℝ := (-x^2 + a * x - a) / Real.exp x

noncomputable def f' (a x : ℝ) : ℝ := (x^2 - (a + 2) * x + 2 * a) / Real.exp x

noncomputable def g (a x : ℝ) : ℝ := (f a x + f' a x) / (x - 1)

theorem g_extreme_points_product_inequality {a x1 x2 : ℝ} 
  (h_cond1 : a > 2)
  (h_cond2 : x1 + x2 = (a + 2) / 2)
  (h_cond3 : x1 * x2 = 1)
  (h_cond4 : x1 ≠ 1 ∧ x2 ≠ 1)
  (h_x1 : x1 ∈ (Set.Ioo 0 1 ∪ Set.Ioi 1))
  (h_x2 : x2 ∈ (Set.Ioo 0 1 ∪ Set.Ioi 1)) :
  g a x1 * g a x2 < 4 / Real.exp 2 :=
sorry

end g_extreme_points_product_inequality_l2234_223495


namespace part1_condition1_implies_a_le_1_condition2_implies_a_le_2_condition3_implies_a_le_1_l2234_223439

section Problem

-- Universal set is ℝ
def universal_set : Set ℝ := Set.univ

-- Set A
def set_A : Set ℝ := { x | x^2 - x - 6 ≤ 0 }

-- Set A complement in ℝ
def complement_A : Set ℝ := universal_set \ set_A

-- Set B
def set_B : Set ℝ := { x | (x - 4)/(x + 1) < 0 }

-- Set C
def set_C (a : ℝ) : Set ℝ := { x | 2 - a < x ∧ x < 2 + a }

-- Prove (complement_A ∩ set_B = (3, 4))
theorem part1 : (complement_A ∩ set_B) = { x | 3 < x ∧ x < 4 } :=
  sorry

-- Assume a definition for real number a (non-negative)
variable (a : ℝ)

-- Prove range of a given the conditions
-- Condition 1: A ∩ C = C implies a ≤ 1
theorem condition1_implies_a_le_1 (h : set_A ∩ set_C a = set_C a) : a ≤ 1 :=
  sorry

-- Condition 2: B ∪ C = B implies a ≤ 2
theorem condition2_implies_a_le_2 (h : set_B ∪ set_C a = set_B) : a ≤ 2 :=
  sorry

-- Condition 3: C ⊆ (A ∩ B) implies a ≤ 1
theorem condition3_implies_a_le_1 (h : set_C a ⊆ set_A ∩ set_B) : a ≤ 1 :=
  sorry

end Problem

end part1_condition1_implies_a_le_1_condition2_implies_a_le_2_condition3_implies_a_le_1_l2234_223439


namespace sheila_weekly_earnings_l2234_223450

theorem sheila_weekly_earnings:
  (∀(m w f : ℕ), (m = 8) → (w = 8) → (f = 8) → 
   ∀(t th : ℕ), (t = 6) → (th = 6) → 
   ∀(h : ℕ), (h = 6) → 
   (m + w + f + t + th) * h = 216) := by
  sorry

end sheila_weekly_earnings_l2234_223450


namespace michael_made_small_balls_l2234_223419

def num_small_balls (total_bands : ℕ) (bands_per_small : ℕ) (bands_per_large : ℕ) (num_large : ℕ) : ℕ :=
  (total_bands - num_large * bands_per_large) / bands_per_small

theorem michael_made_small_balls :
  num_small_balls 5000 50 300 13 = 22 :=
by
  sorry

end michael_made_small_balls_l2234_223419


namespace max_ab_bc_cd_l2234_223471

-- Definitions of nonnegative numbers and their sum condition
variables (a b c d : ℕ) 
variables (h_sum : a + b + c + d = 120)

-- The goal to prove
theorem max_ab_bc_cd : ab + bc + cd <= 3600 :=
sorry

end max_ab_bc_cd_l2234_223471


namespace cut_into_four_and_reassemble_l2234_223409

-- Definitions as per conditions in the problem
def figureArea : ℕ := 36
def nParts : ℕ := 4
def squareArea (s : ℕ) : ℕ := s * s

-- Property to be proved
theorem cut_into_four_and_reassemble :
  ∃ (s : ℕ), squareArea s = figureArea / nParts ∧ s * s = figureArea :=
by
  sorry

end cut_into_four_and_reassemble_l2234_223409


namespace range_of_cars_l2234_223444

def fuel_vehicle_cost_per_km (x : ℕ) : ℚ := (40 * 9) / x
def new_energy_vehicle_cost_per_km (x : ℕ) : ℚ := (60 * 0.6) / x

theorem range_of_cars : ∃ x : ℕ, fuel_vehicle_cost_per_km x = new_energy_vehicle_cost_per_km x + 0.54 ∧ x = 600 := 
by {
  sorry
}

end range_of_cars_l2234_223444


namespace repair_time_l2234_223428

theorem repair_time {x : ℝ} :
  (∀ (a b : ℝ), a = 3 ∧ b = 6 → (((1 / a) + (1 / b)) * x = 1) → x = 2) :=
by
  intros a b hab h
  rcases hab with ⟨ha, hb⟩
  sorry

end repair_time_l2234_223428


namespace sandy_books_second_shop_l2234_223411

theorem sandy_books_second_shop (x : ℕ) (h1 : 65 = 1080 / 16) 
                                (h2 : x * 16 = 840) 
                                (h3 : (1080 + 840) / 16 = 120) : 
                                x = 55 :=
by
  sorry

end sandy_books_second_shop_l2234_223411


namespace bill_initial_amount_l2234_223468

/-- Suppose Ann has $777 and Bill gives Ann $167,
    after which they both have the same amount of money. 
    Prove that Bill initially had $1111. -/
theorem bill_initial_amount (A B : ℕ) (h₁ : A = 777) (h₂ : B - 167 = A + 167) : B = 1111 :=
by
  -- Proof goes here
  sorry

end bill_initial_amount_l2234_223468


namespace net_profit_is_90_l2234_223493

theorem net_profit_is_90
    (cost_seeds cost_soil : ℝ)
    (num_plants : ℕ)
    (price_per_plant : ℝ)
    (h0 : cost_seeds = 2)
    (h1 : cost_soil = 8)
    (h2 : num_plants = 20)
    (h3 : price_per_plant = 5) :
    (num_plants * price_per_plant - (cost_seeds + cost_soil)) = 90 := by
  sorry

end net_profit_is_90_l2234_223493


namespace expr_eval_l2234_223482

theorem expr_eval : 180 / 6 * 2 + 5 = 65 := by
  sorry

end expr_eval_l2234_223482


namespace negative_integer_solution_l2234_223481

theorem negative_integer_solution (x : ℤ) (h : 3 * x + 13 ≥ 0) : x = -1 :=
by
  sorry

end negative_integer_solution_l2234_223481


namespace solve_determinant_l2234_223461

-- Definitions based on the conditions
def determinant (a b c d : ℤ) : ℤ := a * d - b * c

-- The problem translated to Lean 4:
theorem solve_determinant (x : ℤ) 
  (h : determinant (x + 1) x (2 * x - 6) (2 * (x - 1)) = 10) :
  x = 2 :=
sorry -- Proof is skipped

end solve_determinant_l2234_223461


namespace intersection_P_Q_l2234_223479

def P := {x : ℝ | 1 < x ∧ x < 3}
def Q := {x : ℝ | 2 < x}

theorem intersection_P_Q :
  P ∩ Q = {x : ℝ | 2 < x ∧ x < 3} := sorry

end intersection_P_Q_l2234_223479


namespace hours_worked_each_day_l2234_223403

-- Definitions based on problem conditions
def total_hours_worked : ℝ := 8.0
def number_of_days_worked : ℝ := 4.0

-- Theorem statement to prove the number of hours worked each day
theorem hours_worked_each_day :
  total_hours_worked / number_of_days_worked = 2.0 :=
sorry

end hours_worked_each_day_l2234_223403


namespace value_of_x_l2234_223455

theorem value_of_x (x : ℝ) (m : ℕ) (h1 : m = 31) :
  ((x ^ m) / (5 ^ m)) * ((x ^ 16) / (4 ^ 16)) = 1 / (2 * 10 ^ 31) → x = 1 := by
  sorry

end value_of_x_l2234_223455


namespace solve_eq1_solve_eq2_l2234_223440

theorem solve_eq1 (x : ℝ) : (12 * (x - 1) ^ 2 = 3) ↔ (x = 3/2 ∨ x = 1/2) := 
by sorry

theorem solve_eq2 (x : ℝ) : ((x + 1) ^ 3 = 0.125) ↔ (x = -0.5) := 
by sorry

end solve_eq1_solve_eq2_l2234_223440


namespace tan_neg_1140_eq_neg_sqrt3_l2234_223418

theorem tan_neg_1140_eq_neg_sqrt3 
  (tan_neg : ∀ θ : ℝ, Real.tan (-θ) = -Real.tan θ)
  (tan_periodicity : ∀ θ : ℝ, ∀ n : ℤ, Real.tan (θ + n * 180) = Real.tan θ)
  (tan_60 : Real.tan 60 = Real.sqrt 3) :
  Real.tan (-1140) = -Real.sqrt 3 := 
sorry

end tan_neg_1140_eq_neg_sqrt3_l2234_223418


namespace parabola_intersection_diff_l2234_223453

theorem parabola_intersection_diff (a b c d : ℝ) 
  (h₁ : ∀ x y, (3 * x^2 - 2 * x + 1 = y) → (c = x ∨ a = x))
  (h₂ : ∀ x y, (-2 * x^2 + 4 * x + 1 = y) → (c = x ∨ a = x))
  (h₃ : c ≥ a) :
  c - a = 6 / 5 :=
by sorry

end parabola_intersection_diff_l2234_223453


namespace cone_volume_l2234_223413

noncomputable def radius_of_sector : ℝ := 6
noncomputable def arc_length_of_sector : ℝ := (1 / 2) * (2 * Real.pi * radius_of_sector)
noncomputable def radius_of_base : ℝ := arc_length_of_sector / (2 * Real.pi)
noncomputable def slant_height : ℝ := radius_of_sector
noncomputable def height_of_cone : ℝ := Real.sqrt (slant_height^2 - radius_of_base^2)
noncomputable def volume_of_cone : ℝ := (1 / 3) * Real.pi * (radius_of_base^2) * height_of_cone

theorem cone_volume : volume_of_cone = 9 * Real.pi * Real.sqrt 3 := by
  sorry

end cone_volume_l2234_223413
