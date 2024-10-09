import Mathlib

namespace find_nearest_integer_x_minus_y_l1444_144463

variable (x y : ℝ)

theorem find_nearest_integer_x_minus_y
  (h1 : abs x + y = 5)
  (h2 : abs x * y - x^3 = 0)
  (hx : x ≠ 0)
  (hy : y ≠ 0) :
  |x - y| = 5 := sorry

end find_nearest_integer_x_minus_y_l1444_144463


namespace parabola_focus_coincides_hyperbola_focus_l1444_144447

theorem parabola_focus_coincides_hyperbola_focus (p : ℝ) : 
  (∀ x y : ℝ, y^2 = 2 * p * x -> (3,0) = (3,0)) → 
  (∀ x y : ℝ, x^2 / 6 - y^2 / 3 = 1 -> x = 3) → 
  p = 6 :=
by
  sorry

end parabola_focus_coincides_hyperbola_focus_l1444_144447


namespace range_of_x_l1444_144427

noncomputable def f : ℝ → ℝ := sorry  -- f is an even function and decreasing on [0, +∞)

theorem range_of_x (x : ℝ) (h_even : ∀ x, f x = f (-x)) 
  (h_decreasing : ∀ x y, 0 ≤ x → x ≤ y → f x ≥ f y) 
  (h_condition : f (Real.log x) > f 1) : 
  1 / 10 < x ∧ x < 10 := 
sorry

end range_of_x_l1444_144427


namespace division_quotient_l1444_144411

theorem division_quotient (dividend divisor remainder quotient : Nat) 
  (h_dividend : dividend = 109)
  (h_divisor : divisor = 12)
  (h_remainder : remainder = 1)
  (h_division_equation : dividend = divisor * quotient + remainder)
  : quotient = 9 := 
by
  sorry

end division_quotient_l1444_144411


namespace number_of_integers_with_6_or_7_as_digit_in_base9_l1444_144482

/-- 
  There are 729 smallest positive integers written in base 9.
  We want to determine how many of these integers use the digits 6 or 7 (or both) at least once.
-/
theorem number_of_integers_with_6_or_7_as_digit_in_base9 : 
  ∃ n : ℕ, n = 729 ∧ ∃ m : ℕ, m = n - 7^3 := sorry

end number_of_integers_with_6_or_7_as_digit_in_base9_l1444_144482


namespace dice_probability_sum_three_l1444_144466

theorem dice_probability_sum_three (total_outcomes : ℕ := 36) (favorable_outcomes : ℕ := 2) :
  favorable_outcomes / total_outcomes = 1 / 18 :=
by
  sorry

end dice_probability_sum_three_l1444_144466


namespace total_students_in_class_l1444_144497

theorem total_students_in_class
  (S : ℕ)
  (H1 : 5/8 * S = S - 60)
  (H2 : 60 = 3/8 * S) :
  S = 160 :=
by
  sorry

end total_students_in_class_l1444_144497


namespace store_profit_is_33_percent_l1444_144495

noncomputable def store_profit (C : ℝ) : ℝ :=
  let initial_markup := 1.20 * C
  let new_year_markup := initial_markup + 0.25 * initial_markup
  let february_discount := new_year_markup * 0.92
  let shipping_cost := C * 1.05
  (february_discount - shipping_cost)

theorem store_profit_is_33_percent (C : ℝ) : store_profit C = 0.33 * C :=
by
  sorry

end store_profit_is_33_percent_l1444_144495


namespace stratified_sampling_first_level_l1444_144437

-- Definitions from the conditions
def num_senior_teachers : ℕ := 90
def num_first_level_teachers : ℕ := 120
def num_second_level_teachers : ℕ := 170
def total_teachers : ℕ := num_senior_teachers + num_first_level_teachers + num_second_level_teachers
def sample_size : ℕ := 38

-- Definition of the stratified sampling result
def num_first_level_selected : ℕ := (num_first_level_teachers * sample_size) / total_teachers

-- The statement to be proven
theorem stratified_sampling_first_level : num_first_level_selected = 12 :=
by
  sorry

end stratified_sampling_first_level_l1444_144437


namespace passengers_on_board_l1444_144481

/-- 
Given the fractions of passengers from different continents and remaining 42 passengers,
show that the total number of passengers P is 240.
-/
theorem passengers_on_board :
  ∃ P : ℕ,
    (1 / 3) * (P : ℝ) + (1 / 8) * (P : ℝ) + (1 / 5) * (P : ℝ) + (1 / 6) * (P : ℝ) + 42 = (P : ℝ) ∧ P = 240 :=
by
  let P := 240
  have h : (1 / 3) * (P : ℝ) + (1 / 8) * (P : ℝ) + (1 / 5) * (P : ℝ) + (1 / 6) * (P : ℝ) + 42 = (P : ℝ) := sorry
  exact ⟨P, h, rfl⟩

end passengers_on_board_l1444_144481


namespace ratio_eq_one_l1444_144491

theorem ratio_eq_one {a b : ℝ} (h1 : 4 * a^2 = 5 * b^3) (h2 : a ≠ 0 ∧ b ≠ 0) : (a^2 / 5) / (b^3 / 4) = 1 :=
by
  sorry

end ratio_eq_one_l1444_144491


namespace train_crossing_time_l1444_144407

/-- Time for a train of length 1500 meters traveling at 108 km/h to cross an electric pole is 50 seconds -/
theorem train_crossing_time (length : ℕ) (speed_kmph : ℕ) 
    (h₁ : length = 1500) (h₂ : speed_kmph = 108) : 
    (length / ((speed_kmph * 1000) / 3600) = 50) :=
by
  sorry

end train_crossing_time_l1444_144407


namespace mary_change_in_dollars_l1444_144435

theorem mary_change_in_dollars :
  let cost_berries_euros := 7.94
  let cost_peaches_dollars := 6.83
  let exchange_rate := 1.2
  let money_handed_euros := 20
  let money_handed_dollars := 10
  let cost_berries_dollars := cost_berries_euros * exchange_rate
  let total_cost_dollars := cost_berries_dollars + cost_peaches_dollars
  let total_handed_dollars := (money_handed_euros * exchange_rate) + money_handed_dollars
  total_handed_dollars - total_cost_dollars = 17.642 :=
by
  intros
  sorry

end mary_change_in_dollars_l1444_144435


namespace factorize_expr_l1444_144468

theorem factorize_expr (x y : ℝ) : x^3 - 4 * x * y^2 = x * (x + 2 * y) * (x - 2 * y) :=
by
  sorry

end factorize_expr_l1444_144468


namespace sum_of_first_5_terms_is_55_l1444_144458

variable (a : ℕ → ℝ) -- the arithmetic sequence
variable (d : ℝ) -- the common difference
variable (a_2 : a 2 = 7)
variable (a_4 : a 4 = 15)
noncomputable def sum_of_first_5_terms : ℝ := (5 * (a 2 + a 4)) / 2

theorem sum_of_first_5_terms_is_55 :
  sum_of_first_5_terms a = 55 :=
by
  sorry

end sum_of_first_5_terms_is_55_l1444_144458


namespace no_six_digit_numbers_exists_l1444_144417

theorem no_six_digit_numbers_exists :
  ¬(∃ (N : Fin 6 → Fin 720), ∀ (a b c : Fin 6), a ≠ b → a ≠ c → b ≠ c →
  (∃ (i : Fin 6), N i == 720)) := sorry

end no_six_digit_numbers_exists_l1444_144417


namespace fraction_square_eq_decimal_l1444_144474

theorem fraction_square_eq_decimal :
  ∃ (x : ℚ), x^2 = 0.04000000000000001 ∧ x = 1 / 5 :=
by
  sorry

end fraction_square_eq_decimal_l1444_144474


namespace total_arms_collected_l1444_144460

-- Define the conditions as parameters
def arms_of_starfish := 7 * 5
def arms_of_seastar := 14

-- Define the theorem to prove total arms
theorem total_arms_collected : arms_of_starfish + arms_of_seastar = 49 := by
  sorry

end total_arms_collected_l1444_144460


namespace rachel_problems_solved_each_minute_l1444_144400

-- Definitions and conditions
def problems_solved_each_minute (x : ℕ) : Prop :=
  let problems_before_bed := 12 * x
  let problems_at_lunch := 16
  let total_problems := problems_before_bed + problems_at_lunch
  total_problems = 76

-- Theorem to be proved
theorem rachel_problems_solved_each_minute : ∃ x : ℕ, problems_solved_each_minute x ∧ x = 5 :=
by
  sorry

end rachel_problems_solved_each_minute_l1444_144400


namespace annual_increase_rate_l1444_144444

theorem annual_increase_rate (PV FV : ℝ) (n : ℕ) (r : ℝ) :
  PV = 32000 ∧ FV = 40500 ∧ n = 2 ∧ FV = PV * (1 + r)^2 → r = 0.125 :=
by
  sorry

end annual_increase_rate_l1444_144444


namespace min_colors_required_l1444_144408

-- Defining the color type
def Color := ℕ

-- Defining a 6x6 grid
def Grid := Fin 6 → Fin 6 → Color

-- Defining the conditions of the problem for a valid coloring
def is_valid_coloring (c : Grid) : Prop :=
  (∀ i j k, i ≠ j → c i k ≠ c j k) ∧ -- each row has all cells with different colors
  (∀ i j k, i ≠ j → c k i ≠ c k j) ∧ -- each column has all cells with different colors
  (∀ i j, i ≠ j → c i (i+j) ≠ c j (i+j)) ∧ -- each 45° diagonal has all different colors
  (∀ i j, i ≠ j → (i-j ≥ 0 → c (i-j) i ≠ c (i-j) j) ∧ (j-i ≥ 0 → c i (j-i) ≠ c j (j-i))) -- each 135° diagonal has all different colors

-- The formal statement of the math problem
theorem min_colors_required : ∃ (n : ℕ), (∀ c : Grid, is_valid_coloring c → n ≥ 7) :=
sorry

end min_colors_required_l1444_144408


namespace solve_for_x_l1444_144449

-- Definitions and conditions from a) directly 
def f (x : ℝ) : ℝ := 64 * (2 * x - 1) ^ 3

-- Lean 4 statement to prove the problem
theorem solve_for_x (x : ℝ) : f x = 27 → x = 7 / 8 :=
by
  intro h
  -- Placeholder for the actual proof
  sorry

end solve_for_x_l1444_144449


namespace domain_transform_l1444_144455

-- Definitions based on conditions
def domain_f_x_plus_1 : Set ℝ := { x | -2 ≤ x ∧ x ≤ 3 }
def domain_f_id : Set ℝ := { x | -1 ≤ x ∧ x ≤ 4 }
def domain_f_2x_minus_1 : Set ℝ := { x | 0 ≤ x ∧ x ≤ 5/2 }

-- The theorem to prove the mathematically equivalent problem
theorem domain_transform :
  (∀ x, (x + 1) ∈ domain_f_x_plus_1) →
  (∀ y, y ∈ domain_f_2x_minus_1 ↔ 2 * y - 1 ∈ domain_f_id) :=
by
  sorry

end domain_transform_l1444_144455


namespace seating_arrangement_l1444_144465

theorem seating_arrangement (x y : ℕ) (h : 9 * x + 6 * y = 57) : x = 1 :=
sorry

end seating_arrangement_l1444_144465


namespace foci_of_ellipse_l1444_144476

def ellipse_focus (x y : ℝ) : Prop :=
  (x = 0 ∧ (y = 12 ∨ y = -12))

theorem foci_of_ellipse :
  ∀ (x y : ℝ), (x^2)/25 + (y^2)/169 = 1 → ellipse_focus x y :=
by
  intros x y h
  sorry

end foci_of_ellipse_l1444_144476


namespace trigonometric_identity_proof_l1444_144461

theorem trigonometric_identity_proof (alpha : Real)
(h1 : Real.tan (alpha + π / 4) = 1 / 2)
(h2 : -π / 2 < alpha ∧ alpha < 0) :
  (2 * Real.sin alpha ^ 2 + Real.sin (2 * alpha)) / Real.cos (alpha - π / 4) = - (2 * Real.sqrt 5) / 5 :=
by
  sorry

end trigonometric_identity_proof_l1444_144461


namespace fence_perimeter_l1444_144415

-- Definitions for the given conditions
def num_posts : ℕ := 36
def gap_between_posts : ℕ := 6

-- Defining the proof problem to show the perimeter equals 192 feet
theorem fence_perimeter (n : ℕ) (g : ℕ) (sq_field : ℕ → ℕ → Prop) : n = 36 ∧ g = 6 ∧ sq_field n g → 4 * ((n / 4 - 1) * g) = 192 :=
by
  intro h
  sorry

end fence_perimeter_l1444_144415


namespace simplify_fraction_l1444_144483

theorem simplify_fraction (b : ℝ) (h : b ≠ 1) : 
  (b - 1) / (b + b / (b - 1)) = (b - 1) ^ 2 / b ^ 2 := 
by {
  sorry
}

end simplify_fraction_l1444_144483


namespace function_ordering_l1444_144478

-- Definitions for the function and conditions
variable (f : ℝ → ℝ)

-- Assuming properties of the function
axiom odd_function : ∀ x, f (-x) = -f x
axiom periodicity : ∀ x, f (x + 4) = -f x
axiom increasing_on : ∀ ⦃x y⦄, 0 ≤ x → x < y → y ≤ 2 → f x < f y

-- Main theorem statement
theorem function_ordering : f (-25) < f 80 ∧ f 80 < f 11 :=
by 
  sorry

end function_ordering_l1444_144478


namespace value_is_twenty_l1444_144494

theorem value_is_twenty (n : ℕ) (h : n = 16) : 32 - 12 = 20 :=
by {
  -- Simplification of the proof process
  sorry
}

end value_is_twenty_l1444_144494


namespace total_items_l1444_144423

theorem total_items (B M C : ℕ) 
  (h1 : B = 58) 
  (h2 : B = M + 18) 
  (h3 : B = C - 27) : 
  B + M + C = 183 :=
by 
  sorry

end total_items_l1444_144423


namespace solve_for_x_l1444_144439

theorem solve_for_x {x : ℝ} (h : -3 * x - 10 = 4 * x + 5) : x = -15 / 7 :=
  sorry

end solve_for_x_l1444_144439


namespace least_6_digit_number_sum_of_digits_l1444_144434

-- Definitions based on conditions
def is_6_digit (n : ℕ) : Prop := 100000 ≤ n ∧ n < 1000000

def leaves_remainder2 (n : ℕ) (d : ℕ) : Prop := n % d = 2

def sum_of_digits (n : ℕ) : ℕ :=
  (n.digits 10).sum

-- Problem statement
theorem least_6_digit_number_sum_of_digits :
  ∃ n : ℕ, is_6_digit n ∧ leaves_remainder2 n 4 ∧ leaves_remainder2 n 610 ∧ leaves_remainder2 n 15 ∧ sum_of_digits n = 17 :=
sorry

end least_6_digit_number_sum_of_digits_l1444_144434


namespace half_angle_second_quadrant_l1444_144445

theorem half_angle_second_quadrant (k : ℤ) (α : ℝ) (hα : 2 * k * π + π / 2 < α ∧ α < 2 * k * π + π) :
    ∃ j : ℤ, (j * π + π / 4 < α / 2 ∧ α / 2 < j * π + π / 2) ∨ (j * π + 5 * π / 4 < α / 2 ∧ α / 2 < (j + 1) * π / 2) :=
sorry

end half_angle_second_quadrant_l1444_144445


namespace quadratic_roots_and_expression_value_l1444_144457

theorem quadratic_roots_and_expression_value :
  let a := 3 + Real.sqrt 21
  let b := 3 - Real.sqrt 21
  (a ≥ b) →
  (∃ x : ℝ, x^2 - 6 * x + 11 = 23) →
  3 * a + 2 * b = 15 + Real.sqrt 21 :=
by
  intros a b h1 h2
  sorry

end quadratic_roots_and_expression_value_l1444_144457


namespace compute_expression_l1444_144420

theorem compute_expression :
  20 * (150 / 3 + 40 / 5 + 16 / 25 + 2) = 1212.8 :=
by
  -- skipping the proof steps
  sorry

end compute_expression_l1444_144420


namespace sqrt_inequality_l1444_144438

theorem sqrt_inequality : (Real.sqrt 3 + Real.sqrt 7) < 2 * Real.sqrt 5 := 
  sorry

end sqrt_inequality_l1444_144438


namespace lengths_of_angle_bisectors_areas_of_triangles_l1444_144472

-- Given conditions
variables (x y : ℝ) (S1 S2 : ℝ)
variables (hx1 : x + y = 15) (hx2 : x / y = 3 / 2)
variables (hS1 : S1 / S2 = 9 / 4) (hS2 : S1 - S2 = 6)

-- Prove the lengths of the angle bisectors
theorem lengths_of_angle_bisectors :
  x = 9 ∧ y = 6 :=
by sorry

-- Prove the areas of the triangles
theorem areas_of_triangles :
  S1 = 54 / 5 ∧ S2 = 24 / 5 :=
by sorry

end lengths_of_angle_bisectors_areas_of_triangles_l1444_144472


namespace perfect_squares_m_l1444_144469

theorem perfect_squares_m (m : ℕ) (hm_pos : m > 0) (hm_min4_square : ∃ a : ℕ, m - 4 = a^2) (hm_plus5_square : ∃ b : ℕ, m + 5 = b^2) : m = 20 ∨ m = 4 :=
by
  sorry

end perfect_squares_m_l1444_144469


namespace terminal_side_half_angle_l1444_144419

theorem terminal_side_half_angle {k : ℤ} {α : ℝ} 
  (h : 2 * k * π < α ∧ α < 2 * k * π + π / 2) : 
  (k * π < α / 2 ∧ α / 2 < k * π + π / 4) ∨ (k * π + π <= α / 2 ∧ α / 2 < (k + 1) * π + π / 4) :=
sorry

end terminal_side_half_angle_l1444_144419


namespace min_overlap_percentage_l1444_144484

theorem min_overlap_percentage (A B : ℝ) (hA : A = 0.9) (hB : B = 0.8) : ∃ x, x = 0.7 := 
by sorry

end min_overlap_percentage_l1444_144484


namespace Anne_cleaning_time_l1444_144488

theorem Anne_cleaning_time (B A C : ℚ) 
  (h1 : B + A + C = 1 / 6) 
  (h2 : B + 2 * A + 3 * C = 1 / 2)
  (h3 : B + A = 1 / 4)
  (h4 : B + C = 1 / 3) : 
  A = 1 / 6 := 
sorry

end Anne_cleaning_time_l1444_144488


namespace minimum_distinct_values_is_145_l1444_144409

-- Define the conditions
def n_series : ℕ := 2023
def unique_mode_occurrence : ℕ := 15

-- Define the minimum number of distinct values satisfying the conditions
def min_distinct_values (n : ℕ) (mode_count : ℕ) : ℕ :=
  if mode_count < n then 
    (n - mode_count + 13) / 14 + 1
  else
    1

-- The theorem restating the problem to be solved
theorem minimum_distinct_values_is_145 : 
  min_distinct_values n_series unique_mode_occurrence = 145 :=
by
  sorry

end minimum_distinct_values_is_145_l1444_144409


namespace cube_less_than_triple_l1444_144479

theorem cube_less_than_triple : ∀ x : ℤ, (x^3 < 3*x) ↔ (x = 1 ∨ x = -2) :=
by
  sorry

end cube_less_than_triple_l1444_144479


namespace total_peaches_l1444_144493

theorem total_peaches (x : ℕ) (P : ℕ) 
(h1 : P = 6 * x + 57)
(h2 : 6 * x + 57 = 9 * x - 51) : 
  P = 273 :=
by
  sorry

end total_peaches_l1444_144493


namespace jovana_shells_l1444_144418

theorem jovana_shells (initial_shells : ℕ) (added_shells : ℕ) (total_shells : ℕ) 
  (h_initial : initial_shells = 5) (h_added : added_shells = 12) :
  total_shells = 17 :=
by
  sorry

end jovana_shells_l1444_144418


namespace range_of_m_l1444_144489

theorem range_of_m (m : ℝ) :
  (3 * 1 - 2 + m) * (3 * 1 - 1 + m) < 0 →
  -2 < m ∧ m < -1 :=
by
  intro h
  sorry

end range_of_m_l1444_144489


namespace butter_mixture_price_l1444_144405

theorem butter_mixture_price :
  let cost1 := 48 * 150
  let cost2 := 36 * 125
  let cost3 := 24 * 100
  let revenue1 := cost1 + cost1 * (20 / 100)
  let revenue2 := cost2 + cost2 * (30 / 100)
  let revenue3 := cost3 + cost3 * (50 / 100)
  let total_weight := 48 + 36 + 24
  (revenue1 + revenue2 + revenue3) / total_weight = 167.5 :=
by
  sorry

end butter_mixture_price_l1444_144405


namespace find_oranges_to_put_back_l1444_144498

theorem find_oranges_to_put_back (A O x : ℕ) (h₁ : A + O = 15) (h₂ : 40 * A + 60 * O = 720) (h₃ : (360 + 360 - 60 * x) / (15 - x) = 45) : x = 3 := by
  sorry

end find_oranges_to_put_back_l1444_144498


namespace max_value_is_two_over_three_l1444_144410

noncomputable def max_value_expr (x : ℝ) : ℝ := 2^x - 8^x

theorem max_value_is_two_over_three :
  ∃ (x : ℝ), max_value_expr x = 2 / 3 :=
sorry

end max_value_is_two_over_three_l1444_144410


namespace smallest_prime_l1444_144464

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ , m ∣ n → m = 1 ∨ m = n
def is_composite (n : ℕ) : Prop := ∃ d : ℕ, d ∣ n ∧ d ≠ 1 ∧ d ≠ n 

theorem smallest_prime :
  ∃ n : ℕ, n = 29 ∧ 
  n >= 10 ∧ n < 100 ∧
  is_prime n ∧
  ((n / 10) = 3) ∧ 
  is_composite (n % 10 * 10 + n / 10) ∧
  (n % 10 * 10 + n / 10) % 5 = 0 :=
by {
  sorry
}

end smallest_prime_l1444_144464


namespace solution_mixture_l1444_144490

/-
  Let X be a solution that is 10% alcohol by volume.
  Let Y be a solution that is 30% alcohol by volume.
  We define the final solution to be 22% alcohol by volume.
  We need to prove that the amount of solution Y that needs
  to be added to 300 milliliters of solution X to achieve this 
  concentration is 450 milliliters.
-/

theorem solution_mixture (y : ℝ) : 
  (0.10 * 300) + (0.30 * y) = 0.22 * (300 + y) → 
  y = 450 :=
by {
  sorry
}

end solution_mixture_l1444_144490


namespace greatest_possible_average_speed_l1444_144448

def is_palindrome (n : ℕ) : Prop :=
  let s := n.digits 10
  s = s.reverse

theorem greatest_possible_average_speed :
  ∀ (o₁ o₂ : ℕ) (v_max t : ℝ), 
  is_palindrome o₁ → 
  is_palindrome o₂ → 
  o₁ = 12321 → 
  t = 2 ∧ v_max = 65 → 
  (∃ d, d = o₂ - o₁ ∧ d / t <= v_max) → 
  d / t = v_max :=
sorry

end greatest_possible_average_speed_l1444_144448


namespace arithmetic_sequence_fifth_term_l1444_144443

theorem arithmetic_sequence_fifth_term (x : ℝ) (a₂ : ℝ := x) (a₃ : ℝ := 3) 
    (a₁ : ℝ := -1) (h₁ : a₂ = a₁ + (1*(x + 1))) (h₂ : a₃ = a₁ + 2*(x + 1)) : 
    a₁ + 4*(a₃ - a₂ + 1) = 7 :=
by
  sorry

end arithmetic_sequence_fifth_term_l1444_144443


namespace solution_set_of_inequality_l1444_144487

theorem solution_set_of_inequality (x : ℝ) : 2 * x - 6 < 0 ↔ x < 3 := 
by
  sorry

end solution_set_of_inequality_l1444_144487


namespace range_of_a_l1444_144429

noncomputable def f (a x : ℝ) := (1 / 3) * x^3 - x^2 - 3 * x - a

theorem range_of_a (a : ℝ) : 
  (∃ x y z : ℝ, x ≠ y ∧ y ≠ z ∧ z ≠ x ∧ f a x = 0 ∧ f a y = 0 ∧ f a z = 0) ↔ (-9 < a ∧ a < 5 / 3) :=
by apply sorry

end range_of_a_l1444_144429


namespace fraction_simplification_l1444_144492

theorem fraction_simplification (b : ℝ) (hb : b = 3) : 
  (3 * b⁻¹ + b⁻¹ / 3) / b^2 = 10 / 81 :=
by
  rw [hb]
  sorry

end fraction_simplification_l1444_144492


namespace polynomial_perfect_square_value_of_k_l1444_144473

noncomputable def is_perfect_square (p : Polynomial ℝ) : Prop :=
  ∃ (q : Polynomial ℝ), p = q^2

theorem polynomial_perfect_square_value_of_k {k : ℝ} :
  is_perfect_square (Polynomial.X^2 - Polynomial.C k * Polynomial.X + Polynomial.C 25) ↔ (k = 10 ∨ k = -10) :=
by
  sorry

end polynomial_perfect_square_value_of_k_l1444_144473


namespace incorrect_major_premise_l1444_144430

-- Define a structure for Line and Plane
structure Line : Type :=
  (name : String)

structure Plane : Type :=
  (name : String)

-- Define relationships: parallel and contains
def parallel (l1 l2 : Line) : Prop := sorry
def line_in_plane (l : Line) (p : Plane) : Prop := sorry
def parallel_line_plane (l : Line) (p : Plane) : Prop := sorry

-- Conditions
variables 
  (a b : Line) 
  (α : Plane)
  (H1 : line_in_plane a α) 
  (H2 : parallel_line_plane b α)

-- Major premise to disprove
def major_premise (l : Line) (p : Plane) : Prop :=
  ∀ (l_in : Line), line_in_plane l_in p → parallel l l_in

-- State the problem
theorem incorrect_major_premise : ¬major_premise b α :=
sorry

end incorrect_major_premise_l1444_144430


namespace initial_chocolate_amount_l1444_144452

-- Define the problem conditions

def initial_dough (d : ℕ) := d = 36
def left_over_chocolate (lo_choc : ℕ) := lo_choc = 4
def chocolate_percentage (p : ℚ) := p = 0.20
def total_weight (d : ℕ) (c_choc : ℕ) := d + c_choc - 4
def chocolate_used (c_choc : ℕ) (lo_choc : ℕ) := c_choc - lo_choc

-- The main proof goal
theorem initial_chocolate_amount (d : ℕ) (lo_choc : ℕ) (p : ℚ) (C : ℕ) :
  initial_dough d → left_over_chocolate lo_choc → chocolate_percentage p →
  p * (total_weight d C) = chocolate_used C lo_choc → C = 13 :=
by
  intros hd hlc hp h
  sorry

end initial_chocolate_amount_l1444_144452


namespace final_price_for_tiffany_l1444_144450

noncomputable def calculate_final_price (n : ℕ) (c : ℝ) (d : ℝ) (s : ℝ) : ℝ :=
  let total_cost := n * c
  let discount := d * total_cost
  let discounted_price := total_cost - discount
  let sales_tax := s * discounted_price
  let final_price := discounted_price + sales_tax
  final_price

theorem final_price_for_tiffany :
  calculate_final_price 9 4.50 0.20 0.07 = 34.67 :=
by
  sorry

end final_price_for_tiffany_l1444_144450


namespace garage_sale_records_l1444_144422

/--
Roberta started off with 8 vinyl records. Her friends gave her 12
records for her birthday and she bought some more at a garage
sale. It takes her 2 days to listen to 1 record. It will take her
100 days to listen to her record collection. Prove that she bought
30 records at the garage sale.
-/
theorem garage_sale_records :
  let initial_records := 8
  let gift_records := 12
  let days_per_record := 2
  let total_listening_days := 100
  let total_records := total_listening_days / days_per_record
  let records_before_sale := initial_records + gift_records
  let records_bought := total_records - records_before_sale
  records_bought = 30 := 
by
  -- Variable assumptions
  let initial_records := 8
  let gift_records := 12
  let days_per_record := 2
  let total_listening_days := 100

  -- Definitions
  let total_records := total_listening_days / days_per_record
  let records_before_sale := initial_records + gift_records
  let records_bought := total_records - records_before_sale

  -- Conclusion to prove
  show records_bought = 30
  sorry

end garage_sale_records_l1444_144422


namespace equation_has_two_solutions_l1444_144486

theorem equation_has_two_solutions : ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ ∀ x : ℝ, ¬ ( |x - 1| = |x - 2| + |x - 3| ) ↔ (x ≠ x₁ ∧ x ≠ x₂) :=
sorry

end equation_has_two_solutions_l1444_144486


namespace range_of_a_l1444_144425

theorem range_of_a (a : ℝ) (h1 : a > 0) (h2 : ∃ x : ℝ, |x - 4| + |x + 3| < a) : a > 7 :=
sorry

end range_of_a_l1444_144425


namespace minimum_value_of_m_l1444_144432

-- Define a function to determine if a number is a perfect square
def is_perfect_square (n : ℕ) : Prop := ∃ k : ℕ, k * k = n

-- Define a function to determine if a number is a perfect cube
def is_perfect_cube (n : ℕ) : Prop := ∃ k : ℕ, k * k * k = n

-- Our main theorem statement
theorem minimum_value_of_m :
  ∃ m : ℕ, (600 < m ∧ m ≤ 800) ∧
           is_perfect_square (3 * m) ∧
           is_perfect_cube (5 * m) :=
sorry

end minimum_value_of_m_l1444_144432


namespace quadratic_has_real_roots_iff_l1444_144442

theorem quadratic_has_real_roots_iff (k : ℝ) (hk : k ≠ 0) :
  (∃ x : ℝ, k * x^2 - x + 1 = 0) ↔ k ≤ 1 / 4 :=
by
  sorry

end quadratic_has_real_roots_iff_l1444_144442


namespace max_lift_times_l1444_144416

theorem max_lift_times (n : ℕ) :
  (2 * 30 * 10) = (2 * 25 * n) → n = 12 :=
by
  sorry

end max_lift_times_l1444_144416


namespace Y_3_2_eq_1_l1444_144446

def Y (a b : ℕ) : ℕ := a^2 - 2*a*b + b^2

theorem Y_3_2_eq_1 : Y 3 2 = 1 := by
  sorry

end Y_3_2_eq_1_l1444_144446


namespace area_of_smaller_circle_l1444_144402

theorem area_of_smaller_circle (r R : ℝ) (PA AB : ℝ) 
  (h1 : R = 2 * r) (h2 : PA = 4) (h3 : AB = 4) :
  π * r^2 = 2 * π :=
by
  sorry

end area_of_smaller_circle_l1444_144402


namespace find_f_neg_two_l1444_144496

-- Define the function f and its properties
variable (f : ℝ → ℝ)
variable (h1 : ∀ a b : ℝ, f (a + b) = f a * f b)
variable (h2 : ∀ x : ℝ, f x > 0)
variable (h3 : f 1 = 1 / 2)

-- State the theorem to prove that f(-2) = 4
theorem find_f_neg_two : f (-2) = 4 :=
by
  sorry

end find_f_neg_two_l1444_144496


namespace range_of_a_l1444_144412

noncomputable def f (a x : ℝ) : ℝ := x * Real.exp x + (1 / 2) * a * x^2 + a * x

theorem range_of_a (a : ℝ) : 
    (∀ x : ℝ, 2 * Real.exp (f a x) + Real.exp 1 + 2 ≥ 0) ↔ (0 ≤ a ∧ a ≤ 1) := 
sorry

end range_of_a_l1444_144412


namespace new_students_joined_l1444_144424

-- Define conditions
def initial_students : ℕ := 160
def end_year_students : ℕ := 120
def fraction_transferred_out : ℚ := 1 / 3
def total_students_at_start := end_year_students * 3 / 2

-- Theorem statement
theorem new_students_joined : (total_students_at_start - initial_students = 20) :=
by
  -- Placeholder for proof
  sorry

end new_students_joined_l1444_144424


namespace arithmetic_sequence_part_a_arithmetic_sequence_part_b_l1444_144428

theorem arithmetic_sequence_part_a (e u k : ℕ) (n : ℕ) 
  (h1 : e = 1) 
  (h2 : u = 1000) 
  (h3 : k = 343) 
  (h4 : n = 100) : ¬ (∃ d m, e + (m - 1) * d = k ∧ u = e + (n - 1) * d ∧ 1 < m ∧ m < n) :=
by sorry

theorem arithmetic_sequence_part_b (e u k : ℝ) (n : ℕ) 
  (h1 : e = 81 * Real.sqrt 2 - 64 * Real.sqrt 3) 
  (h2 : u = 54 * Real.sqrt 2 - 28 * Real.sqrt 3)
  (h3 : k = 69 * Real.sqrt 2 - 48 * Real.sqrt 3)
  (h4 : n = 100) : (∃ d m, e + (m - 1) * d = k ∧ u = e + (n - 1) * d ∧ 1 < m ∧ m < n) :=
by sorry

end arithmetic_sequence_part_a_arithmetic_sequence_part_b_l1444_144428


namespace problem_c_d_sum_l1444_144433

theorem problem_c_d_sum (C D : ℝ) (h : ∀ x : ℝ, x ≠ 3 → (C / (x - 3) + D * (x - 2) = (5 * x ^ 2 - 8 * x - 6) / (x - 3))) : C + D = 20 :=
sorry

end problem_c_d_sum_l1444_144433


namespace rectangle_area_l1444_144413

theorem rectangle_area (A1 A2 : ℝ) (h1 : A1 = 40) (h2 : A2 = 10) :
    ∃ n : ℕ, n = 240 ∧ ∃ R : ℝ, R = 2 * Real.sqrt (40 / Real.pi) + 2 * Real.sqrt (10 / Real.pi) ∧ 
               (4 * Real.sqrt (10) / Real.sqrt (Real.pi)) * (6 * Real.sqrt (10) / Real.sqrt (Real.pi)) = n / Real.pi :=
by
  sorry

end rectangle_area_l1444_144413


namespace probability_no_defective_pencils_l1444_144471

-- Definitions based on conditions
def total_pencils : ℕ := 11
def defective_pencils : ℕ := 2
def selected_pencils : ℕ := 3

-- Helper function to compute combinations
def combination (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- The proof statement
theorem probability_no_defective_pencils :
  let total_ways := combination total_pencils selected_pencils
  let non_defective_ways := combination (total_pencils - defective_pencils) selected_pencils
  total_ways ≠ 0 → 
  (non_defective_ways / total_ways : ℚ) = 28 / 55 := 
by
  sorry

end probability_no_defective_pencils_l1444_144471


namespace work_done_by_b_l1444_144462

theorem work_done_by_b (x : ℝ) (h1 : (1/6) + (1/13) = (1/x)) : x = 78/7 :=
  sorry

end work_done_by_b_l1444_144462


namespace plant_lamp_arrangement_count_l1444_144440

theorem plant_lamp_arrangement_count :
  let basil_plants := 2
  let aloe_plants := 2
  let white_lamps := 3
  let red_lamps := 3
  (∀ plant, plant = basil_plants ∨ plant = aloe_plants)
  ∧ (∀ lamp, lamp = white_lamps ∨ lamp = red_lamps)
  → (∀ plant, ∃ lamp, plant → lamp)
  → ∃ count, count = 50 := 
by
  sorry

end plant_lamp_arrangement_count_l1444_144440


namespace greatest_divisor_of_arithmetic_sum_l1444_144467

theorem greatest_divisor_of_arithmetic_sum (a d : ℕ) (ha : 0 < a) (hd : 0 < d) :
  ∃ k : ℕ, k = 6 ∧ ∀ a d : ℕ, 12 * a + 66 * d % k = 0 :=
by sorry

end greatest_divisor_of_arithmetic_sum_l1444_144467


namespace xiao_ming_cube_division_l1444_144431

theorem xiao_ming_cube_division (large_edge small_cubes : ℕ)
  (large_edge_eq : large_edge = 4)
  (small_cubes_eq : small_cubes = 29)
  (total_volume : large_edge ^ 3 = 64) :
  ∃ (small_edge_1_cube : ℕ), small_edge_1_cube = 24 ∧ small_cubes = 29 ∧ 
  small_edge_1_cube + (small_cubes - small_edge_1_cube) * 8 = 64 := 
by
  -- We only need to assert the existence here as per the instruction.
  sorry

end xiao_ming_cube_division_l1444_144431


namespace cheaperCandy_cost_is_5_l1444_144406

def cheaperCandy (C : ℝ) : Prop :=
  let expensiveCandyCost := 20 * 8
  let cheaperCandyCost := 40 * C
  let totalWeight := 20 + 40
  let totalCost := 60 * 6
  expensiveCandyCost + cheaperCandyCost = totalCost

theorem cheaperCandy_cost_is_5 : cheaperCandy 5 :=
by
  unfold cheaperCandy
  -- SORRY is a placeholder for the proof steps, which are not required
  sorry 

end cheaperCandy_cost_is_5_l1444_144406


namespace repeating_pattern_sum_23_l1444_144470

def repeating_pattern_sum (n : ℕ) : ℤ :=
  let pattern := [4, -3, 2, -1, 0]
  let block_sum := List.sum pattern
  let complete_blocks := n / pattern.length
  let remainder := n % pattern.length
  complete_blocks * block_sum + List.sum (pattern.take remainder)

theorem repeating_pattern_sum_23 : repeating_pattern_sum 23 = 11 := 
  sorry

end repeating_pattern_sum_23_l1444_144470


namespace geometric_sequence_product_identity_l1444_144403

theorem geometric_sequence_product_identity 
  {a : ℕ → ℝ} (is_geometric_sequence : ∃ r, ∀ n, a (n+1) = a n * r)
  (h : a 3 * a 4 * a 6 * a 7 = 81):
  a 1 * a 9 = 9 :=
by
  sorry

end geometric_sequence_product_identity_l1444_144403


namespace prob_not_same_city_is_056_l1444_144475

def probability_not_same_city (P_A_cityA P_B_cityA : ℝ) : ℝ :=
  let P_A_cityB := 1 - P_A_cityA
  let P_B_cityB := 1 - P_B_cityA
  (P_A_cityA * P_B_cityB) + (P_A_cityB * P_B_cityA)

theorem prob_not_same_city_is_056 :
  probability_not_same_city 0.6 0.2 = 0.56 :=
by
  sorry

end prob_not_same_city_is_056_l1444_144475


namespace dot_product_u_v_l1444_144453

def u : ℝ × ℝ × ℝ × ℝ := (4, -3, 5, -2)
def v : ℝ × ℝ × ℝ × ℝ := (-6, 1, 2, 3)

theorem dot_product_u_v : (4 * -6 + -3 * 1 + 5 * 2 + -2 * 3) = -23 := by
  sorry

end dot_product_u_v_l1444_144453


namespace inequality_proof_l1444_144456

theorem inequality_proof
  (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a^2 / b) + (b^3 / c^2) + (c^4 / a^3) ≥ -a + 2*b + 2*c :=
sorry

end inequality_proof_l1444_144456


namespace students_disliked_menu_l1444_144480

theorem students_disliked_menu (total_students liked_students : ℕ) (h1 : total_students = 400) (h2 : liked_students = 235) : total_students - liked_students = 165 :=
by 
  sorry

end students_disliked_menu_l1444_144480


namespace bees_on_20th_day_l1444_144477

-- Define the conditions
def initial_bees : ℕ := 1

def companions_per_bee : ℕ := 4

-- Define the total number of bees on day n
def total_bees (n : ℕ) : ℕ :=
  (initial_bees + companions_per_bee) ^ n

-- Statement to prove
theorem bees_on_20th_day : total_bees 20 = 5^20 :=
by
  -- The proof is omitted
  sorry

end bees_on_20th_day_l1444_144477


namespace find_u_plus_v_l1444_144485

-- Conditions: 3u - 4v = 17 and 5u - 2v = 1.
-- Question: Find the value of u + v.

theorem find_u_plus_v (u v : ℚ) (h1 : 3 * u - 4 * v = 17) (h2 : 5 * u - 2 * v = 1) : u + v = -8 :=
by
  sorry

end find_u_plus_v_l1444_144485


namespace shortest_chord_eqn_of_circle_l1444_144421

theorem shortest_chord_eqn_of_circle 
    (k x y : ℝ)
    (C_eq : x^2 + y^2 - 2*x - 24 = 0)
    (line_l : y = k * (x - 2) - 1) :
  y = x - 3 :=
by
  sorry

end shortest_chord_eqn_of_circle_l1444_144421


namespace vector_expression_l1444_144499

variables (a b c : ℝ × ℝ)
variables (m n : ℝ)

noncomputable def vec_a : ℝ × ℝ := (1, 1)
noncomputable def vec_b : ℝ × ℝ := (1, -1)
noncomputable def vec_c : ℝ × ℝ := (-1, 2)

/-- Prove that vector c can be expressed in terms of vectors a and b --/
theorem vector_expression : 
  vec_c = m • vec_a + n • vec_b → (m = 1/2 ∧ n = -3/2) :=
sorry

end vector_expression_l1444_144499


namespace xiaoming_problem_l1444_144454

theorem xiaoming_problem (a x : ℝ) 
  (h1 : 20.18 * a - 20.18 = x)
  (h2 : x = 2270.25) : 
  a = 113.5 := 
by 
  sorry

end xiaoming_problem_l1444_144454


namespace delta_solution_l1444_144414

theorem delta_solution : ∃ Δ : ℤ, 4 * (-3) = Δ - 1 ∧ Δ = -11 :=
by
  -- Using the condition 4(-3) = Δ - 1, 
  -- we need to prove that Δ = -11
  sorry

end delta_solution_l1444_144414


namespace chi_squared_test_expected_value_correct_l1444_144426
open ProbabilityTheory

section Part1

def n : ℕ := 400
def a : ℕ := 60
def b : ℕ := 20
def c : ℕ := 180
def d : ℕ := 140
def alpha : ℝ := 0.005
def chi_critical : ℝ := 7.879

noncomputable def chi_squared : ℝ :=
  (n * (a * d - b * c) ^ 2) / ((a + b) * (c + d) * (a + c) * (b + d))

theorem chi_squared_test : chi_squared > chi_critical :=
  sorry

end Part1

section Part2

def reward_med : ℝ := 6  -- 60,000 yuan in 10,000 yuan unit
def reward_small : ℝ := 2  -- 20,000 yuan in 10,000 yuan unit
def total_support : ℕ := 12
def total_rewards : ℕ := 9

noncomputable def dist_table : List (ℝ × ℝ) :=
  [(180, 1 / 220),
   (220, 27 / 220),
   (260, 27 / 55),
   (300, 21 / 55)]

noncomputable def expected_value : ℝ :=
  dist_table.foldr (fun (xi : ℝ × ℝ) acc => acc + xi.1 * xi.2) 0

theorem expected_value_correct : expected_value = 270 :=
  sorry

end Part2

end chi_squared_test_expected_value_correct_l1444_144426


namespace geometric_sequence_a_formula_l1444_144401

noncomputable def a (n : ℕ) : ℤ :=
  if n = 1 then 1
  else if n = 2 then 2
  else n - 2

noncomputable def b (n : ℕ) : ℤ :=
  a (n + 1) - a n

theorem geometric_sequence (n : ℕ) (h : n ≥ 2) : 
  b n = (-1) * b (n - 1) := 
  sorry

theorem a_formula (n : ℕ) : 
  a n = (-1) ^ (n - 1) := 
  sorry

end geometric_sequence_a_formula_l1444_144401


namespace length_of_faster_train_l1444_144459

-- Definitions for the given conditions
def speed_faster_train_kmh : ℝ := 50
def speed_slower_train_kmh : ℝ := 32
def time_seconds : ℝ := 15

theorem length_of_faster_train : 
  let speed_relative_kmh := speed_faster_train_kmh - speed_slower_train_kmh
  let speed_relative_mps := speed_relative_kmh * (1000 / 3600)
  let length_faster_train := speed_relative_mps * time_seconds
  length_faster_train = 75 := 
by 
  sorry 

end length_of_faster_train_l1444_144459


namespace no_seven_sum_possible_l1444_144436

theorem no_seven_sum_possible :
  let outcomes := [-1, -3, -5, 2, 4, 6]
  ∀ (a b : Int), a ∈ outcomes → b ∈ outcomes → a + b ≠ 7 :=
by
  sorry

end no_seven_sum_possible_l1444_144436


namespace binom_computation_l1444_144451

noncomputable def binom : ℕ → ℕ → ℕ
| n, 0       => 1
| 0, k+1     => 0
| n+1, k+1   => binom n k + binom n (k+1)

theorem binom_computation :
  (binom 10 3) * (binom 8 3) = 6720 := by
  sorry

end binom_computation_l1444_144451


namespace regular_octahedron_has_4_pairs_l1444_144441

noncomputable def regular_octahedron_parallel_edges : ℕ :=
  4

theorem regular_octahedron_has_4_pairs
  (h : true) : regular_octahedron_parallel_edges = 4 :=
by
  sorry

end regular_octahedron_has_4_pairs_l1444_144441


namespace correct_op_l1444_144404

-- Declare variables and conditions
variables {a b : ℝ} {m n : ℤ}
variable (ha : a > 0)
variable (hb : b ≠ 0)

-- Define and state the theorem
theorem correct_op (ha : a > 0) (hb : b ≠ 0) : (b / a)^m = a^(-m) * b^m :=
sorry  -- Proof omitted

end correct_op_l1444_144404
