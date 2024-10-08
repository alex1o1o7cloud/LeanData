import Mathlib

namespace minimum_chocolates_l212_212008

theorem minimum_chocolates (x : ℤ) (h1 : x ≥ 150) (h2 : x % 15 = 7) : x = 157 :=
sorry

end minimum_chocolates_l212_212008


namespace range_of_m_l212_212897

theorem range_of_m 
  (h : ∀ x : ℝ, x^2 + m * x + m^2 - 1 > 0) :
  m ∈ (Set.Ioo (-(2 * Real.sqrt 3) / 3) (-(2 * Real.sqrt 3) / 3)).union (Set.Ioi ((2 * Real.sqrt 3) / 3)) := 
sorry

end range_of_m_l212_212897


namespace neither_sufficient_nor_necessary_l212_212633

theorem neither_sufficient_nor_necessary 
  (a b c : ℝ) : 
  ¬ ((∀ x : ℝ, b^2 - 4 * a * c < 0 → a * x^2 + b * x + c > 0) ∧ 
     (∀ x : ℝ, a * x^2 + b * x + c > 0 → b^2 - 4 * a * c < 0)) := 
by
  sorry

end neither_sufficient_nor_necessary_l212_212633


namespace sufficient_but_not_necessary_condition_l212_212013

variables (A B C : Prop)

theorem sufficient_but_not_necessary_condition (h1 : B → A) (h2 : C → B) (h3 : ¬(B → C)) : (C → A) ∧ ¬(A → C) :=
by
  sorry

end sufficient_but_not_necessary_condition_l212_212013


namespace percentage_of_female_employees_l212_212514

theorem percentage_of_female_employees (E : ℕ) (hE : E = 1400) 
  (pct_computer_literate : ℚ) (hpct : pct_computer_literate = 0.62)
  (female_computer_literate : ℕ) (hfcl : female_computer_literate = 588)
  (pct_male_computer_literate : ℚ) (hmcl : pct_male_computer_literate = 0.5) :
  100 * (840 / 1400) = 60 := 
by
  sorry

end percentage_of_female_employees_l212_212514


namespace max_value_4x_plus_y_l212_212014

theorem max_value_4x_plus_y (x y : ℝ) (h : 16 * x^2 + y^2 + 4 * x * y = 3) :
  ∃ (M : ℝ), M = 2 ∧ ∀ (u : ℝ), (∃ (x y : ℝ), 16 * x^2 + y^2 + 4 * x * y = 3 ∧ u = 4 * x + y) → u ≤ M :=
by
  use 2
  sorry

end max_value_4x_plus_y_l212_212014


namespace isosceles_in_27_gon_l212_212609

def vertices := {x : ℕ // x < 27}

def is_isosceles_triangle (a b c : vertices) : Prop :=
  (a.val + c.val) / 2 % 27 = b.val

def is_isosceles_trapezoid (a b c d : vertices) : Prop :=
  (a.val + d.val) / 2 % 27 = (b.val + c.val) / 2 % 27

def seven_points_form_isosceles (s : Finset vertices) : Prop :=
  ∃ (a b c : vertices) (h1 : a ∈ s) (h2 : b ∈ s) (h3 : c ∈ s), is_isosceles_triangle a b c

def seven_points_form_isosceles_trapezoid (s : Finset vertices) : Prop :=
  ∃ (a b c d : vertices) (h1 : a ∈ s) (h2 : b ∈ s) (h3 : c ∈ s) (h4 : d ∈ s), is_isosceles_trapezoid a b c d

theorem isosceles_in_27_gon :
  ∀ (s : Finset vertices), s.card = 7 → 
  (seven_points_form_isosceles s) ∨ (seven_points_form_isosceles_trapezoid s) :=
by sorry

end isosceles_in_27_gon_l212_212609


namespace mary_blue_marbles_l212_212905

theorem mary_blue_marbles (dan_blue_marbles mary_blue_marbles : ℕ)
  (h1 : dan_blue_marbles = 5)
  (h2 : mary_blue_marbles = 2 * dan_blue_marbles) : mary_blue_marbles = 10 := 
by
  sorry

end mary_blue_marbles_l212_212905


namespace nylon_needed_is_192_l212_212781

-- Define the required lengths for the collars
def nylon_needed_for_dog_collar : ℕ := 18
def nylon_needed_for_cat_collar : ℕ := 10

-- Define the number of collars needed
def number_of_dog_collars : ℕ := 9
def number_of_cat_collars : ℕ := 3

-- Define the total nylon needed
def total_nylon_needed : ℕ :=
  (nylon_needed_for_dog_collar * number_of_dog_collars) + (nylon_needed_for_cat_collar * number_of_cat_collars)

-- State the theorem we need to prove
theorem nylon_needed_is_192 : total_nylon_needed = 192 := 
  by
    -- Simplification to match the complete statement for completeness
    sorry

end nylon_needed_is_192_l212_212781


namespace smallest_piece_to_cut_l212_212995

theorem smallest_piece_to_cut (x : ℕ) 
  (h1 : 9 - x > 0) 
  (h2 : 16 - x > 0) 
  (h3 : 18 - x > 0) :
  7 ≤ x ∧ 9 - x + 16 - x ≤ 18 - x :=
by {
  sorry
}

end smallest_piece_to_cut_l212_212995


namespace four_digit_square_number_divisible_by_11_with_unit_1_l212_212160

theorem four_digit_square_number_divisible_by_11_with_unit_1 
  : ∃ y : ℕ, y >= 1000 ∧ y <= 9999 ∧ (∃ n : ℤ, y = n^2) ∧ y % 11 = 0 ∧ y % 10 = 1 ∧ y = 9801 := 
by {
  -- sorry statement to skip the proof.
  sorry 
}

end four_digit_square_number_divisible_by_11_with_unit_1_l212_212160


namespace log_sum_eq_l212_212210

theorem log_sum_eq : ∀ (x y : ℝ), y = 2016 * x ∧ x^y = y^x → (Real.logb 2016 x + Real.logb 2016 y) = 2017 / 2015 :=
by
  intros x y h
  sorry

end log_sum_eq_l212_212210


namespace necessary_but_not_sufficient_condition_l212_212439

theorem necessary_but_not_sufficient_condition :
  (∀ m, -1 < m ∧ m < 5 → ∀ x, 
    x^2 - 2 * m * x + m^2 - 1 = 0 → -2 < x ∧ x < 4) ∧ 
  ¬ (∀ m, -1 < m ∧ m < 5 → ∀ x, 
    x^2 - 2 * m * x + m^2 - 1 = 0 → -2 < x ∧ x < 4) :=
sorry

end necessary_but_not_sufficient_condition_l212_212439


namespace unique_positive_integer_solution_l212_212822

theorem unique_positive_integer_solution (n p : ℕ) (x y : ℕ) :
  (x + p * y = n ∧ x + y = p^2 ∧ x > 0 ∧ y > 0) ↔ 
  (p > 1 ∧ (p - 1) ∣ (n - 1) ∧ ∀ k : ℕ, n ≠ p^k ∧ ∃! t : ℕ × ℕ, (t.1 + p * t.2 = n ∧ t.1 + t.2 = p^2 ∧ t.1 > 0 ∧ t.2 > 0)) :=
by
  sorry

end unique_positive_integer_solution_l212_212822


namespace angela_deliveries_l212_212145

theorem angela_deliveries
  (n_meals : ℕ)
  (h_meals : n_meals = 3)
  (n_packages : ℕ)
  (h_packages : n_packages = 8 * n_meals) :
  n_meals + n_packages = 27 := by
  sorry

end angela_deliveries_l212_212145


namespace polynomial_division_l212_212040

variable (a p x : ℝ)

theorem polynomial_division :
  (p^8 * x^4 - 81 * a^12) / (p^6 * x^3 - 3 * a^3 * p^4 * x^2 + 9 * a^6 * p^2 * x - 27 * a^9) = p^2 * x + 3 * a^3 :=
by sorry

end polynomial_division_l212_212040


namespace cupcakes_left_correct_l212_212675

-- Definitions based on conditions
def total_cupcakes : ℕ := 10 * 12 + 1 * 12 / 2
def total_students : ℕ := 48
def absent_students : ℕ := 6 
def field_trip_students : ℕ := 8
def teachers : ℕ := 2
def teachers_aids : ℕ := 2

-- Function to calculate the number of present people
def total_present_people : ℕ :=
  total_students - absent_students - field_trip_students + teachers + teachers_aids

-- Function to calculate the cupcakes left
def cupcakes_left : ℕ := total_cupcakes - total_present_people

-- The theorem to prove
theorem cupcakes_left_correct : cupcakes_left = 85 := 
by
  -- This is where the proof would go
  sorry

end cupcakes_left_correct_l212_212675


namespace find_C_l212_212622

theorem find_C 
  (m n : ℝ)
  (C : ℝ)
  (h1 : m = 6 * n + C)
  (h2 : m + 2 = 6 * (n + 0.3333333333333333) + C) 
  : C = 0 := by
  sorry

end find_C_l212_212622


namespace rectangle_longer_side_l212_212367

theorem rectangle_longer_side
  (r : ℝ)
  (A_circle : ℝ)
  (A_rectangle : ℝ)
  (shorter_side : ℝ)
  (longer_side : ℝ) :
  r = 5 →
  A_circle = 25 * Real.pi →
  A_rectangle = 3 * A_circle →
  shorter_side = 2 * r →
  longer_side = A_rectangle / shorter_side →
  longer_side = 7.5 * Real.pi :=
by
  intros
  sorry

end rectangle_longer_side_l212_212367


namespace determine_gx_l212_212149

/-
  Given two polynomials f(x) and h(x), we need to show that g(x) is a certain polynomial
  when f(x) + g(x) = h(x).
-/

def f (x : ℝ) : ℝ := 4 * x^5 + 3 * x^3 + x - 2
def h (x : ℝ) : ℝ := 7 * x^3 - 5 * x + 4
def g (x : ℝ) : ℝ := -4 * x^5 + 4 * x^3 - 4 * x + 6

theorem determine_gx (x : ℝ) : f x + g x = h x :=
by
  -- proof will go here
  sorry

end determine_gx_l212_212149


namespace sandy_paid_cost_shop2_l212_212958

-- Define the conditions
def books_shop1 : ℕ := 65
def cost_shop1 : ℕ := 1380
def books_shop2 : ℕ := 55
def avg_price_per_book : ℕ := 19

-- Calculation of the total amount Sandy paid for the books from the second shop
def cost_shop2 (total_books: ℕ) (avg_price: ℕ) (cost1: ℕ) : ℕ :=
  (total_books * avg_price) - cost1

-- Define the theorem we want to prove
theorem sandy_paid_cost_shop2 : cost_shop2 (books_shop1 + books_shop2) avg_price_per_book cost_shop1 = 900 :=
sorry

end sandy_paid_cost_shop2_l212_212958


namespace candy_box_original_price_l212_212056

theorem candy_box_original_price (P : ℝ) (h1 : 1.25 * P = 20) : P = 16 :=
sorry

end candy_box_original_price_l212_212056


namespace fred_speed_5_mph_l212_212815

theorem fred_speed_5_mph (F : ℝ) (h1 : 50 = 25 + 25) (h2 : 25 / 5 = 5) (h3 : 25 / F = 5) : 
  F = 5 :=
by
  -- Since Fred's speed makes meeting with Sam in the same time feasible
  sorry

end fred_speed_5_mph_l212_212815


namespace find_original_workers_and_time_l212_212347

-- Definitions based on the identified conditions
def original_workers (x : ℕ) (y : ℕ) : Prop :=
  (x - 2) * (y + 4) = x * y ∧
  (x + 3) * (y - 2) > x * y ∧
  (x + 4) * (y - 3) > x * y

-- Problem statement to prove
theorem find_original_workers_and_time (x y : ℕ) :
  original_workers x y → x = 6 ∧ y = 8 :=
by
  sorry

end find_original_workers_and_time_l212_212347


namespace max_sum_of_arithmetic_sequence_l212_212925

theorem max_sum_of_arithmetic_sequence 
  (d : ℤ) (a₁ a₃ a₅ a₁₅ : ℤ) (S : ℕ → ℤ)
  (h₁ : d ≠ 0)
  (h₂ : a₃ = a₁ + 2 * d)
  (h₃ : a₅ = a₃ + 2 * d)
  (h₄ : a₁₅ = a₅ + 10 * d)
  (h_geom : a₃ * a₃ = a₅ * a₁₅)
  (h_a₁ : a₁ = 3)
  (h_S : ∀ n, S n = n * a₁ + (n * (n - 1) / 2) * d) :
  ∃ n, S n = 4 :=
by
  sorry

end max_sum_of_arithmetic_sequence_l212_212925


namespace max_sector_area_central_angle_l212_212584

theorem max_sector_area_central_angle (radius arc_length : ℝ) :
  (arc_length + 2 * radius = 20) ∧ (arc_length = 20 - 2 * radius) ∧
  (arc_length / radius = 2) → 
  arc_length / radius = 2 :=
by
  intros h 
  sorry

end max_sector_area_central_angle_l212_212584


namespace max_x_real_nums_l212_212745

theorem max_x_real_nums (x y z : ℝ) (h₁ : x + y + z = 6) (h₂ : x * y + x * z + y * z = 10) : x ≤ 2 :=
sorry

end max_x_real_nums_l212_212745


namespace smallest_sum_squares_edges_is_cube_l212_212077

theorem smallest_sum_squares_edges_is_cube (V : ℝ) (a b c : ℝ)
  (h_vol : a * b * c = V) :
  a^2 + b^2 + c^2 ≥ 3 * (V^(2/3)) := 
sorry

end smallest_sum_squares_edges_is_cube_l212_212077


namespace difference_in_cents_l212_212468

-- Given definitions and conditions
def number_of_coins : ℕ := 3030
def min_nickels : ℕ := 3
def ratio_pennies_to_nickels : ℕ := 10

-- Problem statement: Prove that the difference in cents between the maximum and minimum monetary amounts is 1088
theorem difference_in_cents (p n : ℕ) (h1 : p + n = number_of_coins)
  (h2 : p ≥ ratio_pennies_to_nickels * n) (h3 : n ≥ min_nickels) :
  4 * 275 = 1100 ∧ (3030 + 1100) - (3030 + 4 * 3) = 1088 :=
by {
  sorry
}

end difference_in_cents_l212_212468


namespace range_of_a_l212_212591

  variable {A : Set ℝ} {B : Set ℝ}
  variable {a : ℝ}

  def A_def (a : ℝ) : Set ℝ := { x | a ≤ x ∧ x ≤ 2 * a - 4 }
  def B_def : Set ℝ := { x | -1 < x ∧ x < 6 }

  theorem range_of_a (h : A_def a ∩ B_def = A_def a) : a < 5 :=
  sorry
  
end range_of_a_l212_212591


namespace hot_dogs_remainder_l212_212576

theorem hot_dogs_remainder :
  25197625 % 4 = 1 :=
by
  sorry

end hot_dogs_remainder_l212_212576


namespace hydrogen_atoms_in_compound_l212_212373

theorem hydrogen_atoms_in_compound :
  ∀ (n : ℕ), 98 = 14 + n + 80 → n = 4 :=
by intro n h_eq
   sorry

end hydrogen_atoms_in_compound_l212_212373


namespace integer_solution_system_eq_det_l212_212236

theorem integer_solution_system_eq_det (a b c d : ℤ) 
  (h : ∀ m n : ℤ, ∃ x y : ℤ, a * x + b * y = m ∧ c * x + d * y = n) : 
  a * d - b * c = 1 ∨ a * d - b * c = -1 :=
by
  sorry

end integer_solution_system_eq_det_l212_212236


namespace necessary_but_not_sufficient_condition_l212_212371

theorem necessary_but_not_sufficient_condition (x : ℝ) (h : x > 0) : 
  ((x > 2 ∧ x < 4) ↔ (2 < x ∧ x < 4)) :=
by {
    sorry
}

end necessary_but_not_sufficient_condition_l212_212371


namespace zander_stickers_l212_212041

theorem zander_stickers (S : ℕ) (h1 : 44 = (11 / 25) * S) : S = 100 :=
by
  sorry

end zander_stickers_l212_212041


namespace find_xyz_ratio_l212_212782

theorem find_xyz_ratio (a b c x y z : ℝ) 
  (h1 : x / a + y / b + z / c = 2) 
  (h2 : a^2 / x^2 + b^2 / y^2 + c^2 / z^2 = 1) : 
  x^2 / a^2 + y^2 / b^2 + z^2 / c^2 = 9 :=
by sorry

end find_xyz_ratio_l212_212782


namespace equation_of_circle_l212_212448

def center : ℝ × ℝ := (3, -2)
def radius : ℝ := 5

theorem equation_of_circle (x y : ℝ) :
  (x - center.1)^2 + (y - center.2)^2 = radius^2 ↔
  (x - 3)^2 + (y + 2)^2 = 25 :=
by
  simp [center, radius]
  sorry

end equation_of_circle_l212_212448


namespace one_div_add_one_div_interval_one_div_add_one_div_not_upper_bounded_one_div_add_one_div_in_interval_l212_212399

theorem one_div_add_one_div_interval (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a + b = 2) :
  2 ≤ (1 / a + 1 / b) := 
sorry

theorem one_div_add_one_div_not_upper_bounded (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a + b = 2) :
  ∀ M > 2, ∃ a' b', 0 < a' ∧ 0 < b' ∧ a' + b' = 2 ∧ (1 / a' + 1 / b') > M := 
sorry

theorem one_div_add_one_div_in_interval (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a + b = 2) :
  (2 ≤ (1 / a + 1 / b) ∧ ∀ M > 2, ∃ a' b', 0 < a' ∧ 0 < b' ∧ a' + b' = 2 ∧ (1 / a' + 1 / b') > M) := 
sorry

end one_div_add_one_div_interval_one_div_add_one_div_not_upper_bounded_one_div_add_one_div_in_interval_l212_212399


namespace top_card_probability_spades_or_clubs_l212_212354

-- Definitions
def total_cards : ℕ := 52
def suits : ℕ := 4
def ranks : ℕ := 13
def spades_cards : ℕ := ranks
def clubs_cards : ℕ := ranks
def favorable_outcomes : ℕ := spades_cards + clubs_cards

-- Probability calculation statement
theorem top_card_probability_spades_or_clubs :
  (favorable_outcomes : ℚ) / (total_cards : ℚ) = 1 / 2 :=
  sorry

end top_card_probability_spades_or_clubs_l212_212354


namespace man_l212_212673

theorem man's_salary (S : ℝ) 
  (h_food : S * (1 / 5) > 0)
  (h_rent : S * (1 / 10) > 0)
  (h_clothes : S * (3 / 5) > 0)
  (h_left : S * (1 / 10) = 19000) : 
  S = 190000 := by
  sorry

end man_l212_212673


namespace geometric_sequence_problem_l212_212212

variable (a_n : ℕ → ℝ)

def geometric_sequence (a₁ : ℝ) (q : ℝ) : ℕ → ℝ := λ n => a₁ * q^(n-1)

theorem geometric_sequence_problem (q a_1 : ℝ) (a_1_pos : a_1 = 9)
  (h : ∀ n, a_n n = geometric_sequence a_1 q n)
  (h5 : a_n 5 = a_n 3 * (a_n 4)^2) : 
  a_n 4 = 1/3 ∨ a_n 4 = -1/3 := by 
  sorry

end geometric_sequence_problem_l212_212212


namespace train_length_l212_212545

-- Define the given speeds and time
def train_speed_km_per_h := 25
def man_speed_km_per_h := 2
def crossing_time_sec := 36

-- Convert speeds to m/s
def km_per_h_to_m_per_s (v : ℕ) : ℕ := (v * 1000) / 3600
def train_speed_m_per_s := km_per_h_to_m_per_s train_speed_km_per_h
def man_speed_m_per_s := km_per_h_to_m_per_s man_speed_km_per_h

-- Define the relative speed in m/s
def relative_speed_m_per_s := train_speed_m_per_s + man_speed_m_per_s

-- Theorem to prove the length of the train
theorem train_length : (relative_speed_m_per_s * crossing_time_sec) = 270 :=
by
  -- sorry is used to skip the proof
  sorry

end train_length_l212_212545


namespace remainder_when_y_squared_divided_by_30_l212_212372

theorem remainder_when_y_squared_divided_by_30 (y : ℤ) :
  6 * y ≡ 12 [ZMOD 30] → 5 * y ≡ 25 [ZMOD 30] → y ^ 2 ≡ 19 [ZMOD 30] :=
  by
  intro h1 h2
  sorry

end remainder_when_y_squared_divided_by_30_l212_212372


namespace boris_can_achieve_7_60_cents_l212_212380

/-- Define the conditions as constants -/
def penny_value : ℕ := 1
def dime_value : ℕ := 10
def nickel_value : ℕ := 5
def quarter_value : ℕ := 25

def penny_to_dimes : ℕ := 69
def dime_to_pennies : ℕ := 5
def nickel_to_quarters : ℕ := 120

/-- Function to determine if a value can be produced by a sequence of machine operations -/
def achievable_value (start: ℕ) (target: ℕ) : Prop :=
  ∃ k : ℕ, target = start + k * penny_to_dimes

theorem boris_can_achieve_7_60_cents : achievable_value penny_value 760 :=
  sorry

end boris_can_achieve_7_60_cents_l212_212380


namespace min_value_n_minus_m_l212_212601

noncomputable def f (x : ℝ) : ℝ :=
  if 1 < x then Real.log x else (1 / 2) * x + (1 / 2)

theorem min_value_n_minus_m (m n : ℝ) (hmn : m < n) (hf_eq : f m = f n) : n - m = 3 - 2 * Real.log 2 :=
  sorry

end min_value_n_minus_m_l212_212601


namespace isosceles_triangle_perimeter_l212_212219

theorem isosceles_triangle_perimeter :
  ∃ P : ℕ, (P = 15 ∨ P = 18) ∧ ∀ (a b c : ℕ), (a = 7 ∨ b = 7 ∨ c = 7) ∧ (a = 4 ∨ b = 4 ∨ c = 4) → ((a = 7 ∨ a = 4) ∧ (b = 7 ∨ b = 4) ∧ (c = 7 ∨ c = 4)) ∧ P = a + b + c :=
by
  sorry

end isosceles_triangle_perimeter_l212_212219


namespace average_minutes_run_per_day_l212_212361

theorem average_minutes_run_per_day (f : ℕ) (h_nonzero : f ≠ 0)
  (third_avg fourth_avg fifth_avg : ℕ)
  (third_avg_eq : third_avg = 14)
  (fourth_avg_eq : fourth_avg = 18)
  (fifth_avg_eq : fifth_avg = 8)
  (third_count fourth_count fifth_count : ℕ)
  (third_count_eq : third_count = 3 * fourth_count)
  (fourth_count_eq : fourth_count = f / 2)
  (fifth_count_eq : fifth_count = f) :
  (third_avg * third_count + fourth_avg * fourth_count + fifth_avg * fifth_count) / (third_count + fourth_count + fifth_count) = 38 / 3 :=
by
  sorry

end average_minutes_run_per_day_l212_212361


namespace last_digit_to_appear_mod9_l212_212589

def fib (n : ℕ) : ℕ :=
  if n = 0 then 0
  else if n = 1 then 1
  else fib (n - 1) + fib (n - 2)

def fib_mod9 (n : ℕ) : ℕ :=
  (fib n) % 9

theorem last_digit_to_appear_mod9 :
  ∃ n : ℕ, ∀ m : ℕ, m < n → fib_mod9 m ≠ 0 ∧ fib_mod9 n = 0 :=
sorry

end last_digit_to_appear_mod9_l212_212589


namespace extreme_values_number_of_zeros_l212_212410

noncomputable def f (x : ℝ) : ℝ := 2 * x^3 - 3 * x^2 - 12 * x + 5
noncomputable def g (x m : ℝ) : ℝ := f x - m

theorem extreme_values :
  (∀ x : ℝ, f x ≤ 12) ∧ (f (-1) = 12) ∧ (∀ x : ℝ, -15 ≤ f x) ∧ (f 2 = -15) := 
sorry

theorem number_of_zeros (m : ℝ) :
  (m > 12 ∨ m < -15 → ∃! x : ℝ, g x m = 0) ∧
  (m = 12 ∨ m = -15 → ∃ x y : ℝ, x ≠ y ∧ g x m = 0 ∧ g y m = 0) ∧
  (-15 < m ∧ m < 12 → ∃ x y z : ℝ, x ≠ y ∧ x ≠ z ∧ y ≠ z ∧ g x m = 0 ∧ g y m = 0 ∧ g z m = 0) :=
sorry

end extreme_values_number_of_zeros_l212_212410


namespace mary_thought_animals_l212_212654

-- Definitions based on conditions
def double_counted_sheep : ℕ := 7
def forgotten_pigs : ℕ := 3
def actual_animals : ℕ := 56

-- Statement to be proven
theorem mary_thought_animals (double_counted_sheep forgotten_pigs actual_animals : ℕ) :
  (actual_animals + double_counted_sheep - forgotten_pigs) = 60 := 
by 
  -- Proof goes here
  sorry

end mary_thought_animals_l212_212654


namespace stuffed_animals_total_l212_212327

variable (x y z : ℕ)

theorem stuffed_animals_total :
  let initial := x
  let after_mom := initial + y
  let after_dad := z * after_mom
  let total := after_mom + after_dad
  total = (x + y) * (1 + z) := 
  by 
    let initial := x
    let after_mom := initial + y
    let after_dad := z * after_mom
    let total := after_mom + after_dad
    sorry

end stuffed_animals_total_l212_212327


namespace max_profit_is_45_6_l212_212192

noncomputable def profit_A (x : ℝ) : ℝ := 5.06 * x - 0.15 * x^2
noncomputable def profit_B (x : ℝ) : ℝ := 2 * x

noncomputable def total_profit (x : ℝ) : ℝ :=
  profit_A x + profit_B (15 - x)

theorem max_profit_is_45_6 : 
  ∃ x, 0 ≤ x ∧ x ≤ 15 ∧ total_profit x = 45.6 :=
by
  sorry

end max_profit_is_45_6_l212_212192


namespace percent_of_area_triangle_in_pentagon_l212_212396

-- Defining a structure for the problem statement
structure PentagonAndTriangle where
  s : ℝ -- side length of the equilateral triangle
  side_square : ℝ -- side of the square
  area_triangle : ℝ
  area_square : ℝ
  area_pentagon : ℝ

noncomputable def calculate_areas (s : ℝ) : PentagonAndTriangle :=
  let height_triangle := s * (Real.sqrt 3) / 2
  let area_triangle := Real.sqrt 3 / 4 * s^2
  let area_square := height_triangle^2
  let area_pentagon := area_square + area_triangle
  { s := s, side_square := height_triangle, area_triangle := area_triangle, area_square := area_square, area_pentagon := area_pentagon }

/--
Prove that the percentage of the pentagon's area that is the area of the equilateral triangle is (3 * (Real.sqrt 3 - 1)) / 6 * 100%.
-/
theorem percent_of_area_triangle_in_pentagon 
  (s : ℝ) 
  (pt : PentagonAndTriangle)
  (h₁ : pt = calculate_areas s)
  : pt.area_triangle / pt.area_pentagon = (3 * (Real.sqrt 3 - 1)) / 6 * 100 :=
by
  sorry

end percent_of_area_triangle_in_pentagon_l212_212396


namespace keiths_total_spending_l212_212070

theorem keiths_total_spending :
  let digimon_cost := 4 * 4.45
  let pokemon_cost := 3 * 5.25
  let yugioh_cost := 6 * 3.99
  let mtg_cost := 2 * 6.75
  let baseball_cost := 1 * 6.06
  let total_cost := digimon_cost + pokemon_cost + yugioh_cost + mtg_cost + baseball_cost
  total_cost = 77.05 :=
by
  let digimon_cost := 4 * 4.45
  let pokemon_cost := 3 * 5.25
  let yugioh_cost := 6 * 3.99
  let mtg_cost := 2 * 6.75
  let baseball_cost := 1 * 6.06
  let total_cost := digimon_cost + pokemon_cost + yugioh_cost + mtg_cost + baseball_cost
  have h : total_cost = 77.05 := sorry
  exact h

end keiths_total_spending_l212_212070


namespace range_of_m_l212_212868

theorem range_of_m (m : ℝ) :
  (m + 4 - 4)*(2 + 2 * m - 4) < 0 → 0 < m ∧ m < 1 :=
by
  sorry

end range_of_m_l212_212868


namespace largest_K_is_1_l212_212401

noncomputable def largest_K_vip (K : ℝ) : Prop :=
  ∀ (k : ℝ) (a b c : ℝ), 
  0 ≤ k ∧ k ≤ K → 
  0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c → 
  a^2 + b^2 + c^2 + k * a * b * c = k + 3 → 
  a + b + c ≤ 3

theorem largest_K_is_1 : largest_K_vip 1 :=
sorry

end largest_K_is_1_l212_212401


namespace rectangle_area_l212_212906

theorem rectangle_area (l w r: ℝ) (h1 : l = 2 * r) (h2 : w = r) : l * w = 2 * r^2 :=
by sorry

end rectangle_area_l212_212906


namespace sophia_book_pages_l212_212180

theorem sophia_book_pages:
  ∃ (P : ℕ), (2 / 3 : ℚ) * P = (1 / 3 : ℚ) * P + 30 ∧ P = 90 :=
by
  sorry

end sophia_book_pages_l212_212180


namespace no_common_point_in_all_circles_l212_212915

variable {Point : Type}
variable {Circle : Type}
variable (center : Circle → Point)
variable (contains : Circle → Point → Prop)

-- Given six circles in the plane
variables (C1 C2 C3 C4 C5 C6 : Circle)

-- Condition: None of the circles contain the center of any other circle
axiom condition_1 : ∀ (C D : Circle), C ≠ D → ¬ contains C (center D)

-- Question: Prove that there does not exist a point P that lies in all six circles
theorem no_common_point_in_all_circles : 
  ¬ ∃ (P : Point), (contains C1 P) ∧ (contains C2 P) ∧ (contains C3 P) ∧ (contains C4 P) ∧ (contains C5 P) ∧ (contains C6 P) :=
sorry

end no_common_point_in_all_circles_l212_212915


namespace largest_sum_is_8_over_15_l212_212419

theorem largest_sum_is_8_over_15 :
  max ((1 / 3) + (1 / 6)) (max ((1 / 3) + (1 / 7)) (max ((1 / 3) + (1 / 5)) (max ((1 / 3) + (1 / 9)) ((1 / 3) + (1 / 8))))) = 8 / 15 :=
sorry

end largest_sum_is_8_over_15_l212_212419


namespace pyramid_volume_l212_212946

theorem pyramid_volume (VW WX VZ : ℝ) (h1 : VW = 10) (h2 : WX = 5) (h3 : VZ = 8)
  (h_perp1 : ∀ (V W Z : ℝ), V ≠ W → V ≠ Z → Z ≠ W → W = 0 ∧ Z = 0)
  (h_perp2 : ∀ (V W X : ℝ), V ≠ W → V ≠ X → X ≠ W → W = 0 ∧ X = 0) :
  let area_base := VW * WX
  let height := VZ
  let volume := 1 / 3 * area_base * height
  volume = 400 / 3 := by
  sorry

end pyramid_volume_l212_212946


namespace mike_marbles_l212_212916

theorem mike_marbles (original : ℕ) (given : ℕ) (final : ℕ) 
  (h1 : original = 8) 
  (h2 : given = 4)
  (h3 : final = original - given) : 
  final = 4 :=
by sorry

end mike_marbles_l212_212916


namespace knight_moves_equal_n_seven_l212_212833

def knight_moves (n : ℕ) : ℕ := sorry -- Function to calculate the minimum number of moves for a knight.

theorem knight_moves_equal_n_seven :
  ∀ {n : ℕ}, n = 7 →
    knight_moves n = knight_moves n := by
  -- Conditions: Position on standard checkerboard 
  -- and the knight moves described above.
  sorry

end knight_moves_equal_n_seven_l212_212833


namespace cube_inequality_l212_212359

theorem cube_inequality (a b : ℝ) : a > b ↔ a^3 > b^3 :=
sorry

end cube_inequality_l212_212359


namespace steve_average_speed_l212_212512

-- Define the conditions as constants
def hours1 := 5
def speed1 := 40
def hours2 := 3
def speed2 := 80
def hours3 := 2
def speed3 := 60

-- Define a theorem that calculates average speed and proves the result is 56
theorem steve_average_speed :
  (hours1 * speed1 + hours2 * speed2 + hours3 * speed3) / (hours1 + hours2 + hours3) = 56 := by
  sorry

end steve_average_speed_l212_212512


namespace circle_tangent_l212_212440

theorem circle_tangent (t : ℝ) : 
  (∀ (x y : ℝ), x^2 + y^2 = 4 → (x - t)^2 + y^2 = 1 → |t| = 3) :=
by
  sorry

end circle_tangent_l212_212440


namespace range_of_y_l212_212615

theorem range_of_y :
  ∀ (y x : ℝ), x = 4 - y → (-2 ≤ x ∧ x ≤ -1) → (5 ≤ y ∧ y ≤ 6) :=
by
  intros y x h1 h2
  sorry

end range_of_y_l212_212615


namespace quotient_of_1575_210_l212_212093

theorem quotient_of_1575_210 (a b q : ℕ) (h1 : a = 1575) (h2 : b = a - 1365) (h3 : a % b = 15) : q = 7 :=
by {
  sorry
}

end quotient_of_1575_210_l212_212093


namespace percentage_error_formula_l212_212513

noncomputable def percentage_error_in_area (a b : ℝ) (x y : ℝ) :=
  let actual_area := a * b
  let measured_area := a * (1 + x / 100) * b * (1 + y / 100)
  let error_percentage := ((measured_area - actual_area) / actual_area) * 100
  error_percentage

theorem percentage_error_formula (a b x y : ℝ) :
  percentage_error_in_area a b x y = x + y + (x * y / 100) :=
by
  sorry

end percentage_error_formula_l212_212513


namespace fibonacci_series_sum_l212_212456

noncomputable def fib : ℕ → ℕ
| 0     => 0
| 1     => 1
| (n+2) => fib (n + 1) + fib n

theorem fibonacci_series_sum :
  (∑' n, (fib n : ℝ) / 7^n) = (49 : ℝ) / 287 := 
by
  sorry

end fibonacci_series_sum_l212_212456


namespace lions_after_one_year_l212_212930

def initial_lions : ℕ := 100
def birth_rate : ℕ := 5
def death_rate : ℕ := 1
def months_in_year : ℕ := 12

theorem lions_after_one_year : 
  initial_lions + (birth_rate * months_in_year) - (death_rate * months_in_year) = 148 :=
by
  sorry

end lions_after_one_year_l212_212930


namespace arithmetic_sqrt_sqrt_16_eq_2_l212_212767

theorem arithmetic_sqrt_sqrt_16_eq_2 : Real.sqrt (Real.sqrt 16) = 2 :=
by
  sorry

end arithmetic_sqrt_sqrt_16_eq_2_l212_212767


namespace Theorem3_l212_212377

theorem Theorem3 {f g : ℝ → ℝ} (T1_eq_1 : ∀ x, f (x + 1) = f x)
  (m : ℕ) (h_g_periodic : ∀ x, g (x + 1 / m) = g x) (hm : m > 1) :
  ∃ k : ℕ, k > 0 ∧ (k = 1 ∨ (k ≠ m ∧ ¬(m % k = 0))) ∧ 
    (∀ x, (f x + g x) = (f (x + 1 / k) + g (x + 1 / k))) := 
sorry

end Theorem3_l212_212377


namespace solve_quadratic_l212_212627

theorem solve_quadratic : ∃ x : ℚ, x > 0 ∧ 3 * x^2 + 7 * x - 20 = 0 ∧ x = 5/3 := 
by
  sorry

end solve_quadratic_l212_212627


namespace greatest_3_digit_base_8_divisible_by_7_l212_212176

open Nat

def is_3_digit_base_8 (n : ℕ) : Prop := n < 8^3

def is_divisible_by_7 (n : ℕ) : Prop := 7 ∣ n

theorem greatest_3_digit_base_8_divisible_by_7 :
  ∃ x : ℕ, is_3_digit_base_8 x ∧ is_divisible_by_7 x ∧ x = 7 * (8 * (8 * 7 + 7) + 7) :=
by
  sorry

end greatest_3_digit_base_8_divisible_by_7_l212_212176


namespace sam_original_puppies_count_l212_212493

theorem sam_original_puppies_count 
  (spotted_puppies_start : ℕ)
  (non_spotted_puppies_start : ℕ)
  (spotted_puppies_given : ℕ)
  (non_spotted_puppies_given : ℕ)
  (spotted_puppies_left : ℕ)
  (non_spotted_puppies_left : ℕ)
  (h1 : spotted_puppies_start = 8)
  (h2 : non_spotted_puppies_start = 5)
  (h3 : spotted_puppies_given = 2)
  (h4 : non_spotted_puppies_given = 3)
  (h5 : spotted_puppies_left = spotted_puppies_start - spotted_puppies_given)
  (h6 : non_spotted_puppies_left = non_spotted_puppies_start - non_spotted_puppies_given)
  (h7 : spotted_puppies_left = 6)
  (h8 : non_spotted_puppies_left = 2) :
  spotted_puppies_start + non_spotted_puppies_start = 13 :=
by
  sorry

end sam_original_puppies_count_l212_212493


namespace solve_equation1_solve_equation2_solve_equation3_l212_212167

-- For equation x^2 + 2x = 5
theorem solve_equation1 (x : ℝ) : x^2 + 2 * x = 5 ↔ (x = -1 + Real.sqrt 6) ∨ (x = -1 - Real.sqrt 6) :=
sorry

-- For equation x^2 - 2x - 1 = 0
theorem solve_equation2 (x : ℝ) : x^2 - 2 * x - 1 = 0 ↔ (x = 1 + Real.sqrt 2) ∨ (x = 1 - Real.sqrt 2) :=
sorry

-- For equation 2x^2 + 3x - 5 = 0
theorem solve_equation3 (x : ℝ) : 2 * x^2 + 3 * x - 5 = 0 ↔ (x = -5 / 2) ∨ (x = 1) :=
sorry

end solve_equation1_solve_equation2_solve_equation3_l212_212167


namespace arrangement_count_l212_212726

-- Define the problem conditions: 3 male students and 2 female students.
def male_students : ℕ := 3
def female_students : ℕ := 2
def total_students : ℕ := male_students + female_students

-- Define the condition that female students do not stand at either end.
def valid_positions_for_female : Finset ℕ := {1, 2, 3}
def valid_positions_for_male : Finset ℕ := {0, 4}

-- Theorem statement: the total number of valid arrangements is 36.
theorem arrangement_count : ∃ (n : ℕ), n = 36 := sorry

end arrangement_count_l212_212726


namespace part1_part2_l212_212044

noncomputable def f (x a : ℝ) : ℝ := abs (x - 1) - 2 * abs (x + a)

theorem part1 (x : ℝ) : (∃ a, a = 1) → f x 1 > 1 ↔ -2 < x ∧ x < -(2/3) := by
  sorry

theorem part2 (a : ℝ) : (∀ x, 2 ≤ x → x ≤ 3 → f x a > 0) ↔ (-5/2) < a ∧ a < -2 := by
  sorry

end part1_part2_l212_212044


namespace simplified_expression_evaluates_to_2_l212_212477

-- Definitions based on given conditions:
def x := 2 -- where x = (1/2)^(-1)
def y := 1 -- where y = (-2023)^0

-- Main statement to prove:
theorem simplified_expression_evaluates_to_2 :
  ((2 * x - y) / (x + y) - (x * x - 2 * x * y + y * y) / (x * x - y * y)) / (x - y) / (x + y) = 2 :=
by
  sorry

end simplified_expression_evaluates_to_2_l212_212477


namespace volunteers_per_class_l212_212392

theorem volunteers_per_class (total_needed volunteers teachers_needed : ℕ) (classes : ℕ)
    (h_total : total_needed = 50) (h_teachers : teachers_needed = 13) (h_more_needed : volunteers = 7) (h_classes : classes = 6) :
  (total_needed - teachers_needed - volunteers) / classes = 5 :=
by
  -- calculation and simplification
  sorry

end volunteers_per_class_l212_212392


namespace tangent_lines_inequality_l212_212006

theorem tangent_lines_inequality (k k1 k2 b b1 b2 : ℝ)
  (h1 : k = - (b * b) / 4)
  (h2 : k1 = - (b1 * b1) / 4)
  (h3 : k2 = - (b2 * b2) / 4)
  (h4 : b = b1 + b2) :
  k ≥ 2 * (k1 + k2) := sorry

end tangent_lines_inequality_l212_212006


namespace find_coefficients_l212_212911

theorem find_coefficients (c d : ℝ)
  (h : ∃ u v : ℝ, u ≠ v ∧ (u^3 + c * u^2 + 10 * u + 4 = 0) ∧ (v^3 + c * v^2 + 10 * v + 4 = 0)
     ∧ (u^3 + d * u^2 + 13 * u + 5 = 0) ∧ (v^3 + d * v^2 + 13 * v + 5 = 0)) :
  (c, d) = (7, 8) :=
by
  sorry

end find_coefficients_l212_212911


namespace mul_65_35_l212_212850

theorem mul_65_35 : (65 * 35) = 2275 := by
  -- define a and b
  let a := 50
  let b := 15
  -- use the equivalence (a + b) and (a - b)
  have h1 : 65 = a + b := by rfl
  have h2 : 35 = a - b := by rfl
  -- use the difference of squares formula
  have h_diff_squares : (a + b) * (a - b) = a^2 - b^2 := by sorry
  -- calculate each square
  have ha_sq : a^2 = 2500 := by sorry
  have hb_sq : b^2 = 225 := by sorry
  -- combine the results
  have h_result : a^2 - b^2 = 2500 - 225 := by sorry
  -- finish the proof
  have final_result : (65 * 35) = 2275 := by sorry
  exact final_result

end mul_65_35_l212_212850


namespace matrix_determinant_zero_l212_212154

noncomputable def matrix_example : Matrix (Fin 3) (Fin 3) ℝ := 
  ![
    ![Real.sin 1, Real.sin 2, Real.sin 3],
    ![Real.sin 4, Real.sin 5, Real.sin 6],
    ![Real.sin 7, Real.sin 8, Real.sin 9]
  ]

theorem matrix_determinant_zero : matrix_example.det = 0 := 
by 
  sorry

end matrix_determinant_zero_l212_212154


namespace find_reflection_line_l212_212953

/-*
Triangle ABC has vertices with coordinates A(2,3), B(7,8), and C(-4,6).
The triangle is reflected about line L.
The image points are A'(2,-5), B'(7,-10), and C'(-4,-8).
Prove that the equation of line L is y = -1.
*-/
theorem find_reflection_line :
  ∃ (L : ℝ), (∀ (x : ℝ), (∃ (k : ℝ), L = k) ∧ (L = -1)) :=
by sorry

end find_reflection_line_l212_212953


namespace seulgi_stack_higher_l212_212340

-- Define the conditions
def num_red_boxes : ℕ := 15
def num_yellow_boxes : ℕ := 20
def height_red_box : ℝ := 4.2
def height_yellow_box : ℝ := 3.3

-- Define the total height for each stack
def total_height_hyunjeong : ℝ := num_red_boxes * height_red_box
def total_height_seulgi : ℝ := num_yellow_boxes * height_yellow_box

-- Lean statement to prove the comparison of their heights
theorem seulgi_stack_higher : total_height_seulgi > total_height_hyunjeong :=
by
  -- Proof will be inserted here
  sorry

end seulgi_stack_higher_l212_212340


namespace hour_minute_hand_coincide_at_l212_212606

noncomputable def coinciding_time : ℚ :=
  90 / (6 - 0.5)

theorem hour_minute_hand_coincide_at : coinciding_time = 16 + 4 / 11 := 
  sorry

end hour_minute_hand_coincide_at_l212_212606


namespace carolyn_removal_sum_correct_l212_212461

-- Define the initial conditions
def n : Nat := 10
def initialList : List Nat := List.range (n + 1)  -- equals [0, 1, 2, ..., 10]

-- Given that Carolyn removes specific numbers based on the game rules
def carolynRemovals : List Nat := [6, 10, 8]

-- Sum of numbers removed by Carolyn
def carolynRemovalSum : Nat := carolynRemovals.sum

-- Theorem stating the sum of numbers removed by Carolyn
theorem carolyn_removal_sum_correct : carolynRemovalSum = 24 := by
  sorry

end carolyn_removal_sum_correct_l212_212461


namespace production_days_l212_212351

theorem production_days (n : ℕ) (P : ℕ) (h1: P = n * 50) 
    (h2: (P + 110) / (n + 1) = 55) : n = 11 :=
by
  sorry

end production_days_l212_212351


namespace value_of_a_l212_212323

theorem value_of_a (m : ℝ) :
  (¬ ∃ x : ℝ, x^2 + 2*x + m ≤ 0) → m > 1 :=
sorry

end value_of_a_l212_212323


namespace infinitenat_not_sum_square_prime_l212_212733

theorem infinitenat_not_sum_square_prime : ∀ k : ℕ, ¬ ∃ (n : ℕ) (p : ℕ), Prime p ∧ (3 * k + 2) ^ 2 = n ^ 2 + p :=
by
  intro k
  sorry

end infinitenat_not_sum_square_prime_l212_212733


namespace corrected_mean_is_124_931_l212_212484

/-
Given:
- original_mean : Real = 125.6
- num_observations : Nat = 100
- incorrect_obs1 : Real = 95.3
- incorrect_obs2 : Real = -15.9
- correct_obs1 : Real = 48.2
- correct_obs2 : Real = -35.7

Prove:
- new_mean == 124.931
-/

noncomputable def original_mean : ℝ := 125.6
def num_observations : ℕ := 100
noncomputable def incorrect_obs1 : ℝ := 95.3
noncomputable def incorrect_obs2 : ℝ := -15.9
noncomputable def correct_obs1 : ℝ := 48.2
noncomputable def correct_obs2 : ℝ := -35.7

noncomputable def incorrect_total_sum : ℝ := original_mean * num_observations
noncomputable def sum_incorrect_obs : ℝ := incorrect_obs1 + incorrect_obs2
noncomputable def sum_correct_obs : ℝ := correct_obs1 + correct_obs2
noncomputable def corrected_total_sum : ℝ := incorrect_total_sum - sum_incorrect_obs + sum_correct_obs
noncomputable def new_mean : ℝ := corrected_total_sum / num_observations

theorem corrected_mean_is_124_931 : new_mean = 124.931 := sorry

end corrected_mean_is_124_931_l212_212484


namespace roots_of_cubic_eq_l212_212976

theorem roots_of_cubic_eq (r s t a b c d : ℂ) (h1 : a ≠ 0) (h2 : r ≠ 0) (h3 : s ≠ 0) 
  (h4 : t ≠ 0) (hrst : ∀ x : ℂ, a * x ^ 3 + b * x ^ 2 + c * x + d = 0 → (x = r ∨ x = s ∨ x = t) ∧ (x = r <-> r + s + t - x = -b / a)) 
  (Vieta1 : r + s + t = -b / a) (Vieta2 : r * s + r * t + s * t = c / a) (Vieta3 : r * s * t = -d / a) :
  (1 / r ^ 3 + 1 / s ^ 3 + 1 / t ^ 3 = c ^ 3 / d ^ 3) := 
by sorry

end roots_of_cubic_eq_l212_212976


namespace find_b_l212_212938

theorem find_b (x y z a b : ℝ) (h1 : x + y = 2) (h2 : xy - z^2 = a) (h3 : b = x + y + z) : b = 2 :=
by
  sorry

end find_b_l212_212938


namespace rectangle_long_side_eq_12_l212_212050

theorem rectangle_long_side_eq_12 (s : ℕ) (a b : ℕ) (congruent_triangles : true) (h : a + b = s) (short_side_is_8 : s = 8) : a + b + 4 = 12 :=
by
  sorry

end rectangle_long_side_eq_12_l212_212050


namespace xiaoli_time_l212_212444

variable {t : ℕ} -- Assuming t is a natural number (time in seconds)

theorem xiaoli_time (record_time : ℕ) (t_non_break : t ≥ record_time) (h : record_time = 14) : t ≥ 14 :=
by
  rw [h] at t_non_break
  exact t_non_break

end xiaoli_time_l212_212444


namespace adam_earnings_correct_l212_212834

def total_earnings (lawns_mowed lawns_to_mow : ℕ) (lawn_pay : ℕ)
                   (cars_washed cars_to_wash : ℕ) (car_pay_euros : ℕ) (euro_to_dollar : ℝ)
                   (dogs_walked dogs_to_walk : ℕ) (dog_pay_pesos : ℕ) (peso_to_dollar : ℝ) : ℝ :=
  let lawn_earnings := lawns_mowed * lawn_pay
  let car_earnings := (cars_washed * car_pay_euros : ℝ) * euro_to_dollar
  let dog_earnings := (dogs_walked * dog_pay_pesos : ℝ) * peso_to_dollar
  lawn_earnings + car_earnings + dog_earnings

theorem adam_earnings_correct :
  total_earnings 4 12 9 4 6 10 1.1 3 4 50 0.05 = 87.5 :=
by
  sorry

end adam_earnings_correct_l212_212834


namespace cubic_polynomial_at_zero_l212_212678

noncomputable def f (x : ℝ) : ℝ := by sorry

theorem cubic_polynomial_at_zero :
  (∃ f : ℝ → ℝ, f 2 = 15 ∨ f 2 = -15 ∧
                 f 4 = 15 ∨ f 4 = -15 ∧
                 f 5 = 15 ∨ f 5 = -15 ∧
                 f 6 = 15 ∨ f 6 = -15 ∧
                 f 8 = 15 ∨ f 8 = -15 ∧
                 f 9 = 15 ∨ f 9 = -15 ∧
                 ∀ x, ∃ c a b d, f x = c * x^3 + a * x^2 + b * x + d ) →
  |f 0| = 135 :=
by sorry

end cubic_polynomial_at_zero_l212_212678


namespace remainder_of_k_div_11_l212_212874

theorem remainder_of_k_div_11 {k : ℕ} (hk1 : k % 5 = 2) (hk2 : k % 6 = 5)
  (hk3 : 0 ≤ k % 7 ∧ k % 7 < 7) (hk4 : k < 38) : (k % 11) = 6 := 
by
  sorry

end remainder_of_k_div_11_l212_212874


namespace desk_length_l212_212811

theorem desk_length (width perimeter length : ℤ) (h1 : width = 9) (h2 : perimeter = 46) (h3 : perimeter = 2 * (length + width)) : length = 14 :=
by
  rw [h1, h2] at h3
  sorry

end desk_length_l212_212811


namespace binomial_constant_term_l212_212701

theorem binomial_constant_term : 
  (∃ c : ℕ, ∀ x : ℝ, (x + (1 / (3 * x)))^8 = c * (x ^ (4 * 2 - 8) / 3)) → 
  ∃ c : ℕ, c = 28 :=
sorry

end binomial_constant_term_l212_212701


namespace negative_values_of_x_l212_212279

theorem negative_values_of_x :
  ∃ (n : ℕ), 1 ≤ n ∧ n < 15 ∧ ∀ (x : ℤ), x = n^2 - 200 → x < 0 ∧ (∃k : ℕ, k = 14) :=
by
  sorry

end negative_values_of_x_l212_212279


namespace neg_square_result_l212_212407

-- This definition captures the algebraic expression and its computation rule.
theorem neg_square_result (a : ℝ) : -((-3 * a) ^ 2) = -9 * (a ^ 2) := 
by
  sorry

end neg_square_result_l212_212407


namespace exists_unequal_m_n_l212_212971

theorem exists_unequal_m_n (a b c : ℕ → ℕ) :
  ∃ (m n : ℕ), m ≠ n ∧ a m ≥ a n ∧ b m ≥ b n ∧ c m ≥ c n :=
sorry

end exists_unequal_m_n_l212_212971


namespace polynomial_solution_l212_212693

theorem polynomial_solution (P : ℝ → ℝ) (h₀ : P 0 = 0) (h₁ : ∀ x : ℝ, P x = (1/2) * (P (x+1) + P (x-1))) :
  ∃ a : ℝ, ∀ x : ℝ, P x = a * x :=
sorry

end polynomial_solution_l212_212693


namespace sqrt_square_l212_212229

theorem sqrt_square (n : ℝ) : (Real.sqrt 2023) ^ 2 = 2023 :=
by
  sorry

end sqrt_square_l212_212229


namespace Arianna_time_at_work_l212_212188

theorem Arianna_time_at_work : 
  (24 - (5 + 13)) = 6 := 
by 
  sorry

end Arianna_time_at_work_l212_212188


namespace smallest_base_for_100_l212_212103

theorem smallest_base_for_100 : ∃ b : ℕ, (b^2 ≤ 100 ∧ 100 < b^3) ∧ ∀ c : ℕ, (c^2 ≤ 100 ∧ 100 < c^3) → b ≤ c :=
by
  use 5
  sorry

end smallest_base_for_100_l212_212103


namespace minimum_photos_needed_l212_212809

theorem minimum_photos_needed 
  (total_photos : ℕ) 
  (photos_IV : ℕ)
  (photos_V : ℕ) 
  (photos_VI : ℕ) 
  (photos_VII : ℕ) 
  (photos_I_III : ℕ) 
  (H : total_photos = 130)
  (H_IV : photos_IV = 35)
  (H_V : photos_V = 30)
  (H_VI : photos_VI = 25)
  (H_VII : photos_VII = 20)
  (H_I_III : photos_I_III = total_photos - (photos_IV + photos_V + photos_VI + photos_VII)) :
  77 = 77 :=
by
  sorry

end minimum_photos_needed_l212_212809


namespace ways_to_divide_8_friends_l212_212194

theorem ways_to_divide_8_friends : (4 : ℕ) ^ 8 = 65536 := by
  sorry

end ways_to_divide_8_friends_l212_212194


namespace faye_scored_47_pieces_l212_212830

variable (X : ℕ) -- X is the number of pieces of candy Faye scored on Halloween.

-- Definitions based on the conditions
def initial_candy_count (X : ℕ) : ℕ := X - 25
def after_sister_gave_40 (X : ℕ) : ℕ := initial_candy_count X + 40
def current_candy_count (X : ℕ) : ℕ := after_sister_gave_40 X

-- Theorem to prove the number of pieces of candy Faye scored on Halloween
theorem faye_scored_47_pieces (h : current_candy_count X = 62) : X = 47 :=
by
  sorry

end faye_scored_47_pieces_l212_212830


namespace inequality_proof_l212_212159

theorem inequality_proof (x y z : ℝ) : 
  (x^2 / (x^2 + 2 * y * z)) + (y^2 / (y^2 + 2 * z * x)) + (z^2 / (z^2 + 2 * x * y)) ≥ 1 := 
by
  sorry

end inequality_proof_l212_212159


namespace mul_mixed_number_eq_l212_212970

theorem mul_mixed_number_eq :
  99 + 24 / 25 * -5 = -499 - 4 / 5 :=
by
  sorry

end mul_mixed_number_eq_l212_212970


namespace dice_sum_not_possible_l212_212547

   theorem dice_sum_not_possible (a b c d : ℕ) :
     (1 ≤ a ∧ a ≤ 6) → (1 ≤ b ∧ b ≤ 6) → (1 ≤ c ∧ c ≤ 6) → (1 ≤ d ∧ d ≤ 6) →
     (a * b * c * d = 360) → ¬ (a + b + c + d = 20) :=
   by
     intros ha hb hc hd prod eq_sum
     -- Proof skipped
     sorry
   
end dice_sum_not_possible_l212_212547


namespace orvin_balloons_l212_212813

def regular_price : ℕ := 2
def total_money_initial := 42 * regular_price
def pair_cost := regular_price + (regular_price / 2)
def pairs := total_money_initial / pair_cost
def balloons_from_sale := pairs * 2

def extra_money : ℕ := 18
def price_per_additional_balloon := 2 * regular_price
def additional_balloons := extra_money / price_per_additional_balloon
def greatest_number_of_balloons := balloons_from_sale + additional_balloons

theorem orvin_balloons (pairs balloons_from_sale additional_balloons greatest_number_of_balloons : ℕ) :
  pairs * 2 = 56 →
  additional_balloons = 4 →
  greatest_number_of_balloons = 60 :=
by
  sorry

end orvin_balloons_l212_212813


namespace factory_production_system_l212_212818

theorem factory_production_system (x y : ℕ) (h1 : x + y = 95)
    (h2 : 8*x - 22*y = 0) :
    16*x - 22*y = 0 :=
by
  sorry

end factory_production_system_l212_212818


namespace simplify_expression_l212_212913

theorem simplify_expression : (245^2 - 225^2) / 20 = 470 := by
  sorry

end simplify_expression_l212_212913


namespace mass_percentage_C_in_C6H8Ox_undetermined_l212_212635

-- Define the molar masses of Carbon, Hydrogen, and Oxygen
def molar_mass_C : ℝ := 12.01
def molar_mass_H : ℝ := 1.008
def molar_mass_O : ℝ := 16.00

-- Define the molecular formula
def molar_mass_C6H8O6 : ℝ := (6 * molar_mass_C) + (8 * molar_mass_H) + (6 * molar_mass_O)

-- Given the mass percentage of Carbon in C6H8O6
def mass_percentage_C_in_C6H8O6 : ℝ := 40.91

-- Problem Definition
theorem mass_percentage_C_in_C6H8Ox_undetermined (x : ℕ) : 
  x ≠ 6 → ¬ (∃ p : ℝ, p = (6 * molar_mass_C) / ((6 * molar_mass_C) + (8 * molar_mass_H) + x * molar_mass_O) * 100) :=
by
  intro h1 h2
  sorry

end mass_percentage_C_in_C6H8Ox_undetermined_l212_212635


namespace expansion_dissimilar_terms_count_l212_212968

def number_of_dissimilar_terms (n k : ℕ) : ℕ :=
  Nat.choose (n + k - 1) (k - 1)

theorem expansion_dissimilar_terms_count :
  number_of_dissimilar_terms 7 4 = 120 := by
  sorry

end expansion_dissimilar_terms_count_l212_212968


namespace michael_clean_times_in_one_year_l212_212980

-- Definitions from the conditions
def baths_per_week : ℕ := 2
def showers_per_week : ℕ := 1
def weeks_per_year : ℕ := 52

-- Theorem statement for the proof problem
theorem michael_clean_times_in_one_year :
  (baths_per_week + showers_per_week) * weeks_per_year = 156 :=
by
  sorry

end michael_clean_times_in_one_year_l212_212980


namespace prime_p_satisfies_conditions_l212_212366

theorem prime_p_satisfies_conditions (p : ℕ) (hp : Nat.Prime p) (h1 : Nat.Prime (4 * p^2 + 1)) (h2 : Nat.Prime (6 * p^2 + 1)) : p = 5 :=
sorry

end prime_p_satisfies_conditions_l212_212366


namespace can_form_isosceles_triangle_with_given_sides_l212_212802

-- Define a structure for the sides of a triangle
structure Triangle (α : Type _) :=
  (a b c : α)

-- Define the predicate for the triangle inequality
def triangle_inequality {α : Type _} [LinearOrder α] [Add α] (t : Triangle α) : Prop :=
  t.a + t.b > t.c ∧ t.a + t.c > t.b ∧ t.b + t.c > t.a

-- Define the predicate for an isosceles triangle
def is_isosceles {α : Type _} [DecidableEq α] (t : Triangle α) : Prop :=
  t.a = t.b ∨ t.a = t.c ∨ t.b = t.c

-- Define the main theorem which checks if the given sides can form an isosceles triangle
theorem can_form_isosceles_triangle_with_given_sides
  (t : Triangle ℕ)
  (h_tri : triangle_inequality t)
  (h_iso : is_isosceles t) :
  t = ⟨2, 2, 1⟩ :=
  sorry

end can_form_isosceles_triangle_with_given_sides_l212_212802


namespace cubic_eq_real_roots_roots_product_eq_neg_nine_l212_212902

theorem cubic_eq_real_roots :
  (∀ x, 0 ≤ x ∧ x ≤ Real.sqrt 3 →
    abs (x^3 + (3 / 2) * (1 - a) * x^2 - 3 * a * x + b) ≤ 1) →
  (∃ x1 x2 x3 : ℝ, x1 < x2 ∧ x2 < x3 ∧ 
    x1^3 + (3 / 2) * (1 - a) * x1^2 - 3 * a * x1 + b = 0 ∧
    x2^3 + (3 / 2) * (1 - a) * x2^2 - 3 * a * x2 + b = 0 ∧
    x3^3 + (3 / 2) * (1 - a) * x3^2 - 3 * a * x3 + b = 0) :=
sorry

theorem roots_product_eq_neg_nine :
  let a := 1
  let b := 1
  (∀ x, 0 ≤ x ∧ x ≤ Real.sqrt 3 →
    abs (x^3 + (3 / 2) * (1 - a) * x^2 - 3 * a * x + b) ≤ 1) →
  (∃ x1 x2 x3 : ℝ, x1 < x2 ∧ x2 < x3 ∧ 
    x1^3 - 3 * x1 + 1 = 0 ∧
    x2^3 - 3 * x2 + 1 = 0 ∧
    x3^3 - 3 * x3 + 1 = 0 ∧
    (x1^2 - 2 - x2) * (x2^2 - 2 - x3) * (x3^2 - 2 - x1) = -9) :=
sorry

end cubic_eq_real_roots_roots_product_eq_neg_nine_l212_212902


namespace number_of_yellow_crayons_l212_212225

theorem number_of_yellow_crayons : 
  ∃ (R B Y : ℕ), 
  R = 14 ∧ 
  B = R + 5 ∧ 
  Y = 2 * B - 6 ∧ 
  Y = 32 :=
by
  sorry

end number_of_yellow_crayons_l212_212225


namespace fewer_bands_l212_212690

theorem fewer_bands (J B Y : ℕ) (h1 : J = B + 10) (h2 : B - 4 = 8) (h3 : Y = 24) :
  Y - J = 2 :=
sorry

end fewer_bands_l212_212690


namespace complement_A_union_B_l212_212974

-- Define the universal set U, and the sets A and B
def U : Set ℕ := {1, 2, 3, 4, 5, 6, 7, 8}
def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {3, 4, 5, 6}

-- Lean statement to prove the complement of A ∪ B with respect to U
theorem complement_A_union_B : U \ (A ∪ B) = {7, 8} :=
by
sorry

end complement_A_union_B_l212_212974


namespace find_value_of_x_squared_and_reciprocal_squared_l212_212656

theorem find_value_of_x_squared_and_reciprocal_squared (x : ℝ) (h : x + 1/x = 2) : x^2 + (1/x)^2 = 2 := 
sorry

end find_value_of_x_squared_and_reciprocal_squared_l212_212656


namespace minimum_value_2_only_in_option_b_l212_212307

noncomputable def option_a (x : ℝ) : ℝ := x + 1 / x
noncomputable def option_b (x : ℝ) : ℝ := 3^x + 3^(-x)
noncomputable def option_c (x : ℝ) : ℝ := (Real.log x) + 1 / (Real.log x)
noncomputable def option_d (x : ℝ) : ℝ := (Real.sin x) + 1 / (Real.sin x)

theorem minimum_value_2_only_in_option_b :
  (∀ x > 0, option_a x ≠ 2) ∧
  (∃ x, option_b x = 2) ∧
  (∀ x (h: 0 < x) (h' : x < 1), option_c x ≠ 2) ∧
  (∀ x (h: 0 < x) (h' : x < π / 2), option_d x ≠ 2) :=
by
  sorry

end minimum_value_2_only_in_option_b_l212_212307


namespace luke_money_at_end_of_june_l212_212043

noncomputable def initial_money : ℝ := 48
noncomputable def february_money : ℝ := initial_money - 0.30 * initial_money
noncomputable def march_money : ℝ := february_money - 11 + 21 + 50 * 1.20

noncomputable def april_savings : ℝ := 0.10 * march_money
noncomputable def april_money : ℝ := (march_money - april_savings) - 10 * 1.18 + 0.05 * (march_money - april_savings)

noncomputable def may_savings : ℝ := 0.15 * april_money
noncomputable def may_money : ℝ := (april_money - may_savings) + 100 * 1.22 - 0.25 * ((april_money - may_savings) + 100 * 1.22)

noncomputable def june_savings : ℝ := 0.10 * may_money
noncomputable def june_money : ℝ := (may_money - june_savings) - 0.08 * (may_money - june_savings)
noncomputable def final_money : ℝ := june_money + 0.06 * (may_money - june_savings)

theorem luke_money_at_end_of_june : final_money = 128.15 := sorry

end luke_money_at_end_of_june_l212_212043


namespace not_p_is_sufficient_but_not_necessary_for_q_l212_212619

-- Definitions for the conditions
def p (x : ℝ) : Prop := x^2 > 4
def q (x : ℝ) : Prop := x ≤ 2

-- Definition of ¬p based on the solution derived
def not_p (x : ℝ) : Prop := -2 ≤ x ∧ x ≤ 2

-- The theorem statement
theorem not_p_is_sufficient_but_not_necessary_for_q :
  ∀ x : ℝ, (not_p x → q x) ∧ ¬(q x → not_p x) := sorry

end not_p_is_sufficient_but_not_necessary_for_q_l212_212619


namespace bottle_caps_per_group_l212_212062

theorem bottle_caps_per_group (total_caps : ℕ) (num_groups : ℕ) (caps_per_group : ℕ) 
  (h1 : total_caps = 12) (h2 : num_groups = 6) : 
  total_caps / num_groups = caps_per_group := by
  sorry

end bottle_caps_per_group_l212_212062


namespace find_b_l212_212650

theorem find_b (S : ℕ → ℝ) (a : ℕ → ℝ) (h1 : ∀ n, S n = 3^n + b)
  (h2 : ∀ n ≥ 2, a n = S n - S (n-1))
  (h_geometric : ∃ r, ∀ n ≥ 1, a n = a 1 * r^(n-1)) : b = -1 := 
sorry

end find_b_l212_212650


namespace total_seeds_planted_l212_212587

def number_of_flowerbeds : ℕ := 9
def seeds_per_flowerbed : ℕ := 5

theorem total_seeds_planted : number_of_flowerbeds * seeds_per_flowerbed = 45 :=
by
  sorry

end total_seeds_planted_l212_212587


namespace last_digit_is_zero_last_ten_digits_are_zero_l212_212699

-- Condition: The product includes a factor of 10
def includes_factor_of_10 (n : ℕ) : Prop :=
  ∃ k, n = k * 10

-- Conclusion: The last digit of the product must be 0
theorem last_digit_is_zero (n : ℕ) (h : includes_factor_of_10 n) : 
  n % 10 = 0 :=
sorry

-- Condition: The product includes the factors \(5^{10}\) and \(2^{10}\)
def includes_10_to_the_10 (n : ℕ) : Prop :=
  ∃ k, n = k * 10^10

-- Conclusion: The last ten digits of the product must be 0000000000
theorem last_ten_digits_are_zero (n : ℕ) (h : includes_10_to_the_10 n) : 
  n % 10^10 = 0 :=
sorry

end last_digit_is_zero_last_ten_digits_are_zero_l212_212699


namespace calc_value_of_fraction_l212_212638

theorem calc_value_of_fraction :
  (10^9 / (2 * 5^2 * 10^3)) = 20000 := by
  sorry

end calc_value_of_fraction_l212_212638


namespace smallest_reducible_fraction_l212_212899

theorem smallest_reducible_fraction :
  ∃ n : ℕ, 0 < n ∧ (∃ d > 1, d ∣ (n - 17) ∧ d ∣ (7 * n + 8)) ∧ n = 144 := by
  sorry

end smallest_reducible_fraction_l212_212899


namespace solution_set_for_inequality_l212_212080

theorem solution_set_for_inequality (f : ℝ → ℝ) 
  (h_even : ∀ x, f (-x) = f x)
  (h_decreasing : ∀ ⦃x y⦄, 0 < x → x < y → f y < f x)
  (h_f_neg3 : f (-3) = 1) :
  { x | f x < 1 } = { x | x < -3 ∨ 3 < x } := 
by
  -- TODO: Prove this theorem
  sorry

end solution_set_for_inequality_l212_212080


namespace steve_final_height_l212_212152

-- Define the initial height of Steve in inches.
def initial_height : ℕ := 5 * 12 + 6

-- Define how many inches Steve grew.
def growth : ℕ := 6

-- Define Steve's final height after growing.
def final_height : ℕ := initial_height + growth

-- The final height should be 72 inches.
theorem steve_final_height : final_height = 72 := by
  -- we don't provide the proof here
  sorry

end steve_final_height_l212_212152


namespace sum_of_cuberoots_gt_two_l212_212464

theorem sum_of_cuberoots_gt_two {x₁ x₂ : ℝ} (h₁: x₁^3 = 6 / 5) (h₂: x₂^3 = 5 / 6) : x₁ + x₂ > 2 :=
sorry

end sum_of_cuberoots_gt_two_l212_212464


namespace total_tiles_l212_212413

theorem total_tiles (s : ℕ) (h_black_tiles : 2 * s - 1 = 75) : s^2 = 1444 :=
by {
  sorry
}

end total_tiles_l212_212413


namespace quadratic_factor_transformation_l212_212047

theorem quadratic_factor_transformation (x : ℝ) :
  x^2 - 6 * x + 5 = 0 → (x - 3)^2 = 14 := 
by
  sorry

end quadratic_factor_transformation_l212_212047


namespace probability_X_eq_3_l212_212549

def number_of_ways_to_choose (n k : ℕ) : ℕ :=
  Nat.choose n k

def P_X_eq_3 : ℚ :=
  (number_of_ways_to_choose 5 3) * (number_of_ways_to_choose 3 1) / (number_of_ways_to_choose 8 4)

theorem probability_X_eq_3 : P_X_eq_3 = 3 / 7 := by
  sorry

end probability_X_eq_3_l212_212549


namespace blocks_used_for_fenced_area_l212_212303

theorem blocks_used_for_fenced_area
  (initial_blocks : ℕ) (building_blocks : ℕ) (farmhouse_blocks : ℕ) (remaining_blocks : ℕ) :
  initial_blocks = 344 →
  building_blocks = 80 →
  farmhouse_blocks = 123 →
  remaining_blocks = 84 →
  initial_blocks - building_blocks - farmhouse_blocks - remaining_blocks = 57 :=
by
  intros h1 h2 h3 h4
  sorry

end blocks_used_for_fenced_area_l212_212303


namespace find_floor_at_same_time_l212_212914

def timeTaya (n : ℕ) : ℕ := 15 * (n - 22)
def timeJenna (n : ℕ) : ℕ := 120 + 3 * (n - 22)

theorem find_floor_at_same_time (n : ℕ) : n = 32 :=
by
  -- The goal is to show that Taya and Jenna arrive at the same floor at the same time
  have ht : 15 * (n - 22) = timeTaya n := rfl
  have hj : 120 + 3 * (n - 22) = timeJenna n := rfl
  -- equate the times
  have h : timeTaya n = timeJenna n := by sorry
  -- solving the equation for n = 32
  sorry

end find_floor_at_same_time_l212_212914


namespace inscribed_sphere_radius_eq_l212_212720

noncomputable def inscribed_sphere_radius (b α : ℝ) : ℝ :=
  b * (Real.sin α) / (4 * (Real.cos (α / 4))^2)

theorem inscribed_sphere_radius_eq
  (b α : ℝ) 
  (h1 : 0 < b)
  (h2 : 0 < α ∧ α < Real.pi) 
  : inscribed_sphere_radius b α = b * (Real.sin α) / (4 * (Real.cos (α / 4))^2) :=
sorry

end inscribed_sphere_radius_eq_l212_212720


namespace find_n_l212_212004

-- Define the hyperbola and its properties
def hyperbola (m n : ℝ) : Prop :=
  m > 0 ∧ n > 0 ∧ 2 = (m / (m / 2)) ∧ ∃ f : ℝ × ℝ, f = (m, 0)

-- Define the parabola and its properties
def parabola_focus (m : ℝ) : Prop :=
  (m, 0) = (m, 0)

-- The statement we want to prove
theorem find_n (m : ℝ) (n : ℝ) (H_hyperbola : hyperbola m n) (H_parabola : parabola_focus m) : n = 12 :=
sorry

end find_n_l212_212004


namespace num_ways_choose_officers_8_l212_212282

def numWaysToChooseOfficers (n : ℕ) : ℕ := n * (n - 1) * (n - 2)

theorem num_ways_choose_officers_8 : numWaysToChooseOfficers 8 = 336 := by
  sorry

end num_ways_choose_officers_8_l212_212282


namespace parametric_to_standard_line_parametric_to_standard_ellipse_l212_212038

theorem parametric_to_standard_line (t : ℝ) (x y : ℝ) 
  (h₁ : x = 1 - 3 * t)
  (h₂ : y = 4 * t) :
  4 * x + 3 * y - 4 = 0 := by
sorry

theorem parametric_to_standard_ellipse (θ x y : ℝ) 
  (h₁ : x = 5 * Real.cos θ)
  (h₂ : y = 4 * Real.sin θ) :
  (x^2 / 25) + (y^2 / 16) = 1 := by
sorry

end parametric_to_standard_line_parametric_to_standard_ellipse_l212_212038


namespace quadratic_has_real_solutions_l212_212081

theorem quadratic_has_real_solutions (m : ℝ) : 
  (∃ x : ℝ, (m - 2) * x^2 - 2 * x + 1 = 0) → m ≤ 3 := 
by
  sorry

end quadratic_has_real_solutions_l212_212081


namespace karlson_max_eat_chocolates_l212_212768

noncomputable def maximum_chocolates_eaten : ℕ :=
  34 * (34 - 1) / 2

theorem karlson_max_eat_chocolates : maximum_chocolates_eaten = 561 := by
  sorry

end karlson_max_eat_chocolates_l212_212768


namespace how_many_one_halves_in_two_sevenths_l212_212066

theorem how_many_one_halves_in_two_sevenths : (2 / 7) / (1 / 2) = 4 / 7 := by 
  sorry

end how_many_one_halves_in_two_sevenths_l212_212066


namespace quadrilateral_area_is_8_l212_212765

noncomputable section
open Real

def f1 : ℝ × ℝ := (-2, 0)
def f2 : ℝ × ℝ := (2, 0)

def ellipse (x y : ℝ) : Prop := x^2 / 16 + y^2 / 4 = 1

def origin_symmetric (P Q : ℝ × ℝ) : Prop := P.1 = -Q.1 ∧ P.2 = -Q.2

def distance (A B : ℝ × ℝ) : ℝ := sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)

def is_quadrilateral (P Q F1 F2 : ℝ × ℝ) : Prop :=
  ∃ a b c d, a = P ∧ b = F1 ∧ c = Q ∧ d = F2

def area_of_quadrilateral (A B C D : ℝ × ℝ) : ℝ :=
  0.5 * abs (A.1*B.2 + B.1*C.2 + C.1*D.2 + D.1*A.2 - (B.1*A.2 + C.1*B.2 + D.1*C.2 + A.1*D.2))

theorem quadrilateral_area_is_8 (P Q : ℝ × ℝ) :
  ellipse P.1 P.2 →
  ellipse Q.1 Q.2 →
  origin_symmetric P Q →
  distance P Q = distance f1 f2 →
  is_quadrilateral P Q f1 f2 →
  area_of_quadrilateral P f1 Q f2 = 8 := 
by
  sorry

end quadrilateral_area_is_8_l212_212765


namespace all_numbers_even_l212_212059

theorem all_numbers_even
  (A B C D E : ℤ)
  (h1 : (A + B + C) % 2 = 0)
  (h2 : (A + B + D) % 2 = 0)
  (h3 : (A + B + E) % 2 = 0)
  (h4 : (A + C + D) % 2 = 0)
  (h5 : (A + C + E) % 2 = 0)
  (h6 : (A + D + E) % 2 = 0)
  (h7 : (B + C + D) % 2 = 0)
  (h8 : (B + C + E) % 2 = 0)
  (h9 : (B + D + E) % 2 = 0)
  (h10 : (C + D + E) % 2 = 0) :
  (A % 2 = 0) ∧ (B % 2 = 0) ∧ (C % 2 = 0) ∧ (D % 2 = 0) ∧ (E % 2 = 0) :=
sorry

end all_numbers_even_l212_212059


namespace integer_solutions_for_even_ratio_l212_212997

theorem integer_solutions_for_even_ratio (a : ℤ) (h : ∃ k : ℤ, (a = 2 * k * (1011 - k))): 
  a = 1010 ∨ a = 1012 ∨ a = 1008 ∨ a = 1014 ∨ a = 674 ∨ a = 1348 ∨ a = 0 ∨ a = 2022 :=
sorry

end integer_solutions_for_even_ratio_l212_212997


namespace unique_four_letter_sequence_l212_212623

def alphabet_value (c : Char) : ℕ :=
  if 'A' <= c ∧ c <= 'Z' then (c.toNat - 'A'.toNat + 1) else 0

def sequence_product (s : String) : ℕ :=
  s.foldl (λ acc c => acc * alphabet_value c) 1

theorem unique_four_letter_sequence (s : String) :
  sequence_product "WXYZ" = sequence_product s → s = "WXYZ" :=
by
  sorry

end unique_four_letter_sequence_l212_212623


namespace intersection_eq_l212_212174

open Set

variable (A B : Set ℝ)

def setA : A = {x | -3 < x ∧ x < 2} := sorry

def setB : B = {x | x^2 + 4*x - 5 ≤ 0} := sorry

theorem intersection_eq : A ∩ B = {x | -3 < x ∧ x ≤ 1} :=
sorry

end intersection_eq_l212_212174


namespace capital_formula_minimum_m_l212_212181

-- Define initial conditions
def initial_capital : ℕ := 50000  -- in thousand yuan
def annual_growth_rate : ℝ := 0.5
def submission_amount : ℕ := 10000  -- in thousand yuan

-- Define remaining capital after nth year
noncomputable def remaining_capital (n : ℕ) : ℝ :=
  4500 * (3 / 2)^(n - 1) + 2000  -- in thousand yuan

-- Prove the formula for a_n
theorem capital_formula (n : ℕ) : 
  remaining_capital n = 4500 * (3 / 2)^(n - 1) + 2000 := 
by
  sorry

-- Prove the minimum value of m for which a_m > 30000
theorem minimum_m (m : ℕ) : 
  remaining_capital m > 30000 ↔ m ≥ 6 := 
by
  sorry

end capital_formula_minimum_m_l212_212181


namespace p_over_q_at_neg1_l212_212308

-- Definitions of p(x) and q(x) based on given conditions
noncomputable def q (x : ℝ) := (x + 3) * (x - 2)
noncomputable def p (x : ℝ) := 2 * x

-- Define the main function y = p(x) / q(x)
noncomputable def y (x : ℝ) := p x / q x

-- Statement to prove the value of p(-1) / q(-1)
theorem p_over_q_at_neg1 : y (-1) = (1 : ℝ) / 3 :=
by
  sorry

end p_over_q_at_neg1_l212_212308


namespace solve_inequality_l212_212539

theorem solve_inequality :
  {x : ℝ | -x^2 + 5 * x > 6} = {x : ℝ | 2 < x ∧ x < 3} :=
sorry

end solve_inequality_l212_212539


namespace remaining_apples_l212_212987

def initial_apples : ℕ := 20
def shared_apples : ℕ := 7

theorem remaining_apples : initial_apples - shared_apples = 13 :=
by
  sorry

end remaining_apples_l212_212987


namespace arithmetic_mean_reciprocals_first_four_primes_l212_212364

theorem arithmetic_mean_reciprocals_first_four_primes :
  (1/2 + 1/3 + 1/5 + 1/7) / 4 = 247 / 840 :=
by
  sorry

end arithmetic_mean_reciprocals_first_four_primes_l212_212364


namespace pages_wednesday_l212_212281

-- Given conditions as definitions
def borrow_books := 3
def pages_monday := 20
def pages_tuesday := 12
def total_pages := 51

-- Prove that Nico read 19 pages on Wednesday
theorem pages_wednesday :
  let pages_wednesday := total_pages - (pages_monday + pages_tuesday)
  pages_wednesday = 19 :=
by
  sorry

end pages_wednesday_l212_212281


namespace power_of_two_grows_faster_l212_212420

theorem power_of_two_grows_faster (n : ℕ) (h : n ≥ 5) : 2^n > n^2 :=
sorry

end power_of_two_grows_faster_l212_212420


namespace year_weeks_span_l212_212669

theorem year_weeks_span (days_in_year : ℕ) (h1 : days_in_year = 365 ∨ days_in_year = 366) :
  ∃ W : ℕ, (W = 53 ∨ W = 54) ∧ (days_in_year = 365 → W = 53) ∧ (days_in_year = 366 → W = 53 ∨ W = 54) :=
by
  sorry

end year_weeks_span_l212_212669


namespace sum_of_midpoint_xcoords_l212_212383

theorem sum_of_midpoint_xcoords (a b c : ℝ) (h : a + b + c = 15) :
    (a + b) / 2 + (a + c) / 2 + (b + c) / 2 = 15 :=
  by
    sorry

end sum_of_midpoint_xcoords_l212_212383


namespace minimum_percentage_increase_in_mean_replacing_with_primes_l212_212039

def mean (S : List ℤ) : ℚ :=
  (S.sum : ℚ) / S.length

noncomputable def percentage_increase (original new : ℚ) : ℚ :=
  ((new - original) / original) * 100

theorem minimum_percentage_increase_in_mean_replacing_with_primes :
  let F := [-4, -1, 0, 6, 9] 
  let G := [2, 3, 0, 6, 9] 
  percentage_increase (mean F) (mean G) = 100 :=
by {
  let F := [-4, -1, 0, 6, 9] 
  let G := [2, 3, 0, 6, 9] 
  sorry 
}

end minimum_percentage_increase_in_mean_replacing_with_primes_l212_212039


namespace central_angle_of_sector_in_unit_circle_with_area_1_is_2_l212_212711

theorem central_angle_of_sector_in_unit_circle_with_area_1_is_2 :
  ∀ (θ : ℝ), (∀ (r : ℝ), (r = 1) → (1 / 2 * r^2 * θ = 1) → θ = 2) :=
by
  intros θ r hr h
  sorry

end central_angle_of_sector_in_unit_circle_with_area_1_is_2_l212_212711


namespace find_z_l212_212021

theorem find_z (x y k : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hk : k ≠ 0) (h : 1/x + 1/y = k) :
  ∃ z : ℝ, 1/z = k ∧ z = xy/(x + y) :=
by {
  sorry
}

end find_z_l212_212021


namespace mean_of_data_is_5_l212_212016

theorem mean_of_data_is_5 (h : s^2 = (1 / 4) * ((3.2 - x)^2 + (5.7 - x)^2 + (4.3 - x)^2 + (6.8 - x)^2))
  : x = 5 := 
sorry

end mean_of_data_is_5_l212_212016


namespace xy_leq_half_x_squared_plus_y_squared_l212_212939

theorem xy_leq_half_x_squared_plus_y_squared (x y : ℝ) : x * y ≤ (x^2 + y^2) / 2 := 
by 
  sorry

end xy_leq_half_x_squared_plus_y_squared_l212_212939


namespace inequality_1_inequality_2_l212_212239

noncomputable def f (x : ℝ) : ℝ := abs (x - 1) - abs (x - 2)

theorem inequality_1 (x : ℝ) : f x > 2 * x ↔ x < -1/2 :=
sorry

theorem inequality_2 (t : ℝ) :
  (∃ x : ℝ, f x > t ^ 2 - t + 1) ↔ (0 < t ∧ t < 1) :=
sorry

end inequality_1_inequality_2_l212_212239


namespace binomial_coefficient_10_3_l212_212465

-- Define the binomial coefficient
def binomial_coefficient (n r : ℕ) : ℕ := n.choose r

-- Define the given values for n and r
def n : ℕ := 10
def r : ℕ := 3

-- State the theorem
theorem binomial_coefficient_10_3 : binomial_coefficient n r = 120 := 
by {
  sorry -- This is the proof placeholder
}

end binomial_coefficient_10_3_l212_212465


namespace evaluate_f_5_minus_f_neg_5_l212_212344

def f (x : ℝ) : ℝ := x^4 + x^2 + 5 * x

theorem evaluate_f_5_minus_f_neg_5 : f 5 - f (-5) = 50 := by
  sorry

end evaluate_f_5_minus_f_neg_5_l212_212344


namespace jack_walked_distance_l212_212691

def jack_walking_time: ℝ := 1.25
def jack_walking_rate: ℝ := 3.2
def jack_distance_walked: ℝ := 4

theorem jack_walked_distance:
  jack_walking_rate * jack_walking_time = jack_distance_walked :=
by
  sorry

end jack_walked_distance_l212_212691


namespace karsyn_total_payment_l212_212999

def initial_price : ℝ := 600
def discount_rate : ℝ := 0.20
def phone_case_cost : ℝ := 25
def screen_protector_cost : ℝ := 15
def store_discount_rate : ℝ := 0.05
def sales_tax_rate : ℝ := 0.035

noncomputable def total_payment : ℝ :=
  let discounted_price := discount_rate * initial_price
  let total_cost := discounted_price + phone_case_cost + screen_protector_cost
  let store_discount := store_discount_rate * total_cost
  let discounted_total := total_cost - store_discount
  let tax := sales_tax_rate * discounted_total
  discounted_total + tax

theorem karsyn_total_payment : total_payment = 157.32 := by
  sorry

end karsyn_total_payment_l212_212999


namespace unique_positive_integer_solutions_l212_212345

theorem unique_positive_integer_solutions : 
  ∀ (m n : ℕ), 0 < m ∧ 0 < n ∧ 7 ^ m - 3 * 2 ^ n = 1 ↔ (m = 1 ∧ n = 1) ∨ (m = 2 ∧ n = 4) :=
by sorry

end unique_positive_integer_solutions_l212_212345


namespace xyz_cubed_over_xyz_eq_21_l212_212963

open Complex

theorem xyz_cubed_over_xyz_eq_21 {x y z : ℂ} (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0)
  (h1 : x + y + z = 18)
  (h2 : (x - y)^2 + (x - z)^2 + (y - z)^2 = 2 * x * y * z) :
  (x^3 + y^3 + z^3) / (x * y * z) = 21 :=
sorry

end xyz_cubed_over_xyz_eq_21_l212_212963


namespace negation_proposition_l212_212355

theorem negation_proposition:
  (¬ (∀ x : ℝ, (1 ≤ x) → (x^2 - 2*x + 1 ≥ 0))) ↔ (∃ x : ℝ, (1 ≤ x) ∧ (x^2 - 2*x + 1 < 0)) := 
sorry

end negation_proposition_l212_212355


namespace grandma_mushrooms_l212_212139

theorem grandma_mushrooms (M : ℕ) (h₁ : ∀ t : ℕ, t = 2 * M)
                         (h₂ : ∀ p : ℕ, p = 4 * t)
                         (h₃ : ∀ b : ℕ, b = 4 * p)
                         (h₄ : ∀ r : ℕ, r = b / 3)
                         (h₅ : r = 32) :
  M = 3 :=
by
  -- We are expected to fill the steps here to provide the proof if required
  sorry

end grandma_mushrooms_l212_212139


namespace vanessa_score_l212_212777

theorem vanessa_score (total_points team_score other_players_avg_score: ℝ) : 
  total_points = 72 ∧ team_score = 7 ∧ other_players_avg_score = 4.5 → 
  ∃ vanessa_points: ℝ, vanessa_points = 40.5 :=
by
  sorry

end vanessa_score_l212_212777


namespace sum_of_two_squares_iff_double_sum_of_two_squares_l212_212965

theorem sum_of_two_squares_iff_double_sum_of_two_squares (x : ℤ) :
  (∃ a b : ℤ, x = a^2 + b^2) ↔ (∃ u v : ℤ, 2 * x = u^2 + v^2) :=
by
  sorry

end sum_of_two_squares_iff_double_sum_of_two_squares_l212_212965


namespace interest_rate_proven_l212_212255

structure InvestmentProblem where
  P : ℝ  -- Principal amount
  A : ℝ  -- Accumulated amount
  n : ℕ  -- Number of times interest is compounded per year
  t : ℕ  -- Time in years
  rate : ℝ  -- Interest rate per annum (to be proven)

noncomputable def solve_interest_rate (ip : InvestmentProblem) : ℝ :=
  let half_yearly_rate := ip.rate / 2 / 100
  let amount_formula := ip.P * (1 + half_yearly_rate)^(ip.n * ip.t)
  half_yearly_rate

theorem interest_rate_proven :
  ∀ (P A : ℝ) (n t : ℕ), 
  P = 6000 → 
  A = 6615 → 
  n = 2 → 
  t = 1 → 
  solve_interest_rate {P := P, A := A, n := n, t := t, rate := 10.0952} = 10.0952 := 
by 
  intros
  rw [solve_interest_rate]
  sorry

end interest_rate_proven_l212_212255


namespace add_decimals_l212_212164

theorem add_decimals :
  0.0935 + 0.007 + 0.2 = 0.3005 :=
by sorry

end add_decimals_l212_212164


namespace NumFriendsNextToCaraOnRight_l212_212031

open Nat

def total_people : ℕ := 8
def freds_next_to_Cara : ℕ := 7

theorem NumFriendsNextToCaraOnRight (h : total_people = 8) : freds_next_to_Cara = 7 :=
by
  sorry

end NumFriendsNextToCaraOnRight_l212_212031


namespace joan_paid_230_l212_212728

theorem joan_paid_230 (J K : ℝ) (h1 : J + K = 600) (h2 : 2 * J = K + 90) : J = 230 :=
sorry

end joan_paid_230_l212_212728


namespace find_x_positive_multiple_of_8_l212_212770

theorem find_x_positive_multiple_of_8 (x : ℕ) 
  (h1 : ∃ k, x = 8 * k) 
  (h2 : x^2 > 100) 
  (h3 : x < 20) : x = 16 :=
by
  sorry

end find_x_positive_multiple_of_8_l212_212770


namespace original_number_l212_212881

theorem original_number (x : ℝ) (h : x - x / 3 = 36) : x = 54 :=
by
  sorry

end original_number_l212_212881


namespace negation_of_universal_prop_l212_212418

variable (a : ℝ)

theorem negation_of_universal_prop :
  (¬ ∀ x : ℝ, 0 < x → Real.log x = a) ↔ (∃ x : ℝ, 0 < x ∧ Real.log x ≠ a) :=
by
  sorry

end negation_of_universal_prop_l212_212418


namespace cube_volume_l212_212553

theorem cube_volume {V : ℝ} (x : ℝ) (hV : V = x^3) (hA : 2 * V = 6 * x^2) : V = 27 :=
by
  -- Proof goes here
  sorry

end cube_volume_l212_212553


namespace problem_statement_l212_212200

def f (x : ℝ) : ℝ := 5 * x + 2
def g (x : ℝ) : ℝ := 3 * x - 1

theorem problem_statement : g (f (g (f 1))) = 305 :=
by
  sorry

end problem_statement_l212_212200


namespace angle_ratio_in_triangle_l212_212948

theorem angle_ratio_in_triangle
  (triangle : Type)
  (A B C P Q M : triangle)
  (angle : triangle → triangle → triangle → ℝ)
  (ABC_half : angle A B Q = angle Q B C)
  (BP_BQ_bisect_ABC : angle A B P = angle P B Q)
  (BM_bisects_PBQ : angle M B Q = angle M B P)
  : angle M B Q / angle A B Q = 1 / 4 :=
by 
  sorry

end angle_ratio_in_triangle_l212_212948


namespace range_of_function_l212_212427

noncomputable def range_of_y : Set ℝ :=
  {y | ∃ x : ℝ, y = |x + 5| - |x - 3|}

theorem range_of_function : range_of_y = Set.Icc (-2) 12 :=
by
  sorry

end range_of_function_l212_212427


namespace daysRequired_l212_212339

-- Defining the structure of the problem
structure WallConstruction where
  m1 : ℕ    -- Number of men in the first scenario
  d1 : ℕ    -- Number of days in the first scenario
  m2 : ℕ    -- Number of men in the second scenario

-- Given values
def wallConstructionProblem : WallConstruction :=
  WallConstruction.mk 20 5 30

-- The total work constant
def totalWork (wc : WallConstruction) : ℕ :=
  wc.m1 * wc.d1

-- Proving the number of days required for m2 men
theorem daysRequired (wc : WallConstruction) (k : ℕ) : 
  k = totalWork wc → (wc.m2 * (k / wc.m2 : ℚ) = k) → (k / wc.m2 : ℚ) = 3.3 :=
by
  intro h1 h2
  sorry

end daysRequired_l212_212339


namespace sum_of_reciprocals_l212_212763

variables {a b : ℕ}

def HCF (m n : ℕ) : ℕ := m.gcd n
def LCM (m n : ℕ) : ℕ := m.lcm n

theorem sum_of_reciprocals (h_sum : a + b = 55)
                           (h_hcf : HCF a b = 5)
                           (h_lcm : LCM a b = 120) :
  (1 / a : ℚ) + (1 / b) = 11 / 120 :=
sorry

end sum_of_reciprocals_l212_212763


namespace rod_center_of_gravity_shift_l212_212177

noncomputable def rod_shift (l : ℝ) (s : ℝ) : ℝ := 
  |(l / 2) - ((l - s) / 2)| 

theorem rod_center_of_gravity_shift : 
  rod_shift l 80 = 40 := by
  sorry

end rod_center_of_gravity_shift_l212_212177


namespace acres_used_for_corn_l212_212731

-- Define the conditions given in the problem
def total_land : ℕ := 1034
def ratio_beans : ℕ := 5
def ratio_wheat : ℕ := 2
def ratio_corn : ℕ := 4
def total_ratio_parts : ℕ := ratio_beans + ratio_wheat + ratio_corn
def part_size : ℕ := total_land / total_ratio_parts

-- State the theorem to prove that the land used for corn is 376 acres
theorem acres_used_for_corn : (part_size * ratio_corn = 376) :=
  sorry

end acres_used_for_corn_l212_212731


namespace xy_conditions_l212_212936

theorem xy_conditions (x y : ℝ) (h1 : x + 2 * y = 8) (h2 : x * y = 1) : x^2 + 4 * y^2 = 60 :=
by
  sorry

end xy_conditions_l212_212936


namespace sunflower_count_l212_212030

theorem sunflower_count (r l d : ℕ) (t : ℕ) (h1 : r + l + d = 40) (h2 : t = 160) : 
  t - (r + l + d) = 120 := by
  sorry

end sunflower_count_l212_212030


namespace minimize_distance_l212_212611

-- Definitions of points and lines in the Euclidean plane
structure Point : Type :=
(x : ℝ)
(y : ℝ)

-- Line is defined by a point and a direction vector
structure Line : Type :=
(point : Point)
(direction : Point)

-- Distance function between two points
noncomputable def distance (p1 p2 : Point) : ℝ :=
Real.sqrt ((p2.x - p1.x)^2 + (p2.y - p1.y)^2)

-- Given conditions
variables (a b : Line) -- lines a and b
variables (A1 A2 : Point) -- positions of point A on line a
variables (B1 B2 : Point) -- positions of point B on line b

-- Hypotheses about uniform motion along the lines
def moves_uniformly (A1 A2 : Point) (a : Line) (B1 B2 : Point) (b : Line) : Prop :=
  ∀ t : ℝ, ∃ (At Bt : Point), 
  At.x = A1.x + t * (A2.x - A1.x) ∧ At.y = A1.y + t * (A2.y - A1.y) ∧
  Bt.x = B1.x + t * (B2.x - B1.x) ∧ Bt.y = B1.y + t * (B2.y - B1.y) ∧
  ∀ s : ℝ, At.x + s * (a.direction.x) = Bt.x + s * (b.direction.x) ∧
           At.y + s * (a.direction.y) = Bt.y + s * (b.direction.y)

-- Problem statement: Prove the existence of points such that AB is minimized
theorem minimize_distance (a b : Line) (A1 A2 B1 B2 : Point) (h : moves_uniformly A1 A2 a B1 B2 b) : 
  ∃ (A B : Point), distance A B = Real.sqrt ((A2.x - B2.x) ^ 2 + (A2.y - B2.y) ^ 2) ∧ distance A B ≤ distance A1 B1 ∧ distance A B ≤ distance A2 B2 :=
sorry

end minimize_distance_l212_212611


namespace car_meeting_distance_l212_212908

theorem car_meeting_distance
  (distance_AB : ℝ)
  (speed_A : ℝ)
  (speed_B : ℝ)
  (midpoint_C : ℝ)
  (meeting_distance_from_C : ℝ) 
  (h1 : distance_AB = 245)
  (h2 : speed_A = 70)
  (h3 : speed_B = 90)
  (h4 : midpoint_C = distance_AB / 2) :
  meeting_distance_from_C = 15.31 := 
sorry

end car_meeting_distance_l212_212908


namespace find_x_in_terms_of_y_l212_212438

theorem find_x_in_terms_of_y 
(h₁ : x ≠ 0) 
(h₂ : x ≠ 3) 
(h₃ : y ≠ 0) 
(h₄ : y ≠ 5) 
(h_eq : 3 / x + 2 / y = 1 / 3) : 
x = 9 * y / (y - 6) :=
by
  sorry

end find_x_in_terms_of_y_l212_212438


namespace min_N_such_that_next_person_sits_next_to_someone_l212_212384

def circular_table_has_80_chairs : Prop := ∃ chairs : ℕ, chairs = 80
def N_people_seated (N : ℕ) : Prop := N > 0
def next_person_sits_next_to_someone (N : ℕ) : Prop :=
  ∀ additional_person_seated : ℕ, additional_person_seated ≤ N → additional_person_seated > 0 
  → ∃ adjacent_person : ℕ, adjacent_person ≤ N ∧ adjacent_person > 0
def smallest_value_for_N (N : ℕ) : Prop :=
  (∀ k : ℕ, k < N → ¬next_person_sits_next_to_someone k)

theorem min_N_such_that_next_person_sits_next_to_someone :
  circular_table_has_80_chairs →
  smallest_value_for_N 20 :=
by
  intro h
  sorry

end min_N_such_that_next_person_sits_next_to_someone_l212_212384


namespace domain_of_g_x_l212_212819

theorem domain_of_g_x :
  ∀ x, (x ≤ 6 ∧ x ≥ -19) ↔ -19 ≤ x ∧ x ≤ 6 :=
by 
  -- Statement only, no proof
  sorry

end domain_of_g_x_l212_212819


namespace find_b_l212_212126

theorem find_b (x : ℝ) (b : ℝ) :
  (3 * x + 9 = 0) → (2 * b * x - 15 = -5) → b = -5 / 3 :=
by
  intros h1 h2
  sorry

end find_b_l212_212126


namespace sum_of_intersection_coordinates_l212_212185

noncomputable def h : ℝ → ℝ := sorry

theorem sum_of_intersection_coordinates : 
  (∃ a b : ℝ, h a = h (a + 2) ∧ h 1 = 3 ∧ h (-1) = 3 ∧ a = -1 ∧ b = 3) → -1 + 3 = 2 :=
by
  intro h_assumptions
  sorry

end sum_of_intersection_coordinates_l212_212185


namespace Petya_can_verify_coins_l212_212256

theorem Petya_can_verify_coins :
  ∃ (c₁ c₂ c₃ c₅ : ℕ), 
  (c₁ = 1 ∧ c₂ = 2 ∧ c₃ = 3 ∧ c₅ = 5) ∧
  (∃ (w : ℕ), w = 9) ∧
  (∃ (cond : ℕ → Prop), 
    cond 1 ∧ cond 2 ∧ cond 3 ∧ cond 5) := sorry

end Petya_can_verify_coins_l212_212256


namespace trajectory_of_midpoint_l212_212404

theorem trajectory_of_midpoint 
  (x y : ℝ)
  (P : ℝ × ℝ)
  (M : ℝ × ℝ)
  (hM : (M.fst - 4)^2 + M.snd^2 = 16)
  (hP : P = (x, y))
  (h_mid : M = (2 * P.1 + 4, 2 * P.2 - 8)) :
  x^2 + (y - 4)^2 = 4 :=
by
  sorry

end trajectory_of_midpoint_l212_212404


namespace find_solutions_l212_212588

theorem find_solutions (n k : ℕ) (hn : n > 0) (hk : k > 0) : 
  n! + n = n^k → (n, k) = (2, 2) ∨ (n, k) = (3, 2) ∨ (n, k) = (5, 3) :=
sorry

end find_solutions_l212_212588


namespace slope_of_line_l212_212804

theorem slope_of_line (m : ℤ) (hm : (3 * m - 6) / (1 + m) = 12) : m = -2 := 
sorry

end slope_of_line_l212_212804


namespace find_f_value_l212_212651

theorem find_f_value (f : ℝ → ℝ) 
  (h : ∀ x : ℝ, x ≠ 0 → f (1 - 2 * x) = (1 - x^2) / x^2) : 
  f (1 / 2) = 15 :=
sorry

end find_f_value_l212_212651


namespace platform_length_proof_l212_212506

-- Given conditions
def train_length : ℝ := 300
def time_to_cross_platform : ℝ := 27
def time_to_cross_pole : ℝ := 18

-- The length of the platform L to be proved
def length_of_platform (L : ℝ) : Prop := 
  (train_length / time_to_cross_pole) = (train_length + L) / time_to_cross_platform

theorem platform_length_proof : length_of_platform 150 :=
by
  sorry

end platform_length_proof_l212_212506


namespace matrix_pow_expression_l212_212969

def A : Matrix (Fin 3) (Fin 3) ℝ := !![3, 4, 2; 0, 2, 3; 0, 0, 1]
def I : Matrix (Fin 3) (Fin 3) ℝ := 1

theorem matrix_pow_expression :
  A^5 - 3 • A^4 = !![0, 4, 2; 0, -1, 3; 0, 0, -2] := by
  sorry

end matrix_pow_expression_l212_212969


namespace nn_gt_n1n1_l212_212259

theorem nn_gt_n1n1 (n : ℕ) (h : n > 1) : n^n > (n + 1)^(n - 1) := 
sorry

end nn_gt_n1n1_l212_212259


namespace intersection_points_parabola_l212_212680

noncomputable def parabola : ℝ → ℝ := λ x => x^2

noncomputable def directrix : ℝ → ℝ := λ x => -1

noncomputable def other_line (m c : ℝ) : ℝ → ℝ := λ x => m * x + c

theorem intersection_points_parabola {m c : ℝ} (h1 : ∃ x1 x2 : ℝ, other_line m c x1 = parabola x1 ∧ other_line m c x2 = parabola x2) :
  (∃ x1 x2 : ℝ, parabola x1 = other_line m c x1 ∧ parabola x2 = other_line m c x2 ∧ x1 ≠ x2) → 
  (∃ x1 x2 : ℝ, parabola x1 = other_line m c x1 ∧ parabola x2 = other_line m c x2 ∧ x1 = x2) := 
by
  sorry

end intersection_points_parabola_l212_212680


namespace cost_of_shoes_l212_212497

-- Define the conditions
def saved : Nat := 30
def earn_per_lawn : Nat := 5
def lawns_per_weekend : Nat := 3
def weekends_needed : Nat := 6

-- Prove the total amount saved is the cost of the shoes
theorem cost_of_shoes : saved + (earn_per_lawn * lawns_per_weekend * weekends_needed) = 120 := by
  sorry

end cost_of_shoes_l212_212497


namespace cost_price_percentage_of_marked_price_l212_212594

theorem cost_price_percentage_of_marked_price
  (MP : ℝ) -- Marked Price
  (CP : ℝ) -- Cost Price
  (discount_percent : ℝ) (gain_percent : ℝ)
  (H1 : CP = (x / 100) * MP) -- Cost Price is x percent of Marked Price
  (H2 : discount_percent = 13) -- Discount percentage
  (H3 : gain_percent = 55.35714285714286) -- Gain percentage
  : x = 56 :=
sorry

end cost_price_percentage_of_marked_price_l212_212594


namespace g_f_eval_l212_212505

def f (x : ℤ) := x^3 - 2
def g (x : ℤ) := 3 * x^2 + x + 2

theorem g_f_eval : g (f 3) = 1902 := by
  sorry

end g_f_eval_l212_212505


namespace find_a2_l212_212454

-- Definitions from conditions
def is_arithmetic_sequence (u : ℕ → ℤ) (d : ℤ) : Prop :=
  ∀ n, u (n + 1) = u n + d

def is_geometric_sequence (a b c : ℤ) : Prop :=
  b * b = a * c

-- Main theorem statement
theorem find_a2
  (u : ℕ → ℤ) (a1 a3 a4 : ℤ)
  (h1 : is_arithmetic_sequence u 3)
  (h2 : is_geometric_sequence a1 a3 a4)
  (h3 : a1 = u 1)
  (h4 : a3 = u 3)
  (h5 : a4 = u 4) :
  u 2 = -9 :=
by  
  sorry

end find_a2_l212_212454


namespace original_rectangle_area_l212_212137

-- Define the original rectangle sides, square side, and perimeters of rectangles adjacent to the square
variables {a b x : ℝ}
variable (h1 : a + x = 10)
variable (h2 : b + x = 8)

-- Define the area calculation
def area (a b : ℝ) := a * b

-- The area of the original rectangle should be 80 cm²
theorem original_rectangle_area : area (10 - x) (8 - x) = 80 := by
  sorry

end original_rectangle_area_l212_212137


namespace sum_of_integers_l212_212821

theorem sum_of_integers (numbers : List ℕ) (h1 : numbers.Nodup) 
(h2 : ∃ a b, (a ≠ b ∧ a * b = 16 ∧ a ∈ numbers ∧ b ∈ numbers)) 
(h3 : ∃ c d, (c ≠ d ∧ c * d = 225 ∧ c ∈ numbers ∧ d ∈ numbers)) :
  numbers.sum = 44 :=
sorry

end sum_of_integers_l212_212821


namespace cyclist_first_part_distance_l212_212024

theorem cyclist_first_part_distance
  (T₁ T₂ T₃ : ℝ)
  (D : ℝ)
  (h1 : D = 9 * T₁)
  (h2 : T₂ = 12 / 10)
  (h3 : T₃ = (D + 12) / 7.5)
  (h4 : T₁ + T₂ + T₃ = 7.2) : D = 18 := by
  sorry

end cyclist_first_part_distance_l212_212024


namespace triangle_groups_count_l212_212926

theorem triangle_groups_count (total_points collinear_groups groups_of_three total_combinations : ℕ)
    (h1 : total_points = 12)
    (h2 : collinear_groups = 16)
    (h3 : groups_of_three = (total_points.choose 3))
    (h4 : total_combinations = groups_of_three - collinear_groups) :
    total_combinations = 204 :=
by
  -- This is where the proof would go
  sorry

end triangle_groups_count_l212_212926


namespace simplify_expression_l212_212157

theorem simplify_expression : ( (144^2 - 12^2) / (120^2 - 18^2) * ((120 - 18) * (120 + 18)) / ((144 - 12) * (144 + 12)) ) = 1 :=
by
  sorry

end simplify_expression_l212_212157


namespace train_cross_signal_in_18_sec_l212_212688

-- Definitions of the given conditions
def train_length := 300 -- meters
def platform_length := 350 -- meters
def time_cross_platform := 39 -- seconds

-- Speed of the train
def train_speed := (train_length + platform_length) / time_cross_platform -- meters/second

-- Time to cross the signal pole
def time_cross_signal_pole := train_length / train_speed -- seconds

theorem train_cross_signal_in_18_sec : time_cross_signal_pole = 18 := by sorry

end train_cross_signal_in_18_sec_l212_212688


namespace inverse_proportional_l212_212521

theorem inverse_proportional (p q : ℝ) (k : ℝ) 
  (h1 : ∀ (p q : ℝ), p * q = k)
  (h2 : p = 25)
  (h3 : q = 6) 
  (h4 : q = 15) : 
  p = 10 := 
by
  sorry

end inverse_proportional_l212_212521


namespace not_all_odd_l212_212055

def is_odd (n : ℕ) : Prop := ∃ k : ℕ, n = 2 * k + 1
def divides (a b c d : ℕ) : Prop := a = b * c + d ∧ 0 ≤ d ∧ d < b

theorem not_all_odd (a b c d : ℕ) 
  (h_div : divides a b c d)
  (h_odd_a : is_odd a)
  (h_odd_b : is_odd b)
  (h_odd_c : is_odd c)
  (h_odd_d : is_odd d) :
  False :=
sorry

end not_all_odd_l212_212055


namespace intersection_complements_l212_212026

open Set

variable (U : Set (ℝ × ℝ))
variable (M : Set (ℝ × ℝ))
variable (N : Set (ℝ × ℝ))

noncomputable def complementU (A : Set (ℝ × ℝ)) : Set (ℝ × ℝ) := U \ A

theorem intersection_complements :
  let U := {p : ℝ × ℝ | True}
  let M := {p : ℝ × ℝ | (∃ (x y : ℝ), p = (x, y) ∧ y + 2 = x - 2 ∧ x ≠ 2)}
  let N := {p : ℝ × ℝ | (∃ (x y : ℝ), p = (x, y) ∧ y ≠ x - 4)}
  ((complementU U M) ∩ (complementU U N)) = {(2, -2)} :=
by
  let U := {(x, y) : ℝ × ℝ | True}
  let M := {(x, y) : ℝ × ℝ | (y + 2) = (x - 2) ∧ x ≠ 2}
  let N := {(x, y) : ℝ × ℝ | y ≠ (x - 4)}
  have complement_M := U \ M
  have complement_N := U \ N
  sorry

end intersection_complements_l212_212026


namespace problem1_problem2_l212_212163

-- The first problem
theorem problem1 (x : ℝ) (h : Real.tan x = 3) :
  (2 * Real.sin (Real.pi - x) + 3 * Real.cos (-x)) /
  (Real.sin (x + Real.pi / 2) - Real.sin (x + Real.pi)) = 9 / 4 :=
by
  sorry

-- The second problem
theorem problem2 (x : ℝ) (h : Real.tan x = 3) :
  2 * Real.sin x ^ 2 - Real.sin (2 * x) + Real.cos x ^ 2 = 13 / 10 :=
by
  sorry

end problem1_problem2_l212_212163


namespace floor_length_l212_212337

theorem floor_length (tile_length tile_width : ℕ) (floor_width max_tiles : ℕ)
  (h_tile : tile_length = 25 ∧ tile_width = 16)
  (h_floor_width : floor_width = 120)
  (h_max_tiles : max_tiles = 54) :
  ∃ floor_length : ℕ, 
    (∃ num_cols num_rows : ℕ, 
      num_cols * tile_width = floor_width ∧ 
      num_cols * num_rows = max_tiles ∧ 
      num_rows * tile_length = floor_length) ∧
    floor_length = 175 := 
by
  sorry

end floor_length_l212_212337


namespace relationship_y1_y2_y3_l212_212859

def on_hyperbola (x y k : ℝ) : Prop := y = k / x

theorem relationship_y1_y2_y3 (y1 y2 y3 k : ℝ) (h1 : on_hyperbola (-5) y1 k) (h2 : on_hyperbola (-1) y2 k) (h3 : on_hyperbola 2 y3 k) (hk : k > 0) :
  y2 < y1 ∧ y1 < y3 :=
sorry

end relationship_y1_y2_y3_l212_212859


namespace tenured_professors_percentage_l212_212838

noncomputable def percentage_tenured (W M T TM : ℝ) := W = 0.69 ∧ (1 - W) = M ∧ (M * 0.52) = TM ∧ (W + T - TM) = 0.90 → T = 0.7512

-- Define the mathematical entities
variables (W M T TM : ℝ)

-- The main statement
theorem tenured_professors_percentage : percentage_tenured W M T TM := by
  sorry

end tenured_professors_percentage_l212_212838


namespace jane_cycling_time_difference_l212_212719

theorem jane_cycling_time_difference :
  (3 * 5 / 6.5 - (5 / 10 + 5 / 5 + 5 / 8)) * 60 = 11 :=
by sorry

end jane_cycling_time_difference_l212_212719


namespace find_m_value_l212_212248

-- Define the quadratic function
def quadratic (x m : ℝ) : ℝ := x^2 - 6 * x + m

-- Define the condition that the quadratic function has a minimum value of 1
def has_minimum_value_of_one (m : ℝ) : Prop := ∃ x : ℝ, quadratic x m = 1

-- The main theorem statement
theorem find_m_value : ∀ m : ℝ, has_minimum_value_of_one m → m = 10 :=
by sorry

end find_m_value_l212_212248


namespace no_solution_implies_a_eq_one_l212_212624

theorem no_solution_implies_a_eq_one (a : ℝ) : 
  ¬(∃ x y : ℝ, a * x + y = 1 ∧ x + y = 2) → a = 1 :=
by
  intro h
  sorry

end no_solution_implies_a_eq_one_l212_212624


namespace sum_of_largest_100_l212_212144

theorem sum_of_largest_100 (a : Fin 123 → ℝ) (h1 : (Finset.univ.sum a) = 3813) 
  (h2 : ∀ i j : Fin 123, i ≤ j → a i ≤ a j) : 
  ∃ s : Finset (Fin 123), s.card = 100 ∧ (s.sum a) ≥ 3100 :=
by
  sorry

end sum_of_largest_100_l212_212144


namespace paula_twice_as_old_as_karl_6_years_later_l212_212758

theorem paula_twice_as_old_as_karl_6_years_later
  (P K : ℕ)
  (h1 : P - 5 = 3 * (K - 5))
  (h2 : P + K = 54) :
  P + 6 = 2 * (K + 6) :=
sorry

end paula_twice_as_old_as_karl_6_years_later_l212_212758


namespace five_points_plane_distance_gt3_five_points_space_not_necessarily_gt3_l212_212012

/-
Problem (a): Given five points on a plane, where the distance between any two points is greater than 2. 
             Prove that there exists a distance between some two of them that is greater than 3.
-/
theorem five_points_plane_distance_gt3 (P : Fin 5 → ℝ × ℝ) 
    (h : ∀ i j : Fin 5, i ≠ j → dist (P i) (P j) > 2) : 
    ∃ i j : Fin 5, i ≠ j ∧ dist (P i) (P j) > 3 :=
sorry

/-
Problem (b): Given five points in space, where the distance between any two points is greater than 2. 
             Prove that it is not necessarily true that there exists a distance between some two of them that is greater than 3.
-/
theorem five_points_space_not_necessarily_gt3 (P : Fin 5 → ℝ × ℝ × ℝ) 
    (h : ∀ i j : Fin 5, i ≠ j → dist (P i) (P j) > 2) : 
    ¬ ∃ i j : Fin 5, i ≠ j ∧ dist (P i) (P j) > 3 :=
sorry

end five_points_plane_distance_gt3_five_points_space_not_necessarily_gt3_l212_212012


namespace least_sum_of_bases_l212_212421

theorem least_sum_of_bases :
  ∃ (c d : ℕ), (5 * c + 7 = 7 * d + 5) ∧ (c > 0) ∧ (d > 0) ∧ (c + d = 14) :=
by
  sorry

end least_sum_of_bases_l212_212421


namespace max_correct_answers_l212_212776

variables {a b c : ℕ} -- Define a, b, and c as natural numbers

theorem max_correct_answers : 
  ∀ a b c : ℕ, (a + b + c = 50) → (5 * a - 2 * c = 150) → a ≤ 35 :=
by
  -- Proof steps can be skipped by adding sorry
  sorry

end max_correct_answers_l212_212776


namespace min_dist_on_circle_l212_212992

theorem min_dist_on_circle :
  let P (θ : ℝ) := (2 * Real.cos θ * Real.cos θ, 2 * Real.cos θ * Real.sin θ)
  let M := (0, 2)
  ∃ θ_min : ℝ, 
    (∀ θ : ℝ, 
      let dist (P : ℝ × ℝ) (M : ℝ × ℝ) := Real.sqrt ((P.1 - M.1)^2 + (P.2 - M.2)^2)
      dist (P θ) M ≥ dist (P θ_min) M) ∧ 
    dist (P θ_min) M = Real.sqrt 5 - 1 := sorry

end min_dist_on_circle_l212_212992


namespace minimize_square_sum_l212_212503

theorem minimize_square_sum (x1 x2 x3 : ℝ) (h1 : 0 < x1) (h2 : 0 < x2) (h3 : 0 < x3) 
  (h4 : x1 + 3 * x2 + 5 * x3 = 100) : 
  x1^2 + x2^2 + x3^2 ≥ 2000 / 7 :=
sorry

end minimize_square_sum_l212_212503


namespace min_xsq_ysq_zsq_l212_212338

noncomputable def min_value_x_sq_y_sq_z_sq (x y z : ℝ) (h : x^3 + y^3 + z^3 - 3 * x * y * z = 8) : ℝ :=
  (x^2 + y^2 + z^2)

theorem min_xsq_ysq_zsq (x y z : ℝ) (h : x^3 + y^3 + z^3 - 3 * x * y * z = 8) : 
  min_value_x_sq_y_sq_z_sq x y z h = 40 / 7 :=
  sorry

end min_xsq_ysq_zsq_l212_212338


namespace non_positive_sequence_l212_212851

theorem non_positive_sequence
  (N : ℕ)
  (a : ℕ → ℝ)
  (h₀ : a 0 = 0)
  (hN : a N = 0)
  (h_rec : ∀ i, 1 ≤ i ∧ i ≤ N - 1 → a (i + 1) - 2 * a i + a (i - 1) = a i ^ 2) :
  ∀ i, 1 ≤ i ∧ i ≤ N - 1 → a i ≤ 0 := sorry

end non_positive_sequence_l212_212851


namespace student_solved_correctly_l212_212313

-- Problem conditions as definitions
def sums_attempted : Nat := 96

def sums_correct (x : Nat) : Prop :=
  let sums_wrong := 3 * x
  x + sums_wrong = sums_attempted

-- Lean statement to prove
theorem student_solved_correctly (x : Nat) (h : sums_correct x) : x = 24 :=
  sorry

end student_solved_correctly_l212_212313


namespace least_prime_value_l212_212405

/-- Let q be a set of 12 distinct prime numbers. If the sum of the integers in q is odd,
the product of all the integers in q is divisible by a perfect square, and the number x is a member of q,
then the least value that x can be is 2. -/
theorem least_prime_value (q : Finset ℕ) (hq_distinct : q.card = 12) (hq_prime : ∀ p ∈ q, Nat.Prime p) 
    (hq_odd_sum : q.sum id % 2 = 1) (hq_perfect_square_div : ∃ k, q.prod id % (k * k) = 0) (x : ℕ)
    (hx : x ∈ q) : x = 2 :=
sorry

end least_prime_value_l212_212405


namespace inequality_necessary_not_sufficient_l212_212136

theorem inequality_necessary_not_sufficient (m : ℝ) : 
  (-3 < m ∧ m < 5) → (5 - m > 0 ∧ m + 3 > 0 ∧ 5 - m ≠ m + 3) :=
by
  intro h
  sorry

end inequality_necessary_not_sufficient_l212_212136


namespace calc_fraction_product_l212_212538

theorem calc_fraction_product : 
  (7 / 4) * (8 / 14) * (14 / 8) * (16 / 40) * (35 / 20) * (18 / 45) * (49 / 28) * (32 / 64) = 49 / 200 := 
by sorry

end calc_fraction_product_l212_212538


namespace papers_delivered_to_sunday_only_houses_l212_212864

-- Define the number of houses in the route and the days
def houses_in_route : ℕ := 100
def days_monday_to_saturday : ℕ := 6

-- Define the number of customers that do not get the paper on Sunday
def non_customers_sunday : ℕ := 10
def total_papers_per_week : ℕ := 720

-- Define the required number of papers delivered on Sunday to houses that only get the paper on Sunday
def papers_only_on_sunday : ℕ :=
  total_papers_per_week - (houses_in_route * days_monday_to_saturday) - (houses_in_route - non_customers_sunday)

theorem papers_delivered_to_sunday_only_houses : papers_only_on_sunday = 30 :=
by
  sorry

end papers_delivered_to_sunday_only_houses_l212_212864


namespace new_equation_incorrect_l212_212893

-- Definition of a function to change each digit of a number by +1 or -1 randomly.
noncomputable def modify_digit (num : ℕ) : ℕ := sorry

-- Proposition stating the original problem's condition and conclusion.
theorem new_equation_incorrect (a b : ℕ) (c := a + b) (a' b' c' : ℕ)
    (h1 : a' = modify_digit a)
    (h2 : b' = modify_digit b)
    (h3 : c' = modify_digit c) :
    a' + b' ≠ c' :=
sorry

end new_equation_incorrect_l212_212893


namespace sum_of_first_9_primes_l212_212165

theorem sum_of_first_9_primes : 
  (2 + 3 + 5 + 7 + 11 + 13 + 17 + 19 + 23) = 100 := 
by
  sorry

end sum_of_first_9_primes_l212_212165


namespace measure_of_angle_D_l212_212773

-- Definitions of angles in pentagon ABCDE
variables (A B C D E : ℝ)

-- Conditions
def condition1 := D = A + 30
def condition2 := E = A + 50
def condition3 := B = C
def condition4 := A = B - 45
def condition5 := A + B + C + D + E = 540

-- Theorem to prove
theorem measure_of_angle_D (h1 : condition1 A D)
                           (h2 : condition2 A E)
                           (h3 : condition3 B C)
                           (h4 : condition4 A B)
                           (h5 : condition5 A B C D E) :
  D = 104 :=
sorry

end measure_of_angle_D_l212_212773


namespace fill_tub_together_time_l212_212709

theorem fill_tub_together_time :
  let rate1 := 1 / 4
  let rate2 := 1 / 4
  let rate3 := 1 / 12
  let combined_rate := rate1 + rate2 + rate3
  combined_rate ≠ 0 → (1 / combined_rate = 12 / 7) :=
by
  let rate1 := 1 / 4
  let rate2 := 1 / 4
  let rate3 := 1 / 12
  let combined_rate := rate1 + rate2 + rate3
  sorry

end fill_tub_together_time_l212_212709


namespace working_light_bulbs_count_l212_212852

def lamps := 60
def bulbs_per_lamp := 7

def fraction_with_2_burnt := 1 / 3
def fraction_with_1_burnt := 1 / 4
def fraction_with_3_burnt := 1 / 5

def lamps_with_2_burnt := fraction_with_2_burnt * lamps
def lamps_with_1_burnt := fraction_with_1_burnt * lamps
def lamps_with_3_burnt := fraction_with_3_burnt * lamps
def lamps_with_all_working := lamps - (lamps_with_2_burnt + lamps_with_1_burnt + lamps_with_3_burnt)

def working_bulbs_from_2_burnt := lamps_with_2_burnt * (bulbs_per_lamp - 2)
def working_bulbs_from_1_burnt := lamps_with_1_burnt * (bulbs_per_lamp - 1)
def working_bulbs_from_3_burnt := lamps_with_3_burnt * (bulbs_per_lamp - 3)
def working_bulbs_from_all_working := lamps_with_all_working * bulbs_per_lamp

def total_working_bulbs := working_bulbs_from_2_burnt + working_bulbs_from_1_burnt + working_bulbs_from_3_burnt + working_bulbs_from_all_working

theorem working_light_bulbs_count : total_working_bulbs = 329 := by
  sorry

end working_light_bulbs_count_l212_212852


namespace hyperbola_standard_eq_l212_212551

theorem hyperbola_standard_eq (a c : ℝ) (h1 : a = 5) (h2 : c = 7) :
  (∃ b, b^2 = c^2 - a^2 ∧ (1 = (x^2 / a^2 - y^2 / b^2) ∨ 1 = (y^2 / a^2 - x^2 / b^2))) := by
  sorry

end hyperbola_standard_eq_l212_212551


namespace maximize_profit_l212_212324

-- Definitions from the conditions
def cost_price : ℝ := 16
def initial_selling_price : ℝ := 20
def initial_sales_volume : ℝ := 80
def price_decrease_per_step : ℝ := 0.5
def sales_increase_per_step : ℝ := 20

def functional_relationship (x : ℝ) : ℝ := -40 * x + 880

-- The main theorem we need to prove
theorem maximize_profit :
  (∀ x, 16 ≤ x → x ≤ 20 → functional_relationship x = -40 * x + 880) ∧
  (∃ x, 16 ≤ x ∧ x ≤ 20 ∧ (∀ y, 16 ≤ y → y ≤ 20 → 
    ((-40 * x + 880) * (x - cost_price) ≥ (-40 * y + 880) * (y - cost_price)) ∧
    (-40 * x + 880) * (x - cost_price) = 360 ∧ x = 19)) :=
by
  sorry

end maximize_profit_l212_212324


namespace find_intersection_l212_212722

noncomputable def A : Set ℝ := { x | -4 < x ∧ x < 3 }
noncomputable def B : Set ℝ := { x | x ≤ 2 }

theorem find_intersection : A ∩ B = { x | -4 < x ∧ x ≤ 2 } := sorry

end find_intersection_l212_212722


namespace range_of_a_part1_range_of_a_part2_l212_212592

def set_A (x : ℝ) : Prop := -1 < x ∧ x < 6

def set_B (x : ℝ) (a : ℝ) : Prop := (x ≥ 1 + a) ∨ (x ≤ 1 - a)

def condition_1 (a : ℝ) : Prop :=
  (∀ x, set_A x → ¬ set_B x a) → (a ≥ 5)

def condition_2 (a : ℝ) : Prop :=
  (∀ x, (x ≥ 6 ∨ x ≤ -1) → set_B x a) ∧ (∃ x, set_B x a ∧ ¬ (x ≥ 6 ∨ x ≤ -1)) → (0 < a ∧ a ≤ 2)

theorem range_of_a_part1 (a : ℝ) : condition_1 a :=
  sorry

theorem range_of_a_part2 (a : ℝ) : condition_2 a :=
  sorry

end range_of_a_part1_range_of_a_part2_l212_212592


namespace xiaoxian_mistake_xiaoxuan_difference_l212_212480

-- Define the initial expressions and conditions
def original_expr := (-9) * 3 - 5
def xiaoxian_expr (x : Int) := (-9) * 3 - x
def xiaoxuan_expr := (-9) / 3 - 5

-- Given conditions
variable (result_xiaoxian : Int)
variable (result_original : Int)

-- Proof statement
theorem xiaoxian_mistake (hx : xiaoxian_expr 2 = -29) : 
  xiaoxian_expr 5 = result_xiaoxian := sorry

theorem xiaoxuan_difference : 
  abs (xiaoxuan_expr - original_expr) = 24 := sorry

end xiaoxian_mistake_xiaoxuan_difference_l212_212480


namespace range_of_a_l212_212116

theorem range_of_a (a : ℝ) (h : ∀ x : ℝ, x^2 - a*x + 2*a > 0) : 0 < a ∧ a < 8 :=
by
  sorry

end range_of_a_l212_212116


namespace bouncy_balls_total_l212_212442

theorem bouncy_balls_total :
  let red_packs := 6
  let red_per_pack := 12
  let yellow_packs := 10
  let yellow_per_pack := 8
  let green_packs := 4
  let green_per_pack := 15
  let blue_packs := 3
  let blue_per_pack := 20
  let red_balls := red_packs * red_per_pack
  let yellow_balls := yellow_packs * yellow_per_pack
  let green_balls := green_packs * green_per_pack
  let blue_balls := blue_packs * blue_per_pack
  red_balls + yellow_balls + green_balls + blue_balls = 272 := 
by
  sorry

end bouncy_balls_total_l212_212442


namespace exponent_neg_power_l212_212226

theorem exponent_neg_power (a : ℝ) : -(a^3)^4 = -a^(3 * 4) := 
by
  sorry

end exponent_neg_power_l212_212226


namespace correct_equation_l212_212747

theorem correct_equation : ∃a : ℝ, (-3 * a) ^ 2 = 9 * a ^ 2 :=
by
  use 1
  sorry

end correct_equation_l212_212747


namespace time_to_cross_platform_l212_212692

/-- Definitions of the conditions in the problem. -/
def train_length : ℕ := 1500
def platform_length : ℕ := 1800
def time_to_cross_tree : ℕ := 100
def train_speed : ℕ := train_length / time_to_cross_tree
def total_distance : ℕ := train_length + platform_length

/-- Proof statement: The time for the train to pass the platform. -/
theorem time_to_cross_platform : (total_distance / train_speed) = 220 := by
  sorry

end time_to_cross_platform_l212_212692


namespace total_wait_time_l212_212274

def customs_wait : ℕ := 20
def quarantine_days : ℕ := 14
def hours_per_day : ℕ := 24

theorem total_wait_time :
  customs_wait + quarantine_days * hours_per_day = 356 := 
by
  sorry

end total_wait_time_l212_212274


namespace probability_compensation_l212_212211

-- Define the probabilities of each vehicle getting into an accident
def p1 : ℚ := 1 / 20
def p2 : ℚ := 1 / 21

-- Define the probability of the complementary event
def comp_event : ℚ := (1 - p1) * (1 - p2)

-- Define the overall probability that at least one vehicle gets into an accident
def comp_unit : ℚ := 1 - comp_event

-- The theorem to be proved: the probability that the unit will receive compensation from this insurance within a year is 2 / 21
theorem probability_compensation : comp_unit = 2 / 21 :=
by
  -- giving the proof is not required
  sorry

end probability_compensation_l212_212211


namespace max_sundays_in_51_days_l212_212422

theorem max_sundays_in_51_days (days_in_week: ℕ) (total_days: ℕ) 
  (start_on_first: Bool) (first_day_sunday: Prop) 
  (is_sunday: ℕ → Bool) :
  days_in_week = 7 ∧ total_days = 51 ∧ start_on_first = tt ∧ first_day_sunday → 
  (∃ n, ∀ i < total_days, is_sunday i → n ≤ 8) ∧ 
  (∀ j, j ≤ total_days → is_sunday j → j ≤ 8) := by
  sorry

end max_sundays_in_51_days_l212_212422


namespace jenny_reading_time_l212_212975

theorem jenny_reading_time 
  (days : ℕ)
  (words_first_book : ℕ)
  (words_second_book : ℕ)
  (words_third_book : ℕ)
  (reading_speed : ℕ) : 
  days = 10 →
  words_first_book = 200 →
  words_second_book = 400 →
  words_third_book = 300 →
  reading_speed = 100 →
  (words_first_book + words_second_book + words_third_book) / reading_speed / days * 60 = 54 :=
by
  intros hdays hwords1 hwords2 hwords3 hspeed
  rw [hdays, hwords1, hwords2, hwords3, hspeed]
  norm_num
  sorry

end jenny_reading_time_l212_212975


namespace compute_value_l212_212331

theorem compute_value : (142 + 29 + 26 + 14) * 2 = 422 := 
by 
  sorry

end compute_value_l212_212331


namespace simplify_expression_l212_212790

theorem simplify_expression :
  ( ( (11 / 4) / (11 / 10 + 10 / 3) ) / ( 5 / 2 - ( 4 / 3 ) ) ) /
  ( ( 5 / 7 ) - ( ( (13 / 6 + 9 / 2) * 3 / 8 ) / (11 / 4 - 3 / 2) ) )
  = - (35 / 9) :=
by
  sorry

end simplify_expression_l212_212790


namespace solve_fraction_eq_l212_212889

theorem solve_fraction_eq (x : ℝ) (h₁ : x ≠ 2) (h₂ : x ≠ 3) 
    (h₃ : 3 / (x - 2) = 6 / (x - 3)) : x = 1 :=
by 
  sorry

end solve_fraction_eq_l212_212889


namespace typing_time_l212_212919

-- Definitions based on the problem conditions
def initial_typing_speed : ℕ := 212
def speed_decrease : ℕ := 40
def words_in_document : ℕ := 3440

-- Definition for Barbara's new typing speed
def new_typing_speed : ℕ := initial_typing_speed - speed_decrease

-- Lean proof statement: Proving the time to finish typing is 20 minutes
theorem typing_time :
  (words_in_document / new_typing_speed) = 20 :=
by sorry

end typing_time_l212_212919


namespace find_percentage_reduction_l212_212715

-- Given the conditions of the problem.
def original_price : ℝ := 7500
def current_price: ℝ := 4800
def percentage_reduction (x : ℝ) : Prop := (original_price * (1 - x)^2 = current_price)

-- The statement we need to prove:
theorem find_percentage_reduction (x : ℝ) (h : percentage_reduction x) : x = 0.2 :=
by
  sorry

end find_percentage_reduction_l212_212715


namespace range_of_ϕ_l212_212866

noncomputable def f (x : ℝ) (ϕ : ℝ) : ℝ := 2 * Real.sin (2 * x + ϕ) + 1

theorem range_of_ϕ (ϕ : ℝ) (h1 : abs ϕ ≤ Real.pi / 2) 
    (h2 : ∀ (x : ℝ), -Real.pi / 12 < x ∧ x < Real.pi / 3 → f x ϕ > 1) :
  Real.pi / 6 ≤ ϕ ∧ ϕ ≤ Real.pi / 3 :=
sorry

end range_of_ϕ_l212_212866


namespace average_age_of_choir_l212_212134

theorem average_age_of_choir 
  (num_females : ℕ) (avg_age_females : ℝ)
  (num_males : ℕ) (avg_age_males : ℝ)
  (total_people : ℕ) (total_people_eq : total_people = num_females + num_males) :
  num_females = 12 → avg_age_females = 28 → num_males = 18 → avg_age_males = 38 → total_people = 30 →
  (num_females * avg_age_females + num_males * avg_age_males) / total_people = 34 := by
  intros
  sorry

end average_age_of_choir_l212_212134


namespace star_shell_arrangements_l212_212342

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

-- Conditions
def outward_points : ℕ := 6
def inward_points : ℕ := 6
def total_points : ℕ := outward_points + inward_points
def unique_shells : ℕ := 12

-- The problem statement translated into Lean 4:
theorem star_shell_arrangements : (factorial unique_shells / 12 = 39916800) :=
by
  sorry

end star_shell_arrangements_l212_212342


namespace least_integer_square_eq_12_more_than_three_times_l212_212275

theorem least_integer_square_eq_12_more_than_three_times (x : ℤ) (h : x^2 = 3 * x + 12) : x = -3 :=
sorry

end least_integer_square_eq_12_more_than_three_times_l212_212275


namespace largest_perfect_square_factor_9240_l212_212052

theorem largest_perfect_square_factor_9240 :
  ∃ n : ℕ, n * n = 36 ∧ ∃ m : ℕ, m ∣ 9240 ∧ m = n * n :=
by
  -- We will construct the proof here using the prime factorization
  sorry

end largest_perfect_square_factor_9240_l212_212052


namespace line_equations_l212_212403

theorem line_equations : 
  ∀ (x y : ℝ), (∃ a b c : ℝ, 2 * x + y - 12 = 0 ∨ 2 * x - 5 * y = 0 ∧ (x, y) = (5, 2) ∧ b = 2 * a) :=
by
  sorry

end line_equations_l212_212403


namespace pie_not_crust_percentage_l212_212099

theorem pie_not_crust_percentage (total_weight crust_weight : ℝ) 
  (h1 : total_weight = 200) (h2 : crust_weight = 50) : 
  (total_weight - crust_weight) / total_weight * 100 = 75 :=
by
  sorry

end pie_not_crust_percentage_l212_212099


namespace area_of_rectangular_region_l212_212981

-- Mathematical Conditions
variables (a b c d : ℝ)
variables (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d)

-- Lean 4 Statement of the proof problem
theorem area_of_rectangular_region :
  (a + b) * (2 * d + c) = 2 * a * d + a * c + 2 * b * d + b * c :=
by sorry

end area_of_rectangular_region_l212_212981


namespace robbie_weight_l212_212875

theorem robbie_weight (R P : ℝ) 
  (h1 : P = 4.5 * R - 235)
  (h2 : P = R + 115) :
  R = 100 := 
by 
  sorry

end robbie_weight_l212_212875


namespace white_tulips_multiple_of_seven_l212_212297

/-- Let R be the number of red tulips, which is given as 91. 
    We also know that the greatest number of identical bouquets that can be made without 
    leaving any flowers out is 7.
    Prove that the number of white tulips W is a multiple of 7. -/
theorem white_tulips_multiple_of_seven (R : ℕ) (g : ℕ) (W : ℕ) (hR : R = 91) (hg : g = 7) :
  ∃ w : ℕ, W = 7 * w :=
by
  sorry

end white_tulips_multiple_of_seven_l212_212297


namespace probability_of_continuous_stripe_loop_l212_212138

-- Definitions corresponding to identified conditions:
def cube_faces : ℕ := 6

def diagonal_orientations_per_face : ℕ := 2

def total_stripe_combinations (faces : ℕ) (orientations : ℕ) : ℕ :=
  orientations ^ faces

def satisfying_stripe_combinations : ℕ := 2

-- Proof statement:
theorem probability_of_continuous_stripe_loop :
  (satisfying_stripe_combinations : ℚ) / (total_stripe_combinations cube_faces diagonal_orientations_per_face : ℚ) = 1 / 32 :=
by
  -- Proof goes here
  sorry

end probability_of_continuous_stripe_loop_l212_212138


namespace store_profit_l212_212543

theorem store_profit (m n : ℝ) (hmn : m > n) : 
  let selling_price := (m + n) / 2
  let profit_a := 40 * (selling_price - m)
  let profit_b := 60 * (selling_price - n)
  let total_profit := profit_a + profit_b
  total_profit > 0 :=
by sorry

end store_profit_l212_212543


namespace line_does_not_pass_second_quadrant_l212_212042

theorem line_does_not_pass_second_quadrant 
  (A B C x y : ℝ) 
  (h1 : A * C < 0) 
  (h2 : B * C > 0) 
  (h3 : A * x + B * y + C = 0) :
  ¬ (x < 0 ∧ y > 0) := 
sorry

end line_does_not_pass_second_quadrant_l212_212042


namespace find_C_l212_212078

theorem find_C (A B C : ℕ) (h1 : A + B + C = 500) (h2 : A + C = 200) (h3 : B + C = 350) : C = 50 :=
by
  sorry

end find_C_l212_212078


namespace operation_B_is_correct_l212_212581

theorem operation_B_is_correct (a b x : ℝ) : 
  2 * (a^2) * b * 4 * a * (b^3) = 8 * (a^3) * (b^4) :=
by
  sorry

-- Conditions for incorrect operations
lemma operation_A_is_incorrect (x : ℝ) : 
  x^8 / x^2 ≠ x^4 :=
by
  sorry

lemma operation_C_is_incorrect (x : ℝ) : 
  (-x^5)^4 ≠ -x^20 :=
by
  sorry

lemma operation_D_is_incorrect (a b : ℝ) : 
  (a + b)^2 ≠ a^2 + b^2 :=
by
  sorry

end operation_B_is_correct_l212_212581


namespace expenditure_on_concrete_blocks_l212_212425

def blocks_per_section : ℕ := 30
def cost_per_block : ℕ := 2
def number_of_sections : ℕ := 8

theorem expenditure_on_concrete_blocks : 
  (number_of_sections * blocks_per_section) * cost_per_block = 480 := 
by 
  sorry

end expenditure_on_concrete_blocks_l212_212425


namespace temperature_increase_l212_212766

variable (T_morning T_afternoon : ℝ)

theorem temperature_increase : 
  (T_morning = -3) → (T_afternoon = 5) → (T_afternoon - T_morning = 8) :=
by
intros h1 h2
rw [h1, h2]
sorry

end temperature_increase_l212_212766


namespace diff_one_tenth_and_one_tenth_percent_of_6000_l212_212129

def one_tenth_of_6000 := 6000 / 10
def one_tenth_percent_of_6000 := (1 / 1000) * 6000

theorem diff_one_tenth_and_one_tenth_percent_of_6000 : 
  (one_tenth_of_6000 - one_tenth_percent_of_6000) = 594 :=
by
  sorry

end diff_one_tenth_and_one_tenth_percent_of_6000_l212_212129


namespace circle1_correct_circle2_correct_l212_212197

noncomputable def circle1_eq (x y : ℝ) : ℝ :=
  x^2 + y^2 + 4*x - 6*y - 12

noncomputable def circle2_eq (x y : ℝ) : ℝ :=
  36*x^2 + 36*y^2 - 24*x + 72*y + 31

theorem circle1_correct (x y : ℝ) :
  ((x + 2)^2 + (y - 3)^2 = 25) ↔ (circle1_eq x y = 0) :=
sorry

theorem circle2_correct (x y : ℝ) :
  (36 * ((x - 1/3)^2 + (y + 1)^2) = 9) ↔ (circle2_eq x y = 0) :=
sorry

end circle1_correct_circle2_correct_l212_212197


namespace distribute_pencils_l212_212816

theorem distribute_pencils (number_of_pencils : ℕ) (number_of_people : ℕ)
  (h_pencils : number_of_pencils = 2) (h_people : number_of_people = 5) :
  number_of_distributions = 15 := by
  sorry

end distribute_pencils_l212_212816


namespace no_sum_of_19_l212_212772

theorem no_sum_of_19 (a b c d : ℕ) (ha : 1 ≤ a ∧ a ≤ 6) (hb : 1 ≤ b ∧ b ≤ 6) (hc : 1 ≤ c ∧ c ≤ 6) (hd : 1 ≤ d ∧ d ≤ 6)
  (hprod : a * b * c * d = 180) : a + b + c + d ≠ 19 :=
sorry

end no_sum_of_19_l212_212772


namespace average_of_three_numbers_l212_212171

theorem average_of_three_numbers
  (a b c : ℕ)
  (h1 : 2 * a + b + c = 130)
  (h2 : a + 2 * b + c = 138)
  (h3 : a + b + 2 * c = 152) :
  (a + b + c) / 3 = 35 :=
by
  sorry

end average_of_three_numbers_l212_212171


namespace num_non_congruent_triangles_with_perimeter_12_l212_212246

noncomputable def count_non_congruent_triangles_with_perimeter_12 : ℕ :=
  sorry -- This is where the actual proof or computation would go.

theorem num_non_congruent_triangles_with_perimeter_12 :
  count_non_congruent_triangles_with_perimeter_12 = 3 :=
  sorry -- This is the theorem stating the result we want to prove.

end num_non_congruent_triangles_with_perimeter_12_l212_212246


namespace find_a_perpendicular_lines_l212_212796

theorem find_a_perpendicular_lines 
  (a : ℤ)
  (l1 : ∀ x y : ℤ, a * x + 4 * y + 7 = 0)
  (l2 : ∀ x y : ℤ, 2 * x - 3 * y - 1 = 0) : 
  (∃ a : ℤ, a = 6) :=
by sorry

end find_a_perpendicular_lines_l212_212796


namespace betty_min_sugar_flour_oats_l212_212689

theorem betty_min_sugar_flour_oats :
  ∃ (s f o : ℕ), f ≥ 4 + 2 * s ∧ f ≤ 3 * s ∧ o = f + s ∧ s = 4 :=
by
  sorry

end betty_min_sugar_flour_oats_l212_212689


namespace probability_of_exactly_one_solves_l212_212095

variable (p1 p2 : ℝ)

theorem probability_of_exactly_one_solves (h1 : 0 ≤ p1) (h2 : p1 ≤ 1) (h3 : 0 ≤ p2) (h4 : p2 ≤ 1) :
  (p1 * (1 - p2) + p2 * (1 - p1)) = (p1 * (1 - p2) + p2 * (1 - p1)) :=
by
  sorry

end probability_of_exactly_one_solves_l212_212095


namespace hallie_reads_121_pages_on_fifth_day_l212_212844

-- Definitions for the given conditions.
def book_length : ℕ := 480
def pages_day_one : ℕ := 63
def pages_day_two : ℕ := 95 -- Rounded from 94.5
def pages_day_three : ℕ := 115
def pages_day_four : ℕ := 86 -- Rounded from 86.25

-- Total pages read from day one to day four
def pages_read_first_four_days : ℕ :=
  pages_day_one + pages_day_two + pages_day_three + pages_day_four

-- Conclusion: the number of pages read on the fifth day.
def pages_day_five : ℕ := book_length - pages_read_first_four_days

-- Proof statement: Hallie reads 121 pages on the fifth day.
theorem hallie_reads_121_pages_on_fifth_day :
  pages_day_five = 121 :=
by
  -- Proof omitted
  sorry

end hallie_reads_121_pages_on_fifth_day_l212_212844


namespace initial_group_machines_l212_212727

-- Define the number of bags produced by n machines in one minute and 150 machines in one minute
def bags_produced (machines : ℕ) (bags_per_minute : ℕ) : Prop :=
  machines * bags_per_minute = 45

def bags_produced_150 (bags_produced_in_8_mins : ℕ) : Prop :=
  150 * (bags_produced_in_8_mins / 8) = 450

-- Given the conditions, prove that the number of machines in the initial group is 15
theorem initial_group_machines (n : ℕ) (bags_produced_in_8_mins : ℕ) :
  bags_produced n 45 → bags_produced_150 bags_produced_in_8_mins → n = 15 :=
by
  intro h1 h2
  -- use the conditions to derive the result
  sorry

end initial_group_machines_l212_212727


namespace soccer_league_equation_l212_212224

noncomputable def equation_represents_soccer_league (x : ℕ) : Prop :=
  ∀ x : ℕ, (x * (x - 1)) / 2 = 50

theorem soccer_league_equation (x : ℕ) (h : equation_represents_soccer_league x) :
  (x * (x - 1)) / 2 = 50 :=
  by sorry

end soccer_league_equation_l212_212224


namespace algebraic_identity_l212_212363

theorem algebraic_identity (a b : ℝ) : a^2 - 2 * a * b + b^2 = (a - b)^2 :=
by
  sorry

end algebraic_identity_l212_212363


namespace length_of_train_is_correct_l212_212232

noncomputable def train_length (speed_km_hr : ℝ) (time_sec : ℝ) : ℝ :=
  let speed_m_s := (speed_km_hr * 1000) / 3600
  speed_m_s * time_sec

theorem length_of_train_is_correct (speed_km_hr : ℝ) (time_sec : ℝ) (expected_length : ℝ) :
  speed_km_hr = 60 → time_sec = 21 → expected_length = 350.07 →
  train_length speed_km_hr time_sec = expected_length :=
by
  intros h1 h2 h3
  simp [h1, h2, train_length]
  sorry

end length_of_train_is_correct_l212_212232


namespace sin_cos_acute_angle_lt_one_l212_212272

theorem sin_cos_acute_angle_lt_one (α β : ℝ) (a b c : ℝ) (hα : 0 < α ∧ α < π / 2)
  (hβ : 0 < β ∧ β < π / 2) (h_triangle : a^2 + b^2 = c^2) (h_nonzero_c : c ≠ 0) :
  (a / c < 1) ∧ (b / c < 1) :=
by 
  sorry

end sin_cos_acute_angle_lt_one_l212_212272


namespace james_total_fish_catch_l212_212583

-- Definitions based on conditions
def weight_trout : ℕ := 200
def weight_salmon : ℕ := weight_trout + (60 * weight_trout / 100)
def weight_tuna : ℕ := 2 * weight_trout
def weight_bass : ℕ := 3 * weight_salmon
def weight_catfish : ℚ := weight_tuna / 3

-- Total weight of the fish James caught
def total_weight_fish : ℚ := 
  weight_trout + weight_salmon + weight_tuna + weight_bass + weight_catfish 

-- The theorem statement
theorem james_total_fish_catch : total_weight_fish = 2013.33 := by
  sorry

end james_total_fish_catch_l212_212583


namespace second_caterer_cheaper_l212_212329

theorem second_caterer_cheaper (x : ℕ) (h : x > 33) : 200 + 12 * x < 100 + 15 * x := 
by
  sorry

end second_caterer_cheaper_l212_212329


namespace sin_eq_cos_510_l212_212575

theorem sin_eq_cos_510 (n : ℤ) (h1 : -180 ≤ n ∧ n ≤ 180) (h2 : Real.sin (n * Real.pi / 180) = Real.cos (510 * Real.pi / 180)) :
  n = -60 :=
sorry

end sin_eq_cos_510_l212_212575


namespace hall_length_width_difference_l212_212285

theorem hall_length_width_difference (L W : ℝ) 
  (h1 : W = 1 / 2 * L)
  (h2 : L * W = 450) :
  L - W = 15 :=
sorry

end hall_length_width_difference_l212_212285


namespace polynomial_coeff_properties_l212_212142

theorem polynomial_coeff_properties :
  (∃ a0 a1 a2 a3 a4 a5 a6 a7 : ℤ,
  (∀ x : ℤ, (1 - 2 * x)^7 = a0 + a1 * x + a2 * x^2 + a3 * x^3 + a4 * x^4 + a5 * x^5 + a6 * x^6 + a7 * x^7) ∧
  a0 = 1 ∧
  (a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7 = -1) ∧
  (|a0| + |a1| + |a2| + |a3| + |a4| + |a5| + |a6| + |a7| = 3^7)) :=
sorry

end polynomial_coeff_properties_l212_212142


namespace jeans_discount_rates_l212_212920

theorem jeans_discount_rates
    (M F P : ℝ) 
    (regular_price_moose jeans_regular_price_fox jeans_regular_price_pony : ℝ)
    (moose_count fox_count pony_count : ℕ)
    (total_discount : ℝ) :
    regular_price_moose = 20 →
    regular_price_fox = 15 →
    regular_price_pony = 18 →
    moose_count = 2 →
    fox_count = 3 →
    pony_count = 2 →
    total_discount = 12.48 →
    (M + F + P = 0.32) →
    (F + P = 0.20) →
    (moose_count * M * regular_price_moose + fox_count * F * regular_price_fox + pony_count * P * regular_price_pony = total_discount) →
    M = 0.12 ∧ F = 0.0533 ∧ P = 0.1467 :=
by
  intros
  sorry -- The proof is not required

end jeans_discount_rates_l212_212920


namespace arrange_letters_of_unique_word_l212_212960

-- Define the problem parameters
def unique_word := ["M₁", "I₁", "S₁", "S₂", "I₂", "P₁", "P₂", "I₃"]
def word_length := unique_word.length
def arrangement_count := Nat.factorial word_length

-- Theorem statement corresponding to the problem
theorem arrange_letters_of_unique_word :
  arrangement_count = 40320 :=
by
  sorry

end arrange_letters_of_unique_word_l212_212960


namespace roofing_cost_per_foot_l212_212511

theorem roofing_cost_per_foot:
  ∀ (total_feet needed_feet free_feet : ℕ) (total_cost : ℕ),
  needed_feet = 300 →
  free_feet = 250 →
  total_cost = 400 →
  needed_feet - free_feet = 50 →
  total_cost / (needed_feet - free_feet) = 8 :=
by sorry

end roofing_cost_per_foot_l212_212511


namespace correct_diagram_l212_212223

-- Definitions based on the conditions
def word : String := "KANGAROO"
def diagrams : List (String × Bool) :=
  [("Diagram A", False), ("Diagram B", False), ("Diagram C", False),
   ("Diagram D", False), ("Diagram E", True)]

-- Statement to prove that Diagram E correctly shows "KANGAROO"
theorem correct_diagram :
  ∃ d, (d.1 = "Diagram E") ∧ d.2 = True ∧ d ∈ diagrams :=
by
-- skipping the proof for now
sorry

end correct_diagram_l212_212223


namespace scale_length_l212_212294

theorem scale_length (num_parts : ℕ) (part_length : ℕ) (total_length : ℕ) 
  (h1 : num_parts = 5) (h2 : part_length = 16) : total_length = 80 :=
by
  sorry

end scale_length_l212_212294


namespace value_of_4_ampersand_neg3_l212_212788

-- Define the operation '&'
def ampersand (x y : Int) : Int :=
  x * (y + 2) + x * y

-- State the theorem
theorem value_of_4_ampersand_neg3 : ampersand 4 (-3) = -16 :=
by
  sorry

end value_of_4_ampersand_neg3_l212_212788


namespace ellipse_sum_l212_212792

theorem ellipse_sum (F1 F2 : ℝ × ℝ) (h k a b : ℝ) 
  (hf1 : F1 = (0, 0)) (hf2 : F2 = (6, 0))
  (h_eqn : ∀ P : ℝ × ℝ, dist P F1 + dist P F2 = 10) :
  h + k + a + b = 12 :=
by
  sorry

end ellipse_sum_l212_212792


namespace compare_sqrt_sums_l212_212368

   noncomputable def a : ℝ := Real.sqrt 8 + Real.sqrt 5
   noncomputable def b : ℝ := Real.sqrt 7 + Real.sqrt 6

   theorem compare_sqrt_sums : a < b :=
   by
     sorry
   
end compare_sqrt_sums_l212_212368


namespace skittles_students_division_l212_212524

theorem skittles_students_division (n : ℕ) (h1 : 27 % 3 = 0) (h2 : 27 / 3 = n) : n = 9 := by
  sorry

end skittles_students_division_l212_212524


namespace subset_M_N_l212_212273

def M : Set ℝ := {-1, 1}
def N : Set ℝ := { x | (1 / x < 2) }

theorem subset_M_N : M ⊆ N :=
by
  sorry -- Proof omitted as per the guidelines

end subset_M_N_l212_212273


namespace symmetry_condition_l212_212801

theorem symmetry_condition 
  (p q r s : ℝ) (hp : p ≠ 0) (hq : q ≠ 0) (hr : r ≠ 0) (hs : s ≠ 0) : 
  (∀ a b : ℝ, b = 2 * a → (∃ y, y = (p * (b/2) + 2*q) / (r * (b/2) + 2*s) ∧  b = 2*(y/2) )) → 
  p + r = 0 :=
by
  sorry

end symmetry_condition_l212_212801


namespace interval_of_monotonic_increase_l212_212379

noncomputable def powerFunction (k n x : ℝ) : ℝ := k * x ^ n

variable {k n : ℝ}

theorem interval_of_monotonic_increase
    (h : ∃ k n : ℝ, powerFunction k n 4 = 2) :
    (∀ x y : ℝ, 0 < x ∧ x < y → powerFunction k n x < powerFunction k n y) ∨
    (∀ x y : ℝ, 0 ≤ x ∧ x < y → powerFunction k n x ≤ powerFunction k n y) := sorry

end interval_of_monotonic_increase_l212_212379


namespace triangle_AB_eq_3_halves_CK_l212_212289

/-- Mathematically equivalent problem:
In an acute triangle ABC, rectangle ACGH is constructed with AC as one side, and CG : AC = 2:1.
A square BCEF is constructed with BC as one side. The height CD from A to B intersects GE at point K.
Prove that AB = 3/2 * CK. -/
theorem triangle_AB_eq_3_halves_CK
  (A B C H G E K : Type)
  (triangle_ABC_acute : ∀(A B C : Type), True) 
  (rectangle_ACGH : ∀(A C G H : Type), True) 
  (square_BCEF : ∀(B C E F : Type), True)
  (H_C_G_collinear : ∀(H C G : Type), True)
  (HCG_ratio : ∀ (AC CG : ℝ), CG / AC = 2 / 1)
  (BC_side : ∀ (BC : ℝ), BC = 1)
  (height_CD_intersection : ∀ (A B C D E G : Type), True)
  (intersection_point_K : ∀ (C D G E K : Type), True) :
  ∃ (AB CK : ℝ), AB = 3 / 2 * CK :=
by sorry

end triangle_AB_eq_3_halves_CK_l212_212289


namespace similar_triangle_shortest_side_l212_212238

theorem similar_triangle_shortest_side
  (a₁ : ℕ) (c₁ : ℕ) (c₂ : ℕ)
  (h₁ : a₁ = 15) (h₂ : c₁ = 17) (h₃ : c₂ = 68)
  (right_triangle_1 : a₁^2 + b₁^2 = c₁^2)
  (similar_triangles : ∃ k : ℕ, c₂ = k * c₁) :
  shortest_side = 32 := 
sorry

end similar_triangle_shortest_side_l212_212238


namespace max_edges_convex_polyhedron_l212_212507

theorem max_edges_convex_polyhedron (n : ℕ) (c l e : ℕ) (h1 : c = n) (h2 : c + l = e + 2) (h3 : 2 * e ≥ 3 * l) : e ≤ 3 * n - 6 := 
sorry

end max_edges_convex_polyhedron_l212_212507


namespace mac_total_loss_l212_212151

-- Definitions based on conditions in part a)
def value_dime : ℝ := 0.10
def value_nickel : ℝ := 0.05
def value_quarter : ℝ := 0.25
def dimes_per_quarter : ℕ := 3
def nickels_per_quarter : ℕ := 7
def quarters_traded_dimes : ℕ := 20
def quarters_traded_nickels : ℕ := 20

-- Lean statement for the proof problem
theorem mac_total_loss : (dimes_per_quarter * value_dime * quarters_traded_dimes 
                          + nickels_per_quarter * value_nickel * quarters_traded_nickels
                          - 40 * value_quarter) = 3.00 := 
sorry

end mac_total_loss_l212_212151


namespace chestnuts_distribution_l212_212872

theorem chestnuts_distribution:
  ∃ (chestnuts_Alya chestnuts_Valya chestnuts_Galya : ℕ),
    chestnuts_Alya + chestnuts_Valya + chestnuts_Galya = 70 ∧
    4 * chestnuts_Valya = 3 * chestnuts_Alya ∧
    6 * chestnuts_Galya = 7 * chestnuts_Alya ∧
    chestnuts_Alya = 24 ∧
    chestnuts_Valya = 18 ∧
    chestnuts_Galya = 28 :=
by {
  sorry
}

end chestnuts_distribution_l212_212872


namespace total_capacity_is_correct_l212_212168

-- Define small and large jars capacities
def small_jar_capacity : ℕ := 3
def large_jar_capacity : ℕ := 5

-- Define the total number of jars and the number of small jars
def total_jars : ℕ := 100
def small_jars : ℕ := 62

-- Define the number of large jars based on the total jars and small jars
def large_jars : ℕ := total_jars - small_jars

-- Calculate capacities
def small_jars_total_capacity : ℕ := small_jars * small_jar_capacity
def large_jars_total_capacity : ℕ := large_jars * large_jar_capacity

-- Define the total capacity
def total_capacity : ℕ := small_jars_total_capacity + large_jars_total_capacity

-- Prove that the total capacity is 376 liters
theorem total_capacity_is_correct : total_capacity = 376 := by
  sorry

end total_capacity_is_correct_l212_212168


namespace determine_x_2y_l212_212482

theorem determine_x_2y (x y : ℝ) (h1 : 2 * x + y = 7) (h2 : (x + y) / 3 = 5 / 3) : x + 2 * y = 8 :=
sorry

end determine_x_2y_l212_212482


namespace sin_double_alpha_trig_expression_l212_212326

theorem sin_double_alpha (α : ℝ) (h1 : Real.sin α = -1 / 3) (h2 : π < α ∧ α < 3 * π / 2) :
  Real.sin (2 * α) = 4 * Real.sqrt 2 / 9 :=
sorry

theorem trig_expression (α : ℝ) (h1 : Real.sin α = -1 / 3) (h2 : π < α ∧ α < 3 * π / 2) :
  (Real.sin (α - 2 * π) * Real.cos (2 * π - α)) / (Real.sin (α + π / 2) ^ 2) = Real.sqrt 2 / 4 :=
sorry

end sin_double_alpha_trig_expression_l212_212326


namespace min_area_circle_equation_l212_212498

theorem min_area_circle_equation (x y : ℝ) (hx : 0 < x) (hy : 0 < y) 
  (h : 3 / (2 + x) + 3 / (2 + y) = 1) : (x - 4)^2 + (y - 4)^2 = 256 :=
sorry

end min_area_circle_equation_l212_212498


namespace popularity_order_is_correct_l212_212860

noncomputable def fraction_liking_dodgeball := (13 : ℚ) / 40
noncomputable def fraction_liking_karaoke := (9 : ℚ) / 30
noncomputable def fraction_liking_magicshow := (17 : ℚ) / 60
noncomputable def fraction_liking_quizbowl := (23 : ℚ) / 120

theorem popularity_order_is_correct :
  (fraction_liking_dodgeball ≥ fraction_liking_karaoke) ∧
  (fraction_liking_karaoke ≥ fraction_liking_magicshow) ∧
  (fraction_liking_magicshow ≥ fraction_liking_quizbowl) ∧
  (fraction_liking_dodgeball ≠ fraction_liking_karaoke) ∧
  (fraction_liking_karaoke ≠ fraction_liking_magicshow) ∧
  (fraction_liking_magicshow ≠ fraction_liking_quizbowl) := by
  sorry

end popularity_order_is_correct_l212_212860


namespace ratio_of_points_to_away_home_game_l212_212685

-- Definitions
def first_away_game_points (A : ℕ) : ℕ := A
def second_away_game_points (A : ℕ) : ℕ := A + 18
def third_away_game_points (A : ℕ) : ℕ := A + 20
def last_home_game_points : ℕ := 62
def next_game_points : ℕ := 55
def total_points (A : ℕ) : ℕ := A + (A + 18) + (A + 20) + 62 + 55

-- Given that the total points should be four times the points of the last home game
def target_points : ℕ := 4 * 62

-- The main theorem to prove
theorem ratio_of_points_to_away_home_game : ∀ A : ℕ,
  total_points A = target_points → 62 = 2 * A :=
by
  sorry

end ratio_of_points_to_away_home_game_l212_212685


namespace asymptotes_of_hyperbola_l212_212525

theorem asymptotes_of_hyperbola (a : ℝ) :
  (∃ x y : ℝ, y^2 = 12 * x ∧ (x = 3) ∧ (y = 0)) →
  (a^2 = 9) →
  (∀ b c : ℝ, (b, c) ∈ ({(a, b) | (b = a/3 ∨ b = -a/3)})) :=
by
  intro h_focus_coincides vertex_condition
  sorry

end asymptotes_of_hyperbola_l212_212525


namespace solve_for_x_l212_212737

theorem solve_for_x (x : ℝ) (h : (x / 6) / 3 = 9 / (x / 3)) : x = 9 * Real.sqrt 6 ∨ x = - (9 * Real.sqrt 6) :=
by
  sorry

end solve_for_x_l212_212737


namespace hyperbola_standard_equation_l212_212797

theorem hyperbola_standard_equation (a b : ℝ) (x y : ℝ)
  (H₁ : 2 * a = 2) -- length of the real axis is 2
  (H₂ : y = 2 * x) -- one of its asymptote equations
  : y^2 - 4 * x^2 = 1 :=
sorry

end hyperbola_standard_equation_l212_212797


namespace train_length_l212_212360

theorem train_length (t : ℝ) (v : ℝ) (h1 : t = 13) (h2 : v = 58.15384615384615) : abs (v * t - 756) < 1 :=
by
  sorry

end train_length_l212_212360


namespace problem_1_problem_2_problem_3_l212_212774

theorem problem_1 (avg_daily_production : ℕ) (deviation_wed : ℤ) :
  avg_daily_production = 3000 →
  deviation_wed = -15 →
  avg_daily_production + deviation_wed = 2985 :=
by intros; sorry

theorem problem_2 (avg_daily_production : ℕ) (deviation_sat : ℤ) (deviation_fri : ℤ) :
  avg_daily_production = 3000 →
  deviation_sat = 68 →
  deviation_fri = -20 →
  (avg_daily_production + deviation_sat) - (avg_daily_production + deviation_fri) = 88 :=
by intros; sorry

theorem problem_3 (planned_weekly_production : ℕ) (deviations : List ℤ) :
  planned_weekly_production = 21000 →
  deviations = [35, -12, -15, 30, -20, 68, -9] →
  planned_weekly_production + deviations.sum = 21077 :=
by intros; sorry

end problem_1_problem_2_problem_3_l212_212774


namespace college_application_ways_correct_l212_212423

def college_application_ways : ℕ :=
  -- Scenario 1: Student does not apply to either of the two conflicting colleges
  (Nat.choose 4 3) +
  -- Scenario 2: Student applies to one of the two conflicting colleges
  ((Nat.choose 2 1) * (Nat.choose 4 2))

theorem college_application_ways_correct : college_application_ways = 16 := by
  -- We can skip the proof
  sorry

end college_application_ways_correct_l212_212423


namespace reduced_price_of_oil_l212_212660

/-- 
Given:
1. The original price per kg of oil is P.
2. The reduced price per kg of oil is 0.65P.
3. Rs. 800 can buy 5 kgs more oil at the reduced price than at the original price.
4. The equation 5P - 5 * 0.65P = 800 holds true.

Prove that the reduced price per kg of oil is Rs. 297.14.
-/
theorem reduced_price_of_oil (P : ℝ) (h1 : 5 * P - 5 * 0.65 * P = 800) : 
        0.65 * P = 297.14 := 
    sorry

end reduced_price_of_oil_l212_212660


namespace ivan_speed_ratio_l212_212053

/-- 
A group of tourists started a hike from a campsite. Fifteen minutes later, Ivan returned to the campsite for a flashlight 
and started catching up with the group at a faster constant speed. He reached them 2.5 hours after initially leaving. 
Prove Ivan's speed is 1.2 times the group's speed.
-/
theorem ivan_speed_ratio (d_g d_i : ℝ) (t_g t_i : ℝ) (v_g v_i : ℝ)
    (h1 : t_g = 2.25)       -- Group's travel time (2.25 hours after initial 15 minutes)
    (h2 : t_i = 2.5)        -- Ivan's total travel time
    (h3 : d_g = t_g * v_g)  -- Distance covered by group
    (h4 : d_i = 3 * (v_g * (15 / 60))) -- Ivan's distance covered
    (h5 : d_g = d_i)        -- Ivan eventually catches up with the group
  : v_i / v_g = 1.2 := sorry

end ivan_speed_ratio_l212_212053


namespace impossibility_of_quadratic_conditions_l212_212061

open Real

theorem impossibility_of_quadratic_conditions :
  ∀ (a b c t : ℝ), a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ a ≠ t ∧ b ≠ t ∧ c ≠ t →
  (b * t) ^ 2 - 4 * a * c > 0 →
  c ^ 2 - 4 * b * a > 0 →
  (a * t) ^ 2 - 4 * b * c > 0 →
  false :=
by sorry

end impossibility_of_quadratic_conditions_l212_212061


namespace two_x_plus_two_y_value_l212_212932

theorem two_x_plus_two_y_value (x y : ℝ) (h1 : x^2 - y^2 = 8) (h2 : x - y = 6) : 2 * x + 2 * y = 8 / 3 := 
by sorry

end two_x_plus_two_y_value_l212_212932


namespace total_cupcakes_needed_l212_212382

-- Definitions based on conditions
def cupcakes_per_event : ℝ := 96.0
def number_of_events : ℝ := 8.0

-- Theorem based on the question and the correct answer
theorem total_cupcakes_needed : (cupcakes_per_event * number_of_events) = 768.0 :=
by 
  sorry

end total_cupcakes_needed_l212_212382


namespace sqrt_subtraction_result_l212_212870

theorem sqrt_subtraction_result : 
  (Real.sqrt (49 + 36) - Real.sqrt (36 - 0)) = 4 :=
by
  sorry

end sqrt_subtraction_result_l212_212870


namespace parabola_standard_eq_l212_212734

theorem parabola_standard_eq (p p' : ℝ) (h₁ : p > 0) (h₂ : p' > 0) :
  (∀ (x y : ℝ), (x^2 = 2 * p * y ∨ y^2 = -2 * p' * x) → 
  (x = -2 ∧ y = 4 → (x^2 = y ∨ y^2 = -8 * x))) :=
by
  sorry

end parabola_standard_eq_l212_212734


namespace lucy_packs_of_cake_l212_212481

theorem lucy_packs_of_cake (total_groceries cookies : ℕ) (h1 : total_groceries = 27) (h2 : cookies = 23) :
  total_groceries - cookies = 4 :=
by
  -- In Lean, we would provide the actual proof here, but we'll use sorry to skip the proof as instructed
  sorry

end lucy_packs_of_cake_l212_212481


namespace arc_length_solution_l212_212799

variable (r : ℝ) (α : ℝ)

theorem arc_length_solution (h1 : r = 8) (h2 : α = 5 * Real.pi / 3) : 
    r * α = 40 * Real.pi / 3 := 
by 
    sorry

end arc_length_solution_l212_212799


namespace mary_max_earnings_l212_212744

def max_hours : ℕ := 40
def regular_rate : ℝ := 8
def first_hours : ℕ := 20
def overtime_rate : ℝ := regular_rate + 0.25 * regular_rate

def earnings : ℝ := 
  (first_hours * regular_rate) +
  ((max_hours - first_hours) * overtime_rate)

theorem mary_max_earnings : earnings = 360 := by
  sorry

end mary_max_earnings_l212_212744


namespace percentage_of_rotten_oranges_l212_212573

theorem percentage_of_rotten_oranges
  (total_oranges : ℕ)
  (total_bananas : ℕ)
  (percentage_good_condition : ℕ)
  (rotted_percentage_bananas : ℕ)
  (total_fruits : ℕ)
  (good_condition_fruits : ℕ)
  (rotted_fruits : ℕ)
  (rotted_bananas : ℕ)
  (rotted_oranges : ℕ)
  (percentage_rotten_oranges : ℕ)
  (h1 : total_oranges = 600)
  (h2 : total_bananas = 400)
  (h3 : percentage_good_condition = 89)
  (h4 : rotted_percentage_bananas = 5)
  (h5 : total_fruits = total_oranges + total_bananas)
  (h6 : good_condition_fruits = percentage_good_condition * total_fruits / 100)
  (h7 : rotted_fruits = total_fruits - good_condition_fruits)
  (h8 : rotted_bananas = rotted_percentage_bananas * total_bananas / 100)
  (h9 : rotted_oranges = rotted_fruits - rotted_bananas)
  (h10 : percentage_rotten_oranges = rotted_oranges * 100 / total_oranges) : 
  percentage_rotten_oranges = 15 := 
by
  sorry

end percentage_of_rotten_oranges_l212_212573


namespace complex_expression_l212_212492

theorem complex_expression (x y : ℂ) 
  (h : (x^3 + y^3) / (x^3 - y^3) + (x^3 - y^3) / (x^3 + y^3) = 1) :
  (x^9 + y^9) / (x^9 - y^9) + (x^9 - y^9) / (x^9 + y^9) = 3 / 2 :=
by 
  sorry

end complex_expression_l212_212492


namespace prob_rain_all_days_l212_212387

/--
The probability of rain on Friday, Saturday, and Sunday is given by 
0.40, 0.60, and 0.35 respectively.
We want to prove that the combined probability of rain on all three days,
assuming independence, is 8.4%.
-/
theorem prob_rain_all_days :
  let p_friday := 0.40
  let p_saturday := 0.60
  let p_sunday := 0.35
  p_friday * p_saturday * p_sunday = 0.084 :=
by
  sorry

end prob_rain_all_days_l212_212387


namespace inequality_example_l212_212523

variable {a b c : ℝ} -- Declare a, b, c as real numbers

theorem inequality_example
  (ha : 0 < a)  -- Condition: a is positive
  (hb : 0 < b)  -- Condition: b is positive
  (hc : 0 < c) :  -- Condition: c is positive
  (ab * (a + b) + ac * (a + c) + bc * (b + c)) / (abc) ≥ 6 := 
sorry  -- Proof is skipped

end inequality_example_l212_212523


namespace carson_gold_stars_yesterday_l212_212608

def goldStarsEarnedYesterday (total: ℕ) (earnedToday: ℕ) : ℕ :=
  total - earnedToday

theorem carson_gold_stars_yesterday :
  goldStarsEarnedYesterday 15 9 = 6 :=
by 
  sorry

end carson_gold_stars_yesterday_l212_212608


namespace probability_adjacent_vertices_dodecagon_l212_212900

noncomputable def prob_adjacent_vertices_dodecagon : ℚ :=
  let total_vertices := 12
  let favorable_outcomes := 2  -- adjacent vertices per chosen vertex
  let total_outcomes := total_vertices - 1  -- choosing any other vertex
  favorable_outcomes / total_outcomes

theorem probability_adjacent_vertices_dodecagon :
  prob_adjacent_vertices_dodecagon = 2 / 11 := by
  sorry

end probability_adjacent_vertices_dodecagon_l212_212900


namespace winston_cents_left_l212_212485

-- Definitions based on the conditions in the problem
def quarters := 14
def cents_per_quarter := 25
def half_dollar_in_cents := 50

-- Formulation of the problem statement in Lean
theorem winston_cents_left : (quarters * cents_per_quarter) - half_dollar_in_cents = 300 :=
by sorry

end winston_cents_left_l212_212485


namespace find_number_ge_40_l212_212107

theorem find_number_ge_40 (x : ℝ) : 0.90 * x > 0.80 * 30 + 12 → x > 40 :=
by sorry

end find_number_ge_40_l212_212107


namespace smallest_solution_l212_212983

theorem smallest_solution (x : ℝ) (h₁ : x ≠ 3) (h₂ : x ≠ 4) (h₃ : x ≠ 5) :
  (1 / (x - 3) + 1 / (x - 5) = 4 / (x - 4)) → x = 4 - Real.sqrt 2 :=
sorry

end smallest_solution_l212_212983


namespace lcm_18_27_l212_212266

theorem lcm_18_27 : Nat.lcm 18 27 = 54 :=
by {
  sorry
}

end lcm_18_27_l212_212266


namespace water_evaporation_weight_l212_212808

noncomputable def initial_weight : ℝ := 200
noncomputable def initial_salt_concentration : ℝ := 0.05
noncomputable def final_salt_concentration : ℝ := 0.08

theorem water_evaporation_weight (W_final : ℝ) (evaporation_weight : ℝ) 
  (h1 : W_final = 10 / final_salt_concentration) 
  (h2 : evaporation_weight = initial_weight - W_final) : 
  evaporation_weight = 75 :=
by
  sorry

end water_evaporation_weight_l212_212808


namespace solve_system_l212_212220

theorem solve_system (s t : ℚ) (h1 : 7 * s + 6 * t = 156) (h2 : s = t / 2 + 3) : s = 192 / 19 :=
sorry

end solve_system_l212_212220


namespace usb_drive_available_space_l212_212698

theorem usb_drive_available_space (C P : ℝ) (hC : C = 16) (hP : P = 50) : 
  (1 - P / 100) * C = 8 :=
by
  sorry

end usb_drive_available_space_l212_212698


namespace calculate_volume_and_diagonal_calculate_volume_and_surface_rotation_calculate_radius_given_volume_l212_212115

noncomputable def volume_of_parallelepiped (R : ℝ) : ℝ := R^3 * Real.sqrt 6

noncomputable def diagonal_A_C_prime (R: ℝ) : ℝ := R * Real.sqrt 6

noncomputable def volume_of_rotation (R: ℝ) : ℝ := R^3 * Real.sqrt 12

theorem calculate_volume_and_diagonal (R : ℝ) : 
  volume_of_parallelepiped R = R^3 * Real.sqrt 6 ∧ 
  diagonal_A_C_prime R = R * Real.sqrt 6 :=
by sorry

theorem calculate_volume_and_surface_rotation (R : ℝ) :
  volume_of_rotation R = R^3 * Real.sqrt 12 :=
by sorry

theorem calculate_radius_given_volume (V : ℝ) (h : V = 0.034786) : 
  ∃ R : ℝ, V = volume_of_parallelepiped R :=
by sorry

end calculate_volume_and_diagonal_calculate_volume_and_surface_rotation_calculate_radius_given_volume_l212_212115


namespace increasing_condition_sufficient_not_necessary_l212_212155

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x^3 + a * x

theorem increasing_condition_sufficient_not_necessary (a : ℝ) :
  (∀ x : ℝ, x > 0 → (3 * x^2 + a) ≥ 0) → (a ≥ 0) ∧ ¬ (a > 0 ↔ (∀ x : ℝ, x > 0 → (3 * x^2 + a) ≥ 0)) :=
by
  sorry

end increasing_condition_sufficient_not_necessary_l212_212155


namespace overall_loss_percentage_l212_212295

theorem overall_loss_percentage
  (cost_price : ℝ)
  (discount : ℝ)
  (sales_tax : ℝ)
  (depreciation : ℝ)
  (final_selling_price : ℝ) :
  cost_price = 1900 →
  discount = 0.15 →
  sales_tax = 0.12 →
  depreciation = 0.05 →
  final_selling_price = 1330 →
  ((cost_price - (discount * cost_price)) * (1 + sales_tax) * (1 - depreciation) - final_selling_price) / cost_price * 100 = 20.44 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end overall_loss_percentage_l212_212295


namespace find_b12_l212_212571

noncomputable def seq (b : ℕ → ℤ) : Prop :=
  b 1 = 2 ∧ 
  ∀ m n : ℕ, m > 0 → n > 0 → b (m + n) = b m + b n + (m * n * n)

theorem find_b12 (b : ℕ → ℤ) (h : seq b) : b 12 = 98 := 
by
  sorry

end find_b12_l212_212571


namespace airplane_rows_l212_212555

theorem airplane_rows (r : ℕ) (h1 : ∀ (seats_per_row total_rows : ℕ), seats_per_row = 8 → total_rows = r →
  ∀ occupied_seats : ℕ, occupied_seats = (3 * seats_per_row) / 4 →
  ∀ unoccupied_seats : ℕ, unoccupied_seats = seats_per_row * total_rows - occupied_seats * total_rows →
  unoccupied_seats = 24): 
  r = 12 :=
by
  sorry

end airplane_rows_l212_212555


namespace analyze_a_b_m_n_l212_212317

theorem analyze_a_b_m_n (a b m n : ℕ) (ha : 1 < a) (hb : 1 < b) (hm : 1 < m) (hn : 1 < n)
  (h1 : Prime (a^n - 1))
  (h2 : Prime (b^m + 1)) :
  n = 2 ∧ ∃ k : ℕ, m = 2^k :=
by
  sorry

end analyze_a_b_m_n_l212_212317


namespace find_x_l212_212092

theorem find_x 
  (a : ℝ × ℝ) 
  (b : ℝ × ℝ) 
  (dot_product : ℝ)
  (ha : a = (1, 2)) 
  (hb : b = (x, 3)) 
  (hdot : a.1 * b.1 + a.2 * b.2 = dot_product) 
  (hdot_val : dot_product = 4) : 
  x = -2 :=
by 
  sorry

end find_x_l212_212092


namespace complex_multiplication_l212_212585

variable (i : ℂ)
axiom i_square : i^2 = -1

theorem complex_multiplication : i * (1 + i) = -1 + i :=
by
  sorry

end complex_multiplication_l212_212585


namespace number_exceeds_its_fraction_by_35_l212_212603

theorem number_exceeds_its_fraction_by_35 (x : ℝ) (h : x = (3 / 8) * x + 35) : x = 56 :=
by
  sorry

end number_exceeds_its_fraction_by_35_l212_212603


namespace Malou_average_is_correct_l212_212504

def quiz1_score : ℕ := 91
def quiz2_score : ℕ := 90
def quiz3_score : ℕ := 92
def total_score : ℕ := quiz1_score + quiz2_score + quiz3_score
def number_of_quizzes : ℕ := 3

def Malous_average_score : ℕ := total_score / number_of_quizzes

theorem Malou_average_is_correct : Malous_average_score = 91 := by
  sorry

end Malou_average_is_correct_l212_212504


namespace find_tricycles_l212_212579

noncomputable def number_of_tricycles (w b t : ℕ) : ℕ := t

theorem find_tricycles : ∃ (w b t : ℕ), 
  (w + b + t = 10) ∧ 
  (2 * b + 3 * t = 25) ∧ 
  (number_of_tricycles w b t = 5) :=
  by 
    sorry

end find_tricycles_l212_212579


namespace cubed_inequality_l212_212453

variable {a b : ℝ}

theorem cubed_inequality (h : a > b) : a^3 > b^3 :=
sorry

end cubed_inequality_l212_212453


namespace sin_add_double_alpha_l212_212515

open Real

theorem sin_add_double_alpha (alpha : ℝ) (h : sin (π / 6 - alpha) = 3 / 5) :
  sin (π / 6 + 2 * alpha) = 7 / 25 :=
by
  sorry

end sin_add_double_alpha_l212_212515


namespace fraction_addition_l212_212472

variable (d : ℝ)

theorem fraction_addition (d : ℝ) : (5 + 2 * d) / 8 + 3 = (29 + 2 * d) / 8 := 
sorry

end fraction_addition_l212_212472


namespace line_b_parallel_or_in_plane_l212_212793

def Line : Type := sorry    -- Placeholder for the type of line
def Plane : Type := sorry   -- Placeholder for the type of plane

def is_parallel (a b : Line) : Prop := sorry             -- Predicate for parallel lines
def is_parallel_to_plane (l : Line) (p : Plane) : Prop := sorry   -- Predicate for a line being parallel to a plane
def lies_in_plane (l : Line) (p : Plane) : Prop := sorry          -- Predicate for a line lying in a plane

theorem line_b_parallel_or_in_plane (a b : Line) (α : Plane) 
  (h1 : is_parallel a b) 
  (h2 : is_parallel_to_plane a α) : 
  is_parallel_to_plane b α ∨ lies_in_plane b α :=
sorry

end line_b_parallel_or_in_plane_l212_212793


namespace largest_quantity_l212_212527

theorem largest_quantity (x y z w : ℤ) (h : x + 5 = y - 3 ∧ y - 3 = z + 2 ∧ z + 2 = w - 4) : w > y ∧ w > z ∧ w > x :=
by
  sorry

end largest_quantity_l212_212527


namespace tom_read_chapters_l212_212186

theorem tom_read_chapters (chapters pages: ℕ) (h1: pages = 8 * chapters) (h2: pages = 24):
  chapters = 3 :=
by
  sorry

end tom_read_chapters_l212_212186


namespace timeTakenByBobIs30_l212_212565

-- Define the conditions
def timeTakenByAlice : ℕ := 40
def fractionOfTimeBobTakes : ℚ := 3 / 4

-- Define the statement to be proven
theorem timeTakenByBobIs30 : (fractionOfTimeBobTakes * timeTakenByAlice : ℚ) = 30 := 
by
  sorry

end timeTakenByBobIs30_l212_212565


namespace adjacent_angles_l212_212934

variable (θ : ℝ)

theorem adjacent_angles (h : θ + 3 * θ = 180) : θ = 45 ∧ 3 * θ = 135 :=
by 
  -- This is the place where the proof would go
  -- Here we only declare the statement, not the proof
  sorry

end adjacent_angles_l212_212934


namespace compare_real_numbers_l212_212944

theorem compare_real_numbers (a b c d : ℝ) (h1 : a = -1) (h2 : b = 0) (h3 : c = 1) (h4 : d = 2) :
  d > a ∧ d > b ∧ d > c :=
by
  sorry

end compare_real_numbers_l212_212944


namespace no_square_has_units_digit_seven_l212_212489

theorem no_square_has_units_digit_seven :
  ¬ ∃ n : ℕ, n ≤ 9 ∧ (n^2 % 10) = 7 := by
  sorry

end no_square_has_units_digit_seven_l212_212489


namespace sarah_more_than_cecily_l212_212590

theorem sarah_more_than_cecily (t : ℕ) (ht : t = 144) :
  let s := (1 / 3 : ℚ) * t
  let a := (3 / 8 : ℚ) * t
  let c := t - (s + a)
  s - c = 6 := by
  sorry

end sarah_more_than_cecily_l212_212590


namespace doughnuts_in_shop_l212_212629

def ratio_of_doughnuts_to_muffins : Nat := 5

def number_of_muffins_in_shop : Nat := 10

def number_of_doughnuts (D M : Nat) : Prop :=
  D = ratio_of_doughnuts_to_muffins * M

theorem doughnuts_in_shop :
  number_of_doughnuts D number_of_muffins_in_shop → D = 50 :=
by
  sorry

end doughnuts_in_shop_l212_212629


namespace probability_one_hits_correct_l212_212519

-- Define the probabilities for A hitting and B hitting
noncomputable def P_A : ℝ := 0.4
noncomputable def P_B : ℝ := 0.5

-- Calculate the required probability
noncomputable def probability_one_hits : ℝ :=
  P_A * (1 - P_B) + (1 - P_A) * P_B

-- Statement of the theorem
theorem probability_one_hits_correct :
  probability_one_hits = 0.5 := by 
  sorry

end probability_one_hits_correct_l212_212519


namespace crayons_total_l212_212907

theorem crayons_total (blue_crayons : ℕ) (red_crayons : ℕ) 
  (H1 : red_crayons = 4 * blue_crayons) (H2 : blue_crayons = 3) : 
  blue_crayons + red_crayons = 15 := 
by
  sorry

end crayons_total_l212_212907


namespace positive_difference_l212_212966

/-- Pauline deposits 10,000 dollars into an account with 4% compound interest annually. -/
def Pauline_initial_deposit : ℝ := 10000
def Pauline_interest_rate : ℝ := 0.04
def Pauline_years : ℕ := 12

/-- Quinn deposits 10,000 dollars into an account with 6% simple interest annually. -/
def Quinn_initial_deposit : ℝ := 10000
def Quinn_interest_rate : ℝ := 0.06
def Quinn_years : ℕ := 12

/-- Pauline's balance after 12 years -/
def Pauline_balance : ℝ := Pauline_initial_deposit * (1 + Pauline_interest_rate) ^ Pauline_years

/-- Quinn's balance after 12 years -/
def Quinn_balance : ℝ := Quinn_initial_deposit * (1 + Quinn_interest_rate * Quinn_years)

/-- The positive difference between Pauline's and Quinn's balances after 12 years is $1189 -/
theorem positive_difference :
  |Quinn_balance - Pauline_balance| = 1189 := 
sorry

end positive_difference_l212_212966


namespace parallelogram_smaller_angle_proof_l212_212574

noncomputable def smaller_angle (x : ℝ) : Prop :=
  let larger_angle := x + 120
  let angle_sum := x + larger_angle + x + larger_angle = 360
  angle_sum

theorem parallelogram_smaller_angle_proof (x : ℝ) (h1 : smaller_angle x) : x = 30 := by
  sorry

end parallelogram_smaller_angle_proof_l212_212574


namespace isosceles_right_triangle_solution_l212_212478

theorem isosceles_right_triangle_solution (a b : ℝ) (area : ℝ) 
  (h1 : a = b) (h2 : XY = a * Real.sqrt 2) (h3 : area = (1/2) * a * b) (h4 : area = 36) : 
  XY = 12 :=
by
  sorry

end isosceles_right_triangle_solution_l212_212478


namespace sin_alpha_l212_212894

theorem sin_alpha (α : ℝ) (hα : 0 < α ∧ α < π) (hcos : Real.cos (π + α) = 3 / 5) :
  Real.sin α = 4 / 5 :=
sorry

end sin_alpha_l212_212894


namespace fourth_intersection_point_of_curve_and_circle_l212_212712

theorem fourth_intersection_point_of_curve_and_circle (h k R : ℝ)
  (h1 : (3 - h)^2 + (2 / 3 - k)^2 = R^2)
  (h2 : (-4 - h)^2 + (-1 / 2 - k)^2 = R^2)
  (h3 : (1 / 2 - h)^2 + (4 - k)^2 = R^2) :
  ∃ (x y : ℝ), xy = 2 ∧ (x, y) ≠ (3, 2 / 3) ∧ (x, y) ≠ (-4, -1 / 2) ∧ (x, y) ≠ (1 / 2, 4) ∧ 
    (x - h)^2 + (y - k)^2 = R^2 ∧ (x, y) = (2 / 3, 3) := 
sorry

end fourth_intersection_point_of_curve_and_circle_l212_212712


namespace find_k_range_of_m_l212_212072

-- Given conditions and function definition
def f (x k : ℝ) : ℝ := x^2 + (2*k-3)*x + k^2 - 7

-- Prove that k = 3 when the zeros of f(x) are -1 and -2
theorem find_k (k : ℝ) (h₁ : f (-1) k = 0) (h₂ : f (-2) k = 0) : k = 3 := 
by sorry

-- Prove the range of m such that f(x) < m for x in [-2, 2]
theorem range_of_m (m : ℝ) : (∀ x ∈ Set.Icc (-2 : ℝ) 2, x^2 + 3*x + 2 < m) ↔ 12 < m :=
by sorry

end find_k_range_of_m_l212_212072


namespace largest_angle_heptagon_l212_212353

theorem largest_angle_heptagon :
  ∃ (x : ℝ), 4 * x + 4 * x + 4 * x + 5 * x + 6 * x + 7 * x + 8 * x = 900 ∧ 8 * x = (7200 / 38) := 
by 
  sorry

end largest_angle_heptagon_l212_212353


namespace max_y_value_l212_212276

noncomputable def y (x : ℝ) : ℝ := |x + 1| - 2 * |x| + |x - 2|

theorem max_y_value : ∃ α, (∀ x, -1 ≤ x ∧ x ≤ 2 → y x ≤ α) ∧ α = 3 := by
  sorry

end max_y_value_l212_212276


namespace work_together_time_l212_212723

theorem work_together_time (man_days : ℝ) (son_days : ℝ)
  (h_man : man_days = 5) (h_son : son_days = 7.5) :
  (1 / (1 / man_days + 1 / son_days)) = 3 :=
by
  -- Given the constraints, prove the result
  rw [h_man, h_son]
  sorry

end work_together_time_l212_212723


namespace fraction_of_employees_laid_off_l212_212187

theorem fraction_of_employees_laid_off
    (total_employees : ℕ)
    (salary_per_employee : ℕ)
    (total_payment_after_layoffs : ℕ)
    (h1 : total_employees = 450)
    (h2 : salary_per_employee = 2000)
    (h3 : total_payment_after_layoffs = 600000) :
    (total_employees * salary_per_employee - total_payment_after_layoffs) / (total_employees * salary_per_employee) = 1 / 3 := 
by
    sorry

end fraction_of_employees_laid_off_l212_212187


namespace no_integer_solutions_l212_212909

theorem no_integer_solutions (m n : ℤ) : ¬ (m ^ 3 + 6 * m ^ 2 + 5 * m = 27 * n ^ 3 + 9 * n ^ 2 + 9 * n + 1) :=
sorry

end no_integer_solutions_l212_212909


namespace rational_sqrts_l212_212940

def is_rational (n : ℝ) : Prop := ∃ (q : ℚ), n = q

theorem rational_sqrts 
  (x y z : ℝ) 
  (hxr : is_rational x) 
  (hyr : is_rational y) 
  (hzr : is_rational z)
  (hw : is_rational (Real.sqrt x + Real.sqrt y + Real.sqrt z)) :
  is_rational (Real.sqrt x) ∧ is_rational (Real.sqrt y) ∧ is_rational (Real.sqrt z) :=
sorry

end rational_sqrts_l212_212940


namespace max_value_g_l212_212415

-- Defining the conditions and goal as functions and properties
def condition_1 (f : ℕ → ℕ) : Prop :=
  (Finset.range 43).sum f ≤ 2022

def condition_2 (f g : ℕ → ℕ) : Prop :=
  ∀ a b : ℕ, a >= b → g (a + b) ≤ f a + f b

-- Defining the main theorem to establish the maximum value
theorem max_value_g (f g : ℕ → ℕ) (h1 : condition_1 f) (h2 : condition_2 f g) :
  (Finset.range 85).sum g ≤ 7615 :=
sorry


end max_value_g_l212_212415


namespace solve_diamond_l212_212430

theorem solve_diamond : ∃ (D : ℕ), D < 10 ∧ (D * 9 + 5 = D * 10 + 2) ∧ D = 3 :=
by
  sorry

end solve_diamond_l212_212430


namespace find_k_and_max_ck_largest_ck_for_k0_largest_ck_for_k2_l212_212449

theorem find_k_and_max_ck:
  (∀ (x y z : ℝ), (x > 0) → (y > 0) → (z > 0) →
    ∃ (c_k : ℝ), c_k > 0 ∧ (x^2 + 1) * (y^2 + 1) * (z^2 + 1) ≥ c_k * (x + y + z)^k) →
  (∀ (k : ℝ), 0 ≤ k ∧ k ≤ 2) :=
by
  sorry

theorem largest_ck_for_k0:
  (∀ (x y z : ℝ), (x > 0) → (y > 0) → (z > 0) →
    (x^2 + 1) * (y^2 + 1) * (z^2 + 1) ≥ 1) := 
by
  sorry

theorem largest_ck_for_k2:
  (∀ (x y z : ℝ), (x > 0) → (y > 0) → (z > 0) →
    (x^2 + 1) * (y^2 + 1) * (z^2 + 1) ≥ (8/9) * (x + y + z)^2) :=
by
  sorry

end find_k_and_max_ck_largest_ck_for_k0_largest_ck_for_k2_l212_212449


namespace cannot_determine_right_triangle_l212_212972

-- Definitions of conditions
variables {a b c : ℕ}
variables {angle_A angle_B angle_C : ℕ}

-- Context for the proof
def is_right_angled_triangle_via_sides (a b c : ℕ) : Prop :=
  a^2 + b^2 = c^2

def triangle_angle_sum_theorem (angle_A angle_B angle_C : ℕ) : Prop :=
  angle_A + angle_B + angle_C = 180

-- Statements for conditions as used in the problem
def condition_A (a2 b2 c2 : ℕ) : Prop :=
  a2 = 1 ∧ b2 = 2 ∧ c2 = 3

def condition_B (a b c : ℕ) : Prop :=
  a = 3 ∧ b = 4 ∧ c = 5

def condition_C (angle_A angle_B angle_C : ℕ) : Prop :=
  angle_A + angle_B = angle_C

def condition_D (angle_A angle_B angle_C : ℕ) : Prop :=
  angle_A = 45 ∧ angle_B = 60 ∧ angle_C = 75

-- Proof statement
theorem cannot_determine_right_triangle (a b c angle_A angle_B angle_C : ℕ) :
  condition_D angle_A angle_B angle_C →
  ¬(is_right_angled_triangle_via_sides a b c) :=
sorry

end cannot_determine_right_triangle_l212_212972


namespace ages_of_Mel_and_Lexi_l212_212305

theorem ages_of_Mel_and_Lexi (M L K : ℤ)
  (h1 : M = K - 3)
  (h2 : L = M + 2)
  (h3 : K = 60) :
  M = 57 ∧ L = 59 :=
  by
    -- Proof steps are omitted.
    sorry

end ages_of_Mel_and_Lexi_l212_212305


namespace speed_of_river_l212_212892

theorem speed_of_river (speed_still_water : ℝ) (total_time : ℝ) (total_distance : ℝ) 
  (h_still_water: speed_still_water = 6) 
  (h_total_time: total_time = 1) 
  (h_total_distance: total_distance = 16/3) : 
  ∃ (speed_river : ℝ), speed_river = 2 :=
by 
  -- sorry is used to skip the proof
  sorry

end speed_of_river_l212_212892


namespace combination_sum_l212_212626

theorem combination_sum :
  (Nat.choose 3 2) + (Nat.choose 4 2) + (Nat.choose 5 2) + (Nat.choose 6 2) = 34 :=
by
  sorry

end combination_sum_l212_212626


namespace handshake_count_l212_212135

-- Defining the conditions
def number_of_companies : ℕ := 5
def representatives_per_company : ℕ := 5
def total_participants : ℕ := number_of_companies * representatives_per_company

-- Defining the number of handshakes each person makes
def handshakes_per_person : ℕ := total_participants - 1 - (representatives_per_company - 1)

-- Defining the total number of handshakes
def total_handshakes : ℕ := (total_participants * handshakes_per_person) / 2

theorem handshake_count :
  total_handshakes = 250 :=
by
  sorry

end handshake_count_l212_212135


namespace donation_total_is_correct_l212_212616

-- Definitions and conditions
def Megan_inheritance : ℤ := 1000000
def Dan_inheritance : ℤ := 10000
def donation_percentage : ℚ := 0.1
def Megan_donation := Megan_inheritance * donation_percentage
def Dan_donation := Dan_inheritance * donation_percentage
def total_donation := Megan_donation + Dan_donation

-- Theorem statement
theorem donation_total_is_correct : total_donation = 101000 := by
  sorry

end donation_total_is_correct_l212_212616


namespace product_of_roots_of_quartic_polynomial_l212_212625

theorem product_of_roots_of_quartic_polynomial :
  (∀ x : ℝ, (3 * x^4 - 8 * x^3 + x^2 - 10 * x - 24 = 0) → x = p ∨ x = q ∨ x = r ∨ x = s) →
  (p * q * r * s = -8) :=
by
  intros
  -- proof goes here
  sorry

end product_of_roots_of_quartic_polynomial_l212_212625


namespace distance_Bella_to_Galya_l212_212824

theorem distance_Bella_to_Galya (D_B D_V D_G : ℕ) (BV VG : ℕ)
  (hD_B : D_B = 700)
  (hD_V : D_V = 600)
  (hD_G : D_G = 650)
  (hBV : BV = 100)
  (hVG : VG = 50)
  : BV + VG = 150 := by
  sorry

end distance_Bella_to_Galya_l212_212824


namespace polynomial_inequality_l212_212740

theorem polynomial_inequality
  (x1 x2 x3 a b c : ℝ)
  (h1 : x1 > 0) 
  (h2 : x2 > 0) 
  (h3 : x3 > 0)
  (h4 : x1 + x2 + x3 ≤ 1)
  (h5 : x1^3 + a * x1^2 + b * x1 + c = 0)
  (h6 : x2^3 + a * x2^2 + b * x2 + c = 0)
  (h7 : x3^3 + a * x3^2 + b * x3 + c = 0) :
  a^3 * (1 + a + b) - 9 * c * (3 + 3 * a + a^2) ≤ 0 :=
sorry

end polynomial_inequality_l212_212740


namespace inverse_value_l212_212518

def g (x : ℝ) : ℝ := 4 * x^3 + 5

theorem inverse_value :
  g (-3) = -103 :=
by
  sorry

end inverse_value_l212_212518


namespace train_length_is_correct_l212_212123

-- Definitions
def speed_kmh := 48.0 -- in km/hr
def time_sec := 9.0 -- in seconds

-- Conversion function
def convert_speed (s_kmh : Float) : Float :=
  s_kmh * 1000 / 3600

-- Function to calculate length of train
def length_of_train (speed_kmh : Float) (time_sec : Float) : Float :=
  let speed_ms := convert_speed speed_kmh
  speed_ms * time_sec

-- Proof problem: Given the speed of the train and the time it takes to cross a pole, prove the length of the train
theorem train_length_is_correct : length_of_train speed_kmh time_sec = 119.97 :=
by
  sorry

end train_length_is_correct_l212_212123


namespace determine_m_l212_212567

-- Define the fractional equation condition
def fractional_eq (m x : ℝ) : Prop := (m/(x - 2) + 2*x/(x - 2) = 1)

-- Define the main theorem statement
theorem determine_m (m : ℝ) (h : ∃ (x : ℝ), x > 0 ∧ x ≠ 2 ∧ fractional_eq m x) : m = -4 :=
sorry

end determine_m_l212_212567


namespace simplify_expression_l212_212408

theorem simplify_expression : 
  -2^2003 + (-2)^2004 + 2^2005 - 2^2006 = -3 * 2^2003 := 
  by
    sorry

end simplify_expression_l212_212408


namespace system_solution_l212_212877

theorem system_solution (x y : ℝ) 
  (h1 : (x^2 + x * y + y^2) / (x^2 - x * y + y^2) = 3) 
  (h2 : x^3 + y^3 = 2) : x = 1 ∧ y = 1 :=
  sorry

end system_solution_l212_212877


namespace infinitely_many_n_divide_b_pow_n_plus_1_l212_212148

theorem infinitely_many_n_divide_b_pow_n_plus_1 (b : ℕ) (h1 : b > 2) :
  (∃ᶠ n in at_top, n^2 ∣ b^n + 1) ↔ ¬ ∃ k : ℕ, b + 1 = 2^k :=
sorry

end infinitely_many_n_divide_b_pow_n_plus_1_l212_212148


namespace find_second_sum_l212_212362

theorem find_second_sum (x : ℝ) (h : 24 * x / 100 = (2730 - x) * 15 / 100) : 2730 - x = 1680 := by
  sorry

end find_second_sum_l212_212362


namespace top_and_bottom_edges_same_color_l212_212509

-- Define the vertices for top and bottom pentagonal faces
inductive Vertex
| A1 | A2 | A3 | A4 | A5
| B1 | B2 | B3 | B4 | B5

-- Define the edges
inductive Edge : Type
| TopEdge (v1 v2 : Vertex) (h1 : v1 ∈ [Vertex.A1, Vertex.A2, Vertex.A3, Vertex.A4, Vertex.A5]) (h2 : v2 ∈ [Vertex.A1, Vertex.A2, Vertex.A3, Vertex.A4, Vertex.A5]) : Edge
| BottomEdge (v1 v2 : Vertex) (h1 : v1 ∈ [Vertex.B1, Vertex.B2, Vertex.B3, Vertex.B4, Vertex.B5]) (h2 : v2 ∈ [Vertex.B1, Vertex.B2, Vertex.B3, Vertex.B4, Vertex.B5]) : Edge
| SideEdge (v1 v2 : Vertex) (h1 : v1 ∈ [Vertex.A1, Vertex.A2, Vertex.A3, Vertex.A4, Vertex.A5]) (h2 : v2 ∈ [Vertex.B1, Vertex.B2, Vertex.B3, Vertex.B4, Vertex.B5]) : Edge

-- Define colors
inductive Color
| Red | Blue

-- Define a function that assigns a color to each edge
def edgeColor : Edge → Color := sorry

-- Define a function that checks if a triangle is monochromatic
def isMonochromatic (e1 e2 e3 : Edge) : Prop :=
  edgeColor e1 = edgeColor e2 ∧ edgeColor e2 = edgeColor e3

-- Define our main theorem statement
theorem top_and_bottom_edges_same_color (h : ∀ v1 v2 v3 : Vertex, ¬ isMonochromatic (Edge.TopEdge v1 v2 sorry sorry) (Edge.SideEdge v1 v3 sorry sorry) (Edge.BottomEdge v2 v3 sorry sorry)) : 
  (∀ (v1 v2 : Vertex), v1 ∈ [Vertex.A1, Vertex.A2, Vertex.A3, Vertex.A4, Vertex.A5] → v2 ∈ [Vertex.A1, Vertex.A2, Vertex.A3, Vertex.A4, Vertex.A5] → edgeColor (Edge.TopEdge v1 v2 sorry sorry) = edgeColor (Edge.TopEdge Vertex.A1 Vertex.A2 sorry sorry)) ∧
  (∀ (v1 v2 : Vertex), v1 ∈ [Vertex.B1, Vertex.B2, Vertex.B3, Vertex.B4, Vertex.B5] → v2 ∈ [Vertex.B1, Vertex.B2, Vertex.B3, Vertex.B4, Vertex.B5] → edgeColor (Edge.BottomEdge v1 v2 sorry sorry) = edgeColor (Edge.BottomEdge Vertex.B1 Vertex.B2 sorry sorry)) :=
sorry

end top_and_bottom_edges_same_color_l212_212509


namespace steve_break_even_l212_212544

noncomputable def break_even_performances
  (fixed_overhead : ℕ)
  (min_production_cost max_production_cost : ℕ)
  (venue_capacity percentage_occupied : ℕ)
  (ticket_price : ℕ) : ℕ :=
(fixed_overhead + (percentage_occupied / 100 * venue_capacity * ticket_price)) / (percentage_occupied / 100 * venue_capacity * ticket_price)

theorem steve_break_even
  (fixed_overhead : ℕ := 81000)
  (min_production_cost : ℕ := 5000)
  (max_production_cost : ℕ := 9000)
  (venue_capacity : ℕ := 500)
  (percentage_occupied : ℕ := 80)
  (ticket_price : ℕ := 40)
  (avg_production_cost : ℕ := (min_production_cost + max_production_cost) / 2) :
  break_even_performances fixed_overhead min_production_cost max_production_cost venue_capacity percentage_occupied ticket_price = 9 :=
by
  sorry

end steve_break_even_l212_212544


namespace value_multiplied_by_15_l212_212964

theorem value_multiplied_by_15 (x : ℝ) (h : 3.6 * x = 10.08) : x * 15 = 42 :=
sorry

end value_multiplied_by_15_l212_212964


namespace remainder_of_power_mod_l212_212009

theorem remainder_of_power_mod 
  (n : ℕ)
  (h₁ : 7 ≡ 1 [MOD 6]) : 7^51 ≡ 1 [MOD 6] := 
sorry

end remainder_of_power_mod_l212_212009


namespace mohan_cookies_l212_212835

theorem mohan_cookies :
  ∃ (a : ℕ), 
    (a % 6 = 5) ∧ 
    (a % 7 = 3) ∧ 
    (a % 9 = 7) ∧ 
    (a % 11 = 10) ∧ 
    (a = 1817) :=
sorry

end mohan_cookies_l212_212835


namespace number_of_cities_l212_212542

theorem number_of_cities (n : ℕ) (h : n * (n - 1) / 2 = 15) : n = 6 :=
sorry

end number_of_cities_l212_212542


namespace polynomial_roots_problem_l212_212433

theorem polynomial_roots_problem (γ δ : ℝ) (h₁ : γ^2 - 3*γ + 2 = 0) (h₂ : δ^2 - 3*δ + 2 = 0) :
  8*γ^3 - 6*δ^2 = 48 :=
by
  sorry

end polynomial_roots_problem_l212_212433


namespace scientific_notation_of_sesame_mass_l212_212045

theorem scientific_notation_of_sesame_mass :
  0.00000201 = 2.01 * 10^(-6) :=
sorry

end scientific_notation_of_sesame_mass_l212_212045


namespace problem_solution_l212_212890

-- Define the structure of the dartboard and scoring
structure Dartboard where
  inner_radius : ℝ
  intermediate_radius : ℝ
  outer_radius : ℝ
  regions : List (List ℤ) -- List of lists representing scores in the regions

-- Define the probability calculation function
noncomputable def probability_odd_score (d : Dartboard) : ℚ := sorry

-- Define the specific dartboard with given conditions
def revised_dartboard : Dartboard :=
  { inner_radius := 4.5,
    intermediate_radius := 6.75,
    outer_radius := 9,
    regions := [[3, 2, 2], [2, 1, 1], [1, 1, 3]] }

-- The theorem to prove the solution to the problem
theorem problem_solution : probability_odd_score revised_dartboard = 265 / 855 :=
  sorry

end problem_solution_l212_212890


namespace trig_expression_value_l212_212470

open Real

theorem trig_expression_value (θ : ℝ)
  (h1 : cos (π - θ) > 0)
  (h2 : cos (π / 2 + θ) * (1 - 2 * cos (θ / 2) ^ 2) < 0) :
  (sin θ / |sin θ|) + (|cos θ| / cos θ) + (tan θ / |tan θ|) = -1 :=
by
  sorry

end trig_expression_value_l212_212470


namespace sum_of_integers_satisfying_l212_212687

theorem sum_of_integers_satisfying (x : ℤ) (h : x^2 = 272 + x) : ∃ y : ℤ, y = 1 :=
sorry

end sum_of_integers_satisfying_l212_212687


namespace count_odd_perfect_squares_less_than_16000_l212_212752

theorem count_odd_perfect_squares_less_than_16000 : 
  ∃ n : ℕ, n = 31 ∧ ∀ k < 16000, 
    ∃ b : ℕ, b = 2 * n + 1 ∧ k = (4 * n + 3) ^ 2 ∧ (∃ m : ℕ, m = b + 1 ∧ m % 2 = 0) := 
sorry

end count_odd_perfect_squares_less_than_16000_l212_212752


namespace find_original_cost_price_l212_212683

theorem find_original_cost_price (C S C_new S_new : ℝ) (h1 : S = 1.25 * C) (h2 : C_new = 0.80 * C) (h3 : S_new = S - 16.80) (h4 : S_new = 1.04 * C_new) : C = 80 :=
by
  sorry

end find_original_cost_price_l212_212683


namespace solve_equation_1_solve_equation_2_l212_212414

theorem solve_equation_1 (y: ℝ) : y^2 - 6 * y + 1 = 0 ↔ (y = 3 + 2 * Real.sqrt 2 ∨ y = 3 - 2 * Real.sqrt 2) :=
sorry

theorem solve_equation_2 (x: ℝ) : 2 * (x - 4)^2 = x^2 - 16 ↔ (x = 4 ∨ x = 12) :=
sorry

end solve_equation_1_solve_equation_2_l212_212414


namespace find_max_m_l212_212358

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := (1/2) * Real.exp (2 * x) - a * x

noncomputable def g (x : ℝ) (m : ℝ) : ℝ := (x - m) * f x 1 - (1/4) * Real.exp (2 * x) + x^2 + x

theorem find_max_m (h_inc : ∀ x > 0, g x m ≥ g x m) : m ≤ 1 :=
by
  sorry

end find_max_m_l212_212358


namespace parabola_equation_focus_l212_212529

theorem parabola_equation_focus (p : ℝ) (h₀ : p > 0)
  (h₁ : (p / 2 = 2)) : (y^2 = 2 * p * x) :=
  sorry

end parabola_equation_focus_l212_212529


namespace maria_paper_count_l212_212679

-- Defining the initial number of sheets and the actions taken
variables (x y : ℕ)
def initial_sheets := 50 + 41
def remaining_sheets_after_giving_away := initial_sheets - x
def whole_sheets := remaining_sheets_after_giving_away - y
def half_sheets := y

-- The theorem we want to prove
theorem maria_paper_count (x y : ℕ) :
  whole_sheets x y = initial_sheets - x - y ∧ 
  half_sheets y = y :=
by sorry

end maria_paper_count_l212_212679


namespace total_people_veg_l212_212306

def people_only_veg : ℕ := 13
def people_both_veg_nonveg : ℕ := 8

theorem total_people_veg : people_only_veg + people_both_veg_nonveg = 21 := by
  sorry

end total_people_veg_l212_212306


namespace find_x_plus_y_l212_212836

-- Define the vectors
def vector_a : ℝ × ℝ := (1, 1)
def vector_b (x : ℝ) : ℝ × ℝ := (x, -2)
def vector_c (y : ℝ) : ℝ × ℝ := (-1, y)

-- Define the conditions
def perpendicular (v1 v2 : ℝ × ℝ) : Prop := v1.1 * v2.1 + v1.2 * v2.2 = 0
def parallel (v1 v2 : ℝ × ℝ) : Prop := ∃ k : ℝ, v2.1 = k * v1.1 ∧ v2.2 = k * v1.2

-- State the theorem
theorem find_x_plus_y (x y : ℝ)
  (h1 : perpendicular vector_a (vector_b x))
  (h2 : parallel vector_a (vector_c y)) :
  x + y = 1 :=
sorry

end find_x_plus_y_l212_212836


namespace Heath_current_age_l212_212215

variable (H J : ℕ) -- Declare variables for Heath's and Jude's ages
variable (h1 : J = 2) -- Jude's current age is 2
variable (h2 : H + 5 = 3 * (J + 5)) -- In 5 years, Heath will be 3 times as old as Jude

theorem Heath_current_age : H = 16 :=
by
  -- Proof to be filled in later
  sorry

end Heath_current_age_l212_212215


namespace area_of_rectangle_l212_212721

theorem area_of_rectangle (length width : ℝ) (h1 : length = 15) (h2 : width = length * 0.9) : length * width = 202.5 := by
  sorry

end area_of_rectangle_l212_212721


namespace area_of_rectangle_l212_212921

theorem area_of_rectangle (length width : ℝ) (h_length : length = 47.3) (h_width : width = 24) :
  length * width = 1135.2 :=
by
  sorry -- Skip the proof

end area_of_rectangle_l212_212921


namespace largest_divisor_of_m_l212_212540

-- Definitions
def positive_integer (m : ℕ) : Prop := m > 0
def divisible_by (a b : ℕ) : Prop := ∃ k : ℕ, a = b * k

-- Statement
theorem largest_divisor_of_m (m : ℕ) (h1 : positive_integer m) (h2 : divisible_by (m^2) 54) : ∃ k : ℕ, k = 9 ∧ k ∣ m := 
sorry

end largest_divisor_of_m_l212_212540


namespace single_reduction_equivalent_l212_212945

/-- If a price is first reduced by 25%, and the new price is further reduced by 30%, 
the single percentage reduction equivalent to these two reductions together is 47.5%. -/
theorem single_reduction_equivalent :
  ∀ P : ℝ, (1 - 0.25) * (1 - 0.30) * P = P * (1 - 0.475) :=
by
  intros
  sorry

end single_reduction_equivalent_l212_212945


namespace extreme_point_property_l212_212208

variables (f : ℝ → ℝ) (a b x x₀ x₁ : ℝ) 

-- Define the function f
def func (x : ℝ) := x^3 - a * x - b

-- The main theorem
theorem extreme_point_property (h₀ : ∃ x₀, ∃ x₁, (x₀ ≠ 0) ∧ (x₀^2 = a / 3) ∧ (x₁ ≠ x₀) ∧ (func a b x₀ = func a b x₁)) :
  x₁ + 2 * x₀ = 0 :=
sorry

end extreme_point_property_l212_212208


namespace nth_term_arithmetic_sequence_l212_212730

variable (n r : ℕ)

def S (n : ℕ) : ℕ := 4 * n + 5 * n^2

theorem nth_term_arithmetic_sequence :
  (S r) - (S (r-1)) = 10 * r - 1 :=
by
  sorry

end nth_term_arithmetic_sequence_l212_212730


namespace train_crosses_pole_time_l212_212089

theorem train_crosses_pole_time
  (l : ℕ) (v_kmh : ℕ) (v_ms : ℚ) (t : ℕ)
  (h_l : l = 100)
  (h_v_kmh : v_kmh = 180)
  (h_v_ms_conversion : v_ms = v_kmh * 1000 / 3600)
  (h_v_ms : v_ms = 50) :
  t = l / v_ms := by
  sorry

end train_crosses_pole_time_l212_212089


namespace alice_min_speed_exceeds_45_l212_212343

theorem alice_min_speed_exceeds_45 
  (distance : ℕ)
  (bob_speed : ℕ)
  (alice_delay : ℕ)
  (alice_speed : ℕ)
  (bob_time : ℕ)
  (expected_speed : ℕ) 
  (distance_eq : distance = 180)
  (bob_speed_eq : bob_speed = 40)
  (alice_delay_eq : alice_delay = 1/2)
  (bob_time_eq : bob_time = distance / bob_speed)
  (expected_speed_eq : expected_speed = distance / (bob_time - alice_delay)) :
  alice_speed > expected_speed := 
sorry

end alice_min_speed_exceeds_45_l212_212343


namespace find_number_l212_212036

theorem find_number (x : ℤ) (h : 5 * x + 4 = 19) : x = 3 := sorry

end find_number_l212_212036


namespace sara_height_l212_212649

def Julie := 33
def Mark := Julie + 1
def Roy := Mark + 2
def Joe := Roy + 3
def Sara := Joe + 6

theorem sara_height : Sara = 45 := by
  sorry

end sara_height_l212_212649


namespace remaining_lemons_proof_l212_212562

-- Definitions for initial conditions
def initial_lemons_first_tree   := 15
def initial_lemons_second_tree  := 20
def initial_lemons_third_tree   := 25

def sally_picked_first_tree     := 7
def mary_picked_second_tree     := 9
def tom_picked_first_tree       := 12

def lemons_fell_each_tree       := 4
def animals_eaten_per_tree      := lemons_fell_each_tree / 2

-- Definitions for intermediate calculations
def remaining_lemons_first_tree_full := initial_lemons_first_tree - sally_picked_first_tree - tom_picked_first_tree
def remaining_lemons_first_tree      := if remaining_lemons_first_tree_full < 0 then 0 else remaining_lemons_first_tree_full

def remaining_lemons_second_tree := initial_lemons_second_tree - mary_picked_second_tree

def mary_picked_third_tree := (remaining_lemons_second_tree : ℚ) / 2
def remaining_lemons_third_tree_full := (initial_lemons_third_tree : ℚ) - mary_picked_third_tree
def remaining_lemons_third_tree      := Nat.floor remaining_lemons_third_tree_full

-- Adjusting for fallen and eaten lemons
def final_remaining_lemons_first_tree_full := remaining_lemons_first_tree - lemons_fell_each_tree + animals_eaten_per_tree
def final_remaining_lemons_first_tree      := if final_remaining_lemons_first_tree_full < 0 then 0 else final_remaining_lemons_first_tree_full

def final_remaining_lemons_second_tree     := remaining_lemons_second_tree - lemons_fell_each_tree + animals_eaten_per_tree

def final_remaining_lemons_third_tree_full := remaining_lemons_third_tree - lemons_fell_each_tree + animals_eaten_per_tree
def final_remaining_lemons_third_tree      := if final_remaining_lemons_third_tree_full < 0 then 0 else final_remaining_lemons_third_tree_full

-- Lean 4 statement to prove the equivalence
theorem remaining_lemons_proof :
  final_remaining_lemons_first_tree = 0 ∧
  final_remaining_lemons_second_tree = 9 ∧
  final_remaining_lemons_third_tree = 18 :=
by
  -- The proof is omitted as per the requirement
  sorry

end remaining_lemons_proof_l212_212562


namespace inequality_proof_l212_212125

variables {x y z : ℝ}

theorem inequality_proof 
  (h1 : y ≥ 2 * z) 
  (h2 : 2 * z ≥ 4 * x) 
  (h3 : 2 * (x^3 + y^3 + z^3) + 15 * (x * y^2 + y * z^2 + z * x^2) ≥ 16 * (x^2 * y + y^2 * z + z^2 * x) + 2 * x * y * z) : 
  4 * x + y ≥ 4 * z :=
sorry

end inequality_proof_l212_212125


namespace freddy_spent_10_dollars_l212_212637

theorem freddy_spent_10_dollars 
  (talk_time_dad : ℕ) (talk_time_brother : ℕ) 
  (local_cost_per_minute : ℕ) (international_cost_per_minute : ℕ)
  (conversion_cents_to_dollar : ℕ)
  (h1 : talk_time_dad = 45)
  (h2 : talk_time_brother = 31)
  (h3 : local_cost_per_minute = 5)
  (h4 : international_cost_per_minute = 25)
  (h5 : conversion_cents_to_dollar = 100):
  (local_cost_per_minute * talk_time_dad + international_cost_per_minute * talk_time_brother) / conversion_cents_to_dollar = 10 :=
by
  sorry

end freddy_spent_10_dollars_l212_212637


namespace quadratic_real_solutions_l212_212270

theorem quadratic_real_solutions (p : ℝ) : (∃ x : ℝ, x^2 + p = 0) ↔ p ≤ 0 :=
sorry

end quadratic_real_solutions_l212_212270


namespace sum_first_3k_plus_2_terms_l212_212096

variable (k : ℕ)

def first_term : ℕ := k^2 + 1

def sum_of_sequence (n : ℕ) : ℕ :=
  let a₁ := first_term k
  let aₙ := a₁ + (n - 1)
  n * (a₁ + aₙ) / 2

theorem sum_first_3k_plus_2_terms :
  sum_of_sequence k (3 * k + 2) = 3 * k^3 + 8 * k^2 + 6 * k + 3 :=
by
  -- Here we define the sequence and compute the sum
  sorry

end sum_first_3k_plus_2_terms_l212_212096


namespace isabella_paint_area_l212_212447

theorem isabella_paint_area 
    (bedrooms : ℕ) 
    (length width height doorway_window_area : ℕ) 
    (h1 : bedrooms = 4) 
    (h2 : length = 14) 
    (h3 : width = 12) 
    (h4 : height = 9)
    (h5 : doorway_window_area = 80) :
    (2 * (length * height) + 2 * (width * height) - doorway_window_area) * bedrooms = 1552 := by
       -- Calculate the area of the walls in one bedroom
       -- 2 * (length * height) + 2 * (width * height) - doorway_window_area = 388
       -- The total paintable area for 4 bedrooms = 388 * 4 = 1552
       sorry

end isabella_paint_area_l212_212447


namespace percentage_donated_l212_212184

def income : ℝ := 1200000
def children_percentage : ℝ := 0.20
def wife_percentage : ℝ := 0.30
def remaining : ℝ := income - (children_percentage * 3 * income + wife_percentage * income)
def left_amount : ℝ := 60000
def donated : ℝ := remaining - left_amount

theorem percentage_donated : (donated / remaining) * 100 = 50 := by
  sorry

end percentage_donated_l212_212184


namespace red_marbles_more_than_yellow_l212_212234

-- Define the conditions
def total_marbles : ℕ := 19
def yellow_marbles : ℕ := 5
def ratio_parts_blue : ℕ := 3
def ratio_parts_red : ℕ := 4

-- Prove the number of more red marbles than yellow marbles is equal to 3
theorem red_marbles_more_than_yellow :
  let remainder_marbles := total_marbles - yellow_marbles
  let total_parts := ratio_parts_blue + ratio_parts_red
  let marbles_per_part := remainder_marbles / total_parts
  let blue_marbles := ratio_parts_blue * marbles_per_part
  let red_marbles := ratio_parts_red * marbles_per_part
  red_marbles - yellow_marbles = 3 := by
  sorry

end red_marbles_more_than_yellow_l212_212234


namespace find_k_l212_212457

open Complex

noncomputable def possible_values_of_k (a b c d e : ℂ) (k : ℂ) : Prop :=
  (a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0 ∧ e ≠ 0) ∧
  (a * k^4 + b * k^3 + c * k^2 + d * k + e = 0) ∧
  (b * k^4 + c * k^3 + d * k^2 + e * k + a = 0)

theorem find_k (a b c d e : ℂ) (k : ℂ) :
  possible_values_of_k a b c d e k → k^5 = 1 :=
by
  intro h
  sorry

#check find_k

end find_k_l212_212457


namespace total_sections_formed_l212_212779

theorem total_sections_formed (boys girls : ℕ) (hb : boys = 408) (hg : girls = 264) :
  let gcd := Nat.gcd boys girls
  let boys_sections := boys / gcd
  let girls_sections := girls / gcd
  boys_sections + girls_sections = 28 := 
by
  -- Note: this will assert the theorem, but the proof is omitted with sorry.
  sorry

end total_sections_formed_l212_212779


namespace find_C_l212_212898

theorem find_C (A B C : ℕ) 
  (h1 : A + B + C = 700) 
  (h2 : A + C = 300) 
  (h3 : B + C = 600) 
  : C = 200 := sorry

end find_C_l212_212898


namespace total_amount_before_brokerage_l212_212871

variable (A : ℝ)

theorem total_amount_before_brokerage 
  (cash_realized : ℝ) 
  (brokerage_rate : ℝ) 
  (h1 : cash_realized = 106.25) 
  (h2 : brokerage_rate = 1 / 400) :
  A = 42500 / 399 :=
by
  sorry

end total_amount_before_brokerage_l212_212871


namespace proof_expr1_l212_212837

noncomputable def expr1 : ℝ :=
  (Real.sin (65 * Real.pi / 180) + Real.sin (15 * Real.pi / 180) * Real.sin (10 * Real.pi / 180)) / 
  (Real.sin (25 * Real.pi / 180) - Real.cos (15 * Real.pi / 180) * Real.cos (80 * Real.pi / 180))

theorem proof_expr1 : expr1 = 2 + Real.sqrt 3 :=
by sorry

end proof_expr1_l212_212837


namespace compare_f_g_l212_212658

def R (m n : ℕ) : ℕ := sorry
def L (m n : ℕ) : ℕ := sorry

def f (m n : ℕ) : ℕ := R m n + L m n - sorry
def g (m n : ℕ) : ℕ := R m n + L m n - sorry

theorem compare_f_g (m n : ℕ) : f m n ≤ g m n := sorry

end compare_f_g_l212_212658


namespace num_correct_conclusions_l212_212572

-- Definitions and conditions from the problem
variable {a : ℕ → ℚ}
variable {S : ℕ → ℚ}
variable (n : ℕ)
variable (hSn_eq : S n + S (n + 1) = n ^ 2)

-- Assert the conditions described in the comments
theorem num_correct_conclusions (hSn_eq : ∀ n, S n + S (n + 1) = n ^ 2) :
  (1:ℕ) = 3 ↔
  (-- Conclusion 1
   ¬(∀ n, a (n + 2) - a n = 2) ∧
   -- Conclusion 2: If a_1 = 0, then S_50 = 1225
   (S 50 = 1225) ∧
   -- Conclusion 3: If a_1 = 1, then S_50 = 1224
   (S 50 = 1224) ∧
   -- Conclusion 4: Monotonically increasing sequence
   (∀ a_1, (-1/4 : ℚ) < a_1 ∧ a_1 < 1/4)) :=
by
  sorry

end num_correct_conclusions_l212_212572


namespace value_of_A_l212_212577

def clubsuit (A B : ℕ) := 3 * A + 2 * B + 5

theorem value_of_A (A : ℕ) (h : clubsuit A 7 = 82) : A = 21 :=
by
  sorry

end value_of_A_l212_212577


namespace goats_at_farm_l212_212156

theorem goats_at_farm (G C D P : ℕ) 
  (h1: C = 2 * G)
  (h2: D = (G + C) / 2)
  (h3: P = D / 3)
  (h4: G = P + 33) :
  G = 66 :=
by
  sorry

end goats_at_farm_l212_212156


namespace toys_ratio_l212_212202

theorem toys_ratio (k A M T : ℕ) (h1 : M = 6) (h2 : A = k * M) (h3 : A = T - 2) (h4 : A + M + T = 56):
  A / M = 4 :=
by
  sorry

end toys_ratio_l212_212202


namespace TinaTotalPens_l212_212064

variable (p g b : ℕ)
axiom H1 : p = 12
axiom H2 : g = p - 9
axiom H3 : b = g + 3

theorem TinaTotalPens : p + g + b = 21 := by
  sorry

end TinaTotalPens_l212_212064


namespace barbara_spent_on_other_goods_l212_212049

theorem barbara_spent_on_other_goods
  (cost_tuna : ℝ := 5 * 2)
  (cost_water : ℝ := 4 * 1.5)
  (total_paid : ℝ := 56) :
  total_paid - (cost_tuna + cost_water) = 40 := by
  sorry

end barbara_spent_on_other_goods_l212_212049


namespace sum_of_roots_l212_212079

theorem sum_of_roots (x y : ℝ) (h : ∀ z, z^2 + 2023 * z - 2024 = 0 → z = x ∨ z = y) : x + y = -2023 := 
by
  sorry

end sum_of_roots_l212_212079


namespace union_of_sets_l212_212247

def A : Set ℝ := { x : ℝ | x * (x + 1) ≤ 0 }
def B : Set ℝ := { x : ℝ | -1 < x ∧ x < 1 }
def C : Set ℝ := { x : ℝ | -1 ≤ x ∧ x < 1 }

theorem union_of_sets :
  A ∪ B = C := 
sorry

end union_of_sets_l212_212247


namespace car_B_speed_is_50_l212_212961

def car_speeds (v_A v_B : ℕ) (d_init d_ahead t : ℝ) : Prop :=
  v_A * t = v_B * t + d_init + d_ahead

theorem car_B_speed_is_50 :
  car_speeds 58 50 10 8 2.25 :=
by
  sorry

end car_B_speed_is_50_l212_212961


namespace problem_statement_l212_212895

theorem problem_statement
  (x y : ℝ)
  (h1 : 4 * x + 2 * y = 12)
  (h2 : 2 * x + 4 * y = 20) :
  20 * x^2 + 24 * x * y + 20 * y^2 = 544 :=
  sorry

end problem_statement_l212_212895


namespace average_price_of_dvds_l212_212428

theorem average_price_of_dvds :
  let num_dvds_box1 := 10
  let price_per_dvd_box1 := 2.00
  let num_dvds_box2 := 5
  let price_per_dvd_box2 := 5.00
  let total_cost_box1 := num_dvds_box1 * price_per_dvd_box1
  let total_cost_box2 := num_dvds_box2 * price_per_dvd_box2
  let total_dvds := num_dvds_box1 + num_dvds_box2
  let total_cost := total_cost_box1 + total_cost_box2
  (total_cost / total_dvds) = 3.00 := 
sorry

end average_price_of_dvds_l212_212428


namespace cube_painting_l212_212333

theorem cube_painting (n : ℕ) (h : n > 2) :
  (6 * (n - 2)^2 = (n - 2)^3) ↔ (n = 8) :=
by
  sorry

end cube_painting_l212_212333


namespace probability_of_C_l212_212994

-- Definitions of probabilities for regions A, B, and D
def P_A : ℚ := 3 / 8
def P_B : ℚ := 1 / 4
def P_D : ℚ := 1 / 8

-- Sum of probabilities must be 1
def total_probability : ℚ := 1

-- The main proof statement
theorem probability_of_C : 
  P_A + P_B + P_D + (P_C : ℚ) = total_probability → P_C = 1 / 4 := sorry

end probability_of_C_l212_212994


namespace Bill_Sunday_miles_l212_212931

-- Definitions based on problem conditions
def Bill_Saturday (B : ℕ) : ℕ := B
def Bill_Sunday (B : ℕ) : ℕ := B + 4
def Julia_Sunday (B : ℕ) : ℕ := 2 * (B + 4)
def Alex_Total (B : ℕ) : ℕ := B + 2

-- Total miles equation based on conditions
def total_miles (B : ℕ) : ℕ := Bill_Saturday B + Bill_Sunday B + Julia_Sunday B + Alex_Total B

-- Proof statement
theorem Bill_Sunday_miles (B : ℕ) (h : total_miles B = 54) : Bill_Sunday B = 14 :=
by {
  -- calculations and proof would go here if not omitted
  sorry
}

end Bill_Sunday_miles_l212_212931


namespace KayleeAgeCorrect_l212_212251

-- Define Kaylee's current age
def KayleeCurrentAge (k : ℕ) : Prop :=
  (3 * 5 + (7 - k) = 7)

-- State the theorem
theorem KayleeAgeCorrect : ∃ k : ℕ, KayleeCurrentAge k ∧ k = 8 := 
sorry

end KayleeAgeCorrect_l212_212251


namespace vaclav_multiplication_correct_l212_212761

-- Definitions of the involved numbers and their multiplication consistency.
def a : ℕ := 452
def b : ℕ := 125
def result : ℕ := 56500

-- The main theorem statement proving the correctness of the multiplication.
theorem vaclav_multiplication_correct : a * b = result :=
by sorry

end vaclav_multiplication_correct_l212_212761


namespace inequality_proof_l212_212528

noncomputable def f (a x : ℝ) : ℝ := (1 - x) / (a * x) + Real.log x
noncomputable def g (x : ℝ) : ℝ := Real.log (1 + x) - x

theorem inequality_proof (a b : ℝ) (ha : 1 < a) (hb : 0 < b) : 
  f a (a + b) > f a 1 → g (a / b) < g 0 → 1 / (a + b) < Real.log (a + b) / b ∧ Real.log (a + b) / b < a / b := 
by
  sorry

end inequality_proof_l212_212528


namespace matrix_vector_product_l212_212175

-- Definitions for matrix A and vector v
def A : Matrix (Fin 2) (Fin 2) ℤ := ![
  ![-3, 4],
  ![2, -1]
]

def v : Fin 2 → ℤ := ![2, -2]

-- The theorem to prove
theorem matrix_vector_product :
  (A.mulVec v) = ![-14, 6] :=
by sorry

end matrix_vector_product_l212_212175


namespace tim_books_l212_212315

def has_some_books (Tim Sam : ℕ) : Prop :=
  Sam = 52 ∧ Tim + Sam = 96

theorem tim_books (Tim : ℕ) :
  has_some_books Tim 52 → Tim = 44 := 
by
  intro h
  obtain ⟨hSam, hTogether⟩ := h
  sorry

end tim_books_l212_212315


namespace prism_volume_is_correct_l212_212662

noncomputable def prism_volume 
  (a b c : ℝ) 
  (hab : a * b = 15) 
  (hbc : b * c = 18) 
  (hca : c * a = 20) 
  (hc_longest : c = 2 * a) 
  : ℝ :=
  a * b * c

theorem prism_volume_is_correct 
  (a b c : ℝ) 
  (hab : a * b = 15) 
  (hbc : b * c = 18) 
  (hca : c * a = 20) 
  (hc_longest : c = 2 * a) 
  : prism_volume a b c hab hbc hca hc_longest = 30 * Real.sqrt 10 :=
sorry

end prism_volume_is_correct_l212_212662


namespace angle_BDC_correct_l212_212742

theorem angle_BDC_correct (A B C D : Type) 
  (angle_A : ℝ) (angle_B : ℝ) (angle_DBC : ℝ) : 
  angle_A = 60 ∧ angle_B = 70 ∧ angle_DBC = 40 → 
  ∃ angle_BDC : ℝ, angle_BDC = 100 := 
by
  intro h
  sorry

end angle_BDC_correct_l212_212742


namespace tom_total_payment_l212_212659

def lemon_price : Nat := 2
def papaya_price : Nat := 1
def mango_price : Nat := 4
def discount_per_4_fruits : Nat := 1
def num_lemons : Nat := 6
def num_papayas : Nat := 4
def num_mangos : Nat := 2

theorem tom_total_payment :
  lemon_price * num_lemons + papaya_price * num_papayas + mango_price * num_mangos 
  - (num_lemons + num_papayas + num_mangos) / 4 * discount_per_4_fruits = 21 := 
by sorry

end tom_total_payment_l212_212659


namespace max_value_of_g_l212_212436

noncomputable def g (x : ℝ) : ℝ := 5 * x^2 - 2 * x^4

theorem max_value_of_g : ∃ x ∈ Set.Icc (0:ℝ) 2, g x = 25 / 8 := 
by 
  sorry

end max_value_of_g_l212_212436


namespace polynomial_division_l212_212001

noncomputable def poly1 : Polynomial ℤ := Polynomial.X ^ 13 - Polynomial.X + 100
noncomputable def poly2 : Polynomial ℤ := Polynomial.X ^ 2 + Polynomial.X + 2

theorem polynomial_division : ∃ q : Polynomial ℤ, poly1 = poly2 * q :=
by 
  sorry

end polynomial_division_l212_212001


namespace symmetric_point_m_eq_one_l212_212262

theorem symmetric_point_m_eq_one (m : ℝ) (A B : ℝ × ℝ) 
  (hA : A = (-3, 2 * m - 1))
  (hB : B = (-3, -1))
  (symmetric : A.1 = B.1 ∧ A.2 = -B.2) : 
  m = 1 :=
by
  sorry

end symmetric_point_m_eq_one_l212_212262


namespace minimum_n_value_l212_212023

theorem minimum_n_value : ∃ n : ℕ, n > 0 ∧ ∀ r : ℕ, (2 * n = 5 * r) → n = 5 :=
by
  sorry

end minimum_n_value_l212_212023


namespace total_amount_proof_l212_212569

def total_shared_amount : ℝ :=
  let z := 250
  let y := 1.20 * z
  let x := 1.25 * y
  x + y + z

theorem total_amount_proof : total_shared_amount = 925 :=
by
  sorry

end total_amount_proof_l212_212569


namespace num_ways_to_remove_blocks_l212_212350

-- Definitions based on the problem conditions
def stack_blocks := 85
def block_layers := [1, 4, 16, 64]

-- Theorem statement
theorem num_ways_to_remove_blocks : 
  (∃ f : (ℕ → ℕ), 
    (∀ n, f n = if n = 0 then 1 else if n ≤ 4 then n * f (n - 1) + 3 * (f (n - 1) - 1) else 4^3 * 16) ∧ 
    f 5 = 3384) := sorry

end num_ways_to_remove_blocks_l212_212350


namespace cos_75_deg_l212_212432

theorem cos_75_deg : Real.cos (75 * Real.pi / 180) = (Real.sqrt 6 - Real.sqrt 2) / 4 := by
  sorry

end cos_75_deg_l212_212432


namespace train_speed_l212_212843

noncomputable def train_length : ℝ := 65 -- length of the train in meters
noncomputable def time_to_pass : ℝ := 6.5 -- time to pass the telegraph post in seconds
noncomputable def speed_conversion_factor : ℝ := 18 / 5 -- conversion factor from m/s to km/h

theorem train_speed (h_length : train_length = 65) (h_time : time_to_pass = 6.5) :
  (train_length / time_to_pass) * speed_conversion_factor = 36 :=
by
  simp [h_length, h_time, train_length, time_to_pass, speed_conversion_factor]
  sorry

end train_speed_l212_212843


namespace pow_mod_eq_l212_212265

theorem pow_mod_eq :
  11 ^ 2023 % 5 = 1 :=
by
  sorry

end pow_mod_eq_l212_212265


namespace g_at_5_l212_212762

def g (x : ℝ) : ℝ := sorry

axiom functional_equation : ∀ (x : ℝ), g x + 2 * g (1 - x) = x^2 + 2 * x

theorem g_at_5 : g 5 = -19 / 3 :=
by {
  sorry
}

end g_at_5_l212_212762


namespace sequence_a_5_l212_212764

noncomputable section

-- Definition of the sequence
def a : ℕ → ℕ
| 0       => 1
| 1       => 2
| (n + 2) => a (n + 1) + a n

-- Statement to prove that a 4 = 8 (in Lean, the sequence is zero-indexed, so a 4 is a_5)
theorem sequence_a_5 : a 4 = 8 :=
  by
    sorry

end sequence_a_5_l212_212764


namespace problem_l212_212865

noncomputable def f : ℕ → ℕ := sorry
noncomputable def g : ℕ → ℕ := sorry

theorem problem (surj_f : ∀ y, ∃ x, f x = y) 
                (inj_g : ∀ x1 x2, g x1 = g x2 → x1 = x2)
                (f_ge_g : ∀ n, f n ≥ g n) :
  ∀ n, f n = g n := 
by 
  sorry

end problem_l212_212865


namespace diagonal_length_of_octagon_l212_212195

theorem diagonal_length_of_octagon 
  (r : ℝ) (s : ℝ) (has_symmetry_axes : ℕ) 
  (inscribed : r = 6) (side_length : s = 5) 
  (symmetry_condition : has_symmetry_axes = 4) : 
  ∃ (d : ℝ), d = 2 * Real.sqrt 40 := 
by 
  sorry

end diagonal_length_of_octagon_l212_212195


namespace circle_inscribed_in_square_area_l212_212708

theorem circle_inscribed_in_square_area :
  ∀ (x y : ℝ) (h : 2 * x^2 + 2 * y^2 - 8 * x - 12 * y + 24 = 0),
  ∃ side : ℝ, 4 * (side^2) = 16 :=
by
  sorry

end circle_inscribed_in_square_area_l212_212708


namespace cubic_roots_identity_l212_212290

theorem cubic_roots_identity (x1 x2 p q : ℝ) 
  (h1 : x1^2 + p * x1 + q = 0) 
  (h2 : x2^2 + p * x2 + q = 0) :
  (x1^3 + x2^3 = 3 * p * q - p^3) ∧ 
  (x1^3 - x2^3 = (p^2 - q) * Real.sqrt (p^2 - 4 * q) ∨ 
   x1^3 - x2^3 = -(p^2 - q) * Real.sqrt (p^2 - 4 * q)) :=
by
  sorry

end cubic_roots_identity_l212_212290


namespace melody_initial_food_l212_212429

-- Conditions
variable (dogs : ℕ) (food_per_meal : ℚ) (meals_per_day : ℕ) (days_in_week : ℕ) (food_left : ℚ)
variable (initial_food : ℚ)

-- Values given in the problem statement
axiom h_dogs : dogs = 3
axiom h_food_per_meal : food_per_meal = 1/2
axiom h_meals_per_day : meals_per_day = 2
axiom h_days_in_week : days_in_week = 7
axiom h_food_left : food_left = 9

-- Theorem to prove
theorem melody_initial_food : initial_food = 30 :=
  sorry

end melody_initial_food_l212_212429


namespace landscape_breadth_l212_212998

theorem landscape_breadth (L B : ℕ) (h1 : B = 8 * L)
  (h2 : 3200 = 1 / 9 * (L * B))
  (h3 : B * B = 28800) :
  B = 480 := by
  sorry

end landscape_breadth_l212_212998


namespace students_exceed_pets_l212_212652

-- Defining the conditions
def num_students_per_classroom := 25
def num_rabbits_per_classroom := 3
def num_guinea_pigs_per_classroom := 3
def num_classrooms := 5

-- Main theorem to prove
theorem students_exceed_pets:
  let total_students := num_students_per_classroom * num_classrooms
  let total_rabbits := num_rabbits_per_classroom * num_classrooms
  let total_guinea_pigs := num_guinea_pigs_per_classroom * num_classrooms
  let total_pets := total_rabbits + total_guinea_pigs
  total_students - total_pets = 95 :=
by 
  sorry

end students_exceed_pets_l212_212652


namespace arithmetic_sequence_common_difference_l212_212388

theorem arithmetic_sequence_common_difference (a : ℕ → ℝ) (S : ℕ → ℝ) (d : ℝ) 
    (h1 : ∀ n, S n = (n * (2 * a 1 + (n - 1) * d)) / 2)
    (h2 : (S 2017) / 2017 - (S 17) / 17 = 100) :
    d = 1/10 := 
by sorry

end arithmetic_sequence_common_difference_l212_212388


namespace distribute_items_l212_212412

open Nat

def g (n k : ℕ) : ℕ :=
  -- This is a placeholder for the actual function definition
  sorry

theorem distribute_items (n k : ℕ) (h : n ≥ k ∧ k ≥ 2) :
  g (n + 1) k = k * g n (k - 1) + k * g n k :=
by
  sorry

end distribute_items_l212_212412


namespace grace_have_30_pastries_l212_212217

theorem grace_have_30_pastries (F : ℕ) :
  (2 * (F + 8) + F + (F + 13) = 97) → (F + 13 = 30) :=
by
  sorry

end grace_have_30_pastries_l212_212217


namespace first_term_of_sequence_l212_212605

theorem first_term_of_sequence (S : ℕ → ℕ) (a : ℕ → ℕ) 
  (hS : ∀ n, S n = n^2 + 1) : a 1 = 2 := by
  sorry

end first_term_of_sequence_l212_212605


namespace min_variance_l212_212639

/--
Given a sample x, 1, y, 5 with an average of 2,
prove that the minimum value of the variance of this sample is 3.
-/
theorem min_variance (x y : ℝ) 
  (h_avg : (x + 1 + y + 5) / 4 = 2) :
  3 ≤ (1 / 4) * ((x - 2) ^ 2 + (y - 2) ^ 2 + (1 - 2) ^ 2 + (5 - 2) ^ 2) :=
sorry

end min_variance_l212_212639


namespace final_replacement_weight_l212_212942

theorem final_replacement_weight (W : ℝ) (a b c d e : ℝ) 
  (h1 : a = W / 10)
  (h2 : b = (W - 70 + e) / 10)
  (h3 : b - a = 4)
  (h4 : c = (W - 70 + e - 110 + d) / 10)
  (h5 : c - b = -2)
  (h6 : d = (W - 70 + e - 110 + d + 140 - 90) / 10)
  (h7 : d - c = 5)
  : e = 110 ∧ d = 90 ∧ 140 = e + 50 := sorry

end final_replacement_weight_l212_212942


namespace probability_of_green_apples_l212_212855

def total_apples : ℕ := 8
def red_apples : ℕ := 5
def green_apples : ℕ := 3
def apples_chosen : ℕ := 3
noncomputable def binomial (n k : ℕ) : ℕ := Nat.choose n k

theorem probability_of_green_apples :
  (binomial green_apples apples_chosen : ℚ) / (binomial total_apples apples_chosen : ℚ) = 1 / 56 :=
  sorry

end probability_of_green_apples_l212_212855


namespace Black_Queen_thought_Black_King_asleep_l212_212286

theorem Black_Queen_thought_Black_King_asleep (BK_awake : Prop) (BQ_awake : Prop) :
  (∃ t : ℕ, t = 10 * 60 + 55 → 
  ∀ (BK : Prop) (BQ : Prop),
    ((BK_awake ↔ ¬BK) ∧ (BQ_awake ↔ ¬BQ)) ∧
    (BK → BQ → BQ_awake) ∧
    (¬BK → ¬BQ → BK_awake)) →
  ((BQ ↔ BK) ∧ (BQ_awake ↔ ¬BQ)) →
  (∃ (BQ_thought : Prop), BQ_thought ↔ BK) := 
sorry

end Black_Queen_thought_Black_King_asleep_l212_212286


namespace number_of_four_digit_numbers_l212_212746

theorem number_of_four_digit_numbers : 
  (9999 - 1000 + 1) = 9000 := 
by 
  sorry 

end number_of_four_digit_numbers_l212_212746


namespace kostya_initially_planted_l212_212288

def bulbs_after_planting (n : ℕ) (stages : ℕ) : ℕ :=
  match stages with
  | 0 => n
  | k + 1 => 2 * bulbs_after_planting n k - 1

theorem kostya_initially_planted (n : ℕ) (stages : ℕ) :
  bulbs_after_planting n stages = 113 → n = 15 := 
sorry

end kostya_initially_planted_l212_212288


namespace people_off_second_eq_8_l212_212784

-- Initial number of people on the bus
def initial_people := 50

-- People who got off at the first stop
def people_off_first := 15

-- People who got on at the second stop
def people_on_second := 2

-- People who got off at the second stop (unknown, let's call it x)
variable (x : ℕ)

-- People who got off at the third stop
def people_off_third := 4

-- People who got on at the third stop
def people_on_third := 3

-- Number of people on the bus after the third stop
def people_after_third := 28

-- Equation formed by given conditions
def equation := initial_people - people_off_first - x + people_on_second - people_off_third + people_on_third = people_after_third

-- Goal: Prove the equation with given conditions results in x = 8
theorem people_off_second_eq_8 : equation x → x = 8 := by
  sorry

end people_off_second_eq_8_l212_212784


namespace soda_difference_l212_212620

def regular_soda : ℕ := 67
def diet_soda : ℕ := 9

theorem soda_difference : regular_soda - diet_soda = 58 := 
  by
  sorry

end soda_difference_l212_212620


namespace advertisement_length_l212_212204

noncomputable def movie_length : ℕ := 90
noncomputable def replay_times : ℕ := 6
noncomputable def operation_time : ℕ := 660

theorem advertisement_length : ∃ A : ℕ, 90 * replay_times + 6 * A = operation_time ∧ A = 20 :=
by
  use 20
  sorry

end advertisement_length_l212_212204


namespace cost_of_french_bread_is_correct_l212_212597

noncomputable def cost_of_sandwiches := 2 * 7.75
noncomputable def cost_of_salami := 4.00
noncomputable def cost_of_brie := 3 * cost_of_salami
noncomputable def cost_of_olives := 10.00 * (1/4)
noncomputable def cost_of_feta := 8.00 * (1/2)
noncomputable def total_cost_of_items := cost_of_sandwiches + cost_of_salami + cost_of_brie + cost_of_olives + cost_of_feta
noncomputable def total_spent := 40.00
noncomputable def cost_of_french_bread := total_spent - total_cost_of_items

theorem cost_of_french_bread_is_correct :
  cost_of_french_bread = 2.00 :=
by
  sorry

end cost_of_french_bread_is_correct_l212_212597


namespace ratio_of_books_to_pens_l212_212476

theorem ratio_of_books_to_pens (total_stationery : ℕ) (books : ℕ) (pens : ℕ) 
    (h1 : total_stationery = 400) (h2 : books = 280) (h3 : pens = total_stationery - books) : 
    books / (Nat.gcd books pens) = 7 ∧ pens / (Nat.gcd books pens) = 3 := 
by 
  -- proof steps would go here
  sorry

end ratio_of_books_to_pens_l212_212476


namespace arithmetic_mean_of_scores_l212_212748

theorem arithmetic_mean_of_scores :
  let s1 := 85
  let s2 := 94
  let s3 := 87
  let s4 := 93
  let s5 := 95
  let s6 := 88
  let s7 := 90
  (s1 + s2 + s3 + s4 + s5 + s6 + s7) / 7 = 90.2857142857 :=
by
  sorry

end arithmetic_mean_of_scores_l212_212748


namespace calc_expression_l212_212490

theorem calc_expression : 2 * 0 * 1 + 1 = 1 :=
by
  sorry

end calc_expression_l212_212490


namespace solution_set_inequality_l212_212634

noncomputable def f : ℝ → ℝ := sorry

variable {f : ℝ → ℝ}
variable (hf_diff : Differentiable ℝ f)
variable (hf_ineq : ∀ x, f x > deriv f x)
variable (hf_zero : f 0 = 2)

theorem solution_set_inequality : {x : ℝ | f x < 2 * Real.exp x} = {x | 0 < x} :=
by
  sorry

end solution_set_inequality_l212_212634


namespace not_p_and_p_l212_212949

theorem not_p_and_p (p : Prop) : ¬ (p ∧ ¬ p) :=
by 
  sorry

end not_p_and_p_l212_212949


namespace trees_planted_l212_212568

theorem trees_planted (yard_length : ℕ) (distance_between_trees : ℕ) (n_trees : ℕ) 
  (h1 : yard_length = 434) 
  (h2 : distance_between_trees = 14) 
  (h3 : n_trees = yard_length / distance_between_trees + 1) : 
  n_trees = 32 :=
by
  sorry

end trees_planted_l212_212568


namespace p_or_q_not_necessarily_true_l212_212250

theorem p_or_q_not_necessarily_true (p q : Prop) (hnp : ¬p) (hpq : ¬(p ∧ q)) : ¬(p ∨ q) ∨ (p ∨ q) :=
by
  sorry

end p_or_q_not_necessarily_true_l212_212250


namespace deepak_present_age_l212_212320

/-- Let Rahul and Deepak's current ages be 4x and 3x respectively
  Given that:
  1. The ratio between Rahul and Deepak's ages is 4:3
  2. After 6 years, Rahul's age will be 26 years
  Prove that Deepak's present age is 15 years.
-/
theorem deepak_present_age (x : ℕ) (hx : 4 * x + 6 = 26) : 3 * x = 15 :=
by
  sorry

end deepak_present_age_l212_212320


namespace modulo_remainder_even_l212_212261

theorem modulo_remainder_even (k : ℕ) (h : 1 ≤ k ∧ k ≤ 2018) : 
  ((2019 - k)^12 + 2018) % 2019 = (k^12 + 2018) % 2019 := 
by
  sorry

end modulo_remainder_even_l212_212261


namespace proof_of_min_value_l212_212037

def constraints_on_powers (a b c d : ℝ) : Prop :=
  a^4 + b^4 + c^4 + d^4 = 16

noncomputable def minimum_third_power_sum (a b c d : ℝ) : ℝ :=
  a^3 + b^3 + c^3 + d^3

theorem proof_of_min_value : 
  ∃ a b c d : ℝ, constraints_on_powers a b c d → ∃ min_val : ℝ, min_val = minimum_third_power_sum a b c d :=
sorry -- Further method to rigorously find the minimum value.

end proof_of_min_value_l212_212037


namespace find_a_l212_212736

-- Define the slopes of the lines and the condition that they are perpendicular.
def slope1 (a : ℝ) : ℝ := a
def slope2 (a : ℝ) : ℝ := a + 2

-- The main statement of our problem.
theorem find_a (a : ℝ) (h : slope1 a * slope2 a = -1) : a = -1 :=
sorry

end find_a_l212_212736


namespace find_x_for_which_f_f_x_eq_f_x_l212_212703

noncomputable def f (x : ℝ) : ℝ := x^2 - 5 * x + 6

theorem find_x_for_which_f_f_x_eq_f_x :
  {x : ℝ | f (f x) = f x} = {0, 2, 3, 5} :=
by
  sorry

end find_x_for_which_f_f_x_eq_f_x_l212_212703


namespace Buratino_math_problem_l212_212263

theorem Buratino_math_problem (x : ℝ) : 4 * x + 15 = 15 * x + 4 → x = 1 :=
by
  intro h
  sorry

end Buratino_math_problem_l212_212263


namespace melanie_turnips_l212_212458

theorem melanie_turnips (benny_turnips total_turnips melanie_turnips : ℕ) 
  (h1 : benny_turnips = 113) 
  (h2 : total_turnips = 252) 
  (h3 : total_turnips = benny_turnips + melanie_turnips) : 
  melanie_turnips = 139 :=
by
  sorry

end melanie_turnips_l212_212458


namespace trapezium_second_side_length_l212_212452

theorem trapezium_second_side_length (a b h : ℕ) (Area : ℕ) 
  (h_area : Area = (1 / 2 : ℚ) * (a + b) * h)
  (ha : a = 20) (hh : h = 12) (hA : Area = 228) : b = 18 := by
  sorry

end trapezium_second_side_length_l212_212452


namespace log_product_eq_one_l212_212460

noncomputable def log_base (b a : ℝ) := Real.log a / Real.log b

theorem log_product_eq_one :
  log_base 2 3 * log_base 9 4 = 1 := 
by {
  sorry
}

end log_product_eq_one_l212_212460


namespace fraction_is_perfect_square_l212_212111

theorem fraction_is_perfect_square (a b : ℕ) (hpos_a : 0 < a) (hpos_b : 0 < b) (hdiv : (ab + 1) ∣ (a^2 + b^2)) : 
  ∃ k : ℕ, k^2 = (a^2 + b^2) / (ab + 1) :=
sorry

end fraction_is_perfect_square_l212_212111


namespace train_speed_correct_l212_212235

def train_length : ℝ := 1500
def crossing_time : ℝ := 15
def correct_speed : ℝ := 100

theorem train_speed_correct : (train_length / crossing_time) = correct_speed := by 
  sorry

end train_speed_correct_l212_212235


namespace percent_of_200_is_400_when_whole_is_50_l212_212348

theorem percent_of_200_is_400_when_whole_is_50 (Part Whole : ℕ) (hPart : Part = 200) (hWhole : Whole = 50) :
  (Part / Whole) * 100 = 400 :=
by {
  -- Proof steps go here.
  sorry
}

end percent_of_200_is_400_when_whole_is_50_l212_212348


namespace value_of_expression_l212_212582

theorem value_of_expression 
  (x : ℝ) 
  (h : 7 * x^2 + 6 = 5 * x + 11) 
  : (8 * x - 5)^2 = (2865 - 120 * Real.sqrt 165) / 49 := 
by 
  sorry

end value_of_expression_l212_212582


namespace tenth_term_arithmetic_seq_l212_212090

theorem tenth_term_arithmetic_seq : 
  ∀ (first_term common_diff : ℤ) (n : ℕ), 
    first_term = 10 → common_diff = -2 → n = 10 → 
    (first_term + (n - 1) * common_diff) = -8 :=
by
  sorry

end tenth_term_arithmetic_seq_l212_212090


namespace S_3n_plus_1_l212_212027

noncomputable def S : ℕ → ℝ := sorry  -- S_n is the sum of the first n terms of the sequence {a_n}
noncomputable def a : ℕ → ℝ := sorry  -- Sequence {a_n}

-- Given conditions
axiom S3 : S 3 = 1
axiom S4 : S 4 = 11
axiom a_recurrence (n : ℕ) : a (n + 3) = 2 * a n

-- Define S_{3n+1} in terms of n
theorem S_3n_plus_1 (n : ℕ) : S (3 * n + 1) = 3 * 2^(n+1) - 1 :=
sorry

end S_3n_plus_1_l212_212027


namespace greatest_number_same_remainder_l212_212882

theorem greatest_number_same_remainder (d : ℕ) :
  d ∣ (57 - 25) ∧ d ∣ (105 - 57) ∧ d ∣ (105 - 25) → d ≤ 16 :=
by
  sorry

end greatest_number_same_remainder_l212_212882


namespace sticks_difference_l212_212141

-- Definitions of the conditions
def d := 14  -- number of sticks Dave picked up
def a := 9   -- number of sticks Amy picked up
def total := 50  -- initial total number of sticks in the yard

-- The proof problem statement
theorem sticks_difference : (d + a) - (total - (d + a)) = 4 :=
by
  sorry

end sticks_difference_l212_212141


namespace original_average_weight_l212_212876

theorem original_average_weight 
  (W : ℝ)
  (h1 : 7 * W + 110 + 60 = 9 * 78) : 
  W = 76 := 
by
  sorry

end original_average_weight_l212_212876


namespace more_boys_after_initial_l212_212887

theorem more_boys_after_initial (X Y Z : ℕ) (hX : X = 22) (hY : Y = 35) : Z = Y - X :=
by
  sorry

end more_boys_after_initial_l212_212887


namespace find_quotient_l212_212863

theorem find_quotient (divisor remainder dividend : ℕ) (h_divisor : divisor = 24) (h_remainder : remainder = 5) (h_dividend : dividend = 1565) : 
  (dividend - remainder) / divisor = 65 :=
by
  sorry

end find_quotient_l212_212863


namespace angle_ABC_30_degrees_l212_212033

theorem angle_ABC_30_degrees 
    (angle_CBD : ℝ)
    (angle_ABD : ℝ)
    (angle_ABC : ℝ)
    (h1 : angle_CBD = 90)
    (h2 : angle_ABC + angle_ABD + angle_CBD = 180)
    (h3 : angle_ABD = 60) :
    angle_ABC = 30 :=
by
  sorry

end angle_ABC_30_degrees_l212_212033


namespace find_second_number_l212_212686

theorem find_second_number (A B : ℝ) (h1 : A = 6400) (h2 : 0.05 * A = 0.2 * B + 190) : B = 650 :=
by
  sorry

end find_second_number_l212_212686


namespace nesbitts_inequality_l212_212277

variable (a b c : ℝ)

theorem nesbitts_inequality (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a / (b + c) + b / (c + a) + c / (a + b)) >= 3 / 2 := 
sorry

end nesbitts_inequality_l212_212277


namespace remainder_3_pow_19_mod_10_l212_212122

theorem remainder_3_pow_19_mod_10 : (3 ^ 19) % 10 = 7 := by
  sorry

end remainder_3_pow_19_mod_10_l212_212122


namespace main_inequality_l212_212100

noncomputable def b (c : ℝ) : ℝ := (1 + c) / (2 + c)

def f (c : ℝ) (x : ℝ) : ℝ := sorry

lemma f_continuous (c : ℝ) (h_c : 0 < c) : Continuous (f c) := sorry

lemma condition1 (c : ℝ) (h_c : 0 < c) (x : ℝ) (h_x : 0 ≤ x ∧ x ≤ 1/2) : 
  b c * f c (2 * x) = f c x := sorry

lemma condition2 (c : ℝ) (h_c : 0 < c) (x : ℝ) (h_x : 1/2 ≤ x ∧ x ≤ 1) : 
  f c x = b c + (1 - b c) * f c (2 * x - 1) := sorry

theorem main_inequality (c : ℝ) (h_c : 0 < c) : 
  ∀ x : ℝ, (0 < x ∧ x < 1) → (0 < f c x - x ∧ f c x - x < c) := sorry

end main_inequality_l212_212100


namespace total_stones_l212_212054

theorem total_stones (sent_away kept total : ℕ) (h1 : sent_away = 63) (h2 : kept = 15) (h3 : total = sent_away + kept) : total = 78 :=
by
  sorry

end total_stones_l212_212054


namespace number_is_three_l212_212550

theorem number_is_three (n : ℝ) (h : 4 * n - 7 = 5) : n = 3 :=
by sorry

end number_is_three_l212_212550


namespace roots_numerically_equal_opposite_signs_l212_212563

theorem roots_numerically_equal_opposite_signs
  (a b c : ℝ) (k : ℝ)
  (h : (∃ x : ℝ, x^2 - (b+1) * x ≠ 0) →
    ∃ x : ℝ, x ≠ 0 ∧ x ∈ {x : ℝ | (k+2)*(x^2 - (b+1)*x) = (k-2)*((a+1)*x - c)} ∧ -x ∈ {x : ℝ | (k+2)*(x^2 - (b+1)*x) = (k-2)*((a+1)*x - c)}) :
  k = (-2 * (b - a)) / (b + a + 2) :=
by
  sorry

end roots_numerically_equal_opposite_signs_l212_212563


namespace tenth_term_in_sequence_l212_212935

def seq (n : ℕ) : ℚ :=
  (-1) ^ (n + 1) * ((2 * n - 1) / (n ^ 2 + 1))

theorem tenth_term_in_sequence :
  seq 10 = -19 / 101 :=
by
  -- Proof omitted
  sorry

end tenth_term_in_sequence_l212_212935


namespace total_pages_in_book_l212_212854

-- Given conditions
def pages_first_chapter : ℕ := 13
def pages_second_chapter : ℕ := 68

-- The theorem to prove the total number of pages in the book
theorem total_pages_in_book :
  pages_first_chapter + pages_second_chapter = 81 := by
  sorry

end total_pages_in_book_l212_212854


namespace height_of_spherical_caps_l212_212005

theorem height_of_spherical_caps
  (r q : ℝ)
  (m₁ m₂ m₃ m₄ : ℝ)
  (h1 : m₂ = m₁ * q)
  (h2 : m₃ = m₁ * q^2)
  (h3 : m₄ = m₁ * q^3)
  (h4 : m₁ + m₂ + m₃ + m₄ = 2 * r) :
  m₁ = 2 * r * (q - 1) / (q^4 - 1) := 
sorry

end height_of_spherical_caps_l212_212005


namespace greatest_int_radius_of_circle_l212_212402

theorem greatest_int_radius_of_circle (r : ℝ) (A : ℝ) :
  (A < 200 * Real.pi) ∧ (A = Real.pi * r^2) →
  ∃k : ℕ, (k : ℝ) = 14 ∧ ∀n : ℕ, (n : ℝ) = r → n ≤ k := by
  sorry

end greatest_int_radius_of_circle_l212_212402


namespace total_population_correct_l212_212375

-- Given conditions
def number_of_cities : ℕ := 25
def average_population : ℕ := 3800

-- Statement to prove
theorem total_population_correct : number_of_cities * average_population = 95000 :=
by
  sorry

end total_population_correct_l212_212375


namespace hyperbola_equation_of_midpoint_l212_212751

-- Define the hyperbola E
def hyperbola (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 - y^2 / b^2 = 1

-- Given conditions
variables (a b : ℝ) (hapos : a > 0) (hbpos : b > 0)
variables (F : ℝ × ℝ) (hF : F = (-2, 0))
variables (M : ℝ × ℝ) (hM : M = (-3, -1))

-- The statement requiring proof
theorem hyperbola_equation_of_midpoint (hE : hyperbola a b (-2) 0) 
(hFocus : a^2 + b^2 = 4) : 
  (∃ a' b', a' = 3 ∧ b' = 1 ∧ hyperbola a' b' (-3) (-1)) :=
sorry

end hyperbola_equation_of_midpoint_l212_212751


namespace belfried_payroll_l212_212641

noncomputable def tax_paid (payroll : ℝ) : ℝ :=
  if payroll < 200000 then 0 else 0.002 * (payroll - 200000)

theorem belfried_payroll (payroll : ℝ) (h : tax_paid payroll = 400) : payroll = 400000 :=
by
  sorry

end belfried_payroll_l212_212641


namespace sqrt_fraction_identity_l212_212775

theorem sqrt_fraction_identity (n : ℕ) (h : n > 0) : 
    Real.sqrt ((1 : ℝ) / n - (1 : ℝ) / (n * n)) = Real.sqrt (n - 1) / n :=
by
  sorry

end sqrt_fraction_identity_l212_212775


namespace find_trajectory_l212_212532

noncomputable def trajectory_equation (x y : ℝ) : Prop :=
  (y - 1) * (y + 1) / ((x + 1) * (x - 1)) = -1 / 3

theorem find_trajectory (x y : ℝ) (h : x ≠ 1 ∧ x ≠ -1) :
  trajectory_equation x y → x^2 + 3 * y^2 = 4 :=
by
  sorry

end find_trajectory_l212_212532


namespace remainder_43_pow_97_pow_5_plus_109_mod_163_l212_212233

theorem remainder_43_pow_97_pow_5_plus_109_mod_163 :
    (43 ^ (97 ^ 5) + 109) % 163 = 50 :=
by
  sorry

end remainder_43_pow_97_pow_5_plus_109_mod_163_l212_212233


namespace ferry_travel_time_l212_212857

theorem ferry_travel_time:
  ∀ (v_P v_Q : ℝ) (d_P d_Q : ℝ) (t_P t_Q : ℝ),
    v_P = 8 →
    v_Q = v_P + 1 →
    d_Q = 3 * d_P →
    t_Q = t_P + 5 →
    d_P = v_P * t_P →
    d_Q = v_Q * t_Q →
    t_P = 3 := by
  sorry

end ferry_travel_time_l212_212857


namespace find_a_l212_212035

theorem find_a (a : ℝ) : 
  (∃ x y : ℝ, x - 2 * a * y - 3 = 0 ∧ x^2 + y^2 - 2 * x + 2 * y - 3 = 0) → a = 1 :=
by
  sorry

end find_a_l212_212035


namespace swimmer_distance_l212_212856

noncomputable def effective_speed := 4.4 - 2.5
noncomputable def time := 3.684210526315789
noncomputable def distance := effective_speed * time

theorem swimmer_distance :
  distance = 7 := by
  sorry

end swimmer_distance_l212_212856


namespace denis_neighbors_l212_212153

theorem denis_neighbors :
  ∃ (positions : ℕ → String), 
  (positions 1 = "Borya") ∧ 
  (positions 2 ≠ "Gena") ∧ (positions 2 = "Vera" → positions 3 = "Anya" ∨ positions 3 = "Gena") ∧ 
  (positions 3 ≠ "Borya") ∧ (positions 3 ≠ "Gena") ∧ 
  (positions 5 ≠ "Borya") ∧ (positions 5 ≠ "Anya") → 
  (positions 4 = "Denis" → 
    (positions 3 = "Anya" ∨ positions 5 = "Gena") ∧ 
    (positions 3 ≠ "Gena" ∨ positions 5 = "Anya")) :=
by
  sorry

end denis_neighbors_l212_212153


namespace probability_multiple_4_or_15_l212_212798

-- Definitions of natural number range and a set of multiples
def first_30_nat_numbers : Finset ℕ := Finset.range 30
def multiples_of (n : ℕ) (s : Finset ℕ) : Finset ℕ := s.filter (λ x => x % n = 0)

-- Conditions
def multiples_of_4 := multiples_of 4 first_30_nat_numbers
def multiples_of_15 := multiples_of 15 first_30_nat_numbers

-- Proof that probability of selecting a multiple of 4 or 15 is 3 / 10
theorem probability_multiple_4_or_15 : 
  let favorable_outcomes := (multiples_of_4 ∪ multiples_of_15).card
  let total_outcomes := first_30_nat_numbers.card
  (favorable_outcomes : ℚ) / total_outcomes = 3 / 10 :=
by
  -- correct answer based on the computation
  sorry

end probability_multiple_4_or_15_l212_212798


namespace factorize_polynomial_l212_212827

theorem factorize_polynomial (a b : ℝ) : 
  a^3 * b - 9 * a * b = a * b * (a + 3) * (a - 3) :=
by sorry

end factorize_polynomial_l212_212827


namespace cos_sum_proof_l212_212901

theorem cos_sum_proof (x : ℝ) (h : Real.cos (x - (Real.pi / 6)) = Real.sqrt 3 / 3) :
  Real.cos x + Real.cos (x - Real.pi / 3) = 1 := 
sorry

end cos_sum_proof_l212_212901


namespace boxes_per_case_l212_212697

/-- Let's define the variables for the problem.
    We are given that Shirley sold 10 boxes of trefoils,
    and she needs to deliver 5 cases of boxes. --/
def total_boxes : ℕ := 10
def number_of_cases : ℕ := 5

/-- We need to prove that the number of boxes in each case is 2. --/
theorem boxes_per_case :
  total_boxes / number_of_cases = 2 :=
by
  -- Definition step where we specify the calculation
  unfold total_boxes number_of_cases
  -- The problem requires a division operation
  norm_num
  -- The result should be correct according to the solution steps
  done

end boxes_per_case_l212_212697


namespace gray_region_area_is_96pi_l212_212203

noncomputable def smaller_circle_diameter : ℝ := 4

noncomputable def smaller_circle_radius : ℝ := smaller_circle_diameter / 2

noncomputable def larger_circle_radius : ℝ := 5 * smaller_circle_radius

noncomputable def area_of_larger_circle : ℝ := Real.pi * (larger_circle_radius ^ 2)

noncomputable def area_of_smaller_circle : ℝ := Real.pi * (smaller_circle_radius ^ 2)

noncomputable def area_of_gray_region : ℝ := area_of_larger_circle - area_of_smaller_circle

theorem gray_region_area_is_96pi : area_of_gray_region = 96 * Real.pi := by
  sorry

end gray_region_area_is_96pi_l212_212203


namespace number_of_chlorine_atoms_l212_212978

def molecular_weight_of_aluminum : ℝ := 26.98
def molecular_weight_of_chlorine : ℝ := 35.45
def molecular_weight_of_compound : ℝ := 132.0

theorem number_of_chlorine_atoms :
  ∃ n : ℕ, molecular_weight_of_compound = molecular_weight_of_aluminum + n * molecular_weight_of_chlorine ∧ n = 3 :=
by
  sorry

end number_of_chlorine_atoms_l212_212978


namespace major_axis_length_l212_212316

-- Definitions of the given conditions
structure Ellipse :=
  (focus1 focus2 : ℝ × ℝ)
  (tangent_to_x_axis : Bool)

noncomputable def length_of_major_axis (E : Ellipse) : ℝ :=
  let (x1, y1) := E.focus1
  let (x2, y2) := E.focus2
  Real.sqrt ((x2 - x1) ^ 2 + (y2 + y1) ^ 2)

-- The theorem we want to prove given the conditions
theorem major_axis_length (E : Ellipse)
  (h1 : E.focus1 = (9, 20))
  (h2 : E.focus2 = (49, 55))
  (h3 : E.tangent_to_x_axis = true):
  length_of_major_axis E = 85 :=
by
  sorry

end major_axis_length_l212_212316


namespace average_of_second_set_of_two_numbers_l212_212823

theorem average_of_second_set_of_two_numbers
  (S : ℝ)
  (avg1 avg2 avg3 : ℝ)
  (h1 : S = 6 * 3.95)
  (h2 : avg1 = 3.4)
  (h3 : avg3 = 4.6) :
  (S - (2 * avg1) - (2 * avg3)) / 2 = 3.85 :=
by
  sorry

end average_of_second_set_of_two_numbers_l212_212823


namespace count_young_diagrams_4_count_young_diagrams_5_count_young_diagrams_6_count_young_diagrams_7_l212_212015

-- Define the weight 's'.
variable (s : ℕ)

-- Define the function that counts the number of Young diagrams for a given weight.
def countYoungDiagrams (s : ℕ) : ℕ :=
  -- Placeholder for actual implementation of counting Young diagrams.
  sorry

-- Prove that the count of Young diagrams for s = 4 is 5
theorem count_young_diagrams_4 : countYoungDiagrams 4 = 5 :=
by sorry

-- Prove that the count of Young diagrams for s = 5 is 7
theorem count_young_diagrams_5 : countYoungDiagrams 5 = 7 :=
by sorry

-- Prove that the count of Young diagrams for s = 6 is 11
theorem count_young_diagrams_6 : countYoungDiagrams 6 = 11 :=
by sorry

-- Prove that the count of Young diagrams for s = 7 is 15
theorem count_young_diagrams_7 : countYoungDiagrams 7 = 15 :=
by sorry

end count_young_diagrams_4_count_young_diagrams_5_count_young_diagrams_6_count_young_diagrams_7_l212_212015


namespace value_of_2x_plus_3y_l212_212128

theorem value_of_2x_plus_3y {x y : ℝ} (h1 : 2 * x - 1 = 5) (h2 : 3 * y + 2 = 17) : 2 * x + 3 * y = 21 :=
by
  sorry

end value_of_2x_plus_3y_l212_212128


namespace original_faculty_members_correct_l212_212841

noncomputable def original_faculty_members : ℝ := 282

theorem original_faculty_members_correct:
  ∃ F : ℝ, (0.6375 * F = 180) ∧ (F = original_faculty_members) :=
by
  sorry

end original_faculty_members_correct_l212_212841


namespace rectangle_area_error_percent_l212_212304

theorem rectangle_area_error_percent 
  (L W : ℝ)
  (hL: L > 0)
  (hW: W > 0) :
  let original_area := L * W
  let measured_length := 1.06 * L
  let measured_width := 0.95 * W
  let measured_area := measured_length * measured_width
  let error := measured_area - original_area
  let error_percent := (error / original_area) * 100
  error_percent = 0.7 := by
  let original_area := L * W
  let measured_length := 1.06 * L
  let measured_width := 0.95 * W
  let measured_area := measured_length * measured_width
  let error := measured_area - original_area
  let error_percent := (error / original_area) * 100
  sorry

end rectangle_area_error_percent_l212_212304


namespace omar_total_time_l212_212707

-- Conditions
def lap_distance : ℝ := 400
def first_segment_distance : ℝ := 200
def second_segment_distance : ℝ := 200
def speed_first_segment : ℝ := 6
def speed_second_segment : ℝ := 4
def number_of_laps : ℝ := 7

-- Correct answer we want to prove
def total_time_proven : ℝ := 9 * 60 + 23 -- in seconds

-- Theorem statement claiming total time is 9 minutes and 23 seconds
theorem omar_total_time :
  let time_first_segment := first_segment_distance / speed_first_segment
  let time_second_segment := second_segment_distance / speed_second_segment
  let single_lap_time := time_first_segment + time_second_segment
  let total_time := number_of_laps * single_lap_time
  total_time = total_time_proven := sorry

end omar_total_time_l212_212707


namespace problem_inequality_solution_problem_prove_inequality_l212_212937

-- Function definition for f(x)
def f (x : ℝ) := |2 * x - 3| + |2 * x + 3|

-- Problem 1: Prove the solution set for the inequality f(x) ≤ 8
theorem problem_inequality_solution (x : ℝ) : f x ≤ 8 ↔ -2 ≤ x ∧ x ≤ 2 :=
sorry

-- Problem 2: Prove a + 2b + 3c ≥ 9 given conditions
theorem problem_prove_inequality (a b c : ℝ) (M : ℝ) (h1 : M = 6)
  (h2 : 0 < a) (h3 : 0 < b) (h4 : 0 < c) (h5 : 1 / a + 1 / (2 * b) + 1 / (3 * c) = M / 6) :
  a + 2 * b + 3 * c ≥ 9 :=
sorry

end problem_inequality_solution_problem_prove_inequality_l212_212937


namespace garage_sale_items_count_l212_212672

theorem garage_sale_items_count (n_high n_low: ℕ) :
  n_high = 17 ∧ n_low = 24 → total_items = 40 :=
by
  let n_high: ℕ := 17
  let n_low: ℕ := 24
  let total_items: ℕ := (n_high - 1) + (n_low - 1) + 1
  sorry

end garage_sale_items_count_l212_212672


namespace matrix_exponentiation_l212_212604

theorem matrix_exponentiation (a n : ℕ) (M : Matrix (Fin 3) (Fin 3) ℕ) (N : Matrix (Fin 3) (Fin 3) ℕ) :
  (M^n = N) →
  M = ![
    ![1, 3, a],
    ![0, 1, 5],
    ![0, 0, 1]
  ] →
  N = ![
    ![1, 27, 3060],
    ![0, 1, 45],
    ![0, 0, 1]
  ] →
  a + n = 289 :=
by
  intros h1 h2 h3
  sorry

end matrix_exponentiation_l212_212604


namespace min_handshakes_35_people_l212_212397

theorem min_handshakes_35_people (n : ℕ) (h1 : n = 35) (h2 : ∀ p : ℕ, p < n → p ≥ 3) : ∃ m : ℕ, m = 51 :=
by
  sorry

end min_handshakes_35_people_l212_212397


namespace problem_I_problem_II_l212_212319

-- Problem (I)
theorem problem_I (a : ℝ) (h : ∀ x : ℝ, x^2 - 3 * a * x + 9 > 0) : -2 ≤ a ∧ a ≤ 2 :=
sorry

-- Problem (II)
theorem problem_II (m : ℝ) 
  (h₁ : ∀ x : ℝ, x^2 + 2 * x - 8 < 0 → x - m > 0)
  (h₂ : ∃ x : ℝ, x^2 + 2 * x - 8 < 0) : m ≤ -4 :=
sorry

end problem_I_problem_II_l212_212319


namespace number_of_lucky_numbers_l212_212209

-- Defining the concept of sequence with even number of digit 8
def is_lucky (seq : List ℕ) : Prop :=
  seq.count 8 % 2 = 0

-- Define S(n) recursive formula
noncomputable def S : ℕ → ℝ
| 0 => 0
| n+1 => 4 * (1 - (1 / (2 ^ (n+1))))

theorem number_of_lucky_numbers (n : ℕ) :
  ∀ (seq : List ℕ), (seq.length ≤ n) → is_lucky seq → S n = 4 * (1 - 1 / (2 ^ n)) :=
sorry

end number_of_lucky_numbers_l212_212209


namespace base_b_of_200_has_5_digits_l212_212867

theorem base_b_of_200_has_5_digits : ∃ (b : ℕ), (b^4 ≤ 200) ∧ (200 < b^5) ∧ (b = 3) := by
  sorry

end base_b_of_200_has_5_digits_l212_212867


namespace sum_f_a_seq_positive_l212_212530

noncomputable def f (x : ℝ) : ℝ := sorry
axiom f_odd : ∀ x : ℝ, f (-x) = -f x
axiom f_monotone_decreasing_nonneg : ∀ ⦃x y : ℝ⦄, 0 ≤ x → x ≤ y → f y ≤ f x
axiom a_seq : ∀ n : ℕ, ℝ
axiom a_arithmetic : ∀ m n k : ℕ, m + k = 2 * n → a_seq m + a_seq k = 2 * a_seq n
axiom a3_neg : a_seq 3 < 0

theorem sum_f_a_seq_positive :
    f (a_seq 1) + 
    f (a_seq 2) + 
    f (a_seq 3) + 
    f (a_seq 4) + 
    f (a_seq 5) > 0 :=
sorry

end sum_f_a_seq_positive_l212_212530


namespace rajesh_monthly_savings_l212_212328

theorem rajesh_monthly_savings
  (salary : ℝ)
  (percentage_food : ℝ)
  (percentage_medicines : ℝ)
  (percentage_savings : ℝ)
  (amount_food : ℝ := percentage_food * salary)
  (amount_medicines : ℝ := percentage_medicines * salary)
  (remaining_amount : ℝ := salary - (amount_food + amount_medicines))
  (save_amount : ℝ := percentage_savings * remaining_amount)
  (H_salary : salary = 15000)
  (H_percentage_food : percentage_food = 0.40)
  (H_percentage_medicines : percentage_medicines = 0.20)
  (H_percentage_savings : percentage_savings = 0.60) :
  save_amount = 3600 :=
by
  sorry

end rajesh_monthly_savings_l212_212328


namespace determine_d_minus_b_l212_212431

theorem determine_d_minus_b 
  (a b c d : ℕ) 
  (h1 : a^5 = b^4)
  (h2 : c^3 = d^2)
  (h3 : c - a = 19) 
  (a_pos : 0 < a)
  (b_pos : 0 < b)
  (c_pos : 0 < c)
  (d_pos : 0 < d)
  : d - b = 757 := 
  sorry

end determine_d_minus_b_l212_212431


namespace water_in_maria_jar_after_200_days_l212_212459

def arithmetic_series_sum (a d n : ℕ) : ℕ :=
  (n * (2 * a + (n - 1) * d)) / 2

theorem water_in_maria_jar_after_200_days :
  let initial_volume_maria : ℕ := 1000
  let days : ℕ := 200
  let odd_days : ℕ := days / 2
  let even_days : ℕ := days / 2
  let volume_odd_transfer : ℕ := arithmetic_series_sum 1 2 odd_days
  let volume_even_transfer : ℕ := arithmetic_series_sum 2 2 even_days
  let net_transfer : ℕ := volume_odd_transfer - volume_even_transfer
  let final_volume_maria := initial_volume_maria + net_transfer
  final_volume_maria = 900 :=
by
  sorry

end water_in_maria_jar_after_200_days_l212_212459


namespace simplify_fraction_l212_212985

/-- Given the numbers 180 and 270, prove that 180 / 270 is equal to 2 / 3 -/
theorem simplify_fraction : (180 / 270 : ℚ) = 2 / 3 := 
sorry

end simplify_fraction_l212_212985


namespace transistor_count_2010_l212_212393

-- Define the known constants and conditions
def initial_transistors : ℕ := 2000000
def doubling_period : ℕ := 2
def years_elapsed : ℕ := 2010 - 1995
def number_of_doublings := years_elapsed / doubling_period -- we want floor division

-- The theorem statement we need to prove
theorem transistor_count_2010 : initial_transistors * 2^number_of_doublings = 256000000 := by
  sorry

end transistor_count_2010_l212_212393


namespace vasya_numbers_l212_212455

theorem vasya_numbers (x y : ℚ) (h1 : x + y = x * y) (h2 : x * y = x / y) :
  x = 1 / 2 ∧ y = -1 :=
sorry

end vasya_numbers_l212_212455


namespace problem_statement_l212_212336

noncomputable def p := Real.sqrt 3 + Real.sqrt 5 + Real.sqrt 7
noncomputable def q := -Real.sqrt 3 + Real.sqrt 5 + Real.sqrt 7
noncomputable def r := Real.sqrt 3 - Real.sqrt 5 + Real.sqrt 7
noncomputable def s := -Real.sqrt 3 - Real.sqrt 5 + Real.sqrt 7

theorem problem_statement :
  (1 / p + 1 / q + 1 / r + 1 / s)^2 = 112 / 3481 :=
sorry

end problem_statement_l212_212336


namespace smallest_possible_e_l212_212341

-- Define the polynomial with its roots and integer coefficients
def polynomial (x : ℝ) : ℝ := (x + 4) * (x - 6) * (x - 10) * (2 * x + 1)

-- Define e as the constant term
def e : ℝ := 200 -- based on the final expanded polynomial result

-- The theorem stating the smallest possible value of e
theorem smallest_possible_e : 
  ∃ (e : ℕ), e > 0 ∧ polynomial e = 200 := 
sorry

end smallest_possible_e_l212_212341


namespace determine_a_l212_212979

-- Define the function f as given in the conditions
def f (a : ℝ) (x : ℝ) : ℝ := a * x + 6

-- Formulate the proof statement
theorem determine_a (a : ℝ) (h : f a (-1) = 8) : a = -2 :=
by {
  sorry
}

end determine_a_l212_212979


namespace tan_alpha_eq_7_over_5_l212_212143

theorem tan_alpha_eq_7_over_5
  (α : ℝ)
  (h : Real.tan (α - π / 4) = 1 / 6) :
  Real.tan α = 7 / 5 :=
by
  sorry

end tan_alpha_eq_7_over_5_l212_212143


namespace part2_inequality_l212_212114

-- Define the function f and its conditions
def f (x : ℝ) : ℝ := |x - 1| + |x + 2|

-- The main theorem we want to prove
theorem part2_inequality (a b c : ℝ) (h : a^2 + 2 * b^2 + 3 * c^2 = 6) : 
  |a + 2 * b + 3 * c| ≤ 6 :=
by {
-- Proof goes here
sorry
}

end part2_inequality_l212_212114


namespace chord_constant_l212_212311

theorem chord_constant (
    d : ℝ
) : (∃ t : ℝ, (∀ A B : ℝ × ℝ,
    A.2 = A.1^3 ∧ B.2 = B.1^3 ∧ d = 1/2 ∧
    (C : ℝ × ℝ) = (0, d) ∧ 
    (∀ (AC BC: ℝ),
        AC = dist A C ∧
        BC = dist B C ∧
        t = (1 / (AC^2) + 1 / (BC^2))
    )) → t = 4) := 
sorry

end chord_constant_l212_212311


namespace inequality_is_linear_l212_212754

theorem inequality_is_linear (k : ℝ) (h1 : (|k| - 1) = 1) (h2 : (k + 2) ≠ 0) : k = 2 :=
sorry

end inequality_is_linear_l212_212754


namespace parallelogram_height_l212_212840

theorem parallelogram_height (A B H : ℝ) 
    (h₁ : A = 96) 
    (h₂ : B = 12) 
    (h₃ : A = B * H) :
  H = 8 := 
by {
  sorry
}

end parallelogram_height_l212_212840


namespace medal_allocation_l212_212536

-- Define the participants
inductive Participant
| Jiri
| Vit
| Ota

open Participant

-- Define the medals
inductive Medal
| Gold
| Silver
| Bronze

open Medal

-- Define a structure to capture each person's statement
structure Statements :=
  (Jiri : Prop)
  (Vit : Prop)
  (Ota : Prop)

-- Define the condition based on their statements
def statements (m : Participant → Medal) : Statements :=
  {
    Jiri := m Ota = Gold,
    Vit := m Ota = Silver,
    Ota := (m Ota ≠ Gold ∧ m Ota ≠ Silver)
  }

-- Define the condition for truth-telling and lying based on medals
def truths_and_lies (m : Participant → Medal) (s : Statements) : Prop :=
  (m Jiri = Gold → s.Jiri) ∧ (m Jiri = Bronze → ¬ s.Jiri) ∧
  (m Vit = Gold → s.Vit) ∧ (m Vit = Bronze → ¬ s.Vit) ∧
  (m Ota = Gold → s.Ota) ∧ (m Ota = Bronze → ¬ s.Ota)

-- Define the final theorem to be proven
theorem medal_allocation : 
  ∃ (m : Participant → Medal), 
    truths_and_lies m (statements m) ∧ 
    m Vit = Gold ∧ 
    m Ota = Silver ∧ 
    m Jiri = Bronze := 
sorry

end medal_allocation_l212_212536


namespace angela_age_in_fifteen_years_l212_212087

-- Condition 1: Angela is currently 3 times as old as Beth
def angela_age_three_times_beth (A B : ℕ) := A = 3 * B

-- Condition 2: Angela is half as old as Derek
def angela_half_derek (A D : ℕ) := A = D / 2

-- Condition 3: Twenty years ago, the sum of their ages was equal to Derek's current age
def sum_ages_twenty_years_ago (A B D : ℕ) := (A - 20) + (B - 20) + (D - 20) = D

-- Condition 4: In seven years, the difference in the square root of Angela's age and one-third of Beth's age is a quarter of Derek's age
def age_diff_seven_years (A B D : ℕ) := Real.sqrt (A + 7) - (B + 7) / 3 = D / 4

-- Define the main theorem to be proven
theorem angela_age_in_fifteen_years (A B D : ℕ) 
  (h1 : angela_age_three_times_beth A B)
  (h2 : angela_half_derek A D) 
  (h3 : sum_ages_twenty_years_ago A B D) 
  (h4 : age_diff_seven_years A B D) :
  A + 15 = 60 := 
  sorry

end angela_age_in_fifteen_years_l212_212087


namespace girls_on_debate_team_l212_212254

def number_of_students (groups: ℕ) (group_size: ℕ) : ℕ :=
  groups * group_size

def total_students_debate_team : ℕ :=
  number_of_students 8 9

def number_of_boys : ℕ := 26

def number_of_girls : ℕ :=
  total_students_debate_team - number_of_boys

theorem girls_on_debate_team :
  number_of_girls = 46 :=
by
  sorry

end girls_on_debate_team_l212_212254


namespace find_m_l212_212846

-- Define the hyperbola equation
def hyperbola1 (x y : ℝ) (m : ℝ) : Prop := (x^3 / m) - (y^2 / 3) = 1
def hyperbola2 (x y : ℝ) : Prop := (x^3 / 8) - (y^2 / 4) = 1

-- Define the condition for eccentricity equivalence
def same_eccentricity (m : ℝ) : Prop :=
  let e1_sq := 1 + (4 / 2^2)
  let e2_sq := 1 + (3 / m)
  e1_sq = e2_sq

-- The main theorem statement
theorem find_m (m : ℝ) : hyperbola1 x y m → hyperbola2 x y → same_eccentricity m → m = 6 :=
by
  -- Proof can be skipped with sorry to satisfy the statement-only requirement
  sorry

end find_m_l212_212846


namespace equalize_expenses_l212_212378

/-- Problem Statement:
Given the amount paid by LeRoy (A), Bernardo (B), and Carlos (C),
prove that the amount LeRoy must adjust to share the costs equally is (B + C - 2A) / 3.
-/
theorem equalize_expenses (A B C : ℝ) : 
  (B+C-2*A) / 3 = (A + B + C) / 3 - A :=
by
  sorry

end equalize_expenses_l212_212378


namespace a_plus_b_value_l212_212858

noncomputable def find_a_plus_b (a b : ℕ) (h_neq : a ≠ b) (h_pos : 0 < a ∧ 0 < b) (h_eq : a^2 - b^2 = 2018 - 2 * a) : ℕ :=
  a + b

theorem a_plus_b_value {a b : ℕ} (h_neq : a ≠ b) (h_pos : 0 < a ∧ 0 < b) (h_eq : a^2 - b^2 = 2018 - 2 * a) : find_a_plus_b a b h_neq h_pos h_eq = 672 :=
  sorry

end a_plus_b_value_l212_212858


namespace probability_differ_by_three_is_one_sixth_l212_212486

def probability_of_differ_by_three (outcomes : ℕ) : ℚ :=
  let successful_outcomes := 6
  successful_outcomes / outcomes

theorem probability_differ_by_three_is_one_sixth :
  probability_of_differ_by_three (6 * 6) = 1 / 6 :=
by sorry

end probability_differ_by_three_is_one_sixth_l212_212486


namespace converse_example_l212_212057

theorem converse_example (x : ℝ) (h : x^2 = 1) : x = 1 :=
sorry

end converse_example_l212_212057


namespace sale_price_of_trouser_l212_212831

theorem sale_price_of_trouser : (100 - 0.70 * 100) = 30 := by
  sorry

end sale_price_of_trouser_l212_212831


namespace problem_statement_l212_212293

theorem problem_statement : ∀ (x y : ℝ), |x - 2| + (y + 3)^2 = 0 → (x + y)^2023 = -1 :=
by
  intros x y h
  sorry

end problem_statement_l212_212293


namespace count_negative_x_with_sqrt_pos_int_l212_212443

theorem count_negative_x_with_sqrt_pos_int :
  ∃ (count : ℕ), count = 14 ∧ 
  ∀ (x n : ℤ), (1 ≤ n) ∧ (n * n < 200) ∧ (x = n * n - 200) → x < 0 :=
by sorry

end count_negative_x_with_sqrt_pos_int_l212_212443


namespace volume_of_prism_l212_212105

theorem volume_of_prism (a b c : ℝ) (h1 : a * b = 30) (h2 : a * c = 45) (h3 : b * c = 54) : 
  a * b * c = 270 :=
by
  sorry

end volume_of_prism_l212_212105


namespace sum_of_smallest_x_and_y_for_540_l212_212349

theorem sum_of_smallest_x_and_y_for_540 (x y : ℕ) (hx : 0 < x) (hy : 0 < y)
  (h1 : ∃ k₁, 540 * x = k₁ * k₁)
  (h2 : ∃ k₂, 540 * y = k₂ * k₂ * k₂) :
  x + y = 65 := 
sorry

end sum_of_smallest_x_and_y_for_540_l212_212349


namespace is_not_age_of_child_l212_212450

-- Initial conditions
def mrs_smith_child_ages : List Nat := [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]

-- Given number
def n : Nat := 1124

-- Mrs. Smith's age 
noncomputable def mrs_smith_age : Nat := 46

-- Divisibility check
def is_divisible (n k : Nat) : Bool := n % k = 0

-- Prove the statement
theorem is_not_age_of_child (child_age : Nat) : 
  child_age ∈ mrs_smith_child_ages ∧ ¬ is_divisible n child_age → child_age = 3 :=
by
  intros h
  sorry

end is_not_age_of_child_l212_212450


namespace sequence_general_term_l212_212065

theorem sequence_general_term (a : ℕ → ℕ) (h₁ : a 1 = 1) (h₂ : ∀ n : ℕ, n ≥ 1 → a n = n * (a (n + 1) - a n)) : 
  ∀ n : ℕ, n ≥ 1 → a n = n := 
by 
  sorry

end sequence_general_term_l212_212065


namespace problem_rational_sum_of_powers_l212_212046

theorem problem_rational_sum_of_powers :
  ∃ (a b : ℚ), (1 + Real.sqrt 2)^5 = a + b * Real.sqrt 2 ∧ a + b = 70 :=
by
  sorry

end problem_rational_sum_of_powers_l212_212046


namespace determine_B_l212_212000

-- Declare the sets A and B
def A : Set ℕ := {1, 2}
def B : Set ℕ := {0, 1}

-- The conditions given in the problem
axiom h1 : A ∩ B = {1}
axiom h2 : A ∪ B = {0, 1, 2}

-- The theorem we want to prove
theorem determine_B : B = {0, 1} :=
by
  sorry

end determine_B_l212_212000


namespace length_AB_given_conditions_l212_212034

variable {A B P Q : Type} [LinearOrderedField A] [LinearOrderedField B] [LinearOrderedField P] [LinearOrderedField Q]

def length_of_AB (x y : A) : A := x + y

theorem length_AB_given_conditions (x y u v : A) (hx : y = 4 * x) (hv : 5 * u = 2 * v) (hu : u = x + 3) (hv' : v = y - 3) (hPQ : PQ = 3) : length_of_AB x y = 35 :=
by
  sorry

end length_AB_given_conditions_l212_212034


namespace joan_gave_away_kittens_l212_212352

-- Definitions based on conditions in the problem
def original_kittens : ℕ := 8
def kittens_left : ℕ := 6

-- Mathematical statement to be proved
theorem joan_gave_away_kittens : original_kittens - kittens_left = 2 :=
by
  sorry

end joan_gave_away_kittens_l212_212352


namespace train_length_l212_212022

-- Definitions of speeds and times
def speed_person_A := 5 / 3.6 -- in meters per second
def speed_person_B := 15 / 3.6 -- in meters per second
def time_to_overtake_A := 36 -- in seconds
def time_to_overtake_B := 45 -- in seconds

-- The length of the train
theorem train_length :
  ∃ x : ℝ, x = 500 :=
by
  sorry

end train_length_l212_212022


namespace max_similar_triangles_five_points_l212_212334

-- Let P be a finite set of points on a plane with exactly 5 elements.
def max_similar_triangles(P : Finset (ℝ × ℝ)) : ℕ :=
  if h : P.card = 5 then
    8
  else
    0 -- This is irrelevant for the problem statement, but we need to define it.

-- The main theorem statement
theorem max_similar_triangles_five_points {P : Finset (ℝ × ℝ)} (h : P.card = 5) :
  max_similar_triangles P = 8 :=
sorry

end max_similar_triangles_five_points_l212_212334


namespace range_of_m_l212_212199

theorem range_of_m (m : ℝ) : 
  (∀ x : ℤ, (x > 3 - m) ∧ (x ≤ 5) ↔ (1 ≤ x ∧ x ≤ 5)) →
  (2 < m ∧ m ≤ 3) := 
by
  sorry

end range_of_m_l212_212199


namespace compute_xy_l212_212076

variable (x y : ℝ)

-- Conditions from the problem
def condition1 : Prop := x + y = 10
def condition2 : Prop := x^3 + y^3 = 172

-- Theorem statement to prove the answer
theorem compute_xy (h1 : condition1 x y) (h2 : condition2 x y) : x * y = 41.4 :=
sorry

end compute_xy_l212_212076


namespace problem_statement_l212_212205

noncomputable def tan_plus_alpha_half_pi (α : ℝ) : ℝ := -1 / (Real.tan α)

theorem problem_statement (α : ℝ) (h : tan_plus_alpha_half_pi α = -1 / 2) :
  (2 * Real.sin α + Real.cos α) / (Real.cos α - Real.sin α) = -5 := by
  sorry

end problem_statement_l212_212205


namespace min_value_l212_212002

noncomputable def conditions (x y : ℝ) : Prop :=
  (3^(-x) * y^4 - 2 * y^2 + 3^x ≤ 0) ∧ 
  (27^x + y^4 - 3^x - 1 = 0)

theorem min_value (x y : ℝ) (h : conditions x y) : ∃ x y, (x^3 + y^3 = -1) :=
sorry

end min_value_l212_212002


namespace part_a_solution_l212_212094

theorem part_a_solution (x y : ℤ) : xy + 3 * x - 5 * y = -3 ↔ 
  (x = 6 ∧ y = -21) ∨ 
  (x = -13 ∧ y = -2) ∨ 
  (x = 4 ∧ y = 15) ∨ 
  (x = 23 ∧ y = -4) ∨ 
  (x = 7 ∧ y = -12) ∨ 
  (x = -4 ∧ y = -1) ∨ 
  (x = 3 ∧ y = 6) ∨ 
  (x = 14 ∧ y = -5) ∨ 
  (x = 8 ∧ y = -9) ∨ 
  (x = -1 ∧ y = 0) ∨ 
  (x = 2 ∧ y = 3) ∨ 
  (x = 11 ∧ y = -6) := 
by sorry

end part_a_solution_l212_212094


namespace root_of_function_l212_212613

noncomputable def odd_function (f : ℝ → ℝ) := ∀ x : ℝ, f (-x) = -f (x)

theorem root_of_function (f : ℝ → ℝ) (x₀ : ℝ) (h₀ : odd_function f) (h₁ : f (x₀) = Real.exp (x₀)) :
  (f (-x₀) * Real.exp (-x₀) + 1 = 0) :=
by
  sorry

end root_of_function_l212_212613


namespace metallic_sphere_radius_l212_212522

theorem metallic_sphere_radius 
  (r_wire : ℝ)
  (h_wire : ℝ)
  (r_sphere : ℝ) 
  (V_sphere : ℝ)
  (V_wire : ℝ)
  (h_wire_eq : h_wire = 16)
  (r_wire_eq : r_wire = 12)
  (V_wire_eq : V_wire = π * r_wire^2 * h_wire)
  (V_sphere_eq : V_sphere = (4/3) * π * r_sphere^3)
  (volume_eq : V_sphere = V_wire) :
  r_sphere = 12 :=
by
  sorry

end metallic_sphere_radius_l212_212522


namespace total_squares_in_6x6_grid_l212_212535

theorem total_squares_in_6x6_grid : 
  let size := 6
  let total_squares := (size * size) + ((size - 1) * (size - 1)) + ((size - 2) * (size - 2)) + ((size - 3) * (size - 3)) + ((size - 4) * (size - 4)) + ((size - 5) * (size - 5))
  total_squares = 91 :=
by
  let size := 6
  let total_squares := (size * size) + ((size - 1) * (size - 1)) + ((size - 2) * (size - 2)) + ((size - 3) * (size - 3)) + ((size - 4) * (size - 4)) + ((size - 5) * (size - 5))
  have eqn : total_squares = 91 := sorry
  exact eqn

end total_squares_in_6x6_grid_l212_212535


namespace eval_expr_l212_212600

theorem eval_expr : (2.1 * (49.7 + 0.3)) + 15 = 120 :=
  by
  sorry

end eval_expr_l212_212600


namespace prove_b_plus_m_equals_391_l212_212632

def matrix_A (b : ℕ) : Matrix (Fin 3) (Fin 3) ℕ := ![
  ![1, 3, b],
  ![0, 1, 5],
  ![0, 0, 1]
]

def matrix_power_A (m b : ℕ) : Matrix (Fin 3) (Fin 3) ℕ := 
  (matrix_A b)^(m : ℕ)

def target_matrix : Matrix (Fin 3) (Fin 3) ℕ := ![
  ![1, 21, 3003],
  ![0, 1, 45],
  ![0, 0, 1]
]

theorem prove_b_plus_m_equals_391 (b m : ℕ) (h1 : matrix_power_A m b = target_matrix) : b + m = 391 := by
  sorry

end prove_b_plus_m_equals_391_l212_212632


namespace problem_equiv_proof_l212_212260

theorem problem_equiv_proof : ∀ (i : ℂ), i^2 = -1 → (1 + i^2017) / (1 - i) = i :=
by
  intro i h
  sorry

end problem_equiv_proof_l212_212260


namespace journey_speed_l212_212127

theorem journey_speed
  (v : ℝ) -- Speed during the first four hours
  (total_distance : ℝ) (total_time : ℝ) -- Total distance and time of the journey
  (distance_part1 : ℝ) (time_part1 : ℝ) -- Distance and time for the first part of journey
  (distance_part2 : ℝ) (time_part2 : ℝ) -- Distance and time for the second part of journey
  (speed_part2 : ℝ) : -- Speed during the second part of journey
  total_distance = 24 ∧ total_time = 8 ∧ speed_part2 = 2 ∧ 
  time_part1 = 4 ∧ time_part2 = 4 ∧ 
  distance_part1 = v * time_part1 ∧ distance_part2 = speed_part2 * time_part2 →
  v = 4 := 
by
  sorry

end journey_speed_l212_212127


namespace geometric_seq_arithmetic_triplet_l212_212370

-- Definition of being in a geometric sequence
def is_geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, a (n+1) = a n * q

-- Condition that a_5, a_4, and a_6 form an arithmetic sequence
def is_arithmetic_triplet (a : ℕ → ℝ) (n : ℕ) : Prop :=
  2 * a n = a (n+1) + a (n+2)

-- Our specific problem translated into a Lean statement
theorem geometric_seq_arithmetic_triplet {a : ℕ → ℝ} (q : ℝ) :
  is_geometric_sequence a q →
  is_arithmetic_triplet a 4 →
  q = 1 ∨ q = -2 :=
by
  intros h_geo h_arith
  -- Proof here is omitted
  sorry

end geometric_seq_arithmetic_triplet_l212_212370


namespace min_elements_in_as_l212_212216

noncomputable def min_elems_in_A_s (n : ℕ) (S : Finset ℝ) (hS : S.card = n) : ℕ :=
  if 2 ≤ n then 2 * n - 3 else 0

theorem min_elements_in_as (n : ℕ) (S : Finset ℝ) (hS : S.card = n) (hn: 2 ≤ n) :
  ∃ (A_s : Finset ℝ), A_s.card = min_elems_in_A_s n S hS := sorry

end min_elements_in_as_l212_212216


namespace gino_popsicle_sticks_left_l212_212071

-- Define the initial number of popsicle sticks Gino has
def initial_popsicle_sticks : ℝ := 63.0

-- Define the number of popsicle sticks Gino gives away
def given_away_popsicle_sticks : ℝ := 50.0

-- Expected number of popsicle sticks Gino has left
def expected_remaining_popsicle_sticks : ℝ := 13.0

-- Main theorem to be proven
theorem gino_popsicle_sticks_left :
  initial_popsicle_sticks - given_away_popsicle_sticks = expected_remaining_popsicle_sticks := 
by
  -- This is where the proof would go, but we leave it as 'sorry' for now
  sorry

end gino_popsicle_sticks_left_l212_212071


namespace compare_a_b_c_l212_212252

def a : ℝ := 2^(1/2)
def b : ℝ := 3^(1/3)
def c : ℝ := 5^(1/5)

theorem compare_a_b_c : b > a ∧ a > c :=
  by
  sorry

end compare_a_b_c_l212_212252


namespace batches_of_engines_l212_212451

variable (total_engines : ℕ) (not_defective_engines : ℕ := 300) (engines_per_batch : ℕ := 80)

theorem batches_of_engines (h1 : 3 * total_engines / 4 = not_defective_engines) :
  total_engines / engines_per_batch = 5 := by
sorry

end batches_of_engines_l212_212451


namespace minimum_value_of_x_plus_y_l212_212424

theorem minimum_value_of_x_plus_y
  (x y : ℝ)
  (h1 : x > y)
  (h2 : y > 0)
  (h3 : 1 / (x - y) + 8 / (x + 2 * y) = 1) :
  x + y = 25 / 3 :=
sorry

end minimum_value_of_x_plus_y_l212_212424


namespace dot_product_a_b_l212_212988

-- Define the given vectors
def a : ℝ × ℝ := (1, 1)
def b : ℝ × ℝ := (-1, 2)

-- Define the dot product function
def dot_product (x y : ℝ × ℝ) : ℝ :=
  x.1 * y.1 + x.2 * y.2

-- State the theorem with the correct answer
theorem dot_product_a_b : dot_product a b = 1 :=
by
  sorry

end dot_product_a_b_l212_212988


namespace factorization_of_polynomial_l212_212677

theorem factorization_of_polynomial :
  (x : ℝ) → (x^8 - 4*x^6 + 6*x^4 - 4*x^2 + 1) = ((x - 1)^4 * (x + 1)^4) :=
by
  intro x
  sorry

end factorization_of_polynomial_l212_212677


namespace faye_coloring_books_l212_212131

theorem faye_coloring_books (x : ℕ) : 34 - x + 48 = 79 → x = 3 :=
by
  sorry

end faye_coloring_books_l212_212131


namespace percentage_increase_of_cars_l212_212967

theorem percentage_increase_of_cars :
  ∀ (initial final : ℕ), initial = 24 → final = 48 → ((final - initial) * 100 / initial) = 100 :=
by
  intros
  sorry

end percentage_increase_of_cars_l212_212967


namespace wilson_hamburgers_l212_212508

def hamburger_cost (H : ℕ) := 5 * H
def cola_cost := 6
def discount := 4
def total_cost (H : ℕ) := hamburger_cost H + cola_cost - discount

theorem wilson_hamburgers (H : ℕ) (h : total_cost H = 12) : H = 2 :=
sorry

end wilson_hamburgers_l212_212508


namespace jellybean_ratio_l212_212301

theorem jellybean_ratio (L Tino Arnold : ℕ) (h1 : Tino = L + 24) (h2 : Arnold = 5) (h3 : Tino = 34) :
  Arnold / L = 1 / 2 :=
by
  sorry

end jellybean_ratio_l212_212301


namespace factorization_correct_l212_212700

theorem factorization_correct:
  ∃ a b : ℤ, (25 * x^2 - 85 * x - 150 = (5 * x + a) * (5 * x + b)) ∧ (a + 2 * b = -24) :=
by
  sorry

end factorization_correct_l212_212700


namespace intersection_A_B_l212_212113

def A := {x : ℝ | x < -1 ∨ x > 1}
def B := {x : ℝ | Real.log x / Real.log 2 > 0}

theorem intersection_A_B:
  A ∩ B = {x : ℝ | x > 1} :=
by
  sorry

end intersection_A_B_l212_212113


namespace fraction_division_l212_212280

theorem fraction_division :
  (3 / 7) / (2 / 5) = (15 / 14) :=
by
  sorry

end fraction_division_l212_212280


namespace concave_number_count_l212_212312

def is_concave_number (n : ℕ) : Prop :=
  let hundreds := n / 100
  let tens := (n / 10) % 10
  let units := n % 10
  n >= 100 ∧ n < 1000 ∧ tens < hundreds ∧ tens < units

theorem concave_number_count : ∃ n : ℕ, 
  (∀ m < 1000, is_concave_number m → m = n) ∧ n = 240 :=
by
  sorry

end concave_number_count_l212_212312


namespace recover_original_sequence_l212_212800

theorem recover_original_sequence :
  ∃ (a d : ℤ),
    [a, a + d, a + 2 * d, a + 3 * d, a + 4 * d, a + 5 * d] = [113, 125, 137, 149, 161, 173] :=
by
  sorry

end recover_original_sequence_l212_212800


namespace deepak_age_l212_212314

theorem deepak_age
  (A D : ℕ)
  (h1 : A / D = 2 / 5)  -- the ratio condition
  (h2 : A + 10 = 30)   -- Arun’s age after 10 years will be 30
  : D = 50 :=       -- conclusion Deepak is 50 years old
sorry

end deepak_age_l212_212314


namespace monotonic_decreasing_range_of_a_l212_212488

-- Define the given function
def f (a x : ℝ) := a * x^2 - 3 * x + 4

-- State the proof problem
theorem monotonic_decreasing_range_of_a (a : ℝ) : (∀ x : ℝ, x < 6 → deriv (f a) x ≤ 0) ↔ 0 ≤ a ∧ a ≤ 1/4 :=
sorry

end monotonic_decreasing_range_of_a_l212_212488


namespace max_newsstands_six_corridors_l212_212923

def number_of_intersections (n : ℕ) : ℕ :=
  n * (n - 1) / 2

theorem max_newsstands_six_corridors : number_of_intersections 6 = 15 := 
by sorry

end max_newsstands_six_corridors_l212_212923


namespace algebraic_expression_l212_212140

-- Given conditions in the problem.
variables (x y : ℝ)

-- The statement to be proved: If 2x - 3y = 1, then 6y - 4x + 8 = 6.
theorem algebraic_expression (h : 2 * x - 3 * y = 1) : 6 * y - 4 * x + 8 = 6 :=
by 
  sorry

end algebraic_expression_l212_212140


namespace factorize_expression_l212_212785

theorem factorize_expression (m : ℝ) : 2 * m^2 - 8 = 2 * (m + 2) * (m - 2) :=
sorry

end factorize_expression_l212_212785


namespace opposite_of_neg_2023_l212_212922

-- Definition of the problem
def opposite (n : Int) : Int := -n

-- Theorem statement
theorem opposite_of_neg_2023 : opposite (-2023) = 2023 :=
by
  -- The proof is to be provided
  sorry

end opposite_of_neg_2023_l212_212922


namespace square_side_length_l212_212696

theorem square_side_length (A : ℝ) (h : A = 25) : ∃ s : ℝ, s * s = A ∧ s = 5 :=
by
  sorry

end square_side_length_l212_212696


namespace square_side_length_l212_212283

theorem square_side_length (A : ℝ) (s : ℝ) (hA : A = 64) (h_s : A = s * s) : s = 8 := by
  sorry

end square_side_length_l212_212283


namespace equilateral_triangle_intersection_impossible_l212_212599

noncomputable def trihedral_angle (α β γ : ℝ) : Prop :=
  α + β + γ = 180 ∧ β = 90 ∧ γ = 90 ∧ α > 0

theorem equilateral_triangle_intersection_impossible :
  ¬ ∀ (α : ℝ), ∀ (β γ : ℝ), trihedral_angle α β γ → 
    ∃ (plane : ℝ → ℝ → ℝ), 
      ∀ (x y z : ℝ), plane x y = z → x = y ∧ y = z ∧ z = x ∧ 
                      x + y + z = 60 :=
sorry

end equilateral_triangle_intersection_impossible_l212_212599


namespace greater_number_is_18_l212_212207

theorem greater_number_is_18 (x y : ℝ) 
  (h1 : x + y = 30) 
  (h2 : x - y = 6) 
  (h3 : y ≥ 10) : 
  x = 18 := 
by 
  sorry

end greater_number_is_18_l212_212207


namespace inverse_function_log_base_two_l212_212499

noncomputable def f (a x : ℝ) : ℝ := Real.log x / Real.log a

theorem inverse_function_log_base_two (a : ℝ) (h1 : 0 < a) (h2 : a ≠ 1)
  (h3 : f a (a^2) = a) : f a = fun x => Real.log x / Real.log 2 := 
by
  sorry

end inverse_function_log_base_two_l212_212499


namespace exchange_5_rubles_l212_212196

theorem exchange_5_rubles :
  ¬ ∃ n : ℕ, 1 * n + 2 * n + 3 * n + 5 * n = 500 :=
by 
  sorry

end exchange_5_rubles_l212_212196


namespace right_triangle_area_valid_right_triangle_perimeter_valid_l212_212278

-- Define the basic setup for the right triangle problem
def hypotenuse : ℕ := 13
def leg1 : ℕ := 5
def leg2 : ℕ := 12  -- Calculated from Pythagorean theorem, but assumed here as condition

-- Define the calculated area and perimeter based on the above definitions
def area (a b : ℕ) : ℕ := (1 / 2) * a * b
def perimeter (a b c : ℕ) : ℕ := a + b + c

-- State the proof goals
theorem right_triangle_area_valid : area leg1 leg2 = 30 :=
  by sorry

theorem right_triangle_perimeter_valid : perimeter leg1 leg2 hypotenuse = 30 :=
  by sorry

end right_triangle_area_valid_right_triangle_perimeter_valid_l212_212278


namespace rods_needed_to_complete_6_step_pyramid_l212_212578

def rods_in_step (n : ℕ) : ℕ :=
  16 * n

theorem rods_needed_to_complete_6_step_pyramid (rods_1_step rods_2_step : ℕ) :
  rods_1_step = 16 → rods_2_step = 32 → rods_in_step 6 - rods_in_step 4 = 32 :=
by
  intros h1 h2
  sorry

end rods_needed_to_complete_6_step_pyramid_l212_212578


namespace natasha_quarters_l212_212365

theorem natasha_quarters :
  ∃ n : ℕ, (4 < n) ∧ (n < 40) ∧ (n % 4 = 2) ∧ (n % 5 = 2) ∧ (n % 6 = 2) ∧ (n = 2) := sorry

end natasha_quarters_l212_212365


namespace remainder_of_3_pow_102_mod_101_l212_212534

theorem remainder_of_3_pow_102_mod_101 : (3^102) % 101 = 9 :=
by
  sorry

end remainder_of_3_pow_102_mod_101_l212_212534


namespace polynomial_min_value_l212_212566

noncomputable def poly (x y : ℝ) : ℝ := x^2 + y^2 - 6*x + 8*y + 7

theorem polynomial_min_value : 
  ∃ x y : ℝ, poly x y = -18 :=
by
  sorry

end polynomial_min_value_l212_212566


namespace blue_pill_cost_is_25_l212_212918

variable (blue_pill_cost red_pill_cost : ℕ)

-- Clara takes one blue pill and one red pill each day for 10 days.
-- A blue pill costs $2 more than a red pill.
def pill_cost_condition (blue_pill_cost red_pill_cost : ℕ) : Prop :=
  blue_pill_cost = red_pill_cost + 2 ∧
  10 * blue_pill_cost + 10 * red_pill_cost = 480

-- Prove that the cost of one blue pill is $25.
theorem blue_pill_cost_is_25 (h : pill_cost_condition blue_pill_cost red_pill_cost) : blue_pill_cost = 25 :=
  sorry

end blue_pill_cost_is_25_l212_212918


namespace speed_of_first_train_l212_212501

-- Define the problem conditions
def distance_between_stations : ℝ := 20
def speed_of_second_train : ℝ := 25
def meet_time : ℝ := 8
def start_time_first_train : ℝ := 7
def start_time_second_train : ℝ := 8
def travel_time_first_train : ℝ := meet_time - start_time_first_train

-- The actual proof statement in Lean
theorem speed_of_first_train : ∀ (v : ℝ),
  v * travel_time_first_train = distance_between_stations → v = 20 :=
by
  intro v
  intro h
  sorry

end speed_of_first_train_l212_212501


namespace smaller_inscribed_cube_volume_is_192_sqrt_3_l212_212710

noncomputable def volume_of_smaller_inscribed_cube : ℝ :=
  let edge_length_of_larger_cube := 12
  let diameter_of_sphere := edge_length_of_larger_cube
  let side_length_of_smaller_cube := diameter_of_sphere / Real.sqrt 3
  let volume := side_length_of_smaller_cube ^ 3
  volume

theorem smaller_inscribed_cube_volume_is_192_sqrt_3 : 
  volume_of_smaller_inscribed_cube = 192 * Real.sqrt 3 := 
by
  sorry

end smaller_inscribed_cube_volume_is_192_sqrt_3_l212_212710


namespace white_seeds_per_slice_l212_212862

theorem white_seeds_per_slice (W : ℕ) (black_seeds_per_slice : ℕ) (number_of_slices : ℕ) 
(total_seeds : ℕ) (total_black_seeds : ℕ) (total_white_seeds : ℕ) 
(h1 : black_seeds_per_slice = 20)
(h2 : number_of_slices = 40)
(h3 : total_seeds = 1600)
(h4 : total_black_seeds = black_seeds_per_slice * number_of_slices)
(h5 : total_white_seeds = total_seeds - total_black_seeds)
(h6 : W = total_white_seeds / number_of_slices) :
W = 20 :=
by
  sorry

end white_seeds_per_slice_l212_212862


namespace tetrahedron_coloring_l212_212206

noncomputable def count_distinct_tetrahedron_colorings : ℕ :=
  sorry

theorem tetrahedron_coloring :
  count_distinct_tetrahedron_colorings = 6 :=
  sorry

end tetrahedron_coloring_l212_212206


namespace determine_triangle_value_l212_212570

theorem determine_triangle_value (p : ℕ) (triangle : ℕ) (h1 : triangle + p = 67) (h2 : 3 * (triangle + p) - p = 185) : triangle = 51 := by
  sorry

end determine_triangle_value_l212_212570


namespace chemistry_marks_more_than_physics_l212_212019

theorem chemistry_marks_more_than_physics (M P C x : ℕ) 
  (h1 : M + P = 32) 
  (h2 : (M + C) / 2 = 26) 
  (h3 : C = P + x) : 
  x = 20 := 
by
  sorry

end chemistry_marks_more_than_physics_l212_212019


namespace each_parent_suitcases_l212_212718

namespace SuitcaseProblem

-- Definitions based on conditions
def siblings : Nat := 4
def suitcases_per_sibling : Nat := 2
def total_suitcases : Nat := 14

-- Theorem statement corresponding to the question and correct answer
theorem each_parent_suitcases (suitcases_per_parent : Nat) :
  (siblings * suitcases_per_sibling + 2 * suitcases_per_parent = total_suitcases) →
  suitcases_per_parent = 3 := by
  intro h
  sorry

end SuitcaseProblem

end each_parent_suitcases_l212_212718


namespace book_arrangement_count_l212_212861

-- Conditions
def num_math_books := 4
def num_history_books := 5

-- The number of arrangements is
def arrangements (n m : Nat) : Nat :=
  let choose_end_books := n * (n - 1)
  let choose_middle_book := (n - 2)
  let remaining_books := (n - 3) + m
  choose_end_books * choose_middle_book * Nat.factorial remaining_books

theorem book_arrangement_count (n m : Nat) (h1 : n = num_math_books) (h2 : m = num_history_books) :
  arrangements n m = 120960 :=
by
  rw [h1, h2, arrangements]
  norm_num
  sorry

end book_arrangement_count_l212_212861


namespace pages_in_each_book_l212_212086

variable (BooksRead DaysPerBook TotalDays : ℕ)

theorem pages_in_each_book (h1 : BooksRead = 41) (h2 : DaysPerBook = 12) (h3 : TotalDays = 492) : (TotalDays / DaysPerBook) * DaysPerBook = 492 :=
by
  sorry

end pages_in_each_book_l212_212086


namespace tv_show_duration_l212_212112

theorem tv_show_duration (total_air_time : ℝ) (num_commercials : ℕ) (commercial_duration_min : ℝ) :
  total_air_time = 1.5 ∧ num_commercials = 3 ∧ commercial_duration_min = 10 →
  (total_air_time - (num_commercials * commercial_duration_min / 60)) = 1 :=
by
  sorry

end tv_show_duration_l212_212112


namespace expression_value_l212_212213

noncomputable def expr := (1.90 * (1 / (1 - (3: ℝ)^(1/4)))) + (1 / (1 + (3: ℝ)^(1/4))) + (2 / (1 + (3: ℝ)^(1/2)))

theorem expression_value : expr = -2 := 
by
  sorry

end expression_value_l212_212213


namespace calculate_power_expr_l212_212296

theorem calculate_power_expr :
  let a := (-8 : ℝ)
  let b := (0.125 : ℝ)
  a^2023 * b^2024 = -0.125 :=
by
  sorry

end calculate_power_expr_l212_212296


namespace quadratic_has_one_solution_l212_212385

theorem quadratic_has_one_solution (m : ℝ) : (∃ x : ℝ, 3 * x^2 - 6 * x + m = 0) ∧ (∀ x₁ x₂ : ℝ, (3 * x₁^2 - 6 * x₁ + m = 0) → (3 * x₂^2 - 6 * x₂ + m = 0) → x₁ = x₂) → m = 3 :=
by
  -- intricate steps would go here
  sorry

end quadratic_has_one_solution_l212_212385


namespace not_possible_total_l212_212214

-- Definitions
variables (d r : ℕ)

-- Theorem to prove that 58 cannot be expressed as 26d + 3r
theorem not_possible_total : ¬∃ (d r : ℕ), 26 * d + 3 * r = 58 :=
sorry

end not_possible_total_l212_212214


namespace problem1_problem2_l212_212029

-- Definitions of the sets
def U : Set ℕ := { x | 1 ≤ x ∧ x ≤ 7 }
def A : Set ℕ := { x | 2 ≤ x ∧ x ≤ 5 }
def B : Set ℕ := { x | 3 ≤ x ∧ x ≤ 7 }

-- Problems to prove (statements only, no proofs provided)
theorem problem1 : A ∪ B = {x | 2 ≤ x ∧ x ≤ 7} :=
by
  sorry

theorem problem2 : U \ A ∪ B = {x | (1 ≤ x ∧ x < 2) ∨ (3 ≤ x ∧ x ≤ 7)} :=
by
  sorry

end problem1_problem2_l212_212029


namespace set_union_is_all_real_l212_212051

-- Define the universal set U as the real numbers
def U := ℝ

-- Define the set M as {x | x > 0}
def M : Set ℝ := {x | x > 0}

-- Define the set N as {x | x^2 ≥ x}
def N : Set ℝ := {x | x^2 ≥ x}

-- Prove the relationship M ∪ N = ℝ
theorem set_union_is_all_real : M ∪ N = U := by
  sorry

end set_union_is_all_real_l212_212051


namespace leah_daily_savings_l212_212531

theorem leah_daily_savings 
  (L : ℝ)
  (h1 : 0.25 * 24 = 6)
  (h2 : ∀ (L : ℝ), (L * 20) = 20 * L)
  (h3 : ∀ (L : ℝ), 2 * L * 12 = 24 * L)
  (h4 :  6 + 20 * L + 24 * L = 28) 
: L = 0.5 :=
by
  sorry

end leah_daily_savings_l212_212531


namespace rons_baseball_team_l212_212010

/-- Ron's baseball team scored 270 points in the year. 
    5 players averaged 50 points each, 
    and the remaining players averaged 5 points each.
    Prove that the number of players on the team is 9. -/
theorem rons_baseball_team : (∃ n m : ℕ, 5 * 50 + m * 5 = 270 ∧ n = 5 + m ∧ 5 = 50 ∧ m = 4) :=
sorry

end rons_baseball_team_l212_212010


namespace smallest_negative_integer_solution_l212_212390

theorem smallest_negative_integer_solution :
  ∃ x : ℤ, 45 * x + 8 ≡ 5 [ZMOD 24] ∧ x = -7 :=
sorry

end smallest_negative_integer_solution_l212_212390


namespace heesu_has_greatest_sum_l212_212912

theorem heesu_has_greatest_sum :
  let Sora_sum := 4 + 6
  let Heesu_sum := 7 + 5
  let Jiyeon_sum := 3 + 8
  Heesu_sum > Sora_sum ∧ Heesu_sum > Jiyeon_sum :=
by
  let Sora_sum := 4 + 6
  let Heesu_sum := 7 + 5
  let Jiyeon_sum := 3 + 8
  have h1 : Heesu_sum > Sora_sum := by sorry
  have h2 : Heesu_sum > Jiyeon_sum := by sorry
  exact And.intro h1 h2

end heesu_has_greatest_sum_l212_212912


namespace taco_variants_count_l212_212647

theorem taco_variants_count :
  let toppings := 8
  let meat_variants := 3
  let shell_variants := 2
  2 ^ toppings * meat_variants * shell_variants = 1536 := by
sorry

end taco_variants_count_l212_212647


namespace evaluate_fraction_l212_212025

theorem evaluate_fraction :
  1 + (2 / (3 + (6 / (7 + (8 / 9))))) = 409 / 267 :=
by
  sorry

end evaluate_fraction_l212_212025


namespace no_solution_x_l212_212104

theorem no_solution_x : ¬ ∃ x : ℝ, x * (x - 1) * (x - 2) + (100 - x) * (99 - x) * (98 - x) = 0 := 
sorry

end no_solution_x_l212_212104


namespace optimal_garden_area_l212_212467

variable (l w : ℕ)

/-- Tiffany is building a fence around a rectangular garden. Determine the optimal area, 
    in square feet, that can be enclosed under the conditions. -/
theorem optimal_garden_area 
  (h1 : l >= 100)
  (h2 : w >= 50)
  (h3 : 2 * l + 2 * w = 400) : (l * w) ≤ 7500 := 
sorry

end optimal_garden_area_l212_212467


namespace problem_statement_l212_212927

theorem problem_statement : 15 * 30 + 45 * 15 + 15 * 15 = 1350 :=
by
  sorry

end problem_statement_l212_212927


namespace am_gm_inequality_l212_212011

theorem am_gm_inequality {a1 a2 a3 : ℝ} (h1 : 0 < a1) (h2 : 0 < a2) (h3 : 0 < a3) :
  (a1 * a2 / a3) + (a2 * a3 / a1) + (a3 * a1 / a2) ≥ a1 + a2 + a3 := 
by 
  sorry

end am_gm_inequality_l212_212011


namespace books_remaining_correct_l212_212257

-- Define the total number of books and the number of books read
def total_books : ℕ := 32
def books_read : ℕ := 17

-- Define the number of books remaining to be read
def books_remaining : ℕ := total_books - books_read

-- Prove that the number of books remaining to be read is 15
theorem books_remaining_correct : books_remaining = 15 := by
  sorry

end books_remaining_correct_l212_212257


namespace sum_first_five_terms_l212_212191

noncomputable def geometric_sequence (a1 q : ℝ) (n : ℕ) : ℝ := a1 * q^(n-1)

noncomputable def sum_geometric_sequence (a1 q : ℝ) (n : ℕ) : ℝ := 
  if q = 1 then a1 * n else a1 * (1 - q^n) / (1 - q)

theorem sum_first_five_terms (a1 q : ℝ) 
  (h1 : geometric_sequence a1 q 2 * geometric_sequence a1 q 3 = 2 * a1)
  (h2 : (geometric_sequence a1 q 4 + 2 * geometric_sequence a1 q 7) / 2 = 5 / 4)
  : sum_geometric_sequence a1 q 5 = 31 :=
sorry

end sum_first_five_terms_l212_212191


namespace last_two_digits_of_sum_of_first_15_factorials_eq_13_l212_212101

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def last_two_digits_sum : ℕ :=
  let partial_sum := (factorial 1 % 100) + (factorial 2 % 100) + (factorial 3 % 100) +
                     (factorial 4 % 100) + (factorial 5 % 100) + (factorial 6 % 100) +
                     (factorial 7 % 100) + (factorial 8 % 100) + (factorial 9 % 100)
  partial_sum % 100

theorem last_two_digits_of_sum_of_first_15_factorials_eq_13 : last_two_digits_sum = 13 := by
  sorry

end last_two_digits_of_sum_of_first_15_factorials_eq_13_l212_212101


namespace cube_of_square_of_third_smallest_prime_is_correct_l212_212917

def cube_of_square_of_third_smallest_prime : Nat := 15625

theorem cube_of_square_of_third_smallest_prime_is_correct :
  let third_smallest_prime := 5
  let square := third_smallest_prime ^ 2
  let cube := square ^ 3
  cube = cube_of_square_of_third_smallest_prime :=
by
  let third_smallest_prime := 5
  let square := third_smallest_prime ^ 2
  let cube := square ^ 3
  show cube = 15625
  sorry

end cube_of_square_of_third_smallest_prime_is_correct_l212_212917


namespace tina_work_time_l212_212885

theorem tina_work_time (T : ℕ) (h1 : ∀ Ann_hours, Ann_hours = 9)
                       (h2 : ∀ Tina_worked_hours, Tina_worked_hours = 8)
                       (h3 : ∀ Ann_worked_hours, Ann_worked_hours = 3)
                       (h4 : (8 : ℚ) / T + (1 : ℚ) / 3 = 1) : T = 12 :=
by
  sorry

end tina_work_time_l212_212885


namespace determine_b_l212_212409

theorem determine_b (b : ℝ) : (∀ x : ℝ, (-x^2 + b * x + 1 < 0) ↔ (x < 2 ∨ x > 6)) → b = 8 :=
by sorry

end determine_b_l212_212409


namespace min_value_expression_l212_212617

theorem min_value_expression (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) :
  (6 * z) / (3 * x + y) + (6 * x) / (y + 3 * z) + (2 * y) / (x + 2 * z) ≥ 3 :=
by
  sorry

end min_value_expression_l212_212617


namespace max_value_m_l212_212598

noncomputable def max_m : ℝ := 10

theorem max_value_m (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (hxy : x * y = x + 2 * y) : x * y ≥ max_m - 2 :=
by
  sorry

end max_value_m_l212_212598


namespace total_flour_needed_l212_212310

theorem total_flour_needed (Katie_flour : ℕ) (Sheila_flour : ℕ) 
  (h1 : Katie_flour = 3) 
  (h2 : Sheila_flour = Katie_flour + 2) : 
  Katie_flour + Sheila_flour = 8 := 
  by 
  sorry

end total_flour_needed_l212_212310


namespace smallest_five_digit_perfect_square_and_cube_l212_212607

theorem smallest_five_digit_perfect_square_and_cube :
  ∃ n : ℕ, (10000 ≤ n ∧ n < 100000) ∧ (∃ a : ℕ, n = a^6) ∧ n = 15625 := 
by
  sorry

end smallest_five_digit_perfect_square_and_cube_l212_212607


namespace find_original_price_l212_212287

theorem find_original_price (sale_price : ℕ) (discount : ℕ) (original_price : ℕ) 
  (h1 : sale_price = 60) 
  (h2 : discount = 40) 
  (h3 : original_price = sale_price / ((100 - discount) / 100)) : original_price = 100 :=
by
  sorry

end find_original_price_l212_212287


namespace find_function_l212_212119

noncomputable def f (x : ℝ) : ℝ := sorry

theorem find_function (h : ∀ x y : ℝ, (f x * f y - f (x * y)) / 2 = x + y + 1) : 
  ∀ x : ℝ, f x = x + 2 := sorry

end find_function_l212_212119


namespace apple_cost_l212_212516

theorem apple_cost (rate_cost : ℕ) (rate_weight total_weight : ℕ) (h_rate : rate_cost = 5) (h_weight : rate_weight = 7) (h_total : total_weight = 21) :
  ∃ total_cost : ℕ, total_cost = 15 :=
by
  -- The proof will go here
  sorry

end apple_cost_l212_212516


namespace no_valid_n_for_conditions_l212_212124

theorem no_valid_n_for_conditions :
  ∀ (n : ℕ), (100 ≤ n / 5 ∧ n / 5 ≤ 999) ∧ (100 ≤ 5 * n ∧ 5 * n ≤ 999) → false :=
by
  sorry

end no_valid_n_for_conditions_l212_212124


namespace geom_progression_sum_ratio_l212_212325

theorem geom_progression_sum_ratio (a : ℝ) (r : ℝ) (m : ℕ) :
  r = 5 →
  (a * (1 - r^6) / (1 - r)) / (a * (1 - r^m) / (1 - r)) = 126 →
  m = 3 := by
  sorry

end geom_progression_sum_ratio_l212_212325


namespace bullfinches_are_50_l212_212556

theorem bullfinches_are_50 :
  ∃ N : ℕ, (N > 50 ∨ N < 50 ∨ N ≥ 1) ∧ (¬(N > 50) ∨ ¬(N < 50) ∨ ¬(N ≥ 1)) ∧
  (N > 50 ∧ ¬(N < 50) ∨ N < 50 ∧ ¬(N > 50) ∨ N ≥ 1 ∧ (¬(N > 50) ∧ ¬(N < 50))) ∧
  N = 50 :=
by
  sorry

end bullfinches_are_50_l212_212556


namespace intersection_of_sets_l212_212487

noncomputable def setA : Set ℝ := { x | (x + 2) / (x - 2) ≤ 0 }
noncomputable def setB : Set ℝ := { x | x ≥ 1 }
noncomputable def expectedSet : Set ℝ := { x | 1 ≤ x ∧ x < 2 }

theorem intersection_of_sets : (setA ∩ setB) = expectedSet := by
  sorry

end intersection_of_sets_l212_212487


namespace bruce_total_payment_l212_212670

-- Define the conditions
def quantity_grapes : Nat := 7
def rate_grapes : Nat := 70
def quantity_mangoes : Nat := 9
def rate_mangoes : Nat := 55

-- Define the calculation for total amount paid
def total_amount_paid : Nat :=
  (quantity_grapes * rate_grapes) + (quantity_mangoes * rate_mangoes)

-- Proof statement
theorem bruce_total_payment : total_amount_paid = 985 :=
by
  -- Proof steps would go here
  sorry

end bruce_total_payment_l212_212670


namespace max_min_value_l212_212546

theorem max_min_value (x y z : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : 0 < z) (h4 : x + y + z = 12) (h5 : x * y + y * z + z * x = 30) :
  ∃ n : ℝ, n = min (x * y) (min (y * z) (z * x)) ∧ n = 2 :=
sorry

end max_min_value_l212_212546


namespace factor_difference_of_squares_l212_212466

theorem factor_difference_of_squares (x : ℝ) : x^2 - 64 = (x - 8) * (x + 8) := 
by 
  sorry

end factor_difference_of_squares_l212_212466


namespace find_number_in_parentheses_l212_212240

theorem find_number_in_parentheses :
  ∃ x : ℝ, 3 + 2 * (x - 3) = 24.16 ∧ x = 13.58 :=
by
  sorry

end find_number_in_parentheses_l212_212240


namespace geometric_sequence_a_eq_2_l212_212814

theorem geometric_sequence_a_eq_2 (a : ℝ) (h1 : ¬ a = 0) (h2 : (2 * a) ^ 2 = 8 * a) : a = 2 :=
by {
  sorry -- Proof not required, only the statement.
}

end geometric_sequence_a_eq_2_l212_212814


namespace garden_perimeter_l212_212201

/-- Define the dimensions of the rectangle and triangle in the garden -/
def rectangle_length : ℕ := 10
def rectangle_width : ℕ := 4
def triangle_leg1 : ℕ := 3
def triangle_leg2 : ℕ := 4
def triangle_hypotenuse : ℕ := 5 -- calculated using Pythagorean theorem

/-- Prove that the total perimeter of the combined shape is 28 units -/
theorem garden_perimeter :
  let perimeter := 2 * rectangle_length + rectangle_width + triangle_leg1 + triangle_hypotenuse
  perimeter = 28 :=
by
  let perimeter := 2 * rectangle_length + rectangle_width + triangle_leg1 + triangle_hypotenuse
  have h : perimeter = 28 := sorry
  exact h

end garden_perimeter_l212_212201


namespace emissions_from_tap_water_l212_212218

def carbon_dioxide_emission (x : ℕ) : ℕ := 9 / 10 * x  -- Note: using 9/10 instead of 0.9 to maintain integer type

theorem emissions_from_tap_water : carbon_dioxide_emission 10 = 9 :=
by
  sorry

end emissions_from_tap_water_l212_212218


namespace least_positive_integer_l212_212298

theorem least_positive_integer (x : ℕ) (h : x + 5600 ≡ 325 [MOD 15]) : x = 5 :=
sorry

end least_positive_integer_l212_212298


namespace partI_l212_212924

noncomputable def f (x : ℝ) : ℝ := abs (1 - 1/x)

theorem partI (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a < b) (h4 : f a = f b) :
  a * b > 1 :=
  sorry

end partI_l212_212924


namespace func_above_x_axis_l212_212778

theorem func_above_x_axis (a : ℝ) :
  (∀ x : ℝ, (x^4 + 4*x^3 + a*x^2 - 4*x + 1) > 0) ↔ a > 2 :=
sorry

end func_above_x_axis_l212_212778


namespace shooter_random_event_l212_212483

def eventA := "The sun rises from the east"
def eventB := "A coin thrown up from the ground will fall down"
def eventC := "A shooter hits the target with 10 points in one shot"
def eventD := "Xiao Ming runs at a speed of 30 meters per second"

def is_random_event (event : String) := event = eventC

theorem shooter_random_event : is_random_event eventC := 
by
  sorry

end shooter_random_event_l212_212483


namespace oldest_child_age_l212_212880

theorem oldest_child_age 
  (x : ℕ)
  (h1 : (6 + 8 + 10 + x) / 4 = 9)
  (h2 : 6 + 8 + 10 = 24) :
  x = 12 := 
by 
  sorry

end oldest_child_age_l212_212880


namespace purely_imaginary_complex_number_l212_212738

theorem purely_imaginary_complex_number (a : ℝ) (h : (a^2 - 3 * a + 2) = 0 ∧ (a - 2) ≠ 0) : a = 1 :=
by {
  sorry
}

end purely_imaginary_complex_number_l212_212738


namespace least_value_divisibility_l212_212812

theorem least_value_divisibility : ∃ (x : ℕ), (23 * x) % 3 = 0  ∧ (∀ y : ℕ, ((23 * y) % 3 = 0 → x ≤ y)) := 
  sorry

end least_value_divisibility_l212_212812


namespace simplify_expression_l212_212559

theorem simplify_expression (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) : (2 * x⁻¹ + 3 * y⁻¹)⁻¹ = (x * y) / (2 * y + 3 * x) :=
by sorry

end simplify_expression_l212_212559


namespace find_x_l212_212962

theorem find_x :
  ∃ x : ℕ, (5 * 12) / (x / 3) + 80 = 81 ∧ x = 180 :=
by
  sorry

end find_x_l212_212962


namespace price_of_each_bracelet_l212_212394

-- The conditions
def bike_cost : ℕ := 112
def days_in_two_weeks : ℕ := 14
def bracelets_per_day : ℕ := 8
def total_bracelets := days_in_two_weeks * bracelets_per_day

-- The question and the expected answer
def price_per_bracelet : ℕ := bike_cost / total_bracelets

theorem price_of_each_bracelet :
  price_per_bracelet = 1 := 
by
  sorry

end price_of_each_bracelet_l212_212394


namespace dessert_menu_count_is_192_l212_212161

-- Defining the set of desserts
inductive Dessert
| cake | pie | ice_cream

-- Function to count valid dessert menus (not repeating on consecutive days) with cake on Friday
def countDessertMenus : Nat :=
  -- Let's denote Sunday as day 1 and Saturday as day 7
  let sunday_choices := 3
  let weekday_choices := 2 -- for Monday to Thursday (no repeats consecutive)
  let weekend_choices := 2 -- for Saturday and Sunday after
  sunday_choices * weekday_choices^4 * 1 * weekend_choices^2

-- Theorem stating the number of valid dessert menus for the week
theorem dessert_menu_count_is_192 : countDessertMenus = 192 :=
  by
    -- Actual proof is omitted
    sorry

end dessert_menu_count_is_192_l212_212161


namespace earnings_difference_l212_212826

-- Definitions:
def investments_ratio := (3, 4, 5)
def return_ratio := (6, 5, 4)
def total_earnings := 5800

-- Target statement:
theorem earnings_difference (x y : ℝ)
  (h_investment_ratio : investments_ratio = (3, 4, 5))
  (h_return_ratio : return_ratio = (6, 5, 4))
  (h_total_earnings : (3 * x * 6 * y) / 100 + (4 * x * 5 * y) / 100 + (5 * x * 4 * y) / 100 = total_earnings) :
  ((4 * x * 5 * y) / 100 - (3 * x * 6 * y) / 100) = 200 := 
by
  sorry

end earnings_difference_l212_212826


namespace probability_rain_all_three_days_l212_212462

-- Define the probabilities as constant values
def prob_rain_friday : ℝ := 0.4
def prob_rain_saturday : ℝ := 0.5
def prob_rain_sunday : ℝ := 0.3
def prob_rain_sunday_given_fri_sat : ℝ := 0.6

-- Define the probability of raining all three days considering the conditional probabilities
def prob_rain_all_three_days : ℝ :=
  prob_rain_friday * prob_rain_saturday * prob_rain_sunday_given_fri_sat

-- Prove that the probability of rain on all three days is 12%
theorem probability_rain_all_three_days : prob_rain_all_three_days = 0.12 :=
by
  sorry

end probability_rain_all_three_days_l212_212462


namespace solve_for_ratio_l212_212684

noncomputable def slope_tangent_y_equals_x_squared (x1 : ℝ) : ℝ :=
  2 * x1

noncomputable def slope_tangent_y_equals_x_cubed (x2 : ℝ) : ℝ :=
  3 * x2 * x2

noncomputable def y1_compute (x1 : ℝ) : ℝ :=
  x1 * x1

noncomputable def y2_compute (x2 : ℝ) : ℝ :=
  x2 * x2 * x2

theorem solve_for_ratio (x1 x2 : ℝ)
    (tangent_l_same : slope_tangent_y_equals_x_squared x1 = slope_tangent_y_equals_x_cubed x2)
    (y_tangent_l_same : y1_compute x1 = y2_compute x2) :
  x1 / x2 = 4 / 3 :=
by
  sorry

end solve_for_ratio_l212_212684


namespace prove_b_div_c_equals_one_l212_212951

theorem prove_b_div_c_equals_one
  (a b c d : ℕ)
  (h_a : a > 0 ∧ a < 4)
  (h_b : b > 0 ∧ b < 4)
  (h_c : c > 0 ∧ c < 4)
  (h_d : d > 0 ∧ d < 4)
  (h_eq : 4^a + 3^b + 2^c + 1^d = 78) :
  b / c = 1 :=
by
  sorry

end prove_b_div_c_equals_one_l212_212951


namespace dealer_decision_is_mode_l212_212395

noncomputable def sales_A := 15
noncomputable def sales_B := 22
noncomputable def sales_C := 18
noncomputable def sales_D := 10

def is_mode (sales: List ℕ) (mode_value: ℕ) : Prop :=
  mode_value ∈ sales ∧ ∀ x ∈ sales, x ≤ mode_value

theorem dealer_decision_is_mode : 
  is_mode [sales_A, sales_B, sales_C, sales_D] sales_B :=
by
  sorry

end dealer_decision_is_mode_l212_212395


namespace diff_of_squares_l212_212760

theorem diff_of_squares (a b : ℝ) (h1 : a + b = -2) (h2 : a - b = 4) : a^2 - b^2 = -8 :=
by
  sorry

end diff_of_squares_l212_212760


namespace charcoal_drawings_count_l212_212494

/-- Thomas' drawings problem
  Thomas has 25 drawings in total.
  14 drawings with colored pencils.
  7 drawings with blending markers.
  The rest drawings are made with charcoal.
  We assert that the number of charcoal drawings is 4.
-/
theorem charcoal_drawings_count 
  (total_drawings : ℕ) 
  (colored_pencil_drawings : ℕ) 
  (marker_drawings : ℕ) :
  total_drawings = 25 →
  colored_pencil_drawings = 14 →
  marker_drawings = 7 →
  total_drawings - (colored_pencil_drawings + marker_drawings) = 4 := 
  by
    sorry

end charcoal_drawings_count_l212_212494


namespace balance_relationship_l212_212929

theorem balance_relationship (x : ℕ) (hx : 0 ≤ x ∧ x ≤ 5) : 
  ∃ y : ℝ, y = 200 - 36 * x := 
sorry

end balance_relationship_l212_212929


namespace find_m_l212_212321

noncomputable def vector_a : ℝ × ℝ := (-1, 2)
noncomputable def vector_b (m : ℝ) : ℝ × ℝ := (m, 1)
def is_parallel (v1 v2 : ℝ × ℝ) : Prop := ∃ k : ℝ, v1 = (k * v2.1, k * v2.2)

theorem find_m (m : ℝ) :
  is_parallel (vector_a.1 + 2 * m, vector_a.2 + 2 * 1) (2 * vector_a.1 - m, 2 * vector_a.2 - 1) ↔ m = -1 / 2 := 
by {
  sorry
}

end find_m_l212_212321


namespace solve_for_x_l212_212426

theorem solve_for_x (x : ℝ) (h : 3 * x + 8 = -4 * x - 16) : x = -24 / 7 :=
sorry

end solve_for_x_l212_212426


namespace stratified_sampling_l212_212018

theorem stratified_sampling (lathe_A lathe_B total_samples : ℕ) (hA : lathe_A = 56) (hB : lathe_B = 42) (hTotal : total_samples = 14) :
  ∃ (sample_A sample_B : ℕ), sample_A = 8 ∧ sample_B = 6 :=
by
  sorry

end stratified_sampling_l212_212018


namespace stock_percent_change_l212_212847

theorem stock_percent_change (y : ℝ) : 
  let value_after_day1 := 0.85 * y
  let value_after_day2 := 1.25 * value_after_day1
  (value_after_day2 - y) / y * 100 = 6.25 := by
  sorry

end stock_percent_change_l212_212847


namespace sqrt_expression_equality_l212_212060

theorem sqrt_expression_equality :
  Real.sqrt (25 * Real.sqrt (25 * Real.sqrt 25)) = 5 * 5^(3/4) :=
by
  sorry

end sqrt_expression_equality_l212_212060


namespace alice_unanswered_questions_l212_212561

-- Declare variables for the proof
variables (c w u : ℕ)

-- State the problem in Lean
theorem alice_unanswered_questions :
  50 + 5 * c - 2 * w = 100 ∧
  40 + 7 * c - w - u = 120 ∧
  6 * c + 3 * u = 130 ∧
  c + w + u = 25 →
  u = 20 :=
by
  intros h
  sorry

end alice_unanswered_questions_l212_212561


namespace initial_ratio_of_partners_to_associates_l212_212437

theorem initial_ratio_of_partners_to_associates
  (P : ℕ) (A : ℕ)
  (hP : P = 18)
  (h_ratio_after_hiring : ∀ A, 45 + A = 18 * 34) :
  (P : ℤ) / (A : ℤ) = 2 / 63 := 
sorry

end initial_ratio_of_partners_to_associates_l212_212437


namespace jaylen_has_2_cucumbers_l212_212714

-- Definitions based on given conditions
def carrots_jaylen := 5
def bell_peppers_kristin := 2
def green_beans_kristin := 20
def total_vegetables_jaylen := 18

def bell_peppers_jaylen := 2 * bell_peppers_kristin
def green_beans_jaylen := (green_beans_kristin / 2) - 3

def known_vegetables_jaylen := carrots_jaylen + bell_peppers_jaylen + green_beans_jaylen
def cucumbers_jaylen := total_vegetables_jaylen - known_vegetables_jaylen

-- The theorem to prove
theorem jaylen_has_2_cucumbers : cucumbers_jaylen = 2 :=
by
  -- We'll place the proof here
  sorry

end jaylen_has_2_cucumbers_l212_212714


namespace sum_of_transformed_roots_l212_212491

theorem sum_of_transformed_roots (α β γ : ℂ) (h₁ : α^3 - α + 1 = 0) (h₂ : β^3 - β + 1 = 0) (h₃ : γ^3 - γ + 1 = 0) :
  (1 - α) / (1 + α) + (1 - β) / (1 + β) + (1 - γ) / (1 + γ) = 1 :=
by
  sorry

end sum_of_transformed_roots_l212_212491


namespace range_of_a_l212_212178

theorem range_of_a (a : ℝ) :
  (∃ x : ℝ, (x > 0) ∧ (π^x = (a + 1) / (2 - a))) → (1 / 2 < a ∧ a < 2) :=
by
  sorry

end range_of_a_l212_212178


namespace definite_integral_example_l212_212520

theorem definite_integral_example : ∫ x in (0 : ℝ)..(π/2), 2 * x = π^2 / 4 := 
by 
  sorry

end definite_integral_example_l212_212520


namespace intersection_A_B_l212_212172

def A : Set ℕ := {0, 1, 2, 3}
def B : Set ℕ := {x | 0 < x ∧ x ≤ 2}

theorem intersection_A_B : A ∩ B = {1, 2} :=
by
  sorry

end intersection_A_B_l212_212172


namespace positive_integer_solutions_l212_212725

theorem positive_integer_solutions (n m : ℕ) (h : n > 0 ∧ m > 0) : 
  (n + 1) * m = n! + 1 ↔ (n = 1 ∧ m = 1) ∨ (n = 2 ∧ m = 1) ∨ (n = 4 ∧ m = 5) := by
  sorry

end positive_integer_solutions_l212_212725


namespace train_time_first_platform_correct_l212_212398

-- Definitions
variables (L_train L_first_plat L_second_plat : ℕ) (T_second : ℕ) (T_first : ℕ)

-- Given conditions
def length_train := 350
def length_first_platform := 100
def length_second_platform := 250
def time_second_platform := 20
def expected_time_first_platform := 15

-- Derived values
def total_distance_second_platform := length_train + length_second_platform
def speed := total_distance_second_platform / time_second_platform
def total_distance_first_platform := length_train + length_first_platform
def time_first_platform := total_distance_first_platform / speed

-- Proof Statement
theorem train_time_first_platform_correct : 
  time_first_platform = expected_time_first_platform :=
  by
  sorry

end train_time_first_platform_correct_l212_212398


namespace percentage_of_items_sold_l212_212517

theorem percentage_of_items_sold (total_items price_per_item discount_rate debt creditors_balance remaining_balance : ℕ)
  (H1 : total_items = 2000)
  (H2 : price_per_item = 50)
  (H3 : discount_rate = 80)
  (H4 : debt = 15000)
  (H5 : remaining_balance = 3000) :
  (total_items * (price_per_item - (price_per_item * discount_rate / 100)) + remaining_balance = debt + remaining_balance) →
  (remaining_balance / (price_per_item - (price_per_item * discount_rate / 100)) / total_items * 100 = 90) :=
by
  sorry

end percentage_of_items_sold_l212_212517


namespace polynomial_coefficient_sum_l212_212073

theorem polynomial_coefficient_sum
  (a b c d : ℤ)
  (h1 : (x^2 + a * x + b) * (x^2 + c * x + d) = x^4 + 2 * x^3 - 5 * x^2 + 8 * x - 12) :
  a + b + c + d = 6 := 
sorry

end polynomial_coefficient_sum_l212_212073


namespace remainder_of_x50_div_by_x_sub_1_cubed_l212_212848

theorem remainder_of_x50_div_by_x_sub_1_cubed :
  (x^50 % (x-1)^3) = (1225*x^2 - 2500*x + 1276) :=
sorry

end remainder_of_x50_div_by_x_sub_1_cubed_l212_212848


namespace geometric_sequence_a2_a6_l212_212242

variable (a : ℕ → ℝ) (r : ℝ) (a1 : ℝ)
variable (a_geom_seq : ∀ n, a n = a1 * r^(n-1))
variable (h_a4 : a 4 = 4)

theorem geometric_sequence_a2_a6 : a 2 * a 6 = 16 :=
by
  -- Proof goes here
  sorry

end geometric_sequence_a2_a6_l212_212242


namespace cube_without_lid_configurations_l212_212586

-- Introduce assumption for cube without a lid
structure CubeWithoutLid

-- Define the proof statement
theorem cube_without_lid_configurations : 
  ∃ (configs : Nat), (configs = 8) :=
by
  sorry

end cube_without_lid_configurations_l212_212586


namespace married_fraction_l212_212554

variables (M W N : ℕ)

def married_men : Prop := 2 * M = 3 * N
def married_women : Prop := 3 * W = 5 * N
def total_population : ℕ := M + W
def married_population : ℕ := 2 * N

theorem married_fraction (h1: married_men M N) (h2: married_women W N) :
  (married_population N : ℚ) / (total_population M W : ℚ) = 12 / 19 :=
by sorry

end married_fraction_l212_212554


namespace min_value_of_x_l212_212681

-- Define the conditions and state the problem
theorem min_value_of_x (x : ℝ) : (∀ a : ℝ, a > 0 → x^2 < 1 + a) → x ≥ -1 :=
by
  sorry

end min_value_of_x_l212_212681


namespace problem1_problem2_l212_212879

-- Problem 1: Calculation
theorem problem1 :
  (1:Real) - 1^2 + Real.sqrt 12 + Real.sqrt (4 / 3) = -1 + (8 * Real.sqrt 3) / 3 :=
by
  sorry
  
-- Problem 2: Solve the equation 2x^2 - x - 1 = 0
theorem problem2 (x : Real) :
  (2 * x^2 - x - 1 = 0) → (x = -1/2 ∨ x = 1) :=
by
  sorry

end problem1_problem2_l212_212879


namespace max_sin_angle_F1PF2_on_ellipse_l212_212640

theorem max_sin_angle_F1PF2_on_ellipse
  (x y : ℝ)
  (P : ℝ × ℝ)
  (F1 F2 : ℝ × ℝ)
  (h : P ∈ {Q | Q.1^2 / 9 + Q.2^2 / 5 = 1})
  (F1_is_focus : F1 = (-2, 0))
  (F2_is_focus : F2 = (2, 0)) :
  ∃ sin_max, sin_max = 4 * Real.sqrt 5 / 9 := 
sorry

end max_sin_angle_F1PF2_on_ellipse_l212_212640


namespace inv_g_of_43_div_16_l212_212284

noncomputable def g (x : ℚ) : ℚ := (x^3 - 5) / 4

theorem inv_g_of_43_div_16 : g (3 * (↑7)^(1/3) / 2) = 43 / 16 :=
by 
  sorry

end inv_g_of_43_div_16_l212_212284


namespace problem1_problem2_problem3_l212_212108

noncomputable def f (x a : ℝ) : ℝ := abs x * (x - a)

-- 1. Prove a = 0 if f(x) is odd
theorem problem1 (h: ∀ x : ℝ, f (-x) a = -f x a) : a = 0 :=
sorry

-- 2. Prove a ≤ 0 if f(x) is increasing on the interval [0, 2]
theorem problem2 (h: ∀ x y : ℝ, 0 ≤ x → x ≤ y → y ≤ 2 → f x a ≤ f y a) : a ≤ 0 :=
sorry

-- 3. Prove there exists an a < 0 such that the maximum value of f(x) on [-1, 1/2] is 2, and find a = -3
theorem problem3 (h: ∃ a : ℝ, a < 0 ∧ ∀ x : ℝ, -1 ≤ x → x ≤ 1/2 → f x a ≤ 2 ∧ ∃ x : ℝ, -1 ≤ x → x ≤ 1/2 → f x a = 2) : a = -3 :=
sorry

end problem1_problem2_problem3_l212_212108


namespace lcm_of_48_and_14_is_56_l212_212110

theorem lcm_of_48_and_14_is_56 :
  ∀ n : ℕ, (n = 48 ∧ Nat.gcd n 14 = 12) → Nat.lcm n 14 = 56 :=
by
  intro n h
  sorry

end lcm_of_48_and_14_is_56_l212_212110


namespace monthly_growth_rate_l212_212474

-- Definitions and conditions
def initial_height : ℝ := 20
def final_height : ℝ := 80
def months_in_year : ℕ := 12

-- Theorem stating the monthly growth rate
theorem monthly_growth_rate :
  (final_height - initial_height) / (months_in_year : ℝ) = 5 :=
by 
  sorry

end monthly_growth_rate_l212_212474


namespace tailor_trimming_l212_212118

theorem tailor_trimming (x : ℝ) (A B : ℝ)
  (h1 : ∃ (L : ℝ), L = 22) -- Original length of a side of the cloth is 22 feet
  (h2 : 6 = 6) -- Feet trimmed from two opposite edges
  (h3 : ∃ (remaining_area : ℝ), remaining_area = 120) -- 120 square feet of cloth remain after trimming
  (h4 : A = 22 - 2 * 6) -- New length of the side after trimming 6 feet from opposite edges
  (h5 : B = 22 - x) -- New length of the side after trimming x feet from the other two edges
  (h6 : remaining_area = A * B) -- Relationship of the remaining area
: x = 10 :=
by
  sorry

end tailor_trimming_l212_212118


namespace range_3a_2b_l212_212109

theorem range_3a_2b (a b : ℝ) (h : a^2 + b^2 = 4) : 
  -2 * Real.sqrt 13 ≤ 3 * a + 2 * b ∧ 3 * a + 2 * b ≤ 2 * Real.sqrt 13 := 
by 
  sorry

end range_3a_2b_l212_212109


namespace fish_filets_total_l212_212602

/- Define the number of fish caught by each family member -/
def ben_fish : ℕ := 4
def judy_fish : ℕ := 1
def billy_fish : ℕ := 3
def jim_fish : ℕ := 2
def susie_fish : ℕ := 5

/- Define the number of fish thrown back -/
def fish_thrown_back : ℕ := 3

/- Define the number of filets per fish -/
def filets_per_fish : ℕ := 2

/- Calculate the number of fish filets -/
theorem fish_filets_total : ℕ :=
  let total_fish_caught := ben_fish + judy_fish + billy_fish + jim_fish + susie_fish
  let fish_kept := total_fish_caught - fish_thrown_back
  fish_kept * filets_per_fish

example : fish_filets_total = 24 :=
by {
  /- This 'sorry' placeholder indicates that a proof should be here -/
  sorry
}

end fish_filets_total_l212_212602


namespace analytical_expression_of_f_l212_212074

theorem analytical_expression_of_f (f : ℝ → ℝ) :
  (∀ x : ℝ, x ≠ 0 → f (x + 1 / x) = x^2 + 1 / x^2) →
  (∀ y : ℝ, (y ≥ 2 ∨ y ≤ -2) → f y = y^2 - 2) :=
by
  intro h1 y hy
  sorry

end analytical_expression_of_f_l212_212074


namespace inradius_triangle_l212_212739

theorem inradius_triangle (p A : ℝ) (h1 : p = 39) (h2 : A = 29.25) :
  ∃ r : ℝ, A = (1 / 2) * r * p ∧ r = 1.5 := by
  sorry

end inradius_triangle_l212_212739


namespace original_number_l212_212564

theorem original_number (x : ℝ) (h1 : 268 * 74 = 19732) (h2 : x * 0.74 = 1.9832) : x = 2.68 :=
by
  sorry

end original_number_l212_212564


namespace number_of_Al_atoms_l212_212389

def atomic_weight_Al : ℝ := 26.98
def atomic_weight_Br : ℝ := 79.90
def number_of_Br_atoms : ℕ := 3
def molecular_weight : ℝ := 267

theorem number_of_Al_atoms (x : ℝ) : 
  molecular_weight = (atomic_weight_Al * x) + (atomic_weight_Br * number_of_Br_atoms) → 
  x = 1 :=
by
  sorry

end number_of_Al_atoms_l212_212389


namespace exists_triangle_with_sin_angles_l212_212706

theorem exists_triangle_with_sin_angles (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
    (h : a^4 + b^4 + c^4 + 4*a^2*b^2*c^2 = 2 * (a^2*b^2 + a^2*c^2 + b^2*c^2)) : 
    ∃ (α β γ : ℝ), α + β + γ = Real.pi ∧ Real.sin α = a ∧ Real.sin β = b ∧ Real.sin γ = c :=
by
  sorry

end exists_triangle_with_sin_angles_l212_212706


namespace largest_three_digit_geometric_sequence_l212_212959

-- Definitions based on conditions
def is_three_digit_integer (n : ℕ) : Prop := n >= 100 ∧ n < 1000
def digits_distinct (n : ℕ) : Prop := 
  let d₁ := n / 100
  let d₂ := (n / 10) % 10
  let d₃ := n % 10
  d₁ ≠ d₂ ∧ d₂ ≠ d₃ ∧ d₁ ≠ d₃
def geometric_sequence (n : ℕ) : Prop :=
  let d₁ := n / 100
  let d₂ := (n / 10) % 10
  let d₃ := n % 10
  d₁ != 0 ∧ d₂ != 0  ∧ d₃ != 0 ∧ 
  (∃ r: ℚ, d₂ = d₁ * r ∧ d₃ = d₂ * r)

-- Theorem statement
theorem largest_three_digit_geometric_sequence : 
  ∃ n : ℕ, is_three_digit_integer n ∧ digits_distinct n ∧ geometric_sequence n ∧ n = 964 :=
sorry

end largest_three_digit_geometric_sequence_l212_212959


namespace product_of_two_numbers_l212_212032

theorem product_of_two_numbers (x y : ℕ) (h₁ : x + y = 40) (h₂ : x - y = 16) : x * y = 336 :=
sorry

end product_of_two_numbers_l212_212032


namespace fraction_relation_l212_212642

theorem fraction_relation (n d : ℕ) (h1 : (n + 1 : ℚ) / (d + 1) = 3 / 5) (h2 : (n : ℚ) / d = 5 / 9) :
  ∃ k : ℚ, d = k * 2 * n ∧ k = 9 / 10 :=
by
  sorry

end fraction_relation_l212_212642


namespace equal_cost_sharing_l212_212829

variable (X Y Z : ℝ)
variable (h : X < Y ∧ Y < Z)

theorem equal_cost_sharing :
  ∃ (amount : ℝ), amount = (Y + Z - 2 * X) / 3 := 
sorry

end equal_cost_sharing_l212_212829


namespace smallest_divisor_l212_212166

-- Define the given number and the subtracting number
def original_num : ℕ := 378461
def subtract_num : ℕ := 5

-- Define the resulting number after subtraction
def resulting_num : ℕ := original_num - subtract_num

-- Theorem stating that 47307 is the smallest divisor greater than 5 of 378456
theorem smallest_divisor : ∃ d: ℕ, d > 5 ∧ d ∣ resulting_num ∧ ∀ x: ℕ, x > 5 → x ∣ resulting_num → d ≤ x := 
sorry

end smallest_divisor_l212_212166


namespace window_area_properties_l212_212063

theorem window_area_properties
  (AB : ℝ) (AD : ℝ) (ratio : ℝ)
  (h1 : ratio = 3 / 1)
  (h2 : AB = 40)
  (h3 : AD = 3 * AB) :
  (AD * AB / (π * (AB / 2) ^ 2) = 12 / π) ∧
  (AD * AB + π * (AB / 2) ^ 2 = 4800 + 400 * π) :=
by
  -- Proof will go here
  sorry

end window_area_properties_l212_212063


namespace find_x_l212_212097

theorem find_x (a b x: ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : x > 0)
    (h4 : (4 * a)^(4 * b) = a^b * x^(2 * b)) : x = 16 * a^(3 / 2) := by
  sorry

end find_x_l212_212097


namespace turkey_weight_l212_212406

theorem turkey_weight (total_time_minutes roast_time_per_pound number_of_turkeys : ℕ) 
  (h1 : total_time_minutes = 480) 
  (h2 : roast_time_per_pound = 15)
  (h3 : number_of_turkeys = 2) : 
  (total_time_minutes / number_of_turkeys) / roast_time_per_pound = 16 :=
by
  sorry

end turkey_weight_l212_212406


namespace min_value_ineq_l212_212496

noncomputable def f (x : ℝ) : ℝ := abs (Real.log x)

theorem min_value_ineq (a b : ℝ) (h1 : 0 < b) (h2 : b < a) (h3 : f a = f b) :
  (a^2 + b^2) / (a - b) = 2 * Real.sqrt 2 := by
  sorry

end min_value_ineq_l212_212496


namespace solve_for_x_l212_212435

theorem solve_for_x (x : ℝ) (h1 : 8 * x^2 + 8 * x - 2 = 0) (h2 : 32 * x^2 + 68 * x - 8 = 0) : 
    x = 1 / 8 := 
    sorry

end solve_for_x_l212_212435


namespace cubes_even_sum_even_l212_212993

theorem cubes_even_sum_even (p q : ℕ) (h : Even (p^3 - q^3)) : Even (p + q) := sorry

end cubes_even_sum_even_l212_212993


namespace number_of_first_year_students_to_be_sampled_l212_212910

-- Definitions based on the conditions
def total_students_in_each_grade (x : ℕ) : List ℕ := [4*x, 5*x, 5*x, 6*x]
def total_undergraduate_students (x : ℕ) : ℕ := 4*x + 5*x + 5*x + 6*x
def sample_size : ℕ := 300
def sampling_fraction (x : ℕ) : ℚ := sample_size / total_undergraduate_students x
def first_year_sampling (x : ℕ) : ℕ := (4*x) * sample_size / total_undergraduate_students x

-- Statement to prove
theorem number_of_first_year_students_to_be_sampled {x : ℕ} (hx_pos : x > 0) :
  first_year_sampling x = 60 := 
by
  -- skip the proof
  sorry

end number_of_first_year_students_to_be_sampled_l212_212910


namespace arithmetic_sequence_sum_l212_212996

theorem arithmetic_sequence_sum (a : ℕ → ℕ) (S : ℕ → ℕ)
  (h1 : S 3 = 6)
  (h2 : S 9 = 27) :
  S 6 = 15 :=
sorry

end arithmetic_sequence_sum_l212_212996


namespace necessary_condition_not_sufficient_condition_l212_212991

def P (x : ℝ) := x > 0
def Q (x : ℝ) := x > -2

theorem necessary_condition : ∀ x: ℝ, P x → Q x := 
by sorry

theorem not_sufficient_condition : ∃ x: ℝ, Q x ∧ ¬ P x := 
by sorry

end necessary_condition_not_sufficient_condition_l212_212991


namespace parallel_lines_condition_l212_212130

theorem parallel_lines_condition (k1 k2 b : ℝ) (l1 l2 : ℝ → ℝ) (H1 : ∀ x, l1 x = k1 * x + 1)
  (H2 : ∀ x, l2 x = k2 * x + b) : (∀ x, l1 x = l2 x ↔ k1 = k2 ∧ b = 1) → (k1 = k2) ↔ (∀ x, l1 x ≠ l2 x ∧ l1 x - l2 x = 1 - b) := 
by
  sorry

end parallel_lines_condition_l212_212130


namespace number_of_ways_to_assign_roles_l212_212309

theorem number_of_ways_to_assign_roles :
  let men := 6
  let women := 5
  let male_roles := 3
  let female_roles := 2
  let either_gender_roles := 1
  let total_men := men - male_roles
  let total_women := women - female_roles
  (men.choose male_roles) * (women.choose female_roles) * (total_men + total_women).choose either_gender_roles = 14400 := by 
sorry

end number_of_ways_to_assign_roles_l212_212309


namespace four_c_plus_d_l212_212791

theorem four_c_plus_d (c d : ℝ) (h1 : 2 * c = -6) (h2 : c^2 - d = 1) : 4 * c + d = -4 :=
by
  sorry

end four_c_plus_d_l212_212791


namespace number_division_reduction_l212_212928

theorem number_division_reduction (x : ℕ) (h : x / 3 = x - 48) : x = 72 := 
sorry

end number_division_reduction_l212_212928


namespace range_of_f_l212_212630

noncomputable def f (x : ℝ) : ℝ := 2^x
def valid_range (S : Set ℝ) : Prop := ∃ x ∈ Set.Icc (0 : ℝ) (3 : ℝ), f x ∈ S

theorem range_of_f : valid_range (Set.Icc (1 : ℝ) (8 : ℝ)) :=
sorry

end range_of_f_l212_212630


namespace selection_at_most_one_l212_212146

noncomputable def combination (n k : ℕ) : ℕ :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem selection_at_most_one (A B : ℕ) :
  (combination 5 3) - (combination 3 1) = 7 :=
by
  sorry

end selection_at_most_one_l212_212146


namespace total_weight_of_13_gold_bars_l212_212003

theorem total_weight_of_13_gold_bars
    (C1 C2 C3 C4 C5 C6 C7 C8 C9 C10 C11 C12 C13 : ℝ)
    (w12 w13 w23 w45 w67 w89 w1011 w1213 : ℝ)
    (h1 : w12 = C1 + C2)
    (h2 : w13 = C1 + C3)
    (h3 : w23 = C2 + C3)
    (h4 : w45 = C4 + C5)
    (h5 : w67 = C6 + C7)
    (h6 : w89 = C8 + C9)
    (h7 : w1011 = C10 + C11)
    (h8 : w1213 = C12 + C13) :
    C1 + C2 + C3 + C4 + C5 + C6 + C7 + C8 + C9 + C10 + C11 + C12 + C13 = 
    (C1 + C2 + C3) + (C4 + C5) + (C6 + C7) + (C8 + C9) + (C10 + C11) + (C12 + C13) := 
  by
  sorry

end total_weight_of_13_gold_bars_l212_212003


namespace false_prop_range_of_a_l212_212771

theorem false_prop_range_of_a (a : ℝ) :
  (¬ ∀ x : ℝ, 2 * x^2 - 3 * a * x + 9 ≥ 0) ↔ (a < -2 * Real.sqrt 2 ∨ a > 2 * Real.sqrt 2) :=
by
  sorry

end false_prop_range_of_a_l212_212771


namespace find_sum_a7_a8_l212_212869

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ (a1 q : ℝ), ∀ n : ℕ, a n = a1 * q ^ n

variable (a : ℕ → ℝ)

axiom h_geom : geometric_sequence a
axiom h1 : a 0 + a 1 = 16
axiom h2 : a 2 + a 3 = 32

theorem find_sum_a7_a8 : a 6 + a 7 = 128 :=
sorry

end find_sum_a7_a8_l212_212869


namespace sqrt_7_estimate_l212_212292

theorem sqrt_7_estimate : (2 : Real) < Real.sqrt 7 ∧ Real.sqrt 7 < 3 → (Real.sqrt 7 - 1) / 2 < 1 := 
by
  intro h
  sorry

end sqrt_7_estimate_l212_212292


namespace cassie_nails_claws_total_l212_212269

theorem cassie_nails_claws_total :
  let dogs := 4
  let parrots := 8
  let cats := 2
  let rabbits := 6
  let lizards := 5
  let tortoises := 3

  let dog_nails := dogs * 4 * 4

  let normal_parrots := 6
  let parrot_with_extra_toe := 1
  let parrot_missing_toe := 1
  let parrot_claws := (normal_parrots * 2 * 3) + (parrot_with_extra_toe * 2 * 4) + (parrot_missing_toe * 2 * 2)

  let normal_cats := 1
  let deformed_cat := 1
  let cat_toes := (1 * 4 * 5) + (1 * 4 * 4) + 1 

  let normal_rabbits := 5
  let deformed_rabbit := 1
  let rabbit_nails := (normal_rabbits * 4 * 9) + (3 * 9 + 2)

  let normal_lizards := 4
  let deformed_lizard := 1
  let lizard_toes := (normal_lizards * 4 * 5) + (deformed_lizard * 4 * 4)
  
  let normal_tortoises := 1
  let tortoise_with_extra_claw := 1
  let tortoise_missing_claw := 1
  let tortoise_claws := (normal_tortoises * 4 * 4) + (3 * 4 + 5) + (3 * 4 + 3)

  let total_nails_claws := dog_nails + parrot_claws + cat_toes + rabbit_nails + lizard_toes + tortoise_claws

  total_nails_claws = 524 :=
by
  sorry

end cassie_nails_claws_total_l212_212269


namespace triangle_PR_eq_8_l212_212552

open Real

theorem triangle_PR_eq_8 (P Q R M : ℝ) 
  (PQ QR PM : ℝ) 
  (hPQ : PQ = 6) (hQR : QR = 10) (hPM : PM = 5) 
  (M_midpoint : M = (Q + R) / 2) :
  dist P R = 8 :=
by
  sorry

end triangle_PR_eq_8_l212_212552


namespace at_least_one_of_p_or_q_true_l212_212769

variable (p q : Prop)

theorem at_least_one_of_p_or_q_true (h : ¬(p ∨ q) = false) : p ∨ q :=
by 
  sorry

end at_least_one_of_p_or_q_true_l212_212769


namespace checkerboard_problem_l212_212084

def is_valid_square (size : ℕ) : Prop :=
  size = 4 ∨ size = 5 ∨ size = 6 ∨ size = 7 ∨ size = 8 ∨ size = 9 ∨ size = 10

def check_10_by_10 : ℕ :=
  24 + 36 + 25 + 16 + 9 + 4 + 1

theorem checkerboard_problem :
  ∀ size : ℕ, ( size = 4 ∨ size = 5 ∨ size = 6 ∨ size = 7 ∨ size = 8 ∨ size = 9 ∨ size = 10 ) →
  check_10_by_10 = 115 := 
sorry

end checkerboard_problem_l212_212084


namespace minimal_degree_g_l212_212369

theorem minimal_degree_g {f g h : Polynomial ℝ} 
  (h_eq : 2 * f + 5 * g = h)
  (deg_f : f.degree = 6)
  (deg_h : h.degree = 10) : 
  g.degree = 10 :=
sorry

end minimal_degree_g_l212_212369


namespace range_of_a_values_l212_212548

noncomputable def range_of_a (a : ℝ) : Prop :=
  ∀ x : ℝ, a * x^2 - |x + 1| + 3 * a ≥ 0

theorem range_of_a_values (a : ℝ) : range_of_a a ↔ a ≥ 1/2 :=
by
  sorry

end range_of_a_values_l212_212548


namespace number_of_passed_candidates_l212_212526

theorem number_of_passed_candidates :
  ∀ (P F : ℕ),
  (P + F = 500) →
  (P * 80 + F * 15 = 500 * 60) →
  P = 346 :=
by
  intros P F h1 h2
  sorry

end number_of_passed_candidates_l212_212526


namespace grassy_plot_width_l212_212749

theorem grassy_plot_width (L : ℝ) (P : ℝ) (C : ℝ) (cost_per_sqm : ℝ) (W : ℝ) : 
  L = 110 →
  P = 2.5 →
  C = 510 →
  cost_per_sqm = 0.6 →
  (115 * (W + 5) - 110 * W = C / cost_per_sqm) →
  W = 55 :=
by
  intros hL hP hC hcost_per_sqm harea
  sorry

end grassy_plot_width_l212_212749


namespace fish_game_teams_l212_212780

noncomputable def number_of_possible_teams (n : ℕ) : ℕ := 
  if n = 6 then 5 else sorry

theorem fish_game_teams : number_of_possible_teams 6 = 5 := by
  unfold number_of_possible_teams
  rfl

end fish_game_teams_l212_212780


namespace slower_train_speed_l212_212147

theorem slower_train_speed (faster_speed : ℝ) (time_passed : ℝ) (train_length : ℝ) (slower_speed: ℝ) :
  faster_speed = 50 ∧ time_passed = 15 ∧ train_length = 75 →
  slower_speed = 32 :=
by
  intro h
  sorry

end slower_train_speed_l212_212147


namespace find_a_7_l212_212621

-- Define the arithmetic sequence conditions
variable {a : ℕ → ℤ} -- The sequence a_n
variable (a_4_eq : a 4 = 4)
variable (a_3_a_8_eq : a 3 + a 8 = 5)

-- Prove that a_7 = 1
theorem find_a_7 : a 7 = 1 := by
  sorry

end find_a_7_l212_212621


namespace math_problem_l212_212952

theorem math_problem : 
  ( - (1 / 12 : ℚ) + (1 / 3 : ℚ) - (1 / 2 : ℚ) ) / ( - (1 / 18 : ℚ) ) = 4.5 := 
by
  sorry

end math_problem_l212_212952


namespace alberto_bjorn_distance_difference_l212_212162

-- Definitions based on given conditions
def alberto_speed : ℕ := 12  -- miles per hour
def bjorn_speed : ℕ := 10    -- miles per hour
def total_time : ℕ := 6      -- hours
def bjorn_rest_time : ℕ := 1 -- hours

def alberto_distance : ℕ := alberto_speed * total_time
def bjorn_distance : ℕ := bjorn_speed * (total_time - bjorn_rest_time)

-- The statement to prove
theorem alberto_bjorn_distance_difference :
  (alberto_distance - bjorn_distance) = 22 :=
by
  sorry

end alberto_bjorn_distance_difference_l212_212162


namespace trig_identity_solution_l212_212322

noncomputable def solve_trig_identity (x : ℝ) : Prop :=
  (∃ k : ℤ, x = (Real.pi / 8 * (4 * k + 1))) ∧
  (Real.sin (2 * x))^4 + (Real.cos (2 * x))^4 = Real.sin (2 * x) * Real.cos (2 * x)

theorem trig_identity_solution (x : ℝ) :
  solve_trig_identity x :=
sorry

end trig_identity_solution_l212_212322


namespace revenue_correct_l212_212886

def calculate_revenue : Real :=
  let pumpkin_pie_revenue := 4 * 8 * 5
  let custard_pie_revenue := 5 * 6 * 6
  let apple_pie_revenue := 3 * 10 * 4
  let pecan_pie_revenue := 2 * 12 * 7
  let cookie_revenue := 15 * 2
  let red_velvet_revenue := 6 * 8 * 9
  pumpkin_pie_revenue + custard_pie_revenue + apple_pie_revenue + pecan_pie_revenue + cookie_revenue + red_velvet_revenue

theorem revenue_correct : calculate_revenue = 1090 :=
by
  sorry

end revenue_correct_l212_212886


namespace student_tickets_sold_l212_212356

theorem student_tickets_sold
  (A S : ℕ)
  (h1 : A + S = 846)
  (h2 : 6 * A + 3 * S = 3846) :
  S = 410 :=
sorry

end student_tickets_sold_l212_212356


namespace angle_between_slant_height_and_base_l212_212102

theorem angle_between_slant_height_and_base (R : ℝ) (diam_base_upper diam_base_lower : ℝ) 
(h1 : diam_base_upper + diam_base_lower = 5 * R)
: ∃ θ : ℝ, θ = Real.arcsin (4 / 5) := 
sorry

end angle_between_slant_height_and_base_l212_212102


namespace remy_gallons_l212_212806

noncomputable def gallons_used (R : ℝ) : ℝ :=
  let remy := 3 * R + 1
  let riley := (R + remy) - 2
  let ronan := riley / 2
  R + remy + riley + ronan

theorem remy_gallons : ∃ R : ℝ, gallons_used R = 60 ∧ (3 * R + 1) = 18.85 :=
by
  sorry

end remy_gallons_l212_212806


namespace find_y_ratio_l212_212471

variable {R : Type} [LinearOrderedField R]
variables (x y : R → R) (x1 x2 y1 y2 : R)

-- Condition: x is inversely proportional to y, so xy is constant.
def inversely_proportional (x y : R → R) : Prop := ∀ (a b : R), x a * y a = x b * y b

-- Condition: ∀ nonzero x values, we have these specific ratios
variable (h_inv_prop : inversely_proportional x y)
variable (h_ratio_x : x1 ≠ 0 ∧ x2 ≠ 0 ∧ x1 / x2 = 4 / 5)
variable (h_nonzero_y : y1 ≠ 0 ∧ y2 ≠ 0)

-- Claim to prove
theorem find_y_ratio : (y1 / y2) = 5 / 4 :=
by
  sorry

end find_y_ratio_l212_212471


namespace tangent_sum_l212_212253

theorem tangent_sum (tan : ℝ → ℝ)
  (h1 : ∀ A B, tan (A + B) = (tan A + tan B) / (1 - tan A * tan B))
  (h2 : tan 60 = Real.sqrt 3) :
  tan 20 + tan 40 + Real.sqrt 3 * tan 20 * tan 40 = Real.sqrt 3 := 
by
  sorry

end tangent_sum_l212_212253


namespace jake_peaches_count_l212_212537

-- Define Jill's peaches
def jill_peaches : ℕ := 5

-- Define Steven's peaches based on the condition that Steven has 18 more peaches than Jill
def steven_peaches : ℕ := jill_peaches + 18

-- Define Jake's peaches based on the condition that Jake has 6 fewer peaches than Steven
def jake_peaches : ℕ := steven_peaches - 6

-- The theorem to prove that Jake has 17 peaches
theorem jake_peaches_count : jake_peaches = 17 := by
  sorry

end jake_peaches_count_l212_212537


namespace triangle_perimeter_l212_212729

theorem triangle_perimeter (a b c : ℕ) (ha : a = 14) (hb : b = 8) (hc : c = 9) : a + b + c = 31 := 
by
  sorry

end triangle_perimeter_l212_212729


namespace hexagon_rectangle_ratio_l212_212479

theorem hexagon_rectangle_ratio:
  ∀ (h w : ℕ), 
  (6 * h = 24) → (2 * (2 * w + w) = 24) → 
  (h / w = 1) := by
  intros h w
  intro hex_condition
  intro rect_condition
  sorry

end hexagon_rectangle_ratio_l212_212479


namespace max_a1_l212_212705

theorem max_a1 (a : ℕ → ℝ) (h_pos : ∀ n : ℕ, n > 0 → a n > 0)
  (h_eq : ∀ n : ℕ, n > 0 → 2 + a n * (a (n + 1) - a (n - 1)) = 0 ∨ 2 - a n * (a (n + 1) - a (n - 1)) = 0)
  (h_a20 : a 20 = a 20) :
  ∃ max_a1 : ℝ, max_a1 = 512 := 
sorry

end max_a1_l212_212705


namespace fourth_equation_pattern_l212_212541

theorem fourth_equation_pattern :
  36^2 + 37^2 + 38^2 + 39^2 + 40^2 = 41^2 + 42^2 + 43^2 + 44^2 :=
by
  sorry

end fourth_equation_pattern_l212_212541


namespace find_rate_percent_l212_212179

-- Given conditions as definitions
def SI : ℕ := 128
def P : ℕ := 800
def T : ℕ := 4

-- Define the formula for Simple Interest
def simple_interest (P R T : ℕ) : ℕ := (P * R * T) / 100

-- Define the rate percent we need to prove
def rate_percent : ℕ := 4

-- The theorem statement we need to prove
theorem find_rate_percent (h1 : simple_interest P rate_percent T = SI) : rate_percent = 4 := 
by sorry

end find_rate_percent_l212_212179


namespace continuous_function_nondecreasing_l212_212258

open Set

variable {α : Type*} [LinearOrder ℝ] [Preorder ℝ]

theorem continuous_function_nondecreasing
  (f : (ℝ)→ ℝ) 
  (h_cont : ContinuousOn f (Ioi 0))
  (h_seq : ∀ x > 0, Monotone (fun n : ℕ => f (n*x))):
  ∀ x y, x ≤ y → f x ≤ f y := 
sorry

end continuous_function_nondecreasing_l212_212258


namespace negation_of_proposition_l212_212133

theorem negation_of_proposition : 
  ¬(∀ x : ℝ, x^2 + x + 1 > 0) ↔ ∃ x : ℝ, x^2 + x + 1 ≤ 0 :=
by
  sorry

end negation_of_proposition_l212_212133


namespace gold_coins_equality_l212_212943

theorem gold_coins_equality (pouches : List ℕ) 
  (h_pouches_length : pouches.length = 9)
  (h_pouches_sum : pouches.sum = 60)
  : (∃ s_2 : List (List ℕ), s_2.length = 2 ∧ ∀ l ∈ s_2, l.sum = 30) ∧
    (∃ s_3 : List (List ℕ), s_3.length = 3 ∧ ∀ l ∈ s_3, l.sum = 20) ∧
    (∃ s_4 : List (List ℕ), s_4.length = 4 ∧ ∀ l ∈ s_4, l.sum = 15) ∧
    (∃ s_5 : List (List ℕ), s_5.length = 5 ∧ ∀ l ∈ s_5, l.sum = 12) :=
sorry

end gold_coins_equality_l212_212943


namespace find_multiplicand_l212_212839

theorem find_multiplicand (m : ℕ) 
( h : 32519 * m = 325027405 ) : 
m = 9995 := 
by {
  sorry
}

end find_multiplicand_l212_212839


namespace perfect_square_octal_last_digit_l212_212676

theorem perfect_square_octal_last_digit (a b c : ℕ) (n : ℕ) (h1 : a ≠ 0) (h2 : (abc:ℕ) = n^2) :
  c = 1 :=
sorry

end perfect_square_octal_last_digit_l212_212676


namespace range_of_a_l212_212330

theorem range_of_a (a : ℝ) (f : ℝ → ℝ) (h : ∀ x, f x = abs (x - 2) + abs (x + a) ∧ f x ≥ 3) : a ≤ -5 ∨ a ≥ 1 :=
sorry

end range_of_a_l212_212330


namespace tan_A_tan_B_eq_one_third_l212_212636

theorem tan_A_tan_B_eq_one_third (A B C : ℕ) (hC : C = 120) (hSum : Real.tan A + Real.tan B = 2 * Real.sqrt 3 / 3) : 
  Real.tan A * Real.tan B = 1 / 3 := 
by
  sorry

end tan_A_tan_B_eq_one_third_l212_212636


namespace cost_price_of_product_l212_212189

theorem cost_price_of_product (x y : ℝ)
  (h1 : 0.8 * y - x = 120)
  (h2 : 0.6 * y - x = -20) :
  x = 440 := sorry

end cost_price_of_product_l212_212189


namespace correct_calculation_l212_212346

/-- Conditions for the given calculations -/
def cond_a : Prop := (-2) ^ 3 = 8
def cond_b : Prop := (-3) ^ 2 = -9
def cond_c : Prop := -(3 ^ 2) = -9
def cond_d : Prop := (-2) ^ 2 = 4

/-- Prove that the correct calculation among the given is -3^2 = -9 -/
theorem correct_calculation : cond_c :=
by sorry

end correct_calculation_l212_212346


namespace mr_lee_broke_even_l212_212941

theorem mr_lee_broke_even (sp1 sp2 : ℝ) (p1_loss2 : ℝ) (c1 c2 : ℝ) (h1 : sp1 = 1.50) (h2 : sp2 = 1.50) 
    (h3 : c1 = sp1 / 1.25) (h4 : c2 = sp2 / 0.8333) (h5 : p1_loss2 = (sp1 - c1) + (sp2 - c2)) : 
  p1_loss2 = 0 :=
by 
  sorry

end mr_lee_broke_even_l212_212941


namespace modulo_remainder_l212_212158

theorem modulo_remainder : (7^2023) % 17 = 15 := 
by 
  sorry

end modulo_remainder_l212_212158


namespace minimum_n_for_candy_purchases_l212_212753

theorem minimum_n_for_candy_purchases' {o s p : ℕ} (h1 : 9 * o = 10 * s) (h2 : 9 * o = 20 * p) : 
  ∃ n : ℕ, 30 * n = 180 ∧ ∀ m : ℕ, (30 * m = 9 * o) → n ≤ m :=
by sorry

end minimum_n_for_candy_purchases_l212_212753


namespace at_least_one_nonnegative_l212_212888

theorem at_least_one_nonnegative (x y z : ℝ) : 
  (x^2 + y + 1/4 ≥ 0) ∨ (y^2 + z + 1/4 ≥ 0) ∨ (z^2 + x + 1/4 ≥ 0) :=
sorry

end at_least_one_nonnegative_l212_212888


namespace unique_positive_integer_solution_l212_212411

theorem unique_positive_integer_solution :
  ∃! n : ℕ, n > 0 ∧ ∃ k : ℕ, n^4 - n^3 + 3*n^2 + 5 = k^2 :=
by
  sorry

end unique_positive_integer_solution_l212_212411


namespace developer_lots_l212_212933

theorem developer_lots (acres : ℕ) (cost_per_acre : ℕ) (lot_price : ℕ) 
  (h1 : acres = 4) 
  (h2 : cost_per_acre = 1863) 
  (h3 : lot_price = 828) : 
  ((acres * cost_per_acre) / lot_price) = 9 := 
  by
    sorry

end developer_lots_l212_212933


namespace number_is_45_percent_of_27_l212_212695

theorem number_is_45_percent_of_27 (x : ℝ) (h : 27 / x = 45 / 100) : x = 60 := 
by
  sorry

end number_is_45_percent_of_27_l212_212695


namespace solve_for_x_l212_212789

theorem solve_for_x (x : ℝ) (h : (2 * x + 7) / 7 = 13) : x = 42 :=
sorry

end solve_for_x_l212_212789


namespace sixth_number_is_eight_l212_212267

/- 
  The conditions are:
  1. The sequence is an increasing list of consecutive integers.
  2. The 3rd and 4th numbers add up to 11.
  We need to prove that the 6th number is 8.
-/

theorem sixth_number_is_eight (n : ℕ) (h : n + (n + 1) = 11) : (n + 3) = 8 :=
by
  sorry

end sixth_number_is_eight_l212_212267


namespace expression_equivalence_l212_212820

theorem expression_equivalence :
  (4 + 5) * (4^2 + 5^2) * (4^4 + 5^4) * (4^8 + 5^8) * (4^16 + 5^16) * (4^32 + 5^32) * (4^64 + 5^64) * (4^128 + 5^128) = 5^256 - 4^256 :=
by sorry

end expression_equivalence_l212_212820


namespace total_apples_picked_l212_212655

theorem total_apples_picked (Mike_apples Nancy_apples Keith_apples : ℕ)
  (hMike : Mike_apples = 7)
  (hNancy : Nancy_apples = 3)
  (hKeith : Keith_apples = 6) :
  Mike_apples + Nancy_apples + Keith_apples = 16 :=
by
  sorry

end total_apples_picked_l212_212655


namespace cubic_coeff_relationship_l212_212842

theorem cubic_coeff_relationship (a b c d u v w : ℝ) 
  (h_eq : a * (u^3) + b * (u^2) + c * u + d = 0)
  (h_vieta1 : u + v + w = -(b / a)) 
  (h_vieta2 : u * v + u * w + v * w = c / a) 
  (h_vieta3 : u * v * w = -d / a) 
  (h_condition : u + v = u * v) :
  (c + d) * (b + c + d) = a * d :=
by 
  sorry

end cubic_coeff_relationship_l212_212842


namespace tetrahedron_side_length_l212_212947

theorem tetrahedron_side_length (s : ℝ) (area : ℝ) (d : ℝ) :
  area = 16 → s^2 = area → d = s * Real.sqrt 2 → 4 * Real.sqrt 2 = d :=
by
  intros _ h1 h2
  sorry

end tetrahedron_side_length_l212_212947


namespace fraction_calculation_correct_l212_212957

noncomputable def calculate_fraction : ℚ :=
  let numerator := (1 / 2) - (1 / 3)
  let denominator := (3 / 4) + (1 / 8)
  numerator / denominator

theorem fraction_calculation_correct : calculate_fraction = 4 / 21 := 
  by
    sorry

end fraction_calculation_correct_l212_212957


namespace Nell_cards_difference_l212_212268

-- Definitions
def initial_baseball_cards : ℕ := 438
def initial_ace_cards : ℕ := 18
def given_ace_cards : ℕ := 55
def given_baseball_cards : ℕ := 178

-- Theorem statement
theorem Nell_cards_difference :
  given_baseball_cards - given_ace_cards = 123 := 
by
  sorry

end Nell_cards_difference_l212_212268


namespace maximize_savings_l212_212230

-- Definitions for the conditions
def initial_amount : ℝ := 15000

def discount_option1 (amount : ℝ) : ℝ := 
  let after_first : ℝ := amount * 0.75
  let after_second : ℝ := after_first * 0.90
  after_second * 0.95

def discount_option2 (amount : ℝ) : ℝ := 
  let after_first : ℝ := amount * 0.70
  let after_second : ℝ := after_first * 0.90
  after_second * 0.90

-- Theorem to compare the final amounts
theorem maximize_savings : discount_option2 initial_amount < discount_option1 initial_amount := 
  sorry

end maximize_savings_l212_212230


namespace dan_balloons_l212_212560

theorem dan_balloons (fred_balloons sam_balloons total_balloons dan_balloons : ℕ) 
  (h₁ : fred_balloons = 10) 
  (h₂ : sam_balloons = 46) 
  (h₃ : total_balloons = 72) : 
  dan_balloons = total_balloons - (fred_balloons + sam_balloons) :=
by
  sorry

end dan_balloons_l212_212560


namespace least_number_of_square_tiles_l212_212028

-- Definitions based on conditions
def room_length_cm : ℕ := 672
def room_width_cm : ℕ := 432

-- Correct Answer is 126 tiles

-- Lean Statement for the proof problem
theorem least_number_of_square_tiles : 
  ∃ tile_size tiles_needed, 
    (tile_size = Int.gcd room_length_cm room_width_cm) ∧
    (tiles_needed = (room_length_cm / tile_size) * (room_width_cm / tile_size)) ∧
    tiles_needed = 126 := 
by
  sorry

end least_number_of_square_tiles_l212_212028


namespace part1_part2_part3_l212_212170

variable {α : Type} [LinearOrderedField α]

noncomputable def f (x : α) : α := sorry  -- as we won't define it explicitly, we use sorry

axiom f_conditions : ∀ (u v : α), - 1 ≤ u ∧ u ≤ 1 ∧ - 1 ≤ v ∧ v ≤ 1 → |f u - f v| ≤ |u - v|
axiom f_endpoints : f (-1 : α) = 0 ∧ f (1 : α) = 0

theorem part1 (x : α) (hx : -1 ≤ x ∧ x ≤ 1) : x - 1 ≤ f x ∧ f x ≤ 1 - x := by
  have hf : ∀ (u v : α), -1 ≤ u ∧ u ≤ 1 ∧ -1 ≤ v ∧ v ≤ 1 → |f u - f v| ≤ |u - v| := f_conditions
  sorry

theorem part2 (u v : α) (huv : -1 ≤ u ∧ u ≤ 1 ∧ -1 ≤ v ∧ v ≤ 1) : |f u - f v| ≤ 1 := by
  have hf : ∀ (u v : α), -1 ≤ u ∧ u ≤ 1 ∧ -1 ≤ v ∧ v ≤ 1 → |f u - f v| ≤ |u - v| := f_conditions
  sorry

theorem part3 : ¬ ∃ (f : α → α), (∀ (u v : α), - 1 ≤ u ∧ u ≤ 1 ∧ - 1 ≤ v ∧ v ≤ 1 → |f u - f v| ≤ |u - v| ∧ f (-1 : α) = 0 ∧ f (1 : α) = 0 ∧
  (∀ (x : α), - 1 ≤ x ∧ x ≤ 1 → f (- x) = - f x) ∧ -- odd function condition
  (∀ (u v : α), 0 ≤ u ∧ u ≤ 1/2 ∧ 0 ≤ v ∧ v ≤ 1/2 → |f u - f v| < |u - v|) ∧
  (∀ (u v : α), 1/2 ≤ u ∧ u ≤ 1 ∧ 1/2 ≤ v ∧ v ≤ 1 → |f u - f v| = |u - v|)) := by
  sorry

end part1_part2_part3_l212_212170


namespace abc_value_l212_212264

noncomputable def find_abc (a b c : ℝ) (h1 : a * (b + c) = 152) (h2 : b * (c + a) = 162) (h3 : c * (a + b) = 170) : ℝ :=
  a * b * c

theorem abc_value (a b c : ℝ) (h1 : a * (b + c) = 152) (h2 : b * (c + a) = 162) (h3 : c * (a + b) = 170) :
  a * b * c = 720 :=
by
  -- We skip the proof by providing sorry.
  sorry

end abc_value_l212_212264


namespace brendas_age_l212_212891

theorem brendas_age (A B J : ℝ) 
  (h1 : A = 4 * B)
  (h2 : J = B + 7)
  (h3 : A = J) : 
  B = 7 / 3 := 
by 
  sorry

end brendas_age_l212_212891


namespace set_union_l212_212713

def A : Set ℝ := {x | -1 ≤ x ∧ x ≤ 2}
def B : Set ℝ := {x | 0 ≤ x ∧ x ≤ 4}

theorem set_union : A ∪ B = {x | -1 ≤ x ∧ x ≤ 4} :=
by
  sorry

end set_union_l212_212713


namespace transformed_curve_l212_212810

theorem transformed_curve (x y : ℝ) :
  (y * Real.cos x + 2 * y - 1 = 0) →
  (y - 1) * Real.sin x + 2 * y - 3 = 0 :=
by
  intro h
  sorry

end transformed_curve_l212_212810


namespace unique_symmetric_matrix_pair_l212_212674

theorem unique_symmetric_matrix_pair (a b : ℝ) :
  (∃! M : Matrix (Fin 2) (Fin 2) ℝ, M = M.transpose ∧ Matrix.trace M = a ∧ Matrix.det M = b)
  ↔ (∃ t : ℝ, a = 2 * t ∧ b = t^2) :=
by
  sorry

end unique_symmetric_matrix_pair_l212_212674


namespace seq_max_min_terms_l212_212300

noncomputable def a (n: ℕ) : ℝ := 1 / (2^n - 18)

theorem seq_max_min_terms : (∀ (n : ℕ), n > 5 → a 5 > a n) ∧ (∀ (n : ℕ), n ≠ 4 → a 4 < a n) :=
by 
  sorry

end seq_max_min_terms_l212_212300


namespace graph_passes_through_fixed_point_l212_212645

-- Define the linear function given in the conditions
def linearFunction (k x y : ℝ) : ℝ :=
  (2 * k - 1) * x - (k + 3) * y - (k - 11)

-- Define the fixed point (2, 3)
def fixedPoint : ℝ × ℝ :=
  (2, 3)

-- State the theorem that the graph of the linear function always passes through the fixed point 
theorem graph_passes_through_fixed_point :
  ∀ k : ℝ, linearFunction k fixedPoint.1 fixedPoint.2 = 0 :=
by sorry  -- proof skipped

end graph_passes_through_fixed_point_l212_212645


namespace sum_of_reciprocals_l212_212169

theorem sum_of_reciprocals (x y : ℝ) (h : x + y = 6 * x * y) (hx : x ≠ 0) (hy : y ≠ 0) : 
  (1 / x) + (1 / y) = 2 := 
by
  sorry

end sum_of_reciprocals_l212_212169


namespace ratio_problem_l212_212357

open Classical 

variables {q r s t u : ℚ}

theorem ratio_problem (h1 : q / r = 8) (h2 : s / r = 5) (h3 : s / t = 1 / 4) (h4 : u / t = 3) :
  u / q = 15 / 2 :=
by
  sorry

end ratio_problem_l212_212357


namespace pastries_left_l212_212500

def pastries_baked : ℕ := 4 + 29
def pastries_sold : ℕ := 9

theorem pastries_left : pastries_baked - pastries_sold = 24 :=
by
  -- assume pastries_baked = 33
  -- assume pastries_sold = 9
  -- prove 33 - 9 = 24
  sorry

end pastries_left_l212_212500


namespace max_sides_of_convex_polygon_with_arithmetic_angles_l212_212671

theorem max_sides_of_convex_polygon_with_arithmetic_angles :
  ∀ (n : ℕ), (∃ α : ℝ, α > 0 ∧ α + (n - 1) * 1 < 180) → 
  n * (2 * α + (n - 1)) / 2 = (n - 2) * 180 → n ≤ 27 :=
by
  sorry

end max_sides_of_convex_polygon_with_arithmetic_angles_l212_212671


namespace non_allergic_children_l212_212302

theorem non_allergic_children (T : ℕ) (h1 : T / 2 = n) (h2 : ∀ m : ℕ, 10 = m) (h3 : ∀ k : ℕ, 10 = k) :
  10 = 10 :=
by
  sorry

end non_allergic_children_l212_212302


namespace isabella_total_haircut_length_l212_212245

theorem isabella_total_haircut_length :
  (18 - 14) + (14 - 9) = 9 := 
sorry

end isabella_total_haircut_length_l212_212245


namespace base_length_of_vessel_l212_212237

def volume_of_cube (edge : ℝ) := edge^3

def volume_of_displaced_water (L width rise : ℝ) := L * width * rise

theorem base_length_of_vessel (edge width rise L : ℝ) 
  (h1 : edge = 15) (h2 : width = 15) (h3 : rise = 11.25) 
  (h4 : volume_of_displaced_water L width rise = volume_of_cube edge) : 
  L = 20 :=
by
  sorry

end base_length_of_vessel_l212_212237


namespace find_original_acid_amount_l212_212091

noncomputable def original_amount_of_acid (a w : ℝ) : Prop :=
  3 * a = w + 2 ∧ 5 * a = 3 * w - 10

theorem find_original_acid_amount (a w : ℝ) (h : original_amount_of_acid a w) : a = 4 :=
by
  sorry

end find_original_acid_amount_l212_212091


namespace sum_of_integers_remainders_l212_212446

theorem sum_of_integers_remainders (a b c : ℕ) :
  (a % 15 = 11) →
  (b % 15 = 13) →
  (c % 15 = 14) →
  ((a + b + c) % 15 = 8) ∧ ((a + b + c) % 10 = 8) :=
by
  sorry

end sum_of_integers_remainders_l212_212446


namespace water_pumping_problem_l212_212757

theorem water_pumping_problem :
  let pumpA_rate := 300 -- gallons per hour
  let pumpB_rate := 500 -- gallons per hour
  let combined_rate := pumpA_rate + pumpB_rate -- Combined rate per hour
  let time_duration := 1 / 2 -- Time in hours (30 minutes)
  combined_rate * time_duration = 400 := -- Total volume in gallons
by
  -- Lean proof would go here
  sorry

end water_pumping_problem_l212_212757


namespace number_of_classes_l212_212986

theorem number_of_classes (sheets_per_class_per_day : ℕ) 
                          (total_sheets_per_week : ℕ) 
                          (school_days_per_week : ℕ) 
                          (h1 : sheets_per_class_per_day = 200) 
                          (h2 : total_sheets_per_week = 9000) 
                          (h3 : school_days_per_week = 5) : 
                          total_sheets_per_week / school_days_per_week / sheets_per_class_per_day = 9 :=
by {
  -- Proof steps would go here
  sorry
}

end number_of_classes_l212_212986


namespace second_smallest_prime_perimeter_l212_212665

def is_prime (n : ℕ) : Prop := 
  n > 1 ∧ ∀ m : ℕ, m > 1 → m ∣ n → m = n

def scalene_triangle (a b c : ℕ) : Prop := 
  a ≠ b ∧ b ≠ c ∧ a ≠ c

def prime_perimeter (a b c : ℕ) : Prop := 
  is_prime (a + b + c)

def different_primes (a b c : ℕ) : Prop := 
  is_prime a ∧ is_prime b ∧ is_prime c

theorem second_smallest_prime_perimeter :
  ∃ (a b c : ℕ), 
  scalene_triangle a b c ∧ 
  different_primes a b c ∧ 
  prime_perimeter a b c ∧ 
  a + b + c = 29 := 
sorry

end second_smallest_prime_perimeter_l212_212665


namespace competition_order_l212_212222

variable (A B C D : ℕ)

-- Conditions as given in the problem
axiom cond1 : B + D = 2 * A
axiom cond2 : A + C < B + D
axiom cond3 : A < B + C

-- The desired proof statement
theorem competition_order : D > B ∧ B > A ∧ A > C :=
by
  sorry

end competition_order_l212_212222


namespace b_n_plus_1_eq_2a_n_l212_212558

/-- Definition of binary sequences of length n that do not contain 0, 1, 0 -/
def a_n (n : ℕ) : ℕ := -- specify the actual counting function, placeholder below
  sorry

/-- Definition of binary sequences of length n that do not contain 0, 0, 1, 1 or 1, 1, 0, 0 -/
def b_n (n : ℕ) : ℕ := -- specify the actual counting function, placeholder below
  sorry

/-- Proof statement that for all positive integers n, b_{n+1} = 2a_n -/
theorem b_n_plus_1_eq_2a_n (n : ℕ) (hn : 0 < n) : b_n (n + 1) = 2 * a_n n :=
  sorry

end b_n_plus_1_eq_2a_n_l212_212558


namespace tangent_line_at_zero_range_of_a_l212_212020

noncomputable def f (a x : ℝ) : ℝ := Real.exp x - a * Real.sin x - 1

theorem tangent_line_at_zero (h : ∀ x, f 1 x = Real.exp x - Real.sin x - 1) :
  ∀ x, Real.exp x - Real.sin x - 1 = f 1 x :=
by
  sorry

theorem range_of_a (h : ∀ x, f a x ≥ 0) : a ∈ Set.Iic 1 :=
by
  sorry

end tangent_line_at_zero_range_of_a_l212_212020


namespace nth_term_206_l212_212434

noncomputable def geometric_sequence (a : ℕ → ℝ) : Prop :=
  a 0 = 10 ∧ a 1 = -10 ∧ ∀ n, a (n + 2) = -a n

theorem nth_term_206 (a : ℕ → ℝ) (h : geometric_sequence a) : a 205 = -10 :=
by
  -- Utilizing the sequence property to determine the 206th term
  sorry

end nth_term_206_l212_212434


namespace polynomial_identity_l212_212741

open Polynomial

-- Definition of the non-zero polynomial of interest
noncomputable def p (a : ℝ) : Polynomial ℝ := Polynomial.C a * (Polynomial.X ^ 3 - Polynomial.X)

-- Theorem stating that, for all x, the given equation holds for the polynomial p
theorem polynomial_identity (a : ℝ) (h : a ≠ 0) :
  ∀ x : ℝ, (x - 1) * (p a).eval (x + 1) - (x + 2) * (p a).eval x = 0 :=
by
  sorry

end polynomial_identity_l212_212741


namespace total_games_l212_212374

-- Definitions and conditions
noncomputable def num_teams : ℕ := 12

noncomputable def regular_season_games_each : ℕ := 4

noncomputable def knockout_games_each : ℕ := 2

-- Calculate total number of games
theorem total_games : (num_teams * (num_teams - 1) / 2) * regular_season_games_each + 
                      (num_teams * knockout_games_each / 2) = 276 :=
by
  -- This is the statement to be proven
  sorry

end total_games_l212_212374


namespace length_of_longest_side_l212_212743

theorem length_of_longest_side (l w : ℝ) (h_fencing : 2 * l + 2 * w = 240) (h_area : l * w = 8 * 240) : max l w = 96 :=
by sorry

end length_of_longest_side_l212_212743


namespace domain_of_function_l212_212416

theorem domain_of_function :
  ∀ x, (2 * x - 1 ≥ 0) ∧ (x^2 ≠ 1) → (x ≥ 1/2 ∧ x < 1) ∨ (x > 1) := 
sorry

end domain_of_function_l212_212416


namespace expand_polynomial_l212_212873

theorem expand_polynomial (t : ℝ) : (2 * t^3 - 3 * t + 2) * (-3 * t^2 + 3 * t - 5) = 
  -6 * t^5 + 6 * t^4 - t^3 + 3 * t^2 + 21 * t - 10 :=
by
  sorry

end expand_polynomial_l212_212873


namespace cylinder_sphere_ratio_is_3_2_l212_212381

noncomputable def cylinder_sphere_surface_ratio (r : ℝ) : ℝ :=
  let cylinder_surface_area := 2 * Real.pi * r^2 + 2 * r * Real.pi * (2 * r)
  let sphere_surface_area := 4 * Real.pi * r^2
  cylinder_surface_area / sphere_surface_area

theorem cylinder_sphere_ratio_is_3_2 (r : ℝ) (h : r > 0) :
  cylinder_sphere_surface_ratio r = 3 / 2 :=
by
  sorry

end cylinder_sphere_ratio_is_3_2_l212_212381


namespace smallest_b_for_perfect_square_l212_212291

theorem smallest_b_for_perfect_square : ∃ (b : ℕ), b > 4 ∧ (∃ k, (2 * b + 4) = k * k) ∧
                                             ∀ (b' : ℕ), b' > 4 ∧ (∃ k, (2 * b' + 4) = k * k) → b ≤ b' :=
by
  sorry

end smallest_b_for_perfect_square_l212_212291


namespace inequality_system_no_solution_l212_212990

theorem inequality_system_no_solution (a : ℝ) : ¬ (∃ x : ℝ, x ≤ 5 ∧ x > a) ↔ a ≥ 5 :=
sorry

end inequality_system_no_solution_l212_212990


namespace sqrt_floor_eq_l212_212694

theorem sqrt_floor_eq (n : ℤ) (h : n ≥ 0) : 
  (⌊Real.sqrt n + Real.sqrt (n + 2)⌋) = ⌊Real.sqrt (4 * n + 1)⌋ :=
sorry

end sqrt_floor_eq_l212_212694


namespace distance_origin_to_point_l212_212557

theorem distance_origin_to_point :
  let distance (x1 y1 x2 y2 : ℝ) := Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)
  distance 0 0 8 (-15) = 17 :=
by
  sorry

end distance_origin_to_point_l212_212557


namespace num_colors_l212_212106

def total_balls := 350
def balls_per_color := 35

theorem num_colors :
  total_balls / balls_per_color = 10 := 
by
  sorry

end num_colors_l212_212106


namespace truck_stops_l212_212786

variable (a : ℕ → ℕ)
variable (sum_1 : ℕ)
variable (sum_2 : ℕ)

-- Definition for the first sequence with a common difference of -10
def first_sequence : ℕ → ℕ
| 0       => 40
| (n + 1) => first_sequence n - 10

-- Definition for the second sequence with a common difference of -5
def second_sequence : ℕ → ℕ 
| 0       => 10
| (n + 1) => second_sequence n - 5

-- Summing the first sequence elements before the condition change:
def sum_first_sequence : ℕ → ℕ 
| 0       => 40
| (n + 1) => sum_first_sequence n + first_sequence (n + 1)

-- Summing the second sequence elements after the condition change:
def sum_second_sequence : ℕ → ℕ 
| 0       => second_sequence 0
| (n + 1) => sum_second_sequence n + second_sequence (n + 1)

-- Final sum of distances
def total_distance : ℕ :=
  sum_first_sequence 3 + sum_second_sequence 1

theorem truck_stops (sum_1 sum_2 : ℕ) (h1 : sum_1 = sum_first_sequence 3)
 (h2 : sum_2 = sum_second_sequence 1) : 
  total_distance = 115 := by
  sorry


end truck_stops_l212_212786


namespace find_k_l212_212984

theorem find_k (k : ℕ) (hk : 0 < k) (h : (k + 4) / (k^2 - 1) = 9 / 35) : k = 14 :=
by
  sorry

end find_k_l212_212984


namespace marcia_banana_count_l212_212644

variable (B : ℕ)

-- Conditions
def appleCost := 2
def bananaCost := 1
def orangeCost := 3
def numApples := 12
def numOranges := 4
def avgCost := 2

-- Prove that given the conditions, B equals 4
theorem marcia_banana_count : 
  (24 + 12 + B) / (16 + B) = avgCost → B = 4 :=
by sorry

end marcia_banana_count_l212_212644


namespace flower_beds_fraction_l212_212069

-- Definitions based on given conditions
def yard_length := 30
def yard_width := 6
def trapezoid_parallel_side1 := 20
def trapezoid_parallel_side2 := 30
def flower_bed_leg := (trapezoid_parallel_side2 - trapezoid_parallel_side1) / 2
def flower_bed_area := (1 / 2) * flower_bed_leg ^ 2
def total_flower_bed_area := 2 * flower_bed_area
def yard_area := yard_length * yard_width
def occupied_fraction := total_flower_bed_area / yard_area

-- Statement to prove
theorem flower_beds_fraction :
  occupied_fraction = 5 / 36 :=
by
  -- sorries to skip the proofs
  sorry

end flower_beds_fraction_l212_212069


namespace no_adjacent_black_balls_l212_212190

theorem no_adjacent_black_balls (m n : ℕ) (h : m > n) : 
  (m + 1).choose n = (m + 1).factorial / (n.factorial * (m + 1 - n).factorial) := by
  sorry

end no_adjacent_black_balls_l212_212190


namespace probability_multiple_of_45_l212_212648

def multiples_of_3 := [3, 6, 9]
def primes_less_than_20 := [2, 3, 5, 7, 11, 13, 17, 19]

def favorable_outcomes := (9, 5)
def total_outcomes := (multiples_of_3.length * primes_less_than_20.length)

theorem probability_multiple_of_45 : (multiples_of_3.length = 3 ∧ primes_less_than_20.length = 8) → 
  ∃ w : ℚ, w = 1 / 24 :=
by {
  sorry
}

end probability_multiple_of_45_l212_212648


namespace evaluate_expression_l212_212007

theorem evaluate_expression:
  let a := 3
  let b := 2
  (a^b)^a - (b^a)^b = 665 :=
by
  sorry

end evaluate_expression_l212_212007


namespace capacity_of_other_bottle_l212_212150

theorem capacity_of_other_bottle 
  (total_milk : ℕ) (capacity_bottle_one : ℕ) (fraction_filled_other_bottle : ℚ)
  (equal_fraction : ℚ) (other_bottle_milk : ℚ) (capacity_other_bottle : ℚ) : 
  total_milk = 8 ∧ capacity_bottle_one = 4 ∧ other_bottle_milk = 16/3 ∧ 
  (equal_fraction * capacity_bottle_one + equal_fraction * capacity_other_bottle = total_milk) ∧ 
  (fraction_filled_other_bottle = 5.333333333333333) → capacity_other_bottle = 8 :=
by
  intro h
  sorry

end capacity_of_other_bottle_l212_212150


namespace store_earnings_l212_212614

theorem store_earnings (num_pencils : ℕ) (num_erasers : ℕ) (price_eraser : ℝ) 
  (multiplier : ℝ) (price_pencil : ℝ) (total_earnings : ℝ) :
  num_pencils = 20 →
  price_eraser = 1 →
  num_erasers = num_pencils * 2 →
  price_pencil = (price_eraser * num_erasers) * multiplier →
  multiplier = 2 →
  total_earnings = num_pencils * price_pencil + num_erasers * price_eraser →
  total_earnings = 120 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end store_earnings_l212_212614


namespace distance_from_ground_at_speed_25_is_137_5_l212_212299
noncomputable section

-- Define the initial conditions and givens
def buildingHeight : ℝ := 200
def speedProportionalityConstant : ℝ := 10
def distanceProportionalityConstant : ℝ := 10

-- Define the speed function and distance function
def speed (t : ℝ) : ℝ := speedProportionalityConstant * t
def distance (t : ℝ) : ℝ := distanceProportionalityConstant * (t * t)

-- Define the specific time when speed is 25 m/sec
def timeWhenSpeedIs25 : ℝ := 25 / speedProportionalityConstant

-- Define the distance traveled at this specific time
def distanceTraveledAtTime : ℝ := distance timeWhenSpeedIs25

-- Calculate the distance from the ground
def distanceFromGroundAtSpeed25 : ℝ := buildingHeight - distanceTraveledAtTime

-- State the theorem
theorem distance_from_ground_at_speed_25_is_137_5 :
  distanceFromGroundAtSpeed25 = 137.5 :=
sorry

end distance_from_ground_at_speed_25_is_137_5_l212_212299


namespace functional_equation_solution_l212_212756

noncomputable def f : ℝ → ℝ := sorry

theorem functional_equation_solution :
  (∀ x y : ℝ, f (f x + y) = 2 * x + f (f y - x)) →
  ∃ a : ℝ, ∀ x : ℝ, f x = x - a :=
by
  intro h
  sorry

end functional_equation_solution_l212_212756


namespace ratio_y_share_to_total_l212_212183

theorem ratio_y_share_to_total
  (total_profit : ℝ)
  (diff_share : ℝ)
  (h_total : total_profit = 800)
  (h_diff : diff_share = 160) :
  ∃ (a b : ℝ), (b / (a + b) = 2 / 5) ∧ (|a - b| = (a + b) / 5) :=
by
  sorry

end ratio_y_share_to_total_l212_212183


namespace range_of_m_l212_212724

theorem range_of_m (m : ℝ) (P : ℝ × ℝ) (h : P = (m + 3, m - 5)) (quadrant4 : P.1 > 0 ∧ P.2 < 0) : -3 < m ∧ m < 5 :=
by
  sorry

end range_of_m_l212_212724


namespace find_y_from_expression_l212_212628

theorem find_y_from_expression :
  ∀ y : ℕ, 2^10 + 2^10 + 2^10 + 2^10 = 4^y → y = 6 :=
by
  sorry

end find_y_from_expression_l212_212628


namespace area_Q1RQ3Q5_of_regular_hexagon_l212_212083

noncomputable def area_quadrilateral (s : ℝ) (θ : ℝ) : ℝ := s^2 * Real.sin θ / 2

theorem area_Q1RQ3Q5_of_regular_hexagon :
  let apothem := 3
  let side_length := 6 * Real.sqrt 3
  let θ := Real.pi / 3  -- 60 degrees in radians
  area_quadrilateral (3 * Real.sqrt 3) θ = 27 * Real.sqrt 3 / 2 :=
by
  sorry

end area_Q1RQ3Q5_of_regular_hexagon_l212_212083


namespace exists_consecutive_integers_not_sum_of_two_squares_l212_212241

open Nat

theorem exists_consecutive_integers_not_sum_of_two_squares : 
  ∃ (m : ℕ), ∀ k : ℕ, k < 2017 → ¬(∃ a b : ℤ, (m + k) = a^2 + b^2) := 
sorry

end exists_consecutive_integers_not_sum_of_two_squares_l212_212241


namespace find_a_b_solution_set_l212_212117

-- Given function
def f (x : ℝ) (a b : ℝ) := x^2 - (a + b) * x + 3 * a

-- Part 1: Prove the values of a and b given the solution set of the inequality
theorem find_a_b (a b : ℝ) 
  (h1 : 1^2 - (a + b) * 1 + 3 * 1 = 0)
  (h2 : 3^2 - (a + b) * 3 + 3 * 1 = 0) :
  a = 1 ∧ b = 3 :=
sorry

-- Part 2: Find the solution set of the inequality f(x) > 0 given b = 3
theorem solution_set (a : ℝ)
  (h : b = 3) :
  (a > 3 → (∀ x, f x a 3 > 0 ↔ x < 3 ∨ x > a)) ∧
  (a < 3 → (∀ x, f x a 3 > 0 ↔ x < a ∨ x > 3)) ∧
  (a = 3 → (∀ x, f x a 3 > 0 ↔ x ≠ 3)) :=
sorry

end find_a_b_solution_set_l212_212117


namespace relationship_between_m_and_n_l212_212475

variable (a b m n : ℝ)

axiom h1 : a > b
axiom h2 : b > 0
axiom h3 : m = Real.sqrt a - Real.sqrt b
axiom h4 : n = Real.sqrt (a - b)

theorem relationship_between_m_and_n : m < n :=
by
  -- Lean requires 'sorry' to be used as a placeholder for the proof
  sorry

end relationship_between_m_and_n_l212_212475


namespace yard_length_l212_212989

-- Definition of the problem conditions
def num_trees : Nat := 11
def distance_between_trees : Nat := 15

-- Length of the yard is given by the product of (num_trees - 1) and distance_between_trees
theorem yard_length :
  (num_trees - 1) * distance_between_trees = 150 :=
by
  sorry

end yard_length_l212_212989


namespace root_of_quadratic_eq_when_C_is_3_l212_212805

-- Define the quadratic equation and the roots we are trying to prove
def quadratic_eq (C : ℝ) (x : ℝ) := 3 * x^2 - 6 * x + C = 0

-- Set the constant C to 3
def C : ℝ := 3

-- State the theorem that proves the root of the equation when C=3 is x=1
theorem root_of_quadratic_eq_when_C_is_3 : quadratic_eq C 1 :=
by
  -- Skip the detailed proof
  sorry

end root_of_quadratic_eq_when_C_is_3_l212_212805


namespace simplify_expression_l212_212795

theorem simplify_expression :
  ((45 * 2^10) / (15 * 2^5) * 5) = 480 := by
  sorry

end simplify_expression_l212_212795


namespace molecular_weight_of_compound_l212_212098

theorem molecular_weight_of_compound (total_weight_of_3_moles : ℝ) (n_moles : ℝ) 
  (h1 : total_weight_of_3_moles = 528) (h2 : n_moles = 3) : 
  (total_weight_of_3_moles / n_moles) = 176 :=
by
  sorry

end molecular_weight_of_compound_l212_212098


namespace find_A_l212_212400

theorem find_A (A B : ℕ) (A_digit : A < 10) (B_digit : B < 10) :
  let fourteenA := 100 * 1 + 10 * 4 + A
  let Bseventy3 := 100 * B + 70 + 3
  fourteenA + Bseventy3 = 418 → A = 5 :=
by
  sorry

end find_A_l212_212400


namespace KrystianaChargesForSecondFloorRooms_Theorem_l212_212878

noncomputable def KrystianaChargesForSecondFloorRooms (X : ℝ) : Prop :=
  let costFirstFloor := 3 * 15
  let costThirdFloor := 3 * (2 * 15)
  let totalEarnings := costFirstFloor + 3 * X + costThirdFloor
  totalEarnings = 165 → X = 10

-- This is the statement only. The proof is not included.
theorem KrystianaChargesForSecondFloorRooms_Theorem : KrystianaChargesForSecondFloorRooms 10 :=
sorry

end KrystianaChargesForSecondFloorRooms_Theorem_l212_212878


namespace scientific_notation_of_122254_l212_212702

theorem scientific_notation_of_122254 :
  122254 = 1.22254 * 10^5 :=
sorry

end scientific_notation_of_122254_l212_212702


namespace part_one_solution_part_two_solution_l212_212332

-- Definitions and conditions
def f (x : ℝ) (a : ℝ) : ℝ := |x + 1| - |a * x - 1|

-- Part (1): When a = 1, solution set of the inequality f(x) > 1 is (1/2, +∞)
theorem part_one_solution (x : ℝ) :
  f x 1 > 1 ↔ x > 1 / 2 := sorry

-- Part (2): If the inequality f(x) > x holds for x ∈ (0,1), range of values for a is (0, 2]
theorem part_two_solution (a : ℝ) :
  (∀ x, 0 < x ∧ x < 1 → f x a > x) ↔ 0 < a ∧ a ≤ 2 := sorry

end part_one_solution_part_two_solution_l212_212332


namespace gcd_282_470_l212_212376

theorem gcd_282_470 : Nat.gcd 282 470 = 94 :=
by
  sorry

end gcd_282_470_l212_212376


namespace find_k_l212_212704

variable {a : ℕ → ℝ}

def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop := 
  ∀ n, a (n + 1) = a n + d

def sum_of_sequence (S : ℕ → ℝ) (a : ℕ → ℝ) : Prop :=
  ∀ n, S n = n * (a 1 + a n) / 2

theorem find_k (a : ℕ → ℝ) (S : ℕ → ℝ) (d : ℝ) (k : ℕ)
  (h1 : a 2 = -1)
  (h2 : 2 * a 1 + a 3 = -1)
  (h3 : arithmetic_sequence a d)
  (h4 : sum_of_sequence S a)
  (h5 : S k = -99) :
  k = 11 := 
by
  sorry

end find_k_l212_212704


namespace minimum_time_to_cook_3_pancakes_l212_212750

theorem minimum_time_to_cook_3_pancakes (can_fry_two_pancakes_at_a_time : Prop) 
   (time_to_fully_cook_one_pancake : ℕ) (time_to_cook_one_side : ℕ) :
  can_fry_two_pancakes_at_a_time →
  time_to_fully_cook_one_pancake = 2 →
  time_to_cook_one_side = 1 →
  3 = 3 := 
by
  intros
  sorry

end minimum_time_to_cook_3_pancakes_l212_212750


namespace value_of_a_cube_l212_212977

-- We define the conditions given in the problem.
def A (a : ℤ) : Set ℤ := {5, a^2 + 2 * a + 4}
def a_satisfies (a : ℤ) : Prop := 7 ∈ A a

-- We state the theorem.
theorem value_of_a_cube (a : ℤ) (h1 : a_satisfies a) : a^3 = 1 ∨ a^3 = -27 := by
  sorry

end value_of_a_cube_l212_212977


namespace deepak_share_l212_212231

theorem deepak_share (investment_Anand investment_Deepak total_profit : ℕ)
  (h₁ : investment_Anand = 2250) (h₂ : investment_Deepak = 3200) (h₃ : total_profit = 1380) :
  ∃ share_Deepak, share_Deepak = 810 := sorry

end deepak_share_l212_212231


namespace line_through_two_points_l212_212182

-- Define the points
def p1 : ℝ × ℝ := (1, 0)
def p2 : ℝ × ℝ := (0, -2)

-- Define the equation of the line passing through the points
def line_equation (x y : ℝ) : Prop :=
  2 * x - y - 2 = 0

-- The main theorem
theorem line_through_two_points : ∀ x y, p1 = (1, 0) ∧ p2 = (0, -2) → line_equation x y :=
  by sorry

end line_through_two_points_l212_212182


namespace factorize_expression_triangle_is_isosceles_l212_212618

-- Define the first problem: Factorize the expression.
theorem factorize_expression (a b : ℝ) : a^2 - 4 * a - b^2 + 4 = (a + b - 2) * (a - b - 2) := 
by
  sorry

-- Define the second problem: Determine the shape of the triangle.
theorem triangle_is_isosceles (a b c : ℝ) (h : a^2 - ab - ac + bc = 0) : a = b ∨ a = c :=
by
  sorry

end factorize_expression_triangle_is_isosceles_l212_212618


namespace min_value_3x_plus_4y_l212_212661

theorem min_value_3x_plus_4y (x y : ℝ) (hx : 0 < x) (hy : 0 < y) (h : x + 3 * y = x * y) : 3 * x + 4 * y = 28 :=
sorry

end min_value_3x_plus_4y_l212_212661


namespace length_of_AB_l212_212657

theorem length_of_AB 
  (AB BC CD AD : ℕ)
  (h1 : AB = 1 * BC / 2)
  (h2 : BC = 6 * CD / 5)
  (h3 : AB + BC + CD = 56)
  : AB = 12 := sorry

end length_of_AB_l212_212657


namespace nat_values_of_x_l212_212249

theorem nat_values_of_x :
  (∃ (x : ℕ), 2^(x - 5) = 2 ∧ x = 6) ∧
  (∃ (x : ℕ), 2^x = 512 ∧ x = 9) ∧
  (∃ (x : ℕ), x^5 = 243 ∧ x = 3) ∧
  (∃ (x : ℕ), x^4 = 625 ∧ x = 5) :=
  by {
    sorry
  }

end nat_values_of_x_l212_212249


namespace number_of_pupils_l212_212596

-- Define the conditions.
variables (n : ℕ) -- Number of pupils in the class.

-- Axioms based on the problem statement.
axiom marks_difference : 67 - 45 = 22
axiom avg_increase : (1 / 2 : ℝ) * n = 22 

-- The theorem we need to prove.
theorem number_of_pupils : n = 44 := by
  -- Proof will go here.
  sorry

end number_of_pupils_l212_212596


namespace number_of_employees_l212_212132

-- Definitions
def emily_original_salary : ℕ := 1000000
def emily_new_salary : ℕ := 850000
def employee_original_salary : ℕ := 20000
def employee_new_salary : ℕ := 35000
def salary_difference : ℕ := emily_original_salary - emily_new_salary
def salary_increase_per_employee : ℕ := employee_new_salary - employee_original_salary

-- Theorem: Prove Emily has n employees where n = 10
theorem number_of_employees : salary_difference / salary_increase_per_employee = 10 :=
by sorry

end number_of_employees_l212_212132


namespace green_balls_more_than_red_l212_212075

theorem green_balls_more_than_red
  (total_balls : ℕ) (red_balls : ℕ) (green_balls : ℕ)
  (h1 : total_balls = 66)
  (h2 : red_balls = 30)
  (h3 : green_balls = total_balls - red_balls) : green_balls - red_balls = 6 :=
by
  sorry

end green_balls_more_than_red_l212_212075


namespace remainder_of_expression_l212_212593

theorem remainder_of_expression (k : ℤ) (hk : 0 < k) :
  (4 * k * (2 + 4 + 4 * k) + 3) % 2 = 1 :=
by
  sorry

end remainder_of_expression_l212_212593


namespace slices_per_pizza_l212_212783

-- Definitions based on the conditions
def num_pizzas : Nat := 3
def total_cost : Nat := 72
def cost_per_5_slices : Nat := 10

-- To find the number of slices per pizza
theorem slices_per_pizza (num_pizzas : Nat) (total_cost : Nat) (cost_per_5_slices : Nat): 
  (total_cost / num_pizzas) / (cost_per_5_slices / 5) = 12 :=
by
  sorry

end slices_per_pizza_l212_212783


namespace negation_of_all_cars_are_fast_l212_212580

variable {α : Type} -- Assume α is the type of entities
variable (car fast : α → Prop) -- car and fast are predicates on entities

theorem negation_of_all_cars_are_fast :
  ¬ (∀ x, car x → fast x) ↔ ∃ x, car x ∧ ¬ fast x :=
by sorry

end negation_of_all_cars_are_fast_l212_212580


namespace find_divisor_l212_212794

variable (x y : ℝ)
variable (h1 : (x - 5) / 7 = 7)
variable (h2 : (x - 2) / y = 4)

theorem find_divisor (x y : ℝ) (h1 : (x - 5) / 7 = 7) (h2 : (x - 2) / y = 4) : y = 13 := by
  sorry

end find_divisor_l212_212794


namespace ella_max_book_price_l212_212883

/--
Given that Ella needs to buy 20 identical books and her total budget, 
after deducting the $5 entry fee, is $195. Each book has the same 
cost in whole dollars, and an 8% sales tax is applied to the price of each book. 
Prove that the highest possible price per book that Ella can afford is $9.
-/
theorem ella_max_book_price : 
  ∀ (n : ℕ) (B T : ℝ), n = 20 → B = 195 → T = 1.08 → 
  ∃ (p : ℕ), (↑p ≤ B / T / n) → (9 ≤ p) := 
by 
  sorry

end ella_max_book_price_l212_212883


namespace roberto_outfits_l212_212502

theorem roberto_outfits : 
  let trousers := 5
  let shirts := 5
  let jackets := 3
  (trousers * shirts * jackets = 75) :=
by sorry

end roberto_outfits_l212_212502


namespace dot_product_AB_BC_l212_212121

theorem dot_product_AB_BC (AB BC : ℝ) (B : ℝ) 
  (h1 : AB = 3) (h2 : BC = 4) (h3 : B = π/6) :
  (AB * BC * Real.cos (π - B) = -6 * Real.sqrt 3) :=
by
  rw [h1, h2, h3]
  sorry

end dot_product_AB_BC_l212_212121


namespace product_xyz_equals_1080_l212_212463

noncomputable def xyz_product (x y z : ℝ) : ℝ :=
  if (x > 0) ∧ (y > 0) ∧ (z > 0) ∧ (x * (y + z) = 198) ∧ (y * (z + x) = 216) ∧ (z * (x + y) = 234)
  then x * y * z
  else 0 

theorem product_xyz_equals_1080 {x y z : ℝ} :
  (x > 0) ∧ (y > 0) ∧ (z > 0) ∧ (x * (y + z) = 198) ∧ (y * (z + x) = 216) ∧ (z * (x + y) = 234) →
  xyz_product x y z = 1080 :=
by
  intros h
  -- Proof skipped
  sorry

end product_xyz_equals_1080_l212_212463


namespace probability_drawing_3_one_color_1_other_l212_212896

theorem probability_drawing_3_one_color_1_other (black white : ℕ) (total_balls drawn_balls : ℕ) 
    (total_ways : ℕ) (ways_3_black_1_white : ℕ) (ways_1_black_3_white : ℕ) :
    black = 10 → white = 5 → total_balls = 15 → drawn_balls = 4 →
    total_ways = Nat.choose total_balls drawn_balls →
    ways_3_black_1_white = Nat.choose black 3 * Nat.choose white 1 →
    ways_1_black_3_white = Nat.choose black 1 * Nat.choose white 3 →
    (ways_3_black_1_white + ways_1_black_3_white) / total_ways = 140 / 273 := 
by
  intros h_black h_white h_total_balls h_drawn_balls h_total_ways h_ways_3_black_1_white h_ways_1_black_3_white
  -- The proof would go here, but is not required for this task.
  sorry

end probability_drawing_3_one_color_1_other_l212_212896


namespace John_used_16_bulbs_l212_212244

variable (X : ℕ)

theorem John_used_16_bulbs
  (h1 : 40 - X = 2 * 12) :
  X = 16 := 
sorry

end John_used_16_bulbs_l212_212244


namespace blocks_calculation_l212_212955

theorem blocks_calculation
  (total_amount : ℕ)
  (gift_cost : ℕ)
  (workers_per_block : ℕ)
  (H1  : total_amount = 4000)
  (H2  : gift_cost = 4)
  (H3  : workers_per_block = 100)
  : total_amount / gift_cost / workers_per_block = 10 :=
by
  sorry

end blocks_calculation_l212_212955


namespace delores_initial_money_l212_212085

-- Definitions and conditions based on the given problem
def original_computer_price : ℝ := 400
def original_printer_price : ℝ := 40
def original_headphones_price : ℝ := 60

def computer_discount : ℝ := 0.10
def computer_tax : ℝ := 0.08
def printer_tax : ℝ := 0.05
def headphones_tax : ℝ := 0.06

def leftover_money : ℝ := 10

-- Final proof problem statement
theorem delores_initial_money :
  original_computer_price * (1 - computer_discount) * (1 + computer_tax) +
  original_printer_price * (1 + printer_tax) +
  original_headphones_price * (1 + headphones_tax) + leftover_money = 504.40 := by
  sorry -- Proof is not required

end delores_initial_money_l212_212085


namespace find_f_a_l212_212017

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 0 then 4 * Real.logb 2 (-x) else abs (x^2 + a * x)

theorem find_f_a (a : ℝ) (h : a ≠ 0) (h1 : f a (f a (-Real.sqrt 2)) = 4) : f a a = 8 :=
sorry

end find_f_a_l212_212017


namespace no_integer_in_interval_l212_212828

theorem no_integer_in_interval (n : ℕ) : ¬ ∃ k : ℤ, 
  (n ≠ 0 ∧ (n * Real.sqrt 2 - 1 / (3 * n) < k) ∧ (k < n * Real.sqrt 2 + 1 / (3 * n))) := 
sorry

end no_integer_in_interval_l212_212828


namespace brownie_pieces_count_l212_212903

theorem brownie_pieces_count :
  let tray_length := 24
  let tray_width := 20
  let brownie_length := 3
  let brownie_width := 2
  let tray_area := tray_length * tray_width
  let brownie_area := brownie_length * brownie_width
  let pieces_count := tray_area / brownie_area
  pieces_count = 80 :=
by
  let tray_length := 24
  let tray_width := 20
  let brownie_length := 3
  let brownie_width := 2
  let tray_area := 24 * 20
  let brownie_area := 3 * 2
  let pieces_count := tray_area / brownie_area
  have h1 : tray_length * tray_width = 480 := by norm_num
  have h2 : brownie_length * brownie_width = 6 := by norm_num
  have h3 : pieces_count = 80 := by norm_num
  exact h3

end brownie_pieces_count_l212_212903


namespace pen_price_l212_212832

theorem pen_price (p : ℝ) (h : 30 = 10 * p + 10 * (p / 2)) : p = 2 :=
sorry

end pen_price_l212_212832


namespace base8_to_base10_12345_l212_212954

theorem base8_to_base10_12345 : (1 * 8^4 + 2 * 8^3 + 3 * 8^2 + 4 * 8^1 + 5 * 8^0) = 5349 := by
  sorry

end base8_to_base10_12345_l212_212954


namespace arithmetic_sequence_count_l212_212956

-- Definitions based on the conditions and question
def sequence_count : ℕ := 314 -- The number of common differences for 315-term sequences
def set_size : ℕ := 2014     -- The maximum number in the set {1, 2, 3, ..., 2014}
def min_seq_length : ℕ := 315 -- The length of the arithmetic sequence

-- Lean 4 statement to verify the number of ways to form the required sequence
theorem arithmetic_sequence_count :
  ∃ (ways : ℕ), ways = 5490 ∧
  (∀ (d : ℕ), 1 ≤ d ∧ d ≤ 6 →
  (set_size - (sequence_count * d - 1)) > 0 → 
  ways = (
    if d = 1 then set_size - sequence_count + 1 else
    if d = 2 then set_size - (sequence_count * 2 - 1) + 1 else
    if d = 3 then set_size - (sequence_count * 3 - 1) + 1 else
    if d = 4 then set_size - (sequence_count * 4 - 1) + 1 else
    if d = 5 then set_size - (sequence_count * 5 - 1) + 1 else
    set_size - (sequence_count * 6 - 1) + 1) - 2
  ) :=
sorry

end arithmetic_sequence_count_l212_212956


namespace real_roots_a_set_t_inequality_l212_212533

noncomputable def set_of_a : Set ℝ := {a | -1 ≤ a ∧ a ≤ 7}

theorem real_roots_a_set (x a : ℝ) :
  (∃ x, x^2 - 4 * x + abs (a - 3) = 0) ↔ a ∈ set_of_a := 
by
  sorry

theorem t_inequality (t a : ℝ) (h : ∀ a ∈ set_of_a, t^2 - 2 * a * t + 12 < 0) :
  3 < t ∧ t < 4 := 
by
  sorry

end real_roots_a_set_t_inequality_l212_212533


namespace angle_B_triangle_perimeter_l212_212849

variable {A B C a b c : Real}

-- Definitions and conditions for part 1
def sides_relation (a b c : ℝ) (A : ℝ) : Prop :=
  2 * c = a + 2 * b * Real.cos A

-- Definitions and conditions for part 2
def triangle_area (a b c : ℝ) (B : ℝ) : Prop :=
  (1 / 2) * a * c * Real.sin B = Real.sqrt 3

def side_b_value (b : ℝ) : Prop :=
  b = Real.sqrt 13

-- Theorem statement for part 1 
theorem angle_B (a b c A : ℝ) (h1: sides_relation a b c A) : B = Real.pi / 3 :=
sorry

-- Theorem statement for part 2 
theorem triangle_perimeter (a b c B : ℝ) (h1 : triangle_area a b c B) (h2 : side_b_value b) (h3 : B = Real.pi / 3) : a + b + c = 5 + Real.sqrt 13 :=
sorry

end angle_B_triangle_perimeter_l212_212849


namespace xy_squared_value_l212_212082

theorem xy_squared_value (x y : ℝ) (h1 : x * (x + y) = 22) (h2 : y * (x + y) = 78 - y) :
  (x + y) ^ 2 = 100 :=
  sorry

end xy_squared_value_l212_212082


namespace distance_between_trees_l212_212631

def yard_length : ℕ := 350
def num_trees : ℕ := 26
def num_intervals : ℕ := num_trees - 1

theorem distance_between_trees :
  yard_length / num_intervals = 14 := 
sorry

end distance_between_trees_l212_212631


namespace remove_five_yields_average_10_5_l212_212716

def numberList : List ℕ := [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]

def averageRemaining (l : List ℕ) : ℚ :=
  (List.sum l : ℚ) / l.length

theorem remove_five_yields_average_10_5 :
  averageRemaining (numberList.erase 5) = 10.5 :=
sorry

end remove_five_yields_average_10_5_l212_212716


namespace jinho_total_distance_l212_212391

theorem jinho_total_distance (bus_distance_km : ℝ) (bus_distance_m : ℝ) (walk_distance_m : ℝ) :
  bus_distance_km = 4 → bus_distance_m = 436 → walk_distance_m = 1999 → 
  (2 * (bus_distance_km + bus_distance_m / 1000 + walk_distance_m / 1000)) = 12.87 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end jinho_total_distance_l212_212391


namespace relationship_between_line_and_circle_l212_212441

variables {a b r : ℝ} (M : ℝ × ℝ) (l m : ℝ → ℝ)

def point_inside_circle_not_on_axes 
    (M : ℝ × ℝ) (r : ℝ) : Prop := 
    (M.fst^2 + M.snd^2 < r^2) ∧ (M.fst ≠ 0) ∧ (M.snd ≠ 0)

def line_eq (a b r : ℝ) (x y : ℝ) : Prop := 
    a * x + b * y = r^2

def chord_midpoint (M : ℝ × ℝ) (m : ℝ → ℝ) : Prop := 
    ∃ x1 y1 x2 y2, 
    (M.fst = (x1 + x2) / 2 ∧ M.snd = (y1 + y2) / 2) ∧ 
    (m x1 = y1 ∧ m x2 = y2)

def circle_external (O : ℝ → ℝ) (l : ℝ → ℝ) : Prop := 
    ∀ x y, O x = y → l x ≠ y

theorem relationship_between_line_and_circle
    (M_inside : point_inside_circle_not_on_axes M r)
    (M_chord : chord_midpoint M m)
    (line_eq_l : line_eq a b r M.fst M.snd) :
    (m (M.fst) = - (a / b) * M.snd) ∧ 
    (∀ x, l x ≠ m x) :=
sorry

end relationship_between_line_and_circle_l212_212441


namespace sarah_age_ratio_l212_212717

theorem sarah_age_ratio 
  (S M : ℕ) 
  (h1 : S = 3 * (S / 3))
  (h2 : S - M = 5 * (S / 3 - 2 * M)) : 
  S / M = 27 / 2 := 
sorry

end sarah_age_ratio_l212_212717


namespace prob_not_less_than_30_l212_212612

-- Define the conditions
def prob_less_than_30 : ℝ := 0.3
def prob_between_30_and_40 : ℝ := 0.5

-- State the theorem
theorem prob_not_less_than_30 (h1 : prob_less_than_30 = 0.3) : 1 - prob_less_than_30 = 0.7 :=
by
  sorry

end prob_not_less_than_30_l212_212612


namespace units_digit_squares_eq_l212_212904

theorem units_digit_squares_eq (x y : ℕ) (hx : x % 10 + y % 10 = 10) :
  (x * x) % 10 = (y * y) % 10 :=
by
  sorry

end units_digit_squares_eq_l212_212904


namespace production_value_equation_l212_212120

theorem production_value_equation (x : ℝ) :
  (2000000 * (1 + x)^2) - (2000000 * (1 + x)) = 220000 := 
sorry

end production_value_equation_l212_212120


namespace right_isosceles_triangle_acute_angle_45_l212_212058

theorem right_isosceles_triangle_acute_angle_45
    (a : ℝ)
    (h_leg_conditions : ∀ b : ℝ, a = b)
    (h_hypotenuse_condition : ∀ c : ℝ, c^2 = 2 * (a * a)) :
    ∃ θ : ℝ, θ = 45 :=
by
    sorry

end right_isosceles_triangle_acute_angle_45_l212_212058


namespace question_1_question_2_question_3_l212_212417

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x - 2 * b

-- Question 1
theorem question_1 (a b : ℝ) (h : a = b) (ha : a > 0) :
  ∀ x : ℝ, (f a b x < 0) ↔ (-2 < x ∧ x < 1) :=
sorry

-- Question 2
theorem question_2 (b : ℝ) :
  (∀ x : ℝ, x < 2 → (f 1 b x ≥ 1)) → (b ≤ 2 * Real.sqrt 3 - 4) :=
sorry

-- Question 3
theorem question_3 (a b : ℝ) (h1 : |f a b (-1)| ≤ 1) (h2 : |f a b 1| ≤ 3) :
  (5 / 3 ≤ |a| + |b + 2| ∧ |a| + |b + 2| ≤ 9) :=
sorry

end question_1_question_2_question_3_l212_212417


namespace mark_money_left_l212_212469

theorem mark_money_left (initial_money : ℕ) (cost_book1 cost_book2 cost_book3 : ℕ) (n_book1 n_book2 n_book3 : ℕ) 
  (total_cost : ℕ) (money_left : ℕ) 
  (h1 : initial_money = 85)
  (h2 : cost_book1 = 7)
  (h3 : n_book1 = 3)
  (h4 : cost_book2 = 5)
  (h5 : n_book2 = 4)
  (h6 : cost_book3 = 9)
  (h7 : n_book3 = 2)
  (h8 : total_cost = 21 + 20 + 18)
  (h9 : money_left = initial_money - total_cost):
  money_left = 26 := by
  sorry

end mark_money_left_l212_212469


namespace arithmetic_geometric_sequences_l212_212068

variable {S T : ℕ → ℝ}
variable {a b : ℕ → ℝ}

theorem arithmetic_geometric_sequences (h1 : a 3 = b 3)
  (h2 : a 4 = b 4)
  (h3 : (S 5 - S 3) / (T 4 - T 2) = 5) :
  (a 5 + a 3) / (b 5 + b 3) = - (3 / 5) := by
  sorry

end arithmetic_geometric_sequences_l212_212068


namespace measure_of_angle_D_l212_212318

theorem measure_of_angle_D 
  (A B C D E F : ℝ)
  (h1 : A = B) (h2 : B = C) (h3 : C = F)
  (h4 : D = E) (h5 : A = D - 30) 
  (sum_angles : A + B + C + D + E + F = 720) : 
  D = 140 :=
by
  sorry

end measure_of_angle_D_l212_212318


namespace selected_numbers_satisfy_conditions_l212_212807

theorem selected_numbers_satisfy_conditions :
  ∃ (nums : Finset ℕ), 
  nums = {6, 34, 35, 51, 55, 77} ∧
  (∀ (a b c : ℕ), a ∈ nums → b ∈ nums → c ∈ nums → a ≠ b → a ≠ c → b ≠ c → 
    gcd a b = 1 ∨ gcd b c = 1 ∨ gcd c a = 1) ∧
  (∀ (x y z : ℕ), x ∈ nums → y ∈ nums → z ∈ nums → x ≠ y → x ≠ z → y ≠ z → 
    gcd x y ≠ 1 ∨ gcd y z ≠ 1 ∨ gcd z x ≠ 1) := 
sorry

end selected_numbers_satisfy_conditions_l212_212807


namespace polynomial_solution_l212_212610

open Polynomial

noncomputable def p (x : ℝ) : ℝ := -4 * x^5 + 3 * x^3 - 7 * x^2 + 4 * x - 2

theorem polynomial_solution (x : ℝ) :
  4 * x^5 + 3 * x^3 + 2 * x^2 + (-4 * x^5 + 3 * x^3 - 7 * x^2 + 4 * x - 2) = 6 * x^3 - 5 * x^2 + 4 * x - 2 :=
by
  -- Verification of the equality
  sorry

end polynomial_solution_l212_212610


namespace chad_savings_correct_l212_212243

variable (earnings_mowing : ℝ := 600)
variable (earnings_birthday : ℝ := 250)
variable (earnings_video_games : ℝ := 150)
variable (earnings_odd_jobs : ℝ := 150)
variable (tax_rate : ℝ := 0.10)

noncomputable def total_earnings : ℝ := 
  earnings_mowing + earnings_birthday + earnings_video_games + earnings_odd_jobs

noncomputable def taxes : ℝ := 
  tax_rate * total_earnings

noncomputable def money_after_taxes : ℝ := 
  total_earnings - taxes

noncomputable def savings_mowing : ℝ := 
  0.50 * earnings_mowing

noncomputable def savings_birthday : ℝ := 
  0.30 * earnings_birthday

noncomputable def savings_video_games : ℝ := 
  0.40 * earnings_video_games

noncomputable def savings_odd_jobs : ℝ := 
  0.20 * earnings_odd_jobs

noncomputable def total_savings : ℝ := 
  savings_mowing + savings_birthday + savings_video_games + savings_odd_jobs

theorem chad_savings_correct : total_savings = 465 := by
  sorry

end chad_savings_correct_l212_212243


namespace problem_x_value_l212_212445

theorem problem_x_value (x : ℝ) (h : (max 3 (max 6 (max 9 x)) * min 3 (min 6 (min 9 x)) = 3 + 6 + 9 + x)) : 
    x = 9 / 4 :=
by
  sorry

end problem_x_value_l212_212445


namespace center_of_circle_is_1_2_l212_212048

theorem center_of_circle_is_1_2 :
  ∀ x y : ℝ, x^2 + y^2 - 2 * x - 4 * y = 0 ↔ ∃ (r : ℝ), (x - 1)^2 + (y - 2)^2 = r^2 := by
  sorry

end center_of_circle_is_1_2_l212_212048


namespace compound_interest_rate_is_10_percent_l212_212682

theorem compound_interest_rate_is_10_percent
  (P : ℝ) (CI : ℝ) (t : ℝ) (A : ℝ) (n : ℝ) (r : ℝ)
  (hP : P = 4500) (hCI : CI = 945.0000000000009) (ht : t = 2) (hn : n = 1) (hA : A = P + CI)
  (h_eq : A = P * (1 + r / n)^(n * t)) :
  r = 0.1 :=
by
  sorry

end compound_interest_rate_is_10_percent_l212_212682


namespace unique_integer_for_P5_l212_212510

-- Define the polynomial P with integer coefficients
variable (P : ℤ → ℤ)

-- The conditions given in the problem
variable (x1 x2 x3 : ℤ)
variable (Hx1 : P x1 = 1)
variable (Hx2 : P x2 = 2)
variable (Hx3 : P x3 = 3)

-- The main theorem to prove
theorem unique_integer_for_P5 {P : ℤ → ℤ} {x1 x2 x3 : ℤ}
(Hx1 : P x1 = 1) (Hx2 : P x2 = 2) (Hx3 : P x3 = 3) :
  ∃!(x : ℤ), P x = 5 := sorry

end unique_integer_for_P5_l212_212510


namespace work_completion_days_l212_212193

theorem work_completion_days (A B : ℕ) (hB : B = 12) (work_together_days : ℕ) (work_together : work_together_days = 3) (work_alone_days : ℕ) (work_alone : work_alone_days = 3) : 
  (1 / A + 1 / B) * 3 + (1 / B) * 3 = 1 → A = 6 := 
by 
  intro h
  sorry

end work_completion_days_l212_212193


namespace greatest_common_divisor_is_40_l212_212473

def distance_to_boston : ℕ := 840
def distance_to_atlanta : ℕ := 440

theorem greatest_common_divisor_is_40 :
  Nat.gcd distance_to_boston distance_to_atlanta = 40 :=
by
  -- The theorem statement as described is correct
  -- Proof is omitted as per instructions
  sorry

end greatest_common_divisor_is_40_l212_212473


namespace perpendicular_lines_sufficient_l212_212495

noncomputable def line1_slope (a : ℝ) : ℝ :=
-((a + 2) / (3 * a))

noncomputable def line2_slope (a : ℝ) : ℝ :=
-((a - 2) / (a + 2))

theorem perpendicular_lines_sufficient (a : ℝ) (h : a = -2) :
  line1_slope a * line2_slope a = -1 :=
by
  sorry

end perpendicular_lines_sufficient_l212_212495


namespace solve_cubic_equation_l212_212067

theorem solve_cubic_equation (x : ℝ) (h : x^3 + (x + 2)^3 + (x + 4)^3 = (x + 6)^3) : x = 6 :=
by sorry

end solve_cubic_equation_l212_212067


namespace Ashutosh_completion_time_l212_212668

def Suresh_work_rate := 1 / 15
def Ashutosh_work_rate := 1 / 25
def Suresh_work_time := 9

def job_completed_by_Suresh_in_9_hours := Suresh_work_rate * Suresh_work_time
def remaining_job := 1 - job_completed_by_Suresh_in_9_hours

theorem Ashutosh_completion_time : 
  Ashutosh_work_rate * t = remaining_job -> t = 10 :=
by
  sorry

end Ashutosh_completion_time_l212_212668


namespace g_at_5_l212_212759

def g : ℝ → ℝ := sorry

axiom g_property : ∀ x : ℝ, g x + 3 * g (2 - x) = 4 * x^2 - 3 * x + 2

theorem g_at_5 : g 5 = -20 :=
by {
  apply sorry
}

end g_at_5_l212_212759


namespace height_flagstaff_l212_212653

variables (s_1 s_2 h_2 : ℝ)
variable (h : ℝ)

-- Define the conditions as given
def shadow_flagstaff := s_1 = 40.25
def shadow_building := s_2 = 28.75
def height_building := h_2 = 12.5
def similar_triangles := (h / s_1) = (h_2 / s_2)

-- Prove the height of the flagstaff
theorem height_flagstaff : shadow_flagstaff s_1 ∧ shadow_building s_2 ∧ height_building h_2 ∧ similar_triangles h s_1 h_2 s_2 → h = 17.5 :=
by sorry

end height_flagstaff_l212_212653


namespace minimum_value_l212_212732

theorem minimum_value (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h : x * y * z = 64) : 
  x^2 + 8 * x * y + 16 * y^2 + 4 * z^2 ≥ 192 := 
  sorry

end minimum_value_l212_212732


namespace mileage_in_scientific_notation_l212_212817

noncomputable def scientific_notation_of_mileage : Prop :=
  let mileage := 42000
  mileage = 4.2 * 10^4

theorem mileage_in_scientific_notation :
  scientific_notation_of_mileage :=
by
  sorry

end mileage_in_scientific_notation_l212_212817


namespace amelia_remaining_money_l212_212853

variable {m b n : ℚ}

theorem amelia_remaining_money (h : (1 / 4) * m = (1 / 2) * n * b) : 
  m - n * b = (1 / 2) * m :=
by
  sorry

end amelia_remaining_money_l212_212853


namespace valid_colorings_l212_212643

-- Define the coloring function and the condition
variable (f : ℕ → ℕ) -- f assigns a color (0, 1, or 2) to each natural number
variable (a b c : ℕ)
-- Colors are represented by 0, 1, or 2
variable (colors : Fin 3)

-- Define the condition to be checked
def valid_coloring : Prop :=
  ∀ a b c, 2000 * (a + b) = c → (f a = f b ∧ f b = f c) ∨ (f a ≠ f b ∧ f b ≠ f c ∧ f c ≠ f a)

-- Now define the two possible valid ways of coloring
def all_same_color : Prop :=
  ∃ color, ∀ n, f n = color

def every_third_different : Prop :=
  (∀ k : ℕ, f (3 * k) = 0 ∧ f (3 * k + 1) = 1 ∧ f (3 * k + 2) = 2)

-- Prove that these are the only two valid ways
theorem valid_colorings :
  valid_coloring f →
  all_same_color f ∨ every_third_different f :=
sorry

end valid_colorings_l212_212643


namespace solution_set_inequality_l212_212221

theorem solution_set_inequality : {x : ℝ | (x - 2) * (1 - 2 * x) ≥ 0} = {x : ℝ | 1/2 ≤ x ∧ x ≤ 2} :=
by
  sorry  -- Proof to be provided

end solution_set_inequality_l212_212221


namespace find_b_l212_212227

theorem find_b (A B C : ℝ) (a b c : ℝ)
  (h1 : Real.tan A = 1 / 3)
  (h2 : Real.tan B = 1 / 2)
  (h3 : a = 1)
  (h4 : A + B + C = π) -- This condition is added because angles in a triangle sum up to π.
  : b = Real.sqrt 2 :=
by
  sorry

end find_b_l212_212227


namespace coordinates_of_point_P_l212_212386

-- Define the function y = x^3
def cubic (x : ℝ) : ℝ := x^3

-- Define the derivative of the function
def derivative_cubic (x : ℝ) : ℝ := 3 * x^2

-- Define the condition for the slope of the tangent line to the function at point P
def slope_tangent_line := 3

-- Prove that the coordinates of point P are (1, 1) or (-1, -1) when the slope of the tangent line is 3
theorem coordinates_of_point_P (x : ℝ) (y : ℝ) 
    (h1 : y = cubic x) 
    (h2 : derivative_cubic x = slope_tangent_line) : 
    (x = 1 ∧ y = 1) ∨ (x = -1 ∧ y = -1) :=
by
  sorry

end coordinates_of_point_P_l212_212386


namespace wheel_horizontal_distance_l212_212666

noncomputable def wheel_radius : ℝ := 2
noncomputable def wheel_revolution_fraction : ℝ := 3 / 4
noncomputable def wheel_circumference (r : ℝ) : ℝ := 2 * Real.pi * r

theorem wheel_horizontal_distance :
  wheel_circumference wheel_radius * wheel_revolution_fraction = 3 * Real.pi :=
by
  sorry

end wheel_horizontal_distance_l212_212666


namespace range_a_ff_a_eq_2_f_a_l212_212173

noncomputable def f (x : ℝ) : ℝ :=
if x < 1 then 3 * x - 1 else 2 ^ x

theorem range_a_ff_a_eq_2_f_a :
  {a : ℝ | f (f a) = 2 ^ (f a)} = {a : ℝ | a ≥ 2/3} :=
sorry

end range_a_ff_a_eq_2_f_a_l212_212173


namespace min_rooms_needed_l212_212335

-- Definitions and assumptions from conditions
def max_people_per_room : Nat := 3
def total_fans : Nat := 100
def number_of_teams : Nat := 3
def number_of_genders : Nat := 2
def groups := number_of_teams * number_of_genders

-- Main theorem statement
theorem min_rooms_needed 
  (max_people_per_room: Nat) 
  (total_fans: Nat) 
  (groups: Nat) 
  (h1: max_people_per_room = 3) 
  (h2: total_fans = 100) 
  (h3: groups = 6) : 
  ∃ (rooms: Nat), rooms ≥ 37 :=
by
  sorry

end min_rooms_needed_l212_212335


namespace cone_radius_l212_212950

theorem cone_radius (r l : ℝ)
  (h1 : 6 * Real.pi = Real.pi * r^2 + Real.pi * r * l)
  (h2 : 2 * Real.pi * r = Real.pi * l) :
  r = Real.sqrt 2 :=
by
  sorry

end cone_radius_l212_212950


namespace smallest_portion_is_two_l212_212803

theorem smallest_portion_is_two (a1 a2 a3 a4 a5 : ℕ) (d : ℕ) (h1 : a1 = a3 - 2 * d) (h2 : a2 = a3 - d) (h3 : a4 = a3 + d) (h4 : a5 = a3 + 2 * d) (h5 : a1 + a2 + a3 + a4 + a5 = 120) (h6 : a3 + a4 + a5 = 7 * (a1 + a2)) : a1 = 2 :=
by sorry

end smallest_portion_is_two_l212_212803


namespace minimum_gumballs_needed_l212_212787

/-- Alex wants to buy at least 150 gumballs,
    and have exactly 14 gumballs left after dividing evenly among 17 people.
    Determine the minimum number of gumballs Alex should buy. -/
theorem minimum_gumballs_needed (n : ℕ) (h1 : n ≥ 150) (h2 : n % 17 = 14) : n = 150 :=
sorry

end minimum_gumballs_needed_l212_212787


namespace max_value_of_expr_l212_212271

noncomputable def max_expr_value (x : ℝ) : ℝ :=
  x^6 / (x^12 + 4*x^9 - 6*x^6 + 16*x^3 + 64)

theorem max_value_of_expr : ∀ x : ℝ, max_expr_value x ≤ 1/26 :=
by
  sorry

end max_value_of_expr_l212_212271


namespace range_of_a_l212_212845

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, x^2 - 2*x - 3 < 0 → a < x) → a ≤ -1 :=
by
  sorry

end range_of_a_l212_212845


namespace intersection_A_B_l212_212982

def A : Set ℝ := { x | -1 < x ∧ x < 2 }
def B : Set ℝ := { x | ∃ (n : ℤ), (x : ℝ) = n }

theorem intersection_A_B : A ∩ B = {0, 1} := 
by
  sorry

end intersection_A_B_l212_212982


namespace incorrect_option_D_l212_212088

variable (AB BC BO DO AO CO : ℝ)
variable (DAB : ℝ)
variable (ABCD_is_rectangle ABCD_is_rhombus ABCD_is_square: Prop)

def conditions_statement :=
  AB = BC ∧
  DAB = 90 ∧
  BO = DO ∧
  AO = CO ∧
  (ABCD_is_rectangle ↔ (AB = BC ∧ AB ≠ BC)) ∧
  (ABCD_is_rhombus ↔ AB = BC ∧ AB ≠ BC) ∧
  (ABCD_is_square ↔ ABCD_is_rectangle ∧ ABCD_is_rhombus)

theorem incorrect_option_D
  (h1: BO = DO)
  (h2: AO = CO)
  (h3: ABCD_is_rectangle)
  (h4: conditions_statement AB BC BO DO AO CO DAB ABCD_is_rectangle ABCD_is_rhombus ABCD_is_square):
  ¬ ABCD_is_square :=
by
  sorry
  -- Proof omitted

end incorrect_option_D_l212_212088


namespace max_value_of_g_l212_212973

noncomputable def g (x : ℝ) : ℝ :=
  Real.sqrt (x * (80 - x)) + Real.sqrt (x * (10 - x))

theorem max_value_of_g :
  ∃ y_0 N, (∀ x, 0 ≤ x ∧ x ≤ 10 → g x ≤ N) ∧ g y_0 = N ∧ y_0 = 33.75 ∧ N = 22.5 := 
by
  -- Proof goes here.
  sorry

end max_value_of_g_l212_212973


namespace problem_l212_212755

theorem problem : (1 * (2 + 3) * 4 * 5) = 100 := by
  sorry

end problem_l212_212755


namespace fraction_sum_l212_212884

variable {w x y : ℚ}  -- assuming w, x, and y are rational numbers

theorem fraction_sum (h1 : w / x = 1 / 3) (h2 : w / y = 2 / 3) : (x + y) / y = 3 :=
sorry

end fraction_sum_l212_212884


namespace total_pencils_is_60_l212_212667

def original_pencils : ℕ := 33
def added_pencils : ℕ := 27
def total_pencils : ℕ := original_pencils + added_pencils

theorem total_pencils_is_60 : total_pencils = 60 := by
  sorry

end total_pencils_is_60_l212_212667


namespace q1_q2_l212_212735

variable (a b : ℝ)

-- Definition of the conditions
def conditions : Prop := a + b = 7 ∧ a * b = 6

-- Statement of the first question
theorem q1 (h : conditions a b) : a^2 + b^2 = 37 := sorry

-- Statement of the second question
theorem q2 (h : conditions a b) : a^3 * b - 2 * a^2 * b^2 + a * b^3 = 150 := sorry

end q1_q2_l212_212735


namespace Tom_marble_choices_l212_212595

theorem Tom_marble_choices :
  let total_marbles := 18
  let special_colors := 4
  let choose_one_from_special := (Nat.choose special_colors 1)
  let remaining_marbles := total_marbles - special_colors
  let choose_remaining := (Nat.choose remaining_marbles 5)
  choose_one_from_special * choose_remaining = 8008
:= sorry

end Tom_marble_choices_l212_212595


namespace max_students_equal_division_l212_212198

theorem max_students_equal_division (pens pencils : ℕ) (h_pens : pens = 640) (h_pencils : pencils = 520) : 
  Nat.gcd pens pencils = 40 :=
by
  rw [h_pens, h_pencils]
  have : Nat.gcd 640 520 = 40 := by norm_num
  exact this

end max_students_equal_division_l212_212198


namespace fourth_friend_age_is_8_l212_212646

-- Define the given data
variables (a1 a2 a3 a4 : ℕ)
variables (h_avg : (a1 + a2 + a3 + a4) / 4 = 9)
variables (h1 : a1 = 7) (h2 : a2 = 9) (h3 : a3 = 12)

-- Formalize the theorem to prove that the fourth friend's age is 8
theorem fourth_friend_age_is_8 : a4 = 8 :=
by
  -- Placeholder for the proof
  sorry

end fourth_friend_age_is_8_l212_212646


namespace alpha_in_second_quadrant_l212_212228

variable (α : ℝ)

-- Conditions that P(tan α, cos α) is in the third quadrant
def P_in_third_quadrant (α : ℝ) : Prop := (Real.tan α < 0) ∧ (Real.cos α < 0)

-- Theorem statement
theorem alpha_in_second_quadrant (h : P_in_third_quadrant α) : 
  π/2 < α ∧ α < π :=
sorry

end alpha_in_second_quadrant_l212_212228


namespace dry_grapes_weight_l212_212825

theorem dry_grapes_weight (W_fresh : ℝ) (W_dry : ℝ) (P_water_fresh : ℝ) (P_water_dry : ℝ) :
  W_fresh = 40 → P_water_fresh = 0.80 → P_water_dry = 0.20 → W_dry = 10 := 
by 
  intros hWf hPwf hPwd 
  sorry

end dry_grapes_weight_l212_212825


namespace rhombus_area_of_square_l212_212664

theorem rhombus_area_of_square (h : ∀ (c : ℝ), c = 96) : ∃ (a : ℝ), a = 288 := 
by
  sorry

end rhombus_area_of_square_l212_212664


namespace alice_favorite_number_l212_212663

theorem alice_favorite_number :
  ∃ (n : ℕ), 50 < n ∧ n < 100 ∧ n % 11 = 0 ∧ n % 2 ≠ 0 ∧ (n / 10 + n % 10) % 5 = 0 ∧ n = 55 :=
by
  sorry

end alice_favorite_number_l212_212663
