import Mathlib

namespace change_occurs_in_3_years_l1531_153135

theorem change_occurs_in_3_years (P A1 A2 : ℝ) (R T : ℝ) (h1 : P = 825) (h2 : A1 = 956) (h3 : A2 = 1055)
    (h4 : A1 = P + (P * R * T) / 100)
    (h5 : A2 = P + (P * (R + 4) * T) / 100) : T = 3 :=
by
  sorry

end change_occurs_in_3_years_l1531_153135


namespace max_value_of_y_over_x_l1531_153192

theorem max_value_of_y_over_x
  (x y : ℝ)
  (h1 : x + y ≥ 3)
  (h2 : x - y ≥ -1)
  (h3 : 2 * x - y ≤ 3) :
  (∀ (x y : ℝ), (x + y ≥ 3) ∧ (x - y ≥ -1) ∧ (2 * x - y ≤ 3) → (∀ k, k = y / x → k ≤ 2)) :=
by
  sorry

end max_value_of_y_over_x_l1531_153192


namespace laborer_income_l1531_153191

theorem laborer_income (I : ℕ) (debt : ℕ) 
  (h1 : 6 * I < 420) 
  (h2 : 4 * I = 240 + debt + 30) 
  (h3 : debt = 420 - 6 * I) : 
  I = 69 := by
  sorry

end laborer_income_l1531_153191


namespace melissa_coupe_sale_l1531_153123

theorem melissa_coupe_sale :
  ∃ x : ℝ, (0.02 * x + 0.02 * 2 * x = 1800) ∧ x = 30000 :=
by
  sorry

end melissa_coupe_sale_l1531_153123


namespace logarithmic_expression_range_l1531_153178

theorem logarithmic_expression_range (a : ℝ) : 
  (a - 2 > 0) ∧ (5 - a > 0) ∧ (a - 2 ≠ 1) ↔ (2 < a ∧ a < 3) ∨ (3 < a ∧ a < 5) := 
by
  sorry

end logarithmic_expression_range_l1531_153178


namespace find_f_minus1_plus_f_2_l1531_153120

variable (f : ℝ → ℝ)

def even_function := ∀ x : ℝ, f (-x) = f x

def symmetric_about_origin := ∀ x : ℝ, f (x + 1) = -f (-(x + 1))

def f_value_at_zero := f 0 = 1

theorem find_f_minus1_plus_f_2 :
  even_function f →
  symmetric_about_origin f →
  f_value_at_zero f →
  f (-1) + f 2 = -1 :=
by
  intros
  sorry

end find_f_minus1_plus_f_2_l1531_153120


namespace smallest_four_digit_int_mod_9_l1531_153159

theorem smallest_four_digit_int_mod_9 :
  ∃ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ n % 9 = 5 ∧ ∀ m : ℕ, 1000 ≤ m ∧ m < 10000 ∧ m % 9 = 5 → n ≤ m :=
sorry

end smallest_four_digit_int_mod_9_l1531_153159


namespace sequence_properties_l1531_153151

variables {a : ℕ → ℤ} {b : ℕ → ℤ} {d : ℤ} {q : ℤ}

def arithmetic_sequence (a : ℕ → ℤ) (d : ℤ) : Prop :=
∀ n, a n = a 1 + (n - 1) * d

def geometric_sequence (b : ℕ → ℤ) (q : ℤ) : Prop :=
∀ n, b n = b 1 * q^(n - 1)

theorem sequence_properties
  (ha : arithmetic_sequence a d)
  (hb : geometric_sequence b q)
  (h1 : 2 * a 5 - a 3 = 3)
  (h2 : b 2 = 1)
  (h3 : b 4 = 4) :
  a 7 = 3 ∧ b 6 = 16 ∧ (q = 2 ∨ q = -2) :=
by
  sorry

end sequence_properties_l1531_153151


namespace min_value_of_fraction_l1531_153155

noncomputable def min_val (a b : ℝ) : ℝ :=
  1 / a + 2 * b

theorem min_value_of_fraction (a b : ℝ) (h₁ : a > 0) (h₂ : b > 0) (h₃ : 2 * a * b + 3 = b) :
  min_val a b = 8 + 4 * Real.sqrt 3 :=
sorry

end min_value_of_fraction_l1531_153155


namespace polynomial_divisibility_l1531_153101

noncomputable def polynomial_with_positive_int_coeffs : Type :=
{ f : ℕ → ℕ // ∀ m n : ℕ, f m < f n ↔ m < n }

theorem polynomial_divisibility
  (f : polynomial_with_positive_int_coeffs)
  (n : ℕ) (hn : n > 0) :
  f.1 n ∣ f.1 (f.1 n + 1) ↔ n = 1 :=
sorry

end polynomial_divisibility_l1531_153101


namespace integer_side_lengths_triangle_l1531_153187

theorem integer_side_lengths_triangle :
  ∃ (a b c : ℤ), (abc = 2 * (a - 1) * (b - 1) * (c - 1)) ∧
            (a = 8 ∧ b = 7 ∧ c = 3 ∨ a = 6 ∧ b = 5 ∧ c = 4) := 
by
  sorry

end integer_side_lengths_triangle_l1531_153187


namespace factorizations_of_4050_l1531_153133

theorem factorizations_of_4050 :
  ∃! (a b : ℕ), 10 ≤ a ∧ a < 100 ∧ 10 ≤ b ∧ b < 100 ∧ a * b = 4050 :=
by
  sorry

end factorizations_of_4050_l1531_153133


namespace joan_missed_games_l1531_153183

theorem joan_missed_games :
  ∀ (total_games attended_games missed_games : ℕ),
  total_games = 864 →
  attended_games = 395 →
  missed_games = total_games - attended_games →
  missed_games = 469 :=
by
  intros total_games attended_games missed_games H1 H2 H3
  rw [H1, H2] at H3
  exact H3

end joan_missed_games_l1531_153183


namespace each_sibling_gets_13_pencils_l1531_153165

theorem each_sibling_gets_13_pencils (colored_pencils : ℕ) (black_pencils : ℕ) (siblings : ℕ) (kept_pencils : ℕ) 
  (hyp1 : colored_pencils = 14) (hyp2 : black_pencils = 35) (hyp3 : siblings = 3) (hyp4 : kept_pencils = 10) :
  (colored_pencils + black_pencils - kept_pencils) / siblings = 13 :=
by sorry

end each_sibling_gets_13_pencils_l1531_153165


namespace number_of_people_in_first_group_l1531_153166

variable (W : ℝ)  -- Amount of work
variable (P : ℝ)  -- Number of people in the first group

-- Condition 1: P people can do 3W work in 3 days
def condition1 : Prop := P * (W / 1) * 3 = 3 * W

-- Condition 2: 5 people can do 5W work in 3 days
def condition2 : Prop := 5 * (W / 1) * 3 = 5 * W

-- Theorem to prove: The number of people in the first group is 3
theorem number_of_people_in_first_group (h1 : condition1 W P) (h2 : condition2 W) : P = 3 :=
by
  sorry

end number_of_people_in_first_group_l1531_153166


namespace power_mod_result_l1531_153174

-- Define the modulus and base
def mod : ℕ := 8
def base : ℕ := 7
def exponent : ℕ := 202

-- State the theorem
theorem power_mod_result :
  (base ^ exponent) % mod = 1 :=
by
  sorry

end power_mod_result_l1531_153174


namespace factory_Y_bulbs_proportion_l1531_153190

theorem factory_Y_bulbs_proportion :
  (0.60 * 0.59 + 0.40 * P_Y = 0.62) → (P_Y = 0.665) :=
by
  sorry

end factory_Y_bulbs_proportion_l1531_153190


namespace no_integer_solutions_l1531_153196

theorem no_integer_solutions (x y : ℤ) :
  ¬ (x^2 + 3 * x * y - 2 * y^2 = 122) :=
sorry

end no_integer_solutions_l1531_153196


namespace no_fractional_linear_function_l1531_153156

noncomputable def fractional_linear_function (a b c d x : ℝ) : ℝ := (a * x + b) / (c * x + d)

theorem no_fractional_linear_function (a b c d : ℝ) :
  ∀ x : ℝ, c ≠ 0 → 
  (fractional_linear_function a b c d x + fractional_linear_function b (-d) c (-a) x ≠ -2) :=
by
  sorry

end no_fractional_linear_function_l1531_153156


namespace final_position_D_l1531_153172

open Function

-- Define the original points of the parallelogram
def A : ℝ × ℝ := (3, 4)
def B : ℝ × ℝ := (5, 8)
def C : ℝ × ℝ := (9, 4)
def D : ℝ × ℝ := (7, 0)

-- Define the reflection across the y-axis
def reflect_y_axis (p : ℝ × ℝ) : ℝ × ℝ := (-p.1, p.2)

-- Define the translation by (0, 1)
def translate_up (p : ℝ × ℝ) : ℝ × ℝ := (p.1, p.2 + 1)
def translate_down (p : ℝ × ℝ) : ℝ × ℝ := (p.1, p.2 - 1)

-- Define the reflection across y = x
def reflect_y_eq_x (p : ℝ × ℝ) : ℝ × ℝ := (p.2, p.1)

-- Combine the transformations to get the final reflection across y = x - 1
def reflect_across_y_eq_x_minus_1 (p : ℝ × ℝ) : ℝ × ℝ :=
  translate_down (reflect_y_eq_x (translate_up p))

-- Prove that the final position of D after the two transformations is (1, -8)
theorem final_position_D'' : reflect_across_y_eq_x_minus_1 (reflect_y_axis D) = (1, -8) :=
  sorry

end final_position_D_l1531_153172


namespace speed_of_stream_l1531_153102

-- Definitions based on the conditions provided
def speed_still_water : ℝ := 15
def upstream_time_ratio := 2

-- Proof statement
theorem speed_of_stream (v : ℝ) 
  (h1 : ∀ d t_up t_down, (15 - v) * t_up = d ∧ (15 + v) * t_down = d ∧ t_up = upstream_time_ratio * t_down) : 
  v = 5 :=
sorry

end speed_of_stream_l1531_153102


namespace minimize_travel_time_l1531_153161

theorem minimize_travel_time
  (a b c d : ℝ)
  (v₁ v₂ v₃ v₄ : ℝ)
  (h1 : a > b)
  (h2 : b > c)
  (h3 : c > d)
  (h4 : v₁ > v₂)
  (h5 : v₂ > v₃)
  (h6 : v₃ > v₄) : 
  (a / v₁ + b / v₂ + c / v₃ + d / v₄) ≤ (a / v₁ + b / v₄ + c / v₃ + d / v₂) :=
sorry

end minimize_travel_time_l1531_153161


namespace has_root_in_interval_l1531_153113

def f (x : ℝ) := x^3 - 3*x - 3

theorem has_root_in_interval : ∃ c ∈ (Set.Ioo (2:ℝ) 3), f c = 0 :=
by 
    sorry

end has_root_in_interval_l1531_153113


namespace gasoline_price_increase_percentage_l1531_153115

theorem gasoline_price_increase_percentage : 
  ∀ (highest_price lowest_price : ℝ), highest_price = 24 → lowest_price = 18 → 
  ((highest_price - lowest_price) / lowest_price) * 100 = 33.33 :=
by
  intros highest_price lowest_price h_highest h_lowest
  rw [h_highest, h_lowest]
  -- To be completed in the proof
  sorry

end gasoline_price_increase_percentage_l1531_153115


namespace jellybeans_original_count_l1531_153114

theorem jellybeans_original_count (x : ℝ) (h : (0.75)^3 * x = 27) : x = 64 := 
sorry

end jellybeans_original_count_l1531_153114


namespace buratino_cafe_workdays_l1531_153176

-- Define the conditions as given in the problem statement
def days_in_april (d : Nat) : Prop := d >= 1 ∧ d <= 30
def is_monday (d : Nat) : Prop := d = 1 ∨ d = 8 ∨ d = 15 ∨ d = 22 ∨ d = 29

-- Define the period April 1 to April 13
def period_1_13 (d : Nat) : Prop := d >= 1 ∧ d <= 13

-- Define the statements made by Kolya
def kolya_statement_1 : Prop := ∀ d : Nat, days_in_april d → (d >= 1 ∧ d <= 20) → ¬is_monday d → ∃ n : Nat, n = 18
def kolya_statement_2 : Prop := ∀ d : Nat, days_in_april d → (d >= 10 ∧ d <= 30) → ¬is_monday d → ∃ n : Nat, n = 18

-- Define the condition stating Kolya made a mistake once
def kolya_made_mistake_once : Prop := kolya_statement_1 ∨ kolya_statement_2

-- The proof problem: Prove the number of working days from April 1 to April 13 is 11
theorem buratino_cafe_workdays : period_1_13 (d) → (¬is_monday d → (∃ n : Nat, n = 11)) := sorry

end buratino_cafe_workdays_l1531_153176


namespace mix_solutions_l1531_153160

variables (Vx : ℚ)

def alcohol_content_x (Vx : ℚ) : ℚ := 0.10 * Vx
def alcohol_content_y : ℚ := 0.30 * 450
def final_alcohol_content (Vx : ℚ) : ℚ := 0.22 * (Vx + 450)

theorem mix_solutions (Vx : ℚ) (h : 0.10 * Vx + 0.30 * 450 = 0.22 * (Vx + 450)) :
  Vx = 300 :=
sorry

end mix_solutions_l1531_153160


namespace remainder_correct_l1531_153138

noncomputable def p (x : ℝ) : ℝ := 3 * x ^ 8 - 2 * x ^ 5 + 5 * x ^ 3 - 9
noncomputable def d (x : ℝ) : ℝ := x ^ 2 - 2 * x + 1
noncomputable def r (x : ℝ) : ℝ := 29 * x - 32

theorem remainder_correct (x : ℝ) :
  ∃ q : ℝ → ℝ, p x = d x * q x + r x :=
sorry

end remainder_correct_l1531_153138


namespace positive_diff_after_add_five_l1531_153142

theorem positive_diff_after_add_five (y : ℝ) 
  (h : (45 + y)/2 = 32) : |45 - (y + 5)| = 21 :=
by 
  sorry

end positive_diff_after_add_five_l1531_153142


namespace people_per_car_l1531_153195

theorem people_per_car (total_people : ℝ) (total_cars : ℝ) (h1 : total_people = 189) (h2 : total_cars = 3.0) : total_people / total_cars = 63 := 
by
  sorry

end people_per_car_l1531_153195


namespace suff_but_not_necessary_condition_l1531_153169

theorem suff_but_not_necessary_condition (x y : ℝ) :
  (xy ≠ 6 → x ≠ 2 ∨ y ≠ 3) ∧ ¬ (x ≠ 2 ∨ y ≠ 3 → xy ≠ 6) :=
by
  sorry

end suff_but_not_necessary_condition_l1531_153169


namespace filling_material_heavier_than_sand_l1531_153136

noncomputable def percentage_increase (full_sandbag_weight : ℝ) (partial_fill_percent : ℝ) (full_material_weight : ℝ) : ℝ :=
  let sand_weight := (partial_fill_percent / 100) * full_sandbag_weight
  let material_weight := full_material_weight
  let weight_increase := material_weight - sand_weight
  (weight_increase / sand_weight) * 100

theorem filling_material_heavier_than_sand :
  let full_sandbag_weight := 250
  let partial_fill_percent := 80
  let full_material_weight := 280
  percentage_increase full_sandbag_weight partial_fill_percent full_material_weight = 40 :=
by
  sorry

end filling_material_heavier_than_sand_l1531_153136


namespace cody_books_reading_l1531_153168

theorem cody_books_reading :
  ∀ (total_books first_week_books second_week_books subsequent_week_books : ℕ),
    total_books = 54 →
    first_week_books = 6 →
    second_week_books = 3 →
    subsequent_week_books = 9 →
    (2 + (total_books - (first_week_books + second_week_books)) / subsequent_week_books) = 7 :=
by
  -- Using sorry to mark the proof as incomplete.
  sorry

end cody_books_reading_l1531_153168


namespace geometric_seq_inequality_l1531_153179

theorem geometric_seq_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h : b^2 = a * c) : a^2 + b^2 + c^2 > (a - b + c)^2 :=
by
  sorry

end geometric_seq_inequality_l1531_153179


namespace find_square_tiles_l1531_153127

variables (t s p : ℕ)

theorem find_square_tiles
  (h1 : t + s + p = 30)
  (h2 : 3 * t + 4 * s + 5 * p = 120) :
  s = 10 :=
by
  sorry

end find_square_tiles_l1531_153127


namespace cost_per_book_l1531_153145

theorem cost_per_book (num_animal_books : ℕ) (num_space_books : ℕ) (num_train_books : ℕ) (total_cost : ℕ) 
                      (h1 : num_animal_books = 10) (h2 : num_space_books = 1) (h3 : num_train_books = 3) (h4 : total_cost = 224) :
  (total_cost / (num_animal_books + num_space_books + num_train_books) = 16) :=
by sorry

end cost_per_book_l1531_153145


namespace triplet_solution_l1531_153162

theorem triplet_solution (a b c : ℝ) (h₀ : a ≠ 0) (h₁ : b ≠ 0) (h₂ : c ≠ 0) :
  (a + b + c = (1 / a) + (1 / b) + (1 / c) ∧ a ^ 2 + b ^ 2 + c ^ 2 = (1 / a ^ 2) + (1 / b ^ 2) + (1 / c ^ 2))
  ↔ (∃ x, (a = 1 ∨ a = -1 ∨ a = x ∨ a = 1/x) ∧
           (b = 1 ∨ b = -1 ∨ b = x ∨ b = 1/x) ∧
           (c = 1 ∨ c = -1 ∨ c = x ∨ c = 1/x)) := 
sorry

end triplet_solution_l1531_153162


namespace largest_k_statement_l1531_153146

noncomputable def largest_k (n : ℕ) : ℕ :=
  n - 2

theorem largest_k_statement (S : Finset ℕ) (A : Finset (Finset ℕ)) (h1 : ∀ (A_i : Finset ℕ), A_i ∈ A → 2 ≤ A_i.card ∧ A_i.card < S.card) : 
  largest_k S.card = S.card - 2 :=
by
  sorry

end largest_k_statement_l1531_153146


namespace students_at_start_of_year_l1531_153126

-- Define the initial number of students as a variable S
variables (S : ℕ)

-- Define the conditions
def condition_1 := S - 18 + 14 = 29

-- State the theorem to be proved
theorem students_at_start_of_year (h : condition_1 S) : S = 33 :=
sorry

end students_at_start_of_year_l1531_153126


namespace pond_water_amount_l1531_153158

-- Definitions based on the problem conditions
def initial_gallons := 500
def evaporation_rate := 1
def additional_gallons := 10
def days_period := 35
def additional_days_interval := 7

-- Calculations based on the conditions
def total_evaporation := days_period * evaporation_rate
def total_additional_gallons := (days_period / additional_days_interval) * additional_gallons

-- Theorem stating the final amount of water
theorem pond_water_amount : initial_gallons - total_evaporation + total_additional_gallons = 515 := by
  -- Proof is omitted
  sorry

end pond_water_amount_l1531_153158


namespace quadruple_perimeter_l1531_153104

-- Define the rectangle's original and expanded dimensions and perimeters
def original_perimeter (a b : ℝ) := 2 * (a + b)
def new_perimeter (a b : ℝ) := 2 * ((4 * a) + (4 * b))

-- Statement to be proved
theorem quadruple_perimeter (a b : ℝ) : new_perimeter a b = 4 * original_perimeter a b :=
  sorry

end quadruple_perimeter_l1531_153104


namespace inequality_range_l1531_153139

theorem inequality_range (a : ℝ) : (∀ x : ℝ, x^2 - 1 ≥ a * |x - 1|) → a ≤ -2 :=
by
  sorry

end inequality_range_l1531_153139


namespace matrix_scalars_exist_l1531_153128

namespace MatrixProof

def B : Matrix (Fin 2) (Fin 2) ℝ :=
  ![![0, 1], ![4, -1]]

theorem matrix_scalars_exist :
  ∃ r s : ℝ, B^6 = r • B + s • (1 : Matrix (Fin 2) (Fin 2) ℝ) ∧ r = 0 ∧ s = 64 := by
  sorry

end MatrixProof

end matrix_scalars_exist_l1531_153128


namespace composite_number_property_l1531_153167

theorem composite_number_property (n : ℕ) 
  (h1 : n > 1) 
  (h2 : ¬ Prime n) 
  (h3 : ∀ (d : ℕ), d ∣ n → 1 ≤ d → d < n → n - 20 ≤ d ∧ d ≤ n - 12) :
  n = 21 ∨ n = 25 :=
by
  sorry

end composite_number_property_l1531_153167


namespace unique_zero_function_l1531_153121

theorem unique_zero_function
    (f : ℝ → ℝ)
    (H : ∀ x y : ℝ, x + y ≠ 0 → f (x * y) = (f x + f y) / (x + y)) :
    ∀ x : ℝ, f x = 0 := 
by 
     sorry

end unique_zero_function_l1531_153121


namespace distinct_ball_placement_l1531_153147

def num_distributions (balls boxes : ℕ) : ℕ :=
  if boxes = 3 then 243 - 32 + 16 else 0

theorem distinct_ball_placement : num_distributions 5 3 = 227 :=
by
  sorry

end distinct_ball_placement_l1531_153147


namespace geom_series_min_q_l1531_153140

theorem geom_series_min_q (p q r : ℝ) (hp : 0 < p) (hq : 0 < q) (hr : 0 < r)
  (h_geom : ∃ k : ℝ, q = p * k ∧ r = q * k)
  (hpqr : p * q * r = 216) : q = 6 :=
sorry

end geom_series_min_q_l1531_153140


namespace pos_divisors_180_l1531_153150

theorem pos_divisors_180 : 
  (∃ a b c : ℕ, 180 = (2^a) * (3^b) * (5^c) ∧ a = 2 ∧ b = 2 ∧ c = 1) →
  (∃ n : ℕ, n = 18 ∧ n = (a + 1) * (b + 1) * (c + 1)) := by
  sorry

end pos_divisors_180_l1531_153150


namespace bridge_length_l1531_153132

theorem bridge_length (rate : ℝ) (time_minutes : ℝ) (length : ℝ) 
    (rate_condition : rate = 10) 
    (time_condition : time_minutes = 15) : 
    length = 2.5 := 
by
  sorry

end bridge_length_l1531_153132


namespace sum_of_exponents_correct_l1531_153175

-- Define the initial expression
def original_expr (a b c : ℤ) : ℤ := 40 * a^6 * b^9 * c^14

-- Define the simplified expression outside the radical
def simplified_outside_expr (a b c : ℤ) : ℤ := a * b^3 * c^3

-- Define the sum of the exponents
def sum_of_exponents : ℕ := 1 + 3 + 3

-- Prove that the given conditions lead to the sum of the exponents being 7
theorem sum_of_exponents_correct (a b c : ℤ) :
  original_expr a b c = 40 * a^6 * b^9 * c^14 →
  simplified_outside_expr a b c = a * b^3 * c^3 →
  sum_of_exponents = 7 :=
by
  intros
  -- Proof goes here
  sorry

end sum_of_exponents_correct_l1531_153175


namespace fish_population_l1531_153193

theorem fish_population (x : ℕ) : 
  (1: ℝ) / 45 = (100: ℝ) / ↑x -> x = 1125 :=
by
  sorry

end fish_population_l1531_153193


namespace isosceles_triangle_perimeter_l1531_153185

theorem isosceles_triangle_perimeter (a b c : ℝ) (h₀ : a = 5) (h₁ : b = 10) 
  (h₂ : c = 10 ∨ c = 5) (h₃ : a = b ∨ b = c ∨ c = a) 
  (triangle_inequality : a + b > c ∧ a + c > b ∧ b + c > a) :
  a + b + c = 25 := by
  sorry

end isosceles_triangle_perimeter_l1531_153185


namespace find_a9_l1531_153184

variable (S : ℕ → ℤ) (a : ℕ → ℤ)
variable (d a1 : ℤ)

def arithmetic_seq (n : ℕ) : ℤ :=
  a1 + ↑n * d

def sum_arithmetic_seq (n : ℕ) : ℤ :=
  n * a1 + (n * (n - 1) / 2) * d

axiom h1 : sum_arithmetic_seq 8 = 4 * arithmetic_seq 3
axiom h2 : arithmetic_seq 7 = -2

theorem find_a9 : arithmetic_seq 9 = -6 :=
by
  sorry

end find_a9_l1531_153184


namespace find_a_l1531_153180

theorem find_a (a : ℝ) : 
  let A := {1, 2, 3}
  let B := {x : ℝ | x^2 - (a + 1) * x + a = 0}
  A ∪ B = A → a = 1 ∨ a = 2 ∨ a = 3 :=
by
  intros
  sorry

end find_a_l1531_153180


namespace percent_of_d_is_e_l1531_153189

variable (a b c d e : ℝ)
variable (h1 : d = 0.40 * a)
variable (h2 : d = 0.35 * b)
variable (h3 : e = 0.50 * b)
variable (h4 : e = 0.20 * c)
variable (h5 : c = 0.30 * a)
variable (h6 : c = 0.25 * b)

theorem percent_of_d_is_e : (e / d) * 100 = 15 :=
by sorry

end percent_of_d_is_e_l1531_153189


namespace quadratic_equation_no_real_roots_l1531_153181

theorem quadratic_equation_no_real_roots :
  ∀ (x : ℝ), ¬ (x^2 - 2 * x + 3 = 0) :=
by
  intro x
  sorry

end quadratic_equation_no_real_roots_l1531_153181


namespace range_of_a_l1531_153124

theorem range_of_a (a : ℝ) : 
  (∀ n : ℕ, n ≥ 8 → (a * (n^2) + n + 5) > (a * ((n + 1)^2) + (n + 1) + 5)) → 
  (a * (1^2) + 1 + 5 < a * (2^2) + 2 + 5) →
  (a * (2^2) + 2 + 5 < a * (3^2) + 3 + 5) →
  (a * (3^2) + 3 + 5 < a * (4^2) + 4 + 5) →
  (- (1 / 7) < a ∧ a < - (1 / 17)) :=
by
  sorry

end range_of_a_l1531_153124


namespace roots_calc_l1531_153141

theorem roots_calc {a b c d : ℝ} (h1: a ≠ 0) (h2 : 125 * a + 25 * b + 5 * c + d = 0) (h3 : -27 * a + 9 * b - 3 * c + d = 0) :
  (b + c) / a = -19 :=
by
  sorry

end roots_calc_l1531_153141


namespace calc_15_op_and_op2_l1531_153182

def op1 (x : ℤ) : ℤ := 10 - x
def op2 (x : ℤ) : ℤ := x - 10

theorem calc_15_op_and_op2 :
  op2 (op1 15) = -15 :=
by
  sorry

end calc_15_op_and_op2_l1531_153182


namespace shadow_boundary_l1531_153170

theorem shadow_boundary (r : ℝ) (O P : ℝ × ℝ × ℝ) :
  r = 2 → O = (0, 0, 2) → P = (0, -2, 4) → ∀ x : ℝ, ∃ y : ℝ, y = -10 :=
by sorry

end shadow_boundary_l1531_153170


namespace candy_bar_calories_l1531_153130

theorem candy_bar_calories :
  let calA := 150
  let calB := 200
  let calC := 250
  let countA := 2
  let countB := 3
  let countC := 4
  (countA * calA + countB * calB + countC * calC) = 1900 :=
by
  sorry

end candy_bar_calories_l1531_153130


namespace probability_digits_different_l1531_153117

noncomputable def probability_all_digits_different : ℚ :=
  have tens_digits_probability := (9 / 9) * (8 / 9) * (7 / 9)
  have ones_digits_probability := (10 / 10) * (9 / 10) * (8 / 10)
  (tens_digits_probability * ones_digits_probability)

theorem probability_digits_different :
  probability_all_digits_different = 112 / 225 :=
by 
  -- The proof would go here, but it is not required for this task.
  sorry

end probability_digits_different_l1531_153117


namespace vanya_correct_answers_l1531_153119

theorem vanya_correct_answers (x : ℕ) (q : ℕ) (correct_gain : ℕ) (incorrect_loss : ℕ) (net_change : ℤ) :
  q = 50 ∧ correct_gain = 7 ∧ incorrect_loss = 3 ∧ net_change = 7 * x - 3 * (q - x) ∧ net_change = 0 →
  x = 15 :=
by
  sorry

end vanya_correct_answers_l1531_153119


namespace yellow_balls_are_24_l1531_153134

theorem yellow_balls_are_24 (x y z : ℕ) (h1 : x + y + z = 68) 
                             (h2 : y = 2 * x) (h3 : 3 * z = 4 * y) : y = 24 :=
by
  sorry

end yellow_balls_are_24_l1531_153134


namespace overall_average_score_l1531_153100

-- Definitions based on given conditions
def n_m : ℕ := 8   -- number of male students
def avg_m : ℚ := 87  -- average score of male students
def n_f : ℕ := 12  -- number of female students
def avg_f : ℚ := 92  -- average score of female students

-- The target statement to prove
theorem overall_average_score (n_m : ℕ) (avg_m : ℚ) (n_f : ℕ) (avg_f : ℚ) (overall_avg : ℚ) :
  n_m = 8 ∧ avg_m = 87 ∧ n_f = 12 ∧ avg_f = 92 → overall_avg = 90 :=
by
  sorry

end overall_average_score_l1531_153100


namespace circle_radius_tangent_l1531_153173

theorem circle_radius_tangent (a : ℝ) (R : ℝ) (h1 : a = 25)
  (h2 : ∀ BP DE CP CE, BP = 2 ∧ DE = 2 ∧ CP = 23 ∧ CE = 23 ∧ BP + CP = a ∧ DE + CE = a)
  : R = 17 :=
sorry

end circle_radius_tangent_l1531_153173


namespace remainder_of_452867_div_9_l1531_153106

theorem remainder_of_452867_div_9 : (452867 % 9) = 5 := by
  sorry

end remainder_of_452867_div_9_l1531_153106


namespace average_of_first_13_even_numbers_l1531_153198

-- Definition of the first 13 even numbers
def first_13_even_numbers := [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26]

-- The sum of the first 13 even numbers
def sum_of_first_13_even_numbers : ℕ := 182

-- The number of these even numbers
def number_of_even_numbers : ℕ := 13

-- The average of the first 13 even numbers
theorem average_of_first_13_even_numbers : (sum_of_first_13_even_numbers / number_of_even_numbers) = 14 := by
  sorry

end average_of_first_13_even_numbers_l1531_153198


namespace ira_addition_olya_subtraction_addition_l1531_153144

theorem ira_addition (x : ℤ) (h : (11 + x) / (41 + x : ℚ) = 3 / 8) : x = 7 :=
  sorry

theorem olya_subtraction_addition (y : ℤ) (h : (37 - y) / (63 + y : ℚ) = 3 / 17) : y = 22 :=
  sorry

end ira_addition_olya_subtraction_addition_l1531_153144


namespace unique_real_solution_l1531_153118

theorem unique_real_solution (x y z : ℝ) :
  (x^3 - 3 * x = 4 - y) ∧ 
  (2 * y^3 - 6 * y = 6 - z) ∧ 
  (3 * z^3 - 9 * z = 8 - x) ↔ 
  x = 2 ∧ y = 2 ∧ z = 2 :=
by sorry

end unique_real_solution_l1531_153118


namespace probability_stopping_in_C_l1531_153186

noncomputable def probability_C : ℚ :=
  let P_A := 1 / 5
  let P_B := 1 / 5
  let x := (1 - (P_A + P_B)) / 3
  x

theorem probability_stopping_in_C :
  probability_C = 1 / 5 :=
by
  unfold probability_C
  sorry

end probability_stopping_in_C_l1531_153186


namespace apples_picked_correct_l1531_153116

-- Define the conditions as given in the problem
def apples_given_to_Melanie : ℕ := 27
def apples_left : ℕ := 16

-- Define the problem statement
def total_apples_picked := apples_given_to_Melanie + apples_left

-- Prove that the total apples picked is equal to 43 given the conditions
theorem apples_picked_correct : total_apples_picked = 43 := by
  sorry

end apples_picked_correct_l1531_153116


namespace terminal_side_in_third_quadrant_l1531_153110

def is_equivalent_angle (a b : ℝ) : Prop := ∃ k : ℤ, a = b + k * 360

def in_third_quadrant (θ : ℝ) : Prop :=
  180 < θ ∧ θ < 270

theorem terminal_side_in_third_quadrant : 
  ∀ θ, θ = 600 → in_third_quadrant (θ % 360) :=
by
  intro θ
  intro hθ
  sorry

end terminal_side_in_third_quadrant_l1531_153110


namespace ribbon_fraction_per_box_l1531_153112

theorem ribbon_fraction_per_box 
  (total_ribbon_used : ℚ)
  (number_of_boxes : ℕ)
  (h1 : total_ribbon_used = 5/8)
  (h2 : number_of_boxes = 5) :
  (total_ribbon_used / number_of_boxes = 1/8) :=
by
  sorry

end ribbon_fraction_per_box_l1531_153112


namespace largest_angle_of_triangle_l1531_153177

theorem largest_angle_of_triangle (x : ℝ) (h_ratio : (5 * x) + (6 * x) + (7 * x) = 180) :
  7 * x = 70 := 
sorry

end largest_angle_of_triangle_l1531_153177


namespace no_real_solutions_sufficient_not_necessary_l1531_153197

theorem no_real_solutions_sufficient_not_necessary (m : ℝ) : 
  (|m| < 1) → (m^2 < 4) :=
by
  sorry

end no_real_solutions_sufficient_not_necessary_l1531_153197


namespace students_table_tennis_not_basketball_l1531_153143

variable (total_students : ℕ)
variable (students_like_basketball : ℕ)
variable (students_like_table_tennis : ℕ)
variable (students_dislike_both : ℕ)

theorem students_table_tennis_not_basketball 
  (h_total : total_students = 40)
  (h_basketball : students_like_basketball = 17)
  (h_table_tennis : students_like_table_tennis = 20)
  (h_dislike : students_dislike_both = 8) : 
  ∃ (students_table_tennis_not_basketball : ℕ), students_table_tennis_not_basketball = 15 :=
by
  sorry

end students_table_tennis_not_basketball_l1531_153143


namespace smallest_digit_divisible_by_9_l1531_153125

theorem smallest_digit_divisible_by_9 : 
  ∃ d : ℕ, (∃ m : ℕ, m = 2 + 4 + d + 6 + 0 ∧ m % 9 = 0 ∧ d < 10) ∧ d = 6 :=
by
  sorry

end smallest_digit_divisible_by_9_l1531_153125


namespace cos_beta_value_l1531_153199

theorem cos_beta_value (α β : ℝ) (hα : 0 < α ∧ α < π / 2) (hβ : 0 < β ∧ β < π / 2)
  (hα_cos : Real.cos α = 4 / 5) (hαβ_cos : Real.cos (α + β) = -16 / 65) : 
  Real.cos β = 5 / 13 := 
sorry

end cos_beta_value_l1531_153199


namespace equation_holds_except_two_values_l1531_153188

noncomputable def check_equation (a y : ℝ) (h : a ≠ 0) : Prop :=
  (a / (a + y) + y / (a - y)) / (y / (a + y) - a / (a - y)) = -1 ↔ y ≠ a ∧ y ≠ -a

theorem equation_holds_except_two_values (a y: ℝ) (h: a ≠ 0): check_equation a y h := sorry

end equation_holds_except_two_values_l1531_153188


namespace fraction_identity_l1531_153164

theorem fraction_identity (a b : ℚ) (h : (a - 2 * b) / b = 3 / 5) : a / b = 13 / 5 :=
sorry

end fraction_identity_l1531_153164


namespace sequence_explicit_formula_l1531_153109

theorem sequence_explicit_formula (a : ℕ → ℤ) (n : ℕ) :
  a 0 = 2 →
  (∀ n, a (n+1) = a n - n + 3) →
  a n = -((n * (n + 1)) / 2) + 3 * n + 2 :=
by
  intros h0 h_rec
  sorry

end sequence_explicit_formula_l1531_153109


namespace sufficient_condition_for_perpendicular_l1531_153111

variables (m n : Line) (α β : Plane)

def are_parallel (l1 l2 : Line) : Prop := sorry
def line_perpendicular_to_plane (l : Line) (p : Plane) : Prop := sorry

theorem sufficient_condition_for_perpendicular :
  (are_parallel m n) ∧ (line_perpendicular_to_plane n α) → (line_perpendicular_to_plane m α) :=
sorry

end sufficient_condition_for_perpendicular_l1531_153111


namespace trucks_transport_l1531_153103

variables {x y : ℝ}

theorem trucks_transport (h1 : 2 * x + 3 * y = 15.5)
                         (h2 : 5 * x + 6 * y = 35) :
  3 * x + 2 * y = 17 :=
sorry

end trucks_transport_l1531_153103


namespace inequality_range_l1531_153107

noncomputable def f (x : ℝ) : ℝ := |x - 1| + |x - 2|

theorem inequality_range (a b x: ℝ) (h : a ≠ 0) :
  (|a + b| + |a - b|) ≥ |a| * f x → 1 ≤ x ∧ x ≤ 2 :=
by
  intro h1
  unfold f at h1
  sorry

end inequality_range_l1531_153107


namespace arccos_half_eq_pi_div_three_l1531_153129

theorem arccos_half_eq_pi_div_three : Real.arccos (1 / 2) = Real.pi / 3 := 
sorry

end arccos_half_eq_pi_div_three_l1531_153129


namespace product_of_numbers_eq_120_l1531_153153

theorem product_of_numbers_eq_120 (x y P : ℝ) (h1 : x + y = 23) (h2 : x^2 + y^2 = 289) (h3 : x * y = P) : P = 120 := 
sorry

end product_of_numbers_eq_120_l1531_153153


namespace initial_price_after_markup_l1531_153154

theorem initial_price_after_markup 
  (wholesale_price : ℝ) 
  (h_markup_80 : ∀ P, P = wholesale_price → 1.80 * P = 1.80 * wholesale_price)
  (h_markup_diff : ∀ P, P = wholesale_price → 2.00 * P - 1.80 * P = 3) 
  : 1.80 * wholesale_price = 27 := 
by
  sorry

end initial_price_after_markup_l1531_153154


namespace decreasing_interval_l1531_153171

-- Define the function f
def f (x : ℝ) : ℝ := 3 * x^5 - 5 * x^3

-- Define the derivative of f
def f' (x : ℝ) : ℝ := 15 * x^4 - 15 * x^2

-- State the theorem
theorem decreasing_interval : ∀ x : ℝ, -1 < x ∧ x < 1 → f' x < 0 :=
by sorry

end decreasing_interval_l1531_153171


namespace watermelon_weight_l1531_153194

theorem watermelon_weight (B W : ℝ) (n : ℝ) 
  (h1 : B + n * W = 63) 
  (h2 : B + (n / 2) * W = 34) : 
  n * W = 58 :=
sorry

end watermelon_weight_l1531_153194


namespace tan_of_neg_23_over_3_pi_l1531_153122

theorem tan_of_neg_23_over_3_pi : (Real.tan (- 23 / 3 * Real.pi) = Real.sqrt 3) :=
by
  sorry

end tan_of_neg_23_over_3_pi_l1531_153122


namespace sufficient_but_not_necessary_condition_x_gt_5_x_gt_3_l1531_153108

theorem sufficient_but_not_necessary_condition_x_gt_5_x_gt_3 :
  ∀ x : ℝ, (x > 5 → x > 3) ∧ (∃ x : ℝ, x > 3 ∧ x ≤ 5) :=
by
  sorry

end sufficient_but_not_necessary_condition_x_gt_5_x_gt_3_l1531_153108


namespace find_x_in_equation_l1531_153131

theorem find_x_in_equation :
  ∃ x : ℝ, x / 18 * (x / 162) = 1 ∧ x = 54 :=
by
  sorry

end find_x_in_equation_l1531_153131


namespace inequality_solution_l1531_153137

theorem inequality_solution (x : ℝ) (h : 0 < x) : x^3 - 9*x^2 + 52*x > 0 := 
sorry

end inequality_solution_l1531_153137


namespace tan_45_degrees_l1531_153163

theorem tan_45_degrees : Real.tan (Real.pi / 4) = 1 := by
  sorry

end tan_45_degrees_l1531_153163


namespace carolyn_shared_with_diana_l1531_153149

theorem carolyn_shared_with_diana (initial final shared : ℕ) 
    (h_initial : initial = 47) 
    (h_final : final = 5)
    (h_shared : shared = initial - final) : shared = 42 := by
  rw [h_initial, h_final] at h_shared
  exact h_shared

end carolyn_shared_with_diana_l1531_153149


namespace divisible_by_xyz_l1531_153148

/-- 
Prove that the expression K = (x+y+z)^5 - (-x+y+z)^5 - (x-y+z)^5 - (x+y-z)^5 
is divisible by each of x, y, z.
-/
theorem divisible_by_xyz (x y z : ℝ) :
  ∃ t : ℝ, (x + y + z)^5 - (-x + y + z)^5 - (x - y + z)^5 - (x + y - z)^5 = t * x * y * z :=
by
  -- Proof to be provided
  sorry

end divisible_by_xyz_l1531_153148


namespace cheaper_joint_work_l1531_153152

theorem cheaper_joint_work (r L P : ℝ) (hr_pos : 0 < r) (hL_pos : 0 < L) (hP_pos : 0 < P) : 
  (2 * P * L) / (3 * r) < (3 * P * L) / (4 * r) :=
by
  sorry

end cheaper_joint_work_l1531_153152


namespace volume_of_circumscribed_sphere_l1531_153157

theorem volume_of_circumscribed_sphere (vol_cube : ℝ) (h : vol_cube = 8) :
  ∃ (vol_sphere : ℝ), vol_sphere = 4 * Real.sqrt 3 * Real.pi := 
sorry

end volume_of_circumscribed_sphere_l1531_153157


namespace tan_double_alpha_l1531_153105

theorem tan_double_alpha (α : ℝ) (h : ∀ x : ℝ, (3 * Real.sin x + Real.cos x) ≤ (3 * Real.sin α + Real.cos α)) :
  Real.tan (2 * α) = -3 / 4 :=
sorry

end tan_double_alpha_l1531_153105
