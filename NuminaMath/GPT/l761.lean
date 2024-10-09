import Mathlib

namespace amanda_family_painting_theorem_l761_76176

theorem amanda_family_painting_theorem
  (rooms_with_4_walls : ℕ)
  (walls_per_room_with_4_walls : ℕ)
  (rooms_with_5_walls : ℕ)
  (walls_per_room_with_5_walls : ℕ)
  (walls_per_person : ℕ)
  (total_rooms : ℕ)
  (h1 : rooms_with_4_walls = 5)
  (h2 : walls_per_room_with_4_walls = 4)
  (h3 : rooms_with_5_walls = 4)
  (h4 : walls_per_room_with_5_walls = 5)
  (h5 : walls_per_person = 8)
  (h6 : total_rooms = 9)
  : rooms_with_4_walls * walls_per_room_with_4_walls +
    rooms_with_5_walls * walls_per_room_with_5_walls =
    5 * walls_per_person :=
by
  sorry

end amanda_family_painting_theorem_l761_76176


namespace sum_of_real_values_l761_76101

theorem sum_of_real_values (x : ℝ) (h : |3 * x + 1| = 3 * |x - 3|) : x = 4 / 3 := sorry

end sum_of_real_values_l761_76101


namespace wooden_toys_count_l761_76172

theorem wooden_toys_count :
  ∃ T : ℤ, 
    10 * 40 + 20 * T - (10 * 36 + 17 * T) = 64 ∧ T = 8 :=
by
  use 8
  sorry

end wooden_toys_count_l761_76172


namespace divisibility_condition_l761_76194

theorem divisibility_condition (a m n : ℕ) (ha : 0 < a) (hm : 0 < m) (hn : 0 < n) :
  (a^m + 1) ∣ ((a + 1)^n) ↔ (a = 1 ∧ 1 ≤ m ∧ 1 ≤ n) ∨ (a = 2 ∧ m = 3 ∧ 2 ≤ n) := 
by 
  sorry

end divisibility_condition_l761_76194


namespace ratio_of_spending_is_one_to_two_l761_76118

-- Definitions
def initial_amount : ℕ := 24
def doris_spent : ℕ := 6
def final_amount : ℕ := 15

-- Amount remaining after Doris spent
def remaining_after_doris : ℕ := initial_amount - doris_spent

-- Amount Martha spent
def martha_spent : ℕ := remaining_after_doris - final_amount

-- Ratio of the amounts spent
def ratio_martha_doris : ℕ × ℕ := (martha_spent, doris_spent)

-- Theorem to prove
theorem ratio_of_spending_is_one_to_two : ratio_martha_doris = (1, 2) :=
by
  -- Placeholder for the proof
  sorry

end ratio_of_spending_is_one_to_two_l761_76118


namespace curve_not_parabola_l761_76131

theorem curve_not_parabola (k : ℝ) : ¬ ∃ (x y : ℝ), (x^2 + k * y^2 = 1) ↔ (k = -y / x) :=
by
  sorry

end curve_not_parabola_l761_76131


namespace unique_prime_value_l761_76185

theorem unique_prime_value :
  ∃! n : ℕ, n > 0 ∧ Nat.Prime (n^3 - 7 * n^2 + 17 * n - 11) :=
by {
  sorry
}

end unique_prime_value_l761_76185


namespace primes_in_sequence_are_12_l761_76104

-- Definition of Q
def Q : Nat := (2 * 3 * 5 * 7 * 11 * 13 * 17 * 19 * 23 * 29 * 31 * 37 * 41 * 43 * 47)

-- Set of m values
def ms : List Nat := List.range' 3 101

-- Function to check if Q + m is prime
def is_prime_minus_Q (m : Nat) : Bool := Nat.Prime (Q + m)

-- Counting primes in the sequence
def count_primes_in_sequence : Nat := (ms.filter (λ m => is_prime_minus_Q m = true)).length

theorem primes_in_sequence_are_12 :
  count_primes_in_sequence = 12 := by 
  sorry

end primes_in_sequence_are_12_l761_76104


namespace metal_beams_per_panel_l761_76148

theorem metal_beams_per_panel (panels sheets_per_panel rods_per_sheet rods_needed beams_per_panel rods_per_beam : ℕ)
    (h1 : panels = 10)
    (h2 : sheets_per_panel = 3)
    (h3 : rods_per_sheet = 10)
    (h4 : rods_needed = 380)
    (h5 : rods_per_beam = 4)
    (h6 : beams_per_panel = 2) :
    (panels * sheets_per_panel * rods_per_sheet + panels * beams_per_panel * rods_per_beam = rods_needed) :=
by
  sorry

end metal_beams_per_panel_l761_76148


namespace oak_trees_problem_l761_76127

theorem oak_trees_problem (c t n : ℕ) 
  (h1 : c = 9) 
  (h2 : t = 11) 
  (h3 : t = c + n) 
  : n = 2 := 
by 
  sorry

end oak_trees_problem_l761_76127


namespace line_through_A_parallel_line_through_B_perpendicular_l761_76136

-- 1. Prove the equation of the line passing through point A(2, 1) and parallel to the line 2x + y - 10 = 0 is 2x + y - 5 = 0.
theorem line_through_A_parallel :
  ∃ (l : ℝ → ℝ), (∀ x, 2 * x + l x - 5 = 0) ∧ (l 2 = 1) ∧ (∃ k, ∀ x, l x = -2 * (x - 2) + k) :=
sorry

-- 2. Prove the equation of the line passing through point B(3, 2) and perpendicular to the line 4x + 5y - 8 = 0 is 5x - 4y - 7 = 0.
theorem line_through_B_perpendicular :
  ∃ (m : ℝ) (l : ℝ → ℝ), (∀ x, 5 * x - 4 * l x - 7 = 0) ∧ (l 3 = 2) ∧ (m = -7) :=
sorry

end line_through_A_parallel_line_through_B_perpendicular_l761_76136


namespace sum_of_five_primes_is_145_l761_76130

-- Condition: common difference is 12
def common_difference : ℕ := 12

-- Five prime numbers forming an arithmetic sequence with the given common difference
def a1 : ℕ := 5
def a2 : ℕ := a1 + common_difference
def a3 : ℕ := a2 + common_difference
def a4 : ℕ := a3 + common_difference
def a5 : ℕ := a4 + common_difference

-- The sum of the arithmetic sequence
def sum_of_primes : ℕ := a1 + a2 + a3 + a4 + a5

-- Prove that the sum of these five prime numbers is 145
theorem sum_of_five_primes_is_145 : sum_of_primes = 145 :=
by
  -- Proof goes here
  sorry

end sum_of_five_primes_is_145_l761_76130


namespace distance_to_nearest_river_l761_76188

theorem distance_to_nearest_river (d : ℝ) (h₁ : ¬ (d ≤ 12)) (h₂ : ¬ (d ≥ 15)) (h₃ : ¬ (d ≥ 10)) :
  12 < d ∧ d < 15 :=
by 
  sorry

end distance_to_nearest_river_l761_76188


namespace compare_cube_roots_l761_76165

noncomputable def cube_root (x : ℝ) : ℝ := x^(1/3)

theorem compare_cube_roots : 2 + cube_root 7 < cube_root 60 :=
sorry

end compare_cube_roots_l761_76165


namespace basketball_points_total_l761_76184

variable (Tobee_points Jay_points Sean_points Remy_points Alex_points : ℕ)

def conditions := 
  Tobee_points = 4 ∧
  Jay_points = 2 * Tobee_points + 6 ∧
  Sean_points = Jay_points / 2 ∧
  Remy_points = Tobee_points + Jay_points - 3 ∧
  Alex_points = Sean_points + Remy_points + 4

theorem basketball_points_total 
  (h : conditions Tobee_points Jay_points Sean_points Remy_points Alex_points) :
  Tobee_points + Jay_points + Sean_points + Remy_points + Alex_points = 66 :=
by sorry

end basketball_points_total_l761_76184


namespace units_digit_of_sum_sequence_is_8_l761_76100

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def units_digit_sum_sequence : ℕ :=
  let term (n : ℕ) := (factorial n + n * n) % 10
  (term 1 + term 2 + term 3 + term 4 + term 5 + term 6 + term 7 + term 8 + term 9) % 10

theorem units_digit_of_sum_sequence_is_8 :
  units_digit_sum_sequence = 8 :=
sorry

end units_digit_of_sum_sequence_is_8_l761_76100


namespace trig_expression_l761_76121

theorem trig_expression (α : ℝ) (h : Real.tan α = 2) :
  (2 * Real.sin α + Real.cos α) / (Real.sin α - Real.cos α) = 5 := by
  sorry

end trig_expression_l761_76121


namespace proj_onto_w_equals_correct_l761_76181

open Real

noncomputable def proj (w v : ℝ × ℝ) : ℝ × ℝ :=
  let dot (a b : ℝ × ℝ) := a.1 * b.1 + a.2 * b.2
  let scalar_mul c (a : ℝ × ℝ) := (c * a.1, c * a.2)
  let w_dot_w := dot w w
  if w_dot_w = 0 then (0, 0) else scalar_mul (dot v w / w_dot_w) w

theorem proj_onto_w_equals_correct (v w : ℝ × ℝ)
  (hv : v = (2, 3))
  (hw : w = (-4, 1)) :
  proj w v = (20 / 17, -5 / 17) :=
by
  -- The proof would go here. We add sorry to skip it.
  sorry

end proj_onto_w_equals_correct_l761_76181


namespace f_of_x_l761_76123

theorem f_of_x (f : ℤ → ℤ) (h : ∀ x : ℤ, f (x-1) = 3*x - 1) : ∀ x : ℤ, f x = 3*x + 2 :=
by
  sorry

end f_of_x_l761_76123


namespace speed_limit_correct_l761_76126

def speed_limit_statement (v : ℝ) : Prop :=
  v ≤ 70

theorem speed_limit_correct (v : ℝ) (h : v ≤ 70) : speed_limit_statement v :=
by
  exact h

#print axioms speed_limit_correct

end speed_limit_correct_l761_76126


namespace jills_uncles_medicine_last_time_l761_76193

theorem jills_uncles_medicine_last_time :
  let pills := 90
  let third_of_pill_days := 3
  let days_per_full_pill := 9
  let days_per_month := 30
  let total_days := pills * days_per_full_pill
  let total_months := total_days / days_per_month
  total_months = 27 :=
by {
  sorry
}

end jills_uncles_medicine_last_time_l761_76193


namespace satellite_modular_units_l761_76177

variable (U N S T : ℕ)

def condition1 : Prop := N = (1/8 : ℝ) * S
def condition2 : Prop := T = 4 * S
def condition3 : Prop := U * N = 3 * S

theorem satellite_modular_units
  (h1 : condition1 N S)
  (h2 : condition2 T S)
  (h3 : condition3 U N S) :
  U = 24 :=
sorry

end satellite_modular_units_l761_76177


namespace combines_like_terms_l761_76175

theorem combines_like_terms (a : ℝ) : 2 * a - 5 * a = -3 * a := 
by sorry

end combines_like_terms_l761_76175


namespace square_area_divided_into_equal_rectangles_l761_76174

theorem square_area_divided_into_equal_rectangles (w : ℝ) (a : ℝ) (h : 5 = w) :
  (∃ s : ℝ, s * s = a ∧ s * s / 5 = a / 5) ↔ a = 400 :=
by
  sorry

end square_area_divided_into_equal_rectangles_l761_76174


namespace smallest_n_modulo_l761_76155

theorem smallest_n_modulo :
  ∃ (n : ℕ), 0 < n ∧ 1031 * n % 30 = 1067 * n % 30 ∧ ∀ (m : ℕ), 0 < m ∧ 1031 * m % 30 = 1067 * m % 30 → n ≤ m :=
by
  sorry

end smallest_n_modulo_l761_76155


namespace factorize_expression_l761_76178

-- Variables x and y are real numbers
variables (x y : ℝ)

-- Theorem statement
theorem factorize_expression : 3 * x^2 - 12 * y^2 = 3 * (x - 2 * y) * (x + 2 * y) :=
sorry

end factorize_expression_l761_76178


namespace find_y_l761_76102

def binary_op (a b c d : Int) : Int × Int := (a + d, b - c)

theorem find_y : ∃ y : Int, (binary_op 3 y 2 0) = (3, 4) ↔ y = 6 := by
  sorry

end find_y_l761_76102


namespace total_sales_l761_76182

-- Define sales of Robyn and Lucy
def Robyn_sales : Nat := 47
def Lucy_sales : Nat := 29

-- Prove total sales
theorem total_sales : Robyn_sales + Lucy_sales = 76 :=
by
  sorry

end total_sales_l761_76182


namespace average_length_correct_l761_76163

-- Given lengths of the two pieces
def length1 : ℕ := 2
def length2 : ℕ := 6

-- Define the average length
def average_length (l1 l2 : ℕ) : ℕ := (l1 + l2) / 2

-- State the theorem to prove
theorem average_length_correct : average_length length1 length2 = 4 := 
by 
  sorry

end average_length_correct_l761_76163


namespace arithmetic_mean_is_correct_l761_76140

-- Define the numbers
def num1 : ℕ := 18
def num2 : ℕ := 27
def num3 : ℕ := 45

-- Define the number of terms
def n : ℕ := 3

-- Define the sum of the numbers
def total_sum : ℕ := num1 + num2 + num3

-- Define the arithmetic mean
def arithmetic_mean : ℕ := total_sum / n

-- Theorem stating that the arithmetic mean of the numbers is 30
theorem arithmetic_mean_is_correct : arithmetic_mean = 30 := by
  -- Proof goes here
  sorry

end arithmetic_mean_is_correct_l761_76140


namespace cost_per_lb_of_mixture_l761_76132

def millet_weight : ℝ := 100
def millet_cost_per_lb : ℝ := 0.60
def sunflower_weight : ℝ := 25
def sunflower_cost_per_lb : ℝ := 1.10

theorem cost_per_lb_of_mixture :
  let millet_weight := 100
  let millet_cost_per_lb := 0.60
  let sunflower_weight := 25
  let sunflower_cost_per_lb := 1.10
  let millet_total_cost := millet_weight * millet_cost_per_lb
  let sunflower_total_cost := sunflower_weight * sunflower_cost_per_lb
  let total_cost := millet_total_cost + sunflower_total_cost
  let total_weight := millet_weight + sunflower_weight
  (total_cost / total_weight) = 0.70 :=
by
  sorry

end cost_per_lb_of_mixture_l761_76132


namespace total_truck_loads_needed_l761_76107

noncomputable def truck_loads_of_material : ℝ :=
  let sand := 0.16666666666666666 * Real.pi
  let dirt := 0.3333333333333333 * Real.exp 1
  let cement := 0.16666666666666666 * Real.sqrt 2
  let gravel := 0.25 * Real.log 5 -- log is the natural logarithm in Lean
  sand + dirt + cement + gravel

theorem total_truck_loads_needed : truck_loads_of_material = 1.8401374808985008 := by
  sorry

end total_truck_loads_needed_l761_76107


namespace count_perfect_squares_mul_36_l761_76167

theorem count_perfect_squares_mul_36 (n : ℕ) (h1 : n < 10^7) (h2 : ∃k, n = k^2) (h3 : 36 ∣ n) :
  ∃ m : ℕ, m = 263 :=
by
  sorry

end count_perfect_squares_mul_36_l761_76167


namespace systematic_sampling_questionnaire_B_count_l761_76168

theorem systematic_sampling_questionnaire_B_count (n : ℕ) (N : ℕ) (first_random : ℕ) (range_A_start range_A_end range_B_start range_B_end : ℕ) 
  (h1 : n = 32) (h2 : N = 960) (h3 : first_random = 9) (h4 : range_A_start = 1) (h5 : range_A_end = 460) 
  (h6 : range_B_start = 461) (h7 : range_B_end = 761) :
  ∃ count : ℕ, count = 10 := by
  sorry

end systematic_sampling_questionnaire_B_count_l761_76168


namespace absentees_in_morning_session_is_three_l761_76189

theorem absentees_in_morning_session_is_three
  (registered_morning : ℕ)
  (registered_afternoon : ℕ)
  (absent_afternoon : ℕ)
  (total_students : ℕ)
  (total_registered : ℕ)
  (attended_afternoon : ℕ)
  (attended_morning : ℕ)
  (absent_morning : ℕ) :
  registered_morning = 25 →
  registered_afternoon = 24 →
  absent_afternoon = 4 →
  total_students = 42 →
  total_registered = registered_morning + registered_afternoon →
  attended_afternoon = registered_afternoon - absent_afternoon →
  attended_morning = total_students - attended_afternoon →
  absent_morning = registered_morning - attended_morning →
  absent_morning = 3 :=
by
  intros
  sorry

end absentees_in_morning_session_is_three_l761_76189


namespace train_speed_l761_76150

theorem train_speed
  (train_length : ℕ)
  (man_speed_kmph : ℕ)
  (time_to_pass : ℕ)
  (speed_of_train : ℝ) :
  train_length = 180 →
  man_speed_kmph = 8 →
  time_to_pass = 4 →
  speed_of_train = 154 := 
by
  sorry

end train_speed_l761_76150


namespace exponent_division_l761_76144

theorem exponent_division (h1 : 27 = 3^3) : 3^18 / 27^3 = 19683 := by
  sorry

end exponent_division_l761_76144


namespace number_of_girls_l761_76125

variable (g b : ℕ) -- Number of girls (g) and boys (b) in the class
variable (h_ratio : g / b = 4 / 3) -- The ratio condition
variable (h_total : g + b = 63) -- The total number of students condition

theorem number_of_girls (g b : ℕ) (h_ratio : g / b = 4 / 3) (h_total : g + b = 63) :
    g = 36 :=
sorry

end number_of_girls_l761_76125


namespace Juanita_weekday_spending_l761_76139

/- Defining the variables and conditions in the problem -/

def Grant_spending : ℝ := 200
def Sunday_spending : ℝ := 2
def extra_spending : ℝ := 60

-- We need to prove that Juanita spends $0.50 per day from Monday through Saturday on newspapers.

theorem Juanita_weekday_spending :
  (∃ x : ℝ, 6 * 52 * x + 52 * 2 = Grant_spending + extra_spending) -> (∃ x : ℝ, x = 0.5) := by {
  sorry
}

end Juanita_weekday_spending_l761_76139


namespace triangle_inradius_exradii_relation_l761_76199

theorem triangle_inradius_exradii_relation
  (a b c : ℝ) (S : ℝ) (r r_a r_b r_c : ℝ)
  (h_inradius : S = (1/2) * r * (a + b + c))
  (h_exradii_a : r_a = 2 * S / (b + c - a))
  (h_exradii_b : r_b = 2 * S / (c + a - b))
  (h_exradii_c : r_c = 2 * S / (a + b - c))
  (h_area : S = (1/2) * (a * r_a + b * r_b + c * r_c - a * r - b * r - c * r)) :
  1 / r = 1 / r_a + 1 / r_b + 1 / r_c := 
  by sorry

end triangle_inradius_exradii_relation_l761_76199


namespace infinitely_many_a_not_sum_of_seven_sixth_powers_l761_76122

theorem infinitely_many_a_not_sum_of_seven_sixth_powers :
  ∃ᶠ (a: ℕ) in at_top, (∀ (a_i : ℕ) (h0 : a_i > 0), a ≠ a_i^6 + a_i^6 + a_i^6 + a_i^6 + a_i^6 + a_i^6 + a_i^6 ∧ a % 9 = 8) :=
sorry

end infinitely_many_a_not_sum_of_seven_sixth_powers_l761_76122


namespace rectangle_area_change_l761_76195

theorem rectangle_area_change 
  (L B : ℝ) 
  (A : ℝ := L * B) 
  (L' : ℝ := 1.30 * L) 
  (B' : ℝ := 0.75 * B) 
  (A' : ℝ := L' * B') : 
  A' / A = 0.975 := 
by sorry

end rectangle_area_change_l761_76195


namespace quadratic_properties_l761_76154

def quadratic_function (x : ℝ) : ℝ := -2 * x^2 + 4 * x + 3

theorem quadratic_properties :
  -- 1. The parabola opens downwards.
  (∀ x : ℝ, quadratic_function x < quadratic_function (x + 1) → false) ∧
  -- 2. The axis of symmetry is x = 1.
  (∀ x : ℝ, ∃ y : ℝ, quadratic_function x = quadratic_function y → x = y ∨ x + y = 2) ∧
  -- 3. The vertex coordinates are (1, 5).
  (quadratic_function 1 = 5) ∧
  -- 4. y decreases for x > 1.
  (∀ x : ℝ, x > 1 → quadratic_function x < quadratic_function (x - 1)) :=
by
  sorry

end quadratic_properties_l761_76154


namespace positive_number_sum_square_eq_210_l761_76170

theorem positive_number_sum_square_eq_210 (x : ℕ) (h1 : x^2 + x = 210) (h2 : 0 < x) (h3 : x < 15) : x = 14 :=
by
  sorry

end positive_number_sum_square_eq_210_l761_76170


namespace value_of_difference_power_l761_76112

theorem value_of_difference_power (a b : ℝ) (h₁ : a^3 - 6 * a^2 + 15 * a = 9) 
                                  (h₂ : b^3 - 3 * b^2 + 6 * b = -1) 
                                  : (a - b)^2014 = 1 := 
by sorry

end value_of_difference_power_l761_76112


namespace total_students_l761_76146

theorem total_students (T : ℕ) (h1 : (1/5 : ℚ) * T + (1/4 : ℚ) * T + (1/2 : ℚ) * T + 20 = T) : 
  T = 400 :=
sorry

end total_students_l761_76146


namespace factorize_expression_l761_76158

variables (a b x : ℝ)

theorem factorize_expression :
    5 * a * (x^2 - 1) - 5 * b * (x^2 - 1) = 5 * (x + 1) * (x - 1) * (a - b) := 
by
  sorry

end factorize_expression_l761_76158


namespace original_numbers_l761_76106

theorem original_numbers (a b c d : ℝ) (h1 : a + b + c + d = 45)
    (h2 : ∃ x : ℝ, a + 2 = x ∧ b - 2 = x ∧ 2 * c = x ∧ d / 2 = x) : 
    a = 8 ∧ b = 12 ∧ c = 5 ∧ d = 20 :=
by
  sorry

end original_numbers_l761_76106


namespace wanda_walks_days_per_week_l761_76156

theorem wanda_walks_days_per_week 
  (daily_distance : ℝ) (weekly_distance : ℝ) (weeks : ℕ) (total_distance : ℝ) 
  (h_daily_walk: daily_distance = 2) 
  (h_total_walk: total_distance = 40) 
  (h_weeks: weeks = 4) : 
  ∃ d : ℕ, (d * daily_distance * weeks = total_distance) ∧ (d = 5) := 
by 
  sorry

end wanda_walks_days_per_week_l761_76156


namespace leak_empties_in_24_hours_l761_76152

noncomputable def tap_rate := 1 / 6
noncomputable def combined_rate := 1 / 8
noncomputable def leak_rate := tap_rate - combined_rate
noncomputable def time_to_empty := 1 / leak_rate

theorem leak_empties_in_24_hours :
  time_to_empty = 24 := by
  sorry

end leak_empties_in_24_hours_l761_76152


namespace seqAN_81_eq_640_l761_76197

-- Definitions and hypotheses
def seqAN (n : ℕ) : ℝ := sorry   -- A sequence a_n to be defined properly.

def sumSN (n : ℕ) : ℝ := sorry  -- The sum of the first n terms of a_n.

axiom condition_positivity : ∀ n : ℕ, 0 < seqAN n
axiom condition_a1 : seqAN 1 = 1
axiom condition_sum (n : ℕ) (h : 2 ≤ n) : 
  sumSN n * Real.sqrt (sumSN (n-1)) - sumSN (n-1) * Real.sqrt (sumSN n) = 
  2 * Real.sqrt (sumSN n * sumSN (n-1))

-- Proof problem: 
theorem seqAN_81_eq_640 : seqAN 81 = 640 := by sorry

end seqAN_81_eq_640_l761_76197


namespace remainder_8927_div_11_l761_76124

theorem remainder_8927_div_11 : 8927 % 11 = 8 :=
by
  sorry

end remainder_8927_div_11_l761_76124


namespace smallest_sum_of_consecutive_integers_is_square_l761_76108

-- Define the sum of consecutive integers
def sum_of_consecutive_integers (n : ℕ) : ℕ :=
  (20 * n) + (190 : ℕ)

-- We need to prove there exists an n such that the sum is a perfect square
theorem smallest_sum_of_consecutive_integers_is_square :
  ∃ n : ℕ, ∃ k : ℕ, sum_of_consecutive_integers n = k * k ∧ k * k = 250 :=
sorry

end smallest_sum_of_consecutive_integers_is_square_l761_76108


namespace no_triangles_if_all_horizontal_removed_l761_76128

/-- 
Given a figure that consists of 40 identical toothpicks, making up a symmetric figure with 
additional rows on the top and bottom. We need to prove that removing all 40 horizontal toothpicks 
ensures there are no remaining triangles in the figure.
-/
theorem no_triangles_if_all_horizontal_removed
  (initial_toothpicks : ℕ)
  (horizontal_toothpicks_in_figure : ℕ) 
  (rows : ℕ)
  (top_row : ℕ)
  (second_row : ℕ)
  (third_row : ℕ)
  (fourth_row : ℕ)
  (bottom_row : ℕ)
  (additional_rows : ℕ)
  (triangles_for_upward : ℕ)
  (triangles_for_downward : ℕ):
  initial_toothpicks = 40 →
  horizontal_toothpicks_in_figure = top_row + second_row + third_row + fourth_row + bottom_row →
  rows = 5 →
  top_row = 5 →
  second_row = 10 →
  third_row = 10 →
  fourth_row = 10 →
  bottom_row = 5 →
  additional_rows = 2 →
  triangles_for_upward = 15 →
  triangles_for_downward = 10 →
  horizontal_toothpicks_in_figure = 40 → 
  ∀ toothpicks_removed, toothpicks_removed = 40 →
  no_triangles_remain :=
by
  intros
  sorry

end no_triangles_if_all_horizontal_removed_l761_76128


namespace smaller_odd_number_l761_76161

theorem smaller_odd_number (n : ℤ) (h : n + (n + 2) = 48) : n = 23 :=
by
  sorry

end smaller_odd_number_l761_76161


namespace pentagon_area_correct_l761_76138

-- Define the side lengths of the pentagon
def side1 := 18
def side2 := 25
def side3 := 30
def side4 := 28
def side5 := 22

-- Define the specific angle between the sides of lengths 30 and 28
def angle := 110 -- degrees

-- Define the heights used for the trapezoids and triangle calculations
def height_trapezoid1 := 10
def height_trapezoid2 := 15
def height_triangle := 8

-- Function to calculate the area of a trapezoid
def trapezoid_area (base1 base2 height : ℕ) : ℕ :=
  (base1 + base2) * height / 2

-- Function to calculate the area of a triangle
def triangle_area (base height : ℕ) : ℕ :=
  base * height / 2

-- Calculation of individual areas
def area_trapezoid1 := trapezoid_area side1 side2 height_trapezoid1
def area_trapezoid2 := trapezoid_area side3 side4 height_trapezoid2
def area_triangle := triangle_area side5 height_triangle

-- Total area calculation
def total_area := area_trapezoid1 + area_trapezoid2 + area_triangle

-- Expected total area
def expected_area := 738

-- Lean statement to assert the total area equals the expected value
theorem pentagon_area_correct :
  total_area = expected_area :=
by sorry

end pentagon_area_correct_l761_76138


namespace relationship_among_abc_l761_76157

theorem relationship_among_abc 
  (f : ℝ → ℝ)
  (h_symm : ∀ x, f (x) = f (-x))
  (h_def : ∀ x, 0 < x → f x = |Real.log x / Real.log 2|)
  (a : ℝ) (b : ℝ) (c : ℝ)
  (ha : a = f (1 / 3))
  (hb : b = f (-4))
  (hc : c = f 2) :
  c < a ∧ a < b :=
by
  sorry

end relationship_among_abc_l761_76157


namespace prime_diff_of_cubes_sum_of_square_and_three_times_square_l761_76116

theorem prime_diff_of_cubes_sum_of_square_and_three_times_square 
  (p : ℕ) (a b : ℕ) (h_prime : Nat.Prime p) (h_diff : p = a^3 - b^3) :
  ∃ c d : ℤ, p = c^2 + 3 * d^2 := 
  sorry

end prime_diff_of_cubes_sum_of_square_and_three_times_square_l761_76116


namespace directrix_of_parabola_l761_76151

theorem directrix_of_parabola (x y : ℝ) : (x ^ 2 = y) → (4 * y + 1 = 0) :=
sorry

end directrix_of_parabola_l761_76151


namespace set_intersection_l761_76153

def S : Set ℝ := {x | x^2 - 5 * x + 6 ≥ 0}
def T : Set ℝ := {x | x > 1}
def result : Set ℝ := {x | x ≥ 3 ∨ (1 < x ∧ x ≤ 2)}

theorem set_intersection (x : ℝ) : x ∈ (S ∩ T) ↔ x ∈ result := by
  sorry

end set_intersection_l761_76153


namespace proof_of_value_of_6y_plus_3_l761_76114

theorem proof_of_value_of_6y_plus_3 (y : ℤ) (h : 3 * y + 2 = 11) : 6 * y + 3 = 21 :=
by
  sorry

end proof_of_value_of_6y_plus_3_l761_76114


namespace qs_length_l761_76169

theorem qs_length
  (PQR : Triangle)
  (PQ QR PR : ℝ)
  (h1 : PQ = 7)
  (h2 : QR = 8)
  (h3 : PR = 9)
  (bugs_meet_half_perimeter : PQ + QR + PR = 24)
  (bugs_meet_distance : PQ + qs = 12) :
  qs = 5 :=
by
  sorry

end qs_length_l761_76169


namespace equation_of_l_symmetric_point_l761_76119

/-- Define points O, A, B in the coordinate plane --/
def O := (0, 0)
def A := (2, 0)
def B := (3, 2)

/-- Define midpoint of OA --/
def midpoint_OA := ((O.1 + A.1) / 2, (O.2 + A.2) / 2)

/-- Line l passes through midpoint_OA and B. Prove line l has equation y = x - 1 --/
theorem equation_of_l :
  ∃ l : ℝ → ℝ, (∀ x, l x = x - 1) :=
sorry

/-- Prove the symmetric point of A with respect to line l is (1, 1) --/
theorem symmetric_point :
  ∃ (a b : ℝ), (a, b) = (1, 1) ∧
                (b * (2 - 1)) / (a - 2) = -1 ∧
                b / 2 = (2 + a - 1) / 2 - 1 :=
sorry

end equation_of_l_symmetric_point_l761_76119


namespace max_expr_value_l761_76110

noncomputable def expr (a b c d : ℝ) : ℝ :=
  a + 2 * b + c + 2 * d - a * b - b * c - c * d - d * a

theorem max_expr_value : 
  ∃ (a b c d : ℝ),
    a ∈ Set.Icc (-5 : ℝ) 5 ∧
    b ∈ Set.Icc (-5 : ℝ) 5 ∧
    c ∈ Set.Icc (-5 : ℝ) 5 ∧
    d ∈ Set.Icc (-5 : ℝ) 5 ∧
    expr a b c d = 110 :=
by
  -- Proof omitted
  sorry

end max_expr_value_l761_76110


namespace cost_of_song_book_l761_76187

-- Define the costs as constants
def cost_trumpet : ℝ := 149.16
def cost_music_tool : ℝ := 9.98
def total_spent : ℝ := 163.28

-- Define the statement to prove
theorem cost_of_song_book : total_spent - (cost_trumpet + cost_music_tool) = 4.14 := 
by
  sorry

end cost_of_song_book_l761_76187


namespace binom_mult_l761_76173

open Nat

theorem binom_mult : (Nat.choose 10 3) * (Nat.choose 8 3) = 6720 := by
  sorry

end binom_mult_l761_76173


namespace arrangement_count_l761_76141

noncomputable def count_arrangements (balls : Finset ℕ) (boxes : Finset ℕ) : ℕ :=
  sorry -- The implementation of this function is out of scope for this task

theorem arrangement_count :
  count_arrangements ({1, 2, 3, 4} : Finset ℕ) ({1, 2, 3} : Finset ℕ) = 18 :=
sorry

end arrangement_count_l761_76141


namespace arrange_numbers_l761_76179

noncomputable def a := (10^100)^10
noncomputable def b := 10^(10^10)
noncomputable def c := Nat.factorial 1000000
noncomputable def d := (Nat.factorial 100)^10

theorem arrange_numbers :
  a < d ∧ d < c ∧ c < b := 
sorry

end arrange_numbers_l761_76179


namespace arithmetic_sequence_third_term_l761_76145

theorem arithmetic_sequence_third_term :
  ∀ (a d : ℤ), (a + 4 * d = 2) ∧ (a + 5 * d = 5) → (a + 2 * d = -4) :=
by sorry

end arithmetic_sequence_third_term_l761_76145


namespace smallest_n_for_pencil_purchase_l761_76143

theorem smallest_n_for_pencil_purchase (a b c d n : ℕ)
  (h1 : 6 * a + 10 * b = n)
  (h2 : 6 * c + 10 * d = n + 2)
  (h3 : 7 * a + 12 * b > 7 * c + 12 * d)
  (h4 : 3 * (c - a) + 5 * (d - b) = 1)
  (h5 : d - b > 0) :
  n = 100 :=
by
  sorry

end smallest_n_for_pencil_purchase_l761_76143


namespace problem_l761_76105

-- Define the functions f and g with their properties
noncomputable def f : ℝ → ℝ := sorry
noncomputable def g : ℝ → ℝ := sorry

-- Express the given conditions in Lean
axiom f_even : ∀ x : ℝ, f (-x) = f x
axiom g_odd : ∀ x : ℝ, g (-x) = -g x
axiom g_def : ∀ x : ℝ, g x = f (x - 1)
axiom f_at_2 : f 2 = 2

-- What we need to prove
theorem problem : f 2014 = 2 := 
by sorry

end problem_l761_76105


namespace mr_ray_customers_without_fish_l761_76166

def mr_ray_num_customers_without_fish
  (total_customers : ℕ)
  (total_tuna_weight : ℕ)
  (specific_customers_30lb : ℕ)
  (specific_weight_30lb : ℕ)
  (specific_customers_20lb : ℕ)
  (specific_weight_20lb : ℕ)
  (weight_per_customer : ℕ)
  (remaining_tuna_weight : ℕ)
  (num_customers_served_with_remaining_tuna : ℕ)
  (total_satisfied_customers : ℕ) : ℕ :=
  total_customers - total_satisfied_customers

theorem mr_ray_customers_without_fish :
  mr_ray_num_customers_without_fish 100 2000 10 30 15 20 25 1400 56 81 = 19 :=
by 
  sorry

end mr_ray_customers_without_fish_l761_76166


namespace points_after_perfect_games_l761_76186

theorem points_after_perfect_games (perfect_score : ℕ) (num_games : ℕ) (total_points : ℕ) 
  (h1 : perfect_score = 21) 
  (h2 : num_games = 3) 
  (h3 : total_points = perfect_score * num_games) : 
  total_points = 63 :=
by 
  sorry

end points_after_perfect_games_l761_76186


namespace integer_triplets_satisfy_eq_l761_76198

theorem integer_triplets_satisfy_eq {x y z : ℤ} : 
  x^2 + y^2 + z^2 - x * y - y * z - z * x = 3 ↔ 
  (∃ k : ℤ, (x = k + 2 ∧ y = k + 1 ∧ z = k) ∨ (x = k - 2 ∧ y = k - 1 ∧ z = k)) := 
by
  sorry

end integer_triplets_satisfy_eq_l761_76198


namespace students_number_l761_76111

theorem students_number (x a o : ℕ)
  (h1 : o = 3 * a + 3)
  (h2 : a = 2 * x + 6)
  (h3 : o = 7 * x - 5) :
  x = 26 :=
by sorry

end students_number_l761_76111


namespace circle_equation_exists_l761_76120

theorem circle_equation_exists :
  ∃ (x_c y_c r : ℝ), 
  x_c > 0 ∧ y_c > 0 ∧ 0 < r ∧ r < 5 ∧ (∀ x y : ℝ, (x - x_c)^2 + (y - y_c)^2 = r^2) :=
sorry

end circle_equation_exists_l761_76120


namespace volume_of_region_l761_76164

noncomputable def volume_of_sphere (r : ℝ) : ℝ :=
  (4/3) * Real.pi * r^3

theorem volume_of_region (r_small r_large : ℝ) (h_small : r_small = 4) (h_large : r_large = 7) :
  volume_of_sphere r_large - volume_of_sphere r_small = 372 * Real.pi := by
  rw [h_small, h_large]
  sorry

end volume_of_region_l761_76164


namespace marius_scored_3_more_than_darius_l761_76191

theorem marius_scored_3_more_than_darius 
  (D M T : ℕ) 
  (h1 : D = 10) 
  (h2 : T = D + 5) 
  (h3 : M + D + T = 38) : 
  M = D + 3 := 
by
  sorry

end marius_scored_3_more_than_darius_l761_76191


namespace question1_question2_l761_76137

section

variable (A B C : Set ℝ)
variable (a : ℝ)

-- Condition 1: A = {x | -1 ≤ x < 3}
def setA : Set ℝ := {x | -1 ≤ x ∧ x < 3}

-- Condition 2: B = {x | 2x - 4 ≥ x - 2}
def setB : Set ℝ := {x | x ≥ 2}

-- Condition 3: C = {x | x ≥ a - 1}
def setC (a : ℝ) : Set ℝ := {x | x ≥ a - 1}

-- Question 1: Prove A ∩ B = {x | 2 ≤ x < 3}
theorem question1 : A = setA → B = setB → A ∩ B = {x | 2 ≤ x ∧ x < 3} :=
by intros hA hB; rw [hA, hB]; sorry

-- Question 2: If B ∪ C = C, prove a ∈ (-∞, 3]
theorem question2 : B = setB → C = setC a → (B ∪ C = C) → a ≤ 3 :=
by intros hB hC hBUC; rw [hB, hC] at hBUC; sorry

end

end question1_question2_l761_76137


namespace calculate_retail_price_l761_76133

/-- Define the wholesale price of the machine. -/
def wholesale_price : ℝ := 90

/-- Define the profit rate as 20% of the wholesale price. -/
def profit_rate : ℝ := 0.20

/-- Define the discount rate as 10% of the retail price. -/
def discount_rate : ℝ := 0.10

/-- Calculate the profit based on the wholesale price. -/
def profit : ℝ := profit_rate * wholesale_price

/-- Calculate the selling price after the discount. -/
def selling_price (retail_price : ℝ) : ℝ := retail_price * (1 - discount_rate)

/-- Calculate the total selling price as the wholesale price plus profit. -/
def total_selling_price : ℝ := wholesale_price + profit

/-- State the theorem we need to prove. -/
theorem calculate_retail_price : ∃ R : ℝ, selling_price R = total_selling_price → R = 120 := by
  sorry

end calculate_retail_price_l761_76133


namespace total_balloons_are_48_l761_76160

theorem total_balloons_are_48 
  (brooke_initial : ℕ) (brooke_add : ℕ) (tracy_initial : ℕ) (tracy_add : ℕ)
  (brooke_half_given : ℕ) (tracy_third_popped : ℕ) : 
  brooke_initial = 20 →
  brooke_add = 15 →
  tracy_initial = 10 →
  tracy_add = 35 →
  brooke_half_given = (brooke_initial + brooke_add) / 2 →
  tracy_third_popped = (tracy_initial + tracy_add) / 3 →
  (brooke_initial + brooke_add - brooke_half_given) + (tracy_initial + tracy_add - tracy_third_popped) = 48 := 
by
  intros
  sorry

end total_balloons_are_48_l761_76160


namespace puzzle_pieces_l761_76135

theorem puzzle_pieces
  (total_puzzles : ℕ)
  (pieces_per_10_min : ℕ)
  (total_minutes : ℕ)
  (h1 : total_puzzles = 2)
  (h2 : pieces_per_10_min = 100)
  (h3 : total_minutes = 400) :
  ((total_minutes / 10) * pieces_per_10_min) / total_puzzles = 2000 :=
by
  sorry

end puzzle_pieces_l761_76135


namespace min_sum_of_factors_l761_76115

theorem min_sum_of_factors (a b c : ℕ) (h₁ : a > 0) (h₂ : b > 0) (h₃ : c > 0) (h₄ : a * b * c = 3432) :
  a + b + c ≥ 56 :=
sorry

end min_sum_of_factors_l761_76115


namespace move_line_down_l761_76171

theorem move_line_down (x : ℝ) : (y = -x + 1) → (y = -x - 2) := by
  sorry

end move_line_down_l761_76171


namespace least_possible_number_l761_76103

theorem least_possible_number (k : ℕ) (n : ℕ) (r : ℕ) (h1 : k = 34 * n + r) 
  (h2 : k / 5 = r + 8) (h3 : r < 34) : k = 68 :=
by
  -- Proof to be filled
  sorry

end least_possible_number_l761_76103


namespace max_value_of_k_l761_76142

theorem max_value_of_k (n : ℕ) (k : ℕ) (h : 3^11 = k * (2 * n + k + 1) / 2) : k = 486 :=
sorry

end max_value_of_k_l761_76142


namespace smallest_solution_abs_eq_20_l761_76190

theorem smallest_solution_abs_eq_20 : ∃ x : ℝ, x = -7 ∧ |4 * x + 8| = 20 ∧ (∀ y : ℝ, |4 * y + 8| = 20 → x ≤ y) :=
by
  sorry

end smallest_solution_abs_eq_20_l761_76190


namespace book_original_selling_price_l761_76147

theorem book_original_selling_price (CP SP1 SP2 : ℝ) 
  (h1 : SP1 = 0.9 * CP)
  (h2 : SP2 = 1.1 * CP)
  (h3 : SP2 = 990) : 
  SP1 = 810 :=
by
  sorry

end book_original_selling_price_l761_76147


namespace exp_decreasing_function_range_l761_76196

theorem exp_decreasing_function_range (a : ℝ) (x : ℝ) (h_a : 0 < a ∧ a < 1) (h_f : a^(x+1) ≥ 1) : x ≤ -1 :=
sorry

end exp_decreasing_function_range_l761_76196


namespace fourth_group_trees_l761_76117

theorem fourth_group_trees (x : ℕ) :
  5 * 13 = 12 + 15 + 12 + x + 11 → x = 15 :=
by
  sorry

end fourth_group_trees_l761_76117


namespace total_amount_l761_76109

theorem total_amount (A B C T : ℝ)
  (h1 : A = 1 / 4 * (B + C))
  (h2 : B = 3 / 5 * (A + C))
  (h3 : A = 20) :
  T = A + B + C → T = 100 := by
  sorry

end total_amount_l761_76109


namespace no_nonzero_real_solutions_l761_76162

theorem no_nonzero_real_solutions (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) : 
  ¬ (2 / x + 3 / y = 1 / (x + y)) :=
by sorry

end no_nonzero_real_solutions_l761_76162


namespace solution_exists_unique_l761_76159

theorem solution_exists_unique (x y : ℝ) : (x + y = 2 ∧ x - y = 0) ↔ (x = 1 ∧ y = 1) := 
by
  sorry

end solution_exists_unique_l761_76159


namespace at_least_one_divisible_by_5_l761_76129

theorem at_least_one_divisible_by_5 (k m n : ℕ) (hk : ¬ (5 ∣ k)) (hm : ¬ (5 ∣ m)) (hn : ¬ (5 ∣ n)) : 
  (5 ∣ (k^2 - m^2)) ∨ (5 ∣ (m^2 - n^2)) ∨ (5 ∣ (n^2 - k^2)) :=
by {
    sorry
}

end at_least_one_divisible_by_5_l761_76129


namespace min_val_z_is_7_l761_76149

noncomputable def min_val_z (x y : ℝ) (h : x + 3 * y = 2) : ℝ := 3^x + 27^y + 1

theorem min_val_z_is_7  : ∃ x y : ℝ, x + 3 * y = 2 ∧ min_val_z x y (by sorry) = 7 := sorry

end min_val_z_is_7_l761_76149


namespace totalCostOfCombinedSubscriptions_l761_76192

-- Define the given conditions
def packageACostPerMonth : ℝ := 10
def packageAMonths : ℝ := 6
def packageADiscount : ℝ := 0.10

def packageBCostPerMonth : ℝ := 12
def packageBMonths : ℝ := 9
def packageBDiscount : ℝ := 0.15

-- Define the total cost after discounts
def packageACostAfterDiscount : ℝ := packageACostPerMonth * packageAMonths * (1 - packageADiscount)
def packageBCostAfterDiscount : ℝ := packageBCostPerMonth * packageBMonths * (1 - packageBDiscount)

-- Statement to be proved
theorem totalCostOfCombinedSubscriptions :
  packageACostAfterDiscount + packageBCostAfterDiscount = 145.80 := by
  sorry

end totalCostOfCombinedSubscriptions_l761_76192


namespace range_of_a_l761_76180

theorem range_of_a (a : ℝ) : (∀ x, 1 ≤ x ∧ x ≤ 3 → x^2 - a * x - 3 ≤ 0) ↔ (2 ≤ a) := by
  sorry

end range_of_a_l761_76180


namespace max_possible_N_l761_76134

-- Defining the conditions
def team_size : ℕ := 15

def total_games : ℕ := team_size * team_size

-- Given conditions imply N ways to schedule exactly one game
def ways_to_schedule_one_game (remaining_games : ℕ) : ℕ := remaining_games - 1

-- Maximum possible value of N given the constraints
theorem max_possible_N : ways_to_schedule_one_game (total_games - team_size * (team_size - 1) / 2) = 120 := 
by sorry

end max_possible_N_l761_76134


namespace average_apples_per_hour_l761_76183

theorem average_apples_per_hour (A H : ℝ) (hA : A = 12) (hH : H = 5) : A / H = 2.4 := by
  -- sorry skips the proof
  sorry

end average_apples_per_hour_l761_76183


namespace zoo_ticket_sales_l761_76113

-- Define the number of total people, number of adults, and ticket prices
def total_people : ℕ := 254
def num_adults : ℕ := 51
def adult_ticket_price : ℕ := 28
def kid_ticket_price : ℕ := 12

-- Define the number of kids as the difference between total people and number of adults
def num_kids : ℕ := total_people - num_adults

-- Define the revenue from adult tickets and kid tickets
def revenue_adult_tickets : ℕ := num_adults * adult_ticket_price
def revenue_kid_tickets : ℕ := num_kids * kid_ticket_price

-- Define the total revenue
def total_revenue : ℕ := revenue_adult_tickets + revenue_kid_tickets

-- Theorem to prove the total revenue equals 3864
theorem zoo_ticket_sales : total_revenue = 3864 :=
  by {
    -- sorry allows us to skip the proof
    sorry
  }

end zoo_ticket_sales_l761_76113
