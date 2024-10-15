import Mathlib

namespace NUMINAMATH_GPT_combined_mass_of_individuals_l2301_230129

-- Define constants and assumptions
def boat_length : ℝ := 4 -- in meters
def boat_breadth : ℝ := 3 -- in meters
def sink_depth_first_person : ℝ := 0.01 -- in meters (1 cm)
def sink_depth_second_person : ℝ := 0.02 -- in meters (2 cm)
def density_water : ℝ := 1000 -- in kg/m³ (density of freshwater)

-- Define volumes displaced
def volume_displaced_first : ℝ := boat_length * boat_breadth * sink_depth_first_person
def volume_displaced_both : ℝ := boat_length * boat_breadth * (sink_depth_first_person + sink_depth_second_person)

-- Define weights (which are equal to the masses under the assumption of constant gravity)
def weight_first_person : ℝ := volume_displaced_first * density_water
def weight_both_persons : ℝ := volume_displaced_both * density_water

-- Statement to prove the combined weight
theorem combined_mass_of_individuals : weight_both_persons = 360 :=
by
  -- Skip the proof
  sorry

end NUMINAMATH_GPT_combined_mass_of_individuals_l2301_230129


namespace NUMINAMATH_GPT_cost_of_bench_l2301_230123

variables (cost_table cost_bench : ℕ)

theorem cost_of_bench :
  cost_table + cost_bench = 450 ∧ cost_table = 2 * cost_bench → cost_bench = 150 :=
by
  sorry

end NUMINAMATH_GPT_cost_of_bench_l2301_230123


namespace NUMINAMATH_GPT_degenerate_ellipse_single_point_l2301_230155

theorem degenerate_ellipse_single_point (c : ℝ) :
  (∀ x y : ℝ, 3 * x^2 + y^2 + 6 * x - 12 * y + c = 0 → (x = -1 ∧ y = 6)) ↔ c = -39 :=
by
  sorry

end NUMINAMATH_GPT_degenerate_ellipse_single_point_l2301_230155


namespace NUMINAMATH_GPT_possible_even_and_odd_functions_l2301_230116

def is_even_function (f : ℝ → ℝ) := ∀ x, f (-x) = f x
def is_odd_function (f : ℝ → ℝ) := ∀ x, f (-x) = -f x

theorem possible_even_and_odd_functions :
  ∃ p q : ℝ → ℝ, is_even_function p ∧ is_odd_function (p ∘ q) ∧ (¬(∀ x, p (q x) = 0)) :=
by
  sorry

end NUMINAMATH_GPT_possible_even_and_odd_functions_l2301_230116


namespace NUMINAMATH_GPT_max_value_of_k_l2301_230131

noncomputable def max_possible_k (x y : ℝ) (k : ℝ) : Prop :=
  0 < x ∧ 0 < y ∧ 0 < k ∧
  (3 = k^2 * (x^2 / y^2 + y^2 / x^2) + k * (x / y + y / x))

theorem max_value_of_k (x y : ℝ) (k : ℝ) :
  max_possible_k x y k → k ≤ (-1 + Real.sqrt 7) / 2 :=
sorry

end NUMINAMATH_GPT_max_value_of_k_l2301_230131


namespace NUMINAMATH_GPT_find_bc_l2301_230180

noncomputable def setA : Set ℝ := {x | x^2 + x - 2 ≤ 0}
noncomputable def setB : Set ℝ := {x | 2 < x + 1 ∧ x + 1 ≤ 4}
noncomputable def setAB : Set ℝ := setA ∪ setB
noncomputable def setC (b c : ℝ) : Set ℝ := {x | x^2 + b * x + c > 0}

theorem find_bc (b c : ℝ) :
  (setAB ∩ setC b c = ∅) ∧ (setAB ∪ setC b c = Set.univ) →
  b = -1 ∧ c = -6 :=
by
  sorry

end NUMINAMATH_GPT_find_bc_l2301_230180


namespace NUMINAMATH_GPT_total_earnings_l2301_230169

theorem total_earnings (x y : ℕ) 
  (h1 : 2 * x * y = 250) : 
  58 * (x * y) = 7250 := 
by
  sorry

end NUMINAMATH_GPT_total_earnings_l2301_230169


namespace NUMINAMATH_GPT_daughters_and_granddaughters_without_daughters_l2301_230181

-- Given conditions
def melissa_daughters : ℕ := 10
def half_daughters_with_children : ℕ := melissa_daughters / 2
def grandchildren_per_daughter : ℕ := 4
def total_descendants : ℕ := 50

-- Calculations based on given conditions
def number_of_granddaughters : ℕ := total_descendants - melissa_daughters
def daughters_with_no_children : ℕ := melissa_daughters - half_daughters_with_children
def granddaughters_with_no_children : ℕ := number_of_granddaughters

-- The final result we need to prove
theorem daughters_and_granddaughters_without_daughters : 
  daughters_with_no_children + granddaughters_with_no_children = 45 := by
  sorry

end NUMINAMATH_GPT_daughters_and_granddaughters_without_daughters_l2301_230181


namespace NUMINAMATH_GPT_find_p_l2301_230137

variable (a b c p : ℚ)

theorem find_p (h1 : 5 / (a + b) = p / (a + c)) (h2 : p / (a + c) = 8 / (c - b)) : p = 13 := by
  sorry

end NUMINAMATH_GPT_find_p_l2301_230137


namespace NUMINAMATH_GPT_Sarah_shampoo_conditioner_usage_l2301_230157

theorem Sarah_shampoo_conditioner_usage (daily_shampoo : ℝ) (daily_conditioner : ℝ) (days_in_week : ℝ) (weeks : ℝ) (total_days : ℝ) (daily_total : ℝ) (total_usage : ℝ) :
  daily_shampoo = 1 → 
  daily_conditioner = daily_shampoo / 2 → 
  days_in_week = 7 → 
  weeks = 2 → 
  total_days = days_in_week * weeks → 
  daily_total = daily_shampoo + daily_conditioner → 
  total_usage = daily_total * total_days → 
  total_usage = 21 := by
  sorry

end NUMINAMATH_GPT_Sarah_shampoo_conditioner_usage_l2301_230157


namespace NUMINAMATH_GPT_variance_of_scores_l2301_230138

-- Define the student's scores
def scores : List ℕ := [130, 125, 126, 126, 128]

-- Define a function to calculate the mean
def mean (l : List ℕ) : ℕ :=
  l.sum / l.length

-- Define a function to calculate the variance
def variance (l : List ℕ) : ℕ :=
  let avg := mean l
  (l.map (λ x => (x - avg) * (x - avg))).sum / l.length

-- The proof statement (no proof provided, use sorry)
theorem variance_of_scores : variance scores = 3 := by sorry

end NUMINAMATH_GPT_variance_of_scores_l2301_230138


namespace NUMINAMATH_GPT_sqrt_meaningful_condition_l2301_230187

theorem sqrt_meaningful_condition (a : ℝ) : 2 - a ≥ 0 → a ≤ 2 := by
  sorry

end NUMINAMATH_GPT_sqrt_meaningful_condition_l2301_230187


namespace NUMINAMATH_GPT_smallest_sum_of_squares_l2301_230135

theorem smallest_sum_of_squares (a b : ℕ) (h : a - b = 221) : a + b = 229 :=
sorry

end NUMINAMATH_GPT_smallest_sum_of_squares_l2301_230135


namespace NUMINAMATH_GPT_last_year_ticket_cost_l2301_230199

theorem last_year_ticket_cost (this_year_cost : ℝ) (increase_percentage : ℝ) (last_year_cost : ℝ) :
  this_year_cost = last_year_cost * (1 + increase_percentage) ↔ last_year_cost = 85 :=
by
  let this_year_cost := 102
  let increase_percentage := 0.20
  sorry

end NUMINAMATH_GPT_last_year_ticket_cost_l2301_230199


namespace NUMINAMATH_GPT_paul_packed_total_toys_l2301_230172

def small_box_small_toys : ℕ := 8
def medium_box_medium_toys : ℕ := 12
def large_box_large_toys : ℕ := 7
def large_box_small_toys : ℕ := 3
def small_box_medium_toys : ℕ := 5

def small_box : ℕ := small_box_small_toys + small_box_medium_toys
def medium_box : ℕ := medium_box_medium_toys
def large_box : ℕ := large_box_large_toys + large_box_small_toys

def total_toys : ℕ := small_box + medium_box + large_box

theorem paul_packed_total_toys : total_toys = 35 :=
by sorry

end NUMINAMATH_GPT_paul_packed_total_toys_l2301_230172


namespace NUMINAMATH_GPT_find_a_if_f_is_even_l2301_230143

noncomputable def f (x a : ℝ) : ℝ := (x + a) * (x - 2)

theorem find_a_if_f_is_even (a : ℝ) (h : ∀ x : ℝ, f x a = f (-x) a) : a = 2 := by
  sorry

end NUMINAMATH_GPT_find_a_if_f_is_even_l2301_230143


namespace NUMINAMATH_GPT_find_a_l2301_230195

theorem find_a (a : ℝ) (h : ∃ x : ℝ, x = 2 ∧ x^2 + a * x - 2 = 0) : a = -1 := 
by 
  sorry

end NUMINAMATH_GPT_find_a_l2301_230195


namespace NUMINAMATH_GPT_price_of_second_set_of_knives_l2301_230171

def john_visits_houses_per_day : ℕ := 50
def percent_buying_per_day : ℝ := 0.20
def price_first_set : ℝ := 50
def weekly_sales : ℝ := 5000
def work_days_per_week : ℕ := 5

theorem price_of_second_set_of_knives
  (john_visits_houses_per_day : ℕ)
  (percent_buying_per_day : ℝ)
  (price_first_set : ℝ)
  (weekly_sales : ℝ)
  (work_days_per_week : ℕ) :
  0 < percent_buying_per_day ∧ percent_buying_per_day ≤ 1 ∧
  weekly_sales = 5000 ∧ 
  work_days_per_week = 5 ∧
  john_visits_houses_per_day = 50 ∧
  price_first_set = 50 → 
  (∃ price_second_set : ℝ, price_second_set = 150) :=
  sorry

end NUMINAMATH_GPT_price_of_second_set_of_knives_l2301_230171


namespace NUMINAMATH_GPT_ticket_sales_revenue_l2301_230197

theorem ticket_sales_revenue (total_tickets advance_tickets same_day_tickets price_advance price_same_day: ℕ) 
    (h1: total_tickets = 60) 
    (h2: price_advance = 20) 
    (h3: price_same_day = 30) 
    (h4: advance_tickets = 20) 
    (h5: same_day_tickets = total_tickets - advance_tickets):
    advance_tickets * price_advance + same_day_tickets * price_same_day = 1600 := 
by
  sorry

end NUMINAMATH_GPT_ticket_sales_revenue_l2301_230197


namespace NUMINAMATH_GPT_binary_to_decimal_l2301_230156

/-- The binary number 1011 (base 2) equals 11 (base 10). -/
theorem binary_to_decimal : (1 * 2^3 + 0 * 2^2 + 1 * 2^1 + 1 * 2^0) = 11 := by
  sorry

end NUMINAMATH_GPT_binary_to_decimal_l2301_230156


namespace NUMINAMATH_GPT_identity_proof_l2301_230178

theorem identity_proof (a b c : ℝ) (hab : a ≠ b) (hac : a ≠ c) (hbc : b ≠ c) :
    (b - c) / ((a - b) * (a - c)) + (c - a) / ((b - c) * (b - a)) + (a - b) / ((c - a) * (c - b)) =
    2 / (a - b) + 2 / (b - c) + 2 / (c - a) :=
by
  sorry

end NUMINAMATH_GPT_identity_proof_l2301_230178


namespace NUMINAMATH_GPT_find_F_l2301_230126

theorem find_F (C F : ℝ) (h1 : C = (4 / 7) * (F - 40)) (h2 : C = 35) : F = 101.25 :=
  sorry

end NUMINAMATH_GPT_find_F_l2301_230126


namespace NUMINAMATH_GPT_distance_of_points_in_polar_coordinates_l2301_230151

theorem distance_of_points_in_polar_coordinates
  (A : Real × Real) (B : Real × Real) (θ1 θ2 : Real)
  (hA : A = (5, θ1)) (hB : B = (12, θ2))
  (hθ : θ1 - θ2 = Real.pi / 2) : 
  dist (5 * Real.cos θ1, 5 * Real.sin θ1) (12 * Real.cos θ2, 12 * Real.sin θ2) = 13 := 
by sorry

end NUMINAMATH_GPT_distance_of_points_in_polar_coordinates_l2301_230151


namespace NUMINAMATH_GPT_complex_numbers_right_triangle_l2301_230113

theorem complex_numbers_right_triangle (z : ℂ) (hz : z ≠ 0) :
  (∃ z₁ z₂ : ℂ, z₁ ≠ 0 ∧ z₂ ≠ 0 ∧ z₁^3 = z₂ ∧
                 (∃ θ₁ θ₂ : ℝ, z₁ = Complex.exp (Complex.I * θ₁) ∧
                               z₂ = Complex.exp (Complex.I * θ₂) ∧
                               (θ₂ - θ₁ = π/2 ∨ θ₂ - θ₁ = 3 * π/2))) →
  ∃ n : ℕ, n = 2 :=
by
  sorry

end NUMINAMATH_GPT_complex_numbers_right_triangle_l2301_230113


namespace NUMINAMATH_GPT_partI_partII_l2301_230119

-- Define the absolute value function
def f (x : ℝ) := |x - 1|

-- Part I: Solve the inequality f(x) - f(x+2) < 1
theorem partI (x : ℝ) (h : f x - f (x + 2) < 1) : x > -1 / 2 := 
sorry

-- Part II: Find the range of values for a such that x - f(x + 1 - a) ≤ 1 for all x in [1,2]
theorem partII (a : ℝ) (h : ∀ x : ℝ, 1 ≤ x ∧ x ≤ 2 → x - f (x + 1 - a) ≤ 1) : a ≤ 1 ∨ a ≥ 3 := 
sorry

end NUMINAMATH_GPT_partI_partII_l2301_230119


namespace NUMINAMATH_GPT_minimum_apples_l2301_230148

theorem minimum_apples (n : ℕ) : 
  n % 4 = 1 ∧ n % 5 = 2 ∧ n % 9 = 7 → n = 97 := 
by 
  -- To be proved
  sorry

end NUMINAMATH_GPT_minimum_apples_l2301_230148


namespace NUMINAMATH_GPT_calculate_expression_l2301_230173

theorem calculate_expression :
  (5 * 7 + 10 * 4 - 35 / 5 + 18 / 3 : ℝ) = 74 := by
  sorry

end NUMINAMATH_GPT_calculate_expression_l2301_230173


namespace NUMINAMATH_GPT_cylindrical_to_rectangular_l2301_230160

theorem cylindrical_to_rectangular (r θ z : ℝ) 
  (h₁ : r = 7) (h₂ : θ = 5 * Real.pi / 4) (h₃ : z = 6) : 
  (r * Real.cos θ, r * Real.sin θ, z) = 
  (-7 * Real.sqrt 2 / 2, -7 * Real.sqrt 2 / 2, 6) := 
by 
  sorry

end NUMINAMATH_GPT_cylindrical_to_rectangular_l2301_230160


namespace NUMINAMATH_GPT_f_always_positive_l2301_230194

noncomputable def f (x : ℝ) : ℝ := x^8 - x^5 + x^2 - x + 1

theorem f_always_positive : ∀ x : ℝ, 0 < f x := by
  sorry

end NUMINAMATH_GPT_f_always_positive_l2301_230194


namespace NUMINAMATH_GPT_infinitely_many_n_prime_l2301_230121

theorem infinitely_many_n_prime (p : ℕ) [Fact (Nat.Prime p)] : ∃ᶠ n in at_top, p ∣ 2^n - n := 
sorry

end NUMINAMATH_GPT_infinitely_many_n_prime_l2301_230121


namespace NUMINAMATH_GPT_garden_roller_area_l2301_230153

theorem garden_roller_area (D : ℝ) (A : ℝ) (π : ℝ) (L_new : ℝ) :
  D = 1.4 → A = 88 → π = 22/7 → L_new = 4 → A = 5 * (2 * π * (D / 2) * L_new) :=
by sorry

end NUMINAMATH_GPT_garden_roller_area_l2301_230153


namespace NUMINAMATH_GPT_minimize_J_l2301_230144

noncomputable def H (p q : ℝ) : ℝ :=
  -3 * p * q + 4 * p * (1 - q) + 4 * (1 - p) * q - 5 * (1 - p) * (1 - q)

noncomputable def J (p : ℝ) : ℝ :=
  if p < 0 then 0 else if p > 1 then 1 else if (9 * p - 5 > 4 - 7 * p) then 9 * p - 5 else 4 - 7 * p

theorem minimize_J :
  ∃ (p : ℝ), 0 ≤ p ∧ p ≤ 1 ∧ J p = J (9 / 16) := by
  sorry

end NUMINAMATH_GPT_minimize_J_l2301_230144


namespace NUMINAMATH_GPT_swimming_pool_area_l2301_230146

open Nat

-- Define the width (w) and length (l) with given conditions
def width (w : ℕ) : Prop :=
  exists (l : ℕ), l = 2 * w + 40 ∧ 2 * w + 2 * l = 800

-- Define the area of the swimming pool
def pool_area (w l : ℕ) : ℕ :=
  w * l

theorem swimming_pool_area : 
  ∃ (w l : ℕ), width w ∧ width l -> pool_area w l = 33600 :=
by
  sorry

end NUMINAMATH_GPT_swimming_pool_area_l2301_230146


namespace NUMINAMATH_GPT_sum_of_coordinates_of_point_D_l2301_230139

theorem sum_of_coordinates_of_point_D
  (N : ℝ × ℝ := (6,2))
  (C : ℝ × ℝ := (10, -2))
  (h : ∃ D : ℝ × ℝ, (N = ((C.1 + D.1) / 2, (C.2 + D.2) / 2))) :
  ∃ (D : ℝ × ℝ), D.1 + D.2 = 8 := 
by
  obtain ⟨D, hD⟩ := h
  sorry

end NUMINAMATH_GPT_sum_of_coordinates_of_point_D_l2301_230139


namespace NUMINAMATH_GPT_pairball_playing_time_l2301_230166

-- Define the conditions of the problem
def num_children : ℕ := 7
def total_minutes : ℕ := 105
def total_child_minutes : ℕ := 2 * total_minutes

-- Define the theorem to prove
theorem pairball_playing_time : total_child_minutes / num_children = 30 :=
by sorry

end NUMINAMATH_GPT_pairball_playing_time_l2301_230166


namespace NUMINAMATH_GPT_age_product_difference_is_nine_l2301_230162

namespace ArnoldDanny

def current_age := 4
def product_today (A : ℕ) := A * A
def product_next_year (A : ℕ) := (A + 1) * (A + 1)
def difference (A : ℕ) := product_next_year A - product_today A

theorem age_product_difference_is_nine :
  difference current_age = 9 :=
by
  sorry

end ArnoldDanny

end NUMINAMATH_GPT_age_product_difference_is_nine_l2301_230162


namespace NUMINAMATH_GPT_number_not_equal_54_l2301_230142

def initial_number : ℕ := 12
def target_number : ℕ := 54
def total_time : ℕ := 60

theorem number_not_equal_54 (n : ℕ) (time : ℕ) : (time = total_time) → (n = initial_number) → 
  (∀ t : ℕ, t ≤ time → (n = n * 2 ∨ n = n / 2 ∨ n = n * 3 ∨ n = n / 3)) → n ≠ target_number :=
by
  sorry

end NUMINAMATH_GPT_number_not_equal_54_l2301_230142


namespace NUMINAMATH_GPT_f_7_eq_minus_1_l2301_230174

-- Define the odd function f with the given properties
def is_odd_function (f : ℝ → ℝ) :=
  ∀ x, f (-x) = -f x

def period_2 (f : ℝ → ℝ) :=
  ∀ x, f (x + 2) = -f x

def f_restricted (f : ℝ → ℝ) :=
  ∀ (x : ℝ), 0 ≤ x ∧ x ≤ 1 -> f x = x

-- The main statement: Under the given conditions, f(7) = -1
theorem f_7_eq_minus_1 (f : ℝ → ℝ)
  (H1 : is_odd_function f)
  (H2 : period_2 f)
  (H3 : f_restricted f) :
  f 7 = -1 :=
by
  sorry

end NUMINAMATH_GPT_f_7_eq_minus_1_l2301_230174


namespace NUMINAMATH_GPT_value_not_uniquely_determined_l2301_230120

variables (v : Fin 9 → ℤ) (s : Fin 9 → ℤ)

-- Given conditions
axiom announced_sums : ∀ i, s i = v ((i - 1) % 9) + v ((i + 1) % 9)
axiom sums_sequence : s 0 = 3 ∧ s 1 = 7 ∧ s 2 = 12 ∧ s 3 = 18 ∧ s 4 = 24 ∧ s 5 = 31 ∧ s 6 = 40 ∧ s 7 = 48 ∧ s 8 = 53

-- Statement asserting the indeterminacy of v_5
theorem value_not_uniquely_determined (h: s 3 = 18) : 
  ∃ v : Fin 9 → ℤ, sorry :=
sorry

end NUMINAMATH_GPT_value_not_uniquely_determined_l2301_230120


namespace NUMINAMATH_GPT_complex_number_in_first_quadrant_l2301_230198

noncomputable def z : ℂ := Complex.ofReal 1 + Complex.I

theorem complex_number_in_first_quadrant 
  (h : Complex.ofReal 1 + Complex.I = Complex.I / z) : 
  (0 < z.re ∧ 0 < z.im) :=
  sorry

end NUMINAMATH_GPT_complex_number_in_first_quadrant_l2301_230198


namespace NUMINAMATH_GPT_calculate_120_percent_l2301_230196

theorem calculate_120_percent (x : ℝ) (h : 0.20 * x = 100) : 1.20 * x = 600 :=
sorry

end NUMINAMATH_GPT_calculate_120_percent_l2301_230196


namespace NUMINAMATH_GPT_Bo_knew_percentage_l2301_230150

-- Definitions from the conditions
def total_flashcards := 800
def words_per_day := 16
def days := 40
def total_words_to_learn := words_per_day * days
def known_words := total_flashcards - total_words_to_learn

-- Statement that we need to prove
theorem Bo_knew_percentage : (known_words.toFloat / total_flashcards.toFloat) * 100 = 20 :=
by
  sorry  -- Proof is omitted as per the instructions

end NUMINAMATH_GPT_Bo_knew_percentage_l2301_230150


namespace NUMINAMATH_GPT_arithmetic_mean_solution_l2301_230193

-- Define the Arithmetic Mean statement
theorem arithmetic_mean_solution (x : ℝ) (h : (x + 5 + 17 + 3 * x + 11 + 3 * x + 6) / 5 = 19) : 
  x = 8 :=
by
  sorry -- Proof is not required as per the instructions

end NUMINAMATH_GPT_arithmetic_mean_solution_l2301_230193


namespace NUMINAMATH_GPT_kira_night_songs_l2301_230136

-- Definitions for the conditions
def morning_songs : ℕ := 10
def later_songs : ℕ := 15
def song_size_mb : ℕ := 5
def total_new_songs_memory_mb : ℕ := 140

-- Assert the number of songs Kira downloaded at night
theorem kira_night_songs : (total_new_songs_memory_mb - (morning_songs * song_size_mb + later_songs * song_size_mb)) / song_size_mb = 3 :=
by
  sorry

end NUMINAMATH_GPT_kira_night_songs_l2301_230136


namespace NUMINAMATH_GPT_product_of_four_consecutive_even_numbers_divisible_by_240_l2301_230102

theorem product_of_four_consecutive_even_numbers_divisible_by_240 :
  ∀ (n : ℤ), (n % 2 = 0) →
    (n + 2) % 2 = 0 →
    (n + 4) % 2 = 0 →
    (n + 6) % 2 = 0 →
    ((n * (n + 2) * (n + 4) * (n + 6)) % 240 = 0) :=
by
  intro n hn hnp2 hnp4 hnp6
  sorry

end NUMINAMATH_GPT_product_of_four_consecutive_even_numbers_divisible_by_240_l2301_230102


namespace NUMINAMATH_GPT_exists_root_in_interval_l2301_230168

noncomputable def f (x : ℝ) : ℝ := Real.log x + 2 * x - 5

theorem exists_root_in_interval : ∃ x, (2 < x ∧ x < 3) ∧ f x = 0 := 
by
  -- Assuming f(2) < 0 and f(3) > 0
  have h1 : f 2 < 0 := sorry
  have h2 : f 3 > 0 := sorry
  -- From the intermediate value theorem, there exists a c in (2, 3) such that f(c) = 0
  sorry

end NUMINAMATH_GPT_exists_root_in_interval_l2301_230168


namespace NUMINAMATH_GPT_smallest_N_satisfying_frequencies_l2301_230191

def percentageA := 1 / 5
def percentageB := 3 / 8
def percentageC := 1 / 4
def percentageD := 1 / 8
def percentageE := 1 / 20

def Divisible (n : ℕ) (d : ℕ) : Prop := ∃ (k : ℕ), n = k * d

theorem smallest_N_satisfying_frequencies :
  ∃ N : ℕ, 
    Divisible N 5 ∧ 
    Divisible N 8 ∧ 
    Divisible N 4 ∧ 
    Divisible N 20 ∧ 
    N = 40 := sorry

end NUMINAMATH_GPT_smallest_N_satisfying_frequencies_l2301_230191


namespace NUMINAMATH_GPT_cross_covers_two_rectangles_l2301_230159

def Chessboard := Fin 8 × Fin 8

def is_cross (center : Chessboard) (point : Chessboard) : Prop :=
  (point.1 = center.1 ∧ (point.2 = center.2 - 1 ∨ point.2 = center.2 + 1)) ∨
  (point.2 = center.2 ∧ (point.1 = center.1 - 1 ∨ point.1 = center.1 + 1)) ∨
  (point = center)

def Rectangle_1x3 (rect : Fin 22) : Chessboard → Prop := sorry -- This represents Alina's rectangles
def Rectangle_1x2 (rect : Fin 22) : Chessboard → Prop := sorry -- This represents Polina's rectangles

theorem cross_covers_two_rectangles :
  ∃ center : Chessboard, 
    (∃ (rect_a : Fin 22), ∃ (rect_b : Fin 22), rect_a ≠ rect_b ∧ 
      (∀ p, is_cross center p → Rectangle_1x3 rect_a p) ∧ 
      (∀ p, is_cross center p → Rectangle_1x2 rect_b p)) ∨
    (∃ (rect_a : Fin 22), ∃ (rect_b : Fin 22), rect_a ≠ rect_b ∧ 
      (∀ p, is_cross center p → Rectangle_1x3 rect_a p) ∧ 
      (∀ p, is_cross center p → Rectangle_1x3 rect_b p)) ∨
    (∃ (rect_a : Fin 22), ∃ (rect_b : Fin 22), rect_a ≠ rect_b ∧ 
      (∀ p, is_cross center p → Rectangle_1x2 rect_a p) ∧ 
      (∀ p, is_cross center p → Rectangle_1x2 rect_b p)) :=
sorry

end NUMINAMATH_GPT_cross_covers_two_rectangles_l2301_230159


namespace NUMINAMATH_GPT_exists_root_interval_l2301_230185

def f (x : ℝ) : ℝ := x^2 + 12 * x - 15

theorem exists_root_interval :
  (f 1.1 < 0) ∧ (f 1.2 > 0) → ∃ x : ℝ, 1.1 < x ∧ x < 1.2 ∧ f x = 0 := 
by
  intro h
  sorry

end NUMINAMATH_GPT_exists_root_interval_l2301_230185


namespace NUMINAMATH_GPT_smallest_special_number_gt_3429_l2301_230105

-- Define what it means for a number to be special
def is_special (n : ℕ) : Prop :=
  (List.toFinset (Nat.digits 10 n)).card = 4

-- Define the problem statement in Lean
theorem smallest_special_number_gt_3429 : ∃ n : ℕ, 3429 < n ∧ is_special n ∧ ∀ m : ℕ, 3429 < m ∧ is_special m → n ≤ m := 
  by
  let smallest_n := 3450
  have hn : 3429 < smallest_n := by decide
  have hs : is_special smallest_n := by
    -- digits of 3450 are [3, 4, 5, 0], which are four different digits
    sorry 
  have minimal : ∀ m, 3429 < m ∧ is_special m → smallest_n ≤ m :=
    by
    -- This needs to show that no special number exists between 3429 and 3450
    sorry
  exact ⟨smallest_n, hn, hs, minimal⟩

end NUMINAMATH_GPT_smallest_special_number_gt_3429_l2301_230105


namespace NUMINAMATH_GPT_imo1983_q24_l2301_230140

theorem imo1983_q24 :
  ∃ (S : Finset ℕ), S.card = 1983 ∧ 
    (∀ x ∈ S, x > 0 ∧ x ≤ 10^5) ∧
    (∀ (x y z : ℕ), x ∈ S → y ∈ S → z ∈ S → x ≠ y → x ≠ z → y ≠ z → (x + z ≠ 2 * y)) :=
sorry

end NUMINAMATH_GPT_imo1983_q24_l2301_230140


namespace NUMINAMATH_GPT_pyramid_base_sidelength_l2301_230179

theorem pyramid_base_sidelength (A : ℝ) (h : ℝ) (s : ℝ) 
  (hA : A = 120) (hh : h = 24) (area_eq : A = 1/2 * s * h) : s = 10 := by
  sorry

end NUMINAMATH_GPT_pyramid_base_sidelength_l2301_230179


namespace NUMINAMATH_GPT_projection_ratio_zero_l2301_230188

variables (v w u p q : ℝ → ℝ) -- Assuming vectors are functions from ℝ to ℝ
variables (norm : (ℝ → ℝ) → ℝ) -- norm is a function from vectors to ℝ
variables (proj : (ℝ → ℝ) → (ℝ → ℝ) → (ℝ → ℝ)) -- proj is the projection function

-- Assume the conditions
axiom proj_p : p = proj v w
axiom proj_q : q = proj p u
axiom perp_uv : ∀ t, v t * u t = 0 -- u is perpendicular to v
axiom norm_ratio : norm p / norm v = 3 / 8

theorem projection_ratio_zero : norm q / norm v = 0 :=
by sorry

end NUMINAMATH_GPT_projection_ratio_zero_l2301_230188


namespace NUMINAMATH_GPT_minimum_sum_PE_PC_l2301_230103

noncomputable def point := (ℝ × ℝ)
noncomputable def length (p1 p2 : point) : ℝ := Real.sqrt (((p1.1 - p2.1)^2) + ((p1.2 - p2.2)^2))

theorem minimum_sum_PE_PC :
  let A : point := (0, 3)
  let B : point := (3, 3)
  let C : point := (3, 0)
  let D : point := (0, 0)
  ∃ P E : point, E.1 = 3 ∧ E.2 = 1 ∧ (∃ t : ℝ, t ≥ 0 ∧ t ≤ 3 ∧ P.1 = 3 - t ∧ P.2 = t) ∧
    (length P E + length P C = Real.sqrt 13) :=
by
  sorry

end NUMINAMATH_GPT_minimum_sum_PE_PC_l2301_230103


namespace NUMINAMATH_GPT_correct_answer_l2301_230163

def vector := (Int × Int)

-- Definitions of vectors given in conditions
def m : vector := (2, 1)
def n : vector := (0, -2)

def vec_add (v1 v2 : vector) : vector :=
  (v1.1 + v2.1, v1.2 + v2.2)

def vec_scalar_mult (c : Int) (v : vector) : vector :=
  (c * v.1, c * v.2)

def vec_dot (v1 v2 : vector) : Int :=
  v1.1 * v2.1 + v1.2 * v2.2

-- The condition vector combined
def combined_vector := vec_add m (vec_scalar_mult 2 n)

-- The problem is to prove this:
theorem correct_answer : vec_dot (3, 2) combined_vector = 0 :=
  sorry

end NUMINAMATH_GPT_correct_answer_l2301_230163


namespace NUMINAMATH_GPT_sqrt_expression_simplify_l2301_230141

theorem sqrt_expression_simplify : 
  2 * Real.sqrt 12 * (Real.sqrt 3 / 4) / (10 * Real.sqrt 2) = 3 * Real.sqrt 2 / 20 :=
by 
  sorry

end NUMINAMATH_GPT_sqrt_expression_simplify_l2301_230141


namespace NUMINAMATH_GPT_find_linear_odd_increasing_function_l2301_230108

theorem find_linear_odd_increasing_function (f : ℝ → ℝ)
    (h1 : ∀ x, f (f x) = 4 * x)
    (h2 : ∀ x, f x = -f (-x))
    (h3 : ∀ x y, x < y → f x < f y)
    (h4 : ∃ a : ℝ, a ≠ 0 ∧ ∀ x, f x = a * x) : 
    ∀ x, f x = 2 * x :=
by
  sorry

end NUMINAMATH_GPT_find_linear_odd_increasing_function_l2301_230108


namespace NUMINAMATH_GPT_units_digit_product_l2301_230183

theorem units_digit_product :
  let nums : List Nat := [7, 17, 27, 37, 47, 57, 67, 77, 87, 97]
  let product := nums.prod
  (product % 10) = 9 :=
by
  sorry

end NUMINAMATH_GPT_units_digit_product_l2301_230183


namespace NUMINAMATH_GPT_coefficient_a9_l2301_230115

theorem coefficient_a9 (a0 a1 a2 a3 a4 a5 a6 a7 a8 a9 a10 : ℤ) :
  (x^2 + x^10 = a0 + a1 * (x + 1) + a2 * (x + 1)^2 + a3 * (x + 1)^3 +
   a4 * (x + 1)^4 + a5 * (x + 1)^5 + a6 * (x + 1)^6 + a7 * (x + 1)^7 +
   a8 * (x + 1)^8 + a9 * (x + 1)^9 + a10 * (x + 1)^10) →
  a10 = 1 →
  a9 = -10 :=
by
  sorry

end NUMINAMATH_GPT_coefficient_a9_l2301_230115


namespace NUMINAMATH_GPT_factor_expression_l2301_230190

variable (b : ℤ)

theorem factor_expression : 221 * b^2 + 17 * b = 17 * b * (13 * b + 1) :=
by
  sorry

end NUMINAMATH_GPT_factor_expression_l2301_230190


namespace NUMINAMATH_GPT_division_reciprocal_multiplication_l2301_230184

theorem division_reciprocal_multiplication : (4 / (8 / 13 : ℚ)) = (13 / 2 : ℚ) := 
by
  sorry

end NUMINAMATH_GPT_division_reciprocal_multiplication_l2301_230184


namespace NUMINAMATH_GPT_plane_equation_l2301_230158

theorem plane_equation :
  ∃ (A B C D : ℤ), A > 0 ∧ Int.gcd (Int.natAbs A) (Int.gcd (Int.natAbs B) (Int.gcd (Int.natAbs C) (Int.natAbs D))) = 1 ∧ 
  (∀ (x y z : ℤ), (x, y, z) = (0, 0, 0) ∨ (x, y, z) = (2, 0, -2) → A * x + B * y + C * z + D = 0) ∧ 
  ∀ (x y z : ℤ), (A = 1 ∧ B = -5 ∧ C = 1 ∧ D = 0) := sorry

end NUMINAMATH_GPT_plane_equation_l2301_230158


namespace NUMINAMATH_GPT_seminar_duration_total_l2301_230112

/-- The first part of the seminar lasted 4 hours and 45 minutes -/
def first_part_minutes := 4 * 60 + 45

/-- The second part of the seminar lasted 135 minutes -/
def second_part_minutes := 135

/-- The closing event lasted 500 seconds -/
def closing_event_minutes := 500 / 60

/-- The total duration of the seminar session in minutes, including the closing event, is 428 minutes -/
theorem seminar_duration_total :
  first_part_minutes + second_part_minutes + closing_event_minutes = 428 := by
  sorry

end NUMINAMATH_GPT_seminar_duration_total_l2301_230112


namespace NUMINAMATH_GPT_find_f_7_over_2_l2301_230192

section
variable {f : ℝ → ℝ}

-- Conditions
axiom odd_fn : ∀ x : ℝ, f (-x) = -f (x)
axiom even_shift_fn : ∀ x : ℝ, f (x + 1) = f (1 - x)
axiom range_x : Π x : ℝ, -1 ≤ x ∧ x ≤ 0 → f (x) = 2 * x^2

-- Prove that f(7/2) = 1/2
theorem find_f_7_over_2 : f (7 / 2) = 1 / 2 :=
sorry
end

end NUMINAMATH_GPT_find_f_7_over_2_l2301_230192


namespace NUMINAMATH_GPT_arithmetic_operation_equals_l2301_230134

theorem arithmetic_operation_equals :
  12.1212 + 17.0005 - 9.1103 = 20.0114 := 
by 
  sorry

end NUMINAMATH_GPT_arithmetic_operation_equals_l2301_230134


namespace NUMINAMATH_GPT_olivias_dad_total_spending_l2301_230110

def people : ℕ := 5
def meal_cost : ℕ := 12
def drink_cost : ℕ := 3
def dessert_cost : ℕ := 5

theorem olivias_dad_total_spending : 
  (people * meal_cost) + (people * drink_cost) + (people * dessert_cost) = 100 := 
by
  sorry

end NUMINAMATH_GPT_olivias_dad_total_spending_l2301_230110


namespace NUMINAMATH_GPT_claim1_claim2_l2301_230161

theorem claim1 (n : ℤ) (hs : ∃ l : List ℤ, l.length = n ∧ l.prod = n ∧ l.sum = 0) : 
  ∃ k : ℤ, n = 4 * k := 
sorry

theorem claim2 (n : ℕ) (h : n % 4 = 0) : 
  ∃ l : List ℤ, l.length = n ∧ l.prod = n ∧ l.sum = 0 := 
sorry

end NUMINAMATH_GPT_claim1_claim2_l2301_230161


namespace NUMINAMATH_GPT_shaded_area_l2301_230128

/--
Given a larger square containing a smaller square entirely within it,
where the side length of the smaller square is 5 units
and the side length of the larger square is 10 units,
prove that the area of the shaded region (the area of the larger square minus the area of the smaller square) is 75 square units.
-/
theorem shaded_area :
  let side_length_smaller := 5
  let side_length_larger := 10
  let area_larger := side_length_larger * side_length_larger
  let area_smaller := side_length_smaller * side_length_smaller
  area_larger - area_smaller = 75 := 
by
  let side_length_smaller := 5
  let side_length_larger := 10
  let area_larger := side_length_larger * side_length_larger
  let area_smaller := side_length_smaller * side_length_smaller
  sorry

end NUMINAMATH_GPT_shaded_area_l2301_230128


namespace NUMINAMATH_GPT_traveler_never_returns_home_l2301_230111

variable (City : Type)
variable (Distance : City → City → ℝ)

variables (A B C : City)
variables (C_i C_i_plus_one C_i_minus_one : City)

-- Given conditions
axiom travel_far_from_A : ∀ (C : City), C ≠ B → Distance A B > Distance A C
axiom travel_far_from_B : ∀ (D : City), D ≠ C → Distance B C > Distance B D
axiom increasing_distance : ∀ i : ℕ, Distance C_i C_i_plus_one > Distance C_i_minus_one C_i

-- Given condition that C is not A
axiom C_not_eq_A : C ≠ A

-- Proof statement
theorem traveler_never_returns_home : ∀ i : ℕ, C_i ≠ A := sorry

end NUMINAMATH_GPT_traveler_never_returns_home_l2301_230111


namespace NUMINAMATH_GPT_school_dance_attendance_l2301_230189

theorem school_dance_attendance (P : ℝ)
  (h1 : 0.1 * P = (P - (0.9 * P)))
  (h2 : 0.9 * P = (2/3) * (0.9 * P) + (1/3) * (0.9 * P))
  (h3 : 30 = (1/3) * (0.9 * P)) :
  P = 100 :=
by
  sorry

end NUMINAMATH_GPT_school_dance_attendance_l2301_230189


namespace NUMINAMATH_GPT_units_digit_k_squared_plus_two_exp_k_eq_7_l2301_230182

/-- Define k as given in the problem -/
def k : ℕ := 2010^2 + 2^2010

/-- Final statement that needs to be proved -/
theorem units_digit_k_squared_plus_two_exp_k_eq_7 : (k^2 + 2^k) % 10 = 7 := 
by
  sorry

end NUMINAMATH_GPT_units_digit_k_squared_plus_two_exp_k_eq_7_l2301_230182


namespace NUMINAMATH_GPT_hall_length_l2301_230176

theorem hall_length (L B A : ℝ) (h1 : B = 2 / 3 * L) (h2 : A = 2400) (h3 : A = L * B) : L = 60 := by
  -- proof steps here
  sorry

end NUMINAMATH_GPT_hall_length_l2301_230176


namespace NUMINAMATH_GPT_possible_values_for_N_l2301_230147

theorem possible_values_for_N (N : ℕ) (h₁ : 8 < N) (h₂ : 1 ≤ N - 1) :
  22 < N ∧ N ≤ 25 ↔ (N = 23 ∨ N = 24 ∨ N = 25) :=
by 
  sorry

end NUMINAMATH_GPT_possible_values_for_N_l2301_230147


namespace NUMINAMATH_GPT_increasing_quadratic_l2301_230109

noncomputable def f (a x : ℝ) : ℝ := 3 * x^2 - a * x + 4

theorem increasing_quadratic {a : ℝ} :
  (∀ x ≥ -5, 6 * x - a ≥ 0) ↔ a ≤ -30 :=
by
  sorry

end NUMINAMATH_GPT_increasing_quadratic_l2301_230109


namespace NUMINAMATH_GPT_divisible_by_5_l2301_230118

theorem divisible_by_5 (n : ℕ) : (∃ k : ℕ, 2^n - 1 = 5 * k) ∨ (∃ k : ℕ, 2^n + 1 = 5 * k) ∨ (∃ k : ℕ, 2^(2*n) + 1 = 5 * k) :=
sorry

end NUMINAMATH_GPT_divisible_by_5_l2301_230118


namespace NUMINAMATH_GPT_costco_container_holds_one_gallon_l2301_230152

theorem costco_container_holds_one_gallon
  (costco_cost : ℕ := 8)
  (store_cost_per_bottle : ℕ := 3)
  (savings : ℕ := 16)
  (ounces_per_bottle : ℕ := 16)
  (ounces_per_gallon : ℕ := 128) :
  ∃ (gallons : ℕ), gallons = 1 :=
by
  sorry

end NUMINAMATH_GPT_costco_container_holds_one_gallon_l2301_230152


namespace NUMINAMATH_GPT_gcd_of_repeated_three_digit_number_is_constant_l2301_230114

theorem gcd_of_repeated_three_digit_number_is_constant (m : ℕ) (h1 : 100 ≤ m) (h2 : m < 1000) : 
  ∃ d, d = 1001001 ∧ ∀ n, n = 10010013 * m → (gcd 1001001 n) = 1001001 :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_GPT_gcd_of_repeated_three_digit_number_is_constant_l2301_230114


namespace NUMINAMATH_GPT_one_fourth_of_2_pow_30_eq_2_pow_x_l2301_230133

theorem one_fourth_of_2_pow_30_eq_2_pow_x (x : ℕ) : (1 / 4 : ℝ) * (2:ℝ)^30 = (2:ℝ)^x → x = 28 := by
  sorry

end NUMINAMATH_GPT_one_fourth_of_2_pow_30_eq_2_pow_x_l2301_230133


namespace NUMINAMATH_GPT_correct_addition_result_l2301_230149

-- Define the particular number x and state the condition.
variable (x : ℕ) (h₁ : x + 21 = 52)

-- Assert that the correct result when adding 40 to x is 71.
theorem correct_addition_result : x + 40 = 71 :=
by
  -- Proof would go here; represented as a placeholder for now.
  sorry

end NUMINAMATH_GPT_correct_addition_result_l2301_230149


namespace NUMINAMATH_GPT_intersection_M_N_l2301_230170

def M : Set ℝ := { x | -1 < x ∧ x < 1 }
def N : Set ℝ := { x | x / (x - 1) ≤ 0 }

theorem intersection_M_N :
  M ∩ N = { x | 0 ≤ x ∧ x < 1 } :=
by
  sorry

end NUMINAMATH_GPT_intersection_M_N_l2301_230170


namespace NUMINAMATH_GPT_total_problems_l2301_230177

theorem total_problems (math_pages reading_pages problems_per_page : ℕ) :
  math_pages = 2 →
  reading_pages = 4 →
  problems_per_page = 5 →
  (math_pages + reading_pages) * problems_per_page = 30 :=
by
  sorry

end NUMINAMATH_GPT_total_problems_l2301_230177


namespace NUMINAMATH_GPT_find_initial_number_l2301_230164

theorem find_initial_number (x : ℕ) (h : ∃ y : ℕ, x * y = 4 ∧ y = 2) : x = 2 :=
by
  sorry

end NUMINAMATH_GPT_find_initial_number_l2301_230164


namespace NUMINAMATH_GPT_recurring_division_l2301_230106

def recurring_to_fraction (recurring: ℝ) (part: ℝ): ℝ :=
  part * recurring

theorem recurring_division (recurring: ℝ) (part1 part2: ℝ):
  recurring_to_fraction recurring part1 = 0.63 →
  recurring_to_fraction recurring part2 = 0.18 →
  recurring ≠ 0 →
  (0.63:ℝ)/0.18 = (7:ℝ)/2 :=
by
  intros h1 h2 h3
  rw [recurring_to_fraction] at h1 h2
  sorry

end NUMINAMATH_GPT_recurring_division_l2301_230106


namespace NUMINAMATH_GPT_sum_of_squares_of_roots_l2301_230154

theorem sum_of_squares_of_roots (s₁ s₂ : ℝ) (h1 : s₁ + s₂ = 9) (h2 : s₁ * s₂ = 14) :
  s₁^2 + s₂^2 = 53 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_squares_of_roots_l2301_230154


namespace NUMINAMATH_GPT_ball_max_height_l2301_230124

theorem ball_max_height : 
  (∃ t : ℝ, 
    ∀ u : ℝ, -16 * u ^ 2 + 80 * u + 35 ≤ -16 * t ^ 2 + 80 * t + 35 ∧ 
    -16 * t ^ 2 + 80 * t + 35 = 135) :=
sorry

end NUMINAMATH_GPT_ball_max_height_l2301_230124


namespace NUMINAMATH_GPT_center_square_number_l2301_230167

def in_center_square (grid : Matrix (Fin 3) (Fin 3) ℕ) : ℕ := grid 1 1

theorem center_square_number
  (grid : Matrix (Fin 3) (Fin 3) ℕ)
  (consecutive_share_edge : ∀ (i j : Fin 3) (n : ℕ), 
                              (i < 2 ∨ j < 2) →
                              (∃ d, d ∈ [(-1,0), (1,0), (0,-1), (0,1)] ∧ 
                              grid (i + d.1) (j + d.2) = n + 1))
  (corner_sum_20 : grid 0 0 + grid 0 2 + grid 2 0 + grid 2 2 = 20)
  (diagonal_sum_15 : 
    (grid 0 0 + grid 1 1 + grid 2 2 = 15) 
    ∨ 
    (grid 0 2 + grid 1 1 + grid 2 0 = 15))
  : in_center_square grid = 5 := sorry

end NUMINAMATH_GPT_center_square_number_l2301_230167


namespace NUMINAMATH_GPT_candidate_D_votes_l2301_230100

theorem candidate_D_votes :
  let total_votes := 10000
  let invalid_votes_percentage := 0.25
  let valid_votes := (1 - invalid_votes_percentage) * total_votes
  let candidate_A_percentage := 0.40
  let candidate_B_percentage := 0.30
  let candidate_C_percentage := 0.20
  let candidate_D_percentage := 1.0 - (candidate_A_percentage + candidate_B_percentage + candidate_C_percentage)
  let candidate_D_votes := candidate_D_percentage * valid_votes
  candidate_D_votes = 750 :=
by
  sorry

end NUMINAMATH_GPT_candidate_D_votes_l2301_230100


namespace NUMINAMATH_GPT_common_ratio_is_two_l2301_230117

theorem common_ratio_is_two (a r : ℝ) (h_pos : a > 0) 
  (h_sum : a + a * r + a * r^2 + a * r^3 = 5 * (a + a * r)) : 
  r = 2 := 
by
  sorry

end NUMINAMATH_GPT_common_ratio_is_two_l2301_230117


namespace NUMINAMATH_GPT_solution_set_of_inequality_l2301_230125

noncomputable def f (x : ℝ) : ℝ := 2^x - 2^(-x)

theorem solution_set_of_inequality :
  {x : ℝ | f (2 * x + 1) + f (1) ≥ 0} = {x : ℝ | -1 ≤ x} :=
by
  sorry

end NUMINAMATH_GPT_solution_set_of_inequality_l2301_230125


namespace NUMINAMATH_GPT_smallest_diff_PR_PQ_l2301_230107

theorem smallest_diff_PR_PQ (PQ PR QR : ℤ) (h1 : PQ < PR) (h2 : PR ≤ QR) (h3 : PQ + PR + QR = 2021) : 
  ∃ PQ PR QR : ℤ, PQ < PR ∧ PR ≤ QR ∧ PQ + PR + QR = 2021 ∧ PR - PQ = 1 :=
by
  sorry

end NUMINAMATH_GPT_smallest_diff_PR_PQ_l2301_230107


namespace NUMINAMATH_GPT_mul_mod_correct_l2301_230101

theorem mul_mod_correct :
  (2984 * 3998) % 1000 = 32 :=
by
  sorry

end NUMINAMATH_GPT_mul_mod_correct_l2301_230101


namespace NUMINAMATH_GPT_no_polynomials_exist_l2301_230186

open Polynomial

theorem no_polynomials_exist
  (a b : Polynomial ℂ) (c d : Polynomial ℂ) :
  ¬ (∀ x y : ℂ, 1 + x * y + x^2 * y^2 = a.eval x * c.eval y + b.eval x * d.eval y) :=
sorry

end NUMINAMATH_GPT_no_polynomials_exist_l2301_230186


namespace NUMINAMATH_GPT_find_coefficients_sum_l2301_230145

theorem find_coefficients_sum (a_0 a_1 a_2 a_3 : ℝ) (h : ∀ x : ℝ, x^3 = a_0 + a_1 * (x-2) + a_2 * (x-2)^2 + a_3 * (x-2)^3) :
  a_1 + a_2 + a_3 = 19 :=
by
  sorry

end NUMINAMATH_GPT_find_coefficients_sum_l2301_230145


namespace NUMINAMATH_GPT_josh_total_candies_l2301_230130

def josh_initial_candies (initial_candies given_siblings : ℕ) : Prop :=
  ∃ (remaining_1 best_friend josh_eats share_others : ℕ),
    (remaining_1 = initial_candies - given_siblings) ∧
    (best_friend = remaining_1 / 2) ∧
    (josh_eats = 16) ∧
    (share_others = 19) ∧
    (remaining_1 = 2 * (josh_eats + share_others))

theorem josh_total_candies : josh_initial_candies 100 30 :=
by
  sorry

end NUMINAMATH_GPT_josh_total_candies_l2301_230130


namespace NUMINAMATH_GPT_subcommittee_count_l2301_230175

-- Define the conditions: number of Republicans and Democrats in the Senate committee
def numRepublicans : ℕ := 10
def numDemocrats : ℕ := 8
def chooseRepublicans : ℕ := 4
def chooseDemocrats : ℕ := 3

-- Define the main proof problem based on the conditions and the correct answer
theorem subcommittee_count :
  (Nat.choose numRepublicans chooseRepublicans) * (Nat.choose numDemocrats chooseDemocrats) = 11760 := by
  sorry

end NUMINAMATH_GPT_subcommittee_count_l2301_230175


namespace NUMINAMATH_GPT_polar_to_rectangular_l2301_230104

theorem polar_to_rectangular (r θ : ℝ) (x y : ℝ) 
  (hr : r = 10) 
  (hθ : θ = (3 * Real.pi) / 4) 
  (hx : x = r * Real.cos θ) 
  (hy : y = r * Real.sin θ) 
  :
  x = -5 * Real.sqrt 2 ∧ y = 5 * Real.sqrt 2 := 
by
  -- We assume that the problem is properly stated
  -- Proof omitted here
  sorry

end NUMINAMATH_GPT_polar_to_rectangular_l2301_230104


namespace NUMINAMATH_GPT_total_money_collected_l2301_230127

def number_of_people := 610
def price_adult := 2
def price_child := 1
def number_of_adults := 350

theorem total_money_collected :
  (number_of_people - number_of_adults) * price_child + number_of_adults * price_adult = 960 := by
  sorry

end NUMINAMATH_GPT_total_money_collected_l2301_230127


namespace NUMINAMATH_GPT_average_marks_l2301_230165

theorem average_marks (M P C : ℕ) (h1 : M + P = 60) (h2 : C = P + 10) : (M + C) / 2 = 35 := 
by
  sorry

end NUMINAMATH_GPT_average_marks_l2301_230165


namespace NUMINAMATH_GPT_intersection_polar_coords_l2301_230132

noncomputable def polar_coord_intersection (rho theta : ℝ) : Prop :=
  (rho * (Real.sqrt 3 * Real.cos theta - Real.sin theta) = 2) ∧ (rho = 4 * Real.sin theta)

theorem intersection_polar_coords :
  ∃ (rho theta : ℝ), polar_coord_intersection rho theta ∧ rho = 2 ∧ theta = (Real.pi / 6) := 
sorry

end NUMINAMATH_GPT_intersection_polar_coords_l2301_230132


namespace NUMINAMATH_GPT_triangle_OMN_area_l2301_230122

noncomputable def rho (theta : ℝ) : ℝ := 4 * Real.cos theta + 2 * Real.sin theta

theorem triangle_OMN_area :
  let l1 (x y : ℝ) := y = (Real.sqrt 3 / 3) * x
  let l2 (x y : ℝ) := y = Real.sqrt 3 * x
  let C (x y : ℝ) := (x - 2)^2 + (y - 1)^2 = 5
  let OM := 2 * Real.sqrt 3 + 1
  let ON := 2 + Real.sqrt 3
  let angle_MON := Real.pi / 6
  let area_OMN := (1 / 2) * OM * ON * Real.sin angle_MON
  (4 * (Real.sqrt 3 + 2) + 5 * Real.sqrt 3 = 8 + 5 * Real.sqrt 3) → 
  area_OMN = (8 + 5 * Real.sqrt 3) / 4 :=
sorry

end NUMINAMATH_GPT_triangle_OMN_area_l2301_230122
