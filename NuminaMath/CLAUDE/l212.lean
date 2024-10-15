import Mathlib

namespace NUMINAMATH_CALUDE_baker_extra_donuts_l212_21286

theorem baker_extra_donuts (total_donuts : ℕ) (num_boxes : ℕ) 
  (h1 : total_donuts = 48) 
  (h2 : num_boxes = 7) : 
  total_donuts % num_boxes = 6 := by
  sorry

end NUMINAMATH_CALUDE_baker_extra_donuts_l212_21286


namespace NUMINAMATH_CALUDE_necklace_beads_l212_21209

theorem necklace_beads (total blue red white silver : ℕ) : 
  total = 40 →
  blue = 5 →
  red = 2 * blue →
  white = blue + red →
  total = blue + red + white + silver →
  silver = 10 := by
sorry

end NUMINAMATH_CALUDE_necklace_beads_l212_21209


namespace NUMINAMATH_CALUDE_probability_sum_eight_l212_21244

/-- A fair die with 6 faces -/
structure Die :=
  (faces : Fin 6)

/-- The result of throwing two dice -/
structure TwoDiceThrow :=
  (die1 : Die)
  (die2 : Die)

/-- The sum of the numbers on two dice -/
def diceSum (throw : TwoDiceThrow) : Nat :=
  throw.die1.faces.val + 1 + throw.die2.faces.val + 1

/-- The set of all possible throws of two dice -/
def allThrows : Finset TwoDiceThrow :=
  sorry

/-- The set of throws where the sum is 8 -/
def sumEightThrows : Finset TwoDiceThrow :=
  sorry

/-- The probability of an event occurring when throwing two fair dice -/
def probability (event : Finset TwoDiceThrow) : Rat :=
  (event.card : Rat) / (allThrows.card : Rat)

theorem probability_sum_eight :
  probability sumEightThrows = 5 / 36 :=
sorry

end NUMINAMATH_CALUDE_probability_sum_eight_l212_21244


namespace NUMINAMATH_CALUDE_bottle_capacity_l212_21223

theorem bottle_capacity (V : ℝ) 
  (h1 : V > 0) 
  (h2 : (0.12 * V - 0.24 + 0.12 / V) / V = 0.03) : 
  V = 2 := by
sorry

end NUMINAMATH_CALUDE_bottle_capacity_l212_21223


namespace NUMINAMATH_CALUDE_f_equals_g_l212_21294

-- Define the functions f and g
def f (x : ℝ) : ℝ := x - 1
def g (t : ℝ) : ℝ := t - 1

-- Theorem stating that f and g represent the same function
theorem f_equals_g : ∀ x : ℝ, f x = g x := by
  sorry

end NUMINAMATH_CALUDE_f_equals_g_l212_21294


namespace NUMINAMATH_CALUDE_calculation_proof_equation_solution_l212_21231

-- Problem 1
theorem calculation_proof : -1^2024 + |(-3)| - (Real.pi + 1)^0 = 1 := by sorry

-- Problem 2
theorem equation_solution : 
  ∃ x : ℝ, (2 / (x + 2) = 4 / (x^2 - 4)) ∧ (x = 4) := by sorry

end NUMINAMATH_CALUDE_calculation_proof_equation_solution_l212_21231


namespace NUMINAMATH_CALUDE_borrowed_amount_l212_21247

/-- Calculates the total interest paid over 9 years given the principal amount and interest rates -/
def totalInterest (principal : ℝ) : ℝ :=
  principal * (0.06 * 2 + 0.09 * 3 + 0.14 * 4)

/-- Theorem stating that given the interest rates and total interest paid, the principal amount borrowed is 12000 -/
theorem borrowed_amount (totalInterestPaid : ℝ) 
  (h : totalInterestPaid = 11400) : 
  ∃ principal : ℝ, totalInterest principal = totalInterestPaid ∧ principal = 12000 := by
  sorry

end NUMINAMATH_CALUDE_borrowed_amount_l212_21247


namespace NUMINAMATH_CALUDE_diophantine_equation_solutions_l212_21212

theorem diophantine_equation_solutions :
  ∀ x y z w : ℕ, 2^x * 3^y - 5^x * 7^w = 1 ↔ 
  (x = 1 ∧ y = 0 ∧ z = 0 ∧ w = 0) ∨
  (x = 3 ∧ y = 0 ∧ z = 0 ∧ w = 1) ∨
  (x = 1 ∧ y = 1 ∧ z = 1 ∧ w = 0) ∨
  (x = 2 ∧ y = 2 ∧ z = 1 ∧ w = 1) :=
by sorry

end NUMINAMATH_CALUDE_diophantine_equation_solutions_l212_21212


namespace NUMINAMATH_CALUDE_geometric_sequence_second_term_l212_21237

/-- Given a geometric sequence with common ratio 2 and sum of first 4 terms equal to 60,
    prove that the second term is 8. -/
theorem geometric_sequence_second_term :
  ∀ (a : ℕ → ℝ), 
  (∀ n, a (n + 1) = 2 * a n) →  -- Common ratio q = 2
  (a 1 + a 2 + a 3 + a 4 = 60) →  -- Sum of first 4 terms S_4 = 60
  a 2 = 8 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_second_term_l212_21237


namespace NUMINAMATH_CALUDE_sphere_volume_and_radius_ratio_l212_21226

theorem sphere_volume_and_radius_ratio (V_large V_small : ℝ) (h1 : V_large = 432 * Real.pi) (h2 : V_small = 0.15 * V_large) : 
  ∃ (r_large r_small : ℝ), 
    (4 / 3 * Real.pi * r_large ^ 3 = V_large) ∧ 
    (4 / 3 * Real.pi * r_small ^ 3 = V_small) ∧
    (r_small / r_large = Real.rpow 1.8 (1/3) / 2) ∧
    (V_large + V_small = 496.8 * Real.pi) := by
  sorry

end NUMINAMATH_CALUDE_sphere_volume_and_radius_ratio_l212_21226


namespace NUMINAMATH_CALUDE_min_apples_count_l212_21255

theorem min_apples_count : ∃ n : ℕ, n > 0 ∧
  n % 4 = 1 ∧
  n % 5 = 2 ∧
  n % 9 = 7 ∧
  (∀ m : ℕ, m > 0 ∧ m % 4 = 1 ∧ m % 5 = 2 ∧ m % 9 = 7 → n ≤ m) ∧
  n = 97 := by
sorry

end NUMINAMATH_CALUDE_min_apples_count_l212_21255


namespace NUMINAMATH_CALUDE_lucas_150_mod_5_l212_21265

/-- Lucas sequence -/
def lucas : ℕ → ℕ
  | 0 => 1
  | 1 => 3
  | n + 2 => lucas n + lucas (n + 1)

/-- The 150th term of the Lucas sequence modulo 5 is equal to 3 -/
theorem lucas_150_mod_5 : lucas 149 % 5 = 3 := by
  sorry

end NUMINAMATH_CALUDE_lucas_150_mod_5_l212_21265


namespace NUMINAMATH_CALUDE_semicircle_area_with_inscribed_rectangle_l212_21201

theorem semicircle_area_with_inscribed_rectangle :
  ∀ (r : ℝ),
  r > 0 →
  ∃ (semicircle_area : ℝ),
  (3 : ℝ)^2 + 1^2 = (2 * r)^2 →
  semicircle_area = (13 * π) / 8 :=
by
  sorry

end NUMINAMATH_CALUDE_semicircle_area_with_inscribed_rectangle_l212_21201


namespace NUMINAMATH_CALUDE_cinema_seat_removal_l212_21263

/-- The number of seats that should be removed from a cinema with
    total_seats arranged in rows of seats_per_row, given expected_attendees,
    to minimize unoccupied seats while ensuring full rows. -/
def seats_to_remove (total_seats seats_per_row expected_attendees : ℕ) : ℕ :=
  total_seats - (((expected_attendees + seats_per_row - 1) / seats_per_row) * seats_per_row)

/-- Theorem stating that for the given cinema setup, 88 seats should be removed. -/
theorem cinema_seat_removal :
  seats_to_remove 240 8 150 = 88 := by
  sorry

end NUMINAMATH_CALUDE_cinema_seat_removal_l212_21263


namespace NUMINAMATH_CALUDE_cupcake_cost_proof_l212_21235

theorem cupcake_cost_proof (total_cupcakes : ℕ) (people : ℕ) (cost_per_person : ℚ) :
  total_cupcakes = 12 →
  people = 2 →
  cost_per_person = 9 →
  (people * cost_per_person) / total_cupcakes = 1.5 :=
by sorry

end NUMINAMATH_CALUDE_cupcake_cost_proof_l212_21235


namespace NUMINAMATH_CALUDE_library_book_count_l212_21284

def library_books (initial_books : ℕ) (books_bought_two_years_ago : ℕ) (additional_books_last_year : ℕ) (books_donated : ℕ) : ℕ :=
  initial_books + books_bought_two_years_ago + (books_bought_two_years_ago + additional_books_last_year) - books_donated

theorem library_book_count : 
  library_books 500 300 100 200 = 1000 := by
  sorry

end NUMINAMATH_CALUDE_library_book_count_l212_21284


namespace NUMINAMATH_CALUDE_problem_solution_l212_21206

theorem problem_solution : 
  ((-1)^2023 - Real.sqrt (2 + 1/4) + ((-1 : ℝ)^(1/3 : ℝ)) + 1/2 = -3) ∧ 
  (2 * Real.sqrt 3 + |1 - Real.sqrt 3| - (-1)^2022 + 2 = 3 * Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l212_21206


namespace NUMINAMATH_CALUDE_smallest_four_digit_divisible_by_35_l212_21228

theorem smallest_four_digit_divisible_by_35 : 
  ∀ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ 35 ∣ n → n ≥ 1200 := by
  sorry

end NUMINAMATH_CALUDE_smallest_four_digit_divisible_by_35_l212_21228


namespace NUMINAMATH_CALUDE_inverse_71_mod_83_l212_21243

theorem inverse_71_mod_83 (h : (17⁻¹ : ZMod 83) = 53) : (71⁻¹ : ZMod 83) = 53 := by
  sorry

end NUMINAMATH_CALUDE_inverse_71_mod_83_l212_21243


namespace NUMINAMATH_CALUDE_fraction_equality_l212_21264

theorem fraction_equality : (8 : ℚ) / (4 * 25) = 0.8 / (0.4 * 25) := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l212_21264


namespace NUMINAMATH_CALUDE_work_completion_men_count_l212_21270

/-- Proves that the number of men in the second group is 15, given the conditions of the problem -/
theorem work_completion_men_count : 
  ∀ (work : ℕ) (men1 men2 days1 days2 : ℕ),
    men1 = 18 →
    days1 = 20 →
    days2 = 24 →
    work = men1 * days1 →
    work = men2 * days2 →
    men2 = 15 :=
by
  sorry

end NUMINAMATH_CALUDE_work_completion_men_count_l212_21270


namespace NUMINAMATH_CALUDE_snickers_bought_l212_21272

-- Define the cost of a single Snickers
def snickers_cost : ℚ := 3/2

-- Define the number of M&M packs bought
def mm_packs : ℕ := 3

-- Define the total amount paid
def total_paid : ℚ := 20

-- Define the change received
def change : ℚ := 8

-- Define the relationship between M&M pack cost and Snickers cost
def mm_pack_cost (s : ℚ) : ℚ := 2 * s

-- Theorem to prove
theorem snickers_bought :
  ∃ (n : ℕ), (n : ℚ) * snickers_cost + mm_packs * mm_pack_cost snickers_cost = total_paid - change ∧ n = 2 := by
  sorry


end NUMINAMATH_CALUDE_snickers_bought_l212_21272


namespace NUMINAMATH_CALUDE_triangle_ABC_theorem_l212_21240

open Real

noncomputable def triangle_ABC (a b c : ℝ) (A B C : ℝ) : Prop :=
  0 < a ∧ 0 < b ∧ 0 < c ∧ 
  0 < A ∧ A < π ∧ 0 < B ∧ B < π ∧ 0 < C ∧ C < π ∧
  A + B + C = π

theorem triangle_ABC_theorem 
  (a b c : ℝ) (A B C : ℝ) 
  (h_triangle : triangle_ABC a b c A B C)
  (h_eq : a * sin B - Real.sqrt 3 * b * cos A = 0) :
  A = π / 3 ∧ 
  (a = 3 → 
    (∃ (max_area : ℝ), max_area = 9 * Real.sqrt 3 / 4 ∧
      ∀ (b' c' : ℝ), triangle_ABC 3 b' c' A B C → 
        1/2 * 3 * b' * sin A ≤ max_area ∧
        (1/2 * 3 * b' * sin A = max_area → b' = 3 ∧ c' = 3))) :=
sorry

end NUMINAMATH_CALUDE_triangle_ABC_theorem_l212_21240


namespace NUMINAMATH_CALUDE_trigonometric_equation_solution_l212_21288

theorem trigonometric_equation_solution (x : ℝ) :
  (2 * Real.sin x ^ 3 + 2 * Real.sin x ^ 2 * Real.cos x - Real.sin x * Real.cos x ^ 2 - Real.cos x ^ 3 = 0) ↔
  (∃ n : ℤ, x = -π / 4 + n * π) ∨
  (∃ k : ℤ, x = Real.arctan (Real.sqrt 2 / 2) + k * π) ∨
  (∃ k : ℤ, x = -Real.arctan (Real.sqrt 2 / 2) + k * π) := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_equation_solution_l212_21288


namespace NUMINAMATH_CALUDE_football_team_progress_l212_21251

def team_progress (loss : Int) (gain : Int) : Int :=
  gain - loss

theorem football_team_progress :
  team_progress 5 13 = 8 := by
  sorry

end NUMINAMATH_CALUDE_football_team_progress_l212_21251


namespace NUMINAMATH_CALUDE_total_keys_needed_l212_21220

theorem total_keys_needed 
  (num_complexes : ℕ) 
  (apartments_per_complex : ℕ) 
  (keys_per_apartment : ℕ) 
  (h1 : num_complexes = 2) 
  (h2 : apartments_per_complex = 12) 
  (h3 : keys_per_apartment = 3) : 
  num_complexes * apartments_per_complex * keys_per_apartment = 72 := by
sorry

end NUMINAMATH_CALUDE_total_keys_needed_l212_21220


namespace NUMINAMATH_CALUDE_cubic_quintic_inequality_l212_21222

theorem cubic_quintic_inequality (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (sum_eq_one : a + b + c = 1) : 
  10 * (a^3 + b^3 + c^3) - 9 * (a^5 + b^5 + c^5) ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_cubic_quintic_inequality_l212_21222


namespace NUMINAMATH_CALUDE_wages_calculation_l212_21290

/-- The wages calculation problem -/
theorem wages_calculation 
  (initial_workers : ℕ) 
  (initial_days : ℕ) 
  (initial_wages : ℚ) 
  (new_workers : ℕ) 
  (new_days : ℕ) 
  (h1 : initial_workers = 15) 
  (h2 : initial_days = 6) 
  (h3 : initial_wages = 9450) 
  (h4 : new_workers = 19) 
  (h5 : new_days = 5) : 
  (initial_wages / (initial_workers * initial_days : ℚ)) * (new_workers * new_days) = 9975 :=
by sorry

end NUMINAMATH_CALUDE_wages_calculation_l212_21290


namespace NUMINAMATH_CALUDE_triangle_trip_distance_l212_21268

/-- Given a right-angled triangle DEF with F as the right angle, 
    where DF = 2000 and DE = 4500, prove that DE + EF + DF = 10531 -/
theorem triangle_trip_distance (DE DF EF : ℝ) : 
  DE = 4500 → 
  DF = 2000 → 
  EF ^ 2 = DE ^ 2 - DF ^ 2 → 
  DE + EF + DF = 10531 := by
sorry

end NUMINAMATH_CALUDE_triangle_trip_distance_l212_21268


namespace NUMINAMATH_CALUDE_no_valid_chessboard_config_l212_21298

/-- A chessboard configuration is a function from (Fin 8 × Fin 8) to Fin 64 -/
def ChessboardConfig := Fin 8 × Fin 8 → Fin 64

/-- A 2x2 square on the chessboard -/
structure Square (config : ChessboardConfig) where
  row : Fin 7
  col : Fin 7

/-- The sum of numbers in a 2x2 square -/
def squareSum (config : ChessboardConfig) (square : Square config) : ℕ :=
  (config (square.row, square.col)).val + 1 +
  (config (square.row, square.col.succ)).val + 1 +
  (config (square.row.succ, square.col)).val + 1 +
  (config (square.row.succ, square.col.succ)).val + 1

/-- A valid configuration satisfies the divisibility condition for all 2x2 squares -/
def isValidConfig (config : ChessboardConfig) : Prop :=
  (∀ square : Square config, (squareSum config square) % 5 = 0) ∧
  Function.Injective config

theorem no_valid_chessboard_config : ¬ ∃ config : ChessboardConfig, isValidConfig config := by
  sorry

end NUMINAMATH_CALUDE_no_valid_chessboard_config_l212_21298


namespace NUMINAMATH_CALUDE_sqrt_x_minus_one_real_l212_21275

theorem sqrt_x_minus_one_real (x : ℝ) : (∃ y : ℝ, y ^ 2 = x - 1) ↔ x ≥ 1 := by sorry

end NUMINAMATH_CALUDE_sqrt_x_minus_one_real_l212_21275


namespace NUMINAMATH_CALUDE_manufacturing_employee_percentage_l212_21274

theorem manufacturing_employee_percentage 
  (total_degrees : ℝ) 
  (manufacturing_degrees : ℝ) 
  (h1 : total_degrees = 360) 
  (h2 : manufacturing_degrees = 72) : 
  (manufacturing_degrees / total_degrees) * 100 = 20 := by
sorry

end NUMINAMATH_CALUDE_manufacturing_employee_percentage_l212_21274


namespace NUMINAMATH_CALUDE_square_perimeter_from_diagonal_l212_21295

theorem square_perimeter_from_diagonal (d : ℝ) (h : d = 12) :
  let side := d / Real.sqrt 2
  4 * side = 24 * Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_square_perimeter_from_diagonal_l212_21295


namespace NUMINAMATH_CALUDE_contrapositive_odd_sum_l212_21289

theorem contrapositive_odd_sum (x y : ℤ) :
  (¬(Odd (x + y)) → ¬(Odd x ∧ Odd y)) ↔
  (∀ x y : ℤ, (Odd x ∧ Odd y) → Odd (x + y)) :=
by sorry

end NUMINAMATH_CALUDE_contrapositive_odd_sum_l212_21289


namespace NUMINAMATH_CALUDE_sin_180_degrees_equals_zero_l212_21203

theorem sin_180_degrees_equals_zero : Real.sin (π) = 0 := by
  sorry

end NUMINAMATH_CALUDE_sin_180_degrees_equals_zero_l212_21203


namespace NUMINAMATH_CALUDE_parallel_line_with_chord_l212_21262

/-- Given a line parallel to 3x + 3y + 5 = 0 and intercepted by the circle x² + y² = 20
    with a chord length of 6√2, prove that the equation of the line is x + y ± 2 = 0 -/
theorem parallel_line_with_chord (a b c : ℝ) : 
  (∃ k : ℝ, a = 3 * k ∧ b = 3 * k) → -- Line is parallel to 3x + 3y + 5 = 0
  (∀ x y : ℝ, a * x + b * y + c = 0 → x^2 + y^2 ≤ 20) → -- Line intersects the circle
  (∃ x₁ y₁ x₂ y₂ : ℝ, 
    a * x₁ + b * y₁ + c = 0 ∧
    a * x₂ + b * y₂ + c = 0 ∧
    x₁^2 + y₁^2 = 20 ∧
    x₂^2 + y₂^2 = 20 ∧
    (x₁ - x₂)^2 + (y₁ - y₂)^2 = 72) → -- Chord length is 6√2
  ∃ s : ℝ, (s = 1 ∨ s = -1) ∧ a * x + b * y + c = 0 ↔ x + y + 2 * s = 0 :=
by sorry

end NUMINAMATH_CALUDE_parallel_line_with_chord_l212_21262


namespace NUMINAMATH_CALUDE_probability_two_not_selected_l212_21229

theorem probability_two_not_selected (S : Finset Nat) (a b : Nat) 
  (h1 : S.card = 4) (h2 : a ∈ S) (h3 : b ∈ S) (h4 : a ≠ b) :
  (Finset.filter (λ T : Finset Nat => T.card = 2 ∧ a ∉ T ∧ b ∉ T) (S.powerset)).card / (Finset.filter (λ T : Finset Nat => T.card = 2) (S.powerset)).card = 1 / 6 :=
sorry

end NUMINAMATH_CALUDE_probability_two_not_selected_l212_21229


namespace NUMINAMATH_CALUDE_dividing_chord_length_l212_21227

/-- An octagon inscribed in a circle -/
structure InscribedOctagon :=
  (side_length_1 : ℝ)
  (side_length_2 : ℝ)
  (h1 : side_length_1 > 0)
  (h2 : side_length_2 > 0)

/-- The chord dividing the octagon into two quadrilaterals -/
def dividing_chord (o : InscribedOctagon) : ℝ := sorry

/-- Theorem stating the length of the dividing chord -/
theorem dividing_chord_length (o : InscribedOctagon) 
  (h3 : o.side_length_1 = 4)
  (h4 : o.side_length_2 = 6) : 
  dividing_chord o = 4 := by sorry

end NUMINAMATH_CALUDE_dividing_chord_length_l212_21227


namespace NUMINAMATH_CALUDE_tangent_line_at_2_monotonicity_intervals_l212_21230

/-- The function f(x) = 3x - x^3 -/
def f (x : ℝ) : ℝ := 3 * x - x^3

/-- The derivative of f(x) -/
def f' (x : ℝ) : ℝ := 3 - 3 * x^2

theorem tangent_line_at_2 :
  ∃ (m b : ℝ), m = -9 ∧ b = 16 ∧
  ∀ x, f x = m * (x - 2) + f 2 + b - f 2 :=
sorry

theorem monotonicity_intervals :
  (∀ x y, x < y ∧ y < -1 → f y < f x) ∧
  (∀ x y, -1 < x ∧ x < y ∧ y < 1 → f x < f y) ∧
  (∀ x y, 1 < x ∧ x < y → f y < f x) :=
sorry

end NUMINAMATH_CALUDE_tangent_line_at_2_monotonicity_intervals_l212_21230


namespace NUMINAMATH_CALUDE_first_investment_interest_rate_l212_21208

/-- Proves that the annual simple interest rate of the first investment is 8.5% -/
theorem first_investment_interest_rate 
  (total_income : ℝ) 
  (total_invested : ℝ) 
  (first_investment : ℝ) 
  (second_investment : ℝ) 
  (second_rate : ℝ) :
  total_income = 575 →
  total_invested = 8000 →
  first_investment = 3000 →
  second_investment = 5000 →
  second_rate = 0.064 →
  ∃ (first_rate : ℝ), 
    first_rate = 0.085 ∧ 
    total_income = first_investment * first_rate + second_investment * second_rate :=
by
  sorry

end NUMINAMATH_CALUDE_first_investment_interest_rate_l212_21208


namespace NUMINAMATH_CALUDE_smallest_angle_cosine_equality_l212_21239

theorem smallest_angle_cosine_equality (θ : Real) : 
  (θ > 0) →
  (Real.cos θ = Real.sin (π/4) + Real.cos (π/3) - Real.sin (π/6) - Real.cos (π/12)) →
  (θ = π/6) :=
by sorry

end NUMINAMATH_CALUDE_smallest_angle_cosine_equality_l212_21239


namespace NUMINAMATH_CALUDE_planting_area_difference_l212_21205

/-- Given a village with wheat, rice, and corn planting areas, prove the difference between rice and corn areas. -/
theorem planting_area_difference (m : ℝ) : 
  let wheat_area : ℝ := m
  let rice_area : ℝ := 2 * wheat_area + 3
  let corn_area : ℝ := wheat_area - 5
  rice_area - corn_area = m + 8 := by
  sorry

end NUMINAMATH_CALUDE_planting_area_difference_l212_21205


namespace NUMINAMATH_CALUDE_preservation_time_at_33_l212_21283

/-- The preservation time function -/
noncomputable def preservation_time (k b x : ℝ) : ℝ := Real.exp (k * x + b)

/-- Theorem stating the preservation time at 33°C given conditions -/
theorem preservation_time_at_33 (k b : ℝ) :
  preservation_time k b 0 = 192 →
  preservation_time k b 22 = 48 →
  preservation_time k b 33 = 24 := by
  sorry

end NUMINAMATH_CALUDE_preservation_time_at_33_l212_21283


namespace NUMINAMATH_CALUDE_problem_statement_l212_21207

noncomputable def f (x : ℝ) : ℝ := Real.sin (x - Real.pi / 6) + 1 / 2

theorem problem_statement :
  ∀ x A B C a b c : ℝ,
  -- Part 1 conditions
  (x ∈ Set.Icc 0 (Real.pi / 2)) →
  (f x = 11 / 10) →
  -- Part 1 conclusion
  (Real.cos x = (4 * Real.sqrt 3 - 3) / 10) ∧
  -- Part 2 conditions
  (0 < A ∧ A < Real.pi) →
  (0 < B ∧ B < Real.pi) →
  (0 < C ∧ C < Real.pi) →
  (A + B + C = Real.pi) →
  (a / Real.sin A = b / Real.sin B) →
  (b / Real.sin B = c / Real.sin C) →
  (2 * b * Real.cos A ≤ 2 * c - Real.sqrt 3 * a) →
  -- Part 2 conclusion
  (f B ∈ Set.Ioc 0 (1 / 2)) :=
by sorry

end NUMINAMATH_CALUDE_problem_statement_l212_21207


namespace NUMINAMATH_CALUDE_circle_chord_intersection_area_l212_21232

theorem circle_chord_intersection_area (r : ℝ) (chord_length : ℝ) (intersection_dist : ℝ)
  (h_r : r = 30)
  (h_chord : chord_length = 50)
  (h_dist : intersection_dist = 14) :
  ∃ (m n d : ℕ), 
    (0 < m) ∧ (0 < n) ∧ (0 < d) ∧
    (∀ (p : ℕ), Prime p → ¬(p^2 ∣ d)) ∧
    (∃ (area : ℝ), area = m * Real.pi - n * Real.sqrt d) ∧
    (m + n + d = 162) :=
by sorry

end NUMINAMATH_CALUDE_circle_chord_intersection_area_l212_21232


namespace NUMINAMATH_CALUDE_boat_downstream_time_l212_21279

def boat_problem (boat_speed : ℝ) (stream_speed : ℝ) (upstream_time : ℝ) : Prop :=
  let downstream_speed := boat_speed + stream_speed
  let upstream_speed := boat_speed - stream_speed
  let distance := upstream_speed * upstream_time
  let downstream_time := distance / downstream_speed
  downstream_time = 1

theorem boat_downstream_time :
  boat_problem 15 3 1.5 := by sorry

end NUMINAMATH_CALUDE_boat_downstream_time_l212_21279


namespace NUMINAMATH_CALUDE_males_not_listening_l212_21224

theorem males_not_listening (males_listening : ℕ) (females_not_listening : ℕ) 
  (total_listening : ℕ) (total_not_listening : ℕ) 
  (h1 : males_listening = 45)
  (h2 : females_not_listening = 87)
  (h3 : total_listening = 115)
  (h4 : total_not_listening = 160) : 
  total_listening + total_not_listening - (males_listening + (total_listening - males_listening + females_not_listening)) = 73 :=
by sorry

end NUMINAMATH_CALUDE_males_not_listening_l212_21224


namespace NUMINAMATH_CALUDE_jacket_markup_percentage_l212_21297

/-- Proves that the markup percentage is 40% given the conditions of the jacket sale problem -/
theorem jacket_markup_percentage (purchase_price : ℝ) (selling_price : ℝ) (markup_percentage : ℝ) 
  (sale_discount : ℝ) (gross_profit : ℝ) :
  purchase_price = 48 →
  selling_price = purchase_price + markup_percentage * selling_price →
  sale_discount = 0.2 →
  gross_profit = 16 →
  (1 - sale_discount) * selling_price - purchase_price = gross_profit →
  markup_percentage = 0.4 := by
sorry

end NUMINAMATH_CALUDE_jacket_markup_percentage_l212_21297


namespace NUMINAMATH_CALUDE_locus_is_ellipse_l212_21281

/-- The locus of points P(x,y) satisfying the given conditions forms an ellipse -/
theorem locus_is_ellipse (x y : ℝ) : 
  let A : ℝ × ℝ := (1, 0)
  let directrix : ℝ → Prop := λ t => t = 9
  let dist_ratio : ℝ := 1/3
  let dist_to_A : ℝ := Real.sqrt ((x - A.1)^2 + (y - A.2)^2)
  let dist_to_directrix : ℝ := |x - 9|
  dist_to_A / dist_to_directrix = dist_ratio →
  x^2/9 + y^2/8 = 1 :=
by sorry

end NUMINAMATH_CALUDE_locus_is_ellipse_l212_21281


namespace NUMINAMATH_CALUDE_simplify_expression_l212_21216

theorem simplify_expression (a b : ℝ) : -3 * (a - b) + (2 * a - 3 * b) = -a := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l212_21216


namespace NUMINAMATH_CALUDE_no_divisible_by_nine_l212_21234

def base_n_number (n : ℕ) : ℕ := 3 + 2*n + 1*n^2 + 0*n^3 + 3*n^4 + 2*n^5

theorem no_divisible_by_nine :
  ∀ n : ℕ, 2 ≤ n → n ≤ 100 → ¬(base_n_number n % 9 = 0) := by
sorry

end NUMINAMATH_CALUDE_no_divisible_by_nine_l212_21234


namespace NUMINAMATH_CALUDE_angle_measure_in_special_triangle_l212_21276

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    if b² = a² + ac + c², then the measure of angle B is 120°. -/
theorem angle_measure_in_special_triangle (a b c : ℝ) (h : b^2 = a^2 + a*c + c^2) :
  let angle_B := Real.arccos (-1/2)
  angle_B = 2 * π / 3 := by
  sorry

end NUMINAMATH_CALUDE_angle_measure_in_special_triangle_l212_21276


namespace NUMINAMATH_CALUDE_smallest_multiple_of_5_and_24_l212_21282

theorem smallest_multiple_of_5_and_24 : ∃ n : ℕ, n > 0 ∧ n % 5 = 0 ∧ n % 24 = 0 ∧ ∀ m : ℕ, m > 0 → m % 5 = 0 → m % 24 = 0 → n ≤ m := by
  sorry

end NUMINAMATH_CALUDE_smallest_multiple_of_5_and_24_l212_21282


namespace NUMINAMATH_CALUDE_kaylin_age_is_33_l212_21269

def freyja_age : ℕ := 10
def eli_age : ℕ := freyja_age + 9
def sarah_age : ℕ := 2 * eli_age
def kaylin_age : ℕ := sarah_age - 5

theorem kaylin_age_is_33 : kaylin_age = 33 := by
  sorry

end NUMINAMATH_CALUDE_kaylin_age_is_33_l212_21269


namespace NUMINAMATH_CALUDE_aubree_animal_count_l212_21211

/-- The total number of animals Aubree saw in a day, given the initial counts and changes --/
def total_animals (initial_beavers initial_chipmunks : ℕ) : ℕ :=
  let morning_total := initial_beavers + initial_chipmunks
  let afternoon_beavers := 2 * initial_beavers
  let afternoon_chipmunks := initial_chipmunks - 10
  let afternoon_total := afternoon_beavers + afternoon_chipmunks
  morning_total + afternoon_total

/-- Theorem stating that the total number of animals seen is 130 --/
theorem aubree_animal_count : total_animals 20 40 = 130 := by
  sorry

end NUMINAMATH_CALUDE_aubree_animal_count_l212_21211


namespace NUMINAMATH_CALUDE_stratified_sampling_most_appropriate_l212_21280

-- Define the type for sampling methods
inductive SamplingMethod
| Lottery
| RandomNumber
| Stratified
| Systematic

-- Define the company's production
structure Company where
  sedanModels : Nat
  significantDifferences : Bool

-- Define the appropriateness of a sampling method
def isAppropriate (method : SamplingMethod) (company : Company) : Prop :=
  method = SamplingMethod.Stratified ∧ 
  company.sedanModels > 1 ∧ 
  company.significantDifferences

-- Theorem statement
theorem stratified_sampling_most_appropriate (company : Company) 
  (h1 : company.sedanModels = 3) 
  (h2 : company.significantDifferences = true) :
  isAppropriate SamplingMethod.Stratified company := by
  sorry

end NUMINAMATH_CALUDE_stratified_sampling_most_appropriate_l212_21280


namespace NUMINAMATH_CALUDE_bobbys_remaining_candy_l212_21291

/-- Given Bobby's initial candy count and the amounts eaten, prove that the remaining candy count is 8. -/
theorem bobbys_remaining_candy (initial_candy : ℕ) (first_eaten : ℕ) (second_eaten : ℕ)
  (h1 : initial_candy = 22)
  (h2 : first_eaten = 9)
  (h3 : second_eaten = 5) :
  initial_candy - first_eaten - second_eaten = 8 := by
  sorry

end NUMINAMATH_CALUDE_bobbys_remaining_candy_l212_21291


namespace NUMINAMATH_CALUDE_unique_original_message_exists_l212_21292

/-- Represents a cryptogram as a list of characters -/
def Cryptogram := List Char

/-- Represents a bijective letter substitution -/
def Substitution := Char → Char

/-- The first cryptogram -/
def cryptogram1 : Cryptogram := 
  ['М', 'И', 'М', 'О', 'П', 'Р', 'А', 'С', 'Т', 'Е', 'Т', 'И', 'Р', 'А', 'С', 'И', 'С', 'П', 'Д', 'А', 'И', 'С', 'А', 'Ф', 'Е', 'И', 'И', 'Б', 'О', 'Е', 'Т', 'К', 'Ж', 'Р', 'Г', 'Л', 'Е', 'О', 'Л', 'О', 'И', 'Ш', 'И', 'С', 'А', 'Н', 'Н', 'С', 'Й', 'С', 'А', 'О', 'О', 'Л', 'Т', 'Л', 'Е', 'Я', 'Т', 'У', 'И', 'Ц', 'В', 'Ы', 'И', 'П', 'И', 'Я', 'Д', 'П', 'И', 'Щ', 'П', 'Ь', 'П', 'С', 'Е', 'Ю', 'Я', 'Я']

/-- The second cryptogram -/
def cryptogram2 : Cryptogram := 
  ['У', 'Щ', 'Ф', 'М', 'Ш', 'П', 'Д', 'Р', 'Е', 'Ц', 'Ч', 'Е', 'Ш', 'Ю', 'Ч', 'Д', 'А', 'К', 'Е', 'Ч', 'М', 'Д', 'В', 'К', 'Ш', 'Б', 'Е', 'Е', 'Ч', 'Д', 'Ф', 'Э', 'П', 'Й', 'Щ', 'Г', 'Ш', 'Ф', 'Щ', 'Ц', 'Е', 'Ю', 'Щ', 'Ф', 'П', 'М', 'Е', 'Ч', 'П', 'М', 'Р', 'Р', 'М', 'Е', 'О', 'Ч', 'Х', 'Е', 'Ш', 'Р', 'Т', 'Г', 'И', 'Ф', 'Р', 'С', 'Я', 'Ы', 'Л', 'К', 'Д', 'Ф', 'Ф', 'Е', 'Е']

/-- The original message -/
def original_message : Cryptogram := 
  ['Ш', 'Е', 'С', 'Т', 'А', 'Я', 'О', 'Л', 'И', 'М', 'П', 'И', 'А', 'Д', 'А', 'П', 'О', 'К', 'Р', 'И', 'П', 'Т', 'О', 'Г', 'Р', 'А', 'Ф', 'И', 'И', 'П', 'О', 'С', 'В', 'Я', 'Щ', 'Е', 'Н', 'А', 'С', 'Е', 'М', 'И', 'Д', 'Е', 'С', 'Я', 'Т', 'И', 'П', 'Я', 'Т', 'И', 'Л', 'Е', 'Т', 'И', 'Ю', 'С', 'П', 'Е', 'Ц', 'И', 'А', 'Л', 'Ь', 'Н', 'О', 'Й', 'С', 'Л', 'У', 'Ж', 'Б', 'Ы', 'Р', 'О', 'С', 'С', 'И', 'И']

/-- Predicate to check if a list is a permutation of another list -/
def is_permutation (l1 l2 : List α) : Prop := sorry

/-- Predicate to check if a function is bijective on a given list -/
def is_bijective_on (f : α → β) (l : List α) : Prop := sorry

/-- Main theorem: There exists a unique original message that satisfies the cryptogram conditions -/
theorem unique_original_message_exists : 
  ∃! (msg : Cryptogram), 
    (is_permutation msg cryptogram1) ∧ 
    (∃ (subst : Substitution), 
      (is_bijective_on subst msg) ∧ 
      (cryptogram2 = msg.map subst)) :=
sorry

end NUMINAMATH_CALUDE_unique_original_message_exists_l212_21292


namespace NUMINAMATH_CALUDE_height_less_than_sum_of_distances_l212_21241

/-- Represents a triangle with three unequal sides -/
structure UnequalTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  ha : 0 < a
  hb : 0 < b
  hc : 0 < c
  hab : a ≠ b
  hbc : b ≠ c
  hac : a ≠ c
  longest_side : c > max a b

/-- The height to the longest side of the triangle -/
def height_to_longest_side (t : UnequalTriangle) : ℝ := sorry

/-- Distances from a point on the longest side to the other two sides -/
def distances_to_sides (t : UnequalTriangle) : ℝ × ℝ := sorry

theorem height_less_than_sum_of_distances (t : UnequalTriangle) :
  let x := height_to_longest_side t
  let (y, z) := distances_to_sides t
  x < y + z := by sorry

end NUMINAMATH_CALUDE_height_less_than_sum_of_distances_l212_21241


namespace NUMINAMATH_CALUDE_intersection_k_value_l212_21213

-- Define the two lines
def line1 (x y k : ℝ) : Prop := 3 * x + y = k
def line2 (x y : ℝ) : Prop := -1.2 * x + y = -20

-- Define the theorem
theorem intersection_k_value :
  ∃ (k : ℝ), ∃ (y : ℝ),
    line1 7 y k ∧ line2 7 y ∧ k = 9.4 := by
  sorry

end NUMINAMATH_CALUDE_intersection_k_value_l212_21213


namespace NUMINAMATH_CALUDE_avery_donation_ratio_l212_21257

/-- Proves that the ratio of pants to shirts is 2:1 given the conditions of Avery's donation --/
theorem avery_donation_ratio :
  ∀ (pants : ℕ) (shorts : ℕ),
  let shirts := 4
  shorts = pants / 2 →
  shirts + pants + shorts = 16 →
  pants / shirts = 2 := by
sorry

end NUMINAMATH_CALUDE_avery_donation_ratio_l212_21257


namespace NUMINAMATH_CALUDE_stationery_box_sheets_l212_21253

/-- Represents a box of stationery -/
structure StationeryBox where
  sheets : ℕ
  envelopes : ℕ

/-- Represents a person's stationery usage -/
structure Usage where
  sheetsPerLetter : ℕ
  unusedSheets : ℕ
  unusedEnvelopes : ℕ

theorem stationery_box_sheets (box : StationeryBox) (john mary : Usage) : box.sheets = 240 :=
  by
  have h1 : john.sheetsPerLetter = 2 := by sorry
  have h2 : mary.sheetsPerLetter = 4 := by sorry
  have h3 : john.unusedSheets = 40 := by sorry
  have h4 : mary.unusedEnvelopes = 40 := by sorry
  have h5 : box.sheets = john.sheetsPerLetter * box.envelopes + john.unusedSheets := by sorry
  have h6 : box.sheets = mary.sheetsPerLetter * (box.envelopes - mary.unusedEnvelopes) := by sorry
  sorry

end NUMINAMATH_CALUDE_stationery_box_sheets_l212_21253


namespace NUMINAMATH_CALUDE_new_train_distance_calculation_l212_21277

/-- The distance traveled by the new train given the distance traveled by the old train and the percentage increase -/
def new_train_distance (old_distance : ℝ) (percent_increase : ℝ) : ℝ :=
  old_distance * (1 + percent_increase)

/-- Theorem: Given that a new train travels 30% farther than an old train in the same time,
    and the old train travels 300 miles, the new train travels 390 miles. -/
theorem new_train_distance_calculation :
  new_train_distance 300 0.3 = 390 := by
  sorry

#eval new_train_distance 300 0.3

end NUMINAMATH_CALUDE_new_train_distance_calculation_l212_21277


namespace NUMINAMATH_CALUDE_integral_f_equals_pi_over_2_plus_4_over_3_l212_21293

noncomputable def f (x : ℝ) : ℝ :=
  if -1 ≤ x ∧ x < 1 then Real.sqrt (1 - x^2)
  else if 1 ≤ x ∧ x ≤ 2 then x^2 - 1
  else 0

theorem integral_f_equals_pi_over_2_plus_4_over_3 :
  ∫ x in (-1)..(2), f x = π / 2 + 4 / 3 := by
  sorry

end NUMINAMATH_CALUDE_integral_f_equals_pi_over_2_plus_4_over_3_l212_21293


namespace NUMINAMATH_CALUDE_exemplary_sequences_count_l212_21238

/-- The number of distinct 6-letter sequences from "EXEMPLARY" with given conditions -/
def exemplary_sequences : ℕ :=
  let available_letters := 6  -- X, A, M, P, L, R
  let positions_to_fill := 4  -- positions 2, 3, 4, 5
  Nat.factorial available_letters / Nat.factorial (available_letters - positions_to_fill)

/-- Theorem stating the number of distinct sequences is 360 -/
theorem exemplary_sequences_count :
  exemplary_sequences = 360 := by
  sorry

#eval exemplary_sequences  -- Should output 360

end NUMINAMATH_CALUDE_exemplary_sequences_count_l212_21238


namespace NUMINAMATH_CALUDE_soccer_team_starters_l212_21299

theorem soccer_team_starters (total_players : ℕ) (quadruplets : ℕ) (starters : ℕ) (quadruplets_in_lineup : ℕ) :
  total_players = 16 →
  quadruplets = 4 →
  starters = 6 →
  quadruplets_in_lineup = 3 →
  (Nat.choose quadruplets quadruplets_in_lineup) * (Nat.choose (total_players - quadruplets) (starters - quadruplets_in_lineup)) = 880 :=
by sorry

end NUMINAMATH_CALUDE_soccer_team_starters_l212_21299


namespace NUMINAMATH_CALUDE_deepak_age_l212_21254

/-- Given the ratio of Arun's age to Deepak's age and Arun's future age, prove Deepak's current age -/
theorem deepak_age (arun_age deepak_age : ℕ) : 
  (arun_age : ℚ) / deepak_age = 2 / 5 →
  arun_age + 10 = 30 →
  deepak_age = 50 := by
sorry

end NUMINAMATH_CALUDE_deepak_age_l212_21254


namespace NUMINAMATH_CALUDE_sum_of_numbers_in_ratio_l212_21210

theorem sum_of_numbers_in_ratio (a b c : ℕ) : 
  a ≠ 0 → b ≠ 0 → c ≠ 0 →
  (a : ℚ) / b = 5 →
  (c : ℚ) / b = 4 →
  c = 400 →
  a + b + c = 1000 := by
sorry

end NUMINAMATH_CALUDE_sum_of_numbers_in_ratio_l212_21210


namespace NUMINAMATH_CALUDE_watch_cost_is_20_l212_21296

-- Define the given conditions
def evans_initial_money : ℕ := 1
def money_given_to_evan : ℕ := 12
def additional_money_needed : ℕ := 7

-- Define the cost of the watch
def watch_cost : ℕ := evans_initial_money + money_given_to_evan + additional_money_needed

-- Theorem to prove
theorem watch_cost_is_20 : watch_cost = 20 := by
  sorry

end NUMINAMATH_CALUDE_watch_cost_is_20_l212_21296


namespace NUMINAMATH_CALUDE_min_unique_score_above_90_l212_21260

/-- Represents the scoring system for the modified AHSME exam -/
def score (c w : ℕ) : ℕ := 35 + 5 * c - 2 * w

/-- Represents the total number of questions in the exam -/
def total_questions : ℕ := 35

/-- Theorem stating that 91 is the minimum score above 90 with a unique solution -/
theorem min_unique_score_above_90 :
  ∀ s : ℕ, s > 90 →
  (∃! (c w : ℕ), c + w ≤ total_questions ∧ score c w = s) →
  s ≥ 91 :=
sorry

end NUMINAMATH_CALUDE_min_unique_score_above_90_l212_21260


namespace NUMINAMATH_CALUDE_cone_lateral_surface_l212_21278

theorem cone_lateral_surface (l r : ℝ) (h : l > 0) (k : r > 0) : 
  (2 * π * r) / l = 4 * π / 3 → r / l = 2 / 3 := by
sorry

end NUMINAMATH_CALUDE_cone_lateral_surface_l212_21278


namespace NUMINAMATH_CALUDE_simplify_sqrt_sum_product_l212_21250

theorem simplify_sqrt_sum_product (m n a b : ℝ) 
  (hm : 0 < m) (hn : 0 < n) (ha : 0 < a) (hb : 0 < b) (hab : a > b)
  (hsum : a + b = m) (hprod : a * b = n) :
  Real.sqrt (m + 2 * Real.sqrt n) = Real.sqrt a + Real.sqrt b ∧
  Real.sqrt (m - 2 * Real.sqrt n) = Real.sqrt a - Real.sqrt b :=
by sorry

end NUMINAMATH_CALUDE_simplify_sqrt_sum_product_l212_21250


namespace NUMINAMATH_CALUDE_number_of_reactions_l212_21252

def visible_readings : List ℝ := [2, 2.1, 2, 2.2]

theorem number_of_reactions (x : ℝ) (h1 : (visible_readings.sum + x) / (visible_readings.length + 1) = 2) :
  visible_readings.length + 1 = 5 :=
sorry

end NUMINAMATH_CALUDE_number_of_reactions_l212_21252


namespace NUMINAMATH_CALUDE_dinner_seating_arrangements_l212_21215

/-- The number of ways to choose and seat people at a circular table. -/
def circular_seating_arrangements (total_people : ℕ) (seats : ℕ) : ℕ :=
  total_people * (seats - 1).factorial

/-- The problem statement -/
theorem dinner_seating_arrangements :
  circular_seating_arrangements 8 7 = 5760 := by
  sorry

end NUMINAMATH_CALUDE_dinner_seating_arrangements_l212_21215


namespace NUMINAMATH_CALUDE_complex_number_problem_l212_21266

theorem complex_number_problem (α β : ℂ) : 
  (∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ α + β = x ∧ Complex.I * (α - 3 * β) = y) →
  β = 4 + Complex.I →
  α = 12 - Complex.I := by
sorry

end NUMINAMATH_CALUDE_complex_number_problem_l212_21266


namespace NUMINAMATH_CALUDE_probability_of_red_ball_l212_21249

def total_balls : ℕ := 10
def yellow_balls : ℕ := 1
def green_balls : ℕ := 3

def red_balls : ℕ := total_balls - yellow_balls - green_balls

def probability_red_ball : ℚ := red_balls / total_balls

theorem probability_of_red_ball :
  probability_red_ball = 3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_probability_of_red_ball_l212_21249


namespace NUMINAMATH_CALUDE_binomial_coefficient_ratio_l212_21214

theorem binomial_coefficient_ratio (n k : ℕ) : 
  (Nat.choose n k : ℚ) / (Nat.choose n (k+1) : ℚ) = 1/2 ∧ 
  (Nat.choose n (k+1) : ℚ) / (Nat.choose n (k+2) : ℚ) = 2/3 → 
  n + k = 18 :=
by sorry

end NUMINAMATH_CALUDE_binomial_coefficient_ratio_l212_21214


namespace NUMINAMATH_CALUDE_problem_statement_l212_21204

theorem problem_statement (x y z : ℝ) (h1 : x + y = 5) (h2 : z^2 = x*y + y - 9) :
  x + 2*y + 3*z = 8 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l212_21204


namespace NUMINAMATH_CALUDE_jellybeans_per_child_l212_21217

theorem jellybeans_per_child 
  (initial_jellybeans : ℕ) 
  (normal_class_size : ℕ) 
  (absent_children : ℕ) 
  (remaining_jellybeans : ℕ) 
  (h1 : initial_jellybeans = 100)
  (h2 : normal_class_size = 24)
  (h3 : absent_children = 2)
  (h4 : remaining_jellybeans = 34)
  : (initial_jellybeans - remaining_jellybeans) / (normal_class_size - absent_children) = 3 := by
  sorry

end NUMINAMATH_CALUDE_jellybeans_per_child_l212_21217


namespace NUMINAMATH_CALUDE_cartons_in_load_l212_21245

/-- Represents the weight of vegetables in a store's delivery truck. -/
structure VegetableLoad where
  crate_weight : ℕ
  carton_weight : ℕ
  num_crates : ℕ
  total_weight : ℕ

/-- Calculates the number of cartons in a vegetable load. -/
def num_cartons (load : VegetableLoad) : ℕ :=
  (load.total_weight - load.crate_weight * load.num_crates) / load.carton_weight

/-- Theorem stating that the number of cartons in the specific load is 16. -/
theorem cartons_in_load : 
  let load : VegetableLoad := {
    crate_weight := 4,
    carton_weight := 3,
    num_crates := 12,
    total_weight := 96
  }
  num_cartons load = 16 := by
  sorry


end NUMINAMATH_CALUDE_cartons_in_load_l212_21245


namespace NUMINAMATH_CALUDE_solution_set_for_f_squared_minimum_value_of_g_l212_21248

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - a|

-- Define the function g
def g (a : ℝ) (x : ℝ) : ℝ := f a (2*x + a) + 2 * f a x

-- Part 1
theorem solution_set_for_f_squared (x : ℝ) :
  f 1 x ^ 2 ≤ 2 ↔ 1 - Real.sqrt 2 ≤ x ∧ x ≤ 1 + Real.sqrt 2 :=
sorry

-- Part 2
theorem minimum_value_of_g (a : ℝ) (h : a > 0) :
  (∃ (m : ℝ), m = 4 ∧ ∀ (x : ℝ), g a x ≥ m) → a = 2 :=
sorry

end NUMINAMATH_CALUDE_solution_set_for_f_squared_minimum_value_of_g_l212_21248


namespace NUMINAMATH_CALUDE_binomial_fraction_is_nat_smallest_k_binomial_fraction_is_nat_smallest_k_property_l212_21202

/-- For any natural number m, 1/(m+1) * binomial(2m, m) is a natural number -/
theorem binomial_fraction_is_nat (m : ℕ) : ∃ (k : ℕ), k = (1 : ℚ) / (m + 1 : ℚ) * (Nat.choose (2 * m) m) := by sorry

/-- For any natural numbers m and n where n ≥ m, 
    (2m+1)/(n+m+1) * binomial(2n, n+m) is a natural number -/
theorem smallest_k_binomial_fraction_is_nat (m n : ℕ) (h : n ≥ m) : 
  ∃ (k : ℕ), k = ((2 * m + 1 : ℚ) / (n + m + 1 : ℚ)) * (Nat.choose (2 * n) (n + m)) := by sorry

/-- 2m+1 is the smallest natural number k such that 
    k/(n+m+1) * binomial(2n, n+m) is a natural number for all n ≥ m -/
theorem smallest_k_property (m : ℕ) : 
  ∀ (k : ℕ), (∀ (n : ℕ), n ≥ m → ∃ (j : ℕ), j = (k : ℚ) / (n + m + 1 : ℚ) * (Nat.choose (2 * n) (n + m))) 
  → k ≥ 2 * m + 1 := by sorry

end NUMINAMATH_CALUDE_binomial_fraction_is_nat_smallest_k_binomial_fraction_is_nat_smallest_k_property_l212_21202


namespace NUMINAMATH_CALUDE_sequence_product_l212_21267

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)

/-- A geometric sequence -/
def geometric_sequence (b : ℕ → ℝ) : Prop :=
  ∀ n, b (n + 1) / b n = b (n + 2) / b (n + 1)

/-- The main theorem -/
theorem sequence_product (a : ℕ → ℝ) (b : ℕ → ℝ) :
  arithmetic_sequence a →
  a 1 + a 11 = 8 →
  geometric_sequence b →
  b 7 = a 7 →
  b 6 * b 8 = 16 := by
sorry

end NUMINAMATH_CALUDE_sequence_product_l212_21267


namespace NUMINAMATH_CALUDE_inequality_proof_l212_21218

theorem inequality_proof (x y z : ℝ) 
  (h1 : x ≥ 0) (h2 : y ≥ 0) (h3 : z ≥ 0) 
  (h4 : x + y + z = 1) : 
  2 * (x^2 + y^2 + z^2) + 9 * x * y * z ≥ 1 ∧ 
  x * y + y * z + z * x - 3 * x * y * z ≤ 1/4 := by
  sorry

#check inequality_proof

end NUMINAMATH_CALUDE_inequality_proof_l212_21218


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l212_21258

def A : Set ℝ := {-1, 1, 2, 4}
def B : Set ℝ := {x : ℝ | |x - 1| ≤ 1}

theorem intersection_of_A_and_B : A ∩ B = {1, 2} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l212_21258


namespace NUMINAMATH_CALUDE_floor_ceil_sum_equation_l212_21219

theorem floor_ceil_sum_equation : ∃ (r s : ℝ), 
  (Int.floor r : ℝ) + r + (Int.ceil s : ℝ) = 10.7 ∧ r = 4.7 ∧ s = 2 := by
  sorry

end NUMINAMATH_CALUDE_floor_ceil_sum_equation_l212_21219


namespace NUMINAMATH_CALUDE_circle_line_intersection_l212_21221

theorem circle_line_intersection :
  ∃ (x y : ℝ), (x - 1)^2 + (y - 1)^2 = 2 ∧ x + y = 1 ∧
  ¬(∃ (x y : ℝ), (x - 1)^2 + (y - 1)^2 = 2 ∧ x + y = 1 ∧ x = 1 ∧ y = 1) :=
by sorry

end NUMINAMATH_CALUDE_circle_line_intersection_l212_21221


namespace NUMINAMATH_CALUDE_quadratic_complex_root_l212_21261

/-- Given a quadratic equation x^2 + px + q = 0 with real coefficients,
    if 1 + i is a root, then q = 2. -/
theorem quadratic_complex_root (p q : ℝ) : 
  (∀ x : ℂ, x^2 + p * x + q = 0 ↔ x = (1 + I) ∨ x = (1 - I)) → q = 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_complex_root_l212_21261


namespace NUMINAMATH_CALUDE_total_pages_purchased_l212_21242

def total_budget : ℚ := 10
def cost_per_notepad : ℚ := 5/4
def pages_per_notepad : ℕ := 60

theorem total_pages_purchased :
  (total_budget / cost_per_notepad).floor * pages_per_notepad = 480 :=
by sorry

end NUMINAMATH_CALUDE_total_pages_purchased_l212_21242


namespace NUMINAMATH_CALUDE_t_upper_bound_F_positive_l212_21285

-- Define the functions
noncomputable def f (x : ℝ) : ℝ := Real.log x
noncomputable def g (t : ℝ) (x : ℝ) : ℝ := t / x - f x
noncomputable def F (x : ℝ) : ℝ := f x - 1 / Real.exp x + 2 / (Real.exp 1 * x)

-- Theorem 1
theorem t_upper_bound (t : ℝ) :
  (∀ x > 0, g t x ≤ f x) → t ≤ -2 / Real.exp 1 :=
sorry

-- Theorem 2
theorem F_positive (x : ℝ) :
  x > 0 → F x > 0 :=
sorry

end NUMINAMATH_CALUDE_t_upper_bound_F_positive_l212_21285


namespace NUMINAMATH_CALUDE_nth_equation_holds_l212_21225

theorem nth_equation_holds (n : ℕ) (hn : n > 0) : 
  (4 * n^2 : ℚ) / (2 * n - 1) - (2 * n + 1) = 1 - ((2 * n - 2) : ℚ) / (2 * n - 1) := by
  sorry

end NUMINAMATH_CALUDE_nth_equation_holds_l212_21225


namespace NUMINAMATH_CALUDE_max_value_trigonometric_function_l212_21246

open Real

theorem max_value_trigonometric_function :
  ∃ (M : ℝ), M = 3 - 2 * sqrt 2 ∧
  ∀ θ : ℝ, 0 < θ → θ < π / 2 →
  (1 / sin θ - 1) * (1 / cos θ - 1) ≤ M :=
sorry

end NUMINAMATH_CALUDE_max_value_trigonometric_function_l212_21246


namespace NUMINAMATH_CALUDE_associated_functions_range_l212_21236

/-- Two functions are associated on an interval if their difference has two distinct zeros in that interval. -/
def associated_functions (f g : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, a ≤ x₁ ∧ x₁ < x₂ ∧ x₂ ≤ b ∧ f x₁ = g x₁ ∧ f x₂ = g x₂

/-- The statement of the problem. -/
theorem associated_functions_range (m : ℝ) : 
  let f : ℝ → ℝ := λ x ↦ x^2 - 3*x + 4
  let g : ℝ → ℝ := λ x ↦ 2*x + m
  associated_functions f g 0 3 → -9/4 < m ∧ m ≤ -2 :=
by sorry

end NUMINAMATH_CALUDE_associated_functions_range_l212_21236


namespace NUMINAMATH_CALUDE_quadratic_roots_cube_l212_21273

theorem quadratic_roots_cube (A B C : ℝ) (r s : ℝ) (h1 : A ≠ 0) :
  A * r^2 + B * r + C = 0 →
  A * s^2 + B * s + C = 0 →
  r ≠ s →
  ∃ q, (r^3)^2 + ((B^3 - 3*A*B*C) / A^3) * r^3 + q = 0 ∧
       (s^3)^2 + ((B^3 - 3*A*B*C) / A^3) * s^3 + q = 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_cube_l212_21273


namespace NUMINAMATH_CALUDE_smallest_cube_ending_580_l212_21259

theorem smallest_cube_ending_580 : 
  ∃ n : ℕ, n > 0 ∧ n^3 % 1000 = 580 ∧ ∀ m : ℕ, m > 0 ∧ m^3 % 1000 = 580 → m ≥ n :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_smallest_cube_ending_580_l212_21259


namespace NUMINAMATH_CALUDE_min_value_sum_of_reciprocals_l212_21200

theorem min_value_sum_of_reciprocals (a b c : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) 
  (sum_eq_3 : a + b + c = 3) :
  1 / (a + b)^2 + 1 / (a + c)^2 + 1 / (b + c)^2 ≥ 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_sum_of_reciprocals_l212_21200


namespace NUMINAMATH_CALUDE_divisors_of_cube_l212_21287

theorem divisors_of_cube (n : ℕ) : 
  (∃ (d : Finset ℕ), d = {x : ℕ | x ∣ n} ∧ d.card = 5) →
  (∃ (d : Finset ℕ), d = {x : ℕ | x ∣ n^3} ∧ (d.card = 13 ∨ d.card = 16)) :=
by sorry

end NUMINAMATH_CALUDE_divisors_of_cube_l212_21287


namespace NUMINAMATH_CALUDE_uncovered_area_square_in_square_l212_21233

theorem uncovered_area_square_in_square (large_side : ℝ) (small_side : ℝ) :
  large_side = 10 →
  small_side = 4 →
  large_side ^ 2 - small_side ^ 2 = 84 := by
  sorry

end NUMINAMATH_CALUDE_uncovered_area_square_in_square_l212_21233


namespace NUMINAMATH_CALUDE_prob_first_qualified_on_third_test_l212_21271

/-- The probability of obtaining the first qualified product on the third test. -/
def P_epsilon_3 (pass_rate : ℝ) (fail_rate : ℝ) : ℝ :=
  fail_rate^2 * pass_rate

/-- The theorem stating that P(ε = 3) is equal to (1/4)² × (3/4) given the specified pass and fail rates. -/
theorem prob_first_qualified_on_third_test :
  let pass_rate : ℝ := 3/4
  let fail_rate : ℝ := 1/4
  P_epsilon_3 pass_rate fail_rate = (1/4)^2 * (3/4) :=
by sorry

end NUMINAMATH_CALUDE_prob_first_qualified_on_third_test_l212_21271


namespace NUMINAMATH_CALUDE_star_to_maltese_cross_l212_21256

/-- Represents a four-pointed star -/
structure FourPointedStar :=
  (vertices : Fin 4 → ℝ × ℝ)

/-- Represents a frame -/
structure Frame :=
  (corners : Fin 4 → ℝ × ℝ)

/-- Represents a part of the cut star -/
structure StarPart :=
  (vertices : Fin 3 → ℝ × ℝ)

/-- Represents a Maltese cross -/
structure MalteseCross :=
  (vertices : Fin 8 → ℝ × ℝ)

/-- Function to cut a FourPointedStar into 4 StarParts -/
def cutStar (star : FourPointedStar) : Fin 4 → StarPart :=
  sorry

/-- Function to arrange StarParts in a Frame -/
def arrangeParts (parts : Fin 4 → StarPart) (frame : Frame) : MalteseCross :=
  sorry

/-- Theorem stating that a FourPointedStar can be cut and arranged to form a MalteseCross -/
theorem star_to_maltese_cross (star : FourPointedStar) (frame : Frame) :
  ∃ (arrangement : MalteseCross), arrangement = arrangeParts (cutStar star) frame :=
sorry

end NUMINAMATH_CALUDE_star_to_maltese_cross_l212_21256
