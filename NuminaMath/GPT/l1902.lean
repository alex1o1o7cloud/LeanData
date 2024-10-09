import Mathlib

namespace largest_lucky_number_l1902_190298

theorem largest_lucky_number : 
  let a := 1
  let b := 4
  let lucky_number (x y : ℕ) := x + y + x * y
  let c1 := lucky_number a b
  let c2 := lucky_number b c1
  let c3 := lucky_number c1 c2
  c3 = 499 :=
by
  sorry

end largest_lucky_number_l1902_190298


namespace problem1_problem2_problem3_problem4_l1902_190263

theorem problem1 : 24 - (-16) + (-25) - 32 = -17 := by
  sorry

theorem problem2 : (-1 / 2) * 2 / 2 * (-1 / 2) = 1 / 4 := by
  sorry

theorem problem3 : -2^2 * 5 - (-2)^3 * (1 / 8) + 1 = -18 := by
  sorry

theorem problem4 : ((-1 / 4) - (5 / 6) + (8 / 9)) / (-1 / 6)^2 + (-2)^2 * (-6)= -31 := by
  sorry

end problem1_problem2_problem3_problem4_l1902_190263


namespace Tim_Linda_Mow_Lawn_l1902_190200

theorem Tim_Linda_Mow_Lawn :
  let tim_time := 1.5
  let linda_time := 2
  let tim_rate := 1 / tim_time
  let linda_rate := 1 / linda_time
  let combined_rate := tim_rate + linda_rate
  let combined_time_hours := 1 / combined_rate
  let combined_time_minutes := combined_time_hours * 60
  combined_time_minutes = 51.43 := 
by
    sorry

end Tim_Linda_Mow_Lawn_l1902_190200


namespace consecutive_numbers_probability_l1902_190261

theorem consecutive_numbers_probability :
  let total_ways := Nat.choose 20 5
  let non_consecutive_ways := Nat.choose 16 5
  let probability_of_non_consecutive := (non_consecutive_ways : ℚ) / (total_ways : ℚ)
  let probability_of_consecutive := 1 - probability_of_non_consecutive
  probability_of_consecutive = 232 / 323 :=
by
  sorry

end consecutive_numbers_probability_l1902_190261


namespace total_hours_worked_l1902_190281

-- Definitions based on the conditions
def hours_per_day : ℕ := 3
def days_worked : ℕ := 6

-- Statement of the problem
theorem total_hours_worked : hours_per_day * days_worked = 18 := by
  sorry

end total_hours_worked_l1902_190281


namespace game_points_product_l1902_190240

def g (n : ℕ) : ℕ :=
  if n % 3 = 0 then 12
  else if n % 2 = 0 then 3
  else 0

def allie_rolls : List ℕ := [5, 4, 1, 2, 6]
def betty_rolls : List ℕ := [6, 3, 3, 2, 1]

def calculate_points (rolls : List ℕ) : ℕ :=
  rolls.map g |>.sum

theorem game_points_product :
  calculate_points allie_rolls * calculate_points betty_rolls = 702 :=
by
  sorry

end game_points_product_l1902_190240


namespace range_of_xy_l1902_190275

theorem range_of_xy {x y : ℝ} (h₁ : 0 < x) (h₂ : 0 < y)
    (h₃ : x + 2/x + 3*y + 4/y = 10) : 
    1 ≤ x * y ∧ x * y ≤ 8 / 3 :=
by
  sorry

end range_of_xy_l1902_190275


namespace direct_proportion_b_zero_l1902_190267

theorem direct_proportion_b_zero (b : ℝ) (x y : ℝ) 
  (h : ∀ x, y = x + b → ∃ k, y = k * x) : b = 0 :=
sorry

end direct_proportion_b_zero_l1902_190267


namespace find_vector_at_t4_l1902_190249

def vector_at (t : ℝ) (a d : ℝ × ℝ × ℝ) : ℝ × ℝ × ℝ :=
  let (x, y, z) := a
  let (dx, dy, dz) := d
  (x + t * dx, y + t * dy, z + t * dz)

theorem find_vector_at_t4 :
  ∀ (a d : ℝ × ℝ × ℝ),
    vector_at (-2) a d = (2, 6, 16) →
    vector_at 1 a d = (-1, -5, -10) →
    vector_at 4 a d = (-16, -60, -140) :=
by
  intros a d h1 h2
  sorry

end find_vector_at_t4_l1902_190249


namespace jimmy_more_sheets_than_tommy_l1902_190218

theorem jimmy_more_sheets_than_tommy 
  (jimmy_initial_sheets : ℕ)
  (tommy_initial_sheets : ℕ)
  (additional_sheets : ℕ)
  (h1 : tommy_initial_sheets = jimmy_initial_sheets + 25)
  (h2 : jimmy_initial_sheets = 58)
  (h3 : additional_sheets = 85) :
  (jimmy_initial_sheets + additional_sheets) - tommy_initial_sheets = 60 := 
by
  sorry

end jimmy_more_sheets_than_tommy_l1902_190218


namespace Riley_fewer_pairs_l1902_190232

-- Define the conditions
def Ellie_pairs : ℕ := 8
def Total_pairs : ℕ := 13

-- Prove the statement
theorem Riley_fewer_pairs : (Total_pairs - Ellie_pairs) - Ellie_pairs = 3 :=
by
  -- Skip the proof
  sorry

end Riley_fewer_pairs_l1902_190232


namespace calculation_l1902_190277

theorem calculation :
  (-1:ℤ)^(2022) + (Real.sqrt 9) - 2 * (Real.sin (Real.pi / 6)) = 3 := by
  -- According to the mathematical problem and the given solution.
  -- Here we use essential definitions and facts provided in the problem.
  sorry

end calculation_l1902_190277


namespace homework_problems_left_l1902_190238

def math_problems : ℕ := 43
def science_problems : ℕ := 12
def finished_problems : ℕ := 44

theorem homework_problems_left :
  (math_problems + science_problems - finished_problems) = 11 :=
by
  sorry

end homework_problems_left_l1902_190238


namespace find_m_and_max_profit_l1902_190256

theorem find_m_and_max_profit (m : ℝ) (y : ℝ) (x : ℝ) (ln : ℝ → ℝ) 
    (h1 : y = m * ln x - 1 / 100 * x ^ 2 + 101 / 50 * x + ln 10)
    (h2 : 10 < x) 
    (h3 : y = 35.7) 
    (h4 : x = 20)
    (ln_2 : ln 2 = 0.7) 
    (ln_5 : ln 5 = 1.6) :
    m = -1 ∧ ∃ x, (x = 50 ∧ (-ln x - 1 / 100 * x ^ 2 + 51 / 50 * x + ln 10 - x) = 24.4) := by
  sorry

end find_m_and_max_profit_l1902_190256


namespace smallest_integer_relative_prime_to_2310_l1902_190219

theorem smallest_integer_relative_prime_to_2310 (n : ℕ) : (2 < n → n ≤ 13 → ¬ (n ∣ 2310)) → n = 13 := by
  sorry

end smallest_integer_relative_prime_to_2310_l1902_190219


namespace sum_of_four_digit_multiples_of_5_l1902_190254

theorem sum_of_four_digit_multiples_of_5 :
  let a := 1000
  let l := 9995
  let d := 5
  let n := ((l - a) / d) + 1
  let S := n * (a + l) / 2
  S = 9895500 :=
by
  let a := 1000
  let l := 9995
  let d := 5
  let n := ((l - a) / d) + 1
  let S := n * (a + l) / 2
  sorry

end sum_of_four_digit_multiples_of_5_l1902_190254


namespace largest_prime_divisor_of_1202102_5_l1902_190242

def base_5_to_decimal (n : String) : ℕ := 
  let digits := n.toList.map (λ c => c.toNat - '0'.toNat)
  digits.foldr (λ (digit acc : ℕ) => acc * 5 + digit) 0

def largest_prime_factor (n : ℕ) : ℕ := sorry -- Placeholder for the actual factorization logic.

theorem largest_prime_divisor_of_1202102_5 : 
  largest_prime_factor (base_5_to_decimal "1202102") = 307 := 
sorry

end largest_prime_divisor_of_1202102_5_l1902_190242


namespace probability_calculation_l1902_190278

noncomputable def probability_of_event_A : ℚ := 
  let total_ways := 35 
  let favorable_ways := 6 
  favorable_ways / total_ways

theorem probability_calculation (A_team B_team : Type) [Fintype A_team] [Fintype B_team] [DecidableEq A_team] [DecidableEq B_team] :
  let total_players := 7 
  let selected_players := 4 
  let seeded_A := 2 
  let nonseeded_A := 1 
  let seeded_B := 2 
  let nonseeded_B := 2 
  let event_total_ways := Nat.choose total_players selected_players 
  let event_A_ways := Nat.choose seeded_A 2 * Nat.choose nonseeded_A 2 + Nat.choose seeded_B 2 * Nat.choose nonseeded_B 2 
  probability_of_event_A = 6 / 35 := 
sorry

end probability_calculation_l1902_190278


namespace bagel_pieces_after_10_cuts_l1902_190250

def bagel_pieces_after_cuts (initial_pieces : ℕ) (cuts : ℕ) : ℕ :=
  initial_pieces + cuts

theorem bagel_pieces_after_10_cuts : bagel_pieces_after_cuts 1 10 = 11 := by
  sorry

end bagel_pieces_after_10_cuts_l1902_190250


namespace number_of_rocks_tossed_l1902_190221

-- Conditions
def pebbles : ℕ := 6
def rocks : ℕ := 3
def boulders : ℕ := 2
def pebble_splash : ℚ := 1 / 4
def rock_splash : ℚ := 1 / 2
def boulder_splash : ℚ := 2

-- Total width of the splashes
def total_splash (R : ℕ) : ℚ := 
  pebbles * pebble_splash + R * rock_splash + boulders * boulder_splash

-- Given condition
def total_splash_condition : ℚ := 7

theorem number_of_rocks_tossed : 
  total_splash rocks = total_splash_condition → rocks = 3 :=
by
  intro h
  sorry

end number_of_rocks_tossed_l1902_190221


namespace tens_digit_2023_pow_2024_minus_2025_l1902_190273

theorem tens_digit_2023_pow_2024_minus_2025 :
  (2023 ^ 2024 - 2025) % 100 / 10 % 10 = 5 :=
sorry

end tens_digit_2023_pow_2024_minus_2025_l1902_190273


namespace find_vertex_parabola_l1902_190293

-- Define the quadratic equation of the parabola
def parabola_eq (x y : ℝ) : Prop := x^2 - 4 * x + 3 * y + 10 = 0

-- Definition of the vertex of the parabola
def is_vertex (v : ℝ × ℝ) : Prop :=
  ∀ (x y : ℝ), parabola_eq x y → v = (2, -2)

-- The main statement we want to prove
theorem find_vertex_parabola : 
  ∃ v : ℝ × ℝ, is_vertex v :=
by
  use (2, -2)
  intros x y hyp
  sorry

end find_vertex_parabola_l1902_190293


namespace exist_positive_integers_summing_to_one_l1902_190271

theorem exist_positive_integers_summing_to_one :
  ∃ (x y z : ℕ), 0 < x ∧ 0 < y ∧ 0 < z ∧ (1 / (x:ℚ) + 1 / (y:ℚ) + 1 / (z:ℚ) = 1)
    ∧ ((x = 2 ∧ y = 4 ∧ z = 4) ∨ (x = 2 ∧ y = 3 ∧ z = 6) ∨ (x = 3 ∧ y = 3 ∧ z = 3)) :=
by
  sorry

end exist_positive_integers_summing_to_one_l1902_190271


namespace probability_two_even_balls_l1902_190225

theorem probability_two_even_balls
  (total_balls : ℕ)
  (even_balls : ℕ)
  (h_total : total_balls = 16)
  (h_even : even_balls = 8)
  (first_draw : ℕ → ℚ)
  (second_draw : ℕ → ℚ)
  (h_first : first_draw even_balls = even_balls / total_balls)
  (h_second : second_draw (even_balls - 1) = (even_balls - 1) / (total_balls - 1)) :
  (first_draw even_balls) * (second_draw (even_balls - 1)) = 7 / 30 := 
sorry

end probability_two_even_balls_l1902_190225


namespace select_integers_divisible_l1902_190210

theorem select_integers_divisible (k : ℕ) (s : Finset ℤ) (h₁ : s.card = 2 * 2^k - 1) :
  ∃ t : Finset ℤ, t ⊆ s ∧ t.card = 2^k ∧ (t.sum id) % 2^k = 0 :=
sorry

end select_integers_divisible_l1902_190210


namespace jamshid_taimour_painting_problem_l1902_190247

/-- Jamshid and Taimour Painting Problem -/
theorem jamshid_taimour_painting_problem (T : ℝ) (h1 : T > 0)
  (h2 : 1 / T + 2 / T = 1 / 5) : T = 15 :=
by
  -- solving the theorem
  sorry

end jamshid_taimour_painting_problem_l1902_190247


namespace calculate_expression_l1902_190203

theorem calculate_expression (x : ℝ) (h : x = 3) : (x^2 - 5 * x + 4) / (x - 4) = 2 :=
by
  rw [h]
  sorry

end calculate_expression_l1902_190203


namespace find_higher_percentage_l1902_190257

-- Definitions based on conditions
def principal : ℕ := 8400
def time : ℕ := 2
def rate_0 : ℕ := 10
def delta_interest : ℕ := 840

-- The proof statement
theorem find_higher_percentage (r : ℕ) :
  (principal * rate_0 * time / 100 + delta_interest = principal * r * time / 100) →
  r = 15 :=
by sorry

end find_higher_percentage_l1902_190257


namespace tax_diminished_percentage_l1902_190299

theorem tax_diminished_percentage (T C : ℝ) (hT : T > 0) (hC : C > 0) (X : ℝ) 
  (h : T * (1 - X / 100) * C * 1.15 = T * C * 0.9315) : X = 19 :=
by 
  sorry

end tax_diminished_percentage_l1902_190299


namespace frac_x_y_eq_neg2_l1902_190288

open Real

theorem frac_x_y_eq_neg2 (x y : ℝ) (h1 : 1 < (x - y) / (x + y)) (h2 : (x - y) / (x + y) < 4) (h3 : (x + y) / (x - y) ≠ 1) :
  ∃ t : ℤ, (x / y = t) ∧ (t = -2) :=
by sorry

end frac_x_y_eq_neg2_l1902_190288


namespace greatest_value_l1902_190259

theorem greatest_value (x : ℝ) : -x^2 + 9 * x - 18 ≥ 0 → x ≤ 6 :=
by
  sorry

end greatest_value_l1902_190259


namespace arithmetic_progression_conditions_l1902_190206

theorem arithmetic_progression_conditions (a d : ℝ) :
  let x := a
  let y := a + d
  let z := a + 2 * d
  (y^2 = (x^2 * z^2)^(1/2)) ↔ (d = 0 ∨ d = a * (-2 + Real.sqrt 2) ∨ d = a * (-2 - Real.sqrt 2)) :=
by
  intros
  sorry

end arithmetic_progression_conditions_l1902_190206


namespace prime_square_implies_equal_l1902_190287

theorem prime_square_implies_equal (p : ℕ) (hp : Nat.Prime p) (hp_gt_2 : p > 2)
  (x y : ℕ) (hx : 1 ≤ x ∧ x ≤ (p-1)/2) (hy : 1 ≤ y ∧ y ≤ (p-1)/2)
  (h_square: ∃ k : ℕ, x * (p - x) * y * (p - y) = k ^ 2) : x = y :=
sorry

end prime_square_implies_equal_l1902_190287


namespace largest_number_is_40_l1902_190228

theorem largest_number_is_40 
    (a b c : ℕ) 
    (h1 : a ≠ b)
    (h2 : b ≠ c)
    (h3 : a ≠ c)
    (h4 : a + b + c = 100)
    (h5 : c - b = 8)
    (h6 : b - a = 4) : c = 40 :=
sorry

end largest_number_is_40_l1902_190228


namespace inequality_abc_l1902_190248

theorem inequality_abc (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : 
  (a + b - c) * (b + c - a) * (c + a - b) ≤ a * b * c := 
sorry

end inequality_abc_l1902_190248


namespace sum_of_first_9_terms_45_l1902_190284

-- Define the arithmetic sequence and sum of terms in the sequence
def S (n : ℕ) : ℕ := sorry  -- Placeholder for the sum of the first n terms of the sequence
def a (n : ℕ) : ℕ := sorry  -- Placeholder for the n-th term of the sequence

-- Given conditions
axiom condition1 : a 3 + a 5 + a 7 = 15

-- Proof goal
theorem sum_of_first_9_terms_45 : S 9 = 45 :=
by
  sorry

end sum_of_first_9_terms_45_l1902_190284


namespace min_value_frac_x_y_l1902_190290

theorem min_value_frac_x_y (x y : ℝ) (hx : x > 0) (hy : y > -1) (hxy : x + y = 1) :
  ∃ m, m = 2 + Real.sqrt 3 ∧ ∀ x y, x > 0 → y > -1 → x + y = 1 → (x^2 + 3) / x + y^2 / (y + 1) ≥ m :=
sorry

end min_value_frac_x_y_l1902_190290


namespace milk_transfer_equal_l1902_190227

theorem milk_transfer_equal (A B C x : ℕ) (hA : A = 1200) (hB : B = A - 750) (hC : C = A - B) (h_eq : B + x = C - x) :
  x = 150 :=
by
  sorry

end milk_transfer_equal_l1902_190227


namespace convert_seven_cubic_yards_l1902_190269

-- Define the conversion factor from yards to feet
def yardToFeet : ℝ := 3
-- Define the conversion factor from cubic yards to cubic feet
def cubicYardToCubicFeet : ℝ := yardToFeet ^ 3
-- Define the conversion function from cubic yards to cubic feet
noncomputable def convertVolume (volumeInCubicYards : ℝ) : ℝ :=
  volumeInCubicYards * cubicYardToCubicFeet

-- Statement to prove: 7 cubic yards is equivalent to 189 cubic feet
theorem convert_seven_cubic_yards : convertVolume 7 = 189 := by
  sorry

end convert_seven_cubic_yards_l1902_190269


namespace detergent_per_pound_l1902_190236

-- Define the conditions
def total_ounces_detergent := 18
def total_pounds_clothes := 9

-- Define the question to prove the amount of detergent per pound of clothes
theorem detergent_per_pound : total_ounces_detergent / total_pounds_clothes = 2 := by
  sorry

end detergent_per_pound_l1902_190236


namespace algebraic_expr_value_at_neg_one_l1902_190213

-- Define the expression "3 times the square of x minus 5"
def algebraic_expr (x : ℝ) : ℝ := 3 * x^2 + 5

-- Theorem to state the value when x = -1 is 8
theorem algebraic_expr_value_at_neg_one : algebraic_expr (-1) = 8 := 
by
  -- The steps to prove are skipped with 'sorry'
  sorry

end algebraic_expr_value_at_neg_one_l1902_190213


namespace remainder_equal_to_zero_l1902_190283

def A : ℕ := 270
def B : ℕ := 180
def M : ℕ := 25
def R_A : ℕ := A % M
def R_B : ℕ := B % M
def A_squared_B : ℕ := (A ^ 2 * B) % M
def R_A_R_B : ℕ := (R_A * R_B) % M

theorem remainder_equal_to_zero (h1 : A = 270) (h2 : B = 180) (h3 : M = 25) 
    (h4 : R_A = 20) (h5 : R_B = 5) : 
    A_squared_B = 0 ∧ R_A_R_B = 0 := 
by {
    sorry
}

end remainder_equal_to_zero_l1902_190283


namespace abs_value_sum_l1902_190223

noncomputable def sin_theta_in_bounds (θ : ℝ) : Prop :=
  -1 ≤ Real.sin θ ∧ Real.sin θ ≤ 1

noncomputable def x_satisfies_log_eq (θ x : ℝ) : Prop :=
  Real.log x / Real.log 3 = 1 + Real.sin θ

theorem abs_value_sum (θ x : ℝ) (h1 : x_satisfies_log_eq θ x) (h2 : sin_theta_in_bounds θ) :
  |x - 1| + |x - 9| = 8 :=
sorry

end abs_value_sum_l1902_190223


namespace leakage_empty_time_l1902_190220

variables (a : ℝ) (h1 : a > 0) -- Assuming a is positive for the purposes of the problem

theorem leakage_empty_time (h : 7 * a > 0) : (7 * a) / 6 = 7 * a / 6 :=
by
  sorry

end leakage_empty_time_l1902_190220


namespace solve_x_l1902_190294

theorem solve_x (x : ℝ) (h : (x - 1)^2 = 4) : x = 3 ∨ x = -1 :=
by
  -- proof goes here
  sorry

end solve_x_l1902_190294


namespace inches_per_foot_l1902_190226

-- Definition of the conditions in the problem.
def feet_last_week := 6
def feet_less_this_week := 4
def total_inches := 96

-- Lean statement that proves the number of inches in a foot
theorem inches_per_foot : 
    (total_inches / (feet_last_week + (feet_last_week - feet_less_this_week))) = 12 := 
by sorry

end inches_per_foot_l1902_190226


namespace fraction_to_decimal_l1902_190262

theorem fraction_to_decimal :
  (11:ℚ) / 16 = 0.6875 :=
by
  sorry

end fraction_to_decimal_l1902_190262


namespace roque_commute_time_l1902_190292

theorem roque_commute_time :
  let walk_time := 2
  let bike_time := 1
  let walks_per_week := 3
  let bike_rides_per_week := 2
  let total_walk_time := 2 * walks_per_week * walk_time
  let total_bike_time := 2 * bike_rides_per_week * bike_time
  total_walk_time + total_bike_time = 16 :=
by sorry

end roque_commute_time_l1902_190292


namespace proof_problem_l1902_190282

noncomputable def vector_a (x : ℝ) : ℝ × ℝ := 
  (Real.sin (2 * x), 2 * Real.cos x ^ 2 - 1)

noncomputable def vector_b (θ : ℝ) : ℝ × ℝ := 
  (Real.sin θ, Real.cos θ)

noncomputable def f (x θ : ℝ) : ℝ := 
  (vector_a x).1 * (vector_b θ).1 + (vector_a x).2 * (vector_b θ).2

theorem proof_problem 
  (θ : ℝ) 
  (hθ : 0 < θ ∧ θ < π) 
  (h1 : f (π / 6) θ = 1) 
  (x : ℝ) 
  (hx : -π / 6 ≤ x ∧ x ≤ π / 4) :
  θ = π / 3 ∧
  (∀ x, f x θ = f (x + π) θ) ∧
  (∀ x, -π / 6 ≤ x ∧ x ≤ π / 4 → f x θ ≤ 1) ∧
  (∀ x, -π / 6 ≤ x ∧ x ≤ π / 4 → f x θ ≥ -0.5) :=
by
  sorry

end proof_problem_l1902_190282


namespace problem_statement_l1902_190289

noncomputable def is_integer (x : ℚ) : Prop := ∃ (n : ℤ), x = n

theorem problem_statement (m n p q : ℕ) (h₁ : m ≠ p) (h₂ : is_integer ((mn + pq : ℚ) / (m - p))) :
  is_integer ((mq + np : ℚ) / (m - p)) :=
sorry

end problem_statement_l1902_190289


namespace perimeter_of_one_rectangle_l1902_190272

theorem perimeter_of_one_rectangle (s : ℝ) (rectangle_perimeter rectangle_length rectangle_width : ℝ) (h1 : 4 * s = 240) (h2 : rectangle_width = (1/2) * s) (h3 : rectangle_length = s) (h4 : rectangle_perimeter = 2 * (rectangle_length + rectangle_width)) :
  rectangle_perimeter = 180 := 
sorry

end perimeter_of_one_rectangle_l1902_190272


namespace find_y_l1902_190280

noncomputable def x : ℝ := 0.7142857142857143

def equation (y : ℝ) : Prop :=
  (x * y) / 7 = x^2

theorem find_y : ∃ y : ℝ, equation y ∧ y = 5 :=
by
  use 5
  have h1 : x != 0 := by sorry
  have h2 : (x * 5) / 7 = x^2 := by sorry
  exact ⟨h2, rfl⟩

end find_y_l1902_190280


namespace bob_calories_consumed_l1902_190229

/-- Bob eats half of the pizza with 8 slices, each slice being 300 calories.
   Prove that Bob eats 1200 calories. -/
theorem bob_calories_consumed (total_slices : ℕ) (calories_per_slice : ℕ) (half_slices : ℕ) (calories_consumed : ℕ) 
  (h1 : total_slices = 8)
  (h2 : calories_per_slice = 300)
  (h3 : half_slices = total_slices / 2)
  (h4 : calories_consumed = half_slices * calories_per_slice) 
  : calories_consumed = 1200 := 
sorry

end bob_calories_consumed_l1902_190229


namespace no_cracked_seashells_l1902_190230

theorem no_cracked_seashells (tom_seashells : ℕ) (fred_seashells : ℕ) (total_seashells : ℕ)
  (h1 : tom_seashells = 15) (h2 : fred_seashells = 43) (h3 : total_seashells = 58)
  (h4 : tom_seashells + fred_seashells = total_seashells) : 
  (total_seashells - (tom_seashells + fred_seashells) = 0) :=
by
  sorry

end no_cracked_seashells_l1902_190230


namespace simplify_and_evaluate_expression_l1902_190264

theorem simplify_and_evaluate_expression (x : ℝ) (h : x = -3) : (1 + 1/(x+1)) / ((x^2 + 4*x + 4) / (x+1)) = -1 :=
by
  sorry

end simplify_and_evaluate_expression_l1902_190264


namespace centroid_calculation_correct_l1902_190233

-- Define the vertices of the triangle
def P : ℝ × ℝ := (2, 3)
def Q : ℝ × ℝ := (-1, 4)
def R : ℝ × ℝ := (4, -2)

-- Define the coordinates of the centroid
noncomputable def S : ℝ × ℝ := ((P.1 + Q.1 + R.1) / 3, (P.2 + Q.2 + R.2) / 3)

-- Prove that 7x + 2y = 15 for the centroid
theorem centroid_calculation_correct : 7 * S.1 + 2 * S.2 = 15 :=
by 
  -- Placeholder for the proof steps
  sorry

end centroid_calculation_correct_l1902_190233


namespace geom_seq_sum_first_10_terms_l1902_190252

variable (a : ℕ → ℝ) (a₁ : ℝ) (q : ℝ)
variable (h₀ : a₁ = 1/4)
variable (h₁ : ∀ n, a (n + 1) = a₁ * q ^ n)
variable (S : ℕ → ℝ)
variable (h₂ : S n = a₁ * (1 - q ^ n) / (1 - q))

theorem geom_seq_sum_first_10_terms :
  a 1 = 1 / 4 →
  (a 3) * (a 5) = 4 * ((a 4) - 1) →
  S 10 = 1023 / 4 :=
by
  sorry

end geom_seq_sum_first_10_terms_l1902_190252


namespace k_range_l1902_190297

noncomputable def f (x : ℝ) : ℝ :=
  if 0 < x ∧ x < 1 then -x^3 + 2*x^2 - x
  else if 1 ≤ x then Real.log x
  else 0 -- Technically, we don't care outside (0, +∞), so this else case doesn't matter.

theorem k_range (k : ℝ) :
  (∀ t : ℝ, 0 < t → f t < k * t) ↔ k ∈ (Set.Ioi (1 / Real.exp 1)) :=
by
  sorry

end k_range_l1902_190297


namespace vertex_of_parabola_l1902_190212

theorem vertex_of_parabola : 
  ∀ (x y : ℝ), (y = -x^2 + 3) → (0, 3) ∈ {(h, k) | ∃ (a : ℝ), y = a * (x - h)^2 + k} :=
by
  sorry

end vertex_of_parabola_l1902_190212


namespace sculpture_height_is_34_inches_l1902_190268

-- Define the height of the base in inches
def height_of_base_in_inches : ℕ := 2

-- Define the total height in feet
def total_height_in_feet : ℕ := 3

-- Convert feet to inches (1 foot = 12 inches)
def total_height_in_inches (feet : ℕ) : ℕ := feet * 12

-- The height of the sculpture, given the base and total height
def height_of_sculpture (total_height base_height : ℕ) : ℕ := total_height - base_height

-- State the theorem that the height of the sculpture is 34 inches
theorem sculpture_height_is_34_inches :
  height_of_sculpture (total_height_in_inches total_height_in_feet) height_of_base_in_inches = 34 := by
  sorry

end sculpture_height_is_34_inches_l1902_190268


namespace min_value_fraction_l1902_190295

theorem min_value_fraction (p q r : ℝ) (hp : 0 < p) (hq : 0 < q) (hr : 0 < r) :
  (∃ T : ℝ, T = (5 * r / (3 * p + 2 * q) + 5 * p / (2 * q + 3 * r) + 2 * q / (p + r)) ∧ T = 19 / 4) :=
sorry

end min_value_fraction_l1902_190295


namespace minimum_ab_value_is_two_l1902_190260

noncomputable def minimum_value_ab (a b : ℝ) (h1 : a^2 ≠ 0) (h2 : b ≠ 0)
  (h3 : a^2 * b = a^2 + 1) : ℝ :=
|a * b|

theorem minimum_ab_value_is_two (a b : ℝ) (h1 : a^2 ≠ 0) (h2 : b ≠ 0)
  (h3 : a^2 * b = a^2 + 1) : minimum_value_ab a b h1 h2 h3 = 2 := by
  sorry

end minimum_ab_value_is_two_l1902_190260


namespace methane_reaction_l1902_190285

noncomputable def methane_reacts_with_chlorine
  (moles_CH₄ : ℕ)
  (moles_Cl₂ : ℕ)
  (moles_CCl₄ : ℕ)
  (moles_HCl_produced : ℕ) : Prop :=
  moles_CH₄ = 3 ∧ 
  moles_Cl₂ = 12 ∧ 
  moles_CCl₄ = 3 ∧ 
  moles_HCl_produced = 12

theorem methane_reaction : 
  methane_reacts_with_chlorine 3 12 3 12 :=
by sorry

end methane_reaction_l1902_190285


namespace necessary_but_not_sufficient_l1902_190265

variables (α β : Plane) (m : Line)

-- Define what it means for planes and lines to be perpendicular
def plane_perpendicular (p1 p2 : Plane) : Prop := sorry
def line_perpendicular_plane (l : Line) (p : Plane) : Prop := sorry

-- The main theorem to be established
theorem necessary_but_not_sufficient :
  (plane_perpendicular α β) → (line_perpendicular_plane m β) ∧ ¬ ((plane_perpendicular α β) ↔ (line_perpendicular_plane m β)) :=
sorry

end necessary_but_not_sufficient_l1902_190265


namespace most_suitable_method_l1902_190244

theorem most_suitable_method {x : ℝ} (h : (x - 1) ^ 2 = 4) :
  "Direct method of taking square root" = "Direct method of taking square root" :=
by
  -- We observe that the equation is already in a form 
  -- that is conducive to applying the direct method of taking the square root,
  -- because the equation is already a perfect square on one side and a constant on the other side.
  sorry

end most_suitable_method_l1902_190244


namespace maximise_expression_l1902_190222

theorem maximise_expression {x : ℝ} (hx : 0 < x ∧ x < 1) : 
  ∃ (x_max : ℝ), x_max = 1/2 ∧ 
  (∀ y : ℝ, (0 < y ∧ y < 1) → 3 * y * (1 - y) ≤ 3 * x_max * (1 - x_max)) :=
sorry

end maximise_expression_l1902_190222


namespace same_face_probability_correct_l1902_190270

-- Define the number of sides on the dice
def sides_20 := 20
def sides_16 := 16

-- Define the number of colored sides for each dice category
def maroon_20 := 5
def teal_20 := 8
def cyan_20 := 6
def sparkly_20 := 1

def maroon_16 := 4
def teal_16 := 6
def cyan_16 := 5
def sparkly_16 := 1

-- Define the probabilities of each color matching
def prob_maroon : ℚ := (maroon_20 / sides_20) * (maroon_16 / sides_16)
def prob_teal : ℚ := (teal_20 / sides_20) * (teal_16 / sides_16)
def prob_cyan : ℚ := (cyan_20 / sides_20) * (cyan_16 / sides_16)
def prob_sparkly : ℚ := (sparkly_20 / sides_20) * (sparkly_16 / sides_16)

-- Define the total probability of same face
def prob_same_face := prob_maroon + prob_teal + prob_cyan + prob_sparkly

-- The theorem we need to prove
theorem same_face_probability_correct : 
  prob_same_face = 99 / 320 :=
by
  sorry

end same_face_probability_correct_l1902_190270


namespace parabolas_intersect_with_high_probability_l1902_190296

noncomputable def high_probability_of_intersection : Prop :=
  ∀ (a b c d : ℕ), 1 ≤ a ∧ a ≤ 6 ∧ 1 ≤ b ∧ b ≤ 6 ∧ 1 ≤ c ∧ c ≤ 6 ∧ 1 ≤ d ∧ d ≤ 6 →
  (a - c) ^ 2 + 4 * (b - d) >= 0

theorem parabolas_intersect_with_high_probability : high_probability_of_intersection := sorry

end parabolas_intersect_with_high_probability_l1902_190296


namespace is_divisible_by_7_l1902_190207

theorem is_divisible_by_7 : ∃ k : ℕ, 42 = 7 * k := by
  sorry

end is_divisible_by_7_l1902_190207


namespace minimum_number_of_peanuts_l1902_190276

/--
Five monkeys share a pile of peanuts.
Each monkey divides the peanuts into five piles, leaves one peanut which it eats, and takes away one pile.
This process continues in the same manner until the fifth monkey, who also evenly divides the remaining peanuts into five piles and has one peanut left over.
Prove that the minimum number of peanuts in the pile originally is 3121.
-/
theorem minimum_number_of_peanuts : ∃ N : ℕ, N = 3121 ∧
  (N - 1) % 5 = 0 ∧
  ((4 * ((N - 1) / 5) - 1) % 5 = 0) ∧
  ((4 * ((4 * ((N - 1) / 5) - 1) / 5) - 1) % 5 = 0) ∧
  ((4 * ((4 * ((4 * ((N - 1) / 5) - 1) / 5) - 1) / 5) - 1) % 5 = 0) ∧
  ((4 * ((4 * ((4 * ((4 * ((N - 1) / 5) - 1) / 5) - 1) / 5) - 1) / 5) - 1) / 4) % 5 = 0 :=
by
  sorry

end minimum_number_of_peanuts_l1902_190276


namespace coeff_x2_term_l1902_190291

theorem coeff_x2_term (a b c d e f : ℕ) (h1 : a = 3) (h2 : b = 4) (h3 : c = 5) (h4 : d = 6) (h5 : e = 7) (h6 : f = 8) :
    (a * f + b * e * 1 + c * d) = 82 := 
by
    sorry

end coeff_x2_term_l1902_190291


namespace towels_per_load_l1902_190241

-- Defining the given conditions
def total_towels : ℕ := 42
def number_of_loads : ℕ := 6

-- Defining the problem statement: Prove the number of towels per load
theorem towels_per_load : total_towels / number_of_loads = 7 := by 
  sorry

end towels_per_load_l1902_190241


namespace total_distance_flash_runs_l1902_190224

-- Define the problem with given conditions
theorem total_distance_flash_runs (v k d a : ℝ) (hk : k > 1) : 
  let t := d / (v * (k - 1))
  let distance_to_catch_ace := k * v * t
  let total_distance := distance_to_catch_ace + a
  total_distance = (k * d) / (k - 1) + a := 
by
  sorry

end total_distance_flash_runs_l1902_190224


namespace transform_quadratic_to_squared_form_l1902_190266

theorem transform_quadratic_to_squared_form :
  ∀ x : ℝ, 2 * x^2 - 3 * x + 1 = 0 → (x - 3 / 4)^2 = 1 / 16 :=
by
  intro x h
  sorry

end transform_quadratic_to_squared_form_l1902_190266


namespace problem_statement_l1902_190208

def has_arithmetic_square_root (x : ℝ) : Prop :=
  ∃ y : ℝ, y * y = x

theorem problem_statement :
  (¬ has_arithmetic_square_root (-abs 9)) ∧
  (has_arithmetic_square_root ((-1/4)^2)) ∧
  (has_arithmetic_square_root 0) ∧
  (has_arithmetic_square_root (10^2)) := 
sorry

end problem_statement_l1902_190208


namespace total_price_is_correct_l1902_190255

def total_price_of_hats (total_hats : ℕ) (blue_hat_cost green_hat_cost : ℕ) (num_green_hats : ℕ) : ℕ :=
  let num_blue_hats := total_hats - num_green_hats
  let cost_green_hats := num_green_hats * green_hat_cost
  let cost_blue_hats := num_blue_hats * blue_hat_cost
  cost_green_hats + cost_blue_hats

theorem total_price_is_correct : total_price_of_hats 85 6 7 40 = 550 := 
  sorry

end total_price_is_correct_l1902_190255


namespace sum_x_y_z_l1902_190231

theorem sum_x_y_z (a b : ℝ) (x y z : ℕ) 
  (h_a : a^2 = 16 / 44) 
  (h_b : b^2 = (2 + Real.sqrt 5)^2 / 11) 
  (h_a_neg : a < 0) 
  (h_b_pos : b > 0) 
  (h_expr : (a + b)^3 = x * Real.sqrt y / z) : 
  x + y + z = 181 := 
sorry

end sum_x_y_z_l1902_190231


namespace log_inequality_l1902_190204

variable (a b : ℝ)

theorem log_inequality (h1 : a > b) (h2 : b > 1) : a * Real.log a > b * Real.log b :=
sorry

end log_inequality_l1902_190204


namespace parametric_eq_of_curve_C_max_x_plus_y_on_curve_C_l1902_190214

noncomputable def curve_C (θ : ℝ) : ℝ × ℝ := 
(2 + Real.sqrt 2 * Real.cos θ, 
 2 + Real.sqrt 2 * Real.sin θ)

theorem parametric_eq_of_curve_C (θ : ℝ) : 
    ∃ x y, 
    (x, y) = curve_C θ ∧ 
    (x - 2)^2 + (y - 2)^2 = 2 := by sorry

theorem max_x_plus_y_on_curve_C :
    ∃ x y θ, 
    (x, y) = curve_C θ ∧ 
    (∀ p : ℝ × ℝ, (p.1, p.2) = curve_C θ → 
    p.1 + p.2 ≤ 6) ∧
    x + y = 6 ∧
    x = 3 ∧ 
    y = 3 := by sorry

end parametric_eq_of_curve_C_max_x_plus_y_on_curve_C_l1902_190214


namespace max_curved_sides_l1902_190237

theorem max_curved_sides (n : ℕ) (h : 2 ≤ n) : 
  ∃ m, m = 2 * n - 2 :=
sorry

end max_curved_sides_l1902_190237


namespace polygon_sides_l1902_190258

theorem polygon_sides (n : ℕ) (h : (n - 2) * 180 + 360 = 1800) : n = 10 := by
  sorry

end polygon_sides_l1902_190258


namespace square_area_in_ellipse_l1902_190211

theorem square_area_in_ellipse :
  (∃ t : ℝ, 
    (∀ x y : ℝ, ((x = t ∨ x = -t) ∧ (y = t ∨ y = -t)) → (x^2 / 4 + y^2 / 8 = 1)) 
    ∧ t > 0 
    ∧ ((2 * t)^2 = 32 / 3)) :=
sorry

end square_area_in_ellipse_l1902_190211


namespace total_seashells_found_l1902_190215

-- Defining the conditions
def joan_daily_seashells : ℕ := 6
def jessica_daily_seashells : ℕ := 8
def length_of_vacation : ℕ := 7

-- Stating the theorem
theorem total_seashells_found : 
  (joan_daily_seashells + jessica_daily_seashells) * length_of_vacation = 98 :=
by
  sorry

end total_seashells_found_l1902_190215


namespace pigs_and_dogs_more_than_sheep_l1902_190239

-- Define the number of pigs and sheep
def numberOfPigs : ℕ := 42
def numberOfSheep : ℕ := 48

-- Define the number of dogs such that it is the same as the number of pigs
def numberOfDogs : ℕ := numberOfPigs

-- Define the total number of pigs and dogs
def totalPigsAndDogs : ℕ := numberOfPigs + numberOfDogs

-- State the theorem about the difference between pigs and dogs and the number of sheep
theorem pigs_and_dogs_more_than_sheep :
  totalPigsAndDogs - numberOfSheep = 36 := 
sorry

end pigs_and_dogs_more_than_sheep_l1902_190239


namespace expand_expression_l1902_190274

theorem expand_expression (x : ℝ) : (x - 3) * (x + 3) * (x^2 + 9) = x^4 - 81 := 
by
  sorry

end expand_expression_l1902_190274


namespace eraser_cost_l1902_190243

theorem eraser_cost (initial_money : ℕ) (scissors_count : ℕ) (scissors_price : ℕ) (erasers_count : ℕ) (remaining_money : ℕ) :
    initial_money = 100 →
    scissors_count = 8 →
    scissors_price = 5 →
    erasers_count = 10 →
    remaining_money = 20 →
    (initial_money - scissors_count * scissors_price - remaining_money) / erasers_count = 4 :=
by
  intro h1 h2 h3 h4 h5
  sorry

end eraser_cost_l1902_190243


namespace range_of_a_l1902_190245

theorem range_of_a :
  (¬ ∃ x : ℝ, x^2 + 2 * a * x + a ≤ 0) → (0 < a ∧ a < 1) :=
by
  intros
  sorry

end range_of_a_l1902_190245


namespace shaded_region_is_correct_l1902_190205

noncomputable def area_shaded_region : ℝ :=
  let r_small := (3 : ℝ) / 2
  let r_large := (15 : ℝ) / 2
  let area_small := (1 / 2) * Real.pi * r_small^2
  let area_large := (1 / 2) * Real.pi * r_large^2
  (area_large - 2 * area_small + 3 * area_small)

theorem shaded_region_is_correct :
  area_shaded_region = (117 / 4) * Real.pi :=
by
  -- The proof will go here.
  sorry

end shaded_region_is_correct_l1902_190205


namespace find_y_values_l1902_190234

theorem find_y_values
  (y₁ y₂ y₃ y₄ y₅ : ℝ)
  (h₁ : y₁ + 3 * y₂ + 6 * y₃ + 10 * y₄ + 15 * y₅ = 3)
  (h₂ : 3 * y₁ + 6 * y₂ + 10 * y₃ + 15 * y₄ + 21 * y₅ = 20)
  (h₃ : 6 * y₁ + 10 * y₂ + 15 * y₃ + 21 * y₄ + 28 * y₅ = 86)
  (h₄ : 10 * y₁ + 15 * y₂ + 21 * y₃ + 28 * y₄ + 36 * y₅ = 225) :
  15 * y₁ + 21 * y₂ + 28 * y₃ + 36 * y₄ + 45 * y₅ = 395 :=
by {
  sorry
}

end find_y_values_l1902_190234


namespace find_a_for_opposite_roots_l1902_190235

-- Define the equation and condition using the given problem details
theorem find_a_for_opposite_roots (a : ℝ) 
  (h : ∀ (x : ℝ), x^2 - (a^2 - 2 * a - 15) * x + a - 1 = 0 
    → (∃! (x1 x2 : ℝ), x1 + x2 = 0)) :
  a = -3 := 
sorry

end find_a_for_opposite_roots_l1902_190235


namespace find_c_l1902_190286

theorem find_c (c : ℝ) (h : ∃ β : ℝ, (5 + β = -c) ∧ (5 * β = 45)) : c = -14 := 
  sorry

end find_c_l1902_190286


namespace weeks_to_meet_goal_l1902_190279

def hourly_rate : ℕ := 6
def hours_monday : ℕ := 2
def hours_tuesday : ℕ := 3
def hours_wednesday : ℕ := 4
def hours_thursday : ℕ := 2
def hours_friday : ℕ := 3
def helmet_cost : ℕ := 340
def gloves_cost : ℕ := 45
def initial_savings : ℕ := 40
def misc_expenses : ℕ := 20

theorem weeks_to_meet_goal : 
  let total_needed := helmet_cost + gloves_cost + misc_expenses
  let total_deficit := total_needed - initial_savings
  let total_weekly_hours := hours_monday + hours_tuesday + hours_wednesday + hours_thursday + hours_friday
  let weekly_earnings := total_weekly_hours * hourly_rate
  let weeks_required := Nat.ceil (total_deficit / weekly_earnings)
  weeks_required = 5 := sorry

end weeks_to_meet_goal_l1902_190279


namespace original_denominator_value_l1902_190209

theorem original_denominator_value (d : ℕ) (h1 : 3 + 3 = 6) (h2 : ((6 : ℕ) / (d + 3 : ℕ) = (1 / 3 : ℚ))) : d = 15 :=
sorry

end original_denominator_value_l1902_190209


namespace sequence_formula_l1902_190246

theorem sequence_formula (a : ℕ → ℝ) (h₀ : a 1 = 1) (h₁ : ∀ n : ℕ, 0 < n → a (n + 1) = a n / (1 + a n)) : 
  ∀ n : ℕ, 0 < n → a n = 1 / n := 
by 
  sorry

end sequence_formula_l1902_190246


namespace sin_alpha_cos_alpha_l1902_190253

theorem sin_alpha_cos_alpha {α : ℝ} (h : Real.sin (3 * Real.pi - α) = -2 * Real.sin (Real.pi / 2 + α)) :
  Real.sin α * Real.cos α = -2 / 5 :=
by
  sorry

end sin_alpha_cos_alpha_l1902_190253


namespace sufficient_not_necessary_condition_l1902_190217

theorem sufficient_not_necessary_condition (x : ℝ) (a : ℝ × ℝ) (b : ℝ × ℝ) (hx : x = 2)
    (ha : a = (x, 1)) (hb : b = (4, x)) : 
    (∃ k : ℝ, a = (k * b.1, k * b.2)) ∧ (¬ (∀ k : ℝ, a = (k * b.1, k * b.2))) :=
by 
  sorry

end sufficient_not_necessary_condition_l1902_190217


namespace find_number_l1902_190201

theorem find_number (x : ℝ) (h : 54 / 2 + 3 * x = 75) : x = 16 :=
by
  sorry

end find_number_l1902_190201


namespace terrell_lifting_l1902_190202

theorem terrell_lifting :
  (3 * 25 * 10 = 3 * 20 * 12.5) :=
by
  sorry

end terrell_lifting_l1902_190202


namespace value_of_a_squared_plus_2a_l1902_190251

theorem value_of_a_squared_plus_2a (a x : ℝ) (h1 : x = -5) (h2 : 2 * x + 8 = x / 5 - a) : a^2 + 2 * a = 3 :=
by {
  sorry
}

end value_of_a_squared_plus_2a_l1902_190251


namespace train_late_average_speed_l1902_190216

theorem train_late_average_speed 
  (distance : ℝ) (on_time_speed : ℝ) (late_time_additional : ℝ) 
  (on_time : distance / on_time_speed = 1.75) 
  (late : distance / (on_time_speed * 2/2.5) = 2) :
  distance / 2 = 35 :=
by
  sorry

end train_late_average_speed_l1902_190216
