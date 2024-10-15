import Mathlib

namespace NUMINAMATH_GPT_a4_value_l72_7272

variable (a : ℕ → ℤ)
variable (S : ℕ → ℤ)

-- Condition: The sum of the first n terms of the sequence {a_n} is S_n = n^2 - 1
axiom sum_of_sequence (n : ℕ) : S n = n^2 - 1

-- We need to prove that a_4 = 7
theorem a4_value : a 4 = S 4 - S 3 :=
by 
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_a4_value_l72_7272


namespace NUMINAMATH_GPT_min_gumballs_to_ensure_four_same_color_l72_7280

/-- A structure to represent the number of gumballs of each color. -/
structure Gumballs :=
(red : ℕ)
(white : ℕ)
(blue : ℕ)
(green : ℕ)

def gumball_machine : Gumballs := { red := 10, white := 9, blue := 8, green := 6 }

/-- Theorem to state the minimum number of gumballs required to ensure at least four of any color. -/
theorem min_gumballs_to_ensure_four_same_color 
  (g : Gumballs) 
  (h1 : g.red = 10)
  (h2 : g.white = 9)
  (h3 : g.blue = 8)
  (h4 : g.green = 6) : 
  ∃ n, n = 13 := 
sorry

end NUMINAMATH_GPT_min_gumballs_to_ensure_four_same_color_l72_7280


namespace NUMINAMATH_GPT_sum_fourth_powers_l72_7274

theorem sum_fourth_powers (a b c : ℝ) 
  (h1 : a + b + c = 1)
  (h2 : a^2 + b^2 + c^2 = 2)
  (h3 : a^3 + b^3 + c^3 = 3) : 
  a^4 + b^4 + c^4 = 25 / 6 :=
by sorry

end NUMINAMATH_GPT_sum_fourth_powers_l72_7274


namespace NUMINAMATH_GPT_price_equation_l72_7203

variable (x : ℝ)

def first_discount (x : ℝ) : ℝ := x - 5

def second_discount (price_after_first_discount : ℝ) : ℝ := 0.8 * price_after_first_discount

theorem price_equation
  (hx : second_discount (first_discount x) = 60) :
  0.8 * (x - 5) = 60 := by
  sorry

end NUMINAMATH_GPT_price_equation_l72_7203


namespace NUMINAMATH_GPT_intersection_A_B_at_3_range_of_a_l72_7244

open Set

-- Definitions from the condition
def A (x : ℝ) : Prop := abs x ≥ 2
def B (x a : ℝ) : Prop := (x - 2 * a) * (x + 3) < 0

-- Part (Ⅰ)
theorem intersection_A_B_at_3 :
  let a := 3
  let A := {x : ℝ | abs x ≥ 2}
  let B := {x : ℝ | (x - 6) * (x + 3) < 0}
  {x : ℝ | A x} ∩ {x : ℝ | B x} = {x : ℝ | (-3 < x ∧ x ≤ -2) ∨ (2 ≤ x ∧ x < 6)} :=
by
  sorry

-- Part (Ⅱ)
theorem range_of_a (a : ℝ) (h : a > 0) :
  (∀ x : ℝ, A x ∨ B x a) → a ≥ 1 :=
by
  sorry

end NUMINAMATH_GPT_intersection_A_B_at_3_range_of_a_l72_7244


namespace NUMINAMATH_GPT_discount_price_l72_7261

theorem discount_price (original_price : ℝ) (discount_rate : ℝ) (current_price : ℝ) 
  (h1 : original_price = 120) 
  (h2 : discount_rate = 0.8) 
  (h3 : current_price = original_price * discount_rate) : 
  current_price = 96 := 
by
  sorry

end NUMINAMATH_GPT_discount_price_l72_7261


namespace NUMINAMATH_GPT_z_is_200_percent_of_x_l72_7295

theorem z_is_200_percent_of_x
  (x y z : ℝ)
  (h1 : 0.45 * z = 1.20 * y)
  (h2 : y = 0.75 * x) :
  z = 2 * x :=
sorry

end NUMINAMATH_GPT_z_is_200_percent_of_x_l72_7295


namespace NUMINAMATH_GPT_g6_eq_16_l72_7240

-- Definition of the function g that satisfies the given conditions
variable (g : ℝ → ℝ)

-- Given conditions
axiom functional_eq : ∀ x y : ℝ, g (x + y) = g x * g y
axiom g3_eq_4 : g 3 = 4

-- The goal is to prove g(6) = 16
theorem g6_eq_16 : g 6 = 16 := by
  sorry

end NUMINAMATH_GPT_g6_eq_16_l72_7240


namespace NUMINAMATH_GPT_positive_rational_solutions_condition_l72_7260

-- Definitions used in Lean 4 statement corresponding to conditions in the problem.
variable (a b : ℚ)

-- Lean Statement encapsulating the mathematical proof problem.
theorem positive_rational_solutions_condition :
  ∃ x y : ℚ, x > 0 ∧ y > 0 ∧ x * y = a ∧ x + y = b ↔ (∃ k : ℚ, k^2 = b^2 - 4 * a ∧ k > 0) :=
by
  sorry

end NUMINAMATH_GPT_positive_rational_solutions_condition_l72_7260


namespace NUMINAMATH_GPT_orthogonal_vectors_l72_7284

theorem orthogonal_vectors (x : ℝ) :
  (3 * x - 4 * 6 = 0) → x = 8 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_orthogonal_vectors_l72_7284


namespace NUMINAMATH_GPT_find_n_150_l72_7292

def special_sum (k n : ℕ) : ℕ := (n * (2 * k + n - 1)) / 2

theorem find_n_150 : ∃ n : ℕ, special_sum 3 n = 150 ∧ n = 15 :=
by
  sorry

end NUMINAMATH_GPT_find_n_150_l72_7292


namespace NUMINAMATH_GPT_find_y_l72_7278

theorem find_y (x y : ℕ) (h1 : x > 0 ∧ y > 0) (h2 : x % y = 9) (h3 : (x:ℝ) / (y:ℝ) = 96.45) : y = 20 :=
by
  sorry

end NUMINAMATH_GPT_find_y_l72_7278


namespace NUMINAMATH_GPT_journey_total_time_l72_7255

noncomputable def total_time (D : ℝ) (r_dist : ℕ → ℕ) (r_time : ℕ → ℕ) (u_speed : ℝ) : ℝ :=
  let dist_uphill := D * (r_dist 1) / (r_dist 1 + r_dist 2 + r_dist 3)
  let t_uphill := (dist_uphill / u_speed)
  let k := t_uphill / (r_time 1)
  (r_time 1 + r_time 2 + r_time 3) * k

theorem journey_total_time :
  total_time 50 (fun n => if n = 1 then 1 else if n = 2 then 2 else 3) 
                (fun n => if n = 1 then 4 else if n = 2 then 5 else 6) 
                3 = 10 + 5/12 :=
by
  sorry

end NUMINAMATH_GPT_journey_total_time_l72_7255


namespace NUMINAMATH_GPT_range_of_a_l72_7247

theorem range_of_a (a : ℝ) : (∃ x₀ : ℝ, x₀^2 + (a-1)*x₀ + 1 < 0) ↔ (a < -1 ∨ a > 3) :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l72_7247


namespace NUMINAMATH_GPT_number_of_truthful_people_l72_7237

-- Definitions from conditions
def people := Fin 100
def tells_truth (p : people) : Prop := sorry -- Placeholder definition.

-- Conditions
axiom c1 : ∃ p : people, ¬ tells_truth p
axiom c2 : ∀ p1 p2 : people, p1 ≠ p2 → (tells_truth p1 ∨ tells_truth p2)

-- Goal
theorem number_of_truthful_people : 
  ∃ S : Finset people, S.card = 99 ∧ (∀ p ∈ S, tells_truth p) :=
sorry

end NUMINAMATH_GPT_number_of_truthful_people_l72_7237


namespace NUMINAMATH_GPT_fraction_a_over_d_l72_7256

-- Defining the given conditions as hypotheses
variables (a b c d : ℚ)

-- Conditions
axiom h1 : a / b = 20
axiom h2 : c / b = 5
axiom h3 : c / d = 1 / 15

-- Goal to prove
theorem fraction_a_over_d : a / d = 4 / 15 :=
by
  sorry

end NUMINAMATH_GPT_fraction_a_over_d_l72_7256


namespace NUMINAMATH_GPT_ratio_of_p_to_r_l72_7299

theorem ratio_of_p_to_r (p q r s : ℚ) 
  (h1 : p / q = 5 / 4) 
  (h2 : r / s = 4 / 3) 
  (h3 : s / q = 1 / 8) : 
  p / r = 15 / 2 := 
by 
  sorry

end NUMINAMATH_GPT_ratio_of_p_to_r_l72_7299


namespace NUMINAMATH_GPT_statement_is_true_l72_7234

theorem statement_is_true (a b : ℝ) (h_a : 0 < a) (h_b : 0 < b) 
  (h : ∀ x : ℝ, |x + 2| < b → |(3 * x + 2) + 4| < a) : b ≤ a / 3 :=
by
  sorry

end NUMINAMATH_GPT_statement_is_true_l72_7234


namespace NUMINAMATH_GPT_annual_subscription_cost_l72_7239

theorem annual_subscription_cost :
  (10 * 12) * (1 - 0.2) = 96 :=
by
  sorry

end NUMINAMATH_GPT_annual_subscription_cost_l72_7239


namespace NUMINAMATH_GPT_isabel_ds_games_left_l72_7241

-- Define the initial number of DS games Isabel had
def initial_ds_games : ℕ := 90

-- Define the number of DS games Isabel gave to her friend
def ds_games_given : ℕ := 87

-- Define a function to calculate the remaining DS games
def remaining_ds_games (initial : ℕ) (given : ℕ) : ℕ := initial - given

-- Statement of the theorem we need to prove
theorem isabel_ds_games_left : remaining_ds_games initial_ds_games ds_games_given = 3 := by
  sorry

end NUMINAMATH_GPT_isabel_ds_games_left_l72_7241


namespace NUMINAMATH_GPT_positive_integer_solution_l72_7245

theorem positive_integer_solution (n x y : ℕ) (hn : 0 < n) (hx : 0 < x) (hy : 0 < y) :
  y ^ 2 + x * y + 3 * x = n * (x ^ 2 + x * y + 3 * y) → n = 1 :=
sorry

end NUMINAMATH_GPT_positive_integer_solution_l72_7245


namespace NUMINAMATH_GPT_triangle_perimeter_l72_7230

-- Definitions for the conditions
def inscribed_circle_of_triangle_tangent_at (radius : ℝ) (DP : ℝ) (PE : ℝ) : Prop :=
  radius = 27 ∧ DP = 29 ∧ PE = 33

-- Perimeter calculation theorem
theorem triangle_perimeter (r DP PE : ℝ) (h : inscribed_circle_of_triangle_tangent_at r DP PE) : 
  ∃ perimeter : ℝ, perimeter = 774 :=
by
  sorry

end NUMINAMATH_GPT_triangle_perimeter_l72_7230


namespace NUMINAMATH_GPT_subscription_total_l72_7265

theorem subscription_total (a b c : ℝ) (h1 : a = b + 4000) (h2 : b = c + 5000) (h3 : 15120 / 36000 = a / (a + b + c)) : 
  a + b + c = 50000 :=
by 
  sorry

end NUMINAMATH_GPT_subscription_total_l72_7265


namespace NUMINAMATH_GPT_quadratic_solution_set_l72_7220

theorem quadratic_solution_set (a b c : ℝ) 
  (h : ∀ x : ℝ, ax^2 + bx + c > 0 ↔ x < -2 ∨ x > 3) :
  (a > 0) ∧ 
  (∀ x : ℝ, bx + c > 0 ↔ x < 6) = false ∧ 
  (a + b + c < 0) ∧
  (∀ x : ℝ, cx^2 - bx + a < 0 ↔ x < -1 / 3 ∨ x > 1 / 2) :=
sorry

end NUMINAMATH_GPT_quadratic_solution_set_l72_7220


namespace NUMINAMATH_GPT_ordered_pair_and_sum_of_squares_l72_7264

theorem ordered_pair_and_sum_of_squares :
  ∃ x y : ℚ, 
    6 * x - 48 * y = 2 ∧ 
    3 * y - x = 4 ∧ 
    x ^ 2 + y ^ 2 = 442 / 25 :=
by
  sorry

end NUMINAMATH_GPT_ordered_pair_and_sum_of_squares_l72_7264


namespace NUMINAMATH_GPT_harvest_bushels_l72_7269

def num_rows : ℕ := 5
def stalks_per_row : ℕ := 80
def stalks_per_bushel : ℕ := 8

theorem harvest_bushels : (num_rows * stalks_per_row) / stalks_per_bushel = 50 := by
  sorry

end NUMINAMATH_GPT_harvest_bushels_l72_7269


namespace NUMINAMATH_GPT_cost_per_piece_l72_7246

variable (totalCost : ℝ) (numberOfPizzas : ℝ) (piecesPerPizza : ℝ)

theorem cost_per_piece (h1 : totalCost = 80) (h2 : numberOfPizzas = 4) (h3 : piecesPerPizza = 5) :
  totalCost / numberOfPizzas / piecesPerPizza = 4 := by
sorry

end NUMINAMATH_GPT_cost_per_piece_l72_7246


namespace NUMINAMATH_GPT_hyperbola_condition_l72_7226

-- Definitions and hypotheses
def is_hyperbola (m n : ℝ) (x y : ℝ) : Prop := m * x^2 - n * y^2 = 1

-- Statement of the problem
theorem hyperbola_condition (m n : ℝ) : (∃ x y : ℝ, is_hyperbola m n x y) ↔ m * n > 0 :=
by sorry

end NUMINAMATH_GPT_hyperbola_condition_l72_7226


namespace NUMINAMATH_GPT_connie_tickets_l72_7207

theorem connie_tickets (total_tickets spent_on_koala spent_on_earbuds spent_on_glow_bracelets : ℕ)
  (h1 : total_tickets = 50)
  (h2 : spent_on_koala = total_tickets / 2)
  (h3 : spent_on_earbuds = 10)
  (h4 : total_tickets = spent_on_koala + spent_on_earbuds + spent_on_glow_bracelets) :
  spent_on_glow_bracelets = 15 :=
by
  sorry

end NUMINAMATH_GPT_connie_tickets_l72_7207


namespace NUMINAMATH_GPT_electricity_usage_l72_7218

theorem electricity_usage 
  (total_usage : ℕ) (saved_cost : ℝ) (initial_cost : ℝ) (peak_cost : ℝ) (off_peak_cost : ℝ) 
  (usage_peak : ℕ) (usage_off_peak : ℕ) :
  total_usage = 100 →
  saved_cost = 3 →
  initial_cost = 0.55 →
  peak_cost = 0.6 →
  off_peak_cost = 0.4 →
  usage_peak + usage_off_peak = total_usage →
  (total_usage * initial_cost - (peak_cost * usage_peak + off_peak_cost * usage_off_peak) = saved_cost) →
  usage_peak = 60 ∧ usage_off_peak = 40 :=
by
  intros h1 h2 h3 h4 h5 h6 h7
  sorry

end NUMINAMATH_GPT_electricity_usage_l72_7218


namespace NUMINAMATH_GPT_cave_depth_l72_7200

theorem cave_depth 
  (total_depth : ℕ) 
  (remaining_depth : ℕ) 
  (h1 : total_depth = 974) 
  (h2 : remaining_depth = 386) : 
  total_depth - remaining_depth = 588 := 
by 
  sorry

end NUMINAMATH_GPT_cave_depth_l72_7200


namespace NUMINAMATH_GPT_find_other_number_l72_7259

-- Definitions for the given conditions
def A : ℕ := 500
def LCM : ℕ := 3000
def HCF : ℕ := 100

-- Theorem statement: If A = 500, LCM(A, B) = 3000, and HCF(A, B) = 100, then B = 600.
theorem find_other_number (B : ℕ) (h1 : A = 500) (h2 : Nat.lcm A B = 3000) (h3 : Nat.gcd A B = 100) :
  B = 600 :=
by
  sorry

end NUMINAMATH_GPT_find_other_number_l72_7259


namespace NUMINAMATH_GPT_tom_age_ratio_l72_7297

-- Definitions of the variables
variables (T : ℕ) (N : ℕ)

-- Conditions given in the problem
def condition1 : Prop := T = 2 * (T / 2)
def condition2 : Prop := (T - 3) = 3 * (T / 2 - 12)

-- The ratio theorem to prove
theorem tom_age_ratio (h1 : condition1 T) (h2 : condition2 T) : T / N = 22 :=
by
  sorry

end NUMINAMATH_GPT_tom_age_ratio_l72_7297


namespace NUMINAMATH_GPT_intersection_eq_l72_7251

-- Definitions of sets A and B
def A : Set ℤ := {0, 1}
def B : Set ℤ := {-1, 1}

-- The theorem statement
theorem intersection_eq : A ∩ B = {1} :=
by
  unfold A B
  sorry

end NUMINAMATH_GPT_intersection_eq_l72_7251


namespace NUMINAMATH_GPT_numbers_not_perfect_squares_or_cubes_l72_7285

theorem numbers_not_perfect_squares_or_cubes : 
  let total_numbers := 200
  let perfect_squares := 14
  let perfect_cubes := 5
  let sixth_powers := 1
  total_numbers - (perfect_squares + perfect_cubes - sixth_powers) = 182 :=
by
  sorry

end NUMINAMATH_GPT_numbers_not_perfect_squares_or_cubes_l72_7285


namespace NUMINAMATH_GPT_smallest_r_l72_7210

theorem smallest_r {p q r : ℕ} (h1 : p < q) (h2 : q < r) (h3 : 2 * q = p + r) (h4 : r * r = p * q) : r = 5 :=
sorry

end NUMINAMATH_GPT_smallest_r_l72_7210


namespace NUMINAMATH_GPT_correct_option_l72_7233

theorem correct_option (a b c d : ℝ) (ha : a < 0) (hb : b > 0) (hd : d < 1) 
  (hA : 2 = (a-1)^2 - 2) (hB : 6 = (b-1)^2 - 2) (hC : d = (c-1)^2 - 2) :
  a < c ∧ c < b :=
by
  sorry

end NUMINAMATH_GPT_correct_option_l72_7233


namespace NUMINAMATH_GPT_find_number_l72_7205

def sum := 555 + 445
def difference := 555 - 445
def quotient := 2 * difference
def remainder := 30
def N : ℕ := 220030

theorem find_number (N : ℕ) : 
  N = sum * quotient + remainder :=
  by
    sorry

end NUMINAMATH_GPT_find_number_l72_7205


namespace NUMINAMATH_GPT_grid_to_black_probability_l72_7254

theorem grid_to_black_probability :
  let n := 16
  let p_black_after_rotation := 3 / 4
  (p_black_after_rotation ^ n) = (3 / 4) ^ 16 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_grid_to_black_probability_l72_7254


namespace NUMINAMATH_GPT_compute_g_neg_x_l72_7262

noncomputable def g (x : ℝ) : ℝ := (x^2 + 3*x + 2) / (x^2 - 3*x + 2)

theorem compute_g_neg_x (x : ℝ) (h : x^2 ≠ 2) : g (-x) = 1 / g x := 
  by sorry

end NUMINAMATH_GPT_compute_g_neg_x_l72_7262


namespace NUMINAMATH_GPT_reflection_line_coordinates_sum_l72_7250

theorem reflection_line_coordinates_sum (m b : ℝ)
  (h : ∀ (x y x' y' : ℝ), (x, y) = (-4, 2) → (x', y') = (2, 6) → 
  ∃ (m b : ℝ), y = m * x + b ∧ y' = m * x' + b ∧ ∀ (p q : ℝ), 
  (p, q) = ((x+x')/2, (y+y')/2) → p = ((-4 + 2)/2) ∧ q = ((2 + 6)/2)) :
  m + b = 1 :=
by
  sorry

end NUMINAMATH_GPT_reflection_line_coordinates_sum_l72_7250


namespace NUMINAMATH_GPT_sandwiches_with_ten_loaves_l72_7242

def sandwiches_per_loaf : ℕ := 18 / 3

def num_sandwiches (loaves: ℕ) : ℕ := sandwiches_per_loaf * loaves

theorem sandwiches_with_ten_loaves :
  num_sandwiches 10 = 60 := by
  sorry

end NUMINAMATH_GPT_sandwiches_with_ten_loaves_l72_7242


namespace NUMINAMATH_GPT_waitress_tips_fraction_l72_7231

theorem waitress_tips_fraction
  (S : ℝ) -- salary
  (T : ℝ) -- tips
  (hT : T = (11 / 4) * S) -- tips are 11/4 of salary
  (I : ℝ) -- total income
  (hI : I = S + T) -- total income is the sum of salary and tips
  : (T / I) = (11 / 15) := -- fraction of income from tips is 11/15
by
  sorry

end NUMINAMATH_GPT_waitress_tips_fraction_l72_7231


namespace NUMINAMATH_GPT_find_q_l72_7217

theorem find_q (q: ℕ) (h: 81^10 = 3^q) : q = 40 :=
by
  sorry

end NUMINAMATH_GPT_find_q_l72_7217


namespace NUMINAMATH_GPT_faye_candies_final_count_l72_7286

def initialCandies : ℕ := 47
def candiesEaten : ℕ := 25
def candiesReceived : ℕ := 40

theorem faye_candies_final_count : (initialCandies - candiesEaten + candiesReceived) = 62 :=
by
  sorry

end NUMINAMATH_GPT_faye_candies_final_count_l72_7286


namespace NUMINAMATH_GPT_stream_speed_l72_7268

-- Definitions based on conditions
def speed_in_still_water : ℝ := 5
def distance_downstream : ℝ := 100
def time_downstream : ℝ := 10

-- The required speed of the stream
def speed_of_stream (v : ℝ) : Prop :=
  distance_downstream = (speed_in_still_water + v) * time_downstream

-- Proof statement: the speed of the stream is 5 km/hr
theorem stream_speed : ∃ v, speed_of_stream v ∧ v = 5 := 
by
  use 5
  unfold speed_of_stream
  sorry

end NUMINAMATH_GPT_stream_speed_l72_7268


namespace NUMINAMATH_GPT_skew_lines_sufficient_not_necessary_l72_7296

-- Definitions for the conditions
def skew_lines (l1 l2 : Type) : Prop := sorry -- Definition of skew lines
def do_not_intersect (l1 l2 : Type) : Prop := sorry -- Definition of not intersecting

-- The main theorem statement
theorem skew_lines_sufficient_not_necessary (l1 l2 : Type) :
  (skew_lines l1 l2) → (do_not_intersect l1 l2) ∧ ¬ (do_not_intersect l1 l2 → skew_lines l1 l2) :=
by
  sorry

end NUMINAMATH_GPT_skew_lines_sufficient_not_necessary_l72_7296


namespace NUMINAMATH_GPT_proof_equivalent_problem_l72_7202

variables (a b c : ℝ)
-- Conditions
axiom h1 : a < b
axiom h2 : b < 0
axiom h3 : c > 0

theorem proof_equivalent_problem :
  (a * c < b * c) ∧ (a + b + c < b + c) ∧ (c / a > 1) :=
by
  sorry

end NUMINAMATH_GPT_proof_equivalent_problem_l72_7202


namespace NUMINAMATH_GPT_arithmetic_sequence_sum_l72_7293

noncomputable def a_n (a1 d : ℕ) (n : ℕ) : ℕ := a1 + (n - 1) * d

theorem arithmetic_sequence_sum 
  {a1 d : ℕ} (h_pos_d : d > 0) 
  (h_sum : a1 + (a1 + d) + (a1 + 2 * d) = 15) 
  (h_prod : a1 * (a1 + d) * (a1 + 2 * d) = 80) 
  : a_n a1 d 11 + a_n a1 d 12 + a_n a1 d 13 = 105 :=
sorry

end NUMINAMATH_GPT_arithmetic_sequence_sum_l72_7293


namespace NUMINAMATH_GPT_increasing_exponential_is_necessary_condition_l72_7221

variable {a : ℝ}

theorem increasing_exponential_is_necessary_condition (h : ∀ x y : ℝ, x < y → a ^ x < a ^ y) :
    (a > 1) ∧ (¬ (a > 2 → a > 1)) :=
by
  sorry

end NUMINAMATH_GPT_increasing_exponential_is_necessary_condition_l72_7221


namespace NUMINAMATH_GPT_problem_r_of_3_eq_88_l72_7206

def q (x : ℤ) : ℤ := 2 * x - 5
def r (x : ℤ) : ℤ := x^3 + 2 * x^2 - x - 4

theorem problem_r_of_3_eq_88 : r 3 = 88 :=
by
  sorry

end NUMINAMATH_GPT_problem_r_of_3_eq_88_l72_7206


namespace NUMINAMATH_GPT_black_region_area_is_correct_l72_7252

noncomputable def area_of_black_region : ℕ :=
  let area_large_square := 10 * 10
  let area_first_smaller_square := 4 * 4
  let area_second_smaller_square := 2 * 2
  area_large_square - (area_first_smaller_square + area_second_smaller_square)

theorem black_region_area_is_correct :
  area_of_black_region = 80 :=
by
  sorry

end NUMINAMATH_GPT_black_region_area_is_correct_l72_7252


namespace NUMINAMATH_GPT_find_gross_salary_l72_7211

open Real

noncomputable def bill_take_home_salary : ℝ := 40000
noncomputable def property_tax : ℝ := 2000
noncomputable def sales_tax : ℝ := 3000
noncomputable def income_tax_rate : ℝ := 0.10

theorem find_gross_salary (gross_salary : ℝ) :
  bill_take_home_salary = gross_salary - (income_tax_rate * gross_salary + property_tax + sales_tax) →
  gross_salary = 50000 :=
by
  sorry

end NUMINAMATH_GPT_find_gross_salary_l72_7211


namespace NUMINAMATH_GPT_curve_intersects_at_point_2_3_l72_7235

open Real

theorem curve_intersects_at_point_2_3 :
  ∃ (t₁ t₂ : ℝ), t₁ ≠ t₂ ∧ 
                 (t₁^2 - 4 = t₂^2 - 4) ∧ 
                 (t₁^3 - 6 * t₁ + 3 = t₂^3 - 6 * t₂ + 3) ∧ 
                 (t₁^2 - 4 = 2) ∧ 
                 (t₁^3 - 6 * t₁ + 3 = 3) :=
by
  sorry

end NUMINAMATH_GPT_curve_intersects_at_point_2_3_l72_7235


namespace NUMINAMATH_GPT_wheel_speed_l72_7263

def original_circumference_in_miles := 10 / 5280
def time_factor := 3600
def new_time_factor := 3600 - (1/3)

theorem wheel_speed
  (r : ℝ) 
  (original_speed : r * time_factor = original_circumference_in_miles * 3600)
  (new_speed : (r + 5) * (time_factor - 1/10800) = original_circumference_in_miles * 3600) :
  r = 10 :=
sorry

end NUMINAMATH_GPT_wheel_speed_l72_7263


namespace NUMINAMATH_GPT_inverse_prop_function_through_point_l72_7253

theorem inverse_prop_function_through_point :
  ∀ (f : ℝ → ℝ), (∀ x, f x = k / x) → (f 1 = 2) → (f (-1) = -2) :=
by
  intros f h_inv_prop h_f1
  sorry

end NUMINAMATH_GPT_inverse_prop_function_through_point_l72_7253


namespace NUMINAMATH_GPT_tan_addition_formula_l72_7282

theorem tan_addition_formula (x : ℝ) (h : Real.tan x = Real.sqrt 3) : 
  Real.tan (x + Real.pi / 3) = -Real.sqrt 3 := 
by 
  sorry

end NUMINAMATH_GPT_tan_addition_formula_l72_7282


namespace NUMINAMATH_GPT_prob_of_nine_correct_is_zero_l72_7216

-- Define the necessary components and properties of the problem
def is_correct_placement (letter: ℕ) (envelope: ℕ) : Prop := letter = envelope

def is_random_distribution (letters : Fin 10 → Fin 10) : Prop := true

-- State the theorem formally
theorem prob_of_nine_correct_is_zero (f : Fin 10 → Fin 10) :
  is_random_distribution f →
  (∃ (count : ℕ), count = 9 ∧ (∀ i : Fin 10, is_correct_placement i (f i) ↔ i = count)) → false :=
by
  sorry

end NUMINAMATH_GPT_prob_of_nine_correct_is_zero_l72_7216


namespace NUMINAMATH_GPT_perimeter_region_l72_7238

theorem perimeter_region (rectangle_height : ℕ) (height_eq_sixteen : rectangle_height = 16) (rect_area_eq : 12 * rectangle_height = 192) (total_area_eq : 12 * rectangle_height - 60 = 132):
  (rectangle_height + 12 + 4 + 6 + 10 * 2) = 54 :=
by
  have h1 : 12 * 16 = 192 := by sorry
  exact sorry


end NUMINAMATH_GPT_perimeter_region_l72_7238


namespace NUMINAMATH_GPT_find_s_l72_7222

-- Define the roots of the quadratic equation
variables (a b n r s : ℝ)

-- Conditions from Vieta's formulas
def condition1 : Prop := a + b = n
def condition2 : Prop := a * b = 3

-- Roots of the second quadratic equation
def condition3 : Prop := (a + 1 / b) * (b + 1 / a) = s

-- The theorem statement
theorem find_s
  (h1 : condition1 a b n)
  (h2 : condition2 a b)
  (h3 : condition3 a b s) :
  s = 16 / 3 :=
by
  sorry

end NUMINAMATH_GPT_find_s_l72_7222


namespace NUMINAMATH_GPT_number_of_solutions_in_positive_integers_l72_7232

theorem number_of_solutions_in_positive_integers (x y : ℕ) (h1 : 3 * x + 4 * y = 806) : 
  ∃ n : ℕ, n = 67 := 
sorry

end NUMINAMATH_GPT_number_of_solutions_in_positive_integers_l72_7232


namespace NUMINAMATH_GPT_product_of_two_numbers_eq_a_mul_100_a_l72_7214

def product_of_two_numbers (a : ℝ) (b : ℝ) : ℝ := a * b

theorem product_of_two_numbers_eq_a_mul_100_a (a : ℝ) (b : ℝ) (h : a + b = 100) :
    product_of_two_numbers a b = a * (100 - a) :=
by
  sorry

end NUMINAMATH_GPT_product_of_two_numbers_eq_a_mul_100_a_l72_7214


namespace NUMINAMATH_GPT_ratio_of_times_l72_7270

theorem ratio_of_times (A_work_time B_combined_rate : ℕ) 
  (h1 : A_work_time = 6) 
  (h2 : (1 / (1 / A_work_time + 1 / (B_combined_rate / 2))) = 2) :
  (B_combined_rate : ℝ) / A_work_time = 1 / 2 :=
by
  -- below we add the proof part which we will skip for now with sorry.
  sorry

end NUMINAMATH_GPT_ratio_of_times_l72_7270


namespace NUMINAMATH_GPT_books_total_l72_7275

theorem books_total (J T : ℕ) (hJ : J = 10) (hT : T = 38) : J + T = 48 :=
by {
  sorry
}

end NUMINAMATH_GPT_books_total_l72_7275


namespace NUMINAMATH_GPT_handshakesCountIsCorrect_l72_7273

-- Define the number of gremlins and imps
def numGremlins : ℕ := 30
def numImps : ℕ := 20

-- Define the conditions based on the problem
def handshakesAmongGremlins : ℕ := (numGremlins * (numGremlins - 1)) / 2
def handshakesBetweenImpsAndGremlins : ℕ := numImps * numGremlins

-- Calculate the total handshakes
def totalHandshakes : ℕ := handshakesAmongGremlins + handshakesBetweenImpsAndGremlins

-- Prove that the total number of handshakes equals 1035
theorem handshakesCountIsCorrect : totalHandshakes = 1035 := by
  sorry

end NUMINAMATH_GPT_handshakesCountIsCorrect_l72_7273


namespace NUMINAMATH_GPT_sum_of_coefficients_l72_7267

theorem sum_of_coefficients (a : ℕ → ℤ) (x : ℂ) :
  (2*x - 1)^10 = a 0 + a 1 * x + a 2 * x^2 + a 3 * x^3 + a 4 * x^4 + 
  a 5 * x^5 + a 6 * x^6 + a 7 * x^7 + a 8 * x^8 + a 9 * x^9 + a 10 * x^10 →
  a 0 = 1 →
  a 2 + a 3 + a 4 + a 5 + a 6 + a 7 + a 8 + a 9 + a 10 = 20 :=
sorry

end NUMINAMATH_GPT_sum_of_coefficients_l72_7267


namespace NUMINAMATH_GPT_mod_6_computation_l72_7298

theorem mod_6_computation (a b n : ℕ) (h₁ : a ≡ 35 [MOD 6]) (h₂ : b ≡ 16 [MOD 6]) (h₃ : n = 1723) :
  (a ^ n - b ^ n) % 6 = 1 :=
by 
  -- proofs go here
  sorry

end NUMINAMATH_GPT_mod_6_computation_l72_7298


namespace NUMINAMATH_GPT_g_5_l72_7266

variable (g : ℝ → ℝ)

axiom additivity_condition : ∀ (x y : ℝ), g (x + y) = g x + g y
axiom g_1_nonzero : g 1 ≠ 0

theorem g_5 : g 5 = 5 * g 1 :=
by
  sorry

end NUMINAMATH_GPT_g_5_l72_7266


namespace NUMINAMATH_GPT_find_numbers_l72_7249

theorem find_numbers (a b : ℝ) (h₁ : a - b = 157) (h₂ : a / b = 2) : a = 314 ∧ b = 157 :=
sorry

end NUMINAMATH_GPT_find_numbers_l72_7249


namespace NUMINAMATH_GPT_bottle_caps_per_box_l72_7213

theorem bottle_caps_per_box (total_caps : ℕ) (total_boxes : ℕ) (h_total_caps : total_caps = 60) (h_total_boxes : total_boxes = 60) :
  (total_caps / total_boxes) = 1 :=
by {
  sorry
}

end NUMINAMATH_GPT_bottle_caps_per_box_l72_7213


namespace NUMINAMATH_GPT_one_eighth_of_two_power_36_equals_two_power_x_l72_7291

theorem one_eighth_of_two_power_36_equals_two_power_x (x : ℕ) :
  (1 / 8) * (2 : ℝ) ^ 36 = (2 : ℝ) ^ x → x = 33 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_one_eighth_of_two_power_36_equals_two_power_x_l72_7291


namespace NUMINAMATH_GPT_sum_even_odd_probability_l72_7283

theorem sum_even_odd_probability :
  (∀ (a b : ℕ), ∃ (P_even P_odd : ℚ),
    P_even = 1/2 ∧ P_odd = 1/2 ∧
    (a % 2 = 0 ∧ b % 2 = 0 ↔ (a + b) % 2 = 0) ∧
    (a % 2 = 1 ∧ b % 2 = 1 ↔ (a + b) % 2 = 0) ∧
    ((a % 2 = 0 ∧ b % 2 = 1) ∨ (a % 2 = 1 ∧ b % 2 = 0) ↔ (a + b) % 2 = 1)) :=
sorry

end NUMINAMATH_GPT_sum_even_odd_probability_l72_7283


namespace NUMINAMATH_GPT_total_cost_sandwiches_and_sodas_l72_7243

theorem total_cost_sandwiches_and_sodas :
  let price_sandwich : Real := 2.49
  let price_soda : Real := 1.87
  let quantity_sandwich : ℕ := 2
  let quantity_soda : ℕ := 4
  (quantity_sandwich * price_sandwich + quantity_soda * price_soda) = 12.46 := 
by
  sorry

end NUMINAMATH_GPT_total_cost_sandwiches_and_sodas_l72_7243


namespace NUMINAMATH_GPT_field_area_l72_7224

theorem field_area (x y : ℕ) (h1 : x + y = 700) (h2 : y - x = (1/5) * ((x + y) / 2)) : x = 315 :=
  sorry

end NUMINAMATH_GPT_field_area_l72_7224


namespace NUMINAMATH_GPT_largest_prime_factor_of_18_pow_3_plus_15_pow_4_minus_3_pow_7_is_19_l72_7248

theorem largest_prime_factor_of_18_pow_3_plus_15_pow_4_minus_3_pow_7_is_19 : 
  ∃ p : ℕ, Prime p ∧ p = 19 ∧ ∀ q : ℕ, Prime q → q ∣ (18^3 + 15^4 - 3^7) → q ≤ 19 :=
sorry

end NUMINAMATH_GPT_largest_prime_factor_of_18_pow_3_plus_15_pow_4_minus_3_pow_7_is_19_l72_7248


namespace NUMINAMATH_GPT_three_digit_numbers_with_distinct_digits_avg_condition_l72_7289

theorem three_digit_numbers_with_distinct_digits_avg_condition : 
  ∃ (S : Finset (Fin 1000)), 
  (∀ n ∈ S, (n / 100 ≠ (n / 10 % 10) ∧ (n / 100 ≠ n % 10) ∧ (n / 10 % 10 ≠ n % 10))) ∧
  (∀ n ∈ S, ((n / 100 + n % 10) / 2 = n / 10 % 10)) ∧
  (∀ n ∈ S, abs ((n / 100) - (n / 10 % 10)) ≤ 5 ∧ abs ((n / 10 % 10) - (n % 10)) ≤ 5) ∧
  S.card = 120 :=
sorry

end NUMINAMATH_GPT_three_digit_numbers_with_distinct_digits_avg_condition_l72_7289


namespace NUMINAMATH_GPT_necessary_but_not_sufficient_l72_7212

-- Definitions from conditions
def abs_gt_2 (x : ℝ) : Prop := |x| > 2
def x_lt_neg_2 (x : ℝ) : Prop := x < -2

-- Statement to prove
theorem necessary_but_not_sufficient : 
  ∀ x : ℝ, (abs_gt_2 x → x_lt_neg_2 x) ∧ (¬(x_lt_neg_2 x → abs_gt_2 x)) := 
by 
  sorry

end NUMINAMATH_GPT_necessary_but_not_sufficient_l72_7212


namespace NUMINAMATH_GPT_a5_b5_equals_c5_d5_cannot_conclude_a4_b4_equals_c4_d4_l72_7201

-- Definitions of assumptions
variables (a b c d : ℝ)
axiom h1 : a + b = c + d
axiom h2 : a^3 + b^3 = c^3 + d^3

-- Statement for part (a)
theorem a5_b5_equals_c5_d5 : a^5 + b^5 = c^5 + d^5 :=
by sorry

-- Statement for part (b), we need to state that we cannot conclude a^4 + b^4 = c^4 + d^4 under given conditions
theorem cannot_conclude_a4_b4_equals_c4_d4 : ¬ (a^4 + b^4 = c^4 + d^4) :=
by sorry

end NUMINAMATH_GPT_a5_b5_equals_c5_d5_cannot_conclude_a4_b4_equals_c4_d4_l72_7201


namespace NUMINAMATH_GPT_problem1_problem2_l72_7287

-- Problem 1: Prove the solution set of the given inequality
theorem problem1 (x : ℝ) : (|x - 2| + 2 * |x - 1| > 5) ↔ (x < -1/3 ∨ x > 3) := 
sorry

-- Problem 2: Prove the range of values for 'a' such that the inequality holds
theorem problem2 (a : ℝ) : (∃ x : ℝ, |x - a| + |x - 1| ≤ |a - 2|) ↔ (a ≤ 3/2) :=
sorry

end NUMINAMATH_GPT_problem1_problem2_l72_7287


namespace NUMINAMATH_GPT_problem1_problem2_problem3_problem4_l72_7204

variable (f : ℝ → ℝ)
variables (H1 : f (-1) = 2) 
          (H2 : ∀ x, x < 0 → f x > 1)
          (H3 : ∀ x y, f (x + y) = f x * f y)

-- (1) Prove f(0) = 1
theorem problem1 : f 0 = 1 := sorry

-- (2) Prove f(-4) = 16
theorem problem2 : f (-4) = 16 := sorry

-- (3) Prove f(x) is strictly decreasing
theorem problem3 : ∀ x y, x < y → f x > f y := sorry

-- (4) Solve f(-4x^2)f(10x) ≥ 1/16
theorem problem4 : { x : ℝ | f (-4 * x ^ 2) * f (10 * x) ≥ 1 / 16 } = { x | x ≤ 1 / 2 ∨ 2 ≤ x } := sorry

end NUMINAMATH_GPT_problem1_problem2_problem3_problem4_l72_7204


namespace NUMINAMATH_GPT_find_radius_l72_7236

-- Definitions and conditions
variables (M N r : ℝ) (h1 : M = π * r^2) (h2 : N = 2 * π * r) (h3 : M / N = 25)

-- Theorem statement
theorem find_radius : r = 50 :=
sorry

end NUMINAMATH_GPT_find_radius_l72_7236


namespace NUMINAMATH_GPT_distance_between_trains_l72_7229

theorem distance_between_trains (d1 d2 : ℝ) (t1 t2 : ℝ) (s1 s2 : ℝ) (x : ℝ) :
  d1 = d2 + 100 →
  s1 = 50 →
  s2 = 40 →
  d1 = s1 * t1 →
  d2 = s2 * t2 →
  t1 = t2 →
  d2 = 400 →
  d1 + d2 = 900 :=
by
  intros h1 h2 h3 h4 h5 h6 h7
  sorry

end NUMINAMATH_GPT_distance_between_trains_l72_7229


namespace NUMINAMATH_GPT_greatest_odd_integer_x_l72_7258

theorem greatest_odd_integer_x (x : ℕ) (h1 : x % 2 = 1) (h2 : x^4 / x^2 < 50) : x ≤ 7 :=
sorry

end NUMINAMATH_GPT_greatest_odd_integer_x_l72_7258


namespace NUMINAMATH_GPT_arithmetic_sequence_value_l72_7257

theorem arithmetic_sequence_value 
    (a1 : ℤ) (a2 a3 a4 : ℤ) (a1_a4 : a1 = 18) 
    (b1 b2 b3 : ℤ) 
    (b1_b3 : b3 - b2 = 6 ∧ b2 - b1 = 6 ∧ b2 = 15 ∧ b3 = 21)
    (b1_a3 : a3 = b1 - 6 ∧ a4 = a1 + (a3 - 18) / 3) 
    (c1 c2 c3 c4 : ℝ) 
    (c1_b3 : c1 = a4) 
    (c2 : c2 = -14) 
    (c4 : ∃ m, c4 = b1 - m * (6 :ℝ) + - 0.5) 
    (n : ℝ) : 
    n = -12.5 := by 
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_value_l72_7257


namespace NUMINAMATH_GPT_side_length_of_square_with_circles_l72_7228

noncomputable def side_length_of_square (radius : ℝ) : ℝ :=
  2 * radius + 2 * radius

theorem side_length_of_square_with_circles 
  (radius : ℝ) 
  (h_radius : radius = 2) 
  (h_tangent : ∀ (P Q : ℝ), P = Q + 2 * radius) :
  side_length_of_square radius = 8 :=
by
  sorry

end NUMINAMATH_GPT_side_length_of_square_with_circles_l72_7228


namespace NUMINAMATH_GPT_shortest_chord_length_l72_7276

/-- The shortest chord passing through point D given the conditions provided. -/
theorem shortest_chord_length
  (O : Point) (D : Point) (r : ℝ) (OD : ℝ)
  (h_or : r = 5) (h_od : OD = 3) :
  ∃ (AB : ℝ), AB = 8 := 
  sorry

end NUMINAMATH_GPT_shortest_chord_length_l72_7276


namespace NUMINAMATH_GPT_union_of_sets_l72_7279

-- Defining the sets A and B
def A : Set ℕ := {1, 3, 6}
def B : Set ℕ := {1, 2}

-- The theorem we want to prove
theorem union_of_sets : A ∪ B = {1, 2, 3, 6} := by
  sorry

end NUMINAMATH_GPT_union_of_sets_l72_7279


namespace NUMINAMATH_GPT_complement_union_correct_l72_7223

-- Defining the sets U, A, and B
def U : Set ℕ := {1, 2, 3, 4}
def A : Set ℕ := {1, 2}
def B : Set ℕ := {2, 3}

-- Theorem statement
theorem complement_union_correct : (U \ A) ∪ B = {2, 3, 4} := by
  sorry

end NUMINAMATH_GPT_complement_union_correct_l72_7223


namespace NUMINAMATH_GPT_sum_mod_condition_l72_7227

theorem sum_mod_condition (a b c : ℤ) (h1 : a * b * c % 7 = 2)
                          (h2 : 3 * c % 7 = 1)
                          (h3 : 4 * b % 7 = (2 + b) % 7) :
                          (a + b + c) % 7 = 3 := by
  sorry

end NUMINAMATH_GPT_sum_mod_condition_l72_7227


namespace NUMINAMATH_GPT_substitution_modulo_l72_7271

-- Definitions based on conditions
def total_players := 15
def starting_lineup := 10
def substitutes := 5
def max_substitutions := 2

-- Define the number of substitutions ways for the cases 0, 1, and 2 substitutions
def a_0 := 1
def a_1 := starting_lineup * substitutes
def a_2 := starting_lineup * substitutes * (starting_lineup - 1) * (substitutes - 1)

-- Summing the total number of substitution scenarios
def total_substitution_scenarios := a_0 + a_1 + a_2

-- Theorem statement to verify the result modulo 500
theorem substitution_modulo : total_substitution_scenarios % 500 = 351 := by
  sorry

end NUMINAMATH_GPT_substitution_modulo_l72_7271


namespace NUMINAMATH_GPT_decreasing_condition_l72_7209

noncomputable def f (a x : ℝ) : ℝ := x / (Real.log x) + a * x

theorem decreasing_condition (a : ℝ) :
  (∀ x > 1, (Real.log x - 1) / (Real.log x)^2 + a ≤ 0) → a ≤ -1/4 := by
  sorry

end NUMINAMATH_GPT_decreasing_condition_l72_7209


namespace NUMINAMATH_GPT_factorial_square_product_l72_7290

theorem factorial_square_product : (Real.sqrt (Nat.factorial 6 * Nat.factorial 4)) ^ 2 = 17280 := by
  sorry

end NUMINAMATH_GPT_factorial_square_product_l72_7290


namespace NUMINAMATH_GPT_problems_per_page_l72_7281

def total_problems : ℕ := 60
def finished_problems : ℕ := 20
def remaining_pages : ℕ := 5

theorem problems_per_page :
  (total_problems - finished_problems) / remaining_pages = 8 :=
by
  sorry

end NUMINAMATH_GPT_problems_per_page_l72_7281


namespace NUMINAMATH_GPT_amount_pop_spend_l72_7219

theorem amount_pop_spend
  (total_spent : ℝ)
  (ratio_snap_crackle : ℝ)
  (ratio_crackle_pop : ℝ)
  (spending_eq : total_spent = 150)
  (snap_crackle : ratio_snap_crackle = 2)
  (crackle_pop : ratio_crackle_pop = 3)
  (snap : ℝ)
  (crackle : ℝ)
  (pop : ℝ)
  (snap_eq : snap = ratio_snap_crackle * crackle)
  (crackle_eq : crackle = ratio_crackle_pop * pop)
  (total_eq : snap + crackle + pop = total_spent) :
  pop = 15 := 
by
  sorry

end NUMINAMATH_GPT_amount_pop_spend_l72_7219


namespace NUMINAMATH_GPT_reduced_price_per_kg_of_oil_l72_7288

theorem reduced_price_per_kg_of_oil
  (P : ℝ)
  (h : (1000 / (0.75 * P) - 1000 / P = 5)) :
  0.75 * (1000 / 15) = 50 := 
sorry

end NUMINAMATH_GPT_reduced_price_per_kg_of_oil_l72_7288


namespace NUMINAMATH_GPT_greatest_int_with_gcd_3_l72_7215

theorem greatest_int_with_gcd_3 (n : ℕ) (h1 : n < 150) (h2 : Int.gcd n 24 = 3) : n = 141 := by
  sorry

end NUMINAMATH_GPT_greatest_int_with_gcd_3_l72_7215


namespace NUMINAMATH_GPT_no_such_polynomials_exists_l72_7294

theorem no_such_polynomials_exists :
  ¬ ∃ (f g : Polynomial ℚ), (∀ x y : ℚ, f.eval x * g.eval y = x^200 * y^200 + 1) := 
by 
  sorry

end NUMINAMATH_GPT_no_such_polynomials_exists_l72_7294


namespace NUMINAMATH_GPT_complex_number_z_value_l72_7225

open Complex

theorem complex_number_z_value :
  ∀ (i z : ℂ), i^2 = -1 ∧ z * (1 + i) = 2 * i^2018 → z = -1 + i :=
by
  intros i z h
  have h1 : i^2 = -1 := h.1
  have h2 : z * (1 + i) = 2 * i^2018 := h.2
  sorry

end NUMINAMATH_GPT_complex_number_z_value_l72_7225


namespace NUMINAMATH_GPT_rational_number_theorem_l72_7208

theorem rational_number_theorem (x y : ℚ) 
  (h1 : |(x + 2017 : ℚ)| + (y - 2017) ^ 2 = 0) : 
  (x / y) ^ 2017 = -1 := 
by
  sorry

end NUMINAMATH_GPT_rational_number_theorem_l72_7208


namespace NUMINAMATH_GPT_f_one_eq_minus_one_third_f_of_a_f_is_odd_l72_7277

noncomputable def f (x : ℝ) : ℝ := (1 - 2^x) / (2^x + 1)

theorem f_one_eq_minus_one_third : f 1 = -1/3 := 
by sorry

theorem f_of_a (a : ℝ) : f a = (1 - 2^a) / (2^a + 1) := 
by sorry

theorem f_is_odd : ∀ x, f (-x) = -f x := by sorry

end NUMINAMATH_GPT_f_one_eq_minus_one_third_f_of_a_f_is_odd_l72_7277
