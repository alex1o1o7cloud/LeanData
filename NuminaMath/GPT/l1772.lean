import Mathlib

namespace three_digit_numbers_with_4_and_5_correct_l1772_177295

def count_three_digit_numbers_with_4_and_5 : ℕ :=
  48

theorem three_digit_numbers_with_4_and_5_correct :
  count_three_digit_numbers_with_4_and_5 = 48 :=
by
  sorry -- proof goes here

end three_digit_numbers_with_4_and_5_correct_l1772_177295


namespace necessary_not_sufficient_l1772_177286

theorem necessary_not_sufficient (a b : ℝ) : (a > b - 1) ∧ ¬ (a > b - 1 → a > b) := 
sorry

end necessary_not_sufficient_l1772_177286


namespace man_is_26_years_older_l1772_177213

variable (S : ℕ) (M : ℕ)

-- conditions
def present_age_of_son : Prop := S = 24
def future_age_relation : Prop := M + 2 = 2 * (S + 2)

-- question transformed to a proof problem
theorem man_is_26_years_older
  (h1 : present_age_of_son S)
  (h2 : future_age_relation S M) : M - S = 26 := by
  sorry

end man_is_26_years_older_l1772_177213


namespace rationalize_denominator_l1772_177283

theorem rationalize_denominator :
  (2 + Real.sqrt 5) / (2 - Real.sqrt 5) = -9 - 4 * Real.sqrt 5 :=
by
  -- Commutative field properties and algebraic manipulation will be used here.
  sorry

end rationalize_denominator_l1772_177283


namespace polynomial_proof_l1772_177247

variable (a b : ℝ)

-- Define the given monomial and the resulting polynomial 
def monomial := -3 * a ^ 2 * b
def result := 6 * a ^ 3 * b ^ 2 - 3 * a ^ 2 * b ^ 2 + 9 * a ^ 2 * b

-- Define the polynomial we want to prove
def poly := -2 * a * b + b - 3

-- Statement of the problem in Lean 4
theorem polynomial_proof :
  monomial * poly = result :=
by sorry

end polynomial_proof_l1772_177247


namespace quadratic_inequality_solution_l1772_177268

theorem quadratic_inequality_solution (m : ℝ) : 
  (∀ x : ℝ, x^2 + m * x + 1 ≥ 0) ↔ (-2 ≤ m ∧ m ≤ 2) :=
by
  sorry

end quadratic_inequality_solution_l1772_177268


namespace trig_identity_l1772_177260

theorem trig_identity (x : ℝ) (h : Real.sin (π / 6 - x) = 1 / 2) :
  Real.sin (19 * π / 6 - x) + Real.sin (-2 * π / 3 + x) ^ 2 = 1 / 4 :=
by
  sorry

end trig_identity_l1772_177260


namespace minimum_degree_of_g_l1772_177206

noncomputable def f : Polynomial ℝ := sorry
noncomputable def g : Polynomial ℝ := sorry
noncomputable def h : Polynomial ℝ := sorry

theorem minimum_degree_of_g :
  (5 * f - 3 * g = h) →
  (Polynomial.degree f = 10) →
  (Polynomial.degree h = 11) →
  (Polynomial.degree g = 11) :=
sorry

end minimum_degree_of_g_l1772_177206


namespace kitten_length_l1772_177263

theorem kitten_length (initial_length : ℕ) (doubled_length_1 : ℕ) (doubled_length_2 : ℕ) :
  initial_length = 4 →
  doubled_length_1 = 2 * initial_length →
  doubled_length_2 = 2 * doubled_length_1 →
  doubled_length_2 = 16 :=
by
  intros h1 h2 h3
  rw [h1] at h2
  rw [h2] at h3
  exact h3

end kitten_length_l1772_177263


namespace circumference_of_circle_of_given_area_l1772_177280

theorem circumference_of_circle_of_given_area (A : ℝ) (h : A = 225 * Real.pi) : 
  ∃ C : ℝ, C = 2 * Real.pi * 15 :=
by
  let r := 15
  let C := 2 * Real.pi * r
  use C
  sorry

end circumference_of_circle_of_given_area_l1772_177280


namespace find_judy_rotation_l1772_177289

-- Definition of the problem
def CarlaRotation := 480 % 360 -- This effectively becomes 120
def JudyRotation (y : ℕ) := (360 - 120) % 360 -- This should effectively be 240

-- Theorem stating the problem and solution
theorem find_judy_rotation (y : ℕ) (h : y < 360) : 360 - CarlaRotation = y :=
by 
  dsimp [CarlaRotation, JudyRotation] 
  sorry

end find_judy_rotation_l1772_177289


namespace chuck_vs_dave_ride_time_l1772_177293

theorem chuck_vs_dave_ride_time (D E : ℕ) (h1 : D = 10) (h2 : E = 65) (h3 : E = 13 * C / 10) :
  (C / D = 5) :=
by
  sorry

end chuck_vs_dave_ride_time_l1772_177293


namespace mauve_red_paint_parts_l1772_177253

noncomputable def parts_of_red_in_mauve : ℕ :=
let fuchsia_red_ratio := 5
let fuchsia_blue_ratio := 3
let total_fuchsia := 16
let added_blue := 14
let mauve_blue_ratio := 6

let total_fuchsia_parts := fuchsia_red_ratio + fuchsia_blue_ratio
let red_in_fuchsia := (fuchsia_red_ratio * total_fuchsia) / total_fuchsia_parts
let blue_in_fuchsia := (fuchsia_blue_ratio * total_fuchsia) / total_fuchsia_parts
let blue_in_mauve := blue_in_fuchsia + added_blue
let ratio_red_to_blue_in_mauve := red_in_fuchsia / blue_in_mauve
ratio_red_to_blue_in_mauve * mauve_blue_ratio

theorem mauve_red_paint_parts : parts_of_red_in_mauve = 3 :=
by sorry

end mauve_red_paint_parts_l1772_177253


namespace sum_of_cubes_l1772_177277

theorem sum_of_cubes (x y : ℝ) (h1 : x + y = 11) (h2 : x * y = 12) : x^3 + y^3 = 935 := by
  sorry

end sum_of_cubes_l1772_177277


namespace sum_of_x_coordinates_of_intersections_l1772_177273

def g : ℝ → ℝ := sorry  -- Definition of g is unspecified but it consists of five line segments.

theorem sum_of_x_coordinates_of_intersections 
  (h1 : ∃ x1, g x1 = x1 - 2 ∧ (x1 = -2 ∨ x1 = 1 ∨ x1 = 4))
  (h2 : ∃ x2, g x2 = x2 - 2 ∧ (x2 = -2 ∨ x2 = 1 ∨ x2 = 4))
  (h3 : ∃ x3, g x3 = x3 - 2 ∧ (x3 = -2 ∨ x3 = 1 ∨ x3 = 4)) 
  (hx1x2x3 : x1 ≠ x2 ∧ x2 ≠ x3 ∧ x1 ≠ x3) :
  x1 + x2 + x3 = 3 := by
  -- Proof here
  sorry

end sum_of_x_coordinates_of_intersections_l1772_177273


namespace fraction_not_integer_l1772_177200

theorem fraction_not_integer (a b : ℕ) (h : a ≠ b) (parity: (a % 2 = b % 2)) 
(h_pos_a : 0 < a) (h_pos_b : 0 < b) : ¬ ∃ k : ℕ, (a! + b!) = k * 2^a := 
by sorry

end fraction_not_integer_l1772_177200


namespace unique_polynomial_P_l1772_177211

noncomputable def P : ℝ → ℝ := sorry

axiom P_func_eq (x : ℝ) : P (x^2 + 1) = P x ^ 2 + 1
axiom P_zero : P 0 = 0

theorem unique_polynomial_P (x : ℝ) : P x = x :=
by
  sorry

end unique_polynomial_P_l1772_177211


namespace unique_solution_fraction_l1772_177299

theorem unique_solution_fraction (x : ℝ) :
  (2 * x^2 - 10 * x + 8 ≠ 0) → 
  (∃! (x : ℝ), (3 * x^2 - 15 * x + 12) / (2 * x^2 - 10 * x + 8) = x - 4) :=
by
  sorry

end unique_solution_fraction_l1772_177299


namespace find_x_l1772_177201

theorem find_x (p q x : ℚ) (h1 : p / q = 4 / 5)
    (h2 : 4 / 7 + x / (2 * q + p) = 1) : x = 12 := 
by
  sorry

end find_x_l1772_177201


namespace average_speed_l1772_177262

-- Define the conditions
def initial_reading : ℕ := 2552
def final_reading : ℕ := 2992
def day1_time : ℕ := 6
def day2_time : ℕ := 8

-- Theorem: Proving the average speed is 31 miles per hour.
theorem average_speed :
  final_reading - initial_reading = 440 ∧ day1_time + day2_time = 14 ∧ 
  (final_reading - initial_reading) / (day1_time + day2_time) = 31 :=
by
  sorry

end average_speed_l1772_177262


namespace eugene_total_pencils_l1772_177250

-- Define the initial number of pencils Eugene has
def initial_pencils : ℕ := 51

-- Define the number of pencils Joyce gives to Eugene
def pencils_from_joyce : ℕ := 6

-- Define the expected total number of pencils
def expected_total_pencils : ℕ := 57

-- Theorem to prove the total number of pencils Eugene has
theorem eugene_total_pencils : initial_pencils + pencils_from_joyce = expected_total_pencils := 
by sorry

end eugene_total_pencils_l1772_177250


namespace terminal_side_second_or_third_quadrant_l1772_177236

-- Definitions and conditions directly from part a)
def sin (x : ℝ) : ℝ := sorry
def tan (x : ℝ) : ℝ := sorry
def terminal_side_in_quadrant (x : ℝ) (q : ℕ) : Prop := sorry

-- Proving the mathematically equivalent proof
theorem terminal_side_second_or_third_quadrant (x : ℝ) :
  sin x * tan x < 0 →
  (terminal_side_in_quadrant x 2 ∨ terminal_side_in_quadrant x 3) :=
by
  sorry

end terminal_side_second_or_third_quadrant_l1772_177236


namespace trapezoid_area_ratio_l1772_177290

theorem trapezoid_area_ratio (b h x : ℝ) 
  (base_relation : b + 150 = x)
  (area_ratio : (3 / 7) * h * (b + 75) = (1 / 2) * h * (b + x))
  (mid_segment : x = b + 150) 
  : ⌊x^3 / 1000⌋ = 142 :=
by
  sorry

end trapezoid_area_ratio_l1772_177290


namespace find_number_l1772_177276

theorem find_number (x : ℝ) (h : (x / 6) * 12 = 9) : x = 4.5 :=
by
  sorry

end find_number_l1772_177276


namespace intersection_of_S_and_T_l1772_177223

def S : Set ℝ := {x | x^2 - x ≥ 0}
def T : Set ℝ := {x | 0 < x}

theorem intersection_of_S_and_T : S ∩ T = {x | 1 ≤ x} := by
  sorry

end intersection_of_S_and_T_l1772_177223


namespace gym_hours_tuesday_equals_friday_l1772_177244

-- Definitions
def weekly_gym_hours : ℝ := 5
def monday_hours : ℝ := 1.5
def wednesday_hours : ℝ := 1.5
def friday_hours : ℝ := 1
def total_weekly_hours : ℝ := weekly_gym_hours - (monday_hours + wednesday_hours + friday_hours)

-- Theorem statement
theorem gym_hours_tuesday_equals_friday : 
  total_weekly_hours = friday_hours :=
by
  sorry

end gym_hours_tuesday_equals_friday_l1772_177244


namespace arthur_speed_l1772_177232

/-- Suppose Arthur drives to David's house and aims to arrive exactly on time. 
If he drives at 60 km/h, he arrives 5 minutes late. 
If he drives at 90 km/h, he arrives 5 minutes early. 
We want to find the speed n in km/h at which he arrives exactly on time. -/
theorem arthur_speed (n : ℕ) :
  (∀ t, 1 * (t + 5) = (3 / 2) * (t - 5)) → 
  (60 : ℝ) = 1 →
  (90 : ℝ) = (3 / 2) → 
  n = 72 := by
sorry

end arthur_speed_l1772_177232


namespace inequality_condition_l1772_177205

theorem inequality_condition (x y a : ℝ) (h1 : x < y) (h2 : a * x < a * y) : a > 0 :=
sorry

end inequality_condition_l1772_177205


namespace rational_b_if_rational_a_l1772_177240

theorem rational_b_if_rational_a (x : ℚ) (h_rational : ∃ a : ℚ, a = x / (x^2 - x + 1)) :
  ∃ b : ℚ, b = x^2 / (x^4 - x^2 + 1) :=
by
  sorry

end rational_b_if_rational_a_l1772_177240


namespace ellipse_problem_l1772_177229

theorem ellipse_problem
  (F2 : ℝ) (a : ℝ) (A B : ℝ × ℝ)
  (on_ellipse_A : (A.1 ^ 2) / (a ^ 2) + (25 * (A.2 ^ 2)) / (9 * a ^ 2) = 1)
  (on_ellipse_B : (B.1 ^ 2) / (a ^ 2) + (25 * (B.2 ^ 2)) / (9 * a ^ 2) = 1)
  (focal_distance : |A.1 + F2| + |B.1 + F2| = 8 / 5 * a)
  (midpoint_to_directrix : |(A.1 + B.1) / 2 + 5 / 4 * a| = 3 / 2) :
  a = 1 → (∀ x y, (x^2 + (25 / 9) * y^2 = 1) ↔ ((x^2) / (a^2) + (25 * y^2) / (9 * a^2) = 1)) :=
by
  sorry

end ellipse_problem_l1772_177229


namespace probability_margo_pairing_l1772_177230

-- Definition of the problem
def num_students : ℕ := 32
def num_pairings (n : ℕ) : ℕ := n - 1
def favorable_pairings : ℕ := 2

-- Theorem statement
theorem probability_margo_pairing :
  num_students = 32 →
  ∃ (p : ℚ), p = favorable_pairings / num_pairings num_students ∧ p = 2/31 :=
by
  intros h
  -- The proofs are omitted for brevity.
  sorry

end probability_margo_pairing_l1772_177230


namespace find_y_value_l1772_177275

theorem find_y_value : (12 ^ 2 * 6 ^ 4) / 432 = 432 := by
  sorry

end find_y_value_l1772_177275


namespace ronaldo_current_age_l1772_177219

noncomputable def roonie_age_one_year_ago (R L : ℕ) := 6 * L / 7
noncomputable def new_ratio (R L : ℕ) := (R + 5) * 8 = 7 * (L + 5)

theorem ronaldo_current_age (R L : ℕ) 
  (h1 : R = roonie_age_one_year_ago R L)
  (h2 : new_ratio R L) : L + 1 = 36 :=
by
  sorry

end ronaldo_current_age_l1772_177219


namespace negation_of_exists_solution_l1772_177255

theorem negation_of_exists_solution :
  ¬ (∃ c : ℝ, c > 0 ∧ ∃ x : ℝ, x^2 - x + c = 0) ↔ ∀ c : ℝ, c > 0 → ¬ (∃ x : ℝ, x^2 - x + c = 0) :=
by
  sorry

end negation_of_exists_solution_l1772_177255


namespace power_of_fraction_l1772_177251

theorem power_of_fraction : ((1/3)^5 = (1/243)) :=
by
  sorry

end power_of_fraction_l1772_177251


namespace playground_ratio_l1772_177252

theorem playground_ratio (L B : ℕ) (playground_area landscape_area : ℕ) 
  (h1 : B = 8 * L)
  (h2 : B = 480)
  (h3 : playground_area = 3200)
  (h4 : landscape_area = L * B) : 
  (playground_area : ℚ) / landscape_area = 1 / 9 :=
by
  sorry

end playground_ratio_l1772_177252


namespace number_of_cows_l1772_177271

-- Definitions
variables (a g e c : ℕ)
variables (six_two : 6 * e = 2 * a + 4 * g) (eight_two : 8 * e = 2 * a + 8 * g)

-- Theorem statement
theorem number_of_cows (a g e : ℕ) (six_two : 6 * e = 2 * a + 4 * g) (eight_two : 8 * e = 2 * a + 8 * g) :
  ∃ c : ℕ, c * e * 6 = 6 * a + 36 * g ∧ c = 5 :=
by
  sorry

end number_of_cows_l1772_177271


namespace find_y_l1772_177231

theorem find_y (y: ℕ) (h1: y > 0) (h2: y ≤ 100)
  (h3: (43 + 69 + 87 + y + y) / 5 = 2 * y): 
  y = 25 :=
sorry

end find_y_l1772_177231


namespace one_head_two_tails_probability_l1772_177266

noncomputable def probability_of_one_head_two_tails :=
  let total_outcomes := 8
  let favorable_outcomes := 3
  favorable_outcomes / total_outcomes

theorem one_head_two_tails_probability :
  probability_of_one_head_two_tails = 3 / 8 :=
by
  -- Proof would go here
  sorry

end one_head_two_tails_probability_l1772_177266


namespace coprime_divisors_property_l1772_177204

theorem coprime_divisors_property (n : ℕ) 
  (h : ∀ a b : ℕ, a ∣ n → b ∣ n → gcd a b = 1 → (a + b - 1) ∣ n) : 
  (∃ k : ℕ, ∃ p : ℕ, Nat.Prime p ∧ n = p ^ k) ∨ (n = 12) :=
sorry

end coprime_divisors_property_l1772_177204


namespace real_root_bound_l1772_177226

noncomputable def P (x : ℝ) (n : ℕ) (ns : List ℕ) : ℝ :=
  1 + x^2 + x^5 + ns.foldr (λ n acc => x^n + acc) 0 + x^2008

theorem real_root_bound (n1 n2 : ℕ) (ns : List ℕ) (x : ℝ) :
  5 < n1 →
  List.Chain (λ a b => a < b) n1 (n2 :: ns) →
  n2 < 2008 →
  P x n1 (n2 :: ns) = 0 →
  x ≤ (1 - Real.sqrt 5) / 2 :=
sorry

end real_root_bound_l1772_177226


namespace exists_sum_pair_l1772_177272

theorem exists_sum_pair (n : ℕ) (a b : List ℕ) (h₁ : ∀ x ∈ a, x < n) (h₂ : ∀ y ∈ b, y < n) 
  (h₃ : List.Nodup a) (h₄ : List.Nodup b) (h₅ : a.length + b.length ≥ n) : ∃ x ∈ a, ∃ y ∈ b, x + y = n := by
  sorry

end exists_sum_pair_l1772_177272


namespace jenna_bill_eel_ratio_l1772_177212

theorem jenna_bill_eel_ratio:
  ∀ (B : ℕ), (B + 16 = 64) → (16 / B = 1 / 3) :=
by
  intros B h
  sorry

end jenna_bill_eel_ratio_l1772_177212


namespace ruiz_original_salary_l1772_177233

theorem ruiz_original_salary (S : ℝ) (h : 1.06 * S = 530) : S = 500 :=
by {
  -- Proof goes here
  sorry
}

end ruiz_original_salary_l1772_177233


namespace pirates_total_distance_l1772_177221

def adjusted_distance_1 (d: ℝ) : ℝ := d * 1.10
def adjusted_distance_2 (d: ℝ) : ℝ := d * 1.15
def adjusted_distance_3 (d: ℝ) : ℝ := d * 1.20
def adjusted_distance_4 (d: ℝ) : ℝ := d * 1.25

noncomputable def total_distance : ℝ := 
  let first_island := (adjusted_distance_1 10) + (adjusted_distance_1 15) + (adjusted_distance_1 20)
  let second_island := adjusted_distance_2 40
  let third_island := (adjusted_distance_3 25) + (adjusted_distance_3 20) + (adjusted_distance_3 25) + (adjusted_distance_3 20)
  let fourth_island := adjusted_distance_4 35
  first_island + second_island + third_island + fourth_island

theorem pirates_total_distance : total_distance = 247.25 := by
  sorry

end pirates_total_distance_l1772_177221


namespace find_largest_number_l1772_177220

-- Define what it means for a sequence of 4 numbers to be an arithmetic progression with a given common difference d
def is_arithmetic_progression (a b c d : ℝ) (diff : ℝ) : Prop := (b - a = diff) ∧ (c - b = diff) ∧ (d - c = diff)

-- Define what it means for a sequence of 4 numbers to be a geometric progression
def is_geometric_progression (a b c d : ℝ) : Prop := b / a = c / b ∧ c / b = d / c

-- Given conditions for the sequence of 8 increasing real numbers
def conditions (a : ℕ → ℝ) : Prop :=
  (∀ i j, i < j → a i < a j) ∧
  ∃ i j k, is_arithmetic_progression (a i) (a (i+1)) (a (i+2)) (a (i+3)) 4 ∧
            is_arithmetic_progression (a j) (a (j+1)) (a (j+2)) (a (j+3)) 36 ∧
            is_geometric_progression (a k) (a (k+1)) (a (k+2)) (a (k+3))

-- Prove that under these conditions, the largest number in the sequence is 126
theorem find_largest_number (a : ℕ → ℝ) : conditions a → a 7 = 126 :=
by
  sorry

end find_largest_number_l1772_177220


namespace current_time_is_208_l1772_177207

def minute_hand_position (t : ℝ) : ℝ := 6 * t
def hour_hand_position (t : ℝ) : ℝ := 0.5 * t

theorem current_time_is_208 (t : ℝ) (h1 : 0 < t) (h2 : t < 60) 
  (h3 : minute_hand_position (t + 8) + 60 = hour_hand_position (t + 5)) : 
  t = 8 :=
by sorry

end current_time_is_208_l1772_177207


namespace a4_value_l1772_177222

-- Definitions and helper theorems can go here
variable (S : ℕ → ℕ)
variable (a : ℕ → ℕ)

-- These are our conditions
axiom h1 : S 2 = a 1 + a 2
axiom h2 : a 2 = 3
axiom h3 : ∀ n, S (n + 1) = 2 * S n + 1

theorem a4_value : a 4 = 12 :=
sorry  -- proof to be filled in later

end a4_value_l1772_177222


namespace area_of_square_field_l1772_177257

-- Definitions
def cost_per_meter : ℝ := 1.40
def total_cost : ℝ := 932.40
def gate_width : ℝ := 1.0

-- Problem Statement
theorem area_of_square_field (s : ℝ) (A : ℝ) 
  (h1 : (4 * s - 2 * gate_width) * cost_per_meter = total_cost)
  (h2 : A = s^2) : A = 27889 := 
  sorry

end area_of_square_field_l1772_177257


namespace toothpicks_in_stage_200_l1772_177235

def initial_toothpicks : ℕ := 6
def toothpicks_per_stage : ℕ := 5
def stage_number : ℕ := 200

theorem toothpicks_in_stage_200 :
  initial_toothpicks + (stage_number - 1) * toothpicks_per_stage = 1001 := by
  sorry

end toothpicks_in_stage_200_l1772_177235


namespace total_time_correct_l1772_177241

def greta_time : ℝ := 6.5
def george_time : ℝ := greta_time - 1.5
def gloria_time : ℝ := 2 * george_time
def gary_time : ℝ := (george_time + gloria_time) + 1.75
def gwen_time : ℝ := (greta_time + george_time) - 0.40 * (greta_time + george_time)
def total_time : ℝ := greta_time + george_time + gloria_time + gary_time + gwen_time

theorem total_time_correct : total_time = 45.15 := by
  sorry

end total_time_correct_l1772_177241


namespace correct_incorrect_difference_l1772_177225

variable (x : ℝ)

theorem correct_incorrect_difference : (x - 2152) - (x - 1264) = 888 := by
  sorry

end correct_incorrect_difference_l1772_177225


namespace travel_ways_l1772_177258

theorem travel_ways (highways : ℕ) (railways : ℕ) (n : ℕ) :
  highways = 3 → railways = 2 → n = highways + railways → n = 5 :=
by
  intros h_eq r_eq n_eq
  rw [h_eq, r_eq] at n_eq
  exact n_eq

end travel_ways_l1772_177258


namespace smallest_number_divisible_1_to_10_l1772_177246

theorem smallest_number_divisible_1_to_10 : ∃ n : ℕ, (∀ m ∈ (List.range' 1 10), m ∣ n) ∧ n = 2520 := 
by {
  sorry
}

end smallest_number_divisible_1_to_10_l1772_177246


namespace pencil_length_l1772_177259

theorem pencil_length :
  let purple := 1.5
  let black := 0.5
  let blue := 2
  purple + black + blue = 4 := by sorry

end pencil_length_l1772_177259


namespace coin_difference_is_eight_l1772_177249

theorem coin_difference_is_eight :
  let min_coins := 2  -- two 25-cent coins
  let max_coins := 10 -- ten 5-cent coins
  max_coins - min_coins = 8 :=
by
  sorry

end coin_difference_is_eight_l1772_177249


namespace power_mod_condition_l1772_177218

-- Defining the main problem conditions
theorem power_mod_condition (n: ℕ) : 
  (7^2 ≡ 1 [MOD 12]) →
  (∀ k: ℕ, 7^(2*k+1) ≡ 7 [MOD 12]) →
  (∀ k: ℕ, 7^(2*k) ≡ 1 [MOD 12]) →
  7^135 ≡ 7 [MOD 12] :=
by
  intros h1 h2 h3
  sorry

end power_mod_condition_l1772_177218


namespace average_apples_per_hour_l1772_177294

theorem average_apples_per_hour :
  (5.0 / 3.0) = 1.67 := 
sorry

end average_apples_per_hour_l1772_177294


namespace find_J_l1772_177216

-- Define the problem conditions
def eq1 : Nat := 32
def eq2 : Nat := 4

-- Define the target equation form
def target_eq (J : Nat) : Prop := (eq1^3) * (eq2^3) = 2^J

theorem find_J : ∃ J : Nat, target_eq J ∧ J = 21 :=
by
  -- Rest of the proof goes here
  sorry

end find_J_l1772_177216


namespace stream_speed_l1772_177298

variable (B S : ℝ)

def downstream_eq : Prop := B + S = 13
def upstream_eq : Prop := B - S = 5

theorem stream_speed (h1 : downstream_eq B S) (h2 : upstream_eq B S) : S = 4 :=
by
  sorry

end stream_speed_l1772_177298


namespace problem_l1772_177202

/-
A problem involving natural numbers a and b
where:
1. Their sum is 20000
2. One of them (b) is divisible by 5
3. Erasing the units digit of b gives the other number a

We want to prove their difference is 16358
-/

def nat_sum_and_difference (a b : ℕ) : Prop :=
  a + b = 20000 ∧
  b % 5 = 0 ∧
  (b % 10 = 0 ∧ b / 10 = a ∨ b % 10 = 5 ∧ (b - 5) / 10 = a)

theorem problem (a b : ℕ) (h : nat_sum_and_difference a b) : b - a = 16358 := 
  sorry

end problem_l1772_177202


namespace blue_paint_cans_l1772_177209

noncomputable def ratio_of_blue_to_green := 4 / 1
def total_cans := 50
def fraction_of_blue := 4 / (4 + 1)
def number_of_blue_cans := fraction_of_blue * total_cans

theorem blue_paint_cans : number_of_blue_cans = 40 := by
  sorry

end blue_paint_cans_l1772_177209


namespace triangle_angle_B_l1772_177270

theorem triangle_angle_B (a b A B : ℝ) (h1 : a * Real.cos B = 3 * b * Real.cos A) (h2 : B = A - Real.pi / 6) : 
  B = Real.pi / 6 := by
  sorry

end triangle_angle_B_l1772_177270


namespace least_multiple_of_7_not_lucky_l1772_177242

def sum_of_digits (n : ℕ) : ℕ :=
  n.digits 10 |>.sum

def is_lucky (n : ℕ) : Prop :=
  n % sum_of_digits n = 0

def is_multiple_of_7 (n : ℕ) : Prop :=
  n % 7 = 0

theorem least_multiple_of_7_not_lucky : ∃ n, is_multiple_of_7 n ∧ ¬ is_lucky n ∧ n = 14 :=
by
  sorry

end least_multiple_of_7_not_lucky_l1772_177242


namespace austin_hours_on_mondays_l1772_177215

-- Define the conditions
def earning_per_hour : ℕ := 5
def hours_wednesday : ℕ := 1
def hours_friday : ℕ := 3
def weeks : ℕ := 6
def bicycle_cost : ℕ := 180

-- Define the proof problem
theorem austin_hours_on_mondays (M : ℕ) :
  earning_per_hour * weeks * (M + hours_wednesday + hours_friday) = bicycle_cost → M = 2 :=
by 
  intro h
  sorry

end austin_hours_on_mondays_l1772_177215


namespace fresh_pineapples_left_l1772_177296

namespace PineappleStore

def initial := 86
def sold := 48
def rotten := 9

theorem fresh_pineapples_left (initial sold rotten : ℕ) (h_initial : initial = 86) (h_sold : sold = 48) (h_rotten : rotten = 9) :
  initial - sold - rotten = 29 :=
by sorry

end PineappleStore

end fresh_pineapples_left_l1772_177296


namespace tax_percentage_excess_income_l1772_177284

theorem tax_percentage_excess_income :
  ∀ (rate : ℝ) (total_tax income : ℝ), 
  rate = 0.15 →
  total_tax = 8000 →
  income = 50000 →
  (total_tax - income * rate) / (income - 40000) = 0.2 :=
by
  intros rate total_tax income hrate htotal hincome
  -- proof omitted
  sorry

end tax_percentage_excess_income_l1772_177284


namespace polygon_interior_exterior_eq_l1772_177267

theorem polygon_interior_exterior_eq (n : ℕ) (hn : 3 ≤ n)
  (interior_sum_eq_exterior_sum : (n - 2) * 180 = 360) : n = 4 := by
  sorry

end polygon_interior_exterior_eq_l1772_177267


namespace mouse_jump_less_than_frog_l1772_177265

-- Definitions for the given conditions
def grasshopper_jump : ℕ := 25
def frog_jump : ℕ := grasshopper_jump + 32
def mouse_jump : ℕ := 31

-- The statement we need to prove
theorem mouse_jump_less_than_frog :
  frog_jump - mouse_jump = 26 :=
by
  -- The proof will be filled in here
  sorry

end mouse_jump_less_than_frog_l1772_177265


namespace monotonically_increasing_interval_l1772_177274

noncomputable def f (x : ℝ) : ℝ := x * Real.log x

theorem monotonically_increasing_interval :
  ∀ x : ℝ, x > 1 / Real.exp 1 → (Real.log x + 1) > 0 :=
by
  intros x hx
  sorry

end monotonically_increasing_interval_l1772_177274


namespace possible_to_fill_array_l1772_177285

open BigOperators

theorem possible_to_fill_array :
  ∃ (f : (Fin 10) × (Fin 10) → ℕ),
    (∀ i j : Fin 10, 
      (i ≠ 0 → f (i, j) ∣ f (i - 1, j) ∧ f (i, j) ≠ f (i - 1, j))) ∧
    (∀ i : Fin 10, ∃ n : ℕ, ∀ j : Fin 10, f (i, j) = n + j) :=
sorry

end possible_to_fill_array_l1772_177285


namespace geometric_progression_quadrilateral_exists_l1772_177256

theorem geometric_progression_quadrilateral_exists :
  ∃ (a1 r : ℝ), a1 > 0 ∧ r > 0 ∧ 
  (1 + r + r^2 > r^3) ∧
  (1 + r + r^3 > r^2) ∧
  (1 + r^2 + r^3 > r) ∧
  (r + r^2 + r^3 > 1) := 
sorry

end geometric_progression_quadrilateral_exists_l1772_177256


namespace ratio_of_segments_l1772_177234

theorem ratio_of_segments
  (x y z u v : ℝ)
  (h_triangle : x^2 + y^2 = z^2)
  (h_ratio_legs : 4 * x = 3 * y)
  (h_u : u = x^2 / z)
  (h_v : v = y^2 / z) :
  u / v = 9 / 16 := 
  sorry

end ratio_of_segments_l1772_177234


namespace coefficient_sum_of_squares_is_23456_l1772_177224

theorem coefficient_sum_of_squares_is_23456 
  (p q r s t u : ℤ)
  (h : ∀ x : ℤ, 1728 * x^3 + 64 = (p * x^2 + q * x + r) * (s * x^2 + t * x + u)) :
  p^2 + q^2 + r^2 + s^2 + t^2 + u^2 = 23456 := 
by
  sorry

end coefficient_sum_of_squares_is_23456_l1772_177224


namespace books_sold_on_Tuesday_l1772_177297

theorem books_sold_on_Tuesday 
  (initial_stock : ℕ)
  (books_sold_Monday : ℕ)
  (books_sold_Wednesday : ℕ)
  (books_sold_Thursday : ℕ)
  (books_sold_Friday : ℕ)
  (books_not_sold : ℕ) :
  initial_stock = 800 →
  books_sold_Monday = 60 →
  books_sold_Wednesday = 20 →
  books_sold_Thursday = 44 →
  books_sold_Friday = 66 →
  books_not_sold = 600 →
  ∃ (books_sold_Tuesday : ℕ), books_sold_Tuesday = 10
:= by
  intros h_initial h_monday h_wednesday h_thursday h_friday h_not_sold
  sorry

end books_sold_on_Tuesday_l1772_177297


namespace a_and_b_work_together_l1772_177238
noncomputable def work_rate (days : ℕ) : ℝ := 1 / days

theorem a_and_b_work_together (A_days B_days : ℕ) (hA : A_days = 32) (hB : B_days = 32) :
  (1 / work_rate A_days + 1 / work_rate B_days) = 16 := by
  sorry

end a_and_b_work_together_l1772_177238


namespace jiwon_distance_to_school_l1772_177282

theorem jiwon_distance_to_school
  (taehong_distance_meters jiwon_distance_meters : ℝ)
  (taehong_distance_km : ℝ := 1.05)
  (h1 : taehong_distance_meters = jiwon_distance_meters + 460)
  (h2 : taehong_distance_meters = taehong_distance_km * 1000) :
  jiwon_distance_meters / 1000 = 0.59 := 
sorry

end jiwon_distance_to_school_l1772_177282


namespace basketball_team_win_requirement_l1772_177264

theorem basketball_team_win_requirement :
  ∀ (games_won_first_60 : ℕ) (total_games : ℕ) (win_percentage : ℚ) (remaining_games : ℕ),
    games_won_first_60 = 45 →
    total_games = 110 →
    win_percentage = 0.75 →
    remaining_games = 50 →
    ∃ games_won_remaining, games_won_remaining = 38 ∧
    (games_won_first_60 + games_won_remaining) / total_games = win_percentage :=
by
  intros
  sorry

end basketball_team_win_requirement_l1772_177264


namespace correct_operation_B_l1772_177279

theorem correct_operation_B (x : ℝ) : 
  x - 2 * x = -x :=
sorry

end correct_operation_B_l1772_177279


namespace decrease_in_demand_correct_l1772_177288

noncomputable def proportionate_decrease_in_demand (p e : ℝ) : ℝ :=
  1 - (1 / (1 + e * p))

theorem decrease_in_demand_correct :
  proportionate_decrease_in_demand 0.20 1.5 = 0.23077 :=
by
  sorry

end decrease_in_demand_correct_l1772_177288


namespace constants_sum_l1772_177248

theorem constants_sum (c d : ℝ) (f : ℝ → ℝ) 
  (h₁ : ∀ x : ℝ, f x = if x ≤ 5 then c * x + d else 10 - 2 * x) 
  (h₂ : ∀ x : ℝ, f (f x) = x) : c + d = 6.5 := 
by sorry

end constants_sum_l1772_177248


namespace factor_polynomial_equiv_l1772_177239

theorem factor_polynomial_equiv :
  (x^2 + 2 * x + 1) * (x^2 + 8 * x + 15) + (x^2 + 6 * x - 8) = 
  (x^2 + 7 * x + 1) * (x^2 + 3 * x + 7) :=
by sorry

end factor_polynomial_equiv_l1772_177239


namespace minimum_value_l1772_177208

variables (a b c d : ℝ)
-- Conditions
def condition1 := (b - 2 * a^2 + 3 * Real.log a)^2 = 0
def condition2 := (c - d - 3)^2 = 0

-- Theorem stating the goal
theorem minimum_value (h1 : condition1 a b) (h2 : condition2 c d) : 
  (a - c)^2 + (b - d)^2 = 8 :=
sorry

end minimum_value_l1772_177208


namespace probability_of_event_l1772_177210

noncomputable def interval_probability : ℝ :=
  if 0 ≤ 1 ∧ 1 ≤ 1 then (1 - (1/3)) / (1 - 0) else 0

theorem probability_of_event :
  interval_probability = 2 / 3 :=
by
  rw [interval_probability]
  sorry

end probability_of_event_l1772_177210


namespace find_a_l1772_177292

variable (a : ℝ)

def p (a : ℝ) : Set ℝ := {x | a-1 < x ∧ x < a+1}
def q : Set ℝ := {x | x^2 - 4*x + 3 ≥ 0}
def q_negation : Set ℝ := {x | 1 < x ∧ x < 3}

theorem find_a :
  (∀ x, q_negation x → p a x) → a = 2 := by
  sorry

end find_a_l1772_177292


namespace cindy_olaf_earnings_l1772_177269
noncomputable def total_earnings (apples grapes : ℕ) (price_apple price_grape : ℝ) : ℝ :=
  apples * price_apple + grapes * price_grape

theorem cindy_olaf_earnings :
  total_earnings 15 12 2 1.5 = 48 :=
by
  sorry

end cindy_olaf_earnings_l1772_177269


namespace class_duration_l1772_177217

theorem class_duration (h1 : 8 * 60 + 30 = 510) (h2 : 9 * 60 + 5 = 545) : (545 - 510 = 35) :=
by
  sorry

end class_duration_l1772_177217


namespace lcm_of_three_l1772_177245

theorem lcm_of_three (A1 A2 A3 : ℕ) (D : ℕ)
  (hD : D = Nat.gcd (A1 * A2) (Nat.gcd (A2 * A3) (A3 * A1))) :
  Nat.lcm (Nat.lcm A1 A2) A3 = (A1 * A2 * A3) / D :=
sorry

end lcm_of_three_l1772_177245


namespace fraction_multiplication_division_l1772_177287

-- We will define the fractions and state the equivalence
def fraction_1 : ℚ := 145 / 273
def fraction_2 : ℚ := 2 * (173 / 245) -- equivalent to 2 173/245
def fraction_3 : ℚ := 21 * (13 / 15) -- equivalent to 21 13/15

theorem fraction_multiplication_division :
  (frac1 * frac2 / frac3) = 7395 / 112504 := 
by sorry

end fraction_multiplication_division_l1772_177287


namespace reef_age_in_decimal_l1772_177227

def octal_to_decimal (n: Nat) : Nat :=
  match n with
  | 367 => 7 * (8^0) + 6 * (8^1) + 3 * (8^2)
  | _   => 0  -- Placeholder for other values if needed

theorem reef_age_in_decimal : octal_to_decimal 367 = 247 := by
  sorry

end reef_age_in_decimal_l1772_177227


namespace forty_percent_of_number_l1772_177281

theorem forty_percent_of_number (N : ℝ) (h : (1 / 4) * (1 / 3) * (2 / 5) * N = 20) : 0.40 * N = 240 :=
by
  sorry

end forty_percent_of_number_l1772_177281


namespace f_at_zero_f_on_negative_l1772_177254

-- Define the odd function condition
def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

-- Define the function f(x) for x > 0 condition
def f_on_positive (f : ℝ → ℝ) : Prop :=
  ∀ x, x > 0 → f x = x^2 + x - 1

-- Lean statement for the first proof: f(0) = 0
theorem f_at_zero (f : ℝ → ℝ) (h_odd : is_odd_function f) (h_positive : f_on_positive f) : f 0 = 0 :=
sorry

-- Lean statement for the second proof: for x < 0, f(x) = -x^2 + x + 1
theorem f_on_negative (f : ℝ → ℝ) (h_odd : is_odd_function f) (h_positive : f_on_positive f) :
  ∀ x, x < 0 → f x = -x^2 + x + 1 :=
sorry

end f_at_zero_f_on_negative_l1772_177254


namespace find_a_value_l1772_177261

noncomputable def prob_sum_equals_one (a : ℝ) : Prop :=
  a * (1/2 + 1/4 + 1/8 + 1/16) = 1

theorem find_a_value (a : ℝ) (h : prob_sum_equals_one a) : a = 16/15 :=
sorry

end find_a_value_l1772_177261


namespace slope_of_line_l1772_177278

theorem slope_of_line {x y : ℝ} : 
  (∃ (x y : ℝ), 0 = 3 * x + 4 * y + 12) → ∀ (m : ℝ), m = -3/4 :=
by
  sorry

end slope_of_line_l1772_177278


namespace final_computation_l1772_177237

noncomputable def N := (15 ^ 10 / 15 ^ 9) ^ 3 * 5 ^ 3

theorem final_computation : (N / 3 ^ 3) = 15625 := 
by 
  sorry

end final_computation_l1772_177237


namespace bob_and_bill_same_class_probability_l1772_177203

-- Definitions based on the conditions mentioned in the original problem
def total_people : ℕ := 32
def allowed_per_class : ℕ := 30
def number_chosen : ℕ := 2
def number_of_classes : ℕ := 2
def bob_and_bill_pair : ℕ := 1

-- Binomial coefficient calculation (32 choose 2)
def binomial_coefficient (n k : ℕ) : ℕ := Nat.choose n k
def total_ways := binomial_coefficient total_people number_chosen

-- Probability that Bob and Bill are chosen
def probability_chosen : ℚ := bob_and_bill_pair / total_ways

-- Probability that Bob and Bill are placed in the same class
def probability_same_class : ℚ := 1 / number_of_classes

-- Total combined probability
def combined_probability : ℚ := probability_chosen * probability_same_class

-- Statement of the theorem
theorem bob_and_bill_same_class_probability :
  combined_probability = 1 / 992 := 
sorry

end bob_and_bill_same_class_probability_l1772_177203


namespace book_pages_l1772_177243

theorem book_pages (P : ℝ) (h1 : 2/3 * P = 1/3 * P + 20) : P = 60 :=
by
  sorry

end book_pages_l1772_177243


namespace least_number_remainder_l1772_177214

noncomputable def lcm_12_15_20_54 : ℕ := 540

theorem least_number_remainder :
  ∀ (n r : ℕ), (n = lcm_12_15_20_54 + r) → 
  (n % 12 = r) ∧ (n % 15 = r) ∧ (n % 20 = r) ∧ (n % 54 = r) → 
  r = 0 :=
by
  sorry

end least_number_remainder_l1772_177214


namespace iesha_total_books_l1772_177228

theorem iesha_total_books (schoolBooks sportsBooks : ℕ) (h1 : schoolBooks = 19) (h2 : sportsBooks = 39) : schoolBooks + sportsBooks = 58 :=
by
  sorry

end iesha_total_books_l1772_177228


namespace exists_language_spoken_by_at_least_three_l1772_177291

noncomputable def smallestValue_n (k : ℕ) : ℕ :=
  2 * k + 3

theorem exists_language_spoken_by_at_least_three (k n : ℕ) (P : Fin n → Set ℕ) (K : ℕ → ℕ) :
  (n = smallestValue_n k) →
  (∀ i, (K i) ≤ k) →
  (∀ (x y z : Fin n), ∃ l, l ∈ P x ∧ l ∈ P y ∧ l ∈ P z ∨ l ∈ P y ∧ l ∈ P z ∨ l ∈ P z ∧ l ∈ P x ∨ l ∈ P x ∧ l ∈ P y) →
  ∃ l, ∃ (a b c : Fin n), l ∈ P a ∧ l ∈ P b ∧ l ∈ P c :=
by
  intros h1 h2 h3
  sorry

end exists_language_spoken_by_at_least_three_l1772_177291
