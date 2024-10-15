import Mathlib

namespace NUMINAMATH_GPT_simplify_and_evaluate_l1062_106234

noncomputable def expr (x : ℝ) : ℝ :=
  (x + 3) * (x - 2) + x * (4 - x)

theorem simplify_and_evaluate (x : ℝ) (hx : x = 2) : expr x = 4 :=
by
  rw [hx]
  show expr 2 = 4
  sorry

end NUMINAMATH_GPT_simplify_and_evaluate_l1062_106234


namespace NUMINAMATH_GPT_items_in_storeroom_l1062_106209

-- Conditions definitions
def restocked_items : ℕ := 4458
def sold_items : ℕ := 1561
def total_items_left : ℕ := 3472

-- Statement of the proof
theorem items_in_storeroom : (total_items_left - (restocked_items - sold_items)) = 575 := 
by
  sorry

end NUMINAMATH_GPT_items_in_storeroom_l1062_106209


namespace NUMINAMATH_GPT_ball_bounces_to_less_than_two_feet_l1062_106295

noncomputable def bounce_height (n : ℕ) : ℝ := 20 * (3 / 4) ^ n

theorem ball_bounces_to_less_than_two_feet : ∃ k : ℕ, bounce_height k < 2 ∧ k = 7 :=
by
  -- We need to show that bounce_height k < 2 when k = 7
  sorry

end NUMINAMATH_GPT_ball_bounces_to_less_than_two_feet_l1062_106295


namespace NUMINAMATH_GPT_equal_sum_seq_example_l1062_106282

def EqualSumSeq (a : ℕ → ℕ) (c : ℕ) : Prop := ∀ n, a n + a (n + 1) = c

theorem equal_sum_seq_example (a : ℕ → ℕ) 
  (h1 : EqualSumSeq a 5) 
  (h2 : a 1 = 2) : a 6 = 3 :=
by 
  sorry

end NUMINAMATH_GPT_equal_sum_seq_example_l1062_106282


namespace NUMINAMATH_GPT_lattice_point_condition_l1062_106225

theorem lattice_point_condition (b : ℚ) :
  (∀ (m : ℚ), (1 / 3 < m ∧ m < b) →
    ∀ x : ℤ, (0 < x ∧ x ≤ 200) →
      ¬ ∃ y : ℤ, y = m * x + 3) →
  b = 68 / 203 := 
sorry

end NUMINAMATH_GPT_lattice_point_condition_l1062_106225


namespace NUMINAMATH_GPT_two_square_numbers_difference_133_l1062_106249

theorem two_square_numbers_difference_133 : 
  ∃ (x y : ℤ), x^2 - y^2 = 133 ∧ ((x = 67 ∧ y = 66) ∨ (x = 13 ∧ y = 6)) :=
by {
  sorry
}

end NUMINAMATH_GPT_two_square_numbers_difference_133_l1062_106249


namespace NUMINAMATH_GPT_simplify_expression_l1062_106286

theorem simplify_expression :
  (3 + 4 + 5 + 7) / 3 + (3 * 6 + 9) / 4 = 157 / 12 :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l1062_106286


namespace NUMINAMATH_GPT_transformed_interval_l1062_106273

noncomputable def transformation (x : ℝ) : ℝ := 8 * x - 2

theorem transformed_interval :
  ∀ (x : ℝ), 0 ≤ x ∧ x ≤ 1 → -2 ≤ transformation x ∧ transformation x ≤ 6 := by
  intro x h
  unfold transformation
  sorry

end NUMINAMATH_GPT_transformed_interval_l1062_106273


namespace NUMINAMATH_GPT_true_discount_is_52_l1062_106259

/-- The banker's gain on a bill due 3 years hence at 15% per annum is Rs. 23.4. -/
def BG : ℝ := 23.4

/-- The rate of interest per annum is 15%. -/
def R : ℝ := 15

/-- The time in years is 3. -/
def T : ℝ := 3

/-- The true discount is Rs. 52. -/
theorem true_discount_is_52 : BG * 100 / (R * T) = 52 :=
by
  -- Placeholder for proof. This needs proper calculation.
  sorry

end NUMINAMATH_GPT_true_discount_is_52_l1062_106259


namespace NUMINAMATH_GPT_f_even_l1062_106246

def is_even (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f x = f (-x)

def f (x : ℝ) : ℝ := x^2 + 1

theorem f_even (a : ℝ) (h1 : is_even f) (h2 : ∀ x, -1 ≤ x ∧ x ≤ a) : f a = 2 :=
  sorry

end NUMINAMATH_GPT_f_even_l1062_106246


namespace NUMINAMATH_GPT_option_D_correct_l1062_106268

-- Formal statement in Lean 4
theorem option_D_correct (m : ℝ) : 6 * m + (-2 - 10 * m) = -4 * m - 2 :=
by
  -- Proof is skipped per instruction
  sorry

end NUMINAMATH_GPT_option_D_correct_l1062_106268


namespace NUMINAMATH_GPT_profit_rate_l1062_106207

variables (list_price : ℝ)
          (discount : ℝ := 0.95)
          (selling_increase : ℝ := 1.6)
          (inflation_rate : ℝ := 1.4)

theorem profit_rate (list_price : ℝ) : 
  (selling_increase / (discount * inflation_rate)) - 1 = 0.203 :=
by 
  sorry

end NUMINAMATH_GPT_profit_rate_l1062_106207


namespace NUMINAMATH_GPT_percentage_of_childrens_books_l1062_106299

/-- Conditions: 
- There are 160 books in total.
- 104 of them are for adults.
Prove that the percentage of books intended for children is 35%. --/
theorem percentage_of_childrens_books (total_books : ℕ) (adult_books : ℕ) 
  (h_total : total_books = 160) (h_adult : adult_books = 104) :
  (160 - 104) / 160 * 100 = 35 := 
by {
  sorry -- Proof skipped
}

end NUMINAMATH_GPT_percentage_of_childrens_books_l1062_106299


namespace NUMINAMATH_GPT_trajectory_of_complex_point_l1062_106237

open Complex Topology

theorem trajectory_of_complex_point (z : ℂ) (hz : ‖z‖ ≤ 1) : 
  {w : ℂ | ‖w‖ ≤ 1} = {w : ℂ | w.re * w.re + w.im * w.im ≤ 1} :=
sorry

end NUMINAMATH_GPT_trajectory_of_complex_point_l1062_106237


namespace NUMINAMATH_GPT_nonneg_integer_representation_l1062_106254

theorem nonneg_integer_representation (n : ℕ) : 
  ∃ x y : ℕ, n = (x + y) * (x + y) + 3 * x + y / 2 := 
sorry

end NUMINAMATH_GPT_nonneg_integer_representation_l1062_106254


namespace NUMINAMATH_GPT_alice_walk_time_l1062_106279

theorem alice_walk_time (bob_time : ℝ) 
  (bob_distance : ℝ) 
  (alice_distance1 : ℝ) 
  (alice_distance2 : ℝ) 
  (time_ratio : ℝ) 
  (expected_alice_time : ℝ) :
  bob_time = 36 →
  bob_distance = 6 →
  alice_distance1 = 4 →
  alice_distance2 = 7 →
  time_ratio = 1 / 3 →
  expected_alice_time = 21 →
  (expected_alice_time = alice_distance2 / (alice_distance1 / (bob_time * time_ratio))) := 
by
  intros h1 h2 h3 h4 h5 h6
  rw [h1, h3, h5]
  have h_speed : ℝ := alice_distance1 / (bob_time * time_ratio)
  rw [h4, h6]
  linarith [h_speed]

end NUMINAMATH_GPT_alice_walk_time_l1062_106279


namespace NUMINAMATH_GPT_min_value_of_expression_l1062_106267

theorem min_value_of_expression : ∃ x : ℝ, (8 - x) * (6 - x) * (8 + x) * (6 + x) ≥ -196 :=
by
  sorry

end NUMINAMATH_GPT_min_value_of_expression_l1062_106267


namespace NUMINAMATH_GPT_circle_equation_line_equation_l1062_106208

theorem circle_equation (a b r x y : ℝ) (h1 : a + b = 2 * x + y)
  (h2 : (a, 2*a - 2) = ((1, 2) : ℝ × ℝ))
  (h3 : (a, 2*a - 2) = ((2, 1) : ℝ × ℝ)) :
  (x - 2) ^ 2 + (y - 2) ^ 2 = 1 := sorry

theorem line_equation (x y m : ℝ) (h1 : y + 3 = (x - (-3)) * ((-3) - 0) / (m - (-3)))
  (h2 : (x, y) = (m, 0) ∨ (x, y) = (m, 0))
  (h3 : (m = 1 ∨ m = - 3 / 4)) :
  (3 * x + 4 * y - 3 = 0) ∨ (4 * x + 3 * y + 3 = 0) := sorry

end NUMINAMATH_GPT_circle_equation_line_equation_l1062_106208


namespace NUMINAMATH_GPT_cost_per_vent_l1062_106251

/--
Given that:
1. The total cost of the HVAC system is $20,000.
2. The system includes 2 conditioning zones.
3. Each zone has 5 vents.

Prove that the cost per vent is $2000.
-/
theorem cost_per_vent (total_cost : ℕ) (zones : ℕ) (vents_per_zone : ℕ) (h1 : total_cost = 20000) (h2 : zones = 2) (h3 : vents_per_zone = 5) :
  total_cost / (zones * vents_per_zone) = 2000 := 
sorry

end NUMINAMATH_GPT_cost_per_vent_l1062_106251


namespace NUMINAMATH_GPT_triangle_area_l1062_106277

theorem triangle_area (a b c : ℕ) (h1 : a = 9) (h2 : b = 12) (h3 : c = 15) (right_triangle : a^2 + b^2 = c^2) : 
  (1 / 2) * a * b = 54 := by
    sorry

end NUMINAMATH_GPT_triangle_area_l1062_106277


namespace NUMINAMATH_GPT_efficiency_difference_l1062_106269

variables (Rp Rq : ℚ)

-- Given conditions
def p_rate := Rp = 1 / 21
def combined_rate := Rp + Rq = 1 / 11

-- Define the percentage efficiency difference
def percentage_difference := (Rp - Rq) / Rq * 100

-- Main statement to prove
theorem efficiency_difference : 
  p_rate Rp ∧ 
  combined_rate Rp Rq → 
  percentage_difference Rp Rq = 10 :=
sorry

end NUMINAMATH_GPT_efficiency_difference_l1062_106269


namespace NUMINAMATH_GPT_a_must_not_be_zero_l1062_106256

theorem a_must_not_be_zero (a b c d : ℝ) (h₁ : a / b < -3 * (c / d)) (h₂ : b ≠ 0) (h₃ : d ≠ 0) (h₄ : c = 2 * a) : a ≠ 0 :=
sorry

end NUMINAMATH_GPT_a_must_not_be_zero_l1062_106256


namespace NUMINAMATH_GPT_relationship_between_a_and_b_l1062_106284

-- Define the given linear function
def linear_function (k : ℝ) (x : ℝ) : ℝ := (k^2 + 1) * x + 1

-- Formalize the relationship between a and b given the points and the linear function
theorem relationship_between_a_and_b (a b k : ℝ) 
  (hP : a = linear_function k (-4))
  (hQ : b = linear_function k 2) :
  a < b := 
by
  sorry  -- Proof to be filled in by the theorem prover

end NUMINAMATH_GPT_relationship_between_a_and_b_l1062_106284


namespace NUMINAMATH_GPT_rainfall_difference_l1062_106214

noncomputable def r₁ : ℝ := 26
noncomputable def r₂ : ℝ := 34
noncomputable def r₃ : ℝ := r₂ - 12
noncomputable def avg : ℝ := 140

theorem rainfall_difference : (avg - (r₁ + r₂ + r₃)) = 58 := 
by
  sorry

end NUMINAMATH_GPT_rainfall_difference_l1062_106214


namespace NUMINAMATH_GPT_sum_of_first_8_terms_l1062_106265

-- Define the geometric sequence and its properties
def geometric_sequence (a : ℕ → ℝ) (r : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n * r

-- Define the sum of the first n terms of a sequence
def sum_of_first_n_terms (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  (Finset.range n).sum a

-- Given conditions
def c1 (a : ℕ → ℝ) : Prop := geometric_sequence a 2
def c2 (a : ℕ → ℝ) : Prop := sum_of_first_n_terms a 4 = 1

-- The statement to prove
theorem sum_of_first_8_terms (a : ℕ → ℝ) (h1 : c1 a) (h2 : c2 a) : sum_of_first_n_terms a 8 = 17 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_first_8_terms_l1062_106265


namespace NUMINAMATH_GPT_find_x_l1062_106223

theorem find_x :
  ∃ x : ℝ, (0 < x) ∧ (⌊x⌋ * x + x^2 = 93) ∧ (x = 7.10) :=
by {
   sorry
}

end NUMINAMATH_GPT_find_x_l1062_106223


namespace NUMINAMATH_GPT_trapezoid_area_l1062_106200

theorem trapezoid_area (h : ℝ) : 
  let b1 : ℝ := 4 * h + 2
  let b2 : ℝ := 5 * h
  (b1 + b2) / 2 * h = (9 * h ^ 2 + 2 * h) / 2 :=
by 
  let b1 := 4 * h + 2
  let b2 := 5 * h
  sorry

end NUMINAMATH_GPT_trapezoid_area_l1062_106200


namespace NUMINAMATH_GPT_chord_length_l1062_106288

noncomputable def circle_center (c: ℝ × ℝ) (r: ℝ): Prop := 
  ∃ x y: ℝ, 
    (x - c.1)^2 + (y - c.2)^2 = r^2

noncomputable def line_equation (a b c: ℝ): Prop := 
  ∀ x y: ℝ, 
    a*x + b*y + c = 0

theorem chord_length (a: ℝ): 
  circle_center (2, 1) 2 ∧ line_equation a 1 (-5) ∧
  ∃(chord_len: ℝ), chord_len = 4 → 
  a = 2 :=
by
  sorry

end NUMINAMATH_GPT_chord_length_l1062_106288


namespace NUMINAMATH_GPT_find_p_l1062_106213

theorem find_p (f p : ℂ) (w : ℂ) (h1 : f * p - w = 15000) (h2 : f = 8) (h3 : w = 10 + 200 * Complex.I) : 
  p = 1876.25 + 25 * Complex.I := 
sorry

end NUMINAMATH_GPT_find_p_l1062_106213


namespace NUMINAMATH_GPT_train_speed_l1062_106210

theorem train_speed (v : ℕ) :
    let distance_between_stations := 155
    let speed_of_train_from_A := 20
    let start_time_train_A := 7
    let start_time_train_B := 8
    let meet_time := 11
    let distance_traveled_by_A := speed_of_train_from_A * (meet_time - start_time_train_A)
    let remaining_distance := distance_between_stations - distance_traveled_by_A
    let traveling_time_train_B := meet_time - start_time_train_B
    v * traveling_time_train_B = remaining_distance → v = 25 :=
by
  intros
  sorry

end NUMINAMATH_GPT_train_speed_l1062_106210


namespace NUMINAMATH_GPT_connie_initial_marbles_l1062_106252

theorem connie_initial_marbles (marbles_given : ℕ) (marbles_left : ℕ) (initial_marbles : ℕ) 
    (h1 : marbles_given = 183) (h2 : marbles_left = 593) : initial_marbles = 776 :=
by
  sorry

end NUMINAMATH_GPT_connie_initial_marbles_l1062_106252


namespace NUMINAMATH_GPT_min_value_inverse_sum_l1062_106266

theorem min_value_inverse_sum (a m n : ℝ) (a_pos : 0 < a) (a_ne_one : a ≠ 1) (mn_pos : 0 < m * n) :
  (a^(1-1) = 1) ∧ (m + n = 1) → (1/m + 1/n) = 4 :=
by
  sorry

end NUMINAMATH_GPT_min_value_inverse_sum_l1062_106266


namespace NUMINAMATH_GPT_combined_total_l1062_106248

-- Definitions for the problem conditions
def marks_sandcastles : ℕ := 20
def towers_per_marks_sandcastle : ℕ := 10

def jeffs_multiplier : ℕ := 3
def towers_per_jeffs_sandcastle : ℕ := 5

-- Definitions derived from conditions
def jeffs_sandcastles : ℕ := jeffs_multiplier * marks_sandcastles
def marks_towers : ℕ := marks_sandcastles * towers_per_marks_sandcastle
def jeffs_towers : ℕ := jeffs_sandcastles * towers_per_jeffs_sandcastle

-- Question translated to a Lean theorem
theorem combined_total : 
  (marks_sandcastles + jeffs_sandcastles) + (marks_towers + jeffs_towers) = 580 :=
by
  -- The proof would go here
  sorry

end NUMINAMATH_GPT_combined_total_l1062_106248


namespace NUMINAMATH_GPT_fraction_of_historical_fiction_new_releases_l1062_106244

theorem fraction_of_historical_fiction_new_releases (total_books : ℕ) (p1 p2 p3 : ℕ) (frac_hist_fic : Rat) (frac_new_hist_fic : Rat) (frac_new_non_hist_fic : Rat) 
  (h1 : total_books > 0) (h2 : frac_hist_fic = 40 / 100) (h3 : frac_new_hist_fic = 40 / 100) (h4 : frac_new_non_hist_fic = 40 / 100) 
  (h5 : p1 = frac_hist_fic * total_books) (h6 : p2 = frac_new_hist_fic * p1) (h7 : p3 = frac_new_non_hist_fic * (total_books - p1)) :
  p2 / (p2 + p3) = 2 / 5 :=
by
  sorry

end NUMINAMATH_GPT_fraction_of_historical_fiction_new_releases_l1062_106244


namespace NUMINAMATH_GPT_distance_center_to_line_circle_l1062_106230

noncomputable def circle_center : ℝ × ℝ := (2, Real.pi / 2)

noncomputable def distance_from_center_to_line (radius : ℝ) (center : ℝ × ℝ) : ℝ :=
  radius * Real.sin (center.snd - Real.pi / 3)

theorem distance_center_to_line_circle : distance_from_center_to_line 2 circle_center = 1 := by
  sorry

end NUMINAMATH_GPT_distance_center_to_line_circle_l1062_106230


namespace NUMINAMATH_GPT_repeating_decimal_to_fraction_l1062_106211

theorem repeating_decimal_to_fraction : (0.36 : ℝ) = (11 / 30 : ℝ) :=
sorry

end NUMINAMATH_GPT_repeating_decimal_to_fraction_l1062_106211


namespace NUMINAMATH_GPT_floor_of_neg_seven_fourths_l1062_106219

theorem floor_of_neg_seven_fourths : (Int.floor (-7/4 : ℚ)) = -2 :=
by
  sorry

end NUMINAMATH_GPT_floor_of_neg_seven_fourths_l1062_106219


namespace NUMINAMATH_GPT_area_of_triangle_KBC_l1062_106226

noncomputable def length_FE := 7
noncomputable def length_BC := 7
noncomputable def length_JB := 5
noncomputable def length_BK := 5

theorem area_of_triangle_KBC : (1 / 2 : ℝ) * length_BC * length_BK = 17.5 := by
  -- conditions: 
  -- 1. Hexagon ABCDEF is equilateral with each side of length s.
  -- 2. Squares ABJI and FEHG are formed outside the hexagon with areas 25 and 49 respectively.
  -- 3. Triangle JBK is equilateral.
  -- 4. FE = BC.
  sorry

end NUMINAMATH_GPT_area_of_triangle_KBC_l1062_106226


namespace NUMINAMATH_GPT_train_crossing_time_l1062_106283

-- Definitions from conditions
def length_of_train : ℕ := 120
def length_of_bridge : ℕ := 150
def speed_kmph : ℕ := 36
def speed_mps : ℕ := speed_kmph * 1000 / 3600 -- Convert km/h to m/s
def total_distance : ℕ := length_of_train + length_of_bridge

-- Theorem statement
theorem train_crossing_time : total_distance / speed_mps = 27 := by
  sorry

end NUMINAMATH_GPT_train_crossing_time_l1062_106283


namespace NUMINAMATH_GPT_cassidy_posters_l1062_106291

theorem cassidy_posters (p_two_years_ago : ℕ) (p_double : ℕ) (p_current : ℕ) (p_added : ℕ) 
    (h1 : p_two_years_ago = 14) 
    (h2 : p_double = 2 * p_two_years_ago)
    (h3 : p_current = 22)
    (h4 : p_added = p_double - p_current) : 
    p_added = 6 := 
by
  sorry

end NUMINAMATH_GPT_cassidy_posters_l1062_106291


namespace NUMINAMATH_GPT_sanity_proof_l1062_106278

-- Define the characters and their sanity status as propositions
variables (Griffin QuasiTurtle Lobster : Prop)

-- Conditions
axiom Lobster_thinks : (Griffin ∧ ¬QuasiTurtle ∧ ¬Lobster) ∨ (¬Griffin ∧ QuasiTurtle ∧ ¬Lobster) ∨ (¬Griffin ∧ ¬QuasiTurtle ∧ Lobster)
axiom QuasiTurtle_thinks : Griffin

-- Statement to prove
theorem sanity_proof : ¬Griffin ∧ ¬QuasiTurtle ∧ ¬Lobster :=
by {
  sorry
}

end NUMINAMATH_GPT_sanity_proof_l1062_106278


namespace NUMINAMATH_GPT_tailor_cut_difference_l1062_106236

theorem tailor_cut_difference :
  (7 / 8 + 11 / 12) - (5 / 6 + 3 / 4) = 5 / 24 :=
by
  sorry

end NUMINAMATH_GPT_tailor_cut_difference_l1062_106236


namespace NUMINAMATH_GPT_max_value_of_quadratic_l1062_106247

theorem max_value_of_quadratic :
  ∀ z : ℝ, -6*z^2 + 24*z - 12 ≤ 12 :=
by
  sorry

end NUMINAMATH_GPT_max_value_of_quadratic_l1062_106247


namespace NUMINAMATH_GPT_geometric_sequence_sum_l1062_106222

theorem geometric_sequence_sum (S : ℕ → ℚ) (a : ℕ → ℚ)
  (h1 : S 4 = 1)
  (h2 : S 8 = 3)
  (h3 : ∀ n, S (n + 4) - S n = a (n + 1) + a (n + 2) + a (n + 3) + a (n + 4)) :
  a 17 + a 18 + a 19 + a 20 = 16 :=
by
  -- Insert your proof here.
  sorry

end NUMINAMATH_GPT_geometric_sequence_sum_l1062_106222


namespace NUMINAMATH_GPT_password_encryption_l1062_106239

variables (a b x : ℝ)

theorem password_encryption :
  3 * a * (x^2 - 1) - 3 * b * (x^2 - 1) = 3 * (x + 1) * (x - 1) * (a - b) :=
by sorry

end NUMINAMATH_GPT_password_encryption_l1062_106239


namespace NUMINAMATH_GPT_distance_AB_polar_l1062_106253

open Real

theorem distance_AB_polar (A B : ℝ × ℝ) (θ₁ θ₂ : ℝ) (hA : A = (4, θ₁)) (hB : B = (12, θ₂))
  (hθ : θ₁ - θ₂ = π / 3) : dist (4 * cos θ₁, 4 * sin θ₁) (12 * cos θ₂, 12 * sin θ₂) = 4 * sqrt 13 :=
by
  sorry

end NUMINAMATH_GPT_distance_AB_polar_l1062_106253


namespace NUMINAMATH_GPT_sandy_shopping_l1062_106206

variable (X : ℝ)

theorem sandy_shopping (h : 0.70 * X = 210) : X = 300 := by
  sorry

end NUMINAMATH_GPT_sandy_shopping_l1062_106206


namespace NUMINAMATH_GPT_circle_representation_l1062_106228

theorem circle_representation (m : ℝ) : (∃ x y : ℝ, x^2 + y^2 - 2 * m * x + 2 * m^2 + 2 * m - 3 = 0) ↔ m ∈ Set.Ioo (-3 : ℝ) (1 / 2 : ℝ) :=
by
  sorry

end NUMINAMATH_GPT_circle_representation_l1062_106228


namespace NUMINAMATH_GPT_percentage_more_than_l1062_106212

variable (P Q : ℝ)

-- P gets 20% more than Q
def getsMoreThan (P Q : ℝ) : Prop :=
  P = 1.20 * Q

-- Q gets 20% less than P
def getsLessThan (Q P : ℝ) : Prop :=
  Q = 0.80 * P

theorem percentage_more_than :
  getsLessThan Q P → getsMoreThan P Q := 
sorry

end NUMINAMATH_GPT_percentage_more_than_l1062_106212


namespace NUMINAMATH_GPT_equal_area_of_second_square_l1062_106240

/-- 
In an isosceles right triangle with legs of length 25√2 cm, if a square is inscribed such that two 
of its vertices lie on one leg and one vertex on each of the hypotenuse and the other leg, 
and the area of the square is 625 cm², prove that the area of another inscribed square 
(with one vertex each on the hypotenuse and one leg, and two vertices on the other leg) is also 625 cm².
-/
theorem equal_area_of_second_square 
  (a b : ℝ) (h1 : a = 25 * Real.sqrt 2)  
  (h2 : b = 625) :
  ∃ c : ℝ, c = 625 :=
by
  sorry

end NUMINAMATH_GPT_equal_area_of_second_square_l1062_106240


namespace NUMINAMATH_GPT_Maryann_total_minutes_worked_l1062_106263

theorem Maryann_total_minutes_worked (c a t : ℕ) (h1 : c = 70) (h2 : a = 7 * c) (h3 : t = c + a) : t = 560 := by
  sorry

end NUMINAMATH_GPT_Maryann_total_minutes_worked_l1062_106263


namespace NUMINAMATH_GPT_robot_trajectory_no_intersection_l1062_106238

noncomputable def parabola_equation (x y : ℝ) : Prop := y^2 = 4 * x
noncomputable def line_equation (x y k : ℝ) : Prop := y = k * (x + 1)

theorem robot_trajectory_no_intersection (k : ℝ) :
  (∀ x y : ℝ, parabola_equation x y → ¬ line_equation x y k) →
  (k > 1 ∨ k < -1) :=
by
  sorry

end NUMINAMATH_GPT_robot_trajectory_no_intersection_l1062_106238


namespace NUMINAMATH_GPT_problem_l1062_106216

variable (x y : ℝ)

-- Define the given condition
def condition : Prop := |x + 5| + (y - 4)^2 = 0

-- State the theorem we need to prove
theorem problem (h : condition x y) : (x + y)^99 = -1 := sorry

end NUMINAMATH_GPT_problem_l1062_106216


namespace NUMINAMATH_GPT_mats_length_l1062_106258

open Real

theorem mats_length (r : ℝ) (n : ℤ) (w : ℝ) (y : ℝ) (h₁ : r = 6) (h₂ : n = 8) (h₃ : w = 1):
  y = 6 * sqrt (2 - sqrt 2) :=
sorry

end NUMINAMATH_GPT_mats_length_l1062_106258


namespace NUMINAMATH_GPT_isosceles_right_triangle_ratio_l1062_106272

theorem isosceles_right_triangle_ratio (a : ℝ) (h : a > 0) : (2 * a) / (Real.sqrt (a^2 + a^2)) = Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_GPT_isosceles_right_triangle_ratio_l1062_106272


namespace NUMINAMATH_GPT_percent_decrease_first_year_l1062_106260

theorem percent_decrease_first_year (x : ℝ) (hx : 0 ≤ x ∧ x ≤ 100) 
  (h_second_year : 0.9 * (100 - x) = 54) : x = 40 :=
by sorry

end NUMINAMATH_GPT_percent_decrease_first_year_l1062_106260


namespace NUMINAMATH_GPT_part_one_part_two_part_three_l1062_106241

def f(x : ℝ) := x^2 - 1
def g(a x : ℝ) := a * |x - 1|

-- (I)
theorem part_one (a : ℝ) : 
  ((∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ |f x₁| = g a x₁ ∧ |f x₂| = g a x₂) ↔ (a = 0 ∨ a = 2)) :=
sorry

-- (II)
theorem part_two (a : ℝ) : 
  (∀ x : ℝ, f x ≥ g a x) ↔ (a <= -2) :=
sorry

-- (III)
def G(a x : ℝ) := |f x| + g a x

theorem part_three (a : ℝ) (h : a < 0) : 
  (∀ x ∈ [-2, 2], G a x ≤ if a <= -3 then 0 else 3 + a) :=
sorry

end NUMINAMATH_GPT_part_one_part_two_part_three_l1062_106241


namespace NUMINAMATH_GPT_find_value_l1062_106274

noncomputable def f : ℝ → ℝ := sorry

axiom even_function : ∀ x : ℝ, f x = f (-x)
axiom periodicity_condition : ∀ x : ℝ, f (2 + x) = f (-x)
axiom value_at_half : f (1/2) = 1/2

theorem find_value : f (2023 / 2) = 1/2 := by
  sorry

end NUMINAMATH_GPT_find_value_l1062_106274


namespace NUMINAMATH_GPT_simplify_fraction_l1062_106227

theorem simplify_fraction :
  (144 : ℤ) / (1296 : ℤ) = 1 / 9 := 
by sorry

end NUMINAMATH_GPT_simplify_fraction_l1062_106227


namespace NUMINAMATH_GPT_max_a_value_l1062_106270

noncomputable def f (x : ℝ) (a b : ℝ) : ℝ := x^2 + a * x + b

theorem max_a_value :
  (∀ x : ℝ, ∃ y : ℝ, f y a b = f x a b + y) → a ≤ 1/2 :=
by
  sorry

end NUMINAMATH_GPT_max_a_value_l1062_106270


namespace NUMINAMATH_GPT_articles_selling_price_to_cost_price_eq_l1062_106201

theorem articles_selling_price_to_cost_price_eq (C N : ℝ) (h_gain : 2 * C * N = 20 * C) : N = 10 :=
by
  sorry

end NUMINAMATH_GPT_articles_selling_price_to_cost_price_eq_l1062_106201


namespace NUMINAMATH_GPT_students_speaking_Gujarati_l1062_106243

theorem students_speaking_Gujarati 
  (total_students : ℕ)
  (students_Hindi : ℕ)
  (students_Marathi : ℕ)
  (students_two_languages : ℕ)
  (students_all_three_languages : ℕ)
  (students_total_set: 22 = total_students)
  (students_H_set: 15 = students_Hindi)
  (students_M_set: 6 = students_Marathi)
  (students_two_set: 2 = students_two_languages)
  (students_all_three_set: 1 = students_all_three_languages) :
  ∃ (students_Gujarati : ℕ), 
  22 = students_Gujarati + 15 + 6 - 2 + 1 ∧ students_Gujarati = 2 :=
by
  sorry

end NUMINAMATH_GPT_students_speaking_Gujarati_l1062_106243


namespace NUMINAMATH_GPT_john_remaining_money_l1062_106262

variable (q : ℝ)
variable (number_of_small_pizzas number_of_large_pizzas number_of_drinks : ℕ)
variable (cost_of_drink cost_of_small_pizza cost_of_large_pizza dollars_left : ℝ)

def john_purchases := number_of_small_pizzas = 2 ∧
                      number_of_large_pizzas = 1 ∧
                      number_of_drinks = 4 ∧
                      cost_of_drink = q ∧
                      cost_of_small_pizza = q ∧
                      cost_of_large_pizza = 4 * q ∧
                      dollars_left = 50 - (4 * q + 2 * q + 4 * q)

theorem john_remaining_money : john_purchases q 2 1 4 q q (4 * q) (50 - 10 * q) :=
by
  sorry

end NUMINAMATH_GPT_john_remaining_money_l1062_106262


namespace NUMINAMATH_GPT_angles_with_same_terminal_side_l1062_106290

theorem angles_with_same_terminal_side (k : ℤ) :
  {θ : ℝ | ∃ k : ℤ, θ = k * 360 + 260} = 
  {θ : ℝ | ∃ k : ℤ, θ = k * 360 + (-460 % 360)} :=
by sorry

end NUMINAMATH_GPT_angles_with_same_terminal_side_l1062_106290


namespace NUMINAMATH_GPT_number_of_triangles_for_second_star_l1062_106245

theorem number_of_triangles_for_second_star (a b : ℝ) (h₁ : a + b + 90 = 180) (h₂ : 5 * (360 / 5) = 360) :
  360 / (180 - 90 - (360 / 5)) = 20 :=
by
  sorry

end NUMINAMATH_GPT_number_of_triangles_for_second_star_l1062_106245


namespace NUMINAMATH_GPT_no_b_for_221_square_l1062_106264

theorem no_b_for_221_square (b : ℕ) (h : b ≥ 3) :
  ¬ ∃ n : ℕ, 2 * b^2 + 2 * b + 1 = n^2 :=
by
  sorry

end NUMINAMATH_GPT_no_b_for_221_square_l1062_106264


namespace NUMINAMATH_GPT_wall_ratio_l1062_106202

theorem wall_ratio (V : ℝ) (B : ℝ) (H : ℝ) (x : ℝ) (L : ℝ) :
  V = 12.8 →
  B = 0.4 →
  H = 5 * B →
  L = x * H →
  V = B * H * L →
  x = 4 ∧ L / H = 4 :=
by
  intros hV hB hH hL hVL
  sorry

end NUMINAMATH_GPT_wall_ratio_l1062_106202


namespace NUMINAMATH_GPT_solve_for_x_l1062_106215

theorem solve_for_x (x : ℚ) (h : (3 * x + 5) / 7 = 13) : x = 86 / 3 :=
sorry

end NUMINAMATH_GPT_solve_for_x_l1062_106215


namespace NUMINAMATH_GPT_fraction_of_yellow_balls_l1062_106204

theorem fraction_of_yellow_balls
  (total_balls : ℕ)
  (fraction_green : ℚ)
  (fraction_blue : ℚ)
  (number_blue : ℕ)
  (number_white : ℕ)
  (total_balls_eq : total_balls = number_blue * (1 / fraction_blue))
  (fraction_green_eq : fraction_green = 1 / 4)
  (fraction_blue_eq : fraction_blue = 1 / 8)
  (number_white_eq : number_white = 26)
  (number_blue_eq : number_blue = 6) :
  (total_balls - (total_balls * fraction_green + number_blue + number_white)) / total_balls = 1 / 12 :=
by
  sorry

end NUMINAMATH_GPT_fraction_of_yellow_balls_l1062_106204


namespace NUMINAMATH_GPT_arithmetic_sequence_sum_l1062_106229

variable (a : ℕ → ℝ)
variable (d : ℝ)

-- The sequence is arithmetic
def is_arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) := ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (h_arith : is_arithmetic_sequence a d) (h_condition : a 2 + a 10 = 16) :
  a 4 + a 8 = 16 :=
sorry

end NUMINAMATH_GPT_arithmetic_sequence_sum_l1062_106229


namespace NUMINAMATH_GPT_f_3_neg3div2_l1062_106275

noncomputable def f : ℝ → ℝ :=
sorry

axiom odd_f : ∀ x : ℝ, f (-x) = -f x
axiom symm_f : ∀ t : ℝ, f t = f (1 - t)
axiom restriction_f : ∀ x : ℝ, 0 ≤ x ∧ x ≤ 1/2 → f x = -x^2

theorem f_3_neg3div2 :
  f 3 + f (-3/2) = -1/4 :=
sorry

end NUMINAMATH_GPT_f_3_neg3div2_l1062_106275


namespace NUMINAMATH_GPT_bacteria_count_correct_l1062_106224

-- Define the initial number of bacteria
def initial_bacteria : ℕ := 800

-- Define the doubling time in hours
def doubling_time : ℕ := 3

-- Define the function that calculates the number of bacteria after t hours
noncomputable def bacteria_after (t : ℕ) : ℕ :=
  initial_bacteria * 2 ^ (t / doubling_time)

-- Define the target number of bacteria
def target_bacteria : ℕ := 51200

-- Define the specific time we want to prove the bacteria count equals the target
def specific_time : ℕ := 18

-- Prove that after 18 hours, there will be exactly 51,200 bacteria
theorem bacteria_count_correct : bacteria_after specific_time = target_bacteria :=
  sorry

end NUMINAMATH_GPT_bacteria_count_correct_l1062_106224


namespace NUMINAMATH_GPT_proof_problem_l1062_106287

noncomputable def problem (x y : ℝ) : ℝ :=
  let A := 2 * x + y
  let B := 2 * x - y
  (A ^ 2 - B ^ 2) * (x - 2 * y)

theorem proof_problem : problem (-1) 2 = 80 := by
  sorry

end NUMINAMATH_GPT_proof_problem_l1062_106287


namespace NUMINAMATH_GPT_extra_interest_l1062_106298

def principal : ℝ := 7000
def rate1 : ℝ := 0.18
def rate2 : ℝ := 0.12
def time : ℝ := 2

def interest (P R T : ℝ) : ℝ := P * R * T

theorem extra_interest :
  interest principal rate1 time - interest principal rate2 time = 840 := by
  sorry

end NUMINAMATH_GPT_extra_interest_l1062_106298


namespace NUMINAMATH_GPT_population_at_300pm_l1062_106220

namespace BacteriaProblem

def initial_population : ℕ := 50
def time_increments_to_220pm : ℕ := 4   -- 4 increments of 5 mins each till 2:20 p.m.
def time_increments_to_300pm : ℕ := 2   -- 2 increments of 10 mins each till 3:00 p.m.

def growth_factor_before_220pm : ℕ := 3
def growth_factor_after_220pm : ℕ := 2

theorem population_at_300pm :
  initial_population * growth_factor_before_220pm^time_increments_to_220pm *
  growth_factor_after_220pm^time_increments_to_300pm = 16200 :=
by
  sorry

end BacteriaProblem

end NUMINAMATH_GPT_population_at_300pm_l1062_106220


namespace NUMINAMATH_GPT_susan_took_longer_l1062_106296
variables (M S J T x : ℕ)
theorem susan_took_longer (h1 : M = 2 * S)
                         (h2 : S = J + x)
                         (h3 : J = 30)
                         (h4 : T = M - 7)
                         (h5 : M + S + J + T = 223) : x = 10 :=
sorry

end NUMINAMATH_GPT_susan_took_longer_l1062_106296


namespace NUMINAMATH_GPT_range_of_a_l1062_106218

variable (a : ℝ)
variable (x y : ℝ)

def system_of_equations := 
  (5 * x + 2 * y = 11 * a + 18) ∧ 
  (2 * x - 3 * y = 12 * a - 8) ∧
  (x > 0) ∧ 
  (y > 0)

theorem range_of_a (h : system_of_equations a x y) : 
  - (2:ℝ) / 3 < a ∧ a < 2 :=
sorry

end NUMINAMATH_GPT_range_of_a_l1062_106218


namespace NUMINAMATH_GPT_sqrt_72_eq_6_sqrt_2_l1062_106255

theorem sqrt_72_eq_6_sqrt_2 : Real.sqrt 72 = 6 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_GPT_sqrt_72_eq_6_sqrt_2_l1062_106255


namespace NUMINAMATH_GPT_polynomial_identity_l1062_106235

theorem polynomial_identity (a : ℝ) (h₁ : a^5 + 5 * a^4 + 10 * a^3 + 3 * a^2 - 9 * a - 6 = 0) (h₂ : a ≠ -1) : (a + 1)^3 = 7 :=
sorry

end NUMINAMATH_GPT_polynomial_identity_l1062_106235


namespace NUMINAMATH_GPT_root_conditions_l1062_106289

theorem root_conditions (m : ℝ) : (∃ a b : ℝ, a < 2 ∧ b > 2 ∧ a * b = -1 ∧ a + b = m) ↔ m > 3 / 2 := sorry

end NUMINAMATH_GPT_root_conditions_l1062_106289


namespace NUMINAMATH_GPT_allison_marbles_l1062_106257

theorem allison_marbles (A B C : ℕ) (h1 : B = A + 8) (h2 : C = 3 * B) (h3 : C + A = 136) : 
  A = 28 :=
by
  sorry

end NUMINAMATH_GPT_allison_marbles_l1062_106257


namespace NUMINAMATH_GPT_same_solutions_implies_k_value_l1062_106297

theorem same_solutions_implies_k_value (k : ℤ) : (∀ x : ℤ, 2 * x = 4 ↔ 3 * x + k = -2) → k = -8 :=
by
  sorry

end NUMINAMATH_GPT_same_solutions_implies_k_value_l1062_106297


namespace NUMINAMATH_GPT_hawks_loss_percentage_is_30_l1062_106293

-- Define the variables and the conditions
def matches_won (x : ℕ) : ℕ := 7 * x
def matches_lost (x : ℕ) : ℕ := 3 * x
def total_matches (x : ℕ) : ℕ := matches_won x + matches_lost x
def percent_lost (x : ℕ) : ℕ := (matches_lost x * 100) / total_matches x

-- The goal statement in Lean 4
theorem hawks_loss_percentage_is_30 (x : ℕ) (h : x > 0) : percent_lost x = 30 :=
by sorry

end NUMINAMATH_GPT_hawks_loss_percentage_is_30_l1062_106293


namespace NUMINAMATH_GPT_olympiad_not_possible_l1062_106276

theorem olympiad_not_possible (x : ℕ) (y : ℕ) (h1 : x + y = 1000) (h2 : y = x + 43) : false := by
  sorry

end NUMINAMATH_GPT_olympiad_not_possible_l1062_106276


namespace NUMINAMATH_GPT_quadratic_equation_factored_form_l1062_106271

theorem quadratic_equation_factored_form : 
  ∀ x : ℝ, x^2 - 6 * x - 6 = 0 ↔ (x - 3)^2 = 15 := 
by 
  sorry

end NUMINAMATH_GPT_quadratic_equation_factored_form_l1062_106271


namespace NUMINAMATH_GPT_ratio_a_b_c_l1062_106232

-- Given condition 14(a^2 + b^2 + c^2) = (a + 2b + 3c)^2
theorem ratio_a_b_c (a b c : ℝ) (h : 14 * (a^2 + b^2 + c^2) = (a + 2 * b + 3 * c)^2) : 
  a / b = 1 / 2 ∧ b / c = 2 / 3 :=
by 
  sorry

end NUMINAMATH_GPT_ratio_a_b_c_l1062_106232


namespace NUMINAMATH_GPT_total_germs_l1062_106281

-- Define variables and constants
namespace BiologyLab

def petri_dishes : ℕ := 75
def germs_per_dish : ℕ := 48

-- The goal is to prove that the total number of germs is as expected.
theorem total_germs : (petri_dishes * germs_per_dish) = 3600 :=
by
  -- Proof is omitted for this example
  sorry

end BiologyLab

end NUMINAMATH_GPT_total_germs_l1062_106281


namespace NUMINAMATH_GPT_train_speed_40_l1062_106205

-- Definitions for the conditions
def passes_pole (L V : ℝ) := V = L / 8
def passes_stationary_train (L V : ℝ) := V = (L + 400) / 18

-- The theorem we want to prove
theorem train_speed_40 (L V : ℝ) (h1 : passes_pole L V) (h2 : passes_stationary_train L V) : V = 40 := 
sorry

end NUMINAMATH_GPT_train_speed_40_l1062_106205


namespace NUMINAMATH_GPT_painting_area_l1062_106285

theorem painting_area (wall_height wall_length bookshelf_height bookshelf_length : ℝ)
  (h_wall_height : wall_height = 10)
  (h_wall_length : wall_length = 15)
  (h_bookshelf_height : bookshelf_height = 3)
  (h_bookshelf_length : bookshelf_length = 5) :
  wall_height * wall_length - bookshelf_height * bookshelf_length = 135 := 
by
  sorry

end NUMINAMATH_GPT_painting_area_l1062_106285


namespace NUMINAMATH_GPT_number_of_camels_l1062_106292

theorem number_of_camels (hens goats keepers camel_feet heads total_feet : ℕ)
  (h_hens : hens = 50) (h_goats : goats = 45) (h_keepers : keepers = 15)
  (h_feet_diff : total_feet = heads + 224)
  (h_heads : heads = hens + goats + keepers)
  (h_hens_feet : hens * 2 = 100)
  (h_goats_feet : goats * 4 = 180)
  (h_keepers_feet : keepers * 2 = 30)
  (h_camels_feet : camel_feet = 24)
  (h_total_feet : total_feet = 334)
  (h_feet_without_camels : 100 + 180 + 30 = 310) :
  camel_feet / 4 = 6 := sorry

end NUMINAMATH_GPT_number_of_camels_l1062_106292


namespace NUMINAMATH_GPT_math_proof_problem_l1062_106203

variable (a_n : ℕ → ℤ) (S_n : ℕ → ℤ)
variable (a_1 d : ℤ)
variable (n : ℕ)

def arith_seq : Prop := ∀ n, a_n n = a_1 + (n - 1) * d

def sum_arith_seq : Prop := ∀ n, S_n n = n * (a_1 + (n - 1) * d / 2)

def condition1 : Prop := a_n 5 + a_n 9 = -2

def condition2 : Prop := S_n 3 = 57

noncomputable def general_formula : Prop := ∀ n, a_n n = 27 - 4 * n

noncomputable def max_S_n : Prop := ∀ n, S_n n ≤ 78 ∧ ∃ n, S_n n = 78

theorem math_proof_problem : 
  arith_seq a_n a_1 d ∧ sum_arith_seq S_n a_1 d ∧ condition1 a_n ∧ condition2 S_n 
  → general_formula a_n ∧ max_S_n S_n := 
sorry

end NUMINAMATH_GPT_math_proof_problem_l1062_106203


namespace NUMINAMATH_GPT_Seohyeon_l1062_106250

-- Define the distances in their respective units
def d_Kunwoo_km : ℝ := 3.97
def d_Seohyeon_m : ℝ := 4028

-- Convert Kunwoo's distance to meters
def d_Kunwoo_m : ℝ := d_Kunwoo_km * 1000

-- The main theorem we need to prove
theorem Seohyeon's_distance_longer_than_Kunwoo's :
  d_Seohyeon_m > d_Kunwoo_m :=
by
  sorry

end NUMINAMATH_GPT_Seohyeon_l1062_106250


namespace NUMINAMATH_GPT_max_value_is_27_l1062_106231

noncomputable def max_value_of_expression (a b c : ℝ) : ℝ :=
  (a - b)^2 + (b - c)^2 + (c - a)^2

theorem max_value_is_27 (a b c : ℝ)
  (h : a^2 + b^2 + c^2 = 9) : max_value_of_expression a b c = 27 :=
by
  sorry

end NUMINAMATH_GPT_max_value_is_27_l1062_106231


namespace NUMINAMATH_GPT_twice_brother_age_l1062_106233

theorem twice_brother_age (current_my_age : ℕ) (current_brother_age : ℕ) (years : ℕ) :
  current_my_age = 20 →
  (current_my_age + years) + (current_brother_age + years) = 45 →
  current_my_age + years = 2 * (current_brother_age + years) →
  years = 10 :=
by 
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_twice_brother_age_l1062_106233


namespace NUMINAMATH_GPT_total_sequins_is_162_l1062_106280

/-- Jane sews 6 rows of 8 blue sequins each. -/
def rows_of_blue_sequins : Nat := 6
def sequins_per_blue_row : Nat := 8
def total_blue_sequins : Nat := rows_of_blue_sequins * sequins_per_blue_row

/-- Jane sews 5 rows of 12 purple sequins each. -/
def rows_of_purple_sequins : Nat := 5
def sequins_per_purple_row : Nat := 12
def total_purple_sequins : Nat := rows_of_purple_sequins * sequins_per_purple_row

/-- Jane sews 9 rows of 6 green sequins each. -/
def rows_of_green_sequins : Nat := 9
def sequins_per_green_row : Nat := 6
def total_green_sequins : Nat := rows_of_green_sequins * sequins_per_green_row

/-- The total number of sequins Jane adds to her costume. -/
def total_sequins : Nat := total_blue_sequins + total_purple_sequins + total_green_sequins

theorem total_sequins_is_162 : total_sequins = 162 := 
by
  sorry

end NUMINAMATH_GPT_total_sequins_is_162_l1062_106280


namespace NUMINAMATH_GPT_count_whole_numbers_in_interval_l1062_106294

theorem count_whole_numbers_in_interval :
  let a := (7 / 4)
  let b := (3 * Real.pi)
  ∃ n : ℕ, n = 8 ∧ ∀ x : ℕ, a < x ∧ x < b → 2 ≤ x ∧ x ≤ 9 := by
  sorry

end NUMINAMATH_GPT_count_whole_numbers_in_interval_l1062_106294


namespace NUMINAMATH_GPT_mike_total_cans_l1062_106261

theorem mike_total_cans (monday_cans : ℕ) (tuesday_cans : ℕ) (total_cans : ℕ) : 
  monday_cans = 71 ∧ tuesday_cans = 27 ∧ total_cans = monday_cans + tuesday_cans → total_cans = 98 :=
by
  sorry

end NUMINAMATH_GPT_mike_total_cans_l1062_106261


namespace NUMINAMATH_GPT_opposite_of_neg_half_is_half_l1062_106217

theorem opposite_of_neg_half_is_half : -(-1 / 2) = (1 / 2) :=
by
  sorry

end NUMINAMATH_GPT_opposite_of_neg_half_is_half_l1062_106217


namespace NUMINAMATH_GPT_total_songs_isabel_bought_l1062_106221

theorem total_songs_isabel_bought
  (country_albums pop_albums : ℕ)
  (songs_per_album : ℕ)
  (h1 : country_albums = 6)
  (h2 : pop_albums = 2)
  (h3 : songs_per_album = 9) : 
  (country_albums + pop_albums) * songs_per_album = 72 :=
by
  -- We provide only the statement, no proof as per the instruction
  sorry

end NUMINAMATH_GPT_total_songs_isabel_bought_l1062_106221


namespace NUMINAMATH_GPT_kittens_remaining_l1062_106242

theorem kittens_remaining (original_kittens : ℕ) (kittens_given_away : ℕ) 
  (h1 : original_kittens = 8) (h2 : kittens_given_away = 4) : 
  original_kittens - kittens_given_away = 4 := by
  sorry

end NUMINAMATH_GPT_kittens_remaining_l1062_106242
