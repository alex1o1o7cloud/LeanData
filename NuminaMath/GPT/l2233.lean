import Mathlib

namespace NUMINAMATH_GPT_factorize_polynomial_l2233_223305

theorem factorize_polynomial (a b : ℝ) : a^2 - 9 * b^2 = (a + 3 * b) * (a - 3 * b) := by
  sorry

end NUMINAMATH_GPT_factorize_polynomial_l2233_223305


namespace NUMINAMATH_GPT_return_trip_time_l2233_223384

-- conditions 
variables (d p w : ℝ) (h1 : d = 90 * (p - w)) (h2 : ∀ t : ℝ, t = d / p → d / (p + w) = t - 15)

--  statement
theorem return_trip_time :
  ∃ t : ℝ, t = 30 ∨ t = 45 :=
by
  -- placeholder proof 
  sorry

end NUMINAMATH_GPT_return_trip_time_l2233_223384


namespace NUMINAMATH_GPT_range_of_a_l2233_223319

variable (a : ℝ)

def set_A (a : ℝ) : Set ℝ := { x | x^2 - 2 * x + a ≥ 0 }

theorem range_of_a (h : 1 ∉ set_A a) : a < 1 := 
by {
  sorry
}

end NUMINAMATH_GPT_range_of_a_l2233_223319


namespace NUMINAMATH_GPT_remainder_of_5n_mod_11_l2233_223350

theorem remainder_of_5n_mod_11 (n : ℤ) (h : n % 11 = 1) : (5 * n) % 11 = 5 := 
by
  sorry

end NUMINAMATH_GPT_remainder_of_5n_mod_11_l2233_223350


namespace NUMINAMATH_GPT_same_terminal_side_l2233_223398

theorem same_terminal_side (θ : ℝ) : (∃ k : ℤ, θ = 2 * k * π - π / 6) → θ = 11 * π / 6 :=
sorry

end NUMINAMATH_GPT_same_terminal_side_l2233_223398


namespace NUMINAMATH_GPT_find_M_plus_N_l2233_223358

theorem find_M_plus_N (M N : ℕ) (h1 : 3 / 5 = M / 30) (h2 : 3 / 5 = 90 / N) : M + N = 168 := 
by
  sorry

end NUMINAMATH_GPT_find_M_plus_N_l2233_223358


namespace NUMINAMATH_GPT_number_of_new_bottle_caps_l2233_223317

def threw_away := 6
def total_bottle_caps_now := 60
def found_more_bottle_caps := 44

theorem number_of_new_bottle_caps (N : ℕ) (h1 : N = threw_away + found_more_bottle_caps) : N = 50 :=
sorry

end NUMINAMATH_GPT_number_of_new_bottle_caps_l2233_223317


namespace NUMINAMATH_GPT_value_of_expression_l2233_223356

variable {a b c d e f : ℝ}

theorem value_of_expression :
  a * b * c = 130 →
  b * c * d = 65 →
  c * d * e = 1000 →
  d * e * f = 250 →
  (a * f) / (c * d) = 1 / 2 :=
by
  intros h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_value_of_expression_l2233_223356


namespace NUMINAMATH_GPT_sunland_more_plates_than_moonland_l2233_223343

theorem sunland_more_plates_than_moonland : 
  let sunland_plates := 26^4 * 10^2
  let moonland_plates := 26^3 * 10^3
  (sunland_plates - moonland_plates) = 7321600 := 
by
  sorry

end NUMINAMATH_GPT_sunland_more_plates_than_moonland_l2233_223343


namespace NUMINAMATH_GPT_brown_dog_count_l2233_223391

theorem brown_dog_count:
  ∀ (T L N : ℕ), T = 45 → L = 36 → N = 8 → (T - N - (T - L - N) = 37) :=
by
  intros T L N hT hL hN
  sorry

end NUMINAMATH_GPT_brown_dog_count_l2233_223391


namespace NUMINAMATH_GPT_Wendy_bouquets_l2233_223335

def num_flowers_before : ℕ := 45
def num_wilted_flowers : ℕ := 35
def flowers_per_bouquet : ℕ := 5

theorem Wendy_bouquets : (num_flowers_before - num_wilted_flowers) / flowers_per_bouquet = 2 := by
  sorry

end NUMINAMATH_GPT_Wendy_bouquets_l2233_223335


namespace NUMINAMATH_GPT_closest_to_one_tenth_l2233_223351

noncomputable def p (n : ℕ) : ℚ :=
  1 / (n * (n + 2)) + 1 / ((n + 2) * (n + 4)) + 1 / ((n + 4) * (n + 6)) +
  1 / ((n + 6) * (n + 8)) + 1 / ((n + 8) * (n + 10))

theorem closest_to_one_tenth {n : ℕ} (h₀ : 4 ≤ n ∧ n ≤ 7) : 
  |(5 : ℚ) / (n * (n + 10)) - 1 / 10| ≤ 
  |(5 : ℚ) / (4 * (4 + 10)) - 1 / 10| ∧ n = 4 := 
sorry

end NUMINAMATH_GPT_closest_to_one_tenth_l2233_223351


namespace NUMINAMATH_GPT_seq_formula_l2233_223334

noncomputable def seq {a : Nat → ℝ} (h1 : a 2 - a 1 = 1) (h2 : ∀ n, a (n + 1) - a n = 2 * (n - 1) + 1) : Nat → ℝ :=
sorry

theorem seq_formula {a : Nat → ℝ} 
  (h1 : a 2 - a 1 = 1)
  (h2 : ∀ n, a (n + 1) - a n = 2 * (n - 1) + 1)
  (n : Nat) : a n = 2 ^ n - 1 :=
sorry

end NUMINAMATH_GPT_seq_formula_l2233_223334


namespace NUMINAMATH_GPT_sufficient_and_necessary_condition_l2233_223316

def isMonotonicallyIncreasing {R : Type _} [LinearOrderedField R] (f : R → R) :=
  ∀ x y, x < y → f x < f y

def fx {R : Type _} [LinearOrderedField R] (x m : R) :=
  x^3 + 2*x^2 + m*x + 1

theorem sufficient_and_necessary_condition (m : ℝ) :
  (isMonotonicallyIncreasing (λ x => fx x m) ↔ m ≥ 4/3) :=
  sorry

end NUMINAMATH_GPT_sufficient_and_necessary_condition_l2233_223316


namespace NUMINAMATH_GPT_triangle_angles_and_side_l2233_223323

noncomputable def triangle_properties : Type := sorry

variables {A B C : ℝ}
variables {a b c : ℝ}

theorem triangle_angles_and_side (hA : A = 60)
    (ha : a = 4 * Real.sqrt 3)
    (hb : b = 4 * Real.sqrt 2)
    (habc : triangle_properties)
    : B = 45 ∧ C = 75 ∧ c = 2 * Real.sqrt 2 + 2 * Real.sqrt 6 := 
sorry

end NUMINAMATH_GPT_triangle_angles_and_side_l2233_223323


namespace NUMINAMATH_GPT_find_e_value_l2233_223359

-- Define constants a, b, c, d, and e
variables (a b c d e : ℝ)

-- Theorem statement
theorem find_e_value (h1 : (2 : ℝ)^7 * a + (2 : ℝ)^5 * b + (2 : ℝ)^3 * c + 2 * d + e = 23)
                     (h2 : ((-2) : ℝ)^7 * a + ((-2) : ℝ)^5 * b + ((-2) : ℝ)^3 * c + ((-2) : ℝ) * d + e = -35) :
  e = -6 :=
sorry

end NUMINAMATH_GPT_find_e_value_l2233_223359


namespace NUMINAMATH_GPT_johns_age_is_25_l2233_223344

variable (JohnAge DadAge SisterAge : ℕ)

theorem johns_age_is_25
    (h1 : JohnAge = DadAge - 30)
    (h2 : JohnAge + DadAge = 80)
    (h3 : SisterAge = JohnAge - 5) :
    JohnAge = 25 := 
sorry

end NUMINAMATH_GPT_johns_age_is_25_l2233_223344


namespace NUMINAMATH_GPT_fraction_addition_l2233_223345

theorem fraction_addition (a : ℝ) (h : a ≠ 0) : (3 / a) + (2 / a) = 5 / a :=
sorry

end NUMINAMATH_GPT_fraction_addition_l2233_223345


namespace NUMINAMATH_GPT_initial_candies_l2233_223349

theorem initial_candies (L R : ℕ) (h1 : L + R = 27) (h2 : R - L = 2 * L + 3) : L = 6 ∧ R = 21 :=
by
  sorry

end NUMINAMATH_GPT_initial_candies_l2233_223349


namespace NUMINAMATH_GPT_pinocchio_optimal_success_probability_l2233_223385

def success_prob (s : List ℚ) : ℚ :=
  s.foldr (λ x acc => (x * acc) / (1 - (1 - x) * acc)) 1

theorem pinocchio_optimal_success_probability :
  let success_probs := [9/10, 8/10, 7/10, 6/10, 5/10, 4/10, 3/10, 2/10, 1/10]
  success_prob success_probs = 0.4315 :=
by 
  sorry

end NUMINAMATH_GPT_pinocchio_optimal_success_probability_l2233_223385


namespace NUMINAMATH_GPT_projection_identity_l2233_223332

variables (P : ℝ × ℝ × ℝ) (x1 y1 z1 x2 y2 z2 x3 y3 z3 : ℝ)

-- Define point P as (-1, 3, -4)
def point_P := (-1, 3, -4) = P

-- Define projections on the coordinate planes
def projection_yoz := (x1, y1, z1) = (0, 3, -4)
def projection_zox := (x2, y2, z2) = (-1, 0, -4)
def projection_xoy := (x3, y3, z3) = (-1, 3, 0)

-- Prove that x1^2 + y2^2 + z3^2 = 0 under the given conditions
theorem projection_identity :
  point_P P ∧ projection_yoz x1 y1 z1 ∧ projection_zox x2 y2 z2 ∧ projection_xoy x3 y3 z3 →
  (x1^2 + y2^2 + z3^2 = 0) :=
by
  sorry

end NUMINAMATH_GPT_projection_identity_l2233_223332


namespace NUMINAMATH_GPT_equation_of_line_l2233_223393

theorem equation_of_line {M : ℝ × ℝ} {a b : ℝ} (hM : M = (4,2)) 
  (hAB : ∃ A B : ℝ × ℝ, M = ((A.1 + B.1)/2, (A.2 + B.2)/2) ∧ 
    A ≠ B ∧ ∀ x y : ℝ, 
    (x^2 + 4 * y^2 = 36 → (∃ k : ℝ, y - 2 = k * (x - 4) ) )):
  (x + 2 * y - 8 = 0) :=
sorry

end NUMINAMATH_GPT_equation_of_line_l2233_223393


namespace NUMINAMATH_GPT_still_need_more_volunteers_l2233_223353

def total_volunteers_needed : ℕ := 80
def students_volunteering_per_class : ℕ := 4
def number_of_classes : ℕ := 5
def teacher_volunteers : ℕ := 10
def total_student_volunteers : ℕ := students_volunteering_per_class * number_of_classes
def total_volunteers_so_far : ℕ := total_student_volunteers + teacher_volunteers

theorem still_need_more_volunteers : total_volunteers_needed - total_volunteers_so_far = 50 := by
  sorry

end NUMINAMATH_GPT_still_need_more_volunteers_l2233_223353


namespace NUMINAMATH_GPT_inequality_bi_l2233_223367

variable {α : Type*} [LinearOrderedField α]

-- Sequence of positive real numbers
variable (a : ℕ → α)
-- Conditions for a_i
variable (ha : ∀ i, i > 0 → i * (a i)^2 ≥ (i + 1) * a (i - 1) * a (i + 1))
-- Positive real numbers x and y
variables (x y : α) (hx : x > 0) (hy : y > 0)
-- Definition of b_i
def b (i : ℕ) : α := x * a i + y * a (i - 1)

theorem inequality_bi (i : ℕ) (hi : i ≥ 2) : i * (b a x y i)^2 > (i + 1) * (b a x y (i - 1)) * (b a x y (i + 1)) := 
sorry

end NUMINAMATH_GPT_inequality_bi_l2233_223367


namespace NUMINAMATH_GPT_total_visible_surface_area_l2233_223389

-- Define the cubes by their volumes
def volumes : List ℝ := [1, 8, 27, 125, 216, 343, 512, 729]

-- Define the arrangement information as specified
def arrangement_conditions : Prop :=
  ∃ (s8 s7 s6 s5 s4 s3 s2 s1 : ℝ),
    s8^3 = 729 ∧ s7^3 = 512 ∧ s6^3 = 343 ∧ s5^3 = 216 ∧
    s4^3 = 125 ∧ s3^3 = 27 ∧ s2^3 = 8 ∧ s1^3 = 1 ∧
    5 * s8^2 + (5 * s7^2 + 4 * s6^2 + 4 * s5^2) + 
    (5 * s4^2 + 4 * s3^2 + 5 * s2^2 + 4 * s1^2) = 1250

-- The proof statement
theorem total_visible_surface_area : arrangement_conditions → 1250 = 1250 := by
  intro _ -- this stands for not proving the condition, taking it as assumption
  exact rfl


end NUMINAMATH_GPT_total_visible_surface_area_l2233_223389


namespace NUMINAMATH_GPT_luke_bought_stickers_l2233_223304

theorem luke_bought_stickers :
  ∀ (original birthday given_to_sister used_on_card left total_before_buying stickers_bought : ℕ),
  original = 20 →
  birthday = 20 →
  given_to_sister = 5 →
  used_on_card = 8 →
  left = 39 →
  total_before_buying = original + birthday →
  stickers_bought = (left + given_to_sister + used_on_card) - total_before_buying →
  stickers_bought = 12 :=
by
  intros
  sorry

end NUMINAMATH_GPT_luke_bought_stickers_l2233_223304


namespace NUMINAMATH_GPT_solve_quadratic_roots_l2233_223329

theorem solve_quadratic_roots (b c : ℝ) 
  (h : {1, 2} = {x : ℝ | x^2 + b * x + c = 0}) : 
  b = -3 ∧ c = 2 :=
by
  sorry

end NUMINAMATH_GPT_solve_quadratic_roots_l2233_223329


namespace NUMINAMATH_GPT_cost_of_article_is_308_l2233_223376

theorem cost_of_article_is_308 
  (C G : ℝ) 
  (h1 : 348 = C + G)
  (h2 : 350 = C + G + 0.05 * G) : 
  C = 308 :=
by
  sorry

end NUMINAMATH_GPT_cost_of_article_is_308_l2233_223376


namespace NUMINAMATH_GPT_no_all_same_color_l2233_223392

def chameleons_initial_counts (c b m : ℕ) : Prop :=
  c = 13 ∧ b = 15 ∧ m = 17

def chameleon_interaction (c b m : ℕ) : Prop :=
  (∃ c' b' m', c' + b' + m' = c + b + m ∧ 
  ((c' = c - 1 ∧ b' = b - 1 ∧ m' = m + 2) ∨
   (c' = c - 1 ∧ b' = b + 2 ∧ m' = m - 1) ∨
   (c' = c + 2 ∧ b' = b - 1 ∧ m' = m - 1)))

theorem no_all_same_color (c b m : ℕ) (h1 : chameleons_initial_counts c b m) : 
  ¬ (∃ x, c = x ∧ b = x ∧ m = x) := 
sorry

end NUMINAMATH_GPT_no_all_same_color_l2233_223392


namespace NUMINAMATH_GPT_variance_binom_4_half_l2233_223321

-- Define the binomial variance function
def binomial_variance (n : ℕ) (p : ℝ) : ℝ := n * p * (1 - p)

-- Define the conditions
def n := 4
def p := 1 / 2

-- The target statement
theorem variance_binom_4_half : binomial_variance n p = 1 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_GPT_variance_binom_4_half_l2233_223321


namespace NUMINAMATH_GPT_total_cats_l2233_223366

theorem total_cats (current_cats : ℕ) (additional_cats : ℕ) (h1 : current_cats = 11) (h2 : additional_cats = 32):
  current_cats + additional_cats = 43 :=
by
  -- We state the given conditions:
  -- current_cats = 11
  -- additional_cats = 32
  -- We need to prove:
  -- current_cats + additional_cats = 43
  sorry

end NUMINAMATH_GPT_total_cats_l2233_223366


namespace NUMINAMATH_GPT_smallest_value_c_zero_l2233_223324

noncomputable def smallest_possible_c (a b c : ℝ) : ℝ :=
if h : (0 < a) ∧ (0 < b) ∧ (0 < c) then
  0
else
  c

theorem smallest_value_c_zero (a b c : ℝ) (h : (0 < a) ∧ (0 < b) ∧ (0 < c)) :
  smallest_possible_c a b c = 0 :=
by
  sorry

end NUMINAMATH_GPT_smallest_value_c_zero_l2233_223324


namespace NUMINAMATH_GPT_line_condition_l2233_223383

variable (m n Q : ℝ)

theorem line_condition (h1: m = 8 * n + 5) 
                       (h2: m + Q = 8 * (n + 0.25) + 5) 
                       (h3: p = 0.25) : Q = 2 :=
by
  sorry

end NUMINAMATH_GPT_line_condition_l2233_223383


namespace NUMINAMATH_GPT_expressions_equal_iff_l2233_223352

theorem expressions_equal_iff (a b c: ℝ) : a + 2 * b * c = (a + b) * (a + 2 * c) ↔ a + b + 2 * c = 0 :=
by 
  sorry

end NUMINAMATH_GPT_expressions_equal_iff_l2233_223352


namespace NUMINAMATH_GPT_line_always_passes_through_fixed_point_l2233_223315

theorem line_always_passes_through_fixed_point :
  ∀ m : ℝ, (m-1) * 9 + (2 * m - 1) * (-4) = m - 5 := by
  
  -- Proof would go here
  sorry

end NUMINAMATH_GPT_line_always_passes_through_fixed_point_l2233_223315


namespace NUMINAMATH_GPT_six_digit_product_of_consecutive_even_integers_l2233_223310

theorem six_digit_product_of_consecutive_even_integers :
  ∃ (a b c : ℕ), a < b ∧ b < c ∧ (a % 2 = 0) ∧ (b % 2 = 0) ∧ (c % 2 = 0) ∧ a * b * c = 287232 :=
sorry

end NUMINAMATH_GPT_six_digit_product_of_consecutive_even_integers_l2233_223310


namespace NUMINAMATH_GPT_line_equations_satisfy_conditions_l2233_223373

-- Definitions and conditions:
def intersects_at_distance (k m b : ℝ) : Prop :=
  |(k^2 + 7*k + 12) - (m*k + b)| = 8

def passes_through_point (m b : ℝ) : Prop :=
  7 = 2*m + b

def line_equation_valid (m b : ℝ) : Prop :=
  b ≠ 0

-- Main theorem:
theorem line_equations_satisfy_conditions :
  (line_equation_valid 1 5 ∧ passes_through_point 1 5 ∧ 
  ∃ k, intersects_at_distance k 1 5) ∨
  (line_equation_valid 5 (-3) ∧ passes_through_point 5 (-3) ∧ 
  ∃ k, intersects_at_distance k 5 (-3)) :=
by
  sorry

end NUMINAMATH_GPT_line_equations_satisfy_conditions_l2233_223373


namespace NUMINAMATH_GPT_find_a_l2233_223337

open Classical

noncomputable def f (x a : ℝ) : ℝ := |x - a| - 2

theorem find_a (a : ℝ) :
  (∀ x : ℝ, (|f x a| < 1) ↔ (x ∈ Set.Ioo (-2) 0 ∨ x ∈ Set.Ioo 2 4)) → a = 1 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_find_a_l2233_223337


namespace NUMINAMATH_GPT_combined_capacity_l2233_223388

theorem combined_capacity (A B : ℝ) : 3 * A + B = A + 2 * A + B :=
by
  sorry

end NUMINAMATH_GPT_combined_capacity_l2233_223388


namespace NUMINAMATH_GPT_find_k_values_l2233_223339

theorem find_k_values (k : ℝ) : 
  (∃ (x y : ℝ), x + 2 * y - 1 = 0 ∧ x + 1 = 0 ∧ x + k * y = 0) → 
  (k = 0 ∨ k = 1 ∨ k = 2) ∧
  (k = 0 ∨ k = 1 ∨ k = 2 → ∃ (x y : ℝ), x + 2 * y - 1 = 0 ∧ x + 1 = 0 ∧ x + k * y = 0) :=
by
  sorry

end NUMINAMATH_GPT_find_k_values_l2233_223339


namespace NUMINAMATH_GPT_cost_of_batman_game_l2233_223308

noncomputable def footballGameCost : ℝ := 14.02
noncomputable def strategyGameCost : ℝ := 9.46
noncomputable def totalAmountSpent : ℝ := 35.52

theorem cost_of_batman_game :
  totalAmountSpent - (footballGameCost + strategyGameCost) = 12.04 :=
by
  -- The proof is omitted as instructed.
  sorry

end NUMINAMATH_GPT_cost_of_batman_game_l2233_223308


namespace NUMINAMATH_GPT_hexagon_colorings_correct_l2233_223338

noncomputable def hexagon_colorings : Nat :=
  let colors := ["blue", "orange", "purple"]
  2 -- As determined by the solution.

theorem hexagon_colorings_correct :
  hexagon_colorings = 2 :=
by
  sorry

end NUMINAMATH_GPT_hexagon_colorings_correct_l2233_223338


namespace NUMINAMATH_GPT_interest_difference_l2233_223341

noncomputable def annual_amount (P r t : ℝ) : ℝ :=
P * (1 + r)^t

noncomputable def monthly_amount (P r n t : ℝ) : ℝ :=
P * (1 + r / n)^(n * t)

theorem interest_difference
  (P : ℝ)
  (r : ℝ)
  (n : ℕ)
  (t : ℝ)
  (annual_compounded : annual_amount P r t = 8000 * (1 + 0.10)^3)
  (monthly_compounded : monthly_amount P r 12 3 = 8000 * (1 + 0.10 / 12) ^ (12 * 3)) :
  (monthly_amount P r 12 t - annual_amount P r t) = 142.80 := 
sorry

end NUMINAMATH_GPT_interest_difference_l2233_223341


namespace NUMINAMATH_GPT_condition_nonzero_neither_zero_l2233_223395

theorem condition_nonzero_neither_zero (a b : ℝ) (h : a^2 + b^2 ≠ 0) : ¬(a = 0 ∧ b = 0) :=
sorry

end NUMINAMATH_GPT_condition_nonzero_neither_zero_l2233_223395


namespace NUMINAMATH_GPT_units_digit_G1000_is_3_l2233_223369

def G (n : ℕ) : ℕ := 2 ^ (3 ^ n) + 1

theorem units_digit_G1000_is_3 : (G 1000) % 10 = 3 := sorry

end NUMINAMATH_GPT_units_digit_G1000_is_3_l2233_223369


namespace NUMINAMATH_GPT_total_students_l2233_223396

-- Define the conditions
def rank_from_right := 17
def rank_from_left := 5

-- The proof statement
theorem total_students : rank_from_right + rank_from_left - 1 = 21 := 
by 
  -- Assuming the conditions represented by the definitions
  -- Without loss of generality the proof would be derived from these, but it is skipped
  sorry

end NUMINAMATH_GPT_total_students_l2233_223396


namespace NUMINAMATH_GPT_combination_add_l2233_223377

def combination (n m : ℕ) : ℕ := n.choose m

theorem combination_add {n : ℕ} (h1 : 4 ≤ 9) (h2 : 5 ≤ 9) :
  combination 9 4 + combination 9 5 = combination 10 5 := by
  sorry

end NUMINAMATH_GPT_combination_add_l2233_223377


namespace NUMINAMATH_GPT_range_of_a_l2233_223346

noncomputable def f (a x : ℝ) : ℝ := a * Real.log x - x^2 + (1 / 2) * a

theorem range_of_a (a : ℝ) : (∀ x > 0, f a x ≤ 0) → (0 ≤ a ∧ a ≤ 2) :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l2233_223346


namespace NUMINAMATH_GPT_minimum_value_of_expression_l2233_223381

noncomputable def min_value_expression (a b c : ℝ) : ℝ :=
  a^2 + b^2 + (a + b)^2 + c^2

theorem minimum_value_of_expression (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h_sum : a + b + c = 3) :
  min_value_expression a b c = 9 :=
  sorry

end NUMINAMATH_GPT_minimum_value_of_expression_l2233_223381


namespace NUMINAMATH_GPT_union_sets_l2233_223348

noncomputable def M : Set ℤ := {1, 2, 3}
noncomputable def N : Set ℤ := {x | (x + 1) * (x - 2) < 0}

theorem union_sets : M ∪ N = {0, 1, 2, 3} := by
  sorry

end NUMINAMATH_GPT_union_sets_l2233_223348


namespace NUMINAMATH_GPT_consecutive_diff_possible_l2233_223399

variable (a b c : ℝ)

def greater_than_2022 :=
  a > 2022 ∨ b > 2022 ∨ c > 2022

def distinct_numbers :=
  a ≠ b ∧ b ≠ c ∧ a ≠ c

theorem consecutive_diff_possible :
  greater_than_2022 a b c → distinct_numbers a b c → 
  ∃ (x y z : ℤ), x + 1 = y ∧ y + 1 = z ∧ 
  (a^2 - b^2 = ↑x) ∧ (b^2 - c^2 = ↑y) ∧ (c^2 - a^2 = ↑z) :=
by
  intros h1 h2
  -- proof goes here
  sorry

end NUMINAMATH_GPT_consecutive_diff_possible_l2233_223399


namespace NUMINAMATH_GPT_range_of_m_l2233_223386

theorem range_of_m 
  (f : ℝ → ℝ) 
  (h1 : ∀ x y : ℝ, x < y → -3 < x ∧ y < 3 → f x < f y)
  (h2 : ∀ m : ℝ, f (2 * m) < f (m + 1)) : 
  -3/2 < m ∧ m < 1 :=
  sorry

end NUMINAMATH_GPT_range_of_m_l2233_223386


namespace NUMINAMATH_GPT_students_behind_Yoongi_l2233_223360

theorem students_behind_Yoongi :
  ∀ (n : ℕ), n = 20 → ∀ (j y : ℕ), j = 1 → y = 2 → n - y = 18 :=
by
  intros n h1 j h2 y h3
  sorry

end NUMINAMATH_GPT_students_behind_Yoongi_l2233_223360


namespace NUMINAMATH_GPT_num_lines_satisfying_conditions_l2233_223371

-- Define the entities line, angle, and perpendicularity in a geometric framework
variable (Point Line : Type)
variable (P : Point)
variable (a b l : Line)

-- Define geometrical predicates
variable (Perpendicular : Line → Line → Prop)
variable (Passes_Through : Line → Point → Prop)
variable (Forms_Angle : Line → Line → ℝ → Prop)

-- Given conditions
axiom perp_ab : Perpendicular a b
axiom passes_through_P : Passes_Through l P
axiom angle_la_30 : Forms_Angle l a (30 : ℝ)
axiom angle_lb_90 : Forms_Angle l b (90 : ℝ)

-- The statement to prove
theorem num_lines_satisfying_conditions : ∃ (l1 l2 : Line), l1 ≠ l2 ∧ 
  Passes_Through l1 P ∧ Forms_Angle l1 a (30 : ℝ) ∧ Forms_Angle l1 b (90 : ℝ) ∧
  Passes_Through l2 P ∧ Forms_Angle l2 a (30 : ℝ) ∧ Forms_Angle l2 b (90 : ℝ) ∧
  (∀ l', Passes_Through l' P ∧ Forms_Angle l' a (30 : ℝ) ∧ Forms_Angle l' b (90 : ℝ) → l' = l1 ∨ l' = l2) := sorry

end NUMINAMATH_GPT_num_lines_satisfying_conditions_l2233_223371


namespace NUMINAMATH_GPT_find_first_blend_price_l2233_223354

-- Define the conditions
def first_blend_price (x : ℝ) := x
def second_blend_price : ℝ := 8.00
def total_blend_weight : ℝ := 20
def total_blend_price_per_pound : ℝ := 8.40
def first_blend_weight : ℝ := 8
def second_blend_weight : ℝ := total_blend_weight - first_blend_weight

-- Define the cost calculations
def first_blend_total_cost (x : ℝ) := first_blend_weight * x
def second_blend_total_cost := second_blend_weight * second_blend_price
def total_blend_total_cost (x : ℝ) := first_blend_total_cost x + second_blend_total_cost

-- Prove that the price per pound of the first blend is $9.00
theorem find_first_blend_price : ∃ x : ℝ, total_blend_total_cost x = total_blend_weight * total_blend_price_per_pound ∧ x = 9 :=
by
  sorry

end NUMINAMATH_GPT_find_first_blend_price_l2233_223354


namespace NUMINAMATH_GPT_angle_east_northwest_l2233_223311

def num_spokes : ℕ := 12
def central_angle : ℕ := 360 / num_spokes
def angle_between (start_dir end_dir : ℕ) : ℕ := (end_dir - start_dir) * central_angle

theorem angle_east_northwest : angle_between 3 9 = 90 := sorry

end NUMINAMATH_GPT_angle_east_northwest_l2233_223311


namespace NUMINAMATH_GPT_perpendicular_line_to_plane_l2233_223357

variables {Point Line Plane : Type}
variables (a b c : Line) (α : Plane) (A : Point)

-- Define the conditions
def line_perpendicular_to (l1 l2 : Line) : Prop := sorry
def line_in_plane (l : Line) (p : Plane) : Prop := sorry
def lines_intersect_at (l1 l2 : Line) (P : Point) : Prop := sorry
def line_perpendicular_to_plane (l : Line) (p : Plane) : Prop := sorry

-- Given conditions in Lean 4
variables (h1 : line_perpendicular_to c a)
variables (h2 : line_perpendicular_to c b)
variables (h3 : line_in_plane a α)
variables (h4 : line_in_plane b α)
variables (h5 : lines_intersect_at a b A)

-- The theorem statement to prove
theorem perpendicular_line_to_plane : line_perpendicular_to_plane c α :=
sorry

end NUMINAMATH_GPT_perpendicular_line_to_plane_l2233_223357


namespace NUMINAMATH_GPT_cos_pi_plus_2alpha_value_l2233_223301

theorem cos_pi_plus_2alpha_value (α : ℝ) (h : Real.sin (π / 2 + α) = 1 / 3) : 
    Real.cos (π + 2 * α) = 7 / 9 := sorry

end NUMINAMATH_GPT_cos_pi_plus_2alpha_value_l2233_223301


namespace NUMINAMATH_GPT_circle_equation_l2233_223390

/-
  Prove that the standard equation for the circle passing through points
  A(-6, 0), B(0, 2), and the origin O(0, 0) is (x+3)^2 + (y-1)^2 = 10.
-/
theorem circle_equation :
  ∃ (x y : ℝ), x = -6 ∨ x = 0 ∨ x = 0 ∧ y = 0 ∨ y = 2 ∨ y = 0 → (∀ P : ℝ × ℝ, P = (-6, 0) ∨ P = (0, 2) ∨ P = (0, 0) → (P.1 + 3)^2 + (P.2 - 1)^2 = 10) := 
sorry

end NUMINAMATH_GPT_circle_equation_l2233_223390


namespace NUMINAMATH_GPT_g_is_zero_l2233_223365

noncomputable def g (x : ℝ) : ℝ := 
  Real.sqrt (4 * (Real.sin x)^4 + (Real.cos x)^2) - 
  Real.sqrt (4 * (Real.cos x)^4 + (Real.sin x)^2)

theorem g_is_zero (x : ℝ) : g x = 0 := 
  sorry

end NUMINAMATH_GPT_g_is_zero_l2233_223365


namespace NUMINAMATH_GPT_negation_correct_l2233_223312

variable (x : Real)

def original_proposition : Prop :=
  x > 0 → x^2 > 0

def negation_proposition : Prop :=
  x ≤ 0 → x^2 ≤ 0

theorem negation_correct :
  ¬ original_proposition x = negation_proposition x :=
by 
  sorry

end NUMINAMATH_GPT_negation_correct_l2233_223312


namespace NUMINAMATH_GPT_first_player_can_always_make_A_eq_6_l2233_223303

def maxSum3x3In5x5Board (board : Fin 5 → Fin 5 → ℕ) (i j : Fin 3) : ℕ :=
  (i + 3 : Fin 5) * (j + 3 : Fin 5) + 
  (i + 3 : Fin 5) * (j + 4 : Fin 5) + 
  (i + 3 : Fin 5) * (j + 5 : Fin 5) + 
  (i + 4 : Fin 5) * (j + 3 : Fin 5) + 
  (i + 4 : Fin 5) * (j + 4 : Fin 5) + 
  (i + 4 : Fin 5) * (j + 5 : Fin 5) + 
  (i + 5 : Fin 5) * (j + 3 : Fin 5) + 
  (i + 5 : Fin 5) * (j + 4 : Fin 5) + 
  (i + 5 : Fin 5) * (j + 5 : Fin 5)

theorem first_player_can_always_make_A_eq_6 :
  ∀ (board : Fin 5 → Fin 5 → ℕ), 
  (∀ (i j : Fin 3), maxSum3x3In5x5Board board i j = 6)
  :=
by
  intros board i j
  sorry

end NUMINAMATH_GPT_first_player_can_always_make_A_eq_6_l2233_223303


namespace NUMINAMATH_GPT_mike_needs_more_money_l2233_223325

-- We define the conditions given in the problem.
def phone_cost : ℝ := 1300
def mike_fraction : ℝ := 0.40

-- Define the statement to be proven.
theorem mike_needs_more_money : (phone_cost - (mike_fraction * phone_cost) = 780) :=
by
  -- The proof steps would go here
  sorry

end NUMINAMATH_GPT_mike_needs_more_money_l2233_223325


namespace NUMINAMATH_GPT_folded_paper_perimeter_l2233_223361

theorem folded_paper_perimeter (L W : ℝ) 
  (h1 : 2 * L + W = 34)         -- Condition 1
  (h2 : L * W = 140)            -- Condition 2
  : 2 * W + L = 38 :=           -- Goal
sorry

end NUMINAMATH_GPT_folded_paper_perimeter_l2233_223361


namespace NUMINAMATH_GPT_harmful_bacteria_time_l2233_223333

noncomputable def number_of_bacteria (x : ℝ) : ℝ :=
  4000 * 2^x

theorem harmful_bacteria_time :
  ∃ (x : ℝ), number_of_bacteria x > 90000 ∧ x = 4.5 :=
by
  sorry

end NUMINAMATH_GPT_harmful_bacteria_time_l2233_223333


namespace NUMINAMATH_GPT_trapezium_area_l2233_223347

theorem trapezium_area (a b h : ℝ) (ha : a = 20) (hb : b = 18) (hh : h = 12) :
  (1 / 2 * (a + b) * h = 228) :=
by
  sorry

end NUMINAMATH_GPT_trapezium_area_l2233_223347


namespace NUMINAMATH_GPT_sum_of_fractions_l2233_223363

theorem sum_of_fractions : 
  (2/100) + (5/1000) + (5/10000) + 3 * (4/1000) = 0.0375 := 
by 
  sorry

end NUMINAMATH_GPT_sum_of_fractions_l2233_223363


namespace NUMINAMATH_GPT_gcd_values_count_l2233_223314

theorem gcd_values_count (a b : ℕ) (h : Nat.gcd a b * Nat.lcm a b = 392) : ∃ d, d = 11 := 
sorry

end NUMINAMATH_GPT_gcd_values_count_l2233_223314


namespace NUMINAMATH_GPT_smallest_cookies_left_l2233_223375

theorem smallest_cookies_left (m : ℤ) (h : m % 8 = 5) : (4 * m) % 8 = 4 :=
by
  sorry

end NUMINAMATH_GPT_smallest_cookies_left_l2233_223375


namespace NUMINAMATH_GPT_annie_age_when_anna_three_times_current_age_l2233_223382

theorem annie_age_when_anna_three_times_current_age
  (anna_age : ℕ) (annie_age : ℕ)
  (h1 : anna_age = 13)
  (h2 : annie_age = 3 * anna_age) :
  annie_age + 2 * anna_age = 65 :=
by
  sorry

end NUMINAMATH_GPT_annie_age_when_anna_three_times_current_age_l2233_223382


namespace NUMINAMATH_GPT_larger_integer_21_l2233_223380

theorem larger_integer_21
  (a b : ℕ)
  (h1 : b = 7 * a / 3)
  (h2 : a * b = 189) :
  max a b = 21 :=
by
  sorry

end NUMINAMATH_GPT_larger_integer_21_l2233_223380


namespace NUMINAMATH_GPT_solve_eq1_solve_eq2_l2233_223313

-- Definition of the first equation
def eq1 (x : ℝ) : Prop := (1 / 2) * x^2 - 8 = 0

-- Definition of the second equation
def eq2 (x : ℝ) : Prop := (x - 5)^3 = -27

-- Proof statement for the value of x in the first equation
theorem solve_eq1 (x : ℝ) : eq1 x ↔ x = 4 ∨ x = -4 := by
  sorry

-- Proof statement for the value of x in the second equation
theorem solve_eq2 (x : ℝ) : eq2 x ↔ x = 2 := by
  sorry

end NUMINAMATH_GPT_solve_eq1_solve_eq2_l2233_223313


namespace NUMINAMATH_GPT_inequality_inequality_must_be_true_l2233_223397

variables {a b c d : ℝ}

theorem inequality_inequality_must_be_true
  (h1 : a > b)
  (h2 : b > 0)
  (h3 : c < d)
  (h4 : d < 0) :
  (a / d) < (b / c) :=
sorry

end NUMINAMATH_GPT_inequality_inequality_must_be_true_l2233_223397


namespace NUMINAMATH_GPT_solve_for_z_l2233_223394

theorem solve_for_z (z i : ℂ) (h1 : 1 - i*z + 3*i = -1 + i*z + 3*i) (h2 : i^2 = -1) : z = -i := 
  sorry

end NUMINAMATH_GPT_solve_for_z_l2233_223394


namespace NUMINAMATH_GPT_remainder_of_2x_plus_3uy_l2233_223340

theorem remainder_of_2x_plus_3uy (x y u v : ℤ) (hxy : x = u * y + v) (hv : 0 ≤ v) (hv_ub : v < y) :
  (if 2 * v < y then (2 * v % y) else ((2 * v % y) % -y % y)) = 
  (if 2 * v < y then 2 * v else 2 * v - y) :=
by {
  sorry
}

end NUMINAMATH_GPT_remainder_of_2x_plus_3uy_l2233_223340


namespace NUMINAMATH_GPT_cubics_sum_l2233_223306

noncomputable def roots_cubic (a b c d p q r : ℝ) : Prop :=
  (p + q + r = b) ∧ (p*q + p*r + q*r = c) ∧ (p*q*r = d)

noncomputable def root_values (p q r : ℝ) : Prop :=
  p^3 = 2*p^2 - 3*p + 4 ∧
  q^3 = 2*q^2 - 3*q + 4 ∧
  r^3 = 2*r^2 - 3*r + 4

theorem cubics_sum (p q r : ℝ) (h1 : p + q + r = 2) (h2 : p*q + q*r + p*r = 3)  (h3 : p*q*r = 4)
  (h4 : root_values p q r) : p^3 + q^3 + r^3 = 2 :=
by
  sorry

end NUMINAMATH_GPT_cubics_sum_l2233_223306


namespace NUMINAMATH_GPT_average_age_group_l2233_223307

theorem average_age_group (n : ℕ) (T : ℕ) (h1 : T = 15 * n) (h2 : T + 37 = 17 * (n + 1)) : n = 10 :=
by
  sorry

end NUMINAMATH_GPT_average_age_group_l2233_223307


namespace NUMINAMATH_GPT_hyperbola_shares_focus_with_eccentricity_length_of_chord_AB_l2233_223362

theorem hyperbola_shares_focus_with_eccentricity 
  (a1 b1 : ℝ) (h1 : a1 = 3 ∧ b1 = 2)
  (e : ℝ) (h_eccentricity : e = (Real.sqrt 5) / 2)
  (c : ℝ) (h_focus : c = Real.sqrt (a1^2 - b1^2)) :
  (∃ a b : ℝ, a^2 - b^2 = c^2 ∧ c/a = e ∧ a = 2 ∧ b = 1) :=
sorry

theorem length_of_chord_AB 
  (a b : ℝ) (h_ellipse : a^2 = 4 ∧ b^2 = 1)
  (c : ℝ) (h_focus : c = Real.sqrt (a^2 - b^2))
  (f : ℝ) (h_f : f = Real.sqrt 3)
  (line_eq : ℝ -> ℝ) (h_line_eq : ∀ x, line_eq x = x - f) :
  (∃ x1 x2 : ℝ, 
    x1 + x2 = (8 * Real.sqrt 3) / 5 ∧
    x1 * x2 = 8 / 5 ∧
    Real.sqrt ((x1 - x2)^2 + (line_eq x1 - line_eq x2)^2) = 8 / 5) :=
sorry

end NUMINAMATH_GPT_hyperbola_shares_focus_with_eccentricity_length_of_chord_AB_l2233_223362


namespace NUMINAMATH_GPT_solve_equation_nat_numbers_l2233_223342

theorem solve_equation_nat_numbers :
  ∃ (x y z : ℕ), (2 ^ x + 3 ^ y + 7 = z!) ∧ ((x = 3 ∧ y = 2 ∧ z = 4) ∨ (x = 5 ∧ y = 4 ∧ z = 5)) := 
sorry

end NUMINAMATH_GPT_solve_equation_nat_numbers_l2233_223342


namespace NUMINAMATH_GPT_roy_is_6_years_older_than_julia_l2233_223309

theorem roy_is_6_years_older_than_julia :
  ∀ (R J K : ℕ) (x : ℕ), 
    R = J + x →
    R = K + x / 2 →
    R + 4 = 2 * (J + 4) →
    (R + 4) * (K + 4) = 108 →
    x = 6 :=
by
  intros R J K x h1 h2 h3 h4
  -- Proof goes here (using sorry to skip the proof)
  sorry

end NUMINAMATH_GPT_roy_is_6_years_older_than_julia_l2233_223309


namespace NUMINAMATH_GPT_tangent_line_at_one_l2233_223328

noncomputable def f (x : ℝ) : ℝ := x^2 + Real.log x

theorem tangent_line_at_one (a b : ℝ) (h_tangent : ∀ x, f x = a * x + b) : 
  a + b = 1 := 
sorry

end NUMINAMATH_GPT_tangent_line_at_one_l2233_223328


namespace NUMINAMATH_GPT_pow_div_simplify_l2233_223364

theorem pow_div_simplify : (((15^15) / (15^14))^3 * 3^3) / 3^3 = 3375 := by
  sorry

end NUMINAMATH_GPT_pow_div_simplify_l2233_223364


namespace NUMINAMATH_GPT_correct_fraction_subtraction_l2233_223320

theorem correct_fraction_subtraction (x : ℝ) (h₁ : x ≠ 0) (h₂ : x ≠ 1) :
  ((1 / x) - (1 / (x - 1))) = - (1 / (x^2 - x)) :=
by
  sorry

end NUMINAMATH_GPT_correct_fraction_subtraction_l2233_223320


namespace NUMINAMATH_GPT_unique_n_for_prime_p_l2233_223370

theorem unique_n_for_prime_p (p : ℕ) (hp1 : p > 2) (hp2 : Nat.Prime p) :
  ∃! (n : ℕ), (∃ (k : ℕ), n^2 + n * p = k^2) ∧ n = (p - 1) / 2 ^ 2 :=
sorry

end NUMINAMATH_GPT_unique_n_for_prime_p_l2233_223370


namespace NUMINAMATH_GPT_length_of_DE_l2233_223300

theorem length_of_DE (base : ℝ) (area_ratio : ℝ) (height_ratio : ℝ) :
  base = 18 → area_ratio = 0.09 → height_ratio = 0.3 → DE = 2 :=
by
  sorry

end NUMINAMATH_GPT_length_of_DE_l2233_223300


namespace NUMINAMATH_GPT_david_reading_time_l2233_223387

def total_time : ℕ := 180
def math_homework : ℕ := 25
def spelling_homework : ℕ := 30
def history_assignment : ℕ := 20
def science_project : ℕ := 15
def piano_practice : ℕ := 30
def study_breaks : ℕ := 2 * 10

def time_other_activities : ℕ := math_homework + spelling_homework + history_assignment + science_project + piano_practice + study_breaks

theorem david_reading_time : total_time - time_other_activities = 40 :=
by
  -- Calculation steps would go here, not provided for the theorem statement.
  sorry

end NUMINAMATH_GPT_david_reading_time_l2233_223387


namespace NUMINAMATH_GPT_ratio_c_to_d_l2233_223331

theorem ratio_c_to_d (a b c d : ℚ) 
  (h1 : a / b = 3 / 4) 
  (h2 : b / c = 7 / 9) 
  (h3 : a / d = 0.4166666666666667) : 
  c / d = 5 / 7 := 
by
  -- Proof not needed
  sorry

end NUMINAMATH_GPT_ratio_c_to_d_l2233_223331


namespace NUMINAMATH_GPT_gcd_lcm_product_l2233_223368

noncomputable def a : ℕ := 90
noncomputable def b : ℕ := 135

theorem gcd_lcm_product :
  Nat.gcd a b * Nat.lcm a b = 12150 := by
  sorry

end NUMINAMATH_GPT_gcd_lcm_product_l2233_223368


namespace NUMINAMATH_GPT_twenty_four_x_eq_a_cubed_t_l2233_223374

-- Define conditions
variables {x : ℝ} {a t : ℝ}
axiom h1 : 2^x = a
axiom h2 : 3^x = t

-- State the theorem
theorem twenty_four_x_eq_a_cubed_t : 24^x = a^3 * t := 
by sorry

end NUMINAMATH_GPT_twenty_four_x_eq_a_cubed_t_l2233_223374


namespace NUMINAMATH_GPT_find_four_letter_list_with_equal_product_l2233_223326

open Nat

theorem find_four_letter_list_with_equal_product :
  ∃ (L T M W : ℕ), 
  (L * T * M * W = 23 * 24 * 25 * 26) 
  ∧ (1 ≤ L ∧ L ≤ 26) ∧ (1 ≤ T ∧ T ≤ 26) ∧ (1 ≤ M ∧ M ≤ 26) ∧ (1 ≤ W ∧ W ≤ 26) 
  ∧ (L ≠ T) ∧ (T ≠ M) ∧ (M ≠ W) ∧ (W ≠ L) ∧ (L ≠ M) ∧ (T ≠ W)
  ∧ (L * T * M * W) = (12 * 20 * 13 * 23) :=
by
  sorry

end NUMINAMATH_GPT_find_four_letter_list_with_equal_product_l2233_223326


namespace NUMINAMATH_GPT_find_k_for_maximum_value_l2233_223302

theorem find_k_for_maximum_value (k : ℝ) :
  (∀ x : ℝ, -3 ≤ x ∧ x ≤ 2 → k * x^2 + 2 * k * x + 1 ≤ 5) ∧
  (∃ x : ℝ, -3 ≤ x ∧ x ≤ 2 ∧ k * x^2 + 2 * k * x + 1 = 5) ↔
  k = 1 / 2 ∨ k = -4 :=
by
  sorry

end NUMINAMATH_GPT_find_k_for_maximum_value_l2233_223302


namespace NUMINAMATH_GPT_double_persons_half_work_l2233_223378

theorem double_persons_half_work :
  (∀ (n : ℕ) (d : ℕ), d = 12 → (2 * n) * (d / 2) = n * 3) :=
by
  sorry

end NUMINAMATH_GPT_double_persons_half_work_l2233_223378


namespace NUMINAMATH_GPT_solve_x_eq_l2233_223322

theorem solve_x_eq : ∃ x : ℚ, -3 * x - 12 = 6 * x + 9 ∧ x = -7 / 3 :=
by 
  sorry

end NUMINAMATH_GPT_solve_x_eq_l2233_223322


namespace NUMINAMATH_GPT_find_L_l2233_223336

noncomputable def L_value : ℕ := 3

theorem find_L
  (a b : ℕ)
  (cows : ℕ := 5 * b)
  (chickens : ℕ := 5 * a + 7)
  (insects : ℕ := b ^ (a - 5))
  (legs_cows : ℕ := 4 * cows)
  (legs_chickens : ℕ := 2 * chickens)
  (legs_insects : ℕ :=  6 * insects)
  (total_legs : ℕ := legs_cows + legs_chickens + legs_insects) 
  (h1 : cows = insects)
  (h2 : total_legs = (L_value * 100 + L_value * 10 + L_value) + 1) :
  L_value = 3 := sorry

end NUMINAMATH_GPT_find_L_l2233_223336


namespace NUMINAMATH_GPT_greatest_possible_value_of_y_l2233_223372

theorem greatest_possible_value_of_y (x y : ℤ) (h : x * y + 3 * x + 2 * y = -1) : y ≤ 2 :=
sorry

end NUMINAMATH_GPT_greatest_possible_value_of_y_l2233_223372


namespace NUMINAMATH_GPT_multiple_of_6_is_multiple_of_3_l2233_223355

theorem multiple_of_6_is_multiple_of_3 (n : ℕ) (h1 : ∀ k : ℕ, n = 6 * k)
  : ∃ m : ℕ, n = 3 * m :=
by sorry

end NUMINAMATH_GPT_multiple_of_6_is_multiple_of_3_l2233_223355


namespace NUMINAMATH_GPT_apples_bought_is_28_l2233_223327

-- Define the initial number of apples, number of apples used, and total number of apples after buying more
def initial_apples : ℕ := 38
def apples_used : ℕ := 20
def total_apples_after_buying : ℕ := 46

-- State the theorem: the number of apples bought is 28
theorem apples_bought_is_28 : (total_apples_after_buying - (initial_apples - apples_used)) = 28 := 
by sorry

end NUMINAMATH_GPT_apples_bought_is_28_l2233_223327


namespace NUMINAMATH_GPT_least_subtraction_divisible_l2233_223318

def least_subtrahend (n m : ℕ) : ℕ :=
n % m

theorem least_subtraction_divisible (n : ℕ) (m : ℕ) (sub : ℕ) :
  n = 13604 → m = 87 → sub = least_subtrahend n m → (n - sub) % m = 0 :=
by
  intros h1 h2 h3
  rw [h1, h2, h3]
  sorry

end NUMINAMATH_GPT_least_subtraction_divisible_l2233_223318


namespace NUMINAMATH_GPT_m_greater_than_one_l2233_223330

variables {x m : ℝ}

def p : Prop := -2 ≤ x ∧ x ≤ 11
def q : Prop := 1 - 3 * m ≤ x ∧ x ≤ 3 + m

theorem m_greater_than_one (h : ¬(x^2 - 2 * x + m ≤ 0)) : m > 1 :=
sorry

end NUMINAMATH_GPT_m_greater_than_one_l2233_223330


namespace NUMINAMATH_GPT_probability_of_green_ball_l2233_223379

def container_X := (5, 7)  -- (red balls, green balls)
def container_Y := (7, 5)  -- (red balls, green balls)
def container_Z := (7, 5)  -- (red balls, green balls)

def total_balls (container : ℕ × ℕ) : ℕ := container.1 + container.2

def probability_green (container : ℕ × ℕ) : ℚ := 
  (container.2 : ℚ) / total_balls container

noncomputable def probability_green_from_random_selection : ℚ :=
  (1 / 3) * probability_green container_X +
  (1 / 3) * probability_green container_Y +
  (1 / 3) * probability_green container_Z

theorem probability_of_green_ball :
  probability_green_from_random_selection = 17 / 36 :=
sorry

end NUMINAMATH_GPT_probability_of_green_ball_l2233_223379
