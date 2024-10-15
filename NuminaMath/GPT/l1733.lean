import Mathlib

namespace NUMINAMATH_GPT_us_more_than_canada_l1733_173324

/-- Define the total number of supermarkets -/
def total_supermarkets : ℕ := 84

/-- Define the number of supermarkets in the US -/
def us_supermarkets : ℕ := 49

/-- Define the number of supermarkets in Canada -/
def canada_supermarkets : ℕ := total_supermarkets - us_supermarkets

/-- The proof problem: Prove that there are 14 more supermarkets in the US than in Canada -/
theorem us_more_than_canada : us_supermarkets - canada_supermarkets = 14 := by
  sorry

end NUMINAMATH_GPT_us_more_than_canada_l1733_173324


namespace NUMINAMATH_GPT_area_inequality_equality_condition_l1733_173328

variable (a b c d S : ℝ)
variable (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : d > 0) 
variable (s : ℝ) (h5 : s = (a + b + c + d) / 2)
variable (h6 : S = Real.sqrt ((s - a) * (s - b) * (s - c) * (s - d)))

theorem area_inequality (h : S = Real.sqrt ((s - a) * (s - b) * (s - c) * (s - d)) ∧ s = (a + b + c + d) / 2) :
  S ≤ Real.sqrt (a * b * c * d) :=
sorry

theorem equality_condition (h : S = Real.sqrt ((s - a) * (s - b) * (s - c) * (s - d)) ∧ s = (a + b + c + d) / 2) :
  (S = Real.sqrt (a * b * c * d)) ↔ (a = c ∧ b = d ∨ a = d ∧ b = c) :=
sorry

end NUMINAMATH_GPT_area_inequality_equality_condition_l1733_173328


namespace NUMINAMATH_GPT_max_min_value_of_fg_l1733_173323

noncomputable def f (x : ℝ) : ℝ := 4 - x^2
noncomputable def g (x : ℝ) : ℝ := 3 * x
noncomputable def min' (a b : ℝ) : ℝ := if a < b then a else b

theorem max_min_value_of_fg : ∃ x : ℝ, min' (f x) (g x) = 3 :=
by
  sorry

end NUMINAMATH_GPT_max_min_value_of_fg_l1733_173323


namespace NUMINAMATH_GPT_ratio_of_areas_l1733_173370

theorem ratio_of_areas (s L : ℝ) (h1 : (π * L^2) / (π * s^2) = 9 / 4) : L - s = (1/2) * s :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_areas_l1733_173370


namespace NUMINAMATH_GPT_range_of_a_for_three_tangents_curve_through_point_l1733_173382

noncomputable def curve (a : ℝ) (x : ℝ) : ℝ := x^3 + 3 * x^2 + a * x + a - 2

noncomputable def curve_derivative (a : ℝ) (x : ℝ) : ℝ := 3 * x^2 + 6 * x + a

theorem range_of_a_for_three_tangents_curve_through_point :
  ∀ (a : ℝ), (∀ x0 : ℝ, 2 * x0^3 + 3 * x0^2 + 4 - a = 0 → 
    ((2 * -1^3 + 3 * -1^2 + 4 - a > 0) ∧ (2 * 0^3 + 3 * 0^2 + 4 - a < 0))) ↔ (4 < a ∧ a < 5) :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_for_three_tangents_curve_through_point_l1733_173382


namespace NUMINAMATH_GPT_fourth_term_of_geometric_sequence_l1733_173341

theorem fourth_term_of_geometric_sequence 
  (a r : ℕ) 
  (h₁ : a = 3)
  (h₂ : a * r^2 = 75) :
  a * r^3 = 375 := 
by
  sorry

end NUMINAMATH_GPT_fourth_term_of_geometric_sequence_l1733_173341


namespace NUMINAMATH_GPT_max_positive_n_l1733_173308

noncomputable def is_arithmetic_sequence (a : ℕ → ℤ) : Prop :=
∀ n m, a (n + 1) - a n = a (m + 1) - a m

noncomputable def sequence_condition (a : ℕ → ℤ) : Prop :=
a 1010 / a 1009 < -1

noncomputable def sum_of_first_n_terms (a : ℕ → ℤ) (n : ℕ) : ℤ :=
n * (a 1 + a n) / 2

theorem max_positive_n (a : ℕ → ℤ) (h1 : is_arithmetic_sequence a) 
    (h2 : sequence_condition a) : n = 2018 ∧ sum_of_first_n_terms a 2018 > 0 := sorry

end NUMINAMATH_GPT_max_positive_n_l1733_173308


namespace NUMINAMATH_GPT_sum_of_squares_of_roots_l1733_173316

theorem sum_of_squares_of_roots 
  (x1 x2 : ℝ) 
  (h₁ : 5 * x1^2 - 6 * x1 - 4 = 0)
  (h₂ : 5 * x2^2 - 6 * x2 - 4 = 0)
  (h₃ : x1 ≠ x2) :
  x1^2 + x2^2 = 76 / 25 := sorry

end NUMINAMATH_GPT_sum_of_squares_of_roots_l1733_173316


namespace NUMINAMATH_GPT_highest_prob_red_ball_l1733_173319

-- Definitions
def total_red_balls : ℕ := 5
def total_white_balls : ℕ := 12
def total_balls : ℕ := total_red_balls + total_white_balls

-- Condition that neither bag is empty
def neither_bag_empty (r1 w1 r2 w2 : ℕ) : Prop :=
  (r1 + w1 > 0) ∧ (r2 + w2 > 0)

-- Define the probability of drawing a red ball from a bag
def prob_red (r w : ℕ) : ℚ :=
  if (r + w) = 0 then 0 else r / (r + w)

-- Define the overall probability if choosing either bag with equal probability
def overall_prob_red (r1 w1 r2 w2 : ℕ) : ℚ :=
  (prob_red r1 w1 + prob_red r2 w2) / 2

-- Problem statement to be proved
theorem highest_prob_red_ball :
  ∃ (r1 w1 r2 w2 : ℕ),
    neither_bag_empty r1 w1 r2 w2 ∧
    r1 + r2 = total_red_balls ∧
    w1 + w2 = total_white_balls ∧
    (overall_prob_red r1 w1 r2 w2 = 0.625) :=
sorry

end NUMINAMATH_GPT_highest_prob_red_ball_l1733_173319


namespace NUMINAMATH_GPT_hershey_kisses_to_kitkats_ratio_l1733_173361

-- Definitions based on the conditions
def kitkats : ℕ := 5
def nerds : ℕ := 8
def lollipops : ℕ := 11
def baby_ruths : ℕ := 10
def reeses : ℕ := baby_ruths / 2
def candy_total_before : ℕ := kitkats + nerds + lollipops + baby_ruths + reeses
def candy_remaining : ℕ := 49
def lollipops_given : ℕ := 5
def total_candy_before : ℕ := candy_remaining + lollipops_given
def hershey_kisses : ℕ := total_candy_before - candy_total_before

-- Theorem to prove the desired ratio
theorem hershey_kisses_to_kitkats_ratio : hershey_kisses / kitkats = 3 := by
  sorry

end NUMINAMATH_GPT_hershey_kisses_to_kitkats_ratio_l1733_173361


namespace NUMINAMATH_GPT_inradius_of_right_triangle_l1733_173360

-- Define the side lengths of the triangle
def a : ℕ := 9
def b : ℕ := 40
def c : ℕ := 41

-- Define the semiperimeter of the triangle
def s : ℕ := (a + b + c) / 2

-- Define the area of a right triangle
def A : ℕ := (a * b) / 2

-- Define the inradius of the triangle
def inradius : ℕ := A / s

theorem inradius_of_right_triangle : inradius = 4 :=
by
  -- The proof is omitted since only the statement is requested
  sorry

end NUMINAMATH_GPT_inradius_of_right_triangle_l1733_173360


namespace NUMINAMATH_GPT_apple_consumption_l1733_173363

-- Definitions for the portions of the apple above and below water
def portion_above_water := 1 / 5
def portion_below_water := 4 / 5

-- Rates of consumption by fish and bird
def fish_rate := 120  -- grams per minute
def bird_rate := 60  -- grams per minute

-- The question statements with the correct answers
theorem apple_consumption :
  (portion_below_water * (fish_rate / (fish_rate + bird_rate)) = 2 / 3) ∧ 
  (portion_above_water * (bird_rate / (fish_rate + bird_rate)) = 1 / 3) := 
sorry

end NUMINAMATH_GPT_apple_consumption_l1733_173363


namespace NUMINAMATH_GPT_substitution_correct_l1733_173326

theorem substitution_correct (x y : ℝ) (h1 : y = x - 1) (h2 : x - 2 * y = 7) :
  x - 2 * x + 2 = 7 :=
by
  sorry

end NUMINAMATH_GPT_substitution_correct_l1733_173326


namespace NUMINAMATH_GPT_iggy_running_hours_l1733_173315

theorem iggy_running_hours :
  ∀ (monday tuesday wednesday thursday friday pace_in_minutes total_minutes_in_hour : ℕ),
  monday = 3 → tuesday = 4 → wednesday = 6 → thursday = 8 → friday = 3 →
  pace_in_minutes = 10 → total_minutes_in_hour = 60 →
  ((monday + tuesday + wednesday + thursday + friday) * pace_in_minutes) / total_minutes_in_hour = 4 :=
by
  intros monday tuesday wednesday thursday friday pace_in_minutes total_minutes_in_hour
  sorry

end NUMINAMATH_GPT_iggy_running_hours_l1733_173315


namespace NUMINAMATH_GPT_common_ratio_q_l1733_173366

variable {a_n : ℕ → ℝ} {S_n : ℕ → ℝ} {q : ℝ}

axiom a5_condition : a_n 5 = 2 * S_n 4 + 3
axiom a6_condition : a_n 6 = 2 * S_n 5 + 3

theorem common_ratio_q : q = 3 :=
by
  sorry

end NUMINAMATH_GPT_common_ratio_q_l1733_173366


namespace NUMINAMATH_GPT_seated_ways_alice_between_bob_and_carol_l1733_173304

-- Define the necessary entities and conditions for the problem.
def num_people : Nat := 7
def alice := "Alice"
def bob := "Bob"
def carol := "Carol"

-- The main theorem
theorem seated_ways_alice_between_bob_and_carol :
  ∃ (ways : Nat), ways = 48 := by
  sorry

end NUMINAMATH_GPT_seated_ways_alice_between_bob_and_carol_l1733_173304


namespace NUMINAMATH_GPT_number_50_is_sample_size_l1733_173317

def number_of_pairs : ℕ := 50
def is_sample_size (n : ℕ) : Prop := n = number_of_pairs

-- We are to show that 50 represents the sample size
theorem number_50_is_sample_size : is_sample_size 50 :=
sorry

end NUMINAMATH_GPT_number_50_is_sample_size_l1733_173317


namespace NUMINAMATH_GPT_necessary_but_not_sufficient_condition_l1733_173380

theorem necessary_but_not_sufficient_condition (x : ℝ) : (x > 5) → (x > 4) :=
by 
  intro h
  linarith

end NUMINAMATH_GPT_necessary_but_not_sufficient_condition_l1733_173380


namespace NUMINAMATH_GPT_convert_speed_72_kmph_to_mps_l1733_173365

theorem convert_speed_72_kmph_to_mps :
  let kmph := 72
  let factor_km_to_m := 1000
  let factor_hr_to_s := 3600
  (kmph * factor_km_to_m) / factor_hr_to_s = 20 := by
  -- (72 kmph * (1000 meters / 1 kilometer)) / (3600 seconds / 1 hour) = 20 meters per second
  sorry

end NUMINAMATH_GPT_convert_speed_72_kmph_to_mps_l1733_173365


namespace NUMINAMATH_GPT_calculation_result_l1733_173359

theorem calculation_result :
  -Real.sqrt 4 + abs (-Real.sqrt 2 - 1) + (Real.pi - 2013) ^ 0 - (1/5) ^ 0 = Real.sqrt 2 - 1 :=
by
  sorry

end NUMINAMATH_GPT_calculation_result_l1733_173359


namespace NUMINAMATH_GPT_find_largest_number_l1733_173331

noncomputable def largest_number (a b c : ℚ) : ℚ :=
  if a + b + c = 77 ∧ c - b = 9 ∧ b - a = 5 then c else 0

theorem find_largest_number (a b c : ℚ) 
  (h1 : a + b + c = 77) 
  (h2 : c - b = 9) 
  (h3 : b - a = 5) : 
  c = 100 / 3 := 
sorry

end NUMINAMATH_GPT_find_largest_number_l1733_173331


namespace NUMINAMATH_GPT_remaining_food_can_cater_children_l1733_173318

theorem remaining_food_can_cater_children (A C : ℝ) 
  (h_food_adults : 70 * A = 90 * C) 
  (h_35_adults_ate : ∀ n: ℝ, (n = 35) → 35 * A = 35 * (9/7) * C) : 
  70 * A - 35 * A = 45 * C :=
by
  sorry

end NUMINAMATH_GPT_remaining_food_can_cater_children_l1733_173318


namespace NUMINAMATH_GPT_opposite_of_half_l1733_173379

theorem opposite_of_half : - (1 / 2) = -1 / 2 := 
by
  sorry

end NUMINAMATH_GPT_opposite_of_half_l1733_173379


namespace NUMINAMATH_GPT_volume_of_bag_l1733_173303

-- Define the dimensions of the cuboid
def width : ℕ := 9
def length : ℕ := 4
def height : ℕ := 7

-- Define the volume calculation function for a cuboid
def volume (l w h : ℕ) : ℕ :=
  l * w * h

-- Provide the theorem to prove the volume is 252 cubic centimeters
theorem volume_of_bag : volume length width height = 252 := by
  -- Since the proof is not requested, insert sorry to complete the statement.
  sorry

end NUMINAMATH_GPT_volume_of_bag_l1733_173303


namespace NUMINAMATH_GPT_burmese_pythons_required_l1733_173348

theorem burmese_pythons_required (single_python_rate : ℕ) (total_alligators : ℕ) (total_weeks : ℕ) (required_pythons : ℕ) :
  single_python_rate = 1 →
  total_alligators = 15 →
  total_weeks = 3 →
  required_pythons = total_alligators / total_weeks →
  required_pythons = 5 :=
by
  intros h1 h2 h3 h4
  rw [h1, h2, h3] at *
  simp at h4
  sorry

end NUMINAMATH_GPT_burmese_pythons_required_l1733_173348


namespace NUMINAMATH_GPT_find_common_divisor_l1733_173321

open Int

theorem find_common_divisor (n : ℕ) (h1 : 2287 % n = 2028 % n)
  (h2 : 2028 % n = 1806 % n) : n = Int.gcd (Int.gcd 259 222) 481 := by
  sorry -- Proof goes here

end NUMINAMATH_GPT_find_common_divisor_l1733_173321


namespace NUMINAMATH_GPT_no_solution_2023_l1733_173354

theorem no_solution_2023 (a b c : ℕ) (h₁ : a + b + c = 2023) (h₂ : (b + c) ∣ a) (h₃ : (b - c + 1) ∣ (b + c)) : false :=
by
  sorry

end NUMINAMATH_GPT_no_solution_2023_l1733_173354


namespace NUMINAMATH_GPT_find_y_intersection_of_tangents_l1733_173392

-- Define the parabola
def parabola (x : ℝ) : ℝ := x^2 - 2 * x - 3

-- Define the tangent slope at a point on the parabola
def tangent_slope (x : ℝ) : ℝ := 2 * (x - 1)

-- Define the perpendicular condition for tangents at points A and B
def perpendicular_condition (a b : ℝ) : Prop := (a - 1) * (b - 1) = -1 / 4

-- Define the y-coordinate of the intersection point P of the tangents at A and B
def y_coordinate_of_intersection (a b : ℝ) : ℝ := a * b - a - b + 2

-- Theorem to be proved
theorem find_y_intersection_of_tangents (a b : ℝ) 
  (ha : parabola a = a ^ 2 - 2 * a - 3) 
  (hb : parabola b = b ^ 2 - 2 * b - 3) 
  (hp : perpendicular_condition a b) :
  y_coordinate_of_intersection a b = -1 / 4 :=
sorry

end NUMINAMATH_GPT_find_y_intersection_of_tangents_l1733_173392


namespace NUMINAMATH_GPT_minimum_value_of_fraction_plus_variable_l1733_173356

theorem minimum_value_of_fraction_plus_variable (a : ℝ) (h : a > 1) : ∃ m, (∀ b, b > 1 → (4 / (b - 1) + b) ≥ m) ∧ m = 5 :=
by
  use 5
  sorry

end NUMINAMATH_GPT_minimum_value_of_fraction_plus_variable_l1733_173356


namespace NUMINAMATH_GPT_households_with_dvd_player_l1733_173306

noncomputable def numHouseholds : ℕ := 100
noncomputable def numWithCellPhone : ℕ := 90
noncomputable def numWithMP3Player : ℕ := 55
noncomputable def greatestWithAllThree : ℕ := 55 -- maximum x
noncomputable def differenceX_Y : ℕ := 25 -- x - y = 25

def numberOfDVDHouseholds : ℕ := 15

theorem households_with_dvd_player : ∀ (D : ℕ),
  D + 25 - D = 55 - 20 →
  D = numberOfDVDHouseholds :=
by
  intro D h
  sorry

end NUMINAMATH_GPT_households_with_dvd_player_l1733_173306


namespace NUMINAMATH_GPT_total_investment_is_10000_l1733_173398

open Real

-- Definitions of conditions
def interest_rate_8 : Real := 0.08
def interest_rate_9 : Real := 0.09
def combined_interest : Real := 840
def investment_8 : Real := 6000
def total_interest (x : Real) : Real := (interest_rate_8 * investment_8 + interest_rate_9 * x)
def investment_9 : Real := 4000

-- Theorem stating the problem
theorem total_investment_is_10000 :
    (∀ x : Real,
        total_interest x = combined_interest → x = investment_9) →
    investment_8 + investment_9 = 10000 := 
by
    intros
    sorry

end NUMINAMATH_GPT_total_investment_is_10000_l1733_173398


namespace NUMINAMATH_GPT_du_chin_fraction_of_sales_l1733_173385

theorem du_chin_fraction_of_sales :
  let pies := 200
  let price_per_pie := 20
  let remaining_money := 1600
  let total_sales := pies * price_per_pie
  let used_for_ingredients := total_sales - remaining_money
  let fraction_used_for_ingredients := used_for_ingredients / total_sales
  fraction_used_for_ingredients = (3 / 5) := by
    sorry

end NUMINAMATH_GPT_du_chin_fraction_of_sales_l1733_173385


namespace NUMINAMATH_GPT_log_minus_one_has_one_zero_l1733_173390

theorem log_minus_one_has_one_zero : ∃! x : ℝ, x > 0 ∧ (Real.log x - 1 = 0) :=
sorry

end NUMINAMATH_GPT_log_minus_one_has_one_zero_l1733_173390


namespace NUMINAMATH_GPT_three_digit_even_with_sum_twelve_l1733_173378

theorem three_digit_even_with_sum_twelve :
  ∃ n: ℕ, n = 36 ∧ 
    (∀ x, 100 ≤ x ∧ x ≤ 999 ∧ x % 2 = 0 ∧ 
          ((x / 10) % 10 + x % 10 = 12) → x = n) :=
sorry

end NUMINAMATH_GPT_three_digit_even_with_sum_twelve_l1733_173378


namespace NUMINAMATH_GPT_expression_undefined_count_l1733_173314

theorem expression_undefined_count : ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ ∀ x : ℝ,
  ((x = x1 ∨ x = x2) ↔ (x^2 - 2*x - 3 = 0 ∨ x - 3 = 0)) ∧ 
  ((x^2 - 2*x - 3) * (x - 3) = 0 → (x = x1 ∨ x = x2)) :=
by
  sorry

end NUMINAMATH_GPT_expression_undefined_count_l1733_173314


namespace NUMINAMATH_GPT_line_tangent_to_ellipse_l1733_173333

theorem line_tangent_to_ellipse (m : ℝ) : 
  (∀ x y : ℝ, y = m * x + 2 ∧ x^2 + 9 * y^2 = 1 → m^2 = 35/9) := 
sorry

end NUMINAMATH_GPT_line_tangent_to_ellipse_l1733_173333


namespace NUMINAMATH_GPT_projection_of_AB_on_AC_l1733_173301

noncomputable def A : ℝ × ℝ := (-1, 1)
noncomputable def B : ℝ × ℝ := (0, 3)
noncomputable def C : ℝ × ℝ := (3, 4)

noncomputable def vectorAB := (B.1 - A.1, B.2 - A.2)
noncomputable def vectorAC := (C.1 - A.1, C.2 - A.2)

noncomputable def dotProduct (u v : ℝ × ℝ) : ℝ :=
  u.1 * v.1 + u.2 * v.2

noncomputable def magnitude (v : ℝ × ℝ) : ℝ :=
  Real.sqrt (v.1 ^ 2 + v.2 ^ 2)

theorem projection_of_AB_on_AC :
  (dotProduct vectorAB vectorAC) / (magnitude vectorAC) = 2 :=
  sorry

end NUMINAMATH_GPT_projection_of_AB_on_AC_l1733_173301


namespace NUMINAMATH_GPT_exp_fixed_point_l1733_173374

theorem exp_fixed_point (a : ℝ) (ha1 : a > 0) (ha2 : a ≠ 1) : a^0 = 1 :=
by
  exact one_pow 0

end NUMINAMATH_GPT_exp_fixed_point_l1733_173374


namespace NUMINAMATH_GPT_number_of_ordered_pairs_l1733_173381

theorem number_of_ordered_pairs {x y: ℕ} (h1 : x < y) (h2 : 2 * x * y / (x + y) = 4^30) : 
  ∃ n, n = 61 :=
sorry

end NUMINAMATH_GPT_number_of_ordered_pairs_l1733_173381


namespace NUMINAMATH_GPT_infection_probability_l1733_173310

theorem infection_probability
  (malaria_percent : ℝ)
  (zika_percent : ℝ)
  (vaccine_reduction : ℝ)
  (prob_random_infection : ℝ)
  (P : ℝ) :
  malaria_percent = 0.40 →
  zika_percent = 0.20 →
  vaccine_reduction = 0.50 →
  prob_random_infection = 0.15 →
  0.15 = (0.40 * 0.50 * P) + (0.20 * P) →
  P = 0.375 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end NUMINAMATH_GPT_infection_probability_l1733_173310


namespace NUMINAMATH_GPT_permutations_of_3_3_3_7_7_l1733_173367

noncomputable def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

theorem permutations_of_3_3_3_7_7 : 
  (factorial 5) / (factorial 3 * factorial 2) = 10 :=
by
  sorry

end NUMINAMATH_GPT_permutations_of_3_3_3_7_7_l1733_173367


namespace NUMINAMATH_GPT_soccer_match_outcome_l1733_173347

theorem soccer_match_outcome :
  ∃ n : ℕ, n = 4 ∧
  (∃ (num_wins num_draws num_losses : ℕ),
     num_wins * 3 + num_draws * 1 + num_losses * 0 = 19 ∧
     num_wins + num_draws + num_losses = 14) :=
sorry

end NUMINAMATH_GPT_soccer_match_outcome_l1733_173347


namespace NUMINAMATH_GPT_rook_placements_5x5_l1733_173387

/-- The number of ways to place five distinct rooks on a 
  5x5 chess board such that each column and row of the 
  board contains exactly one rook is 120. -/
theorem rook_placements_5x5 : 
  ∃! (f : Fin 5 → Fin 5), Function.Bijective f :=
by
  sorry

end NUMINAMATH_GPT_rook_placements_5x5_l1733_173387


namespace NUMINAMATH_GPT_intersection_of_M_and_N_l1733_173371

noncomputable def M : Set ℝ := {x | -1 < x ∧ x < 5}
noncomputable def N : Set ℝ := {x | x * (x - 4) > 0}

theorem intersection_of_M_and_N :
  M ∩ N = { x : ℝ | (-1 < x ∧ x < 0) ∨ (4 < x ∧ x < 5) } := by
  sorry

end NUMINAMATH_GPT_intersection_of_M_and_N_l1733_173371


namespace NUMINAMATH_GPT_certain_number_value_l1733_173372

theorem certain_number_value (x : ℝ) (certain_number : ℝ) 
  (h1 : x = 0.25) 
  (h2 : 625^(-x) + 25^(-2 * x) + certain_number^(-4 * x) = 11) : 
  certain_number = 5 / 53 := 
sorry

end NUMINAMATH_GPT_certain_number_value_l1733_173372


namespace NUMINAMATH_GPT_last_two_nonzero_digits_of_70_factorial_are_04_l1733_173335

-- Given conditions
def factorial (n : ℕ) : ℕ := if n = 0 then 1 else n * factorial (n - 1)

-- Theorem statement
theorem last_two_nonzero_digits_of_70_factorial_are_04 :
  let n := 70;
  ∀ t : ℕ, 
    t = factorial n → t % 100 ≠ 0 → (t % 100) / 10 != 0 → 
    (t % 100) = 04 :=
sorry

end NUMINAMATH_GPT_last_two_nonzero_digits_of_70_factorial_are_04_l1733_173335


namespace NUMINAMATH_GPT_domain_of_x_l1733_173358

-- Conditions
def is_defined_num (x : ℝ) : Prop := x + 1 >= 0
def not_zero_den (x : ℝ) : Prop := x ≠ 2

-- Proof problem statement
theorem domain_of_x (x : ℝ) : (is_defined_num x ∧ not_zero_den x) ↔ (x >= -1 ∧ x ≠ 2) := by
  sorry

end NUMINAMATH_GPT_domain_of_x_l1733_173358


namespace NUMINAMATH_GPT_range_of_m_l1733_173394

theorem range_of_m (m : ℝ) : (1^2 + 2*1 - m ≤ 0) ∧ (2^2 + 2*2 - m > 0) → 3 ≤ m ∧ m < 8 := by
  sorry

end NUMINAMATH_GPT_range_of_m_l1733_173394


namespace NUMINAMATH_GPT_total_liquid_consumption_l1733_173305

-- Define the given conditions
def elijah_drink_pints : ℝ := 8.5
def emilio_drink_pints : ℝ := 9.5
def isabella_drink_liters : ℝ := 3
def xavier_drink_gallons : ℝ := 2
def pint_to_cups : ℝ := 2
def liter_to_cups : ℝ := 4.22675
def gallon_to_cups : ℝ := 16
def xavier_soda_fraction : ℝ := 0.60
def xavier_fruit_punch_fraction : ℝ := 0.40

-- Define the converted amounts
def elijah_cups := elijah_drink_pints * pint_to_cups
def emilio_cups := emilio_drink_pints * pint_to_cups
def isabella_cups := isabella_drink_liters * liter_to_cups
def xavier_total_cups := xavier_drink_gallons * gallon_to_cups
def xavier_soda_cups := xavier_soda_fraction * xavier_total_cups
def xavier_fruit_punch_cups := xavier_fruit_punch_fraction * xavier_total_cups

-- Total amount calculation
def total_cups := elijah_cups + emilio_cups + isabella_cups + xavier_soda_cups + xavier_fruit_punch_cups

-- Proof statement
theorem total_liquid_consumption : total_cups = 80.68025 := by
  sorry

end NUMINAMATH_GPT_total_liquid_consumption_l1733_173305


namespace NUMINAMATH_GPT_firstDiscountIsTenPercent_l1733_173388

def listPrice : ℝ := 70
def finalPrice : ℝ := 56.16
def secondDiscount : ℝ := 10.857142857142863

theorem firstDiscountIsTenPercent (x : ℝ) : 
    finalPrice = listPrice * (1 - x / 100) * (1 - secondDiscount / 100) ↔ x = 10 := 
by
  sorry

end NUMINAMATH_GPT_firstDiscountIsTenPercent_l1733_173388


namespace NUMINAMATH_GPT_S_equals_l1733_173334
noncomputable def S : Real :=
  1 / (5 - Real.sqrt 23) + 1 / (Real.sqrt 23 - Real.sqrt 20) - 1 / (Real.sqrt 20 - 4) -
  1 / (4 - Real.sqrt 15) + 1 / (Real.sqrt 15 - Real.sqrt 12) - 1 / (Real.sqrt 12 - 3)

theorem S_equals : S = 2 * Real.sqrt 23 - 2 :=
by
  sorry

end NUMINAMATH_GPT_S_equals_l1733_173334


namespace NUMINAMATH_GPT_probability_of_draw_l1733_173344

-- Let P be the probability of the game ending in a draw.
-- Let PA be the probability of Player A winning.

def PA_not_losing := 0.8
def PB_not_losing := 0.7

theorem probability_of_draw : ¬ (1 - PA_not_losing + PB_not_losing ≠ 1.5) → PA_not_losing + (1 - PB_not_losing) = 1.5 → PB_not_losing + 0.5 = 1 := by
  intros
  sorry

end NUMINAMATH_GPT_probability_of_draw_l1733_173344


namespace NUMINAMATH_GPT_knights_max_seated_between_knights_l1733_173369

theorem knights_max_seated_between_knights {n k : ℕ} (h1 : n = 40) (h2 : k = 10) (h3 : ∃ (x : ℕ), x = 7) :
  ∃ (m : ℕ), m = 32 :=
by
  sorry

end NUMINAMATH_GPT_knights_max_seated_between_knights_l1733_173369


namespace NUMINAMATH_GPT_probability_one_black_one_red_l1733_173376

theorem probability_one_black_one_red (R B : Finset ℕ) (hR : R.card = 2) (hB : B.card = 3) :
  (2 : ℚ) / 5 = (6 + 6) / (5 * 4) := by
  sorry

end NUMINAMATH_GPT_probability_one_black_one_red_l1733_173376


namespace NUMINAMATH_GPT_roger_initial_money_l1733_173346

theorem roger_initial_money (spent_on_game : ℕ) (cost_per_toy : ℕ) (num_toys : ℕ) (total_money_spent : ℕ) :
  spent_on_game = 48 →
  cost_per_toy = 3 →
  num_toys = 5 →
  total_money_spent = spent_on_game + num_toys * cost_per_toy →
  total_money_spent = 63 :=
by
  intros h_game h_toy_cost h_num_toys h_total_spent
  rw [h_game, h_toy_cost, h_num_toys] at h_total_spent
  exact h_total_spent

end NUMINAMATH_GPT_roger_initial_money_l1733_173346


namespace NUMINAMATH_GPT_constant_k_independent_of_b_l1733_173352

noncomputable def algebraic_expression (a b k : ℝ) : ℝ :=
  a * b * (5 * k * a - 3 * b) - (k * a - b) * (3 * a * b - 4 * a^2)

theorem constant_k_independent_of_b (a : ℝ) : (algebraic_expression a b 2) = (algebraic_expression a 1 2) :=
by
  sorry

end NUMINAMATH_GPT_constant_k_independent_of_b_l1733_173352


namespace NUMINAMATH_GPT_faster_pipe_rate_l1733_173391

-- Set up our variables and the condition
variable (F S : ℝ)
variable (n : ℕ)

-- Given conditions
axiom S_rate : S = 1 / 180
axiom combined_rate : F + S = 1 / 36
axiom faster_rate : F = n * S

-- Theorem to prove
theorem faster_pipe_rate : n = 4 := by
  sorry

end NUMINAMATH_GPT_faster_pipe_rate_l1733_173391


namespace NUMINAMATH_GPT_arithmetic_sequence_sum_l1733_173338

variable {a : ℕ → ℝ}

noncomputable def sum_of_first_ten_terms (a : ℕ → ℝ) : ℝ :=
  (10 / 2) * (a 1 + a 10)

theorem arithmetic_sequence_sum (h : a 5 + a 6 = 28) :
  sum_of_first_ten_terms a = 140 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_sum_l1733_173338


namespace NUMINAMATH_GPT_route_comparison_l1733_173393

-- Definitions based on given conditions

def time_uphill : ℕ := 6
def time_path : ℕ := 2 * time_uphill
def total_first_two_stages : ℕ := time_uphill + time_path
def time_final_stage : ℕ := total_first_two_stages / 3
def total_time_first_route : ℕ := total_first_two_stages + time_final_stage

def time_flat_path : ℕ := 14
def time_second_stage : ℕ := 2 * time_flat_path
def total_time_second_route : ℕ := time_flat_path + time_second_stage

-- Statement we want to prove
theorem route_comparison : 
  total_time_second_route - total_time_first_route = 18 := by
  sorry

end NUMINAMATH_GPT_route_comparison_l1733_173393


namespace NUMINAMATH_GPT_intersection_M_N_l1733_173307

def M (x : ℝ) : Prop := Real.log x / Real.log 2 ≥ 0
def N (x : ℝ) : Prop := -2 ≤ x ∧ x ≤ 2

theorem intersection_M_N :
  {x : ℝ | M x} ∩ {x | N x} = {x | 1 ≤ x ∧ x ≤ 2} :=
by
  sorry

end NUMINAMATH_GPT_intersection_M_N_l1733_173307


namespace NUMINAMATH_GPT_max_value_x3y2z_l1733_173350

theorem max_value_x3y2z
  (x y z : ℝ)
  (hx_pos : 0 < x)
  (hy_pos : 0 < y)
  (hz_pos : 0 < z)
  (h_total : x + 2 * y + 3 * z = 1)
  : x^3 * y^2 * z ≤ 2048 / 11^6 := 
by
  sorry

end NUMINAMATH_GPT_max_value_x3y2z_l1733_173350


namespace NUMINAMATH_GPT_max_ad_minus_bc_l1733_173311

theorem max_ad_minus_bc (a b c d : ℤ) (ha : a ∈ Set.image (fun x => x) {(-1), 1, 2})
                         (hb : b ∈ Set.image (fun x => x) {(-1), 1, 2})
                         (hc : c ∈ Set.image (fun x => x) {(-1), 1, 2})
                         (hd : d ∈ Set.image (fun x => x) {(-1), 1, 2}) :
  ad - bc ≤ 6 :=
sorry

end NUMINAMATH_GPT_max_ad_minus_bc_l1733_173311


namespace NUMINAMATH_GPT_find_n_l1733_173395

open Classical

theorem find_n (n : ℕ) (h : (8 * Nat.choose n 3) = 8 * (2 * Nat.choose n 1)) : n = 5 := by
  sorry

end NUMINAMATH_GPT_find_n_l1733_173395


namespace NUMINAMATH_GPT_total_amount_spent_is_300_l1733_173397

-- Definitions of conditions
def S : ℕ := 97
def H : ℕ := 2 * S + 9

-- The total amount spent
def total_spent : ℕ := S + H

-- Proof statement
theorem total_amount_spent_is_300 : total_spent = 300 :=
by
  sorry

end NUMINAMATH_GPT_total_amount_spent_is_300_l1733_173397


namespace NUMINAMATH_GPT_sam_initial_investment_is_6000_l1733_173368

variables (P : ℝ)
noncomputable def final_amount (P : ℝ) : ℝ :=
  P * (1 + 0.10 / 2) ^ (2 * 1)

theorem sam_initial_investment_is_6000 :
  final_amount 6000 = 6615 :=
by
  unfold final_amount
  sorry

end NUMINAMATH_GPT_sam_initial_investment_is_6000_l1733_173368


namespace NUMINAMATH_GPT_maximum_value_x_squared_plus_2y_l1733_173349

theorem maximum_value_x_squared_plus_2y (x y b : ℝ) (h_curve : x^2 / 4 + y^2 / b^2 = 1) (h_b_positive : b > 0) : 
  x^2 + 2 * y ≤ max (b^2 / 4 + 4) (2 * b) :=
sorry

end NUMINAMATH_GPT_maximum_value_x_squared_plus_2y_l1733_173349


namespace NUMINAMATH_GPT_find_f3_l1733_173389

noncomputable def f : ℝ → ℝ := sorry
noncomputable def g : ℝ → ℝ := sorry

def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x
def is_even (g : ℝ → ℝ) : Prop := ∀ x, g (-x) = g x

theorem find_f3 
  (hf : is_odd f) 
  (hg : is_even g) 
  (h : ∀ x, f x + g x = 1 / (x - 1)) : 
  f 3 = 3 / 8 :=
by 
  sorry

end NUMINAMATH_GPT_find_f3_l1733_173389


namespace NUMINAMATH_GPT_quadratic_eq_real_roots_m_ge_neg1_quadratic_eq_real_roots_cond_l1733_173325

theorem quadratic_eq_real_roots_m_ge_neg1 (m : ℝ) :
  (∃ x1 x2 : ℝ, x1^2 + 2*(m+1)*x1 + m^2 - 1 = 0 ∧ x2^2 + 2*(m+1)*x2 + m^2 - 1 = 0) →
  m ≥ -1 :=
sorry

theorem quadratic_eq_real_roots_cond (m : ℝ) (x1 x2 : ℝ) :
  x1^2 + 2*(m+1)*x1 + m^2 - 1 = 0 ∧ x2^2 + 2*(m+1)*x2 + m^2 - 1 = 0 ∧
  (x1 - x2)^2 = 16 - x1 * x2 →
  m = 1 :=
sorry

end NUMINAMATH_GPT_quadratic_eq_real_roots_m_ge_neg1_quadratic_eq_real_roots_cond_l1733_173325


namespace NUMINAMATH_GPT_exists_positive_integer_solution_l1733_173351

theorem exists_positive_integer_solution (m : ℕ) (hm : 0 < m) :
  ∃ n : ℕ, 0 < n ∧ n / m = ⌊(n^2 : ℝ)^(1/3)⌋ + ⌊(n : ℝ)^(1/2)⌋ + 1 := 
by
  sorry

end NUMINAMATH_GPT_exists_positive_integer_solution_l1733_173351


namespace NUMINAMATH_GPT_largest_positive_integer_l1733_173384

def binary_op (n : ℕ) : ℤ := n - (n * 5)

theorem largest_positive_integer (n : ℕ) (h : binary_op n < 21) : n ≤ 1 := 
sorry

end NUMINAMATH_GPT_largest_positive_integer_l1733_173384


namespace NUMINAMATH_GPT_function_properties_l1733_173399

noncomputable def f (x : ℝ) : ℝ := x^2

theorem function_properties :
  (∀ x1 x2 : ℝ, f (x1 * x2) = f x1 * f x2) ∧
  (∀ x : ℝ, 0 < x → deriv f x > 0) ∧
  (∀ x : ℝ, deriv f (-x) = -deriv f x) :=
by
  sorry

end NUMINAMATH_GPT_function_properties_l1733_173399


namespace NUMINAMATH_GPT_relation_between_p_and_q_l1733_173309

theorem relation_between_p_and_q (p q : ℝ) (α : ℝ) 
  (h1 : α + 2 * α = -p) 
  (h2 : α * (2 * α) = q) : 
  2 * p^2 = 9 * q := 
by 
  -- simplifying the provided conditions
  sorry

end NUMINAMATH_GPT_relation_between_p_and_q_l1733_173309


namespace NUMINAMATH_GPT_units_digit_product_is_2_l1733_173362

def units_digit_product : ℕ := 
  (10 * 11 * 12 * 13 * 14 * 15 * 16) / 800 % 10

theorem units_digit_product_is_2 : units_digit_product = 2 := 
by
  sorry

end NUMINAMATH_GPT_units_digit_product_is_2_l1733_173362


namespace NUMINAMATH_GPT_percentage_increase_l1733_173332

theorem percentage_increase (L : ℝ) (h : L + 60 = 240) : ((60 / L) * 100 = 33 + (1 / 3) * 100) :=
by
  sorry

end NUMINAMATH_GPT_percentage_increase_l1733_173332


namespace NUMINAMATH_GPT_two_a_minus_b_values_l1733_173337

theorem two_a_minus_b_values (a b : ℝ) (h1 : |a| = 4) (h2 : |b| = 5) (h3 : |a + b| = -(a + b)) :
  (2 * a - b = 13) ∨ (2 * a - b = -3) :=
sorry

end NUMINAMATH_GPT_two_a_minus_b_values_l1733_173337


namespace NUMINAMATH_GPT_average_speed_is_9_mph_l1733_173342

-- Define the conditions
def distance_north_ft := 5280
def north_speed_min_per_mile := 3
def rest_time_min := 10
def south_speed_miles_per_min := 3

-- Define a function to convert feet to miles
def feet_to_miles (ft : ℕ) : ℕ := ft / 5280

-- Define the time calculation for north and south trips
def time_north_min (speed : ℕ) (distance_ft : ℕ) : ℕ :=
  speed * feet_to_miles distance_ft

def time_south_min (speed_miles_per_min : ℕ) (distance_ft : ℕ) : ℕ :=
  (feet_to_miles distance_ft) / speed_miles_per_min

def total_time_min (time_north rest_time time_south : ℕ) : Rat :=
  time_north + rest_time + time_south

-- Convert total time into hours
def total_time_hr (total_time_min : Rat) : Rat :=
  total_time_min / 60

-- Define the total distance in miles
def total_distance_miles (distance_ft : ℕ) : ℕ :=
  2 * feet_to_miles distance_ft

-- Calculate the average speed
def average_speed (total_distance : ℕ) (total_time_hr : Rat) : Rat :=
  total_distance / total_time_hr

-- Prove the average speed is 9 miles per hour
theorem average_speed_is_9_mph : 
  average_speed (total_distance_miles distance_north_ft)
                (total_time_hr (total_time_min (time_north_min north_speed_min_per_mile distance_north_ft)
                                              rest_time_min
                                              (time_south_min south_speed_miles_per_min distance_north_ft)))
    = 9 := by
  sorry

end NUMINAMATH_GPT_average_speed_is_9_mph_l1733_173342


namespace NUMINAMATH_GPT_quadratic_two_distinct_real_roots_l1733_173340

theorem quadratic_two_distinct_real_roots (k : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ (k+2) * x^2 + 4 * x + 1 = 0 ∧ (k+2) * y^2 + 4 * y + 1 = 0) ↔ 
  (k < 2 ∧ k ≠ -2) :=
sorry

end NUMINAMATH_GPT_quadratic_two_distinct_real_roots_l1733_173340


namespace NUMINAMATH_GPT_flat_fee_for_solar_panel_equipment_l1733_173330

theorem flat_fee_for_solar_panel_equipment
  (land_acreage : ℕ)
  (land_cost_per_acre : ℕ)
  (house_cost : ℕ)
  (num_cows : ℕ)
  (cow_cost_per_cow : ℕ)
  (num_chickens : ℕ)
  (chicken_cost_per_chicken : ℕ)
  (installation_hours : ℕ)
  (installation_cost_per_hour : ℕ)
  (total_cost : ℕ)
  (total_spent : ℕ) :
  land_acreage * land_cost_per_acre + house_cost +
  num_cows * cow_cost_per_cow + num_chickens * chicken_cost_per_chicken +
  installation_hours * installation_cost_per_hour = total_spent →
  total_cost = total_spent →
  total_cost - (land_acreage * land_cost_per_acre + house_cost +
  num_cows * cow_cost_per_cow + num_chickens * chicken_cost_per_chicken +
  installation_hours * installation_cost_per_hour) = 26000 := by 
  sorry

end NUMINAMATH_GPT_flat_fee_for_solar_panel_equipment_l1733_173330


namespace NUMINAMATH_GPT_triangle_division_point_distances_l1733_173345

theorem triangle_division_point_distances 
  {a b c : ℝ} 
  (h1 : a = 13) 
  (h2 : b = 17) 
  (h3 : c = 24)
  (h4 : ∃ p q : ℝ, p = 9 ∧ q = 11) : 
  ∃ p q : ℝ, p = 9 ∧ q = 11 :=
  sorry

end NUMINAMATH_GPT_triangle_division_point_distances_l1733_173345


namespace NUMINAMATH_GPT_max_value_of_trig_expression_l1733_173383

open Real

theorem max_value_of_trig_expression : ∀ x : ℝ, 3 * cos x + 4 * sin x ≤ 5 :=
sorry

end NUMINAMATH_GPT_max_value_of_trig_expression_l1733_173383


namespace NUMINAMATH_GPT_smallest_number_of_slices_l1733_173313

-- Definition of the number of slices in each type of cheese package
def slices_of_cheddar : ℕ := 12
def slices_of_swiss : ℕ := 28

-- Predicate stating that the smallest number of slices of each type Randy could have bought is 84
theorem smallest_number_of_slices : Nat.lcm slices_of_cheddar slices_of_swiss = 84 := by
  sorry

end NUMINAMATH_GPT_smallest_number_of_slices_l1733_173313


namespace NUMINAMATH_GPT_find_functions_l1733_173336

def is_non_decreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x ≤ y → f x ≤ f y

theorem find_functions (f : ℝ × ℝ → ℝ) :
  (is_non_decreasing (λ x => f (0, x))) →
  (∀ x y, f (x, y) = f (y, x)) →
  (∀ x y z, (f (x, y) - f (y, z)) * (f (y, z) - f (z, x)) * (f (z, x) - f (x, y)) = 0) →
  (∀ x y a, f (x + a, y + a) = f (x, y) + a) →
  (∃ a : ℝ, (∀ x y, f (x, y) = a + min x y) ∨ (∀ x y, f (x, y) = a + max x y)) :=
  by sorry

end NUMINAMATH_GPT_find_functions_l1733_173336


namespace NUMINAMATH_GPT_correct_graph_is_C_l1733_173302

-- Define the years and corresponding remote work percentages
def percentages : List (ℕ × ℝ) := [
  (1960, 0.1),
  (1970, 0.15),
  (1980, 0.12),
  (1990, 0.25),
  (2000, 0.4)
]

-- Define the property of the graph trend
def isCorrectGraph (p : List (ℕ × ℝ)) : Prop :=
  p = [
    (1960, 0.1),
    (1970, 0.15),
    (1980, 0.12),
    (1990, 0.25),
    (2000, 0.4)
  ]

-- State the theorem
theorem correct_graph_is_C : isCorrectGraph percentages = True :=
  sorry

end NUMINAMATH_GPT_correct_graph_is_C_l1733_173302


namespace NUMINAMATH_GPT_tower_height_count_l1733_173329

theorem tower_height_count (bricks : ℕ) (height1 height2 height3 : ℕ) :
  height1 = 3 → height2 = 11 → height3 = 18 → bricks = 100 →
  (∃ (h : ℕ),  h = 1404) :=
by
  sorry

end NUMINAMATH_GPT_tower_height_count_l1733_173329


namespace NUMINAMATH_GPT_fish_added_l1733_173375

theorem fish_added (T C : ℕ) (h1 : T + C = 20) (h2 : C = T - 4) : C = 8 :=
by
  sorry

end NUMINAMATH_GPT_fish_added_l1733_173375


namespace NUMINAMATH_GPT_initial_bones_count_l1733_173322

theorem initial_bones_count (B : ℕ) (h1 : B + 8 = 23) : B = 15 :=
sorry

end NUMINAMATH_GPT_initial_bones_count_l1733_173322


namespace NUMINAMATH_GPT_decomposition_sum_of_cubes_l1733_173396

theorem decomposition_sum_of_cubes 
  (a b c d e : ℤ) 
  (h : (512 : ℤ) * x ^ 3 + 27 = (a * x + b) * (c * x ^ 2 + d * x + e)) :
  a + b + c + d + e = 60 := 
sorry

end NUMINAMATH_GPT_decomposition_sum_of_cubes_l1733_173396


namespace NUMINAMATH_GPT_yolkino_palkino_l1733_173339

open Nat

/-- On every kilometer of the highway between the villages Yolkino and Palkino, there is a post with a sign.
    On one side of the sign, the distance to Yolkino is written, and on the other side, the distance to Palkino is written.
    The sum of all the digits on each post equals 13.
    Prove that the distance from Yolkino to Palkino is 49 kilometers. -/
theorem yolkino_palkino (n : ℕ) (h : ∀ k : ℕ, k ≤ n → (digits 10 k).sum + (digits 10 (n - k)).sum = 13) : n = 49 :=
by
  sorry

end NUMINAMATH_GPT_yolkino_palkino_l1733_173339


namespace NUMINAMATH_GPT_total_kids_in_camp_l1733_173300

-- Definitions from the conditions
variables (X : ℕ)
def kids_going_to_soccer_camp := X / 2
def kids_going_to_soccer_camp_morning := kids_going_to_soccer_camp / 4
def kids_going_to_soccer_camp_afternoon := kids_going_to_soccer_camp - kids_going_to_soccer_camp_morning

-- Given condition that 750 kids are going to soccer camp in the afternoon
axiom h : kids_going_to_soccer_camp_afternoon X = 750

-- The statement to prove that X = 2000
theorem total_kids_in_camp : X = 2000 :=
sorry

end NUMINAMATH_GPT_total_kids_in_camp_l1733_173300


namespace NUMINAMATH_GPT_geometric_series_sum_l1733_173320

-- Define the first term and common ratio
def a : ℚ := 5 / 3
def r : ℚ := -1 / 6

-- Prove the sum of the infinite geometric series
theorem geometric_series_sum : (∑' n : ℕ, a * r^n) = 10 / 7 := by
  sorry

end NUMINAMATH_GPT_geometric_series_sum_l1733_173320


namespace NUMINAMATH_GPT_calculation_proof_l1733_173364

theorem calculation_proof :
  5^(Real.log 9 / Real.log 5) + (1 / 2) * (Real.log 32 / Real.log 2) - Real.log (Real.log 8 / Real.log 2) / Real.log 3 = 21 / 2 := 
  sorry

end NUMINAMATH_GPT_calculation_proof_l1733_173364


namespace NUMINAMATH_GPT_avg_weekly_income_500_l1733_173386

theorem avg_weekly_income_500 :
  let base_salary := 350
  let income_past_5_weeks := [406, 413, 420, 436, 495]
  let commission_next_2_weeks_avg := 315
  let total_income_past_5_weeks := income_past_5_weeks.sum
  let total_base_salary_next_2_weeks := base_salary * 2
  let total_commission_next_2_weeks := commission_next_2_weeks_avg * 2
  let total_income := total_income_past_5_weeks + total_base_salary_next_2_weeks + total_commission_next_2_weeks
  let avg_weekly_income := total_income / 7
  avg_weekly_income = 500 := by
{
  sorry
}

end NUMINAMATH_GPT_avg_weekly_income_500_l1733_173386


namespace NUMINAMATH_GPT_four_points_no_obtuse_triangle_l1733_173355

noncomputable def probability_no_obtuse_triangle : ℝ :=
1 / 64

theorem four_points_no_obtuse_triangle (A B C D : circle) :
  (∀ (P Q : circle) (PQ_angle : ℝ), PQ_angle < π/2) → 
  probability_no_obtuse_triangle = 1 / 64 :=
sorry

end NUMINAMATH_GPT_four_points_no_obtuse_triangle_l1733_173355


namespace NUMINAMATH_GPT_other_root_of_quadratic_l1733_173327

theorem other_root_of_quadratic (m : ℝ) (h : (m + 2) * 0^2 - 0 + m^2 - 4 = 0) : 
  ∃ x : ℝ, (m + 2) * x^2 - x + m^2 - 4 = 0 ∧ x ≠ 0 ∧ x = 1/4 := 
sorry

end NUMINAMATH_GPT_other_root_of_quadratic_l1733_173327


namespace NUMINAMATH_GPT_cardboard_box_height_l1733_173353

theorem cardboard_box_height :
  ∃ (x : ℕ), x ≥ 0 ∧ 10 * x^2 + 4 * x ≥ 130 ∧ (2 * x + 1) = 9 :=
sorry

end NUMINAMATH_GPT_cardboard_box_height_l1733_173353


namespace NUMINAMATH_GPT_fixed_point_exists_trajectory_M_trajectory_equation_l1733_173357

variable (m : ℝ)
def line_l (x y : ℝ) : Prop := 2 * x + (1 + m) * y + 2 * m = 0
def point_P (x y : ℝ) : Prop := x = -1 ∧ y = 0

theorem fixed_point_exists :
  ∃ x y : ℝ, (line_l m x y ∧ x = 1 ∧ y = -2) :=
by
  sorry

theorem trajectory_M :
  ∃ (M: ℝ × ℝ), (line_l m M.1 M.2 ∧ M = (0, -1)) :=
by
  sorry

theorem trajectory_equation (x y : ℝ) :
  ∃ (x y : ℝ), (x + 1) ^ 2  + y ^ 2 = 2 :=
by
  sorry

end NUMINAMATH_GPT_fixed_point_exists_trajectory_M_trajectory_equation_l1733_173357


namespace NUMINAMATH_GPT_cricket_run_target_l1733_173312

theorem cricket_run_target
  (run_rate_1st_period : ℝ)
  (overs_1st_period : ℕ)
  (run_rate_2nd_period : ℝ)
  (overs_2nd_period : ℕ)
  (target_runs : ℝ)
  (h1 : run_rate_1st_period = 3.2)
  (h2 : overs_1st_period = 10)
  (h3 : run_rate_2nd_period = 5)
  (h4 : overs_2nd_period = 50) :
  target_runs = (run_rate_1st_period * overs_1st_period) + (run_rate_2nd_period * overs_2nd_period) :=
by
  sorry

end NUMINAMATH_GPT_cricket_run_target_l1733_173312


namespace NUMINAMATH_GPT_x_coordinate_of_equidistant_point_l1733_173377

theorem x_coordinate_of_equidistant_point (x : ℝ) : 
  ((-3 - x)^2 + (-2 - 0)^2) = ((2 - x)^2 + (-6 - 0)^2) → x = 2.7 :=
by
  sorry

end NUMINAMATH_GPT_x_coordinate_of_equidistant_point_l1733_173377


namespace NUMINAMATH_GPT_sin_theta_of_triangle_l1733_173373

theorem sin_theta_of_triangle (area : ℝ) (side : ℝ) (median : ℝ) (θ : ℝ)
  (h_area : area = 30)
  (h_side : side = 10)
  (h_median : median = 9) :
  Real.sin θ = 2 / 3 := by
  sorry

end NUMINAMATH_GPT_sin_theta_of_triangle_l1733_173373


namespace NUMINAMATH_GPT_problem_l1733_173343

variables (x : ℝ)

-- Define the condition
def condition (x : ℝ) : Prop :=
  0.3 * (0.2 * x) = 24

-- Define the target statement
def target (x : ℝ) : Prop :=
  0.2 * (0.3 * x) = 24

-- The theorem we want to prove
theorem problem (x : ℝ) (h : condition x) : target x :=
sorry

end NUMINAMATH_GPT_problem_l1733_173343
