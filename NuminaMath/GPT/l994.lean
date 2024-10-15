import Mathlib

namespace NUMINAMATH_GPT_blue_tissue_length_exists_l994_99452

theorem blue_tissue_length_exists (B R : ℝ) (h1 : R = B + 12) (h2 : 2 * R = 3 * B) : B = 24 := 
by
  sorry

end NUMINAMATH_GPT_blue_tissue_length_exists_l994_99452


namespace NUMINAMATH_GPT_part1_part2_l994_99453

variable (a b c : ℝ)
variable (h1 : a > 0) (h2 : b > 0) (h3 : c > 0)
variable (h_sum : a + b + c = 1)

theorem part1 : 2 * a * b + b * c + c * a + c ^ 2 / 2 ≤ 1 / 2 := sorry

theorem part2 : (a ^ 2 + c ^ 2) / b + (b ^ 2 + a ^ 2) / c + (c ^ 2 + b ^ 2) / a ≥ 2 := sorry

end NUMINAMATH_GPT_part1_part2_l994_99453


namespace NUMINAMATH_GPT_red_balls_count_l994_99435

-- Define the conditions
def white_red_ratio : ℕ × ℕ := (5, 3)
def num_white_balls : ℕ := 15

-- Define the theorem to prove
theorem red_balls_count (r : ℕ) : r = num_white_balls / (white_red_ratio.1) * (white_red_ratio.2) :=
by sorry

end NUMINAMATH_GPT_red_balls_count_l994_99435


namespace NUMINAMATH_GPT_subtraction_example_l994_99478

theorem subtraction_example : 34.256 - 12.932 - 1.324 = 20.000 := 
by
  sorry

end NUMINAMATH_GPT_subtraction_example_l994_99478


namespace NUMINAMATH_GPT_find_coefficients_l994_99449

theorem find_coefficients (A B : ℝ) (h_roots : (x^2 + A * x + B = 0 ∧ (x = A ∨ x = B))) :
  (A = 0 ∧ B = 0) ∨ (A = 1 ∧ B = -2) :=
by sorry

end NUMINAMATH_GPT_find_coefficients_l994_99449


namespace NUMINAMATH_GPT_pieces_present_l994_99439

def total_pieces : ℕ := 32
def missing_pieces : ℕ := 10

theorem pieces_present : total_pieces - missing_pieces = 22 :=
by {
  sorry
}

end NUMINAMATH_GPT_pieces_present_l994_99439


namespace NUMINAMATH_GPT_percentage_of_games_not_won_is_40_l994_99479

def ratio_games_won_to_lost (games_won games_lost : ℕ) : Prop := 
  games_won / gcd games_won games_lost = 3 ∧ games_lost / gcd games_won games_lost = 2

def total_games (games_won games_lost ties : ℕ) : ℕ :=
  games_won + games_lost + ties

def percentage_games_not_won (games_won games_lost ties : ℕ) : ℕ :=
  ((games_lost + ties) * 100) / (games_won + games_lost + ties)

theorem percentage_of_games_not_won_is_40
  (games_won games_lost ties : ℕ)
  (h_ratio : ratio_games_won_to_lost games_won games_lost)
  (h_ties : ties = 5)
  (h_no_other_games : games_won + games_lost + ties = total_games games_won games_lost ties) :
  percentage_games_not_won games_won games_lost ties = 40 := 
sorry

end NUMINAMATH_GPT_percentage_of_games_not_won_is_40_l994_99479


namespace NUMINAMATH_GPT_storm_first_thirty_minutes_rain_l994_99434

theorem storm_first_thirty_minutes_rain 
  (R: ℝ)
  (H1: R + (R / 2) + (1 / 2) = 8)
  : R = 5 :=
by
  sorry

end NUMINAMATH_GPT_storm_first_thirty_minutes_rain_l994_99434


namespace NUMINAMATH_GPT_maximum_sum_of_composites_l994_99482

def is_composite (n : ℕ) : Prop :=
  ∃ a b : ℕ, 1 < a ∧ 1 < b ∧ n = a * b

def pairwise_coprime (A B C : ℕ) : Prop :=
  Nat.gcd A B = 1 ∧ Nat.gcd A C = 1 ∧ Nat.gcd B C = 1

theorem maximum_sum_of_composites (A B C : ℕ)
  (hA : is_composite A) (hB : is_composite B) (hC : is_composite C)
  (h_pairwise : pairwise_coprime A B C)
  (h_prod_eq : A * B * C = 11011 * 28) :
  A + B + C = 1626 := 
sorry

end NUMINAMATH_GPT_maximum_sum_of_composites_l994_99482


namespace NUMINAMATH_GPT_combined_score_is_75_l994_99433

variable (score1 : ℕ) (total1 : ℕ)
variable (score2 : ℕ) (total2 : ℕ)
variable (score3 : ℕ) (total3 : ℕ)

-- Conditions: Antonette's scores and the number of problems in each test
def Antonette_scores : Prop :=
  score1 = 60 * total1 / 100 ∧ total1 = 15 ∧
  score2 = 85 * total2 / 100 ∧ total2 = 20 ∧
  score3 = 75 * total3 / 100 ∧ total3 = 25

-- Theorem to prove the combined score is 75% (45 out of 60) rounded to the nearest percent
theorem combined_score_is_75
  (h : Antonette_scores score1 total1 score2 total2 score3 total3) :
  100 * (score1 + score2 + score3) / (total1 + total2 + total3) = 75 :=
by sorry

end NUMINAMATH_GPT_combined_score_is_75_l994_99433


namespace NUMINAMATH_GPT_complex_solution_l994_99496

open Complex

theorem complex_solution (z : ℂ) (h : z + Complex.abs z = 1 + Complex.I) : z = Complex.I := 
by
  sorry

end NUMINAMATH_GPT_complex_solution_l994_99496


namespace NUMINAMATH_GPT_max_ab_l994_99411

theorem max_ab (a b : ℝ) (h1 : a + 2 * b = 1) (h2 : 0 < a) (h3 : 0 < b) : 
  ∃ (M : ℝ), M = 1 / 8 ∧ ∀ (a b : ℝ), (a + 2 * b = 1) → 0 < a → 0 < b → ab ≤ M :=
sorry

end NUMINAMATH_GPT_max_ab_l994_99411


namespace NUMINAMATH_GPT_harry_travel_ratio_l994_99477

theorem harry_travel_ratio
  (bus_initial_time : ℕ)
  (bus_rest_time : ℕ)
  (total_travel_time : ℕ)
  (walking_time : ℕ := total_travel_time - (bus_initial_time + bus_rest_time))
  (bus_total_time : ℕ := bus_initial_time + bus_rest_time)
  (ratio : ℚ := walking_time / bus_total_time)
  (h1 : bus_initial_time = 15)
  (h2 : bus_rest_time = 25)
  (h3 : total_travel_time = 60)
  : ratio = (1 / 2) := 
sorry

end NUMINAMATH_GPT_harry_travel_ratio_l994_99477


namespace NUMINAMATH_GPT_log_inequality_l994_99428

noncomputable def log (x : ℝ) : ℝ := Real.log x / Real.log 10

theorem log_inequality (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : a ≠ b) (h5 : b ≠ c) (h6 : c ≠ a) :
  log ((a + b) / 2) + log ((b + c) / 2) + log ((c + a) / 2) > log a + log b + log c :=
by
  sorry

end NUMINAMATH_GPT_log_inequality_l994_99428


namespace NUMINAMATH_GPT_tangent_line_of_ellipse_l994_99455

noncomputable def ellipse_tangent_line (a b x0 y0 x y : ℝ) : Prop :=
  x0 * x / a^2 + y0 * y / b^2 = 1

theorem tangent_line_of_ellipse
  (a b x0 y0 : ℝ)
  (h_ellipse : x0^2 / a^2 + y0^2 / b^2 = 1)
  (h_a_pos : a > 0)
  (h_b_pos : b > 0)
  (h_a_b : a > b) :
  ellipse_tangent_line a b x0 y0 x y :=
sorry

end NUMINAMATH_GPT_tangent_line_of_ellipse_l994_99455


namespace NUMINAMATH_GPT_luka_age_difference_l994_99401

theorem luka_age_difference (a l : ℕ) (h1 : a = 8) (h2 : ∀ m : ℕ, m = 6 → l = m + 4) : l - a = 2 :=
by
  -- Assume Aubrey's age is 8
  have ha : a = 8 := h1
  -- Assume Max's age at Aubrey's 8th birthday is 6
  have hl : l = 10 := h2 6 rfl
  -- Hence, Luka is 2 years older than Aubrey
  sorry

end NUMINAMATH_GPT_luka_age_difference_l994_99401


namespace NUMINAMATH_GPT_rate_is_five_l994_99436

noncomputable def rate_per_sq_meter (total_cost : ℕ) (total_area : ℕ) : ℕ :=
  total_cost / total_area

theorem rate_is_five :
  let length := 80
  let breadth := 60
  let road_width := 10
  let total_cost := 6500
  let area_road1 := road_width * breadth
  let area_road2 := road_width * length
  let area_intersection := road_width * road_width
  let total_area := area_road1 + area_road2 - area_intersection
  rate_per_sq_meter total_cost total_area = 5 :=
by
  sorry

end NUMINAMATH_GPT_rate_is_five_l994_99436


namespace NUMINAMATH_GPT_time_to_store_vaccine_l994_99420

def final_temp : ℤ := -24
def current_temp : ℤ := -4
def rate_of_change : ℤ := -5

theorem time_to_store_vaccine : 
  ∃ t : ℤ, current_temp + rate_of_change * t = final_temp ∧ t = 4 :=
by
  use 4
  sorry

end NUMINAMATH_GPT_time_to_store_vaccine_l994_99420


namespace NUMINAMATH_GPT_subtract_fractions_correct_l994_99427

theorem subtract_fractions_correct :
  (3 / 8 + 5 / 12 - 1 / 6) = (5 / 8) := by
sorry

end NUMINAMATH_GPT_subtract_fractions_correct_l994_99427


namespace NUMINAMATH_GPT_correct_statement_B_l994_99458

/-- Define the diameter of a sphere -/
def diameter (d : ℝ) (s : Set (ℝ × ℝ × ℝ)) : Prop :=
∃ x y : ℝ × ℝ × ℝ, x ∈ s ∧ y ∈ s ∧ dist x y = d ∧ ∀ z ∈ s, dist x y ≥ dist x z ∧ dist x y ≥ dist z y

/-- Define that a line segment connects two points on the sphere's surface and passes through the center -/
def connects_diameter (center : ℝ × ℝ × ℝ) (radius : ℝ) (x y : ℝ × ℝ × ℝ) : Prop :=
dist center x = radius ∧ dist center y = radius ∧ (x + y) / 2 = center

/-- A sphere is the set of all points at a fixed distance from the center -/
def sphere (center : ℝ × ℝ × ℝ) (radius : ℝ) : Set (ℝ × ℝ × ℝ) :=
{x | dist center x = radius}

theorem correct_statement_B (center : ℝ × ℝ × ℝ) (radius : ℝ) (x y : ℝ × ℝ × ℝ):
  (∀ (s : Set (ℝ × ℝ × ℝ)), sphere center radius = s → diameter (2 * radius) s)
  → connects_diameter center radius x y
  → (∃ d : ℝ, diameter d (sphere center radius)) := 
by
  intros
  sorry

end NUMINAMATH_GPT_correct_statement_B_l994_99458


namespace NUMINAMATH_GPT_geometric_sequence_sum_l994_99431

theorem geometric_sequence_sum (a r : ℚ) (h_a : a = 1/3) (h_r : r = 1/2) (S_n : ℚ) (h_S_n : S_n = 80/243) : ∃ n : ℕ, S_n = a * ((1 - r^n) / (1 - r)) ∧ n = 4 := by
  sorry

end NUMINAMATH_GPT_geometric_sequence_sum_l994_99431


namespace NUMINAMATH_GPT_bus_seat_problem_l994_99443

theorem bus_seat_problem 
  (left_seats : ℕ) 
  (right_seats := left_seats - 3) 
  (left_capacity := 3 * left_seats)
  (right_capacity := 3 * right_seats)
  (back_seat_capacity := 12)
  (total_capacity := left_capacity + right_capacity + back_seat_capacity)
  (h1 : total_capacity = 93) 
  : left_seats = 15 := 
by 
  sorry

end NUMINAMATH_GPT_bus_seat_problem_l994_99443


namespace NUMINAMATH_GPT_tan_frac_a_pi_six_eq_sqrt_three_l994_99468

theorem tan_frac_a_pi_six_eq_sqrt_three (a : ℝ) (h : (a, 9) ∈ { p : ℝ × ℝ | p.2 = 3 ^ p.1 }) : 
  Real.tan (a * Real.pi / 6) = Real.sqrt 3 := 
by
  sorry

end NUMINAMATH_GPT_tan_frac_a_pi_six_eq_sqrt_three_l994_99468


namespace NUMINAMATH_GPT_parabola_vertex_l994_99494

theorem parabola_vertex :
  ∀ (x : ℝ), (∃ y : ℝ, y = 2 * (x - 5)^2 + 3) → (5, 3) = (5, 3) :=
by
  intros x y_eq
  sorry

end NUMINAMATH_GPT_parabola_vertex_l994_99494


namespace NUMINAMATH_GPT_arithmetic_sequence_general_term_l994_99474

theorem arithmetic_sequence_general_term
  (d : ℕ) (a : ℕ → ℕ)
  (ha4 : a 4 = 14)
  (hd : d = 3) :
  ∃ a₁, ∀ n, a n = a₁ + (n - 1) * d := by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_general_term_l994_99474


namespace NUMINAMATH_GPT_prime_square_minus_one_divisible_by_24_l994_99425

theorem prime_square_minus_one_divisible_by_24 (p : ℕ) (hp : p ≥ 5) (prime_p : Nat.Prime p) : 
  ∃ k : ℤ, p^2 - 1 = 24 * k :=
  sorry

end NUMINAMATH_GPT_prime_square_minus_one_divisible_by_24_l994_99425


namespace NUMINAMATH_GPT_cone_lateral_area_l994_99498

/--
Given that the radius of the base of a cone is 3 cm and the slant height is 6 cm,
prove that the lateral area of this cone is 18π cm².
-/
theorem cone_lateral_area {r l : ℝ} (h_radius : r = 3) (h_slant_height : l = 6) :
  (π * r * l) = 18 * π :=
by
  have h1 : r = 3 := h_radius
  have h2 : l = 6 := h_slant_height
  rw [h1, h2]
  norm_num
  sorry

end NUMINAMATH_GPT_cone_lateral_area_l994_99498


namespace NUMINAMATH_GPT_geometric_sequence_value_sum_l994_99442

variable {a : ℕ → ℝ}

noncomputable def is_geometric_sequence (a : ℕ → ℝ) :=
  ∀ n m, a (n + m) * a 0 = a n * a m

theorem geometric_sequence_value_sum {a : ℕ → ℝ}
  (hpos : ∀ n, a n > 0)
  (geom : is_geometric_sequence a)
  (given : a 0 * a 2 + 2 * a 1 * a 3 + a 2 * a 4 = 16) 
  : a 1 + a 3 = 4 :=
sorry

end NUMINAMATH_GPT_geometric_sequence_value_sum_l994_99442


namespace NUMINAMATH_GPT_total_afternoon_evening_emails_l994_99491

-- Definitions based on conditions
def afternoon_emails : ℕ := 5
def evening_emails : ℕ := 8

-- Statement to be proven
theorem total_afternoon_evening_emails : afternoon_emails + evening_emails = 13 :=
by 
  sorry

end NUMINAMATH_GPT_total_afternoon_evening_emails_l994_99491


namespace NUMINAMATH_GPT_rational_roots_of_polynomial_l994_99467

theorem rational_roots_of_polynomial :
  { x : ℚ | (x + 1) * (x - (2 / 3)) * (x^2 - 2) = 0 } = {-1, 2 / 3} :=
by
  sorry

end NUMINAMATH_GPT_rational_roots_of_polynomial_l994_99467


namespace NUMINAMATH_GPT_no_common_period_l994_99400

theorem no_common_period (g h : ℝ → ℝ) 
  (hg : ∀ x, g (x + 2) = g x) 
  (hh : ∀ x, h (x + π/2) = h x) : 
  ¬ (∃ T > 0, ∀ x, g (x + T) + h (x + T) = g x + h x) :=
sorry

end NUMINAMATH_GPT_no_common_period_l994_99400


namespace NUMINAMATH_GPT_arrangement_count_l994_99464

-- Definitions from the conditions
def people : Nat := 5
def valid_positions_for_A : Finset Nat := Finset.range 5 \ {0, 4}

-- The theorem that states the question equals the correct answer given the conditions
theorem arrangement_count (A_positions : Finset Nat := valid_positions_for_A) : 
  ∃ (total_arrangements : Nat), total_arrangements = 72 :=
by
  -- Placeholder for the proof
  sorry

end NUMINAMATH_GPT_arrangement_count_l994_99464


namespace NUMINAMATH_GPT_complex_magnitude_add_reciprocals_l994_99497

open Complex

theorem complex_magnitude_add_reciprocals
  (z w : ℂ)
  (hz : Complex.abs z = 2)
  (hw : Complex.abs w = 4)
  (hz_plus_w : Complex.abs (z + w) = 6) :
  Complex.abs (1 / z + 1 / w) = 3 / 4 := by
  sorry

end NUMINAMATH_GPT_complex_magnitude_add_reciprocals_l994_99497


namespace NUMINAMATH_GPT_sin_60_eq_sqrt3_div_2_l994_99489

theorem sin_60_eq_sqrt3_div_2 : Real.sin (Real.pi / 3) = Real.sqrt 3 / 2 := 
by
  sorry

end NUMINAMATH_GPT_sin_60_eq_sqrt3_div_2_l994_99489


namespace NUMINAMATH_GPT_area_of_parallelogram_l994_99487

theorem area_of_parallelogram (base height : ℝ) (h_base : base = 12) (h_height : height = 8) :
  base * height = 96 :=
by
  rw [h_base, h_height]
  norm_num

end NUMINAMATH_GPT_area_of_parallelogram_l994_99487


namespace NUMINAMATH_GPT_sequences_cover_naturals_without_repetition_l994_99422

theorem sequences_cover_naturals_without_repetition
  (x y : Real) 
  (hx : Irrational x) 
  (hy : Irrational y) 
  (hxy : 1/x + 1/y = 1) :
  (∀ n : ℕ, ∃! k : ℕ, (⌊k * x⌋ = n) ∨ (⌊k * y⌋ = n)) :=
sorry

end NUMINAMATH_GPT_sequences_cover_naturals_without_repetition_l994_99422


namespace NUMINAMATH_GPT_gcd_multiples_l994_99475

theorem gcd_multiples (p q : ℕ) (hp : p > 0) (hq : q > 0) (h : Nat.gcd p q = 15) : Nat.gcd (8 * p) (18 * q) = 30 :=
by sorry

end NUMINAMATH_GPT_gcd_multiples_l994_99475


namespace NUMINAMATH_GPT_female_with_advanced_degrees_l994_99459

theorem female_with_advanced_degrees
  (total_employees : ℕ)
  (total_females : ℕ)
  (total_employees_with_advanced_degrees : ℕ)
  (total_employees_with_college_degree_only : ℕ)
  (total_males_with_college_degree_only : ℕ)
  (h1 : total_employees = 180)
  (h2 : total_females = 110)
  (h3 : total_employees_with_advanced_degrees = 90)
  (h4 : total_employees_with_college_degree_only = 90)
  (h5 : total_males_with_college_degree_only = 35) :
  ∃ (female_with_advanced_degrees : ℕ), female_with_advanced_degrees = 55 :=
by
  -- the proof goes here
  sorry

end NUMINAMATH_GPT_female_with_advanced_degrees_l994_99459


namespace NUMINAMATH_GPT_max_value_of_f_l994_99429

theorem max_value_of_f :
  ∀ (x : ℝ), -5 ≤ x ∧ x ≤ 13 → ∃ (y : ℝ), y = x - 5 ∧ y ≤ 8 ∧ y >= -10 ∧ 
  (∀ (z : ℝ), z = (x - 5) → z ≤ 8) := 
by
  sorry

end NUMINAMATH_GPT_max_value_of_f_l994_99429


namespace NUMINAMATH_GPT_additional_boys_went_down_slide_l994_99485

theorem additional_boys_went_down_slide (initial_boys total_boys additional_boys : ℕ) (h1 : initial_boys = 22) (h2 : total_boys = 35) : additional_boys = 13 :=
by {
    -- Proof body will be here
    sorry
}

end NUMINAMATH_GPT_additional_boys_went_down_slide_l994_99485


namespace NUMINAMATH_GPT_number_of_trees_l994_99409

theorem number_of_trees (l d : ℕ) (h_l : l = 441) (h_d : d = 21) : (l / d) + 1 = 22 :=
by
  sorry

end NUMINAMATH_GPT_number_of_trees_l994_99409


namespace NUMINAMATH_GPT_michael_watermelon_weight_l994_99410

theorem michael_watermelon_weight (m c j : ℝ) (h1 : c = 3 * m) (h2 : j = c / 2) (h3 : j = 12) : m = 8 :=
by
  sorry

end NUMINAMATH_GPT_michael_watermelon_weight_l994_99410


namespace NUMINAMATH_GPT_find_AG_l994_99476

-- Defining constants and variables
variables (DE EC AD BC FB AG : ℚ)
variables (BC_def : BC = (1 / 3) * AD)
variables (FB_def : FB = (2 / 3) * AD)
variables (DE_val : DE = 8)
variables (EC_val : EC = 6)
variables (sum_AD : BC + FB = AD)

-- The theorem statement
theorem find_AG : AG = 56 / 9 :=
by
  -- Placeholder for the proof
  sorry

end NUMINAMATH_GPT_find_AG_l994_99476


namespace NUMINAMATH_GPT_seats_needed_l994_99402

theorem seats_needed (children seats_per_seat : ℕ) (h1 : children = 58) (h2 : seats_per_seat = 2) : children / seats_per_seat = 29 :=
by sorry

end NUMINAMATH_GPT_seats_needed_l994_99402


namespace NUMINAMATH_GPT_mia_has_largest_final_value_l994_99465

def daniel_final : ℕ := (12 * 2 - 3 + 5)
def mia_final : ℕ := ((15 - 2) * 2 + 3)
def carlos_final : ℕ := (13 * 2 - 4 + 6)

theorem mia_has_largest_final_value : mia_final > daniel_final ∧ mia_final > carlos_final := by
  sorry

end NUMINAMATH_GPT_mia_has_largest_final_value_l994_99465


namespace NUMINAMATH_GPT_distinct_positive_least_sum_seven_integers_prod_2016_l994_99419

theorem distinct_positive_least_sum_seven_integers_prod_2016 :
  ∃ (n1 n2 n3 n4 n5 n6 n7 : ℕ),
    n1 < n2 ∧ n2 < n3 ∧ n3 < n4 ∧ n4 < n5 ∧ n5 < n6 ∧ n6 < n7 ∧
    (n1 * n2 * n3 * n4 * n5 * n6 * n7) % 2016 = 0 ∧
    n1 + n2 + n3 + n4 + n5 + n6 + n7 = 31 :=
sorry

end NUMINAMATH_GPT_distinct_positive_least_sum_seven_integers_prod_2016_l994_99419


namespace NUMINAMATH_GPT_product_lcm_gcd_l994_99450

theorem product_lcm_gcd (a b : ℕ) (h_a : a = 24) (h_b : b = 36):
  Nat.lcm a b * Nat.gcd a b = 864 :=
by
  rw [h_a, h_b]
  sorry

end NUMINAMATH_GPT_product_lcm_gcd_l994_99450


namespace NUMINAMATH_GPT_terms_are_equal_l994_99407

theorem terms_are_equal (n : ℕ) (a b : ℕ → ℕ)
  (h_n : n ≥ 2018)
  (h_a : ∀ i j : ℕ, i ≠ j → a i ≠ a j)
  (h_b : ∀ i j : ℕ, i ≠ j → b i ≠ b j)
  (h_a_pos : ∀ i : ℕ, 1 ≤ i ∧ i ≤ n → a i > 0)
  (h_b_pos : ∀ i : ℕ, 1 ≤ i ∧ i ≤ n → b i > 0)
  (h_a_le : ∀ i : ℕ, 1 ≤ i ∧ i ≤ n → a i ≤ 5 * n)
  (h_b_le : ∀ i : ℕ, 1 ≤ i ∧ i ≤ n → b i ≤ 5 * n)
  (h_arith : ∀ i j : ℕ, 1 ≤ i ∧ i ≤ n ∧ 1 ≤ j ∧ j ≤ n → (a j * b i - a i * b j) * (j - i) = 0):
  ∀ i j : ℕ, 1 ≤ i ∧ i ≤ n ∧ 1 ≤ j ∧ j ≤ n → a i * b j = a j * b i :=
by
  sorry

end NUMINAMATH_GPT_terms_are_equal_l994_99407


namespace NUMINAMATH_GPT_max_fraction_l994_99483

theorem max_fraction (x y : ℝ) (hx : -3 ≤ x ∧ x ≤ -1) (hy : 3 ≤ y ∧ y ≤ 6) :
  1 + y / x ≤ -2 :=
sorry

end NUMINAMATH_GPT_max_fraction_l994_99483


namespace NUMINAMATH_GPT_cocktail_cost_l994_99437

noncomputable def costPerLitreCocktail (cost_mixed_fruit_juice : ℝ) (cost_acai_juice : ℝ) (volume_mixed_fruit : ℝ) (volume_acai : ℝ) : ℝ :=
  let total_cost := cost_mixed_fruit_juice * volume_mixed_fruit + cost_acai_juice * volume_acai
  let total_volume := volume_mixed_fruit + volume_acai
  total_cost / total_volume

theorem cocktail_cost : costPerLitreCocktail 262.85 3104.35 32 21.333333333333332 = 1399.99 :=
  by
    sorry

end NUMINAMATH_GPT_cocktail_cost_l994_99437


namespace NUMINAMATH_GPT_whale_crossing_time_l994_99499

theorem whale_crossing_time
  (speed_fast : ℝ)
  (speed_slow : ℝ)
  (length_slow : ℝ)
  (h_fast : speed_fast = 18)
  (h_slow : speed_slow = 15)
  (h_length : length_slow = 45) :
  (length_slow / (speed_fast - speed_slow) = 15) :=
by
  sorry

end NUMINAMATH_GPT_whale_crossing_time_l994_99499


namespace NUMINAMATH_GPT_simplify_expression_l994_99480
theorem simplify_expression (c : ℝ) : 
    (3 * c + 6 - 6 * c) / 3 = -c + 2 := 
by 
    sorry

end NUMINAMATH_GPT_simplify_expression_l994_99480


namespace NUMINAMATH_GPT_sum_of_repeating_decimals_l994_99403

-- Define the two repeating decimals
def repeating_decimal_0_2 : ℚ := 2 / 9
def repeating_decimal_0_03 : ℚ := 1 / 33

-- Define the problem as a proof statement
theorem sum_of_repeating_decimals : repeating_decimal_0_2 + repeating_decimal_0_03 = 25 / 99 := 
by sorry

end NUMINAMATH_GPT_sum_of_repeating_decimals_l994_99403


namespace NUMINAMATH_GPT_triangle_inequality_proof_l994_99424

theorem triangle_inequality_proof (a b c : ℝ) (PA QA PB QB PC QC : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (hpa : PA ≥ 0) (hqa : QA ≥ 0) (hpb : PB ≥ 0) (hqb : QB ≥ 0) 
  (hpc : PC ≥ 0) (hqc : QC ≥ 0):
  a * PA * QA + b * PB * QB + c * PC * QC ≥ a * b * c := 
sorry

end NUMINAMATH_GPT_triangle_inequality_proof_l994_99424


namespace NUMINAMATH_GPT_candy_per_day_eq_eight_l994_99404

def candy_received_from_neighbors : ℝ := 11.0
def candy_received_from_sister : ℝ := 5.0
def days_candy_lasted : ℝ := 2.0

theorem candy_per_day_eq_eight :
  (candy_received_from_neighbors + candy_received_from_sister) / days_candy_lasted = 8.0 :=
by
  sorry

end NUMINAMATH_GPT_candy_per_day_eq_eight_l994_99404


namespace NUMINAMATH_GPT_sum_of_repeating_decimals_l994_99473

def repeatingDecimalToFraction (str : String) (base : ℕ) : ℚ := sorry

noncomputable def expressSumAsFraction : ℚ :=
  let x := repeatingDecimalToFraction "2" 10
  let y := repeatingDecimalToFraction "03" 100
  let z := repeatingDecimalToFraction "0004" 10000
  x + y + z

theorem sum_of_repeating_decimals : expressSumAsFraction = 843 / 3333 := by
  sorry

end NUMINAMATH_GPT_sum_of_repeating_decimals_l994_99473


namespace NUMINAMATH_GPT_range_of_x_l994_99451

theorem range_of_x (x : ℝ) : x + 2 ≥ 0 ∧ x - 3 ≠ 0 → x ≥ -2 ∧ x ≠ 3 :=
by
  sorry

end NUMINAMATH_GPT_range_of_x_l994_99451


namespace NUMINAMATH_GPT_yen_exchange_rate_l994_99445

theorem yen_exchange_rate (yen_per_dollar : ℕ) (dollars : ℕ) (y : ℕ) (h1 : yen_per_dollar = 120) (h2 : dollars = 10) : y = 1200 :=
by
  have h3 : y = yen_per_dollar * dollars := by sorry
  rw [h1, h2] at h3
  exact h3

end NUMINAMATH_GPT_yen_exchange_rate_l994_99445


namespace NUMINAMATH_GPT_smallest_positive_integer_l994_99414

theorem smallest_positive_integer 
  (x : ℤ) (h1 : x % 6 = 3) (h2 : x % 8 = 2) : x = 33 :=
sorry

end NUMINAMATH_GPT_smallest_positive_integer_l994_99414


namespace NUMINAMATH_GPT_find_a_l994_99454

noncomputable def A : Set ℝ := {x | x^2 - x - 2 = 0}
noncomputable def B (a : ℝ) : Set ℝ := {x | a * x - 1 = 0}
def is_solution (a : ℝ) : Prop := ∀ b, b ∈ B a → b ∈ A

theorem find_a (a : ℝ) : (B a ⊆ A) → a = 0 ∨ a = -1 ∨ a = 1/2 := by
  intro h
  sorry

end NUMINAMATH_GPT_find_a_l994_99454


namespace NUMINAMATH_GPT_Rajesh_work_completion_time_l994_99471

-- Definitions based on conditions in a)
def Mahesh_rate := 1 / 60 -- Mahesh's rate of work (work per day)
def Mahesh_work := 20 * Mahesh_rate -- Work completed by Mahesh in 20 days
def Rajesh_time_to_complete_remaining_work := 30 -- Rajesh time to complete remaining work (days)
def Remaining_work := 1 - Mahesh_work -- Remaining work after Mahesh's contribution

-- Statement that needs to be proved
theorem Rajesh_work_completion_time :
  (Rajesh_time_to_complete_remaining_work : ℝ) * (1 / Remaining_work) = 45 :=
sorry

end NUMINAMATH_GPT_Rajesh_work_completion_time_l994_99471


namespace NUMINAMATH_GPT_adam_students_in_10_years_l994_99440

-- Define the conditions
def teaches_per_year : Nat := 50
def first_year_students : Nat := 40
def years_teaching : Nat := 10

-- Define the total number of students Adam will teach in 10 years
def total_students (first_year: Nat) (rest_years: Nat) (students_per_year: Nat) : Nat :=
  first_year + (rest_years * students_per_year)

-- State the theorem
theorem adam_students_in_10_years :
  total_students first_year_students (years_teaching - 1) teaches_per_year = 490 :=
by
  sorry

end NUMINAMATH_GPT_adam_students_in_10_years_l994_99440


namespace NUMINAMATH_GPT_equalSumSeqDefinition_l994_99486

def isEqualSumSeq (s : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, n > 0 → s (n - 1) + s n = s (n + 1)

theorem equalSumSeqDefinition (s : ℕ → ℝ) :
  isEqualSumSeq s ↔ 
  ∀ n : ℕ, n > 0 → s n = s (n - 1) + s (n + 1) :=
by
  sorry

end NUMINAMATH_GPT_equalSumSeqDefinition_l994_99486


namespace NUMINAMATH_GPT_people_present_l994_99444

-- Define the number of parents, pupils, and teachers as constants
def p := 73
def s := 724
def t := 744

-- The theorem to prove the total number of people present
theorem people_present : p + s + t = 1541 := 
by
  -- Proof is inserted here
  sorry

end NUMINAMATH_GPT_people_present_l994_99444


namespace NUMINAMATH_GPT_students_per_group_l994_99415

-- Defining the conditions
def total_students : ℕ := 256
def number_of_teachers : ℕ := 8

-- The statement to prove
theorem students_per_group :
  total_students / number_of_teachers = 32 :=
by
  sorry

end NUMINAMATH_GPT_students_per_group_l994_99415


namespace NUMINAMATH_GPT_rob_total_cards_l994_99490

variables (r r_d j_d : ℕ)

-- Definitions of conditions
def condition1 : Prop := r_d = r / 3
def condition2 : Prop := j_d = 5 * r_d
def condition3 : Prop := j_d = 40

-- Problem Statement
theorem rob_total_cards (h1 : condition1 r r_d)
                        (h2 : condition2 r_d j_d)
                        (h3 : condition3 j_d) :
  r = 24 :=
by
  sorry

end NUMINAMATH_GPT_rob_total_cards_l994_99490


namespace NUMINAMATH_GPT_fraction_difference_eq_l994_99405

theorem fraction_difference_eq (a b : ℝ) (ha : 0 < a) (hb : 0 < b)
  (h : a / (1 + a) + b / (1 + b) = 1) :
  a / (1 + b^2) - b / (1 + a^2) = a - b :=
sorry

end NUMINAMATH_GPT_fraction_difference_eq_l994_99405


namespace NUMINAMATH_GPT_number_of_outfits_l994_99466

theorem number_of_outfits : (5 * 4 * 6 * 3) = 360 := by
  sorry

end NUMINAMATH_GPT_number_of_outfits_l994_99466


namespace NUMINAMATH_GPT_tangent_line_through_origin_eq_ex_l994_99457

theorem tangent_line_through_origin_eq_ex :
  ∃ (k : ℝ), (∀ x : ℝ, y = e^x) ∧ (∃ x₀ : ℝ, y - e^x₀ = e^x₀ * (x - x₀)) ∧ 
  (y = k * x) :=
sorry

end NUMINAMATH_GPT_tangent_line_through_origin_eq_ex_l994_99457


namespace NUMINAMATH_GPT_smallest_abcd_value_l994_99408

theorem smallest_abcd_value (A B C D : ℕ) (h1 : A ≠ B) (h2 : 1 ≤ A) (h3 : A ≤ 9) (h4 : 0 ≤ B) 
                            (h5 : B ≤ 9) (h6 : 1 ≤ C) (h7 : C ≤ 9) (h8 : 1 ≤ D) (h9 : D ≤ 9)
                            (h10 : 10 * A * A + A * B = 1000 * A + 100 * B + 10 * C + D)
                            (h11 : A ≠ C) (h12 : A ≠ D) (h13 : B ≠ C) (h14 : B ≠ D) (h15 : C ≠ D) :
  1000 * A + 100 * B + 10 * C + D = 2046 :=
sorry

end NUMINAMATH_GPT_smallest_abcd_value_l994_99408


namespace NUMINAMATH_GPT_sofia_running_time_l994_99492

theorem sofia_running_time :
  let distance_first_section := 100 -- meters
  let speed_first_section := 5 -- meters per second
  let distance_second_section := 300 -- meters
  let speed_second_section := 4 -- meters per second
  let num_laps := 6
  let time_first_section := distance_first_section / speed_first_section -- in seconds
  let time_second_section := distance_second_section / speed_second_section -- in seconds
  let time_per_lap := time_first_section + time_second_section -- in seconds
  let total_time_seconds := num_laps * time_per_lap -- in seconds
  let total_time_minutes := total_time_seconds / 60 -- integer division for minutes
  let remaining_seconds := total_time_seconds % 60 -- modulo for remaining seconds
  total_time_minutes = 9 ∧ remaining_seconds = 30 := 
  by
  sorry

end NUMINAMATH_GPT_sofia_running_time_l994_99492


namespace NUMINAMATH_GPT_equilateral_triangle_isosceles_triangle_l994_99438

noncomputable def is_equilateral (a b c : ℝ) : Prop :=
  a = b ∧ b = c

noncomputable def is_isosceles (a b c : ℝ) : Prop :=
  a = b ∨ b = c ∨ a = c

theorem equilateral_triangle (a b c : ℝ) (h : abs (a - b) + abs (b - c) = 0) : is_equilateral a b c :=
  sorry

theorem isosceles_triangle (a b c : ℝ) (h : (a - b) * (b - c) = 0) : is_isosceles a b c :=
  sorry

end NUMINAMATH_GPT_equilateral_triangle_isosceles_triangle_l994_99438


namespace NUMINAMATH_GPT_least_number_to_add_to_246835_l994_99417

-- Define relevant conditions and computations
def lcm_of_169_and_289 : ℕ := Nat.lcm 169 289
def remainder_246835_mod_lcm : ℕ := 246835 % lcm_of_169_and_289
def least_number_to_add : ℕ := lcm_of_169_and_289 - remainder_246835_mod_lcm

-- The theorem statement
theorem least_number_to_add_to_246835 : least_number_to_add = 52 :=
by
  sorry

end NUMINAMATH_GPT_least_number_to_add_to_246835_l994_99417


namespace NUMINAMATH_GPT_find_n_l994_99426

variable (P : ℕ → ℝ) (n : ℕ)

def polynomialDegree (P : ℕ → ℝ) (deg : ℕ) : Prop :=
  ∀ k, k > deg → P k = 0

def zeroValues (P : ℕ → ℝ) (n : ℕ) : Prop :=
  ∀ i, i ∈ (List.range (2 * n + 1)).map (λ k => 2 * k) → P i = 0

def twoValues (P : ℕ → ℝ) (n : ℕ) : Prop :=
  ∀ i, i ∈ (List.range (2 * n)).map (λ k => 2 * k + 1) → P i = 2

def specialValue (P : ℕ → ℝ) (n : ℕ) : Prop :=
  P (2 * n + 1) = -30

theorem find_n :
  (∃ n, polynomialDegree P (2 * n) ∧ zeroValues P n ∧ twoValues P n ∧ specialValue P n) →
  n = 2 :=
by
  sorry

end NUMINAMATH_GPT_find_n_l994_99426


namespace NUMINAMATH_GPT_base6_problem_l994_99416

theorem base6_problem
  (x y : ℕ)
  (h1 : 453 = 2 * x * 10 + y) -- Constraint from base-6 to base-10 conversion
  (h2 : 0 ≤ x ∧ x ≤ 9) -- x is a base-10 digit
  (h3 : 0 ≤ y ∧ y ≤ 9) -- y is a base-10 digit
  (h4 : 4 * 6^2 + 5 * 6 + 3 = 177) -- Conversion result for 453_6
  (h5 : 2 * x * 10 + y = 177) -- Conversion from condition
  (hx : x = 7) -- x value from solution
  (hy : y = 7) -- y value from solution
  : (x * y) / 10 = 49 / 10 := 
by 
  sorry

end NUMINAMATH_GPT_base6_problem_l994_99416


namespace NUMINAMATH_GPT_remainder_g_x12_div_g_x_l994_99495

-- Define the polynomial g
noncomputable def g (x : ℂ) : ℂ := x^5 + x^4 + x^3 + x^2 + x + 1

-- Proving the remainder when g(x^12) is divided by g(x) is 6
theorem remainder_g_x12_div_g_x : 
  (g (x^12) % g x) = 6 :=
sorry

end NUMINAMATH_GPT_remainder_g_x12_div_g_x_l994_99495


namespace NUMINAMATH_GPT_matrix_cubic_l994_99472

noncomputable def matrix_a : Matrix (Fin 2) (Fin 2) ℤ := ![
  ![2, -1],
  ![1, 1]
]

theorem matrix_cubic :
  matrix_a ^ 3 = ![
    ![3, -6],
    ![6, -3]
  ] := by
  sorry

end NUMINAMATH_GPT_matrix_cubic_l994_99472


namespace NUMINAMATH_GPT_calculate_seasons_l994_99412

theorem calculate_seasons :
  ∀ (episodes_per_season : ℕ) (episodes_per_day : ℕ) (days : ℕ),
  episodes_per_season = 20 →
  episodes_per_day = 2 →
  days = 30 →
  (episodes_per_day * days) / episodes_per_season = 3 :=
by
  intros episodes_per_season episodes_per_day days h_eps h_epd h_d
  sorry

end NUMINAMATH_GPT_calculate_seasons_l994_99412


namespace NUMINAMATH_GPT_yellow_highlighters_count_l994_99423

theorem yellow_highlighters_count 
  (Y : ℕ) 
  (pink_highlighters : ℕ := Y + 7) 
  (blue_highlighters : ℕ := Y + 12) 
  (total_highlighters : ℕ := Y + pink_highlighters + blue_highlighters) : 
  total_highlighters = 40 → Y = 7 :=
by
  sorry

end NUMINAMATH_GPT_yellow_highlighters_count_l994_99423


namespace NUMINAMATH_GPT_pencil_case_costs_l994_99463

variable {x y : ℝ}

theorem pencil_case_costs :
  (2 * x + 3 * y = 108) ∧ (5 * x = 6 * y) → 
  (x = 24) ∧ (y = 20) :=
by
  intros h
  obtain ⟨h1, h2⟩ := h
  sorry

end NUMINAMATH_GPT_pencil_case_costs_l994_99463


namespace NUMINAMATH_GPT_infinite_pairs_exists_l994_99484

noncomputable def exists_infinite_pairs : Prop :=
  ∃ (a b : ℕ), (a + b ∣ a * b + 1) ∧ (a - b ∣ a * b - 1) ∧ b > 1 ∧ a > b * Real.sqrt 3 - 1

theorem infinite_pairs_exists : ∃ (count : ℕ) (a b : ℕ), ∀ n < count, exists_infinite_pairs :=
sorry

end NUMINAMATH_GPT_infinite_pairs_exists_l994_99484


namespace NUMINAMATH_GPT_fraction_Renz_Miles_l994_99432

-- Given definitions and conditions
def Mitch_macarons : ℕ := 20
def Joshua_diff : ℕ := 6
def kids : ℕ := 68
def macarons_per_kid : ℕ := 2
def total_macarons_given : ℕ := kids * macarons_per_kid
def Joshua_macarons : ℕ := Mitch_macarons + Joshua_diff
def Miles_macarons : ℕ := 2 * Joshua_macarons
def Mitch_Joshua_Miles_macarons : ℕ := Mitch_macarons + Joshua_macarons + Miles_macarons
def Renz_macarons : ℕ := total_macarons_given - Mitch_Joshua_Miles_macarons

-- The theorem to prove
theorem fraction_Renz_Miles : (Renz_macarons : ℚ) / (Miles_macarons : ℚ) = 19 / 26 :=
by
  sorry

end NUMINAMATH_GPT_fraction_Renz_Miles_l994_99432


namespace NUMINAMATH_GPT_students_on_right_side_l994_99456

-- Define the total number of students and the number of students on the left side
def total_students : ℕ := 63
def left_students : ℕ := 36

-- Define the number of students on the right side using subtraction
def right_students (total_students left_students : ℕ) : ℕ := total_students - left_students

-- Theorem: Prove that the number of students on the right side is 27
theorem students_on_right_side : right_students total_students left_students = 27 := by
  sorry

end NUMINAMATH_GPT_students_on_right_side_l994_99456


namespace NUMINAMATH_GPT_min_value_eval_l994_99462

noncomputable def min_value_expr (x y : ℝ) := 
  (x + 1/y) * (x + 1/y - 100) + (y + 1/x) * (y + 1/x - 100)

theorem min_value_eval (x y : ℝ) (hx : 0 < x) (hy : 0 < y) : 
  x = y → min_value_expr x y = -2500 :=
by
  intros hxy
  -- Insert proof steps here
  sorry

end NUMINAMATH_GPT_min_value_eval_l994_99462


namespace NUMINAMATH_GPT_angle_measure_l994_99418

theorem angle_measure (A B C : ℝ) (h1 : A = B) (h2 : A + B = 110 ∨ (A = 180 - 110)) :
  A = 70 ∨ A = 55 := by
  sorry

end NUMINAMATH_GPT_angle_measure_l994_99418


namespace NUMINAMATH_GPT_option_B_correct_l994_99441

theorem option_B_correct : 1 ∈ ({0, 1} : Set ℕ) := 
by
  sorry

end NUMINAMATH_GPT_option_B_correct_l994_99441


namespace NUMINAMATH_GPT_Diane_bakes_160_gingerbreads_l994_99470

-- Definitions
def trays1Count : Nat := 4
def gingerbreads1PerTray : Nat := 25
def trays2Count : Nat := 3
def gingerbreads2PerTray : Nat := 20

def totalGingerbreads : Nat :=
  (trays1Count * gingerbreads1PerTray) + (trays2Count * gingerbreads2PerTray)

-- Problem statement
theorem Diane_bakes_160_gingerbreads :
  totalGingerbreads = 160 := by
  sorry

end NUMINAMATH_GPT_Diane_bakes_160_gingerbreads_l994_99470


namespace NUMINAMATH_GPT_find_other_number_l994_99461

theorem find_other_number (x : ℕ) (h : x + 42 = 96) : x = 54 :=
by {
  sorry
}

end NUMINAMATH_GPT_find_other_number_l994_99461


namespace NUMINAMATH_GPT_number_of_ordered_pairs_l994_99493

theorem number_of_ordered_pairs (h : ∀ (m n : ℕ), 0 < m → 0 < n → 6/m + 3/n = 1 → true) : 
∃! (s : Finset (ℕ × ℕ)), s.card = 4 ∧ ∀ (x : ℕ × ℕ), x ∈ s → 0 < x.1 ∧ 0 < x.2 ∧ 6 / ↑x.1 + 3 / ↑x.2 = 1 :=
by
-- Sorry, skipping the proof
  sorry

end NUMINAMATH_GPT_number_of_ordered_pairs_l994_99493


namespace NUMINAMATH_GPT_minimal_coach_handshakes_l994_99460

theorem minimal_coach_handshakes (n k1 k2 : ℕ) (h1 : k1 < n) (h2 : k2 < n)
  (hn : (n * (n - 1)) / 2 + k1 + k2 = 300) : k1 + k2 = 0 := by
  sorry

end NUMINAMATH_GPT_minimal_coach_handshakes_l994_99460


namespace NUMINAMATH_GPT_inheritance_amount_l994_99430

def is_inheritance_amount (x : ℝ) : Prop :=
  let federal_tax := 0.25 * x
  let remaining_after_fed := x - federal_tax
  let state_tax := 0.12 * remaining_after_fed
  let total_tax_paid := federal_tax + state_tax
  total_tax_paid = 15600

theorem inheritance_amount : 
  ∃ x, is_inheritance_amount x ∧ x = 45882 := 
by
  sorry

end NUMINAMATH_GPT_inheritance_amount_l994_99430


namespace NUMINAMATH_GPT_candidate_p_wage_difference_l994_99406

theorem candidate_p_wage_difference
  (P Q : ℝ)    -- Candidate p's hourly wage is P, Candidate q's hourly wage is Q
  (H : ℝ)      -- Candidate p's working hours
  (total_payment : ℝ)
  (wage_ratio : P = 1.5 * Q)  -- Candidate p is paid 50% more per hour than candidate q
  (hours_diff : Q * (H + 10) = total_payment)  -- Candidate q's total payment equation
  (candidate_q_payment : Q * (H + 10) = 480)   -- total payment for candidate q
  (candidate_p_payment : 1.5 * Q * H = 480)    -- total payment for candidate p
  : P - Q = 8 := sorry

end NUMINAMATH_GPT_candidate_p_wage_difference_l994_99406


namespace NUMINAMATH_GPT_complex_ratio_real_l994_99481

theorem complex_ratio_real (a b : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0)
  (h3 : ∃ z : ℂ, z = a + b * Complex.I ∧ (z * (1 - 2 * Complex.I)).im = 0) :
  a / b = 1 / 2 :=
sorry

end NUMINAMATH_GPT_complex_ratio_real_l994_99481


namespace NUMINAMATH_GPT_runs_by_running_percentage_l994_99488

def total_runs := 125
def boundaries := 5
def boundary_runs := boundaries * 4
def sixes := 5
def sixes_runs := sixes * 6
def runs_by_running := total_runs - (boundary_runs + sixes_runs)
def percentage_runs_by_running := (runs_by_running : ℚ) / total_runs * 100

theorem runs_by_running_percentage :
  percentage_runs_by_running = 60 := by sorry

end NUMINAMATH_GPT_runs_by_running_percentage_l994_99488


namespace NUMINAMATH_GPT_additional_water_added_l994_99448

variable (M W : ℕ)

theorem additional_water_added (M W : ℕ) (initial_mix : ℕ) (initial_ratio : ℕ × ℕ) (new_ratio : ℚ) :
  initial_mix = 45 →
  initial_ratio = (4, 1) →
  new_ratio = 4 / 3 →
  (4 / 5) * initial_mix = M →
  (1 / 5) * initial_mix + W = 3 / 4 * M →
  W = 18 :=
by
  sorry

end NUMINAMATH_GPT_additional_water_added_l994_99448


namespace NUMINAMATH_GPT_sophomores_more_than_first_graders_l994_99421

def total_students : ℕ := 95
def first_graders : ℕ := 32
def second_graders : ℕ := total_students - first_graders

theorem sophomores_more_than_first_graders : second_graders - first_graders = 31 := by
  sorry

end NUMINAMATH_GPT_sophomores_more_than_first_graders_l994_99421


namespace NUMINAMATH_GPT_loss_per_meter_calculation_l994_99447

/-- Define the given constants and parameters. --/
def total_meters : ℕ := 600
def selling_price : ℕ := 18000
def cost_price_per_meter : ℕ := 35

/-- Now we define the total cost price, total loss and loss per meter --/
def total_cost_price : ℕ := cost_price_per_meter * total_meters
def total_loss : ℕ := total_cost_price - selling_price
def loss_per_meter : ℕ := total_loss / total_meters

/-- State the theorem we need to prove. --/
theorem loss_per_meter_calculation : loss_per_meter = 5 :=
by
  sorry

end NUMINAMATH_GPT_loss_per_meter_calculation_l994_99447


namespace NUMINAMATH_GPT_calculate_hardcover_volumes_l994_99413

theorem calculate_hardcover_volumes (h p : ℕ) 
  (h_total_volumes : h + p = 12)
  (h_cost_equation : 27 * h + 16 * p = 284)
  (h_p_relation : p = 12 - h) : h = 8 :=
by
  sorry

end NUMINAMATH_GPT_calculate_hardcover_volumes_l994_99413


namespace NUMINAMATH_GPT_sum_of_integers_is_106_l994_99469

theorem sum_of_integers_is_106 (n m : ℕ) 
  (h1: n * (n + 1) = 1320) 
  (h2: m * (m + 1) * (m + 2) = 1320) : 
  n + (n + 1) + m + (m + 1) + (m + 2) = 106 :=
  sorry

end NUMINAMATH_GPT_sum_of_integers_is_106_l994_99469


namespace NUMINAMATH_GPT_cos_sub_sin_alpha_l994_99446

theorem cos_sub_sin_alpha (alpha : ℝ) (h1 : π / 4 < alpha) (h2 : alpha < π / 2)
    (h3 : Real.sin (2 * alpha) = 24 / 25) : Real.cos alpha - Real.sin alpha = -1 / 5 :=
by
  sorry

end NUMINAMATH_GPT_cos_sub_sin_alpha_l994_99446
