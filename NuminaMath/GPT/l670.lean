import Mathlib

namespace NUMINAMATH_GPT_find_value_l670_67087

theorem find_value (x y : ℚ) (hx : x = 5 / 7) (hy : y = 7 / 5) :
  (1 / 3 * x^8 * y^9 + 1 / 7) = 64 / 105 := by
  sorry

end NUMINAMATH_GPT_find_value_l670_67087


namespace NUMINAMATH_GPT_real_number_set_condition_l670_67058

theorem real_number_set_condition (x : ℝ) :
  (x ≠ 1) ∧ (x^2 - x ≠ 1) ∧ (x^2 - x ≠ x) →
  x ≠ 0 ∧ x ≠ 1 ∧ x ≠ 2 ∧ x ≠ (1 + Real.sqrt 5) / 2 ∧ x ≠ (1 - Real.sqrt 5) / 2 := 
by
  sorry

end NUMINAMATH_GPT_real_number_set_condition_l670_67058


namespace NUMINAMATH_GPT_value_of_number_l670_67053

theorem value_of_number (x : ℤ) (number : ℚ) (h₁ : x = 32) (h₂ : 35 - (23 - (15 - x)) = 12 * number / (1/2)) : number = -5/6 :=
by
  sorry

end NUMINAMATH_GPT_value_of_number_l670_67053


namespace NUMINAMATH_GPT_vasya_numbers_l670_67091

theorem vasya_numbers :
  ∃ x y : ℝ, x + y = x * y ∧ x + y = x / y ∧ x = 1/2 ∧ y = -1 :=
by
  sorry

end NUMINAMATH_GPT_vasya_numbers_l670_67091


namespace NUMINAMATH_GPT_ophelia_average_pay_l670_67069

theorem ophelia_average_pay : ∀ (n : ℕ), 
  (51 + 100 * (n - 1)) / n = 93 ↔ n = 7 :=
by
  sorry

end NUMINAMATH_GPT_ophelia_average_pay_l670_67069


namespace NUMINAMATH_GPT_smallest_integer_in_set_l670_67048

theorem smallest_integer_in_set :
  ∀ (n : ℤ), (n + 6 > 2 * ((n + (n+1) + (n+2) + (n+3) + (n+4) + (n+5) + (n+6)) / 7)) → n = -1 :=
by
  intros n h
  sorry

end NUMINAMATH_GPT_smallest_integer_in_set_l670_67048


namespace NUMINAMATH_GPT_hyperbola_eccentricity_range_l670_67024

theorem hyperbola_eccentricity_range
  (a b t : ℝ)
  (h_a_pos : a > 0)
  (h_b_pos : b > 0)
  (h_condition : a > b) :
  ∃ e : ℝ, e = Real.sqrt (1 + (b / a)^2) ∧ 1 < e ∧ e < Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_GPT_hyperbola_eccentricity_range_l670_67024


namespace NUMINAMATH_GPT_ratio_norm_lisa_l670_67005

-- Define the number of photos taken by each photographer.
variable (L M N : ℕ)

-- Given conditions
def norm_photos : Prop := N = 110
def photo_sum_condition : Prop := L + M = M + N - 60

-- Prove the ratio of Norm's photos to Lisa's photos.
theorem ratio_norm_lisa (h1 : norm_photos N) (h2 : photo_sum_condition L M N) : N / L = 11 / 5 := 
by
  sorry

end NUMINAMATH_GPT_ratio_norm_lisa_l670_67005


namespace NUMINAMATH_GPT_total_number_of_balls_l670_67059

theorem total_number_of_balls 
(b : ℕ) (P_blue : ℚ) (h1 : b = 8) (h2 : P_blue = 1/3) : 
  ∃ g : ℕ, b + g = 24 := by
  sorry

end NUMINAMATH_GPT_total_number_of_balls_l670_67059


namespace NUMINAMATH_GPT_liked_product_B_l670_67038

-- Define the conditions as assumptions
variables (X : ℝ)

-- Assumptions
axiom liked_both : 23 = 23
axiom liked_neither : 23 = 23

-- The main theorem that needs to be proven
theorem liked_product_B (X : ℝ) : ∃ Y : ℝ, Y = 100 - X :=
by sorry

end NUMINAMATH_GPT_liked_product_B_l670_67038


namespace NUMINAMATH_GPT_angle_between_bisectors_is_zero_l670_67071

-- Let's define the properties of the triangle and the required proof.

open Real

-- Define the side lengths of the isosceles triangle
def is_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ a + c > b ∧ b + c > a ∧ a > 0 ∧ b > 0 ∧ c > 0

def is_isosceles (a b c : ℝ) : Prop :=
  (a = b ∨ a = c ∨ b = c) ∧ is_triangle a b c

-- Define the specific isosceles triangle in the problem
def triangle_ABC : Prop := is_isosceles 5 5 6

-- Prove that the angle φ between the two lines is 0°
theorem angle_between_bisectors_is_zero :
  triangle_ABC → ∃ φ : ℝ, φ = 0 :=
by sorry

end NUMINAMATH_GPT_angle_between_bisectors_is_zero_l670_67071


namespace NUMINAMATH_GPT_drunk_drivers_count_l670_67052

theorem drunk_drivers_count (D S : ℕ) (h1 : S = 7 * D - 3) (h2 : D + S = 45) : D = 6 :=
by
  sorry

end NUMINAMATH_GPT_drunk_drivers_count_l670_67052


namespace NUMINAMATH_GPT_folded_paper_area_ratio_l670_67037

theorem folded_paper_area_ratio (s : ℝ) (h : s > 0) :
  let A := s^2
  let rectangle_area := (s / 2) * s
  let triangle_area := (1 / 2) * (s / 2) * s
  let folded_area := 4 * rectangle_area - triangle_area
  (folded_area / A) = 7 / 4 :=
by
  let A := s^2
  let rectangle_area := (s / 2) * s
  let triangle_area := (1 / 2) * (s / 2) * s
  let folded_area := 4 * rectangle_area - triangle_area
  show (folded_area / A) = 7 / 4
  sorry

end NUMINAMATH_GPT_folded_paper_area_ratio_l670_67037


namespace NUMINAMATH_GPT_total_songs_performed_l670_67033

theorem total_songs_performed :
  ∃ N : ℕ, 
  (∃ e d o : ℕ, 
     (e > 3 ∧ e < 9) ∧ (d > 3 ∧ d < 9) ∧ (o > 3 ∧ o < 9)
      ∧ N = (9 + 3 + e + d + o) / 4) ∧ N = 6 :=
sorry

end NUMINAMATH_GPT_total_songs_performed_l670_67033


namespace NUMINAMATH_GPT_necessary_but_not_sufficient_for_gt_one_l670_67017

variable (x : ℝ)

theorem necessary_but_not_sufficient_for_gt_one (h : x^2 > 1) : ¬(x^2 > 1 ↔ x > 1) ∧ (x > 1 → x^2 > 1) :=
by
  sorry

end NUMINAMATH_GPT_necessary_but_not_sufficient_for_gt_one_l670_67017


namespace NUMINAMATH_GPT_sum_or_difference_div_by_100_l670_67046

theorem sum_or_difference_div_by_100 (s : Finset ℤ) (h_card : s.card = 52) :
  ∃ (a b : ℤ), a ∈ s ∧ b ∈ s ∧ (a ≠ b) ∧ (100 ∣ (a + b) ∨ 100 ∣ (a - b)) :=
by
  sorry

end NUMINAMATH_GPT_sum_or_difference_div_by_100_l670_67046


namespace NUMINAMATH_GPT_father_l670_67004

variable (R F M : ℕ)
variable (h1 : F = 4 * R)
variable (h2 : 4 * R + 8 = M * (R + 8))
variable (h3 : 4 * R + 16 = 2 * (R + 16))

theorem father's_age_ratio (hR : R = 8) : (F + 8) / (R + 8) = 5 / 2 := by
  sorry

end NUMINAMATH_GPT_father_l670_67004


namespace NUMINAMATH_GPT_negation_of_existence_l670_67026

theorem negation_of_existence (m : ℤ) :
  (¬ ∃ x : ℤ, x^2 + 2*x + m ≤ 0) ↔ (∀ x : ℤ, x^2 + 2*x + m > 0) :=
by
  sorry

end NUMINAMATH_GPT_negation_of_existence_l670_67026


namespace NUMINAMATH_GPT_Faye_crayons_l670_67054

theorem Faye_crayons (rows crayons_per_row : ℕ) (h_rows : rows = 7) (h_crayons_per_row : crayons_per_row = 30) : rows * crayons_per_row = 210 :=
by
  sorry

end NUMINAMATH_GPT_Faye_crayons_l670_67054


namespace NUMINAMATH_GPT_fraction_of_square_shaded_is_half_l670_67073

theorem fraction_of_square_shaded_is_half {s : ℝ} (h : s > 0) :
  let O := (0, 0)
  let P := (0, s)
  let Q := (s, s / 2)
  let area_square := s^2
  let area_triangle_OPQ := 1 / 2 * s^2 / 2
  let shaded_area := area_square - area_triangle_OPQ
  (shaded_area / area_square) = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_fraction_of_square_shaded_is_half_l670_67073


namespace NUMINAMATH_GPT_circle_outside_hexagon_area_l670_67090

theorem circle_outside_hexagon_area :
  let r := (Real.sqrt 2) / 2
  let s := 1
  let area_circle := π * r^2
  let area_hexagon := 3 * Real.sqrt 3 / 2 * s^2
  area_circle - area_hexagon = (π / 2) - (3 * Real.sqrt 3 / 2) :=
by
  sorry

end NUMINAMATH_GPT_circle_outside_hexagon_area_l670_67090


namespace NUMINAMATH_GPT_total_reading_materials_l670_67097

def reading_materials (magazines newspapers books pamphlets : ℕ) : ℕ :=
  magazines + newspapers + books + pamphlets

theorem total_reading_materials:
  reading_materials 425 275 150 75 = 925 := by
  sorry

end NUMINAMATH_GPT_total_reading_materials_l670_67097


namespace NUMINAMATH_GPT_min_points_tenth_game_l670_67028

-- Defining the scores for each segment of games
def first_five_games : List ℕ := [18, 15, 13, 17, 19]
def next_four_games : List ℕ := [14, 20, 12, 21]

-- Calculating the total score after 9 games
def total_score_after_nine_games : ℕ := first_five_games.sum + next_four_games.sum

-- Defining the required total points after 10 games for an average greater than 17
def required_total_points := 171

-- Proving the number of points needed in the 10th game
theorem min_points_tenth_game (s₁ s₂ : List ℕ) (h₁ : s₁ = first_five_games) (h₂ : s₂ = next_four_games) :
    s₁.sum + s₂.sum + x ≥ required_total_points → x ≥ 22 :=
  sorry

end NUMINAMATH_GPT_min_points_tenth_game_l670_67028


namespace NUMINAMATH_GPT_find_eccentricity_l670_67002

noncomputable def ellipse_gamma (a b : ℝ) (ha_gt : a > 0) (hb_gt : b > 0) (h : a > b) : Prop :=
∀ (x y : ℝ), x^2 / a^2 + y^2 / b^2 = 1

def ellipse_focus (a b : ℝ) : Prop :=
∀ (x y : ℝ), x = 3 → y = 0

def vertex_A (b : ℝ) : Prop :=
∀ (x y : ℝ), x = 0 → y = b

def vertex_B (b : ℝ) : Prop :=
∀ (x y : ℝ), x = 0 → y = -b

def point_N : Prop :=
∀ (x y : ℝ), x = 12 → y = 0

theorem find_eccentricity : 
∀ (a b : ℝ) (ha_gt : a > 0) (hb_gt : b > 0) (h : a > b), 
  ellipse_gamma a b ha_gt hb_gt h → 
  ellipse_focus a b → 
  vertex_A b → 
  vertex_B b → 
  point_N → 
  ∃ e : ℝ, e = 1 / 2 := 
by 
  sorry

end NUMINAMATH_GPT_find_eccentricity_l670_67002


namespace NUMINAMATH_GPT_arithmetic_sequence_k_l670_67014

theorem arithmetic_sequence_k (d : ℤ) (h_d : d ≠ 0) (a : ℕ → ℤ) 
  (h_seq : ∀ n, a n = 0 + n * d) (h_k : a 21 = a 0 + a 1 + a 2 + a 3 + a 4 + a 5 + a 6):
  21 = 21 :=
by
  -- This would be the problem setup
  -- The proof would go here
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_k_l670_67014


namespace NUMINAMATH_GPT_line_increase_is_110_l670_67012

noncomputable def original_lines (increased_lines : ℕ) (percentage_increase : ℚ) : ℚ :=
  increased_lines / (1 + percentage_increase)

theorem line_increase_is_110
  (L' : ℕ)
  (percentage_increase : ℚ)
  (hL' : L' = 240)
  (hp : percentage_increase = 0.8461538461538461) :
  L' - original_lines L' percentage_increase = 110 :=
by
  sorry

end NUMINAMATH_GPT_line_increase_is_110_l670_67012


namespace NUMINAMATH_GPT_find_usual_time_l670_67032

variable (R T : ℝ)

theorem find_usual_time
  (h_condition :  R * T = (9 / 8) * R * (T - 4)) :
  T = 36 :=
by
  sorry

end NUMINAMATH_GPT_find_usual_time_l670_67032


namespace NUMINAMATH_GPT_f_comp_f_neg1_l670_67088

noncomputable def f (x : ℝ) : ℝ :=
if x < 1 then (1 / 4) ^ x else Real.log x / Real.log (1 / 2)

theorem f_comp_f_neg1 : f (f (-1)) = -2 := 
by
  sorry

end NUMINAMATH_GPT_f_comp_f_neg1_l670_67088


namespace NUMINAMATH_GPT_solution_exists_real_solution_31_l670_67082

theorem solution_exists_real_solution_31 :
  ∃ x : ℝ, (2 * x + 1) * (3 * x + 1) * (5 * x + 1) * (30 * x + 1) = 10 ∧ 
            (x = (-4 + Real.sqrt 31) / 15 ∨ x = (-4 - Real.sqrt 31) / 15) :=
sorry

end NUMINAMATH_GPT_solution_exists_real_solution_31_l670_67082


namespace NUMINAMATH_GPT_average_side_length_of_squares_l670_67000

noncomputable def side_length (area : ℝ) : ℝ :=
  Real.sqrt area

noncomputable def average_side_length (areas : List ℝ) : ℝ :=
  (areas.map side_length).sum / (areas.length : ℝ)

theorem average_side_length_of_squares :
  average_side_length [25, 64, 144] = 25 / 3 :=
by
  sorry

end NUMINAMATH_GPT_average_side_length_of_squares_l670_67000


namespace NUMINAMATH_GPT_prob_point_in_region_l670_67027

theorem prob_point_in_region :
  let rect_area := 18
  let intersect_area := 15 / 2
  let probability := intersect_area / rect_area
  probability = 5 / 12 :=
by
  sorry

end NUMINAMATH_GPT_prob_point_in_region_l670_67027


namespace NUMINAMATH_GPT_blue_pill_cost_l670_67008

theorem blue_pill_cost :
  ∃ y : ℝ, ∀ (red_pill_cost blue_pill_cost : ℝ),
    (blue_pill_cost = red_pill_cost + 2) ∧
    (21 * (blue_pill_cost + red_pill_cost) = 819) →
    blue_pill_cost = 20.5 :=
by sorry

end NUMINAMATH_GPT_blue_pill_cost_l670_67008


namespace NUMINAMATH_GPT_minimum_number_of_odd_integers_among_six_l670_67019

theorem minimum_number_of_odd_integers_among_six : 
  ∀ (x y a b m n : ℤ), 
    x + y = 28 →
    x + y + a + b = 45 →
    x + y + a + b + m + n = 63 →
    ∃ (odd_count : ℕ), odd_count = 1 :=
by sorry

end NUMINAMATH_GPT_minimum_number_of_odd_integers_among_six_l670_67019


namespace NUMINAMATH_GPT_common_factor_polynomials_l670_67092

theorem common_factor_polynomials (a : ℝ) :
  (∀ p : ℝ, p ≠ 0 ∧ 
           (p^3 - p - a = 0) ∧ 
           (p^2 + p - a = 0)) → 
  (a = 0 ∨ a = 10 ∨ a = -2) := by
  sorry

end NUMINAMATH_GPT_common_factor_polynomials_l670_67092


namespace NUMINAMATH_GPT_compare_neg_rationals_l670_67025

theorem compare_neg_rationals : - (3 / 4 : ℚ) > - (6 / 5 : ℚ) :=
by sorry

end NUMINAMATH_GPT_compare_neg_rationals_l670_67025


namespace NUMINAMATH_GPT_range_of_angle_of_inclination_l670_67081

theorem range_of_angle_of_inclination (α : ℝ) :
  ∃ θ : ℝ, θ ∈ (Set.Icc 0 (Real.pi / 4) ∪ Set.Ico (3 * Real.pi / 4) Real.pi) ∧
           ∀ x : ℝ, ∃ y : ℝ, y = x * Real.sin α + 1 := by
  sorry

end NUMINAMATH_GPT_range_of_angle_of_inclination_l670_67081


namespace NUMINAMATH_GPT_both_reunions_l670_67062

theorem both_reunions (U O H B : ℕ) 
  (hU : U = 100) 
  (hO : O = 50) 
  (hH : H = 62) 
  (attend_one : U = O + H - B) :  
  B = 12 := 
by 
  sorry

end NUMINAMATH_GPT_both_reunions_l670_67062


namespace NUMINAMATH_GPT_rahul_salary_l670_67036

variable (X : ℝ)

def house_rent_deduction (salary : ℝ) : ℝ := salary * 0.8
def education_expense (remaining_after_rent : ℝ) : ℝ := remaining_after_rent * 0.9
def clothing_expense (remaining_after_education : ℝ) : ℝ := remaining_after_education * 0.9

theorem rahul_salary : (X * 0.8 * 0.9 * 0.9 = 1377) → X = 2125 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_rahul_salary_l670_67036


namespace NUMINAMATH_GPT_sum_fractions_lt_one_l670_67044

theorem sum_fractions_lt_one (a b c d : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) :
  0 < (a / (b + c + d) + b / (a + c + d) + c / (a + b + d) + d / (a + b + c)) ∧
  (a / (b + c + d) + b / (a + c + d) + c / (a + b + d) + d / (a + b + c)) < 1 :=
by
  sorry

end NUMINAMATH_GPT_sum_fractions_lt_one_l670_67044


namespace NUMINAMATH_GPT_polynomial_degree_rational_coefficients_l670_67098

theorem polynomial_degree_rational_coefficients :
  ∃ p : Polynomial ℚ,
    (Polynomial.aeval (2 - 3 * Real.sqrt 3) p = 0) ∧
    (Polynomial.aeval (-2 - 3 * Real.sqrt 3) p = 0) ∧
    (Polynomial.aeval (3 + Real.sqrt 11) p = 0) ∧
    (Polynomial.aeval (3 - Real.sqrt 11) p = 0) ∧
    p.degree = 6 :=
sorry

end NUMINAMATH_GPT_polynomial_degree_rational_coefficients_l670_67098


namespace NUMINAMATH_GPT_cone_volume_l670_67076

theorem cone_volume (R h : ℝ) (hR : 0 ≤ R) (hh : 0 ≤ h) : 
  (∫ x in (0 : ℝ)..h, π * (R / h * x)^2) = (1 / 3) * π * R^2 * h :=
by
  sorry

end NUMINAMATH_GPT_cone_volume_l670_67076


namespace NUMINAMATH_GPT_kamal_chemistry_marks_l670_67093

variables (english math physics biology average total numSubjects : ℕ)

theorem kamal_chemistry_marks 
  (marks_in_english : english = 66)
  (marks_in_math : math = 65)
  (marks_in_physics : physics = 77)
  (marks_in_biology : biology = 75)
  (avg_marks : average = 69)
  (number_of_subjects : numSubjects = 5)
  (total_marks_known : total = 283) :
  ∃ chemistry : ℕ, chemistry = 62 := 
by 
  sorry

end NUMINAMATH_GPT_kamal_chemistry_marks_l670_67093


namespace NUMINAMATH_GPT_trig_identity_simplified_l670_67010

open Real

theorem trig_identity_simplified :
  (sin (15 * π / 180) + cos (15 * π / 180)) * (sin (15 * π / 180) - cos (15 * π / 180)) = - (sqrt 3 / 2) :=
by
  sorry

end NUMINAMATH_GPT_trig_identity_simplified_l670_67010


namespace NUMINAMATH_GPT_inverse_proportional_p_q_l670_67039

theorem inverse_proportional_p_q (k : ℚ)
  (h1 : ∀ p q : ℚ, p * q = k)
  (h2 : (30 : ℚ) * (4 : ℚ) = k) :
  p = 12 ↔ (10 : ℚ) * p = k :=
by
  sorry

end NUMINAMATH_GPT_inverse_proportional_p_q_l670_67039


namespace NUMINAMATH_GPT_carousel_rotation_time_l670_67055

-- Definitions and Conditions
variables (a v U x : ℝ)

-- Conditions given in the problem
def condition1 : Prop := (U * a - v * a = 2 * Real.pi)
def condition2 : Prop := (v * a = U * (x - a / 2))

-- Statement to prove
theorem carousel_rotation_time :
  condition1 a v U ∧ condition2 a v U x → x = 2 * a / 3 :=
by
  intro h
  have c1 := h.1
  have c2 := h.2
  sorry

end NUMINAMATH_GPT_carousel_rotation_time_l670_67055


namespace NUMINAMATH_GPT_polynomial_expansion_l670_67021

theorem polynomial_expansion :
  (7 * X^2 + 5 * X - 3) * (3 * X^3 + 2 * X^2 + 1) = 
  21 * X^5 + 29 * X^4 + X^3 + X^2 + 5 * X - 3 :=
sorry

end NUMINAMATH_GPT_polynomial_expansion_l670_67021


namespace NUMINAMATH_GPT_unique_necklace_arrangements_l670_67057

-- Definitions
def num_beads : Nat := 7

def factorial (n : Nat) : Nat :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

-- The number of unique ways to arrange the beads on a necklace
-- considering rotations and reflections
theorem unique_necklace_arrangements : (factorial num_beads) / (num_beads * 2) = 360 := 
by
  sorry

end NUMINAMATH_GPT_unique_necklace_arrangements_l670_67057


namespace NUMINAMATH_GPT_area_of_inscribed_triangle_l670_67099

noncomputable def area_of_triangle_inscribed_in_circle_with_arcs (a b c : ℕ) := 
  let circum := a + b + c
  let r := circum / (2 * Real.pi)
  let θ := 360 / (a + b + c)
  let angle1 := 4 * θ
  let angle2 := 6 * θ
  let angle3 := 8 * θ
  let sin80 := Real.sin (80 * Real.pi / 180)
  let sin120 := Real.sin (120 * Real.pi / 180)
  let sin160 := Real.sin (160 * Real.pi / 180)
  let approx_vals := sin80 + sin120 + sin160
  (1 / 2) * r^2 * approx_vals

theorem area_of_inscribed_triangle : 
  area_of_triangle_inscribed_in_circle_with_arcs 4 6 8 = 90.33 / Real.pi^2 :=
by sorry

end NUMINAMATH_GPT_area_of_inscribed_triangle_l670_67099


namespace NUMINAMATH_GPT_values_of_z_l670_67061

theorem values_of_z (z : ℤ) (hz : 0 < z) :
  (z^2 - 50 * z + 550 ≤ 10) ↔ (20 ≤ z ∧ z ≤ 30) := sorry

end NUMINAMATH_GPT_values_of_z_l670_67061


namespace NUMINAMATH_GPT_least_x_divisibility_l670_67011

theorem least_x_divisibility :
  ∃ x : ℕ, (x > 0) ∧ ((x^2 + 164) % 3 = 0) ∧ ((x^2 + 164) % 4 = 0) ∧ ((x^2 + 164) % 5 = 0) ∧
  ((x^2 + 164) % 6 = 0) ∧ ((x^2 + 164) % 7 = 0) ∧ ((x^2 + 164) % 8 = 0) ∧ 
  ((x^2 + 164) % 9 = 0) ∧ ((x^2 + 164) % 10 = 0) ∧ ((x^2 + 164) % 11 = 0) ∧ x = 166 → 
  3 = 3 :=
by
  sorry

end NUMINAMATH_GPT_least_x_divisibility_l670_67011


namespace NUMINAMATH_GPT_perimeter_of_8_sided_figure_l670_67022

theorem perimeter_of_8_sided_figure (n : ℕ) (len : ℕ) (h1 : n = 8) (h2 : len = 2) :
  n * len = 16 := by
  sorry

end NUMINAMATH_GPT_perimeter_of_8_sided_figure_l670_67022


namespace NUMINAMATH_GPT_population_difference_l670_67035

variable (A B C : ℝ)

-- Conditions
def population_condition (A B C : ℝ) : Prop := A + B = B + C + 5000

-- The proof statement
theorem population_difference (h : population_condition A B C) : A - C = 5000 :=
by sorry

end NUMINAMATH_GPT_population_difference_l670_67035


namespace NUMINAMATH_GPT_total_lawns_mowed_l670_67001

theorem total_lawns_mowed (earned_per_lawn forgotten_lawns total_earned : ℕ) 
    (h1 : earned_per_lawn = 9) 
    (h2 : forgotten_lawns = 8) 
    (h3 : total_earned = 54) : 
    ∃ (total_lawns : ℕ), total_lawns = 14 :=
by
    sorry

end NUMINAMATH_GPT_total_lawns_mowed_l670_67001


namespace NUMINAMATH_GPT_max_x_minus_y_l670_67015

theorem max_x_minus_y (x y : ℝ) (h : x^2 + y^2 - 4 * x - 2 * y - 4 = 0) : (x - y) ≤ 1 + 3 * (Real.sqrt 2) :=
sorry

end NUMINAMATH_GPT_max_x_minus_y_l670_67015


namespace NUMINAMATH_GPT_find_x_pos_integer_l670_67084

theorem find_x_pos_integer (x : ℕ) (h : 0 < x) (n d : ℕ)
    (h1 : n = x^2 + 4 * x + 29)
    (h2 : d = 4 * x + 9)
    (h3 : n = d * x + 13) : 
    x = 2 := 
sorry

end NUMINAMATH_GPT_find_x_pos_integer_l670_67084


namespace NUMINAMATH_GPT_negation_of_existential_l670_67030

theorem negation_of_existential (P : ℝ → Prop) :
  (¬ ∃ x : ℝ, x^2 - x > 0) ↔ (∀ x : ℝ, x^2 - x ≤ 0) :=
by
  sorry

end NUMINAMATH_GPT_negation_of_existential_l670_67030


namespace NUMINAMATH_GPT_slope_of_tangent_line_at_zero_l670_67067

noncomputable def f (x : ℝ) : ℝ := Real.exp x * Real.cos x

theorem slope_of_tangent_line_at_zero : (deriv f 0) = 1 := 
by
  sorry

end NUMINAMATH_GPT_slope_of_tangent_line_at_zero_l670_67067


namespace NUMINAMATH_GPT_find_minimum_value_l670_67034

noncomputable def fixed_point_at_2_2 (a : ℝ) (ha_pos : a > 0) (ha_ne : a ≠ 1) : Prop :=
∀ (x : ℝ), a^(2-x) + 1 = 2 ↔ x = 2

noncomputable def point_on_line (m n : ℝ) (hmn_pos : m * n > 0) : Prop :=
2 * m + 2 * n = 1

theorem find_minimum_value (m n : ℝ) (hmn_pos : m * n > 0) :
  (fixed_point_at_2_2 a ha_pos ha_ne) → (point_on_line m n hmn_pos) → (1/m + 1/n ≥ 8) :=
sorry

end NUMINAMATH_GPT_find_minimum_value_l670_67034


namespace NUMINAMATH_GPT_odd_positive_multiples_of_7_with_units_digit_1_lt_200_count_l670_67074

theorem odd_positive_multiples_of_7_with_units_digit_1_lt_200_count : 
  ∃ (count : ℕ), count = 3 ∧
  ∀ n : ℕ, (n % 2 = 1) → (n % 7 = 0) → (n < 200) → (n % 10 = 1) → count = 3 :=
sorry

end NUMINAMATH_GPT_odd_positive_multiples_of_7_with_units_digit_1_lt_200_count_l670_67074


namespace NUMINAMATH_GPT_geometric_sequence_max_product_l670_67072

theorem geometric_sequence_max_product
  (b : ℕ → ℝ) (q : ℝ) (b1 : ℝ)
  (h_b1_pos : b1 > 0)
  (h_q : 0 < q ∧ q < 1)
  (h_b : ∀ n, b (n + 1) = b n * q)
  (h_b7_gt_1 : b 7 > 1)
  (h_b8_lt_1 : b 8 < 1) :
  (∀ (n : ℕ), n = 7 → b 1 * b 2 * b 3 * b 4 * b 5 * b 6 * b 7 = b 1 * b 2 * b 3 * b 4 * b 5 * b 6 * b 7) :=
by {
  sorry
}

end NUMINAMATH_GPT_geometric_sequence_max_product_l670_67072


namespace NUMINAMATH_GPT_sequence_formula_l670_67051

noncomputable def a (n : ℕ) : ℕ := n

theorem sequence_formula (n : ℕ) (h : 0 < n) (S_n : ℕ → ℕ) 
  (hSn : ∀ m : ℕ, S_n m = (1 / 2 : ℚ) * (a m)^2 + (1 / 2 : ℚ) * m) : a n = n :=
by
  sorry

end NUMINAMATH_GPT_sequence_formula_l670_67051


namespace NUMINAMATH_GPT_total_lambs_l670_67040

def num_initial_lambs : ℕ := 6
def num_baby_lambs_per_mother : ℕ := 2
def num_mothers : ℕ := 2
def traded_lambs : ℕ := 3
def extra_lambs : ℕ := 7

theorem total_lambs :
  num_initial_lambs + (num_baby_lambs_per_mother * num_mothers) - traded_lambs + extra_lambs = 14 :=
by
  sorry

end NUMINAMATH_GPT_total_lambs_l670_67040


namespace NUMINAMATH_GPT_evaluate_expression_l670_67095

/-
  Define the expressions from the conditions.
  We define the numerator and denominator separately.
-/
def expr_numerator : ℚ := 1 - (1 / 4)
def expr_denominator : ℚ := 1 - (1 / 3)

/-
  Define the original expression to be proven.
  This is our main expression to evaluate.
-/
def expr : ℚ := expr_numerator / expr_denominator

/-
  State the final proof problem that the expression is equal to 9/8.
-/
theorem evaluate_expression : expr = 9 / 8 := sorry

end NUMINAMATH_GPT_evaluate_expression_l670_67095


namespace NUMINAMATH_GPT_three_sport_players_l670_67006

def total_members := 50
def B := 22
def T := 28
def Ba := 18
def BT := 10
def BBa := 8
def TBa := 12
def N := 4
def All := 8

theorem three_sport_players : B + T + Ba - (BT + BBa + TBa) + All = total_members - N :=
by
suffices h : 22 + 28 + 18 - (10 + 8 + 12) + 8 = 50 - 4
exact h
-- The detailed proof is left as an exercise
sorry

end NUMINAMATH_GPT_three_sport_players_l670_67006


namespace NUMINAMATH_GPT_sum_of_three_smallest_two_digit_primes_l670_67066

theorem sum_of_three_smallest_two_digit_primes :
  11 + 13 + 17 = 41 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_three_smallest_two_digit_primes_l670_67066


namespace NUMINAMATH_GPT_angle_between_diagonals_l670_67056

open Real

theorem angle_between_diagonals
  (a b c : ℝ) :
  ∃ θ : ℝ, θ = arccos (a^2 / sqrt ((a^2 + b^2) * (a^2 + c^2))) :=
by
  -- Placeholder for the proof
  sorry

end NUMINAMATH_GPT_angle_between_diagonals_l670_67056


namespace NUMINAMATH_GPT_students_per_bus_l670_67075

/-- The number of students who can be accommodated in each bus -/
theorem students_per_bus (total_students : ℕ) (students_in_cars : ℕ) (num_buses : ℕ) 
(h1 : total_students = 375) (h2 : students_in_cars = 4) (h3 : num_buses = 7) : 
(total_students - students_in_cars) / num_buses = 53 :=
by
  sorry

end NUMINAMATH_GPT_students_per_bus_l670_67075


namespace NUMINAMATH_GPT_lulu_cash_left_l670_67050

-- Define the initial amount
def initial_amount : ℕ := 65

-- Define the amount spent on ice cream
def spent_on_ice_cream : ℕ := 5

-- Define the amount spent on a t-shirt
def spent_on_tshirt (remaining_after_ice_cream : ℕ) : ℕ := remaining_after_ice_cream / 2

-- Define the amount deposited in the bank
def deposited_in_bank (remaining_after_tshirt : ℕ) : ℕ := remaining_after_tshirt / 5

-- Define the remaining cash after all transactions
def remaining_cash (initial : ℕ) (spent_ice_cream : ℕ) (spent_tshirt: ℕ) (deposited: ℕ) :ℕ :=
  initial - spent_ice_cream - spent_tshirt - deposited

-- Theorem statement to prove
theorem lulu_cash_left : remaining_cash initial_amount spent_on_ice_cream (spent_on_tshirt (initial_amount - spent_on_ice_cream)) 
(deposited_in_bank ((initial_amount - spent_on_ice_cream) - (spent_on_tshirt (initial_amount - spent_on_ice_cream)))) = 24 :=
by
  sorry

end NUMINAMATH_GPT_lulu_cash_left_l670_67050


namespace NUMINAMATH_GPT_scoops_for_mom_l670_67078

/-- 
  Each scoop of ice cream costs $2.
  Pierre gets 3 scoops.
  The total bill is $14.
  Prove that Pierre's mom gets 4 scoops.
-/
theorem scoops_for_mom
  (scoop_cost : ℕ)
  (pierre_scoops : ℕ)
  (total_bill : ℕ) :
  scoop_cost = 2 → pierre_scoops = 3 → total_bill = 14 → 
  (total_bill - pierre_scoops * scoop_cost) / scoop_cost = 4 := 
by
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_scoops_for_mom_l670_67078


namespace NUMINAMATH_GPT_hair_cut_off_length_l670_67003

def initial_hair_length : ℕ := 18
def hair_length_after_haircut : ℕ := 9

theorem hair_cut_off_length :
  initial_hair_length - hair_length_after_haircut = 9 :=
sorry

end NUMINAMATH_GPT_hair_cut_off_length_l670_67003


namespace NUMINAMATH_GPT_dividend_is_5336_l670_67079

theorem dividend_is_5336 (D Q R : ℕ) (h1 : D = 10 * Q) (h2 : D = 5 * R) (h3 : R = 46) :
  (D * Q + R) = 5336 :=
by {
  sorry
}

end NUMINAMATH_GPT_dividend_is_5336_l670_67079


namespace NUMINAMATH_GPT_find_real_x_l670_67043

noncomputable def solution_set (x : ℝ) := (5 ≤ x) ∧ (x < 5.25)

theorem find_real_x (x : ℝ) :
  (⌊x * ⌊x⌋⌋ = 20) ↔ solution_set x :=
by
  sorry

end NUMINAMATH_GPT_find_real_x_l670_67043


namespace NUMINAMATH_GPT_g_of_5_l670_67013

noncomputable def g (x : ℝ) : ℝ := -2 / x

theorem g_of_5 (x : ℝ) : g (g (g (g (g x)))) = -2 / x :=
by
  sorry

end NUMINAMATH_GPT_g_of_5_l670_67013


namespace NUMINAMATH_GPT_gcd_12569_36975_l670_67086

-- Define the integers for which we need to find the gcd
def num1 : ℕ := 12569
def num2 : ℕ := 36975

-- The statement that the gcd of these two numbers is 1
theorem gcd_12569_36975 : Nat.gcd num1 num2 = 1 := by
  sorry

end NUMINAMATH_GPT_gcd_12569_36975_l670_67086


namespace NUMINAMATH_GPT_max_rect_area_with_given_perimeter_l670_67096

-- Define the variables used in the problem
def length_of_wire := 12
def max_area (x : ℝ) := -(x - 3)^2 + 9

-- Lean Statement for the problem
theorem max_rect_area_with_given_perimeter : ∃ (A : ℝ), (∀ (x : ℝ), 0 < x ∧ x < 6 → (x * (6 - x) ≤ A)) ∧ A = 9 :=
by
  sorry

end NUMINAMATH_GPT_max_rect_area_with_given_perimeter_l670_67096


namespace NUMINAMATH_GPT_shortest_side_of_right_triangle_l670_67070

theorem shortest_side_of_right_triangle (a b : ℝ) (h : a = 9 ∧ b = 12) : ∃ c : ℝ, (c = min a b) ∧ c = 9 :=
by
  sorry

end NUMINAMATH_GPT_shortest_side_of_right_triangle_l670_67070


namespace NUMINAMATH_GPT_sum_of_coefficients_l670_67029

theorem sum_of_coefficients (x y : ℝ) : 
  (2 * x - 3 * y) ^ 9 = -1 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_coefficients_l670_67029


namespace NUMINAMATH_GPT_cindy_correct_answer_l670_67007

theorem cindy_correct_answer (x : ℕ) (h : (x - 9) / 3 = 43) : (x - 3) / 9 = 15 :=
by
sorry

end NUMINAMATH_GPT_cindy_correct_answer_l670_67007


namespace NUMINAMATH_GPT_values_of_x_for_g_l670_67064

def g (x : ℝ) : ℝ := x^2 - 5 * x

theorem values_of_x_for_g (x : ℝ) :
  g (g x) = g x ↔ x = 0 ∨ x = 5 ∨ x = 6 ∨ x = -1 := by
    sorry

end NUMINAMATH_GPT_values_of_x_for_g_l670_67064


namespace NUMINAMATH_GPT_political_exam_pass_l670_67009

-- Define the students' statements.
def A_statement (C_passed : Prop) : Prop := C_passed
def B_statement (B_passed : Prop) : Prop := ¬ B_passed
def C_statement (A_statement : Prop) : Prop := A_statement

-- Define the problem conditions.
def condition_1 (A_passed B_passed C_passed : Prop) : Prop := ¬A_passed ∨ ¬B_passed ∨ ¬C_passed
def condition_2 (A_passed B_passed C_passed : Prop) := A_statement C_passed
def condition_3 (A_passed B_passed C_passed : Prop) := B_statement B_passed
def condition_4 (A_passed B_passed C_passed : Prop) := C_statement (A_statement C_passed)
def condition_5 (A_statement_true B_statement_true C_statement_true : Prop) : Prop := 
  (¬A_statement_true ∧ B_statement_true ∧ C_statement_true) ∨
  (A_statement_true ∧ ¬B_statement_true ∧ C_statement_true) ∨
  (A_statement_true ∧ B_statement_true ∧ ¬C_statement_true)

-- Define the proof problem.
theorem political_exam_pass : 
  ∀ (A_passed B_passed C_passed : Prop),
  condition_1 A_passed B_passed C_passed →
  condition_2 A_passed B_passed C_passed →
  condition_3 A_passed B_passed C_passed →
  condition_4 A_passed B_passed C_passed →
  ∃ (A_statement_true B_statement_true C_statement_true : Prop), 
  condition_5 A_statement_true B_statement_true C_statement_true →
  ¬A_passed
:= by { sorry }

end NUMINAMATH_GPT_political_exam_pass_l670_67009


namespace NUMINAMATH_GPT_find_sum_lent_l670_67083

theorem find_sum_lent (P : ℝ) : 
  (∃ R T : ℝ, R = 4 ∧ T = 8 ∧ I = P - 170 ∧ I = (P * 8) / 25) → P = 250 :=
by
  sorry

end NUMINAMATH_GPT_find_sum_lent_l670_67083


namespace NUMINAMATH_GPT_right_triangle_perimeter_l670_67020

-- Conditions
variable (a : ℝ) (b : ℝ) (c : ℝ)
variable (h_area : 1 / 2 * 15 * b = 150)
variable (h_pythagorean : a^2 + b^2 = c^2)
variable (h_a : a = 15)

-- The theorem to prove the perimeter is 60 units
theorem right_triangle_perimeter : a + b + c = 60 := by
  sorry

end NUMINAMATH_GPT_right_triangle_perimeter_l670_67020


namespace NUMINAMATH_GPT_cumulative_profit_exceeds_technical_renovation_expressions_for_A_n_B_n_l670_67089

noncomputable def A_n (n : ℕ) : ℝ :=
  490 * n - 10 * n^2

noncomputable def B_n (n : ℕ) : ℝ :=
  500 * n + 400 - 500 / 2^(n-1)

theorem cumulative_profit_exceeds_technical_renovation :
  ∀ n : ℕ, n ≥ 4 → B_n n > A_n n :=
by
  sorry  -- Proof goes here

theorem expressions_for_A_n_B_n (n : ℕ) :
  A_n n = 490 * n - 10 * n^2 ∧
  B_n n = 500 * n + 400 - 500 / 2^(n-1) :=
by
  sorry  -- Proof goes here

end NUMINAMATH_GPT_cumulative_profit_exceeds_technical_renovation_expressions_for_A_n_B_n_l670_67089


namespace NUMINAMATH_GPT_henry_collected_points_l670_67042

def points_from_wins (wins : ℕ) : ℕ := wins * 5
def points_from_losses (losses : ℕ) : ℕ := losses * 2
def points_from_draws (draws : ℕ) : ℕ := draws * 3

def total_points (wins losses draws : ℕ) : ℕ := 
  points_from_wins wins + points_from_losses losses + points_from_draws draws

theorem henry_collected_points :
  total_points 2 2 10 = 44 := by
  -- The proof will go here
  sorry

end NUMINAMATH_GPT_henry_collected_points_l670_67042


namespace NUMINAMATH_GPT_monotone_function_sol_l670_67060

noncomputable def monotone_function (f : ℤ → ℤ) :=
  ∀ x y : ℤ, f x ≤ f y → x ≤ y

theorem monotone_function_sol
  (f : ℤ → ℤ)
  (H1 : monotone_function f)
  (H2 : ∀ x y : ℤ, f (x^2005 + y^2005) = f x ^ 2005 + f y ^ 2005) :
  (∀ x : ℤ, f x = x) ∨ (∀ x : ℤ, f x = -x) :=
sorry

end NUMINAMATH_GPT_monotone_function_sol_l670_67060


namespace NUMINAMATH_GPT_benny_turnips_l670_67018

-- Definitions and conditions
def melanie_turnips : ℕ := 139
def total_turnips : ℕ := 252

-- Question to prove
theorem benny_turnips : ∃ b : ℕ, b = total_turnips - melanie_turnips ∧ b = 113 :=
by {
    sorry
}

end NUMINAMATH_GPT_benny_turnips_l670_67018


namespace NUMINAMATH_GPT_total_distance_traveled_l670_67049

def trip_duration : ℕ := 8
def speed_first_half : ℕ := 70
def speed_second_half : ℕ := 85
def time_each_half : ℕ := trip_duration / 2

theorem total_distance_traveled :
  let distance_first_half := time_each_half * speed_first_half
  let distance_second_half := time_each_half * speed_second_half
  let total_distance := distance_first_half + distance_second_half
  total_distance = 620 := by
  sorry

end NUMINAMATH_GPT_total_distance_traveled_l670_67049


namespace NUMINAMATH_GPT_stocking_stuffers_total_l670_67023

theorem stocking_stuffers_total 
  (candy_canes_per_child beanie_babies_per_child books_per_child : ℕ)
  (num_children : ℕ)
  (h1 : candy_canes_per_child = 4)
  (h2 : beanie_babies_per_child = 2)
  (h3 : books_per_child = 1)
  (h4 : num_children = 3) :
  candy_canes_per_child + beanie_babies_per_child + books_per_child * num_children = 21 :=
by
  sorry

end NUMINAMATH_GPT_stocking_stuffers_total_l670_67023


namespace NUMINAMATH_GPT_find_f_minus_half_l670_67080

-- Definitions
def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

def function_definition (f : ℝ → ℝ) : Prop :=
  ∀ x, x > 0 → f x = 4^x

-- Theorem statement
theorem find_f_minus_half {f : ℝ → ℝ}
  (h_odd : is_odd_function f)
  (h_def : function_definition f) :
  f (-1/2) = -2 :=
by
  sorry

end NUMINAMATH_GPT_find_f_minus_half_l670_67080


namespace NUMINAMATH_GPT_problem_1_problem_2_l670_67065

theorem problem_1 (α : ℝ) (hα : Real.tan α = 2) :
  Real.tan (α + Real.pi / 4) = -3 :=
by
  sorry

theorem problem_2 (α : ℝ) (hα : Real.tan α = 2) :
  (6 * Real.sin α + Real.cos α) / (3 * Real.sin α - 2 * Real.cos α) = 13 / 4 :=
by
  sorry

end NUMINAMATH_GPT_problem_1_problem_2_l670_67065


namespace NUMINAMATH_GPT_students_in_grade6_l670_67016

noncomputable def num_students_total : ℕ := 100
noncomputable def num_students_grade4 : ℕ := 30
noncomputable def num_students_grade5 : ℕ := 35
noncomputable def num_students_grade6 : ℕ := num_students_total - (num_students_grade4 + num_students_grade5)

theorem students_in_grade6 : num_students_grade6 = 35 := by
  sorry

end NUMINAMATH_GPT_students_in_grade6_l670_67016


namespace NUMINAMATH_GPT_find_a_l670_67094

-- Define the conditions of the problem
def line1 (a : ℝ) : ℝ × ℝ × ℝ := (a + 3, 1, -3) -- Coefficients of line1: (a+3)x + y - 3 = 0
def line2 (a : ℝ) : ℝ × ℝ × ℝ := (5, a - 3, 4)  -- Coefficients of line2: 5x + (a-3)y + 4 = 0

-- Definition of direction vector and normal vector
def direction_vector (a : ℝ) : ℝ × ℝ := (1, -(a + 3))
def normal_vector (a : ℝ) : ℝ × ℝ := (5, a - 3)

-- Proof statement
theorem find_a (a : ℝ) : (direction_vector a = normal_vector a) → a = -2 :=
by {
  -- Insert proof here
  sorry
}

end NUMINAMATH_GPT_find_a_l670_67094


namespace NUMINAMATH_GPT_number_of_passed_candidates_l670_67045

theorem number_of_passed_candidates
  (P F : ℕ) 
  (h1 : P + F = 120)
  (h2 : 39 * P + 15 * F = 4200) : P = 100 :=
sorry

end NUMINAMATH_GPT_number_of_passed_candidates_l670_67045


namespace NUMINAMATH_GPT_total_fish_in_pond_l670_67085

theorem total_fish_in_pond (N : ℕ) (h1 : 80 ≤ N) (h2 : 5 ≤ 150) (h_marked_dist : (5 : ℚ) / 150 = (80 : ℚ) / N) : N = 2400 := by
  sorry

end NUMINAMATH_GPT_total_fish_in_pond_l670_67085


namespace NUMINAMATH_GPT_no_super_squarish_numbers_l670_67063

def is_super_squarish (M : ℕ) : Prop :=
  let a := M / 100000 % 100
  let b := M / 1000 % 1000
  let c := M % 100
  (M ≥ 1000000 ∧ M < 10000000) ∧
  (M % 10 ≠ 0 ∧ (M / 10) % 10 ≠ 0 ∧ (M / 100) % 10 ≠ 0 ∧ (M / 1000) % 10 ≠ 0 ∧
    (M / 10000) % 10 ≠ 0 ∧ (M / 100000) % 10 ≠ 0 ∧ (M / 1000000) % 10 ≠ 0) ∧
  (∃ y : ℕ, y * y = M) ∧
  (∃ f g : ℕ, f * f = a ∧ 2 * f * g = b ∧ g * g = c) ∧
  (10 ≤ a ∧ a ≤ 99) ∧
  (100 ≤ b ∧ b ≤ 999) ∧
  (10 ≤ c ∧ c ≤ 99)

theorem no_super_squarish_numbers : ∀ M : ℕ, is_super_squarish M → false :=
sorry

end NUMINAMATH_GPT_no_super_squarish_numbers_l670_67063


namespace NUMINAMATH_GPT_chocolate_bars_remaining_l670_67041

theorem chocolate_bars_remaining (total_bars sold_week1 sold_week2 : ℕ) (h_total : total_bars = 18) (h_sold1 : sold_week1 = 5) (h_sold2 : sold_week2 = 7) : total_bars - (sold_week1 + sold_week2) = 6 :=
by {
  sorry
}

end NUMINAMATH_GPT_chocolate_bars_remaining_l670_67041


namespace NUMINAMATH_GPT_length_of_segment_NB_l670_67031

variable (L W x : ℝ)
variable (h1 : 0 < L) (h2 : 0 < W) (h3 : x * W / 2 = 0.4 * (L * W))

theorem length_of_segment_NB (L W x : ℝ) (h1 : 0 < L) (h2 : 0 < W) (h3 : x * W / 2 = 0.4 * (L * W)) : 
  x = 0.8 * L :=
by
  sorry

end NUMINAMATH_GPT_length_of_segment_NB_l670_67031


namespace NUMINAMATH_GPT_number_of_diagonals_octagon_heptagon_diff_l670_67047

def number_of_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

theorem number_of_diagonals_octagon_heptagon_diff :
  let A := number_of_diagonals 8
  let B := number_of_diagonals 7
  A - B = 6 :=
by
  sorry

end NUMINAMATH_GPT_number_of_diagonals_octagon_heptagon_diff_l670_67047


namespace NUMINAMATH_GPT_range_of_a_l670_67077

-- Define the function f
def f (x a : ℝ) : ℝ := |x - 1| + |x - a|

-- State the problem
theorem range_of_a (a : ℝ) : (∀ x : ℝ, f x a ≥ 2) → (a ≤ -1 ∨ a ≥ 3) :=
by 
  sorry

end NUMINAMATH_GPT_range_of_a_l670_67077


namespace NUMINAMATH_GPT_integer_count_in_interval_l670_67068

theorem integer_count_in_interval : 
  let lower_bound := Int.floor (-7 * Real.pi)
  let upper_bound := Int.ceil (12 * Real.pi)
  upper_bound - lower_bound + 1 = 61 :=
by
  let lower_bound := Int.floor (-7 * Real.pi)
  let upper_bound := Int.ceil (12 * Real.pi)
  have : upper_bound - lower_bound + 1 = 61 := sorry
  exact this

end NUMINAMATH_GPT_integer_count_in_interval_l670_67068
