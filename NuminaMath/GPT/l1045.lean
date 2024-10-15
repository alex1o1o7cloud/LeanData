import Mathlib

namespace NUMINAMATH_GPT_arithmetic_sequence_formula_l1045_104560

-- Define the sequence and its properties
def is_arithmetic_sequence (a : ℤ) (u : ℕ → ℤ) : Prop :=
  u 0 = a - 1 ∧ u 1 = a + 1 ∧ u 2 = 2 * a + 3 ∧ ∀ n, u (n + 1) - u n = u 1 - u 0

theorem arithmetic_sequence_formula (a : ℤ) :
  ∃ u : ℕ → ℤ, is_arithmetic_sequence a u ∧ (∀ n, u n = 2 * n - 3) :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_formula_l1045_104560


namespace NUMINAMATH_GPT_negation_of_implication_l1045_104520

theorem negation_of_implication {r p q : Prop} :
  ¬ (r → (p ∨ q)) ↔ (¬ r → (¬ p ∧ ¬ q)) :=
by sorry

end NUMINAMATH_GPT_negation_of_implication_l1045_104520


namespace NUMINAMATH_GPT_problem_l1045_104530

def g (x : ℝ) (d e f : ℝ) := d * x^2 + e * x + f

theorem problem (d e f : ℝ) (h_vertex : ∀ x : ℝ, g d e f (x + 2) = -1 * (x + 2)^2 + 5) :
  d + e + 3 * f = 14 := 
sorry

end NUMINAMATH_GPT_problem_l1045_104530


namespace NUMINAMATH_GPT_value_of_k_l1045_104591

theorem value_of_k (a b k : ℝ) (h1 : 2^a = k) (h2 : 3^b = k) (h3 : k ≠ 1) (h4 : 2 * a + b = 2 * a * b) : k = 3 * Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_GPT_value_of_k_l1045_104591


namespace NUMINAMATH_GPT_cover_square_floor_l1045_104527

theorem cover_square_floor (x : ℕ) (h : 2 * x - 1 = 37) : x^2 = 361 :=
by
  sorry

end NUMINAMATH_GPT_cover_square_floor_l1045_104527


namespace NUMINAMATH_GPT_gcd_2183_1947_l1045_104512

theorem gcd_2183_1947 : Nat.gcd 2183 1947 = 59 := 
by 
  sorry

end NUMINAMATH_GPT_gcd_2183_1947_l1045_104512


namespace NUMINAMATH_GPT_sum_infinite_series_l1045_104504

noncomputable def series_term (n : ℕ) : ℚ := 
  (2 * n + 3) / (n * (n + 1) * (n + 2))

noncomputable def partial_fractions (n : ℕ) : ℚ := 
  (3 / 2) / n - 1 / (n + 1) - (1 / 2) / (n + 2)

theorem sum_infinite_series : 
  (∑' n : ℕ, series_term (n + 1)) = 5 / 4 := 
by
  sorry

end NUMINAMATH_GPT_sum_infinite_series_l1045_104504


namespace NUMINAMATH_GPT_common_point_arithmetic_progression_l1045_104595

theorem common_point_arithmetic_progression (a b c : ℝ) (h : 2 * b = a + c) :
  ∃ (x y : ℝ), (∀ x, y = a * x^2 + b * x + c) ∧ x = -2 ∧ y = 0 :=
by
  sorry

end NUMINAMATH_GPT_common_point_arithmetic_progression_l1045_104595


namespace NUMINAMATH_GPT_june_vs_christopher_l1045_104546

namespace SwordLength

def christopher_length : ℕ := 15
def jameson_length : ℕ := 3 + 2 * christopher_length
def june_length : ℕ := 5 + jameson_length

theorem june_vs_christopher : june_length - christopher_length = 23 := by
  show 5 + (3 + 2 * christopher_length) - christopher_length = 23
  sorry

end SwordLength

end NUMINAMATH_GPT_june_vs_christopher_l1045_104546


namespace NUMINAMATH_GPT_range_of_a_l1045_104584

theorem range_of_a (x a : ℝ) (h₀ : x < 0) (h₁ : 2^x - a = 1 / (x - 1)) : 0 < a ∧ a < 2 :=
sorry

end NUMINAMATH_GPT_range_of_a_l1045_104584


namespace NUMINAMATH_GPT_range_of_f_l1045_104555

noncomputable def f (x : ℝ) := Real.arcsin (x ^ 2 - x)

theorem range_of_f :
  Set.range f = Set.Icc (-Real.arcsin (1/4)) (Real.pi / 2) :=
sorry

end NUMINAMATH_GPT_range_of_f_l1045_104555


namespace NUMINAMATH_GPT_joan_half_dollars_spent_on_wednesday_l1045_104537

variable (x : ℝ)
variable (h1 : x * 0.5 + 14 * 0.5 = 9)

theorem joan_half_dollars_spent_on_wednesday :
  x = 4 :=
by
  -- The proof is not required, hence using sorry
  sorry

end NUMINAMATH_GPT_joan_half_dollars_spent_on_wednesday_l1045_104537


namespace NUMINAMATH_GPT_digit_150th_of_17_div_70_is_7_l1045_104557

noncomputable def decimal_representation (n d : ℚ) : ℚ := n / d

-- We define the repeating part of the decimal representation.
def repeating_part : ℕ := 242857

-- Define the function to get the nth digit of the repeating part.
def get_nth_digit_of_repeating (n : ℕ) : ℕ :=
  let sequence := [2, 4, 2, 8, 5, 7]
  sequence.get! (n % sequence.length)

theorem digit_150th_of_17_div_70_is_7 : get_nth_digit_of_repeating 149 = 7 :=
by
  sorry

end NUMINAMATH_GPT_digit_150th_of_17_div_70_is_7_l1045_104557


namespace NUMINAMATH_GPT_simplify_expression_l1045_104503

theorem simplify_expression (a b : ℕ) (h : a / b = 1 / 3) : 
    1 - (a - b) / (a - 2 * b) / ((a ^ 2 - b ^ 2) / (a ^ 2 - 4 * a * b + 4 * b ^ 2)) = 3 / 4 := 
by sorry

end NUMINAMATH_GPT_simplify_expression_l1045_104503


namespace NUMINAMATH_GPT_problem_statement_l1045_104510

-- Definition of the arithmetic sequence {a_n}
def a (n : ℕ) : ℕ := n

-- Definition of the geometric sequence {b_n}
def b (n : ℕ) : ℕ := 2^n

-- Definition of the sequence {c_n}
def c (n : ℕ) : ℕ := a n + b n

-- Sum of first n terms of the sequence {c_n}
def S (n : ℕ) : ℕ := (n * (n + 1)) / 2 + 2^(n + 1) - 2

-- Prove the problem statement
theorem problem_statement :
  (a 1 + a 2 = 3) ∧
  (a 4 - a 3 = 1) ∧
  (b 2 = a 4) ∧
  (b 3 = a 8) ∧
  (∀ n : ℕ, c n = a n + b n) ∧
  (∀ n : ℕ, S n = (n * (n + 1)) / 2 + 2^(n + 1) - 2) :=
by {
  sorry -- Proof goes here
}

end NUMINAMATH_GPT_problem_statement_l1045_104510


namespace NUMINAMATH_GPT_pow_four_inequality_l1045_104553

theorem pow_four_inequality (x y : ℝ) : x^4 + y^4 ≥ x * y * (x + y)^2 :=
by
  sorry

end NUMINAMATH_GPT_pow_four_inequality_l1045_104553


namespace NUMINAMATH_GPT_men_wages_l1045_104539

def men := 5
def women := 5
def boys := 7
def total_wages := 90
def wage_man := 7.5

theorem men_wages (men women boys : ℕ) (total_wages wage_man : ℝ)
  (h1 : 5 = women) (h2 : women = boys) (h3 : 5 * wage_man + 1 * wage_man + 7 * wage_man = total_wages) :
  5 * wage_man = 37.5 :=
  sorry

end NUMINAMATH_GPT_men_wages_l1045_104539


namespace NUMINAMATH_GPT_sum_possible_values_l1045_104540

theorem sum_possible_values (p q r s : ℝ) (h : (p - q) * (r - s) / ((q - r) * (s - p)) = 3 / 4) :
  (p - r) * (q - s) / ((p - q) * (r - s)) = -1 := 
by
  sorry

end NUMINAMATH_GPT_sum_possible_values_l1045_104540


namespace NUMINAMATH_GPT_percent_error_l1045_104523

theorem percent_error (x : ℝ) (h : x > 0) :
  (abs ((12 * x) - (x / 3)) / (x / 3)) * 100 = 3500 :=
by
  sorry

end NUMINAMATH_GPT_percent_error_l1045_104523


namespace NUMINAMATH_GPT_ratio_used_to_total_apples_l1045_104500

noncomputable def total_apples_bonnie : ℕ := 8
noncomputable def total_apples_samuel : ℕ := total_apples_bonnie + 20
noncomputable def eaten_apples_samuel : ℕ := total_apples_samuel / 2
noncomputable def used_for_pie_samuel : ℕ := total_apples_samuel - eaten_apples_samuel - 10

theorem ratio_used_to_total_apples : used_for_pie_samuel / (Nat.gcd used_for_pie_samuel total_apples_samuel) = 1 ∧
                                     total_apples_samuel / (Nat.gcd used_for_pie_samuel total_apples_samuel) = 7 := by
  sorry

end NUMINAMATH_GPT_ratio_used_to_total_apples_l1045_104500


namespace NUMINAMATH_GPT_list_price_of_article_l1045_104549

theorem list_price_of_article
  (P : ℝ)
  (discount1 : ℝ)
  (discount2 : ℝ)
  (final_price : ℝ)
  (h1 : discount1 = 0.10)
  (h2 : discount2 = 0.01999999999999997)
  (h3 : final_price = 61.74) :
  P = 70 :=
by
  sorry

end NUMINAMATH_GPT_list_price_of_article_l1045_104549


namespace NUMINAMATH_GPT_stadium_surface_area_correct_l1045_104501

noncomputable def stadium_length_yards : ℝ := 62
noncomputable def stadium_width_yards : ℝ := 48
noncomputable def stadium_height_yards : ℝ := 30

noncomputable def stadium_length_feet : ℝ := stadium_length_yards * 3
noncomputable def stadium_width_feet : ℝ := stadium_width_yards * 3
noncomputable def stadium_height_feet : ℝ := stadium_height_yards * 3

def total_surface_area_stadium (length : ℝ) (width : ℝ) (height : ℝ) : ℝ :=
  2 * (length * width + width * height + height * length)

theorem stadium_surface_area_correct :
  total_surface_area_stadium stadium_length_feet stadium_width_feet stadium_height_feet = 110968 := by
  sorry

end NUMINAMATH_GPT_stadium_surface_area_correct_l1045_104501


namespace NUMINAMATH_GPT_equation_is_hyperbola_l1045_104564

theorem equation_is_hyperbola : 
  ∀ x y : ℝ, (x^2 - 25*y^2 - 10*x + 50 = 0) → 
  (∃ a b h k : ℝ, (a ≠ 0 ∧ b ≠ 0 ∧ (x - h)^2 / a^2 - (y - k)^2 / b^2 = -1)) :=
by
  sorry

end NUMINAMATH_GPT_equation_is_hyperbola_l1045_104564


namespace NUMINAMATH_GPT_vector_identity_l1045_104531

namespace VectorAddition

variable {V : Type*} [AddCommGroup V]

theorem vector_identity
  (AD DC AB BC : V)
  (h1 : AD + DC = AC)
  (h2 : AC - AB = BC) :
  AD + DC - AB = BC :=
by
  sorry

end VectorAddition

end NUMINAMATH_GPT_vector_identity_l1045_104531


namespace NUMINAMATH_GPT_fran_speed_l1045_104594

-- Definitions for conditions
def joann_speed : ℝ := 15 -- in miles per hour
def joann_time : ℝ := 4 -- in hours
def fran_time : ℝ := 2 -- in hours
def joann_distance : ℝ := joann_speed * joann_time -- distance Joann traveled

-- Proof Goal Statement
theorem fran_speed (hf: fran_time ≠ 0) : (joann_speed * joann_time) / fran_time = 30 :=
by
  -- Sorry placeholder skips the proof steps
  sorry

end NUMINAMATH_GPT_fran_speed_l1045_104594


namespace NUMINAMATH_GPT_will_initially_bought_seven_boxes_l1045_104592

theorem will_initially_bought_seven_boxes :
  let given_away_pieces := 3 * 4
  let total_initial_pieces := given_away_pieces + 16
  let initial_boxes := total_initial_pieces / 4
  initial_boxes = 7 := 
by
  sorry

end NUMINAMATH_GPT_will_initially_bought_seven_boxes_l1045_104592


namespace NUMINAMATH_GPT_inequality_lemma_l1045_104590

-- Define the conditions: x and y are positive numbers and x > y
variables (x y : ℝ)
variables (hx_pos : 0 < x) (hy_pos : 0 < y) (hxy : x > y)

-- State the theorem to be proved
theorem inequality_lemma : 2 * x + 1 / (x^2 - 2*x*y + y^2) >= 2 * y + 3 :=
by
  sorry

end NUMINAMATH_GPT_inequality_lemma_l1045_104590


namespace NUMINAMATH_GPT_find_fraction_squares_l1045_104516

theorem find_fraction_squares (x y z a b c : ℝ) 
  (h1 : x / a + y / b + z / c = 4) 
  (h2 : a / x + b / y + c / z = 0) : 
  x^2 / a^2 + y^2 / b^2 + z^2 / c^2 = 16 := 
by
  sorry

end NUMINAMATH_GPT_find_fraction_squares_l1045_104516


namespace NUMINAMATH_GPT_exists_positive_integer_k_l1045_104543

theorem exists_positive_integer_k :
  ∃ k : ℕ, k > 0 ∧ ∀ n : ℕ, n > 0 → ¬ Nat.Prime (2^n * k + 1) ∧ 2^n * k + 1 > 1 :=
by
  sorry

end NUMINAMATH_GPT_exists_positive_integer_k_l1045_104543


namespace NUMINAMATH_GPT_sum_smallest_largest_3_digit_numbers_made_up_of_1_2_5_l1045_104506

theorem sum_smallest_largest_3_digit_numbers_made_up_of_1_2_5 :
  let smallest := 125
  let largest := 521
  smallest + largest = 646 := by
  sorry

end NUMINAMATH_GPT_sum_smallest_largest_3_digit_numbers_made_up_of_1_2_5_l1045_104506


namespace NUMINAMATH_GPT_factorize_x4_minus_4x2_l1045_104509

theorem factorize_x4_minus_4x2 (x : ℝ) : 
  x^4 - 4 * x^2 = x^2 * (x - 2) * (x + 2) :=
by
  sorry

end NUMINAMATH_GPT_factorize_x4_minus_4x2_l1045_104509


namespace NUMINAMATH_GPT_count_squares_below_graph_l1045_104532

theorem count_squares_below_graph (x y: ℕ) (h : 5 * x + 195 * y = 975) :
  ∃ n : ℕ, n = 388 ∧ 
  ∀ a b : ℕ, 0 ≤ a ∧ a ≤ 195 ∧ 0 ≤ b ∧ b ≤ 5 →
    1 * a + 1 * b < 195 * 5 →
    n = 388 := 
sorry

end NUMINAMATH_GPT_count_squares_below_graph_l1045_104532


namespace NUMINAMATH_GPT_abs_inequality_solution_l1045_104528

theorem abs_inequality_solution (x : ℝ) : 
  3 < |x + 2| ∧ |x + 2| ≤ 6 ↔ (1 < x ∧ x ≤ 4) ∨ (-8 ≤ x ∧ x < -5) := 
by
  sorry

end NUMINAMATH_GPT_abs_inequality_solution_l1045_104528


namespace NUMINAMATH_GPT_cosine_identity_l1045_104579

theorem cosine_identity (α : ℝ) (h : Real.sin α = 1 / 3) : Real.cos (π / 2 + α) = -1 / 3 := by
  sorry

end NUMINAMATH_GPT_cosine_identity_l1045_104579


namespace NUMINAMATH_GPT_tan_five_pi_over_four_l1045_104541

theorem tan_five_pi_over_four : Real.tan (5 * Real.pi / 4) = 1 := by
  sorry

end NUMINAMATH_GPT_tan_five_pi_over_four_l1045_104541


namespace NUMINAMATH_GPT_largest_class_is_28_l1045_104548

-- definition and conditions
def largest_class_students (x : ℕ) : Prop :=
  let total_students := x + (x - 2) + (x - 4) + (x - 6) + (x - 8)
  total_students = 120

-- statement to prove
theorem largest_class_is_28 : ∃ x : ℕ, largest_class_students x ∧ x = 28 :=
by
  sorry

end NUMINAMATH_GPT_largest_class_is_28_l1045_104548


namespace NUMINAMATH_GPT_insurance_slogan_equivalence_l1045_104524

variables (H I : Prop)

theorem insurance_slogan_equivalence :
  (∀ x, x → H → I) ↔ (∀ y, y → ¬I → ¬H) :=
sorry

end NUMINAMATH_GPT_insurance_slogan_equivalence_l1045_104524


namespace NUMINAMATH_GPT_angle_ABC_is_83_l1045_104563

-- Define a structure for the quadrilateral ABCD 
structure Quadrilateral (A B C D : Type) :=
  (angle_BAC : ℝ) -- Measure in degrees
  (angle_CAD : ℝ) -- Measure in degrees
  (angle_ACD : ℝ) -- Measure in degrees
  (side_AB : ℝ) -- Lengths of sides
  (side_AD : ℝ)
  (side_AC : ℝ)

-- Define the conditions from the problem
variable {A B C D : Type}
variable (quad : Quadrilateral A B C D)
variable (h1 : quad.angle_BAC = 60)
variable (h2 : quad.angle_CAD = 60)
variable (h3 : quad.angle_ACD = 23)
variable (h4 : quad.side_AB + quad.side_AD = quad.side_AC)

-- State the theorem to be proved
theorem angle_ABC_is_83 : quad.angle_ACD = 23 → quad.angle_CAD = 60 → 
                           quad.angle_BAC = 60 → quad.side_AB + quad.side_AD = quad.side_AC → 
                           ∃ angle_ABC : ℝ, angle_ABC = 83 := by
  sorry

end NUMINAMATH_GPT_angle_ABC_is_83_l1045_104563


namespace NUMINAMATH_GPT_arithmetic_sequence_sum_l1045_104558

variable {a : ℕ → ℝ}

theorem arithmetic_sequence_sum (h1 : a 2 + a 3 = 2) (h2 : a 4 + a 5 = 6) : a 5 + a 6 = 8 :=
sorry

end NUMINAMATH_GPT_arithmetic_sequence_sum_l1045_104558


namespace NUMINAMATH_GPT_prove_math_problem_l1045_104535

noncomputable def math_problem (x y : ℝ) (h1 : x^2 = x + 1) (h2 : y^2 = y + 1) (h3 : x ≠ y) : Prop :=
  (x + y = 1) ∧ (x^5 + y^5 = 11)

theorem prove_math_problem (x y : ℝ) (h1 : x^2 = x + 1) (h2 : y^2 = y + 1) (h3 : x ≠ y) : math_problem x y h1 h2 h3 :=
  sorry

end NUMINAMATH_GPT_prove_math_problem_l1045_104535


namespace NUMINAMATH_GPT_remaining_nails_after_repairs_l1045_104583

def fraction_used (perc : ℤ) (total : ℤ) : ℤ :=
  (total * perc) / 100

def after_kitchen (nails : ℤ) : ℤ :=
  nails - fraction_used 35 nails

def after_fence (nails : ℤ) : ℤ :=
  let remaining := after_kitchen nails
  remaining - fraction_used 75 remaining

def after_table (nails : ℤ) : ℤ :=
  let remaining := after_fence nails
  remaining - fraction_used 55 remaining

def after_floorboard (nails : ℤ) : ℤ :=
  let remaining := after_table nails
  remaining - fraction_used 30 remaining

theorem remaining_nails_after_repairs :
  after_floorboard 400 = 21 :=
by
  sorry

end NUMINAMATH_GPT_remaining_nails_after_repairs_l1045_104583


namespace NUMINAMATH_GPT_calculate_total_earnings_l1045_104567

theorem calculate_total_earnings :
  let num_floors := 10
  let rooms_per_floor := 20
  let hours_per_room := 8
  let earnings_per_hour := 20
  let total_rooms := num_floors * rooms_per_floor
  let total_hours := total_rooms * hours_per_room
  let total_earnings := total_hours * earnings_per_hour
  total_earnings = 32000 := by sorry

end NUMINAMATH_GPT_calculate_total_earnings_l1045_104567


namespace NUMINAMATH_GPT_largest_divisor_of_n_l1045_104582

theorem largest_divisor_of_n (n : ℕ) (hn : n > 0) (h : 72 ∣ n^2) : 12 ∣ n :=
by sorry

end NUMINAMATH_GPT_largest_divisor_of_n_l1045_104582


namespace NUMINAMATH_GPT_concentric_circles_ratio_l1045_104565

theorem concentric_circles_ratio (d1 d2 d3 : ℝ) (h1 : d1 = 2) (h2 : d2 = 4) (h3 : d3 = 6) :
  let r1 := d1 / 2
  let r2 := d2 / 2
  let r3 := d3 / 2
  let A_red := π * r1 ^ 2
  let A_middle := π * r2 ^ 2
  let A_large := π * r3 ^ 2
  let A_blue := A_middle - A_red
  let A_green := A_large - A_middle
  (A_green / A_blue) = 5 / 3 := 
by
  sorry

end NUMINAMATH_GPT_concentric_circles_ratio_l1045_104565


namespace NUMINAMATH_GPT_average_speed_is_75_l1045_104568

-- Define the conditions
def speed_first_hour : ℕ := 90
def speed_second_hour : ℕ := 60
def total_time : ℕ := 2

-- Define the average speed and prove it is equal to the given answer
theorem average_speed_is_75 : 
  (speed_first_hour + speed_second_hour) / total_time = 75 := 
by 
  -- We will skip the proof for now
  sorry

end NUMINAMATH_GPT_average_speed_is_75_l1045_104568


namespace NUMINAMATH_GPT_triangle_base_length_l1045_104571

theorem triangle_base_length (h : ℝ) (A : ℝ) (b : ℝ) (h_eq : h = 6) (A_eq : A = 13.5) (area_eq : A = (b * h) / 2) : b = 4.5 :=
by
  sorry

end NUMINAMATH_GPT_triangle_base_length_l1045_104571


namespace NUMINAMATH_GPT_leonardo_sleep_fraction_l1045_104576

theorem leonardo_sleep_fraction (h : 60 ≠ 0) : (12 / 60 : ℚ) = (1 / 5 : ℚ) :=
by
  sorry

end NUMINAMATH_GPT_leonardo_sleep_fraction_l1045_104576


namespace NUMINAMATH_GPT_rain_is_random_event_l1045_104581

def is_random_event (p : ℝ) : Prop := p > 0 ∧ p < 1

theorem rain_is_random_event (p : ℝ) (h : p = 0.75) : is_random_event p :=
by
  -- Here we will provide the necessary proof eventually.
  sorry

end NUMINAMATH_GPT_rain_is_random_event_l1045_104581


namespace NUMINAMATH_GPT_some_zen_not_cen_l1045_104529

variable {Zen Ben Cen : Type}
variables (P Q R : Zen → Prop)

theorem some_zen_not_cen (h1 : ∀ x, P x → Q x)
                        (h2 : ∃ x, Q x ∧ ¬ (R x)) :
  ∃ x, P x ∧ ¬ (R x) :=
  sorry

end NUMINAMATH_GPT_some_zen_not_cen_l1045_104529


namespace NUMINAMATH_GPT_percentage_increase_l1045_104536

theorem percentage_increase (original new : ℕ) (h₀ : original = 60) (h₁ : new = 120) :
  ((new - original) / original) * 100 = 100 := by
  sorry

end NUMINAMATH_GPT_percentage_increase_l1045_104536


namespace NUMINAMATH_GPT_find_x_l1045_104556

-- Definitions based on conditions
variables (A B C M O : Type)
variables (OA OB OC OM : vector_space O)
variables (x : ℚ) -- Rational number type for x

-- Condition (1): M lies in the plane ABC
-- Condition (2): OM = x * OA + 1/3 * OB + 1/2 * OC
axiom H : OM = x • OA + (1 / 3 : ℚ) • OB + (1 / 2 : ℚ) • OC

-- The theorem statement
theorem find_x :
  x = 1 / 6 :=
sorry -- Proof is to be provided

end NUMINAMATH_GPT_find_x_l1045_104556


namespace NUMINAMATH_GPT_swimming_pool_length_l1045_104547

noncomputable def solveSwimmingPoolLength : ℕ :=
  let w_pool := 22
  let w_deck := 3
  let total_area := 728
  let total_width := w_pool + 2 * w_deck
  let L := (total_area / total_width) - 2 * w_deck
  L

theorem swimming_pool_length : solveSwimmingPoolLength = 20 := 
  by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_swimming_pool_length_l1045_104547


namespace NUMINAMATH_GPT_average_inside_time_l1045_104598

def jonsey_awake_hours := 24 * (2/3)
def jonsey_inside_fraction := 1 - (1/2)
def jonsey_inside_hours := jonsey_awake_hours * jonsey_inside_fraction

def riley_awake_hours := 24 * (3/4)
def riley_inside_fraction := 1 - (1/3)
def riley_inside_hours := riley_awake_hours * riley_inside_fraction

def total_inside_hours := jonsey_inside_hours + riley_inside_hours
def number_of_people := 2
def average_inside_hours := total_inside_hours / number_of_people

theorem average_inside_time (jonsey_awake_hrs : ℝ) (jonsey_inside_frac : ℝ) 
  (jonsey_inside_hrs : ℝ) (riley_awake_hrs : ℝ) (riley_inside_frac : ℝ) 
  (riley_inside_hrs : ℝ) (total_inside_hrs : ℝ) (num_people : ℝ) 
  (avg_inside_hrs : ℝ) :
  jonsey_awake_hrs = 24 * (2 / 3) → 
  jonsey_inside_frac = 1 - (1 / 2) →
  jonsey_inside_hrs = jonsey_awake_hrs * jonsey_inside_frac →
  riley_awake_hrs = 24 * (3 / 4) →
  riley_inside_frac = 1 - (1 / 3) →
  riley_inside_hrs = riley_awake_hrs * riley_inside_frac →
  total_inside_hrs = jonsey_inside_hrs + riley_inside_hrs →
  num_people = 2 →
  avg_inside_hrs = total_inside_hrs / num_people →
  avg_inside_hrs = 10 := 
by
  intros
  sorry

end NUMINAMATH_GPT_average_inside_time_l1045_104598


namespace NUMINAMATH_GPT_fifth_boy_pays_l1045_104575

def problem_conditions (a b c d e : ℝ) : Prop :=
  d = 20 ∧
  a = (1 / 3) * (b + c + d + e) ∧
  b = (1 / 4) * (a + c + d + e) ∧
  c = (1 / 5) * (a + b + d + e) ∧
  a + b + c + d + e = 120 

theorem fifth_boy_pays (a b c d e : ℝ) (h : problem_conditions a b c d e) : 
  e = 35 :=
sorry

end NUMINAMATH_GPT_fifth_boy_pays_l1045_104575


namespace NUMINAMATH_GPT_m_range_positive_solution_l1045_104521

theorem m_range_positive_solution (m : ℝ) : (∃ x : ℝ, x > 0 ∧ (2 * x + m) / (x - 2) + (x - 1) / (2 - x) = 3) ↔ (m > -7 ∧ m ≠ -3) := by
  sorry

end NUMINAMATH_GPT_m_range_positive_solution_l1045_104521


namespace NUMINAMATH_GPT_alice_age_30_l1045_104515

variable (A T : ℕ)

def tom_younger_alice (A T : ℕ) := T = A - 15
def ten_years_ago (A T : ℕ) := A - 10 = 4 * (T - 10)

theorem alice_age_30 (A T : ℕ) (h1 : tom_younger_alice A T) (h2 : ten_years_ago A T) : A = 30 := 
by sorry

end NUMINAMATH_GPT_alice_age_30_l1045_104515


namespace NUMINAMATH_GPT_solution_set_inequality_l1045_104593

theorem solution_set_inequality (x : ℝ) :
  ((x + (1 / 2)) * ((3 / 2) - x) ≥ 0) ↔ (- (1 / 2) ≤ x ∧ x ≤ (3 / 2)) :=
by sorry

end NUMINAMATH_GPT_solution_set_inequality_l1045_104593


namespace NUMINAMATH_GPT_no_real_roots_of_quad_eq_l1045_104552

theorem no_real_roots_of_quad_eq (k : ℝ) : ¬(k ≠ 0 ∧ ∃ x : ℝ, x^2 + k * x + 2 * k^2 = 0) :=
by
  sorry

end NUMINAMATH_GPT_no_real_roots_of_quad_eq_l1045_104552


namespace NUMINAMATH_GPT_quadratic_to_standard_form_div_l1045_104514

theorem quadratic_to_standard_form_div (b c : ℤ)
  (h : ∀ x : ℤ, x^2 - 2100 * x - 8400 = (x + b)^2 + c) :
  c / b = 1058 :=
sorry

end NUMINAMATH_GPT_quadratic_to_standard_form_div_l1045_104514


namespace NUMINAMATH_GPT_arithmetic_sequence_angles_sum_l1045_104526

theorem arithmetic_sequence_angles_sum (A B C : ℝ) (h₁ : A + B + C = 180) (h₂ : 2 * B = A + C) :
  A + C = 120 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_angles_sum_l1045_104526


namespace NUMINAMATH_GPT_cost_of_gravelling_the_path_l1045_104519

-- Define the problem conditions
def plot_length : ℝ := 110
def plot_width : ℝ := 65
def path_width : ℝ := 2.5
def cost_per_sq_meter : ℝ := 0.70

-- Define the dimensions of the grassy area without the path
def grassy_length : ℝ := plot_length - 2 * path_width
def grassy_width : ℝ := plot_width - 2 * path_width

-- Define the area of the entire plot and the grassy area without the path
def area_entire_plot : ℝ := plot_length * plot_width
def area_grassy_area : ℝ := grassy_length * grassy_width

-- Define the area of the path
def area_path : ℝ := area_entire_plot - area_grassy_area

-- Define the cost of gravelling the path
def cost_gravelling_path : ℝ := area_path * cost_per_sq_meter

-- State the theorem
theorem cost_of_gravelling_the_path : cost_gravelling_path = 595 := 
by
  -- The proof is omitted
  sorry

end NUMINAMATH_GPT_cost_of_gravelling_the_path_l1045_104519


namespace NUMINAMATH_GPT_people_on_williams_bus_l1045_104522

theorem people_on_williams_bus
  (P : ℕ)
  (dutch_people : ℕ)
  (dutch_americans : ℕ)
  (window_seats : ℕ)
  (h1 : dutch_people = (3 * P) / 5)
  (h2 : dutch_americans = dutch_people / 2)
  (h3 : window_seats = dutch_americans / 3)
  (h4 : window_seats = 9) : 
  P = 90 :=
sorry

end NUMINAMATH_GPT_people_on_williams_bus_l1045_104522


namespace NUMINAMATH_GPT_avg_age_of_14_students_l1045_104596

theorem avg_age_of_14_students (avg_age_25 : ℕ) (avg_age_10 : ℕ) (age_25th : ℕ) (total_students : ℕ) (remaining_students : ℕ) :
  avg_age_25 = 25 →
  avg_age_10 = 22 →
  age_25th = 13 →
  total_students = 25 →
  remaining_students = 14 →
  ( (total_students * avg_age_25) - (10 * avg_age_10) - age_25th ) / remaining_students = 28 :=
by
  intros
  sorry

end NUMINAMATH_GPT_avg_age_of_14_students_l1045_104596


namespace NUMINAMATH_GPT_original_student_count_l1045_104573

variable (A B C N D : ℕ)
variable (hA : A = 40)
variable (hB : B = 32)
variable (hC : C = 36)
variable (hD : D = N * A)
variable (hNewSum : D + 8 * B = (N + 8) * C)

theorem original_student_count (hA : A = 40) (hB : B = 32) (hC : C = 36) (hD : D = N * A) (hNewSum : D + 8 * B = (N + 8) * C) : 
  N = 8 :=
by
  sorry

end NUMINAMATH_GPT_original_student_count_l1045_104573


namespace NUMINAMATH_GPT_correct_articles_l1045_104572

-- Define the given conditions
def specific_experience : Prop := true
def countable_noun : Prop := true

-- Problem statement: given the conditions, choose the correct articles to fill in the blanks
theorem correct_articles (h1 : specific_experience) (h2 : countable_noun) : 
  "the; a" = "the; a" :=
by
  sorry

end NUMINAMATH_GPT_correct_articles_l1045_104572


namespace NUMINAMATH_GPT_emma_time_l1045_104599

theorem emma_time (E : ℝ) (h1 : 2 * E + E = 60) : E = 20 :=
sorry

end NUMINAMATH_GPT_emma_time_l1045_104599


namespace NUMINAMATH_GPT_B_days_finish_work_l1045_104505

theorem B_days_finish_work :
  ∀ (W : ℝ) (A_work B_work B_days : ℝ),
  (A_work = W / 9) → 
  (B_work = W / B_days) →
  (3 * (W / 9) + 10 * (W / B_days) = W) →
  B_days = 15 :=
by
  intros W A_work B_work B_days hA_work hB_work hTotal
  sorry

end NUMINAMATH_GPT_B_days_finish_work_l1045_104505


namespace NUMINAMATH_GPT_find_x_solution_l1045_104533

theorem find_x_solution (x b c : ℝ) (h_eq : x^2 + c^2 = (b - x)^2):
  x = (b^2 - c^2) / (2 * b) :=
sorry

end NUMINAMATH_GPT_find_x_solution_l1045_104533


namespace NUMINAMATH_GPT_find_x_satisfying_floor_eq_l1045_104570

theorem find_x_satisfying_floor_eq (x : ℝ) (hx: ⌊x⌋ * x = 152) : x = 38 / 3 :=
sorry

end NUMINAMATH_GPT_find_x_satisfying_floor_eq_l1045_104570


namespace NUMINAMATH_GPT_sequence_positions_l1045_104513

noncomputable def position_of_a4k1 (x : ℕ) : ℕ := 4 * x + 1
noncomputable def position_of_a4k2 (x : ℕ) : ℕ := 4 * x + 2
noncomputable def position_of_a4k3 (x : ℕ) : ℕ := 4 * x + 3
noncomputable def position_of_a4k (x : ℕ) : ℕ := 4 * x

theorem sequence_positions (k : ℕ) :
  (6 + 1964 = 1970 ∧ position_of_a4k1 1964 = 7857) ∧
  (6 + 1965 = 1971 ∧ position_of_a4k1 1965 = 7861) ∧
  (8 + 1962 = 1970 ∧ position_of_a4k2 1962 = 7850) ∧
  (8 + 1963 = 1971 ∧ position_of_a4k2 1963 = 7854) ∧
  (16 + 2 * 977 = 1970 ∧ position_of_a4k3 977 = 3911) ∧
  (14 + 2 * (979 - 1) = 1970 ∧ position_of_a4k 979 = 3916) :=
by sorry

end NUMINAMATH_GPT_sequence_positions_l1045_104513


namespace NUMINAMATH_GPT_Lizette_average_above_94_l1045_104577

noncomputable def Lizette_new_weighted_average
  (score3: ℝ) (avg3: ℝ) (weight3: ℝ) (score_new1 score_new2: ℝ) (weight_new: ℝ) :=
  let total_points3 := avg3 * 3
  let total_weight3 := 3 * weight3
  let total_points := total_points3 + score_new1 + score_new2
  let total_weight := total_weight3 + 2 * weight_new
  total_points / total_weight

theorem Lizette_average_above_94:
  ∀ (score3 avg3 weight3 score_new1 score_new2 weight_new: ℝ),
  score3 = 92 →
  avg3 = 94 →
  weight3 = 0.15 →
  score_new1 > 94 →
  score_new2 > 94 →
  weight_new = 0.20 →
  Lizette_new_weighted_average score3 avg3 weight3 score_new1 score_new2 weight_new > 94 :=
by
  intros score3 avg3 weight3 score_new1 score_new2 weight_new h1 h2 h3 h4 h5 h6
  sorry

end NUMINAMATH_GPT_Lizette_average_above_94_l1045_104577


namespace NUMINAMATH_GPT_juniors_score_l1045_104511

theorem juniors_score (n : ℕ) (j s : ℕ) (avg_score students_avg seniors_avg : ℕ)
  (h1 : 0 < n)
  (h2 : j = n / 5)
  (h3 : s = 4 * n / 5)
  (h4 : avg_score = 80)
  (h5 : seniors_avg = 78)
  (h6 : students_avg = avg_score)
  (h7 : n * students_avg = n * avg_score)
  (h8 : s * seniors_avg = 78 * s) :
  (800 - 624) / j = 88 := by
  sorry

end NUMINAMATH_GPT_juniors_score_l1045_104511


namespace NUMINAMATH_GPT_union_of_A_and_B_l1045_104561

open Set

def A : Set ℕ := {2, 5, 6}
def B : Set ℕ := {3, 5}

theorem union_of_A_and_B : A ∪ B = {2, 3, 5, 6} := sorry

end NUMINAMATH_GPT_union_of_A_and_B_l1045_104561


namespace NUMINAMATH_GPT_jason_total_hours_l1045_104585

variables (hours_after_school hours_total : ℕ)

def earnings_after_school := 4 * hours_after_school
def earnings_saturday := 6 * 8
def total_earnings := earnings_after_school + earnings_saturday

theorem jason_total_hours :
  4 * hours_after_school + earnings_saturday = 88 →
  hours_total = hours_after_school + 8 →
  total_earnings = 88 →
  hours_total = 18 :=
by
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_jason_total_hours_l1045_104585


namespace NUMINAMATH_GPT_arrangements_15_cents_l1045_104502

def numArrangements (n : ℕ) : ℕ :=
  sorry  -- Function definition which outputs the number of arrangements for sum n

theorem arrangements_15_cents : numArrangements 15 = X :=
  sorry  -- Replace X with the correct calculated number

end NUMINAMATH_GPT_arrangements_15_cents_l1045_104502


namespace NUMINAMATH_GPT_complement_intersection_is_correct_l1045_104588

open Set

noncomputable def U : Set ℕ := {0, 1, 2, 3, 4, 5, 6}
noncomputable def A : Set ℕ := {1, 2}
noncomputable def B : Set ℕ := {0, 2, 5}
noncomputable def complementA := (U \ A)

theorem complement_intersection_is_correct :
  complementA ∩ B = {0, 5} :=
by
  sorry

end NUMINAMATH_GPT_complement_intersection_is_correct_l1045_104588


namespace NUMINAMATH_GPT_totalCorrectQuestions_l1045_104597

-- Definitions for the conditions
def mathQuestions : ℕ := 40
def mathCorrectPercentage : ℕ := 75
def englishQuestions : ℕ := 50
def englishCorrectPercentage : ℕ := 98

-- Function to calculate the number of correctly answered questions
def correctQuestions (totalQuestions : ℕ) (percentage : ℕ) : ℕ :=
  (percentage * totalQuestions) / 100

-- Main theorem to prove the total number of correct questions
theorem totalCorrectQuestions : 
  correctQuestions mathQuestions mathCorrectPercentage +
  correctQuestions englishQuestions englishCorrectPercentage = 79 :=
by
  sorry

end NUMINAMATH_GPT_totalCorrectQuestions_l1045_104597


namespace NUMINAMATH_GPT_counties_under_50k_perc_l1045_104534

def percentage (s: String) : ℝ := match s with
  | "20k_to_49k" => 45
  | "less_than_20k" => 30
  | _ => 0

theorem counties_under_50k_perc : percentage "20k_to_49k" + percentage "less_than_20k" = 75 := by
  sorry

end NUMINAMATH_GPT_counties_under_50k_perc_l1045_104534


namespace NUMINAMATH_GPT_swimming_speed_in_still_water_l1045_104574

theorem swimming_speed_in_still_water (v : ℝ) 
  (h_current_speed : 2 = 2) 
  (h_time_distance : 7 = 7) 
  (h_effective_speed : v - 2 = 14 / 7) : 
  v = 4 :=
sorry

end NUMINAMATH_GPT_swimming_speed_in_still_water_l1045_104574


namespace NUMINAMATH_GPT_jason_money_determination_l1045_104589

theorem jason_money_determination (fred_last_week : ℕ) (fred_earned : ℕ) (fred_now : ℕ) (jason_last_week : ℕ → Prop)
  (h1 : fred_last_week = 23)
  (h2 : fred_earned = 63)
  (h3 : fred_now = 86) :
  ¬ ∃ x, jason_last_week x :=
by
  sorry

end NUMINAMATH_GPT_jason_money_determination_l1045_104589


namespace NUMINAMATH_GPT_speed_of_second_train_l1045_104507

theorem speed_of_second_train
  (t₁ : ℕ := 2)  -- Time the first train sets off (2:00 pm in hours)
  (s₁ : ℝ := 70) -- Speed of the first train in km/h
  (t₂ : ℕ := 3)  -- Time the second train sets off (3:00 pm in hours)
  (t₃ : ℕ := 10) -- Time when the second train catches the first train (10:00 pm in hours)
  : ∃ S : ℝ, S = 80 := sorry

end NUMINAMATH_GPT_speed_of_second_train_l1045_104507


namespace NUMINAMATH_GPT_seq_nth_term_2009_l1045_104551

theorem seq_nth_term_2009 (n x : ℤ) (h : 2 * x - 3 = 5 ∧ 5 * x - 11 = 9 ∧ 3 * x + 1 = 13) :
  n = 502 ↔ 2009 = (2 * x - 3) + (n - 1) * ((5 * x - 11) - (2 * x - 3)) :=
sorry

end NUMINAMATH_GPT_seq_nth_term_2009_l1045_104551


namespace NUMINAMATH_GPT_largest_of_a_b_c_d_e_l1045_104544

theorem largest_of_a_b_c_d_e (a b c d e : ℝ)
  (h1 : a - 2 = b + 3)
  (h2 : a - 2 = c - 4)
  (h3 : a - 2 = d + 5)
  (h4 : a - 2 = e - 6) :
  e > a ∧ e > b ∧ e > c ∧ e > d :=
by
  sorry

end NUMINAMATH_GPT_largest_of_a_b_c_d_e_l1045_104544


namespace NUMINAMATH_GPT_major_axis_length_l1045_104508

theorem major_axis_length (x y : ℝ) (h : 16 * x^2 + 9 * y^2 = 144) : 8 = 8 :=
by sorry

end NUMINAMATH_GPT_major_axis_length_l1045_104508


namespace NUMINAMATH_GPT_difference_of_integers_l1045_104554

theorem difference_of_integers :
  ∀ (x y : ℤ), (x = 32) → (y = 5*x + 2) → (y - x = 130) :=
by
  intros x y hx hy
  sorry

end NUMINAMATH_GPT_difference_of_integers_l1045_104554


namespace NUMINAMATH_GPT_value_of_product_of_sums_of_roots_l1045_104550

theorem value_of_product_of_sums_of_roots 
    (a b c : ℂ)
    (h1 : a + b + c = 15)
    (h2 : a * b + b * c + c * a = 22)
    (h3 : a * b * c = 8) :
    (1 + a) * (1 + b) * (1 + c) = 46 := by
  sorry

end NUMINAMATH_GPT_value_of_product_of_sums_of_roots_l1045_104550


namespace NUMINAMATH_GPT_valid_triples_l1045_104587

theorem valid_triples :
  ∀ (a b c : ℕ), 1 ≤ a → 1 ≤ b → 1 ≤ c →
  (∃ k : ℕ, 32 * a + 3 * b + 48 * c = 4 * k * a * b * c) ↔ 
  (a = 1 ∧ b = 20 ∧ c = 1) ∨ (a = 1 ∧ b = 4 ∧ c = 1) ∨ (a = 3 ∧ b = 4 ∧ c = 1) := 
by
  sorry

end NUMINAMATH_GPT_valid_triples_l1045_104587


namespace NUMINAMATH_GPT_max_length_polyline_l1045_104542

-- Definition of the grid and problem
def grid_rows : ℕ := 6
def grid_cols : ℕ := 10

-- The maximum length of a closed, non-self-intersecting polyline
theorem max_length_polyline (rows cols : ℕ) 
  (h_rows : rows = grid_rows) (h_cols : cols = grid_cols) :
  ∃ length : ℕ, length = 76 :=
by {
  sorry
}

end NUMINAMATH_GPT_max_length_polyline_l1045_104542


namespace NUMINAMATH_GPT_problem_1_problem_2_l1045_104525

open Real

-- Part 1
theorem problem_1 (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a + b = 4) : 
  (1 / a + 1 / (b + 1) ≥ 4 / 5) :=
sorry

-- Part 2
theorem problem_2 : 
  ∃ (a b : ℝ), 0 < a ∧ 0 < b ∧ a + b = 4 ∧ (4 / (a * b) + a / b = (1 + sqrt 5) / 2) :=
sorry

end NUMINAMATH_GPT_problem_1_problem_2_l1045_104525


namespace NUMINAMATH_GPT_Betty_will_pay_zero_l1045_104562

-- Definitions of the conditions
def Doug_age : ℕ := 40
def Alice_age (D : ℕ) : ℕ := D / 2
def Betty_age (B D A : ℕ) : Prop := B + D + A = 130
def Cost_of_pack_of_nuts (C B : ℕ) : Prop := C = 2 * B
def Decrease_rate : ℕ := 5
def New_cost (C B A : ℕ) : ℕ := max 0 (C - (B - A) * Decrease_rate)
def Total_cost (packs cost_per_pack: ℕ) : ℕ := packs * cost_per_pack

-- The main proposition
theorem Betty_will_pay_zero :
  ∃ B A C, 
    (C = 2 * B) ∧
    (A = Doug_age / 2) ∧
    (B + Doug_age + A = 130) ∧
    (Total_cost 20 (max 0 (C - (B - A) * Decrease_rate)) = 0) :=
by sorry

end NUMINAMATH_GPT_Betty_will_pay_zero_l1045_104562


namespace NUMINAMATH_GPT_power_sum_l1045_104566

theorem power_sum : 2^4 + 2^4 + 2^5 + 2^5 = 96 := 
by
  sorry

end NUMINAMATH_GPT_power_sum_l1045_104566


namespace NUMINAMATH_GPT_remainder_when_divided_by_22_l1045_104580

theorem remainder_when_divided_by_22 (n : ℤ) (h : (2 * n) % 11 = 2) : n % 22 = 1 :=
by
  sorry

end NUMINAMATH_GPT_remainder_when_divided_by_22_l1045_104580


namespace NUMINAMATH_GPT_temperature_at_midnight_l1045_104578

theorem temperature_at_midnight :
  ∀ (morning_temp noon_rise midnight_drop midnight_temp : ℤ),
    morning_temp = -3 →
    noon_rise = 6 →
    midnight_drop = -7 →
    midnight_temp = morning_temp + noon_rise + midnight_drop →
    midnight_temp = -4 :=
by
  intros
  sorry

end NUMINAMATH_GPT_temperature_at_midnight_l1045_104578


namespace NUMINAMATH_GPT_compound_proposition_p_or_q_l1045_104586

theorem compound_proposition_p_or_q : 
  (∃ (n : ℝ), ∀ (m : ℝ), m * n = m) ∨ 
  (∀ (n : ℝ), ∃ (m : ℝ), m^2 < n) := 
by
  sorry

end NUMINAMATH_GPT_compound_proposition_p_or_q_l1045_104586


namespace NUMINAMATH_GPT_average_payment_correct_l1045_104538

-- Definitions based on conditions in the problem
def first_payments_num : ℕ := 20
def first_payment_amount : ℕ := 450

def second_payments_num : ℕ := 30
def increment_after_first : ℕ := 80

def third_payments_num : ℕ := 40
def increment_after_second : ℕ := 65

def fourth_payments_num : ℕ := 50
def increment_after_third : ℕ := 105

def fifth_payments_num : ℕ := 60
def increment_after_fourth : ℕ := 95

def total_payments : ℕ := first_payments_num + second_payments_num + third_payments_num + fourth_payments_num + fifth_payments_num

-- Function to calculate total paid amount
def total_amount_paid : ℕ :=
  (first_payments_num * first_payment_amount) +
  (second_payments_num * (first_payment_amount + increment_after_first)) +
  (third_payments_num * (first_payment_amount + increment_after_first + increment_after_second)) +
  (fourth_payments_num * (first_payment_amount + increment_after_first + increment_after_second + increment_after_third)) +
  (fifth_payments_num * (first_payment_amount + increment_after_first + increment_after_second + increment_after_third + increment_after_fourth))

-- Function to calculate average payment
def average_payment : ℕ := total_amount_paid / total_payments

-- The theorem to be proved
theorem average_payment_correct : average_payment = 657 := by
  sorry

end NUMINAMATH_GPT_average_payment_correct_l1045_104538


namespace NUMINAMATH_GPT_values_of_x_l1045_104518

theorem values_of_x (x : ℝ) : (-2 < x ∧ x < 2) ↔ (x^2 < |x| + 2) := by
  sorry

end NUMINAMATH_GPT_values_of_x_l1045_104518


namespace NUMINAMATH_GPT_percentage_deducted_from_list_price_l1045_104545

noncomputable def cost_price : ℝ := 47.50
noncomputable def list_price : ℝ := 65.97
noncomputable def selling_price : ℝ := 65.97
noncomputable def required_profit_percent : ℝ := 25

theorem percentage_deducted_from_list_price :
  let desired_selling_price := cost_price * (1 + required_profit_percent / 100)
  let discount_percentage := 100 * (1 - desired_selling_price / list_price)
  discount_percentage = 10.02 :=
by
  sorry

end NUMINAMATH_GPT_percentage_deducted_from_list_price_l1045_104545


namespace NUMINAMATH_GPT_find_normal_monthly_charge_l1045_104569

-- Define the conditions
def normal_monthly_charge (x : ℕ) : Prop :=
  let first_month_charge := x / 3
  let fourth_month_charge := x + 15
  let other_months_charge := 4 * x
  (first_month_charge + fourth_month_charge + other_months_charge = 175)

-- The statement to prove
theorem find_normal_monthly_charge : ∃ x : ℕ, normal_monthly_charge x ∧ x = 30 := by
  sorry

end NUMINAMATH_GPT_find_normal_monthly_charge_l1045_104569


namespace NUMINAMATH_GPT_franks_daily_reading_l1045_104517

-- Define the conditions
def total_pages : ℕ := 612
def days_to_finish : ℕ := 6

-- State the theorem we want to prove
theorem franks_daily_reading : (total_pages / days_to_finish) = 102 :=
by
  sorry

end NUMINAMATH_GPT_franks_daily_reading_l1045_104517


namespace NUMINAMATH_GPT_solve_purchase_price_problem_l1045_104559

def purchase_price_problem : Prop :=
  ∃ P : ℝ, (0.10 * P + 12 = 35) ∧ (P = 230)

theorem solve_purchase_price_problem : purchase_price_problem :=
  by
    sorry

end NUMINAMATH_GPT_solve_purchase_price_problem_l1045_104559
