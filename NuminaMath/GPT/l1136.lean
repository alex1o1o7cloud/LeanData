import Mathlib

namespace NUMINAMATH_GPT_ellipse_foci_x_axis_l1136_113650

theorem ellipse_foci_x_axis (k : ℝ) : 
  (0 < k ∧ k < 2) ↔ (∃ a b : ℝ, a > 0 ∧ b > 0 ∧ ∀ x y : ℝ, (x^2 / a^2 + y^2 / b^2 = 1) ∧ a > b) := 
sorry

end NUMINAMATH_GPT_ellipse_foci_x_axis_l1136_113650


namespace NUMINAMATH_GPT_div_by_72_l1136_113660

theorem div_by_72 (x : ℕ) (y : ℕ) (h1 : 0 ≤ x ∧ x ≤ 9) (h2 : x = 4)
    (h3 : 0 ≤ y ∧ y ≤ 9) (h4 : y = 6) : 
    72 ∣ (9834800 + 1000 * x + 10 * y) :=
by 
  sorry

end NUMINAMATH_GPT_div_by_72_l1136_113660


namespace NUMINAMATH_GPT_total_savings_percentage_l1136_113627

theorem total_savings_percentage
  (original_coat_price : ℕ) (original_pants_price : ℕ)
  (coat_discount_percent : ℚ) (pants_discount_percent : ℚ)
  (original_total_price : ℕ) (total_savings : ℕ)
  (savings_percentage : ℚ) :
  original_coat_price = 120 →
  original_pants_price = 60 →
  coat_discount_percent = 0.30 →
  pants_discount_percent = 0.60 →
  original_total_price = original_coat_price + original_pants_price →
  total_savings = original_coat_price * coat_discount_percent + original_pants_price * pants_discount_percent →
  savings_percentage = (total_savings / original_total_price) * 100 →
  savings_percentage = 40 := 
by
  intros
  sorry

end NUMINAMATH_GPT_total_savings_percentage_l1136_113627


namespace NUMINAMATH_GPT_negation_proof_l1136_113603

open Real

theorem negation_proof :
  (¬ ∃ x : ℕ, exp x - x - 1 ≤ 0) ↔ (∀ x : ℕ, exp x - x - 1 > 0) :=
by
  sorry

end NUMINAMATH_GPT_negation_proof_l1136_113603


namespace NUMINAMATH_GPT_exponentiation_rule_l1136_113630

theorem exponentiation_rule (x : ℝ) : (x^5)^2 = x^10 :=
by {
  sorry
}

end NUMINAMATH_GPT_exponentiation_rule_l1136_113630


namespace NUMINAMATH_GPT_internal_diagonal_cubes_l1136_113696

-- Define the dimensions of the rectangular solid
def x_dimension : ℕ := 168
def y_dimension : ℕ := 350
def z_dimension : ℕ := 390

-- Define the GCD calculations for the given dimensions
def gcd_xy : ℕ := Nat.gcd x_dimension y_dimension
def gcd_yz : ℕ := Nat.gcd y_dimension z_dimension
def gcd_zx : ℕ := Nat.gcd z_dimension x_dimension
def gcd_xyz : ℕ := Nat.gcd (Nat.gcd x_dimension y_dimension) z_dimension

-- Define a statement that the internal diagonal passes through a certain number of cubes
theorem internal_diagonal_cubes :
  x_dimension + y_dimension + z_dimension - gcd_xy - gcd_yz - gcd_zx + gcd_xyz = 880 :=
by
  -- Configuration of conditions and proof skeleton with sorry
  sorry

end NUMINAMATH_GPT_internal_diagonal_cubes_l1136_113696


namespace NUMINAMATH_GPT_matias_fewer_cards_l1136_113654

theorem matias_fewer_cards (J M C : ℕ) (h1 : J = M) (h2 : C = 20) (h3 : C + M + J = 48) : C - M = 6 :=
by
-- To be proven
  sorry

end NUMINAMATH_GPT_matias_fewer_cards_l1136_113654


namespace NUMINAMATH_GPT_jon_total_cost_l1136_113653
-- Import the complete Mathlib library

-- Define the conditions
def MSRP : ℝ := 30
def insurance_rate : ℝ := 0.20
def tax_rate : ℝ := 0.50

-- Calculate intermediate values based on conditions
noncomputable def insurance_cost : ℝ := insurance_rate * MSRP
noncomputable def subtotal_before_tax : ℝ := MSRP + insurance_cost
noncomputable def state_tax : ℝ := tax_rate * subtotal_before_tax
noncomputable def total_cost : ℝ := subtotal_before_tax + state_tax

-- The theorem we need to prove
theorem jon_total_cost : total_cost = 54 := by
  -- Proof is omitted
  sorry

end NUMINAMATH_GPT_jon_total_cost_l1136_113653


namespace NUMINAMATH_GPT_arithmetic_progression_sum_15_terms_l1136_113679

def arithmetic_progression_sum (a₁ d : ℚ) : ℚ :=
  15 * (2 * a₁ + (15 - 1) * d) / 2

def am_prog3_and_9_sum_and_product (a₁ d : ℚ) : Prop :=
  (a₁ + 2 * d) + (a₁ + 8 * d) = 6 ∧ (a₁ + 2 * d) * (a₁ + 8 * d) = 135 / 16

theorem arithmetic_progression_sum_15_terms (a₁ d : ℚ)
  (h : am_prog3_and_9_sum_and_product a₁ d) :
  arithmetic_progression_sum a₁ d = 37.5 ∨ arithmetic_progression_sum a₁ d = 52.5 :=
sorry

end NUMINAMATH_GPT_arithmetic_progression_sum_15_terms_l1136_113679


namespace NUMINAMATH_GPT_perpendicular_lines_a_value_l1136_113618

theorem perpendicular_lines_a_value :
  (∃ (a : ℝ), ∀ (x y : ℝ), (3 * y + x + 5 = 0) ∧ (4 * y + a * x + 3 = 0) → a = -12) :=
by
  sorry

end NUMINAMATH_GPT_perpendicular_lines_a_value_l1136_113618


namespace NUMINAMATH_GPT_no_integer_solutions_l1136_113626

theorem no_integer_solutions (x y z : ℤ) (h : ¬ (x = 0 ∧ y = 0 ∧ z = 0)) : 2 * x^4 + y^4 ≠ 7 * z^4 :=
sorry

end NUMINAMATH_GPT_no_integer_solutions_l1136_113626


namespace NUMINAMATH_GPT_min_distance_from_origin_to_line_l1136_113647

open Real

theorem min_distance_from_origin_to_line :
    ∀ x y : ℝ, (3 * x + 4 * y - 4 = 0) -> dist (0, 0) (x, y) = 4 / 5 :=
by
  sorry

end NUMINAMATH_GPT_min_distance_from_origin_to_line_l1136_113647


namespace NUMINAMATH_GPT_define_interval_l1136_113605

theorem define_interval (x : ℝ) : 
  (0 < x + 2) → (0 < 5 - x) → (-2 < x ∧ x < 5) :=
by
  intros h1 h2
  sorry

end NUMINAMATH_GPT_define_interval_l1136_113605


namespace NUMINAMATH_GPT_equation1_solution_equation2_solutions_l1136_113666

theorem equation1_solution (x : ℝ) : (x - 2) * (x - 3) = x - 2 → (x = 2 ∨ x = 4) :=
by
  intro h
  have h1 : (x - 2) * (x - 3) - (x - 2) = 0 := by sorry
  have h2 : (x - 2) * (x - 4) = 0 := by sorry
  have h3 : x - 2 = 0 ∨ x - 4 = 0 := by sorry
  cases h3 with
  | inl h4 => left; exact eq_of_sub_eq_zero h4
  | inr h5 => right; exact eq_of_sub_eq_zero h5

theorem equation2_solutions (x : ℝ) : 2 * x^2 - 5 * x + 1 = 0 → (x = (5 + Real.sqrt 17) / 4 ∨ x = (5 - Real.sqrt 17) / 4) :=
by
  intro h
  have h1 : (-5)^2 - 4 * 2 * 1 = 17 := by sorry
  have h2 : 2 * x^2 - 5 * x + 1 = 2 * ((x - (5 + Real.sqrt 17) / 4) * (x - (5 - Real.sqrt 17) / 4)) := by sorry
  have h3 : (x = (5 + Real.sqrt 17) / 4 ∨ x = (5 - Real.sqrt 17) / 4) := by sorry
  exact h3

end NUMINAMATH_GPT_equation1_solution_equation2_solutions_l1136_113666


namespace NUMINAMATH_GPT_generatrix_length_of_cone_l1136_113686

theorem generatrix_length_of_cone (r l : ℝ) (π : ℝ) (sqrt : ℝ → ℝ) 
  (hx : r = sqrt 2)
  (h_baseline_length : 2 * π * r = π * l) : 
  l = 2 * sqrt 2 :=
by
  sorry

end NUMINAMATH_GPT_generatrix_length_of_cone_l1136_113686


namespace NUMINAMATH_GPT_sum_first_11_terms_l1136_113632

-- Define the arithmetic sequence and sum formula
def arithmetic_sequence (a1 d : ℤ) (n : ℕ) : ℤ :=
  a1 + (n - 1) * d

def sum_arithmetic_sequence (a1 d : ℤ) (n : ℕ) : ℤ :=
  n * (2 * a1 + (n - 1) * d) / 2

-- Conditions given
variables (a1 d : ℤ)
axiom condition : (a1 + d) + (a1 + 9 * d) = 4

-- Proof statement
theorem sum_first_11_terms : sum_arithmetic_sequence a1 d 11 = 22 :=
by
  -- Placeholder for the actual proof
  sorry

end NUMINAMATH_GPT_sum_first_11_terms_l1136_113632


namespace NUMINAMATH_GPT_find_math_marks_l1136_113629

theorem find_math_marks
  (e p c b : ℕ)
  (n : ℕ)
  (a : ℚ)
  (M : ℕ) :
  e = 96 →
  p = 82 →
  c = 87 →
  b = 92 →
  n = 5 →
  a = 90.4 →
  (a * n = (e + p + c + b + M)) →
  M = 95 :=
by intros
   sorry

end NUMINAMATH_GPT_find_math_marks_l1136_113629


namespace NUMINAMATH_GPT_root_interval_sum_l1136_113698

theorem root_interval_sum (a b : Int) (h1 : b - a = 1) (h2 : ∃ x, a < x ∧ x < b ∧ (x^3 - x + 1) = 0) : a + b = -3 := 
sorry

end NUMINAMATH_GPT_root_interval_sum_l1136_113698


namespace NUMINAMATH_GPT_seeds_in_first_plot_l1136_113622

theorem seeds_in_first_plot (x : ℕ) (h1 : 0 < x)
  (h2 : 200 = 200)
  (h3 : 0.25 * (x : ℝ) = 0.25 * (x : ℝ))
  (h4 : 0.35 * 200 = 70)
  (h5 : (0.25 * (x : ℝ) + 70) / (x + 200) = 0.29) :
  x = 300 :=
by sorry

end NUMINAMATH_GPT_seeds_in_first_plot_l1136_113622


namespace NUMINAMATH_GPT_findDivisor_l1136_113661

def addDivisorProblem : Prop :=
  ∃ d : ℕ, ∃ n : ℕ, n = 172835 + 21 ∧ d ∣ n ∧ d = 21

theorem findDivisor : addDivisorProblem :=
by
  sorry

end NUMINAMATH_GPT_findDivisor_l1136_113661


namespace NUMINAMATH_GPT_power_inequality_l1136_113669

theorem power_inequality 
( a b : ℝ )
( h1 : 0 < a )
( h2 : 0 < b )
( h3 : a ^ 1999 + b ^ 2000 ≥ a ^ 2000 + b ^ 2001 ) :
  a ^ 2000 + b ^ 2000 ≤ 2 :=
sorry

end NUMINAMATH_GPT_power_inequality_l1136_113669


namespace NUMINAMATH_GPT_smallest_integer_not_expressible_in_form_l1136_113652

theorem smallest_integer_not_expressible_in_form :
  ∀ (n : ℕ), (0 < n ∧ (∀ a b c d : ℕ, n ≠ (2^a - 2^b) / (2^c - 2^d))) ↔ n = 11 :=
by
  sorry

end NUMINAMATH_GPT_smallest_integer_not_expressible_in_form_l1136_113652


namespace NUMINAMATH_GPT_expression_equals_neg_one_l1136_113671

theorem expression_equals_neg_one (a b c : ℝ) (h : a + b + c = 0) :
  (|a| / a) + (|b| / b) + (|c| / c) + (|a * b| / (a * b)) + (|a * c| / (a * c)) + (|b * c| / (b * c)) + (|a * b * c| / (a * b * c)) = -1 :=
  sorry

end NUMINAMATH_GPT_expression_equals_neg_one_l1136_113671


namespace NUMINAMATH_GPT_pie_charts_cannot_show_changes_l1136_113657

def pie_chart_shows_part_whole (P : Type) := true
def bar_chart_shows_amount (B : Type) := true
def line_chart_shows_amount_and_changes (L : Type) := true

theorem pie_charts_cannot_show_changes (P B L : Type) :
  pie_chart_shows_part_whole P ∧ bar_chart_shows_amount B ∧ line_chart_shows_amount_and_changes L →
  ¬ (pie_chart_shows_part_whole P ∧ ¬ line_chart_shows_amount_and_changes P) :=
by sorry

end NUMINAMATH_GPT_pie_charts_cannot_show_changes_l1136_113657


namespace NUMINAMATH_GPT_area_of_grey_part_l1136_113645

theorem area_of_grey_part :
  let area1 := 8 * 10
  let area2 := 12 * 9
  let area_black := 37
  let area_white := 43
  area2 - area_white = 65 :=
by
  let area1 := 8 * 10
  let area2 := 12 * 9
  let area_black := 37
  let area_white := 43
  have : area2 - area_white = 65 := by sorry
  exact this

end NUMINAMATH_GPT_area_of_grey_part_l1136_113645


namespace NUMINAMATH_GPT_meena_cookies_left_l1136_113668

-- Define the given conditions in terms of Lean definitions
def total_cookies_baked := 5 * 12
def cookies_sold_to_stone := 2 * 12
def cookies_bought_by_brock := 7
def cookies_bought_by_katy := 2 * cookies_bought_by_brock

-- Define the total cookies sold
def total_cookies_sold := cookies_sold_to_stone + cookies_bought_by_brock + cookies_bought_by_katy

-- Define the number of cookies left
def cookies_left := total_cookies_baked - total_cookies_sold

-- Prove that the number of cookies left is 15
theorem meena_cookies_left : cookies_left = 15 := by
  -- The proof is omitted (sorry is used to skip proof)
  sorry

end NUMINAMATH_GPT_meena_cookies_left_l1136_113668


namespace NUMINAMATH_GPT_add_least_number_l1136_113691

theorem add_least_number (n : ℕ) (h1 : n = 1789) (h2 : ∃ k : ℕ, 5 * k = n + 11) (h3 : ∃ j : ℕ, 6 * j = n + 11) (h4 : ∃ m : ℕ, 4 * m = n + 11) (h5 : ∃ l : ℕ, 11 * l = n + 11) : 11 = 11 :=
by
  sorry

end NUMINAMATH_GPT_add_least_number_l1136_113691


namespace NUMINAMATH_GPT_product_remainder_l1136_113670

theorem product_remainder
    (a b c : ℕ)
    (h₁ : a % 36 = 16)
    (h₂ : b % 36 = 8)
    (h₃ : c % 36 = 24) :
    (a * b * c) % 36 = 12 := 
    by
    sorry

end NUMINAMATH_GPT_product_remainder_l1136_113670


namespace NUMINAMATH_GPT_female_students_in_sample_l1136_113643

/-- In a high school, there are 500 male students and 400 female students in the first grade. 
    If a random sample of size 45 is taken from the students of this grade using stratified sampling by gender, 
    the number of female students in the sample is 20. -/
theorem female_students_in_sample 
  (num_male : ℕ) (num_female : ℕ) (sample_size : ℕ)
  (h_male : num_male = 500)
  (h_female : num_female = 400)
  (h_sample : sample_size = 45)
  (total_students : ℕ := num_male + num_female)
  (sample_ratio : ℚ := sample_size / total_students) :
  num_female * sample_ratio = 20 := 
sorry

end NUMINAMATH_GPT_female_students_in_sample_l1136_113643


namespace NUMINAMATH_GPT_find_a1_l1136_113688

theorem find_a1 (a b : ℕ → ℝ) (h1 : ∀ n ≥ 1, a (n + 1) + b (n + 1) = (a n + b n) / 2) 
  (h2 : ∀ n ≥ 1, a (n + 1) * b (n + 1) = (a n * b n) ^ (1/2)) 
  (hb2016 : b 2016 = 1) (ha1_pos : a 1 > 0) :
  a 1 = 2^2015 :=
sorry

end NUMINAMATH_GPT_find_a1_l1136_113688


namespace NUMINAMATH_GPT_min_integer_solution_l1136_113649

theorem min_integer_solution (x : ℤ) (h1 : 3 - x > 0) (h2 : (4 * x / 3 : ℚ) + 3 / 2 > -(x / 6)) : x = 0 := by
  sorry

end NUMINAMATH_GPT_min_integer_solution_l1136_113649


namespace NUMINAMATH_GPT_max_basketballs_l1136_113699

theorem max_basketballs (x : ℕ) (h1 : 80 * x + 50 * (40 - x) ≤ 2800) : x ≤ 26 := sorry

end NUMINAMATH_GPT_max_basketballs_l1136_113699


namespace NUMINAMATH_GPT_quadratic_inequality_solution_set_l1136_113664

theorem quadratic_inequality_solution_set :
  {x : ℝ | x^2 - x - 2 < 0} = {x : ℝ | -1 < x ∧ x < 2} :=
by
  sorry

end NUMINAMATH_GPT_quadratic_inequality_solution_set_l1136_113664


namespace NUMINAMATH_GPT_plane_equation_l1136_113648

variable (a b c : ℝ)
variable (ha : a ≠ 0)
variable (hb : b ≠ 0)
variable (hc : c ≠ 0)

theorem plane_equation (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) :
  ∃ x y z : ℝ, (x / a + y / b + z / c = 1) :=
sorry

end NUMINAMATH_GPT_plane_equation_l1136_113648


namespace NUMINAMATH_GPT_distinct_arrangements_balloon_l1136_113667

-- Let's define the basic conditions:
def total_letters : Nat := 7
def repeats_l : Nat := 2
def repeats_o : Nat := 2

-- Now let's state the problem.
theorem distinct_arrangements_balloon : 
  (Nat.factorial total_letters) / ((Nat.factorial repeats_l) * (Nat.factorial repeats_o)) = 1260 := 
by
  sorry

end NUMINAMATH_GPT_distinct_arrangements_balloon_l1136_113667


namespace NUMINAMATH_GPT_Jimin_weight_l1136_113682

variable (T J : ℝ)

theorem Jimin_weight (h1 : T - J = 4) (h2 : T + J = 88) : J = 42 :=
sorry

end NUMINAMATH_GPT_Jimin_weight_l1136_113682


namespace NUMINAMATH_GPT_find_percentage_l1136_113634

variable (x p : ℝ)
variable (h1 : 0.25 * x = (p / 100) * 1500 - 20)
variable (h2 : x = 820)

theorem find_percentage : p = 15 :=
by
  sorry

end NUMINAMATH_GPT_find_percentage_l1136_113634


namespace NUMINAMATH_GPT_rhombus_diagonal_length_l1136_113639

theorem rhombus_diagonal_length
  (d1 d2 A : ℝ)
  (h1 : d1 = 20)
  (h2 : A = 250)
  (h3 : A = (d1 * d2) / 2) :
  d2 = 25 :=
by
  sorry

end NUMINAMATH_GPT_rhombus_diagonal_length_l1136_113639


namespace NUMINAMATH_GPT_real_number_solution_l1136_113600

theorem real_number_solution : ∃ x : ℝ, x = 3 + 6 / (1 + 6 / x) ∧ x = 3 * Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_GPT_real_number_solution_l1136_113600


namespace NUMINAMATH_GPT_hexagon_side_lengths_l1136_113684

theorem hexagon_side_lengths (a b c d e f : ℕ) (h : a ≠ b ∧ b ≠ c ∧ c ≠ d ∧ d ≠ e ∧ e ≠ f)
(h1: a = 7 ∧ b = 5 ∧ (a + b + c + d + e + f = 38)) : 
(a + b + c + d + e + f = 38 ∧ a + b + c + d + e + f = 7 + 7 + 7 + 7 + 5 + 5) → 
(a + b + c + d + e + f = (4 * 7) + (2 * 5)) :=
sorry

end NUMINAMATH_GPT_hexagon_side_lengths_l1136_113684


namespace NUMINAMATH_GPT_william_probability_l1136_113606

def probability_of_correct_answer (p : ℚ) (q : ℚ) (n : ℕ) : ℚ :=
  1 - q^n

theorem william_probability :
  let p := 1 / 5
  let q := 4 / 5
  let n := 6
  probability_of_correct_answer p q n = 11529 / 15625 :=
by
  let p := 1 / 5
  let q := 4 / 5
  let n := 6
  unfold probability_of_correct_answer
  sorry

end NUMINAMATH_GPT_william_probability_l1136_113606


namespace NUMINAMATH_GPT_t_minus_s_equals_neg_17_25_l1136_113614

noncomputable def t : ℝ := (60 + 30 + 20 + 5 + 5) / 5
noncomputable def s : ℝ := (60 * (60 / 120) + 30 * (30 / 120) + 20 * (20 / 120) + 5 * (5 / 120) + 5 * (5 / 120))
noncomputable def t_minus_s : ℝ := t - s

theorem t_minus_s_equals_neg_17_25 : t_minus_s = -17.25 := by
  sorry

end NUMINAMATH_GPT_t_minus_s_equals_neg_17_25_l1136_113614


namespace NUMINAMATH_GPT_remaining_painting_time_l1136_113662

-- Define the conditions
def total_rooms : ℕ := 10
def hours_per_room : ℕ := 8
def rooms_painted : ℕ := 8

-- Define what we want to prove
theorem remaining_painting_time : (total_rooms - rooms_painted) * hours_per_room = 16 :=
by
  -- Here is where you would provide the proof
  sorry

end NUMINAMATH_GPT_remaining_painting_time_l1136_113662


namespace NUMINAMATH_GPT_isosceles_right_triangle_area_l1136_113665

theorem isosceles_right_triangle_area (h : ℝ) (l : ℝ) (A : ℝ)
  (h_def : h = 6 * Real.sqrt 2)
  (rel_leg_hypotenuse : h = Real.sqrt 2 * l)
  (area_def : A = 1 / 2 * l * l) :
  A = 18 :=
by
  sorry

end NUMINAMATH_GPT_isosceles_right_triangle_area_l1136_113665


namespace NUMINAMATH_GPT_field_dimensions_l1136_113642

theorem field_dimensions (W L : ℕ) (h1 : L = 2 * W) (h2 : 2 * L + 2 * W = 600) : W = 100 ∧ L = 200 :=
sorry

end NUMINAMATH_GPT_field_dimensions_l1136_113642


namespace NUMINAMATH_GPT_sum_of_fractions_l1136_113638

theorem sum_of_fractions : 
  (1 / 10) + (2 / 10) + (3 / 10) + (4 / 10) + (10 / 10) + (11 / 10) + (15 / 10) + (20 / 10) + (25 / 10) + (50 / 10) = 14.1 :=
by sorry

end NUMINAMATH_GPT_sum_of_fractions_l1136_113638


namespace NUMINAMATH_GPT_euler_conjecture_counter_example_l1136_113690

theorem euler_conjecture_counter_example :
  ∃ (n : ℕ), 133^5 + 110^5 + 84^5 + 27^5 = n^5 ∧ n = 144 :=
by
  sorry

end NUMINAMATH_GPT_euler_conjecture_counter_example_l1136_113690


namespace NUMINAMATH_GPT_smallest_four_digit_divisible_by_4_and_5_l1136_113675

theorem smallest_four_digit_divisible_by_4_and_5 : 
  ∃ n, (n % 4 = 0) ∧ (n % 5 = 0) ∧ 1000 ≤ n ∧ n < 10000 ∧ 
  ∀ m, (m % 4 = 0) ∧ (m % 5 = 0) ∧ 1000 ≤ m ∧ m < 10000 → n ≤ m :=
by
  sorry

end NUMINAMATH_GPT_smallest_four_digit_divisible_by_4_and_5_l1136_113675


namespace NUMINAMATH_GPT_income_growth_relation_l1136_113673

-- Define all the conditions
def initial_income : ℝ := 1.3
def third_week_income : ℝ := 2
def growth_rate (x : ℝ) : ℝ := (1 + x)^2  -- Compound interest style growth over 2 weeks.

-- Theorem: proving the relationship given the conditions
theorem income_growth_relation (x : ℝ) : initial_income * growth_rate x = third_week_income :=
by
  unfold initial_income third_week_income growth_rate
  sorry  -- Proof not required.

end NUMINAMATH_GPT_income_growth_relation_l1136_113673


namespace NUMINAMATH_GPT_marys_total_cards_l1136_113651

def initial_cards : ℕ := 18
def torn_cards : ℕ := 8
def cards_from_fred : ℕ := 26
def cards_bought_by_mary : ℕ := 40

theorem marys_total_cards :
  initial_cards - torn_cards + cards_from_fred + cards_bought_by_mary = 76 :=
by
  sorry

end NUMINAMATH_GPT_marys_total_cards_l1136_113651


namespace NUMINAMATH_GPT_range_of_a_l1136_113681

theorem range_of_a (a : ℝ) : (∀ x : ℝ, x ≥ a → |x - 1| < 1) → (∃ x : ℝ, |x - 1| < 1 ∧ x < a) → a ≤ 0 := 
sorry

end NUMINAMATH_GPT_range_of_a_l1136_113681


namespace NUMINAMATH_GPT_chord_constant_sum_l1136_113612

theorem chord_constant_sum (d : ℝ) (h : d = 1/2) :
  ∀ A B : ℝ × ℝ, (A.2 = A.1^2) → (B.2 = B.1^2) →
  (∃ m : ℝ, A.2 = m * A.1 + d ∧ B.2 = m * B.1 + d) →
  (∃ D : ℝ × ℝ, D = (0, d) ∧ (∃ s : ℝ,
    s = (1 / ((A.1 - D.1)^2 + (A.2 - D.2)^2) + 1 / ((B.1 - D.1)^2 + (B.2 - D.2)^2)) ∧ s = 4)) :=
by 
  sorry

end NUMINAMATH_GPT_chord_constant_sum_l1136_113612


namespace NUMINAMATH_GPT_parabola_passes_through_A_C_l1136_113625

theorem parabola_passes_through_A_C : ∃ (a b : ℝ), (2 = a * 1^2 + b * 1 + 1) ∧ (1 = a * 2^2 + b * 2 + 1) :=
by {
  sorry
}

end NUMINAMATH_GPT_parabola_passes_through_A_C_l1136_113625


namespace NUMINAMATH_GPT_find_x_range_l1136_113619

variable (f : ℝ → ℝ)

def even_function (f : ℝ → ℝ) := ∀ x : ℝ, f x = f (-x)

def decreasing_on_nonnegative (f : ℝ → ℝ) :=
  ∀ x1 x2 : ℝ, x1 ≥ 0 → x2 ≥ 0 → x1 ≠ x2 → (f x2 - f x1) / (x2 - x1) < 0

theorem find_x_range (f : ℝ → ℝ)
  (h1 : even_function f)
  (h2 : decreasing_on_nonnegative f)
  (h3 : f (1/3) = 3/4)
  (h4 : ∀ x : ℝ, 4 * f (Real.logb (1/8) x) > 3) :
  ∀ x : ℝ, (1/2 < x ∧ x < 2) ↔ True := sorry

end NUMINAMATH_GPT_find_x_range_l1136_113619


namespace NUMINAMATH_GPT_count_two_digit_integers_congruent_to_2_mod_4_l1136_113640

theorem count_two_digit_integers_congruent_to_2_mod_4 : 
  ∃ n : ℕ, (∀ x : ℕ, 10 ≤ x ∧ x ≤ 99 → x % 4 = 2 → x = 4 * k + 2) ∧ n = 23 := 
by
  sorry

end NUMINAMATH_GPT_count_two_digit_integers_congruent_to_2_mod_4_l1136_113640


namespace NUMINAMATH_GPT_temperature_at_night_l1136_113637

theorem temperature_at_night 
  (T_morning : ℝ) 
  (T_rise_noon : ℝ) 
  (T_drop_night : ℝ) 
  (h1 : T_morning = 22) 
  (h2 : T_rise_noon = 6) 
  (h3 : T_drop_night = 10) : 
  (T_morning + T_rise_noon - T_drop_night = 18) :=
by 
  sorry

end NUMINAMATH_GPT_temperature_at_night_l1136_113637


namespace NUMINAMATH_GPT_range_of_a_l1136_113693

def p (x : ℝ) : Prop := 1 / 2 ≤ x ∧ x ≤ 1

def q (x a : ℝ) : Prop := (x - a) * (x - a - 1) ≤ 0

theorem range_of_a (a x : ℝ) 
  (hp : ∀ x, ¬ (1 / 2 ≤ x ∧ x ≤ 1) → (x < 1 / 2 ∨ x > 1))
  (hq : ∀ x, ¬ ((x - a) * (x - a - 1) ≤ 0) → (x < a ∨ x > a + 1))
  (h : ∀ x, (q x a) → (p x)) :
  0 ≤ a ∧ a ≤ 1 / 2 := 
sorry

end NUMINAMATH_GPT_range_of_a_l1136_113693


namespace NUMINAMATH_GPT_line_equation_l1136_113617

noncomputable def projection (u v : ℝ × ℝ) : ℝ × ℝ :=
  let dot_uv := u.1 * v.1 + u.2 * v.2
  let norm_u_sq := u.1 * u.1 + u.2 * u.2
  (dot_uv / norm_u_sq) • u

theorem line_equation :
  ∀ (x y : ℝ), projection (4, 3) (x, y) = (-4, -3) → y = (-4 / 3) * x - 25 / 3 :=
by
  intros x y h
  sorry

end NUMINAMATH_GPT_line_equation_l1136_113617


namespace NUMINAMATH_GPT_george_boxes_l1136_113677

-- Define the problem conditions and the question's expected outcome.
def total_blocks : ℕ := 12
def blocks_per_box : ℕ := 6
def expected_num_boxes : ℕ := 2

-- The proof statement that needs to be proved: George has the expected number of boxes.
theorem george_boxes : total_blocks / blocks_per_box = expected_num_boxes := 
  sorry

end NUMINAMATH_GPT_george_boxes_l1136_113677


namespace NUMINAMATH_GPT_tire_circumference_l1136_113631

/-- If a tire rotates at 400 revolutions per minute and the car is traveling at 48 km/h, 
    prove that the circumference of the tire in meters is 2. -/
theorem tire_circumference (speed_kmh : ℕ) (revolutions_per_min : ℕ)
  (h1 : speed_kmh = 48) (h2 : revolutions_per_min = 400) : 
  (circumference : ℕ) = 2 := 
sorry

end NUMINAMATH_GPT_tire_circumference_l1136_113631


namespace NUMINAMATH_GPT_batteries_on_flashlights_l1136_113609

variable (b_flashlights b_toys b_controllers b_total : ℕ)

theorem batteries_on_flashlights :
  b_toys = 15 → 
  b_controllers = 2 → 
  b_total = 19 → 
  b_total = b_flashlights + b_toys + b_controllers → 
  b_flashlights = 4 :=
by
  intros h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_batteries_on_flashlights_l1136_113609


namespace NUMINAMATH_GPT_find_smallest_value_of_sum_of_squares_l1136_113602
noncomputable def smallest_value (x y z : ℚ) := x^2 + y^2 + z^2

theorem find_smallest_value_of_sum_of_squares :
  ∃ (x y z : ℚ), (x + 4) * (y - 4) = 0 ∧ 3 * z - 2 * y = 5 ∧ smallest_value x y z = 457 / 9 :=
by
  sorry

end NUMINAMATH_GPT_find_smallest_value_of_sum_of_squares_l1136_113602


namespace NUMINAMATH_GPT_geometric_sum_first_six_terms_l1136_113676

variable (a_n : ℕ → ℝ)

axiom geometric_seq (r a1 : ℝ) : ∀ n, a_n n = a1 * r ^ (n - 1)
axiom a2_val : a_n 2 = 2
axiom a5_val : a_n 5 = 16

theorem geometric_sum_first_six_terms (S6 : ℝ) : S6 = 1 * (1 - 2^6) / (1 - 2) := by
  sorry

end NUMINAMATH_GPT_geometric_sum_first_six_terms_l1136_113676


namespace NUMINAMATH_GPT_factorization_eq1_factorization_eq2_l1136_113658

-- Definitions for the given conditions
variables (a b x y m : ℝ)

-- The problem statement as Lean definitions and the goal theorems
def expr1 : ℝ := -6 * a * b + 3 * a^2 + 3 * b^2
def factored1 : ℝ := 3 * (a - b)^2

def expr2 : ℝ := y^2 * (2 - m) + x^2 * (m - 2)
def factored2 : ℝ := (m - 2) * (x + y) * (x - y)

-- Theorem statements for equivalence
theorem factorization_eq1 : expr1 a b = factored1 a b :=
by
  sorry

theorem factorization_eq2 : expr2 x y m = factored2 x y m :=
by
  sorry

end NUMINAMATH_GPT_factorization_eq1_factorization_eq2_l1136_113658


namespace NUMINAMATH_GPT_find_constants_for_matrix_condition_l1136_113687

noncomputable section

def B : Matrix (Fin 3) (Fin 3) ℝ :=
  ![![1, 2, 3], ![0, 1, 2], ![1, 0, 1]]

def I : Matrix (Fin 3) (Fin 3) ℝ :=
  1

theorem find_constants_for_matrix_condition :
  ∃ p q r : ℝ, B^3 + p • B^2 + q • B + r • I = 0 :=
by
  use -5, 3, -6
  sorry

end NUMINAMATH_GPT_find_constants_for_matrix_condition_l1136_113687


namespace NUMINAMATH_GPT_solve_for_y_l1136_113636

theorem solve_for_y (y : ℚ) (h : |5 * y - 6| = 0) : y = 6 / 5 :=
by 
  sorry

end NUMINAMATH_GPT_solve_for_y_l1136_113636


namespace NUMINAMATH_GPT_smallest_positive_period_of_f_range_of_a_l1136_113692

noncomputable def f (x : ℝ) : ℝ := Real.sin (2 * x) + Real.cos (2 * x)

theorem smallest_positive_period_of_f : (∃ T > 0, ∀ x, f (x + T) = f x) ∧ (T = π) :=
by
  sorry

theorem range_of_a (a : ℝ) : (∀ x, f x ≤ a) → a ≥ Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_GPT_smallest_positive_period_of_f_range_of_a_l1136_113692


namespace NUMINAMATH_GPT_percentage_of_failed_candidates_l1136_113608

theorem percentage_of_failed_candidates :
  let total_candidates := 2000
  let girls := 900
  let boys := total_candidates - girls
  let boys_passed := 32 / 100 * boys
  let girls_passed := 32 / 100 * girls
  let total_passed := boys_passed + girls_passed
  let total_failed := total_candidates - total_passed
  let percentage_failed := (total_failed / total_candidates) * 100
  percentage_failed = 68 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_percentage_of_failed_candidates_l1136_113608


namespace NUMINAMATH_GPT_log_exp_identity_l1136_113633

theorem log_exp_identity (a : ℝ) (h : a = Real.log 5 / Real.log 4) : 
  (2^a + 2^(-a) = 6 * Real.sqrt 5 / 5) :=
by {
  -- a = log_4 (5) can be rewritten using change-of-base formula: log 5 / log 4
  -- so, it can be used directly in the theorem
  sorry
}

end NUMINAMATH_GPT_log_exp_identity_l1136_113633


namespace NUMINAMATH_GPT_least_number_l1136_113683

theorem least_number (n p q r s : ℕ) : 
  (n + p) % 24 = 0 ∧ 
  (n + q) % 32 = 0 ∧ 
  (n + r) % 36 = 0 ∧
  (n + s) % 54 = 0 →
  n = 863 :=
sorry

end NUMINAMATH_GPT_least_number_l1136_113683


namespace NUMINAMATH_GPT_square_tiles_count_l1136_113655

theorem square_tiles_count 
  (h s : ℕ)
  (total_tiles : h + s = 30)
  (total_edges : 6 * h + 4 * s = 128) : 
  s = 26 :=
by
  sorry

end NUMINAMATH_GPT_square_tiles_count_l1136_113655


namespace NUMINAMATH_GPT_find_q_l1136_113607

theorem find_q (q : ℤ) (x : ℤ) (y : ℤ) (h1 : x = 55 + 2 * q) (h2 : y = 4 * q + 41) (h3 : x = y) : q = 7 :=
by
  sorry

end NUMINAMATH_GPT_find_q_l1136_113607


namespace NUMINAMATH_GPT_abs_lt_two_nec_but_not_suff_l1136_113635

theorem abs_lt_two_nec_but_not_suff (x : ℝ) :
  (|x - 1| < 2) → (0 < x ∧ x < 3) ∧ ¬((0 < x ∧ x < 3) → (|x - 1| < 2)) := sorry

end NUMINAMATH_GPT_abs_lt_two_nec_but_not_suff_l1136_113635


namespace NUMINAMATH_GPT_pow_ge_double_l1136_113610

theorem pow_ge_double (n : ℕ) : 2^n ≥ 2 * n := sorry

end NUMINAMATH_GPT_pow_ge_double_l1136_113610


namespace NUMINAMATH_GPT_AM_GM_inequality_l1136_113601

theorem AM_GM_inequality (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (h_not_all_eq : x ≠ y ∨ y ≠ z ∨ z ≠ x) :
  (x + y) * (y + z) * (z + x) > 8 * x * y * z :=
by
  sorry

end NUMINAMATH_GPT_AM_GM_inequality_l1136_113601


namespace NUMINAMATH_GPT_numbers_not_as_difference_of_squares_l1136_113620

theorem numbers_not_as_difference_of_squares :
  {n : ℕ | ¬ ∃ x y : ℕ, x^2 - y^2 = n} = {1, 4} ∪ {4*k + 2 | k : ℕ} :=
by sorry

end NUMINAMATH_GPT_numbers_not_as_difference_of_squares_l1136_113620


namespace NUMINAMATH_GPT_evaluate_polynomial_at_minus_two_l1136_113689

noncomputable def polynomial (x : ℝ) : ℝ := 2 * x^4 + 3 * x^3 - x^2 + 2 * x + 5

theorem evaluate_polynomial_at_minus_two : polynomial (-2) = 5 := by
  sorry

end NUMINAMATH_GPT_evaluate_polynomial_at_minus_two_l1136_113689


namespace NUMINAMATH_GPT_range_of_a_l1136_113674

open Set

def p (a x : ℝ) : Prop := x^2 - 4 * a * x + 3 * a^2 < 0
def q (x : ℝ) : Prop := x^2 + 2 * x - 8 > 0

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, p a x → q x) →
  ({ x : ℝ | p a x } ⊆ { x : ℝ | q x }) →
  a ≤ -4 ∨ a ≥ 2 ∨ a = 0 :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l1136_113674


namespace NUMINAMATH_GPT_school_fee_l1136_113685

theorem school_fee (a b c d e f g h i j k l : ℕ) (h1 : a = 2) (h2 : b = 100) (h3 : c = 1) (h4 : d = 50) (h5 : e = 5) (h6 : f = 20) (h7 : g = 3) (h8 : h = 10) (h9 : i = 4) (h10 : j = 5) (h11 : k = 4 ) (h12 : l = 50) :
  a * b + c * d + e * f + g * h + i * j + 3 * b + k * d + 2 * f + l * h + 6 * j = 980 := sorry

end NUMINAMATH_GPT_school_fee_l1136_113685


namespace NUMINAMATH_GPT_negation_of_proposition_l1136_113659

theorem negation_of_proposition :
  (¬∃ x₀ ∈ Set.Ioo 0 (π/2), Real.cos x₀ > Real.sin x₀) ↔ ∀ x ∈ Set.Ioo 0 (π / 2), Real.cos x ≤ Real.sin x :=
by
  sorry

end NUMINAMATH_GPT_negation_of_proposition_l1136_113659


namespace NUMINAMATH_GPT_blocks_eaten_correct_l1136_113656

def initial_blocks : ℕ := 55
def remaining_blocks : ℕ := 26

-- How many blocks were eaten by the hippopotamus?
def blocks_eaten_by_hippopotamus : ℕ := initial_blocks - remaining_blocks

theorem blocks_eaten_correct :
  blocks_eaten_by_hippopotamus = 29 := by
  sorry

end NUMINAMATH_GPT_blocks_eaten_correct_l1136_113656


namespace NUMINAMATH_GPT_sum_log_base_5_divisors_l1136_113646

theorem sum_log_base_5_divisors (n : ℕ) (h : n * (n + 1) / 2 = 264) : n = 23 :=
by
  sorry

end NUMINAMATH_GPT_sum_log_base_5_divisors_l1136_113646


namespace NUMINAMATH_GPT_ratio_of_surface_areas_l1136_113613

theorem ratio_of_surface_areas (s : ℝ) :
  let cube_surface_area := 6 * s^2
  let tetrahedron_edge := s * Real.sqrt 2
  let tetrahedron_face_area := (Real.sqrt 3 / 4) * (tetrahedron_edge)^2
  let tetrahedron_surface_area := 4 * tetrahedron_face_area
  (cube_surface_area / tetrahedron_surface_area) = Real.sqrt 3 :=
by
  let cube_surface_area := 6 * s^2
  let tetrahedron_edge := s * Real.sqrt 2
  let tetrahedron_face_area := (Real.sqrt 3 / 4) * (tetrahedron_edge)^2
  let tetrahedron_surface_area := 4 * tetrahedron_face_area
  show (cube_surface_area / tetrahedron_surface_area) = Real.sqrt 3
  sorry

end NUMINAMATH_GPT_ratio_of_surface_areas_l1136_113613


namespace NUMINAMATH_GPT_sampling_method_selection_l1136_113621

-- Define the sampling methods as data type
inductive SamplingMethod
| SimpleRandomSampling : SamplingMethod
| SystematicSampling : SamplingMethod
| StratifiedSampling : SamplingMethod
| SamplingWithReplacement : SamplingMethod

-- Define our conditions
def basketballs : Nat := 10
def is_random_selection : Bool := true
def no_obvious_stratification : Bool := true

-- The theorem to prove the correct sampling method
theorem sampling_method_selection 
  (b : Nat) 
  (random_selection : Bool) 
  (no_stratification : Bool) : 
  SamplingMethod :=
  if b = 10 ∧ random_selection ∧ no_stratification then SamplingMethod.SimpleRandomSampling 
  else sorry

-- Prove the correct sampling method given our conditions
example : sampling_method_selection basketballs is_random_selection no_obvious_stratification = SamplingMethod.SimpleRandomSampling := 
by
-- skipping the proof here with sorry
sorry

end NUMINAMATH_GPT_sampling_method_selection_l1136_113621


namespace NUMINAMATH_GPT_total_wrappers_collected_l1136_113697

theorem total_wrappers_collected :
  let Andy_wrappers := 34
  let Max_wrappers := 15
  let Zoe_wrappers := 25
  Andy_wrappers + Max_wrappers + Zoe_wrappers = 74 :=
by
  let Andy_wrappers := 34
  let Max_wrappers := 15
  let Zoe_wrappers := 25
  show Andy_wrappers + Max_wrappers + Zoe_wrappers = 74
  sorry

end NUMINAMATH_GPT_total_wrappers_collected_l1136_113697


namespace NUMINAMATH_GPT_find_cd_l1136_113628

noncomputable def g (c d : ℝ) (x : ℝ) : ℝ := c * x^3 - 8 * x^2 + d * x - 7

theorem find_cd (c d : ℝ) 
  (h1 : g c d 2 = -7) 
  (h2 : g c d (-1) = -25) : 
  (c, d) = (2, 8) := 
by
  sorry

end NUMINAMATH_GPT_find_cd_l1136_113628


namespace NUMINAMATH_GPT_quadratic_root_and_coefficient_l1136_113678

theorem quadratic_root_and_coefficient (k : ℝ) :
  (∃ x : ℝ, 5 * x^2 + k * x - 6 = 0 ∧ x = 2) →
  (∃ x₁ : ℝ, (5 * x₁^2 + k * x₁ - 6 = 0 ∧ x₁ ≠ 2) ∧ x₁ = -3/5 ∧ k = -7) :=
by
  sorry

end NUMINAMATH_GPT_quadratic_root_and_coefficient_l1136_113678


namespace NUMINAMATH_GPT_line_passes_through_circle_center_l1136_113624

theorem line_passes_through_circle_center
  (a : ℝ)
  (h_line : ∀ (x y : ℝ), 3 * x + y + a = 0 → (x, y) = (-1, 2))
  (h_circle : ∀ (x y : ℝ), x^2 + y^2 + 2 * x - 4 * y = 0 → (x, y) = (-1, 2)) :
  a = 1 :=
by
  sorry

end NUMINAMATH_GPT_line_passes_through_circle_center_l1136_113624


namespace NUMINAMATH_GPT_g_is_even_l1136_113616

noncomputable def g (x : ℝ) := 2 ^ (x ^ 2 - 4) - |x|

theorem g_is_even : ∀ x : ℝ, g (-x) = g x :=
by
  sorry

end NUMINAMATH_GPT_g_is_even_l1136_113616


namespace NUMINAMATH_GPT_green_tiles_in_50th_row_l1136_113641

-- Conditions
def tiles_in_row (n : ℕ) : ℕ := 2 * n - 1

def green_tiles_in_row (n : ℕ) : ℕ := (tiles_in_row n - 1) / 2

-- Prove the number of green tiles in the 50th row
theorem green_tiles_in_50th_row : green_tiles_in_row 50 = 49 :=
by
  -- Placeholder proof
  sorry

end NUMINAMATH_GPT_green_tiles_in_50th_row_l1136_113641


namespace NUMINAMATH_GPT_smallest_k_l1136_113644

-- Definitions used in the conditions
def poly1 (z : ℂ) : ℂ := z^12 + z^11 + z^8 + z^7 + z^6 + z^3 + 1
def poly2 (z : ℂ) (k : ℕ) : ℂ := z^k - 1

-- Lean 4 statement for the problem
theorem smallest_k (k : ℕ) (hk : k = 120) :
  ∀ z : ℂ, poly1 z ∣ poly2 z k :=
sorry

end NUMINAMATH_GPT_smallest_k_l1136_113644


namespace NUMINAMATH_GPT_parabola_point_distance_condition_l1136_113611

theorem parabola_point_distance_condition (k : ℝ) (p : ℝ) (h_p_gt_0 : p > 0) (focus : ℝ × ℝ) (vertex : ℝ × ℝ) :
  vertex = (0, 0) → focus = (0, p/2) → (k^2 = -2 * p * (-2)) → dist (k, -2) focus = 4 → k = 4 ∨ k = -4 :=
by
  sorry

end NUMINAMATH_GPT_parabola_point_distance_condition_l1136_113611


namespace NUMINAMATH_GPT_coyote_time_lemma_l1136_113694

theorem coyote_time_lemma (coyote_speed darrel_speed : ℝ) (catch_up_time t : ℝ) 
  (h1 : coyote_speed = 15) (h2 : darrel_speed = 30) (h3 : catch_up_time = 1) (h4 : darrel_speed * catch_up_time = coyote_speed * t) :
  t = 2 :=
by
  sorry

end NUMINAMATH_GPT_coyote_time_lemma_l1136_113694


namespace NUMINAMATH_GPT_Kevin_ends_with_54_cards_l1136_113615

/-- Kevin starts with 7 cards and finds another 47 cards. 
    This theorem proves that Kevin ends with 54 cards. -/
theorem Kevin_ends_with_54_cards :
  let initial_cards := 7
  let found_cards := 47
  initial_cards + found_cards = 54 := 
by
  let initial_cards := 7
  let found_cards := 47
  sorry

end NUMINAMATH_GPT_Kevin_ends_with_54_cards_l1136_113615


namespace NUMINAMATH_GPT_smallest_b_value_l1136_113604

variable {a b c d : ℝ}

-- Definitions based on conditions
def is_arithmetic_series (a b c : ℝ) (d : ℝ) : Prop :=
  a = b - d ∧ c = b + d

def abc_product (a b c : ℝ) : Prop :=
  a * b * c = 216

theorem smallest_b_value (a b c d : ℝ) 
  (pos_a : a > 0) (pos_b : b > 0) (pos_c : c > 0)
  (arith_series : is_arithmetic_series a b c d)
  (abc_216 : abc_product a b c) : 
  b ≥ 6 :=
by
  sorry

end NUMINAMATH_GPT_smallest_b_value_l1136_113604


namespace NUMINAMATH_GPT_train_length_l1136_113623

variable (L V : ℝ)

-- Given conditions
def condition1 : Prop := V = L / 24
def condition2 : Prop := V = (L + 650) / 89

theorem train_length : condition1 L V → condition2 L V → L = 240 := by
  intro h1 h2
  sorry

end NUMINAMATH_GPT_train_length_l1136_113623


namespace NUMINAMATH_GPT_salary_increase_l1136_113695

theorem salary_increase (S : ℝ) (P : ℝ) (H0 : P > 0 )  
  (saved_last_year : ℝ := 0.10 * S)
  (salary_this_year : ℝ := S * (1 + P / 100))
  (saved_this_year : ℝ := 0.15 * salary_this_year)
  (H1 : saved_this_year = 1.65 * saved_last_year) :
  P = 10 :=
by
  sorry

end NUMINAMATH_GPT_salary_increase_l1136_113695


namespace NUMINAMATH_GPT_pipe_tank_fill_time_l1136_113680

/-- 
Given:
1. Pipe A fills the tank in 2 hours.
2. The leak empties the tank in 4 hours.
Prove: 
The tank is filled in 4 hours when both Pipe A and the leak are working together.
 -/
theorem pipe_tank_fill_time :
  let A := 1 / 2 -- rate at which Pipe A fills the tank (tank per hour)
  let L := 1 / 4 -- rate at which the leak empties the tank (tank per hour)
  let net_rate := A - L -- net rate of filling the tank
  net_rate > 0 → (1 / net_rate) = 4 := 
by
  intros
  sorry

end NUMINAMATH_GPT_pipe_tank_fill_time_l1136_113680


namespace NUMINAMATH_GPT_cells_at_end_of_9th_day_l1136_113672

def initial_cells : ℕ := 4
def split_ratio : ℕ := 3
def total_days : ℕ := 9
def days_per_split : ℕ := 3

def num_terms : ℕ := total_days / days_per_split

noncomputable def number_of_cells (initial_cells split_ratio num_terms : ℕ) : ℕ :=
  initial_cells * split_ratio ^ (num_terms - 1)

theorem cells_at_end_of_9th_day :
  number_of_cells initial_cells split_ratio num_terms = 36 :=
by
  sorry

end NUMINAMATH_GPT_cells_at_end_of_9th_day_l1136_113672


namespace NUMINAMATH_GPT_polygon_properties_l1136_113663

def interior_angle_sum (n : ℕ) : ℝ :=
  (n - 2) * 180

def exterior_angle_sum : ℝ :=
  360

theorem polygon_properties (n : ℕ) (h : interior_angle_sum n = 3 * exterior_angle_sum + 180) :
  n = 9 ∧ interior_angle_sum n / n = 140 :=
by
  sorry

end NUMINAMATH_GPT_polygon_properties_l1136_113663
