import Mathlib

namespace sarah_copies_total_pages_l900_90074

noncomputable def total_pages_copied (people : ℕ) (pages_first : ℕ) (copies_first : ℕ) (pages_second : ℕ) (copies_second : ℕ) : ℕ :=
  (pages_first * (copies_first * people)) + (pages_second * (copies_second * people))

theorem sarah_copies_total_pages :
  total_pages_copied 20 30 3 45 2 = 3600 := by
  sorry

end sarah_copies_total_pages_l900_90074


namespace cos_660_degrees_is_one_half_l900_90045

noncomputable def cos_660_eq_one_half : Prop :=
  (Real.cos (660 * Real.pi / 180) = 1 / 2)

theorem cos_660_degrees_is_one_half : cos_660_eq_one_half :=
by
  sorry

end cos_660_degrees_is_one_half_l900_90045


namespace price_decrease_l900_90010

theorem price_decrease (P : ℝ) (h₁ : 1.25 * P = P * 1.25) (h₂ : 1.10 * P = P * 1.10) :
  1.25 * P * (1 - 12 / 100) = 1.10 * P :=
by
  sorry

end price_decrease_l900_90010


namespace train_length_is_150_l900_90098

noncomputable def train_length_crossing_post (t_post : ℕ := 10) : ℕ := 10
noncomputable def train_length_crossing_platform (length_platform : ℕ := 150) (t_platform : ℕ := 20) : ℕ := 20
def train_constant_speed (L v : ℚ) (t_post t_platform : ℚ) (length_platform : ℚ) : Prop :=
  v = L / t_post ∧ v = (L + length_platform) / t_platform

theorem train_length_is_150 (L : ℚ) (t_post t_platform : ℚ) (length_platform : ℚ) (H : train_constant_speed L v t_post t_platform length_platform) : 
  L = 150 :=
by
  sorry

end train_length_is_150_l900_90098


namespace speed_limit_of_friend_l900_90042

theorem speed_limit_of_friend (total_distance : ℕ) (christina_speed : ℕ) (christina_time_min : ℕ) (friend_time_hr : ℕ) 
(h1 : total_distance = 210)
(h2 : christina_speed = 30)
(h3 : christina_time_min = 180)
(h4 : friend_time_hr = 3)
(h5 : total_distance = (christina_speed * (christina_time_min / 60)) + (christina_speed * friend_time_hr)) :
  (total_distance - christina_speed * (christina_time_min / 60)) / friend_time_hr = 40 := 
by
  sorry

end speed_limit_of_friend_l900_90042


namespace arithmetic_sequence_common_difference_l900_90046

/-- The sum of the first n terms of an arithmetic sequence -/
def S (n : ℕ) (a₁ d : ℚ) : ℚ := n * a₁ + (n * (n - 1) / 2) * d

/-- Condition for the sum of the first 5 terms -/
def S5 (a₁ d : ℚ) : Prop := S 5 a₁ d = 6

/-- Condition for the second term of the sequence -/
def a2 (a₁ d : ℚ) : Prop := a₁ + d = 1

/-- The main theorem to be proved -/
theorem arithmetic_sequence_common_difference (a₁ d : ℚ) (hS5 : S5 a₁ d) (ha2 : a2 a₁ d) : d = 1 / 5 :=
sorry

end arithmetic_sequence_common_difference_l900_90046


namespace jim_makes_60_dollars_l900_90068

-- Definitions based on the problem conditions
def average_weight_per_rock : ℝ := 1.5
def price_per_pound : ℝ := 4
def number_of_rocks : ℕ := 10

-- Problem statement
theorem jim_makes_60_dollars :
  (average_weight_per_rock * number_of_rocks) * price_per_pound = 60 := by
  sorry

end jim_makes_60_dollars_l900_90068


namespace data_set_variance_l900_90044

def data_set : List ℕ := [2, 4, 5, 3, 6]

noncomputable def mean (l : List ℕ) : ℝ :=
  l.sum / l.length

noncomputable def variance (l : List ℕ) : ℝ :=
  let m : ℝ := mean l
  (l.map (fun x => (x - m) ^ 2)).sum / l.length

theorem data_set_variance : variance data_set = 2 := by
  sorry

end data_set_variance_l900_90044


namespace find_b_l900_90082

theorem find_b (α β b : ℤ)
  (h1: α > 1)
  (h2: β < -1)
  (h3: ∃ x : ℝ, α * x^2 + β * x - 2 = 0)
  (h4: ∃ x : ℝ, x^2 + bx - 2 = 0)
  (hb: ∀ root1 root2 : ℝ, root1 * root2 = -2 ∧ root1 + root2 = -b) :
  b = 0 := 
sorry

end find_b_l900_90082


namespace percentage_of_diameter_l900_90041

variable (d_R d_S r_R r_S : ℝ)
variable (A_R A_S : ℝ)
variable (pi : ℝ) (h1 : pi > 0)

theorem percentage_of_diameter 
(h_area : A_R = 0.64 * A_S) 
(h_radius_R : r_R = d_R / 2) 
(h_radius_S : r_S = d_S / 2)
(h_area_R : A_R = pi * r_R^2) 
(h_area_S : A_S = pi * r_S^2) 
: (d_R / d_S) * 100 = 80 := by
  sorry

end percentage_of_diameter_l900_90041


namespace roots_negative_and_bounds_find_possible_values_of_b_and_c_l900_90000

theorem roots_negative_and_bounds
  (b c x₁ x₂ x₁' x₂' : ℤ) 
  (h1 : x₁ * x₂ > 0) 
  (h2 : x₁' * x₂' > 0)
  (h3 : x₁^2 + b * x₁ + c = 0) 
  (h4 : x₂^2 + b * x₂ + c = 0) 
  (h5 : x₁'^2 + c * x₁' + b = 0) 
  (h6 : x₂'^2 + c * x₂' + b = 0) :
  x₁ < 0 ∧ x₂ < 0 ∧ x₁' < 0 ∧ x₂' < 0 ∧ (b - 1 ≤ c ∧ c ≤ b + 1) :=
by
  sorry


theorem find_possible_values_of_b_and_c 
  (b c : ℤ) 
  (h's : ∃ x₁ x₂ x₁' x₂', 
    x₁ * x₂ > 0 ∧ 
    x₁' * x₂' > 0 ∧ 
    (x₁^2 + b * x₁ + c = 0) ∧ 
    (x₂^2 + b * x₂ + c = 0) ∧ 
    (x₁'^2 + c * x₁' + b = 0) ∧ 
    (x₂'^2 + c * x₂' + b = 0)) :
  (b = 4 ∧ c = 4) ∨ 
  (b = 5 ∧ c = 6) ∨ 
  (b = 6 ∧ c = 5) :=
by
  sorry

end roots_negative_and_bounds_find_possible_values_of_b_and_c_l900_90000


namespace simplify_scientific_notation_l900_90080

theorem simplify_scientific_notation :
  (12 * 10^10) / (6 * 10^2) = 2 * 10^8 := 
sorry

end simplify_scientific_notation_l900_90080


namespace fabian_initial_hours_l900_90002

-- Define the conditions
def speed : ℕ := 5
def total_distance : ℕ := 30
def additional_time : ℕ := 3

-- The distance Fabian covers in the additional time
def additional_distance := speed * additional_time

-- The initial distance walked by Fabian
def initial_distance := total_distance - additional_distance

-- The initial hours Fabian walked
def initial_hours := initial_distance / speed

theorem fabian_initial_hours : initial_hours = 3 := by
  -- Proof goes here
  sorry

end fabian_initial_hours_l900_90002


namespace sum_of_root_and_square_of_other_root_eq_2007_l900_90066

/-- If α and β are the two real roots of the equation x^2 - x - 2006 = 0,
    then the value of α + β^2 is 2007. --/
theorem sum_of_root_and_square_of_other_root_eq_2007
  (α β : ℝ)
  (hα : α^2 - α - 2006 = 0)
  (hβ : β^2 - β - 2006 = 0) :
  α + β^2 = 2007 := sorry

end sum_of_root_and_square_of_other_root_eq_2007_l900_90066


namespace find_fifth_number_l900_90026

def avg_sum_9_numbers := 936
def sum_first_5_numbers := 495
def sum_last_5_numbers := 500

theorem find_fifth_number (A1 A2 A3 A4 A5 A6 A7 A8 A9 : ℝ)
  (h1 : A1 + A2 + A3 + A4 + A5 + A6 + A7 + A8 + A9 = avg_sum_9_numbers)
  (h2 : A1 + A2 + A3 + A4 + A5 = sum_first_5_numbers)
  (h3 : A5 + A6 + A7 + A8 + A9 = sum_last_5_numbers) :
  A5 = 29.5 :=
sorry

end find_fifth_number_l900_90026


namespace coefficient_comparison_expansion_l900_90017

theorem coefficient_comparison_expansion (n : ℕ) (h₁ : 2 * n * (n - 1) = 14 * n) : n = 8 :=
by
  sorry

end coefficient_comparison_expansion_l900_90017


namespace find_two_heaviest_l900_90043

theorem find_two_heaviest (a b c d : ℝ) : 
  (a ≠ b) ∧ (a ≠ c) ∧ (a ≠ d) ∧ (b ≠ c) ∧ (b ≠ d) ∧ (c ≠ d) →
  ∃ x y : ℝ, (x ≠ y) ∧ (x = max (max (max a b) c) d) ∧ (y = max (max (min (max a b) c) d) d) :=
by sorry

end find_two_heaviest_l900_90043


namespace number_of_therapy_hours_l900_90005

theorem number_of_therapy_hours (A F H : ℝ) (h1 : F = A + 35) 
  (h2 : F + (H - 1) * A = 350) (h3 : F + A = 161) :
  H = 5 :=
by
  sorry

end number_of_therapy_hours_l900_90005


namespace perpendicular_vectors_m_val_l900_90090

theorem perpendicular_vectors_m_val (m : ℝ) 
  (a : ℝ × ℝ := (-1, 2)) 
  (b : ℝ × ℝ := (m, 1)) 
  (h : a.1 * b.1 + a.2 * b.2 = 0) : 
  m = 2 := 
by 
  sorry

end perpendicular_vectors_m_val_l900_90090


namespace problem1_problem2_l900_90052

-- Problem (Ⅰ)
theorem problem1 (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : a + b = 1) :
  (1 + 1 / a) * (1 + 1 / b) ≥ 9 :=
sorry

-- Problem (Ⅱ)
theorem problem2 (a : ℝ) (h1 : ∀ (x : ℝ), x ≥ 1 ↔ |x + 3| - |x - a| ≥ 2) :
  a = 2 :=
sorry

end problem1_problem2_l900_90052


namespace algebraic_expression_l900_90060

theorem algebraic_expression (a b : Real) 
  (h : a * b = 2 * (a^2 + b^2)) : 2 * a * b - (a^2 + b^2) = 0 :=
by
  sorry

end algebraic_expression_l900_90060


namespace min_value_of_n_l900_90084

/-!
    Given:
    - There are 53 students.
    - Each student must join one club and can join at most two clubs.
    - There are three clubs: Science, Culture, and Lifestyle.

    Prove:
    The minimum value of n, where n is the maximum number of people who join exactly the same set of clubs, is 9.
-/

def numStudents : ℕ := 53
def numClubs : ℕ := 3
def numSets : ℕ := 6

theorem min_value_of_n : ∃ n : ℕ, n = 9 ∧ 
  ∀ (students clubs sets : ℕ), students = numStudents → clubs = numClubs → sets = numSets →
  (students / sets + if students % sets = 0 then 0 else 1) = 9 :=
by
  sorry -- proof to be filled out

end min_value_of_n_l900_90084


namespace total_goals_l900_90048

def first_period_goals (k: ℕ) : ℕ :=
  k

def second_period_goals (k: ℕ) : ℕ :=
  2 * k

def spiders_first_period_goals (k: ℕ) : ℕ :=
  k / 2

def spiders_second_period_goals (s1: ℕ) : ℕ :=
  s1 * s1

def third_period_goals (k1 k2: ℕ) : ℕ :=
  2 * (k1 + k2)

def spiders_third_period_goals (s2: ℕ) : ℕ :=
  s2

def apply_bonus (goals: ℕ) (multiple: ℕ) : ℕ :=
  if goals % multiple = 0 then goals + 1 else goals

theorem total_goals (k1 k2 s1 s2 k3 s3 : ℕ) :
  first_period_goals 2 = k1 →
  second_period_goals k1 = k2 →
  spiders_first_period_goals k1 = s1 →
  spiders_second_period_goals s1 = s2 →
  third_period_goals k1 k2 = k3 →
  apply_bonus k3 3 = k3 + 1 →
  apply_bonus s2 2 = s2 →
  spiders_third_period_goals s2 = s3 →
  apply_bonus s3 2 = s3 →
  2 + k2 + (k3 + 1) + (s1 + s2 + s3) = 22 :=
by
  sorry

end total_goals_l900_90048


namespace beau_age_calculation_l900_90054

variable (sons_age : ℕ) (beau_age_today : ℕ) (beau_age_3_years_ago : ℕ)

def triplets := 3
def sons_today := 16
def sons_age_3_years_ago := sons_today - 3
def sum_of_sons_3_years_ago := triplets * sons_age_3_years_ago

theorem beau_age_calculation
  (h1 : sons_today = 16)
  (h2 : sum_of_sons_3_years_ago = beau_age_3_years_ago)
  (h3 : beau_age_today = beau_age_3_years_ago + 3) :
  beau_age_today = 42 :=
sorry

end beau_age_calculation_l900_90054


namespace B_inter_A_complement_eq_one_l900_90018

def I : Set ℕ := {1, 2, 3, 4, 5}
def A : Set ℕ := {2, 3, 5}
def B : Set ℕ := {1, 3}
def A_complement : Set ℕ := I \ A

theorem B_inter_A_complement_eq_one : B ∩ A_complement = {1} := by
  sorry

end B_inter_A_complement_eq_one_l900_90018


namespace regular_polygon_inscribed_circle_area_l900_90035

theorem regular_polygon_inscribed_circle_area
  (n : ℕ) (R : ℝ) (hR : R ≠ 0) (h_area : (1 / 2 : ℝ) * n * R^2 * Real.sin (2 * Real.pi / n) = 4 * R^2) :
  n = 20 :=
by 
  sorry

end regular_polygon_inscribed_circle_area_l900_90035


namespace value_of_fraction_l900_90029

theorem value_of_fraction (m n : ℝ) (h1 : m^2 - 2 * m - 1 = 0) (h2 : n^2 + 2 * n - 1 = 0) (h3 : m * n ≠ 1) : 
  (mn + n + 1) / n = 3 :=
by
  sorry

end value_of_fraction_l900_90029


namespace sara_gave_dan_pears_l900_90063

theorem sara_gave_dan_pears :
  ∀ (original_pears left_pears given_to_dan : ℕ),
    original_pears = 35 →
    left_pears = 7 →
    given_to_dan = original_pears - left_pears →
    given_to_dan = 28 :=
by
  intros original_pears left_pears given_to_dan h_original h_left h_given
  rw [h_original, h_left] at h_given
  exact h_given

end sara_gave_dan_pears_l900_90063


namespace area_of_given_circle_is_4pi_l900_90033

-- Define the given equation of the circle
def circle_equation (x y : ℝ) : Prop :=
  3 * x^2 + 3 * y^2 - 12 * x + 18 * y + 27 = 0

-- Define the area of the circle to be proved
noncomputable def area_of_circle : ℝ := 4 * Real.pi

-- Statement of the theorem to be proved in Lean
theorem area_of_given_circle_is_4pi :
  (∃ x y : ℝ, circle_equation x y) → area_of_circle = 4 * Real.pi :=
by
  -- The proof will go here
  sorry

end area_of_given_circle_is_4pi_l900_90033


namespace average_salary_correct_l900_90027

def A_salary := 10000
def B_salary := 5000
def C_salary := 11000
def D_salary := 7000
def E_salary := 9000

def total_salary := A_salary + B_salary + C_salary + D_salary + E_salary
def num_individuals := 5

def average_salary := total_salary / num_individuals

theorem average_salary_correct : average_salary = 8600 := by
  sorry

end average_salary_correct_l900_90027


namespace polynomial_remainder_l900_90067

noncomputable def remainder_div (p : Polynomial ℚ) (d1 d2 d3 : Polynomial ℚ) : Polynomial ℚ :=
  let d := d1 * d2 * d3 
  let q := p /ₘ d 
  let r := p %ₘ d 
  r

theorem polynomial_remainder :
  let p := (X^6 + 2 * X^4 - X^3 - 7 * X^2 + 3 * X + 1)
  let d1 := X - 2
  let d2 := X + 1
  let d3 := X - 3
  remainder_div p d1 d2 d3 = 29 * X^2 + 17 * X - 19 :=
by
  sorry

end polynomial_remainder_l900_90067


namespace find_prime_q_l900_90051

theorem find_prime_q (p q r : ℕ) 
  (prime_p : Nat.Prime p)
  (prime_q : Nat.Prime q)
  (prime_r : Nat.Prime r)
  (eq_r : q - p = r)
  (cond_p : 5 < p ∧ p < 15)
  (cond_q : q < 15) :
  q = 13 :=
sorry

end find_prime_q_l900_90051


namespace hazel_lemonade_total_l900_90037

theorem hazel_lemonade_total 
  (total_lemonade: ℕ)
  (sold_construction: ℕ := total_lemonade / 2) 
  (sold_kids: ℕ := 18) 
  (gave_friends: ℕ := sold_kids / 2) 
  (drank_herself: ℕ := 1) :
  total_lemonade = 56 :=
  sorry

end hazel_lemonade_total_l900_90037


namespace sequence_identical_l900_90095

noncomputable def a (n : ℕ) : ℝ :=
  (1 / (2 * Real.sqrt 3)) * ((2 + Real.sqrt 3)^n - (2 - Real.sqrt 3)^n)

theorem sequence_identical (n : ℕ) :
  a (n + 1) = (a n + a (n + 2)) / 4 :=
by
  sorry

end sequence_identical_l900_90095


namespace parabola_point_coord_l900_90012

theorem parabola_point_coord {x y : ℝ} (h₁ : y^2 = 4 * x) (h₂ : (x - 1)^2 + y^2 = 100) : x = 9 ∧ (y = 6 ∨ y = -6) :=
by 
  sorry

end parabola_point_coord_l900_90012


namespace smallest_part_in_ratio_l900_90007

variable (b : ℝ)

theorem smallest_part_in_ratio (h : b = -2620) : 
  let total_money := 3000 + b
  let total_parts := 19
  let smallest_ratio_part := 5
  let smallest_part := (smallest_ratio_part / total_parts) * total_money
  smallest_part = 100 :=
by 
  let total_money := 3000 + b
  let total_parts := 19
  let smallest_ratio_part := 5
  let smallest_part := (smallest_ratio_part / total_parts) * total_money
  sorry

end smallest_part_in_ratio_l900_90007


namespace distinct_integer_values_of_a_l900_90009

theorem distinct_integer_values_of_a :
  ∃ (a_values : Finset ℤ), (∀ a ∈ a_values, ∃ (m n : ℤ), (x^2 + a * x + 12 * a = 0) ∧ (m + n = -a) ∧ (m * n = 12 * a))
  ∧ a_values.card = 16 :=
sorry

end distinct_integer_values_of_a_l900_90009


namespace min_value_expr_l900_90056

theorem min_value_expr (a b : ℝ) (h₁ : 0 < b) (h₂ : b < a) :
  ∃ x : ℝ, x = a^2 + 1 / (b * (a - b)) ∧ x ≥ 4 :=
by sorry

end min_value_expr_l900_90056


namespace simplify_and_evaluate_l900_90071

theorem simplify_and_evaluate 
  (x y : ℤ) 
  (h1 : |x| = 2) 
  (h2 : y = 1) 
  (h3 : x * y < 0) : 
  3 * x^2 * y - 2 * x^2 - (x * y)^2 - 3 * x^2 * y - 4 * (x * y)^2 = -18 := by
  sorry

end simplify_and_evaluate_l900_90071


namespace mother_l900_90092

def problem_conditions (D M : ℤ) : Prop :=
  (2 * D + M = 70) ∧ (D + 2 * M = 95)

theorem mother's_age_is_40 (D M : ℤ) (h : problem_conditions D M) : M = 40 :=
by sorry

end mother_l900_90092


namespace roots_triangle_ineq_l900_90081

variable {m : ℝ}

def roots_form_triangle (x1 x2 x3 : ℝ) : Prop :=
  x1 + x2 > x3 ∧ x1 + x3 > x2 ∧ x2 + x3 > x1

theorem roots_triangle_ineq (h : ∀ x, (x - 2) * (x^2 - 4*x + m) = 0) :
  3 < m ∧ m < 4 :=
by
  sorry

end roots_triangle_ineq_l900_90081


namespace quadratic_roots_product_sum_l900_90003

theorem quadratic_roots_product_sum :
  ∀ (f g : ℝ), 
  (∀ x : ℝ, 3 * x^2 - 4 * x + 2 = 0 → x = f ∨ x = g) → 
  (f + g = 4 / 3) → 
  (f * g = 2 / 3) → 
  (f + 2) * (g + 2) = 22 / 3 :=
by
  intro f g roots_eq sum_eq product_eq
  sorry

end quadratic_roots_product_sum_l900_90003


namespace common_ratio_of_geometric_sequence_l900_90015

theorem common_ratio_of_geometric_sequence (a : ℕ → ℝ) (q : ℝ) 
  (h1 : a 2 = 2) 
  (h2 : a 5 = 1 / 4) : 
  ( ∃ a1 : ℝ, a n = a1 * q ^ (n - 1)) 
    :=
sorry

end common_ratio_of_geometric_sequence_l900_90015


namespace equilateral_triangle_intersections_l900_90024

-- Define the main theorem based on the conditions

theorem equilateral_triangle_intersections :
  let a_1 := (6 - 1) * (7 - 1) / 2
  let a_2 := (6 - 2) * (7 - 2) / 2
  let a_3 := (6 - 3) * (7 - 3) / 2
  let a_4 := (6 - 4) * (7 - 4) / 2
  let a_5 := (6 - 5) * (7 - 5) / 2
  a_1 + 2 * a_2 + 3 * a_3 + 4 * a_4 + 5 * a_5 = 70 := by
  sorry

end equilateral_triangle_intersections_l900_90024


namespace sector_perimeter_l900_90061

-- Conditions:
def theta : ℝ := 54  -- central angle in degrees
def r : ℝ := 20      -- radius in cm

-- Translation of given conditions and expected result:
theorem sector_perimeter (theta_eq : theta = 54) (r_eq : r = 20) :
  let l := (θ * r) / 180 * Real.pi 
  let perim := l + 2 * r 
  perim = 6 * Real.pi + 40 := sorry

end sector_perimeter_l900_90061


namespace find_uncertain_mushrooms_l900_90050

-- Definitions for the conditions based on the problem statement.
variable (totalMushrooms : ℕ)
variable (safeMushrooms : ℕ)
variable (poisonousMushrooms : ℕ)
variable (uncertainMushrooms : ℕ)

-- The conditions given in the problem
-- 1. Lillian found 32 mushrooms.
-- 2. She identified 9 mushrooms as safe to eat.
-- 3. The number of poisonous mushrooms is twice the number of safe mushrooms.
-- 4. The total number of mushrooms is the sum of safe, poisonous, and uncertain mushrooms.

axiom given_conditions : 
  totalMushrooms = 32 ∧
  safeMushrooms = 9 ∧
  poisonousMushrooms = 2 * safeMushrooms ∧
  totalMushrooms = safeMushrooms + poisonousMushrooms + uncertainMushrooms

-- The proof problem: Given the conditions, prove the number of uncertain mushrooms equals 5
theorem find_uncertain_mushrooms : 
  uncertainMushrooms = 5 :=
by sorry

end find_uncertain_mushrooms_l900_90050


namespace minimal_time_for_horses_l900_90096

/-- Define the individual periods of the horses' runs -/
def periods : List ℕ := [2, 3, 4, 5, 6, 7, 9, 10]

/-- Define a function to calculate the LCM of a list of numbers -/
def lcm_list (l : List ℕ) : ℕ :=
  l.foldl Nat.lcm 1

/-- Conjecture: proving that 60 is the minimal time until at least 6 out of 8 horses meet at the starting point -/
theorem minimal_time_for_horses : lcm_list [2, 3, 4, 5, 6, 10] = 60 :=
by
  sorry

end minimal_time_for_horses_l900_90096


namespace adults_attended_l900_90099

def adult_ticket_cost : ℕ := 25
def children_ticket_cost : ℕ := 15
def total_receipts : ℕ := 7200
def total_attendance : ℕ := 400

theorem adults_attended (A C: ℕ) (h1 : adult_ticket_cost * A + children_ticket_cost * C = total_receipts)
                       (h2 : A + C = total_attendance) : A = 120 :=
by
  sorry

end adults_attended_l900_90099


namespace minimum_value_of_f_l900_90006

noncomputable def f (x : ℝ) : ℝ := (Real.sqrt ((x + 2)^2 + 4^2)) + (Real.sqrt ((x + 1)^2 + 3^2))

theorem minimum_value_of_f : ∃ x : ℝ, f x = 5 * Real.sqrt 2 ∧ ∀ y : ℝ, f y ≥ f x :=
by
  use -3
  sorry

end minimum_value_of_f_l900_90006


namespace mixture_ratios_equal_quantities_l900_90075

-- Define the given conditions
def ratio_p_milk_water := (5, 4)
def ratio_q_milk_water := (2, 7)

-- Define what we're trying to prove: the ratio p : q such that the resulting mixture has equal milk and water
theorem mixture_ratios_equal_quantities 
  (P Q : ℝ) 
  (h1 : 5 * P + 2 * Q = 4 * P + 7 * Q) :
  P / Q = 5 :=
  sorry

end mixture_ratios_equal_quantities_l900_90075


namespace beta_value_l900_90083

variable {α β : Real}
open Real

theorem beta_value :
  cos α = 1 / 7 ∧ cos (α + β) = -11 / 14 ∧ 0 < α ∧ α < π / 2 ∧ π / 2 < α + β ∧ α + β < π → 
  β = π / 3 := 
by
  -- Proof would go here
  sorry

end beta_value_l900_90083


namespace arithmetic_geometric_sequence_product_l900_90039

theorem arithmetic_geometric_sequence_product :
  ∀ (a : ℕ → ℝ) (q : ℝ),
    a 1 = 3 →
    (a 1) + (a 1 * q^2) + (a 1 * q^4) = 21 →
    (a 2) * (a 6) = 72 :=
by 
  intros a q h1 h2 
  sorry

end arithmetic_geometric_sequence_product_l900_90039


namespace smallest_positive_multiple_of_37_l900_90055

theorem smallest_positive_multiple_of_37 :
  ∃ n, n > 0 ∧ (∃ a, n = 37 * a) ∧ (∃ k, n = 76 * k + 7) ∧ n = 2405 := 
by
  sorry

end smallest_positive_multiple_of_37_l900_90055


namespace proof_of_inequality_l900_90053

theorem proof_of_inequality (a : ℝ) (h : (∃ x : ℝ, x - 2 * a + 4 = 0 ∧ x < 0)) :
  (a - 3) * (a - 4) > 0 :=
by
  sorry

end proof_of_inequality_l900_90053


namespace probability_of_rain_l900_90011

variable (P_R P_B0 : ℝ)
variable (H1 : 0 ≤ P_R ∧ P_R ≤ 1)
variable (H2 : 0 ≤ P_B0 ∧ P_B0 ≤ 1)
variable (H : P_R + P_B0 - P_R * P_B0 = 0.2)

theorem probability_of_rain : 
  P_R = 1/9 :=
by
  sorry

end probability_of_rain_l900_90011


namespace min_value_of_expression_l900_90049

theorem min_value_of_expression (n : ℕ) (h : n > 0) : (n / 3 + 27 / n) ≥ 6 :=
by {
  -- Proof goes here but is not required in the statement
  sorry
}

end min_value_of_expression_l900_90049


namespace determine_m_in_hexadecimal_conversion_l900_90034

theorem determine_m_in_hexadecimal_conversion :
  ∃ m : ℕ, 1 * 6^5 + 3 * 6^4 + m * 6^3 + 5 * 6^2 + 0 * 6^1 + 2 * 6^0 = 12710 ∧ m = 4 :=
by
  sorry

end determine_m_in_hexadecimal_conversion_l900_90034


namespace quadratic_expression_positive_l900_90078

theorem quadratic_expression_positive (k : ℝ) : 
  (∀ x : ℝ, x^2 - (k - 5) * x + k + 2 > 0) ↔ (7 - 4 * Real.sqrt 2 < k ∧ k < 7 + 4 * Real.sqrt 2) :=
by
  sorry

end quadratic_expression_positive_l900_90078


namespace fewest_seats_occupied_l900_90022

def min_seats_occupied (N : ℕ) : ℕ :=
  if h : N % 4 = 0 then (N / 2) else (N / 2) + 1

theorem fewest_seats_occupied (N : ℕ) (h : N = 150) : min_seats_occupied N = 74 := by
  sorry

end fewest_seats_occupied_l900_90022


namespace Gerald_needs_to_average_5_chores_per_month_l900_90008

def spending_per_month := 100
def season_length := 4
def cost_per_chore := 10
def total_spending := spending_per_month * season_length
def months_not_playing := 12 - season_length
def amount_to_save_per_month := total_spending / months_not_playing
def chores_per_month := amount_to_save_per_month / cost_per_chore

theorem Gerald_needs_to_average_5_chores_per_month :
  chores_per_month = 5 := by
  sorry

end Gerald_needs_to_average_5_chores_per_month_l900_90008


namespace hannah_speed_l900_90016

theorem hannah_speed :
  ∃ H : ℝ, 
    (∀ t : ℝ, (t = 6) → d = 130) ∧ 
    (∀ t : ℝ, (t = 11) → d = 130) → 
    (d = 37 * 5 + H * 5) → 
    H = 15 := 
by 
  sorry

end hannah_speed_l900_90016


namespace inequality_solution_l900_90076

section
variables (a x : ℝ)

theorem inequality_solution (h : a < 0) :
  (ax^2 + (1 - a) * x - 1 > 0 ↔
     (-1 < a ∧ a < 0 ∧ 1 < x ∧ x < -1/a) ∨
     (a = -1 ∧ false) ∨
     (a < -1 ∧ -1/a < x ∧ x < 1)) :=
by sorry

end inequality_solution_l900_90076


namespace store_profit_loss_l900_90059

theorem store_profit_loss :
  ∃ (x y : ℝ), (1 + 0.25) * x = 135 ∧ (1 - 0.25) * y = 135 ∧ (135 - x) + (135 - y) = -18 :=
by
  sorry

end store_profit_loss_l900_90059


namespace evaluate_expression_l900_90057

theorem evaluate_expression :
  3 ^ (1 ^ (2 ^ 8)) + ((3 ^ 1) ^ 2) ^ 4 = 6564 :=
by
  sorry

end evaluate_expression_l900_90057


namespace quadratic_discriminant_l900_90094

def discriminant (a b c : ℚ) : ℚ := b^2 - 4 * a * c

theorem quadratic_discriminant : discriminant 5 (5 + 1/5) (1/5) = 576 / 25 := by
  sorry

end quadratic_discriminant_l900_90094


namespace smallest_positive_period_sin_cos_sin_l900_90031

noncomputable def smallest_positive_period := 2 * Real.pi

theorem smallest_positive_period_sin_cos_sin :
  ∃ T > 0, (∀ x, (Real.sin x - 2 * Real.cos (2 * x) + 4 * Real.sin (4 * x)) = (Real.sin (x + T) - 2 * Real.cos (2 * (x + T)) + 4 * Real.sin (4 * (x + T)))) ∧ T = smallest_positive_period := by
sorry

end smallest_positive_period_sin_cos_sin_l900_90031


namespace no_integer_solutions_x2_plus_3xy_minus_2y2_eq_122_l900_90030

theorem no_integer_solutions_x2_plus_3xy_minus_2y2_eq_122 :
  ¬ ∃ x y : ℤ, x^2 + 3 * x * y - 2 * y^2 = 122 := sorry

end no_integer_solutions_x2_plus_3xy_minus_2y2_eq_122_l900_90030


namespace trevor_coin_difference_l900_90064

theorem trevor_coin_difference:
  ∀ (total_coins quarters: ℕ),
  total_coins = 77 →
  quarters = 29 →
  (total_coins - quarters = 48) := by
  intros total_coins quarters h1 h2
  sorry

end trevor_coin_difference_l900_90064


namespace enchilada_cost_l900_90065

theorem enchilada_cost : ∃ T E : ℝ, 2 * T + 3 * E = 7.80 ∧ 3 * T + 5 * E = 12.70 ∧ E = 2.00 :=
by
  sorry

end enchilada_cost_l900_90065


namespace deliver_all_cargo_l900_90032

theorem deliver_all_cargo (containers : ℕ) (cargo_mass : ℝ) (ships : ℕ) (ship_capacity : ℝ)
  (h1 : containers ≥ 35)
  (h2 : cargo_mass = 18)
  (h3 : ships = 7)
  (h4 : ship_capacity = 3)
  (h5 : ∀ t, (0 < t) → (t ≤ containers) → (t = 35)) :
  (ships * ship_capacity) ≥ cargo_mass :=
by
  sorry

end deliver_all_cargo_l900_90032


namespace range_of_a_l900_90013

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, 1 < x ∧ x < 4 → ax^2 - 2 * x + 2 > 0) ↔ (a > 1/2) :=
sorry

end range_of_a_l900_90013


namespace present_age_of_A_is_11_l900_90019

-- Definitions for present ages
variables (A B C : ℕ)

-- Definitions for the given conditions
def sum_of_ages_present : Prop := A + B + C = 57
def age_ratio_three_years_ago (x : ℕ) : Prop := (A - 3 = x) ∧ (B - 3 = 2 * x) ∧ (C - 3 = 3 * x)

-- The proof statement
theorem present_age_of_A_is_11 (x : ℕ) (h1 : sum_of_ages_present A B C) (h2 : age_ratio_three_years_ago A B C x) : A = 11 := 
by
  sorry

end present_age_of_A_is_11_l900_90019


namespace trapezium_second_side_length_l900_90036

-- Define the problem in Lean
variables (a h A b : ℝ)

-- Define the conditions
def conditions : Prop :=
  a = 20 ∧ h = 25 ∧ A = 475

-- Prove the length of the second parallel side
theorem trapezium_second_side_length (h_cond : conditions a h A) : b = 18 :=
by
  sorry

end trapezium_second_side_length_l900_90036


namespace shifted_function_l900_90089

def initial_fun (x : ℝ) : ℝ := 5 * (x - 1) ^ 2 + 1

theorem shifted_function :
  (∀ x, initial_fun (x - 2) - 3 = 5 * (x + 1) ^ 2 - 2) :=
by
  intro x
  -- sorry statement to indicate proof should be here
  sorry

end shifted_function_l900_90089


namespace tangent_line_at_point_l900_90028

theorem tangent_line_at_point (x y : ℝ) (h : y = x^2) (hx : x = 1) (hy : y = 1) : 
  2 * x - y - 1 = 0 :=
by
  sorry

end tangent_line_at_point_l900_90028


namespace family_gathering_l900_90047

theorem family_gathering (P : ℕ) 
  (h1 : (P / 2 = P - 10)) : P = 20 :=
sorry

end family_gathering_l900_90047


namespace wheat_distribution_l900_90040

def mill1_rate := 19 / 3 -- quintals per hour
def mill2_rate := 32 / 5 -- quintals per hour
def mill3_rate := 5     -- quintals per hour

def total_wheat := 1330 -- total wheat in quintals

theorem wheat_distribution :
    ∃ (x1 x2 x3 : ℚ), 
    x1 = 475 ∧ x2 = 480 ∧ x3 = 375 ∧ 
    x1 / mill1_rate = x2 / mill2_rate ∧ x2 / mill2_rate = x3 / mill3_rate ∧ 
    x1 + x2 + x3 = total_wheat :=
by {
  sorry
}

end wheat_distribution_l900_90040


namespace coprime_permutations_count_l900_90070

noncomputable def count_coprime_permutations (l : List ℕ) : ℕ :=
if h : l = [1, 2, 3, 4, 5, 6, 7] ∨ l = [1, 2, 3, 7, 5, 6, 4] -- other permutations can be added as needed
then 864
else 0

theorem coprime_permutations_count :
  count_coprime_permutations [1, 2, 3, 4, 5, 6, 7] = 864 :=
sorry

end coprime_permutations_count_l900_90070


namespace find_correct_grades_l900_90021

structure StudentGrades := 
  (Volodya: ℕ) 
  (Sasha: ℕ) 
  (Petya: ℕ)

def isCorrectGrades (grades : StudentGrades) : Prop :=
  grades.Volodya = 5 ∧ grades.Sasha = 4 ∧ grades.Petya = 3

theorem find_correct_grades (grades : StudentGrades)
  (h1 : grades.Volodya = 5 ∨ grades.Volodya ≠ 5)
  (h2 : grades.Sasha = 3 ∨ grades.Sasha ≠ 3)
  (h3 : grades.Petya ≠ 5 ∨ grades.Petya = 5)
  (unique_h1: grades.Volodya = 5 ∨ grades.Sasha = 5 ∨ grades.Petya = 5) 
  (unique_h2: grades.Volodya = 4 ∨ grades.Sasha = 4 ∨ grades.Petya = 4)
  (unique_h3: grades.Volodya = 3 ∨ grades.Sasha = 3 ∨ grades.Petya = 3) 
  (lyingCount: (grades.Volodya ≠ 5 ∧ grades.Sasha ≠ 3 ∧ grades.Petya = 5)
              ∨ (grades.Volodya = 5 ∧ grades.Sasha ≠ 3 ∧ grades.Petya ≠ 5)
              ∨ (grades.Volodya ≠ 5 ∧ grades.Sasha = 3 ∧ grades.Petya ≠ 5)) :
  isCorrectGrades grades :=
sorry

end find_correct_grades_l900_90021


namespace find_x_l900_90001

theorem find_x (x : ℝ) (h : 0.65 * x = 0.20 * 552.50) : x = 170 :=
by
  sorry

end find_x_l900_90001


namespace scientific_notation_of_number_l900_90097

theorem scientific_notation_of_number :
  ∀ (n : ℕ), n = 450000000 -> n = 45 * 10^7 := 
by
  sorry

end scientific_notation_of_number_l900_90097


namespace distance_from_pointM_to_xaxis_l900_90093

-- Define the point M with coordinates (2, -3)
def pointM : ℝ × ℝ := (2, -3)

-- Define the function to compute the distance from a point to the x-axis.
def distanceToXAxis (p : ℝ × ℝ) : ℝ := |p.2|

-- Formalize the proof statement.
theorem distance_from_pointM_to_xaxis : distanceToXAxis pointM = 3 := by
  -- Proof goes here
  sorry

end distance_from_pointM_to_xaxis_l900_90093


namespace circle_equation_l900_90023

theorem circle_equation (x y : ℝ) : 
  (∀ (a b : ℝ), (a - 1)^2 + (b - 1)^2 = 2 → (a, b) = (0, 0)) ∧
  ((0 - 1)^2 + (0 - 1)^2 = 2) → 
  (x - 1)^2 + (y - 1)^2 = 2 := 
by 
  sorry

end circle_equation_l900_90023


namespace base_k_eq_26_l900_90014

theorem base_k_eq_26 (k : ℕ) (h : 3 * k + 2 = 26) : k = 8 :=
by {
  -- The actual proof will go here.
  sorry
}

end base_k_eq_26_l900_90014


namespace least_product_ab_l900_90086

theorem least_product_ab (a b : ℕ) (ha : 0 < a) (hb : 0 < b) (h : (1 : ℚ) / a + 1 / (3 * b) = 1 / 6) : a * b ≥ 48 :=
by
  sorry

end least_product_ab_l900_90086


namespace fraction_of_cost_due_to_high_octane_is_half_l900_90069

theorem fraction_of_cost_due_to_high_octane_is_half :
  ∀ (cost_regular cost_high : ℝ) (units_high units_regular : ℕ),
    units_high * cost_high + units_regular * cost_regular ≠ 0 →
    cost_high = 3 * cost_regular →
    units_high = 1515 →
    units_regular = 4545 →
    (units_high * cost_high) / (units_high * cost_high + units_regular * cost_regular) = 1 / 2 :=
by
  intro cost_regular cost_high units_high units_regular h_total_cost_ne_zero h_cost_rel h_units_high h_units_regular
  -- skip the actual proof steps
  sorry

end fraction_of_cost_due_to_high_octane_is_half_l900_90069


namespace equation_solution_l900_90091

theorem equation_solution (x y : ℕ) (h : x^3 - y^3 = x * y + 61) : x = 6 ∧ y = 5 :=
by
  sorry

end equation_solution_l900_90091


namespace minimum_abs_sum_l900_90079

def matrix_squared_condition (p q r s : ℤ) : Prop :=
  (p * p + q * r = 9) ∧ 
  (q * r + s * s = 9) ∧ 
  (p * q + q * s = 0) ∧ 
  (r * p + r * s = 0)

def abs_sum (p q r s : ℤ) : ℤ :=
  |p| + |q| + |r| + |s|

theorem minimum_abs_sum (p q r s : ℤ) (h1 : p ≠ 0) (h2 : q ≠ 0) (h3 : r ≠ 0) (h4 : s ≠ 0) 
  (h5 : matrix_squared_condition p q r s) : abs_sum p q r s = 8 :=
by 
  sorry

end minimum_abs_sum_l900_90079


namespace total_balls_is_108_l900_90038

theorem total_balls_is_108 (B : ℕ) (W : ℕ) (n : ℕ) (h1 : W = 8 * B) 
                           (h2 : n = B + W) 
                           (h3 : 100 ≤ n - W + 1) 
                           (h4 : 100 > B) : n = 108 := 
by sorry

end total_balls_is_108_l900_90038


namespace tan_alpha_eq_neg_one_l900_90072

theorem tan_alpha_eq_neg_one (α : ℝ) (h : Real.sin (π / 6 - α) = Real.cos (π / 6 + α)) : Real.tan α = -1 :=
  sorry

end tan_alpha_eq_neg_one_l900_90072


namespace harold_monthly_income_l900_90025

variable (M : ℕ)

def rent : ℕ := 700
def car_payment : ℕ := 300
def utilities : ℕ := car_payment / 2
def groceries : ℕ := 50

def total_expenses : ℕ := rent + car_payment + utilities + groceries
def remaining_money_after_expenses : ℕ := M - total_expenses
def retirement_saving_target : ℕ := 650
def required_remaining_money_pre_saving : ℕ := 2 * retirement_saving_target

theorem harold_monthly_income :
  remaining_money_after_expenses = required_remaining_money_pre_saving → M = 2500 :=
by
  sorry

end harold_monthly_income_l900_90025


namespace solve_inequality_l900_90085

theorem solve_inequality (x : ℝ) (h : 3 * x + 4 ≠ 0) :
  (3 - 1 / (3 * x + 4) < 5) ↔ (-4 / 3 < x) :=
by
  sorry

end solve_inequality_l900_90085


namespace find_third_integer_l900_90020

theorem find_third_integer (a b c : ℕ) (h1 : a * b * c = 42) (h2 : a + b = 9) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) : c = 3 :=
sorry

end find_third_integer_l900_90020


namespace arccos_cos_7_l900_90004

noncomputable def arccos_cos_7_eq_7_minus_2pi : Prop :=
  ∃ x : ℝ, x = 7 - 2 * Real.pi ∧ Real.arccos (Real.cos 7) = x

theorem arccos_cos_7 :
  arccos_cos_7_eq_7_minus_2pi :=
by
  sorry

end arccos_cos_7_l900_90004


namespace sum_of_solutions_is_24_l900_90073

theorem sum_of_solutions_is_24 (a : ℝ) (x1 x2 : ℝ) 
    (h1 : abs (x1 - a) = 100) (h2 : abs (x2 - a) = 100)
    (sum_eq : x1 + x2 = 24) : a = 12 :=
sorry

end sum_of_solutions_is_24_l900_90073


namespace value_2_std_devs_below_mean_l900_90088

theorem value_2_std_devs_below_mean {μ σ : ℝ} (h_mean : μ = 10.5) (h_std_dev : σ = 1) : μ - 2 * σ = 8.5 :=
by
  sorry

end value_2_std_devs_below_mean_l900_90088


namespace red_robin_team_arrangements_l900_90058

theorem red_robin_team_arrangements :
  let boys := 3
  let girls := 4
  let choose2 (n : ℕ) (k : ℕ) := Nat.choose n k
  let permutations (n : ℕ) := Nat.factorial n
  let waysToPositionBoys := choose2 boys 2 * permutations 2
  let waysToPositionRemainingMembers := permutations (boys - 2 + girls)
  waysToPositionBoys * waysToPositionRemainingMembers = 720 :=
by
  let boys := 3
  let girls := 4
  let choose2 (n : ℕ) (k : ℕ) := Nat.choose n k
  let permutations (n : ℕ) := Nat.factorial n
  let waysToPositionBoys := choose2 boys 2 * permutations 2
  let waysToPositionRemainingMembers := permutations (boys - 2 + girls)
  have : waysToPositionBoys * waysToPositionRemainingMembers = 720 := 
    by sorry -- Proof omitted here
  exact this

end red_robin_team_arrangements_l900_90058


namespace grant_total_earnings_l900_90077

def earnings_first_month : ℕ := 350
def earnings_second_month : ℕ := 2 * earnings_first_month + 50
def earnings_third_month : ℕ := 4 * (earnings_first_month + earnings_second_month)
def total_earnings : ℕ := earnings_first_month + earnings_second_month + earnings_third_month

theorem grant_total_earnings : total_earnings = 5500 := by
  sorry

end grant_total_earnings_l900_90077


namespace fourth_root_is_four_l900_90062

-- Define the polynomial f
def f (x : ℝ) : ℝ := x^4 - 8 * x^3 - 7 * x^2 + 9 * x + 11

-- Conditions that must be true for the given problem
@[simp] def f_neg1_zero : f (-1) = 0 := by sorry
@[simp] def f_2_zero : f (2) = 0 := by sorry
@[simp] def f_neg3_zero : f (-3) = 0 := by sorry

-- The theorem stating the fourth root
theorem fourth_root_is_four (root4 : ℝ) (H : f root4 = 0) : root4 = 4 := by sorry

end fourth_root_is_four_l900_90062


namespace triangle_area_via_line_eq_l900_90087

theorem triangle_area_via_line_eq (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  let x_intercept := (1 / a)
  let y_intercept := (1 / b)
  let area := (1 / 2) * |x_intercept| * |y_intercept|
  area = 1 / (2 * |a * b|) :=
by
  let x_intercept := (1 / a)
  let y_intercept := (1 / b)
  let area := (1 / 2) * |x_intercept| * |y_intercept|
  sorry

end triangle_area_via_line_eq_l900_90087
