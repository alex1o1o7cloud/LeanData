import Mathlib

namespace flood_damage_conversion_l1097_109734

-- Define the conversion rate and the damage in Indian Rupees as given
def rupees_to_pounds (rupees : ℕ) : ℕ := rupees / 75
def damage_in_rupees : ℕ := 45000000

-- Define the expected damage in British Pounds
def expected_damage_in_pounds : ℕ := 600000

-- The theorem to prove that the damage in British Pounds is as expected, given the conditions.
theorem flood_damage_conversion :
  rupees_to_pounds damage_in_rupees = expected_damage_in_pounds :=
by
  -- The proof goes here, but we'll use sorry to skip it as instructed.
  sorry

end flood_damage_conversion_l1097_109734


namespace sum_a_c_eq_13_l1097_109770

noncomputable def conditions (a b c d k : ℤ) :=
  d = a * b * c ∧
  1 < a ∧ a < b ∧ b < c ∧
  233 = d * k + 79

theorem sum_a_c_eq_13 (a b c d k : ℤ) (h : conditions a b c d k) : a + c = 13 := by
  sorry

end sum_a_c_eq_13_l1097_109770


namespace value_A_minus_B_l1097_109752

-- Conditions definitions
def A : ℕ := (1 * 1000) + (16 * 100) + (28 * 10)
def B : ℕ := 355 + 245 * 3

-- Theorem statement
theorem value_A_minus_B : A - B = 1790 := by
  sorry

end value_A_minus_B_l1097_109752


namespace f_at_neg_2_l1097_109715

noncomputable def f (x : ℝ) (a b : ℝ) : ℝ := x^5 + a * x^3 + x^2 + b * x + 2

-- Given the condition
def f_at_2_eq_3 (a b : ℝ) : Prop := f 2 a b = 3

-- Prove the value of f(-2)
theorem f_at_neg_2 (a b : ℝ) (h : f_at_2_eq_3 a b) : f (-2) a b = 1 :=
sorry

end f_at_neg_2_l1097_109715


namespace ratio_of_a_b_to_b_c_l1097_109729

theorem ratio_of_a_b_to_b_c (a b c : ℝ) (h₁ : b / a = 3) (h₂ : c / b = 2) : 
  (a + b) / (b + c) = 4 / 9 := by
  sorry

end ratio_of_a_b_to_b_c_l1097_109729


namespace first_divisor_l1097_109788

-- Definitions
def is_divisible_by (n d : ℕ) : Prop := d > 0 ∧ n % d = 0

-- Theorem to prove
theorem first_divisor (x : ℕ) (h₁ : ∃ l, l = Nat.lcm x 35 ∧ is_divisible_by 1400 l ∧ 1400 / l = 8) : 
  x = 25 := 
sorry

end first_divisor_l1097_109788


namespace largest_three_digit_number_l1097_109756

theorem largest_three_digit_number :
  ∃ (n : ℕ), (n < 1000) ∧ (n % 7 = 1) ∧ (n % 8 = 4) ∧ (∀ (m : ℕ), (m < 1000) ∧ (m % 7 = 1) ∧ (m % 8 = 4) → m ≤ n) :=
sorry

end largest_three_digit_number_l1097_109756


namespace hypotenuse_length_l1097_109780

theorem hypotenuse_length (x y : ℝ) 
  (h1 : (1/3) * Real.pi * y^2 * x = 1080 * Real.pi) 
  (h2 : (1/3) * Real.pi * x^2 * y = 2430 * Real.pi) : 
  Real.sqrt (x^2 + y^2) = 6 * Real.sqrt 13 := 
  sorry

end hypotenuse_length_l1097_109780


namespace dennis_years_of_teaching_l1097_109779

variable (V A D E N : ℕ)

def combined_years_taught : Prop :=
  V + A + D + E + N = 225

def virginia_adrienne_relation : Prop :=
  V = A + 9

def virginia_dennis_relation : Prop :=
  V = D - 15

def elijah_adrienne_relation : Prop :=
  E = A - 3

def elijah_nadine_relation : Prop :=
  E = N + 7

theorem dennis_years_of_teaching 
  (h1 : combined_years_taught V A D E N) 
  (h2 : virginia_adrienne_relation V A)
  (h3 : virginia_dennis_relation V D)
  (h4 : elijah_adrienne_relation E A) 
  (h5 : elijah_nadine_relation E N) : 
  D = 65 :=
  sorry

end dennis_years_of_teaching_l1097_109779


namespace parabola_vertex_sum_l1097_109761

variable (a b c : ℝ)

def parabola_eq (x y : ℝ) : Prop :=
  x = a * y^2 + b * y + c

def vertex (v : ℝ × ℝ) : Prop :=
  v = (-3, 2)

def passes_through (p : ℝ × ℝ) : Prop :=
  p = (-1, 0)

theorem parabola_vertex_sum :
  ∀ (a b c : ℝ),
  (∃ v : ℝ × ℝ, vertex v) ∧
  (∃ p : ℝ × ℝ, passes_through p) →
  a + b + c = -7/2 :=
by
  intros a b c
  intro conditions
  sorry

end parabola_vertex_sum_l1097_109761


namespace geometric_to_arithmetic_l1097_109731

theorem geometric_to_arithmetic (a_1 a_2 a_3 b_1 b_2 b_3: ℝ) (ha: a_1 > 0 ∧ a_2 > 0 ∧ a_3 > 0 ∧ b_1 > 0 ∧ b_2 > 0 ∧ b_3 > 0)
  (h_geometric_a : ∃ q : ℝ, a_2 = a_1 * q ∧ a_3 = a_1 * q^2)
  (h_geometric_b : ∃ q₁ : ℝ, b_2 = b_1 * q₁ ∧ b_3 = b_1 * q₁^2)
  (h_sum : a_1 + a_2 + a_3 = b_1 + b_2 + b_3)
  (h_arithmetic : 2 * a_2 * b_2 = a_1 * b_1 + a_3 * b_3) : 
  a_2 = b_2 :=
by
  sorry

end geometric_to_arithmetic_l1097_109731


namespace y_intercept_with_z_3_l1097_109746

theorem y_intercept_with_z_3 : 
  ∀ x y : ℝ, (4 * x + 6 * y - 2 * 3 = 24) → (x = 0) → y = 5 :=
by
  intros x y h1 h2
  sorry

end y_intercept_with_z_3_l1097_109746


namespace quadratic_inequality_solution_range_l1097_109797

theorem quadratic_inequality_solution_range (a : ℝ) :
  (∀ x : ℝ, a*x^2 + 2*a*x - 4 < 0) ↔ -4 < a ∧ a < 0 := 
by
  sorry

end quadratic_inequality_solution_range_l1097_109797


namespace part1_part2_l1097_109707

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := Real.exp (x - 1) + a
noncomputable def g (x : ℝ) (a : ℝ) : ℝ := a * x + Real.log x

theorem part1 (x : ℝ) (hx : 0 < x) :
  f x 0 ≥ g x 0 + 1 := sorry

theorem part2 {x0 : ℝ} (hx0 : ∃ y0 : ℝ, f x0 0 = g x0 0 ∧ ∀ x ≠ x0, f x 0 ≠ g x 0) :
  x0 < 2 := sorry

end part1_part2_l1097_109707


namespace bill_took_six_naps_l1097_109747

def total_hours (days : Nat) : Nat := days * 24

def hours_left (total : Nat) (worked : Nat) : Nat := total - worked

def naps_taken (remaining : Nat) (duration : Nat) : Nat := remaining / duration

theorem bill_took_six_naps :
  let days := 4
  let hours_worked := 54
  let nap_duration := 7
  naps_taken (hours_left (total_hours days) hours_worked) nap_duration = 6 := 
by {
  sorry
}

end bill_took_six_naps_l1097_109747


namespace rectangle_solution_l1097_109719

-- Define the given conditions
variables (x y : ℚ)

-- Given equations
def condition1 := (Real.sqrt (x - y) = 2 / 5)
def condition2 := (Real.sqrt (x + y) = 2)

-- Solution
theorem rectangle_solution (x y : ℚ) (h1 : condition1 x y) (h2 : condition2 x y) : 
  x = 52 / 25 ∧ y = 48 / 25 ∧ (Real.sqrt ((52 / 25) * (48 / 25)) = 8 / 25) :=
by
  sorry

end rectangle_solution_l1097_109719


namespace find_x_l1097_109710

theorem find_x (x : ℕ) (hx : x > 0) : 1^(x + 3) + 2^(x + 2) + 3^x + 4^(x + 1) = 1958 → x = 4 :=
sorry

end find_x_l1097_109710


namespace problem_statement_l1097_109765

noncomputable def f (x : ℝ) : ℝ := x - Real.log x
noncomputable def g (x : ℝ) : ℝ := Real.log x / x

theorem problem_statement (x : ℝ) (h : 0 < x ∧ x ≤ Real.exp 1) : 
  f x > g x + 1/2 :=
sorry

end problem_statement_l1097_109765


namespace solution_exists_for_any_y_l1097_109702

theorem solution_exists_for_any_y (z : ℝ) : (∀ y : ℝ, ∃ x : ℝ, x^2 + y^2 + 4*z^2 + 2*x*y*z - 9 = 0) ↔ |z| ≤ 3 / 2 := 
sorry

end solution_exists_for_any_y_l1097_109702


namespace sequence_4951_l1097_109723

theorem sequence_4951 :
  (∃ (a : ℕ → ℕ), a 1 = 1 ∧ (∀ n : ℕ, 0 < n → a (n + 1) = a n + n) ∧ a 100 = 4951) :=
sorry

end sequence_4951_l1097_109723


namespace radius_of_inner_circle_l1097_109758

theorem radius_of_inner_circle (R a x : ℝ) (hR : 0 < R) (ha : 0 ≤ a) (haR : a < R) :
  (a ≠ R ∧ a ≠ 0) → x = (R^2 - a^2) / (2 * R) :=
by
  sorry

end radius_of_inner_circle_l1097_109758


namespace range_of_y_l1097_109794

theorem range_of_y (y : ℝ) (h₁ : y < 0) (h₂ : ⌈y⌉ * ⌊y⌋ = 110) : -11 < y ∧ y < -10 := 
sorry

end range_of_y_l1097_109794


namespace crayons_initial_total_l1097_109784

theorem crayons_initial_total 
  (lost_given : ℕ) (left : ℕ) (initial : ℕ) 
  (h1 : lost_given = 70) (h2 : left = 183) : 
  initial = lost_given + left := 
by
  sorry

end crayons_initial_total_l1097_109784


namespace find_c_l1097_109763

variable (a b c : ℕ)

theorem find_c (h1 : a = 9) (h2 : b = 2) (h3 : Odd c) (h4 : a + b > c) (h5 : a - b < c) (h6 : b + c > a) (h7 : b - c < a) : c = 9 :=
sorry

end find_c_l1097_109763


namespace product_of_first_nine_terms_l1097_109754

-- Declare the geometric sequence and given condition
variable {α : Type*} [Field α]
variable {a : ℕ → α}
variable (r : α) (a1 : α)

-- Define that the sequence is geometric
def is_geometric_sequence (a : ℕ → α) (r : α) (a1 : α) : Prop :=
  ∀ n : ℕ, a n = a1 * r ^ n

-- Given a_5 = -2 in the sequence
def geometric_sequence_with_a5 (a : ℕ → α) (r : α) (a1 : α) : Prop :=
  is_geometric_sequence a r a1 ∧ a 5 = -2

-- Prove that the product of the first 9 terms is -512
theorem product_of_first_nine_terms 
  (a : ℕ → α) 
  (r : α) 
  (a₁ : α) 
  (h : geometric_sequence_with_a5 a r a₁) : 
  (a 0) * (a 1) * (a 2) * (a 3) * (a 4) * (a 5) * (a 6) * (a 7) * (a 8) = -512 := 
by
  sorry

end product_of_first_nine_terms_l1097_109754


namespace find_a5_l1097_109796

-- Define an arithmetic sequence with a given common difference
def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) := ∀ n : ℕ, a (n + 1) = a n + d

-- Define that three terms form a geometric sequence
def geometric_sequence (x y z : ℝ) := y^2 = x * z

-- Given conditions for the problem
def a₁ : ℝ := 1  -- found from the geometric sequence condition
def d : ℝ := 2

-- The definition of the sequence {a_n} based on the common difference
noncomputable def a_n (n : ℕ) : ℝ := a₁ + n * d

-- Given that a_1, a_2, a_5 form a geometric sequence
axiom geo_progression : geometric_sequence a₁ (a_n 1) (a_n 4)

-- The proof goal
theorem find_a5 : a_n 4 = 9 :=
by
  -- the proof is skipped
  sorry

end find_a5_l1097_109796


namespace find_k_l1097_109751

theorem find_k (k : ℝ) (h : (-2)^2 - k * (-2) - 6 = 0) : k = 1 :=
by
  sorry

end find_k_l1097_109751


namespace combined_volume_of_all_cubes_l1097_109776

/-- Lily has 4 cubes each with side length 3, Mark has 3 cubes each with side length 4,
    and Zoe has 2 cubes each with side length 5. Prove that the combined volume of all
    the cubes is 550. -/
theorem combined_volume_of_all_cubes 
  (lily_cubes : ℕ := 4) (lily_side_length : ℕ := 3)
  (mark_cubes : ℕ := 3) (mark_side_length : ℕ := 4)
  (zoe_cubes : ℕ := 2) (zoe_side_length : ℕ := 5) :
  (lily_cubes * lily_side_length ^ 3) + 
  (mark_cubes * mark_side_length ^ 3) + 
  (zoe_cubes * zoe_side_length ^ 3) = 550 :=
by
  have lily_volume : ℕ := lily_cubes * lily_side_length ^ 3
  have mark_volume : ℕ := mark_cubes * mark_side_length ^ 3
  have zoe_volume : ℕ := zoe_cubes * zoe_side_length ^ 3
  have total_volume : ℕ := lily_volume + mark_volume + zoe_volume
  sorry

end combined_volume_of_all_cubes_l1097_109776


namespace quadratic_real_roots_and_a_value_l1097_109721

-- Define the quadratic equation (a-5)x^2 - 4x - 1 = 0
def quadratic_eq (a : ℝ) (x : ℝ) := (a - 5) * x^2 - 4 * x - 1

-- Define the discriminant for the quadratic equation
def discriminant (a : ℝ) := 4 - 4 * (a - 5) * (-1)

-- Main theorem statement
theorem quadratic_real_roots_and_a_value
    (a : ℝ) (x1 x2 : ℝ) 
    (h_roots : (a - 5) ≠ 0)
    (h_eq : quadratic_eq a x1 = 0 ∧ quadratic_eq a x2 = 0)
    (h_sum_product : x1 + x2 + x1 * x2 = 3) :
    (a ≥ 1) ∧ (a = 6) :=
  sorry

end quadratic_real_roots_and_a_value_l1097_109721


namespace exists_integer_point_touching_x_axis_l1097_109735

-- Define the context for the problem
variable {p q : ℤ}

-- Condition: The quadratic trinomial touches x-axis, i.e., discriminant is zero.
axiom discriminant_zero (p q : ℤ) : p^2 - 4 * q = 0

-- Theorem statement: Proving the existence of such an integer point.
theorem exists_integer_point_touching_x_axis :
  ∃ a b : ℤ, (a = -p ∧ b = q) ∧ (∀ (x : ℝ), x^2 + a * x + b = 0 → (a * a - 4 * b) = 0) :=
sorry

end exists_integer_point_touching_x_axis_l1097_109735


namespace total_male_students_combined_l1097_109783

/-- The number of first-year students is 695, of which 329 are female students. 
If the number of male second-year students is 254, prove that the number of male students in the first-year and second-year combined is 620. -/
theorem total_male_students_combined (first_year_students : ℕ) (female_first_year_students : ℕ) (male_second_year_students : ℕ) :
  first_year_students = 695 →
  female_first_year_students = 329 →
  male_second_year_students = 254 →
  (first_year_students - female_first_year_students + male_second_year_students) = 620 := by
  sorry

end total_male_students_combined_l1097_109783


namespace elena_pens_l1097_109713

theorem elena_pens (X Y : ℝ) 
  (h1 : X + Y = 12) 
  (h2 : 4 * X + 2.80 * Y = 40) :
  X = 5 :=
by
  sorry

end elena_pens_l1097_109713


namespace ilya_incorrect_l1097_109749

theorem ilya_incorrect (s t : ℝ) : ¬ (s + t = s * t ∧ s * t = s / t) :=
by
  sorry

end ilya_incorrect_l1097_109749


namespace sum_of_k_l1097_109743

theorem sum_of_k (k : ℕ) :
  ((∃ x, x^2 - 4 * x + 3 = 0 ∧ x^2 - 7 * x + k = 0) →
  (k = 6 ∨ k = 12)) →
  (6 + 12 = 18) :=
by sorry

end sum_of_k_l1097_109743


namespace xiaoyangs_scores_l1097_109700

theorem xiaoyangs_scores (average : ℕ) (diff : ℕ) (h_average : average = 96) (h_diff : diff = 8) :
  ∃ chinese_score math_score : ℕ, chinese_score = 92 ∧ math_score = 100 :=
by
  sorry

end xiaoyangs_scores_l1097_109700


namespace slices_left_for_lunch_tomorrow_l1097_109724

def pizza_slices : ℕ := 12
def lunch_slices : ℕ := pizza_slices / 2
def remaining_after_lunch : ℕ := pizza_slices - lunch_slices
def dinner_slices : ℕ := remaining_after_lunch * 1/3
def slices_left : ℕ := remaining_after_lunch - dinner_slices

theorem slices_left_for_lunch_tomorrow : slices_left = 4 :=
by
  sorry

end slices_left_for_lunch_tomorrow_l1097_109724


namespace julia_tuesday_kids_l1097_109727

-- Definitions based on conditions
def kids_on_monday : ℕ := 11
def tuesday_more_than_monday : ℕ := 1

-- The main statement to be proved
theorem julia_tuesday_kids : (kids_on_monday + tuesday_more_than_monday) = 12 := by
  sorry

end julia_tuesday_kids_l1097_109727


namespace min_value_of_quadratic_l1097_109799

theorem min_value_of_quadratic :
  ∃ (x y : ℝ), 2 * x^2 + 4 * x * y + 5 * y^2 - 4 * x - 6 * y + 1 = -3 :=
sorry

end min_value_of_quadratic_l1097_109799


namespace planting_area_correct_l1097_109755

def garden_area : ℕ := 18 * 14
def pond_area : ℕ := 4 * 2
def flower_bed_area : ℕ := (1 / 2) * 3 * 2
def planting_area : ℕ := garden_area - pond_area - flower_bed_area

theorem planting_area_correct : planting_area = 241 := by
  -- proof would go here
  sorry

end planting_area_correct_l1097_109755


namespace power_greater_than_any_l1097_109726

theorem power_greater_than_any {p M : ℝ} (hp : p > 0) (hM : M > 0) : ∃ n : ℕ, (1 + p)^n > M :=
by
  sorry

end power_greater_than_any_l1097_109726


namespace length_of_living_room_l1097_109762

theorem length_of_living_room (width area : ℝ) (h_width : width = 14) (h_area : area = 215.6) :
  ∃ length : ℝ, length = 15.4 ∧ area = length * width :=
by
  sorry

end length_of_living_room_l1097_109762


namespace area_of_square_l1097_109737

theorem area_of_square (r s l b : ℝ) (h1 : l = (2/5) * r)
                               (h2 : r = s)
                               (h3 : b = 10)
                               (h4 : l * b = 220) :
  s^2 = 3025 :=
by
  -- proof goes here
  sorry

end area_of_square_l1097_109737


namespace greatest_common_divisor_of_three_common_divisors_l1097_109745

theorem greatest_common_divisor_of_three_common_divisors (m : ℕ) :
  (∀ d, d ∣ 126 ∧ d ∣ m → d = 1 ∨ d = 3 ∨ d = 9) →
  gcd 126 m = 9 := 
sorry

end greatest_common_divisor_of_three_common_divisors_l1097_109745


namespace vec_op_l1097_109709

def a : ℝ × ℝ := (2, 1)
def b : ℝ × ℝ := (2, -2)
def two_a : ℝ × ℝ := (2 * 2, 2 * 1)
def result : ℝ × ℝ := (two_a.1 - b.1, two_a.2 - b.2)

theorem vec_op : (2 * a.1 - b.1, 2 * a.2 - b.2) = (2, 4) := by
  sorry

end vec_op_l1097_109709


namespace orchestra_members_l1097_109739

theorem orchestra_members :
  ∃ (n : ℕ), 
    150 < n ∧ n < 250 ∧ 
    n % 4 = 2 ∧ 
    n % 5 = 3 ∧ 
    n % 7 = 4 :=
by
  use 158
  repeat {split};
  sorry

end orchestra_members_l1097_109739


namespace impossible_score_53_l1097_109790

def quizScoring (total_questions correct_answers incorrect_answers unanswered_questions score: ℤ) : Prop :=
  total_questions = 15 ∧
  correct_answers + incorrect_answers + unanswered_questions = 15 ∧
  score = 4 * correct_answers - incorrect_answers ∧
  unanswered_questions ≥ 0 ∧ correct_answers ≥ 0 ∧ incorrect_answers ≥ 0

theorem impossible_score_53 :
  ¬ ∃ (correct_answers incorrect_answers unanswered_questions : ℤ), quizScoring 15 correct_answers incorrect_answers unanswered_questions 53 := 
sorry

end impossible_score_53_l1097_109790


namespace basketball_game_l1097_109793

theorem basketball_game (a r b d : ℕ) (r_gt_1 : r > 1) (d_gt_0 : d > 0)
  (H1 : a = b)
  (H2 : a * (1 + r) * (1 + r^2) = 4 * b + 6 * d + 2)
  (H3 : a * (1 + r) * (1 + r^2) ≤ 100)
  (H4 : 4 * b + 6 * d ≤ 98) :
  (a + a * r) + (b + (b + d)) = 43 := 
sorry

end basketball_game_l1097_109793


namespace min_value_expression_l1097_109704

theorem min_value_expression (x y : ℝ) : ∃ (a b : ℝ), x = a ∧ y = b ∧ (x^2 + y^2 - 8*x - 6*y + 30 = 5) :=
by
  sorry

end min_value_expression_l1097_109704


namespace solution_exists_l1097_109786

variable (x y : ℝ)

noncomputable def condition (x y : ℝ) : Prop :=
  (3 + 5 * x = -4 + 6 * y) ∧ (2 + (-6) * x = 6 + 8 * y)

theorem solution_exists : ∃ (x y : ℝ), condition x y ∧ x = -20 / 19 ∧ y = 11 / 38 := 
  by
  sorry

end solution_exists_l1097_109786


namespace students_in_school_B_l1097_109775

theorem students_in_school_B 
    (A B C : ℕ) 
    (h1 : A + C = 210) 
    (h2 : A = 4 * B) 
    (h3 : C = 3 * B) : 
    B = 30 := 
by 
    sorry

end students_in_school_B_l1097_109775


namespace elise_initial_dog_food_l1097_109757

variable (initial_dog_food : ℤ)
variable (bought_first_bag : ℤ := 15)
variable (bought_second_bag : ℤ := 10)
variable (final_dog_food : ℤ := 40)

theorem elise_initial_dog_food :
  initial_dog_food + bought_first_bag + bought_second_bag = final_dog_food →
  initial_dog_food = 15 :=
by
  sorry

end elise_initial_dog_food_l1097_109757


namespace transform_uniform_random_l1097_109744

theorem transform_uniform_random (a_1 : ℝ) (h : 0 ≤ a_1 ∧ a_1 ≤ 1) : -2 ≤ a_1 * 8 - 2 ∧ a_1 * 8 - 2 ≤ 6 :=
by sorry

end transform_uniform_random_l1097_109744


namespace inverse_of_p_l1097_109725

variables {p q r : Prop}

theorem inverse_of_p (m n : Prop) (hp : p = (m → n)) (hq : q = (¬m → ¬n)) (hr : r = (n → m)) : r = p ∧ r = (n → m) :=
by
  sorry

end inverse_of_p_l1097_109725


namespace b_3_value_S_m_formula_l1097_109768

-- Definition of the sequences a_n and b_n
def a_n (n : ℕ) : ℕ := if n = 0 then 0 else 3 ^ n
def b_m (m : ℕ) : ℕ := a_n (3 * m)

-- Given b_m = 3^(2m) for m in ℕ*
lemma b_m_formula (m : ℕ) (h : m > 0) : b_m m = 3 ^ (2 * m) :=
by sorry -- (This proof step will later ensure that b_m m is defined as required)

-- Prove b_3 = 729
theorem b_3_value : b_m 3 = 729 :=
by sorry

-- Sum of the first m terms of the sequence b_n
def S_m (m : ℕ) : ℕ := (Finset.range m).sum (λ i => if i = 0 then 0 else b_m (i + 1))

-- Prove S_m = (3/8)(9^m - 1)
theorem S_m_formula (m : ℕ) : S_m m = (3 / 8) * (9 ^ m - 1) :=
by sorry

end b_3_value_S_m_formula_l1097_109768


namespace train_passing_time_l1097_109772

noncomputable def speed_in_m_per_s : ℝ := (60 * 1000) / 3600

variable (L : ℝ) (S : ℝ)
variable (train_length : L = 500)
variable (train_speed : S = speed_in_m_per_s)

theorem train_passing_time : L / S = 30 := by
  sorry

end train_passing_time_l1097_109772


namespace probability_at_least_one_l1097_109741

theorem probability_at_least_one (p1 p2 : ℝ) (hp1 : 0 ≤ p1) (hp2 : 0 ≤ p2) (hp1p2 : p1 ≤ 1) (hp2p2 : p2 ≤ 1)
  (h0 : 0 ≤ 1 - p1) (h1 : 0 ≤ 1 - p2) (h2 : 1 - (1 - p1) ≥ 0) (h3 : 1 - (1 - p2) ≥ 0) :
  1 - (1 - p1) * (1 - p2) = 1 - (1 - p1) * (1 - p2) := by
  sorry

end probability_at_least_one_l1097_109741


namespace speaker_is_tweedledee_l1097_109714

-- Definitions
variable (Speaks : Prop) (is_tweedledum : Prop) (has_black_card : Prop)

-- Condition: If the speaker is Tweedledum, then the card in the speaker's pocket is not a black suit.
axiom A1 : is_tweedledum → ¬ has_black_card

-- Goal: Prove that the speaker is Tweedledee.
theorem speaker_is_tweedledee (h1 : Speaks) : ¬ is_tweedledum :=
by
  sorry

end speaker_is_tweedledee_l1097_109714


namespace range_of_a_l1097_109785

open Set

variable {a : ℝ} 

def M (a : ℝ) : Set ℝ := {x : ℝ | -4 * x + 4 * a < 0 }

theorem range_of_a (hM : 2 ∉ M a) : a ≥ 2 :=
by
  sorry

end range_of_a_l1097_109785


namespace find_quadratic_function_l1097_109736

def quad_function (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

theorem find_quadratic_function : ∃ (a b c : ℝ), 
  (∀ x : ℝ, quad_function a b c x = 2 * x^2 + 4 * x - 1) ∧ 
  (quad_function a b c (-1) = -3) ∧ 
  (quad_function a b c 1 = 5) :=
sorry

end find_quadratic_function_l1097_109736


namespace unique_rational_solution_l1097_109740

theorem unique_rational_solution (x y z : ℚ) (h : x^3 + 3*y^3 + 9*z^3 - 9*x*y*z = 0) : x = 0 ∧ y = 0 ∧ z = 0 := 
by {
  sorry
}

end unique_rational_solution_l1097_109740


namespace matrix_determinant_l1097_109777

variable {a b c d : ℝ}
variable (h : a * d - b * c = 4)

theorem matrix_determinant :
  (a * (7 * c + 3 * d) - c * (7 * a + 3 * b)) = 12 := by
  sorry

end matrix_determinant_l1097_109777


namespace A_minus_B_l1097_109716

theorem A_minus_B (A B : ℚ) (n : ℕ) :
  (A : ℚ) = 1 / 6 →
  (B : ℚ) = -1 / 12 →
  A - B = 1 / 4 :=
by
  intro hA hB
  rw [hA, hB]
  norm_num

end A_minus_B_l1097_109716


namespace find_a11_times_a55_l1097_109750

noncomputable def a_ij (i j : ℕ) : ℝ := 
  if i = 4 ∧ j = 1 then -2 else
  if i = 4 ∧ j = 3 then 10 else
  if i = 2 ∧ j = 4 then 4 else sorry

theorem find_a11_times_a55 
  (arithmetic_first_row : ∀ j, a_ij 1 (j + 1) = a_ij 1 1 + (j * 6))
  (geometric_columns : ∀ i j, a_ij (i + 1) j = a_ij 1 j * (2 ^ i) ∨ a_ij (i + 1) j = a_ij 1 j * ((-2) ^ i))
  (a24_eq_4 : a_ij 2 4 = 4)
  (a41_eq_neg2 : a_ij 4 1 = -2)
  (a43_eq_10 : a_ij 4 3 = 10) :
  a_ij 1 1 * a_ij 5 5 = -11 :=
by sorry

end find_a11_times_a55_l1097_109750


namespace sequence_solution_l1097_109769

theorem sequence_solution (a : ℕ → ℝ) (n : ℕ) (h1 : a 1 = 2) (h_rec : ∀ n > 0, a (n + 1) = a n ^ 2) : 
  a n = 2 ^ 2 ^ (n - 1) :=
by
  sorry

end sequence_solution_l1097_109769


namespace f_decreasing_l1097_109759

open Real

noncomputable def f (x : ℝ) : ℝ := 1 / x^2 + 3

theorem f_decreasing (x1 x2 : ℝ) (h1 : 0 < x1) (h2 : 0 < x2) (h : x1 < x2) : f x1 > f x2 := 
by
  sorry

end f_decreasing_l1097_109759


namespace smallest_integer_ending_in_6_divisible_by_13_l1097_109778

theorem smallest_integer_ending_in_6_divisible_by_13 (n : ℤ) (h1 : ∃ n : ℤ, 10 * n + 6 = x) (h2 : x % 13 = 0) : x = 26 :=
  sorry

end smallest_integer_ending_in_6_divisible_by_13_l1097_109778


namespace Option_C_correct_l1097_109771

theorem Option_C_correct (x y : ℝ) : 3 * x * y^2 - 4 * x * y^2 = - x * y^2 :=
by
  sorry

end Option_C_correct_l1097_109771


namespace find_a_and_b_l1097_109722

theorem find_a_and_b (a b : ℝ) :
  {-1, 3} = {x : ℝ | x^2 + a * x + b = 0} ↔ a = -2 ∧ b = -3 :=
by 
  sorry

end find_a_and_b_l1097_109722


namespace find_last_two_digits_l1097_109706

noncomputable def tenth_digit (d1 d2 d3 d4 d5 d6 d7 d8 : ℕ) : ℕ :=
d7 + d8

noncomputable def ninth_digit (d1 d2 d3 d4 d5 d6 d7 : ℕ) : ℕ :=
d6 + d7

theorem find_last_two_digits :
  ∃ d9 d10 : ℕ, d9 = ninth_digit 1 1 2 3 5 8 13 ∧ d10 = tenth_digit 1 1 2 3 5 8 13 21 :=
by
  sorry

end find_last_two_digits_l1097_109706


namespace necessary_and_sufficient_condition_l1097_109792

theorem necessary_and_sufficient_condition :
  ∀ a b : ℝ, (a + b > 0) ↔ ((a ^ 3) + (b ^ 3) > 0) :=
by
  sorry

end necessary_and_sufficient_condition_l1097_109792


namespace unit_vector_same_direction_l1097_109718

-- Define the coordinates of points A and B as given in the conditions
def A : ℝ × ℝ := (1, 3)
def B : ℝ × ℝ := (4, -1)

-- Define the vector AB
def vectorAB : ℝ × ℝ := (B.1 - A.1, B.2 - A.2)

-- Define the magnitude of vector AB
noncomputable def magnitudeAB : ℝ := Real.sqrt (vectorAB.1^2 + vectorAB.2^2)

-- Define the unit vector in the direction of AB
noncomputable def unitVectorAB : ℝ × ℝ := (vectorAB.1 / magnitudeAB, vectorAB.2 / magnitudeAB)

-- The theorem we want to prove
theorem unit_vector_same_direction :
  unitVectorAB = (3 / 5, -4 / 5) :=
sorry

end unit_vector_same_direction_l1097_109718


namespace wang_hao_not_last_l1097_109703

theorem wang_hao_not_last (total_players : ℕ) (players_to_choose : ℕ) 
  (wang_hao : ℕ) (ways_to_choose_if_not_last : ℕ) : 
  total_players = 6 ∧ players_to_choose = 3 → 
  ways_to_choose_if_not_last = 100 := 
by
  sorry

end wang_hao_not_last_l1097_109703


namespace find_original_price_l1097_109781

theorem find_original_price (x y : ℝ) 
  (h1 : 60 * x + 75 * y = 2700)
  (h2 : 60 * 0.85 * x + 75 * 0.90 * y = 2370) : 
  x = 20 ∧ y = 20 :=
sorry

end find_original_price_l1097_109781


namespace total_weight_of_oranges_l1097_109795

theorem total_weight_of_oranges :
  let capacity1 := 80
  let capacity2 := 50
  let capacity3 := 60
  let filled1 := 3 / 4
  let filled2 := 3 / 5
  let filled3 := 2 / 3
  let weight_per_orange1 := 0.25
  let weight_per_orange2 := 0.30
  let weight_per_orange3 := 0.40
  let num_oranges1 := capacity1 * filled1
  let num_oranges2 := capacity2 * filled2
  let num_oranges3 := capacity3 * filled3
  let total_weight1 := num_oranges1 * weight_per_orange1
  let total_weight2 := num_oranges2 * weight_per_orange2
  let total_weight3 := num_oranges3 * weight_per_orange3
  total_weight1 + total_weight2 + total_weight3 = 40 := by
  sorry

end total_weight_of_oranges_l1097_109795


namespace min_surface_area_base_edge_length_l1097_109789

noncomputable def min_base_edge_length (V : ℝ) : ℝ :=
  2 * (V / (2 * Real.pi))^(1/3)

theorem min_surface_area_base_edge_length (V : ℝ) : 
  min_base_edge_length V = (4 * V)^(1/3) :=
by
  sorry

end min_surface_area_base_edge_length_l1097_109789


namespace find_two_digit_divisors_l1097_109738

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100

def has_remainder (a b r : ℕ) : Prop := a = b * (a / b) + r

theorem find_two_digit_divisors (n : ℕ) (h1 : is_two_digit n) (h2 : has_remainder 723 n 30) :
  n = 33 ∨ n = 63 ∨ n = 77 ∨ n = 99 :=
sorry

end find_two_digit_divisors_l1097_109738


namespace number_of_fish_bought_each_year_l1097_109720

-- Define the conditions
def initial_fish : ℕ := 2
def net_gain_each_year (x : ℕ) : ℕ := x - 1
def years : ℕ := 5
def final_fish : ℕ := 7

-- Define the problem statement as a Lean theorem
theorem number_of_fish_bought_each_year (x : ℕ) : 
  initial_fish + years * net_gain_each_year x = final_fish → x = 2 := 
sorry

end number_of_fish_bought_each_year_l1097_109720


namespace angle_in_second_quadrant_l1097_109766

theorem angle_in_second_quadrant (x : ℝ) (hx1 : Real.tan x < 0) (hx2 : Real.sin x - Real.cos x > 0) : 
  (∃ k : ℤ, x = 2 * k * Real.pi + Real.pi / 2 ∨ x = 2 * k * Real.pi + 3 * Real.pi / 2) :=
sorry

end angle_in_second_quadrant_l1097_109766


namespace general_term_formula_l1097_109787

theorem general_term_formula (n : ℕ) (a : ℕ → ℚ) :
  (∀ n, a n = (-1)^n * (n^2)/(2 * n - 1)) :=
sorry

end general_term_formula_l1097_109787


namespace ali_total_money_l1097_109760

-- Definitions based on conditions
def bills_of_5_dollars : ℕ := 7
def bills_of_10_dollars : ℕ := 1
def value_of_5_dollar_bill : ℕ := 5
def value_of_10_dollar_bill : ℕ := 10

-- Prove that Ali's total amount of money is $45
theorem ali_total_money : (bills_of_5_dollars * value_of_5_dollar_bill) + (bills_of_10_dollars * value_of_10_dollar_bill) = 45 := 
by
  sorry

end ali_total_money_l1097_109760


namespace total_grapes_l1097_109705

theorem total_grapes (r a n : ℕ) (h1 : r = 25) (h2 : a = r + 2) (h3 : n = a + 4) : r + a + n = 83 := by
  sorry

end total_grapes_l1097_109705


namespace solve_7_at_8_l1097_109712

theorem solve_7_at_8 : (7 * 8) / (7 + 8 + 3) = 28 / 9 := by
  sorry

end solve_7_at_8_l1097_109712


namespace pauline_total_spending_l1097_109717

theorem pauline_total_spending
  (total_before_tax : ℝ)
  (sales_tax_rate : ℝ)
  (h₁ : total_before_tax = 150)
  (h₂ : sales_tax_rate = 0.08) :
  total_before_tax + total_before_tax * sales_tax_rate = 162 :=
by {
  -- Proof here
  sorry
}

end pauline_total_spending_l1097_109717


namespace problem_l1097_109764

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  (1 / 2) * a * x^2 - (x - 1) * Real.exp x

theorem problem (a : ℝ) :
  (∀ x1 x2 x3 : ℝ, 0 ≤ x1 ∧ x1 ≤ 1 ∧ 0 ≤ x2 ∧ x2 ≤ 1 ∧ 0 ≤ x3 ∧ x3 ≤ 1 →
                  f a x1 + f a x2 ≥ f a x3) →
  1 ≤ a ∧ a ≤ 4 :=
sorry

end problem_l1097_109764


namespace smallest_range_l1097_109742

-- Define the conditions
def estate (A B C : ℝ) : Prop :=
  A = 20000 ∧
  abs (A - B) > 0.3 * A ∧
  abs (A - C) > 0.3 * A ∧
  abs (B - C) > 0.3 * A

-- Define the statement to prove
theorem smallest_range (A B C : ℝ) (h : estate A B C) : 
  ∃ r : ℝ, r = 12000 :=
sorry

end smallest_range_l1097_109742


namespace num_of_ordered_pairs_l1097_109728

theorem num_of_ordered_pairs (a b : ℕ) (h1 : 0 < a) (h2 : 0 < b) (h3 : b > a)
(h4 : (a-2)*(b-2) = (ab / 2)) : (a, b) = (5, 12) ∨ (a, b) = (6, 8) :=
by
  sorry

end num_of_ordered_pairs_l1097_109728


namespace flowers_per_day_l1097_109774

-- Definitions for conditions
def total_flowers := 360
def days := 6

-- Proof that the number of flowers Miriam can take care of in one day is 60
theorem flowers_per_day : total_flowers / days = 60 := by
  sorry

end flowers_per_day_l1097_109774


namespace total_capacity_of_two_tanks_l1097_109711

-- Conditions
def tank_A_initial_fullness : ℚ := 3 / 4
def tank_A_final_fullness : ℚ := 7 / 8
def tank_A_added_volume : ℚ := 5

def tank_B_initial_fullness : ℚ := 2 / 3
def tank_B_final_fullness : ℚ := 5 / 6
def tank_B_added_volume : ℚ := 3

-- Proof statement
theorem total_capacity_of_two_tanks :
  let tank_A_total_capacity := tank_A_added_volume / (tank_A_final_fullness - tank_A_initial_fullness)
  let tank_B_total_capacity := tank_B_added_volume / (tank_B_final_fullness - tank_B_initial_fullness)
  tank_A_total_capacity + tank_B_total_capacity = 58 := 
sorry

end total_capacity_of_two_tanks_l1097_109711


namespace expand_binomial_square_l1097_109782

variables (x : ℝ)

theorem expand_binomial_square (x : ℝ) : (2 - x) ^ 2 = 4 - 4 * x + x ^ 2 := 
sorry

end expand_binomial_square_l1097_109782


namespace problem_l1097_109708

noncomputable def f (x : ℝ) : ℝ :=
  2 * Real.sin (3 * x - Real.pi / 3)

theorem problem 
  (x₁ x₂ : ℝ)
  (hx₁x₂ : |f x₁ - f x₂| = 4)
  (x : ℝ)
  (hx : 0 ≤ x ∧ x ≤ Real.pi / 6)
  (m : ℝ) : m ≥ 1 / 3 :=
sorry

end problem_l1097_109708


namespace abs_sub_eq_three_l1097_109791

theorem abs_sub_eq_three {m n : ℝ} (h1 : m * n = 4) (h2 : m + n = 5) : |m - n| = 3 := 
sorry

end abs_sub_eq_three_l1097_109791


namespace max_sum_first_n_terms_is_S_5_l1097_109730

open Nat

-- Define the arithmetic sequence and the conditions.
variable {a : ℕ → ℝ} -- The arithmetic sequence {a_n}
variable {d : ℝ} -- The common difference of the arithmetic sequence
variable {S : ℕ → ℝ} -- The sum of the first n terms of the sequence a

-- Hypotheses corresponding to the conditions in the problem
lemma a_5_positive : a 5 > 0 := sorry
lemma a_4_plus_a_7_negative : a 4 + a 7 < 0 := sorry

-- Statement to prove that the maximum value of the sum of the first n terms is S_5 given the conditions
theorem max_sum_first_n_terms_is_S_5 :
  (∀ (n : ℕ), S n ≤ S 5) :=
sorry

end max_sum_first_n_terms_is_S_5_l1097_109730


namespace security_deposit_amount_correct_l1097_109798

noncomputable def daily_rate : ℝ := 125.00
noncomputable def pet_fee : ℝ := 100.00
noncomputable def service_cleaning_fee_rate : ℝ := 0.20
noncomputable def security_deposit_rate : ℝ := 0.50
noncomputable def weeks : ℝ := 2
noncomputable def days_per_week : ℝ := 7

noncomputable def number_of_days : ℝ := weeks * days_per_week
noncomputable def total_rental_fee : ℝ := number_of_days * daily_rate
noncomputable def total_rental_fee_with_pet : ℝ := total_rental_fee + pet_fee
noncomputable def service_cleaning_fee : ℝ := service_cleaning_fee_rate * total_rental_fee_with_pet
noncomputable def total_cost : ℝ := total_rental_fee_with_pet + service_cleaning_fee

theorem security_deposit_amount_correct : 
    security_deposit_rate * total_cost = 1110.00 := 
by 
  sorry

end security_deposit_amount_correct_l1097_109798


namespace logarithmic_inequality_l1097_109733

noncomputable def a : ℝ := Real.log 2 / Real.log 3
noncomputable def b : ℝ := Real.log 3 / Real.log 2
noncomputable def c : ℝ := Real.log (1 / 3) / Real.log 4

theorem logarithmic_inequality :
  Real.log a < (1 / 2)^b := by
  sorry

end logarithmic_inequality_l1097_109733


namespace find_k_l1097_109773

def S (n : ℕ) : ℤ := n^2 - 9 * n

def a (n : ℕ) : ℤ := 
  if n = 1 then S 1
  else S n - S (n - 1)

theorem find_k (k : ℕ) (h1 : 5 < a k) (h2 : a k < 8) : k = 8 := by
  sorry

end find_k_l1097_109773


namespace probability_red_bean_l1097_109753

section ProbabilityRedBean

-- Initially, there are 5 red beans and 9 black beans in a bag.
def initial_red_beans : ℕ := 5
def initial_black_beans : ℕ := 9
def initial_total_beans : ℕ := initial_red_beans + initial_black_beans

-- Then, 3 red beans and 3 black beans are added to the bag.
def added_red_beans : ℕ := 3
def added_black_beans : ℕ := 3
def final_red_beans : ℕ := initial_red_beans + added_red_beans
def final_black_beans : ℕ := initial_black_beans + added_black_beans
def final_total_beans : ℕ := final_red_beans + final_black_beans

-- The probability of drawing a red bean should be 2/5
theorem probability_red_bean :
  (final_red_beans : ℚ) / final_total_beans = 2 / 5 := by
  sorry

end ProbabilityRedBean

end probability_red_bean_l1097_109753


namespace tom_seashells_left_l1097_109748

def initial_seashells : ℕ := 5
def given_away_seashells : ℕ := 2

theorem tom_seashells_left : (initial_seashells - given_away_seashells) = 3 :=
by
  sorry

end tom_seashells_left_l1097_109748


namespace black_ants_employed_l1097_109732

theorem black_ants_employed (total_ants : ℕ) (red_ants : ℕ) 
  (h1 : total_ants = 900) (h2 : red_ants = 413) :
    total_ants - red_ants = 487 :=
by
  -- The proof is given below.
  sorry

end black_ants_employed_l1097_109732


namespace bob_total_profit_l1097_109701

def initial_cost (num_dogs : ℕ) (cost_per_dog : ℕ) : ℕ := num_dogs * cost_per_dog

def revenue (num_puppies : ℕ) (price_per_puppy : ℕ) : ℕ := num_puppies * price_per_puppy

def total_profit (initial_cost : ℕ) (revenue : ℕ) : ℕ := revenue - initial_cost

theorem bob_total_profit (c1 : initial_cost 2 250 = 500)
                        (c2 : revenue 6 350 = 2100)
                        (c3 : total_profit 500 2100 = 1600) :
  total_profit (initial_cost 2 250) (revenue 6 350) = 1600 := by
  sorry

end bob_total_profit_l1097_109701


namespace symmetric_point_y_axis_l1097_109767

def M : ℝ × ℝ := (-5, 2)
def symmetric_point (p : ℝ × ℝ) : ℝ × ℝ := (-p.1, p.2)
theorem symmetric_point_y_axis :
  symmetric_point M = (5, 2) :=
by
  sorry

end symmetric_point_y_axis_l1097_109767
