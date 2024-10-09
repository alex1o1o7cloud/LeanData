import Mathlib

namespace halfway_between_3_4_and_5_7_l1151_115164

-- Define the two fractions
def frac1 := 3/4
def frac2 := 5/7

-- Define the average function for two fractions
def halfway_fract (a b : ℚ) : ℚ := (a + b) / 2

-- Prove that the halfway fraction between 3/4 and 5/7 is 41/56
theorem halfway_between_3_4_and_5_7 : 
  halfway_fract frac1 frac2 = 41/56 := 
by 
  sorry

end halfway_between_3_4_and_5_7_l1151_115164


namespace incorrect_expression_l1151_115155

variable (x y : ℝ)

theorem incorrect_expression (h : x > y) (hnx : x < 0) (hny : y < 0) : x^2 - 3 ≤ y^2 - 3 := by
sorry

end incorrect_expression_l1151_115155


namespace sqrt_twentyfive_eq_five_l1151_115148

theorem sqrt_twentyfive_eq_five : Real.sqrt 25 = 5 := by
  sorry

end sqrt_twentyfive_eq_five_l1151_115148


namespace rational_solution_cos_eq_l1151_115176

theorem rational_solution_cos_eq {q : ℚ} (h0 : 0 < q) (h1 : q < 1) (heq : Real.cos (3 * Real.pi * q) + 2 * Real.cos (2 * Real.pi * q) = 0) : 
  q = 2 / 3 := 
sorry

end rational_solution_cos_eq_l1151_115176


namespace pencil_notebook_cost_l1151_115146

variable {p n : ℝ}

theorem pencil_notebook_cost (hp1 : 9 * p + 11 * n = 6.05) (hp2 : 6 * p + 4 * n = 2.68) :
  18 * p + 13 * n = 8.45 :=
sorry

end pencil_notebook_cost_l1151_115146


namespace original_average_age_l1151_115141

-- Definitions based on conditions
def original_strength : ℕ := 12
def new_student_count : ℕ := 12
def new_student_average_age : ℕ := 32
def age_decrease : ℕ := 4
def total_student_count : ℕ := original_strength + new_student_count
def combined_total_age (A : ℕ) : ℕ := original_strength * A + new_student_count * new_student_average_age
def new_average_age (A : ℕ) : ℕ := A - age_decrease

-- Statement of the problem
theorem original_average_age (A : ℕ) (h : combined_total_age A / total_student_count = new_average_age A) : A = 40 := 
by 
  sorry

end original_average_age_l1151_115141


namespace max_axbycz_value_l1151_115189

theorem max_axbycz_value (a b c : ℝ) (x y z : ℝ) 
  (h_triangle: a + b > c ∧ b + c > a ∧ c + a > b)
  (h_positive: 0 < x ∧ 0 < y ∧ 0 < z)
  (h_sum : x + y + z = 1) : 
  a * x * y + b * y * z + c * z * x ≤ (a * b * c) / (2 * a * b + 2 * b * c + 2 * c * a - a^2 - b^2 - c^2) :=
  sorry

end max_axbycz_value_l1151_115189


namespace linear_regression_passes_through_centroid_l1151_115168

noncomputable def linear_regression (a b : ℝ) (x : ℝ) : ℝ := a + b * x

theorem linear_regression_passes_through_centroid 
  (a b : ℝ) (x_bar y_bar : ℝ) 
  (h_centroid : ∀ (x y : ℝ), (x = x_bar ∧ y = y_bar) → y = linear_regression a b x) :
  linear_regression a b x_bar = y_bar :=
by
  -- proof omitted
  sorry

end linear_regression_passes_through_centroid_l1151_115168


namespace find_number_l1151_115100

theorem find_number (x : ℝ) (n : ℝ) (h1 : x = 12) (h2 : (27 / n) * x - 18 = 3 * x + 27) : n = 4 :=
sorry

end find_number_l1151_115100


namespace sarah_homework_problems_l1151_115162

theorem sarah_homework_problems (math_pages reading_pages problems_per_page : ℕ) 
  (h1 : math_pages = 4) 
  (h2 : reading_pages = 6) 
  (h3 : problems_per_page = 4) : 
  (math_pages + reading_pages) * problems_per_page = 40 :=
by 
  sorry

end sarah_homework_problems_l1151_115162


namespace total_students_l1151_115142

theorem total_students (ratio_boys_girls : ℕ) (girls : ℕ) (boys : ℕ) (total_students : ℕ)
  (h1 : ratio_boys_girls = 2)     -- The simplified ratio of boys to girls
  (h2 : girls = 200)              -- There are 200 girls
  (h3 : boys = ratio_boys_girls * girls) -- Number of boys is ratio * number of girls
  (h4 : total_students = boys + girls)   -- Total number of students is the sum of boys and girls
  : total_students = 600 :=             -- Prove that the total number of students is 600
sorry

end total_students_l1151_115142


namespace rectangle_length_reduction_30_percent_l1151_115183

variables (L W : ℝ) (x : ℝ)

theorem rectangle_length_reduction_30_percent
  (h : 1 = (1 - x / 100) * 1.4285714285714287) :
  x = 30 :=
sorry

end rectangle_length_reduction_30_percent_l1151_115183


namespace rose_flyers_l1151_115136

theorem rose_flyers (total_flyers made: ℕ) (flyers_jack: ℕ) (flyers_left: ℕ) 
(h1 : total_flyers = 1236)
(h2 : flyers_jack = 120)
(h3 : flyers_left = 796)
: total_flyers - flyers_jack - flyers_left = 320 :=
by
  sorry

end rose_flyers_l1151_115136


namespace flour_cups_l1151_115114

theorem flour_cups (f : ℚ) (h : f = 4 + 3/4) : (1/3) * f = 1 + 7/12 := by
  sorry

end flour_cups_l1151_115114


namespace smallest_b_base_l1151_115149

theorem smallest_b_base :
  ∃ b : ℕ, b^2 ≤ 25 ∧ 25 < b^3 ∧ (∀ c : ℕ, c < b → ¬(c^2 ≤ 25 ∧ 25 < c^3)) :=
sorry

end smallest_b_base_l1151_115149


namespace equidistant_trajectory_l1151_115113

theorem equidistant_trajectory {x y z : ℝ} :
  (x + 1)^2 + (y - 1)^2 + z^2 = (x - 2)^2 + (y + 1)^2 + (z + 1)^2 →
  3 * x - 2 * y - z = 2 :=
sorry

end equidistant_trajectory_l1151_115113


namespace breakable_iff_composite_l1151_115196

-- Definitions directly from the problem conditions
def is_breakable (n : ℕ) : Prop :=
  ∃ (a b x y : ℕ), a > 0 ∧ b > 0 ∧ x > 0 ∧ y > 0 ∧ a + b = n ∧ (x / a : ℚ) + (y / b : ℚ) = 1

def is_composite (n : ℕ) : Prop :=
  ∃ (s t : ℕ), s > 1 ∧ t > 1 ∧ n = s * t

-- The proof statement
theorem breakable_iff_composite (n : ℕ) : is_breakable n ↔ is_composite n := sorry

end breakable_iff_composite_l1151_115196


namespace find_BM_length_l1151_115101

variables (A M C : Type) (MA CA BC BM : ℝ) (x d h : ℝ)

-- Conditions
def condition1 : Prop := MA + (BC - BM) = 2 * CA
def condition2 : Prop := MA = x
def condition3 : Prop := CA = d
def condition4 : Prop := BC = h

-- The proof problem statement
theorem find_BM_length (A M C : Type) (MA CA BC BM : ℝ) (x d h : ℝ)
  (h1 : condition1 MA CA BC BM)
  (h2 : condition2 MA x)
  (h3 : condition3 CA d)
  (h4 : condition4 BC h) :
  BM = 2 * d :=
sorry

end find_BM_length_l1151_115101


namespace problem_statement_l1151_115134

def nabla (a b : ℕ) : ℕ := 3 + b ^ a

theorem problem_statement : nabla (nabla 2 3) 4 = 16777219 := by
  sorry

end problem_statement_l1151_115134


namespace sum_of_abcd_l1151_115118

variable (a b c d : ℚ)

def condition (x : ℚ) : Prop :=
  x = a + 3 ∧
  x = b + 7 ∧
  x = c + 5 ∧
  x = d + 9 ∧
  x = a + b + c + d + 13

theorem sum_of_abcd (x : ℚ) (h : condition a b c d x) : a + b + c + d = -28 / 3 := 
by sorry

end sum_of_abcd_l1151_115118


namespace quadrilateral_sides_equal_l1151_115151

theorem quadrilateral_sides_equal (a b c d : ℕ) (h1 : a ∣ b + c + d) (h2 : b ∣ a + c + d) (h3 : c ∣ a + b + d) (h4 : d ∣ a + b + c) : a = b ∨ a = c ∨ a = d ∨ b = c ∨ b = d ∨ c = d :=
sorry

end quadrilateral_sides_equal_l1151_115151


namespace Stonewall_marching_band_max_members_l1151_115179

theorem Stonewall_marching_band_max_members (n : ℤ) (h1 : 30 * n % 34 = 2) (h2 : 30 * n < 1500) : 30 * n = 1260 :=
by
  sorry

end Stonewall_marching_band_max_members_l1151_115179


namespace anthony_has_more_pairs_l1151_115194

theorem anthony_has_more_pairs (scott_pairs : ℕ) (anthony_pairs : ℕ) (jim_pairs : ℕ) :
  (scott_pairs = 7) →
  (anthony_pairs = 3 * scott_pairs) →
  (jim_pairs = anthony_pairs - 2) →
  (anthony_pairs - jim_pairs = 2) :=
by
  intro h_scott h_anthony h_jim
  sorry

end anthony_has_more_pairs_l1151_115194


namespace no_b_satisfies_condition_l1151_115122

noncomputable def f (b x : ℝ) : ℝ :=
  x^2 + 3 * b * x + 5 * b

theorem no_b_satisfies_condition :
  ∀ b : ℝ, ¬ (∃ x : ℝ, ∀ y : ℝ, |f b y| ≤ 5 → y = x) :=
by
  sorry

end no_b_satisfies_condition_l1151_115122


namespace inverse_of_inverse_at_9_l1151_115154

noncomputable def f (x : ℝ) : ℝ := 4 * x + 5

noncomputable def f_inv (x : ℝ) : ℝ := (x - 5) / 4

theorem inverse_of_inverse_at_9 : f_inv (f_inv 9) = -1 :=
by
  sorry

end inverse_of_inverse_at_9_l1151_115154


namespace joe_first_lift_weight_l1151_115170

variable (x y : ℕ)

def joe_lift_conditions (x y : ℕ) : Prop :=
  x + y = 600 ∧ 2 * x = y + 300

theorem joe_first_lift_weight (x y : ℕ) (h : joe_lift_conditions x y) : x = 300 :=
by
  sorry

end joe_first_lift_weight_l1151_115170


namespace main_inequality_l1151_115165

theorem main_inequality (a b c d : ℝ) (h_a_pos : 0 < a) (h_b_pos : 0 < b) (h_c_pos : 0 < c) (h_d_pos : 0 < d) (h_sum : a + b + c + d = 1) :
  (b * c * d) / (1 - a)^2 + (c * d * a) / (1 - b)^2 + (d * a * b) / (1 - c)^2 + (a * b * c) / (1 - d)^2 ≤ 1 / 9 :=
by
  sorry

end main_inequality_l1151_115165


namespace y_days_do_work_l1151_115174

theorem y_days_do_work (d : ℝ) (h : (1 / 30) + (1 / d) = 1 / 18) : d = 45 := 
by
  sorry

end y_days_do_work_l1151_115174


namespace tan_alpha_proof_l1151_115171

theorem tan_alpha_proof 
  (α : ℝ) 
  (h1 : 0 < α) 
  (h2 : α < Real.pi / 2)
  (h3 : Real.tan (2 * α) = Real.cos α / (2 - Real.sin α)) : 
  Real.tan α = Real.sqrt 15 / 15 :=
by
  sorry

end tan_alpha_proof_l1151_115171


namespace problem1_problem2_l1151_115140

-- Problem 1: Solution set for x(7 - x) >= 12
theorem problem1 (x : ℝ) : x * (7 - x) ≥ 12 ↔ (3 ≤ x ∧ x ≤ 4) :=
by
  sorry

-- Problem 2: Solution set for x^2 > 2(x - 1)
theorem problem2 (x : ℝ) : x^2 > 2 * (x - 1) ↔ true :=
by
  sorry

end problem1_problem2_l1151_115140


namespace train_distance_difference_l1151_115180

theorem train_distance_difference (t : ℝ) (D₁ D₂ : ℝ)
(h_speed1 : D₁ = 20 * t)
(h_speed2 : D₂ = 25 * t)
(h_total_dist : D₁ + D₂ = 540) :
  D₂ - D₁ = 60 :=
by {
  -- These are the conditions as stated in step c)
  sorry
}

end train_distance_difference_l1151_115180


namespace part_I_part_II_l1151_115125

noncomputable def A : Set ℝ := {x | 2*x^2 - 5*x - 3 <= 0}
noncomputable def B (a : ℝ) : Set ℝ := {x | (x - (2*a + 1)) * (x - (a - 1)) < 0}

theorem part_I :
  (A ∪ B 0 = {x : ℝ | -1 < x ∧ x ≤ 3}) :=
by sorry

theorem part_II (a : ℝ) :
  (A ∩ B a = ∅) →
  (a ≤ -3/4 ∨ a ≥ 4) ∧ a ≠ -2 :=
by sorry


end part_I_part_II_l1151_115125


namespace softball_players_count_l1151_115192

theorem softball_players_count :
  ∀ (cricket hockey football total_players softball : ℕ),
  cricket = 15 →
  hockey = 12 →
  football = 13 →
  total_players = 55 →
  total_players = cricket + hockey + football + softball →
  softball = 15 :=
by
  intros cricket hockey football total_players softball h_cricket h_hockey h_football h_total_players h_total
  sorry

end softball_players_count_l1151_115192


namespace remainder_division_Q_l1151_115137

noncomputable def Q_rest : Polynomial ℝ := -(Polynomial.X : Polynomial ℝ) + 125

theorem remainder_division_Q (Q : Polynomial ℝ) :
  Q.eval 20 = 105 ∧ Q.eval 105 = 20 →
  ∃ R : Polynomial ℝ, Q = (Polynomial.X - 20) * (Polynomial.X - 105) * R + Q_rest :=
by sorry

end remainder_division_Q_l1151_115137


namespace cyclist_trip_time_l1151_115124

variable (a v : ℝ)
variable (h1 : a / v = 5)

theorem cyclist_trip_time
  (increase_factor : ℝ := 1.25) :
  (a / (2 * v) + a / (2 * (increase_factor * v)) = 4.5) :=
sorry

end cyclist_trip_time_l1151_115124


namespace square_area_l1151_115198

theorem square_area : ∃ (s: ℝ), (∀ x: ℝ, x^2 + 4*x + 1 = 7 → ∃ t: ℝ, t = x ∧ ∃ x2: ℝ, (x2 - x)^2 = s^2 ∧ ∀ y : ℝ, y = 7 ∧ y = x2^2 + 4*x2 + 1) ∧ s^2 = 40 :=
by
  sorry

end square_area_l1151_115198


namespace transformed_polynomial_l1151_115167

theorem transformed_polynomial (x y : ℝ) (h : y = x + 1 / x) :
  (x^4 - 2*x^3 - 3*x^2 + 2*x + 1 = 0) → (x^2 * (y^2 - y - 3) = 0) :=
by
  sorry

end transformed_polynomial_l1151_115167


namespace initial_number_of_men_l1151_115111

theorem initial_number_of_men (M : ℝ) (P : ℝ) (h1 : P = M * 20) (h2 : P = (M + 200) * 16.67) : M = 1000 :=
by
  sorry

end initial_number_of_men_l1151_115111


namespace ratio_of_scores_l1151_115172

theorem ratio_of_scores 
  (u v : ℝ) 
  (h1 : u > v) 
  (h2 : u - v = (u + v) / 2) 
  : v / u = 1 / 3 :=
sorry

end ratio_of_scores_l1151_115172


namespace total_outlets_needed_l1151_115128

-- Definitions based on conditions:
def outlets_per_room : ℕ := 6
def number_of_rooms : ℕ := 7

-- Theorem to prove the total number of outlets is 42
theorem total_outlets_needed : outlets_per_room * number_of_rooms = 42 := by
  -- Simple proof with mathematics:
  sorry

end total_outlets_needed_l1151_115128


namespace max_profit_l1151_115131

noncomputable def fixed_cost : ℝ := 2.5

noncomputable def cost (x : ℝ) : ℝ :=
  if x < 80 then (1/3) * x^2 + 10 * x
  else 51 * x + 10000 / x - 1450

noncomputable def revenue (x : ℝ) : ℝ := 0.05 * 1000 * x

noncomputable def profit (x : ℝ) : ℝ :=
  revenue x - cost x - fixed_cost * 10

theorem max_profit : ∃ x_opt : ℝ, ∀ x : ℝ, 0 < x → 
  profit x ≤ profit 100 ∧ x_opt = 100 :=
by
  sorry

end max_profit_l1151_115131


namespace total_weight_of_plastic_rings_l1151_115182

-- Conditions
def orange_ring_weight : ℝ := 0.08
def purple_ring_weight : ℝ := 0.33
def white_ring_weight : ℝ := 0.42

-- Proof Statement
theorem total_weight_of_plastic_rings :
  orange_ring_weight + purple_ring_weight + white_ring_weight = 0.83 := by
  sorry

end total_weight_of_plastic_rings_l1151_115182


namespace solve_quadratic_l1151_115186

theorem solve_quadratic (x : ℝ) :
  25 * x^2 - 10 * x - 1000 = 0 → ∃ r s, (x + r)^2 = s ∧ s = 40.04 :=
by
  intro h
  sorry

end solve_quadratic_l1151_115186


namespace monkey_farm_l1151_115175

theorem monkey_farm (x y : ℕ) 
  (h1 : y = 14 * x + 48) 
  (h2 : y = 18 * x - 64) : 
  x = 28 ∧ y = 440 := 
by 
  sorry

end monkey_farm_l1151_115175


namespace range_of_a_l1151_115150

theorem range_of_a (a : ℝ) : 
  (∃ x : ℝ, (x^2 + a*x + 4 < 0)) ↔ (a < -4 ∨ a > 4) :=
by
  sorry

end range_of_a_l1151_115150


namespace exists_permutation_ab_minus_cd_ge_two_l1151_115143

theorem exists_permutation_ab_minus_cd_ge_two (p q r s : ℝ) 
  (h1 : p + q + r + s = 9) 
  (h2 : p^2 + q^2 + r^2 + s^2 = 21) :
  ∃ (a b c d : ℝ), (a, b, c, d) = (p, q, r, s) ∨ (a, b, c, d) = (p, q, s, r) ∨ 
  (a, b, c, d) = (p, r, q, s) ∨ (a, b, c, d) = (p, r, s, q) ∨ 
  (a, b, c, d) = (p, s, q, r) ∨ (a, b, c, d) = (p, s, r, q) ∨ 
  (a, b, c, d) = (q, p, r, s) ∨ (a, b, c, d) = (q, p, s, r) ∨ 
  (a, b, c, d) = (q, r, p, s) ∨ (a, b, c, d) = (q, r, s, p) ∨ 
  (a, b, c, d) = (q, s, p, r) ∨ (a, b, c, d) = (q, s, r, p) ∨ 
  (a, b, c, d) = (r, p, q, s) ∨ (a, b, c, d) = (r, p, s, q) ∨ 
  (a, b, c, d) = (r, q, p, s) ∨ (a, b, c, d) = (r, q, s, p) ∨ 
  (a, b, c, d) = (r, s, p, q) ∨ (a, b, c, d) = (r, s, q, p) ∨ 
  (a, b, c, d) = (s, p, q, r) ∨ (a, b, c, d) = (s, p, r, q) ∨ 
  (a, b, c, d) = (s, q, p, r) ∨ (a, b, c, d) = (s, q, r, p) ∨ 
  (a, b, c, d) = (s, r, p, q) ∨ (a, b, c, d) = (s, r, q, p) ∧ ab - cd ≥ 2 :=
sorry

end exists_permutation_ab_minus_cd_ge_two_l1151_115143


namespace simplify_fraction_l1151_115158

variable (x : ℕ)

theorem simplify_fraction (h : x = 3) : (x^10 + 15 * x^5 + 125) / (x^5 + 5) = 248 + 25 / 62 := by
  sorry

end simplify_fraction_l1151_115158


namespace find_third_month_sale_l1151_115126

def sales_first_month : ℕ := 3435
def sales_second_month : ℕ := 3927
def sales_fourth_month : ℕ := 4230
def sales_fifth_month : ℕ := 3562
def sales_sixth_month : ℕ := 1991
def required_average_sale : ℕ := 3500

theorem find_third_month_sale (S3 : ℕ) :
  (sales_first_month + sales_second_month + S3 + sales_fourth_month + sales_fifth_month + sales_sixth_month) / 6 = required_average_sale →
  S3 = 3855 := by
  sorry

end find_third_month_sale_l1151_115126


namespace num_stripes_on_us_flag_l1151_115160

-- Definitions based on conditions in the problem
def num_stars : ℕ := 50

def num_circles : ℕ := (num_stars / 2) - 3

def num_squares (S : ℕ) : ℕ := 2 * S + 6

def total_shapes (num_squares : ℕ) : ℕ := num_circles + num_squares

-- The theorem stating the number of stripes
theorem num_stripes_on_us_flag (S : ℕ) (h1 : num_circles = 22) (h2 : total_shapes (num_squares S) = 54) : S = 13 := by
  sorry

end num_stripes_on_us_flag_l1151_115160


namespace negation_of_P_is_exists_Q_l1151_115130

def P (x : ℝ) : Prop := x^2 - x + 3 > 0

theorem negation_of_P_is_exists_Q :
  (¬ (∀ x : ℝ, P x)) ↔ (∃ x : ℝ, ¬ P x) :=
sorry

end negation_of_P_is_exists_Q_l1151_115130


namespace complement_of_A_union_B_in_U_l1151_115184

def U : Set ℝ := { x | -5 < x ∧ x < 5 }
def A : Set ℝ := { x | x^2 - 4*x - 5 < 0 }
def B : Set ℝ := { x | -2 < x ∧ x < 4 }

theorem complement_of_A_union_B_in_U :
  (U \ (A ∪ B)) = { x | -5 < x ∧ x ≤ -2 } := by
  sorry

end complement_of_A_union_B_in_U_l1151_115184


namespace larger_circle_radius_l1151_115185

theorem larger_circle_radius (r R : ℝ) 
  (h : (π * R^2) / (π * r^2) = 5 / 2) : 
  R = r * Real.sqrt 2.5 :=
sorry

end larger_circle_radius_l1151_115185


namespace average_rst_l1151_115112

variable (r s t : ℝ)

theorem average_rst
  (h : (4 / 3) * (r + s + t) = 12) :
  (r + s + t) / 3 = 3 :=
sorry

end average_rst_l1151_115112


namespace difference_red_white_l1151_115144

/-
Allie picked 100 wildflowers. The categories of flowers are given as below:
- 13 of the flowers were yellow and white
- 17 of the flowers were red and yellow
- 14 of the flowers were red and white
- 16 of the flowers were blue and yellow
- 9 of the flowers were blue and white
- 8 of the flowers were red, blue, and yellow
- 6 of the flowers were red, white, and blue

The goal is to define the number of flowers containing red and white, and
prove that the difference between the number of flowers containing red and 
those containing white is 3.
-/

def total_flowers : ℕ := 100
def yellow_and_white : ℕ := 13
def red_and_yellow : ℕ := 17
def red_and_white : ℕ := 14
def blue_and_yellow : ℕ := 16
def blue_and_white : ℕ := 9
def red_blue_and_yellow : ℕ := 8
def red_white_and_blue : ℕ := 6

def flowers_with_red : ℕ := red_and_yellow + red_and_white + red_blue_and_yellow + red_white_and_blue
def flowers_with_white : ℕ := yellow_and_white + red_and_white + blue_and_white + red_white_and_blue

theorem difference_red_white : flowers_with_red - flowers_with_white = 3 := by
  rw [flowers_with_red, flowers_with_white]
  sorry

end difference_red_white_l1151_115144


namespace car_rental_cost_l1151_115153

theorem car_rental_cost
  (rent_per_day : ℝ) (cost_per_mile : ℝ) (days_rented : ℕ) (miles_driven : ℝ)
  (h1 : rent_per_day = 30)
  (h2 : cost_per_mile = 0.25)
  (h3 : days_rented = 5)
  (h4 : miles_driven = 500) :
  rent_per_day * days_rented + cost_per_mile * miles_driven = 275 := 
  by
  sorry

end car_rental_cost_l1151_115153


namespace usual_time_is_20_l1151_115139

-- Define the problem
variables (T T': ℕ)

-- Conditions
axiom condition1 : T' = T + 5
axiom condition2 : T' = 5 * T / 4

-- Proof statement
theorem usual_time_is_20 : T = 20 :=
  sorry

end usual_time_is_20_l1151_115139


namespace arithmetic_sequence_sum_l1151_115156

theorem arithmetic_sequence_sum (S : ℕ → ℝ) (a : ℕ → ℝ) (n : ℕ)
  (h1 : ∀ n, S n = n * ((a 1 + a n) / 2))
  (h2 : S 9 = 27) :
  a 4 + a 6 = 6 := 
sorry

end arithmetic_sequence_sum_l1151_115156


namespace prime_product_is_2009_l1151_115188

theorem prime_product_is_2009 (a b c : ℕ) 
  (h_primeA : Prime a) 
  (h_primeB : Prime b) 
  (h_primeC : Prime c)
  (h_div1 : a ∣ (b + 8)) 
  (h_div2a : a ∣ (b^2 - 1)) 
  (h_div2c : c ∣ (b^2 - 1)) 
  (h_sum : b + c = a^2 - 1) : 
  a * b * c = 2009 := 
sorry

end prime_product_is_2009_l1151_115188


namespace bread_pieces_total_l1151_115157

def initial_slices : ℕ := 2
def pieces_per_slice (n : ℕ) : ℕ := n * 4

theorem bread_pieces_total : pieces_per_slice initial_slices = 8 :=
by
  sorry

end bread_pieces_total_l1151_115157


namespace product_odd_integers_lt_20_l1151_115178

/--
The product of all odd positive integers strictly less than 20 is a positive number ending with the digit 5.
-/
theorem product_odd_integers_lt_20 :
  let nums := [1, 3, 5, 7, 9, 11, 13, 15, 17, 19]
  let product := List.prod nums
  (product > 0) ∧ (product % 10 = 5) :=
by
  sorry

end product_odd_integers_lt_20_l1151_115178


namespace average_age_of_school_l1151_115108

theorem average_age_of_school 
  (total_students : ℕ)
  (average_age_boys : ℕ)
  (average_age_girls : ℕ)
  (number_of_girls : ℕ)
  (number_of_boys : ℕ := total_students - number_of_girls)
  (total_age_boys : ℕ := average_age_boys * number_of_boys)
  (total_age_girls : ℕ := average_age_girls * number_of_girls)
  (total_age_students : ℕ := total_age_boys + total_age_girls) :
  total_students = 640 →
  average_age_boys = 12 →
  average_age_girls = 11 →
  number_of_girls = 160 →
  (total_age_students : ℝ) / (total_students : ℝ) = 11.75 :=
by
  intros h1 h2 h3 h4
  sorry

end average_age_of_school_l1151_115108


namespace spa_polish_total_digits_l1151_115102

theorem spa_polish_total_digits (girls : ℕ) (digits_per_girl : ℕ) (total_digits : ℕ)
  (h1 : girls = 5) (h2 : digits_per_girl = 20) : total_digits = 100 :=
by
  sorry

end spa_polish_total_digits_l1151_115102


namespace parabola_focus_l1151_115159

-- Define the given conditions
def parabola_equation (x : ℝ) : ℝ := 4 * x^2 - 8 * x + 5

-- The proof statement that we need to show the focus of the given parabola
theorem parabola_focus :
  (∃ (h k : ℝ), (k = 1) ∧ (h = 1) ∧ (parabola_equation h = k) ∧ ((h, k + 1 / (4 * 4)) = (1, 17 / 16))) := 
sorry

end parabola_focus_l1151_115159


namespace difference_five_three_numbers_specific_number_condition_l1151_115193

def is_five_three_number (A : ℕ) : Prop :=
  let a := A / 1000
  let b := (A % 1000) / 100
  let c := (A % 100) / 10
  let d := A % 10
  a = 5 + c ∧ b = 3 + d

def M (A : ℕ) : ℕ :=
  let a := A / 1000
  let b := (A % 1000) / 100
  let c := (A % 100) / 10
  let d := A % 10
  a + c + 2 * (b + d)

def N (A : ℕ) : ℕ :=
  let b := (A % 1000) / 100
  b - 3

noncomputable def largest_five_three_number := 9946
noncomputable def smallest_five_three_number := 5300

theorem difference_five_three_numbers :
  largest_five_three_number - smallest_five_three_number = 4646 := by
  sorry

noncomputable def specific_five_three_number := 5401

theorem specific_number_condition {A : ℕ} (hA : is_five_three_number A) :
  (M A) % (N A) = 0 ∧ (M A) / (N A) % 5 = 0 → A = specific_five_three_number := by
  sorry

end difference_five_three_numbers_specific_number_condition_l1151_115193


namespace system_solution_l1151_115104

theorem system_solution (x y : ℝ) :
  (x + y = 4) ∧ (2 * x - y = 2) → x = 2 ∧ y = 2 := by 
sorry

end system_solution_l1151_115104


namespace correct_calculation_l1151_115127

theorem correct_calculation : (Real.sqrt 3) ^ 2 = 3 := by
  sorry

end correct_calculation_l1151_115127


namespace average_value_of_powers_l1151_115119

theorem average_value_of_powers (z : ℝ) : 
  (z^2 + 3*z^2 + 6*z^2 + 12*z^2 + 24*z^2) / 5 = 46*z^2 / 5 :=
by
  sorry

end average_value_of_powers_l1151_115119


namespace edge_length_box_l1151_115199

theorem edge_length_box (n : ℝ) (h : n = 999.9999999999998) : 
  ∃ (L : ℝ), L = 1 ∧ ((L * 100) ^ 3 / 10 ^ 3) = n := 
sorry

end edge_length_box_l1151_115199


namespace simple_interest_years_l1151_115129

theorem simple_interest_years (P : ℝ) (hP : P > 0) (R : ℝ := 2.5) (SI : ℝ := P / 5) : 
  ∃ T : ℝ, P * R * T / 100 = SI ∧ T = 8 :=
by
  sorry

end simple_interest_years_l1151_115129


namespace zhang_qiu_jian_problem_l1151_115107

-- Define the arithmetic sequence
def arithmeticSequence (a1 d : ℚ) (n : ℕ) : ℚ :=
  a1 + (n - 1) * d

-- Sum of first n terms of an arithmetic sequence
def sumArithmeticSequence (a1 d : ℚ) (n : ℕ) : ℚ :=
  n * a1 + (n * (n - 1) / 2) * d

theorem zhang_qiu_jian_problem :
  sumArithmeticSequence 5 (16 / 29) 30 = 390 := 
by 
  sorry

end zhang_qiu_jian_problem_l1151_115107


namespace smallest_d_for_inverse_l1151_115163

def g (x : ℝ) : ℝ := (x - 3) ^ 2 - 7

theorem smallest_d_for_inverse (d : ℝ) : 
  (∀ x1 x2 : ℝ, d ≤ x1 → d ≤ x2 → g x1 = g x2 → x1 = x2) → d = 3 :=
by
  sorry

end smallest_d_for_inverse_l1151_115163


namespace grocer_sales_l1151_115166

theorem grocer_sales (sale1 sale2 sale3 sale4 sale5 sale6 : ℕ)
  (h1 : sale2 = 900)
  (h2 : sale3 = 1000)
  (h3 : sale4 = 700)
  (h4 : sale5 = 800)
  (h5 : sale6 = 900)
  (h6 : (sale1 + sale2 + sale3 + sale4 + sale5 + sale6) / 6 = 850) :
  sale1 = 800 :=
by
  sorry

end grocer_sales_l1151_115166


namespace equal_intercepts_no_second_quadrant_l1151_115187

/- Given line equation (a + 1)x + y + 2 - a = 0 and a \in ℝ. -/
def line_eq (a x y : ℝ) : Prop := (a + 1) * x + y + 2 - a = 0

/- If the line l has equal intercepts on both coordinate axes, 
   then a = 0 or a = 2. -/
theorem equal_intercepts (a : ℝ) :
  (∃ x y : ℝ, line_eq a x 0 ∧ line_eq a 0 y ∧ x = y) →
  a = 0 ∨ a = 2 :=
sorry

/- If the line l does not pass through the second quadrant,
   then a ≤ -1. -/
theorem no_second_quadrant (a : ℝ) :
  (∀ x y : ℝ, x > 0 → y > 0 → ¬ line_eq a x y) →
  a ≤ -1 :=
sorry

end equal_intercepts_no_second_quadrant_l1151_115187


namespace range_of_func_l1151_115195

noncomputable def func (x : ℝ) : ℝ := 1 / (x - 1)

theorem range_of_func :
  (∀ y : ℝ, 
    (∃ x : ℝ, (x < 1 ∨ (2 ≤ x ∧ x < 5)) ∧ y = func x) ↔ 
    (y < 0 ∨ (1/4 < y ∧ y ≤ 1))) :=
by
  sorry

end range_of_func_l1151_115195


namespace oil_bill_january_l1151_115191

theorem oil_bill_january (F J : ℝ) (h1 : F / J = 3 / 2) (h2 : (F + 30) / J = 5 / 3) : J = 180 :=
by
  sorry

end oil_bill_january_l1151_115191


namespace chairlift_halfway_l1151_115152

theorem chairlift_halfway (total_chairs current_chair halfway_chair : ℕ) 
  (h_total_chairs : total_chairs = 96)
  (h_current_chair : current_chair = 66) : halfway_chair = 18 :=
sorry

end chairlift_halfway_l1151_115152


namespace find_value_of_expression_l1151_115123

theorem find_value_of_expression (a : ℝ) (h : a^2 - a - 1 = 0) : a^3 - a^2 - a + 2023 = 2023 :=
by
  sorry

end find_value_of_expression_l1151_115123


namespace intersection_length_l1151_115120

theorem intersection_length 
  (A B : ℝ × ℝ) 
  (hA : A.1^2 + A.2^2 = 1) 
  (hB : B.1^2 + B.2^2 = 1) 
  (hA_on_line : A.1 = A.2) 
  (hB_on_line : B.1 = B.2) 
  (hAB : A ≠ B) :
  dist A B = 2 :=
by sorry

end intersection_length_l1151_115120


namespace absolute_value_inequality_solution_l1151_115177

theorem absolute_value_inequality_solution (x : ℝ) :
  abs ((3 * x + 2) / (x + 2)) > 3 ↔ (x < -2) ∨ (-2 < x ∧ x < -4 / 3) :=
by
  sorry

end absolute_value_inequality_solution_l1151_115177


namespace frac_square_between_half_and_one_l1151_115117

theorem frac_square_between_half_and_one :
  let fraction := (11 : ℝ) / 12
  let expr := fraction^2
  (1 / 2) < expr ∧ expr < 1 :=
by
  let fraction := (11 : ℝ) / 12
  let expr := fraction^2
  have h1 : (1 / 2) < expr := sorry
  have h2 : expr < 1 := sorry
  exact ⟨h1, h2⟩

end frac_square_between_half_and_one_l1151_115117


namespace car_speed_l1151_115181

theorem car_speed (v : ℝ) (h : (1 / v) * 3600 = (1 / 450) * 3600 + 2) : v = 360 :=
by
  sorry

end car_speed_l1151_115181


namespace laborers_employed_l1151_115145

theorem laborers_employed 
    (H L : ℕ) 
    (h1 : H + L = 35) 
    (h2 : 140 * H + 90 * L = 3950) : 
    L = 19 :=
by
  sorry

end laborers_employed_l1151_115145


namespace maria_score_l1151_115173

theorem maria_score (m j : ℕ) (h1 : m = j + 50) (h2 : (m + j) / 2 = 112) : m = 137 :=
by
  sorry

end maria_score_l1151_115173


namespace banana_ratio_proof_l1151_115138

-- Definitions based on conditions
def initial_bananas := 310
def bananas_left_on_tree := 100
def bananas_eaten := 70

-- Auxiliary calculations for clarity
def bananas_cut := initial_bananas - bananas_left_on_tree
def bananas_remaining := bananas_cut - bananas_eaten

-- Theorem we need to prove
theorem banana_ratio_proof :
  bananas_remaining / bananas_eaten = 2 :=
by
  sorry

end banana_ratio_proof_l1151_115138


namespace alice_students_count_l1151_115190

variable (S : ℕ)
variable (students_with_own_vests := 0.20 * S)
variable (students_needing_vests := 0.80 * S)
variable (instructors : ℕ := 10)
variable (life_vests_on_hand : ℕ := 20)
variable (additional_life_vests_needed : ℕ := 22)
variable (total_life_vests_needed := life_vests_on_hand + additional_life_vests_needed)
variable (life_vests_needed_for_instructors := instructors)
variable (life_vests_needed_for_students := total_life_vests_needed - life_vests_needed_for_instructors)

theorem alice_students_count : S = 40 :=
by
  -- proof steps would go here
  sorry

end alice_students_count_l1151_115190


namespace fifth_number_l1151_115103

def sequence_sum (a b : ℕ) : ℕ :=
  a + b + (a + b) + (a + 2 * b) + (2 * a + 3 * b) + (3 * a + 5 * b)

theorem fifth_number (a b : ℕ) (h : sequence_sum a b = 2008) : 2 * a + 3 * b = 502 := by
  sorry

end fifth_number_l1151_115103


namespace central_symmetry_preserves_distance_l1151_115147

variables {Point : Type} [MetricSpace Point]

def central_symmetry (O A A' B B' : Point) : Prop :=
  dist O A = dist O A' ∧ dist O B = dist O B'

theorem central_symmetry_preserves_distance {O A A' B B' : Point}
  (h : central_symmetry O A A' B B') : dist A B = dist A' B' :=
sorry

end central_symmetry_preserves_distance_l1151_115147


namespace city_map_scale_l1151_115121

theorem city_map_scale 
  (map_length : ℝ) (actual_length_km : ℝ) (actual_length_cm : ℝ) (conversion_factor : ℝ)
  (h1 : map_length = 240) 
  (h2 : actual_length_km = 18)
  (h3 : actual_length_cm = actual_length_km * conversion_factor)
  (h4 : conversion_factor = 100000) :
  map_length / actual_length_cm = 1 / 7500 :=
by
  sorry

end city_map_scale_l1151_115121


namespace sum_of_first_seven_terms_l1151_115115

variable (a : ℕ → ℝ)
variable (d : ℝ)

-- Define the arithmetic sequence
def arithmetic_sequence (a : ℕ → ℝ) (d : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n + d

-- Given condition
axiom a3_a4_a5_sum : a 3 + a 4 + a 5 = 12

-- Statement to prove
theorem sum_of_first_seven_terms (h : arithmetic_sequence a d) : 
  a 1 + a 2 + a 3 + a 4 + a 5 + a 6 + a 7 = 28 :=
sorry

end sum_of_first_seven_terms_l1151_115115


namespace garden_area_is_correct_l1151_115132

def width_of_property : ℕ := 1000
def length_of_property : ℕ := 2250

def width_of_garden : ℕ := width_of_property / 8
def length_of_garden : ℕ := length_of_property / 10

def area_of_garden : ℕ := width_of_garden * length_of_garden

theorem garden_area_is_correct : area_of_garden = 28125 := by
  -- Skipping proof for the purpose of this example
  sorry

end garden_area_is_correct_l1151_115132


namespace product_divisible_by_12_l1151_115109

theorem product_divisible_by_12 (a b c d : ℤ) :
  12 ∣ (b - a) * (c - a) * (d - a) * (d - c) * (d - b) * (c - b) := 
by {
  sorry
}

end product_divisible_by_12_l1151_115109


namespace abs_f_x_minus_f_a_lt_l1151_115133

variable {R : Type*} [LinearOrderedField R]

def f (x : R) (c : R) := x ^ 2 - x + c

theorem abs_f_x_minus_f_a_lt (x a c : R) (h : abs (x - a) < 1) : 
  abs (f x c - f a c) < 2 * (abs a + 1) :=
by
  sorry

end abs_f_x_minus_f_a_lt_l1151_115133


namespace steve_take_home_pay_l1151_115116

def annual_salary : ℝ := 40000
def tax_rate : ℝ := 0.20
def healthcare_rate : ℝ := 0.10
def union_dues : ℝ := 800

theorem steve_take_home_pay : 
  (annual_salary - (annual_salary * tax_rate + annual_salary * healthcare_rate + union_dues)) = 27200 := 
by 
  sorry

end steve_take_home_pay_l1151_115116


namespace smallest_positive_real_number_l1151_115161

noncomputable def smallest_x : ℝ := 71 / 8

theorem smallest_positive_real_number (x : ℝ) (h₁ : ∀ y : ℝ, 0 < y ∧ (⌊y^2⌋ - y * ⌊y⌋ = 7) → x ≤ y) (h₂ : 0 < x) (h₃ : ⌊x^2⌋ - x * ⌊x⌋ = 7) : x = smallest_x :=
sorry

end smallest_positive_real_number_l1151_115161


namespace Leila_donated_2_bags_l1151_115135

theorem Leila_donated_2_bags (L : ℕ) (h1 : 25 * L + 7 = 57) : L = 2 :=
by
  sorry

end Leila_donated_2_bags_l1151_115135


namespace like_terms_exponent_equality_l1151_115110

theorem like_terms_exponent_equality (m n : ℕ) (a b : ℝ) 
    (H : 3 * a^m * b^2 = 2/3 * a * b^n) : m = 1 ∧ n = 2 :=
by
  sorry

end like_terms_exponent_equality_l1151_115110


namespace negation_of_proposition_l1151_115197
open Real

theorem negation_of_proposition :
  ¬ (∃ x₀ : ℝ, (2/x₀) + log x₀ ≤ 0) ↔ ∀ x : ℝ, (2/x) + log x > 0 :=
by
  sorry

end negation_of_proposition_l1151_115197


namespace preimage_of_mapping_l1151_115105

def f (a b : ℝ) : ℝ × ℝ := (a + 2 * b, 2 * a - b)

theorem preimage_of_mapping : ∃ (a b : ℝ), f a b = (3, 1) ∧ (a, b) = (1, 1) :=
by
  sorry

end preimage_of_mapping_l1151_115105


namespace Michael_points_l1151_115106

theorem Michael_points (total_points : ℕ) (num_other_players : ℕ) (avg_points : ℕ) (Michael_points : ℕ) 
  (h1 : total_points = 75)
  (h2 : num_other_players = 5)
  (h3 : avg_points = 6)
  (h4 : Michael_points = total_points - num_other_players * avg_points) :
  Michael_points = 45 := by
  sorry

end Michael_points_l1151_115106


namespace difference_in_tiles_l1151_115169

theorem difference_in_tiles (n : ℕ) (hn : n = 9) : (n + 1)^2 - n^2 = 19 :=
by sorry

end difference_in_tiles_l1151_115169
