import Mathlib

namespace students_with_uncool_parents_l783_78308

theorem students_with_uncool_parents (class_size : ℕ) (cool_dads : ℕ) (cool_moms : ℕ) (both_cool_parents : ℕ) : 
  class_size = 40 → cool_dads = 18 → cool_moms = 20 → both_cool_parents = 10 → 
  (class_size - (cool_dads - both_cool_parents + cool_moms - both_cool_parents + both_cool_parents) = 12) :=
by
  sorry

end students_with_uncool_parents_l783_78308


namespace min_employees_needed_l783_78347

-- Define the conditions
variable (W A : Finset ℕ)
variable (n_W n_A n_WA : ℕ)

-- Assume the given condition values
def sizeW := 95
def sizeA := 80
def sizeWA := 30

-- Define the proof problem
theorem min_employees_needed :
  (sizeW + sizeA - sizeWA) = 145 :=
by sorry

end min_employees_needed_l783_78347


namespace circle_area_irrational_of_rational_radius_l783_78344

theorem circle_area_irrational_of_rational_radius (r : ℚ) : ¬ ∃ A : ℚ, A = π * (r:ℝ) * (r:ℝ) :=
by sorry

end circle_area_irrational_of_rational_radius_l783_78344


namespace proof_problem_l783_78315

-- Definitions of the conditions
def cond1 (r : ℕ) : Prop := 2^r = 16
def cond2 (s : ℕ) : Prop := 5^s = 25

-- Statement of the problem
theorem proof_problem (r s : ℕ) (h₁ : cond1 r) (h₂ : cond2 s) : r + s = 6 :=
by
  sorry

end proof_problem_l783_78315


namespace floor_x_floor_x_eq_20_l783_78317

theorem floor_x_floor_x_eq_20 (x : ℝ) : ⌊x * ⌊x⌋⌋ = 20 ↔ 5 ≤ x ∧ x < 5.25 := 
sorry

end floor_x_floor_x_eq_20_l783_78317


namespace problem1_line_equation_problem2_circle_equation_l783_78338

-- Problem 1: Equation of a specific line
def line_intersection (x y : ℝ) : Prop := 
  2 * x + y - 8 = 0 ∧ x - 2 * y + 1 = 0

def line_perpendicular (x y : ℝ) : Prop :=
  6 * x - 8 * y + 3 = 0

noncomputable def find_line (x y : ℝ) : Prop :=
  ∃ (l : ℝ), (8 * x + 6 * y + l = 0) ∧ 
  line_intersection x y ∧ line_perpendicular x y

theorem problem1_line_equation : ∃ (x y : ℝ), find_line x y :=
sorry

-- Problem 2: Equation of a specific circle
def point_A (x y : ℝ) : Prop := 
  x = 5 ∧ y = 2

def point_B (x y : ℝ) : Prop := 
  x = 3 ∧ y = -2

def center_on_line (x y : ℝ) : Prop :=
  2 * x - y = 3

noncomputable def find_circle (x y r : ℝ) : Prop :=
  ((x - 2)^2 + (y - 1)^2 = r) ∧
  ∃ x1 y1 x2 y2, point_A x1 y1 ∧ point_B x2 y2 ∧ center_on_line x y ∧ ((x1 - x)^2 + (y1 - y)^2 = r)

theorem problem2_circle_equation : ∃ (x y r : ℝ), find_circle x y 10 :=
sorry

end problem1_line_equation_problem2_circle_equation_l783_78338


namespace solve_equation_l783_78365

theorem solve_equation (a : ℝ) (x : ℝ) : (2 * a * x + 3) / (a - x) = 3 / 4 → x = 1 → a = -3 :=
by
  intros h h1
  rw [h1] at h
  sorry

end solve_equation_l783_78365


namespace base_number_l783_78359

theorem base_number (a x : ℕ) (h1 : a ^ x - a ^ (x - 2) = 3 * 2 ^ 11) (h2 : x = 13) : a = 2 :=
by
  sorry

end base_number_l783_78359


namespace solve_inequality_l783_78322

theorem solve_inequality (x : ℝ) :
  (4 * x^4 + x^2 + 4 * x - 5 * x^2 * |x + 2| + 4) ≥ 0 ↔ 
  x ∈ Set.Iic (-1) ∪ Set.Icc ((1 - Real.sqrt 33) / 8) ((1 + Real.sqrt 33) / 8) ∪ Set.Ici 2 :=
by
  sorry

end solve_inequality_l783_78322


namespace common_root_conds_l783_78375

theorem common_root_conds (α a b c d : ℝ) (h₁ : a ≠ c)
  (h₂ : α^2 + a * α + b = 0)
  (h₃ : α^2 + c * α + d = 0) :
  α = (d - b) / (a - c) :=
by 
  sorry

end common_root_conds_l783_78375


namespace coupon1_greater_l783_78395

variable (x : ℝ)

def coupon1_discount (x : ℝ) : ℝ := 0.15 * x
def coupon2_discount : ℝ := 50
def coupon3_discount (x : ℝ) : ℝ := 0.25 * x - 62.5

theorem coupon1_greater (x : ℝ) (hx1 : 333.33 < x ∧ x < 625) : 
  coupon1_discount x > coupon2_discount ∧ coupon1_discount x > coupon3_discount x := by
  sorry

end coupon1_greater_l783_78395


namespace compute_g_f_1_l783_78348

def f (x : ℝ) : ℝ := x^3 - 2 * x + 3
def g (x : ℝ) : ℝ := 2 * x^2 - 3 * x + 1

theorem compute_g_f_1 : g (f 1) = 3 :=
by
  sorry

end compute_g_f_1_l783_78348


namespace product_divisible_by_14_l783_78397

theorem product_divisible_by_14 (a b c d : ℤ) (h : 7 * a + 8 * b = 14 * c + 28 * d) : 14 ∣ a * b := 
sorry

end product_divisible_by_14_l783_78397


namespace sufficient_but_not_necessary_condition_subset_condition_l783_78380

open Set

variable (a : ℝ)
def U : Set ℝ := univ
def A : Set ℝ := {x : ℝ | 1 ≤ x ∧ x ≤ 5}
def B (a : ℝ) : Set ℝ := {x : ℝ | -1-2*a ≤ x ∧ x ≤ a-2}

theorem sufficient_but_not_necessary_condition (H : ∃ x ∈ A, x ∉ B a) : a ≥ 7 := sorry

theorem subset_condition (H : B a ⊆ A) : a < 1/3 := sorry

end sufficient_but_not_necessary_condition_subset_condition_l783_78380


namespace solve_equation_5x_plus1_div_2x_sq_plus_5x_minus3_eq_2x_div_2x_minus1_l783_78391

theorem solve_equation_5x_plus1_div_2x_sq_plus_5x_minus3_eq_2x_div_2x_minus1 :
  ∀ x : ℝ, 2 * x ^ 2 + 5 * x - 3 ≠ 0 ∧ 2 * x - 1 ≠ 0 → 
  (5 * x + 1) / (2 * x ^ 2 + 5 * x - 3) = (2 * x) / (2 * x - 1) → 
  x = -1 :=
by
  intro x h_cond h_eq
  sorry

end solve_equation_5x_plus1_div_2x_sq_plus_5x_minus3_eq_2x_div_2x_minus1_l783_78391


namespace sum_geq_4k_l783_78399

theorem sum_geq_4k (a b k : ℕ) (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_k : k > 1)
  (h_lcm_gcd : Nat.lcm a b + Nat.gcd a b = k * (a + b)) : a + b ≥ 4 * k := 
by 
  sorry

end sum_geq_4k_l783_78399


namespace geometric_sequence_S9_l783_78335

theorem geometric_sequence_S9 (S : ℕ → ℝ) (S3_eq : S 3 = 2) (S6_eq : S 6 = 6) : S 9 = 14 :=
by
  sorry

end geometric_sequence_S9_l783_78335


namespace gamma_max_success_ratio_l783_78358

theorem gamma_max_success_ratio (x y z w : ℕ) (h_yw : y + w = 500)
    (h_gamma_first_day : 0 < x ∧ x < 170 * y / 280)
    (h_gamma_second_day : 0 < z ∧ z < 150 * w / 220)
    (h_less_than_500 : (28 * x + 22 * z) / 17 < 500) :
    (x + z) ≤ 170 := 
sorry

end gamma_max_success_ratio_l783_78358


namespace ninth_term_arithmetic_sequence_l783_78390

variable (a d : ℕ)

def arithmetic_sequence_sum (a d : ℕ) : ℕ :=
  a + (a + d) + (a + 2 * d) + (a + 3 * d) + (a + 4 * d) + (a + 5 * d)

theorem ninth_term_arithmetic_sequence (h1 : arithmetic_sequence_sum a d = 21) (h2 : a + 6 * d = 7) : a + 8 * d = 9 :=
by
  sorry

end ninth_term_arithmetic_sequence_l783_78390


namespace solution_correct_l783_78392

def mascot_options := ["A Xiang", "A He", "A Ru", "A Yi", "Le Yangyang"]

def volunteer_options := ["A", "B", "C", "D", "E"]

noncomputable def count_valid_assignments (mascots : List String) (volunteers : List String) : Nat :=
  let all_assignments := mascots.permutations
  let valid_assignments := all_assignments.filter (λ p =>
    (p.get! 0 = "A Xiang" ∨ p.get! 1 = "A Xiang") ∧ p.get! 2 ≠ "Le Yangyang")
  valid_assignments.length

theorem solution_correct :
  count_valid_assignments mascot_options volunteer_options = 36 :=
by
  sorry

end solution_correct_l783_78392


namespace sector_angle_l783_78382

theorem sector_angle (l S : ℝ) (r α : ℝ) 
  (h_arc_length : l = 6)
  (h_area : S = 6)
  (h_area_formula : S = 1/2 * l * r)
  (h_arc_formula : l = r * α) : 
  α = 3 :=
by
  sorry

end sector_angle_l783_78382


namespace ab_greater_than_a_plus_b_l783_78303

theorem ab_greater_than_a_plus_b (a b : ℝ) (ha : a ≥ 2) (hb : b > 2) : a * b > a + b :=
by
  sorry

end ab_greater_than_a_plus_b_l783_78303


namespace range_of_a_l783_78388

theorem range_of_a (a : ℝ) 
  (h : ¬ ∃ x : ℝ, Real.exp x ≤ 2 * x + a) : a < 2 - 2 * Real.log 2 := 
  sorry

end range_of_a_l783_78388


namespace Linda_has_24_classmates_l783_78330

theorem Linda_has_24_classmates 
  (cookies_per_student : ℕ := 10)
  (cookies_per_batch : ℕ := 48)
  (chocolate_chip_batches : ℕ := 2)
  (oatmeal_raisin_batches : ℕ := 1)
  (additional_batches : ℕ := 2) : 
  (chocolate_chip_batches * cookies_per_batch + oatmeal_raisin_batches * cookies_per_batch + additional_batches * cookies_per_batch) / cookies_per_student = 24 := 
by 
  sorry

end Linda_has_24_classmates_l783_78330


namespace proof_problem_l783_78311

theorem proof_problem :
  ∀ (X : ℝ), 213 * 16 = 3408 → (213 * 16) + (1.6 * 2.13) = X → X - (5 / 2) * 1.25 = 3408.283 :=
by
  intros X h1 h2
  sorry

end proof_problem_l783_78311


namespace sequence_general_formula_l783_78332

theorem sequence_general_formula (n : ℕ) (h : n ≥ 1) :
  ∃ a : ℕ → ℝ, a 1 = 1 ∧ (∀ n ≥ 1, a (n + 1) = a n / (1 + a n)) ∧ a n = (1 : ℝ) / n :=
by
  sorry

end sequence_general_formula_l783_78332


namespace correspond_half_l783_78373

theorem correspond_half (m n : ℕ) 
  (H : ∀ h : Fin m, ∃ g_set : Finset (Fin n), (g_set.card = n / 2) ∧ (∀ g : Fin n, g ∈ g_set))
  (G : ∀ g : Fin n, ∃ h_set : Finset (Fin m), (h_set.card ≤ m / 2) ∧ (∀ h : Fin m, h ∈ h_set)) :
  (∀ h : Fin m, ∀ g_set : Finset (Fin n), g_set.card = n / 2) ∧ (∀ g : Fin n, ∀ h_set : Finset (Fin m), h_set.card = m / 2) :=
by
  sorry

end correspond_half_l783_78373


namespace no_both_squares_l783_78361

theorem no_both_squares {x y : ℕ} (hx : x > 0) (hy : y > 0) : ¬ (∃ a b : ℕ, a^2 = x^2 + 2 * y ∧ b^2 = y^2 + 2 * x) :=
by
  sorry

end no_both_squares_l783_78361


namespace integer_solutions_count_for_equation_l783_78370

theorem integer_solutions_count_for_equation :
  (∃ n : ℕ, (∀ x y : ℤ, (1/x + 1/y = 1/7) → (x ≠ 0) → (y ≠ 0) → n = 5 )) :=
sorry

end integer_solutions_count_for_equation_l783_78370


namespace range_of_a_l783_78319

theorem range_of_a (a : ℝ) (h : (∀ x1 x2 : ℝ, x1 < x2 → (2 * a - 1) ^ x1 > (2 * a - 1) ^ x2)) :
  1 / 2 < a ∧ a < 1 :=
by
  sorry

end range_of_a_l783_78319


namespace handshake_count_250_l783_78337

theorem handshake_count_250 (n m : ℕ) (h1 : n = 5) (h2 : m = 5) :
  (n * m * (n * m - 1 - (n - 1))) / 2 = 250 :=
by
  -- Traditionally the theorem proof part goes here but it is omitted
  sorry

end handshake_count_250_l783_78337


namespace raghu_investment_l783_78393

theorem raghu_investment
  (R trishul vishal : ℝ)
  (h1 : trishul = 0.90 * R)
  (h2 : vishal = 0.99 * R)
  (h3 : R + trishul + vishal = 6647) :
  R = 2299.65 :=
by
  sorry

end raghu_investment_l783_78393


namespace find_angle_degree_l783_78340

-- Define the angle
variable {x : ℝ}

-- Define the conditions
def complement (x : ℝ) : ℝ := 90 - x
def supplement (x : ℝ) : ℝ := 180 - x

-- Define the given condition
def condition (x : ℝ) : Prop := complement x = (1/3) * (supplement x)

-- The theorem statement
theorem find_angle_degree (x : ℝ) (h : condition x) : x = 45 :=
by
  sorry

end find_angle_degree_l783_78340


namespace quadratic_roots_ratio_l783_78326

theorem quadratic_roots_ratio (m n p : ℝ) (h₁ : m ≠ 0) (h₂ : n ≠ 0) (h₃ : p ≠ 0)
    (h₄ : ∀ (s₁ s₂ : ℝ), s₁ + s₂ = -p ∧ s₁ * s₂ = m ∧ 3 * s₁ + 3 * s₂ = -m ∧ 9 * s₁ * s₂ = n) :
    n / p = 27 :=
sorry

end quadratic_roots_ratio_l783_78326


namespace Teena_speed_is_55_l783_78341

def Teena_speed (Roe_speed T : ℝ) (initial_gap final_gap time : ℝ) : Prop :=
  Roe_speed * time + initial_gap + final_gap = T * time

theorem Teena_speed_is_55 :
  Teena_speed 40 55 7.5 15 1.5 :=
by 
  sorry

end Teena_speed_is_55_l783_78341


namespace carl_typing_speed_l783_78387

theorem carl_typing_speed (words_per_day: ℕ) (minutes_per_day: ℕ) (total_words: ℕ) (days: ℕ) : 
  words_per_day = total_words / days ∧ 
  minutes_per_day = 4 * 60 ∧ 
  (words_per_day / minutes_per_day) = 50 :=
by 
  sorry

end carl_typing_speed_l783_78387


namespace selected_female_athletes_l783_78346

-- Definitions based on conditions
def total_male_athletes := 56
def total_female_athletes := 42
def selected_male_athletes := 8
def male_to_female_ratio := 4 / 3

-- Problem statement: Prove that the number of selected female athletes is 6
theorem selected_female_athletes :
  selected_male_athletes * (3 / 4) = 6 :=
by 
  -- Placeholder for the proof
  sorry

end selected_female_athletes_l783_78346


namespace smallest_percentage_all_correct_l783_78320

theorem smallest_percentage_all_correct (p1 p2 p3 : ℝ) 
  (h1 : p1 = 0.9) 
  (h2 : p2 = 0.8)
  (h3 : p3 = 0.7) :
  ∃ x, x = 0.4 ∧ (x ≤ 1 - ((1 - p1) + (1 - p2) + (1 - p3))) :=
by 
  sorry

end smallest_percentage_all_correct_l783_78320


namespace shelves_used_l783_78316

-- Define the initial conditions
def initial_stock : Float := 40.0
def additional_stock : Float := 20.0
def books_per_shelf : Float := 4.0

-- Define the total number of books
def total_books : Float := initial_stock + additional_stock

-- Define the number of shelves
def number_of_shelves : Float := total_books / books_per_shelf

-- The proof statement that needs to be proven
theorem shelves_used : number_of_shelves = 15.0 :=
by
  -- The proof will go here
  sorry

end shelves_used_l783_78316


namespace sqrt_54_sub_sqrt_6_l783_78314

theorem sqrt_54_sub_sqrt_6 : Real.sqrt 54 - Real.sqrt 6 = 2 * Real.sqrt 6 := by
  sorry

end sqrt_54_sub_sqrt_6_l783_78314


namespace third_chapter_pages_l783_78345

theorem third_chapter_pages (x : ℕ) (h : 18 = x + 15) : x = 3 :=
by
  sorry

end third_chapter_pages_l783_78345


namespace perpendicular_slope_l783_78342

theorem perpendicular_slope (k : ℝ) : (∀ x, y = k*x) ∧ (∀ x, y = 2*x + 1) → k = -1 / 2 :=
by
  intro h
  sorry

end perpendicular_slope_l783_78342


namespace count_two_digit_primes_with_given_conditions_l783_78318

def is_two_digit_prime (n : ℕ) : Prop :=
  n ≥ 10 ∧ n < 100 ∧ Prime n

def sum_of_digits_is_nine (n : ℕ) : Prop :=
  let tens := n / 10
  let units := n % 10
  tens + units = 9

def tens_greater_than_units (n : ℕ) : Prop :=
  let tens := n / 10
  let units := n % 10
  tens > units

theorem count_two_digit_primes_with_given_conditions :
  ∃ count : ℕ, count = 0 ∧ ∀ n, is_two_digit_prime n ∧ sum_of_digits_is_nine n ∧ tens_greater_than_units n → false :=
by
  -- proof goes here
  sorry

end count_two_digit_primes_with_given_conditions_l783_78318


namespace lin_reg_proof_l783_78302

variable (x y : List ℝ)
variable (n : ℝ := 10)
variable (sum_x : ℝ := 80)
variable (sum_y : ℝ := 20)
variable (sum_xy : ℝ := 184)
variable (sum_x2 : ℝ := 720)

noncomputable def mean (lst: List ℝ) (n: ℝ) : ℝ := (List.sum lst) / n

noncomputable def lin_reg_slope (n sum_x sum_y sum_xy sum_x2 : ℝ) : ℝ :=
  (sum_xy - n * (sum_x / n) * (sum_y / n)) / (sum_x2 - n * (sum_x / n) ^ 2)

noncomputable def lin_reg_intercept (sum_x sum_y : ℝ) (slope : ℝ) (n : ℝ) : ℝ :=
  (sum_y / n) - slope * (sum_x / n)

theorem lin_reg_proof :
  lin_reg_slope n sum_x sum_y sum_xy sum_x2 = 0.3 ∧ 
  lin_reg_intercept sum_x sum_y 0.3 n = -0.4 ∧ 
  (0.3 * 7 - 0.4 = 1.7) :=
by
  sorry

end lin_reg_proof_l783_78302


namespace balloons_lost_l783_78356

-- Definitions corresponding to the conditions
def initial_balloons : ℕ := 7
def current_balloons : ℕ := 4

-- The mathematically equivalent proof problem
theorem balloons_lost : initial_balloons - current_balloons = 3 := by
  -- proof steps would go here, but we use sorry to skip them 
  sorry

end balloons_lost_l783_78356


namespace inscribed_circle_radius_isosceles_triangle_l783_78364

noncomputable def isosceles_triangle_base : ℝ := 30 -- base AC
noncomputable def isosceles_triangle_equal_side : ℝ := 39 -- equal sides AB and BC

theorem inscribed_circle_radius_isosceles_triangle :
  ∀ (AC AB BC: ℝ), 
  AC = isosceles_triangle_base → 
  AB = isosceles_triangle_equal_side →
  BC = isosceles_triangle_equal_side →
  ∃ r : ℝ, r = 10 := 
by
  intros AC AB BC hAC hAB hBC
  sorry

end inscribed_circle_radius_isosceles_triangle_l783_78364


namespace largest_divisible_by_88_l783_78383

theorem largest_divisible_by_88 (n : ℕ) (h₁ : n = 9999) (h₂ : n % 88 = 55) : n - 55 = 9944 := by
  sorry

end largest_divisible_by_88_l783_78383


namespace total_sticks_needed_l783_78329

theorem total_sticks_needed :
  let simon_sticks := 36
  let gerry_sticks := 2 * (simon_sticks / 3)
  let total_simon_and_gerry := simon_sticks + gerry_sticks
  let micky_sticks := total_simon_and_gerry + 9
  total_simon_and_gerry + micky_sticks = 129 :=
by
  sorry

end total_sticks_needed_l783_78329


namespace solve_for_q_l783_78343

theorem solve_for_q (m n q : ℕ) (h1 : 7/8 = m/96) (h2 : 7/8 = (n + m)/112) (h3 : 7/8 = (q - m)/144) :
  q = 210 :=
sorry

end solve_for_q_l783_78343


namespace transform_quadratic_l783_78333

theorem transform_quadratic (x m n : ℝ) 
  (h : x^2 - 6 * x - 1 = 0) : 
  (x + m)^2 = n ↔ (m = 3 ∧ n = 10) :=
by sorry

end transform_quadratic_l783_78333


namespace slices_per_person_is_correct_l783_78377

-- Conditions
def slices_per_tomato : Nat := 8
def total_tomatoes : Nat := 20
def people_for_meal : Nat := 8

-- Calculate number of slices for a single person
def slices_needed_for_single_person (slices_per_tomato : Nat) (total_tomatoes : Nat) (people_for_meal : Nat) : Nat :=
  (slices_per_tomato * total_tomatoes) / people_for_meal

-- The statement to be proved
theorem slices_per_person_is_correct : slices_needed_for_single_person slices_per_tomato total_tomatoes people_for_meal = 20 :=
by
  sorry

end slices_per_person_is_correct_l783_78377


namespace eval_expression_l783_78396

theorem eval_expression : 7^3 + 3 * 7^2 + 3 * 7 + 1 = 512 := 
by 
  sorry

end eval_expression_l783_78396


namespace simplify_expression_l783_78351

theorem simplify_expression (x : ℝ) : (3 * x + 15) + (100 * x + 15) + (10 * x - 5) = 113 * x + 25 :=
by
  sorry

end simplify_expression_l783_78351


namespace impossible_to_empty_pile_l783_78334

theorem impossible_to_empty_pile (a b c : ℕ) (h : a = 1993 ∧ b = 199 ∧ c = 19) : 
  ¬ (∃ x y z : ℕ, (x + y + z = 0) ∧ (x = a ∨ x = b ∨ x = c ∧ y = a ∨ y = b ∨ y = c ∧ z = a ∨ z = b ∨ z = c)) := 
sorry

end impossible_to_empty_pile_l783_78334


namespace distance_between_foci_is_six_l783_78327

-- Lean 4 Statement
noncomputable def distance_between_foci (p1 p2 p3 : ℝ × ℝ) : ℝ :=
  if (p1 = (1, 3) ∧ p2 = (6, -1) ∧ p3 = (11, 3)) then 6 else 0

theorem distance_between_foci_is_six : distance_between_foci (1, 3) (6, -1) (11, 3) = 6 :=
by
  sorry

end distance_between_foci_is_six_l783_78327


namespace relatively_prime_dates_in_september_l783_78355

-- Define a condition to check if two numbers are relatively prime
def relatively_prime (a b : ℕ) : Prop := Nat.gcd a b = 1

-- Define the number of days in September
def days_in_september := 30

-- Define the month of September as the 9th month
def month_of_september := 9

-- Define the proposition that the number of relatively prime dates in September is 20
theorem relatively_prime_dates_in_september : 
  ∃ count, (count = 20 ∧ ∀ day, day ∈ Finset.range (days_in_september + 1) → relatively_prime month_of_september day → count = 20) := sorry

end relatively_prime_dates_in_september_l783_78355


namespace five_a_plus_five_b_eq_neg_twenty_five_thirds_l783_78369

variable (g f : ℝ → ℝ)
variable (a b : ℝ)
axiom g_def : ∀ x, g x = 3 * x + 5
axiom g_inv_rel : ∀ x, g x = (f⁻¹ x) - 1
axiom f_def : ∀ x, f x = a * x + b
axiom f_inv_def : ∀ x, f⁻¹ (f x) = x

theorem five_a_plus_five_b_eq_neg_twenty_five_thirds :
    5 * a + 5 * b = -25 / 3 :=
sorry

end five_a_plus_five_b_eq_neg_twenty_five_thirds_l783_78369


namespace hexagon_ratio_l783_78305

noncomputable def ratio_of_hexagon_areas (s : ℝ) : ℝ :=
  let area_ABCDEF := (3 * Real.sqrt 3 / 2) * s^2
  let side_smaller := (3 * s) / 2
  let area_smaller := (3 * Real.sqrt 3 / 2) * side_smaller^2
  area_smaller / area_ABCDEF

theorem hexagon_ratio (s : ℝ) : ratio_of_hexagon_areas s = 9 / 4 :=
by
  sorry

end hexagon_ratio_l783_78305


namespace max_profit_price_range_for_minimum_profit_l783_78307

noncomputable def functional_relationship (x : ℝ) : ℝ :=
-10 * x^2 + 2000 * x - 84000

theorem max_profit :
  ∃ x, (∀ x₀, x₀ ≠ x → functional_relationship x₀ < functional_relationship x) ∧
  functional_relationship x = 16000 := 
sorry

theorem price_range_for_minimum_profit :
  ∀ (x : ℝ), 
  -10 * (x - 100)^2 + 16000 - 1750 ≥ 12000 → 
  85 ≤ x ∧ x ≤ 115 :=
sorry

end max_profit_price_range_for_minimum_profit_l783_78307


namespace ratio_of_areas_l783_78336

theorem ratio_of_areas (len_rect width_rect area_tri : ℝ) (h1 : len_rect = 6) (h2 : width_rect = 4) (h3 : area_tri = 60) :
    (len_rect * width_rect) / area_tri = 2 / 5 :=
by
  rw [h1, h2, h3]
  norm_num

end ratio_of_areas_l783_78336


namespace least_positive_integer_greater_than_100_l783_78301

theorem least_positive_integer_greater_than_100 : ∃ n : ℕ, n > 100 ∧ (∀ k ∈ [2, 3, 4, 5, 6, 7, 8, 9, 10], n % k = 1) ∧ n = 2521 :=
by
  sorry

end least_positive_integer_greater_than_100_l783_78301


namespace convert_to_scientific_notation_l783_78366

theorem convert_to_scientific_notation :
  (448000 : ℝ) = 4.48 * 10^5 :=
by
  sorry

end convert_to_scientific_notation_l783_78366


namespace max_integer_is_110003_l783_78379

def greatest_integer : Prop :=
  let a := 100004
  let b := 110003
  let c := 102002
  let d := 100301
  let e := 100041
  b > a ∧ b > c ∧ b > d ∧ b > e

theorem max_integer_is_110003 : greatest_integer :=
by
  sorry

end max_integer_is_110003_l783_78379


namespace polynomial_divisible_by_x_minus_2_l783_78325

theorem polynomial_divisible_by_x_minus_2 (k : ℝ) :
  (2 * (2 : ℝ)^3 - 8 * (2 : ℝ)^2 + k * (2 : ℝ) - 10 = 0) → 
  k = 13 :=
by 
  intro h
  sorry

end polynomial_divisible_by_x_minus_2_l783_78325


namespace min_value_a_is_1_or_100_l783_78313

noncomputable def f (x : ℝ) : ℝ := x + 100 / x

theorem min_value_a_is_1_or_100 (a : ℝ) (m1 m2 : ℝ) 
  (h1 : a > 0) 
  (h_m1 : ∀ x, 0 < x ∧ x ≤ a → f x ≥ m1)
  (h_m1_min : ∃ x, 0 < x ∧ x ≤ a ∧ f x = m1)
  (h_m2 : ∀ x, a ≤ x → f x ≥ m2)
  (h_m2_min : ∃ x, a ≤ x ∧ f x = m2)
  (h_prod : m1 * m2 = 2020) : 
  a = 1 ∨ a = 100 :=
sorry

end min_value_a_is_1_or_100_l783_78313


namespace unique_c1_c2_exists_l783_78354

theorem unique_c1_c2_exists (a_0 a_1 x_1 x_2 : ℝ) (h_distinct : x_1 ≠ x_2) : 
  ∃! (c_1 c_2 : ℝ), ∀ n : ℕ, a_n = c_1 * x_1^n + c_2 * x_2^n :=
sorry

end unique_c1_c2_exists_l783_78354


namespace find_m_n_condition_l783_78304

theorem find_m_n_condition (m n : ℕ) :
  m ≥ 1 ∧ n > m ∧ (42 ^ n ≡ 42 ^ m [MOD 100]) ∧ m + n = 24 :=
sorry

end find_m_n_condition_l783_78304


namespace min_baseball_cards_divisible_by_15_l783_78352

theorem min_baseball_cards_divisible_by_15 :
  ∀ (j m c e t : ℕ),
    j = m →
    m = c - 6 →
    c = 20 →
    e = 2 * (j + m) →
    t = c + m + j + e →
    t ≥ 104 →
    ∃ k : ℕ, t = 15 * k ∧ t = 105 :=
by
  intros j m c e t h1 h2 h3 h4 h5 h6
  sorry

end min_baseball_cards_divisible_by_15_l783_78352


namespace round_robin_teams_l783_78378

theorem round_robin_teams (x : ℕ) (h : (x * (x - 1)) / 2 = 45) : x = 10 := 
by
  sorry

end round_robin_teams_l783_78378


namespace find_a_l783_78321

theorem find_a (a r s : ℝ) (h1 : r^2 = a) (h2 : 2 * r * s = 24) (h3 : s^2 = 9) : a = 16 :=
sorry

end find_a_l783_78321


namespace zero_in_interval_l783_78309

noncomputable def f (x : ℝ) : ℝ := 2 * x - 8 + Real.logb 3 x

theorem zero_in_interval : 
  (0 < 3) ∧ (3 < 4) → (f 3 < 0) ∧ (f 4 > 0) → ∃ x, 3 < x ∧ x < 4 ∧ f x = 0 :=
by
  intro h1 h2
  obtain ⟨h3, h4⟩ := h2
  sorry

end zero_in_interval_l783_78309


namespace hostel_cost_for_23_days_l783_78331

theorem hostel_cost_for_23_days :
  let first_week_days := 7
  let additional_days := 23 - first_week_days
  let cost_first_week := 18 * first_week_days
  let cost_additional_weeks := 11 * additional_days
  23 * ((cost_first_week + cost_additional_weeks) / 23) = 302 :=
by sorry

end hostel_cost_for_23_days_l783_78331


namespace problem_inequality_l783_78363

theorem problem_inequality 
  (a b c d : ℝ) 
  (h_pos_a : 0 < a) (h_le_a : a ≤ 1)
  (h_pos_b : 0 < b) (h_le_b : b ≤ 1)
  (h_pos_c : 0 < c) (h_le_c : c ≤ 1)
  (h_pos_d : 0 < d) (h_le_d : d ≤ 1) :
  (1 / (a^2 + b^2 + c^2 + d^2)) ≥ (1 / 4) + (1 - a) * (1 - b) * (1 - c) * (1 - d) :=
by
  sorry

end problem_inequality_l783_78363


namespace age_difference_l783_78386

theorem age_difference (a b c : ℕ) (h : a + b = b + c + 10) : c = a - 10 :=
by
  sorry

end age_difference_l783_78386


namespace equivalent_terminal_side_l783_78381

theorem equivalent_terminal_side (k : ℤ) : 
    (∃ k : ℤ, (5 * π / 3 = -π / 3 + 2 * π * k)) :=
sorry

end equivalent_terminal_side_l783_78381


namespace correct_calculation_l783_78328

theorem correct_calculation :
  (∀ x : ℤ, x^5 + x^3 ≠ x^8) ∧
  (∀ x : ℤ, x^5 - x^3 ≠ x^2) ∧
  (∀ x : ℤ, x^5 * x^3 = x^8) ∧
  (∀ x : ℤ, (-3 * x)^3 ≠ -9 * x^3) :=
by
  sorry

end correct_calculation_l783_78328


namespace average_marks_of_passed_l783_78310

theorem average_marks_of_passed
  (total_boys : ℕ)
  (average_all : ℕ)
  (average_failed : ℕ)
  (passed_boys : ℕ)
  (num_boys := 120)
  (avg_all := 37)
  (avg_failed := 15)
  (passed := 110)
  (failed_boys := total_boys - passed_boys)
  (total_marks_all := average_all * total_boys)
  (total_marks_failed := average_failed * failed_boys)
  (total_marks_passed := total_marks_all - total_marks_failed)
  (average_passed := total_marks_passed / passed_boys) :
  average_passed = 39 :=
by
  -- start of proof
  sorry

end average_marks_of_passed_l783_78310


namespace alyssa_cookie_count_l783_78372

variable (Aiyanna_cookies Alyssa_cookies : ℕ)
variable (h1 : Aiyanna_cookies = 140)
variable (h2 : Aiyanna_cookies = Alyssa_cookies + 11)

theorem alyssa_cookie_count : Alyssa_cookies = 129 := by
  -- We can use the given conditions to prove the theorem
  sorry

end alyssa_cookie_count_l783_78372


namespace correct_inequality_l783_78368

variables {a b c : ℝ}
def quadratic_function (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem correct_inequality (h_a_pos : a > 0) (h_discriminant_pos : b^2 - 4 * a * c > 0) (h_c_neg : c < 0) (h_b_neg : b < 0) :
  a * b * c > 0 :=
sorry

end correct_inequality_l783_78368


namespace GCF_seven_eight_factorial_l783_78362

-- Given conditions
def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

-- Calculating 7! and 8!
def seven_factorial := factorial 7
def eight_factorial := factorial 8

-- Proof statement
theorem GCF_seven_eight_factorial : ∃ g, g = seven_factorial ∧ g = Nat.gcd seven_factorial eight_factorial ∧ g = 5040 :=
by sorry

end GCF_seven_eight_factorial_l783_78362


namespace fraction_eq_zero_iff_x_eq_6_l783_78374

theorem fraction_eq_zero_iff_x_eq_6 (x : ℝ) : (x - 6) / (5 * x) = 0 ↔ x = 6 :=
by
  sorry

end fraction_eq_zero_iff_x_eq_6_l783_78374


namespace walking_area_calculation_l783_78376

noncomputable def walking_area_of_park (park_length park_width fountain_radius : ℝ) : ℝ :=
  let park_area := park_length * park_width
  let fountain_area := Real.pi * fountain_radius^2
  park_area - fountain_area

theorem walking_area_calculation :
  walking_area_of_park 50 30 5 = 1500 - 25 * Real.pi :=
by
  sorry

end walking_area_calculation_l783_78376


namespace min_cuts_for_100_quadrilaterals_l783_78300

theorem min_cuts_for_100_quadrilaterals : ∃ n : ℕ, (∃ q : ℕ, q = 100 ∧ n + 1 = q + 99) ∧ n = 1699 :=
sorry

end min_cuts_for_100_quadrilaterals_l783_78300


namespace find_c_l783_78339

def p (x : ℝ) := 4 * x - 9
def q (x : ℝ) (c : ℝ) := 5 * x - c

theorem find_c : ∃ (c : ℝ), p (q 3 c) = 14 ∧ c = 9.25 :=
by
  sorry

end find_c_l783_78339


namespace div_by_6_l783_78350

theorem div_by_6 (m : ℕ) : 6 ∣ (m^3 + 11 * m) :=
sorry

end div_by_6_l783_78350


namespace apples_count_l783_78357

def total_apples (mike_apples nancy_apples keith_apples : Nat) : Nat :=
  mike_apples + nancy_apples + keith_apples

theorem apples_count :
  total_apples 7 3 6 = 16 :=
by
  rfl

end apples_count_l783_78357


namespace solve_for_x_l783_78367

theorem solve_for_x (x : ℂ) (i : ℂ) (h : i ^ 2 = -1) (eqn : 3 + i * x = 5 - 2 * i * x) : x = i / 3 :=
sorry

end solve_for_x_l783_78367


namespace three_digit_divisible_by_11_l783_78353

theorem three_digit_divisible_by_11 {x y z : ℕ} 
  (h1 : 0 ≤ x ∧ x < 10) 
  (h2 : 0 ≤ y ∧ y < 10) 
  (h3 : 0 ≤ z ∧ z < 10) 
  (h4 : x + z = y) : 
  (100 * x + 10 * y + z) % 11 = 0 := 
by 
  sorry

end three_digit_divisible_by_11_l783_78353


namespace b_2056_l783_78398

noncomputable def b (n : ℕ) : ℝ := sorry

-- Conditions
axiom h1 : b 1 = 2 + Real.sqrt 8
axiom h2 : b 2023 = 15 + Real.sqrt 8
axiom recurrence : ∀ n, n ≥ 2 → b n = b (n - 1) * b (n + 1)

-- Problem statement to prove
theorem b_2056 : b 2056 = (2 + Real.sqrt 8)^2 / (15 + Real.sqrt 8) :=
sorry

end b_2056_l783_78398


namespace bricks_in_chimney_900_l783_78385

theorem bricks_in_chimney_900 (h : ℕ) :
  let Brenda_rate := h / 9
  let Brandon_rate := h / 10
  let combined_rate := (Brenda_rate + Brandon_rate) - 10
  5 * combined_rate = h → h = 900 :=
by
  intros Brenda_rate Brandon_rate combined_rate
  sorry

end bricks_in_chimney_900_l783_78385


namespace intersection_of_sets_l783_78306

def setM : Set ℝ := { x | x^2 - 3 * x - 4 ≤ 0 }
def setN : Set ℝ := { x | Real.log x ≥ 0 }

theorem intersection_of_sets : (setM ∩ setN) = { x | 1 ≤ x ∧ x ≤ 4 } := 
by {
  sorry
}

end intersection_of_sets_l783_78306


namespace convert_base_5_to_decimal_l783_78389

-- Define the base-5 number 44 and its decimal equivalent
def base_5_number : ℕ := 4 * 5^1 + 4 * 5^0

-- Prove that the base-5 number 44 equals 24 in decimal
theorem convert_base_5_to_decimal : base_5_number = 24 := by
  sorry

end convert_base_5_to_decimal_l783_78389


namespace percent_increase_in_area_l783_78324

theorem percent_increase_in_area (s : ℝ) (h_s : s > 0) :
  let medium_area := s^2
  let large_length := 1.20 * s
  let large_width := 1.25 * s
  let large_area := large_length * large_width 
  let percent_increase := ((large_area - medium_area) / medium_area) * 100
  percent_increase = 50 := by
    sorry

end percent_increase_in_area_l783_78324


namespace slope_of_AB_l783_78312

theorem slope_of_AB (A B : (ℕ × ℕ)) (hA : A = (3, 4)) (hB : B = (2, 3)) : 
  (B.2 - A.2) / (B.1 - A.1) = 1 := 
by 
  sorry

end slope_of_AB_l783_78312


namespace find_radius_l783_78371

-- Define the given conditions as variables
variables (l A r : ℝ)

-- Conditions from the problem
-- 1. The arc length of the sector is 2 cm
def arc_length_eq : Prop := l = 2

-- 2. The area of the sector is 2 cm²
def area_eq : Prop := A = 2

-- Formula for the area of the sector
def sector_area (l r : ℝ) : ℝ := 0.5 * l * r

-- Define the goal to prove the radius is 2 cm
theorem find_radius (h₁ : arc_length_eq l) (h₂ : area_eq A) : r = 2 :=
by {
  sorry -- proof omitted
}

end find_radius_l783_78371


namespace minimum_n_minus_m_l783_78384

noncomputable def f (x : Real) : Real :=
    (Real.sin x) * (Real.sin (x + Real.pi / 3)) - 1 / 4

theorem minimum_n_minus_m (m n : Real) (h : m < n) 
  (h_domain : ∀ x, m ≤ x ∧ x ≤ n → -1 / 2 ≤ f x ∧ f x ≤ 1 / 4) :
  n - m = 2 * Real.pi / 3 :=
by
  sorry

end minimum_n_minus_m_l783_78384


namespace fraction_sum_l783_78349

theorem fraction_sum :
  (3 / 30 : ℝ) + (5 / 300) + (7 / 3000) = 0.119 := by
  sorry

end fraction_sum_l783_78349


namespace solve_for_x_l783_78323

theorem solve_for_x (x : ℝ) : (x - 55) / 3 = (2 - 3*x + x^2) / 4 → (x = 20 / 3 ∨ x = -11) :=
by
  intro h
  sorry

end solve_for_x_l783_78323


namespace sasha_remainder_is_20_l783_78394

theorem sasha_remainder_is_20 (n a b c d : ℤ) (h1 : n = 102 * a + b) (h2 : n = 103 * c + d) (h3 : d = 20 - a) : b = 20 :=
by
  sorry

end sasha_remainder_is_20_l783_78394


namespace find_x_positive_integers_l783_78360

theorem find_x_positive_integers (a b c x : ℕ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a + b + c = x * a * b * c) → (x = 1 ∧ a = 1 ∧ b = 2 ∧ c = 3) ∨
  (x = 2 ∧ a = 1 ∧ b = 1 ∧ c = 2) ∨
  (x = 3 ∧ a = 1 ∧ b = 1 ∧ c = 1) :=
  sorry

end find_x_positive_integers_l783_78360
