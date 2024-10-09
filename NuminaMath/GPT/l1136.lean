import Mathlib

namespace hyperbola_sufficiency_l1136_113673

open Real

theorem hyperbola_sufficiency (k : ℝ) : 
  (9 - k < 0 ∧ k - 4 > 0) → 
  (∃ x y : ℝ, (x^2) / (9 - k) + (y^2) / (k - 4) = 1) :=
by
  intro hk
  sorry

end hyperbola_sufficiency_l1136_113673


namespace sqrt_four_l1136_113666

theorem sqrt_four : {x : ℝ | x ^ 2 = 4} = {-2, 2} := by
  sorry

end sqrt_four_l1136_113666


namespace zigzag_lines_divide_regions_l1136_113607

-- Define the number of regions created by n zigzag lines
def regions (n : ℕ) : ℕ := (2 * n * (2 * n + 1)) / 2 + 1 - 2 * n

-- Main theorem
theorem zigzag_lines_divide_regions (n : ℕ) : ∃ k : ℕ, k = regions n := by
  sorry

end zigzag_lines_divide_regions_l1136_113607


namespace determine_a_l1136_113619

theorem determine_a (a x : ℝ) (h : x = 1) (h_eq : a * x + 2 * x = 3) : a = 1 :=
by
  subst h
  simp at h_eq
  linarith

end determine_a_l1136_113619


namespace cos_expression_range_l1136_113634

theorem cos_expression_range (A B C : ℝ) (hA : 0 < A) (hB : 0 < B) (hC : 0 < C) (hSum : A + B + C = Real.pi) :
  -25 / 16 < 3 * Real.cos A + 2 * Real.cos (2 * B) + Real.cos (3 * C) ∧ 3 * Real.cos A + 2 * Real.cos (2 * B) + Real.cos (3 * C) < 6 :=
sorry

end cos_expression_range_l1136_113634


namespace arithmetic_sequence_sum_l1136_113608

variable {α : Type*} [LinearOrderedField α]

noncomputable def S (n a_1 d : α) : α :=
  (n / 2) * (2 * a_1 + (n - 1) * d)

theorem arithmetic_sequence_sum (a_1 d : α) :
  S 5 a_1 d = 5 → S 9 a_1 d = 27 → S 7 a_1 d = 14 :=
by
  sorry

end arithmetic_sequence_sum_l1136_113608


namespace negation_proposition_l1136_113604

-- Definitions based on the conditions
def original_proposition : Prop := ∃ x : ℝ, x^2 + 3*x + 2 < 0

-- Theorem requiring proof
theorem negation_proposition : (¬ original_proposition) = ∀ x : ℝ, x^2 + 3*x + 2 ≥ 0 :=
by
  sorry

end negation_proposition_l1136_113604


namespace difference_two_numbers_l1136_113623

theorem difference_two_numbers (a b : ℕ) (h₁ : a + b = 20250) (h₂ : b % 15 = 0) (h₃ : a = b / 3) : b - a = 10130 :=
by 
  sorry

end difference_two_numbers_l1136_113623


namespace area_of_square_l1136_113653

-- Define the problem setting and the conditions
def square (side_length : ℝ) : Prop :=
  ∃ (width height : ℝ), width * height = side_length^2
    ∧ width = 5
    ∧ side_length / height = 5 / height

-- State the theorem to be proven
theorem area_of_square (side_length : ℝ) (width height : ℝ) (h1 : width = 5) (h2: side_length = 5 + 2 * height): 
  square side_length → side_length^2 = 400 :=
by
  intro h
  sorry

end area_of_square_l1136_113653


namespace possible_sums_of_products_neg11_l1136_113601

theorem possible_sums_of_products_neg11 (a b c : ℤ) (h : a * b * c = -11) :
  a + b + c = -9 ∨ a + b + c = 11 ∨ a + b + c = 13 :=
sorry

end possible_sums_of_products_neg11_l1136_113601


namespace intersect_count_l1136_113670

noncomputable def f (x : ℝ) : ℝ := sorry  -- Function f defined for all real x.
noncomputable def f_inv (y : ℝ) : ℝ := sorry  -- Inverse function of f.

theorem intersect_count : 
  (∃ a b : ℝ, a ≠ b ∧ f (a^2) = f (a^3) ∧ f (b^2) = f (b^3)) :=
by sorry

end intersect_count_l1136_113670


namespace inequalities_not_all_hold_l1136_113640

theorem inequalities_not_all_hold (a b c d : ℝ) (h₀ : 0 < a) (h₁ : 0 < b) (h₂ : 0 < c) (h₃ : 0 < d) :
    ¬ (a + b < c + d ∧ (a + b) * (c + d) < a * b + c * d ∧ (a + b) * c * d < a * b * (c + d)) :=
by
  sorry

end inequalities_not_all_hold_l1136_113640


namespace relationship_y1_y2_y3_l1136_113676

theorem relationship_y1_y2_y3 
  (y_1 y_2 y_3 : ℝ)
  (h1 : y_1 = (-2)^2 + 2*(-2) + 2)
  (h2 : y_2 = (-1)^2 + 2*(-1) + 2)
  (h3 : y_3 = 2^2 + 2*2 + 2) :
  y_2 < y_1 ∧ y_1 < y_3 := 
sorry

end relationship_y1_y2_y3_l1136_113676


namespace value_of_six_inch_cube_l1136_113609

theorem value_of_six_inch_cube :
  let four_inch_cube_value := 400
  let four_inch_side_length := 4
  let six_inch_side_length := 6
  let volume (s : ℕ) : ℕ := s ^ 3
  (volume six_inch_side_length / volume four_inch_side_length) * four_inch_cube_value = 1350 := by
sorry

end value_of_six_inch_cube_l1136_113609


namespace sum_fraction_series_eq_l1136_113696

noncomputable def sum_fraction_series : ℝ :=
  ∑' n, (1 / (n * (n + 3)))

theorem sum_fraction_series_eq :
  sum_fraction_series = 11 / 18 :=
sorry

end sum_fraction_series_eq_l1136_113696


namespace photos_difference_is_120_l1136_113615

theorem photos_difference_is_120 (initial_photos : ℕ) (final_photos : ℕ) (first_day_factor : ℕ) (first_day_photos : ℕ) (second_day_photos : ℕ) : 
  initial_photos = 400 → 
  final_photos = 920 → 
  first_day_factor = 2 →
  first_day_photos = initial_photos / first_day_factor →
  final_photos = initial_photos + first_day_photos + second_day_photos →
  second_day_photos - first_day_photos = 120 :=
by
  intros h1 h2 h3 h4 h5
  sorry

end photos_difference_is_120_l1136_113615


namespace expand_product_polynomials_l1136_113602

noncomputable def poly1 : Polynomial ℤ := 5 * Polynomial.X + 3
noncomputable def poly2 : Polynomial ℤ := 7 * Polynomial.X^2 + 2 * Polynomial.X + 4
noncomputable def expanded_form : Polynomial ℤ := 35 * Polynomial.X^3 + 31 * Polynomial.X^2 + 26 * Polynomial.X + 12

theorem expand_product_polynomials :
  poly1 * poly2 = expanded_form := 
by
  sorry

end expand_product_polynomials_l1136_113602


namespace fourth_buoy_distance_with_current_l1136_113677

-- Define the initial conditions
def first_buoy_distance : ℕ := 20
def second_buoy_additional_distance : ℕ := 24
def third_buoy_additional_distance : ℕ := 28
def common_difference_increment : ℕ := 4
def ocean_current_push_per_segment : ℕ := 3
def number_of_segments : ℕ := 3

-- Define the mathematical proof problem
theorem fourth_buoy_distance_with_current :
  let fourth_buoy_additional_distance := third_buoy_additional_distance + common_difference_increment
  let first_to_second_buoy := first_buoy_distance + second_buoy_additional_distance
  let second_to_third_buoy := first_to_second_buoy + third_buoy_additional_distance
  let distance_before_current := second_to_third_buoy + fourth_buoy_additional_distance
  let total_current_push := ocean_current_push_per_segment * number_of_segments
  let final_distance := distance_before_current - total_current_push
  final_distance = 95 := by
  sorry

end fourth_buoy_distance_with_current_l1136_113677


namespace domain_ln_x_plus_one_l1136_113662

theorem domain_ln_x_plus_one : 
  { x : ℝ | ∃ y : ℝ, y = x + 1 ∧ y > 0 } = { x : ℝ | x > -1 } :=
by
  sorry

end domain_ln_x_plus_one_l1136_113662


namespace div_by_11_l1136_113648

theorem div_by_11 (x y : ℤ) (k : ℤ) (h : 14 * x + 13 * y = 11 * k) : 11 ∣ (19 * x + 9 * y) :=
by
  sorry

end div_by_11_l1136_113648


namespace part1_part2_l1136_113698

open Set

def A : Set ℝ := {x | x^2 - 8 * x + 15 = 0}
def B (a : ℝ) : Set ℝ := {x | a * x = 1}

theorem part1 : B (1/5) ⊆ A ∧ ¬ A ⊆ B (1/5) := by
  sorry
  
theorem part2 (a : ℝ) : (B a ⊆ A) ↔ a ∈ ({0, 1/3, 1/5} : Set ℝ) := by
  sorry

end part1_part2_l1136_113698


namespace negation_universal_to_particular_l1136_113624

theorem negation_universal_to_particular :
  (¬ ∀ x : ℝ, x^2 ≥ 0) ↔ ∃ x : ℝ, x^2 < 0 :=
by
  sorry

end negation_universal_to_particular_l1136_113624


namespace negation_proof_l1136_113685

theorem negation_proof (x : ℝ) : ¬ (∃ x : ℝ, x^2 + 2*x + 2 ≤ 0) ↔ ∀ x : ℝ, x^2 + 2*x + 2 > 0 :=
by
  sorry

end negation_proof_l1136_113685


namespace no_three_digit_number_l1136_113682

theorem no_three_digit_number (N : ℕ) : 
  (100 ≤ N ∧ N < 1000 ∧ 
   (∀ k, k ∈ [1,2,3] → 5 < (N / 10^(k - 1) % 10)) ∧ 
   (N % 6 = 0) ∧ (N % 5 = 0)) → 
  false :=
by
sorry

end no_three_digit_number_l1136_113682


namespace danny_age_l1136_113647

theorem danny_age (D : ℕ) (h : D - 19 = 3 * (26 - 19)) : D = 40 := by
  sorry

end danny_age_l1136_113647


namespace solution_set_is_circle_with_exclusion_l1136_113616

noncomputable 
def system_solutions_set (x y : ℝ) : Prop :=
  ∃ a : ℝ, (a * x + y = 2 * a + 3) ∧ (x - a * y = a + 4)

noncomputable 
def solution_circle (x y : ℝ) : Prop :=
  (x - 3)^2 + (y - 1)^2 = 5

theorem solution_set_is_circle_with_exclusion :
  ∀ (x y : ℝ), (system_solutions_set x y ↔ solution_circle x y) ∧ 
  ¬(x = 2 ∧ y = -1) :=
by
  sorry

end solution_set_is_circle_with_exclusion_l1136_113616


namespace math_proof_problem_l1136_113637

theorem math_proof_problem (a : ℝ) : 
  (a^8 / a^4 ≠ a^4) ∧ ((a^2)^3 ≠ a^6) ∧ ((3*a)^3 ≠ 9*a^3) ∧ ((-a)^3 * (-a)^5 = a^8) := 
by 
  sorry

end math_proof_problem_l1136_113637


namespace friends_Sarah_brought_l1136_113659

def total_people_in_house : Nat := 15
def in_bedroom : Nat := 2
def living_room : Nat := 8
def Sarah : Nat := 1

theorem friends_Sarah_brought :
  total_people_in_house - (in_bedroom + Sarah + living_room) = 4 := by
  sorry

end friends_Sarah_brought_l1136_113659


namespace cryptarithmetic_puzzle_sol_l1136_113645

theorem cryptarithmetic_puzzle_sol (A B C D : ℕ) 
  (h1 : A + B + C = D) 
  (h2 : B + C = 7) 
  (h3 : A - B = 1) : D = 9 := 
by 
  sorry

end cryptarithmetic_puzzle_sol_l1136_113645


namespace find_m_l1136_113629

theorem find_m (m : ℝ) (h : ∀ x : ℝ, x - m > 5 ↔ x > 2) : m = -3 := by
  sorry

end find_m_l1136_113629


namespace polynomial_coefficients_l1136_113611

theorem polynomial_coefficients (x a₄ a₃ a₂ a₁ a₀ : ℝ) (h : (x - 1)^4 = a₄ * x^4 + a₃ * x^3 + a₂ * x^2 + a₁ * x + a₀)
  : a₄ - a₃ + a₂ - a₁ = 15 := by
  sorry

end polynomial_coefficients_l1136_113611


namespace find_hours_hired_l1136_113664

def hourly_rate : ℝ := 15
def tip_rate : ℝ := 0.20
def total_paid : ℝ := 54

theorem find_hours_hired (h : ℝ) : 15 * h + 0.20 * 15 * h = 54 → h = 3 :=
by
  sorry

end find_hours_hired_l1136_113664


namespace find_n_l1136_113688

-- Define the function to sum the digits of a natural number n
def digit_sum (n : ℕ) : ℕ := 
  -- This is a dummy implementation for now
  -- Normally, we would implement the sum of the digits of n
  sorry 

-- The main theorem that we want to prove
theorem find_n : ∃ (n : ℕ), digit_sum n + n = 2011 ∧ n = 1991 :=
by
  -- Proof steps would go here, but we're skipping those with sorry.
  sorry

end find_n_l1136_113688


namespace quadratic_function_through_point_l1136_113644

theorem quadratic_function_through_point : 
  (∃ (a : ℝ), ∀ (x y : ℝ), y = a * x ^ 2 ∧ ((x, y) = (-1, 4)) → y = 4 * x ^ 2) :=
sorry

end quadratic_function_through_point_l1136_113644


namespace molecular_weight_one_mole_l1136_113684

theorem molecular_weight_one_mole
  (molecular_weight_7_moles : ℝ)
  (mole_count : ℝ)
  (h : molecular_weight_7_moles = 126)
  (k : mole_count = 7)
  : molecular_weight_7_moles / mole_count = 18 := 
sorry

end molecular_weight_one_mole_l1136_113684


namespace largest_three_digit_number_l1136_113642

def divisible_by_each_digit (n : ℕ) : Prop :=
  let digits := [n / 100, (n / 10) % 10, n % 10]
  ∀ d ∈ digits, d ≠ 0 ∧ n % d = 0

def sum_of_digits_divisible_by (n : ℕ) (k : ℕ) : Prop :=
  let sum := (n / 100) + ((n / 10) % 10) + (n % 10)
  sum % k = 0

theorem largest_three_digit_number : ∃ n : ℕ, n = 936 ∧
  n >= 100 ∧ n < 1000 ∧
  divisible_by_each_digit n ∧
  sum_of_digits_divisible_by n 6 :=
by
  -- Proof details are omitted
  sorry

end largest_three_digit_number_l1136_113642


namespace parabola_focus_coordinates_l1136_113617

theorem parabola_focus_coordinates (a : ℝ) (h : a ≠ 0) :
  ∃ x y : ℝ, y = 4 * a * x^2 → (x, y) = (0, 1 / (16 * a)) :=
by
  sorry

end parabola_focus_coordinates_l1136_113617


namespace polynomial_divisibility_l1136_113618

theorem polynomial_divisibility (C D : ℝ)
  (h : ∀ x, x^2 + x + 1 = 0 → x^102 + C * x + D = 0) :
  C + D = -1 := 
by 
  sorry

end polynomial_divisibility_l1136_113618


namespace height_of_frustum_l1136_113639

-- Definitions based on the given conditions
def cuts_parallel_to_base (height: ℕ) (ratio: ℕ) : ℕ := 
  height * ratio

-- Define the problem
theorem height_of_frustum 
  (height_smaller_pyramid : ℕ) 
  (ratio_upper_to_lower: ℕ) 
  (h : height_smaller_pyramid = 3) 
  (r : ratio_upper_to_lower = 4) 
  : (cuts_parallel_to_base 3 2) - height_smaller_pyramid = 3 := 
by
  sorry

end height_of_frustum_l1136_113639


namespace husk_estimation_l1136_113635

-- Define the conditions: total rice, sample size, and number of husks in the sample
def total_rice : ℕ := 1520
def sample_size : ℕ := 144
def husks_in_sample : ℕ := 18

-- Define the expected amount of husks in the total batch of rice
def expected_husks : ℕ := 190

-- The theorem stating the problem
theorem husk_estimation 
  (h : (husks_in_sample / sample_size) * total_rice = expected_husks) :
  (18 / 144) * 1520 = 190 := 
sorry

end husk_estimation_l1136_113635


namespace cost_comparison_for_30_pens_l1136_113687

def cost_store_a (x : ℕ) : ℝ :=
  if x > 10 then 0.9 * x + 6
  else 1.5 * x

def cost_store_b (x : ℕ) : ℝ :=
  1.2 * x

theorem cost_comparison_for_30_pens :
  cost_store_a 30 < cost_store_b 30 :=
by
  have store_a_cost : cost_store_a 30 = 0.9 * 30 + 6 := by rfl
  have store_b_cost : cost_store_b 30 = 1.2 * 30 := by rfl
  rw [store_a_cost, store_b_cost]
  sorry

end cost_comparison_for_30_pens_l1136_113687


namespace solve_eq1_solve_eq2_l1136_113668

theorem solve_eq1 (x : ℤ) : x - 2 * (5 + x) = -4 → x = -6 := by
  sorry

theorem solve_eq2 (x : ℤ) : (2 * x - 1) / 2 = 1 - (3 - x) / 4 → x = 1 := by
  sorry

end solve_eq1_solve_eq2_l1136_113668


namespace red_balls_in_box_l1136_113694

theorem red_balls_in_box {n : ℕ} (h : n = 6) (p : (∃ (r : ℕ), r / 6 = 1 / 3)) : ∃ r, r = 2 :=
by
  sorry

end red_balls_in_box_l1136_113694


namespace problem_solution_correct_l1136_113661

def proposition_p : Prop :=
  ∃ x : ℝ, Real.tan x = 1

def proposition_q : Prop :=
  {x : ℝ | x^2 - 3 * x + 2 < 0} = {x : ℝ | 1 < x ∧ x < 2}

theorem problem_solution_correct :
  (proposition_p ∧ proposition_q) ∧
  (proposition_p ∧ ¬proposition_q) = false ∧
  (¬proposition_p ∨ proposition_q) ∧
  (¬proposition_p ∨ ¬proposition_q) = false :=
by
  sorry

end problem_solution_correct_l1136_113661


namespace only_n_eq_1_divides_2_pow_n_minus_1_l1136_113660

theorem only_n_eq_1_divides_2_pow_n_minus_1 (n : ℕ) (h1 : 1 ≤ n) (h2 : n ∣ 2^n - 1) : n = 1 :=
sorry

end only_n_eq_1_divides_2_pow_n_minus_1_l1136_113660


namespace find_DF_l1136_113650

noncomputable def triangle (a b c : ℝ) : Prop :=
a + b > c ∧ b + c > a ∧ c + a > b

noncomputable def median (a b : ℝ) : ℝ := a / 2

theorem find_DF {DE EF DM DF : ℝ} (h1 : DE = 7) (h2 : EF = 10) (h3 : DM = 5) :
  DF = Real.sqrt 51 :=
by
  sorry

end find_DF_l1136_113650


namespace mary_visited_two_shops_l1136_113689

-- Define the costs of items
def cost_shirt : ℝ := 13.04
def cost_jacket : ℝ := 12.27
def total_cost : ℝ := 25.31

-- Define the number of shops visited
def number_of_shops : ℕ := 2

-- Proof that Mary visited 2 shops given the conditions
theorem mary_visited_two_shops (h_shirt : cost_shirt = 13.04) (h_jacket : cost_jacket = 12.27) (h_total : cost_shirt + cost_jacket = total_cost) : number_of_shops = 2 :=
by
  sorry

end mary_visited_two_shops_l1136_113689


namespace terry_tomato_types_l1136_113657

theorem terry_tomato_types (T : ℕ) (h1 : 2 * T * 4 * 2 = 48) : T = 3 :=
by
  -- Proof goes here
  sorry

end terry_tomato_types_l1136_113657


namespace money_allocation_l1136_113632

theorem money_allocation (x y : ℝ) (h1 : x + 1/2 * y = 50) (h2 : y + 2/3 * x = 50) : 
  x + 1/2 * y = 50 ∧ y + 2/3 * x = 50 :=
by
  exact ⟨h1, h2⟩

end money_allocation_l1136_113632


namespace student_attempted_sums_l1136_113626

theorem student_attempted_sums (right wrong : ℕ) (h1 : wrong = 2 * right) (h2 : right = 12) : right + wrong = 36 := sorry

end student_attempted_sums_l1136_113626


namespace extrema_of_function_l1136_113656

noncomputable def f (x : ℝ) := x / 8 + 2 / x

theorem extrema_of_function : 
  ∀ x ∈ Set.Ioo (-5 : ℝ) (10),
  (x ≠ 0) →
  (f (-4) = -1 ∧ f 4 = 1) ∧
  (∀ x ∈ Set.Ioc (-5) 0, f x ≤ -1) ∧
  (∀ x ∈ Set.Ioo 0 10, f x ≥ 1) := by
  sorry

end extrema_of_function_l1136_113656


namespace employees_females_l1136_113630

theorem employees_females
  (total_employees : ℕ)
  (adv_deg_employees : ℕ)
  (coll_deg_employees : ℕ)
  (males_coll_deg : ℕ)
  (females_adv_deg : ℕ)
  (females_coll_deg : ℕ)
  (h1 : total_employees = 180)
  (h2 : adv_deg_employees = 90)
  (h3 : coll_deg_employees = 180 - 90)
  (h4 : males_coll_deg = 35)
  (h5 : females_adv_deg = 55)
  (h6 : females_coll_deg = 90 - 35) :
  females_coll_deg + females_adv_deg = 110 :=
by
  sorry

end employees_females_l1136_113630


namespace locus_of_C_l1136_113600

variables (a b : ℝ)
variables (h1 : a > 0) (h2 : b > 0)

theorem locus_of_C :
  ∀ (C : ℝ × ℝ), (C.2 = (b / a) * C.1 ∧ (a * b / Real.sqrt (a ^ 2 + b ^ 2) ≤ C.1) ∧ (C.1 ≤ a)) :=
sorry

end locus_of_C_l1136_113600


namespace required_decrease_l1136_113628

noncomputable def price_after_increases (P : ℝ) : ℝ :=
  let P1 := 1.20 * P
  let P2 := 1.10 * P1
  1.15 * P2

noncomputable def price_after_discount (P : ℝ) : ℝ :=
  0.95 * price_after_increases P

noncomputable def price_after_tax (P : ℝ) : ℝ :=
  1.07 * price_after_discount P

theorem required_decrease (P : ℝ) (D : ℝ) : 
  (1 - D / 100) * price_after_tax P = P ↔ D = 35.1852 :=
by
  sorry

end required_decrease_l1136_113628


namespace train_length_l1136_113620

theorem train_length 
  (t1 t2 : ℝ)
  (d2 : ℝ)
  (L : ℝ)
  (V : ℝ)
  (h1 : t1 = 18)
  (h2 : t2 = 27)
  (h3 : d2 = 150.00000000000006)
  (h4 : V = L / t1)
  (h5 : V = (L + d2) / t2) :
  L = 300.0000000000001 :=
by
  sorry

end train_length_l1136_113620


namespace total_balls_donated_l1136_113625

def num_elem_classes_A := 4
def num_middle_classes_A := 5
def num_elem_classes_B := 5
def num_middle_classes_B := 3
def num_elem_classes_C := 6
def num_middle_classes_C := 4
def balls_per_class := 5

theorem total_balls_donated :
  (num_elem_classes_A + num_middle_classes_A) * balls_per_class +
  (num_elem_classes_B + num_middle_classes_B) * balls_per_class +
  (num_elem_classes_C + num_middle_classes_C) * balls_per_class =
  135 :=
by
  sorry

end total_balls_donated_l1136_113625


namespace animal_shelter_cats_l1136_113643

theorem animal_shelter_cats (D C x : ℕ) (h1 : 15 * C = 7 * D) (h2 : 15 * (C + x) = 11 * D) (h3 : D = 60) : x = 16 :=
by
  sorry

end animal_shelter_cats_l1136_113643


namespace complex_number_is_3i_quadratic_equation_roots_l1136_113679

open Complex

-- Given complex number z satisfies 2z + |z| = 3 + 6i
-- We need to prove that z = 3i
theorem complex_number_is_3i (z : ℂ) (h : 2 * z + abs z = 3 + 6 * I) : z = 3 * I :=
sorry

-- Given that z = 3i is a root of the quadratic equation with real coefficients
-- Prove that b - c = -9
theorem quadratic_equation_roots (b c : ℝ) (h1 : 3 * I + -3 * I = -b)
  (h2 : 3 * I * -3 * I = c) : b - c = -9 :=
sorry

end complex_number_is_3i_quadratic_equation_roots_l1136_113679


namespace determine_N_l1136_113699

theorem determine_N (N : ℕ) :
    995 + 997 + 999 + 1001 + 1003 = 5005 - N → N = 5 := 
by 
  sorry

end determine_N_l1136_113699


namespace area_of_rhombus_l1136_113612

theorem area_of_rhombus (d1 d2 : ℝ) (h1 : d1 = 22) (h2 : d2 = 30) : 
  (d1 * d2) / 2 = 330 :=
by
  rw [h1, h2]
  norm_num

-- Here we state the theorem about the area of the rhombus given its diagonal lengths.

end area_of_rhombus_l1136_113612


namespace balloon_ratio_l1136_113674

theorem balloon_ratio 
  (initial_blue : ℕ) (initial_purple : ℕ) (balloons_left : ℕ)
  (h1 : initial_blue = 303)
  (h2 : initial_purple = 453)
  (h3 : balloons_left = 378) :
  (balloons_left / (initial_blue + initial_purple) : ℚ) = (1 / 2 : ℚ) :=
by
  sorry

end balloon_ratio_l1136_113674


namespace rectangle_area_in_cm_l1136_113671

theorem rectangle_area_in_cm (length_in_m : ℝ) (width_in_m : ℝ) 
  (h_length : length_in_m = 0.5) (h_width : width_in_m = 0.36) : 
  (100 * length_in_m) * (100 * width_in_m) = 1800 :=
by
  -- We skip the proof for now
  sorry

end rectangle_area_in_cm_l1136_113671


namespace prove_inequality_l1136_113652

variable (x y z : ℝ)
variable (h₁ : x > 0)
variable (h₂ : y > 0)
variable (h₃ : z > 0)
variable (h₄ : x + y + z = 1)

theorem prove_inequality :
  (3 * x^2 - x) / (1 + x^2) +
  (3 * y^2 - y) / (1 + y^2) +
  (3 * z^2 - z) / (1 + z^2) ≥ 0 :=
by
  sorry

end prove_inequality_l1136_113652


namespace triangle_formation_segments_l1136_113691

theorem triangle_formation_segments (a b c : ℝ) (h_sum : a + b + c = 1) (h_a : a < 1/2) (h_b : b < 1/2) (h_c : c < 1/2) : 
  (a + b > c) ∧ (a + c > b) ∧ (b + c > a) := 
by
  sorry

end triangle_formation_segments_l1136_113691


namespace proof_equilateral_inscribed_circle_l1136_113683

variables {A B C : Type*}
variables (r : ℝ) (D : ℝ)

def is_equilateral_triangle (A B C : Type*) : Prop := 
  -- Define the equilateral condition, where all sides are equal
  true

def is_inscribed_circle_radius (D r : ℝ) : Prop := 
  -- Define the property that D is the center and r is the radius 
  true

def distance_center_to_vertex (D r x : ℝ) : Prop := 
  x = 3 * r

theorem proof_equilateral_inscribed_circle 
  (A B C : Type*) 
  (r D : ℝ) 
  (h1 : is_equilateral_triangle A B C) 
  (h2 : is_inscribed_circle_radius D r) : 
  distance_center_to_vertex D r (1 / 16) :=
by sorry

end proof_equilateral_inscribed_circle_l1136_113683


namespace problem1_problem2_problem3_l1136_113605

theorem problem1 : 128 + 52 / 13 = 132 :=
by
  sorry

theorem problem2 : 132 / 11 * 29 - 178 = 170 :=
by
  sorry

theorem problem3 : 45 * (320 / (4 * 5)) = 720 :=
by
  sorry

end problem1_problem2_problem3_l1136_113605


namespace washed_shirts_l1136_113613

-- Definitions based on the conditions
def short_sleeve_shirts : ℕ := 39
def long_sleeve_shirts : ℕ := 47
def unwashed_shirts : ℕ := 66

-- The total number of shirts is the sum of short and long sleeve shirts
def total_shirts : ℕ := short_sleeve_shirts + long_sleeve_shirts

-- The problem to prove that Oliver washed 20 shirts
theorem washed_shirts :
  total_shirts - unwashed_shirts = 20 := 
sorry

end washed_shirts_l1136_113613


namespace arithmetic_progression_12th_term_l1136_113658

theorem arithmetic_progression_12th_term (a d n : ℤ) (h_a : a = 2) (h_d : d = 8) (h_n : n = 12) :
  a + (n - 1) * d = 90 :=
by
  -- sorry is a placeholder for the actual proof
  sorry

end arithmetic_progression_12th_term_l1136_113658


namespace cole_trip_time_l1136_113697

theorem cole_trip_time 
  (D : ℕ) -- The distance D from home to work
  (T_total : ℕ) -- The total round trip time in hours
  (S1 S2 : ℕ) -- The average speeds (S1, S2) in km/h
  (h1 : S1 = 80) -- The average speed from home to work
  (h2 : S2 = 120) -- The average speed from work to home
  (h3 : T_total = 2) -- The total round trip time is 2 hours
  : (D : ℝ) / 80 + (D : ℝ) / 120 = 2 →
    (T_work : ℝ) = (D : ℝ) / 80 →
    (T_work * 60) = 72 := 
by {
  sorry
}

end cole_trip_time_l1136_113697


namespace cake_pieces_kept_l1136_113690

theorem cake_pieces_kept (total_pieces : ℕ) (two_fifths_eaten : ℕ) (extra_pieces_eaten : ℕ)
  (h1 : total_pieces = 35)
  (h2 : two_fifths_eaten = 2 * total_pieces / 5)
  (h3 : extra_pieces_eaten = 3)
  (correct_answer : ℕ)
  (h4 : correct_answer = total_pieces - (two_fifths_eaten + extra_pieces_eaten)) :
  correct_answer = 18 := by
  sorry

end cake_pieces_kept_l1136_113690


namespace alpha_beta_sum_two_l1136_113695

theorem alpha_beta_sum_two (α β : ℝ) 
  (hα : α^3 - 3 * α^2 + 5 * α - 17 = 0)
  (hβ : β^3 - 3 * β^2 + 5 * β + 11 = 0) : 
  α + β = 2 :=
by
  sorry

end alpha_beta_sum_two_l1136_113695


namespace describe_random_event_l1136_113614

def idiom_A : Prop := "海枯石烂" = "extremely improbable or far into the future, not random"
def idiom_B : Prop := "守株待兔" = "represents a random event"
def idiom_C : Prop := "画饼充饥" = "unreal hopes, not random"
def idiom_D : Prop := "瓜熟蒂落" = "natural or expected outcome, not random"

theorem describe_random_event : idiom_B := 
by
  -- Proof omitted; conclusion follows from the given definitions
  sorry

end describe_random_event_l1136_113614


namespace hexagon_side_length_l1136_113646

theorem hexagon_side_length (p : ℕ) (s : ℕ) (h₁ : p = 24) (h₂ : s = 6) : p / s = 4 := by
  sorry

end hexagon_side_length_l1136_113646


namespace find_interest_rate_l1136_113651

-- Given conditions
def P : ℝ := 4099.999999999999
def t : ℕ := 2
def CI_minus_SI : ℝ := 41

-- Formulas for Simple Interest and Compound Interest
def SI (P : ℝ) (r : ℝ) (t : ℕ) : ℝ := P * r * (t : ℝ)
def CI (P : ℝ) (r : ℝ) (t : ℕ) : ℝ := P * ((1 + r) ^ t) - P

-- Main theorem to prove: the interest rate r is 0.1 (i.e., 10%)
theorem find_interest_rate (r : ℝ) : 
  (CI P r t - SI P r t = CI_minus_SI) → r = 0.1 :=
by
  sorry

end find_interest_rate_l1136_113651


namespace arithmetic_sequence_mod_l1136_113655

theorem arithmetic_sequence_mod :
  let a := 2
  let d := 5
  let l := 137
  let n := (l - a) / d + 1
  let S := n * (2 * a + (n - 1) * d) / 2
  n = 28 ∧ S = 1946 →
  S % 20 = 6 :=
by
  intros h
  sorry

end arithmetic_sequence_mod_l1136_113655


namespace find_fourth_number_l1136_113686

theorem find_fourth_number (a : ℕ → ℕ) (h1 : a 7 = 42) (h2 : a 9 = 110)
  (h3 : ∀ n, n ≥ 3 → a n = a (n - 1) + a (n - 2)) : a 4 = 10 := 
sorry

end find_fourth_number_l1136_113686


namespace pauly_omelets_l1136_113649

/-- Pauly is making omelets for his family. There are three dozen eggs, and he plans to use them all. 
Each omelet requires 4 eggs. Including himself, there are 3 people. 
Prove that each person will get 3 omelets. -/

def total_eggs := 3 * 12

def eggs_per_omelet := 4

def total_omelets := total_eggs / eggs_per_omelet

def number_of_people := 3

def omelets_per_person := total_omelets / number_of_people

theorem pauly_omelets : omelets_per_person = 3 :=
by
  -- Placeholder proof
  sorry

end pauly_omelets_l1136_113649


namespace meeting_attendance_l1136_113654

theorem meeting_attendance (A B : ℕ) (h1 : 2 * A + B = 7) (h2 : A + 2 * B = 11) : A + B = 6 :=
sorry

end meeting_attendance_l1136_113654


namespace least_number_of_cubes_is_10_l1136_113669

noncomputable def volume_of_block (length width height : ℕ) : ℕ :=
  length * width * height

noncomputable def gcd_three_numbers (a b c : ℕ) : ℕ :=
  Nat.gcd a (Nat.gcd b c)

noncomputable def volume_of_cube (side : ℕ) : ℕ :=
  side ^ 3

noncomputable def least_number_of_cubes (length width height : ℕ) : ℕ := 
  volume_of_block length width height / volume_of_cube (gcd_three_numbers length width height)

theorem least_number_of_cubes_is_10 : least_number_of_cubes 15 30 75 = 10 := by
  sorry

end least_number_of_cubes_is_10_l1136_113669


namespace diameter_is_twice_radius_l1136_113663

theorem diameter_is_twice_radius {r d : ℝ} (h : d = 2 * r) : d = 2 * r :=
by {
  sorry
}

end diameter_is_twice_radius_l1136_113663


namespace find_inradius_of_scalene_triangle_l1136_113667

noncomputable def side_a := 32
noncomputable def side_b := 40
noncomputable def side_c := 24
noncomputable def ic := 18
noncomputable def expected_inradius := 2 * Real.sqrt 17

theorem find_inradius_of_scalene_triangle (a b c : ℝ) (h : a = side_a) (h1 : b = side_b) (h2 : c = side_c) (ic_length : ℝ) (h3: ic_length = ic) : (Real.sqrt (ic_length ^ 2 - (b - ((a + b - c) / 2)) ^ 2)) = expected_inradius :=
by
  sorry

end find_inradius_of_scalene_triangle_l1136_113667


namespace shaded_region_area_l1136_113606

theorem shaded_region_area
  (n : ℕ) (d : ℝ) 
  (h₁ : n = 25) 
  (h₂ : d = 10) 
  (h₃ : n > 0) : 
  (d^2 / n = 2) ∧ (n * (d^2 / (2 * n)) = 50) :=
by 
  sorry

end shaded_region_area_l1136_113606


namespace minimize_quadratic_expression_l1136_113681

theorem minimize_quadratic_expression : ∃ x : ℝ, (∀ y : ℝ, 3 * x^2 - 18 * x + 7 ≤ 3 * y^2 - 18 * y + 7) ∧ x = 3 :=
by
  sorry

end minimize_quadratic_expression_l1136_113681


namespace zachary_cans_first_day_l1136_113678

theorem zachary_cans_first_day :
  ∃ (first_day_cans : ℕ),
    ∃ (second_day_cans : ℕ),
      ∃ (third_day_cans : ℕ),
        ∃ (seventh_day_cans : ℕ),
          second_day_cans = 9 ∧
          third_day_cans = 14 ∧
          (∀ (n : ℕ), 2 ≤ n ∧ n < 7 → third_day_cans = second_day_cans + 5) →
          seventh_day_cans = 34 ∧
          first_day_cans = second_day_cans - 5 ∧
          first_day_cans = 4 :=

by
  sorry

end zachary_cans_first_day_l1136_113678


namespace equal_pieces_length_l1136_113641

theorem equal_pieces_length (total_length_cm : ℕ) (num_pieces : ℕ) (num_equal_pieces : ℕ) (length_remaining_piece_mm : ℕ) :
  total_length_cm = 1165 ∧ num_pieces = 154 ∧ num_equal_pieces = 150 ∧ length_remaining_piece_mm = 100 →
  (total_length_cm * 10 - (num_pieces - num_equal_pieces) * length_remaining_piece_mm) / num_equal_pieces = 75 :=
by
  sorry

end equal_pieces_length_l1136_113641


namespace solution_set_f_prime_pos_l1136_113633

noncomputable def f (x : ℝ) : ℝ := x^2 - 2*x - 4*Real.log x

theorem solution_set_f_prime_pos : 
  {x : ℝ | 0 < x ∧ (deriv f x > 0)} = {x : ℝ | 2 < x} :=
by
  sorry

end solution_set_f_prime_pos_l1136_113633


namespace arithmetic_sequence_sum_l1136_113610

theorem arithmetic_sequence_sum (a : ℕ → ℕ) (S : ℕ → ℕ) (d : ℤ) (n : ℕ)
  (h₁ : ∀ n : ℕ, a n = a 1 + (n - 1) * d)
  (h₂ : ∀ n : ℕ, S n = n * (a 1 + a n) / 2)
  (h₃ : 3 * a 5 - a 1 = 10) :
  S 13 = 117 := 
sorry

end arithmetic_sequence_sum_l1136_113610


namespace Samantha_last_name_length_l1136_113603

theorem Samantha_last_name_length :
  ∃ (S B : ℕ), S = B - 3 ∧ B - 2 = 2 * 4 ∧ S = 7 :=
by
  sorry

end Samantha_last_name_length_l1136_113603


namespace coefficient_of_expansion_l1136_113693

theorem coefficient_of_expansion (m : ℝ) (h : m^3 * (Nat.choose 6 3) = -160) : m = -2 := by
  sorry

end coefficient_of_expansion_l1136_113693


namespace value_of_f_5_l1136_113672

noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := a * x^3 - b * Real.sin x - 2

theorem value_of_f_5 (a b : ℝ) (hf : f a b (-5) = 17) : f a b 5 = -21 := by
  sorry

end value_of_f_5_l1136_113672


namespace cream_ratio_l1136_113692

noncomputable def John_creme_amount : ℚ := 3
noncomputable def Janet_initial_amount : ℚ := 8
noncomputable def Janet_creme_added : ℚ := 3
noncomputable def Janet_total_mixture : ℚ := Janet_initial_amount + Janet_creme_added
noncomputable def Janet_creme_ratio : ℚ := Janet_creme_added / Janet_total_mixture
noncomputable def Janet_drank_amount : ℚ := 3
noncomputable def Janet_drank_creme : ℚ := Janet_drank_amount * Janet_creme_ratio
noncomputable def Janet_creme_remaining : ℚ := Janet_creme_added - Janet_drank_creme

theorem cream_ratio :
  (John_creme_amount / Janet_creme_remaining) = (11 / 5) :=
by
  sorry

end cream_ratio_l1136_113692


namespace union_of_A_and_B_l1136_113680

open Set

-- Definitions for the conditions
def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {2, 3, 4}

-- Statement of the theorem
theorem union_of_A_and_B : A ∪ B = {1, 2, 3, 4} :=
by
  sorry

end union_of_A_and_B_l1136_113680


namespace compute_abs_difference_l1136_113621

theorem compute_abs_difference (x y : ℝ) 
  (h1 : ⌊x⌋ + (y - ⌊y⌋) = 3.6)
  (h2 : (x - ⌊x⌋) + ⌊y⌋ = 4.5) : 
  |x - y| = 1.1 :=
by 
  sorry

end compute_abs_difference_l1136_113621


namespace find_a_plus_d_l1136_113622

variables (a b c d e : ℝ)

theorem find_a_plus_d :
  a + b = 12 ∧ b + c = 9 ∧ c + d = 3 ∧ d + e = 7 ∧ e + a = 10 → a + d = 6 :=
by
  intros h
  have h1 : a + b = 12 := h.1
  have h2 : b + c = 9 := h.2.1
  have h3 : c + d = 3 := h.2.2.1
  have h4 : d + e = 7 := h.2.2.2.1
  have h5 : e + a = 10 := h.2.2.2.2
  sorry

end find_a_plus_d_l1136_113622


namespace max_value_of_expression_l1136_113627

theorem max_value_of_expression (A M C : ℕ) (hA : 0 < A) (hM : 0 < M) (hC : 0 < C) (hSum : A + M + C = 15) : 
  A * M * C + A * M + M * C + C * A + A + M + C ≤ 215 :=
sorry

end max_value_of_expression_l1136_113627


namespace find_integer_n_l1136_113665

noncomputable def cubic_expr_is_pure_integer (n : ℤ) : Prop :=
  (729 * n ^ 6 - 540 * n ^ 4 + 240 * n ^ 2 - 64 : ℂ).im = 0

theorem find_integer_n :
  ∃! n : ℤ, cubic_expr_is_pure_integer n := 
sorry

end find_integer_n_l1136_113665


namespace smallest_enclosing_sphere_radius_l1136_113636

noncomputable def radius_of_enclosing_sphere (r : ℝ) : ℝ :=
  let s := 6 -- side length of the cube
  let d := s * Real.sqrt 3 -- space diagonal of the cube
  (d + 2 * r) / 2

theorem smallest_enclosing_sphere_radius :
  radius_of_enclosing_sphere 2 = 3 * Real.sqrt 3 + 2 :=
by
  -- skipping the proof with sorry
  sorry

end smallest_enclosing_sphere_radius_l1136_113636


namespace soccer_balls_percentage_holes_l1136_113675

variable (x : ℕ)

theorem soccer_balls_percentage_holes 
    (h1 : ∃ x, 0 ≤ x ∧ x ≤ 100)
    (h2 : 48 = 80 * (100 - x) / 100) : 
  x = 40 := sorry

end soccer_balls_percentage_holes_l1136_113675


namespace koala_fiber_eaten_l1136_113631

-- Definitions based on conditions
def absorbs_percentage : ℝ := 0.40
def fiber_absorbed : ℝ := 12

-- The theorem statement to prove the total amount of fiber eaten
theorem koala_fiber_eaten : 
  (fiber_absorbed / absorbs_percentage) = 30 :=
by 
  sorry

end koala_fiber_eaten_l1136_113631


namespace perfume_weight_is_six_ounces_l1136_113638

def weight_in_pounds (ounces : ℕ) : ℕ := ounces / 16

def initial_weight := 5  -- Initial suitcase weight in pounds
def final_weight := 11   -- Final suitcase weight in pounds
def chocolate := 4       -- Weight of chocolate in pounds
def soap := 2 * 5        -- Weight of 2 bars of soap in ounces
def jam := 2 * 8         -- Weight of 2 jars of jam in ounces

def total_additional_weight :=
  chocolate + (weight_in_pounds soap) + (weight_in_pounds jam)

def perfume_weight_in_pounds := final_weight - initial_weight - total_additional_weight

def perfume_weight_in_ounces := perfume_weight_in_pounds * 16

theorem perfume_weight_is_six_ounces : perfume_weight_in_ounces = 6 := by sorry

end perfume_weight_is_six_ounces_l1136_113638
