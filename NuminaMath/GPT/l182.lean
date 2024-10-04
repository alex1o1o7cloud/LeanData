import Mathlib

namespace sufficient_food_l182_182330

-- Definitions for a large package and a small package as amounts of food lasting days
def largePackageDay : ℕ := L
def smallPackageDay : ℕ := S

-- Condition: One large package and four small packages last 14 days
axiom condition : largePackageDay + 4 * smallPackageDay = 14

-- Theorem to be proved: One large package and three small packages last at least 11 days
theorem sufficient_food : largePackageDay + 3 * smallPackageDay ≥ 11 :=
by
  sorry

end sufficient_food_l182_182330


namespace a10_eq_neg12_l182_182618

variable (a_n : ℕ → ℤ)
variable (S_n : ℕ → ℤ)
variable (d a1 : ℤ)

-- Conditions of the problem
axiom arithmetic_sequence : ∀ n : ℕ, a_n n = a1 + (n - 1) * d
axiom sum_of_first_n_terms : ∀ n : ℕ, S_n n = n * (2 * a1 + (n - 1) * d) / 2
axiom a2_eq_4 : a_n 2 = 4
axiom S8_eq_neg8 : S_n 8 = -8

-- The statement to prove
theorem a10_eq_neg12 : a_n 10 = -12 :=
sorry

end a10_eq_neg12_l182_182618


namespace symmetric_points_addition_l182_182509

theorem symmetric_points_addition 
  (m n : ℝ)
  (A : (ℝ × ℝ)) (B : (ℝ × ℝ))
  (hA : A = (2, m)) 
  (hB : B = (n, -1))
  (symmetry : A.1 = B.1 ∧ A.2 = -B.2) : 
  m + n = 3 :=
by
  sorry

end symmetric_points_addition_l182_182509


namespace spilled_bag_candies_l182_182300

theorem spilled_bag_candies (c1 c2 c3 c4 c5 c6 c7 : ℕ) (avg_candies_per_bag : ℕ) (x : ℕ) 
  (h_counts : c1 = 12 ∧ c2 = 14 ∧ c3 = 18 ∧ c4 = 22 ∧ c5 = 24 ∧ c6 = 26 ∧ c7 = 29)
  (h_avg : avg_candies_per_bag = 22)
  (h_total : c1 + c2 + c3 + c4 + c5 + c6 + c7 + x = 8 * avg_candies_per_bag) : x = 31 := 
by
  sorry

end spilled_bag_candies_l182_182300


namespace num_four_digit_numbers_greater_than_3999_with_product_of_middle_two_digits_exceeding_10_l182_182631

def num_valid_pairs : Nat := 34

def num_valid_first_digits : Nat := 6

def num_valid_last_digits : Nat := 10

theorem num_four_digit_numbers_greater_than_3999_with_product_of_middle_two_digits_exceeding_10 :
  (num_valid_first_digits * num_valid_pairs * num_valid_last_digits) = 2040 :=
by
  sorry

end num_four_digit_numbers_greater_than_3999_with_product_of_middle_two_digits_exceeding_10_l182_182631


namespace find_x_l182_182006

-- Defining the sum of integers from 30 to 40 inclusive
def sum_30_to_40 : ℕ := (30 + 31 + 32 + 33 + 34 + 35 + 36 + 37 + 38 + 39 + 40)

-- Defining the number of even integers from 30 to 40 inclusive
def count_even_30_to_40 : ℕ := 6

-- Given that x + y = 391, and y = count_even_30_to_40
-- Prove that x is equal to 385
theorem find_x (h : sum_30_to_40 + count_even_30_to_40 = 391) : sum_30_to_40 = 385 :=
by
  simp [sum_30_to_40, count_even_30_to_40] at h
  sorry

end find_x_l182_182006


namespace compare_neg_rational_l182_182111

def neg_one_third : ℚ := -1 / 3
def neg_one_half : ℚ := -1 / 2

theorem compare_neg_rational : neg_one_third > neg_one_half :=
by sorry

end compare_neg_rational_l182_182111


namespace tank_capacity_is_24_l182_182731

noncomputable def tank_capacity_proof : Prop :=
  ∃ (C : ℝ), (∃ (v : ℝ), (v / C = 1 / 6) ∧ ((v + 4) / C = 1 / 3)) ∧ C = 24

theorem tank_capacity_is_24 : tank_capacity_proof := sorry

end tank_capacity_is_24_l182_182731


namespace maximum_sin_C_in_triangle_l182_182644

theorem maximum_sin_C_in_triangle 
  (A B C : ℝ)
  (h1 : A + B + C = π) 
  (h2 : 1 / Real.tan A + 1 / Real.tan B = 6 / Real.tan C) : 
  Real.sin C = Real.sqrt 15 / 4 :=
sorry

end maximum_sin_C_in_triangle_l182_182644


namespace shaded_region_area_l182_182412

theorem shaded_region_area (RS : ℝ) (n_shaded : ℕ)
  (h1 : RS = 10) (h2 : n_shaded = 20) :
  (20 * (RS / (2 * Real.sqrt 2))^2) = 250 :=
by
  sorry

end shaded_region_area_l182_182412


namespace rectangle_area_l182_182447

theorem rectangle_area (A1 A2 : ℝ) (h1 : A1 = 40) (h2 : A2 = 10) :
    ∃ n : ℕ, n = 240 ∧ ∃ R : ℝ, R = 2 * Real.sqrt (40 / Real.pi) + 2 * Real.sqrt (10 / Real.pi) ∧ 
               (4 * Real.sqrt (10) / Real.sqrt (Real.pi)) * (6 * Real.sqrt (10) / Real.sqrt (Real.pi)) = n / Real.pi :=
by
  sorry

end rectangle_area_l182_182447


namespace terminal_side_of_half_angle_quadrant_l182_182501

def is_angle_in_third_quadrant (α : ℝ) (k : ℤ) : Prop :=
  k * 360 + 180 < α ∧ α < k * 360 + 270

def is_terminal_side_of_half_angle_in_quadrant (α : ℝ) : Prop :=
  (∃ n : ℤ, n * 360 + 90 < α / 2 ∧ α / 2 < n * 360 + 135) ∨ 
  (∃ n : ℤ, n * 360 + 270 < α / 2 ∧ α / 2 < n * 360 + 315)

theorem terminal_side_of_half_angle_quadrant (α : ℝ) (k : ℤ) :
  is_angle_in_third_quadrant α k → is_terminal_side_of_half_angle_in_quadrant α := 
sorry

end terminal_side_of_half_angle_quadrant_l182_182501


namespace saroj_age_proof_l182_182714

def saroj_present_age (vimal_age_6_years_ago saroj_age_6_years_ago : ℕ) : ℕ :=
  sorry    -- calculation logic would be here but is not needed per instruction

noncomputable def question_conditions (vimal_age_6_years_ago saroj_age_6_years_ago : ℕ) : Prop :=
  vimal_age_6_years_ago / 6 = saroj_age_6_years_ago / 5 ∧
  (vimal_age_6_years_ago + 10) / 11 = (saroj_age_6_years_ago + 10) / 10 ∧
  saroj_present_age vimal_age_6_years_ago saroj_age_6_years_ago = 16

theorem saroj_age_proof (vimal_age_6_years_ago saroj_age_6_years_ago : ℕ) :
  question_conditions vimal_age_6_years_ago saroj_age_6_years_ago :=
  sorry

end saroj_age_proof_l182_182714


namespace find_integers_l182_182920

theorem find_integers (a b m : ℕ) (ha : 0 < a) (hb : 0 < b) :
  (a + b^2) * (b + a^2) = 2^m → a = 1 ∧ b = 1 ∧ m = 2 :=
by
  sorry

end find_integers_l182_182920


namespace pills_per_week_l182_182517

theorem pills_per_week (hours_per_pill : ℕ) (hours_per_day : ℕ) (days_per_week : ℕ) 
(h1: hours_per_pill = 6) (h2: hours_per_day = 24) (h3: days_per_week = 7) :
(hours_per_day / hours_per_pill) * days_per_week = 28 :=
by
  sorry

end pills_per_week_l182_182517


namespace S_inter_T_eq_T_l182_182960

-- Define the sets S and T
def S := { s : ℤ | ∃ n : ℤ, s = 2 * n + 1 }
def T := { t : ℤ | ∃ n : ℤ, t = 4 * n + 1 }

-- Prove that S ∩ T = T
theorem S_inter_T_eq_T : S ∩ T = T := 
by
  sorry

end S_inter_T_eq_T_l182_182960


namespace find_x_l182_182497

open Real

def a : ℝ × ℝ := (3, 4)
def b : ℝ × ℝ := (2, -1)

noncomputable def dot_product (u v : ℝ × ℝ) : ℝ := u.1 * v.1 + u.2 * v.2

theorem find_x (x : ℝ) : 
  let ab := (a.1 + x * b.1, a.2 + x * b.2)
  let minus_b := (-b.1, -b.2)
  dot_product ab minus_b = 0 
  → x = -2 / 5 :=
by
  intros
  sorry

end find_x_l182_182497


namespace largest_constant_inequality_l182_182335

theorem largest_constant_inequality :
  ∃ C : ℝ, (∀ x y z : ℝ, x^2 + y^2 + z^2 + 1 ≥ C * (x + y + z)) ∧ C = Real.sqrt (4 / 3) :=
by {
  sorry
}

end largest_constant_inequality_l182_182335


namespace xiaoMing_xiaoHong_diff_university_l182_182785

-- Definitions based on problem conditions
inductive Student
| XiaoMing
| XiaoHong
| StudentC
| StudentD
deriving DecidableEq

inductive University
| A
| B
deriving DecidableEq

-- Definition for the problem
def num_ways_diff_university : Nat :=
  4 -- The correct answer based on the solution steps

-- Problem statement
theorem xiaoMing_xiaoHong_diff_university :
  let students := [Student.XiaoMing, Student.XiaoHong, Student.StudentC, Student.StudentD]
  let universities := [University.A, University.B]
  (∃ (assign : Student → University),
    assign Student.XiaoMing ≠ assign Student.XiaoHong ∧
    (assign Student.StudentC ≠ assign Student.StudentD ∨
     assign Student.XiaoMing ≠ assign Student.StudentD ∨
     assign Student.XiaoHong ≠ assign Student.StudentC ∨
     assign Student.XiaoMing ≠ assign Student.StudentC)) →
  num_ways_diff_university = 4 :=
by
  sorry

end xiaoMing_xiaoHong_diff_university_l182_182785


namespace find_rate_of_interest_l182_182283

-- Conditions
def principal : ℕ := 4200
def time : ℕ := 2
def interest_12 : ℕ := principal * 12 * time / 100
def additional_interest : ℕ := 504
def total_interest_r : ℕ := interest_12 + additional_interest

-- Theorem Statement
theorem find_rate_of_interest (r : ℕ) (h : 1512 = principal * r * time / 100) : r = 18 :=
by sorry

end find_rate_of_interest_l182_182283


namespace algebraic_expression_value_l182_182482

theorem algebraic_expression_value 
  (x y : ℝ) 
  (h : 2 * x + y = 1) : 
  (y + 1) ^ 2 - (y ^ 2 - 4 * x + 4) = -1 := 
by 
  sorry

end algebraic_expression_value_l182_182482


namespace compare_fractions_l182_182118

theorem compare_fractions : (-1 / 3) > (-1 / 2) := sorry

end compare_fractions_l182_182118


namespace fraction_shaded_in_cube_l182_182604

theorem fraction_shaded_in_cube :
  let side_length := 2
  let face_area := side_length * side_length
  let total_surface_area := 6 * face_area
  let shaded_faces := 3
  let shaded_face_area := face_area / 2
  let total_shaded_area := shaded_faces * shaded_face_area
  total_shaded_area / total_surface_area = 1 / 4 :=
by
  sorry

end fraction_shaded_in_cube_l182_182604


namespace power_of_two_l182_182831

theorem power_of_two (m n : ℕ) (h_m_pos : 0 < m) (h_n_pos : 0 < n) 
  (h_prime : Prime (m^(4^n + 1) - 1)) : 
  ∃ t : ℕ, n = 2^t :=
sorry

end power_of_two_l182_182831


namespace baguette_orderings_l182_182281

theorem baguette_orderings :
  ((Finset.card (Finset.powersetLen 3 (Finset.range 4))) * 3.factorial) = 24 :=
by
  sorry

end baguette_orderings_l182_182281


namespace second_group_num_persons_l182_182885

def man_hours (num_persons : ℕ) (days : ℕ) (hours_per_day : ℕ) : ℕ :=
  num_persons * days * hours_per_day

theorem second_group_num_persons :
  ∀ (x : ℕ),
    let first_group_man_hours := man_hours 36 12 5
    let second_group_days := 12
    let second_group_hours_per_day := 6
    (first_group_man_hours = man_hours x second_group_days second_group_hours_per_day) →
    x = 30 :=
by
  intros x first_group_man_hours second_group_days second_group_hours_per_day h
  sorry

end second_group_num_persons_l182_182885


namespace asymptote_equation_l182_182680

-- Defining the hyperbola equation
def hyperbola : Prop := ∀ x y : ℝ, 4 * x^2 - y^2 = 1

-- Stating the theorem that needs to be proved
theorem asymptote_equation (x y : ℝ) (h : hyperbola x y) : 2 * x + y = 0 :=
by sorry

end asymptote_equation_l182_182680


namespace remainder_of_91_pow_92_mod_100_l182_182551

theorem remainder_of_91_pow_92_mod_100 : (91 ^ 92) % 100 = 81 :=
by
  sorry

end remainder_of_91_pow_92_mod_100_l182_182551


namespace root_diff_sum_leq_l182_182882

-- Given conditions
def monic_quadratic_trinomial (f : Polynomial ℝ) : Prop :=
  f.degree = 2 ∧ leadingCoeff f = 1

def has_two_roots (f : Polynomial ℝ) : Prop :=
  ∃ a b, (f = Polynomial.C (a * b) * Polynomial.C (a + b))

def root_difference (f : Polynomial ℝ) : ℝ :=
  let ⟨a, b, h⟩ := exists_pair f in a - b

axiom monic_quadratic_f (f : Polynomial ℝ) : monic_quadratic_trinomial f
axiom monic_quadratic_g (g : Polynomial ℝ) : monic_quadratic_trinomial g
axiom two_root_f (f : Polynomial ℝ) : has_two_roots f
axiom two_root_g (g : Polynomial ℝ) : has_two_roots g
axiom two_root_f_plus_g (f g : Polynomial ℝ) : has_two_roots (f + g)
axiom root_diff_eq (f g : Polynomial ℝ) : root_difference f = root_difference g

-- Statement to prove
theorem root_diff_sum_leq (f g : Polynomial ℝ) :
  root_difference (f + g) ≤ root_difference f :=
by sorry

end root_diff_sum_leq_l182_182882


namespace license_plate_combinations_l182_182625

def consonants_count := 21
def vowels_count := 5
def digits_count := 10

theorem license_plate_combinations : 
  consonants_count * vowels_count * consonants_count * digits_count * vowels_count = 110250 :=
by
  sorry

end license_plate_combinations_l182_182625


namespace probability_recruitment_l182_182916

-- Definitions for conditions
def P_A : ℚ := 2/3
def P_A_not_and_B_not : ℚ := 1/12
def P_B_and_C : ℚ := 3/8

-- Independence of A, B, and C
axiom independence_A_B_C : ∀ {P_A P_B P_C : Prop}, 
  (P_A ∧ P_B ∧ P_C) → (P_A ∧ P_B) ∧ (P_A ∧ P_C) ∧ (P_B ∧ P_C)

-- Definition of probabilities of B and C
def P_B : ℚ := 3/4
def P_C : ℚ := 1/2

-- Main theorem
theorem probability_recruitment : 
  P_A = 2/3 ∧ 
  P_A_not_and_B_not = 1/12 ∧ 
  P_B_and_C = 3/8 ∧ 
  (∀ {P_A P_B P_C : Prop}, 
    (P_A ∧ P_B ∧ P_C) → (P_A ∧ P_B) ∧ (P_A ∧ P_C) ∧ (P_B ∧ P_C)) → 
  (P_B = 3/4 ∧ P_C = 1/2) ∧ 
  (2/3 * 3/4 * 1/2 + 1/3 * 3/4 * 1/2 + 2/3 * 1/4 * 1/2 + 2/3 * 3/4 * 1/2 = 17/24) := 
by sorry

end probability_recruitment_l182_182916


namespace jimin_rank_l182_182512

theorem jimin_rank (seokjin_rank : ℕ) (h1 : seokjin_rank = 4) (h2 : ∃ jimin_rank, jimin_rank = seokjin_rank + 1) : 
  ∃ jimin_rank, jimin_rank = 5 := 
by
  sorry

end jimin_rank_l182_182512


namespace cos_90_equals_0_l182_182128

-- Define the question: cos(90 degrees)
def cos_90_degrees : ℝ := Real.cos (Real.pi / 2)

-- State that cos(90 degrees) equals 0
theorem cos_90_equals_0 : cos_90_degrees = 0 :=
by sorry

end cos_90_equals_0_l182_182128


namespace rectangular_prism_diagonal_inequality_l182_182522

theorem rectangular_prism_diagonal_inequality 
  (a b c l : ℝ) 
  (h : l^2 = a^2 + b^2 + c^2) :
  (l^4 - a^4) * (l^4 - b^4) * (l^4 - c^4) ≥ 512 * a^4 * b^4 * c^4 := 
by sorry

end rectangular_prism_diagonal_inequality_l182_182522


namespace isosceles_triangle_base_length_l182_182569

noncomputable def equilateral_side_length (p_eq : ℕ) : ℕ := p_eq / 3

theorem isosceles_triangle_base_length (p_eq p_iso s b : ℕ) 
  (h1 : p_eq = 45)
  (h2 : p_iso = 40)
  (h3 : s = equilateral_side_length p_eq)
  (h4 : p_iso = s + s + b)
  : b = 10 :=
by
  simp [h1, h2, h3] at h4
  -- steps to solve for b would be written here
  sorry

end isosceles_triangle_base_length_l182_182569


namespace cos_90_eq_zero_l182_182154

-- Define cosine function and specify its behavior on the unit circle.
def cos (θ : ℝ) : ℝ :=
  -- Cosine gives the x-coordinate of the point on the unit circle.
  sorry -- Definition of cosine function is assumed.

-- Statement to be proved that involves the cosine function.
theorem cos_90_eq_zero : cos (90 * π / 180) = 0 := 
by
  sorry -- Proof is omitted.

end cos_90_eq_zero_l182_182154


namespace slope_of_line_joining_solutions_l182_182798

theorem slope_of_line_joining_solutions (x1 x2 y1 y2 : ℝ) :
  (4 / x1 + 5 / y1 = 1) → (4 / x2 + 5 / y2 = 1) →
  (x1 ≠ x2) → (y1 = 5 * x1 / (4 * x1 - 1)) → (y2 = 5 * x2 / (4 * x2 - 1)) →
  (x1 ≠ 1 / 4) → (x2 ≠ 1 / 4) →
  ((y2 - y1) / (x2 - x1) = - (5 / 21)) :=
by
  intros h_eq1 h_eq2 h_neq h_y1 h_y2 h_x1 h_x2
  -- Proof omitted for brevity
  sorry

end slope_of_line_joining_solutions_l182_182798


namespace best_sampling_method_l182_182284

theorem best_sampling_method :
  let elderly := 27
  let middle_aged := 54
  let young := 81
  let total_population := elderly + middle_aged + young
  let sample_size := 36
  let sampling_methods := ["simple random sampling", "systematic sampling", "stratified sampling"]
  stratified_sampling
:=
by
  sorry

end best_sampling_method_l182_182284


namespace lcm_condition_implies_all_two_l182_182768

theorem lcm_condition_implies_all_two (x : ℕ → ℕ)
  (h₁ : ∀ (i : ℕ), 1 ≤ i ∧ i ≤ 20 → 
        x (i + 2) ^ 2 = Nat.lcm (x (i + 1)) (x i) + Nat.lcm (x i) (x (i - 1)))
  (h₂ : x 0 = x 20)
  (h₃ : x 21 = x 1)
  (h₄ : x 22 = x 2) :
  ∀ i, 1 ≤ i ∧ i ≤ 20 → x i = 2 := 
sorry

end lcm_condition_implies_all_two_l182_182768


namespace line_parabola_one_intersection_l182_182503

theorem line_parabola_one_intersection (k : ℝ) : 
  ((∃ (x y : ℝ), y = k * x - 1 ∧ y^2 = 4 * x ∧ (∀ u v : ℝ, u ≠ x → v = k * u - 1 → v^2 ≠ 4 * u)) ↔ (k = 0 ∨ k = 1)) := 
sorry

end line_parabola_one_intersection_l182_182503


namespace jill_has_6_more_dolls_than_jane_l182_182557

theorem jill_has_6_more_dolls_than_jane
  (total_dolls : ℕ) 
  (jane_dolls : ℕ) 
  (more_dolls_than : ℕ → ℕ → Prop)
  (h1 : total_dolls = 32) 
  (h2 : jane_dolls = 13) 
  (jill_dolls : ℕ)
  (h3 : more_dolls_than jill_dolls jane_dolls) :
  (jill_dolls - jane_dolls) = 6 :=
by
  -- the proof goes here
  sorry

end jill_has_6_more_dolls_than_jane_l182_182557


namespace op_two_four_l182_182832

def op (a b : ℝ) : ℝ := 5 * a + 2 * b

theorem op_two_four : op 2 4 = 18 := by
  sorry

end op_two_four_l182_182832


namespace S_inter_T_eq_T_l182_182964

-- Define the sets S and T
def S := { s : ℤ | ∃ n : ℤ, s = 2 * n + 1 }
def T := { t : ℤ | ∃ n : ℤ, t = 4 * n + 1 }

-- Prove that S ∩ T = T
theorem S_inter_T_eq_T : S ∩ T = T := 
by
  sorry

end S_inter_T_eq_T_l182_182964


namespace marbles_in_jar_l182_182574

theorem marbles_in_jar (M : ℕ) (h1 : M / 24 = 24 * 26 / 26) (h2 : M / 26 + 1 = M / 24) : M = 312 := by
  sorry

end marbles_in_jar_l182_182574


namespace initial_bottles_l182_182757

-- Define the conditions
def drank_bottles : ℕ := 144
def left_bottles : ℕ := 157

-- Define the total_bottles function
def total_bottles : ℕ := drank_bottles + left_bottles

-- State the theorem to be proven
theorem initial_bottles : total_bottles = 301 :=
by
  sorry

end initial_bottles_l182_182757


namespace calculate_candy_bars_l182_182014

theorem calculate_candy_bars
  (soft_drink_calories : ℕ)
  (percent_added_sugar : ℕ)
  (recommended_intake : ℕ)
  (exceeded_percentage : ℕ)
  (candy_bar_calories : ℕ)
  (soft_drink_calories = 2500)
  (percent_added_sugar = 5)
  (recommended_intake = 150)
  (exceeded_percentage = 100)
  (candy_bar_calories = 25) :
  let added_sugar_from_drink := soft_drink_calories * percent_added_sugar / 100,
      exceeded_amount := recommended_intake * exceeded_percentage / 100,
      total_added_sugar := recommended_intake + exceeded_amount,
      added_sugar_from_candy_bars := total_added_sugar - added_sugar_from_drink in
  added_sugar_from_candy_bars / candy_bar_calories = 7 :=
by 
  -- proof
  sorry

end calculate_candy_bars_l182_182014


namespace number_of_adults_in_sleeper_class_l182_182640

-- Number of passengers in the train
def total_passengers : ℕ := 320

-- Percentage of passengers who are adults
def percentage_adults : ℚ := 75 / 100

-- Percentage of adults who are in the sleeper class
def percentage_adults_sleeper_class : ℚ := 15 / 100

-- Mathematical statement to prove
theorem number_of_adults_in_sleeper_class :
  (total_passengers * percentage_adults * percentage_adults_sleeper_class) = 36 :=
by
  sorry

end number_of_adults_in_sleeper_class_l182_182640


namespace cos_ninety_degrees_l182_182124

theorem cos_ninety_degrees : Real.cos (90 * Real.pi / 180) = 0 := 
by 
  sorry

end cos_ninety_degrees_l182_182124


namespace club_members_addition_l182_182725

theorem club_members_addition
  (current_members : ℕ := 10)
  (desired_members : ℕ := 2 * current_members + 5)
  (additional_members : ℕ := desired_members - current_members) :
  additional_members = 15 :=
by
  -- proof placeholder
  sorry

end club_members_addition_l182_182725


namespace Cora_book_reading_problem_l182_182333

theorem Cora_book_reading_problem
  (total_pages: ℕ)
  (read_monday: ℕ)
  (read_tuesday: ℕ)
  (read_wednesday: ℕ)
  (H: total_pages = 158 ∧ read_monday = 23 ∧ read_tuesday = 38 ∧ read_wednesday = 61) :
  ∃ P: ℕ, 23 + 38 + 61 + P + 2 * P = total_pages ∧ P = 12 :=
  sorry

end Cora_book_reading_problem_l182_182333


namespace notebook_area_l182_182066

variable (w h : ℝ)

def width_to_height_ratio (w h : ℝ) : Prop := w / h = 7 / 5
def perimeter (w h : ℝ) : Prop := 2 * w + 2 * h = 48
def area (w h : ℝ) : ℝ := w * h

theorem notebook_area (w h : ℝ) (ratio : width_to_height_ratio w h) (peri : perimeter w h) :
  area w h = 140 :=
by
  sorry

end notebook_area_l182_182066


namespace cone_volume_l182_182596

theorem cone_volume (diameter height : ℝ) (h_diam : diameter = 14) (h_height : height = 12) :
  (1 / 3 : ℝ) * Real.pi * ((diameter / 2) ^ 2) * height = 196 * Real.pi := by
  sorry

end cone_volume_l182_182596


namespace exists_fixed_point_sequence_l182_182234

theorem exists_fixed_point_sequence (N : ℕ) (hN : 0 < N) (a : ℕ → ℕ)
  (ha_conditions : ∀ i < N, a i % 2^(N+1) ≠ 0) :
  ∃ M, ∀ n ≥ M, a n = a M :=
sorry

end exists_fixed_point_sequence_l182_182234


namespace curves_intersect_at_l182_182071

noncomputable def f (x : ℝ) : ℝ := 3 * x^2 - 4 * x + 2
noncomputable def g (x : ℝ) : ℝ := -x^3 + 9 * x^2 - 4 * x + 2

theorem curves_intersect_at :
  (∃ x : ℝ, f x = g x) ↔ ([(0, 2), (6, 86)] = [(0, 2), (6, 86)]) :=
by
  sorry

end curves_intersect_at_l182_182071


namespace max_crate_weight_on_single_trip_l182_182579

-- Define the conditions
def trailer_capacity := {n | n = 3 ∨ n = 4 ∨ n = 5}
def min_crate_weight : ℤ := 1250

-- Define the maximum weight calculation
def max_weight (n : ℤ) (w : ℤ) : ℤ := n * w

-- Proof statement
theorem max_crate_weight_on_single_trip :
  ∃ w, (5 ∈ trailer_capacity) → max_weight 5 min_crate_weight = w ∧ w = 6250 := 
by
  sorry

end max_crate_weight_on_single_trip_l182_182579


namespace discount_is_five_l182_182457
-- Importing the needed Lean Math library

-- Defining the problem conditions
def costPrice : ℝ := 100
def profit_percent_with_discount : ℝ := 0.2
def profit_percent_without_discount : ℝ := 0.25

-- Calculating the respective selling prices
def sellingPrice_with_discount := costPrice * (1 + profit_percent_with_discount)
def sellingPrice_without_discount := costPrice * (1 + profit_percent_without_discount)

-- Calculating the discount 
def calculated_discount := sellingPrice_without_discount - sellingPrice_with_discount

-- Proving that the discount is $5
theorem discount_is_five : calculated_discount = 5 := by
  -- Proof omitted
  sorry

end discount_is_five_l182_182457


namespace solution_l182_182791

def p : Prop := ∀ x > 0, Real.log (x + 1) > 0
def q : Prop := ∀ a b : ℝ, a > b → a^2 > b^2

theorem solution : p ∧ ¬ q := by
  sorry

end solution_l182_182791


namespace sale_price_per_bearing_before_bulk_discount_l182_182516

-- Define the given conditions
def machines : ℕ := 10
def ball_bearings_per_machine : ℕ := 30
def total_ball_bearings : ℕ := machines * ball_bearings_per_machine

def normal_cost_per_bearing : ℝ := 1
def total_normal_cost : ℝ := total_ball_bearings * normal_cost_per_bearing

def bulk_discount : ℝ := 0.20
def sale_savings : ℝ := 120

-- The theorem we need to prove
theorem sale_price_per_bearing_before_bulk_discount (P : ℝ) :
  total_normal_cost - (total_ball_bearings * P * (1 - bulk_discount)) = sale_savings → 
  P = 0.75 :=
by sorry

end sale_price_per_bearing_before_bulk_discount_l182_182516


namespace intersection_eq_T_l182_182985

def S : Set ℤ := {s | ∃ n : ℤ, s = 2 * n + 1}
def T : Set ℤ := {t | ∃ n : ℤ, t = 4 * n + 1}

theorem intersection_eq_T : S ∩ T = T := sorry

end intersection_eq_T_l182_182985


namespace find_repair_charge_l182_182908

theorem find_repair_charge
    (cost_oil_change : ℕ)
    (cost_car_wash : ℕ)
    (num_oil_changes : ℕ)
    (num_repairs : ℕ)
    (num_car_washes : ℕ)
    (total_earnings : ℕ)
    (R : ℕ) :
    (cost_oil_change = 20) →
    (cost_car_wash = 5) →
    (num_oil_changes = 5) →
    (num_repairs = 10) →
    (num_car_washes = 15) →
    (total_earnings = 475) →
    5 * cost_oil_change + 10 * R + 15 * cost_car_wash = total_earnings →
    R = 30 :=
by
    intros h1 h2 h3 h4 h5 h6 h7
    sorry

end find_repair_charge_l182_182908


namespace probability_product_of_rolls_l182_182096

theorem probability_product_of_rolls :
  let dice := [1, 2, 3, 4, 5, 6] in
  (∃ (rolls : list ℕ) (h : rolls.length = 8),
    (∀ r ∈ rolls, r ∈ dice) ∧ 
    ((∀ r ∈ rolls, r % 2 = 1) ∨ 
     (∃! r ∈ rolls, r = 2 ∧ ∀ s ∈ (rolls.erase r), s % 2 = 1))
  ) →
  (list.prob (λ rolls, (∀ r ∈ rolls, r % 2 = 1) ∨ 
                    (∃! r ∈ rolls, r = 2 ∧ ∀ s ∈ (rolls.erase r), s % 2 = 1))
              (list.replicate 8 dice)) = 11 / 768 :=
sorry

end probability_product_of_rolls_l182_182096


namespace remaining_stickers_l182_182425

def stickers_per_page : ℕ := 20
def pages : ℕ := 12
def lost_pages : ℕ := 1

theorem remaining_stickers : 
  (pages * stickers_per_page - lost_pages * stickers_per_page) = 220 :=
  by
    sorry

end remaining_stickers_l182_182425


namespace intersection_point_of_line_and_y_axis_l182_182411

theorem intersection_point_of_line_and_y_axis :
  {p : ℝ × ℝ | ∃ x, p = (x, 2 * x + 1) ∧ x = 0} = {(0, 1)} :=
by sorry

end intersection_point_of_line_and_y_axis_l182_182411


namespace fraction_to_decimal_subtraction_l182_182909

theorem fraction_to_decimal_subtraction 
    (h : (3 : ℚ) / 40 = 0.075) : 
    0.075 - 0.005 = 0.070 := 
by 
    sorry

end fraction_to_decimal_subtraction_l182_182909


namespace intersection_eq_T_l182_182997

def S : Set ℤ := {s | ∃ n : ℤ, s = 2 * n + 1}
def T : Set ℤ := {t | ∃ n : ℤ, t = 4 * n + 1}

theorem intersection_eq_T : S ∩ T = T := sorry

end intersection_eq_T_l182_182997


namespace health_risk_probability_l182_182817

theorem health_risk_probability :
  let a := 0.08 * 500
  let b := 0.08 * 500
  let c := 0.08 * 500
  let d := 0.18 * 500
  let e := 0.18 * 500
  let f := 0.18 * 500
  let g := 0.05 * 500
  let h := 500 - (3 * 40 + 3 * 90 + 25)
  let q := 500 - (a + d + e + g)
  let p := 1
  let q := 3
  p + q = 4 := sorry

end health_risk_probability_l182_182817


namespace spherical_to_rectangular_l182_182910

theorem spherical_to_rectangular :
  let ρ := 6
  let θ := 7 * Real.pi / 4
  let φ := Real.pi / 3
  let x := ρ * Real.sin φ * Real.cos θ
  let y := ρ * Real.sin φ * Real.sin θ
  let z := ρ * Real.cos φ
  (x, y, z) = (-3 * Real.sqrt 6, -3 * Real.sqrt 6, 3) :=
by
  sorry

end spherical_to_rectangular_l182_182910


namespace min_value_y_l182_182217

theorem min_value_y (x : ℝ) (h : x > 5 / 4) : 
  ∃ y, y = 4*x - 1 + 1 / (4*x - 5) ∧ y ≥ 6 :=
by
  sorry

end min_value_y_l182_182217


namespace tan_315_eq_neg_1_l182_182589

theorem tan_315_eq_neg_1 : Real.tan (315 * Real.pi / 180) = -1 :=
by
  sorry

end tan_315_eq_neg_1_l182_182589


namespace intersection_of_sets_l182_182971

open Set Int

theorem intersection_of_sets (S T : Set Int) (hS : S = {s | ∃ n : ℤ, s = 2 * n + 1}) (hT : T = {t | ∃ n : ℤ, t = 4 * n + 1}) : S ∩ T = T :=
  by
  sorry

end intersection_of_sets_l182_182971


namespace log_sum_l182_182609

theorem log_sum : (Real.log 0.01 / Real.log 10) + (Real.log 16 / Real.log 2) = 2 := by
  sorry

end log_sum_l182_182609


namespace max_value_of_f_l182_182594

noncomputable theory
open Real

def f (x : ℝ) : ℝ := 2 + sin (3 * x)

theorem max_value_of_f : ∃ x : ℝ, f x = 3 :=
begin
  use (π / 6),
  unfold f,
  rw [sin_mul, sin_pi_div_six, cos_pi_div_six],
  norm_num,
end

end max_value_of_f_l182_182594


namespace intersection_eq_T_l182_182934

def S : Set ℤ := {s | ∃ n : ℤ, s = 2 * n + 1}
def T : Set ℤ := {t | ∃ n : ℤ, t = 4 * n + 1}

theorem intersection_eq_T : S ∩ T = T := 
sorry

end intersection_eq_T_l182_182934


namespace intersection_eq_T_l182_182986

def S : Set ℤ := {s | ∃ n : ℤ, s = 2 * n + 1}
def T : Set ℤ := {t | ∃ n : ℤ, t = 4 * n + 1}

theorem intersection_eq_T : S ∩ T = T := sorry

end intersection_eq_T_l182_182986


namespace length_of_train_is_correct_l182_182878

-- Definitions based on conditions
def speed_kmh := 90
def time_sec := 10

-- Convert speed from km/hr to m/s
def speed_ms := speed_kmh * (1000 / 3600)

-- Calculate the length of the train
def length_of_train := speed_ms * time_sec

-- Theorem to prove the length of the train
theorem length_of_train_is_correct : length_of_train = 250 := by
  sorry

end length_of_train_is_correct_l182_182878


namespace sum_first_20_integers_l182_182561

def sum_first_n_integers (n : ℕ) : ℕ :=
  n * (n + 1) / 2

theorem sum_first_20_integers : sum_first_n_integers 20 = 210 :=
by
  -- Provided proof omitted
  sorry

end sum_first_20_integers_l182_182561


namespace intersection_eq_T_l182_182996

def S : Set ℤ := {s | ∃ n : ℤ, s = 2 * n + 1}
def T : Set ℤ := {t | ∃ n : ℤ, t = 4 * n + 1}

theorem intersection_eq_T : S ∩ T = T := sorry

end intersection_eq_T_l182_182996


namespace vertical_asymptote_condition_l182_182344

theorem vertical_asymptote_condition (c : ℝ) :
  (∀ x : ℝ, (x = 3 ∨ x = -6) → (x^2 - x + c = 0)) → 
  (c = -6 ∨ c = -42) :=
by
  sorry

end vertical_asymptote_condition_l182_182344


namespace eggs_per_day_second_store_l182_182661

-- Define the number of eggs in a dozen
def eggs_in_a_dozen : ℕ := 12

-- Define the number of dozen eggs supplied to the first store each day
def dozen_per_day_first_store : ℕ := 5

-- Define the number of eggs supplied to the first store each day
def eggs_per_day_first_store : ℕ := dozen_per_day_first_store * eggs_in_a_dozen

-- Define the number of days in a week
def days_in_week : ℕ := 7

-- Calculate the weekly supply to the first store
def weekly_supply_first_store : ℕ := eggs_per_day_first_store * days_in_week

-- Define the total weekly supply to both stores
def total_weekly_supply : ℕ := 630

-- Calculate the weekly supply to the second store
def weekly_supply_second_store : ℕ := total_weekly_supply - weekly_supply_first_store

-- Define the theorem to prove the number of eggs supplied to the second store each day
theorem eggs_per_day_second_store : weekly_supply_second_store / days_in_week = 30 := by
  sorry

end eggs_per_day_second_store_l182_182661


namespace problem_solution_l182_182241

theorem problem_solution (x1 x2 : ℝ) (h1 : x1^2 + x1 - 4 = 0) (h2 : x2^2 + x2 - 4 = 0) (h3 : x1 + x2 = -1) : 
  x1^3 - 5 * x2^2 + 10 = -19 := 
by 
  sorry

end problem_solution_l182_182241


namespace max_area_of_right_triangle_with_hypotenuse_4_l182_182638

theorem max_area_of_right_triangle_with_hypotenuse_4 : 
  (∀ (a b : ℝ), a^2 + b^2 = 16 → (∃ S, S = 1/2 * a * b ∧ S ≤ 4)) ∧ 
  (∃ a b : ℝ, a^2 + b^2 = 16 ∧ a = b ∧ 1/2 * a * b = 4) :=
by
  sorry

end max_area_of_right_triangle_with_hypotenuse_4_l182_182638


namespace smallest_bottles_needed_l182_182907

/-- Christine needs at least 60 fluid ounces of milk, the store sells milk in 250 milliliter bottles,
and there are 32 fluid ounces in 1 liter. The smallest number of bottles Christine should purchase
is 8. -/
theorem smallest_bottles_needed
  (fl_oz_needed : ℕ := 60)
  (ml_per_bottle : ℕ := 250)
  (fl_oz_per_liter : ℕ := 32) :
  let liters_needed := fl_oz_needed / fl_oz_per_liter
  let ml_needed := liters_needed * 1000
  let bottles := (ml_needed + ml_per_bottle - 1) / ml_per_bottle
  bottles = 8 :=
by
  sorry

end smallest_bottles_needed_l182_182907


namespace daughter_age_is_10_l182_182450

variable (D : ℕ)

-- Conditions
def father_current_age (D : ℕ) : ℕ := 4 * D
def father_age_in_20_years (D : ℕ) : ℕ := father_current_age D + 20
def daughter_age_in_20_years (D : ℕ) : ℕ := D + 20

-- Theorem statement
theorem daughter_age_is_10 :
  father_current_age D = 40 →
  father_age_in_20_years D = 2 * daughter_age_in_20_years D →
  D = 10 :=
by
  -- Here would be the proof steps to show that D = 10 given the conditions
  sorry

end daughter_age_is_10_l182_182450


namespace july_percentage_is_correct_l182_182054

def total_scientists : ℕ := 120
def july_scientists : ℕ := 16
def july_percentage : ℚ := (july_scientists : ℚ) / (total_scientists : ℚ) * 100

theorem july_percentage_is_correct : july_percentage = 13.33 := 
by 
  -- Provides the proof directly as a statement
  sorry

end july_percentage_is_correct_l182_182054


namespace cos_90_eq_zero_l182_182148

theorem cos_90_eq_zero : (cos 90) = 0 := by
  -- Condition: The angle 90 degrees corresponds to the point (0, 1) on the unit circle.
  -- We claim cos 90 degrees is 0
  sorry

end cos_90_eq_zero_l182_182148


namespace convex_15gon_smallest_angle_arith_seq_l182_182053

noncomputable def smallest_angle (n : ℕ) (avg_angle d : ℕ) : ℕ :=
156 - 7 * d

theorem convex_15gon_smallest_angle_arith_seq :
  let n := 15 in
  ∀ (a d : ℕ), 
  (a = 156 - 7 * d) ∧
  (avg_angle = (13 * 180) / n) ∧
  (forall i : ℕ, 1 ≤ i ∧ i < n → d < 24 / 7) →
  a = 135 :=
sorry

end convex_15gon_smallest_angle_arith_seq_l182_182053


namespace fraction_equality_l182_182346

theorem fraction_equality (a b : ℝ) (h : a / b = 2 / 3) : a / (a + b) = 2 / 5 :=
by
  sorry

end fraction_equality_l182_182346


namespace cos_90_eq_0_l182_182140

theorem cos_90_eq_0 :
  ∃ (p : ℝ × ℝ), p = (0, 1) ∧ ∀ θ : ℝ, θ = 90 → cos θ = p.1 :=
by
  let p := (0, 1)
  use p
  split
  · rfl
  · intros θ h
    rw h
    sorry

end cos_90_eq_0_l182_182140


namespace salt_percentage_l182_182597

theorem salt_percentage (salt water : ℝ) (h_salt : salt = 10) (h_water : water = 40) : 
  salt / water = 0.2 :=
by
  sorry

end salt_percentage_l182_182597


namespace nonagon_angles_l182_182184

/-- Determine the angles of the nonagon given specified conditions -/
theorem nonagon_angles (a : ℝ) (x : ℝ) 
  (h_angle_eq : ∀ (AIH BCD HGF : ℝ), AIH = x → BCD = x → HGF = x)
  (h_internal_sum : 7 * 180 = 1260)
  (h_tessellation : x + x + x + (360 - x) + (360 - x) + (360 - x) = 1080) :
  True := sorry

end nonagon_angles_l182_182184


namespace compare_neg_thirds_and_halves_l182_182115

theorem compare_neg_thirds_and_halves : (-1 : ℚ) / 3 > (-1 : ℚ) / 2 :=
by
  sorry

end compare_neg_thirds_and_halves_l182_182115


namespace total_pencils_l182_182693

-- Define the initial conditions
def initial_pencils : ℕ := 41
def added_pencils : ℕ := 30

-- Define the statement to be proven
theorem total_pencils :
  initial_pencils + added_pencils = 71 :=
by
  sorry

end total_pencils_l182_182693


namespace vec_a_squared_minus_vec_b_squared_l182_182802

variable (a b : ℝ × ℝ)
variable (h1 : a + b = (-3, 6))
variable (h2 : a - b = (-3, 2))

theorem vec_a_squared_minus_vec_b_squared : (a.1 * a.1 + a.2 * a.2) - (b.1 * b.1 + b.2 * b.2) = 32 :=
sorry

end vec_a_squared_minus_vec_b_squared_l182_182802


namespace determine_parabola_l182_182494

-- Define the parabola passing through point P(1,1)
def parabola_passing_through (a b c : ℝ) :=
  (1:ℝ)^2 * a + 1 * b + c = 1

-- Define the condition that the tangent line at Q(2, -1) has a slope parallel to y = x - 3, which means slope = 1
def tangent_slope_at_Q (a b : ℝ) :=
  4 * a + b = 1

-- Define the parabola passing through point Q(2, -1)
def parabola_passing_through_Q (a b c : ℝ) :=
  (2:ℝ)^2 * a + (2:ℝ) * b + c = -1

-- The proof statement
theorem determine_parabola (a b c : ℝ):
  parabola_passing_through a b c ∧ 
  tangent_slope_at_Q a b ∧ 
  parabola_passing_through_Q a b c → 
  a = 3 ∧ b = -11 ∧ c = 9 :=
by
  sorry

end determine_parabola_l182_182494


namespace intersection_of_sets_l182_182884

theorem intersection_of_sets (A B : Set ℕ) (hA : A = {1, 2, 3}) (hB : B = {3, 4}) :
  A ∩ B = {3} :=
by
  rw [hA, hB]
  exact sorry

end intersection_of_sets_l182_182884


namespace total_receipts_l182_182458

theorem total_receipts 
  (x y : ℕ) 
  (h1 : x + y = 64)
  (h2 : y ≥ 8) 
  : 3 * x + 4 * y = 200 := 
by
  sorry

end total_receipts_l182_182458


namespace equal_areas_of_cyclic_quadrilateral_and_orthocenter_quadrilateral_l182_182643

noncomputable def Midpoint (A B : Point) : Point := sorry
noncomputable def Orthocenter (A B C : Triangle) : Point := sorry
noncomputable def Area (P Q R S : Quadrilateral) : ℝ := sorry

theorem equal_areas_of_cyclic_quadrilateral_and_orthocenter_quadrilateral
  (A B C D E F G H W X Y Z : Point)
  (h1 : CyclicQuadrilateral A B C D)
  (h2 : Midpoint A B = E)
  (h3 : Midpoint B C = F)
  (h4 : Midpoint C D = G)
  (h5 : Midpoint D A = H)
  (h6 : Orthocenter A H E = W)
  (h7 : Orthocenter B E F = X)
  (h8 : Orthocenter C F G = Y)
  (h9 : Orthocenter D G H = Z)
  : Area A B C D = Area W X Y Z :=
sorry

end equal_areas_of_cyclic_quadrilateral_and_orthocenter_quadrilateral_l182_182643


namespace conversion_1_conversion_2_conversion_3_l182_182083

theorem conversion_1 : 2 * 1000 = 2000 := sorry

theorem conversion_2 : 9000 / 1000 = 9 := sorry

theorem conversion_3 : 8 * 1000 = 8000 := sorry

end conversion_1_conversion_2_conversion_3_l182_182083


namespace sufficient_food_supply_l182_182314

variable {L S : ℝ}

theorem sufficient_food_supply (h1 : L + 4 * S = 14) (h2 : L > S) : L + 3 * S ≥ 11 :=
by
  sorry

end sufficient_food_supply_l182_182314


namespace calculate_chord_length_l182_182256

noncomputable def chord_length_of_tangent (r1 r2 : ℝ) (c : ℝ) : Prop :=
  r1^2 - r2^2 = 18 ∧ (c / 2)^2 = 18

theorem calculate_chord_length (r1 r2 : ℝ) (h : chord_length_of_tangent r1 r2 (6 * Real.sqrt 2)) :
  (6 * Real.sqrt 2) = 6 * Real.sqrt 2 :=
by
  sorry

end calculate_chord_length_l182_182256


namespace number_of_paths_l182_182361

-- Definition of vertices
inductive Vertex
| A | B | C | D | E | F | G

-- Edges based on the description
def edges : List (Vertex × Vertex) := [
  (Vertex.A, Vertex.G), (Vertex.G, Vertex.C), (Vertex.G, Vertex.D), (Vertex.C, Vertex.B),
  (Vertex.D, Vertex.C), (Vertex.D, Vertex.F), (Vertex.D, Vertex.E), (Vertex.E, Vertex.F),
  (Vertex.F, Vertex.B), (Vertex.C, Vertex.F), (Vertex.A, Vertex.C), (Vertex.A, Vertex.D)
]

-- Function to count paths from A to B without revisiting any vertex
def countPaths (start : Vertex) (goal : Vertex) (adj : List (Vertex × Vertex)) : Nat :=
sorry

-- The theorem statement
theorem number_of_paths : countPaths Vertex.A Vertex.B edges = 10 :=
sorry

end number_of_paths_l182_182361


namespace digital_earth_functionalities_l182_182472

def digital_earth_allows_internet_navigation : Prop := 
  ∀ (f : String), f ∈ ["Receive distance education", "Shop online", "Seek medical advice online"]

def digital_earth_does_not_allow_physical_travel : Prop := 
  ¬ (∀ (f : String), f ∈ ["Travel around the world"])

theorem digital_earth_functionalities :
  digital_earth_allows_internet_navigation ∧ digital_earth_does_not_allow_physical_travel →
  ∀(f : String), f ∈ ["Receive distance education", "Shop online", "Seek medical advice online"] :=
by
  sorry

end digital_earth_functionalities_l182_182472


namespace volume_of_cube_l182_182085

theorem volume_of_cube (A : ℝ) (s V : ℝ) 
  (hA : A = 150) 
  (h_surface_area : A = 6 * s^2) 
  (h_side_length : s = 5) :
  V = s^3 →
  V = 125 :=
by
  sorry

end volume_of_cube_l182_182085


namespace max_popsicles_is_13_l182_182386

/-- Pablo's budgets and prices for buying popsicles. -/
structure PopsicleStore where
  single_popsicle_cost : ℕ
  three_popsicle_box_cost : ℕ
  five_popsicle_box_cost : ℕ
  starting_budget : ℕ

/-- The maximum number of popsicles Pablo can buy given the store's prices and his budget. -/
def maxPopsicles (store : PopsicleStore) : ℕ :=
  let num_five_popsicle_boxes := store.starting_budget / store.five_popsicle_box_cost
  let remaining_after_five_boxes := store.starting_budget % store.five_popsicle_box_cost
  let num_three_popsicle_boxes := remaining_after_five_boxes / store.three_popsicle_box_cost
  let remaining_after_three_boxes := remaining_after_five_boxes % store.three_popsicle_box_cost
  let num_single_popsicles := remaining_after_three_boxes / store.single_popsicle_cost
  num_five_popsicle_boxes * 5 + num_three_popsicle_boxes * 3 + num_single_popsicles

theorem max_popsicles_is_13 :
  maxPopsicles { single_popsicle_cost := 1, 
                 three_popsicle_box_cost := 2, 
                 five_popsicle_box_cost := 3, 
                 starting_budget := 8 } = 13 := by
  sorry

end max_popsicles_is_13_l182_182386


namespace rectangle_perimeter_l182_182854

-- Definitions based on the conditions
def length : ℕ := 15
def width : ℕ := 8

-- Definition of the perimeter function
def perimeter (l w : ℕ) : ℕ := 2 * (l + w)

-- Statement of the theorem we need to prove
theorem rectangle_perimeter : perimeter length width = 46 := by
  sorry

end rectangle_perimeter_l182_182854


namespace cat_food_sufficiency_l182_182309

variable (L S : ℝ)

theorem cat_food_sufficiency (h : L + 4 * S = 14) : L + 3 * S >= 11 :=
sorry

end cat_food_sufficiency_l182_182309


namespace gift_card_amount_l182_182246

theorem gift_card_amount (original_price final_price : ℝ) 
  (discount1 discount2 : ℝ) 
  (discounted_price1 discounted_price2 : ℝ) :
  original_price = 2000 →
  discount1 = 0.15 →
  discount2 = 0.10 →
  discounted_price1 = original_price - (discount1 * original_price) →
  discounted_price2 = discounted_price1 - (discount2 * discounted_price1) →
  final_price = 1330 →
  discounted_price2 - final_price = 200 := 
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end gift_card_amount_l182_182246


namespace de_morgan_birth_year_jenkins_birth_year_l182_182032

open Nat

theorem de_morgan_birth_year
  (x : ℕ) (hx : x = 43) (hx_square : x * x = 1849) :
  1849 - 43 = 1806 :=
by
  sorry

theorem jenkins_birth_year
  (a b : ℕ) (ha : a = 5) (hb : b = 6) (m : ℕ) (hm : m = 31) (n : ℕ) (hn : n = 5)
  (ha_sq : a * a = 25) (hb_sq : b * b = 36) (ha4 : a * a * a * a = 625)
  (hb4 : b * b * b * b = 1296) (hm2 : m * m = 961) (hn4 : n * n * n * n = 625) :
  1921 - 61 = 1860 ∧
  1922 - 62 = 1860 ∧
  1875 - 15 = 1860 :=
by
  sorry

end de_morgan_birth_year_jenkins_birth_year_l182_182032


namespace difference_of_squares_l182_182065

theorem difference_of_squares {a b : ℝ} (h1 : a + b = 75) (h2 : a - b = 15) : a^2 - b^2 = 1125 :=
by
  sorry

end difference_of_squares_l182_182065


namespace marbles_total_l182_182476

theorem marbles_total (fabian kyle miles : ℕ) (h1 : fabian = 3 * kyle) (h2 : fabian = 5 * miles) (h3 : fabian = 15) : kyle + miles = 8 := by
  sorry

end marbles_total_l182_182476


namespace trig_expression_equality_l182_182107

theorem trig_expression_equality :
  let tan_60 := Real.sqrt 3
  let tan_45 := 1
  let cos_30 := Real.sqrt 3 / 2
  2 * tan_60 + tan_45 - 4 * cos_30 = 1 := by
  let tan_60 := Real.sqrt 3
  let tan_45 := 1
  let cos_30 := Real.sqrt 3 / 2
  sorry

end trig_expression_equality_l182_182107


namespace sin_cos_expr1_sin_cos_expr2_l182_182617

variable {x : ℝ}
variable (hx : Real.tan x = 2)

theorem sin_cos_expr1 : (2 / 3) * (Real.sin x)^2 + (1 / 4) * (Real.cos x)^2 = 7 / 12 := by
  sorry

theorem sin_cos_expr2 : 2 * (Real.sin x)^2 - (Real.sin x) * (Real.cos x) + (Real.cos x)^2 = 7 / 5 := by
  sorry

end sin_cos_expr1_sin_cos_expr2_l182_182617


namespace cos_90_eq_zero_l182_182150

-- Define cosine function and specify its behavior on the unit circle.
def cos (θ : ℝ) : ℝ :=
  -- Cosine gives the x-coordinate of the point on the unit circle.
  sorry -- Definition of cosine function is assumed.

-- Statement to be proved that involves the cosine function.
theorem cos_90_eq_zero : cos (90 * π / 180) = 0 := 
by
  sorry -- Proof is omitted.

end cos_90_eq_zero_l182_182150


namespace perimeter_shaded_region_l182_182230

theorem perimeter_shaded_region (r: ℝ) (circumference: ℝ) (h1: circumference = 36) (h2: {x // x = 3 * (circumference / 6)}) : x = 18 :=
by
  sorry

end perimeter_shaded_region_l182_182230


namespace sum_of_angles_divisible_by_360_l182_182728

theorem sum_of_angles_divisible_by_360 {n : ℕ} (h : n ≠ 0) :
  let sides := 2 * n in
  (sides - 2) * 180 = 360 * (n - 1) :=
by
  have sides_eq_2n : sides = 2 * n := rfl
  sorry

end sum_of_angles_divisible_by_360_l182_182728


namespace cos_90_deg_eq_zero_l182_182135

noncomputable def cos_90_degrees : ℝ :=
cos (90 * real.pi / 180)

theorem cos_90_deg_eq_zero : cos_90_degrees = 0 := 
by
  -- Definitions related to the conditions
  let point_0 := (1 : ℝ, 0 : ℝ)
  let point_90 := (0 : ℝ, 1 : ℝ)

  -- Rotating the point (1, 0) by 90 degrees counterclockwise results in the point (0, 1)
  have h_rotate_90 : (real.cos (90 * real.pi / 180), real.sin (90 * real.pi / 180)) = point_90 :=
    by
      -- Rotation logic is abstracted out in this property of cos and sin
      have h_cos_90 : real.cos (real.pi / 2) = 0 := by sorry
      have h_sin_90 : real.sin (real.pi / 2) = 1 := by sorry
      exact ⟨h_cos_90, h_sin_90⟩

  -- By definition, cos 90 degrees is the x-coordinate of the rotated point
  show cos_90_degrees = 0,
  from congr_arg Prod.fst h_rotate_90

end cos_90_deg_eq_zero_l182_182135


namespace smallest_w_factor_l182_182568

theorem smallest_w_factor:
  ∃ w : ℕ, (∃ n : ℕ, n = 936 * w ∧ 
              2 ^ 5 ∣ n ∧ 
              3 ^ 3 ∣ n ∧ 
              14 ^ 2 ∣ n) ∧ 
              w = 1764 :=
sorry

end smallest_w_factor_l182_182568


namespace paint_total_gallons_l182_182287

theorem paint_total_gallons
  (white_paint_gallons : ℕ)
  (blue_paint_gallons : ℕ)
  (h_wp : white_paint_gallons = 660)
  (h_bp : blue_paint_gallons = 6029) :
  white_paint_gallons + blue_paint_gallons = 6689 := 
by
  sorry

end paint_total_gallons_l182_182287


namespace find_a_value_l182_182924

theorem find_a_value (a x1 x2 : ℝ) (h1 : a > 0) (h2 : x1 + x2 = 15) 
  (h3 : ∀ x, x^2 - 2 * a * x - 8 * a^2 < 0) : a = 15 / 2 :=
  sorry

end find_a_value_l182_182924


namespace problem1_problem2_l182_182800

noncomputable def f (x : ℝ) (a : ℝ) : ℝ := x + a^2 / x
noncomputable def g (x : ℝ) : ℝ := x + Real.log x
noncomputable def h (x : ℝ) (a : ℝ) : ℝ := f x a + g x

-- Problem 1: Prove that a = sqrt(3) given that x = 1 is an extremum point for h(x, a)
theorem problem1 (a : ℝ) (h_extremum : ∀ x : ℝ, x = 1 → 0 = (2 - a^2 / x^2 + 1 / x)) : a = Real.sqrt 3 := sorry

-- Problem 2: Prove the range of a is [ (e + 1) / 2, +∞ ) such that for any x1, x2 ∈ [1, e], f(x1, a) ≥ g(x2)
theorem problem2 (a : ℝ) :
  (∀ x1 x2 : ℝ, 1 ≤ x1 ∧ x1 ≤ Real.exp 1 ∧ 1 ≤ x2 ∧ x2 ≤ Real.exp 1 → f x1 a ≥ g x2) →
  (Real.exp 1 + 1) / 2 ≤ a :=
sorry

end problem1_problem2_l182_182800


namespace rectangular_prism_edge_properties_l182_182364

-- Define a rectangular prism and the concept of parallel and perpendicular pairs of edges.
structure RectangularPrism :=
  (vertices : Fin 8 → Fin 3 → ℝ)
  -- Additional necessary conditions on the structure could be added here.

-- Define the number of parallel edges in a rectangular prism
def number_of_parallel_edge_pairs (rp : RectangularPrism) : ℕ :=
  -- Formula or logic to count parallel edge pairs.
  8 -- Placeholder for actual logic computation, based on problem conditions.

-- Define the number of perpendicular edges in a rectangular prism
def number_of_perpendicular_edge_pairs (rp : RectangularPrism) : ℕ :=
  -- Formula or logic to count perpendicular edge pairs.
  20 -- Placeholder for actual logic computation, based on problem conditions.

-- Theorem that asserts the requirement based on conditions
theorem rectangular_prism_edge_properties (rp : RectangularPrism) :
  number_of_parallel_edge_pairs rp = 8 ∧ number_of_perpendicular_edge_pairs rp = 20 :=
  by
    -- Placeholder proof that establishes the theorem
    sorry

end rectangular_prism_edge_properties_l182_182364


namespace find_values_l182_182463

noncomputable def value_of_a (a : ℚ) : Prop :=
  4 + a = 2

noncomputable def value_of_b (b : ℚ) : Prop :=
  b^2 - 2 * b = 24 ∧ 4 * b^2 - 2 * b = 72

theorem find_values (a b : ℚ) (h1 : value_of_a a) (h2 : value_of_b b) :
  a = -2 ∧ b = -4 :=
by
  sorry

end find_values_l182_182463


namespace min_guesses_correct_l182_182278

def min_guesses (n k : ℕ) : ℕ :=
  if n = 2 * k then 2 else 1

theorem min_guesses_correct (n k : ℕ) (h : n > k) :
  (min_guesses n k = 2 ↔ n = 2 * k) ∧ (min_guesses n k = 1 ↔ n ≠ 2 * k) := by
  sorry

end min_guesses_correct_l182_182278


namespace b_over_a_squared_eq_seven_l182_182576

theorem b_over_a_squared_eq_seven (a b k : ℕ) (ha : a > 1) (hb : b = a * (10^k + 1)) (hdiv : a^2 ∣ b) :
  b / a^2 = 7 :=
sorry

end b_over_a_squared_eq_seven_l182_182576


namespace intersection_M_N_l182_182792

def M : Set ℝ := { x | x ≤ 4 }
def N : Set ℝ := { x | 0 < x }

theorem intersection_M_N : M ∩ N = { x | 0 < x ∧ x ≤ 4 } := 
by 
  sorry

end intersection_M_N_l182_182792


namespace probability_of_odd_product_is_zero_l182_182756

-- Define the spinners
def spinnerC : List ℕ := [1, 3, 5, 7]
def spinnerD : List ℕ := [2, 4, 6]

-- Define the condition that the odds and evens have a specific product property
axiom odd_times_even_is_even {a b : ℕ} (ha : a % 2 = 1) (hb : b % 2 = 0) : (a * b) % 2 = 0

-- Define the probability of getting an odd product
noncomputable def probability_odd_product : ℕ :=
  if ∃ a ∈ spinnerC, ∃ b ∈ spinnerD, (a * b) % 2 = 1 then 1 else 0

-- Main theorem
theorem probability_of_odd_product_is_zero : probability_odd_product = 0 := by
  sorry

end probability_of_odd_product_is_zero_l182_182756


namespace sum_of_reciprocal_AP_l182_182415

theorem sum_of_reciprocal_AP (a1 a2 a3 : ℝ) (d : ℝ)
  (h1 : a1 + a2 + a3 = 11/18)
  (h2 : 1/a1 + 1/a2 + 1/a3 = 18)
  (h3 : 1/a2 = 1/a1 + d)
  (h4 : 1/a3 = 1/a1 + 2*d) :
  (a1 = 1/9 ∧ a2 = 1/6 ∧ a3 = 1/3) ∨ (a1 = 1/3 ∧ a2 = 1/6 ∧ a3 = 1/9) :=
sorry

end sum_of_reciprocal_AP_l182_182415


namespace product_of_y_coordinates_l182_182667

theorem product_of_y_coordinates (k : ℝ) (hk : k > 0) :
    let y1 := 2 + Real.sqrt (k^2 - 64)
    let y2 := 2 - Real.sqrt (k^2 - 64)
    y1 * y2 = 68 - k^2 :=
by 
  sorry

end product_of_y_coordinates_l182_182667


namespace cost_of_ice_cream_scoop_l182_182191

theorem cost_of_ice_cream_scoop
  (num_meals : ℕ) (meal_cost : ℕ) (total_money : ℕ)
  (total_meals_cost : num_meals * meal_cost = 30)
  (remaining_money : total_money - 30 = 15)
  (num_ice_cream_scoops : ℕ) (cost_per_scoop : ℕ)
  (total_cost : 30 + 15 = total_money)
  (total_ice_cream_cost : num_ice_cream_scoops * cost_per_scoop = remaining_money) :
  cost_per_scoop = 5 :=
by
  have h_num_meals : num_meals = 3 := by sorry
  have h_meal_cost : meal_cost = 10 := by sorry
  have h_total_money : total_money = 45 := by sorry
  have h_num_ice_cream_scoops : num_ice_cream_scoops = 3 := by sorry
  exact sorry

end cost_of_ice_cream_scoop_l182_182191


namespace cube_cut_off_edges_l182_182187

theorem cube_cut_off_edges :
  let original_edges := 12
  let new_edges_per_vertex := 3
  let vertices := 8
  let new_edges := new_edges_per_vertex * vertices
  (original_edges + new_edges) = 36 :=
by
  sorry

end cube_cut_off_edges_l182_182187


namespace add_neg_two_and_three_l182_182305

theorem add_neg_two_and_three : -2 + 3 = 1 :=
by
  sorry

end add_neg_two_and_three_l182_182305


namespace intersection_S_T_eq_T_l182_182949

def S : Set ℤ := { s | ∃ n : ℤ, s = 2 * n + 1 }
def T : Set ℤ := { t | ∃ n : ℤ, t = 4 * n + 1 }

theorem intersection_S_T_eq_T : S ∩ T = T := by
  sorry

end intersection_S_T_eq_T_l182_182949


namespace problem_statement_l182_182001

def g (x : ℝ) : ℝ := 3 * x + 2

theorem problem_statement : g (g (g 3)) = 107 := by
  sorry

end problem_statement_l182_182001


namespace length_breadth_difference_l182_182862

theorem length_breadth_difference (L W : ℝ) 
  (h1 : W = 1/2 * L) 
  (h2 : L * W = 288) : L - W = 12 :=
by
  sorry

end length_breadth_difference_l182_182862


namespace victor_cannot_escape_k4_l182_182099

theorem victor_cannot_escape_k4
  (r : ℝ)
  (speed_A : ℝ)
  (speed_B : ℝ) 
  (k : ℝ)
  (hr : r = 1)
  (hk : k = 4)
  (hA_speed : speed_A = 4 * speed_B)
  (B_starts_at_center : ∃ (B : ℝ), B = 0):
  ¬(∃ (escape_strategy : ℝ → ℝ), escape_strategy 0 = 0 → escape_strategy r = 1) :=
sorry

end victor_cannot_escape_k4_l182_182099


namespace d_is_multiple_of_4_c_minus_d_is_multiple_of_4_c_minus_d_is_multiple_of_2_l182_182536

variable (c d : ℕ)

-- Conditions: c is a multiple of 4 and d is a multiple of 8
def is_multiple_of_4 (n : ℕ) : Prop := ∃ k : ℕ, n = 4 * k
def is_multiple_of_8 (n : ℕ) : Prop := ∃ k : ℕ, n = 8 * k

-- Statements to prove:

-- A. d is a multiple of 4
theorem d_is_multiple_of_4 {c d : ℕ} (h1 : is_multiple_of_4 c) (h2 : is_multiple_of_8 d) : is_multiple_of_4 d :=
sorry

-- B. c - d is a multiple of 4
theorem c_minus_d_is_multiple_of_4 {c d : ℕ} (h1 : is_multiple_of_4 c) (h2 : is_multiple_of_8 d) : is_multiple_of_4 (c - d) :=
sorry

-- D. c - d is a multiple of 2
theorem c_minus_d_is_multiple_of_2 {c d : ℕ} (h1 : is_multiple_of_4 c) (h2 : is_multiple_of_8 d) : ∃ k : ℕ, c - d = 2 * k :=
sorry

end d_is_multiple_of_4_c_minus_d_is_multiple_of_4_c_minus_d_is_multiple_of_2_l182_182536


namespace stratified_sampling_correct_l182_182086

def num_students := 500
def num_male_students := 500
def num_female_students := 400
def ratio_male_female := num_male_students / num_female_students

def selected_male_students := 25
def selected_female_students := (selected_male_students * num_female_students) / num_male_students

theorem stratified_sampling_correct :
  selected_female_students = 20 :=
by
  sorry

end stratified_sampling_correct_l182_182086


namespace ezekiel_new_shoes_l182_182605

-- condition Ezekiel bought 3 pairs of shoes
def pairs_of_shoes : ℕ := 3

-- condition Each pair consists of 2 shoes
def shoes_per_pair : ℕ := 2

-- proving the number of new shoes Ezekiel has
theorem ezekiel_new_shoes (pairs_of_shoes shoes_per_pair : ℕ) : pairs_of_shoes * shoes_per_pair = 6 :=
by
  sorry

end ezekiel_new_shoes_l182_182605


namespace infinitely_many_digitally_divisible_integers_l182_182098

theorem infinitely_many_digitally_divisible_integers :
  ∀ n : ℕ, ∃ k : ℕ, k = (10 ^ (3 ^ n) - 1) / 9 ∧ (3 ^ n ∣ k) :=
by
  sorry

end infinitely_many_digitally_divisible_integers_l182_182098


namespace farm_own_more_horses_than_cows_after_transaction_l182_182035

theorem farm_own_more_horses_than_cows_after_transaction :
  ∀ (x : Nat), 
    3 * (3 * x - 15) = 5 * (x + 15) →
    75 - 45 = 30 :=
by
  intro x h
  -- This is a placeholder for the proof steps which we skip.
  sorry

end farm_own_more_horses_than_cows_after_transaction_l182_182035


namespace dawn_monthly_savings_l182_182181

-- Definitions for the conditions
def annual_salary := 48000
def months_in_year := 12
def saving_rate := 0.10

-- Define the monthly salary
def monthly_salary := annual_salary / months_in_year

-- Define the monthly savings
def monthly_savings := monthly_salary * saving_rate

-- The theorem to prove
theorem dawn_monthly_savings : monthly_savings = 400 :=
by
  -- Proof details skipped
  sorry

end dawn_monthly_savings_l182_182181


namespace compare_ab_1_to_a_b_two_pow_b_eq_one_div_b_sign_of_expression_l182_182918

-- Part 1
theorem compare_ab_1_to_a_b {a b : ℝ} (h1 : a^b * b^a + Real.log b / Real.log a = 0) (ha : a > 0) (hb : b > 0) : ab + 1 < a + b := sorry

-- Part 2
theorem two_pow_b_eq_one_div_b {b : ℝ} (h : 2^b * b^2 + Real.log b / Real.log 2 = 0) : 2^b = 1 / b := sorry

-- Part 3
theorem sign_of_expression {b : ℝ} (h : 2^b * b^2 + Real.log b / Real.log 2 = 0) : (2 * b + 1 - Real.sqrt 5) * (3 * b - 2) < 0 := sorry

end compare_ab_1_to_a_b_two_pow_b_eq_one_div_b_sign_of_expression_l182_182918


namespace students_height_order_valid_after_rearrangement_l182_182394
open List

variable {n : ℕ} -- number of students in each row
variable (a b : Fin n → ℝ) -- heights of students in each row

/-- Prove Gábor's observation remains valid after rearrangement: 
    each student in the back row is taller than the student in front of them.
    Given:
    - ∀ i, b i < a i (initial condition)
    - ∀ i < j, a i ≤ a j (rearrangement condition)
    Prove:
    - ∀ i, b i < a i (remains valid after rearrangement)
-/
theorem students_height_order_valid_after_rearrangement
  (h₁ : ∀ i : Fin n, b i < a i)
  (h₂ : ∀ (i j : Fin n), i < j → a i ≤ a j) :
  ∀ i : Fin n, b i < a i :=
by sorry

end students_height_order_valid_after_rearrangement_l182_182394


namespace cos_90_eq_zero_l182_182153

-- Define cosine function and specify its behavior on the unit circle.
def cos (θ : ℝ) : ℝ :=
  -- Cosine gives the x-coordinate of the point on the unit circle.
  sorry -- Definition of cosine function is assumed.

-- Statement to be proved that involves the cosine function.
theorem cos_90_eq_zero : cos (90 * π / 180) = 0 := 
by
  sorry -- Proof is omitted.

end cos_90_eq_zero_l182_182153


namespace cos_90_eq_0_l182_182157

theorem cos_90_eq_0 : Real.cos (90 * Real.pi / 180) = 0 := by
  sorry

end cos_90_eq_0_l182_182157


namespace max_det_A_l182_182194

open Real

-- Define the matrix and the determinant expression
noncomputable def A (θ : ℝ) : Matrix (Fin 3) (Fin 3) ℝ :=
  ![
    ![1, 1, 1],
    ![1, 1 + cos θ, 1],
    ![1 + sin θ, 1, 1]
  ]

-- Lean statement to prove the maximum value of the determinant of matrix A
theorem max_det_A : ∃ θ : ℝ, (Matrix.det (A θ)) ≤ 1/2 := by
  sorry

end max_det_A_l182_182194


namespace isosceles_triangle_perimeter_l182_182227

theorem isosceles_triangle_perimeter (a b : ℝ) (h1 : a = 5) (h2 : b = 11) (h3 : a = b ∨ b = b) :
  (5 + 11 + 11 = 27) := 
by {
  sorry
}

end isosceles_triangle_perimeter_l182_182227


namespace point_in_third_quadrant_l182_182809

theorem point_in_third_quadrant (m : ℝ) (h1 : m < 0) (h2 : 4 + 2 * m < 0) : m < -2 :=
by
  sorry

end point_in_third_quadrant_l182_182809


namespace fourth_term_geometric_progression_l182_182807

theorem fourth_term_geometric_progression (x : ℝ) (h : ∀ n : ℕ, 0 < n → 
  (x ≠ 0 ∧ (2 * (x) + 2 * (n - 1)) ≠ 0 ∧ (3 * (x) + 3 * (n - 1)) ≠ 0)
  → ((2 * x + 2) / x) = (3 * x + 3) / (2 * x + 2)) : 
  ∃ r : ℝ, r = -13.5 := 
by 
  sorry

end fourth_term_geometric_progression_l182_182807


namespace triangle_two_acute_angles_l182_182073

theorem triangle_two_acute_angles (A B C : ℝ) (h_triangle : A + B + C = 180) (h_pos : A > 0 ∧ B > 0 ∧ C > 0)
  (h_acute_triangle: A < 90 ∨ B < 90 ∨ C < 90): A < 90 ∧ B < 90 ∨ A < 90 ∧ C < 90 ∨ B < 90 ∧ C < 90 :=
by
  sorry

end triangle_two_acute_angles_l182_182073


namespace rancher_total_animals_l182_182895

theorem rancher_total_animals
  (H C : ℕ) (h1 : C = 5 * H) (h2 : C = 140) :
  C + H = 168 := 
sorry

end rancher_total_animals_l182_182895


namespace man_reaches_home_at_11_pm_l182_182456

theorem man_reaches_home_at_11_pm :
  let start_time := 15 -- represents 3 pm in 24-hour format
  let level_speed := 4 -- km/hr
  let uphill_speed := 3 -- km/hr
  let downhill_speed := 6 -- km/hr
  let total_distance := 12 -- km
  let level_distance := 4 -- km
  let uphill_distance := 4 -- km
  let downhill_distance := 4 -- km
  let level_time := level_distance / level_speed -- time for 4 km on level ground
  let uphill_time := uphill_distance / uphill_speed -- time for 4 km uphill
  let downhill_time := downhill_distance / downhill_speed -- time for 4 km downhill
  let total_time_one_way := level_time + uphill_time + downhill_time + level_time
  let destination_time := start_time + total_time_one_way
  let return_time := destination_time + total_time_one_way
  return_time = 23 := -- represents 11 pm in 24-hour format
by
  sorry

end man_reaches_home_at_11_pm_l182_182456


namespace find_y_l182_182808

theorem find_y (x y: ℤ) (h1: x^2 - 3 * x + 2 = y + 6) (h2: x = -4) : y = 24 :=
by
  sorry

end find_y_l182_182808


namespace initial_number_is_31_l182_182608

theorem initial_number_is_31 (N : ℕ) (h : ∃ k : ℕ, N - 10 = 21 * k) : N = 31 :=
sorry

end initial_number_is_31_l182_182608


namespace percent_children_with_both_colors_l182_182441

theorem percent_children_with_both_colors
  (F : ℕ) (C : ℕ) 
  (even_F : F % 2 = 0)
  (children_pick_two_flags : C = F / 2)
  (sixty_percent_blue : 6 * C / 10 = 6 * C / 10)
  (fifty_percent_red : 5 * C / 10 = 5 * C / 10)
  : (6 * C / 10) + (5 * C / 10) - C = C / 10 :=
by
  sorry

end percent_children_with_both_colors_l182_182441


namespace num_positive_int_values_l182_182778

theorem num_positive_int_values (N : ℕ) :
  (∃ m : ℕ, N = m ∧ m > 0 ∧ 48 % (m + 3) = 0) ↔ (N < 7) :=
sorry

end num_positive_int_values_l182_182778


namespace ellipse_hyperbola_tangent_n_values_are_62_20625_and_1_66875_l182_182610

theorem ellipse_hyperbola_tangent_n_values_are_62_20625_and_1_66875:
  let is_ellipse (x y n : ℝ) := x^2 + n*(y - 1)^2 = n
  let is_hyperbola (x y : ℝ) := x^2 - 4*(y + 3)^2 = 4
  ∃ (n1 n2 : ℝ),
    n1 = 62.20625 ∧ n2 = 1.66875 ∧
    (∀ (x y : ℝ), is_ellipse x y n1 → is_hyperbola x y → 
       is_ellipse x y n2 → is_hyperbola x y → 
       (4 + n1)*(y^2 - 2*y + 1) = 4*(y^2 + 6*y + 9) + 4 ∧ 
       ((24 - 2*n1)^2 - 4*(4 + n1)*40 = 0) ∧
       (4 + n2)*(y^2 - 2*y + 1) = 4*(y^2 + 6*y + 9) + 4 ∧ 
       ((24 - 2*n2)^2 - 4*(4 + n2)*40 = 0))
:= sorry

end ellipse_hyperbola_tangent_n_values_are_62_20625_and_1_66875_l182_182610


namespace football_goals_l182_182452

variable (A : ℚ) (G : ℚ)

theorem football_goals (A G : ℚ) 
    (h1 : G = 14 * A)
    (h2 : G + 3 = (A + 0.08) * 15) :
    G = 25.2 :=
by
  -- Proof here
  sorry

end football_goals_l182_182452


namespace average_minutes_heard_l182_182095

theorem average_minutes_heard :
  let total_audience := 200
  let duration := 90
  let percent_entire := 0.15
  let percent_slept := 0.15
  let percent_half := 0.25
  let percent_one_fourth := 0.75
  let total_entire := total_audience * percent_entire
  let total_slept := total_audience * percent_slept
  let remaining := total_audience - total_entire - total_slept
  let total_half := remaining * percent_half
  let total_one_fourth := remaining * percent_one_fourth
  let minutes_entire := total_entire * duration
  let minutes_half := total_half * (duration / 2)
  let minutes_one_fourth := total_one_fourth * (duration / 4)
  let total_minutes_heard := minutes_entire + 0 + minutes_half + minutes_one_fourth
  let average_minutes := total_minutes_heard / total_audience
  average_minutes = 33 :=
by
  sorry

end average_minutes_heard_l182_182095


namespace intersection_of_sets_l182_182972

open Set Int

theorem intersection_of_sets (S T : Set Int) (hS : S = {s | ∃ n : ℤ, s = 2 * n + 1}) (hT : T = {t | ∃ n : ℤ, t = 4 * n + 1}) : S ∩ T = T :=
  by
  sorry

end intersection_of_sets_l182_182972


namespace cos_90_eq_0_l182_182158

theorem cos_90_eq_0 : Real.cos (90 * Real.pi / 180) = 0 := by
  sorry

end cos_90_eq_0_l182_182158


namespace triangle_angle_l182_182007

variable (a b c : ℝ)
variable (C : ℝ)

theorem triangle_angle (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (h : (a^2 + b^2) * (a^2 + b^2 - c^2) = 3 * a^2 * b^2) :
  C = Real.arccos ((a^4 + b^4 - a^2 * b^2) / (2 * a * b * (a^2 + b^2))) :=
sorry

end triangle_angle_l182_182007


namespace num_positive_int_values_for_expression_is_7_l182_182782

theorem num_positive_int_values_for_expression_is_7 :
  {N : ℕ // 0 < N ∧ ∃ k : ℕ, 48 = k * (N + 3)}.card = 7 := 
sorry

end num_positive_int_values_for_expression_is_7_l182_182782


namespace no_solution_system_l182_182005

theorem no_solution_system (a : ℝ) : 
  (∀ x : ℝ, (x - 2 * a > 0) → (3 - 2 * x > x - 6) → false) ↔ a ≥ 3 / 2 :=
by
  sorry

end no_solution_system_l182_182005


namespace segment_length_l182_182076

def cbrt (x : ℝ) : ℝ := x^(1/3)

theorem segment_length (x : ℝ) 
  (h : |x - cbrt 27| = 5) : (abs ((cbrt 27 + 5) - (cbrt 27 - 5)) = 10) :=
by
  sorry

end segment_length_l182_182076


namespace intersection_S_T_eq_T_l182_182943

def S : Set ℤ := { s | ∃ n : ℤ, s = 2 * n + 1 }
def T : Set ℤ := { t | ∃ n : ℤ, t = 4 * n + 1 }

theorem intersection_S_T_eq_T : S ∩ T = T := by
  sorry

end intersection_S_T_eq_T_l182_182943


namespace B_and_C_complementary_l182_182196

def EventA (selected : List String) : Prop :=
  selected.count "boy" = 1

def EventB (selected : List String) : Prop :=
  selected.count "boy" ≥ 1

def EventC (selected : List String) : Prop :=
  selected.count "girl" = 2

theorem B_and_C_complementary :
  ∀ selected : List String,
    (selected.length = 2 ∧ (EventB selected ∨ EventC selected)) ∧ 
    (¬ (EventB selected ∧ EventC selected)) →
    (EventB selected → ¬ EventC selected) ∧ (EventC selected → ¬ EventB selected) :=
  sorry

end B_and_C_complementary_l182_182196


namespace inequality_proof_l182_182484

-- Define the conditions and the theorem statement
variables {a b c d : ℝ}

theorem inequality_proof (h1 : c < d) (h2 : a > b) (h3 : b > 0) : a - c > b - d :=
by
  sorry

end inequality_proof_l182_182484


namespace intersection_of_sets_l182_182976

open Set Int

theorem intersection_of_sets (S T : Set Int) (hS : S = {s | ∃ n : ℤ, s = 2 * n + 1}) (hT : T = {t | ∃ n : ℤ, t = 4 * n + 1}) : S ∩ T = T :=
  by
  sorry

end intersection_of_sets_l182_182976


namespace graph_not_pass_through_second_quadrant_l182_182262

theorem graph_not_pass_through_second_quadrant :
  ¬ ∃ x y : ℝ, y = 2 * x - 3 ∧ x < 0 ∧ y > 0 :=
by sorry

end graph_not_pass_through_second_quadrant_l182_182262


namespace iodine_atomic_weight_l182_182770

noncomputable def atomic_weight_of_iodine : ℝ :=
  127.01

theorem iodine_atomic_weight
  (mw_AlI3 : ℝ := 408)
  (aw_Al : ℝ := 26.98)
  (formula_mw_AlI3 : mw_AlI3 = aw_Al + 3 * atomic_weight_of_iodine) :
  atomic_weight_of_iodine = 127.01 :=
by sorry

end iodine_atomic_weight_l182_182770


namespace quadratic_has_solution_l182_182391

theorem quadratic_has_solution (a b : ℝ) : ∃ x : ℝ, (a^6 - b^6) * x^2 + 2 * (a^5 - b^5) * x + (a^4 - b^4) = 0 :=
  by sorry

end quadratic_has_solution_l182_182391


namespace cos_90_eq_0_l182_182159

theorem cos_90_eq_0 : Real.cos (90 * Real.pi / 180) = 0 := by
  sorry

end cos_90_eq_0_l182_182159


namespace problem_intersection_l182_182211

open Set

variable {x : ℝ}

def A : Set ℝ := {x | 2 * x - 5 ≥ 0}
def B : Set ℝ := {x | x^2 - 4 * x + 3 < 0}
def C : Set ℝ := {x | (5 / 2) ≤ x ∧ x < 3}

theorem problem_intersection : A ∩ B = C := by
  sorry

end problem_intersection_l182_182211


namespace incorrect_statement_c_l182_182279

-- Define even function
def is_even (f : ℝ → ℝ) := ∀ x : ℝ, f (-x) = f x

-- Define odd function
def is_odd (f : ℝ → ℝ) := ∀ x : ℝ, f (-x) = -f x

-- Function definitions
def f1 (x : ℝ) : ℝ := x^4 + x^2
def f2 (x : ℝ) : ℝ := x^3 + x^2

-- Main theorem statement
theorem incorrect_statement_c : ¬ is_odd f2 := sorry

end incorrect_statement_c_l182_182279


namespace smallest_angle_in_15_sided_polygon_arithmetic_sequence_l182_182052

theorem smallest_angle_in_15_sided_polygon_arithmetic_sequence
  (a d : ℕ) 
  (angles : Fin 15 → ℕ)
  (h_seq : ∀ i : Fin 15, angles i = a + i * d)
  (h_convex : ∀ i : Fin 15, angles i < 180)
  (h_sum : ∑ i, angles i = 2340) : 
  a = 135 := 
sorry

end smallest_angle_in_15_sided_polygon_arithmetic_sequence_l182_182052


namespace smaller_angle_at_9_am_l182_182753

-- Define the angular positions of the minute and hour hands
def minute_hand_angle (minute : Nat) : ℕ := 0  -- At the 12 position
def hour_hand_angle (hour : Nat) : ℕ := hour * 30  -- 30 degrees per hour

-- Define the function to get the smaller angle between two angles on the clock from 0 to 360 degrees
def smaller_angle (angle1 angle2 : ℕ) : ℕ :=
  let angle_diff := Int.natAbs (angle1 - angle2)
  min angle_diff (360 - angle_diff)

-- The theorem to prove
theorem smaller_angle_at_9_am : smaller_angle (minute_hand_angle 0) (hour_hand_angle 9) = 90 := sorry

end smaller_angle_at_9_am_l182_182753


namespace cos_90_eq_zero_l182_182149

theorem cos_90_eq_zero : (cos 90) = 0 := by
  -- Condition: The angle 90 degrees corresponds to the point (0, 1) on the unit circle.
  -- We claim cos 90 degrees is 0
  sorry

end cos_90_eq_zero_l182_182149


namespace four_digit_numbers_with_product_exceeds_10_l182_182633

theorem four_digit_numbers_with_product_exceeds_10 : 
  (∃ count : ℕ, 
    count = (6 * 58 * 10) ∧
    count = 3480) := 
by {
  sorry
}

end four_digit_numbers_with_product_exceeds_10_l182_182633


namespace ways_A_to_C_via_B_l182_182223

def ways_A_to_B : Nat := 2
def ways_B_to_C : Nat := 3

theorem ways_A_to_C_via_B : ways_A_to_B * ways_B_to_C = 6 := by
  sorry

end ways_A_to_C_via_B_l182_182223


namespace func_increasing_l182_182185

noncomputable def func (x : ℝ) : ℝ :=
  x^3 + x + 1

theorem func_increasing : ∀ x : ℝ, deriv func x > 0 := by
  sorry

end func_increasing_l182_182185


namespace extracurricular_popularity_order_l182_182461

def fraction_likes_drama := 9 / 28
def fraction_likes_music := 13 / 36
def fraction_likes_art := 11 / 24

theorem extracurricular_popularity_order :
  fraction_likes_art > fraction_likes_music ∧ 
  fraction_likes_music > fraction_likes_drama :=
by
  sorry

end extracurricular_popularity_order_l182_182461


namespace greatest_possible_length_l182_182560

theorem greatest_possible_length :
  ∃ (g : ℕ), g = Nat.gcd 700 (Nat.gcd 385 1295) ∧ g = 35 :=
by
  sorry

end greatest_possible_length_l182_182560


namespace num_positive_int_values_for_expression_is_7_l182_182783

theorem num_positive_int_values_for_expression_is_7 :
  {N : ℕ // 0 < N ∧ ∃ k : ℕ, 48 = k * (N + 3)}.card = 7 := 
sorry

end num_positive_int_values_for_expression_is_7_l182_182783


namespace find_constant_l182_182055

theorem find_constant (x1 x2 : ℝ) (C : ℝ) :
  x1 - x2 = 5.5 ∧
  x1 + x2 = -5 / 2 ∧
  x1 * x2 = C / 2 →
  C = -12 :=
by
  -- proof goes here
  sorry

end find_constant_l182_182055


namespace quadratic_solution_exists_for_any_a_b_l182_182389

theorem quadratic_solution_exists_for_any_a_b (a b : ℝ) : 
  ∃ x : ℝ, (a^6 - b^6)*x^2 + 2*(a^5 - b^5)*x + (a^4 - b^4) = 0 := 
by
  -- The proof would go here
  sorry

end quadratic_solution_exists_for_any_a_b_l182_182389


namespace cafeteria_ordered_red_apples_l182_182688

theorem cafeteria_ordered_red_apples
  (R : ℕ) 
  (h : (R + 17) - 10 = 32) : 
  R = 25 :=
sorry

end cafeteria_ordered_red_apples_l182_182688


namespace tan_ratio_is_7_over_3_l182_182237

open Real

theorem tan_ratio_is_7_over_3 (a b : ℝ) (h1 : sin (a + b) = 5 / 8) (h2 : sin (a - b) = 1 / 4) : (tan a / tan b) = 7 / 3 :=
by
  sorry

end tan_ratio_is_7_over_3_l182_182237


namespace compare_neg_thirds_and_halves_l182_182116

theorem compare_neg_thirds_and_halves : (-1 : ℚ) / 3 > (-1 : ℚ) / 2 :=
by
  sorry

end compare_neg_thirds_and_halves_l182_182116


namespace cos_90_eq_zero_l182_182145

theorem cos_90_eq_zero : (cos 90) = 0 := by
  -- Condition: The angle 90 degrees corresponds to the point (0, 1) on the unit circle.
  -- We claim cos 90 degrees is 0
  sorry

end cos_90_eq_zero_l182_182145


namespace problem_lean_l182_182612

variable (α : ℝ)

-- Given condition
axiom given_cond : (1 + Real.sin α) * (1 - Real.cos α) = 1

-- Proof to be proven
theorem problem_lean : (1 - Real.sin α) * (1 + Real.cos α) = 1 - Real.sin (2 * α) := by
  sorry

end problem_lean_l182_182612


namespace Charles_speed_with_music_l182_182110

theorem Charles_speed_with_music (S : ℝ) (h1 : 40 / 60 + 30 / 60 = 70 / 60) (h2 : S * (40 / 60) + 4 * (30 / 60) = 6) : S = 8 :=
by
  sorry

end Charles_speed_with_music_l182_182110


namespace intersection_eq_T_l182_182998

def S : Set ℤ := {s | ∃ n : ℤ, s = 2 * n + 1}
def T : Set ℤ := {t | ∃ n : ℤ, t = 4 * n + 1}

theorem intersection_eq_T : S ∩ T = T := sorry

end intersection_eq_T_l182_182998


namespace distinct_digits_solution_l182_182263

theorem distinct_digits_solution (A B C : ℕ)
  (h1 : A + B = 10)
  (h2 : C + A = 9)
  (h3 : B + C = 9)
  (h4 : A ≠ B)
  (h5 : B ≠ C)
  (h6 : C ≠ A)
  (h7 : 0 < A)
  (h8 : 0 < B)
  (h9 : 0 < C)
  : A = 1 ∧ B = 9 ∧ C = 8 := 
  by sorry

end distinct_digits_solution_l182_182263


namespace vertical_coordinate_intersection_l182_182692

def original_function (x : ℝ) := x^2 + 2 * x + 1

def shifted_function (x : ℝ) := (x + 3)^2 + 3

theorem vertical_coordinate_intersection :
  shifted_function 0 = 12 :=
by
  sorry

end vertical_coordinate_intersection_l182_182692


namespace hotel_towels_l182_182736

def num_rooms : Nat := 10
def people_per_room : Nat := 3
def towels_per_person : Nat := 2

theorem hotel_towels : num_rooms * people_per_room * towels_per_person = 60 :=
by
  sorry

end hotel_towels_l182_182736


namespace find_principal_l182_182436

noncomputable def compoundPrincipal (A : ℝ) (r : ℝ) (n : ℕ) (t : ℕ) : ℝ :=
  A / (1 + r / n) ^ (n * t)

theorem find_principal :
  let A := 3969
  let r := 0.05
  let n := 1
  let t := 2
  compoundPrincipal A r n t = 3600 :=
by
  sorry

end find_principal_l182_182436


namespace friend_spent_l182_182567

theorem friend_spent (x you friend total: ℝ) (h1 : total = you + friend) (h2 : friend = you + 3) (h3 : total = 11) : friend = 7 := by
  sorry

end friend_spent_l182_182567


namespace non_degenerate_ellipse_condition_l182_182595

theorem non_degenerate_ellipse_condition (x y k a : ℝ) :
  (3 * x^2 + 9 * y^2 - 12 * x + 27 * y = k) ∧
  (∃ h : ℝ, 3 * (x - h)^2 + 9 * (y + 3/2)^2 = k + 129/4) ∧
  (k > a) ↔ (a = -129 / 4) :=
by
  sorry

end non_degenerate_ellipse_condition_l182_182595


namespace total_stamps_l182_182101

-- Definitions for the conditions.
def snowflake_stamps : ℕ := 11
def truck_stamps : ℕ := snowflake_stamps + 9
def rose_stamps : ℕ := truck_stamps - 13

-- Statement to prove the total number of stamps.
theorem total_stamps : (snowflake_stamps + truck_stamps + rose_stamps) = 38 :=
by
  sorry

end total_stamps_l182_182101


namespace value_of_d_minus_2_times_e_minus_2_l182_182238

-- Define the given quadratic equation
def quadratic_eq (a b c x : ℝ) := a * x^2 + b * x + c = 0

-- State that our equation is 3x² + 4x - 7 = 0
def given_quadratic_eq := quadratic_eq 3 4 (-7)

-- Using Vieta's formulas
def sum_roots (d e : ℝ) := d + e = - (4 / 3)
def product_roots (d e : ℝ) := d * e = - (7 / 3)

-- Prove the required value
theorem value_of_d_minus_2_times_e_minus_2 (d e : ℝ) (h1 : sum_roots d e) (h2 : product_roots d e) :
  (d - 2) * (e - 2) = 13 / 3 := by
  sorry

end value_of_d_minus_2_times_e_minus_2_l182_182238


namespace coin_flip_sequences_count_l182_182084

theorem coin_flip_sequences_count : (2 ^ 16) = 65536 :=
by
  sorry

end coin_flip_sequences_count_l182_182084


namespace max_area_triangle_l182_182646

def area_max (a b c : ℝ) := 
  (1 / 2) * a * b * (Real.sin (Real.arccos ((a^2 + c^2 - b^2)/(2*a*c))))

theorem max_area_triangle (a b : ℝ) :
(∃ (A C: ℝ) (h₁: a / 2 = ℝ.cos A / (2 - ℝ.cos C)), 
  (∀ (c = 2), 
    ∀ b, (Real.cos C = (a^2 + c^2 - b^2)/(2*a*c)) →
      (Real.sin C = sqrt (1 - ((a^2 + c^2 - b^2)/(2*a*c))^2)) → 
        (area_max a a 2 = 4/3))) :=
begin
  sorry
end

end max_area_triangle_l182_182646


namespace trig_identity_example_l182_182337

theorem trig_identity_example :
  (Real.sin (36 * Real.pi / 180) * Real.cos (6 * Real.pi / 180) - 
   Real.sin (54 * Real.pi / 180) * Real.cos (84 * Real.pi / 180)) = 
  1 / 2 :=
by
  sorry

end trig_identity_example_l182_182337


namespace sum_a3_a7_l182_182010

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem sum_a3_a7 (a : ℕ → ℝ)
  (h₁ : arithmetic_sequence a)
  (h₂ : a 1 + a 9 + a 2 + a 8 = 20) :
  a 3 + a 7 = 10 :=
sorry

end sum_a3_a7_l182_182010


namespace value_of_z_l182_182659

theorem value_of_z (x y z : ℤ) (h1 : x^2 = y - 4) (h2 : x = -6) (h3 : y = z + 2) : z = 38 := 
by
  -- Proof skipped
  sorry

end value_of_z_l182_182659


namespace octagon_ratio_l182_182542

theorem octagon_ratio (total_area : ℝ) (area_below_PQ : ℝ) (triangle_base : ℝ) (triangle_height : ℝ) (XQ QY : ℝ) :
  total_area = 10 ∧
  area_below_PQ = 5 ∧
  triangle_base = 5 ∧
  triangle_height = 8 / 5 ∧
  area_below_PQ = 1 + (1 / 2) * triangle_base * triangle_height ∧
  XQ + QY = triangle_base ∧
  (1 / 2) * (XQ + QY) * triangle_height = 5
  → (XQ / QY) = 2 / 3 := 
sorry

end octagon_ratio_l182_182542


namespace intersection_eq_T_l182_182936

def S : Set ℤ := {s | ∃ n : ℤ, s = 2 * n + 1}
def T : Set ℤ := {t | ∃ n : ℤ, t = 4 * n + 1}

theorem intersection_eq_T : S ∩ T = T := 
sorry

end intersection_eq_T_l182_182936


namespace cos_90_eq_zero_l182_182151

-- Define cosine function and specify its behavior on the unit circle.
def cos (θ : ℝ) : ℝ :=
  -- Cosine gives the x-coordinate of the point on the unit circle.
  sorry -- Definition of cosine function is assumed.

-- Statement to be proved that involves the cosine function.
theorem cos_90_eq_zero : cos (90 * π / 180) = 0 := 
by
  sorry -- Proof is omitted.

end cos_90_eq_zero_l182_182151


namespace sine_minus_cosine_l182_182201

variable {α : ℝ}

theorem sine_minus_cosine (h1 : sin α + cos α = 7 / 5) (h2 : π / 4 < α) (h3 : α < π / 2) :
  sin α - cos α = 1 / 5 :=
sorry

end sine_minus_cosine_l182_182201


namespace bacteria_reach_target_l182_182286

def bacteria_growth (initial : ℕ) (target : ℕ) (doubling_time : ℕ) (delay : ℕ) : ℕ :=
  let doubling_count := Nat.log2 (target / initial)
  doubling_count * doubling_time + delay

theorem bacteria_reach_target : 
  bacteria_growth 800 25600 5 3 = 28 := by
  sorry

end bacteria_reach_target_l182_182286


namespace value_of_f_at_5_l182_182366

def f (x : ℝ) : ℝ := 4 * x + 2

theorem value_of_f_at_5 : f 5 = 22 :=
by
  sorry

end value_of_f_at_5_l182_182366


namespace S_inter_T_eq_T_l182_182965

-- Define the sets S and T
def S := { s : ℤ | ∃ n : ℤ, s = 2 * n + 1 }
def T := { t : ℤ | ∃ n : ℤ, t = 4 * n + 1 }

-- Prove that S ∩ T = T
theorem S_inter_T_eq_T : S ∩ T = T := 
by
  sorry

end S_inter_T_eq_T_l182_182965


namespace cos_90_equals_0_l182_182126

-- Define the question: cos(90 degrees)
def cos_90_degrees : ℝ := Real.cos (Real.pi / 2)

-- State that cos(90 degrees) equals 0
theorem cos_90_equals_0 : cos_90_degrees = 0 :=
by sorry

end cos_90_equals_0_l182_182126


namespace no_value_of_a_l182_182342

theorem no_value_of_a (a : ℝ) (x y : ℝ) : ¬∃ x1 x2 : ℝ, (x1 ≠ x2) ∧ (x1^2 + y^2 + 2 * x1 = abs (x1 - a) - 1) ∧ (x2^2 + y^2 + 2 * x2 = abs (x2 - a) - 1) := 
by
  sorry

end no_value_of_a_l182_182342


namespace general_term_of_sequence_l182_182548

theorem general_term_of_sequence (n : ℕ) :
  ∃ (a : ℕ → ℚ),
    a 1 = 1 / 2 ∧ 
    a 2 = -2 ∧ 
    a 3 = 9 / 2 ∧ 
    a 4 = -8 ∧ 
    a 5 = 25 / 2 ∧ 
    ∀ n, a n = (-1) ^ (n + 1) * (n ^ 2 / 2) := 
by
  sorry

end general_term_of_sequence_l182_182548


namespace inequality_one_inequality_two_l182_182202

-- Definitions of the three positive real numbers and their sum of reciprocals squared is equal to 1
variables {a b c : ℝ}
variable (h_pos : 0 < a ∧ 0 < b ∧ 0 < c)
variable (h_sum : (1 / a^2) + (1 / b^2) + (1 / c^2) = 1)

-- First proof that (1/a + 1/b + 1/c) <= sqrt(3)
theorem inequality_one (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_sum : (1 / a^2) + (1 / b^2) + (1 / c^2) = 1) :
  (1 / a) + (1 / b) + (1 / c) ≤ Real.sqrt 3 :=
sorry

-- Second proof that (a^2/b^4) + (b^2/c^4) + (c^2/a^4) >= 1
theorem inequality_two (h_pos : 0 < a ∧ 0 < b ∧ 0 < c) (h_sum : (1 / a^2) + (1 / b^2) + (1 / c^2) = 1) :
  (a^2 / b^4) + (b^2 / c^4) + (c^2 / a^4) ≥ 1 :=
sorry

end inequality_one_inequality_two_l182_182202


namespace functional_equation_solution_l182_182593

theorem functional_equation_solution {
  f : ℝ → ℝ
} (h : ∀ x y : ℝ, f ((x - y)^2) = x^2 - 2 * y * f x + (f y)^2) :
  (∀ x : ℝ, f x = x) ∨ (∀ x : ℝ, f x = x + 1) :=
sorry

end functional_equation_solution_l182_182593


namespace non_negative_sequence_l182_182686

theorem non_negative_sequence
  (a : Fin 100 → ℝ)
  (h₁ : a 0 = a 99)
  (h₂ : ∀ i : Fin 97, a i - 2 * a (i+1) + a (i+2) ≤ 0)
  (h₃ : a 0 ≥ 0) :
  ∀ i : Fin 100, a i ≥ 0 :=
by
  sorry

end non_negative_sequence_l182_182686


namespace interest_rate_per_annum_l182_182543
noncomputable def interest_rate_is_10 : ℝ := 10
theorem interest_rate_per_annum (P R : ℝ) : 
  (1200 * ((1 + R / 100)^2 - 1) - 1200 * R * 2 / 100 = 12) → P = 1200 → R = 10 := 
by sorry

end interest_rate_per_annum_l182_182543


namespace point_not_on_graph_and_others_on_l182_182276

theorem point_not_on_graph_and_others_on (y : ℝ → ℝ) (h₁ : ∀ x, y x = x / (x - 1))
  : ¬ (1 = (1 : ℝ) / ((1 : ℝ) - 1)) 
  ∧ (2 = (2 : ℝ) / ((2 : ℝ) - 1)) 
  ∧ ((-1 : ℝ) = (1/2 : ℝ) / ((1/2 : ℝ) - 1)) 
  ∧ (0 = (0 : ℝ) / ((0 : ℝ) - 1)) 
  ∧ (3/2 = (3 : ℝ) / ((3 : ℝ) - 1)) := 
sorry

end point_not_on_graph_and_others_on_l182_182276


namespace multiplicative_inverse_185_mod_341_l182_182180

theorem multiplicative_inverse_185_mod_341 :
  ∃ (b: ℕ), b ≡ 74466 [MOD 341] ∧ 185 * b ≡ 1 [MOD 341] :=
sorry

end multiplicative_inverse_185_mod_341_l182_182180


namespace cube_side_length_l182_182815

def cube_volume (side : ℝ) : ℝ := side ^ 3

theorem cube_side_length (volume : ℝ) (h : volume = 729) : ∃ (side : ℝ), side = 9 ∧ cube_volume side = volume :=
by
  sorry

end cube_side_length_l182_182815


namespace calculate_exponent_product_l182_182904

theorem calculate_exponent_product :
  (2^0.5) * (2^0.3) * (2^0.2) * (2^0.1) * (2^0.9) = 4 :=
by
  sorry

end calculate_exponent_product_l182_182904


namespace total_cupcakes_needed_l182_182554

-- Definitions based on conditions
def cupcakes_per_event : ℝ := 96.0
def number_of_events : ℝ := 8.0

-- Theorem based on the question and the correct answer
theorem total_cupcakes_needed : (cupcakes_per_event * number_of_events) = 768.0 :=
by 
  sorry

end total_cupcakes_needed_l182_182554


namespace find_positive_integer_n_l182_182485

theorem find_positive_integer_n (S : ℕ → ℚ) (hS : ∀ n, S n = n / (n + 1))
  (h : ∃ n : ℕ, S n * S (n + 1) = 3 / 4) : 
  ∃ n : ℕ, n = 6 := 
by {
  sorry
}

end find_positive_integer_n_l182_182485


namespace weight_loss_l182_182376

def initial_weight : ℕ := 69
def current_weight : ℕ := 34

theorem weight_loss :
  initial_weight - current_weight = 35 :=
by
  sorry

end weight_loss_l182_182376


namespace _l182_182271

lemma power_of_a_point_theorem (AP BP CP DP : ℝ) (hAP : AP = 5) (hCP : CP = 2) (h_theorem : AP * BP = CP * DP) :
  BP / DP = 2 / 5 :=
by
  sorry

end _l182_182271


namespace more_roses_than_orchids_l182_182428

theorem more_roses_than_orchids (roses orchids : ℕ) (h1 : roses = 12) (h2 : orchids = 2) : roses - orchids = 10 := by
  sorry

end more_roses_than_orchids_l182_182428


namespace bob_day3_miles_l182_182588

noncomputable def total_miles : ℕ := 70
noncomputable def day1_miles : ℕ := total_miles * 20 / 100
noncomputable def remaining_after_day1 : ℕ := total_miles - day1_miles
noncomputable def day2_miles : ℕ := remaining_after_day1 * 50 / 100
noncomputable def remaining_after_day2 : ℕ := remaining_after_day1 - day2_miles
noncomputable def day3_miles : ℕ := remaining_after_day2

theorem bob_day3_miles : day3_miles = 28 :=
by
  -- Insert proof here
  sorry

end bob_day3_miles_l182_182588


namespace solve_rational_inequality_l182_182535

theorem solve_rational_inequality :
  {x : ℝ | (9*x^2 + 18 * x - 60) / ((3 * x - 4) * (x + 5)) < 4} =
  {x : ℝ | (-10 < x ∧ x < -5) ∨ (2/3 < x ∧ x < 4/3) ∨ (4/3 < x)} :=
by
  sorry

end solve_rational_inequality_l182_182535


namespace segment_length_l182_182075

theorem segment_length (x : ℝ) (y : ℝ) (u : ℝ) (v : ℝ) :
  (|x - u| = 5 ∧ |y - u| = 5 ∧ u = √[3]{27} ∧ v = √[3]{27}) → (|x - y| = 10) :=
by
  sorry

end segment_length_l182_182075


namespace jane_baking_time_l182_182648

-- Definitions based on the conditions
variables (J : ℝ) (J_time : J > 0) -- J is the time it takes Jane to bake cakes individually
variables (Roy_time : 5 > 0) -- Roy can bake cakes in 5 hours
variables (together_time : 2 > 0) -- They work together for 2 hours
variables (remaining_time : 0.4 > 0) -- Jane completes the remaining task in 0.4 hours alone

-- Lean statement to prove Jane's individual baking time
theorem jane_baking_time : 
  (2 * (1 / J + 1 / 5) + 0.4 * (1 / J) = 1) → 
  J = 4 :=
by 
  sorry

end jane_baking_time_l182_182648


namespace train_time_first_platform_correct_l182_182745

-- Definitions
variables (L_train L_first_plat L_second_plat : ℕ) (T_second : ℕ) (T_first : ℕ)

-- Given conditions
def length_train := 350
def length_first_platform := 100
def length_second_platform := 250
def time_second_platform := 20
def expected_time_first_platform := 15

-- Derived values
def total_distance_second_platform := length_train + length_second_platform
def speed := total_distance_second_platform / time_second_platform
def total_distance_first_platform := length_train + length_first_platform
def time_first_platform := total_distance_first_platform / speed

-- Proof Statement
theorem train_time_first_platform_correct : 
  time_first_platform = expected_time_first_platform :=
  by
  sorry

end train_time_first_platform_correct_l182_182745


namespace cos_90_eq_0_l182_182130

theorem cos_90_eq_0 : Real.cos (Float.pi / 2) = 0 := 
sorry

end cos_90_eq_0_l182_182130


namespace rational_operations_l182_182039

theorem rational_operations (a b n p : ℤ) (hb : b ≠ 0) (hp : p ≠ 0) (hn : n ≠ 0) :
  (∃ q : ℚ, q = (a : ℚ) / b + (n : ℚ) / p) ∧
  (∃ q : ℚ, q = (a : ℚ) / b - (n : ℚ) / p) ∧
  (∃ q : ℚ, q = (a : ℚ) / b * (n : ℚ) / p) ∧
  (∃ q : ℚ, q = (a : ℚ) / b / ((n : ℚ) / p)) :=
by 
  sorry

end rational_operations_l182_182039


namespace prob_is_correct_l182_182575

def total_balls : ℕ := 500
def white_balls : ℕ := 200
def green_balls : ℕ := 100
def yellow_balls : ℕ := 70
def blue_balls : ℕ := 50
def red_balls : ℕ := 30
def purple_balls : ℕ := 20
def orange_balls : ℕ := 30

noncomputable def probability_green_yellow_blue : ℚ :=
  (green_balls + yellow_balls + blue_balls) / total_balls

theorem prob_is_correct :
  probability_green_yellow_blue = 0.44 := 
  by
  sorry

end prob_is_correct_l182_182575


namespace jo_page_an_hour_ago_l182_182826

variables (total_pages current_page hours_left : ℕ)
variables (steady_reading_rate : ℕ)
variables (page_an_hour_ago : ℕ)

-- Conditions
def conditions := 
  steady_reading_rate * hours_left = total_pages - current_page ∧
  total_pages = 210 ∧
  current_page = 90 ∧
  hours_left = 4 ∧
  page_an_hour_ago = current_page - steady_reading_rate

-- Theorem to prove that Jo was on page 60 an hour ago
theorem jo_page_an_hour_ago (h : conditions total_pages current_page hours_left steady_reading_rate page_an_hour_ago) : 
  page_an_hour_ago = 60 :=
sorry

end jo_page_an_hour_ago_l182_182826


namespace shaded_area_proof_l182_182011

-- Given Definitions
def rectangle_area (length : ℕ) (width : ℕ) : ℕ := length * width
def triangle_area (base : ℕ) (height : ℕ) : ℕ := (base * height) / 2

-- Conditions
def grid_area : ℕ :=
  rectangle_area 2 3 + rectangle_area 3 4 + rectangle_area 4 5

def unshaded_triangle_area : ℕ := triangle_area 12 4

-- Question
def shaded_area : ℕ := grid_area - unshaded_triangle_area

-- Proof statement
theorem shaded_area_proof : shaded_area = 14 := by
  sorry

end shaded_area_proof_l182_182011


namespace trucks_in_yard_l182_182506

/-- The number of trucks in the yard is 23, given the conditions. -/
theorem trucks_in_yard (T : ℕ) (H1 : ∃ n : ℕ, n > 0)
  (H2 : ∃ k : ℕ, k = 5 * T)
  (H3 : T + 5 * T = 140) : T = 23 :=
sorry

end trucks_in_yard_l182_182506


namespace intersection_eq_T_l182_182930

def S : Set ℤ := {s | ∃ n : ℤ, s = 2 * n + 1}
def T : Set ℤ := {t | ∃ n : ℤ, t = 4 * n + 1}

theorem intersection_eq_T : S ∩ T = T := 
sorry

end intersection_eq_T_l182_182930


namespace cos_pi_half_eq_zero_l182_182172

theorem cos_pi_half_eq_zero : Real.cos (Float.pi / 2) = 0 :=
by
  sorry

end cos_pi_half_eq_zero_l182_182172


namespace cat_food_problem_l182_182320

noncomputable def L : ℝ
noncomputable def S : ℝ

theorem cat_food_problem (L S : ℝ) (h : L + 4 * S = 14) : L + 3 * S ≥ 11 :=
by 
  sorry

end cat_food_problem_l182_182320


namespace intersection_of_sets_l182_182978

open Set Int

theorem intersection_of_sets (S T : Set Int) (hS : S = {s | ∃ n : ℤ, s = 2 * n + 1}) (hT : T = {t | ∃ n : ℤ, t = 4 * n + 1}) : S ∩ T = T :=
  by
  sorry

end intersection_of_sets_l182_182978


namespace distinct_arrangements_of_beads_l182_182642

noncomputable def factorial (n : Nat) : Nat := if h : n = 0 then 1 else n * factorial (n - 1)

theorem distinct_arrangements_of_beads : 
  ∃ (arrangements : Nat), arrangements = factorial 8 / (8 * 2) ∧ arrangements = 2520 := 
by
  -- Sorry to skip the proof, only requiring the statement.
  sorry

end distinct_arrangements_of_beads_l182_182642


namespace cos_90_deg_eq_zero_l182_182139

noncomputable def cos_90_degrees : ℝ :=
cos (90 * real.pi / 180)

theorem cos_90_deg_eq_zero : cos_90_degrees = 0 := 
by
  -- Definitions related to the conditions
  let point_0 := (1 : ℝ, 0 : ℝ)
  let point_90 := (0 : ℝ, 1 : ℝ)

  -- Rotating the point (1, 0) by 90 degrees counterclockwise results in the point (0, 1)
  have h_rotate_90 : (real.cos (90 * real.pi / 180), real.sin (90 * real.pi / 180)) = point_90 :=
    by
      -- Rotation logic is abstracted out in this property of cos and sin
      have h_cos_90 : real.cos (real.pi / 2) = 0 := by sorry
      have h_sin_90 : real.sin (real.pi / 2) = 1 := by sorry
      exact ⟨h_cos_90, h_sin_90⟩

  -- By definition, cos 90 degrees is the x-coordinate of the rotated point
  show cos_90_degrees = 0,
  from congr_arg Prod.fst h_rotate_90

end cos_90_deg_eq_zero_l182_182139


namespace cos_90_equals_0_l182_182127

-- Define the question: cos(90 degrees)
def cos_90_degrees : ℝ := Real.cos (Real.pi / 2)

-- State that cos(90 degrees) equals 0
theorem cos_90_equals_0 : cos_90_degrees = 0 :=
by sorry

end cos_90_equals_0_l182_182127


namespace z_squared_in_second_quadrant_l182_182395
open Complex Real

noncomputable def z : ℂ := exp (π * I / 3)

theorem z_squared_in_second_quadrant : (z^2).re < 0 ∧ (z^2).im > 0 :=
by
  sorry

end z_squared_in_second_quadrant_l182_182395


namespace positive_difference_of_numbers_l182_182690

theorem positive_difference_of_numbers :
  ∃ x y : ℕ, x + y = 50 ∧ 3 * y - 4 * x = 10 ∧ y - x = 10 :=
by
  sorry

end positive_difference_of_numbers_l182_182690


namespace cos_pi_half_eq_zero_l182_182170

theorem cos_pi_half_eq_zero : Real.cos (Float.pi / 2) = 0 :=
by
  sorry

end cos_pi_half_eq_zero_l182_182170


namespace marching_band_total_weight_l182_182821

def weight_trumpets := 5
def weight_clarinets := 5
def weight_trombones := 10
def weight_tubas := 20
def weight_drums := 15

def count_trumpets := 6
def count_clarinets := 9
def count_trombones := 8
def count_tubas := 3
def count_drums := 2

theorem marching_band_total_weight :
  (count_trumpets * weight_trumpets) + (count_clarinets * weight_clarinets) + (count_trombones * weight_trombones) + 
  (count_tubas * weight_tubas) + (count_drums * weight_drums) = 245 :=
by
  sorry

end marching_band_total_weight_l182_182821


namespace cos_90_eq_0_l182_182133

theorem cos_90_eq_0 : Real.cos (Float.pi / 2) = 0 := 
sorry

end cos_90_eq_0_l182_182133


namespace simplify_expression_l182_182674

theorem simplify_expression (x : ℚ) (h1 : x ≠ 3) (h2 : x ≠ 2) (h3 : x ≠ 4) (h4 : x ≠ 5) :
  ( (x^2 - 4*x + 3) / (x^2 - 6*x + 9) ) / ( (x^2 - 6*x + 8) / (x^2 - 8*x + 15) ) =
  ( (x - 1) * (x - 5) ) / ( (x - 3) * (x - 4) * (x - 2) ) :=
by
  sorry

end simplify_expression_l182_182674


namespace total_number_of_fish_l182_182109

theorem total_number_of_fish :
  let goldfish := 8
  let angelfish := goldfish + 4
  let guppies := 2 * angelfish
  let tetras := goldfish - 3
  let bettas := tetras + 5
  goldfish + angelfish + guppies + tetras + bettas = 59 := by
  -- Provide the proof here.
  sorry

end total_number_of_fish_l182_182109


namespace marathon_fraction_l182_182905

theorem marathon_fraction :
  ∃ (f : ℚ), (2 * 7) = (6 + (6 + 6 * f)) ∧ f = 1 / 3 :=
by 
  sorry

end marathon_fraction_l182_182905


namespace expected_turns_formula_l182_182451

noncomputable def expected_turns (n : ℕ) : ℝ :=
  n +  1 / 2 - (n - 1 / 2) * (1 / Real.sqrt (Real.pi * (n - 1)))

theorem expected_turns_formula (n : ℕ) (h : n ≥ 1) :
  expected_turns n = n + 1 / 2 - (n - 1 / 2) * (1 / Real.sqrt (Real.pi * (n - 1))) :=
by
  sorry

end expected_turns_formula_l182_182451


namespace find_interest_rate_l182_182385

theorem find_interest_rate
  (P A : ℝ) (n t : ℕ) (r : ℝ)
  (hP : P = 100)
  (hA : A = 121.00000000000001)
  (hn : n = 2)
  (ht : t = 1)
  (compound_interest : A = P * (1 + r / n) ^ (n * t)) :
  r = 0.2 :=
by
  sorry

end find_interest_rate_l182_182385


namespace cos_90_eq_zero_l182_182164

def point_after_rotation (θ : ℝ) : ℝ × ℝ :=
  let x := cos θ
  let y := sin θ
  (x, y)

theorem cos_90_eq_zero : cos (real.pi / 2) = 0 :=
  sorry

end cos_90_eq_zero_l182_182164


namespace intersection_of_sets_l182_182975

open Set Int

theorem intersection_of_sets (S T : Set Int) (hS : S = {s | ∃ n : ℤ, s = 2 * n + 1}) (hT : T = {t | ∃ n : ℤ, t = 4 * n + 1}) : S ∩ T = T :=
  by
  sorry

end intersection_of_sets_l182_182975


namespace intersection_of_sets_l182_182984

open Set Int

theorem intersection_of_sets (S T : Set Int) (hS : S = {s | ∃ n : ℤ, s = 2 * n + 1}) (hT : T = {t | ∃ n : ℤ, t = 4 * n + 1}) : S ∩ T = T :=
  by
  sorry

end intersection_of_sets_l182_182984


namespace S_inter_T_eq_T_l182_182962

-- Define the sets S and T
def S := { s : ℤ | ∃ n : ℤ, s = 2 * n + 1 }
def T := { t : ℤ | ∃ n : ℤ, t = 4 * n + 1 }

-- Prove that S ∩ T = T
theorem S_inter_T_eq_T : S ∩ T = T := 
by
  sorry

end S_inter_T_eq_T_l182_182962


namespace virginia_more_years_l182_182559

variable {V A D x : ℕ}

theorem virginia_more_years (h1 : V + A + D = 75) (h2 : D = 34) (h3 : V = A + x) (h4 : V = D - x) : x = 9 :=
by
  sorry

end virginia_more_years_l182_182559


namespace min_value_of_A2_minus_B2_nonneg_l182_182239

noncomputable def A (x y z : ℝ) : ℝ :=
  Real.sqrt (x + 4) + Real.sqrt (y + 7) + Real.sqrt (z + 13)

noncomputable def B (x y z : ℝ) : ℝ :=
  Real.sqrt (x + 3) + Real.sqrt (y + 3) + Real.sqrt (z + 3)

theorem min_value_of_A2_minus_B2_nonneg (x y z : ℝ) (hx : 0 ≤ x) (hy : 0 ≤ y) (hz : 0 ≤ z) : 
  (A x y z) ^ 2 - (B x y z) ^ 2 ≥ 36 :=
by
  sorry

end min_value_of_A2_minus_B2_nonneg_l182_182239


namespace distance_from_P_to_AB_l182_182236

-- Let \(ABC\) be an isosceles triangle where \(AB\) is the base. 
-- An altitude from vertex \(C\) to base \(AB\) measures 6 units.
-- A line drawn through a point \(P\) inside the triangle, parallel to base \(AB\), 
-- divides the triangle into two regions of equal area.
-- The vertex angle at \(C\) is a right angle.
-- Prove that the distance from \(P\) to \(AB\) is 3 units.

theorem distance_from_P_to_AB :
  ∀ (A B C P : Type)
    (distance_AB distance_AC distance_BC : ℝ)
    (is_isosceles : distance_AC = distance_BC)
    (right_angle_C : distance_AC^2 + distance_BC^2 = distance_AB^2)
    (altitude_C : distance_BC = 6)
    (line_through_P_parallel_to_AB : ∃ (P_x : ℝ), 0 < P_x ∧ P_x < distance_BC),
  ∃ (distance_P_to_AB : ℝ), distance_P_to_AB = 3 :=
by
  sorry

end distance_from_P_to_AB_l182_182236


namespace second_reduction_percentage_l182_182407

variable (P : ℝ) -- Original price
variable (x : ℝ) -- Second reduction percentage

-- Condition 1: After a 25% reduction
def first_reduction (P : ℝ) : ℝ := 0.75 * P

-- Condition 3: Combined reduction equivalent to 47.5%
def combined_reduction (P : ℝ) : ℝ := 0.525 * P

-- Question: Given the conditions, prove that the second reduction is 0.3
theorem second_reduction_percentage (P : ℝ) (x : ℝ) :
  (1 - x) * first_reduction P = combined_reduction P → x = 0.3 :=
by
  intro h
  sorry

end second_reduction_percentage_l182_182407


namespace final_expression_in_simplest_form_l182_182213

variable (x : ℝ)

theorem final_expression_in_simplest_form : 
  ((3 * x + 6 - 5 * x + 10) / 5) = (-2 / 5) * x + 16 / 5 :=
by
  sorry

end final_expression_in_simplest_form_l182_182213


namespace cat_food_sufficiency_l182_182326

variable (L S : ℝ)

theorem cat_food_sufficiency (h : L + 4 * S = 14) : L + 3 * S ≥ 11 :=
sorry

end cat_food_sufficiency_l182_182326


namespace color_changes_probability_l182_182578

-- Define the durations of the traffic lights
def green_duration := 40
def yellow_duration := 5
def red_duration := 45

-- Define the total cycle duration
def total_cycle_duration := green_duration + yellow_duration + red_duration

-- Define the duration of the interval Mary watches
def watch_duration := 4

-- Define the change windows where the color changes can be witnessed
def change_windows :=
  [green_duration - watch_duration,
   green_duration + yellow_duration - watch_duration,
   green_duration + yellow_duration + red_duration - watch_duration]

-- Define the total change window duration
def total_change_window_duration := watch_duration * (change_windows.length)

-- Calculate the probability of witnessing a change
def probability_witnessing_change := (total_change_window_duration : ℚ) / total_cycle_duration

-- The theorem to prove
theorem color_changes_probability :
  probability_witnessing_change = 2 / 15 := by sorry

end color_changes_probability_l182_182578


namespace club_additional_members_l182_182719

theorem club_additional_members (current_members : ℕ) (additional_members : ℕ) 
  (h1 : current_members = 10) 
  (h2 : additional_members = 5 + 2 * current_members - current_members) : 
  additional_members = 15 :=
by 
  rw [h1] at h2
  norm_num at h2
  exact h2

end club_additional_members_l182_182719


namespace sum_of_coeffs_l182_182857

-- Define the polynomial with real coefficients
def poly (p q r s : ℝ) : Polynomial ℝ := Polynomial.C 1 + Polynomial.C p * Polynomial.X + Polynomial.C q * Polynomial.X^2 + Polynomial.C r * Polynomial.X^3 + Polynomial.C s * Polynomial.X^4

-- Given conditions
def g (x : ℂ) : Polynomial ℂ := x^4 + p * x^3 + q * x^2 + r * x + s

theorem sum_of_coeffs (p q r s : ℝ)
  (h1 : g (Complex.I * 3) = 0)
  (h2 : g (1 + 2 * Complex.I) = 0) :
  p + q + r + s = -41 :=
sorry

end sum_of_coeffs_l182_182857


namespace weekly_caloric_deficit_l182_182016

-- Define the conditions
def daily_calories (day : String) : Nat :=
  if day = "Saturday" then 3500 else 2500

def daily_burn : Nat := 3000

-- Define the total calories consumed in a week
def total_weekly_consumed : Nat :=
  (2500 * 6) + 3500

-- Define the total calories burned in a week
def total_weekly_burned : Nat :=
  daily_burn * 7

-- Define the weekly deficit
def weekly_deficit : Nat :=
  total_weekly_burned - total_weekly_consumed

-- The proof goal
theorem weekly_caloric_deficit : weekly_deficit = 2500 :=
by
  -- Proof steps would go here; however, per instructions, we use sorry
  sorry

end weekly_caloric_deficit_l182_182016


namespace c_share_correct_l182_182707

def investment_a : ℕ := 5000
def investment_b : ℕ := 15000
def investment_c : ℕ := 30000
def total_profit : ℕ := 5000

def total_investment : ℕ := investment_a + investment_b + investment_c
def c_ratio : ℚ := investment_c / total_investment
def c_share : ℚ := total_profit * c_ratio

theorem c_share_correct : c_share = 3000 := by
  sorry

end c_share_correct_l182_182707


namespace intersection_eq_T_l182_182937

def S : Set ℤ := {s | ∃ n : ℤ, s = 2 * n + 1}
def T : Set ℤ := {t | ∃ n : ℤ, t = 4 * n + 1}

theorem intersection_eq_T : S ∩ T = T := 
sorry

end intersection_eq_T_l182_182937


namespace intersection_S_T_eq_T_l182_182956

def S : Set ℤ := { s | ∃ n : ℤ, s = 2 * n + 1 }
def T : Set ℤ := { t | ∃ n : ℤ, t = 4 * n + 1 }

theorem intersection_S_T_eq_T : S ∩ T = T := by
  sorry

end intersection_S_T_eq_T_l182_182956


namespace club_members_addition_l182_182724

theorem club_members_addition
  (current_members : ℕ := 10)
  (desired_members : ℕ := 2 * current_members + 5)
  (additional_members : ℕ := desired_members - current_members) :
  additional_members = 15 :=
by
  -- proof placeholder
  sorry

end club_members_addition_l182_182724


namespace cos_90_equals_0_l182_182129

-- Define the question: cos(90 degrees)
def cos_90_degrees : ℝ := Real.cos (Real.pi / 2)

-- State that cos(90 degrees) equals 0
theorem cos_90_equals_0 : cos_90_degrees = 0 :=
by sorry

end cos_90_equals_0_l182_182129


namespace molly_age_l182_182553

theorem molly_age
  (avg_age : ℕ)
  (hakimi_age : ℕ)
  (jared_age : ℕ)
  (molly_age : ℕ)
  (h1 : avg_age = 40)
  (h2 : hakimi_age = 40)
  (h3 : jared_age = hakimi_age + 10)
  (h4 : 3 * avg_age = hakimi_age + jared_age + molly_age) :
  molly_age = 30 :=
by
  sorry

end molly_age_l182_182553


namespace paint_red_and_cut_then_count_l182_182063

def initial_cube_side_length : ℕ := 4

def cube_painted_faces (side_length : ℕ) : Prop :=
∀ (f : Fin 6 → set (Fin₃ × Fin₃ × Fin₃)), 
  (∀ p : Fin₃ × Fin₃ × Fin₃, ∃ f₁, f p) → 
  side_length = initial_cube_side_length

def cut_into_small_cubes (side_length small_cube_size : ℕ) : Prop :=
side_length % small_cube_size = 0

noncomputable def number_of_cubes_with_at_least_two_red_faces : ℕ :=
56

theorem paint_red_and_cut_then_count :
  ∀ (n : ℕ), cube_painted_faces n ∧ cut_into_small_cubes n 1 → n = initial_cube_side_length →
  number_of_cubes_with_at_least_two_red_faces = 56 :=
sorry

end paint_red_and_cut_then_count_l182_182063


namespace mark_candy_bars_consumption_l182_182013

theorem mark_candy_bars_consumption 
  (recommended_intake : ℕ := 150)
  (soft_drink_calories : ℕ := 2500)
  (soft_drink_added_sugar_percent : ℕ := 5)
  (candy_bar_added_sugar_calories : ℕ := 25)
  (exceeded_percentage : ℕ := 100)
  (actual_intake := recommended_intake + (recommended_intake * exceeded_percentage / 100))
  (soft_drink_added_sugar := soft_drink_calories * soft_drink_added_sugar_percent / 100)
  (candy_bars_added_sugar := actual_intake - soft_drink_added_sugar)
  (number_of_bars := candy_bars_added_sugar / candy_bar_added_sugar_calories) : 
  number_of_bars = 7 := 
by
  sorry

end mark_candy_bars_consumption_l182_182013


namespace total_payment_is_correct_l182_182334

-- Define the number of friends
def number_of_friends : ℕ := 7

-- Define the amount each friend paid
def amount_per_friend : ℝ := 70.0

-- Define the total amount paid
def total_amount_paid : ℝ := number_of_friends * amount_per_friend

-- Prove that the total amount paid is 490.0
theorem total_payment_is_correct : total_amount_paid = 490.0 := by 
  -- Here, the proof would be filled in
  sorry

end total_payment_is_correct_l182_182334


namespace max_combined_subject_marks_l182_182750

theorem max_combined_subject_marks :
  let total_marks_math := (130 + 14) / 0.36,
      total_marks_physics := (120 + 20) / 0.40,
      total_marks_chemistry := (160 + 10) / 0.45,
      max_total_marks := total_marks_math + total_marks_physics + total_marks_chemistry in
  ⌊(total_marks_math + total_marks_physics + total_marks_chemistry)⌋ = 1127 :=
by
  -- The proof should be written here
  sorry

end max_combined_subject_marks_l182_182750


namespace packs_needed_is_six_l182_182532

variable (l_bedroom l_bathroom l_kitchen l_basement : ℕ)

def total_bulbs_needed := l_bedroom + l_bathroom + l_kitchen + l_basement
def garage_bulbs_needed := total_bulbs_needed / 2
def total_bulbs_with_garage := total_bulbs_needed + garage_bulbs_needed
def packs_needed := total_bulbs_with_garage / 2

theorem packs_needed_is_six
    (h1 : l_bedroom = 2)
    (h2 : l_bathroom = 1)
    (h3 : l_kitchen = 1)
    (h4 : l_basement = 4) :
    packs_needed l_bedroom l_bathroom l_kitchen l_basement = 6 := by
  sorry

end packs_needed_is_six_l182_182532


namespace total_football_games_l182_182864

theorem total_football_games (months : ℕ) (games_per_month : ℕ) (season_length : months = 17 ∧ games_per_month = 19) :
  (months * games_per_month) = 323 :=
by
  sorry

end total_football_games_l182_182864


namespace percent_sales_other_l182_182255

theorem percent_sales_other (percent_notebooks : ℕ) (percent_markers : ℕ) (h1 : percent_notebooks = 42) (h2 : percent_markers = 26) :
    100 - (percent_notebooks + percent_markers) = 32 := by
  sorry

end percent_sales_other_l182_182255


namespace molecularWeight_correct_l182_182432

noncomputable def molecularWeight (nC nH nO nN: ℤ) 
    (wC wH wO wN : ℚ) : ℚ := nC * wC + nH * wH + nO * wO + nN * wN

theorem molecularWeight_correct : 
    molecularWeight 5 12 3 1 12.01 1.008 16.00 14.01 = 134.156 := by
  sorry

end molecularWeight_correct_l182_182432


namespace intersection_S_T_eq_T_l182_182999

open Set

def S : Set ℤ := {s | ∃ n : ℤ, s = 2 * n + 1 }
def T : Set ℤ := {t | ∃ n : ℤ, t = 4 * n + 1 }

theorem intersection_S_T_eq_T : S ∩ T = T := 
by 
  sorry

end intersection_S_T_eq_T_l182_182999


namespace find_a_l182_182637

noncomputable def quadratic_function (a : ℝ) (x : ℝ) : ℝ :=
  x^2 - 2*(a-1)*x + 2

theorem find_a (a : ℝ) :
  (∀ x1 x2 : ℝ, x1 ≤ 2 → 2 ≤ x2 → quadratic_function a x1 ≥ quadratic_function a 2 ∧ quadratic_function a 2 ≤ quadratic_function a x2) →
  a = 3 :=
by
  sorry

end find_a_l182_182637


namespace cos_90_deg_eq_zero_l182_182136

noncomputable def cos_90_degrees : ℝ :=
cos (90 * real.pi / 180)

theorem cos_90_deg_eq_zero : cos_90_degrees = 0 := 
by
  -- Definitions related to the conditions
  let point_0 := (1 : ℝ, 0 : ℝ)
  let point_90 := (0 : ℝ, 1 : ℝ)

  -- Rotating the point (1, 0) by 90 degrees counterclockwise results in the point (0, 1)
  have h_rotate_90 : (real.cos (90 * real.pi / 180), real.sin (90 * real.pi / 180)) = point_90 :=
    by
      -- Rotation logic is abstracted out in this property of cos and sin
      have h_cos_90 : real.cos (real.pi / 2) = 0 := by sorry
      have h_sin_90 : real.sin (real.pi / 2) = 1 := by sorry
      exact ⟨h_cos_90, h_sin_90⟩

  -- By definition, cos 90 degrees is the x-coordinate of the rotated point
  show cos_90_degrees = 0,
  from congr_arg Prod.fst h_rotate_90

end cos_90_deg_eq_zero_l182_182136


namespace trader_loss_percent_l182_182219

noncomputable def CP1 : ℝ := 325475 / 1.13
noncomputable def CP2 : ℝ := 325475 / 0.87
noncomputable def TCP : ℝ := CP1 + CP2
noncomputable def TSP : ℝ := 325475 * 2
noncomputable def profit_or_loss : ℝ := TSP - TCP
noncomputable def profit_or_loss_percent : ℝ := (profit_or_loss / TCP) * 100

theorem trader_loss_percent : profit_or_loss_percent = -1.684 := by 
  sorry

end trader_loss_percent_l182_182219


namespace algorithm_correct_l182_182266

def algorithm_output (x : Int) : Int :=
  let y := Int.natAbs x
  (2 ^ y) - y

theorem algorithm_correct : 
  algorithm_output (-3) = 5 :=
  by sorry

end algorithm_correct_l182_182266


namespace boys_without_calculators_l182_182244

theorem boys_without_calculators :
    ∀ (total_boys students_with_calculators girls_with_calculators : ℕ),
    total_boys = 16 →
    students_with_calculators = 22 →
    girls_with_calculators = 13 →
    total_boys - (students_with_calculators - girls_with_calculators) = 7 :=
by
  intros
  sorry

end boys_without_calculators_l182_182244


namespace chocolate_bar_cost_l182_182339

-- Definitions based on the conditions given in the problem.
def total_bars : ℕ := 7
def remaining_bars : ℕ := 4
def total_money : ℚ := 9
def bars_sold : ℕ := total_bars - remaining_bars
def cost_per_bar := total_money / bars_sold

-- The theorem that needs to be proven.
theorem chocolate_bar_cost : cost_per_bar = 3 := by
  -- proof placeholder
  sorry

end chocolate_bar_cost_l182_182339


namespace benny_birthday_money_l182_182466

-- Define conditions
def spent_on_gear : ℕ := 47
def left_over : ℕ := 32

-- Define the total amount Benny received
def total_money_received : ℕ := 79

-- Theorem statement
theorem benny_birthday_money (spent_on_gear : ℕ) (left_over : ℕ) : spent_on_gear + left_over = total_money_received :=
by
  sorry

end benny_birthday_money_l182_182466


namespace no_sum_of_three_squares_l182_182250

theorem no_sum_of_three_squares (n : ℤ) (h : n % 8 = 7) : 
  ¬ ∃ a b c : ℤ, a^2 + b^2 + c^2 = n :=
by 
sorry

end no_sum_of_three_squares_l182_182250


namespace jamies_father_days_to_lose_weight_l182_182825

def calories_per_pound : ℕ := 3500
def pounds_to_lose : ℕ := 5
def calories_burned_per_day : ℕ := 2500
def calories_consumed_per_day : ℕ := 2000
def net_calories_burned_per_day : ℕ := calories_burned_per_day - calories_consumed_per_day
def total_calories_to_burn : ℕ := pounds_to_lose * calories_per_pound
def days_to_burn_calories := total_calories_to_burn / net_calories_burned_per_day

theorem jamies_father_days_to_lose_weight : days_to_burn_calories = 35 := by
  sorry

end jamies_father_days_to_lose_weight_l182_182825


namespace part1_part2_l182_182025

variables (a b c : ℝ)
-- Assuming a, b, c are positive and satisfy the given equation
variable (h1 : 0 < a)
variable (h2 : 0 < b)
variable (h3 : 0 < c)
variable (h_eq : 4 * a ^ 2 + b ^ 2 + 16 * c ^ 2 = 1)

-- Statement for the first part: 0 < ab < 1/4
theorem part1 : 0 < a * b ∧ a * b < 1 / 4 :=
  sorry

-- Statement for the second part: 1/a² + 1/b² + 1/(4abc²) > 49
theorem part2 : 1 / (a ^ 2) + 1 / (b ^ 2) + 1 / (4 * a * b * c ^ 2) > 49 :=
  sorry

end part1_part2_l182_182025


namespace S_inter_T_eq_T_l182_182958

-- Define the sets S and T
def S := { s : ℤ | ∃ n : ℤ, s = 2 * n + 1 }
def T := { t : ℤ | ∃ n : ℤ, t = 4 * n + 1 }

-- Prove that S ∩ T = T
theorem S_inter_T_eq_T : S ∩ T = T := 
by
  sorry

end S_inter_T_eq_T_l182_182958


namespace problem_statement_l182_182002

def g (x : ℝ) : ℝ := 3 * x + 2

theorem problem_statement : g (g (g 3)) = 107 := by
  sorry

end problem_statement_l182_182002


namespace problem_solution_l182_182380

theorem problem_solution (f : ℝ → ℝ)
    (h : ∀ x y : ℝ, f ((x - y) ^ 2) = f x ^ 2 - 2 * x * f y + y ^ 2) :
    ∃ n s : ℕ, 
    (n = 2) ∧ 
    (s = 3) ∧
    (n * s = 6) :=
sorry

end problem_solution_l182_182380


namespace max_ab_l182_182365

theorem max_ab {a b : ℝ} (h₁ : a > 0) (h₂ : b > 0) (h₃ : a + b = 6) : ab ≤ 9 :=
sorry

end max_ab_l182_182365


namespace cat_food_sufficiency_l182_182323

variable (L S : ℝ)

theorem cat_food_sufficiency (h : L + 4 * S = 14) : L + 3 * S ≥ 11 :=
sorry

end cat_food_sufficiency_l182_182323


namespace total_goals_correct_l182_182599

-- Define the number of goals scored by each team in each period
def kickers_first_period_goals : ℕ := 2
def kickers_second_period_goals : ℕ := 2 * kickers_first_period_goals
def spiders_first_period_goals : ℕ := (1 / 2) * kickers_first_period_goals
def spiders_second_period_goals : ℕ := 2 * kickers_second_period_goals

-- Define the total goals scored by both teams
def total_goals : ℕ :=
  kickers_first_period_goals + 
  kickers_second_period_goals + 
  spiders_first_period_goals + 
  spiders_second_period_goals

-- State the theorem to be proved
theorem total_goals_correct : total_goals = 15 := by
  sorry

end total_goals_correct_l182_182599


namespace find_y_l182_182677

-- Suppose C > A > B > 0
-- and A is y% smaller than C.
-- Also, C = 2B.
-- We need to show that y = 100 - 50 * (A / B).

variable (A B C : ℝ)
variable (y : ℝ)

-- Conditions
axiom h1 : C > A
axiom h2 : A > B
axiom h3 : B > 0
axiom h4 : C = 2 * B
axiom h5 : A = (1 - y / 100) * C

-- Goal
theorem find_y : y = 100 - 50 * (A / B) :=
by
  sorry

end find_y_l182_182677


namespace intersection_S_T_eq_T_l182_182944

def S : Set ℤ := { s | ∃ n : ℤ, s = 2 * n + 1 }
def T : Set ℤ := { t | ∃ n : ℤ, t = 4 * n + 1 }

theorem intersection_S_T_eq_T : S ∩ T = T := by
  sorry

end intersection_S_T_eq_T_l182_182944


namespace max_distance_between_sparkling_points_l182_182091

theorem max_distance_between_sparkling_points (a₁ b₁ a₂ b₂ : ℝ) 
  (h₁ : a₁^2 + b₁^2 = 1) (h₂ : a₂^2 + b₂^2 = 1) :
  ∃ d, d = 2 ∧ ∀ (x y : ℝ), x = a₂ - a₁ ∧ y = b₂ - b₁ → (x ^ 2 + y ^ 2 = d ^ 2) :=
by
  sorry

end max_distance_between_sparkling_points_l182_182091


namespace cos_90_eq_0_l182_182144

theorem cos_90_eq_0 :
  ∃ (p : ℝ × ℝ), p = (0, 1) ∧ ∀ θ : ℝ, θ = 90 → cos θ = p.1 :=
by
  let p := (0, 1)
  use p
  split
  · rfl
  · intros θ h
    rw h
    sorry

end cos_90_eq_0_l182_182144


namespace dice_sum_probability_l182_182272

theorem dice_sum_probability :
  let outcomes := {(x, y) | x ∈ {1, 2, 3, 4, 5, 6} ∧ y ∈ {1, 2, 3, 4, 5, 6}}
  let favorable := {(x, y) | (x, y) ∈ outcomes ∧ (x + y = 4 ∨ x + y = 8 ∨ x + y = 12)}
  (favorable.card.to_rat / outcomes.card.to_rat) = 1 / 4 :=
by
  -- The proof would be filled here
  sorry

end dice_sum_probability_l182_182272


namespace polynomial_solution_l182_182913

theorem polynomial_solution (P : ℝ → ℝ) :
  (∀ a b c : ℝ, (a * b + b * c + c * a = 0) → 
  (P (a - b) + P (b - c) + P (c - a) = 2 * P (a + b + c))) →
  ∃ α β : ℝ, ∀ x : ℝ, P x = α * x ^ 4 + β * x ^ 2 :=
by
  sorry

end polynomial_solution_l182_182913


namespace cos_90_eq_0_l182_182141

theorem cos_90_eq_0 :
  ∃ (p : ℝ × ℝ), p = (0, 1) ∧ ∀ θ : ℝ, θ = 90 → cos θ = p.1 :=
by
  let p := (0, 1)
  use p
  split
  · rfl
  · intros θ h
    rw h
    sorry

end cos_90_eq_0_l182_182141


namespace angle_same_after_minutes_l182_182500

def angle_between_hands (H M : ℝ) : ℝ :=
  abs (30 * H - 5.5 * M)

theorem angle_same_after_minutes (x : ℝ) :
  x = 54 + 6 / 11 → 
  angle_between_hands (5 + (x / 60)) x = 150 :=
by
  sorry

end angle_same_after_minutes_l182_182500


namespace hotel_towels_l182_182733

theorem hotel_towels (num_rooms : ℕ) (num_people_per_room : ℕ) (towels_per_person : ℕ)
  (h1 : num_rooms = 10) (h2 : num_people_per_room = 3) (h3 : towels_per_person = 2) :
  num_rooms * num_people_per_room * towels_per_person = 60 :=
by
  sorry

end hotel_towels_l182_182733


namespace sum_of_first_five_integers_l182_182758

theorem sum_of_first_five_integers : (1 + 2 + 3 + 4 + 5) = 15 := 
by 
  sorry

end sum_of_first_five_integers_l182_182758


namespace harmonic_progression_l182_182429

theorem harmonic_progression (a b c : ℕ) (h_pos : 0 < a ∧ 0 < b ∧ 0 < c)
(h_harm : 1 / (a : ℝ) + 1 / (c : ℝ) = 2 / (b : ℝ))
(h_div : c % b = 0)
(h_inc : a < b ∧ b < c) :
  a = 20 → 
  (b, c) = (30, 60) ∨ (b, c) = (35, 140) ∨ (b, c) = (36, 180) ∨ (b, c) = (38, 380) ∨ (b, c) = (39, 780) :=
by sorry

end harmonic_progression_l182_182429


namespace smallest_x_for_equation_l182_182760

theorem smallest_x_for_equation :
  ∃ x : ℝ, x = -15 ∧ (∀ y : ℝ, 3*y^2 + 39*y - 75 = y*(y + 16) → x ≤ y) ∧ 
  3*(-15)^2 + 39*(-15) - 75 = -15*(-15 + 16) :=
sorry

end smallest_x_for_equation_l182_182760


namespace intersection_of_sets_l182_182983

open Set Int

theorem intersection_of_sets (S T : Set Int) (hS : S = {s | ∃ n : ℤ, s = 2 * n + 1}) (hT : T = {t | ∃ n : ℤ, t = 4 * n + 1}) : S ∩ T = T :=
  by
  sorry

end intersection_of_sets_l182_182983


namespace max_servings_l182_182573

-- Define available chunks for each type of fruit
def available_cantaloupe := 150
def available_honeydew := 135
def available_pineapple := 60
def available_watermelon := 220

-- Define the required chunks per serving for each type of fruit
def chunks_per_serving_cantaloupe := 3
def chunks_per_serving_honeydew := 2
def chunks_per_serving_pineapple := 1
def chunks_per_serving_watermelon := 4

-- Define the minimum required servings
def minimum_servings := 50

-- Prove the greatest number of servings that can be made while maintaining the specific ratio
theorem max_servings : 
  ∀ s : ℕ, 
  s * chunks_per_serving_cantaloupe ≤ available_cantaloupe ∧
  s * chunks_per_serving_honeydew ≤ available_honeydew ∧
  s * chunks_per_serving_pineapple ≤ available_pineapple ∧
  s * chunks_per_serving_watermelon ≤ available_watermelon ∧ 
  s ≥ minimum_servings → 
  s = 50 :=
by
  sorry

end max_servings_l182_182573


namespace bus_ride_cost_l182_182897

/-- The cost of a bus ride from town P to town Q, given that the cost of a train ride is $2.35 more 
    than a bus ride, and the combined cost of one train ride and one bus ride is $9.85. -/
theorem bus_ride_cost (B : ℝ) (h1 : ∃T, T = B + 2.35) (h2 : ∃T, T + B = 9.85) : B = 3.75 :=
by
  obtain ⟨T1, hT1⟩ := h1
  obtain ⟨T2, hT2⟩ := h2
  simp only [hT1, add_right_inj] at hT2
  sorry

end bus_ride_cost_l182_182897


namespace card_game_fairness_l182_182430

theorem card_game_fairness :
  let deck_size := 52
  let aces := 2
  let total_pairings := Nat.choose deck_size aces  -- Number of ways to choose 2 positions from 52
  let tie_cases := deck_size - 1                  -- Number of ways for consecutive pairs
  let non_tie_outcomes := total_pairings - tie_cases
  non_tie_outcomes / 2 = non_tie_outcomes / 2
:= sorry

end card_game_fairness_l182_182430


namespace purchase_costs_10_l182_182197

def total_cost (a b c d e : ℝ) := a + b + c + d + e
def cost_dates (a : ℝ) := 3 * a
def cost_cantaloupe (a b : ℝ) := a - b
def cost_eggs (b c : ℝ) := b + c

theorem purchase_costs_10 (a b c d e : ℝ) 
  (h_total_cost : total_cost a b c d e = 30)
  (h_cost_dates : d = cost_dates a)
  (h_cost_cantaloupe : c = cost_cantaloupe a b)
  (h_cost_eggs : e = cost_eggs b c) :
  b + c + e = 10 :=
by
  have := h_total_cost
  have := h_cost_dates
  have := h_cost_cantaloupe
  have := h_cost_eggs
  sorry

end purchase_costs_10_l182_182197


namespace number_of_stickers_after_losing_page_l182_182421

theorem number_of_stickers_after_losing_page (stickers_per_page : ℕ) (initial_pages : ℕ) (pages_lost : ℕ) :
  stickers_per_page = 20 → initial_pages = 12 → pages_lost = 1 → (initial_pages - pages_lost) * stickers_per_page = 220 :=
by
  intros
  sorry

end number_of_stickers_after_losing_page_l182_182421


namespace cos_pi_half_eq_zero_l182_182173

theorem cos_pi_half_eq_zero : Real.cos (Float.pi / 2) = 0 :=
by
  sorry

end cos_pi_half_eq_zero_l182_182173


namespace sum_of_numbers_l182_182685

theorem sum_of_numbers (x y z : ℝ) (h1 : x ≤ y) (h2 : y ≤ z) 
  (h3 : y = 5) (h4 : (x + y + z) / 3 = x + 10) (h5 : (x + y + z) / 3 = z - 15) : 
  x + y + z = 30 := 
by 
  sorry

end sum_of_numbers_l182_182685


namespace evaluate_expression_l182_182439

-- Define the conditions
def num : ℤ := 900^2
def a : ℤ := 306
def b : ℤ := 294
def denom : ℤ := a^2 - b^2

-- State the theorem to be proven
theorem evaluate_expression : (num : ℚ) / denom = 112.5 :=
by
  -- proof is skipped
  sorry

end evaluate_expression_l182_182439


namespace find_root_interval_l182_182842

noncomputable def f (x : ℝ) : ℝ := x^3 - 3*x + 1

theorem find_root_interval : ∃ k : ℕ, (f 1 < 0 ∧ f 2 > 0) → k = 1 :=
by
  sorry

end find_root_interval_l182_182842


namespace cube_faces_l182_182362

theorem cube_faces : ∀ (c : {s : Type | ∃ (x y z : ℝ), s = ({ (x0, y0, z0) : ℝ × ℝ × ℝ | x0 ≤ x ∧ y0 ≤ y ∧ z0 ≤ z}) }), 
  ∃ (f : ℕ), f = 6 :=
by 
  -- proof would be written here
  sorry

end cube_faces_l182_182362


namespace probability_of_consonant_initials_is_10_over_13_l182_182505

def is_vowel (c : Char) : Prop :=
  c = 'A' ∨ c = 'E' ∨ c = 'I' ∨ c = 'O' ∨ c = 'U' ∨ c = 'Y'

def is_consonant (c : Char) : Prop :=
  ¬(is_vowel c) ∧ c ≠ 'W' 

noncomputable def probability_of_consonant_initials : ℚ :=
  let total_letters := 26
  let number_of_vowels := 6
  let number_of_consonants := total_letters - number_of_vowels
  number_of_consonants / total_letters

theorem probability_of_consonant_initials_is_10_over_13 :
  probability_of_consonant_initials = 10 / 13 :=
by
  sorry

end probability_of_consonant_initials_is_10_over_13_l182_182505


namespace ratio_of_x_to_y_l182_182759

theorem ratio_of_x_to_y (x y : ℝ) (h : (12 * x - 5 * y) / (15 * x - 3 * y) = 4 / 7) : x / y = 23 / 24 :=
sorry

end ratio_of_x_to_y_l182_182759


namespace total_spent_is_correct_l182_182653

-- Declare the constants for the prices and quantities
def wallet_cost : ℕ := 50
def sneakers_cost_per_pair : ℕ := 100
def sneakers_pairs : ℕ := 2
def backpack_cost : ℕ := 100
def jeans_cost_per_pair : ℕ := 50
def jeans_pairs : ℕ := 2

-- Define the total amounts spent by Leonard and Michael
def leonard_total : ℕ := wallet_cost + sneakers_cost_per_pair * sneakers_pairs
def michael_total : ℕ := backpack_cost + jeans_cost_per_pair * jeans_pairs

-- The total amount spent by Leonard and Michael
def total_spent : ℕ := leonard_total + michael_total

-- The proof statement
theorem total_spent_is_correct : total_spent = 450 :=
by 
  -- This part is where the proof would go
  sorry

end total_spent_is_correct_l182_182653


namespace Alejandra_overall_score_l182_182459

theorem Alejandra_overall_score :
  let score1 := (60/100 : ℝ) * 20
  let score2 := (75/100 : ℝ) * 30
  let score3 := (85/100 : ℝ) * 40
  let total_score := score1 + score2 + score3
  let total_questions := 90
  let overall_percentage := (total_score / total_questions) * 100
  round overall_percentage = 77 :=
by
  sorry

end Alejandra_overall_score_l182_182459


namespace trigonometric_identity_l182_182838

noncomputable def π := Real.pi
noncomputable def tan (x : ℝ) := Real.sin x / Real.cos x

theorem trigonometric_identity (α β : ℝ) 
  (hα : 0 < α ∧ α < π / 2) 
  (hβ : 0 < β ∧ β < π / 2) 
  (h : tan α = (1 + Real.sin β) / Real.cos β) :
  2 * α - β = π / 2 := 
sorry

end trigonometric_identity_l182_182838


namespace pilot_fish_speed_theorem_l182_182829

noncomputable def pilot_fish_speed 
    (keanu_speed : ℕ)
    (shark_factor : ℕ) 
    (pilot_fish_factor : ℕ) 
    : ℕ :=
    let shark_speed_increase := keanu_speed * (shark_factor - 1) in
    let pilot_fish_speed_increase := shark_speed_increase / pilot_fish_factor in
    keanu_speed + pilot_fish_speed_increase

theorem pilot_fish_speed_theorem : 
    pilot_fish_speed 20 2 2 = 30 :=
by 
    simp [pilot_fish_speed]
    sorry  -- proof steps are omitted.

end pilot_fish_speed_theorem_l182_182829


namespace trajectory_perimeter_range_area_ratio_range_l182_182789

noncomputable def trajectory_eq (x y : ℝ) := x^2 + y^2 = 4

theorem trajectory (x y : ℝ) :
  (abs (sqrt ((x+4)^2 + y^2)) = 2 * abs (sqrt ((x+1)^2 + y^2))) ↔ trajectory_eq x y := 
by
  sorry -- Proof omitted

theorem perimeter_range (x y : ℝ) (h : trajectory_eq x y) :
  6 < abs (sqrt ((x+4)^2 + y^2)) + abs (sqrt ((x+1)^2 + y^2)) < 12 := 
by
  sorry -- Proof omitted

theorem area_ratio_range (x y m : ℝ) (h : trajectory_eq x y) :
  let y2 := (-2 * m + m^2 + 1) / (m^2 + 1)
  let t := abs y / abs y2
  (1 / 3) < t ∧ t < 3 :=
by
  sorry -- Proof omitted

end trajectory_perimeter_range_area_ratio_range_l182_182789


namespace scale_model_height_l182_182370

theorem scale_model_height 
  (scale_ratio : ℚ) (actual_height : ℚ)
  (h_ratio : scale_ratio = 1/30)
  (h_actual_height : actual_height = 305) 
  : Int.ceil (actual_height * scale_ratio) = 10 := by
  -- Define variables and the necessary conditions
  let height_of_model: ℚ := actual_height * scale_ratio
  -- Skip the proof steps
  sorry

end scale_model_height_l182_182370


namespace systematic_sampling_correct_l182_182280

-- Conditions as definitions
def total_bags : ℕ := 50
def num_samples : ℕ := 5
def interval (total num : ℕ) : ℕ := total / num
def correct_sequence : List ℕ := [5, 15, 25, 35, 45]

-- Statement
theorem systematic_sampling_correct :
  ∃ l : List ℕ, (l.length = num_samples) ∧ 
               (∀ i ∈ l, i ≤ total_bags) ∧
               (∀ i j, i < j → l.indexOf i < l.indexOf j → j - i = interval total_bags num_samples) ∧
               l = correct_sequence :=
by
  sorry

end systematic_sampling_correct_l182_182280


namespace cos_90_eq_0_l182_182134

theorem cos_90_eq_0 : Real.cos (Float.pi / 2) = 0 := 
sorry

end cos_90_eq_0_l182_182134


namespace largest_n_exists_unique_k_l182_182868

theorem largest_n_exists_unique_k (n k : ℕ) :
  (∃! k, (8 : ℚ) / 15 < (n : ℚ) / (n + k) ∧ (n : ℚ) / (n + k) < 7 / 13) →
  n ≤ 112 :=
sorry

end largest_n_exists_unique_k_l182_182868


namespace quadratic_function_condition_l182_182788

theorem quadratic_function_condition (m : ℝ) (h1 : |m| = 2) (h2 : m - 2 ≠ 0) : m = -2 :=
  sorry

end quadratic_function_condition_l182_182788


namespace minimum_value_of_expression_l182_182195

theorem minimum_value_of_expression (x y : ℝ) (hx : x > 1) (hy : y > 1) :
    (x^4 / (y - 1)) + (y^4 / (x - 1)) ≥ 12 := 
sorry

end minimum_value_of_expression_l182_182195


namespace cube_root_opposite_zero_l182_182853

theorem cube_root_opposite_zero (x : ℝ) (h : x^(1/3) = -x) : x = 0 :=
sorry

end cube_root_opposite_zero_l182_182853


namespace flammable_ice_storage_capacity_l182_182064

theorem flammable_ice_storage_capacity (billion : ℕ) (h : billion = 10^9) : (800 * billion = 8 * 10^11) :=
by
  sorry

end flammable_ice_storage_capacity_l182_182064


namespace bowling_ball_weight_l182_182773

def weight_of_canoe : ℕ := 32
def weight_of_canoes (n : ℕ) := n * weight_of_canoe
def weight_of_bowling_balls (n : ℕ) := 128

theorem bowling_ball_weight :
  (128 / 5 : ℚ) = (weight_of_bowling_balls 5 / 5 : ℚ) :=
by
  -- Theorems and calculations would typically be carried out here
  sorry

end bowling_ball_weight_l182_182773


namespace values_of_N_l182_182781

theorem values_of_N (N : ℕ) : (∃ k, k ∈ ({4, 6, 8, 12, 16, 24, 48} : set ℕ) ∧ k = N + 3) ↔ (N ∈ {1, 3, 5, 9, 13, 21, 45} : set ℕ) :=
by 
  sorry

#eval values_of_N 4 -- Example usage: should give true if N = 1

end values_of_N_l182_182781


namespace green_dots_fifth_row_l182_182665

variable (R : ℕ → ℕ)

-- Define the number of green dots according to the pattern
def pattern (n : ℕ) : ℕ := 3 * n

-- Define conditions for rows
axiom row_1 : R 1 = 3
axiom row_2 : R 2 = 6
axiom row_3 : R 3 = 9
axiom row_4 : R 4 = 12

-- The theorem
theorem green_dots_fifth_row : R 5 = 15 :=
by
  -- Row 5 follows the pattern and should satisfy the condition R 5 = R 4 + 3
  sorry

end green_dots_fifth_row_l182_182665


namespace stickers_after_loss_l182_182418

-- Conditions
def stickers_per_page : ℕ := 20
def initial_pages : ℕ := 12
def lost_pages : ℕ := 1

-- Problem statement
theorem stickers_after_loss : (initial_pages - lost_pages) * stickers_per_page = 220 := by
  sorry

end stickers_after_loss_l182_182418


namespace solve_problem_l182_182381

theorem solve_problem
    (x y z : ℝ)
    (h1 : x > 0)
    (h2 : y > 0)
    (h3 : z > 0)
    (h4 : x^2 + x * y + y^2 = 2)
    (h5 : y^2 + y * z + z^2 = 5)
    (h6 : z^2 + z * x + x^2 = 3) :
    x * y + y * z + z * x = 2 * Real.sqrt 2 := 
by
  sorry

end solve_problem_l182_182381


namespace age_difference_l182_182852

theorem age_difference (P M Mo : ℕ) (h1 : P = (3 * M) / 5) (h2 : Mo = (4 * M) / 3) (h3 : P + M + Mo = 88) : Mo - P = 22 := 
by sorry

end age_difference_l182_182852


namespace songs_in_each_album_l182_182566

variable (X : ℕ)

theorem songs_in_each_album (h : 6 * X + 2 * X = 72) : X = 9 :=
by sorry

end songs_in_each_album_l182_182566


namespace intersection_of_sets_l182_182980

open Set Int

theorem intersection_of_sets (S T : Set Int) (hS : S = {s | ∃ n : ℤ, s = 2 * n + 1}) (hT : T = {t | ∃ n : ℤ, t = 4 * n + 1}) : S ∩ T = T :=
  by
  sorry

end intersection_of_sets_l182_182980


namespace homework_problems_left_l182_182528

def math_problems : ℕ := 43
def science_problems : ℕ := 12
def finished_problems : ℕ := 44

theorem homework_problems_left :
  (math_problems + science_problems - finished_problems) = 11 :=
by
  sorry

end homework_problems_left_l182_182528


namespace sqrt_nine_factorial_over_72_eq_l182_182105

noncomputable def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

theorem sqrt_nine_factorial_over_72_eq : 
  Real.sqrt ((factorial 9) / 72) = 12 * Real.sqrt 35 :=
by
  sorry

end sqrt_nine_factorial_over_72_eq_l182_182105


namespace evaluate_expression_l182_182340

theorem evaluate_expression : (827 * 827) - (826 * 828) + 2 = 3 := by
  sorry

end evaluate_expression_l182_182340


namespace axel_vowels_written_l182_182902

theorem axel_vowels_written (total_alphabets number_of_vowels n : ℕ) (h1 : total_alphabets = 10) (h2 : number_of_vowels = 5) (h3 : total_alphabets = number_of_vowels * n) : n = 2 :=
by
  sorry

end axel_vowels_written_l182_182902


namespace intersection_of_sets_l182_182973

open Set Int

theorem intersection_of_sets (S T : Set Int) (hS : S = {s | ∃ n : ℤ, s = 2 * n + 1}) (hT : T = {t | ∃ n : ℤ, t = 4 * n + 1}) : S ∩ T = T :=
  by
  sorry

end intersection_of_sets_l182_182973


namespace find_A_l182_182701

theorem find_A (A7B : ℕ) (H1 : (A7B % 100) / 10 = 7) (H2 : A7B + 23 = 695) : (A7B / 100) = 6 := 
  sorry

end find_A_l182_182701


namespace intersection_S_T_eq_T_l182_182947

def S : Set ℤ := { s | ∃ n : ℤ, s = 2 * n + 1 }
def T : Set ℤ := { t | ∃ n : ℤ, t = 4 * n + 1 }

theorem intersection_S_T_eq_T : S ∩ T = T := by
  sorry

end intersection_S_T_eq_T_l182_182947


namespace square_can_be_divided_into_40_smaller_squares_l182_182437

theorem square_can_be_divided_into_40_smaller_squares 
: ∃ (n : ℕ), n * n = 40 := 
sorry

end square_can_be_divided_into_40_smaller_squares_l182_182437


namespace percentage_less_than_l182_182089

theorem percentage_less_than (x y : ℝ) (h : x = 8 * y) : ((x - y) / x) * 100 = 87.5 := 
by sorry

end percentage_less_than_l182_182089


namespace function_single_intersection_l182_182221

theorem function_single_intersection (a : ℝ) : 
  (∃ x : ℝ, ax^2 - x + 1 = 0 ∧ ∀ y : ℝ, (ax^2 - x + 1 = 0 → y = x)) ↔ (a = 0 ∨ a = 1/4) :=
sorry

end function_single_intersection_l182_182221


namespace owner_overtakes_thief_l182_182744

theorem owner_overtakes_thief :
  ∀ (speed_thief speed_owner : ℕ) (time_theft_discovered : ℝ), 
    speed_thief = 45 →
    speed_owner = 50 →
    time_theft_discovered = 0.5 →
    (time_theft_discovered + (45 * 0.5) / (speed_owner - speed_thief)) = 5 := 
by
  intros speed_thief speed_owner time_theft_discovered h1 h2 h3
  rw [h1, h2, h3]
  norm_num
  done

end owner_overtakes_thief_l182_182744


namespace school_trip_seat_count_l182_182410

theorem school_trip_seat_count :
  ∀ (classrooms students_per_classroom seats_per_bus : ℕ),
  classrooms = 87 →
  students_per_classroom = 58 →
  seats_per_bus = 29 →
  ∀ (total_students total_buses_needed : ℕ),
  total_students = classrooms * students_per_classroom →
  total_buses_needed = (total_students + seats_per_bus - 1) / seats_per_bus →
  seats_per_bus = 29 := by
  intros classrooms students_per_classroom seats_per_bus
  intros h1 h2 h3
  intros total_students total_buses_needed
  intros h4 h5
  sorry

end school_trip_seat_count_l182_182410


namespace frac_equality_l182_182348

variables (a b : ℚ) -- Declare the variables as rational numbers

-- State the theorem with the given condition and the proof goal
theorem frac_equality (h : a / b = 2 / 3) : a / (a + b) = 2 / 5 :=
by
  sorry -- proof goes here

end frac_equality_l182_182348


namespace range_of_a_l182_182814

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, x^2 + (a - 1) * x + 1 ≥ 0) ↔ (-1 ≤ a ∧ a ≤ 3) :=
by
  sorry

end range_of_a_l182_182814


namespace car_repair_cost_l182_182443

noncomputable def total_cost (first_mechanic_rate: ℝ) (first_mechanic_hours: ℕ) 
    (first_mechanic_days: ℕ) (second_mechanic_rate: ℝ) 
    (second_mechanic_hours: ℕ) (second_mechanic_days: ℕ) 
    (discount_first: ℝ) (discount_second: ℝ) 
    (parts_cost: ℝ) (sales_tax_rate: ℝ): ℝ :=
  let first_mechanic_cost := first_mechanic_rate * first_mechanic_hours * first_mechanic_days
  let second_mechanic_cost := second_mechanic_rate * second_mechanic_hours * second_mechanic_days
  let first_mechanic_discounted := first_mechanic_cost - (discount_first * first_mechanic_cost)
  let second_mechanic_discounted := second_mechanic_cost - (discount_second * second_mechanic_cost)
  let total_before_tax := first_mechanic_discounted + second_mechanic_discounted + parts_cost
  let sales_tax := sales_tax_rate * total_before_tax
  total_before_tax + sales_tax

theorem car_repair_cost :
  total_cost 60 8 14 75 6 10 0.15 0.10 3200 0.07 = 13869.34 := by
  sorry

end car_repair_cost_l182_182443


namespace probability_between_20_and_30_l182_182663

open Probability

-- Definition of standard six-sided die
def die := {i : ℕ // 1 ≤ i ∧ i ≤ 6}

-- Probability of not rolling a 2 on a die
def probability_not_two : ℚ := 5 / 6

-- Probability that neither of the two dice shows a 2
def probability_neither_two : ℚ :=
  probability_not_two * probability_not_two

-- Probability that at least one die shows a 2
def probability_at_least_one_two : ℚ :=
  1 - probability_neither_two

-- Theorem to prove the desired probability is 11/36
theorem probability_between_20_and_30 : probability_at_least_one_two = 11 / 36 :=
by
  sorry

end probability_between_20_and_30_l182_182663


namespace smallest_k_for_perfect_cube_l182_182404

noncomputable def isPerfectCube (m : ℕ) : Prop :=
  ∃ n : ℤ, n^3 = m

theorem smallest_k_for_perfect_cube :
  ∃ k : ℕ, k > 0 ∧ (∀ m : ℕ, ((2^4) * (3^2) * (5^5) * k = m) → isPerfectCube m) ∧ k = 60 :=
sorry

end smallest_k_for_perfect_cube_l182_182404


namespace carrots_left_over_l182_182584

theorem carrots_left_over (c g : ℕ) (h₁ : c = 47) (h₂ : g = 4) : c % g = 3 :=
by
  sorry

end carrots_left_over_l182_182584


namespace volume_of_cylindrical_block_l182_182369

variable (h_cylindrical : ℕ) (combined_value : ℝ)

theorem volume_of_cylindrical_block (h_cylindrical : ℕ) (combined_value : ℝ):
  h_cylindrical = 3 → combined_value / 5 * h_cylindrical = 15.42 := by
suffices combined_value / 5 = 5.14 from sorry
suffices 5.14 * 3 = 15.42 from sorry
suffices h_cylindrical = 3 from sorry
suffices 25.7 = combined_value from sorry
sorry

end volume_of_cylindrical_block_l182_182369


namespace masha_dolls_l182_182662

theorem masha_dolls (n : ℕ) (h : (n / 2) * 1 + (n / 4) * 2 + (n / 4) * 4 = 24) : n = 12 :=
sorry

end masha_dolls_l182_182662


namespace no_valid_x_for_given_circle_conditions_l182_182009

theorem no_valid_x_for_given_circle_conditions :
  ∀ x : ℝ,
    ¬ ((x - 15)^2 + 18^2 = 225 ∧ (x - 15)^2 + (-18)^2 = 225) :=
by
  sorry

end no_valid_x_for_given_circle_conditions_l182_182009


namespace intersection_eq_T_l182_182988

def S : Set ℤ := {s | ∃ n : ℤ, s = 2 * n + 1}
def T : Set ℤ := {t | ∃ n : ℤ, t = 4 * n + 1}

theorem intersection_eq_T : S ∩ T = T := sorry

end intersection_eq_T_l182_182988


namespace no_such_integers_l182_182023

noncomputable def omega : ℂ := Complex.exp (2 * Real.pi * Complex.I / 5)

theorem no_such_integers (a b c d k : ℤ) (h : k > 1) :
  (a + b * omega + c * omega^2 + d * omega^3)^k ≠ 1 + omega :=
sorry

end no_such_integers_l182_182023


namespace correct_solution_l182_182247

theorem correct_solution : 
  ∀ (x y a b : ℚ), (a = 1) → (b = 1 / 2) → 
  (a * x + y = 2) → (2 * x - b * y = 1) → 
  (x = 4 / 5 ∧ y = 6 / 5) := 
by
  intros x y a b ha hb h1 h2
  sorry

end correct_solution_l182_182247


namespace central_angle_probability_l182_182285

theorem central_angle_probability (A : ℝ) (x : ℝ)
  (h1 : A > 0)
  (h2 : (x / 360) * A / A = 1 / 8) : 
  x = 45 := 
by
  sorry

end central_angle_probability_l182_182285


namespace num_four_digit_numbers_greater_than_3999_with_product_of_middle_two_digits_exceeding_10_l182_182630

def num_valid_pairs : Nat := 34

def num_valid_first_digits : Nat := 6

def num_valid_last_digits : Nat := 10

theorem num_four_digit_numbers_greater_than_3999_with_product_of_middle_two_digits_exceeding_10 :
  (num_valid_first_digits * num_valid_pairs * num_valid_last_digits) = 2040 :=
by
  sorry

end num_four_digit_numbers_greater_than_3999_with_product_of_middle_two_digits_exceeding_10_l182_182630


namespace smallest_angle_of_convex_15_gon_arithmetic_sequence_l182_182050

theorem smallest_angle_of_convex_15_gon_arithmetic_sequence :
  ∃ (a d : ℕ), (∀ k : ℕ, k < 15 → (let angle := a + k * d in angle < 180)) ∧
  (∀ i j : ℕ, i < j → i < 15 → j < 15 → (a + i * d) < (a + j * d)) ∧
  (let sequence_sum := 15 * a + d * 7 * 14 in sequence_sum = 2340) ∧
  (d = 3) ∧
  (a = 135) :=
by
  sorry

end smallest_angle_of_convex_15_gon_arithmetic_sequence_l182_182050


namespace problem_correct_l182_182379

noncomputable def S : Set ℕ := {x | x^2 - x = 0}
noncomputable def T : Set ℕ := {x | x ∈ Set.univ ∧ 6 % (x - 2) = 0}

theorem problem_correct : S ∩ T = ∅ :=
by sorry

end problem_correct_l182_182379


namespace quadrilateral_area_lemma_l182_182740

-- Define the coordinates of the vertices
structure Point where
  x : ℤ
  y : ℤ

def A : Point := ⟨1, 3⟩
def B : Point := ⟨1, 1⟩
def C : Point := ⟨2, 1⟩
def D : Point := ⟨2006, 2007⟩

-- Function to calculate the area of a quadrilateral given its vertices
def quadrilateral_area (A B C D : Point) : ℤ := 
  let triangle_area (P Q R : Point) : ℤ :=
    (Q.x - P.x) * (R.y - P.y) - (Q.y - P.y) * (R.x - P.x) / 2
  triangle_area A B C + triangle_area A C D

-- The statement to be proved
theorem quadrilateral_area_lemma : quadrilateral_area A B C D = 3008 := 
  sorry

end quadrilateral_area_lemma_l182_182740


namespace john_total_cost_l182_182890

def base_cost : ℤ := 25
def text_cost_per_message : ℤ := 8
def extra_minute_cost_per_minute : ℤ := 15
def international_minute_cost : ℤ := 100

def texts_sent : ℤ := 200
def total_hours : ℤ := 42
def international_minutes : ℤ := 10

-- Calculate the number of extra minutes
def extra_minutes : ℤ := (total_hours - 40) * 60

noncomputable def total_cost : ℤ :=
  base_cost +
  (texts_sent * text_cost_per_message) / 100 +
  (extra_minutes * extra_minute_cost_per_minute) / 100 +
  international_minutes * (international_minute_cost / 100)

theorem john_total_cost :
  total_cost = 69 := by
    sorry

end john_total_cost_l182_182890


namespace room_width_l182_182260

theorem room_width (w : ℝ) (h1 : 21 > 0) (h2 : 2 > 0) 
  (h3 : (25 * (w + 4) - 21 * w = 148)) : w = 12 :=
by {
  sorry
}

end room_width_l182_182260


namespace stickers_after_loss_l182_182419

-- Conditions
def stickers_per_page : ℕ := 20
def initial_pages : ℕ := 12
def lost_pages : ℕ := 1

-- Problem statement
theorem stickers_after_loss : (initial_pages - lost_pages) * stickers_per_page = 220 := by
  sorry

end stickers_after_loss_l182_182419


namespace chess_group_players_l182_182863

noncomputable def number_of_players_in_chess_group (n : ℕ) : Prop := 
  n * (n - 1) / 2 = 21

theorem chess_group_players : ∃ n : ℕ, number_of_players_in_chess_group n ∧ n = 7 := 
by
  use 7
  split
  show number_of_players_in_chess_group 7
  unfold number_of_players_in_chess_group
  norm_num
  show 7 = 7
  rfl

end chess_group_players_l182_182863


namespace intersection_eq_T_l182_182929

def S : Set ℤ := {s | ∃ n : ℤ, s = 2 * n + 1}
def T : Set ℤ := {t | ∃ n : ℤ, t = 4 * n + 1}

theorem intersection_eq_T : S ∩ T = T := 
sorry

end intersection_eq_T_l182_182929


namespace shaded_area_is_28_l182_182257

theorem shaded_area_is_28 (A B : ℕ) (h1 : A = 64) (h2 : B = 28) : B = 28 := by
  sorry

end shaded_area_is_28_l182_182257


namespace fabric_ratio_l182_182592

theorem fabric_ratio
  (d_m : ℕ) (d_t : ℕ) (d_w : ℕ) (cost : ℕ) (total_revenue : ℕ) (revenue_monday : ℕ) (revenue_tuesday : ℕ) (revenue_wednesday : ℕ)
  (h_d_m : d_m = 20)
  (h_cost : cost = 2)
  (h_d_w : d_w = d_t / 4)
  (h_total_revenue : total_revenue = 140)
  (h_revenue : revenue_monday + revenue_tuesday + revenue_wednesday = total_revenue)
  (h_r_m : revenue_monday = d_m * cost)
  (h_r_t : revenue_tuesday = d_t * cost) 
  (h_r_w : revenue_wednesday = d_w * cost) :
  (d_t / d_m = 1) :=
by
  sorry

end fabric_ratio_l182_182592


namespace range_of_m_l182_182367

theorem range_of_m (m : ℝ) :
  (∀ x : ℝ, x ≤ -1 -> (m^2 - m) * 2^x - (1/2)^x < 1) →
  -2 < m ∧ m < 3 :=
by
  sorry

end range_of_m_l182_182367


namespace S_inter_T_eq_T_l182_182957

-- Define the sets S and T
def S := { s : ℤ | ∃ n : ℤ, s = 2 * n + 1 }
def T := { t : ℤ | ∃ n : ℤ, t = 4 * n + 1 }

-- Prove that S ∩ T = T
theorem S_inter_T_eq_T : S ∩ T = T := 
by
  sorry

end S_inter_T_eq_T_l182_182957


namespace percent_gain_is_5_333_l182_182732

noncomputable def calculate_percent_gain (total_sheep : ℕ) 
                                         (sold_sheep : ℕ) 
                                         (price_paid_sheep : ℕ) 
                                         (sold_remaining_sheep : ℕ)
                                         (remaining_sheep : ℕ) 
                                         (total_cost : ℝ) 
                                         (initial_revenue : ℝ) 
                                         (remaining_revenue : ℝ) : ℝ :=
  (remaining_revenue + initial_revenue - total_cost) / total_cost * 100

theorem percent_gain_is_5_333
  (x : ℝ)
  (total_sheep : ℕ := 800)
  (sold_sheep : ℕ := 750)
  (price_paid_sheep : ℕ := 790)
  (remaining_sheep : ℕ := 50)
  (total_cost : ℝ := (800 : ℝ) * x)
  (initial_revenue : ℝ := (790 : ℝ) * x)
  (remaining_revenue : ℝ := (50 : ℝ) * ((790 : ℝ) * x / 750)) :
  calculate_percent_gain total_sheep sold_sheep price_paid_sheep remaining_sheep 50 total_cost initial_revenue remaining_revenue = 5.333 := by
  sorry

end percent_gain_is_5_333_l182_182732


namespace Kim_total_hours_l182_182020

-- Define the initial conditions
def initial_classes : ℕ := 4
def hours_per_class : ℕ := 2
def dropped_class : ℕ := 1

-- The proof problem: Given the initial conditions, prove the total hours of classes per day is 6
theorem Kim_total_hours : (initial_classes - dropped_class) * hours_per_class = 6 := by
  sorry

end Kim_total_hours_l182_182020


namespace auditorium_total_chairs_l182_182511

theorem auditorium_total_chairs 
  (n : ℕ)
  (h1 : 2 + 5 - 1 = n)   -- n is the number of rows which is equal to 6
  (h2 : 3 + 4 - 1 = n)   -- n is the number of chairs per row which is also equal to 6
  : n * n = 36 :=        -- the total number of chairs is 36
by
  sorry

end auditorium_total_chairs_l182_182511


namespace smallest_positive_period_of_f_max_min_values_of_f_l182_182623

noncomputable def a (x : ℝ) : ℝ × ℝ := (Real.cos x, -1 / 2)
noncomputable def b (x : ℝ) : ℝ × ℝ := (Real.sqrt 3 * Real.sin x, Real.cos (2 * x))
noncomputable def f (x : ℝ) : ℝ := (a x).1 * (b x).1 + (a x).2 * (b x).2

--(I) Prove the smallest positive period of f(x) is π.
theorem smallest_positive_period_of_f : ∀ x : ℝ, f (x + π) = f x :=
sorry

--(II) Prove the maximum and minimum values of f(x) on [0, π / 2] are 1 and -1/2 respectively.
theorem max_min_values_of_f : ∃ max min : ℝ, (max = 1) ∧ (min = -1 / 2) ∧
  (∀ x ∈ Set.Icc 0 (π / 2), f x ≤ max ∧ f x ≥ min) :=
sorry

end smallest_positive_period_of_f_max_min_values_of_f_l182_182623


namespace greatest_expression_l182_182198

theorem greatest_expression 
  (x1 x2 y1 y2 : ℝ) 
  (hx1 : 0 < x1) 
  (hx2 : x1 < x2) 
  (hx12 : x1 + x2 = 1) 
  (hy1 : 0 < y1) 
  (hy2 : y1 < y2) 
  (hy12 : y1 + y2 = 1) : 
  x1 * y1 + x2 * y2 > max (x1 * x2 + y1 * y2) (max (x1 * y2 + x2 * y1) (1/2)) := 
sorry

end greatest_expression_l182_182198


namespace cos_90_eq_zero_l182_182160

def point_after_rotation (θ : ℝ) : ℝ × ℝ :=
  let x := cos θ
  let y := sin θ
  (x, y)

theorem cos_90_eq_zero : cos (real.pi / 2) = 0 :=
  sorry

end cos_90_eq_zero_l182_182160


namespace Lulu_blueberry_pies_baked_l182_182523

-- Definitions of conditions
def Lola_mini_cupcakes := 13
def Lola_pop_tarts := 10
def Lola_blueberry_pies := 8
def Lola_total_pastries := Lola_mini_cupcakes + Lola_pop_tarts + Lola_blueberry_pies
def Lulu_mini_cupcakes := 16
def Lulu_pop_tarts := 12
def total_pastries := 73

-- Prove that Lulu baked 14 blueberry pies
theorem Lulu_blueberry_pies_baked : 
  ∃ (Lulu_blueberry_pies : Nat), 
    Lola_total_pastries + Lulu_mini_cupcakes + Lulu_pop_tarts + Lulu_blueberry_pies = total_pastries ∧ 
    Lulu_blueberry_pies = 14 := by
  sorry

end Lulu_blueberry_pies_baked_l182_182523


namespace leading_coefficient_of_f_l182_182060

noncomputable def polynomial : Type := ℕ → ℝ

def satisfies_condition (f : polynomial) : Prop :=
  ∀ (x : ℕ), f (x + 1) - f x = 6 * x + 4

theorem leading_coefficient_of_f (f : polynomial) (h : satisfies_condition f) : 
  ∃ a b c : ℝ, (∀ (x : ℕ), f x = a * (x^2) + b * x + c) ∧ a = 3 := 
by
  sorry

end leading_coefficient_of_f_l182_182060


namespace carol_maximizes_chance_of_winning_l182_182899

noncomputable def carol_optimal_choice : ℝ :=
0.725

theorem carol_maximizes_chance_of_winning :
  ∀ (a b d : ℝ), (0 ≤ a ∧ a ≤ 1) ∧ (0.25 ≤ b ∧ b ≤ 0.75) ∧ (0 ≤ d ∧ d ≤ 1) →
  (∃ c : ℝ, (0.6 ≤ c ∧ c ≤ 0.9) ∧ (c = 0.725) ∧
  (c > a ∧ c < b ∨ 
   c < a ∧ c > b) ∧
  (c > d ∨ c < d)) :=
sorry

end carol_maximizes_chance_of_winning_l182_182899


namespace original_price_of_item_l182_182893

theorem original_price_of_item (P : ℝ) 
(selling_price : ℝ) 
(h1 : 0.9 * P = selling_price) 
(h2 : selling_price = 675) : 
P = 750 := sorry

end original_price_of_item_l182_182893


namespace min_value_of_b_plus_2_div_a_l182_182489

theorem min_value_of_b_plus_2_div_a (a : ℝ) (b : ℝ) (h₁ : 0 < a) 
  (h₂ : ∀ x : ℝ, 0 < x → (ax - 1) * (x^2 + bx - 4) ≥ 0) : 
  ∃ a' b', (a' > 0 ∧ b' = 4 * a' - 1 / a') ∧ b' + 2 / a' = 4 :=
by
  sorry

end min_value_of_b_plus_2_div_a_l182_182489


namespace common_volume_of_tetrahedra_l182_182823

open Real

noncomputable def volume_of_common_part (a b c : ℝ) : ℝ :=
  min (a * sqrt 3 / 12) (min (b * sqrt 3 / 12) (c * sqrt 3 / 12))

theorem common_volume_of_tetrahedra (a b c : ℝ) :
  volume_of_common_part a b c =
  min (a * sqrt 3 / 12) (min (b * sqrt 3 / 12) (c * sqrt 3 / 12)) :=
by sorry

end common_volume_of_tetrahedra_l182_182823


namespace sequence_sum_l182_182928

theorem sequence_sum (a : ℕ → ℚ) (S : ℕ → ℚ)
  (h1 : a 1 = 1)
  (h2 : ∀ n : ℕ, S n = n^2 * a n) :
  ∀ n : ℕ, S n = 2 * n / (n + 1) := 
by 
  sorry

end sequence_sum_l182_182928


namespace central_angle_of_sector_l182_182048

theorem central_angle_of_sector (R r n : ℝ) (h_lateral_area : 2 * π * r^2 = π * r * R) 
  (h_arc_length : (n * π * R) / 180 = 2 * π * r) : n = 180 :=
by 
  sorry

end central_angle_of_sector_l182_182048


namespace savings_per_month_l182_182182

noncomputable def annual_salary : ℝ := 48000
noncomputable def monthly_payments : ℝ := 12
noncomputable def savings_percentage : ℝ := 0.10

theorem savings_per_month :
  (annual_salary / monthly_payments) * savings_percentage = 400 :=
by
  sorry

end savings_per_month_l182_182182


namespace sector_central_angle_l182_182927

theorem sector_central_angle (r l α : ℝ) 
  (h1 : 2 * r + l = 6) 
  (h2 : 0.5 * l * r = 2) :
  α = l / r → α = 4 ∨ α = 1 :=
sorry

end sector_central_angle_l182_182927


namespace minimum_students_for_same_vote_l182_182675

theorem minimum_students_for_same_vote (n : ℕ) (k : ℕ) (h1 : n = 10) (h2 : k = 2) :
  ∃ m, m = 46 ∧ ∀ (students : Finset (Finset ℕ)), students.card = m → 
    (∃ s1 s2, s1 ≠ s2 ∧ s1.card = k ∧ s2.card = k ∧ s1 ⊆ (Finset.range n) ∧ s2 ⊆ (Finset.range n) ∧ s1 = s2) :=
by 
  sorry

end minimum_students_for_same_vote_l182_182675


namespace sam_total_distance_l182_182382

-- Definitions based on conditions
def first_half_distance : ℕ := 120
def first_half_time : ℕ := 3
def second_half_distance : ℕ := 80
def second_half_time : ℕ := 2
def sam_time : ℚ := 5.5

-- Marguerite's overall average speed
def marguerite_average_speed : ℚ := (first_half_distance + second_half_distance) / (first_half_time + second_half_time)

-- Theorem statement: Sam's total distance driven
theorem sam_total_distance : ∀ (d : ℚ), d = (marguerite_average_speed * sam_time) ↔ d = 220 := by
  intro d
  sorry

end sam_total_distance_l182_182382


namespace cat_food_sufficiency_l182_182315

theorem cat_food_sufficiency (L S : ℝ) (h : L + 4 * S = 14) : L + 3 * S ≥ 11 :=
sorry

end cat_food_sufficiency_l182_182315


namespace stratified_sampling_is_reasonable_l182_182449

-- Defining our conditions and stating our theorem
def flat_land := 150
def ditch_land := 30
def sloped_land := 90
def total_acres := 270
def sampled_acres := 18
def sampling_ratio := sampled_acres / total_acres

def flat_land_sampled := flat_land * sampling_ratio
def ditch_land_sampled := ditch_land * sampling_ratio
def sloped_land_sampled := sloped_land * sampling_ratio

theorem stratified_sampling_is_reasonable :
  flat_land_sampled = 10 ∧
  ditch_land_sampled = 2 ∧
  sloped_land_sampled = 6 := 
by
  sorry

end stratified_sampling_is_reasonable_l182_182449


namespace problem_statement_l182_182833

variables {a b c p q r : ℝ}

-- Given conditions
axiom h1 : 19 * p + b * q + c * r = 0
axiom h2 : a * p + 29 * q + c * r = 0
axiom h3 : a * p + b * q + 56 * r = 0
axiom h4 : a ≠ 19
axiom h5 : p ≠ 0

-- Statement to prove
theorem problem_statement : 
  (a / (a - 19)) + (b / (b - 29)) + (c / (c - 56)) = 1 :=
sorry

end problem_statement_l182_182833


namespace circle_table_acquaintance_impossible_l182_182426

theorem circle_table_acquaintance_impossible (P : Finset ℕ) (hP : P.card = 40) :
  ¬ (∀ (a b : ℕ), (a ∈ P) → (b ∈ P) → (∃ k, 2 * k ≠ 0) → (∃ c, c ∈ P) ∧ (a ≠ b) ∧ (c = a ∨ c = b)
       ↔ ¬(∃ k, 2 * k + 1 ≠ 0)) :=
by
  sorry

end circle_table_acquaintance_impossible_l182_182426


namespace MsSatosClassRatioProof_l182_182526

variable (g b : ℕ) -- g is the number of girls, b is the number of boys

def MsSatosClassRatioProblem : Prop :=
  (g = b + 6) ∧ (g + b = 32) → g / b = 19 / 13

theorem MsSatosClassRatioProof : MsSatosClassRatioProblem g b := by
  sorry

end MsSatosClassRatioProof_l182_182526


namespace log_base_change_l182_182787

theorem log_base_change (a b : ℝ) (h₁ : Real.log 5 / Real.log 3 = a) (h₂ : Real.log 7 / Real.log 3 = b) :
    Real.log 35 / Real.log 15 = (a + b) / (1 + a) :=
by
  sorry

end log_base_change_l182_182787


namespace option_c_incorrect_l182_182564

theorem option_c_incorrect (a : ℝ) : a + a^2 ≠ a^3 :=
sorry

end option_c_incorrect_l182_182564


namespace cos_of_90_degrees_l182_182169

-- Definition of cosine of 90 degrees
def cos_90_degrees : ℝ := Real.cos (90 * (Float.pi / 180))

-- Condition: Cosine is the x-coordinate of the point (0, 1) after 90 degree rotation
theorem cos_of_90_degrees : cos_90_degrees = 0 :=
by
  -- Proof would go here, but we skip it with sorry
  sorry

end cos_of_90_degrees_l182_182169


namespace circle_ratio_new_diameter_circumference_l182_182504

theorem circle_ratio_new_diameter_circumference (r : ℝ) :
  let new_radius := r + 2
  let new_diameter := 2 * new_radius
  let new_circumference := 2 * Real.pi * new_radius
  new_circumference / new_diameter = Real.pi := 
by
  sorry

end circle_ratio_new_diameter_circumference_l182_182504


namespace set_complement_l182_182045

variable {U : Set ℝ} (A : Set ℝ)

theorem set_complement :
  (U = {x : ℝ | x > 1}) →
  (A ⊆ U) →
  (U \ A = {x : ℝ | x > 9}) →
  (A = {x : ℝ | 1 < x ∧ x ≤ 9}) :=
by
  intros hU hA hC
  sorry

end set_complement_l182_182045


namespace rabbit_carrots_l182_182611

theorem rabbit_carrots (r f : ℕ) (hr : 3 * r = 5 * f) (hf : f = r - 6) : 3 * r = 45 :=
by
  sorry

end rabbit_carrots_l182_182611


namespace decreasing_interval_0_pi_over_4_l182_182357

noncomputable def f (x : ℝ) (φ : ℝ) : ℝ := Real.cos (x + φ)

theorem decreasing_interval_0_pi_over_4 (φ : ℝ) (hφ1 : 0 < |φ| ∧ |φ| < Real.pi / 2)
  (hodd : ∀ x : ℝ, f (x + Real.pi / 4) φ = -f (-x + Real.pi / 4) φ) :
  ∀ x : ℝ, 0 < x ∧ x < Real.pi / 4 → f x φ > f (x + 1e-6) φ :=
by sorry

end decreasing_interval_0_pi_over_4_l182_182357


namespace tan_pi9_2pi9_4pi9_l182_182078

theorem tan_pi9_2pi9_4pi9 :
  tan (Real.pi / 9) * tan (2 * Real.pi / 9) * tan (4 * Real.pi / 9) = Real.sqrt 3 :=
by
  sorry

end tan_pi9_2pi9_4pi9_l182_182078


namespace S_inter_T_eq_T_l182_182966

-- Define the sets S and T
def S := { s : ℤ | ∃ n : ℤ, s = 2 * n + 1 }
def T := { t : ℤ | ∃ n : ℤ, t = 4 * n + 1 }

-- Prove that S ∩ T = T
theorem S_inter_T_eq_T : S ∩ T = T := 
by
  sorry

end S_inter_T_eq_T_l182_182966


namespace monotonicity_of_f_range_of_a_l182_182755

noncomputable def f (a x : ℝ) : ℝ := 2 * a * Real.log x - x^2 + a

theorem monotonicity_of_f (a : ℝ) :
  (∀ x > 0, (a ≤ 0 → f a x ≤ f a (x - 1)) ∧ 
           (a > 0 → ((x < Real.sqrt a → f a x ≤ f a (x + 1)) ∨ 
                     (x > Real.sqrt a → f a x ≥ f a (x - 1))))) := sorry

theorem range_of_a (a : ℝ) :
  (∀ x > 0, f a x ≤ 0) → (0 ≤ a ∧ a ≤ 1) := sorry

end monotonicity_of_f_range_of_a_l182_182755


namespace sandwich_cost_is_five_l182_182375

-- Define the cost of each sandwich
variables (x : ℝ)

-- Conditions
def jack_orders_sandwiches (cost_per_sandwich : ℝ) : Prop :=
  3 * cost_per_sandwich = 15

-- Proof problem statement (no proof provided)
theorem sandwich_cost_is_five (h : jack_orders_sandwiches x) : x = 5 :=
sorry

end sandwich_cost_is_five_l182_182375


namespace university_diploma_percentage_l182_182371

variables (population : ℝ)
          (U : ℝ) -- percentage of people with a university diploma
          (J : ℝ := 0.40) -- percentage of people with the job of their choice
          (S : ℝ := 0.10) -- percentage of people with a secondary school diploma pursuing further education

-- Condition 1: 18% of the people do not have a university diploma but have the job of their choice.
-- Condition 2: 25% of the people who do not have the job of their choice have a university diploma.
-- Condition 3: 10% of the people have a secondary school diploma and are pursuing further education.
-- Condition 4: 60% of the people with secondary school diploma have the job of their choice.
-- Condition 5: 30% of the people in further education have a job of their choice as well.
-- Condition 6: 40% of the people have the job of their choice.

axiom condition_1 : 0.18 * population = (0.18 * (1 - U)) * (population)
axiom condition_2 : 0.25 * (100 - J * 100) = 0.25 * (population - J * population)
axiom condition_3 : S * population = 0.10 * population
axiom condition_4 : 0.60 * S * population = (0.60 * S) * population
axiom condition_5 : 0.30 * S * population = (0.30 * S) * population
axiom condition_6 : J * population = 0.40 * population

theorem university_diploma_percentage : U * 100 = 37 :=
by sorry

end university_diploma_percentage_l182_182371


namespace Cora_pages_to_read_on_Thursday_l182_182332

theorem Cora_pages_to_read_on_Thursday
  (total_pages : ℕ)
  (read_monday : ℕ)
  (read_tuesday : ℕ)
  (read_wednesday : ℕ)
  (pages_left : ℕ)
  (read_friday : ℕ)
  (thursday_pages : ℕ) :
  total_pages = 158 →
  read_monday = 23 →
  read_tuesday = 38 →
  read_wednesday = 61 →
  pages_left = total_pages - (read_monday + read_tuesday + read_wednesday) →
  read_friday = 2 * thursday_pages →
  pages_left = thursday_pages + read_friday →
  thursday_pages = 12 :=
begin
  -- Proof is not required
  sorry
end

end Cora_pages_to_read_on_Thursday_l182_182332


namespace fill_cistern_time_l182_182435

theorem fill_cistern_time (F E : ℝ) (hF : F = 1/2) (hE : E = 1/4) : 
  (1 / (F - E)) = 4 :=
by
  -- Definitions of F and E are used as hypotheses hF and hE
  -- Prove the actual theorem stating the time to fill the cistern is 4 hours
  sorry

end fill_cistern_time_l182_182435


namespace a_10_eq_18_l182_182510

variable {a : ℕ → ℕ}

-- Conditions
def is_arithmetic_sequence (a : ℕ → ℕ) : Prop :=
  ∃ d, ∀ n, a (n + 1) = a n + d

axiom a2 : a 2 = 2
axiom a3 : a 3 = 4
axiom arithmetic_seq : is_arithmetic_sequence a

-- problem: prove a_{10} = 18
theorem a_10_eq_18 : a 10 = 18 :=
sorry

end a_10_eq_18_l182_182510


namespace six_digit_number_l182_182765

/-- 
Find a six-digit number that starts with the digit 1 and such that if this digit is moved to the end, the resulting number is three times the original number.
-/
theorem six_digit_number (N : ℕ) (h₁ : 100000 ≤ N ∧ N < 1000000) (h₂ : ∃ x : ℕ, N = 1 * 10^5 + x ∧ 10 * x + 1 = 3 * N) : N = 142857 :=
by sorry

end six_digit_number_l182_182765


namespace smallest_positive_solution_eq_sqrt_29_l182_182433

theorem smallest_positive_solution_eq_sqrt_29 :
  ∃ x : ℝ, 0 < x ∧ x^4 - 58 * x^2 + 841 = 0 ∧ x = Real.sqrt 29 :=
by
  sorry

end smallest_positive_solution_eq_sqrt_29_l182_182433


namespace find_x_positive_integers_l182_182766

theorem find_x_positive_integers (a b c x : ℕ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a + b + c = x * a * b * c) → (x = 1 ∧ a = 1 ∧ b = 2 ∧ c = 3) ∨
  (x = 2 ∧ a = 1 ∧ b = 1 ∧ c = 2) ∨
  (x = 3 ∧ a = 1 ∧ b = 1 ∧ c = 1) :=
  sorry

end find_x_positive_integers_l182_182766


namespace expression_evaluation_l182_182468

def e : Int := -(-1) + 3^2 / (1 - 4) * 2

theorem expression_evaluation : e = -5 := 
by
  unfold e
  sorry

end expression_evaluation_l182_182468


namespace cat_food_sufficiency_l182_182308

variable (L S : ℝ)

theorem cat_food_sufficiency (h : L + 4 * S = 14) : L + 3 * S >= 11 :=
sorry

end cat_food_sufficiency_l182_182308


namespace intersection_eq_T_l182_182992

def S : Set ℤ := {s | ∃ n : ℤ, s = 2 * n + 1}
def T : Set ℤ := {t | ∃ n : ℤ, t = 4 * n + 1}

theorem intersection_eq_T : S ∩ T = T := sorry

end intersection_eq_T_l182_182992


namespace gambler_difference_eq_two_l182_182454

theorem gambler_difference_eq_two (x y : ℕ) (x_lost y_lost : ℕ) :
  20 * x + 100 * y = 3000 ∧
  x + y = 14 ∧
  20 * (14 - y_lost) + 100 * y_lost = 760 →
  (x_lost - y_lost = 2) := sorry

end gambler_difference_eq_two_l182_182454


namespace circle_center_l182_182921

theorem circle_center (x y: ℝ) : 
  (x + 2)^2 + (y + 3)^2 = 29 ↔ (∃ c1 c2 : ℝ, c1 = -2 ∧ c2 = -3) :=
by sorry

end circle_center_l182_182921


namespace rationalize_denominator_correct_l182_182670

noncomputable def rationalize_denominator : Prop :=
  (50 + Real.sqrt 8) / (Real.sqrt 50 + Real.sqrt 8) = (50 * (Real.sqrt 50 - Real.sqrt 8) + 12) / 42

theorem rationalize_denominator_correct : rationalize_denominator :=
by
  sorry

end rationalize_denominator_correct_l182_182670


namespace tobias_time_spent_at_pool_l182_182069

-- Define the conditions
def distance_per_interval : ℕ := 100
def time_per_interval : ℕ := 5
def pause_interval : ℕ := 25
def pause_time : ℕ := 5
def total_distance : ℕ := 3000
def total_time_in_hours : ℕ := 3

-- Hypotheses based on the problem conditions
def swimming_time_without_pauses := (total_distance / distance_per_interval) * time_per_interval
def number_of_pauses := (swimming_time_without_pauses / pause_interval)
def total_pause_time := number_of_pauses * pause_time
def total_time := swimming_time_without_pauses + total_pause_time

-- Proof statement
theorem tobias_time_spent_at_pool : total_time / 60 = total_time_in_hours :=
by 
  -- Put proof here
  sorry

end tobias_time_spent_at_pool_l182_182069


namespace hotel_towels_l182_182734

theorem hotel_towels (num_rooms : ℕ) (num_people_per_room : ℕ) (towels_per_person : ℕ)
  (h1 : num_rooms = 10) (h2 : num_people_per_room = 3) (h3 : towels_per_person = 2) :
  num_rooms * num_people_per_room * towels_per_person = 60 :=
by
  sorry

end hotel_towels_l182_182734


namespace triple_composition_l182_182000

def g (x : ℤ) : ℤ := 3 * x + 2

theorem triple_composition :
  g (g (g 3)) = 107 :=
by
  sorry

end triple_composition_l182_182000


namespace evaluate_expression_l182_182872

theorem evaluate_expression (x : ℝ) (h : x = 3) : (x^2 - 3 * x - 10) / (x - 5) = 5 :=
by
  sorry

end evaluate_expression_l182_182872


namespace intersection_eq_T_l182_182931

def S : Set ℤ := {s | ∃ n : ℤ, s = 2 * n + 1}
def T : Set ℤ := {t | ∃ n : ℤ, t = 4 * n + 1}

theorem intersection_eq_T : S ∩ T = T := 
sorry

end intersection_eq_T_l182_182931


namespace distinct_necklace_arrangements_8_beads_l182_182641

theorem distinct_necklace_arrangements_8_beads : 
  (nat.factorial 8 / (8 * 2)) = 2520 := by
  sorry

end distinct_necklace_arrangements_8_beads_l182_182641


namespace factorize_x2_plus_2x_l182_182192

theorem factorize_x2_plus_2x (x : ℝ) : x^2 + 2*x = x * (x + 2) :=
by sorry

end factorize_x2_plus_2x_l182_182192


namespace box_ratio_l182_182233

theorem box_ratio (h : ℤ) (l : ℤ) (w : ℤ) (v : ℤ)
  (H_height : h = 12)
  (H_length : l = 3 * h)
  (H_volume : l * w * h = 3888)
  (H_length_multiple : ∃ m, l = m * w) :
  l / w = 4 := by
  sorry

end box_ratio_l182_182233


namespace ratio_of_segments_intersecting_chords_l182_182558

open Real

variables (EQ FQ HQ GQ : ℝ)

theorem ratio_of_segments_intersecting_chords 
  (h1 : EQ = 5) 
  (h2 : GQ = 7) 
  (h3 : EQ * FQ = GQ * HQ) : 
  FQ / HQ = 7 / 5 :=
by
  sorry

end ratio_of_segments_intersecting_chords_l182_182558


namespace reflection_over_line_y_eq_x_l182_182400

theorem reflection_over_line_y_eq_x {x y x' y' : ℝ} (h_c : (x, y) = (6, -5)) (h_reflect : (x', y') = (y, x)) :
  (x', y') = (-5, 6) :=
  by
    simp [h_c, h_reflect]
    sorry

end reflection_over_line_y_eq_x_l182_182400


namespace intersection_eq_T_l182_182989

def S : Set ℤ := {s | ∃ n : ℤ, s = 2 * n + 1}
def T : Set ℤ := {t | ∃ n : ℤ, t = 4 * n + 1}

theorem intersection_eq_T : S ∩ T = T := sorry

end intersection_eq_T_l182_182989


namespace club_additional_members_l182_182723

theorem club_additional_members (current_members : ℕ) (additional_members : ℕ) :
  current_members = 10 →
  additional_members = (2 * current_members) + 5 - current_members →
  additional_members = 15 :=
by
  intro h1 h2
  rw [h1] at h2
  norm_num at h2
  exact h2

end club_additional_members_l182_182723


namespace percentage_of_students_in_60_to_69_range_is_20_l182_182453

theorem percentage_of_students_in_60_to_69_range_is_20 :
  let scores := [4, 8, 6, 5, 2]
  let total_students := scores.sum
  let students_in_60_to_69 := 5
  (students_in_60_to_69 * 100 / total_students) = 20 := by
  sorry

end percentage_of_students_in_60_to_69_range_is_20_l182_182453


namespace brick_width_l182_182572

def courtyard :=
  let length_m := 25
  let width_m := 16
  let length_cm := length_m * 100
  let width_cm := width_m * 100
  length_cm * width_cm

def total_bricks := 20000

def brick :=
  let length_cm := 20
  let width_cm := 10  -- This is our hypothesis to prove
  length_cm * width_cm

theorem brick_width (courtyard_area : courtyard) (total_required_bricks : total_bricks) :
  total_required_bricks * brick = courtyard_area := by
  sorry

end brick_width_l182_182572


namespace area_of_circle_l182_182867

theorem area_of_circle (x y : ℝ) :
  x^2 + y^2 + 8 * x + 10 * y = -9 → 
  ∃ a : ℝ, a = 32 * Real.pi :=
by
  sorry

end area_of_circle_l182_182867


namespace total_pies_eaten_l182_182694

variable (Adam Bill Sierra : ℕ)

axiom condition1 : Adam = Bill + 3
axiom condition2 : Sierra = 2 * Bill
axiom condition3 : Sierra = 12

theorem total_pies_eaten : Adam + Bill + Sierra = 27 :=
by
  -- Sorry is used to skip the proof
  sorry

end total_pies_eaten_l182_182694


namespace total_goals_scored_l182_182601

-- Definitions based on the problem conditions
def kickers_first_period_goals : ℕ := 2
def kickers_second_period_goals : ℕ := 2 * kickers_first_period_goals
def spiders_first_period_goals : ℕ := kickers_first_period_goals / 2
def spiders_second_period_goals : ℕ := 2 * kickers_second_period_goals

-- The theorem we need to prove
theorem total_goals_scored : 
  kickers_first_period_goals + kickers_second_period_goals +
  spiders_first_period_goals + spiders_second_period_goals = 15 := 
by
  -- proof steps will go here
  sorry

end total_goals_scored_l182_182601


namespace prob_four_sons_four_daughters_l182_182384

open Finset

theorem prob_four_sons_four_daughters 
  (n : ℕ := 8) 
  (p : ℚ := 1 / 2) 
  (k : ℕ := 4) :
  let total_combinations := 2^n in
  let favorable_combinations := nat.choose n k in
  (favorable_combinations : ℚ) / (total_combinations : ℚ) = 35 / 128 := 
by
  sorry

end prob_four_sons_four_daughters_l182_182384


namespace hotel_towels_l182_182735

def num_rooms : Nat := 10
def people_per_room : Nat := 3
def towels_per_person : Nat := 2

theorem hotel_towels : num_rooms * people_per_room * towels_per_person = 60 :=
by
  sorry

end hotel_towels_l182_182735


namespace billy_boxes_of_candy_l182_182104

theorem billy_boxes_of_candy (pieces_per_box total_pieces : ℕ) (h1 : pieces_per_box = 3) (h2 : total_pieces = 21) :
  total_pieces / pieces_per_box = 7 := 
by
  sorry

end billy_boxes_of_candy_l182_182104


namespace fathers_age_after_further_8_years_l182_182438

variable (R F : ℕ)

def age_relation_1 : Prop := F = 4 * R
def age_relation_2 : Prop := F + 8 = (5 * (R + 8)) / 2

theorem fathers_age_after_further_8_years (h1 : age_relation_1 R F) (h2 : age_relation_2 R F) : (F + 16) = 2 * (R + 16) :=
by 
  sorry

end fathers_age_after_further_8_years_l182_182438


namespace quadratic_solution_exists_for_any_a_b_l182_182388

theorem quadratic_solution_exists_for_any_a_b (a b : ℝ) : 
  ∃ x : ℝ, (a^6 - b^6)*x^2 + 2*(a^5 - b^5)*x + (a^4 - b^4) = 0 := 
by
  -- The proof would go here
  sorry

end quadratic_solution_exists_for_any_a_b_l182_182388


namespace range_of_a_l182_182702

theorem range_of_a (x a : ℝ) (h₁ : x > 1) (h₂ : a ≤ x + 1 / (x - 1)) : 
  a < 3 :=
sorry

end range_of_a_l182_182702


namespace negative_only_option_B_l182_182077

theorem negative_only_option_B :
  (0 > -3) ∧ 
  (|-3| = 3) ∧ 
  (0 < 3) ∧
  (0 < (1/3)) ∧
  ∀ x, x = -3 → x < 0 :=
by
  sorry

end negative_only_option_B_l182_182077


namespace log_eq_solution_l182_182354

open Real

noncomputable def solve_log_eq : Real :=
  let x := 62.5^(1/3)
  x

theorem log_eq_solution (x : Real) (hx : 3 * log x - 4 * log 5 = -1) :
  x = solve_log_eq :=
by
  sorry

end log_eq_solution_l182_182354


namespace intersection_S_T_eq_T_l182_182955

def S : Set ℤ := { s | ∃ n : ℤ, s = 2 * n + 1 }
def T : Set ℤ := { t | ∃ n : ℤ, t = 4 * n + 1 }

theorem intersection_S_T_eq_T : S ∩ T = T := by
  sorry

end intersection_S_T_eq_T_l182_182955


namespace rhombus_area_l182_182855

-- Declare the lengths of the diagonals
def diagonal1 := 6
def diagonal2 := 8

-- Define the area function for a rhombus
def area_of_rhombus (d1 d2 : ℕ) : ℕ :=
  (d1 * d2) / 2

-- State the theorem
theorem rhombus_area : area_of_rhombus diagonal1 diagonal2 = 24 := by sorry

end rhombus_area_l182_182855


namespace initial_guppies_l182_182295

theorem initial_guppies (total_gups : ℕ) (dozen_gups : ℕ) (extra_gups : ℕ) (baby_gups_initial : ℕ) (baby_gups_later : ℕ) :
  total_gups = 52 → dozen_gups = 12 → extra_gups = 3 → baby_gups_initial = 3 * 12 → baby_gups_later = 9 → 
  total_gups - (baby_gups_initial + baby_gups_later) = 7 :=
by
  intro h1 h2 h3 h4 h5
  sorry

end initial_guppies_l182_182295


namespace value_of_expression_at_three_l182_182870

theorem value_of_expression_at_three (x : ℝ) (h : x = 3) : (x^2 - 3 * x - 10) / (x - 5) = 5 := 
by
  sorry

end value_of_expression_at_three_l182_182870


namespace market_value_of_share_l182_182079

-- Definitions from the conditions
def nominal_value : ℝ := 48
def dividend_rate : ℝ := 0.09
def desired_interest_rate : ℝ := 0.12

-- The proof problem (theorem statement) in Lean 4
theorem market_value_of_share : (nominal_value * dividend_rate / desired_interest_rate * 100) = 36 := 
by
  sorry

end market_value_of_share_l182_182079


namespace cat_food_sufficiency_l182_182325

variable (L S : ℝ)

theorem cat_food_sufficiency (h : L + 4 * S = 14) : L + 3 * S ≥ 11 :=
sorry

end cat_food_sufficiency_l182_182325


namespace students_per_group_l182_182265

-- Definitions for conditions
def number_of_boys : ℕ := 28
def number_of_girls : ℕ := 4
def number_of_groups : ℕ := 8
def total_students : ℕ := number_of_boys + number_of_girls

-- The Theorem we want to prove
theorem students_per_group : total_students / number_of_groups = 4 := by
  sorry

end students_per_group_l182_182265


namespace height_of_triangle_l182_182046

theorem height_of_triangle
    (A : ℝ) (b : ℝ) (h : ℝ)
    (h1 : A = 30)
    (h2 : b = 12)
    (h3 : A = (b * h) / 2) :
    h = 5 :=
by
  sorry

end height_of_triangle_l182_182046


namespace common_root_l182_182345

theorem common_root (p : ℝ) :
  (∃ x : ℝ, x^2 - (p+2)*x + 2*p + 6 = 0 ∧ 2*x^2 - (p+4)*x + 2*p + 3 = 0) ↔ (p = -3 ∨ p = 9) :=
by
  sorry

end common_root_l182_182345


namespace solution_of_inequality_l182_182552

open Set

theorem solution_of_inequality (x : ℝ) :
  x^2 - 2 * x - 3 > 0 ↔ x < -1 ∨ x > 3 :=
by
  sorry

end solution_of_inequality_l182_182552


namespace intersection_S_T_eq_T_l182_182951

def S : Set ℤ := { s | ∃ n : ℤ, s = 2 * n + 1 }
def T : Set ℤ := { t | ∃ n : ℤ, t = 4 * n + 1 }

theorem intersection_S_T_eq_T : S ∩ T = T := by
  sorry

end intersection_S_T_eq_T_l182_182951


namespace smallest_stable_triangle_side_length_l182_182830

/-- The smallest possible side length that can appear in any stable triangle with side lengths that 
are multiples of 5, 80, and 112, respectively, is 20. -/
theorem smallest_stable_triangle_side_length {a b c : ℕ} 
  (hab : ∃ k₁, a = 5 * k₁) 
  (hbc : ∃ k₂, b = 80 * k₂) 
  (hac : ∃ k₃, c = 112 * k₃) 
  (abc_triangle : a + b > c ∧ a + c > b ∧ b + c > a) : 
  a = 20 ∨ b = 20 ∨ c = 20 :=
sorry

end smallest_stable_triangle_side_length_l182_182830


namespace sean_needs_six_packs_l182_182530

/-- 
 Sean needs to replace 2 light bulbs in the bedroom, 
 1 in the bathroom, 1 in the kitchen, and 4 in the basement. 
 He also needs to replace 1/2 of that amount in the garage. 
 The bulbs come 2 per pack. 
 -/
def bedroom_bulbs: ℕ := 2
def bathroom_bulbs: ℕ := 1
def kitchen_bulbs: ℕ := 1
def basement_bulbs: ℕ := 4
def bulbs_per_pack: ℕ := 2

noncomputable def total_bulbs_needed_including_garage: ℕ := 
  let total_rooms_bulbs := bedroom_bulbs + bathroom_bulbs + kitchen_bulbs + basement_bulbs
  let garage_bulbs := total_rooms_bulbs / 2
  total_rooms_bulbs + garage_bulbs

noncomputable def total_packs_needed: ℕ := total_bulbs_needed_including_garage / bulbs_per_pack

theorem sean_needs_six_packs : total_packs_needed = 6 :=
by
  sorry

end sean_needs_six_packs_l182_182530


namespace side_length_square_l182_182570

theorem side_length_square (x : ℝ) (h1 : x^2 = 2 * (4 * x)) : x = 8 :=
by
  sorry

end side_length_square_l182_182570


namespace dealership_sales_prediction_l182_182462

theorem dealership_sales_prediction (sports_cars_sold sedans SUVs : ℕ) 
    (ratio_sc_sedans : 3 * sedans = 5 * sports_cars_sold) 
    (ratio_sc_SUVs : sports_cars_sold = 2 * SUVs) 
    (sports_cars_sold_next_month : sports_cars_sold = 36) :
    (sedans = 60 ∧ SUVs = 72) :=
sorry

end dealership_sales_prediction_l182_182462


namespace soda_cans_purchasable_l182_182253

theorem soda_cans_purchasable (S Q : ℕ) (t D : ℝ) (hQ_pos : Q > 0) :
    let quarters_from_dollars := 4 * D
    let total_quarters_with_tax := quarters_from_dollars * (1 + t)
    (total_quarters_with_tax / Q) * S = (4 * D * S * (1 + t)) / Q :=
sorry

end soda_cans_purchasable_l182_182253


namespace cos_90_eq_0_l182_182143

theorem cos_90_eq_0 :
  ∃ (p : ℝ × ℝ), p = (0, 1) ∧ ∀ θ : ℝ, θ = 90 → cos θ = p.1 :=
by
  let p := (0, 1)
  use p
  split
  · rfl
  · intros θ h
    rw h
    sorry

end cos_90_eq_0_l182_182143


namespace count_valid_numbers_l182_182626

-- Let n be the number of four-digit numbers greater than 3999 with the product of the middle two digits exceeding 10.
def n : ℕ := 3480

-- Formalize the given conditions:
def is_valid_four_digit (a b c d : ℕ) : Prop :=
  (4 ≤ a ∧ a ≤ 9) ∧ (0 ≤ d ∧ d ≤ 9) ∧ (1 ≤ b ∧ b ≤ 9) ∧ (1 ≤ c ∧ c ≤ 9) ∧ (b * c > 10)

-- The theorem to prove the number of valid four-digit numbers is 3480
theorem count_valid_numbers : 
  (∑ (a b c d : ℕ) in finset.range 10 × finset.range 10 × finset.range 10 × finset.range 10,
    if is_valid_four_digit a b c d then 1 else 0) = n := sorry

end count_valid_numbers_l182_182626


namespace cos_of_90_degrees_l182_182167

-- Definition of cosine of 90 degrees
def cos_90_degrees : ℝ := Real.cos (90 * (Float.pi / 180))

-- Condition: Cosine is the x-coordinate of the point (0, 1) after 90 degree rotation
theorem cos_of_90_degrees : cos_90_degrees = 0 :=
by
  -- Proof would go here, but we skip it with sorry
  sorry

end cos_of_90_degrees_l182_182167


namespace solve_for_question_mark_l182_182883

theorem solve_for_question_mark :
  let question_mark := 4135 / 45
  (45 * question_mark) + (625 / 25) - (300 * 4) = 2950 + (1500 / (75 * 2)) :=
by
  let question_mark := 4135 / 45
  sorry

end solve_for_question_mark_l182_182883


namespace girl_from_grade_4_probability_l182_182818

-- Number of girls and boys in grade 3
def girls_grade_3 := 28
def boys_grade_3 := 35
def total_grade_3 := girls_grade_3 + boys_grade_3

-- Number of girls and boys in grade 4
def girls_grade_4 := 45
def boys_grade_4 := 42
def total_grade_4 := girls_grade_4 + boys_grade_4

-- Number of girls and boys in grade 5
def girls_grade_5 := 38
def boys_grade_5 := 51
def total_grade_5 := girls_grade_5 + boys_grade_5

-- Total number of children in playground
def total_children := total_grade_3 + total_grade_4 + total_grade_5

-- Probability that a randomly selected child is a girl from grade 4
def probability_girl_grade_4 := (girls_grade_4: ℚ) / total_children

theorem girl_from_grade_4_probability :
  probability_girl_grade_4 = 45 / 239 := by
  sorry

end girl_from_grade_4_probability_l182_182818


namespace kim_hours_of_classes_per_day_l182_182019

-- Definitions based on conditions
def original_classes : Nat := 4
def hours_per_class : Nat := 2
def dropped_classes : Nat := 1

-- Prove that Kim now has 6 hours of classes per day
theorem kim_hours_of_classes_per_day : (original_classes - dropped_classes) * hours_per_class = 6 := by
  sorry

end kim_hours_of_classes_per_day_l182_182019


namespace complex_multiplication_l182_182401

variable (i : ℂ)
axiom i_square : i^2 = -1

theorem complex_multiplication : i * (1 + i) = -1 + i :=
by
  sorry

end complex_multiplication_l182_182401


namespace parabola_inequality_l182_182622

theorem parabola_inequality (a c y1 y2 : ℝ) (h1 : a < 0)
  (h2 : y1 = a * (-1 - 1)^2 + c)
  (h3 : y2 = a * (4 - 1)^2 + c) :
  y1 > y2 :=
sorry

end parabola_inequality_l182_182622


namespace exists_non_prime_form_l182_182901

theorem exists_non_prime_form (n : ℕ) : ∃ n : ℕ, ¬Nat.Prime (n^2 + n + 41) :=
sorry

end exists_non_prime_form_l182_182901


namespace random_variable_prob_l182_182242

theorem random_variable_prob (n : ℕ) (h : (3 : ℝ) / n = 0.3) : n = 10 :=
sorry

end random_variable_prob_l182_182242


namespace parabola_equation_l182_182681

theorem parabola_equation (a b c d e f : ℤ)
  (h1 : a = 0 )    -- The equation should have no x^2 term
  (h2 : b = 0 )    -- The equation should have no xy term
  (h3 : c > 0)     -- The coefficient of y^2 should be positive
  (h4 : d = -2)    -- The coefficient of x in the final form should be -2
  (h5 : e = -8)    -- The coefficient of y in the final form should be -8
  (h6 : f = 16)    -- The constant term in the final form should be 16
  (pass_through : (2 : ℤ) = k * (6 - 4) ^ 2)
  (vertex : (0 : ℤ) = k * (sym_axis - 4) ^ 2)
  (symmetry_axis_parallel_x : True)
  (vertex_on_y_axis : True):
  ax^2 + bxy + cy^2 + dx + ey + f = 0 :=
by
  sorry

end parabola_equation_l182_182681


namespace distinct_solutions_of_transformed_eq_l182_182545

open Function

variable {R : Type} [Field R]

def cubic_func (a b c d : R) (x : R) : R := a*x^3 + b*x^2 + c*x + d

noncomputable def three_distinct_roots {a b c d : R} (f : R → R)
  (h : ∀ x, f x = a*x^3 + b*x^2 + c*x + d) : Prop :=
∃ α β γ, α ≠ β ∧ β ≠ γ ∧ γ ≠ α ∧ f α = 0 ∧ f β = 0 ∧ f γ = 0

theorem distinct_solutions_of_transformed_eq
  {a b c d : R} (h : ∃ α β γ, α ≠ β ∧ β ≠ γ ∧ γ ≠ α ∧ (cubic_func a b c d α) = 0 ∧ (cubic_func a b c d β) = 0 ∧ (cubic_func a b c d γ) = 0) :
  ∃ p q, p ≠ q ∧ (4 * (cubic_func a b c d p) * (3 * a * p + b) = (3 * a * p^2 + 2 * b * p + c)^2) ∧ 
              (4 * (cubic_func a b c d q) * (3 * a * q + b) = (3 * a * q^2 + 2 * b * q + c)^2) := sorry

end distinct_solutions_of_transformed_eq_l182_182545


namespace stickers_after_loss_l182_182417

-- Conditions
def stickers_per_page : ℕ := 20
def initial_pages : ℕ := 12
def lost_pages : ℕ := 1

-- Problem statement
theorem stickers_after_loss : (initial_pages - lost_pages) * stickers_per_page = 220 := by
  sorry

end stickers_after_loss_l182_182417


namespace matthews_contribution_l182_182524

theorem matthews_contribution 
  (total_cost : ℝ) (yen_amount : ℝ) (conversion_rate : ℝ)
  (h1 : total_cost = 18)
  (h2 : yen_amount = 2500)
  (h3 : conversion_rate = 140) :
  (total_cost - (yen_amount / conversion_rate)) = 0.143 :=
by sorry

end matthews_contribution_l182_182524


namespace cat_food_sufficiency_l182_182307

variable (L S : ℝ)

theorem cat_food_sufficiency (h : L + 4 * S = 14) : L + 3 * S >= 11 :=
sorry

end cat_food_sufficiency_l182_182307


namespace simplification_and_evaluation_l182_182848

theorem simplification_and_evaluation (a : ℚ) (h : a = -1 / 2) :
  (3 * a + 2) * (a - 1) - 4 * a * (a + 1) = 1 / 4 := 
by
  sorry

end simplification_and_evaluation_l182_182848


namespace time_difference_for_x_miles_l182_182036

def time_old_shoes (n : Nat) : Int := 10 * n
def time_new_shoes (n : Nat) : Int := 13 * n
def time_difference_for_5_miles : Int := time_new_shoes 5 - time_old_shoes 5

theorem time_difference_for_x_miles (x : Nat) (h : time_difference_for_5_miles = 15) : 
  time_new_shoes x - time_old_shoes x = 3 * x := 
by
  sorry

end time_difference_for_x_miles_l182_182036


namespace cos_90_eq_zero_l182_182178

theorem cos_90_eq_zero : Real.cos (90 * Real.pi / 180) = 0 := by 
  sorry

end cos_90_eq_zero_l182_182178


namespace cos_90_equals_0_l182_182125

-- Define the question: cos(90 degrees)
def cos_90_degrees : ℝ := Real.cos (Real.pi / 2)

-- State that cos(90 degrees) equals 0
theorem cos_90_equals_0 : cos_90_degrees = 0 :=
by sorry

end cos_90_equals_0_l182_182125


namespace cat_food_problem_l182_182322

noncomputable def L : ℝ
noncomputable def S : ℝ

theorem cat_food_problem (L S : ℝ) (h : L + 4 * S = 14) : L + 3 * S ≥ 11 :=
by 
  sorry

end cat_food_problem_l182_182322


namespace radius_of_sphere_touching_four_l182_182925

noncomputable def r_sphere_internally_touching_four := Real.sqrt (3 / 2) + 1
noncomputable def r_sphere_externally_touching_four := Real.sqrt (3 / 2) - 1

theorem radius_of_sphere_touching_four (r : ℝ) (R := Real.sqrt (3 / 2)) :
  r = R + 1 ∨ r = R - 1 :=
by
  sorry

end radius_of_sphere_touching_four_l182_182925


namespace number_of_cars_l182_182550

theorem number_of_cars (b c : ℕ) (h1 : b = c / 10) (h2 : c - b = 90) : c = 100 :=
by
  sorry

end number_of_cars_l182_182550


namespace find_theta_l182_182483

open Real

noncomputable def vector_a : ℝ × ℝ := (1, 2)
noncomputable def vector_b : ℝ × ℝ := (1, -1)

noncomputable def vector_c : ℝ × ℝ := (2 • vector_a.1 + vector_b.1, 2 • vector_a.2 + vector_b.2)
noncomputable def vector_d : ℝ × ℝ := (vector_a.1 - vector_b.1, vector_a.2 - vector_b.2)

-- Definition of the dot product
noncomputable def dot_prod (v1 v2 : ℝ × ℝ) : ℝ := v1.1 * v2.1 + v1.2 * v2.2

-- Definition of the magnitude of a vector
noncomputable def magnitude (v : ℝ × ℝ) : ℝ := real.sqrt (v.1 * v.1 + v.2 * v.2)

-- Definition of cosθ
noncomputable def cos_theta : ℝ := dot_prod vector_c vector_d / (magnitude vector_c * magnitude vector_d)

theorem find_theta : acos cos_theta = π / 4 :=
by
  sorry

end find_theta_l182_182483


namespace cos_90_eq_zero_l182_182177

theorem cos_90_eq_zero : Real.cos (90 * Real.pi / 180) = 0 := by 
  sorry

end cos_90_eq_zero_l182_182177


namespace compare_neg_rational_l182_182112

def neg_one_third : ℚ := -1 / 3
def neg_one_half : ℚ := -1 / 2

theorem compare_neg_rational : neg_one_third > neg_one_half :=
by sorry

end compare_neg_rational_l182_182112


namespace range_of_a_l182_182490

theorem range_of_a (a : ℝ) (f : ℝ → ℝ) (x1 x2 : ℝ)
  (h1 : 0 < x1) (h2 : x1 < x2) (h3 : x2 ≤ 1) (f_def : ∀ x, f x = a * x - x^3)
  (condition : f x2 - f x1 > x2 - x1) :
  a ≥ 4 :=
by sorry

end range_of_a_l182_182490


namespace marbles_total_l182_182477

theorem marbles_total (fabian kyle miles : ℕ) (h1 : fabian = 3 * kyle) (h2 : fabian = 5 * miles) (h3 : fabian = 15) : kyle + miles = 8 := by
  sorry

end marbles_total_l182_182477


namespace club_members_addition_l182_182726

theorem club_members_addition
  (current_members : ℕ := 10)
  (desired_members : ℕ := 2 * current_members + 5)
  (additional_members : ℕ := desired_members - current_members) :
  additional_members = 15 :=
by
  -- proof placeholder
  sorry

end club_members_addition_l182_182726


namespace cos_90_eq_zero_l182_182152

-- Define cosine function and specify its behavior on the unit circle.
def cos (θ : ℝ) : ℝ :=
  -- Cosine gives the x-coordinate of the point on the unit circle.
  sorry -- Definition of cosine function is assumed.

-- Statement to be proved that involves the cosine function.
theorem cos_90_eq_zero : cos (90 * π / 180) = 0 := 
by
  sorry -- Proof is omitted.

end cos_90_eq_zero_l182_182152


namespace min_students_green_eyes_backpack_no_glasses_l182_182919

theorem min_students_green_eyes_backpack_no_glasses
  (S G B Gl : ℕ)
  (h_S : S = 25)
  (h_G : G = 15)
  (h_B : B = 18)
  (h_Gl : Gl = 6)
  : ∃ x, x ≥ 8 ∧ x + Gl ≤ S ∧ x ≤ min G B :=
sorry

end min_students_green_eyes_backpack_no_glasses_l182_182919


namespace cost_of_ice_cream_scoop_l182_182190

theorem cost_of_ice_cream_scoop
  (num_meals : ℕ) (meal_cost : ℕ) (total_money : ℕ)
  (total_meals_cost : num_meals * meal_cost = 30)
  (remaining_money : total_money - 30 = 15)
  (num_ice_cream_scoops : ℕ) (cost_per_scoop : ℕ)
  (total_cost : 30 + 15 = total_money)
  (total_ice_cream_cost : num_ice_cream_scoops * cost_per_scoop = remaining_money) :
  cost_per_scoop = 5 :=
by
  have h_num_meals : num_meals = 3 := by sorry
  have h_meal_cost : meal_cost = 10 := by sorry
  have h_total_money : total_money = 45 := by sorry
  have h_num_ice_cream_scoops : num_ice_cream_scoops = 3 := by sorry
  exact sorry

end cost_of_ice_cream_scoop_l182_182190


namespace compare_sums_of_sines_l182_182373

theorem compare_sums_of_sines {A B C : ℝ} 
  (hA : 0 < A) (hB : 0 < B) (hC : 0 < C)
  (h_sum : A + B + C = π) :
  (if A < π / 2 ∧ B < π / 2 ∧ C < π / 2 then
    (1 / (Real.sin (2 * A)) + 1 / (Real.sin (2 * B)) + 1 / (Real.sin (2 * C))
      ≥ 1 / (Real.sin A) + 1 / (Real.sin B) + 1 / (Real.sin C))
  else
    (1 / (Real.sin (2 * A)) + 1 / (Real.sin (2 * B)) + 1 / (Real.sin (2 * C))
      < 1 / (Real.sin A) + 1 / (Real.sin B) + 1 / (Real.sin C))) :=
sorry

end compare_sums_of_sines_l182_182373


namespace max_value_of_f_l182_182549

noncomputable def f (theta x : ℝ) : ℝ :=
  (Real.cos theta)^2 - 2 * x * Real.cos theta - 1

noncomputable def M (x : ℝ) : ℝ :=
  if 0 <= x then 
    2 * x
  else 
    -2 * x

theorem max_value_of_f {x : ℝ} : 
  ∃ theta : ℝ, Real.cos theta ∈ [-1, 1] ∧ f theta x = M x :=
by
  sorry

end max_value_of_f_l182_182549


namespace equation_of_chord_l182_182790

-- Define the ellipse equation and point P
def ellipse_eq (x y : ℝ) : Prop := 4 * x^2 + 9 * y^2 = 144
def P : ℝ × ℝ := (3, 2)
def is_midpoint (A B P : ℝ × ℝ) : Prop := A.1 + B.1 = 2 * P.1 ∧ A.2 + B.2 = 2 * P.2
def on_chord (A B : ℝ × ℝ) (x y : ℝ) : Prop := (x - A.1) * (B.2 - A.2) = (y - A.2) * (B.1 - A.1)

-- Lean Statement
theorem equation_of_chord :
  ∀ A B : ℝ × ℝ,
    ellipse_eq A.1 A.2 →
    ellipse_eq B.1 B.2 →
    is_midpoint A B P →
    ∀ x y : ℝ,
      on_chord A B x y →
      2 * x + 3 * y = 12 :=
by
  sorry

end equation_of_chord_l182_182790


namespace problem_statement_l182_182218

theorem problem_statement (a b : ℝ) (h : |a - 1| + (b + 2)^2 = 0) : (a + b) ^ 2002 + a ^ 2001 = 2 := 
by 
  sorry

end problem_statement_l182_182218


namespace value_of_a_l182_182613

theorem value_of_a (a : ℝ) :
  (∀ x : ℝ, -2 ≤ x ∧ x ≤ 1 → |a * x + 1| ≤ 3) ↔ a = 2 :=
by
  sorry

end value_of_a_l182_182613


namespace values_of_m_zero_rain_l182_182374

def f (x y : ℝ) : ℝ := abs (x^3 + 2*x^2*y - 5*x*y^2 - 6*y^3)

theorem values_of_m_zero_rain :
  {m : ℝ | ∀ x : ℝ, f x (m * x) = 0} = {-1, 1/2, -1/3} :=
sorry

end values_of_m_zero_rain_l182_182374


namespace pins_after_one_month_l182_182845

def avg_pins_per_day : ℕ := 10
def delete_pins_per_week_per_person : ℕ := 5
def group_size : ℕ := 20
def initial_pins : ℕ := 1000

theorem pins_after_one_month
  (avg_pins_per_day_pos : avg_pins_per_day = 10)
  (delete_pins_per_week_per_person_pos : delete_pins_per_week_per_person = 5)
  (group_size_pos : group_size = 20)
  (initial_pins_pos : initial_pins = 1000) : 
  1000 + (avg_pins_per_day * group_size * 30) - (delete_pins_per_week_per_person * group_size * 4) = 6600 :=
by
  sorry

end pins_after_one_month_l182_182845


namespace schools_participating_l182_182603

noncomputable def num_schools (students_per_school : ℕ) (total_students : ℕ) : ℕ :=
  total_students / students_per_school

theorem schools_participating (students_per_school : ℕ) (beth_rank : ℕ) 
  (carla_rank : ℕ) (highest_on_team : ℕ) (n : ℕ) :
  students_per_school = 4 ∧ beth_rank = 46 ∧ carla_rank = 79 ∧
  (∀ i, i ≤ 46 → highest_on_team = 40) → 
  num_schools students_per_school ((2 * highest_on_team) - 1) = 19 := 
by
  intros h
  sorry

end schools_participating_l182_182603


namespace BD_value_l182_182645

noncomputable def triangleBD (AC BC AD CD : ℝ) : ℝ :=
  let θ := Real.arccos ((3 ^ 2 + 9 ^ 2 - 7 ^ 2) / (2 * 3 * 9))
  let ψ := Real.pi - θ
  let cosψ := Real.cos ψ
  let x := (-1.026 + Real.sqrt ((1.026 ^ 2) + 4 * 40)) / 2
  if x > 0 then x else 5.8277 -- confirmed manually as positive root.

theorem BD_value : (triangleBD 7 7 9 3) = 5.8277 :=
by
  apply sorry

end BD_value_l182_182645


namespace magic_square_exists_l182_182290

theorem magic_square_exists : 
  ∃ (a b c d e f g h : ℕ),
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧ a ≠ g ∧ a ≠ h ∧ 
    b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧ b ≠ g ∧ b ≠ h ∧ 
    c ≠ d ∧ c ≠ e ∧ c ≠ f ∧ c ≠ g ∧ c ≠ h ∧ 
    d ≠ e ∧ d ≠ f ∧ d ≠ g ∧ d ≠ h ∧ 
    e ≠ f ∧ e ≠ g ∧ e ≠ h ∧ 
    f ≠ g ∧ f ≠ h ∧
    g ≠ h ∧
    a + b + c = 12 ∧ d + e + f = 12 ∧ g + h + 0 = 12 ∧
    a + d + g = 12 ∧ b + 0 + h = 12 ∧ c + f + 0 = 12 :=
sorry

end magic_square_exists_l182_182290


namespace intersection_eq_T_l182_182939

def S : Set ℤ := {s | ∃ n : ℤ, s = 2 * n + 1}
def T : Set ℤ := {t | ∃ n : ℤ, t = 4 * n + 1}

theorem intersection_eq_T : S ∩ T = T := 
sorry

end intersection_eq_T_l182_182939


namespace min_value_of_quadratic_l182_182471

theorem min_value_of_quadratic (x : ℝ) : ∃ y, y = x^2 + 14*x + 20 ∧ ∀ z, z = x^2 + 14*x + 20 → z ≥ -29 :=
by
  sorry

end min_value_of_quadratic_l182_182471


namespace discount_difference_l182_182094

theorem discount_difference (P : ℝ) (h₁ : 0 < P) : 
  let actual_combined_discount := 1 - (0.75 * 0.85)
  let claimed_discount := 0.40
  actual_combined_discount - claimed_discount = 0.0375 :=
by 
  sorry

end discount_difference_l182_182094


namespace max_value_2x_plus_y_l182_182200

def max_poly_value : ℝ :=
  sorry

theorem max_value_2x_plus_y (x y : ℝ) (h1 : x + 2 * y ≤ 3) (h2 : 0 ≤ x) (h3 : 0 ≤ y) : 
  2 * x + y ≤ 6 :=
sorry

example (x y : ℝ) (h1 : x + 2 * y ≤ 3) (h2 : 0 ≤ x) (h3 : 0 ≤ y) : 2 * x + y = 6 
  ↔ x = 3 ∧ y = 0 :=
by exact sorry

end max_value_2x_plus_y_l182_182200


namespace S_inter_T_eq_T_l182_182959

-- Define the sets S and T
def S := { s : ℤ | ∃ n : ℤ, s = 2 * n + 1 }
def T := { t : ℤ | ∃ n : ℤ, t = 4 * n + 1 }

-- Prove that S ∩ T = T
theorem S_inter_T_eq_T : S ∩ T = T := 
by
  sorry

end S_inter_T_eq_T_l182_182959


namespace greatest_possible_perimeter_l182_182819

theorem greatest_possible_perimeter :
  ∃ (x : ℤ), x ≥ 4 ∧ x ≤ 5 ∧ (x + 4 * x + 18 = 43 ∧
    ∀ (y : ℤ), y ≥ 4 ∧ y ≤ 5 → y + 4 * y + 18 ≤ 43) :=
by
  sorry

end greatest_possible_perimeter_l182_182819


namespace tetrahedron_probability_correct_l182_182741

noncomputable def tetrahedron_probability : ℚ :=
  let total_arrangements := 16
  let suitable_arrangements := 2
  suitable_arrangements / total_arrangements

theorem tetrahedron_probability_correct : tetrahedron_probability = 1 / 8 :=
by
  sorry

end tetrahedron_probability_correct_l182_182741


namespace point_in_third_quadrant_l182_182810

theorem point_in_third_quadrant (m : ℝ) (h1 : m < 0) (h2 : 4 + 2 * m < 0) : m < -2 :=
by
  sorry

end point_in_third_quadrant_l182_182810


namespace intersection_eq_T_l182_182990

def S : Set ℤ := {s | ∃ n : ℤ, s = 2 * n + 1}
def T : Set ℤ := {t | ∃ n : ℤ, t = 4 * n + 1}

theorem intersection_eq_T : S ∩ T = T := sorry

end intersection_eq_T_l182_182990


namespace evaluate_expression_l182_182873

theorem evaluate_expression (x : ℝ) (h : x = 3) : (x^2 - 3 * x - 10) / (x - 5) = 5 :=
by
  sorry

end evaluate_expression_l182_182873


namespace gnome_count_l182_182396

theorem gnome_count (g_R: ℕ) (g_W: ℕ) (h1: g_R = 4 * g_W) (h2: g_W = 20) : g_R - (40 * g_R / 100) = 48 := by
  sorry

end gnome_count_l182_182396


namespace John_l182_182015

theorem John's_net_profit 
  (gross_income : ℕ)
  (car_purchase_cost : ℕ)
  (car_maintenance : ℕ → ℕ → ℕ)
  (car_insurance : ℕ)
  (car_tire_replacement : ℕ)
  (trade_in_value : ℕ)
  (tax_rate : ℚ)
  (total_taxes : ℕ)
  (monthly_maintenance_cost : ℕ)
  (months : ℕ)
  (net_profit : ℕ) :
  gross_income = 30000 →
  car_purchase_cost = 20000 →
  car_maintenance monthly_maintenance_cost months = 3600 →
  car_insurance = 1200 →
  car_tire_replacement = 400 →
  trade_in_value = 6000 →
  tax_rate = 15/100 →
  total_taxes = 4500 →
  monthly_maintenance_cost = 300 →
  months = 12 →
  net_profit = gross_income - (car_purchase_cost + car_maintenance monthly_maintenance_cost months + car_insurance + car_tire_replacement + total_taxes) + trade_in_value →
  net_profit = 6300 := 
by 
  sorry -- Proof to be provided

end John_l182_182015


namespace blocks_per_tree_l182_182040

def trees_per_day : ℕ := 2
def blocks_after_5_days : ℕ := 30
def days : ℕ := 5

theorem blocks_per_tree : (blocks_after_5_days / (trees_per_day * days)) = 3 :=
by
  sorry

end blocks_per_tree_l182_182040


namespace total_spending_is_450_l182_182656

-- Define the costs of items bought by Leonard
def leonard_wallet_cost : ℕ := 50
def pair_of_sneakers_cost : ℕ := 100
def pairs_of_sneakers : ℕ := 2

-- Define the costs of items bought by Michael
def michael_backpack_cost : ℕ := 100
def pair_of_jeans_cost : ℕ := 50
def pairs_of_jeans : ℕ := 2

-- Define the total spending of Leonard and Michael 
def total_spent : ℕ :=
  leonard_wallet_cost + (pair_of_sneakers_cost * pairs_of_sneakers) + 
  michael_backpack_cost + (pair_of_jeans_cost * pairs_of_jeans)

-- The proof statement
theorem total_spending_is_450 : total_spent = 450 := 
by
  sorry

end total_spending_is_450_l182_182656


namespace green_ball_removal_l182_182715

variable (total_balls : ℕ)
variable (initial_green_balls : ℕ)
variable (initial_yellow_balls : ℕ)
variable (desired_green_percentage : ℚ)
variable (removals : ℕ)

theorem green_ball_removal :
  initial_green_balls = 420 → 
  total_balls = 600 → 
  desired_green_percentage = 3 / 5 →
  (420 - removals) / (600 - removals) = desired_green_percentage → 
  removals = 150 :=
sorry

end green_ball_removal_l182_182715


namespace normal_dist_probability_l182_182057

-- Definitions
def xi : Type := ℝ -- The type for our random variable, which is real numbers
def ξ_pdf (x : ℝ) : ℝ := 
  1 / (5 * (real.pi ^ (1 / 2))) * real.exp (-((x - 100) ^ 2) / (2 * 5 ^ 2))

-- Given conditions
axiom H_normal : ∀ x : ℝ, pdf xi x = ξ_pdf x
axiom P_xi_110 : P(λ x, x < 110) xi = 0.98

-- The statement we need to prove
theorem normal_dist_probability :
  P (λ x, 90 < x ∧ x < 100) xi = 0.48 :=
sorry

end normal_dist_probability_l182_182057


namespace one_inch_cubes_with_red_paint_at_least_two_faces_l182_182062

theorem one_inch_cubes_with_red_paint_at_least_two_faces
  (number_of_one_inch_cubes : ℕ)
  (cubes_with_three_faces : ℕ)
  (cubes_with_two_faces : ℕ)
  (total_cubes_with_at_least_two_faces : ℕ) :
  number_of_one_inch_cubes = 64 →
  cubes_with_three_faces = 8 →
  cubes_with_two_faces = 24 →
  total_cubes_with_at_least_two_faces = cubes_with_three_faces + cubes_with_two_faces →
  total_cubes_with_at_least_two_faces = 32 :=
by
  sorry

end one_inch_cubes_with_red_paint_at_least_two_faces_l182_182062


namespace sum_of_three_numbers_l182_182683

variable (x y z : ℝ)

theorem sum_of_three_numbers :
  y = 5 → 
  (x + y + z) / 3 = x + 10 →
  (x + y + z) / 3 = z - 15 →
  x + y + z = 30 :=
by
  intros hy h1 h2
  rw [hy] at h1 h2
  sorry

end sum_of_three_numbers_l182_182683


namespace teams_in_league_l182_182427

def number_of_teams (n : ℕ) := n * (n - 1) / 2

theorem teams_in_league : ∃ n : ℕ, number_of_teams n = 36 ∧ n = 9 := by
  sorry

end teams_in_league_l182_182427


namespace at_most_2n_div_3_good_triangles_l182_182616

-- Definitions based on problem conditions
universe u

structure Polygon (α : Type u) :=
(vertices : List α)
(convex : True)  -- Placeholder for convexity condition

-- Definition for a good triangle
structure Triangle (α : Type u) :=
(vertices : Fin 3 → α)
(unit_length : (Fin 3) → (Fin 3) → Bool)  -- Placeholder for unit length side condition

noncomputable def count_good_triangles {α : Type u} [Inhabited α] (P : Polygon α) : Nat := sorry

theorem at_most_2n_div_3_good_triangles {α : Type u} [Inhabited α] (P : Polygon α) :
  count_good_triangles P ≤ P.vertices.length * 2 / 3 := 
sorry

end at_most_2n_div_3_good_triangles_l182_182616


namespace grazing_months_b_l182_182879

theorem grazing_months_b (a_oxen a_months b_oxen c_oxen c_months total_rent c_share : ℕ) (x : ℕ) 
  (h_a : a_oxen = 10) (h_am : a_months = 7) (h_b : b_oxen = 12) 
  (h_c : c_oxen = 15) (h_cm : c_months = 3) (h_tr : total_rent = 105) 
  (h_cs : c_share = 27) : 
  45 * 105 = 27 * (70 + 12 * x + 45) → x = 5 :=
by
  sorry

end grazing_months_b_l182_182879


namespace area_of_triangle_l182_182352

theorem area_of_triangle (m : ℝ) 
  (h : ∀ x y : ℝ, ((m + 3) * x + y = 3 * m - 4) → 
                  (7 * x + (5 - m) * y - 8 ≠ 0)
  ) : ((m = -2) → (1/2) * 2 * 2 = 2) := 
by {
  sorry
}

end area_of_triangle_l182_182352


namespace wage_difference_l182_182708

-- Definitions of the problem
variables (P Q h : ℝ)
axiom total_pay : P * h = 480
axiom wage_relation : P = 1.5 * Q
axiom time_relation : Q * (h + 10) = 480

-- Theorem to prove the hourly wage difference
theorem wage_difference : P - Q = 8 :=
by
  sorry

end wage_difference_l182_182708


namespace dante_age_l182_182591

def combined_age (D : ℕ) : ℕ := D + D / 2 + (D + 1)

theorem dante_age :
  ∃ D : ℕ, combined_age D = 31 ∧ D = 12 :=
by
  sorry

end dante_age_l182_182591


namespace find_initial_volume_l182_182891

noncomputable def initial_volume_of_solution (V : ℝ) : Prop :=
  let initial_jasmine := 0.05 * V
  let added_jasmine := 8
  let added_water := 2
  let new_total_volume := V + added_jasmine + added_water
  let new_jasmine := 0.125 * new_total_volume
  initial_jasmine + added_jasmine = new_jasmine

theorem find_initial_volume : ∃ V : ℝ, initial_volume_of_solution V ∧ V = 90 :=
by
  use 90
  unfold initial_volume_of_solution
  sorry

end find_initial_volume_l182_182891


namespace one_heads_one_tails_probability_l182_182875

def outcomes : List (String × String) := [("H", "H"), ("H", "T"), ("T", "H"), ("T", "T")]

def favorable_outcomes (outcome : String × String) : Bool :=
  (outcome = ("H", "T")) ∨ (outcome = ("T", "H"))

def probability_of_favorable_event : ℚ :=
  ⟨2, 4⟩  -- 2 favorable outcomes out of 4 possible outcomes (as a rational number simplified to 1/2)

theorem one_heads_one_tails_probability :
  ∃ (p : ℚ), p = probability_of_favorable_event :=
begin
  use ⟨1, 2⟩,
  sorry
end

end one_heads_one_tails_probability_l182_182875


namespace packs_needed_is_six_l182_182533

variable (l_bedroom l_bathroom l_kitchen l_basement : ℕ)

def total_bulbs_needed := l_bedroom + l_bathroom + l_kitchen + l_basement
def garage_bulbs_needed := total_bulbs_needed / 2
def total_bulbs_with_garage := total_bulbs_needed + garage_bulbs_needed
def packs_needed := total_bulbs_with_garage / 2

theorem packs_needed_is_six
    (h1 : l_bedroom = 2)
    (h2 : l_bathroom = 1)
    (h3 : l_kitchen = 1)
    (h4 : l_basement = 4) :
    packs_needed l_bedroom l_bathroom l_kitchen l_basement = 6 := by
  sorry

end packs_needed_is_six_l182_182533


namespace point_in_third_quadrant_l182_182811

theorem point_in_third_quadrant (m : ℝ) (h1 : m < 0) (h2 : 4 + 2 * m < 0) : m < -2 := by
  sorry

end point_in_third_quadrant_l182_182811


namespace max_n_l182_182205

def sum_first_n_terms (S n : ℕ) (a : ℕ → ℕ) : Prop :=
  S = 2 * a n - n

theorem max_n (S : ℕ) (a : ℕ → ℕ) :
  (∀ n, sum_first_n_terms S n a) → ∀ n, (2 ^ n - 1 ≤ 10 * n) → n ≤ 5 :=
by
  sorry

end max_n_l182_182205


namespace parametric_to_standard_form_l182_182495

theorem parametric_to_standard_form (t : ℝ) (x y : ℝ)
    (param_eq1 : x = 1 + t)
    (param_eq2 : y = -1 + t) :
    x - y - 2 = 0 :=
sorry

end parametric_to_standard_form_l182_182495


namespace point_position_after_time_l182_182289

noncomputable def final_position (initial : ℝ × ℝ) (velocity : ℝ × ℝ) (time : ℝ) : ℝ × ℝ :=
  (initial.1 + velocity.1 * time, initial.2 + velocity.2 * time)

theorem point_position_after_time :
  final_position (-10, 10) (4, -3) 5 = (10, -5) :=
by
  sorry

end point_position_after_time_l182_182289


namespace total_goals_correct_l182_182598

-- Define the number of goals scored by each team in each period
def kickers_first_period_goals : ℕ := 2
def kickers_second_period_goals : ℕ := 2 * kickers_first_period_goals
def spiders_first_period_goals : ℕ := (1 / 2) * kickers_first_period_goals
def spiders_second_period_goals : ℕ := 2 * kickers_second_period_goals

-- Define the total goals scored by both teams
def total_goals : ℕ :=
  kickers_first_period_goals + 
  kickers_second_period_goals + 
  spiders_first_period_goals + 
  spiders_second_period_goals

-- State the theorem to be proved
theorem total_goals_correct : total_goals = 15 := by
  sorry

end total_goals_correct_l182_182598


namespace hcf_lcm_product_l182_182408

theorem hcf_lcm_product (a b : ℕ) (H : a * b = 45276) (L : Nat.lcm a b = 2058) : Nat.gcd a b = 22 :=
by 
  -- The proof steps go here
  sorry

end hcf_lcm_product_l182_182408


namespace product_of_three_numbers_l182_182689

theorem product_of_three_numbers (a b c : ℕ) (h1 : a + b + c = 210) (h2 : 5 * a = b - 11) (h3 : 5 * a = c + 11) : a * b * c = 168504 :=
  sorry

end product_of_three_numbers_l182_182689


namespace find_value_of_sum_of_squares_l182_182520

theorem find_value_of_sum_of_squares
  (a b c : ℝ)
  (h1 : a ≠ 0)
  (h2 : b ≠ 0)
  (h3 : c ≠ 0)
  (h4 : a + b + c = 0)
  (h5 : a^4 + b^4 + c^4 = a^6 + b^6 + c^6) :
  a^2 + b^2 + c^2 = 6 / 5 := by
  sorry

end find_value_of_sum_of_squares_l182_182520


namespace intersection_S_T_eq_T_l182_182952

def S : Set ℤ := { s | ∃ n : ℤ, s = 2 * n + 1 }
def T : Set ℤ := { t | ∃ n : ℤ, t = 4 * n + 1 }

theorem intersection_S_T_eq_T : S ∩ T = T := by
  sorry

end intersection_S_T_eq_T_l182_182952


namespace no_x_satisfies_arithmetic_mean_l182_182097

theorem no_x_satisfies_arithmetic_mean :
  ¬ ∃ x : ℝ, (3 + 117 + 915 + 138 + 2114 + x) / 6 = 12 :=
by
  sorry

end no_x_satisfies_arithmetic_mean_l182_182097


namespace set_equality_l182_182801

open Set

namespace Proof

variables (U M N : Set ℕ) 
variables (U_univ : U = {1, 2, 3, 4, 5, 6})
variables (M_set : M = {2, 3})
variables (N_set : N = {1, 3})

theorem set_equality :
  {4, 5, 6} = (U \ M) ∩ (U \ N) :=
by
  rw [U_univ, M_set, N_set]
  sorry

end Proof

end set_equality_l182_182801


namespace sufficient_not_necessary_condition_l182_182248

-- Definitions of propositions
def propA (x : ℝ) : Prop := (x - 1)^2 < 9
def propB (x a : ℝ) : Prop := (x + 2) * (x + a) < 0

-- Lean statement of the problem
theorem sufficient_not_necessary_condition (a : ℝ) : 
  (∀ x, propA x → propB x a) ∧ (∃ x, ¬ propA x ∧ propB x a) ↔ a < -4 :=
sorry

end sufficient_not_necessary_condition_l182_182248


namespace alma_carrots_leftover_l182_182582

/-- Alma has 47 baby carrots and wishes to distribute them equally among 4 goats.
    We need to prove that the number of leftover carrots after such distribution is 3. -/
theorem alma_carrots_leftover (total_carrots : ℕ) (goats : ℕ) (leftover : ℕ) 
  (h1 : total_carrots = 47) (h2 : goats = 4) (h3 : leftover = total_carrots % goats) : 
  leftover = 3 :=
by
  sorry

end alma_carrots_leftover_l182_182582


namespace number_of_valid_N_count_valid_N_is_seven_l182_182777

theorem number_of_valid_N (N : ℕ) :  ∃ (m : ℕ), (m > 3) ∧ (m ∣ 48) ∧ (N = m - 3) := sorry

theorem count_valid_N_is_seven : 
  (∃ f : Fin 7 → ℕ, ∀ k, ∃ m, (m > 3) ∧ (m ∣ 48) ∧ (f k = m - 3)) ∧
  (∀ g : Fin 8 → ℕ, (∀ k, ∃ m, (m > 3) ∧ (m ∣ 48) ∧ (g k = m - 3)) → False) := sorry

end number_of_valid_N_count_valid_N_is_seven_l182_182777


namespace solve_natural_numbers_system_l182_182252

theorem solve_natural_numbers_system :
  ∃ a b c : ℕ, (a^3 - b^3 - c^3 = 3 * a * b * c) ∧ (a^2 = 2 * (a + b + c)) ∧
  ((a = 4 ∧ b = 1 ∧ c = 3) ∨ (a = 4 ∧ b = 2 ∧ c = 2) ∨ (a = 4 ∧ b = 3 ∧ c = 1)) :=
by
  sorry

end solve_natural_numbers_system_l182_182252


namespace fewest_printers_l182_182727

theorem fewest_printers (x y : ℕ) (h : 8 * x = 7 * y) : x + y = 15 :=
sorry

end fewest_printers_l182_182727


namespace registration_methods_for_5_students_l182_182886

def number_of_registration_methods (students groups : ℕ) : ℕ :=
  groups ^ students

theorem registration_methods_for_5_students : number_of_registration_methods 5 2 = 32 := by
  sorry

end registration_methods_for_5_students_l182_182886


namespace sum_series_eq_four_l182_182754

noncomputable def series_sum : ℝ :=
  ∑' n : ℕ, if n = 0 then 0 else (3 * (n + 1) + 2) / ((n + 1) * ((n + 1) + 1) * ((n + 1) + 3))

theorem sum_series_eq_four :
  series_sum = 4 :=
by
  sorry

end sum_series_eq_four_l182_182754


namespace parker_shorter_than_daisy_l182_182387

noncomputable def solve_height_difference : Nat :=
  let R := 60
  let D := R + 8
  let avg := 64
  ((3 * avg) - (D + R))

theorem parker_shorter_than_daisy :
  let P := solve_height_difference
  D - P = 4 := by
  sorry

end parker_shorter_than_daisy_l182_182387


namespace platform_length_l182_182887

theorem platform_length
  (train_length : ℕ)
  (pole_time : ℕ)
  (platform_time : ℕ)
  (h1 : train_length = 300)
  (h2 : pole_time = 18)
  (h3 : platform_time = 39) :
  let speed := train_length / pole_time in
  let total_distance := speed * platform_time in
  total_distance - train_length = 350 :=
by 
  sorry

end platform_length_l182_182887


namespace magician_card_pairs_l182_182288

theorem magician_card_pairs:
  ∃ (f : Fin 65 → Fin 65 × Fin 65), 
  (∀ m n : Fin 65, ∃ k l : Fin 65, (f m = (k, l) ∧ f n = (l, k))) := 
sorry

end magician_card_pairs_l182_182288


namespace rectangle_area_90_l182_182538

theorem rectangle_area_90 {x y : ℝ} (h1 : (x + 3) * (y - 1) = x * y) (h2 : (x - 3) * (y + 1.5) = x * y) : x * y = 90 := 
  sorry

end rectangle_area_90_l182_182538


namespace art_gallery_ratio_l182_182748

theorem art_gallery_ratio (A : ℕ) (D : ℕ) (S_not_displayed : ℕ) (P_not_displayed : ℕ)
  (h1 : A = 2700)
  (h2 : 1 / 6 * D = D / 6)
  (h3 : P_not_displayed = S_not_displayed / 3)
  (h4 : S_not_displayed = 1200) :
  D / A = 11 / 27 := by
  sorry

end art_gallery_ratio_l182_182748


namespace number_of_solution_pairs_l182_182914

theorem number_of_solution_pairs : 
  ∃ n, (∀ x y : ℕ, 4 * x + 7 * y = 548 → (x > 0 ∧ y > 0) → n = 19) :=
sorry

end number_of_solution_pairs_l182_182914


namespace total_stamps_l182_182100

-- Definitions for the conditions.
def snowflake_stamps : ℕ := 11
def truck_stamps : ℕ := snowflake_stamps + 9
def rose_stamps : ℕ := truck_stamps - 13

-- Statement to prove the total number of stamps.
theorem total_stamps : (snowflake_stamps + truck_stamps + rose_stamps) = 38 :=
by
  sorry

end total_stamps_l182_182100


namespace scientific_notation_gdp_l182_182746

theorem scientific_notation_gdp :
  8837000000 = 8.837 * 10^9 := 
by
  sorry

end scientific_notation_gdp_l182_182746


namespace g_10_plus_g_neg10_eq_6_l182_182834

variable (a b c : ℝ)
noncomputable def g : ℝ → ℝ := λ x => a * x ^ 8 + b * x ^ 6 - c * x ^ 4 + 5

theorem g_10_plus_g_neg10_eq_6 (h : g a b c 10 = 3) : g a b c 10 + g a b c (-10) = 6 :=
by
  -- Proof goes here
  sorry

end g_10_plus_g_neg10_eq_6_l182_182834


namespace charge_two_hours_l182_182717

def charge_first_hour (F A : ℝ) : Prop := F = A + 25
def total_charge_five_hours (F A : ℝ) : Prop := F + 4 * A = 250
def total_charge_two_hours (F A : ℝ) : Prop := F + A = 115

theorem charge_two_hours (F A : ℝ) 
  (h1 : charge_first_hour F A)
  (h2 : total_charge_five_hours F A) : 
  total_charge_two_hours F A :=
by
  sorry

end charge_two_hours_l182_182717


namespace ratio_of_female_contestants_l182_182299

theorem ratio_of_female_contestants (T M F : ℕ) (hT : T = 18) (hM : M = 12) (hF : F = T - M) :
  F / T = 1 / 3 :=
by
  sorry

end ratio_of_female_contestants_l182_182299


namespace last_three_digits_of_7_pow_210_l182_182769

theorem last_three_digits_of_7_pow_210 : (7^210) % 1000 = 599 := by
  sorry

end last_three_digits_of_7_pow_210_l182_182769


namespace cat_food_sufficiency_l182_182317

theorem cat_food_sufficiency (L S : ℝ) (h : L + 4 * S = 14) : L + 3 * S ≥ 11 :=
sorry

end cat_food_sufficiency_l182_182317


namespace smallest_angle_convex_15_polygon_l182_182051

theorem smallest_angle_convex_15_polygon :
  ∃ (a : ℕ) (d : ℕ), (∀ n : ℕ, n ∈ Finset.range 15 → (a + n * d < 180)) ∧
  15 * (a + 7 * d) = 2340 ∧ 15 * d <= 24 -> a = 135 :=
by
  -- Proof omitted
  sorry

end smallest_angle_convex_15_polygon_l182_182051


namespace solve_abs_inequality_l182_182850

theorem solve_abs_inequality (x : ℝ) (h : x ≠ 1) : 
  abs ((3 * x - 2) / (x - 1)) > 3 ↔ (5 / 6 < x ∧ x < 1) ∨ (x > 1) := 
by 
  sorry

end solve_abs_inequality_l182_182850


namespace roots_of_polynomial_l182_182336

theorem roots_of_polynomial : 
  (∀ x : ℝ, (x^3 - 6*x^2 + 11*x - 6) * (x - 2) = 0 ↔ x = 1 ∨ x = 2 ∨ x = 3) :=
by
  intro x
  sorry

end roots_of_polynomial_l182_182336


namespace find_intended_number_l182_182639

theorem find_intended_number (n : ℕ) (h : 6 * n + 382 = 988) : n = 101 := 
by {
  sorry
}

end find_intended_number_l182_182639


namespace cos_90_eq_0_l182_182155

theorem cos_90_eq_0 : Real.cos (90 * Real.pi / 180) = 0 := by
  sorry

end cos_90_eq_0_l182_182155


namespace total_wage_calculation_l182_182738

def basic_pay_rate : ℝ := 20
def weekly_hours : ℝ := 40
def overtime_rate : ℝ := basic_pay_rate * 1.25
def total_hours_worked : ℝ := 48
def overtime_hours : ℝ := total_hours_worked - weekly_hours

theorem total_wage_calculation : 
  (weekly_hours * basic_pay_rate) + (overtime_hours * overtime_rate) = 1000 :=
by
  sorry

end total_wage_calculation_l182_182738


namespace sulfuric_acid_percentage_l182_182647

theorem sulfuric_acid_percentage 
  (total_volume : ℝ)
  (first_solution_percentage : ℝ)
  (final_solution_percentage : ℝ)
  (second_solution_volume : ℝ)
  (expected_second_solution_percentage : ℝ) :
  total_volume = 60 ∧
  first_solution_percentage = 0.02 ∧
  final_solution_percentage = 0.05 ∧
  second_solution_volume = 18 →
  expected_second_solution_percentage = 12 :=
by
  sorry

end sulfuric_acid_percentage_l182_182647


namespace platform_length_is_350_l182_182888

variables (L : ℕ)

def train_length := 300
def time_to_cross_pole := 18
def time_to_cross_platform := 39

-- Speed of the train when crossing the pole
def speed_cross_pole : ℚ := train_length / time_to_cross_pole

-- Speed of the train when crossing the platform
def speed_cross_platform (L : ℕ) : ℚ := (train_length + L) / time_to_cross_platform

-- The main goal is to prove that the length of the platform is 350 meters
theorem platform_length_is_350 (L : ℕ) (h : speed_cross_pole = speed_cross_platform L) : L = 350 := sorry

end platform_length_is_350_l182_182888


namespace find_positive_real_solutions_l182_182607

theorem find_positive_real_solutions (x : ℝ) (h1 : 0 < x) 
(h2 : 3 / 5 * (2 * x ^ 2 - 2) = (x ^ 2 - 40 * x - 8) * (x ^ 2 + 20 * x + 4)) :
    x = (40 + Real.sqrt 1636) / 2 ∨ x = (-20 + Real.sqrt 388) / 2 := by
  sorry

end find_positive_real_solutions_l182_182607


namespace sum_of_two_numbers_l182_182416

theorem sum_of_two_numbers (x y : ℝ) (h1 : x + y = 30) (h2 : x - y = 10) (h3 : x * y = 200) : (x + y = 30) :=
by sorry

end sum_of_two_numbers_l182_182416


namespace cos_pi_half_eq_zero_l182_182174

theorem cos_pi_half_eq_zero : Real.cos (Float.pi / 2) = 0 :=
by
  sorry

end cos_pi_half_eq_zero_l182_182174


namespace factorize_expression_l182_182763

theorem factorize_expression (a b : ℝ) :
  4 * a^3 * b - a * b = a * b * (2 * a + 1) * (2 * a - 1) :=
by
  sorry

end factorize_expression_l182_182763


namespace intersection_S_T_eq_T_l182_182953

def S : Set ℤ := { s | ∃ n : ℤ, s = 2 * n + 1 }
def T : Set ℤ := { t | ∃ n : ℤ, t = 4 * n + 1 }

theorem intersection_S_T_eq_T : S ∩ T = T := by
  sorry

end intersection_S_T_eq_T_l182_182953


namespace fencing_cost_proof_l182_182056

noncomputable def totalCostOfFencing (length : ℕ) (breadth : ℕ) (costPerMeter : ℚ) : ℚ :=
  2 * (length + breadth) * costPerMeter

theorem fencing_cost_proof : totalCostOfFencing 56 (56 - 12) 26.50 = 5300 := by
  sorry

end fencing_cost_proof_l182_182056


namespace smallest_n_for_2007_l182_182264

/-- The smallest number of positive integers \( n \) such that their product is 2007 and their sum is 2007.
Given that \( n > 1 \), we need to show 1337 is the smallest such \( n \).
-/
theorem smallest_n_for_2007 (n : ℕ) (H : n > 1) :
  (∃ s : Finset ℕ, (s.sum id = 2007) ∧ (s.prod id = 2007) ∧ (s.card = n)) → (n = 1337) :=
sorry

end smallest_n_for_2007_l182_182264


namespace cape_may_vs_daytona_shark_sightings_diff_l182_182108

-- Definitions based on the conditions
def total_shark_sightings := 40
def cape_may_sightings : ℕ := 24
def daytona_beach_sightings : ℕ := total_shark_sightings - cape_may_sightings

-- The main theorem stating the problem in Lean
theorem cape_may_vs_daytona_shark_sightings_diff :
  (2 * daytona_beach_sightings - cape_may_sightings) = 8 := by
  sorry

end cape_may_vs_daytona_shark_sightings_diff_l182_182108


namespace remaining_stickers_l182_182423

def stickers_per_page : ℕ := 20
def pages : ℕ := 12
def lost_pages : ℕ := 1

theorem remaining_stickers : 
  (pages * stickers_per_page - lost_pages * stickers_per_page) = 220 :=
  by
    sorry

end remaining_stickers_l182_182423


namespace four_digit_number_count_l182_182628

def count_suitable_four_digit_numbers : Prop :=
  let validFirstDigits := [4, 5, 6, 7, 8, 9] -- First digit choices (4 to 9) = 6 choices
  let validLastDigits := [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] -- Last digit choices (0 to 9) = 10 choices
  let validMiddlePairs := (do
    d1 ← [1, 2, 3, 4, 5, 6, 7, 8, 9], -- Middle digit choices (1 to 9)
    d2 ← [1, 2, 3, 4, 5, 6, 7, 8, 9], -- Middle digit choices (1 to 9)
    guard (d1 * d2 > 10), 
    [d1, d2]).length -- Count the valid pairs whose product exceeds 10
  
  3660 = validFirstDigits.length * validMiddlePairs * validLastDigits.length

theorem four_digit_number_count : count_suitable_four_digit_numbers :=
by
  -- Hint: skipping actual proof
  sorry

end four_digit_number_count_l182_182628


namespace cos_of_90_degrees_l182_182168

-- Definition of cosine of 90 degrees
def cos_90_degrees : ℝ := Real.cos (90 * (Float.pi / 180))

-- Condition: Cosine is the x-coordinate of the point (0, 1) after 90 degree rotation
theorem cos_of_90_degrees : cos_90_degrees = 0 :=
by
  -- Proof would go here, but we skip it with sorry
  sorry

end cos_of_90_degrees_l182_182168


namespace S_inter_T_eq_T_l182_182969

-- Define the sets S and T
def S := { s : ℤ | ∃ n : ℤ, s = 2 * n + 1 }
def T := { t : ℤ | ∃ n : ℤ, t = 4 * n + 1 }

-- Prove that S ∩ T = T
theorem S_inter_T_eq_T : S ∩ T = T := 
by
  sorry

end S_inter_T_eq_T_l182_182969


namespace cos_ninety_degrees_l182_182122

theorem cos_ninety_degrees : Real.cos (90 * Real.pi / 180) = 0 := 
by 
  sorry

end cos_ninety_degrees_l182_182122


namespace find_integer_n_l182_182431

theorem find_integer_n :
  ∃ n : ℕ, 0 ≤ n ∧ n < 201 ∧ 200 * n ≡ 144 [MOD 101] ∧ n = 29 := 
by
  sorry

end find_integer_n_l182_182431


namespace license_plate_count_l182_182363

theorem license_plate_count : 
  let consonants := 20
  let vowels := 6
  let digits := 10
  4 * consonants * vowels * consonants * digits = 24000 :=
by
  sorry

end license_plate_count_l182_182363


namespace high_quality_chip_prob_l182_182716

variable (chipsA chipsB chipsC : ℕ)
variable (qualityA qualityB qualityC : ℝ)
variable (totalChips : ℕ)

noncomputable def probability_of_high_quality_chip (chipsA chipsB chipsC : ℕ) (qualityA qualityB qualityC : ℝ) (totalChips : ℕ) : ℝ :=
  (chipsA / totalChips) * qualityA + (chipsB / totalChips) * qualityB + (chipsC / totalChips) * qualityC

theorem high_quality_chip_prob :
  let chipsA := 5
  let chipsB := 10
  let chipsC := 10
  let qualityA := 0.8
  let qualityB := 0.8
  let qualityC := 0.7
  let totalChips := 25
  probability_of_high_quality_chip chipsA chipsB chipsC qualityA qualityB qualityC totalChips = 0.76 :=
by
  sorry

end high_quality_chip_prob_l182_182716


namespace range_of_a_l182_182799

noncomputable def f (a b x : ℝ) : ℝ := Real.log x - (1/2) * a * x^2 - b * x

theorem range_of_a (a b x : ℝ) (h1 : ∀ x > 0, (1/x) - a * x - b ≠ 0) (h2 : ∀ x > 0, x = 1 → (1/x) - a * x - b = 0) : 
  (1 - a) = b ∧ a > -1 :=
by
  sorry

end range_of_a_l182_182799


namespace value_of_expression_at_three_l182_182871

theorem value_of_expression_at_three (x : ℝ) (h : x = 3) : (x^2 - 3 * x - 10) / (x - 5) = 5 := 
by
  sorry

end value_of_expression_at_three_l182_182871


namespace monotonic_intervals_maximum_of_k_l182_182620

def f (x : ℝ) (m : ℝ) : ℝ := (m + Real.log x) / x

def h (x : ℝ) : ℝ := (x + 1) * (4 + Real.log x) / x

theorem monotonic_intervals (m : ℝ) (h1 : x > 1) :
  (m ≥ 1 ∧ ∀ x > 1, deriv (λ x, f x m) x ≤ 0) ∨ 
  (m < 1 ∧ ∀ x ∈ Ioo 1 (Real.exp (1 - m)), deriv (λ x, f x m) x > 0 ∧ ∀ x ∈ Ioi (Real.exp (1 - m)), deriv (λ x, f x m) x < 0) :=
sorry

theorem maximum_of_k (h1 : ∀ x > 1, (k / (x + 1) < f x 4)) :
  k ≤ 6 :=
sorry

end monotonic_intervals_maximum_of_k_l182_182620


namespace cos_of_90_degrees_l182_182165

-- Definition of cosine of 90 degrees
def cos_90_degrees : ℝ := Real.cos (90 * (Float.pi / 180))

-- Condition: Cosine is the x-coordinate of the point (0, 1) after 90 degree rotation
theorem cos_of_90_degrees : cos_90_degrees = 0 :=
by
  -- Proof would go here, but we skip it with sorry
  sorry

end cos_of_90_degrees_l182_182165


namespace cos_90_deg_eq_zero_l182_182137

noncomputable def cos_90_degrees : ℝ :=
cos (90 * real.pi / 180)

theorem cos_90_deg_eq_zero : cos_90_degrees = 0 := 
by
  -- Definitions related to the conditions
  let point_0 := (1 : ℝ, 0 : ℝ)
  let point_90 := (0 : ℝ, 1 : ℝ)

  -- Rotating the point (1, 0) by 90 degrees counterclockwise results in the point (0, 1)
  have h_rotate_90 : (real.cos (90 * real.pi / 180), real.sin (90 * real.pi / 180)) = point_90 :=
    by
      -- Rotation logic is abstracted out in this property of cos and sin
      have h_cos_90 : real.cos (real.pi / 2) = 0 := by sorry
      have h_sin_90 : real.sin (real.pi / 2) = 1 := by sorry
      exact ⟨h_cos_90, h_sin_90⟩

  -- By definition, cos 90 degrees is the x-coordinate of the rotated point
  show cos_90_degrees = 0,
  from congr_arg Prod.fst h_rotate_90

end cos_90_deg_eq_zero_l182_182137


namespace CatCafePawRatio_l182_182906

-- Define the context
def CatCafeMeow (P : ℕ) := 3 * P
def CatCafePaw (P : ℕ) := P
def CatCafeCool := 5
def TotalCats (P : ℕ) := CatCafeMeow P + CatCafePaw P

-- State the theorem
theorem CatCafePawRatio (P : ℕ) (n : ℕ) : 
  CatCafeCool = 5 →
  CatCafeMeow P = 3 * CatCafePaw P →
  TotalCats P = 40 →
  P = 10 →
  n * CatCafeCool = P →
  n = 2 :=
by
  intros
  sorry

end CatCafePawRatio_l182_182906


namespace geometric_sequence_S8_l182_182797

theorem geometric_sequence_S8 (S : ℕ → ℝ) (hs2 : S 2 = 4) (hs4 : S 4 = 16) : 
  S 8 = 160 := by
  sorry

end geometric_sequence_S8_l182_182797


namespace area_of_enclosed_shape_l182_182679

noncomputable def enclosed_area : ℝ :=
  ∫ x in (0 : ℝ)..(2 : ℝ), (4 * x - x^3)

theorem area_of_enclosed_shape : enclosed_area = 4 := by
  sorry

end area_of_enclosed_shape_l182_182679


namespace jonathan_weekly_caloric_deficit_l182_182017

def jonathan_caloric_deficit 
  (daily_calories : ℕ) (extra_calories_saturday : ℕ) (daily_burn : ℕ) 
  (days : ℕ) (saturday : ℕ) : ℕ :=
  let total_consumed := daily_calories * days + (daily_calories + extra_calories_saturday) * saturday in
  let total_burned := daily_burn * (days + saturday) in
  total_burned - total_consumed

theorem jonathan_weekly_caloric_deficit :
  jonathan_caloric_deficit 2500 1000 3000 6 1 = 2500 :=
by
  sorry

end jonathan_weekly_caloric_deficit_l182_182017


namespace price_of_first_oil_is_54_l182_182713

/-- Let x be the price per litre of the first oil.
Given that 10 litres of the first oil are mixed with 5 litres of second oil priced at Rs. 66 per litre,
resulting in a 15-litre mixture costing Rs. 58 per litre, prove that x = 54. -/
theorem price_of_first_oil_is_54 :
  (∃ x : ℝ, x = 54) ↔
  (10 * x + 5 * 66 = 15 * 58) :=
by
  sorry

end price_of_first_oil_is_54_l182_182713


namespace total_goals_scored_l182_182600

-- Definitions based on the problem conditions
def kickers_first_period_goals : ℕ := 2
def kickers_second_period_goals : ℕ := 2 * kickers_first_period_goals
def spiders_first_period_goals : ℕ := kickers_first_period_goals / 2
def spiders_second_period_goals : ℕ := 2 * kickers_second_period_goals

-- The theorem we need to prove
theorem total_goals_scored : 
  kickers_first_period_goals + kickers_second_period_goals +
  spiders_first_period_goals + spiders_second_period_goals = 15 := 
by
  -- proof steps will go here
  sorry

end total_goals_scored_l182_182600


namespace compare_fractions_l182_182117

theorem compare_fractions : (-1 / 3) > (-1 / 2) := sorry

end compare_fractions_l182_182117


namespace four_digit_number_count_l182_182629

def count_suitable_four_digit_numbers : Prop :=
  let validFirstDigits := [4, 5, 6, 7, 8, 9] -- First digit choices (4 to 9) = 6 choices
  let validLastDigits := [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] -- Last digit choices (0 to 9) = 10 choices
  let validMiddlePairs := (do
    d1 ← [1, 2, 3, 4, 5, 6, 7, 8, 9], -- Middle digit choices (1 to 9)
    d2 ← [1, 2, 3, 4, 5, 6, 7, 8, 9], -- Middle digit choices (1 to 9)
    guard (d1 * d2 > 10), 
    [d1, d2]).length -- Count the valid pairs whose product exceeds 10
  
  3660 = validFirstDigits.length * validMiddlePairs * validLastDigits.length

theorem four_digit_number_count : count_suitable_four_digit_numbers :=
by
  -- Hint: skipping actual proof
  sorry

end four_digit_number_count_l182_182629


namespace exists_consecutive_integers_not_sum_of_two_squares_l182_182762

open Nat

theorem exists_consecutive_integers_not_sum_of_two_squares : 
  ∃ (m : ℕ), ∀ k : ℕ, k < 2017 → ¬(∃ a b : ℤ, (m + k) = a^2 + b^2) := 
sorry

end exists_consecutive_integers_not_sum_of_two_squares_l182_182762


namespace shorter_piece_length_l182_182440

theorem shorter_piece_length (x : ℝ) (h : 3 * x = 60) : x = 20 :=
by
  sorry

end shorter_piece_length_l182_182440


namespace number_of_valid_N_count_valid_N_is_seven_l182_182776

theorem number_of_valid_N (N : ℕ) :  ∃ (m : ℕ), (m > 3) ∧ (m ∣ 48) ∧ (N = m - 3) := sorry

theorem count_valid_N_is_seven : 
  (∃ f : Fin 7 → ℕ, ∀ k, ∃ m, (m > 3) ∧ (m ∣ 48) ∧ (f k = m - 3)) ∧
  (∀ g : Fin 8 → ℕ, (∀ k, ∃ m, (m > 3) ∧ (m ∣ 48) ∧ (g k = m - 3)) → False) := sorry

end number_of_valid_N_count_valid_N_is_seven_l182_182776


namespace gnomes_remaining_in_ravenswood_l182_182398

theorem gnomes_remaining_in_ravenswood 
  (westerville_gnomes : ℕ)
  (ravenswood_initial_gnomes : ℕ)
  (taken_gnomes : ℕ)
  (remaining_gnomes : ℕ)
  (h1 : westerville_gnomes = 20)
  (h2 : ravenswood_initial_gnomes = 4 * westerville_gnomes)
  (h3 : taken_gnomes = (40 * ravenswood_initial_gnomes) / 100)
  (h4 : remaining_gnomes = ravenswood_initial_gnomes - taken_gnomes) :
  remaining_gnomes = 48 :=
by
  sorry

end gnomes_remaining_in_ravenswood_l182_182398


namespace abc_mod_n_l182_182521

theorem abc_mod_n (n : ℕ) (a b c : ℤ) (hn : 0 < n)
  (h1 : a * b ≡ 1 [ZMOD n])
  (h2 : c ≡ b [ZMOD n]) : (a * b * c) ≡ 1 [ZMOD n] := sorry

end abc_mod_n_l182_182521


namespace line_intersects_hyperbola_l182_182222

theorem line_intersects_hyperbola (k : Real) : 
  (∃ x y : Real, y = k * x ∧ (x^2) / 9 - (y^2) / 4 = 1) ↔ (-2 / 3 < k ∧ k < 2 / 3) := 
sorry

end line_intersects_hyperbola_l182_182222


namespace total_fish_l182_182881

theorem total_fish {lilly_fish rosy_fish : ℕ} (h1 : lilly_fish = 10) (h2 : rosy_fish = 11) : 
lilly_fish + rosy_fish = 21 :=
by 
  sorry

end total_fish_l182_182881


namespace cos_ninety_degrees_l182_182123

theorem cos_ninety_degrees : Real.cos (90 * Real.pi / 180) = 0 := 
by 
  sorry

end cos_ninety_degrees_l182_182123


namespace lego_set_cost_l182_182786

-- Definitions and conditions
def price_per_car := 5
def cars_sold := 3
def action_figures_sold := 2
def total_earnings := 120

-- Derived prices
def price_per_action_figure := 2 * price_per_car
def price_per_board_game := price_per_action_figure + price_per_car

-- Total cost of sold items (cars, action figures, and board game)
def total_cost_of_sold_items := 
  (cars_sold * price_per_car) + 
  (action_figures_sold * price_per_action_figure) + 
  price_per_board_game

-- Cost of Lego set
theorem lego_set_cost : 
  total_earnings - total_cost_of_sold_items = 70 :=
by
  -- Proof omitted
  sorry

end lego_set_cost_l182_182786


namespace cat_food_sufficiency_l182_182324

variable (L S : ℝ)

theorem cat_food_sufficiency (h : L + 4 * S = 14) : L + 3 * S ≥ 11 :=
sorry

end cat_food_sufficiency_l182_182324


namespace second_candidate_more_marks_30_l182_182442

noncomputable def total_marks : ℝ := 600
def passing_marks_approx : ℝ := 240

def candidate_marks (percentage : ℝ) (total : ℝ) : ℝ :=
  percentage * total

def more_marks (second_candidate : ℝ) (passing : ℝ) : ℝ :=
  second_candidate - passing

theorem second_candidate_more_marks_30 :
  more_marks (candidate_marks 0.45 total_marks) passing_marks_approx = 30 := by
  sorry

end second_candidate_more_marks_30_l182_182442


namespace non_neg_sequence_l182_182235

theorem non_neg_sequence (a : ℝ) (x : ℕ → ℝ) (h0 : x 0 = 0)
  (h1 : ∀ n, x (n + 1) = 1 - a * Real.exp (x n)) (ha : a ≤ 1) :
  ∀ n, x n ≥ 0 := 
  sorry

end non_neg_sequence_l182_182235


namespace smallest_number_of_white_marbles_l182_182581

theorem smallest_number_of_white_marbles
  (n : ℕ)
  (hn1 : n > 0)
  (orange_marbles : ℕ := n / 5)
  (hn_orange : n % 5 = 0)
  (purple_marbles : ℕ := n / 6)
  (hn_purple : n % 6 = 0)
  (green_marbles : ℕ := 9)
  : (n - (orange_marbles + purple_marbles + green_marbles)) = 10 → n = 30 :=
by
  sorry

end smallest_number_of_white_marbles_l182_182581


namespace average_speed_l182_182861

theorem average_speed (speed1 speed2: ℝ) (time1 time2: ℝ) (h1: speed1 = 90) (h2: speed2 = 40) (h3: time1 = 1) (h4: time2 = 1) :
  (speed1 * time1 + speed2 * time2) / (time1 + time2) = 65 := by
  sorry

end average_speed_l182_182861


namespace intersection_of_sets_l182_182981

open Set Int

theorem intersection_of_sets (S T : Set Int) (hS : S = {s | ∃ n : ℤ, s = 2 * n + 1}) (hT : T = {t | ∃ n : ℤ, t = 4 * n + 1}) : S ∩ T = T :=
  by
  sorry

end intersection_of_sets_l182_182981


namespace expression_value_l182_182660

-- Define the problem statement
theorem expression_value (x y z : ℝ) (h1 : x ≠ 0) (h2 : y ≠ 0) (h3 : z ≠ 0) 
  (h4 : (x + y) / z = (y + z) / x) (h5 : (y + z) / x = (z + x) / y) :
  ∃ k : ℝ, k = 8 ∨ k = -1 := 
sorry

end expression_value_l182_182660


namespace total_people_in_office_even_l182_182671

theorem total_people_in_office_even (M W : ℕ) (h_even : M = W) (h_meeting_women : 6 = 20 / 100 * W) : 
  M + W = 60 :=
by
  sorry

end total_people_in_office_even_l182_182671


namespace not_square_of_expression_l182_182668

theorem not_square_of_expression (n : ℕ) (h : n > 0) : ¬ ∃ m : ℤ, m * m = 2 * n * n + 2 - n :=
by
  sorry

end not_square_of_expression_l182_182668


namespace jim_caught_fish_l182_182465

variable (ben judy billy susie jim caught_back total_filets : ℕ)

def caught_fish : ℕ :=
  ben + judy + billy + susie + jim - caught_back

theorem jim_caught_fish (h_ben : ben = 4)
                        (h_judy : judy = 1)
                        (h_billy : billy = 3)
                        (h_susie : susie = 5)
                        (h_caught_back : caught_back = 3)
                        (h_total_filets : total_filets = 24)
                        (h_filets_per_fish : ∀ f : ℕ, total_filets = f * 2 → caught_fish ben judy billy susie jim caught_back = f) :
  jim = 2 :=
by
  -- Proof goes here
  sorry

end jim_caught_fish_l182_182465


namespace sample_size_stratified_sampling_l182_182803

theorem sample_size_stratified_sampling (n : ℕ) 
  (total_employees : ℕ) 
  (middle_aged_employees : ℕ) 
  (middle_aged_sample : ℕ)
  (stratified_sampling : n * middle_aged_employees = middle_aged_sample * total_employees)
  (total_employees_pos : total_employees = 750)
  (middle_aged_employees_pos : middle_aged_employees = 250) :
  n = 15 := 
by
  rw [total_employees_pos, middle_aged_employees_pos] at stratified_sampling
  sorry

end sample_size_stratified_sampling_l182_182803


namespace general_term_sequence_l182_182712

theorem general_term_sequence (a : ℕ → ℝ) (h₁ : a 1 = 1) (hn : ∀ (n : ℕ), a (n + 1) = (10 + 4 * a n) / (1 + a n)) :
  ∀ n : ℕ, a n = 5 - 7 / (1 + (3 / 4) * (-6)^(n - 1)) := 
sorry

end general_term_sequence_l182_182712


namespace sufficient_food_l182_182327

-- Definitions for a large package and a small package as amounts of food lasting days
def largePackageDay : ℕ := L
def smallPackageDay : ℕ := S

-- Condition: One large package and four small packages last 14 days
axiom condition : largePackageDay + 4 * smallPackageDay = 14

-- Theorem to be proved: One large package and three small packages last at least 11 days
theorem sufficient_food : largePackageDay + 3 * smallPackageDay ≥ 11 :=
by
  sorry

end sufficient_food_l182_182327


namespace num_valid_four_digit_numbers_l182_182634

def is_valid_number (n : ℕ) : Prop :=
  n >= 4000 ∧ n < 10000 ∧ let d1 := n / 1000,
                             d2 := (n / 100) % 10,
                             d3 := (n / 10) % 10,
                             _ := n % 10 in
                            d1 >= 4 ∧ (d2 > 0 ∧ d2 < 10) ∧ (d3 > 0 ∧ d3 < 10) ∧ (d2 * d3 > 10)

theorem num_valid_four_digit_numbers : 
  (finset.filter is_valid_number (finset.range 10000)).card = 4260 :=
sorry

end num_valid_four_digit_numbers_l182_182634


namespace puppy_weight_l182_182092

variable (p s l r : ℝ)

theorem puppy_weight :
  p + s + l + r = 40 ∧ 
  p^2 + l^2 = 4 * s ∧ 
  p^2 + s^2 = l^2 → 
  p = Real.sqrt 2 :=
sorry

end puppy_weight_l182_182092


namespace chelsea_total_time_l182_182469

def num_batches := 4
def bake_time_per_batch := 20  -- minutes
def ice_time_per_batch := 30   -- minutes
def cupcakes_per_batch := 6
def additional_time_first_batch := 10 -- per cupcake
def additional_time_second_batch := 15 -- per cupcake
def additional_time_third_batch := 12 -- per cupcake
def additional_time_fourth_batch := 20 -- per cupcake

def total_bake_ice_time := bake_time_per_batch + ice_time_per_batch
def total_bake_ice_time_all_batches := total_bake_ice_time * num_batches

def total_additional_time_first_batch := additional_time_first_batch * cupcakes_per_batch
def total_additional_time_second_batch := additional_time_second_batch * cupcakes_per_batch
def total_additional_time_third_batch := additional_time_third_batch * cupcakes_per_batch
def total_additional_time_fourth_batch := additional_time_fourth_batch * cupcakes_per_batch

def total_additional_time := 
  total_additional_time_first_batch +
  total_additional_time_second_batch +
  total_additional_time_third_batch +
  total_additional_time_fourth_batch

def total_time := total_bake_ice_time_all_batches + total_additional_time

theorem chelsea_total_time : total_time = 542 := by
  sorry

end chelsea_total_time_l182_182469


namespace proof_expression_C_equals_negative_one_l182_182296

def A : ℤ := abs (-1)
def B : ℤ := -(-1)
def C : ℤ := -(1^2)
def D : ℤ := (-1)^2

theorem proof_expression_C_equals_negative_one : C = -1 :=
by 
  sorry

end proof_expression_C_equals_negative_one_l182_182296


namespace number_line_distance_l182_182527

theorem number_line_distance (x : ℝ) : |x + 1| = 6 ↔ (x = 5 ∨ x = -7) :=
by
  sorry

end number_line_distance_l182_182527


namespace pq_condition_l182_182029

theorem pq_condition (p q : ℝ) (h1 : p * q = 16) (h2 : p + q = 10) : (p - q)^2 = 36 :=
by
  sorry

end pq_condition_l182_182029


namespace cos_90_eq_0_l182_182142

theorem cos_90_eq_0 :
  ∃ (p : ℝ × ℝ), p = (0, 1) ∧ ∀ θ : ℝ, θ = 90 → cos θ = p.1 :=
by
  let p := (0, 1)
  use p
  split
  · rfl
  · intros θ h
    rw h
    sorry

end cos_90_eq_0_l182_182142


namespace alice_acorns_purchase_l182_182898

variable (bob_payment : ℕ) (alice_payment_rate : ℕ) (price_per_acorn : ℕ)

-- Given conditions
def bob_paid : Prop := bob_payment = 6000
def alice_paid : Prop := alice_payment_rate = 9
def acorn_price : Prop := price_per_acorn = 15

-- Proof statement
theorem alice_acorns_purchase
  (h1 : bob_paid bob_payment)
  (h2 : alice_paid alice_payment_rate)
  (h3 : acorn_price price_per_acorn) :
  ∃ n : ℕ, n = (alice_payment_rate * bob_payment) / price_per_acorn ∧ n = 3600 := 
by
  sorry

end alice_acorns_purchase_l182_182898


namespace hike_up_days_l182_182455

theorem hike_up_days (R_up R_down D_down D_up : ℝ) 
  (H1 : R_up = 8) 
  (H2 : R_down = 1.5 * R_up)
  (H3 : D_down = 24)
  (H4 : D_up / R_up = D_down / R_down) : 
  D_up / R_up = 2 :=
by
  sorry

end hike_up_days_l182_182455


namespace distance_lines_eq_2_l182_182208

-- Define the first line in standard form
def line1 (x y : ℝ) : Prop := 3 * x + 4 * y + 3 = 0

-- Define the second line in standard form, established based on the parallel condition
def line2 (x y : ℝ) : Prop := 6 * x + 8 * y - 14 = 0

-- Define the condition for parallel lines which gives m
axiom parallel_lines_condition : ∀ (x y : ℝ), (line1 x y) → (line2 x y)

-- Define the distance between two parallel lines formula
noncomputable def distance_between_parallel_lines (a b c1 c2 : ℝ) : ℝ :=
  abs (c2 - c1) / (Real.sqrt (a ^ 2 + b ^ 2))

-- Prove the distance between the given lines is 2
theorem distance_lines_eq_2 : distance_between_parallel_lines 3 4 (-3) 7 = 2 :=
by
  -- Details of proof are omitted, but would show how to manipulate and calculate distances
  sorry

end distance_lines_eq_2_l182_182208


namespace intersection_S_T_eq_T_l182_182954

def S : Set ℤ := { s | ∃ n : ℤ, s = 2 * n + 1 }
def T : Set ℤ := { t | ∃ n : ℤ, t = 4 * n + 1 }

theorem intersection_S_T_eq_T : S ∩ T = T := by
  sorry

end intersection_S_T_eq_T_l182_182954


namespace compare_fractions_l182_182119

theorem compare_fractions : (-1 / 3) > (-1 / 2) := sorry

end compare_fractions_l182_182119


namespace evaluate_g_neg_1_l182_182206

noncomputable def g (x : ℝ) : ℝ := -2 * x^2 + 5 * x - 7

theorem evaluate_g_neg_1 : g (-1) = -14 := 
by
  sorry

end evaluate_g_neg_1_l182_182206


namespace janets_garden_area_l182_182649

theorem janets_garden_area :
  ∃ (s l : ℕ), 2 * (s + l) = 24 ∧ (l + 1) = 3 * (s + 1) ∧ 6 * (s + 1 - 1) * 6 * (l + 1 - 1) = 576 := 
by
  sorry

end janets_garden_area_l182_182649


namespace intersection_S_T_eq_T_l182_182948

def S : Set ℤ := { s | ∃ n : ℤ, s = 2 * n + 1 }
def T : Set ℤ := { t | ∃ n : ℤ, t = 4 * n + 1 }

theorem intersection_S_T_eq_T : S ∩ T = T := by
  sorry

end intersection_S_T_eq_T_l182_182948


namespace circular_garden_radius_l182_182877

theorem circular_garden_radius
  (r : ℝ) -- radius of the circular garden
  (h : 2 * Real.pi * r = (1 / 6) * Real.pi * r^2) :
  r = 12 := 
by {
  sorry
}

end circular_garden_radius_l182_182877


namespace sequence_98th_term_l182_182254

-- Definitions of the rules
def rule1 (n : ℕ) : ℕ := n * 9
def rule2 (n : ℕ) : ℕ := n / 2
def rule3 (n : ℕ) : ℕ := n - 5

-- Function to compute the next term in the sequence based on the current term
def next_term (n : ℕ) : ℕ :=
  if n < 10 then rule1 n
  else if n % 2 = 0 then rule2 n
  else rule3 n

-- Function to compute the nth term of the sequence starting with the initial term
def nth_term (start : ℕ) (n : ℕ) : ℕ :=
  Nat.iterate next_term n start

-- Theorem to prove that the 98th term of the sequence starting at 98 is 27
theorem sequence_98th_term : nth_term 98 98 = 27 := by
  sorry

end sequence_98th_term_l182_182254


namespace probability_of_being_closer_to_origin_l182_182894

noncomputable def probability_closer_to_origin 
  (rect : Set (ℝ × ℝ) := {p | 0 ≤ p.1 ∧ p.1 ≤ 3 ∧ 0 ≤ p.2 ∧ p.2 ≤ 2})
  (origin : ℝ × ℝ := (0, 0))
  (point : ℝ × ℝ := (4, 2))
  : ℚ :=
1/3

theorem probability_of_being_closer_to_origin :
  probability_closer_to_origin = 1/3 :=
by sorry

end probability_of_being_closer_to_origin_l182_182894


namespace friend_gain_percentage_l182_182088

noncomputable def gain_percentage (original_cost_price sold_price_friend : ℝ) : ℝ :=
  ((sold_price_friend - (original_cost_price - 0.12 * original_cost_price)) / (original_cost_price - 0.12 * original_cost_price)) * 100

theorem friend_gain_percentage (original_cost_price sold_price_friend gain_pct : ℝ) 
  (H1 : original_cost_price = 51136.36) 
  (H2 : sold_price_friend = 54000) 
  (H3 : gain_pct = 20) : 
  gain_percentage original_cost_price sold_price_friend = gain_pct := 
by
  sorry

end friend_gain_percentage_l182_182088


namespace gnomes_remaining_in_ravenswood_l182_182399

theorem gnomes_remaining_in_ravenswood 
  (westerville_gnomes : ℕ)
  (ravenswood_initial_gnomes : ℕ)
  (taken_gnomes : ℕ)
  (remaining_gnomes : ℕ)
  (h1 : westerville_gnomes = 20)
  (h2 : ravenswood_initial_gnomes = 4 * westerville_gnomes)
  (h3 : taken_gnomes = (40 * ravenswood_initial_gnomes) / 100)
  (h4 : remaining_gnomes = ravenswood_initial_gnomes - taken_gnomes) :
  remaining_gnomes = 48 :=
by
  sorry

end gnomes_remaining_in_ravenswood_l182_182399


namespace plane_equation_parametric_l182_182090

theorem plane_equation_parametric 
  (s t : ℝ)
  (v : ℝ × ℝ × ℝ)
  (x y z : ℝ) 
  (A B C D : ℤ)
  (h1 : v = (2 + s + 2 * t, 3 + 2 * s - t, 1 + s + 3 * t))
  (h2 : A = 7)
  (h3 : B = -1)
  (h4 : C = -5)
  (h5 : D = -6)
  (h6 : A > 0)
  (h7 : Int.gcd A (Int.gcd B (Int.gcd C D)) = 1) :
  7 * x - y - 5 * z - 6 = 0 := 
sorry

end plane_equation_parametric_l182_182090


namespace today_is_thursday_l182_182460

-- Define the days of the week as an enumerated type
inductive DayOfWeek
| Monday : DayOfWeek
| Tuesday : DayOfWeek
| Wednesday : DayOfWeek
| Thursday : DayOfWeek
| Friday : DayOfWeek
| Saturday : DayOfWeek
| Sunday : DayOfWeek

open DayOfWeek

-- Define the conditions for the lion and the unicorn
def lion_truth (d: DayOfWeek) : Bool :=
match d with
| Monday | Tuesday | Wednesday => false
| _ => true

def unicorn_truth (d: DayOfWeek) : Bool :=
match d with
| Thursday | Friday | Saturday => false
| _ => true

-- The statement made by the lion and the unicorn
def lion_statement (today: DayOfWeek) : Bool :=
match today with
| Monday => lion_truth Sunday
| Tuesday => lion_truth Monday
| Wednesday => lion_truth Tuesday
| Thursday => lion_truth Wednesday
| Friday => lion_truth Thursday
| Saturday => lion_truth Friday
| Sunday => lion_truth Saturday

def unicorn_statement (today: DayOfWeek) : Bool :=
match today with
| Monday => unicorn_truth Sunday
| Tuesday => unicorn_truth Monday
| Wednesday => unicorn_truth Tuesday
| Thursday => unicorn_truth Wednesday
| Friday => unicorn_truth Thursday
| Saturday => unicorn_truth Friday
| Sunday => unicorn_truth Saturday

-- Main theorem to prove the current day
theorem today_is_thursday (d: DayOfWeek) (lion_said: lion_statement d = false) (unicorn_said: unicorn_statement d = false) : d = Thursday :=
by
  -- Placeholder for actual proof
  sorry

end today_is_thursday_l182_182460


namespace kyle_and_miles_marbles_l182_182474

theorem kyle_and_miles_marbles (f k m : ℕ) 
  (h1 : f = 3 * k) 
  (h2 : f = 5 * m) 
  (h3 : f = 15) : 
  k + m = 8 := 
by 
  sorry

end kyle_and_miles_marbles_l182_182474


namespace sqrt_factorial_div_l182_182302

theorem sqrt_factorial_div:
  Real.sqrt (↑(Nat.factorial 9) / 90) = 4 * Real.sqrt 42 := 
by
  -- Steps of the proof
  sorry

end sqrt_factorial_div_l182_182302


namespace trigonometric_identity_l182_182793

open Real

theorem trigonometric_identity
  (x : ℝ)
  (h1 : sin x * cos x = 1 / 8)
  (h2 : π / 4 < x)
  (h3 : x < π / 2) :
  cos x - sin x = - (sqrt 3 / 2) :=
sorry

end trigonometric_identity_l182_182793


namespace sum_of_coordinates_eq_nine_halves_l182_182537

theorem sum_of_coordinates_eq_nine_halves {f : ℝ → ℝ} 
  (h₁ : 2 = (f 1) / 2) :
  (4 + (1 / 2) = 9 / 2) :=
by 
  sorry

end sum_of_coordinates_eq_nine_halves_l182_182537


namespace geometric_progression_solution_l182_182343

theorem geometric_progression_solution (p : ℝ) :
  (3 * p + 1)^2 = (9 * p + 10) * |p - 3| ↔ p = -1 ∨ p = 29 / 18 :=
by
  sorry

end geometric_progression_solution_l182_182343


namespace cos_90_eq_0_l182_182132

theorem cos_90_eq_0 : Real.cos (Float.pi / 2) = 0 := 
sorry

end cos_90_eq_0_l182_182132


namespace g_infinite_distinct_values_l182_182912

def g (x : ℝ) : ℝ :=
  (∑ k in finset.range 11, (⌊k * x⌋ - (k + 1) * ⌊x⌋)) + 3 * x

theorem g_infinite_distinct_values : ∀ x : ℝ, x ≥ 0 → set.infinite (set.range g) := 
by sorry

end g_infinite_distinct_values_l182_182912


namespace b_sequence_is_constant_l182_182353

noncomputable def b_sequence_formula (a b : ℕ → ℝ) : Prop :=
  (∀ n : ℕ, n > 0 → ∃ d q : ℝ, (∀ n : ℕ, a (n + 1) = a n + d) ∧ (∀ n : ℕ, b (n + 1) = b n * q)) ∧
  (∀ n : ℕ, n > 0 → a (n + 1) / a n = b n) ∧
  (∀ n : ℕ, n > 0 → b n = 1)

theorem b_sequence_is_constant (a b : ℕ → ℝ) (h : b_sequence_formula a b) : ∀ n : ℕ, n > 0 → b n = 1 :=
  by
    sorry

end b_sequence_is_constant_l182_182353


namespace deepak_present_age_l182_182709

def rahul_age (x : ℕ) : ℕ := 4 * x
def deepak_age (x : ℕ) : ℕ := 3 * x

theorem deepak_present_age (x : ℕ) (h1 : rahul_age x + 10 = 26) : deepak_age x = 12 :=
by sorry

end deepak_present_age_l182_182709


namespace num_letters_with_line_no_dot_l182_182368

theorem num_letters_with_line_no_dot :
  ∀ (total_letters with_dot_and_line : ℕ) (with_dot_only with_line_only : ℕ),
    (total_letters = 60) →
    (with_dot_and_line = 20) →
    (with_dot_only = 4) →
    (total_letters = with_dot_and_line + with_dot_only + with_line_only) →
    with_line_only = 36 :=
by
  intros total_letters with_dot_and_line with_dot_only with_line_only
  intros h1 h2 h3 h4
  rw [h1, h2, h3] at h4
  sorry

end num_letters_with_line_no_dot_l182_182368


namespace johnnys_hourly_wage_l182_182827

def totalEarnings : ℝ := 26
def totalHours : ℝ := 8
def hourlyWage : ℝ := 3.25

theorem johnnys_hourly_wage : totalEarnings / totalHours = hourlyWage :=
by
  sorry

end johnnys_hourly_wage_l182_182827


namespace sufficient_food_l182_182329

-- Definitions for a large package and a small package as amounts of food lasting days
def largePackageDay : ℕ := L
def smallPackageDay : ℕ := S

-- Condition: One large package and four small packages last 14 days
axiom condition : largePackageDay + 4 * smallPackageDay = 14

-- Theorem to be proved: One large package and three small packages last at least 11 days
theorem sufficient_food : largePackageDay + 3 * smallPackageDay ≥ 11 :=
by
  sorry

end sufficient_food_l182_182329


namespace gnome_count_l182_182397

theorem gnome_count (g_R: ℕ) (g_W: ℕ) (h1: g_R = 4 * g_W) (h2: g_W = 20) : g_R - (40 * g_R / 100) = 48 := by
  sorry

end gnome_count_l182_182397


namespace sufficient_food_supply_l182_182312

variable {L S : ℝ}

theorem sufficient_food_supply (h1 : L + 4 * S = 14) (h2 : L > S) : L + 3 * S ≥ 11 :=
by
  sorry

end sufficient_food_supply_l182_182312


namespace minimum_valid_N_exists_l182_182767

theorem minimum_valid_N_exists (N : ℝ) (a : ℕ → ℕ) :
  (∀ n : ℕ, a n > 0) →
  (∀ n : ℕ, a n < a (n+1)) →
  (∀ n : ℕ, (a (2*n - 1) + a (2*n)) / a n = N) →
  N ≥ 4 :=
by
  sorry

end minimum_valid_N_exists_l182_182767


namespace sum_of_three_numbers_l182_182682

variable (x y z : ℝ)

theorem sum_of_three_numbers :
  y = 5 → 
  (x + y + z) / 3 = x + 10 →
  (x + y + z) / 3 = z - 15 →
  x + y + z = 30 :=
by
  intros hy h1 h2
  rw [hy] at h1 h2
  sorry

end sum_of_three_numbers_l182_182682


namespace largest_n_exists_l182_182186

theorem largest_n_exists :
  ∃ (n : ℕ), (∃ (x y z : ℕ), n^2 = x^2 + y^2 + z^2 + 2 * x * y + 2 * y * z + 2 * z * x + 3 * x + 3 * y + 3 * z - 8) ∧
    ∀ (m : ℕ), (∃ (x y z : ℕ), m^2 = x^2 + y^2 + z^2 + 2 * x * y + 2 * y * z + 2 * z * x + 3 * x + 3 * y + 3 * z - 8) →
    n ≥ m :=
  sorry

end largest_n_exists_l182_182186


namespace parabola_focus_distance_l182_182402

theorem parabola_focus_distance (p : ℝ) (h : 2 * p = 8) : p = 4 :=
  by
  sorry

end parabola_focus_distance_l182_182402


namespace eat_jar_together_time_l182_182446

-- Define the rate of the child
def child_rate := 1 / 6

-- Define the rate of Karlson who eats twice as fast as the child
def karlson_rate := 2 * child_rate

-- Define the combined rate when both eat together
def combined_rate := child_rate + karlson_rate

-- Prove that the time taken together to eat one jar is 2 minutes
theorem eat_jar_together_time : (1 / combined_rate) = 2 :=
by
  -- Add the proof steps here
  sorry

end eat_jar_together_time_l182_182446


namespace sean_needs_six_packs_l182_182531

/-- 
 Sean needs to replace 2 light bulbs in the bedroom, 
 1 in the bathroom, 1 in the kitchen, and 4 in the basement. 
 He also needs to replace 1/2 of that amount in the garage. 
 The bulbs come 2 per pack. 
 -/
def bedroom_bulbs: ℕ := 2
def bathroom_bulbs: ℕ := 1
def kitchen_bulbs: ℕ := 1
def basement_bulbs: ℕ := 4
def bulbs_per_pack: ℕ := 2

noncomputable def total_bulbs_needed_including_garage: ℕ := 
  let total_rooms_bulbs := bedroom_bulbs + bathroom_bulbs + kitchen_bulbs + basement_bulbs
  let garage_bulbs := total_rooms_bulbs / 2
  total_rooms_bulbs + garage_bulbs

noncomputable def total_packs_needed: ℕ := total_bulbs_needed_including_garage / bulbs_per_pack

theorem sean_needs_six_packs : total_packs_needed = 6 :=
by
  sorry

end sean_needs_six_packs_l182_182531


namespace socorro_training_hours_l182_182849

theorem socorro_training_hours :
  let daily_multiplication_time := 10  -- in minutes
  let daily_division_time := 20        -- in minutes
  let training_days := 10              -- in days
  let minutes_per_hour := 60           -- minutes in an hour
  let daily_total_time := daily_multiplication_time + daily_division_time
  let total_training_time := daily_total_time * training_days
  total_training_time / minutes_per_hour = 5 :=
by sorry

end socorro_training_hours_l182_182849


namespace inequality_proof_l182_182203

theorem inequality_proof (a b c d : ℝ) (h1 : b < a) (h2 : d < c) : a + c > b + d :=
  sorry

end inequality_proof_l182_182203


namespace units_digit_m_squared_plus_3_to_m_l182_182836

theorem units_digit_m_squared_plus_3_to_m (m : ℕ) (h : m = 2021^2 + 3^2021) : (m^2 + 3^m) % 10 = 7 :=
by
  sorry

end units_digit_m_squared_plus_3_to_m_l182_182836


namespace no_sport_members_count_l182_182226

theorem no_sport_members_count (n B T B_and_T : ℕ) (h1 : n = 27) (h2 : B = 17) (h3 : T = 19) (h4 : B_and_T = 11) : 
  n - (B + T - B_and_T) = 2 :=
by
  sorry

end no_sport_members_count_l182_182226


namespace find_V_y_l182_182251

-- Define the volumes and percentages given in the problem
def V_x : ℕ := 300
def percent_x : ℝ := 0.10
def percent_y : ℝ := 0.30
def desired_percent : ℝ := 0.22

-- Define the alcohol volumes in the respective solutions
def alcohol_x := percent_x * V_x
def total_volume (V_y : ℕ) := V_x + V_y
def desired_alcohol (V_y : ℕ) := desired_percent * (total_volume V_y)

-- Define our main statement
theorem find_V_y : ∃ (V_y : ℕ), alcohol_x + (percent_y * V_y) = desired_alcohol V_y ∧ V_y = 450 :=
by
  sorry

end find_V_y_l182_182251


namespace simplify_expression_l182_182042

theorem simplify_expression :
  (1 / (Real.sqrt 8 + Real.sqrt 11) +
   1 / (Real.sqrt 11 + Real.sqrt 14) +
   1 / (Real.sqrt 14 + Real.sqrt 17) +
   1 / (Real.sqrt 17 + Real.sqrt 20) +
   1 / (Real.sqrt 20 + Real.sqrt 23) +
   1 / (Real.sqrt 23 + Real.sqrt 26) +
   1 / (Real.sqrt 26 + Real.sqrt 29) +
   1 / (Real.sqrt 29 + Real.sqrt 32)) = 
  (2 * Real.sqrt 2 / 3) :=
by sorry

end simplify_expression_l182_182042


namespace domain_of_f_l182_182544

def domain_of_log_func := Set ℝ

def is_valid (x : ℝ) : Prop := x - 1 > 0

def func_domain (f : ℝ → ℝ) : domain_of_log_func := {x : ℝ | is_valid x}

theorem domain_of_f :
  func_domain (λ x => Real.log (x - 1)) = {x : ℝ | 1 < x} := by
  sorry

end domain_of_f_l182_182544


namespace find_m_l182_182199

theorem find_m (m : ℝ) (h : (4 * (-1)^3 + 3 * m * (-1)^2 + 6 * (-1) = 2)) :
  m = 4 :=
by
  sorry

end find_m_l182_182199


namespace xy_sum_is_one_l182_182249

theorem xy_sum_is_one (x y : ℝ) (h : x^2 + y^2 + x * y = 12 * x - 8 * y + 2) : x + y = 1 :=
sorry

end xy_sum_is_one_l182_182249


namespace andres_possibilities_10_dollars_l182_182751

theorem andres_possibilities_10_dollars : 
  (∃ (num_1_coins num_2_coins num_5_bills : ℕ),
    num_1_coins + 2 * num_2_coins + 5 * num_5_bills = 10) → 
  ∃ (ways : ℕ), ways = 10 :=
by
  -- The proof can be provided here, but we'll use sorry to skip it in this template.
  sorry

end andres_possibilities_10_dollars_l182_182751


namespace find_f3_l182_182193

theorem find_f3 (f : ℝ → ℝ) 
  (h : ∀ x : ℝ, f x + 3 * f (1 - x) = 4 * x^3) : f 3 = -25.5 :=
sorry

end find_f3_l182_182193


namespace total_wheels_l182_182865

def num_wheels_in_garage : Nat :=
  let cars := 2 * 4
  let lawnmower := 4
  let bicycles := 3 * 2
  let tricycle := 3
  let unicycle := 1
  let skateboard := 4
  let wheelbarrow := 1
  let wagon := 4
  let dolly := 2
  let shopping_cart := 4
  let scooter := 2
  cars + lawnmower + bicycles + tricycle + unicycle + skateboard + wheelbarrow + wagon + dolly + shopping_cart + scooter

theorem total_wheels : num_wheels_in_garage = 39 := by
  sorry

end total_wheels_l182_182865


namespace eq_irrational_parts_l182_182669

theorem eq_irrational_parts (a b c d : ℝ) (h : a + b * (Real.sqrt 5) = c + d * (Real.sqrt 5)) : a = c ∧ b = d := 
by 
  sorry

end eq_irrational_parts_l182_182669


namespace student_incorrect_answer_l182_182224

theorem student_incorrect_answer (D I : ℕ) (h1 : D / 63 = I) (h2 : D / 36 = 42) : I = 24 := by
  sorry

end student_incorrect_answer_l182_182224


namespace Brenda_bakes_20_cakes_a_day_l182_182903

-- Define the conditions
variables (x : ℕ)

-- Other necessary definitions
def cakes_baked_in_9_days (x : ℕ) : ℕ := 9 * x
def cakes_after_selling_half (total_cakes : ℕ) : ℕ := total_cakes.div2

-- Given condition that Brenda has 90 cakes after selling half
def final_cakes_after_selling : ℕ := 90

-- Mathematical statement we want to prove
theorem Brenda_bakes_20_cakes_a_day (x : ℕ) (h : cakes_after_selling_half (cakes_baked_in_9_days x) = final_cakes_after_selling) : x = 20 :=
by sorry

end Brenda_bakes_20_cakes_a_day_l182_182903


namespace icing_cubes_count_31_l182_182889

def cake_cubed (n : ℕ) := n^3

noncomputable def slabs_with_icing (n : ℕ): ℕ := 
    let num_faces := 3
    let edge_per_face := n - 1
    let edges_with_icing := num_faces * edge_per_face * (n - 2)
    edges_with_icing + (n - 2) * 4 * (n - 2)

theorem icing_cubes_count_31 : ∀ (n : ℕ), n = 5 → slabs_with_icing n = 31 :=
by
  intros n hn
  revert hn
  sorry

end icing_cubes_count_31_l182_182889


namespace average_age_l182_182258

theorem average_age (avg_age_students : ℝ) (num_students : ℕ) (avg_age_teachers : ℝ) (num_teachers : ℕ) :
  avg_age_students = 13 → 
  num_students = 40 → 
  avg_age_teachers = 42 → 
  num_teachers = 60 → 
  (num_students * avg_age_students + num_teachers * avg_age_teachers) / (num_students + num_teachers) = 30.4 :=
by
  intros h1 h2 h3 h4
  sorry

end average_age_l182_182258


namespace range_of_a_l182_182774

-- Define the condition function
def inequality (a x : ℝ) : Prop := a^2 * x - 2 * (a - x - 4) < 0

-- Prove that given the inequality always holds for any real x, the range of a is (-2, 2]
theorem range_of_a {a : ℝ} (h : ∀ x : ℝ, inequality a x) : -2 < a ∧ a ≤ 2 := by
  sorry

end range_of_a_l182_182774


namespace fish_initial_numbers_l182_182514

theorem fish_initial_numbers (x y : ℕ) (h1 : x + y = 100) (h2 : x - 30 = y - 40) : x = 45 ∧ y = 55 :=
by
  sorry

end fish_initial_numbers_l182_182514


namespace possible_orange_cells_l182_182282

theorem possible_orange_cells :
  ∃ (n : ℕ), n = 2021 * 2020 ∨ n = 2022 * 2020 := 
sorry

end possible_orange_cells_l182_182282


namespace smallest_angle_in_convex_15sided_polygon_l182_182049

def isConvexPolygon (n : ℕ) (angles : Fin n → ℚ) : Prop :=
  ∑ i, angles i = (n - 2) * 180 ∧ ∀ i,  angles i < 180

def arithmeticSequence (angles : Fin 15 → ℚ) : Prop :=
  ∃ a d : ℚ, ∀ i : Fin 15, angles i = a + i * d

def increasingSequence (angles : Fin 15 → ℚ) : Prop :=
  ∀ i j : Fin 15, i < j → angles i < angles j

def integerSequence (angles : Fin 15 → ℚ) : Prop :=
  ∀ i : Fin 15, (angles i : ℚ) = angles i

theorem smallest_angle_in_convex_15sided_polygon :
  ∃ (angles : Fin 15 → ℚ),
    isConvexPolygon 15 angles ∧
    arithmeticSequence angles ∧
    increasingSequence angles ∧
    integerSequence angles ∧
    angles 0 = 135 :=
by
  sorry

end smallest_angle_in_convex_15sided_polygon_l182_182049


namespace find_k_l182_182004

theorem find_k (k : ℝ) (h1 : k > 0) (h2 : ∀ x ∈ Set.Icc (2 : ℝ) 4, y = k / x → y ≥ 5) : k = 20 :=
sorry

end find_k_l182_182004


namespace sum_of_numbers_l182_182409

theorem sum_of_numbers : ∃ (a b : ℕ), (a + b = 21) ∧ (a / b = 3 / 4) ∧ (max a b = 12) :=
by
  sorry

end sum_of_numbers_l182_182409


namespace sally_balloons_l182_182041

theorem sally_balloons (F S : ℕ) (h1 : F = 3 * S) (h2 : F = 18) : S = 6 :=
by sorry

end sally_balloons_l182_182041


namespace prism_volume_l182_182562

theorem prism_volume 
    (x y z : ℝ) 
    (h_xy : x * y = 18) 
    (h_yz : y * z = 12) 
    (h_xz : x * z = 8) 
    (h_longest_shortest : max x (max y z) = 2 * min x (min y z)) : 
    x * y * z = 16 := 
  sorry

end prism_volume_l182_182562


namespace simplify_exponent_fraction_l182_182275

theorem simplify_exponent_fraction : (3 ^ 2015 + 3 ^ 2013) / (3 ^ 2015 - 3 ^ 2013) = 5 / 4 := by
  sorry

end simplify_exponent_fraction_l182_182275


namespace total_legs_of_passengers_l182_182087

theorem total_legs_of_passengers :
  ∀ (total_heads cats cat_legs human_heads normal_human_legs one_legged_captain_legs : ℕ),
  total_heads = 15 →
  cats = 7 →
  cat_legs = 4 →
  human_heads = (total_heads - cats) →
  normal_human_legs = 2 →
  one_legged_captain_legs = 1 →
  ((cats * cat_legs) + ((human_heads - 1) * normal_human_legs) + one_legged_captain_legs) = 43 :=
by
  intros total_heads cats cat_legs human_heads normal_human_legs one_legged_captain_legs h1 h2 h3 h4 h5 h6
  sorry

end total_legs_of_passengers_l182_182087


namespace problem_1_problem_2_l182_182359

-- Definition of the function f
def f (x a : ℝ) : ℝ := |x - a| + 3 * x

-- Problem I
theorem problem_1 (x : ℝ) : (f x 1 ≥ 3 * x + 2) ↔ (x ≥ 3 ∨ x ≤ -1) :=
by sorry

-- Problem II
theorem problem_2 (a : ℝ) (h : ∀ x : ℝ, f x a ≤ 0 → x ≤ -3) : a = 6 :=
by sorry

end problem_1_problem_2_l182_182359


namespace correct_equation_l182_182703

theorem correct_equation : ∃a : ℝ, (-3 * a) ^ 2 = 9 * a ^ 2 :=
by
  use 1
  sorry

end correct_equation_l182_182703


namespace length_of_second_platform_l182_182580

-- Given conditions
def length_of_train : ℕ := 310
def length_of_first_platform : ℕ := 110
def time_to_cross_first_platform : ℕ := 15
def time_to_cross_second_platform : ℕ := 20

-- Calculated based on conditions
def total_distance_first_platform : ℕ :=
  length_of_train + length_of_first_platform

def speed_of_train : ℕ :=
  total_distance_first_platform / time_to_cross_first_platform

def total_distance_second_platform : ℕ :=
  speed_of_train * time_to_cross_second_platform

-- Statement to prove
theorem length_of_second_platform :
  total_distance_second_platform = length_of_train + 250 := sorry

end length_of_second_platform_l182_182580


namespace projection_non_ambiguity_l182_182231

theorem projection_non_ambiguity 
    (a b c : ℝ) 
    (theta : ℝ) 
    (h : a^2 = b^2 + c^2 - 2 * b * c * Real.cos theta) : 
    ∃ (c' : ℝ), c' = c * Real.cos theta ∧ a^2 = b^2 + c^2 + 2 * b * c' := 
sorry

end projection_non_ambiguity_l182_182231


namespace range_of_a_l182_182003

-- Define the function f(x)
def f (a x : ℝ) : ℝ := log a (a * x^2 - 4 * x + 9)

-- Define the interval [1, 3]
def interval := Icc 1 3

-- Define the condition for function g(x) = ax^2 - 4x + 9 to be positive in [1, 3]
def g_pos (a x : ℝ) : Prop := a * x^2 - 4 * x + 9 > 0

-- State the theorem
theorem range_of_a (a : ℝ) (h_pos : 0 < a) (h_neq : a ≠ 1) :
  (∀ x ∈ interval, g_pos a x) →
  (strict_mono_on (f a) interval ↔ a ∈ (Ioc (1/3 : ℝ) (2/3)) ∪ Ici (2 : ℝ)) :=
by sorry

end range_of_a_l182_182003


namespace bob_miles_run_on_day_three_l182_182587

theorem bob_miles_run_on_day_three :
  ∀ (total_miles miles_day1 miles_day2 miles_day3 : ℝ),
    total_miles = 70 →
    miles_day1 = 0.20 * total_miles →
    miles_day2 = 0.50 * (total_miles - miles_day1) →
    miles_day3 = total_miles - miles_day1 - miles_day2 →
    miles_day3 = 28 :=
by
  intros total_miles miles_day1 miles_day2 miles_day3 ht hm1 hm2 hm3
  rw [ht, hm1, hm2, hm3]
  sorry

end bob_miles_run_on_day_three_l182_182587


namespace find_percentage_l182_182444

theorem find_percentage (P : ℝ) (h1 : (P / 100) * 200 = 30 + 0.60 * 50) : P = 30 :=
by
  sorry

end find_percentage_l182_182444


namespace intersection_of_sets_l182_182979

open Set Int

theorem intersection_of_sets (S T : Set Int) (hS : S = {s | ∃ n : ℤ, s = 2 * n + 1}) (hT : T = {t | ∃ n : ℤ, t = 4 * n + 1}) : S ∩ T = T :=
  by
  sorry

end intersection_of_sets_l182_182979


namespace intersection_eq_T_l182_182991

def S : Set ℤ := {s | ∃ n : ℤ, s = 2 * n + 1}
def T : Set ℤ := {t | ∃ n : ℤ, t = 4 * n + 1}

theorem intersection_eq_T : S ∩ T = T := sorry

end intersection_eq_T_l182_182991


namespace intersection_eq_T_l182_182940

def S : Set ℤ := {s | ∃ n : ℤ, s = 2 * n + 1}
def T : Set ℤ := {t | ∃ n : ℤ, t = 4 * n + 1}

theorem intersection_eq_T : S ∩ T = T := 
sorry

end intersection_eq_T_l182_182940


namespace needs_debugging_defective_parts_count_parts_exceeding_200_12_l182_182012

noncomputable def normal_distribution (μ σ : ℝ) : Type := sorry

def part_inner_diameter := normal_distribution 200 0.06

def valid_inner_diameter_range := (199.82, 200.18)

def measured_diameters := [199.87, 199.91, 199.99, 200.13, 200.19]

theorem needs_debugging : ¬ (∀ x ∈ measured_diameters, 199.82 < x ∧ x < 200.18) :=
sorry

def total_parts := 10000

def defective p : ℕ → ℝ := (1 - p)

theorem defective_parts_count (p : ℝ) (n : ℕ) (h1 : p = 0.003) (h2 : n = total_parts) :
  30 ≤ n * p ∧ n * p < 31 :=
sorry

theorem parts_exceeding_200_12 (p : ℝ) (h : p = 0.0225) :
  225 ≤ total_parts * p :=
sorry

end needs_debugging_defective_parts_count_parts_exceeding_200_12_l182_182012


namespace S_inter_T_eq_T_l182_182967

-- Define the sets S and T
def S := { s : ℤ | ∃ n : ℤ, s = 2 * n + 1 }
def T := { t : ℤ | ∃ n : ℤ, t = 4 * n + 1 }

-- Prove that S ∩ T = T
theorem S_inter_T_eq_T : S ∩ T = T := 
by
  sorry

end S_inter_T_eq_T_l182_182967


namespace max_value_of_expression_l182_182678

def real_numbers (m n : ℝ) := m > 0 ∧ n < 0 ∧ (1 / m + 1 / n = 1)

theorem max_value_of_expression (m n : ℝ) (h : real_numbers m n) : 4 * m + n ≤ 1 :=
  sorry

end max_value_of_expression_l182_182678


namespace intersection_of_sets_l182_182982

open Set Int

theorem intersection_of_sets (S T : Set Int) (hS : S = {s | ∃ n : ℤ, s = 2 * n + 1}) (hT : T = {t | ∃ n : ℤ, t = 4 * n + 1}) : S ∩ T = T :=
  by
  sorry

end intersection_of_sets_l182_182982


namespace flour_masses_l182_182479

theorem flour_masses (x : ℝ) (h: 
    (x * (1 + x / 100) + (x + 10) * (1 + (x + 10) / 100) = 112.5)) :
    x = 35 ∧ (x + 10) = 45 :=
by 
  sorry

end flour_masses_l182_182479


namespace evaluate_f_l182_182207

def f (x : ℚ) : ℚ := (2 * x - 3) / (3 * x ^ 2 - 1)

theorem evaluate_f :
  f (-2) = -7 / 11 ∧ f (0) = 3 ∧ f (1) = -1 / 2 :=
by
  sorry

end evaluate_f_l182_182207


namespace quadratic_has_solution_l182_182390

theorem quadratic_has_solution (a b : ℝ) : ∃ x : ℝ, (a^6 - b^6) * x^2 + 2 * (a^5 - b^5) * x + (a^4 - b^4) = 0 :=
  by sorry

end quadratic_has_solution_l182_182390


namespace S_inter_T_eq_T_l182_182970

-- Define the sets S and T
def S := { s : ℤ | ∃ n : ℤ, s = 2 * n + 1 }
def T := { t : ℤ | ∃ n : ℤ, t = 4 * n + 1 }

-- Prove that S ∩ T = T
theorem S_inter_T_eq_T : S ∩ T = T := 
by
  sorry

end S_inter_T_eq_T_l182_182970


namespace minimum_value_of_absolute_sum_l182_182658

theorem minimum_value_of_absolute_sum (x : ℝ) :
  ∃ y : ℝ, (∀ x : ℝ, y ≤ |x + 1| + |x + 2| + |x + 3| + |x + 4| + |x + 5|) ∧ y = 6 :=
sorry

end minimum_value_of_absolute_sum_l182_182658


namespace largest_x_l182_182074

-- Define the condition of the problem.
def equation_holds (x : ℝ) : Prop :=
  (5 * x - 20) / (4 * x - 5) ^ 2 + (5 * x - 20) / (4 * x - 5) = 20

-- State the theorem to prove the largest value of x is 9/5.
theorem largest_x : ∃ x : ℝ, equation_holds x ∧ ∀ y : ℝ, equation_holds y → y ≤ 9 / 5 :=
by
  sorry

end largest_x_l182_182074


namespace cost_price_computer_table_l182_182059

theorem cost_price_computer_table (C S : ℝ) (hS1 : S = 1.25 * C) (hS2 : S = 1000) : C = 800 :=
by
  sorry

end cost_price_computer_table_l182_182059


namespace renu_work_rate_l182_182846

theorem renu_work_rate (R : ℝ) :
  (∀ (renu_rate suma_rate combined_rate : ℝ),
    renu_rate = 1 / R ∧
    suma_rate = 1 / 6 ∧
    combined_rate = 1 / 3 ∧    
    combined_rate = renu_rate + suma_rate) → 
    R = 6 :=
by
  sorry

end renu_work_rate_l182_182846


namespace Kim_total_hours_l182_182021

-- Define the initial conditions
def initial_classes : ℕ := 4
def hours_per_class : ℕ := 2
def dropped_class : ℕ := 1

-- The proof problem: Given the initial conditions, prove the total hours of classes per day is 6
theorem Kim_total_hours : (initial_classes - dropped_class) * hours_per_class = 6 := by
  sorry

end Kim_total_hours_l182_182021


namespace three_digit_numbers_l182_182606

theorem three_digit_numbers (n : ℕ) :
  (100 ≤ n ∧ n ≤ 999) → 
  (n * n % 1000 = n % 1000) ↔ 
  (n = 625 ∨ n = 376) :=
by 
  sorry

end three_digit_numbers_l182_182606


namespace club_additional_members_l182_182720

theorem club_additional_members (current_members : ℕ) (additional_members : ℕ) 
  (h1 : current_members = 10) 
  (h2 : additional_members = 5 + 2 * current_members - current_members) : 
  additional_members = 15 :=
by 
  rw [h1] at h2
  norm_num at h2
  exact h2

end club_additional_members_l182_182720


namespace fewest_cookies_l182_182298

theorem fewest_cookies
  (r a s d1 d2 : ℝ)
  (hr_pos : r > 0)
  (ha_pos : a > 0)
  (hs_pos : s > 0)
  (hd1_pos : d1 > 0)
  (hd2_pos : d2 > 0)
  (h_Alice_cookies : 15 = 15)
  (h_same_dough : true) :
  15 < (15 * (Real.pi * r^2)) / (a^2) ∧
  15 < (15 * (Real.pi * r^2)) / ((3 * Real.sqrt 3 / 2) * s^2) ∧
  15 < (15 * (Real.pi * r^2)) / ((1 / 2) * d1 * d2) :=
by
  sorry

end fewest_cookies_l182_182298


namespace phoenix_flight_l182_182539

theorem phoenix_flight : ∃ n : ℕ, 3 ^ n > 6560 ∧ ∀ m < n, 3 ^ m ≤ 6560 :=
by sorry

end phoenix_flight_l182_182539


namespace fraction_equality_l182_182347

theorem fraction_equality (a b : ℝ) (h : a / b = 2 / 3) : a / (a + b) = 2 / 5 :=
by
  sorry

end fraction_equality_l182_182347


namespace triangle_area_l182_182413

def is_right_triangle (a b c : ℕ) : Prop :=
  a^2 + b^2 = c^2

theorem triangle_area (a b c : ℕ) (h1 : a = 9) (h2 : b = 40) (h3 : c = 41) (h4 : is_right_triangle a b c) :
  (1 / 2 : ℝ) * a * b = 180 :=
by sorry

end triangle_area_l182_182413


namespace quadratic_roots_expression_eq_zero_l182_182614

theorem quadratic_roots_expression_eq_zero
  (a b c : ℝ)
  (h : ∀ x : ℝ, a * x^2 + b * x + c = 0)
  (x1 x2 : ℝ)
  (hx1 : a * x1^2 + b * x1 + c = 0)
  (hx2 : a * x2^2 + b * x2 + c = 0)
  (s1 s2 s3 : ℝ)
  (h_s1 : s1 = x1 + x2)
  (h_s2 : s2 = x1^2 + x2^2)
  (h_s3 : s3 = x1^3 + x2^3) :
  a * s3 + b * s2 + c * s1 = 0 := sorry

end quadratic_roots_expression_eq_zero_l182_182614


namespace proof_problem_l182_182028

variables {a b c : ℝ}

theorem proof_problem (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : 4 * a^2 + b^2 + 16 * c^2 = 1) :
  (0 < a * b ∧ a * b < 1 / 4) ∧ (1 / a^2 + 1 / b^2 + 1 / (4 * a * b * c^2) > 49) :=
by
  sorry

end proof_problem_l182_182028


namespace clock_90_degree_angle_times_l182_182393

noncomputable def first_time_90_degree_angle (t : ℝ) : Prop := 5.5 * t = 90

noncomputable def second_time_90_degree_angle (t : ℝ) : Prop := 5.5 * t = 270

theorem clock_90_degree_angle_times :
  ∃ t₁ t₂ : ℝ,
  first_time_90_degree_angle t₁ ∧ 
  second_time_90_degree_angle t₂ ∧ 
  t₁ = (180 / 11 : ℝ) ∧ 
  t₂ = (540 / 11 : ℝ) :=
by
  sorry

end clock_90_degree_angle_times_l182_182393


namespace fixed_point_of_inverse_l182_182216

-- Define an odd function f on ℝ
def odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = - f (x)

-- Define the transformed function g
def g (f : ℝ → ℝ) (x : ℝ) := f (x + 1) - 2

-- Define the condition for a point to be on the inverse of a function
def inv_contains (f : ℝ → ℝ) (p : ℝ × ℝ) : Prop :=
  p.2 = f p.1

-- The theorem statement
theorem fixed_point_of_inverse (f : ℝ → ℝ) 
  (Hf_odd : odd_function f) :
  inv_contains (λ y => g f (y)) (-2, -1) :=
sorry

end fixed_point_of_inverse_l182_182216


namespace determine_a_l182_182491

theorem determine_a (a : ℝ) (f : ℝ → ℝ) (h : f = fun x => a * x^3 - 2 * x) (pt : f (-1) = 4) : a = -2 := by
  sorry

end determine_a_l182_182491


namespace cos_90_eq_zero_l182_182163

def point_after_rotation (θ : ℝ) : ℝ × ℝ :=
  let x := cos θ
  let y := sin θ
  (x, y)

theorem cos_90_eq_zero : cos (real.pi / 2) = 0 :=
  sorry

end cos_90_eq_zero_l182_182163


namespace intersection_eq_T_l182_182935

def S : Set ℤ := {s | ∃ n : ℤ, s = 2 * n + 1}
def T : Set ℤ := {t | ∃ n : ℤ, t = 4 * n + 1}

theorem intersection_eq_T : S ∩ T = T := 
sorry

end intersection_eq_T_l182_182935


namespace length_each_stitch_l182_182650

theorem length_each_stitch 
  (hem_length_feet : ℝ) 
  (stitches_per_minute : ℝ) 
  (hem_time_minutes : ℝ) 
  (hem_length_inches : ℝ) 
  (total_stitches : ℝ) 
  (stitch_length_inches : ℝ) 
  (h1 : hem_length_feet = 3) 
  (h2 : stitches_per_minute = 24) 
  (h3 : hem_time_minutes = 6) 
  (h4 : hem_length_inches = hem_length_feet * 12) 
  (h5 : total_stitches = stitches_per_minute * hem_time_minutes) 
  (h6 : stitch_length_inches = hem_length_inches / total_stitches) :
  stitch_length_inches = 0.25 :=
by
  sorry

end length_each_stitch_l182_182650


namespace cos_ninety_degrees_l182_182120

theorem cos_ninety_degrees : Real.cos (90 * Real.pi / 180) = 0 := 
by 
  sorry

end cos_ninety_degrees_l182_182120


namespace ratio_avg_eq_42_l182_182061

theorem ratio_avg_eq_42 (a b c d : ℕ)
  (h1 : ∃ k : ℕ, a = 2 * k ∧ b = 3 * k ∧ c = 4 * k ∧ d = 5 * k)
  (h2 : (a + b + c + d) / 4 = 42) : a = 24 :=
by sorry

end ratio_avg_eq_42_l182_182061


namespace correct_answer_l182_182761

-- Definitions of the groups
def group_1_well_defined : Prop := false -- Smaller numbers
def group_2_well_defined : Prop := true  -- Non-negative even numbers not greater than 10
def group_3_well_defined : Prop := true  -- All triangles
def group_4_well_defined : Prop := false -- Tall male students

-- Propositions representing the options
def option_A : Prop := group_1_well_defined ∧ group_4_well_defined
def option_B : Prop := group_2_well_defined ∧ group_3_well_defined
def option_C : Prop := group_2_well_defined
def option_D : Prop := group_3_well_defined

-- Theorem stating Option B is the correct answer
theorem correct_answer : option_B ∧ ¬option_A ∧ ¬option_C ∧ ¬option_D := by
  sorry

end correct_answer_l182_182761


namespace compare_neg_rational_l182_182113

def neg_one_third : ℚ := -1 / 3
def neg_one_half : ℚ := -1 / 2

theorem compare_neg_rational : neg_one_third > neg_one_half :=
by sorry

end compare_neg_rational_l182_182113


namespace quadratic_inequality_l182_182784

theorem quadratic_inequality (a : ℝ) (h : ∀ x : ℝ, x^2 - a * x + a > 0) : 0 < a ∧ a < 4 :=
sorry

end quadratic_inequality_l182_182784


namespace total_balloons_l182_182070

def tom_balloons : Nat := 9
def sara_balloons : Nat := 8

theorem total_balloons : tom_balloons + sara_balloons = 17 := 
by
  sorry

end total_balloons_l182_182070


namespace original_grain_correct_l182_182577

-- Define the initial quantities
def grain_spilled : ℕ := 49952
def grain_remaining : ℕ := 918

-- Define the original amount of grain expected
def original_grain : ℕ := 50870

-- Prove that the original amount of grain was correct
theorem original_grain_correct : grain_spilled + grain_remaining = original_grain := 
by
  sorry

end original_grain_correct_l182_182577


namespace find_alpha_l182_182372

noncomputable section

open Real 

def curve_C1 (x y : ℝ) : Prop := x + y = 1
def curve_C2 (x y φ : ℝ) : Prop := x = 2 + 2 * cos φ ∧ y = 2 * sin φ 

def polar_coordinate_eq1 (ρ θ : ℝ) : Prop := ρ * sin (θ + π / 4) = sqrt 2 / 2
def polar_coordinate_eq2 (ρ θ : ℝ) : Prop := ρ = 4 * cos θ

def line_l (ρ θ α : ℝ)  (hα: α > 0 ∧ α < π / 2) : Prop := θ = α ∧ ρ > 0 

def OB_div_OA_eq_4 (ρA ρB α : ℝ) : Prop := ρB / ρA = 4

theorem find_alpha (α : ℝ) (hα: α > 0 ∧ α < π / 2)
  (h₁: ∀ (x y ρ θ: ℝ), curve_C1 x y → polar_coordinate_eq1 ρ θ) 
  (h₂: ∀ (x y φ ρ θ: ℝ), curve_C2 x y φ → polar_coordinate_eq2 ρ θ) 
  (h₃: ∀ (ρ θ: ℝ), line_l ρ θ α hα) 
  (h₄: ∀ (ρA ρB : ℝ), OB_div_OA_eq_4 ρA ρB α → ρA = 1 / (cos α + sin α) ∧ ρB = 4 * cos α ): 
  α = 3 * π / 8 :=
by
  sorry

end find_alpha_l182_182372


namespace find_square_l182_182058

theorem find_square (s : ℕ) : 
    (7863 / 13 = 604 + (s / 13)) → s = 11 :=
by
  sorry

end find_square_l182_182058


namespace sum_of_coefficients_l182_182024

theorem sum_of_coefficients (p : ℕ) (hp : Nat.Prime p) (a b c : ℕ)
    (f : ℕ → ℕ) (hf : ∀ x, f x = a * x ^ 2 + b * x + c)
    (h_range : 0 < a ∧ a ≤ p ∧ 0 < b ∧ b ≤ p ∧ 0 < c ∧ c ≤ p)
    (h_div : ∀ x, x > 0 → p ∣ (f x)) : 
    a + b + c = 3 * p := 
sorry

end sum_of_coefficients_l182_182024


namespace intersection_eq_T_l182_182932

def S : Set ℤ := {s | ∃ n : ℤ, s = 2 * n + 1}
def T : Set ℤ := {t | ∃ n : ℤ, t = 4 * n + 1}

theorem intersection_eq_T : S ∩ T = T := 
sorry

end intersection_eq_T_l182_182932


namespace largest_lambda_inequality_l182_182922

theorem largest_lambda_inequality :
  ∀ (a b c d e : ℝ), 0 ≤ a → 0 ≤ b → 0 ≤ c → 0 ≤ d → 0 ≤ e →
  (a^2 + b^2 + c^2 + d^2 + e^2 ≥ a * b + (5/4) * b * c + c * d + d * e) :=
by
  sorry

end largest_lambda_inequality_l182_182922


namespace recurring_decimal_sum_l182_182341

-- Definitions based on the conditions identified
def recurringDecimal (n : ℕ) : ℚ := n / 9
def r8 := recurringDecimal 8
def r2 := recurringDecimal 2
def r6 := recurringDecimal 6
def r6_simplified : ℚ := 2 / 3

-- The theorem to prove
theorem recurring_decimal_sum : r8 + r2 - r6_simplified = 4 / 9 :=
by
  -- Proof steps will go here (but are omitted because of the problem requirements)
  sorry

end recurring_decimal_sum_l182_182341


namespace condition_eq_l182_182513

-- We are given a triangle ABC with sides opposite angles A, B, and C being a, b, and c respectively.
variable (A B C a b c : ℝ)

-- Conditions for the problem
def sin_eq (A B : ℝ) := Real.sin A = Real.sin B
def cos_eq (A B : ℝ) := Real.cos A = Real.cos B
def sin2_eq (A B : ℝ) := Real.sin (2 * A) = Real.sin (2 * B)
def cos2_eq (A B : ℝ) := Real.cos (2 * A) = Real.cos (2 * B)

-- The main statement we need to prove
theorem condition_eq (h1 : sin_eq A B) (h2 : cos_eq A B) (h4 : cos2_eq A B) : a = b :=
sorry

end condition_eq_l182_182513


namespace cos_90_eq_zero_l182_182176

theorem cos_90_eq_zero : Real.cos (90 * Real.pi / 180) = 0 := by 
  sorry

end cos_90_eq_zero_l182_182176


namespace bubbleSort_iter_count_l182_182355

/-- Bubble sort iterates over the list repeatedly, swapping adjacent elements if they are in the wrong order. -/
def bubbleSortSteps (lst : List Int) : List (List Int) :=
sorry -- Implementation of bubble sort to capture each state after each iteration

/-- Prove that sorting [6, -3, 0, 15] in descending order using bubble sort requires exactly 3 iterations. -/
theorem bubbleSort_iter_count : 
  (bubbleSortSteps [6, -3, 0, 15]).length = 3 :=
sorry

end bubbleSort_iter_count_l182_182355


namespace part_I_part_II_l182_182358

-- Part I
theorem part_I (x : ℝ) : (|x + 1| + |x - 4| ≤ 2 * |x - 4|) ↔ (x < 1.5) :=
sorry

-- Part II
theorem part_II (a : ℝ) : (∀ x : ℝ, |x + a| + |x - 4| ≥ 3) → (a ≤ -7 ∨ a ≥ -1) :=
sorry

end part_I_part_II_l182_182358


namespace tan_sum_eq_l182_182214

theorem tan_sum_eq (α : ℝ) (h : Real.tan (α + Real.pi / 4) = 2) :
  (Real.sin α - Real.cos α) / (Real.sin α + Real.cos α) = -1/2 :=
by sorry

end tan_sum_eq_l182_182214


namespace intersection_of_sets_l182_182977

open Set Int

theorem intersection_of_sets (S T : Set Int) (hS : S = {s | ∃ n : ℤ, s = 2 * n + 1}) (hT : T = {t | ∃ n : ℤ, t = 4 * n + 1}) : S ∩ T = T :=
  by
  sorry

end intersection_of_sets_l182_182977


namespace cos_90_eq_zero_l182_182147

theorem cos_90_eq_zero : (cos 90) = 0 := by
  -- Condition: The angle 90 degrees corresponds to the point (0, 1) on the unit circle.
  -- We claim cos 90 degrees is 0
  sorry

end cos_90_eq_zero_l182_182147


namespace difference_of_squares_l182_182869

theorem difference_of_squares : 
  let a := 625
  let b := 575
  (a^2 - b^2) = 60000 :=
by 
  let a := 625
  let b := 575
  sorry

end difference_of_squares_l182_182869


namespace sin_double_angle_l182_182794

theorem sin_double_angle (θ : ℝ) (h : Real.sin (π / 4 + θ) = 1 / 3) : Real.sin (2 * θ) = -7 / 9 :=
by
  sorry

end sin_double_angle_l182_182794


namespace average_of_first_two_numbers_l182_182047

theorem average_of_first_two_numbers (s1 s2 s3 s4 s5 s6 a b c : ℝ) 
  (h_average_six : (s1 + s2 + s3 + s4 + s5 + s6) / 6 = 4.6)
  (h_average_set2 : (s3 + s4) / 2 = 3.8)
  (h_average_set3 : (s5 + s6) / 2 = 6.6)
  (h_total_sum : s1 + s2 + s3 + s4 + s5 + s6 = 27.6) : 
  (s1 + s2) / 2 = 3.4 :=
sorry

end average_of_first_two_numbers_l182_182047


namespace least_positive_value_tan_inv_k_l182_182837

theorem least_positive_value_tan_inv_k 
  (a b : ℝ) 
  (x : ℝ) 
  (h1 : Real.tan x = a / b) 
  (h2 : Real.tan (2 * x) = 2 * b / (a + 2 * b)) 
  : x = Real.arctan 1 := 
sorry

end least_positive_value_tan_inv_k_l182_182837


namespace pentagon_area_l182_182806

noncomputable def area_of_pentagon (a b c d e : ℕ) : ℕ := 
  let area_triangle := (1/2) * a * b
  let area_trapezoid := (1/2) * (c + e) * d
  area_triangle + area_trapezoid

theorem pentagon_area : area_of_pentagon 18 25 30 28 25 = 995 :=
by sorry

end pentagon_area_l182_182806


namespace log_det_solution_l182_182487

theorem log_det_solution (x : ℝ) : 
  (log (Real.sqrt 2) (abs (x - 11)) < 0) ↔ (0 < x ∧ x < 1) ∨ (1 < x ∧ x < 2) := 
by 
  sorry

end log_det_solution_l182_182487


namespace sibling_age_difference_l182_182268

theorem sibling_age_difference (Y : ℝ) (Y_eq : Y = 25.75) (avg_age_eq : (Y + (Y + 3) + (Y + 6) + (Y + x)) / 4 = 30) : (Y + 6) - Y = 6 :=
by
  sorry

end sibling_age_difference_l182_182268


namespace disjoint_subsets_mod_1000_l182_182519

open Nat

theorem disjoint_subsets_mod_1000 :
  let T := Finset.range 13
  let m := (3^12 - 2 * 2^12 + 1) / 2
  m % 1000 = 625 := 
by
  let T := Finset.range 13
  let m := (3^12 - 2 * 2^12 + 1) / 2
  have : m % 1000 = 625 := sorry
  exact this

end disjoint_subsets_mod_1000_l182_182519


namespace gcf_and_multiples_l182_182274

theorem gcf_and_multiples (a b gcf : ℕ) : 
  (a = 90) → (b = 135) → gcd a b = gcf → 
  (gcf = 45) ∧ (45 % gcf = 0) ∧ (90 % gcf = 0) ∧ (135 % gcf = 0) := 
by
  intros ha hb hgcf
  rw [ha, hb] at hgcf
  sorry

end gcf_and_multiples_l182_182274


namespace cost_of_ice_cream_l182_182188

theorem cost_of_ice_cream 
  (meal_cost : ℕ)
  (number_of_people : ℕ)
  (total_money : ℕ)
  (total_cost : ℕ := meal_cost * number_of_people) 
  (remaining_money : ℕ := total_money - total_cost) 
  (ice_cream_cost_per_scoop : ℕ := remaining_money / number_of_people) :
  meal_cost = 10 ∧ number_of_people = 3 ∧ total_money = 45 →
  ice_cream_cost_per_scoop = 5 :=
by
  intros
  sorry

end cost_of_ice_cream_l182_182188


namespace total_teachers_in_all_departments_is_637_l182_182392

noncomputable def total_teachers : ℕ :=
  let major_departments := 9
  let minor_departments := 8
  let teachers_per_major := 45
  let teachers_per_minor := 29
  (major_departments * teachers_per_major) + (minor_departments * teachers_per_minor)

theorem total_teachers_in_all_departments_is_637 : total_teachers = 637 := 
  by
  sorry

end total_teachers_in_all_departments_is_637_l182_182392


namespace beginner_trigonometry_probability_l182_182225

def BC := ℝ
def AC := ℝ
def IC := ℝ
def BT := ℝ
def AT := ℝ
def IT := ℝ
def T := 5000

theorem beginner_trigonometry_probability :
  ∀ (BC AC IC BT AT IT : ℝ),
  (BC + AC + IC = 0.60 * T) →
  (BT + AT + IT = 0.40 * T) →
  (BC + BT = 0.45 * T) →
  (AC + AT = 0.35 * T) →
  (IC + IT = 0.20 * T) →
  (BC = 1.25 * BT) →
  (IC + AC = 1.20 * (IT + AT)) →
  (BT / T = 1/5) :=
by
  intros
  sorry

end beginner_trigonometry_probability_l182_182225


namespace club_additional_members_l182_182718

theorem club_additional_members (current_members : ℕ) (additional_members : ℕ) 
  (h1 : current_members = 10) 
  (h2 : additional_members = 5 + 2 * current_members - current_members) : 
  additional_members = 15 :=
by 
  rw [h1] at h2
  norm_num at h2
  exact h2

end club_additional_members_l182_182718


namespace club_additional_members_l182_182722

theorem club_additional_members (current_members : ℕ) (additional_members : ℕ) :
  current_members = 10 →
  additional_members = (2 * current_members) + 5 - current_members →
  additional_members = 15 :=
by
  intro h1 h2
  rw [h1] at h2
  norm_num at h2
  exact h2

end club_additional_members_l182_182722


namespace proof_x_plus_y_l182_182851

variables (x y : ℝ)

-- Definitions for the given conditions
def cond1 (x y : ℝ) : Prop := 2 * |x| + x + y = 18
def cond2 (x y : ℝ) : Prop := x + 2 * |y| - y = 14

theorem proof_x_plus_y (x y : ℝ) (h1 : cond1 x y) (h2 : cond2 x y) : x + y = 14 := by
  sorry

end proof_x_plus_y_l182_182851


namespace proof_problem_l182_182027

variables {a b c : ℝ}

theorem proof_problem (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : 4 * a^2 + b^2 + 16 * c^2 = 1) :
  (0 < a * b ∧ a * b < 1 / 4) ∧ (1 / a^2 + 1 / b^2 + 1 / (4 * a * b * c^2) > 49) :=
by
  sorry

end proof_problem_l182_182027


namespace common_difference_of_arithmetic_sequence_l182_182229

variable {a : ℕ → ℝ} (a2 a5 : ℝ)
variable (h1 : a 2 = 9) (h2 : a 5 = 33)

theorem common_difference_of_arithmetic_sequence :
  ∃ d : ℝ, ∀ n : ℕ, a n = a 1 + (n - 1) * d ∧ d = 8 := by
  sorry

end common_difference_of_arithmetic_sequence_l182_182229


namespace compare_neg_thirds_and_halves_l182_182114

theorem compare_neg_thirds_and_halves : (-1 : ℚ) / 3 > (-1 : ℚ) / 2 :=
by
  sorry

end compare_neg_thirds_and_halves_l182_182114


namespace sum_of_numbers_l182_182684

theorem sum_of_numbers (x y z : ℝ) (h1 : x ≤ y) (h2 : y ≤ z) 
  (h3 : y = 5) (h4 : (x + y + z) / 3 = x + 10) (h5 : (x + y + z) / 3 = z - 15) : 
  x + y + z = 30 := 
by 
  sorry

end sum_of_numbers_l182_182684


namespace gray_region_area_l182_182590

noncomputable def area_of_gray_region (C_center D_center : ℝ × ℝ) (C_radius D_radius : ℝ) :=
  let rect_area := 35
  let semicircle_C_area := (25 * Real.pi) / 2
  let quarter_circle_D_area := (16 * Real.pi) / 4
  rect_area - semicircle_C_area - quarter_circle_D_area

theorem gray_region_area :
  area_of_gray_region (5, 5) (12, 5) 5 4 = 35 - 16.5 * Real.pi :=
by
  simp [area_of_gray_region]
  sorry

end gray_region_area_l182_182590


namespace value_of_m_l182_182338

theorem value_of_m (m : ℝ) : (∀ x : ℝ, (x^2 + 2 * m * x + m > 3 / 16)) ↔ (1 / 4 < m ∧ m < 3 / 4) :=
by sorry

end value_of_m_l182_182338


namespace factorial_fraction_integer_l182_182534

open Nat

theorem factorial_fraction_integer (m n : ℕ) : 
  ∃ k : ℕ, k = (2 * m).factorial * (2 * n).factorial / (m.factorial * n.factorial * (m + n).factorial) := 
sorry

end factorial_fraction_integer_l182_182534


namespace intersection_A_B_l182_182209

def set_A (x : ℝ) : Prop := x^2 - 4 * x - 5 < 0
def set_B (x : ℝ) : Prop := 2 < x ∧ x < 4

theorem intersection_A_B (x : ℝ) :
  (set_A x ∧ set_B x) ↔ 2 < x ∧ x < 4 :=
by sorry

end intersection_A_B_l182_182209


namespace factor_100_minus_16y2_l182_182478

theorem factor_100_minus_16y2 (y : ℝ) : 100 - 16 * y^2 = 4 * (5 - 2 * y) * (5 + 2 * y) := 
by sorry

end factor_100_minus_16y2_l182_182478


namespace remainder_when_divided_by_x_minus_2_l182_182772

def p (x : ℤ) : ℤ := x^5 + x^3 + x + 3

theorem remainder_when_divided_by_x_minus_2 :
  p 2 = 45 :=
by
  sorry

end remainder_when_divided_by_x_minus_2_l182_182772


namespace total_stamps_is_38_l182_182103

-- Definitions based directly on conditions
def snowflake_stamps := 11
def truck_stamps := snowflake_stamps + 9
def rose_stamps := truck_stamps - 13
def total_stamps := snowflake_stamps + truck_stamps + rose_stamps

-- Statement to be proved
theorem total_stamps_is_38 : total_stamps = 38 := 
by 
  sorry

end total_stamps_is_38_l182_182103


namespace circumference_of_back_wheel_l182_182259

theorem circumference_of_back_wheel
  (C_f : ℝ) (C_b : ℝ) (D : ℝ) (N_b : ℝ)
  (h1 : C_f = 30)
  (h2 : D = 1650)
  (h3 : (N_b + 5) * C_f = D)
  (h4 : N_b * C_b = D) :
  C_b = 33 :=
sorry

end circumference_of_back_wheel_l182_182259


namespace number_of_stickers_after_losing_page_l182_182422

theorem number_of_stickers_after_losing_page (stickers_per_page : ℕ) (initial_pages : ℕ) (pages_lost : ℕ) :
  stickers_per_page = 20 → initial_pages = 12 → pages_lost = 1 → (initial_pages - pages_lost) * stickers_per_page = 220 :=
by
  intros
  sorry

end number_of_stickers_after_losing_page_l182_182422


namespace cost_of_first_20_kgs_l182_182880

theorem cost_of_first_20_kgs 
  (l m n : ℕ) 
  (hl1 : 30 * l +  3 * m = 333) 
  (hl2 : 30 * l +  6 * m = 366) 
  (hl3 : 30 * l + 15 * m = 465) 
  (hl4 : 30 * l + 20 * m = 525) 
  : 20 * l = 200 :=
by
  sorry

end cost_of_first_20_kgs_l182_182880


namespace system_solution_l182_182813

theorem system_solution :
  ∃ x y : ℝ, (3 * x + y = 11 ∧ x - y = 1) ∧ (x = 3 ∧ y = 2) := 
by
  sorry

end system_solution_l182_182813


namespace a_in_s_l182_182360

-- Defining the sets and the condition
def S : Set ℕ := {1, 2}
def T (a : ℕ) : Set ℕ := {a}

-- The Lean theorem statement
theorem a_in_s (a : ℕ) (h : S ∪ T a = S) : a = 1 ∨ a = 2 := 
by 
  sorry

end a_in_s_l182_182360


namespace find_m_of_quadratic_fn_l182_182547

theorem find_m_of_quadratic_fn (m : ℚ) (h : 2 * m - 1 = 2) : m = 3 / 2 :=
by
  sorry

end find_m_of_quadratic_fn_l182_182547


namespace range_of_x_satisfying_inequality_l182_182687

theorem range_of_x_satisfying_inequality (x : ℝ) : x^2 < |x| ↔ (x > -1 ∧ x < 0) ∨ (x > 0 ∧ x < 1) :=
by
  sorry

end range_of_x_satisfying_inequality_l182_182687


namespace sufficient_food_supply_l182_182313

variable {L S : ℝ}

theorem sufficient_food_supply (h1 : L + 4 * S = 14) (h2 : L > S) : L + 3 * S ≥ 11 :=
by
  sorry

end sufficient_food_supply_l182_182313


namespace tank_capacity_is_24_l182_182730

noncomputable def tank_capacity_proof : Prop :=
  ∃ (C : ℝ), (∃ (v : ℝ), (v / C = 1 / 6) ∧ ((v + 4) / C = 1 / 3)) ∧ C = 24

theorem tank_capacity_is_24 : tank_capacity_proof := sorry

end tank_capacity_is_24_l182_182730


namespace compute_f_at_5_l182_182215

def f : ℝ → ℝ := sorry

axiom f_property : ∀ x : ℝ, f (10 ^ x) = x

theorem compute_f_at_5 : f 5 = Real.log 5 / Real.log 10 :=
by
  sorry

end compute_f_at_5_l182_182215


namespace outliers_in_data_set_l182_182541

-- Define the data set
def dataSet : List ℕ := [6, 19, 33, 33, 39, 41, 41, 43, 51, 57]

-- Define the given quartiles
def Q1 : ℕ := 33
def Q3 : ℕ := 43

-- Define the interquartile range
def IQR : ℕ := Q3 - Q1

-- Define the outlier thresholds
def lowerOutlierThreshold : ℕ := Q1 - 3 / 2 * IQR
def upperOutlierThreshold : ℕ := Q3 + 3 / 2 * IQR

-- Define what it means to be an outlier
def isOutlier (x : ℕ) : Bool :=
  x < lowerOutlierThreshold ∨ x > upperOutlierThreshold

-- Count the number of outliers in the data set
def countOutliers (data : List ℕ) : ℕ :=
  (data.filter isOutlier).length

theorem outliers_in_data_set :
  countOutliers dataSet = 1 :=
by
  sorry

end outliers_in_data_set_l182_182541


namespace apples_needed_l182_182652

-- Define a simple equivalence relation between the weights of oranges and apples.
def weight_equivalent (oranges apples : ℕ) : Prop :=
  8 * apples = 6 * oranges
  
-- State the main theorem based on the given conditions
theorem apples_needed (oranges_count : ℕ) (h : weight_equivalent 1 1) : oranges_count = 32 → ∃ apples_count, apples_count = 24 :=
by
  sorry

end apples_needed_l182_182652


namespace value_of_x_when_y_is_six_l182_182081

theorem value_of_x_when_y_is_six 
  (k : ℝ) -- The constant of variation
  (h1 : ∀ y : ℝ, x = k / y^2) -- The inverse relationship
  (h2 : y = 2)
  (h3 : x = 1)
  : x = 1 / 9 :=
by
  sorry

end value_of_x_when_y_is_six_l182_182081


namespace daily_increase_in_weaving_l182_182711

open Nat

theorem daily_increase_in_weaving :
  let d := 16 / 29
  (30 : ℝ) * 10 + (30 * 29 / 2 : ℝ) * d = 600 :=
by
  let d := 16 / 29
  have h1 : (30 : ℝ) * 10 + (30 * 29 / 2 : ℝ) * d = 600 := sorry
  exact h1

end daily_increase_in_weaving_l182_182711


namespace stuffed_dogs_count_l182_182183

theorem stuffed_dogs_count (D : ℕ) (h1 : 14 + D % 7 = 0) : D = 7 :=
by {
  sorry
}

end stuffed_dogs_count_l182_182183


namespace cement_bought_l182_182624

-- Define the three conditions given in the problem
def original_cement : ℕ := 98
def son_contribution : ℕ := 137
def total_cement : ℕ := 450

-- Using those conditions, state that the amount of cement he bought is 215 lbs
theorem cement_bought :
  original_cement + son_contribution = 235 ∧ total_cement - (original_cement + son_contribution) = 215 := 
by {
  sorry
}

end cement_bought_l182_182624


namespace find_f7_l182_182204

noncomputable def f : ℝ → ℝ := sorry

-- The conditions provided in the problem
axiom odd_function : ∀ x : ℝ, f (-x) = -f x
axiom periodic_function : ∀ x : ℝ, f (x + 4) = f x
axiom function_in_interval : ∀ x, 0 < x ∧ x < 2 → f x = 2 * x^2

-- The final proof goal
theorem find_f7 : f 7 = -2 :=
by sorry

end find_f7_l182_182204


namespace intersection_of_A_and_B_l182_182486

open Set Int

def A : Set ℝ := { x | x ^ 2 - 6 * x + 8 ≤ 0 }
def B : Set ℤ := { x | abs (x - 3) < 2 }

theorem intersection_of_A_and_B :
  (A ∩ (coe '' B) = { x : ℝ | x = 2 ∨ x = 3 ∨ x = 4 }) :=
by
  sorry

end intersection_of_A_and_B_l182_182486


namespace arrangement_count_l182_182697

theorem arrangement_count (n : ℕ) (hn : n = 9) (p1 p2 p3 p4 : Finset ℕ) (hp1 : p1 = {1, 2}) (hp2 : p2 = {3, 4}) :
let book_arrangements := (n - 2)! * 4 in
book_arrangements = 4 * (n - 2)! :=
by
  sorry

end arrangement_count_l182_182697


namespace solve_quadratic_l182_182044

theorem solve_quadratic :
  ∀ x, (x^2 - x - 12 = 0) → (x = -3 ∨ x = 4) :=
by
  intro x
  intro h
  sorry

end solve_quadratic_l182_182044


namespace problem_2003_divisibility_l182_182673

theorem problem_2003_divisibility :
  let N := (List.range' 1 1001).prod + (List.range' 1002 1001).prod
  N % 2003 = 0 := by
  sorry

end problem_2003_divisibility_l182_182673


namespace solution_set_f_neg_x_l182_182619

noncomputable def f (a b x : ℝ) : ℝ := (a * x - 1) * (x + b)

theorem solution_set_f_neg_x (a b : ℝ) (h : ∀ x, f a b x > 0 ↔ -1 < x ∧ x < 3) :
  ∀ x, f a b (-x) < 0 ↔ (x < -3 ∨ x > 1) :=
by
  intro x
  specialize h (-x)
  sorry

end solution_set_f_neg_x_l182_182619


namespace rod_length_difference_l182_182297

theorem rod_length_difference (L₁ L₂ : ℝ) (h1 : L₁ + L₂ = 33)
    (h2 : (∀ x : ℝ, x = (2 / 3) * L₁ ∧ x = (4 / 5) * L₂)) :
    abs (L₁ - L₂) = 3 := by
  sorry

end rod_length_difference_l182_182297


namespace part1_part2_l182_182026

variables (a b c : ℝ)
-- Assuming a, b, c are positive and satisfy the given equation
variable (h1 : 0 < a)
variable (h2 : 0 < b)
variable (h3 : 0 < c)
variable (h_eq : 4 * a ^ 2 + b ^ 2 + 16 * c ^ 2 = 1)

-- Statement for the first part: 0 < ab < 1/4
theorem part1 : 0 < a * b ∧ a * b < 1 / 4 :=
  sorry

-- Statement for the second part: 1/a² + 1/b² + 1/(4abc²) > 49
theorem part2 : 1 / (a ^ 2) + 1 / (b ^ 2) + 1 / (4 * a * b * c ^ 2) > 49 :=
  sorry

end part1_part2_l182_182026


namespace max_value_of_largest_integer_l182_182080

theorem max_value_of_largest_integer (a1 a2 a3 a4 a5 a6 a7 : ℕ) (h1 : a1 + a2 + a3 + a4 + a5 + a6 + a7 = 560) (h2 : a7 - a1 = 20) : a7 ≤ 21 :=
sorry

end max_value_of_largest_integer_l182_182080


namespace sufficient_food_supply_l182_182311

variable {L S : ℝ}

theorem sufficient_food_supply (h1 : L + 4 * S = 14) (h2 : L > S) : L + 3 * S ≥ 11 :=
by
  sorry

end sufficient_food_supply_l182_182311


namespace intersection_eq_T_l182_182987

def S : Set ℤ := {s | ∃ n : ℤ, s = 2 * n + 1}
def T : Set ℤ := {t | ∃ n : ℤ, t = 4 * n + 1}

theorem intersection_eq_T : S ∩ T = T := sorry

end intersection_eq_T_l182_182987


namespace mark_donates_cans_l182_182243

-- Definitions coming directly from the conditions
def num_shelters : ℕ := 6
def people_per_shelter : ℕ := 30
def cans_per_person : ℕ := 10

-- The final statement to be proven
theorem mark_donates_cans : (num_shelters * people_per_shelter * cans_per_person) = 1800 :=
by sorry

end mark_donates_cans_l182_182243


namespace count_squares_below_line_l182_182261

theorem count_squares_below_line (units : ℕ) :
  let intercept_x := 221;
  let intercept_y := 7;
  let total_squares := intercept_x * intercept_y;
  let diagonal_squares := intercept_x - 1 + intercept_y - 1 + 1; 
  let non_diag_squares := total_squares - diagonal_squares;
  let below_line := non_diag_squares / 2;
  below_line = 660 :=
by
  sorry

end count_squares_below_line_l182_182261


namespace cost_of_ice_cream_l182_182189

theorem cost_of_ice_cream 
  (meal_cost : ℕ)
  (number_of_people : ℕ)
  (total_money : ℕ)
  (total_cost : ℕ := meal_cost * number_of_people) 
  (remaining_money : ℕ := total_money - total_cost) 
  (ice_cream_cost_per_scoop : ℕ := remaining_money / number_of_people) :
  meal_cost = 10 ∧ number_of_people = 3 ∧ total_money = 45 →
  ice_cream_cost_per_scoop = 5 :=
by
  intros
  sorry

end cost_of_ice_cream_l182_182189


namespace ned_initial_lives_l182_182565

variable (lost_lives : ℕ) (current_lives : ℕ) 
variable (initial_lives : ℕ)

theorem ned_initial_lives (h_lost: lost_lives = 13) (h_current: current_lives = 70) :
  initial_lives = current_lives + lost_lives := by
  sorry

end ned_initial_lives_l182_182565


namespace dollar_triple_60_l182_182911

-- Define the function $N
def dollar (N : Real) : Real :=
  0.4 * N + 2

-- Proposition proving that $$(($60)) = 6.96
theorem dollar_triple_60 : dollar (dollar (dollar 60)) = 6.96 := by
  sorry

end dollar_triple_60_l182_182911


namespace gum_needed_l182_182518

-- Definitions based on problem conditions
def num_cousins : ℕ := 4
def gum_per_cousin : ℕ := 5

-- Proposition that we need to prove
theorem gum_needed : num_cousins * gum_per_cousin = 20 := by
  sorry

end gum_needed_l182_182518


namespace prove_non_negative_axbycz_l182_182496

variable {a b c x y z : ℝ}

theorem prove_non_negative_axbycz
  (h1 : (a + b + c) * (x + y + z) = 3)
  (h2 : (a^2 + b^2 + c^2) * (x^2 + y^2 + z^2) = 4) :
  a * x + b * y + c * z ≥ 0 := 
sorry

end prove_non_negative_axbycz_l182_182496


namespace exponentiation_identity_l182_182106

theorem exponentiation_identity (x : ℝ) : (-x^7)^4 = x^28 := 
sorry

end exponentiation_identity_l182_182106


namespace range_function_1_l182_182493

theorem range_function_1 (y : ℝ) : 
  (∃ x : ℝ, x ≥ -1 ∧ y = (1/3) ^ x) ↔ (0 < y ∧ y ≤ 3) :=
sorry

end range_function_1_l182_182493


namespace customer_difference_l182_182292

theorem customer_difference (before after : ℕ) (h1 : before = 19) (h2 : after = 4) : before - after = 15 :=
by
  sorry

end customer_difference_l182_182292


namespace amount_paid_l182_182525

def hamburger_cost : ℕ := 4
def onion_rings_cost : ℕ := 2
def smoothie_cost : ℕ := 3
def change_received : ℕ := 11

theorem amount_paid (h_cost : ℕ := hamburger_cost) (o_cost : ℕ := onion_rings_cost) (s_cost : ℕ := smoothie_cost) (change : ℕ := change_received) :
  h_cost + o_cost + s_cost + change = 20 := by
  sorry

end amount_paid_l182_182525


namespace fraction_area_of_shaded_square_in_larger_square_is_one_eighth_l182_182245

theorem fraction_area_of_shaded_square_in_larger_square_is_one_eighth :
  let side_larger_square := 4
  let area_larger_square := side_larger_square^2
  let side_shaded_square := Real.sqrt (1^2 + 1^2)
  let area_shaded_square := side_shaded_square^2
  area_shaded_square / area_larger_square = 1 / 8 := 
by 
  sorry

end fraction_area_of_shaded_square_in_larger_square_is_one_eighth_l182_182245


namespace shirley_eggs_start_l182_182672

theorem shirley_eggs_start (eggs_end : ℕ) (eggs_bought : ℕ) (eggs_start : ℕ) (h_end : eggs_end = 106) (h_bought : eggs_bought = 8) :
  eggs_start = eggs_end - eggs_bought → eggs_start = 98 :=
by
  intros h_start
  rw [h_end, h_bought] at h_start
  exact h_start

end shirley_eggs_start_l182_182672


namespace cos_90_eq_zero_l182_182162

def point_after_rotation (θ : ℝ) : ℝ × ℝ :=
  let x := cos θ
  let y := sin θ
  (x, y)

theorem cos_90_eq_zero : cos (real.pi / 2) = 0 :=
  sorry

end cos_90_eq_zero_l182_182162


namespace number_of_orange_ribbons_l182_182816

/-- Define the total number of ribbons -/
def total_ribbons (yellow purple orange black total : ℕ) : Prop :=
  yellow + purple + orange + black = total

/-- Define the fractions -/
def fractions (total_ribbons yellow purple orange black : ℕ) : Prop :=
  yellow = total_ribbons / 4 ∧ purple = total_ribbons / 3 ∧ orange = total_ribbons / 12 ∧ black = 40

/-- Define the black ribbons fraction -/
def black_fraction (total_ribbons : ℕ) : Prop :=
  40 = total_ribbons / 3

theorem number_of_orange_ribbons :
  ∃ (total : ℕ), total_ribbons (total / 4) (total / 3) (total / 12) 40 total ∧ black_fraction total ∧ (total / 12 = 10) :=
by
  sorry

end number_of_orange_ribbons_l182_182816


namespace intersection_S_T_eq_T_l182_182946

def S : Set ℤ := { s | ∃ n : ℤ, s = 2 * n + 1 }
def T : Set ℤ := { t | ∃ n : ℤ, t = 4 * n + 1 }

theorem intersection_S_T_eq_T : S ∩ T = T := by
  sorry

end intersection_S_T_eq_T_l182_182946


namespace base8_to_base10_l182_182676

theorem base8_to_base10 {a b : ℕ} (h1 : 3 * 64 + 7 * 8 + 4 = 252) (h2 : 252 = a * 10 + b) :
  (a + b : ℝ) / 20 = 0.35 :=
sorry

end base8_to_base10_l182_182676


namespace exists_integer_n_l182_182844

theorem exists_integer_n (k : ℕ) (hk : 0 < k) : 
  ∃ n : ℤ, (n + 1981^k)^(1/2 : ℝ) + (n : ℝ)^(1/2 : ℝ) = (1982^(1/2 : ℝ) + 1) ^ k :=
sorry

end exists_integer_n_l182_182844


namespace cat_food_sufficiency_l182_182310

variable (L S : ℝ)

theorem cat_food_sufficiency (h : L + 4 * S = 14) : L + 3 * S >= 11 :=
sorry

end cat_food_sufficiency_l182_182310


namespace minimum_value_of_fraction_l182_182795

variable {a b : ℝ}

theorem minimum_value_of_fraction (h1 : a > b) (h2 : a * b = 1) : 
  ∃ (c : ℝ), c = 2 * Real.sqrt 2 ∧ ∀ x > b, a * x = 1 -> 
  (x - b + 2 / (x - b) ≥ c) :=
by
  sorry

end minimum_value_of_fraction_l182_182795


namespace evaluate_dollar_l182_182470

variable {R : Type} [Field R]

def dollar (a b : R) : R := (a - b) ^ 2

theorem evaluate_dollar (x y : R) : dollar (2 * x + 3 * y) (3 * x - 4 * y) = x ^ 2 - 14 * x * y + 49 * y ^ 2 := by
  sorry

end evaluate_dollar_l182_182470


namespace sufficient_food_l182_182328

-- Definitions for a large package and a small package as amounts of food lasting days
def largePackageDay : ℕ := L
def smallPackageDay : ℕ := S

-- Condition: One large package and four small packages last 14 days
axiom condition : largePackageDay + 4 * smallPackageDay = 14

-- Theorem to be proved: One large package and three small packages last at least 11 days
theorem sufficient_food : largePackageDay + 3 * smallPackageDay ≥ 11 :=
by
  sorry

end sufficient_food_l182_182328


namespace factorization_x8_minus_81_l182_182331

theorem factorization_x8_minus_81 (x : ℝ) : 
  x^8 - 81 = (x^4 + 9) * (x^2 + 3) * (x + real.sqrt 3) * (x - real.sqrt 3) :=
by
  sorry

end factorization_x8_minus_81_l182_182331


namespace S_inter_T_eq_T_l182_182968

-- Define the sets S and T
def S := { s : ℤ | ∃ n : ℤ, s = 2 * n + 1 }
def T := { t : ℤ | ∃ n : ℤ, t = 4 * n + 1 }

-- Prove that S ∩ T = T
theorem S_inter_T_eq_T : S ∩ T = T := 
by
  sorry

end S_inter_T_eq_T_l182_182968


namespace coffee_shop_spending_l182_182923

variable (R S : ℝ)

theorem coffee_shop_spending (h1 : S = 0.60 * R) (h2 : R = S + 12.50) : R + S = 50 :=
by
  sorry

end coffee_shop_spending_l182_182923


namespace compound_interest_rate_l182_182414

theorem compound_interest_rate (SI CI : ℝ) (P1 P2 : ℝ) (T1 T2 : ℝ) (R1 : ℝ) (R : ℝ) 
    (H1 : SI = (P1 * R1 * T1) / 100)
    (H2 : CI = 2 * SI)
    (H3 : CI = P2 * ((1 + R/100)^2 - 1))
    (H4 : P1 = 1272)
    (H5 : P2 = 5000)
    (H6 : T1 = 5)
    (H7 : T2 = 2)
    (H8 : R1 = 10) :
  R = 12 :=
by
  sorry

end compound_interest_rate_l182_182414


namespace point_in_third_quadrant_l182_182812

theorem point_in_third_quadrant (m : ℝ) (h1 : m < 0) (h2 : 4 + 2 * m < 0) : m < -2 := by
  sorry

end point_in_third_quadrant_l182_182812


namespace correct_division_result_l182_182563

theorem correct_division_result (x : ℝ) (h : 4 * x = 166.08) : x / 4 = 10.38 :=
by
  sorry

end correct_division_result_l182_182563


namespace taxi_ride_total_cost_l182_182291

theorem taxi_ride_total_cost :
  let base_fee := 1.50
  let cost_per_mile := 0.25
  let distance1 := 5
  let distance2 := 8
  let distance3 := 3
  let cost1 := base_fee + distance1 * cost_per_mile
  let cost2 := base_fee + distance2 * cost_per_mile
  let cost3 := base_fee + distance3 * cost_per_mile
  cost1 + cost2 + cost3 = 8.50 := sorry

end taxi_ride_total_cost_l182_182291


namespace total_spent_is_correct_l182_182654

-- Declare the constants for the prices and quantities
def wallet_cost : ℕ := 50
def sneakers_cost_per_pair : ℕ := 100
def sneakers_pairs : ℕ := 2
def backpack_cost : ℕ := 100
def jeans_cost_per_pair : ℕ := 50
def jeans_pairs : ℕ := 2

-- Define the total amounts spent by Leonard and Michael
def leonard_total : ℕ := wallet_cost + sneakers_cost_per_pair * sneakers_pairs
def michael_total : ℕ := backpack_cost + jeans_cost_per_pair * jeans_pairs

-- The total amount spent by Leonard and Michael
def total_spent : ℕ := leonard_total + michael_total

-- The proof statement
theorem total_spent_is_correct : total_spent = 450 :=
by 
  -- This part is where the proof would go
  sorry

end total_spent_is_correct_l182_182654


namespace prime_even_intersection_l182_182378

-- Define P as the set of prime numbers
def is_prime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def P : Set ℕ := { n | is_prime n }

-- Define Q as the set of even numbers
def Q : Set ℕ := { n | n % 2 = 0 }

-- Statement to prove
theorem prime_even_intersection : P ∩ Q = {2} :=
by
  sorry

end prime_even_intersection_l182_182378


namespace donation_total_is_correct_l182_182841

-- Definitions and conditions
def Megan_inheritance : ℤ := 1000000
def Dan_inheritance : ℤ := 10000
def donation_percentage : ℚ := 0.1
def Megan_donation := Megan_inheritance * donation_percentage
def Dan_donation := Dan_inheritance * donation_percentage
def total_donation := Megan_donation + Dan_donation

-- Theorem statement
theorem donation_total_is_correct : total_donation = 101000 := by
  sorry

end donation_total_is_correct_l182_182841


namespace poly_real_coeff_l182_182858

noncomputable def polynomial_g (p q r s : ℝ) : (ℝ[X]) :=
  X^4 + C p * X^3 + C q * X^2 + C r * X + C s

theorem poly_real_coeff (p q r s : ℝ) 
  (h1 : polynomial_g p q r s.eval (3 * Complex.i) = 0)
  (h2 : polynomial_g p q r s.eval (1 + 2 * Complex.i) = 0) :
  p + q + r + s = 39 := 
sorry

end poly_real_coeff_l182_182858


namespace cos_90_eq_0_l182_182156

theorem cos_90_eq_0 : Real.cos (90 * Real.pi / 180) = 0 := by
  sorry

end cos_90_eq_0_l182_182156


namespace number_of_stickers_after_losing_page_l182_182420

theorem number_of_stickers_after_losing_page (stickers_per_page : ℕ) (initial_pages : ℕ) (pages_lost : ℕ) :
  stickers_per_page = 20 → initial_pages = 12 → pages_lost = 1 → (initial_pages - pages_lost) * stickers_per_page = 220 :=
by
  intros
  sorry

end number_of_stickers_after_losing_page_l182_182420


namespace alma_carrots_leftover_l182_182583

/-- Alma has 47 baby carrots and wishes to distribute them equally among 4 goats.
    We need to prove that the number of leftover carrots after such distribution is 3. -/
theorem alma_carrots_leftover (total_carrots : ℕ) (goats : ℕ) (leftover : ℕ) 
  (h1 : total_carrots = 47) (h2 : goats = 4) (h3 : leftover = total_carrots % goats) : 
  leftover = 3 :=
by
  sorry

end alma_carrots_leftover_l182_182583


namespace du_chin_remaining_money_l182_182037

noncomputable def du_chin_revenue_over_week : ℝ := 
  let day0_revenue := 200 * 20
  let day0_cost := 3 / 5 * day0_revenue
  let day0_remaining := day0_revenue - day0_cost

  let day1_revenue := day0_remaining * 1.10
  let day1_cost := day0_cost * 1.10
  let day1_remaining := day1_revenue - day1_cost

  let day2_revenue := day1_remaining * 0.95
  let day2_cost := day1_cost * 0.90
  let day2_remaining := day2_revenue - day2_cost

  let day3_revenue := day2_remaining
  let day3_cost := day2_cost
  let day3_remaining := day3_revenue - day3_cost

  let day4_revenue := day3_remaining * 1.15
  let day4_cost := day3_cost * 1.05
  let day4_remaining := day4_revenue - day4_cost

  let day5_revenue := day4_remaining * 0.92
  let day5_cost := day4_cost * 0.95
  let day5_remaining := day5_revenue - day5_cost

  let day6_revenue := day5_remaining * 1.05
  let day6_cost := day5_cost
  let day6_remaining := day6_revenue - day6_cost

  day0_remaining + day1_remaining + day2_remaining + day3_remaining + day4_remaining + day5_remaining + day6_remaining

theorem du_chin_remaining_money : du_chin_revenue_over_week = 13589.08 := 
  sorry

end du_chin_remaining_money_l182_182037


namespace profit_share_difference_l182_182277

noncomputable def ratio (x y : ℕ) : ℕ := x / Nat.gcd x y

def capital_A : ℕ := 8000
def capital_B : ℕ := 10000
def capital_C : ℕ := 12000
def profit_share_B : ℕ := 1900
def total_parts : ℕ := 15  -- Sum of the ratio parts (4 for A, 5 for B, 6 for C)
def part_amount : ℕ := profit_share_B / 5  -- 5 parts of B

def profit_share_A : ℕ := 4 * part_amount
def profit_share_C : ℕ := 6 * part_amount

theorem profit_share_difference :
  (profit_share_C - profit_share_A) = 760 := by
  sorry

end profit_share_difference_l182_182277


namespace sin_double_angle_shift_l182_182926

variable (θ : Real)

theorem sin_double_angle_shift (h : Real.cos (θ + Real.pi) = -1 / 3) :
  Real.sin (2 * θ + Real.pi / 2) = -7 / 9 := 
by 
  sorry

end sin_double_angle_shift_l182_182926


namespace combined_marble_remainder_l182_182874

theorem combined_marble_remainder (l j : ℕ) (h_l : l % 8 = 5) (h_j : j % 8 = 6) : (l + j) % 8 = 3 := by
  sorry

end combined_marble_remainder_l182_182874


namespace num_positive_int_values_l182_182779

theorem num_positive_int_values (N : ℕ) :
  (∃ m : ℕ, N = m ∧ m > 0 ∧ 48 % (m + 3) = 0) ↔ (N < 7) :=
sorry

end num_positive_int_values_l182_182779


namespace solve_for_x_l182_182820

theorem solve_for_x 
  (a b c d x y z w : ℝ) 
  (H1 : x + y + z + w = 360)
  (H2 : a = x + y / 2) 
  (H3 : b = y + z / 2) 
  (H4 : c = z + w / 2) 
  (H5 : d = w + x / 2) : 
  x = (16 / 15) * (a - b / 2 + c / 4 - d / 8) :=
sorry


end solve_for_x_l182_182820


namespace total_stamps_is_38_l182_182102

-- Definitions based directly on conditions
def snowflake_stamps := 11
def truck_stamps := snowflake_stamps + 9
def rose_stamps := truck_stamps - 13
def total_stamps := snowflake_stamps + truck_stamps + rose_stamps

-- Statement to be proved
theorem total_stamps_is_38 : total_stamps = 38 := 
by 
  sorry

end total_stamps_is_38_l182_182102


namespace inequality_holds_for_all_x_l182_182480

theorem inequality_holds_for_all_x (x : ℝ) : 3 * x^2 + 9 * x ≥ -12 := by
  sorry

end inequality_holds_for_all_x_l182_182480


namespace sum_powers_divisible_by_5_iff_l182_182843

theorem sum_powers_divisible_by_5_iff (n : ℕ) (h_pos : n > 0) :
  (1^n + 2^n + 3^n + 4^n) % 5 = 0 ↔ n % 4 ≠ 0 := 
sorry

end sum_powers_divisible_by_5_iff_l182_182843


namespace pie_eaten_after_four_trips_l182_182876

theorem pie_eaten_after_four_trips : 
  let trip1 := (1 / 3 : ℝ)
  let trip2 := (1 / 3^2 : ℝ)
  let trip3 := (1 / 3^3 : ℝ)
  let trip4 := (1 / 3^4 : ℝ)
  trip1 + trip2 + trip3 + trip4 = (40 / 81 : ℝ) :=
by
  sorry

end pie_eaten_after_four_trips_l182_182876


namespace true_proposition_among_provided_l182_182586

theorem true_proposition_among_provided :
  ∃ (x0 : ℝ), |x0| ≤ 0 :=
by
  exists 0
  simp

end true_proposition_among_provided_l182_182586


namespace negation_of_universal_statement_l182_182856

theorem negation_of_universal_statement :
  (¬ ∀ x : ℝ, x^2 ≥ 0) ↔ ∃ x : ℝ, x^2 < 0 :=
by
  sorry

end negation_of_universal_statement_l182_182856


namespace cat_food_sufficiency_l182_182318

theorem cat_food_sufficiency (L S : ℝ) (h : L + 4 * S = 14) : L + 3 * S ≥ 11 :=
sorry

end cat_food_sufficiency_l182_182318


namespace distance_A_to_B_l182_182739

theorem distance_A_to_B : 
  ∀ (D : ℕ),
    let boat_speed_with_wind := 21
    let boat_speed_against_wind := 17
    let time_for_round_trip := 7
    let stream_speed_ab := 3
    let stream_speed_ba := 2
    let effective_speed_ab := boat_speed_with_wind + stream_speed_ab
    let effective_speed_ba := boat_speed_against_wind - stream_speed_ba
    D / effective_speed_ab + D / effective_speed_ba = time_for_round_trip →
    D = 65 :=
by
  sorry

end distance_A_to_B_l182_182739


namespace intersection_eq_T_l182_182941

def S : Set ℤ := {s | ∃ n : ℤ, s = 2 * n + 1}
def T : Set ℤ := {t | ∃ n : ℤ, t = 4 * n + 1}

theorem intersection_eq_T : S ∩ T = T := 
sorry

end intersection_eq_T_l182_182941


namespace solve_equation_nat_numbers_l182_182710

theorem solve_equation_nat_numbers (a b : ℕ) (h : (a, b) = (11, 170) ∨ (a, b) = (22, 158) ∨ (a, b) = (33, 146) ∨
                                    (a, b) = (44, 134) ∨ (a, b) = (55, 122) ∨ (a, b) = (66, 110) ∨
                                    (a, b) = (77, 98) ∨ (a, b) = (88, 86) ∨ (a, b) = (99, 74) ∨
                                    (a, b) = (110, 62) ∨ (a, b) = (121, 50) ∨ (a, b) = (132, 38) ∨
                                    (a, b) = (143, 26) ∨ (a, b) = (154, 14) ∨ (a, b) = (165, 2)) :
  12 * a + 11 * b = 2002 :=
by
  sorry

end solve_equation_nat_numbers_l182_182710


namespace percentage_loss_l182_182892

theorem percentage_loss (selling_price_with_loss : ℝ)
    (desired_selling_price_for_profit : ℝ)
    (profit_percentage : ℝ) (actual_selling_price : ℝ)
    (calculated_loss_percentage : ℝ) :
    selling_price_with_loss = 16 →
    desired_selling_price_for_profit = 21.818181818181817 →
    profit_percentage = 20 →
    actual_selling_price = 18.181818181818182 →
    calculated_loss_percentage = 12 → 
    calculated_loss_percentage = (actual_selling_price - selling_price_with_loss) / actual_selling_price * 100 := 
sorry

end percentage_loss_l182_182892


namespace intersection_S_T_eq_T_l182_182945

def S : Set ℤ := { s | ∃ n : ℤ, s = 2 * n + 1 }
def T : Set ℤ := { t | ∃ n : ℤ, t = 4 * n + 1 }

theorem intersection_S_T_eq_T : S ∩ T = T := by
  sorry

end intersection_S_T_eq_T_l182_182945


namespace jonah_profit_is_correct_l182_182377

noncomputable def jonah_profit : Real :=
  let pineapples := 6
  let pricePerPineapple := 3
  let pineappleCostWithoutDiscount := pineapples * pricePerPineapple
  let discount := if pineapples > 4 then 0.20 * pineappleCostWithoutDiscount else 0
  let totalCostAfterDiscount := pineappleCostWithoutDiscount - discount
  let ringsPerPineapple := 10
  let totalRings := pineapples * ringsPerPineapple
  let ringsSoldIndividually := 2
  let pricePerIndividualRing := 5
  let revenueFromIndividualRings := ringsSoldIndividually * pricePerIndividualRing
  let ringsLeft := totalRings - ringsSoldIndividually
  let ringsPerSet := 4
  let setsSold := ringsLeft / ringsPerSet -- This should be interpreted as an integer division
  let pricePerSet := 16
  let revenueFromSets := setsSold * pricePerSet
  let totalRevenue := revenueFromIndividualRings + revenueFromSets
  let profit := totalRevenue - totalCostAfterDiscount
  profit
  
theorem jonah_profit_is_correct :
  jonah_profit = 219.60 := by
  sorry

end jonah_profit_is_correct_l182_182377


namespace integer_count_n_l182_182804

theorem integer_count_n (n : ℤ) (H1 : n % 3 = 0) (H2 : 3 * n ≥ 1) (H3 : 3 * n ≤ 1000) : 
  ∃ k : ℕ, k = 111 := by
  sorry

end integer_count_n_l182_182804


namespace original_number_is_neg2_l182_182696

theorem original_number_is_neg2 (x : ℚ) (h : 2 - 1/x = 5/2) : x = -2 :=
sorry

end original_number_is_neg2_l182_182696


namespace no_such_arrangement_exists_l182_182822

theorem no_such_arrangement_exists :
  ¬ ∃ (f : ℕ → ℕ) (c : ℕ), 
    (∀ n, 1 ≤ f n ∧ f n ≤ 1331) ∧
    (∀ x y z, f (x + 11 * y + 121 * z) = c → f ((x+1) + 11 * y + 121 * z) = c + 8) ∧
    (∀ x y z, f (x + 11 * y + 121 * z) = c → f (x + 11 * (y+1) + 121 * z) = c + 9) :=
sorry

end no_such_arrangement_exists_l182_182822


namespace cos_90_deg_eq_zero_l182_182138

noncomputable def cos_90_degrees : ℝ :=
cos (90 * real.pi / 180)

theorem cos_90_deg_eq_zero : cos_90_degrees = 0 := 
by
  -- Definitions related to the conditions
  let point_0 := (1 : ℝ, 0 : ℝ)
  let point_90 := (0 : ℝ, 1 : ℝ)

  -- Rotating the point (1, 0) by 90 degrees counterclockwise results in the point (0, 1)
  have h_rotate_90 : (real.cos (90 * real.pi / 180), real.sin (90 * real.pi / 180)) = point_90 :=
    by
      -- Rotation logic is abstracted out in this property of cos and sin
      have h_cos_90 : real.cos (real.pi / 2) = 0 := by sorry
      have h_sin_90 : real.sin (real.pi / 2) = 1 := by sorry
      exact ⟨h_cos_90, h_sin_90⟩

  -- By definition, cos 90 degrees is the x-coordinate of the rotated point
  show cos_90_degrees = 0,
  from congr_arg Prod.fst h_rotate_90

end cos_90_deg_eq_zero_l182_182138


namespace intersection_eq_T_l182_182995

def S : Set ℤ := {s | ∃ n : ℤ, s = 2 * n + 1}
def T : Set ℤ := {t | ∃ n : ℤ, t = 4 * n + 1}

theorem intersection_eq_T : S ∩ T = T := sorry

end intersection_eq_T_l182_182995


namespace Wendy_earned_45_points_l182_182698

-- Definitions for the conditions
def points_per_bag : Nat := 5
def total_bags : Nat := 11
def unrecycled_bags : Nat := 2

-- The variable for recycled bags and total points earned
def recycled_bags := total_bags - unrecycled_bags
def total_points := recycled_bags * points_per_bag

theorem Wendy_earned_45_points : total_points = 45 :=
by
  -- Proof goes here
  sorry

end Wendy_earned_45_points_l182_182698


namespace square_traffic_sign_perimeter_l182_182742

-- Define the side length of the square
def side_length : ℕ := 4

-- Define the number of sides of the square
def number_of_sides : ℕ := 4

-- Define the perimeter of the square
def perimeter (l : ℕ) (n : ℕ) : ℕ := l * n

-- The theorem to be proved
theorem square_traffic_sign_perimeter : perimeter side_length number_of_sides = 16 :=
by
  sorry

end square_traffic_sign_perimeter_l182_182742


namespace brick_width_correct_l182_182571

theorem brick_width_correct
  (courtyard_length_m : ℕ) (courtyard_width_m : ℕ) (brick_length_cm : ℕ) (num_bricks : ℕ)
  (total_area_cm : ℕ) (brick_width_cm : ℕ) :
  courtyard_length_m = 25 →
  courtyard_width_m = 16 →
  brick_length_cm = 20 →
  num_bricks = 20000 →
  total_area_cm = courtyard_length_m * 100 * courtyard_width_m * 100 →
  total_area_cm = num_bricks * brick_length_cm * brick_width_cm →
  brick_width_cm = 10 :=
by
  intros h1 h2 h3 h4 h5 h6
  sorry

end brick_width_correct_l182_182571


namespace divides_six_ab_l182_182657

theorem divides_six_ab 
  (a b n : ℕ) 
  (hb : b < 10) 
  (hn : n > 3) 
  (h_eq : 2^n = 10 * a + b) : 
  6 ∣ (a * b) :=
sorry

end divides_six_ab_l182_182657


namespace hot_dogs_remainder_l182_182502

theorem hot_dogs_remainder :
  let n := 16789537
  let d := 5
  n % d = 2 :=
by
  sorry

end hot_dogs_remainder_l182_182502


namespace find_m_l182_182546

theorem find_m :
  ∃ m : ℝ, (∀ x : ℝ, x > 0 → (m^2 - m - 5) * x^(m - 1) > 0) ∧ m = 3 :=
sorry

end find_m_l182_182546


namespace distance_between_B_and_C_l182_182555

theorem distance_between_B_and_C
  (A B C : Type)
  (AB : ℝ)
  (angle_A : ℝ)
  (angle_B : ℝ)
  (h_AB : AB = 10)
  (h_angle_A : angle_A = 60)
  (h_angle_B : angle_B = 75) :
  ∃ BC : ℝ, BC = 5 * Real.sqrt 6 :=
by
  sorry

end distance_between_B_and_C_l182_182555


namespace no_solution_eqn_l182_182220

theorem no_solution_eqn (m : ℝ) :
  ¬ ∃ x : ℝ, (3 - 2 * x) / (x - 3) - (m * x - 2) / (3 - x) = -1 ↔ m = 1 :=
by
  sorry

end no_solution_eqn_l182_182220


namespace train_avg_speed_without_stoppages_l182_182704

/-- A train with stoppages has an average speed of 125 km/h. Given that the train stops for 30 minutes per hour,
the average speed of the train without stoppages is 250 km/h. -/
theorem train_avg_speed_without_stoppages (avg_speed_with_stoppages : ℝ) 
  (stoppage_time_per_hour : ℝ) (no_stoppage_speed : ℝ) 
  (h1 : avg_speed_with_stoppages = 125) (h2 : stoppage_time_per_hour = 0.5) : 
  no_stoppage_speed = 250 :=
sorry

end train_avg_speed_without_stoppages_l182_182704


namespace vertical_asymptote_sum_l182_182403

theorem vertical_asymptote_sum : 
  let f (x : ℝ) := (4 * x ^ 2 + 1) / (6 * x ^ 2 + 7 * x + 3)
  ∃ p q : ℝ, (6 * p ^ 2 + 7 * p + 3 = 0) ∧ (6 * q ^ 2 + 7 * q + 3 = 0) ∧ p + q = -11 / 6 :=
by
  let f (x : ℝ) := (4 * x ^ 2 + 1) / (6 * x ^ 2 + 7 * x + 3)
  exact sorry

end vertical_asymptote_sum_l182_182403


namespace merchant_gross_profit_l182_182706

-- Define the purchase price and markup rate
def purchase_price : ℝ := 42
def markup_rate : ℝ := 0.30
def discount_rate : ℝ := 0.20

-- Define the selling price equation given the purchase price and markup rate
def selling_price (S : ℝ) : Prop := S = purchase_price + markup_rate * S

-- Define the discounted selling price given the selling price and discount rate
def discounted_selling_price (S : ℝ) : ℝ := S - discount_rate * S

-- Define the gross profit as the difference between the discounted selling price and purchase price
def gross_profit (S : ℝ) : ℝ := discounted_selling_price S - purchase_price

theorem merchant_gross_profit : ∃ S : ℝ, selling_price S ∧ gross_profit S = 6 :=
by
  sorry

end merchant_gross_profit_l182_182706


namespace room_volume_correct_l182_182508

variable (Length Width Height : ℕ) (Volume : ℕ)

-- Define the dimensions of the room
def roomLength := 100
def roomWidth := 10
def roomHeight := 10

-- Define the volume function
def roomVolume (l w h : ℕ) : ℕ := l * w * h

-- Theorem to prove the volume of the room
theorem room_volume_correct : roomVolume roomLength roomWidth roomHeight = 10000 := 
by
  -- roomVolume 100 10 10 = 10000
  sorry

end room_volume_correct_l182_182508


namespace bridge_length_l182_182705

theorem bridge_length (rate : ℝ) (time_minutes : ℝ) (length : ℝ) 
    (rate_condition : rate = 10) 
    (time_condition : time_minutes = 15) : 
    length = 2.5 := 
by
  sorry

end bridge_length_l182_182705


namespace father_age_l182_182764

variable (F C1 C2 : ℕ)

theorem father_age (h1 : F = 3 * (C1 + C2))
  (h2 : F + 5 = 2 * (C1 + 5 + C2 + 5)) :
  F = 45 := by
  sorry

end father_age_l182_182764


namespace coefficient_x2_term_l182_182540

open Polynomial

noncomputable def poly1 : Polynomial ℝ := (X - 1)^3
noncomputable def poly2 : Polynomial ℝ := (X - 1)^4

theorem coefficient_x2_term :
  coeff (poly1 + poly2) 2 = 3 :=
sorry

end coefficient_x2_term_l182_182540


namespace cat_food_problem_l182_182319

noncomputable def L : ℝ
noncomputable def S : ℝ

theorem cat_food_problem (L S : ℝ) (h : L + 4 * S = 14) : L + 3 * S ≥ 11 :=
by 
  sorry

end cat_food_problem_l182_182319


namespace calculate_expression_l182_182306

theorem calculate_expression : 
  (10 - 9 * 8 + 7^2 / 2 - 3 * 4 + 6 - 5 = -48.5) :=
by
  -- Proof goes here
  sorry

end calculate_expression_l182_182306


namespace sum_of_interior_angles_divisible_by_360_l182_182729

theorem sum_of_interior_angles_divisible_by_360
  (n : ℕ)
  (h : n > 0) :
  ∃ k : ℤ, ((2 * n - 2) * 180) = 360 * k :=
by
  sorry

end sum_of_interior_angles_divisible_by_360_l182_182729


namespace eight_dice_probability_equal_split_l182_182602

theorem eight_dice_probability_equal_split :
  let n := 8 in
  let p := (1 / 2) in
  (nat.choose n (n / 2) * p ^ n) = (35 / 128)
  :=
by
  sorry

end eight_dice_probability_equal_split_l182_182602


namespace max_marks_obtainable_l182_182749

theorem max_marks_obtainable 
  (math_pass_percentage : ℝ := 36 / 100)
  (phys_pass_percentage : ℝ := 40 / 100)
  (chem_pass_percentage : ℝ := 45 / 100)
  (math_marks : ℕ := 130)
  (math_fail_margin : ℕ := 14)
  (phys_marks : ℕ := 120)
  (phys_fail_margin : ℕ := 20)
  (chem_marks : ℕ := 160)
  (chem_fail_margin : ℕ := 10) : 
  ∃ max_total_marks : ℤ, max_total_marks = 1127 := 
by 
  sorry  -- Proof not required

end max_marks_obtainable_l182_182749


namespace jimin_class_students_l182_182473

theorem jimin_class_students 
    (total_distance : ℝ)
    (interval_distance : ℝ)
    (h1 : total_distance = 242)
    (h2 : interval_distance = 5.5) :
    (total_distance / interval_distance) + 1 = 45 :=
by sorry

end jimin_class_students_l182_182473


namespace intersection_eq_T_l182_182994

def S : Set ℤ := {s | ∃ n : ℤ, s = 2 * n + 1}
def T : Set ℤ := {t | ∃ n : ℤ, t = 4 * n + 1}

theorem intersection_eq_T : S ∩ T = T := sorry

end intersection_eq_T_l182_182994


namespace min_val_of_a2_plus_b2_l182_182228

variable (a b : ℝ)

def condition := 3 * a - 4 * b - 2 = 0

theorem min_val_of_a2_plus_b2 : condition a b → (∃ a b : ℝ, a^2 + b^2 = 4 / 25) := by 
  sorry

end min_val_of_a2_plus_b2_l182_182228


namespace frac_equality_l182_182349

variables (a b : ℚ) -- Declare the variables as rational numbers

-- State the theorem with the given condition and the proof goal
theorem frac_equality (h : a / b = 2 / 3) : a / (a + b) = 2 / 5 :=
by
  sorry -- proof goes here

end frac_equality_l182_182349


namespace solve_equation_l182_182043

theorem solve_equation (x : ℝ) : 3 * x * (x + 3) = 2 * (x + 3) ↔ (x = -3 ∨ x = 2/3) :=
by
  sorry

end solve_equation_l182_182043


namespace intersection_eq_T_l182_182993

def S : Set ℤ := {s | ∃ n : ℤ, s = 2 * n + 1}
def T : Set ℤ := {t | ∃ n : ℤ, t = 4 * n + 1}

theorem intersection_eq_T : S ∩ T = T := sorry

end intersection_eq_T_l182_182993


namespace value_of_expression_l182_182700

theorem value_of_expression : (1 * 2 * 3 * 4 * 5 * 6 : ℚ) / (1 + 2 + 3 + 4 + 5 + 6) = 240 / 7 := 
by 
  sorry

end value_of_expression_l182_182700


namespace jimmy_bought_3_pens_l182_182515

def cost_of_notebooks (num_notebooks : ℕ) (price_per_notebook : ℕ) : ℕ := num_notebooks * price_per_notebook
def cost_of_folders (num_folders : ℕ) (price_per_folder : ℕ) : ℕ := num_folders * price_per_folder
def total_cost (cost_notebooks cost_folders : ℕ) : ℕ := cost_notebooks + cost_folders
def total_spent (initial_money change : ℕ) : ℕ := initial_money - change
def cost_of_pens (total_spent amount_for_items : ℕ) : ℕ := total_spent - amount_for_items
def num_pens (cost_pens price_per_pen : ℕ) : ℕ := cost_pens / price_per_pen

theorem jimmy_bought_3_pens :
  let pen_price := 1
  let notebook_price := 3
  let num_notebooks := 4
  let folder_price := 5
  let num_folders := 2
  let initial_money := 50
  let change := 25
  let cost_notebooks := cost_of_notebooks num_notebooks notebook_price
  let cost_folders := cost_of_folders num_folders folder_price
  let total_items_cost := total_cost cost_notebooks cost_folders
  let amount_spent := total_spent initial_money change
  let pen_cost := cost_of_pens amount_spent total_items_cost
  num_pens pen_cost pen_price = 3 :=
by
  sorry

end jimmy_bought_3_pens_l182_182515


namespace remaining_stickers_l182_182424

def stickers_per_page : ℕ := 20
def pages : ℕ := 12
def lost_pages : ℕ := 1

theorem remaining_stickers : 
  (pages * stickers_per_page - lost_pages * stickers_per_page) = 220 :=
  by
    sorry

end remaining_stickers_l182_182424


namespace find_first_term_l182_182615

-- Define the arithmetic sequence and its properties
def arithmetic_seq (a d : ℤ) (n : ℕ) : ℤ := a + n * d

variable (a1 a3 a9 d : ℤ)

-- Given conditions
axiom h1 : arithmetic_seq a1 d 2 = 30
axiom h2 : arithmetic_seq a1 d 8 = 60

theorem find_first_term : a1 = 20 :=
by
  -- mathematical proof steps here
  sorry

end find_first_term_l182_182615


namespace cos_90_eq_zero_l182_182161

def point_after_rotation (θ : ℝ) : ℝ × ℝ :=
  let x := cos θ
  let y := sin θ
  (x, y)

theorem cos_90_eq_zero : cos (real.pi / 2) = 0 :=
  sorry

end cos_90_eq_zero_l182_182161


namespace values_of_N_l182_182780

theorem values_of_N (N : ℕ) : (∃ k, k ∈ ({4, 6, 8, 12, 16, 24, 48} : set ℕ) ∧ k = N + 3) ↔ (N ∈ {1, 3, 5, 9, 13, 21, 45} : set ℕ) :=
by 
  sorry

#eval values_of_N 4 -- Example usage: should give true if N = 1

end values_of_N_l182_182780


namespace number_of_pebbles_l182_182270

theorem number_of_pebbles (P : ℕ) : 
  (P * (1/4 : ℝ) + 3 * (1/2 : ℝ) + 2 * 2 = 7) → P = 6 := by
  sorry

end number_of_pebbles_l182_182270


namespace value_of_x_l182_182082

-- Define variables and conditions
def consecutive (x y z : ℤ) : Prop := x = z + 2 ∧ y = z + 1

-- Main proposition
theorem value_of_x (x y z : ℤ) (h1 : consecutive x y z) (h2 : z = 2) (h3 : 2 * x + 3 * y + 3 * z = 5 * y + 8) : x = 4 :=
by
  sorry

end value_of_x_l182_182082


namespace joan_remaining_kittens_l182_182232

-- Definitions based on the given conditions
def original_kittens : Nat := 8
def kittens_given_away : Nat := 2

-- Statement to prove
theorem joan_remaining_kittens : original_kittens - kittens_given_away = 6 := 
by
  -- Proof skipped
  sorry

end joan_remaining_kittens_l182_182232


namespace ratio_of_perimeters_l182_182072

theorem ratio_of_perimeters (s S : ℝ) 
  (h1 : S = 3 * s) : 
  (4 * S) / (4 * s) = 3 :=
by
  sorry

end ratio_of_perimeters_l182_182072


namespace inf_solutions_integers_l182_182805

theorem inf_solutions_integers (x y z : ℕ) : ∃ (n : ℕ), ∀ n > 0, (x = 2^(32 + 72 * n)) ∧ (y = 2^(28 + 63 * n)) ∧ (z = 2^(25 + 56 * n)) → x^7 + y^8 = z^9 :=
by {
  sorry
}

end inf_solutions_integers_l182_182805


namespace pool_students_count_l182_182664

noncomputable def total_students (total_women : ℕ) (female_students : ℕ) (extra_men : ℕ) (non_student_men : ℕ) : ℕ := 
  let total_men := total_women + extra_men
  let male_students := total_men - non_student_men
  female_students + male_students

theorem pool_students_count
  (total_women : ℕ := 1518)
  (female_students : ℕ := 536)
  (extra_men : ℕ := 525)
  (non_student_men : ℕ := 1257) :
  total_students total_women female_students extra_men non_student_men = 1322 := 
by
  sorry

end pool_students_count_l182_182664


namespace cos_90_eq_zero_l182_182179

theorem cos_90_eq_zero : Real.cos (90 * Real.pi / 180) = 0 := by 
  sorry

end cos_90_eq_zero_l182_182179


namespace make_fraction_meaningful_l182_182866

theorem make_fraction_meaningful (x : ℝ) : (x - 1) ≠ 0 ↔ x ≠ 1 :=
by
  sorry

end make_fraction_meaningful_l182_182866


namespace sum_first_10_terms_l182_182796

-- Define the conditions for the problem
def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n, a (n + 1) = a n * q

def arithmetic_sequence (b c d : ℝ) : Prop :=
  2 * c = b + d

def conditions (a : ℕ → ℝ) (q : ℝ) : Prop :=
  a 1 = 1 ∧
  geometric_sequence a q ∧
  arithmetic_sequence (4 * a 1) (2 * a 2) (a 3)

-- Define the sum of the first n terms of a geometric sequence
def sum_first_n_terms (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  (Finset.range n).sum a

-- Prove the final result
theorem sum_first_10_terms (a : ℕ → ℝ) (q : ℝ) (h : conditions a q) :
  sum_first_n_terms a 10 = 1023 :=
sorry

end sum_first_10_terms_l182_182796


namespace gcd_of_q_and_r_l182_182636

theorem gcd_of_q_and_r (p q r : ℕ) (hpq : p > 0) (hqr : q > 0) (hpr : r > 0)
    (gcd_pq : Nat.gcd p q = 240) (gcd_pr : Nat.gcd p r = 540) : Nat.gcd q r = 60 := by
  sorry

end gcd_of_q_and_r_l182_182636


namespace inequality_ac2_bc2_implies_a_b_l182_182775

theorem inequality_ac2_bc2_implies_a_b (a b c : ℝ) (h : a * c^2 > b * c^2) : a > b :=
sorry

end inequality_ac2_bc2_implies_a_b_l182_182775


namespace power_identity_l182_182481

theorem power_identity (x y a b : ℝ) (h1 : 10^x = a) (h2 : 10^y = b) : 10^(3*x + 2*y) = a^3 * b^2 := 
by 
  sorry

end power_identity_l182_182481


namespace carrots_left_over_l182_182585

theorem carrots_left_over (c g : ℕ) (h₁ : c = 47) (h₂ : g = 4) : c % g = 3 :=
by
  sorry

end carrots_left_over_l182_182585


namespace sum_A_B_C_l182_182859

noncomputable def number_B (A : ℕ) : ℕ := (A * 5) / 2
noncomputable def number_C (B : ℕ) : ℕ := (B * 7) / 4

theorem sum_A_B_C (A B C : ℕ) (h1 : A = 16) (h2 : A * 5 = B * 2) (h3 : B * 7 = C * 4) :
  A + B + C = 126 :=
by
  sorry

end sum_A_B_C_l182_182859


namespace log_equation_positive_x_l182_182434

theorem log_equation_positive_x (x : ℝ) (hx : 0 < x) (hx1 : x ≠ 1) : 
  (Real.log x / Real.log 2) * (Real.log 9 / Real.log x) = Real.log 9 / Real.log 2 :=
by sorry

end log_equation_positive_x_l182_182434


namespace max_xy_of_perpendicular_l182_182498

open Real

noncomputable def vector_a (x : ℝ) : ℝ × ℝ := (1, x - 1)
noncomputable def vector_b (y : ℝ) : ℝ × ℝ := (y, 2)

def dot_product (v1 v2 : ℝ × ℝ) : ℝ := v1.1 * v2.1 + v1.2 * v2.2 

theorem max_xy_of_perpendicular (x y : ℝ) 
  (h_perp : dot_product (vector_a x) (vector_b y) = 0) : xy ≤ 1/2 :=
by
  sorry

end max_xy_of_perpendicular_l182_182498


namespace cos_90_eq_zero_l182_182175

theorem cos_90_eq_zero : Real.cos (90 * Real.pi / 180) = 0 := by 
  sorry

end cos_90_eq_zero_l182_182175


namespace vinny_fifth_month_loss_l182_182273

theorem vinny_fifth_month_loss (start_weight : ℝ) (end_weight : ℝ) (first_month_loss : ℝ) (second_month_loss : ℝ) (third_month_loss : ℝ) (fourth_month_loss : ℝ) (total_loss : ℝ):
  start_weight = 300 ∧
  first_month_loss = 20 ∧
  second_month_loss = first_month_loss / 2 ∧
  third_month_loss = second_month_loss / 2 ∧
  fourth_month_loss = third_month_loss / 2 ∧
  (start_weight - end_weight) = total_loss ∧
  end_weight = 250.5 →
  (total_loss - (first_month_loss + second_month_loss + third_month_loss + fourth_month_loss)) = 12 :=
by
  sorry

end vinny_fifth_month_loss_l182_182273


namespace problem1_problem2_problem3_l182_182917

section Problem1

variable (a b : ℝ)
variable (h_condition : a^b * b^a + log a b = 0)
variable (ha_pos : 0 < a)
variable (hb_pos : 0 < b)

theorem problem1 : ab + 1 < a + b :=
sorry

end Problem1

section Problem2

variable (b : ℝ)
variable (h_condition : 2^b * b^2 + log 2 b = 0)
variable (hb_pos : 0 < b)

theorem problem2 : 2^b < 1 / b :=
sorry

end Problem2

section Problem3

variable (b : ℝ)
variable (h_condition : 2^b * b^2 + log 2 b = 0)
variable (hb_pos : 0 < b)

theorem problem3 : (2 * b + 1 - real.sqrt 5) * (3 * b - 2) < 0 :=
sorry

end Problem3

end problem1_problem2_problem3_l182_182917


namespace fraction_of_sum_after_6_years_l182_182743

-- Define the principal amount, rate, and time period as given in the conditions
def P : ℝ := 1
def R : ℝ := 0.02777777777777779
def T : ℕ := 6

-- Definition of the Simple Interest calculation
def simple_interest (P R : ℝ) (T : ℕ) : ℝ :=
  P * R * T

-- Definition of the total amount after 6 years
def total_amount (P SI : ℝ) : ℝ :=
  P + SI

-- The main theorem to prove
theorem fraction_of_sum_after_6_years :
  total_amount P (simple_interest P R T) = 1.1666666666666667 :=
by
  sorry

end fraction_of_sum_after_6_years_l182_182743


namespace chemistry_problem_l182_182445

theorem chemistry_problem 
(C : ℝ)  -- concentration of the original salt solution
(h_mix : 1 * C / 100 = 15 * 2 / 100) : 
  C = 30 := 
sorry

end chemistry_problem_l182_182445


namespace result_l182_182464

def problem : Float :=
  let sum := 78.652 + 24.3981
  let diff := sum - 0.025
  Float.round (diff * 100) / 100

theorem result :
  problem = 103.03 := by
  sorry

end result_l182_182464


namespace sum_series_75_to_99_l182_182301

theorem sum_series_75_to_99 : 
  let a := 75
  let l := 99
  let n := l - a + 1
  let s := n * (a + l) / 2
  s = 2175 :=
by
  sorry

end sum_series_75_to_99_l182_182301


namespace calculate_g_l182_182840

def g (a b c : ℚ) : ℚ := (2 * c + a) / (b - c)

theorem calculate_g : g 3 6 (-1) = 1 / 7 :=
by
    -- Proof is not included
    sorry

end calculate_g_l182_182840


namespace cos_ninety_degrees_l182_182121

theorem cos_ninety_degrees : Real.cos (90 * Real.pi / 180) = 0 := 
by 
  sorry

end cos_ninety_degrees_l182_182121


namespace polygon_sides_l182_182267

theorem polygon_sides (h1 : 1260 - 360 = 900) (h2 : (n - 2) * 180 = 900) : n = 7 :=
by 
  sorry

end polygon_sides_l182_182267


namespace marta_sold_on_saturday_l182_182031

-- Definitions of conditions
def initial_shipment : ℕ := 1000
def rotten_tomatoes : ℕ := 200
def second_shipment : ℕ := 2000
def tomatoes_on_tuesday : ℕ := 2500
def x := 300

-- Total tomatoes on Monday after the second shipment
def tomatoes_on_monday (sold_tomatoes : ℕ) : ℕ :=
  initial_shipment - sold_tomatoes - rotten_tomatoes + second_shipment

-- Theorem statement to prove
theorem marta_sold_on_saturday : (tomatoes_on_monday x = tomatoes_on_tuesday) -> (x = 300) :=
by 
  intro h
  sorry

end marta_sold_on_saturday_l182_182031


namespace mean_score_of_seniors_l182_182666

theorem mean_score_of_seniors 
  (s n : ℕ)
  (ms mn : ℝ)
  (h1 : s + n = 120)
  (h2 : n = 2 * s)
  (h3 : ms = 1.5 * mn)
  (h4 : (s : ℝ) * ms + (n : ℝ) * mn = 13200)
  : ms = 141.43 :=
by
  sorry

end mean_score_of_seniors_l182_182666


namespace problem_statement_l182_182835

variable (a b c : ℝ)

def g (x : ℝ) : ℝ := a * x^8 + b * x^6 - c * x^4 + 5

theorem problem_statement (h : g a b c 10 = 3) : g a b c 10 + g a b c (-10) = 6 := by
  have h_even : ∀ x : ℝ, g a b c x = g a b c (-x) := by
    intro x
    simp [g]
  have h_neg10 : g a b c (-10) = g a b c 10 := h_even 10
  rw [h_neg10, h]
  norm_num
  sorry

end problem_statement_l182_182835


namespace intersection_eq_T_l182_182933

def S : Set ℤ := {s | ∃ n : ℤ, s = 2 * n + 1}
def T : Set ℤ := {t | ∃ n : ℤ, t = 4 * n + 1}

theorem intersection_eq_T : S ∩ T = T := 
sorry

end intersection_eq_T_l182_182933


namespace minimum_employees_needed_l182_182448

noncomputable def employees_needed (total_days : ℕ) (work_days : ℕ) (rest_days : ℕ) (min_on_duty : ℕ) : ℕ :=
  let comb := (total_days.choose rest_days)
  min_on_duty * comb / work_days

theorem minimum_employees_needed {total_days work_days rest_days min_on_duty : ℕ} (h_total_days: total_days = 7) (h_work_days: work_days = 5) (h_rest_days: rest_days = 2) (h_min_on_duty: min_on_duty = 45) : 
  employees_needed total_days work_days rest_days min_on_duty = 63 := by
  rw [h_total_days, h_work_days, h_rest_days, h_min_on_duty]
  -- detailed computation and proofs steps omitted
  -- the critical part is to ensure 63 is derived correctly based on provided values
  sorry

end minimum_employees_needed_l182_182448


namespace smaller_angle_at_9_00_am_l182_182752

-- Definitions based on conditions identified in Step A
def minute_hand_position : ℝ := 0
def hour_hand_position : ℝ := 270
def degrees_per_hour : ℝ := 30

-- The main theorem statement
theorem smaller_angle_at_9_00_am : 
  let smaller_angle := min (abs (minute_hand_position - hour_hand_position)) (360 - abs (minute_hand_position - hour_hand_position))
  in smaller_angle = 90 := 
by 
  -- Since we skip the proof, we just put 'sorry'
  sorry

end smaller_angle_at_9_00_am_l182_182752


namespace intersection_S_T_eq_T_l182_182950

def S : Set ℤ := { s | ∃ n : ℤ, s = 2 * n + 1 }
def T : Set ℤ := { t | ∃ n : ℤ, t = 4 * n + 1 }

theorem intersection_S_T_eq_T : S ∩ T = T := by
  sorry

end intersection_S_T_eq_T_l182_182950


namespace total_distance_biked_l182_182747

theorem total_distance_biked :
  let monday_distance := 12
  let tuesday_distance := 2 * monday_distance - 3
  let wednesday_distance := 2 * 11
  let thursday_distance := wednesday_distance + 2
  let friday_distance := thursday_distance + 2
  let saturday_distance := friday_distance + 2
  let sunday_distance := 3 * 6
  monday_distance + tuesday_distance + wednesday_distance + thursday_distance + friday_distance + saturday_distance + sunday_distance = 151 := 
by
  sorry

end total_distance_biked_l182_182747


namespace sqrt_factorial_div_90_eq_l182_182303

open Real

theorem sqrt_factorial_div_90_eq : sqrt (realOfNat (Nat.factorial 9) / 90) = 24 * sqrt 7 := by
  sorry

end sqrt_factorial_div_90_eq_l182_182303


namespace min_length_M_inter_N_l182_182210

def setM (m : ℝ) : Set ℝ := {x | m ≤ x ∧ x ≤ m + 3 / 4}
def setN (n : ℝ) : Set ℝ := {x | n - 1 / 3 ≤ x ∧ x ≤ n}
def setP : Set ℝ := {x | 0 ≤ x ∧ x ≤ 1}

theorem min_length_M_inter_N (m n : ℝ) 
  (hm : 0 ≤ m ∧ m + 3 / 4 ≤ 1) 
  (hn : 1 / 3 ≤ n ∧ n ≤ 1) : 
  let I := (setM m ∩ setN n)
  ∃ Iinf Isup : ℝ, I = {x | Iinf ≤ x ∧ x ≤ Isup} ∧ Isup - Iinf = 1 / 12 :=
  sorry

end min_length_M_inter_N_l182_182210


namespace intersection_eq_T_l182_182938

def S : Set ℤ := {s | ∃ n : ℤ, s = 2 * n + 1}
def T : Set ℤ := {t | ∃ n : ℤ, t = 4 * n + 1}

theorem intersection_eq_T : S ∩ T = T := 
sorry

end intersection_eq_T_l182_182938


namespace subtracted_result_correct_l182_182651

theorem subtracted_result_correct (n : ℕ) (h1 : 96 / n = 6) : 34 - n = 18 :=
by
  sorry

end subtracted_result_correct_l182_182651


namespace part_a_part_b_l182_182030

theorem part_a (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (h : x + y + z ≥ 3) :
  ¬ (1/x + 1/y + 1/z ≤ 3) :=
sorry

theorem part_b (x y z : ℝ) (hx : 0 < x) (hy : 0 < y) (hz : 0 < z) (h : x + y + z ≤ 3) :
  1/x + 1/y + 1/z ≥ 3 :=
sorry

end part_a_part_b_l182_182030


namespace max_value_of_product_l182_182240

theorem max_value_of_product (x y z w : ℝ) (h_pos : 0 < x ∧ 0 < y ∧ 0 < z ∧ 0 < w) (h_sum : x + y + z + w = 1) : 
  x^2 * y^2 * z^2 * w ≤ 64 / 823543 :=
by
  sorry

end max_value_of_product_l182_182240


namespace decreasing_interval_l182_182350

def f (a x : ℝ) : ℝ := x^2 + 2*(a - 1)*x + 2

theorem decreasing_interval (a : ℝ) : (∀ x y : ℝ, x ≤ y → y ≤ 4 → f a y ≤ f a x) ↔ a < -3 := 
by
  sorry

end decreasing_interval_l182_182350


namespace find_second_number_l182_182556

theorem find_second_number (x : ℕ) (h1 : ∀ d : ℕ, d ∣ 60 → d ∣ x → d ∣ 18) 
                           (h2 : 60 % 18 = 6) (h3 : x % 18 = 10) 
                           (h4 : x > 60) : 
  x = 64 := 
by
  sorry

end find_second_number_l182_182556


namespace total_spending_is_450_l182_182655

-- Define the costs of items bought by Leonard
def leonard_wallet_cost : ℕ := 50
def pair_of_sneakers_cost : ℕ := 100
def pairs_of_sneakers : ℕ := 2

-- Define the costs of items bought by Michael
def michael_backpack_cost : ℕ := 100
def pair_of_jeans_cost : ℕ := 50
def pairs_of_jeans : ℕ := 2

-- Define the total spending of Leonard and Michael 
def total_spent : ℕ :=
  leonard_wallet_cost + (pair_of_sneakers_cost * pairs_of_sneakers) + 
  michael_backpack_cost + (pair_of_jeans_cost * pairs_of_jeans)

-- The proof statement
theorem total_spending_is_450 : total_spent = 450 := 
by
  sorry

end total_spending_is_450_l182_182655


namespace club_additional_members_l182_182721

theorem club_additional_members (current_members : ℕ) (additional_members : ℕ) :
  current_members = 10 →
  additional_members = (2 * current_members) + 5 - current_members →
  additional_members = 15 :=
by
  intro h1 h2
  rw [h1] at h2
  norm_num at h2
  exact h2

end club_additional_members_l182_182721


namespace max_profit_l182_182293

noncomputable def C (x : ℝ) : ℝ :=
  if x < 80 then (1/2) * x^2 + 40 * x
  else 101 * x + 8100 / x - 2180

noncomputable def profit (x : ℝ) : ℝ :=
  if x < 80 then 100 * x - C x - 500
  else 100 * x - C x - 500

theorem max_profit :
  (∀ x, (0 < x ∧ x < 80) → profit x = - (1/2) * x^2 + 60 * x - 500) ∧
  (∀ x, (80 ≤ x) → profit x = 1680 - (x + 8100 / x)) ∧
  (∃ x, x = 90 ∧ profit x = 1500) :=
by {
  -- Proof here
  sorry
}

end max_profit_l182_182293


namespace sequence_solution_l182_182824

theorem sequence_solution 
  (a : ℕ → ℝ)
  (h1 : ∀ n, a (n + 1) = a n / (2 + a n))
  (h2 : a 1 = 1) :
  ∀ n, a n = 1 / (2^n - 1) :=
sorry

end sequence_solution_l182_182824


namespace binomial_sum_mod_3_l182_182038

open BigOperators

theorem binomial_sum_mod_3 :
  let n := 6002
  ∑ k in finset.Ico (0 : ℕ) (n / 6), nat.choose n (k * 6 + 1) % 3 = 1 :=
by
  sorry

end binomial_sum_mod_3_l182_182038


namespace fraction_subtraction_l182_182699

theorem fraction_subtraction : (3 + 5 + 7) / (2 + 4 + 6) - (2 - 4 + 6) / (3 - 5 + 7) = 9 / 20 :=
by
  sorry

end fraction_subtraction_l182_182699


namespace stacy_height_now_l182_182022

-- Definitions based on the given conditions
def S_initial : ℕ := 50
def J_initial : ℕ := 45
def J_growth : ℕ := 1
def S_growth : ℕ := J_growth + 6

-- Prove statement about Stacy's current height
theorem stacy_height_now : S_initial + S_growth = 57 := by
  sorry

end stacy_height_now_l182_182022


namespace greatest_drop_in_price_l182_182068

theorem greatest_drop_in_price (jan feb mar apr may jun : ℝ)
  (h_jan : jan = -0.50)
  (h_feb : feb = 2.00)
  (h_mar : mar = -2.50)
  (h_apr : apr = 3.00)
  (h_may : may = -0.50)
  (h_jun : jun = -2.00) :
  mar = -2.50 ∧ (mar ≤ jan ∧ mar ≤ may ∧ mar ≤ jun) :=
by
  sorry

end greatest_drop_in_price_l182_182068


namespace simplify_expression_l182_182294

theorem simplify_expression : 
  8 - (-3) + (-5) + (-7) = 3 + 8 - 7 - 5 := 
by
  sorry

end simplify_expression_l182_182294


namespace kim_hours_of_classes_per_day_l182_182018

-- Definitions based on conditions
def original_classes : Nat := 4
def hours_per_class : Nat := 2
def dropped_classes : Nat := 1

-- Prove that Kim now has 6 hours of classes per day
theorem kim_hours_of_classes_per_day : (original_classes - dropped_classes) * hours_per_class = 6 := by
  sorry

end kim_hours_of_classes_per_day_l182_182018


namespace integers_abs_le_3_l182_182406

theorem integers_abs_le_3 :
  {x : ℤ | |x| ≤ 3} = { -3, -2, -1, 0, 1, 2, 3 } :=
by
  sorry

end integers_abs_le_3_l182_182406


namespace count_valid_numbers_l182_182627

-- Let n be the number of four-digit numbers greater than 3999 with the product of the middle two digits exceeding 10.
def n : ℕ := 3480

-- Formalize the given conditions:
def is_valid_four_digit (a b c d : ℕ) : Prop :=
  (4 ≤ a ∧ a ≤ 9) ∧ (0 ≤ d ∧ d ≤ 9) ∧ (1 ≤ b ∧ b ≤ 9) ∧ (1 ≤ c ∧ c ≤ 9) ∧ (b * c > 10)

-- The theorem to prove the number of valid four-digit numbers is 3480
theorem count_valid_numbers : 
  (∑ (a b c d : ℕ) in finset.range 10 × finset.range 10 × finset.range 10 × finset.range 10,
    if is_valid_four_digit a b c d then 1 else 0) = n := sorry

end count_valid_numbers_l182_182627


namespace num_valid_four_digit_numbers_l182_182635

def is_valid_number (n : ℕ) : Prop :=
  n >= 4000 ∧ n < 10000 ∧ let d1 := n / 1000,
                             d2 := (n / 100) % 10,
                             d3 := (n / 10) % 10,
                             _ := n % 10 in
                            d1 >= 4 ∧ (d2 > 0 ∧ d2 < 10) ∧ (d3 > 0 ∧ d3 < 10) ∧ (d2 * d3 > 10)

theorem num_valid_four_digit_numbers : 
  (finset.filter is_valid_number (finset.range 10000)).card = 4260 :=
sorry

end num_valid_four_digit_numbers_l182_182635


namespace max_area_isosceles_triangle_l182_182008

theorem max_area_isosceles_triangle (b : ℝ) (h : ℝ) (area : ℝ) 
  (h_cond : h^2 = 1 - b^2 / 4)
  (area_def : area = 1 / 2 * b * h) : 
  area ≤ 2 * Real.sqrt 2 / 3 := 
sorry

end max_area_isosceles_triangle_l182_182008


namespace monotonicity_m_eq_zero_range_of_m_l182_182492

noncomputable def f (x : ℝ) (m : ℝ) : ℝ := Real.exp x - m * x^2 - 2 * x

theorem monotonicity_m_eq_zero :
  ∀ x : ℝ, (x < Real.log 2 → f x 0 < f (x + 1) 0) ∧ (x > Real.log 2 → f x 0 > f (x - 1) 0) := 
sorry

theorem range_of_m :
  ∀ x : ℝ, x ∈ Set.Ici 0 → f x m > (Real.exp 1 / 2 - 1) → m < (Real.exp 1 / 2 - 1) := 
sorry

end monotonicity_m_eq_zero_range_of_m_l182_182492


namespace k_greater_than_half_l182_182529

-- Definition of the problem conditions
variables {a b c k : ℝ}

-- Assume a, b, c are the sides of a triangle
axiom triangle_inequality : a + b > c

-- Given condition
axiom sides_condition : a^2 + b^2 = k * c^2

-- The theorem to prove k > 0.5
theorem k_greater_than_half (h1 : a + b > c) (h2 : a^2 + b^2 = k * c^2) : k > 0.5 :=
by
  sorry

end k_greater_than_half_l182_182529


namespace peacocks_in_zoo_l182_182507

theorem peacocks_in_zoo :
  ∃ p t : ℕ, 2 * p + 4 * t = 54 ∧ p + t = 17 ∧ p = 7 :=
by
  sorry

end peacocks_in_zoo_l182_182507


namespace cat_food_problem_l182_182321

noncomputable def L : ℝ
noncomputable def S : ℝ

theorem cat_food_problem (L S : ℝ) (h : L + 4 * S = 14) : L + 3 * S ≥ 11 :=
by 
  sorry

end cat_food_problem_l182_182321


namespace ratio_noah_to_joe_l182_182034

def noah_age_after_10_years : ℕ := 22
def years_elapsed : ℕ := 10
def joe_age : ℕ := 6
def noah_age : ℕ := noah_age_after_10_years - years_elapsed

theorem ratio_noah_to_joe : noah_age / joe_age = 2 := by
  -- calculation omitted for brevity
  sorry

end ratio_noah_to_joe_l182_182034


namespace balloons_in_package_initially_l182_182067

-- Definition of conditions
def friends : ℕ := 5
def balloons_given_back : ℕ := 11
def balloons_after_giving_back : ℕ := 39

-- Calculation for original balloons each friend had
def original_balloons_each_friend := balloons_after_giving_back + balloons_given_back

-- Theorem: Number of balloons in the package initially
theorem balloons_in_package_initially : 
  (original_balloons_each_friend * friends) = 250 :=
by
  sorry

end balloons_in_package_initially_l182_182067


namespace intersection_of_sets_l182_182974

open Set Int

theorem intersection_of_sets (S T : Set Int) (hS : S = {s | ∃ n : ℤ, s = 2 * n + 1}) (hT : T = {t | ∃ n : ℤ, t = 4 * n + 1}) : S ∩ T = T :=
  by
  sorry

end intersection_of_sets_l182_182974


namespace sum_in_base4_l182_182405

def dec_to_base4 (n : ℕ) : ℕ :=
  if n = 0 then 0 else
  let rec convert (n : ℕ) (acc : ℕ) (power : ℕ) :=
    if n = 0 then acc
    else convert (n / 4) (acc + (n % 4) * power) (power * 10)
  convert n 0 1

theorem sum_in_base4 : dec_to_base4 (234 + 78) = 13020 :=
  sorry

end sum_in_base4_l182_182405


namespace cos_sin_identity_l182_182304

theorem cos_sin_identity : 
  (Real.cos (75 * Real.pi / 180) + Real.sin (75 * Real.pi / 180)) * 
  (Real.cos (75 * Real.pi / 180) - Real.sin (75 * Real.pi / 180)) = -Real.sqrt 3 / 2 := 
  sorry

end cos_sin_identity_l182_182304


namespace tan_sum_pi_div_4_sin_fraction_simplifies_to_1_l182_182488

variable (α : ℝ)
variable (π : ℝ) [Fact (π > 0)]

-- Assume condition
axiom tan_alpha_eq_2 : Real.tan α = 2

-- Goal (1): Prove that tan(α + π/4) = -3
theorem tan_sum_pi_div_4 : Real.tan (α + π / 4) = -3 :=
by
  sorry

-- Goal (2): Prove that (sin(2α) / (sin^2(α) + sin(α) * cos(α) - cos(2α) - 1)) = 1
theorem sin_fraction_simplifies_to_1 :
  (Real.sin (2 * α)) / (Real.sin (α)^2 + Real.sin (α) * Real.cos (α) - Real.cos (2 * α) - 1) = 1 :=
by
  sorry

end tan_sum_pi_div_4_sin_fraction_simplifies_to_1_l182_182488


namespace kyle_and_miles_marbles_l182_182475

theorem kyle_and_miles_marbles (f k m : ℕ) 
  (h1 : f = 3 * k) 
  (h2 : f = 5 * m) 
  (h3 : f = 15) : 
  k + m = 8 := 
by 
  sorry

end kyle_and_miles_marbles_l182_182475


namespace next_time_angle_150_degrees_l182_182499

theorem next_time_angle_150_degrees (x : ℝ) : (∃ x, x = 329/6) :=
by
  let θ := λ H M : ℝ, abs (30 * H - 5.5 * M)
  let initial_angle := θ 5 0
  have eq1 : initial_angle = 150 :=
    by sorry
  let H := 5 + x / 60
  have eq2 : θ H x = 150 :=
    by sorry
  have eq3 : abs (150 - 5 * x) = 150 :=
    by sorry
  have eq4 : abs (150 - 5 * x) = 150 := by sorry
  have solution : x = 54 + 6 / 11 :=
    by sorry
  existsi solution
  sorry

end next_time_angle_150_degrees_l182_182499


namespace first_week_tickets_calc_l182_182691

def total_tickets : ℕ := 90
def second_week_tickets : ℕ := 17
def tickets_left : ℕ := 35

theorem first_week_tickets_calc : total_tickets - (second_week_tickets + tickets_left) = 38 := by
  sorry

end first_week_tickets_calc_l182_182691


namespace hyperbola_eccentricity_is_sqrt2_l182_182621

noncomputable def hyperbola_eccentricity (a b : ℝ) (hyp1 : a > 0) (hyp2 : b > 0) 
(hyp3 : b = a) : ℝ :=
    let c := Real.sqrt (2) * a
    c / a

theorem hyperbola_eccentricity_is_sqrt2 
(a b : ℝ) (hyp1 : a > 0) (hyp2 : b > 0) (hyp3 : b = a) :
hyperbola_eccentricity a b hyp1 hyp2 hyp3 = Real.sqrt 2 := sorry

end hyperbola_eccentricity_is_sqrt2_l182_182621


namespace original_candle_length_l182_182269

theorem original_candle_length (current_length : ℝ) (factor : ℝ) (h_current : current_length = 48) (h_factor : factor = 1.33) :
  (current_length * factor = 63.84) :=
by
  -- The proof goes here
  sorry

end original_candle_length_l182_182269


namespace number_of_people_tasting_apple_pies_l182_182847

/-- Sedrach's apple pie problem -/
def apple_pies : ℕ := 13
def halves_per_apple_pie : ℕ := 2
def bite_size_samples_per_half : ℕ := 5

theorem number_of_people_tasting_apple_pies :
    (apple_pies * halves_per_apple_pie * bite_size_samples_per_half) = 130 :=
by
  sorry

end number_of_people_tasting_apple_pies_l182_182847


namespace cat_food_sufficiency_l182_182316

theorem cat_food_sufficiency (L S : ℝ) (h : L + 4 * S = 14) : L + 3 * S ≥ 11 :=
sorry

end cat_food_sufficiency_l182_182316


namespace S_inter_T_eq_T_l182_182961

-- Define the sets S and T
def S := { s : ℤ | ∃ n : ℤ, s = 2 * n + 1 }
def T := { t : ℤ | ∃ n : ℤ, t = 4 * n + 1 }

-- Prove that S ∩ T = T
theorem S_inter_T_eq_T : S ∩ T = T := 
by
  sorry

end S_inter_T_eq_T_l182_182961


namespace banks_investments_count_l182_182033

-- Conditions
def revenue_per_investment_banks := 500
def revenue_per_investment_elizabeth := 900
def number_of_investments_elizabeth := 5
def extra_revenue_elizabeth := 500

-- Total revenue calculations
def total_revenue_elizabeth := number_of_investments_elizabeth * revenue_per_investment_elizabeth
def total_revenue_banks := total_revenue_elizabeth - extra_revenue_elizabeth

-- Number of investments for Mr. Banks
def number_of_investments_banks := total_revenue_banks / revenue_per_investment_banks

theorem banks_investments_count : number_of_investments_banks = 8 := by
  sorry

end banks_investments_count_l182_182033


namespace intersection_eq_T_l182_182942

def S : Set ℤ := {s | ∃ n : ℤ, s = 2 * n + 1}
def T : Set ℤ := {t | ∃ n : ℤ, t = 4 * n + 1}

theorem intersection_eq_T : S ∩ T = T := 
sorry

end intersection_eq_T_l182_182942


namespace problem_l182_182356

noncomputable def f (x : ℝ) (a : ℝ) (α : ℝ) (b : ℝ) (β : ℝ) : ℝ :=
  a * Real.sin (Real.pi * x + α) + b * Real.cos (Real.pi * x + β)

theorem problem {a α b β : ℝ} (h : f 2001 a α b β = 3) : f 2012 a α b β = -3 := by
  sorry

end problem_l182_182356


namespace pilot_fish_speed_when_moved_away_l182_182828

/-- Conditions -/
def keanu_speed : ℕ := 20
def shark_new_speed (k : ℕ) : ℕ := 2 * k
def pilot_fish_increase_speed (k s_new : ℕ) : ℕ := k + (s_new - k) / 2

/-- The problem statement to prove -/
theorem pilot_fish_speed_when_moved_away (k : ℕ) (s_new : ℕ) (p_new : ℕ) 
  (h1 : k = 20) 
  (h2 : s_new = shark_new_speed k) 
  (h3 : p_new = pilot_fish_increase_speed k s_new) : 
  p_new = 30 :=
by
  rw [h1] at h2
  rw [h2, h1] at h3
  rw [h3]
  sorry

end pilot_fish_speed_when_moved_away_l182_182828


namespace cos_pi_half_eq_zero_l182_182171

theorem cos_pi_half_eq_zero : Real.cos (Float.pi / 2) = 0 :=
by
  sorry

end cos_pi_half_eq_zero_l182_182171


namespace commission_rate_change_amount_l182_182896

theorem commission_rate_change_amount :
  ∃ X : ℝ, (∀ S : ℝ, ∀ commission : ℝ, S = 15885.42 → commission = (S - 15000) →
  commission = 0.10 * X + 0.05 * (S - X) → X = 1822.98) :=
sorry

end commission_rate_change_amount_l182_182896


namespace S_inter_T_eq_T_l182_182963

-- Define the sets S and T
def S := { s : ℤ | ∃ n : ℤ, s = 2 * n + 1 }
def T := { t : ℤ | ∃ n : ℤ, t = 4 * n + 1 }

-- Prove that S ∩ T = T
theorem S_inter_T_eq_T : S ∩ T = T := 
by
  sorry

end S_inter_T_eq_T_l182_182963


namespace cos_of_90_degrees_l182_182166

-- Definition of cosine of 90 degrees
def cos_90_degrees : ℝ := Real.cos (90 * (Float.pi / 180))

-- Condition: Cosine is the x-coordinate of the point (0, 1) after 90 degree rotation
theorem cos_of_90_degrees : cos_90_degrees = 0 :=
by
  -- Proof would go here, but we skip it with sorry
  sorry

end cos_of_90_degrees_l182_182166


namespace remainder_division_l182_182771

theorem remainder_division (x : ℂ) (β : ℂ) (hβ : β^7 = 1) :
  (x^6 + x^5 + x^4 + x^3 + x^2 + x + 1) = 0 ->
  (x^63 + x^49 + x^35 + x^14 + 1) % (x^6 + x^5 + x^4 + x^3 + x^2 + x + 1) = 5 :=
by
  intro h
  sorry

end remainder_division_l182_182771


namespace value_of_f_at_2_l182_182467

def f (x : ℝ) : ℝ := 2 * x^4 + 3 * x^3 + 5 * x - 4

theorem value_of_f_at_2 : f 2 = 62 :=
by
  -- The proof will be inserted here, it follows Horner's method steps shown in the solution
  sorry

end value_of_f_at_2_l182_182467


namespace cos_90_eq_0_l182_182131

theorem cos_90_eq_0 : Real.cos (Float.pi / 2) = 0 := 
sorry

end cos_90_eq_0_l182_182131


namespace cos_90_eq_zero_l182_182146

theorem cos_90_eq_zero : (cos 90) = 0 := by
  -- Condition: The angle 90 degrees corresponds to the point (0, 1) on the unit circle.
  -- We claim cos 90 degrees is 0
  sorry

end cos_90_eq_zero_l182_182146


namespace cycle_cost_price_l182_182737

theorem cycle_cost_price (SP : ℝ) (loss_percentage : ℝ) (C : ℝ) 
  (h1 : SP = 1360) 
  (h2 : loss_percentage = 0.15) :
  SP = (1 - loss_percentage) * C → C = 1600 :=
by
  sorry

end cycle_cost_price_l182_182737


namespace multiples_of_4_between_200_and_500_l182_182212
-- Import the necessary library

open Nat

theorem multiples_of_4_between_200_and_500 : 
  ∃ n, n = (500 / 4 - 200 / 4) :=
by
  sorry

end multiples_of_4_between_200_and_500_l182_182212


namespace four_digit_numbers_with_product_exceeds_10_l182_182632

theorem four_digit_numbers_with_product_exceeds_10 : 
  (∃ count : ℕ, 
    count = (6 * 58 * 10) ∧
    count = 3480) := 
by {
  sorry
}

end four_digit_numbers_with_product_exceeds_10_l182_182632


namespace overlap_length_l182_182860

noncomputable def total_length_red_segments : ℝ := 98
noncomputable def total_spanning_distance : ℝ := 83
noncomputable def number_of_overlaps : ℕ := 6

theorem overlap_length (x : ℝ) : number_of_overlaps * x = total_length_red_segments - total_spanning_distance → 
  x = (total_length_red_segments - total_spanning_distance) / number_of_overlaps := 
  by
    sorry

end overlap_length_l182_182860


namespace amber_age_l182_182900

theorem amber_age 
  (a g : ℕ)
  (h1 : g = 15 * a)
  (h2 : g - a = 70) :
  a = 5 :=
by
  sorry

end amber_age_l182_182900


namespace mikes_age_is_18_l182_182383

-- Define variables for Mike's age (m) and his uncle's age (u)
variables (m u : ℕ)

-- Condition 1: Mike is 18 years younger than his uncle
def condition1 : Prop := m = u - 18

-- Condition 2: The sum of their ages is 54 years
def condition2 : Prop := m + u = 54

-- Statement: Prove that Mike's age is 18 given the conditions
theorem mikes_age_is_18 (h1 : condition1 m u) (h2 : condition2 m u) : m = 18 :=
by
  -- Proof skipped with sorry
  sorry

end mikes_age_is_18_l182_182383


namespace correct_ignition_time_l182_182695

noncomputable def ignition_time_satisfying_condition (initial_length : ℝ) (l : ℝ) : ℕ :=
  let burn_rate1 := l / 240
  let burn_rate2 := l / 360
  let stub1 t := l - burn_rate1 * t
  let stub2 t := l - burn_rate2 * t
  let stub_length_condition t := stub2 t = 3 * stub1 t
  let time_difference_at_6AM := 360 -- 6 AM is 360 minutes after midnight
  360 - 180 -- time to ignite the candles

theorem correct_ignition_time : ignition_time_satisfying_condition l 6 = 180 := 
by sorry

end correct_ignition_time_l182_182695


namespace area_of_region_bounded_by_circle_l182_182915

theorem area_of_region_bounded_by_circle :
  (∃ x y : ℝ, x^2 + y^2 + 2 * x - 4 * y + 9 = 0) →
  ∃ (area : ℝ), area = 4 * Real.pi :=
by
  sorry

end area_of_region_bounded_by_circle_l182_182915


namespace perpendicular_vectors_dot_product_zero_l182_182839

theorem perpendicular_vectors_dot_product_zero (m : ℝ) :
  let a := (1, 2)
  let b := (m + 1, -m)
  (a.1 * b.1 + a.2 * b.2 = 0) → m = 1 :=
by
  intros a b h_eq
  sorry

end perpendicular_vectors_dot_product_zero_l182_182839


namespace find_x_l182_182351

variable (A B : Set ℕ)
variable (x : ℕ)

theorem find_x (hA : A = {1, 3}) (hB : B = {2, x}) (hUnion : A ∪ B = {1, 2, 3, 4}) : x = 4 := by
  sorry

end find_x_l182_182351


namespace additional_track_length_l182_182093

theorem additional_track_length (h : ℝ) (g1 g2 : ℝ) (L1 L2 : ℝ)
  (rise_eq : h = 800) 
  (orig_grade : g1 = 0.04) 
  (new_grade : g2 = 0.025) 
  (L1_eq : L1 = h / g1) 
  (L2_eq : L2 = h / g2)
  : (L2 - L1 = 12000) := 
sorry

end additional_track_length_l182_182093
