import Mathlib

namespace NUMINAMATH_GPT_composite_of_n_gt_one_l1065_106549

theorem composite_of_n_gt_one (n : ℕ) (h : n > 1) : ∃ a b : ℕ, a > 1 ∧ b > 1 ∧ n^4 + 4 = a * b :=
by
  sorry

end NUMINAMATH_GPT_composite_of_n_gt_one_l1065_106549


namespace NUMINAMATH_GPT_vertex_of_f_C_l1065_106531

def f_A (x : ℝ) : ℝ := (x + 4) ^ 2 - 3
def f_B (x : ℝ) : ℝ := (x + 4) ^ 2 + 3
def f_C (x : ℝ) : ℝ := (x - 4) ^ 2 - 3
def f_D (x : ℝ) : ℝ := (x - 4) ^ 2 + 3

theorem vertex_of_f_C : ∃ (h k : ℝ), h = 4 ∧ k = -3 ∧ ∀ x, f_C x = (x - h) ^ 2 + k :=
by
  sorry

end NUMINAMATH_GPT_vertex_of_f_C_l1065_106531


namespace NUMINAMATH_GPT_find_range_of_a_l1065_106516

def quadratic_function (a x : ℝ) : ℝ :=
  x^2 + 2 * (a - 1) * x + 2

def is_decreasing_on (f : ℝ → ℝ) (I : Set ℝ) : Prop :=
  ∀ ⦃x y⦄, x ∈ I → y ∈ I → x ≤ y → f x ≥ f y

def is_increasing_on (f : ℝ → ℝ) (I : Set ℝ) : Prop :=
  ∀ ⦃x y⦄, x ∈ I → y ∈ I → x ≤ y → f x ≤ f y

def is_monotonic_on (f : ℝ → ℝ) (I : Set ℝ) : Prop :=
  is_decreasing_on f I ∨ is_increasing_on f I

theorem find_range_of_a (a : ℝ) :
  is_monotonic_on (quadratic_function a) (Set.Icc (-4) 4) ↔ (a ≤ -3 ∨ a ≥ 5) :=
sorry

end NUMINAMATH_GPT_find_range_of_a_l1065_106516


namespace NUMINAMATH_GPT_value_of_expression_l1065_106541

theorem value_of_expression (x y : ℤ) (hx : x = 2) (hy : y = 1) : 2 * x - 3 * y = 1 :=
by
  -- Substitute the given values into the expression and calculate
  sorry

end NUMINAMATH_GPT_value_of_expression_l1065_106541


namespace NUMINAMATH_GPT_average_of_three_numbers_is_165_l1065_106562

variable (x y z : ℕ)
variable (hy : y = 90)
variable (h1 : z = 4 * y)
variable (h2 : y = 2 * x)

theorem average_of_three_numbers_is_165 : (x + y + z) / 3 = 165 := by
  sorry

end NUMINAMATH_GPT_average_of_three_numbers_is_165_l1065_106562


namespace NUMINAMATH_GPT_lemons_needed_for_3_dozen_is_9_l1065_106500

-- Define the conditions
def lemon_tbs : ℕ := 4
def juice_needed_per_dozen : ℕ := 12
def dozens_needed : ℕ := 3
def total_juice_needed : ℕ := juice_needed_per_dozen * dozens_needed

-- The number of lemons needed to make 3 dozen cupcakes
def lemons_needed (total_juice : ℕ) (lemon_juice : ℕ) : ℕ :=
  total_juice / lemon_juice

-- Prove the number of lemons needed == 9
theorem lemons_needed_for_3_dozen_is_9 : lemons_needed total_juice_needed lemon_tbs = 9 :=
  by sorry

end NUMINAMATH_GPT_lemons_needed_for_3_dozen_is_9_l1065_106500


namespace NUMINAMATH_GPT_smallest_positive_multiple_l1065_106535

/-- Prove that the smallest positive multiple of 15 that is 7 more than a multiple of 65 is 255. -/
theorem smallest_positive_multiple : 
  ∃ n : ℕ, n > 0 ∧ n % 15 = 0 ∧ n % 65 = 7 ∧ n = 255 :=
sorry

end NUMINAMATH_GPT_smallest_positive_multiple_l1065_106535


namespace NUMINAMATH_GPT_sum_of_other_endpoint_coordinates_l1065_106570

theorem sum_of_other_endpoint_coordinates :
  ∃ (x y: ℤ), (8 + x) / 2 = 6 ∧ y / 2 = -10 ∧ x + y = -16 :=
by
  sorry

end NUMINAMATH_GPT_sum_of_other_endpoint_coordinates_l1065_106570


namespace NUMINAMATH_GPT_smallest_nonprime_in_range_l1065_106548

def smallest_nonprime_with_no_prime_factors_less_than_20 (m : ℕ) : Prop :=
  ¬(Nat.Prime m) ∧ m > 10 ∧ ∀ p : ℕ, Nat.Prime p → p < 20 → ¬(p ∣ m)

theorem smallest_nonprime_in_range :
  smallest_nonprime_with_no_prime_factors_less_than_20 529 ∧ 520 < 529 ∧ 529 ≤ 540 := 
by 
  sorry

end NUMINAMATH_GPT_smallest_nonprime_in_range_l1065_106548


namespace NUMINAMATH_GPT_fish_to_corn_value_l1065_106515

/-- In an island kingdom, five fish can be traded for three jars of honey, 
    and a jar of honey can be traded for six cobs of corn. 
    Prove that one fish is worth 3.6 cobs of corn. -/

theorem fish_to_corn_value (f h c : ℕ) (h1 : 5 * f = 3 * h) (h2 : h = 6 * c) : f = 18 * c / 5 := by
  sorry

end NUMINAMATH_GPT_fish_to_corn_value_l1065_106515


namespace NUMINAMATH_GPT_quadratic_real_roots_iff_find_m_given_condition_l1065_106527

noncomputable def quadratic_discriminant (a b c : ℝ) : ℝ := b^2 - 4 * a * c

noncomputable def roots (a b c : ℝ) : ℝ × ℝ :=
  let disc := quadratic_discriminant a b c
  if disc < 0 then (0, 0)
  else ((-b + disc.sqrt) / (2 * a), (-b - disc.sqrt) / (2 * a))

theorem quadratic_real_roots_iff (m : ℝ) :
  (quadratic_discriminant 1 (-2 * (m + 1)) (m ^ 2 + 5) ≥ 0) ↔ (m ≥ 2) :=
by sorry

theorem find_m_given_condition (x1 x2 m : ℝ) (h1 : x1 + x2 = 2 * (m + 1)) (h2 : x1 * x2 = m ^ 2 + 5) (h3 : (x1 - 1) * (x2 - 1) = 28) :
  m = 6 :=
by sorry

end NUMINAMATH_GPT_quadratic_real_roots_iff_find_m_given_condition_l1065_106527


namespace NUMINAMATH_GPT_simplify_expression_l1065_106501

variables (a b : ℝ) (h₁ : a ≠ 0) (h₂ : b ≠ 0) (h₃ : a ≠ b)

theorem simplify_expression :
  (a^4 - a^2 * b^2) / (a - b)^2 / (a * (a + b) / b^2) * (b^2 / a) = b^4 / (a - b) :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l1065_106501


namespace NUMINAMATH_GPT_min_value_of_x2_y2_sub_xy_l1065_106589

theorem min_value_of_x2_y2_sub_xy (x y : ℝ) (h : x^2 + y^2 + x * y = 315) : 
  ∃ m : ℝ, (∀ (u v : ℝ), u^2 + v^2 + u * v = 315 → u^2 + v^2 - u * v ≥ m) ∧ m = 105 :=
sorry

end NUMINAMATH_GPT_min_value_of_x2_y2_sub_xy_l1065_106589


namespace NUMINAMATH_GPT_benzene_carbon_mass_percentage_l1065_106523

noncomputable def carbon_mass_percentage_in_benzene 
  (carbon_atomic_mass : ℝ) (hydrogen_atomic_mass : ℝ) 
  (benzene_formula_ratio : (ℕ × ℕ)) : ℝ := 
    let (num_carbon_atoms, num_hydrogen_atoms) := benzene_formula_ratio
    let total_carbon_mass := num_carbon_atoms * carbon_atomic_mass
    let total_hydrogen_mass := num_hydrogen_atoms * hydrogen_atomic_mass
    let total_mass := total_carbon_mass + total_hydrogen_mass
    100 * (total_carbon_mass / total_mass)

theorem benzene_carbon_mass_percentage 
  (carbon_atomic_mass : ℝ := 12.01) 
  (hydrogen_atomic_mass : ℝ := 1.008) 
  (benzene_formula_ratio : (ℕ × ℕ) := (6, 6)) : 
    carbon_mass_percentage_in_benzene carbon_atomic_mass hydrogen_atomic_mass benzene_formula_ratio = 92.23 :=
by 
  unfold carbon_mass_percentage_in_benzene
  sorry

end NUMINAMATH_GPT_benzene_carbon_mass_percentage_l1065_106523


namespace NUMINAMATH_GPT_infinitely_many_pairs_l1065_106520

theorem infinitely_many_pairs : ∀ b : ℕ, ∃ a : ℕ, 2019 < 2^a / 3^b ∧ 2^a / 3^b < 2020 := 
by
  sorry

end NUMINAMATH_GPT_infinitely_many_pairs_l1065_106520


namespace NUMINAMATH_GPT_Max_students_count_l1065_106569

variables (M J : ℕ)

theorem Max_students_count :
  (M = 2 * J + 100) → 
  (M + J = 5400) → 
  M = 3632 := 
by 
  intros h1 h2
  sorry

end NUMINAMATH_GPT_Max_students_count_l1065_106569


namespace NUMINAMATH_GPT_apples_given_to_Larry_l1065_106522

-- Define the initial conditions
def initial_apples : ℕ := 75
def remaining_apples : ℕ := 23

-- The statement that we need to prove
theorem apples_given_to_Larry : initial_apples - remaining_apples = 52 :=
by
  -- skip the proof
  sorry

end NUMINAMATH_GPT_apples_given_to_Larry_l1065_106522


namespace NUMINAMATH_GPT_trajectory_of_intersection_l1065_106542

-- Define the conditions and question in Lean
structure Point where
  x : ℝ
  y : ℝ

def on_circle (C : Point) : Prop :=
  C.x^2 + C.y^2 = 1

def perp_to_x_axis (C D : Point) : Prop :=
  C.x = D.x ∧ C.y = -D.y

theorem trajectory_of_intersection (A B C D M : Point)
  (hA : A = {x := -1, y := 0})
  (hB : B = {x := 1, y := 0})
  (hC : on_circle C)
  (hD : on_circle D)
  (hCD : perp_to_x_axis C D)
  (hM : ∃ m n : ℝ, C = {x := m, y := n} ∧ M = {x := 1 / m, y := n / m}) :
  M.x^2 - M.y^2 = 1 ∧ M.y ≠ 0 :=
by
  sorry

end NUMINAMATH_GPT_trajectory_of_intersection_l1065_106542


namespace NUMINAMATH_GPT_validate_model_and_profit_range_l1065_106598

noncomputable def is_exponential_model_valid (x y : ℝ) : Prop :=
  ∃ T a : ℝ, T > 0 ∧ a > 1 ∧ y = T * a^x

noncomputable def is_profitable_for_at_least_one_billion (x : ℝ) : Prop :=
  (∃ T a : ℝ, T > 0 ∧ a > 1 ∧ 1/5 * (Real.sqrt 2)^x ≥ 10 ∧ 0 < x ∧ x ≤ 12) ∨
  (-0.2 * (x - 12) * (x - 17) + 12.8 ≥ 10 ∧ x > 12)

theorem validate_model_and_profit_range :
  (is_exponential_model_valid 2 0.4) ∧
  (is_exponential_model_valid 4 0.8) ∧
  (is_exponential_model_valid 12 12.8) ∧
  is_profitable_for_at_least_one_billion 11.3 ∧
  is_profitable_for_at_least_one_billion 19 :=
by
  sorry

end NUMINAMATH_GPT_validate_model_and_profit_range_l1065_106598


namespace NUMINAMATH_GPT_filling_tank_with_pipes_l1065_106550

theorem filling_tank_with_pipes :
  let Ra := 1 / 70
  let Rb := 2 * Ra
  let Rc := 2 * Rb
  let Rtotal := Ra + Rb + Rc
  Rtotal = 1 / 10 →  -- Given the combined rate fills the tank in 10 hours
  3 = 3 :=  -- Number of pipes used to fill the tank
by
  intros Ra Rb Rc Rtotal h
  simp [Ra, Rb, Rc] at h
  sorry

end NUMINAMATH_GPT_filling_tank_with_pipes_l1065_106550


namespace NUMINAMATH_GPT_tangent_product_constant_l1065_106573

variable (a x₁ x₂ y₁ y₂ : ℝ)

def point_on_parabola (x y : ℝ) := x^2 = 4 * y
def point_P := (a, -2)
def point_A := (x₁, y₁)
def point_B := (x₂, y₂)

theorem tangent_product_constant
  (h₁ : point_on_parabola x₁ y₁)
  (h₂ : point_on_parabola x₂ y₂)
  (h₃ : ∃ k₁ k₂ : ℝ, 
        (y₁ + 2 = k₁ * (x₁ - a) ∧ y₂ + 2 = k₂ * (x₂ - a)) 
        ∧ (k₁ * k₂ = -2)) :
  x₁ * x₂ + y₁ * y₂ = -4 :=
sorry

end NUMINAMATH_GPT_tangent_product_constant_l1065_106573


namespace NUMINAMATH_GPT_multiple_of_8_and_12_l1065_106566

theorem multiple_of_8_and_12 (x y : ℤ) (hx : ∃ k : ℤ, x = 8 * k) (hy : ∃ k : ℤ, y = 12 * k) :
  (∃ k : ℤ, y = 4 * k) ∧ (∃ k : ℤ, x - y = 4 * k) :=
by
  /- Proof goes here, based on the given conditions -/
  sorry

end NUMINAMATH_GPT_multiple_of_8_and_12_l1065_106566


namespace NUMINAMATH_GPT_Wilsons_number_l1065_106546

theorem Wilsons_number (N : ℝ) (h : N - N / 3 = 16 / 3) : N = 8 := sorry

end NUMINAMATH_GPT_Wilsons_number_l1065_106546


namespace NUMINAMATH_GPT_greatest_3_digit_base8_num_div_by_7_eq_511_l1065_106586

noncomputable def greatest_base8_number_divisible_by_7 : ℕ := 7 * 73

theorem greatest_3_digit_base8_num_div_by_7_eq_511 : 
  greatest_base8_number_divisible_by_7 = 511 :=
by 
  sorry

end NUMINAMATH_GPT_greatest_3_digit_base8_num_div_by_7_eq_511_l1065_106586


namespace NUMINAMATH_GPT_three_digit_integers_count_l1065_106526

theorem three_digit_integers_count : 
  ∃ (n : ℕ), n = 24 ∧
  (∃ (digits : Finset ℕ), digits = {2, 4, 7, 9} ∧
  (∀ a b c : ℕ, a ∈ digits → b ∈ digits → c ∈ digits → a ≠ b → b ≠ c → a ≠ c → 
  100 * a + 10 * b + c ∈ {y | 100 ≤ y ∧ y < 1000} → 4 * 3 * 2 = 24)) :=
by
  sorry

end NUMINAMATH_GPT_three_digit_integers_count_l1065_106526


namespace NUMINAMATH_GPT_log_sum_equality_l1065_106559

noncomputable def evaluate_log_sum : ℝ :=
  3 / (Real.log 1000^4 / Real.log 8) + 4 / (Real.log 1000^4 / Real.log 10)

theorem log_sum_equality :
  evaluate_log_sum = (9 * Real.log 2 / Real.log 10 + 4) / 12 :=
by
  sorry

end NUMINAMATH_GPT_log_sum_equality_l1065_106559


namespace NUMINAMATH_GPT_sampling_methods_suitability_l1065_106528

-- Define sample sizes and population sizes
def n1 := 2  -- Number of students to be selected in sample ①
def N1 := 10  -- Population size for sample ①
def n2 := 50  -- Number of students to be selected in sample ②
def N2 := 1000  -- Population size for sample ②

-- Define what it means for a sampling method to be suitable
def is_simple_random_sampling_suitable (n N : Nat) : Prop :=
  N <= 50 ∧ n < N

def is_systematic_sampling_suitable (n N : Nat) : Prop :=
  N > 50 ∧ n < N ∧ n ≥ 50 / 1000 * N  -- Ensuring suitable systematic sampling size

-- The proof statement
theorem sampling_methods_suitability :
  is_simple_random_sampling_suitable n1 N1 ∧ is_systematic_sampling_suitable n2 N2 :=
by
  -- Sorry blocks are used to skip the proofs
  sorry

end NUMINAMATH_GPT_sampling_methods_suitability_l1065_106528


namespace NUMINAMATH_GPT_students_between_jimin_yuna_l1065_106579

theorem students_between_jimin_yuna 
  (total_students : ℕ) 
  (jimin_position : ℕ) 
  (yuna_position : ℕ) 
  (h1 : total_students = 32) 
  (h2 : jimin_position = 27) 
  (h3 : yuna_position = 11) 
  : (jimin_position - yuna_position - 1) = 15 := 
by
  sorry

end NUMINAMATH_GPT_students_between_jimin_yuna_l1065_106579


namespace NUMINAMATH_GPT_tina_sales_ratio_l1065_106551

theorem tina_sales_ratio (katya_sales ricky_sales t_sold_more : ℕ) 
  (h_katya : katya_sales = 8) 
  (h_ricky : ricky_sales = 9) 
  (h_tina_sold : t_sold_more = katya_sales + 26) 
  (h_tina_multiple : ∃ m : ℕ, t_sold_more = m * (katya_sales + ricky_sales)) :
  t_sold_more / (katya_sales + ricky_sales) = 2 := 
by 
  sorry

end NUMINAMATH_GPT_tina_sales_ratio_l1065_106551


namespace NUMINAMATH_GPT_KellyGamesLeft_l1065_106513

def initialGames : ℕ := 121
def gamesGivenAway : ℕ := 99

theorem KellyGamesLeft : initialGames - gamesGivenAway = 22 := by
  sorry

end NUMINAMATH_GPT_KellyGamesLeft_l1065_106513


namespace NUMINAMATH_GPT_find_time_same_height_l1065_106529

noncomputable def height_ball (t : ℝ) : ℝ := 60 - 9 * t - 8 * t^2
noncomputable def height_bird (t : ℝ) : ℝ := 3 * t^2 + 4 * t

theorem find_time_same_height : ∃ t : ℝ, t = 20 / 11 ∧ height_ball t = height_bird t := 
by
  use 20 / 11
  sorry

end NUMINAMATH_GPT_find_time_same_height_l1065_106529


namespace NUMINAMATH_GPT_sqrt_of_sixteen_l1065_106514

theorem sqrt_of_sixteen (x : ℝ) (h : x^2 = 16) : x = 4 ∨ x = -4 := 
sorry

end NUMINAMATH_GPT_sqrt_of_sixteen_l1065_106514


namespace NUMINAMATH_GPT_passes_through_fixed_point_l1065_106585

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  a^(x-2) - 3

theorem passes_through_fixed_point (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  f a 2 = -2 :=
by
  sorry

end NUMINAMATH_GPT_passes_through_fixed_point_l1065_106585


namespace NUMINAMATH_GPT_geometric_series_sum_l1065_106502

/-- The first term of the geometric series. -/
def a : ℚ := 3

/-- The common ratio of the geometric series. -/
def r : ℚ := -3 / 4

/-- The sum of the geometric series is equal to 12/7. -/
theorem geometric_series_sum : (∑' n : ℕ, a * r^n) = 12 / 7 := 
by
  /- The Sum function and its properties for the geometric series will be used here. -/
  sorry

end NUMINAMATH_GPT_geometric_series_sum_l1065_106502


namespace NUMINAMATH_GPT_value_of_expression_l1065_106555

theorem value_of_expression : (2112 - 2021) ^ 2 / 169 = 49 := by
  sorry

end NUMINAMATH_GPT_value_of_expression_l1065_106555


namespace NUMINAMATH_GPT_beam_equation_correctness_l1065_106571

-- Define the conditions
def total_selling_price : ℕ := 6210
def freight_per_beam : ℕ := 3

-- Define the unknown quantity
variable (x : ℕ)

-- State the theorem
theorem beam_equation_correctness
  (h1 : total_selling_price = 6210)
  (h2 : freight_per_beam = 3) :
  freight_per_beam * (x - 1) = total_selling_price / x := 
sorry

end NUMINAMATH_GPT_beam_equation_correctness_l1065_106571


namespace NUMINAMATH_GPT_scientific_notation_of_star_diameter_l1065_106582

theorem scientific_notation_of_star_diameter:
    (∃ (c : ℝ) (n : ℕ), 1 ≤ c ∧ c < 10 ∧ 16600000000 = c * 10^n) → 
    16600000000 = 1.66 * 10^10 :=
by
  sorry

end NUMINAMATH_GPT_scientific_notation_of_star_diameter_l1065_106582


namespace NUMINAMATH_GPT_find_s_l_l1065_106583

theorem find_s_l :
  ∃ s l : ℝ, ∀ t : ℝ, 
  (-8 + l * t, s + -6 * t) ∈ {p : ℝ × ℝ | ∃ x : ℝ, p.snd = 3 / 4 * x + 2 ∧ p.fst = x} ∧ 
  (s = -4 ∧ l = -8) :=
by
  sorry

end NUMINAMATH_GPT_find_s_l_l1065_106583


namespace NUMINAMATH_GPT_difference_students_rabbits_l1065_106557

-- Define the number of students per classroom
def students_per_classroom := 22

-- Define the number of rabbits per classroom
def rabbits_per_classroom := 4

-- Define the number of classrooms
def classrooms := 6

-- Calculate the total number of students
def total_students := students_per_classroom * classrooms

-- Calculate the total number of rabbits
def total_rabbits := rabbits_per_classroom * classrooms

-- Prove the difference between the number of students and rabbits is 108
theorem difference_students_rabbits : total_students - total_rabbits = 108 := by
  sorry

end NUMINAMATH_GPT_difference_students_rabbits_l1065_106557


namespace NUMINAMATH_GPT_count_yellow_highlighters_l1065_106545

-- Definitions of the conditions
def pink_highlighters : ℕ := 9
def blue_highlighters : ℕ := 5
def total_highlighters : ℕ := 22

-- Definition based on the question
def yellow_highlighters : ℕ := total_highlighters - (pink_highlighters + blue_highlighters)

-- The theorem to prove the number of yellow highlighters
theorem count_yellow_highlighters : yellow_highlighters = 8 :=
by
  -- Proof omitted as instructed
  sorry

end NUMINAMATH_GPT_count_yellow_highlighters_l1065_106545


namespace NUMINAMATH_GPT_selection_methods_l1065_106511

theorem selection_methods :
  ∃ (ways_with_girls : ℕ), ways_with_girls = Nat.choose 6 4 - Nat.choose 4 4 ∧ ways_with_girls = 14 := by
  sorry

end NUMINAMATH_GPT_selection_methods_l1065_106511


namespace NUMINAMATH_GPT_lesser_fraction_exists_l1065_106509

theorem lesser_fraction_exists (x y : ℚ) (h_sum : x + y = 3/4) (h_prod : x * y = 1/8) : x = 1/4 ∨ y = 1/4 := by
  sorry

end NUMINAMATH_GPT_lesser_fraction_exists_l1065_106509


namespace NUMINAMATH_GPT_exists_solution_negation_correct_l1065_106540

theorem exists_solution_negation_correct :
  (∃ x : ℝ, x^2 - x = 0) ↔ (∃ x : ℝ, True) ∧ (∀ x : ℝ, ¬ (x^2 - x = 0)) :=
by
  sorry

end NUMINAMATH_GPT_exists_solution_negation_correct_l1065_106540


namespace NUMINAMATH_GPT_area_covered_by_congruent_rectangles_l1065_106510

-- Definitions of conditions
def length_AB : ℕ := 12
def width_AD : ℕ := 8
def area_rect (l w : ℕ) : ℕ := l * w

-- Center of the first rectangle
def center_ABCD : ℕ × ℕ := (length_AB / 2, width_AD / 2)

-- Proof statement
theorem area_covered_by_congruent_rectangles 
  (length_ABCD length_EFGH width_ABCD width_EFGH : ℕ)
  (congruent : length_ABCD = length_EFGH ∧ width_ABCD = width_EFGH)
  (center_E : ℕ × ℕ)
  (H_center_E : center_E = center_ABCD) :
  area_rect length_ABCD width_ABCD + area_rect length_EFGH width_EFGH - length_ABCD * width_ABCD / 2 = 168 := by
  sorry

end NUMINAMATH_GPT_area_covered_by_congruent_rectangles_l1065_106510


namespace NUMINAMATH_GPT_triangle_side_length_range_l1065_106599

theorem triangle_side_length_range (x : ℝ) : 
  (1 < x) ∧ (x < 9) → ¬ (x = 10) :=
by
  sorry

end NUMINAMATH_GPT_triangle_side_length_range_l1065_106599


namespace NUMINAMATH_GPT_tangent_line_eq_l1065_106524

theorem tangent_line_eq (x y : ℝ) (h_curve : y = x^3 - x + 1) (h_point : (x, y) = (0, 1)) : x + y - 1 = 0 := 
sorry

end NUMINAMATH_GPT_tangent_line_eq_l1065_106524


namespace NUMINAMATH_GPT_no_positive_integer_solutions_l1065_106587

theorem no_positive_integer_solutions :
  ∀ (A : ℕ), 1 ≤ A ∧ A ≤ 9 → ¬∃ x y : ℕ, x > 0 ∧ y > 0 ∧ x * y = A * 10 + A ∧ x + y = 10 * A + 1 := by
  sorry

end NUMINAMATH_GPT_no_positive_integer_solutions_l1065_106587


namespace NUMINAMATH_GPT_find_f_x_l1065_106505

theorem find_f_x (f : ℤ → ℤ) (h : ∀ x : ℤ, f (x + 1) = 2 * x - 1) : 
  ∀ x : ℤ, f x = 2 * x - 3 :=
sorry

end NUMINAMATH_GPT_find_f_x_l1065_106505


namespace NUMINAMATH_GPT_correct_result_after_mistakes_l1065_106536

theorem correct_result_after_mistakes (n : ℕ) (f : ℕ → ℕ → ℕ) (g : ℕ → ℕ → ℕ)
    (h1 : f n 4 * 4 + 18 = g 12 18) : 
    g (f n 4 * 4) 18 = 498 :=
by
  sorry

end NUMINAMATH_GPT_correct_result_after_mistakes_l1065_106536


namespace NUMINAMATH_GPT_rate_of_interest_l1065_106508

theorem rate_of_interest (P A T SI : ℝ) (h1 : P = 750) (h2 : A = 900) (h3 : T = 2)
  (h4 : SI = A - P) (h5 : SI = (P * R * T) / 100) : R = 10 :=
by
  sorry

end NUMINAMATH_GPT_rate_of_interest_l1065_106508


namespace NUMINAMATH_GPT_find_intended_number_l1065_106595

theorem find_intended_number (x : ℕ) 
    (condition : 3 * x = (10 * 3 * x + 2) / 19 + 7) : 
    x = 5 :=
sorry

end NUMINAMATH_GPT_find_intended_number_l1065_106595


namespace NUMINAMATH_GPT_kelly_can_buy_ten_pounds_of_mangoes_l1065_106512

theorem kelly_can_buy_ten_pounds_of_mangoes (h : 0.5 * 1.2 = 0.60) : 12 / (2 * 0.60) = 10 :=
  by
    sorry

end NUMINAMATH_GPT_kelly_can_buy_ten_pounds_of_mangoes_l1065_106512


namespace NUMINAMATH_GPT_find_x_solutions_l1065_106521

theorem find_x_solutions (x : ℝ) :
  let f (x : ℝ) := x^2 - 4*x + 1
  let f2 (x : ℝ) := (f x)^2
  f (f x) = f2 x ↔ x = 2 + (Real.sqrt 13) / 2 ∨ x = 2 - (Real.sqrt 13) / 2 := by
  sorry

end NUMINAMATH_GPT_find_x_solutions_l1065_106521


namespace NUMINAMATH_GPT_intercept_condition_l1065_106596

theorem intercept_condition (a b c : ℝ) (h_nonzero : a ≠ 0 ∧ b ≠ 0) :
  (∃ x y : ℝ, a * x + b * y + c = 0 ∧ x = -c / a ∧ y = -c / b ∧ x = y) → (c = 0 ∨ a = b) :=
by
  sorry

end NUMINAMATH_GPT_intercept_condition_l1065_106596


namespace NUMINAMATH_GPT_combined_weight_of_two_new_students_l1065_106568

theorem combined_weight_of_two_new_students (W : ℕ) (X : ℕ) 
  (cond1 : (W - 150 + X) / 8 = (W / 8) - 2) :
  X = 134 := 
sorry

end NUMINAMATH_GPT_combined_weight_of_two_new_students_l1065_106568


namespace NUMINAMATH_GPT_find_number_l1065_106547

theorem find_number (x : ℝ) (h : 5020 - 502 / x = 5015) : x = 100.4 :=
by
  sorry

end NUMINAMATH_GPT_find_number_l1065_106547


namespace NUMINAMATH_GPT_smallest_n_watches_l1065_106578

variable {n d : ℕ}

theorem smallest_n_watches (h1 : d > 0)
  (h2 : 10 * n - 30 = 100) : n = 13 :=
by
  sorry

end NUMINAMATH_GPT_smallest_n_watches_l1065_106578


namespace NUMINAMATH_GPT_ball_returns_to_bella_after_13_throws_l1065_106591

theorem ball_returns_to_bella_after_13_throws:
  ∀ (girls : Fin 13) (n : ℕ), (∃ k, k > 0 ∧ (1 + k * 5) % 13 = 1) → (n = 13) :=
by
  sorry

end NUMINAMATH_GPT_ball_returns_to_bella_after_13_throws_l1065_106591


namespace NUMINAMATH_GPT_square_distance_from_B_to_center_l1065_106507

noncomputable def distance_squared (a b : ℝ) : ℝ := a^2 + b^2

theorem square_distance_from_B_to_center :
  ∀ (a b : ℝ),
    (a^2 + (b + 8)^2 = 75) →
    ((a + 2)^2 + b^2 = 75) →
    distance_squared a b = 122 :=
by
  intros a b h1 h2
  sorry

end NUMINAMATH_GPT_square_distance_from_B_to_center_l1065_106507


namespace NUMINAMATH_GPT_average_hours_l1065_106564

def hours_studied (week1 week2 week3 week4 week5 week6 week7 : ℕ) : ℕ :=
  week1 + week2 + week3 + week4 + week5 + week6 + week7

theorem average_hours (x : ℕ)
  (h1 : hours_studied 8 10 9 11 10 7 x / 7 = 9) :
  x = 8 :=
by
  sorry

end NUMINAMATH_GPT_average_hours_l1065_106564


namespace NUMINAMATH_GPT_square_triangle_same_area_l1065_106556

theorem square_triangle_same_area (perimeter_square height_triangle : ℤ) (same_area : ℚ) 
  (h_perimeter_square : perimeter_square = 64) 
  (h_height_triangle : height_triangle = 64)
  (h_same_area : same_area = 256) :
  ∃ x : ℚ, x = 8 :=
by
  sorry

end NUMINAMATH_GPT_square_triangle_same_area_l1065_106556


namespace NUMINAMATH_GPT_find_a_plus_b_l1065_106534

noncomputable def f (a b x : ℝ) := a * x + b
noncomputable def g (x : ℝ) := 3 * x - 4

theorem find_a_plus_b (a b : ℝ) (h : ∀ (x : ℝ), g (f a b x) = 4 * x + 5) : a + b = 13 / 3 := 
  sorry

end NUMINAMATH_GPT_find_a_plus_b_l1065_106534


namespace NUMINAMATH_GPT_distinct_roots_iff_l1065_106554

def has_two_distinct_roots (a : ℝ) : Prop :=
  ∃ x1 x2 : ℝ, x1 ≠ x2 ∧ (x1 + |x1| = 2 * Real.sqrt (3 + 2*a*x1 - 4*a)) ∧
                       (x2 + |x2| = 2 * Real.sqrt (3 + 2*a*x2 - 4*a))

theorem distinct_roots_iff (a : ℝ) :
  has_two_distinct_roots a ↔ (a ∈ Set.Ioo 0 (3 / 4 : ℝ) ∨ 3 < a) :=
sorry

end NUMINAMATH_GPT_distinct_roots_iff_l1065_106554


namespace NUMINAMATH_GPT_smallest_unit_of_money_correct_l1065_106593

noncomputable def smallest_unit_of_money (friends : ℕ) (total_bill paid_amount : ℚ) : ℚ :=
  if (total_bill % friends : ℚ) = 0 then
    total_bill / friends
  else
    1 % 100

theorem smallest_unit_of_money_correct :
  smallest_unit_of_money 9 124.15 124.11 = 1 % 100 := 
by
  sorry

end NUMINAMATH_GPT_smallest_unit_of_money_correct_l1065_106593


namespace NUMINAMATH_GPT_one_cow_one_bag_in_46_days_l1065_106565

-- Defining the conditions
def cows_eat_husk (n_cows n_bags n_days : ℕ) := n_cows = n_bags ∧ n_cows = n_days ∧ n_bags = n_days

-- The main theorem to be proved
theorem one_cow_one_bag_in_46_days (h : cows_eat_husk 46 46 46) : 46 = 46 := by
  sorry

end NUMINAMATH_GPT_one_cow_one_bag_in_46_days_l1065_106565


namespace NUMINAMATH_GPT_unicorn_rope_length_l1065_106580

noncomputable def a : ℕ := 90
noncomputable def b : ℕ := 1500
noncomputable def c : ℕ := 3

theorem unicorn_rope_length : a + b + c = 1593 :=
by
  -- The steps to prove the theorem should go here, but as stated, we skip this with "sorry".
  sorry

end NUMINAMATH_GPT_unicorn_rope_length_l1065_106580


namespace NUMINAMATH_GPT_function_solution_l1065_106553

noncomputable def f (c : ℝ) (x : ℝ) : ℝ :=
if x = 0 then c else if x = 1 then 3 - 2 * c else (-x^3 + 3 * x^2 + 2) / (3 * x * (1 - x))

theorem function_solution (f : ℝ → ℝ) :
  (∀ x ≠ 0, f x + 2 * f ((x - 1) / x) = 3 * x) →
  (∃ c : ℝ, ∀ x : ℝ, f x = if x = 0 then c else if x = 1 then 3 - 2 * c else (-x^3 + 3 * x^2 + 2) / (3 * x * (1 - x))) :=
by
  intro h
  use (f 0)
  intro x
  split_ifs with h0 h1
  rotate_left -- to handle the cases x ≠ 0, 1 at first.
  sorry -- Additional proof steps required here.
  sorry -- Use the given conditions and functional equation to conclude f(0) = c.
  sorry -- Use the given conditions and functional equation to conclude f(1) = 3 - 2c.

end NUMINAMATH_GPT_function_solution_l1065_106553


namespace NUMINAMATH_GPT_range_of_a_l1065_106538

-- Define the assumptions and target proof
theorem range_of_a {f : ℝ → ℝ}
  (h_odd : ∀ x : ℝ, f (-x) = -f x)
  (h_monotonic : ∀ x1 x2 : ℝ, x1 < x2 → f x1 > f x2)
  (h_condition : ∀ a : ℝ, f (2 - a) + f (4 - a) < 0)
  : ∀ a : ℝ, f (2 - a) + f (4 - a) < 0 → a < 3 :=
by
  intro a h
  sorry

end NUMINAMATH_GPT_range_of_a_l1065_106538


namespace NUMINAMATH_GPT_possible_to_position_guards_l1065_106543

-- Define the conditions
def guard_sees (d : ℝ) : Prop := d = 100

-- Prove that it is possible to arrange guards around a point object so that neither the object nor the guards can be approached unnoticed
theorem possible_to_position_guards (num_guards : ℕ) (d : ℝ) (h : guard_sees d) : 
  (0 < num_guards) → 
  (∀ θ : ℕ, θ < num_guards → (θ * (360 / num_guards)) < 360) → 
  True :=
by 
  -- Details of the proof would go here
  sorry

end NUMINAMATH_GPT_possible_to_position_guards_l1065_106543


namespace NUMINAMATH_GPT_find_x_l1065_106504

-- Definitions based on conditions
def parabola_eq (y x p : ℝ) : Prop := y^2 = 2 * p * x
def point_on_parabola (p : ℝ) : Prop := ∃ y x, parabola_eq y x p ∧ (x = 1) ∧ (y = 2)
def valid_p (p : ℝ) : Prop := p > 0
def dist_to_focus (x : ℝ) : ℝ := 1
def dist_to_line (x : ℝ) : ℝ := abs (x + 1)

-- Main statement to be proven
theorem find_x (p : ℝ) (h1 : point_on_parabola p) (h2 : valid_p p) :
  ∃ x, dist_to_focus x = dist_to_line x ∧ x = 1 :=
sorry

end NUMINAMATH_GPT_find_x_l1065_106504


namespace NUMINAMATH_GPT_prism_surface_area_l1065_106572

theorem prism_surface_area (a : ℝ) : 
  let surface_area_cubes := 6 * a^2 * 3
  let surface_area_shared_faces := 4 * a^2
  surface_area_cubes - surface_area_shared_faces = 14 * a^2 := 
by 
  let surface_area_cubes := 6 * a^2 * 3
  let surface_area_shared_faces := 4 * a^2
  have : surface_area_cubes - surface_area_shared_faces = 14 * a^2 := sorry
  exact this

end NUMINAMATH_GPT_prism_surface_area_l1065_106572


namespace NUMINAMATH_GPT_crease_length_l1065_106567

noncomputable def length_of_crease (theta : ℝ) : ℝ :=
  8 * Real.sin theta

theorem crease_length (theta : ℝ) (hθ : 0 ≤ theta ∧ theta ≤ π / 2) : 
  length_of_crease theta = 8 * Real.sin theta :=
by sorry

end NUMINAMATH_GPT_crease_length_l1065_106567


namespace NUMINAMATH_GPT_max_apartment_size_l1065_106537

theorem max_apartment_size (rental_price_per_sqft : ℝ) (budget : ℝ) (h1 : rental_price_per_sqft = 1.20) (h2 : budget = 720) : 
  budget / rental_price_per_sqft = 600 :=
by 
  sorry

end NUMINAMATH_GPT_max_apartment_size_l1065_106537


namespace NUMINAMATH_GPT_arithmetic_sequence_length_correct_l1065_106581

noncomputable def arithmetic_sequence_length (a d last_term : ℕ) : ℕ :=
  ((last_term - a) / d) + 1

theorem arithmetic_sequence_length_correct :
  arithmetic_sequence_length 2 3 2014 = 671 :=
by
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_length_correct_l1065_106581


namespace NUMINAMATH_GPT_sid_spent_on_snacks_l1065_106563

theorem sid_spent_on_snacks :
  let original_money := 48
  let money_spent_on_computer_accessories := 12
  let money_left_after_computer_accessories := original_money - money_spent_on_computer_accessories
  let remaining_money_after_purchases := 4 + original_money / 2
  ∃ snacks_cost, money_left_after_computer_accessories - snacks_cost = remaining_money_after_purchases ∧ snacks_cost = 8 :=
by
  sorry

end NUMINAMATH_GPT_sid_spent_on_snacks_l1065_106563


namespace NUMINAMATH_GPT_find_x_for_equation_l1065_106533

theorem find_x_for_equation 
  (x : ℝ)
  (h : (32 : ℝ)^(x-2) / (8 : ℝ)^(x-2) = (512 : ℝ)^(3 * x)) : 
  x = -4/25 :=
by
  sorry

end NUMINAMATH_GPT_find_x_for_equation_l1065_106533


namespace NUMINAMATH_GPT_triangle_abs_simplification_l1065_106588

theorem triangle_abs_simplification
  (x y z : ℝ)
  (h1 : x + y > z)
  (h2 : y + z > x)
  (h3 : x + z > y) :
  |x + y - z| - 2 * |y - x - z| = -x + 3 * y - 3 * z :=
by
  sorry

end NUMINAMATH_GPT_triangle_abs_simplification_l1065_106588


namespace NUMINAMATH_GPT_area_percentage_of_smaller_square_l1065_106519

theorem area_percentage_of_smaller_square 
  (radius : ℝ)
  (a A O B: ℝ)
  (side_length_larger_square side_length_smaller_square : ℝ) 
  (hyp1 : side_length_larger_square = 4)
  (hyp2 : radius = 2 * Real.sqrt 2)
  (hyp3 : a = 4) 
  (hyp4 : A = 2 + side_length_smaller_square / 4)
  (hyp5 : O = 2 * Real.sqrt 2)
  (hyp6 : side_length_smaller_square = 0.8) :
  (side_length_smaller_square^2 / side_length_larger_square^2) = 0.04 :=
by
  sorry

end NUMINAMATH_GPT_area_percentage_of_smaller_square_l1065_106519


namespace NUMINAMATH_GPT_ratio_of_girls_to_boys_l1065_106518

theorem ratio_of_girls_to_boys (total_students girls boys : ℕ) (h_ratio : girls = 4 * (girls + boys) / 7) (h_total : total_students = 70) : 
  girls = 40 ∧ boys = 30 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_girls_to_boys_l1065_106518


namespace NUMINAMATH_GPT_shortest_time_to_camp_l1065_106575

/-- 
Given:
- The width of the river is 1 km.
- The camp is 1 km away from the point directly across the river.
- Swimming speed is 2 km/hr.
- Walking speed is 3 km/hr.

Prove the shortest time required to reach the camp is (2 + √5) / 6 hours.
--/
theorem shortest_time_to_camp :
  ∃ t : ℝ, t = (2 + Real.sqrt 5) / 6 := 
sorry

end NUMINAMATH_GPT_shortest_time_to_camp_l1065_106575


namespace NUMINAMATH_GPT_digit_68th_is_1_l1065_106517

noncomputable def largest_n : ℕ :=
  (10^100 - 1) / 14

def digit_at_68th_place (n : ℕ) : ℕ :=
  (n / 10^(68 - 1)) % 10

theorem digit_68th_is_1 : digit_at_68th_place largest_n = 1 :=
sorry

end NUMINAMATH_GPT_digit_68th_is_1_l1065_106517


namespace NUMINAMATH_GPT_pure_imaginary_solution_l1065_106594

-- Defining the main problem as a theorem in Lean 4

theorem pure_imaginary_solution (m : ℝ) : 
  (∃ a b : ℝ, (m^2 - m = a ∧ a = 0) ∧ (m^2 - 3 * m + 2 = b ∧ b ≠ 0)) → 
  m = 0 :=
sorry -- Proof is omitted as per the instructions

end NUMINAMATH_GPT_pure_imaginary_solution_l1065_106594


namespace NUMINAMATH_GPT_simplify_and_evaluate_expression_l1065_106592

theorem simplify_and_evaluate_expression : 
  ∀ a : ℚ, a = -1/2 → (a + 3)^2 - (a + 1) * (a - 1) - 2 * (2 * a + 4) = 1 := 
by
  intro a ha
  simp only [ha]
  sorry

end NUMINAMATH_GPT_simplify_and_evaluate_expression_l1065_106592


namespace NUMINAMATH_GPT_necessary_but_not_sufficient_condition_l1065_106525

noncomputable def condition (m : ℝ) : Prop := 1 < m ∧ m < 3

def represents_ellipse (m : ℝ) (x y : ℝ) : Prop :=
  (x ^ 2) / (m - 1) + (y ^ 2) / (3 - m) = 1

theorem necessary_but_not_sufficient_condition (m : ℝ) :
  (∃ x y, represents_ellipse m x y) → condition m :=
sorry

end NUMINAMATH_GPT_necessary_but_not_sufficient_condition_l1065_106525


namespace NUMINAMATH_GPT_evaluate_expression_l1065_106590

theorem evaluate_expression : 
  let a := 45
  let b := 15
  (a + b)^2 - (a^2 + b^2 + 2 * a * 5) = 900 :=
by
  let a := 45
  let b := 15
  sorry

end NUMINAMATH_GPT_evaluate_expression_l1065_106590


namespace NUMINAMATH_GPT_shaded_area_is_correct_l1065_106539

-- Conditions definition
def shaded_numbers : ℕ := 2015
def boundary_properties (segment : ℕ) : Prop := 
  segment = 1 ∨ segment = 2

theorem shaded_area_is_correct : ∀ n : ℕ, n = shaded_numbers → boundary_properties n → 
  (∃ area : ℚ, area = 47.5) :=
by
  sorry

end NUMINAMATH_GPT_shaded_area_is_correct_l1065_106539


namespace NUMINAMATH_GPT_min_performances_l1065_106576

theorem min_performances (total_singers : ℕ) (m : ℕ) (n_pairs : ℕ := 28) (pairs_performance : ℕ := 6)
  (condition : total_singers = 108) 
  (const_pairs : ∀ (r : ℕ), (n_pairs * r = pairs_performance * m)) : m ≥ 14 :=
by
  sorry

end NUMINAMATH_GPT_min_performances_l1065_106576


namespace NUMINAMATH_GPT_semi_circle_radius_l1065_106597

theorem semi_circle_radius (P : ℝ) (π : ℝ) (r : ℝ) (hP : P = 10.797344572538567) (hπ : π = 3.14159) :
  (π + 2) * r = P → r = 2.1 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_semi_circle_radius_l1065_106597


namespace NUMINAMATH_GPT_butter_needed_for_original_recipe_l1065_106561

-- Define the conditions
def butter_to_flour_ratio : ℚ := 12 / 56

def flour_for_original_recipe : ℚ := 14

def butter_for_original_recipe (ratio : ℚ) (flour : ℚ) : ℚ :=
  ratio * flour

-- State the theorem
theorem butter_needed_for_original_recipe :
  butter_for_original_recipe butter_to_flour_ratio flour_for_original_recipe = 3 := 
sorry

end NUMINAMATH_GPT_butter_needed_for_original_recipe_l1065_106561


namespace NUMINAMATH_GPT_interest_difference_l1065_106560

noncomputable def difference_between_interest (P : ℝ) (R : ℝ) (T : ℝ) (n : ℕ) : ℝ :=
  let SI := P * R * T / 100
  let CI := P * (1 + (R / (n*100)))^(n * T) - P
  CI - SI

theorem interest_difference (P : ℝ) (R : ℝ) (T : ℝ) (n : ℕ) (hP : P = 1200) (hR : R = 10) (hT : T = 1) (hn : n = 2) :
  difference_between_interest P R T n = -59.25 := by
  sorry

end NUMINAMATH_GPT_interest_difference_l1065_106560


namespace NUMINAMATH_GPT_find_principal_l1065_106552

-- Defining the conditions
def A : ℝ := 5292
def r : ℝ := 0.05
def n : ℝ := 1
def t : ℝ := 2

-- The theorem statement
theorem find_principal :
  ∃ (P : ℝ), A = P * (1 + r / n) ^ (n * t) ∧ P = 4800 :=
by
  sorry

end NUMINAMATH_GPT_find_principal_l1065_106552


namespace NUMINAMATH_GPT_smallest_prime_factor_in_setB_l1065_106584

def setB : Set ℕ := {55, 57, 58, 59, 61}

def smallest_prime_factor (n : ℕ) : ℕ :=
  if h : n = 2 then 2 else (Nat.minFac (Nat.pred n)).succ

theorem smallest_prime_factor_in_setB :
  ∃ n ∈ setB, smallest_prime_factor n = 2 := by
  sorry

end NUMINAMATH_GPT_smallest_prime_factor_in_setB_l1065_106584


namespace NUMINAMATH_GPT_problem_solution_l1065_106577

theorem problem_solution (a b c : ℝ) (h1 : a + b + c = 1) (h2 : a^2 + b^2 + c^2 = 2) (h3 : a^3 + b^3 + c^3 = 3) :
  (a * b * c = 1 / 6) ∧ (a^4 + b^4 + c^4 = 25 / 6) :=
by {
  sorry
}

end NUMINAMATH_GPT_problem_solution_l1065_106577


namespace NUMINAMATH_GPT_largest_of_three_numbers_l1065_106544

noncomputable def largest_root (p q r : ℝ) (h1 : p + q + r = 3) (h2 : p * q + p * r + q * r = -8) (h3 : p * q * r = -20) : ℝ :=
  max p (max q r)

theorem largest_of_three_numbers (p q r : ℝ) 
  (h1 : p + q + r = 3) 
  (h2 : p * q + p * r + q * r = -8) 
  (h3 : p * q * r = -20) :
  largest_root p q r h1 h2 h3 = ( -1 + Real.sqrt 21 ) / 2 :=
by
  sorry

end NUMINAMATH_GPT_largest_of_three_numbers_l1065_106544


namespace NUMINAMATH_GPT_sam_seashells_l1065_106532

def seashells_problem := 
  let mary_seashells := 47
  let total_seashells := 65
  (total_seashells - mary_seashells) = 18

theorem sam_seashells :
  seashells_problem :=
by
  sorry

end NUMINAMATH_GPT_sam_seashells_l1065_106532


namespace NUMINAMATH_GPT_ratio_equivalence_l1065_106506

theorem ratio_equivalence (x : ℝ) :
  ((20 / 10) * 100 = (25 / x) * 100) → x = 12.5 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_ratio_equivalence_l1065_106506


namespace NUMINAMATH_GPT_simplest_quadratic_radical_l1065_106530

theorem simplest_quadratic_radical :
  let a := Real.sqrt 12
  let b := Real.sqrt (2 / 3)
  let c := Real.sqrt 0.3
  let d := Real.sqrt 7
  d < a ∧ d < b ∧ d < c := 
by {
  -- the proof steps will go here, but we use sorry for now
  sorry
}

end NUMINAMATH_GPT_simplest_quadratic_radical_l1065_106530


namespace NUMINAMATH_GPT_sum_of_triangle_angles_sin_halves_leq_one_l1065_106503

theorem sum_of_triangle_angles_sin_halves_leq_one (A B C : ℝ) 
  (hA : 0 < A) (hB : 0 < B) (hC : 0 < C) 
  (hABC : A + B + C = Real.pi) : 
  8 * Real.sin (A / 2) * Real.sin (B / 2) * Real.sin (C / 2) ≤ 1 := 
sorry 

end NUMINAMATH_GPT_sum_of_triangle_angles_sin_halves_leq_one_l1065_106503


namespace NUMINAMATH_GPT_min_value_sin_cos_expr_l1065_106574

open Real

theorem min_value_sin_cos_expr :
  (∀ x : ℝ, sin x ^ 4 + (3 / 2) * cos x ^ 4 ≥ 3 / 5) ∧ 
  (∃ x : ℝ, sin x ^ 4 + (3 / 2) * cos x ^ 4 = 3 / 5) :=
by
  sorry

end NUMINAMATH_GPT_min_value_sin_cos_expr_l1065_106574


namespace NUMINAMATH_GPT_minimum_value_of_fraction_l1065_106558

theorem minimum_value_of_fraction {x : ℝ} (hx : x ≥ 3/2) :
  ∃ y : ℝ, y = (2 * (x - 1) + (1 / (x - 1)) + 2) ∧ y = 2 * Real.sqrt 2 + 2 :=
sorry

end NUMINAMATH_GPT_minimum_value_of_fraction_l1065_106558
