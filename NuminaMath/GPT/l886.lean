import Mathlib

namespace NUMINAMATH_GPT_dog_nails_per_foot_l886_88633

-- Definitions from conditions
def number_of_dogs := 4
def number_of_parrots := 8
def total_nails_to_cut := 113
def parrots_claws := 8

-- Derived calculations from the solution but only involving given conditions
def dogs_claws (nails_per_foot : ℕ) := 16 * nails_per_foot
def parrots_total_claws := number_of_parrots * parrots_claws

-- The main theorem to prove the number of nails per dog foot
theorem dog_nails_per_foot :
  ∃ x : ℚ, 16 * x + parrots_total_claws = total_nails_to_cut :=
by {
  -- Directly state the expected answer
  use 3.0625,
  -- Placeholder for proof
  sorry
}

end NUMINAMATH_GPT_dog_nails_per_foot_l886_88633


namespace NUMINAMATH_GPT_average_mpg_correct_l886_88675

noncomputable def average_mpg (initial_miles final_miles : ℕ) (refill1 refill2 refill3 : ℕ) : ℚ :=
  let distance := final_miles - initial_miles
  let total_gallons := refill1 + refill2 + refill3
  distance / total_gallons

theorem average_mpg_correct :
  average_mpg 32000 33100 15 10 22 = 23.4 :=
by
  sorry

end NUMINAMATH_GPT_average_mpg_correct_l886_88675


namespace NUMINAMATH_GPT_students_total_l886_88678

def num_girls : ℕ := 11
def num_boys : ℕ := num_girls + 5

theorem students_total : num_girls + num_boys = 27 := by
  sorry

end NUMINAMATH_GPT_students_total_l886_88678


namespace NUMINAMATH_GPT_complex_number_problem_l886_88654

open Complex -- Open the complex numbers namespace

theorem complex_number_problem 
  (z1 z2 : ℂ) 
  (h_z1 : z1 = 2 - I) 
  (h_z2 : z2 = -I) : 
  z1 / z2 + Complex.abs z2 = 2 + 2 * I := by
-- Definitions and conditions directly from (a)
  rw [h_z1, h_z2] -- Replace z1 and z2 with their given values
  sorry -- Proof to be filled in place of the solution steps

end NUMINAMATH_GPT_complex_number_problem_l886_88654


namespace NUMINAMATH_GPT_min_sum_real_possible_sums_int_l886_88632

-- Lean 4 statement for the real numbers case
theorem min_sum_real (x y : ℝ) (hx : x + y + 2 * x * y = 5) (hx_pos : x > 0) (hy_pos : y > 0) :
  x + y ≥ Real.sqrt 11 - 1 := 
sorry

-- Lean 4 statement for the integers case
theorem possible_sums_int (x y : ℤ) (hx : x + y + 2 * x * y = 5) :
  x + y = 5 ∨ x + y = -7 :=
sorry

end NUMINAMATH_GPT_min_sum_real_possible_sums_int_l886_88632


namespace NUMINAMATH_GPT_sachin_borrowed_amount_l886_88612

variable (P : ℝ) (gain : ℝ)
variable (interest_rate_borrow : ℝ := 4 / 100)
variable (interest_rate_lend : ℝ := 25 / 4 / 100)
variable (time_period : ℝ := 2)
variable (gain_provided : ℝ := 112.5)

theorem sachin_borrowed_amount (h : gain = 0.0225 * P) : P = 5000 :=
by sorry

end NUMINAMATH_GPT_sachin_borrowed_amount_l886_88612


namespace NUMINAMATH_GPT_equilateral_triangle_data_l886_88693

theorem equilateral_triangle_data
  (A : ℝ)
  (b : ℝ)
  (ha : A = 450)
  (hb : b = 25)
  (equilateral : ∀ (a b c : ℝ), a = b ∧ b = c ∧ c = a) :
  ∃ (h P : ℝ), h = 36 ∧ P = 75 := by
  sorry

end NUMINAMATH_GPT_equilateral_triangle_data_l886_88693


namespace NUMINAMATH_GPT_find_table_price_l886_88619

noncomputable def chair_price (C T : ℝ) : Prop := 2 * C + T = 0.6 * (C + 2 * T)
noncomputable def chair_table_sum (C T : ℝ) : Prop := C + T = 64

theorem find_table_price (C T : ℝ) (h1 : chair_price C T) (h2 : chair_table_sum C T) : T = 56 :=
by sorry

end NUMINAMATH_GPT_find_table_price_l886_88619


namespace NUMINAMATH_GPT_construct_right_triangle_l886_88673

theorem construct_right_triangle (hypotenuse : ℝ) (ε : ℝ) (h_positive : 0 < ε) (h_less_than_ninety : ε < 90) :
    ∃ α β : ℝ, α + β = 90 ∧ α - β = ε ∧ 45 < α ∧ α < 90 :=
by
  sorry

end NUMINAMATH_GPT_construct_right_triangle_l886_88673


namespace NUMINAMATH_GPT_inequality_relation_l886_88694

open Real

theorem inequality_relation (x : ℝ) :
  ¬ ((∀ x, (x - 1) * (x + 3) < 0 → (x + 1) * (x - 3) < 0) ∧
     (∀ x, (x + 1) * (x - 3) < 0 → (x - 1) * (x + 3) < 0)) := 
by
  sorry

end NUMINAMATH_GPT_inequality_relation_l886_88694


namespace NUMINAMATH_GPT_min_product_sum_l886_88646

theorem min_product_sum (a : Fin 7 → ℕ) (b : Fin 7 → ℕ) 
  (h2 : ∀ i, 2 ≤ a i) 
  (h3 : ∀ i, a i ≤ 166) 
  (h4 : ∀ i, a i ^ b i % 167 = a (i + 1) % 7 + 1 ^ 2 % 167) : 
  b 0 * b 1 * b 2 * b 3 * b 4 * b 5 * b 6 * (b 0 + b 1 + b 2 + b 3 + b 4 + b 5 + b 6) = 675 := sorry

end NUMINAMATH_GPT_min_product_sum_l886_88646


namespace NUMINAMATH_GPT_car_speed_l886_88635

theorem car_speed (d t : ℝ) (h_d : d = 624) (h_t : t = 3) : d / t = 208 := by
  sorry

end NUMINAMATH_GPT_car_speed_l886_88635


namespace NUMINAMATH_GPT_quadratic_roots_l886_88642

-- Define the condition for the quadratic equation
def quadratic_eq (x m : ℝ) : Prop := x^2 - 4*x + m + 2 = 0

-- Define the discriminant condition
def discriminant_pos (m : ℝ) : Prop := (4^2 - 4 * (m + 2)) > 0

-- Define the condition range for m
def m_range (m : ℝ) : Prop := m < 2

-- Define the condition for m as a positive integer
def m_positive_integer (m : ℕ) : Prop := m = 1

-- The main theorem stating the problem
theorem quadratic_roots : 
  (∀ (m : ℝ), discriminant_pos m → m_range m) ∧ 
  (∀ m : ℕ, m_positive_integer m → (∃ x1 x2 : ℝ, quadratic_eq x1 m ∧ quadratic_eq x2 m ∧ x1 = 1 ∧ x2 = 3)) := 
by 
  sorry

end NUMINAMATH_GPT_quadratic_roots_l886_88642


namespace NUMINAMATH_GPT_part_I_part_II_l886_88659

noncomputable def f (x : ℝ) := Real.sin x
noncomputable def f' (x : ℝ) := Real.cos x

theorem part_I (x : ℝ) (h : 0 < x) : f' x > 1 - x^2 / 2 := sorry

theorem part_II (a : ℝ) : (∀ x, 0 < x ∧ x < Real.pi / 2 → f x + f x / f' x > a * x) ↔ a ≤ 2 := sorry

end NUMINAMATH_GPT_part_I_part_II_l886_88659


namespace NUMINAMATH_GPT_part1_part2_l886_88685

variable (a b : ℝ)

theorem part1 : ((-a)^2 * (a^2)^2 / a^3) = a^3 := sorry

theorem part2 : (a + b) * (a - b) - (a - b)^2 = 2 * a * b - 2 * b^2 := sorry

end NUMINAMATH_GPT_part1_part2_l886_88685


namespace NUMINAMATH_GPT_f_g_of_neg2_l886_88688

def f (x : ℤ) : ℤ := 3 * x + 2
def g (x : ℤ) : ℤ := (x - 1)^2

theorem f_g_of_neg2 : f (g (-2)) = 29 := by
  -- We need to show f(g(-2)) = 29 given the definitions of f and g
  sorry

end NUMINAMATH_GPT_f_g_of_neg2_l886_88688


namespace NUMINAMATH_GPT_area_of_rectangular_plot_l886_88605

theorem area_of_rectangular_plot (B L : ℕ) (h1 : L = 3 * B) (h2 : B = 18) : L * B = 972 := by
  sorry

end NUMINAMATH_GPT_area_of_rectangular_plot_l886_88605


namespace NUMINAMATH_GPT_high_heels_height_l886_88660

theorem high_heels_height (x : ℝ) :
  let height := 157
  let lower_limbs := 95
  let golden_ratio := 0.618
  (95 + x) / (157 + x) = 0.618 → x = 5.3 :=
sorry

end NUMINAMATH_GPT_high_heels_height_l886_88660


namespace NUMINAMATH_GPT_solve_for_star_l886_88690

theorem solve_for_star 
  (x : ℝ) 
  (h : 45 - (28 - (37 - (15 - x))) = 58) : 
  x = 19 :=
by
  -- Proof goes here. Currently incomplete, so we use sorry.
  sorry

end NUMINAMATH_GPT_solve_for_star_l886_88690


namespace NUMINAMATH_GPT_center_cell_value_l886_88601

theorem center_cell_value
  (a b c d e f g h i : ℝ)
  (h_pos : 0 < a ∧ 0 < b ∧ 0 < c ∧ 0 < d ∧ 0 < e ∧ 0 < f ∧ 0 < g ∧ 0 < h ∧ 0 < i)
  (h_row1 : a * b * c = 1)
  (h_row2 : d * e * f = 1)
  (h_row3 : g * h * i = 1)
  (h_col1 : a * d * g = 1)
  (h_col2 : b * e * h = 1)
  (h_col3 : c * f * i = 1)
  (h_square1 : a * b * d * e = 2)
  (h_square2 : b * c * e * f = 2)
  (h_square3 : d * e * g * h = 2)
  (h_square4 : e * f * h * i = 2) :
  e = 1 :=
  sorry

end NUMINAMATH_GPT_center_cell_value_l886_88601


namespace NUMINAMATH_GPT_sodium_chloride_solution_l886_88631

theorem sodium_chloride_solution (n y : ℝ) (h1 : n > 30) 
  (h2 : 0.01 * n * n = 0.01 * (n - 8) * (n + y)) : 
  y = 8 * n / (n + 8) :=
sorry

end NUMINAMATH_GPT_sodium_chloride_solution_l886_88631


namespace NUMINAMATH_GPT_problem1_problem2_l886_88637

theorem problem1 (x y : ℝ) (h₀ : y = Real.log (2 * x)) (h₁ : x + y = 2) : Real.exp x + Real.exp y > 2 * Real.exp 1 :=
by {
  sorry -- Proof goes here
}

theorem problem2 (x y : ℝ) (h₀ : y = Real.log (2 * x)) (h₁ : x + y = 2) : x * Real.log x + y * Real.log y > 0 :=
by {
  sorry -- Proof goes here
}

end NUMINAMATH_GPT_problem1_problem2_l886_88637


namespace NUMINAMATH_GPT_max_prime_product_l886_88614

theorem max_prime_product : 
  ∃ (x y z : ℕ), 
    Prime x ∧ Prime y ∧ Prime z ∧ 
    x ≠ y ∧ y ≠ z ∧ z ≠ x ∧ 
    x + y + z = 49 ∧ 
    x * y * z = 4199 := 
by
  sorry

end NUMINAMATH_GPT_max_prime_product_l886_88614


namespace NUMINAMATH_GPT_bug_total_distance_l886_88636

def total_distance (p1 p2 p3 p4 : ℤ) : ℤ :=
  abs (p2 - p1) + abs (p3 - p2) + abs (p4 - p3)

theorem bug_total_distance : total_distance (-3) (-8) 0 6 = 19 := 
by sorry

end NUMINAMATH_GPT_bug_total_distance_l886_88636


namespace NUMINAMATH_GPT_fraction_value_sin_cos_value_l886_88610

open Real

-- Let alpha be an angle in radians satisfying the given condition
variable (α : ℝ)

-- Given condition
def condition  : Prop := sin α = 2 * cos α

-- First question
theorem fraction_value (h : condition α) : 
  (sin α - 4 * cos α) / (5 * sin α + 2 * cos α) = -1 / 6 :=
sorry

-- Second question
theorem sin_cos_value (h : condition α) : 
  sin α ^ 2 + 2 * sin α * cos α = 8 / 5 :=
sorry

end NUMINAMATH_GPT_fraction_value_sin_cos_value_l886_88610


namespace NUMINAMATH_GPT_an_geometric_l886_88691

-- Define the functions and conditions
def f (x : ℝ) (b : ℝ) : ℝ := b * x + 1

def g (n : ℕ) (b : ℝ) : ℝ :=
  match n with
  | 0 => 1
  | n + 1 => f (g n b) b

-- Define the sequence a_n
def a (n : ℕ) (b : ℝ) : ℝ :=
  g (n + 1) b - g n b

-- Prove that a_n is a geometric sequence
theorem an_geometric (b : ℝ) (h : b ≠ 1) : 
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) b = q * a n b :=
sorry

end NUMINAMATH_GPT_an_geometric_l886_88691


namespace NUMINAMATH_GPT_find_x_l886_88667

theorem find_x (x : ℚ) (h : (35 / 100) * x = (40 / 100) * 50) : 
  x = 400 / 7 :=
sorry

end NUMINAMATH_GPT_find_x_l886_88667


namespace NUMINAMATH_GPT_find_number_l886_88657

theorem find_number (x n : ℤ) (h1 : |x| = 9 * x - n) (h2 : x = 2) : n = 16 := by 
  sorry

end NUMINAMATH_GPT_find_number_l886_88657


namespace NUMINAMATH_GPT_evaluate_expression_l886_88668

variable (b x : ℝ)

theorem evaluate_expression (h : x = b + 9) : x - b + 4 = 13 := by
  sorry

end NUMINAMATH_GPT_evaluate_expression_l886_88668


namespace NUMINAMATH_GPT_calculate_nabla_l886_88645

def nabla (a b : ℚ) : ℚ := (a + b) / (1 + a * b)

theorem calculate_nabla : nabla (nabla 2 3) 4 = 11 / 9 :=
by
  sorry

end NUMINAMATH_GPT_calculate_nabla_l886_88645


namespace NUMINAMATH_GPT_prob_one_tails_in_three_consecutive_flips_l886_88699

-- Define the probability of heads and tails
def P_H : ℝ := 0.5
def P_T : ℝ := 0.5

-- Define the probability of a sequence of coin flips resulting in exactly one tails in three flips
def P_one_tails_in_three_flips : ℝ :=
  P_H * P_H * P_T + P_H * P_T * P_H + P_T * P_H * P_H

-- The statement we need to prove
theorem prob_one_tails_in_three_consecutive_flips :
  P_one_tails_in_three_flips = 0.375 :=
by
  sorry

end NUMINAMATH_GPT_prob_one_tails_in_three_consecutive_flips_l886_88699


namespace NUMINAMATH_GPT_factorization1_factorization2_factorization3_l886_88674

-- (1) Prove x^3 - 6x^2 + 9x == x(x-3)^2
theorem factorization1 (x : ℝ) : x^3 - 6 * x^2 + 9 * x = x * (x - 3)^2 :=
by sorry

-- (2) Prove (x-2)^2 - x + 2 == (x-2)(x-3)
theorem factorization2 (x : ℝ) : (x - 2)^2 - x + 2 = (x - 2) * (x - 3) :=
by sorry

-- (3) Prove (x^2 + y^2)^2 - 4x^2*y^2 == (x + y)^2(x - y)^2
theorem factorization3 (x y : ℝ) : (x^2 + y^2)^2 - 4 * x^2 * y^2 = (x + y)^2 * (x - y)^2 :=
by sorry

end NUMINAMATH_GPT_factorization1_factorization2_factorization3_l886_88674


namespace NUMINAMATH_GPT_sky_color_changes_l886_88607

theorem sky_color_changes (h1 : 1 = 1) 
  (colors_interval : ℕ := 10) 
  (hours_duration : ℕ := 2)
  (minutes_per_hour : ℕ := 60) :
  (hours_duration * minutes_per_hour) / colors_interval = 12 :=
by {
  -- multiplications and division
  sorry
}

end NUMINAMATH_GPT_sky_color_changes_l886_88607


namespace NUMINAMATH_GPT_problem_divisible_by_480_l886_88666

theorem problem_divisible_by_480 (a : ℕ) (h1 : a % 10 = 4) (h2 : ¬ (a % 4 = 0)) : ∃ k : ℕ, a * (a^2 - 1) * (a^2 - 4) = 480 * k :=
by
  sorry

end NUMINAMATH_GPT_problem_divisible_by_480_l886_88666


namespace NUMINAMATH_GPT_total_beetles_eaten_each_day_l886_88671

-- Definitions from the conditions
def birds_eat_per_day : ℕ := 12
def snakes_eat_per_day : ℕ := 3
def jaguars_eat_per_day : ℕ := 5
def number_of_jaguars : ℕ := 6

-- Theorem statement
theorem total_beetles_eaten_each_day :
  (number_of_jaguars * jaguars_eat_per_day) * snakes_eat_per_day * birds_eat_per_day = 1080 :=
by sorry

end NUMINAMATH_GPT_total_beetles_eaten_each_day_l886_88671


namespace NUMINAMATH_GPT_ratio_of_amounts_l886_88687

theorem ratio_of_amounts (B J P : ℝ) (hB : B = 60) (hP : P = (1 / 3) * B) (hJ : J = B - 20) : J / P = 2 :=
by
  have hP_val : P = 20 := by sorry
  have hJ_val : J = 40 := by sorry
  have ratio : J / P = 40 / 20 := by sorry
  show J / P = 2
  sorry

end NUMINAMATH_GPT_ratio_of_amounts_l886_88687


namespace NUMINAMATH_GPT_cuckoo_chime_78_l886_88676

-- Define the arithmetic sum for the cuckoo clock problem
def cuckoo_chime_sum (n a l : Nat) : Nat :=
  n * (a + l) / 2

-- Main theorem
theorem cuckoo_chime_78 : 
  cuckoo_chime_sum 12 1 12 = 78 := 
by
  -- Proof part can be written here
  sorry

end NUMINAMATH_GPT_cuckoo_chime_78_l886_88676


namespace NUMINAMATH_GPT_distance_from_original_position_l886_88648

/-- Definition of initial problem conditions and parameters --/
def square_area (l : ℝ) : Prop :=
  l * l = 18

def folded_area_relation (x : ℝ) : Prop :=
  0.5 * x^2 = 2 * (18 - 0.5 * x^2)

/-- The main statement that needs to be proved --/
theorem distance_from_original_position :
  ∃ (A_initial A_folded_dist : ℝ),
    square_area A_initial ∧
    (∃ x : ℝ, folded_area_relation x ∧ A_folded_dist = 2 * Real.sqrt 6 * Real.sqrt 2) ∧
    A_folded_dist = 4 * Real.sqrt 3 :=
by
  -- The proof is omitted here; providing structure for the problem.
  sorry

end NUMINAMATH_GPT_distance_from_original_position_l886_88648


namespace NUMINAMATH_GPT_problem_statement_l886_88672

def product_of_first_n (n : ℕ) : ℕ := List.prod (List.range' 1 n)

def sum_of_first_n (n : ℕ) : ℕ := List.sum (List.range' 1 n)

theorem problem_statement : 
  let numerator := product_of_first_n 9  -- product of numbers 1 through 8
  let denominator := sum_of_first_n 9  -- sum of numbers 1 through 8
  numerator / denominator = 1120 :=
by {
  sorry
}

end NUMINAMATH_GPT_problem_statement_l886_88672


namespace NUMINAMATH_GPT_forty_percent_of_number_l886_88686

theorem forty_percent_of_number (N : ℝ) (h : (1/4) * (1/3) * (2/5) * N = 16) : (40/100) * N = 192 :=
by
  sorry

end NUMINAMATH_GPT_forty_percent_of_number_l886_88686


namespace NUMINAMATH_GPT_largest_common_term_l886_88625

theorem largest_common_term (n m : ℕ) (k : ℕ) (a : ℕ) 
  (h1 : a = 7 + 7 * n) 
  (h2 : a = 8 + 12 * m) 
  (h3 : 56 + 84 * k < 500) : a = 476 :=
  sorry

end NUMINAMATH_GPT_largest_common_term_l886_88625


namespace NUMINAMATH_GPT_part1_part2_part3_l886_88606

def f (x : ℝ) : ℝ := x^2 - 1
def g (a x : ℝ) : ℝ := a * |x - 1|

theorem part1 :
  ∀ x : ℝ, |f x| = |x - 1| → x = -2 ∨ x = 0 ∨ x = 1 :=
sorry

theorem part2 (a : ℝ) :
  (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ |f x1| = g a x1 ∧ |f x2| = g a x2) ↔ (a = 0 ∨ a = 2) :=
sorry

theorem part3 (a : ℝ) :
  (∀ x : ℝ, f x ≥ g a x) ↔ (a ≤ -2) :=
sorry

end NUMINAMATH_GPT_part1_part2_part3_l886_88606


namespace NUMINAMATH_GPT_boys_girls_relationship_l886_88641

theorem boys_girls_relationship (b g : ℕ): (4 + 2 * b = g) → (b = (g - 4) / 2) :=
by
  intros h
  sorry

end NUMINAMATH_GPT_boys_girls_relationship_l886_88641


namespace NUMINAMATH_GPT_remainder_is_zero_l886_88608

theorem remainder_is_zero (D R r : ℕ) (h1 : D = 12 * 42 + R)
                           (h2 : D = 21 * 24 + r)
                           (h3 : r < 21) :
                           r = 0 :=
by 
  sorry

end NUMINAMATH_GPT_remainder_is_zero_l886_88608


namespace NUMINAMATH_GPT_prove_a₈_l886_88697

noncomputable def first_term (a : ℕ → ℝ) : Prop := a 1 = 3
noncomputable def arithmetic_b (a b : ℕ → ℝ) : Prop := ∀ n, b n = a (n + 1) - a n
noncomputable def b_conditions (b : ℕ → ℝ) : Prop := b 3 = -2 ∧ b 10 = 12

theorem prove_a₈ (a b : ℕ → ℝ) (h1 : first_term a) (h2 : arithmetic_b a b) (h3 : b_conditions b) :
  a 8 = 3 :=
sorry

end NUMINAMATH_GPT_prove_a₈_l886_88697


namespace NUMINAMATH_GPT_tan_2alpha_value_beta_value_l886_88653

variable (α β : ℝ)
variable (h1 : 0 < β ∧ β < α ∧ α < π / 2)
variable (h2 : Real.cos α = 1 / 7)
variable (h3 : Real.cos (α - β) = 13 / 14)

theorem tan_2alpha_value : Real.tan (2 * α) = - (8 * Real.sqrt 3 / 47) :=
by
  sorry

theorem beta_value : β = π / 3 :=
by
  sorry

end NUMINAMATH_GPT_tan_2alpha_value_beta_value_l886_88653


namespace NUMINAMATH_GPT_arithmetic_sequence_problem_l886_88658

theorem arithmetic_sequence_problem : 
  ∀ (a : ℕ → ℕ) (d : ℕ), 
  a 1 = 1 →
  (a 3 + a 4 + a 5 + a 6 = 20) →
  a 8 = 9 :=
by
  intros a d h₁ h₂
  -- We skip the proof, leaving a placeholder.
  sorry

end NUMINAMATH_GPT_arithmetic_sequence_problem_l886_88658


namespace NUMINAMATH_GPT_f_monotonic_increasing_l886_88655

noncomputable def f (x : ℝ) : ℝ := 2 - 3 / x

theorem f_monotonic_increasing :
  ∀ (x1 x2 : ℝ), 0 < x1 → 0 < x2 → x1 > x2 → f x1 > f x2 :=
by
  intros x1 x2 hx1 hx2 h
  sorry

end NUMINAMATH_GPT_f_monotonic_increasing_l886_88655


namespace NUMINAMATH_GPT_find_c_value_l886_88664

theorem find_c_value (x1 y1 x2 y2 : ℝ) (h1 : x1 = 1) (h2 : y1 = 4) (h3 : x2 = 5) (h4 : y2 = 0) (c : ℝ)
  (h5 : 3 * ((x1 + x2) / 2) - 2 * ((y1 + y2) / 2) = c) : c = 5 :=
sorry

end NUMINAMATH_GPT_find_c_value_l886_88664


namespace NUMINAMATH_GPT_find_z_l886_88639

variable {x y z w : ℝ}

theorem find_z (h : (1/x) + (1/y) = (1/z) + w) : z = (x * y) / (x + y - w * x * y) :=
by sorry

end NUMINAMATH_GPT_find_z_l886_88639


namespace NUMINAMATH_GPT_fruit_bowl_oranges_l886_88602

theorem fruit_bowl_oranges :
  ∀ (bananas apples oranges : ℕ),
    bananas = 2 →
    apples = 2 * bananas →
    bananas + apples + oranges = 12 →
    oranges = 6 :=
by
  intros bananas apples oranges h1 h2 h3
  sorry

end NUMINAMATH_GPT_fruit_bowl_oranges_l886_88602


namespace NUMINAMATH_GPT_last_two_digits_of_sum_l886_88623

def factorial (n : ℕ) : ℕ :=
  if n = 0 then 1 else n * factorial (n - 1)

def last_two_digits (n : ℕ) : ℕ := n % 100

theorem last_two_digits_of_sum :
  last_two_digits (factorial 4 + factorial 5 + factorial 6 + factorial 7 + factorial 8 + factorial 9) = 4 :=
by
  sorry

end NUMINAMATH_GPT_last_two_digits_of_sum_l886_88623


namespace NUMINAMATH_GPT_max_area_rect_bamboo_fence_l886_88665

theorem max_area_rect_bamboo_fence (a b : ℝ) (h : a + b = 10) : a * b ≤ 24 :=
by
  sorry

end NUMINAMATH_GPT_max_area_rect_bamboo_fence_l886_88665


namespace NUMINAMATH_GPT_ratio_of_tetrahedrons_volume_l886_88629

theorem ratio_of_tetrahedrons_volume (d R s s' V_ratio m n : ℕ) (h1 : d = 4)
  (h2 : R = 2)
  (h3 : s = 4 * R / Real.sqrt 6)
  (h4 : s' = s / Real.sqrt 8)
  (h5 : V_ratio = (s' / s) ^ 3)
  (hm : m = 1)
  (hn : n = 32)
  (h_ratio : V_ratio = m / n) :
  m + n = 33 :=
by
  sorry

end NUMINAMATH_GPT_ratio_of_tetrahedrons_volume_l886_88629


namespace NUMINAMATH_GPT_select_team_l886_88670

-- Definition of the problem conditions 
def boys : Nat := 10
def girls : Nat := 12
def team_size : Nat := 8
def boys_in_team : Nat := 4
def girls_in_team : Nat := 4

-- Given conditions reflect in the Lean statement that needs proof
theorem select_team : 
  (Nat.choose boys boys_in_team) * (Nat.choose girls girls_in_team) = 103950 :=
by
  sorry

end NUMINAMATH_GPT_select_team_l886_88670


namespace NUMINAMATH_GPT_projection_of_vectors_l886_88613

variables {a b : ℝ}

noncomputable def vector_projection (a b : ℝ) : ℝ :=
  (a * b) / b^2 * b

theorem projection_of_vectors
  (ha : abs a = 6)
  (hb : abs b = 3)
  (hab : a * b = -12) : vector_projection a b = -4 :=
sorry

end NUMINAMATH_GPT_projection_of_vectors_l886_88613


namespace NUMINAMATH_GPT_jasper_time_l886_88661

theorem jasper_time {omar_time : ℕ} {omar_height : ℕ} {jasper_height : ℕ} 
  (h1 : omar_time = 12)
  (h2 : omar_height = 240)
  (h3 : jasper_height = 600)
  (h4 : ∃ t : ℕ, t = (jasper_height * omar_time) / (3 * omar_height))
  : t = 10 :=
by sorry

end NUMINAMATH_GPT_jasper_time_l886_88661


namespace NUMINAMATH_GPT_max_tickets_sold_l886_88630

theorem max_tickets_sold (bus_capacity : ℕ) (num_stations : ℕ) (max_capacity : bus_capacity = 25) 
  (total_stations : num_stations = 14) : 
  ∃ (tickets : ℕ), tickets = 67 :=
by 
  sorry

end NUMINAMATH_GPT_max_tickets_sold_l886_88630


namespace NUMINAMATH_GPT_compute_expression_l886_88627

theorem compute_expression :
  (4 + 8 - 16 + 32 + 64 - 128 + 256) / (8 + 16 - 32 + 64 + 128 - 256 + 512) = 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_compute_expression_l886_88627


namespace NUMINAMATH_GPT_intersection_P_Q_intersection_complementP_Q_l886_88663

-- Define the universal set U
def U := Set.univ (ℝ)

-- Define set P
def P := {x : ℝ | |x| > 2}

-- Define set Q
def Q := {x : ℝ | x^2 - 4*x + 3 < 0}

-- Complement of P with respect to U
def complement_P : Set ℝ := {x : ℝ | |x| ≤ 2}

theorem intersection_P_Q : P ∩ Q = ({x : ℝ | 2 < x ∧ x < 3}) :=
by {
  sorry
}

theorem intersection_complementP_Q : complement_P ∩ Q = ({x : ℝ | 1 < x ∧ x ≤ 2}) :=
by {
  sorry
}

end NUMINAMATH_GPT_intersection_P_Q_intersection_complementP_Q_l886_88663


namespace NUMINAMATH_GPT_smallest_k_satisfies_l886_88628

noncomputable def sqrt (x : ℝ) : ℝ := x ^ (1 / 2 : ℝ)

theorem smallest_k_satisfies (x y : ℝ) (hx : 0 ≤ x) (hy : 0 ≤ y) :
  (sqrt (x * y)) + (1 / 2) * (sqrt (abs (x - y))) ≥ (x + y) / 2 :=
by
  sorry

end NUMINAMATH_GPT_smallest_k_satisfies_l886_88628


namespace NUMINAMATH_GPT_solve_for_x_l886_88638

theorem solve_for_x (x : ℝ) (h : (5 - 3 * x)^5 = -1) : x = 2 := by
sorry

end NUMINAMATH_GPT_solve_for_x_l886_88638


namespace NUMINAMATH_GPT_simplify_expression_l886_88626

theorem simplify_expression :
  (6^7 + 4^6) * (1^5 - (-1)^5)^10 = 290938368 :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l886_88626


namespace NUMINAMATH_GPT_room_length_l886_88662

theorem room_length (width : ℝ) (total_cost : ℝ) (cost_per_sq_meter : ℝ) (length : ℝ) : 
  width = 3.75 ∧ total_cost = 14437.5 ∧ cost_per_sq_meter = 700 → length = 5.5 :=
by
  sorry

end NUMINAMATH_GPT_room_length_l886_88662


namespace NUMINAMATH_GPT_reciprocal_relation_l886_88696

theorem reciprocal_relation (x : ℝ) (h : 1 / (x + 3) = 2) : 1 / (x + 5) = 2 / 5 := 
by
  sorry

end NUMINAMATH_GPT_reciprocal_relation_l886_88696


namespace NUMINAMATH_GPT_total_teachers_correct_l886_88643

-- Define the number of departments and the total number of teachers
def num_departments : ℕ := 7
def total_teachers : ℕ := 140

-- Proving that the total number of teachers is 140
theorem total_teachers_correct : total_teachers = 140 := 
by
  sorry

end NUMINAMATH_GPT_total_teachers_correct_l886_88643


namespace NUMINAMATH_GPT_quadratic_max_value_l886_88604

open Real

variables (a b c x : ℝ)
noncomputable def f (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem quadratic_max_value (h₀ : a < 0) (x₀ : ℝ) (h₁ : 2 * a * x₀ + b = 0) : 
  ∀ x : ℝ, f a b c x ≤ f a b c x₀ := sorry

end NUMINAMATH_GPT_quadratic_max_value_l886_88604


namespace NUMINAMATH_GPT_required_moles_of_H2O_l886_88677

-- Definition of the balanced chemical reaction
def balanced_reaction_na_to_naoh_and_H2 : Prop :=
  ∀ (NaH H2O NaOH H2 : ℕ), NaH + H2O = NaOH + H2

-- The given moles of NaH
def moles_NaH : ℕ := 2

-- Assertion that we need to prove: amount of H2O required is 2 moles
theorem required_moles_of_H2O (balanced : balanced_reaction_na_to_naoh_and_H2) : 
  (2 * 1) = 2 :=
by
  sorry

end NUMINAMATH_GPT_required_moles_of_H2O_l886_88677


namespace NUMINAMATH_GPT_units_digit_of_result_l886_88651

def tens_plus_one (a b : ℕ) : Prop := a = b + 1

theorem units_digit_of_result (a b : ℕ) (h : tens_plus_one a b) :
  ((10 * a + b) - (10 * b + a)) % 10 = 9 :=
by
  -- Let's mark this part as incomplete using sorry.
  sorry

end NUMINAMATH_GPT_units_digit_of_result_l886_88651


namespace NUMINAMATH_GPT_polar_line_through_center_perpendicular_to_axis_l886_88621

-- We define our conditions
def circle_in_polar (ρ θ : ℝ) : Prop := ρ = 4 * Real.cos θ

def center_of_circle (C : ℝ × ℝ) : Prop := C = (2, 0)

def line_in_rectangular (x : ℝ) : Prop := x = 2

-- We now state the proof problem
theorem polar_line_through_center_perpendicular_to_axis (ρ θ : ℝ) : 
  (∃ C, center_of_circle C ∧ (∃ x, line_in_rectangular x)) →
  (circle_in_polar ρ θ → ρ * Real.cos θ = 2) :=
by
  sorry

end NUMINAMATH_GPT_polar_line_through_center_perpendicular_to_axis_l886_88621


namespace NUMINAMATH_GPT_non_adjacent_ball_arrangements_l886_88683

-- Statement only, proof is omitted
theorem non_adjacent_ball_arrangements :
  let n := (3: ℕ) -- Number of identical yellow balls
  let white_red_positions := (4: ℕ) -- Positions around the yellow unit
  let choose_positions := Nat.choose white_red_positions 2
  let arrange_balls := (2: ℕ) -- Ways to arrange the white and red balls in the chosen positions
  let total_arrangements := choose_positions * arrange_balls
  total_arrangements = 12 := 
by
  sorry

end NUMINAMATH_GPT_non_adjacent_ball_arrangements_l886_88683


namespace NUMINAMATH_GPT_juicy_12_juicy_20_l886_88617

def is_juicy (n : ℕ) : Prop :=
  ∃ (a b c d : ℕ), a ≠ b ∧ b ≠ c ∧ c ≠ d ∧ 1 = (1 / a) + (1 / b) + (1 / c) + (1 / d) ∧ a * b * c * d = n

theorem juicy_12 : is_juicy 12 :=
sorry

theorem juicy_20 : is_juicy 20 :=
sorry

end NUMINAMATH_GPT_juicy_12_juicy_20_l886_88617


namespace NUMINAMATH_GPT_sequence_general_term_correct_l886_88603

open Nat

def S (n : ℕ) : ℤ := 3 * (n : ℤ) * (n : ℤ) - 2 * (n : ℤ) + 1

def a (n : ℕ) : ℤ :=
  if n = 1 then 2
  else 6 * (n : ℤ) - 5

theorem sequence_general_term_correct : ∀ n, (S n - S (n - 1) = a n) :=
by
  intros
  sorry

end NUMINAMATH_GPT_sequence_general_term_correct_l886_88603


namespace NUMINAMATH_GPT_simplify_expression_l886_88634

theorem simplify_expression (x y : ℝ) (hx : x = -1/2) (hy : y = 2022) :
  ((2*x - y)^2 - (2*x + y)*(2*x - y)) / (2*y) = 2023 :=
by
  sorry

end NUMINAMATH_GPT_simplify_expression_l886_88634


namespace NUMINAMATH_GPT_car_speed_travel_l886_88684

theorem car_speed_travel (v : ℝ) :
  600 = 3600 / 6 ∧
  (6 : ℝ) = (3600 / v) + 2 →
  v = 900 :=
by
  sorry

end NUMINAMATH_GPT_car_speed_travel_l886_88684


namespace NUMINAMATH_GPT_coordinate_relationship_l886_88616

theorem coordinate_relationship (x y : ℝ) (h : |x| - |y| = 0) : (|x| - |y| = 0) :=
by
    sorry

end NUMINAMATH_GPT_coordinate_relationship_l886_88616


namespace NUMINAMATH_GPT_range_of_m_l886_88656

-- Define the discriminant of a quadratic equation
def discriminant(a b c : ℝ) : ℝ := b^2 - 4 * a * c

-- Proposition p: The equation x^2 - 2x + m = 0 has two distinct real roots
def p (m : ℝ) : Prop := discriminant 1 (-2) m > 0

-- Proposition q: The function y = (m + 2)x - 1 is monotonically increasing
def q (m : ℝ) : Prop := m + 2 > 0

-- The main theorem stating the conditions and proving the range of m
theorem range_of_m (m : ℝ) (hpq : p m ∨ q m) (hpnq : ¬(p m ∧ q m)) : m ≤ -2 ∨ m ≥ 1 := sorry

end NUMINAMATH_GPT_range_of_m_l886_88656


namespace NUMINAMATH_GPT_price_and_max_units_proof_l886_88698

/-- 
Given the conditions of purchasing epidemic prevention supplies: 
- 60 units of type A and 45 units of type B costing 1140 yuan
- 45 units of type A and 30 units of type B costing 840 yuan
- A total of 600 units with a cost not exceeding 8000 yuan

Prove:
1. The price of each unit of type A is 16 yuan, and type B is 4 yuan.
2. The maximum number of units of type A that can be purchased is 466.
--/
theorem price_and_max_units_proof 
  (x y : ℕ) 
  (m : ℕ)
  (h1 : 60 * x + 45 * y = 1140) 
  (h2 : 45 * x + 30 * y = 840) 
  (h3 : 16 * m + 4 * (600 - m) ≤ 8000) 
  (h4 : m ≤ 600) :
  x = 16 ∧ y = 4 ∧ m = 466 := 
by 
  sorry

end NUMINAMATH_GPT_price_and_max_units_proof_l886_88698


namespace NUMINAMATH_GPT_point_not_on_line_l886_88650

theorem point_not_on_line (m b : ℝ) (h : m * b > 0) : ¬ ((2023, 0) ∈ {p : ℝ × ℝ | p.2 = m * p.1 + b}) :=
by
  -- proof is omitted
  sorry

end NUMINAMATH_GPT_point_not_on_line_l886_88650


namespace NUMINAMATH_GPT_dans_average_rate_l886_88640

/-- Dan's average rate for the entire trip, given the conditions, equals 0.125 miles per minute --/
theorem dans_average_rate :
  ∀ (d_run d_swim : ℝ) (r_run r_swim : ℝ) (time_run time_swim : ℝ),
  d_run = 3 ∧ d_swim = 3 ∧ r_run = 10 ∧ r_swim = 6 ∧ 
  time_run = (d_run / r_run) * 60 ∧ time_swim = (d_swim / r_swim) * 60 →
  ((d_run + d_swim) / (time_run + time_swim)) = 0.125 :=
by
  intros d_run d_swim r_run r_swim time_run time_swim h
  sorry

end NUMINAMATH_GPT_dans_average_rate_l886_88640


namespace NUMINAMATH_GPT_omega_terms_sum_to_zero_l886_88692

theorem omega_terms_sum_to_zero {ω : ℂ} (h1 : ω^5 = 1) (h2 : ω ≠ 1) :
  ω^12 + ω^15 + ω^18 + ω^21 + ω^24 = 0 :=
by sorry

end NUMINAMATH_GPT_omega_terms_sum_to_zero_l886_88692


namespace NUMINAMATH_GPT_sum_of_first_n_terms_l886_88652

-- Define the sequence aₙ
def a (n : ℕ) : ℕ := 2 * n - 1

-- Prove that the sum of the first n terms of the sequence is n²
theorem sum_of_first_n_terms (n : ℕ) : (Finset.range (n+1)).sum a = n^2 :=
by sorry -- Proof is skipped

end NUMINAMATH_GPT_sum_of_first_n_terms_l886_88652


namespace NUMINAMATH_GPT_problem_l886_88680

theorem problem (X Y Z : ℕ) (hX : 0 < X) (hY : 0 < Y) (hZ : 0 < Z)
  (coprime : Nat.gcd X (Nat.gcd Y Z) = 1)
  (h : X * Real.log 3 / Real.log 100 + Y * Real.log 4 / Real.log 100 = Z):
  X + Y + Z = 4 :=
sorry

end NUMINAMATH_GPT_problem_l886_88680


namespace NUMINAMATH_GPT_annual_donation_amount_l886_88618

-- Define the conditions
variables (age_start age_end : ℕ)
variables (total_donations : ℕ)

-- Define the question (prove the annual donation amount) given these conditions
theorem annual_donation_amount (h1 : age_start = 13) (h2 : age_end = 33) (h3 : total_donations = 105000) :
  total_donations / (age_end - age_start) = 5250 :=
by
   sorry

end NUMINAMATH_GPT_annual_donation_amount_l886_88618


namespace NUMINAMATH_GPT_solve_first_train_length_l886_88649

noncomputable def first_train_length (time: ℝ) (speed1_kmh: ℝ) (speed2_kmh: ℝ) (length2: ℝ) : ℝ :=
  let speed1_ms := speed1_kmh * 1000 / 3600
  let speed2_ms := speed2_kmh * 1000 / 3600
  let relative_speed := speed1_ms + speed2_ms
  let total_distance := relative_speed * time
  total_distance - length2

theorem solve_first_train_length :
  first_train_length 7.0752960452818945 80 65 165 = 120.28 :=
by
  simp [first_train_length]
  norm_num
  sorry

end NUMINAMATH_GPT_solve_first_train_length_l886_88649


namespace NUMINAMATH_GPT_max_in_circle_eqn_l886_88681

theorem max_in_circle_eqn : 
  ∀ (x y : ℝ), (x ≥ 0) → (y ≥ 0) → (4 * x + 3 * y ≤ 12) → (x - 1)^2 + (y - 1)^2 = 1 :=
by
  intros x y hx hy hineq
  sorry

end NUMINAMATH_GPT_max_in_circle_eqn_l886_88681


namespace NUMINAMATH_GPT_range_of_a_l886_88647

theorem range_of_a
  (P : Prop := ∃ x : ℝ, x^2 + 2 * a * x + a ≤ 0) :
  ¬P → 0 < a ∧ a < 1 :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l886_88647


namespace NUMINAMATH_GPT_remaining_oranges_l886_88679

/-- Define the conditions of the problem. -/
def oranges_needed_Michaela : ℕ := 20
def oranges_needed_Cassandra : ℕ := 2 * oranges_needed_Michaela
def total_oranges_picked : ℕ := 90

/-- State the proof problem. -/
theorem remaining_oranges : total_oranges_picked - (oranges_needed_Michaela + oranges_needed_Cassandra) = 30 := 
sorry

end NUMINAMATH_GPT_remaining_oranges_l886_88679


namespace NUMINAMATH_GPT_find_C_l886_88689

theorem find_C
  (A B C : ℕ)
  (h1 : A + B + C = 1000)
  (h2 : A + C = 700)
  (h3 : B + C = 600) :
  C = 300 := by
  sorry

end NUMINAMATH_GPT_find_C_l886_88689


namespace NUMINAMATH_GPT_triangle_side_length_uniqueness_l886_88622

-- Define the conditions as axioms
variable (n : ℕ)
variable (h : n > 0)
variable (A1 : 3 * n + 9 > 5 * n - 4)
variable (A2 : 5 * n - 4 > 4 * n + 6)

-- The theorem stating the constraints and expected result
theorem triangle_side_length_uniqueness :
  (4 * n + 6) + (3 * n + 9) > (5 * n - 4) ∧
  (3 * n + 9) + (5 * n - 4) > (4 * n + 6) ∧
  (5 * n - 4) + (4 * n + 6) > (3 * n + 9) ∧
  3 * n + 9 > 5 * n - 4 ∧
  5 * n - 4 > 4 * n + 6 → 
  n = 11 :=
by {
  -- Proof steps can be filled here
  sorry
}

end NUMINAMATH_GPT_triangle_side_length_uniqueness_l886_88622


namespace NUMINAMATH_GPT_tangent_addition_l886_88600

open Real

theorem tangent_addition (x : ℝ) (h : tan x = 3) :
  tan (x + π / 6) = - (5 * (sqrt 3 + 3)) / 3 := by
  -- Providing a brief outline of the proof steps is not necessary for the statement
  sorry

end NUMINAMATH_GPT_tangent_addition_l886_88600


namespace NUMINAMATH_GPT_prove_all_perfect_squares_l886_88695

noncomputable def is_perfect_square (n : ℕ) : Prop :=
∃ k : ℕ, k^2 = n

noncomputable def all_distinct (l : List ℕ) : Prop :=
l.Nodup

noncomputable def pairwise_products_are_perfect_squares (l : List ℕ) : Prop :=
∀ i j, i < l.length → j < l.length → i ≠ j → is_perfect_square (l.nthLe i sorry * l.nthLe j sorry)

theorem prove_all_perfect_squares :
  ∀ l : List ℕ, l.length = 25 →
  (∀ x ∈ l, x ≤ 1000 ∧ 0 < x) →
  all_distinct l →
  pairwise_products_are_perfect_squares l →
  ∀ x ∈ l, is_perfect_square x := 
by
  intros l h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_prove_all_perfect_squares_l886_88695


namespace NUMINAMATH_GPT_students_in_johnsons_class_l886_88615

-- Define the conditions as constants/variables
def studentsInFinleysClass : ℕ := 24
def studentsAdditionalInJohnsonsClass : ℕ := 10

-- State the problem as a theorem
theorem students_in_johnsons_class : 
  let halfFinleysClass := studentsInFinleysClass / 2
  let johnsonsClass := halfFinleysClass + studentsAdditionalInJohnsonsClass
  johnsonsClass = 22 :=
by
  sorry

end NUMINAMATH_GPT_students_in_johnsons_class_l886_88615


namespace NUMINAMATH_GPT_certain_fraction_ratio_l886_88669

theorem certain_fraction_ratio :
  (∃ (x y : ℚ), (x / y) / (6 / 5) = (2 / 5) / 0.14285714285714288) →
  (∃ (x y : ℚ), x / y = 84 / 25) := 
  by
    intros h_ratio
    have h_rat := h_ratio
    sorry

end NUMINAMATH_GPT_certain_fraction_ratio_l886_88669


namespace NUMINAMATH_GPT_emily_initial_marbles_l886_88609

open Nat

theorem emily_initial_marbles (E : ℕ) (h : 3 * E - (3 * E / 2 + 1) = 8) : E = 6 :=
sorry

end NUMINAMATH_GPT_emily_initial_marbles_l886_88609


namespace NUMINAMATH_GPT_fraction_meaningful_l886_88611

theorem fraction_meaningful (x : ℝ) : (x ≠ 1) ↔ ∃ y, y = 1 / (x - 1) :=
by
  sorry

end NUMINAMATH_GPT_fraction_meaningful_l886_88611


namespace NUMINAMATH_GPT_circle_radius_triple_area_l886_88682

noncomputable def circle_radius (n : ℝ) : ℝ :=
  let r := (n * (Real.sqrt 3 + 1)) / 2
  r

theorem circle_radius_triple_area (r n : ℝ) (h : π * (r + n)^2 = 3 * π * r^2) :
  r = (n * (Real.sqrt 3 + 1)) / 2 :=
by
  sorry

end NUMINAMATH_GPT_circle_radius_triple_area_l886_88682


namespace NUMINAMATH_GPT_inequality_proof_l886_88620

theorem inequality_proof (x y z : ℝ) (hx : 0 ≤ x ∧ x ≤ 1) (hy : 0 ≤ y ∧ y ≤ 1) (hz : 0 ≤ z ∧ z ≤ 1) :
  (x / (y + z + 1)) + (y / (z + x + 1)) + (z / (x + y + 1)) ≤ 1 - (1 - x) * (1 - y) * (1 - z) :=
sorry

end NUMINAMATH_GPT_inequality_proof_l886_88620


namespace NUMINAMATH_GPT_find_f_g_3_l886_88644

def f (x : ℝ) : ℝ := x - 2
def g (x : ℝ) : ℝ := x^2 - 4 * x + 3

theorem find_f_g_3 :
  f (g 3) = -2 := by
  sorry

end NUMINAMATH_GPT_find_f_g_3_l886_88644


namespace NUMINAMATH_GPT_lives_per_player_l886_88624

-- Definitions based on the conditions
def initial_players : Nat := 2
def joined_players : Nat := 2
def total_lives : Nat := 24

-- Derived condition
def total_players : Nat := initial_players + joined_players

-- Proof statement
theorem lives_per_player : total_lives / total_players = 6 :=
by
  sorry

end NUMINAMATH_GPT_lives_per_player_l886_88624
