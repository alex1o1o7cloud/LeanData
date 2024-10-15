import Mathlib

namespace NUMINAMATH_GPT_stratified_sampling_l1566_156618

-- Definitions of the classes and their student counts
def class1_students : Nat := 54
def class2_students : Nat := 42

-- Definition of total students to be sampled
def total_sampled_students : Nat := 16

-- Definition of the number of students to be selected from each class
def students_selected_from_class1 : Nat := 9
def students_selected_from_class2 : Nat := 7

-- The proof problem
theorem stratified_sampling :
  students_selected_from_class1 + students_selected_from_class2 = total_sampled_students ∧ 
  students_selected_from_class1 * (class2_students + class1_students) = class1_students * total_sampled_students :=
by
  sorry

end NUMINAMATH_GPT_stratified_sampling_l1566_156618


namespace NUMINAMATH_GPT_inequality_proof_l1566_156679

theorem inequality_proof (x1 x2 x3 : ℝ) (hx1 : 0 < x1) (hx2 : 0 < x2) (hx3 : 0 < x3) :
  (x1^2 + x2^2 + x3^2)^3 / (x1^3 + x2^3 + x3^3)^2 ≤ 3 :=
sorry

end NUMINAMATH_GPT_inequality_proof_l1566_156679


namespace NUMINAMATH_GPT_rectangle_area_l1566_156621

-- Conditions
def radius : ℝ := 6
def diameter : ℝ := 2 * radius
def width : ℝ := diameter
def ratio_length_to_width : ℝ := 3

-- Given the ratio of the length to the width is 3:1
def length : ℝ := ratio_length_to_width * width

-- Theorem stating the area of the rectangle
theorem rectangle_area :
  let area := length * width
  area = 432 := by
    sorry

end NUMINAMATH_GPT_rectangle_area_l1566_156621


namespace NUMINAMATH_GPT_percentage_fraction_l1566_156685

theorem percentage_fraction (P : ℚ) (hP : P < 35) (h : (P / 100) * 180 = 42) : P = 7 / 30 * 100 :=
by
  sorry

end NUMINAMATH_GPT_percentage_fraction_l1566_156685


namespace NUMINAMATH_GPT_cost_per_slice_in_cents_l1566_156652

def loaves : ℕ := 3
def slices_per_loaf : ℕ := 20
def total_payment : ℕ := 2 * 20
def change : ℕ := 16
def total_cost : ℕ := total_payment - change
def total_slices : ℕ := loaves * slices_per_loaf

theorem cost_per_slice_in_cents :
  (total_cost : ℕ) * 100 / total_slices = 40 :=
by
  sorry

end NUMINAMATH_GPT_cost_per_slice_in_cents_l1566_156652


namespace NUMINAMATH_GPT_problem1_problem2_problem3_problem4_l1566_156664

-- statement for problem 1
theorem problem1 : -5 + 8 - 2 = 1 := by
  sorry

-- statement for problem 2
theorem problem2 : (-3) * (5/6) / (-1/4) = 10 := by
  sorry

-- statement for problem 3
theorem problem3 : -3/17 + (-3.75) + (-14/17) + (15/4) = -1 := by
  sorry

-- statement for problem 4
theorem problem4 : -(1^10) - ((13/14) - (11/12)) * (4 - (-2)^2) + (1/2) / 3 = -(5/6) := by
  sorry

end NUMINAMATH_GPT_problem1_problem2_problem3_problem4_l1566_156664


namespace NUMINAMATH_GPT_smallest_n_integer_price_l1566_156668

theorem smallest_n_integer_price (p : ℚ) (h : ∃ x : ℕ, p = x ∧ 1.06 * p = n) : n = 53 :=
sorry

end NUMINAMATH_GPT_smallest_n_integer_price_l1566_156668


namespace NUMINAMATH_GPT_triangle_inequality_theorem_l1566_156688

def is_triangle (a b c : ℕ) : Prop :=
  (a + b > c) ∧ (a + c > b) ∧ (b + c > a)

theorem triangle_inequality_theorem :
  ¬ is_triangle 2 3 5 ∧ is_triangle 5 6 10 ∧ ¬ is_triangle 1 1 3 ∧ ¬ is_triangle 3 4 9 :=
by {
  -- Proof goes here
  sorry
}

end NUMINAMATH_GPT_triangle_inequality_theorem_l1566_156688


namespace NUMINAMATH_GPT_smallest_sum_of_squares_l1566_156699

theorem smallest_sum_of_squares (x y : ℕ) (h : x^2 - y^2 = 221) : x^2 + y^2 ≥ 229 :=
sorry

end NUMINAMATH_GPT_smallest_sum_of_squares_l1566_156699


namespace NUMINAMATH_GPT_one_three_digit_cube_divisible_by_16_l1566_156655

theorem one_three_digit_cube_divisible_by_16 :
  ∃! (n : ℕ), (100 ≤ n ∧ n < 1000 ∧ ∃ (k : ℕ), n = k^3 ∧ 16 ∣ n) :=
sorry

end NUMINAMATH_GPT_one_three_digit_cube_divisible_by_16_l1566_156655


namespace NUMINAMATH_GPT_eqn_of_line_through_intersection_parallel_eqn_of_line_perpendicular_distance_l1566_156613

-- Proof 1: Line through intersection and parallel
theorem eqn_of_line_through_intersection_parallel :
  ∃ k : ℝ, (9 : ℝ) * (x: ℝ) + (18: ℝ) * (y: ℝ) - 4 = 0 ∧
           (∀ x y : ℝ, (2 * x + 3 * y - 5 = 0) → (7 * x + 15 * y + 1 = 0) → (x + 2 * y + k = 0)) :=
sorry

-- Proof 2: Line perpendicular and specific distance from origin
theorem eqn_of_line_perpendicular_distance :
  ∃ k : ℝ, (∃ m : ℝ, (k = 30 ∨ k = -30) ∧ (4 * (x: ℝ) - 3 * (y: ℝ) + m = 0 ∧ (∃ d : ℝ, d = 6 ∧ (|m| / (4 ^ 2 + (-3) ^ 2).sqrt) = d))) :=
sorry

end NUMINAMATH_GPT_eqn_of_line_through_intersection_parallel_eqn_of_line_perpendicular_distance_l1566_156613


namespace NUMINAMATH_GPT_cost_of_each_art_book_l1566_156627

-- Define the conditions
def total_cost : ℕ := 30
def cost_per_math_and_science_book : ℕ := 3
def num_math_books : ℕ := 2
def num_art_books : ℕ := 3
def num_science_books : ℕ := 6

-- The proof problem statement
theorem cost_of_each_art_book :
  (total_cost - (num_math_books * cost_per_math_and_science_book + num_science_books * cost_per_math_and_science_book)) / num_art_books = 2 :=
by
  sorry -- proof goes here,

end NUMINAMATH_GPT_cost_of_each_art_book_l1566_156627


namespace NUMINAMATH_GPT_three_pair_probability_l1566_156651

theorem three_pair_probability :
  let total_combinations := Nat.choose 52 5
  let three_pair_combinations := 13 * 4 * 12 * 4
  total_combinations = 2598960 ∧ three_pair_combinations = 2496 →
  (three_pair_combinations : ℚ) / total_combinations = 2496 / 2598960 :=
by
  -- Definitions and computations can be added here if necessary
  sorry

end NUMINAMATH_GPT_three_pair_probability_l1566_156651


namespace NUMINAMATH_GPT_find_equation_AC_l1566_156634

noncomputable def triangleABC (A B C : (ℝ × ℝ)) : Prop :=
  B = (-2, 0) ∧ 
  ∃ (lineAB : ℝ × ℝ → ℝ), ∀ P, lineAB P = 3 * P.1 - P.2 + 6 

noncomputable def conditions (A B : (ℝ × ℝ)) : Prop :=
  (3 * B.1 - B.2 + 6 = 0) ∧ 
  (B.1 + 3 * B.2 - 26 = 0) ∧
  (A.1 + A.2 - 2 = 0)

noncomputable def equationAC (A C : (ℝ × ℝ)) : Prop :=
  (C.1 - 3 * C.2 + 10 = 0)

theorem find_equation_AC (A B C : (ℝ × ℝ)) (h₁ : triangleABC A B C) (h₂ : conditions A B) : 
  equationAC A C :=
sorry

end NUMINAMATH_GPT_find_equation_AC_l1566_156634


namespace NUMINAMATH_GPT_power_mod_residue_l1566_156622

theorem power_mod_residue (n : ℕ) (h : n = 1234) : (7^n) % 19 = 9 := by
  sorry

end NUMINAMATH_GPT_power_mod_residue_l1566_156622


namespace NUMINAMATH_GPT_perpendicular_lines_l1566_156615

theorem perpendicular_lines (a : ℝ) : 
  ∀ x y : ℝ, 3 * y - x + 4 = 0 → 4 * y + a * x + 5 = 0 → a = 12 :=
by
  sorry

end NUMINAMATH_GPT_perpendicular_lines_l1566_156615


namespace NUMINAMATH_GPT_problem_statement_l1566_156693

noncomputable def f (x k : ℝ) := x^3 / (2^x + k * 2^(-x))

def is_even_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

def k2_eq_1_is_nec_but_not_suff (f : ℝ → ℝ) (k : ℝ) : Prop :=
  (k^2 = 1) → (is_even_function f → k = -1 ∧ ¬(k = 1))

theorem problem_statement (k : ℝ) :
  k2_eq_1_is_nec_but_not_suff (λ x => f x k) k :=
by
  sorry

end NUMINAMATH_GPT_problem_statement_l1566_156693


namespace NUMINAMATH_GPT_find_m_value_l1566_156600

theorem find_m_value (x m : ℝ)
  (h1 : -3 * x = -5 * x + 4)
  (h2 : m^x - 9 = 0) :
  m = 3 ∨ m = -3 := 
sorry

end NUMINAMATH_GPT_find_m_value_l1566_156600


namespace NUMINAMATH_GPT_find_other_number_l1566_156644

-- Given conditions
def sum_of_numbers (x y : ℕ) : Prop := x + y = 72
def number_difference (x y : ℕ) : Prop := x = y + 12
def one_number_is_30 (x : ℕ) : Prop := x = 30

-- Theorem to prove
theorem find_other_number (y : ℕ) : 
  sum_of_numbers y 30 ∧ number_difference 30 y → y = 18 := by
  sorry

end NUMINAMATH_GPT_find_other_number_l1566_156644


namespace NUMINAMATH_GPT_five_crows_two_hours_l1566_156673

-- Define the conditions and the question as hypotheses
def crows_worms (crows worms hours : ℕ) := 
  (crows = 3) ∧ (worms = 30) ∧ (hours = 1)

theorem five_crows_two_hours 
  (c: ℕ) (w: ℕ) (h: ℕ)
  (H: crows_worms c w h)
  : ∃ worms_eaten : ℕ, worms_eaten = 100 :=
by
  sorry

end NUMINAMATH_GPT_five_crows_two_hours_l1566_156673


namespace NUMINAMATH_GPT_no_solution_to_a_l1566_156691

theorem no_solution_to_a (x : ℝ) :
  (4 * x - 1) / 6 - (5 * x - 2 / 3) / 10 + (9 - x / 2) / 3 ≠ 101 / 20 := 
sorry

end NUMINAMATH_GPT_no_solution_to_a_l1566_156691


namespace NUMINAMATH_GPT_smallest_positive_n_l1566_156624

theorem smallest_positive_n (n : ℕ) : n > 0 → (3 * n ≡ 1367 [MOD 26]) → n = 5 :=
by
  intros _ _
  sorry

end NUMINAMATH_GPT_smallest_positive_n_l1566_156624


namespace NUMINAMATH_GPT_kosher_clients_count_l1566_156609

def T := 30
def V := 7
def VK := 3
def Neither := 18

theorem kosher_clients_count (K : ℕ) : T - Neither = V + K - VK → K = 8 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_kosher_clients_count_l1566_156609


namespace NUMINAMATH_GPT_rectangle_area_l1566_156625

theorem rectangle_area (side_length : ℝ) (rect_width : ℝ) (rect_length : ℝ) : 
  side_length^2 = 64 → 
  rect_width = side_length →
  rect_length = 3 * rect_width →
  rect_width * rect_length = 192 := 
by
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_rectangle_area_l1566_156625


namespace NUMINAMATH_GPT_tablet_battery_life_l1566_156650

noncomputable def battery_life_remaining
  (no_use_life : ℝ) (use_life : ℝ) (total_on_time : ℝ) (use_time : ℝ) : ℝ :=
  let no_use_consumption_rate := 1 / no_use_life
  let use_consumption_rate := 1 / use_life
  let no_use_time := total_on_time - use_time
  let total_battery_used := no_use_time * no_use_consumption_rate + use_time * use_consumption_rate
  let remaining_battery := 1 - total_battery_used
  remaining_battery / no_use_consumption_rate

theorem tablet_battery_life (no_use_life : ℝ) (use_life : ℝ) (total_on_time : ℝ) (use_time : ℝ) :
  battery_life_remaining no_use_life use_life total_on_time use_time = 6 :=
by
  -- The proof will go here, we use sorry for now to skip the proof step.
  sorry

end NUMINAMATH_GPT_tablet_battery_life_l1566_156650


namespace NUMINAMATH_GPT_extracellular_proof_l1566_156640

-- Define the components
def component1 : Set String := {"Na＋", "antibodies", "plasma proteins"}
def component2 : Set String := {"Hemoglobin", "O2", "glucose"}
def component3 : Set String := {"glucose", "CO2", "insulin"}
def component4 : Set String := {"Hormones", "neurotransmitter vesicles", "amino acids"}

-- Define the properties of being a part of the extracellular fluid
def is_extracellular (x : Set String) : Prop :=
  x = component1 ∨ x = component3

-- State the theorem to prove
theorem extracellular_proof : is_extracellular component1 ∧ ¬is_extracellular component2 ∧ is_extracellular component3 ∧ ¬is_extracellular component4 :=
by
  sorry

end NUMINAMATH_GPT_extracellular_proof_l1566_156640


namespace NUMINAMATH_GPT_two_digit_decimal_bounds_l1566_156635

def is_approximate (original approx : ℝ) : Prop :=
  abs (original - approx) < 0.05

theorem two_digit_decimal_bounds :
  ∃ max min : ℝ, is_approximate 15.6 max ∧ max = 15.64 ∧ is_approximate 15.6 min ∧ min = 15.55 :=
by
  sorry

end NUMINAMATH_GPT_two_digit_decimal_bounds_l1566_156635


namespace NUMINAMATH_GPT_cost_in_chinese_yuan_l1566_156695

theorem cost_in_chinese_yuan
  (usd_to_nad : ℝ := 8)
  (usd_to_cny : ℝ := 5)
  (sculpture_cost_nad : ℝ := 160) :
  sculpture_cost_nad / usd_to_nad * usd_to_cny = 100 := 
by
  sorry

end NUMINAMATH_GPT_cost_in_chinese_yuan_l1566_156695


namespace NUMINAMATH_GPT_remaining_puppies_l1566_156601

def initial_puppies : Nat := 8
def given_away_puppies : Nat := 4

theorem remaining_puppies : initial_puppies - given_away_puppies = 4 := 
by 
  sorry

end NUMINAMATH_GPT_remaining_puppies_l1566_156601


namespace NUMINAMATH_GPT_kaleb_candy_problem_l1566_156682

-- Define the initial problem with given conditions

theorem kaleb_candy_problem :
  ∀ (total_boxes : ℕ) (given_away_boxes : ℕ) (pieces_per_box : ℕ),
    total_boxes = 14 →
    given_away_boxes = 5 →
    pieces_per_box = 6 →
    (total_boxes - given_away_boxes) * pieces_per_box = 54 :=
by
  intros total_boxes given_away_boxes pieces_per_box
  intros h1 h2 h3
  -- Use assumptions
  sorry

end NUMINAMATH_GPT_kaleb_candy_problem_l1566_156682


namespace NUMINAMATH_GPT_jawbreakers_in_package_correct_l1566_156603

def jawbreakers_ate : Nat := 20
def jawbreakers_left : Nat := 4
def jawbreakers_in_package : Nat := jawbreakers_ate + jawbreakers_left

theorem jawbreakers_in_package_correct : jawbreakers_in_package = 24 := by
  sorry

end NUMINAMATH_GPT_jawbreakers_in_package_correct_l1566_156603


namespace NUMINAMATH_GPT_complex_number_first_quadrant_l1566_156645

theorem complex_number_first_quadrant (z : ℂ) (h : z = (i - 1) / i) : 
  ∃ x y : ℝ, z = x + y * I ∧ x > 0 ∧ y > 0 := 
sorry

end NUMINAMATH_GPT_complex_number_first_quadrant_l1566_156645


namespace NUMINAMATH_GPT_ratio_b_a_4_l1566_156642

theorem ratio_b_a_4 (a b : ℚ) (h1 : b / a = 4) (h2 : b = 15 - 6 * a) : a = 3 / 2 :=
by
  sorry

end NUMINAMATH_GPT_ratio_b_a_4_l1566_156642


namespace NUMINAMATH_GPT_inverse_negation_l1566_156639

theorem inverse_negation :
  (∀ x : ℝ, x ≥ 3 → x < 0) ↔ (∃ x : ℝ, x ≥ 0 ∧ ¬ (x < 3)) :=
by
  sorry

end NUMINAMATH_GPT_inverse_negation_l1566_156639


namespace NUMINAMATH_GPT_seed_germination_probability_l1566_156663

-- Define necessary values and variables
def n : ℕ := 3
def p : ℚ := 0.7
def k : ℕ := 2

-- Define the binomial probability formula
def binomial_probability (n : ℕ) (k : ℕ) (p : ℚ) : ℚ :=
  (Nat.choose n k) * (p^k) * ((1 - p)^(n - k))

-- State the proof problem
theorem seed_germination_probability :
  binomial_probability n k p = 0.441 := 
sorry

end NUMINAMATH_GPT_seed_germination_probability_l1566_156663


namespace NUMINAMATH_GPT_vector_perpendicular_solution_l1566_156612

noncomputable def a (m : ℝ) : ℝ × ℝ := (1, m)
def b : ℝ × ℝ := (3, -2)

def dot_product (v w : ℝ × ℝ) : ℝ :=
  v.1 * w.1 + v.2 * w.2

theorem vector_perpendicular_solution (m : ℝ) (h : dot_product (a m + b) b = 0) : m = 8 := by
  sorry

end NUMINAMATH_GPT_vector_perpendicular_solution_l1566_156612


namespace NUMINAMATH_GPT_average_cost_per_trip_is_correct_l1566_156607

def oldest_pass_cost : ℕ := 100
def second_oldest_pass_cost : ℕ := 90
def third_oldest_pass_cost : ℕ := 80
def youngest_pass_cost : ℕ := 70

def oldest_trips : ℕ := 35
def second_oldest_trips : ℕ := 25
def third_oldest_trips : ℕ := 20
def youngest_trips : ℕ := 15

def total_cost : ℕ := oldest_pass_cost + second_oldest_pass_cost + third_oldest_pass_cost + youngest_pass_cost
def total_trips : ℕ := oldest_trips + second_oldest_trips + third_oldest_trips + youngest_trips

def average_cost_per_trip : ℚ := total_cost / total_trips

theorem average_cost_per_trip_is_correct : average_cost_per_trip = 340 / 95 :=
by sorry

end NUMINAMATH_GPT_average_cost_per_trip_is_correct_l1566_156607


namespace NUMINAMATH_GPT_inequality_proof_l1566_156610

noncomputable def a (x1 x2 x3 x4 x5 : ℝ) := x1 + x2 + x3 + x4 + x5
noncomputable def b (x1 x2 x3 x4 x5 : ℝ) := x1 * x2 + x1 * x3 + x1 * x4 + x1 * x5 + x2 * x3 + x2 * x4 + x2 * x5 + x3 * x4 + x3 * x5 + x4 * x5
noncomputable def c (x1 x2 x3 x4 x5 : ℝ) := x1 * x2 * x3 + x1 * x2 * x4 + x1 * x2 * x5 + x1 * x3 * x4 + x1 * x3 * x5 + x1 * x4 * x5 + x2 * x3 * x4 + x2 * x3 * x5 + x2 * x4 * x5 + x3 * x4 * x5
noncomputable def d (x1 x2 x3 x4 x5 : ℝ) := x1 * x2 * x3 * x4 + x1 * x2 * x3 * x5 + x1 * x2 * x4 * x5 + x1 * x3 * x4 * x5 + x2 * x3 * x4 * x5

theorem inequality_proof (x1 x2 x3 x4 x5 : ℝ) (hx1x2x3x4x5 : x1 * x2 * x3 * x4 * x5 = 1) :
  (1 / a x1 x2 x3 x4 x5) + (1 / b x1 x2 x3 x4 x5) + (1 / c x1 x2 x3 x4 x5) + (1 / d x1 x2 x3 x4 x5) ≤ 3 / 5 := 
sorry

end NUMINAMATH_GPT_inequality_proof_l1566_156610


namespace NUMINAMATH_GPT_least_money_Moe_l1566_156614

theorem least_money_Moe (Bo Coe Flo Jo Moe Zoe : ℝ)
  (H1 : Flo > Jo) 
  (H2 : Flo > Bo) 
  (H3 : Bo > Zoe) 
  (H4 : Coe > Zoe) 
  (H5 : Jo > Zoe) 
  (H6 : Bo > Jo) 
  (H7 : Zoe > Moe) : 
  (Moe < Bo) ∧ (Moe < Coe) ∧ (Moe < Flo) ∧ (Moe < Jo) ∧ (Moe < Zoe) :=
by
  sorry

end NUMINAMATH_GPT_least_money_Moe_l1566_156614


namespace NUMINAMATH_GPT_factor_expression_l1566_156666

theorem factor_expression (b : ℝ) : 56 * b^3 + 168 * b^2 = 56 * b^2 * (b + 3) :=
by
  sorry

end NUMINAMATH_GPT_factor_expression_l1566_156666


namespace NUMINAMATH_GPT_smallest_fraction_numerator_l1566_156681

theorem smallest_fraction_numerator :
  ∃ (a b : ℕ), (10 ≤ a ∧ a ≤ 99) ∧ (10 ≤ b ∧ b ≤ 99) ∧ (a * 4 > b * 3) ∧ (a = 73) := 
sorry

end NUMINAMATH_GPT_smallest_fraction_numerator_l1566_156681


namespace NUMINAMATH_GPT_actual_price_of_food_l1566_156677

theorem actual_price_of_food (P : ℝ) (h : 1.32 * P = 132) : P = 100 := 
by
  sorry

end NUMINAMATH_GPT_actual_price_of_food_l1566_156677


namespace NUMINAMATH_GPT_solve_for_x_l1566_156620
-- Import the entire Mathlib library

-- Define the condition
def condition (x : ℝ) := (72 - x)^2 = x^2

-- State the theorem
theorem solve_for_x : ∃ x : ℝ, condition x ∧ x = 36 :=
by {
  -- The proof will be provided here
  sorry
}

end NUMINAMATH_GPT_solve_for_x_l1566_156620


namespace NUMINAMATH_GPT_max_product_of_slopes_l1566_156676

theorem max_product_of_slopes 
  (m₁ m₂ : ℝ)
  (h₁ : m₂ = 3 * m₁)
  (h₂ : abs ((m₂ - m₁) / (1 + m₁ * m₂)) = Real.sqrt 3) :
  m₁ * m₂ ≤ 2 :=
sorry

end NUMINAMATH_GPT_max_product_of_slopes_l1566_156676


namespace NUMINAMATH_GPT_chord_segments_division_l1566_156626

-- Definitions based on the conditions
variables (R OM : ℝ) (AB : ℝ)
-- Setting the values as the problem provides 
def radius : ℝ := 15
def distance_from_center : ℝ := 13
def chord_length : ℝ := 18

-- Formulate the problem statement as a theorem
theorem chord_segments_division :
  ∃ (AM MB : ℝ), AM = 14 ∧ MB = 4 :=
by
  let CB := chord_length / 2
  let OC := Real.sqrt (radius^2 - CB^2)
  let MC := Real.sqrt (distance_from_center^2 - OC^2)
  let AM := CB + MC
  let MB := CB - MC
  use AM, MB
  sorry

end NUMINAMATH_GPT_chord_segments_division_l1566_156626


namespace NUMINAMATH_GPT_solve_equations_l1566_156686

theorem solve_equations (x y : ℝ) (h1 : (x + y) / x = y / (x + y)) (h2 : x = 2 * y) :
  x = 0 ∧ y = 0 :=
by
  sorry

end NUMINAMATH_GPT_solve_equations_l1566_156686


namespace NUMINAMATH_GPT_sin2_cos3_tan4_lt_zero_l1566_156608

theorem sin2_cos3_tan4_lt_zero (h1 : Real.sin 2 > 0) (h2 : Real.cos 3 < 0) (h3 : Real.tan 4 > 0) : Real.sin 2 * Real.cos 3 * Real.tan 4 < 0 :=
sorry

end NUMINAMATH_GPT_sin2_cos3_tan4_lt_zero_l1566_156608


namespace NUMINAMATH_GPT_common_root_iff_cond_l1566_156643

theorem common_root_iff_cond (p1 p2 q1 q2 : ℂ) :
  (∃ x : ℂ, x^2 + p1 * x + q1 = 0 ∧ x^2 + p2 * x + q2 = 0) ↔
  (q2 - q1)^2 + (p1 - p2) * (p1 * q2 - q1 * p2) = 0 :=
by
  sorry

end NUMINAMATH_GPT_common_root_iff_cond_l1566_156643


namespace NUMINAMATH_GPT_relationship_of_y_values_l1566_156648

noncomputable def quadratic_function (x : ℝ) (c : ℝ) := x^2 - 6*x + c

theorem relationship_of_y_values (c : ℝ) (y1 y2 y3 : ℝ) :
  quadratic_function 1 c = y1 →
  quadratic_function (2 * Real.sqrt 2) c = y2 →
  quadratic_function 4 c = y3 →
  y3 < y2 ∧ y2 < y1 :=
by
  intros hA hB hC
  sorry

end NUMINAMATH_GPT_relationship_of_y_values_l1566_156648


namespace NUMINAMATH_GPT_ratio_of_fuji_trees_l1566_156647

variable (F T : ℕ) -- Declaring F as number of pure Fuji trees, T as total number of trees
variables (C : ℕ) -- Declaring C as number of cross-pollinated trees 

theorem ratio_of_fuji_trees 
  (h1: 10 * C = T) 
  (h2: F + C = 221) 
  (h3: T = F + 39 + C) : 
  F * 52 = 39 * T := 
sorry

end NUMINAMATH_GPT_ratio_of_fuji_trees_l1566_156647


namespace NUMINAMATH_GPT_nurse_distribution_l1566_156633

theorem nurse_distribution (nurses hospitals : ℕ) (h1 : nurses = 3) (h2 : hospitals = 6) 
  (h3 : ∀ (a b c : ℕ), a = b → b = c → a = c → a ≤ 2) : 
  (hospitals^nurses - hospitals) = 210 := 
by 
  sorry

end NUMINAMATH_GPT_nurse_distribution_l1566_156633


namespace NUMINAMATH_GPT_parallel_segments_have_equal_slopes_l1566_156611

theorem parallel_segments_have_equal_slopes
  (A B X : ℝ × ℝ)
  (Y : ℝ × ℝ)
  (hA : A = (-5, -1))
  (hB : B = (2, -8))
  (hX : X = (2, 10))
  (hY1 : Y.1 = 20)
  (h_parallel : (B.2 - A.2) / (B.1 - A.1) = (Y.2 - X.2) / (Y.1 - X.1)) :
  Y.2 = -8 :=
by
  sorry

end NUMINAMATH_GPT_parallel_segments_have_equal_slopes_l1566_156611


namespace NUMINAMATH_GPT_eval_expr_l1566_156631

theorem eval_expr : 3^2 * 4 * 6^3 * Nat.factorial 7 = 39191040 := by
  -- the proof will be filled in here
  sorry

end NUMINAMATH_GPT_eval_expr_l1566_156631


namespace NUMINAMATH_GPT_resulting_figure_has_25_sides_l1566_156604

/-- Consider a sequential construction starting with an isosceles triangle, adding a rectangle 
    on one side, then a regular hexagon on a non-adjacent side of the rectangle, followed by a
    regular heptagon, another regular hexagon, and finally, a regular nonagon. -/
def sides_sequence : List ℕ := [3, 4, 6, 7, 6, 9]

/-- The number of sides exposed to the outside in the resulting figure. -/
def exposed_sides (sides : List ℕ) : ℕ :=
  let total_sides := sides.sum
  let adjacent_count := 2 + 2 + 2 + 2 + 1
  total_sides - adjacent_count

theorem resulting_figure_has_25_sides :
  exposed_sides sides_sequence = 25 := 
by
  sorry

end NUMINAMATH_GPT_resulting_figure_has_25_sides_l1566_156604


namespace NUMINAMATH_GPT_find_a_l1566_156671

noncomputable def A (a : ℝ) : ℝ × ℝ := (a, 2)
def B : ℝ × ℝ := (5, 1)
noncomputable def C (a : ℝ) : ℝ × ℝ := (-4, 2 * a)

def collinear (A B C : ℝ × ℝ) : Prop :=
  let (x1, y1) := A
  let (x2, y2) := B
  let (x3, y3) := C
  (y2 - y1) * (x3 - x1) = (y3 - y1) * (x2 - x1)

theorem find_a (a : ℝ) : collinear (A a) B (C a) ↔ a = 4 :=
by
  sorry

end NUMINAMATH_GPT_find_a_l1566_156671


namespace NUMINAMATH_GPT_paul_and_paula_cookies_l1566_156623

-- Define the number of cookies per pack type
def cookies_in_pack (pack : ℕ) : ℕ :=
  match pack with
  | 1 => 15
  | 2 => 30
  | 3 => 45
  | 4 => 60
  | _ => 0

-- Paul's purchase: 2 packs of Pack B and 1 pack of Pack A
def pauls_cookies : ℕ :=
  2 * cookies_in_pack 2 + cookies_in_pack 1

-- Paula's purchase: 1 pack of Pack A and 1 pack of Pack C
def paulas_cookies : ℕ :=
  cookies_in_pack 1 + cookies_in_pack 3

-- Total number of cookies Paul and Paula have
def total_cookies : ℕ :=
  pauls_cookies + paulas_cookies

theorem paul_and_paula_cookies : total_cookies = 135 :=
by
  sorry

end NUMINAMATH_GPT_paul_and_paula_cookies_l1566_156623


namespace NUMINAMATH_GPT_volleyball_team_selection_l1566_156670

noncomputable def volleyball_squad_count (n m k : ℕ) : ℕ :=
  n * (Nat.choose m k)

theorem volleyball_team_selection :
  volleyball_squad_count 12 11 7 = 3960 :=
by
  sorry

end NUMINAMATH_GPT_volleyball_team_selection_l1566_156670


namespace NUMINAMATH_GPT_relationship_between_p_and_q_l1566_156675

theorem relationship_between_p_and_q (p q : ℝ) 
  (h : ∃ x : ℝ, (x^2 + p*x + q = 0) ∧ (2*x)^2 + p*(2*x) + q = 0) :
  2 * p^2 = 9 * q :=
sorry

end NUMINAMATH_GPT_relationship_between_p_and_q_l1566_156675


namespace NUMINAMATH_GPT_even_number_representation_l1566_156678

-- Definitions for conditions
def even_number (k : Int) : Prop := ∃ m : Int, k = 2 * m
def perfect_square (n : Int) : Prop := ∃ p : Int, n = p * p
def sum_representation (a b : Int) : Prop := ∃ k : Int, a + b = 2 * k ∧ perfect_square (a * b)
def difference_representation (d k e : Int) : Prop := d * (d - 2 * k) = e * e

-- The theorem statement
theorem even_number_representation {k : Int} (hk : even_number k) :
  (∃ a b : Int, sum_representation a b ∧ 2 * k = a + b) ∨
  (∃ d e : Int, difference_representation d k e ∧ d ≠ 0) :=
sorry

end NUMINAMATH_GPT_even_number_representation_l1566_156678


namespace NUMINAMATH_GPT_find_divisor_l1566_156669

-- Condition Definitions
def dividend : ℕ := 725
def quotient : ℕ := 20
def remainder : ℕ := 5

-- Target Proof Statement
theorem find_divisor (divisor : ℕ) (h : dividend = divisor * quotient + remainder) : divisor = 36 := by
  sorry

end NUMINAMATH_GPT_find_divisor_l1566_156669


namespace NUMINAMATH_GPT_shopkeeper_gain_percentage_l1566_156646

noncomputable def gain_percentage (false_weight: ℕ) (true_weight: ℕ) : ℝ :=
  (↑(true_weight - false_weight) / ↑false_weight) * 100

theorem shopkeeper_gain_percentage :
  gain_percentage 960 1000 = 4.166666666666667 := 
sorry

end NUMINAMATH_GPT_shopkeeper_gain_percentage_l1566_156646


namespace NUMINAMATH_GPT_chips_probability_l1566_156606

/-- A bag contains 4 green, 3 orange, and 5 blue chips. If the 12 chips are randomly drawn from
    the bag, one at a time and without replacement, the probability that the chips are drawn such
    that the 4 green chips are drawn consecutively, the 3 orange chips are drawn consecutively,
    and the 5 blue chips are drawn consecutively, but not necessarily in the green-orange-blue
    order, is 1/4620. -/
theorem chips_probability :
  let total_chips := 12
  let factorial := Nat.factorial
  let favorable_outcomes := (factorial 3) * (factorial 4) * (factorial 3) * (factorial 5)
  let total_outcomes := factorial total_chips
  favorable_outcomes / total_outcomes = 1 / 4620 :=
by
  -- proof goes here, but we skip it
  sorry

end NUMINAMATH_GPT_chips_probability_l1566_156606


namespace NUMINAMATH_GPT_rent_for_additional_hour_l1566_156662

theorem rent_for_additional_hour (x : ℝ) :
  (25 + 10 * x = 125) → (x = 10) :=
by 
  sorry

end NUMINAMATH_GPT_rent_for_additional_hour_l1566_156662


namespace NUMINAMATH_GPT_genuine_items_count_l1566_156602

def total_purses : ℕ := 26
def total_handbags : ℕ := 24
def fake_purses : ℕ := total_purses / 2
def fake_handbags : ℕ := total_handbags / 4
def genuine_purses : ℕ := total_purses - fake_purses
def genuine_handbags : ℕ := total_handbags - fake_handbags

theorem genuine_items_count : genuine_purses + genuine_handbags = 31 := by
  sorry

end NUMINAMATH_GPT_genuine_items_count_l1566_156602


namespace NUMINAMATH_GPT_cost_D_to_E_l1566_156641

def distance_DF (DF DE EF : ℝ) : Prop :=
  DE^2 = DF^2 + EF^2

def cost_to_fly (distance : ℝ) (per_kilometer_cost booking_fee : ℝ) : ℝ :=
  distance * per_kilometer_cost + booking_fee

noncomputable def total_cost_to_fly_from_D_to_E : ℝ :=
  let DE := 3750 -- Distance from D to E (km)
  let booking_fee := 120 -- Booking fee in dollars
  let per_kilometer_cost := 0.12 -- Cost per kilometer in dollars
  cost_to_fly DE per_kilometer_cost booking_fee

theorem cost_D_to_E : total_cost_to_fly_from_D_to_E = 570 := by
  sorry

end NUMINAMATH_GPT_cost_D_to_E_l1566_156641


namespace NUMINAMATH_GPT_perimeter_of_square_is_32_l1566_156667

-- Given conditions
def radius := 4
def diameter := 2 * radius
def side_length_of_square := diameter

-- Question: What is the perimeter of the square?
def perimeter_of_square := 4 * side_length_of_square

-- Proof statement
theorem perimeter_of_square_is_32 : perimeter_of_square = 32 :=
sorry

end NUMINAMATH_GPT_perimeter_of_square_is_32_l1566_156667


namespace NUMINAMATH_GPT_arithmetic_sequence_properties_l1566_156616

-- Definitions and conditions
def S (n : ℕ) : ℤ := -2 * n^2 + 15 * n

-- Statement of the problem as a theorem
theorem arithmetic_sequence_properties :
  (∀ n : ℕ, S (n + 1) - S n = 17 - 4 * (n + 1)) ∧
  (∃ n : ℕ, S n = 28 ∧ ∀ m : ℕ, S m ≤ S n) :=
by {sorry}

end NUMINAMATH_GPT_arithmetic_sequence_properties_l1566_156616


namespace NUMINAMATH_GPT_set_union_proof_l1566_156619

  open Set

  def M : Set ℕ := {0, 1, 3}
  def N : Set ℕ := {x | ∃ a, a ∈ M ∧ x = 3 * a}

  theorem set_union_proof : M ∪ N = {0, 1, 3, 9} :=
  by
    sorry
  
end NUMINAMATH_GPT_set_union_proof_l1566_156619


namespace NUMINAMATH_GPT_Mike_watches_TV_every_day_l1566_156683

theorem Mike_watches_TV_every_day :
  (∃ T : ℝ, 
  (3 * (T / 2) + 7 * T = 34) 
  → T = 4) :=
by
  let T := 4
  sorry

end NUMINAMATH_GPT_Mike_watches_TV_every_day_l1566_156683


namespace NUMINAMATH_GPT_total_players_is_59_l1566_156665

-- Define the number of players from each sport.
def cricket_players : ℕ := 16
def hockey_players : ℕ := 12
def football_players : ℕ := 18
def softball_players : ℕ := 13

-- Define the total number of players as the sum of the above.
def total_players : ℕ :=
  cricket_players + hockey_players + football_players + softball_players

-- Prove that the total number of players is 59.
theorem total_players_is_59 :
  total_players = 59 :=
by
  unfold total_players
  unfold cricket_players
  unfold hockey_players
  unfold football_players
  unfold softball_players
  sorry

end NUMINAMATH_GPT_total_players_is_59_l1566_156665


namespace NUMINAMATH_GPT_area_triangle_BRS_l1566_156674

def point := ℝ × ℝ
def x_intercept (p : point) : ℝ := p.1
def y_intercept (p : point) : ℝ := p.2

noncomputable def distance_from_y_axis (p : point) : ℝ := abs p.1

theorem area_triangle_BRS (B R S : point)
  (hB : B = (4, 10))
  (h_perp : ∃ m₁ m₂, m₁ * m₂ = -1)
  (h_sum_zero : x_intercept R + x_intercept S = 0)
  (h_dist : distance_from_y_axis B = 10) :
  ∃ area : ℝ, area = 60 := 
sorry

end NUMINAMATH_GPT_area_triangle_BRS_l1566_156674


namespace NUMINAMATH_GPT_A_finish_work_in_6_days_l1566_156617

theorem A_finish_work_in_6_days :
  ∃ (x : ℕ), (1 / (12:ℚ) + 1 / (x:ℚ) = 1 / (4:ℚ)) → x = 6 :=
by
  sorry

end NUMINAMATH_GPT_A_finish_work_in_6_days_l1566_156617


namespace NUMINAMATH_GPT_hemisphere_surface_area_l1566_156649

theorem hemisphere_surface_area (r : ℝ) (π : ℝ) (area_base : ℝ) (surface_area_sphere : ℝ) (Q : ℝ) : 
  area_base = 3 ∧ surface_area_sphere = 4 * π * r^2 → Q = 9 :=
by
  sorry

end NUMINAMATH_GPT_hemisphere_surface_area_l1566_156649


namespace NUMINAMATH_GPT_value_of_a_if_1_in_S_l1566_156692

variable (a : ℤ)
def S := { x : ℤ | 3 * x + a = 0 }

theorem value_of_a_if_1_in_S (h : 1 ∈ S a) : a = -3 :=
sorry

end NUMINAMATH_GPT_value_of_a_if_1_in_S_l1566_156692


namespace NUMINAMATH_GPT_orangeade_price_l1566_156656

theorem orangeade_price (O W : ℝ) (h1 : O = W) (price_day1 : ℝ) (price_day2 : ℝ) 
    (volume_day1 : ℝ) (volume_day2 : ℝ) (revenue_day1 : ℝ) (revenue_day2 : ℝ) : 
    volume_day1 = 2 * O ∧ volume_day2 = 3 * O ∧ revenue_day1 = revenue_day2 ∧ price_day1 = 0.82 
    → price_day2 = 0.55 :=
by
    intros
    sorry

end NUMINAMATH_GPT_orangeade_price_l1566_156656


namespace NUMINAMATH_GPT_proof_no_natural_solutions_l1566_156689

noncomputable def no_natural_solutions : Prop :=
  ∀ x y : ℕ, y^2 ≠ x^2 + x + 1

theorem proof_no_natural_solutions : no_natural_solutions :=
by
  intros x y
  sorry

end NUMINAMATH_GPT_proof_no_natural_solutions_l1566_156689


namespace NUMINAMATH_GPT_cost_per_pack_l1566_156658

theorem cost_per_pack (total_bill : ℕ) (change_given : ℕ) (packs : ℕ) (total_cost := total_bill - change_given) (cost_per_pack := total_cost / packs) 
  (h1 : total_bill = 20) 
  (h2 : change_given = 11) 
  (h3 : packs = 3) : 
  cost_per_pack = 3 := by
  sorry

end NUMINAMATH_GPT_cost_per_pack_l1566_156658


namespace NUMINAMATH_GPT_shortest_remaining_side_l1566_156684

theorem shortest_remaining_side (a b c : ℝ) (h₁ : a = 5) (h₂ : c = 13) (h₃ : a^2 + b^2 = c^2) : b = 12 :=
by
  rw [h₁, h₂] at h₃
  sorry

end NUMINAMATH_GPT_shortest_remaining_side_l1566_156684


namespace NUMINAMATH_GPT_find_p_q_r_l1566_156690

def f (x : ℝ) : ℝ := x^2 + 2*x + 2
def g (x p q r : ℝ) : ℝ := x^3 + 2*x^2 + 6*p*x + 4*q*x + r

noncomputable def roots_sum_f := -2
noncomputable def roots_product_f := 2

theorem find_p_q_r (p q r : ℝ) (h1 : ∀ x, f x = 0 → g x p q r = 0) :
  (p + q) * r = 0 :=
sorry

end NUMINAMATH_GPT_find_p_q_r_l1566_156690


namespace NUMINAMATH_GPT_savings_by_going_earlier_l1566_156697

/-- Define the cost of evening ticket -/
def evening_ticket_cost : ℝ := 10

/-- Define the cost of large popcorn & drink combo -/
def food_combo_cost : ℝ := 10

/-- Define the discount percentage on tickets from 12 noon to 3 pm -/
def ticket_discount : ℝ := 0.20

/-- Define the discount percentage on food combos from 12 noon to 3 pm -/
def food_combo_discount : ℝ := 0.50

/-- Prove that the total savings Trip could achieve by going to the earlier movie is $7 -/
theorem savings_by_going_earlier : 
  (ticket_discount * evening_ticket_cost) + (food_combo_discount * food_combo_cost) = 7 := by
  sorry

end NUMINAMATH_GPT_savings_by_going_earlier_l1566_156697


namespace NUMINAMATH_GPT_Ava_watched_television_for_240_minutes_l1566_156605

-- Define the conditions
def hours (h : ℕ) := h = 4

-- Define the conversion factor from hours to minutes
def convert_hours_to_minutes (h : ℕ) : ℕ := h * 60

-- State the theorem
theorem Ava_watched_television_for_240_minutes (h : ℕ) (hh : hours h) : convert_hours_to_minutes h = 240 :=
by
  -- The proof goes here but is skipped
  sorry

end NUMINAMATH_GPT_Ava_watched_television_for_240_minutes_l1566_156605


namespace NUMINAMATH_GPT_nearest_integer_x_sub_y_l1566_156636

theorem nearest_integer_x_sub_y (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) 
  (h1 : |x| - y = 4) 
  (h2 : |x| * y - x^3 = 1) : 
  abs (x - y - 4) < 1 :=
sorry

end NUMINAMATH_GPT_nearest_integer_x_sub_y_l1566_156636


namespace NUMINAMATH_GPT_total_opaque_stackings_l1566_156659

-- Define the glass pane and its rotation
inductive Rotation
| deg_0 | deg_90 | deg_180 | deg_270
deriving DecidableEq, Repr

-- The property of opacity for a stack of glass panes
def isOpaque (stack : List (List Rotation)) : Bool :=
  -- The implementation of this part depends on the specific condition in the problem
  -- and here is abstracted out for the problem statement.
  sorry

-- The main problem stating the required number of ways
theorem total_opaque_stackings : ∃ (n : ℕ), n = 7200 :=
  sorry

end NUMINAMATH_GPT_total_opaque_stackings_l1566_156659


namespace NUMINAMATH_GPT_twelfth_term_l1566_156661

noncomputable def a (n : ℕ) : ℕ := 
  if n = 0 then 0 
  else (n * (n + 2)) - ((n - 1) * (n + 1))

theorem twelfth_term : a 12 = 25 :=
by sorry

end NUMINAMATH_GPT_twelfth_term_l1566_156661


namespace NUMINAMATH_GPT_largest_common_term_in_range_l1566_156628

theorem largest_common_term_in_range :
  ∃ (a : ℕ), a < 150 ∧ (∃ (n : ℕ), a = 3 + 8 * n) ∧ (∃ (n : ℕ), a = 5 + 9 * n) ∧ a = 131 :=
by
  sorry

end NUMINAMATH_GPT_largest_common_term_in_range_l1566_156628


namespace NUMINAMATH_GPT_seq_a_general_term_seq_b_general_term_inequality_k_l1566_156630

def seq_a (n : ℕ) : ℕ :=
if n = 1 then 2 else 2 * n - 1

def S (n : ℕ) : ℕ := 
match n with
| 0       => 0
| (n + 1) => S n + seq_a (n + 1)

def seq_b (n : ℕ) : ℕ := 3 ^ n

def T (n : ℕ) : ℕ := (3 ^ (n + 1) - 3) / 2

theorem seq_a_general_term (n : ℕ) : seq_a n = if n = 1 then 2 else 2 * n - 1 :=
sorry

theorem seq_b_general_term (n : ℕ) : seq_b n = 3 ^ n :=
sorry

theorem inequality_k (k : ℝ) : (∀ n : ℕ, n > 0 → (T n + 3/2 : ℝ) * k ≥ 3 * n - 6) ↔ k ≥ 2 / 27 :=
sorry

end NUMINAMATH_GPT_seq_a_general_term_seq_b_general_term_inequality_k_l1566_156630


namespace NUMINAMATH_GPT_two_digit_number_l1566_156680

theorem two_digit_number (x y : Nat) : 
  10 * x + y = 10 * x + y := 
by 
  sorry

end NUMINAMATH_GPT_two_digit_number_l1566_156680


namespace NUMINAMATH_GPT_mutually_exclusive_iff_complementary_l1566_156687

variables {Ω : Type} (A₁ A₂ : Set Ω) (S : Set Ω)

/-- Proposition A: Events A₁ and A₂ are mutually exclusive. -/
def mutually_exclusive : Prop := A₁ ∩ A₂ = ∅

/-- Proposition B: Events A₁ and A₂ are complementary. -/
def complementary : Prop := A₁ ∩ A₂ = ∅ ∧ A₁ ∪ A₂ = S

/-- Proposition A is a necessary but not sufficient condition for Proposition B. -/
theorem mutually_exclusive_iff_complementary :
  mutually_exclusive A₁ A₂ → (complementary A₁ A₂ S → mutually_exclusive A₁ A₂) ∧
  (¬(mutually_exclusive A₁ A₂ → complementary A₁ A₂ S)) :=
by
  sorry

end NUMINAMATH_GPT_mutually_exclusive_iff_complementary_l1566_156687


namespace NUMINAMATH_GPT_number_of_license_plates_l1566_156654

-- Define the alphabet size and digit size constants.
def num_letters : ℕ := 26
def num_digits : ℕ := 10

-- Define the number of letters in the license plate.
def letters_in_plate : ℕ := 3

-- Define the number of digits in the license plate.
def digits_in_plate : ℕ := 4

-- Calculating the total number of license plates possible as (26^3) * (10^4).
theorem number_of_license_plates : 
  (num_letters ^ letters_in_plate) * (num_digits ^ digits_in_plate) = 175760000 :=
by
  sorry

end NUMINAMATH_GPT_number_of_license_plates_l1566_156654


namespace NUMINAMATH_GPT_total_hours_proof_l1566_156694

-- Definitions and conditions
def kate_hours : ℕ := 22
def pat_hours : ℕ := 2 * kate_hours
def mark_hours : ℕ := kate_hours + 110

-- Statement of the proof problem
theorem total_hours_proof : pat_hours + kate_hours + mark_hours = 198 := by
  sorry

end NUMINAMATH_GPT_total_hours_proof_l1566_156694


namespace NUMINAMATH_GPT_cookies_none_of_ingredients_l1566_156698

theorem cookies_none_of_ingredients (c : ℕ) (o : ℕ) (r : ℕ) (a : ℕ) (total_cookies : ℕ) :
  total_cookies = 48 ∧ c = total_cookies / 3 ∧ o = (3 * total_cookies + 4) / 5 ∧ r = total_cookies / 2 ∧ a = total_cookies / 8 → 
  ∃ n, n = 19 ∧ (∀ k, k = total_cookies - max c (max o (max r a)) → k ≤ n) :=
by sorry

end NUMINAMATH_GPT_cookies_none_of_ingredients_l1566_156698


namespace NUMINAMATH_GPT_problem_equivalence_l1566_156632

theorem problem_equivalence : 4 * 4^3 - 16^60 / 16^57 = -3840 := by
  sorry

end NUMINAMATH_GPT_problem_equivalence_l1566_156632


namespace NUMINAMATH_GPT_second_yellow_probability_l1566_156657

-- Define the conditions in Lean
def BagA : Type := {marble : Int // marble ≥ 0}
def BagB : Type := {marble : Int // marble ≥ 0}
def BagC : Type := {marble : Int // marble ≥ 0}
def BagD : Type := {marble : Int // marble ≥ 0}

noncomputable def marbles_in_A := 4 + 5 + 2
noncomputable def marbles_in_B := 7 + 5
noncomputable def marbles_in_C := 3 + 7
noncomputable def marbles_in_D := 8 + 2

-- Probabilities of drawing specific colors from Bag A
noncomputable def prob_white_A := 4 / 11
noncomputable def prob_black_A := 5 / 11
noncomputable def prob_red_A := 2 / 11

-- Probabilities of drawing a yellow marble from Bags B, C and D
noncomputable def prob_yellow_B := 7 / 12
noncomputable def prob_yellow_C := 3 / 10
noncomputable def prob_yellow_D := 8 / 10

-- Expected probability that the second marble is yellow
noncomputable def prob_second_yellow : ℚ :=
  (prob_white_A * prob_yellow_B) + (prob_black_A * prob_yellow_C) + (prob_red_A * prob_yellow_D)

/-- Prove that the total probability the second marble drawn is yellow is 163/330. -/
theorem second_yellow_probability :
  prob_second_yellow = 163 / 330 := sorry

end NUMINAMATH_GPT_second_yellow_probability_l1566_156657


namespace NUMINAMATH_GPT_candidate_lost_by_votes_l1566_156660

theorem candidate_lost_by_votes :
  let candidate_votes := (31 / 100) * 6450
  let rival_votes := (69 / 100) * 6450
  candidate_votes <= 6450 ∧ rival_votes <= 6450 ∧ rival_votes - candidate_votes = 2451 :=
by
  let candidate_votes := (31 / 100) * 6450
  let rival_votes := (69 / 100) * 6450
  have h1: candidate_votes <= 6450 := sorry
  have h2: rival_votes <= 6450 := sorry
  have h3: rival_votes - candidate_votes = 2451 := sorry
  exact ⟨h1, h2, h3⟩

end NUMINAMATH_GPT_candidate_lost_by_votes_l1566_156660


namespace NUMINAMATH_GPT_fraction_sum_geq_zero_l1566_156638

theorem fraction_sum_geq_zero (a b c : ℝ) (h1 : a > b) (h2 : b > c) : 
  (1 / (a - b) + 1 / (b - c) + 4 / (c - a)) ≥ 0 := 
by 
  sorry

end NUMINAMATH_GPT_fraction_sum_geq_zero_l1566_156638


namespace NUMINAMATH_GPT_tutors_meet_again_l1566_156696

theorem tutors_meet_again (tim uma victor xavier: ℕ) (h1: tim = 5) (h2: uma = 6) (h3: victor = 9) (h4: xavier = 8) :
  Nat.lcm (Nat.lcm tim uma) (Nat.lcm victor xavier) = 360 := 
by 
  rw [h1, h2, h3, h4]
  show Nat.lcm (Nat.lcm 5 6) (Nat.lcm 9 8) = 360
  sorry

end NUMINAMATH_GPT_tutors_meet_again_l1566_156696


namespace NUMINAMATH_GPT_opposite_of_neg_three_l1566_156629

theorem opposite_of_neg_three : -(-3) = 3 := by
  sorry

end NUMINAMATH_GPT_opposite_of_neg_three_l1566_156629


namespace NUMINAMATH_GPT_smallest_n_l1566_156653

variable {a : ℕ → ℝ} -- the arithmetic sequence
noncomputable def d := a 2 - a 1  -- common difference

variable {S : ℕ → ℝ}  -- sum of the first n terms

-- conditions
axiom cond1 : a 66 < 0
axiom cond2 : a 67 > 0
axiom cond3 : a 67 > abs (a 66)

-- sum of the first n terms of the arithmetic sequence
noncomputable def sum_n (n : ℕ) : ℝ :=
  n * (a 1 + a n) / 2

theorem smallest_n (n : ℕ) : S n > 0 → n = 132 :=
by
  sorry

end NUMINAMATH_GPT_smallest_n_l1566_156653


namespace NUMINAMATH_GPT_complex_quadrant_l1566_156637

open Complex

theorem complex_quadrant (z : ℂ) (h : z = (2 - I) / (2 + I)) : 
  z.re > 0 ∧ z.im < 0 := 
by
  sorry

end NUMINAMATH_GPT_complex_quadrant_l1566_156637


namespace NUMINAMATH_GPT_compute_A_3_2_l1566_156672

namespace Ackermann

def A : ℕ → ℕ → ℕ
| 0, n     => n + 1
| m + 1, 0 => A m 1
| m + 1, n + 1 => A m (A (m + 1) n)

theorem compute_A_3_2 : A 3 2 = 12 :=
sorry

end Ackermann

end NUMINAMATH_GPT_compute_A_3_2_l1566_156672
