import Mathlib

namespace NUMINAMATH_GPT_people_per_car_l724_72495

theorem people_per_car (total_people : ℝ) (total_cars : ℝ) (h1 : total_people = 189) (h2 : total_cars = 3.0) : total_people / total_cars = 63 := 
by
  sorry

end NUMINAMATH_GPT_people_per_car_l724_72495


namespace NUMINAMATH_GPT_sum_geometric_series_l724_72476

theorem sum_geometric_series :
  let a := (1 : ℚ) / 5
  let r := (1 : ℚ) / 5
  let n := 8
  let S_n := a * (1 - r^n) / (1 - r)
  S_n = 195312 / 781250 := by
    sorry

end NUMINAMATH_GPT_sum_geometric_series_l724_72476


namespace NUMINAMATH_GPT_rectangle_area_is_588_l724_72442

-- Definitions based on the conditions of the problem
def radius : ℝ := 7
def diameter : ℝ := 2 * radius
def width : ℝ := diameter
def length : ℝ := 3 * width

-- The statement to prove that the area of the rectangle is 588
theorem rectangle_area_is_588 : length * width = 588 :=
by
  -- Omitted proof
  sorry

end NUMINAMATH_GPT_rectangle_area_is_588_l724_72442


namespace NUMINAMATH_GPT_rate_percent_is_10_l724_72430

theorem rate_percent_is_10
  (SI : ℕ) (P : ℕ) (T : ℕ) (R : ℕ) 
  (h1 : SI = 2500) (h2 : P = 5000) (h3 : T = 5) :
  R = 10 :=
by
  sorry

end NUMINAMATH_GPT_rate_percent_is_10_l724_72430


namespace NUMINAMATH_GPT_solve_system_of_equations_l724_72454

theorem solve_system_of_equations (x y m : ℝ) 
  (h1 : 2 * x + y = 7)
  (h2 : x + 2 * y = m - 3) 
  (h3 : x - y = 2) : m = 8 :=
by
  -- Proof part is replaced with sorry as mentioned
  sorry

end NUMINAMATH_GPT_solve_system_of_equations_l724_72454


namespace NUMINAMATH_GPT_ram_efficiency_eq_27_l724_72439

theorem ram_efficiency_eq_27 (R : ℕ) (h1 : ∀ Krish, 2 * (1 / (R : ℝ)) = 1 / Krish) 
  (h2 : ∀ s, 3 * (1 / (R : ℝ)) * s = 1 ↔ s = (9 : ℝ)) : R = 27 :=
sorry

end NUMINAMATH_GPT_ram_efficiency_eq_27_l724_72439


namespace NUMINAMATH_GPT_chord_length_l724_72478

-- Definitions and conditions for the problem
variables (A D B C G E F : Point)

-- Lengths and radii in the problem
noncomputable def radius : Real := 10
noncomputable def AB : Real := 20
noncomputable def BC : Real := 20
noncomputable def CD : Real := 20

-- Centers of circles
variables (O N P : Circle) (AN ND : Real)

-- Tangent properties and intersection points
variable (tangent_AG : Tangent AG P G)
variable (intersect_AG_N : Intersects AG N E F)

-- Given the geometry setup, prove the length of chord EF.
theorem chord_length (EF_length : Real) :
  EF_length = 2 * Real.sqrt 93.75 := sorry

end NUMINAMATH_GPT_chord_length_l724_72478


namespace NUMINAMATH_GPT_find_n_l724_72463

theorem find_n (n : ℕ) (h : n > 0) : 
  (3^n + 5^n) % (3^(n-1) + 5^(n-1)) = 0 ↔ n = 1 := 
by sorry

end NUMINAMATH_GPT_find_n_l724_72463


namespace NUMINAMATH_GPT_problem_lean_statement_l724_72429

open Real

noncomputable def f (x : ℝ) : ℝ := sqrt 3 * sin (2 * x) + cos (2 * x)
noncomputable def g (x : ℝ) : ℝ := f (x + π / 6)

theorem problem_lean_statement : 
  (∀ x, g x = 2 * cos (2 * x)) ∧ (∀ x, g (x) = g (-x)) ∧ (∀ x, g (x + π) = g (x)) :=
  sorry

end NUMINAMATH_GPT_problem_lean_statement_l724_72429


namespace NUMINAMATH_GPT_square_inscribed_in_right_triangle_side_length_l724_72492

theorem square_inscribed_in_right_triangle_side_length
  (A B C X Y Z W : ℝ × ℝ)
  (AB BC AC : ℝ)
  (square_side : ℝ)
  (h : 0 < square_side) :
  -- Define the lengths of sides of the triangle.
  AB = 3 ∧ BC = 4 ∧ AC = 5 ∧

  -- Define the square inscribed in the triangle
  (W.1 - A.1)^2 + (W.2 - A.2)^2 = square_side^2 ∧
  (X.1 - W.1)^2 + (X.2 - W.2)^2 = square_side^2 ∧
  (Y.1 - X.1)^2 + (Y.2 - X.2)^2 = square_side^2 ∧
  (Z.1 - W.1)^2 + (Z.2 - W.2)^2 = square_side^2 ∧
  (Z.1 - C.1)^2 + (Z.2 - C.2)^2 = square_side^2 ∧

  -- Points where square meets triangle sides
  X.1 = A.1 ∧ Z.1 = C.1 ∧ Y.1 = X.1 ∧ W.1 = Z.1 ∧ Z.2 = Y.2 ∧

  -- Right triangle condition
  (B.1 - A.1)^2 + (B.2 - A.2)^2 = AB^2 ∧
  (C.1 - B.1)^2 + (C.2 - B.2)^2 = BC^2 ∧
  (C.1 - A.1)^2 + (C.2 - A.2)^2 = AC^2 ∧
  
  -- Right angle at vertex B
  (B.1 - A.1) * (B.1 - C.1) + (B.2 - A.2) * (B.2 - C.2) = 0
  →
  -- Prove the side length of the inscribed square
  square_side = 60 / 37 :=
sorry

end NUMINAMATH_GPT_square_inscribed_in_right_triangle_side_length_l724_72492


namespace NUMINAMATH_GPT_minimum_value_of_g_gm_equal_10_implies_m_is_5_l724_72453

/-- Condition: Definition of the function y in terms of x and m -/
def y (x m : ℝ) : ℝ := x^2 + m * x - 4

/-- Theorem about finding the minimum value of g(m) -/
theorem minimum_value_of_g (m : ℝ) :
  ∃ g : ℝ, g = (if m ≥ -4 then 2 * m
      else if -8 < m ∧ m < -4 then -m^2 / 4 - 4
      else 4 * m + 12) := by
  sorry

/-- Theorem that if the minimum value of g(m) is 10, then m must be 5 -/
theorem gm_equal_10_implies_m_is_5 :
  ∃ m, (if m ≥ -4 then 2 * m
       else if -8 < m ∧ m < -4 then -m^2 / 4 - 4
       else 4 * m + 12) = 10 := by
  use 5
  sorry

end NUMINAMATH_GPT_minimum_value_of_g_gm_equal_10_implies_m_is_5_l724_72453


namespace NUMINAMATH_GPT_geometric_seq_inequality_l724_72472

theorem geometric_seq_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h : b^2 = a * c) : a^2 + b^2 + c^2 > (a - b + c)^2 :=
by
  sorry

end NUMINAMATH_GPT_geometric_seq_inequality_l724_72472


namespace NUMINAMATH_GPT_complex_power_difference_l724_72445

theorem complex_power_difference (i : ℂ) (h : i^2 = -1) : (1 + i)^10 - (1 - i)^10 = 64 * i := 
by sorry

end NUMINAMATH_GPT_complex_power_difference_l724_72445


namespace NUMINAMATH_GPT_max_value_sin2x_cos2x_l724_72409

open Real

theorem max_value_sin2x_cos2x (x : ℝ) (h : 0 ≤ x ∧ x ≤ π / 2) :
  (sin (2 * x) + cos (2 * x) ≤ sqrt 2) ∧
  (∃ y, (0 ≤ y ∧ y ≤ π / 2) ∧ (sin (2 * y) + cos (2 * y) = sqrt 2)) :=
by
  sorry

end NUMINAMATH_GPT_max_value_sin2x_cos2x_l724_72409


namespace NUMINAMATH_GPT_mul_102_102_l724_72408

theorem mul_102_102 : 102 * 102 = 10404 := by
  sorry

end NUMINAMATH_GPT_mul_102_102_l724_72408


namespace NUMINAMATH_GPT_symmetric_points_product_l724_72498

theorem symmetric_points_product (a b : ℝ) 
    (h1 : a + 2 = -4) 
    (h2 : b = 2) : 
    a * b = -12 := 
sorry

end NUMINAMATH_GPT_symmetric_points_product_l724_72498


namespace NUMINAMATH_GPT_simplify_expression_l724_72449

theorem simplify_expression (a b : ℂ) (x : ℂ) (hb : b ≠ 0) (ha : a ≠ b) (hx : x = a / b) :
  (a^2 + b^2) / (a^2 - b^2) = (x^2 + 1) / (x^2 - 1) :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_simplify_expression_l724_72449


namespace NUMINAMATH_GPT_factory_Y_bulbs_proportion_l724_72468

theorem factory_Y_bulbs_proportion :
  (0.60 * 0.59 + 0.40 * P_Y = 0.62) → (P_Y = 0.665) :=
by
  sorry

end NUMINAMATH_GPT_factory_Y_bulbs_proportion_l724_72468


namespace NUMINAMATH_GPT_prob_three_blue_is_correct_l724_72411

-- Definitions corresponding to the problem conditions
def total_jellybeans : ℕ := 20
def blue_jellybeans_start : ℕ := 10
def red_jellybeans : ℕ := 10

-- Probabilities calculation steps as definitions
def prob_first_blue : ℚ := blue_jellybeans_start / total_jellybeans
def prob_second_blue_given_first_blue : ℚ := (blue_jellybeans_start - 1) / (total_jellybeans - 1)
def prob_third_blue_given_first_two_blue : ℚ := (blue_jellybeans_start - 2) / (total_jellybeans - 2)

-- Total probability of drawing three blue jellybeans
def prob_three_blue : ℚ := 
  prob_first_blue *
  prob_second_blue_given_first_blue *
  prob_third_blue_given_first_two_blue

-- Formal statement of the proof problem
theorem prob_three_blue_is_correct : prob_three_blue = 2 / 19 :=
by
  -- Fill the proof here
  sorry

end NUMINAMATH_GPT_prob_three_blue_is_correct_l724_72411


namespace NUMINAMATH_GPT_no_integer_solutions_l724_72458

theorem no_integer_solutions (x y : ℤ) :
  ¬ (x^2 + 3 * x * y - 2 * y^2 = 122) :=
sorry

end NUMINAMATH_GPT_no_integer_solutions_l724_72458


namespace NUMINAMATH_GPT_joan_missed_games_l724_72452

theorem joan_missed_games :
  ∀ (total_games attended_games missed_games : ℕ),
  total_games = 864 →
  attended_games = 395 →
  missed_games = total_games - attended_games →
  missed_games = 469 :=
by
  intros total_games attended_games missed_games H1 H2 H3
  rw [H1, H2] at H3
  exact H3

end NUMINAMATH_GPT_joan_missed_games_l724_72452


namespace NUMINAMATH_GPT_quadratic_two_distinct_real_roots_l724_72446

theorem quadratic_two_distinct_real_roots (a : ℝ) :
    (∃ x1 x2 : ℝ, x1 ≠ x2 ∧ (a * x1^2 + 2 * x1 - 3 = 0) ∧ (a * x2^2 + 2 * x2 - 3 = 0)) ↔ a > -1 / 3 := by
  sorry

end NUMINAMATH_GPT_quadratic_two_distinct_real_roots_l724_72446


namespace NUMINAMATH_GPT_fish_population_l724_72461

theorem fish_population (x : ℕ) : 
  (1: ℝ) / 45 = (100: ℝ) / ↑x -> x = 1125 :=
by
  sorry

end NUMINAMATH_GPT_fish_population_l724_72461


namespace NUMINAMATH_GPT_poly_div_factor_l724_72477

theorem poly_div_factor (c : ℚ) : 2 * x + 7 ∣ 8 * x^4 + 27 * x^3 + 6 * x^2 + c * x - 49 ↔
  c = 47.25 :=
  sorry

end NUMINAMATH_GPT_poly_div_factor_l724_72477


namespace NUMINAMATH_GPT_watermelon_weight_l724_72462

theorem watermelon_weight (B W : ℝ) (n : ℝ) 
  (h1 : B + n * W = 63) 
  (h2 : B + (n / 2) * W = 34) : 
  n * W = 58 :=
sorry

end NUMINAMATH_GPT_watermelon_weight_l724_72462


namespace NUMINAMATH_GPT_find_c_for_two_solutions_in_real_l724_72400

noncomputable def system_two_solutions (x y c : ℝ) : Prop := (|x + y| = 2007 ∧ |x - y| = c)

theorem find_c_for_two_solutions_in_real : ∃ c : ℝ, (∀ x y : ℝ, system_two_solutions x y c) ↔ (c = 0) :=
by
  sorry

end NUMINAMATH_GPT_find_c_for_two_solutions_in_real_l724_72400


namespace NUMINAMATH_GPT_smallest_n_for_common_factor_l724_72407

theorem smallest_n_for_common_factor : ∃ n : ℕ, n > 0 ∧ (Nat.gcd (11 * n - 3) (8 * n + 4) > 1) ∧ n = 42 := 
by
  sorry

end NUMINAMATH_GPT_smallest_n_for_common_factor_l724_72407


namespace NUMINAMATH_GPT_sum_difference_even_odd_l724_72465

-- Define the sum of even integers from 2 to 100
def sum_even (n : ℕ) : ℕ := (n / 2) * (2 + n)

-- Define the sum of odd integers from 1 to 99
def sum_odd (n : ℕ) : ℕ := (n / 2) * (1 + n)

theorem sum_difference_even_odd:
  let a := sum_even 100
  let b := sum_odd 99
  a - b = 50 :=
by
  sorry

end NUMINAMATH_GPT_sum_difference_even_odd_l724_72465


namespace NUMINAMATH_GPT_extra_coverage_calculation_l724_72424

/-- Define the conditions -/
def bag_coverage : ℕ := 500
def lawn_length : ℕ := 35
def lawn_width : ℕ := 48
def number_of_bags : ℕ := 6

/-- Define the main theorem to prove -/
theorem extra_coverage_calculation :
  number_of_bags * bag_coverage - (lawn_length * lawn_width) = 1320 := 
by
  sorry

end NUMINAMATH_GPT_extra_coverage_calculation_l724_72424


namespace NUMINAMATH_GPT_max_value_of_y_over_x_l724_72497

theorem max_value_of_y_over_x
  (x y : ℝ)
  (h1 : x + y ≥ 3)
  (h2 : x - y ≥ -1)
  (h3 : 2 * x - y ≤ 3) :
  (∀ (x y : ℝ), (x + y ≥ 3) ∧ (x - y ≥ -1) ∧ (2 * x - y ≤ 3) → (∀ k, k = y / x → k ≤ 2)) :=
by
  sorry

end NUMINAMATH_GPT_max_value_of_y_over_x_l724_72497


namespace NUMINAMATH_GPT_vasya_claim_false_l724_72444

theorem vasya_claim_false :
  ∀ (weights : List ℕ), weights = [1, 2, 3, 4, 5, 6, 7] →
  (¬ ∃ (subset : List ℕ), subset.length = 3 ∧ 1 ∈ subset ∧
  ((weights.sum - subset.sum) = 14) ∧ (14 = 14)) :=
by
  sorry

end NUMINAMATH_GPT_vasya_claim_false_l724_72444


namespace NUMINAMATH_GPT_find_vector_at_t_zero_l724_72490

def vector_at_t (a d : ℝ × ℝ) (t : ℝ) : ℝ × ℝ :=
  (a.1 + t*d.1, a.2 + t*d.2)

theorem find_vector_at_t_zero :
  ∃ (a d : ℝ × ℝ),
    vector_at_t a d 1 = (2, 3) ∧
    vector_at_t a d 4 = (8, -5) ∧
    vector_at_t a d 5 = (10, -9) ∧
    vector_at_t a d 0 = (0, 17/3) :=
by
  sorry

end NUMINAMATH_GPT_find_vector_at_t_zero_l724_72490


namespace NUMINAMATH_GPT_p_adic_valuation_of_factorial_l724_72422

noncomputable def digit_sum (n p : ℕ) : ℕ :=
  -- Definition for sum of digits of n in base p
  sorry

def p_adic_valuation (n factorial : ℕ) (p : ℕ) : ℕ :=
  -- Representation of p-adic valuation of n!
  sorry

theorem p_adic_valuation_of_factorial (n p : ℕ) (hp: p > 1):
  p_adic_valuation n.factorial p = (n - digit_sum n p) / (p - 1) :=
sorry

end NUMINAMATH_GPT_p_adic_valuation_of_factorial_l724_72422


namespace NUMINAMATH_GPT_equal_share_of_tea_l724_72435

def totalCups : ℕ := 10
def totalPeople : ℕ := 5
def cupsPerPerson : ℕ := totalCups / totalPeople

theorem equal_share_of_tea : cupsPerPerson = 2 := by
  sorry

end NUMINAMATH_GPT_equal_share_of_tea_l724_72435


namespace NUMINAMATH_GPT_average_of_first_13_even_numbers_l724_72467

-- Definition of the first 13 even numbers
def first_13_even_numbers := [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26]

-- The sum of the first 13 even numbers
def sum_of_first_13_even_numbers : ℕ := 182

-- The number of these even numbers
def number_of_even_numbers : ℕ := 13

-- The average of the first 13 even numbers
theorem average_of_first_13_even_numbers : (sum_of_first_13_even_numbers / number_of_even_numbers) = 14 := by
  sorry

end NUMINAMATH_GPT_average_of_first_13_even_numbers_l724_72467


namespace NUMINAMATH_GPT_smallest_possible_sector_angle_l724_72427

theorem smallest_possible_sector_angle : ∃ a₁ d : ℕ, 2 * a₁ + 9 * d = 72 ∧ a₁ = 9 :=
by
  sorry

end NUMINAMATH_GPT_smallest_possible_sector_angle_l724_72427


namespace NUMINAMATH_GPT_undefined_expression_real_val_l724_72484

theorem undefined_expression_real_val (a : ℝ) :
  a = 2 → (a^3 - 8 = 0) :=
by
  intros
  sorry

end NUMINAMATH_GPT_undefined_expression_real_val_l724_72484


namespace NUMINAMATH_GPT_geometric_sequence_sum_l724_72443

theorem geometric_sequence_sum {a : ℕ → ℤ} (r : ℤ) (h1 : a 1 = 1) (h2 : r = -2) 
(h3 : ∀ n, a (n + 1) = a n * r) : 
  a 1 + |a 2| + |a 3| + a 4 = 15 := 
by sorry

end NUMINAMATH_GPT_geometric_sequence_sum_l724_72443


namespace NUMINAMATH_GPT_parts_of_diagonal_in_rectangle_l724_72425

/-- Proving that a 24x60 rectangle divided by its diagonal results in 1512 parts --/

theorem parts_of_diagonal_in_rectangle :
  let m := 24
  let n := 60
  let gcd_mn := gcd m n
  let unit_squares := m * n
  let diagonal_intersections := m + n - gcd_mn
  unit_squares + diagonal_intersections = 1512 :=
by
  sorry

end NUMINAMATH_GPT_parts_of_diagonal_in_rectangle_l724_72425


namespace NUMINAMATH_GPT_walnut_trees_l724_72432

theorem walnut_trees (logs_per_pine logs_per_maple logs_per_walnut pine_trees maple_trees total_logs walnut_trees : ℕ)
  (h1 : logs_per_pine = 80)
  (h2 : logs_per_maple = 60)
  (h3 : logs_per_walnut = 100)
  (h4 : pine_trees = 8)
  (h5 : maple_trees = 3)
  (h6 : total_logs = 1220)
  (h7 : total_logs = pine_trees * logs_per_pine + maple_trees * logs_per_maple + walnut_trees * logs_per_walnut) :
  walnut_trees = 4 :=
by
  sorry

end NUMINAMATH_GPT_walnut_trees_l724_72432


namespace NUMINAMATH_GPT_laborer_income_l724_72496

theorem laborer_income (I : ℕ) (debt : ℕ) 
  (h1 : 6 * I < 420) 
  (h2 : 4 * I = 240 + debt + 30) 
  (h3 : debt = 420 - 6 * I) : 
  I = 69 := by
  sorry

end NUMINAMATH_GPT_laborer_income_l724_72496


namespace NUMINAMATH_GPT_find_s_for_g_eq_0_l724_72434

def g (x s : ℝ) : ℝ := 3 * x^5 + 2 * x^4 - x^3 + 2 * x^2 - 5 * x + s

theorem find_s_for_g_eq_0 : ∃ (s : ℝ), g 3 s = 0 → s = -867 :=
by
  sorry

end NUMINAMATH_GPT_find_s_for_g_eq_0_l724_72434


namespace NUMINAMATH_GPT_decimal_to_fraction_correct_l724_72413

-- Define a structure representing our initial decimal to fraction conversion
structure DecimalFractionConversion :=
  (decimal: ℚ)
  (vulgar_fraction: ℚ)
  (simplified_fraction: ℚ)

-- Define the conditions provided in the problem
def conversion_conditions : DecimalFractionConversion :=
  { decimal := 35 / 100,
    vulgar_fraction := 35 / 100,
    simplified_fraction := 7 / 20 }

-- State the theorem we aim to prove
theorem decimal_to_fraction_correct :
  conversion_conditions.simplified_fraction = 7 / 20 := by
  sorry

end NUMINAMATH_GPT_decimal_to_fraction_correct_l724_72413


namespace NUMINAMATH_GPT_cos_beta_value_l724_72480

theorem cos_beta_value (α β : ℝ) (hα : 0 < α ∧ α < π / 2) (hβ : 0 < β ∧ β < π / 2)
  (hα_cos : Real.cos α = 4 / 5) (hαβ_cos : Real.cos (α + β) = -16 / 65) : 
  Real.cos β = 5 / 13 := 
sorry

end NUMINAMATH_GPT_cos_beta_value_l724_72480


namespace NUMINAMATH_GPT_red_paint_quarts_l724_72479

theorem red_paint_quarts (r g w : ℕ) (ratio_rw : r * 5 = w * 4) (w_quarts : w = 15) : r = 12 :=
by 
  -- We provide the skeleton of the proof here: the detailed steps are skipped (as instructed).
  sorry

end NUMINAMATH_GPT_red_paint_quarts_l724_72479


namespace NUMINAMATH_GPT_product_remainder_l724_72404

theorem product_remainder (a b c : ℕ) (h1 : a % 7 = 2) (h2 : b % 7 = 3) (h3 : c % 7 = 5) (h4 : (a + b + c) % 7 = 3) : 
  (a * b * c) % 7 = 2 := 
by sorry

end NUMINAMATH_GPT_product_remainder_l724_72404


namespace NUMINAMATH_GPT_distance_proof_l724_72423

/-- Maxwell's walking speed in km/h. -/
def Maxwell_speed := 4

/-- Time Maxwell walks before meeting Brad in hours. -/
def Maxwell_time := 10

/-- Brad's running speed in km/h. -/
def Brad_speed := 6

/-- Time Brad runs before meeting Maxwell in hours. -/
def Brad_time := 9

/-- Distance between Maxwell and Brad's homes in km. -/
def distance_between_homes : ℕ := 94

/-- Prove the distance between their homes is 94 km given the conditions. -/
theorem distance_proof 
  (h1 : Maxwell_speed * Maxwell_time = 40)
  (h2 : Brad_speed * Brad_time = 54) :
  Maxwell_speed * Maxwell_time + Brad_speed * Brad_time = distance_between_homes := 
by 
  sorry

end NUMINAMATH_GPT_distance_proof_l724_72423


namespace NUMINAMATH_GPT_dennis_took_away_l724_72464

-- Define the initial and remaining number of cards
def initial_cards : ℕ := 67
def remaining_cards : ℕ := 58

-- Define the number of cards taken away
def cards_taken_away (n m : ℕ) : ℕ := n - m

-- Prove that the number of cards taken away is 9
theorem dennis_took_away :
  cards_taken_away initial_cards remaining_cards = 9 :=
by
  -- Placeholder proof
  sorry

end NUMINAMATH_GPT_dennis_took_away_l724_72464


namespace NUMINAMATH_GPT_integer_side_lengths_triangle_l724_72457

theorem integer_side_lengths_triangle :
  ∃ (a b c : ℤ), (abc = 2 * (a - 1) * (b - 1) * (c - 1)) ∧
            (a = 8 ∧ b = 7 ∧ c = 3 ∨ a = 6 ∧ b = 5 ∧ c = 4) := 
by
  sorry

end NUMINAMATH_GPT_integer_side_lengths_triangle_l724_72457


namespace NUMINAMATH_GPT_harmonic_mean_pairs_count_l724_72406

theorem harmonic_mean_pairs_count :
  ∃ (s : Finset (ℕ × ℕ)), (∀ p ∈ s, p.1 < p.2 ∧ 2 * p.1 * p.2 = 4^15 * (p.1 + p.2)) ∧ s.card = 29 :=
sorry

end NUMINAMATH_GPT_harmonic_mean_pairs_count_l724_72406


namespace NUMINAMATH_GPT_gcd_m_n_l724_72419

def m := 122^2 + 234^2 + 346^2 + 458^2
def n := 121^2 + 233^2 + 345^2 + 457^2

theorem gcd_m_n : Int.gcd m n = 1 := 
by sorry

end NUMINAMATH_GPT_gcd_m_n_l724_72419


namespace NUMINAMATH_GPT_intersection_P_M_l724_72410

open Set Int

def P : Set ℤ := {x | 0 ≤ x ∧ x < 3}

def M : Set ℤ := {x | x^2 ≤ 9}

theorem intersection_P_M : P ∩ M = {0, 1, 2} := by
  sorry

end NUMINAMATH_GPT_intersection_P_M_l724_72410


namespace NUMINAMATH_GPT_triangle_ineq_l724_72481

theorem triangle_ineq
  (a b c : ℝ)
  (triangle_sides : 0 < a ∧ 0 < b ∧ 0 < c)
  (triangle_ineq : a + b + c = 1) :
  a^2 + b^2 + c^2 + 4 * a * b * c < 1 / 2 :=
by
  sorry

end NUMINAMATH_GPT_triangle_ineq_l724_72481


namespace NUMINAMATH_GPT_hypotenuse_length_right_triangle_l724_72485

theorem hypotenuse_length_right_triangle :
  ∃ (x : ℝ), (x > 7) ∧ ((x - 7)^2 + x^2 = (x + 2)^2) ∧ (x + 2 = 17) :=
by {
  sorry
}

end NUMINAMATH_GPT_hypotenuse_length_right_triangle_l724_72485


namespace NUMINAMATH_GPT_horner_value_v2_l724_72405

def poly (x : ℤ) : ℤ := 208 + 9 * x^2 + 6 * x^4 + x^6

theorem horner_value_v2 : poly (-4) = ((((0 + -4) * -4 + 6) * -4 + 9) * -4 + 208) :=
by
  sorry

end NUMINAMATH_GPT_horner_value_v2_l724_72405


namespace NUMINAMATH_GPT_Prudence_sleep_weeks_l724_72448

def Prudence_sleep_per_week : Nat := 
  let nights_sleep_weekday := 6
  let nights_sleep_weekend := 9
  let weekday_nights := 5
  let weekend_nights := 2
  let naps := 1
  let naps_days := 2
  weekday_nights * nights_sleep_weekday + weekend_nights * nights_sleep_weekend + naps_days * naps

theorem Prudence_sleep_weeks (w : Nat) (h : w * Prudence_sleep_per_week = 200) : w = 4 :=
by
  sorry

end NUMINAMATH_GPT_Prudence_sleep_weeks_l724_72448


namespace NUMINAMATH_GPT_bottles_per_case_l724_72491

theorem bottles_per_case (total_bottles : ℕ) (total_cases : ℕ) (h1 : total_bottles = 60000) (h2 : total_cases = 12000) :
  total_bottles / total_cases = 5 :=
by
  -- Using the given problem, so steps from the solution are not required here
  sorry

end NUMINAMATH_GPT_bottles_per_case_l724_72491


namespace NUMINAMATH_GPT_tile_count_l724_72473

theorem tile_count (a : ℕ) (h1 : ∃ b : ℕ, b = 2 * a) (h2 : 2 * (Int.floor (a * Real.sqrt 5)) - 1 = 49) :
  2 * a^2 = 50 :=
by
  sorry

end NUMINAMATH_GPT_tile_count_l724_72473


namespace NUMINAMATH_GPT_perfect_square_for_x_l724_72431

def expr (x : ℝ) : ℝ := 11.98 * 11.98 + 11.98 * x + 0.02 * 0.02

theorem perfect_square_for_x : expr 0.04 = (11.98 + 0.02) ^ 2 :=
by
  sorry

end NUMINAMATH_GPT_perfect_square_for_x_l724_72431


namespace NUMINAMATH_GPT_isosceles_triangle_perimeter_l724_72493

theorem isosceles_triangle_perimeter (a b c : ℝ) (h₀ : a = 5) (h₁ : b = 10) 
  (h₂ : c = 10 ∨ c = 5) (h₃ : a = b ∨ b = c ∨ c = a) 
  (triangle_inequality : a + b > c ∧ a + c > b ∧ b + c > a) :
  a + b + c = 25 := by
  sorry

end NUMINAMATH_GPT_isosceles_triangle_perimeter_l724_72493


namespace NUMINAMATH_GPT_f_plus_one_odd_l724_72474

noncomputable def f : ℝ → ℝ := sorry

theorem f_plus_one_odd (f : ℝ → ℝ)
  (h : ∀ x₁ x₂ : ℝ, f (x₁ + x₂) = f x₁ + f x₂ + 1) :
  ∀ x : ℝ, f x + 1 = -(f (-x) + 1) :=
sorry

end NUMINAMATH_GPT_f_plus_one_odd_l724_72474


namespace NUMINAMATH_GPT_equation_holds_except_two_values_l724_72483

noncomputable def check_equation (a y : ℝ) (h : a ≠ 0) : Prop :=
  (a / (a + y) + y / (a - y)) / (y / (a + y) - a / (a - y)) = -1 ↔ y ≠ a ∧ y ≠ -a

theorem equation_holds_except_two_values (a y: ℝ) (h: a ≠ 0): check_equation a y h := sorry

end NUMINAMATH_GPT_equation_holds_except_two_values_l724_72483


namespace NUMINAMATH_GPT_imaginary_part_of_product_l724_72460

def imaginary_unit : ℂ := Complex.I

def z : ℂ := 2 + imaginary_unit

theorem imaginary_part_of_product : (z * imaginary_unit).im = 2 := by
  sorry

end NUMINAMATH_GPT_imaginary_part_of_product_l724_72460


namespace NUMINAMATH_GPT_probability_stopping_in_C_l724_72494

noncomputable def probability_C : ℚ :=
  let P_A := 1 / 5
  let P_B := 1 / 5
  let x := (1 - (P_A + P_B)) / 3
  x

theorem probability_stopping_in_C :
  probability_C = 1 / 5 :=
by
  unfold probability_C
  sorry

end NUMINAMATH_GPT_probability_stopping_in_C_l724_72494


namespace NUMINAMATH_GPT_sum_of_fractions_l724_72436

theorem sum_of_fractions :
  (1/15 + 2/15 + 3/15 + 4/15 + 5/15 + 6/15 + 7/15 + 8/15 + 9/15 + 46/15) = (91/15) := by
  sorry

end NUMINAMATH_GPT_sum_of_fractions_l724_72436


namespace NUMINAMATH_GPT_basketball_children_l724_72440

/-- Given:
  1. total spectators is 10,000
  2. 7,000 of them were men
  3. Of the remaining spectators, there were 5 times as many children as women

Prove that the number of children was 2,500. -/
theorem basketball_children (total_spectators : ℕ) (men : ℕ) (women_children : ℕ) (women children : ℕ) 
  (h1 : total_spectators = 10000) 
  (h2 : men = 7000) 
  (h3 : women_children = total_spectators - men) 
  (h4 : women + 5 * women = women_children) 
  : children = 5 * 500 := 
  by 
  sorry

end NUMINAMATH_GPT_basketball_children_l724_72440


namespace NUMINAMATH_GPT_phantom_needs_more_money_l724_72470

def amount_phantom_has : ℤ := 50
def cost_black : ℤ := 11
def count_black : ℕ := 2
def cost_red : ℤ := 15
def count_red : ℕ := 3
def cost_yellow : ℤ := 13
def count_yellow : ℕ := 2

def total_cost : ℤ := cost_black * count_black + cost_red * count_red + cost_yellow * count_yellow
def additional_amount_needed : ℤ := total_cost - amount_phantom_has

theorem phantom_needs_more_money : additional_amount_needed = 43 := by
  sorry

end NUMINAMATH_GPT_phantom_needs_more_money_l724_72470


namespace NUMINAMATH_GPT_find_a_l724_72455

theorem find_a (a : ℝ) : 
  let A := {1, 2, 3}
  let B := {x : ℝ | x^2 - (a + 1) * x + a = 0}
  A ∪ B = A → a = 1 ∨ a = 2 ∨ a = 3 :=
by
  intros
  sorry

end NUMINAMATH_GPT_find_a_l724_72455


namespace NUMINAMATH_GPT_quadrilateral_possible_rods_l724_72437

theorem quadrilateral_possible_rods (rods : Finset ℕ) (a b c : ℕ) (ha : a = 3) (hb : b = 7) (hc : c = 15)
  (hrods : rods = (Finset.range 31 \ {3, 7, 15})) :
  ∃ d, d ∈ rods ∧ 5 < d ∧ d < 25 ∧ rods.card - 2 = 17 := 
by
  sorry

end NUMINAMATH_GPT_quadrilateral_possible_rods_l724_72437


namespace NUMINAMATH_GPT_system_has_integer_solution_l724_72475

theorem system_has_integer_solution (a b : ℤ) : 
  ∃ x y z t : ℤ, x + y + 2 * z + 2 * t = a ∧ 2 * x - 2 * y + z - t = b :=
by
  sorry

end NUMINAMATH_GPT_system_has_integer_solution_l724_72475


namespace NUMINAMATH_GPT_termites_ate_black_squares_l724_72415

def chessboard_black_squares_eaten : Nat :=
  12

theorem termites_ate_black_squares :
  let rows := 8;
  let cols := 8;
  let total_squares := rows * cols / 2; -- This simplistically assumes half the squares are black.
  (total_squares = 32) → 
  chessboard_black_squares_eaten = 12 :=
by
  intros h
  sorry

end NUMINAMATH_GPT_termites_ate_black_squares_l724_72415


namespace NUMINAMATH_GPT_no_real_solutions_sufficient_not_necessary_l724_72466

theorem no_real_solutions_sufficient_not_necessary (m : ℝ) : 
  (|m| < 1) → (m^2 < 4) :=
by
  sorry

end NUMINAMATH_GPT_no_real_solutions_sufficient_not_necessary_l724_72466


namespace NUMINAMATH_GPT_tina_first_hour_coins_l724_72416

variable (X : ℕ)

theorem tina_first_hour_coins :
  let first_hour_coins := X
  let second_third_hour_coins := 30 + 30
  let fourth_hour_coins := 40
  let fifth_hour_removed_coins := 20
  let total_coins := first_hour_coins + second_third_hour_coins + fourth_hour_coins - fifth_hour_removed_coins
  total_coins = 100 → X = 20 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_tina_first_hour_coins_l724_72416


namespace NUMINAMATH_GPT_incorrect_conclusion_l724_72487

def y (x : ℝ) : ℝ := -2 * x + 3

theorem incorrect_conclusion : ∀ (x : ℝ), y x = 0 → x ≠ 0 := 
by
  sorry

end NUMINAMATH_GPT_incorrect_conclusion_l724_72487


namespace NUMINAMATH_GPT_find_f_of_3_l724_72420

theorem find_f_of_3 (f : ℝ → ℝ) (h1 : ∀ x y : ℝ, f (x * f y - y) = x * y - f y) 
  (h2 : f 0 = 0) (h3 : ∀ x : ℝ, f (-x) = -f x) : f 3 = 3 :=
sorry

end NUMINAMATH_GPT_find_f_of_3_l724_72420


namespace NUMINAMATH_GPT_area_of_plot_area_in_terms_of_P_l724_72438

-- Conditions and definitions.
variables (P : ℝ) (l w : ℝ)
noncomputable def perimeter := 2 * (l + w)
axiom h_perimeter : perimeter l w = 120
axiom h_equality : l = 2 * w

-- Proofs statements
theorem area_of_plot : l + w = 60 → l = 2 * w → (4 * w)^2 = 6400 := by
  sorry

theorem area_in_terms_of_P : (4 * (P / 6))^2 = (2 * P / 3)^2 → (2 * P / 3)^2 = 4 * P^2 / 9 := by
  sorry

end NUMINAMATH_GPT_area_of_plot_area_in_terms_of_P_l724_72438


namespace NUMINAMATH_GPT_logarithmic_expression_range_l724_72471

theorem logarithmic_expression_range (a : ℝ) : 
  (a - 2 > 0) ∧ (5 - a > 0) ∧ (a - 2 ≠ 1) ↔ (2 < a ∧ a < 3) ∨ (3 < a ∧ a < 5) := 
by
  sorry

end NUMINAMATH_GPT_logarithmic_expression_range_l724_72471


namespace NUMINAMATH_GPT_convex_parallelogram_faces_1992_l724_72414

theorem convex_parallelogram_faces_1992 (n : ℕ) (h : n > 0) : (n * (n - 1) ≠ 1992) := 
by
  sorry

end NUMINAMATH_GPT_convex_parallelogram_faces_1992_l724_72414


namespace NUMINAMATH_GPT_geometric_progression_terms_l724_72447

theorem geometric_progression_terms (a b r : ℝ) (n : ℕ) (h1 : 0 < r) (h2: a ≠ 0) (h3 : b = a * r^(n-1)) :
  n = 1 + (Real.log (b / a)) / (Real.log r) :=
by sorry

end NUMINAMATH_GPT_geometric_progression_terms_l724_72447


namespace NUMINAMATH_GPT_percentage_students_on_trip_l724_72421

variable (total_students : ℕ)
variable (students_more_than_100 : ℕ)
variable (students_on_trip : ℕ)
variable (percentage_more_than_100 : ℝ)
variable (percentage_not_more_than_100 : ℝ)

-- Given conditions
def condition_1 := percentage_more_than_100 = 0.16
def condition_2 := percentage_not_more_than_100 = 0.75

-- The final proof statement
theorem percentage_students_on_trip :
  percentage_more_than_100 * (total_students : ℝ) /
  ((1 - percentage_not_more_than_100)) / (total_students : ℝ) * 100 = 64 :=
by
  sorry

end NUMINAMATH_GPT_percentage_students_on_trip_l724_72421


namespace NUMINAMATH_GPT_area_of_square_on_AD_l724_72417

theorem area_of_square_on_AD :
  ∃ (AB BC CD AD : ℝ),
    (∃ AB_sq BC_sq CD_sq AD_sq : ℝ,
      AB_sq = 25 ∧ BC_sq = 49 ∧ CD_sq = 64 ∧ 
      AB = Real.sqrt AB_sq ∧ BC = Real.sqrt BC_sq ∧ CD = Real.sqrt CD_sq ∧
      AD_sq = AB^2 + BC^2 + CD^2 ∧ AD = Real.sqrt AD_sq ∧ AD_sq = 138
    ) :=
by
  sorry

end NUMINAMATH_GPT_area_of_square_on_AD_l724_72417


namespace NUMINAMATH_GPT_angle_in_second_quadrant_l724_72433

/-- If α is an angle in the first quadrant, then π - α is an angle in the second quadrant -/
theorem angle_in_second_quadrant (α : Real) (h : 0 < α ∧ α < π / 2) : π - α > π / 2 ∧ π - α < π :=
by
  sorry

end NUMINAMATH_GPT_angle_in_second_quadrant_l724_72433


namespace NUMINAMATH_GPT_calculate_expression_l724_72499

theorem calculate_expression :
  2 * (-1 / 4) - |1 - Real.sqrt 3| + (-2023)^0 = 3 / 2 - Real.sqrt 3 :=
by
  sorry

end NUMINAMATH_GPT_calculate_expression_l724_72499


namespace NUMINAMATH_GPT_find_number_l724_72426

theorem find_number (x : ℝ) (h : x / 5 + 10 = 21) : x = 55 :=
by
  sorry

end NUMINAMATH_GPT_find_number_l724_72426


namespace NUMINAMATH_GPT_range_of_a_l724_72451

variable (a : ℝ)

def p : Prop := ∀ (x : ℝ), 1 ≤ x ∧ x ≤ 2 → x^2 - a ≥ 0
def q : Prop := ∃ (x : ℝ), x^2 + 2 * a * x + 2 - a = 0

theorem range_of_a (h : p a ∧ q a) : a ∈ Set.Iic (-2) ∪ {1} := by
  sorry

end NUMINAMATH_GPT_range_of_a_l724_72451


namespace NUMINAMATH_GPT_intersection_of_A_and_B_l724_72459

def set_A : Set ℝ := {x | x^2 ≤ 4 * x}
def set_B : Set ℝ := {x | |x| ≥ 2}

theorem intersection_of_A_and_B :
  {x | x ∈ set_A ∧ x ∈ set_B} = {x | 2 ≤ x ∧ x ≤ 4} :=
sorry

end NUMINAMATH_GPT_intersection_of_A_and_B_l724_72459


namespace NUMINAMATH_GPT_inequality_proof_l724_72418

theorem inequality_proof (α : ℝ) (h1 : 0 < α) (h2 : α < Real.pi / 2) : 
  2 * Real.sin α + Real.tan α > 3 * α := 
by
  sorry

end NUMINAMATH_GPT_inequality_proof_l724_72418


namespace NUMINAMATH_GPT_ladder_base_l724_72441

theorem ladder_base (h : ℝ) (b : ℝ) (l : ℝ)
  (h_eq : h = 12) (l_eq : l = 15) : b = 9 :=
by
  have hypotenuse := l
  have height := h
  have base := b
  have pythagorean_theorem : height^2 + base^2 = hypotenuse^2 := by sorry 
  sorry

end NUMINAMATH_GPT_ladder_base_l724_72441


namespace NUMINAMATH_GPT_g_range_l724_72401

variable {R : Type*} [LinearOrderedRing R]

-- Let y = f(x) be a function defined on R with a period of 1
def periodic (f : R → R) : Prop :=
  ∀ x, f (x + 1) = f x

-- If g(x) = f(x) + 2x
def g (f : R → R) (x : R) : R := f x + 2 * x

-- If the range of g(x) on the interval [1,2] is [-1,5]
def rangeCondition (f : R → R) : Prop :=
  ∀ x, 1 ≤ x ∧ x ≤ 2 → -1 ≤ g f x ∧ g f x ≤ 5

-- Then the range of the function g(x) on the interval [-2020,2020] is [-4043,4041]
theorem g_range (f : R → R) 
  (hf_periodic : periodic f) 
  (hf_range : rangeCondition f) : 
  ∀ x, -2020 ≤ x ∧ x ≤ 2020 → -4043 ≤ g f x ∧ g f x ≤ 4041 :=
sorry

end NUMINAMATH_GPT_g_range_l724_72401


namespace NUMINAMATH_GPT_geometric_sequence_log_sum_l724_72489

noncomputable def log_base_three (x : ℝ) : ℝ := Real.log x / Real.log 3

theorem geometric_sequence_log_sum (a : ℕ → ℝ) (h1 : ∀ n, a n > 0)
  (h2 : ∃ r, ∀ n, a (n + 1) = a n * r)
  (h3 : a 6 * a 7 = 9) :
  log_base_three (a 1) + log_base_three (a 2) + log_base_three (a 3) +
  log_base_three (a 4) + log_base_three (a 5) + log_base_three (a 6) +
  log_base_three (a 7) + log_base_three (a 8) + log_base_three (a 9) +
  log_base_three (a 10) + log_base_three (a 11) + log_base_three (a 12) = 12 :=
  sorry

end NUMINAMATH_GPT_geometric_sequence_log_sum_l724_72489


namespace NUMINAMATH_GPT_find_a9_l724_72482

variable (S : ℕ → ℤ) (a : ℕ → ℤ)
variable (d a1 : ℤ)

def arithmetic_seq (n : ℕ) : ℤ :=
  a1 + ↑n * d

def sum_arithmetic_seq (n : ℕ) : ℤ :=
  n * a1 + (n * (n - 1) / 2) * d

axiom h1 : sum_arithmetic_seq 8 = 4 * arithmetic_seq 3
axiom h2 : arithmetic_seq 7 = -2

theorem find_a9 : arithmetic_seq 9 = -6 :=
by
  sorry

end NUMINAMATH_GPT_find_a9_l724_72482


namespace NUMINAMATH_GPT_y_works_in_40_days_l724_72402

theorem y_works_in_40_days :
  ∃ d, (d > 0) ∧ 
  (1/20 + 1/d = 3/40) ∧ 
  d = 40 :=
by
  use 40
  sorry

end NUMINAMATH_GPT_y_works_in_40_days_l724_72402


namespace NUMINAMATH_GPT_triangles_may_or_may_not_be_congruent_triangles_may_have_equal_areas_l724_72412

-- Given the conditions: two sides of one triangle are equal to two sides of another triangle.
-- And an angle opposite to one of these sides is equal to the angle opposite to the corresponding side.
variables {A B C D E F : Type}
variables {AB DE BC EF : ℝ} (h_AB_DE : AB = DE) (h_BC_EF : BC = EF)
variables {angle_A angle_D : ℝ} (h_angle_A_D : angle_A = angle_D)

-- Prove that the triangles may or may not be congruent
theorem triangles_may_or_may_not_be_congruent :
  ∃ (AB DE BC EF : ℝ) (angle_A angle_D : ℝ), AB = DE → BC = EF → angle_A = angle_D →
  (triangle_may_be_congruent_or_not : Prop) :=
sorry

-- Prove that the triangles may have equal areas
theorem triangles_may_have_equal_areas :
  ∃ (AB DE BC EF : ℝ) (angle_A angle_D : ℝ), AB = DE → BC = EF → angle_A = angle_D →
  (triangle_may_have_equal_areas : Prop) :=
sorry

end NUMINAMATH_GPT_triangles_may_or_may_not_be_congruent_triangles_may_have_equal_areas_l724_72412


namespace NUMINAMATH_GPT_solutions_of_system_l724_72403

theorem solutions_of_system (x y z : ℝ) :
    (x^2 - y = z^2) → (y^2 - z = x^2) → (z^2 - x = y^2) →
    (x = 0 ∧ y = 0 ∧ z = 0) ∨ 
    (x = 1 ∧ y = 0 ∧ z = -1) ∨ 
    (x = 0 ∧ y = -1 ∧ z = 1) ∨ 
    (x = -1 ∧ y = 1 ∧ z = 0) := by
  sorry

end NUMINAMATH_GPT_solutions_of_system_l724_72403


namespace NUMINAMATH_GPT_sum_of_fourth_powers_correct_l724_72450

noncomputable def sum_of_fourth_powers (x : ℤ) : ℤ :=
  x^4 + (x+1)^4 + (x+2)^4

theorem sum_of_fourth_powers_correct (x : ℤ) (h : x * (x+1) * (x+2) = 36 * x + 12) : 
  sum_of_fourth_powers x = 98 :=
sorry

end NUMINAMATH_GPT_sum_of_fourth_powers_correct_l724_72450


namespace NUMINAMATH_GPT_final_selling_price_l724_72428

-- Conditions
variable (x : ℝ)
def original_price : ℝ := x
def first_discount : ℝ := 0.8 * x
def additional_reduction : ℝ := 10

-- Statement of the problem
theorem final_selling_price (x : ℝ) : (0.8 * x) - 10 = 0.8 * x - 10 :=
by sorry

end NUMINAMATH_GPT_final_selling_price_l724_72428


namespace NUMINAMATH_GPT_total_fencing_cost_l724_72456

-- Conditions
def length : ℝ := 55
def cost_per_meter : ℝ := 26.50

-- We derive breadth from the given conditions
def breadth : ℝ := length - 10

-- Calculate the perimeter of the rectangular plot
def perimeter : ℝ := 2 * (length + breadth)

-- Calculate the total cost of fencing the plot
def total_cost : ℝ := cost_per_meter * perimeter

-- The theorem to prove that total cost is equal to 5300
theorem total_fencing_cost : total_cost = 5300 := by
  -- Calculation goes here
  sorry

end NUMINAMATH_GPT_total_fencing_cost_l724_72456


namespace NUMINAMATH_GPT_percentage_increase_area_l724_72488

theorem percentage_increase_area (L W : ℝ) (hL : 0 < L) (hW : 0 < W) :
  let A := L * W
  let A' := (1.35 * L) * (1.35 * W)
  let percentage_increase := ((A' - A) / A) * 100
  percentage_increase = 82.25 :=
by
  sorry

end NUMINAMATH_GPT_percentage_increase_area_l724_72488


namespace NUMINAMATH_GPT_find_AB_l724_72486

variables {AB CD AD BC AP PD APD PQ Q: ℝ}

def is_rectangle (ABCD : Prop) := ABCD

variables (P_on_BC : Prop)
variable (BP CP: ℝ)
variable (tan_angle_APD: ℝ)

theorem find_AB (ABCD : Prop) (P_on_BC : Prop) (BP CP: ℝ) (tan_angle_APD: ℝ) : 
  is_rectangle ABCD →
  P_on_BC →
  BP = 24 →
  CP = 12 →
  tan_angle_APD = 2 →
  AB = 27 := 
by
  sorry

end NUMINAMATH_GPT_find_AB_l724_72486


namespace NUMINAMATH_GPT_area_of_region_AGF_l724_72469

theorem area_of_region_AGF 
  (ABCD_area : ℝ)
  (hABCD_area : ABCD_area = 160)
  (E F G : ℝ)
  (hE_midpoint : E = (A + B) / 2)
  (hF_midpoint : F = (C + D) / 2)
  (EF_divides : EF_area = ABCD_area / 2)
  (hEF_midpoint : G = (E + F) / 2)
  (AG_divides_upper : AG_area = EF_area / 2) :
  AGF_area = 40 := 
sorry

end NUMINAMATH_GPT_area_of_region_AGF_l724_72469
