import Mathlib

namespace NUMINAMATH_GPT_series_sum_correct_l2416_241662

noncomputable def geometric_series_sum (a r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r ^ n) / (1 - r)

theorem series_sum_correct :
  geometric_series_sum (1 / 2) (-1 / 3) 6 = 91 / 243 :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_GPT_series_sum_correct_l2416_241662


namespace NUMINAMATH_GPT_no_positive_integer_n_ge_2_1001_n_is_square_of_prime_l2416_241651

noncomputable def is_square_of_prime (m : ℕ) : Prop :=
  ∃ p : ℕ, Prime p ∧ m = p * p

theorem no_positive_integer_n_ge_2_1001_n_is_square_of_prime :
  ∀ n : ℕ, n ≥ 2 → ¬ is_square_of_prime (n^3 + 1) :=
by
  intro n hn
  sorry

end NUMINAMATH_GPT_no_positive_integer_n_ge_2_1001_n_is_square_of_prime_l2416_241651


namespace NUMINAMATH_GPT_k_equals_three_fourths_l2416_241667

theorem k_equals_three_fourths : ∀ a b c d : ℝ, a ∈ Set.Ici (-1) → b ∈ Set.Ici (-1) → c ∈ Set.Ici (-1) → d ∈ Set.Ici (-1) →
  a^3 + b^3 + c^3 + d^3 + 1 ≥ (3 / 4) * (a + b + c + d) :=
by
  intros
  sorry

end NUMINAMATH_GPT_k_equals_three_fourths_l2416_241667


namespace NUMINAMATH_GPT_bees_on_second_day_l2416_241654

-- Define the number of bees on the first day
def bees_first_day : ℕ := 144

-- Define the multiplication factor
def multiplication_factor : ℕ := 3

-- Define the number of bees on the second day
def bees_second_day : ℕ := bees_first_day * multiplication_factor

-- Theorem stating the number of bees on the second day is 432
theorem bees_on_second_day : bees_second_day = 432 := 
by
  sorry

end NUMINAMATH_GPT_bees_on_second_day_l2416_241654


namespace NUMINAMATH_GPT_sum_of_coordinates_of_reflected_points_l2416_241621

theorem sum_of_coordinates_of_reflected_points (C D : ℝ × ℝ) (hx : C.1 = 3) (hy : C.2 = 8) (hD : D = (-C.1, C.2)) :
  C.1 + C.2 + D.1 + D.2 = 16 := by
  sorry

end NUMINAMATH_GPT_sum_of_coordinates_of_reflected_points_l2416_241621


namespace NUMINAMATH_GPT_taylor_family_reunion_adults_l2416_241608

def number_of_kids : ℕ := 45
def number_of_tables : ℕ := 14
def people_per_table : ℕ := 12
def total_people := number_of_tables * people_per_table

theorem taylor_family_reunion_adults : total_people - number_of_kids = 123 := by
  sorry

end NUMINAMATH_GPT_taylor_family_reunion_adults_l2416_241608


namespace NUMINAMATH_GPT_find_x_l2416_241646

theorem find_x 
  (x : ℝ)
  (h : 3.5 * ((3.6 * 0.48 * 2.50) / (0.12 * x * 0.5)) = 2800.0000000000005) : 
  x = 0.225 := 
sorry

end NUMINAMATH_GPT_find_x_l2416_241646


namespace NUMINAMATH_GPT_square_of_number_l2416_241659

theorem square_of_number (x : ℝ) (h : 2 * x = x / 5 + 9) : x^2 = 25 := 
sorry

end NUMINAMATH_GPT_square_of_number_l2416_241659


namespace NUMINAMATH_GPT_negate_exists_implies_forall_l2416_241634

-- Define the original proposition
def prop1 (x : ℝ) : Prop := x^2 + 2 * x + 2 < 0

-- The negation of the proposition
def neg_prop1 := ∀ x : ℝ, x^2 + 2 * x + 2 ≥ 0

-- Statement of the equivalence
theorem negate_exists_implies_forall :
  ¬(∃ x : ℝ, prop1 x) ↔ neg_prop1 := by
  sorry

end NUMINAMATH_GPT_negate_exists_implies_forall_l2416_241634


namespace NUMINAMATH_GPT_terms_before_one_l2416_241693

-- Define the sequence parameters
def a : ℤ := 100
def d : ℤ := -7
def nth_term (n : ℕ) : ℤ := a + (n - 1) * d

-- Define the target term we are interested in
def target_term : ℤ := 1

-- Define the main theorem
theorem terms_before_one : ∃ n : ℕ, nth_term n = target_term ∧ (n - 1) = 14 := by
  sorry

end NUMINAMATH_GPT_terms_before_one_l2416_241693


namespace NUMINAMATH_GPT_sara_basketball_loss_l2416_241691

theorem sara_basketball_loss (total_games : ℕ) (games_won : ℕ) (games_lost : ℕ) 
  (h1 : total_games = 16) 
  (h2 : games_won = 12) 
  (h3 : games_lost = total_games - games_won) : 
  games_lost = 4 :=
by
  sorry

end NUMINAMATH_GPT_sara_basketball_loss_l2416_241691


namespace NUMINAMATH_GPT_largest_possible_sum_l2416_241642

theorem largest_possible_sum (a b : ℤ) (h : a^2 - b^2 = 144) : a + b ≤ 72 :=
sorry

end NUMINAMATH_GPT_largest_possible_sum_l2416_241642


namespace NUMINAMATH_GPT_problem1_problem2_l2416_241676

noncomputable def f (x : ℝ) : ℝ :=
  if h : 1 ≤ x then x else 1 / x

noncomputable def g (x : ℝ) (a : ℝ) : ℝ :=
  a * f x - |x - 2|

def problem1_statement (b : ℝ) : Prop :=
  ∀ x, x > 0 → g x 0 ≤ |x - 1| + b

def problem2_statement : Prop :=
  ∃ x, (0 < x) ∧ ∀ y, (0 < y) → g y 1 ≥ g x 1

theorem problem1 : ∀ b : ℝ, problem1_statement b ↔ b ∈ Set.Ici (-1) := sorry

theorem problem2 : ∃ x, problem2_statement ∧ g x 1 = 0 := sorry

end NUMINAMATH_GPT_problem1_problem2_l2416_241676


namespace NUMINAMATH_GPT_find_positive_number_l2416_241600

theorem find_positive_number (x : ℝ) (hx : 0 < x) (h : (2 / 3) * x = (144 / 216) * (1 / x)) : x = 1 := by
  sorry

end NUMINAMATH_GPT_find_positive_number_l2416_241600


namespace NUMINAMATH_GPT_telephone_charges_equal_l2416_241637

theorem telephone_charges_equal (m : ℝ) :
  (9 + 0.25 * m = 12 + 0.20 * m) → m = 60 :=
by
  intro h
  sorry

end NUMINAMATH_GPT_telephone_charges_equal_l2416_241637


namespace NUMINAMATH_GPT_problem_1_problem_2_l2416_241692

noncomputable def f (x a : ℝ) : ℝ := abs x + 2 * abs (x - a)

theorem problem_1 (x : ℝ) : (f x 1 ≤ 4) ↔ (- 2 / 3 ≤ x ∧ x ≤ 2) := 
sorry

theorem problem_2 (a : ℝ) : (∀ x : ℝ, f x a ≥ 4) ↔ (4 ≤ a) := 
sorry

end NUMINAMATH_GPT_problem_1_problem_2_l2416_241692


namespace NUMINAMATH_GPT_value_of_x_plus_y_l2416_241645

theorem value_of_x_plus_y (x y : ℝ) 
  (h1 : 2 * x - y = -1) 
  (h2 : x + 4 * y = 22) : 
  x + y = 7 :=
sorry

end NUMINAMATH_GPT_value_of_x_plus_y_l2416_241645


namespace NUMINAMATH_GPT_circle_area_l2416_241605

theorem circle_area (r : ℝ) (h : 8 * (1 / (2 * π * r)) = (2 * r) ^ 2) : π * r ^ 2 = π ^ (1 / 3) :=
by
  sorry

end NUMINAMATH_GPT_circle_area_l2416_241605


namespace NUMINAMATH_GPT_area_of_square_l2416_241684

theorem area_of_square (a : ℝ) (h : a = 12) : a * a = 144 := by
  rw [h]
  norm_num

end NUMINAMATH_GPT_area_of_square_l2416_241684


namespace NUMINAMATH_GPT_remainder_349_div_13_l2416_241609

theorem remainder_349_div_13 : 349 % 13 = 11 := 
by 
  sorry

end NUMINAMATH_GPT_remainder_349_div_13_l2416_241609


namespace NUMINAMATH_GPT_bouquet_cost_l2416_241613

theorem bouquet_cost (c₁ : ℕ) (r₁ r₂ : ℕ) (c_discount : ℕ) (discount_percentage: ℕ) :
  (c₁ = 30) → (r₁ = 15) → (r₂ = 45) → (c_discount = 81) → (discount_percentage = 10) → 
  ((c₂ : ℕ) → (c₂ = (c₁ * r₂) / r₁) → (r₂ > 30) → 
  (c_discount = c₂ - (c₂ * discount_percentage / 100))) → 
  c_discount = 81 :=
by
  intros h1 h2 h3 h4 h5
  subst_vars
  sorry

end NUMINAMATH_GPT_bouquet_cost_l2416_241613


namespace NUMINAMATH_GPT_problem_1_problem_2_problem_3_l2416_241627

noncomputable def f (x : ℝ) (k : ℝ) : ℝ := 8 * x^2 + 16 * x - k
noncomputable def g (x : ℝ) : ℝ := 2 * x^3 + 5 * x^2 + 4 * x
noncomputable def h (x : ℝ) (k : ℝ) : ℝ := g x - f x k

theorem problem_1 (k : ℝ) : (∀ x : ℝ, -3 ≤ x ∧ x ≤ 3 → f x k ≤ g x) → 45 ≤ k := by
  sorry

theorem problem_2 (k : ℝ) : (∃ x : ℝ, -3 ≤ x ∧ x ≤ 3 ∧ f x k ≤ g x) → -7 ≤ k := by
  sorry

theorem problem_3 (k : ℝ) : (∀ x1 x2 : ℝ, (-3 ≤ x1 ∧ x1 ≤ 3) ∧ (-3 ≤ x2 ∧ x2 ≤ 3) → f x1 k ≤ g x2) → 141 ≤ k := by
  sorry

end NUMINAMATH_GPT_problem_1_problem_2_problem_3_l2416_241627


namespace NUMINAMATH_GPT_trader_sold_80_meters_l2416_241687

variable (x : ℕ)
variable (selling_price_per_meter profit_per_meter cost_price_per_meter total_selling_price : ℕ)

theorem trader_sold_80_meters
  (h_cost_price : cost_price_per_meter = 118)
  (h_profit : profit_per_meter = 7)
  (h_selling_price : selling_price_per_meter = cost_price_per_meter + profit_per_meter)
  (h_total_selling_price : total_selling_price = 10000)
  (h_eq : selling_price_per_meter * x = total_selling_price) :
  x = 80 := by
    sorry

end NUMINAMATH_GPT_trader_sold_80_meters_l2416_241687


namespace NUMINAMATH_GPT_greatest_distance_between_vertices_l2416_241628

theorem greatest_distance_between_vertices 
    (inner_perimeter outer_perimeter : ℝ) 
    (inner_square_perimeter_eq : inner_perimeter = 16)
    (outer_square_perimeter_eq : outer_perimeter = 40)
    : ∃ max_distance, max_distance = 2 * Real.sqrt 34 :=
by
  sorry

end NUMINAMATH_GPT_greatest_distance_between_vertices_l2416_241628


namespace NUMINAMATH_GPT_no_nonneg_rational_sol_for_equation_l2416_241681

theorem no_nonneg_rational_sol_for_equation :
  ¬ ∃ (x y z : ℚ), 0 ≤ x ∧ 0 ≤ y ∧ 0 ≤ z ∧ x^5 + 2 * y^5 + 5 * z^5 = 11 :=
by
  sorry

end NUMINAMATH_GPT_no_nonneg_rational_sol_for_equation_l2416_241681


namespace NUMINAMATH_GPT_smallest_area_right_triangle_l2416_241680

-- We define the two sides of the triangle
def side1 : ℕ := 6
def side2 : ℕ := 8

-- Define the area calculation for a right triangle
def area (a b : ℕ) : ℕ := (a * b) / 2

-- The theorem to prove the smallest area is 24 square units
theorem smallest_area_right_triangle : ∃ (c : ℕ), side1 * side1 + side2 * side2 = c * c ∧ area side1 side2 = 24 :=
by
  sorry

end NUMINAMATH_GPT_smallest_area_right_triangle_l2416_241680


namespace NUMINAMATH_GPT_problem_a5_value_l2416_241629

def Sn (n : ℕ) : ℕ := 2 * n^2 + 3 * n - 1

theorem problem_a5_value : Sn 5 - Sn 4 = 21 := by
  sorry

end NUMINAMATH_GPT_problem_a5_value_l2416_241629


namespace NUMINAMATH_GPT_log_expression_evaluation_l2416_241624

theorem log_expression_evaluation : 
  (4 * Real.log 2 + 3 * Real.log 5 - Real.log (1/5)) = 4 := 
  sorry

end NUMINAMATH_GPT_log_expression_evaluation_l2416_241624


namespace NUMINAMATH_GPT_exists_same_color_points_distance_one_l2416_241612

theorem exists_same_color_points_distance_one
    (color : ℝ × ℝ → Fin 3)
    (h : ∀ p q : ℝ × ℝ, dist p q = 1 → color p ≠ color q) :
  ∃ p q : ℝ × ℝ, dist p q = 1 ∧ color p = color q :=
sorry

end NUMINAMATH_GPT_exists_same_color_points_distance_one_l2416_241612


namespace NUMINAMATH_GPT_range_of_f_lt_zero_l2416_241614

noncomputable
def f : ℝ → ℝ := sorry

theorem range_of_f_lt_zero 
  (hf_even : ∀ x, f x = f (-x))
  (hf_decreasing : ∀ x y, x < y ∧ y ≤ 0 → f x > f y)
  (hf_at_neg2_zero : f (-2) = 0) :
  {x : ℝ | f x < 0} = {x : ℝ | -2 < x ∧ x < 2} :=
by
  sorry

end NUMINAMATH_GPT_range_of_f_lt_zero_l2416_241614


namespace NUMINAMATH_GPT_largest_integer_n_l2416_241697

theorem largest_integer_n (n : ℤ) (h : n^2 - 13 * n + 40 < 0) : n = 7 :=
by
  sorry

end NUMINAMATH_GPT_largest_integer_n_l2416_241697


namespace NUMINAMATH_GPT_average_excluding_highest_lowest_l2416_241619

-- Define the conditions
def batting_average : ℚ := 59
def innings : ℕ := 46
def highest_score : ℕ := 156
def score_difference : ℕ := 150
def lowest_score : ℕ := highest_score - score_difference

-- Prove the average excluding the highest and lowest innings is 58
theorem average_excluding_highest_lowest :
  let total_runs := batting_average * innings
  let runs_excluding := total_runs - highest_score - lowest_score
  let effective_innings := innings - 2
  runs_excluding / effective_innings = 58 := by
  -- Insert proof here
  sorry

end NUMINAMATH_GPT_average_excluding_highest_lowest_l2416_241619


namespace NUMINAMATH_GPT_simplify_fraction_l2416_241655

theorem simplify_fraction (x : ℝ) : (2 * x - 3) / 4 + (4 * x + 5) / 3 = (22 * x + 11) / 12 := by
  sorry

end NUMINAMATH_GPT_simplify_fraction_l2416_241655


namespace NUMINAMATH_GPT_joe_lowest_test_score_dropped_l2416_241617

theorem joe_lowest_test_score_dropped 
  (A B C D : ℝ) 
  (h1 : A + B + C + D = 360) 
  (h2 : A + B + C = 255) :
  D = 105 :=
sorry

end NUMINAMATH_GPT_joe_lowest_test_score_dropped_l2416_241617


namespace NUMINAMATH_GPT_ratio_of_earnings_l2416_241666

theorem ratio_of_earnings (jacob_hourly: ℕ) (jake_total: ℕ) (days: ℕ) (hours_per_day: ℕ) (jake_hourly: ℕ) (ratio: ℕ) 
  (h_jacob: jacob_hourly = 6)
  (h_jake_total: jake_total = 720)
  (h_days: days = 5)
  (h_hours_per_day: hours_per_day = 8)
  (h_jake_hourly: jake_hourly = jake_total / (days * hours_per_day))
  (h_ratio: ratio = jake_hourly / jacob_hourly) :
  ratio = 3 := 
sorry

end NUMINAMATH_GPT_ratio_of_earnings_l2416_241666


namespace NUMINAMATH_GPT_part_a_part_b_l2416_241665

-- Definition based on conditions
def S (n k : ℕ) : ℕ :=
  -- Placeholder: Actual definition would count the coefficients
  -- of (x+1)^n that are not divisible by k.
  sorry

-- Part (a) proof statement
theorem part_a : S 2012 3 = 324 :=
by sorry

-- Part (b) proof statement
theorem part_b : 2012 ∣ S (2012^2011) 2011 :=
by sorry

end NUMINAMATH_GPT_part_a_part_b_l2416_241665


namespace NUMINAMATH_GPT_max_E_l2416_241610

def E (x₁ x₂ x₃ x₄ : ℝ) : ℝ :=
  x₁ + x₂ + x₃ + x₄ -
  x₁ * x₂ - x₁ * x₃ - x₁ * x₄ -
  x₂ * x₃ - x₂ * x₄ - x₃ * x₄ +
  x₁ * x₂ * x₃ + x₁ * x₂ * x₄ +
  x₁ * x₃ * x₄ + x₂ * x₃ * x₄ -
  x₁ * x₂ * x₃ * x₄

theorem max_E (x₁ x₂ x₃ x₄ : ℝ) (h₁ : 0 ≤ x₁) (h₂ : x₁ ≤ 1) (h₃ : 0 ≤ x₂) (h₄ : x₂ ≤ 1) (h₅ : 0 ≤ x₃) (h₆ : x₃ ≤ 1) (h₇ : 0 ≤ x₄) (h₈ : x₄ ≤ 1) : 
  E x₁ x₂ x₃ x₄ ≤ 1 :=
sorry

end NUMINAMATH_GPT_max_E_l2416_241610


namespace NUMINAMATH_GPT_f_is_periodic_l2416_241630

-- Define the conditions for the function f
def f (x : ℝ) : ℝ := sorry
axiom f_defined : ∀ x : ℝ, f x ≠ 0
axiom f_property : ∃ a : ℝ, a > 0 ∧ ∀ x : ℝ, f (x - a) = 1 / f x

-- Formal problem statement to be proven
theorem f_is_periodic : ∃ a : ℝ, a > 0 ∧ ∀ x : ℝ, f x = f (x + 2 * a) :=
by {
  sorry
}

end NUMINAMATH_GPT_f_is_periodic_l2416_241630


namespace NUMINAMATH_GPT_smallest_integer_condition_l2416_241695

theorem smallest_integer_condition :
  ∃ (x : ℕ) (d : ℕ) (n : ℕ) (p : ℕ), x = 1350 ∧ d = 1 ∧ n = 450 ∧ p = 2 ∧
  x = 10^p * d + n ∧
  n = x / 19 ∧
  (1 ≤ d ∧ d ≤ 9 ∧ 10^p * d % 18 = 0) :=
sorry

end NUMINAMATH_GPT_smallest_integer_condition_l2416_241695


namespace NUMINAMATH_GPT_find_percentage_l2416_241638

theorem find_percentage (P : ℕ) (h1 : 0.20 * 650 = 130) (h2 : P * 800 / 100 = 320) : P = 40 := 
by { 
  sorry 
}

end NUMINAMATH_GPT_find_percentage_l2416_241638


namespace NUMINAMATH_GPT_train_length_is_correct_l2416_241671

-- Definitions of speeds and time
def speedTrain_kmph := 100
def speedMotorbike_kmph := 64
def overtakingTime_s := 20

-- Calculate speeds in m/s
def speedTrain_mps := speedTrain_kmph * 1000 / 3600
def speedMotorbike_mps := speedMotorbike_kmph * 1000 / 3600

-- Calculate relative speed
def relativeSpeed_mps := speedTrain_mps - speedMotorbike_mps

-- Calculate the length of the train
def length_of_train := relativeSpeed_mps * overtakingTime_s

-- Theorem: Verifying the length of the train is 200 meters
theorem train_length_is_correct : length_of_train = 200 := by
  -- Sorry placeholder for proof
  sorry

end NUMINAMATH_GPT_train_length_is_correct_l2416_241671


namespace NUMINAMATH_GPT_rectangle_length_fraction_l2416_241679

theorem rectangle_length_fraction 
  (s r : ℝ) 
  (A b ℓ : ℝ)
  (area_square : s * s = 1600)
  (radius_eq_side : r = s)
  (area_rectangle : A = ℓ * b)
  (breadth_rect : b = 10)
  (area_rect_val : A = 160) :
  ℓ / r = 2 / 5 := 
by
  sorry

end NUMINAMATH_GPT_rectangle_length_fraction_l2416_241679


namespace NUMINAMATH_GPT_ratio_enlarged_by_nine_l2416_241685

theorem ratio_enlarged_by_nine (a b : ℕ) (h : b ≠ 0) :
  (3 * a) / (b / 3) = 9 * (a / b) :=
by
  have h1 : b / 3 ≠ 0 := by sorry
  have h2 : a * 3 ≠ 0 := by sorry
  sorry

end NUMINAMATH_GPT_ratio_enlarged_by_nine_l2416_241685


namespace NUMINAMATH_GPT_tangent_line_through_B_l2416_241670

theorem tangent_line_through_B (x : ℝ) (y : ℝ) (x₀ : ℝ) (y₀ : ℝ) :
  (y₀ = x₀^2) →
  (y - y₀ = 2*x₀*(x - x₀)) →
  (3, 5) ∈ ({p : ℝ × ℝ | ∃ t, p.2 - t^2 = 2*t*(p.1 - t)}) →
  (x = 2 * x₀) ∧ (y = y₀) →
  (2*x - y - 1 = 0 ∨ 10*x - y - 25 = 0) :=
by
  intros h1 h2 h3 h4
  sorry

end NUMINAMATH_GPT_tangent_line_through_B_l2416_241670


namespace NUMINAMATH_GPT_number_of_white_stones_is_3600_l2416_241625

-- Definitions and conditions
def total_stones : ℕ := 6000
def total_difference_to_4800 : ℕ := 4800
def W : ℕ := 3600

-- Conditions
def condition1 (B : ℕ) : Prop := total_stones - W + B = total_difference_to_4800
def condition2 (B : ℕ) : Prop := W + B = total_stones
def condition3 (B : ℕ) : Prop := W > B

-- Theorem statement
theorem number_of_white_stones_is_3600 :
  ∃ B : ℕ, condition1 B ∧ condition2 B ∧ condition3 B :=
by
  -- TODO: Complete the proof
  sorry

end NUMINAMATH_GPT_number_of_white_stones_is_3600_l2416_241625


namespace NUMINAMATH_GPT_parabola_no_intersect_l2416_241604

theorem parabola_no_intersect (m : ℝ) : 
  (¬ ∃ x : ℝ, -x^2 - 6*x + m = 0 ) ↔ m < -9 :=
by
  sorry

end NUMINAMATH_GPT_parabola_no_intersect_l2416_241604


namespace NUMINAMATH_GPT_four_fours_to_seven_l2416_241644

theorem four_fours_to_seven :
  (∃ eq1 eq2 : ℕ, eq1 ≠ eq2 ∧
    (eq1 = 4 + 4 - (4 / 4) ∧
     eq2 = 44 / 4 - 4 ∧ eq1 = 7 ∧ eq2 = 7)) :=
by
  existsi (4 + 4 - (4 / 4))
  existsi (44 / 4 - 4)
  sorry

end NUMINAMATH_GPT_four_fours_to_seven_l2416_241644


namespace NUMINAMATH_GPT_complex_sum_power_l2416_241668

noncomputable def z : ℂ := sorry

theorem complex_sum_power (hz : z^2 + z + 1 = 0) :
  z^100 + z^101 + z^102 + z^103 + z^104 = -1 :=
sorry

end NUMINAMATH_GPT_complex_sum_power_l2416_241668


namespace NUMINAMATH_GPT_max_value_of_y_l2416_241623

theorem max_value_of_y (x : ℝ) (h : 0 < x ∧ x < 1 / 2) : (∃ y, y = x^2 * (1 - 2*x) ∧ y ≤ 1 / 27) :=
sorry

end NUMINAMATH_GPT_max_value_of_y_l2416_241623


namespace NUMINAMATH_GPT_octagon_area_difference_l2416_241633

theorem octagon_area_difference (side_length : ℝ) (h : side_length = 1) : 
  let A := 2 * (1 + Real.sqrt 2)
  let triangle_area := (1 / 2) * (1 / 2) * (1 / 2)
  let gray_area := 4 * triangle_area
  let part_with_lines := A - gray_area
  (gray_area - part_with_lines) = 1 / 4 :=
by
  sorry

end NUMINAMATH_GPT_octagon_area_difference_l2416_241633


namespace NUMINAMATH_GPT_algebraic_expression_interpretation_l2416_241635

def donations_interpretation (m n : ℝ) : ℝ := 5 * m + 2 * n
def plazas_area_interpretation (a : ℝ) : ℝ := 6 * a^2

theorem algebraic_expression_interpretation (m n a : ℝ) :
  donations_interpretation m n = 5 * m + 2 * n ∧ plazas_area_interpretation a = 6 * a^2 :=
by
  sorry

end NUMINAMATH_GPT_algebraic_expression_interpretation_l2416_241635


namespace NUMINAMATH_GPT_externally_tangent_internally_tangent_common_chord_and_length_l2416_241650

-- Definitions of Circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 - 2*x - 6*y - 1 = 0
def circle2 (x y : ℝ) (m : ℝ) : Prop := x^2 + y^2 - 10*x - 12*y + m = 0

-- Proof problem 1: Externally tangent
theorem externally_tangent (m : ℝ) : (∃ x y : ℝ, circle1 x y ∧ circle2 x y m) → m = 25 + 10 * Real.sqrt 11 :=
sorry

-- Proof problem 2: Internally tangent
theorem internally_tangent (m : ℝ) : (∃ x y : ℝ, circle1 x y ∧ circle2 x y m) → m = 25 - 10 * Real.sqrt 11 :=
sorry

-- Proof problem 3: Common chord and length when m = 45
theorem common_chord_and_length :
  (∃ x y : ℝ, circle2 x y 45) →
  (∃ l : ℝ, l = 4 * Real.sqrt 7 ∧ ∀ x y : ℝ, (circle1 x y ∧ circle2 x y 45) → (4*x + 3*y - 23 = 0)) :=
sorry

end NUMINAMATH_GPT_externally_tangent_internally_tangent_common_chord_and_length_l2416_241650


namespace NUMINAMATH_GPT_positive_difference_of_b_l2416_241660

def g (n : Int) : Int :=
  if n < 0 then n^2 + 3 else 2 * n - 25

theorem positive_difference_of_b :
  let s := g (-3) + g 3
  let t b := g b = -s
  ∃ a b, t a ∧ t b ∧ a ≠ b ∧ |a - b| = 18 :=
by
  sorry

end NUMINAMATH_GPT_positive_difference_of_b_l2416_241660


namespace NUMINAMATH_GPT_number_of_fish_disappeared_l2416_241663

-- First, define initial amounts of each type of fish
def goldfish_initial := 7
def catfish_initial := 12
def guppies_initial := 8
def angelfish_initial := 5

-- Define the total initial number of fish
def total_fish_initial := goldfish_initial + catfish_initial + guppies_initial + angelfish_initial

-- Define the current number of fish
def fish_current := 27

-- Define the number of fish disappeared
def fish_disappeared := total_fish_initial - fish_current

-- Proof statement
theorem number_of_fish_disappeared:
  fish_disappeared = 5 :=
by
  -- Sorry is a placeholder that indicates the proof is omitted.
  sorry

end NUMINAMATH_GPT_number_of_fish_disappeared_l2416_241663


namespace NUMINAMATH_GPT_min_time_to_cook_noodles_l2416_241657

/-- 
Li Ming needs to cook noodles, following these steps: 
① Boil the noodles for 4 minutes; 
② Wash vegetables for 5 minutes; 
③ Prepare the noodles and condiments for 2 minutes; 
④ Boil the water in the pot for 10 minutes; 
⑤ Wash the pot and add water for 2 minutes. 
Apart from step ④, only one step can be performed at a time. 
Prove that the minimum number of minutes needed to complete these tasks is 16.
-/
def total_time : Nat :=
  let t5 := 2 -- Wash the pot and add water
  let t4 := 10 -- Boil the water in the pot
  let t2 := 5 -- Wash vegetables
  let t3 := 2 -- Prepare the noodles and condiments
  let t1 := 4 -- Boil the noodles
  t5 + t4.max (t2 + t3) + t1

theorem min_time_to_cook_noodles : total_time = 16 :=
by
  sorry

end NUMINAMATH_GPT_min_time_to_cook_noodles_l2416_241657


namespace NUMINAMATH_GPT_souvenirs_total_cost_l2416_241626

theorem souvenirs_total_cost (T : ℝ) (H1 : 347 = T + 146) : T + 347 = 548 :=
by
  -- To ensure the validity of the Lean statement but without the proof.
  sorry

end NUMINAMATH_GPT_souvenirs_total_cost_l2416_241626


namespace NUMINAMATH_GPT_janet_total_earnings_l2416_241694

def hourly_wage_exterminator := 70
def hourly_work_exterminator := 20
def sculpture_price_per_pound := 20
def sculpture_1_weight := 5
def sculpture_2_weight := 7

theorem janet_total_earnings :
  (hourly_wage_exterminator * hourly_work_exterminator) +
  (sculpture_price_per_pound * sculpture_1_weight) +
  (sculpture_price_per_pound * sculpture_2_weight) = 1640 := by
  sorry

end NUMINAMATH_GPT_janet_total_earnings_l2416_241694


namespace NUMINAMATH_GPT_no_solution_k_l2416_241675

theorem no_solution_k (k : ℝ) : 
  (∀ t s : ℝ, 
    ∃ (a : ℝ × ℝ) (b : ℝ × ℝ) (c : ℝ × ℝ) (d : ℝ × ℝ), 
      (a = (2, 7)) ∧ 
      (b = (5, -9)) ∧ 
      (c = (4, -3)) ∧ 
      (d = (-2, k)) ∧ 
      (a + t • b ≠ c + s • d)) ↔ k = 18 / 5 := 
by
  sorry

end NUMINAMATH_GPT_no_solution_k_l2416_241675


namespace NUMINAMATH_GPT_John_total_amount_l2416_241616

theorem John_total_amount (x : ℝ)
  (h1 : ∃ x : ℝ, (3 * x * 5 * 3 * x) = 300):
  (x + 3 * x + 15 * x) = 380 := by
  sorry

end NUMINAMATH_GPT_John_total_amount_l2416_241616


namespace NUMINAMATH_GPT_cube_with_holes_l2416_241607

-- Definitions and conditions
def edge_length_cube : ℝ := 4
def side_length_hole : ℝ := 2
def depth_hole : ℝ := 1
def number_of_holes : ℕ := 6

-- Prove that the total surface area including inside surfaces is 144 square meters
def total_surface_area_including_inside_surfaces : ℝ :=
  let original_surface_area := 6 * (edge_length_cube ^ 2)
  let area_removed_per_hole := side_length_hole ^ 2
  let area_exposed_inside_per_hole := 2 * (side_length_hole * depth_hole) + area_removed_per_hole
  original_surface_area - number_of_holes * area_removed_per_hole + number_of_holes * area_exposed_inside_per_hole

-- Prove that the total volume of material removed is 24 cubic meters
def total_volume_removed : ℝ :=
  number_of_holes * (side_length_hole ^ 2 * depth_hole)

theorem cube_with_holes :
  total_surface_area_including_inside_surfaces = 144 ∧ total_volume_removed = 24 :=
by
  sorry

end NUMINAMATH_GPT_cube_with_holes_l2416_241607


namespace NUMINAMATH_GPT_train_length_l2416_241631

open Real

theorem train_length 
  (v : ℝ) -- speed of the train in km/hr
  (t : ℝ) -- time in seconds
  (d : ℝ) -- length of the bridge in meters
  (h_v : v = 36) -- condition 1
  (h_t : t = 50) -- condition 2
  (h_d : d = 140) -- condition 3
  : (v * 1000 / 3600) * t = 360 + 140 := 
sorry

end NUMINAMATH_GPT_train_length_l2416_241631


namespace NUMINAMATH_GPT_sum_squares_condition_l2416_241636

theorem sum_squares_condition
  (a b c : ℝ)
  (h1 : a^2 + b^2 + c^2 = 75)
  (h2 : ab + bc + ca = 40)
  (h3 : c = 5) :
  a + b + c = 5 * Real.sqrt 62 :=
by sorry

end NUMINAMATH_GPT_sum_squares_condition_l2416_241636


namespace NUMINAMATH_GPT_ceil_and_floor_difference_l2416_241615

theorem ceil_and_floor_difference (x : ℝ) (ε : ℝ) 
  (h_cond : ⌈x + ε⌉ - ⌊x + ε⌋ = 1) (h_eps : 0 < ε ∧ ε < 1) :
  ⌈x + ε⌉ - (x + ε) = 1 - ε :=
sorry

end NUMINAMATH_GPT_ceil_and_floor_difference_l2416_241615


namespace NUMINAMATH_GPT_range_of_a_l2416_241658

open Real

noncomputable def f (x : ℝ) : ℝ := abs (log x)

noncomputable def g (x : ℝ) : ℝ := 
  if 0 < x ∧ x ≤ 1 then 0 
  else abs (x^2 - 4) - 2

noncomputable def h (x : ℝ) : ℝ := f x + g x

theorem range_of_a (a : ℝ) : (∀ x : ℝ, |h x| = a → has_four_real_roots : Prop) ↔ (1 ≤ a ∧ a < 2 - log 2) := sorry

end NUMINAMATH_GPT_range_of_a_l2416_241658


namespace NUMINAMATH_GPT_value_of_b_l2416_241632

theorem value_of_b (b : ℝ) (x : ℝ) (h : x = 1) (h_eq : 3 * x^2 - b * x + 3 = 0) : b = 6 :=
by
  sorry

end NUMINAMATH_GPT_value_of_b_l2416_241632


namespace NUMINAMATH_GPT_asymptotes_of_hyperbola_l2416_241647

theorem asymptotes_of_hyperbola (k : ℤ) (h1 : (k - 2016) * (k - 2018) < 0) :
  ∀ x y: ℝ, (x ^ 2) - (y ^ 2) = 1 → ∃ a b: ℝ, y = x * a ∨ y = x * b :=
by
  sorry

end NUMINAMATH_GPT_asymptotes_of_hyperbola_l2416_241647


namespace NUMINAMATH_GPT_existence_of_point_N_l2416_241622

-- Given conditions
def is_point_on_ellipse (x y a b : ℝ) : Prop := 
  (x^2 / a^2) + (y^2 / b^2) = 1

def is_ellipse (a b : ℝ) : Prop := 
  a > b ∧ b > 0 ∧ (a^2 = b^2 + (a * (Real.sqrt 2) / 2)^2)

def passes_through_point (x y a b : ℝ) (px py : ℝ) : Prop :=
  (px^2 / a^2) + (py^2 / b^2) = 1

def ellipse_with_eccentricity (a : ℝ) : Prop :=
  (Real.sqrt 2) / 2 = (Real.sqrt (a^2 - (a * (Real.sqrt 2) / 2)^2)) / a

def line_through_point (k : ℝ) (x y : ℝ) : Prop :=
  y = k * x + 1

def lines_intersect_ellipse (k a b : ℝ) : Prop :=
  ∃ x1 y1 x2 y2, line_through_point k x1 y1 ∧ line_through_point k x2 y2 ∧ is_point_on_ellipse x1 y1 a b ∧ is_point_on_ellipse x2 y2 a b

def angle_condition (k t a b : ℝ) : Prop :=
  ∃ x1 y1 x2 y2, line_through_point k x1 y1 ∧ line_through_point k x2 y2 ∧ is_point_on_ellipse x1 y1 a b ∧ is_point_on_ellipse x2 y2 a b ∧ 
  ((y1 - t) / x1) + ((y2 - t) / x2) = 0

-- Lean 4 statement
theorem existence_of_point_N (a b k t : ℝ) (hx : is_ellipse a b) (hp : passes_through_point 2 (Real.sqrt 2) a b 2 (Real.sqrt 2)) (he : ellipse_with_eccentricity a) (hl : ∀ (x1 y1 x2 y2 : ℝ), lines_intersect_ellipse k a b) :
  ∃ (N : ℝ), N = 4 ∧ angle_condition k N a b :=
sorry

end NUMINAMATH_GPT_existence_of_point_N_l2416_241622


namespace NUMINAMATH_GPT_solve_system1_solve_system2_l2416_241648

theorem solve_system1 (x y : ℝ) (h1 : y = x - 4) (h2 : x + y = 6) : x = 5 ∧ y = 1 :=
by sorry

theorem solve_system2 (x y : ℝ) (h1 : 2 * x + y = 1) (h2 : 4 * x - y = 5) : x = 1 ∧ y = -1 :=
by sorry

end NUMINAMATH_GPT_solve_system1_solve_system2_l2416_241648


namespace NUMINAMATH_GPT_analytical_expression_of_odd_function_l2416_241649

noncomputable def f (x : ℝ) : ℝ :=
if x > 0 then x^2 - 2 * x + 3 
else if x = 0 then 0 
else -x^2 - 2 * x - 3

theorem analytical_expression_of_odd_function :
  ∀ x : ℝ, f x =
    if x > 0 then x^2 - 2 * x + 3 
    else if x = 0 then 0 
    else -x^2 - 2 * x - 3 :=
by
  sorry

end NUMINAMATH_GPT_analytical_expression_of_odd_function_l2416_241649


namespace NUMINAMATH_GPT_ratio_passengers_i_to_ii_l2416_241603

-- Definitions: Conditions from the problem
variables (total_fare : ℕ) (fare_ii_class : ℕ) (fare_i_class_ratio_to_ii : ℕ)

-- Given conditions
axiom total_fare_collected : total_fare = 1325
axiom fare_collected_from_ii_class : fare_ii_class = 1250
axiom i_to_ii_fare_ratio : fare_i_class_ratio_to_ii = 3

-- Define the fare for I class and II class passengers
def fare_i_class := 3 * (fare_ii_class / fare_i_class_ratio_to_ii)

-- Statement of the proof problem translating the question, conditions, and answer
theorem ratio_passengers_i_to_ii (x y : ℕ) (h1 : 3 * fare_i_class * x = total_fare - fare_ii_class)
    (h2 : (fare_ii_class / fare_i_class_ratio_to_ii) * y = fare_ii_class) : x = y / 50 :=
by
  sorry

end NUMINAMATH_GPT_ratio_passengers_i_to_ii_l2416_241603


namespace NUMINAMATH_GPT_people_on_trolley_l2416_241618

-- Given conditions
variable (X : ℕ)

def initial_people : ℕ := 10

def second_stop_people : ℕ := initial_people - 3 + 20

def third_stop_people : ℕ := second_stop_people - 18 + 2

def fourth_stop_people : ℕ := third_stop_people - 5 + X

-- Prove the current number of people on the trolley is 6 + X
theorem people_on_trolley (X : ℕ) : 
  fourth_stop_people X = 6 + X := 
by 
  unfold fourth_stop_people
  unfold third_stop_people
  unfold second_stop_people
  unfold initial_people
  sorry

end NUMINAMATH_GPT_people_on_trolley_l2416_241618


namespace NUMINAMATH_GPT_seating_arrangement_fixed_pairs_l2416_241640

theorem seating_arrangement_fixed_pairs 
  (total_chairs : ℕ) 
  (total_people : ℕ) 
  (specific_pair_adjacent : Prop)
  (comb : ℕ) 
  (four_factorial : ℕ) 
  (two_factorial : ℕ) 
  : total_chairs = 6 → total_people = 5 → specific_pair_adjacent → comb = Nat.choose 6 4 → 
    four_factorial = Nat.factorial 4 → two_factorial = Nat.factorial 2 → 
    Nat.choose 6 4 * Nat.factorial 4 * Nat.factorial 2 = 720 
  := by
  intros
  sorry

end NUMINAMATH_GPT_seating_arrangement_fixed_pairs_l2416_241640


namespace NUMINAMATH_GPT_triangle_ABC_two_solutions_l2416_241656

theorem triangle_ABC_two_solutions (x : ℝ) (h1 : x > 0) : 
  2 < x ∧ x < 2 * Real.sqrt 2 ↔
  (∃ a b B, a = x ∧ b = 2 ∧ B = Real.pi / 4 ∧ a * Real.sin B < b ∧ b < a) := by
  sorry

end NUMINAMATH_GPT_triangle_ABC_two_solutions_l2416_241656


namespace NUMINAMATH_GPT_min_value_fraction_l2416_241611

theorem min_value_fraction {x y : ℝ} (hx : x > 0) (hy : y > 0) (h : x + 3 * y = 1) :
  (∀ y : ℝ,  y > 0 → (∀ x : ℝ, x > 0 → x + 3 * y = 1 → (1/x + 1/(3*y)) ≥ 4)) :=
sorry

end NUMINAMATH_GPT_min_value_fraction_l2416_241611


namespace NUMINAMATH_GPT_bhanu_income_percentage_l2416_241639

variable {I P : ℝ}

theorem bhanu_income_percentage (h₁ : 300 = (P / 100) * I)
                                  (h₂ : 210 = 0.3 * (I - 300)) :
  P = 30 :=
by
  sorry

end NUMINAMATH_GPT_bhanu_income_percentage_l2416_241639


namespace NUMINAMATH_GPT_puppy_price_l2416_241688

theorem puppy_price (P : ℕ) (kittens_price : ℕ) (total_earnings : ℕ) :
  (kittens_price = 2 * 6) → (total_earnings = 17) → (kittens_price + P = total_earnings) → P = 5 :=
by
  intros h1 h2 h3
  sorry

end NUMINAMATH_GPT_puppy_price_l2416_241688


namespace NUMINAMATH_GPT_intersection_of_A_and_B_l2416_241669

def A : Set ℝ := {-1, 0, 1, 2}
def B : Set ℝ := {x : ℝ | -1 < x ∧ x ≤ 1}

theorem intersection_of_A_and_B :
  ∀ x : ℝ, (x ∈ A ∩ B) ↔ (x = 0 ∨ x = 1) := by
  sorry

end NUMINAMATH_GPT_intersection_of_A_and_B_l2416_241669


namespace NUMINAMATH_GPT_jackson_running_increase_l2416_241620

theorem jackson_running_increase
    (initial_miles_per_day : ℕ)
    (final_miles_per_day : ℕ)
    (weeks_increasing : ℕ)
    (total_weeks : ℕ)
    (h1 : initial_miles_per_day = 3)
    (h2 : final_miles_per_day = 7)
    (h3 : weeks_increasing = 4)
    (h4 : total_weeks = 5) :
    (final_miles_per_day - initial_miles_per_day) / weeks_increasing = 1 := 
by
  -- provided steps from solution
  sorry

end NUMINAMATH_GPT_jackson_running_increase_l2416_241620


namespace NUMINAMATH_GPT_minimum_value_of_expression_l2416_241661

theorem minimum_value_of_expression {k x1 x2 : ℝ} 
  (h1 : x1 + x2 = -2 * k)
  (h2 : x1 * x2 = k^2 + k + 3) : 
  (x1 - 1)^2 + (x2 - 1)^2 ≥ 8 :=
sorry

end NUMINAMATH_GPT_minimum_value_of_expression_l2416_241661


namespace NUMINAMATH_GPT_no_intersections_root_of_quadratic_l2416_241689

theorem no_intersections_root_of_quadratic (x : ℝ) :
  ¬(∃ x, (y = x) ∧ (y = x - 3)) ↔ (x^2 - 3 * x = 0) := by
  sorry

end NUMINAMATH_GPT_no_intersections_root_of_quadratic_l2416_241689


namespace NUMINAMATH_GPT_range_of_a_l2416_241641

theorem range_of_a (a : ℝ) :
  (∃ A : Finset ℝ, 
    (∀ x, x ∈ A ↔ x^3 - 2 * x^2 + a * x = 0) ∧ A.card = 3) ↔ (a < 0 ∨ (0 < a ∧ a < 1)) :=
by
  sorry

end NUMINAMATH_GPT_range_of_a_l2416_241641


namespace NUMINAMATH_GPT_perpendicular_slope_l2416_241699

theorem perpendicular_slope (x y : ℝ) (h : 5 * x - 2 * y = 10) :
  ∀ (m' : ℝ), m' = -2 / 5 :=
by
  sorry

end NUMINAMATH_GPT_perpendicular_slope_l2416_241699


namespace NUMINAMATH_GPT_initial_value_l2416_241664

theorem initial_value (x k : ℤ) (h : x + 294 = k * 456) : x = 162 :=
sorry

end NUMINAMATH_GPT_initial_value_l2416_241664


namespace NUMINAMATH_GPT_min_value_inequality_l2416_241674

theorem min_value_inequality (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : a + 3 * b = 1) :
  1 / a + 3 / b ≥ 16 := 
by
  sorry

end NUMINAMATH_GPT_min_value_inequality_l2416_241674


namespace NUMINAMATH_GPT_impossible_to_form_11x12x13_parallelepiped_l2416_241677

def is_possible_to_form_parallelepiped
  (brick_shapes_form_unit_cubes : Prop)
  (dimensions : ℕ × ℕ × ℕ) : Prop :=
  ∃ bricks : ℕ, 
    (bricks * 4 = dimensions.fst * dimensions.snd * dimensions.snd.fst)

theorem impossible_to_form_11x12x13_parallelepiped 
  (dimensions := (11, 12, 13)) 
  (brick_shapes_form_unit_cubes : Prop) : 
  ¬ is_possible_to_form_parallelepiped brick_shapes_form_unit_cubes dimensions := 
sorry

end NUMINAMATH_GPT_impossible_to_form_11x12x13_parallelepiped_l2416_241677


namespace NUMINAMATH_GPT_packed_oranges_l2416_241673

theorem packed_oranges (oranges_per_box : ℕ) (boxes_used : ℕ) (total_oranges : ℕ) 
  (h1 : oranges_per_box = 10) (h2 : boxes_used = 265) : 
  total_oranges = 2650 :=
by 
  sorry

end NUMINAMATH_GPT_packed_oranges_l2416_241673


namespace NUMINAMATH_GPT_white_roses_count_l2416_241682

def total_flowers : ℕ := 6284
def red_roses : ℕ := 1491
def yellow_carnations : ℕ := 3025
def white_roses : ℕ := total_flowers - (red_roses + yellow_carnations)

theorem white_roses_count :
  white_roses = 1768 := by
  sorry

end NUMINAMATH_GPT_white_roses_count_l2416_241682


namespace NUMINAMATH_GPT_theater_total_seats_l2416_241690

theorem theater_total_seats
  (occupied_seats : ℕ) (empty_seats : ℕ) 
  (h1 : occupied_seats = 532) (h2 : empty_seats = 218) :
  occupied_seats + empty_seats = 750 := 
by
  -- This is the placeholder for the proof
  sorry

end NUMINAMATH_GPT_theater_total_seats_l2416_241690


namespace NUMINAMATH_GPT_closest_multiple_of_18_2021_l2416_241643

def is_multiple_of (n k : ℕ) : Prop := ∃ m : ℕ, n = k * m

def closest_multiple_of (n k : ℕ) : ℕ :=
if (n % k) * 2 < k then n - (n % k) else n + (k - n % k)

theorem closest_multiple_of_18_2021 :
  closest_multiple_of 2021 18 = 2016 := by
    sorry

end NUMINAMATH_GPT_closest_multiple_of_18_2021_l2416_241643


namespace NUMINAMATH_GPT_total_grains_in_grey_regions_l2416_241606

def total_grains_circle1 : ℕ := 87
def total_grains_circle2 : ℕ := 110
def white_grains_circle1 : ℕ := 68
def white_grains_circle2 : ℕ := 68

theorem total_grains_in_grey_regions : total_grains_circle1 - white_grains_circle1 + (total_grains_circle2 - white_grains_circle2) = 61 :=
by
  sorry

end NUMINAMATH_GPT_total_grains_in_grey_regions_l2416_241606


namespace NUMINAMATH_GPT_pencil_cost_l2416_241678

theorem pencil_cost (P : ℕ) (h1 : ∀ p : ℕ, p = 80) (h2 : ∀ p_est, ((16 * P) + (20 * 80)) = p_est → p_est = 2000) (h3 : 36 = 16 + 20) :
    P = 25 :=
  sorry

end NUMINAMATH_GPT_pencil_cost_l2416_241678


namespace NUMINAMATH_GPT_chairs_left_proof_l2416_241672

def red_chairs : ℕ := 4
def yellow_chairs : ℕ := 2 * red_chairs
def blue_chairs : ℕ := 3 * yellow_chairs
def green_chairs : ℕ := blue_chairs / 2
def orange_chairs : ℕ := green_chairs + 2
def total_chairs : ℕ := red_chairs + yellow_chairs + blue_chairs + green_chairs + orange_chairs
def borrowed_chairs : ℕ := 5 + 3
def chairs_left : ℕ := total_chairs - borrowed_chairs

theorem chairs_left_proof : chairs_left = 54 := by
  -- This is where the proof would go
  sorry

end NUMINAMATH_GPT_chairs_left_proof_l2416_241672


namespace NUMINAMATH_GPT_arithmetic_geometric_mean_inequality_l2416_241683

open BigOperators

noncomputable def A (a : Fin n → ℝ) : ℝ := (Finset.univ.sum a) / n

noncomputable def G (a : Fin n → ℝ) : ℝ := (Finset.univ.prod a) ^ (1 / n)

theorem arithmetic_geometric_mean_inequality (n : ℕ) (a : Fin n → ℝ) (h : ∀ i, 0 < a i) : A a ≥ G a :=
  sorry

end NUMINAMATH_GPT_arithmetic_geometric_mean_inequality_l2416_241683


namespace NUMINAMATH_GPT_find_a_for_inverse_proportion_l2416_241601

theorem find_a_for_inverse_proportion (a : ℝ)
  (h_A : ∃ k : ℝ, 4 = k / (-1))
  (h_B : ∃ k : ℝ, 2 = k / a) :
  a = -2 :=
sorry

end NUMINAMATH_GPT_find_a_for_inverse_proportion_l2416_241601


namespace NUMINAMATH_GPT_percentage_of_money_spent_is_80_l2416_241698

-- Define the cost of items
def cheeseburger_cost : ℕ := 3
def milkshake_cost : ℕ := 5
def cheese_fries_cost : ℕ := 8

-- Define the amount of money Jim and his cousin brought
def jim_money : ℕ := 20
def cousin_money : ℕ := 10

-- Define the total cost of the meal
def total_cost : ℕ :=
  2 * cheeseburger_cost + 2 * milkshake_cost + cheese_fries_cost

-- Define the combined money they brought
def combined_money : ℕ := jim_money + cousin_money

-- Define the percentage of combined money spent
def percentage_spent : ℕ :=
  (total_cost * 100) / combined_money

theorem percentage_of_money_spent_is_80 :
  percentage_spent = 80 :=
by
  -- proof goes here
  sorry

end NUMINAMATH_GPT_percentage_of_money_spent_is_80_l2416_241698


namespace NUMINAMATH_GPT_smallest_n_transform_l2416_241652

open Real

noncomputable def line1_angle : ℝ := π / 30
noncomputable def line2_angle : ℝ := π / 40
noncomputable def line_slope : ℝ := 2 / 45
noncomputable def transform_angle (theta : ℝ) (n : ℕ) : ℝ := theta + n * (7 * π / 120)

theorem smallest_n_transform (theta : ℝ) (n : ℕ) (m : ℕ)
  (h_line1 : line1_angle = π / 30)
  (h_line2 : line2_angle = π / 40)
  (h_slope : tan theta = line_slope)
  (h_transform : transform_angle theta n = theta + m * 2 * π) :
  n = 120 := 
sorry

end NUMINAMATH_GPT_smallest_n_transform_l2416_241652


namespace NUMINAMATH_GPT_circle_radius_l2416_241686

theorem circle_radius :
  ∃ r : ℝ, ∀ x y : ℝ, (x^2 - 8 * x + y^2 + 4 * y + 16 = 0) → r = 2 :=
sorry

end NUMINAMATH_GPT_circle_radius_l2416_241686


namespace NUMINAMATH_GPT_find_number_l2416_241602

theorem find_number (x : ℕ) (h : x / 4 + 15 = 27) : x = 48 :=
sorry

end NUMINAMATH_GPT_find_number_l2416_241602


namespace NUMINAMATH_GPT_amy_school_year_hours_l2416_241653

noncomputable def summer_hours_per_week := 40
noncomputable def summer_weeks := 8
noncomputable def summer_earnings := 3200
noncomputable def school_year_weeks := 32
noncomputable def school_year_earnings_needed := 4800

theorem amy_school_year_hours
  (H1 : summer_earnings = summer_hours_per_week * summer_weeks * (summer_earnings / (summer_hours_per_week * summer_weeks)))
  (H2 : school_year_earnings_needed = school_year_weeks * (summer_earnings / (summer_hours_per_week * summer_weeks)))
  : (school_year_earnings_needed / school_year_weeks / (summer_earnings / (summer_hours_per_week * summer_weeks))) = 15 :=
by
  sorry

end NUMINAMATH_GPT_amy_school_year_hours_l2416_241653


namespace NUMINAMATH_GPT_combined_time_in_pool_l2416_241696

theorem combined_time_in_pool : 
    ∀ (Jerry_time Elaine_time George_time Kramer_time : ℕ), 
    Jerry_time = 3 →
    Elaine_time = 2 * Jerry_time →
    George_time = Elaine_time / 3 →
    Kramer_time = 0 →
    Jerry_time + Elaine_time + George_time + Kramer_time = 11 :=
by 
  intros Jerry_time Elaine_time George_time Kramer_time hJerry hElaine hGeorge hKramer
  sorry

end NUMINAMATH_GPT_combined_time_in_pool_l2416_241696
