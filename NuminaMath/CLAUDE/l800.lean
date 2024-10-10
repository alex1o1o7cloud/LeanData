import Mathlib

namespace divisibility_check_l800_80034

theorem divisibility_check : 
  (5641713 % 29 ≠ 0) ∧ (1379235 % 11 = 0) := by sorry

end divisibility_check_l800_80034


namespace unique_solution_quadratic_l800_80005

/-- The equation 9x^2 + nx + 1 = 0 has exactly one solution in x if and only if n = 6 -/
theorem unique_solution_quadratic (n : ℝ) : 
  (∃! x : ℝ, 9 * x^2 + n * x + 1 = 0) ↔ n = 6 := by
sorry

end unique_solution_quadratic_l800_80005


namespace stock_price_increase_l800_80022

theorem stock_price_increase (initial_price : ℝ) (h : initial_price > 0) : 
  let price_after_year1 := initial_price * 1.2
  let price_after_year2 := price_after_year1 * 0.75
  let price_after_year3 := initial_price * 1.26
  (price_after_year3 / price_after_year2 - 1) * 100 = 40 := by
sorry

end stock_price_increase_l800_80022


namespace abcd_sum_proof_l800_80096

/-- Given four different digits A, B, C, and D forming a four-digit number ABCD,
    prove that if ABCD + ABCD = 7314, then ABCD = 3657 -/
theorem abcd_sum_proof (A B C D : ℕ) (h1 : A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D)
    (h2 : 1000 ≤ A * 1000 + B * 100 + C * 10 + D ∧ A * 1000 + B * 100 + C * 10 + D < 10000)
    (h3 : (A * 1000 + B * 100 + C * 10 + D) + (A * 1000 + B * 100 + C * 10 + D) = 7314) :
  A * 1000 + B * 100 + C * 10 + D = 3657 := by
  sorry

end abcd_sum_proof_l800_80096


namespace parabola_transformation_l800_80007

def original_parabola (x : ℝ) : ℝ := 3 * x^2

def transformed_parabola (x : ℝ) : ℝ := 3 * (x - 3)^2 - 1

theorem parabola_transformation :
  ∀ x : ℝ, transformed_parabola x = original_parabola (x - 3) - 1 :=
by sorry

end parabola_transformation_l800_80007


namespace quadratic_roots_l800_80057

theorem quadratic_roots : ∀ x : ℝ, x^2 - 49 = 0 ↔ x = 7 ∨ x = -7 := by
  sorry

end quadratic_roots_l800_80057


namespace max_value_2sin_l800_80028

theorem max_value_2sin (x : ℝ) : ∃ (M : ℝ), M = 2 ∧ ∀ y : ℝ, 2 * Real.sin y ≤ M := by
  sorry

end max_value_2sin_l800_80028


namespace max_candies_l800_80086

theorem max_candies (vitya maria sasha : ℕ) : 
  vitya = 35 →
  maria < vitya →
  sasha = vitya + maria →
  Even sasha →
  vitya + maria + sasha ≤ 136 :=
by sorry

end max_candies_l800_80086


namespace parallel_vectors_y_value_l800_80009

/-- Two vectors are parallel if and only if their components are proportional -/
def parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

/-- Given two parallel vectors a = (2, 3) and b = (4, y + 1), prove that y = 5 -/
theorem parallel_vectors_y_value :
  let a : ℝ × ℝ := (2, 3)
  let b : ℝ × ℝ := (4, y + 1)
  parallel a b → y = 5 := by
sorry

end parallel_vectors_y_value_l800_80009


namespace sum_parts_is_24_l800_80004

/-- A rectangular prism with two opposite corners colored red -/
structure ColoredRectangularPrism where
  length : ℕ
  width : ℕ
  height : ℕ
  red_corners : ℕ
  h_red_corners : red_corners = 2

/-- The sum of edges, non-red corners, and faces of a colored rectangular prism -/
def sum_parts (prism : ColoredRectangularPrism) : ℕ :=
  12 + (8 - prism.red_corners) + 6

theorem sum_parts_is_24 (prism : ColoredRectangularPrism) :
  sum_parts prism = 24 :=
sorry

end sum_parts_is_24_l800_80004


namespace quadratic_properties_l800_80042

-- Define the quadratic function
def f (x : ℝ) : ℝ := x^2 + 2*x - 3

-- State the theorem
theorem quadratic_properties :
  (∀ x, f x ≥ -4) ∧  -- Minimum value is -4
  (f (-1) = -4) ∧    -- Minimum occurs at x = -1
  (f 0 = -3) ∧       -- Passes through (0, -3)
  (f 1 = 0) ∧        -- Intersects x-axis at (1, 0)
  (f (-3) = 0) ∧     -- Intersects x-axis at (-3, 0)
  (∀ x, -2 ≤ x ∧ x ≤ 2 → f x ≤ 5) ∧  -- Maximum value in [-2, 2] is 5
  (f 2 = 5)  -- Maximum value occurs at x = 2
  := by sorry


end quadratic_properties_l800_80042


namespace hyperbola_asymptotes_l800_80085

/-- Given a hyperbola with equation x²/a² - y²/b² = 1 and eccentricity √3,
    its asymptotes are given by y = ±√2 x -/
theorem hyperbola_asymptotes (a b : ℝ) (h : a > 0) (k : b > 0) :
  (∀ x y : ℝ, x^2 / a^2 - y^2 / b^2 = 1) →
  ((a^2 + b^2) / a^2 = 3) →
  (∃ c : ℝ, ∀ x : ℝ, (y = c * x ∨ y = -c * x) ↔ (x / a = y / b ∨ x / a = -y / b)) ∧
  c = Real.sqrt 2 :=
sorry

end hyperbola_asymptotes_l800_80085


namespace red_lucky_stars_count_l800_80070

theorem red_lucky_stars_count (blue : ℕ) (yellow : ℕ) (red : ℕ) :
  blue = 20 →
  yellow = 15 →
  (red : ℚ) / (red + blue + yellow : ℚ) = 1/2 →
  red = 35 := by
sorry

end red_lucky_stars_count_l800_80070


namespace log_12_5_value_l800_80048

-- Define the given conditions
axiom a : ℝ
axiom b : ℝ
axiom lg_2_eq_a : Real.log 2 = a
axiom ten_pow_b_eq_3 : (10 : ℝ)^b = 3

-- State the theorem to be proved
theorem log_12_5_value : Real.log 5 / Real.log 12 = (1 - a) / (2 * a + b) := by sorry

end log_12_5_value_l800_80048


namespace roots_star_zero_l800_80027

-- Define the new operation ※
def star (a b : ℝ) : ℝ := a * b - a - b

-- Define the theorem
theorem roots_star_zero {x₁ x₂ : ℝ} (h : x₁^2 + x₁ - 1 = 0 ∧ x₂^2 + x₂ - 1 = 0) : 
  star x₁ x₂ = 0 := by
  sorry

-- Note: The proof is omitted as per instructions

end roots_star_zero_l800_80027


namespace least_positive_linear_combination_l800_80089

theorem least_positive_linear_combination : 
  ∃ (n : ℕ), n > 0 ∧ (∀ (m : ℤ), (∃ (x y : ℤ), m = 24*x + 16*y) → m = 0 ∨ m.natAbs ≥ n) ∧ 
  (∃ (x y : ℤ), n = 24*x + 16*y) :=
by sorry

end least_positive_linear_combination_l800_80089


namespace intersects_iff_m_ge_neg_one_l800_80075

/-- A quadratic function f(x) = x^2 + 2x - m -/
def f (m : ℝ) (x : ℝ) : ℝ := x^2 + 2*x - m

/-- The graph of f intersects the x-axis -/
def intersects_x_axis (m : ℝ) : Prop :=
  ∃ x : ℝ, f m x = 0

/-- Theorem: The graph of f(x) = x^2 + 2x - m intersects the x-axis
    if and only if m ≥ -1 -/
theorem intersects_iff_m_ge_neg_one (m : ℝ) :
  intersects_x_axis m ↔ m ≥ -1 := by sorry

end intersects_iff_m_ge_neg_one_l800_80075


namespace exam_students_count_l800_80039

theorem exam_students_count (total_average : ℝ) (excluded_average : ℝ) (new_average : ℝ) 
  (excluded_count : ℕ) (h1 : total_average = 80) (h2 : excluded_average = 40) 
  (h3 : new_average = 90) (h4 : excluded_count = 5) : 
  ∃ (N : ℕ), N = 25 ∧ 
  (N : ℝ) * total_average = (N - excluded_count : ℝ) * new_average + 
    (excluded_count : ℝ) * excluded_average :=
by
  sorry

#check exam_students_count

end exam_students_count_l800_80039


namespace quadratic_inequality_solution_l800_80016

theorem quadratic_inequality_solution (a : ℝ) : 
  (∀ x, x ≠ -(1/a) → a*x^2 + 2*x + a > 0) ∧ 
  (∃ x, a*x^2 + 2*x + a ≤ 0) → 
  a = 1 := by
sorry

end quadratic_inequality_solution_l800_80016


namespace negation_of_all_seated_l800_80054

universe u

-- Define the predicates
variable (in_room : α → Prop)
variable (seated : α → Prop)

-- State the theorem
theorem negation_of_all_seated :
  ¬(∀ (x : α), in_room x → seated x) ↔ ∃ (x : α), in_room x ∧ ¬(seated x) :=
by sorry

end negation_of_all_seated_l800_80054


namespace post_office_mail_count_l800_80029

/- Define the daily intake of letters and packages -/
def letters_per_day : ℕ := 60
def packages_per_day : ℕ := 20

/- Define the number of days in a month and the number of months -/
def days_per_month : ℕ := 30
def months : ℕ := 6

/- Define the total pieces of mail per day -/
def mail_per_day : ℕ := letters_per_day + packages_per_day

/- Theorem to prove -/
theorem post_office_mail_count :
  mail_per_day * days_per_month * months = 14400 :=
by sorry

end post_office_mail_count_l800_80029


namespace apple_consumption_l800_80055

theorem apple_consumption (x : ℝ) : 
  x > 0 ∧ x + 2*x + x/2 = 14 → x = 4 := by
  sorry

end apple_consumption_l800_80055


namespace sum_of_perfect_square_integers_l800_80053

theorem sum_of_perfect_square_integers : ∃ (S : Finset ℕ), 
  (∀ n ∈ S, ∃ k : ℕ, n^2 - 19*n + 99 = k^2) ∧ 
  (∀ n : ℕ, n ∉ S → ¬∃ k : ℕ, n^2 - 19*n + 99 = k^2) ∧
  (S.sum id = 38) := by
  sorry

end sum_of_perfect_square_integers_l800_80053


namespace inverse_of_twelve_point_five_l800_80074

theorem inverse_of_twelve_point_five (x : ℝ) : 1 / x = 12.5 → x = 0.08 := by
  sorry

end inverse_of_twelve_point_five_l800_80074


namespace mushroom_ratio_l800_80012

theorem mushroom_ratio (total : ℕ) (safe : ℕ) (uncertain : ℕ) 
  (h1 : total = 32) 
  (h2 : safe = 9) 
  (h3 : uncertain = 5) : 
  (total - safe - uncertain) / safe = 2 := by
  sorry

end mushroom_ratio_l800_80012


namespace vasya_lives_on_fifth_floor_l800_80036

/-- The number of steps Petya walks from the first to the third floor -/
def petya_steps : ℕ := 36

/-- The number of steps Vasya walks from the first floor to his floor -/
def vasya_steps : ℕ := 72

/-- The floor on which Vasya lives -/
def vasya_floor : ℕ := 5

/-- Theorem stating that Vasya lives on the 5th floor given the conditions -/
theorem vasya_lives_on_fifth_floor :
  (petya_steps / 2 = vasya_steps / vasya_floor) →
  vasya_floor = 5 := by
  sorry

end vasya_lives_on_fifth_floor_l800_80036


namespace problem_statement_l800_80067

theorem problem_statement (a b c : ℝ) (h1 : a ≠ b) (h2 : b ≠ c) (h3 : a ≠ c)
  (h4 : a / (b - c) + b / (c - a) + c / (a - b) = -1) :
  a^3 / (b - c)^2 + b^3 / (c - a)^2 + c^3 / (a - b)^2 = 0 := by
  sorry

end problem_statement_l800_80067


namespace bouncing_ball_original_height_l800_80037

/-- Represents the behavior of a bouncing ball -/
def BouncingBall (originalHeight : ℝ) : Prop :=
  let reboundFactor := (1/2 : ℝ)
  let totalTravel := originalHeight +
                     2 * (reboundFactor * originalHeight) +
                     2 * (reboundFactor^2 * originalHeight)
  totalTravel = 250

/-- Theorem stating the original height of the ball -/
theorem bouncing_ball_original_height :
  ∃ (h : ℝ), BouncingBall h ∧ h = 100 := by
  sorry

end bouncing_ball_original_height_l800_80037


namespace n_squared_divisible_by_144_l800_80082

theorem n_squared_divisible_by_144 (n : ℕ+) (h : ∃ t : ℕ+, t = 12 ∧ ∀ k : ℕ+, k ∣ n → k ≤ t) :
  144 ∣ n^2 := by
  sorry

end n_squared_divisible_by_144_l800_80082


namespace total_students_shaking_hands_l800_80030

/-- The number of students from each school who participated in the debate --/
structure SchoolParticipation where
  school1 : ℕ
  school2 : ℕ
  school3 : ℕ

/-- The conditions of the debate participation --/
def debateConditions (p : SchoolParticipation) : Prop :=
  p.school1 = 2 * p.school2 ∧
  p.school2 = p.school3 + 40 ∧
  p.school3 = 200

/-- The theorem stating the total number of students who shook the mayor's hand --/
theorem total_students_shaking_hands (p : SchoolParticipation) 
  (h : debateConditions p) : p.school1 + p.school2 + p.school3 = 920 := by
  sorry

end total_students_shaking_hands_l800_80030


namespace series_sum_equals_20_over_3_l800_80097

/-- The sum of the series (7n+2)/k^n from n=1 to infinity -/
noncomputable def series_sum (k : ℝ) : ℝ := ∑' n, (7 * n + 2) / k^n

/-- Theorem stating that if k > 1 and the series sum equals 20/3, then k = 2.9 -/
theorem series_sum_equals_20_over_3 (k : ℝ) (h1 : k > 1) (h2 : series_sum k = 20/3) : k = 2.9 := by
  sorry

end series_sum_equals_20_over_3_l800_80097


namespace value_of_expression_l800_80000

theorem value_of_expression (a : ℝ) (h : a^2 + 2*a + 1 = 0) : 2*a^2 + 4*a - 3 = -5 := by
  sorry

end value_of_expression_l800_80000


namespace negation_equivalence_l800_80083

theorem negation_equivalence :
  (¬ ∃ a : ℝ, a < 0 ∧ a + 4 / a ≤ -4) ↔ (∀ a : ℝ, a < 0 → a + 4 / a > -4) := by
  sorry

end negation_equivalence_l800_80083


namespace new_bill_total_l800_80031

/-- Calculates the new bill total after substitutions and additional charges -/
def calculate_new_bill (original_order : ℝ) 
                       (tomato_old : ℝ) (tomato_new : ℝ)
                       (lettuce_old : ℝ) (lettuce_new : ℝ)
                       (celery_old : ℝ) (celery_new : ℝ)
                       (delivery_tip : ℝ) : ℝ :=
  original_order + (tomato_new - tomato_old) + (lettuce_new - lettuce_old) + 
  (celery_new - celery_old) + delivery_tip

/-- Theorem stating that the new bill total is $35.00 -/
theorem new_bill_total : 
  calculate_new_bill 25 0.99 2.20 1.00 1.75 1.96 2.00 8.00 = 35 := by
  sorry

end new_bill_total_l800_80031


namespace six_digit_numbers_count_l800_80092

/-- The number of ways to choose 2 items from 4 items -/
def choose_4_2 : ℕ := 6

/-- The number of ways to choose 1 item from 2 items -/
def choose_2_1 : ℕ := 2

/-- The number of ways to arrange 3 items -/
def arrange_3_3 : ℕ := 6

/-- The number of ways to choose 2 positions from 4 positions -/
def insert_2_in_4 : ℕ := 6

/-- The total number of valid six-digit numbers -/
def total_numbers : ℕ := choose_4_2 * choose_2_1 * arrange_3_3 * insert_2_in_4

theorem six_digit_numbers_count : total_numbers = 432 := by
  sorry

end six_digit_numbers_count_l800_80092


namespace no_simultaneous_integer_fractions_l800_80080

theorem no_simultaneous_integer_fractions :
  ¬ ∃ (n : ℤ), (∃ (k m : ℤ), (n - 6 : ℚ) / 15 = k ∧ (n - 5 : ℚ) / 24 = m) :=
by sorry

end no_simultaneous_integer_fractions_l800_80080


namespace circle_equation_l800_80025

-- Define the circle C
def circle_C : Set (ℝ × ℝ) := {p : ℝ × ℝ | ∃ (h : ℝ), (p.1 - h)^2 + p.2^2 = (h - 1)^2 + 1^2}

-- Define points A and B
def point_A : ℝ × ℝ := (5, 2)
def point_B : ℝ × ℝ := (-1, 4)

-- Theorem statement
theorem circle_equation :
  (∀ p : ℝ × ℝ, p ∈ circle_C ↔ (p.1 - 1)^2 + p.2^2 = 20) ∧
  point_A ∈ circle_C ∧
  point_B ∈ circle_C ∧
  (∃ h : ℝ, ∀ p : ℝ × ℝ, p ∈ circle_C → p.2 = 0 → p.1 = h) :=
sorry

end circle_equation_l800_80025


namespace sheela_monthly_income_l800_80018

theorem sheela_monthly_income (deposit : ℝ) (percentage : ℝ) (monthly_income : ℝ) :
  deposit = 4500 →
  percentage = 28 →
  deposit = percentage / 100 * monthly_income →
  monthly_income = 16071.43 := by
  sorry

end sheela_monthly_income_l800_80018


namespace fraction_simplification_l800_80095

theorem fraction_simplification :
  (3/6 + 4/5) / (5/12 + 1/4) = 39/20 := by sorry

end fraction_simplification_l800_80095


namespace calculation_proof_l800_80035

theorem calculation_proof : (35 / (8 + 3 - 5) - 2) * 4 = 46 / 3 := by
  sorry

end calculation_proof_l800_80035


namespace geometric_progression_problem_l800_80072

theorem geometric_progression_problem (a b c : ℝ) : 
  a > 0 ∧ b > 0 ∧ c > 0 →  -- Positive real numbers
  (∃ r : ℝ, b = a * r ∧ c = b * r) →  -- Geometric progression
  a * b * c = 64 →  -- Product is 64
  (a + b + c) / 3 = 14 / 3 →  -- Arithmetic mean is 14/3
  ((a = 2 ∧ b = 4 ∧ c = 8) ∨ (a = 8 ∧ b = 4 ∧ c = 2)) :=
by sorry

end geometric_progression_problem_l800_80072


namespace george_candy_count_l800_80026

/-- The number of bags of candy -/
def num_bags : ℕ := 8

/-- The number of candy pieces in each bag -/
def pieces_per_bag : ℕ := 81

/-- The total number of candy pieces -/
def total_pieces : ℕ := num_bags * pieces_per_bag

theorem george_candy_count : total_pieces = 648 := by
  sorry

end george_candy_count_l800_80026


namespace nh4cl_molecular_weight_l800_80091

/-- The molecular weight of NH4Cl in grams per mole -/
def molecular_weight_NH4Cl : ℝ := 53

/-- The number of moles given in the problem -/
def moles : ℝ := 8

/-- The total weight of the given moles of NH4Cl in grams -/
def total_weight : ℝ := 424

/-- Theorem: The molecular weight of NH4Cl is 53 grams/mole -/
theorem nh4cl_molecular_weight :
  molecular_weight_NH4Cl = total_weight / moles :=
by sorry

end nh4cl_molecular_weight_l800_80091


namespace sqrt_x_squared_plus_two_is_quadratic_radical_l800_80038

-- Define what it means for an expression to be a quadratic radical
def is_quadratic_radical (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, ∃ y : ℝ, f x = y ∧ y ≥ 0

-- Theorem statement
theorem sqrt_x_squared_plus_two_is_quadratic_radical :
  is_quadratic_radical (λ x : ℝ => Real.sqrt (x^2 + 2)) :=
by
  sorry

end sqrt_x_squared_plus_two_is_quadratic_radical_l800_80038


namespace hilt_water_fountain_trips_l800_80068

/-- The number of times Mrs. Hilt will go to the water fountain -/
def water_fountain_trips (distance_to_fountain : ℕ) (total_distance : ℕ) : ℕ :=
  total_distance / (2 * distance_to_fountain)

/-- Theorem: Mrs. Hilt will go to the water fountain 2 times -/
theorem hilt_water_fountain_trips :
  water_fountain_trips 30 120 = 2 := by
  sorry

end hilt_water_fountain_trips_l800_80068


namespace propositions_truth_l800_80050

theorem propositions_truth :
  (∀ x : ℝ, x^2 - x + 1 > 0) ∧
  (∃ x₀ : ℝ, x₀ > 0 ∧ Real.log (1 / x₀) > -x₀ + 1) ∧
  (¬ ∃ x₀ : ℝ, x₀ > 0 ∧ Real.log x₀ > x₀ - 1) ∧
  (¬ ∀ x : ℝ, x > 0 → (1/2)^x > Real.log x / Real.log (1/2)) :=
by sorry

end propositions_truth_l800_80050


namespace smallest_n_greater_than_20_l800_80078

/-- g(n) is the sum of the digits of 1/(6^n) to the right of the decimal point -/
def g (n : ℕ+) : ℕ :=
  sorry

theorem smallest_n_greater_than_20 :
  (∀ k : ℕ+, k < 4 → g k ≤ 20) ∧ g 4 > 20 :=
sorry

end smallest_n_greater_than_20_l800_80078


namespace probability_at_least_two_defective_l800_80001

/-- The probability of selecting at least 2 defective items from a batch of products -/
theorem probability_at_least_two_defective (total : Nat) (good : Nat) (defective : Nat) 
  (selected : Nat) (h1 : total = good + defective) (h2 : total = 10) (h3 : good = 6) 
  (h4 : defective = 4) (h5 : selected = 3) : 
  (Nat.choose defective 2 * Nat.choose good 1 + Nat.choose defective 3) / 
  Nat.choose total selected = 1 / 3 := by
  sorry

end probability_at_least_two_defective_l800_80001


namespace square_region_perimeter_l800_80061

theorem square_region_perimeter (area : ℝ) (num_squares : ℕ) (rows : ℕ) (cols : ℕ) :
  area = 392 →
  num_squares = 8 →
  rows = 2 →
  cols = 4 →
  let side_length := Real.sqrt (area / num_squares)
  let perimeter := 2 * (rows * side_length + cols * side_length)
  perimeter = 126 := by
  sorry

end square_region_perimeter_l800_80061


namespace max_blocks_in_box_l800_80060

/-- Represents the dimensions of a rectangular box -/
structure BoxDimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Represents the dimensions of a block -/
structure BlockDimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Calculates the maximum number of blocks that can fit in a box -/
def maxBlocks (box : BoxDimensions) (block : BlockDimensions) : ℕ :=
  sorry

/-- The theorem stating the maximum number of blocks that can fit in the given box -/
theorem max_blocks_in_box :
  let box := BoxDimensions.mk 4 3 2
  let block := BlockDimensions.mk 3 1 1
  maxBlocks box block = 6 :=
sorry

end max_blocks_in_box_l800_80060


namespace equation_represents_line_and_hyperbola_l800_80011

-- Define the equation
def equation (x y : ℝ) : Prop := y^6 - 6*x^6 = 3*y^2 - 8

-- Define what it means for the equation to represent a line
def represents_line (eq : (ℝ → ℝ → Prop)) : Prop :=
  ∃ a b : ℝ, ∀ x y : ℝ, eq x y → y = a*x + b

-- Define what it means for the equation to represent a hyperbola
def represents_hyperbola (eq : (ℝ → ℝ → Prop)) : Prop :=
  ∃ a b c d e f : ℝ, a ≠ 0 ∧ b ≠ 0 ∧ a*b < 0 ∧
    ∀ x y : ℝ, eq x y → a*x^2 + b*y^2 + c*x*y + d*x + e*y + f = 0

-- Theorem statement
theorem equation_represents_line_and_hyperbola :
  represents_line equation ∧ represents_hyperbola equation :=
sorry

end equation_represents_line_and_hyperbola_l800_80011


namespace tetrahedron_acute_angles_l800_80058

/-- A tetrahedron with vertices S, A, B, and C -/
structure Tetrahedron where
  S : Point
  A : Point
  B : Point
  C : Point

/-- The dihedral angle between two faces of a tetrahedron -/
def dihedralAngle (t : Tetrahedron) (face1 face2 : Fin 4) : ℝ := sorry

/-- The planar angle at a vertex of a face in a tetrahedron -/
def planarAngle (t : Tetrahedron) (face : Fin 4) (vertex : Fin 3) : ℝ := sorry

/-- A predicate stating that an angle is acute -/
def isAcute (angle : ℝ) : Prop := angle > 0 ∧ angle < Real.pi / 2

theorem tetrahedron_acute_angles (t : Tetrahedron) :
  (∀ face1 face2, isAcute (dihedralAngle t face1 face2)) →
  (∀ face vertex, isAcute (planarAngle t face vertex)) := by
  sorry

end tetrahedron_acute_angles_l800_80058


namespace calculate_expression_l800_80023

theorem calculate_expression : 500 * 997 * 0.0997 * (10^2) = 5 * 997^2 := by
  sorry

end calculate_expression_l800_80023


namespace arithmetic_sequence_property_l800_80019

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

theorem arithmetic_sequence_property (a : ℕ → ℝ) 
  (h_arithmetic : arithmetic_sequence a)
  (h_sum : a 2 + a 4 + a 6 + a 8 + a 10 = 80) :
  a 7 - (1/2) * a 8 = 8 := by
sorry

end arithmetic_sequence_property_l800_80019


namespace book_purchases_l800_80073

/-- The number of people who purchased only book A -/
def v : ℕ := sorry

/-- The number of people who purchased only book B -/
def x : ℕ := sorry

/-- The number of people who purchased book B (both only and with book A) -/
def y : ℕ := sorry

/-- The number of people who purchased both books A and B -/
def both : ℕ := 500

theorem book_purchases : 
  (y = x + both) ∧ 
  (v = 2 * y) ∧ 
  (both = 2 * x) →
  v = 1500 := by sorry

end book_purchases_l800_80073


namespace impossible_transformation_l800_80043

/-- Represents the color and position of a token -/
inductive Token
  | Red : Token
  | BlueEven : Token
  | BlueOdd : Token

/-- Converts a token to its numeric representation -/
def tokenValue : Token → Int
  | Token.Red => 0
  | Token.BlueEven => 1
  | Token.BlueOdd => -1

/-- Represents the state of the line as a list of tokens -/
def Line := List Token

/-- Calculates the sum of the numeric representations of tokens in a line -/
def lineSum (l : Line) : Int :=
  l.map tokenValue |>.sum

/-- Represents a valid operation on the line -/
inductive Operation
  | Insert : Token → Token → Operation
  | Remove : Token → Token → Operation

/-- Applies an operation to a line -/
def applyOperation (l : Line) (op : Operation) : Line :=
  match op with
  | Operation.Insert t1 t2 => sorry
  | Operation.Remove t1 t2 => sorry

/-- Theorem: It's impossible to transform the initial state to the desired final state -/
theorem impossible_transformation : ∀ (ops : List Operation),
  let initial : Line := [Token.Red, Token.BlueEven]
  let final : Line := [Token.BlueOdd, Token.Red]
  (lineSum initial = lineSum (ops.foldl applyOperation initial)) ∧
  (ops.foldl applyOperation initial ≠ final) := by
  sorry

end impossible_transformation_l800_80043


namespace valid_words_count_l800_80010

def alphabet_size : ℕ := 25
def max_word_length : ℕ := 5

def total_words (n : ℕ) (k : ℕ) : ℕ :=
  (n^1 + n^2 + n^3 + n^4 + n^5)

def words_without_specific_letter (n : ℕ) (k : ℕ) : ℕ :=
  ((n-1)^1 + (n-1)^2 + (n-1)^3 + (n-1)^4 + (n-1)^5)

theorem valid_words_count :
  total_words alphabet_size max_word_length - words_without_specific_letter alphabet_size max_word_length = 1678698 :=
by sorry

end valid_words_count_l800_80010


namespace geometric_series_sum_l800_80090

theorem geometric_series_sum (a r : ℚ) (n : ℕ) (h : r ≠ 1) :
  let series_sum := a * (1 - r^n) / (1 - r)
  let a := (1 : ℚ) / 4
  let r := -(1 : ℚ) / 4
  let n := 5
  series_sum = 205 / 1024 := by sorry

end geometric_series_sum_l800_80090


namespace sector_max_area_l800_80040

/-- Given a sector with perimeter 40, its maximum area is 100 -/
theorem sector_max_area (r l : ℝ) (h_perimeter : 2 * r + l = 40) :
  (1 / 2) * l * r ≤ 100 := by
  sorry

end sector_max_area_l800_80040


namespace cos_sin_identity_l800_80062

open Real

theorem cos_sin_identity : 
  cos (89 * π / 180) * cos (π / 180) + sin (91 * π / 180) * sin (181 * π / 180) = 0 := by
  sorry

end cos_sin_identity_l800_80062


namespace probability_theorem_l800_80049

/-- The number of roots of unity for z^1997 - 1 = 0 --/
def n : ℕ := 1997

/-- The set of complex roots of z^1997 - 1 = 0 --/
def roots : Set ℂ := {z : ℂ | z^n = 1}

/-- The condition that needs to be satisfied --/
def condition (v w : ℂ) : Prop := Real.sqrt (2 + Real.sqrt 3) ≤ Complex.abs (v + w)

/-- The number of pairs (v, w) satisfying the condition --/
def satisfying_pairs : ℕ := 332 * (n - 1)

/-- The total number of possible pairs (v, w) --/
def total_pairs : ℕ := n * (n - 1)

/-- The theorem to be proved --/
theorem probability_theorem :
  (satisfying_pairs : ℚ) / total_pairs = 83 / 499 := by sorry

end probability_theorem_l800_80049


namespace circle_diameter_ratio_l800_80084

theorem circle_diameter_ratio (R S : ℝ) (hR : R > 0) (hS : S > 0)
  (h_area : π * R^2 = 0.25 * (π * S^2)) :
  2 * R = 0.5 * (2 * S) := by
sorry

end circle_diameter_ratio_l800_80084


namespace sufficient_not_necessary_l800_80071

theorem sufficient_not_necessary 
  (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  (∃ x y, x + y > a + b ∧ x * y > a * b ∧ ¬(x > a ∧ y > b)) ∧ 
  (∀ x y, x > a ∧ y > b → x + y > a + b ∧ x * y > a * b) := by
sorry

end sufficient_not_necessary_l800_80071


namespace regular_polygon_2022_probability_l800_80002

/-- A regular polygon with 2022 sides -/
structure RegularPolygon2022 where
  area : ℝ
  sides : Nat
  is_regular : sides = 2022

/-- A point on the perimeter of a polygon -/
structure PerimeterPoint (P : RegularPolygon2022) where
  x : ℝ
  y : ℝ
  on_perimeter : True  -- This is a placeholder for the actual condition

/-- The distance between two points -/
def distance (A B : PerimeterPoint P) : ℝ := sorry

/-- The probability of an event -/
def probability (event : Prop) : ℝ := sorry

theorem regular_polygon_2022_probability 
  (P : RegularPolygon2022) 
  (h : P.area = 1) :
  probability (
    ∀ (A B : PerimeterPoint P), 
    distance A B ≥ Real.sqrt (2 / Real.pi)
  ) = 1/2 := by sorry

end regular_polygon_2022_probability_l800_80002


namespace right_triangle_area_l800_80044

/-- The area of a right triangle with hypotenuse 10√2 cm and one angle 45° is 50 cm² -/
theorem right_triangle_area (h : ℝ) (α : ℝ) (A : ℝ) : 
  h = 10 * Real.sqrt 2 →  -- hypotenuse is 10√2 cm
  α = 45 * π / 180 →      -- one angle is 45°
  A = h^2 / 4 →           -- area formula for 45-45-90 triangle
  A = 50 := by
  sorry


end right_triangle_area_l800_80044


namespace age_problem_l800_80056

theorem age_problem (a b c : ℕ) : 
  a ≠ b ∧ b ≠ c ∧ a ≠ c →
  a > 0 ∧ b > 0 ∧ c > 0 →
  ∃ k : ℤ, (b - 1) / (a - 1) = k ∧ (b + 1) / (a + 1) = k + 1 →
  ∃ m : ℤ, (c - 1) / (b - 1) = m ∧ (c + 1) / (b + 1) = m + 1 →
  a + b + c ≤ 150 →
  a = 2 ∧ b = 7 ∧ c = 49 :=
by sorry

end age_problem_l800_80056


namespace candy_ratio_l800_80003

theorem candy_ratio : ∀ (red yellow blue : ℕ),
  red = 40 →
  yellow = 3 * red - 20 →
  red + blue = 90 →
  blue * 2 = yellow :=
by sorry

end candy_ratio_l800_80003


namespace greatest_third_side_proof_l800_80093

/-- The greatest integer length of the third side of a triangle with two sides of 7 cm and 15 cm -/
def greatest_third_side : ℕ := 21

/-- Triangle inequality theorem for our specific case -/
axiom triangle_inequality (a b c : ℝ) : 
  (a = 7 ∧ b = 15) → (c < a + b ∧ c > |a - b|)

theorem greatest_third_side_proof : 
  ∀ c : ℝ, (c < 22 ∧ c > 8) → c ≤ greatest_third_side := by sorry

end greatest_third_side_proof_l800_80093


namespace grid_black_probability_l800_80064

/-- Represents a 4x4 grid where each cell can be either black or white -/
def Grid := Fin 4 → Fin 4 → Bool

/-- The probability of a single cell being black initially -/
def initial_black_prob : ℚ := 1/2

/-- Rotates the grid 90 degrees clockwise -/
def rotate (g : Grid) : Grid := sorry

/-- Applies the repainting rule after rotation -/
def repaint (g : Grid) : Grid := sorry

/-- The probability that the entire grid becomes black after rotation and repainting -/
def prob_all_black_after_process : ℚ := sorry

/-- Theorem stating the probability of the grid becoming entirely black -/
theorem grid_black_probability : 
  prob_all_black_after_process = 1 / 65536 := by sorry

end grid_black_probability_l800_80064


namespace quadratic_inequality_no_solution_l800_80047

theorem quadratic_inequality_no_solution (m : ℝ) (h : m ≤ 1) :
  ¬∃ x : ℝ, x^2 + 2*x + 2 - m < 0 :=
by sorry

end quadratic_inequality_no_solution_l800_80047


namespace minute_hand_rotation_1h50m_l800_80033

/-- Represents the rotation of a clock's minute hand in degrees -/
def minute_hand_rotation (hours : ℕ) (minutes : ℕ) : ℤ :=
  -(hours * 360 + (minutes * 360) / 60)

/-- Theorem stating that for 1 hour and 50 minutes, the minute hand rotates -660 degrees -/
theorem minute_hand_rotation_1h50m : 
  minute_hand_rotation 1 50 = -660 := by
  sorry

end minute_hand_rotation_1h50m_l800_80033


namespace unique_satisfying_function_l800_80077

open Real

/-- A function f: ℝ₊ → ℝ₊ satisfying the given conditions -/
def SatisfyingFunction (f : ℝ → ℝ) : Prop :=
  (∀ x > 0, f x > 0) ∧ 
  (∀ x > 0, f x < 2*x - x / (1 + x^(3/2))) ∧
  (∀ x > 0, f (f x) = (5/2) * f x - x)

/-- The theorem stating that the only function satisfying the conditions is f(x) = x/2 -/
theorem unique_satisfying_function :
  ∀ f : ℝ → ℝ, SatisfyingFunction f → (∀ x > 0, f x = x/2) :=
by sorry

end unique_satisfying_function_l800_80077


namespace min_value_sum_reciprocals_l800_80065

/-- Given a line 2ax + by - 2 = 0 where a > 0 and b > 0, and the line passes through the point (1, 2),
    the minimum value of 1/a + 1/b is 4. -/
theorem min_value_sum_reciprocals (a b : ℝ) : 
  a > 0 → b > 0 → 2*a + b*2 = 2 → (∀ x y : ℝ, x > 0 → y > 0 → 2*x + y*2 = 2 → 1/a + 1/b ≤ 1/x + 1/y) → 
  1/a + 1/b = 4 := by sorry

end min_value_sum_reciprocals_l800_80065


namespace diophantine_equation_solution_l800_80063

theorem diophantine_equation_solution (x y z : ℤ) :
  x ≠ 0 → y ≠ 0 → z ≠ 0 → x + y + z ≠ 0 →
  (1 : ℚ) / x + (1 : ℚ) / y + (1 : ℚ) / z = (1 : ℚ) / (x + y + z) →
  (z = -x - y) ∨ (y = -x - z) ∨ (x = -y - z) :=
by sorry

end diophantine_equation_solution_l800_80063


namespace log_base_1024_integer_count_l800_80041

theorem log_base_1024_integer_count : 
  ∃! (S : Finset ℕ), 
    (∀ b ∈ S, b > 0 ∧ ∃ n : ℕ, n > 0 ∧ b ^ n = 1024) ∧ 
    (∀ b : ℕ, b > 0 → (∃ n : ℕ, n > 0 ∧ b ^ n = 1024) → b ∈ S) ∧
    S.card = 4 :=
by sorry

end log_base_1024_integer_count_l800_80041


namespace money_ratio_problem_l800_80076

theorem money_ratio_problem (ram_money gopal_money krishan_money : ℕ) :
  ram_money = 588 →
  krishan_money = 3468 →
  gopal_money * 17 = krishan_money * 7 →
  ∃ (a b : ℕ), a * gopal_money = b * ram_money ∧ a = 3 ∧ b = 7 :=
by sorry

end money_ratio_problem_l800_80076


namespace geometric_sequence_common_ratio_l800_80013

/-- A geometric sequence -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = q * a n

theorem geometric_sequence_common_ratio
  (a : ℕ → ℝ)
  (h_geom : GeometricSequence a)
  (h_a3 : a 3 = 1)
  (h_a5 : a 5 = 4) :
  ∃ q : ℝ, (q = 2 ∨ q = -2) ∧ ∀ n : ℕ, a (n + 1) = q * a n :=
sorry

end geometric_sequence_common_ratio_l800_80013


namespace residue_neg_1234_mod_31_l800_80059

theorem residue_neg_1234_mod_31 : Int.mod (-1234) 31 = 6 := by
  sorry

end residue_neg_1234_mod_31_l800_80059


namespace split_tree_sum_lower_bound_l800_80094

/-- Represents a tree where each node splits into two children that sum to the parent -/
inductive SplitTree : Nat → Type
  | leaf : SplitTree 1
  | node : (n : Nat) → (left right : Nat) → left + right = n → 
           SplitTree left → SplitTree right → SplitTree n

/-- The sum of all numbers in a SplitTree -/
def treeSum : {n : Nat} → SplitTree n → Nat
  | _, SplitTree.leaf => 1
  | n, SplitTree.node _ left right _ leftTree rightTree => 
      n + treeSum leftTree + treeSum rightTree

/-- Theorem: The sum of all numbers in a SplitTree starting with 2^n is at least n * 2^n -/
theorem split_tree_sum_lower_bound (n : Nat) (tree : SplitTree (2^n)) :
  treeSum tree ≥ n * 2^n := by
  sorry

end split_tree_sum_lower_bound_l800_80094


namespace complex_modulus_identity_l800_80087

theorem complex_modulus_identity 
  (z₁ z₂ z₃ z₄ : ℂ) 
  (h₁ : Complex.abs z₁ = 1) 
  (h₂ : Complex.abs z₂ = 1) 
  (h₃ : Complex.abs z₃ = 1) 
  (h₄ : Complex.abs z₄ = 1) : 
  Complex.abs (z₁ - z₂) ^ 2 * Complex.abs (z₃ - z₄) ^ 2 + 
  Complex.abs (z₁ + z₄) ^ 2 * Complex.abs (z₃ - z₂) ^ 2 = 
  Complex.abs (z₁ * (z₂ - z₃) + z₃ * (z₂ - z₁) + z₄ * (z₁ - z₃)) ^ 2 := by
  sorry

end complex_modulus_identity_l800_80087


namespace shirt_shoe_cost_multiple_l800_80098

/-- The multiple of the cost of the shirt that represents the cost of the shoes -/
def multiple_of_shirt_cost (total_cost shirt_cost shoe_cost : ℚ) : ℚ :=
  (shoe_cost - 9) / shirt_cost

theorem shirt_shoe_cost_multiple :
  let total_cost : ℚ := 300
  let shirt_cost : ℚ := 97
  let shoe_cost : ℚ := total_cost - shirt_cost
  shoe_cost = multiple_of_shirt_cost total_cost shirt_cost shoe_cost * shirt_cost + 9 →
  multiple_of_shirt_cost total_cost shirt_cost shoe_cost = 2 :=
by sorry

end shirt_shoe_cost_multiple_l800_80098


namespace john_duck_profit_l800_80099

/-- Calculates the profit from selling ducks given the following conditions:
  * number_of_ducks: The number of ducks bought and sold
  * cost_per_duck: The cost of each duck when buying
  * weight_per_duck: The weight of each duck in pounds
  * price_per_pound: The selling price per pound of duck
-/
def duck_profit (number_of_ducks : ℕ) (cost_per_duck : ℚ) (weight_per_duck : ℚ) (price_per_pound : ℚ) : ℚ :=
  let total_cost := number_of_ducks * cost_per_duck
  let revenue_per_duck := weight_per_duck * price_per_pound
  let total_revenue := number_of_ducks * revenue_per_duck
  total_revenue - total_cost

/-- Theorem stating that under the given conditions, the profit is $300 -/
theorem john_duck_profit :
  duck_profit 30 10 4 5 = 300 := by
  sorry

end john_duck_profit_l800_80099


namespace distance_to_grandmas_house_l800_80014

-- Define the car's efficiency in miles per gallon
def car_efficiency : ℝ := 20

-- Define the amount of gas needed to reach Grandma's house in gallons
def gas_needed : ℝ := 5

-- Theorem to prove the distance to Grandma's house
theorem distance_to_grandmas_house : car_efficiency * gas_needed = 100 := by
  sorry

end distance_to_grandmas_house_l800_80014


namespace function_properties_l800_80006

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

def is_increasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x < f y

def is_symmetric_about (f : ℝ → ℝ) (c : ℝ) : Prop :=
  ∀ x, f (c + x) = f (c - x)

theorem function_properties (f : ℝ → ℝ) 
  (h1 : is_even f)
  (h2 : ∀ x, f (x + 1) = -f x)
  (h3 : is_increasing_on f (-1) 0) :
  (∀ x, f (x + 2) = f x) ∧ 
  (is_symmetric_about f 1) ∧
  (f 2 = f 0) := by
  sorry

end function_properties_l800_80006


namespace original_number_is_nine_l800_80032

theorem original_number_is_nine (N : ℕ) : (N - 4) % 5 = 0 → N = 9 := by
  sorry

end original_number_is_nine_l800_80032


namespace circle_radius_is_three_l800_80020

/-- Given a circle where the product of three inches and its circumference
    is twice its area, prove that its radius is 3 inches. -/
theorem circle_radius_is_three (r : ℝ) (h : 3 * (2 * π * r) = 2 * (π * r^2)) : r = 3 := by
  sorry

end circle_radius_is_three_l800_80020


namespace log_function_k_range_l800_80021

theorem log_function_k_range (a : ℝ) (h_a : a > 0) :
  {k : ℝ | ∀ x > a, x > max a (k * a)} = {k : ℝ | -1 ≤ k ∧ k ≤ 1} := by
sorry

end log_function_k_range_l800_80021


namespace recurrence_relations_hold_l800_80066

def circle_radius : ℝ := 1

def perimeter_circumscribed (n : ℕ) : ℝ := sorry

def perimeter_inscribed (n : ℕ) : ℝ := sorry

theorem recurrence_relations_hold (n : ℕ) (h : n ≥ 3) :
  perimeter_circumscribed (2 * n) = (2 * perimeter_circumscribed n * perimeter_inscribed n) / (perimeter_circumscribed n + perimeter_inscribed n) ∧
  perimeter_inscribed (2 * n) = Real.sqrt (perimeter_inscribed n * perimeter_circumscribed (2 * n)) :=
sorry

end recurrence_relations_hold_l800_80066


namespace pure_imaginary_quotient_l800_80015

theorem pure_imaginary_quotient (a : ℝ) : 
  let z₁ : ℂ := a + 2*I
  let z₂ : ℂ := 3 - 4*I
  (∃ (b : ℝ), z₁ / z₂ = b*I) → a = 8/3 := by
  sorry

end pure_imaginary_quotient_l800_80015


namespace conference_hall_tables_l800_80046

theorem conference_hall_tables (chairs_per_table : ℕ) (chair_legs : ℕ) (table_legs : ℕ) (total_legs : ℕ) :
  chairs_per_table = 8 →
  chair_legs = 3 →
  table_legs = 5 →
  total_legs = 580 →
  ∃ (num_tables : ℕ), num_tables = 20 ∧ 
    chairs_per_table * num_tables * chair_legs + num_tables * table_legs = total_legs :=
by sorry

end conference_hall_tables_l800_80046


namespace parallel_vectors_angle_l800_80069

theorem parallel_vectors_angle (x : Real) : 
  let a : ℝ × ℝ := (Real.sin x, 3/4)
  let b : ℝ × ℝ := (1/3, (1/2) * Real.cos x)
  (∃ (k : ℝ), k ≠ 0 ∧ a = k • b) → 
  0 < x ∧ x < π/2 → 
  x = π/4 := by
  sorry

end parallel_vectors_angle_l800_80069


namespace lake_shore_distance_l800_80081

/-- Given two points A and B on the shore of a lake, and a point C chosen such that
    CA = 50 meters, CB = 30 meters, and ∠ACB = 120°, prove that the distance AB is 70 meters. -/
theorem lake_shore_distance (A B C : ℝ × ℝ) : 
  let CA := Real.sqrt ((A.1 - C.1)^2 + (A.2 - C.2)^2)
  let CB := Real.sqrt ((B.1 - C.1)^2 + (B.2 - C.2)^2)
  let AB := Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)
  let cos_ACB := ((A.1 - C.1) * (B.1 - C.1) + (A.2 - C.2) * (B.2 - C.2)) / (CA * CB)
  CA = 50 ∧ CB = 30 ∧ cos_ACB = -1/2 → AB = 70 := by
  sorry


end lake_shore_distance_l800_80081


namespace jack_sugar_usage_l800_80088

/-- Represents the amount of sugar Jack has initially -/
def initial_sugar : ℕ := 65

/-- Represents the amount of sugar Jack buys after usage -/
def sugar_bought : ℕ := 50

/-- Represents the final amount of sugar Jack has -/
def final_sugar : ℕ := 97

/-- Represents the amount of sugar Jack uses -/
def sugar_used : ℕ := 18

theorem jack_sugar_usage :
  initial_sugar - sugar_used + sugar_bought = final_sugar :=
by sorry

end jack_sugar_usage_l800_80088


namespace expression_evaluation_l800_80008

theorem expression_evaluation (c a b d : ℚ) 
  (h1 : d = a + 1)
  (h2 : a = b - 3)
  (h3 : b = c + 5)
  (h4 : c = 6)
  (h5 : d + 3 ≠ 0)
  (h6 : a + 2 ≠ 0)
  (h7 : b - 5 ≠ 0)
  (h8 : c + 7 ≠ 0) :
  ((d + 5) / (d + 3)) * ((a + 3) / (a + 2)) * ((b - 3) / (b - 5)) * ((c + 10) / (c + 7)) = 1232 / 585 := by
  sorry

end expression_evaluation_l800_80008


namespace f_expression_f_range_l800_80052

/-- A quadratic function f satisfying the given conditions -/
def f (x : ℝ) : ℝ := sorry

/-- The property that f(x+1) - f(x) = 2x -/
axiom f_diff (x : ℝ) : f (x + 1) - f x = 2 * x

/-- The property that f(0) = 1 -/
axiom f_zero : f 0 = 1

/-- Theorem: The analytical expression of f(x) -/
theorem f_expression (x : ℝ) : f x = x^2 - x + 1 := sorry

/-- Theorem: The range of f(x) when x ∈ [-1, 1] -/
theorem f_range : Set.Icc (3/4 : ℝ) 3 = { y | ∃ x ∈ Set.Icc (-1 : ℝ) 1, f x = y } := sorry

end f_expression_f_range_l800_80052


namespace officer_selection_theorem_l800_80017

def total_candidates : ℕ := 20
def officer_positions : ℕ := 6
def past_officers : ℕ := 5

theorem officer_selection_theorem :
  (Nat.choose past_officers 1 * Nat.choose (total_candidates - past_officers) (officer_positions - 1)) +
  (Nat.choose past_officers 2 * Nat.choose (total_candidates - past_officers) (officer_positions - 2)) +
  (Nat.choose past_officers 3 * Nat.choose (total_candidates - past_officers) (officer_positions - 3)) = 33215 := by
  sorry

end officer_selection_theorem_l800_80017


namespace water_filter_capacity_l800_80079

/-- The total capacity of a cylindrical water filter in liters. -/
def total_capacity : ℝ := 120

/-- The amount of water in the filter when it is partially filled, in liters. -/
def partial_amount : ℝ := 36

/-- The fraction of the filter that is filled when it contains the partial amount. -/
def partial_fraction : ℝ := 0.30

/-- Theorem stating that the total capacity of the water filter is 120 liters,
    given that it contains 36 liters when it is 30% full. -/
theorem water_filter_capacity :
  total_capacity * partial_fraction = partial_amount :=
by sorry

end water_filter_capacity_l800_80079


namespace diseased_corn_plants_l800_80045

theorem diseased_corn_plants (grid_size : Nat) (h : grid_size = 2015) :
  let center := grid_size / 2 + 1
  let days_to_corner := center - 1
  days_to_corner * 2 = 2014 :=
sorry

end diseased_corn_plants_l800_80045


namespace person_age_puzzle_l800_80051

theorem person_age_puzzle : ∃ (x : ℝ), x > 0 ∧ x = 4 * (x + 4) - 4 * (x - 4) + (1/2) * (x - 6) ∧ x = 58 := by
  sorry

end person_age_puzzle_l800_80051


namespace hamburger_meat_price_per_pound_l800_80024

/-- Given the following grocery items and their prices:
    - 2 pounds of hamburger meat (price unknown)
    - 1 pack of hamburger buns for $1.50
    - A head of lettuce for $1.00
    - A 1.5-pound tomato priced at $2.00 per pound
    - A jar of pickles that cost $2.50 with a $1.00 off coupon
    And given that Lauren paid with a $20 bill and got $6 change back,
    prove that the price per pound of hamburger meat is $3.50. -/
theorem hamburger_meat_price_per_pound
  (hamburger_meat_weight : ℝ)
  (buns_price : ℝ)
  (lettuce_price : ℝ)
  (tomato_weight : ℝ)
  (tomato_price_per_pound : ℝ)
  (pickles_price : ℝ)
  (pickles_discount : ℝ)
  (paid_amount : ℝ)
  (change_amount : ℝ)
  (h1 : hamburger_meat_weight = 2)
  (h2 : buns_price = 1.5)
  (h3 : lettuce_price = 1)
  (h4 : tomato_weight = 1.5)
  (h5 : tomato_price_per_pound = 2)
  (h6 : pickles_price = 2.5)
  (h7 : pickles_discount = 1)
  (h8 : paid_amount = 20)
  (h9 : change_amount = 6) :
  (paid_amount - change_amount - (buns_price + lettuce_price + tomato_weight * tomato_price_per_pound + pickles_price - pickles_discount)) / hamburger_meat_weight = 3.5 := by
  sorry

end hamburger_meat_price_per_pound_l800_80024
