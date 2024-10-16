import Mathlib

namespace NUMINAMATH_CALUDE_consecutive_numbers_sum_product_l1114_111496

/-- Given four consecutive natural numbers x-1, x, x+1, and x+2, 
    if the product of their sum and the sum of their squares 
    equals three times the sum of their cubes, then x = 5. -/
theorem consecutive_numbers_sum_product (x : ℕ) : 
  (x - 1 + x + (x + 1) + (x + 2)) * 
  ((x - 1)^2 + x^2 + (x + 1)^2 + (x + 2)^2) = 
  3 * ((x - 1)^3 + x^3 + (x + 1)^3 + (x + 2)^3) → 
  x = 5 := by
  sorry

#check consecutive_numbers_sum_product

end NUMINAMATH_CALUDE_consecutive_numbers_sum_product_l1114_111496


namespace NUMINAMATH_CALUDE_rectangular_field_ratio_l1114_111474

theorem rectangular_field_ratio (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a < b) : 
  (a + b = 3 * a) → 
  (a + b - Real.sqrt (a^2 + b^2) = b / 3) → 
  a / b = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_rectangular_field_ratio_l1114_111474


namespace NUMINAMATH_CALUDE_twelve_rolls_in_case_l1114_111454

/-- Calculates the number of rolls in a case of paper towels given the case price, individual roll price, and savings percentage. -/
def rolls_in_case (case_price : ℚ) (roll_price : ℚ) (savings_percent : ℚ) : ℚ :=
  case_price / (roll_price * (1 - savings_percent / 100))

/-- Theorem stating that there are 12 rolls in the case under the given conditions. -/
theorem twelve_rolls_in_case :
  rolls_in_case 9 1 25 = 12 := by
  sorry

end NUMINAMATH_CALUDE_twelve_rolls_in_case_l1114_111454


namespace NUMINAMATH_CALUDE_unique_prime_squared_plus_minus_six_prime_l1114_111449

theorem unique_prime_squared_plus_minus_six_prime :
  ∃! p : ℕ, Nat.Prime p ∧ Nat.Prime (p^2 - 6) ∧ Nat.Prime (p^2 + 6) :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_unique_prime_squared_plus_minus_six_prime_l1114_111449


namespace NUMINAMATH_CALUDE_sheet_length_l1114_111408

theorem sheet_length (width : ℝ) (side_margin : ℝ) (top_bottom_margin : ℝ) (typing_percentage : ℝ) :
  width = 20 →
  side_margin = 2 →
  top_bottom_margin = 3 →
  typing_percentage = 0.64 →
  ∃ length : ℝ,
    length = 30 ∧
    (width - 2 * side_margin) * (length - 2 * top_bottom_margin) = typing_percentage * width * length :=
by sorry

end NUMINAMATH_CALUDE_sheet_length_l1114_111408


namespace NUMINAMATH_CALUDE_quadratic_equation_from_means_l1114_111466

theorem quadratic_equation_from_means (a b : ℝ) : 
  (a + b) / 2 = 8 → 
  Real.sqrt (a * b) = 10 → 
  ∀ x, x^2 - 16*x + 100 = 0 ↔ (x = a ∨ x = b) := by
sorry

end NUMINAMATH_CALUDE_quadratic_equation_from_means_l1114_111466


namespace NUMINAMATH_CALUDE_empty_solution_set_iff_k_ge_one_l1114_111457

-- Define the function representing the left side of the inequality
def f (k x : ℝ) : ℝ := k * x^2 - 2 * |x - 1| + 3 * k

-- Define the property of having an empty solution set
def has_empty_solution_set (k : ℝ) : Prop :=
  ∀ x : ℝ, f k x ≥ 0

-- State the theorem
theorem empty_solution_set_iff_k_ge_one :
  ∀ k : ℝ, has_empty_solution_set k ↔ k ≥ 1 := by sorry

end NUMINAMATH_CALUDE_empty_solution_set_iff_k_ge_one_l1114_111457


namespace NUMINAMATH_CALUDE_min_max_abs_expressions_l1114_111447

theorem min_max_abs_expressions (x y : ℝ) :
  ∃ (x₀ y₀ : ℝ), max (|2 * x₀ + y₀|) (max (|x₀ - y₀|) (|1 + y₀|)) = (1/2 : ℝ) ∧
  ∀ (x y : ℝ), (1/2 : ℝ) ≤ max (|2 * x + y|) (max (|x - y|) (|1 + y|)) :=
by sorry

end NUMINAMATH_CALUDE_min_max_abs_expressions_l1114_111447


namespace NUMINAMATH_CALUDE_exam_score_calculation_l1114_111478

/-- Proves that the number of marks awarded for a correct answer is 4 in the given exam scenario -/
theorem exam_score_calculation (total_questions : ℕ) (correct_answers : ℕ) (total_score : ℕ) 
  (h1 : total_questions = 150)
  (h2 : correct_answers = 120)
  (h3 : total_score = 420)
  (h4 : ∀ x : ℕ, x * correct_answers - 2 * (total_questions - correct_answers) = total_score → x = 4) :
  ∃ x : ℕ, x * correct_answers - 2 * (total_questions - correct_answers) = total_score ∧ x = 4 :=
by sorry

end NUMINAMATH_CALUDE_exam_score_calculation_l1114_111478


namespace NUMINAMATH_CALUDE_regression_result_l1114_111400

/-- The regression equation -/
def regression_equation (x : ℝ) : ℝ := 4.75 * x + 2.57

/-- Theorem: For the given regression equation, when x = 28, y = 135.57 -/
theorem regression_result : regression_equation 28 = 135.57 := by
  sorry

end NUMINAMATH_CALUDE_regression_result_l1114_111400


namespace NUMINAMATH_CALUDE_gum_pieces_per_package_l1114_111425

theorem gum_pieces_per_package (total_packages : ℕ) (total_pieces : ℕ) : 
  total_packages = 9 → total_pieces = 135 → total_pieces / total_packages = 15 := by
  sorry

end NUMINAMATH_CALUDE_gum_pieces_per_package_l1114_111425


namespace NUMINAMATH_CALUDE_cover_cost_is_77_l1114_111458

/-- Represents the cost of printing a book in kopecks -/
def book_cost (cover_cost : ℕ) (page_cost : ℕ) (num_pages : ℕ) : ℕ :=
  (cover_cost * 100 + page_cost * num_pages + 99) / 100 * 100

/-- The problem statement -/
theorem cover_cost_is_77 : 
  ∃ (cover_cost page_cost : ℕ),
    (∀ n, book_cost cover_cost page_cost n = ((cover_cost * 100 + page_cost * n + 99) / 100) * 100) ∧
    book_cost cover_cost page_cost 104 = 134 * 100 ∧
    book_cost cover_cost page_cost 192 = 181 * 100 ∧
    cover_cost = 77 :=
by
  sorry

end NUMINAMATH_CALUDE_cover_cost_is_77_l1114_111458


namespace NUMINAMATH_CALUDE_hyperbola_asymptote_slope_l1114_111482

/-- The slope of the asymptotes for the hyperbola (x^2 / 144) - (y^2 / 81) = 1 is 3/4 -/
theorem hyperbola_asymptote_slope :
  let hyperbola := fun (x y : ℝ) => x^2 / 144 - y^2 / 81 = 1
  ∃ m : ℝ, m = 3/4 ∧ 
    ∀ (x y : ℝ), hyperbola x y → (y = m * x ∨ y = -m * x) := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_asymptote_slope_l1114_111482


namespace NUMINAMATH_CALUDE_salary_fraction_on_food_l1114_111435

theorem salary_fraction_on_food 
  (salary : ℝ) 
  (rent_fraction : ℝ) 
  (clothes_fraction : ℝ) 
  (amount_left : ℝ) 
  (h1 : salary = 160000)
  (h2 : rent_fraction = 1/10)
  (h3 : clothes_fraction = 3/5)
  (h4 : amount_left = 16000)
  (h5 : salary * rent_fraction + salary * clothes_fraction + amount_left + salary * food_fraction = salary) :
  food_fraction = 1/5 := by
sorry

end NUMINAMATH_CALUDE_salary_fraction_on_food_l1114_111435


namespace NUMINAMATH_CALUDE_cameron_typing_speed_l1114_111433

/-- The number of words Cameron could type per minute before breaking his arm -/
def words_before : ℕ := 10

/-- The difference in words typed in 5 minutes before and after breaking his arm -/
def word_difference : ℕ := 10

/-- The number of words Cameron could type per minute after breaking his arm -/
def words_after : ℕ := 8

/-- Proof that Cameron could type 8 words per minute after breaking his arm -/
theorem cameron_typing_speed :
  words_after = 8 ∧
  words_before * 5 - words_after * 5 = word_difference :=
by sorry

end NUMINAMATH_CALUDE_cameron_typing_speed_l1114_111433


namespace NUMINAMATH_CALUDE_beakers_with_no_metal_ions_l1114_111453

theorem beakers_with_no_metal_ions (total_beakers : Nat) (copper_beakers : Nat) (silver_beakers : Nat)
  (drops_a_per_beaker : Nat) (drops_b_per_beaker : Nat) (total_drops_a : Nat) (total_drops_b : Nat) :
  total_beakers = 50 →
  copper_beakers = 10 →
  silver_beakers = 5 →
  drops_a_per_beaker = 3 →
  drops_b_per_beaker = 4 →
  total_drops_a = 106 →
  total_drops_b = 80 →
  total_beakers - copper_beakers - silver_beakers = 15 :=
by sorry

end NUMINAMATH_CALUDE_beakers_with_no_metal_ions_l1114_111453


namespace NUMINAMATH_CALUDE_brian_watching_time_l1114_111468

/-- The total time Brian spends watching animal videos -/
def total_watching_time (cat_video_length : ℕ) : ℕ :=
  let dog_video_length := 2 * cat_video_length
  let gorilla_video_length := 2 * (cat_video_length + dog_video_length)
  cat_video_length + dog_video_length + gorilla_video_length

/-- Theorem stating that Brian spends 36 minutes watching animal videos -/
theorem brian_watching_time : total_watching_time 4 = 36 := by
  sorry

end NUMINAMATH_CALUDE_brian_watching_time_l1114_111468


namespace NUMINAMATH_CALUDE_fraction_to_decimal_l1114_111471

theorem fraction_to_decimal : (7 : ℚ) / 16 = (4375 : ℚ) / 10000 := by sorry

end NUMINAMATH_CALUDE_fraction_to_decimal_l1114_111471


namespace NUMINAMATH_CALUDE_sum_of_three_equal_numbers_l1114_111418

theorem sum_of_three_equal_numbers (a b c d e : ℝ) : 
  (a + b + c + d + e) / 5 = 20 → 
  a = 12 → 
  b = 24 → 
  c = d → 
  d = e → 
  c + d + e = 64 := by
sorry

end NUMINAMATH_CALUDE_sum_of_three_equal_numbers_l1114_111418


namespace NUMINAMATH_CALUDE_f_positive_all_reals_f_positive_interval_l1114_111455

/-- The quadratic function f(x) = x^2 + 2(a-2)x + 4 -/
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + 2*(a-2)*x + 4

/-- Theorem 1: f(x) > 0 for all x ∈ ℝ if and only if 0 < a < 4 -/
theorem f_positive_all_reals (a : ℝ) :
  (∀ x : ℝ, f a x > 0) ↔ (0 < a ∧ a < 4) :=
sorry

/-- Theorem 2: f(x) > 0 for x ∈ [-3, 1] if and only if a ∈ (-1/2, 4) -/
theorem f_positive_interval (a : ℝ) :
  (∀ x : ℝ, x ∈ Set.Icc (-3) 1 → f a x > 0) ↔ (a > -1/2 ∧ a < 4) :=
sorry

end NUMINAMATH_CALUDE_f_positive_all_reals_f_positive_interval_l1114_111455


namespace NUMINAMATH_CALUDE_certain_number_equation_l1114_111498

theorem certain_number_equation : ∃ x : ℚ, (40 * x + (12 + 8) * 3 / 5 = 1212) ∧ x = 30 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_equation_l1114_111498


namespace NUMINAMATH_CALUDE_angelinas_journey_l1114_111475

/-- Angelina's journey with varying speeds -/
theorem angelinas_journey (distance_home_grocery : ℝ) (distance_grocery_gym : ℝ) (time_difference : ℝ) :
  distance_home_grocery = 840 →
  distance_grocery_gym = 480 →
  time_difference = 40 →
  ∃ (speed_home_grocery : ℝ),
    speed_home_grocery > 0 ∧
    distance_home_grocery / speed_home_grocery - distance_grocery_gym / (2 * speed_home_grocery) = time_difference ∧
    2 * speed_home_grocery = 30 :=
by sorry

end NUMINAMATH_CALUDE_angelinas_journey_l1114_111475


namespace NUMINAMATH_CALUDE_drum_size_correct_l1114_111486

/-- Represents the size of the drum in gallons -/
def D : ℝ := 54.99

/-- Represents the amount of 100% antifreeze used in gallons -/
def pure_antifreeze : ℝ := 6.11

/-- Represents the percentage of antifreeze in the final mixture -/
def final_mixture_percent : ℝ := 0.20

/-- Represents the percentage of antifreeze in the initial diluted mixture -/
def initial_diluted_percent : ℝ := 0.10

/-- Theorem stating that the given conditions result in the correct drum size -/
theorem drum_size_correct : 
  pure_antifreeze + (D - pure_antifreeze) * initial_diluted_percent = D * final_mixture_percent := by
  sorry

#check drum_size_correct

end NUMINAMATH_CALUDE_drum_size_correct_l1114_111486


namespace NUMINAMATH_CALUDE_sum_of_fractions_equals_target_l1114_111443

theorem sum_of_fractions_equals_target : 
  (1/3 : ℚ) + (1/2 : ℚ) + (-5/6 : ℚ) + (1/5 : ℚ) + (1/4 : ℚ) + (-9/20 : ℚ) + (2/15 : ℚ) = 
  (13333333333333333 : ℚ) / (100000000000000000 : ℚ) := by sorry

end NUMINAMATH_CALUDE_sum_of_fractions_equals_target_l1114_111443


namespace NUMINAMATH_CALUDE_simplify_expression_l1114_111499

theorem simplify_expression (a b : ℝ) (h1 : a ≠ -b) (h2 : a ≠ 2*b) (h3 : a ≠ b) :
  (a + 2*b) / (a + b) - (a - b) / (a - 2*b) / ((a^2 - b^2) / (a^2 - 4*a*b + 4*b^2)) = 4*b / (a + b) :=
by sorry

end NUMINAMATH_CALUDE_simplify_expression_l1114_111499


namespace NUMINAMATH_CALUDE_quadratic_inequality_l1114_111480

-- Define the quadratic function
def f (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

-- Define the solution set of f ≥ 0
def solution_set (a b c : ℝ) : Set ℝ := {x | x ≤ -3 ∨ x ≥ 4}

-- Define the theorem
theorem quadratic_inequality (a b c : ℝ) 
  (h : ∀ x, x ∈ solution_set a b c ↔ f a b c x ≥ 0) :
  a > 0 ∧ 
  (∀ x, cx^2 - bx + a < 0 ↔ x < -1/4 ∨ x > 1/3) :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l1114_111480


namespace NUMINAMATH_CALUDE_expansion_unique_solution_l1114_111461

/-- The number of terms in the expansion of (a+b+c+d+e+1)^n that include all five variables
    a, b, c, d, e, each to some positive power. -/
def numTerms (n : ℕ) : ℕ := Nat.choose n 5

/-- The proposition that 16 is the unique positive integer n such that the expansion of
    (a+b+c+d+e+1)^n contains exactly 2002 terms with all five variables a, b, c, d, e
    each to some positive power. -/
theorem expansion_unique_solution : 
  ∃! (n : ℕ), n > 0 ∧ numTerms n = 2002 ∧ n = 16 := by sorry

end NUMINAMATH_CALUDE_expansion_unique_solution_l1114_111461


namespace NUMINAMATH_CALUDE_xyz_value_l1114_111434

theorem xyz_value (x y z : ℝ) (h1 : x^2 * y * z^3 = 7^3) (h2 : x * y^2 = 7^9) : 
  x * y * z = 7^4 := by
sorry

end NUMINAMATH_CALUDE_xyz_value_l1114_111434


namespace NUMINAMATH_CALUDE_cylinder_with_same_volume_and_surface_area_l1114_111421

theorem cylinder_with_same_volume_and_surface_area 
  (r m : ℝ) (hr : r > 0) (hm : m > 0) :
  ∃ (r₁ m₁ : ℝ),
    r₁ > 0 ∧ m₁ > 0 ∧
    r₁ ≠ r ∧ m₁ ≠ m ∧
    r₁ = (Real.sqrt (r^2 + 4*m*r) - r) / 2 ∧
    m₁ = 4*m*r^2 / (Real.sqrt (r^2 + 4*m*r) - r)^2 ∧
    π * r^2 * m = π * r₁^2 * m₁ ∧
    2 * π * r^2 + 2 * π * r * m = 2 * π * r₁^2 + 2 * π * r₁ * m₁ :=
by sorry

#check cylinder_with_same_volume_and_surface_area

end NUMINAMATH_CALUDE_cylinder_with_same_volume_and_surface_area_l1114_111421


namespace NUMINAMATH_CALUDE_axis_of_symmetry_l1114_111460

def is_symmetric_about (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x, f (a + x) = f (a - x)

theorem axis_of_symmetry (f : ℝ → ℝ) (h : ∀ x, f x = f (6 - x)) :
  is_symmetric_about f 3 := by sorry

end NUMINAMATH_CALUDE_axis_of_symmetry_l1114_111460


namespace NUMINAMATH_CALUDE_correct_games_attended_l1114_111422

/-- Calculates the number of football games attended given the planned games and missed games. -/
def games_attended (this_month : ℕ) (last_month : ℕ) (missed : ℕ) : ℕ :=
  this_month + last_month - missed

/-- Theorem stating that the number of games attended is correct given the problem conditions. -/
theorem correct_games_attended :
  games_attended 11 17 16 = 12 := by
  sorry

end NUMINAMATH_CALUDE_correct_games_attended_l1114_111422


namespace NUMINAMATH_CALUDE_initial_men_count_prove_initial_men_count_l1114_111416

/-- The number of men initially working on a project -/
def initial_men : ℕ := 12

/-- The number of hours worked per day by the initial group -/
def initial_hours_per_day : ℕ := 8

/-- The number of days worked by the initial group -/
def initial_days : ℕ := 10

/-- The number of men in the second group -/
def second_group_men : ℕ := 5

/-- The number of hours worked per day by the second group -/
def second_hours_per_day : ℕ := 16

/-- The number of days worked by the second group -/
def second_days : ℕ := 12

theorem initial_men_count : 
  initial_men * initial_hours_per_day * initial_days = 
  second_group_men * second_hours_per_day * second_days :=
by sorry

/-- The main theorem proving the number of men initially working -/
theorem prove_initial_men_count : initial_men = 12 :=
by sorry

end NUMINAMATH_CALUDE_initial_men_count_prove_initial_men_count_l1114_111416


namespace NUMINAMATH_CALUDE_expression_simplification_l1114_111497

theorem expression_simplification (m : ℝ) (h : m = Real.sqrt 2) :
  (1 + 1 / (m - 2)) / ((m^2 - m) / (m - 2)) = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l1114_111497


namespace NUMINAMATH_CALUDE_shoe_price_ratio_l1114_111465

/-- Given a shoe with a marked price, a discount of 1/4 off, and a cost that is 2/3 of the actual selling price, 
    the ratio of the cost to the marked price is 1/2. -/
theorem shoe_price_ratio (marked_price : ℝ) (marked_price_pos : 0 < marked_price) : 
  let selling_price := (3/4) * marked_price
  let cost := (2/3) * selling_price
  cost / marked_price = 1/2 := by
sorry

end NUMINAMATH_CALUDE_shoe_price_ratio_l1114_111465


namespace NUMINAMATH_CALUDE_decimal_addition_l1114_111492

theorem decimal_addition : (0.9 : ℝ) + 0.09 = 0.99 := by
  sorry

end NUMINAMATH_CALUDE_decimal_addition_l1114_111492


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l1114_111493

/-- Given a hyperbola mx^2 + 5y^2 = 5m with eccentricity e = 2, prove that m = -15 -/
theorem hyperbola_eccentricity (m : ℝ) : 
  (∃ (x y : ℝ), m*x^2 + 5*y^2 = 5*m) → -- Hyperbola equation
  (∃ (e : ℝ), e = 2 ∧ e^2 = 1 - m/5) → -- Eccentricity definition
  m = -15 := by
sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l1114_111493


namespace NUMINAMATH_CALUDE_max_n_for_L_perfect_square_l1114_111402

/-- Definition of L(n): the number of permutations of {1,2,...,n} with exactly one landmark point -/
def L (n : ℕ) : ℕ := 4 * (2^(n-2) - 1)

/-- Theorem stating that 3 is the maximum n ≥ 3 for which L(n) is a perfect square -/
theorem max_n_for_L_perfect_square :
  ∀ n : ℕ, n ≥ 3 → (∃ k : ℕ, L n = k^2) → n ≤ 3 :=
sorry

end NUMINAMATH_CALUDE_max_n_for_L_perfect_square_l1114_111402


namespace NUMINAMATH_CALUDE_function_max_value_l1114_111485

-- Define the function f
def f (x m : ℝ) : ℝ := -x^3 + 3*x^2 + 9*x + m

-- State the theorem
theorem function_max_value (m : ℝ) :
  (∀ x ∈ Set.Icc (-2 : ℝ) 2, f x m ≤ 20) ∧
  (∃ x ∈ Set.Icc (-2 : ℝ) 2, f x m = 20) →
  m = -2 ∧ ∀ x ∈ Set.Icc (-2 : ℝ) 2, f x (-2) ≤ 20 :=
by sorry

end NUMINAMATH_CALUDE_function_max_value_l1114_111485


namespace NUMINAMATH_CALUDE_all_chameleons_green_chameleon_color_convergence_l1114_111470

/-- Represents the colors of chameleons --/
inductive Color
| Yellow
| Red
| Green

/-- Represents the state of chameleons on the island --/
structure ChameleonState where
  yellow : Nat
  red : Nat
  green : Nat

/-- The initial state of chameleons --/
def initialState : ChameleonState :=
  { yellow := 7, red := 10, green := 17 }

/-- The total number of chameleons --/
def totalChameleons : Nat := 34

/-- Function to model the color change when two different colored chameleons meet --/
def colorChange (c1 c2 : Color) : Color :=
  match c1, c2 with
  | Color.Yellow, Color.Red => Color.Green
  | Color.Red, Color.Yellow => Color.Green
  | Color.Yellow, Color.Green => Color.Red
  | Color.Green, Color.Yellow => Color.Red
  | Color.Red, Color.Green => Color.Yellow
  | Color.Green, Color.Red => Color.Yellow
  | _, _ => c1  -- No change if same color

/-- Theorem stating that all chameleons will eventually be green --/
theorem all_chameleons_green (finalState : ChameleonState) : 
  (finalState.yellow + finalState.red + finalState.green = totalChameleons) →
  (finalState.yellow = 0 ∧ finalState.red = 0 ∧ finalState.green = totalChameleons) :=
sorry

/-- Main theorem to prove --/
theorem chameleon_color_convergence :
  ∃ (finalState : ChameleonState),
    (finalState.yellow + finalState.red + finalState.green = totalChameleons) ∧
    (finalState.yellow = 0 ∧ finalState.red = 0 ∧ finalState.green = totalChameleons) :=
sorry

end NUMINAMATH_CALUDE_all_chameleons_green_chameleon_color_convergence_l1114_111470


namespace NUMINAMATH_CALUDE_square_sum_implies_abs_sum_l1114_111405

theorem square_sum_implies_abs_sum (a b : ℝ) :
  a^2 + b^2 > 1 → |a| + |b| > 1 := by sorry

end NUMINAMATH_CALUDE_square_sum_implies_abs_sum_l1114_111405


namespace NUMINAMATH_CALUDE_hyperbola_triangle_area_l1114_111479

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 - y^2/8 = 1

-- Define the foci F₁ and F₂
def F₁ : ℝ × ℝ := sorry
def F₂ : ℝ × ℝ := sorry

-- Define a point P on the hyperbola
def P : ℝ × ℝ := sorry

-- Define the distances |PF₁| and |PF₂|
def PF₁ : ℝ := sorry
def PF₂ : ℝ := sorry

-- Theorem statement
theorem hyperbola_triangle_area :
  hyperbola P.1 P.2 →
  PF₁ / PF₂ = 3 / 4 →
  (1/2) * ‖F₁ - F₂‖ * ‖P - (F₁ + F₂)/2‖ = 8 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_triangle_area_l1114_111479


namespace NUMINAMATH_CALUDE_olympic_volunteer_allocation_l1114_111412

theorem olympic_volunteer_allocation :
  let n : ℕ := 5  -- number of volunteers
  let k : ℕ := 4  -- number of projects
  let allocations : ℕ := (n.choose 2) * (k.factorial)
  allocations = 240 :=
by sorry

end NUMINAMATH_CALUDE_olympic_volunteer_allocation_l1114_111412


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l1114_111411

-- Define the arithmetic sequence
def arithmetic_sequence (n : ℕ) : ℤ := 3 - 2 * n

-- Define the sum of the first k terms
def sum_of_terms (k : ℕ) : ℤ := k * (arithmetic_sequence 1 + arithmetic_sequence k) / 2

-- Theorem statement
theorem arithmetic_sequence_problem :
  (arithmetic_sequence 1 = 1) ∧
  (arithmetic_sequence 3 = -3) ∧
  (∃ k : ℕ, sum_of_terms k = -35 ∧ k = 7) :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l1114_111411


namespace NUMINAMATH_CALUDE_committee_probability_l1114_111464

/-- The number of members in the Literature club -/
def total_members : ℕ := 24

/-- The number of boys in the Literature club -/
def num_boys : ℕ := 12

/-- The number of girls in the Literature club -/
def num_girls : ℕ := 12

/-- The size of the committee to be chosen -/
def committee_size : ℕ := 5

/-- The probability of choosing a committee with at least 2 boys and at least 2 girls -/
theorem committee_probability : 
  (Nat.choose total_members committee_size - 
   (2 * Nat.choose num_boys committee_size + 
    Nat.choose num_boys 1 * Nat.choose num_girls 4 + 
    Nat.choose num_girls 1 * Nat.choose num_boys 4)) / 
   Nat.choose total_members committee_size = 121 / 177 := by
  sorry

end NUMINAMATH_CALUDE_committee_probability_l1114_111464


namespace NUMINAMATH_CALUDE_population_growth_over_three_years_l1114_111456

/-- Represents the demographic rates for a given year -/
structure YearlyRates where
  birth_rate : ℝ
  death_rate : ℝ
  in_migration : ℝ
  out_migration : ℝ

/-- Calculates the net growth rate for a given year -/
def net_growth_rate (rates : YearlyRates) : ℝ :=
  rates.birth_rate + rates.in_migration - rates.death_rate - rates.out_migration

/-- Theorem stating the net percentage increase in population over three years -/
theorem population_growth_over_three_years 
  (year1 : YearlyRates)
  (year2 : YearlyRates)
  (year3 : YearlyRates)
  (h1 : year1 = { birth_rate := 0.025, death_rate := 0.01, in_migration := 0.03, out_migration := 0.02 })
  (h2 : year2 = { birth_rate := 0.02, death_rate := 0.015, in_migration := 0.04, out_migration := 0.035 })
  (h3 : year3 = { birth_rate := 0.022, death_rate := 0.008, in_migration := 0.025, out_migration := 0.01 })
  : ∃ (ε : ℝ), abs ((1 + net_growth_rate year1) * (1 + net_growth_rate year2) * (1 + net_growth_rate year3) - 1 - 0.065675) < ε ∧ ε > 0 := by
  sorry

end NUMINAMATH_CALUDE_population_growth_over_three_years_l1114_111456


namespace NUMINAMATH_CALUDE_decimal_to_percentage_l1114_111448

theorem decimal_to_percentage (x : ℝ) (h : x = 0.02) : x * 100 = 2 := by
  sorry

end NUMINAMATH_CALUDE_decimal_to_percentage_l1114_111448


namespace NUMINAMATH_CALUDE_johnny_money_left_l1114_111427

def johnny_savings (september october november : ℕ) : ℕ := september + october + november

theorem johnny_money_left (september october november spending : ℕ) 
  (h1 : september = 30)
  (h2 : october = 49)
  (h3 : november = 46)
  (h4 : spending = 58) :
  johnny_savings september october november - spending = 67 := by
  sorry

end NUMINAMATH_CALUDE_johnny_money_left_l1114_111427


namespace NUMINAMATH_CALUDE_quarter_circles_limit_l1114_111409

/-- The limit of the sum of quarter-circle lengths approaches the original circumference -/
theorem quarter_circles_limit (C : ℝ) (h : C > 0) :
  ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, |2 * n * (C / (2 * n)) - C| < ε :=
sorry

end NUMINAMATH_CALUDE_quarter_circles_limit_l1114_111409


namespace NUMINAMATH_CALUDE_min_value_S_l1114_111445

theorem min_value_S (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h_sum : x^2 + y^2 + z^2 = 1) :
  ∀ (x' y' z' : ℝ), x' > 0 → y' > 0 → z' > 0 → x'^2 + y'^2 + z'^2 = 1 →
    (1 + z) / (2 * x * y * z) ≤ (1 + z') / (2 * x' * y' * z') →
    (1 + z) / (2 * x * y * z) ≥ 4 :=
sorry

end NUMINAMATH_CALUDE_min_value_S_l1114_111445


namespace NUMINAMATH_CALUDE_collinear_points_reciprocal_sum_l1114_111452

theorem collinear_points_reciprocal_sum (a b : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  (∃ (t : ℝ), (2 + t * (a - 2), 2 + t * (-2)) = (0, b)) →
  1 / a + 1 / b = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_collinear_points_reciprocal_sum_l1114_111452


namespace NUMINAMATH_CALUDE_rook_placement_exists_l1114_111472

/-- Represents a chessboard with rook placements -/
structure Chessboard (n : ℕ) :=
  (placements : Fin n → Fin n)
  (colors : Fin n → Fin n → Fin (n^2 / 2))

/-- Predicate to check if rook placements are valid -/
def valid_placements (n : ℕ) (board : Chessboard n) : Prop :=
  (∀ i j : Fin n, i ≠ j → board.placements i ≠ board.placements j) ∧
  (∀ i j : Fin n, i ≠ j → board.colors i (board.placements i) ≠ board.colors j (board.placements j))

/-- Predicate to check if the coloring is valid -/
def valid_coloring (n : ℕ) (board : Chessboard n) : Prop :=
  ∀ c : Fin (n^2 / 2), ∃! (i j k l : Fin n), 
    (i, j) ≠ (k, l) ∧ board.colors i j = c ∧ board.colors k l = c

/-- Main theorem -/
theorem rook_placement_exists (n : ℕ) (h_even : Even n) (h_gt_2 : n > 2) :
  ∃ (board : Chessboard n), valid_placements n board ∧ valid_coloring n board :=
sorry

end NUMINAMATH_CALUDE_rook_placement_exists_l1114_111472


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l1114_111431

theorem sufficient_not_necessary_condition (m : ℝ) :
  (∀ x > 1, x^2 - m*x + 1 > 0) ↔ m ≤ 2 :=
sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l1114_111431


namespace NUMINAMATH_CALUDE_square_area_ratio_l1114_111404

/-- The ratio of the area of a square with side length 2y to the area of a square with side length 8y is 1/16 -/
theorem square_area_ratio (y : ℝ) (y_pos : y > 0) : 
  (2 * y)^2 / (8 * y)^2 = 1 / 16 := by
  sorry

end NUMINAMATH_CALUDE_square_area_ratio_l1114_111404


namespace NUMINAMATH_CALUDE_parallel_lines_b_value_l1114_111440

/-- Given two lines in the xy-plane, this theorem proves that if they are parallel,
    then the value of b must be 6. -/
theorem parallel_lines_b_value (b : ℝ) :
  (∀ x y, 3 * y - 3 * b = 9 * x) →
  (∀ x y, y - 2 = (b - 3) * x) →
  (∃ k : ℝ, ∀ x y, 3 * y - 3 * b = 9 * x ↔ y - 2 = k * (x - 0)) →
  b = 6 := by
  sorry

end NUMINAMATH_CALUDE_parallel_lines_b_value_l1114_111440


namespace NUMINAMATH_CALUDE_cylinder_surface_area_l1114_111419

theorem cylinder_surface_area (h w : ℝ) (h_pos : h > 0) (w_pos : w > 0) 
  (hπ : h = 6 * Real.pi) (wπ : w = 4 * Real.pi) :
  ∃ (r : ℝ), (r = 3 ∨ r = 2) ∧ 
    (2 * Real.pi * r * h + 2 * Real.pi * r^2 = 24 * Real.pi^2 + 18 * Real.pi ∨
     2 * Real.pi * r * h + 2 * Real.pi * r^2 = 24 * Real.pi^2 + 8 * Real.pi) :=
by sorry

end NUMINAMATH_CALUDE_cylinder_surface_area_l1114_111419


namespace NUMINAMATH_CALUDE_jellybeans_left_specific_l1114_111444

/-- Calculates the number of jellybeans left in a jar after some children eat them. -/
def jellybeans_left (total : ℕ) (normal_class_size : ℕ) (absent : ℕ) (absent_eat : ℕ)
  (group1_size : ℕ) (group1_eat : ℕ) (group2_size : ℕ) (group2_eat : ℕ) : ℕ :=
  total - (group1_size * group1_eat + group2_size * group2_eat)

/-- Theorem stating the number of jellybeans left in the jar under specific conditions. -/
theorem jellybeans_left_specific : 
  jellybeans_left 250 24 2 7 12 5 10 4 = 150 := by
  sorry

end NUMINAMATH_CALUDE_jellybeans_left_specific_l1114_111444


namespace NUMINAMATH_CALUDE_tom_catch_equals_16_l1114_111481

def melanie_catch : ℕ := 8

def tom_catch_multiplier : ℕ := 2

def tom_catch : ℕ := tom_catch_multiplier * melanie_catch

theorem tom_catch_equals_16 : tom_catch = 16 := by
  sorry

end NUMINAMATH_CALUDE_tom_catch_equals_16_l1114_111481


namespace NUMINAMATH_CALUDE_solve_for_m_l1114_111441

/-- Given that (x, y) = (2, -3) is a solution of the equation mx + 3y = 1, prove that m = 5 -/
theorem solve_for_m (x y m : ℝ) (h1 : x = 2) (h2 : y = -3) (h3 : m * x + 3 * y = 1) : m = 5 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_m_l1114_111441


namespace NUMINAMATH_CALUDE_paint_calculation_l1114_111401

/-- The amount of white paint needed in ounces -/
def white_paint : ℕ := 20

/-- The amount of green paint needed in ounces -/
def green_paint : ℕ := 15

/-- The amount of brown paint needed in ounces -/
def brown_paint : ℕ := 34

/-- The total amount of paint needed in ounces -/
def total_paint : ℕ := white_paint + green_paint + brown_paint

theorem paint_calculation : total_paint = 69 := by
  sorry

end NUMINAMATH_CALUDE_paint_calculation_l1114_111401


namespace NUMINAMATH_CALUDE_calculate_Y_l1114_111476

theorem calculate_Y : ∀ A B Y : ℚ,
  A = 3081 / 4 →
  B = A * 2 →
  Y = A - B →
  Y = -770.25 := by
sorry

end NUMINAMATH_CALUDE_calculate_Y_l1114_111476


namespace NUMINAMATH_CALUDE_mink_coat_problem_l1114_111451

theorem mink_coat_problem (initial_minks : ℕ) (babies_per_mink : ℕ) (coats_made : ℕ) : 
  initial_minks = 30 →
  babies_per_mink = 6 →
  coats_made = 7 →
  (initial_minks * (babies_per_mink + 1) / 2) / coats_made = 15 := by
sorry

end NUMINAMATH_CALUDE_mink_coat_problem_l1114_111451


namespace NUMINAMATH_CALUDE_diophantine_equation_solution_l1114_111462

theorem diophantine_equation_solution :
  ∀ (p q r k : ℕ),
    p.Prime ∧ q.Prime ∧ r.Prime ∧ k > 0 →
    p^2 + q^2 + 49*r^2 = 9*k^2 - 101 →
    ((p = 3 ∧ q = 5 ∧ r = 3 ∧ k = 8) ∨ (p = 5 ∧ q = 3 ∧ r = 3 ∧ k = 8)) :=
by sorry

end NUMINAMATH_CALUDE_diophantine_equation_solution_l1114_111462


namespace NUMINAMATH_CALUDE_rectangle_side_ratio_l1114_111490

theorem rectangle_side_ratio (a b : ℝ) (h : (a - b) / (a + b) = 1 / 3) : (a / b) ^ 2 = 4 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_side_ratio_l1114_111490


namespace NUMINAMATH_CALUDE_experimental_fertilizer_height_is_135_l1114_111429

-- Define the heights of plants with different fertilizers
def control_height : ℝ := 36

def bone_meal_height : ℝ := 1.25 * control_height

def cow_manure_height : ℝ := 2 * bone_meal_height

def experimental_fertilizer_height : ℝ := 1.5 * cow_manure_height

-- Theorem to prove
theorem experimental_fertilizer_height_is_135 :
  experimental_fertilizer_height = 135 := by
  sorry

end NUMINAMATH_CALUDE_experimental_fertilizer_height_is_135_l1114_111429


namespace NUMINAMATH_CALUDE_paving_job_units_l1114_111477

theorem paving_job_units (worker1_rate worker2_rate reduced_efficiency total_time : ℝ) 
  (h1 : worker1_rate = 1 / 8)
  (h2 : worker2_rate = 1 / 12)
  (h3 : reduced_efficiency = 8)
  (h4 : total_time = 6) :
  let combined_rate := worker1_rate + worker2_rate - reduced_efficiency / total_time
  total_time * combined_rate = 192 := by
sorry

end NUMINAMATH_CALUDE_paving_job_units_l1114_111477


namespace NUMINAMATH_CALUDE_shortest_distance_curve_to_line_l1114_111463

/-- The shortest distance from a point on the curve y = 2ln x to the line 2x - y + 3 = 0 is √5 -/
theorem shortest_distance_curve_to_line :
  let curve := fun x : ℝ => 2 * Real.log x
  let line := fun x y : ℝ => 2 * x - y + 3 = 0
  ∃ d : ℝ, d = Real.sqrt 5 ∧
    ∀ x y : ℝ, curve x = y →
      d ≤ (|2 * x - y + 3| / Real.sqrt (2^2 + (-1)^2)) ∧
      ∃ x₀ y₀ : ℝ, curve x₀ = y₀ ∧
        d = (|2 * x₀ - y₀ + 3| / Real.sqrt (2^2 + (-1)^2)) :=
by sorry

end NUMINAMATH_CALUDE_shortest_distance_curve_to_line_l1114_111463


namespace NUMINAMATH_CALUDE_perpendicular_line_equation_l1114_111423

/-- A line passing through (-1, 2) and perpendicular to y = 2/3x has the equation 3x + 2y - 1 = 0 -/
theorem perpendicular_line_equation :
  let l : Set (ℝ × ℝ) := {(x, y) | 3 * x + 2 * y - 1 = 0}
  let point : ℝ × ℝ := (-1, 2)
  let perpendicular_slope : ℝ := 2 / 3
  (point ∈ l) ∧
  (∀ (x y : ℝ), (x, y) ∈ l → (3 : ℝ) * perpendicular_slope = -1) :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_line_equation_l1114_111423


namespace NUMINAMATH_CALUDE_longest_side_of_obtuse_consecutive_integer_triangle_l1114_111420

-- Define a triangle with consecutive integer side lengths
def ConsecutiveIntegerSidedTriangle (a b c : ℕ) : Prop :=
  (b = a + 1) ∧ (c = b + 1) ∧ (a ≥ 1)

-- Define an obtuse triangle
def ObtuseTriangle (a b c : ℕ) : Prop :=
  (a^2 + b^2 < c^2) ∨ (a^2 + c^2 < b^2) ∨ (b^2 + c^2 < a^2)

theorem longest_side_of_obtuse_consecutive_integer_triangle :
  ∀ a b c : ℕ,
  ConsecutiveIntegerSidedTriangle a b c →
  ObtuseTriangle a b c →
  c = 4 :=
sorry

end NUMINAMATH_CALUDE_longest_side_of_obtuse_consecutive_integer_triangle_l1114_111420


namespace NUMINAMATH_CALUDE_a_profit_share_l1114_111450

/-- Calculates the share of profit for an investor given the investments and time periods -/
def calculate_profit_share (investment_a : ℕ) (investment_b : ℕ) (months_a : ℕ) (months_b : ℕ) (total_profit : ℕ) : ℚ :=
  let investment_time_a := investment_a * months_a
  let investment_time_b := investment_b * months_b
  let total_investment_time := investment_time_a + investment_time_b
  let ratio_a := investment_time_a / total_investment_time
  ratio_a * total_profit

theorem a_profit_share :
  calculate_profit_share 300 200 12 6 100 = 75 := by
  sorry

end NUMINAMATH_CALUDE_a_profit_share_l1114_111450


namespace NUMINAMATH_CALUDE_simplify_expression_l1114_111436

theorem simplify_expression (a : ℝ) : (3 * a^2)^2 = 9 * a^4 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1114_111436


namespace NUMINAMATH_CALUDE_f_monotone_increasing_l1114_111442

/-- The function f(x) = x^3 - 1/x is monotonically increasing for x > 0 -/
theorem f_monotone_increasing :
  ∀ x₁ x₂ : ℝ, 0 < x₁ → 0 < x₂ → x₁ < x₂ → x₁^3 - 1/x₁ < x₂^3 - 1/x₂ := by
  sorry

end NUMINAMATH_CALUDE_f_monotone_increasing_l1114_111442


namespace NUMINAMATH_CALUDE_isosceles_triangle_area_l1114_111484

/-- An isosceles triangle with two sides of length 5 and a median to the base of length 4 has an area of 12 -/
theorem isosceles_triangle_area (a b c : ℝ) (m : ℝ) (h_isosceles : a = b) (h_side : a = 5) (h_median : m = 4) : 
  (1/2 : ℝ) * c * m = 12 := by
  sorry

end NUMINAMATH_CALUDE_isosceles_triangle_area_l1114_111484


namespace NUMINAMATH_CALUDE_classroom_seats_count_l1114_111446

/-- Represents a rectangular classroom with seats arranged in rows and columns. -/
structure Classroom where
  seats_left : ℕ  -- Number of seats to the left of the chosen seat
  seats_right : ℕ  -- Number of seats to the right of the chosen seat
  rows_front : ℕ  -- Number of rows in front of the chosen seat
  rows_back : ℕ  -- Number of rows behind the chosen seat

/-- Calculates the total number of seats in the classroom. -/
def total_seats (c : Classroom) : ℕ :=
  (c.seats_left + c.seats_right + 1) * (c.rows_front + c.rows_back + 1)

/-- Theorem stating that a classroom with the given properties has 399 seats. -/
theorem classroom_seats_count :
  ∀ (c : Classroom),
    c.seats_left = 6 →
    c.seats_right = 12 →
    c.rows_front = 7 →
    c.rows_back = 13 →
    total_seats c = 399 := by
  sorry

end NUMINAMATH_CALUDE_classroom_seats_count_l1114_111446


namespace NUMINAMATH_CALUDE_ellipse_equation_l1114_111495

-- Define the ellipse
structure Ellipse where
  foci : (ℝ × ℝ) × (ℝ × ℝ)
  distance_sum : ℝ

-- Define the standard form of an ellipse equation
def standard_ellipse_equation (a b : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1

-- Theorem statement
theorem ellipse_equation (e : Ellipse) (x y : ℝ) : 
  e.foci = ((-4, 0), (4, 0)) →
  e.distance_sum = 10 →
  standard_ellipse_equation 25 9 x y :=
sorry

end NUMINAMATH_CALUDE_ellipse_equation_l1114_111495


namespace NUMINAMATH_CALUDE_system_equivalent_to_line_l1114_111437

/-- The system of equations -/
def system (x y : ℝ) : Prop :=
  x - 2*y = 1 ∧ x^3 - 6*x*y - 8*y^3 = 1

/-- The line representing the solution -/
def solution_line (x y : ℝ) : Prop :=
  y = (x - 1) / 2

/-- Theorem stating that the system is equivalent to the solution line -/
theorem system_equivalent_to_line :
  ∀ x y : ℝ, system x y ↔ solution_line x y :=
sorry

end NUMINAMATH_CALUDE_system_equivalent_to_line_l1114_111437


namespace NUMINAMATH_CALUDE_five_digit_sum_l1114_111417

theorem five_digit_sum (x : ℕ) : 
  (1 + 3 + 4 + 6 + x) * (5 * 4 * 3 * 2 * 1) = 2640 → x = 8 := by
  sorry

end NUMINAMATH_CALUDE_five_digit_sum_l1114_111417


namespace NUMINAMATH_CALUDE_crow_tree_problem_l1114_111414

theorem crow_tree_problem (x y : ℕ) : 
  (3 * y + 5 = x) → (5 * (y - 1) = x) → (x = 20 ∧ y = 5) := by
  sorry

end NUMINAMATH_CALUDE_crow_tree_problem_l1114_111414


namespace NUMINAMATH_CALUDE_total_hamburger_combinations_l1114_111487

/-- The number of different condiments available -/
def num_condiments : ℕ := 10

/-- The number of choices for meat patties -/
def meat_patty_choices : ℕ := 3

/-- The number of choices for buns -/
def bun_choices : ℕ := 2

/-- Theorem: The total number of different hamburger combinations -/
theorem total_hamburger_combinations : 
  (2 ^ num_condiments) * meat_patty_choices * bun_choices = 6144 := by
  sorry

end NUMINAMATH_CALUDE_total_hamburger_combinations_l1114_111487


namespace NUMINAMATH_CALUDE_solution_set_inequality_l1114_111432

theorem solution_set_inequality (x : ℝ) :
  {x : ℝ | x * (9 - x) > 0} = Set.Ioo 0 9 := by sorry

end NUMINAMATH_CALUDE_solution_set_inequality_l1114_111432


namespace NUMINAMATH_CALUDE_race_time_proof_l1114_111488

/-- 
Given a race with three participants Patrick, Manu, and Amy:
- Patrick's race time is 60 seconds
- Manu's race time is 12 seconds more than Patrick's
- Amy's speed is twice Manu's speed
Prove that Amy's race time is 36 seconds
-/
theorem race_time_proof (patrick_time manu_time amy_time : ℝ) : 
  patrick_time = 60 →
  manu_time = patrick_time + 12 →
  amy_time * 2 = manu_time →
  amy_time = 36 := by
sorry

end NUMINAMATH_CALUDE_race_time_proof_l1114_111488


namespace NUMINAMATH_CALUDE_two_digit_number_problem_l1114_111473

theorem two_digit_number_problem :
  ∃! n : ℕ, 10 ≤ n ∧ n < 100 ∧
  ∃ k : ℤ, 3 * n - 4 = 10 * k ∧
  60 < 4 * n - 15 ∧ 4 * n - 15 < 100 ∧
  n = 28 :=
sorry

end NUMINAMATH_CALUDE_two_digit_number_problem_l1114_111473


namespace NUMINAMATH_CALUDE_divisibility_theorem_l1114_111403

/-- The set of natural numbers m for which 3^m - 1 is divisible by 2^m -/
def S : Set ℕ := {m : ℕ | ∃ k : ℕ, 3^m - 1 = k * 2^m}

/-- The set of natural numbers m for which 31^m - 1 is divisible by 2^m -/
def T : Set ℕ := {m : ℕ | ∃ k : ℕ, 31^m - 1 = k * 2^m}

theorem divisibility_theorem :
  S = {1, 2, 4} ∧ T = {1, 2, 4, 6, 8} := by sorry

end NUMINAMATH_CALUDE_divisibility_theorem_l1114_111403


namespace NUMINAMATH_CALUDE_M_on_inscribed_square_l1114_111406

/-- Right triangle with squares and inscribed square -/
structure RightTriangleWithSquares where
  -- Right triangle ABC
  a : ℝ
  b : ℝ
  c : ℝ
  -- Pythagorean theorem
  pythagoras : a^2 + b^2 = c^2
  -- Positivity of sides
  a_pos : a > 0
  b_pos : b > 0
  c_pos : c > 0
  -- Inscribed square side length
  x : ℝ
  x_def : x = (a * b) / (a + b)
  -- Point M
  M : ℝ × ℝ

/-- The theorem stating that M lies on the perimeter of the inscribed square -/
theorem M_on_inscribed_square (t : RightTriangleWithSquares) :
  t.M.1 = t.x ∧ 0 ≤ t.M.2 ∧ t.M.2 ≤ t.x := by
  sorry

end NUMINAMATH_CALUDE_M_on_inscribed_square_l1114_111406


namespace NUMINAMATH_CALUDE_solution_equality_implies_k_equals_one_l1114_111426

theorem solution_equality_implies_k_equals_one :
  ∀ x k : ℝ,
  (2 * x - 1 = 3 * x - 2) →
  (4 - (k * x + 2) / 3 = 3 * k - (2 - 2 * x) / 4) →
  k = 1 := by
sorry

end NUMINAMATH_CALUDE_solution_equality_implies_k_equals_one_l1114_111426


namespace NUMINAMATH_CALUDE_expression_evaluation_l1114_111410

theorem expression_evaluation :
  let x : ℚ := -1/2
  let y : ℚ := 1
  (2*x + 3*y)^2 - (2*x + y)*(2*x - y) = 4 := by sorry

end NUMINAMATH_CALUDE_expression_evaluation_l1114_111410


namespace NUMINAMATH_CALUDE_even_function_properties_l1114_111489

-- Define the function f
def f (x m : ℝ) : ℝ := (x - 1) * (x + m)

-- State the theorem
theorem even_function_properties (m : ℝ) :
  (∀ x, f x m = f (-x) m) →
  (m = 1 ∧ (∀ x, f x m = 0 ↔ x = 1 ∨ x = -1)) :=
by sorry

end NUMINAMATH_CALUDE_even_function_properties_l1114_111489


namespace NUMINAMATH_CALUDE_inverse_of_B_cubed_l1114_111438

theorem inverse_of_B_cubed (B : Matrix (Fin 2) (Fin 2) ℝ) :
  B⁻¹ = !![3, -2; 1, 0] →
  (B^3)⁻¹ = !![15, -14; 7, -6] := by
sorry

end NUMINAMATH_CALUDE_inverse_of_B_cubed_l1114_111438


namespace NUMINAMATH_CALUDE_cos_sum_inequality_l1114_111459

theorem cos_sum_inequality (x y : Real) :
  x ∈ Set.Icc 0 (Real.pi / 2) →
  y ∈ Set.Icc 0 Real.pi →
  Real.cos (x + y) ≤ Real.cos x * Real.cos y :=
by sorry

end NUMINAMATH_CALUDE_cos_sum_inequality_l1114_111459


namespace NUMINAMATH_CALUDE_hyperbola_parabola_coincidence_l1114_111424

-- Define the hyperbola
def hyperbola (a : ℝ) (x y : ℝ) : Prop :=
  x^2 / a^2 - y^2 / 3 = 1

-- Define the parabola
def parabola (x y : ℝ) : Prop :=
  y^2 = 8 * x

-- Define the focus of the parabola
def parabola_focus : ℝ × ℝ := (2, 0)

-- Define the right vertex of the hyperbola
def hyperbola_right_vertex (a : ℝ) : ℝ × ℝ := (a, 0)

theorem hyperbola_parabola_coincidence (a : ℝ) (h : a > 0) :
  hyperbola_right_vertex a = parabola_focus → a = 2 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_parabola_coincidence_l1114_111424


namespace NUMINAMATH_CALUDE_inequality_solution_set_l1114_111428

def solution_set : Set ℝ := {x | -1 < x ∧ x ≤ 2}

theorem inequality_solution_set :
  ∀ x : ℝ, (x - 2) / (x + 1) ≤ 0 ↔ x ∈ solution_set :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l1114_111428


namespace NUMINAMATH_CALUDE_min_value_of_expression_l1114_111413

theorem min_value_of_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  (1 / a + 2) * (1 / b + 2) ≥ 16 ∧
  ((1 / a + 2) * (1 / b + 2) = 16 ↔ a = 1/2 ∧ b = 1/2) :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l1114_111413


namespace NUMINAMATH_CALUDE_restaurant_bill_calculation_l1114_111467

def restaurant_bill (num_people_1 num_people_2 : ℕ) (cost_1 cost_2 service_charge : ℚ) 
  (discount_rate tip_rate : ℚ) : ℚ :=
  let meal_cost := num_people_1 * cost_1 + num_people_2 * cost_2
  let total_before_discount := meal_cost + service_charge
  let discount := discount_rate * meal_cost
  let total_after_discount := total_before_discount - discount
  let tip := tip_rate * total_before_discount
  total_after_discount + tip

theorem restaurant_bill_calculation :
  restaurant_bill 10 5 18 25 50 (5/100) (10/100) = 375.25 := by
  sorry

end NUMINAMATH_CALUDE_restaurant_bill_calculation_l1114_111467


namespace NUMINAMATH_CALUDE_max_value_inequality_l1114_111491

theorem max_value_inequality (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (x * y + y * z) / (x^2 + y^2 + z^2) ≤ Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_max_value_inequality_l1114_111491


namespace NUMINAMATH_CALUDE_complex_equation_implies_sum_of_squares_l1114_111483

/-- The imaginary unit -/
def i : ℂ := Complex.I

/-- Given complex equation -/
def complex_equation (a b : ℝ) : Prop :=
  (4 - 3 * i) * (a + b * i) = 25 * i

/-- Theorem stating that the given complex equation implies a² + b² = 25 -/
theorem complex_equation_implies_sum_of_squares (a b : ℝ) :
  complex_equation a b → a^2 + b^2 = 25 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_implies_sum_of_squares_l1114_111483


namespace NUMINAMATH_CALUDE_system_solution_l1114_111415

theorem system_solution (a b c : ℝ) : 
  (∀ x y, a * x + b * y = 2 ∧ c * x - 7 * y = 8) →
  (a * 3 + b * (-2) = 2 ∧ c * 3 - 7 * (-2) = 8) →
  (a * (-2) + b * 2 = 2) →
  (a = 4 ∧ b = 5 ∧ c = -2) := by
sorry

end NUMINAMATH_CALUDE_system_solution_l1114_111415


namespace NUMINAMATH_CALUDE_power_product_six_three_l1114_111407

theorem power_product_six_three : (6 : ℕ)^5 * (3 : ℕ)^5 = 1889568 := by
  sorry

end NUMINAMATH_CALUDE_power_product_six_three_l1114_111407


namespace NUMINAMATH_CALUDE_vector_simplification_l1114_111494

variable {V : Type*} [AddCommGroup V]

variable (A B C M O : V)

theorem vector_simplification :
  (B - A) + (B - M) + (O - B) + (C - B) + (M - O) = C - A :=
by sorry

end NUMINAMATH_CALUDE_vector_simplification_l1114_111494


namespace NUMINAMATH_CALUDE_fraction_equality_l1114_111430

theorem fraction_equality (a b : ℝ) (h1 : a ≠ b) (h2 : b ≠ 0) :
  let x := a / b
  (a^2 + b^2) / (a^2 - b^2) = (x^2 + 1) / (x^2 - 1) := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l1114_111430


namespace NUMINAMATH_CALUDE_greatest_two_digit_with_digit_product_16_l1114_111439

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def digit_product (n : ℕ) : ℕ :=
  (n / 10) * (n % 10)

theorem greatest_two_digit_with_digit_product_16 :
  ∀ n : ℕ, is_two_digit n → digit_product n = 16 → n ≤ 82 :=
sorry

end NUMINAMATH_CALUDE_greatest_two_digit_with_digit_product_16_l1114_111439


namespace NUMINAMATH_CALUDE_binomial_coefficient_equality_l1114_111469

theorem binomial_coefficient_equality (k : ℕ) : 
  (Nat.choose 18 k = Nat.choose 18 (2 * k - 3)) ↔ (k = 3 ∨ k = 7) := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_equality_l1114_111469
