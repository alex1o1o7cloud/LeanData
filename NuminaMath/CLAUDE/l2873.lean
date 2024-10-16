import Mathlib

namespace NUMINAMATH_CALUDE_tangent_line_equation_l2873_287384

/-- The equation of the tangent line to y = ln x + x^2 at (1, 1) is 3x - y - 2 = 0 -/
theorem tangent_line_equation (x y : ℝ) : 
  let f : ℝ → ℝ := λ t => Real.log t + t^2
  let f' : ℝ → ℝ := λ t => 1/t + 2*t
  let slope : ℝ := f' 1
  let point : ℝ × ℝ := (1, 1)
  3*x - y - 2 = 0 ↔ y - point.2 = slope * (x - point.1) :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_equation_l2873_287384


namespace NUMINAMATH_CALUDE_rose_incorrect_answers_l2873_287394

theorem rose_incorrect_answers
  (total_items : ℕ)
  (liza_percentage : ℚ)
  (rose_additional_correct : ℕ)
  (h1 : total_items = 60)
  (h2 : liza_percentage = 90 / 100)
  (h3 : rose_additional_correct = 2)
  : total_items - (liza_percentage * total_items + rose_additional_correct) = 4 := by
  sorry

end NUMINAMATH_CALUDE_rose_incorrect_answers_l2873_287394


namespace NUMINAMATH_CALUDE_inequality_and_equality_condition_l2873_287393

theorem inequality_and_equality_condition (x y : ℝ) :
  5 * x^2 + y^2 + 1 ≥ 4 * x * y + 2 * x ∧
  (5 * x^2 + y^2 + 1 = 4 * x * y + 2 * x ↔ x = 1 ∧ y = 2) :=
by sorry

end NUMINAMATH_CALUDE_inequality_and_equality_condition_l2873_287393


namespace NUMINAMATH_CALUDE_newspaper_pieces_not_all_found_l2873_287358

theorem newspaper_pieces_not_all_found :
  ¬∃ (k p v : ℕ), 1988 = k + 4 * p + 8 * v ∧ k > 0 := by
  sorry

end NUMINAMATH_CALUDE_newspaper_pieces_not_all_found_l2873_287358


namespace NUMINAMATH_CALUDE_rectangle_ellipse_theorem_l2873_287309

/-- Represents a rectangle ABCD with an inscribed ellipse K -/
structure RectangleWithEllipse where
  -- Length of side AB
  ab : ℝ
  -- Length of side AD
  ad : ℝ
  -- Point M on AB where the minor axis of K intersects
  am : ℝ
  -- Point L on AB where the minor axis of K intersects
  lb : ℝ
  -- Ensure AB = 2
  ab_eq_two : ab = 2
  -- Ensure AD < √2
  ad_lt_sqrt_two : ad < Real.sqrt 2
  -- Ensure M and L are on AB
  m_l_on_ab : am + lb = ab

/-- The theorem to be proved -/
theorem rectangle_ellipse_theorem (rect : RectangleWithEllipse) :
  rect.am^2 - rect.lb^2 = -8 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_ellipse_theorem_l2873_287309


namespace NUMINAMATH_CALUDE_amanda_friends_count_l2873_287355

def total_tickets : ℕ := 80
def tickets_per_friend : ℕ := 4
def second_day_tickets : ℕ := 32
def third_day_tickets : ℕ := 28

theorem amanda_friends_count :
  ∃ (friends : ℕ), 
    friends * tickets_per_friend + second_day_tickets + third_day_tickets = total_tickets ∧
    friends = 5 := by
  sorry

end NUMINAMATH_CALUDE_amanda_friends_count_l2873_287355


namespace NUMINAMATH_CALUDE_y1_greater_than_y2_l2873_287311

/-- A linear function f(x) = -x + 1 -/
def f (x : ℝ) : ℝ := -x + 1

theorem y1_greater_than_y2 (y1 y2 : ℝ) 
  (h1 : f (-2) = y1) 
  (h2 : f 2 = y2) : 
  y1 > y2 := by
  sorry

end NUMINAMATH_CALUDE_y1_greater_than_y2_l2873_287311


namespace NUMINAMATH_CALUDE_elementary_symmetric_polynomials_l2873_287337

variable (x y z : ℝ)

/-- Elementary symmetric polynomial of degree 1 -/
def σ₁ (x y z : ℝ) : ℝ := x + y + z

/-- Elementary symmetric polynomial of degree 2 -/
def σ₂ (x y z : ℝ) : ℝ := x*y + y*z + z*x

/-- Elementary symmetric polynomial of degree 3 -/
def σ₃ (x y z : ℝ) : ℝ := x*y*z

theorem elementary_symmetric_polynomials (x y z : ℝ) :
  ((x + y) * (y + z) * (x + z) = σ₂ x y z * σ₁ x y z - σ₃ x y z) ∧
  (x^3 + y^3 + z^3 - 3*x*y*z = σ₁ x y z * (σ₁ x y z^2 - 3 * σ₂ x y z)) ∧
  (x^3 + y^3 = σ₁ x y 0^3 - 3 * σ₁ x y 0 * σ₂ x y 0) ∧
  ((x^2 + y^2) * (y^2 + z^2) * (x^2 + z^2) = 
    σ₁ x y z^2 * σ₂ x y z^2 + 4 * σ₁ x y z * σ₂ x y z * σ₃ x y z - 
    2 * σ₂ x y z^3 - 2 * σ₁ x y z^3 * σ₃ x y z - σ₃ x y z^2) ∧
  (x^4 + y^4 + z^4 = 
    σ₁ x y z^4 - 4 * σ₁ x y z^2 * σ₂ x y z + 2 * σ₂ x y z^2 + 4 * σ₁ x y z * σ₃ x y z) :=
by sorry

end NUMINAMATH_CALUDE_elementary_symmetric_polynomials_l2873_287337


namespace NUMINAMATH_CALUDE_series_sum_equals_three_l2873_287342

theorem series_sum_equals_three (k : ℝ) (hk : k > 1) :
  (∑' n : ℕ, (n^2 + 3*n - 2) / k^n) = 2 → k = 3 := by
  sorry

end NUMINAMATH_CALUDE_series_sum_equals_three_l2873_287342


namespace NUMINAMATH_CALUDE_anna_bob_numbers_not_equal_l2873_287370

/-- Represents a number formed by concatenating consecutive positive integers -/
def ConsecutiveIntegerNumber (start : ℕ) (count : ℕ) : ℕ := sorry

/-- Anna's number is formed by 20 consecutive positive integers -/
def AnnaNumber (start : ℕ) : ℕ := ConsecutiveIntegerNumber start 20

/-- Bob's number is formed by 21 consecutive positive integers -/
def BobNumber (start : ℕ) : ℕ := ConsecutiveIntegerNumber start 21

/-- Theorem stating that Anna's and Bob's numbers cannot be equal -/
theorem anna_bob_numbers_not_equal :
  ∀ (a b : ℕ), AnnaNumber a ≠ BobNumber b :=
sorry

end NUMINAMATH_CALUDE_anna_bob_numbers_not_equal_l2873_287370


namespace NUMINAMATH_CALUDE_multiply_divide_multiply_l2873_287348

theorem multiply_divide_multiply : 8 * 7 / 8 * 7 = 49 := by
  sorry

end NUMINAMATH_CALUDE_multiply_divide_multiply_l2873_287348


namespace NUMINAMATH_CALUDE_population_after_four_years_l2873_287300

def population_after_n_years (initial_population : ℕ) (new_people : ℕ) (people_leaving : ℕ) (years : ℕ) : ℕ :=
  let population_after_changes := initial_population + new_people - people_leaving
  (population_after_changes / 2^years : ℕ)

theorem population_after_four_years :
  population_after_n_years 780 100 400 4 = 30 := by
  sorry

end NUMINAMATH_CALUDE_population_after_four_years_l2873_287300


namespace NUMINAMATH_CALUDE_sons_age_l2873_287362

theorem sons_age (father_age son_age : ℕ) : 
  father_age = 3 * son_age →
  (father_age - 8) = 4 * (son_age - 8) →
  son_age = 24 := by
sorry

end NUMINAMATH_CALUDE_sons_age_l2873_287362


namespace NUMINAMATH_CALUDE_dividend_divisor_quotient_problem_l2873_287304

theorem dividend_divisor_quotient_problem :
  ∀ (dividend divisor quotient : ℕ),
    dividend = 6 * divisor →
    divisor = 6 * quotient →
    dividend = divisor * quotient →
    dividend = 216 ∧ divisor = 36 ∧ quotient = 6 := by
  sorry

end NUMINAMATH_CALUDE_dividend_divisor_quotient_problem_l2873_287304


namespace NUMINAMATH_CALUDE_sequence_term_exists_l2873_287303

theorem sequence_term_exists : ∃ n : ℕ, n * (n + 2) = 99 := by
  sorry

end NUMINAMATH_CALUDE_sequence_term_exists_l2873_287303


namespace NUMINAMATH_CALUDE_real_part_of_i_times_i_minus_one_l2873_287314

theorem real_part_of_i_times_i_minus_one :
  Complex.re (Complex.I * (Complex.I - 1)) = -1 := by
  sorry

end NUMINAMATH_CALUDE_real_part_of_i_times_i_minus_one_l2873_287314


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l2873_287387

theorem quadratic_inequality_solution (m : ℝ) : 
  (∀ x : ℝ, (- (1/2) * x^2 + 2*x > m*x) ↔ (0 < x ∧ x < 2)) → m = 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l2873_287387


namespace NUMINAMATH_CALUDE_simplify_square_roots_l2873_287376

theorem simplify_square_roots : 81^(1/2) - 144^(1/2) = -63 := by sorry

end NUMINAMATH_CALUDE_simplify_square_roots_l2873_287376


namespace NUMINAMATH_CALUDE_total_cans_of_peas_l2873_287301

-- Define the number of cans per box
def cans_per_box : ℕ := 4

-- Define the number of boxes ordered
def boxes_ordered : ℕ := 203

-- Theorem to prove
theorem total_cans_of_peas : cans_per_box * boxes_ordered = 812 := by
  sorry

end NUMINAMATH_CALUDE_total_cans_of_peas_l2873_287301


namespace NUMINAMATH_CALUDE_absent_student_percentage_l2873_287367

theorem absent_student_percentage (total_students : ℕ) (boys : ℕ) (girls : ℕ) 
  (h1 : total_students = 120)
  (h2 : boys = 70)
  (h3 : girls = 50)
  (h4 : total_students = boys + girls)
  (boys_absent_fraction : ℚ)
  (girls_absent_fraction : ℚ)
  (h5 : boys_absent_fraction = 1 / 7)
  (h6 : girls_absent_fraction = 1 / 5) :
  (boys_absent_fraction * boys + girls_absent_fraction * girls) / total_students = 1 / 6 := by
  sorry

end NUMINAMATH_CALUDE_absent_student_percentage_l2873_287367


namespace NUMINAMATH_CALUDE_fred_seashell_count_l2873_287341

/-- The number of seashells Tom found -/
def tom_seashells : ℕ := 15

/-- The difference between Fred's and Tom's seashell counts -/
def fred_tom_difference : ℕ := 28

/-- The number of seashells Fred found -/
def fred_seashells : ℕ := tom_seashells + fred_tom_difference

theorem fred_seashell_count : fred_seashells = 43 := by
  sorry

end NUMINAMATH_CALUDE_fred_seashell_count_l2873_287341


namespace NUMINAMATH_CALUDE_triangle_area_l2873_287363

/-- The area of a triangle with base 4 and height 8 is 16 -/
theorem triangle_area : ℝ → ℝ → ℝ → Prop :=
  fun base height area =>
    base = 4 ∧ height = 8 →
    area = (base * height) / 2 →
    area = 16

/-- Proof of the theorem -/
lemma prove_triangle_area : triangle_area 4 8 16 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l2873_287363


namespace NUMINAMATH_CALUDE_parabola_zeros_difference_l2873_287347

/-- Represents a parabola of the form y = ax² + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The y-coordinate for a given x on the parabola -/
def Parabola.y (p : Parabola) (x : ℝ) : ℝ :=
  p.a * x^2 + p.b * x + p.c

/-- The zeros of the parabola -/
def Parabola.zeros (p : Parabola) : Set ℝ :=
  {x : ℝ | p.y x = 0}

theorem parabola_zeros_difference (p : Parabola) :
  p.y 1 = -2 →  -- Vertex at (1, -2)
  p.y 3 = 10 →  -- Point (3, 10) on the parabola
  ∃ m n : ℝ,
    m ∈ p.zeros ∧
    n ∈ p.zeros ∧
    m > n ∧
    m - n = 2 * Real.sqrt 6 / 3 := by
  sorry

end NUMINAMATH_CALUDE_parabola_zeros_difference_l2873_287347


namespace NUMINAMATH_CALUDE_fish_selection_probabilities_l2873_287385

/-- The number of fish in the aquarium -/
def total_fish : ℕ := 6

/-- The number of black fish in the aquarium -/
def black_fish : ℕ := 4

/-- The number of red fish in the aquarium -/
def red_fish : ℕ := 2

/-- The number of days the teacher has classes -/
def class_days : ℕ := 4

/-- The probability of selecting fish of the same color in two consecutive draws -/
def prob_same_color : ℚ := 5 / 9

/-- The probability of selecting fish of different colors in two consecutive draws on exactly 2 out of 4 days -/
def prob_diff_color_two_days : ℚ := 800 / 2187

theorem fish_selection_probabilities :
  (prob_same_color = (black_fish / total_fish) ^ 2 + (red_fish / total_fish) ^ 2) ∧
  (prob_diff_color_two_days = 
    (class_days.choose 2 : ℚ) * prob_same_color ^ 2 * (1 - prob_same_color) ^ 2) := by
  sorry

end NUMINAMATH_CALUDE_fish_selection_probabilities_l2873_287385


namespace NUMINAMATH_CALUDE_negation_of_implication_l2873_287378

theorem negation_of_implication (x y : ℝ) :
  (¬(x = y → Real.sqrt x = Real.sqrt y)) ↔ (x ≠ y → Real.sqrt x ≠ Real.sqrt y) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_implication_l2873_287378


namespace NUMINAMATH_CALUDE_prime_sum_theorem_l2873_287328

theorem prime_sum_theorem : ∃ (A B : ℕ), 
  0 < A ∧ 0 < B ∧
  Nat.Prime A ∧ 
  Nat.Prime B ∧ 
  Nat.Prime (A - B) ∧ 
  Nat.Prime (A - 2*B) ∧
  A + B + (A - B) + (A - 2*B) = 17 := by
sorry

end NUMINAMATH_CALUDE_prime_sum_theorem_l2873_287328


namespace NUMINAMATH_CALUDE_green_marbles_after_replacement_l2873_287392

/-- Represents the number of marbles of each color in a jar -/
structure MarbleJar where
  red : ℕ
  green : ℕ
  blue : ℕ
  yellow : ℕ
  purple : ℕ
  white : ℕ

/-- Calculates the total number of marbles in the jar -/
def totalMarbles (jar : MarbleJar) : ℕ :=
  jar.red + jar.green + jar.blue + jar.yellow + jar.purple + jar.white

/-- Represents the percentage of each color in the jar -/
structure MarblePercentages where
  red : ℚ
  green : ℚ
  blue : ℚ
  yellow : ℚ
  purple : ℚ

/-- Theorem stating the final number of green marbles after replacement -/
theorem green_marbles_after_replacement (jar : MarbleJar) (percentages : MarblePercentages) :
  percentages.red = 25 / 100 →
  percentages.green = 15 / 100 →
  percentages.blue = 20 / 100 →
  percentages.yellow = 10 / 100 →
  percentages.purple = 15 / 100 →
  jar.white = 35 →
  (jar.red : ℚ) / (totalMarbles jar : ℚ) = percentages.red →
  (jar.green : ℚ) / (totalMarbles jar : ℚ) = percentages.green →
  (jar.blue : ℚ) / (totalMarbles jar : ℚ) = percentages.blue →
  (jar.yellow : ℚ) / (totalMarbles jar : ℚ) = percentages.yellow →
  (jar.purple : ℚ) / (totalMarbles jar : ℚ) = percentages.purple →
  (jar.white : ℚ) / (totalMarbles jar : ℚ) = 1 - (percentages.red + percentages.green + percentages.blue + percentages.yellow + percentages.purple) →
  jar.green + jar.red / 3 = 55 := by
  sorry

end NUMINAMATH_CALUDE_green_marbles_after_replacement_l2873_287392


namespace NUMINAMATH_CALUDE_cosine_equality_l2873_287366

theorem cosine_equality (n : ℤ) (h1 : 0 ≤ n) (h2 : n ≤ 180) :
  Real.cos (n * π / 180) = Real.cos (317 * π / 180) → n = 43 := by
  sorry

end NUMINAMATH_CALUDE_cosine_equality_l2873_287366


namespace NUMINAMATH_CALUDE_valid_prices_count_l2873_287330

def valid_digits : List Nat := [1, 1, 4, 5, 6, 6]

def is_valid_start (n : Nat) : Bool :=
  n ≥ 4

def count_valid_prices (digits : List Nat) : Nat :=
  digits.filter is_valid_start
    |>.map (λ d => (digits.erase d).permutations.length)
    |>.sum

theorem valid_prices_count :
  count_valid_prices valid_digits = 90 := by
  sorry

end NUMINAMATH_CALUDE_valid_prices_count_l2873_287330


namespace NUMINAMATH_CALUDE_cherry_soda_count_l2873_287398

theorem cherry_soda_count (total : ℕ) (cherry : ℕ) (orange : ℕ) 
  (h1 : total = 24)
  (h2 : orange = 2 * cherry)
  (h3 : total = cherry + orange) : cherry = 8 := by
  sorry

end NUMINAMATH_CALUDE_cherry_soda_count_l2873_287398


namespace NUMINAMATH_CALUDE_ice_skating_rinks_and_ski_resorts_l2873_287361

theorem ice_skating_rinks_and_ski_resorts (x y : ℕ) : 
  x + y = 1230 →
  2 * x + y + 500 = 2560 →
  x = 830 ∧ y = 400 := by
sorry

end NUMINAMATH_CALUDE_ice_skating_rinks_and_ski_resorts_l2873_287361


namespace NUMINAMATH_CALUDE_polynomial_identity_l2873_287369

theorem polynomial_identity (x : ℝ) (h : x + 1/x = 3) :
  x^12 - 7*x^6 + x^2 = 45363*x - 17327 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_identity_l2873_287369


namespace NUMINAMATH_CALUDE_infinitely_many_solutions_l2873_287302

theorem infinitely_many_solutions (d : ℝ) : 
  (∀ y : ℝ, 3 * (5 + d * y) = 15 * y + 15) ↔ d = 5 := by
  sorry

end NUMINAMATH_CALUDE_infinitely_many_solutions_l2873_287302


namespace NUMINAMATH_CALUDE_horner_method_for_f_at_2_l2873_287352

def f (x : ℝ) : ℝ := 2 * x^5 + 3 * x^4 + 2 * x^3 - 4 * x + 5

theorem horner_method_for_f_at_2 : f 2 = 125 := by
  sorry

end NUMINAMATH_CALUDE_horner_method_for_f_at_2_l2873_287352


namespace NUMINAMATH_CALUDE_monkey_climb_time_l2873_287349

/-- A monkey climbing a tree problem -/
theorem monkey_climb_time (tree_height : ℕ) (hop_distance : ℕ) (slip_distance : ℕ) 
  (h1 : tree_height = 20)
  (h2 : hop_distance = 3)
  (h3 : slip_distance = 2) :
  ∃ (time : ℕ), time = 17 ∧ 
  time * (hop_distance - slip_distance) + hop_distance ≥ tree_height :=
by
  sorry

end NUMINAMATH_CALUDE_monkey_climb_time_l2873_287349


namespace NUMINAMATH_CALUDE_bertolli_farm_tomatoes_bertolli_farm_tomatoes_proof_l2873_287377

theorem bertolli_farm_tomatoes : ℕ → Prop :=
  fun tomatoes =>
    let corn : ℕ := 4112
    let onions : ℕ := 985
    let onions_difference : ℕ := 5200
    onions = tomatoes + corn - onions_difference →
    tomatoes = 2073

-- The proof is omitted
theorem bertolli_farm_tomatoes_proof : bertolli_farm_tomatoes 2073 := by
  sorry

end NUMINAMATH_CALUDE_bertolli_farm_tomatoes_bertolli_farm_tomatoes_proof_l2873_287377


namespace NUMINAMATH_CALUDE_chosen_number_l2873_287346

theorem chosen_number (x : ℝ) : x / 8 - 100 = 6 → x = 848 := by
  sorry

end NUMINAMATH_CALUDE_chosen_number_l2873_287346


namespace NUMINAMATH_CALUDE_alternating_sum_coefficients_l2873_287383

theorem alternating_sum_coefficients :
  ∀ (a a₁ a₂ a₃ a₄ a₅ a₆ a₇ : ℝ),
  (∀ x : ℝ, (1 + 2*x)^2 * (1 - x)^5 = a + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 + a₆*x^6 + a₇*x^7) →
  a₁ - a₂ + a₃ - a₄ + a₅ - a₆ + a₇ = -31 :=
by sorry

end NUMINAMATH_CALUDE_alternating_sum_coefficients_l2873_287383


namespace NUMINAMATH_CALUDE_median_sum_bounds_l2873_287359

/-- Given a triangle with sides a, b, c and medians m_a, m_b, m_c, 
    prove that the sum of any two medians is bounded by the perimeter and semi-perimeter. -/
theorem median_sum_bounds (a b c m_a m_b m_c : ℝ) 
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b)
  (h_median_a : m_a^2 = (2*b^2 + 2*c^2 - a^2) / 4)
  (h_median_b : m_b^2 = (2*c^2 + 2*a^2 - b^2) / 4)
  (h_median_c : m_c^2 = (2*a^2 + 2*b^2 - c^2) / 4) :
  m_a + m_b ≤ 3/4 * (a + b + c) ∧ 
  m_a + m_b ≥ 3/4 * ((a + b + c) / 2) := by
  sorry

end NUMINAMATH_CALUDE_median_sum_bounds_l2873_287359


namespace NUMINAMATH_CALUDE_election_votes_l2873_287373

theorem election_votes (total_votes : ℕ) (invalid_percent : ℚ) (winner_percent : ℚ) 
  (h1 : total_votes = 7000)
  (h2 : invalid_percent = 1/5)
  (h3 : winner_percent = 11/20) :
  ↑total_votes * (1 - invalid_percent) * (1 - winner_percent) = 2520 := by
  sorry

end NUMINAMATH_CALUDE_election_votes_l2873_287373


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l2873_287338

def A : Set ℝ := {x | x^2 - 4*x > 0}
def B : Set ℝ := {x | |x - 1| ≤ 2}

theorem intersection_of_A_and_B : A ∩ B = {x | -1 ≤ x ∧ x < 0} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l2873_287338


namespace NUMINAMATH_CALUDE_is_2013_in_sequence_l2873_287343

/-- An arithmetic sequence containing 13, 25, and 41 as terms (not necessarily consecutive) -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  is_arithmetic : ∀ n m, a (n + 1) - a n = a (m + 1) - a m
  has_13 : ∃ i, a i = 13
  has_25 : ∃ j, a j = 25
  has_41 : ∃ k, a k = 41

/-- 2013 is a term in the arithmetic sequence -/
theorem is_2013_in_sequence (seq : ArithmeticSequence) : ∃ n, seq.a n = 2013 := by
  sorry

end NUMINAMATH_CALUDE_is_2013_in_sequence_l2873_287343


namespace NUMINAMATH_CALUDE_pool_length_l2873_287365

theorem pool_length (width : ℝ) (area : ℝ) (length : ℝ) : 
  width = 3 → 
  area = 30 → 
  area = length * width → 
  length = 10 := by
sorry

end NUMINAMATH_CALUDE_pool_length_l2873_287365


namespace NUMINAMATH_CALUDE_coloring_schemes_formula_l2873_287396

/-- The number of different coloring schemes for n connected regions using m colors -/
def coloringSchemes (m n : ℕ) : ℕ :=
  ((-1 : ℤ) ^ n * (m - 1 : ℤ) + (m - 1 : ℤ) ^ n).natAbs

/-- Theorem stating the number of different coloring schemes for n connected regions using m colors -/
theorem coloring_schemes_formula (m n : ℕ) (hm : m ≥ 2) (hn : n ≥ 2) :
  coloringSchemes m n = ((-1 : ℤ) ^ n * (m - 1 : ℤ) + (m - 1 : ℤ) ^ n).natAbs := by
  sorry

end NUMINAMATH_CALUDE_coloring_schemes_formula_l2873_287396


namespace NUMINAMATH_CALUDE_total_harvest_l2873_287374

def tomato_harvest (day1 : ℕ) (extra_day2 : ℕ) : ℕ := 
  let day2 := day1 + extra_day2
  let day3 := 2 * day2
  day1 + day2 + day3

theorem total_harvest : tomato_harvest 120 50 = 630 := by
  sorry

end NUMINAMATH_CALUDE_total_harvest_l2873_287374


namespace NUMINAMATH_CALUDE_perpendicular_line_equation_l2873_287327

/-- Given a line L1 with equation mx - m^2y = 1 and a point P(2,1) on L1,
    prove that the line L2 perpendicular to L1 at P has equation x + y - 3 = 0 -/
theorem perpendicular_line_equation (m : ℝ) :
  let L1 : ℝ → ℝ → Prop := λ x y ↦ m * x - m^2 * y = 1
  let P : ℝ × ℝ := (2, 1)
  let L2 : ℝ → ℝ → Prop := λ x y ↦ x + y - 3 = 0
  L1 P.1 P.2 → L2 = (λ x y ↦ x + y - 3 = 0) :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_line_equation_l2873_287327


namespace NUMINAMATH_CALUDE_max_k_value_l2873_287310

theorem max_k_value (x y k : ℝ) (hx : x > 0) (hy : y > 0) (hk : k > 0)
  (h : 5 = k^2 * ((x^2 / y^2) + (y^2 / x^2)) + k * (x / y + y / x)) :
  k ≤ (-1 + Real.sqrt 22) / 2 := by
sorry

end NUMINAMATH_CALUDE_max_k_value_l2873_287310


namespace NUMINAMATH_CALUDE_jake_peaches_count_l2873_287368

-- Define the variables
def steven_peaches : ℕ := 13
def steven_apples : ℕ := 52
def jake_apples : ℕ := steven_apples + 84

-- Define Jake's peaches in terms of Steven's
def jake_peaches : ℕ := steven_peaches - 10

-- Theorem to prove
theorem jake_peaches_count : jake_peaches = 3 := by
  sorry

end NUMINAMATH_CALUDE_jake_peaches_count_l2873_287368


namespace NUMINAMATH_CALUDE_mistake_correction_l2873_287354

theorem mistake_correction (x : ℝ) : 8 * x + 8 = 56 → x / 8 = 0.75 := by
  sorry

end NUMINAMATH_CALUDE_mistake_correction_l2873_287354


namespace NUMINAMATH_CALUDE_unique_x_value_l2873_287305

theorem unique_x_value (x : ℝ) : x^2 ∈ ({0, 1, x} : Set ℝ) → x = -1 := by
  sorry

end NUMINAMATH_CALUDE_unique_x_value_l2873_287305


namespace NUMINAMATH_CALUDE_factorial_ratio_equals_119_factorial_l2873_287312

theorem factorial_ratio_equals_119_factorial : (Nat.factorial (Nat.factorial 5)) / (Nat.factorial 5) = Nat.factorial 119 := by
  sorry

end NUMINAMATH_CALUDE_factorial_ratio_equals_119_factorial_l2873_287312


namespace NUMINAMATH_CALUDE_constant_speed_calculation_l2873_287325

/-- Proves that a journey of 2304 kilometers completed in 36 hours at a constant speed results in a speed of 64 km/h -/
theorem constant_speed_calculation (distance : ℝ) (time : ℝ) (speed : ℝ) :
  distance = 2304 →
  time = 36 →
  speed = distance / time →
  speed = 64 := by
sorry

end NUMINAMATH_CALUDE_constant_speed_calculation_l2873_287325


namespace NUMINAMATH_CALUDE_upstream_rate_calculation_l2873_287375

/-- Represents the rowing rates and current speed in kilometers per hour. -/
structure RowingScenario where
  downstream_rate : ℝ
  still_water_rate : ℝ
  current_rate : ℝ

/-- Calculates the upstream rate given a RowingScenario. -/
def upstream_rate (scenario : RowingScenario) : ℝ :=
  scenario.still_water_rate - scenario.current_rate

/-- Theorem stating that for the given scenario, the upstream rate is 10 kmph. -/
theorem upstream_rate_calculation (scenario : RowingScenario) 
  (h1 : scenario.downstream_rate = 30)
  (h2 : scenario.still_water_rate = 20)
  (h3 : scenario.current_rate = 10) :
  upstream_rate scenario = 10 := by
  sorry

#check upstream_rate_calculation

end NUMINAMATH_CALUDE_upstream_rate_calculation_l2873_287375


namespace NUMINAMATH_CALUDE_deck_size_proof_l2873_287388

theorem deck_size_proof (r b : ℕ) : 
  (r : ℚ) / (r + b) = 1/4 →
  (r : ℚ) / (r + b + 6) = 1/6 →
  r + b = 12 := by
sorry

end NUMINAMATH_CALUDE_deck_size_proof_l2873_287388


namespace NUMINAMATH_CALUDE_nils_geese_count_l2873_287395

/-- Represents the number of days the feed lasts -/
def FeedDuration : ℕ → ℕ → ℕ
  | n, k => k

/-- Represents the change in feed duration when selling geese -/
def SellGeese (n : ℕ) : ℕ := FeedDuration (n - 50) (FeedDuration n 0 + 20)

/-- Represents the change in feed duration when buying geese -/
def BuyGeese (n : ℕ) : ℕ := FeedDuration (n + 100) (FeedDuration n 0 - 10)

/-- The theorem stating that Nils has 300 geese -/
theorem nils_geese_count :
  ∃ (n : ℕ), n = 300 ∧ 
  SellGeese n = FeedDuration n 0 + 20 ∧
  BuyGeese n = FeedDuration n 0 - 10 :=
sorry

end NUMINAMATH_CALUDE_nils_geese_count_l2873_287395


namespace NUMINAMATH_CALUDE_coefficient_x_cubed_sum_binomial_l2873_287334

theorem coefficient_x_cubed_sum_binomial (n : ℕ) (hn : n ≥ 3) :
  (Finset.range (n - 2)).sum (fun k => Nat.choose (k + 3) 3) = Nat.choose (n + 1) 4 := by
  sorry

#check coefficient_x_cubed_sum_binomial 2005

end NUMINAMATH_CALUDE_coefficient_x_cubed_sum_binomial_l2873_287334


namespace NUMINAMATH_CALUDE_time_addition_theorem_l2873_287332

/-- Represents a date and time --/
structure DateTime where
  year : Nat
  month : Nat
  day : Nat
  hour : Nat
  minute : Nat

/-- Adds minutes to a DateTime --/
def addMinutes (dt : DateTime) (minutes : Nat) : DateTime :=
  sorry

/-- Checks if two DateTimes are equal --/
def dateTimeEqual (dt1 dt2 : DateTime) : Prop :=
  dt1.year = dt2.year ∧
  dt1.month = dt2.month ∧
  dt1.day = dt2.day ∧
  dt1.hour = dt2.hour ∧
  dt1.minute = dt2.minute

theorem time_addition_theorem :
  let start := DateTime.mk 2023 7 4 12 0
  let end_time := DateTime.mk 2023 7 6 21 36
  dateTimeEqual (addMinutes start 3456) end_time :=
by sorry

end NUMINAMATH_CALUDE_time_addition_theorem_l2873_287332


namespace NUMINAMATH_CALUDE_quadratic_completing_square_l2873_287333

theorem quadratic_completing_square :
  ∀ x : ℝ, x^2 - 6*x - 5 = 0 ↔ (x - 3)^2 = 14 := by
sorry

end NUMINAMATH_CALUDE_quadratic_completing_square_l2873_287333


namespace NUMINAMATH_CALUDE_function_equality_l2873_287323

/-- Given a function f such that f(2x) = 2 / (2 + x) for all x > 0,
    prove that 2f(x) = 8 / (4 + x) -/
theorem function_equality (f : ℝ → ℝ) 
    (h : ∀ x > 0, f (2 * x) = 2 / (2 + x)) :
  ∀ x > 0, 2 * f x = 8 / (4 + x) := by
  sorry

end NUMINAMATH_CALUDE_function_equality_l2873_287323


namespace NUMINAMATH_CALUDE_solve_stuffed_animals_l2873_287307

def stuffed_animals_problem (mckenna kenley tenly : ℕ) : Prop :=
  (mckenna = 34) ∧
  (kenley = 2 * mckenna) ∧
  (tenly = kenley + 5) ∧
  (mckenna + kenley + tenly = 175)

theorem solve_stuffed_animals :
  ∃ (mckenna kenley tenly : ℕ), stuffed_animals_problem mckenna kenley tenly :=
by
  sorry

end NUMINAMATH_CALUDE_solve_stuffed_animals_l2873_287307


namespace NUMINAMATH_CALUDE_boxes_loaded_is_100_l2873_287380

/-- The number of boxes loaded on a truck given its capacity and other items --/
def boxes_loaded (truck_capacity : ℕ) (box_weight crate_weight sack_weight bag_weight : ℕ)
  (num_crates num_sacks num_bags : ℕ) : ℕ :=
  (truck_capacity - (crate_weight * num_crates + sack_weight * num_sacks + bag_weight * num_bags)) / box_weight

/-- Theorem stating that 100 boxes were loaded given the specific conditions --/
theorem boxes_loaded_is_100 :
  boxes_loaded 13500 100 60 50 40 10 50 10 = 100 := by
  sorry

end NUMINAMATH_CALUDE_boxes_loaded_is_100_l2873_287380


namespace NUMINAMATH_CALUDE_boat_speed_in_still_water_l2873_287372

/-- The speed of a boat in still water, given downstream travel information -/
theorem boat_speed_in_still_water : 
  let current_speed : ℝ := 6
  let downstream_distance : ℝ := 10.67
  let downstream_time : ℝ := 1/3
  let boat_speed : ℝ := 26.01
  (boat_speed + current_speed) * downstream_time = downstream_distance :=
by sorry

end NUMINAMATH_CALUDE_boat_speed_in_still_water_l2873_287372


namespace NUMINAMATH_CALUDE_a_zero_necessary_not_sufficient_l2873_287351

def is_pure_imaginary (z : ℂ) : Prop := ∃ b : ℝ, z = (0 : ℝ) + b * Complex.I

theorem a_zero_necessary_not_sufficient :
  (∀ a b : ℝ, is_pure_imaginary (Complex.ofReal a + Complex.I * b) → a = 0) ∧
  ¬(∀ a b : ℝ, a = 0 → is_pure_imaginary (Complex.ofReal a + Complex.I * b)) :=
by sorry

end NUMINAMATH_CALUDE_a_zero_necessary_not_sufficient_l2873_287351


namespace NUMINAMATH_CALUDE_product_increased_by_four_l2873_287335

theorem product_increased_by_four (x : ℝ) (h : x = 3) : 5 * x + 4 = 19 := by
  sorry

end NUMINAMATH_CALUDE_product_increased_by_four_l2873_287335


namespace NUMINAMATH_CALUDE_black_area_after_three_changes_l2873_287360

/-- The fraction of area that remains black after one change -/
def change_fraction : ℚ := 5/6 * 9/10

/-- The fraction of area that remains black after n changes -/
def remaining_black (n : ℕ) : ℚ := (change_fraction ^ n)

theorem black_area_after_three_changes : 
  remaining_black 3 = 27/64 := by sorry

end NUMINAMATH_CALUDE_black_area_after_three_changes_l2873_287360


namespace NUMINAMATH_CALUDE_sequence_x_perfect_square_l2873_287386

def perfect_square_sequence (s : ℕ → ℕ) : Prop :=
  ∀ n, ∃ m, s n = m^2

def sequence_x : ℕ → ℤ
| 0 => 0
| 1 => 3
| (n + 2) => 4 * sequence_x (n + 1) - sequence_x n

theorem sequence_x_perfect_square :
  perfect_square_sequence (λ n => (sequence_x (n + 1) * sequence_x (n - 1) + 9).natAbs) := by
  sorry

end NUMINAMATH_CALUDE_sequence_x_perfect_square_l2873_287386


namespace NUMINAMATH_CALUDE_hyperbola_incenter_theorem_l2873_287339

/-- Hyperbola C: x²/4 - y²/5 = 1 -/
def hyperbola (x y : ℝ) : Prop := x^2 / 4 - y^2 / 5 = 1

/-- Point P is on the hyperbola in the first quadrant -/
structure PointOnHyperbola where
  x : ℝ
  y : ℝ
  on_hyperbola : hyperbola x y
  first_quadrant : x > 0 ∧ y > 0

/-- F₁ and F₂ are the left and right foci of the hyperbola -/
structure Foci where
  f₁ : ℝ × ℝ
  f₂ : ℝ × ℝ

/-- I is the incenter of triangle PF₁F₂ -/
def incenter (p : PointOnHyperbola) (f : Foci) : ℝ × ℝ := sorry

/-- |PF₁| = 2|PF₂| -/
def focal_distance_condition (p : PointOnHyperbola) (f : Foci) : Prop :=
  let (x₁, y₁) := f.f₁
  let (x₂, y₂) := f.f₂
  ((p.x - x₁)^2 + (p.y - y₁)^2) = 4 * ((p.x - x₂)^2 + (p.y - y₂)^2)

/-- Vector PI = x * Vector PF₁ + y * Vector PF₂ -/
def vector_condition (p : PointOnHyperbola) (f : Foci) (x y : ℝ) : Prop :=
  let i := incenter p f
  let (x₁, y₁) := f.f₁
  let (x₂, y₂) := f.f₂
  (i.1 - p.x, i.2 - p.y) = (x * (x₁ - p.x) + y * (x₂ - p.x), x * (y₁ - p.y) + y * (y₂ - p.y))

/-- Main theorem -/
theorem hyperbola_incenter_theorem (p : PointOnHyperbola) (f : Foci) (x y : ℝ) :
  focal_distance_condition p f →
  vector_condition p f x y →
  y - x = 2/9 := by sorry

end NUMINAMATH_CALUDE_hyperbola_incenter_theorem_l2873_287339


namespace NUMINAMATH_CALUDE_machine_retail_price_l2873_287379

/-- The retail price of a machine -/
def retail_price : ℝ := 132

/-- The wholesale price of the machine -/
def wholesale_price : ℝ := 99

/-- The discount rate applied to the retail price -/
def discount_rate : ℝ := 0.1

/-- The profit rate as a percentage of the wholesale price -/
def profit_rate : ℝ := 0.2

theorem machine_retail_price :
  retail_price = wholesale_price * (1 + profit_rate) / (1 - discount_rate) :=
by sorry

end NUMINAMATH_CALUDE_machine_retail_price_l2873_287379


namespace NUMINAMATH_CALUDE_sqrt_sum_greater_than_sqrt_of_sum_l2873_287313

theorem sqrt_sum_greater_than_sqrt_of_sum {a b : ℝ} (ha : a > 0) (hb : b > 0) :
  Real.sqrt a + Real.sqrt b > Real.sqrt (a + b) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_greater_than_sqrt_of_sum_l2873_287313


namespace NUMINAMATH_CALUDE_min_distance_line_parabola_l2873_287306

/-- The minimum distance between a point on the line y = (5/12)x - 7 and a point on the parabola y = x^2 is 4007/624. -/
theorem min_distance_line_parabola :
  let line := fun x : ℝ => (5/12) * x - 7
  let parabola := fun x : ℝ => x^2
  let distance := fun (x₁ x₂ : ℝ) => 
    Real.sqrt ((x₂ - x₁)^2 + (parabola x₂ - line x₁)^2)
  ∃ (x₁ x₂ : ℝ), ∀ (y₁ y₂ : ℝ), distance x₁ x₂ ≤ distance y₁ y₂ ∧ 
    distance x₁ x₂ = 4007/624 :=
by sorry

end NUMINAMATH_CALUDE_min_distance_line_parabola_l2873_287306


namespace NUMINAMATH_CALUDE_equation_solutions_l2873_287356

theorem equation_solutions :
  (∃ x : ℝ, 5 * x - 2 = 2 * (x + 2) ∧ x = 2) ∧
  (∃ x : ℝ, 2 * x + (x - 3) / 2 = (2 - x) / 3 - 5 ∧ x = -1) := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l2873_287356


namespace NUMINAMATH_CALUDE_remainder_theorem_l2873_287390

theorem remainder_theorem (x y u v : ℕ) (h1 : x > 0) (h2 : y > 0) 
  (h3 : x = u * y + v) (h4 : v < y) : 
  (x + (2 * u + 1) * y) % y = v := by
  sorry

end NUMINAMATH_CALUDE_remainder_theorem_l2873_287390


namespace NUMINAMATH_CALUDE_equidistant_points_of_f_l2873_287340

/-- A point (x, y) is equidistant if |x| = |y| -/
def is_equidistant (x y : ℝ) : Prop := abs x = abs y

/-- The function f(x) = x^2 + 2x -/
def f (x : ℝ) : ℝ := x^2 + 2*x

/-- Theorem: The points (0,0), (-1,-1), and (-3,3) are equidistant points of f -/
theorem equidistant_points_of_f :
  is_equidistant 0 (f 0) ∧
  is_equidistant (-1) (f (-1)) ∧
  is_equidistant (-3) (f (-3)) :=
sorry

end NUMINAMATH_CALUDE_equidistant_points_of_f_l2873_287340


namespace NUMINAMATH_CALUDE_inscribed_circle_theorem_l2873_287321

/-- A point in the plane -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- A quadrilateral defined by four points -/
structure Quadrilateral :=
  (A B C D : Point)

/-- Checks if a quadrilateral is convex -/
def isConvex (q : Quadrilateral) : Prop :=
  sorry

/-- Checks if a point lies on a line segment between two other points -/
def liesBetween (P Q R : Point) : Prop :=
  sorry

/-- Checks if a quadrilateral has an inscribed circle -/
def hasInscribedCircle (q : Quadrilateral) : Prop :=
  sorry

/-- The main theorem -/
theorem inscribed_circle_theorem (A B C D E F G H P : Point) 
  (q : Quadrilateral) (h1 : q = Quadrilateral.mk A B C D) 
  (h2 : isConvex q)
  (h3 : liesBetween A E B)
  (h4 : liesBetween B F C)
  (h5 : liesBetween C G D)
  (h6 : liesBetween D H A)
  (h7 : P = sorry) -- P is the intersection of EG and FH
  (h8 : hasInscribedCircle (Quadrilateral.mk H A E P))
  (h9 : hasInscribedCircle (Quadrilateral.mk E B F P))
  (h10 : hasInscribedCircle (Quadrilateral.mk F C G P))
  (h11 : hasInscribedCircle (Quadrilateral.mk G D H P)) :
  hasInscribedCircle q :=
sorry

end NUMINAMATH_CALUDE_inscribed_circle_theorem_l2873_287321


namespace NUMINAMATH_CALUDE_unique_solution_is_201_l2873_287319

theorem unique_solution_is_201 : ∃! (n : ℕ+), 
  (Finset.sum (Finset.range n) (λ k => 4*k + 1)) / (Finset.sum (Finset.range n) (λ k => 4*(k + 1))) = 100 / 101 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_unique_solution_is_201_l2873_287319


namespace NUMINAMATH_CALUDE_square_of_binomial_formula_l2873_287317

theorem square_of_binomial_formula (x y : ℝ) :
  (2*x + y) * (y - 2*x) = y^2 - (2*x)^2 :=
by sorry

end NUMINAMATH_CALUDE_square_of_binomial_formula_l2873_287317


namespace NUMINAMATH_CALUDE_smallest_integer_divisible_by_24_and_8_l2873_287350

theorem smallest_integer_divisible_by_24_and_8 : ∃ n : ℕ+, 
  (∀ m : ℕ+, m < n → (¬(24 ∣ m^2) ∨ ¬(8 ∣ m))) ∧ 
  24 ∣ n^2 ∧ 
  8 ∣ n ∧
  ∀ d : ℕ+, d ∣ n → d ≤ 8 :=
by sorry

end NUMINAMATH_CALUDE_smallest_integer_divisible_by_24_and_8_l2873_287350


namespace NUMINAMATH_CALUDE_ellipse_major_axis_length_l2873_287326

/-- The length of the major axis of an ellipse formed by the intersection of a plane and a right circular cylinder -/
def major_axis_length (cylinder_radius : ℝ) (major_axis_ratio : ℝ) : ℝ :=
  2 * cylinder_radius * major_axis_ratio

/-- Theorem: The length of the major axis of the ellipse is 5 -/
theorem ellipse_major_axis_length :
  major_axis_length 2 1.25 = 5 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_major_axis_length_l2873_287326


namespace NUMINAMATH_CALUDE_truncated_cone_radii_relation_l2873_287344

/-- Represents a truncated cone with given dimensions and properties -/
structure TruncatedCone where
  top_radius : ℝ
  bottom_radius : ℝ
  section_ratio : ℝ

/-- Theorem stating the relationship between the radii of a truncated cone
    given specific conditions on its section -/
theorem truncated_cone_radii_relation (cone : TruncatedCone)
  (h1 : cone.top_radius = 5)
  (h2 : cone.section_ratio = 1/2) :
  cone.bottom_radius = 25 := by
  sorry

#check truncated_cone_radii_relation

end NUMINAMATH_CALUDE_truncated_cone_radii_relation_l2873_287344


namespace NUMINAMATH_CALUDE_valid_solutions_l2873_287324

-- Define digits as natural numbers from 0 to 9
def Digit : Type := { n : ℕ // n ≤ 9 }

-- Define the conditions for each case
def case1_conditions (x y z : Digit) : Prop :=
  (10 * x.val + y.val = 3 * (10 * y.val + z.val)) ∧
  (x.val + y.val = y.val + z.val + 3)

def case2_conditions (x y z : Digit) : Prop :=
  (10 * x.val + y.val = 3 * (10 * z.val + x.val)) ∧
  (x.val + y.val = x.val + z.val + 3 ∨ x.val + y.val = x.val + z.val - 3)

def case3_conditions (x y z : Digit) : Prop :=
  (10 * x.val + y.val = 3 * (10 * x.val + z.val)) ∧
  (x.val + y.val = x.val + z.val + 3 ∨ x.val + y.val = x.val + z.val - 3)

def case4_conditions (x y z : Digit) : Prop :=
  (10 * x.val + y.val = 3 * (10 * z.val + y.val)) ∧
  (x.val + y.val = z.val + y.val + 3 ∨ x.val + y.val = z.val + y.val - 3)

-- Main theorem
theorem valid_solutions :
  ∀ (a b : ℕ) (x y z : Digit),
    a > b →
    (case1_conditions x y z ∨ case2_conditions x y z ∨ case3_conditions x y z ∨ case4_conditions x y z) →
    ((a = 72 ∧ b = 24) ∨ (a = 45 ∧ b = 15)) := by
  sorry

end NUMINAMATH_CALUDE_valid_solutions_l2873_287324


namespace NUMINAMATH_CALUDE_binomial_expansion_equal_terms_l2873_287345

theorem binomial_expansion_equal_terms (p q : ℝ) (hp : p > 0) (hq : q > 0) : 
  10 * p^9 * q = 45 * p^8 * q^2 → p + 2*q = 1 → p = 9/13 := by sorry

end NUMINAMATH_CALUDE_binomial_expansion_equal_terms_l2873_287345


namespace NUMINAMATH_CALUDE_video_game_problem_l2873_287318

theorem video_game_problem (total_games : ℕ) (price_per_game : ℕ) (total_earnings : ℕ) :
  total_games - (total_earnings / price_per_game) = total_games - (total_earnings / price_per_game) :=
by sorry

end NUMINAMATH_CALUDE_video_game_problem_l2873_287318


namespace NUMINAMATH_CALUDE_ordering_of_constants_l2873_287364

theorem ordering_of_constants : 
  Real.log 17 < 3 ∧ 3 < Real.exp (Real.sqrt 2) := by sorry

end NUMINAMATH_CALUDE_ordering_of_constants_l2873_287364


namespace NUMINAMATH_CALUDE_triangle_area_maximum_l2873_287336

/-- The area of a triangle with two fixed sides is maximized when the angle between these sides is 90°. -/
theorem triangle_area_maximum (a b : ℝ) (h1 : a > 0) (h2 : b > 0) :
  ∃ θ : ℝ, θ ∈ Set.Icc 0 π ∧
    ∀ φ : ℝ, φ ∈ Set.Icc 0 π →
      (1 / 2) * a * b * Real.sin θ ≥ (1 / 2) * a * b * Real.sin φ :=
  sorry

end NUMINAMATH_CALUDE_triangle_area_maximum_l2873_287336


namespace NUMINAMATH_CALUDE_sine_of_vertex_angle_is_four_fifths_l2873_287329

/-- An isosceles triangle with a special property regarding inscribed rectangles. -/
structure SpecialIsoscelesTriangle where
  -- The vertex angle of the isosceles triangle
  vertex_angle : ℝ
  -- A function that takes two real numbers (representing the sides of a rectangle)
  -- and returns whether that rectangle can be inscribed in the triangle
  is_inscribable : ℝ → ℝ → Prop
  -- The constant perimeter of inscribable rectangles
  constant_perimeter : ℝ
  -- Property: The triangle is isosceles
  is_isosceles : Prop
  -- Property: Any rectangle that can be inscribed has the constant perimeter
  perimeter_is_constant : ∀ x y, is_inscribable x y → x + y = constant_perimeter

/-- 
The main theorem: In a special isosceles triangle where all inscribable rectangles 
have a constant perimeter, the sine of the vertex angle is 4/5.
-/
theorem sine_of_vertex_angle_is_four_fifths (t : SpecialIsoscelesTriangle) : 
  Real.sin t.vertex_angle = 4/5 := by
  sorry


end NUMINAMATH_CALUDE_sine_of_vertex_angle_is_four_fifths_l2873_287329


namespace NUMINAMATH_CALUDE_y_equation_solution_l2873_287381

theorem y_equation_solution (y : ℝ) (c d : ℕ+) 
  (h1 : y^2 + 2*y + 2/y + 1/y^2 = 20)
  (h2 : y = c + Real.sqrt d) : 
  c + d = 5 := by
sorry

end NUMINAMATH_CALUDE_y_equation_solution_l2873_287381


namespace NUMINAMATH_CALUDE_second_object_length_l2873_287315

def tape_length : ℕ := 5
def object1_length : ℕ := 100
def object2_length : ℕ := 780

theorem second_object_length :
  (tape_length ∣ object1_length) ∧ 
  (tape_length ∣ object2_length) ∧ 
  (∃ k : ℕ, k * tape_length = object2_length) →
  object2_length = 780 :=
by sorry

end NUMINAMATH_CALUDE_second_object_length_l2873_287315


namespace NUMINAMATH_CALUDE_sum_of_squares_16_to_30_l2873_287308

def sum_of_squares (n : ℕ) : ℕ := n * (n + 1) * (2 * n + 1) / 6

theorem sum_of_squares_16_to_30 :
  sum_of_squares 30 - sum_of_squares 15 = 8215 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_16_to_30_l2873_287308


namespace NUMINAMATH_CALUDE_alex_jellybean_possibilities_l2873_287353

def total_money : ℕ := 100  -- in pence

def toffee_price : ℕ := 5
def bubblegum_price : ℕ := 3
def jellybean_price : ℕ := 2

def min_toffee_spend : ℕ := 35  -- ⌈100 / 3⌉ rounded up to nearest multiple of 5
def min_bubblegum_spend : ℕ := 27  -- ⌈100 / 4⌉ rounded up to nearest multiple of 3
def min_jellybean_spend : ℕ := 10  -- 100 / 10

def possible_jellybean_counts : Set ℕ := {5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 19}

theorem alex_jellybean_possibilities :
  ∀ n : ℕ, n ∈ possible_jellybean_counts ↔
    ∃ (t b j : ℕ),
      t * toffee_price + b * bubblegum_price + n * jellybean_price = total_money ∧
      t * toffee_price ≥ min_toffee_spend ∧
      b * bubblegum_price ≥ min_bubblegum_spend ∧
      n * jellybean_price ≥ min_jellybean_spend :=
by sorry

end NUMINAMATH_CALUDE_alex_jellybean_possibilities_l2873_287353


namespace NUMINAMATH_CALUDE_coin_ratio_is_equal_l2873_287320

/-- Represents the types of coins in the bag -/
inductive CoinType
  | OneRupee
  | FiftyPaise
  | TwentyFivePaise

/-- The value of a coin in rupees -/
def coinValue (c : CoinType) : Rat :=
  match c with
  | CoinType.OneRupee => 1
  | CoinType.FiftyPaise => 1/2
  | CoinType.TwentyFivePaise => 1/4

/-- The number of coins of each type -/
def numCoins : Nat := 80

/-- The total value of all coins in rupees -/
def totalValue : Rat := 140

/-- Theorem stating that the ratio of coin counts is 1:1:1 -/
theorem coin_ratio_is_equal :
  let oneRupeeCount := numCoins
  let fiftyPaiseCount := numCoins
  let twentyFivePaiseCount := numCoins
  let totalCalculatedValue := oneRupeeCount * coinValue CoinType.OneRupee +
                              fiftyPaiseCount * coinValue CoinType.FiftyPaise +
                              twentyFivePaiseCount * coinValue CoinType.TwentyFivePaise
  totalCalculatedValue = totalValue →
  oneRupeeCount = fiftyPaiseCount ∧ fiftyPaiseCount = twentyFivePaiseCount :=
by
  sorry

#check coin_ratio_is_equal

end NUMINAMATH_CALUDE_coin_ratio_is_equal_l2873_287320


namespace NUMINAMATH_CALUDE_job_completion_days_l2873_287331

/-- Represents the job scenario with given parameters -/
structure JobScenario where
  total_days : ℕ
  initial_workers : ℕ
  days_worked : ℕ
  work_done_fraction : ℚ
  fired_workers : ℕ

/-- Calculates the remaining days to complete the job -/
def remaining_days (job : JobScenario) : ℕ :=
  sorry

/-- Theorem statement for the job completion problem -/
theorem job_completion_days (job : JobScenario) 
  (h1 : job.total_days = 100)
  (h2 : job.initial_workers = 10)
  (h3 : job.days_worked = 20)
  (h4 : job.work_done_fraction = 1/4)
  (h5 : job.fired_workers = 2) :
  remaining_days job = 75 := by sorry

end NUMINAMATH_CALUDE_job_completion_days_l2873_287331


namespace NUMINAMATH_CALUDE_eighth_row_interior_sum_l2873_287316

/-- Sum of all elements in row n of Pascal's Triangle -/
def pascal_row_sum (n : ℕ) : ℕ := 2^(n-1)

/-- Sum of interior numbers in row n of Pascal's Triangle -/
def pascal_interior_sum (n : ℕ) : ℕ := pascal_row_sum n - 2

theorem eighth_row_interior_sum :
  pascal_interior_sum 8 = 126 := by sorry

end NUMINAMATH_CALUDE_eighth_row_interior_sum_l2873_287316


namespace NUMINAMATH_CALUDE_golf_strokes_over_par_l2873_287382

/-- Calculates the number of strokes over par for a golfer -/
def strokes_over_par (holes : ℕ) (avg_strokes_per_hole : ℕ) (par_per_hole : ℕ) : ℕ :=
  (holes * avg_strokes_per_hole) - (holes * par_per_hole)

theorem golf_strokes_over_par :
  strokes_over_par 9 4 3 = 9 := by
  sorry

end NUMINAMATH_CALUDE_golf_strokes_over_par_l2873_287382


namespace NUMINAMATH_CALUDE_base_conversion_l2873_287389

theorem base_conversion (b : ℝ) : 
  b > 0 ∧ (5 * 6 + 4 = 1 * b^2 + 2 * b + 1) → b = -1 + Real.sqrt 34 := by
  sorry

end NUMINAMATH_CALUDE_base_conversion_l2873_287389


namespace NUMINAMATH_CALUDE_third_valid_number_is_105_l2873_287399

def is_valid_number (n : ℕ) : Bool :=
  n < 600

def find_third_valid_number (sequence : List ℕ) : Option ℕ :=
  let valid_numbers := sequence.filter is_valid_number
  valid_numbers.get? 2

theorem third_valid_number_is_105 (sequence : List ℕ) :
  sequence = [59, 16, 95, 55, 67, 19, 98, 10, 50, 71] →
  find_third_valid_number sequence = some 105 := by
  sorry

end NUMINAMATH_CALUDE_third_valid_number_is_105_l2873_287399


namespace NUMINAMATH_CALUDE_multiple_of_n_divisible_by_60_l2873_287397

theorem multiple_of_n_divisible_by_60 (n : ℕ) :
  0 < n →
  n < 200 →
  (∃ k : ℕ, k > 0 ∧ 60 ∣ (k * n)) →
  (∃ p q r : ℕ, Nat.Prime p ∧ Nat.Prime q ∧ Nat.Prime r ∧ p ≠ q ∧ q ≠ r ∧ p ≠ r ∧ n = p * q * r) →
  (∃ m : ℕ, m > 0 ∧ 60 ∣ (m * n) ∧ ∀ k : ℕ, (k > 0 ∧ 60 ∣ (k * n)) → m ≤ k) →
  (∃ m : ℕ, m > 0 ∧ 60 ∣ (m * n) ∧ ∀ k : ℕ, (k > 0 ∧ 60 ∣ (k * n)) → m ≤ k) ∧ m = 60 :=
by sorry

end NUMINAMATH_CALUDE_multiple_of_n_divisible_by_60_l2873_287397


namespace NUMINAMATH_CALUDE_five_line_configurations_l2873_287357

/-- Represents a configuration of five lines in a plane -/
structure LineConfiguration where
  /-- The number of intersection points -/
  intersections : ℕ
  /-- The number of sets of parallel lines -/
  parallel_sets : ℕ

/-- The total count is the sum of intersection points and parallel sets -/
def total_count (config : LineConfiguration) : ℕ :=
  config.intersections + config.parallel_sets

/-- Possible configurations of five lines in a plane -/
def possible_configurations : List LineConfiguration :=
  [
    ⟨0, 1⟩,  -- All 5 lines parallel
    ⟨4, 1⟩,  -- 4 parallel lines and 1 intersecting
    ⟨6, 2⟩,  -- Two sets of parallel lines (2 and 3)
    ⟨7, 1⟩,  -- 3 parallel lines and 2 intersecting
    ⟨8, 2⟩,  -- Two pairs of parallel lines
    ⟨9, 1⟩,  -- 1 pair of parallel lines
    ⟨10, 0⟩  -- No parallel lines
  ]

theorem five_line_configurations :
  (possible_configurations.map total_count).toFinset = {1, 5, 8, 10} := by sorry

end NUMINAMATH_CALUDE_five_line_configurations_l2873_287357


namespace NUMINAMATH_CALUDE_sandy_marks_per_correct_sum_l2873_287391

theorem sandy_marks_per_correct_sum :
  ∀ (marks_per_correct : ℕ) (marks_per_incorrect : ℕ) (total_attempts : ℕ) (total_marks : ℕ) (correct_attempts : ℕ),
    marks_per_incorrect = 2 →
    total_attempts = 30 →
    total_marks = 50 →
    correct_attempts = 22 →
    marks_per_correct * correct_attempts - marks_per_incorrect * (total_attempts - correct_attempts) = total_marks →
    marks_per_correct = 3 := by
  sorry

end NUMINAMATH_CALUDE_sandy_marks_per_correct_sum_l2873_287391


namespace NUMINAMATH_CALUDE_area_less_than_one_third_l2873_287371

theorem area_less_than_one_third (a : ℝ) (h : 1 < a ∧ a < 2) : 
  let f (x : ℝ) := 1 - |x - 1|
  let g (x : ℝ) := |2*x - a|
  let area := (1/6) * |(a - 1)*(a - 2)|
  area < (1/3) :=
by sorry

end NUMINAMATH_CALUDE_area_less_than_one_third_l2873_287371


namespace NUMINAMATH_CALUDE_carries_strawberry_harvest_l2873_287322

/-- Represents a rectangular garden with strawberry plants -/
structure StrawberryGarden where
  length : ℝ
  width : ℝ
  plants_per_sqft : ℝ
  strawberries_per_plant : ℝ

/-- Calculates the expected total number of strawberries in the garden -/
def total_strawberries (garden : StrawberryGarden) : ℝ :=
  garden.length * garden.width * garden.plants_per_sqft * garden.strawberries_per_plant

/-- Theorem stating the expected number of strawberries in Carrie's garden -/
theorem carries_strawberry_harvest :
  let garden : StrawberryGarden := {
    length := 10,
    width := 15,
    plants_per_sqft := 5,
    strawberries_per_plant := 12
  }
  total_strawberries garden = 9000 := by
  sorry

end NUMINAMATH_CALUDE_carries_strawberry_harvest_l2873_287322
