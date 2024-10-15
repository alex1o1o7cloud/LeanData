import Mathlib

namespace NUMINAMATH_CALUDE_range_of_a_for_intersecting_circles_l2876_287678

/-- Two circles intersect if and only if the distance between their centers is greater than
    the absolute difference of their radii and less than the sum of their radii -/
axiom circles_intersect_iff_distance_between_centers (x₁ y₁ x₂ y₂ r₁ r₂ : ℝ) :
  ((x₁ - x₂)^2 + (y₁ - y₂)^2 > (r₁ - r₂)^2) ∧
  ((x₁ - x₂)^2 + (y₁ - y₂)^2 < (r₁ + r₂)^2)

/-- The range of a for intersecting circles -/
theorem range_of_a_for_intersecting_circles (a : ℝ) :
  (∃ x y : ℝ, (x + 2)^2 + (y - a)^2 = 1 ∧ (x - a)^2 + (y - 5)^2 = 16) ↔
  1 < a ∧ a < 2 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_for_intersecting_circles_l2876_287678


namespace NUMINAMATH_CALUDE_simplify_expression_l2876_287692

theorem simplify_expression : (625 : ℝ) ^ (1/4 : ℝ) * (125 : ℝ) ^ (1/3 : ℝ) = 25 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2876_287692


namespace NUMINAMATH_CALUDE_subset_family_bound_l2876_287691

theorem subset_family_bound (n k m : ℕ) (B : Fin m → Finset (Fin n)) :
  (∀ i, (B i).card = k) →
  (k ≥ 2) →
  (∀ i j, i < j → (B i ∩ B j).card ≤ 1) →
  m ≤ ⌊(n : ℝ) / k * ⌊(n - 1 : ℝ) / (k - 1)⌋⌋ :=
by sorry

end NUMINAMATH_CALUDE_subset_family_bound_l2876_287691


namespace NUMINAMATH_CALUDE_unique_four_digit_number_l2876_287610

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

def digit_sum (n : ℕ) : ℕ := 
  (n / 1000) + ((n / 100) % 10) + ((n / 10) % 10) + (n % 10)

def middle_digits_sum (n : ℕ) : ℕ := 
  ((n / 100) % 10) + ((n / 10) % 10)

def thousands_minus_units (n : ℕ) : ℤ := 
  (n / 1000 : ℤ) - (n % 10 : ℤ)

theorem unique_four_digit_number : 
  ∃! n : ℕ, 
    is_four_digit n ∧ 
    digit_sum n = 18 ∧ 
    middle_digits_sum n = 10 ∧ 
    thousands_minus_units n = 2 ∧ 
    n % 9 = 0 ∧
    n = 5643 :=
by
  sorry

end NUMINAMATH_CALUDE_unique_four_digit_number_l2876_287610


namespace NUMINAMATH_CALUDE_perpendicular_bisector_equation_l2876_287674

/-- The perpendicular bisector of a line segment with endpoints (x₁, y₁) and (x₂, y₂) -/
def perpendicular_bisector (x₁ y₁ x₂ y₂ : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - x₁)^2 + (p.2 - y₁)^2 = (p.1 - x₂)^2 + (p.2 - y₂)^2}

theorem perpendicular_bisector_equation :
  perpendicular_bisector 1 3 5 (-1) = {p : ℝ × ℝ | p.1 - p.2 - 2 = 0} := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_bisector_equation_l2876_287674


namespace NUMINAMATH_CALUDE_binomial_expansion_properties_l2876_287607

/-- Given the expansion of (1+2x)^10, prove properties about its coefficients -/
theorem binomial_expansion_properties :
  let n : ℕ := 10
  let expansion := fun (k : ℕ) => (n.choose k) * (2^k)
  let sum_first_three := 1 + 2 * n.choose 1 + 4 * n.choose 2
  -- Condition: sum of coefficients of first three terms is 201
  sum_first_three = 201 →
  -- 1. The binomial coefficient is largest for the 6th term
  (∀ k, k ≠ 5 → n.choose 5 ≥ n.choose k) ∧
  -- 2. The coefficient is largest for the 8th term
  (∀ k, k ≠ 7 → expansion 7 ≥ expansion k) :=
by sorry

end NUMINAMATH_CALUDE_binomial_expansion_properties_l2876_287607


namespace NUMINAMATH_CALUDE_find_m_l2876_287643

-- Define the function f
def f (x : ℝ) : ℝ := 4 * x + 7

-- State the theorem
theorem find_m (m : ℝ) : (∀ x, f (1/2 * x - 1) = 2 * x + 3) → f m = 6 → m = -1/4 := by
  sorry

end NUMINAMATH_CALUDE_find_m_l2876_287643


namespace NUMINAMATH_CALUDE_square_plus_reciprocal_square_l2876_287638

theorem square_plus_reciprocal_square (x : ℝ) (h : x + 1/x = 8) : x^2 + 1/x^2 = 62 := by
  sorry

end NUMINAMATH_CALUDE_square_plus_reciprocal_square_l2876_287638


namespace NUMINAMATH_CALUDE_sammy_has_twenty_caps_l2876_287664

/-- Represents the number of bottle caps each person has -/
structure BottleCaps where
  sammy : ℕ
  janine : ℕ
  billie : ℕ
  tommy : ℕ

/-- The initial state of bottle caps -/
def initial_state (b : ℕ) : BottleCaps :=
  { sammy := 3 * b + 2
    janine := 3 * b
    billie := b
    tommy := 0 }

/-- The final state of bottle caps after Billie's gift -/
def final_state (b : ℕ) : BottleCaps :=
  { sammy := 3 * b + 2
    janine := 3 * b
    billie := b - 4
    tommy := 4 }

/-- The theorem stating Sammy has 20 bottle caps -/
theorem sammy_has_twenty_caps :
  ∃ b : ℕ,
    (final_state b).tommy = 2 * (final_state b).billie ∧
    (final_state b).sammy = 20 := by
  sorry

#check sammy_has_twenty_caps

end NUMINAMATH_CALUDE_sammy_has_twenty_caps_l2876_287664


namespace NUMINAMATH_CALUDE_sin_fifth_power_coefficients_sum_of_squares_l2876_287669

theorem sin_fifth_power_coefficients_sum_of_squares :
  ∃ (b₁ b₂ b₃ b₄ b₅ : ℝ),
    (∀ θ : ℝ, (Real.sin θ)^5 = b₁ * Real.sin θ + b₂ * Real.sin (2 * θ) + b₃ * Real.sin (3 * θ) + b₄ * Real.sin (4 * θ) + b₅ * Real.sin (5 * θ)) →
    b₁^2 + b₂^2 + b₃^2 + b₄^2 + b₅^2 = 63 / 128 := by
  sorry

end NUMINAMATH_CALUDE_sin_fifth_power_coefficients_sum_of_squares_l2876_287669


namespace NUMINAMATH_CALUDE_line_hyperbola_intersection_l2876_287681

theorem line_hyperbola_intersection (k : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ < x₂ ∧ x₁ < -1 ∧ x₂ < -1 ∧
    (x₁^2 - (k*x₁ - 1)^2 = 1) ∧
    (x₂^2 - (k*x₂ - 1)^2 = 1)) ↔
  -Real.sqrt 2 < k ∧ k < -1 := by
sorry

end NUMINAMATH_CALUDE_line_hyperbola_intersection_l2876_287681


namespace NUMINAMATH_CALUDE_ball_probability_l2876_287697

theorem ball_probability (total : ℕ) (white green yellow red purple : ℕ) 
  (h_total : total = 100)
  (h_white : white = 50)
  (h_green : green = 20)
  (h_yellow : yellow = 10)
  (h_red : red = 17)
  (h_purple : purple = 3)
  (h_sum : white + green + yellow + red + purple = total) :
  (white + green + yellow : ℚ) / total = 0.8 := by
sorry

end NUMINAMATH_CALUDE_ball_probability_l2876_287697


namespace NUMINAMATH_CALUDE_expense_increase_percentage_l2876_287657

theorem expense_increase_percentage (salary : ℝ) (initial_savings_rate : ℝ) (new_savings : ℝ) :
  salary = 5500 →
  initial_savings_rate = 0.2 →
  new_savings = 220 →
  let initial_savings := salary * initial_savings_rate
  let initial_expenses := salary - initial_savings
  let expense_increase := initial_savings - new_savings
  (expense_increase / initial_expenses) * 100 = 20 := by
  sorry

end NUMINAMATH_CALUDE_expense_increase_percentage_l2876_287657


namespace NUMINAMATH_CALUDE_jeffs_towers_count_l2876_287631

/-- The number of sandcastles on Mark's beach -/
def marks_sandcastles : ℕ := 20

/-- The number of towers per sandcastle on Mark's beach -/
def marks_towers_per_castle : ℕ := 10

/-- The ratio of Jeff's sandcastles to Mark's sandcastles -/
def jeff_to_mark_ratio : ℕ := 3

/-- The total number of sandcastles and towers on both beaches -/
def total_objects : ℕ := 580

/-- The number of towers per sandcastle on Jeff's beach -/
def jeffs_towers_per_castle : ℕ := 5

theorem jeffs_towers_count : jeffs_towers_per_castle = 5 := by
  sorry

end NUMINAMATH_CALUDE_jeffs_towers_count_l2876_287631


namespace NUMINAMATH_CALUDE_rowing_conference_votes_l2876_287635

theorem rowing_conference_votes 
  (num_coaches : ℕ) 
  (num_rowers : ℕ) 
  (votes_per_coach : ℕ) 
  (h1 : num_coaches = 36) 
  (h2 : num_rowers = 60) 
  (h3 : votes_per_coach = 5) : 
  (num_coaches * votes_per_coach) / num_rowers = 3 :=
by sorry

end NUMINAMATH_CALUDE_rowing_conference_votes_l2876_287635


namespace NUMINAMATH_CALUDE_sum_of_fractions_equals_two_ninths_l2876_287687

theorem sum_of_fractions_equals_two_ninths :
  (1 / (3 * 4 : ℚ)) + (1 / (4 * 5 : ℚ)) + (1 / (5 * 6 : ℚ)) +
  (1 / (6 * 7 : ℚ)) + (1 / (7 * 8 : ℚ)) + (1 / (8 * 9 : ℚ)) = 2 / 9 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fractions_equals_two_ninths_l2876_287687


namespace NUMINAMATH_CALUDE_journey_speed_calculation_l2876_287606

/-- Proves that given a journey of 12 hours covering 560 km, where the first half of the distance 
is traveled at 35 kmph, the speed for the second half of the journey is 70 kmph. -/
theorem journey_speed_calculation (total_time : ℝ) (total_distance : ℝ) (first_half_speed : ℝ) :
  total_time = 12 →
  total_distance = 560 →
  first_half_speed = 35 →
  (total_distance / 2) / first_half_speed + (total_distance / 2) / ((total_distance / 2) / (total_time - (total_distance / 2) / first_half_speed)) = total_time →
  (total_distance / 2) / (total_time - (total_distance / 2) / first_half_speed) = 70 := by
  sorry

#check journey_speed_calculation

end NUMINAMATH_CALUDE_journey_speed_calculation_l2876_287606


namespace NUMINAMATH_CALUDE_union_of_M_and_N_l2876_287658

-- Define the sets M and N
def M : Set ℕ := {a : ℕ | a = 0 ∨ ∃ x, x = a}
def N : Set ℕ := {1, 2}

-- State the theorem
theorem union_of_M_and_N :
  (∃ a : ℕ, M = {a, 0}) →  -- M = {a, 0}
  N = {1, 2} →             -- N = {1, 2}
  M ∩ N = {1} →            -- M ∩ N = {1}
  M ∪ N = {0, 1, 2} :=     -- M ∪ N = {0, 1, 2}
by sorry

end NUMINAMATH_CALUDE_union_of_M_and_N_l2876_287658


namespace NUMINAMATH_CALUDE_cyclist_speed_l2876_287625

/-- The cyclist's problem -/
theorem cyclist_speed :
  ∀ (expected_speed actual_speed : ℝ),
  expected_speed > 0 →
  actual_speed > 0 →
  actual_speed = expected_speed + 1 →
  96 / actual_speed = 96 / expected_speed - 2 →
  96 / expected_speed = 1.25 →
  actual_speed = 16 := by
  sorry

end NUMINAMATH_CALUDE_cyclist_speed_l2876_287625


namespace NUMINAMATH_CALUDE_triangle_formation_l2876_287677

/-- Triangle inequality theorem check for three sides --/
def triangle_inequality (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

/-- Check if three lengths can form a triangle --/
def can_form_triangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ triangle_inequality a b c

theorem triangle_formation :
  can_form_triangle 2 3 4 ∧
  ¬can_form_triangle 5 1 3 ∧
  ¬can_form_triangle 2 4 2 ∧
  ¬can_form_triangle 3 3 7 :=
by sorry

end NUMINAMATH_CALUDE_triangle_formation_l2876_287677


namespace NUMINAMATH_CALUDE_no_even_three_digit_sum_27_l2876_287616

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def digit_sum (n : ℕ) : ℕ :=
  (n / 100) + ((n / 10) % 10) + (n % 10)

theorem no_even_three_digit_sum_27 :
  ¬ ∃ n : ℕ, is_three_digit n ∧ digit_sum n = 27 ∧ Even n :=
sorry

end NUMINAMATH_CALUDE_no_even_three_digit_sum_27_l2876_287616


namespace NUMINAMATH_CALUDE_quadrilateral_sum_l2876_287685

/-- Given a quadrilateral PQRS with vertices P(a, b), Q(a, -b), R(-a, -b), and S(-a, b),
    where a and b are positive integers and a > b, if the area of PQRS is 32,
    then a + b = 5. -/
theorem quadrilateral_sum (a b : ℕ) (ha : a > b) (hb : b > 0)
  (harea : (2 * a) * (2 * b) = 32) : a + b = 5 := by
  sorry

#check quadrilateral_sum

end NUMINAMATH_CALUDE_quadrilateral_sum_l2876_287685


namespace NUMINAMATH_CALUDE_square_diff_roots_l2876_287667

theorem square_diff_roots : 
  (Real.sqrt (625681 + 1000) - Real.sqrt 1000)^2 = 
    626681 - 2 * Real.sqrt 626681 * 31.622776601683793 + 1000 := by
  sorry

end NUMINAMATH_CALUDE_square_diff_roots_l2876_287667


namespace NUMINAMATH_CALUDE_factorization_proof_l2876_287647

theorem factorization_proof (x y : ℝ) : 
  y^2 * (x - 2) + 16 * (2 - x) = (x - 2) * (y + 4) * (y - 4) := by
  sorry

end NUMINAMATH_CALUDE_factorization_proof_l2876_287647


namespace NUMINAMATH_CALUDE_train_length_l2876_287603

/-- Given a train that crosses a tree in 120 seconds and takes 200 seconds to pass
    a platform 800 m long, moving at a constant speed, prove that the length of the train
    is 1200 meters. -/
theorem train_length (tree_time platform_time platform_length : ℝ)
    (h1 : tree_time = 120)
    (h2 : platform_time = 200)
    (h3 : platform_length = 800) :
  let train_length := (platform_time * platform_length) / (platform_time - tree_time)
  train_length = 1200 := by
  sorry

end NUMINAMATH_CALUDE_train_length_l2876_287603


namespace NUMINAMATH_CALUDE_mikes_total_spending_l2876_287639

/-- Represents Mike's shopping expenses -/
structure ShoppingExpenses where
  food : ℝ
  wallet : ℝ
  shirt : ℝ

/-- Calculates the total spending given Mike's shopping expenses -/
def totalSpending (expenses : ShoppingExpenses) : ℝ :=
  expenses.food + expenses.wallet + expenses.shirt

/-- Theorem stating Mike's total spending given the problem conditions -/
theorem mikes_total_spending :
  ∀ (expenses : ShoppingExpenses),
    expenses.food = 30 →
    expenses.wallet = expenses.food + 60 →
    expenses.shirt = expenses.wallet / 3 →
    totalSpending expenses = 150 := by
  sorry


end NUMINAMATH_CALUDE_mikes_total_spending_l2876_287639


namespace NUMINAMATH_CALUDE_total_cost_with_tax_l2876_287623

def earbuds_cost : ℝ := 200
def smartwatch_cost : ℝ := 300
def earbuds_tax_rate : ℝ := 0.15
def smartwatch_tax_rate : ℝ := 0.12

theorem total_cost_with_tax : 
  earbuds_cost * (1 + earbuds_tax_rate) + smartwatch_cost * (1 + smartwatch_tax_rate) = 566 := by
  sorry

end NUMINAMATH_CALUDE_total_cost_with_tax_l2876_287623


namespace NUMINAMATH_CALUDE_intersection_and_lines_l2876_287689

-- Define the lines l₁ and l₂
def l₁ (x y : ℝ) : Prop := 2 * x - y + 7 = 0
def l₂ (x y : ℝ) : Prop := x + y - 1 = 0

-- Define the intersection point A(m, n)
def m : ℝ := -2
def n : ℝ := 3

-- Define line l
def l (x y : ℝ) : Prop := 2 * x - 3 * y - 1 = 0

-- Theorem statement
theorem intersection_and_lines :
  (l₁ m n ∧ l₂ m n) ∧  -- A is the intersection of l₁ and l₂
  (∀ x y : ℝ, x + 2 * y - 4 = 0 ↔ (x - m) * 2 + (y - n) * 1 = 0) ∧  -- l₃ equation
  (∀ x y : ℝ, 2 * x - 3 * y + 13 = 0 ↔ (y - n) = (2 / 3) * (x - m)) :=  -- l₄ equation
by sorry

end NUMINAMATH_CALUDE_intersection_and_lines_l2876_287689


namespace NUMINAMATH_CALUDE_three_lines_exist_l2876_287688

/-- A line in the 2D plane represented by its slope and y-intercept -/
structure Line where
  slope : Option ℝ
  intercept : ℝ

/-- The hyperbola x^2 - y^2 = 2 -/
def hyperbola (x y : ℝ) : Prop := x^2 - y^2 = 2

/-- A line passes through the point (√2, 0) -/
def passesThrough (l : Line) : Prop :=
  match l.slope with
  | none => l.intercept = Real.sqrt 2
  | some m => l.intercept = -m * Real.sqrt 2

/-- A line has exactly one common point with the hyperbola -/
def hasOneCommonPoint (l : Line) : Prop :=
  ∃! p : ℝ × ℝ, (match l.slope with
    | none => p.1 = l.intercept ∧ p.2 = 0
    | some m => p.2 = m * p.1 + l.intercept) ∧ hyperbola p.1 p.2

/-- The main theorem: there are exactly 3 lines satisfying the conditions -/
theorem three_lines_exist :
  ∃! (lines : Finset Line), lines.card = 3 ∧
    ∀ l ∈ lines, passesThrough l ∧ hasOneCommonPoint l :=
sorry

end NUMINAMATH_CALUDE_three_lines_exist_l2876_287688


namespace NUMINAMATH_CALUDE_smallest_zucchini_count_l2876_287686

def is_divisible_by_6 (n : ℕ) : Prop := ∃ k : ℕ, n = 6 * k

def is_perfect_square (n : ℕ) : Prop := ∃ k : ℕ, n = k * k

def is_perfect_cube (n : ℕ) : Prop := ∃ k : ℕ, n = k * k * k

def satisfies_conditions (n : ℕ) : Prop :=
  is_divisible_by_6 n ∧ is_perfect_square (n / 2) ∧ is_perfect_cube (n / 3)

theorem smallest_zucchini_count :
  satisfies_conditions 648 ∧ ∀ m : ℕ, m < 648 → ¬(satisfies_conditions m) :=
by sorry

end NUMINAMATH_CALUDE_smallest_zucchini_count_l2876_287686


namespace NUMINAMATH_CALUDE_expected_sixes_is_one_third_l2876_287613

-- Define a die as having 6 sides
def die_sides : ℕ := 6

-- Define the probability of rolling a 6 on a single die
def prob_six : ℚ := 1 / die_sides

-- Define the probability of not rolling a 6 on a single die
def prob_not_six : ℚ := 1 - prob_six

-- Define the expected number of 6's when rolling two dice
def expected_sixes : ℚ := 
  2 * (prob_six * prob_six) + 
  1 * (2 * prob_six * prob_not_six) + 
  0 * (prob_not_six * prob_not_six)

-- Theorem statement
theorem expected_sixes_is_one_third : expected_sixes = 1 / 3 :=
sorry

end NUMINAMATH_CALUDE_expected_sixes_is_one_third_l2876_287613


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l2876_287672

-- Define the sets A and B
def A : Set ℝ := {x | -1 < x ∧ x < 1}
def B : Set ℝ := {x | x > 0}

-- State the theorem
theorem intersection_of_A_and_B : A ∩ B = {x : ℝ | 0 < x ∧ x < 1} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l2876_287672


namespace NUMINAMATH_CALUDE_fraction_zero_l2876_287660

theorem fraction_zero (x : ℝ) (h : x ≠ 1) : (x^2 - 1) / (x - 1) = 0 ↔ x = -1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_zero_l2876_287660


namespace NUMINAMATH_CALUDE_range_of_3x_plus_2y_l2876_287611

theorem range_of_3x_plus_2y (x y : ℝ) 
  (h1 : 1 ≤ x + y) (h2 : x + y ≤ 3) 
  (h3 : -1 ≤ x - y) (h4 : x - y ≤ 4) : 
  2 ≤ 3*x + 2*y ∧ 3*x + 2*y ≤ 9.5 :=
by sorry

end NUMINAMATH_CALUDE_range_of_3x_plus_2y_l2876_287611


namespace NUMINAMATH_CALUDE_binomial_constant_term_l2876_287651

theorem binomial_constant_term (n : ℕ) : 
  (∃ r : ℕ, r ≤ n ∧ 4*n = 5*r) ↔ n = 10 := by sorry

end NUMINAMATH_CALUDE_binomial_constant_term_l2876_287651


namespace NUMINAMATH_CALUDE_book_sale_revenue_l2876_287652

/-- Given a collection of books where 2/3 were sold for $3.50 each and 40 remained unsold,
    prove that the total amount received for the sold books is $280. -/
theorem book_sale_revenue (total_books : ℕ) (price_per_book : ℚ) :
  (2 : ℚ) / 3 * total_books + 40 = total_books →
  price_per_book = (7 : ℚ) / 2 →
  ((2 : ℚ) / 3 * total_books) * price_per_book = 280 := by
  sorry

end NUMINAMATH_CALUDE_book_sale_revenue_l2876_287652


namespace NUMINAMATH_CALUDE_fruit_candy_cost_difference_l2876_287670

/-- The cost difference between two schools purchasing fruit candy --/
theorem fruit_candy_cost_difference : 
  let school_a_quantity : ℝ := 56
  let school_a_price_per_kg : ℝ := 8.06
  let price_reduction : ℝ := 0.56
  let free_candy_percentage : ℝ := 0.05
  
  let school_b_price_per_kg : ℝ := school_a_price_per_kg - price_reduction
  let school_b_quantity : ℝ := school_a_quantity / (1 + free_candy_percentage)
  
  let school_a_total_cost : ℝ := school_a_quantity * school_a_price_per_kg
  let school_b_total_cost : ℝ := school_b_quantity * school_b_price_per_kg
  
  school_a_total_cost - school_b_total_cost = 51.36 := by
  sorry

end NUMINAMATH_CALUDE_fruit_candy_cost_difference_l2876_287670


namespace NUMINAMATH_CALUDE_inequality_solution_set_l2876_287668

theorem inequality_solution_set (x : ℝ) : (3 * x - 1) / (2 - x) ≥ 1 ↔ 3 / 4 ≤ x ∧ x < 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l2876_287668


namespace NUMINAMATH_CALUDE_orange_ring_weight_l2876_287624

/-- The weight of the orange ring in an experiment -/
theorem orange_ring_weight (purple_weight white_weight total_weight : ℚ)
  (h1 : purple_weight = 33/100)
  (h2 : white_weight = 21/50)
  (h3 : total_weight = 83/100) :
  total_weight - (purple_weight + white_weight) = 2/25 := by
  sorry

#eval (83/100 : ℚ) - ((33/100 : ℚ) + (21/50 : ℚ))

end NUMINAMATH_CALUDE_orange_ring_weight_l2876_287624


namespace NUMINAMATH_CALUDE_inverse_f_128_l2876_287656

/-- Given a function f: ℝ → ℝ satisfying f(4) = 2 and f(2x) = 2f(x) for all x,
    prove that f⁻¹(128) = 256 -/
theorem inverse_f_128 (f : ℝ → ℝ) (h1 : f 4 = 2) (h2 : ∀ x, f (2 * x) = 2 * f x) :
  f⁻¹ 128 = 256 := by
  sorry

end NUMINAMATH_CALUDE_inverse_f_128_l2876_287656


namespace NUMINAMATH_CALUDE_gabby_shopping_funds_l2876_287626

theorem gabby_shopping_funds (total_cost available_funds : ℕ) : 
  total_cost = 165 → available_funds = 110 → total_cost - available_funds = 55 := by
  sorry

end NUMINAMATH_CALUDE_gabby_shopping_funds_l2876_287626


namespace NUMINAMATH_CALUDE_triangle_properties_l2876_287619

-- Define the triangle ABC
def A : ℝ × ℝ := (-4, 0)
def B : ℝ × ℝ := (0, 2)
def C : ℝ × ℝ := (2, -2)

-- Define the altitude line equation
def altitude_equation (x y : ℝ) : Prop :=
  2 * x + y - 2 = 0

-- Define the circumcircle equation
def circumcircle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 + 4 * x + 4 * y - 8 = 0

-- Theorem statement
theorem triangle_properties :
  (∀ x y : ℝ, altitude_equation x y ↔ 
    (x - A.1) * (B.2 - C.2) = (y - A.2) * (B.1 - C.1)) ∧
  (∀ x y : ℝ, circumcircle_equation x y ↔ 
    (x - A.1)^2 + (y - A.2)^2 = (x - B.1)^2 + (y - B.2)^2 ∧
    (x - B.1)^2 + (y - B.2)^2 = (x - C.1)^2 + (y - C.2)^2) :=
by sorry

end NUMINAMATH_CALUDE_triangle_properties_l2876_287619


namespace NUMINAMATH_CALUDE_simplify_fraction_l2876_287632

theorem simplify_fraction : (121 : ℚ) / 13310 = 1 / 110 := by sorry

end NUMINAMATH_CALUDE_simplify_fraction_l2876_287632


namespace NUMINAMATH_CALUDE_chess_points_theorem_l2876_287684

theorem chess_points_theorem :
  ∃! (s : Finset ℕ), s.card = 2 ∧
  (∀ x ∈ s, ∃ n : ℕ, 11 * n + x * (100 - n) = 800) ∧
  (3 ∈ s ∧ 4 ∈ s) := by
  sorry

end NUMINAMATH_CALUDE_chess_points_theorem_l2876_287684


namespace NUMINAMATH_CALUDE_f_composition_negative_three_l2876_287676

noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 0 then 1 / (5 - x) else Real.log x / Real.log 4

theorem f_composition_negative_three (f : ℝ → ℝ) :
  (∀ x ≤ 0, f x = 1 / (5 - x)) →
  (∀ x > 0, f x = Real.log x / Real.log 4) →
  f (f (-3)) = -3/2 := by
  sorry

end NUMINAMATH_CALUDE_f_composition_negative_three_l2876_287676


namespace NUMINAMATH_CALUDE_microtron_stock_price_l2876_287648

/-- Represents the stock market scenario with Microtron and Dynaco stocks -/
structure StockMarket where
  microtron_price : ℝ
  dynaco_price : ℝ
  total_shares_sold : ℕ
  average_price : ℝ
  dynaco_shares_sold : ℕ

/-- Theorem stating the price of Microtron stock given the market conditions -/
theorem microtron_stock_price (market : StockMarket) 
  (h1 : market.dynaco_price = 44)
  (h2 : market.total_shares_sold = 300)
  (h3 : market.average_price = 40)
  (h4 : market.dynaco_shares_sold = 150) :
  market.microtron_price = 36 := by
  sorry

end NUMINAMATH_CALUDE_microtron_stock_price_l2876_287648


namespace NUMINAMATH_CALUDE_exists_abs_leq_zero_l2876_287698

theorem exists_abs_leq_zero : ∃ x : ℝ, |x| ≤ 0 := by
  sorry

end NUMINAMATH_CALUDE_exists_abs_leq_zero_l2876_287698


namespace NUMINAMATH_CALUDE_gcd_12345_6789_l2876_287633

theorem gcd_12345_6789 : Nat.gcd 12345 6789 = 3 := by
  sorry

end NUMINAMATH_CALUDE_gcd_12345_6789_l2876_287633


namespace NUMINAMATH_CALUDE_gcf_5_factorial_6_factorial_l2876_287601

def factorial (n : ℕ) : ℕ := Nat.factorial n

theorem gcf_5_factorial_6_factorial : 
  Nat.gcd (factorial 5) (factorial 6) = factorial 5 := by
  sorry

end NUMINAMATH_CALUDE_gcf_5_factorial_6_factorial_l2876_287601


namespace NUMINAMATH_CALUDE_rectangular_solid_diagonal_l2876_287679

/-- Given a rectangular solid with side lengths x, y, and z, 
    if the surface area is 11 and the sum of the lengths of all edges is 24, 
    then the length of one of its diagonals is 5. -/
theorem rectangular_solid_diagonal 
  (x y z : ℝ) 
  (h1 : 2*x*y + 2*y*z + 2*x*z = 11) 
  (h2 : 4*(x + y + z) = 24) : 
  Real.sqrt (x^2 + y^2 + z^2) = 5 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_solid_diagonal_l2876_287679


namespace NUMINAMATH_CALUDE_latch_caught_14_necklaces_l2876_287693

/-- The number of necklaces caught by Boudreaux -/
def boudreaux_necklaces : ℕ := 12

/-- The number of necklaces caught by Rhonda -/
def rhonda_necklaces : ℕ := boudreaux_necklaces / 2

/-- The number of necklaces caught by Latch -/
def latch_necklaces : ℕ := 3 * rhonda_necklaces - 4

/-- Theorem stating that Latch caught 14 necklaces -/
theorem latch_caught_14_necklaces : latch_necklaces = 14 := by
  sorry

end NUMINAMATH_CALUDE_latch_caught_14_necklaces_l2876_287693


namespace NUMINAMATH_CALUDE_train_platform_crossing_time_l2876_287614

/-- Calculates the time required for a train to cross a platform -/
theorem train_platform_crossing_time
  (train_speed_kmph : ℝ)
  (train_speed_ms : ℝ)
  (time_to_pass_man : ℝ)
  (platform_length : ℝ)
  (h1 : train_speed_kmph = 72)
  (h2 : train_speed_ms = 20)
  (h3 : time_to_pass_man = 16)
  (h4 : platform_length = 280)
  (h5 : train_speed_ms = train_speed_kmph * 1000 / 3600) :
  let train_length := train_speed_ms * time_to_pass_man
  let total_distance := train_length + platform_length
  total_distance / train_speed_ms = 30 := by
  sorry

end NUMINAMATH_CALUDE_train_platform_crossing_time_l2876_287614


namespace NUMINAMATH_CALUDE_student_distribution_l2876_287645

/-- The number of ways to distribute n indistinguishable objects into k distinct boxes,
    with at least one object in each box -/
def distribute (n k : ℕ) : ℕ := sorry

/-- There are 5 college students -/
def num_students : ℕ := 5

/-- There are 3 factories -/
def num_factories : ℕ := 3

/-- The theorem stating that there are 150 ways to distribute 5 students among 3 factories
    with at least one student in each factory -/
theorem student_distribution : distribute num_students num_factories = 150 := by sorry

end NUMINAMATH_CALUDE_student_distribution_l2876_287645


namespace NUMINAMATH_CALUDE_ellipse_semi_minor_axis_l2876_287650

/-- Given an ellipse with center at the origin, one focus at (0, -2), and one endpoint
    of a semi-major axis at (0, 5), its semi-minor axis has length √21. -/
theorem ellipse_semi_minor_axis (c a b : ℝ) : 
  c = 2 →  -- distance from center to focus
  a = 5 →  -- length of semi-major axis
  b^2 = a^2 - c^2 →  -- relationship between a, b, and c in an ellipse
  b = Real.sqrt 21 := by
sorry

end NUMINAMATH_CALUDE_ellipse_semi_minor_axis_l2876_287650


namespace NUMINAMATH_CALUDE_sum_of_cubes_l2876_287637

theorem sum_of_cubes (a b : ℝ) (h1 : a + b = 10) (h2 : a * b = 17) : a^3 + b^3 = 490 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_cubes_l2876_287637


namespace NUMINAMATH_CALUDE_multiplication_addition_equality_l2876_287602

theorem multiplication_addition_equality : 85 * 1500 + (1 / 2) * 1500 = 128250 := by
  sorry

end NUMINAMATH_CALUDE_multiplication_addition_equality_l2876_287602


namespace NUMINAMATH_CALUDE_origin_not_in_convex_hull_probability_l2876_287642

/-- The unit circle in the complex plane -/
def S1 : Set ℂ := {z : ℂ | Complex.abs z = 1}

/-- The probability that the origin is not contained in the convex hull of n randomly selected points from S¹ -/
noncomputable def probability (n : ℕ) : ℝ := 1 - (n : ℝ) / 2^(n - 1)

/-- Theorem: The probability that the origin is not contained in the convex hull of seven randomly selected points from S¹ is 57/64 -/
theorem origin_not_in_convex_hull_probability :
  probability 7 = 57 / 64 := by sorry

end NUMINAMATH_CALUDE_origin_not_in_convex_hull_probability_l2876_287642


namespace NUMINAMATH_CALUDE_compound_oxygen_atoms_l2876_287690

/-- Represents the atomic weights of elements in atomic mass units (amu) -/
structure AtomicWeight where
  Cu : ℝ
  C : ℝ
  O : ℝ

/-- Represents a compound with Cu, C, and O atoms -/
structure Compound where
  Cu : ℕ
  C : ℕ
  O : ℕ

/-- Calculates the molecular weight of a compound given atomic weights -/
def molecularWeight (c : Compound) (w : AtomicWeight) : ℝ :=
  c.Cu * w.Cu + c.C * w.C + c.O * w.O

/-- Theorem stating that a compound with 1 Cu, 1 C, and n O atoms
    with a molecular weight of 124 amu has 3 O atoms -/
theorem compound_oxygen_atoms
  (w : AtomicWeight)
  (h1 : w.Cu = 63.55)
  (h2 : w.C = 12.01)
  (h3 : w.O = 16.00)
  (c : Compound)
  (h4 : c.Cu = 1)
  (h5 : c.C = 1)
  (h6 : molecularWeight c w = 124) :
  c.O = 3 := by
  sorry

end NUMINAMATH_CALUDE_compound_oxygen_atoms_l2876_287690


namespace NUMINAMATH_CALUDE_red_marbles_count_l2876_287661

theorem red_marbles_count :
  ∀ (total blue red yellow : ℕ),
    total = 85 →
    blue = 3 * red →
    yellow = 29 →
    total = red + blue + yellow →
    red = 14 := by
  sorry

end NUMINAMATH_CALUDE_red_marbles_count_l2876_287661


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l2876_287683

def M : Set ℝ := {x | x^2 ≥ 4}
def N : Set ℝ := {-3, 0, 1, 3, 4}

theorem intersection_of_M_and_N :
  M ∩ N = {-3, 3, 4} := by sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l2876_287683


namespace NUMINAMATH_CALUDE_translation_point_difference_l2876_287630

/-- Given points A and B, their translations A₁ and B₁, prove that a - b = -8 -/
theorem translation_point_difference (A B A₁ B₁ : ℝ × ℝ) (a b : ℝ) 
  (h1 : A = (1, -3))
  (h2 : B = (2, 1))
  (h3 : A₁ = (a, 2))
  (h4 : B₁ = (-1, b))
  (h5 : ∃ (v : ℝ × ℝ), A₁ = A + v ∧ B₁ = B + v) :
  a - b = -8 := by
  sorry

end NUMINAMATH_CALUDE_translation_point_difference_l2876_287630


namespace NUMINAMATH_CALUDE_rebus_unique_solution_l2876_287696

/-- Represents a four-digit number ABCD where A, B, C, D are distinct non-zero digits. -/
structure Rebus where
  a : Nat
  b : Nat
  c : Nat
  d : Nat
  h_distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d
  h_nonzero : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0
  h_digits : a < 10 ∧ b < 10 ∧ c < 10 ∧ d < 10

/-- The rebus equation ABCA = 182 * CD holds. -/
def rebusEquation (r : Rebus) : Prop :=
  1000 * r.a + 100 * r.b + 10 * r.c + r.a = 182 * (10 * r.c + r.d)

/-- The unique solution to the rebus is 2916. -/
theorem rebus_unique_solution :
  ∃! r : Rebus, rebusEquation r ∧ r.a = 2 ∧ r.b = 9 ∧ r.c = 1 ∧ r.d = 6 := by
  sorry

end NUMINAMATH_CALUDE_rebus_unique_solution_l2876_287696


namespace NUMINAMATH_CALUDE_max_leap_years_in_period_l2876_287628

/-- A calendrical system where leap years occur every 5 years -/
structure CalendarSystem where
  leap_year_interval : ℕ
  leap_year_interval_eq : leap_year_interval = 5

/-- The number of years in the period we're considering -/
def period_length : ℕ := 200

/-- The maximum number of leap years in the given period -/
def max_leap_years (c : CalendarSystem) : ℕ := period_length / c.leap_year_interval

/-- Theorem stating that the maximum number of leap years in a 200-year period is 40 -/
theorem max_leap_years_in_period (c : CalendarSystem) : max_leap_years c = 40 := by
  sorry

end NUMINAMATH_CALUDE_max_leap_years_in_period_l2876_287628


namespace NUMINAMATH_CALUDE_cubic_root_sum_l2876_287604

theorem cubic_root_sum (p q r : ℝ) : 
  p^3 - 8*p^2 + 9*p - 3 = 0 →
  q^3 - 8*q^2 + 9*q - 3 = 0 →
  r^3 - 8*r^2 + 9*r - 3 = 0 →
  p/(q*r + 1) + q/(p*r + 1) + r/(p*q + 1) = 83/43 := by
sorry

end NUMINAMATH_CALUDE_cubic_root_sum_l2876_287604


namespace NUMINAMATH_CALUDE_diagonal_pigeonhole_l2876_287655

/-- The number of vertices in the regular polygon -/
def n : ℕ := 2017

/-- The number of distinct diagonal lengths in a regular n-gon -/
def distinct_lengths (n : ℕ) : ℕ := (n - 3) / 2

/-- The smallest number of diagonals to guarantee two of the same length -/
def smallest_n (n : ℕ) : ℕ := distinct_lengths n + 1

theorem diagonal_pigeonhole :
  smallest_n n = 1008 :=
sorry

end NUMINAMATH_CALUDE_diagonal_pigeonhole_l2876_287655


namespace NUMINAMATH_CALUDE_largest_sum_is_253_33_l2876_287620

/-- Represents a trapezium ABCD with specific angle properties -/
structure Trapezium where
  -- Internal angles in arithmetic progression
  b : ℝ
  e : ℝ
  -- Smallest angle is 35°
  smallest_angle : b = 35
  -- Sum of internal angles is 360°
  angle_sum : 4 * b + 6 * e = 360

/-- The largest possible sum of the two largest angles in the trapezium -/
def largest_sum_of_two_largest_angles (t : Trapezium) : ℝ :=
  2 * t.b + 5 * t.e

/-- Theorem stating the largest possible sum of the two largest angles -/
theorem largest_sum_is_253_33 (t : Trapezium) :
  largest_sum_of_two_largest_angles t = 253.33 := by
  sorry


end NUMINAMATH_CALUDE_largest_sum_is_253_33_l2876_287620


namespace NUMINAMATH_CALUDE_characterization_of_brazilian_triples_l2876_287680

def is_brazilian (a b c : ℕ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧
  (∃ k : ℕ, b * c + 1 = k * a) ∧
  (∃ k : ℕ, a * c + 1 = k * b) ∧
  (∃ k : ℕ, a * b + 1 = k * c)

def brazilian_triples : Set (ℕ × ℕ × ℕ) :=
  {(3, 2, 1), (2, 3, 1), (1, 3, 2), (2, 1, 3), (1, 2, 3), (3, 1, 2),
   (7, 3, 2), (3, 7, 2), (2, 7, 3), (3, 2, 7), (2, 3, 7), (7, 2, 3),
   (2, 1, 1), (1, 2, 1), (1, 1, 2),
   (1, 1, 1)}

theorem characterization_of_brazilian_triples :
  ∀ a b c : ℕ, is_brazilian a b c ↔ (a, b, c) ∈ brazilian_triples :=
sorry

end NUMINAMATH_CALUDE_characterization_of_brazilian_triples_l2876_287680


namespace NUMINAMATH_CALUDE_B_completes_work_in_8_days_l2876_287605

/-- The number of days B takes to complete the work alone -/
def B : ℕ := 8

/-- The rate at which A completes the work -/
def rate_A : ℚ := 1 / 20

/-- The rate at which B completes the work -/
def rate_B : ℚ := 1 / B

/-- The amount of work completed by A and B together in 3 days -/
def work_together : ℚ := 3 * (rate_A + rate_B)

/-- The amount of work completed by B alone in 3 days -/
def work_B_alone : ℚ := 3 * rate_B

theorem B_completes_work_in_8_days :
  work_together + work_B_alone = 1 ∧ B = 8 := by
  sorry

end NUMINAMATH_CALUDE_B_completes_work_in_8_days_l2876_287605


namespace NUMINAMATH_CALUDE_negation_equivalence_l2876_287666

theorem negation_equivalence :
  (¬ ∃ x : ℝ, x > 1 ∧ x^2 + 2 < 0) ↔ (∀ x : ℝ, x > 1 → x^2 + 2 ≥ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_equivalence_l2876_287666


namespace NUMINAMATH_CALUDE_joan_remaining_kittens_l2876_287636

def initial_kittens : ℕ := 8
def kittens_given_away : ℕ := 2

theorem joan_remaining_kittens :
  initial_kittens - kittens_given_away = 6 := by
  sorry

end NUMINAMATH_CALUDE_joan_remaining_kittens_l2876_287636


namespace NUMINAMATH_CALUDE_twelfth_term_equals_three_over_512_l2876_287646

/-- The nth term of a geometric sequence -/
def geometricSequenceTerm (a₁ : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a₁ * r^(n - 1)

/-- The 12th term of the specific geometric sequence -/
def twelfthTerm : ℚ :=
  geometricSequenceTerm 12 (1/2) 12

theorem twelfth_term_equals_three_over_512 :
  twelfthTerm = 3/512 := by
  sorry

end NUMINAMATH_CALUDE_twelfth_term_equals_three_over_512_l2876_287646


namespace NUMINAMATH_CALUDE_cube_distance_to_plane_l2876_287653

/-- Given a cube with side length 10 and three vertices adjacent to the closest vertex A
    at heights 10, 11, and 12 above a plane, prove that the distance from A to the plane
    is (33-√294)/3 -/
theorem cube_distance_to_plane (cube_side : ℝ) (height_1 height_2 height_3 : ℝ) :
  cube_side = 10 →
  height_1 = 10 →
  height_2 = 11 →
  height_3 = 12 →
  ∃ (distance : ℝ), distance = (33 - Real.sqrt 294) / 3 ∧
    distance = min height_1 (min height_2 height_3) - 
      Real.sqrt ((cube_side^2 - (height_2 - height_1)^2) / 4 +
                 (cube_side^2 - (height_3 - height_1)^2) / 4 +
                 (cube_side^2 - (height_3 - height_2)^2) / 4) := by
  sorry

end NUMINAMATH_CALUDE_cube_distance_to_plane_l2876_287653


namespace NUMINAMATH_CALUDE_gcd_lcm_product_24_54_l2876_287617

theorem gcd_lcm_product_24_54 : Nat.gcd 24 54 * Nat.lcm 24 54 = 1296 := by
  sorry

end NUMINAMATH_CALUDE_gcd_lcm_product_24_54_l2876_287617


namespace NUMINAMATH_CALUDE_exists_number_divisible_by_2n_with_only_1_and_2_l2876_287621

/-- A function that checks if a natural number only contains digits 1 and 2 in its decimal representation -/
def onlyOneAndTwo (n : ℕ) : Prop :=
  ∀ d : ℕ, d ∈ n.digits 10 → d = 1 ∨ d = 2

/-- For every natural number n, there exists a number divisible by 2^n 
    whose decimal representation uses only the digits 1 and 2 -/
theorem exists_number_divisible_by_2n_with_only_1_and_2 :
  ∀ n : ℕ, ∃ N : ℕ, 2^n ∣ N ∧ onlyOneAndTwo N :=
by sorry

end NUMINAMATH_CALUDE_exists_number_divisible_by_2n_with_only_1_and_2_l2876_287621


namespace NUMINAMATH_CALUDE_feed_animals_count_l2876_287663

/-- Represents the number of pairs of animals in the zoo -/
def num_pairs : ℕ := 5

/-- Calculates the number of ways to feed all animals in the zoo -/
def feed_animals : ℕ :=
  (num_pairs) *  -- Choose from 5 females
  (num_pairs - 1) * (num_pairs - 1) *  -- Choose from 4 males, then 4 females
  (num_pairs - 2) * (num_pairs - 2) *  -- Choose from 3 males, then 3 females
  (num_pairs - 3) * (num_pairs - 3) *  -- Choose from 2 males, then 2 females
  (num_pairs - 4) * (num_pairs - 4)    -- Choose from 1 male, then 1 female

/-- Theorem stating the number of ways to feed all animals -/
theorem feed_animals_count : feed_animals = 2880 := by
  sorry

end NUMINAMATH_CALUDE_feed_animals_count_l2876_287663


namespace NUMINAMATH_CALUDE_trapezium_side_length_first_parallel_side_length_l2876_287673

theorem trapezium_side_length : ℝ → Prop :=
  fun x =>
    let area : ℝ := 247
    let other_side : ℝ := 18
    let height : ℝ := 13
    area = (1 / 2) * (x + other_side) * height →
    x = 20

/-- The length of the first parallel side of the trapezium is 20 cm. -/
theorem first_parallel_side_length : trapezium_side_length 20 := by
  sorry

end NUMINAMATH_CALUDE_trapezium_side_length_first_parallel_side_length_l2876_287673


namespace NUMINAMATH_CALUDE_largest_sum_of_digits_l2876_287665

/-- Represents a digital time display in 24-hour format -/
structure TimeDisplay where
  hours : Nat
  minutes : Nat
  seconds : Nat
  hours_valid : hours < 24
  minutes_valid : minutes < 60
  seconds_valid : seconds < 60

/-- Calculates the sum of digits for a natural number -/
def sumOfDigits (n : Nat) : Nat :=
  n.repr.foldl (fun sum c => sum + c.toNat - '0'.toNat) 0

/-- Calculates the sum of digits for a time display -/
def sumOfTimeDigits (t : TimeDisplay) : Nat :=
  sumOfDigits t.hours + sumOfDigits t.minutes + sumOfDigits t.seconds

/-- The largest possible sum of digits in a 24-hour format digital watch display is 38 -/
theorem largest_sum_of_digits : ∀ t : TimeDisplay, sumOfTimeDigits t ≤ 38 := by
  sorry

end NUMINAMATH_CALUDE_largest_sum_of_digits_l2876_287665


namespace NUMINAMATH_CALUDE_triangle_similarity_BL_calculation_l2876_287627

theorem triangle_similarity_BL_calculation (AD BC AL BL LD LC : ℝ) 
  (h_similar : AD / BC = AL / BL ∧ AL / BL = LD / LC) :
  (∀ AB BD : ℝ, 
    (AB = 6 * Real.sqrt 13 ∧ AD = 6 ∧ BD = 12 * Real.sqrt 3) → 
    BL = 16 * Real.sqrt 3 - 12) ∧
  (∀ AB BD : ℝ, 
    (AB = 30 ∧ AD = 6 ∧ BD = 12 * Real.sqrt 6) → 
    BL = (16 * Real.sqrt 6 - 6) / 5) := by
  sorry

end NUMINAMATH_CALUDE_triangle_similarity_BL_calculation_l2876_287627


namespace NUMINAMATH_CALUDE_dice_probability_l2876_287640

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

def dice_outcome := ℕ × ℕ

def favorable_outcome (outcome : dice_outcome) : Prop :=
  is_prime outcome.1 ∧ is_perfect_square outcome.2

def total_outcomes : ℕ := 36

def favorable_outcomes : ℕ := 6

theorem dice_probability :
  (favorable_outcomes : ℚ) / total_outcomes = 1 / 6 :=
sorry

end NUMINAMATH_CALUDE_dice_probability_l2876_287640


namespace NUMINAMATH_CALUDE_emma_calculation_l2876_287641

/-- Reverses the digits of a two-digit number -/
def reverse_digits (n : ℕ) : ℕ :=
  10 * (n % 10) + (n / 10)

/-- Checks if a number is two-digit -/
def is_two_digit (n : ℕ) : Prop :=
  10 ≤ n ∧ n ≤ 99

theorem emma_calculation (a b : ℕ) (ha : is_two_digit a) (hb : b > 0) :
  (reverse_digits a * b - 18 = 120) → (a * b = 192) := by
  sorry

end NUMINAMATH_CALUDE_emma_calculation_l2876_287641


namespace NUMINAMATH_CALUDE_first_shipment_size_l2876_287695

/-- The size of the first shipment of tomatoes in kg -/
def first_shipment : ℕ := sorry

/-- The amount of tomatoes sold on Saturday in kg -/
def sold_saturday : ℕ := 300

/-- The amount of tomatoes that rotted on Sunday in kg -/
def rotted_sunday : ℕ := 200

/-- The total amount of tomatoes available on Tuesday in kg -/
def total_tuesday : ℕ := 2500

theorem first_shipment_size :
  first_shipment - sold_saturday - rotted_sunday + 2 * first_shipment = total_tuesday ∧
  first_shipment = 1000 := by sorry

end NUMINAMATH_CALUDE_first_shipment_size_l2876_287695


namespace NUMINAMATH_CALUDE_B_2_2_l2876_287644

def B : ℕ → ℕ → ℕ
| 0, n => n + 2
| m + 1, 0 => B m 2
| m + 1, n + 1 => B m (B (m + 1) n)

theorem B_2_2 : B 2 2 = 16 := by
  sorry

end NUMINAMATH_CALUDE_B_2_2_l2876_287644


namespace NUMINAMATH_CALUDE_cubic_is_closed_log_not_closed_sqrt_closed_condition_l2876_287654

-- Define a closed function
def is_closed_function (f : ℝ → ℝ) : Prop :=
  (∃ (a b : ℝ), a < b ∧ 
    (∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → (f x ≤ f y ∨ f y ≤ f x)) ∧
    (∀ y, a ≤ y ∧ y ≤ b → ∃ x, a ≤ x ∧ x ≤ b ∧ f x = y))

-- Theorem for the cubic function
theorem cubic_is_closed : is_closed_function (fun x => -x^3) :=
sorry

-- Theorem for the logarithmic function
theorem log_not_closed : ¬ is_closed_function (fun x => 2*x - Real.log x) :=
sorry

-- Theorem for the square root function
theorem sqrt_closed_condition (k : ℝ) : 
  is_closed_function (fun x => k + Real.sqrt (x + 2)) ↔ -9/4 < k ∧ k ≤ -2 :=
sorry

end NUMINAMATH_CALUDE_cubic_is_closed_log_not_closed_sqrt_closed_condition_l2876_287654


namespace NUMINAMATH_CALUDE_ellipse_equation_l2876_287629

theorem ellipse_equation (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a > b)
  (h4 : c / a = 2 / 3) (h5 : a = 3) :
  (∀ x y : ℝ, x^2 / a^2 + y^2 / b^2 = 1 ↔ x^2 / 9 + y^2 / 5 = 1) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_equation_l2876_287629


namespace NUMINAMATH_CALUDE_divide_by_reciprocal_twelve_divided_by_one_twelfth_l2876_287612

theorem divide_by_reciprocal (a b : ℚ) (h : b ≠ 0) : a / b = a * (1 / b) := by sorry

theorem twelve_divided_by_one_twelfth : 12 / (1 / 12) = 144 := by sorry

end NUMINAMATH_CALUDE_divide_by_reciprocal_twelve_divided_by_one_twelfth_l2876_287612


namespace NUMINAMATH_CALUDE_train_length_l2876_287699

/-- The length of a train given its speed, a man's walking speed, and the time taken to pass the man. -/
theorem train_length (train_speed : Real) (man_speed : Real) (passing_time : Real) :
  train_speed = 63 →
  man_speed = 3 →
  passing_time = 44.99640028797696 →
  (train_speed - man_speed) * passing_time * (1000 / 3600) = 750 := by
  sorry

end NUMINAMATH_CALUDE_train_length_l2876_287699


namespace NUMINAMATH_CALUDE_scatter_plot_placement_l2876_287671

/-- Represents a variable in a scatter plot -/
inductive Variable
  | Explanatory
  | Forecast

/-- Represents an axis in a scatter plot -/
inductive Axis
  | X
  | Y

/-- Represents the correct placement of variables on axes in a scatter plot -/
def correct_placement (v : Variable) (a : Axis) : Prop :=
  match v, a with
  | Variable.Explanatory, Axis.X => True
  | Variable.Forecast, Axis.Y => True
  | _, _ => False

/-- Theorem stating the correct placement of variables in a scatter plot -/
theorem scatter_plot_placement :
  ∀ (v : Variable) (a : Axis),
    correct_placement v a ↔
      ((v = Variable.Explanatory ∧ a = Axis.X) ∨
       (v = Variable.Forecast ∧ a = Axis.Y)) :=
by sorry

end NUMINAMATH_CALUDE_scatter_plot_placement_l2876_287671


namespace NUMINAMATH_CALUDE_max_value_cubic_function_l2876_287659

theorem max_value_cubic_function (m : ℝ) :
  (∃ (x : ℝ), ∀ (y : ℝ), -y^3 + 6*y^2 - m ≤ -x^3 + 6*x^2 - m) ∧
  (∃ (z : ℝ), -z^3 + 6*z^2 - m = 12) →
  m = 20 := by
sorry

end NUMINAMATH_CALUDE_max_value_cubic_function_l2876_287659


namespace NUMINAMATH_CALUDE_mn_solutions_l2876_287694

theorem mn_solutions (m n : ℤ) : 
  m * n ≥ 0 → m^3 + n^3 + 99*m*n = 33^3 → 
  ((m + n = 33 ∧ m ≥ 0 ∧ n ≥ 0) ∨ (m = -33 ∧ n = -33)) := by
  sorry

end NUMINAMATH_CALUDE_mn_solutions_l2876_287694


namespace NUMINAMATH_CALUDE_num_possible_lists_eq_50625_l2876_287634

/-- The number of balls in the bin -/
def num_balls : ℕ := 15

/-- The number of draws -/
def num_draws : ℕ := 4

/-- The number of possible lists when drawing 'num_draws' times from 'num_balls' with replacement -/
def num_possible_lists : ℕ := num_balls ^ num_draws

/-- Theorem stating that the number of possible lists is 50625 -/
theorem num_possible_lists_eq_50625 : num_possible_lists = 50625 := by
  sorry

end NUMINAMATH_CALUDE_num_possible_lists_eq_50625_l2876_287634


namespace NUMINAMATH_CALUDE_complex_number_in_fourth_quadrant_l2876_287662

theorem complex_number_in_fourth_quadrant :
  let z : ℂ := 1 / (1 + 3 * Complex.I)
  (z.re > 0) ∧ (z.im < 0) :=
by sorry

end NUMINAMATH_CALUDE_complex_number_in_fourth_quadrant_l2876_287662


namespace NUMINAMATH_CALUDE_ferry_journey_difference_l2876_287615

/-- Represents the properties of a ferry journey -/
structure FerryJourney where
  speed : ℝ
  time : ℝ
  distance : ℝ

/-- The ferry problem setup -/
def ferryProblem : Prop :=
  ∃ (P Q : FerryJourney),
    -- Ferry P properties
    P.speed = 6 ∧
    P.time = 3 ∧
    P.distance = P.speed * P.time ∧
    -- Ferry Q properties
    Q.distance = 2 * P.distance ∧
    Q.speed = P.speed + 3 ∧
    Q.time = Q.distance / Q.speed ∧
    -- The time difference is 1 hour
    Q.time - P.time = 1

/-- Theorem stating the solution to the ferry problem -/
theorem ferry_journey_difference : ferryProblem := by
  sorry

end NUMINAMATH_CALUDE_ferry_journey_difference_l2876_287615


namespace NUMINAMATH_CALUDE_mini_van_capacity_correct_l2876_287649

/-- Represents the capacity of a mini-van's tank in liters -/
def mini_van_capacity : ℝ := 65

/-- Represents the service cost per vehicle in dollars -/
def service_cost : ℝ := 2.10

/-- Represents the fuel cost per liter in dollars -/
def fuel_cost : ℝ := 0.60

/-- Represents the number of mini-vans -/
def num_mini_vans : ℕ := 3

/-- Represents the number of trucks -/
def num_trucks : ℕ := 2

/-- Represents the total cost in dollars -/
def total_cost : ℝ := 299.1

/-- Represents the ratio of truck tank capacity to mini-van tank capacity -/
def truck_capacity_ratio : ℝ := 2.2

theorem mini_van_capacity_correct :
  service_cost * (num_mini_vans + num_trucks) +
  fuel_cost * (num_mini_vans * mini_van_capacity + num_trucks * (truck_capacity_ratio * mini_van_capacity)) =
  total_cost := by sorry

end NUMINAMATH_CALUDE_mini_van_capacity_correct_l2876_287649


namespace NUMINAMATH_CALUDE_lawn_care_time_l2876_287600

/-- The time it takes Max to mow the lawn, in minutes -/
def mow_time : ℕ := 40

/-- The time it takes Max to fertilize the lawn, in minutes -/
def fertilize_time : ℕ := 2 * mow_time

/-- The total time it takes Max to both mow and fertilize the lawn, in minutes -/
def total_time : ℕ := mow_time + fertilize_time

theorem lawn_care_time : total_time = 120 := by
  sorry

end NUMINAMATH_CALUDE_lawn_care_time_l2876_287600


namespace NUMINAMATH_CALUDE_subset_implies_m_equals_one_l2876_287675

def A (m : ℝ) : Set ℝ := {-1, 3, 2*m-1}
def B (m : ℝ) : Set ℝ := {3, m^2}

theorem subset_implies_m_equals_one (m : ℝ) : B m ⊆ A m → m = 1 := by
  sorry

end NUMINAMATH_CALUDE_subset_implies_m_equals_one_l2876_287675


namespace NUMINAMATH_CALUDE_nth_root_two_inequality_l2876_287618

theorem nth_root_two_inequality (n : ℕ) (h : n ≥ 2) :
  (2 : ℝ) ^ (1 / n) - 1 ≤ Real.sqrt (2 / (n * (n - 1))) := by
  sorry

end NUMINAMATH_CALUDE_nth_root_two_inequality_l2876_287618


namespace NUMINAMATH_CALUDE_pythagorean_triple_identification_l2876_287608

def is_pythagorean_triple (a b c : ℕ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a * a + b * b = c * c

theorem pythagorean_triple_identification :
  ¬ is_pythagorean_triple 3 4 5 ∧
  ¬ is_pythagorean_triple 3 4 6 ∧
  is_pythagorean_triple 5 12 13 ∧
  ¬ is_pythagorean_triple 9 12 15 :=
by sorry

end NUMINAMATH_CALUDE_pythagorean_triple_identification_l2876_287608


namespace NUMINAMATH_CALUDE_factorial_equation_l2876_287609

theorem factorial_equation : 6 * 10 * 4 * 168 = Nat.factorial 8 := by
  sorry

end NUMINAMATH_CALUDE_factorial_equation_l2876_287609


namespace NUMINAMATH_CALUDE_water_consumption_l2876_287682

theorem water_consumption (W : ℝ) : 
  W > 0 →
  (W - 0.2 * W - 0.35 * (W - 0.2 * W) = 130) →
  W = 250 := by
sorry

end NUMINAMATH_CALUDE_water_consumption_l2876_287682


namespace NUMINAMATH_CALUDE_smallest_positive_leading_coeff_l2876_287622

/-- A quadratic polynomial that takes integer values for all integer inputs. -/
def IntegerValuedQuadratic (a b c : ℚ) : ℤ → ℤ :=
  fun x => ⌊a * x^2 + b * x + c⌋

/-- The property that a quadratic polynomial takes integer values for all integer inputs. -/
def IsIntegerValued (a b c : ℚ) : Prop :=
  ∀ x : ℤ, (IntegerValuedQuadratic a b c x : ℚ) = a * x^2 + b * x + c

/-- The smallest positive leading coefficient of an integer-valued quadratic polynomial is 1/2. -/
theorem smallest_positive_leading_coeff :
  (∃ a b c : ℚ, a > 0 ∧ IsIntegerValued a b c) ∧
  (∀ a b c : ℚ, a > 0 → IsIntegerValued a b c → a ≥ 1/2) ∧
  (∃ b c : ℚ, IsIntegerValued (1/2) b c) :=
sorry

end NUMINAMATH_CALUDE_smallest_positive_leading_coeff_l2876_287622
