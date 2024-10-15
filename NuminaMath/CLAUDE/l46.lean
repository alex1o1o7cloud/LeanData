import Mathlib

namespace NUMINAMATH_CALUDE_sum_of_roots_zero_l46_4626

theorem sum_of_roots_zero (z₁ z₂ z₃ : ℝ) : 
  (4096 * z₁^3 + 16 * z₁ - 9 = 0) →
  (4096 * z₂^3 + 16 * z₂ - 9 = 0) →
  (4096 * z₃^3 + 16 * z₃ - 9 = 0) →
  z₁ + z₂ + z₃ = 0 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_roots_zero_l46_4626


namespace NUMINAMATH_CALUDE_pe_class_size_l46_4619

theorem pe_class_size (fourth_grade_classes : ℕ) (students_per_class : ℕ) (total_cupcakes : ℕ) :
  fourth_grade_classes = 3 →
  students_per_class = 30 →
  total_cupcakes = 140 →
  total_cupcakes - (fourth_grade_classes * students_per_class) = 50 :=
by
  sorry

end NUMINAMATH_CALUDE_pe_class_size_l46_4619


namespace NUMINAMATH_CALUDE_quadratic_inequality_l46_4615

theorem quadratic_inequality (x : ℝ) : -15 * x^2 + 10 * x + 5 > 0 ↔ -1/3 < x ∧ x < 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l46_4615


namespace NUMINAMATH_CALUDE_min_red_beads_l46_4647

/-- Represents a necklace with blue and red beads. -/
structure Necklace where
  blue_count : ℕ
  red_count : ℕ
  cyclic : Bool
  segment_condition : Bool

/-- Checks if a necklace satisfies the given conditions. -/
def is_valid_necklace (n : Necklace) : Prop :=
  n.blue_count = 50 ∧
  n.cyclic ∧
  n.segment_condition

/-- Theorem stating the minimum number of red beads required. -/
theorem min_red_beads (n : Necklace) :
  is_valid_necklace n → n.red_count ≥ 29 := by
  sorry

#check min_red_beads

end NUMINAMATH_CALUDE_min_red_beads_l46_4647


namespace NUMINAMATH_CALUDE_simplify_polynomial_l46_4624

theorem simplify_polynomial (x : ℝ) : 
  4 * x^3 + 5 * x + 6 * x^2 + 10 - (3 - 6 * x^2 - 4 * x^3 + 2 * x) = 
  8 * x^3 + 12 * x^2 + 3 * x + 7 := by sorry

end NUMINAMATH_CALUDE_simplify_polynomial_l46_4624


namespace NUMINAMATH_CALUDE_female_advanced_under_40_l46_4699

theorem female_advanced_under_40 (total_employees : ℕ) (female_employees : ℕ) (male_employees : ℕ)
  (advanced_degrees : ℕ) (college_degrees : ℕ) (high_school_diplomas : ℕ)
  (male_advanced : ℕ) (male_college : ℕ) (male_high_school : ℕ)
  (female_under_40_ratio : ℚ) :
  total_employees = 280 →
  female_employees = 160 →
  male_employees = 120 →
  advanced_degrees = 120 →
  college_degrees = 100 →
  high_school_diplomas = 60 →
  male_advanced = 50 →
  male_college = 35 →
  male_high_school = 35 →
  female_under_40_ratio = 3/4 →
  ⌊(advanced_degrees - male_advanced : ℚ) * female_under_40_ratio⌋ = 52 :=
by sorry

end NUMINAMATH_CALUDE_female_advanced_under_40_l46_4699


namespace NUMINAMATH_CALUDE_number_of_subsets_l46_4638

universe u

def card {α : Type u} (s : Set α) : ℕ := sorry

theorem number_of_subsets (M A B : Set ℕ) : 
  card M = 10 →
  A ⊆ M →
  B ⊆ M →
  A ∩ B = ∅ →
  card A = 2 →
  card B = 3 →
  card {X : Set ℕ | A ⊆ X ∧ X ⊆ M} = 256 := by sorry

end NUMINAMATH_CALUDE_number_of_subsets_l46_4638


namespace NUMINAMATH_CALUDE_intersection_implies_a_value_l46_4625

def set_A (a : ℝ) : Set ℝ := {-4, 2*a-1, a^2}
def set_B (a : ℝ) : Set ℝ := {a-5, 1-a, 9}

theorem intersection_implies_a_value :
  ∀ a : ℝ, set_A a ∩ set_B a = {9} → a = -3 :=
by
  sorry

end NUMINAMATH_CALUDE_intersection_implies_a_value_l46_4625


namespace NUMINAMATH_CALUDE_swimmers_passing_count_l46_4616

/-- Represents the swimming scenario -/
structure SwimmingScenario where
  poolLength : ℝ
  speedA : ℝ
  speedB : ℝ
  totalTime : ℝ
  turnDelay : ℝ

/-- Calculates the number of times swimmers pass each other -/
def passingCount (s : SwimmingScenario) : ℕ :=
  sorry

/-- The main theorem stating the number of times swimmers pass each other -/
theorem swimmers_passing_count :
  let s : SwimmingScenario := {
    poolLength := 100,
    speedA := 4,
    speedB := 3,
    totalTime := 900,  -- 15 minutes in seconds
    turnDelay := 5
  }
  passingCount s = 63 := by sorry

end NUMINAMATH_CALUDE_swimmers_passing_count_l46_4616


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l46_4696

def g (p q r s : ℝ) (x : ℂ) : ℂ :=
  x^4 + p*x^3 + q*x^2 + r*x + s

theorem sum_of_coefficients 
  (p q r s : ℝ) 
  (h1 : g p q r s (3*I) = 0)
  (h2 : g p q r s (1 + 2*I) = 0) : 
  p + q + r + s = -41 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l46_4696


namespace NUMINAMATH_CALUDE_linear_function_through_points_and_m_l46_4635

/-- Represents a point in 2D space -/
structure Point where
  x : ℚ
  y : ℚ

/-- Represents a linear function y = kx + b -/
structure LinearFunction where
  k : ℚ
  b : ℚ

/-- Check if a point lies on a linear function -/
def pointOnFunction (p : Point) (f : LinearFunction) : Prop :=
  p.y = f.k * p.x + f.b

theorem linear_function_through_points_and_m
  (A : Point)
  (B : Point)
  (C : Point)
  (h1 : A.x = 3 ∧ A.y = 5)
  (h2 : B.x = -4 ∧ B.y = -9)
  (h3 : C.y = 2) :
  ∃ (f : LinearFunction),
    pointOnFunction A f ∧
    pointOnFunction B f ∧
    pointOnFunction C f ∧
    f.k = 2 ∧
    f.b = -1 ∧
    C.x = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_linear_function_through_points_and_m_l46_4635


namespace NUMINAMATH_CALUDE_expression_greater_than_30_l46_4698

theorem expression_greater_than_30 :
  ∃ (expr : ℝ),
    (expr = 20 / (2 - Real.sqrt 2)) ∧
    (expr > 30) := by
  sorry

end NUMINAMATH_CALUDE_expression_greater_than_30_l46_4698


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l46_4681

theorem sum_of_coefficients (a a₁ a₂ a₃ a₄ a₅ : ℝ) :
  (∀ x : ℝ, (2*x - 1)^5 = a*x^5 + a₁*x^4 + a₂*x^3 + a₃*x^2 + a₄*x + a₅) →
  a₁ + a₃ + a₅ = -121 := by
sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l46_4681


namespace NUMINAMATH_CALUDE_eleventh_number_with_digit_sum_13_l46_4610

/-- A function that returns the sum of digits of a positive integer -/
def digit_sum (n : ℕ+) : ℕ :=
  sorry

/-- A function that returns the nth positive integer whose digits sum to 13 -/
def nth_number_with_digit_sum_13 (n : ℕ+) : ℕ+ :=
  sorry

/-- Theorem stating that the 11th number with digit sum 13 is 145 -/
theorem eleventh_number_with_digit_sum_13 :
  nth_number_with_digit_sum_13 11 = 145 :=
sorry

end NUMINAMATH_CALUDE_eleventh_number_with_digit_sum_13_l46_4610


namespace NUMINAMATH_CALUDE_bakery_sugar_amount_l46_4657

/-- Given the ratios of ingredients in a bakery storage room, prove the amount of sugar. -/
theorem bakery_sugar_amount (sugar flour baking_soda : ℚ) 
  (h1 : sugar / flour = 5 / 2)
  (h2 : flour / baking_soda = 10 / 1)
  (h3 : flour / (baking_soda + 60) = 8 / 1) :
  sugar = 6000 := by
  sorry

end NUMINAMATH_CALUDE_bakery_sugar_amount_l46_4657


namespace NUMINAMATH_CALUDE_initial_games_count_l46_4659

theorem initial_games_count (initial remaining given : ℕ) : 
  remaining = initial - given → 
  given = 7 → 
  remaining = 91 → 
  initial = 98 := by sorry

end NUMINAMATH_CALUDE_initial_games_count_l46_4659


namespace NUMINAMATH_CALUDE_solution_set_f_less_than_2_range_of_a_for_solution_exists_l46_4634

-- Define the function f
def f (x : ℝ) : ℝ := |2*x + 1| - |x - 1|

-- Theorem for the solution set of f(x) < 2
theorem solution_set_f_less_than_2 :
  {x : ℝ | f x < 2} = Set.Ioo (-4 : ℝ) (2/3) := by sorry

-- Theorem for the range of a where f(x) ≤ a - a²/2 has a solution
theorem range_of_a_for_solution_exists :
  {a : ℝ | ∃ x, f x ≤ a - a^2/2} = Set.Icc (-1 : ℝ) 3 := by sorry

end NUMINAMATH_CALUDE_solution_set_f_less_than_2_range_of_a_for_solution_exists_l46_4634


namespace NUMINAMATH_CALUDE_equation_solutions_l46_4641

theorem equation_solutions :
  (∃ (s₁ s₂ : Set ℝ),
    s₁ = {x : ℝ | (5 - 2*x)^2 - 16 = 0} ∧
    s₂ = {x : ℝ | 2*(x - 3) = x^2 - 9} ∧
    s₁ = {1/2, 9/2} ∧
    s₂ = {3, -1}) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l46_4641


namespace NUMINAMATH_CALUDE_problem_solution_l46_4667

def S : Set ℝ := {x | (x + 2) / (x - 5) < 0}
def P (a : ℝ) : Set ℝ := {x | a + 1 < x ∧ x < 2*a + 15}

theorem problem_solution :
  (S = {x : ℝ | -2 < x ∧ x < 5}) ∧
  (∀ a : ℝ, S ⊆ P a ↔ -5 ≤ a ∧ a ≤ -3) := by sorry

end NUMINAMATH_CALUDE_problem_solution_l46_4667


namespace NUMINAMATH_CALUDE_equal_expressions_l46_4617

theorem equal_expressions : 2007 * 2011 - 2008 * 2010 = 2008 * 2012 - 2009 * 2011 := by
  sorry

end NUMINAMATH_CALUDE_equal_expressions_l46_4617


namespace NUMINAMATH_CALUDE_course_selection_schemes_l46_4660

theorem course_selection_schemes :
  let total_courses : ℕ := 4
  let student_a_choices : ℕ := 2
  let student_b_choices : ℕ := 3
  let student_c_choices : ℕ := 3
  
  (Nat.choose total_courses student_a_choices) *
  (Nat.choose total_courses student_b_choices) *
  (Nat.choose total_courses student_c_choices) = 96 :=
by
  sorry

end NUMINAMATH_CALUDE_course_selection_schemes_l46_4660


namespace NUMINAMATH_CALUDE_fixed_point_on_all_parabolas_l46_4678

/-- The parabola family defined by a real parameter t -/
def parabola (t : ℝ) (x : ℝ) : ℝ := 4 * x^2 + 2 * t * x - 3 * t

/-- The fixed point through which all parabolas pass -/
def fixed_point : ℝ × ℝ := (3, 36)

/-- Theorem stating that the fixed point lies on all parabolas in the family -/
theorem fixed_point_on_all_parabolas :
  ∀ t : ℝ, parabola t (fixed_point.1) = fixed_point.2 := by sorry

end NUMINAMATH_CALUDE_fixed_point_on_all_parabolas_l46_4678


namespace NUMINAMATH_CALUDE_correct_probabilities_l46_4683

def ball_probabilities (total_balls : ℕ) (p_red p_black_or_yellow p_yellow_or_green : ℚ) : Prop :=
  let p_black := 1/4
  let p_yellow := 1/6
  let p_green := 1/4
  total_balls = 12 ∧
  p_red = 1/3 ∧
  p_black_or_yellow = 5/12 ∧
  p_yellow_or_green = 5/12 ∧
  p_red + p_black + p_yellow + p_green = 1 ∧
  p_black_or_yellow = p_black + p_yellow ∧
  p_yellow_or_green = p_yellow + p_green

theorem correct_probabilities : 
  ∀ (total_balls : ℕ) (p_red p_black_or_yellow p_yellow_or_green : ℚ),
  ball_probabilities total_balls p_red p_black_or_yellow p_yellow_or_green := by
  sorry

end NUMINAMATH_CALUDE_correct_probabilities_l46_4683


namespace NUMINAMATH_CALUDE_jack_weight_l46_4621

theorem jack_weight (total_weight sam_weight jack_weight : ℕ) : 
  total_weight = 96 →
  jack_weight = sam_weight + 8 →
  total_weight = sam_weight + jack_weight →
  jack_weight = 52 := by
sorry

end NUMINAMATH_CALUDE_jack_weight_l46_4621


namespace NUMINAMATH_CALUDE_trajectory_and_PQ_length_l46_4632

-- Define the circle C₁
def C₁ (x y : ℝ) : Prop := x^2 + y^2 = 9

-- Define the point A on C₁
def A_on_C₁ (x₀ y₀ : ℝ) : Prop := C₁ x₀ y₀

-- Define the perpendicular condition for AN
def AN_perp_x (x₀ y₀ : ℝ) : Prop := ∃ (N : ℝ × ℝ), N.1 = x₀ ∧ N.2 = 0

-- Define the condition for point M
def M_condition (x y x₀ y₀ : ℝ) : Prop :=
  ∃ (N : ℝ × ℝ), N.1 = x₀ ∧ N.2 = 0 ∧
  (x, y) + 2 * (x - x₀, y - y₀) = (2 * Real.sqrt 2 - 2) • (x₀, 0)

-- Define the curve C
def C (x y : ℝ) : Prop := x^2 / 8 + y^2 / 4 = 1

-- Define the intersection of line l with curve C
def l_intersects_C (P Q : ℝ × ℝ) : Prop :=
  C P.1 P.2 ∧ C Q.1 Q.2 ∧ P ≠ Q

-- Define the condition for circle PQ passing through O
def circle_PQ_through_O (P Q : ℝ × ℝ) : Prop :=
  P.1 * Q.1 + P.2 * Q.2 = 0

theorem trajectory_and_PQ_length :
  ∀ (x y x₀ y₀ : ℝ) (P Q : ℝ × ℝ),
  A_on_C₁ x₀ y₀ →
  AN_perp_x x₀ y₀ →
  M_condition x y x₀ y₀ →
  l_intersects_C P Q →
  circle_PQ_through_O P Q →
  (C x y ∧ 
   (4 * Real.sqrt 6 / 3)^2 ≤ ((P.1 - Q.1)^2 + (P.2 - Q.2)^2) ∧
   ((P.1 - Q.1)^2 + (P.2 - Q.2)^2) ≤ (2 * Real.sqrt 3)^2) :=
by sorry

end NUMINAMATH_CALUDE_trajectory_and_PQ_length_l46_4632


namespace NUMINAMATH_CALUDE_largest_n_binomial_sum_equals_binomial_l46_4685

theorem largest_n_binomial_sum_equals_binomial (n : ℕ) : 
  (Nat.choose 9 4 + Nat.choose 9 5 = Nat.choose 10 n) → n ≤ 5 :=
by sorry

end NUMINAMATH_CALUDE_largest_n_binomial_sum_equals_binomial_l46_4685


namespace NUMINAMATH_CALUDE_min_value_expression_lower_bound_achievable_l46_4663

theorem min_value_expression (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c)
  (h_ineq : 21 * a * b + 2 * b * c + 8 * c * a ≤ 12) :
  1 / a + 2 / b + 3 / c ≥ 15 / 2 := by
  sorry

theorem lower_bound_achievable :
  ∃ a b c : ℝ, 0 < a ∧ 0 < b ∧ 0 < c ∧
  21 * a * b + 2 * b * c + 8 * c * a ≤ 12 ∧
  1 / a + 2 / b + 3 / c = 15 / 2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_lower_bound_achievable_l46_4663


namespace NUMINAMATH_CALUDE_number_divided_by_24_is_19_l46_4675

theorem number_divided_by_24_is_19 (x : ℤ) : (x / 24 = 19) → x = 456 := by
  sorry

end NUMINAMATH_CALUDE_number_divided_by_24_is_19_l46_4675


namespace NUMINAMATH_CALUDE_unique_prime_factors_count_l46_4654

def product : ℕ := 102 * 103 * 105 * 107

theorem unique_prime_factors_count :
  (Nat.factors product).toFinset.card = 7 := by sorry

end NUMINAMATH_CALUDE_unique_prime_factors_count_l46_4654


namespace NUMINAMATH_CALUDE_probability_even_8_sided_die_l46_4637

/-- A fair 8-sided die -/
def fair_8_sided_die : Finset ℕ := Finset.range 8

/-- The set of even outcomes on the die -/
def even_outcomes : Finset ℕ := Finset.filter (λ x => x % 2 = 0) fair_8_sided_die

/-- The probability of an event occurring when rolling the die -/
def probability (event : Finset ℕ) : ℚ :=
  (event.card : ℚ) / (fair_8_sided_die.card : ℚ)

theorem probability_even_8_sided_die :
  probability even_outcomes = 1/2 := by sorry

end NUMINAMATH_CALUDE_probability_even_8_sided_die_l46_4637


namespace NUMINAMATH_CALUDE_subtract_percentage_equivalent_to_multiply_l46_4627

theorem subtract_percentage_equivalent_to_multiply (a : ℝ) : 
  a - (0.04 * a) = 0.96 * a := by sorry

end NUMINAMATH_CALUDE_subtract_percentage_equivalent_to_multiply_l46_4627


namespace NUMINAMATH_CALUDE_waiter_income_fraction_l46_4653

/-- Given a waiter's salary and tips, where the tips are 7/4 of the salary,
    prove that the fraction of total income from tips is 7/11. -/
theorem waiter_income_fraction (salary : ℚ) (tips : ℚ) (h : tips = (7 / 4) * salary) :
  tips / (salary + tips) = 7 / 11 := by
  sorry

end NUMINAMATH_CALUDE_waiter_income_fraction_l46_4653


namespace NUMINAMATH_CALUDE_p_and_not_q_is_true_l46_4680

-- Define proposition p
def p : Prop := ∃ x : ℝ, x - 2 > 0

-- Define proposition q
def q : Prop := ∀ x : ℝ, Real.sqrt x > x

-- Theorem to prove
theorem p_and_not_q_is_true : p ∧ ¬q := by
  sorry

end NUMINAMATH_CALUDE_p_and_not_q_is_true_l46_4680


namespace NUMINAMATH_CALUDE_orchids_after_planting_l46_4656

/-- The number of orchid bushes in the park after planting -/
def total_orchids (initial : ℕ) (planted : ℕ) : ℕ :=
  initial + planted

/-- Theorem: The park will have 6 orchid bushes after planting -/
theorem orchids_after_planting :
  total_orchids 2 4 = 6 := by
sorry

end NUMINAMATH_CALUDE_orchids_after_planting_l46_4656


namespace NUMINAMATH_CALUDE_age_difference_l46_4682

/-- The difference in years between individuals a and c -/
def R (a c : ℕ) : ℕ := a - c

/-- The age of an individual after 5 years -/
def L (x : ℕ) : ℕ := x + 5

theorem age_difference (a b c d : ℕ) :
  (L a + L b = L b + L c + 10) →
  (c + d = a + d - 12) →
  R a c = 12 := by
  sorry

end NUMINAMATH_CALUDE_age_difference_l46_4682


namespace NUMINAMATH_CALUDE_events_mutually_exclusive_not_opposite_l46_4609

structure Bag where
  red : Nat
  white : Nat
  black : Nat

def draw_two_balls (b : Bag) : Nat := b.red + b.white + b.black - 2

def exactly_one_white (b : Bag) : Prop := 
  ∃ (x : Nat), x = 1 ∧ x ≤ b.white ∧ x ≤ draw_two_balls b

def exactly_two_white (b : Bag) : Prop := 
  ∃ (x : Nat), x = 2 ∧ x ≤ b.white ∧ x ≤ draw_two_balls b

def mutually_exclusive (p q : Prop) : Prop :=
  ¬(p ∧ q)

def opposite (p q : Prop) : Prop :=
  (p ↔ ¬q) ∧ (q ↔ ¬p)

theorem events_mutually_exclusive_not_opposite 
  (b : Bag) (h1 : b.red = 3) (h2 : b.white = 2) (h3 : b.black = 1) : 
  mutually_exclusive (exactly_one_white b) (exactly_two_white b) ∧ 
  ¬(opposite (exactly_one_white b) (exactly_two_white b)) := by
  sorry

end NUMINAMATH_CALUDE_events_mutually_exclusive_not_opposite_l46_4609


namespace NUMINAMATH_CALUDE_duke_dvd_count_l46_4611

/-- Represents the number of DVDs Duke found in the first box -/
def first_box_count : ℕ := sorry

/-- Represents the price of each DVD in the first box -/
def first_box_price : ℚ := 2

/-- Represents the number of DVDs Duke found in the second box -/
def second_box_count : ℕ := 5

/-- Represents the price of each DVD in the second box -/
def second_box_price : ℚ := 5

/-- Represents the average price of all DVDs bought -/
def average_price : ℚ := 3

theorem duke_dvd_count : first_box_count = 5 := by
  sorry

end NUMINAMATH_CALUDE_duke_dvd_count_l46_4611


namespace NUMINAMATH_CALUDE_scientific_notation_361000000_l46_4607

/-- Express 361000000 in scientific notation -/
theorem scientific_notation_361000000 : ∃ (a : ℝ) (n : ℤ), 
  361000000 = a * (10 : ℝ) ^ n ∧ 1 ≤ a ∧ a < 10 ∧ a = 3.61 ∧ n = 8 := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_361000000_l46_4607


namespace NUMINAMATH_CALUDE_least_perimeter_triangle_l46_4687

theorem least_perimeter_triangle (a b x : ℕ) (ha : a = 24) (hb : b = 37) : 
  (a + b > x ∧ a + x > b ∧ b + x > a) → (∀ y : ℕ, (a + b > y ∧ a + y > b ∧ b + y > a) → x ≤ y) →
  a + b + x = 75 :=
sorry

end NUMINAMATH_CALUDE_least_perimeter_triangle_l46_4687


namespace NUMINAMATH_CALUDE_dress_shop_inventory_l46_4622

/-- Proves that given a total space of 200 dresses and 83 red dresses,
    the number of additional blue dresses compared to red dresses is 34. -/
theorem dress_shop_inventory (total_space : Nat) (red_dresses : Nat)
    (h1 : total_space = 200)
    (h2 : red_dresses = 83) :
    total_space - red_dresses - red_dresses = 34 := by
  sorry

end NUMINAMATH_CALUDE_dress_shop_inventory_l46_4622


namespace NUMINAMATH_CALUDE_roots_sum_of_squares_l46_4665

theorem roots_sum_of_squares (a b : ℝ) : 
  (a^2 - 7*a + 7 = 0) → (b^2 - 7*b + 7 = 0) → a^2 + b^2 = 35 := by
  sorry

end NUMINAMATH_CALUDE_roots_sum_of_squares_l46_4665


namespace NUMINAMATH_CALUDE_polygon_sides_from_angle_sum_l46_4697

theorem polygon_sides_from_angle_sum (n : ℕ) (angle_sum : ℝ) : 
  angle_sum = 180 * (n - 2) → angle_sum = 1080 → n = 8 := by
  sorry

end NUMINAMATH_CALUDE_polygon_sides_from_angle_sum_l46_4697


namespace NUMINAMATH_CALUDE_visitors_growth_rate_l46_4605

theorem visitors_growth_rate (x : ℝ) : 
  (420000 : ℝ) * (1 + x)^2 = 1339100 ↔ 42 * (1 + x)^2 = 133.91 :=
by sorry

end NUMINAMATH_CALUDE_visitors_growth_rate_l46_4605


namespace NUMINAMATH_CALUDE_no_linear_term_in_product_l46_4612

theorem no_linear_term_in_product (m : ℝ) : 
  (∀ x : ℝ, ∃ a b : ℝ, (x + m) * (2 - x) = a * x^2 + b) → m = 2 := by
  sorry

end NUMINAMATH_CALUDE_no_linear_term_in_product_l46_4612


namespace NUMINAMATH_CALUDE_sixth_graders_count_l46_4686

theorem sixth_graders_count (seventh_graders : ℕ) (seventh_percent : ℚ) (sixth_percent : ℚ) 
  (h1 : seventh_graders = 64)
  (h2 : seventh_percent = 32 / 100)
  (h3 : sixth_percent = 38 / 100)
  (h4 : seventh_graders = (seventh_percent * (seventh_graders / seventh_percent)).floor) :
  (sixth_percent * (seventh_graders / seventh_percent)).floor = 76 := by
  sorry

end NUMINAMATH_CALUDE_sixth_graders_count_l46_4686


namespace NUMINAMATH_CALUDE_no_integer_triangle_with_integer_altitudes_and_perimeter_1995_l46_4684

theorem no_integer_triangle_with_integer_altitudes_and_perimeter_1995 :
  ¬ ∃ (a b c h_a h_b h_c : ℕ), 
    (a + b + c = 1995) ∧ 
    (h_a^2 * (4*a^2) = 2*a^2*b^2 + 2*a^2*c^2 + 2*c^2*b^2 - a^4 - b^4 - c^4) ∧
    (h_b^2 * (4*b^2) = 2*a^2*b^2 + 2*b^2*c^2 + 2*c^2*a^2 - a^4 - b^4 - c^4) ∧
    (h_c^2 * (4*c^2) = 2*a^2*c^2 + 2*b^2*c^2 + 2*a^2*b^2 - a^4 - b^4 - c^4) :=
by sorry


end NUMINAMATH_CALUDE_no_integer_triangle_with_integer_altitudes_and_perimeter_1995_l46_4684


namespace NUMINAMATH_CALUDE_cheese_cookies_per_box_l46_4651

/-- The number of boxes in a carton -/
def boxes_per_carton : ℕ := 12

/-- The price of a pack of cheese cookies in dollars -/
def price_per_pack : ℕ := 1

/-- The cost of a dozen cartons in dollars -/
def cost_dozen_cartons : ℕ := 1440

/-- The number of packs of cheese cookies in each box -/
def packs_per_box : ℕ := 10

theorem cheese_cookies_per_box :
  packs_per_box = 10 := by sorry

end NUMINAMATH_CALUDE_cheese_cookies_per_box_l46_4651


namespace NUMINAMATH_CALUDE_floor_of_4_7_l46_4676

theorem floor_of_4_7 : ⌊(4.7 : ℝ)⌋ = 4 := by sorry

end NUMINAMATH_CALUDE_floor_of_4_7_l46_4676


namespace NUMINAMATH_CALUDE_line_equation_through_point_with_inclination_l46_4672

/-- The equation of a line passing through (1, 2) with a 45° inclination angle -/
theorem line_equation_through_point_with_inclination (x y : ℝ) : 
  (x - y + 1 = 0) ↔ 
  (∃ (t : ℝ), x = 1 + t ∧ y = 2 + t) ∧ 
  (∀ (x₁ y₁ x₂ y₂ : ℝ), x₁ - x₂ ≠ 0 → (y₁ - y₂) / (x₁ - x₂) = 1) :=
by sorry

end NUMINAMATH_CALUDE_line_equation_through_point_with_inclination_l46_4672


namespace NUMINAMATH_CALUDE_total_seashells_l46_4694

def sam_seashells : ℕ := 18
def mary_seashells : ℕ := 47
def john_seashells : ℕ := 32
def emily_seashells : ℕ := 26

theorem total_seashells : 
  sam_seashells + mary_seashells + john_seashells + emily_seashells = 123 := by
  sorry

end NUMINAMATH_CALUDE_total_seashells_l46_4694


namespace NUMINAMATH_CALUDE_mary_marbles_l46_4623

def dan_marbles : ℕ := 5
def mary_multiplier : ℕ := 2

theorem mary_marbles : 
  dan_marbles * mary_multiplier = 10 := by sorry

end NUMINAMATH_CALUDE_mary_marbles_l46_4623


namespace NUMINAMATH_CALUDE_no_valid_b_exists_l46_4628

/-- Given a point P and its symmetric point Q about the origin, 
    prove that there is no real value of b for which both points 
    satisfy the inequality 2x - by + 1 ≤ 0. -/
theorem no_valid_b_exists (P : ℝ × ℝ) (Q : ℝ × ℝ) : 
  P = (1, -2) → 
  Q.1 = -P.1 → 
  Q.2 = -P.2 → 
  ¬∃ b : ℝ, (2 * P.1 - b * P.2 + 1 ≤ 0) ∧ (2 * Q.1 - b * Q.2 + 1 ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_no_valid_b_exists_l46_4628


namespace NUMINAMATH_CALUDE_inequality_equivalence_l46_4602

theorem inequality_equivalence (x : ℝ) : (1 / (x - 1) > 1) ↔ (1 < x ∧ x < 2) :=
sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l46_4602


namespace NUMINAMATH_CALUDE_sales_difference_prove_sales_difference_l46_4691

def morning_sales (remy_bottles : ℕ) (nick_bottles : ℕ) (price_per_bottle : ℚ) : ℚ :=
  (remy_bottles + nick_bottles) * price_per_bottle

theorem sales_difference (remy_morning_bottles : ℕ) (evening_sales : ℚ) : ℚ :=
  let nick_morning_bottles : ℕ := remy_morning_bottles - 6
  let price_per_bottle : ℚ := 1/2
  let morning_total : ℚ := morning_sales remy_morning_bottles nick_morning_bottles price_per_bottle
  evening_sales - morning_total

theorem prove_sales_difference : sales_difference 55 55 = 3 := by
  sorry

end NUMINAMATH_CALUDE_sales_difference_prove_sales_difference_l46_4691


namespace NUMINAMATH_CALUDE_triangle_side_relationship_l46_4679

/-- Given a triangle with perimeter 12 and one side 5, prove the relationship between the other two sides -/
theorem triangle_side_relationship (x y : ℝ) : 
  (0 < x ∧ x < 6) → 
  (0 < y ∧ y < 6) → 
  (5 + x + y = 12) → 
  y = 7 - x :=
by sorry

end NUMINAMATH_CALUDE_triangle_side_relationship_l46_4679


namespace NUMINAMATH_CALUDE_fraction_simplification_l46_4600

theorem fraction_simplification (a b : ℕ) (h : b ≠ 0) : 
  (4 * a) / (4 * b) = a / b :=
sorry

end NUMINAMATH_CALUDE_fraction_simplification_l46_4600


namespace NUMINAMATH_CALUDE_correct_paintball_spending_l46_4662

/-- Represents the paintball spending calculation for John --/
def paintball_spending (regular_plays_per_month : ℕ) 
                       (boxes_per_play : ℕ) 
                       (price_1_5 : ℚ) 
                       (price_6_11 : ℚ) 
                       (price_12_plus : ℚ) 
                       (discount_12_plus : ℚ) 
                       (regular_maintenance : ℚ) 
                       (peak_maintenance : ℚ) 
                       (travel_week1 : ℚ) 
                       (travel_week2 : ℚ) 
                       (travel_week3 : ℚ) 
                       (travel_week4 : ℚ) : ℚ × ℚ :=
  let regular_boxes := regular_plays_per_month * boxes_per_play
  let peak_boxes := 2 * regular_boxes
  let travel_cost := travel_week1 + travel_week2 + travel_week3 + travel_week4
  
  let regular_paintball_cost := 
    if regular_boxes ≤ 5 then regular_boxes * price_1_5
    else if regular_boxes ≤ 11 then regular_boxes * price_6_11
    else let cost := regular_boxes * price_12_plus
         cost - (cost * discount_12_plus)
  
  let peak_paintball_cost := 
    let cost := peak_boxes * price_12_plus
    cost - (cost * discount_12_plus)
  
  let regular_total := regular_paintball_cost + regular_maintenance + travel_cost
  let peak_total := peak_paintball_cost + peak_maintenance + travel_cost
  
  (regular_total, peak_total)

/-- Theorem stating the correct paintball spending for John --/
theorem correct_paintball_spending :
  paintball_spending 3 3 25 23 22 (1/10) 40 60 10 15 12 8 = (292, 461.4) :=
sorry

end NUMINAMATH_CALUDE_correct_paintball_spending_l46_4662


namespace NUMINAMATH_CALUDE_shot_radius_l46_4618

/-- Given a sphere of radius 4 cm from which 64 equal-sized spherical shots can be made,
    the radius of each shot is 1 cm. -/
theorem shot_radius (R : ℝ) (N : ℕ) (r : ℝ) : R = 4 → N = 64 → (R / r)^3 = N → r = 1 := by
  sorry

end NUMINAMATH_CALUDE_shot_radius_l46_4618


namespace NUMINAMATH_CALUDE_diamond_eight_three_l46_4661

-- Define the diamond operation
def diamond (x y : ℤ) : ℤ :=
  sorry

-- State the theorem
theorem diamond_eight_three : diamond 8 3 = 39 := by
  sorry

-- Define the properties of the diamond operation
axiom diamond_zero (x : ℤ) : diamond x 0 = x

axiom diamond_comm (x y : ℤ) : diamond x y = diamond y x

axiom diamond_recursive (x y : ℤ) : diamond (x + 2) y = diamond x y + 2 * y + 3

end NUMINAMATH_CALUDE_diamond_eight_three_l46_4661


namespace NUMINAMATH_CALUDE_f_is_even_f_is_decreasing_f_minimum_on_interval_l46_4670

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - 2*a*x + 1

-- Theorem 1: f(x) is even iff a = 0
theorem f_is_even (a : ℝ) : (∀ x, f a x = f a (-x)) ↔ a = 0 := by sorry

-- Theorem 2: f(x) is decreasing on (-∞, 4] iff a ≥ 4
theorem f_is_decreasing (a : ℝ) : (∀ x y, x ≤ y ∧ y ≤ 4 → f a x ≥ f a y) ↔ a ≥ 4 := by sorry

-- Theorem 3: Minimum value of f(x) on [1, 2]
theorem f_minimum_on_interval (a : ℝ) :
  (∀ x ∈ [1, 2], f a x ≥ 
    (if a ≤ 1 then 2 - 2*a
     else if a < 2 then 1 - a^2
     else 5 - 4*a)) ∧
  (∃ x ∈ [1, 2], f a x = 
    (if a ≤ 1 then 2 - 2*a
     else if a < 2 then 1 - a^2
     else 5 - 4*a)) := by sorry

end NUMINAMATH_CALUDE_f_is_even_f_is_decreasing_f_minimum_on_interval_l46_4670


namespace NUMINAMATH_CALUDE_inequalities_for_negative_numbers_l46_4603

theorem inequalities_for_negative_numbers (a b : ℝ) (h : a < b ∧ b < 0) :
  (1 / a > 1 / b) ∧ (Real.sqrt (-a) > Real.sqrt (-b)) ∧ (abs a > -b) := by
  sorry

end NUMINAMATH_CALUDE_inequalities_for_negative_numbers_l46_4603


namespace NUMINAMATH_CALUDE_bricks_used_l46_4655

/-- Calculates the total number of bricks used in a construction project --/
theorem bricks_used (courses_per_wall : ℕ) (bricks_per_course : ℕ) (total_walls : ℕ) 
  (h1 : courses_per_wall = 15)
  (h2 : bricks_per_course = 25)
  (h3 : total_walls = 8) : 
  (total_walls - 1) * courses_per_wall * bricks_per_course + 
  (courses_per_wall - 1) * bricks_per_course = 2975 := by
  sorry

end NUMINAMATH_CALUDE_bricks_used_l46_4655


namespace NUMINAMATH_CALUDE_cos_225_degrees_l46_4629

theorem cos_225_degrees : 
  Real.cos (225 * π / 180) = -1 / Real.sqrt 2 := by
  have cos_addition : ∀ θ, Real.cos (π + θ) = -Real.cos θ := sorry
  have cos_45_degrees : Real.cos (45 * π / 180) = 1 / Real.sqrt 2 := sorry
  sorry

end NUMINAMATH_CALUDE_cos_225_degrees_l46_4629


namespace NUMINAMATH_CALUDE_picture_area_calculation_l46_4674

/-- Given a sheet of paper with width, length, and margin, calculates the area of the picture. -/
def picture_area (paper_width paper_length margin : ℝ) : ℝ :=
  (paper_width - 2 * margin) * (paper_length - 2 * margin)

/-- Theorem stating that for a paper of 8.5 by 10 inches with a 1.5-inch margin, 
    the picture area is 38.5 square inches. -/
theorem picture_area_calculation :
  picture_area 8.5 10 1.5 = 38.5 := by
  sorry

#eval picture_area 8.5 10 1.5

end NUMINAMATH_CALUDE_picture_area_calculation_l46_4674


namespace NUMINAMATH_CALUDE_polynomial_roots_l46_4666

theorem polynomial_roots : 
  ∀ z : ℂ, z^4 - 6*z^2 + z + 8 = 0 ↔ z = -2 ∨ z = 1 ∨ z = Complex.I * Real.sqrt 7 ∨ z = -Complex.I * Real.sqrt 7 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_roots_l46_4666


namespace NUMINAMATH_CALUDE_largest_non_factor_product_of_100_l46_4646

def is_factor (a b : ℕ) : Prop := b % a = 0

theorem largest_non_factor_product_of_100 :
  ∀ x y : ℕ,
    x ≠ y →
    x > 0 →
    y > 0 →
    is_factor x 100 →
    is_factor y 100 →
    ¬ is_factor (x * y) 100 →
    x * y ≤ 40 :=
by sorry

end NUMINAMATH_CALUDE_largest_non_factor_product_of_100_l46_4646


namespace NUMINAMATH_CALUDE_complex_calculations_l46_4669

theorem complex_calculations : 
  (∀ x : ℝ, x^2 = 3 → (1 + x) * (2 - x) = -1 + x) ∧
  (Real.sqrt 36 * Real.sqrt 12 / Real.sqrt 3 = 12) ∧
  (Real.sqrt 18 - Real.sqrt 8 + Real.sqrt (1/8) = 5 * Real.sqrt 2 / 4) ∧
  ((3 * Real.sqrt 18 + (1/5) * Real.sqrt 50 - 4 * Real.sqrt (1/2)) / Real.sqrt 32 = 2) :=
by sorry

end NUMINAMATH_CALUDE_complex_calculations_l46_4669


namespace NUMINAMATH_CALUDE_cos_sum_fifth_circle_l46_4604

theorem cos_sum_fifth_circle : Real.cos (2 * Real.pi / 5) + Real.cos (4 * Real.pi / 5) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_cos_sum_fifth_circle_l46_4604


namespace NUMINAMATH_CALUDE_circle_equation_from_parabola_focus_l46_4690

/-- The equation of a circle with its center at the focus of the parabola y² = 4x
    and passing through the origin is x² + y² - 2x = 0. -/
theorem circle_equation_from_parabola_focus (x y : ℝ) : 
  (∃ (h : ℝ), y^2 = 4*x ∧ h = 1) →  -- Focus of parabola y² = 4x is at (1, 0)
  (0^2 + 0^2 = (x - 1)^2 + y^2) →  -- Circle passes through origin (0, 0)
  (x^2 + y^2 - 2*x = 0) :=
by sorry

end NUMINAMATH_CALUDE_circle_equation_from_parabola_focus_l46_4690


namespace NUMINAMATH_CALUDE_investment_principal_is_200_l46_4648

/-- Represents the simple interest investment scenario -/
structure SimpleInterestInvestment where
  principal : ℝ
  rate : ℝ
  amount_after_2_years : ℝ
  amount_after_5_years : ℝ

/-- The simple interest investment satisfies the given conditions -/
def satisfies_conditions (investment : SimpleInterestInvestment) : Prop :=
  investment.amount_after_2_years = investment.principal * (1 + 2 * investment.rate) ∧
  investment.amount_after_5_years = investment.principal * (1 + 5 * investment.rate) ∧
  investment.amount_after_2_years = 260 ∧
  investment.amount_after_5_years = 350

/-- Theorem stating that the investment with the given conditions has a principal of $200 -/
theorem investment_principal_is_200 :
  ∃ (investment : SimpleInterestInvestment), 
    satisfies_conditions investment ∧ investment.principal = 200 := by
  sorry

end NUMINAMATH_CALUDE_investment_principal_is_200_l46_4648


namespace NUMINAMATH_CALUDE_percentage_problem_l46_4693

theorem percentage_problem (x : ℝ) (h : 24 = (75 / 100) * x) : x = 32 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l46_4693


namespace NUMINAMATH_CALUDE_abs_five_point_five_minus_pi_l46_4677

theorem abs_five_point_five_minus_pi :
  |5.5 - Real.pi| = 5.5 - Real.pi :=
by sorry

end NUMINAMATH_CALUDE_abs_five_point_five_minus_pi_l46_4677


namespace NUMINAMATH_CALUDE_polynomial_evaluation_l46_4688

theorem polynomial_evaluation : 
  let a : ℝ := 2
  (3 * a^3 - 7 * a^2 + a - 5) * (4 * a - 6) = -14 := by
sorry

end NUMINAMATH_CALUDE_polynomial_evaluation_l46_4688


namespace NUMINAMATH_CALUDE_unfair_coin_probability_l46_4695

theorem unfair_coin_probability (p : ℝ) : 
  (0 ≤ p ∧ p ≤ 1) →
  (35 * p^4 * (1-p)^3 = 343/3125) →
  p = 0.7 := by
sorry

end NUMINAMATH_CALUDE_unfair_coin_probability_l46_4695


namespace NUMINAMATH_CALUDE_cube_fraction_product_l46_4620

theorem cube_fraction_product : 
  (((7^3 - 1) / (7^3 + 1)) * 
   ((8^3 - 1) / (8^3 + 1)) * 
   ((9^3 - 1) / (9^3 + 1)) * 
   ((10^3 - 1) / (10^3 + 1)) * 
   ((11^3 - 1) / (11^3 + 1))) = 931 / 946 := by
  sorry

end NUMINAMATH_CALUDE_cube_fraction_product_l46_4620


namespace NUMINAMATH_CALUDE_z_in_first_quadrant_l46_4664

theorem z_in_first_quadrant (z : ℂ) (h : z * (1 - 3*I) = 5 - 5*I) : 
  0 < z.re ∧ 0 < z.im :=
sorry

end NUMINAMATH_CALUDE_z_in_first_quadrant_l46_4664


namespace NUMINAMATH_CALUDE_base_eight_23456_equals_10030_l46_4639

def base_eight_to_ten (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (8 ^ i)) 0

theorem base_eight_23456_equals_10030 :
  base_eight_to_ten [6, 5, 4, 3, 2] = 10030 := by
  sorry

end NUMINAMATH_CALUDE_base_eight_23456_equals_10030_l46_4639


namespace NUMINAMATH_CALUDE_jack_two_queens_probability_l46_4650

-- Define a standard deck of cards
def standardDeck : ℕ := 52

-- Define the number of Jacks in a standard deck
def numJacks : ℕ := 4

-- Define the number of Queens in a standard deck
def numQueens : ℕ := 4

-- Define the probability of the specific draw
def probJackTwoQueens : ℚ := 2 / 5525

-- Theorem statement
theorem jack_two_queens_probability :
  (numJacks / standardDeck) * (numQueens / (standardDeck - 1)) * ((numQueens - 1) / (standardDeck - 2)) = probJackTwoQueens := by
  sorry

end NUMINAMATH_CALUDE_jack_two_queens_probability_l46_4650


namespace NUMINAMATH_CALUDE_spanish_test_average_score_l46_4645

theorem spanish_test_average_score (marco_score margaret_score average_score : ℝ) : 
  marco_score = 0.9 * average_score →
  margaret_score = marco_score + 5 →
  margaret_score = 86 →
  average_score = 90 := by
sorry

end NUMINAMATH_CALUDE_spanish_test_average_score_l46_4645


namespace NUMINAMATH_CALUDE_car_fuel_efficiency_l46_4640

/-- Given a car that travels 140 kilometers using 3.5 gallons of gasoline,
    prove that the car's fuel efficiency is 40 kilometers per gallon. -/
theorem car_fuel_efficiency :
  let distance : ℝ := 140  -- Total distance in kilometers
  let fuel : ℝ := 3.5      -- Fuel used in gallons
  let efficiency : ℝ := distance / fuel  -- Fuel efficiency in km/gallon
  efficiency = 40 := by sorry

end NUMINAMATH_CALUDE_car_fuel_efficiency_l46_4640


namespace NUMINAMATH_CALUDE_one_fifths_in_one_fourth_l46_4652

theorem one_fifths_in_one_fourth : (1 : ℚ) / 4 / ((1 : ℚ) / 5) = 5 / 4 := by
  sorry

end NUMINAMATH_CALUDE_one_fifths_in_one_fourth_l46_4652


namespace NUMINAMATH_CALUDE_money_sharing_calculation_l46_4658

/-- Proves that given a money sharing scenario with a specific ratio and known amount,
    the total shared amount can be calculated. -/
theorem money_sharing_calculation (mark_ratio nina_ratio oliver_ratio : ℕ)
                                  (nina_amount : ℕ) :
  mark_ratio = 2 →
  nina_ratio = 3 →
  oliver_ratio = 9 →
  nina_amount = 60 →
  ∃ (total : ℕ), total = 280 ∧ 
    nina_amount * (mark_ratio + nina_ratio + oliver_ratio) = total * nina_ratio :=
by
  sorry

end NUMINAMATH_CALUDE_money_sharing_calculation_l46_4658


namespace NUMINAMATH_CALUDE_sector_area_l46_4636

theorem sector_area (n : Real) (r : Real) (h1 : n = 120) (h2 : r = 3) :
  (n * π * r^2) / 360 = 3 * π := by
  sorry

end NUMINAMATH_CALUDE_sector_area_l46_4636


namespace NUMINAMATH_CALUDE_chord_line_equation_l46_4643

/-- The equation of a line passing through a chord of an ellipse -/
theorem chord_line_equation (x₁ y₁ x₂ y₂ : ℝ) :
  (x₁^2 / 36 + y₁^2 / 9 = 1) →
  (x₂^2 / 36 + y₂^2 / 9 = 1) →
  ((x₁ + x₂) / 2 = 4) →
  ((y₁ + y₂) / 2 = 2) →
  (∀ x y : ℝ, y - 2 = -(1/2) * (x - 4) ↔ x + 2*y - 8 = 0) :=
by sorry

end NUMINAMATH_CALUDE_chord_line_equation_l46_4643


namespace NUMINAMATH_CALUDE_litter_patrol_collection_l46_4642

theorem litter_patrol_collection (glass_bottles : ℕ) (aluminum_cans : ℕ) 
  (h1 : glass_bottles = 10) (h2 : aluminum_cans = 8) : 
  glass_bottles + aluminum_cans = 18 := by
  sorry

end NUMINAMATH_CALUDE_litter_patrol_collection_l46_4642


namespace NUMINAMATH_CALUDE_rectangular_garden_area_rectangular_garden_area_proof_l46_4608

/-- A rectangular garden with length three times its width and perimeter 72 meters has an area of 243 square meters. -/
theorem rectangular_garden_area : ℝ → Prop :=
  fun w : ℝ =>
    w > 0 →                   -- width is positive
    2 * (w + 3 * w) = 72 →    -- perimeter is 72 meters
    w * (3 * w) = 243         -- area is 243 square meters

/-- Proof of the rectangular_garden_area theorem -/
theorem rectangular_garden_area_proof : ∃ w : ℝ, rectangular_garden_area w :=
  sorry

end NUMINAMATH_CALUDE_rectangular_garden_area_rectangular_garden_area_proof_l46_4608


namespace NUMINAMATH_CALUDE_man_speed_against_current_l46_4644

/-- Given a man's speed with the current and the speed of the current,
    calculate the man's speed against the current. -/
def speed_against_current (speed_with_current : ℝ) (current_speed : ℝ) : ℝ :=
  speed_with_current - 2 * current_speed

/-- Theorem stating that given the specific conditions,
    the man's speed against the current is 18 kmph. -/
theorem man_speed_against_current :
  speed_against_current 20 1 = 18 := by
  sorry

#eval speed_against_current 20 1

end NUMINAMATH_CALUDE_man_speed_against_current_l46_4644


namespace NUMINAMATH_CALUDE_f_value_for_specific_inputs_l46_4649

-- Define the function f
def f (m n k p : ℕ) : ℤ := (n^2 - m) * (n^k - m^p)

-- Theorem statement
theorem f_value_for_specific_inputs :
  f 5 3 2 3 = -464 :=
by sorry

end NUMINAMATH_CALUDE_f_value_for_specific_inputs_l46_4649


namespace NUMINAMATH_CALUDE_alex_calculation_l46_4606

theorem alex_calculation (x : ℝ) : x / 6 - 18 = 24 → x * 6 + 18 = 1530 := by
  sorry

end NUMINAMATH_CALUDE_alex_calculation_l46_4606


namespace NUMINAMATH_CALUDE_complex_fraction_sum_l46_4692

theorem complex_fraction_sum (a b : ℝ) : 
  (Complex.I : ℂ) ^ 2 = -1 →
  (2 + Complex.I) / (1 + Complex.I) = Complex.mk a b →
  a + b = 1 := by
sorry

end NUMINAMATH_CALUDE_complex_fraction_sum_l46_4692


namespace NUMINAMATH_CALUDE_cos_2alpha_plus_pi_third_l46_4668

theorem cos_2alpha_plus_pi_third (α : ℝ) (h : Real.sin (α - π/3) = 2/3) :
  Real.cos (2*α + π/3) = -1/9 := by
  sorry

end NUMINAMATH_CALUDE_cos_2alpha_plus_pi_third_l46_4668


namespace NUMINAMATH_CALUDE_division_zero_implies_divisor_greater_l46_4630

theorem division_zero_implies_divisor_greater (d : ℕ) :
  2016 / d = 0 → d > 2016 := by
sorry

end NUMINAMATH_CALUDE_division_zero_implies_divisor_greater_l46_4630


namespace NUMINAMATH_CALUDE_inverse_f_at_10_l46_4614

-- Define the function f
def f (x : ℝ) : ℝ := x^2 + 1

-- Define the domain of f
def f_domain (x : ℝ) : Prop := x ≥ 1

-- State the theorem
theorem inverse_f_at_10 (f_inv : ℝ → ℝ) 
  (h1 : ∀ x, f_domain x → f_inv (f x) = x) 
  (h2 : ∀ y, y ≥ 1 → f (f_inv y) = y) : 
  f_inv 10 = 3 := by sorry

end NUMINAMATH_CALUDE_inverse_f_at_10_l46_4614


namespace NUMINAMATH_CALUDE_x_range_l46_4673

theorem x_range (x y z : ℝ) (h1 : x ≥ y) (h2 : y ≥ z) (h3 : x + y + z = 1) (h4 : x^2 + y^2 + z^2 = 3) :
  x ∈ Set.Icc 1 (5/3) :=
sorry

end NUMINAMATH_CALUDE_x_range_l46_4673


namespace NUMINAMATH_CALUDE_ratio_equality_l46_4633

theorem ratio_equality (a b c x y z : ℝ) 
  (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c)
  (pos_x : 0 < x) (pos_y : 0 < y) (pos_z : 0 < z)
  (sum_abc : a^2 + b^2 + c^2 = 25)
  (sum_xyz : x^2 + y^2 + z^2 = 36)
  (sum_axbycz : a*x + b*y + c*z = 30) :
  (a + b + c) / (x + y + z) = 5/6 := by
sorry

end NUMINAMATH_CALUDE_ratio_equality_l46_4633


namespace NUMINAMATH_CALUDE_third_degree_polynomial_property_l46_4689

/-- A third-degree polynomial with real coefficients. -/
def ThirdDegreePolynomial := ℝ → ℝ

/-- The property that |f(1)| = |f(2)| = |f(4)| = 10 -/
def SatisfiesCondition (f : ThirdDegreePolynomial) : Prop :=
  |f 1| = 10 ∧ |f 2| = 10 ∧ |f 4| = 10

theorem third_degree_polynomial_property (f : ThirdDegreePolynomial) 
  (h : SatisfiesCondition f) : |f 0| = 34/3 := by
  sorry

end NUMINAMATH_CALUDE_third_degree_polynomial_property_l46_4689


namespace NUMINAMATH_CALUDE_lap_time_improvement_l46_4613

-- Define the initial swimming scenario
def initial_total_time : ℚ := 29
def initial_break_time : ℚ := 3
def initial_laps : ℚ := 14

-- Define the current swimming scenario
def current_total_time : ℚ := 28
def current_laps : ℚ := 16

-- Define the lap time calculation function
def lap_time (total_time : ℚ) (break_time : ℚ) (laps : ℚ) : ℚ :=
  (total_time - break_time) / laps

-- State the theorem
theorem lap_time_improvement :
  lap_time initial_total_time initial_break_time initial_laps -
  lap_time current_total_time 0 current_laps = 3 / 28 := by
  sorry

end NUMINAMATH_CALUDE_lap_time_improvement_l46_4613


namespace NUMINAMATH_CALUDE_nested_fraction_evaluation_l46_4631

theorem nested_fraction_evaluation : 
  (1 : ℚ) / (2 + 1 / (3 + 1 / 4)) = 13 / 30 := by
  sorry

end NUMINAMATH_CALUDE_nested_fraction_evaluation_l46_4631


namespace NUMINAMATH_CALUDE_toothpicks_stage_20_l46_4601

/-- The number of toothpicks in stage n of the pattern -/
def toothpicks (n : ℕ) : ℕ :=
  4 + 3 * (n - 1)

theorem toothpicks_stage_20 :
  toothpicks 20 = 61 := by
  sorry

end NUMINAMATH_CALUDE_toothpicks_stage_20_l46_4601


namespace NUMINAMATH_CALUDE_female_officers_count_l46_4671

theorem female_officers_count (total_on_duty : ℕ) (female_on_duty_ratio : ℚ) (female_total_ratio : ℚ) :
  total_on_duty = 500 →
  female_on_duty_ratio = 1/2 →
  female_total_ratio = 1/4 →
  (female_on_duty_ratio * total_on_duty : ℚ) / female_total_ratio = 1000 := by
  sorry

end NUMINAMATH_CALUDE_female_officers_count_l46_4671
