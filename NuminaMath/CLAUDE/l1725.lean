import Mathlib

namespace sin_pi_minus_alpha_l1725_172580

theorem sin_pi_minus_alpha (α : Real) 
  (h1 : π / 2 < α) 
  (h2 : α < π) 
  (h3 : 3 * Real.sin (2 * α) = 2 * Real.sin α) : 
  Real.sin (π - α) = 2 * Real.sqrt 2 / 3 := by
  sorry

end sin_pi_minus_alpha_l1725_172580


namespace cubic_roots_relation_l1725_172565

/-- Given a cubic equation x^3 + px^2 + qx + r = 0 with roots α, β, γ, 
    returns a function that computes expressions involving these roots -/
def cubicRootRelations (p q r : ℝ) : 
  (ℝ → ℝ → ℝ → ℝ) → ℝ := sorry

theorem cubic_roots_relation (a b c s t : ℝ) : 
  cubicRootRelations 3 4 (-11) (fun x y z => x) = a ∧
  cubicRootRelations 3 4 (-11) (fun x y z => y) = b ∧
  cubicRootRelations 3 4 (-11) (fun x y z => z) = c ∧
  cubicRootRelations (-2) s t (fun x y z => x) = a + b ∧
  cubicRootRelations (-2) s t (fun x y z => y) = b + c ∧
  cubicRootRelations (-2) s t (fun x y z => z) = c + a →
  s = 8 ∧ t = 23 := by sorry

end cubic_roots_relation_l1725_172565


namespace max_of_three_l1725_172555

theorem max_of_three (a b c : ℝ) :
  let x := max a b
  ∀ m : ℝ, (m = max a (max b c) ↔ (m = x ∨ (c > x ∧ m = c))) :=
by sorry

end max_of_three_l1725_172555


namespace hold_age_ratio_l1725_172576

theorem hold_age_ratio (mother_age : ℕ) (son_age : ℕ) (h1 : mother_age = 36) (h2 : mother_age = 3 * son_age) :
  (mother_age - 8) / (son_age - 8) = 7 := by
  sorry

end hold_age_ratio_l1725_172576


namespace solve_equation_l1725_172558

theorem solve_equation : ∃ x : ℝ, 25 - (4 + 3) = 5 + x ∧ x = 13 := by sorry

end solve_equation_l1725_172558


namespace nancys_contribution_is_36_l1725_172516

/-- The number of bottle caps Marilyn had initially -/
def initial_caps : ℝ := 51.0

/-- The number of bottle caps Marilyn had after Nancy's contribution -/
def final_caps : ℝ := 87.0

/-- The number of bottle caps Nancy gave to Marilyn -/
def nancys_contribution : ℝ := final_caps - initial_caps

theorem nancys_contribution_is_36 : nancys_contribution = 36 := by
  sorry

end nancys_contribution_is_36_l1725_172516


namespace geometric_sum_value_l1725_172524

theorem geometric_sum_value (x : ℝ) (h1 : x^2023 - 3*x + 2 = 0) (h2 : x ≠ 1) :
  x^2022 + x^2021 + x^2020 + x^2019 + x^2018 + x^2017 + x^2016 + x^2015 + x^2014 + x^2013 + 
  x^2012 + x^2011 + x^2010 + x^2009 + x^2008 + x^2007 + x^2006 + x^2005 + x^2004 + x^2003 + 
  x^2002 + x^2001 + x^2000 + x^1999 + x^1998 + x^1997 + x^1996 + x^1995 + x^1994 + x^1993 + 
  x^1992 + x^1991 + x^1990 + x^1989 + x^1988 + x^1987 + x^1986 + x^1985 + x^1984 + x^1983 + 
  x^1982 + x^1981 + x^1980 + x^1979 + x^1978 + x^1977 + x^1976 + x^1975 + x^1974 + x^1973 + 
  -- ... (continuing the pattern)
  x^22 + x^21 + x^20 + x^19 + x^18 + x^17 + x^16 + x^15 + x^14 + x^13 + 
  x^12 + x^11 + x^10 + x^9 + x^8 + x^7 + x^6 + x^5 + x^4 + x^3 + x^2 + x + 1 = 3 := by
sorry

end geometric_sum_value_l1725_172524


namespace probability_geometry_second_draw_l1725_172584

/-- Represents the set of questions in the problem -/
structure QuestionSet where
  total : ℕ
  algebra : ℕ
  geometry : ℕ
  algebra_first_draw : Prop

/-- The probability of selecting a geometry question on the second draw,
    given an algebra question was selected on the first draw -/
def conditional_probability (qs : QuestionSet) : ℚ :=
  qs.geometry / (qs.total - 1)

/-- The main theorem to prove -/
theorem probability_geometry_second_draw 
  (qs : QuestionSet) 
  (h1 : qs.total = 5) 
  (h2 : qs.algebra = 3) 
  (h3 : qs.geometry = 2) 
  (h4 : qs.algebra_first_draw) : 
  conditional_probability qs = 1/2 := by
  sorry

end probability_geometry_second_draw_l1725_172584


namespace wedding_cost_theorem_l1725_172521

/-- Calculates the total cost of a wedding given the venue cost, cost per guest, 
    John's desired number of guests, and the percentage increase desired by John's wife. -/
def wedding_cost (venue_cost : ℕ) (cost_per_guest : ℕ) (john_guests : ℕ) (wife_increase_percent : ℕ) : ℕ :=
  let total_guests := john_guests + john_guests * wife_increase_percent / 100
  venue_cost + cost_per_guest * total_guests

/-- Proves that the total cost of the wedding is $50,000 given the specified conditions. -/
theorem wedding_cost_theorem : 
  wedding_cost 10000 500 50 60 = 50000 := by
  sorry

#eval wedding_cost 10000 500 50 60

end wedding_cost_theorem_l1725_172521


namespace parabola_unique_coefficients_l1725_172534

/-- A parabola is defined by the equation y = ax^2 + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The y-coordinate of a point on the parabola given its x-coordinate -/
def Parabola.y_at (p : Parabola) (x : ℝ) : ℝ :=
  p.a * x^2 + p.b * x + p.c

/-- The slope of the tangent line to the parabola at a given x-coordinate -/
def Parabola.slope_at (p : Parabola) (x : ℝ) : ℝ :=
  2 * p.a * x + p.b

/-- Theorem: For a parabola y = ax^2 + bx + c, if it passes through (1, 1),
    and the slope of the tangent line at (2, -1) is 1,
    then a = 3, b = -11, and c = 9 -/
theorem parabola_unique_coefficients (p : Parabola) 
    (h1 : p.y_at 1 = 1)
    (h2 : p.y_at 2 = -1)
    (h3 : p.slope_at 2 = 1) :
    p.a = 3 ∧ p.b = -11 ∧ p.c = 9 := by
  sorry

end parabola_unique_coefficients_l1725_172534


namespace fish_apple_equivalence_l1725_172528

/-- Represents the value of one fish in terms of apples -/
def fish_value (f l r a : ℚ) : Prop :=
  5 * f = 3 * l ∧ l = 6 * r ∧ 3 * r = 2 * a ∧ f = 12/5 * a

/-- Theorem stating that under the given trading conditions, one fish is worth 12/5 apples -/
theorem fish_apple_equivalence :
  ∀ f l r a : ℚ, fish_value f l r a :=
by
  sorry

#check fish_apple_equivalence

end fish_apple_equivalence_l1725_172528


namespace parking_cost_theorem_l1725_172537

/-- The number of hours for the initial parking cost. -/
def h : ℝ := 2

/-- The initial parking cost in dollars. -/
def initial_cost : ℝ := 10

/-- The additional cost per hour after the initial period in dollars. -/
def additional_cost_per_hour : ℝ := 1.75

/-- The total number of hours parked. -/
def total_hours : ℝ := 9

/-- The average cost per hour for the total parking time. -/
def average_cost_per_hour : ℝ := 2.4722222222222223

theorem parking_cost_theorem :
  h = 2 ∧
  initial_cost = 10 ∧
  additional_cost_per_hour = 1.75 ∧
  total_hours = 9 ∧
  average_cost_per_hour = 2.4722222222222223 →
  initial_cost + additional_cost_per_hour * (total_hours - h) = average_cost_per_hour * total_hours :=
by sorry

end parking_cost_theorem_l1725_172537


namespace B_power_97_l1725_172562

def B : Matrix (Fin 3) (Fin 3) ℝ := !![1, 0, 0; 0, 0, -2; 0, 2, 0]

theorem B_power_97 : 
  B^97 = !![1, 0, 0; 0, 0, -2 * 16^24; 0, 2 * 16^24, 0] := by sorry

end B_power_97_l1725_172562


namespace peters_class_size_l1725_172513

theorem peters_class_size :
  ∀ (hands_without_peter : ℕ) (hands_per_student : ℕ),
    hands_without_peter = 20 →
    hands_per_student = 2 →
    hands_without_peter / hands_per_student + 1 = 11 :=
by
  sorry

end peters_class_size_l1725_172513


namespace binomial_probability_theorem_l1725_172532

/-- A random variable following a binomial distribution -/
structure BinomialRV where
  n : ℕ
  p : ℝ
  h1 : 0 ≤ p ∧ p ≤ 1

/-- The expectation of a binomial random variable -/
def expectation (X : BinomialRV) : ℝ := X.n * X.p

/-- The variance of a binomial random variable -/
def variance (X : BinomialRV) : ℝ := X.n * X.p * (1 - X.p)

/-- The probability of a binomial random variable being equal to k -/
def probability (X : BinomialRV) (k : ℕ) : ℝ :=
  (X.n.choose k) * (X.p ^ k) * ((1 - X.p) ^ (X.n - k))

theorem binomial_probability_theorem (X : BinomialRV) 
  (h2 : expectation X = 2)
  (h3 : variance X = 4/3) :
  probability X 2 = 80/243 := by
  sorry

end binomial_probability_theorem_l1725_172532


namespace tangent_slope_implies_a_l1725_172525

-- Define the curve
def f (a x : ℝ) : ℝ := x^4 + a*x^2 + 1

-- Define the derivative of the curve
def f' (a x : ℝ) : ℝ := 4*x^3 + 2*a*x

theorem tangent_slope_implies_a (a : ℝ) :
  f a (-1) = a + 2 → f' a (-1) = 8 → a = -6 := by
  sorry

end tangent_slope_implies_a_l1725_172525


namespace q_implies_k_range_p_or_q_and_not_p_and_q_implies_k_range_l1725_172569

-- Define proposition p
def p (k : ℝ) : Prop := ∀ x : ℝ, x^2 - k*x + 2*k + 5 ≥ 0

-- Define proposition q
def q (k : ℝ) : Prop := ∃ a b : ℝ, a > 0 ∧ b < 0 ∧ a = 4 - k ∧ b = 1 - k

-- Theorem 1
theorem q_implies_k_range (k : ℝ) : q k → 1 < k ∧ k < 4 := by sorry

-- Theorem 2
theorem p_or_q_and_not_p_and_q_implies_k_range (k : ℝ) : 
  (p k ∨ q k) ∧ ¬(p k ∧ q k) → (-2 ≤ k ∧ k ≤ 1) ∨ (4 ≤ k ∧ k ≤ 10) := by sorry

end q_implies_k_range_p_or_q_and_not_p_and_q_implies_k_range_l1725_172569


namespace arithmetic_mean_of_special_set_l1725_172502

def set_of_numbers : List ℕ := [8, 88, 888, 8888, 88888, 888888, 8888888, 88888888, 888888888]

theorem arithmetic_mean_of_special_set :
  let n := set_of_numbers.length
  let sum := set_of_numbers.sum
  sum / n = 98765432 := by sorry

end arithmetic_mean_of_special_set_l1725_172502


namespace third_root_of_cubic_l1725_172571

theorem third_root_of_cubic (a b : ℝ) : 
  (∀ x : ℝ, a * x^3 + (a + 2*b) * x^2 + (b - 3*a) * x + (8 - a) = 0 ↔ x = -2 ∨ x = 3 ∨ x = 4/3) →
  ∃ x : ℝ, x ≠ -2 ∧ x ≠ 3 ∧ a * x^3 + (a + 2*b) * x^2 + (b - 3*a) * x + (8 - a) = 0 ∧ x = 4/3 :=
by sorry

end third_root_of_cubic_l1725_172571


namespace fourteenSidedFigureArea_l1725_172574

/-- A point in 2D space represented by its x and y coordinates -/
structure Point where
  x : ℝ
  y : ℝ

/-- The vertices of the fourteen-sided figure -/
def vertices : List Point := [
  ⟨1, 2⟩, ⟨1, 3⟩, ⟨2, 4⟩, ⟨3, 5⟩, ⟨4, 6⟩, ⟨5, 5⟩, ⟨6, 5⟩,
  ⟨7, 4⟩, ⟨7, 3⟩, ⟨6, 2⟩, ⟨5, 1⟩, ⟨4, 1⟩, ⟨3, 1⟩, ⟨2, 2⟩
]

/-- Calculate the area of a polygon given its vertices -/
def polygonArea (vertices : List Point) : ℝ :=
  sorry -- Implement the calculation of polygon area

/-- Theorem stating that the area of the fourteen-sided figure is 14 square centimeters -/
theorem fourteenSidedFigureArea : polygonArea vertices = 14 := by
  sorry

end fourteenSidedFigureArea_l1725_172574


namespace unique_four_digit_number_l1725_172517

/-- Represents a 4-digit number as a tuple of its digits -/
def FourDigitNumber := (Nat × Nat × Nat × Nat)

/-- Checks if a tuple represents a valid 4-digit number -/
def isValidFourDigitNumber (n : FourDigitNumber) : Prop :=
  let (a, b, c, d) := n
  1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ 0 ≤ c ∧ c ≤ 9 ∧ 0 ≤ d ∧ d ≤ 9

/-- Checks if a 4-digit number satisfies the given conditions -/
def satisfiesConditions (n : FourDigitNumber) : Prop :=
  let (a, b, c, d) := n
  b = 3 * a ∧ c = a + b ∧ d = 3 * b

theorem unique_four_digit_number :
  ∃! (n : FourDigitNumber), isValidFourDigitNumber n ∧ satisfiesConditions n ∧ n = (1, 3, 4, 9) :=
by sorry

end unique_four_digit_number_l1725_172517


namespace one_third_of_seven_times_nine_l1725_172547

theorem one_third_of_seven_times_nine : (1 / 3 : ℚ) * (7 * 9) = 21 := by
  sorry

end one_third_of_seven_times_nine_l1725_172547


namespace partial_fraction_decomposition_l1725_172523

theorem partial_fraction_decomposition (C D : ℚ) :
  (∀ x : ℚ, x ≠ 7 ∧ x ≠ -2 →
    (5 * x - 3) / (x^2 - 5*x - 14) = C / (x - 7) + D / (x + 2)) →
  C = 32/9 ∧ D = 13/9 := by
sorry

end partial_fraction_decomposition_l1725_172523


namespace sin_plus_cos_value_l1725_172554

theorem sin_plus_cos_value (A : Real) (h : Real.sin (2 * A) = 2/3) :
  Real.sin A + Real.cos A = Real.sqrt (5/3) := by
  sorry

end sin_plus_cos_value_l1725_172554


namespace total_carrots_grown_l1725_172590

/-- The total number of carrots grown by Joan, Jessica, and Michael is 77. -/
theorem total_carrots_grown (joan_carrots : ℕ) (jessica_carrots : ℕ) (michael_carrots : ℕ)
  (h1 : joan_carrots = 29)
  (h2 : jessica_carrots = 11)
  (h3 : michael_carrots = 37) :
  joan_carrots + jessica_carrots + michael_carrots = 77 :=
by sorry

end total_carrots_grown_l1725_172590


namespace complex_equation_magnitude_l1725_172540

theorem complex_equation_magnitude (z : ℂ) (a b : ℝ) (n : ℕ) 
  (h : a * z^n + b * Complex.I * z^(n-1) + b * Complex.I * z - a = 0) : 
  Complex.abs z = 1 := by
  sorry

end complex_equation_magnitude_l1725_172540


namespace carlos_summer_reading_l1725_172501

/-- Carlos' summer reading challenge -/
theorem carlos_summer_reading 
  (july_books august_books total_goal : ℕ) 
  (h1 : july_books = 28)
  (h2 : august_books = 30)
  (h3 : total_goal = 100) :
  total_goal - (july_books + august_books) = 42 := by
  sorry

end carlos_summer_reading_l1725_172501


namespace distinct_three_digit_count_base_6_l1725_172552

/-- The number of three-digit numbers with distinct digits in base b -/
def distinct_three_digit_count (b : ℕ) : ℕ := (b - 1)^2 * (b - 2)

/-- Theorem: In base 6, there are exactly 100 three-digit numbers with distinct digits -/
theorem distinct_three_digit_count_base_6 : distinct_three_digit_count 6 = 100 := by
  sorry

#eval distinct_three_digit_count 6  -- This should evaluate to 100

end distinct_three_digit_count_base_6_l1725_172552


namespace competition_problem_l1725_172568

theorem competition_problem : ((7^2 - 3^2)^4) = 2560000 := by
  sorry

end competition_problem_l1725_172568


namespace harkamal_grapes_purchase_l1725_172510

/-- The amount of grapes purchased by Harkamal -/
def grapes_kg : ℝ := 8

/-- The cost of grapes per kg -/
def grapes_cost_per_kg : ℝ := 70

/-- The cost of mangoes per kg -/
def mangoes_cost_per_kg : ℝ := 60

/-- The amount of mangoes purchased by Harkamal -/
def mangoes_kg : ℝ := 9

/-- The total amount paid by Harkamal -/
def total_paid : ℝ := 1100

theorem harkamal_grapes_purchase :
  grapes_kg * grapes_cost_per_kg + mangoes_kg * mangoes_cost_per_kg = total_paid :=
by sorry

end harkamal_grapes_purchase_l1725_172510


namespace compound_ratio_proof_l1725_172599

theorem compound_ratio_proof : 
  let r1 : ℚ := 2/3
  let r2 : ℚ := 6/7
  let r3 : ℚ := 1/3
  let r4 : ℚ := 3/8
  (r1 * r2 * r3 * r4 : ℚ) = 0.07142857142857142 :=
by sorry

end compound_ratio_proof_l1725_172599


namespace count_valid_programs_l1725_172578

/-- Represents the available courses --/
inductive Course
| English
| Algebra
| Geometry
| History
| Art
| Latin
| Science

/-- Checks if a course is a mathematics course --/
def isMathCourse (c : Course) : Bool :=
  match c with
  | Course.Algebra => true
  | Course.Geometry => true
  | _ => false

/-- Checks if a course is a science course --/
def isScienceCourse (c : Course) : Bool :=
  match c with
  | Course.Science => true
  | _ => false

/-- Represents a program of 4 courses --/
structure Program :=
  (courses : Finset Course)
  (size_eq : courses.card = 4)
  (has_english : Course.English ∈ courses)
  (has_math : ∃ c ∈ courses, isMathCourse c)
  (has_science : ∃ c ∈ courses, isScienceCourse c)

/-- The set of all valid programs --/
def validPrograms : Finset Program := sorry

theorem count_valid_programs :
  validPrograms.card = 19 := by sorry

end count_valid_programs_l1725_172578


namespace nested_subtract_201_l1725_172509

/-- Recursive function to represent nested subtractions -/
def nestedSubtract (x : ℝ) : ℕ → ℝ
  | 0 => x - 1
  | n + 1 => x - nestedSubtract x n

/-- Theorem stating that the nested subtraction equals 1 iff x = 201 -/
theorem nested_subtract_201 (x : ℝ) :
  nestedSubtract x 199 = 1 ↔ x = 201 := by
  sorry

#check nested_subtract_201

end nested_subtract_201_l1725_172509


namespace cuboid_edge_lengths_l1725_172519

theorem cuboid_edge_lengths (a b c : ℝ) : 
  (a * b : ℝ) / (b * c) = 16 / 21 →
  (a * b : ℝ) / (a * c) = 16 / 28 →
  a^2 + b^2 + c^2 = 29^2 →
  a = 16 ∧ b = 12 ∧ c = 21 := by
  sorry

end cuboid_edge_lengths_l1725_172519


namespace trig_identity_l1725_172583

theorem trig_identity : (1 / (2 * Real.sin (10 * π / 180))) - 2 * Real.sin (70 * π / 180) = 1 := by
  sorry

end trig_identity_l1725_172583


namespace square_area_increase_l1725_172588

/-- Given a square with initial side length 4, if the side length increases by x
    and the area increases by y, then y = x^2 + 8x -/
theorem square_area_increase (x y : ℝ) : 
  (4 + x)^2 - 4^2 = y → y = x^2 + 8*x := by
  sorry

end square_area_increase_l1725_172588


namespace sugar_per_larger_cookie_l1725_172541

/-- Proves that if 40 cookies each use 1/8 cup of sugar, and the same total amount of sugar
    is used to make 25 larger cookies, then each larger cookie will contain 1/5 cup of sugar. -/
theorem sugar_per_larger_cookie :
  let small_cookies : ℕ := 40
  let large_cookies : ℕ := 25
  let sugar_per_small : ℚ := 1 / 8
  let total_sugar : ℚ := small_cookies * sugar_per_small
  let sugar_per_large : ℚ := total_sugar / large_cookies
  sugar_per_large = 1 / 5 := by
  sorry

end sugar_per_larger_cookie_l1725_172541


namespace complex_magnitude_proof_l1725_172549

theorem complex_magnitude_proof (z : ℂ) (h : z = 1 + Complex.I) : 
  Complex.abs (z^2 - 2*z) = 2 := by
  sorry

end complex_magnitude_proof_l1725_172549


namespace power_product_equality_l1725_172553

theorem power_product_equality : 2^3 * 3 * 5^3 * 7 = 21000 := by
  sorry

end power_product_equality_l1725_172553


namespace cans_display_rows_l1725_172529

/-- The number of cans in the nth row of the display -/
def cans_in_row (n : ℕ) : ℕ := 2 * n + 1

/-- The total number of cans in a display with n rows -/
def total_cans (n : ℕ) : ℕ := n * (n + 2)

/-- The number of rows in the display -/
def num_rows : ℕ := 12

theorem cans_display_rows :
  (cans_in_row 1 = 3) ∧
  (∀ n : ℕ, n > 0 → cans_in_row (n + 1) = cans_in_row n + 2) ∧
  (total_cans num_rows = 169) ∧
  (∀ m : ℕ, m ≠ num_rows → total_cans m ≠ 169) :=
by sorry

end cans_display_rows_l1725_172529


namespace initial_socks_count_l1725_172591

theorem initial_socks_count (S : ℕ) : 
  (S ≥ 4) →
  (∃ (remaining : ℕ), remaining = S - 4) →
  (∃ (after_donation : ℕ), after_donation = (remaining : ℚ) * (1 / 3 : ℚ)) →
  (after_donation + 13 = 25) →
  S = 40 :=
by sorry

end initial_socks_count_l1725_172591


namespace fixed_point_of_exponential_function_l1725_172566

theorem fixed_point_of_exponential_function (a : ℝ) (ha : a > 0) :
  let f : ℝ → ℝ := λ x ↦ a^(x-1) + 3
  f 1 = 4 := by sorry

end fixed_point_of_exponential_function_l1725_172566


namespace parabola_line_intersection_theorem_l1725_172544

/-- Represents a parabola with focus on the x-axis and vertex at the origin -/
structure Parabola where
  p : ℝ
  equation : ℝ → ℝ → Prop := λ x y => y^2 = 2 * p * x

/-- Represents a line in the form x = my + b -/
structure Line where
  m : ℝ
  b : ℝ
  equation : ℝ → ℝ → Prop := λ x y => x = m * y + b

/-- Theorem stating the existence of a specific line intersecting the parabola -/
theorem parabola_line_intersection_theorem (C : Parabola) (h1 : C.equation 2 1) :
  ∃ (l : Line), l.b = 2 ∧
    (∃ (A B : ℝ × ℝ),
      C.equation A.1 A.2 ∧
      C.equation B.1 B.2 ∧
      l.equation A.1 A.2 ∧
      l.equation B.1 B.2 ∧
      let M := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)
      let N := (M.1, Real.sqrt (2 * C.p * M.1))
      (N.1 - A.1) * (N.1 - B.1) + (N.2 - A.2) * (N.2 - B.2) = 0) ∧
    (l.m = 2 ∨ l.m = -2) := by
  sorry

end parabola_line_intersection_theorem_l1725_172544


namespace triangle_abc_properties_l1725_172536

-- Define the triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the theorem
theorem triangle_abc_properties (t : Triangle) 
  (h1 : t.a = Real.sqrt 2)
  (h2 : t.b = Real.sqrt 3)
  (h3 : t.B = 60 * π / 180) :
  t.A = 45 * π / 180 ∧ 
  t.C = 75 * π / 180 ∧ 
  t.c = (Real.sqrt 2 + Real.sqrt 6) / 2 := by
  sorry


end triangle_abc_properties_l1725_172536


namespace min_burgers_recovery_l1725_172504

/-- The minimum whole number of burgers Sarah must sell to recover her initial investment -/
def min_burgers : ℕ := 637

/-- Sarah's initial investment in dollars -/
def initial_investment : ℕ := 7000

/-- Sarah's earnings per burger in dollars -/
def earnings_per_burger : ℕ := 15

/-- Sarah's ingredient cost per burger in dollars -/
def ingredient_cost_per_burger : ℕ := 4

/-- Theorem stating that min_burgers is the minimum whole number of burgers
    Sarah must sell to recover her initial investment -/
theorem min_burgers_recovery :
  (min_burgers * (earnings_per_burger - ingredient_cost_per_burger) ≥ initial_investment) ∧
  ∀ n : ℕ, n < min_burgers → n * (earnings_per_burger - ingredient_cost_per_burger) < initial_investment :=
by sorry

end min_burgers_recovery_l1725_172504


namespace algebraic_identities_l1725_172514

theorem algebraic_identities (x : ℝ) (h : x + 1/x = 8) :
  (x^2 + 1/x^2 = 62) ∧ (x^3 + 1/x^3 = 488) := by
  sorry

end algebraic_identities_l1725_172514


namespace books_bought_at_yard_sale_l1725_172539

-- Define the initial number of books
def initial_books : ℕ := 35

-- Define the final number of books
def final_books : ℕ := 56

-- Theorem: The number of books bought at the yard sale is 21
theorem books_bought_at_yard_sale :
  final_books - initial_books = 21 :=
by sorry

end books_bought_at_yard_sale_l1725_172539


namespace isosceles_triangle_area_l1725_172586

/-- An isosceles triangle PQR with given side lengths and altitude properties -/
structure IsoscelesTriangle where
  /-- Length of equal sides PQ and PR -/
  side : ℝ
  /-- Length of base QR -/
  base : ℝ
  /-- Altitude PS bisects base QR -/
  altitude_bisects_base : True

/-- The area of the isosceles triangle PQR is 360 square units -/
theorem isosceles_triangle_area
  (t : IsoscelesTriangle)
  (h1 : t.side = 41)
  (h2 : t.base = 18) :
  t.side * t.base / 2 = 360 :=
sorry

end isosceles_triangle_area_l1725_172586


namespace M_mod_1500_l1725_172572

/-- A sequence of positive integers whose binary representation has exactly 9 ones -/
def T : Nat → Nat := sorry

/-- The 1500th number in the sequence T -/
def M : Nat := T 1500

/-- The remainder when M is divided by 1500 -/
theorem M_mod_1500 : M % 1500 = 500 := by sorry

end M_mod_1500_l1725_172572


namespace factorization_of_8a_squared_minus_2_l1725_172538

theorem factorization_of_8a_squared_minus_2 (a : ℝ) : 8 * a^2 - 2 = 2 * (2*a + 1) * (2*a - 1) := by
  sorry

end factorization_of_8a_squared_minus_2_l1725_172538


namespace inequality_solution_l1725_172507

theorem inequality_solution (x : ℝ) : 
  (9 * x^2 + 18 * x - 60) / ((3 * x - 4) * (x + 5)) < 2 ↔ 
  (x > -5 ∧ x < -20/3) ∨ (x > 2/3 ∧ x < 4/3) := by
sorry

end inequality_solution_l1725_172507


namespace frank_candies_l1725_172567

def frank_tickets_game1 : ℕ := 33
def frank_tickets_game2 : ℕ := 9
def candy_cost : ℕ := 6

theorem frank_candies : 
  (frank_tickets_game1 + frank_tickets_game2) / candy_cost = 7 := by sorry

end frank_candies_l1725_172567


namespace constant_term_binomial_expansion_l1725_172535

theorem constant_term_binomial_expansion :
  ∃ (c : ℝ), c = 7 ∧ 
  ∀ (x : ℝ), x ≠ 0 → 
  ∃ (f : ℝ → ℝ), (λ x => (x^(1/3) + 1/(2*x))^8) = 
    (λ x => c + f x) ∧ (∀ (y : ℝ), y ≠ 0 → f y ≠ 0) :=
by sorry

end constant_term_binomial_expansion_l1725_172535


namespace inequality_system_solution_set_l1725_172530

theorem inequality_system_solution_set (x : ℝ) :
  (x + 1 > 0 ∧ x - 3 > 0) ↔ x > 3 := by
sorry

end inequality_system_solution_set_l1725_172530


namespace simplification_proofs_l1725_172543

theorem simplification_proofs :
  (∀ x : ℝ, x ≥ 0 → Real.sqrt (x^2) = x) ∧
  ((5 * Real.sqrt 5)^2 = 125) ∧
  (Real.sqrt ((-1/7)^2) = 1/7) ∧
  ((-Real.sqrt (2/3))^2 = 2/3) := by
  sorry

end simplification_proofs_l1725_172543


namespace smallest_number_with_remainder_l1725_172593

theorem smallest_number_with_remainder (n : ℕ) : 
  n = 1996 ↔ 
  (n > 1992 ∧ 
   n % 9 = 7 ∧ 
   ∀ m, m > 1992 ∧ m % 9 = 7 → n ≤ m) :=
by sorry

end smallest_number_with_remainder_l1725_172593


namespace arctan_equation_solution_l1725_172597

theorem arctan_equation_solution :
  ∃ y : ℝ, 2 * Real.arctan (1/3) + Real.arctan (1/15) + Real.arctan (1/y) = π/3 ∧ y = 13.25 := by
  sorry

end arctan_equation_solution_l1725_172597


namespace flea_can_reach_all_points_l1725_172595

/-- The length of the k-th jump for the flea -/
def jumpLength (k : ℕ) : ℕ := 2^k + 1

/-- A jump is represented by its length and direction -/
structure Jump where
  length : ℕ
  direction : Bool  -- true for right, false for left

/-- The final position after a sequence of jumps -/
def finalPosition (jumps : List Jump) : ℤ :=
  jumps.foldl (fun pos jump => 
    if jump.direction then pos + jump.length else pos - jump.length) 0

/-- Theorem: For any natural number n, there exists a sequence of jumps
    that allows the flea to move from point 0 to point n -/
theorem flea_can_reach_all_points (n : ℕ) : 
  ∃ (jumps : List Jump), finalPosition jumps = n := by
  sorry

end flea_can_reach_all_points_l1725_172595


namespace spinsters_cats_ratio_l1725_172582

theorem spinsters_cats_ratio : 
  ∀ (spinsters cats : ℕ),
    spinsters = 12 →
    cats = spinsters + 42 →
    (spinsters : ℚ) / cats = 2 / 9 := by
  sorry

end spinsters_cats_ratio_l1725_172582


namespace smallest_solution_of_equation_l1725_172500

theorem smallest_solution_of_equation : 
  ∃ x : ℝ, x = -15 ∧ 
  (∀ y : ℝ, 3 * y^2 + 39 * y - 75 = y * (y + 16) → y ≥ x) :=
by sorry

end smallest_solution_of_equation_l1725_172500


namespace cubic_sum_theorem_l1725_172551

theorem cubic_sum_theorem (x y z : ℝ) 
  (h1 : x + y + z = 3) 
  (h2 : x*y + y*z + z*x = -3) 
  (h3 : x*y*z = -3) : 
  x^3 + y^3 + z^3 = 45 := by
sorry

end cubic_sum_theorem_l1725_172551


namespace quadratic_equation_linear_coefficient_l1725_172596

/-- The exponent of x in the first term of the equation -/
def exponent (m : ℝ) : ℝ := m^2 - 2*m - 1

/-- The equation is quadratic when the exponent equals 2 -/
def is_quadratic (m : ℝ) : Prop := exponent m = 2

/-- The coefficient of x in the equation -/
def linear_coefficient (m : ℝ) : ℝ := -m

theorem quadratic_equation_linear_coefficient :
  ∀ m : ℝ, (m ≠ 3) → is_quadratic m → linear_coefficient m = 1 := by sorry

end quadratic_equation_linear_coefficient_l1725_172596


namespace larger_number_proof_l1725_172506

theorem larger_number_proof (x y : ℤ) : 
  y = 2 * x + 3 → 
  x + y = 27 → 
  max x y = 19 := by
sorry

end larger_number_proof_l1725_172506


namespace matthew_initial_cakes_l1725_172557

/-- The number of crackers Matthew had initially -/
def initial_crackers : ℕ := 29

/-- The number of friends Matthew gave crackers and cakes to -/
def num_friends : ℕ := 2

/-- The number of cakes each person ate -/
def cakes_eaten_per_person : ℕ := 15

/-- The number of cakes Matthew had initially -/
def initial_cakes : ℕ := initial_crackers + num_friends * cakes_eaten_per_person

theorem matthew_initial_cakes : initial_cakes = 59 := by sorry

end matthew_initial_cakes_l1725_172557


namespace workshop_ratio_l1725_172550

theorem workshop_ratio (total : ℕ) (novelists : ℕ) (poets : ℕ) : 
  total = 24 → novelists = 15 → poets = total - novelists → 
  ∃ (a b : ℕ), a = 3 ∧ b = 5 ∧ poets * b = novelists * a :=
sorry

end workshop_ratio_l1725_172550


namespace triangle_side_length_l1725_172527

theorem triangle_side_length (A B C : ℝ) (a b c : ℝ) : 
  A = π / 6 →  -- 30° in radians
  B = π / 4 →  -- 45° in radians
  a = Real.sqrt 2 →
  (Real.sin A) * b = (Real.sin B) * a →
  b = 2 := by
sorry

end triangle_side_length_l1725_172527


namespace candy_mixture_price_l1725_172545

theorem candy_mixture_price (price1 price2 : ℝ) (h1 : price1 = 10) (h2 : price2 = 15) : 
  let weight_ratio := 3
  let total_weight := weight_ratio + 1
  let total_cost := price1 * weight_ratio + price2
  total_cost / total_weight = 11.25 := by
sorry

end candy_mixture_price_l1725_172545


namespace train_passing_time_l1725_172511

/-- Proves that a train of given length and speed takes the calculated time to pass a stationary point. -/
theorem train_passing_time (train_length : ℝ) (train_speed_kmh : ℝ) (passing_time : ℝ) : 
  train_length = 280 →
  train_speed_kmh = 72 →
  passing_time = 14 →
  passing_time = train_length / (train_speed_kmh * 1000 / 3600) := by
  sorry

#check train_passing_time

end train_passing_time_l1725_172511


namespace least_k_cube_divisible_by_168_l1725_172573

theorem least_k_cube_divisible_by_168 :
  ∀ k : ℕ, k > 0 → k^3 % 168 = 0 → k ≥ 42 :=
by sorry

end least_k_cube_divisible_by_168_l1725_172573


namespace road_trip_distance_l1725_172533

/-- Represents Rick's road trip with 5 destinations -/
structure RoadTrip where
  leg1 : ℝ
  leg2 : ℝ
  leg3 : ℝ
  leg4 : ℝ
  leg5 : ℝ

/-- Conditions of Rick's road trip -/
def validRoadTrip (trip : RoadTrip) : Prop :=
  trip.leg2 = 2 * trip.leg1 ∧
  trip.leg3 = 40 ∧
  trip.leg3 = trip.leg1 / 2 ∧
  trip.leg4 = 2 * (trip.leg1 + trip.leg2 + trip.leg3) ∧
  trip.leg5 = 1.5 * trip.leg4

/-- The total distance of the road trip -/
def totalDistance (trip : RoadTrip) : ℝ :=
  trip.leg1 + trip.leg2 + trip.leg3 + trip.leg4 + trip.leg5

/-- Theorem stating that the total distance of a valid road trip is 1680 miles -/
theorem road_trip_distance (trip : RoadTrip) (h : validRoadTrip trip) :
  totalDistance trip = 1680 := by
  sorry

end road_trip_distance_l1725_172533


namespace ellipse_equation_l1725_172503

/-- An ellipse with center at the origin, focus on the y-axis, eccentricity 1/2, and focal length 8 has the equation y²/64 + x²/48 = 1 -/
theorem ellipse_equation (x y : ℝ) : 
  let center := (0 : ℝ × ℝ)
  let focus_on_y_axis := true
  let eccentricity := (1 : ℝ) / 2
  let focal_length := (8 : ℝ)
  (y^2 / 64 + x^2 / 48 = 1) ↔ 
    ∃ (a b c : ℝ), 
      a > 0 ∧ b > 0 ∧
      c = focal_length / 2 ∧
      eccentricity = c / a ∧
      b^2 = a^2 - c^2 ∧
      y^2 / a^2 + x^2 / b^2 = 1 :=
by sorry

end ellipse_equation_l1725_172503


namespace gcd_diff_is_square_l1725_172589

theorem gcd_diff_is_square (x y z : ℕ) (h : (1 : ℚ) / x - (1 : ℚ) / y = (1 : ℚ) / z) :
  ∃ k : ℕ, Nat.gcd x (Nat.gcd y z) * (y - x) = k ^ 2 := by
sorry

end gcd_diff_is_square_l1725_172589


namespace cut_cylinder_volume_l1725_172518

/-- Represents a right cylinder with a vertical planar cut -/
structure CutCylinder where
  height : ℝ
  baseRadius : ℝ
  cutArea : ℝ

/-- The volume of the larger piece of a cut cylinder -/
def largerPieceVolume (c : CutCylinder) : ℝ := sorry

theorem cut_cylinder_volume 
  (c : CutCylinder) 
  (h_height : c.height = 20)
  (h_radius : c.baseRadius = 5)
  (h_cut_area : c.cutArea = 100 * Real.sqrt 2) :
  largerPieceVolume c = 250 + 375 * Real.pi := by sorry

end cut_cylinder_volume_l1725_172518


namespace operations_result_l1725_172522

-- Define operation S
def S (a b : ℤ) : ℤ := 4*a + 6*b

-- Define operation T
def T (a b : ℤ) : ℤ := 5*a + 3*b

-- Theorem to prove
theorem operations_result : (S 6 3 = 42) ∧ (T 6 3 = 39) := by
  sorry

end operations_result_l1725_172522


namespace triangle_area_l1725_172561

theorem triangle_area (a b c : ℝ) (h_perimeter : a + b + c = 10 + 2 * Real.sqrt 7)
  (h_ratio : ∃ (k : ℝ), a = 2 * k ∧ b = 3 * k ∧ c = k * Real.sqrt 7) :
  let S := Real.sqrt ((1/4) * (c^2 * a^2 - ((c^2 + a^2 - b^2)/2)^2))
  S = 6 * Real.sqrt 3 := by
sorry

end triangle_area_l1725_172561


namespace jeremy_wednesday_oranges_l1725_172515

/-- The number of oranges Jeremy picked on different days and the total -/
structure OrangePicks where
  monday : ℕ
  tuesday : ℕ
  wednesday : ℕ
  total : ℕ

/-- Given the conditions of Jeremy's orange picking, prove that he picked 70 oranges on Wednesday -/
theorem jeremy_wednesday_oranges (picks : OrangePicks) 
  (h1 : picks.monday = 100)
  (h2 : picks.tuesday = 3 * picks.monday)
  (h3 : picks.total = 470)
  (h4 : picks.total = picks.monday + picks.tuesday + picks.wednesday) :
  picks.wednesday = 70 := by
  sorry


end jeremy_wednesday_oranges_l1725_172515


namespace increase_by_percentage_l1725_172581

theorem increase_by_percentage (initial : ℝ) (percentage : ℝ) (final : ℝ) :
  initial = 250 →
  percentage = 75 →
  final = initial * (1 + percentage / 100) →
  final = 437.5 := by
sorry

end increase_by_percentage_l1725_172581


namespace fraction_simplification_l1725_172564

theorem fraction_simplification :
  (2 * (Real.sqrt 2 + Real.sqrt 6)) / (3 * Real.sqrt (2 + Real.sqrt 3)) = 4 / 3 := by
  sorry

end fraction_simplification_l1725_172564


namespace cubic_sum_theorem_l1725_172577

theorem cubic_sum_theorem (p q r : ℝ) 
  (sum_eq : p + q + r = 4)
  (sum_prod_eq : p * q + p * r + q * r = 6)
  (prod_eq : p * q * r = -8) : 
  p^3 + q^3 + r^3 = 8 := by
sorry

end cubic_sum_theorem_l1725_172577


namespace max_value_sqrt_expression_l1725_172531

theorem max_value_sqrt_expression (x : ℝ) (h1 : 0 ≤ x) (h2 : x ≤ 16) :
  Real.sqrt (x + 64) + Real.sqrt (16 - x) + 2 * Real.sqrt x ≤ 4 * Real.sqrt 5 + 8 :=
by sorry

end max_value_sqrt_expression_l1725_172531


namespace michael_passes_donovan_in_four_laps_l1725_172546

/-- Represents the race conditions and calculates the number of laps for Michael to pass Donovan -/
def raceLaps (trackLength : ℕ) (donovanNormalTime : ℕ) (michaelNormalTime : ℕ) 
              (obstacles : ℕ) (donovanObstacleTime : ℕ) (michaelObstacleTime : ℕ) : ℕ :=
  let donovanLapTime := donovanNormalTime + obstacles * donovanObstacleTime
  let michaelLapTime := michaelNormalTime + obstacles * michaelObstacleTime
  let timeDiffPerLap := donovanLapTime - michaelLapTime
  let lapsToPass := (donovanLapTime + timeDiffPerLap - 1) / timeDiffPerLap
  lapsToPass

/-- Theorem stating that Michael needs 4 laps to pass Donovan under the given conditions -/
theorem michael_passes_donovan_in_four_laps :
  raceLaps 300 45 40 3 10 5 = 4 := by
  sorry

end michael_passes_donovan_in_four_laps_l1725_172546


namespace sum_of_zeros_is_six_l1725_172512

/-- A function with the property f(1-x) = f(3+x) for all x -/
def symmetric_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (1 - x) = f (3 + x)

/-- The set of zeros of a function -/
def zeros (f : ℝ → ℝ) : Set ℝ :=
  {x | f x = 0}

/-- Theorem: If f is a symmetric function with exactly three distinct zeros,
    then the sum of these zeros is 6 -/
theorem sum_of_zeros_is_six (f : ℝ → ℝ) 
    (h_sym : symmetric_function f) 
    (h_zeros : ∃ a b c : ℝ, a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ zeros f = {a, b, c}) :
  ∃ a b c : ℝ, zeros f = {a, b, c} ∧ a + b + c = 6 := by
  sorry

end sum_of_zeros_is_six_l1725_172512


namespace hendricks_guitar_price_l1725_172594

theorem hendricks_guitar_price 
  (gerald_price : ℝ) 
  (discount_percentage : ℝ) 
  (h1 : gerald_price = 250) 
  (h2 : discount_percentage = 20) :
  gerald_price * (1 - discount_percentage / 100) = 200 := by
  sorry

end hendricks_guitar_price_l1725_172594


namespace jamie_score_l1725_172570

theorem jamie_score (team_total : ℝ) (num_players : ℕ) (other_players_avg : ℝ) 
  (h1 : team_total = 60)
  (h2 : num_players = 6)
  (h3 : other_players_avg = 4.8) : 
  team_total - (num_players - 1) * other_players_avg = 36 :=
by sorry

end jamie_score_l1725_172570


namespace space_station_arrangements_count_l1725_172563

/-- The number of ways to distribute n distinguishable objects into k distinguishable boxes,
    with each box containing at least min and at most max objects. -/
def distribute (n k min max : ℕ) : ℕ := sorry

/-- The number of ways to distribute 6 distinguishable objects into 3 distinguishable boxes,
    with each box containing at least 1 and at most 3 objects. -/
def space_station_arrangements : ℕ := distribute 6 3 1 3

theorem space_station_arrangements_count : space_station_arrangements = 450 := by sorry

end space_station_arrangements_count_l1725_172563


namespace energetic_cycling_hours_l1725_172505

theorem energetic_cycling_hours 
  (total_distance : ℝ) 
  (total_time : ℝ) 
  (energetic_speed : ℝ) 
  (fatigued_speed : ℝ) 
  (h1 : total_distance = 150) 
  (h2 : total_time = 12) 
  (h3 : energetic_speed = 15) 
  (h4 : fatigued_speed = 10) : 
  ∃ (energetic_hours : ℝ), 
    energetic_hours * energetic_speed + (total_time - energetic_hours) * fatigued_speed = total_distance ∧ 
    energetic_hours = 6 := by
  sorry

end energetic_cycling_hours_l1725_172505


namespace path_area_and_cost_l1725_172520

/-- Calculates the area of a rectangular path around a field -/
def path_area (field_length field_width path_width : ℝ) : ℝ :=
  (field_length + 2 * path_width) * (field_width + 2 * path_width) - field_length * field_width

/-- Calculates the cost of constructing a path given its area and cost per unit area -/
def construction_cost (path_area cost_per_unit : ℝ) : ℝ :=
  path_area * cost_per_unit

theorem path_area_and_cost (field_length field_width path_width cost_per_unit : ℝ) 
  (h1 : field_length = 85)
  (h2 : field_width = 55)
  (h3 : path_width = 2.5)
  (h4 : cost_per_unit = 2) : 
  path_area field_length field_width path_width = 725 ∧ 
  construction_cost (path_area field_length field_width path_width) cost_per_unit = 1450 := by
  sorry

#eval path_area 85 55 2.5
#eval construction_cost (path_area 85 55 2.5) 2

end path_area_and_cost_l1725_172520


namespace pencil_cost_theorem_l1725_172556

/-- Calculates the average cost per pencil in cents, rounded to the nearest cent -/
def averageCostPerPencil (pencilCount : ℕ) (pencilCost : ℚ) (shippingCost : ℚ) (discount : ℚ) : ℕ :=
  let totalCost := pencilCost + shippingCost - discount
  let totalCostInCents := (totalCost * 100).floor
  ((totalCostInCents + pencilCount / 2) / pencilCount).toNat

theorem pencil_cost_theorem :
  let pencilCount : ℕ := 150
  let pencilCost : ℚ := 15.5
  let shippingCost : ℚ := 5.75
  let discount : ℚ := 1

  averageCostPerPencil pencilCount pencilCost shippingCost discount = 14 := by
    sorry

#eval averageCostPerPencil 150 15.5 5.75 1

end pencil_cost_theorem_l1725_172556


namespace cab_driver_income_l1725_172579

def average_income : ℝ := 440
def num_days : ℕ := 5
def known_incomes : List ℝ := [250, 650, 400, 500]

theorem cab_driver_income :
  let total_income := average_income * num_days
  let known_total := known_incomes.sum
  total_income - known_total = 400 := by sorry

end cab_driver_income_l1725_172579


namespace ratio_equality_l1725_172560

theorem ratio_equality (a b c x y z : ℝ) 
  (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c) 
  (pos_x : 0 < x) (pos_y : 0 < y) (pos_z : 0 < z)
  (sum_squares_abc : a^2 + b^2 + c^2 = 49)
  (sum_squares_xyz : x^2 + y^2 + z^2 = 64)
  (dot_product : a*x + b*y + c*z = 56) :
  (a + b + c) / (x + y + z) = 7/8 := by
sorry

end ratio_equality_l1725_172560


namespace line_perpendicular_to_plane_l1725_172587

-- Define the types for plane and line
variable (Plane Line : Type)

-- Define the relations
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Line → Line → Prop)

-- State the theorem
theorem line_perpendicular_to_plane 
  (α : Plane) (a b : Line) :
  perpendicular a α → parallel a b → perpendicular b α := by
  sorry

end line_perpendicular_to_plane_l1725_172587


namespace parallelogram_base_length_l1725_172598

-- Define the parallelogram properties
def parallelogram_area : ℝ := 360
def parallelogram_height : ℝ := 12

-- Theorem statement
theorem parallelogram_base_length :
  parallelogram_area / parallelogram_height = 30 :=
by sorry

end parallelogram_base_length_l1725_172598


namespace iphone_price_decrease_l1725_172559

theorem iphone_price_decrease (initial_price : ℝ) (second_month_decrease : ℝ) (final_price : ℝ) :
  initial_price = 1000 →
  second_month_decrease = 20 →
  final_price = 720 →
  ∃ (first_month_decrease : ℝ),
    first_month_decrease = 10 ∧
    final_price = initial_price * (1 - first_month_decrease / 100) * (1 - second_month_decrease / 100) :=
by sorry


end iphone_price_decrease_l1725_172559


namespace equal_areas_imply_all_equal_l1725_172548

-- Define a square
structure Square where
  side : ℝ
  area : ℝ
  area_eq : area = side * side

-- Define the four parts of the square
structure SquareParts where
  square : Square
  part1 : ℝ
  part2 : ℝ
  part3 : ℝ
  part4 : ℝ
  sum_eq_area : part1 + part2 + part3 + part4 = square.area

-- Define the perpendicular lines
structure PerpendicularLines where
  line1 : ℝ → ℝ
  line2 : ℝ → ℝ
  perpendicular : ∀ x y, line1 x * line2 y = -1

-- Theorem statement
theorem equal_areas_imply_all_equal (sq : Square) (parts : SquareParts) (lines : PerpendicularLines)
  (h1 : parts.square = sq)
  (h2 : parts.part1 = parts.part2)
  (h3 : parts.part2 = parts.part3)
  (h4 : ∃ x y, x ∈ Set.Icc 0 sq.side ∧ y ∈ Set.Icc 0 sq.side ∧ 
       lines.line1 x = lines.line2 y) :
  parts.part1 = parts.part2 ∧ parts.part2 = parts.part3 ∧ parts.part3 = parts.part4 :=
by sorry

end equal_areas_imply_all_equal_l1725_172548


namespace elijah_score_l1725_172585

/-- Proves that Elijah's score is 43 points given the team's total score,
    number of players, and average score of other players. -/
theorem elijah_score (total_score : ℕ) (num_players : ℕ) (other_avg : ℕ) 
  (h1 : total_score = 85)
  (h2 : num_players = 8)
  (h3 : other_avg = 6) :
  total_score - (num_players - 1) * other_avg = 43 := by
  sorry

#check elijah_score

end elijah_score_l1725_172585


namespace remainder_sum_mod_three_l1725_172592

theorem remainder_sum_mod_three
  (a b c d : ℕ)
  (ha : a % 6 = 4)
  (hb : b % 6 = 4)
  (hc : c % 6 = 4)
  (hd : d % 6 = 4) :
  (a + b + c + d) % 3 = 1 := by
  sorry

end remainder_sum_mod_three_l1725_172592


namespace fish_to_buy_l1725_172508

def current_fish : ℕ := 212
def desired_total : ℕ := 280

theorem fish_to_buy : desired_total - current_fish = 68 := by sorry

end fish_to_buy_l1725_172508


namespace arthur_walk_distance_l1725_172526

/-- The number of blocks Arthur walks east -/
def blocks_east : ℕ := 8

/-- The number of blocks Arthur walks north -/
def blocks_north : ℕ := 15

/-- The number of blocks Arthur walks west -/
def blocks_west : ℕ := 3

/-- The length of each block in miles -/
def block_length : ℚ := 1/2

/-- The total distance Arthur walks in miles -/
def total_distance : ℚ := (blocks_east + blocks_north + blocks_west : ℚ) * block_length

theorem arthur_walk_distance : total_distance = 13 := by
  sorry

end arthur_walk_distance_l1725_172526


namespace ellipse_focus_l1725_172575

theorem ellipse_focus (center : ℝ × ℝ) (major_axis : ℝ) (minor_axis : ℝ) :
  center = (3, -1) →
  major_axis = 6 →
  minor_axis = 4 →
  let focus_distance := Real.sqrt ((major_axis / 2)^2 - (minor_axis / 2)^2)
  let focus_x := center.1 + focus_distance
  (focus_x, center.2) = (3 + Real.sqrt 5, -1) :=
by sorry

end ellipse_focus_l1725_172575


namespace indefinite_integral_proof_l1725_172542

theorem indefinite_integral_proof (x : ℝ) (h1 : x ≠ -2) (h2 : x ≠ 2) :
  let f := fun (x : ℝ) => (x^3 - 6*x^2 + 11*x - 10) / ((x+2)*(x-2)^3)
  let F := fun (x : ℝ) => Real.log (abs (x+2)) + 1 / (2*(x-2)^2)
  deriv F x = f x := by sorry

end indefinite_integral_proof_l1725_172542
