import Mathlib

namespace NUMINAMATH_CALUDE_parabola_with_directrix_neg_two_l234_23478

/-- A parabola is defined by its directrix and focus. -/
structure Parabola where
  directrix : ℝ  -- y-coordinate of the directrix
  focus : ℝ × ℝ  -- (x, y) coordinates of the focus

/-- The standard equation of a parabola with a vertical axis of symmetry. -/
def standardEquation (p : Parabola) : Prop :=
  ∀ x y : ℝ, (x ^ 2 = 4 * p.focus.2 * y) ↔ (y - p.focus.2) ^ 2 = (x - p.focus.1) ^ 2 + (y - p.directrix) ^ 2

theorem parabola_with_directrix_neg_two (p : Parabola) 
  (h : p.directrix = -2) : 
  standardEquation p ↔ ∀ x y : ℝ, x ^ 2 = 8 * y :=
sorry

end NUMINAMATH_CALUDE_parabola_with_directrix_neg_two_l234_23478


namespace NUMINAMATH_CALUDE_square_sum_product_l234_23463

theorem square_sum_product (x : ℝ) :
  (Real.sqrt (9 + x) + Real.sqrt (16 - x) = 8) →
  ((9 + x) * (16 - x) = 380.25) := by
  sorry

end NUMINAMATH_CALUDE_square_sum_product_l234_23463


namespace NUMINAMATH_CALUDE_negation_of_proposition_l234_23435

theorem negation_of_proposition :
  (¬ ∀ x : ℝ, x > 0 → x^2 - x ≤ 0) ↔ (∃ x : ℝ, x > 0 ∧ x^2 - x > 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_proposition_l234_23435


namespace NUMINAMATH_CALUDE_function_minimum_implies_inequality_l234_23412

/-- Given a function f(x) = ax^2 + bx - ln(x) where a, b ∈ ℝ,
    if a > 0 and for any x > 0, f(x) ≥ f(1), then ln(a) < -2b -/
theorem function_minimum_implies_inequality (a b : ℝ) :
  a > 0 →
  (∀ x > 0, a * x^2 + b * x - Real.log x ≥ a + b) →
  Real.log a < -2 * b :=
by sorry

end NUMINAMATH_CALUDE_function_minimum_implies_inequality_l234_23412


namespace NUMINAMATH_CALUDE_pascal_high_school_students_l234_23432

/-- The number of students at Pascal High School -/
def total_students : ℕ := sorry

/-- The number of students who went on the first trip -/
def first_trip : ℕ := sorry

/-- The number of students who went on the second trip -/
def second_trip : ℕ := sorry

/-- The number of students who went on the third trip -/
def third_trip : ℕ := sorry

/-- The number of students who went on all three trips -/
def all_three_trips : ℕ := 160

theorem pascal_high_school_students :
  (first_trip = total_students / 2) ∧
  (second_trip = (total_students * 4) / 5) ∧
  (third_trip = (total_students * 9) / 10) ∧
  (all_three_trips = 160) ∧
  (∀ s, s ∈ Finset.range total_students →
    (s ∈ Finset.range first_trip ∧ s ∈ Finset.range second_trip) ∨
    (s ∈ Finset.range first_trip ∧ s ∈ Finset.range third_trip) ∨
    (s ∈ Finset.range second_trip ∧ s ∈ Finset.range third_trip) ∨
    (s ∈ Finset.range all_three_trips)) →
  total_students = 800 := by sorry

end NUMINAMATH_CALUDE_pascal_high_school_students_l234_23432


namespace NUMINAMATH_CALUDE_centroid_altitude_distance_l234_23401

-- Define the triangle ABC
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define the lengths of the sides
def side_lengths (t : Triangle) : ℝ × ℝ × ℝ :=
  (7, 15, 20)

-- Define the centroid G
def centroid (t : Triangle) : ℝ × ℝ := sorry

-- Define the foot of the altitude P
def altitude_foot (t : Triangle) (G : ℝ × ℝ) : ℝ × ℝ := sorry

-- Define the distance between two points
def distance (p1 p2 : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem centroid_altitude_distance (t : Triangle) :
  let G := centroid t
  let P := altitude_foot t G
  distance G P = 1.4 := by sorry

end NUMINAMATH_CALUDE_centroid_altitude_distance_l234_23401


namespace NUMINAMATH_CALUDE_product_of_second_and_third_smallest_l234_23480

theorem product_of_second_and_third_smallest (a b c : ℕ) (h1 : a = 10) (h2 : b = 11) (h3 : c = 12) :
  (max a (max b c)) * (max (min a b) (min (max a b) c)) = 132 := by
  sorry

end NUMINAMATH_CALUDE_product_of_second_and_third_smallest_l234_23480


namespace NUMINAMATH_CALUDE_octal_sum_units_digit_l234_23472

/-- The units digit of the sum of two octal numbers -/
def octal_units_digit_sum (a b : ℕ) : ℕ :=
  (a % 8 + b % 8) % 8

/-- Theorem: The units digit of 53₈ + 64₈ in base 8 is 7 -/
theorem octal_sum_units_digit :
  octal_units_digit_sum 53 64 = 7 := by
  sorry

end NUMINAMATH_CALUDE_octal_sum_units_digit_l234_23472


namespace NUMINAMATH_CALUDE_tangent_line_y_intercept_l234_23474

/-- The y-intercept of the tangent line to y = x^3 + 11 at (1,12) is 9 -/
theorem tangent_line_y_intercept : 
  let f (x : ℝ) := x^3 + 11
  let P : ℝ × ℝ := (1, 12)
  let m := (deriv f) P.1
  let tangent_line (x : ℝ) := m * (x - P.1) + P.2
  tangent_line 0 = 9 := by sorry

end NUMINAMATH_CALUDE_tangent_line_y_intercept_l234_23474


namespace NUMINAMATH_CALUDE_basic_structures_are_sequential_conditional_loop_modular_not_basic_structure_l234_23464

/-- The set of basic algorithm structures -/
inductive AlgorithmStructure
  | Sequential
  | Conditional
  | Loop
  | Modular

/-- The set of basic algorithm structures contains exactly Sequential, Conditional, and Loop -/
def basic_structures : Set AlgorithmStructure :=
  {AlgorithmStructure.Sequential, AlgorithmStructure.Conditional, AlgorithmStructure.Loop}

/-- The theorem stating that the basic structures are exactly Sequential, Conditional, and Loop -/
theorem basic_structures_are_sequential_conditional_loop :
  basic_structures = {AlgorithmStructure.Sequential, AlgorithmStructure.Conditional, AlgorithmStructure.Loop} :=
by sorry

/-- The theorem stating that Modular is not a basic structure -/
theorem modular_not_basic_structure :
  AlgorithmStructure.Modular ∉ basic_structures :=
by sorry

end NUMINAMATH_CALUDE_basic_structures_are_sequential_conditional_loop_modular_not_basic_structure_l234_23464


namespace NUMINAMATH_CALUDE_quadratic_radical_rule_l234_23499

theorem quadratic_radical_rule (n : ℕ+) : 
  Real.sqrt (n + 1 / (n + 2)) = (n + 1) * Real.sqrt (1 / (n + 2)) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_radical_rule_l234_23499


namespace NUMINAMATH_CALUDE_exists_multiple_with_odd_digit_sum_l234_23448

/-- Sum of digits of a natural number in decimal notation -/
def sumOfDigits (n : ℕ) : ℕ := sorry

/-- Predicate to check if a number is odd -/
def isOdd (n : ℕ) : Prop := n % 2 = 1

/-- Theorem: For any natural number M, there exists a multiple of M with an odd sum of digits -/
theorem exists_multiple_with_odd_digit_sum (M : ℕ) : 
  ∃ k : ℕ, M ∣ k ∧ isOdd (sumOfDigits k) := by sorry

end NUMINAMATH_CALUDE_exists_multiple_with_odd_digit_sum_l234_23448


namespace NUMINAMATH_CALUDE_polynomial_remainder_l234_23417

/-- Given a polynomial q(x) = Ax^6 + Bx^4 + Cx^2 + 10, if the remainder when
    q(x) is divided by x - 2 is 20, then the remainder when q(x) is divided
    by x + 2 is also 20. -/
theorem polynomial_remainder (A B C : ℝ) : 
  let q : ℝ → ℝ := λ x ↦ A * x^6 + B * x^4 + C * x^2 + 10
  (q 2 = 20) → (q (-2) = 20) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_remainder_l234_23417


namespace NUMINAMATH_CALUDE_necklace_profit_l234_23455

/-- Calculate the profit from selling necklaces -/
theorem necklace_profit
  (charms_per_necklace : ℕ)
  (charm_cost : ℕ)
  (selling_price : ℕ)
  (necklaces_sold : ℕ)
  (h1 : charms_per_necklace = 10)
  (h2 : charm_cost = 15)
  (h3 : selling_price = 200)
  (h4 : necklaces_sold = 30) :
  (selling_price - charms_per_necklace * charm_cost) * necklaces_sold = 1500 :=
by sorry

end NUMINAMATH_CALUDE_necklace_profit_l234_23455


namespace NUMINAMATH_CALUDE_remainder_theorem_remainder_is_16_l234_23423

/-- The polynomial f(x) = x^4 - 6x^3 + 11x^2 + 12x - 20 -/
def f (x : ℝ) : ℝ := x^4 - 6*x^3 + 11*x^2 + 12*x - 20

/-- The remainder when f(x) is divided by (x - 2) is equal to f(2) -/
theorem remainder_theorem (x : ℝ) : 
  ∃ (q : ℝ → ℝ), f x = (x - 2) * q x + f 2 := by sorry

/-- The remainder when x^4 - 6x^3 + 11x^2 + 12x - 20 is divided by x - 2 is 16 -/
theorem remainder_is_16 : f 2 = 16 := by sorry

end NUMINAMATH_CALUDE_remainder_theorem_remainder_is_16_l234_23423


namespace NUMINAMATH_CALUDE_books_lost_l234_23449

/-- Given that Sandy has 10 books, Tim has 33 books, and they now have 19 books together,
    prove that Benny lost 24 books. -/
theorem books_lost (sandy_books tim_books total_books_now : ℕ)
  (h1 : sandy_books = 10)
  (h2 : tim_books = 33)
  (h3 : total_books_now = 19) :
  sandy_books + tim_books - total_books_now = 24 := by
  sorry

end NUMINAMATH_CALUDE_books_lost_l234_23449


namespace NUMINAMATH_CALUDE_max_customers_interviewed_l234_23447

theorem max_customers_interviewed (total : ℕ) (impulsive : ℕ) (ad_influence_ratio : ℚ) (consultant_ratio : ℚ) : 
  total ≤ 50 ∧
  impulsive = 7 ∧
  ad_influence_ratio = 3/4 ∧
  consultant_ratio = 1/3 ∧
  (∃ k : ℕ, total - impulsive = 4 * k) ∧
  (ad_influence_ratio * (total - impulsive)).isInt ∧
  (consultant_ratio * ad_influence_ratio * (total - impulsive)).isInt →
  total ≤ 47 ∧ 
  (∃ max_total : ℕ, max_total = 47 ∧ 
    max_total ≤ 50 ∧
    (∃ k : ℕ, max_total - impulsive = 4 * k) ∧
    (ad_influence_ratio * (max_total - impulsive)).isInt ∧
    (consultant_ratio * ad_influence_ratio * (max_total - impulsive)).isInt) :=
by sorry

end NUMINAMATH_CALUDE_max_customers_interviewed_l234_23447


namespace NUMINAMATH_CALUDE_isosceles_triangle_sides_l234_23438

/-- An isosceles triangle with perimeter 20 -/
structure IsoscelesTriangle where
  base : ℝ
  leg : ℝ
  perimeter_eq : base + 2 * leg = 20

/-- The lengths of the sides when each leg is twice the base -/
def legsTwiceBase (t : IsoscelesTriangle) : Prop :=
  t.leg = 2 * t.base ∧ t.base = 4 ∧ t.leg = 8

/-- The lengths of the sides when one side is 6 -/
def oneSideSix (t : IsoscelesTriangle) : Prop :=
  (t.base = 6 ∧ t.leg = 7) ∨ (t.base = 8 ∧ t.leg = 6)

theorem isosceles_triangle_sides :
  (∀ t : IsoscelesTriangle, t.leg = 2 * t.base → legsTwiceBase t) ∧
  (∀ t : IsoscelesTriangle, (t.base = 6 ∨ t.leg = 6) → oneSideSix t) := by sorry

end NUMINAMATH_CALUDE_isosceles_triangle_sides_l234_23438


namespace NUMINAMATH_CALUDE_max_number_after_two_moves_l234_23482

def initial_number : ℕ := 4597

def swap_adjacent_digits (n : ℕ) (i : ℕ) : ℕ := 
  sorry

def subtract_100 (n : ℕ) : ℕ := 
  sorry

def make_move (n : ℕ) (i : ℕ) : ℕ := 
  subtract_100 (swap_adjacent_digits n i)

def max_after_moves (n : ℕ) (moves : ℕ) : ℕ := 
  sorry

theorem max_number_after_two_moves : 
  max_after_moves initial_number 2 = 4659 := by
  sorry

end NUMINAMATH_CALUDE_max_number_after_two_moves_l234_23482


namespace NUMINAMATH_CALUDE_negation_equivalence_l234_23409

theorem negation_equivalence :
  (¬ ∃ x : ℝ, x^3 - x^2 + 1 ≤ 0) ↔ (∀ x : ℝ, x^3 - x^2 + 1 > 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_equivalence_l234_23409


namespace NUMINAMATH_CALUDE_absolute_value_and_roots_calculation_l234_23410

theorem absolute_value_and_roots_calculation : 
  |(-3)| + (1/2)^0 - Real.sqrt 8 * Real.sqrt 2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_and_roots_calculation_l234_23410


namespace NUMINAMATH_CALUDE_max_reciprocal_sum_of_zeros_l234_23489

/-- Piecewise function f(x) as defined in the problem -/
noncomputable def f (k : ℝ) (x : ℝ) : ℝ :=
  if 0 < x ∧ x ≤ 1 then k * x^2 + 2 * x - 1
  else if x > 1 then k * x + 1
  else 0  -- Define f(x) as 0 for x ≤ 0 to make it total

/-- Theorem stating the maximum value of 1/x₁ + 1/x₂ -/
theorem max_reciprocal_sum_of_zeros (k : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f k x₁ = 0 ∧ f k x₂ = 0) →
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f k x₁ = 0 ∧ f k x₂ = 0 ∧ 1/x₁ + 1/x₂ ≤ 9/4) ∧
  (∃ k₀ : ℝ, ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f k₀ x₁ = 0 ∧ f k₀ x₂ = 0 ∧ 1/x₁ + 1/x₂ = 9/4) :=
by sorry


end NUMINAMATH_CALUDE_max_reciprocal_sum_of_zeros_l234_23489


namespace NUMINAMATH_CALUDE_equation_solution_l234_23468

theorem equation_solution : 
  {x : ℝ | (5 - 2*x)^(x + 1) = 1} = {-1, 2, 3} := by sorry

end NUMINAMATH_CALUDE_equation_solution_l234_23468


namespace NUMINAMATH_CALUDE_imaginary_part_of_i_over_one_minus_i_l234_23425

/-- The imaginary part of the complex number i / (1 - i) is 1/2 -/
theorem imaginary_part_of_i_over_one_minus_i : Complex.im (Complex.I / (1 - Complex.I)) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_i_over_one_minus_i_l234_23425


namespace NUMINAMATH_CALUDE_inverse_functions_l234_23456

/-- A function type representing the described graphs --/
inductive FunctionGraph
  | Parabola
  | StraightLine
  | HorizontalLine
  | Semicircle
  | CubicFunction

/-- Predicate to determine if a function graph has an inverse --/
def has_inverse (f : FunctionGraph) : Prop :=
  match f with
  | FunctionGraph.StraightLine => true
  | FunctionGraph.Semicircle => true
  | _ => false

/-- Theorem stating which function graphs have inverses --/
theorem inverse_functions (f : FunctionGraph) :
  has_inverse f ↔ (f = FunctionGraph.StraightLine ∨ f = FunctionGraph.Semicircle) :=
sorry

end NUMINAMATH_CALUDE_inverse_functions_l234_23456


namespace NUMINAMATH_CALUDE_largest_mu_inequality_l234_23481

theorem largest_mu_inequality :
  ∃ (μ : ℝ), μ = 3/4 ∧ 
  (∀ (a b c d : ℝ), a ≥ 0 → b ≥ 0 → c ≥ 0 → d ≥ 0 →
    a^2 + b^2 + c^2 + d^2 ≥ a*b + μ*(b*c + d*a) + c*d) ∧
  (∀ (μ' : ℝ), μ' > μ →
    ∃ (a b c d : ℝ), a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ d ≥ 0 ∧
      a^2 + b^2 + c^2 + d^2 < a*b + μ'*(b*c + d*a) + c*d) :=
by sorry

end NUMINAMATH_CALUDE_largest_mu_inequality_l234_23481


namespace NUMINAMATH_CALUDE_complex_arithmetic_equality_l234_23427

theorem complex_arithmetic_equality : 
  2004 - (2003 - 2004 * (2003 - 2002 * (2003 - 2004)^2004)) = 2005 := by
  sorry

end NUMINAMATH_CALUDE_complex_arithmetic_equality_l234_23427


namespace NUMINAMATH_CALUDE_parabola_properties_l234_23492

def parabola (x : ℝ) : ℝ := (x + 2)^2 - 1

theorem parabola_properties :
  (∀ x y : ℝ, parabola x = y → y = (x + 2)^2 - 1) ∧
  (∀ x₁ x₂ : ℝ, x₁ < x₂ → parabola x₁ < parabola x₂) ∧
  (∀ x : ℝ, parabola x ≥ parabola (-2)) ∧
  (parabola (-2) = -1) ∧
  (∀ x₁ x₂ : ℝ, x₁ < -2 ∧ -2 < x₂ → parabola x₁ = parabola x₂ → x₁ + x₂ = -4) ∧
  (∀ x : ℝ, x > -2 → ∀ h : ℝ, h > 0 → parabola (x + h) > parabola x) :=
by sorry

end NUMINAMATH_CALUDE_parabola_properties_l234_23492


namespace NUMINAMATH_CALUDE_sam_read_100_pages_l234_23485

def minimum_assigned : ℕ := 25

def harrison_extra : ℕ := 10

def pam_extra : ℕ := 15

def sam_multiplier : ℕ := 2

def harrison_pages : ℕ := minimum_assigned + harrison_extra

def pam_pages : ℕ := harrison_pages + pam_extra

def sam_pages : ℕ := sam_multiplier * pam_pages

theorem sam_read_100_pages : sam_pages = 100 := by
  sorry

end NUMINAMATH_CALUDE_sam_read_100_pages_l234_23485


namespace NUMINAMATH_CALUDE_chord_length_l234_23446

theorem chord_length (r d : ℝ) (hr : r = 5) (hd : d = 4) :
  let chord_length := 2 * Real.sqrt (r^2 - d^2)
  chord_length = 6 := by sorry

end NUMINAMATH_CALUDE_chord_length_l234_23446


namespace NUMINAMATH_CALUDE_sapphire_percentage_l234_23418

def total_gems : ℕ := 12000
def diamonds : ℕ := 1800
def rubies : ℕ := 4000
def emeralds : ℕ := 3500

def sapphires : ℕ := total_gems - (diamonds + rubies + emeralds)

theorem sapphire_percentage :
  (sapphires : ℚ) / total_gems * 100 = 22.5 := by sorry

end NUMINAMATH_CALUDE_sapphire_percentage_l234_23418


namespace NUMINAMATH_CALUDE_first_day_of_month_l234_23419

-- Define the days of the week
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

def next_day (d : DayOfWeek) : DayOfWeek :=
  match d with
  | DayOfWeek.Sunday => DayOfWeek.Monday
  | DayOfWeek.Monday => DayOfWeek.Tuesday
  | DayOfWeek.Tuesday => DayOfWeek.Wednesday
  | DayOfWeek.Wednesday => DayOfWeek.Thursday
  | DayOfWeek.Thursday => DayOfWeek.Friday
  | DayOfWeek.Friday => DayOfWeek.Saturday
  | DayOfWeek.Saturday => DayOfWeek.Sunday

def day_after_n_days (d : DayOfWeek) (n : Nat) : DayOfWeek :=
  match n with
  | 0 => d
  | n + 1 => next_day (day_after_n_days d n)

theorem first_day_of_month (d : DayOfWeek) :
  day_after_n_days d 27 = DayOfWeek.Tuesday → d = DayOfWeek.Wednesday :=
by sorry

end NUMINAMATH_CALUDE_first_day_of_month_l234_23419


namespace NUMINAMATH_CALUDE_manuscript_revisions_l234_23462

/-- The number of pages revised twice in a manuscript -/
def pages_revised_twice (total_pages : ℕ) (pages_revised_once : ℕ) (cost_first_typing : ℕ) (cost_revision : ℕ) (total_cost : ℕ) : ℕ :=
  let cost_all_first_typing := total_pages * cost_first_typing
  let cost_revisions_once := pages_revised_once * cost_revision
  let remaining_cost := total_cost - cost_all_first_typing - cost_revisions_once
  remaining_cost / (2 * cost_revision)

theorem manuscript_revisions (total_pages : ℕ) (pages_revised_once : ℕ) (cost_first_typing : ℕ) (cost_revision : ℕ) (total_cost : ℕ)
  (h1 : total_pages = 100)
  (h2 : pages_revised_once = 30)
  (h3 : cost_first_typing = 10)
  (h4 : cost_revision = 5)
  (h5 : total_cost = 1350) :
  pages_revised_twice total_pages pages_revised_once cost_first_typing cost_revision total_cost = 20 := by
  sorry

end NUMINAMATH_CALUDE_manuscript_revisions_l234_23462


namespace NUMINAMATH_CALUDE_age_difference_l234_23460

theorem age_difference (A B C : ℤ) (h1 : C = A - 12) : A + B - (B + C) = 12 := by
  sorry

end NUMINAMATH_CALUDE_age_difference_l234_23460


namespace NUMINAMATH_CALUDE_gcd_of_B_is_five_l234_23497

/-- The set of all numbers that can be represented as the sum of five consecutive positive integers -/
def B : Set ℕ := {n : ℕ | ∃ y : ℕ, y > 0 ∧ n = (y - 2) + (y - 1) + y + (y + 1) + (y + 2)}

/-- The greatest common divisor of all numbers in B is 5 -/
theorem gcd_of_B_is_five : ∃ d : ℕ, d > 0 ∧ (∀ n ∈ B, d ∣ n) ∧ (∀ m : ℕ, (∀ n ∈ B, m ∣ n) → m ∣ d) ∧ d = 5 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_B_is_five_l234_23497


namespace NUMINAMATH_CALUDE_num_category_B_prob_both_categories_l234_23477

/- Define the number of category A housekeepers -/
def category_A : ℕ := 12

/- Define the total number of housekeepers selected for training -/
def selected_total : ℕ := 20

/- Define the number of category B housekeepers selected for training -/
def selected_B : ℕ := 16

/- Define the number of category A housekeepers available for hiring -/
def available_A : ℕ := 3

/- Define the number of category B housekeepers available for hiring -/
def available_B : ℕ := 2

/- Theorem for the number of category B housekeepers -/
theorem num_category_B : ∃ x : ℕ, 
  (category_A * selected_B) / (selected_total - selected_B) = x :=
sorry

/- Theorem for the probability of hiring from both categories -/
theorem prob_both_categories : 
  (available_A * available_B) / ((available_A + available_B) * (available_A + available_B - 1) / 2) = 3/5 :=
sorry

end NUMINAMATH_CALUDE_num_category_B_prob_both_categories_l234_23477


namespace NUMINAMATH_CALUDE_minimum_male_students_l234_23411

theorem minimum_male_students (num_benches : ℕ) (students_per_bench : ℕ) :
  num_benches = 29 →
  students_per_bench = 5 →
  ∃ (male_students : ℕ),
    male_students ≥ 29 ∧
    male_students * 5 ≥ num_benches * students_per_bench ∧
    ∀ m : ℕ, m < 29 → m * 5 < num_benches * students_per_bench :=
by sorry

end NUMINAMATH_CALUDE_minimum_male_students_l234_23411


namespace NUMINAMATH_CALUDE_min_product_of_three_min_product_is_neg_720_l234_23458

def S : Finset Int := {-10, -7, -3, 0, 2, 4, 8, 9}

theorem min_product_of_three (a b c : Int) 
  (ha : a ∈ S) (hb : b ∈ S) (hc : c ∈ S) 
  (hab : a ≠ b) (hbc : b ≠ c) (hac : a ≠ c) :
  ∀ x y z : Int, x ∈ S → y ∈ S → z ∈ S → 
  x ≠ y → y ≠ z → x ≠ z → 
  a * b * c ≥ x * y * z :=
by
  sorry

theorem min_product_is_neg_720 : 
  ∃ a b c : Int, a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ 
  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ 
  a * b * c = -720 ∧
  (∀ x y z : Int, x ∈ S → y ∈ S → z ∈ S → 
   x ≠ y → y ≠ z → x ≠ z → 
   x * y * z ≥ -720) :=
by
  sorry

end NUMINAMATH_CALUDE_min_product_of_three_min_product_is_neg_720_l234_23458


namespace NUMINAMATH_CALUDE_min_value_theorem_l234_23405

theorem min_value_theorem (x y z : ℝ) 
  (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (hsum : x + y + z = 9) :
  (x^2 + y^2)/(x + y) + (x^2 + z^2)/(x + z) + (y^2 + z^2)/(y + z) ≥ 9 ∧
  ((x^2 + y^2)/(x + y) + (x^2 + z^2)/(x + z) + (y^2 + z^2)/(y + z) = 9 ↔ x = 3 ∧ y = 3 ∧ z = 3) :=
by sorry

end NUMINAMATH_CALUDE_min_value_theorem_l234_23405


namespace NUMINAMATH_CALUDE_book_arrangement_theorem_l234_23400

/-- The number of ways to arrange books on a shelf -/
def arrange_books (math_books : ℕ) (english_books : ℕ) (science_books : ℕ) : ℕ :=
  (Nat.factorial 3) * (Nat.factorial math_books) * (Nat.factorial english_books) * (Nat.factorial science_books)

/-- Theorem: The number of ways to arrange 4 math books, 6 English books, and 2 science books
    on a shelf, where all books of the same subject must stay together and the books within
    each subject are different, is equal to 207360. -/
theorem book_arrangement_theorem :
  arrange_books 4 6 2 = 207360 := by
  sorry

end NUMINAMATH_CALUDE_book_arrangement_theorem_l234_23400


namespace NUMINAMATH_CALUDE_exists_unreachable_number_l234_23440

/-- A function that returns true if a number is a 4-digit integer -/
def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

/-- A function that counts the number of differing digits between two numbers -/
def digit_difference (n m : ℕ) : ℕ := sorry

/-- The theorem stating that there exists a 4-digit number that cannot be changed 
    into a multiple of 1992 by changing 3 of its digits -/
theorem exists_unreachable_number : 
  ∃ n : ℕ, is_four_digit n ∧ 
    ∀ m : ℕ, is_four_digit m → m % 1992 = 0 → digit_difference n m > 3 := by
  sorry

end NUMINAMATH_CALUDE_exists_unreachable_number_l234_23440


namespace NUMINAMATH_CALUDE_books_per_shelf_l234_23450

/-- Given four shelves with books and a round-trip distance, 
    prove the number of books on each shelf. -/
theorem books_per_shelf 
  (num_shelves : ℕ) 
  (round_trip_distance : ℕ) 
  (h1 : num_shelves = 4)
  (h2 : round_trip_distance = 3200)
  (h3 : ∃ (books_per_shelf : ℕ), 
    num_shelves * books_per_shelf = round_trip_distance / 2) :
  ∃ (books_per_shelf : ℕ), books_per_shelf = 400 :=
by
  sorry

end NUMINAMATH_CALUDE_books_per_shelf_l234_23450


namespace NUMINAMATH_CALUDE_sum_congruence_mod_9_l234_23496

theorem sum_congruence_mod_9 : 
  (2 + 33 + 444 + 5555 + 66666 + 777777 + 8888888 + 99999999) % 9 = 6 := by
  sorry

end NUMINAMATH_CALUDE_sum_congruence_mod_9_l234_23496


namespace NUMINAMATH_CALUDE_tree_height_problem_l234_23436

theorem tree_height_problem (h1 h2 : ℝ) : 
  h1 = h2 + 20 →  -- One tree is 20 feet taller than the other
  h2 / h1 = 5 / 7 →  -- The heights are in the ratio 5:7
  h1 = 70 := by  -- The height of the taller tree is 70 feet
sorry

end NUMINAMATH_CALUDE_tree_height_problem_l234_23436


namespace NUMINAMATH_CALUDE_power_two_equality_l234_23470

theorem power_two_equality (m : ℕ) : 2^m = 2 * 16^2 * 4^3 * 8 → m = 18 := by
  sorry

end NUMINAMATH_CALUDE_power_two_equality_l234_23470


namespace NUMINAMATH_CALUDE_factorization_equality_l234_23461

theorem factorization_equality (x y : ℝ) : 
  y^2 + x*y - 3*x - y - 6 = (y - 3) * (y + 2 + x) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l234_23461


namespace NUMINAMATH_CALUDE_simplify_expression_l234_23491

theorem simplify_expression (x y : ℝ) (hx : x ≠ 0) :
  ((x + y)^2 - y * (2*x + y) - 6*x) / (2*x) = x/2 - 3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l234_23491


namespace NUMINAMATH_CALUDE_perfect_square_prob_l234_23465

/-- A function that represents the number of ways to roll a 10-sided die n times
    such that the product of the rolls is a perfect square -/
def b : ℕ → ℕ
  | 0 => 1
  | n + 1 => 10^n + 2 * b n

/-- The probability of rolling a 10-sided die 4 times and getting a product
    that is a perfect square -/
def prob_perfect_square : ℚ :=
  b 4 / 10^4

theorem perfect_square_prob :
  prob_perfect_square = 316 / 2500 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_prob_l234_23465


namespace NUMINAMATH_CALUDE_counterexamples_exist_l234_23403

def is_counterexample (n : ℕ) : Prop :=
  ¬(Nat.Prime n) ∧ ¬(Nat.Prime (n - 3))

theorem counterexamples_exist : 
  is_counterexample 18 ∧ is_counterexample 24 :=
by sorry

end NUMINAMATH_CALUDE_counterexamples_exist_l234_23403


namespace NUMINAMATH_CALUDE_day_1500_is_sunday_l234_23486

/-- Given that the first day is a Friday, prove that the 1500th day is a Sunday -/
theorem day_1500_is_sunday (first_day : Nat) (h : first_day % 7 = 5) : 
  (first_day + 1499) % 7 = 0 := by
  sorry

#check day_1500_is_sunday

end NUMINAMATH_CALUDE_day_1500_is_sunday_l234_23486


namespace NUMINAMATH_CALUDE_square_diagonal_side_area_l234_23407

/-- Given a square with diagonal length 4, prove its side length and area. -/
theorem square_diagonal_side_area :
  ∃ (side_length area : ℝ),
    4^2 = 2 * side_length^2 ∧
    side_length = 2 * Real.sqrt 2 ∧
    area = 8 := by
  sorry

end NUMINAMATH_CALUDE_square_diagonal_side_area_l234_23407


namespace NUMINAMATH_CALUDE_cube_side_length_l234_23494

-- Define the constants
def paint_cost_per_kg : ℝ := 60
def area_covered_per_kg : ℝ := 20
def total_paint_cost : ℝ := 1800
def num_cube_sides : ℕ := 6

-- Define the theorem
theorem cube_side_length :
  ∃ (side_length : ℝ),
    side_length > 0 ∧
    side_length^2 * num_cube_sides * paint_cost_per_kg / area_covered_per_kg = total_paint_cost ∧
    side_length = 10 := by
  sorry

end NUMINAMATH_CALUDE_cube_side_length_l234_23494


namespace NUMINAMATH_CALUDE_perpendicular_lines_from_perpendicular_planes_l234_23433

-- Define the space
variable (Space : Type)

-- Define lines and planes
variable (Line Plane : Type)

-- Define the relationships between lines and planes
variable (skew : Line → Line → Prop)
variable (parallel : Line → Line → Prop)
variable (parallel_line_plane : Line → Plane → Prop)
variable (parallel_plane : Plane → Plane → Prop)
variable (perpendicular_line_plane : Line → Plane → Prop)
variable (perpendicular_plane : Plane → Plane → Prop)
variable (perpendicular_line : Line → Line → Prop)

-- Define the lines and planes
variable (l m n : Line)
variable (α β : Plane)

-- State that l, m, n are different lines
variable (h_diff_lm : l ≠ m)
variable (h_diff_ln : l ≠ n)
variable (h_diff_mn : m ≠ n)

-- State that α and β are non-coincident planes
variable (h_non_coincident : α ≠ β)

-- State the theorem to be proved
theorem perpendicular_lines_from_perpendicular_planes :
  (perpendicular_plane α β ∧ perpendicular_line_plane l α ∧ perpendicular_line_plane m β) →
  perpendicular_line l m :=
sorry

end NUMINAMATH_CALUDE_perpendicular_lines_from_perpendicular_planes_l234_23433


namespace NUMINAMATH_CALUDE_divisors_18_product_and_sum_l234_23487

def divisors (n : ℕ) : List ℕ :=
  (List.range (n + 1)).filter (λ d => n % d = 0)

theorem divisors_18_product_and_sum :
  (divisors 18).prod = 5832 ∧ (divisors 18).sum = 39 := by
  sorry

end NUMINAMATH_CALUDE_divisors_18_product_and_sum_l234_23487


namespace NUMINAMATH_CALUDE_color_tv_price_l234_23406

theorem color_tv_price (x : ℝ) : 
  (1 + 0.4) * x * 0.8 - x = 144 → x = 1200 := by
  sorry

end NUMINAMATH_CALUDE_color_tv_price_l234_23406


namespace NUMINAMATH_CALUDE_smallest_difference_for_8_factorial_l234_23416

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

theorem smallest_difference_for_8_factorial :
  ∀ a b c : ℕ+,
  a * b * c = factorial 8 →
  a < b →
  b < c →
  ∀ a' b' c' : ℕ+,
  a' * b' * c' = factorial 8 →
  a' < b' →
  b' < c' →
  c - a ≤ c' - a' :=
sorry

end NUMINAMATH_CALUDE_smallest_difference_for_8_factorial_l234_23416


namespace NUMINAMATH_CALUDE_abc_equation_l234_23467

theorem abc_equation (a b c p : ℝ) 
  (h_distinct : a ≠ b ∧ b ≠ c ∧ a ≠ c)
  (h_nonzero : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0)
  (h_eq1 : a + 2/b = p)
  (h_eq2 : b + 2/c = p)
  (h_eq3 : c + 2/a = p) :
  a * b * c + 2 * p = 0 := by
sorry

end NUMINAMATH_CALUDE_abc_equation_l234_23467


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l234_23424

theorem imaginary_part_of_complex_fraction : 
  let z : ℂ := 2 * Complex.I / (3 - 2 * Complex.I)
  Complex.im z = 6 / 13 := by
sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l234_23424


namespace NUMINAMATH_CALUDE_mersenne_last_two_digits_l234_23437

/-- The exponent used in the Mersenne prime -/
def p : ℕ := 82589933

/-- The Mersenne number -/
def mersenne_number : ℕ := 2^p - 1

/-- The last two digits of a number -/
def last_two_digits (n : ℕ) : ℕ := n % 100

theorem mersenne_last_two_digits : last_two_digits mersenne_number = 91 := by
  sorry

end NUMINAMATH_CALUDE_mersenne_last_two_digits_l234_23437


namespace NUMINAMATH_CALUDE_impossibleToMakeAllMultiplesOfTen_l234_23483

/-- Represents an 8x8 grid of integers -/
def Grid := Fin 8 → Fin 8 → ℤ

/-- Represents an operation on the grid -/
inductive Operation
| threeByThree (i j : Fin 8) : Operation
| fourByFour (i j : Fin 8) : Operation

/-- Apply an operation to a grid -/
def applyOperation (g : Grid) (op : Operation) : Grid :=
  sorry

/-- Check if all numbers in the grid are multiples of 10 -/
def allMultiplesOfTen (g : Grid) : Prop :=
  ∀ i j, ∃ k, g i j = 10 * k

/-- The main theorem -/
theorem impossibleToMakeAllMultiplesOfTen :
  ∃ (g : Grid),
    (∀ i j, g i j ≥ 0) ∧
    ¬∃ (ops : List Operation), allMultiplesOfTen (ops.foldl applyOperation g) :=
  sorry

end NUMINAMATH_CALUDE_impossibleToMakeAllMultiplesOfTen_l234_23483


namespace NUMINAMATH_CALUDE_sector_angle_l234_23408

/-- Given a circular sector with arc length 4 and area 4, 
    prove that the absolute value of its central angle in radians is 2. -/
theorem sector_angle (r : ℝ) (θ : ℝ) (h1 : r * θ = 4) (h2 : (1/2) * r^2 * θ = 4) : 
  |θ| = 2 := by
sorry

end NUMINAMATH_CALUDE_sector_angle_l234_23408


namespace NUMINAMATH_CALUDE_sugar_recipe_reduction_l234_23426

theorem sugar_recipe_reduction : 
  let full_recipe : ℚ := 5 + 3/4
  let reduced_recipe : ℚ := full_recipe / 3
  reduced_recipe = 1 + 11/12 := by sorry

end NUMINAMATH_CALUDE_sugar_recipe_reduction_l234_23426


namespace NUMINAMATH_CALUDE_second_customer_regular_hours_l234_23453

/-- Represents the hourly rates and customer data for an online service -/
structure OnlineService where
  regularRate : ℝ
  premiumRate : ℝ
  customer1PremiumHours : ℝ
  customer1RegularHours : ℝ
  customer1TotalCharge : ℝ
  customer2PremiumHours : ℝ
  customer2TotalCharge : ℝ

/-- Calculates the number of regular hours for the second customer -/
def calculateCustomer2RegularHours (service : OnlineService) : ℝ :=
  -- Implementation not required for the statement
  sorry

/-- Theorem stating that the second customer spent 3 regular hours -/
theorem second_customer_regular_hours (service : OnlineService) 
  (h1 : service.customer1PremiumHours = 2)
  (h2 : service.customer1RegularHours = 9)
  (h3 : service.customer1TotalCharge = 28)
  (h4 : service.customer2PremiumHours = 3)
  (h5 : service.customer2TotalCharge = 27) :
  calculateCustomer2RegularHours service = 3 := by
  sorry

#eval "Lean 4 statement generated successfully."

end NUMINAMATH_CALUDE_second_customer_regular_hours_l234_23453


namespace NUMINAMATH_CALUDE_ralph_peanuts_l234_23471

-- Define the initial number of peanuts
def initial_peanuts : ℕ := 74

-- Define the number of peanuts lost
def peanuts_lost : ℕ := 59

-- Theorem to prove
theorem ralph_peanuts : initial_peanuts - peanuts_lost = 15 := by
  sorry

end NUMINAMATH_CALUDE_ralph_peanuts_l234_23471


namespace NUMINAMATH_CALUDE_rectangle_area_comparison_l234_23439

theorem rectangle_area_comparison (a b : ℝ) (ha : a = 8) (hb : b = 15) : 
  let d := Real.sqrt (a^2 + b^2)
  let new_rectangle_area := (d + b) * (d - b)
  let square_area := (a + b)^2
  new_rectangle_area ≠ square_area := by
sorry

end NUMINAMATH_CALUDE_rectangle_area_comparison_l234_23439


namespace NUMINAMATH_CALUDE_sqrt_expression_equality_l234_23488

theorem sqrt_expression_equality : 
  Real.sqrt 8 - (1/3)⁻¹ / Real.sqrt 3 + (1 - Real.sqrt 2)^2 = 3 - Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_expression_equality_l234_23488


namespace NUMINAMATH_CALUDE_smallest_divisible_by_1_to_10_l234_23452

theorem smallest_divisible_by_1_to_10 : ∃ (n : ℕ), n > 0 ∧ (∀ k : ℕ, 1 ≤ k ∧ k ≤ 10 → k ∣ n) ∧ (∀ m : ℕ, m > 0 ∧ (∀ k : ℕ, 1 ≤ k ∧ k ≤ 10 → k ∣ m) → n ≤ m) ∧ n = 2520 := by
  sorry

end NUMINAMATH_CALUDE_smallest_divisible_by_1_to_10_l234_23452


namespace NUMINAMATH_CALUDE_inequality_proof_l234_23414

theorem inequality_proof (a b : ℝ) (ha : a > 1) (hb : b > 1) :
  a^2 / (b - 1) + b^2 / (a - 1) ≥ 8 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l234_23414


namespace NUMINAMATH_CALUDE_min_red_chips_is_72_l234_23428

/-- Represents the number of chips of each color in the box -/
structure ChipCounts where
  white : ℕ
  blue : ℕ
  red : ℕ

/-- Checks if the chip counts satisfy the given conditions -/
def valid_counts (c : ChipCounts) : Prop :=
  c.blue ≥ c.white / 3 ∧
  c.blue ≤ c.red / 4 ∧
  c.white + c.blue ≥ 72

/-- The minimum number of red chips required -/
def min_red_chips : ℕ := 72

/-- Theorem stating that the minimum number of red chips is 72 -/
theorem min_red_chips_is_72 :
  ∀ c : ChipCounts, valid_counts c → c.red ≥ min_red_chips :=
by sorry

end NUMINAMATH_CALUDE_min_red_chips_is_72_l234_23428


namespace NUMINAMATH_CALUDE_weight_lifting_equivalence_l234_23473

/-- Given that Max originally lifts two 30-pound weights 10 times, this theorem proves
    that he needs to lift two 25-pound weights 12 times to match the original total weight. -/
theorem weight_lifting_equivalence :
  let original_weight : ℕ := 30
  let original_reps : ℕ := 10
  let new_weight : ℕ := 25
  let total_weight : ℕ := 2 * original_weight * original_reps
  ∃ (n : ℕ), 2 * new_weight * n = total_weight ∧ n = 12 :=
by sorry

end NUMINAMATH_CALUDE_weight_lifting_equivalence_l234_23473


namespace NUMINAMATH_CALUDE_quadratic_real_roots_condition_l234_23469

theorem quadratic_real_roots_condition (m : ℝ) :
  (∃ x : ℝ, x^2 - 3*x + m = 0) → m ≤ 9/4 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_real_roots_condition_l234_23469


namespace NUMINAMATH_CALUDE_otimes_inequality_solutions_l234_23421

-- Define the ⊗ operation
def otimes (a b : ℝ) : ℝ := a * (a - b) + 1

-- Define the set of non-negative integers satisfying the inequality
def solution_set : Set ℕ := {x | otimes 2 ↑x ≥ 3}

-- Theorem statement
theorem otimes_inequality_solutions :
  solution_set = {0, 1} := by sorry

end NUMINAMATH_CALUDE_otimes_inequality_solutions_l234_23421


namespace NUMINAMATH_CALUDE_path_area_l234_23466

/-- The area of a ring-shaped path around a circular lawn -/
theorem path_area (r : ℝ) (w : ℝ) (h1 : r = 35) (h2 : w = 7) :
  (π * (r + w)^2 - π * r^2) = 539 * π :=
sorry

end NUMINAMATH_CALUDE_path_area_l234_23466


namespace NUMINAMATH_CALUDE_concentric_circles_chords_l234_23457

/-- Given two concentric circles with chords of the larger circle tangent to the smaller circle,
    if the angle between two adjacent chords is 60°, then the minimum number of such chords
    needed to complete a full circle is 3. -/
theorem concentric_circles_chords (angle_between_chords : ℝ) (n : ℕ) :
  angle_between_chords = 60 →
  (n : ℝ) * (180 - angle_between_chords) = 360 →
  n = 3 :=
sorry

end NUMINAMATH_CALUDE_concentric_circles_chords_l234_23457


namespace NUMINAMATH_CALUDE_fibonacci_mod_4_2022_l234_23476

def fibonacci : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | (n + 2) => fibonacci (n + 1) + fibonacci n

def b (n : ℕ) : ℕ := fibonacci n % 4

theorem fibonacci_mod_4_2022 : b 2022 = 0 := by
  sorry

end NUMINAMATH_CALUDE_fibonacci_mod_4_2022_l234_23476


namespace NUMINAMATH_CALUDE_tshirt_cost_l234_23420

def amusement_park_problem (initial_amount ticket_cost food_cost remaining_amount : ℕ) : Prop :=
  let total_spent := ticket_cost + food_cost + (initial_amount - ticket_cost - food_cost - remaining_amount)
  total_spent = initial_amount - remaining_amount

theorem tshirt_cost (initial_amount ticket_cost food_cost remaining_amount : ℕ) 
  (h1 : initial_amount = 75)
  (h2 : ticket_cost = 30)
  (h3 : food_cost = 13)
  (h4 : remaining_amount = 9)
  (h5 : amusement_park_problem initial_amount ticket_cost food_cost remaining_amount) :
  initial_amount - ticket_cost - food_cost - remaining_amount = 23 := by
  sorry

end NUMINAMATH_CALUDE_tshirt_cost_l234_23420


namespace NUMINAMATH_CALUDE_laser_path_distance_correct_l234_23413

/-- The total distance traveled by a laser beam with specified bounces -/
def laser_path_distance : ℝ := 12

/-- Starting point of the laser -/
def start_point : ℝ × ℝ := (4, 6)

/-- Final point of the laser -/
def end_point : ℝ × ℝ := (8, 6)

/-- Theorem stating that the laser path distance is correct -/
theorem laser_path_distance_correct :
  let path := laser_path_distance
  let start := start_point
  let end_ := end_point
  (path = ‖(start.1 + end_.1, start.2 - end_.2)‖) ∧
  (path > 0) ∧
  (start.1 > 0) ∧
  (start.2 > 0) ∧
  (end_.1 > 0) ∧
  (end_.2 > 0) :=
by sorry

end NUMINAMATH_CALUDE_laser_path_distance_correct_l234_23413


namespace NUMINAMATH_CALUDE_four_digit_numbers_extrema_l234_23459

theorem four_digit_numbers_extrema :
  let sum_of_numbers : ℕ := 106656
  let is_valid_number : (ℕ → Bool) :=
    λ n => n ≥ 1000 ∧ n ≤ 9999 ∧ 
           (let digits := [n / 1000, (n / 100) % 10, (n / 10) % 10, n % 10]
            digits.all (· ≠ 0) ∧ digits.Nodup)
  let valid_numbers := (List.range 10000).filter is_valid_number
  sum_of_numbers = valid_numbers.sum →
  (∀ n ∈ valid_numbers, n ≤ 9421 ∧ n ≥ 1249) ∧
  9421 ∈ valid_numbers ∧ 1249 ∈ valid_numbers :=
by sorry

end NUMINAMATH_CALUDE_four_digit_numbers_extrema_l234_23459


namespace NUMINAMATH_CALUDE_lemonade_stand_profit_l234_23415

/-- Represents the profit calculation for a lemonade stand -/
theorem lemonade_stand_profit :
  ∀ (lemon_cost sugar_cost cup_cost : ℕ) 
    (price_per_cup cups_sold : ℕ),
  lemon_cost = 10 →
  sugar_cost = 5 →
  cup_cost = 3 →
  price_per_cup = 4 →
  cups_sold = 21 →
  cups_sold * price_per_cup - (lemon_cost + sugar_cost + cup_cost) = 66 := by
sorry

end NUMINAMATH_CALUDE_lemonade_stand_profit_l234_23415


namespace NUMINAMATH_CALUDE_fruit_weights_correct_l234_23422

structure Fruit where
  name : String
  weight : Nat

def banana : Fruit := ⟨"banana", 170⟩
def orange : Fruit := ⟨"orange", 180⟩
def watermelon : Fruit := ⟨"watermelon", 1400⟩
def kiwi : Fruit := ⟨"kiwi", 200⟩
def apple : Fruit := ⟨"apple", 210⟩

def fruits : List Fruit := [banana, orange, watermelon, kiwi, apple]

theorem fruit_weights_correct : 
  (∀ f ∈ fruits, f.weight ∈ [170, 180, 200, 210, 1400]) ∧ 
  (watermelon.weight > banana.weight + orange.weight + kiwi.weight + apple.weight) ∧
  (orange.weight + kiwi.weight = banana.weight + apple.weight) ∧
  (orange.weight > banana.weight) ∧
  (orange.weight < kiwi.weight) := by
  sorry

end NUMINAMATH_CALUDE_fruit_weights_correct_l234_23422


namespace NUMINAMATH_CALUDE_problem_solution_l234_23475

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - a|

-- Define the theorem
theorem problem_solution :
  -- Given conditions
  (∀ x : ℝ, (f 2 x ≤ 3) ↔ (-1 ≤ x ∧ x ≤ 5)) →
  -- Part 1: Prove that a = 2
  (∃! a : ℝ, ∀ x : ℝ, (f a x ≤ 3) ↔ (-1 ≤ x ∧ x ≤ 5)) ∧
  -- Part 2: Prove that the minimum value of f(x) + f(x+5) is 5
  (∀ x : ℝ, f 2 x + f 2 (x + 5) ≥ 5) ∧
  (∃ x : ℝ, f 2 x + f 2 (x + 5) = 5) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l234_23475


namespace NUMINAMATH_CALUDE_find_m_value_l234_23495

theorem find_m_value (m : ℤ) : 
  (∃ (a b : ℝ), ∀ (x : ℝ), (2/3 * (m + 4) * x^(|m| - 3) + 6 > 0) ↔ (a * x + b > 0)) →
  (m + 4 ≠ 0) →
  m = 4 := by
sorry

end NUMINAMATH_CALUDE_find_m_value_l234_23495


namespace NUMINAMATH_CALUDE_total_area_after_expansion_l234_23484

/-- Theorem: Total area of two houses after expansion -/
theorem total_area_after_expansion (small_house large_house expansion : ℕ) 
  (h1 : small_house = 5200)
  (h2 : large_house = 7300)
  (h3 : expansion = 3500) :
  small_house + large_house + expansion = 16000 := by
  sorry

#check total_area_after_expansion

end NUMINAMATH_CALUDE_total_area_after_expansion_l234_23484


namespace NUMINAMATH_CALUDE_clothes_washing_time_l234_23434

/-- Represents the time in minutes for washing different types of laundry -/
structure LaundryTime where
  clothes : ℕ
  towels : ℕ
  sheets : ℕ

/-- Defines the conditions for the laundry washing problem -/
def valid_laundry_time (t : LaundryTime) : Prop :=
  t.towels = 2 * t.clothes ∧
  t.sheets = t.towels - 15 ∧
  t.clothes + t.towels + t.sheets = 135

/-- Theorem stating that the time to wash clothes is 30 minutes -/
theorem clothes_washing_time (t : LaundryTime) :
  valid_laundry_time t → t.clothes = 30 := by
  sorry

end NUMINAMATH_CALUDE_clothes_washing_time_l234_23434


namespace NUMINAMATH_CALUDE_complex_expression_eighth_root_of_unity_l234_23402

theorem complex_expression_eighth_root_of_unity :
  let z := (Complex.tan (Real.pi / 4) + Complex.I) / (Complex.tan (Real.pi / 4) - Complex.I)
  z = Complex.I ∧
  z^8 = 1 ∧
  ∃ n : ℕ, n = 2 ∧ z = Complex.exp (Complex.I * (2 * ↑n * Real.pi / 8)) := by
  sorry

end NUMINAMATH_CALUDE_complex_expression_eighth_root_of_unity_l234_23402


namespace NUMINAMATH_CALUDE_xyz_value_l234_23479

theorem xyz_value (x y z : ℝ) 
  (h1 : (x + y + z) * (x * y + x * z + y * z) = 24)
  (h2 : x^2 * (y + z) + y^2 * (x + z) + z^2 * (x + y) = 9)
  (h3 : x + y + z = 3) :
  x * y * z = 5 := by
  sorry

end NUMINAMATH_CALUDE_xyz_value_l234_23479


namespace NUMINAMATH_CALUDE_equilateral_triangle_inscribed_circle_radius_l234_23431

/-- Given an equilateral triangle inscribed in a circle with area 81 cm²,
    prove that the radius of the circle is 6 * (3^(1/4)) cm. -/
theorem equilateral_triangle_inscribed_circle_radius 
  (S : ℝ) (r : ℝ) (h1 : S = 81) :
  r = 6 * (3 : ℝ)^(1/4) :=
by
  sorry


end NUMINAMATH_CALUDE_equilateral_triangle_inscribed_circle_radius_l234_23431


namespace NUMINAMATH_CALUDE_amoeba_bacteria_ratio_l234_23493

-- Define the initial number of amoebas and bacteria
def initial_amoeba : ℕ := sorry
def initial_bacteria : ℕ := sorry

-- Define the number of days
def days : ℕ := 100

-- Define the function for the number of amoebas on day n
def amoeba (n : ℕ) : ℕ := 2^(n-1) * initial_amoeba

-- Define the function for the number of bacteria on day n after predation
def bacteria_after_predation (n : ℕ) : ℕ := 2^(n-1) * (initial_bacteria - initial_amoeba)

theorem amoeba_bacteria_ratio :
  bacteria_after_predation days = 0 → initial_amoeba = initial_bacteria := by sorry

end NUMINAMATH_CALUDE_amoeba_bacteria_ratio_l234_23493


namespace NUMINAMATH_CALUDE_expression_value_l234_23490

theorem expression_value (a b : ℝ) (h : a^2 + 2*a*b + b^2 = 0) :
  a*(a + 4*b) - (a + 2*b)*(a - 2*b) = 0 := by
sorry

end NUMINAMATH_CALUDE_expression_value_l234_23490


namespace NUMINAMATH_CALUDE_symmetric_point_coordinates_l234_23444

structure Point where
  x : ℝ
  y : ℝ

def translate_left (p : Point) (d : ℝ) : Point :=
  ⟨p.x - d, p.y⟩

def symmetric_origin (p : Point) : Point :=
  ⟨-p.x, -p.y⟩

theorem symmetric_point_coordinates :
  let A : Point := ⟨1, 2⟩
  let B : Point := translate_left A 2
  let C : Point := symmetric_origin B
  C = ⟨1, -2⟩ := by sorry

end NUMINAMATH_CALUDE_symmetric_point_coordinates_l234_23444


namespace NUMINAMATH_CALUDE_smallest_a_is_9_l234_23442

-- Define the arithmetic sequence
def is_arithmetic_sequence (a b c : ℕ) : Prop := b - a = c - b

-- Define the function f
def f (a b c : ℕ) (x : ℝ) : ℝ := a * x^2 + b * x + c

-- State the theorem
theorem smallest_a_is_9 
  (a b c : ℕ) 
  (r s : ℝ) 
  (h_arith : is_arithmetic_sequence a b c)
  (h_positive : 0 < a ∧ 0 < b ∧ 0 < c)
  (h_order : a < b ∧ b < c)
  (h_f_r : f a b c r = s)
  (h_f_s : f a b c s = r)
  (h_rs : r * s = 2017)
  (h_distinct : r ≠ s) :
  ∀ a' : ℕ, (∃ b' c' : ℕ, 
    is_arithmetic_sequence a' b' c' ∧ 
    0 < a' ∧ 0 < b' ∧ 0 < c' ∧
    a' < b' ∧ b' < c' ∧
    (∃ r' s' : ℝ, f a' b' c' r' = s' ∧ f a' b' c' s' = r' ∧ r' * s' = 2017 ∧ r' ≠ s')) →
  a' ≥ 9 :=
sorry

end NUMINAMATH_CALUDE_smallest_a_is_9_l234_23442


namespace NUMINAMATH_CALUDE_parallel_vectors_magnitude_l234_23445

/-- Given two vectors a and b in ℝ², if a is parallel to b, then the magnitude of b is 2√5. -/
theorem parallel_vectors_magnitude (a b : ℝ × ℝ) : 
  a = (1, 2) → 
  b.1 = -2 → 
  ∃ (t : ℝ), a = t • b → 
  ‖b‖ = 2 * Real.sqrt 5 := by
sorry

end NUMINAMATH_CALUDE_parallel_vectors_magnitude_l234_23445


namespace NUMINAMATH_CALUDE_deck_width_proof_l234_23451

/-- Proves that for a rectangular pool of 20 feet by 22 feet, surrounded by a deck of uniform width,
    if the total area of the pool and deck is 728 square feet, then the width of the deck is 3 feet. -/
theorem deck_width_proof (w : ℝ) : 
  (20 + 2*w) * (22 + 2*w) = 728 → w = 3 :=
by sorry

end NUMINAMATH_CALUDE_deck_width_proof_l234_23451


namespace NUMINAMATH_CALUDE_nine_sided_polygon_diagonals_l234_23429

/-- The number of diagonals in a polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- Theorem: A nine-sided polygon has 27 diagonals -/
theorem nine_sided_polygon_diagonals :
  num_diagonals 9 = 27 := by sorry

end NUMINAMATH_CALUDE_nine_sided_polygon_diagonals_l234_23429


namespace NUMINAMATH_CALUDE_proportional_enlargement_l234_23404

/-- Given a rectangle that is enlarged proportionally, this theorem proves
    that the new height can be calculated from the original dimensions
    and the new width. -/
theorem proportional_enlargement
  (original_width original_height new_width : ℝ)
  (h_positive : original_width > 0 ∧ original_height > 0 ∧ new_width > 0)
  (h_original_width : original_width = 2)
  (h_original_height : original_height = 1.5)
  (h_new_width : new_width = 8) :
  let new_height := original_height * (new_width / original_width)
  new_height = 6 := by
sorry

end NUMINAMATH_CALUDE_proportional_enlargement_l234_23404


namespace NUMINAMATH_CALUDE_dollar_op_neg_three_neg_four_l234_23498

def dollar_op (x y : Int) : Int := x * (y + 1) + x * y

theorem dollar_op_neg_three_neg_four : dollar_op (-3) (-4) = 21 := by
  sorry

end NUMINAMATH_CALUDE_dollar_op_neg_three_neg_four_l234_23498


namespace NUMINAMATH_CALUDE_parallel_tangents_sum_bound_l234_23441

noncomputable def f (k : ℝ) (x : ℝ) : ℝ := (k + 4/k) * Real.log x + (4 - x^2) / x

theorem parallel_tangents_sum_bound (k : ℝ) (x₁ x₂ : ℝ) (h_k : k ≥ 4) 
  (h_distinct : x₁ ≠ x₂) (h_positive : x₁ > 0 ∧ x₂ > 0) 
  (h_parallel : (deriv (f k)) x₁ = (deriv (f k)) x₂) :
  x₁ + x₂ > 16/5 := by
  sorry

end NUMINAMATH_CALUDE_parallel_tangents_sum_bound_l234_23441


namespace NUMINAMATH_CALUDE_exam_results_l234_23443

theorem exam_results (total_students : ℕ) 
  (percent_8_or_more : ℚ) (percent_5_or_less : ℚ) :
  total_students = 40 →
  percent_8_or_more = 20 / 100 →
  percent_5_or_less = 45 / 100 →
  (1 : ℚ) - percent_8_or_more - percent_5_or_less = 35 / 100 := by
  sorry

end NUMINAMATH_CALUDE_exam_results_l234_23443


namespace NUMINAMATH_CALUDE_least_exponent_sum_for_2023_l234_23430

def is_sum_of_distinct_powers_of_two (n : ℕ) (exponents : List ℕ) : Prop :=
  n = (exponents.map (fun e => 2^e)).sum ∧ exponents.Nodup

theorem least_exponent_sum_for_2023 :
  ∃ (exponents : List ℕ),
    is_sum_of_distinct_powers_of_two 2023 exponents ∧
    exponents.sum = 48 ∧
    ∀ (other_exponents : List ℕ),
      is_sum_of_distinct_powers_of_two 2023 other_exponents →
      other_exponents.sum ≥ 48 :=
sorry

end NUMINAMATH_CALUDE_least_exponent_sum_for_2023_l234_23430


namespace NUMINAMATH_CALUDE_problem_solution_l234_23454

theorem problem_solution : (42 / (9 - 2 + 3)) * 7 = 29.4 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l234_23454
