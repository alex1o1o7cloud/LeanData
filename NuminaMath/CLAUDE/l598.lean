import Mathlib

namespace NUMINAMATH_CALUDE_solution_set_f_plus_x_positive_range_of_a_for_full_solution_set_l598_59832

def f (x : ℝ) := |x - 2| - |x + 1|

theorem solution_set_f_plus_x_positive :
  {x : ℝ | f x + x > 0} = Set.union (Set.union (Set.Ioo (-3) (-1)) (Set.Ico (-1) 1)) (Set.Ioi 3) :=
sorry

theorem range_of_a_for_full_solution_set :
  {a : ℝ | ∀ x, f x ≤ a^2 - 2*a} = Set.union (Set.Iic (-1)) (Set.Ici 3) :=
sorry

end NUMINAMATH_CALUDE_solution_set_f_plus_x_positive_range_of_a_for_full_solution_set_l598_59832


namespace NUMINAMATH_CALUDE_smallest_sum_c_d_l598_59866

theorem smallest_sum_c_d (c d : ℝ) : 
  c > 0 → d > 0 → 
  (∃ x : ℝ, x^2 + c*x + 3*d = 0) → 
  (∃ x : ℝ, x^2 + 3*d*x + c = 0) → 
  c + d ≥ 16/3 ∧ 
  ∃ c₀ d₀ : ℝ, c₀ > 0 ∧ d₀ > 0 ∧ 
    (∃ x : ℝ, x^2 + c₀*x + 3*d₀ = 0) ∧ 
    (∃ x : ℝ, x^2 + 3*d₀*x + c₀ = 0) ∧ 
    c₀ + d₀ = 16/3 :=
by sorry

end NUMINAMATH_CALUDE_smallest_sum_c_d_l598_59866


namespace NUMINAMATH_CALUDE_circle_equation_l598_59856

-- Define the circle C
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the lines
def line1 (x y : ℝ) : Prop := x - y = 0
def line2 (x y : ℝ) : Prop := x - y - 4 = 0
def line3 (x y : ℝ) : Prop := x + y = 0

-- Define tangency
def isTangent (c : Circle) (l : (ℝ → ℝ → Prop)) : Prop :=
  ∃ (x y : ℝ), l x y ∧ ((x - c.center.1)^2 + (y - c.center.2)^2 = c.radius^2)

-- Main theorem
theorem circle_equation (C : Circle) 
  (h1 : isTangent C line1)
  (h2 : isTangent C line2)
  (h3 : line3 C.center.1 C.center.2) :
  ∀ (x y : ℝ), (x - 1)^2 + (y + 1)^2 = 2 ↔ ((x - C.center.1)^2 + (y - C.center.2)^2 = C.radius^2) :=
sorry

end NUMINAMATH_CALUDE_circle_equation_l598_59856


namespace NUMINAMATH_CALUDE_not_necessarily_divisible_by_48_l598_59805

theorem not_necessarily_divisible_by_48 (k : ℤ) :
  let n := k * (k + 1) * (k + 2) * (k + 3)
  ∃ (n : ℤ), (8 ∣ n) ∧ ¬(48 ∣ n) := by
  sorry

end NUMINAMATH_CALUDE_not_necessarily_divisible_by_48_l598_59805


namespace NUMINAMATH_CALUDE_charlie_feathers_l598_59860

theorem charlie_feathers (total_needed : ℕ) (still_needed : ℕ) 
  (h1 : total_needed = 900)
  (h2 : still_needed = 513) :
  total_needed - still_needed = 387 := by
  sorry

end NUMINAMATH_CALUDE_charlie_feathers_l598_59860


namespace NUMINAMATH_CALUDE_mango_problem_l598_59854

theorem mango_problem (alexis_mangoes : ℕ) (dilan_ashley_mangoes : ℕ) : 
  alexis_mangoes = 60 → 
  alexis_mangoes = 4 * dilan_ashley_mangoes → 
  alexis_mangoes + dilan_ashley_mangoes = 75 := by
sorry

end NUMINAMATH_CALUDE_mango_problem_l598_59854


namespace NUMINAMATH_CALUDE_pizza_leftover_slices_pizza_leftover_slices_proof_l598_59870

/-- Given two pizzas, each cut into 12 slices, with Dean eating half of one pizza,
    Frank eating 3 slices of the same pizza, and Sammy eating a third of the other pizza,
    prove that the total number of slices left over is 11. -/
theorem pizza_leftover_slices : ℕ :=
  let total_pizzas : ℕ := 2
  let slices_per_pizza : ℕ := 12
  let dean_eaten : ℕ := slices_per_pizza / 2
  let frank_eaten : ℕ := 3
  let sammy_eaten : ℕ := slices_per_pizza / 3
  let total_slices : ℕ := total_pizzas * slices_per_pizza
  let hawaiian_leftover : ℕ := slices_per_pizza - (dean_eaten + frank_eaten)
  let cheese_leftover : ℕ := slices_per_pizza - sammy_eaten
  let total_leftover : ℕ := hawaiian_leftover + cheese_leftover
  11

theorem pizza_leftover_slices_proof : pizza_leftover_slices = 11 := by sorry

end NUMINAMATH_CALUDE_pizza_leftover_slices_pizza_leftover_slices_proof_l598_59870


namespace NUMINAMATH_CALUDE_largest_digit_divisible_by_6_l598_59848

def is_divisible_by_6 (n : ℕ) : Prop := n % 6 = 0

def is_single_digit (n : ℕ) : Prop := n < 10

theorem largest_digit_divisible_by_6 :
  ∀ M : ℕ, is_single_digit M →
    (is_divisible_by_6 (45670 + M) → M ≤ 8) ∧
    (is_divisible_by_6 (45678)) := by
  sorry

end NUMINAMATH_CALUDE_largest_digit_divisible_by_6_l598_59848


namespace NUMINAMATH_CALUDE_probability_at_least_one_correct_l598_59859

theorem probability_at_least_one_correct (total_questions : Nat) (options_per_question : Nat) (guessed_questions : Nat) : 
  total_questions = 30 → 
  options_per_question = 6 → 
  guessed_questions = 5 → 
  (1 - (options_per_question - 1 : ℚ) / options_per_question ^ guessed_questions) = 4651 / 7776 := by
sorry

end NUMINAMATH_CALUDE_probability_at_least_one_correct_l598_59859


namespace NUMINAMATH_CALUDE_base_b_square_l598_59857

theorem base_b_square (b : ℕ) (h : b > 1) :
  (3 * b + 5) ^ 2 = b ^ 3 + 3 * b ^ 2 + 3 * b + 1 → b = 7 :=
by sorry

end NUMINAMATH_CALUDE_base_b_square_l598_59857


namespace NUMINAMATH_CALUDE_hundredth_digit_of_seven_twenty_sixths_l598_59885

theorem hundredth_digit_of_seven_twenty_sixths (n : ℕ) : n = 100 → 
  (7 : ℚ) / 26 * 10^n % 10 = 9 := by sorry

end NUMINAMATH_CALUDE_hundredth_digit_of_seven_twenty_sixths_l598_59885


namespace NUMINAMATH_CALUDE_geometric_sequence_general_term_l598_59882

/-- Geometric sequence with common ratio 4 and sum of first three terms equal to 21 -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  (∀ n, a (n + 1) = 4 * a n) ∧ (a 1 + a 2 + a 3 = 21)

/-- The general term of the geometric sequence -/
def general_term (n : ℕ) : ℝ := 4^(n - 1)

theorem geometric_sequence_general_term (a : ℕ → ℝ) :
  geometric_sequence a → ∀ n : ℕ, a n = general_term n := by sorry

end NUMINAMATH_CALUDE_geometric_sequence_general_term_l598_59882


namespace NUMINAMATH_CALUDE_hyperbola_and_line_properties_l598_59841

/-- Hyperbola with given properties -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_pos_a : a > 0
  h_pos_b : b > 0
  h_focus : 2 = Real.sqrt (a^2 + b^2)
  h_eccentricity : 2 = Real.sqrt (a^2 + b^2) / a

/-- Line intersecting the hyperbola -/
structure IntersectingLine where
  k : ℝ
  m : ℝ
  h_slope : k = 1
  h_distinct : ∃ (x₁ y₁ x₂ y₂ : ℝ), x₁ ≠ x₂ ∧ 
    y₁ = k * x₁ + m ∧ y₂ = k * x₂ + m ∧
    x₁^2 - y₁^2/3 = 1 ∧ x₂^2 - y₂^2/3 = 1
  h_area : ∃ (x₀ y₀ : ℝ), 
    x₀ = (k * m) / (3 - k^2) ∧
    y₀ = (3 * m) / (3 - k^2) ∧
    1/2 * |4 * k * m / (3 - k^2)| * |4 * m / (3 - k^2)| = 4

/-- Main theorem -/
theorem hyperbola_and_line_properties (C : Hyperbola) (l : IntersectingLine) :
  (C.a = 1 ∧ C.b = Real.sqrt 3) ∧ 
  (l.m = Real.sqrt 2 ∨ l.m = -Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_and_line_properties_l598_59841


namespace NUMINAMATH_CALUDE_stockholm_uppsala_distance_l598_59847

/-- Calculates the actual distance between Stockholm and Uppsala based on map measurements and scales. -/
def actual_distance (map_distance : ℝ) (first_part : ℝ) (scale1 : ℝ) (scale2 : ℝ) : ℝ :=
  first_part * scale1 + (map_distance - first_part) * scale2

/-- Theorem stating that the actual distance between Stockholm and Uppsala is 375 km. -/
theorem stockholm_uppsala_distance :
  let map_distance : ℝ := 45
  let first_part : ℝ := 15
  let scale1 : ℝ := 5
  let scale2 : ℝ := 10
  actual_distance map_distance first_part scale1 scale2 = 375 := by
  sorry


end NUMINAMATH_CALUDE_stockholm_uppsala_distance_l598_59847


namespace NUMINAMATH_CALUDE_main_theorem_l598_59890

/-- The set of natural numbers with an odd number of 1s in their binary representation up to 2^n - 1 -/
def A (n : ℕ) : Finset ℕ :=
  sorry

/-- The set of natural numbers with an even number of 1s in their binary representation up to 2^n - 1 -/
def B (n : ℕ) : Finset ℕ :=
  sorry

/-- The difference between the sum of nth powers of numbers in A and B -/
def S (n : ℕ) : ℤ :=
  (A n).sum (fun x => x^n) - (B n).sum (fun x => x^n)

/-- The main theorem stating the closed form of S(n) -/
theorem main_theorem (n : ℕ) : S n = (-1)^(n-1) * (n.factorial : ℤ) * 2^(n*(n-1)/2) :=
  sorry

end NUMINAMATH_CALUDE_main_theorem_l598_59890


namespace NUMINAMATH_CALUDE_greeting_card_exchange_l598_59874

theorem greeting_card_exchange (n : ℕ) (h : n * (n - 1) = 90) : n = 10 := by
  sorry

end NUMINAMATH_CALUDE_greeting_card_exchange_l598_59874


namespace NUMINAMATH_CALUDE_parabola_directrix_equation_l598_59808

/-- Proves that for a parabola y^2 = 2px with focus (2, 0) coinciding with the right focus of the ellipse x^2/9 + y^2/5 = 1, the equation of the directrix is x = -2. -/
theorem parabola_directrix_equation (p : ℝ) : 
  (∀ x y : ℝ, y^2 = 2*p*x → (2 : ℝ) = p/2) → 
  (∀ x y : ℝ, x^2/9 + y^2/5 = 1 → (2 : ℝ) = Real.sqrt (9 - 5)) → 
  (∀ x : ℝ, x = -p/2 ↔ x = -2) := by
sorry

end NUMINAMATH_CALUDE_parabola_directrix_equation_l598_59808


namespace NUMINAMATH_CALUDE_polynomial_substitution_l598_59843

theorem polynomial_substitution (x y : ℝ) :
  y = x + 1 →
  3 * x^3 + 7 * x^2 + 9 * x + 6 = 3 * y^3 - 2 * y^2 + 4 * y + 1 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_substitution_l598_59843


namespace NUMINAMATH_CALUDE_linear_function_x_axis_intersection_l598_59879

/-- A linear function f(x) = -x + 2 -/
def f (x : ℝ) : ℝ := -x + 2

/-- The x-coordinate of the intersection point with the x-axis -/
def x_intersection : ℝ := 2

theorem linear_function_x_axis_intersection :
  f x_intersection = 0 ∧ x_intersection = 2 := by
  sorry

end NUMINAMATH_CALUDE_linear_function_x_axis_intersection_l598_59879


namespace NUMINAMATH_CALUDE_largest_three_digit_multiple_of_3_and_5_l598_59893

theorem largest_three_digit_multiple_of_3_and_5 : ∃ n : ℕ, n = 990 ∧ 
  n % 3 = 0 ∧ n % 5 = 0 ∧ 
  n < 1000 ∧
  ∀ m : ℕ, m < 1000 → m % 3 = 0 → m % 5 = 0 → m ≤ n :=
by sorry

end NUMINAMATH_CALUDE_largest_three_digit_multiple_of_3_and_5_l598_59893


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l598_59877

theorem simplify_and_evaluate :
  (∀ x y : ℚ, x = -2 ∧ y = -3 → 6*x - 5*y + 3*y - 2*x = -2) ∧
  (∀ a : ℚ, a = -1/2 → 1/4*(-4*a^2 + 2*a - 8) - (1/2*a - 2) = -1/4) := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l598_59877


namespace NUMINAMATH_CALUDE_christian_age_when_brian_is_40_l598_59889

/-- Represents a person's age --/
structure Age where
  current : ℕ
  future : ℕ

/-- Represents the ages of Christian and Brian --/
structure AgeRelation where
  christian : Age
  brian : Age
  yearsUntilFuture : ℕ

/-- The conditions of the problem --/
def problemConditions (ages : AgeRelation) : Prop :=
  ages.christian.current = 2 * ages.brian.current ∧
  ages.brian.future = 40 ∧
  ages.christian.future = 72 ∧
  ages.christian.future = ages.christian.current + ages.yearsUntilFuture ∧
  ages.brian.future = ages.brian.current + ages.yearsUntilFuture

/-- The theorem to prove --/
theorem christian_age_when_brian_is_40 (ages : AgeRelation) :
  problemConditions ages → ages.christian.future = 72 := by
  sorry


end NUMINAMATH_CALUDE_christian_age_when_brian_is_40_l598_59889


namespace NUMINAMATH_CALUDE_arithmetic_sequence_seventh_term_l598_59802

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_seventh_term
  (a : ℕ → ℝ)
  (h_arith : ArithmeticSequence a)
  (h_eq1 : a 2 + a 4 + a 5 = a 3 + a 6)
  (h_eq2 : a 9 + a 10 = 3) :
  a 7 = 1 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_seventh_term_l598_59802


namespace NUMINAMATH_CALUDE_sin_decreasing_interval_l598_59821

theorem sin_decreasing_interval :
  ∀ x ∈ Set.Icc (π / 2) (3 * π / 2),
    ∀ y ∈ Set.Icc (π / 2) (3 * π / 2),
      x ≤ y → Real.sin x ≥ Real.sin y :=
by sorry

end NUMINAMATH_CALUDE_sin_decreasing_interval_l598_59821


namespace NUMINAMATH_CALUDE_restaurant_bill_division_l598_59831

theorem restaurant_bill_division (total_bill : ℝ) (num_people : ℕ) (individual_share : ℝ) :
  total_bill = 135 →
  num_people = 3 →
  individual_share = total_bill / num_people →
  individual_share = 45 := by
  sorry

end NUMINAMATH_CALUDE_restaurant_bill_division_l598_59831


namespace NUMINAMATH_CALUDE_dodecagon_diagonals_l598_59863

/-- The number of diagonals in a convex polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- A dodecagon has 12 sides -/
def dodecagon_sides : ℕ := 12

theorem dodecagon_diagonals :
  num_diagonals dodecagon_sides = 54 := by sorry

end NUMINAMATH_CALUDE_dodecagon_diagonals_l598_59863


namespace NUMINAMATH_CALUDE_special_sequence_2011_l598_59887

/-- A sequence satisfying the given conditions -/
def special_sequence (a : ℕ → ℤ) : Prop :=
  a 201 = 2 ∧ ∀ n : ℕ, n > 0 → a n + a (n + 1) = 0

/-- The 2011th term of the special sequence is 2 -/
theorem special_sequence_2011 (a : ℕ → ℤ) (h : special_sequence a) : a 2011 = 2 := by
  sorry

end NUMINAMATH_CALUDE_special_sequence_2011_l598_59887


namespace NUMINAMATH_CALUDE_coin_toss_sequences_l598_59836

/-- The number of coin tosses in the sequence -/
def n : ℕ := 15

/-- The number of "HH" (heads followed by heads) in the sequence -/
def hh_count : ℕ := 2

/-- The number of "HT" (heads followed by tails) in the sequence -/
def ht_count : ℕ := 3

/-- The number of "TH" (tails followed by heads) in the sequence -/
def th_count : ℕ := 4

/-- The number of "TT" (tails followed by tails) in the sequence -/
def tt_count : ℕ := 5

/-- The total number of distinct sequences -/
def total_sequences : ℕ := 2522520

/-- Theorem stating that the number of distinct sequences of n coin tosses
    with exactly hh_count "HH", ht_count "HT", th_count "TH", and tt_count "TT"
    is equal to total_sequences -/
theorem coin_toss_sequences :
  (Nat.factorial (n - 1)) / (Nat.factorial hh_count * Nat.factorial ht_count *
  Nat.factorial th_count * Nat.factorial tt_count) = total_sequences := by
  sorry

end NUMINAMATH_CALUDE_coin_toss_sequences_l598_59836


namespace NUMINAMATH_CALUDE_triangle_side_ratio_l598_59851

theorem triangle_side_ratio (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c)
  (h4 : b + c ≤ 2 * a) (h5 : c + a ≤ 2 * b) (h6 : a < b + c) (h7 : b < c + a) :
  2 / 3 < b / a ∧ b / a < 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_ratio_l598_59851


namespace NUMINAMATH_CALUDE_fraction_to_decimal_l598_59878

theorem fraction_to_decimal : (45 : ℚ) / (5^3) = (360 : ℚ) / 1000 := by sorry

end NUMINAMATH_CALUDE_fraction_to_decimal_l598_59878


namespace NUMINAMATH_CALUDE_gcd_of_30_and_45_l598_59898

theorem gcd_of_30_and_45 : Nat.gcd 30 45 = 15 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_30_and_45_l598_59898


namespace NUMINAMATH_CALUDE_volunteer_allocation_schemes_l598_59828

/-- The number of ways to allocate volunteers to projects -/
def allocate_volunteers (n_volunteers : ℕ) (n_projects : ℕ) : ℕ :=
  (n_volunteers.choose 2) * (n_projects.factorial)

/-- Theorem stating that allocating 5 volunteers to 4 projects results in 240 schemes -/
theorem volunteer_allocation_schemes :
  allocate_volunteers 5 4 = 240 :=
by sorry

end NUMINAMATH_CALUDE_volunteer_allocation_schemes_l598_59828


namespace NUMINAMATH_CALUDE_sqrt_2_irrational_l598_59852

theorem sqrt_2_irrational : Irrational (Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_2_irrational_l598_59852


namespace NUMINAMATH_CALUDE_power_difference_equals_l598_59868

theorem power_difference_equals (a b c : ℕ) :
  3^456 - 9^5 / 9^3 = 3^456 - 81 := by sorry

end NUMINAMATH_CALUDE_power_difference_equals_l598_59868


namespace NUMINAMATH_CALUDE_arithmetic_mean_of_one_and_four_l598_59891

theorem arithmetic_mean_of_one_and_four :
  (1 + 4) / 2 = 5/2 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_of_one_and_four_l598_59891


namespace NUMINAMATH_CALUDE_quadruple_base_exponent_l598_59867

theorem quadruple_base_exponent (a b x y s : ℝ) 
  (ha : a > 0) (hb : b > 0) (hx : x > 0) (hy : y > 0) (hs : s > 0)
  (h1 : s = (4*a)^(4*b))
  (h2 : s = a^b * y^b)
  (h3 : y = 4*x) : 
  x = 64 * a^3 := by sorry

end NUMINAMATH_CALUDE_quadruple_base_exponent_l598_59867


namespace NUMINAMATH_CALUDE_quadratic_inequality_implies_a_bound_l598_59830

theorem quadratic_inequality_implies_a_bound (a : ℝ) :
  (∀ x ∈ Set.Ioo (0 : ℝ) (1/2), x^2 + a*x + 1 ≥ 0) → a ≥ -5/2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_implies_a_bound_l598_59830


namespace NUMINAMATH_CALUDE_unique_solution_l598_59834

/-- Represents a five-digit number -/
def FiveDigitNumber := { n : ℕ // 10000 ≤ n ∧ n < 100000 }

/-- Represents a four-digit number -/
def FourDigitNumber := { n : ℕ // 1000 ≤ n ∧ n < 10000 }

/-- Given a five-digit number, returns all possible four-digit numbers
    that can be formed by removing one digit -/
def removeSingleDigit (n : FiveDigitNumber) : Set FourDigitNumber :=
  sorry

/-- The property that defines our solution -/
def isSolution (n : FiveDigitNumber) : Prop :=
  ∃ (m : FourDigitNumber), m ∈ removeSingleDigit n ∧ n.val + m.val = 54321

/-- Theorem stating that 49383 is the unique solution -/
theorem unique_solution :
  ∃! (n : FiveDigitNumber), isSolution n ∧ n.val = 49383 :=
sorry

end NUMINAMATH_CALUDE_unique_solution_l598_59834


namespace NUMINAMATH_CALUDE_max_a4_in_geometric_sequence_l598_59819

/-- A positive geometric sequence -/
def IsPositiveGeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r > 0, ∀ n, a (n + 1) = r * a n ∧ a n > 0

/-- The theorem statement -/
theorem max_a4_in_geometric_sequence
  (a : ℕ → ℝ)
  (h_geo : IsPositiveGeometricSequence a)
  (h_sum : a 3 + a 5 = 4) :
  ∀ b : ℝ, a 4 ≤ b → b ≤ 2 :=
sorry

end NUMINAMATH_CALUDE_max_a4_in_geometric_sequence_l598_59819


namespace NUMINAMATH_CALUDE_clothes_percentage_is_25_percent_l598_59884

def monthly_income : ℝ := 90000
def household_percentage : ℝ := 0.50
def medicine_percentage : ℝ := 0.15
def savings : ℝ := 9000

theorem clothes_percentage_is_25_percent :
  let clothes_expense := monthly_income - (household_percentage * monthly_income + medicine_percentage * monthly_income + savings)
  clothes_expense / monthly_income = 0.25 := by
sorry

end NUMINAMATH_CALUDE_clothes_percentage_is_25_percent_l598_59884


namespace NUMINAMATH_CALUDE_one_circle_exists_l598_59801

def circle_equation (a x y : ℝ) : Prop :=
  x^2 + y^2 + a*x + 2*a*y + 2*a^2 + a - 1 = 0

def is_circle (a : ℝ) : Prop :=
  ∃ (center_x center_y radius : ℝ), 
    radius > 0 ∧
    ∀ (x y : ℝ), circle_equation a x y ↔ (x - center_x)^2 + (y - center_y)^2 = radius^2

def a_set : Set ℝ := {-2, 0, 1, 3/4}

theorem one_circle_exists :
  ∃! (a : ℝ), a ∈ a_set ∧ is_circle a :=
sorry

end NUMINAMATH_CALUDE_one_circle_exists_l598_59801


namespace NUMINAMATH_CALUDE_f_of_x_minus_3_l598_59807

theorem f_of_x_minus_3 (x : ℝ) : (fun (x : ℝ) => x^2) (x - 3) = x^2 - 6*x + 9 := by
  sorry

end NUMINAMATH_CALUDE_f_of_x_minus_3_l598_59807


namespace NUMINAMATH_CALUDE_reflection_line_sum_l598_59888

/-- Given a line y = mx + b, if the reflection of point (2, 2) across this line is (8, 6), then m + b = 10 -/
theorem reflection_line_sum (m b : ℝ) : 
  (∀ (x y : ℝ), (x - 2)^2 + (y - 2)^2 = (8 - x)^2 + (6 - y)^2 → y = m*x + b) →
  m + b = 10 := by sorry

end NUMINAMATH_CALUDE_reflection_line_sum_l598_59888


namespace NUMINAMATH_CALUDE_solution_set_and_range_l598_59875

def f (x : ℝ) : ℝ := |2 * x - 1| + 1

theorem solution_set_and_range :
  (∀ x : ℝ, f x ≤ 6 ↔ -2 ≤ x ∧ x ≤ 3) ∧
  (∀ m : ℝ, (∃ n : ℝ, f n ≤ m - f (-n)) ↔ 4 ≤ m) := by sorry

end NUMINAMATH_CALUDE_solution_set_and_range_l598_59875


namespace NUMINAMATH_CALUDE_mike_siblings_l598_59869

-- Define the characteristics
inductive EyeColor
| Blue
| Green

inductive HairColor
| Black
| Blonde

inductive Sport
| Soccer
| Basketball

-- Define a child's characteristics
structure ChildCharacteristics where
  eyeColor : EyeColor
  hairColor : HairColor
  favoriteSport : Sport

-- Define the children
def Lily : ChildCharacteristics := ⟨EyeColor.Green, HairColor.Black, Sport.Soccer⟩
def Mike : ChildCharacteristics := ⟨EyeColor.Blue, HairColor.Blonde, Sport.Basketball⟩
def Oliver : ChildCharacteristics := ⟨EyeColor.Blue, HairColor.Black, Sport.Soccer⟩
def Emma : ChildCharacteristics := ⟨EyeColor.Green, HairColor.Blonde, Sport.Basketball⟩
def Jacob : ChildCharacteristics := ⟨EyeColor.Blue, HairColor.Blonde, Sport.Soccer⟩
def Sophia : ChildCharacteristics := ⟨EyeColor.Green, HairColor.Blonde, Sport.Soccer⟩

-- Define a function to check if two children share at least one characteristic
def shareCharacteristic (child1 child2 : ChildCharacteristics) : Prop :=
  child1.eyeColor = child2.eyeColor ∨ 
  child1.hairColor = child2.hairColor ∨ 
  child1.favoriteSport = child2.favoriteSport

-- Define the theorem
theorem mike_siblings : 
  shareCharacteristic Mike Emma ∧ 
  shareCharacteristic Mike Jacob ∧ 
  shareCharacteristic Emma Jacob ∧
  ¬(shareCharacteristic Mike Lily ∧ shareCharacteristic Mike Oliver ∧ shareCharacteristic Mike Sophia) :=
by sorry

end NUMINAMATH_CALUDE_mike_siblings_l598_59869


namespace NUMINAMATH_CALUDE_sarah_wide_reflections_correct_l598_59826

/-- The number of times Sarah sees her reflection in the room with tall mirrors -/
def sarah_tall_reflections : ℕ := 10

/-- The number of times Ellie sees her reflection in the room with tall mirrors -/
def ellie_tall_reflections : ℕ := 6

/-- The number of times Ellie sees her reflection in the room with wide mirrors -/
def ellie_wide_reflections : ℕ := 3

/-- The number of times they both passed through the room with tall mirrors -/
def tall_mirror_passes : ℕ := 3

/-- The number of times they both passed through the room with wide mirrors -/
def wide_mirror_passes : ℕ := 5

/-- The total number of reflections seen by both Sarah and Ellie -/
def total_reflections : ℕ := 88

/-- The number of times Sarah sees her reflection in the room with wide mirrors -/
def sarah_wide_reflections : ℕ := 5

theorem sarah_wide_reflections_correct :
  sarah_tall_reflections * tall_mirror_passes +
  sarah_wide_reflections * wide_mirror_passes +
  ellie_tall_reflections * tall_mirror_passes +
  ellie_wide_reflections * wide_mirror_passes = total_reflections :=
by sorry

end NUMINAMATH_CALUDE_sarah_wide_reflections_correct_l598_59826


namespace NUMINAMATH_CALUDE_angle_bisector_length_l598_59876

-- Define the triangle DEF
structure Triangle :=
  (DE : ℝ)
  (EF : ℝ)
  (DF : ℝ)

-- Define the angle bisector EG
def angleBisector (t : Triangle) : ℝ := sorry

-- Theorem statement
theorem angle_bisector_length (t : Triangle) 
  (h1 : t.DE = 4)
  (h2 : t.EF = 5)
  (h3 : t.DF = 6) :
  angleBisector t = 3 * Real.sqrt 6 := by sorry

end NUMINAMATH_CALUDE_angle_bisector_length_l598_59876


namespace NUMINAMATH_CALUDE_max_distance_circle_to_line_l598_59861

/-- The maximum distance from the center of the circle x² + y² = 4 to the line mx + (5-2m)y - 2 = 0, where m ∈ ℝ, is 2√5/5. -/
theorem max_distance_circle_to_line :
  let circle := {(x, y) : ℝ × ℝ | x^2 + y^2 = 4}
  let line (m : ℝ) := {(x, y) : ℝ × ℝ | m*x + (5 - 2*m)*y - 2 = 0}
  ∀ m : ℝ, (⨆ p ∈ line m, dist (0, 0) p) = 2 * Real.sqrt 5 / 5 :=
by sorry

end NUMINAMATH_CALUDE_max_distance_circle_to_line_l598_59861


namespace NUMINAMATH_CALUDE_square_area_from_diagonal_l598_59858

/-- Given a square with diagonal length 40, prove its area is 800 -/
theorem square_area_from_diagonal (d : ℝ) (h : d = 40) : d^2 / 2 = 800 := by
  sorry

end NUMINAMATH_CALUDE_square_area_from_diagonal_l598_59858


namespace NUMINAMATH_CALUDE_slope_of_line_l598_59892

theorem slope_of_line (x y : ℝ) :
  x + 2 * y - 4 = 0 → (∃ m b : ℝ, y = m * x + b ∧ m = -1/2) :=
by
  sorry

end NUMINAMATH_CALUDE_slope_of_line_l598_59892


namespace NUMINAMATH_CALUDE_min_value_theorem_l598_59823

theorem min_value_theorem (x : ℝ) : 
  (x^2 + 9) / Real.sqrt (x^2 + 5) ≥ 4 ∧ 
  ∃ y : ℝ, (y^2 + 9) / Real.sqrt (y^2 + 5) = 4 := by
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l598_59823


namespace NUMINAMATH_CALUDE_boat_speed_in_still_water_l598_59853

/-- A boat traveling downstream with the help of a stream. -/
structure BoatTrip where
  boat_speed : ℝ      -- Speed of the boat in still water (km/hr)
  stream_speed : ℝ    -- Speed of the stream (km/hr)
  time : ℝ            -- Time taken for the trip (hours)
  distance : ℝ        -- Distance traveled (km)

/-- The theorem stating the boat's speed in still water given the conditions. -/
theorem boat_speed_in_still_water (trip : BoatTrip)
  (h1 : trip.stream_speed = 5)
  (h2 : trip.time = 5)
  (h3 : trip.distance = 135) :
  trip.boat_speed = 22 := by
  sorry

end NUMINAMATH_CALUDE_boat_speed_in_still_water_l598_59853


namespace NUMINAMATH_CALUDE_part_one_part_two_l598_59864

-- Define the & operation
def ampersand (a b : ℚ) : ℚ := b^2 - a*b

-- Part 1
theorem part_one : ampersand (2/3) (-1/2) = 7/12 := by sorry

-- Part 2
theorem part_two (x y : ℚ) (h : |x + 1| + (y - 3)^2 = 0) : 
  ampersand x y = 12 := by sorry

end NUMINAMATH_CALUDE_part_one_part_two_l598_59864


namespace NUMINAMATH_CALUDE_min_value_fraction_l598_59839

theorem min_value_fraction (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  (1 / a + 2 / b) ≥ 3 + 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_fraction_l598_59839


namespace NUMINAMATH_CALUDE_boat_speed_in_still_water_l598_59837

/-- The speed of a boat in still water, given its downstream and upstream distances traveled in one hour. -/
theorem boat_speed_in_still_water (downstream upstream : ℝ) (h1 : downstream = 11) (h2 : upstream = 5) :
  let boat_speed := (downstream + upstream) / 2
  boat_speed = 8 := by sorry

end NUMINAMATH_CALUDE_boat_speed_in_still_water_l598_59837


namespace NUMINAMATH_CALUDE_symmetrical_line_slope_range_l598_59894

/-- Given a line l: y = kx - 1 intersecting with x + y - 1 = 0,
    the range of k for which a symmetrical line can be derived is (1, +∞) -/
theorem symmetrical_line_slope_range (k : ℝ) : 
  (∃ (x y : ℝ), y = k * x - 1 ∧ x + y - 1 = 0) →
  (∃ (m : ℝ), m ≠ k ∧ ∃ (x₀ y₀ : ℝ), (∀ (x y : ℝ), y - y₀ = m * (x - x₀) ↔ y = k * x - 1)) ↔
  k > 1 :=
sorry

end NUMINAMATH_CALUDE_symmetrical_line_slope_range_l598_59894


namespace NUMINAMATH_CALUDE_revenue_decrease_l598_59865

theorem revenue_decrease (T C : ℝ) (hT : T > 0) (hC : C > 0) :
  let new_tax := 0.7 * T
  let new_consumption := 1.2 * C
  let original_revenue := T * C
  let new_revenue := new_tax * new_consumption
  (original_revenue - new_revenue) / original_revenue = 0.16 := by
  sorry

end NUMINAMATH_CALUDE_revenue_decrease_l598_59865


namespace NUMINAMATH_CALUDE_triangle_existence_l598_59833

theorem triangle_existence (a b c A B C : ℝ) 
  (h1 : a > 0 ∧ b > 0 ∧ c > 0)
  (h2 : A > 0 ∧ B > 0 ∧ C > 0)
  (h3 : a + b > c ∧ b + c > a ∧ c + a > b)
  (h4 : A + B > C ∧ B + C > A ∧ C + A > B) :
  ∃ (x y z : ℝ), 
    x = Real.sqrt (a^2 + A^2) ∧
    y = Real.sqrt (b^2 + B^2) ∧
    z = Real.sqrt (c^2 + C^2) ∧
    x + y > z ∧ y + z > x ∧ z + x > y :=
sorry

end NUMINAMATH_CALUDE_triangle_existence_l598_59833


namespace NUMINAMATH_CALUDE_volume_ratio_of_rotated_triangle_l598_59896

/-- Given a right-angled triangle with perpendicular sides of lengths a and b,
    the ratio of the volume of the solid formed by rotating around side a
    to the volume of the solid formed by rotating around side b is b : a. -/
theorem volume_ratio_of_rotated_triangle (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  (1 / 3 * π * b^2 * a) / (1 / 3 * π * a^2 * b) = b / a :=
sorry

end NUMINAMATH_CALUDE_volume_ratio_of_rotated_triangle_l598_59896


namespace NUMINAMATH_CALUDE_tank_capacity_l598_59824

theorem tank_capacity (initial_fraction : Rat) (final_fraction : Rat) (used_gallons : ℕ) 
  (h1 : initial_fraction = 3/4)
  (h2 : final_fraction = 1/3)
  (h3 : used_gallons = 18) :
  ∃ (capacity : ℕ), 
    capacity * initial_fraction - capacity * final_fraction = used_gallons ∧ 
    capacity = 43 := by
sorry


end NUMINAMATH_CALUDE_tank_capacity_l598_59824


namespace NUMINAMATH_CALUDE_batsman_average_increase_l598_59872

/-- Represents a batsman's performance -/
structure Batsman where
  innings : ℕ
  totalRuns : ℕ
  lastInningRuns : ℕ

/-- Calculate the average runs per inning -/
def average (b : Batsman) : ℚ :=
  b.totalRuns / b.innings

/-- Calculate the increase in average -/
def averageIncrease (before after : ℚ) : ℚ :=
  after - before

theorem batsman_average_increase 
  (b : Batsman) 
  (h1 : b.innings = 11) 
  (h2 : b.lastInningRuns = 90) 
  (h3 : average b = 40) :
  averageIncrease 
    (average { innings := b.innings - 1, totalRuns := b.totalRuns - b.lastInningRuns, lastInningRuns := 0 }) 
    (average b) = 5 := by
  sorry

end NUMINAMATH_CALUDE_batsman_average_increase_l598_59872


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l598_59800

/-- An arithmetic sequence is a sequence where the difference between any two consecutive terms is constant. -/
def ArithmeticSequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

/-- The sum of the 5th through 7th terms of an arithmetic sequence. -/
def SumFifthToSeventh (a : ℕ → ℤ) : ℤ := a 5 + a 6 + a 7

theorem arithmetic_sequence_sum (a : ℕ → ℤ) :
  ArithmeticSequence a →
  a 8 = 16 →
  a 9 = 22 →
  a 10 = 28 →
  SumFifthToSeventh a = 12 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l598_59800


namespace NUMINAMATH_CALUDE_correct_factorization_l598_59845

theorem correct_factorization (a : ℝ) : -1 + 4 * a^2 = (2*a + 1) * (2*a - 1) := by
  sorry

end NUMINAMATH_CALUDE_correct_factorization_l598_59845


namespace NUMINAMATH_CALUDE_candy_calculation_correct_l598_59886

/-- Calculates the number of candy pieces Haley's sister gave her. -/
def candy_from_sister (initial : ℕ) (eaten : ℕ) (final : ℕ) : ℕ :=
  final - (initial - eaten)

/-- Proves that the calculation of candy pieces from Haley's sister is correct. -/
theorem candy_calculation_correct (initial eaten final : ℕ) 
  (h1 : initial ≥ eaten) 
  (h2 : final ≥ initial - eaten) : 
  candy_from_sister initial eaten final = final - (initial - eaten) :=
by sorry

end NUMINAMATH_CALUDE_candy_calculation_correct_l598_59886


namespace NUMINAMATH_CALUDE_sanitizer_theorem_l598_59809

/-- Represents the prices and quantities of hand sanitizer and disinfectant -/
structure SanitizerProblem where
  x : ℚ  -- Price of hand sanitizer
  y : ℚ  -- Price of 84 disinfectant
  eq1 : 100 * x + 150 * y = 1500
  eq2 : 120 * x + 160 * y = 1720
  promotion : ℕ → ℕ
  promotion_def : ∀ n : ℕ, promotion n = n / 150 * 10

/-- The solution to the sanitizer problem -/
def sanitizer_solution (p : SanitizerProblem) : Prop :=
  p.x = 9 ∧ p.y = 4 ∧ 
  9 * 150 + 4 * (60 - p.promotion 150) = 1550

/-- The main theorem stating that the solution is correct -/
theorem sanitizer_theorem (p : SanitizerProblem) : sanitizer_solution p := by
  sorry

end NUMINAMATH_CALUDE_sanitizer_theorem_l598_59809


namespace NUMINAMATH_CALUDE_quadratic_intersection_count_l598_59820

/-- The quadratic function f(x) = x^2 - 2x + 1 -/
def f (x : ℝ) : ℝ := x^2 - 2*x + 1

/-- The number of intersection points between f and the coordinate axes -/
def intersection_count : ℕ := 2

theorem quadratic_intersection_count :
  (∃! x, f x = 0) ∧ (f 0 ≠ 0) → intersection_count = 2 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_intersection_count_l598_59820


namespace NUMINAMATH_CALUDE_damien_tall_cupboard_glasses_l598_59880

/-- Represents the number of glasses in different cupboards --/
structure Cupboards where
  tall : ℕ
  wide : ℕ
  narrow : ℕ

/-- The setup of Damien's glass collection --/
def damien_cupboards : Cupboards where
  tall := 5
  wide := 10
  narrow := 10

/-- Theorem stating the number of glasses in Damien's tall cupboard --/
theorem damien_tall_cupboard_glasses :
  ∃ (c : Cupboards), 
    c.wide = 2 * c.tall ∧ 
    c.narrow = 10 ∧ 
    15 % 3 = 0 ∧ 
    c = damien_cupboards :=
by
  sorry

#check damien_tall_cupboard_glasses

end NUMINAMATH_CALUDE_damien_tall_cupboard_glasses_l598_59880


namespace NUMINAMATH_CALUDE_ned_candy_boxes_l598_59806

/-- The number of candy pieces Ned gave to his little brother -/
def pieces_given : ℝ := 7.0

/-- The number of candy pieces in each box -/
def pieces_per_box : ℝ := 6.0

/-- The number of candy pieces Ned still has -/
def pieces_left : ℕ := 42

/-- The number of boxes Ned bought initially -/
def boxes_bought : ℕ := 8

theorem ned_candy_boxes : 
  ⌊(pieces_given + pieces_left : ℝ) / pieces_per_box⌋ = boxes_bought := by
  sorry

end NUMINAMATH_CALUDE_ned_candy_boxes_l598_59806


namespace NUMINAMATH_CALUDE_perpendicular_bisector_c_value_l598_59883

/-- The value of c for which the line 3x - y = c is a perpendicular bisector
    of the line segment from (2,4) to (6,8) -/
theorem perpendicular_bisector_c_value :
  ∃ c : ℝ,
    (∀ x y : ℝ, 3 * x - y = c → 
      ((x - 4) ^ 2 + (y - 6) ^ 2 = 8) ∧ 
      (3 * (x - 4) + (y - 6) = 0)) →
    c = 6 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_bisector_c_value_l598_59883


namespace NUMINAMATH_CALUDE_f_even_implies_a_eq_two_l598_59844

/-- A function f is even if f(-x) = f(x) for all x in its domain -/
def IsEven (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = f x

/-- The function f(x) = x * exp(x) / (exp(a*x) - 1) -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  x * Real.exp x / (Real.exp (a * x) - 1)

/-- If f(x) = x * exp(x) / (exp(a*x) - 1) is an even function, then a = 2 -/
theorem f_even_implies_a_eq_two (a : ℝ) :
  IsEven (f a) → a = 2 := by
  sorry

end NUMINAMATH_CALUDE_f_even_implies_a_eq_two_l598_59844


namespace NUMINAMATH_CALUDE_train_bridge_time_l598_59862

/-- Given a train of length 18 meters that passes a pole in 9 seconds,
    prove that it takes 27 seconds to pass a bridge of length 36 meters. -/
theorem train_bridge_time (train_length : ℝ) (pole_pass_time : ℝ) (bridge_length : ℝ) :
  train_length = 18 →
  pole_pass_time = 9 →
  bridge_length = 36 →
  (train_length + bridge_length) / (train_length / pole_pass_time) = 27 := by
  sorry

end NUMINAMATH_CALUDE_train_bridge_time_l598_59862


namespace NUMINAMATH_CALUDE_max_integer_a_for_real_roots_l598_59881

theorem max_integer_a_for_real_roots : 
  ∀ a : ℤ, 
    a ≠ 1 →
    (∃ x : ℝ, (a - 1) * x^2 - 2 * x + 3 = 0) →
    a ≤ 0 :=
by sorry

end NUMINAMATH_CALUDE_max_integer_a_for_real_roots_l598_59881


namespace NUMINAMATH_CALUDE_prime_arithmetic_progression_difference_divisibility_l598_59855

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

def arithmetic_progression (a : ℕ → ℕ) (d : ℕ) :=
  ∀ n : ℕ, a (n + 1) = a n + d

theorem prime_arithmetic_progression_difference_divisibility
  (p : ℕ → ℕ)
  (d : ℕ)
  (h_prime : ∀ n, n ∈ Finset.range 15 → is_prime (p n))
  (h_increasing : ∀ n, n ∈ Finset.range 14 → p n < p (n + 1))
  (h_arith_prog : arithmetic_progression p d) :
  ∃ k : ℕ, d = k * (2 * 3 * 5 * 7 * 11 * 13) :=
sorry

end NUMINAMATH_CALUDE_prime_arithmetic_progression_difference_divisibility_l598_59855


namespace NUMINAMATH_CALUDE_isosceles_triangles_not_necessarily_congruent_l598_59825

/-- Represents an isosceles triangle -/
structure IsoscelesTriangle where
  /-- Length of the equal sides -/
  side_length : ℝ
  /-- Length of the base -/
  base_length : ℝ
  /-- Radius of the inscribed circle -/
  incircle_radius : ℝ
  /-- side_length is positive -/
  side_length_pos : 0 < side_length
  /-- base_length is positive -/
  base_length_pos : 0 < base_length
  /-- incircle_radius is positive -/
  incircle_radius_pos : 0 < incircle_radius
  /-- The base cannot be longer than twice the side length -/
  base_bound : base_length ≤ 2 * side_length
  /-- Relation between side length, base length, and incircle radius -/
  geometry_constraint : incircle_radius = (base_length * Real.sqrt (side_length^2 - (base_length/2)^2)) / (2 * side_length + base_length)

/-- Two isosceles triangles with the same side length and incircle radius are not necessarily congruent -/
theorem isosceles_triangles_not_necessarily_congruent :
  ∃ (t1 t2 : IsoscelesTriangle), 
    t1.side_length = t2.side_length ∧ 
    t1.incircle_radius = t2.incircle_radius ∧ 
    t1.base_length ≠ t2.base_length :=
by sorry

end NUMINAMATH_CALUDE_isosceles_triangles_not_necessarily_congruent_l598_59825


namespace NUMINAMATH_CALUDE_quadratic_equation_root_l598_59816

theorem quadratic_equation_root (x : ℝ) : x^2 + 6*x + 4 = 0 ↔ x = Real.sqrt 5 - 3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_root_l598_59816


namespace NUMINAMATH_CALUDE_g_inequality_l598_59814

/-- A quadratic function f(x) = ax^2 + a that is even on the interval [-a, a^2] -/
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + a

/-- The function g(x) = f(x-1) -/
def g (a : ℝ) (x : ℝ) : ℝ := f a (x - 1)

/-- Theorem stating the relationship between g(3/2), g(0), and g(3) -/
theorem g_inequality (a : ℝ) (h1 : a ≠ 0) 
  (h2 : ∀ x ∈ Set.Icc (-a) (a^2), f a x = f a (-x)) : 
  g a (3/2) < g a 0 ∧ g a 0 < g a 3 := by
  sorry

end NUMINAMATH_CALUDE_g_inequality_l598_59814


namespace NUMINAMATH_CALUDE_greatest_common_measure_l598_59815

theorem greatest_common_measure (a b c : ℕ) (ha : a = 729000) (hb : b = 1242500) (hc : c = 32175) :
  Nat.gcd a (Nat.gcd b c) = 225 := by
  sorry

end NUMINAMATH_CALUDE_greatest_common_measure_l598_59815


namespace NUMINAMATH_CALUDE_total_tabs_is_sixty_l598_59810

/-- Calculates the total number of tabs opened across all browsers -/
def totalTabs (numBrowsers : ℕ) (windowsPerBrowser : ℕ) (tabsPerWindow : ℕ) : ℕ :=
  numBrowsers * windowsPerBrowser * tabsPerWindow

/-- Theorem: Given the specified conditions, the total number of tabs is 60 -/
theorem total_tabs_is_sixty :
  totalTabs 2 3 10 = 60 := by
  sorry

end NUMINAMATH_CALUDE_total_tabs_is_sixty_l598_59810


namespace NUMINAMATH_CALUDE_circle_equation_perpendicular_chord_values_l598_59827

-- Define the circle
def circle_center : ℝ × ℝ := (2, 0)
def circle_radius : ℝ := 5

-- Define the tangent line
def tangent_line (x y : ℝ) : Prop := 4 * x + 3 * y - 33 = 0

-- Define the intersecting line
def intersecting_line (a x y : ℝ) : Prop := a * x - y - 7 = 0

-- Theorem for the circle equation
theorem circle_equation : 
  ∀ x y : ℝ, (x - circle_center.1)^2 + (y - circle_center.2)^2 = circle_radius^2 ↔ 
  (x - 2)^2 + y^2 = 25 := by sorry

-- Theorem for the values of a
theorem perpendicular_chord_values (a : ℝ) :
  (∃ A B : ℝ × ℝ, 
    (A.1 - circle_center.1)^2 + (A.2 - circle_center.2)^2 = circle_radius^2 ∧
    (B.1 - circle_center.1)^2 + (B.2 - circle_center.2)^2 = circle_radius^2 ∧
    intersecting_line a A.1 A.2 ∧
    intersecting_line a B.1 B.2 ∧
    ((A.1 - circle_center.1) * (B.1 - circle_center.1) + (A.2 - circle_center.2) * (B.2 - circle_center.2) = 0)) →
  (a = 1 ∨ a = -73/17) := by sorry

end NUMINAMATH_CALUDE_circle_equation_perpendicular_chord_values_l598_59827


namespace NUMINAMATH_CALUDE_double_plus_five_positive_l598_59871

theorem double_plus_five_positive (m : ℝ) :
  (2 * m + 5 > 0) ↔ (∃ x > 0, x = 2 * m + 5) :=
by sorry

end NUMINAMATH_CALUDE_double_plus_five_positive_l598_59871


namespace NUMINAMATH_CALUDE_pizza_price_l598_59804

theorem pizza_price (num_pizzas : ℕ) (tip : ℝ) (bill : ℝ) (change : ℝ) :
  num_pizzas = 4 ∧ tip = 5 ∧ bill = 50 ∧ change = 5 →
  ∃ (price : ℝ), price = 10 ∧ num_pizzas * price + tip = bill - change :=
by sorry

end NUMINAMATH_CALUDE_pizza_price_l598_59804


namespace NUMINAMATH_CALUDE_triangle_ratio_l598_59811

/-- Given a triangle ABC with angle A = 60°, side b = 1, and area = √3,
    prove that (a+b+c)/(sin A + sin B + sin C) = 2√39/3 -/
theorem triangle_ratio (a b c A B C : ℝ) : 
  A = π/3 → 
  b = 1 → 
  (1/2) * b * c * Real.sin A = Real.sqrt 3 →
  (a + b + c) / (Real.sin A + Real.sin B + Real.sin C) = 2 * Real.sqrt 39 / 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_ratio_l598_59811


namespace NUMINAMATH_CALUDE_certain_amount_proof_l598_59840

theorem certain_amount_proof (A : ℝ) : 
  (0.20 * 1050 = 0.15 * 1500 - A) → A = 15 := by
  sorry

end NUMINAMATH_CALUDE_certain_amount_proof_l598_59840


namespace NUMINAMATH_CALUDE_product_of_base9_digits_9876_l598_59897

/-- Converts a base 10 number to base 9 --/
def toBase9 (n : ℕ) : List ℕ :=
  sorry

/-- Calculates the product of a list of natural numbers --/
def productOfList (l : List ℕ) : ℕ :=
  sorry

/-- Theorem: The product of the digits in the base 9 representation of 9876₁₀ is 192 --/
theorem product_of_base9_digits_9876 :
  productOfList (toBase9 9876) = 192 := by
  sorry

end NUMINAMATH_CALUDE_product_of_base9_digits_9876_l598_59897


namespace NUMINAMATH_CALUDE_constant_t_value_l598_59835

theorem constant_t_value : ∃! t : ℝ, ∀ x : ℝ, 
  (3*x^2 - 4*x + 5) * (5*x^2 + t*x + 15) = 15*x^4 - 47*x^3 + 115*x^2 - 110*x + 75 ∧ t = -10 := by
  sorry

end NUMINAMATH_CALUDE_constant_t_value_l598_59835


namespace NUMINAMATH_CALUDE_new_person_weight_l598_59850

theorem new_person_weight (initial_count : ℕ) (weight_increase : ℝ) (replaced_weight : ℝ) :
  initial_count = 8 →
  weight_increase = 2.5 →
  replaced_weight = 85 →
  ∃ (new_weight : ℝ), new_weight = 105 ∧
    (initial_count : ℝ) * weight_increase = new_weight - replaced_weight :=
by sorry

end NUMINAMATH_CALUDE_new_person_weight_l598_59850


namespace NUMINAMATH_CALUDE_f_increasing_iff_a_nonnegative_l598_59822

/-- A function f is increasing on an interval [a, +∞) if for all x, y in the interval with x < y, f(x) < f(y) -/
def IncreasingOnInterval (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x y, a ≤ x → x < y → f x < f y

/-- The function f(x) = x^2 + 2(a-1)x + 3 -/
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + 2*(a-1)*x + 3

theorem f_increasing_iff_a_nonnegative :
  ∀ a : ℝ, IncreasingOnInterval (f a) 1 ↔ a ∈ Set.Ici 0 := by
  sorry

end NUMINAMATH_CALUDE_f_increasing_iff_a_nonnegative_l598_59822


namespace NUMINAMATH_CALUDE_bookseller_display_windows_l598_59813

/-- Given the conditions of the bookseller's display windows problem, prove that the number of non-fiction books is 2. -/
theorem bookseller_display_windows (fiction_books : ℕ) (display_fiction : ℕ) (total_configs : ℕ) :
  fiction_books = 4 →
  display_fiction = 3 →
  total_configs = 36 →
  ∃ n : ℕ, n = 2 ∧ (Nat.factorial fiction_books / Nat.factorial (fiction_books - display_fiction)) * Nat.factorial n = total_configs :=
by sorry

end NUMINAMATH_CALUDE_bookseller_display_windows_l598_59813


namespace NUMINAMATH_CALUDE_quadratic_roots_sum_product_l598_59829

theorem quadratic_roots_sum_product (p q : ℝ) : 
  (∀ x : ℝ, 3 * x^2 - p * x + q = 0 → 
    (∃ r₁ r₂ : ℝ, r₁ + r₂ = 9 ∧ r₁ * r₂ = 15 ∧ 
      (3 * r₁^2 - p * r₁ + q = 0) ∧ (3 * r₂^2 - p * r₂ + q = 0))) →
  p + q = 72 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_sum_product_l598_59829


namespace NUMINAMATH_CALUDE_quadratic_other_x_intercept_l598_59803

/-- Given a quadratic function f(x) with vertex (5,10) and one x-intercept at (1,0),
    the x-coordinate of the other x-intercept is 9. -/
theorem quadratic_other_x_intercept 
  (f : ℝ → ℝ) 
  (h1 : ∃ a b c : ℝ, ∀ x, f x = a * x^2 + b * x + c) 
  (h2 : f 1 = 0) 
  (h3 : ∃ y, f 5 = y ∧ y = 10) :
  ∃ x, f x = 0 ∧ x = 9 :=
sorry

end NUMINAMATH_CALUDE_quadratic_other_x_intercept_l598_59803


namespace NUMINAMATH_CALUDE_odd_divisor_of_4a_squared_minus_1_l598_59812

theorem odd_divisor_of_4a_squared_minus_1 (n : ℤ) (h : Odd n) :
  ∃ a : ℤ, n ∣ (4 * a^2 - 1) := by
  sorry

end NUMINAMATH_CALUDE_odd_divisor_of_4a_squared_minus_1_l598_59812


namespace NUMINAMATH_CALUDE_chef_michel_pies_l598_59846

/-- Represents the number of pies sold given the number of pieces and customers -/
def pies_sold (pieces : ℕ) (customers : ℕ) : ℕ :=
  (customers + pieces - 1) / pieces

/-- The total number of pies sold by Chef Michel -/
def total_pies : ℕ :=
  pies_sold 4 52 + pies_sold 8 76 + pies_sold 5 80 + pies_sold 10 130

/-- Theorem stating that Chef Michel sold 52 pies in total -/
theorem chef_michel_pies :
  total_pies = 52 := by
  sorry

#eval total_pies

end NUMINAMATH_CALUDE_chef_michel_pies_l598_59846


namespace NUMINAMATH_CALUDE_count_valid_labelings_l598_59849

/-- A labeling of the edges of a rectangular prism with 0s and 1s. -/
def Labeling := Fin 12 → Fin 2

/-- The set of faces of a rectangular prism. -/
def Face := Fin 6

/-- The edges that make up each face of the rectangular prism. -/
def face_edges : Face → Finset (Fin 12) :=
  sorry

/-- The sum of labels on a given face for a given labeling. -/
def face_sum (l : Labeling) (f : Face) : Nat :=
  (face_edges f).sum (fun e => l e)

/-- A labeling is valid if the sum of labels on each face is exactly 2. -/
def is_valid_labeling (l : Labeling) : Prop :=
  ∀ f : Face, face_sum l f = 2

/-- The set of all valid labelings. -/
def valid_labelings : Finset Labeling :=
  sorry

theorem count_valid_labelings :
  valid_labelings.card = 16 :=
sorry

end NUMINAMATH_CALUDE_count_valid_labelings_l598_59849


namespace NUMINAMATH_CALUDE_hundredth_term_difference_l598_59817

/-- An arithmetic sequence with specific properties -/
structure ArithmeticSequence where
  terms : ℕ
  min_value : ℝ
  max_value : ℝ
  sum : ℝ

/-- The properties of our specific arithmetic sequence -/
def our_sequence : ArithmeticSequence where
  terms := 350
  min_value := 5
  max_value := 150
  sum := 38500

/-- The 100th term of an arithmetic sequence -/
def hundredth_term (a d : ℝ) : ℝ := a + 99 * d

/-- Theorem stating the difference between max and min possible 100th terms -/
theorem hundredth_term_difference (seq : ArithmeticSequence) 
  (h_seq : seq = our_sequence) : 
  ∃ (L G : ℝ), 
    (∀ (a d : ℝ), 
      (seq.min_value ≤ a) ∧ 
      (a + (seq.terms - 1) * d ≤ seq.max_value) ∧
      (seq.sum = (seq.terms : ℝ) * (2 * a + (seq.terms - 1) * d) / 2) →
      (L ≤ hundredth_term a d ∧ hundredth_term a d ≤ G)) ∧
    (G - L = 60.225) := by
  sorry

end NUMINAMATH_CALUDE_hundredth_term_difference_l598_59817


namespace NUMINAMATH_CALUDE_kalebs_savings_l598_59818

/-- The amount of money Kaleb needs to buy the toys -/
def total_cost (num_toys : ℕ) (price_per_toy : ℕ) : ℕ := num_toys * price_per_toy

/-- The amount of money Kaleb has saved initially -/
def initial_savings (total_cost additional_money : ℕ) : ℕ := total_cost - additional_money

/-- Theorem stating Kaleb's initial savings -/
theorem kalebs_savings (num_toys price_per_toy additional_money : ℕ) 
  (h1 : num_toys = 6)
  (h2 : price_per_toy = 6)
  (h3 : additional_money = 15) :
  initial_savings (total_cost num_toys price_per_toy) additional_money = 21 := by
  sorry

#check kalebs_savings

end NUMINAMATH_CALUDE_kalebs_savings_l598_59818


namespace NUMINAMATH_CALUDE_age_difference_l598_59842

theorem age_difference (alice_age bob_age : ℕ) : 
  alice_age + 5 = 19 →
  alice_age + 6 = 2 * (bob_age + 6) →
  alice_age - bob_age = 10 := by
sorry

end NUMINAMATH_CALUDE_age_difference_l598_59842


namespace NUMINAMATH_CALUDE_max_sum_of_squares_l598_59838

theorem max_sum_of_squares (a b c d : ℝ) : 
  a + b = 20 →
  a * b + c + d = 90 →
  a * d + b * c = 210 →
  c * d = 125 →
  a^2 + b^2 + c^2 + d^2 ≤ 1450 ∧ 
  ∃ (a' b' c' d' : ℝ), 
    a' + b' = 20 ∧
    a' * b' + c' + d' = 90 ∧
    a' * d' + b' * c' = 210 ∧
    c' * d' = 125 ∧
    a'^2 + b'^2 + c'^2 + d'^2 = 1450 :=
by sorry

end NUMINAMATH_CALUDE_max_sum_of_squares_l598_59838


namespace NUMINAMATH_CALUDE_min_value_equals_gcd_l598_59899

theorem min_value_equals_gcd (a b c : ℕ+) :
  (∃ (x y z : ℤ), ∀ (x' y' z' : ℤ), a * x + b * y + c * z ≤ a * x' + b * y' + c * z' ∧ 0 < a * x + b * y + c * z) →
  (∃ (x y z : ℤ), a * x + b * y + c * z = Nat.gcd a.val (Nat.gcd b.val c.val)) := by
  sorry

end NUMINAMATH_CALUDE_min_value_equals_gcd_l598_59899


namespace NUMINAMATH_CALUDE_face_moisturizer_cost_l598_59873

/-- Proves that the cost of each face moisturizer is $50 given the problem conditions -/
theorem face_moisturizer_cost (tanya_face_moisturizer_cost : ℝ) 
  (h1 : 2 * (2 * tanya_face_moisturizer_cost + 4 * 60) = 2 * tanya_face_moisturizer_cost + 4 * 60 + 1020) : 
  tanya_face_moisturizer_cost = 50 := by
  sorry

end NUMINAMATH_CALUDE_face_moisturizer_cost_l598_59873


namespace NUMINAMATH_CALUDE_combined_salary_proof_l598_59895

/-- The combined salary of two people A and B -/
def combinedSalary (salaryA salaryB : ℝ) : ℝ := salaryA + salaryB

/-- The savings of a person given their salary and spending percentage -/
def savings (salary spendingPercentage : ℝ) : ℝ := salary * (1 - spendingPercentage)

theorem combined_salary_proof (salaryA salaryB : ℝ) 
  (hSpendA : savings salaryA 0.8 = savings salaryB 0.85)
  (hSalaryB : salaryB = 8000) :
  combinedSalary salaryA salaryB = 14000 := by
  sorry

end NUMINAMATH_CALUDE_combined_salary_proof_l598_59895
