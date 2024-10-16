import Mathlib

namespace NUMINAMATH_CALUDE_billboard_fully_lit_probability_l3261_326190

/-- The number of words in the billboard text -/
def num_words : ℕ := 5

/-- The probability of seeing the billboard fully lit -/
def fully_lit_probability : ℚ := 1 / num_words

/-- Theorem stating that the probability of seeing the billboard fully lit is 1/5 -/
theorem billboard_fully_lit_probability :
  fully_lit_probability = 1 / 5 := by sorry

end NUMINAMATH_CALUDE_billboard_fully_lit_probability_l3261_326190


namespace NUMINAMATH_CALUDE_pure_imaginary_solution_l3261_326153

-- Define the complex number i
def i : ℂ := Complex.I

-- Define the complex-valued function z
def z (m : ℝ) : ℂ := (1 + i) * m^2 - (4 + i) * m + 3

-- Theorem statement
theorem pure_imaginary_solution (m : ℝ) :
  (z m).re = 0 ∧ (z m).im ≠ 0 → m = 3 :=
sorry

end NUMINAMATH_CALUDE_pure_imaginary_solution_l3261_326153


namespace NUMINAMATH_CALUDE_greatest_common_multiple_15_20_less_than_150_l3261_326199

def is_common_multiple (m n k : ℕ) : Prop := k % m = 0 ∧ k % n = 0

theorem greatest_common_multiple_15_20_less_than_150 : 
  ∃ (k : ℕ), k = 120 ∧ 
  is_common_multiple 15 20 k ∧ 
  k < 150 ∧ 
  ∀ (m : ℕ), is_common_multiple 15 20 m → m < 150 → m ≤ k :=
sorry

end NUMINAMATH_CALUDE_greatest_common_multiple_15_20_less_than_150_l3261_326199


namespace NUMINAMATH_CALUDE_brown_eyed_brunettes_l3261_326185

theorem brown_eyed_brunettes (total : ℕ) (blondes : ℕ) (brunettes : ℕ) (blue_eyed : ℕ) (brown_eyed : ℕ) (blue_eyed_blondes : ℕ) :
  total = 60 →
  blondes + brunettes = total →
  blue_eyed + brown_eyed = total →
  brunettes = 35 →
  blue_eyed_blondes = 20 →
  brown_eyed = 22 →
  brown_eyed - (blondes - blue_eyed_blondes) = 17 :=
by sorry

end NUMINAMATH_CALUDE_brown_eyed_brunettes_l3261_326185


namespace NUMINAMATH_CALUDE_defective_units_percentage_l3261_326157

theorem defective_units_percentage 
  (shipped_defective_ratio : Real) 
  (total_shipped_defective_ratio : Real) 
  (h1 : shipped_defective_ratio = 0.04)
  (h2 : total_shipped_defective_ratio = 0.0016) : 
  total_shipped_defective_ratio / shipped_defective_ratio = 0.04 := by
sorry

end NUMINAMATH_CALUDE_defective_units_percentage_l3261_326157


namespace NUMINAMATH_CALUDE_triangle_sides_from_heights_l3261_326181

/-- Given a triangle with heights d, e, and f corresponding to sides a, b, and c respectively,
    this theorem states the relationship between the sides and heights. -/
theorem triangle_sides_from_heights (d e f : ℝ) (hd : d > 0) (he : e > 0) (hf : f > 0) :
  ∃ (a b c : ℝ),
    let A := ((1/d + 1/e + 1/f) * (-1/d + 1/e + 1/f) * (1/d - 1/e + 1/f) * (1/d + 1/e - 1/f))
    a = 2 / (d * Real.sqrt A) ∧
    b = 2 / (e * Real.sqrt A) ∧
    c = 2 / (f * Real.sqrt A) :=
sorry

end NUMINAMATH_CALUDE_triangle_sides_from_heights_l3261_326181


namespace NUMINAMATH_CALUDE_distance_from_center_l3261_326106

-- Define the circle and points
def Circle := {p : ℝ × ℝ | p.1^2 + p.2^2 = 50}

-- Define the conditions
def Conditions (A B C : ℝ × ℝ) : Prop :=
  A ∈ Circle ∧ 
  C ∈ Circle ∧ 
  (A.1 - B.1)^2 + (A.2 - B.2)^2 = 36 ∧  -- AB = 6
  (C.1 - B.1)^2 + (C.2 - B.2)^2 = 4 ∧   -- BC = 2
  (A.1 - B.1) * (C.1 - B.1) + (A.2 - B.2) * (C.2 - B.2) = 0  -- Angle ABC is right

-- State the theorem
theorem distance_from_center (A B C : ℝ × ℝ) 
  (h : Conditions A B C) : B.1^2 + B.2^2 = 26 := by
  sorry

end NUMINAMATH_CALUDE_distance_from_center_l3261_326106


namespace NUMINAMATH_CALUDE_negation_equivalence_l3261_326154

theorem negation_equivalence (x y : ℝ) : 
  ¬(x^2 + y^2 = 0 → x = 0 ∧ y = 0) ↔ (x^2 + y^2 ≠ 0 → ¬(x = 0 ∧ y = 0)) :=
by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l3261_326154


namespace NUMINAMATH_CALUDE_first_term_is_two_l3261_326171

/-- An arithmetic sequence with the given properties -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  monotone_increasing : ∀ n, a n < a (n + 1)
  is_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)
  sum_first_three : a 1 + a 2 + a 3 = 12
  product_first_three : a 1 * a 2 * a 3 = 48

/-- The first term of the arithmetic sequence is 2 -/
theorem first_term_is_two (seq : ArithmeticSequence) : seq.a 1 = 2 := by
  sorry

end NUMINAMATH_CALUDE_first_term_is_two_l3261_326171


namespace NUMINAMATH_CALUDE_intercept_sum_lower_bound_l3261_326167

/-- A line passing through (1,3) intersecting positive x and y axes -/
structure InterceptLine where
  a : ℝ
  b : ℝ
  a_pos : 0 < a
  b_pos : 0 < b
  passes_through_P : 1 / a + 3 / b = 1

/-- The sum of intercepts is at least 4 + 2√3 -/
theorem intercept_sum_lower_bound (l : InterceptLine) : l.a + l.b ≥ 4 + 2 * Real.sqrt 3 := by
  sorry

#check intercept_sum_lower_bound

end NUMINAMATH_CALUDE_intercept_sum_lower_bound_l3261_326167


namespace NUMINAMATH_CALUDE_factor_of_polynomial_l3261_326133

theorem factor_of_polynomial (x : ℝ) : 
  ∃ (q : ℝ → ℝ), (x^6 + 8 : ℝ) = (x^2 + 2) * q x := by
sorry

end NUMINAMATH_CALUDE_factor_of_polynomial_l3261_326133


namespace NUMINAMATH_CALUDE_no_convincing_statement_when_guilty_l3261_326143

/-- Represents a statement made in court -/
def Statement : Type := String

/-- Represents the state of being guilty or innocent -/
inductive GuiltState
| Guilty
| Innocent

/-- Represents a jury's belief about guilt -/
inductive JuryBelief
| BelievesGuilty
| BelievesInnocent

/-- A function that models how a rational jury processes a statement -/
def rationalJuryProcess : Statement → GuiltState → JuryBelief := sorry

/-- The theorem stating that it's impossible to convince a rational jury of innocence when guilty -/
theorem no_convincing_statement_when_guilty :
  ∀ (s : Statement), rationalJuryProcess s GuiltState.Guilty ≠ JuryBelief.BelievesInnocent := by
  sorry

end NUMINAMATH_CALUDE_no_convincing_statement_when_guilty_l3261_326143


namespace NUMINAMATH_CALUDE_weight_difference_l3261_326111

theorem weight_difference (n : ℕ) (joe_weight : ℝ) (initial_avg : ℝ) (new_avg : ℝ) :
  joe_weight = 42 →
  initial_avg = 30 →
  new_avg = 31 →
  (n * initial_avg + joe_weight) / (n + 1) = new_avg →
  let total_weight := n * initial_avg + joe_weight
  let remaining_students := n - 1
  ∃ (x : ℝ), (total_weight - 2 * x) / remaining_students = initial_avg →
  |x - joe_weight| = 6 :=
by sorry

end NUMINAMATH_CALUDE_weight_difference_l3261_326111


namespace NUMINAMATH_CALUDE_right_triangle_integer_sides_l3261_326137

theorem right_triangle_integer_sides (a b c : ℕ) : 
  a^2 + b^2 = c^2 →  -- Pythagorean theorem (right triangle condition)
  Nat.gcd a (Nat.gcd b c) = 1 →  -- GCD of sides is 1
  ∃ m n : ℕ, 
    (a = 2*m*n ∧ b = m^2 - n^2 ∧ c = m^2 + n^2) ∨ 
    (b = 2*m*n ∧ a = m^2 - n^2 ∧ c = m^2 + n^2) :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_integer_sides_l3261_326137


namespace NUMINAMATH_CALUDE_quadratic_increasing_condition_l3261_326112

/-- A quadratic function of the form y = x^2 + (1-m)x + 1 -/
def quadratic_function (m : ℝ) (x : ℝ) : ℝ := x^2 + (1-m)*x + 1

/-- The derivative of the quadratic function -/
def quadratic_derivative (m : ℝ) (x : ℝ) : ℝ := 2*x + (1-m)

theorem quadratic_increasing_condition (m : ℝ) :
  (∀ x > 1, quadratic_derivative m x > 0) ↔ m ≤ 3 :=
sorry

end NUMINAMATH_CALUDE_quadratic_increasing_condition_l3261_326112


namespace NUMINAMATH_CALUDE_negation_of_forall_exp_positive_l3261_326135

theorem negation_of_forall_exp_positive :
  (¬ ∀ x : ℝ, Real.exp x > 0) ↔ (∃ x : ℝ, Real.exp x ≤ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_forall_exp_positive_l3261_326135


namespace NUMINAMATH_CALUDE_food_expense_percentage_l3261_326138

/-- Proves that the percentage of salary spent on food is 32% given the specified conditions --/
theorem food_expense_percentage
  (salary : ℝ)
  (medicine_percentage : ℝ)
  (savings_percentage : ℝ)
  (savings_amount : ℝ)
  (h1 : salary = 15000)
  (h2 : medicine_percentage = 20)
  (h3 : savings_percentage = 60)
  (h4 : savings_amount = 4320)
  (h5 : savings_amount = (salary - (medicine_percentage / 100) * salary - food_expense) * (savings_percentage / 100))
  : (food_expense / salary) * 100 = 32 := by
  sorry

end NUMINAMATH_CALUDE_food_expense_percentage_l3261_326138


namespace NUMINAMATH_CALUDE_rectangle_lengths_l3261_326191

/-- Given a square and two rectangles with specific properties, prove their lengths -/
theorem rectangle_lengths (square_side : ℝ) (rect1_width rect2_width : ℝ) 
  (h1 : square_side = 6)
  (h2 : rect1_width = 4)
  (h3 : rect2_width = 3)
  (h4 : square_side * square_side = rect1_width * (square_side * square_side / rect1_width))
  (h5 : rect2_width * (square_side * square_side / (2 * rect2_width)) = square_side * square_side / 2) :
  (square_side * square_side / rect1_width, square_side * square_side / (2 * rect2_width)) = (9, 6) := by
  sorry

#check rectangle_lengths

end NUMINAMATH_CALUDE_rectangle_lengths_l3261_326191


namespace NUMINAMATH_CALUDE_movie_ticket_revenue_l3261_326107

/-- Calculates the total revenue from movie ticket sales --/
theorem movie_ticket_revenue
  (adult_price : ℕ)
  (child_price : ℕ)
  (total_tickets : ℕ)
  (child_tickets : ℕ)
  (h1 : adult_price = 7)
  (h2 : child_price = 4)
  (h3 : total_tickets = 900)
  (h4 : child_tickets = 400) :
  adult_price * (total_tickets - child_tickets) + child_price * child_tickets = 5100 :=
by sorry

end NUMINAMATH_CALUDE_movie_ticket_revenue_l3261_326107


namespace NUMINAMATH_CALUDE_cricket_team_size_l3261_326117

theorem cricket_team_size :
  ∀ n : ℕ,
  n > 0 →
  let avg_age : ℚ := 26
  let wicket_keeper_age : ℚ := avg_age + 3
  let remaining_avg_age : ℚ := avg_age - 1
  (n : ℚ) * avg_age = wicket_keeper_age + avg_age + (n - 2 : ℚ) * remaining_avg_age →
  n = 5 := by
sorry

end NUMINAMATH_CALUDE_cricket_team_size_l3261_326117


namespace NUMINAMATH_CALUDE_mango_rate_proof_l3261_326170

def grape_quantity : ℕ := 10
def grape_rate : ℕ := 70
def mango_quantity : ℕ := 9
def total_paid : ℕ := 1195

theorem mango_rate_proof :
  (total_paid - grape_quantity * grape_rate) / mango_quantity = 55 := by
  sorry

end NUMINAMATH_CALUDE_mango_rate_proof_l3261_326170


namespace NUMINAMATH_CALUDE_problem_solution_l3261_326151

theorem problem_solution (x : ℝ) 
  (h : x + Real.sqrt (x^2 - 4) + 1 / (x - Real.sqrt (x^2 - 4)) = 30) :
  x^2 + Real.sqrt (x^4 - 16) + 1 / (x^2 + Real.sqrt (x^4 - 16)) = 52441/900 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l3261_326151


namespace NUMINAMATH_CALUDE_friends_receiving_pens_correct_l3261_326147

/-- Calculate the number of friends who will receive pens --/
def friends_receiving_pens (kendra_packs tony_packs maria_packs : ℕ)
                           (kendra_pens_per_pack tony_pens_per_pack maria_pens_per_pack : ℕ)
                           (pens_kept_per_person : ℕ) : ℕ :=
  let kendra_total := kendra_packs * kendra_pens_per_pack
  let tony_total := tony_packs * tony_pens_per_pack
  let maria_total := maria_packs * maria_pens_per_pack
  let total_pens := kendra_total + tony_total + maria_total
  let total_kept := 3 * pens_kept_per_person
  total_pens - total_kept

theorem friends_receiving_pens_correct :
  friends_receiving_pens 7 5 9 4 6 5 3 = 94 := by
  sorry

end NUMINAMATH_CALUDE_friends_receiving_pens_correct_l3261_326147


namespace NUMINAMATH_CALUDE_smallest_d_for_factorization_l3261_326144

theorem smallest_d_for_factorization : 
  (∃ (p q : ℤ), x^2 + 107*x + 2050 = (x + p) * (x + q)) ∧ 
  (∀ (d : ℕ), d < 107 → ¬∃ (p q : ℤ), x^2 + d*x + 2050 = (x + p) * (x + q)) := by
  sorry

end NUMINAMATH_CALUDE_smallest_d_for_factorization_l3261_326144


namespace NUMINAMATH_CALUDE_gift_payment_l3261_326118

theorem gift_payment (a b c d : ℝ) 
  (h1 : a + b + c + d = 84)
  (h2 : a = (1/3) * (b + c + d))
  (h3 : b = (1/4) * (a + c + d))
  (h4 : c = (1/5) * (a + b + d))
  (h5 : a ≥ 0) (h6 : b ≥ 0) (h7 : c ≥ 0) (h8 : d ≥ 0) : 
  d = 40 := by
  sorry

end NUMINAMATH_CALUDE_gift_payment_l3261_326118


namespace NUMINAMATH_CALUDE_complex_sum_zero_l3261_326165

theorem complex_sum_zero (z : ℂ) (h : z = Complex.exp (6 * Real.pi * I / 11)) :
  z^2 / (1 + z^3) + z^4 / (1 + z^6) + z^5 / (1 + z^9) = 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_sum_zero_l3261_326165


namespace NUMINAMATH_CALUDE_advanced_math_group_arrangements_l3261_326142

/-- The number of students in the advanced mathematics study group -/
def total_students : ℕ := 5

/-- The number of boys in the group -/
def num_boys : ℕ := 3

/-- The number of girls in the group -/
def num_girls : ℕ := 2

/-- Student A -/
def student_A : ℕ := 1

/-- Student B -/
def student_B : ℕ := 2

/-- The number of arrangements where A and B must stand next to each other -/
def arrangements_adjacent : ℕ := 48

/-- The number of arrangements where A and B must not stand next to each other -/
def arrangements_not_adjacent : ℕ := 72

/-- The number of arrangements where A cannot stand at the far left and B cannot stand at the far right -/
def arrangements_restricted : ℕ := 78

theorem advanced_math_group_arrangements :
  (total_students = num_boys + num_girls) ∧
  (arrangements_adjacent = 48) ∧
  (arrangements_not_adjacent = 72) ∧
  (arrangements_restricted = 78) := by
  sorry

end NUMINAMATH_CALUDE_advanced_math_group_arrangements_l3261_326142


namespace NUMINAMATH_CALUDE_largest_prime_factor_of_S_l3261_326146

/-- The product of non-zero digits of a positive integer -/
def p (n : ℕ+) : ℕ :=
  sorry

/-- The sum of p(n) from 1 to 999 -/
def S : ℕ :=
  (Finset.range 999).sum (λ i => p ⟨i + 1, Nat.succ_pos i⟩)

/-- The largest prime factor of S -/
theorem largest_prime_factor_of_S :
  ∃ (q : ℕ), Nat.Prime q ∧ q ∣ S ∧ ∀ (p : ℕ), Nat.Prime p → p ∣ S → p ≤ q :=
  sorry

end NUMINAMATH_CALUDE_largest_prime_factor_of_S_l3261_326146


namespace NUMINAMATH_CALUDE_combined_average_correct_l3261_326169

-- Define the percentages for each city
def springfield : Fin 4 → ℚ
  | 0 => 12
  | 1 => 18
  | 2 => 25
  | 3 => 40

def shelbyville : Fin 4 → ℚ
  | 0 => 10
  | 1 => 15
  | 2 => 23
  | 3 => 35

-- Define the years
def years : Fin 4 → ℕ
  | 0 => 1990
  | 1 => 2000
  | 2 => 2010
  | 3 => 2020

-- Define the combined average function
def combinedAverage (i : Fin 4) : ℚ :=
  (springfield i + shelbyville i) / 2

-- Theorem statement
theorem combined_average_correct :
  (combinedAverage 0 = 11) ∧
  (combinedAverage 1 = 33/2) ∧
  (combinedAverage 2 = 24) ∧
  (combinedAverage 3 = 75/2) := by
  sorry

end NUMINAMATH_CALUDE_combined_average_correct_l3261_326169


namespace NUMINAMATH_CALUDE_equation_solution_l3261_326103

theorem equation_solution :
  ∃ x : ℝ, x = 4/3 ∧ 
    (3*x^2)/(x-2) - (3*x + 8)/4 + (6 - 9*x)/(x-2) + 2 = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3261_326103


namespace NUMINAMATH_CALUDE_a_zero_necessary_not_sufficient_l3261_326180

def is_pure_imaginary (z : ℂ) : Prop := z.re = 0

theorem a_zero_necessary_not_sufficient :
  ∃ (a b : ℝ), (is_pure_imaginary (a + b * I) → a = 0) ∧
  ¬(a = 0 → is_pure_imaginary (a + b * I)) :=
sorry

end NUMINAMATH_CALUDE_a_zero_necessary_not_sufficient_l3261_326180


namespace NUMINAMATH_CALUDE_condition_for_proposition_l3261_326176

def A : Set ℝ := {x | 1 ≤ x ∧ x ≤ 2}

theorem condition_for_proposition (a : ℝ) :
  (∀ x ∈ A, x^2 - a ≤ 0) ↔ a ≥ 4 := by sorry

end NUMINAMATH_CALUDE_condition_for_proposition_l3261_326176


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l3261_326193

theorem complex_fraction_simplification (x y : ℚ) (hx : x = 3) (hy : y = 4) :
  (x / (y + 1)) / (y / (x + 2)) = 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l3261_326193


namespace NUMINAMATH_CALUDE_smallest_x_for_inequality_l3261_326123

theorem smallest_x_for_inequality : ∃ x : ℕ, (∀ y : ℕ, 27^y > 3^24 → x ≤ y) ∧ 27^x > 3^24 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_smallest_x_for_inequality_l3261_326123


namespace NUMINAMATH_CALUDE_rectangle_area_l3261_326150

/-- A rectangle with length twice its width and perimeter 84 cm has an area of 392 cm² -/
theorem rectangle_area (width : ℝ) (length : ℝ) (perimeter : ℝ) (area : ℝ) : 
  length = 2 * width →
  perimeter = 84 →
  perimeter = 2 * (length + width) →
  area = length * width →
  area = 392 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l3261_326150


namespace NUMINAMATH_CALUDE_polynomial_division_theorem_l3261_326177

theorem polynomial_division_theorem (x : ℝ) : 
  (x^5 + x^4 + x^3 + x^2 + x + 1) * (x - 1) + 9 = x^6 + 8 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_division_theorem_l3261_326177


namespace NUMINAMATH_CALUDE_magnitude_of_complex_power_l3261_326196

theorem magnitude_of_complex_power (z : ℂ) : 
  z = 4 + 2 * Complex.I * Real.sqrt 5 → Complex.abs (z^4) = 1296 := by
  sorry

end NUMINAMATH_CALUDE_magnitude_of_complex_power_l3261_326196


namespace NUMINAMATH_CALUDE_ceiling_2023_ceiling_quadratic_inequality_ceiling_equality_distance_l3261_326128

-- Define the ceiling function
noncomputable def ceiling (x : ℝ) : ℤ :=
  Int.ceil x

-- Theorem 1
theorem ceiling_2023 (x : ℝ) :
  ceiling x = 2023 → x ∈ Set.Ioo 2022 2023 := by sorry

-- Theorem 2
theorem ceiling_quadratic_inequality (x : ℝ) :
  (ceiling x)^2 - 5*(ceiling x) + 6 ≤ 0 → x ∈ Set.Ioo 1 3 := by sorry

-- Theorem 3
theorem ceiling_equality_distance (x y : ℝ) :
  ceiling x = ceiling y → |x - y| < 1 := by sorry

end NUMINAMATH_CALUDE_ceiling_2023_ceiling_quadratic_inequality_ceiling_equality_distance_l3261_326128


namespace NUMINAMATH_CALUDE_initial_coins_l3261_326182

/-- Given a box of coins, prove that the initial number of coins is 21 when 8 coins are added and the total becomes 29. -/
theorem initial_coins (initial_coins added_coins total_coins : ℕ) 
  (h1 : added_coins = 8)
  (h2 : total_coins = 29)
  (h3 : initial_coins + added_coins = total_coins) : 
  initial_coins = 21 := by
sorry

end NUMINAMATH_CALUDE_initial_coins_l3261_326182


namespace NUMINAMATH_CALUDE_parabola_equation_l3261_326155

/-- A parabola is defined by its directrix and focus. -/
structure Parabola where
  directrix : ℝ  -- y-coordinate of the directrix
  focus : ℝ      -- y-coordinate of the focus

/-- The standard equation of a parabola. -/
def standardEquation (p : Parabola) : Prop :=
  ∀ x y : ℝ, (x^2 = -4 * p.directrix * y) ↔ (y = p.directrix ∨ (x^2 + (y - p.focus)^2 = (y - p.directrix)^2))

/-- Theorem: For a parabola with directrix y = 4, its standard equation is x² = -16y. -/
theorem parabola_equation (p : Parabola) (h : p.directrix = 4) : 
  standardEquation p ↔ ∀ x y : ℝ, x^2 = -16*y ↔ (y = 4 ∨ (x^2 + (y - p.focus)^2 = (y - 4)^2)) :=
sorry

end NUMINAMATH_CALUDE_parabola_equation_l3261_326155


namespace NUMINAMATH_CALUDE_ellipse_major_axis_length_l3261_326100

/-- Definition of the ellipse -/
def is_ellipse (x y : ℝ) : Prop := x^2 / 4 + y^2 / 9 = 1

/-- Definition of the major axis length -/
def major_axis_length (f : ℝ → ℝ → Prop) : ℝ := sorry

/-- Theorem: The length of the major axis of the ellipse is 6 -/
theorem ellipse_major_axis_length :
  major_axis_length is_ellipse = 6 := by sorry

end NUMINAMATH_CALUDE_ellipse_major_axis_length_l3261_326100


namespace NUMINAMATH_CALUDE_largest_common_measure_l3261_326152

theorem largest_common_measure (segment1 segment2 : ℕ) 
  (h1 : segment1 = 15) (h2 : segment2 = 12) : 
  ∃ (m : ℕ), m > 0 ∧ m ∣ segment1 ∧ m ∣ segment2 ∧ 
  ∀ (n : ℕ), n > m → (n ∣ segment1 ∧ n ∣ segment2) → False :=
by sorry

end NUMINAMATH_CALUDE_largest_common_measure_l3261_326152


namespace NUMINAMATH_CALUDE_unique_n_for_prime_sequence_l3261_326119

theorem unique_n_for_prime_sequence : ∃! (n : ℕ), 
  n > 0 ∧ 
  Nat.Prime (n + 1) ∧ 
  Nat.Prime (n + 3) ∧ 
  Nat.Prime (n + 7) ∧ 
  Nat.Prime (n + 9) ∧ 
  Nat.Prime (n + 13) ∧ 
  Nat.Prime (n + 15) :=
by sorry

end NUMINAMATH_CALUDE_unique_n_for_prime_sequence_l3261_326119


namespace NUMINAMATH_CALUDE_smallest_sum_with_conditions_l3261_326162

theorem smallest_sum_with_conditions (a b : ℕ+) 
  (h1 : Nat.gcd (a + b) 330 = 1)
  (h2 : ∃ k : ℕ, a^(a:ℕ) = k * b^(b:ℕ))
  (h3 : ¬∃ m : ℕ, a = m * b) :
  ∀ (x y : ℕ+), 
    (Nat.gcd (x + y) 330 = 1) → 
    (∃ k : ℕ, x^(x:ℕ) = k * y^(y:ℕ)) → 
    (¬∃ m : ℕ, x = m * y) → 
    (a + b : ℕ) ≤ (x + y : ℕ) ∧ 
    (a + b : ℕ) = 507 := by
  sorry

end NUMINAMATH_CALUDE_smallest_sum_with_conditions_l3261_326162


namespace NUMINAMATH_CALUDE_polygon_sides_count_l3261_326197

-- Define the number of sides of the polygon
variable (n : ℕ)

-- Define the sum of interior angles
def sum_interior_angles (n : ℕ) : ℝ := (n - 2) * 180

-- Define the sum of exterior angles (always 360°)
def sum_exterior_angles : ℝ := 360

-- State the theorem
theorem polygon_sides_count :
  sum_interior_angles n = sum_exterior_angles + 720 → n = 8 := by
  sorry

end NUMINAMATH_CALUDE_polygon_sides_count_l3261_326197


namespace NUMINAMATH_CALUDE_selection_ways_l3261_326108

def boys : ℕ := 5
def girls : ℕ := 3
def total_subjects : ℕ := 5

def choose (n k : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))
def permute (n k : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial (n - k))

theorem selection_ways : 
  choose (boys + girls - 2) (total_subjects - 2) * choose (total_subjects - 2) 1 * permute (total_subjects - 2) (total_subjects - 2) = 360 := by
  sorry

end NUMINAMATH_CALUDE_selection_ways_l3261_326108


namespace NUMINAMATH_CALUDE_college_board_committee_count_l3261_326149

/-- Represents a college board. -/
structure Board :=
  (total_members : ℕ)
  (professors : ℕ)
  (non_professors : ℕ)
  (h_total : total_members = professors + non_professors)

/-- Represents a committee formed from the board. -/
structure Committee :=
  (size : ℕ)
  (min_professors : ℕ)

/-- Calculates the number of valid committees for a given board and committee requirements. -/
def count_valid_committees (board : Board) (committee : Committee) : ℕ :=
  sorry

/-- The specific board in the problem. -/
def college_board : Board :=
  { total_members := 15
  , professors := 7
  , non_professors := 8
  , h_total := by rfl }

/-- The specific committee requirements in the problem. -/
def required_committee : Committee :=
  { size := 5
  , min_professors := 2 }

theorem college_board_committee_count :
  count_valid_committees college_board required_committee = 2457 :=
sorry

end NUMINAMATH_CALUDE_college_board_committee_count_l3261_326149


namespace NUMINAMATH_CALUDE_pepperoni_pizzas_sold_l3261_326187

/-- A pizza type -/
inductive PizzaType
  | Pepperoni
  | Bacon
  | Cheese

/-- Representation of pizza sales -/
structure PizzaSales where
  pepperoni : ℕ
  bacon : ℕ
  cheese : ℕ

/-- The theorem stating the number of pepperoni pizzas sold -/
theorem pepperoni_pizzas_sold (sales : PizzaSales) :
  sales.bacon = 6 →
  sales.cheese = 6 →
  sales.pepperoni + sales.bacon + sales.cheese = 14 →
  sales.pepperoni = 2 := by
  sorry

end NUMINAMATH_CALUDE_pepperoni_pizzas_sold_l3261_326187


namespace NUMINAMATH_CALUDE_abc_inequality_l3261_326175

theorem abc_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h : a^(3/2) + b^(3/2) + c^(3/2) = 1) :
  abc ≤ 1/9 ∧ a/(b+c) + b/(a+c) + c/(a+b) ≤ 1/(2 * Real.sqrt (abc)) := by
  sorry

end NUMINAMATH_CALUDE_abc_inequality_l3261_326175


namespace NUMINAMATH_CALUDE_complex_simplification_l3261_326134

/-- The imaginary unit -/
def i : ℂ := Complex.I

/-- Theorem stating that the given complex expression simplifies to 5 -/
theorem complex_simplification : 2 * (3 - i) + i * (2 + i) = 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_simplification_l3261_326134


namespace NUMINAMATH_CALUDE_percentage_markup_proof_l3261_326168

def selling_price : ℚ := 8587
def cost_price : ℚ := 6925

theorem percentage_markup_proof :
  let markup := selling_price - cost_price
  let percentage_markup := (markup / cost_price) * 100
  ∃ ε > 0, abs (percentage_markup - 23.99) < ε := by
sorry

end NUMINAMATH_CALUDE_percentage_markup_proof_l3261_326168


namespace NUMINAMATH_CALUDE_A_eq_union_l3261_326160

/-- The set of real numbers a > 0 such that either y = a^x is not monotonically increasing on R
    or ax^2 - ax + 1 > 0 does not hold for all x ∈ R, but at least one of these conditions is true. -/
def A : Set ℝ :=
  {a : ℝ | a > 0 ∧
    (¬(∀ x y : ℝ, x < y → a^x < a^y) ∨ ¬(∀ x : ℝ, a*x^2 - a*x + 1 > 0)) ∧
    ((∀ x y : ℝ, x < y → a^x < a^y) ∨ (∀ x : ℝ, a*x^2 - a*x + 1 > 0))}

/-- The theorem stating that A is equal to the interval (0,1] union [4,+∞) -/
theorem A_eq_union : A = Set.Ioo 0 1 ∪ Set.Ici 4 := by sorry

end NUMINAMATH_CALUDE_A_eq_union_l3261_326160


namespace NUMINAMATH_CALUDE_student_average_greater_than_true_average_l3261_326120

theorem student_average_greater_than_true_average 
  (u v w x y : ℝ) 
  (h : u ≤ v ∧ v ≤ w ∧ w ≤ x ∧ x ≤ y) : 
  ((u + v + w) / 3 + x + y) / 3 > (u + v + w + x + y) / 5 := by
  sorry

end NUMINAMATH_CALUDE_student_average_greater_than_true_average_l3261_326120


namespace NUMINAMATH_CALUDE_range_of_f_l3261_326104

def f (x : ℝ) := x^2 + 4*x + 6

theorem range_of_f : 
  ∀ y ∈ Set.Icc 2 6, ∃ x ∈ Set.Ico (-3) 0, f x = y ∧
  ∀ x ∈ Set.Ico (-3) 0, 2 ≤ f x ∧ f x < 6 :=
sorry

end NUMINAMATH_CALUDE_range_of_f_l3261_326104


namespace NUMINAMATH_CALUDE_haley_concert_spending_l3261_326174

/-- Calculates the total cost of concert tickets based on a pricing structure -/
def calculate_total_cost (initial_price : ℕ) (discounted_price : ℕ) (initial_quantity : ℕ) (discounted_quantity : ℕ) : ℕ :=
  initial_price * initial_quantity + discounted_price * discounted_quantity

/-- Proves that Haley's total spending on concert tickets is $27 -/
theorem haley_concert_spending :
  let initial_price : ℕ := 4
  let discounted_price : ℕ := 3
  let initial_quantity : ℕ := 3
  let discounted_quantity : ℕ := 5
  calculate_total_cost initial_price discounted_price initial_quantity discounted_quantity = 27 :=
by
  sorry

#eval calculate_total_cost 4 3 3 5

end NUMINAMATH_CALUDE_haley_concert_spending_l3261_326174


namespace NUMINAMATH_CALUDE_rectangle_max_area_l3261_326148

/-- Given a rectangle with perimeter 60, its maximum possible area is 225 -/
theorem rectangle_max_area :
  ∀ x y : ℝ,
  x > 0 → y > 0 →
  2 * x + 2 * y = 60 →
  x * y ≤ 225 :=
by
  sorry

end NUMINAMATH_CALUDE_rectangle_max_area_l3261_326148


namespace NUMINAMATH_CALUDE_eight_elevenths_rounded_l3261_326124

/-- Rounds a rational number to the specified number of decimal places -/
def round_to_decimal_places (q : ℚ) (places : ℕ) : ℚ :=
  (⌊q * 10^places + 1/2⌋ : ℚ) / 10^places

/-- Proves that 8/11 rounded to 3 decimal places is equal to 0.727 -/
theorem eight_elevenths_rounded : round_to_decimal_places (8/11) 3 = 727/1000 := by
  sorry

end NUMINAMATH_CALUDE_eight_elevenths_rounded_l3261_326124


namespace NUMINAMATH_CALUDE_amy_bob_games_l3261_326121

/-- Represents the total number of players -/
def total_players : ℕ := 12

/-- Represents the number of players in each game -/
def players_per_game : ℕ := 6

/-- Represents the number of players that are always together (Chris and Dave) -/
def always_together : ℕ := 2

/-- Represents the number of specific players we're interested in (Amy and Bob) -/
def specific_players : ℕ := 2

/-- Theorem stating that the number of games where Amy and Bob play together
    is equal to the number of ways to choose 2 players from the remaining 8 players -/
theorem amy_bob_games :
  (total_players - specific_players - always_together).choose 2 =
  Nat.choose 8 2 := by sorry

end NUMINAMATH_CALUDE_amy_bob_games_l3261_326121


namespace NUMINAMATH_CALUDE_strawberry_milk_probability_l3261_326184

theorem strawberry_milk_probability : 
  let n : ℕ := 7  -- number of trials
  let k : ℕ := 5  -- number of successes
  let p : ℚ := 3/4  -- probability of success in each trial
  Nat.choose n k * p^k * (1-p)^(n-k) = 5103/16384 := by
  sorry

end NUMINAMATH_CALUDE_strawberry_milk_probability_l3261_326184


namespace NUMINAMATH_CALUDE_algebraic_simplification_l3261_326126

theorem algebraic_simplification (x y : ℝ) :
  (3 * x - 2 * y - 4) * (x + y + 5) - (x + 2 * y + 5) * (3 * x - y - 1) =
  -4 * x * y - 3 * x - 7 * y - 15 := by
  sorry

end NUMINAMATH_CALUDE_algebraic_simplification_l3261_326126


namespace NUMINAMATH_CALUDE_unique_prime_product_sum_l3261_326194

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

def distinct_primes (p q r : ℕ) : Prop := is_prime p ∧ is_prime q ∧ is_prime r ∧ p ≠ q ∧ p ≠ r ∧ q ≠ r

theorem unique_prime_product_sum (p q r : ℕ) : 
  5401 = p * q * r → 
  distinct_primes p q r →
  ∃! n : ℕ, ∃ p1 p2 p3 : ℕ, 
    n = p1 * p2 * p3 ∧ 
    distinct_primes p1 p2 p3 ∧ 
    p1 + p2 + p3 = p + q + r ∧
    n ≠ 5401 :=
sorry

end NUMINAMATH_CALUDE_unique_prime_product_sum_l3261_326194


namespace NUMINAMATH_CALUDE_tan_sum_simplification_l3261_326192

theorem tan_sum_simplification :
  (∀ x y, Real.tan (x + y) = (Real.tan x + Real.tan y) / (1 - Real.tan x * Real.tan y)) →
  Real.tan (45 * π / 180) = 1 →
  (1 + Real.tan (10 * π / 180)) * (1 + Real.tan (35 * π / 180)) = 2 := by
  sorry

end NUMINAMATH_CALUDE_tan_sum_simplification_l3261_326192


namespace NUMINAMATH_CALUDE_change_received_l3261_326101

/-- The change received when buying gum and a protractor -/
theorem change_received (gum_cost protractor_cost amount_paid : ℕ) : 
  gum_cost = 350 → protractor_cost = 500 → amount_paid = 1000 → 
  amount_paid - (gum_cost + protractor_cost) = 150 := by
  sorry

end NUMINAMATH_CALUDE_change_received_l3261_326101


namespace NUMINAMATH_CALUDE_select_parents_count_l3261_326131

/-- The number of ways to select 4 parents out of 12 (6 couples), 
    such that exactly one pair of the chosen 4 are a couple -/
def selectParents : ℕ := sorry

/-- The total number of couples -/
def totalCouples : ℕ := 6

/-- The total number of parents -/
def totalParents : ℕ := 12

/-- The number of parents to be selected -/
def parentsToSelect : ℕ := 4

theorem select_parents_count : 
  selectParents = 240 := by sorry

end NUMINAMATH_CALUDE_select_parents_count_l3261_326131


namespace NUMINAMATH_CALUDE_imaginary_part_of_one_minus_i_squared_l3261_326164

theorem imaginary_part_of_one_minus_i_squared :
  Complex.im ((1 - Complex.I) ^ 2) = -2 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_one_minus_i_squared_l3261_326164


namespace NUMINAMATH_CALUDE_dave_lost_tickets_l3261_326178

/-- Prove that Dave lost 2 tickets at the arcade -/
theorem dave_lost_tickets (initial_tickets : ℕ) (spent_tickets : ℕ) (remaining_tickets : ℕ) 
  (h1 : initial_tickets = 14)
  (h2 : spent_tickets = 10)
  (h3 : remaining_tickets = 2) :
  initial_tickets - (spent_tickets + remaining_tickets) = 2 := by
  sorry

end NUMINAMATH_CALUDE_dave_lost_tickets_l3261_326178


namespace NUMINAMATH_CALUDE_shaded_area_square_with_quarter_circles_l3261_326145

/-- The area of the shaded region inside a square with side length 16 cm but outside
    four quarter circles with radius 6 cm at each corner is 256 - 36π cm². -/
theorem shaded_area_square_with_quarter_circles (π : ℝ) :
  let square_side : ℝ := 16
  let circle_radius : ℝ := 6
  let square_area : ℝ := square_side ^ 2
  let quarter_circle_area : ℝ := π * circle_radius ^ 2 / 4
  let total_quarter_circles_area : ℝ := 4 * quarter_circle_area
  let shaded_area : ℝ := square_area - total_quarter_circles_area
  shaded_area = 256 - 36 * π :=
by sorry

end NUMINAMATH_CALUDE_shaded_area_square_with_quarter_circles_l3261_326145


namespace NUMINAMATH_CALUDE_max_rectangle_area_l3261_326179

/-- The equation that the vertices' coordinates must satisfy -/
def vertex_equation (x y : ℝ) : Prop :=
  |y + 1| * (y^2 + 2*y + 28) + |x - 2| = 9 * (y^2 + 2*y + 4)

/-- The area function of the rectangle -/
def rectangle_area (x : ℝ) : ℝ :=
  -4 * x * (x - 3)^3

/-- The theorem stating the maximum area of the rectangle -/
theorem max_rectangle_area :
  ∃ (x y : ℝ), vertex_equation x y ∧
  ∀ (x' y' : ℝ), vertex_equation x' y' → rectangle_area x ≥ rectangle_area x' ∧
  rectangle_area x = 34.171875 := by
sorry

end NUMINAMATH_CALUDE_max_rectangle_area_l3261_326179


namespace NUMINAMATH_CALUDE_burger_cost_l3261_326141

theorem burger_cost (burger soda : ℕ) 
  (alice_purchase : 4 * burger + 3 * soda = 440)
  (bob_purchase : 3 * burger + 2 * soda = 330) :
  burger = 110 := by
sorry

end NUMINAMATH_CALUDE_burger_cost_l3261_326141


namespace NUMINAMATH_CALUDE_arithmetic_square_root_of_16_l3261_326125

theorem arithmetic_square_root_of_16 : 
  ∃ (x : ℝ), x ≥ 0 ∧ x ^ 2 = 16 ∧ x = 4 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_square_root_of_16_l3261_326125


namespace NUMINAMATH_CALUDE_arctan_of_tan_difference_l3261_326130

theorem arctan_of_tan_difference (θ : Real) : 
  θ ∈ Set.Icc 0 180 → 
  Real.arctan (Real.tan (80 * π / 180) - 3 * Real.tan (30 * π / 180)) = 50 * π / 180 := by
  sorry

end NUMINAMATH_CALUDE_arctan_of_tan_difference_l3261_326130


namespace NUMINAMATH_CALUDE_fibonacci_type_sequence_count_l3261_326186

/-- A Fibonacci-type sequence is an infinite sequence of integers where each term is the sum of the two preceding ones. -/
def FibonacciTypeSequence (a : ℤ → ℤ) : Prop :=
  ∀ n : ℤ, a n = a (n - 1) + a (n - 2)

/-- Count of Fibonacci-type sequences with two consecutive terms strictly positive and ≤ N -/
def countFibonacciTypeSequences (N : ℕ) : ℕ :=
  if N % 2 = 0 then
    (N / 2) * (N / 2 + 1)
  else
    ((N + 1) / 2) ^ 2

theorem fibonacci_type_sequence_count (N : ℕ) :
  (∃ a : ℤ → ℤ, FibonacciTypeSequence a ∧
    ∃ n : ℤ, 0 < a n ∧ 0 < a (n + 1) ∧ a n ≤ N ∧ a (n + 1) ≤ N) →
  countFibonacciTypeSequences N = 
    if N % 2 = 0 then
      (N / 2) * (N / 2 + 1)
    else
      ((N + 1) / 2) ^ 2 :=
by sorry

#check fibonacci_type_sequence_count

end NUMINAMATH_CALUDE_fibonacci_type_sequence_count_l3261_326186


namespace NUMINAMATH_CALUDE_total_pictures_l3261_326113

theorem total_pictures (randy peter quincy : ℕ) : 
  randy = 5 → 
  peter = randy + 3 → 
  quincy = peter + 20 → 
  randy + peter + quincy = 41 := by sorry

end NUMINAMATH_CALUDE_total_pictures_l3261_326113


namespace NUMINAMATH_CALUDE_apples_remaining_l3261_326102

/-- Calculates the number of apples remaining on a tree after three days of picking -/
theorem apples_remaining (total : ℕ) (day1_fraction : ℚ) (day2_multiplier : ℕ) (day3_addition : ℕ) : 
  total = 200 →
  day1_fraction = 1 / 5 →
  day2_multiplier = 2 →
  day3_addition = 20 →
  total - (total * day1_fraction).floor - day2_multiplier * (total * day1_fraction).floor - ((total * day1_fraction).floor + day3_addition) = 20 := by
sorry

end NUMINAMATH_CALUDE_apples_remaining_l3261_326102


namespace NUMINAMATH_CALUDE_harry_cookies_per_batch_l3261_326132

/-- Calculates the number of cookies in a batch given the total chips, number of batches, and chips per cookie. -/
def cookies_per_batch (total_chips : ℕ) (num_batches : ℕ) (chips_per_cookie : ℕ) : ℕ :=
  (total_chips / num_batches) / chips_per_cookie

/-- Proves that the number of cookies in a batch is 3 given the specified conditions. -/
theorem harry_cookies_per_batch :
  cookies_per_batch 81 3 9 = 3 := by
  sorry

end NUMINAMATH_CALUDE_harry_cookies_per_batch_l3261_326132


namespace NUMINAMATH_CALUDE_cylinder_minus_cones_volume_l3261_326136

/-- The volume of a cylinder minus the volume of three congruent cones -/
theorem cylinder_minus_cones_volume (r h : ℝ) (hr : r = 8) (hh : h = 24) :
  π * r^2 * h - 3 * (1/3 * π * r^2 * (h/3)) = 1024 * π := by
  sorry

end NUMINAMATH_CALUDE_cylinder_minus_cones_volume_l3261_326136


namespace NUMINAMATH_CALUDE_uncle_height_difference_l3261_326115

/-- Given James was initially 2/3 as tall as his uncle who is 72 inches tall,
    and James grew 10 inches, prove that his uncle is now 14 inches taller than James. -/
theorem uncle_height_difference (james_initial_ratio : ℚ) (uncle_height : ℕ) (james_growth : ℕ) :
  james_initial_ratio = 2 / 3 →
  uncle_height = 72 →
  james_growth = 10 →
  uncle_height - (james_initial_ratio * uncle_height + james_growth) = 14 :=
by
  sorry

end NUMINAMATH_CALUDE_uncle_height_difference_l3261_326115


namespace NUMINAMATH_CALUDE_largest_x_abs_equation_l3261_326163

theorem largest_x_abs_equation : ∃ (x : ℝ), x = 7 ∧ |x + 3| = 10 ∧ ∀ y : ℝ, |y + 3| = 10 → y ≤ x := by
  sorry

end NUMINAMATH_CALUDE_largest_x_abs_equation_l3261_326163


namespace NUMINAMATH_CALUDE_rectangle_triangle_ef_length_l3261_326105

/-- Given a rectangle ABCD with side lengths AB and BC, and a triangle DEF inside it
    where DE = DF and the area of DEF is one-third of ABCD's area,
    prove that EF has length 12 when AB = 9 and BC = 12. -/
theorem rectangle_triangle_ef_length
  (AB BC : ℝ)
  (DE DF EF : ℝ)
  (h_ab : AB = 9)
  (h_bc : BC = 12)
  (h_de_df : DE = DF)
  (h_area : (1/2) * DE * DF = (1/3) * AB * BC) :
  EF = 12 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_triangle_ef_length_l3261_326105


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l3261_326158

theorem quadratic_equation_solution : ∃ x₁ x₂ : ℝ, 
  (x₁ = 3 ∧ x₂ = -5) ∧ 
  (x₁^2 + 2*x₁ - 15 = 0) ∧ 
  (x₂^2 + 2*x₂ - 15 = 0) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l3261_326158


namespace NUMINAMATH_CALUDE_two_digit_powers_of_three_l3261_326139

theorem two_digit_powers_of_three :
  ∃! (s : Finset ℕ), (∀ n ∈ s, 10 ≤ 3^n ∧ 3^n ≤ 99) ∧ s.card = 2 := by
  sorry

end NUMINAMATH_CALUDE_two_digit_powers_of_three_l3261_326139


namespace NUMINAMATH_CALUDE_odd_function_property_l3261_326198

/-- A function f is odd if f(-x) = -f(x) for all x -/
def IsOdd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

theorem odd_function_property (f : ℝ → ℝ) (h1 : IsOdd f) (h2 : f (-3) = -2) :
  f 3 + f 0 = 2 := by
  sorry

end NUMINAMATH_CALUDE_odd_function_property_l3261_326198


namespace NUMINAMATH_CALUDE_completing_square_quadratic_l3261_326189

theorem completing_square_quadratic (x : ℝ) :
  (x^2 - 4*x - 2 = 0) ↔ ((x - 2)^2 = 6) :=
sorry

end NUMINAMATH_CALUDE_completing_square_quadratic_l3261_326189


namespace NUMINAMATH_CALUDE_max_c_magnitude_l3261_326116

theorem max_c_magnitude (a b c : ℝ × ℝ) : 
  (‖a‖ = 1) → 
  (‖b‖ = 1) → 
  (a • b = 1/2) → 
  (‖a - b + c‖ ≤ 1) →
  (∃ (c : ℝ × ℝ), ‖c‖ = 2) ∧ 
  (∀ (c : ℝ × ℝ), ‖a - b + c‖ ≤ 1 → ‖c‖ ≤ 2) :=
by sorry

end NUMINAMATH_CALUDE_max_c_magnitude_l3261_326116


namespace NUMINAMATH_CALUDE_topmost_triangle_values_all_possible_values_achievable_l3261_326188

/-- Represents the six white triangles in the diagram -/
structure WhiteTriangles :=
  (w₁ w₂ w₃ w₄ w₅ w₆ : ℤ)

/-- Checks if the sum of three integers is divisible by 5 -/
def sumDivisibleBy5 (a b c : ℤ) : Prop :=
  (a + b + c) % 5 = 0

/-- Represents the conditions given in the problem -/
def validConfiguration (wt : WhiteTriangles) : Prop :=
  wt.w₁ = 12 ∧
  wt.w₃ = 3 ∧
  sumDivisibleBy5 wt.w₁ wt.w₂ wt.w₄ ∧
  sumDivisibleBy5 wt.w₂ wt.w₃ wt.w₅ ∧
  sumDivisibleBy5 wt.w₄ wt.w₅ wt.w₆

/-- The set of possible values for the topmost white triangle -/
def possibleTopValues : Set ℤ :=
  {0, 1, 2, 3, 4}

/-- Theorem stating that the topmost white triangle can only contain values from the set {0, 1, 2, 3, 4} -/
theorem topmost_triangle_values (wt : WhiteTriangles) :
  validConfiguration wt → wt.w₆ ∈ possibleTopValues :=
by
  sorry

/-- Theorem stating that all values in the set {0, 1, 2, 3, 4} are possible for the topmost white triangle -/
theorem all_possible_values_achievable :
  ∀ n ∈ possibleTopValues, ∃ wt : WhiteTriangles, validConfiguration wt ∧ wt.w₆ = n :=
by
  sorry

end NUMINAMATH_CALUDE_topmost_triangle_values_all_possible_values_achievable_l3261_326188


namespace NUMINAMATH_CALUDE_distinct_roots_iff_m_lt_half_m_value_when_inverse_sum_neg_two_l3261_326127

/-- Given a quadratic equation x^2 - 2(m-1)x + m^2 = 0 -/
def quadratic_equation (m x : ℝ) : Prop :=
  x^2 - 2*(m-1)*x + m^2 = 0

/-- The discriminant of the quadratic equation -/
def discriminant (m : ℝ) : ℝ :=
  (-2*(m-1))^2 - 4*m^2

theorem distinct_roots_iff_m_lt_half (m : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ quadratic_equation m x₁ ∧ quadratic_equation m x₂) ↔ m < 1/2 :=
sorry

theorem m_value_when_inverse_sum_neg_two (m : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ quadratic_equation m x₁ ∧ quadratic_equation m x₂ ∧ 1/x₁ + 1/x₂ = -2) →
  m = (-1 - Real.sqrt 5) / 2 :=
sorry

end NUMINAMATH_CALUDE_distinct_roots_iff_m_lt_half_m_value_when_inverse_sum_neg_two_l3261_326127


namespace NUMINAMATH_CALUDE_g_values_l3261_326173

-- Define the function g
def g (x : ℝ) : ℝ := -2 * x^2 - 3 * x + 1

-- State the theorem
theorem g_values : g (-1) = 2 ∧ g (-2) = -1 := by
  sorry

end NUMINAMATH_CALUDE_g_values_l3261_326173


namespace NUMINAMATH_CALUDE_math_correct_percentage_l3261_326109

/-- Represents the number of questions in the math test -/
def math_questions : ℕ := 40

/-- Represents the number of questions in the English test -/
def english_questions : ℕ := 50

/-- Represents the percentage of English questions answered correctly -/
def english_correct_percentage : ℚ := 98 / 100

/-- Represents the total number of questions answered correctly across both tests -/
def total_correct : ℕ := 79

/-- Theorem stating that the percentage of math questions answered correctly is 75% -/
theorem math_correct_percentage :
  (total_correct - (english_correct_percentage * english_questions).num) / math_questions = 75 / 100 := by
  sorry

end NUMINAMATH_CALUDE_math_correct_percentage_l3261_326109


namespace NUMINAMATH_CALUDE_wire_cut_ratio_l3261_326195

theorem wire_cut_ratio (a b : ℝ) (h : a > 0 ∧ b > 0) : 
  (a^2 / 16 = b^2 / (4 * Real.pi)) → a / b = 2 / Real.sqrt Real.pi := by
  sorry

end NUMINAMATH_CALUDE_wire_cut_ratio_l3261_326195


namespace NUMINAMATH_CALUDE_transformations_result_l3261_326183

/-- Rotates a point (x, y) by 180° counterclockwise around (2, 3) -/
def rotate180 (x y : ℝ) : ℝ × ℝ :=
  (4 - x, 6 - y)

/-- Reflects a point (x, y) about the line y = x -/
def reflectAboutYEqX (x y : ℝ) : ℝ × ℝ :=
  (y, x)

/-- Applies both transformations to a point (x, y) -/
def applyTransformations (x y : ℝ) : ℝ × ℝ :=
  let (x', y') := rotate180 x y
  reflectAboutYEqX x' y'

theorem transformations_result (c d : ℝ) :
  applyTransformations c d = (1, -4) → d - c = 7 := by
  sorry

end NUMINAMATH_CALUDE_transformations_result_l3261_326183


namespace NUMINAMATH_CALUDE_morning_speed_calculation_l3261_326114

theorem morning_speed_calculation 
  (total_time : ℝ) 
  (distance : ℝ) 
  (evening_speed : ℝ) 
  (h1 : total_time = 1) 
  (h2 : distance = 18) 
  (h3 : evening_speed = 30) : 
  ∃ morning_speed : ℝ, 
    distance / morning_speed + distance / evening_speed = total_time ∧ 
    morning_speed = 45 := by
  sorry

end NUMINAMATH_CALUDE_morning_speed_calculation_l3261_326114


namespace NUMINAMATH_CALUDE_hyperbola_real_axis_length_l3261_326129

/-- The length of the real axis of the hyperbola 2x^2 - y^2 = 8 is 4 -/
theorem hyperbola_real_axis_length :
  let hyperbola := {(x, y) : ℝ × ℝ | 2 * x^2 - y^2 = 8}
  ∃ a : ℝ, a > 0 ∧ (∀ (x y : ℝ), (x, y) ∈ hyperbola → x^2 / a^2 - y^2 / (2*a^2) = 1) ∧ 2*a = 4 :=
sorry

end NUMINAMATH_CALUDE_hyperbola_real_axis_length_l3261_326129


namespace NUMINAMATH_CALUDE_fliers_remaining_l3261_326140

theorem fliers_remaining (total : ℕ) (morning_fraction : ℚ) (afternoon_fraction : ℚ)
  (h_total : total = 2000)
  (h_morning : morning_fraction = 1 / 10)
  (h_afternoon : afternoon_fraction = 1 / 4) :
  total - (total * morning_fraction).floor - ((total - (total * morning_fraction).floor) * afternoon_fraction).floor = 1350 :=
by sorry

end NUMINAMATH_CALUDE_fliers_remaining_l3261_326140


namespace NUMINAMATH_CALUDE_distinct_roots_imply_distinct_roots_l3261_326156

theorem distinct_roots_imply_distinct_roots (p q : ℝ) 
  (hp : p > 0) 
  (hq : q > 0) 
  (h1 : (p^2 - 4*q) > 0) 
  (h2 : (q^2 - 4*p) > 0) : 
  ((p + q)^2 - 8*(p + q)) > 0 := by
sorry


end NUMINAMATH_CALUDE_distinct_roots_imply_distinct_roots_l3261_326156


namespace NUMINAMATH_CALUDE_multiples_of_five_most_representative_l3261_326172

/-- Represents a sampling method for the math test --/
inductive SamplingMethod
  | TopStudents
  | BottomStudents
  | FemaleStudents
  | MultiplesOfFive

/-- Represents a student in the seventh grade --/
structure Student where
  id : Nat
  gender : Bool  -- True for female, False for male
  score : Nat

/-- The population of students who took the test --/
def population : Finset Student := sorry

/-- The total number of students in the population --/
axiom total_students : Finset.card population = 400

/-- Defines what makes a sampling method representative --/
def is_representative (method : SamplingMethod) : Prop := sorry

/-- Theorem stating that selecting students with numbers that are multiples of 5 
    is the most representative sampling method --/
theorem multiples_of_five_most_representative : 
  is_representative SamplingMethod.MultiplesOfFive ∧ 
  ∀ m : SamplingMethod, m ≠ SamplingMethod.MultiplesOfFive → 
    ¬(is_representative m) :=
sorry

end NUMINAMATH_CALUDE_multiples_of_five_most_representative_l3261_326172


namespace NUMINAMATH_CALUDE_circle_equal_circumference_area_diameter_l3261_326122

/-- A circle with numerically equal circumference and area has a diameter of 4 -/
theorem circle_equal_circumference_area_diameter (r : ℝ) (h : r > 0) :
  π * (2 * r) = π * r^2 → 2 * r = 4 := by
  sorry

end NUMINAMATH_CALUDE_circle_equal_circumference_area_diameter_l3261_326122


namespace NUMINAMATH_CALUDE_compound_molecular_weight_l3261_326161

/-- Atomic weight in atomic mass units (amu) -/
def atomic_weight (element : String) : ℝ :=
  match element with
  | "H" => 1.008
  | "Br" => 79.904
  | "O" => 15.999
  | "C" => 12.011
  | "N" => 14.007
  | "S" => 32.065
  | _ => 0  -- Default case for unknown elements

/-- Number of atoms of each element in the compound -/
def atom_count (element : String) : ℕ :=
  match element with
  | "H" => 2
  | "Br" => 1
  | "O" => 3
  | "C" => 1
  | "N" => 1
  | "S" => 2
  | _ => 0  -- Default case for elements not in the compound

/-- Calculate the molecular weight of the compound -/
def molecular_weight : ℝ :=
  (atomic_weight "H" * atom_count "H") +
  (atomic_weight "Br" * atom_count "Br") +
  (atomic_weight "O" * atom_count "O") +
  (atomic_weight "C" * atom_count "C") +
  (atomic_weight "N" * atom_count "N") +
  (atomic_weight "S" * atom_count "S")

/-- Theorem stating that the molecular weight of the compound is 220.065 amu -/
theorem compound_molecular_weight : molecular_weight = 220.065 := by
  sorry

end NUMINAMATH_CALUDE_compound_molecular_weight_l3261_326161


namespace NUMINAMATH_CALUDE_largest_n_satisfying_property_l3261_326166

/-- A function that checks if a number is prime -/
def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → n % m ≠ 0

/-- A function that checks if a number is an odd prime -/
def isOddPrime (p : ℕ) : Prop := isPrime p ∧ p % 2 ≠ 0

/-- The property that n satisfies: for any odd prime p < n, n - p is prime -/
def satisfiesProperty (n : ℕ) : Prop :=
  ∀ p : ℕ, p < n → isOddPrime p → isPrime (n - p)

theorem largest_n_satisfying_property :
  (satisfiesProperty 10) ∧ 
  (∀ m : ℕ, m > 10 → ¬(satisfiesProperty m)) :=
sorry

end NUMINAMATH_CALUDE_largest_n_satisfying_property_l3261_326166


namespace NUMINAMATH_CALUDE_factor_polynomial_l3261_326110

theorem factor_polynomial (x : ℝ) : 60 * x^4 - 150 * x^8 = -30 * x^4 * (5 * x^4 - 2) := by
  sorry

end NUMINAMATH_CALUDE_factor_polynomial_l3261_326110


namespace NUMINAMATH_CALUDE_aubrey_garden_yield_l3261_326159

/-- Represents Aubrey's garden layout and plant yields --/
structure Garden where
  total_rows : Nat
  tomato_plants_per_row : Nat
  cucumber_plants_per_row : Nat
  bell_pepper_plants_per_row : Nat
  tomato_yield_first_last : Nat
  tomato_yield_middle : Nat
  cucumber_yield_a : Nat
  cucumber_yield_b : Nat
  bell_pepper_yield : Nat

/-- Calculates the total yield of vegetables in Aubrey's garden --/
def calculate_yield (g : Garden) : Nat × Nat × Nat :=
  let pattern_rows := 4
  let patterns := g.total_rows / pattern_rows
  let tomato_rows := patterns
  let cucumber_rows := 2 * patterns
  let bell_pepper_rows := patterns

  let tomatoes_per_row := 2 * g.tomato_yield_first_last + (g.tomato_plants_per_row - 2) * g.tomato_yield_middle
  let cucumbers_per_row := (g.cucumber_plants_per_row / 2) * (g.cucumber_yield_a + g.cucumber_yield_b)
  let bell_peppers_per_row := g.bell_pepper_plants_per_row * g.bell_pepper_yield

  let total_tomatoes := tomato_rows * tomatoes_per_row
  let total_cucumbers := cucumber_rows * cucumbers_per_row
  let total_bell_peppers := bell_pepper_rows * bell_peppers_per_row

  (total_tomatoes, total_cucumbers, total_bell_peppers)

/-- Theorem stating the total yield of Aubrey's garden --/
theorem aubrey_garden_yield (g : Garden)
  (h1 : g.total_rows = 20)
  (h2 : g.tomato_plants_per_row = 8)
  (h3 : g.cucumber_plants_per_row = 6)
  (h4 : g.bell_pepper_plants_per_row = 12)
  (h5 : g.tomato_yield_first_last = 6)
  (h6 : g.tomato_yield_middle = 4)
  (h7 : g.cucumber_yield_a = 4)
  (h8 : g.cucumber_yield_b = 5)
  (h9 : g.bell_pepper_yield = 2) :
  calculate_yield g = (180, 270, 120) := by
  sorry

#eval calculate_yield {
  total_rows := 20,
  tomato_plants_per_row := 8,
  cucumber_plants_per_row := 6,
  bell_pepper_plants_per_row := 12,
  tomato_yield_first_last := 6,
  tomato_yield_middle := 4,
  cucumber_yield_a := 4,
  cucumber_yield_b := 5,
  bell_pepper_yield := 2
}

end NUMINAMATH_CALUDE_aubrey_garden_yield_l3261_326159
