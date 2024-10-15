import Mathlib

namespace NUMINAMATH_CALUDE_odd_coefficients_equals_two_pow_binary_ones_l2983_298304

/-- The number of 1s in the binary representation of a natural number -/
def binaryOnes (n : ℕ) : ℕ := sorry

/-- The number of odd coefficients in the polynomial expansion of (1+x)^n -/
def oddCoefficients (n : ℕ) : ℕ := sorry

/-- Theorem: The number of odd coefficients in (1+x)^n is 2^d, where d is the number of 1s in n's binary representation -/
theorem odd_coefficients_equals_two_pow_binary_ones (n : ℕ) :
  oddCoefficients n = 2^(binaryOnes n) := by sorry

end NUMINAMATH_CALUDE_odd_coefficients_equals_two_pow_binary_ones_l2983_298304


namespace NUMINAMATH_CALUDE_no_increasing_function_with_properties_l2983_298357

theorem no_increasing_function_with_properties :
  ¬ ∃ (f : ℕ → ℕ),
    (∀ (a b : ℕ), a < b → f a < f b) ∧
    (f 2 = 2) ∧
    (∀ (n m : ℕ), f (n * m) = f n + f m) := by
  sorry

end NUMINAMATH_CALUDE_no_increasing_function_with_properties_l2983_298357


namespace NUMINAMATH_CALUDE_solutions_satisfy_system_l2983_298318

/-- The system of equations -/
def system (x y : ℝ) : Prop :=
  26 * x^2 + 42 * x * y + 17 * y^2 = 10 ∧
  10 * x^2 + 18 * x * y + 8 * y^2 = 6

/-- The solutions to the system of equations -/
def solutions : List (ℝ × ℝ) :=
  [(-1, 2), (-11, 14), (11, -14), (1, -2)]

/-- Theorem stating that the given points are solutions to the system -/
theorem solutions_satisfy_system :
  ∀ (p : ℝ × ℝ), p ∈ solutions → system p.1 p.2 := by
  sorry

end NUMINAMATH_CALUDE_solutions_satisfy_system_l2983_298318


namespace NUMINAMATH_CALUDE_total_questions_is_60_l2983_298348

-- Define the scoring system
def correct_score : ℕ := 4
def incorrect_score : ℕ := 1

-- Define the given information
def total_score : ℕ := 140
def correct_answers : ℕ := 40

-- Define the total number of questions attempted
def total_questions : ℕ := correct_answers + (correct_score * correct_answers - total_score)

-- Theorem to prove
theorem total_questions_is_60 : total_questions = 60 := by
  sorry

end NUMINAMATH_CALUDE_total_questions_is_60_l2983_298348


namespace NUMINAMATH_CALUDE_geometric_sum_not_always_geometric_arithmetic_and_geometric_is_constant_sum_power_not_always_arithmetic_or_geometric_arithmetic_sequence_no_equal_terms_l2983_298305

-- Definition of a geometric sequence
def is_geometric_sequence (a b c d : ℝ) : Prop :=
  (b / a = c / b) ∧ (c / b = d / c)

-- Definition of an arithmetic sequence
def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

-- Definition of a geometric sequence (for infinite sequences)
def is_geometric_sequence_inf (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

-- Definition of a constant sequence
def is_constant_sequence (a : ℕ → ℝ) : Prop :=
  ∃ c : ℝ, ∀ n : ℕ, a n = c

theorem geometric_sum_not_always_geometric :
  ∃ a b c d : ℝ, is_geometric_sequence a b c d ∧
  ¬ is_geometric_sequence (a + b) (b + c) (c + d) (d + a) :=
sorry

theorem arithmetic_and_geometric_is_constant (a : ℕ → ℝ) :
  is_arithmetic_sequence a → is_geometric_sequence_inf a → is_constant_sequence a :=
sorry

theorem sum_power_not_always_arithmetic_or_geometric :
  ∃ (a : ℝ) (S : ℕ → ℝ), (∀ n : ℕ, S n = a^n - 1) ∧
  ¬ (is_arithmetic_sequence S ∨ is_geometric_sequence_inf S) :=
sorry

theorem arithmetic_sequence_no_equal_terms (a : ℕ → ℝ) (d : ℝ) :
  is_arithmetic_sequence a → d ≠ 0 → ∀ m n : ℕ, m ≠ n → a m ≠ a n :=
sorry

end NUMINAMATH_CALUDE_geometric_sum_not_always_geometric_arithmetic_and_geometric_is_constant_sum_power_not_always_arithmetic_or_geometric_arithmetic_sequence_no_equal_terms_l2983_298305


namespace NUMINAMATH_CALUDE_twentieth_term_of_sequence_l2983_298345

def arithmetic_sequence (a₁ : ℕ) (d : ℕ) (n : ℕ) : ℕ :=
  a₁ + (n - 1) * d

theorem twentieth_term_of_sequence : arithmetic_sequence 3 5 20 = 98 := by
  sorry

end NUMINAMATH_CALUDE_twentieth_term_of_sequence_l2983_298345


namespace NUMINAMATH_CALUDE_reflect_M_across_y_axis_l2983_298386

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Reflects a point across the y-axis -/
def reflectAcrossYAxis (p : Point) : Point :=
  { x := -p.x, y := p.y }

/-- The theorem stating that reflecting M(3,2) across the y-axis results in (-3,2) -/
theorem reflect_M_across_y_axis :
  let M : Point := { x := 3, y := 2 }
  reflectAcrossYAxis M = { x := -3, y := 2 } := by
  sorry

end NUMINAMATH_CALUDE_reflect_M_across_y_axis_l2983_298386


namespace NUMINAMATH_CALUDE_payment_combinations_eq_six_l2983_298339

/-- Represents the number of ways to make a payment of 230 yuan using given bills -/
def payment_combinations : ℕ :=
  (Finset.filter (fun (x, y, z) => 
    50 * x + 20 * y + 10 * z = 230 ∧ 
    x ≤ 5 ∧ y ≤ 6 ∧ z ≤ 7)
    (Finset.product (Finset.range 6) (Finset.product (Finset.range 7) (Finset.range 8)))).card

/-- The theorem stating that there are exactly 6 ways to make the payment -/
theorem payment_combinations_eq_six : payment_combinations = 6 := by
  sorry

end NUMINAMATH_CALUDE_payment_combinations_eq_six_l2983_298339


namespace NUMINAMATH_CALUDE_tarun_worked_days_l2983_298361

/-- Represents the number of days it takes for Arun and Tarun to complete the work together -/
def combined_days : ℝ := 10

/-- Represents the number of days it takes for Arun to complete the work alone -/
def arun_alone_days : ℝ := 60

/-- Represents the number of days Arun worked alone after Tarun left -/
def arun_remaining_days : ℝ := 36

/-- Represents the total amount of work to be done -/
def total_work : ℝ := 1

/-- Theorem stating that Tarun worked for 4 days before leaving -/
theorem tarun_worked_days : 
  ∃ (t : ℝ), 
    t > 0 ∧ 
    t < combined_days ∧ 
    (t / combined_days + arun_remaining_days / arun_alone_days = total_work) ∧ 
    t = 4 := by
  sorry


end NUMINAMATH_CALUDE_tarun_worked_days_l2983_298361


namespace NUMINAMATH_CALUDE_percentage_passed_both_l2983_298353

theorem percentage_passed_both (failed_hindi : ℝ) (failed_english : ℝ) (failed_both : ℝ)
  (h1 : failed_hindi = 20)
  (h2 : failed_english = 70)
  (h3 : failed_both = 10) :
  100 - (failed_hindi + failed_english - failed_both) = 20 := by
  sorry

end NUMINAMATH_CALUDE_percentage_passed_both_l2983_298353


namespace NUMINAMATH_CALUDE_david_money_left_l2983_298330

/-- Represents the money situation of a person on a trip -/
def MoneyOnTrip (initial_amount spent_amount remaining_amount : ℕ) : Prop :=
  (initial_amount = spent_amount + remaining_amount) ∧
  (remaining_amount = spent_amount - 800)

theorem david_money_left :
  ∃ (spent_amount : ℕ), MoneyOnTrip 1800 spent_amount 500 :=
sorry

end NUMINAMATH_CALUDE_david_money_left_l2983_298330


namespace NUMINAMATH_CALUDE_quadratic_radicals_same_type_l2983_298399

-- Define the two quadratic expressions
def f (a : ℝ) : ℝ := 3 * a - 8
def g (a : ℝ) : ℝ := 17 - 2 * a

-- Theorem statement
theorem quadratic_radicals_same_type :
  ∃ (a : ℝ), a = 5 ∧ f a = g a :=
sorry

end NUMINAMATH_CALUDE_quadratic_radicals_same_type_l2983_298399


namespace NUMINAMATH_CALUDE_store_profit_calculation_l2983_298315

/-- Represents the pricing strategy and profit calculation for a store selling turtleneck sweaters -/
theorem store_profit_calculation (C : ℝ) (h : C > 0) :
  let initial_markup := 1.20
  let new_year_markup := 1.25
  let february_discount := 0.80
  let final_price := C * initial_markup * new_year_markup * february_discount
  final_price = 1.20 * C ∧ (final_price - C) / C = 0.20 := by
  sorry

end NUMINAMATH_CALUDE_store_profit_calculation_l2983_298315


namespace NUMINAMATH_CALUDE_lemming_average_distance_l2983_298375

/-- The average distance from a point to the sides of a square --/
theorem lemming_average_distance (side_length : ℝ) (diagonal_distance : ℝ) (turn_angle : ℝ) (final_distance : ℝ) : 
  side_length = 12 →
  diagonal_distance = 7.8 →
  turn_angle = 60 * π / 180 →
  final_distance = 3 →
  let d := (diagonal_distance / (side_length * Real.sqrt 2))
  let x := d * side_length + final_distance * Real.cos (π/2 - turn_angle)
  let y := d * side_length + final_distance * Real.sin (π/2 - turn_angle)
  (x + y + (side_length - x) + (side_length - y)) / 4 = 6 := by
sorry

end NUMINAMATH_CALUDE_lemming_average_distance_l2983_298375


namespace NUMINAMATH_CALUDE_min_teachers_is_16_l2983_298395

/-- Represents the number of teachers in each subject --/
structure TeacherCounts where
  maths : Nat
  physics : Nat
  chemistry : Nat

/-- Calculates the minimum number of teachers required --/
def minTeachersRequired (counts : TeacherCounts) : Nat :=
  counts.maths + counts.physics + counts.chemistry

/-- Theorem stating the minimum number of teachers required --/
theorem min_teachers_is_16 (counts : TeacherCounts) 
  (h_maths : counts.maths = 6)
  (h_physics : counts.physics = 5)
  (h_chemistry : counts.chemistry = 5) :
  minTeachersRequired counts = 16 := by
  sorry

end NUMINAMATH_CALUDE_min_teachers_is_16_l2983_298395


namespace NUMINAMATH_CALUDE_total_books_read_is_36sc_l2983_298363

/-- The number of books read by the entire student body in one year -/
def total_books_read (c s : ℕ) : ℕ :=
  let books_per_month : ℕ := 3
  let months_per_year : ℕ := 12
  let books_per_student_per_year : ℕ := books_per_month * months_per_year
  let total_students : ℕ := c * s
  books_per_student_per_year * total_students

/-- Theorem stating that the total number of books read is 36 * s * c -/
theorem total_books_read_is_36sc (c s : ℕ) :
  total_books_read c s = 36 * s * c := by
  sorry

end NUMINAMATH_CALUDE_total_books_read_is_36sc_l2983_298363


namespace NUMINAMATH_CALUDE_folded_polyhedron_volume_l2983_298381

/-- Represents a polyhedron formed by folding four squares and two equilateral triangles -/
structure FoldedPolyhedron where
  square_side_length : ℝ
  triangle_side_length : ℝ
  h_square_side : square_side_length = 2
  h_triangle_side : triangle_side_length = Real.sqrt 8

/-- Calculates the volume of the folded polyhedron -/
noncomputable def volume (p : FoldedPolyhedron) : ℝ :=
  (16 * Real.sqrt 2 / 3) * Real.sqrt (2 - Real.sqrt 2)

/-- Theorem stating that the volume of the folded polyhedron is correct -/
theorem folded_polyhedron_volume (p : FoldedPolyhedron) :
  volume p = (16 * Real.sqrt 2 / 3) * Real.sqrt (2 - Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_folded_polyhedron_volume_l2983_298381


namespace NUMINAMATH_CALUDE_cubic_polynomials_inequality_l2983_298394

/-- A cubic polynomial with real coefficients -/
structure CubicPolynomial where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The roots of a cubic polynomial -/
def roots (p : CubicPolynomial) : Finset ℝ := sorry

/-- Check if all roots of a polynomial are positive -/
def all_roots_positive (p : CubicPolynomial) : Prop :=
  ∀ r ∈ roots p, r > 0

/-- Given two cubic polynomials, check if the roots of one are reciprocals of the other -/
def roots_are_reciprocals (p q : CubicPolynomial) : Prop :=
  ∀ r ∈ roots p, (1 / r) ∈ roots q

theorem cubic_polynomials_inequality (p q : CubicPolynomial) 
  (h_positive : all_roots_positive p)
  (h_reciprocal : roots_are_reciprocals p q) :
  p.a * q.a > 9 ∧ p.b * q.b > 9 := by
  sorry

end NUMINAMATH_CALUDE_cubic_polynomials_inequality_l2983_298394


namespace NUMINAMATH_CALUDE_flower_count_l2983_298347

theorem flower_count (roses tulips lilies : ℕ) : 
  roses = 58 ∧ 
  roses = tulips + 15 ∧ 
  roses = lilies - 25 → 
  roses + tulips + lilies = 184 := by
sorry

end NUMINAMATH_CALUDE_flower_count_l2983_298347


namespace NUMINAMATH_CALUDE_average_lawn_cuts_l2983_298396

/-- The number of months in a year -/
def months_in_year : ℕ := 12

/-- The number of months Mr. Roper cuts his lawn 15 times -/
def high_frequency_months : ℕ := 6

/-- The number of months Mr. Roper cuts his lawn 3 times -/
def low_frequency_months : ℕ := 6

/-- The number of times Mr. Roper cuts his lawn in high frequency months -/
def high_frequency_cuts : ℕ := 15

/-- The number of times Mr. Roper cuts his lawn in low frequency months -/
def low_frequency_cuts : ℕ := 3

/-- Theorem stating the average number of times Mr. Roper cuts his yard per month -/
theorem average_lawn_cuts :
  (high_frequency_months * high_frequency_cuts + low_frequency_months * low_frequency_cuts) / months_in_year = 9 := by
  sorry

end NUMINAMATH_CALUDE_average_lawn_cuts_l2983_298396


namespace NUMINAMATH_CALUDE_moles_of_ki_formed_l2983_298380

/-- Represents the chemical reaction NH4I + KOH → NH3 + KI + H2O -/
structure ChemicalReaction where
  nh4i : ℝ  -- moles of NH4I
  koh : ℝ   -- moles of KOH
  nh3 : ℝ   -- moles of NH3
  ki : ℝ    -- moles of KI
  h2o : ℝ   -- moles of H2O

/-- The molar mass of NH4I in g/mol -/
def molar_mass_nh4i : ℝ := 144.95

/-- The total mass of NH4I in grams -/
def total_mass_nh4i : ℝ := 435

/-- Theorem stating that the number of moles of KI formed is 3 -/
theorem moles_of_ki_formed
  (reaction : ChemicalReaction)
  (h1 : reaction.koh = 3)
  (h2 : reaction.nh3 = 3)
  (h3 : reaction.h2o = 3)
  (h4 : reaction.nh4i = total_mass_nh4i / molar_mass_nh4i)
  (h5 : reaction.nh4i = reaction.koh) :
  reaction.ki = 3 := by
  sorry

end NUMINAMATH_CALUDE_moles_of_ki_formed_l2983_298380


namespace NUMINAMATH_CALUDE_chocolate_candy_difference_l2983_298327

/-- The difference in cost between chocolate and candy bar -/
def cost_difference (chocolate_cost candy_cost : ℕ) : ℕ :=
  chocolate_cost - candy_cost

/-- Theorem stating the difference in cost between chocolate and candy bar -/
theorem chocolate_candy_difference :
  cost_difference 7 2 = 5 := by sorry

end NUMINAMATH_CALUDE_chocolate_candy_difference_l2983_298327


namespace NUMINAMATH_CALUDE_g_neg_two_l2983_298369

def g (x : ℝ) : ℝ := x^3 - 3*x^2 + 4

theorem g_neg_two : g (-2) = -16 := by
  sorry

end NUMINAMATH_CALUDE_g_neg_two_l2983_298369


namespace NUMINAMATH_CALUDE_not_p_and_q_l2983_298322

-- Define proposition p
def p : Prop := ∀ (a b c : ℝ), a < b → a * c^2 < b * c^2

-- Define proposition q
def q : Prop := ∃ (x₀ : ℝ), x₀ > 0 ∧ x₀ - 1 - Real.log x₀ = 0

-- Theorem statement
theorem not_p_and_q : (¬p) ∧ q := by sorry

end NUMINAMATH_CALUDE_not_p_and_q_l2983_298322


namespace NUMINAMATH_CALUDE_reciprocal_roots_quadratic_l2983_298392

theorem reciprocal_roots_quadratic (k : ℝ) : 
  (∃ r₁ r₂ : ℝ, r₁ ≠ 0 ∧ r₂ ≠ 0 ∧ r₁ * r₂ = 1 ∧ 
    (∀ x : ℝ, 5.2 * x * x + 14.3 * x + k = 0 ↔ (x = r₁ ∨ x = r₂))) → 
  k = 5.2 := by
sorry


end NUMINAMATH_CALUDE_reciprocal_roots_quadratic_l2983_298392


namespace NUMINAMATH_CALUDE_solution_set_quadratic_inequality_l2983_298371

theorem solution_set_quadratic_inequality :
  ∀ x : ℝ, x * (x - 2) ≤ 0 ↔ 0 ≤ x ∧ x ≤ 2 := by sorry

end NUMINAMATH_CALUDE_solution_set_quadratic_inequality_l2983_298371


namespace NUMINAMATH_CALUDE_green_peaches_per_basket_l2983_298310

/-- Given 7 baskets with a total of 14 green peaches evenly distributed,
    prove that each basket contains 2 green peaches. -/
theorem green_peaches_per_basket :
  ∀ (num_baskets : ℕ) (total_green : ℕ) (green_per_basket : ℕ),
    num_baskets = 7 →
    total_green = 14 →
    total_green = num_baskets * green_per_basket →
    green_per_basket = 2 := by
  sorry

end NUMINAMATH_CALUDE_green_peaches_per_basket_l2983_298310


namespace NUMINAMATH_CALUDE_triangle_inequality_theorem_l2983_298390

-- Define a triangle ABC
structure Triangle :=
  (a b c : ℝ)
  (positive_sides : a > 0 ∧ b > 0 ∧ c > 0)
  (triangle_inequality : a < b + c ∧ b < c + a ∧ c < a + b)

-- Define the angle bisector points
structure AngleBisectorPoints (t : Triangle) :=
  (A₁ B₁ C₁ : ℝ × ℝ)

-- Define the property of points being concyclic
def are_concyclic (p₁ p₂ p₃ p₄ : ℝ × ℝ) : Prop := sorry

-- Define the theorem
theorem triangle_inequality_theorem (t : Triangle) (abp : AngleBisectorPoints t) :
  are_concyclic abp.A₁ abp.B₁ abp.C₁ (0, t.b) →
  (t.a / (t.b + t.c)) + (t.b / (t.c + t.a)) + (t.c / (t.a + t.b)) ≥ (Real.sqrt 17 - 1) / 2 :=
by sorry

end NUMINAMATH_CALUDE_triangle_inequality_theorem_l2983_298390


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l2983_298336

-- Define sets A and B
def A : Set ℝ := {x : ℝ | -2 ≤ x ∧ x ≤ 3}
def B : Set ℝ := {x : ℝ | (x + 1) * (x - 4) > 0}

-- Theorem statement
theorem intersection_of_A_and_B :
  A ∩ B = {x : ℝ | -2 ≤ x ∧ x < -1} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l2983_298336


namespace NUMINAMATH_CALUDE_balls_picked_is_two_l2983_298397

/-- The number of red balls in the bag -/
def red_balls : ℕ := 3

/-- The number of blue balls in the bag -/
def blue_balls : ℕ := 2

/-- The number of green balls in the bag -/
def green_balls : ℕ := 3

/-- The total number of balls in the bag -/
def total_balls : ℕ := red_balls + blue_balls + green_balls

/-- The probability of picking two red balls -/
def prob_two_red : ℚ := 3 / 28

/-- The number of balls picked at random -/
def balls_picked : ℕ := 2

/-- Theorem stating that the number of balls picked is 2 given the conditions -/
theorem balls_picked_is_two :
  (red_balls = 3 ∧ blue_balls = 2 ∧ green_balls = 3) →
  (prob_two_red = 3 / 28) →
  (balls_picked = 2) := by sorry

end NUMINAMATH_CALUDE_balls_picked_is_two_l2983_298397


namespace NUMINAMATH_CALUDE_sqrt_two_irrational_others_rational_l2983_298376

theorem sqrt_two_irrational_others_rational : 
  (∃ (q : ℚ), Real.sqrt 2 = (q : ℝ)) ∧ 
  (∃ (q : ℚ), (1 : ℝ) = (q : ℝ)) ∧ 
  (∃ (q : ℚ), (0 : ℝ) = (q : ℝ)) ∧ 
  (∃ (q : ℚ), (-1 : ℝ) = (q : ℝ)) →
  False :=
by sorry

end NUMINAMATH_CALUDE_sqrt_two_irrational_others_rational_l2983_298376


namespace NUMINAMATH_CALUDE_mary_money_left_l2983_298342

/-- The amount of money Mary has left after purchasing pizzas and drinks -/
def money_left (p : ℝ) : ℝ :=
  let initial_money := 50
  let drink_cost := p
  let medium_pizza_cost := 3 * p
  let large_pizza_cost := 5 * p
  let num_drinks := 4
  let num_medium_pizzas := 3
  let num_large_pizzas := 2
  initial_money - (num_drinks * drink_cost + num_medium_pizzas * medium_pizza_cost + num_large_pizzas * large_pizza_cost)

/-- Theorem stating that Mary has 50 - 23p dollars left after her purchases -/
theorem mary_money_left (p : ℝ) : money_left p = 50 - 23 * p := by
  sorry

end NUMINAMATH_CALUDE_mary_money_left_l2983_298342


namespace NUMINAMATH_CALUDE_combustion_reaction_result_l2983_298358

-- Define the thermochemical equations
def nitrobenzene_combustion (x : ℝ) : ℝ := 3094.88 * x
def aniline_combustion (y : ℝ) : ℝ := 3392.15 * y
def ethanol_combustion (z : ℝ) : ℝ := 1370 * z

-- Define the relationship between x and y based on nitrogen production
def nitrogen_production (x y : ℝ) : Prop := 0.5 * x + 0.5 * y = 0.15

-- Define the total heat released
def total_heat_released (x y z : ℝ) : Prop :=
  nitrobenzene_combustion x + aniline_combustion y + ethanol_combustion z = 1467.4

-- Define the mass of the solution
def solution_mass (x : ℝ) : ℝ := 470 * x

-- Define the theorem
theorem combustion_reaction_result :
  ∃ (x y z : ℝ),
    nitrogen_production x y ∧
    total_heat_released x y z ∧
    x = 0.1 ∧
    solution_mass x = 47 := by
  sorry

end NUMINAMATH_CALUDE_combustion_reaction_result_l2983_298358


namespace NUMINAMATH_CALUDE_minimum_score_raises_average_l2983_298308

def scores : List ℕ := [92, 88, 74, 65, 80]

def current_average : ℚ := (scores.sum : ℚ) / scores.length

def target_average : ℚ := current_average + 5

def minimum_score : ℕ := 110

theorem minimum_score_raises_average : 
  (((scores.sum + minimum_score) : ℚ) / (scores.length + 1)) = target_average := by sorry

end NUMINAMATH_CALUDE_minimum_score_raises_average_l2983_298308


namespace NUMINAMATH_CALUDE_jesse_mall_trip_l2983_298338

def mall_trip (initial_amount novel_cost : ℕ) : ℕ :=
  let lunch_cost := 2 * novel_cost
  let total_spent := novel_cost + lunch_cost
  initial_amount - total_spent

theorem jesse_mall_trip :
  mall_trip 50 7 = 29 := by
  sorry

end NUMINAMATH_CALUDE_jesse_mall_trip_l2983_298338


namespace NUMINAMATH_CALUDE_day5_sale_correct_l2983_298311

/-- Represents the sales data for a grocer over 6 days -/
structure GrocerSales where
  average_target : ℕ  -- Target average sale for 5 consecutive days
  day1 : ℕ
  day2 : ℕ
  day3 : ℕ
  day5 : ℕ  -- The day we want to calculate
  day6 : ℕ
  total_days : ℕ  -- Number of days for average calculation

/-- Calculates the required sale on the fifth day to meet the average target -/
def calculate_day5_sale (sales : GrocerSales) : ℕ :=
  sales.average_target * sales.total_days - (sales.day1 + sales.day2 + sales.day3 + sales.day5 + sales.day6)

/-- Theorem stating that the calculated sale for day 5 is correct -/
theorem day5_sale_correct (sales : GrocerSales) 
  (h1 : sales.average_target = 625)
  (h2 : sales.day1 = 435)
  (h3 : sales.day2 = 927)
  (h4 : sales.day3 = 855)
  (h5 : sales.day5 = 562)
  (h6 : sales.day6 = 741)
  (h7 : sales.total_days = 5) :
  calculate_day5_sale sales = 167 := by
  sorry

#eval calculate_day5_sale { 
  average_target := 625, 
  day1 := 435, 
  day2 := 927, 
  day3 := 855, 
  day5 := 562, 
  day6 := 741, 
  total_days := 5 
}

end NUMINAMATH_CALUDE_day5_sale_correct_l2983_298311


namespace NUMINAMATH_CALUDE_triangle_area_approx_l2983_298332

/-- The area of a triangle with sides 35 cm, 23 cm, and 41 cm is approximately 402.65 cm² --/
theorem triangle_area_approx (a b c : ℝ) (ha : a = 35) (hb : b = 23) (hc : c = 41) :
  ∃ (area : ℝ), abs (area - ((a * b * Real.sqrt (1 - ((a^2 + b^2 - c^2) / (2*a*b))^2)) / 2) - 402.65) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_approx_l2983_298332


namespace NUMINAMATH_CALUDE_no_integer_roots_l2983_298351

/-- A polynomial with integer coefficients -/
def IntPolynomial := ℤ → ℤ

/-- Evaluates a polynomial at a given integer -/
def eval (p : IntPolynomial) (x : ℤ) : ℤ := p x

/-- Predicate for odd integers -/
def is_odd (n : ℤ) : Prop := ∃ k : ℤ, n = 2 * k + 1

theorem no_integer_roots (p : IntPolynomial) 
  (h0 : is_odd (eval p 0)) 
  (h1 : is_odd (eval p 1)) : 
  ∀ k : ℤ, eval p k ≠ 0 := by
sorry

end NUMINAMATH_CALUDE_no_integer_roots_l2983_298351


namespace NUMINAMATH_CALUDE_find_a_l2983_298312

def U (a : ℝ) : Set ℝ := {2, 3, a^2 - 2*a - 3}
def A (a : ℝ) : Set ℝ := {2, |a - 7|}

theorem find_a : ∀ a : ℝ, (U a) \ (A a) = {5} → a = 4 ∨ a = -2 := by sorry

end NUMINAMATH_CALUDE_find_a_l2983_298312


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l2983_298378

def A : Set ℝ := {x | x > 0}
def B : Set ℝ := {x | -1 < x ∧ x ≤ 2}

theorem intersection_of_A_and_B : A ∩ B = {x : ℝ | 0 < x ∧ x ≤ 2} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l2983_298378


namespace NUMINAMATH_CALUDE_girth_bound_l2983_298320

/-- The minimum degree of a graph G -/
def min_degree (G : Type*) : ℕ := sorry

/-- The girth of a graph G -/
def girth (G : Type*) : ℕ := sorry

/-- The number of vertices in a graph G -/
def num_vertices (G : Type*) : ℕ := sorry

/-- Theorem: For any graph G with minimum degree ≥ 3, the girth is less than 2 log |G| -/
theorem girth_bound (G : Type*) (h : min_degree G ≥ 3) : 
  girth G < 2 * Real.log (num_vertices G) := by
  sorry

end NUMINAMATH_CALUDE_girth_bound_l2983_298320


namespace NUMINAMATH_CALUDE_quadratic_discriminant_l2983_298335

/-- The discriminant of a quadratic equation ax^2 + bx + c is b^2 - 4ac -/
def discriminant (a b c : ℝ) : ℝ := b^2 - 4*a*c

/-- Theorem: The discriminant of the quadratic equation 5x^2 - 2x - 7 is 144 -/
theorem quadratic_discriminant :
  discriminant 5 (-2) (-7) = 144 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_discriminant_l2983_298335


namespace NUMINAMATH_CALUDE_division_multiplication_problem_l2983_298362

theorem division_multiplication_problem : (0.45 / 0.005) * 2 = 180 := by
  sorry

end NUMINAMATH_CALUDE_division_multiplication_problem_l2983_298362


namespace NUMINAMATH_CALUDE_cookie_sheet_perimeter_l2983_298333

/-- The perimeter of a rectangular cookie sheet -/
theorem cookie_sheet_perimeter (width : ℝ) (length : ℝ) (inch_to_cm : ℝ) : 
  width = 15.2 ∧ length = 3.7 ∧ inch_to_cm = 2.54 →
  2 * (width * inch_to_cm + length * inch_to_cm) = 96.012 := by
  sorry

end NUMINAMATH_CALUDE_cookie_sheet_perimeter_l2983_298333


namespace NUMINAMATH_CALUDE_scout_camp_chocolate_cost_l2983_298344

/-- The cost of chocolate bars for a scout camp out --/
def chocolate_cost (bar_cost : ℚ) (sections_per_bar : ℕ) (num_scouts : ℕ) (smores_per_scout : ℕ) : ℚ :=
  let total_smores := num_scouts * smores_per_scout
  let bars_needed := (total_smores + sections_per_bar - 1) / sections_per_bar
  bars_needed * bar_cost

/-- Theorem: The cost of chocolate bars for the given scout camp out is $15.00 --/
theorem scout_camp_chocolate_cost :
  chocolate_cost (3/2) 3 15 2 = 15 := by
  sorry

end NUMINAMATH_CALUDE_scout_camp_chocolate_cost_l2983_298344


namespace NUMINAMATH_CALUDE_largest_three_digit_with_seven_hundreds_l2983_298316

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def has_seven_in_hundreds_place (n : ℕ) : Prop := (n / 100) % 10 = 7

theorem largest_three_digit_with_seven_hundreds : 
  ∀ n : ℕ, is_three_digit n → has_seven_in_hundreds_place n → n ≤ 799 :=
by sorry

end NUMINAMATH_CALUDE_largest_three_digit_with_seven_hundreds_l2983_298316


namespace NUMINAMATH_CALUDE_min_value_and_fraction_sum_l2983_298323

-- Define the function f(x)
def f (x : ℝ) : ℝ := |x + 1| + |2*x - 1|

-- Theorem statement
theorem min_value_and_fraction_sum :
  (∃ a : ℝ, (∀ x : ℝ, f x ≥ a) ∧ (∃ x₀ : ℝ, f x₀ = a) ∧ a = 3/2) ∧
  (∀ m n : ℝ, m > 0 → n > 0 → m + n = 3/2 → 1/m + 4/n ≥ 6) ∧
  (∃ m₀ n₀ : ℝ, m₀ > 0 ∧ n₀ > 0 ∧ m₀ + n₀ = 3/2 ∧ 1/m₀ + 4/n₀ = 6) :=
by sorry

end NUMINAMATH_CALUDE_min_value_and_fraction_sum_l2983_298323


namespace NUMINAMATH_CALUDE_inscribed_triangle_area_l2983_298367

theorem inscribed_triangle_area (a b c : ℝ) (h_positive : a > 0 ∧ b > 0 ∧ c > 0) :
  (∃ (r : ℝ), r = 4 ∧ ∃ (A B C : ℝ), 
    a / Real.sin A = b / Real.sin B ∧ b / Real.sin B = c / Real.sin C ∧ 
    c / Real.sin C = 2 * r) →
  a * b * c = 16 * Real.sqrt 2 →
  (1 / 2) * a * b * Real.sin (Real.arcsin (c / 8)) = Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_inscribed_triangle_area_l2983_298367


namespace NUMINAMATH_CALUDE_smallest_divisible_by_1_to_10_is_correct_l2983_298360

/-- The smallest positive integer divisible by all integers from 1 to 10 -/
def smallest_divisible_by_1_to_10 : ℕ := 2520

/-- Predicate to check if a number is divisible by all integers from 1 to 10 -/
def divisible_by_1_to_10 (n : ℕ) : Prop :=
  ∀ i : ℕ, 1 ≤ i → i ≤ 10 → n % i = 0

theorem smallest_divisible_by_1_to_10_is_correct :
  (divisible_by_1_to_10 smallest_divisible_by_1_to_10) ∧
  (∀ n : ℕ, n > 0 → divisible_by_1_to_10 n → n ≥ smallest_divisible_by_1_to_10) :=
by sorry

end NUMINAMATH_CALUDE_smallest_divisible_by_1_to_10_is_correct_l2983_298360


namespace NUMINAMATH_CALUDE_sphere_surface_area_l2983_298313

/-- The surface area of a sphere, given specific conditions for a hemisphere --/
theorem sphere_surface_area (r : ℝ) (h1 : π * r^2 = 3) (h2 : 3 * π * r^2 = 9) :
  ∃ S : ℝ → ℝ, ∀ x : ℝ, S x = 4 * π * x^2 := by sorry

end NUMINAMATH_CALUDE_sphere_surface_area_l2983_298313


namespace NUMINAMATH_CALUDE_prob_A_wins_3_1_l2983_298350

/-- The probability of Team A winning a single game -/
def prob_A_win : ℚ := 1/2

/-- The probability of Team B winning a single game -/
def prob_B_win : ℚ := 1/2

/-- The number of games in a best-of-five series where one team wins 3-1 -/
def games_played : ℕ := 4

/-- The number of ways to arrange 3 wins in 4 games -/
def winning_scenarios : ℕ := 3

/-- The probability of Team A winning with a score of 3:1 in a best-of-five series -/
theorem prob_A_wins_3_1 : 
  (prob_A_win ^ 3 * prob_B_win) * winning_scenarios = 3/16 := by
  sorry

end NUMINAMATH_CALUDE_prob_A_wins_3_1_l2983_298350


namespace NUMINAMATH_CALUDE_max_sum_of_cubes_l2983_298391

/-- Given real numbers a, b, c, d satisfying the condition,
    the sum of their cubes is bounded above by 4√10 -/
theorem max_sum_of_cubes (a b c d : ℝ) 
  (h : a^2 + b^2 + c^2 + d^2 + a*b + a*c + a*d + b*c + b*d + c*d = 10) : 
  a^3 + b^3 + c^3 + d^3 ≤ 4 * Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_max_sum_of_cubes_l2983_298391


namespace NUMINAMATH_CALUDE_sum_of_fourth_powers_is_square_l2983_298341

theorem sum_of_fourth_powers_is_square (a b c : ℤ) (h : a + b + c = 0) :
  2 * (a^4 + b^4 + c^4) = (a^2 + b^2 + c^2)^2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_fourth_powers_is_square_l2983_298341


namespace NUMINAMATH_CALUDE_arithmetic_geometric_mean_inequality_l2983_298373

theorem arithmetic_geometric_mean_inequality (x y z : ℝ) 
  (hx : x > 0) (hy : y > 0) (hz : z > 0) : 
  (x + y + z) / 3 ≥ (x * y * z) ^ (1/3) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_mean_inequality_l2983_298373


namespace NUMINAMATH_CALUDE_three_digit_numbers_divisible_by_11_equal_sum_of_squares_of_digits_l2983_298328

/-- A 3-digit number -/
def ThreeDigitNumber (n : ℕ) : Prop :=
  100 ≤ n ∧ n < 1000

/-- The sum of squares of digits of a natural number -/
def SumOfSquaresOfDigits (n : ℕ) : ℕ :=
  let hundreds := n / 100
  let tens := (n / 10) % 10
  let ones := n % 10
  hundreds * hundreds + tens * tens + ones * ones

/-- The main theorem -/
theorem three_digit_numbers_divisible_by_11_equal_sum_of_squares_of_digits :
  ∃! (s : Finset ℕ), s.card = 2 ∧ 
    ∀ n ∈ s, ThreeDigitNumber n ∧ 
             n % 11 = 0 ∧
             n / 11 = SumOfSquaresOfDigits n :=
by sorry

end NUMINAMATH_CALUDE_three_digit_numbers_divisible_by_11_equal_sum_of_squares_of_digits_l2983_298328


namespace NUMINAMATH_CALUDE_john_shoe_purchase_cost_l2983_298389

/-- Calculate the total cost including tax for two items -/
def total_cost (price1 : ℝ) (price2 : ℝ) (tax_rate : ℝ) : ℝ :=
  let total_before_tax := price1 + price2
  let tax_amount := total_before_tax * tax_rate
  total_before_tax + tax_amount

/-- Theorem stating the total cost for the given problem -/
theorem john_shoe_purchase_cost :
  total_cost 150 120 0.1 = 297 := by
  sorry

end NUMINAMATH_CALUDE_john_shoe_purchase_cost_l2983_298389


namespace NUMINAMATH_CALUDE_m_range_l2983_298384

theorem m_range (x m : ℝ) :
  (∀ x, (1/3 < x ∧ x < 1/2) ↔ |x - m| < 1) →
  -1/2 ≤ m ∧ m ≤ 4/3 :=
by sorry

end NUMINAMATH_CALUDE_m_range_l2983_298384


namespace NUMINAMATH_CALUDE_quadratic_form_sum_l2983_298366

/-- Given that 2x^2 - 8x + 1 can be expressed as a(x-h)^2 + k, prove that a + h + k = -3 -/
theorem quadratic_form_sum (a h k : ℝ) : 
  (∀ x, 2*x^2 - 8*x + 1 = a*(x-h)^2 + k) → a + h + k = -3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_form_sum_l2983_298366


namespace NUMINAMATH_CALUDE_cube_surface_area_proof_l2983_298302

-- Define the edge length of the cube
def edge_length : ℝ → ℝ := λ a => 7 * a

-- Define the surface area of a cube given its edge length
def cube_surface_area (edge : ℝ) : ℝ := 6 * edge^2

-- Theorem statement
theorem cube_surface_area_proof (a : ℝ) :
  cube_surface_area (edge_length a) = 294 * a^2 := by
  sorry

end NUMINAMATH_CALUDE_cube_surface_area_proof_l2983_298302


namespace NUMINAMATH_CALUDE_quadratic_transformation_l2983_298326

-- Define the original quadratic function
def original_function (x : ℝ) : ℝ := x^2

-- Define the transformation
def transform (f : ℝ → ℝ) (horizontal_shift : ℝ) (vertical_shift : ℝ) : ℝ → ℝ :=
  λ x => f (x - horizontal_shift) + vertical_shift

-- Define the new function after transformation
def new_function : ℝ → ℝ := transform original_function 3 3

-- Theorem stating the equivalence
theorem quadratic_transformation :
  ∀ x : ℝ, new_function x = (x - 3)^2 + 3 :=
by
  sorry

end NUMINAMATH_CALUDE_quadratic_transformation_l2983_298326


namespace NUMINAMATH_CALUDE_ceiling_of_negative_three_point_seven_l2983_298370

theorem ceiling_of_negative_three_point_seven :
  ⌈(-3.7 : ℝ)⌉ = -3 := by sorry

end NUMINAMATH_CALUDE_ceiling_of_negative_three_point_seven_l2983_298370


namespace NUMINAMATH_CALUDE_merchant_profit_theorem_l2983_298324

/-- Calculates the profit percentage given the ratio of articles sold to articles bought --/
def profit_percentage (articles_sold : ℕ) (articles_bought : ℕ) : ℚ :=
  ((articles_bought : ℚ) / (articles_sold : ℚ) - 1) * 100

/-- Proves that when 25 articles' cost price equals 18 articles' selling price, the profit is (7/18) * 100 percent --/
theorem merchant_profit_theorem (cost_price selling_price : ℚ) 
  (h : 25 * cost_price = 18 * selling_price) : 
  profit_percentage 18 25 = (7 / 18) * 100 := by
  sorry

#eval profit_percentage 18 25

end NUMINAMATH_CALUDE_merchant_profit_theorem_l2983_298324


namespace NUMINAMATH_CALUDE_max_sector_area_l2983_298317

/-- Sector represents a circular sector with radius and central angle -/
structure Sector where
  radius : ℝ
  angle : ℝ

/-- The perimeter of a sector -/
def sectorPerimeter (s : Sector) : ℝ := s.radius * s.angle + 2 * s.radius

/-- The area of a sector -/
def sectorArea (s : Sector) : ℝ := 0.5 * s.radius^2 * s.angle

/-- Theorem: Maximum area of a sector with perimeter 40 -/
theorem max_sector_area (s : Sector) (h : sectorPerimeter s = 40) :
  sectorArea s ≤ 100 ∧ (sectorArea s = 100 ↔ s.angle = 2) := by sorry

end NUMINAMATH_CALUDE_max_sector_area_l2983_298317


namespace NUMINAMATH_CALUDE_sticker_distribution_l2983_298349

/-- The number of ways to distribute n identical objects into k distinct groups,
    where each group must have at least one object -/
def distribute (n k : ℕ) : ℕ :=
  Nat.choose (n - 1) (k - 1)

/-- The number of stickers -/
def num_stickers : ℕ := 10

/-- The number of sheets of paper -/
def num_sheets : ℕ := 5

theorem sticker_distribution :
  distribute num_stickers num_sheets = 126 := by
  sorry

end NUMINAMATH_CALUDE_sticker_distribution_l2983_298349


namespace NUMINAMATH_CALUDE_optimal_price_and_profit_l2983_298377

/-- Represents the daily sales volume as a function of price -/
def sales_volume (x : ℝ) : ℝ := -20 * x + 1600

/-- Represents the daily profit as a function of price -/
def daily_profit (x : ℝ) : ℝ := (x - 40) * (sales_volume x)

/-- The selling price must not be less than 45 yuan -/
def min_price : ℝ := 45

/-- Theorem stating the optimal price and maximum profit -/
theorem optimal_price_and_profit :
  ∃ (x : ℝ), x ≥ min_price ∧
  (∀ y : ℝ, y ≥ min_price → daily_profit y ≤ daily_profit x) ∧
  x = 60 ∧ daily_profit x = 8000 := by
  sorry

#check optimal_price_and_profit

end NUMINAMATH_CALUDE_optimal_price_and_profit_l2983_298377


namespace NUMINAMATH_CALUDE_factor_of_polynomial_l2983_298368

theorem factor_of_polynomial (x : ℝ) : 
  (x - 1/2) ∣ (8*x^3 + 17*x^2 + 2*x - 3) := by
  sorry

end NUMINAMATH_CALUDE_factor_of_polynomial_l2983_298368


namespace NUMINAMATH_CALUDE_initial_flea_distance_l2983_298334

/-- Represents a flea's position on a 2D plane -/
structure FleaPosition where
  x : ℝ
  y : ℝ

/-- Represents the jump pattern of a flea -/
inductive JumpDirection
  | Right
  | Up
  | Left
  | Down

/-- Calculates the position of a flea after n jumps -/
def flea_position_after_jumps (initial_pos : FleaPosition) (direction : JumpDirection) (n : ℕ) : FleaPosition :=
  sorry

/-- Calculates the distance between two points on a 2D plane -/
def distance (p1 p2 : FleaPosition) : ℝ :=
  sorry

/-- Theorem stating the initial distance between the fleas -/
theorem initial_flea_distance (flea1_start flea2_start : FleaPosition)
  (h1 : flea_position_after_jumps flea1_start JumpDirection.Right 100 = 
        FleaPosition.mk (flea1_start.x - 50) (flea1_start.y - 50))
  (h2 : flea_position_after_jumps flea2_start JumpDirection.Left 100 = 
        FleaPosition.mk (flea2_start.x + 50) (flea2_start.y - 50))
  (h3 : distance (flea_position_after_jumps flea1_start JumpDirection.Right 100)
                 (flea_position_after_jumps flea2_start JumpDirection.Left 100) = 300) :
  distance flea1_start flea2_start = 2 :=
sorry

end NUMINAMATH_CALUDE_initial_flea_distance_l2983_298334


namespace NUMINAMATH_CALUDE_max_min_f_on_interval_l2983_298398

-- Define the function f(x) = x³ - 3x
def f (x : ℝ) := x^3 - 3*x

-- Define the interval [-2, 0]
def interval := Set.Icc (-2 : ℝ) 0

-- Theorem statement
theorem max_min_f_on_interval :
  ∃ (max min : ℝ),
    (∀ x ∈ interval, f x ≤ max) ∧
    (∃ x ∈ interval, f x = max) ∧
    (∀ x ∈ interval, min ≤ f x) ∧
    (∃ x ∈ interval, f x = min) ∧
    max = 2 ∧ min = -2 := by
  sorry

end NUMINAMATH_CALUDE_max_min_f_on_interval_l2983_298398


namespace NUMINAMATH_CALUDE_nicholas_crackers_l2983_298388

theorem nicholas_crackers (marcus_crackers mona_crackers nicholas_crackers : ℕ) : 
  marcus_crackers = 3 * mona_crackers →
  nicholas_crackers = mona_crackers + 6 →
  marcus_crackers = 27 →
  nicholas_crackers = 15 := by sorry

end NUMINAMATH_CALUDE_nicholas_crackers_l2983_298388


namespace NUMINAMATH_CALUDE_set_A_enumeration_l2983_298340

def A : Set ℚ := {z | ∃ p q : ℕ+, z = p / q ∧ p + q = 5}

theorem set_A_enumeration : A = {1/4, 2/3, 3/2, 4} := by
  sorry

end NUMINAMATH_CALUDE_set_A_enumeration_l2983_298340


namespace NUMINAMATH_CALUDE_simplify_fraction_l2983_298355

theorem simplify_fraction (b : ℝ) (h : b = 5) : 15 * b^4 / (75 * b^3) = 1 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l2983_298355


namespace NUMINAMATH_CALUDE_abcd_product_magnitude_l2983_298372

theorem abcd_product_magnitude (a b c d : ℝ) : 
  a ≠ 0 → b ≠ 0 → c ≠ 0 → d ≠ 0 →
  a ≠ b → a ≠ c → a ≠ d → b ≠ c → b ≠ d → c ≠ d →
  a^2 + 1/b = b^2 + 1/c → b^2 + 1/c = c^2 + 1/d → c^2 + 1/d = d^2 + 1/a →
  |a*b*c*d| = 1 := by
sorry

end NUMINAMATH_CALUDE_abcd_product_magnitude_l2983_298372


namespace NUMINAMATH_CALUDE_sequence_a_2006_bounds_l2983_298343

def sequence_a : ℕ → ℚ
  | 0 => 1/2
  | n+1 => sequence_a n + (1/2006) * (sequence_a n)^2

theorem sequence_a_2006_bounds : 
  1 - 1/2008 < sequence_a 2006 ∧ sequence_a 2006 < 1 := by
  sorry

end NUMINAMATH_CALUDE_sequence_a_2006_bounds_l2983_298343


namespace NUMINAMATH_CALUDE_max_imaginary_part_of_roots_l2983_298352

def polynomial (z : ℂ) : ℂ := z^9 + z^7 - z^5 + z^3 - z

def is_root (z : ℂ) : Prop := polynomial z = 0

def imaginary_part (z : ℂ) : ℝ := z.im

theorem max_imaginary_part_of_roots :
  ∃ (θ : ℝ), 
    -π/2 ≤ θ ∧ θ ≤ π/2 ∧
    (∀ (z : ℂ), is_root z → imaginary_part z ≤ Real.sin θ) ∧
    θ = π/2 :=
sorry

end NUMINAMATH_CALUDE_max_imaginary_part_of_roots_l2983_298352


namespace NUMINAMATH_CALUDE_decimal_to_fraction_sum_l2983_298387

theorem decimal_to_fraction_sum (a b : ℕ+) :
  (a : ℚ) / (b : ℚ) = 324375 / 1000000 ∧
  (∀ (c d : ℕ+), (c : ℚ) / (d : ℚ) = 324375 / 1000000 → a ≤ c) →
  (a : ℕ) + (b : ℕ) = 2119 := by
sorry

end NUMINAMATH_CALUDE_decimal_to_fraction_sum_l2983_298387


namespace NUMINAMATH_CALUDE_ana_guarantee_l2983_298329

/-- The hat game setup -/
structure HatGame where
  n : ℕ
  h_n_gt_1 : n > 1

/-- The minimum number of correct guesses Ana can guarantee -/
def min_correct_guesses (game : HatGame) : ℕ :=
  (game.n - 1) / 2

/-- The theorem stating Ana's guarantee -/
theorem ana_guarantee (game : HatGame) :
  ∃ (strategy : Type),
    ∀ (bob_distribution : Type),
      ∃ (correct_guesses : ℕ),
        correct_guesses ≥ min_correct_guesses game :=
sorry


end NUMINAMATH_CALUDE_ana_guarantee_l2983_298329


namespace NUMINAMATH_CALUDE_not_all_zero_equiv_one_nonzero_l2983_298346

theorem not_all_zero_equiv_one_nonzero (a b c : ℝ) :
  ¬(a = 0 ∧ b = 0 ∧ c = 0) ↔ (a ≠ 0 ∨ b ≠ 0 ∨ c ≠ 0) := by
  sorry

end NUMINAMATH_CALUDE_not_all_zero_equiv_one_nonzero_l2983_298346


namespace NUMINAMATH_CALUDE_green_peaches_count_l2983_298319

/-- The number of green peaches in a basket, given the number of red, yellow, and total green and yellow peaches. -/
def num_green_peaches (red : ℕ) (yellow : ℕ) (green_and_yellow : ℕ) : ℕ :=
  green_and_yellow - yellow

/-- Theorem stating that there are 6 green peaches in the basket. -/
theorem green_peaches_count :
  let red : ℕ := 5
  let yellow : ℕ := 14
  let green_and_yellow : ℕ := 20
  num_green_peaches red yellow green_and_yellow = 6 := by
  sorry

end NUMINAMATH_CALUDE_green_peaches_count_l2983_298319


namespace NUMINAMATH_CALUDE_distance_between_points_l2983_298359

/-- Two cars traveling towards each other -/
structure CarProblem where
  /-- Speed of Car A in km/h -/
  speed_a : ℝ
  /-- Speed of Car B in km/h -/
  speed_b : ℝ
  /-- Time in hours until cars meet -/
  time_to_meet : ℝ
  /-- Additional time for Car A to reach point B after meeting -/
  additional_time : ℝ

/-- The theorem stating the distance between points A and B -/
theorem distance_between_points (p : CarProblem)
  (h1 : p.speed_a = p.speed_b + 20)
  (h2 : p.time_to_meet = 4)
  (h3 : p.additional_time = 3) :
  p.speed_a * p.time_to_meet + p.speed_b * p.time_to_meet = 240 := by
  sorry

#check distance_between_points

end NUMINAMATH_CALUDE_distance_between_points_l2983_298359


namespace NUMINAMATH_CALUDE_inequality_equivalence_l2983_298303

theorem inequality_equivalence (x : ℝ) : 
  |x - 2| + |x + 3| < 7 ↔ -4 < x ∧ x < 3 := by
sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l2983_298303


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l2983_298383

theorem simplify_and_evaluate (a b : ℚ) (h1 : a = -1) (h2 : b = 1/4) :
  (a + 2*b)^2 + (a + 2*b)*(a - 2*b) = 1 := by sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l2983_298383


namespace NUMINAMATH_CALUDE_common_root_of_quadratic_equations_l2983_298382

theorem common_root_of_quadratic_equations (a b x : ℝ) :
  (x^2 + 2019*a*x + b = 0) ∧
  (x^2 + 2019*b*x + a = 0) ∧
  (a ≠ b) →
  x = 1/2019 :=
by sorry

end NUMINAMATH_CALUDE_common_root_of_quadratic_equations_l2983_298382


namespace NUMINAMATH_CALUDE_abs_and_recip_of_neg_one_point_two_l2983_298307

theorem abs_and_recip_of_neg_one_point_two :
  let x : ℝ := -1.2
  abs x = 1.2 ∧ x⁻¹ = -5/6 := by
  sorry

end NUMINAMATH_CALUDE_abs_and_recip_of_neg_one_point_two_l2983_298307


namespace NUMINAMATH_CALUDE_cube_as_difference_of_squares_l2983_298301

theorem cube_as_difference_of_squares (n : ℤ) (h : n > 1) :
  ∃ (a b : ℤ), a > 0 ∧ b > 0 ∧ n^3 = a^2 - b^2 := by
  sorry

end NUMINAMATH_CALUDE_cube_as_difference_of_squares_l2983_298301


namespace NUMINAMATH_CALUDE_largest_three_digit_congruence_l2983_298374

theorem largest_three_digit_congruence :
  ∃ m : ℕ,
    100 ≤ m ∧ m ≤ 999 ∧
    40 * m ≡ 120 [MOD 200] ∧
    ∀ n : ℕ, 100 ≤ n ∧ n ≤ 999 ∧ 40 * n ≡ 120 [MOD 200] → n ≤ m ∧
    m = 998 :=
by sorry

end NUMINAMATH_CALUDE_largest_three_digit_congruence_l2983_298374


namespace NUMINAMATH_CALUDE_alcohol_solution_proof_l2983_298331

theorem alcohol_solution_proof (initial_volume : ℝ) (initial_percentage : ℝ) 
  (target_percentage : ℝ) (added_alcohol : ℝ) : 
  initial_volume = 100 ∧ 
  initial_percentage = 0.20 ∧ 
  target_percentage = 0.30 ∧
  added_alcohol = 14.2857 →
  (initial_volume * initial_percentage + added_alcohol) / (initial_volume + added_alcohol) = target_percentage :=
by
  sorry

end NUMINAMATH_CALUDE_alcohol_solution_proof_l2983_298331


namespace NUMINAMATH_CALUDE_complex_arithmetic_l2983_298314

theorem complex_arithmetic (A M S : ℂ) (P : ℝ) 
  (hA : A = 5 - 2*I) 
  (hM : M = -3 + 2*I) 
  (hS : S = 2*I) 
  (hP : P = 3) : 
  A - M + S - (P : ℂ) = 5 - 2*I := by
  sorry

end NUMINAMATH_CALUDE_complex_arithmetic_l2983_298314


namespace NUMINAMATH_CALUDE_water_channel_length_l2983_298337

theorem water_channel_length : ∀ L : ℝ,
  L > 0 →
  (3/4 * L - 5/28 * L) = 4/7 * L →
  (4/7 * L - 2/7 * L) = 2/7 * L →
  2/7 * L = 100 →
  L = 350 := by
sorry

end NUMINAMATH_CALUDE_water_channel_length_l2983_298337


namespace NUMINAMATH_CALUDE_proportion_equality_l2983_298365

theorem proportion_equality (x : ℝ) : (x / 5 = 1.2 / 8) → x = 0.75 := by
  sorry

end NUMINAMATH_CALUDE_proportion_equality_l2983_298365


namespace NUMINAMATH_CALUDE_research_budget_allocation_l2983_298309

theorem research_budget_allocation (microphotonics : ℝ) (home_electronics : ℝ)
  (genetically_modified_microorganisms : ℝ) (industrial_lubricants : ℝ)
  (basic_astrophysics_degrees : ℝ) :
  microphotonics = 14 →
  home_electronics = 24 →
  genetically_modified_microorganisms = 29 →
  industrial_lubricants = 8 →
  basic_astrophysics_degrees = 18 →
  ∃ (food_additives : ℝ),
    food_additives = 20 ∧
    microphotonics + home_electronics + genetically_modified_microorganisms +
    industrial_lubricants + (basic_astrophysics_degrees / 360 * 100) + food_additives = 100 :=
by sorry

end NUMINAMATH_CALUDE_research_budget_allocation_l2983_298309


namespace NUMINAMATH_CALUDE_toms_age_ratio_l2983_298364

theorem toms_age_ratio (T M : ℝ) : T > 0 → M > 0 → T - M = 3 * (T - 4 * M) → T / M = 11 / 2 := by
  sorry

end NUMINAMATH_CALUDE_toms_age_ratio_l2983_298364


namespace NUMINAMATH_CALUDE_A_is_irrational_l2983_298321

/-- The sequence of consecutive prime numbers -/
def consecutive_primes : ℕ → ℕ := sorry

/-- The decimal representation of our number -/
def A : ℝ := sorry

/-- Dirichlet's theorem on arithmetic progressions -/
axiom dirichlet_theorem : ∃ (infinitely_many : Set ℕ), ∀ p ∈ infinitely_many, 
  ∃ (n x : ℕ), p = 10^(n+1) * x + 1 ∧ Prime p

/-- The main theorem: A is irrational -/
theorem A_is_irrational : Irrational A := sorry

end NUMINAMATH_CALUDE_A_is_irrational_l2983_298321


namespace NUMINAMATH_CALUDE_qt_length_l2983_298354

/-- Square with side length 4 and special points T and U -/
structure SpecialSquare where
  -- Square PQRS with side length 4
  side : ℝ
  side_eq : side = 4

  -- Point T on side PQ
  t : ℝ × ℝ
  t_on_pq : t.1 ≥ 0 ∧ t.1 ≤ side ∧ t.2 = 0

  -- Point U on side PS
  u : ℝ × ℝ
  u_on_ps : u.1 = 0 ∧ u.2 ≥ 0 ∧ u.2 ≤ side

  -- Lines QT and SU divide the square into four equal areas
  equal_areas : (side * t.1) / 2 = (side * side) / 4

/-- The length of QT in a SpecialSquare is 2√3 -/
theorem qt_length (sq : SpecialSquare) : 
  Real.sqrt ((sq.side - sq.t.1)^2 + sq.t.1^2) = 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_qt_length_l2983_298354


namespace NUMINAMATH_CALUDE_arithmetic_calculation_l2983_298379

theorem arithmetic_calculation : (28 * 9 + 18 * 19 + 8 * 29) / 14 = 59 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculation_l2983_298379


namespace NUMINAMATH_CALUDE_negation_equivalence_l2983_298325

theorem negation_equivalence (m : ℤ) : 
  (¬ ∃ x : ℤ, x^2 + 2*x + m ≤ 0) ↔ (∀ x : ℤ, x^2 + 2*x + m > 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_equivalence_l2983_298325


namespace NUMINAMATH_CALUDE_muffin_cost_calculation_l2983_298300

/-- Given a purchase of 3 items of equal cost and one item of known cost,
    with a discount applied, prove the original cost of each equal-cost item. -/
theorem muffin_cost_calculation (M : ℝ) : 
  (∃ (M : ℝ), 
    (0.85 * (3 * M + 1.45) = 3.70) ∧ 
    (abs (M - 0.97) < 0.01)) := by
  sorry

end NUMINAMATH_CALUDE_muffin_cost_calculation_l2983_298300


namespace NUMINAMATH_CALUDE_book_price_increase_l2983_298356

theorem book_price_increase (final_price : ℝ) (increase_percentage : ℝ) 
  (h1 : final_price = 360)
  (h2 : increase_percentage = 20) :
  let original_price := final_price / (1 + increase_percentage / 100)
  original_price = 300 := by
sorry

end NUMINAMATH_CALUDE_book_price_increase_l2983_298356


namespace NUMINAMATH_CALUDE_marys_max_earnings_l2983_298393

/-- Mary's work schedule and pay structure --/
structure WorkSchedule where
  maxHours : Nat
  regularHours : Nat
  regularRate : ℚ
  overtimeRateIncrease : ℚ

/-- Calculate Mary's maximum weekly earnings --/
def calculateMaxEarnings (schedule : WorkSchedule) : ℚ :=
  let regularEarnings := schedule.regularRate * schedule.regularHours
  let overtimeHours := schedule.maxHours - schedule.regularHours
  let overtimeRate := schedule.regularRate * (1 + schedule.overtimeRateIncrease)
  let overtimeEarnings := overtimeRate * overtimeHours
  regularEarnings + overtimeEarnings

/-- Mary's specific work schedule --/
def marysSchedule : WorkSchedule :=
  { maxHours := 40
  , regularHours := 20
  , regularRate := 8
  , overtimeRateIncrease := 1/4 }

/-- Theorem: Mary's maximum weekly earnings are $360 --/
theorem marys_max_earnings :
  calculateMaxEarnings marysSchedule = 360 := by
  sorry

end NUMINAMATH_CALUDE_marys_max_earnings_l2983_298393


namespace NUMINAMATH_CALUDE_cricket_team_right_handed_players_l2983_298306

theorem cricket_team_right_handed_players 
  (total_players : ℕ) 
  (throwers : ℕ) 
  (h1 : total_players = 67)
  (h2 : throwers = 37)
  (h3 : throwers ≤ total_players)
  (h4 : (total_players - throwers) % 3 = 0) :
  throwers + (total_players - throwers) * 2 / 3 = 57 := by
  sorry

end NUMINAMATH_CALUDE_cricket_team_right_handed_players_l2983_298306


namespace NUMINAMATH_CALUDE_f_symmetry_l2983_298385

/-- Given a function f(x) = x^2005 + ax^3 - b/x - 8, where a and b are real constants,
    if f(-2) = 10, then f(2) = -26 -/
theorem f_symmetry (a b : ℝ) : 
  let f : ℝ → ℝ := λ x => x^2005 + a*x^3 - b/x - 8
  f (-2) = 10 → f 2 = -26 := by
sorry

end NUMINAMATH_CALUDE_f_symmetry_l2983_298385
