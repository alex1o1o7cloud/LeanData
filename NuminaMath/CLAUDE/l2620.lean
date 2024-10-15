import Mathlib

namespace NUMINAMATH_CALUDE_triangle_area_prove_triangle_area_l2620_262082

/-- The area of the triangle formed by the lines y = 3x - 3, y = -2x + 18, and the y-axis -/
theorem triangle_area : ℝ → Prop :=
  fun A => 
    let line1 := fun x : ℝ => 3 * x - 3
    let line2 := fun x : ℝ => -2 * x + 18
    let y_axis := fun x : ℝ => 0
    let intersection_x := (21 : ℝ) / 5
    let intersection_y := line1 intersection_x
    let base := line2 0 - line1 0
    let height := intersection_x
    A = (1 / 2) * base * height ∧ A = 441 / 10

/-- Proof of the theorem -/
theorem prove_triangle_area : ∃ A : ℝ, triangle_area A :=
  sorry

end NUMINAMATH_CALUDE_triangle_area_prove_triangle_area_l2620_262082


namespace NUMINAMATH_CALUDE_equilateral_triangle_area_perimeter_ratio_l2620_262056

/-- The ratio of the area to the square of the perimeter for an equilateral triangle with side length 10 -/
theorem equilateral_triangle_area_perimeter_ratio :
  let side_length : ℝ := 10
  let perimeter : ℝ := 3 * side_length
  let height : ℝ := side_length * (Real.sqrt 3 / 2)
  let area : ℝ := (1 / 2) * side_length * height
  area / (perimeter ^ 2) = Real.sqrt 3 / 36 := by
sorry

end NUMINAMATH_CALUDE_equilateral_triangle_area_perimeter_ratio_l2620_262056


namespace NUMINAMATH_CALUDE_quadratic_inequality_implies_a_range_l2620_262022

theorem quadratic_inequality_implies_a_range (a : ℝ) : 
  (∀ x : ℝ, x^2 + 2*a*x + 1 ≥ 0) → -1 ≤ a ∧ a ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_implies_a_range_l2620_262022


namespace NUMINAMATH_CALUDE_num_selections_with_A_or_B_l2620_262077

/-- The number of key projects -/
def num_key_projects : ℕ := 4

/-- The number of general projects -/
def num_general_projects : ℕ := 6

/-- The number of key projects to be selected -/
def select_key : ℕ := 2

/-- The number of general projects to be selected -/
def select_general : ℕ := 2

/-- Theorem stating the number of selection methods with at least one of A or B selected -/
theorem num_selections_with_A_or_B : 
  (Nat.choose (num_key_projects - 1) (select_key - 1) * Nat.choose (num_general_projects - 1) select_general) +
  (Nat.choose (num_key_projects - 1) select_key * Nat.choose (num_general_projects - 1) (select_general - 1)) +
  (Nat.choose (num_key_projects - 1) (select_key - 1) * Nat.choose (num_general_projects - 1) (select_general - 1)) = 60 := by
  sorry


end NUMINAMATH_CALUDE_num_selections_with_A_or_B_l2620_262077


namespace NUMINAMATH_CALUDE_total_age_of_couple_l2620_262088

def bride_age : ℕ := 102
def age_difference : ℕ := 19

theorem total_age_of_couple : 
  bride_age + (bride_age - age_difference) = 185 := by sorry

end NUMINAMATH_CALUDE_total_age_of_couple_l2620_262088


namespace NUMINAMATH_CALUDE_leftover_coin_value_l2620_262008

/-- The number of quarters in a roll -/
def quarters_per_roll : ℕ := 30

/-- The number of dimes in a roll -/
def dimes_per_roll : ℕ := 40

/-- The number of quarters James has -/
def james_quarters : ℕ := 77

/-- The number of dimes James has -/
def james_dimes : ℕ := 138

/-- The number of quarters Lindsay has -/
def lindsay_quarters : ℕ := 112

/-- The number of dimes Lindsay has -/
def lindsay_dimes : ℕ := 244

/-- The value of a quarter in dollars -/
def quarter_value : ℚ := 0.25

/-- The value of a dime in dollars -/
def dime_value : ℚ := 0.10

/-- The theorem stating the value of leftover coins -/
theorem leftover_coin_value :
  let total_quarters := james_quarters + lindsay_quarters
  let total_dimes := james_dimes + lindsay_dimes
  let leftover_quarters := total_quarters % quarters_per_roll
  let leftover_dimes := total_dimes % dimes_per_roll
  (leftover_quarters : ℚ) * quarter_value + (leftover_dimes : ℚ) * dime_value = 2.45 := by
  sorry


end NUMINAMATH_CALUDE_leftover_coin_value_l2620_262008


namespace NUMINAMATH_CALUDE_unique_solution_quadratic_l2620_262070

theorem unique_solution_quadratic (a : ℝ) :
  (∃! x : ℝ, a * x^2 - 3 * x + 2 = 0) ↔ (a = 0 ∨ a = 9/8) := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_quadratic_l2620_262070


namespace NUMINAMATH_CALUDE_fermat_prime_equation_solutions_l2620_262066

/-- A Fermat's Prime is a prime number of the form 2^α + 1, for α a positive integer -/
def IsFermatPrime (p : ℕ) : Prop :=
  Nat.Prime p ∧ ∃ α : ℕ, α > 0 ∧ p = 2^α + 1

/-- The main theorem statement -/
theorem fermat_prime_equation_solutions :
  ∀ p n k : ℕ,
  p > 0 ∧ n > 0 ∧ k > 0 →
  IsFermatPrime p →
  p^n + n = (n+1)^k →
  (p = 3 ∧ n = 1 ∧ k = 2) ∨ (p = 5 ∧ n = 2 ∧ k = 3) :=
by sorry

end NUMINAMATH_CALUDE_fermat_prime_equation_solutions_l2620_262066


namespace NUMINAMATH_CALUDE_equation_solution_l2620_262080

theorem equation_solution : 
  ∃! x : ℝ, x^2 + 6*x + 6*x * Real.sqrt (x + 5) = 24 ∧ 
  x = (-17 + Real.sqrt 277) / 2 := by
sorry

end NUMINAMATH_CALUDE_equation_solution_l2620_262080


namespace NUMINAMATH_CALUDE_equation_has_real_roots_l2620_262096

theorem equation_has_real_roots (a b : ℝ) : 
  ∃ x : ℝ, (x^2 / (x^2 - a^2) + x^2 / (x^2 - b^2) = 4) := by sorry

end NUMINAMATH_CALUDE_equation_has_real_roots_l2620_262096


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_range_l2620_262043

/-- Given a hyperbola C: (y^2 / a^2) - (x^2 / b^2) = 1 with a > 0 and b > 0,
    whose asymptotes intersect with the circle x^2 + (y - 2)^2 = 1,
    the eccentricity e of C satisfies 1 < e < 2√3/3. -/
theorem hyperbola_eccentricity_range (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let C := {(x, y) : ℝ × ℝ | y^2 / a^2 - x^2 / b^2 = 1}
  let asymptotes := {(x, y) : ℝ × ℝ | a * x = b * y ∨ a * x = -b * y}
  let circle := {(x, y) : ℝ × ℝ | x^2 + (y - 2)^2 = 1}
  let e := Real.sqrt (1 + b^2 / a^2)
  (∃ p ∈ asymptotes, p ∈ circle) →
  1 < e ∧ e < 2 * Real.sqrt 3 / 3 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_range_l2620_262043


namespace NUMINAMATH_CALUDE_inequality_proof_l2620_262037

theorem inequality_proof (x y : ℝ) (h1 : x > y) (h2 : x * y = 1) :
  (x^2 + y^2) / (x - y) ≥ 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2620_262037


namespace NUMINAMATH_CALUDE_no_constant_term_implies_n_not_eight_l2620_262099

theorem no_constant_term_implies_n_not_eight (n : ℕ) :
  (∀ r : ℕ, r ≤ n → n ≠ 4 / 3 * r) →
  n ≠ 8 := by
  sorry

end NUMINAMATH_CALUDE_no_constant_term_implies_n_not_eight_l2620_262099


namespace NUMINAMATH_CALUDE_imaginary_part_of_2_minus_3i_l2620_262092

theorem imaginary_part_of_2_minus_3i :
  Complex.im (2 - 3 * Complex.I) = -3 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_2_minus_3i_l2620_262092


namespace NUMINAMATH_CALUDE_game_cost_l2620_262033

/-- 
Given:
- Will made 104 dollars mowing lawns
- He spent 41 dollars on new mower blades
- He bought 7 games with the remaining money
Prove that each game cost 9 dollars
-/
theorem game_cost (total_earned : ℕ) (spent_on_blades : ℕ) (num_games : ℕ) :
  total_earned = 104 →
  spent_on_blades = 41 →
  num_games = 7 →
  (total_earned - spent_on_blades) / num_games = 9 :=
by sorry

end NUMINAMATH_CALUDE_game_cost_l2620_262033


namespace NUMINAMATH_CALUDE_theater_ticket_price_l2620_262095

/-- The price of tickets at a theater with discounts for children and seniors -/
theorem theater_ticket_price :
  ∀ (adult_price : ℝ),
  (6 * adult_price + 5 * (adult_price / 2) + 3 * (adult_price * 0.75) = 42) →
  (10 * adult_price + 8 * (adult_price / 2) + 4 * (adult_price * 0.75) = 58.65) :=
by
  sorry

end NUMINAMATH_CALUDE_theater_ticket_price_l2620_262095


namespace NUMINAMATH_CALUDE_quadratic_complete_square_l2620_262015

theorem quadratic_complete_square (x : ℝ) : 
  x^2 + 10*x + 7 = 0 → ∃ c d : ℝ, (x + c)^2 = d ∧ d = 18 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_complete_square_l2620_262015


namespace NUMINAMATH_CALUDE_intersection_of_P_and_Q_l2620_262048

def P : Set ℝ := {-2, 0, 2, 4}
def Q : Set ℝ := {x | 0 < x ∧ x < 3}

theorem intersection_of_P_and_Q : P ∩ Q = {2} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_P_and_Q_l2620_262048


namespace NUMINAMATH_CALUDE_prob_red_then_green_is_two_ninths_l2620_262090

def num_red_balls : ℕ := 2
def num_green_balls : ℕ := 1
def total_balls : ℕ := num_red_balls + num_green_balls

def probability_red_then_green : ℚ :=
  (num_red_balls : ℚ) / total_balls * (num_green_balls : ℚ) / total_balls

theorem prob_red_then_green_is_two_ninths :
  probability_red_then_green = 2 / 9 := by
  sorry

end NUMINAMATH_CALUDE_prob_red_then_green_is_two_ninths_l2620_262090


namespace NUMINAMATH_CALUDE_parking_lot_vehicles_l2620_262010

/-- Given a parking lot with tricycles and bicycles, prove the number of each type --/
theorem parking_lot_vehicles (total_vehicles : ℕ) (total_wheels : ℕ) 
  (h1 : total_vehicles = 15)
  (h2 : total_wheels = 40) :
  ∃ (tricycles bicycles : ℕ),
    tricycles + bicycles = total_vehicles ∧
    3 * tricycles + 2 * bicycles = total_wheels ∧
    tricycles = 10 ∧
    bicycles = 5 := by
  sorry

end NUMINAMATH_CALUDE_parking_lot_vehicles_l2620_262010


namespace NUMINAMATH_CALUDE_man_double_son_age_l2620_262036

/-- Represents the age difference between a man and his son -/
def age_difference : ℕ := 35

/-- Represents the son's present age -/
def son_present_age : ℕ := 33

/-- Calculates the number of years until the man's age is twice his son's age -/
def years_until_double_age : ℕ := 2

/-- Theorem stating that the number of years until the man's age is twice his son's age is 2 -/
theorem man_double_son_age :
  (son_present_age + years_until_double_age) * 2 = 
  (son_present_age + age_difference + years_until_double_age) :=
by sorry

end NUMINAMATH_CALUDE_man_double_son_age_l2620_262036


namespace NUMINAMATH_CALUDE_total_drawings_l2620_262076

/-- The number of neighbors Shiela has -/
def num_neighbors : ℕ := 6

/-- The number of drawings each neighbor receives -/
def drawings_per_neighbor : ℕ := 9

/-- Theorem: The total number of drawings Shiela made is 54 -/
theorem total_drawings : num_neighbors * drawings_per_neighbor = 54 := by
  sorry

end NUMINAMATH_CALUDE_total_drawings_l2620_262076


namespace NUMINAMATH_CALUDE_artist_paintings_l2620_262093

/-- Calculates the number of paintings an artist can complete in a given number of weeks. -/
def paintings_completed (hours_per_week : ℕ) (hours_per_painting : ℕ) (num_weeks : ℕ) : ℕ :=
  (hours_per_week / hours_per_painting) * num_weeks

/-- Proves that an artist spending 30 hours per week painting, taking 3 hours per painting,
    can complete 40 paintings in 4 weeks. -/
theorem artist_paintings : paintings_completed 30 3 4 = 40 := by
  sorry

end NUMINAMATH_CALUDE_artist_paintings_l2620_262093


namespace NUMINAMATH_CALUDE_integer_divisibility_problem_l2620_262012

theorem integer_divisibility_problem :
  ∀ a b c : ℕ,
    1 < a → a < b → b < c →
    (((a - 1) * (b - 1) * (c - 1)) ∣ (a * b * c - 1)) →
    ((a = 2 ∧ b = 4 ∧ c = 8) ∨ (a = 3 ∧ b = 5 ∧ c = 15)) :=
by sorry

end NUMINAMATH_CALUDE_integer_divisibility_problem_l2620_262012


namespace NUMINAMATH_CALUDE_count_satisfying_numbers_is_45_l2620_262073

/-- A function that checks if a three-digit number satisfies the condition -/
def satisfiesCondition (n : Nat) : Bool :=
  let a := n / 100
  let b := (n / 10) % 10
  let c := n % 10
  b = a + c ∧ 100 ≤ n ∧ n ≤ 999

/-- The count of three-digit numbers satisfying the condition -/
def countSatisfyingNumbers : Nat :=
  (List.range 900).map (· + 100)
    |>.filter satisfiesCondition
    |>.length

/-- Theorem stating that the count of satisfying numbers is 45 -/
theorem count_satisfying_numbers_is_45 : countSatisfyingNumbers = 45 := by
  sorry

end NUMINAMATH_CALUDE_count_satisfying_numbers_is_45_l2620_262073


namespace NUMINAMATH_CALUDE_sum_of_digits_of_greatest_prime_divisor_l2620_262067

def n : ℕ := 16385

-- Define a function to get the greatest prime divisor
def greatest_prime_divisor (m : ℕ) : ℕ := sorry

-- Define a function to sum the digits of a number
def sum_of_digits (m : ℕ) : ℕ := sorry

-- Theorem statement
theorem sum_of_digits_of_greatest_prime_divisor :
  sum_of_digits (greatest_prime_divisor n) = 13 := by sorry

end NUMINAMATH_CALUDE_sum_of_digits_of_greatest_prime_divisor_l2620_262067


namespace NUMINAMATH_CALUDE_complex_fraction_division_l2620_262045

theorem complex_fraction_division : 
  (5 / (8 / 13)) / (10 / 7) = 91 / 16 := by sorry

end NUMINAMATH_CALUDE_complex_fraction_division_l2620_262045


namespace NUMINAMATH_CALUDE_max_sum_of_pairwise_sums_l2620_262001

theorem max_sum_of_pairwise_sums (a b c d e : ℝ) 
  (h : (a + b) + (a + c) + (b + c) + (d + e) = 1096) :
  (a + d) + (a + e) + (b + d) + (b + e) + (c + d) + (c + e) ≤ 4384 := by
  sorry

end NUMINAMATH_CALUDE_max_sum_of_pairwise_sums_l2620_262001


namespace NUMINAMATH_CALUDE_ellipse_condition_l2620_262089

/-- The equation of the graph is 9x^2 + y^2 - 36x + 8y = k -/
def graph_equation (x y k : ℝ) : Prop :=
  9 * x^2 + y^2 - 36 * x + 8 * y = k

/-- A non-degenerate ellipse has positive denominators in its standard form -/
def is_non_degenerate_ellipse (k : ℝ) : Prop :=
  k + 52 > 0

theorem ellipse_condition (k : ℝ) :
  (∀ x y, graph_equation x y k → is_non_degenerate_ellipse k) ↔ k > -52 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_condition_l2620_262089


namespace NUMINAMATH_CALUDE_swimmers_pass_23_times_l2620_262014

/-- Represents the number of times two swimmers pass each other in a pool --/
def swimmers_passing_count (pool_length : ℝ) (speed_a speed_b : ℝ) (total_time : ℝ) : ℕ :=
  sorry

/-- Theorem stating the number of times swimmers pass each other under given conditions --/
theorem swimmers_pass_23_times :
  swimmers_passing_count 120 4 3 (15 * 60) = 23 := by
  sorry

end NUMINAMATH_CALUDE_swimmers_pass_23_times_l2620_262014


namespace NUMINAMATH_CALUDE_f_sum_symmetric_l2620_262031

/-- For the function f(x) = x + sin(x) + 1, f(x) + f(-x) = 2 for all real x -/
theorem f_sum_symmetric (x : ℝ) : let f : ℝ → ℝ := λ x ↦ x + Real.sin x + 1
  f x + f (-x) = 2 := by
  sorry

end NUMINAMATH_CALUDE_f_sum_symmetric_l2620_262031


namespace NUMINAMATH_CALUDE_b_range_l2620_262006

-- Define the quadratic equation
def quadratic (x b c : ℝ) : Prop := x^2 + 2*b*x + c = 0

-- Define the condition for roots in [-1, 1]
def roots_in_range (b c : ℝ) : Prop :=
  ∃ x : ℝ, x ∈ Set.Icc (-1) 1 ∧ quadratic x b c

-- Define the inequality condition
def inequality_condition (b c : ℝ) : Prop :=
  0 ≤ 4*b + c ∧ 4*b + c ≤ 3

-- Theorem statement
theorem b_range (b c : ℝ) :
  roots_in_range b c → inequality_condition b c → b ∈ Set.Icc (-1) 2 := by
  sorry


end NUMINAMATH_CALUDE_b_range_l2620_262006


namespace NUMINAMATH_CALUDE_student_score_l2620_262027

theorem student_score (max_score : ℕ) (pass_threshold : ℚ) (fail_margin : ℕ) (student_score : ℕ) : 
  max_score = 500 →
  pass_threshold = 33 / 100 →
  fail_margin = 40 →
  student_score = ⌊max_score * pass_threshold⌋ - fail_margin →
  student_score = 125 := by
sorry

end NUMINAMATH_CALUDE_student_score_l2620_262027


namespace NUMINAMATH_CALUDE_count_non_negative_rationals_l2620_262003

def rational_list : List ℚ := [-15, 5 + 1/3, -23/100, 0, 76/10, 2, -1/3, 314/100]

theorem count_non_negative_rationals :
  (rational_list.filter (λ x => x ≥ 0)).length = 5 := by
  sorry

end NUMINAMATH_CALUDE_count_non_negative_rationals_l2620_262003


namespace NUMINAMATH_CALUDE_quinary_444_equals_octal_174_l2620_262009

/-- Converts a quinary (base-5) number to decimal --/
def quinary_to_decimal (q : ℕ) : ℕ := sorry

/-- Converts a decimal number to octal --/
def decimal_to_octal (d : ℕ) : ℕ := sorry

/-- Theorem: The quinary number 444₅ is equal to the octal number 174₈ --/
theorem quinary_444_equals_octal_174 : 
  decimal_to_octal (quinary_to_decimal 444) = 174 := by sorry

end NUMINAMATH_CALUDE_quinary_444_equals_octal_174_l2620_262009


namespace NUMINAMATH_CALUDE_min_tan_product_l2620_262028

theorem min_tan_product (α β γ : Real) (h_acute : α ∈ Set.Ioo 0 (π/2) ∧ β ∈ Set.Ioo 0 (π/2) ∧ γ ∈ Set.Ioo 0 (π/2)) 
  (h_cos_sum : Real.cos α ^ 2 + Real.cos β ^ 2 + Real.cos γ ^ 2 = 1) :
  ∃ (min : Real), 
    (∀ (α' β' γ' : Real), 
      α' ∈ Set.Ioo 0 (π/2) → β' ∈ Set.Ioo 0 (π/2) → γ' ∈ Set.Ioo 0 (π/2) →
      Real.cos α' ^ 2 + Real.cos β' ^ 2 + Real.cos γ' ^ 2 = 1 →
      Real.tan α' * Real.tan β' * Real.tan γ' ≥ min) ∧
    Real.tan α * Real.tan β * Real.tan γ = min ∧
    min = 2 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_min_tan_product_l2620_262028


namespace NUMINAMATH_CALUDE_solve_equation_l2620_262047

theorem solve_equation (x : ℝ) (h : (128 / x) + (75 / x) + (57 / x) = 6.5) : x = 40 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l2620_262047


namespace NUMINAMATH_CALUDE_exists_diverse_line_l2620_262078

/-- Represents a 17x17 table with integers from 1 to 17 -/
def Table := Fin 17 → Fin 17 → Fin 17

/-- Predicate to check if a table is valid according to the problem conditions -/
def is_valid_table (t : Table) : Prop :=
  ∀ n : Fin 17, (Finset.univ.filter (λ (i : Fin 17 × Fin 17) => t i.1 i.2 = n)).card = 17

/-- Counts the number of different elements in a list -/
def count_different (l : List (Fin 17)) : Nat :=
  (l.toFinset).card

/-- Theorem stating the existence of a row or column with at least 5 different numbers -/
theorem exists_diverse_line (t : Table) (h : is_valid_table t) :
  (∃ i : Fin 17, count_different (List.ofFn (λ j => t i j)) ≥ 5) ∨
  (∃ j : Fin 17, count_different (List.ofFn (λ i => t i j)) ≥ 5) := by
  sorry

end NUMINAMATH_CALUDE_exists_diverse_line_l2620_262078


namespace NUMINAMATH_CALUDE_quadratic_sum_l2620_262079

/-- The quadratic function we're working with -/
def f (x : ℝ) : ℝ := 8*x^2 + 48*x + 200

/-- The general form of a quadratic after completing the square -/
def g (a b c x : ℝ) : ℝ := a*(x+b)^2 + c

theorem quadratic_sum (a b c : ℝ) : 
  (∀ x, f x = g a b c x) → a + b + c = 139 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_sum_l2620_262079


namespace NUMINAMATH_CALUDE_not_divisible_by_121_l2620_262055

theorem not_divisible_by_121 (n : ℤ) : ¬(∃ (k : ℤ), n^2 + 3*n + 5 = 121*k ∨ n^2 - 3*n + 5 = 121*k) := by
  sorry

end NUMINAMATH_CALUDE_not_divisible_by_121_l2620_262055


namespace NUMINAMATH_CALUDE_prob_odd_product_six_rolls_main_theorem_l2620_262052

/-- A standard die has six faces numbered 1 through 6 -/
def standardDie : Finset Nat := {1, 2, 3, 4, 5, 6}

/-- The probability of rolling an odd number on a standard die -/
def probOddRoll : Rat := 1/2

/-- The number of times the die is rolled -/
def numRolls : Nat := 6

/-- Theorem: The probability of rolling a standard die six times and obtaining an odd product is 1/64 -/
theorem prob_odd_product_six_rolls :
  (probOddRoll ^ numRolls : Rat) = 1/64 := by
  sorry

/-- Main theorem: The probability of rolling a standard die six times and obtaining an odd product is 1/64 -/
theorem main_theorem :
  ∃ (p : Rat), p = (probOddRoll ^ numRolls) ∧ p = 1/64 := by
  sorry

end NUMINAMATH_CALUDE_prob_odd_product_six_rolls_main_theorem_l2620_262052


namespace NUMINAMATH_CALUDE_polynomial_coefficient_properties_l2620_262030

theorem polynomial_coefficient_properties (a₀ a₁ a₂ a₃ a₄ a₅ a₆ : ℝ) :
  (∀ x, (x - 2)^6 = a₀ + a₁*x + a₂*x^2 + a₃*x^3 + a₄*x^4 + a₅*x^5 + a₆*x^6) →
  (a₄ = 60 ∧ a₁ + a₂ + a₃ + a₄ + a₅ + a₆ = -63) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_coefficient_properties_l2620_262030


namespace NUMINAMATH_CALUDE_larger_cube_volume_l2620_262020

theorem larger_cube_volume (n : ℕ) (small_cube_volume : ℝ) (surface_area_diff : ℝ) :
  n = 216 →
  small_cube_volume = 1 →
  surface_area_diff = 1080 →
  (n : ℝ) * 6 * small_cube_volume^(2/3) - 6 * ((n : ℝ) * small_cube_volume)^(2/3) = surface_area_diff →
  (n : ℝ) * small_cube_volume = 216 :=
by sorry

end NUMINAMATH_CALUDE_larger_cube_volume_l2620_262020


namespace NUMINAMATH_CALUDE_trailing_zeroes_1500_factorial_l2620_262017

def trailingZeroes (n : ℕ) : ℕ :=
  (n / 5) + (n / 25) + (n / 125) + (n / 625)

theorem trailing_zeroes_1500_factorial :
  trailingZeroes 1500 = 374 := by
  sorry

end NUMINAMATH_CALUDE_trailing_zeroes_1500_factorial_l2620_262017


namespace NUMINAMATH_CALUDE_remainder_97_pow_50_mod_100_l2620_262032

theorem remainder_97_pow_50_mod_100 : 97^50 % 100 = 49 := by
  sorry

end NUMINAMATH_CALUDE_remainder_97_pow_50_mod_100_l2620_262032


namespace NUMINAMATH_CALUDE_nine_powers_equal_three_power_l2620_262063

theorem nine_powers_equal_three_power (n : ℕ) : 
  9^n + 9^n + 9^n + 9^n + 9^n + 9^n + 9^n + 9^n + 9^n = 3^2012 → n = 1005 := by
  sorry

end NUMINAMATH_CALUDE_nine_powers_equal_three_power_l2620_262063


namespace NUMINAMATH_CALUDE_willys_work_problem_l2620_262039

/-- Willy's work problem -/
theorem willys_work_problem (total_days : ℕ) (daily_wage : ℚ) (daily_fine : ℚ) 
  (h_total_days : total_days = 30)
  (h_daily_wage : daily_wage = 8)
  (h_daily_fine : daily_fine = 10)
  (h_no_money_owed : ∃ (days_worked : ℚ), 
    0 ≤ days_worked ∧ 
    days_worked ≤ total_days ∧ 
    days_worked * daily_wage = (total_days - days_worked) * daily_fine) :
  ∃ (days_worked : ℚ) (days_missed : ℚ),
    days_worked = 50 / 3 ∧
    days_missed = 40 / 3 ∧
    days_worked + days_missed = total_days ∧
    days_worked * daily_wage = days_missed * daily_fine :=
sorry

end NUMINAMATH_CALUDE_willys_work_problem_l2620_262039


namespace NUMINAMATH_CALUDE_salt_mixture_problem_l2620_262069

/-- Proves that the amount of initial 20% salt solution is 30 ounces when mixed with 30 ounces of 60% salt solution to create a 40% salt solution. -/
theorem salt_mixture_problem (x : ℝ) :
  (0.20 * x + 0.60 * 30 = 0.40 * (x + 30)) → x = 30 := by
  sorry

end NUMINAMATH_CALUDE_salt_mixture_problem_l2620_262069


namespace NUMINAMATH_CALUDE_tangent_implies_a_equals_two_l2620_262002

noncomputable section

-- Define the line and curve equations
def line (x : ℝ) : ℝ := x + 1
def curve (a : ℝ) (x : ℝ) : ℝ := Real.log (x + a)

-- Define the tangency condition
def is_tangent (a : ℝ) : Prop :=
  ∃ x : ℝ, 
    line x = curve a x ∧ 
    (deriv (curve a)) x = (deriv line) x

-- Theorem statement
theorem tangent_implies_a_equals_two :
  ∀ a : ℝ, is_tangent a → a = 2 :=
sorry

end

end NUMINAMATH_CALUDE_tangent_implies_a_equals_two_l2620_262002


namespace NUMINAMATH_CALUDE_cube_volume_from_total_edge_length_l2620_262061

/-- Given a cube where the sum of the lengths of all edges is 48 cm, 
    prove that its volume is 64 cubic centimeters. -/
theorem cube_volume_from_total_edge_length : 
  ∀ (edge_length : ℝ), 
    (12 * edge_length = 48) →
    (edge_length^3 = 64) := by
  sorry

end NUMINAMATH_CALUDE_cube_volume_from_total_edge_length_l2620_262061


namespace NUMINAMATH_CALUDE_expression_value_l2620_262040

theorem expression_value : 
  let a : ℝ := 5
  let b : ℝ := 7
  let c : ℝ := 3
  (2*a - (3*b - 4*c)) - ((2*a - 3*b) - 4*c) = 24 := by
sorry

end NUMINAMATH_CALUDE_expression_value_l2620_262040


namespace NUMINAMATH_CALUDE_inequality_system_solution_set_l2620_262004

theorem inequality_system_solution_set :
  let S := {x : ℝ | x + 6 ≤ 8 ∧ x - 7 < 2 * (x - 3)}
  S = {x : ℝ | -1 < x ∧ x ≤ 2} := by
  sorry

end NUMINAMATH_CALUDE_inequality_system_solution_set_l2620_262004


namespace NUMINAMATH_CALUDE_gcf_lcm_sum_8_12_l2620_262053

theorem gcf_lcm_sum_8_12 : Nat.gcd 8 12 + Nat.lcm 8 12 = 28 := by
  sorry

end NUMINAMATH_CALUDE_gcf_lcm_sum_8_12_l2620_262053


namespace NUMINAMATH_CALUDE_supplement_of_complement_of_75_degrees_l2620_262072

/-- Given a 75-degree angle, prove that the degree measure of the supplement of its complement is 165°. -/
theorem supplement_of_complement_of_75_degrees :
  let angle : ℝ := 75
  let complement : ℝ := 90 - angle
  let supplement : ℝ := 180 - complement
  supplement = 165 := by
  sorry

end NUMINAMATH_CALUDE_supplement_of_complement_of_75_degrees_l2620_262072


namespace NUMINAMATH_CALUDE_greatest_divisor_four_consecutive_integers_l2620_262075

theorem greatest_divisor_four_consecutive_integers :
  ∀ n : ℕ, n > 0 →
  ∃ m : ℕ, m = 12 ∧ 
  (∀ k : ℕ, k > m → ¬(k ∣ (n * (n + 1) * (n + 2) * (n + 3)))) ∧
  (12 ∣ (n * (n + 1) * (n + 2) * (n + 3))) := by
sorry

end NUMINAMATH_CALUDE_greatest_divisor_four_consecutive_integers_l2620_262075


namespace NUMINAMATH_CALUDE_arithmetic_mean_of_fractions_l2620_262046

theorem arithmetic_mean_of_fractions :
  let f1 : ℚ := 3 / 8
  let f2 : ℚ := 5 / 9
  let f3 : ℚ := 7 / 12
  let mean : ℚ := (f1 + f2 + f3) / 3
  mean = 109 / 216 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_of_fractions_l2620_262046


namespace NUMINAMATH_CALUDE_interest_rate_calculation_l2620_262057

/-- Given a principal amount and an interest rate, proves that if the simple interest
    for 2 years is 40 and the compound interest for 2 years is 41, then the interest rate is 5% -/
theorem interest_rate_calculation (P : ℝ) (r : ℝ) 
    (h1 : P * r * 2 = 40)  -- Simple interest condition
    (h2 : P * ((1 + r)^2 - 1) = 41) -- Compound interest condition
    : r = 0.05 := by
  sorry

#check interest_rate_calculation

end NUMINAMATH_CALUDE_interest_rate_calculation_l2620_262057


namespace NUMINAMATH_CALUDE_complement_of_A_relative_to_U_l2620_262013

-- Define the universal set U
def U : Set ℝ := {x | x < 3}

-- Define the subset A
def A : Set ℝ := {x | x < 1}

-- Define the complement of A relative to U
def complement_U_A : Set ℝ := {x ∈ U | x ∉ A}

-- Theorem statement
theorem complement_of_A_relative_to_U :
  complement_U_A = {x | 1 ≤ x ∧ x < 3} :=
sorry

end NUMINAMATH_CALUDE_complement_of_A_relative_to_U_l2620_262013


namespace NUMINAMATH_CALUDE_consecutive_product_not_power_l2620_262081

theorem consecutive_product_not_power (n m : ℕ) (h : m > 1) :
  ¬ ∃ k : ℕ, (n - 1) * n * (n + 1) = k ^ m := by
  sorry

end NUMINAMATH_CALUDE_consecutive_product_not_power_l2620_262081


namespace NUMINAMATH_CALUDE_parabola_properties_l2620_262058

def is_valid_parabola (a b c : ℝ) : Prop :=
  a ≠ 0 ∧
  a * (-1)^2 + b * (-1) + c = -1 ∧
  c = 1 ∧
  a * (-2)^2 + b * (-2) + c > 1

theorem parabola_properties (a b c : ℝ) 
  (h : is_valid_parabola a b c) : 
  a * b * c > 0 ∧ 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ a * x₁^2 + b * x₁ + c - 3 = 0 ∧ a * x₂^2 + b * x₂ + c - 3 = 0) ∧
  a + b + c > 7 := by
  sorry

end NUMINAMATH_CALUDE_parabola_properties_l2620_262058


namespace NUMINAMATH_CALUDE_intersection_x_product_l2620_262083

/-- Given a line y = mx + k and a parabola y = ax² + bx + c that intersect at two points,
    the product of the x-coordinates of these intersection points is equal to (c - k) / a. -/
theorem intersection_x_product (a m b c k : ℝ) (ha : a ≠ 0) :
  let f (x : ℝ) := a * x^2 + b * x + c
  let g (x : ℝ) := m * x + k
  let h (x : ℝ) := f x - g x
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ h x₁ = 0 ∧ h x₂ = 0 →
  x₁ * x₂ = (c - k) / a :=
sorry

end NUMINAMATH_CALUDE_intersection_x_product_l2620_262083


namespace NUMINAMATH_CALUDE_intersection_length_l2620_262005

-- Define the line l
def line_l (t : ℝ) : ℝ × ℝ := (1 + t, 2 + t)

-- Define the circle C in polar form
def circle_C (ρ θ : ℝ) : Prop := ρ^2 + 2*ρ*(Real.sin θ) = 3

-- Define the intersection points
def intersection_points (l : ℝ → ℝ × ℝ) (C : ℝ → ℝ → Prop) : Set (ℝ × ℝ) :=
  {p | ∃ t, l t = p ∧ ∃ ρ θ, C ρ θ ∧ p.1 = ρ * (Real.cos θ) ∧ p.2 = ρ * (Real.sin θ)}

-- Theorem statement
theorem intersection_length :
  let points := intersection_points line_l circle_C
  ∃ M N : ℝ × ℝ, M ∈ points ∧ N ∈ points ∧ M ≠ N ∧
    Real.sqrt ((M.1 - N.1)^2 + (M.2 - N.2)^2) = 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_intersection_length_l2620_262005


namespace NUMINAMATH_CALUDE_great_pyramid_dimensions_l2620_262059

/-- The Great Pyramid of Giza's dimensions and sum of height and width -/
theorem great_pyramid_dimensions :
  let height := 500 + 20
  let width := height + 234
  height + width = 1274 := by sorry

end NUMINAMATH_CALUDE_great_pyramid_dimensions_l2620_262059


namespace NUMINAMATH_CALUDE_inequality_proof_l2620_262042

theorem inequality_proof (a b c : ℝ) (h1 : a > b) (h2 : b > c) :
  a^2 * (b - c) + b^2 * (c - a) + c^2 * (a - b) > 0 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2620_262042


namespace NUMINAMATH_CALUDE_flock_max_weight_l2620_262064

/-- Represents the types of swallows --/
inductive SwallowType
| American
| European

/-- Calculates the maximum weight a swallow can carry based on its type --/
def maxWeightCarried (s : SwallowType) : ℕ :=
  match s with
  | SwallowType.American => 5
  | SwallowType.European => 10

/-- The total number of swallows in the flock --/
def totalSwallows : ℕ := 90

/-- The ratio of American to European swallows --/
def americanToEuropeanRatio : ℕ := 2

/-- Theorem stating the maximum combined weight the flock can carry --/
theorem flock_max_weight :
  let europeanCount := totalSwallows / (americanToEuropeanRatio + 1)
  let americanCount := totalSwallows - europeanCount
  europeanCount * maxWeightCarried SwallowType.European +
  americanCount * maxWeightCarried SwallowType.American = 600 := by
  sorry


end NUMINAMATH_CALUDE_flock_max_weight_l2620_262064


namespace NUMINAMATH_CALUDE_earth_inhabitable_fraction_l2620_262084

theorem earth_inhabitable_fraction :
  let earth_surface := 1
  let land_fraction := (1 : ℚ) / 3
  let inhabitable_land_fraction := (1 : ℚ) / 3
  (land_fraction * inhabitable_land_fraction) * earth_surface = (1 : ℚ) / 9 := by
  sorry

end NUMINAMATH_CALUDE_earth_inhabitable_fraction_l2620_262084


namespace NUMINAMATH_CALUDE_cylinder_cone_surface_area_l2620_262049

/-- The total surface area of a cylinder topped with a cone -/
theorem cylinder_cone_surface_area (h_cyl h_cone r : ℝ) (h_cyl_pos : h_cyl > 0) (h_cone_pos : h_cone > 0) (r_pos : r > 0) :
  let cylinder_base_area := π * r^2
  let cylinder_lateral_area := 2 * π * r * h_cyl
  let cone_slant_height := Real.sqrt (r^2 + h_cone^2)
  let cone_lateral_area := π * r * cone_slant_height
  cylinder_base_area + cylinder_lateral_area + cone_lateral_area = 175 * π + 5 * π * Real.sqrt 89 :=
by sorry

end NUMINAMATH_CALUDE_cylinder_cone_surface_area_l2620_262049


namespace NUMINAMATH_CALUDE_f_15_equals_227_l2620_262041

/-- Given a function f(n) = n^2 - n + 17, prove that f(15) = 227 -/
theorem f_15_equals_227 (f : ℕ → ℕ) (h : ∀ n, f n = n^2 - n + 17) : f 15 = 227 := by
  sorry

end NUMINAMATH_CALUDE_f_15_equals_227_l2620_262041


namespace NUMINAMATH_CALUDE_stock_price_after_two_years_l2620_262065

def initial_price : ℝ := 200
def first_year_increase : ℝ := 0.50
def second_year_decrease : ℝ := 0.30

theorem stock_price_after_two_years :
  let price_after_first_year := initial_price * (1 + first_year_increase)
  let final_price := price_after_first_year * (1 - second_year_decrease)
  final_price = 210 := by sorry

end NUMINAMATH_CALUDE_stock_price_after_two_years_l2620_262065


namespace NUMINAMATH_CALUDE_largest_digit_divisible_by_six_l2620_262007

theorem largest_digit_divisible_by_six :
  ∃ (N : ℕ), N ≤ 9 ∧ (3456 * 10 + N) % 6 = 0 ∧
  ∀ (M : ℕ), M ≤ 9 ∧ (3456 * 10 + M) % 6 = 0 → M ≤ N :=
by sorry

end NUMINAMATH_CALUDE_largest_digit_divisible_by_six_l2620_262007


namespace NUMINAMATH_CALUDE_equal_intercept_line_properties_l2620_262062

/-- A line passing through (1, 2) with equal intercepts on both axes -/
def equal_intercept_line : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 + p.2 = 3}

theorem equal_intercept_line_properties :
  (1, 2) ∈ equal_intercept_line ∧
  ∃ a : ℝ, a ≠ 0 ∧ (a, 0) ∈ equal_intercept_line ∧ (0, a) ∈ equal_intercept_line :=
by sorry

end NUMINAMATH_CALUDE_equal_intercept_line_properties_l2620_262062


namespace NUMINAMATH_CALUDE_minsu_running_time_l2620_262060

theorem minsu_running_time 
  (total_distance : Real) 
  (speed : Real) 
  (distance_remaining : Real) : Real :=
  let distance_run := total_distance - distance_remaining
  let time_elapsed := distance_run / speed
  have h1 : total_distance = 120 := by sorry
  have h2 : speed = 4 := by sorry
  have h3 : distance_remaining = 20 := by sorry
  have h4 : time_elapsed = 25 := by sorry
  time_elapsed

#check minsu_running_time

end NUMINAMATH_CALUDE_minsu_running_time_l2620_262060


namespace NUMINAMATH_CALUDE_line_intersects_ellipse_l2620_262044

/-- The line kx+y+k+1=0 intersects the ellipse x^2/25 + y^2/16 = 1 for all real values of k -/
theorem line_intersects_ellipse (k : ℝ) : ∃ (x y : ℝ), 
  (k * x + y + k + 1 = 0) ∧ (x^2 / 25 + y^2 / 16 = 1) := by sorry

end NUMINAMATH_CALUDE_line_intersects_ellipse_l2620_262044


namespace NUMINAMATH_CALUDE_john_needs_two_sets_l2620_262019

/-- The number of metal bars in each set -/
def bars_per_set : ℕ := 7

/-- The total number of metal bars -/
def total_bars : ℕ := 14

/-- The number of sets of metal bars John needs -/
def sets_needed : ℕ := total_bars / bars_per_set

theorem john_needs_two_sets : sets_needed = 2 := by
  sorry

end NUMINAMATH_CALUDE_john_needs_two_sets_l2620_262019


namespace NUMINAMATH_CALUDE_maxwells_walking_speed_l2620_262098

/-- Proves that Maxwell's walking speed is 3 km/h given the problem conditions --/
theorem maxwells_walking_speed 
  (total_distance : ℝ) 
  (maxwell_distance : ℝ) 
  (brad_speed : ℝ) 
  (h1 : total_distance = 36) 
  (h2 : maxwell_distance = 12) 
  (h3 : brad_speed = 6) : 
  maxwell_distance / (total_distance - maxwell_distance) * brad_speed = 3 := by
  sorry

end NUMINAMATH_CALUDE_maxwells_walking_speed_l2620_262098


namespace NUMINAMATH_CALUDE_money_ratio_proof_l2620_262086

theorem money_ratio_proof (natasha_money carla_money cosima_money : ℚ) :
  natasha_money = 3 * carla_money →
  carla_money = cosima_money →
  natasha_money = 60 →
  (7 / 5) * (natasha_money + carla_money + cosima_money) - (natasha_money + carla_money + cosima_money) = 36 →
  carla_money / cosima_money = 1 := by
  sorry

end NUMINAMATH_CALUDE_money_ratio_proof_l2620_262086


namespace NUMINAMATH_CALUDE_tan_alpha_value_l2620_262024

theorem tan_alpha_value (α : Real) (h : Real.tan α = -1/2) :
  (1 + 2 * Real.sin α * Real.cos α) / (Real.sin α ^ 2 - Real.cos α ^ 2) = -1/3 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_value_l2620_262024


namespace NUMINAMATH_CALUDE_complex_absolute_value_l2620_262021

/-- Given a complex number z such that (1 + 2i) / z = 1 + i,
    prove that the absolute value of z is equal to √10 / 2. -/
theorem complex_absolute_value (z : ℂ) (h : (1 + 2 * Complex.I) / z = 1 + Complex.I) :
  Complex.abs z = Real.sqrt 10 / 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_absolute_value_l2620_262021


namespace NUMINAMATH_CALUDE_garden_area_l2620_262016

theorem garden_area (length_distance width_distance : ℝ) 
  (h1 : length_distance * 30 = 1500)
  (h2 : (2 * length_distance + 2 * width_distance) * 12 = 1500) :
  length_distance * width_distance = 625 := by
  sorry

end NUMINAMATH_CALUDE_garden_area_l2620_262016


namespace NUMINAMATH_CALUDE_a_gt_one_sufficient_not_necessary_l2620_262087

theorem a_gt_one_sufficient_not_necessary (a : ℝ) (h : a ≠ 0) :
  (∀ a, a > 1 → a > 1/a) ∧ (∃ a, a > 1/a ∧ a ≤ 1) := by sorry

end NUMINAMATH_CALUDE_a_gt_one_sufficient_not_necessary_l2620_262087


namespace NUMINAMATH_CALUDE_algebraic_expression_value_l2620_262034

theorem algebraic_expression_value (a b : ℝ) (h1 : a = 3) (h2 : a - b = 1) : a^2 - a*b = 3 := by
  sorry

end NUMINAMATH_CALUDE_algebraic_expression_value_l2620_262034


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l2620_262051

/-- Represents an arithmetic sequence -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  d : ℝ      -- Common difference
  h : ∀ n, a (n + 1) = a n + d

/-- Sum of the first n terms of an arithmetic sequence -/
def S (seq : ArithmeticSequence) : ℕ → ℝ
  | 0 => 0
  | n + 1 => S seq n + seq.a (n + 1)

theorem arithmetic_sequence_property (seq : ArithmeticSequence) (m : ℕ) 
  (h1 : S seq (m - 1) = -2)
  (h2 : S seq m = 0)
  (h3 : S seq (m + 1) = 3) :
  seq.d = 1 ∧ m = 5 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l2620_262051


namespace NUMINAMATH_CALUDE_shobha_current_age_l2620_262035

/-- Given the ratio of Shekhar's age to Shobha's age and Shekhar's future age, 
    prove Shobha's current age -/
theorem shobha_current_age 
  (shekhar_age shobha_age : ℕ) 
  (age_ratio : shekhar_age / shobha_age = 4 / 3) 
  (shekhar_future_age : shekhar_age + 6 = 26) : 
  shobha_age = 15 := by
sorry

end NUMINAMATH_CALUDE_shobha_current_age_l2620_262035


namespace NUMINAMATH_CALUDE_Ba_atomic_weight_l2620_262018

def atomic_weight_Ba (molecular_weight : ℝ) (atomic_weight_O : ℝ) : ℝ :=
  molecular_weight - atomic_weight_O

theorem Ba_atomic_weight :
  let molecular_weight : ℝ := 153
  let atomic_weight_O : ℝ := 16
  atomic_weight_Ba molecular_weight atomic_weight_O = 137 := by
sorry

end NUMINAMATH_CALUDE_Ba_atomic_weight_l2620_262018


namespace NUMINAMATH_CALUDE_floor_power_divisibility_l2620_262000

theorem floor_power_divisibility (n : ℕ) : 
  (2^(n+1) : ℤ) ∣ ⌊(1 + Real.sqrt 3)^(2*n + 1)⌋ := by
  sorry

end NUMINAMATH_CALUDE_floor_power_divisibility_l2620_262000


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l2620_262011

/-- Given sets M and N, prove their intersection -/
theorem intersection_of_M_and_N :
  let M := {x : ℝ | |x - 1| < 2}
  let N := {x : ℝ | x * (x - 3) < 0}
  M ∩ N = {x : ℝ | 0 < x ∧ x < 3} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l2620_262011


namespace NUMINAMATH_CALUDE_triangle_centroid_product_l2620_262050

theorem triangle_centroid_product (AP PD BP PE CP PF : ℝ) 
  (h : AP / PD + BP / PE + CP / PF = 90) : 
  AP / PD * BP / PE * CP / PF = 94 := by
  sorry

end NUMINAMATH_CALUDE_triangle_centroid_product_l2620_262050


namespace NUMINAMATH_CALUDE_min_diagonal_pairs_l2620_262026

/-- Represents a triangle of cells arranged in rows -/
structure CellTriangle where
  rows : ℕ

/-- Calculates the total number of cells in the triangle -/
def totalCells (t : CellTriangle) : ℕ :=
  t.rows * (t.rows + 1) / 2

/-- Calculates the number of rows with an odd number of cells -/
def oddRows (t : CellTriangle) : ℕ :=
  t.rows / 2

/-- Theorem: The minimum number of diagonal pairs in a cell triangle
    with 5784 rows is equal to the number of rows with an odd number of cells -/
theorem min_diagonal_pairs (t : CellTriangle) (h : t.rows = 5784) :
  oddRows t = 2892 := by sorry

end NUMINAMATH_CALUDE_min_diagonal_pairs_l2620_262026


namespace NUMINAMATH_CALUDE_round_trip_average_speed_l2620_262074

/-- Calculates the average speed for a round trip given uphill speed, uphill time, and downhill time -/
theorem round_trip_average_speed 
  (uphill_speed : ℝ) 
  (uphill_time : ℝ) 
  (downhill_time : ℝ) 
  (h1 : uphill_speed = 2.5) 
  (h2 : uphill_time = 3) 
  (h3 : downhill_time = 2) : 
  (2 * uphill_speed * uphill_time) / (uphill_time + downhill_time) = 3 := by
  sorry

#check round_trip_average_speed

end NUMINAMATH_CALUDE_round_trip_average_speed_l2620_262074


namespace NUMINAMATH_CALUDE_concyclicity_equivalence_l2620_262054

-- Define the points
variable (A B C D P E F G H O₁ O₂ O₃ O₄ : EuclideanPlane)

-- Define the quadrilateral ABCD
def is_convex_quadrilateral (A B C D : EuclideanPlane) : Prop := sorry

-- Define the intersection of diagonals
def diagonals_intersect_at (A B C D P : EuclideanPlane) : Prop := sorry

-- Define midpoints
def is_midpoint (M A B : EuclideanPlane) : Prop := sorry

-- Define circumcenter
def is_circumcenter (O P Q R : EuclideanPlane) : Prop := sorry

-- Define concyclicity
def are_concyclic (P Q R S : EuclideanPlane) : Prop := sorry

-- Main theorem
theorem concyclicity_equivalence 
  (h_quad : is_convex_quadrilateral A B C D)
  (h_diag : diagonals_intersect_at A B C D P)
  (h_mid_E : is_midpoint E A B)
  (h_mid_F : is_midpoint F B C)
  (h_mid_G : is_midpoint G C D)
  (h_mid_H : is_midpoint H D A)
  (h_circ_O₁ : is_circumcenter O₁ P H E)
  (h_circ_O₂ : is_circumcenter O₂ P E F)
  (h_circ_O₃ : is_circumcenter O₃ P F G)
  (h_circ_O₄ : is_circumcenter O₄ P G H) :
  are_concyclic O₁ O₂ O₃ O₄ ↔ are_concyclic A B C D := by sorry

end NUMINAMATH_CALUDE_concyclicity_equivalence_l2620_262054


namespace NUMINAMATH_CALUDE_distance_center_to_point_l2620_262038

/-- Given a circle with polar equation ρ = 4cosθ and a point P with polar coordinates (4, π/3),
    prove that the distance between the center of the circle and point P is 2√3. -/
theorem distance_center_to_point (θ : Real) (ρ : Real → Real) (P : Real × Real) :
  (ρ = fun θ => 4 * Real.cos θ) →
  P = (4, Real.pi / 3) →
  ∃ C : Real × Real, 
    (C.1 - P.1)^2 + (C.2 - P.2)^2 = 12 :=
by sorry

end NUMINAMATH_CALUDE_distance_center_to_point_l2620_262038


namespace NUMINAMATH_CALUDE_solution_set_x_squared_leq_four_l2620_262094

theorem solution_set_x_squared_leq_four :
  {x : ℝ | x^2 ≤ 4} = {x : ℝ | -2 ≤ x ∧ x ≤ 2} := by
  sorry

end NUMINAMATH_CALUDE_solution_set_x_squared_leq_four_l2620_262094


namespace NUMINAMATH_CALUDE_committee_selection_l2620_262085

theorem committee_selection (n : ℕ) (k : ℕ) (h1 : n = 12) (h2 : k = 5) :
  Nat.choose n k = 792 := by
  sorry

end NUMINAMATH_CALUDE_committee_selection_l2620_262085


namespace NUMINAMATH_CALUDE_john_share_l2620_262071

def total_amount : ℕ := 6000
def john_ratio : ℕ := 2
def jose_ratio : ℕ := 4
def binoy_ratio : ℕ := 6

theorem john_share :
  let total_ratio := john_ratio + jose_ratio + binoy_ratio
  (john_ratio : ℚ) / total_ratio * total_amount = 1000 := by sorry

end NUMINAMATH_CALUDE_john_share_l2620_262071


namespace NUMINAMATH_CALUDE_divisibility_property_l2620_262097

theorem divisibility_property (n : ℕ) : 
  ∃ (a b : ℕ), (a * n + 1)^6 + b ≡ 0 [MOD (n^2 + n + 1)] :=
by
  use 2, 27
  sorry

end NUMINAMATH_CALUDE_divisibility_property_l2620_262097


namespace NUMINAMATH_CALUDE_triangle_isosceles_from_equation_l2620_262023

/-- A triangle with sides a, b, and c is isosceles if two of its sides are equal. -/
def IsIsosceles (a b c : ℝ) : Prop :=
  a = b ∨ b = c ∨ a = c

/-- 
  Theorem: If the three sides a, b, c of a triangle ABC satisfy a²-ac-b²+bc=0, 
  then the triangle is isosceles.
-/
theorem triangle_isosceles_from_equation 
  (a b c : ℝ) 
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0) 
  (h_triangle : a + b > c ∧ b + c > a ∧ a + c > b) 
  (h_eq : a^2 - a*c - b^2 + b*c = 0) : 
  IsIsosceles a b c := by
  sorry


end NUMINAMATH_CALUDE_triangle_isosceles_from_equation_l2620_262023


namespace NUMINAMATH_CALUDE_sunglasses_cap_probability_l2620_262025

theorem sunglasses_cap_probability (total_sunglasses : ℕ) (total_caps : ℕ) 
  (prob_sunglasses_given_cap : ℚ) :
  total_sunglasses = 70 →
  total_caps = 45 →
  prob_sunglasses_given_cap = 3/9 →
  (prob_sunglasses_given_cap * total_caps : ℚ) / total_sunglasses = 3/14 :=
by sorry

end NUMINAMATH_CALUDE_sunglasses_cap_probability_l2620_262025


namespace NUMINAMATH_CALUDE_watchtower_probability_l2620_262029

/-- Represents a searchlight with a given rotation speed in revolutions per minute -/
structure Searchlight where
  speed : ℝ
  speed_positive : speed > 0

/-- The setup of the watchtower problem -/
structure WatchtowerSetup where
  searchlight1 : Searchlight
  searchlight2 : Searchlight
  searchlight3 : Searchlight
  path_time : ℝ
  sl1_speed : searchlight1.speed = 2
  sl2_speed : searchlight2.speed = 3
  sl3_speed : searchlight3.speed = 4
  path_time_value : path_time = 30

/-- The probability of all searchlights not completing a revolution within the given time is 0 -/
theorem watchtower_probability (setup : WatchtowerSetup) :
  ∃ (s : Searchlight), s ∈ [setup.searchlight1, setup.searchlight2, setup.searchlight3] ∧
  (60 / s.speed ≤ setup.path_time) :=
sorry

end NUMINAMATH_CALUDE_watchtower_probability_l2620_262029


namespace NUMINAMATH_CALUDE_sector_area_l2620_262068

/-- Given a circular sector with central angle 120° and arc length 6π, its area is 27π -/
theorem sector_area (θ : ℝ) (arc_length : ℝ) (area : ℝ) : 
  θ = 120 * π / 180 →  -- Convert 120° to radians
  arc_length = 6 * π → 
  area = 27 * π :=
by
  sorry


end NUMINAMATH_CALUDE_sector_area_l2620_262068


namespace NUMINAMATH_CALUDE_sin_minus_cos_for_specific_tan_l2620_262091

theorem sin_minus_cos_for_specific_tan (θ : Real) 
  (h1 : θ ∈ Set.Ioo 0 (π / 2)) 
  (h2 : Real.tan θ = 1 / 3) : 
  Real.sin θ - Real.cos θ = -Real.sqrt 10 / 5 := by
  sorry

end NUMINAMATH_CALUDE_sin_minus_cos_for_specific_tan_l2620_262091
