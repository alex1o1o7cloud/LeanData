import Mathlib

namespace frog_max_hop_sum_l4094_409421

/-- The maximum sum of hop lengths for a frog hopping on integers -/
theorem frog_max_hop_sum (n : ℕ+) : 
  ∃ (S : ℕ), S = (4^n.val - 1) / 3 ∧ 
  ∀ (hop_lengths : List ℕ), 
    (∀ l ∈ hop_lengths, ∃ k : ℕ, l = 2^k) →
    (∀ p ∈ List.range (2^n.val), List.count p (List.scanl (λ acc x => (acc + x) % (2^n.val)) 0 hop_lengths) ≤ 1) →
    List.sum hop_lengths ≤ S :=
sorry

end frog_max_hop_sum_l4094_409421


namespace solve_linear_equation_l4094_409485

theorem solve_linear_equation (x : ℝ) : 3 * x + 7 = -2 → x = -3 := by
  sorry

end solve_linear_equation_l4094_409485


namespace shelter_dogs_l4094_409453

theorem shelter_dogs (dogs cats : ℕ) : 
  (dogs : ℚ) / cats = 15 / 7 →
  dogs / (cats + 16) = 15 / 11 →
  dogs = 60 :=
by sorry

end shelter_dogs_l4094_409453


namespace reciprocal_expression_l4094_409474

theorem reciprocal_expression (a b : ℝ) (h : a * b = 1) :
  a^2 * b - (a - 2023) = 2023 := by sorry

end reciprocal_expression_l4094_409474


namespace two_digit_cube_l4094_409417

theorem two_digit_cube (x : ℕ) : x = 93 ↔ 
  (x ≥ 10 ∧ x < 100) ∧ 
  (∃ n : ℕ, 101010 * x + 1 = n^3) ∧
  (101010 * x + 1 ≥ 1000000 ∧ 101010 * x + 1 < 10000000) := by
sorry

end two_digit_cube_l4094_409417


namespace ballpoint_pen_price_relation_l4094_409462

/-- Proves the relationship between price and number of pens for a specific box of ballpoint pens -/
theorem ballpoint_pen_price_relation :
  let box_pens : ℕ := 16
  let box_price : ℚ := 24
  let unit_price : ℚ := box_price / box_pens
  ∀ (x : ℚ) (y : ℚ), y = unit_price * x → y = (3/2 : ℚ) * x := by
  sorry

end ballpoint_pen_price_relation_l4094_409462


namespace fraction_simplification_l4094_409428

theorem fraction_simplification : 
  ((2^1004)^2 - (2^1002)^2) / ((2^1003)^2 - (2^1001)^2) = 4 := by
  sorry

end fraction_simplification_l4094_409428


namespace robie_has_five_boxes_l4094_409479

/-- Calculates the number of boxes Robie has left after giving some away -/
def robies_boxes (total_cards : ℕ) (cards_per_box : ℕ) (unboxed_cards : ℕ) (boxes_given_away : ℕ) : ℕ :=
  ((total_cards - unboxed_cards) / cards_per_box) - boxes_given_away

/-- Theorem stating that Robie has 5 boxes left given the initial conditions -/
theorem robie_has_five_boxes :
  robies_boxes 75 10 5 2 = 5 := by
  sorry

end robie_has_five_boxes_l4094_409479


namespace sum_of_cubics_degree_at_most_3_l4094_409446

-- Define a cubic polynomial
def CubicPolynomial (R : Type*) [CommRing R] := {p : Polynomial R // p.degree ≤ 3}

-- Theorem statement
theorem sum_of_cubics_degree_at_most_3 {R : Type*} [CommRing R] 
  (A B : CubicPolynomial R) : 
  (A.val + B.val).degree ≤ 3 := by
  sorry

end sum_of_cubics_degree_at_most_3_l4094_409446


namespace probability_at_least_one_defective_l4094_409470

theorem probability_at_least_one_defective (total : ℕ) (defective : ℕ) (chosen : ℕ) 
  (h1 : total = 20) (h2 : defective = 4) (h3 : chosen = 2) :
  (1 : ℚ) - (Nat.choose (total - defective) chosen : ℚ) / (Nat.choose total chosen : ℚ) = 7/19 := by
  sorry

end probability_at_least_one_defective_l4094_409470


namespace negative_square_cubed_l4094_409457

theorem negative_square_cubed (a : ℝ) : (-a^2)^3 = -a^6 := by
  sorry

end negative_square_cubed_l4094_409457


namespace exists_determining_question_l4094_409402

/-- Represents the type of being (Human or Zombie) --/
inductive Being
| Human
| Zombie

/-- Represents a possible response to a question --/
inductive Response
| Bal
| Yes
| No

/-- Represents a question that can be asked --/
def Question := Being → Response

/-- A function that determines the type of being based on a response --/
def DetermineBeing := Response → Being

/-- Humans always tell the truth, zombies always lie --/
axiom truth_telling (q : Question) :
  ∀ (b : Being), 
    (b = Being.Human → q b = Response.Bal) ∧
    (b = Being.Zombie → q b ≠ Response.Bal)

/-- There exists a question that can determine the type of being --/
theorem exists_determining_question :
  ∃ (q : Question) (d : DetermineBeing),
    ∀ (b : Being), d (q b) = b :=
sorry

end exists_determining_question_l4094_409402


namespace x_and_y_negative_l4094_409403

theorem x_and_y_negative (x y : ℝ) 
  (h1 : 2 * x - 3 * y > x) 
  (h2 : x + 4 * y < 3 * y) : 
  x < 0 ∧ y < 0 := by
  sorry

end x_and_y_negative_l4094_409403


namespace sum_greater_than_double_l4094_409411

theorem sum_greater_than_double (a b : ℝ) (h1 : a > b) (h2 : b > 0) : a + b > 2*b := by
  sorry

end sum_greater_than_double_l4094_409411


namespace base_5_reversed_in_base_7_l4094_409487

/-- Converts a base 5 number to base 10 -/
def toBase10FromBase5 (a b c : Nat) : Nat :=
  25 * a + 5 * b + c

/-- Converts a base 7 number to base 10 -/
def toBase10FromBase7 (a b c : Nat) : Nat :=
  49 * c + 7 * b + a

/-- Checks if a number is a valid digit in base 5 -/
def isValidBase5Digit (n : Nat) : Prop :=
  n ≤ 4

theorem base_5_reversed_in_base_7 :
  ∃! (a₁ b₁ c₁ a₂ b₂ c₂ : Nat),
    isValidBase5Digit a₁ ∧ isValidBase5Digit b₁ ∧ isValidBase5Digit c₁ ∧
    isValidBase5Digit a₂ ∧ isValidBase5Digit b₂ ∧ isValidBase5Digit c₂ ∧
    toBase10FromBase5 a₁ b₁ c₁ = toBase10FromBase7 c₁ b₁ a₁ ∧
    toBase10FromBase5 a₂ b₂ c₂ = toBase10FromBase7 c₂ b₂ a₂ ∧
    a₁ ≠ 0 ∧ a₂ ≠ 0 ∧
    toBase10FromBase5 a₁ b₁ c₁ + toBase10FromBase5 a₂ b₂ c₂ = 153 :=
  sorry

end base_5_reversed_in_base_7_l4094_409487


namespace log8_three_point_five_equals_512_l4094_409489

-- Define the logarithm base 8
noncomputable def log8 (x : ℝ) : ℝ := Real.log x / Real.log 8

-- State the theorem
theorem log8_three_point_five_equals_512 :
  ∀ x : ℝ, x > 0 → log8 x = 3.5 → x = 512 := by
  sorry

end log8_three_point_five_equals_512_l4094_409489


namespace greatest_common_divisor_with_digit_sum_l4094_409435

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sum_of_digits (n / 10)

def has_same_remainder (a b n : ℕ) : Prop :=
  a % n = b % n

theorem greatest_common_divisor_with_digit_sum (a b : ℕ) :
  ∃ (n : ℕ), n > 0 ∧
  n ∣ (a - b) ∧
  has_same_remainder a b n ∧
  sum_of_digits n = 4 ∧
  (∀ m : ℕ, m > n → m ∣ (a - b) → sum_of_digits m ≠ 4) →
  1120 ∣ n ∧ ∀ k : ℕ, k < 1120 → ¬(n ∣ k) :=
sorry

end greatest_common_divisor_with_digit_sum_l4094_409435


namespace nested_square_root_equality_l4094_409490

theorem nested_square_root_equality : Real.sqrt (25 * Real.sqrt (15 * Real.sqrt 9)) = 5 * Real.sqrt 15 := by
  sorry

end nested_square_root_equality_l4094_409490


namespace expression_factorization_l4094_409477

theorem expression_factorization (x : ℝ) :
  (12 * x^3 + 90 * x - 6) - (-3 * x^3 + 5 * x - 6) = 5 * x * (3 * x^2 + 17) :=
by sorry

end expression_factorization_l4094_409477


namespace min_value_sum_reciprocals_l4094_409451

theorem min_value_sum_reciprocals (a b c : ℝ) 
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_sum : a + b + c = 2) : 
  (1 / (2*a + b) + 1 / (2*b + c) + 1 / (2*c + a)) ≥ 27/8 ∧ 
  ∃ (a₀ b₀ c₀ : ℝ), 0 < a₀ ∧ 0 < b₀ ∧ 0 < c₀ ∧ a₀ + b₀ + c₀ = 2 ∧
    1 / (2*a₀ + b₀) + 1 / (2*b₀ + c₀) + 1 / (2*c₀ + a₀) = 27/8 := by
  sorry

end min_value_sum_reciprocals_l4094_409451


namespace circumscribed_circle_radius_of_specific_trapezoid_l4094_409440

/-- An isosceles trapezoid with given dimensions -/
structure IsoscelesTrapezoid where
  base1 : ℝ
  base2 : ℝ
  height : ℝ

/-- The radius of the circumscribed circle of an isosceles trapezoid -/
def circumscribedCircleRadius (t : IsoscelesTrapezoid) : ℝ :=
  sorry

/-- Theorem stating that for a trapezoid with given dimensions, 
    the radius of its circumscribed circle is 10.625 -/
theorem circumscribed_circle_radius_of_specific_trapezoid : 
  let t : IsoscelesTrapezoid := { base1 := 9, base2 := 21, height := 8 }
  circumscribedCircleRadius t = 10.625 := by
  sorry

end circumscribed_circle_radius_of_specific_trapezoid_l4094_409440


namespace quadratic_inequality_and_range_l4094_409424

theorem quadratic_inequality_and_range (a b : ℝ) (k : ℝ) : 
  (∀ x, (x < 1 ∨ x > b) ↔ a * x^2 - 3 * x + 2 > 0) →
  (∀ x y, x > 0 → y > 0 → a / x + b / y = 1 → 2 * x + y ≥ k^2 + k + 2) →
  a = 1 ∧ b = 2 ∧ -3 ≤ k ∧ k ≤ 2 := by
  sorry

end quadratic_inequality_and_range_l4094_409424


namespace f_value_at_2_l4094_409497

-- Define the function f
def f (x a b : ℝ) : ℝ := x^5 + a*x^3 + b*x - 8

-- State the theorem
theorem f_value_at_2 (a b : ℝ) : 
  (f (-2) a b = 3) → (f 2 a b = -19) := by
  sorry

end f_value_at_2_l4094_409497


namespace problem_statement_l4094_409483

theorem problem_statement (p : Prop) (q : Prop)
  (hp : p ↔ ∃ x₀ : ℝ, Real.exp x₀ ≤ 0)
  (hq : q ↔ ∀ x : ℝ, 2^x > x^2) :
  (¬p) ∨ q := by sorry

end problem_statement_l4094_409483


namespace problem_solution_l4094_409491

theorem problem_solution (m n : ℕ) 
  (h1 : m + 8 < n + 3)
  (h2 : (m + (m + 3) + (m + 8) + (n + 3) + (n + 4) + 2*n) / 6 = n)
  (h3 : ((m + 8) + (n + 3)) / 2 = n) : 
  m + n = 53 := by sorry

end problem_solution_l4094_409491


namespace parabola_coefficient_l4094_409492

/-- Given a parabola y = ax^2 + bx + c with vertex (p, kp) and y-intercept (0, -kp),
    where p ≠ 0 and k is a non-zero constant, prove that b = 4k/p -/
theorem parabola_coefficient (a b c p k : ℝ) (h1 : p ≠ 0) (h2 : k ≠ 0) : 
  (∀ x, a * x^2 + b * x + c = a * (x - p)^2 + k * p) →
  (a * 0^2 + b * 0 + c = -k * p) →
  b = 4 * k / p := by
  sorry

end parabola_coefficient_l4094_409492


namespace coefficient_sum_equality_l4094_409443

theorem coefficient_sum_equality (n : ℕ) (h : n ≥ 5) :
  (Finset.range (n - 4)).sum (λ k => Nat.choose (k + 5) 5) = Nat.choose (n + 1) 6 := by
  sorry

end coefficient_sum_equality_l4094_409443


namespace games_per_box_l4094_409465

theorem games_per_box (initial_games : ℕ) (sold_games : ℕ) (num_boxes : ℕ) :
  initial_games = 76 →
  sold_games = 46 →
  num_boxes = 6 →
  (initial_games - sold_games) / num_boxes = 5 := by
  sorry

end games_per_box_l4094_409465


namespace negative_abs_comparison_l4094_409471

theorem negative_abs_comparison : -|(-8 : ℤ)| < -6 := by
  sorry

end negative_abs_comparison_l4094_409471


namespace triangle_angle_difference_l4094_409401

theorem triangle_angle_difference (a b c : ℝ) (A B C : ℝ) :
  a > 0 → b > 0 → c > 0 →
  A > 0 → B > 0 → C > 0 →
  A + B + C = π →
  b = a * Real.tan B →
  A > π / 2 →
  A - B = π / 2 := by
sorry

end triangle_angle_difference_l4094_409401


namespace consecutive_odd_numbers_divisibility_l4094_409405

/-- Two natural numbers are consecutive odd numbers -/
def ConsecutiveOddNumbers (p q : ℕ) : Prop :=
  ∃ k : ℕ, p = 2*k + 1 ∧ q = 2*k + 3

/-- A number a is divisible by a number b -/
def IsDivisibleBy (a b : ℕ) : Prop :=
  ∃ k : ℕ, a = b * k

theorem consecutive_odd_numbers_divisibility (p q : ℕ) :
  ConsecutiveOddNumbers p q → IsDivisibleBy (p^q + q^p) (p + q) := by
  sorry

end consecutive_odd_numbers_divisibility_l4094_409405


namespace smaller_solution_quadratic_l4094_409466

theorem smaller_solution_quadratic (x : ℝ) : 
  (x^2 - 13*x - 30 = 0) → 
  (∃ y : ℝ, y ≠ x ∧ y^2 - 13*y - 30 = 0) → 
  (x = -2 ∨ x = 15) ∧ 
  (∀ y : ℝ, y^2 - 13*y - 30 = 0 → y ≥ -2) :=
by sorry

end smaller_solution_quadratic_l4094_409466


namespace book_arrangement_count_l4094_409439

theorem book_arrangement_count :
  let n_math_books : ℕ := 4
  let n_english_books : ℕ := 6
  let all_math_books_together : Prop := True
  let all_english_books_together : Prop := True
  let specific_english_book_at_left : Prop := True
  let all_books_distinct : Prop := True
  (n_math_books.factorial * (n_english_books - 1).factorial) = 2880 :=
by sorry

end book_arrangement_count_l4094_409439


namespace consecutive_sum_equality_l4094_409469

theorem consecutive_sum_equality :
  ∃ (a b : ℕ), 
    a ≥ 1 ∧ 
    5 * (a + 2) = 2 * b + 1 ∧ 
    ∀ (x : ℕ), x < a → ¬∃ (y : ℕ), 5 * (x + 2) = 2 * y + 1 :=
by sorry

end consecutive_sum_equality_l4094_409469


namespace half_diamond_four_thirds_l4094_409498

-- Define the diamond operation
noncomputable def diamond (a b : ℝ) : ℝ := sorry

-- Axioms
axiom diamond_def {a b : ℝ} (ha : 0 < a) (hb : 0 < b) : 
  diamond (a * b) b = a * (diamond b b)

axiom diamond_identity {a : ℝ} (ha : 0 < a) : 
  diamond (diamond a 1) a = diamond a 1

axiom diamond_one : diamond 1 1 = 1

-- Theorem to prove
theorem half_diamond_four_thirds : 
  diamond (1/2) (4/3) = 2/3 := by sorry

end half_diamond_four_thirds_l4094_409498


namespace machine_probabilities_theorem_l4094_409427

/-- Machine processing probabilities -/
structure MachineProbabilities where
  A : ℝ  -- Probability for machine A
  B : ℝ  -- Probability for machine B
  C : ℝ  -- Probability for machine C

/-- Given conditions -/
def conditions (p : MachineProbabilities) : Prop :=
  p.A * (1 - p.B) = 1/4 ∧
  p.B * (1 - p.C) = 1/12 ∧
  p.A * p.C = 2/9

/-- Theorem statement -/
theorem machine_probabilities_theorem (p : MachineProbabilities) 
  (h : conditions p) :
  p.A = 1/3 ∧ p.B = 1/4 ∧ p.C = 2/3 ∧
  1 - (1 - p.A) * (1 - p.B) * (1 - p.C) = 5/6 := by
  sorry

end machine_probabilities_theorem_l4094_409427


namespace function_property_l4094_409455

def is_positive_integer (x : ℝ) : Prop := ∃ n : ℕ, x = n ∧ n > 0

theorem function_property (f : ℝ → ℝ) 
  (h1 : ∀ x y, x < y → f x < f y)  -- monotonically increasing
  (h2 : ∀ n : ℕ, n > 0 → is_positive_integer (f n))  -- f(n) is a positive integer for positive integer n
  (h3 : ∀ n : ℕ, n > 0 → f (f n) = 2 * n + 1)  -- f(f(n)) = 2n + 1 for positive integer n
  : f 1 = 2 ∧ f 2 = 3 := by
  sorry

end function_property_l4094_409455


namespace min_projection_value_l4094_409463

theorem min_projection_value (a b : ℝ × ℝ) : 
  let norm_a := Real.sqrt ((a.1 ^ 2) + (a.2 ^ 2))
  let norm_b := Real.sqrt ((b.1 ^ 2) + (b.2 ^ 2))
  let dot_product := a.1 * b.1 + a.2 * b.2
  let cos_theta := dot_product / (norm_a * norm_b)
  norm_a = Real.sqrt 6 ∧ 
  ((a.1 + 2 * b.1) ^ 2 + (a.2 + 2 * b.2) ^ 2) = ((3 * a.1 - 4 * b.1) ^ 2 + (3 * a.2 - 4 * b.2) ^ 2) →
  ∃ (min_value : ℝ), min_value = 12 / 7 ∧ ∀ θ : ℝ, norm_a * |cos_theta| ≥ min_value := by
sorry

end min_projection_value_l4094_409463


namespace jean_carter_books_l4094_409442

/-- Prove that given 12 total volumes, paperback price of $18, hardcover price of $30, 
    and total spent of $312, the number of hardcover volumes bought is 8. -/
theorem jean_carter_books 
  (total_volumes : ℕ) 
  (paperback_price hardcover_price : ℚ) 
  (total_spent : ℚ) 
  (h : total_volumes = 12)
  (hp : paperback_price = 18)
  (hh : hardcover_price = 30)
  (hs : total_spent = 312) :
  ∃ (hardcover_count : ℕ), 
    hardcover_count * hardcover_price + (total_volumes - hardcover_count) * paperback_price = total_spent ∧ 
    hardcover_count = 8 :=
by sorry

end jean_carter_books_l4094_409442


namespace fireworks_display_total_l4094_409494

/-- Calculate the total number of fireworks in a New Year's Eve display --/
def totalFireworks : ℕ :=
  let yearDigits : ℕ := 4
  let yearFireworksPerDigit : ℕ := 6
  let happyNewYearLetters : ℕ := 12
  let regularLetterFireworks : ℕ := 5
  let helloFireworks : ℕ := 8 + 7 + 6 + 6 + 9
  let additionalBoxes : ℕ := 100
  let fireworksPerBox : ℕ := 10

  yearDigits * yearFireworksPerDigit +
  happyNewYearLetters * regularLetterFireworks +
  helloFireworks +
  additionalBoxes * fireworksPerBox

theorem fireworks_display_total :
  totalFireworks = 1120 := by sorry

end fireworks_display_total_l4094_409494


namespace necessary_but_not_sufficient_l4094_409467

theorem necessary_but_not_sufficient : 
  (∀ x : ℝ, |x - 1| < 1 → x^2 - 5*x < 0) ∧ 
  (∃ x : ℝ, x^2 - 5*x < 0 ∧ |x - 1| ≥ 1) := by
  sorry

end necessary_but_not_sufficient_l4094_409467


namespace pint_cost_is_eight_l4094_409472

/-- The cost of a pint of paint given the number of doors, cost of a gallon, and savings -/
def pint_cost (num_doors : ℕ) (gallon_cost : ℚ) (savings : ℚ) : ℚ :=
  (gallon_cost + savings) / num_doors

theorem pint_cost_is_eight :
  pint_cost 8 55 9 = 8 := by
  sorry

end pint_cost_is_eight_l4094_409472


namespace quadratic_equations_solutions_l4094_409496

theorem quadratic_equations_solutions :
  (∃ x₁ x₂ : ℝ, x₁^2 - 4*x₁ - 12 = 0 ∧ x₂^2 - 4*x₂ - 12 = 0 ∧ x₁ = 6 ∧ x₂ = -2) ∧
  (∃ y₁ y₂ : ℝ, y₁^2 - 4*y₁ - 3 = 0 ∧ y₂^2 - 4*y₂ - 3 = 0 ∧ y₁ = 2 + Real.sqrt 7 ∧ y₂ = 2 - Real.sqrt 7) :=
by sorry

end quadratic_equations_solutions_l4094_409496


namespace inverse_function_sum_l4094_409476

-- Define the functions g and g_inv
def g (c d : ℝ) (x : ℝ) : ℝ := c * x + d
def g_inv (c d : ℝ) (x : ℝ) : ℝ := d * x + c

-- State the theorem
theorem inverse_function_sum (c d : ℝ) :
  (∀ x : ℝ, g c d (g_inv c d x) = x) →
  (∀ x : ℝ, g_inv c d (g c d x) = x) →
  c + d = -2 := by
  sorry

end inverse_function_sum_l4094_409476


namespace jays_change_l4094_409464

def book_cost : ℕ := 25
def pen_cost : ℕ := 4
def ruler_cost : ℕ := 1
def amount_paid : ℕ := 50

def total_cost : ℕ := book_cost + pen_cost + ruler_cost

theorem jays_change (change : ℕ) : change = amount_paid - total_cost → change = 20 := by
  sorry

end jays_change_l4094_409464


namespace mcq_options_count_l4094_409422

theorem mcq_options_count (p_all_correct : ℚ) (p_tf_correct : ℚ) (n : ℕ) : 
  p_all_correct = 1 / 12 →
  p_tf_correct = 1 / 2 →
  (1 / n : ℚ) * p_tf_correct * p_tf_correct = p_all_correct →
  n = 3 := by
  sorry

end mcq_options_count_l4094_409422


namespace parabola_vertex_l4094_409432

/-- The vertex of the parabola y = x^2 - 1 has coordinates (0, -1) -/
theorem parabola_vertex : 
  let f : ℝ → ℝ := fun x ↦ x^2 - 1
  ∃! p : ℝ × ℝ, p.1 = 0 ∧ p.2 = -1 ∧ ∀ x : ℝ, f x ≥ f p.1 :=
by sorry

end parabola_vertex_l4094_409432


namespace distance_P_to_x_axis_l4094_409447

def point_to_x_axis_distance (p : ℝ × ℝ) : ℝ :=
  |p.2|

theorem distance_P_to_x_axis :
  let P : ℝ × ℝ := (-3, 2)
  point_to_x_axis_distance P = 2 := by
  sorry

end distance_P_to_x_axis_l4094_409447


namespace total_kittens_l4094_409458

/-- Given an initial number of kittens and additional kittens, 
    prove that the total number of kittens is their sum. -/
theorem total_kittens (initial additional : ℕ) :
  initial + additional = initial + additional :=
by sorry

end total_kittens_l4094_409458


namespace peanuts_added_l4094_409416

theorem peanuts_added (initial_peanuts final_peanuts : ℕ) 
  (h1 : initial_peanuts = 4)
  (h2 : final_peanuts = 12) : 
  final_peanuts - initial_peanuts = 8 := by
  sorry

end peanuts_added_l4094_409416


namespace pentagon_sum_edges_vertices_l4094_409454

/-- A pentagon is a polygon with 5 sides -/
structure Pentagon where
  edges : ℕ
  vertices : ℕ
  is_pentagon : edges = 5 ∧ vertices = 5

/-- The sum of edges and vertices in a pentagon is 10 -/
theorem pentagon_sum_edges_vertices (p : Pentagon) : p.edges + p.vertices = 10 := by
  sorry

end pentagon_sum_edges_vertices_l4094_409454


namespace chess_tournament_players_l4094_409481

/-- Represents the possible total scores recorded by the scorers -/
def possible_scores : List ℕ := [1979, 1980, 1984, 1985]

/-- Calculates the total number of games in a tournament with n players -/
def total_games (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem: The number of players in the chess tournament is 45 -/
theorem chess_tournament_players : ∃ (n : ℕ), n = 45 ∧ 
  ∃ (score : ℕ), score ∈ possible_scores ∧ score = 2 * total_games n := by
  sorry


end chess_tournament_players_l4094_409481


namespace imaginary_part_of_complex_fraction_l4094_409408

theorem imaginary_part_of_complex_fraction (i : ℂ) (h : i^2 = -1) :
  Complex.im (i^3 / (2*i - 1)) = 1/5 := by
  sorry

end imaginary_part_of_complex_fraction_l4094_409408


namespace tape_length_calculation_l4094_409456

/-- Calculates the length of tape wrapped around a cylindrical core -/
theorem tape_length_calculation 
  (initial_diameter : ℝ) 
  (tape_width : ℝ) 
  (num_wraps : ℕ) 
  (final_diameter : ℝ) 
  (h1 : initial_diameter = 4)
  (h2 : tape_width = 4)
  (h3 : num_wraps = 800)
  (h4 : final_diameter = 16) :
  (π / 2) * (initial_diameter + final_diameter) * num_wraps = 80 * π := by
  sorry

#check tape_length_calculation

end tape_length_calculation_l4094_409456


namespace quadratic_factorization_l4094_409409

theorem quadratic_factorization (a x : ℝ) : a * x^2 - 2 * a * x + a = a * (x - 1)^2 := by
  sorry

end quadratic_factorization_l4094_409409


namespace complex_purely_imaginary_iff_m_eq_one_l4094_409499

def is_purely_imaginary (z : ℂ) : Prop := z.re = 0

theorem complex_purely_imaginary_iff_m_eq_one (m : ℝ) :
  is_purely_imaginary (m^2 - m + m * Complex.I) ↔ m = 1 := by
  sorry

end complex_purely_imaginary_iff_m_eq_one_l4094_409499


namespace factorization_equality_l4094_409415

theorem factorization_equality (a : ℝ) : 3 * a^2 - 6 * a + 3 = 3 * (a - 1)^2 := by
  sorry

end factorization_equality_l4094_409415


namespace line_not_in_third_quadrant_l4094_409444

/-- A point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a line in the 2D plane using the general form ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Checks if a point is in the first quadrant -/
def isInFirstQuadrant (p : Point) : Prop :=
  p.x > 0 ∧ p.y > 0

/-- Checks if a line passes through the third quadrant -/
def passesThroughThirdQuadrant (l : Line) : Prop :=
  ∃ (x y : ℝ), x < 0 ∧ y < 0 ∧ l.a * x + l.b * y + l.c = 0

/-- The main theorem -/
theorem line_not_in_third_quadrant 
  (a b : ℝ) 
  (h_first_quadrant : isInFirstQuadrant ⟨a*b, a+b⟩) :
  ¬passesThroughThirdQuadrant ⟨b, a, -a*b⟩ :=
by sorry

end line_not_in_third_quadrant_l4094_409444


namespace john_grandpa_money_l4094_409437

theorem john_grandpa_money (x : ℝ) : 
  x > 0 ∧ x + 3 * x = 120 → x = 30 := by sorry

end john_grandpa_money_l4094_409437


namespace exam_score_calculation_l4094_409473

theorem exam_score_calculation (total_questions : ℕ) (correct_answers : ℕ) (total_marks : ℕ) 
  (h1 : total_questions = 60)
  (h2 : correct_answers = 40)
  (h3 : total_marks = 140)
  (h4 : correct_answers ≤ total_questions) :
  ∃ (marks_per_correct : ℕ), 
    marks_per_correct * correct_answers - (total_questions - correct_answers) = total_marks ∧ 
    marks_per_correct = 4 :=
by
  sorry

end exam_score_calculation_l4094_409473


namespace lip_gloss_coverage_l4094_409410

theorem lip_gloss_coverage 
  (num_tubs : ℕ) 
  (tubes_per_tub : ℕ) 
  (total_people : ℕ) 
  (h1 : num_tubs = 6) 
  (h2 : tubes_per_tub = 2) 
  (h3 : total_people = 36) : 
  total_people / (num_tubs * tubes_per_tub) = 3 := by
sorry

end lip_gloss_coverage_l4094_409410


namespace cafeteria_pies_l4094_409420

def number_of_pies (initial_apples : ℕ) (handed_out : ℕ) (apples_per_pie : ℕ) : ℕ :=
  (initial_apples - handed_out) / apples_per_pie

theorem cafeteria_pies : number_of_pies 51 41 5 = 2 := by
  sorry

end cafeteria_pies_l4094_409420


namespace milk_for_nine_cookies_l4094_409430

-- Define the relationship between cookies and quarts of milk
def milk_for_cookies (cookies : ℕ) : ℚ :=
  (3 : ℚ) * cookies / 18

-- Define the conversion from quarts to pints
def quarts_to_pints (quarts : ℚ) : ℚ :=
  2 * quarts

theorem milk_for_nine_cookies :
  quarts_to_pints (milk_for_cookies 9) = 3 := by
  sorry

end milk_for_nine_cookies_l4094_409430


namespace points_collinear_l4094_409482

/-- Three points are collinear if the slope between any two pairs of points is the same. -/
def collinear (p1 p2 p3 : ℝ × ℝ) : Prop :=
  let (x1, y1) := p1
  let (x2, y2) := p2
  let (x3, y3) := p3
  (y2 - y1) * (x3 - x2) = (y3 - y2) * (x2 - x1)

theorem points_collinear : collinear (1, 2) (3, -2) (4, -4) := by
  sorry

end points_collinear_l4094_409482


namespace project_completion_time_l4094_409461

/-- Given a project that A can complete in 20 days, and A and B together can complete in 15 days
    with A quitting 5 days before completion, prove that B can complete the project alone in 30 days. -/
theorem project_completion_time (a_rate b_rate : ℚ) : 
  a_rate = (1 : ℚ) / 20 →                          -- A's work rate
  a_rate + b_rate = (1 : ℚ) / 15 →                 -- A and B's combined work rate
  10 * (a_rate + b_rate) + 5 * b_rate = 1 →        -- Total work done
  b_rate = (1 : ℚ) / 30                            -- B's work rate (reciprocal of completion time)
  := by sorry

end project_completion_time_l4094_409461


namespace cube_split_2017_l4094_409412

theorem cube_split_2017 (m : ℕ) (h1 : m > 1) : 
  (m^3 = (m - 1)*(m^2 + m + 1) + (m - 1)^2 + (m - 1)^2 + 1) → 
  ((m - 1)*(m^2 + m + 1) = 2017 ∨ (m - 1)^2 = 2017 ∨ (m - 1)^2 + 2 = 2017) → 
  m = 46 := by
sorry

end cube_split_2017_l4094_409412


namespace vector_angle_solution_l4094_409486

/-- Given two plane vectors a and b with unit length and 60° angle between them,
    prove that t = 0 is a valid solution when the angle between a+b and ta-b is obtuse. -/
theorem vector_angle_solution (a b : ℝ × ℝ) (t : ℝ) :
  (norm a = 1) →
  (norm b = 1) →
  (a • b = 1 / 2) →  -- cos 60° = 1/2
  ((a + b) • (t • a - b) < 0) →
  (t = 0) →
  True := by sorry

end vector_angle_solution_l4094_409486


namespace log_equation_solution_l4094_409413

theorem log_equation_solution : 
  let f : ℝ → ℝ := λ x ↦ Real.log x / Real.log 5
  ∀ x : ℝ, f (x^2 - 25*x) = 3 ↔ x = 5*(5 + 3*Real.sqrt 5) ∨ x = 5*(5 - 3*Real.sqrt 5) := by
  sorry

end log_equation_solution_l4094_409413


namespace investor_share_calculation_l4094_409407

/-- Calculates the share of profit for an investor given the investments and time periods. -/
theorem investor_share_calculation
  (investment_a investment_b total_profit : ℚ)
  (time_a time_b total_time : ℕ)
  (h1 : investment_a = 150)
  (h2 : investment_b = 200)
  (h3 : total_profit = 100)
  (h4 : time_a = 12)
  (h5 : time_b = 6)
  (h6 : total_time = 12)
  : (investment_a * time_a) / ((investment_a * time_a) + (investment_b * time_b)) * total_profit = 60 := by
  sorry

#check investor_share_calculation

end investor_share_calculation_l4094_409407


namespace surface_area_of_specific_block_l4094_409449

/-- Represents a rectangular solid block made of unit cubes -/
structure RectangularBlock where
  length : Nat
  width : Nat
  height : Nat
  total_cubes : Nat

/-- Calculates the surface area of a rectangular block -/
def surface_area (block : RectangularBlock) : Nat :=
  2 * (block.length * block.width + block.length * block.height + block.width * block.height)

/-- Theorem stating that the surface area of the specific block is 66 square units -/
theorem surface_area_of_specific_block :
  ∃ (block : RectangularBlock),
    block.length = 5 ∧
    block.width = 3 ∧
    block.height = 1 ∧
    block.total_cubes = 15 ∧
    surface_area block = 66 := by
  sorry

end surface_area_of_specific_block_l4094_409449


namespace solution_to_equation_l4094_409436

theorem solution_to_equation : ∃ x y : ℤ, 5 * x + 4 * y = 14 ∧ x = 2 ∧ y = 1 := by
  sorry

end solution_to_equation_l4094_409436


namespace cake_box_theorem_l4094_409406

/-- Represents the dimensions of a rectangular box -/
structure BoxDimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Calculates the volume of a box given its dimensions -/
def boxVolume (d : BoxDimensions) : ℕ :=
  d.length * d.width * d.height

/-- Represents the problem of fitting cake boxes into a carton -/
structure CakeBoxProblem where
  carton : BoxDimensions
  cakeBox : BoxDimensions

/-- Calculates the maximum number of cake boxes that can fit in a carton -/
def maxCakeBoxes (p : CakeBoxProblem) : ℕ :=
  (boxVolume p.carton) / (boxVolume p.cakeBox)

/-- The main theorem stating the maximum number of cake boxes that can fit in the given carton -/
theorem cake_box_theorem (p : CakeBoxProblem) 
  (h_carton : p.carton = ⟨25, 42, 60⟩) 
  (h_cake_box : p.cakeBox = ⟨8, 7, 5⟩) : 
  maxCakeBoxes p = 225 := by
  sorry

#eval maxCakeBoxes ⟨⟨25, 42, 60⟩, ⟨8, 7, 5⟩⟩

end cake_box_theorem_l4094_409406


namespace expression_evaluation_l4094_409452

theorem expression_evaluation (x y : ℝ) 
  (h : |x + 1/2| + (y - 1)^2 = 0) : 
  5 * x^2 * y - (6 * x * y - 2 * (x * y - 2 * x^2 * y) - x * y^2) + 4 * x * y = -1/4 :=
by sorry

end expression_evaluation_l4094_409452


namespace g_50_solutions_l4094_409488

def g₀ (x : ℝ) : ℝ := x + |x - 50| - |x + 50|

def g (n : ℕ) (x : ℝ) : ℝ := 
  match n with
  | 0 => g₀ x
  | n + 1 => |g n x| - 2

theorem g_50_solutions : 
  ∃ (S : Finset ℝ), (∀ x ∈ S, g 50 x = 0) ∧ (∀ x ∉ S, g 50 x ≠ 0) ∧ Finset.card S = 4 := by
  sorry

end g_50_solutions_l4094_409488


namespace remainder_problem_l4094_409423

theorem remainder_problem (n : ℤ) (h : n % 8 = 3) : (4 * n - 10) % 8 = 2 := by
  sorry

end remainder_problem_l4094_409423


namespace linear_function_proof_l4094_409493

/-- A linear function passing through points (1, -1) and (-2, 8) -/
def f (x : ℝ) : ℝ := -3 * x + 2

theorem linear_function_proof :
  (f 1 = -1) ∧ 
  (f (-2) = 8) ∧ 
  (∀ x : ℝ, f x = -3 * x + 2) ∧
  (f (-10) = 32) := by
  sorry


end linear_function_proof_l4094_409493


namespace fraction_problem_l4094_409448

theorem fraction_problem :
  ∃ (n d : ℚ), n + d = 5.25 ∧ (n + 3) / (2 * d) = 1/3 ∧ n/d = 2/33 := by
  sorry

end fraction_problem_l4094_409448


namespace rational_expression_iff_zero_l4094_409445

theorem rational_expression_iff_zero (x : ℝ) : 
  ∃ (q : ℚ), x + Real.sqrt (x^2 + 4) - 1 / (x + Real.sqrt (x^2 + 4)) = q ↔ x = 0 := by
  sorry

end rational_expression_iff_zero_l4094_409445


namespace simple_interest_rate_proof_l4094_409441

/-- Given a principal amount and a simple interest rate, 
    if the amount becomes 7/6 of itself after 4 years, 
    then the rate is 1/24 -/
theorem simple_interest_rate_proof 
  (P : ℝ) (R : ℝ) (P_pos : P > 0) :
  P * (1 + 4 * R) = 7/6 * P → R = 1/24 := by
  sorry

end simple_interest_rate_proof_l4094_409441


namespace chord_midpoint_theorem_l4094_409431

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 16 + y^2 / 4 = 1

-- Define the point P
def P : ℝ × ℝ := (1, 1)

-- Define the chord equation
def chord_equation (x y : ℝ) : Prop := x + 4*y - 5 = 0

-- Theorem statement
theorem chord_midpoint_theorem :
  ∀ (A B : ℝ × ℝ),
  ellipse A.1 A.2 ∧ ellipse B.1 B.2 →
  (A.1 + B.1) / 2 = P.1 ∧ (A.2 + B.2) / 2 = P.2 →
  ∀ (x y : ℝ), chord_equation x y ↔ ∃ t : ℝ, (x, y) = (1 - t, 1 + t/4) :=
sorry

end chord_midpoint_theorem_l4094_409431


namespace min_value_sum_reciprocals_l4094_409460

theorem min_value_sum_reciprocals (p q r s t u : ℝ) 
  (hp : p > 0) (hq : q > 0) (hr : r > 0) (hs : s > 0) (ht : t > 0) (hu : u > 0)
  (hsum : p + q + r + s + t + u = 10) :
  1/p + 9/q + 25/r + 49/s + 81/t + 121/u ≥ 129.6 := by
  sorry


end min_value_sum_reciprocals_l4094_409460


namespace work_completion_time_l4094_409475

/-- Given a work that can be completed by person A in 20 days, and 0.375 of the work
    can be completed by A and B together in 5 days, prove that person B can complete
    the work alone in 40 days. -/
theorem work_completion_time (work_rate_A work_rate_B : ℝ) : 
  work_rate_A = 1 / 20 →
  5 * (work_rate_A + work_rate_B) = 0.375 →
  1 / work_rate_B = 40 := by
sorry

end work_completion_time_l4094_409475


namespace five_integers_average_l4094_409426

theorem five_integers_average (a₁ a₂ a₃ a₄ a₅ : ℤ) :
  (a₁ + a₂) + (a₁ + a₃) + (a₁ + a₄) + (a₁ + a₅) + 
  (a₂ + a₃) + (a₂ + a₄) + (a₂ + a₅) + 
  (a₃ + a₄) + (a₃ + a₅) + 
  (a₄ + a₅) = 2020 →
  (a₁ + a₂ + a₃ + a₄ + a₅) / 5 = 101 := by
sorry

#eval (2020 : ℤ) / 4  -- To verify that 2020 / 4 = 505

end five_integers_average_l4094_409426


namespace stratified_sampling_appropriate_for_subgroups_investigation1_uses_stratified_sampling_investigation2_uses_simple_random_sampling_l4094_409425

/-- Represents a sampling method -/
inductive SamplingMethod
  | StratifiedSampling
  | SimpleRandomSampling
  | SystematicSampling

/-- Represents a region with its number of outlets -/
structure Region where
  name : String
  outlets : Nat

/-- Represents the company's sales outlet distribution -/
structure CompanyDistribution where
  regions : List Region
  totalOutlets : Nat

/-- Represents a sampling scenario -/
structure SamplingScenario where
  population : CompanyDistribution
  sampleSize : Nat
  hasDistinctSubgroups : Bool

/-- Determines the most appropriate sampling method for a given scenario -/
def appropriateSamplingMethod (scenario : SamplingScenario) : SamplingMethod :=
  if scenario.hasDistinctSubgroups then
    SamplingMethod.StratifiedSampling
  else
    SamplingMethod.SimpleRandomSampling

/-- Theorem: Stratified sampling is more appropriate for populations with distinct subgroups -/
theorem stratified_sampling_appropriate_for_subgroups 
  (scenario : SamplingScenario) 
  (h : scenario.hasDistinctSubgroups = true) : 
  appropriateSamplingMethod scenario = SamplingMethod.StratifiedSampling :=
sorry

/-- Company distribution for the given problem -/
def companyDistribution : CompanyDistribution :=
  { regions := [
      { name := "A", outlets := 150 },
      { name := "B", outlets := 120 },
      { name := "C", outlets := 180 },
      { name := "D", outlets := 150 }
    ],
    totalOutlets := 600
  }

/-- Sampling scenario for investigation (1) -/
def investigation1 : SamplingScenario :=
  { population := companyDistribution,
    sampleSize := 100,
    hasDistinctSubgroups := true
  }

/-- Sampling scenario for investigation (2) -/
def investigation2 : SamplingScenario :=
  { population := { regions := [{ name := "C_large", outlets := 20 }], totalOutlets := 20 },
    sampleSize := 7,
    hasDistinctSubgroups := false
  }

/-- Theorem: The appropriate sampling method for investigation (1) is Stratified Sampling -/
theorem investigation1_uses_stratified_sampling :
  appropriateSamplingMethod investigation1 = SamplingMethod.StratifiedSampling :=
sorry

/-- Theorem: The appropriate sampling method for investigation (2) is Simple Random Sampling -/
theorem investigation2_uses_simple_random_sampling :
  appropriateSamplingMethod investigation2 = SamplingMethod.SimpleRandomSampling :=
sorry

end stratified_sampling_appropriate_for_subgroups_investigation1_uses_stratified_sampling_investigation2_uses_simple_random_sampling_l4094_409425


namespace wallet_total_l4094_409404

theorem wallet_total (nada ali john : ℕ) : 
  ali = nada - 5 →
  john = 4 * nada →
  john = 48 →
  ali + nada + john = 67 := by
sorry

end wallet_total_l4094_409404


namespace inequality_proof_l4094_409478

theorem inequality_proof (p q r x y θ : ℝ) :
  p * x^(q - y) + q * x^(r - y) + r * x^(y - θ) ≥ p + q + r := by
  sorry

end inequality_proof_l4094_409478


namespace cross_number_puzzle_l4094_409418

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n < 1000

def second_digit (n : ℕ) : ℕ := (n / 10) % 10

theorem cross_number_puzzle : 
  ∃! d : ℕ, d < 10 ∧ 
  (∃ m : ℕ, is_three_digit (3^m) ∧ second_digit (3^m) = d) ∧
  (∃ n : ℕ, is_three_digit (7^n) ∧ second_digit (7^n) = d) ∧
  d = 4 :=
sorry

end cross_number_puzzle_l4094_409418


namespace percentage_problem_l4094_409429

theorem percentage_problem (N : ℝ) (P : ℝ) : 
  N = 100 → 
  (P / 100) * N = (3 / 5) * N - 10 → 
  P = 50 := by
sorry

end percentage_problem_l4094_409429


namespace delta_4_zero_delta_3_nonzero_l4094_409419

def u (n : ℕ) : ℤ := n^3 + n

def delta_1 (u : ℕ → ℤ) (n : ℕ) : ℤ := u (n + 1) - u n

def delta (k : ℕ) (u : ℕ → ℤ) : ℕ → ℤ :=
  match k with
  | 0 => u
  | 1 => delta_1 u
  | k + 1 => delta_1 (delta k u)

theorem delta_4_zero_delta_3_nonzero :
  (∀ n, delta 4 u n = 0) ∧ (∃ n, delta 3 u n ≠ 0) := by sorry

end delta_4_zero_delta_3_nonzero_l4094_409419


namespace domain_of_composed_function_l4094_409450

-- Define the function f
def f : ℝ → ℝ := sorry

-- State the theorem
theorem domain_of_composed_function :
  (∀ x ∈ Set.Icc 0 3, f (x + 1) ∈ Set.range f) →
  {x : ℝ | f (2^x) ∈ Set.range f} = Set.Icc 0 2 := by
  sorry

end domain_of_composed_function_l4094_409450


namespace operation_state_theorem_l4094_409414

/-- Represents the state of a student operating a computer -/
def OperationState := Fin 5 → Fin 5 → Bool

/-- The given condition that the product of diagonal elements is 0 -/
def DiagonalProductZero (a : OperationState) : Prop :=
  (a 0 0) && (a 1 1) && (a 2 2) && (a 3 3) && (a 4 4) = false

/-- At least one student is not operating their own computer -/
def AtLeastOneNotOwnComputer (a : OperationState) : Prop :=
  ∃ i : Fin 5, a i i = false

/-- 
If the product of diagonal elements in the operation state matrix is 0,
then at least one student is not operating their own computer.
-/
theorem operation_state_theorem (a : OperationState) :
  DiagonalProductZero a → AtLeastOneNotOwnComputer a := by
  sorry

end operation_state_theorem_l4094_409414


namespace equation_equivalence_l4094_409400

theorem equation_equivalence (x : ℝ) :
  (x - 2)^5 + (x - 6)^5 = 32 →
  let z := x - 4
  z^5 + 40*z^3 + 80*z - 32 = 0 := by
sorry

end equation_equivalence_l4094_409400


namespace gcf_lcm_sum_4_6_l4094_409459

theorem gcf_lcm_sum_4_6 : Nat.gcd 4 6 + Nat.lcm 4 6 = 14 := by
  sorry

end gcf_lcm_sum_4_6_l4094_409459


namespace power_function_not_through_origin_l4094_409480

theorem power_function_not_through_origin (m : ℝ) : 
  let f : ℝ → ℝ := λ x ↦ (m^2 - 3*m + 3) * x^(m^2 - m - 2)
  (∀ x ≠ 0, f x ≠ 0) → (m = 1 ∨ m = 2) :=
by
  sorry

end power_function_not_through_origin_l4094_409480


namespace cos_two_alpha_zero_l4094_409495

theorem cos_two_alpha_zero (α : Real) 
  (h : Real.sin (π / 6 - α) = Real.cos (π / 6 + α)) : 
  Real.cos (2 * α) = 0 := by
  sorry

end cos_two_alpha_zero_l4094_409495


namespace geometric_sequence_fourth_term_l4094_409438

/-- A geometric sequence with common ratio r -/
def geometric_sequence (a : ℕ → ℝ) (r : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_fourth_term
  (a : ℕ → ℝ) (r : ℝ) 
  (h_geo : geometric_sequence a r)
  (h_roots : a 2 * a 6 = 64) :
  a 4 = 8 := by
  sorry

end geometric_sequence_fourth_term_l4094_409438


namespace A_equals_set_l4094_409484

def A : Set ℕ := {x | 0 ≤ x ∧ x < 3}

theorem A_equals_set : A = {0, 1, 2} := by sorry

end A_equals_set_l4094_409484


namespace cake_division_theorem_l4094_409468

/-- Represents a piece of cake -/
structure CakePiece where
  cookies : ℕ
  roses : ℕ

/-- Represents the whole cake -/
structure Cake where
  totalCookies : ℕ
  totalRoses : ℕ
  pieces : ℕ

/-- Checks if a cake can be evenly divided -/
def isEvenlyDivisible (c : Cake) : Prop :=
  c.totalCookies % c.pieces = 0 ∧ c.totalRoses % c.pieces = 0

/-- Calculates the content of each piece when the cake is evenly divided -/
def pieceContent (c : Cake) (h : isEvenlyDivisible c) : CakePiece :=
  { cookies := c.totalCookies / c.pieces
  , roses := c.totalRoses / c.pieces }

/-- Theorem: If a cake with 48 cookies and 4 roses is cut into 4 equal pieces,
    each piece will have 12 cookies and 1 rose -/
theorem cake_division_theorem (c : Cake)
    (h1 : c.totalCookies = 48)
    (h2 : c.totalRoses = 4)
    (h3 : c.pieces = 4)
    (h4 : isEvenlyDivisible c) :
    pieceContent c h4 = { cookies := 12, roses := 1 } := by
  sorry


end cake_division_theorem_l4094_409468


namespace angle_measure_in_triangle_l4094_409434

theorem angle_measure_in_triangle (P Q R : ℝ) 
  (h1 : P = 75)
  (h2 : Q = 2 * R - 15)
  (h3 : P + Q + R = 180) :
  R = 40 := by
  sorry

end angle_measure_in_triangle_l4094_409434


namespace computer_off_time_l4094_409433

/-- Represents days of the week -/
inductive Day
  | Friday
  | Saturday

/-- Represents time of day in hours (0-23) -/
def Time := Fin 24

/-- Represents a specific moment (day and time) -/
structure Moment where
  day : Day
  time : Time

/-- Adds hours to a given moment, wrapping to the next day if necessary -/
def addHours (m : Moment) (h : Nat) : Moment :=
  let totalHours := m.time.val + h
  let newDay := if totalHours ≥ 24 then Day.Saturday else m.day
  let newTime := Fin.ofNat (totalHours % 24)
  { day := newDay, time := newTime }

theorem computer_off_time 
  (start : Moment) 
  (h : Nat) 
  (start_day : start.day = Day.Friday)
  (start_time : start.time = ⟨14, sorry⟩)
  (duration : h = 30) :
  addHours start h = { day := Day.Saturday, time := ⟨20, sorry⟩ } := by
  sorry

#check computer_off_time

end computer_off_time_l4094_409433
