import Mathlib

namespace second_street_sales_l880_88082

/-- Represents the sales data for a door-to-door salesman selling security systems. -/
structure SalesData where
  commission_per_sale : ℕ
  total_commission : ℕ
  streets : Fin 4 → ℕ
  second_street_sales : ℕ

/-- The conditions of the sales problem. -/
def sales_conditions (data : SalesData) : Prop :=
  data.commission_per_sale = 25 ∧
  data.total_commission = 175 ∧
  data.streets 0 = data.second_street_sales / 2 ∧
  data.streets 1 = data.second_street_sales ∧
  data.streets 2 = 0 ∧
  data.streets 3 = 1

/-- Theorem stating that under the given conditions, the number of security systems sold on the second street is 4. -/
theorem second_street_sales (data : SalesData) :
  sales_conditions data → data.second_street_sales = 4 := by
  sorry

end second_street_sales_l880_88082


namespace square_side_equals_pi_l880_88021

theorem square_side_equals_pi : ∃ x : ℝ, 
  4 * x = 2 * π * 2 ∧ x = π := by
  sorry

end square_side_equals_pi_l880_88021


namespace percentage_difference_l880_88055

theorem percentage_difference : (0.6 * 40) - (4 / 5 * 25) = 4 := by
  sorry

end percentage_difference_l880_88055


namespace polygon_sides_l880_88031

theorem polygon_sides (n : ℕ) : 
  (n - 2) * 180 = 3 * 360 → n = 8 := by
  sorry

end polygon_sides_l880_88031


namespace part_one_part_two_l880_88089

-- Define the function f
def f (x a b : ℝ) : ℝ := |2*x + a| + |2*x - 2*b| + 3

-- Part I
theorem part_one :
  let a : ℝ := 1
  let b : ℝ := 1
  {x : ℝ | f x a b > 8} = {x : ℝ | x < -1 ∨ x > 1.5} := by sorry

-- Part II
theorem part_two :
  ∀ (a b : ℝ), a > 0 → b > 0 →
  (∀ x : ℝ, f x a b ≥ 5) →
  (∃ x : ℝ, f x a b = 5) →
  (∀ a' b' : ℝ, a' > 0 → b' > 0 →
    (∀ x : ℝ, f x a' b' ≥ 5) →
    (∃ x : ℝ, f x a' b' = 5) →
    1/a + 1/b ≤ 1/a' + 1/b') →
  1/a + 1/b = (3 + 2 * Real.sqrt 2) / 2 := by sorry

end part_one_part_two_l880_88089


namespace skate_cost_is_65_l880_88016

/-- The cost of renting skates for one visit -/
def rental_cost : ℚ := 2.5

/-- The number of visits needed to justify buying skates -/
def visits : ℕ := 26

/-- The cost of a new pair of skates -/
def skate_cost : ℚ := rental_cost * visits

/-- Theorem stating that the cost of a new pair of skates is $65 -/
theorem skate_cost_is_65 : skate_cost = 65 := by sorry

end skate_cost_is_65_l880_88016


namespace divisible_by_three_divisible_by_nine_l880_88015

/-- Represents the decimal digits of a non-negative integer -/
def DecimalDigits : Type := List Nat

/-- Returns the sum of digits in a DecimalDigits representation -/
def sum_of_digits (digits : DecimalDigits) : Nat :=
  digits.sum

/-- Converts a non-negative integer to its DecimalDigits representation -/
def to_decimal_digits (n : Nat) : DecimalDigits :=
  sorry

/-- Converts a DecimalDigits representation back to the original number -/
def from_decimal_digits (digits : DecimalDigits) : Nat :=
  sorry

/-- Theorem: A number is divisible by 3 iff the sum of its digits is divisible by 3 -/
theorem divisible_by_three (n : Nat) :
  n % 3 = 0 ↔ (sum_of_digits (to_decimal_digits n)) % 3 = 0 :=
  sorry

/-- Theorem: A number is divisible by 9 iff the sum of its digits is divisible by 9 -/
theorem divisible_by_nine (n : Nat) :
  n % 9 = 0 ↔ (sum_of_digits (to_decimal_digits n)) % 9 = 0 :=
  sorry

end divisible_by_three_divisible_by_nine_l880_88015


namespace smallest_multiplier_for_perfect_square_l880_88085

def y : ℕ := 2^3 * 3^3 * 4^4 * 5^5 * 6^6 * 7^7 * 8^8 * 11^3

theorem smallest_multiplier_for_perfect_square :
  ∃! k : ℕ, k > 0 ∧ 
  (∃ m : ℕ, k * y = m^2) ∧
  (∀ j : ℕ, j > 0 → j < k → ¬∃ n : ℕ, j * y = n^2) ∧
  k = 2310 :=
sorry

end smallest_multiplier_for_perfect_square_l880_88085


namespace triangle_third_side_l880_88096

theorem triangle_third_side (a b : ℝ) (n : ℕ) : 
  a = 3.14 → b = 0.67 → 
  (n : ℝ) + b > a ∧ (n : ℝ) + a > b ∧ a + b > (n : ℝ) →
  n = 3 := by
sorry

end triangle_third_side_l880_88096


namespace fishing_earnings_l880_88053

/-- Calculates the total earnings from fishing over a period including a specific day --/
theorem fishing_earnings (rate : ℝ) (past_catch : ℝ) (today_multiplier : ℝ) :
  let past_earnings := rate * past_catch
  let today_catch := past_catch * today_multiplier
  let today_earnings := rate * today_catch
  let total_earnings := past_earnings + today_earnings
  (rate = 20 ∧ past_catch = 80 ∧ today_multiplier = 2) →
  total_earnings = 4800 :=
by
  sorry

end fishing_earnings_l880_88053


namespace smallest_angle_solution_l880_88098

theorem smallest_angle_solution : 
  ∃ x : ℝ, x > 0 ∧ 
    6 * Real.sin x * (Real.cos x)^3 - 6 * (Real.sin x)^3 * Real.cos x = 3 * Real.sqrt 3 / 2 ∧
    ∀ y : ℝ, y > 0 → 
      6 * Real.sin y * (Real.cos y)^3 - 6 * (Real.sin y)^3 * Real.cos y = 3 * Real.sqrt 3 / 2 → 
      x ≤ y ∧
    x = π / 12 :=
sorry

end smallest_angle_solution_l880_88098


namespace sequence_length_l880_88090

/-- Calculates the number of terms in an arithmetic sequence -/
def arithmeticSequenceLength (a₁ : ℕ) (aₙ : ℕ) (d : ℕ) : ℕ :=
  (aₙ - a₁) / d + 1

/-- The number of terms in the arithmetic sequence from 5 to 200 with common difference 3 -/
theorem sequence_length : arithmeticSequenceLength 5 200 3 = 66 := by
  sorry

#eval arithmeticSequenceLength 5 200 3

end sequence_length_l880_88090


namespace abs_sum_minus_product_equals_two_l880_88051

theorem abs_sum_minus_product_equals_two
  (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  |a| / a + |b| / b + |c| / c - (a * b * c) / |a * b * c| = 2 := by
  sorry

end abs_sum_minus_product_equals_two_l880_88051


namespace recipe_total_cups_l880_88000

/-- The total number of cups needed for a recipe with cereal, milk, and nuts. -/
def total_cups (cereal_servings milk_servings nuts_servings : ℝ)
               (cereal_cups_per_serving milk_cups_per_serving nuts_cups_per_serving : ℝ) : ℝ :=
  cereal_servings * cereal_cups_per_serving +
  milk_servings * milk_cups_per_serving +
  nuts_servings * nuts_cups_per_serving

/-- Theorem stating that the total cups needed for the given recipe is 57.0 cups. -/
theorem recipe_total_cups :
  total_cups 18.0 12.0 6.0 2.0 1.5 0.5 = 57.0 := by
  sorry

end recipe_total_cups_l880_88000


namespace tommy_savings_needed_l880_88060

def number_of_books : ℕ := 8
def cost_per_book : ℕ := 5
def current_savings : ℕ := 13

theorem tommy_savings_needed : 
  number_of_books * cost_per_book - current_savings = 27 := by
  sorry

end tommy_savings_needed_l880_88060


namespace chord_inequality_l880_88036

/-- Given a semicircle with unit radius and four consecutive chords with lengths a, b, c, d,
    prove that a^2 + b^2 + c^2 + d^2 + abc + bcd < 4 -/
theorem chord_inequality (a b c d : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d)
  (hchords : ∃ (A B C D E : ℝ × ℝ), 
    (A.1 - B.1)^2 + (A.2 - B.2)^2 = a^2 ∧
    (B.1 - C.1)^2 + (B.2 - C.2)^2 = b^2 ∧
    (C.1 - D.1)^2 + (C.2 - D.2)^2 = c^2 ∧
    (D.1 - E.1)^2 + (D.2 - E.2)^2 = d^2 ∧
    A.1^2 + A.2^2 = 1 ∧ B.1^2 + B.2^2 = 1 ∧ C.1^2 + C.2^2 = 1 ∧ 
    D.1^2 + D.2^2 = 1 ∧ E.1^2 + E.2^2 = 1 ∧
    (A.2 ≥ 0 ∧ B.2 ≥ 0 ∧ C.2 ≥ 0 ∧ D.2 ≥ 0 ∧ E.2 ≥ 0)) :
  a^2 + b^2 + c^2 + d^2 + a*b*c + b*c*d < 4 := by
  sorry

end chord_inequality_l880_88036


namespace chocolate_milk_probability_l880_88091

/-- The probability of exactly k successes in n independent trials with probability p of success in each trial. -/
def binomial_probability (n k : ℕ) (p : ℚ) : ℚ :=
  (n.choose k : ℚ) * p ^ k * (1 - p) ^ (n - k)

/-- The probability of exactly 5 successes in 7 independent trials with 3/4 probability of success in each trial is 5103/16384. -/
theorem chocolate_milk_probability :
  binomial_probability 7 5 (3/4) = 5103/16384 := by
  sorry

end chocolate_milk_probability_l880_88091


namespace quadratic_polynomial_conditions_l880_88043

theorem quadratic_polynomial_conditions (p : ℝ → ℝ) :
  (∀ x, p x = 1.8 * x^2 - 5.4 * x - 32.4) →
  p (-3) = 0 ∧ p 6 = 0 ∧ p 7 = 18 := by
  sorry

end quadratic_polynomial_conditions_l880_88043


namespace parallel_line_plane_not_all_parallel_l880_88058

-- Define the basic types
variable (Point : Type) (Line : Type) (Plane : Type)

-- Define the relations
variable (contains : Plane → Line → Prop)
variable (parallel : Line → Plane → Prop)
variable (parallel_lines : Line → Line → Prop)

-- Define the specific objects
variable (a b : Line) (α : Plane)

-- State the theorem
theorem parallel_line_plane_not_all_parallel 
  (h1 : ¬ contains α b)
  (h2 : contains α a)
  (h3 : parallel b α) :
  ¬ (∀ (l : Line), contains α l → parallel_lines b l) := by
  sorry


end parallel_line_plane_not_all_parallel_l880_88058


namespace union_of_P_and_Q_l880_88007

def P : Set ℕ := {1, 2}
def Q : Set ℕ := {y | ∃ a ∈ P, y = 2*a - 1}

theorem union_of_P_and_Q : P ∪ Q = {1, 2, 3} := by sorry

end union_of_P_and_Q_l880_88007


namespace quadratic_inequality_solution_l880_88079

/-- Given a quadratic inequality ax^2 + bx + 1 > 0 with solution set (-1, 1/3), 
    prove that a - b = 3 -/
theorem quadratic_inequality_solution (a b : ℝ) : 
  (∀ x : ℝ, ax^2 + b*x + 1 > 0 ↔ -1 < x ∧ x < 1/3) → 
  a - b = 3 :=
by sorry

end quadratic_inequality_solution_l880_88079


namespace reach_power_of_three_l880_88037

/-- Represents the possible operations on the blackboard -/
inductive Operation
  | triple_minus_one : Operation  -- 3k - 1
  | double_plus_one : Operation   -- 2k + 1
  | half : Operation              -- k / 2

/-- Applies an operation to a number if the result is an integer -/
def apply_operation (k : ℤ) (op : Operation) : Option ℤ :=
  match op with
  | Operation.triple_minus_one => some (3 * k - 1)
  | Operation.double_plus_one => some (2 * k + 1)
  | Operation.half => if k % 2 = 0 then some (k / 2) else none

/-- Represents a sequence of operations -/
def OperationSequence := List Operation

/-- Applies a sequence of operations to a number -/
def apply_sequence (n : ℤ) (seq : OperationSequence) : Option ℤ :=
  seq.foldl (fun acc op => acc.bind (fun k => apply_operation k op)) (some n)

/-- The main theorem -/
theorem reach_power_of_three (n : ℤ) (h : n ≥ 1) :
  ∃ (seq : OperationSequence), apply_sequence n seq = some (3^2023) :=
sorry

end reach_power_of_three_l880_88037


namespace squares_in_figure_50_l880_88010

/-- The number of squares in the nth figure -/
def f (n : ℕ) : ℕ :=
  3 * n^2 + 3 * n + 1

/-- The sequence of squares follows the given pattern -/
axiom pattern_holds : f 0 = 1 ∧ f 1 = 7 ∧ f 2 = 19 ∧ f 3 = 37

/-- The number of squares in figure 50 is 7651 -/
theorem squares_in_figure_50 : f 50 = 7651 := by
  sorry

end squares_in_figure_50_l880_88010


namespace compound_molecular_weight_l880_88046

/-- Atomic weight of Copper in g/mol -/
def Cu_weight : ℝ := 63.546

/-- Atomic weight of Carbon in g/mol -/
def C_weight : ℝ := 12.011

/-- Atomic weight of Oxygen in g/mol -/
def O_weight : ℝ := 15.999

/-- Number of Copper atoms in the compound -/
def Cu_count : ℕ := 1

/-- Number of Carbon atoms in the compound -/
def C_count : ℕ := 1

/-- Number of Oxygen atoms in the compound -/
def O_count : ℕ := 3

/-- Molecular weight of the compound in g/mol -/
def molecular_weight : ℝ := Cu_count * Cu_weight + C_count * C_weight + O_count * O_weight

theorem compound_molecular_weight : molecular_weight = 123.554 := by
  sorry

end compound_molecular_weight_l880_88046


namespace product_ab_l880_88047

theorem product_ab (a b : ℚ) (h1 : 2 * a + 5 * b = 40) (h2 : 4 * a + 3 * b = 41) :
  a * b = 3315 / 98 := by
sorry

end product_ab_l880_88047


namespace factorization_equality_l880_88069

theorem factorization_equality (x : ℝ) :
  (x^4 + x^2 - 4) * (x^4 + x^2 + 3) + 10 = (x^4 + x^2 + 1) * (x^2 + 2) * (x + 1) * (x - 1) := by
  sorry

end factorization_equality_l880_88069


namespace art_gallery_display_ratio_l880_88039

/-- Theorem about the ratio of displayed art pieces to total pieces in a gallery -/
theorem art_gallery_display_ratio 
  (total_pieces : ℕ)
  (sculptures_not_displayed : ℕ)
  (h_total : total_pieces = 3150)
  (h_sculptures_not_displayed : sculptures_not_displayed = 1400)
  (h_display_ratio : ∀ d : ℕ, d > 0 → (d : ℚ) / 6 = (sculptures_not_displayed : ℚ))
  (h_not_display_ratio : ∀ n : ℕ, n > 0 → (n : ℚ) / 3 = ((total_pieces - sculptures_not_displayed) : ℚ)) :
  (total_pieces : ℚ) / 3 = (total_pieces - sculptures_not_displayed * 3 / 2 : ℚ) :=
by sorry

end art_gallery_display_ratio_l880_88039


namespace average_milk_production_per_cow_l880_88035

theorem average_milk_production_per_cow (num_cows : ℕ) (total_milk : ℕ) (num_days : ℕ) 
  (h_cows : num_cows = 40)
  (h_milk : total_milk = 12000)
  (h_days : num_days = 30) :
  total_milk / num_cows / num_days = 10 := by
  sorry

end average_milk_production_per_cow_l880_88035


namespace polynomial_division_remainder_l880_88034

theorem polynomial_division_remainder : ∃ q : Polynomial ℝ, 
  X^4 + 2*X^2 + 1 = (X^2 - 2*X + 4) * q + (-4*X - 7) := by
  sorry

end polynomial_division_remainder_l880_88034


namespace p_neither_sufficient_nor_necessary_for_q_l880_88083

-- Define the propositions p and q
def p (a b : ℝ) : Prop := a + b > 0
def q (a b : ℝ) : Prop := a * b > 0

-- Theorem stating that p is neither sufficient nor necessary for q
theorem p_neither_sufficient_nor_necessary_for_q :
  (∃ a b : ℝ, p a b ∧ ¬q a b) ∧ (∃ a b : ℝ, q a b ∧ ¬p a b) :=
sorry

end p_neither_sufficient_nor_necessary_for_q_l880_88083


namespace monotonic_function_range_l880_88052

def f (a x : ℝ) : ℝ := x^2 + 2*(a-1)*x + 2

def monotonic_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x < f y ∨ (∀ z, a ≤ z ∧ z ≤ b → f z = f x)

theorem monotonic_function_range (a : ℝ) :
  monotonic_on (f a) (-1) 2 → a ≤ -1 ∨ a ≥ 2 := by
  sorry

end monotonic_function_range_l880_88052


namespace sum_of_squares_problem_l880_88099

theorem sum_of_squares_problem (a b c : ℝ) : 
  a ≥ 0 → b ≥ 0 → c ≥ 0 → 
  a^2 + b^2 + c^2 = 64 → 
  a*b + b*c + c*a = 30 → 
  a + b + c = 2 * Real.sqrt 31 := by
sorry

end sum_of_squares_problem_l880_88099


namespace k_of_five_eq_eight_point_five_l880_88088

noncomputable def h (x : ℝ) : ℝ := 5 / (3 - x)

noncomputable def h_inverse (x : ℝ) : ℝ := 3 - 5 / x

noncomputable def k (x : ℝ) : ℝ := 1 / (h_inverse x) + 8

theorem k_of_five_eq_eight_point_five : k 5 = 8.5 := by
  sorry

end k_of_five_eq_eight_point_five_l880_88088


namespace a_minus_c_values_l880_88067

theorem a_minus_c_values (a b c : ℕ) 
  (h1 : a > b) 
  (h2 : a^2 - a*b - a*c + b*c = 7) : 
  a - c = 1 ∨ a - c = 7 := by
sorry

end a_minus_c_values_l880_88067


namespace train_length_l880_88080

/-- Calculates the length of a train given its speed and time to cross a pole -/
theorem train_length (speed_kmh : ℝ) (time_s : ℝ) : 
  speed_kmh = 56 → time_s = 9 → ∃ (length_m : ℝ), abs (length_m - 140.04) < 0.01 := by
  sorry

end train_length_l880_88080


namespace f_pi_8_equals_sqrt_2_l880_88003

noncomputable def f (x : ℝ) : ℝ := 
  1 / (2 * Real.tan x) + (Real.sin (x/2) * Real.cos (x/2)) / (2 * Real.cos (x/2)^2 - 1)

theorem f_pi_8_equals_sqrt_2 : f (π/8) = Real.sqrt 2 := by
  sorry

end f_pi_8_equals_sqrt_2_l880_88003


namespace integer_operation_problem_l880_88097

theorem integer_operation_problem : ∃! x : ℤ, 
  ∃ r : ℤ, 0 ≤ r ∧ r < 7 ∧ ((x - 77) * 8 = 37 * 7 + r) ∧ x = 110 := by
  sorry

end integer_operation_problem_l880_88097


namespace white_daisies_count_l880_88081

theorem white_daisies_count :
  ∀ (white pink red : ℕ),
    pink = 9 * white →
    red = 4 * pink - 3 →
    white + pink + red = 273 →
    white = 6 :=
by
  sorry

end white_daisies_count_l880_88081


namespace parabola_and_intersection_l880_88038

/-- Parabola with vertex at origin and focus on x-axis -/
structure Parabola where
  p : ℝ
  equation : ℝ → ℝ → Prop := fun x y => y^2 = 2 * p * x

/-- Point on the parabola -/
structure PointOnParabola (par : Parabola) where
  x : ℝ
  y : ℝ
  on_parabola : par.equation x y

/-- Line intersecting the parabola -/
structure IntersectingLine (par : Parabola) where
  k : ℝ
  b : ℝ
  equation : ℝ → ℝ → Prop := fun x y => y = k * x + b
  intersects_twice : ∃ (p1 p2 : PointOnParabola par), p1 ≠ p2 ∧ 
    equation p1.x p1.y ∧ equation p2.x p2.y

theorem parabola_and_intersection 
    (par : Parabola) 
    (A : PointOnParabola par)
    (h1 : A.x = 4)
    (h2 : (A.x + par.p / 2)^2 + A.y^2 = 6^2)
    (line : IntersectingLine par)
    (h3 : line.b = -2)
    (h4 : ∃ (B : PointOnParabola par), 
      line.equation B.x B.y ∧ (A.x + B.x) / 2 = 2) :
  par.p = 4 ∧ line.k = 2 := by sorry

end parabola_and_intersection_l880_88038


namespace students_in_all_classes_l880_88068

/-- Represents the number of students in each class combination --/
structure ClassCombinations where
  code_only : ℕ
  chess_only : ℕ
  photo_only : ℕ
  code_chess : ℕ
  code_photo : ℕ
  chess_photo : ℕ
  all_three : ℕ

/-- The problem statement and conditions --/
theorem students_in_all_classes 
  (total_students : ℕ)
  (code_students : ℕ)
  (chess_students : ℕ)
  (photo_students : ℕ)
  (multi_class_students : ℕ)
  (h1 : total_students = 25)
  (h2 : code_students = 12)
  (h3 : chess_students = 15)
  (h4 : photo_students = 10)
  (h5 : multi_class_students = 10)
  (combinations : ClassCombinations)
  (h6 : total_students = 
    combinations.code_only + combinations.chess_only + combinations.photo_only + 
    combinations.code_chess + combinations.code_photo + combinations.chess_photo + 
    combinations.all_three)
  (h7 : code_students = 
    combinations.code_only + combinations.code_chess + combinations.code_photo + 
    combinations.all_three)
  (h8 : chess_students = 
    combinations.chess_only + combinations.code_chess + combinations.chess_photo + 
    combinations.all_three)
  (h9 : photo_students = 
    combinations.photo_only + combinations.code_photo + combinations.chess_photo + 
    combinations.all_three)
  (h10 : multi_class_students = 
    combinations.code_chess + combinations.code_photo + combinations.chess_photo + 
    combinations.all_three) :
  combinations.all_three = 4 := by
  sorry


end students_in_all_classes_l880_88068


namespace parabola_point_range_l880_88059

-- Define the parabola function
def f (x : ℝ) : ℝ := x^2 - 4*x + 3

-- Define the theorem
theorem parabola_point_range :
  ∃ (m : ℝ), 
    (m > 0) ∧
    (∀ (x₁ x₂ : ℝ),
      (-1 < x₁ ∧ x₁ < 1) →
      (m - 1 < x₂ ∧ x₂ < m) →
      (f x₁ ≠ f x₂)) ∧
    ((2 ≤ m ∧ m ≤ 3) ∨ m ≥ 6) :=
sorry

end parabola_point_range_l880_88059


namespace bcm_hens_percentage_l880_88001

/-- Given a farm with chickens, calculate the percentage of Black Copper Marans hens -/
theorem bcm_hens_percentage 
  (total_chickens : ℕ) 
  (bcm_percentage : ℚ) 
  (bcm_hens : ℕ) 
  (h1 : total_chickens = 100) 
  (h2 : bcm_percentage = 1/5) 
  (h3 : bcm_hens = 16) : 
  (bcm_hens : ℚ) / (bcm_percentage * total_chickens) = 4/5 := by
sorry

end bcm_hens_percentage_l880_88001


namespace overlook_distance_proof_l880_88008

/-- The distance to Mount Overlook in miles -/
def distance_to_overlook : ℝ := 12

/-- Jeannie's hiking speed to Mount Overlook in miles per hour -/
def speed_to_overlook : ℝ := 4

/-- Jeannie's hiking speed from Mount Overlook in miles per hour -/
def speed_from_overlook : ℝ := 6

/-- Total time of the hike in hours -/
def total_time : ℝ := 5

theorem overlook_distance_proof :
  distance_to_overlook = 12 ∧
  (distance_to_overlook / speed_to_overlook + distance_to_overlook / speed_from_overlook = total_time) :=
by sorry

end overlook_distance_proof_l880_88008


namespace female_employees_count_l880_88076

/-- The total number of female employees in a company, given specific conditions. -/
theorem female_employees_count (total_employees : ℕ) (male_employees : ℕ) (female_managers : ℕ) :
  female_managers = 280 →
  (2 : ℚ) / 5 * total_employees = female_managers + (2 : ℚ) / 5 * male_employees →
  total_employees = male_employees + 700 →
  700 = total_employees - male_employees :=
by sorry

end female_employees_count_l880_88076


namespace greatest_common_multiple_15_10_under_150_l880_88077

def is_common_multiple (m n k : ℕ) : Prop := k % m = 0 ∧ k % n = 0

theorem greatest_common_multiple_15_10_under_150 :
  ∃ (k : ℕ), k < 150 ∧ 
             is_common_multiple 15 10 k ∧ 
             ∀ (j : ℕ), j < 150 → is_common_multiple 15 10 j → j ≤ k :=
by
  use 120
  sorry

#eval 120  -- Expected output: 120

end greatest_common_multiple_15_10_under_150_l880_88077


namespace square_area_from_perimeter_l880_88033

/-- Given a square with perimeter 48 meters, its area is 144 square meters. -/
theorem square_area_from_perimeter :
  ∀ s : ℝ,
  s > 0 →
  4 * s = 48 →
  s * s = 144 :=
by
  sorry

end square_area_from_perimeter_l880_88033


namespace geometric_sequence_fourth_term_l880_88054

theorem geometric_sequence_fourth_term (a₁ a₂ a₃ : ℝ) (h₁ : a₁ = 2^(1/4))
    (h₂ : a₂ = 2^(1/6)) (h₃ : a₃ = 2^(1/12)) :
  let r := a₂ / a₁
  a₃ * r = 1 := by
sorry

end geometric_sequence_fourth_term_l880_88054


namespace min_value_sum_reciprocals_l880_88073

theorem min_value_sum_reciprocals (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) 
  (h4 : a + b + c = 3) : 
  (∀ x y z : ℝ, x > 0 → y > 0 → z > 0 → x + y + z = 3 → 
    1 / (x + y) + 1 / z ≥ 1 / (a + b) + 1 / c) → 
  1 / (a + b) + 1 / c = 4 / 3 := by
sorry

end min_value_sum_reciprocals_l880_88073


namespace exam_students_count_l880_88032

theorem exam_students_count :
  ∀ (N : ℕ) (T : ℝ),
  (T = 80 * N) →
  ((T - 350) / (N - 5) = 90) →
  N = 10 := by
sorry

end exam_students_count_l880_88032


namespace calculation_proof_l880_88029

theorem calculation_proof :
  (13 + (-7) + (-6) = 0) ∧
  ((-8) * (-4/3) * (-0.125) * (5/4) = -5/3) := by
sorry

end calculation_proof_l880_88029


namespace value_range_sqrt_16_minus_4_pow_x_l880_88094

theorem value_range_sqrt_16_minus_4_pow_x :
  ∀ x : ℝ, 0 ≤ Real.sqrt (16 - 4^x) ∧ Real.sqrt (16 - 4^x) < 4 := by
  sorry

end value_range_sqrt_16_minus_4_pow_x_l880_88094


namespace sally_out_of_pocket_l880_88062

theorem sally_out_of_pocket (provided : ℕ) (book_cost : ℕ) (students : ℕ) :
  provided = 320 →
  book_cost = 12 →
  students = 30 →
  (students * book_cost - provided : ℤ) = 40 := by
  sorry

end sally_out_of_pocket_l880_88062


namespace prob_not_all_same_value_l880_88012

/-- The number of sides on each die -/
def sides : ℕ := 8

/-- The number of dice rolled -/
def num_dice : ℕ := 5

/-- The probability that five fair 8-sided dice won't all show the same number -/
def prob_not_all_same : ℚ :=
  1 - (sides : ℚ) / sides ^ num_dice

theorem prob_not_all_same_value :
  prob_not_all_same = 4095 / 4096 :=
sorry

end prob_not_all_same_value_l880_88012


namespace abs_neg_two_l880_88020

theorem abs_neg_two : |(-2 : ℝ)| = 2 := by sorry

end abs_neg_two_l880_88020


namespace parabola_theorem_l880_88087

/-- A parabola passing through specific points with given conditions -/
def Parabola (a b : ℝ) : ℝ → ℝ := fun x ↦ a * x^2 + b * x + 1

theorem parabola_theorem (a b : ℝ) (m : ℝ) :
  Parabola a b 1 = -2 ∧ 
  Parabola a b (-2) = 13 ∧ 
  ∃ y₁ y₂ : ℝ, Parabola a b 5 = y₁ ∧ 
             Parabola a b m = y₂ ∧ 
             y₂ = 12 - y₁ ∧ 
             y₁ ≠ y₂ 
  → a = 1 ∧ b = -4 ∧ m = -1 := by
  sorry

end parabola_theorem_l880_88087


namespace percentage_first_division_l880_88092

theorem percentage_first_division (total_students : ℕ) 
  (second_division_percent : ℚ) (just_passed : ℕ) :
  total_students = 300 →
  second_division_percent = 54 / 100 →
  just_passed = 48 →
  ∃ (first_division_percent : ℚ),
    first_division_percent + second_division_percent + (just_passed : ℚ) / total_students = 1 ∧
    first_division_percent = 30 / 100 := by
  sorry

end percentage_first_division_l880_88092


namespace new_failing_grades_saturday_is_nine_l880_88024

/-- The number of new failing grades that appear on Saturday -/
def new_failing_grades_saturday : ℕ :=
  let group1_students : ℕ := 7
  let group2_students : ℕ := 9
  let days_mon_to_sat : ℕ := 6
  let failing_grades_mon_to_fri : ℕ := 30
  let group1_failing_grades := group1_students * (days_mon_to_sat / 2)
  let group2_failing_grades := group2_students * (days_mon_to_sat / 3)
  let total_failing_grades := group1_failing_grades + group2_failing_grades
  total_failing_grades - failing_grades_mon_to_fri

theorem new_failing_grades_saturday_is_nine :
  new_failing_grades_saturday = 9 := by
  sorry

end new_failing_grades_saturday_is_nine_l880_88024


namespace quartic_roots_sum_product_l880_88040

theorem quartic_roots_sum_product (a b : ℝ) : 
  a^4 - 6*a - 2 = 0 → b^4 - 6*b - 2 = 0 → a * b + a + b = 4 := by
  sorry

end quartic_roots_sum_product_l880_88040


namespace sample_standard_deviation_l880_88057

/-- Given a sample of 5 individuals with values a, 0, 1, 2, 3, where the average is 1,
    prove that the standard deviation of the sample is √2. -/
theorem sample_standard_deviation (a : ℝ) : 
  (a + 0 + 1 + 2 + 3) / 5 = 1 →
  Real.sqrt (((a - 1)^2 + (-1)^2 + 0^2 + 1^2 + 2^2) / 5) = Real.sqrt 2 := by
  sorry

end sample_standard_deviation_l880_88057


namespace magnitude_of_one_minus_i_to_eighth_l880_88086

theorem magnitude_of_one_minus_i_to_eighth : Complex.abs ((1 - Complex.I) ^ 8) = 16 := by sorry

end magnitude_of_one_minus_i_to_eighth_l880_88086


namespace arc_length_problem_l880_88072

theorem arc_length_problem (r : ℝ) (θ : ℝ) (a : ℝ) :
  r = 18 →
  θ = π / 3 →
  r * θ = a * π →
  a = 6 := by
  sorry

end arc_length_problem_l880_88072


namespace circle_properties_l880_88042

theorem circle_properties (x y : ℝ) (h : x^2 + y^2 - 4*x + 1 = 0) :
  (∃ k : ℝ, k = Real.sqrt 3 ∧ 
    (∀ t : ℝ, x ≠ 0 → y / x ≤ k) ∧
    (∃ x₀ y₀ : ℝ, x₀ ≠ 0 ∧ x₀^2 + y₀^2 - 4*x₀ + 1 = 0 ∧ y₀ / x₀ = k)) ∧
  (∃ k : ℝ, k = -Real.sqrt 3 ∧ 
    (∀ t : ℝ, x ≠ 0 → k ≤ y / x) ∧
    (∃ x₀ y₀ : ℝ, x₀ ≠ 0 ∧ x₀^2 + y₀^2 - 4*x₀ + 1 = 0 ∧ y₀ / x₀ = k)) ∧
  (∃ k : ℝ, k = -2 + Real.sqrt 6 ∧ 
    (∀ t : ℝ, y - x ≤ k) ∧
    (∃ x₀ y₀ : ℝ, x₀^2 + y₀^2 - 4*x₀ + 1 = 0 ∧ y₀ - x₀ = k)) ∧
  (∃ k : ℝ, k = -2 - Real.sqrt 6 ∧ 
    (∀ t : ℝ, k ≤ y - x) ∧
    (∃ x₀ y₀ : ℝ, x₀^2 + y₀^2 - 4*x₀ + 1 = 0 ∧ y₀ - x₀ = k)) ∧
  (∃ k : ℝ, k = 7 + 4 * Real.sqrt 3 ∧ 
    (∀ t : ℝ, x^2 + y^2 ≤ k) ∧
    (∃ x₀ y₀ : ℝ, x₀^2 + y₀^2 - 4*x₀ + 1 = 0 ∧ x₀^2 + y₀^2 = k)) ∧
  (∃ k : ℝ, k = 7 - 4 * Real.sqrt 3 ∧ 
    (∀ t : ℝ, k ≤ x^2 + y^2) ∧
    (∃ x₀ y₀ : ℝ, x₀^2 + y₀^2 - 4*x₀ + 1 = 0 ∧ x₀^2 + y₀^2 = k)) :=
by sorry

end circle_properties_l880_88042


namespace sqrt_x_plus_5_real_l880_88004

theorem sqrt_x_plus_5_real (x : ℝ) : (∃ y : ℝ, y^2 = x + 5) ↔ x ≥ -5 := by sorry

end sqrt_x_plus_5_real_l880_88004


namespace problem_statement_l880_88026

theorem problem_statement :
  ((-2023)^0 : ℝ) - 4 * Real.sin (π/4) + |(-Real.sqrt 8)| = 1 := by sorry

end problem_statement_l880_88026


namespace correct_production_l880_88093

/-- Represents the production of each shift in a car manufacturing plant. -/
structure ShiftProduction where
  day : ℕ
  second : ℕ
  third : ℕ

/-- Checks if the given shift production satisfies the problem conditions. -/
def satisfiesConditions (p : ShiftProduction) : Prop :=
  p.day = 4 * p.second ∧
  p.third = (3 * p.second) / 2 ∧
  p.day + p.second + p.third = 8000

/-- Theorem stating that the given production numbers satisfy the problem conditions. -/
theorem correct_production : satisfiesConditions ⟨4923, 1231, 1846⟩ := by
  sorry

#check correct_production

end correct_production_l880_88093


namespace batch_size_calculation_l880_88022

theorem batch_size_calculation (sample_size : ℕ) (probability : ℚ) (total : ℕ) : 
  sample_size = 30 →
  probability = 1/4 →
  (sample_size : ℚ) / probability = total →
  total = 120 := by
sorry

end batch_size_calculation_l880_88022


namespace shopping_mall_problem_l880_88023

/-- Shopping mall product purchase problem -/
theorem shopping_mall_problem 
  (cost_price_A cost_price_B : ℚ)
  (quantity_A quantity_B : ℕ)
  (selling_price_A selling_price_B : ℚ) :
  cost_price_A = cost_price_B - 2 →
  80 / cost_price_A = 100 / cost_price_B →
  quantity_A = 3 * quantity_B - 5 →
  quantity_A + quantity_B ≤ 95 →
  selling_price_A = 12 →
  selling_price_B = 15 →
  (selling_price_A - cost_price_A) * quantity_A + 
  (selling_price_B - cost_price_B) * quantity_B > 380 →
  (cost_price_A = 8 ∧ cost_price_B = 10) ∧
  (∀ n : ℕ, n ≤ quantity_B → n ≤ 25) ∧
  ((quantity_A = 67 ∧ quantity_B = 24) ∨ 
   (quantity_A = 70 ∧ quantity_B = 25)) := by
sorry


end shopping_mall_problem_l880_88023


namespace wizard_elixir_combinations_l880_88095

-- Define the number of herbs and gems
def num_herbs : ℕ := 4
def num_gems : ℕ := 6

-- Define the number of incompatible combinations for one gem
def incompatible_combinations : ℕ := 3

-- Define the number of herbs that can be used with the specific gem
def specific_gem_combinations : ℕ := 1

-- Theorem statement
theorem wizard_elixir_combinations :
  let total_combinations := num_herbs * num_gems
  let remaining_after_incompatible := total_combinations - incompatible_combinations
  let valid_combinations := remaining_after_incompatible - (num_herbs - specific_gem_combinations)
  valid_combinations = 18 := by
  sorry


end wizard_elixir_combinations_l880_88095


namespace inequality_solution_l880_88061

-- Define the parameter a
variable (a : ℝ)

-- Define the condition a < -1
variable (h : a < -1)

-- Define the solution set
def solution_set (x : ℝ) : Prop :=
  x ∈ (Set.Iio (-1) ∪ Set.Ioi (1/a))

-- State the theorem
theorem inequality_solution :
  ∀ x, (a * x - 1) / (x + 1) < 0 ↔ solution_set a x :=
by sorry

end inequality_solution_l880_88061


namespace no_prime_solution_l880_88078

theorem no_prime_solution :
  ∀ p : ℕ, Prime p → 2 * p^3 - 5 * p + 14 ≠ 0 := by
  sorry

end no_prime_solution_l880_88078


namespace triangle_area_l880_88009

theorem triangle_area (a b c : ℝ) (α : ℝ) (h1 : a = 14)
  (h2 : α = Real.pi / 3) (h3 : b / c = 8 / 5) :
  (1 / 2) * b * c * Real.sin α = 40 * Real.sqrt 3 := by
  sorry

end triangle_area_l880_88009


namespace intersection_point_sum_l880_88030

-- Define the function f
variable (f : ℝ → ℝ)

-- State the theorem
theorem intersection_point_sum (h1 : f (-3) = 3) (h2 : f 1 = 3)
  (h3 : ∃! p : ℝ × ℝ, f p.1 = f (p.1 - 4) ∧ f p.1 = p.2) :
  ∃ p : ℝ × ℝ, f p.1 = f (p.1 - 4) ∧ f p.1 = p.2 ∧ p.1 + p.2 = 4 := by
sorry

end intersection_point_sum_l880_88030


namespace isosceles_triangle_from_tangent_sum_l880_88071

/-- If the sum of tangents of angle differences in a triangle is zero, then the triangle is isosceles. -/
theorem isosceles_triangle_from_tangent_sum (A B C : ℝ) 
  (h_triangle : A + B + C = π) 
  (h_tangent_sum : Real.tan (A - B) + Real.tan (B - C) + Real.tan (C - A) = 0) : 
  A = B ∨ B = C ∨ C = A :=
sorry

end isosceles_triangle_from_tangent_sum_l880_88071


namespace unique_m_satisfying_lcm_conditions_l880_88006

theorem unique_m_satisfying_lcm_conditions : 
  ∃! m : ℕ+, (Nat.lcm 36 m.val = 180) ∧ (Nat.lcm m.val 45 = 225) ∧ (m.val = 25) :=
by sorry

end unique_m_satisfying_lcm_conditions_l880_88006


namespace sin_theta_value_l880_88049

theorem sin_theta_value (θ : Real) (h1 : 0 < θ ∧ θ < π / 2) (h2 : Real.sin (θ - π / 3) = 5 / 13) :
  Real.sin θ = (5 + 12 * Real.sqrt 3) / 26 := by
  sorry

end sin_theta_value_l880_88049


namespace nathans_earnings_186_l880_88013

/-- Calculates the total earnings from Nathan's harvest --/
def nathans_earnings (strawberry_plants : ℕ) (tomato_plants : ℕ) 
  (strawberries_per_plant : ℕ) (tomatoes_per_plant : ℕ) 
  (fruits_per_basket : ℕ) (strawberry_basket_price : ℕ) (tomato_basket_price : ℕ) : ℕ :=
  let total_strawberries := strawberry_plants * strawberries_per_plant
  let total_tomatoes := tomato_plants * tomatoes_per_plant
  let strawberry_baskets := total_strawberries / fruits_per_basket
  let tomato_baskets := total_tomatoes / fruits_per_basket
  let strawberry_earnings := strawberry_baskets * strawberry_basket_price
  let tomato_earnings := tomato_baskets * tomato_basket_price
  strawberry_earnings + tomato_earnings

/-- Theorem stating that Nathan's earnings from his harvest equal $186 --/
theorem nathans_earnings_186 :
  nathans_earnings 5 7 14 16 7 9 6 = 186 := by
  sorry

end nathans_earnings_186_l880_88013


namespace seating_theorem_l880_88070

/-- The number of seats in the row -/
def n : ℕ := 8

/-- The number of people to be seated -/
def k : ℕ := 2

/-- The number of different seating arrangements for k people in n seats,
    with empty seats required on both sides of each person -/
def seating_arrangements (n k : ℕ) : ℕ := sorry

/-- Theorem stating that the number of seating arrangements
    for 2 people in 8 seats is 20 -/
theorem seating_theorem : seating_arrangements n k = 20 := by
  sorry

end seating_theorem_l880_88070


namespace quadratic_downward_solution_nonempty_l880_88017

/-- A quadratic function f(x) = ax² + bx + c opens downwards if a < 0 -/
def opens_downwards (a b c : ℝ) : Prop := a < 0

/-- The solution set of ax² + bx + c < 0 is not empty -/
def solution_set_nonempty (a b c : ℝ) : Prop :=
  ∃ x : ℝ, a * x^2 + b * x + c < 0

/-- If a quadratic function opens downwards, its solution set for f(x) < 0 is not empty -/
theorem quadratic_downward_solution_nonempty (a b c : ℝ) :
  opens_downwards a b c → solution_set_nonempty a b c := by
  sorry

end quadratic_downward_solution_nonempty_l880_88017


namespace bench_cost_proof_l880_88025

/-- The cost of the bench in dollars -/
def bench_cost : ℝ := 150

/-- The cost of the garden table in dollars -/
def table_cost : ℝ := 2 * bench_cost

/-- The combined cost of the bench and garden table in dollars -/
def combined_cost : ℝ := 450

theorem bench_cost_proof : bench_cost = 150 := by sorry

end bench_cost_proof_l880_88025


namespace cos_two_beta_equals_one_l880_88014

theorem cos_two_beta_equals_one (α β : ℝ) 
  (h : Real.sin (α - β) * Real.cos α - Real.cos (α - β) * Real.sin α = 0) : 
  Real.cos (2 * β) = 1 := by
  sorry

end cos_two_beta_equals_one_l880_88014


namespace f_is_even_l880_88041

def isOdd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x

def isEven (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

theorem f_is_even (g : ℝ → ℝ) (h : isOdd g) :
  isEven (fun x ↦ |g (x^4)|) := by
  sorry

end f_is_even_l880_88041


namespace inequality_proof_l880_88050

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  (a * b) / (a^2 + 3 * b^2) + (b * c) / (b^2 + 3 * c^2) + (c * a) / (c^2 + 3 * a^2) ≤ 3/4 := by
  sorry

end inequality_proof_l880_88050


namespace inequality_proof_l880_88066

theorem inequality_proof (a b : ℝ) (n : ℕ) (ha : a > 0) (hb : b > 0) (hn : n > 0) :
  (1 + a / b) ^ n + (1 + b / a) ^ n ≥ 2^(n + 1) := by
  sorry

end inequality_proof_l880_88066


namespace group_meal_cost_l880_88064

/-- Calculates the total cost for a group to eat at a restaurant --/
def total_cost (total_people : ℕ) (num_kids : ℕ) (adult_meal_cost : ℕ) : ℕ :=
  (total_people - num_kids) * adult_meal_cost

/-- Theorem: The total cost for the given group is $15 --/
theorem group_meal_cost : total_cost 12 7 3 = 15 := by
  sorry

end group_meal_cost_l880_88064


namespace joes_team_draws_l880_88044

/-- Represents a team's performance in a soccer tournament --/
structure TeamPerformance where
  wins : ℕ
  draws : ℕ

/-- Calculates the total points for a team --/
def calculatePoints (team : TeamPerformance) : ℕ :=
  3 * team.wins + team.draws

theorem joes_team_draws : 
  ∀ (joes_team first_place : TeamPerformance),
    joes_team.wins = 1 →
    first_place.wins = 2 →
    first_place.draws = 2 →
    calculatePoints first_place = calculatePoints joes_team + 2 →
    joes_team.draws = 3 := by
  sorry


end joes_team_draws_l880_88044


namespace zeroth_power_of_nonzero_is_one_l880_88005

theorem zeroth_power_of_nonzero_is_one (x : ℚ) (h : x ≠ 0) : x^0 = 1 := by
  sorry

end zeroth_power_of_nonzero_is_one_l880_88005


namespace probability_at_least_two_green_l880_88084

theorem probability_at_least_two_green (total : ℕ) (red : ℕ) (green : ℕ) (yellow : ℕ) :
  total = 10 ∧ red = 5 ∧ green = 3 ∧ yellow = 2 →
  (Nat.choose total 3 : ℚ) ≠ 0 →
  (Nat.choose green 2 * Nat.choose (total - green) 1 + Nat.choose green 3 : ℚ) / Nat.choose total 3 = 11 / 60 :=
by sorry

end probability_at_least_two_green_l880_88084


namespace polynomial_remainder_l880_88019

theorem polynomial_remainder (q : ℝ → ℝ) (h1 : ∃ r1 : ℝ → ℝ, ∀ x, q x = (x - 1) * (r1 x) + 10)
  (h2 : ∃ r2 : ℝ → ℝ, ∀ x, q x = (x + 3) * (r2 x) - 8) :
  ∃ r : ℝ → ℝ, ∀ x, q x = (x - 1) * (x + 3) * (r x) + 4.5 * x + 5.5 := by
sorry

end polynomial_remainder_l880_88019


namespace xy_value_l880_88045

theorem xy_value (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h_eq : Real.sqrt (Real.log x) + Real.sqrt (Real.log y) + 
          Real.log (Real.sqrt x) + Real.log (Real.sqrt y) + 
          Real.log (x^(1/4)) + Real.log (y^(1/4)) = 150)
  (h_int1 : ∃ n : ℤ, Real.sqrt (Real.log x) = n)
  (h_int2 : ∃ n : ℤ, Real.sqrt (Real.log y) = n)
  (h_int3 : ∃ n : ℤ, Real.log (Real.sqrt x) = n)
  (h_int4 : ∃ n : ℤ, Real.log (Real.sqrt y) = n)
  (h_int5 : ∃ n : ℤ, Real.log (x^(1/4)) = n)
  (h_int6 : ∃ n : ℤ, Real.log (y^(1/4)) = n) :
  x * y = Real.exp 340 := by
  sorry

end xy_value_l880_88045


namespace quadratic_function_property_l880_88002

/-- Given a quadratic function f(x) = ax^2 + bx + c with a ≠ 0,
    if f(r) = f(s) = k for two distinct points r and s,
    then f(r + s) = c -/
theorem quadratic_function_property
  (a b c r s k : ℝ)
  (h_a : a ≠ 0)
  (h_distinct : r ≠ s)
  (h_fr : a * r^2 + b * r + c = k)
  (h_fs : a * s^2 + b * s + c = k) :
  a * (r + s)^2 + b * (r + s) + c = c :=
sorry

end quadratic_function_property_l880_88002


namespace max_value_2x_3y_l880_88065

theorem max_value_2x_3y (x y : ℝ) (h : x^2 + y^2 = 16*x + 8*y + 20) :
  ∃ (M : ℝ), M = 33 ∧ 2*x + 3*y ≤ M ∧ ∃ (x₀ y₀ : ℝ), 2*x₀ + 3*y₀ = M ∧ x₀^2 + y₀^2 = 16*x₀ + 8*y₀ + 20 :=
sorry

end max_value_2x_3y_l880_88065


namespace parallel_lines_unique_m_l880_88075

/-- Given two lines l₁ and l₂, prove that m = -4 is the only value that makes them parallel -/
theorem parallel_lines_unique_m : ∃! m : ℝ, 
  (∀ x y : ℝ, (m - 2) * x - 3 * y - 1 = 0 ↔ ((m - 2) / 3) * x - 1 / 3 = y) ∧ 
  (∀ x y : ℝ, m * x + (m + 2) * y + 1 = 0 ↔ (-m / (m + 2)) * x - 1 / (m + 2) = y) ∧
  ((m - 2) / 3 = -m / (m + 2)) ∧
  (m - 2 ≠ -m) ∧
  m = -4 := by
  sorry

end parallel_lines_unique_m_l880_88075


namespace homework_problem_l880_88027

theorem homework_problem (total : ℕ) (ratio_incomplete : ℕ) (ratio_complete : ℕ) 
  (h_total : total = 15)
  (h_ratio : ratio_incomplete = 3 ∧ ratio_complete = 2) :
  ∃ (completed : ℕ), completed = 6 ∧ 
    ratio_incomplete * completed = ratio_complete * (total - completed) :=
by sorry

end homework_problem_l880_88027


namespace smallest_integer_sum_product_squares_l880_88048

theorem smallest_integer_sum_product_squares :
  ∃ (a : ℕ), a > 0 ∧ 
  (∃ (b : ℕ), 10 + a = b^2) ∧ 
  (∃ (c : ℕ), 10 * a = c^2) ∧
  (∀ (x : ℕ), x > 0 ∧ x < a → 
    (¬∃ (y : ℕ), 10 + x = y^2) ∨ 
    (¬∃ (z : ℕ), 10 * x = z^2)) ∧
  a = 90 := by
sorry

end smallest_integer_sum_product_squares_l880_88048


namespace factory_growth_rate_l880_88018

theorem factory_growth_rate (P a b x : ℝ) (h1 : P > 0) (h2 : a > -1) (h3 : b > -1)
  (h4 : (1 + x)^2 = (1 + a) * (1 + b)) : x ≤ max a b := by
  sorry

end factory_growth_rate_l880_88018


namespace M_equals_N_l880_88056

def M : Set ℤ := {u | ∃ m n l : ℤ, u = 12*m + 8*n + 4*l}
def N : Set ℤ := {u | ∃ p q r : ℤ, u = 20*p + 16*q + 12*r}

theorem M_equals_N : M = N := by
  sorry

end M_equals_N_l880_88056


namespace winningScoresCount_is_nineteen_l880_88011

/-- Represents a cross country meet between two teams -/
structure CrossCountryMeet where
  /-- Number of runners in each team -/
  runnersPerTeam : Nat
  /-- Total number of runners -/
  totalRunners : Nat
  /-- The sum of all positions -/
  totalPositionSum : Nat
  /-- Condition that there are 2 teams -/
  twoTeams : totalRunners = 2 * runnersPerTeam
  /-- Condition that the total sum of positions is correct -/
  validTotalSum : totalPositionSum = totalRunners * (totalRunners + 1) / 2

/-- The number of different winning scores possible in a cross country meet -/
def winningScoresCount (meet : CrossCountryMeet) : Nat :=
  meet.totalPositionSum / 2 - (meet.runnersPerTeam * (meet.runnersPerTeam + 1) / 2) + 1

/-- Theorem stating that the number of different winning scores is 19 -/
theorem winningScoresCount_is_nineteen :
  ∀ (meet : CrossCountryMeet),
    meet.runnersPerTeam = 6 →
    winningScoresCount meet = 19 :=
by
  sorry


end winningScoresCount_is_nineteen_l880_88011


namespace remainder_of_N_mod_45_l880_88063

def concatenate_integers (n : ℕ) : ℕ :=
  -- Definition of concatenating integers from 1 to n
  sorry

def N : ℕ := concatenate_integers 44

theorem remainder_of_N_mod_45 : N % 45 = 9 := by
  sorry

end remainder_of_N_mod_45_l880_88063


namespace eight_teams_twentyeight_games_unique_solution_eight_teams_l880_88028

/-- The number of games played when each team in a conference plays every other team once -/
def games_played (n : ℕ) : ℕ := n * (n - 1) / 2

/-- The theorem stating that 8 teams results in 28 games played -/
theorem eight_teams_twentyeight_games :
  ∃ (n : ℕ), n > 0 ∧ games_played n = 28 ∧ n = 8 := by
  sorry

/-- The theorem proving that 8 is the only positive integer satisfying the conditions -/
theorem unique_solution_eight_teams :
  ∀ (n : ℕ), n > 0 → games_played n = 28 → n = 8 := by
  sorry

end eight_teams_twentyeight_games_unique_solution_eight_teams_l880_88028


namespace parallel_vectors_l880_88074

/-- Two vectors in ℝ² are parallel if and only if their cross product is zero -/
axiom parallel_iff_cross_zero {u v : ℝ × ℝ} : 
  (∃ (k : ℝ), u = k • v ∨ v = k • u) ↔ u.1 * v.2 - u.2 * v.1 = 0

/-- Given vectors a and b, prove that a is parallel to b if and only if y = -6 -/
theorem parallel_vectors (a b : ℝ × ℝ) (h1 : a = (-1, 3)) (h2 : b = (2, y)) :
  (∃ (k : ℝ), a = k • b ∨ b = k • a) ↔ y = -6 :=
by sorry

end parallel_vectors_l880_88074
