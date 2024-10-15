import Mathlib

namespace NUMINAMATH_CALUDE_power_of_two_equality_l970_97003

theorem power_of_two_equality : (2^36 / 8 = 2^x) → x = 33 := by
  sorry

end NUMINAMATH_CALUDE_power_of_two_equality_l970_97003


namespace NUMINAMATH_CALUDE_square_1369_product_l970_97016

theorem square_1369_product (x : ℤ) (h : x^2 = 1369) : (x + 3) * (x - 3) = 1360 := by
  sorry

end NUMINAMATH_CALUDE_square_1369_product_l970_97016


namespace NUMINAMATH_CALUDE_gcd_problem_l970_97027

theorem gcd_problem : Nat.gcd (122^2 + 234^2 + 345^2 + 10) (123^2 + 233^2 + 347^2 + 10) = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_problem_l970_97027


namespace NUMINAMATH_CALUDE_least_five_digit_palindrome_div_25_l970_97093

/-- A function that checks if a natural number is a five-digit palindrome -/
def is_five_digit_palindrome (n : ℕ) : Prop :=
  10000 ≤ n ∧ n < 100000 ∧ 
  (n / 10000 = n % 10) ∧ 
  ((n / 1000) % 10 = (n / 10) % 10)

/-- The theorem stating that 10201 is the least five-digit palindrome divisible by 25 -/
theorem least_five_digit_palindrome_div_25 :
  ∀ n : ℕ, is_five_digit_palindrome n ∧ n % 25 = 0 → n ≥ 10201 :=
sorry

end NUMINAMATH_CALUDE_least_five_digit_palindrome_div_25_l970_97093


namespace NUMINAMATH_CALUDE_work_completion_theorem_l970_97011

/-- Represents the number of men originally employed -/
def original_men : ℕ := 17

/-- Represents the number of days originally required to finish the work -/
def original_days : ℕ := 8

/-- Represents the number of additional men who joined -/
def additional_men : ℕ := 10

/-- Represents the number of days saved after additional men joined -/
def days_saved : ℕ := 3

/-- Theorem stating that the given conditions lead to the correct number of original men -/
theorem work_completion_theorem :
  (original_men * original_days = (original_men + additional_men) * (original_days - days_saved)) ∧
  (original_men ≥ 1) ∧
  (∀ m : ℕ, m < original_men →
    m * original_days ≠ (m + additional_men) * (original_days - days_saved)) :=
by sorry

end NUMINAMATH_CALUDE_work_completion_theorem_l970_97011


namespace NUMINAMATH_CALUDE_sum_of_n_for_perfect_square_l970_97069

theorem sum_of_n_for_perfect_square : ∃ (S : Finset ℕ),
  (∀ n ∈ S, n < 2023 ∧ ∃ k : ℕ, 2 * n^2 + 3 * n = k^2) ∧
  (∀ n : ℕ, n < 2023 → (∃ k : ℕ, 2 * n^2 + 3 * n = k^2) → n ∈ S) ∧
  S.sum id = 444 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_n_for_perfect_square_l970_97069


namespace NUMINAMATH_CALUDE_next_simultaneous_occurrence_l970_97083

def museum_interval : ℕ := 18
def library_interval : ℕ := 24
def town_hall_interval : ℕ := 30

def minutes_in_hour : ℕ := 60

theorem next_simultaneous_occurrence :
  ∃ (h : ℕ), h * minutes_in_hour = lcm museum_interval (lcm library_interval town_hall_interval) ∧ h = 6 := by
  sorry

end NUMINAMATH_CALUDE_next_simultaneous_occurrence_l970_97083


namespace NUMINAMATH_CALUDE_youngest_child_age_l970_97008

/-- Represents the age of the youngest child -/
def youngest_age : ℕ → Prop := λ x =>
  -- There are 5 children
  -- Children are born at intervals of 3 years each
  -- The sum of their ages is 60 years
  x + (x + 3) + (x + 6) + (x + 9) + (x + 12) = 60

/-- Proves that the age of the youngest child is 6 years -/
theorem youngest_child_age : youngest_age 6 := by
  sorry

end NUMINAMATH_CALUDE_youngest_child_age_l970_97008


namespace NUMINAMATH_CALUDE_orthic_triangle_inradius_bound_l970_97087

/-- Given a triangle ABC with circumradius R = 1 and inradius r, 
    the inradius P of its orthic triangle A'B'C' satisfies P ≤ 1 - (1/3)(1+r)^2 -/
theorem orthic_triangle_inradius_bound (R r P : ℝ) : 
  R = 1 → 0 < r → r ≤ 1/2 → P ≤ 1 - (1/3) * (1 + r)^2 := by
  sorry

end NUMINAMATH_CALUDE_orthic_triangle_inradius_bound_l970_97087


namespace NUMINAMATH_CALUDE_shooting_game_propositions_l970_97019

variable (p₁ p₂ : Prop)

theorem shooting_game_propositions :
  -- Both shots hit the airplane
  (p₁ ∧ p₂) = (p₁ ∧ p₂) ∧
  -- Both shots missed the airplane
  (¬p₁ ∧ ¬p₂) = (¬p₁ ∧ ¬p₂) ∧
  -- Exactly one shot hit the airplane
  ((p₁ ∧ ¬p₂) ∨ (p₂ ∧ ¬p₁)) = ((p₁ ∧ ¬p₂) ∨ (p₂ ∧ ¬p₁)) ∧
  -- At least one shot hit the airplane
  (p₁ ∨ p₂) = (p₁ ∨ p₂) := by sorry

end NUMINAMATH_CALUDE_shooting_game_propositions_l970_97019


namespace NUMINAMATH_CALUDE_midpoint_quadrilateral_perimeter_l970_97024

/-- A rectangle with a diagonal of 8 units -/
structure Rectangle :=
  (diagonal : ℝ)
  (diagonal_eq : diagonal = 8)

/-- A quadrilateral formed by connecting the midpoints of the sides of a rectangle -/
def MidpointQuadrilateral (rect : Rectangle) : Set (ℝ × ℝ) :=
  sorry

/-- The perimeter of a quadrilateral -/
def perimeter (quad : Set (ℝ × ℝ)) : ℝ :=
  sorry

theorem midpoint_quadrilateral_perimeter (rect : Rectangle) :
  perimeter (MidpointQuadrilateral rect) = 16 :=
sorry

end NUMINAMATH_CALUDE_midpoint_quadrilateral_perimeter_l970_97024


namespace NUMINAMATH_CALUDE_min_distance_theorem_l970_97064

/-- Given a line segment AB of length 2 with midpoint C, where A moves on the x-axis and B moves on the y-axis. -/
def line_segment (A B C : ℝ × ℝ) : Prop :=
  norm (A - B) = 2 ∧ C = (A + B) / 2 ∧ A.2 = 0 ∧ B.1 = 0

/-- The trajectory of point C is a circle with equation x² + y² = 1 -/
def trajectory (C : ℝ × ℝ) : Prop :=
  C.1^2 + C.2^2 = 1

/-- The line √2ax + by = 1 intersects the trajectory at points C and D -/
def intersecting_line (a b : ℝ) (C D : ℝ × ℝ) : Prop :=
  trajectory C ∧ trajectory D ∧ 
  Real.sqrt 2 * a * C.1 + b * C.2 = 1 ∧
  Real.sqrt 2 * a * D.1 + b * D.2 = 1

/-- Triangle COD is a right-angled triangle with O as the origin -/
def right_triangle (C D : ℝ × ℝ) : Prop :=
  (C.1 * D.1 + C.2 * D.2) = 0

/-- Point P has coordinates (a, b) -/
def point_P (P : ℝ × ℝ) (a b : ℝ) : Prop :=
  P = (a, b)

/-- The main theorem: The minimum distance between P(a, b) and (0, 1) is √2 - 1 -/
theorem min_distance_theorem (A B C D P : ℝ × ℝ) (a b : ℝ) :
  line_segment A B C →
  trajectory C →
  intersecting_line a b C D →
  right_triangle C D →
  point_P P a b →
  (∃ (min_dist : ℝ), ∀ (a' b' : ℝ), 
    norm ((a', b') - (0, 1)) ≥ min_dist ∧
    min_dist = Real.sqrt 2 - 1) :=
sorry

end NUMINAMATH_CALUDE_min_distance_theorem_l970_97064


namespace NUMINAMATH_CALUDE_percent_palindromes_with_seven_l970_97037

/-- A palindrome between 1000 and 2000 -/
structure Palindrome :=
  (x y : Fin 10)

/-- Checks if a palindrome contains at least one 7 -/
def containsSeven (p : Palindrome) : Prop :=
  p.x = 7 ∨ p.y = 7

/-- The set of all palindromes between 1000 and 2000 -/
def allPalindromes : Finset Palindrome :=
  sorry

/-- The set of palindromes containing at least one 7 -/
def palindromesWithSeven : Finset Palindrome :=
  sorry

theorem percent_palindromes_with_seven :
  (palindromesWithSeven.card : ℚ) / allPalindromes.card = 19 / 100 := by
  sorry

end NUMINAMATH_CALUDE_percent_palindromes_with_seven_l970_97037


namespace NUMINAMATH_CALUDE_school_population_l970_97085

theorem school_population (num_boys : ℕ) (difference : ℕ) (num_girls : ℕ) : 
  num_boys = 1145 → 
  num_boys = num_girls + difference → 
  difference = 510 → 
  num_girls = 635 := by
sorry

end NUMINAMATH_CALUDE_school_population_l970_97085


namespace NUMINAMATH_CALUDE_complex_modulus_problem_l970_97055

theorem complex_modulus_problem : 
  let i : ℂ := Complex.I
  let z : ℂ := 1 + 5 / (2 - i) * i
  Complex.abs z = Real.sqrt 10 := by sorry

end NUMINAMATH_CALUDE_complex_modulus_problem_l970_97055


namespace NUMINAMATH_CALUDE_abc_product_l970_97045

theorem abc_product (a b c : ℤ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : c ≠ 0)
  (h4 : a + b + c = 30) (h5 : 1 / a + 1 / b + 1 / c + 450 / (a * b * c) = 1) :
  a * b * c = 1912 := by
  sorry

end NUMINAMATH_CALUDE_abc_product_l970_97045


namespace NUMINAMATH_CALUDE_area_of_triangle_BCD_l970_97015

/-- Given a triangle ABC with area 36 and base 6, and a triangle BCD sharing the same height as ABC
    with base 34, prove that the area of triangle BCD is 204. -/
theorem area_of_triangle_BCD (area_ABC : ℝ) (base_AC : ℝ) (base_CD : ℝ) (height : ℝ) :
  area_ABC = 36 →
  base_AC = 6 →
  base_CD = 34 →
  area_ABC = (1/2) * base_AC * height →
  (1/2) * base_CD * height = 204 := by
  sorry

end NUMINAMATH_CALUDE_area_of_triangle_BCD_l970_97015


namespace NUMINAMATH_CALUDE_composition_equation_solution_l970_97030

theorem composition_equation_solution (α β : ℝ → ℝ) (h1 : ∀ x, α x = 4 * x + 9) 
  (h2 : ∀ x, β x = 9 * x + 6) (h3 : α (β x) = 8) : x = -25/36 := by
  sorry

end NUMINAMATH_CALUDE_composition_equation_solution_l970_97030


namespace NUMINAMATH_CALUDE_seventh_term_of_specific_geometric_sequence_l970_97017

/-- A geometric sequence is defined by its first term and common ratio -/
def geometric_sequence (a₁ : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a₁ * r ^ (n - 1)

/-- The seventh term of a geometric sequence with first term 3 and second term -3/2 is 3/64 -/
theorem seventh_term_of_specific_geometric_sequence :
  let a₁ : ℚ := 3
  let a₂ : ℚ := -3/2
  let r : ℚ := a₂ / a₁
  geometric_sequence a₁ r 7 = 3/64 := by
sorry

end NUMINAMATH_CALUDE_seventh_term_of_specific_geometric_sequence_l970_97017


namespace NUMINAMATH_CALUDE_best_of_five_more_advantageous_l970_97048

/-- The probability of the stronger player winning in a best-of-three format -/
def prob_best_of_three (p : ℝ) : ℝ := p^2 + 2*p^2*(1-p)

/-- The probability of the stronger player winning in a best-of-five format -/
def prob_best_of_five (p : ℝ) : ℝ := p^3 + 3*p^3*(1-p) + 6*p^3*(1-p)^2

/-- Theorem stating that the best-of-five format is more advantageous for selecting the strongest player -/
theorem best_of_five_more_advantageous (p : ℝ) (h : 0.5 < p ∧ p ≤ 1) :
  prob_best_of_three p < prob_best_of_five p :=
sorry

end NUMINAMATH_CALUDE_best_of_five_more_advantageous_l970_97048


namespace NUMINAMATH_CALUDE_real_roots_iff_k_leq_five_l970_97066

theorem real_roots_iff_k_leq_five (k : ℝ) :
  (∃ x : ℝ, (k - 3) * x^2 - 4 * x + 2 = 0) ↔ k ≤ 5 := by
  sorry

end NUMINAMATH_CALUDE_real_roots_iff_k_leq_five_l970_97066


namespace NUMINAMATH_CALUDE_mans_age_twice_sons_age_l970_97089

theorem mans_age_twice_sons_age (son_age : ℕ) (age_difference : ℕ) (years : ℕ) : 
  son_age = 16 →
  age_difference = 18 →
  (son_age + age_difference + years) = 2 * (son_age + years) →
  years = 2 := by
sorry

end NUMINAMATH_CALUDE_mans_age_twice_sons_age_l970_97089


namespace NUMINAMATH_CALUDE_incorrect_statement_l970_97080

theorem incorrect_statement :
  ¬(∀ (p q : Prop), (¬p ∧ ¬q) → ¬(p ∧ q)) :=
sorry

end NUMINAMATH_CALUDE_incorrect_statement_l970_97080


namespace NUMINAMATH_CALUDE_quadratic_equation_roots_l970_97067

theorem quadratic_equation_roots :
  let f : ℝ → ℝ := fun x ↦ 3 * x^2 - x
  ∃ x₁ x₂ : ℝ, x₁ = 0 ∧ x₂ = 1/3 ∧ (∀ x : ℝ, f x = 0 ↔ x = x₁ ∨ x = x₂) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_roots_l970_97067


namespace NUMINAMATH_CALUDE_absolute_value_inequality_l970_97029

-- Define the solution set
def solution_set : Set ℝ := {x | x < -5 ∨ x > 1}

-- State the theorem
theorem absolute_value_inequality :
  {x : ℝ | |x + 2| > 3} = solution_set :=
by sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_l970_97029


namespace NUMINAMATH_CALUDE_bmw_sales_l970_97041

/-- Proves that the number of BMWs sold is 135 given the specified conditions -/
theorem bmw_sales (total : ℕ) (audi_percent : ℚ) (toyota_percent : ℚ) (acura_percent : ℚ)
  (h_total : total = 300)
  (h_audi : audi_percent = 12 / 100)
  (h_toyota : toyota_percent = 25 / 100)
  (h_acura : acura_percent = 18 / 100)
  (h_sum : audi_percent + toyota_percent + acura_percent < 1) :
  ↑total * (1 - (audi_percent + toyota_percent + acura_percent)) = 135 := by
  sorry


end NUMINAMATH_CALUDE_bmw_sales_l970_97041


namespace NUMINAMATH_CALUDE_josh_money_left_l970_97020

/-- The amount of money Josh has left after selling bracelets and buying cookies -/
def money_left (cost_per_bracelet : ℚ) (sell_price : ℚ) (num_bracelets : ℕ) (cookie_cost : ℚ) : ℚ :=
  (sell_price - cost_per_bracelet) * num_bracelets - cookie_cost

/-- Theorem stating that Josh has $3 left after selling bracelets and buying cookies -/
theorem josh_money_left :
  money_left 1 1.5 12 3 = 3 := by
  sorry

end NUMINAMATH_CALUDE_josh_money_left_l970_97020


namespace NUMINAMATH_CALUDE_tangent_line_at_one_l970_97023

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x < 0 then Real.log (-x) + 3 * x else -Real.log x + 3 * x

-- State the theorem
theorem tangent_line_at_one (h : ∀ x, f (-x) = -f x) :
  let tangent_line (x : ℝ) := 2 * x + 1
  ∀ x, tangent_line x = f 1 + (tangent_line 1 - f 1) * (x - 1) := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_at_one_l970_97023


namespace NUMINAMATH_CALUDE_function_properties_l970_97050

-- Define the properties of the function f
def is_odd (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = -f x
def is_even (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x
def has_period (f : ℝ → ℝ) (p : ℝ) : Prop := ∀ x, f (x + p) = f x

-- State the theorem
theorem function_properties (f : ℝ → ℝ) (a b : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hab : a ≠ b) :
  (is_odd (fun x => f (x - a)) ∧ is_odd (fun x => f (x + a)) → has_period f (4 * a)) ∧
  (is_odd (fun x => f (x - a)) ∧ is_even (fun x => f (x - b)) → has_period f (4 * |a - b|)) :=
by sorry

end NUMINAMATH_CALUDE_function_properties_l970_97050


namespace NUMINAMATH_CALUDE_least_distinct_values_l970_97071

/-- Given a list of positive integers with the specified properties,
    the least number of distinct values is 218. -/
theorem least_distinct_values (list : List ℕ+) : 
  (list.length = 3042) →
  (∃! m, list.count m = 15 ∧ ∀ n, list.count n ≤ list.count m) →
  (list.toFinset.card ≥ 218 ∧ ∀ k, k < 218 → ¬(list.toFinset.card = k)) := by
  sorry

end NUMINAMATH_CALUDE_least_distinct_values_l970_97071


namespace NUMINAMATH_CALUDE_tangent_line_to_circle_l970_97051

/-- Given a point A(2,4) and a circle x^2 + y^2 = 4, 
    the tangent line from A to the circle has equation x = 2 or 3x - 4y + 10 = 0 -/
theorem tangent_line_to_circle (A : ℝ × ℝ) (circle : Set (ℝ × ℝ)) :
  A = (2, 4) →
  circle = {(x, y) | x^2 + y^2 = 4} →
  (∃ (k : ℝ), (∀ (x y : ℝ), (x, y) ∈ circle → 
    (x = 2 ∨ 3*x - 4*y + 10 = 0) ↔ 
    ((x - 2)^2 + (y - 4)^2 = ((x - 0)^2 + (y - 0)^2 - 4) / 4))) := by
  sorry


end NUMINAMATH_CALUDE_tangent_line_to_circle_l970_97051


namespace NUMINAMATH_CALUDE_sin_geq_x_on_unit_interval_l970_97049

theorem sin_geq_x_on_unit_interval (x : Real) (h : x ∈ Set.Icc 0 1) :
  Real.sqrt 2 * Real.sin x ≥ x := by
  sorry

end NUMINAMATH_CALUDE_sin_geq_x_on_unit_interval_l970_97049


namespace NUMINAMATH_CALUDE_x_equals_six_l970_97090

def factorial (n : ℕ) : ℕ := Nat.factorial n

theorem x_equals_six : ∃ x : ℕ, x * factorial x + 2 * factorial x = 40320 ∧ x = 6 := by
  sorry

end NUMINAMATH_CALUDE_x_equals_six_l970_97090


namespace NUMINAMATH_CALUDE_a_5_plus_a_6_equals_152_l970_97099

-- Define the sequence a_n
def a (n : ℕ) : ℤ := 3 * n^2 - 3 * n + 1

-- Define the partial sum S_n
def S (n : ℕ) : ℤ := n^3

-- State the theorem
theorem a_5_plus_a_6_equals_152 : a 5 + a 6 = 152 := by
  sorry

end NUMINAMATH_CALUDE_a_5_plus_a_6_equals_152_l970_97099


namespace NUMINAMATH_CALUDE_power_difference_divisibility_l970_97012

theorem power_difference_divisibility (a b : ℤ) (h : 100 ∣ (a - b)) :
  10000 ∣ (a^100 - b^100) := by
  sorry

end NUMINAMATH_CALUDE_power_difference_divisibility_l970_97012


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l970_97001

-- Define the quadratic inequality
def quadratic_inequality (x : ℝ) : Prop := (x - 2) * (x + 2) < 5

-- Define the solution set
def solution_set : Set ℝ := {x | -3 < x ∧ x < 3}

-- Theorem statement
theorem quadratic_inequality_solution :
  {x : ℝ | quadratic_inequality x} = solution_set :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l970_97001


namespace NUMINAMATH_CALUDE_television_selection_count_l970_97056

def type_a_count : ℕ := 4
def type_b_count : ℕ := 5
def selection_size : ℕ := 3

theorem television_selection_count :
  (type_a_count.choose 1) * (type_b_count.choose 1) * ((type_a_count + type_b_count - 2).choose 1) = 140 := by
  sorry

end NUMINAMATH_CALUDE_television_selection_count_l970_97056


namespace NUMINAMATH_CALUDE_game_converges_to_black_hole_l970_97025

/-- Represents a three-digit number in the game --/
structure GameNumber :=
  (hundreds : Nat)
  (tens : Nat)
  (ones : Nat)

/-- Counts the number of even digits in a natural number --/
def countEvenDigits (n : Nat) : Nat :=
  sorry

/-- Counts the number of odd digits in a natural number --/
def countOddDigits (n : Nat) : Nat :=
  sorry

/-- Counts the total number of digits in a natural number --/
def countDigits (n : Nat) : Nat :=
  sorry

/-- Converts a natural number to a GameNumber --/
def natToGameNumber (n : Nat) : GameNumber :=
  { hundreds := countEvenDigits n,
    tens := countOddDigits n,
    ones := countDigits n }

/-- Converts a GameNumber to a natural number --/
def gameNumberToNat (g : GameNumber) : Nat :=
  g.hundreds * 100 + g.tens * 10 + g.ones

/-- Applies one step of the game rules --/
def gameStep (n : Nat) : Nat :=
  gameNumberToNat (natToGameNumber n)

/-- The final number reached in the game --/
def blackHoleNumber : Nat := 123

/-- Theorem: The game always ends with the black hole number --/
theorem game_converges_to_black_hole (start : Nat) : 
  ∃ k : Nat, (gameStep^[k] start) = blackHoleNumber :=
sorry

end NUMINAMATH_CALUDE_game_converges_to_black_hole_l970_97025


namespace NUMINAMATH_CALUDE_coin_toss_problem_l970_97013

def binomial_probability (n k : ℕ) (p : ℝ) : ℝ :=
  (Nat.choose n k : ℝ) * p^k * (1 - p)^(n - k)

theorem coin_toss_problem (n : ℕ) :
  binomial_probability n 3 0.5 = 0.25 → n = 4 := by
  sorry

end NUMINAMATH_CALUDE_coin_toss_problem_l970_97013


namespace NUMINAMATH_CALUDE_fixed_distance_to_H_l970_97096

/-- Given a parabola y^2 = 4x, with O as the origin and moving points A and B on the parabola,
    such that OA ⊥ OB, and OH ⊥ AB where H is the foot of the perpendicular,
    prove that the point (2,0) has a fixed distance to point H. -/
theorem fixed_distance_to_H (O A B H : ℝ × ℝ) :
  O = (0, 0) →
  (∀ (x y : ℝ), A = (x, y) → y^2 = 4*x) →
  (∀ (x y : ℝ), B = (x, y) → y^2 = 4*x) →
  (A.1 * B.1 + A.2 * B.2 = 0) →  -- OA ⊥ OB
  (∃ (m n : ℝ), ∀ (x y : ℝ), (x = m*y + n) ↔ ((x, y) = A ∨ (x, y) = B)) →  -- Line AB: x = my + n
  (O.1 * H.1 + O.2 * H.2 = 0) →  -- OH ⊥ AB
  (∃ (d : ℝ), ∀ (H' : ℝ × ℝ), 
    (O.1 * H'.1 + O.2 * H'.2 = 0) →  -- OH' ⊥ AB
    (∃ (m n : ℝ), ∀ (x y : ℝ), (x = m*y + n) ↔ ((x, y) = A ∨ (x, y) = B)) →
    ((2 - H'.1)^2 + H'.2^2 = d^2)) :=
by sorry

end NUMINAMATH_CALUDE_fixed_distance_to_H_l970_97096


namespace NUMINAMATH_CALUDE_leaves_collection_time_l970_97098

/-- The time taken to collect leaves under given conditions -/
def collect_leaves_time (total_leaves : ℕ) (collect_rate : ℕ) (scatter_rate : ℕ) (cycle_time : ℚ) : ℚ :=
  let net_increase := collect_rate - scatter_rate
  let full_cycles := (total_leaves - net_increase) / net_increase
  (full_cycles * cycle_time + cycle_time) / 60

/-- The problem statement -/
theorem leaves_collection_time :
  collect_leaves_time 50 5 3 (45 / 60) = 75 / 4 :=
sorry

end NUMINAMATH_CALUDE_leaves_collection_time_l970_97098


namespace NUMINAMATH_CALUDE_max_perimeter_triangle_l970_97010

/-- Given a triangle with two sides of length 7 and 9 units, and the third side of length x units
    (where x is an integer), the maximum perimeter of the triangle is 31 units. -/
theorem max_perimeter_triangle (x : ℤ) : 
  (7 : ℝ) + 9 > x ∧ (7 : ℝ) + x > 9 ∧ (9 : ℝ) + x > 7 → 
  x > 0 →
  (∀ y : ℤ, ((7 : ℝ) + 9 > y ∧ (7 : ℝ) + y > 9 ∧ (9 : ℝ) + y > 7 → y ≤ x)) →
  (7 : ℝ) + 9 + x = 31 := by
  sorry

end NUMINAMATH_CALUDE_max_perimeter_triangle_l970_97010


namespace NUMINAMATH_CALUDE_solution_set_of_inequality_l970_97046

theorem solution_set_of_inequality (x : ℝ) :
  Set.Icc (-5 : ℝ) 3 \ {3} = {x | (x + 5) / (3 - x) ≥ 0} := by
  sorry

end NUMINAMATH_CALUDE_solution_set_of_inequality_l970_97046


namespace NUMINAMATH_CALUDE_z_in_fourth_quadrant_l970_97060

theorem z_in_fourth_quadrant (z : ℂ) (h : z * Complex.I = 2 + 3 * Complex.I) :
  Real.sign (z.re) = 1 ∧ Real.sign (z.im) = -1 :=
sorry

end NUMINAMATH_CALUDE_z_in_fourth_quadrant_l970_97060


namespace NUMINAMATH_CALUDE_ant_farm_problem_l970_97053

/-- Represents the number of ants of a specific species on a given day -/
def ant_count (initial : ℕ) (growth_rate : ℕ) (days : ℕ) : ℕ :=
  initial * (growth_rate ^ days)

theorem ant_farm_problem :
  ∀ a b c : ℕ,
  a + b + c = 50 →
  ant_count a 2 4 + ant_count b 3 4 + ant_count c 5 4 = 6230 →
  ant_count a 2 4 = 736 :=
by
  sorry

#check ant_farm_problem

end NUMINAMATH_CALUDE_ant_farm_problem_l970_97053


namespace NUMINAMATH_CALUDE_gratuity_percentage_l970_97022

theorem gratuity_percentage
  (num_people : ℕ)
  (total_bill : ℝ)
  (avg_cost_before_gratuity : ℝ)
  (h1 : num_people = 7)
  (h2 : total_bill = 840)
  (h3 : avg_cost_before_gratuity = 100) :
  (total_bill - num_people * avg_cost_before_gratuity) / (num_people * avg_cost_before_gratuity) * 100 = 20 := by
sorry

end NUMINAMATH_CALUDE_gratuity_percentage_l970_97022


namespace NUMINAMATH_CALUDE_max_value_f_l970_97077

open Real

/-- The maximum value of f(m, n) given the conditions -/
theorem max_value_f (f g : ℝ → ℝ) (m n : ℝ) :
  (∀ x > 0, f x = log x) →
  (∀ x, g x = (2*m + 3)*x + n) →
  (∀ x > 0, f x ≤ g x) →
  let f_mn := (2*m + 3) * n
  ∃ (min_f_mn : ℝ), f_mn ≥ min_f_mn ∧ 
    (∀ m' n', (∀ x > 0, log x ≤ (2*m' + 3)*x + n') → (2*m' + 3) * n' ≥ min_f_mn) →
  (∃ (max_value : ℝ), max_value = 1 / Real.exp 2 ∧
    ∀ m' n', (∀ x > 0, log x ≤ (2*m' + 3)*x + n') →
      let f_m'n' := (2*m' + 3) * n'
      ∃ (min_f_m'n' : ℝ), f_m'n' ≥ min_f_m'n' ∧
        (∀ m'' n'', (∀ x > 0, log x ≤ (2*m'' + 3)*x + n'') → (2*m'' + 3) * n'' ≥ min_f_m'n') →
      min_f_m'n' ≤ max_value) :=
by sorry

end NUMINAMATH_CALUDE_max_value_f_l970_97077


namespace NUMINAMATH_CALUDE_converse_is_false_l970_97052

theorem converse_is_false : ¬∀ x : ℝ, x > 0 → x - 3 > 0 := by sorry

end NUMINAMATH_CALUDE_converse_is_false_l970_97052


namespace NUMINAMATH_CALUDE_specific_regular_polygon_l970_97070

/-- Properties of a regular polygon -/
structure RegularPolygon where
  perimeter : ℝ
  side_length : ℝ
  sides : ℕ
  interior_angle : ℝ

/-- The theorem about the specific regular polygon -/
theorem specific_regular_polygon :
  ∃ (p : RegularPolygon),
    p.perimeter = 180 ∧
    p.side_length = 15 ∧
    p.sides = 12 ∧
    p.interior_angle = 150 := by
  sorry

end NUMINAMATH_CALUDE_specific_regular_polygon_l970_97070


namespace NUMINAMATH_CALUDE_distance_point_to_line_is_correct_l970_97002

def point_A : ℝ × ℝ × ℝ := (0, 3, -1)
def point_B : ℝ × ℝ × ℝ := (1, 2, 1)
def point_C : ℝ × ℝ × ℝ := (2, 4, 0)

def line_direction : ℝ × ℝ × ℝ := (point_C.1 - point_B.1, point_C.2.1 - point_B.2.1, point_C.2.2 - point_B.2.2)

def distance_point_to_line (A B C : ℝ × ℝ × ℝ) : ℝ := sorry

theorem distance_point_to_line_is_correct :
  distance_point_to_line point_A point_B point_C = (3 * Real.sqrt 2) / 2 := by sorry

end NUMINAMATH_CALUDE_distance_point_to_line_is_correct_l970_97002


namespace NUMINAMATH_CALUDE_carrie_farm_earnings_l970_97038

def total_money (num_tomatoes : ℕ) (num_carrots : ℕ) (price_tomato : ℚ) (price_carrot : ℚ) : ℚ :=
  num_tomatoes * price_tomato + num_carrots * price_carrot

theorem carrie_farm_earnings :
  total_money 200 350 1 (3/2) = 725 := by
  sorry

end NUMINAMATH_CALUDE_carrie_farm_earnings_l970_97038


namespace NUMINAMATH_CALUDE_prob_at_least_one_passes_eq_0_995_l970_97005

/-- The probability that at least one candidate passes the test -/
def prob_at_least_one_passes (prob_A prob_B prob_C : ℝ) : ℝ :=
  1 - (1 - prob_A) * (1 - prob_B) * (1 - prob_C)

/-- Theorem stating that the probability of at least one candidate passing is 0.995 -/
theorem prob_at_least_one_passes_eq_0_995 :
  prob_at_least_one_passes 0.9 0.8 0.75 = 0.995 := by
  sorry

#eval prob_at_least_one_passes 0.9 0.8 0.75

end NUMINAMATH_CALUDE_prob_at_least_one_passes_eq_0_995_l970_97005


namespace NUMINAMATH_CALUDE_gcf_154_252_l970_97018

theorem gcf_154_252 : Nat.gcd 154 252 = 14 := by
  sorry

end NUMINAMATH_CALUDE_gcf_154_252_l970_97018


namespace NUMINAMATH_CALUDE_tens_digit_of_1998_pow_2003_minus_1995_l970_97042

theorem tens_digit_of_1998_pow_2003_minus_1995 :
  (1998^2003 - 1995) % 100 / 10 = 0 := by
  sorry

end NUMINAMATH_CALUDE_tens_digit_of_1998_pow_2003_minus_1995_l970_97042


namespace NUMINAMATH_CALUDE_thermometer_distribution_count_l970_97004

/-- The number of senior classes -/
def num_classes : ℕ := 10

/-- The total number of thermometers to distribute -/
def total_thermometers : ℕ := 23

/-- The minimum number of thermometers each class must receive -/
def min_thermometers_per_class : ℕ := 2

/-- The number of remaining thermometers after initial distribution -/
def remaining_thermometers : ℕ := total_thermometers - num_classes * min_thermometers_per_class

/-- The number of spaces between items for divider placement -/
def spaces_for_dividers : ℕ := remaining_thermometers - 1

/-- The number of dividers needed -/
def num_dividers : ℕ := num_classes - 1

theorem thermometer_distribution_count :
  (spaces_for_dividers.choose num_dividers) = 220 := by
  sorry

end NUMINAMATH_CALUDE_thermometer_distribution_count_l970_97004


namespace NUMINAMATH_CALUDE_star_five_three_l970_97039

def star (a b : ℝ) : ℝ := 4 * a - 2 * b

theorem star_five_three : star 5 3 = 14 := by
  sorry

end NUMINAMATH_CALUDE_star_five_three_l970_97039


namespace NUMINAMATH_CALUDE_forty_people_skating_wheels_l970_97009

/-- The number of wheels on the floor when a given number of people are roller skating. -/
def wheels_on_floor (people : ℕ) : ℕ :=
  people * 2 * 4

/-- Theorem: When 40 people are roller skating, there are 320 wheels on the floor. -/
theorem forty_people_skating_wheels : wheels_on_floor 40 = 320 := by
  sorry

#eval wheels_on_floor 40

end NUMINAMATH_CALUDE_forty_people_skating_wheels_l970_97009


namespace NUMINAMATH_CALUDE_quadratic_equation_properties_l970_97082

theorem quadratic_equation_properties (m : ℝ) :
  (∀ m, ∃ x : ℝ, x^2 - (m-1)*x + (m-2) = 0) ∧
  (∃ x : ℝ, x^2 - (m-1)*x + (m-2) = 0 ∧ x > 6 → m > 8) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_properties_l970_97082


namespace NUMINAMATH_CALUDE_selection_count_l970_97031

def class_size : ℕ := 38
def selection_size : ℕ := 5
def remaining_students : ℕ := class_size - 2  -- Excluding students A and B
def remaining_selection_size : ℕ := selection_size - 1  -- We always select student A

def binomial (n k : ℕ) : ℕ := 
  if k > n then 0
  else Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

theorem selection_count : 
  binomial remaining_students remaining_selection_size = 58905 := by
  sorry

end NUMINAMATH_CALUDE_selection_count_l970_97031


namespace NUMINAMATH_CALUDE_mashed_potatoes_count_mashed_potatoes_proof_l970_97021

theorem mashed_potatoes_count : ℕ → ℕ → ℕ → Prop :=
  fun bacon_count difference mashed_count =>
    (bacon_count = 269) →
    (difference = 61) →
    (mashed_count = bacon_count + difference) →
    (mashed_count = 330)

-- The proof is omitted
theorem mashed_potatoes_proof : mashed_potatoes_count 269 61 330 := by
  sorry

end NUMINAMATH_CALUDE_mashed_potatoes_count_mashed_potatoes_proof_l970_97021


namespace NUMINAMATH_CALUDE_reciprocal_of_one_l970_97000

-- Define the concept of reciprocal
def is_reciprocal (a b : ℝ) : Prop := a * b = 1

-- Theorem statement
theorem reciprocal_of_one : is_reciprocal 1 1 := by
  sorry

end NUMINAMATH_CALUDE_reciprocal_of_one_l970_97000


namespace NUMINAMATH_CALUDE_exponent_simplification_l970_97007

theorem exponent_simplification (a : ℝ) : (36 * a^9)^4 * (63 * a^9)^4 = a^4 := by
  sorry

end NUMINAMATH_CALUDE_exponent_simplification_l970_97007


namespace NUMINAMATH_CALUDE_max_value_of_f_l970_97072

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^3 + 3*x^2 + a

-- State the theorem
theorem max_value_of_f (a : ℝ) :
  (∃ x ∈ Set.Icc (-3 : ℝ) 3, ∀ y ∈ Set.Icc (-3 : ℝ) 3, f a x ≤ f a y) ∧ 
  (∀ x ∈ Set.Icc (-3 : ℝ) 3, 3 ≤ f a x) →
  ∃ x ∈ Set.Icc (-3 : ℝ) 3, ∀ y ∈ Set.Icc (-3 : ℝ) 3, f a y ≤ f a x ∧ f a x = 57 :=
by
  sorry


end NUMINAMATH_CALUDE_max_value_of_f_l970_97072


namespace NUMINAMATH_CALUDE_product_of_five_reals_l970_97033

theorem product_of_five_reals (a b c d e : ℝ) 
  (h_distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ c ≠ d ∧ c ≠ e ∧ d ≠ e)
  (h1 : a * b + b = a * c + a)
  (h2 : b * c + c = b * d + b)
  (h3 : c * d + d = c * e + c)
  (h4 : d * e + e = d * a + d) :
  a * b * c * d * e = 1 := by
sorry

end NUMINAMATH_CALUDE_product_of_five_reals_l970_97033


namespace NUMINAMATH_CALUDE_f_properties_l970_97054

open Real

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := x * (x - 1) * (x - a)

-- Define the derivative f'(x)
def f_prime (a : ℝ) (x : ℝ) : ℝ := 3 * x^2 - 2 * (1 + a) * x + a

theorem f_properties (a : ℝ) (h : a > 1) :
  -- 1. The derivative of f(x) is f'(x)
  (∀ x, deriv (f a) x = f_prime a x) ∧
  -- 2. f(x) has two different critical points
  (∃ x₁ x₂, x₁ ≠ x₂ ∧ f_prime a x₁ = 0 ∧ f_prime a x₂ = 0) ∧
  -- 3. f(x₁) + f(x₂) ≤ 0 holds if and only if a ≥ 2
  (∀ x₁ x₂, f_prime a x₁ = 0 → f_prime a x₂ = 0 → 
    (f a x₁ + f a x₂ ≤ 0 ↔ a ≥ 2)) := by
  sorry

end NUMINAMATH_CALUDE_f_properties_l970_97054


namespace NUMINAMATH_CALUDE_brick_length_proof_l970_97044

/-- Given a courtyard and brick specifications, prove the length of each brick -/
theorem brick_length_proof (courtyard_length : ℝ) (courtyard_width : ℝ) 
  (brick_width : ℝ) (total_bricks : ℕ) :
  courtyard_length = 25 →
  courtyard_width = 18 →
  brick_width = 0.1 →
  total_bricks = 22500 →
  ∃ (brick_length : ℝ),
    brick_length = 0.2 ∧
    courtyard_length * courtyard_width * 10000 = 
      total_bricks * brick_length * brick_width := by
  sorry

end NUMINAMATH_CALUDE_brick_length_proof_l970_97044


namespace NUMINAMATH_CALUDE_lydia_apple_eating_age_l970_97047

/-- The age at which Lydia will first eat an apple from her tree -/
def apple_eating_age (tree_maturity : ℕ) (planting_age : ℕ) : ℕ :=
  planting_age + tree_maturity

theorem lydia_apple_eating_age
  (tree_maturity : ℕ)
  (planting_age : ℕ)
  (current_age : ℕ)
  (h1 : tree_maturity = 7)
  (h2 : planting_age = 4)
  (h3 : current_age = 9) :
  apple_eating_age tree_maturity planting_age = 11 := by
sorry

end NUMINAMATH_CALUDE_lydia_apple_eating_age_l970_97047


namespace NUMINAMATH_CALUDE_interior_edge_sum_is_ten_l970_97073

/-- Represents a rectangular picture frame -/
structure PictureFrame where
  width : ℝ
  outerLength : ℝ
  outerWidth : ℝ
  frameArea : ℝ

/-- Calculates the sum of interior edge lengths of a picture frame -/
def interiorEdgeSum (frame : PictureFrame) : ℝ :=
  2 * (frame.outerLength - 2 * frame.width) + 2 * (frame.outerWidth - 2 * frame.width)

/-- Theorem stating that for a frame with given properties, the sum of interior edges is 10 inches -/
theorem interior_edge_sum_is_ten (frame : PictureFrame) 
    (h1 : frame.width = 2)
    (h2 : frame.outerLength = 7)
    (h3 : frame.frameArea = 36)
    (h4 : frame.frameArea = frame.outerLength * frame.outerWidth - (frame.outerLength - 2 * frame.width) * (frame.outerWidth - 2 * frame.width)) :
  interiorEdgeSum frame = 10 := by
  sorry

#check interior_edge_sum_is_ten

end NUMINAMATH_CALUDE_interior_edge_sum_is_ten_l970_97073


namespace NUMINAMATH_CALUDE_expression_equality_l970_97086

theorem expression_equality (x : ℝ) (h : x ≥ 1) :
  let expr := Real.sqrt (x + 2 * Real.sqrt (x - 1)) + Real.sqrt (x - 2 * Real.sqrt (x - 1))
  (x ≤ 2 → expr = 2) ∧ (x > 2 → expr = 2 * Real.sqrt (x - 1)) := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l970_97086


namespace NUMINAMATH_CALUDE_best_optimistic_coefficient_l970_97026

theorem best_optimistic_coefficient 
  (a b c x : ℝ) 
  (h1 : a < b) 
  (h2 : 0 < x) 
  (h3 : x < 1) 
  (h4 : c = a + x * (b - a)) 
  (h5 : (c - a)^2 = (b - c) * (b - a)) : 
  x = (Real.sqrt 5 - 1) / 2 := by
sorry

end NUMINAMATH_CALUDE_best_optimistic_coefficient_l970_97026


namespace NUMINAMATH_CALUDE_equilateral_triangle_area_perimeter_ratio_l970_97035

/-- The ratio of area to perimeter for an equilateral triangle with side length 10 -/
theorem equilateral_triangle_area_perimeter_ratio :
  let s : ℝ := 10
  let perimeter : ℝ := 3 * s
  let area : ℝ := (Real.sqrt 3 / 4) * s^2
  area / perimeter = 5 * Real.sqrt 3 / 6 := by
  sorry

end NUMINAMATH_CALUDE_equilateral_triangle_area_perimeter_ratio_l970_97035


namespace NUMINAMATH_CALUDE_shooting_scenarios_l970_97097

theorem shooting_scenarios (n : ℕ) (n₁ n₂ n₃ n₄ : ℕ) 
  (h_total : n = n₁ + n₂ + n₃ + n₄)
  (h_n : n = 10)
  (h_n₁ : n₁ = 2)
  (h_n₂ : n₂ = 4)
  (h_n₃ : n₃ = 3)
  (h_n₄ : n₄ = 1) :
  (Nat.factorial n) / (Nat.factorial n₁ * Nat.factorial n₂ * Nat.factorial n₃ * Nat.factorial n₄) = 12600 :=
by sorry

end NUMINAMATH_CALUDE_shooting_scenarios_l970_97097


namespace NUMINAMATH_CALUDE_existence_of_subsets_l970_97006

/-- The set M containing integers from 1 to 10000 -/
def M : Set ℕ := Finset.range 10000

/-- The property that defines the required subsets -/
def has_unique_intersection (A : Finset (Set ℕ)) : Prop :=
  ∀ z ∈ M, ∃ B : Finset (Set ℕ), B ⊆ A ∧ B.card = 8 ∧ (⋂₀ B.toSet : Set ℕ) = {z}

/-- The main theorem stating the existence of 16 subsets with the required property -/
theorem existence_of_subsets : ∃ A : Finset (Set ℕ), A.card = 16 ∧ has_unique_intersection A := by
  sorry

end NUMINAMATH_CALUDE_existence_of_subsets_l970_97006


namespace NUMINAMATH_CALUDE_perfect_square_factors_count_l970_97061

def is_perfect_square (n : ℕ) : Prop := ∃ k : ℕ, n = k^2

def count_perfect_square_factors (a b c : ℕ) : ℕ :=
  (a/2 + 1) * ((b/2 + 1) * (c/2 + 1))

theorem perfect_square_factors_count :
  count_perfect_square_factors 10 12 15 = 336 := by sorry

end NUMINAMATH_CALUDE_perfect_square_factors_count_l970_97061


namespace NUMINAMATH_CALUDE_rectangular_field_dimensions_l970_97034

theorem rectangular_field_dimensions :
  ∀ (width length : ℝ),
  width > 0 →
  length = 2 * width →
  width * length = 800 →
  width = 20 ∧ length = 40 := by
sorry

end NUMINAMATH_CALUDE_rectangular_field_dimensions_l970_97034


namespace NUMINAMATH_CALUDE_system_solution_l970_97078

theorem system_solution : ∃ (x y : ℝ), 
  (2 * x^2 - 3 * x * y + y^2 = 3 ∧ x^2 + 2 * x * y - 2 * y^2 = 6) ∧
  ((x = 2 ∧ y = 1) ∨ (x = -2 ∧ y = -1)) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l970_97078


namespace NUMINAMATH_CALUDE_prove_nested_max_min_l970_97014

/-- Given distinct real numbers p, q, r, s, t satisfying p < q < r < s < t,
    prove that M(M(p, m(q, s)), m(r, m(p, t))) = q -/
theorem prove_nested_max_min (p q r s t : ℝ) 
  (h_distinct : p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ p ≠ t ∧ q ≠ r ∧ q ≠ s ∧ q ≠ t ∧ r ≠ s ∧ r ≠ t ∧ s ≠ t) 
  (h_order : p < q ∧ q < r ∧ r < s ∧ s < t) : 
  max (max p (min q s)) (min r (min p t)) = q := by
  sorry

end NUMINAMATH_CALUDE_prove_nested_max_min_l970_97014


namespace NUMINAMATH_CALUDE_rectangle_tileable_iff_divisible_l970_97079

/-- An (0, b)-tile is a 2 × b rectangle. -/
structure ZeroBTile (b : ℕ) :=
  (width : Fin 2)
  (height : Fin b)

/-- A tiling of an m × n rectangle with (0, b)-tiles. -/
def Tiling (m n b : ℕ) := List (ZeroBTile b)

/-- Predicate to check if a tiling is valid for an m × n rectangle. -/
def IsValidTiling (m n b : ℕ) (t : Tiling m n b) : Prop :=
  sorry  -- Definition of valid tiling omitted for brevity

/-- An m × n rectangle is (0, b)-tileable if there exists a valid tiling. -/
def IsTileable (m n b : ℕ) : Prop :=
  ∃ t : Tiling m n b, IsValidTiling m n b t

/-- Main theorem: An m × n rectangle is (0, b)-tileable iff 2b divides m or 2b divides n. -/
theorem rectangle_tileable_iff_divisible (m n b : ℕ) (hm : m > 0) (hn : n > 0) (hb : b > 0) :
  IsTileable m n b ↔ (2 * b ∣ m) ∨ (2 * b ∣ n) :=
sorry

end NUMINAMATH_CALUDE_rectangle_tileable_iff_divisible_l970_97079


namespace NUMINAMATH_CALUDE_inequality_proof_l970_97062

theorem inequality_proof (a b c : ℝ) : a^2 + 4*b^2 + 8*c^2 - 3*a*b - 4*b*c - 2*c*a ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l970_97062


namespace NUMINAMATH_CALUDE_box_volume_ratio_l970_97036

/-- The volume of a rectangular box -/
def box_volume (length width height : ℝ) : ℝ := length * width * height

/-- Alex's box dimensions -/
def alex_length : ℝ := 8
def alex_width : ℝ := 6
def alex_height : ℝ := 12

/-- Felicia's box dimensions -/
def felicia_length : ℝ := 12
def felicia_width : ℝ := 6
def felicia_height : ℝ := 8

theorem box_volume_ratio :
  (box_volume alex_length alex_width alex_height) / (box_volume felicia_length felicia_width felicia_height) = 1 := by
  sorry

end NUMINAMATH_CALUDE_box_volume_ratio_l970_97036


namespace NUMINAMATH_CALUDE_waiter_earnings_l970_97088

theorem waiter_earnings (total_customers : ℕ) (non_tippers : ℕ) (tip_amount : ℕ) : 
  total_customers = 9 → 
  non_tippers = 5 → 
  tip_amount = 8 → 
  (total_customers - non_tippers) * tip_amount = 32 := by
sorry

end NUMINAMATH_CALUDE_waiter_earnings_l970_97088


namespace NUMINAMATH_CALUDE_sum_of_roots_quadratic_sum_of_roots_specific_equation_l970_97059

theorem sum_of_roots_quadratic (a b c : ℝ) (h : a ≠ 0) :
  let equation := fun x : ℝ => a * x^2 + b * x + c
  let sum_of_roots := -b / a
  (∃ x y : ℝ, x ≠ y ∧ equation x = 0 ∧ equation y = 0) →
  sum_of_roots = x + y :=
sorry

theorem sum_of_roots_specific_equation :
  let equation := fun x : ℝ => x^2 + 2023 * x - 2024
  let sum_of_roots := -2023
  (∃ x y : ℝ, x ≠ y ∧ equation x = 0 ∧ equation y = 0) →
  sum_of_roots = x + y :=
sorry

end NUMINAMATH_CALUDE_sum_of_roots_quadratic_sum_of_roots_specific_equation_l970_97059


namespace NUMINAMATH_CALUDE_bookshelf_theorem_l970_97068

def bookshelf_problem (yoongi_notebooks jungkook_notebooks hoseok_notebooks : ℕ) : Prop :=
  yoongi_notebooks = 3 ∧ jungkook_notebooks = 3 ∧ hoseok_notebooks = 3 →
  yoongi_notebooks + jungkook_notebooks + hoseok_notebooks = 9

theorem bookshelf_theorem : bookshelf_problem 3 3 3 := by
  sorry

end NUMINAMATH_CALUDE_bookshelf_theorem_l970_97068


namespace NUMINAMATH_CALUDE_min_value_ab_min_value_ab_is_nine_l970_97074

theorem min_value_ab (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a + b - a*b + 3 = 0) :
  ∀ x y : ℝ, x > 0 → y > 0 → x + y - x*y + 3 = 0 → a*b ≤ x*y :=
by sorry

theorem min_value_ab_is_nine (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a + b - a*b + 3 = 0) :
  a*b = 9 :=
by sorry

end NUMINAMATH_CALUDE_min_value_ab_min_value_ab_is_nine_l970_97074


namespace NUMINAMATH_CALUDE_f_2_equals_17_l970_97032

/-- A function f with an extremum of 9 at x = 1 -/
def f (a b : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + b*x + a^2 - 1

/-- The function f has an extremum of 9 at x = 1 -/
def has_extremum_at_1 (a b : ℝ) : Prop :=
  f a b 1 = 9 ∧ ∃ ε > 0, ∀ x ∈ Set.Ioo (1 - ε) (1 + ε), f a b x ≤ f a b 1

theorem f_2_equals_17 (a b : ℝ) (h : has_extremum_at_1 a b) : f a b 2 = 17 := by
  sorry

end NUMINAMATH_CALUDE_f_2_equals_17_l970_97032


namespace NUMINAMATH_CALUDE_smallest_four_digit_divisible_by_53_l970_97065

theorem smallest_four_digit_divisible_by_53 : 
  ∀ n : ℕ, 1000 ≤ n ∧ n < 10000 ∧ n % 53 = 0 → n ≥ 1007 :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_four_digit_divisible_by_53_l970_97065


namespace NUMINAMATH_CALUDE_sum_of_15th_set_l970_97081

/-- The first element of the nth set in the sequence -/
def first_element (n : ℕ) : ℕ := 1 + n * (n - 1) / 2

/-- The last element of the nth set in the sequence -/
def last_element (n : ℕ) : ℕ := first_element n + n - 1

/-- The sum of elements in the nth set -/
def S (n : ℕ) : ℕ := n * (first_element n + last_element n) / 2

/-- Theorem stating that the sum of elements in the 15th set is 1695 -/
theorem sum_of_15th_set : S 15 = 1695 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_15th_set_l970_97081


namespace NUMINAMATH_CALUDE_max_nmmm_value_l970_97075

/-- Represents a three-digit number where all digits are the same -/
def three_digit_same (d : ℕ) : ℕ := 100 * d + 10 * d + d

/-- Represents a four-digit number NMMM where the last three digits are the same -/
def four_digit_nmmm (n m : ℕ) : ℕ := 1000 * n + 100 * m + 10 * m + m

/-- The maximum value of NMMM given the problem conditions -/
theorem max_nmmm_value :
  ∀ m : ℕ,
  1 ≤ m → m ≤ 9 →
  (∃ n : ℕ, four_digit_nmmm n m = m * three_digit_same m) →
  (∀ k : ℕ, k ≤ 9 → 
    (∃ l : ℕ, four_digit_nmmm l k = k * three_digit_same k) →
    four_digit_nmmm l k ≤ 3996) :=
by sorry

end NUMINAMATH_CALUDE_max_nmmm_value_l970_97075


namespace NUMINAMATH_CALUDE_joanna_estimate_l970_97076

theorem joanna_estimate (u v ε₁ ε₂ : ℝ) (h1 : u > v) (h2 : v > 0) (h3 : ε₁ > 0) (h4 : ε₂ > 0) :
  (u + ε₁) - (v - ε₂) > u - v := by
  sorry

end NUMINAMATH_CALUDE_joanna_estimate_l970_97076


namespace NUMINAMATH_CALUDE_overlapping_segment_length_l970_97057

theorem overlapping_segment_length (tape_length : ℝ) (total_length : ℝ) (num_tapes : ℕ) :
  tape_length = 250 →
  total_length = 925 →
  num_tapes = 4 →
  ∃ (overlap_length : ℝ),
    overlap_length * (num_tapes - 1) = num_tapes * tape_length - total_length ∧
    overlap_length = 25 := by
  sorry

end NUMINAMATH_CALUDE_overlapping_segment_length_l970_97057


namespace NUMINAMATH_CALUDE_dracula_is_alive_l970_97040

-- Define the possible states of a person
inductive PersonState
| Sane
| MadVampire
| Other

-- Define the Transylvanian's statement
def transylvanianStatement (personState : PersonState) (draculaAlive : Prop) : Prop :=
  (personState = PersonState.Sane ∨ personState = PersonState.MadVampire) → draculaAlive

-- Theorem to prove
theorem dracula_is_alive : ∃ (personState : PersonState), transylvanianStatement personState (∃ dracula, dracula = "alive") :=
sorry

end NUMINAMATH_CALUDE_dracula_is_alive_l970_97040


namespace NUMINAMATH_CALUDE_quadratic_roots_imply_negative_a_abs_function_solutions_l970_97063

-- Define the quadratic equation
def quadratic (a : ℝ) (x : ℝ) : ℝ := x^2 + (a - 3) * x + a

-- Define the absolute value function
def abs_function (x : ℝ) : ℝ := |3 - x^2|

theorem quadratic_roots_imply_negative_a (a : ℝ) :
  (∃ x y : ℝ, x > 0 ∧ y < 0 ∧ quadratic a x = 0 ∧ quadratic a y = 0) → a < 0 :=
sorry

theorem abs_function_solutions (a : ℝ) :
  ¬(∃! x : ℝ, abs_function x = a) :=
sorry

end NUMINAMATH_CALUDE_quadratic_roots_imply_negative_a_abs_function_solutions_l970_97063


namespace NUMINAMATH_CALUDE_gift_wrapping_combinations_l970_97091

/-- Represents the number of varieties of wrapping paper -/
def wrapping_paper_varieties : ℕ := 10

/-- Represents the number of colors of ribbon -/
def ribbon_colors : ℕ := 3

/-- Represents the number of types of gift cards -/
def gift_card_types : ℕ := 4

/-- Represents the number of designs of gift tags -/
def gift_tag_designs : ℕ := 5

/-- Calculates the total number of gift wrapping combinations -/
def total_combinations : ℕ := wrapping_paper_varieties * ribbon_colors * gift_card_types * gift_tag_designs

/-- Theorem stating that the total number of gift wrapping combinations is 600 -/
theorem gift_wrapping_combinations : total_combinations = 600 := by sorry

end NUMINAMATH_CALUDE_gift_wrapping_combinations_l970_97091


namespace NUMINAMATH_CALUDE_power_of_three_mod_seven_l970_97084

theorem power_of_three_mod_seven : 3^1503 % 7 = 6 := by
  sorry

end NUMINAMATH_CALUDE_power_of_three_mod_seven_l970_97084


namespace NUMINAMATH_CALUDE_toms_age_ratio_l970_97095

/-- Theorem representing Tom's age problem -/
theorem toms_age_ratio (T N : ℚ) : 
  (T > 0) →  -- Tom's age is positive
  (N > 0) →  -- N is positive (years in the past)
  (T - N > 0) →  -- Tom's age N years ago was positive
  (T - 4*N ≥ 0) →  -- Sum of children's ages N years ago was non-negative
  (T - N = 3*(T - 4*N)) →  -- Condition relating Tom's age N years ago to his children's ages
  T / N = 11 / 2 := by
  sorry

end NUMINAMATH_CALUDE_toms_age_ratio_l970_97095


namespace NUMINAMATH_CALUDE_conditional_probability_B_given_A_l970_97058

-- Define the sample space for a single die roll
def Die : Type := Fin 6

-- Define the probability space for two dice rolls
def TwoDice : Type := Die × Die

-- Define event A: sum of dice is even
def eventA (roll : TwoDice) : Prop :=
  (roll.1.val + 1 + roll.2.val + 1) % 2 = 0

-- Define event B: sum of dice is less than 7
def eventB (roll : TwoDice) : Prop :=
  roll.1.val + 1 + roll.2.val + 1 < 7

-- Define the probability measure
def P : Set TwoDice → ℝ := sorry

-- State the theorem
theorem conditional_probability_B_given_A :
  P {roll : TwoDice | eventB roll ∧ eventA roll} / P {roll : TwoDice | eventA roll} = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_conditional_probability_B_given_A_l970_97058


namespace NUMINAMATH_CALUDE_area_of_ω_l970_97028

-- Define the circle ω
def ω : Set (ℝ × ℝ) := sorry

-- Define points A and B
def A : ℝ × ℝ := (4, 15)
def B : ℝ × ℝ := (14, 9)

-- Assume A and B lie on ω
axiom A_on_ω : A ∈ ω
axiom B_on_ω : B ∈ ω

-- Define the tangent lines at A and B
def tangent_A : Set (ℝ × ℝ) := sorry
def tangent_B : Set (ℝ × ℝ) := sorry

-- Assume the intersection point of tangents is on the x-axis
axiom tangents_intersect_x_axis : ∃ x : ℝ, (x, 0) ∈ tangent_A ∩ tangent_B

-- Define the area of a circle
def circle_area (c : Set (ℝ × ℝ)) : ℝ := sorry

-- Theorem statement
theorem area_of_ω : 
  |circle_area ω - 154.73 * Real.pi| < 0.01 := sorry

end NUMINAMATH_CALUDE_area_of_ω_l970_97028


namespace NUMINAMATH_CALUDE_certain_number_is_sixteen_l970_97094

theorem certain_number_is_sixteen :
  ∃ x : ℝ, (213 * x = 3408) ∧ (21.3 * x = 340.8) → x = 16 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_is_sixteen_l970_97094


namespace NUMINAMATH_CALUDE_decreasing_function_a_range_l970_97092

-- Define the piecewise function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then (5 * a - 1) * x + 4 * a else Real.log x / Real.log a

-- Theorem statement
theorem decreasing_function_a_range :
  ∀ a : ℝ, (∀ x y : ℝ, x < y → f a x > f a y) →
  a ∈ Set.Icc (1 / 9 : ℝ) (1 / 5 : ℝ) := by
  sorry


end NUMINAMATH_CALUDE_decreasing_function_a_range_l970_97092


namespace NUMINAMATH_CALUDE_factorial_ratio_equals_sixty_sevenths_l970_97043

theorem factorial_ratio_equals_sixty_sevenths :
  (Nat.factorial 10 * Nat.factorial 6 * Nat.factorial 3) / (Nat.factorial 9 * Nat.factorial 7) = 60 / 7 := by
  sorry

end NUMINAMATH_CALUDE_factorial_ratio_equals_sixty_sevenths_l970_97043
