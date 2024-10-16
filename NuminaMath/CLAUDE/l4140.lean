import Mathlib

namespace NUMINAMATH_CALUDE_sqrt_inequality_l4140_414066

theorem sqrt_inequality : Real.sqrt 3 - Real.sqrt 2 > Real.sqrt 6 - Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_inequality_l4140_414066


namespace NUMINAMATH_CALUDE_height_difference_l4140_414091

theorem height_difference (parker daisy reese : ℕ) 
  (h1 : daisy = reese + 8)
  (h2 : reese = 60)
  (h3 : (parker + daisy + reese) / 3 = 64) :
  daisy - parker = 4 := by
sorry

end NUMINAMATH_CALUDE_height_difference_l4140_414091


namespace NUMINAMATH_CALUDE_exam_full_marks_l4140_414056

theorem exam_full_marks (A B C D F : ℝ) 
  (hA : A = 0.9 * B)
  (hB : B = 1.25 * C)
  (hC : C = 0.8 * D)
  (hAmarks : A = 360)
  (hDpercent : D = 0.8 * F) : 
  F = 500 := by
sorry

end NUMINAMATH_CALUDE_exam_full_marks_l4140_414056


namespace NUMINAMATH_CALUDE_star_two_three_l4140_414070

-- Define the star operation
def star (a b : ℝ) : ℝ := a^2 * b^2 - a + 1

-- Theorem statement
theorem star_two_three : star 2 3 = 35 := by
  sorry

end NUMINAMATH_CALUDE_star_two_three_l4140_414070


namespace NUMINAMATH_CALUDE_complement_of_M_l4140_414059

def U : Set ℕ := {1, 2, 3, 4, 5, 6}
def M : Set ℕ := {1, 2, 5}

theorem complement_of_M : 
  (U \ M) = {3, 4, 6} := by sorry

end NUMINAMATH_CALUDE_complement_of_M_l4140_414059


namespace NUMINAMATH_CALUDE_matthew_egg_rolls_l4140_414089

/-- The number of egg rolls eaten by each person -/
structure EggRolls where
  kimberly : ℕ
  alvin : ℕ
  patrick : ℕ
  matthew : ℕ

/-- The conditions of the egg roll problem -/
def EggRollConditions (e : EggRolls) : Prop :=
  e.kimberly = 5 ∧
  e.alvin = 2 * e.kimberly - 1 ∧
  e.patrick = e.alvin / 2 ∧
  e.matthew = 2 * e.patrick

theorem matthew_egg_rolls (e : EggRolls) (h : EggRollConditions e) : e.matthew = 8 := by
  sorry

#check matthew_egg_rolls

end NUMINAMATH_CALUDE_matthew_egg_rolls_l4140_414089


namespace NUMINAMATH_CALUDE_cubic_function_c_range_l4140_414026

theorem cubic_function_c_range (a b c : ℝ) :
  let f := fun x => x^3 + a*x^2 + b*x + c
  (0 < f (-1) ∧ f (-1) = f (-2) ∧ f (-2) = f (-3) ∧ f (-3) ≤ 3) →
  (6 < c ∧ c ≤ 9) := by
sorry

end NUMINAMATH_CALUDE_cubic_function_c_range_l4140_414026


namespace NUMINAMATH_CALUDE_trig_identity_l4140_414062

theorem trig_identity (a : ℝ) (h : Real.sin (π / 3 - a) = 1 / 3) :
  Real.cos (5 * π / 6 - a) = -1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l4140_414062


namespace NUMINAMATH_CALUDE_algae_free_day_l4140_414012

/-- Represents the coverage of algae on the pond for a given day -/
def algaeCoverage (day : ℕ) : ℚ :=
  1 / 2^(30 - day)

/-- The day when the pond is 75% algae-free -/
def targetDay : ℕ := 28

theorem algae_free_day :
  (algaeCoverage targetDay = 1/4) ∧ 
  (∀ d : ℕ, d < targetDay → algaeCoverage d < 1/4) ∧
  (∀ d : ℕ, d > targetDay → algaeCoverage d > 1/4) :=
by sorry

end NUMINAMATH_CALUDE_algae_free_day_l4140_414012


namespace NUMINAMATH_CALUDE_weight_problem_l4140_414024

/-- Proves that the initial number of students is 19 given the conditions of the weight problem. -/
theorem weight_problem (initial_avg : ℝ) (new_avg : ℝ) (new_student_weight : ℝ)
  (h1 : initial_avg = 15)
  (h2 : new_avg = 14.6)
  (h3 : new_student_weight = 7) :
  ∃ n : ℕ, n = 19 ∧ 
    (n : ℝ) * initial_avg + new_student_weight = (n + 1 : ℝ) * new_avg :=
by sorry

end NUMINAMATH_CALUDE_weight_problem_l4140_414024


namespace NUMINAMATH_CALUDE_abs_value_of_specific_complex_l4140_414000

/-- Given a complex number z = (1-i)/i, prove that its absolute value |z| is equal to √2 -/
theorem abs_value_of_specific_complex : let z : ℂ := (1 - Complex.I) / Complex.I
  Complex.abs z = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_abs_value_of_specific_complex_l4140_414000


namespace NUMINAMATH_CALUDE_raghu_investment_l4140_414013

theorem raghu_investment (raghu trishul vishal : ℝ) : 
  trishul = 0.9 * raghu →
  vishal = 1.1 * trishul →
  raghu + trishul + vishal = 6936 →
  raghu = 2400 := by
sorry

end NUMINAMATH_CALUDE_raghu_investment_l4140_414013


namespace NUMINAMATH_CALUDE_total_crickets_l4140_414088

/-- The total number of crickets given an initial and additional amount -/
theorem total_crickets (initial : ℝ) (additional : ℝ) :
  initial = 7.5 → additional = 11.25 → initial + additional = 18.75 :=
by sorry

end NUMINAMATH_CALUDE_total_crickets_l4140_414088


namespace NUMINAMATH_CALUDE_feuerbach_centers_parallelogram_or_collinear_l4140_414030

/-- A point in the plane -/
structure Point := (x y : ℝ)

/-- A quadrilateral defined by four points -/
structure Quadrilateral := (A B C D : Point)

/-- The intersection point of the diagonals of a quadrilateral -/
def diagonalIntersection (q : Quadrilateral) : Point := sorry

/-- The center of the Feuerbach circle of a triangle -/
def feuerbachCenter (A B C : Point) : Point := sorry

/-- Predicate to check if four points form a parallelogram -/
def isParallelogram (P Q R S : Point) : Prop := sorry

/-- Predicate to check if four points are collinear -/
def areCollinear (P Q R S : Point) : Prop := sorry

/-- Main theorem -/
theorem feuerbach_centers_parallelogram_or_collinear (q : Quadrilateral) :
  let E := diagonalIntersection q
  let F1 := feuerbachCenter q.A q.B E
  let F2 := feuerbachCenter q.B q.C E
  let F3 := feuerbachCenter q.C q.D E
  let F4 := feuerbachCenter q.D q.A E
  isParallelogram F1 F2 F3 F4 ∨ areCollinear F1 F2 F3 F4 := by
  sorry

end NUMINAMATH_CALUDE_feuerbach_centers_parallelogram_or_collinear_l4140_414030


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l4140_414010

theorem imaginary_part_of_z (z : ℂ) : z = 2 / (-1 + Complex.I) → z.im = -1 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l4140_414010


namespace NUMINAMATH_CALUDE_congruence_solution_l4140_414057

theorem congruence_solution (x a m : ℕ) : m ≥ 2 → a < m → (15 * x + 2) % 20 = 7 % 20 → x % m = a % m → a + m = 7 := by
  sorry

end NUMINAMATH_CALUDE_congruence_solution_l4140_414057


namespace NUMINAMATH_CALUDE_increasing_f_implies_k_leq_one_l4140_414077

-- Define the function f
def f (k : ℝ) (x : ℝ) : ℝ := x^2 - 2*k*x + 1

-- State the theorem
theorem increasing_f_implies_k_leq_one :
  ∀ k : ℝ, (∀ x y : ℝ, 1 ≤ x ∧ x < y ∧ y ≤ 3 → f k x < f k y) → k ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_increasing_f_implies_k_leq_one_l4140_414077


namespace NUMINAMATH_CALUDE_ratio_hcf_to_lcm_l4140_414099

/-- Given three positive integers a, b, and c in the ratio 3:4:5 with HCF 40, their LCM is 2400 -/
theorem ratio_hcf_to_lcm (a b c : ℕ+) : 
  (a : ℚ) / 3 = (b : ℚ) / 4 ∧ (b : ℚ) / 4 = (c : ℚ) / 5 → 
  Nat.gcd a.val (Nat.gcd b.val c.val) = 40 →
  Nat.lcm a.val (Nat.lcm b.val c.val) = 2400 := by
sorry

end NUMINAMATH_CALUDE_ratio_hcf_to_lcm_l4140_414099


namespace NUMINAMATH_CALUDE_expected_games_at_negative_one_l4140_414052

/-- The expected number of games in a best-of-five series -/
def f (x : ℝ) : ℝ :=
  3 * (x^3 + (1-x)^3) + 
  4 * (3*x^3*(1-x) + 3*(1-x)^3*x) + 
  5 * (6*x^2*(1-x)^2)

/-- Theorem: The expected number of games when x = -1 is 21 -/
theorem expected_games_at_negative_one : f (-1) = 21 := by
  sorry

end NUMINAMATH_CALUDE_expected_games_at_negative_one_l4140_414052


namespace NUMINAMATH_CALUDE_fiftieth_term_of_sequence_l4140_414079

/-- The nth term of an arithmetic sequence -/
def arithmeticSequenceTerm (a₁ : ℤ) (d : ℤ) (n : ℕ) : ℤ :=
  a₁ + (n - 1 : ℤ) * d

theorem fiftieth_term_of_sequence (a₁ d : ℤ) (h₁ : a₁ = 48) (h₂ : d = -2) :
  arithmeticSequenceTerm a₁ d 50 = -50 := by
  sorry

end NUMINAMATH_CALUDE_fiftieth_term_of_sequence_l4140_414079


namespace NUMINAMATH_CALUDE_quadratic_completing_square_l4140_414019

theorem quadratic_completing_square (x : ℝ) : 
  (∃ p q : ℝ, 16 * x^2 + 32 * x - 512 = 0 ↔ (x + p)^2 = q) → 
  (∃ q : ℝ, (∀ x : ℝ, 16 * x^2 + 32 * x - 512 = 0 ↔ (x + 1)^2 = q) ∧ q = 33) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_completing_square_l4140_414019


namespace NUMINAMATH_CALUDE_floor_sqrt_27_squared_l4140_414032

theorem floor_sqrt_27_squared : ⌊Real.sqrt 27⌋^2 = 25 := by
  sorry

end NUMINAMATH_CALUDE_floor_sqrt_27_squared_l4140_414032


namespace NUMINAMATH_CALUDE_figure_100_squares_l4140_414096

/-- The number of nonoverlapping unit squares in figure n -/
def f (n : ℕ) : ℕ := 2 * n^3 + n^2 + 2 * n + 1

theorem figure_100_squares :
  f 0 = 1 ∧ f 1 = 7 ∧ f 2 = 25 ∧ f 3 = 63 → f 100 = 2010201 := by
  sorry

end NUMINAMATH_CALUDE_figure_100_squares_l4140_414096


namespace NUMINAMATH_CALUDE_complex_purely_imaginary_l4140_414072

theorem complex_purely_imaginary (a : ℝ) :
  let z : ℂ := (a + Complex.I) / (1 - Complex.I)
  (∃ (b : ℝ), z = Complex.I * b) → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_purely_imaginary_l4140_414072


namespace NUMINAMATH_CALUDE_greatest_y_l4140_414084

theorem greatest_y (y : ℕ) (h1 : y > 0) (h2 : ∃ k : ℕ, y = 4 * k) (h3 : y^3 < 8000) :
  y ≤ 16 ∧ ∃ y' : ℕ, y' > 0 ∧ (∃ k : ℕ, y' = 4 * k) ∧ y'^3 < 8000 ∧ y' = 16 :=
sorry

end NUMINAMATH_CALUDE_greatest_y_l4140_414084


namespace NUMINAMATH_CALUDE_f_neither_even_nor_odd_l4140_414046

-- Define the function f on the given domain
def f : {x : ℝ | -1 < x ∧ x ≤ 1} → ℝ := fun x => x.val ^ 2

-- State the theorem
theorem f_neither_even_nor_odd :
  ¬(∀ x : {x : ℝ | -1 < x ∧ x ≤ 1}, f ⟨-x.val, by sorry⟩ = f x) ∧
  ¬(∀ x : {x : ℝ | -1 < x ∧ x ≤ 1}, f ⟨-x.val, by sorry⟩ = -f x) :=
by sorry

end NUMINAMATH_CALUDE_f_neither_even_nor_odd_l4140_414046


namespace NUMINAMATH_CALUDE_two_digit_average_decimal_l4140_414007

theorem two_digit_average_decimal (m n : ℕ) : 
  (10 ≤ m ∧ m < 100) →
  (10 ≤ n ∧ n < 100) →
  (m + n) / 2 = m + n / 10 →
  m = n :=
by sorry

end NUMINAMATH_CALUDE_two_digit_average_decimal_l4140_414007


namespace NUMINAMATH_CALUDE_remainder_of_1742_base12_div_9_l4140_414014

/-- Converts a base-12 digit to base-10 --/
def base12ToBase10(digit : Nat) : Nat :=
  if digit < 12 then digit else 0

/-- Converts a base-12 number to base-10 --/
def convertBase12ToBase10(n : List Nat) : Nat :=
  n.enum.foldl (fun acc (i, d) => acc + base12ToBase10 d * 12^i) 0

/-- The base-12 representation of 1742₁₂ --/
def base12Num : List Nat := [2, 4, 7, 1]

theorem remainder_of_1742_base12_div_9 :
  (convertBase12ToBase10 base12Num) % 9 = 3 := by
  sorry

end NUMINAMATH_CALUDE_remainder_of_1742_base12_div_9_l4140_414014


namespace NUMINAMATH_CALUDE_unique_quadratic_solution_l4140_414048

theorem unique_quadratic_solution (a : ℝ) :
  (∃! x : ℝ, a * x^2 + 2 * x - 1 = 0) → a = 0 ∨ a = -1 := by
  sorry

end NUMINAMATH_CALUDE_unique_quadratic_solution_l4140_414048


namespace NUMINAMATH_CALUDE_expression_evaluation_l4140_414051

theorem expression_evaluation : 
  let a : ℝ := 2 * Real.sin (π / 4) + (1 / 2)⁻¹
  ((a^2 - 4) / a) / ((4 * a - 4) / a - a) + 2 / (a - 2) = -1 - Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l4140_414051


namespace NUMINAMATH_CALUDE_smallest_result_l4140_414087

def S : Finset ℕ := {2, 5, 8, 11, 14}

def process (a b c : ℕ) : ℕ := (a + b) * c

theorem smallest_result :
  ∃ (a b c : ℕ), a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
  process a b c = 26 ∧
  ∀ (x y z : ℕ), x ∈ S → y ∈ S → z ∈ S → x ≠ y → y ≠ z → x ≠ z →
  process x y z ≥ 26 :=
by sorry

end NUMINAMATH_CALUDE_smallest_result_l4140_414087


namespace NUMINAMATH_CALUDE_shaded_circle_is_six_l4140_414016

/-- Represents the circle positions in the diagram --/
inductive Position
| Top
| Left
| Right
| Bottom
| Shaded
| Other

/-- Checks if a number is prime --/
def isPrime (n : Nat) : Bool :=
  n > 1 && (Nat.factors n).length == 1

/-- Represents the arrangement of numbers in the circles --/
def Arrangement := Position → Nat

/-- Checks if an arrangement is valid according to the problem conditions --/
def isValidArrangement (arr : Arrangement) : Prop :=
  arr Position.Top = 5 ∧
  ({6, 7, 8, 9, 10} : Set Nat) = {arr Position.Left, arr Position.Right, arr Position.Bottom, arr Position.Shaded, arr Position.Other} ∧
  (∀ p q : Position, p ≠ q → isPrime (arr p + arr q))

theorem shaded_circle_is_six (arr : Arrangement) (h : isValidArrangement arr) : 
  arr Position.Shaded = 6 := by
  sorry

#check shaded_circle_is_six

end NUMINAMATH_CALUDE_shaded_circle_is_six_l4140_414016


namespace NUMINAMATH_CALUDE_smallest_n_for_non_prime_2n_plus_1_l4140_414097

theorem smallest_n_for_non_prime_2n_plus_1 :
  ∃ n : ℕ+, (∀ k < n, Nat.Prime (2 * k + 1)) ∧ ¬Nat.Prime (2 * n + 1) ∧ n = 4 := by
  sorry

end NUMINAMATH_CALUDE_smallest_n_for_non_prime_2n_plus_1_l4140_414097


namespace NUMINAMATH_CALUDE_cube_edge_length_l4140_414050

/-- Given the cost of paint, coverage per quart, and total cost to paint a cube,
    prove that the edge length of the cube is 10 feet. -/
theorem cube_edge_length
  (paint_cost_per_quart : ℝ)
  (coverage_per_quart : ℝ)
  (total_cost : ℝ)
  (h1 : paint_cost_per_quart = 3.2)
  (h2 : coverage_per_quart = 60)
  (h3 : total_cost = 32)
  : ∃ (edge_length : ℝ), edge_length = 10 ∧ 6 * edge_length^2 = total_cost / paint_cost_per_quart * coverage_per_quart :=
by sorry

end NUMINAMATH_CALUDE_cube_edge_length_l4140_414050


namespace NUMINAMATH_CALUDE_quadratic_roots_difference_squared_l4140_414029

theorem quadratic_roots_difference_squared :
  ∀ a b : ℝ,
  (6 * a^2 + 13 * a - 28 = 0) →
  (6 * b^2 + 13 * b - 28 = 0) →
  (a - b)^2 = 841 / 36 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_difference_squared_l4140_414029


namespace NUMINAMATH_CALUDE_rectangle_width_and_ratio_l4140_414068

-- Define the rectangle
structure Rectangle where
  initial_length : ℝ
  new_length : ℝ
  new_perimeter : ℝ

-- Define the theorem
theorem rectangle_width_and_ratio 
  (rect : Rectangle) 
  (h1 : rect.initial_length = 8) 
  (h2 : rect.new_length = 12) 
  (h3 : rect.new_perimeter = 36) : 
  ∃ (new_width : ℝ), 
    new_width = 6 ∧ 
    new_width / rect.new_length = 1 / 2 := by
  sorry


end NUMINAMATH_CALUDE_rectangle_width_and_ratio_l4140_414068


namespace NUMINAMATH_CALUDE_olivia_napkins_l4140_414027

theorem olivia_napkins (initial_napkins final_napkins : ℕ) 
  (h1 : initial_napkins = 15)
  (h2 : final_napkins = 45)
  (h3 : ∃ (o : ℕ), final_napkins = initial_napkins + o + 2*o) :
  ∃ (o : ℕ), o = 10 ∧ final_napkins = initial_napkins + o + 2*o :=
by sorry

end NUMINAMATH_CALUDE_olivia_napkins_l4140_414027


namespace NUMINAMATH_CALUDE_min_value_of_expression_l4140_414075

theorem min_value_of_expression (x : ℝ) :
  (x + 1) * (x + 2) * (x + 3) * (x + 4) + 2019 ≥ 2018 ∧
  ∃ x : ℝ, (x + 1) * (x + 2) * (x + 3) * (x + 4) + 2019 = 2018 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l4140_414075


namespace NUMINAMATH_CALUDE_log_28_5_equals_fraction_l4140_414043

-- Define the common logarithm (base 10)
noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 10

-- Define the given conditions
variable (a b : ℝ)
variable (h1 : lg 2 = a)
variable (h2 : lg 7 = b)

-- Define the logarithm with base 28
noncomputable def log_28 (x : ℝ) : ℝ := Real.log x / Real.log 28

-- Theorem statement
theorem log_28_5_equals_fraction : log_28 5 = (1 - a) / (2 * a + b) := by
  sorry

end NUMINAMATH_CALUDE_log_28_5_equals_fraction_l4140_414043


namespace NUMINAMATH_CALUDE_tom_car_washing_earnings_l4140_414009

/-- The amount of money Tom had last week -/
def initial_amount : ℕ := 74

/-- The amount of money Tom has now -/
def current_amount : ℕ := 86

/-- The amount of money Tom made washing cars -/
def money_made : ℕ := current_amount - initial_amount

theorem tom_car_washing_earnings : 
  money_made = current_amount - initial_amount :=
by sorry

end NUMINAMATH_CALUDE_tom_car_washing_earnings_l4140_414009


namespace NUMINAMATH_CALUDE_sqrt_difference_equals_two_sqrt_three_l4140_414041

theorem sqrt_difference_equals_two_sqrt_three :
  Real.sqrt (7 + 4 * Real.sqrt 3) - Real.sqrt (7 - 4 * Real.sqrt 3) = 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_difference_equals_two_sqrt_three_l4140_414041


namespace NUMINAMATH_CALUDE_sqrt_six_over_sqrt_two_equals_sqrt_three_l4140_414001

theorem sqrt_six_over_sqrt_two_equals_sqrt_three : 
  Real.sqrt 6 / Real.sqrt 2 = Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_six_over_sqrt_two_equals_sqrt_three_l4140_414001


namespace NUMINAMATH_CALUDE_circle_center_l4140_414095

/-- The polar equation of a circle is given by ρ = √2(cos θ + sin θ).
    This theorem proves that the center of this circle is at the point (1, π/4) in polar coordinates. -/
theorem circle_center (ρ θ : ℝ) : 
  ρ = Real.sqrt 2 * (Real.cos θ + Real.sin θ) → 
  ∃ (r θ₀ : ℝ), r = 1 ∧ θ₀ = π / 4 ∧ 
    (∀ (x y : ℝ), x = r * Real.cos θ₀ ∧ y = r * Real.sin θ₀ → 
      (x - Real.sqrt 2 / 2)^2 + (y - Real.sqrt 2 / 2)^2 = 1) :=
by sorry

end NUMINAMATH_CALUDE_circle_center_l4140_414095


namespace NUMINAMATH_CALUDE_students_studying_both_subjects_l4140_414071

theorem students_studying_both_subjects (total : ℕ) 
  (physics_min physics_max chemistry_min chemistry_max : ℕ) : 
  total = 2500 →
  physics_min = 1750 →
  physics_max = 1875 →
  chemistry_min = 875 →
  chemistry_max = 1125 →
  ∃ (m M : ℕ),
    m = physics_min + chemistry_min - total ∧
    M = physics_max + chemistry_max - total ∧
    M - m = 375 :=
by sorry

end NUMINAMATH_CALUDE_students_studying_both_subjects_l4140_414071


namespace NUMINAMATH_CALUDE_price_reduction_equation_l4140_414067

/-- Proves the correct equation for a price reduction scenario -/
theorem price_reduction_equation (x : ℝ) : 
  (∃ (original_price final_price : ℝ),
    original_price = 200 ∧ 
    final_price = 162 ∧ 
    final_price = original_price * (1 - x)^2) ↔ 
  200 * (1 - x)^2 = 162 := by
sorry

end NUMINAMATH_CALUDE_price_reduction_equation_l4140_414067


namespace NUMINAMATH_CALUDE_not_square_for_any_base_l4140_414045

-- Define the representation of a number in base b
def base_b_representation (b : ℕ) : ℕ := b^2 + 3*b + 3

-- Theorem statement
theorem not_square_for_any_base :
  ∀ b : ℕ, b ≥ 2 → ¬ ∃ n : ℕ, base_b_representation b = n^2 :=
by sorry

end NUMINAMATH_CALUDE_not_square_for_any_base_l4140_414045


namespace NUMINAMATH_CALUDE_rain_on_monday_l4140_414086

theorem rain_on_monday (tuesday_rain : Real) (no_rain : Real) (both_rain : Real) 
  (h1 : tuesday_rain = 0.55)
  (h2 : no_rain = 0.35)
  (h3 : both_rain = 0.60) : 
  ∃ monday_rain : Real, monday_rain = 0.70 := by
  sorry

end NUMINAMATH_CALUDE_rain_on_monday_l4140_414086


namespace NUMINAMATH_CALUDE_dividend_calculation_l4140_414085

theorem dividend_calculation (divisor quotient remainder : ℕ) 
  (h1 : divisor = 21)
  (h2 : quotient = 14)
  (h3 : remainder = 7) :
  divisor * quotient + remainder = 301 :=
by sorry

end NUMINAMATH_CALUDE_dividend_calculation_l4140_414085


namespace NUMINAMATH_CALUDE_cone_height_from_circular_sector_l4140_414063

/-- The height of a cone formed by rolling one of four congruent sectors cut from a circular sheet of paper. -/
theorem cone_height_from_circular_sector (r : ℝ) (h : r = 10) :
  let sector_angle : ℝ := 2 * Real.pi / 4
  let base_radius : ℝ := r * sector_angle / (2 * Real.pi)
  let height : ℝ := Real.sqrt (r^2 - base_radius^2)
  height = (5 * Real.sqrt 15) / 2 := by
  sorry

end NUMINAMATH_CALUDE_cone_height_from_circular_sector_l4140_414063


namespace NUMINAMATH_CALUDE_binary_110011_equals_51_l4140_414083

def binary_to_decimal (b₅ b₄ b₃ b₂ b₁ b₀ : ℕ) : ℕ :=
  b₀ + 2 * b₁ + 2^2 * b₂ + 2^3 * b₃ + 2^4 * b₄ + 2^5 * b₅

theorem binary_110011_equals_51 : binary_to_decimal 1 1 0 0 1 1 = 51 := by
  sorry

end NUMINAMATH_CALUDE_binary_110011_equals_51_l4140_414083


namespace NUMINAMATH_CALUDE_work_completion_time_l4140_414049

/-- Proves that if person A can complete a work in 40 days, and together with person B they can complete 0.25 part of the work in 6 days, then person B can complete the work alone in 60 days. -/
theorem work_completion_time (a b : ℝ) (ha : a = 40) (hab : 1 / a + 1 / b = 1 / 24) : b = 60 := by
  sorry

end NUMINAMATH_CALUDE_work_completion_time_l4140_414049


namespace NUMINAMATH_CALUDE_gwen_book_count_l4140_414055

/-- The number of books on each shelf. -/
def books_per_shelf : ℕ := 4

/-- The number of shelves containing mystery books. -/
def mystery_shelves : ℕ := 5

/-- The number of shelves containing picture books. -/
def picture_shelves : ℕ := 3

/-- The total number of books Gwen has. -/
def total_books : ℕ := books_per_shelf * (mystery_shelves + picture_shelves)

theorem gwen_book_count : total_books = 32 := by
  sorry

end NUMINAMATH_CALUDE_gwen_book_count_l4140_414055


namespace NUMINAMATH_CALUDE_friendship_theorem_l4140_414002

/-- Represents a simple undirected graph with 6 vertices -/
def Graph := Fin 6 → Fin 6 → Bool

/-- The friendship relation is symmetric -/
def symmetric (g : Graph) : Prop :=
  ∀ i j : Fin 6, g i j = g j i

/-- A set of three vertices form a triangle in the graph -/
def isTriangle (g : Graph) (v1 v2 v3 : Fin 6) : Prop :=
  v1 ≠ v2 ∧ v2 ≠ v3 ∧ v1 ≠ v3 ∧
  ((g v1 v2 ∧ g v2 v3 ∧ g v1 v3) ∨ (¬g v1 v2 ∧ ¬g v2 v3 ∧ ¬g v1 v3))

/-- Main theorem: any graph with 6 vertices contains a monochromatic triangle -/
theorem friendship_theorem (g : Graph) (h : symmetric g) :
  ∃ v1 v2 v3 : Fin 6, isTriangle g v1 v2 v3 := by
  sorry

end NUMINAMATH_CALUDE_friendship_theorem_l4140_414002


namespace NUMINAMATH_CALUDE_mans_age_twice_sons_l4140_414025

/-- Proves that it takes 2 years for a man's age to be twice his son's age -/
theorem mans_age_twice_sons (mans_age son_age : ℕ) (y : ℕ) : 
  mans_age = son_age + 20 →
  son_age = 18 →
  mans_age + y = 2 * (son_age + y) →
  y = 2 := by
sorry

end NUMINAMATH_CALUDE_mans_age_twice_sons_l4140_414025


namespace NUMINAMATH_CALUDE_eight_million_two_hundred_thousand_scientific_notation_l4140_414047

/-- Scientific notation representation of a number -/
structure ScientificNotation where
  significand : ℝ
  exponent : ℤ
  is_valid : 1 ≤ |significand| ∧ |significand| < 10

/-- Convert a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem eight_million_two_hundred_thousand_scientific_notation :
  toScientificNotation 8200000 = ScientificNotation.mk 8.2 6 (by sorry) :=
sorry

end NUMINAMATH_CALUDE_eight_million_two_hundred_thousand_scientific_notation_l4140_414047


namespace NUMINAMATH_CALUDE_inequality_implication_l4140_414015

theorem inequality_implication (a b : ℝ) (h : a > b) : -5 * a < -5 * b := by
  sorry

end NUMINAMATH_CALUDE_inequality_implication_l4140_414015


namespace NUMINAMATH_CALUDE_count_monomials_l4140_414065

-- Define what a monomial is
def is_monomial (expr : String) : Bool :=
  match expr with
  | "0" => true
  | "2x-1" => false
  | "a" => true
  | "1/x" => false
  | "-2/3" => true
  | "(x-y)/2" => false
  | "2x/5" => true
  | _ => false

-- Define the set of expressions
def expressions : List String :=
  ["0", "2x-1", "a", "1/x", "-2/3", "(x-y)/2", "2x/5"]

-- Theorem statement
theorem count_monomials :
  (expressions.filter is_monomial).length = 4 := by sorry

end NUMINAMATH_CALUDE_count_monomials_l4140_414065


namespace NUMINAMATH_CALUDE_remainder_s_mod_6_l4140_414008

theorem remainder_s_mod_6 (s t : ℕ) (hs : s > t) (h_mod : (s - t) % 6 = 5) : s % 6 = 5 := by
  sorry

end NUMINAMATH_CALUDE_remainder_s_mod_6_l4140_414008


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l4140_414098

theorem imaginary_part_of_z (z : ℂ) (h : (z - Complex.I) / (z - 2) = Complex.I) :
  z.im = -1/2 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l4140_414098


namespace NUMINAMATH_CALUDE_trajectory_equation_l4140_414078

-- Define the circle O
def circle_O (x y : ℝ) : Prop := x^2 + y^2 = 4

-- Define point P
def point_P : ℝ × ℝ := (-1, 0)

-- Define a point Q on the circle
def point_Q (x y : ℝ) : Prop := circle_O x y

-- Define the midpoint M of PQ
def midpoint_M (x y : ℝ) : Prop :=
  ∃ (qx qy : ℝ), point_Q qx qy ∧ x = (qx + point_P.1) / 2 ∧ y = (qy + point_P.2) / 2

-- Theorem: The trajectory of M forms the equation (x + 1/2)² + y² = 1
theorem trajectory_equation :
  ∀ (x y : ℝ), midpoint_M x y ↔ (x + 1/2)^2 + y^2 = 1 :=
sorry

end NUMINAMATH_CALUDE_trajectory_equation_l4140_414078


namespace NUMINAMATH_CALUDE_divisibility_implies_difference_one_l4140_414054

theorem divisibility_implies_difference_one
  (a b c d : ℕ)
  (h1 : (a * b - c * d) ∣ a)
  (h2 : (a * b - c * d) ∣ b)
  (h3 : (a * b - c * d) ∣ c)
  (h4 : (a * b - c * d) ∣ d) :
  a * b - c * d = 1 := by
sorry

end NUMINAMATH_CALUDE_divisibility_implies_difference_one_l4140_414054


namespace NUMINAMATH_CALUDE_box_volume_l4140_414074

theorem box_volume (x y z : ℝ) 
  (h1 : 2*x + 2*y = 26) 
  (h2 : x + z = 10) 
  (h3 : y + z = 7) : 
  x * y * z = 80 := by
  sorry

end NUMINAMATH_CALUDE_box_volume_l4140_414074


namespace NUMINAMATH_CALUDE_simplify_expression_l4140_414040

theorem simplify_expression (x : ℝ) : 120 * x - 75 * x = 45 * x := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l4140_414040


namespace NUMINAMATH_CALUDE_thursday_is_only_valid_start_day_l4140_414006

-- Define the days of the week
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

def DayOfWeek.next (d : DayOfWeek) : DayOfWeek :=
  match d with
  | .Sunday => .Monday
  | .Monday => .Tuesday
  | .Tuesday => .Wednesday
  | .Wednesday => .Thursday
  | .Thursday => .Friday
  | .Friday => .Saturday
  | .Saturday => .Sunday

def DayOfWeek.addDays (d : DayOfWeek) (n : Nat) : DayOfWeek :=
  match n with
  | 0 => d
  | n + 1 => (d.addDays n).next

def isOpen (d : DayOfWeek) : Bool :=
  match d with
  | .Sunday => false
  | .Monday => false
  | _ => true

def validRedemptionSchedule (startDay : DayOfWeek) : Bool :=
  let schedule := List.range 8 |>.map (fun i => startDay.addDays (i * 7))
  schedule.all isOpen

theorem thursday_is_only_valid_start_day :
  ∀ (d : DayOfWeek), validRedemptionSchedule d ↔ d = DayOfWeek.Thursday :=
sorry

#check thursday_is_only_valid_start_day

end NUMINAMATH_CALUDE_thursday_is_only_valid_start_day_l4140_414006


namespace NUMINAMATH_CALUDE_simplify_expression_l4140_414034

theorem simplify_expression (x : ℝ) (h : x ≥ 2) :
  |2 - x| + (Real.sqrt (x - 2))^2 - Real.sqrt (4 * x^2 - 4 * x + 1) = -3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l4140_414034


namespace NUMINAMATH_CALUDE_sequence_is_arithmetic_l4140_414035

/-- Given a sequence {aₙ} satisfying 4aₙ₊₁ - 4aₙ - 9 = 0 for all n,
    prove that {aₙ} is an arithmetic sequence with a common difference of 9/4. -/
theorem sequence_is_arithmetic (a : ℕ → ℝ) 
    (h : ∀ n, 4 * a (n + 1) - 4 * a n - 9 = 0) :
    ∃ d, d = 9/4 ∧ ∀ n, a (n + 1) - a n = d :=
sorry

end NUMINAMATH_CALUDE_sequence_is_arithmetic_l4140_414035


namespace NUMINAMATH_CALUDE_derivative_f_l4140_414076

noncomputable def f (x : ℝ) : ℝ := (x + 1/x)^5

theorem derivative_f (x : ℝ) (hx : x ≠ 0) :
  deriv f x = 5 * (x + 1/x)^4 * (1 - 1/x^2) :=
by sorry

end NUMINAMATH_CALUDE_derivative_f_l4140_414076


namespace NUMINAMATH_CALUDE_complex_quadratic_equation_solution_l4140_414005

theorem complex_quadratic_equation_solution :
  ∃ (z₁ z₂ : ℂ), 
    (z₁ = 1 + Real.sqrt 3 - (Real.sqrt 3 / 2) * Complex.I) ∧
    (z₂ = 1 - Real.sqrt 3 + (Real.sqrt 3 / 2) * Complex.I) ∧
    (∀ z : ℂ, 3 * z^2 - 2 * z = 7 - 3 * Complex.I ↔ z = z₁ ∨ z = z₂) :=
by sorry

end NUMINAMATH_CALUDE_complex_quadratic_equation_solution_l4140_414005


namespace NUMINAMATH_CALUDE_speed_ratio_l4140_414044

/-- Represents the scenario of Xiaoqing and Xiaoqiang's journey --/
structure Journey where
  distance : ℝ
  walking_speed : ℝ
  motorcycle_speed : ℝ
  (walking_speed_pos : walking_speed > 0)
  (motorcycle_speed_pos : motorcycle_speed > 0)
  (distance_pos : distance > 0)

/-- The time taken for the entire journey is 2.5 times the direct trip --/
def journey_time_constraint (j : Journey) : Prop :=
  (j.distance / j.motorcycle_speed) * 2.5 = 
    (j.distance / j.motorcycle_speed) + 
    (j.distance / j.motorcycle_speed - j.distance / j.walking_speed)

/-- The theorem stating the ratio of speeds --/
theorem speed_ratio (j : Journey) 
  (h : journey_time_constraint j) : 
  j.motorcycle_speed / j.walking_speed = 3 := by
  sorry


end NUMINAMATH_CALUDE_speed_ratio_l4140_414044


namespace NUMINAMATH_CALUDE_abs_difference_of_product_and_sum_l4140_414018

theorem abs_difference_of_product_and_sum (p q : ℝ) 
  (h1 : p * q = 6) 
  (h2 : p + q = 7) : 
  |p - q| = Real.sqrt 37 := by
  sorry

end NUMINAMATH_CALUDE_abs_difference_of_product_and_sum_l4140_414018


namespace NUMINAMATH_CALUDE_sequences_sum_and_diff_total_l4140_414038

def arithmetic_sum (a₁ : ℤ) (d : ℤ) (n : ℕ) : ℤ :=
  n * (2 * a₁ + (n - 1) * d) / 2

def sequence1_sum : ℤ := arithmetic_sum 4 10 6
def sequence2_sum : ℤ := arithmetic_sum 12 10 6

theorem sequences_sum_and_diff_total : 
  (sequence1_sum + sequence2_sum) + (sequence2_sum - sequence1_sum) = 444 := by
  sorry

end NUMINAMATH_CALUDE_sequences_sum_and_diff_total_l4140_414038


namespace NUMINAMATH_CALUDE_sin_600_degrees_l4140_414053

theorem sin_600_degrees : Real.sin (600 * π / 180) = -Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_600_degrees_l4140_414053


namespace NUMINAMATH_CALUDE_sum_of_angles_equals_540_l4140_414036

-- Define the angles as real numbers
variable (a b c d e f g : ℝ)

-- Define the straight lines (we don't need to explicitly define them, 
-- but we'll use their properties in the theorem statement)

-- State the theorem
theorem sum_of_angles_equals_540 :
  a + b + c + d + e + f + g = 540 := by
  sorry


end NUMINAMATH_CALUDE_sum_of_angles_equals_540_l4140_414036


namespace NUMINAMATH_CALUDE_cosine_double_angle_equation_cosine_double_angle_special_case_l4140_414061

theorem cosine_double_angle_equation (a b c : ℝ) (x : ℝ) 
  (h : a * (Real.cos x)^2 + b * Real.cos x + c = 0) :
  (1/4) * a^2 * (Real.cos (2*x))^2 + 
  (1/2) * (a^2 - b^2 + 2*a*c) * Real.cos (2*x) + 
  (1/4) * (a^2 + 4*a*c + 4*c^2 - 2*b^2) = 0 := by
  sorry

-- Special case
theorem cosine_double_angle_special_case (x : ℝ) 
  (h : 4 * (Real.cos x)^2 + 2 * Real.cos x - 1 = 0) :
  4 * (Real.cos (2*x))^2 + 2 * Real.cos (2*x) - 1 = 0 := by
  sorry

end NUMINAMATH_CALUDE_cosine_double_angle_equation_cosine_double_angle_special_case_l4140_414061


namespace NUMINAMATH_CALUDE_total_money_problem_l4140_414090

theorem total_money_problem (brad : ℝ) (josh : ℝ) (doug : ℝ) 
  (h1 : brad = 12.000000000000002)
  (h2 : josh = 2 * brad)
  (h3 : josh = (3/4) * doug) : 
  brad + josh + doug = 68.00000000000001 := by
  sorry

end NUMINAMATH_CALUDE_total_money_problem_l4140_414090


namespace NUMINAMATH_CALUDE_range_of_dot_product_trajectory_of_P_l4140_414011

noncomputable section

-- Define the hyperbola C
def C (x y : ℝ) : Prop := x^2 / 2 - y^2 / 3 = 1

-- Define the foci F₁ and F₂
def F₁ : ℝ × ℝ := (-Real.sqrt 5, 0)
def F₂ : ℝ × ℝ := (Real.sqrt 5, 0)

-- Define a point M on the right branch of C
def M (x y : ℝ) : Prop := C x y ∧ x ≥ Real.sqrt 2

-- Define the dot product of vectors
def dot_product (v₁ v₂ : ℝ × ℝ) : ℝ :=
  v₁.1 * v₂.1 + v₁.2 * v₂.2

-- Theorem 1: Range of OM · F₁M
theorem range_of_dot_product (x y : ℝ) :
  M x y → dot_product (x, y) (x + F₁.1, y + F₁.2) ≥ 2 + Real.sqrt 10 := by sorry

-- Define a point P with constant sum of distances from F₁ and F₂
def P (x y : ℝ) : Prop :=
  ∃ (k : ℝ), Real.sqrt ((x - F₁.1)^2 + (y - F₁.2)^2) +
             Real.sqrt ((x - F₂.1)^2 + (y - F₂.2)^2) = k

-- Define the cosine of angle F₁PF₂
def cos_F₁PF₂ (x y : ℝ) : ℝ :=
  let d₁ := Real.sqrt ((x - F₁.1)^2 + (y - F₁.2)^2)
  let d₂ := Real.sqrt ((x - F₂.1)^2 + (y - F₂.2)^2)
  ((x - F₁.1) * (x - F₂.1) + (y - F₁.2) * (y - F₂.2)) / (d₁ * d₂)

-- Theorem 2: Trajectory of P
theorem trajectory_of_P (x y : ℝ) :
  P x y ∧ (∀ (u v : ℝ), P u v → cos_F₁PF₂ x y ≤ cos_F₁PF₂ u v) ∧ cos_F₁PF₂ x y = -1/9
  → x^2/9 + y^2/4 = 1 := by sorry

end NUMINAMATH_CALUDE_range_of_dot_product_trajectory_of_P_l4140_414011


namespace NUMINAMATH_CALUDE_power_inequality_l4140_414042

theorem power_inequality : 3^44 > 4^33 ∧ 4^33 > 5^22 := by
  sorry

end NUMINAMATH_CALUDE_power_inequality_l4140_414042


namespace NUMINAMATH_CALUDE_problem_1_problem_2_l4140_414023

theorem problem_1 : 3 * Real.sqrt 3 - Real.sqrt 8 + Real.sqrt 2 - Real.sqrt 27 = -Real.sqrt 2 := by sorry

theorem problem_2 : (4 * Real.sqrt 6 - 6 * Real.sqrt 3) / (2 * Real.sqrt 3) = 2 * Real.sqrt 2 - 3 := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_l4140_414023


namespace NUMINAMATH_CALUDE_gear_alignment_theorem_l4140_414093

/-- Represents a gear with a certain number of teeth and ground-off pairs -/
structure Gear where
  initial_teeth : Nat
  ground_off_pairs : Nat

/-- Calculates the number of remaining teeth on a gear -/
def remaining_teeth (g : Gear) : Nat :=
  g.initial_teeth - g.ground_off_pairs

/-- Calculates the number of possible alignment positions -/
def alignment_positions (g : Gear) : Nat :=
  g.initial_teeth - g.ground_off_pairs + 1

/-- Theorem stating that there exists exactly one position where a hole in one gear
    aligns with a whole tooth on the other gear -/
theorem gear_alignment_theorem (g1 g2 : Gear)
  (h1 : g1.initial_teeth = 32)
  (h2 : g2.initial_teeth = 32)
  (h3 : g1.ground_off_pairs = 6)
  (h4 : g2.ground_off_pairs = 6)
  : ∃! position, position ≤ alignment_positions g1 ∧
    (position ≠ 0 → 
      (∃ hole_in_g1 whole_tooth_in_g2, 
        hole_in_g1 ≤ g1.ground_off_pairs ∧
        whole_tooth_in_g2 ≤ remaining_teeth g2 ∧
        hole_in_g1 ≠ whole_tooth_in_g2)) :=
  sorry

end NUMINAMATH_CALUDE_gear_alignment_theorem_l4140_414093


namespace NUMINAMATH_CALUDE_pizza_slices_l4140_414058

theorem pizza_slices (coworkers : ℕ) (pizzas : ℕ) (slices_per_person : ℕ) :
  coworkers = 12 →
  pizzas = 3 →
  slices_per_person = 2 →
  (coworkers * slices_per_person) / pizzas = 8 :=
by
  sorry

end NUMINAMATH_CALUDE_pizza_slices_l4140_414058


namespace NUMINAMATH_CALUDE_right_triangle_sets_l4140_414073

/-- A function that checks if three numbers can form a right-angled triangle -/
def is_right_triangle (a b c : ℕ) : Prop :=
  a * a + b * b = c * c ∨ a * a + c * c = b * b ∨ b * b + c * c = a * a

/-- The theorem stating that among the given sets, only (5, 12, 13) forms a right-angled triangle -/
theorem right_triangle_sets : 
  is_right_triangle 5 12 13 ∧
  ¬is_right_triangle 2 3 4 ∧
  ¬is_right_triangle 4 5 6 ∧
  ¬is_right_triangle 3 4 6 :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_sets_l4140_414073


namespace NUMINAMATH_CALUDE_largest_n_divisibility_l4140_414003

theorem largest_n_divisibility : ∃ (n : ℕ), n = 890 ∧ 
  (∀ m : ℕ, m > n → ¬(m + 10 ∣ m^3 + 100)) ∧ 
  (n + 10 ∣ n^3 + 100) := by
  sorry

end NUMINAMATH_CALUDE_largest_n_divisibility_l4140_414003


namespace NUMINAMATH_CALUDE_quadratic_coefficients_l4140_414080

def is_vertex (f : ℝ → ℝ) (x₀ y₀ : ℝ) : Prop :=
  ∀ x, f x ≥ f x₀ ∧ f x₀ = y₀

def has_vertical_symmetry_axis (f : ℝ → ℝ) (x₀ : ℝ) : Prop :=
  ∀ x, f (x₀ + x) = f (x₀ - x)

theorem quadratic_coefficients 
  (f : ℝ → ℝ) (a b c : ℝ) :
  (∀ x, f x = a * x^2 + b * x + c) →
  is_vertex f (-2) 5 →
  has_vertical_symmetry_axis f (-2) →
  f 0 = 9 →
  a = 1 ∧ b = 4 ∧ c = 9 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_coefficients_l4140_414080


namespace NUMINAMATH_CALUDE_lake_width_correct_l4140_414031

/-- The width of the lake in miles -/
def lake_width : ℝ := 60

/-- The speed of the faster boat in miles per hour -/
def fast_boat_speed : ℝ := 30

/-- The speed of the slower boat in miles per hour -/
def slow_boat_speed : ℝ := 12

/-- The time difference in hours between the arrivals of the two boats -/
def time_difference : ℝ := 3

/-- Theorem stating that the lake width is correct given the boat speeds and time difference -/
theorem lake_width_correct :
  lake_width / slow_boat_speed = lake_width / fast_boat_speed + time_difference :=
by sorry

end NUMINAMATH_CALUDE_lake_width_correct_l4140_414031


namespace NUMINAMATH_CALUDE_tarantula_perimeter_is_16_l4140_414022

/-- Represents a rectangle with width and height in inches -/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Represents the tarantula-shaped figure -/
structure TarantulaShape where
  body : Rectangle
  legs : Rectangle

/-- Calculates the perimeter of the tarantula-shaped figure -/
def tarantulaPerimeter (t : TarantulaShape) : ℝ :=
  2 * (t.body.width + t.body.height)

theorem tarantula_perimeter_is_16 :
  ∀ t : TarantulaShape,
    t.body.width = 3 ∧
    t.body.height = 10 ∧
    t.legs.width = 5 ∧
    t.legs.height = 3 →
    tarantulaPerimeter t = 16 := by
  sorry

#check tarantula_perimeter_is_16

end NUMINAMATH_CALUDE_tarantula_perimeter_is_16_l4140_414022


namespace NUMINAMATH_CALUDE_arithmetic_sequence_max_ratio_l4140_414082

theorem arithmetic_sequence_max_ratio (a : ℕ → ℝ) (S : ℕ → ℝ) : 
  (∀ n, S n = n * (a 1 + a n) / 2) →  -- Definition of S_n for arithmetic sequence
  (∀ n, a (n + 1) - a n = a 2 - a 1) →  -- Definition of arithmetic sequence
  S 17 > 0 →
  S 18 < 0 →
  (∀ k ∈ Finset.range 15, S (k + 1) / a (k + 1) ≤ S 9 / a 9) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_max_ratio_l4140_414082


namespace NUMINAMATH_CALUDE_lily_cups_count_l4140_414017

/-- Represents Gina's cup painting rates and order details -/
structure PaintingOrder where
  rose_rate : ℕ  -- Roses painted per hour
  lily_rate : ℕ  -- Lilies painted per hour
  rose_order : ℕ  -- Number of rose cups ordered
  total_pay : ℕ  -- Total payment for the order in dollars
  hourly_rate : ℕ  -- Gina's hourly rate in dollars

/-- Calculates the number of lily cups in the order -/
def lily_cups (order : PaintingOrder) : ℕ :=
  let total_hours := order.total_pay / order.hourly_rate
  let rose_hours := order.rose_order / order.rose_rate
  let lily_hours := total_hours - rose_hours
  lily_hours * order.lily_rate

/-- Theorem stating that for the given order, the number of lily cups is 14 -/
theorem lily_cups_count (order : PaintingOrder) 
  (h1 : order.rose_rate = 6)
  (h2 : order.lily_rate = 7)
  (h3 : order.rose_order = 6)
  (h4 : order.total_pay = 90)
  (h5 : order.hourly_rate = 30) :
  lily_cups order = 14 := by
  sorry

#eval lily_cups { rose_rate := 6, lily_rate := 7, rose_order := 6, total_pay := 90, hourly_rate := 30 }

end NUMINAMATH_CALUDE_lily_cups_count_l4140_414017


namespace NUMINAMATH_CALUDE_floor_plus_self_eq_seventeen_point_five_l4140_414094

theorem floor_plus_self_eq_seventeen_point_five (s : ℝ) : 
  ⌊s⌋ + s = 17.5 ↔ s = 8.5 := by sorry

end NUMINAMATH_CALUDE_floor_plus_self_eq_seventeen_point_five_l4140_414094


namespace NUMINAMATH_CALUDE_smallest_valid_number_l4140_414064

def is_odd (n : ℕ) : Bool := n % 2 = 1

def is_even (n : ℕ) : Bool := n % 2 = 0

def digit_count (n : ℕ) : ℕ := (String.length (toString n))

def sum_of_digits (n : ℕ) : ℕ :=
  (toString n).toList.map (fun c => c.toNat - '0'.toNat) |>.sum

def is_valid_number (n : ℕ) : Bool :=
  digit_count n = 4 ∧
  n % 9 = 0 ∧
  (is_odd (n / 1000 % 10) + is_odd (n / 100 % 10) + is_odd (n / 10 % 10) + is_odd (n % 10) = 3) ∧
  (is_even (n / 1000 % 10) + is_even (n / 100 % 10) + is_even (n / 10 % 10) + is_even (n % 10) = 1)

theorem smallest_valid_number : 
  (∀ m : ℕ, 1000 ≤ m ∧ m < 1215 → ¬ is_valid_number m) ∧ is_valid_number 1215 := by sorry

end NUMINAMATH_CALUDE_smallest_valid_number_l4140_414064


namespace NUMINAMATH_CALUDE_x_2007_equals_2_l4140_414092

def x : ℕ → ℚ
  | 0 => 1
  | 1 => 1
  | (n + 2) => (1 + x (n + 1)) / x n

theorem x_2007_equals_2 : x 2007 = 2 := by
  sorry

end NUMINAMATH_CALUDE_x_2007_equals_2_l4140_414092


namespace NUMINAMATH_CALUDE_D_2021_2022_2023_odd_l4140_414060

def D : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | 2 => 1
  | n + 3 => D (n + 2) + D (n + 1)

theorem D_2021_2022_2023_odd :
  Odd (D 2021) ∧ Odd (D 2022) ∧ Odd (D 2023) := by
  sorry

end NUMINAMATH_CALUDE_D_2021_2022_2023_odd_l4140_414060


namespace NUMINAMATH_CALUDE_range_of_f_l4140_414069

-- Define the function f(x) = x^3
def f (x : ℝ) : ℝ := x^3

-- Define the domain
def domain : Set ℝ := {x : ℝ | x ≥ -1}

-- Define the range
def range : Set ℝ := {y : ℝ | ∃ x ∈ domain, f x = y}

-- Theorem statement
theorem range_of_f : range = {y : ℝ | y ≥ -1} := by sorry

end NUMINAMATH_CALUDE_range_of_f_l4140_414069


namespace NUMINAMATH_CALUDE_triangle_area_l4140_414039

/-- Given a triangle ABC with side length a = 6, angle B = 30°, and angle C = 120°,
    prove that its area is 9√3. -/
theorem triangle_area (a b c : ℝ) (A B C : Real) : 
  a = 6 → B = 30 * Real.pi / 180 → C = 120 * Real.pi / 180 →
  (1/2) * a * b * Real.sin C = 9 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l4140_414039


namespace NUMINAMATH_CALUDE_shaded_area_square_with_circles_l4140_414020

/-- The area of the shaded region not covered by four circles centered at the vertices of a square -/
theorem shaded_area_square_with_circles (side_length radius : ℝ) (h1 : side_length = 8) (h2 : radius = 3) :
  side_length ^ 2 - 4 * Real.pi * radius ^ 2 = 64 - 36 * Real.pi := by
  sorry

#check shaded_area_square_with_circles

end NUMINAMATH_CALUDE_shaded_area_square_with_circles_l4140_414020


namespace NUMINAMATH_CALUDE_y_intercept_of_line_l4140_414028

/-- The y-intercept of a line with slope 1 passing through the midpoint of a line segment --/
theorem y_intercept_of_line (x₁ y₁ x₂ y₂ : ℝ) :
  let midpoint_x := (x₁ + x₂) / 2
  let midpoint_y := (y₁ + y₂) / 2
  let slope := 1
  let y_intercept := midpoint_y - slope * midpoint_x
  x₁ = 2 ∧ y₁ = 8 ∧ x₂ = 14 ∧ y₂ = 4 →
  y_intercept = -2 := by
sorry

end NUMINAMATH_CALUDE_y_intercept_of_line_l4140_414028


namespace NUMINAMATH_CALUDE_susans_books_l4140_414037

/-- Proves that Susan has 600 books given the conditions of the problem -/
theorem susans_books (susan_books : ℕ) (lidia_books : ℕ) : 
  lidia_books = 4 * susan_books → -- Lidia's collection is four times bigger than Susan's
  susan_books + lidia_books = 3000 → -- Total books is 3000
  susan_books = 600 := by
sorry

end NUMINAMATH_CALUDE_susans_books_l4140_414037


namespace NUMINAMATH_CALUDE_unique_solution_l4140_414081

/-- The number of positive integer solutions to the equation 2x + 3y = 8 -/
def solution_count : ℕ :=
  (Finset.filter (fun p : ℕ × ℕ => 
    p.1 > 0 ∧ p.2 > 0 ∧ 2 * p.1 + 3 * p.2 = 8) (Finset.product (Finset.range 9) (Finset.range 9))).card

/-- Theorem stating that there is exactly one positive integer solution to 2x + 3y = 8 -/
theorem unique_solution : solution_count = 1 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_l4140_414081


namespace NUMINAMATH_CALUDE_stratified_sampling_second_year_l4140_414021

theorem stratified_sampling_second_year (total_students : ℕ) (second_year_students : ℕ) (sample_size : ℕ) :
  total_students = 3600 →
  second_year_students = 900 →
  sample_size = 720 →
  (second_year_students * sample_size) / total_students = 180 :=
by sorry

end NUMINAMATH_CALUDE_stratified_sampling_second_year_l4140_414021


namespace NUMINAMATH_CALUDE_limit_evaluation_l4140_414033

theorem limit_evaluation : 
  ∀ ε > 0, ∃ N : ℕ, ∀ n ≥ N, 
    |((1 + 2*n : ℝ)^3 - 8*n^5) / ((1 + 2*n)^2 + 4*n^2) + 1| < ε :=
by sorry

end NUMINAMATH_CALUDE_limit_evaluation_l4140_414033


namespace NUMINAMATH_CALUDE_sarah_initial_cupcakes_l4140_414004

/-- The number of cupcakes Todd ate -/
def cupcakes_eaten : ℕ := 14

/-- The number of packages Sarah could make after Todd ate some cupcakes -/
def packages : ℕ := 3

/-- The number of cupcakes in each package -/
def cupcakes_per_package : ℕ := 8

/-- The initial number of cupcakes Sarah baked -/
def initial_cupcakes : ℕ := cupcakes_eaten + packages * cupcakes_per_package

theorem sarah_initial_cupcakes : initial_cupcakes = 38 := by
  sorry

end NUMINAMATH_CALUDE_sarah_initial_cupcakes_l4140_414004
