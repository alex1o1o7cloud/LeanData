import Mathlib

namespace NUMINAMATH_CALUDE_smallest_matches_l340_34065

/-- Represents the types of regular polygons --/
inductive Polygon
  | Triangle
  | Square
  | Pentagon
  | Hexagon

/-- Returns the number of sides for a given polygon --/
def sides (p : Polygon) : Nat :=
  match p with
  | .Triangle => 3
  | .Square => 4
  | .Pentagon => 5
  | .Hexagon => 6

/-- Checks if it's possible to form a pair of polygons with a given number of matches --/
def canFormPair (n : Nat) (p1 p2 : Polygon) : Prop :=
  ∃ (x y : Nat), x > 0 ∧ y > 0 ∧ x * sides p1 + y * sides p2 = n

/-- States that 11 matches can form certain pairs but not others --/
axiom eleven_matches :
  (canFormPair 11 Polygon.Triangle Polygon.Pentagon) ∧
  (canFormPair 11 Polygon.Pentagon Polygon.Hexagon) ∧
  (canFormPair 11 Polygon.Square Polygon.Triangle) ∧
  ¬(canFormPair 11 Polygon.Triangle Polygon.Hexagon) ∧
  ¬(canFormPair 11 Polygon.Square Polygon.Pentagon) ∧
  ¬(canFormPair 11 Polygon.Square Polygon.Hexagon)

/-- The main theorem stating that 36 is the smallest number of matches --/
theorem smallest_matches :
  (∀ (p1 p2 : Polygon), canFormPair 36 p1 p2) ∧
  (∀ (n : Nat), n < 36 → ∃ (p1 p2 : Polygon), ¬(canFormPair n p1 p2)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_matches_l340_34065


namespace NUMINAMATH_CALUDE_no_real_roots_for_geometric_sequence_quadratic_l340_34036

/-- If a, b, c form a geometric sequence, then ax^2 + bx + c = 0 has no real roots -/
theorem no_real_roots_for_geometric_sequence_quadratic 
  (a b c : ℝ) (h : b^2 = a*c) : 
  ∀ x : ℝ, a*x^2 + b*x + c ≠ 0 := by
sorry

end NUMINAMATH_CALUDE_no_real_roots_for_geometric_sequence_quadratic_l340_34036


namespace NUMINAMATH_CALUDE_sin_330_degrees_l340_34063

theorem sin_330_degrees : Real.sin (330 * π / 180) = -1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_330_degrees_l340_34063


namespace NUMINAMATH_CALUDE_carter_reading_rate_l340_34048

/-- The number of pages Oliver can read in 1 hour -/
def oliver_pages : ℕ := 40

/-- The number of pages Lucy can read in 1 hour -/
def lucy_pages : ℕ := oliver_pages + 20

/-- The number of pages Carter can read in 1 hour -/
def carter_pages : ℕ := lucy_pages / 2

/-- Proves that Carter can read 30 pages in 1 hour given the conditions -/
theorem carter_reading_rate : carter_pages = 30 := by
  sorry

end NUMINAMATH_CALUDE_carter_reading_rate_l340_34048


namespace NUMINAMATH_CALUDE_jenny_rommel_age_difference_l340_34083

/-- Given the ages and relationships of Tim, Rommel, and Jenny, prove that Jenny is 2 years older than Rommel -/
theorem jenny_rommel_age_difference :
  ∀ (tim_age rommel_age jenny_age : ℕ),
  tim_age = 5 →
  rommel_age = 3 * tim_age →
  jenny_age = tim_age + 12 →
  jenny_age - rommel_age = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_jenny_rommel_age_difference_l340_34083


namespace NUMINAMATH_CALUDE_trigonometric_equation_solution_l340_34091

theorem trigonometric_equation_solution (x : ℝ) :
  (5.32 * Real.sin (2 * x) * Real.sin (6 * x) * Real.cos (4 * x) + (1/4) * Real.cos (12 * x) = 0) ↔
  (∃ k : ℤ, x = (π / 8) * (2 * k + 1)) ∨
  (∃ k : ℤ, x = (π / 12) * (6 * k + 1) ∨ x = (π / 12) * (6 * k - 1)) :=
by sorry

end NUMINAMATH_CALUDE_trigonometric_equation_solution_l340_34091


namespace NUMINAMATH_CALUDE_choose_cooks_l340_34097

theorem choose_cooks (n : ℕ) (k : ℕ) (h1 : n = 10) (h2 : k = 3) :
  Nat.choose n k = 120 := by
  sorry

end NUMINAMATH_CALUDE_choose_cooks_l340_34097


namespace NUMINAMATH_CALUDE_smallest_prime_factor_of_1471_l340_34035

theorem smallest_prime_factor_of_1471 :
  (Nat.minFac 1471 = 13) := by
  sorry

end NUMINAMATH_CALUDE_smallest_prime_factor_of_1471_l340_34035


namespace NUMINAMATH_CALUDE_inequality_relations_l340_34017

theorem inequality_relations (r p q : ℝ) 
  (hr : r > 0) (hp : p > 0) (hq : q > 0) (hpq : p^2 * r > q^2 * r) : 
  p > q ∧ |p| > |q| ∧ 1/p < 1/q := by sorry

end NUMINAMATH_CALUDE_inequality_relations_l340_34017


namespace NUMINAMATH_CALUDE_product_equality_l340_34015

theorem product_equality : 469111 * 9999 = 4690428889 := by
  sorry

end NUMINAMATH_CALUDE_product_equality_l340_34015


namespace NUMINAMATH_CALUDE_max_value_reciprocal_sum_l340_34047

theorem max_value_reciprocal_sum (x y a b : ℝ) 
  (ha : a > 1) 
  (hb : b > 1) 
  (hax : a^x = 3) 
  (hby : b^y = 3) 
  (hab : a + b = 2 * Real.sqrt 3) : 
  (∃ (z : ℝ), ∀ (w : ℝ), 1/x + 1/y ≤ w → w ≤ z) ∧ 
  (∃ (x0 y0 : ℝ), 1/x0 + 1/y0 = 1 ∧ 
    a^x0 = 3 ∧ b^y0 = 3 ∧ a + b = 2 * Real.sqrt 3) :=
sorry

end NUMINAMATH_CALUDE_max_value_reciprocal_sum_l340_34047


namespace NUMINAMATH_CALUDE_cubic_expression_value_l340_34023

theorem cubic_expression_value (p q : ℝ) : 
  3 * p^2 - 5 * p - 2 = 0 →
  3 * q^2 - 5 * q - 2 = 0 →
  p ≠ q →
  (9 * p^3 + 9 * q^3) / (p - q) = 215 / (3 * (p - q)) := by
  sorry

end NUMINAMATH_CALUDE_cubic_expression_value_l340_34023


namespace NUMINAMATH_CALUDE_quadratic_function_max_value_l340_34038

theorem quadratic_function_max_value (a : ℝ) (h1 : a ≠ 0) :
  (∀ x ∈ Set.Icc 0 3, a * x^2 - 2 * a * x ≤ 3) ∧
  (∃ x ∈ Set.Icc 0 3, a * x^2 - 2 * a * x = 3) →
  a = -3 ∨ a = 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_max_value_l340_34038


namespace NUMINAMATH_CALUDE_total_pencils_l340_34056

/-- The number of colors in a rainbow -/
def rainbow_colors : ℕ := 7

/-- The number of Chloe's friends who bought the same color box -/
def friends : ℕ := 5

/-- The total number of people who bought color boxes (Chloe and her friends) -/
def total_people : ℕ := friends + 1

/-- The number of pencils in each color box -/
def pencils_per_box : ℕ := rainbow_colors

theorem total_pencils :
  pencils_per_box * total_people = 42 :=
by sorry

end NUMINAMATH_CALUDE_total_pencils_l340_34056


namespace NUMINAMATH_CALUDE_fractional_equation_solution_l340_34068

theorem fractional_equation_solution (x : ℝ) (hx : x ≠ 0) (hx2 : 2 * x - 1 ≠ 0) :
  (2 / x = 3 / (2 * x - 1)) ↔ x = 2 := by sorry

end NUMINAMATH_CALUDE_fractional_equation_solution_l340_34068


namespace NUMINAMATH_CALUDE_number_of_divisors_of_30_l340_34010

theorem number_of_divisors_of_30 : Nat.card {d : ℕ | d > 0 ∧ 30 % d = 0} = 8 := by
  sorry

end NUMINAMATH_CALUDE_number_of_divisors_of_30_l340_34010


namespace NUMINAMATH_CALUDE_min_value_of_3a_plus_2_l340_34007

theorem min_value_of_3a_plus_2 (a : ℝ) (h : 8 * a^2 + 7 * a + 6 = 5) :
  ∃ (m : ℝ), (3 * a + 2 ≥ m) ∧ (∀ x, 8 * x^2 + 7 * x + 6 = 5 → 3 * x + 2 ≥ m) ∧ m = -1 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_3a_plus_2_l340_34007


namespace NUMINAMATH_CALUDE_quadratic_root_product_l340_34022

theorem quadratic_root_product (x₁ x₂ : ℝ) : 
  (x₁^2 - 4*x₁ + 1 = 0) → 
  (x₂^2 - 4*x₂ + 1 = 0) → 
  x₁ * x₂ = 1 := by
sorry

end NUMINAMATH_CALUDE_quadratic_root_product_l340_34022


namespace NUMINAMATH_CALUDE_stone_121_is_10_l340_34076

/-- The number of stones in the sequence -/
def n : ℕ := 11

/-- The length of a full cycle (left-to-right and right-to-left) -/
def cycle_length : ℕ := 2 * n - 1

/-- The position of a stone in the original left-to-right count, given its count number -/
def stone_position (count : ℕ) : ℕ :=
  (count - 1) % cycle_length + 1

/-- The theorem stating that the 121st count corresponds to the 10th stone -/
theorem stone_121_is_10 : stone_position 121 = 10 := by
  sorry

end NUMINAMATH_CALUDE_stone_121_is_10_l340_34076


namespace NUMINAMATH_CALUDE_office_staff_composition_l340_34049

/-- Represents the number of officers in an office. -/
def num_officers : ℕ := 15

/-- Represents the number of non-officers in the office. -/
def num_non_officers : ℕ := 480

/-- Represents the average salary of all employees in Rs/month. -/
def avg_salary_all : ℕ := 120

/-- Represents the average salary of officers in Rs/month. -/
def avg_salary_officers : ℕ := 440

/-- Represents the average salary of non-officers in Rs/month. -/
def avg_salary_non_officers : ℕ := 110

/-- Theorem stating that the number of officers is 15, given the conditions of the problem. -/
theorem office_staff_composition :
  num_officers = 15 ∧
  num_non_officers = 480 ∧
  avg_salary_all * (num_officers + num_non_officers) = 
    avg_salary_officers * num_officers + avg_salary_non_officers * num_non_officers :=
by sorry


end NUMINAMATH_CALUDE_office_staff_composition_l340_34049


namespace NUMINAMATH_CALUDE_average_monthly_income_l340_34033

def monthly_expense_first_3 : ℕ := 1700
def monthly_expense_next_4 : ℕ := 1550
def monthly_expense_last_5 : ℕ := 1800
def annual_savings : ℕ := 5200

def total_expenses : ℕ := 
  monthly_expense_first_3 * 3 + 
  monthly_expense_next_4 * 4 + 
  monthly_expense_last_5 * 5

def total_income : ℕ := total_expenses + annual_savings

theorem average_monthly_income : 
  total_income / 12 = 2125 := by sorry

end NUMINAMATH_CALUDE_average_monthly_income_l340_34033


namespace NUMINAMATH_CALUDE_friendship_ratio_theorem_l340_34044

/-- Represents a boy in the school -/
structure Boy where
  id : Nat

/-- Represents a girl in the school -/
structure Girl where
  id : Nat

/-- The number of girls who know a given boy -/
def d_Boy (b : Boy) : ℕ := sorry

/-- The number of boys who know a given girl -/
def d_Girl (g : Girl) : ℕ := sorry

/-- Represents that a boy and a girl know each other -/
def knows (b : Boy) (g : Girl) : Prop := sorry

theorem friendship_ratio_theorem 
  (n m : ℕ) 
  (boys : Finset Boy) 
  (girls : Finset Girl) 
  (h_boys : boys.card = n) 
  (h_girls : girls.card = m) 
  (h_girls_know_boy : ∀ g : Girl, ∃ b : Boy, knows b g) :
  ∃ (b : Boy) (g : Girl), 
    knows b g ∧ (d_Boy b : ℚ) / (d_Girl g : ℚ) ≥ (m : ℚ) / (n : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_friendship_ratio_theorem_l340_34044


namespace NUMINAMATH_CALUDE_fraction_subtraction_l340_34094

theorem fraction_subtraction : 
  (3 + 6 + 9) / (2 + 5 + 8) - (2 + 5 + 8) / (3 + 6 + 9) = 11 / 30 := by
  sorry

end NUMINAMATH_CALUDE_fraction_subtraction_l340_34094


namespace NUMINAMATH_CALUDE_division_problem_l340_34062

theorem division_problem (dividend quotient remainder divisor : ℕ) : 
  dividend = 141 →
  quotient = 8 →
  remainder = 5 →
  dividend = divisor * quotient + remainder →
  divisor = 17 := by
sorry

end NUMINAMATH_CALUDE_division_problem_l340_34062


namespace NUMINAMATH_CALUDE_term_2007_is_6019_l340_34053

/-- An arithmetic sequence with first term 1, second term 4, and third term 7 -/
def arithmetic_sequence (n : ℕ) : ℕ :=
  1 + 3 * (n - 1)

/-- Theorem stating that the 2007th term of the sequence is 6019 -/
theorem term_2007_is_6019 : arithmetic_sequence 2007 = 6019 := by
  sorry

end NUMINAMATH_CALUDE_term_2007_is_6019_l340_34053


namespace NUMINAMATH_CALUDE_paint_distribution_l340_34024

def paint_problem (total : ℚ) (blue_ratio green_ratio white_ratio : ℕ) : Prop :=
  let total_ratio := blue_ratio + green_ratio + white_ratio
  let blue_amount := (blue_ratio : ℚ) * total / total_ratio
  let green_amount := (green_ratio : ℚ) * total / total_ratio
  let white_amount := (white_ratio : ℚ) * total / total_ratio
  blue_amount = 15 ∧ green_amount = 9 ∧ white_amount = 21

theorem paint_distribution :
  paint_problem 45 5 3 7 := by
  sorry

end NUMINAMATH_CALUDE_paint_distribution_l340_34024


namespace NUMINAMATH_CALUDE_sum_of_A_and_C_is_five_l340_34082

-- Define the multiplication problem
def multiplication_problem (A B C D : Nat) : Prop :=
  A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D ∧
  A < 10 ∧ B < 10 ∧ C < 10 ∧ D < 10 ∧
  (100 * A + 10 * B + A) * D = 1000 * C + 100 * B + 10 * A + D

-- State the theorem
theorem sum_of_A_and_C_is_five :
  ∀ A B C D : Nat, multiplication_problem A B C D → A + C = 5 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_A_and_C_is_five_l340_34082


namespace NUMINAMATH_CALUDE_total_earnings_calculation_l340_34016

theorem total_earnings_calculation 
  (x y : ℝ) 
  (h1 : 4 * x * (5 * y / 100) = 3 * x * (6 * y / 100) + 350) 
  (h2 : x * y = 17500) : 
  (3 * x * (6 * y / 100) + 4 * x * (5 * y / 100) + 5 * x * (4 * y / 100)) = 10150 := by
  sorry

end NUMINAMATH_CALUDE_total_earnings_calculation_l340_34016


namespace NUMINAMATH_CALUDE_smallest_w_l340_34093

def is_factor (a b : ℕ) : Prop := ∃ k : ℕ, b = a * k

theorem smallest_w (w : ℕ) : w > 0 →
  is_factor (2^7) (2880 * w) →
  is_factor (3^4) (2880 * w) →
  is_factor (5^3) (2880 * w) →
  is_factor (7^3) (2880 * w) →
  is_factor (11^2) (2880 * w) →
  w ≥ 37348700 :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_w_l340_34093


namespace NUMINAMATH_CALUDE_envelope_height_l340_34099

theorem envelope_height (width : ℝ) (area : ℝ) (height : ℝ) : 
  width = 6 → area = 36 → area = width * height → height = 6 := by
  sorry

end NUMINAMATH_CALUDE_envelope_height_l340_34099


namespace NUMINAMATH_CALUDE_four_digit_solution_l340_34074

def is_four_digit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n ≤ 9999

def digit_value (n : ℕ) (place : ℕ) : ℕ :=
  (n / (10^place)) % 10

def number_from_digits (a b c d : ℕ) : ℕ :=
  1000 * a + 100 * b + 10 * c + d

theorem four_digit_solution :
  let abcd := 2996
  let dcba := number_from_digits (digit_value abcd 0) (digit_value abcd 1) (digit_value abcd 2) (digit_value abcd 3)
  is_four_digit abcd ∧ is_four_digit dcba ∧ 2 * abcd + 1000 = dcba := by
  sorry

end NUMINAMATH_CALUDE_four_digit_solution_l340_34074


namespace NUMINAMATH_CALUDE_system_solution_l340_34030

theorem system_solution :
  ∃ (x y : ℚ), 
    (2 * x - 3 * y = 1) ∧
    (5 * x + 4 * y = 6) ∧
    (x + 2 * y = 2) ∧
    (x = 2/3) ∧
    (y = 2/3) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l340_34030


namespace NUMINAMATH_CALUDE_plane_perpendicular_condition_l340_34001

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular and parallel relations
variable (perp : Plane → Plane → Prop)
variable (para : Plane → Plane → Prop)

-- State the theorem
theorem plane_perpendicular_condition 
  (a b : Line) (α β γ : Plane) :
  perp α γ → para γ β → perp α β :=
by sorry

end NUMINAMATH_CALUDE_plane_perpendicular_condition_l340_34001


namespace NUMINAMATH_CALUDE_train_carriage_seats_l340_34009

theorem train_carriage_seats : 
  ∀ (seats_per_carriage : ℕ),
  (3 * 4 * (seats_per_carriage + 10) = 420) →
  seats_per_carriage = 25 := by
sorry

end NUMINAMATH_CALUDE_train_carriage_seats_l340_34009


namespace NUMINAMATH_CALUDE_min_value_sum_reciprocals_l340_34088

theorem min_value_sum_reciprocals (a b c d e f : ℝ) 
  (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c) 
  (pos_d : 0 < d) (pos_e : 0 < e) (pos_f : 0 < f)
  (sum_eq_10 : a + b + c + d + e + f = 10) :
  (4/a + 9/b + 16/c + 25/d + 36/e + 49/f) ≥ 72.9 := by
  sorry

end NUMINAMATH_CALUDE_min_value_sum_reciprocals_l340_34088


namespace NUMINAMATH_CALUDE_break_even_point_manuals_l340_34000

/-- The break-even point for manual production -/
theorem break_even_point_manuals :
  let average_cost (Q : ℝ) := 100 + 100000 / Q
  let planned_price := 300
  ∃ Q : ℝ, Q > 0 ∧ average_cost Q = planned_price ∧ Q = 500 :=
by sorry

end NUMINAMATH_CALUDE_break_even_point_manuals_l340_34000


namespace NUMINAMATH_CALUDE_red_ball_probability_l340_34096

/-- The probability of selecting a red ball from a bag -/
def probability_red_ball (total_balls : ℕ) (red_balls : ℕ) : ℚ :=
  red_balls / total_balls

/-- Theorem: The probability of selecting a red ball from a bag with 15 balls, 
    of which 3 are red, is 1/5 -/
theorem red_ball_probability :
  probability_red_ball 15 3 = 1 / 5 := by
  sorry

#eval probability_red_ball 15 3

end NUMINAMATH_CALUDE_red_ball_probability_l340_34096


namespace NUMINAMATH_CALUDE_parabola_intersection_slope_product_l340_34064

-- Define the parabola C
def C (p : ℝ) (x y : ℝ) : Prop := y^2 = 2*p*x ∧ p > 0

-- Define that C passes through (2,2)
def C_passes_through_2_2 (p : ℝ) : Prop := C p 2 2

-- Define the line passing through (2,0)
def line_through_2_0 (m : ℝ) (x y : ℝ) : Prop := x = m*y + 2

-- Define the intersection points A and B
def intersection_points (p m x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  C p x₁ y₁ ∧ C p x₂ y₂ ∧ line_through_2_0 m x₁ y₁ ∧ line_through_2_0 m x₂ y₂

-- Define the slopes of OA and OB
def slopes (x₁ y₁ x₂ y₂ k₁ k₂ : ℝ) : Prop :=
  k₁ = y₁ / x₁ ∧ k₂ = y₂ / x₂

-- The main theorem
theorem parabola_intersection_slope_product :
  ∀ (p m x₁ y₁ x₂ y₂ k₁ k₂ : ℝ),
    C_passes_through_2_2 p →
    intersection_points p m x₁ y₁ x₂ y₂ →
    slopes x₁ y₁ x₂ y₂ k₁ k₂ →
    k₁ * k₂ = -1 := by
  sorry

end NUMINAMATH_CALUDE_parabola_intersection_slope_product_l340_34064


namespace NUMINAMATH_CALUDE_largest_difference_l340_34041

def P : ℕ := 3 * 2003^2004
def Q : ℕ := 2003^2004
def R : ℕ := 2002 * 2003^2003
def S : ℕ := 3 * 2003^2003
def T : ℕ := 2003^2003
def U : ℕ := 2003^2002

theorem largest_difference (P Q R S T U : ℕ) 
  (hP : P = 3 * 2003^2004)
  (hQ : Q = 2003^2004)
  (hR : R = 2002 * 2003^2003)
  (hS : S = 3 * 2003^2003)
  (hT : T = 2003^2003)
  (hU : U = 2003^2002) :
  P - Q > Q - R ∧ P - Q > R - S ∧ P - Q > S - T ∧ P - Q > T - U :=
by
  sorry

end NUMINAMATH_CALUDE_largest_difference_l340_34041


namespace NUMINAMATH_CALUDE_f_properties_l340_34011

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (a * x / Real.exp x) + (1/2) * x^2 - x

def monotonic_intervals (a : ℝ) : Prop :=
  (a ≤ 0 → (∀ x y, x < y → x < 1 → f a x > f a y) ∧ 
            (∀ x y, x < y → 1 < x → f a x < f a y)) ∧
  (a = Real.exp 1 → (∀ x y, x < y → f a x < f a y)) ∧
  (0 < a ∧ a < Real.exp 1 → 
    (∀ x y, x < y → y < Real.log a → f a x < f a y) ∧
    (∀ x y, x < y → Real.log a < x ∧ y < 1 → f a x > f a y) ∧
    (∀ x y, x < y → 1 < x → f a x < f a y)) ∧
  (Real.exp 1 < a → 
    (∀ x y, x < y → y < 1 → f a x < f a y) ∧
    (∀ x y, x < y → 1 < x ∧ y < Real.log a → f a x > f a y) ∧
    (∀ x y, x < y → Real.log a < x → f a x < f a y))

def number_of_zeros (a : ℝ) : Prop :=
  (Real.exp 1 / 2 < a → ∃! x, f a x = 0) ∧
  ((a = 1 ∨ a = Real.exp 1 / 2) → ∃ x y, x ≠ y ∧ f a x = 0 ∧ f a y = 0 ∧ 
    (∀ z, f a z = 0 → z = x ∨ z = y)) ∧
  (1 < a ∧ a < Real.exp 1 / 2 → ∃ x y z, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧ 
    f a x = 0 ∧ f a y = 0 ∧ f a z = 0 ∧ 
    (∀ w, f a w = 0 → w = x ∨ w = y ∨ w = z))

theorem f_properties (a : ℝ) (h : 1 ≤ a) : 
  monotonic_intervals a ∧ number_of_zeros a := by sorry

end NUMINAMATH_CALUDE_f_properties_l340_34011


namespace NUMINAMATH_CALUDE_y_derivative_l340_34040

noncomputable def y (x : ℝ) : ℝ :=
  (3^x * (Real.log 3 * Real.sin (2*x) - 2 * Real.cos (2*x))) / ((Real.log 3)^2 + 4)

theorem y_derivative (x : ℝ) :
  deriv y x = 3^x * Real.sin (2*x) :=
by sorry

end NUMINAMATH_CALUDE_y_derivative_l340_34040


namespace NUMINAMATH_CALUDE_book_price_problem_l340_34059

theorem book_price_problem (original_price : ℝ) : 
  original_price * (1 - 0.25) + original_price * (1 - 0.40) = 66 → 
  original_price = 48.89 := by
sorry

end NUMINAMATH_CALUDE_book_price_problem_l340_34059


namespace NUMINAMATH_CALUDE_absolute_value_of_negative_l340_34026

theorem absolute_value_of_negative (x : ℝ) : x < 0 → |x| = -x := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_of_negative_l340_34026


namespace NUMINAMATH_CALUDE_coefficient_x_squared_expansion_l340_34075

/-- The coefficient of x^2 in the expansion of (1/√x + x)^8 -/
def coefficient_x_squared : ℕ := 70

/-- The binomial coefficient (8 choose 4) -/
def binomial_8_4 : ℕ := 70

theorem coefficient_x_squared_expansion :
  coefficient_x_squared = binomial_8_4 := by sorry

end NUMINAMATH_CALUDE_coefficient_x_squared_expansion_l340_34075


namespace NUMINAMATH_CALUDE_remainder_8457_mod_9_l340_34089

/-- The sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- A number is congruent to the sum of its digits modulo 9 -/
axiom sum_of_digits_mod_9 (n : ℕ) : n ≡ sum_of_digits n [MOD 9]

/-- The remainder when 8457 is divided by 9 is 6 -/
theorem remainder_8457_mod_9 : 8457 % 9 = 6 := by sorry

end NUMINAMATH_CALUDE_remainder_8457_mod_9_l340_34089


namespace NUMINAMATH_CALUDE_nancy_target_amount_l340_34070

def hourly_rate (total_earnings : ℚ) (hours_worked : ℚ) : ℚ :=
  total_earnings / hours_worked

def target_amount (rate : ℚ) (target_hours : ℚ) : ℚ :=
  rate * target_hours

theorem nancy_target_amount 
  (initial_earnings : ℚ) 
  (initial_hours : ℚ) 
  (target_hours : ℚ) 
  (h1 : initial_earnings = 28)
  (h2 : initial_hours = 4)
  (h3 : target_hours = 10) :
  target_amount (hourly_rate initial_earnings initial_hours) target_hours = 70 :=
by
  sorry

end NUMINAMATH_CALUDE_nancy_target_amount_l340_34070


namespace NUMINAMATH_CALUDE_supersonic_pilot_l340_34025

theorem supersonic_pilot (total_distance : ℝ) 
  (dupon_distance dupon_remaining duran_distance duran_remaining : ℝ) : 
  (dupon_distance + dupon_remaining = total_distance) →
  (duran_distance + duran_remaining = total_distance) →
  (2 * dupon_distance + dupon_remaining / 1.5 = total_distance) →
  (duran_distance / 1.5 + 2 * duran_remaining = total_distance) →
  (duran_distance = 3 * duran_remaining) →
  (duran_distance = 3 / 4 * total_distance) :=
by sorry

end NUMINAMATH_CALUDE_supersonic_pilot_l340_34025


namespace NUMINAMATH_CALUDE_smallest_divisor_of_28_l340_34006

theorem smallest_divisor_of_28 : ∀ d : ℕ, d > 0 → d ∣ 28 → d ≥ 1 :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_divisor_of_28_l340_34006


namespace NUMINAMATH_CALUDE_sum_of_squares_l340_34071

theorem sum_of_squares (x y : ℝ) (h1 : x + y = 18) (h2 : x * y = 72) : x^2 + y^2 = 180 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_l340_34071


namespace NUMINAMATH_CALUDE_alissa_has_more_present_difference_l340_34037

/-- The number of presents Ethan has -/
def ethan_presents : ℕ := 31

/-- The number of presents Alissa has -/
def alissa_presents : ℕ := 53

/-- Alissa has more presents than Ethan -/
theorem alissa_has_more : alissa_presents > ethan_presents := by sorry

/-- The difference between Alissa's and Ethan's presents is 22 -/
theorem present_difference : alissa_presents - ethan_presents = 22 := by sorry

end NUMINAMATH_CALUDE_alissa_has_more_present_difference_l340_34037


namespace NUMINAMATH_CALUDE_problem_solution_l340_34004

theorem problem_solution : -1^6 + 8 / (-2)^2 - |(-4) * 3| = -9 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l340_34004


namespace NUMINAMATH_CALUDE_intersection_point_satisfies_equations_l340_34095

theorem intersection_point_satisfies_equations :
  let x : ℚ := 75 / 8
  let y : ℚ := 15 / 8
  (3 * x^2 - 12 * y^2 = 48) ∧ (y = -1/3 * x + 5) := by
  sorry

end NUMINAMATH_CALUDE_intersection_point_satisfies_equations_l340_34095


namespace NUMINAMATH_CALUDE_alyssa_cans_collected_l340_34012

theorem alyssa_cans_collected (total_cans : ℕ) (abigail_cans : ℕ) (cans_needed : ℕ) 
  (h1 : total_cans = 100)
  (h2 : abigail_cans = 43)
  (h3 : cans_needed = 27) :
  total_cans - (abigail_cans + cans_needed) = 30 := by
  sorry

end NUMINAMATH_CALUDE_alyssa_cans_collected_l340_34012


namespace NUMINAMATH_CALUDE_marks_radiator_cost_l340_34031

/-- The total cost of replacing a car radiator -/
def total_cost (work_duration : ℕ) (hourly_rate : ℕ) (part_cost : ℕ) : ℕ :=
  work_duration * hourly_rate + part_cost

/-- Proof that Mark's total cost for replacing his car radiator is $300 -/
theorem marks_radiator_cost :
  total_cost 2 75 150 = 300 := by
  sorry

end NUMINAMATH_CALUDE_marks_radiator_cost_l340_34031


namespace NUMINAMATH_CALUDE_convention_handshakes_l340_34066

/-- The number of handshakes in a convention --/
def number_of_handshakes (num_companies : ℕ) (reps_per_company : ℕ) : ℕ :=
  let total_people := num_companies * reps_per_company
  let handshakes_per_person := total_people - reps_per_company
  (total_people * handshakes_per_person) / 2

/-- Theorem stating the number of handshakes for the given convention --/
theorem convention_handshakes :
  number_of_handshakes 5 3 = 90 := by
  sorry

end NUMINAMATH_CALUDE_convention_handshakes_l340_34066


namespace NUMINAMATH_CALUDE_chord_equation_l340_34020

/-- Given a circle with equation x² + y² = 9 and a chord PQ with midpoint (1,2),
    the equation of line PQ is x + 2y - 5 = 0 -/
theorem chord_equation (P Q : ℝ × ℝ) : 
  (∀ (x y : ℝ), x^2 + y^2 = 9 → (x - 1)^2 + (y - 2)^2 = ((P.1 - 1)^2 + (P.2 - 2)^2)) →
  ((P.1 + Q.1) / 2 = 1 ∧ (P.2 + Q.2) / 2 = 2) →
  ∃ (k : ℝ), ∀ (x y : ℝ), (x = P.1 ∧ y = P.2) ∨ (x = Q.1 ∧ y = Q.2) → x + 2*y - 5 = k := by
  sorry

end NUMINAMATH_CALUDE_chord_equation_l340_34020


namespace NUMINAMATH_CALUDE_no_rational_roots_for_odd_coefficients_l340_34008

theorem no_rational_roots_for_odd_coefficients (p q : ℤ) (hp : Odd p) (hq : Odd q) :
  ¬∃ (x : ℚ), x^2 + 2*↑p*x + 2*↑q = 0 := by
  sorry

end NUMINAMATH_CALUDE_no_rational_roots_for_odd_coefficients_l340_34008


namespace NUMINAMATH_CALUDE_prime_divisor_equality_l340_34039

theorem prime_divisor_equality (p q : ℕ) (hp : Nat.Prime p) (hq : Nat.Prime q) (hdiv : p ∣ q) : p = q := by
  sorry

end NUMINAMATH_CALUDE_prime_divisor_equality_l340_34039


namespace NUMINAMATH_CALUDE_tv_cost_l340_34098

theorem tv_cost (savings : ℝ) (furniture_fraction : ℝ) (tv_cost : ℝ) : 
  savings = 880 → 
  furniture_fraction = 3/4 → 
  tv_cost = savings * (1 - furniture_fraction) → 
  tv_cost = 220 := by
sorry

end NUMINAMATH_CALUDE_tv_cost_l340_34098


namespace NUMINAMATH_CALUDE_three_digit_numbers_divisible_by_nine_l340_34077

/-- Given nonzero digits a, b, and c that form 6 distinct three-digit numbers,
    if the sum of these numbers is 5994, then each number is divisible by 9. -/
theorem three_digit_numbers_divisible_by_nine 
  (a b c : ℕ) 
  (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (hdistinct : a ≠ b ∧ b ≠ c ∧ a ≠ c)
  (hsum : 100 * (a + b + c) + 10 * (a + b + c) + (a + b + c) = 5994) :
  let numbers := [100*a + 10*b + c, 100*a + 10*c + b, 
                  100*b + 10*a + c, 100*b + 10*c + a, 
                  100*c + 10*a + b, 100*c + 10*b + a]
  ∀ n ∈ numbers, n % 9 = 0 :=
by sorry

end NUMINAMATH_CALUDE_three_digit_numbers_divisible_by_nine_l340_34077


namespace NUMINAMATH_CALUDE_percentage_of_defective_meters_l340_34029

theorem percentage_of_defective_meters (total_meters examined_meters : ℕ) 
  (h1 : total_meters = 120) (h2 : examined_meters = 12) :
  (examined_meters : ℝ) / total_meters * 100 = 10 := by
  sorry

end NUMINAMATH_CALUDE_percentage_of_defective_meters_l340_34029


namespace NUMINAMATH_CALUDE_parallel_segments_k_value_l340_34080

/-- Given four points on a Cartesian plane where segment AB is parallel to segment XY, 
    prove that k = -8. -/
theorem parallel_segments_k_value 
  (A B X Y : ℝ × ℝ)
  (hA : A = (-6, 0))
  (hB : B = (0, -6))
  (hX : X = (0, 10))
  (hY : Y = (18, k))
  (h_parallel : (B.2 - A.2) * (Y.1 - X.1) = (Y.2 - X.2) * (B.1 - A.1)) :
  k = -8 :=
by sorry

end NUMINAMATH_CALUDE_parallel_segments_k_value_l340_34080


namespace NUMINAMATH_CALUDE_opposite_of_negative_five_l340_34073

theorem opposite_of_negative_five : -((-5 : ℤ)) = 5 := by sorry

end NUMINAMATH_CALUDE_opposite_of_negative_five_l340_34073


namespace NUMINAMATH_CALUDE_population_increase_l340_34067

theorem population_increase (a b c : ℝ) :
  let increase_0_to_1 := 1 + a / 100
  let increase_1_to_2 := 1 + b / 100
  let increase_2_to_3 := 1 + c / 100
  let total_increase := increase_0_to_1 * increase_1_to_2 * increase_2_to_3 - 1
  total_increase * 100 = a + b + c + (a * b + b * c + a * c) / 100 + a * b * c / 10000 :=
by sorry

end NUMINAMATH_CALUDE_population_increase_l340_34067


namespace NUMINAMATH_CALUDE_find_first_number_l340_34018

theorem find_first_number (x : ℝ) (y : ℝ) : 
  (28 + x + 42 + 78 + 104) / 5 = 90 →
  (y + 255 + 511 + 1023 + x) / 5 = 423 →
  y = 128 := by
sorry

end NUMINAMATH_CALUDE_find_first_number_l340_34018


namespace NUMINAMATH_CALUDE_tangent_lines_equal_implies_a_equals_one_l340_34057

noncomputable section

-- Define the functions f and g
def f (x : ℝ) : ℝ := x^2 - 2
def g (a : ℝ) (x : ℝ) : ℝ := 3 * Real.log x - a * x

-- Define the derivatives of f and g
def f' (x : ℝ) : ℝ := 2 * x
def g' (a : ℝ) (x : ℝ) : ℝ := 3 / x - a

-- Theorem statement
theorem tangent_lines_equal_implies_a_equals_one :
  ∃ (x : ℝ), x > 0 ∧ f x = g 1 x ∧ f' x = g' 1 x :=
sorry

end

end NUMINAMATH_CALUDE_tangent_lines_equal_implies_a_equals_one_l340_34057


namespace NUMINAMATH_CALUDE_power_division_eight_sixtyfour_l340_34051

theorem power_division_eight_sixtyfour : 8^15 / 64^7 = 8 := by
  sorry

end NUMINAMATH_CALUDE_power_division_eight_sixtyfour_l340_34051


namespace NUMINAMATH_CALUDE_max_intersections_theorem_l340_34086

/-- A polygon in a plane -/
structure Polygon where
  sides : ℕ

/-- A regular polygon -/
structure RegularPolygon extends Polygon

/-- An irregular polygon -/
structure IrregularPolygon extends Polygon

/-- Two polygons that overlap but share no complete side -/
structure OverlappingPolygons where
  P₁ : RegularPolygon
  P₂ : IrregularPolygon
  overlap : Bool
  no_shared_side : Bool

/-- The maximum number of intersection points between two polygons -/
def max_intersections (op : OverlappingPolygons) : ℕ :=
  op.P₁.sides * op.P₂.sides

/-- Theorem: The maximum number of intersections between a regular polygon P₁
    and an irregular polygon P₂, where they overlap but share no complete side,
    is the product of their number of sides -/
theorem max_intersections_theorem (op : OverlappingPolygons)
    (h : op.P₁.sides ≤ op.P₂.sides) :
    max_intersections op = op.P₁.sides * op.P₂.sides :=
  sorry

end NUMINAMATH_CALUDE_max_intersections_theorem_l340_34086


namespace NUMINAMATH_CALUDE_number_equality_l340_34072

theorem number_equality (x : ℚ) : 
  (35 / 100) * x = (30 / 100) * 50 → x = 300 / 7 := by
sorry

end NUMINAMATH_CALUDE_number_equality_l340_34072


namespace NUMINAMATH_CALUDE_calculation_proof_l340_34060

theorem calculation_proof : -3^2 - (-1)^4 * 5 / (-5/3) = -6 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l340_34060


namespace NUMINAMATH_CALUDE_sum_of_valid_starting_values_l340_34090

def transform (n : ℕ) : ℕ :=
  if n % 2 = 0 then n / 2 else 4 * n + 1

def apply_transform (n : ℕ) (times : ℕ) : ℕ :=
  match times with
  | 0 => n
  | m + 1 => transform (apply_transform n m)

def valid_starting_values : List ℕ :=
  (List.range 100).filter (λ n => apply_transform n 6 = 1)

theorem sum_of_valid_starting_values :
  valid_starting_values.sum = 85 := by sorry

end NUMINAMATH_CALUDE_sum_of_valid_starting_values_l340_34090


namespace NUMINAMATH_CALUDE_prop_evaluation_l340_34013

-- Define the propositions p and q
def p (x y : ℝ) : Prop := (x > y) → (-x < -y)
def q (x y : ℝ) : Prop := (x < y) → (x^2 < y^2)

-- State the theorem
theorem prop_evaluation : ∃ (x y : ℝ), (p x y ∨ q x y) ∧ (p x y ∧ ¬(q x y)) := by
  sorry

end NUMINAMATH_CALUDE_prop_evaluation_l340_34013


namespace NUMINAMATH_CALUDE_find_a_l340_34087

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then Real.log x else -x

-- State the theorem
theorem find_a : ∃ (a : ℝ), f (1/3) = (1/3) * f a ∧ a = 1/27 := by
  sorry

end NUMINAMATH_CALUDE_find_a_l340_34087


namespace NUMINAMATH_CALUDE_division_problem_l340_34034

theorem division_problem (total : ℚ) (a b c d : ℚ) : 
  total = 2880 →
  a = (1/3) * b →
  b = (2/5) * c →
  c = (3/4) * d →
  a + b + c + d = total →
  b = 403.2 := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l340_34034


namespace NUMINAMATH_CALUDE_sufficient_but_not_necessary_condition_l340_34014

/-- A sequence of 8 positive real numbers -/
def Sequence := Fin 8 → ℝ

/-- Predicate to check if a sequence is positive -/
def is_positive (s : Sequence) : Prop :=
  ∀ i, s i > 0

/-- Predicate to check if a sequence is geometric -/
def is_geometric (s : Sequence) : Prop :=
  ∃ q : ℝ, q > 0 ∧ ∀ i : Fin 7, s (i + 1) = q * s i

theorem sufficient_but_not_necessary_condition (s : Sequence) 
  (h_positive : is_positive s) :
  (s 0 + s 7 < s 3 + s 4 → ¬ is_geometric s) ∧
  ∃ s' : Sequence, is_positive s' ∧ ¬ is_geometric s' ∧ s' 0 + s' 7 ≥ s' 3 + s' 4 :=
by sorry

end NUMINAMATH_CALUDE_sufficient_but_not_necessary_condition_l340_34014


namespace NUMINAMATH_CALUDE_mixed_lubricant_price_l340_34042

/-- Represents an oil type with its volume, price, and discount or tax -/
structure OilType where
  volume : ℝ
  price : ℝ
  discount_or_tax : ℝ
  is_discount : Bool

/-- Calculates the total cost of an oil type after applying discount or tax -/
def calculate_cost (oil : OilType) : ℝ :=
  let base_cost := oil.volume * oil.price
  if oil.is_discount then
    base_cost * (1 - oil.discount_or_tax)
  else
    base_cost * (1 + oil.discount_or_tax)

/-- Theorem stating that the final price per litre of the mixed lubricant oil is approximately 52.80 -/
theorem mixed_lubricant_price (oils : List OilType) 
  (h1 : oils.length = 6)
  (h2 : oils[0] = OilType.mk 70 43 0.15 true)
  (h3 : oils[1] = OilType.mk 50 51 0.10 false)
  (h4 : oils[2] = OilType.mk 15 60 0.08 true)
  (h5 : oils[3] = OilType.mk 25 62 0.12 false)
  (h6 : oils[4] = OilType.mk 40 67 0.05 true)
  (h7 : oils[5] = OilType.mk 10 75 0.18 true) :
  let total_cost := oils.map calculate_cost |>.sum
  let total_volume := oils.map (·.volume) |>.sum
  abs (total_cost / total_volume - 52.80) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_mixed_lubricant_price_l340_34042


namespace NUMINAMATH_CALUDE_suraya_kayla_difference_l340_34061

/-- The number of apples picked by each person -/
structure ApplePicks where
  suraya : ℕ
  caleb : ℕ
  kayla : ℕ

/-- The conditions of the apple-picking scenario -/
def apple_picking_conditions (a : ApplePicks) : Prop :=
  a.suraya = a.caleb + 12 ∧
  a.caleb + 5 = a.kayla ∧
  a.kayla = 20

/-- The theorem stating that Suraya picked 7 more apples than Kayla -/
theorem suraya_kayla_difference (a : ApplePicks) 
  (h : apple_picking_conditions a) : a.suraya - a.kayla = 7 := by
  sorry


end NUMINAMATH_CALUDE_suraya_kayla_difference_l340_34061


namespace NUMINAMATH_CALUDE_quadratic_monotone_iff_a_geq_one_l340_34084

/-- A quadratic function f(x) = x^2 + 2ax + b is monotonically increasing
    on [-1, +∞) if and only if a ≥ 1 -/
theorem quadratic_monotone_iff_a_geq_one (a b : ℝ) :
  (∀ x y : ℝ, -1 ≤ x → x ≤ y → x^2 + 2*a*x + b ≤ y^2 + 2*a*y + b) ↔ a ≥ 1 :=
sorry

end NUMINAMATH_CALUDE_quadratic_monotone_iff_a_geq_one_l340_34084


namespace NUMINAMATH_CALUDE_solution_count_decrease_l340_34045

/-- The system of equations has fewer than four solutions if and only if a = ±1 or a = ±√2 -/
theorem solution_count_decrease (a : ℝ) : 
  (∃ x y : ℝ, x^2 - y^2 = 0 ∧ (x - a)^2 + y^2 = 1) ∧ 
  (∀ x₁ y₁ x₂ y₂ x₃ y₃ x₄ y₄ : ℝ, 
    (x₁^2 - y₁^2 = 0 ∧ (x₁ - a)^2 + y₁^2 = 1) → 
    (x₂^2 - y₂^2 = 0 ∧ (x₂ - a)^2 + y₂^2 = 1) → 
    (x₃^2 - y₃^2 = 0 ∧ (x₃ - a)^2 + y₃^2 = 1) → 
    (x₄^2 - y₄^2 = 0 ∧ (x₄ - a)^2 + y₄^2 = 1) → 
    (x₁ = x₂ ∧ y₁ = y₂) ∨ (x₁ = x₃ ∧ y₁ = y₃) ∨ (x₁ = x₄ ∧ y₁ = y₄) ∨ 
    (x₂ = x₃ ∧ y₂ = y₃) ∨ (x₂ = x₄ ∧ y₂ = y₄) ∨ (x₃ = x₄ ∧ y₃ = y₄)) ↔ 
  a = 1 ∨ a = -1 ∨ a = Real.sqrt 2 ∨ a = -Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_solution_count_decrease_l340_34045


namespace NUMINAMATH_CALUDE_coin_denomination_l340_34058

/-- Given a total bill of 285 pesos, paid with 11 20-peso bills and 11 coins of unknown denomination,
    prove that the denomination of the coins must be 5 pesos. -/
theorem coin_denomination (total_bill : ℕ) (bill_value : ℕ) (num_bills : ℕ) (num_coins : ℕ) 
  (h1 : total_bill = 285)
  (h2 : bill_value = 20)
  (h3 : num_bills = 11)
  (h4 : num_coins = 11) :
  ∃ (coin_value : ℕ), coin_value = 5 ∧ total_bill = num_bills * bill_value + num_coins * coin_value :=
by
  sorry

#check coin_denomination

end NUMINAMATH_CALUDE_coin_denomination_l340_34058


namespace NUMINAMATH_CALUDE_cryptarithmetic_solution_l340_34043

def is_distinct (a b c d e : ℕ) : Prop :=
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧
  b ≠ c ∧ b ≠ d ∧ b ≠ e ∧
  c ≠ d ∧ c ≠ e ∧
  d ≠ e

def is_digit (n : ℕ) : Prop := n < 10

def is_nonzero_digit (n : ℕ) : Prop := 0 < n ∧ n < 10

theorem cryptarithmetic_solution :
  ∃! (X Y B M C : ℕ),
    is_distinct X Y B M C ∧
    is_nonzero_digit X ∧
    is_digit Y ∧
    is_nonzero_digit B ∧
    is_digit M ∧
    is_digit C ∧
    X * 1000 + Y * 100 + 70 + B * 100 + M * 10 + C =
    B * 1000 + M * 100 + C * 10 + 0 :=
by sorry

end NUMINAMATH_CALUDE_cryptarithmetic_solution_l340_34043


namespace NUMINAMATH_CALUDE_sword_length_difference_l340_34092

theorem sword_length_difference (christopher_sword : ℕ) (jameson_sword : ℕ) (june_sword : ℕ) : 
  christopher_sword = 15 →
  jameson_sword = 2 * christopher_sword + 3 →
  june_sword = jameson_sword + 5 →
  june_sword - christopher_sword = 23 :=
by
  sorry

end NUMINAMATH_CALUDE_sword_length_difference_l340_34092


namespace NUMINAMATH_CALUDE_sandy_marbles_count_l340_34028

/-- The number of red marbles Jessica has -/
def jessica_marbles : ℕ := 3 * 12

/-- The factor by which Sandy has more marbles than Jessica -/
def sandy_factor : ℕ := 4

/-- The number of red marbles Sandy has -/
def sandy_marbles : ℕ := sandy_factor * jessica_marbles

theorem sandy_marbles_count : sandy_marbles = 144 := by
  sorry

end NUMINAMATH_CALUDE_sandy_marbles_count_l340_34028


namespace NUMINAMATH_CALUDE_farmer_land_usage_l340_34021

theorem farmer_land_usage (beans wheat corn total : ℕ) : 
  beans + wheat + corn = total →
  5 * wheat = 2 * beans →
  2 * corn = beans →
  corn = 376 →
  total = 1034 := by
sorry

end NUMINAMATH_CALUDE_farmer_land_usage_l340_34021


namespace NUMINAMATH_CALUDE_monomial_properties_l340_34005

/-- The coefficient of a monomial -/
def coefficient (m : ℚ) (x y : ℚ) : ℚ := m

/-- The degree of a monomial -/
def degree (x y : ℚ) : ℕ := 2 + 1

theorem monomial_properties :
  let m : ℚ := -π / 7
  let x : ℚ := 0  -- Placeholder value, not used in computation
  let y : ℚ := 0  -- Placeholder value, not used in computation
  (coefficient m x y = -π / 7) ∧ (degree x y = 3) := by
  sorry

end NUMINAMATH_CALUDE_monomial_properties_l340_34005


namespace NUMINAMATH_CALUDE_polygon_deformable_to_triangle_l340_34054

/-- A planar polygon represented by its vertices -/
structure PlanarPolygon where
  vertices : List (ℝ × ℝ)
  n : ℕ
  h_n : vertices.length = n

/-- A function that checks if a polygon can be deformed into a triangle -/
def can_deform_to_triangle (p : PlanarPolygon) : Prop :=
  ∃ (v1 v2 v3 : ℝ × ℝ), v1 ∈ p.vertices ∧ v2 ∈ p.vertices ∧ v3 ∈ p.vertices ∧
    v1 ≠ v2 ∧ v2 ≠ v3 ∧ v1 ≠ v3

/-- The main theorem stating that any planar polygon with more than 4 vertices
    can be deformed into a triangle -/
theorem polygon_deformable_to_triangle (p : PlanarPolygon) (h : p.n > 4) :
  can_deform_to_triangle p := by
  sorry

end NUMINAMATH_CALUDE_polygon_deformable_to_triangle_l340_34054


namespace NUMINAMATH_CALUDE_monotonic_f_range_a_l340_34078

/-- Piecewise function f(x) -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then -x^2 + a*x + a/4 else a^x

/-- Theorem stating the range of a for monotonically increasing f(x) -/
theorem monotonic_f_range_a (a : ℝ) :
  (∀ x y : ℝ, x < y → f a x < f a y) → 2 ≤ a ∧ a ≤ 4 := by
  sorry

end NUMINAMATH_CALUDE_monotonic_f_range_a_l340_34078


namespace NUMINAMATH_CALUDE_ellipse_equation_l340_34085

-- Define the line
def line (x y : ℝ) : Prop := 2 * x - y + 2 = 0

-- Define the general form of the ellipse
def ellipse (x y A B : ℝ) : Prop := (x^2 / A) + (y^2 / B) = 1

-- State the theorem
theorem ellipse_equation 
  (x y A B : ℝ) 
  (h1 : A > 0) 
  (h2 : B > 0) 
  (h3 : ∃ (xf yf xv yv : ℝ), 
    line xf yf ∧ line xv yv ∧ 
    ellipse xf yf A B ∧ ellipse xv yv A B ∧ 
    ((xf = 0 ∧ xv ≠ 0) ∨ (yf = 0 ∧ yv ≠ 0))) :
  ((A = 5 ∧ B = 4) ∨ (A = 1 ∧ B = 5)) :=
sorry

end NUMINAMATH_CALUDE_ellipse_equation_l340_34085


namespace NUMINAMATH_CALUDE_johns_class_boys_count_l340_34003

theorem johns_class_boys_count :
  ∀ (g b : ℕ),
  g + b = 28 →
  g = (3 * b) / 4 →
  b = 16 :=
by
  sorry

end NUMINAMATH_CALUDE_johns_class_boys_count_l340_34003


namespace NUMINAMATH_CALUDE_parallelogram_condition_l340_34069

/-- The condition for the existence of a parallelogram inscribed in an ellipse and tangent to a circle -/
theorem parallelogram_condition (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a > b) :
  (∀ (x y : ℝ), x^2 + y^2 = 1 →
    ∃ (P : ℝ × ℝ), P.1^2 / a^2 + P.2^2 / b^2 = 1 ∧
      ∃ (Q R S : ℝ × ℝ),
        Q.1^2 / a^2 + Q.2^2 / b^2 = 1 ∧
        R.1^2 / a^2 + R.2^2 / b^2 = 1 ∧
        S.1^2 / a^2 + S.2^2 / b^2 = 1 ∧
        (P.1 - Q.1) * (R.1 - S.1) + (P.2 - Q.2) * (R.2 - S.2) = 0 ∧
        (P.1 - R.1) * (Q.1 - S.1) + (P.2 - R.2) * (Q.2 - S.2) = 0 ∧
        ((P.1 - x)^2 + (P.2 - y)^2 = 1 ∨
         (Q.1 - x)^2 + (Q.2 - y)^2 = 1 ∨
         (R.1 - x)^2 + (R.2 - y)^2 = 1 ∨
         (S.1 - x)^2 + (S.2 - y)^2 = 1)) ↔
  1 / a^2 + 1 / b^2 = 1 :=
by sorry

end NUMINAMATH_CALUDE_parallelogram_condition_l340_34069


namespace NUMINAMATH_CALUDE_james_arthur_muffin_ratio_l340_34079

theorem james_arthur_muffin_ratio :
  let arthur_muffins : ℕ := 115
  let james_muffins : ℕ := 1380
  (james_muffins : ℚ) / (arthur_muffins : ℚ) = 12 := by sorry

end NUMINAMATH_CALUDE_james_arthur_muffin_ratio_l340_34079


namespace NUMINAMATH_CALUDE_school_students_count_l340_34055

theorem school_students_count :
  let blue_percent : ℝ := 0.45
  let red_percent : ℝ := 0.23
  let green_percent : ℝ := 0.15
  let other_count : ℕ := 102
  let total_count : ℕ := 600
  blue_percent + red_percent + green_percent + (other_count : ℝ) / total_count = 1 ∧
  (other_count : ℝ) / total_count = 1 - (blue_percent + red_percent + green_percent) :=
by sorry

end NUMINAMATH_CALUDE_school_students_count_l340_34055


namespace NUMINAMATH_CALUDE_total_oranges_l340_34027

def oranges_per_box : ℝ := 10.0
def boxes_packed : ℝ := 2650.0

theorem total_oranges : oranges_per_box * boxes_packed = 26500.0 := by
  sorry

end NUMINAMATH_CALUDE_total_oranges_l340_34027


namespace NUMINAMATH_CALUDE_area_smallest_rectangle_radius_6_l340_34052

/-- The area of the smallest rectangle containing a circle of given radius -/
def smallest_rectangle_area (radius : ℝ) : ℝ :=
  (2 * radius) * (3 * radius)

/-- Theorem: The area of the smallest rectangle containing a circle of radius 6 is 216 -/
theorem area_smallest_rectangle_radius_6 :
  smallest_rectangle_area 6 = 216 := by
  sorry

end NUMINAMATH_CALUDE_area_smallest_rectangle_radius_6_l340_34052


namespace NUMINAMATH_CALUDE_quadratic_roots_after_modification_l340_34081

theorem quadratic_roots_after_modification (a b t l : ℝ) :
  -1 < t → t < 0 →
  (∀ x, x^2 + a*x + b = 0 ↔ x = t ∨ x = l) →
  ∃ r₁ r₂, r₁ ≠ r₂ ∧ ∀ x, x^2 + (a+t)*x + (b+t) = 0 ↔ x = r₁ ∨ x = r₂ :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_after_modification_l340_34081


namespace NUMINAMATH_CALUDE_andrews_eggs_l340_34032

theorem andrews_eggs (initial_eggs bought_eggs final_eggs : ℕ) : 
  bought_eggs = 62 → final_eggs = 70 → initial_eggs + bought_eggs = final_eggs → initial_eggs = 8 := by
  sorry

end NUMINAMATH_CALUDE_andrews_eggs_l340_34032


namespace NUMINAMATH_CALUDE_temp_increase_proof_l340_34019

-- Define the temperatures
def last_night_temp : Int := -5
def current_temp : Int := 3

-- Define the temperature difference function
def temp_difference (t1 t2 : Int) : Int := t2 - t1

-- Theorem to prove
theorem temp_increase_proof : 
  temp_difference last_night_temp current_temp = 8 := by
  sorry

end NUMINAMATH_CALUDE_temp_increase_proof_l340_34019


namespace NUMINAMATH_CALUDE_complex_equation_solution_l340_34046

theorem complex_equation_solution (a : ℝ) : 
  (Complex.mk 2 a) * (Complex.mk a (-2)) = Complex.I * (-4) → a = 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l340_34046


namespace NUMINAMATH_CALUDE_girls_average_weight_is_27_l340_34050

/-- Given a class with boys and girls, calculates the average weight of girls -/
def average_weight_of_girls (total_students : ℕ) (num_boys : ℕ) (boys_avg_weight : ℚ) (class_avg_weight : ℚ) : ℚ :=
  let total_weight := class_avg_weight * total_students
  let boys_total_weight := boys_avg_weight * num_boys
  let girls_total_weight := total_weight - boys_total_weight
  let num_girls := total_students - num_boys
  girls_total_weight / num_girls

/-- Theorem stating that the average weight of girls is 27 kgs given the problem conditions -/
theorem girls_average_weight_is_27 : 
  average_weight_of_girls 25 15 48 45 = 27 := by
  sorry

end NUMINAMATH_CALUDE_girls_average_weight_is_27_l340_34050


namespace NUMINAMATH_CALUDE_f_is_convex_l340_34002

/-- The function f(x) = x^4 - 2x^3 + 36x^2 - x + 7 -/
def f (x : ℝ) : ℝ := x^4 - 2*x^3 + 36*x^2 - x + 7

/-- The second derivative of f(x) -/
def f'' (x : ℝ) : ℝ := 12*x^2 - 12*x + 72

theorem f_is_convex : ConvexOn ℝ Set.univ f := by
  sorry

end NUMINAMATH_CALUDE_f_is_convex_l340_34002
