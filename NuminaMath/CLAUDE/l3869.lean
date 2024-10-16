import Mathlib

namespace NUMINAMATH_CALUDE_smallest_odd_with_three_prime_factors_l3869_386964

-- Define a function to check if a number has exactly three distinct prime factors
def has_three_distinct_prime_factors (n : ℕ) : Prop :=
  ∃ p q r : ℕ, Nat.Prime p ∧ Nat.Prime q ∧ Nat.Prime r ∧
  p ≠ q ∧ p ≠ r ∧ q ≠ r ∧
  n = p * q * r

-- State the theorem
theorem smallest_odd_with_three_prime_factors :
  (∀ m : ℕ, m < 105 → m % 2 = 1 → ¬(has_three_distinct_prime_factors m)) ∧
  (105 % 2 = 1) ∧
  (has_three_distinct_prime_factors 105) :=
sorry

end NUMINAMATH_CALUDE_smallest_odd_with_three_prime_factors_l3869_386964


namespace NUMINAMATH_CALUDE_max_value_of_f_on_interval_l3869_386924

def f (x : ℝ) := 2 * x^2 + 4 * x - 1

theorem max_value_of_f_on_interval : 
  ∃ (c : ℝ), c ∈ Set.Icc (-2) 2 ∧ 
  (∀ x, x ∈ Set.Icc (-2) 2 → f x ≤ f c) ∧
  f c = 15 :=
sorry

end NUMINAMATH_CALUDE_max_value_of_f_on_interval_l3869_386924


namespace NUMINAMATH_CALUDE_rectangle_area_l3869_386967

theorem rectangle_area (a b : ℝ) (h : a^2 + b^2 - 8*a - 6*b + 25 = 0) : a * b = 12 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l3869_386967


namespace NUMINAMATH_CALUDE_max_value_inequality_l3869_386913

theorem max_value_inequality (x y : ℝ) : 
  (x + 3*y + 4) / Real.sqrt (x^2 + y^2 + x + 1) ≤ Real.sqrt 26 := by
  sorry

end NUMINAMATH_CALUDE_max_value_inequality_l3869_386913


namespace NUMINAMATH_CALUDE_original_jellybean_count_l3869_386902

-- Define the daily reduction rate
def daily_reduction_rate : ℝ := 0.8

-- Define the function that calculates the remaining quantity after n days
def remaining_after_days (initial : ℝ) (days : ℕ) : ℝ :=
  initial * (daily_reduction_rate ^ days)

-- State the theorem
theorem original_jellybean_count :
  ∃ (initial : ℝ), remaining_after_days initial 2 = 32 ∧ initial = 50 := by
  sorry

end NUMINAMATH_CALUDE_original_jellybean_count_l3869_386902


namespace NUMINAMATH_CALUDE_smallest_multiple_with_divisors_l3869_386974

/-- The number of positive integral divisors of n -/
def num_divisors (n : ℕ) : ℕ := sorry

/-- n is a multiple of m -/
def is_multiple (n m : ℕ) : Prop := ∃ k, n = m * k

theorem smallest_multiple_with_divisors :
  ∃ n : ℕ,
    is_multiple n 75 ∧
    num_divisors n = 36 ∧
    (∀ m : ℕ, is_multiple m 75 → num_divisors m = 36 → n ≤ m) ∧
    n / 75 = 162 := by
  sorry

end NUMINAMATH_CALUDE_smallest_multiple_with_divisors_l3869_386974


namespace NUMINAMATH_CALUDE_cube_surface_area_increase_l3869_386927

/-- Proves that a cube with edge length 7 cm, when cut into 1 cm cubes, results in a 600% increase in surface area. -/
theorem cube_surface_area_increase : 
  let original_edge_length : ℝ := 7
  let original_surface_area : ℝ := 6 * original_edge_length^2
  let new_surface_area : ℝ := 6 * original_edge_length^3
  new_surface_area = 7 * original_surface_area := by
  sorry

#check cube_surface_area_increase

end NUMINAMATH_CALUDE_cube_surface_area_increase_l3869_386927


namespace NUMINAMATH_CALUDE_largest_quotient_l3869_386968

def S : Set ℤ := {-36, -6, -4, 3, 7, 9}

def quotient (a b : ℤ) : ℚ := (a : ℚ) / (b : ℚ)

def valid_quotient (q : ℚ) : Prop :=
  ∃ (a b : ℤ), a ∈ S ∧ b ∈ S ∧ b ≠ 0 ∧ q = quotient a b

theorem largest_quotient :
  ∃ (max_q : ℚ), valid_quotient max_q ∧ 
  (∀ (q : ℚ), valid_quotient q → q ≤ max_q) ∧
  max_q = 9 := by sorry

end NUMINAMATH_CALUDE_largest_quotient_l3869_386968


namespace NUMINAMATH_CALUDE_limit_at_two_l3869_386943

/-- The limit of (3x^2 - 5x - 2) / (x - 2) as x approaches 2 is 7 -/
theorem limit_at_two :
  ∀ ε > 0, ∃ δ > 0, ∀ x : ℝ, x ≠ 2 →
    0 < |x - 2| ∧ |x - 2| < δ →
    |((3 * x^2 - 5 * x - 2) / (x - 2)) - 7| < ε :=
by sorry

end NUMINAMATH_CALUDE_limit_at_two_l3869_386943


namespace NUMINAMATH_CALUDE_smallest_abs_z_l3869_386945

theorem smallest_abs_z (z : ℂ) (h : Complex.abs (z - 8) + Complex.abs (z + 6 * Complex.I) = 17) :
  ∃ (w : ℂ), Complex.abs w ≤ Complex.abs z ∧ Complex.abs (w - 8) + Complex.abs (w + 6 * Complex.I) = 17 ∧ Complex.abs w = 48 / 17 :=
sorry

end NUMINAMATH_CALUDE_smallest_abs_z_l3869_386945


namespace NUMINAMATH_CALUDE_valid_speaking_orders_eq_1080_l3869_386918

-- Define the number of candidates
def num_candidates : ℕ := 8

-- Define the number of speakers to be selected
def num_speakers : ℕ := 4

-- Define a function to calculate the number of valid speaking orders
def valid_speaking_orders : ℕ :=
  -- Number of orders where only one of A or B participates
  (Nat.choose 2 1) * (Nat.choose 6 3) * (Nat.factorial 4) +
  -- Number of orders where both A and B participate with one person between them
  (Nat.choose 2 2) * (Nat.choose 6 2) * (Nat.choose 2 1) * (Nat.factorial 2) * (Nat.factorial 2)

-- Theorem stating that the number of valid speaking orders is 1080
theorem valid_speaking_orders_eq_1080 : valid_speaking_orders = 1080 := by
  sorry

end NUMINAMATH_CALUDE_valid_speaking_orders_eq_1080_l3869_386918


namespace NUMINAMATH_CALUDE_nearest_multiple_21_l3869_386977

theorem nearest_multiple_21 (x : ℤ) : 
  (∀ y : ℤ, y % 21 = 0 → |x - 2319| ≤ |x - y|) → x = 2318 :=
sorry

end NUMINAMATH_CALUDE_nearest_multiple_21_l3869_386977


namespace NUMINAMATH_CALUDE_scarlet_savings_l3869_386910

theorem scarlet_savings (initial_savings : ℕ) (earrings_cost : ℕ) (necklace_cost : ℕ) : 
  initial_savings = 80 → earrings_cost = 23 → necklace_cost = 48 → 
  initial_savings - (earrings_cost + necklace_cost) = 9 := by
sorry

end NUMINAMATH_CALUDE_scarlet_savings_l3869_386910


namespace NUMINAMATH_CALUDE_smallest_n_for_square_not_cube_l3869_386922

def is_perfect_square (x : ℕ) : Prop :=
  ∃ y : ℕ, y * y = x

def is_perfect_cube (x : ℕ) : Prop :=
  ∃ y : ℕ, y * y * y = x

def expression (n k : ℕ) : ℕ :=
  3^k + n^k + (3*n)^k + 2014^k

theorem smallest_n_for_square_not_cube :
  ∃ n : ℕ, n > 0 ∧
    (∀ k : ℕ, is_perfect_square (expression n k)) ∧
    (∀ k : ℕ, ¬ is_perfect_cube (expression n k)) ∧
    (∀ m : ℕ, m > 0 ∧ m < n →
      ¬(∀ k : ℕ, is_perfect_square (expression m k)) ∨
      ¬(∀ k : ℕ, ¬ is_perfect_cube (expression m k))) ∧
    n = 2 :=
  sorry

end NUMINAMATH_CALUDE_smallest_n_for_square_not_cube_l3869_386922


namespace NUMINAMATH_CALUDE_negation_equivalence_l3869_386998

-- Define a triangle
structure Triangle where
  angles : Fin 3 → ℝ
  sum_180 : angles 0 + angles 1 + angles 2 = 180
  all_positive : ∀ i, angles i > 0

-- Define what it means for an angle to be obtuse
def is_obtuse (angle : ℝ) : Prop := angle > 90

-- Define the proposition "A triangle has at most one obtuse angle"
def at_most_one_obtuse (t : Triangle) : Prop :=
  (is_obtuse (t.angles 0) → ¬is_obtuse (t.angles 1) ∧ ¬is_obtuse (t.angles 2)) ∧
  (is_obtuse (t.angles 1) → ¬is_obtuse (t.angles 0) ∧ ¬is_obtuse (t.angles 2)) ∧
  (is_obtuse (t.angles 2) → ¬is_obtuse (t.angles 0) ∧ ¬is_obtuse (t.angles 1))

-- Define the negation of the proposition
def negation_at_most_one_obtuse (t : Triangle) : Prop :=
  ¬(at_most_one_obtuse t)

-- Define the condition "There are at least two obtuse angles in the triangle"
def at_least_two_obtuse (t : Triangle) : Prop :=
  (is_obtuse (t.angles 0) ∧ is_obtuse (t.angles 1)) ∨
  (is_obtuse (t.angles 1) ∧ is_obtuse (t.angles 2)) ∨
  (is_obtuse (t.angles 0) ∧ is_obtuse (t.angles 2))

-- Theorem: The negation of "at most one obtuse angle" is equivalent to "at least two obtuse angles"
theorem negation_equivalence (t : Triangle) :
  negation_at_most_one_obtuse t ↔ at_least_two_obtuse t :=
sorry

end NUMINAMATH_CALUDE_negation_equivalence_l3869_386998


namespace NUMINAMATH_CALUDE_sum_1423_9_and_711_9_in_base3_l3869_386907

/-- Converts a number from base 9 to base 10 -/
def base9To10 (n : ℕ) : ℕ := sorry

/-- Converts a number from base 10 to base 3 -/
def base10To3 (n : ℕ) : ℕ := sorry

/-- The sum of 1423 in base 9 and 711 in base 9, converted to base 3 -/
def sumInBase3 : ℕ := base10To3 (base9To10 1423 + base9To10 711)

theorem sum_1423_9_and_711_9_in_base3 :
  sumInBase3 = 2001011 := by sorry

end NUMINAMATH_CALUDE_sum_1423_9_and_711_9_in_base3_l3869_386907


namespace NUMINAMATH_CALUDE_donation_to_second_home_l3869_386957

theorem donation_to_second_home 
  (total_donation : ℝ)
  (first_home : ℝ)
  (third_home : ℝ)
  (h1 : total_donation = 700)
  (h2 : first_home = 245)
  (h3 : third_home = 230) :
  total_donation - first_home - third_home = 225 :=
by sorry

end NUMINAMATH_CALUDE_donation_to_second_home_l3869_386957


namespace NUMINAMATH_CALUDE_meeting_percentage_is_37_5_l3869_386920

/-- Represents the duration of a workday in minutes -/
def workday_duration : ℕ := 8 * 60

/-- Represents the duration of the first meeting in minutes -/
def first_meeting_duration : ℕ := 60

/-- Represents the duration of the second meeting in minutes -/
def second_meeting_duration : ℕ := 2 * first_meeting_duration

/-- Represents the total duration of both meetings in minutes -/
def total_meeting_duration : ℕ := first_meeting_duration + second_meeting_duration

/-- Represents the percentage of the workday spent in meetings -/
def meeting_percentage : ℚ := (total_meeting_duration : ℚ) / (workday_duration : ℚ) * 100

theorem meeting_percentage_is_37_5 : meeting_percentage = 37.5 := by
  sorry

end NUMINAMATH_CALUDE_meeting_percentage_is_37_5_l3869_386920


namespace NUMINAMATH_CALUDE_dorothy_doughnut_price_l3869_386991

/-- Given Dorothy's doughnut business scenario, prove the selling price per doughnut. -/
theorem dorothy_doughnut_price 
  (ingredient_cost : ℚ) 
  (num_doughnuts : ℕ) 
  (profit : ℚ) 
  (h1 : ingredient_cost = 53)
  (h2 : num_doughnuts = 25)
  (h3 : profit = 22) :
  (ingredient_cost + profit) / num_doughnuts = 3 := by
  sorry

#eval (53 + 22) / 25

end NUMINAMATH_CALUDE_dorothy_doughnut_price_l3869_386991


namespace NUMINAMATH_CALUDE_mode_of_student_dishes_l3869_386905

def student_dishes : List ℕ := [3, 5, 4, 6, 3, 3, 4]

def mode (l : List ℕ) : ℕ :=
  l.foldl (fun acc x => if l.count x > l.count acc then x else acc) 0

theorem mode_of_student_dishes :
  mode student_dishes = 3 := by sorry

end NUMINAMATH_CALUDE_mode_of_student_dishes_l3869_386905


namespace NUMINAMATH_CALUDE_pyramid_levels_6_l3869_386989

/-- Defines the number of cubes in a pyramid with n levels -/
def pyramid_cubes (n : ℕ) : ℕ := n * (n + 1) * (2 * n + 1) / 6

/-- Theorem stating that a pyramid with 6 levels contains 225 cubes -/
theorem pyramid_levels_6 : pyramid_cubes 6 = 225 := by sorry

end NUMINAMATH_CALUDE_pyramid_levels_6_l3869_386989


namespace NUMINAMATH_CALUDE_geometric_series_sum_l3869_386997

theorem geometric_series_sum : 
  let a := 2  -- first term
  let r := 3  -- common ratio
  let n := 7  -- number of terms
  a * (r^n - 1) / (r - 1) = 2186 := by
sorry

end NUMINAMATH_CALUDE_geometric_series_sum_l3869_386997


namespace NUMINAMATH_CALUDE_company_car_replacement_l3869_386979

theorem company_car_replacement (x : ℕ) : 
  let initial_fleet := 20
  let retired_per_year := 5
  let years := 2
  let old_cars_after_two_years := initial_fleet - years * retired_per_year
  let new_cars_after_two_years := years * x
  let total_fleet_after_two_years := old_cars_after_two_years + new_cars_after_two_years
  (old_cars_after_two_years < total_fleet_after_two_years / 2) → x ≥ 6 :=
by sorry

end NUMINAMATH_CALUDE_company_car_replacement_l3869_386979


namespace NUMINAMATH_CALUDE_rectangle_area_problem_l3869_386925

theorem rectangle_area_problem (l w : ℚ) : 
  (l + 3) * (w - 1) = l * w ∧ (l - 3) * (w + 2) = l * w → l * w = -90 / 121 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_problem_l3869_386925


namespace NUMINAMATH_CALUDE_frog_jumps_theorem_l3869_386941

-- Define the jump sequences for each frog
def SmallFrogJumps : List Int := [2, 3]
def MediumFrogJumps : List Int := [2, 4]
def LargeFrogJumps : List Int := [6, 9]

-- Define the target rungs for each frog
def SmallFrogTarget : Int := 7
def MediumFrogTarget : Int := 1
def LargeFrogTarget : Int := 3

-- Function to check if a target can be reached using given jumps
def canReachTarget (jumps : List Int) (target : Int) : Prop :=
  ∃ (sequence : List Int), 
    (∀ x ∈ sequence, x ∈ jumps ∨ -x ∈ jumps) ∧ 
    sequence.sum = target

theorem frog_jumps_theorem :
  (canReachTarget SmallFrogJumps SmallFrogTarget) ∧
  ¬(canReachTarget MediumFrogJumps MediumFrogTarget) ∧
  (canReachTarget LargeFrogJumps LargeFrogTarget) := by
  sorry


end NUMINAMATH_CALUDE_frog_jumps_theorem_l3869_386941


namespace NUMINAMATH_CALUDE_curve_E_equation_line_l_equation_l3869_386966

/-- The curve E is defined by the constant sum of distances to two fixed points -/
def CurveE (P : ℝ × ℝ) : Prop :=
  let F₁ : ℝ × ℝ := (-Real.sqrt 3, 0)
  let F₂ : ℝ × ℝ := (Real.sqrt 3, 0)
  Real.sqrt ((P.1 - F₁.1)^2 + (P.2 - F₁.2)^2) +
  Real.sqrt ((P.1 - F₂.1)^2 + (P.2 - F₂.2)^2) = 4

/-- The line l passes through (0, -2) and intersects curve E at points C and D -/
def LineL (l : ℝ → ℝ) (C D : ℝ × ℝ) : Prop :=
  l 0 = -2 ∧ CurveE C ∧ CurveE D ∧ C.2 = l C.1 ∧ D.2 = l D.1

/-- The dot product of OC and OD is zero -/
def OrthogonalIntersection (C D : ℝ × ℝ) : Prop :=
  C.1 * D.1 + C.2 * D.2 = 0

theorem curve_E_equation (P : ℝ × ℝ) (h : CurveE P) :
  P.1^2 / 4 + P.2^2 = 1 :=
sorry

theorem line_l_equation (l : ℝ → ℝ) (C D : ℝ × ℝ)
  (hl : LineL l C D) (horth : OrthogonalIntersection C D) :
  (∀ x, l x = 2*x - 2) ∨ (∀ x, l x = -2*x - 2) :=
sorry

end NUMINAMATH_CALUDE_curve_E_equation_line_l_equation_l3869_386966


namespace NUMINAMATH_CALUDE_weekly_expenditure_is_3500_l3869_386915

/-- Represents the daily expenditures for a week -/
structure WeeklyExpenditure where
  mon : ℕ
  tue : ℕ
  wed : ℕ
  thu : ℕ
  fri : ℕ
  sat : ℕ
  sun : ℕ

/-- Calculates the total expenditure for the week -/
def totalExpenditure (w : WeeklyExpenditure) : ℕ :=
  w.mon + w.tue + w.wed + w.thu + w.fri + w.sat + w.sun

/-- Represents the items purchased on Friday -/
structure FridayPurchases where
  earphone : ℕ
  pen : ℕ
  notebook : ℕ

/-- Calculates the total cost of Friday's purchases -/
def fridayTotal (f : FridayPurchases) : ℕ :=
  f.earphone + f.pen + f.notebook

/-- Theorem stating that the total weekly expenditure is 3500 -/
theorem weekly_expenditure_is_3500 
  (w : WeeklyExpenditure)
  (f : FridayPurchases)
  (h1 : w.mon = 450)
  (h2 : w.tue = 600)
  (h3 : w.wed = 400)
  (h4 : w.thu = 500)
  (h5 : w.sat = 550)
  (h6 : w.sun = 300)
  (h7 : f.earphone = 620)
  (h8 : f.pen = 30)
  (h9 : f.notebook = 50)
  (h10 : w.fri = fridayTotal f) :
  totalExpenditure w = 3500 := by
  sorry

#check weekly_expenditure_is_3500

end NUMINAMATH_CALUDE_weekly_expenditure_is_3500_l3869_386915


namespace NUMINAMATH_CALUDE_min_value_theorem_l3869_386939

theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 2) :
  1/a + 2/b ≥ (3 + 2 * Real.sqrt 2) / 2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_l3869_386939


namespace NUMINAMATH_CALUDE_opposite_sides_range_l3869_386926

def line_equation (x y a : ℝ) : ℝ := x - 2*y + a

theorem opposite_sides_range (a : ℝ) : 
  (∃ (x₁ y₁ x₂ y₂ : ℝ), 
    x₁ = 2 ∧ y₁ = -1 ∧ x₂ = -3 ∧ y₂ = 2 ∧ 
    (line_equation x₁ y₁ a) * (line_equation x₂ y₂ a) < 0) ↔ 
  -4 < a ∧ a < 7 :=
sorry

end NUMINAMATH_CALUDE_opposite_sides_range_l3869_386926


namespace NUMINAMATH_CALUDE_floor_of_4_7_l3869_386962

theorem floor_of_4_7 : ⌊(4.7 : ℝ)⌋ = 4 := by sorry

end NUMINAMATH_CALUDE_floor_of_4_7_l3869_386962


namespace NUMINAMATH_CALUDE_equation_one_solution_equation_two_solution_l3869_386973

-- Equation 1
theorem equation_one_solution :
  ∃ x : ℝ, (x ≠ 0 ∧ x ≠ 1) ∧ (9 / x = 8 / (x - 1)) → x = 9 :=
sorry

-- Equation 2
theorem equation_two_solution :
  ∃ x : ℝ, (x ≠ 2) ∧ (1 / (x - 2) - 3 = (x - 1) / (2 - x)) → x = 3 :=
sorry

end NUMINAMATH_CALUDE_equation_one_solution_equation_two_solution_l3869_386973


namespace NUMINAMATH_CALUDE_triangle_area_l3869_386995

/-- The area of a triangle with vertices at (2,-3), (-4,2), and (3,-7) is 19/2 -/
theorem triangle_area : 
  let A : ℝ × ℝ := (2, -3)
  let B : ℝ × ℝ := (-4, 2)
  let C : ℝ × ℝ := (3, -7)
  let area := abs ((C.1 - A.1) * (B.2 - A.2) - (C.2 - A.2) * (B.1 - A.1)) / 2
  area = 19 / 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l3869_386995


namespace NUMINAMATH_CALUDE_M_equals_N_l3869_386996

/-- The set M of integers of the form 12m + 8n + 4l where m, n, l are integers -/
def M : Set ℤ := {u | ∃ m n l : ℤ, u = 12*m + 8*n + 4*l}

/-- The set N of integers of the form 20p + 16q + 12r where p, q, r are integers -/
def N : Set ℤ := {u | ∃ p q r : ℤ, u = 20*p + 16*q + 12*r}

/-- Theorem stating that M equals N -/
theorem M_equals_N : M = N := by
  sorry

end NUMINAMATH_CALUDE_M_equals_N_l3869_386996


namespace NUMINAMATH_CALUDE_circle_area_diameter_13_l3869_386959

/-- The area of a circle with diameter 13 meters is π * (13/2)^2 square meters. -/
theorem circle_area_diameter_13 :
  let diameter : ℝ := 13
  let radius : ℝ := diameter / 2
  let area : ℝ := π * radius ^ 2
  area = π * (13 / 2) ^ 2 := by sorry

end NUMINAMATH_CALUDE_circle_area_diameter_13_l3869_386959


namespace NUMINAMATH_CALUDE_seventh_term_largest_implies_n_l3869_386921

/-- The binomial coefficient -/
def binomial_coefficient (n k : ℕ) : ℕ := sorry

/-- Predicate to check if the 7th term has the largest binomial coefficient -/
def seventh_term_largest (n : ℕ) : Prop :=
  ∀ k, k ≠ 6 → binomial_coefficient n 6 ≥ binomial_coefficient n k

/-- Theorem stating the possible values of n when the 7th term has the largest binomial coefficient -/
theorem seventh_term_largest_implies_n (n : ℕ) :
  seventh_term_largest n → n = 11 ∨ n = 12 ∨ n = 13 := by sorry

end NUMINAMATH_CALUDE_seventh_term_largest_implies_n_l3869_386921


namespace NUMINAMATH_CALUDE_maggies_share_l3869_386935

def total_sum : ℝ := 6000
def debby_percentage : ℝ := 0.25

theorem maggies_share :
  let debby_share := debby_percentage * total_sum
  let maggie_share := total_sum - debby_share
  maggie_share = 4500 := by sorry

end NUMINAMATH_CALUDE_maggies_share_l3869_386935


namespace NUMINAMATH_CALUDE_cos_shift_l3869_386988

theorem cos_shift (x : ℝ) : 
  Real.cos (1/2 * x + π/3) = Real.cos (1/2 * (x + 2*π/3)) := by
  sorry

end NUMINAMATH_CALUDE_cos_shift_l3869_386988


namespace NUMINAMATH_CALUDE_rectangular_sheet_area_l3869_386975

theorem rectangular_sheet_area (area1 area2 : ℝ) : 
  area1 = 4 * area2 →  -- First part is four times larger than the second
  area1 - area2 = 2208 →  -- First part is 2208 cm² larger than the second
  area1 + area2 = 3680 :=  -- Total area of the sheet
by sorry

end NUMINAMATH_CALUDE_rectangular_sheet_area_l3869_386975


namespace NUMINAMATH_CALUDE_sum_of_sequences_l3869_386954

def arithmetic_sum (a₁ : ℕ) (d : ℕ) (n : ℕ) : ℕ :=
  n * (2 * a₁ + (n - 1) * d) / 2

theorem sum_of_sequences : 
  (arithmetic_sum 2 10 5) + (arithmetic_sum 10 10 5) = 260 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_sequences_l3869_386954


namespace NUMINAMATH_CALUDE_number_problem_l3869_386985

theorem number_problem (x : ℚ) : (3 * x / 2) + 6 = 11 → x = 10/3 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l3869_386985


namespace NUMINAMATH_CALUDE_intersection_slope_of_circles_l3869_386972

-- Define the equations of the two circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 - 6*x + 4*y - 20 = 0
def circle2 (x y : ℝ) : Prop := x^2 + y^2 - 2*x - 6*y + 10 = 0

-- Define the slope of the line passing through the intersection points
def intersection_slope : ℝ := 0.4

-- Theorem statement
theorem intersection_slope_of_circles :
  ∀ x y : ℝ, circle1 x y ∧ circle2 x y → 
  ∃ m b : ℝ, m = intersection_slope ∧ y = m * x + b :=
by sorry

end NUMINAMATH_CALUDE_intersection_slope_of_circles_l3869_386972


namespace NUMINAMATH_CALUDE_circle_equation_proof_l3869_386912

/-- Represents a point in 2D space -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- Represents the equation of a circle -/
structure CircleEquation where
  D : ℝ
  E : ℝ
  F : ℝ

/-- Checks if a point satisfies the circle equation -/
def satisfiesCircleEquation (p : Point2D) (c : CircleEquation) : Prop :=
  p.x^2 + p.y^2 + c.D * p.x + c.E * p.y + c.F = 0

/-- Theorem: The equation x^2 + y^2 - 4x - 6y = 0 represents a circle passing through (0,0), (4,0), and (-1,1) -/
theorem circle_equation_proof :
  let c : CircleEquation := ⟨-4, -6, 0⟩
  satisfiesCircleEquation ⟨0, 0⟩ c ∧
  satisfiesCircleEquation ⟨4, 0⟩ c ∧
  satisfiesCircleEquation ⟨-1, 1⟩ c :=
by sorry

end NUMINAMATH_CALUDE_circle_equation_proof_l3869_386912


namespace NUMINAMATH_CALUDE_cubic_equation_solutions_l3869_386970

def is_solution (x y z : ℤ) : Prop :=
  x^3 + y^3 + z^3 - 3*x*y*z = 2003

theorem cubic_equation_solutions :
  ∀ x y z : ℤ, is_solution x y z ↔ 
    ((x = 668 ∧ y = 668 ∧ z = 667) ∨
     (x = 668 ∧ y = 667 ∧ z = 668) ∨
     (x = 667 ∧ y = 668 ∧ z = 668)) :=
by sorry

end NUMINAMATH_CALUDE_cubic_equation_solutions_l3869_386970


namespace NUMINAMATH_CALUDE_ball_max_height_l3869_386923

-- Define the function representing the ball's height
def f (t : ℝ) : ℝ := -20 * t^2 + 40 * t + 20

-- Theorem stating that the maximum height is 40 feet
theorem ball_max_height :
  ∃ (max : ℝ), max = 40 ∧ ∀ (t : ℝ), f t ≤ max :=
sorry

end NUMINAMATH_CALUDE_ball_max_height_l3869_386923


namespace NUMINAMATH_CALUDE_no_solution_condition_l3869_386914

theorem no_solution_condition (m : ℚ) : 
  (∀ x : ℚ, x ≠ 5 ∧ x ≠ -5 → 1 / (x - 5) + m / (x + 5) ≠ (m + 5) / (x^2 - 25)) ↔ 
  m = -1 ∨ m = 5 ∨ m = -5/11 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_condition_l3869_386914


namespace NUMINAMATH_CALUDE_triangle_properties_l3869_386986

/-- Given a triangle ABC with side lengths a, b, c opposite to angles A, B, C respectively. -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The perimeter of the triangle is √2 + 1 -/
def perimeter_condition (t : Triangle) : Prop :=
  t.a + t.b + t.c = Real.sqrt 2 + 1

/-- The sum of sines condition -/
def sine_sum_condition (t : Triangle) : Prop :=
  Real.sin t.A + Real.sin t.B = Real.sqrt 2 * Real.sin t.C

/-- The area of the triangle is (1/6) * sin C -/
def area_condition (t : Triangle) : Prop :=
  (1/2) * t.a * t.b * Real.sin t.C = (1/6) * Real.sin t.C

theorem triangle_properties (t : Triangle) 
  (h_perimeter : perimeter_condition t)
  (h_sine_sum : sine_sum_condition t)
  (h_area : area_condition t) :
  t.c = 1 ∧ t.C = π/3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l3869_386986


namespace NUMINAMATH_CALUDE_water_jars_problem_l3869_386947

theorem water_jars_problem (C1 C2 C3 W : ℚ) : 
  W > 0 ∧ C1 > 0 ∧ C2 > 0 ∧ C3 > 0 →
  W = (1/7) * C1 ∧ W = (2/9) * C2 ∧ W = (3/11) * C3 →
  C3 ≥ C1 ∧ C3 ≥ C2 →
  (3 * W) / C3 = 9/11 := by
sorry


end NUMINAMATH_CALUDE_water_jars_problem_l3869_386947


namespace NUMINAMATH_CALUDE_kendra_shirts_theorem_l3869_386953

/-- Represents the number of shirts Kendra needs for two weeks -/
def shirts_needed : ℕ :=
  let school_days := 5
  let club_days := 3
  let saturday_shirts := 1
  let sunday_shirts := 2
  let weeks := 2
  (school_days + club_days + saturday_shirts + sunday_shirts) * weeks

/-- Theorem stating that Kendra needs 22 shirts to do laundry once every two weeks -/
theorem kendra_shirts_theorem : shirts_needed = 22 := by
  sorry

end NUMINAMATH_CALUDE_kendra_shirts_theorem_l3869_386953


namespace NUMINAMATH_CALUDE_a_6_equals_one_half_l3869_386994

def a (n : ℕ+) : ℚ := (3 * n.val - 2) / (2 ^ (n.val - 1))

theorem a_6_equals_one_half : a 6 = 1 / 2 := by sorry

end NUMINAMATH_CALUDE_a_6_equals_one_half_l3869_386994


namespace NUMINAMATH_CALUDE_line_circle_intersection_l3869_386956

theorem line_circle_intersection (k : ℝ) : 
  ∃ (x y : ℝ), y = k * x + 1 ∧ x^2 + y^2 = 4 := by
  sorry

end NUMINAMATH_CALUDE_line_circle_intersection_l3869_386956


namespace NUMINAMATH_CALUDE_sqrt_simplification_l3869_386931

theorem sqrt_simplification (a : ℝ) (ha : a ≥ 0) :
  Real.sqrt (a^(1/2) * Real.sqrt (a^(1/2) * Real.sqrt a)) = a^(1/2) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_simplification_l3869_386931


namespace NUMINAMATH_CALUDE_geometric_series_first_term_l3869_386952

theorem geometric_series_first_term (a r : ℝ) (h1 : |r| < 1) 
  (h2 : a / (1 - r) = 30) (h3 : a^2 / (1 - r^2) = 90) : a = 60 / 11 := by
  sorry

end NUMINAMATH_CALUDE_geometric_series_first_term_l3869_386952


namespace NUMINAMATH_CALUDE_salary_increase_l3869_386965

/-- Represents the regression line for a worker's monthly salary based on labor productivity -/
def regression_line (x : ℝ) : ℝ := 60 + 90 * x

/-- Theorem stating that an increase of 1 unit in labor productivity results in a 90 yuan increase in salary -/
theorem salary_increase (x : ℝ) : 
  regression_line (x + 1) - regression_line x = 90 := by
  sorry

end NUMINAMATH_CALUDE_salary_increase_l3869_386965


namespace NUMINAMATH_CALUDE_product_of_y_coordinates_l3869_386942

/-- Given a point P on the line x = -3 that is 10 units from (5, 2),
    the product of all possible y-coordinates of P is -32. -/
theorem product_of_y_coordinates : ∀ y₁ y₂ : ℝ,
  ((-3 - 5)^2 + (y₁ - 2)^2 = 10^2) →
  ((-3 - 5)^2 + (y₂ - 2)^2 = 10^2) →
  y₁ ≠ y₂ →
  y₁ * y₂ = -32 := by
  sorry

end NUMINAMATH_CALUDE_product_of_y_coordinates_l3869_386942


namespace NUMINAMATH_CALUDE_least_divisible_by_first_ten_l3869_386919

def first_ten_integers : Finset ℕ := Finset.range 10

theorem least_divisible_by_first_ten : ∃ n : ℕ, n > 0 ∧ (∀ i ∈ first_ten_integers, i ∣ n) ∧ 
  (∀ m : ℕ, m > 0 ∧ (∀ i ∈ first_ten_integers, i ∣ m) → n ≤ m) ∧ n = 2520 :=
sorry

end NUMINAMATH_CALUDE_least_divisible_by_first_ten_l3869_386919


namespace NUMINAMATH_CALUDE_point_transformation_l3869_386960

/-- Rotate a point (x, y) counterclockwise by 90° around (h, k) -/
def rotate90 (x y h k : ℝ) : ℝ × ℝ :=
  (h - (y - k), k + (x - h))

/-- Reflect a point (x, y) about the line y = x -/
def reflectAboutYEqX (x y : ℝ) : ℝ × ℝ :=
  (y, x)

theorem point_transformation (c d : ℝ) :
  let (x₁, y₁) := rotate90 c d 2 3
  let (x₂, y₂) := reflectAboutYEqX x₁ y₁
  (x₂ = -3 ∧ y₂ = 8) → d - c = 1 := by
  sorry

end NUMINAMATH_CALUDE_point_transformation_l3869_386960


namespace NUMINAMATH_CALUDE_horse_distribution_l3869_386949

theorem horse_distribution (total_horses : ℕ) (son1_horses son2_horses son3_horses : ℕ) :
  total_horses = 17 ∧
  son1_horses = 9 ∧
  son2_horses = 6 ∧
  son3_horses = 2 →
  son1_horses / total_horses = 1/2 ∧
  son2_horses / total_horses = 1/3 ∧
  son3_horses / total_horses = 1/9 ∧
  son1_horses + son2_horses + son3_horses = total_horses :=
by
  sorry

#check horse_distribution

end NUMINAMATH_CALUDE_horse_distribution_l3869_386949


namespace NUMINAMATH_CALUDE_new_combined_total_capacity_l3869_386951

/-- Represents a weightlifter's lifting capacities -/
structure Lifter where
  cleanAndJerk : ℝ
  snatch : ℝ

/-- Represents the improvement rates for a lifter -/
structure Improvement where
  cleanAndJerkRate : ℝ
  snatchRate : ℝ

/-- Calculates the new lifting capacities after improvement -/
def improve (lifter : Lifter) (imp : Improvement) : Lifter where
  cleanAndJerk := lifter.cleanAndJerk * (1 + imp.cleanAndJerkRate)
  snatch := lifter.snatch * (1 + imp.snatchRate)

/-- Calculates the total lifting capacity of a lifter -/
def totalCapacity (lifter : Lifter) : ℝ :=
  lifter.cleanAndJerk + lifter.snatch

/-- The main theorem to prove -/
theorem new_combined_total_capacity
  (john : Lifter)
  (alice : Lifter)
  (mark : Lifter)
  (johnImp : Improvement)
  (aliceImp : Improvement)
  (markImp : Improvement)
  (h1 : john.cleanAndJerk = 80)
  (h2 : john.snatch = 50)
  (h3 : alice.cleanAndJerk = 90)
  (h4 : alice.snatch = 55)
  (h5 : mark.cleanAndJerk = 100)
  (h6 : mark.snatch = 65)
  (h7 : johnImp.cleanAndJerkRate = 1)  -- doubled means 100% increase
  (h8 : johnImp.snatchRate = 0.8)
  (h9 : aliceImp.cleanAndJerkRate = 0.5)
  (h10 : aliceImp.snatchRate = 0.9)
  (h11 : markImp.cleanAndJerkRate = 0.75)
  (h12 : markImp.snatchRate = 0.7)
  : totalCapacity (improve john johnImp) +
    totalCapacity (improve alice aliceImp) +
    totalCapacity (improve mark markImp) = 775 := by
  sorry

end NUMINAMATH_CALUDE_new_combined_total_capacity_l3869_386951


namespace NUMINAMATH_CALUDE_quadratic_roots_l3869_386990

theorem quadratic_roots (a b c : ℝ) (h : b^2 - 4*a*c > 0) :
  let f (x : ℝ) := 3*a*x^2 + 2*(a + b)*x + (b + c)
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ f x₁ = 0 ∧ f x₂ = 0 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_l3869_386990


namespace NUMINAMATH_CALUDE_woodburning_profit_l3869_386933

/-- Calculate the profit from selling woodburnings -/
theorem woodburning_profit (num_sold : ℕ) (price_per_item : ℕ) (wood_cost : ℕ) :
  num_sold = 20 →
  price_per_item = 15 →
  wood_cost = 100 →
  num_sold * price_per_item - wood_cost = 200 := by
sorry

end NUMINAMATH_CALUDE_woodburning_profit_l3869_386933


namespace NUMINAMATH_CALUDE_geometric_sequence_sum_implies_a_eq_neg_one_l3869_386911

/-- A geometric sequence with sum of first n terms given by Sn = 4^n + a -/
def GeometricSequence (a : ℝ) := ℕ → ℝ

/-- Sum of first n terms of the geometric sequence -/
def SumFirstNTerms (seq : GeometricSequence a) (n : ℕ) : ℝ := 4^n + a

/-- The ratio between consecutive terms in a geometric sequence is constant -/
def IsGeometric (seq : GeometricSequence a) : Prop :=
  ∀ n : ℕ, seq (n + 2) / seq (n + 1) = seq (n + 1) / seq n

theorem geometric_sequence_sum_implies_a_eq_neg_one :
  ∀ a : ℝ, ∃ seq : GeometricSequence a,
    (∀ n : ℕ, SumFirstNTerms seq n = 4^n + a) →
    IsGeometric seq →
    a = -1 :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_sum_implies_a_eq_neg_one_l3869_386911


namespace NUMINAMATH_CALUDE_edward_initial_amount_l3869_386944

def initial_amount (books_cost pens_cost remaining : ℕ) : ℕ :=
  books_cost + pens_cost + remaining

theorem edward_initial_amount :
  initial_amount 6 16 19 = 41 :=
by sorry

end NUMINAMATH_CALUDE_edward_initial_amount_l3869_386944


namespace NUMINAMATH_CALUDE_min_abs_value_plus_constant_l3869_386999

theorem min_abs_value_plus_constant (x : ℝ) :
  ∀ y : ℝ, |x - 2| + 2023 ≤ |y - 2| + 2023 ↔ x = 2 := by
  sorry

end NUMINAMATH_CALUDE_min_abs_value_plus_constant_l3869_386999


namespace NUMINAMATH_CALUDE_optimal_transport_plan_l3869_386982

/-- Represents the transportation problem for fruits A, B, and C -/
structure FruitTransport where
  total_trucks : ℕ
  total_tons : ℕ
  tons_per_truck_A : ℕ
  tons_per_truck_B : ℕ
  tons_per_truck_C : ℕ
  profit_per_ton_A : ℕ
  profit_per_ton_B : ℕ
  profit_per_ton_C : ℕ

/-- Calculates the profit for a given transportation plan -/
def calculate_profit (ft : FruitTransport) (trucks_A trucks_B trucks_C : ℕ) : ℕ :=
  trucks_A * ft.tons_per_truck_A * ft.profit_per_ton_A +
  trucks_B * ft.tons_per_truck_B * ft.profit_per_ton_B +
  trucks_C * ft.tons_per_truck_C * ft.profit_per_ton_C

/-- The main theorem stating the optimal transportation plan and maximum profit -/
theorem optimal_transport_plan (ft : FruitTransport)
  (h1 : ft.total_trucks = 20)
  (h2 : ft.total_tons = 100)
  (h3 : ft.tons_per_truck_A = 6)
  (h4 : ft.tons_per_truck_B = 5)
  (h5 : ft.tons_per_truck_C = 4)
  (h6 : ft.profit_per_ton_A = 500)
  (h7 : ft.profit_per_ton_B = 600)
  (h8 : ft.profit_per_ton_C = 400) :
  ∃ (trucks_A trucks_B trucks_C : ℕ),
    trucks_A + trucks_B + trucks_C = ft.total_trucks ∧
    trucks_A * ft.tons_per_truck_A + trucks_B * ft.tons_per_truck_B + trucks_C * ft.tons_per_truck_C = ft.total_tons ∧
    trucks_A ≥ 2 ∧ trucks_B ≥ 2 ∧ trucks_C ≥ 2 ∧
    trucks_A = 2 ∧ trucks_B = 16 ∧ trucks_C = 2 ∧
    calculate_profit ft trucks_A trucks_B trucks_C = 57200 ∧
    ∀ (a b c : ℕ), a + b + c = ft.total_trucks →
      a * ft.tons_per_truck_A + b * ft.tons_per_truck_B + c * ft.tons_per_truck_C = ft.total_tons →
      a ≥ 2 → b ≥ 2 → c ≥ 2 →
      calculate_profit ft a b c ≤ calculate_profit ft trucks_A trucks_B trucks_C :=
by sorry

end NUMINAMATH_CALUDE_optimal_transport_plan_l3869_386982


namespace NUMINAMATH_CALUDE_coefficient_x3y3_equals_15_l3869_386901

/-- The coefficient of x^3 * y^3 in the expansion of (x + y^2/x)(x + y)^5 -/
def coefficient_x3y3 (x y : ℝ) : ℝ :=
  let expanded := (x + y^2/x) * (x + y)^5
  sorry

theorem coefficient_x3y3_equals_15 :
  ∀ x y, x ≠ 0 → coefficient_x3y3 x y = 15 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_x3y3_equals_15_l3869_386901


namespace NUMINAMATH_CALUDE_jake_final_balance_l3869_386928

/-- Represents Jake's bitcoin transactions and calculates his final balance --/
def jake_bitcoin_balance (initial_fortune : ℚ) (investment : ℚ) (first_donation : ℚ) 
  (brother_return : ℚ) (second_donation : ℚ) : ℚ :=
  let after_investment := initial_fortune - investment
  let after_first_donation := after_investment - (first_donation / 2)
  let after_giving_to_brother := after_first_donation / 2
  let after_brother_return := after_giving_to_brother + brother_return
  let after_quadrupling := after_brother_return * 4
  after_quadrupling - (second_donation * 4)

/-- Theorem stating that Jake ends up with 95 bitcoins --/
theorem jake_final_balance : 
  jake_bitcoin_balance 120 40 25 5 15 = 95 := by
  sorry

end NUMINAMATH_CALUDE_jake_final_balance_l3869_386928


namespace NUMINAMATH_CALUDE_jason_bookcase_weight_difference_l3869_386930

/-- Represents the bookcase and Jason's collection of items -/
structure Bookcase :=
  (shelves : Nat)
  (shelf_weight_limit : Nat)
  (hardcover_books : Nat)
  (textbooks : Nat)
  (knick_knacks : Nat)
  (max_hardcover_weight : Real)
  (max_textbook_weight : Real)
  (max_knick_knack_weight : Real)

/-- Calculates the maximum weight of the collection minus the bookcase's weight limit -/
def weight_difference (b : Bookcase) : Real :=
  b.hardcover_books * b.max_hardcover_weight +
  b.textbooks * b.max_textbook_weight +
  b.knick_knacks * b.max_knick_knack_weight -
  b.shelves * b.shelf_weight_limit

/-- Theorem stating that the weight difference for Jason's collection is 195 pounds -/
theorem jason_bookcase_weight_difference :
  ∃ (b : Bookcase),
    b.shelves = 4 ∧
    b.shelf_weight_limit = 20 ∧
    b.hardcover_books = 70 ∧
    b.textbooks = 30 ∧
    b.knick_knacks = 10 ∧
    b.max_hardcover_weight = 1.5 ∧
    b.max_textbook_weight = 3 ∧
    b.max_knick_knack_weight = 8 ∧
    weight_difference b = 195 :=
  sorry

end NUMINAMATH_CALUDE_jason_bookcase_weight_difference_l3869_386930


namespace NUMINAMATH_CALUDE_prob_different_cities_l3869_386978

/-- The probability that student A attends university in city A -/
def prob_A_cityA : ℝ := 0.6

/-- The probability that student B attends university in city A -/
def prob_B_cityA : ℝ := 0.3

/-- The theorem stating that the probability of A and B not attending university 
    in the same city is 0.54, given the probabilities of each student 
    attending city A -/
theorem prob_different_cities (h1 : 0 ≤ prob_A_cityA ∧ prob_A_cityA ≤ 1) 
                               (h2 : 0 ≤ prob_B_cityA ∧ prob_B_cityA ≤ 1) : 
  prob_A_cityA * (1 - prob_B_cityA) + (1 - prob_A_cityA) * prob_B_cityA = 0.54 := by
  sorry

end NUMINAMATH_CALUDE_prob_different_cities_l3869_386978


namespace NUMINAMATH_CALUDE_unique_intersection_values_l3869_386955

-- Define the complex plane
variable (z : ℂ)

-- Define the condition from the original problem
def intersection_condition (k : ℝ) : Prop :=
  ∃! z : ℂ, Complex.abs (z - 4) = 3 * Complex.abs (z + 4) ∧ Complex.abs z = k

-- State the theorem
theorem unique_intersection_values :
  ∀ k : ℝ, intersection_condition k ↔ (k = 13 - Real.sqrt 153 ∨ k = 13 + Real.sqrt 153) :=
sorry

end NUMINAMATH_CALUDE_unique_intersection_values_l3869_386955


namespace NUMINAMATH_CALUDE_rectangle_dimensions_l3869_386946

theorem rectangle_dimensions (x : ℝ) : 
  (2*x - 3) * (3*x + 4) = 20*x - 12 → x = 7/2 := by
sorry

end NUMINAMATH_CALUDE_rectangle_dimensions_l3869_386946


namespace NUMINAMATH_CALUDE_three_greater_than_negative_four_l3869_386969

theorem three_greater_than_negative_four : 3 > -4 := by
  sorry

end NUMINAMATH_CALUDE_three_greater_than_negative_four_l3869_386969


namespace NUMINAMATH_CALUDE_ones_digit_of_8_to_47_l3869_386940

theorem ones_digit_of_8_to_47 : (8^47 : ℕ) % 10 = 2 := by sorry

end NUMINAMATH_CALUDE_ones_digit_of_8_to_47_l3869_386940


namespace NUMINAMATH_CALUDE_points_on_line_l3869_386937

/-- Given points M(a, 1/b) and N(b, 1/c) on the line x + y = 1,
    prove that points P(c, 1/a) and Q(1/c, b) are also on the same line. -/
theorem points_on_line (a b c : ℝ) (ha : a + 1/b = 1) (hb : b + 1/c = 1) :
  c + 1/a = 1 ∧ 1/c + b = 1 := by
  sorry

end NUMINAMATH_CALUDE_points_on_line_l3869_386937


namespace NUMINAMATH_CALUDE_decrease_amount_l3869_386916

theorem decrease_amount (x y : ℝ) : x = 50 → (1/5) * x - y = 5 → y = 5 := by
  sorry

end NUMINAMATH_CALUDE_decrease_amount_l3869_386916


namespace NUMINAMATH_CALUDE_unique_sequence_l3869_386963

/-- A strictly increasing sequence of natural numbers -/
def StrictlyIncreasingSeq (a : ℕ → ℕ) : Prop :=
  ∀ n m : ℕ, n < m → a n < a m

/-- The property that a₂ = 2 -/
def SecondTermIsTwo (a : ℕ → ℕ) : Prop :=
  a 2 = 2

/-- The property that aₙₘ = aₙ * aₘ for any natural numbers n and m -/
def MultiplicativeProperty (a : ℕ → ℕ) : Prop :=
  ∀ n m : ℕ, a (n * m) = a n * a m

/-- The theorem stating that the only sequence satisfying all conditions is aₙ = n -/
theorem unique_sequence :
  ∀ a : ℕ → ℕ,
    StrictlyIncreasingSeq a →
    SecondTermIsTwo a →
    MultiplicativeProperty a →
    ∀ n : ℕ, a n = n :=
by sorry

end NUMINAMATH_CALUDE_unique_sequence_l3869_386963


namespace NUMINAMATH_CALUDE_negation_of_universal_quantifier_l3869_386992

theorem negation_of_universal_quantifier :
  (¬ ∀ x : ℝ, x ≥ Real.sqrt 2 → x^2 ≥ 2) ↔ (∃ x : ℝ, x ≥ Real.sqrt 2 ∧ x^2 < 2) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_universal_quantifier_l3869_386992


namespace NUMINAMATH_CALUDE_parabola_translation_l3869_386980

/-- Given two parabolas, prove that one is a translation of the other -/
theorem parabola_translation (x : ℝ) :
  (x^2 + 4*x + 5) = ((x + 2)^2 + 1) := by sorry

end NUMINAMATH_CALUDE_parabola_translation_l3869_386980


namespace NUMINAMATH_CALUDE_inequality_solution_range_l3869_386938

theorem inequality_solution_range (a : ℝ) : 
  (∃ x : ℝ, x^2 - a*x - a ≤ -3) ↔ (a ≤ -6 ∨ a ≥ 2) := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_range_l3869_386938


namespace NUMINAMATH_CALUDE_horizontal_arrangement_possible_l3869_386993

/-- Represents a domino on the board -/
structure Domino where
  x : ℕ
  y : ℕ
  horizontal : Bool

/-- Represents the chessboard with an extra cell -/
structure Board where
  cells : ℕ
  dominoes : List Domino

/-- Checks if a given board configuration is valid -/
def is_valid_board (b : Board) : Prop :=
  b.cells = 65 ∧ b.dominoes.length = 32

/-- Checks if all dominoes on the board are horizontal -/
def all_horizontal (b : Board) : Prop :=
  b.dominoes.all (λ d => d.horizontal)

/-- Represents the ability to move dominoes on the board -/
def can_move_domino (b : Board) : Prop :=
  ∀ d : Domino, ∃ d' : Domino, d' ∈ b.dominoes

/-- Main theorem: It's possible to arrange all dominoes horizontally -/
theorem horizontal_arrangement_possible (b : Board) 
  (h_valid : is_valid_board b) (h_move : can_move_domino b) : 
  ∃ b' : Board, is_valid_board b' ∧ all_horizontal b' :=
sorry

end NUMINAMATH_CALUDE_horizontal_arrangement_possible_l3869_386993


namespace NUMINAMATH_CALUDE_park_expansion_area_ratio_l3869_386903

theorem park_expansion_area_ratio :
  ∀ s : ℝ, s > 0 →
  (s^2) / ((3*s)^2) = 1/9 := by
sorry

end NUMINAMATH_CALUDE_park_expansion_area_ratio_l3869_386903


namespace NUMINAMATH_CALUDE_cory_candy_packs_l3869_386932

/-- The number of packs of candies Cory wants to buy -/
def num_packs : ℕ := sorry

/-- The amount of money Cory has -/
def cory_money : ℚ := 20

/-- The cost of each pack of candies -/
def pack_cost : ℚ := 49

/-- The additional amount Cory needs -/
def additional_money : ℚ := 78

theorem cory_candy_packs : num_packs = 2 := by sorry

end NUMINAMATH_CALUDE_cory_candy_packs_l3869_386932


namespace NUMINAMATH_CALUDE_expected_same_color_edges_l3869_386971

/-- Represents a 3x3 board -/
def Board := Fin 3 → Fin 3 → Bool

/-- The number of squares in the board -/
def boardSize : Nat := 9

/-- The number of squares to be blackened -/
def blackSquares : Nat := 5

/-- The total number of pairs of adjacent squares -/
def totalAdjacentPairs : Nat := 12

/-- Calculates the probability that two adjacent squares have the same color -/
noncomputable def probSameColor : ℚ := 4 / 9

/-- Theorem: The expected number of edges between two squares of the same color 
    in a 3x3 board with 5 randomly blackened squares is 16/3 -/
theorem expected_same_color_edges :
  (totalAdjacentPairs : ℚ) * probSameColor = 16 / 3 := by sorry

end NUMINAMATH_CALUDE_expected_same_color_edges_l3869_386971


namespace NUMINAMATH_CALUDE_less_crowded_detector_time_is_ten_l3869_386950

/-- Represents the time Mark spends on courthouse activities in a week -/
structure CourthouseTime where
  workDays : ℕ
  parkingTime : ℕ
  walkingTime : ℕ
  crowdedDetectorDays : ℕ
  crowdedDetectorTime : ℕ
  totalWeeklyTime : ℕ

/-- Calculates the time it takes to get through the metal detector on less crowded days -/
def lessCrowdedDetectorTime (ct : CourthouseTime) : ℕ :=
  let weeklyParkingTime := ct.workDays * ct.parkingTime
  let weeklyWalkingTime := ct.workDays * ct.walkingTime
  let weeklyCrowdedDetectorTime := ct.crowdedDetectorDays * ct.crowdedDetectorTime
  let remainingTime := ct.totalWeeklyTime - weeklyParkingTime - weeklyWalkingTime - weeklyCrowdedDetectorTime
  remainingTime / (ct.workDays - ct.crowdedDetectorDays)

theorem less_crowded_detector_time_is_ten (ct : CourthouseTime)
  (h1 : ct.workDays = 5)
  (h2 : ct.parkingTime = 5)
  (h3 : ct.walkingTime = 3)
  (h4 : ct.crowdedDetectorDays = 2)
  (h5 : ct.crowdedDetectorTime = 30)
  (h6 : ct.totalWeeklyTime = 130) :
  lessCrowdedDetectorTime ct = 10 := by
  sorry

#eval lessCrowdedDetectorTime ⟨5, 5, 3, 2, 30, 130⟩

end NUMINAMATH_CALUDE_less_crowded_detector_time_is_ten_l3869_386950


namespace NUMINAMATH_CALUDE_common_power_theorem_l3869_386958

theorem common_power_theorem (a b x y : ℕ) : 
  a > 1 → b > 1 → x > 1 → y > 1 → 
  Nat.gcd a b = 1 → 
  x^a = y^b → 
  ∃ n : ℕ, n > 1 ∧ x = n^b ∧ y = n^a := by
sorry

end NUMINAMATH_CALUDE_common_power_theorem_l3869_386958


namespace NUMINAMATH_CALUDE_sample_correlation_coefficient_range_l3869_386917

/-- The sample correlation coefficient -/
def sample_correlation_coefficient : ℝ → Prop :=
  λ r => r ≥ -1 ∧ r ≤ 1

/-- Theorem: The sample correlation coefficient is not strictly between -1 and 1 -/
theorem sample_correlation_coefficient_range :
  ¬ (∀ r : ℝ, sample_correlation_coefficient r → r > -1 ∧ r < 1) :=
by
  sorry

end NUMINAMATH_CALUDE_sample_correlation_coefficient_range_l3869_386917


namespace NUMINAMATH_CALUDE_right_triangle_area_l3869_386987

/-- The area of a right triangle with hypotenuse 5 and one leg 3 is 6 -/
theorem right_triangle_area : ∀ (a b c : ℝ), 
  a = 3 → 
  c = 5 → 
  a^2 + b^2 = c^2 → 
  (1/2) * a * b = 6 :=
by
  sorry

end NUMINAMATH_CALUDE_right_triangle_area_l3869_386987


namespace NUMINAMATH_CALUDE_geometric_sequence_ratio_l3869_386981

/-- 
Given a geometric sequence {a_n} with common ratio q,
prove that if a₂ = 1 and a₁ + a₃ = -2, then q = -1.
-/
theorem geometric_sequence_ratio (a : ℕ → ℝ) (q : ℝ) :
  (∀ n, a (n + 1) = a n * q) →  -- geometric sequence condition
  a 2 = 1 →                    -- a₂ = 1
  a 1 + a 3 = -2 →             -- a₁ + a₃ = -2
  q = -1 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_ratio_l3869_386981


namespace NUMINAMATH_CALUDE_no_primes_in_range_l3869_386929

theorem no_primes_in_range (n : ℕ) (h : n > 1) :
  ∀ k, n! + 1 < k ∧ k < n! + 2*n → ¬ Nat.Prime k := by
  sorry

end NUMINAMATH_CALUDE_no_primes_in_range_l3869_386929


namespace NUMINAMATH_CALUDE_f_composition_quarter_l3869_386976

noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then Real.log x / Real.log 4
  else 2^x

theorem f_composition_quarter : f (f (1/4)) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_f_composition_quarter_l3869_386976


namespace NUMINAMATH_CALUDE_unique_quadratic_solution_l3869_386961

/-- 
Given a quadratic equation bx^2 + 6x + d = 0 with exactly one solution,
where b + d = 7 and b < d, prove that b = (7 - √13) / 2 and d = (7 + √13) / 2
-/
theorem unique_quadratic_solution (b d : ℝ) : 
  (∃! x, b * x^2 + 6 * x + d = 0) →
  b + d = 7 →
  b < d →
  b = (7 - Real.sqrt 13) / 2 ∧ d = (7 + Real.sqrt 13) / 2 := by
  sorry

end NUMINAMATH_CALUDE_unique_quadratic_solution_l3869_386961


namespace NUMINAMATH_CALUDE_jimmy_stair_climbing_time_l3869_386900

/-- Represents the time taken to climb stairs with an increasing time for each flight -/
def stair_climbing_time (initial_time : ℕ) (time_increase : ℕ) (num_flights : ℕ) : ℕ :=
  (num_flights * (2 * initial_time + (num_flights - 1) * time_increase)) / 2

/-- Theorem stating the total time Jimmy takes to climb eight flights of stairs -/
theorem jimmy_stair_climbing_time :
  stair_climbing_time 30 10 8 = 520 := by
  sorry

end NUMINAMATH_CALUDE_jimmy_stair_climbing_time_l3869_386900


namespace NUMINAMATH_CALUDE_back_seat_tickets_sold_l3869_386908

/-- Proves the number of back seat tickets sold at a concert --/
theorem back_seat_tickets_sold (total_seats : ℕ) (main_price back_price : ℕ) (total_revenue : ℕ) :
  total_seats = 20000 →
  main_price = 55 →
  back_price = 45 →
  total_revenue = 955000 →
  ∃ (main_seats back_seats : ℕ),
    main_seats + back_seats = total_seats ∧
    main_price * main_seats + back_price * back_seats = total_revenue ∧
    back_seats = 14500 := by
  sorry

#check back_seat_tickets_sold

end NUMINAMATH_CALUDE_back_seat_tickets_sold_l3869_386908


namespace NUMINAMATH_CALUDE_recurrence_necessary_not_sufficient_l3869_386936

/-- A sequence of real numbers -/
def Sequence := ℕ → ℝ

/-- The property that a sequence satisfies a_n = 2a_{n-1} for n ≥ 2 -/
def SatisfiesRecurrence (a : Sequence) : Prop :=
  ∀ n : ℕ, n ≥ 2 → a n = 2 * a (n - 1)

/-- The property that a sequence is geometric with common ratio 2 -/
def IsGeometricSequenceWithRatio2 (a : Sequence) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ ∀ n : ℕ, a n = r * (2 ^ n)

/-- The main theorem stating that SatisfiesRecurrence is necessary but not sufficient
    for IsGeometricSequenceWithRatio2 -/
theorem recurrence_necessary_not_sufficient :
  (∀ a : Sequence, IsGeometricSequenceWithRatio2 a → SatisfiesRecurrence a) ∧
  (∃ a : Sequence, SatisfiesRecurrence a ∧ ¬IsGeometricSequenceWithRatio2 a) :=
by sorry

end NUMINAMATH_CALUDE_recurrence_necessary_not_sufficient_l3869_386936


namespace NUMINAMATH_CALUDE_number_of_divisors_of_360_l3869_386948

theorem number_of_divisors_of_360 : Finset.card (Nat.divisors 360) = 24 := by
  sorry

end NUMINAMATH_CALUDE_number_of_divisors_of_360_l3869_386948


namespace NUMINAMATH_CALUDE_normal_distribution_probability_l3869_386904

-- Define a random variable X following N(0,1) distribution
def X : Real → Real := sorry

-- Define the probability measure for X
def P (s : Set Real) : Real := sorry

-- State the theorem
theorem normal_distribution_probability 
  (h1 : ∀ s, P s = ∫ x in s, X x)
  (h2 : P {x | x ≤ 1} = 0.8413) :
  P {x | -1 < x ∧ x < 0} = 0.3413 := by
  sorry

end NUMINAMATH_CALUDE_normal_distribution_probability_l3869_386904


namespace NUMINAMATH_CALUDE_lower_circle_radius_is_153_l3869_386983

/-- Configuration of circles and square between parallel lines -/
structure GeometricConfiguration where
  -- Distance between parallel lines
  line_distance : ℝ
  -- Side length of the square
  square_side : ℝ
  -- Radius of the upper circle
  upper_radius : ℝ
  -- The configuration satisfies the given conditions
  h1 : line_distance = 400
  h2 : square_side = 279
  h3 : upper_radius = 65

/-- Calculate the radius of the lower circle -/
def lower_circle_radius (config : GeometricConfiguration) : ℝ :=
  -- Placeholder for the actual calculation
  153

/-- Theorem stating that the radius of the lower circle is 153 units -/
theorem lower_circle_radius_is_153 (config : GeometricConfiguration) :
  lower_circle_radius config = 153 := by
  sorry

#check lower_circle_radius_is_153

end NUMINAMATH_CALUDE_lower_circle_radius_is_153_l3869_386983


namespace NUMINAMATH_CALUDE_pure_imaginary_ratio_l3869_386909

theorem pure_imaginary_ratio (a b : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) 
  (h3 : ∃ (y : ℝ), (3 - 8*I) * (a + b*I) = y*I) : 
  a / b = -8 / 3 := by
sorry

end NUMINAMATH_CALUDE_pure_imaginary_ratio_l3869_386909


namespace NUMINAMATH_CALUDE_second_term_is_negative_x_cubed_l3869_386906

/-- A line on a two-dimensional coordinate plane defined by a = x^2 - x^3 -/
def line (x : ℝ) : ℝ := x^2 - x^3

/-- The line touches the x-axis in 2 places -/
axiom touches_x_axis_twice : ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ line x₁ = 0 ∧ line x₂ = 0

/-- The second term of the equation representing the line is -x^3 -/
theorem second_term_is_negative_x_cubed :
  ∃ f : ℝ → ℝ, (∀ x, line x = f x - x^3) ∧ (∀ x, f x = x^2) :=
sorry

end NUMINAMATH_CALUDE_second_term_is_negative_x_cubed_l3869_386906


namespace NUMINAMATH_CALUDE_fraction_simplification_l3869_386984

theorem fraction_simplification :
  (15 : ℚ) / 35 * 28 / 45 * 75 / 28 = 5 / 7 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l3869_386984


namespace NUMINAMATH_CALUDE_point_in_second_quadrant_l3869_386934

def second_quadrant (x y : ℝ) : Prop := x < 0 ∧ y > 0

theorem point_in_second_quadrant (a : ℝ) : second_quadrant (-1) (a^2 + 1) := by
  sorry

end NUMINAMATH_CALUDE_point_in_second_quadrant_l3869_386934
