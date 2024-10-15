import Mathlib

namespace NUMINAMATH_CALUDE_simplification_proofs_l3470_347021

theorem simplification_proofs :
  (∀ x : ℝ, x ≥ 0 → Real.sqrt (x^2) = x) ∧
  ((5 * Real.sqrt 5)^2 = 125) ∧
  (Real.sqrt ((-1/7)^2) = 1/7) ∧
  ((-Real.sqrt (2/3))^2 = 2/3) := by
  sorry

end NUMINAMATH_CALUDE_simplification_proofs_l3470_347021


namespace NUMINAMATH_CALUDE_jeremy_wednesday_oranges_l3470_347000

/-- The number of oranges Jeremy picked on different days and the total -/
structure OrangePicks where
  monday : ℕ
  tuesday : ℕ
  wednesday : ℕ
  total : ℕ

/-- Given the conditions of Jeremy's orange picking, prove that he picked 70 oranges on Wednesday -/
theorem jeremy_wednesday_oranges (picks : OrangePicks) 
  (h1 : picks.monday = 100)
  (h2 : picks.tuesday = 3 * picks.monday)
  (h3 : picks.total = 470)
  (h4 : picks.total = picks.monday + picks.tuesday + picks.wednesday) :
  picks.wednesday = 70 := by
  sorry


end NUMINAMATH_CALUDE_jeremy_wednesday_oranges_l3470_347000


namespace NUMINAMATH_CALUDE_intersection_implies_a_geq_two_l3470_347035

-- Define sets A and B
def A (a : ℝ) : Set ℝ := {x : ℝ | x < a}
def B : Set ℝ := {x : ℝ | x^2 - 3*x + 2 < 0}

-- State the theorem
theorem intersection_implies_a_geq_two (a : ℝ) (h : A a ∩ B = B) : a ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_intersection_implies_a_geq_two_l3470_347035


namespace NUMINAMATH_CALUDE_arithmetic_mean_of_special_set_l3470_347009

def set_of_numbers : List ℕ := [8, 88, 888, 8888, 88888, 888888, 8888888, 88888888, 888888888]

theorem arithmetic_mean_of_special_set :
  let n := set_of_numbers.length
  let sum := set_of_numbers.sum
  sum / n = 98765432 := by sorry

end NUMINAMATH_CALUDE_arithmetic_mean_of_special_set_l3470_347009


namespace NUMINAMATH_CALUDE_parabola_line_intersection_theorem_l3470_347022

/-- Represents a parabola with focus on the x-axis and vertex at the origin -/
structure Parabola where
  p : ℝ
  equation : ℝ → ℝ → Prop := λ x y => y^2 = 2 * p * x

/-- Represents a line in the form x = my + b -/
structure Line where
  m : ℝ
  b : ℝ
  equation : ℝ → ℝ → Prop := λ x y => x = m * y + b

/-- Theorem stating the existence of a specific line intersecting the parabola -/
theorem parabola_line_intersection_theorem (C : Parabola) (h1 : C.equation 2 1) :
  ∃ (l : Line), l.b = 2 ∧
    (∃ (A B : ℝ × ℝ),
      C.equation A.1 A.2 ∧
      C.equation B.1 B.2 ∧
      l.equation A.1 A.2 ∧
      l.equation B.1 B.2 ∧
      let M := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)
      let N := (M.1, Real.sqrt (2 * C.p * M.1))
      (N.1 - A.1) * (N.1 - B.1) + (N.2 - A.2) * (N.2 - B.2) = 0) ∧
    (l.m = 2 ∨ l.m = -2) := by
  sorry

end NUMINAMATH_CALUDE_parabola_line_intersection_theorem_l3470_347022


namespace NUMINAMATH_CALUDE_consecutive_integers_product_l3470_347079

theorem consecutive_integers_product (a b c d e : ℕ) : 
  a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ e > 0 ∧
  b = a + 1 ∧ c = b + 1 ∧ d = c + 1 ∧ e = d + 1 ∧
  a * b * c * d * e = 2520 →
  e = 7 := by
sorry

end NUMINAMATH_CALUDE_consecutive_integers_product_l3470_347079


namespace NUMINAMATH_CALUDE_two_thirds_of_five_times_nine_l3470_347001

theorem two_thirds_of_five_times_nine : (2 / 3 : ℚ) * (5 * 9) = 30 := by
  sorry

end NUMINAMATH_CALUDE_two_thirds_of_five_times_nine_l3470_347001


namespace NUMINAMATH_CALUDE_units_digit_of_sequence_sum_l3470_347091

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def sequence_term (n : ℕ) : ℕ := factorial n + 10

def sequence_sum (n : ℕ) : ℕ := (List.range n).map sequence_term |>.sum

theorem units_digit_of_sequence_sum :
  sequence_sum 10 % 10 = 3 := by sorry

end NUMINAMATH_CALUDE_units_digit_of_sequence_sum_l3470_347091


namespace NUMINAMATH_CALUDE_geometric_sum_value_l3470_347043

theorem geometric_sum_value (x : ℝ) (h1 : x^2023 - 3*x + 2 = 0) (h2 : x ≠ 1) :
  x^2022 + x^2021 + x^2020 + x^2019 + x^2018 + x^2017 + x^2016 + x^2015 + x^2014 + x^2013 + 
  x^2012 + x^2011 + x^2010 + x^2009 + x^2008 + x^2007 + x^2006 + x^2005 + x^2004 + x^2003 + 
  x^2002 + x^2001 + x^2000 + x^1999 + x^1998 + x^1997 + x^1996 + x^1995 + x^1994 + x^1993 + 
  x^1992 + x^1991 + x^1990 + x^1989 + x^1988 + x^1987 + x^1986 + x^1985 + x^1984 + x^1983 + 
  x^1982 + x^1981 + x^1980 + x^1979 + x^1978 + x^1977 + x^1976 + x^1975 + x^1974 + x^1973 + 
  -- ... (continuing the pattern)
  x^22 + x^21 + x^20 + x^19 + x^18 + x^17 + x^16 + x^15 + x^14 + x^13 + 
  x^12 + x^11 + x^10 + x^9 + x^8 + x^7 + x^6 + x^5 + x^4 + x^3 + x^2 + x + 1 = 3 := by
sorry

end NUMINAMATH_CALUDE_geometric_sum_value_l3470_347043


namespace NUMINAMATH_CALUDE_luke_paula_commute_l3470_347092

/-- The problem of Luke and Paula's commute times -/
theorem luke_paula_commute :
  -- Luke's bus time to work
  ∀ (luke_bus : ℕ),
  -- Paula's bus time as a fraction of Luke's
  ∀ (paula_fraction : ℚ),
  -- Total travel time for both
  ∀ (total_time : ℕ),
  -- Conditions
  luke_bus = 70 →
  paula_fraction = 3/5 →
  total_time = 504 →
  -- Conclusion
  ∃ (bike_multiple : ℚ),
    -- Luke's bike time = bus time * bike_multiple
    (luke_bus * bike_multiple).floor +
    -- Luke's bus time to work
    luke_bus +
    -- Paula's bus time to work
    (paula_fraction * luke_bus).floor +
    -- Paula's bus time back home
    (paula_fraction * luke_bus).floor = total_time ∧
    bike_multiple = 5 :=
by sorry

end NUMINAMATH_CALUDE_luke_paula_commute_l3470_347092


namespace NUMINAMATH_CALUDE_line_perpendicular_to_plane_l3470_347067

-- Define the types for plane and line
variable (Plane Line : Type)

-- Define the relations
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Line → Line → Prop)

-- State the theorem
theorem line_perpendicular_to_plane 
  (α : Plane) (a b : Line) :
  perpendicular a α → parallel a b → perpendicular b α := by
  sorry

end NUMINAMATH_CALUDE_line_perpendicular_to_plane_l3470_347067


namespace NUMINAMATH_CALUDE_sugar_per_larger_cookie_l3470_347019

/-- Proves that if 40 cookies each use 1/8 cup of sugar, and the same total amount of sugar
    is used to make 25 larger cookies, then each larger cookie will contain 1/5 cup of sugar. -/
theorem sugar_per_larger_cookie :
  let small_cookies : ℕ := 40
  let large_cookies : ℕ := 25
  let sugar_per_small : ℚ := 1 / 8
  let total_sugar : ℚ := small_cookies * sugar_per_small
  let sugar_per_large : ℚ := total_sugar / large_cookies
  sugar_per_large = 1 / 5 := by
  sorry

end NUMINAMATH_CALUDE_sugar_per_larger_cookie_l3470_347019


namespace NUMINAMATH_CALUDE_unique_four_digit_number_l3470_347005

/-- Represents a 4-digit number as a tuple of its digits -/
def FourDigitNumber := (Nat × Nat × Nat × Nat)

/-- Checks if a tuple represents a valid 4-digit number -/
def isValidFourDigitNumber (n : FourDigitNumber) : Prop :=
  let (a, b, c, d) := n
  1 ≤ a ∧ a ≤ 9 ∧ 0 ≤ b ∧ b ≤ 9 ∧ 0 ≤ c ∧ c ≤ 9 ∧ 0 ≤ d ∧ d ≤ 9

/-- Checks if a 4-digit number satisfies the given conditions -/
def satisfiesConditions (n : FourDigitNumber) : Prop :=
  let (a, b, c, d) := n
  b = 3 * a ∧ c = a + b ∧ d = 3 * b

theorem unique_four_digit_number :
  ∃! (n : FourDigitNumber), isValidFourDigitNumber n ∧ satisfiesConditions n ∧ n = (1, 3, 4, 9) :=
by sorry

end NUMINAMATH_CALUDE_unique_four_digit_number_l3470_347005


namespace NUMINAMATH_CALUDE_cubic_sum_theorem_l3470_347038

theorem cubic_sum_theorem (x y z : ℝ) 
  (h1 : x + y + z = 3) 
  (h2 : x*y + y*z + z*x = -3) 
  (h3 : x*y*z = -3) : 
  x^3 + y^3 + z^3 = 45 := by
sorry

end NUMINAMATH_CALUDE_cubic_sum_theorem_l3470_347038


namespace NUMINAMATH_CALUDE_remainder_sum_mod_three_l3470_347049

theorem remainder_sum_mod_three
  (a b c d : ℕ)
  (ha : a % 6 = 4)
  (hb : b % 6 = 4)
  (hc : c % 6 = 4)
  (hd : d % 6 = 4) :
  (a + b + c + d) % 3 = 1 := by
  sorry

end NUMINAMATH_CALUDE_remainder_sum_mod_three_l3470_347049


namespace NUMINAMATH_CALUDE_constant_term_binomial_expansion_l3470_347018

theorem constant_term_binomial_expansion :
  ∃ (c : ℝ), c = 7 ∧ 
  ∀ (x : ℝ), x ≠ 0 → 
  ∃ (f : ℝ → ℝ), (λ x => (x^(1/3) + 1/(2*x))^8) = 
    (λ x => c + f x) ∧ (∀ (y : ℝ), y ≠ 0 → f y ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_constant_term_binomial_expansion_l3470_347018


namespace NUMINAMATH_CALUDE_ellipse_focus_l3470_347052

theorem ellipse_focus (center : ℝ × ℝ) (major_axis : ℝ) (minor_axis : ℝ) :
  center = (3, -1) →
  major_axis = 6 →
  minor_axis = 4 →
  let focus_distance := Real.sqrt ((major_axis / 2)^2 - (minor_axis / 2)^2)
  let focus_x := center.1 + focus_distance
  (focus_x, center.2) = (3 + Real.sqrt 5, -1) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_focus_l3470_347052


namespace NUMINAMATH_CALUDE_nathan_ate_twenty_gumballs_l3470_347074

/-- The number of gumballs in each package -/
def gumballs_per_package : ℕ := 5

/-- The number of packages Nathan finished -/
def packages_finished : ℕ := 4

/-- The total number of gumballs Nathan ate -/
def gumballs_eaten : ℕ := gumballs_per_package * packages_finished

theorem nathan_ate_twenty_gumballs : gumballs_eaten = 20 := by
  sorry

end NUMINAMATH_CALUDE_nathan_ate_twenty_gumballs_l3470_347074


namespace NUMINAMATH_CALUDE_sarah_trucks_l3470_347011

theorem sarah_trucks (trucks_to_jeff trucks_to_amy trucks_left : ℕ) 
  (h1 : trucks_to_jeff = 13)
  (h2 : trucks_to_amy = 21)
  (h3 : trucks_left = 38) :
  trucks_to_jeff + trucks_to_amy + trucks_left = 72 := by
  sorry

end NUMINAMATH_CALUDE_sarah_trucks_l3470_347011


namespace NUMINAMATH_CALUDE_cubic_double_root_value_l3470_347085

theorem cubic_double_root_value (a b : ℝ) (p q r : ℝ) : 
  (∀ x : ℝ, x^3 + p*x^2 + q*x + r = (x - a)^2 * (x - b)) →
  p = -6 →
  q = 9 →
  r = 0 ∨ r = -4 :=
sorry

end NUMINAMATH_CALUDE_cubic_double_root_value_l3470_347085


namespace NUMINAMATH_CALUDE_valid_pairings_count_l3470_347087

def number_of_bowls : ℕ := 5
def number_of_glasses : ℕ := 5
def number_of_colors : ℕ := 5

def total_pairings : ℕ := number_of_bowls * number_of_glasses

def invalid_pairings : ℕ := 1

theorem valid_pairings_count : 
  total_pairings - invalid_pairings = 24 :=
sorry

end NUMINAMATH_CALUDE_valid_pairings_count_l3470_347087


namespace NUMINAMATH_CALUDE_xiaolis_estimate_l3470_347082

theorem xiaolis_estimate (p q a b : ℝ) 
  (h1 : p > q) (h2 : q > 0) (h3 : a > b) (h4 : b > 0) : 
  (p + a) - (q + b) > p - q := by
  sorry

end NUMINAMATH_CALUDE_xiaolis_estimate_l3470_347082


namespace NUMINAMATH_CALUDE_sequence_eventually_periodic_l3470_347088

def is_valid_sequence (a : ℕ → ℕ) : Prop :=
  ∀ n > 2, a (n + 1) = (a n ^ n + a (n - 1)) % 10

def is_eventually_periodic (a : ℕ → ℕ) : Prop :=
  ∃ n₀ p, p > 0 ∧ ∀ k ≥ n₀, a (k + p) = a k

theorem sequence_eventually_periodic (a : ℕ → ℕ) (h : is_valid_sequence a) :
  is_eventually_periodic a := by
  sorry

#check sequence_eventually_periodic

end NUMINAMATH_CALUDE_sequence_eventually_periodic_l3470_347088


namespace NUMINAMATH_CALUDE_container_volume_comparison_l3470_347033

theorem container_volume_comparison (a r : ℝ) (ha : a > 0) (hr : r > 0) 
  (h_eq : (2 * a)^3 = (4/3) * Real.pi * r^3) : 
  (2*a + 2)^3 > (4/3) * Real.pi * (r + 1)^3 := by
  sorry

end NUMINAMATH_CALUDE_container_volume_comparison_l3470_347033


namespace NUMINAMATH_CALUDE_third_root_of_cubic_l3470_347008

theorem third_root_of_cubic (a b : ℝ) : 
  (∀ x : ℝ, a * x^3 + (a + 2*b) * x^2 + (b - 3*a) * x + (8 - a) = 0 ↔ x = -2 ∨ x = 3 ∨ x = 4/3) →
  ∃ x : ℝ, x ≠ -2 ∧ x ≠ 3 ∧ a * x^3 + (a + 2*b) * x^2 + (b - 3*a) * x + (8 - a) = 0 ∧ x = 4/3 :=
by sorry

end NUMINAMATH_CALUDE_third_root_of_cubic_l3470_347008


namespace NUMINAMATH_CALUDE_hold_age_ratio_l3470_347053

theorem hold_age_ratio (mother_age : ℕ) (son_age : ℕ) (h1 : mother_age = 36) (h2 : mother_age = 3 * son_age) :
  (mother_age - 8) / (son_age - 8) = 7 := by
  sorry

end NUMINAMATH_CALUDE_hold_age_ratio_l3470_347053


namespace NUMINAMATH_CALUDE_count_valid_programs_l3470_347015

/-- Represents the available courses --/
inductive Course
| English
| Algebra
| Geometry
| History
| Art
| Latin
| Science

/-- Checks if a course is a mathematics course --/
def isMathCourse (c : Course) : Bool :=
  match c with
  | Course.Algebra => true
  | Course.Geometry => true
  | _ => false

/-- Checks if a course is a science course --/
def isScienceCourse (c : Course) : Bool :=
  match c with
  | Course.Science => true
  | _ => false

/-- Represents a program of 4 courses --/
structure Program :=
  (courses : Finset Course)
  (size_eq : courses.card = 4)
  (has_english : Course.English ∈ courses)
  (has_math : ∃ c ∈ courses, isMathCourse c)
  (has_science : ∃ c ∈ courses, isScienceCourse c)

/-- The set of all valid programs --/
def validPrograms : Finset Program := sorry

theorem count_valid_programs :
  validPrograms.card = 19 := by sorry

end NUMINAMATH_CALUDE_count_valid_programs_l3470_347015


namespace NUMINAMATH_CALUDE_elijah_score_l3470_347065

/-- Proves that Elijah's score is 43 points given the team's total score,
    number of players, and average score of other players. -/
theorem elijah_score (total_score : ℕ) (num_players : ℕ) (other_avg : ℕ) 
  (h1 : total_score = 85)
  (h2 : num_players = 8)
  (h3 : other_avg = 6) :
  total_score - (num_players - 1) * other_avg = 43 := by
  sorry

#check elijah_score

end NUMINAMATH_CALUDE_elijah_score_l3470_347065


namespace NUMINAMATH_CALUDE_fixed_point_of_exponential_function_l3470_347063

theorem fixed_point_of_exponential_function (a : ℝ) (ha : a > 0) :
  let f : ℝ → ℝ := λ x ↦ a^(x-1) + 3
  f 1 = 4 := by sorry

end NUMINAMATH_CALUDE_fixed_point_of_exponential_function_l3470_347063


namespace NUMINAMATH_CALUDE_inscribed_prism_surface_area_l3470_347002

/-- The surface area of a right square prism inscribed in a sphere -/
theorem inscribed_prism_surface_area (r h : ℝ) (hr : r = Real.sqrt 6) (hh : h = 4) :
  let a := Real.sqrt ((r^2 - h^2/4) / 2)
  2 * a^2 + 4 * a * h = 40 :=
sorry

end NUMINAMATH_CALUDE_inscribed_prism_surface_area_l3470_347002


namespace NUMINAMATH_CALUDE_sqrt_500_simplification_l3470_347094

theorem sqrt_500_simplification : Real.sqrt 500 = 10 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_500_simplification_l3470_347094


namespace NUMINAMATH_CALUDE_initial_socks_count_l3470_347048

theorem initial_socks_count (S : ℕ) : 
  (S ≥ 4) →
  (∃ (remaining : ℕ), remaining = S - 4) →
  (∃ (after_donation : ℕ), after_donation = (remaining : ℚ) * (1 / 3 : ℚ)) →
  (after_donation + 13 = 25) →
  S = 40 :=
by sorry

end NUMINAMATH_CALUDE_initial_socks_count_l3470_347048


namespace NUMINAMATH_CALUDE_line_circle_intersection_l3470_347034

/-- Given a point (a, b) outside a circle and a line ax + by = r^2, 
    prove that the line intersects the circle but doesn't pass through the center. -/
theorem line_circle_intersection (a b r : ℝ) (h : a^2 + b^2 > r^2) :
  ∃ (x y : ℝ), x^2 + y^2 = r^2 ∧ a*x + b*y = r^2 ∧ (x ≠ 0 ∨ y ≠ 0) := by
  sorry

end NUMINAMATH_CALUDE_line_circle_intersection_l3470_347034


namespace NUMINAMATH_CALUDE_cuboid_edge_lengths_l3470_347058

theorem cuboid_edge_lengths (a b c : ℝ) : 
  (a * b : ℝ) / (b * c) = 16 / 21 →
  (a * b : ℝ) / (a * c) = 16 / 28 →
  a^2 + b^2 + c^2 = 29^2 →
  a = 16 ∧ b = 12 ∧ c = 21 := by
  sorry

end NUMINAMATH_CALUDE_cuboid_edge_lengths_l3470_347058


namespace NUMINAMATH_CALUDE_circle_diameter_l3470_347069

theorem circle_diameter (C : ℝ) (h : C = 36) : 
  (C / π) = (36 : ℝ) / π := by sorry

end NUMINAMATH_CALUDE_circle_diameter_l3470_347069


namespace NUMINAMATH_CALUDE_parabola_unique_coefficients_l3470_347041

/-- A parabola is defined by the equation y = ax^2 + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The y-coordinate of a point on the parabola given its x-coordinate -/
def Parabola.y_at (p : Parabola) (x : ℝ) : ℝ :=
  p.a * x^2 + p.b * x + p.c

/-- The slope of the tangent line to the parabola at a given x-coordinate -/
def Parabola.slope_at (p : Parabola) (x : ℝ) : ℝ :=
  2 * p.a * x + p.b

/-- Theorem: For a parabola y = ax^2 + bx + c, if it passes through (1, 1),
    and the slope of the tangent line at (2, -1) is 1,
    then a = 3, b = -11, and c = 9 -/
theorem parabola_unique_coefficients (p : Parabola) 
    (h1 : p.y_at 1 = 1)
    (h2 : p.y_at 2 = -1)
    (h3 : p.slope_at 2 = 1) :
    p.a = 3 ∧ p.b = -11 ∧ p.c = 9 := by
  sorry

end NUMINAMATH_CALUDE_parabola_unique_coefficients_l3470_347041


namespace NUMINAMATH_CALUDE_triangle_problem_l3470_347089

theorem triangle_problem (a b c A B C : ℝ) (h1 : c * Real.cos A - Real.sqrt 3 * a * Real.sin C - c = 0)
  (h2 : a = 2) (h3 : (1/2) * b * c * Real.sin A = Real.sqrt 3) :
  A = π/3 ∧ b = 2 ∧ c = 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_problem_l3470_347089


namespace NUMINAMATH_CALUDE_square_area_increase_l3470_347032

/-- Given a square with initial side length 4, if the side length increases by x
    and the area increases by y, then y = x^2 + 8x -/
theorem square_area_increase (x y : ℝ) : 
  (4 + x)^2 - 4^2 = y → y = x^2 + 8*x := by
  sorry

end NUMINAMATH_CALUDE_square_area_increase_l3470_347032


namespace NUMINAMATH_CALUDE_fish_apple_equivalence_l3470_347036

/-- Represents the value of one fish in terms of apples -/
def fish_value (f l r a : ℚ) : Prop :=
  5 * f = 3 * l ∧ l = 6 * r ∧ 3 * r = 2 * a ∧ f = 12/5 * a

/-- Theorem stating that under the given trading conditions, one fish is worth 12/5 apples -/
theorem fish_apple_equivalence :
  ∀ f l r a : ℚ, fish_value f l r a :=
by
  sorry

#check fish_apple_equivalence

end NUMINAMATH_CALUDE_fish_apple_equivalence_l3470_347036


namespace NUMINAMATH_CALUDE_triangle_side_length_l3470_347026

theorem triangle_side_length (A B C : ℝ) (a b c : ℝ) : 
  A = π / 6 →  -- 30° in radians
  B = π / 4 →  -- 45° in radians
  a = Real.sqrt 2 →
  (Real.sin A) * b = (Real.sin B) * a →
  b = 2 := by
sorry

end NUMINAMATH_CALUDE_triangle_side_length_l3470_347026


namespace NUMINAMATH_CALUDE_exponent_division_l3470_347095

theorem exponent_division (a : ℝ) : 2 * a^3 / a^2 = 2 * a := by sorry

end NUMINAMATH_CALUDE_exponent_division_l3470_347095


namespace NUMINAMATH_CALUDE_not_all_greater_than_quarter_l3470_347047

theorem not_all_greater_than_quarter (a b c : ℝ) 
  (ha : 0 < a ∧ a < 1) (hb : 0 < b ∧ b < 1) (hc : 0 < c ∧ c < 1) : 
  ¬ ((1 - a) * b > 1/4 ∧ (1 - b) * c > 1/4 ∧ (1 - c) * a > 1/4) := by
  sorry

end NUMINAMATH_CALUDE_not_all_greater_than_quarter_l3470_347047


namespace NUMINAMATH_CALUDE_quadratic_polynomial_satisfies_conditions_l3470_347013

theorem quadratic_polynomial_satisfies_conditions :
  ∃ (p : ℝ → ℝ),
    (∀ x, p x = (14 * x^2 + 4 * x + 12) / 15) ∧
    p (-2) = 4 ∧
    p 1 = 2 ∧
    p 3 = 10 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_polynomial_satisfies_conditions_l3470_347013


namespace NUMINAMATH_CALUDE_cab_driver_income_l3470_347016

def average_income : ℝ := 440
def num_days : ℕ := 5
def known_incomes : List ℝ := [250, 650, 400, 500]

theorem cab_driver_income :
  let total_income := average_income * num_days
  let known_total := known_incomes.sum
  total_income - known_total = 400 := by sorry

end NUMINAMATH_CALUDE_cab_driver_income_l3470_347016


namespace NUMINAMATH_CALUDE_triangle_area_calculation_l3470_347093

theorem triangle_area_calculation (a b : Real) (C : Real) :
  a = 45 ∧ b = 60 ∧ C = 37 →
  abs ((1/2) * a * b * Real.sin (C * Real.pi / 180) - 812.45) < 0.01 :=
by sorry

end NUMINAMATH_CALUDE_triangle_area_calculation_l3470_347093


namespace NUMINAMATH_CALUDE_cubic_roots_relation_l3470_347031

/-- Given a cubic equation x^3 + px^2 + qx + r = 0 with roots α, β, γ, 
    returns a function that computes expressions involving these roots -/
def cubicRootRelations (p q r : ℝ) : 
  (ℝ → ℝ → ℝ → ℝ) → ℝ := sorry

theorem cubic_roots_relation (a b c s t : ℝ) : 
  cubicRootRelations 3 4 (-11) (fun x y z => x) = a ∧
  cubicRootRelations 3 4 (-11) (fun x y z => y) = b ∧
  cubicRootRelations 3 4 (-11) (fun x y z => z) = c ∧
  cubicRootRelations (-2) s t (fun x y z => x) = a + b ∧
  cubicRootRelations (-2) s t (fun x y z => y) = b + c ∧
  cubicRootRelations (-2) s t (fun x y z => z) = c + a →
  s = 8 ∧ t = 23 := by sorry

end NUMINAMATH_CALUDE_cubic_roots_relation_l3470_347031


namespace NUMINAMATH_CALUDE_shape_to_square_possible_l3470_347084

/-- Represents a shape on a graph paper --/
structure Shape :=
  (area : ℝ)

/-- Represents a triangle --/
structure Triangle :=
  (area : ℝ)

/-- Represents a square --/
structure Square :=
  (side_length : ℝ)

/-- Function to divide a shape into triangles --/
def divide_into_triangles (s : Shape) : List Triangle := sorry

/-- Function to assemble triangles into a square --/
def assemble_square (triangles : List Triangle) : Option Square := sorry

/-- Theorem stating that the shape can be divided into 5 triangles and assembled into a square --/
theorem shape_to_square_possible (s : Shape) : 
  ∃ (triangles : List Triangle) (sq : Square), 
    divide_into_triangles s = triangles ∧ 
    triangles.length = 5 ∧ 
    assemble_square triangles = some sq :=
by sorry

end NUMINAMATH_CALUDE_shape_to_square_possible_l3470_347084


namespace NUMINAMATH_CALUDE_product_sum_fractions_l3470_347077

theorem product_sum_fractions : (3 * 4 * 5) * (1/3 + 1/4 - 1/5) = 23 := by
  sorry

end NUMINAMATH_CALUDE_product_sum_fractions_l3470_347077


namespace NUMINAMATH_CALUDE_isosceles_triangle_area_l3470_347066

/-- An isosceles triangle PQR with given side lengths and altitude properties -/
structure IsoscelesTriangle where
  /-- Length of equal sides PQ and PR -/
  side : ℝ
  /-- Length of base QR -/
  base : ℝ
  /-- Altitude PS bisects base QR -/
  altitude_bisects_base : True

/-- The area of the isosceles triangle PQR is 360 square units -/
theorem isosceles_triangle_area
  (t : IsoscelesTriangle)
  (h1 : t.side = 41)
  (h2 : t.base = 18) :
  t.side * t.base / 2 = 360 :=
sorry

end NUMINAMATH_CALUDE_isosceles_triangle_area_l3470_347066


namespace NUMINAMATH_CALUDE_complex_equation_magnitude_l3470_347007

theorem complex_equation_magnitude (z : ℂ) (a b : ℝ) (n : ℕ) 
  (h : a * z^n + b * Complex.I * z^(n-1) + b * Complex.I * z - a = 0) : 
  Complex.abs z = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_magnitude_l3470_347007


namespace NUMINAMATH_CALUDE_road_trip_distance_l3470_347040

/-- Represents Rick's road trip with 5 destinations -/
structure RoadTrip where
  leg1 : ℝ
  leg2 : ℝ
  leg3 : ℝ
  leg4 : ℝ
  leg5 : ℝ

/-- Conditions of Rick's road trip -/
def validRoadTrip (trip : RoadTrip) : Prop :=
  trip.leg2 = 2 * trip.leg1 ∧
  trip.leg3 = 40 ∧
  trip.leg3 = trip.leg1 / 2 ∧
  trip.leg4 = 2 * (trip.leg1 + trip.leg2 + trip.leg3) ∧
  trip.leg5 = 1.5 * trip.leg4

/-- The total distance of the road trip -/
def totalDistance (trip : RoadTrip) : ℝ :=
  trip.leg1 + trip.leg2 + trip.leg3 + trip.leg4 + trip.leg5

/-- Theorem stating that the total distance of a valid road trip is 1680 miles -/
theorem road_trip_distance (trip : RoadTrip) (h : validRoadTrip trip) :
  totalDistance trip = 1680 := by
  sorry

end NUMINAMATH_CALUDE_road_trip_distance_l3470_347040


namespace NUMINAMATH_CALUDE_tangent_slope_implies_a_l3470_347024

-- Define the curve
def f (a x : ℝ) : ℝ := x^4 + a*x^2 + 1

-- Define the derivative of the curve
def f' (a x : ℝ) : ℝ := 4*x^3 + 2*a*x

theorem tangent_slope_implies_a (a : ℝ) :
  f a (-1) = a + 2 → f' a (-1) = 8 → a = -6 := by
  sorry

end NUMINAMATH_CALUDE_tangent_slope_implies_a_l3470_347024


namespace NUMINAMATH_CALUDE_B_power_97_l3470_347028

def B : Matrix (Fin 3) (Fin 3) ℝ := !![1, 0, 0; 0, 0, -2; 0, 2, 0]

theorem B_power_97 : 
  B^97 = !![1, 0, 0; 0, 0, -2 * 16^24; 0, 2 * 16^24, 0] := by sorry

end NUMINAMATH_CALUDE_B_power_97_l3470_347028


namespace NUMINAMATH_CALUDE_most_suitable_sampling_methods_l3470_347078

/-- Represents different sampling methods --/
inductive SamplingMethod
  | SystematicSampling
  | StratifiedSampling
  | SimpleRandomSampling

/-- Represents a survey scenario --/
structure SurveyScenario where
  description : String
  sampleSize : Nat

/-- Determines the most suitable sampling method for a given scenario --/
def mostSuitableSamplingMethod (scenario : SurveyScenario) : SamplingMethod :=
  sorry

/-- The three survey scenarios described in the problem --/
def scenario1 : SurveyScenario :=
  { description := "First-year high school students' mathematics learning, 2 students from each class",
    sampleSize := 2 }

def scenario2 : SurveyScenario :=
  { description := "Math competition results, 12 students selected from different score ranges",
    sampleSize := 12 }

def scenario3 : SurveyScenario :=
  { description := "Sports meeting, arranging tracks for 6 students in 400m race",
    sampleSize := 6 }

/-- Theorem stating the most suitable sampling methods for the given scenarios --/
theorem most_suitable_sampling_methods :
  (mostSuitableSamplingMethod scenario1 = SamplingMethod.SystematicSampling) ∧
  (mostSuitableSamplingMethod scenario2 = SamplingMethod.StratifiedSampling) ∧
  (mostSuitableSamplingMethod scenario3 = SamplingMethod.SimpleRandomSampling) :=
by sorry

end NUMINAMATH_CALUDE_most_suitable_sampling_methods_l3470_347078


namespace NUMINAMATH_CALUDE_binomial_probability_theorem_l3470_347003

/-- A random variable following a binomial distribution -/
structure BinomialRV where
  n : ℕ
  p : ℝ
  h1 : 0 ≤ p ∧ p ≤ 1

/-- The expectation of a binomial random variable -/
def expectation (X : BinomialRV) : ℝ := X.n * X.p

/-- The variance of a binomial random variable -/
def variance (X : BinomialRV) : ℝ := X.n * X.p * (1 - X.p)

/-- The probability of a binomial random variable being equal to k -/
def probability (X : BinomialRV) (k : ℕ) : ℝ :=
  (X.n.choose k) * (X.p ^ k) * ((1 - X.p) ^ (X.n - k))

theorem binomial_probability_theorem (X : BinomialRV) 
  (h2 : expectation X = 2)
  (h3 : variance X = 4/3) :
  probability X 2 = 80/243 := by
  sorry

end NUMINAMATH_CALUDE_binomial_probability_theorem_l3470_347003


namespace NUMINAMATH_CALUDE_max_value_sin_tan_function_l3470_347073

theorem max_value_sin_tan_function :
  ∀ x : ℝ, 2 * Real.sin x ^ 2 - Real.tan x ^ 2 ≤ 3 - 2 * Real.sqrt 2 ∧
  ∃ x : ℝ, 2 * Real.sin x ^ 2 - Real.tan x ^ 2 = 3 - 2 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_max_value_sin_tan_function_l3470_347073


namespace NUMINAMATH_CALUDE_pencil_cost_l3470_347072

theorem pencil_cost (initial_amount : ℕ) (amount_left : ℕ) (candy_cost : ℕ) : 
  initial_amount = 43 → amount_left = 18 → candy_cost = 5 → 
  initial_amount - amount_left - candy_cost = 20 := by
  sorry

end NUMINAMATH_CALUDE_pencil_cost_l3470_347072


namespace NUMINAMATH_CALUDE_walking_problem_l3470_347046

/-- Proves that the given conditions lead to the correct system of equations --/
theorem walking_problem (x y : ℝ) : 
  (∀ t : ℝ, t * x < t * y) → -- Xiao Wang walks faster than Xiao Zhang
  (3 * x + 210 = 5 * y) →    -- Distance condition after 3 and 5 minutes
  (10 * y - 10 * x = 100) →  -- Time and initial distance condition
  (∃ d : ℝ, d > 0 ∧ 10 * x = d ∧ 10 * y = d + 100) → -- Both reach the museum in 10 minutes
  (3 * x + 210 = 5 * y ∧ 10 * y - 10 * x = 100) :=
by sorry

end NUMINAMATH_CALUDE_walking_problem_l3470_347046


namespace NUMINAMATH_CALUDE_candy_mixture_price_l3470_347023

theorem candy_mixture_price (price1 price2 : ℝ) (h1 : price1 = 10) (h2 : price2 = 15) : 
  let weight_ratio := 3
  let total_weight := weight_ratio + 1
  let total_cost := price1 * weight_ratio + price2
  total_cost / total_weight = 11.25 := by
sorry

end NUMINAMATH_CALUDE_candy_mixture_price_l3470_347023


namespace NUMINAMATH_CALUDE_smallest_solution_of_equation_l3470_347055

theorem smallest_solution_of_equation : 
  ∃ x : ℝ, x = -15 ∧ 
  (∀ y : ℝ, 3 * y^2 + 39 * y - 75 = y * (y + 16) → y ≥ x) :=
by sorry

end NUMINAMATH_CALUDE_smallest_solution_of_equation_l3470_347055


namespace NUMINAMATH_CALUDE_parabola_sum_l3470_347099

/-- A parabola with equation y = ax^2 + bx + c, vertex (3, -2), and passing through (0, 5) -/
structure Parabola where
  a : ℚ
  b : ℚ
  c : ℚ
  vertex_x : ℚ := 3
  vertex_y : ℚ := -2
  point_x : ℚ := 0
  point_y : ℚ := 5
  eq_at_vertex : -2 = a * 3^2 + b * 3 + c
  eq_at_point : 5 = c
  vertex_formula : b = -2 * a * vertex_x

theorem parabola_sum (p : Parabola) : p.a + p.b + p.c = 10/9 := by
  sorry

end NUMINAMATH_CALUDE_parabola_sum_l3470_347099


namespace NUMINAMATH_CALUDE_jamie_score_l3470_347051

theorem jamie_score (team_total : ℝ) (num_players : ℕ) (other_players_avg : ℝ) 
  (h1 : team_total = 60)
  (h2 : num_players = 6)
  (h3 : other_players_avg = 4.8) : 
  team_total - (num_players - 1) * other_players_avg = 36 :=
by sorry

end NUMINAMATH_CALUDE_jamie_score_l3470_347051


namespace NUMINAMATH_CALUDE_indefinite_integral_proof_l3470_347020

theorem indefinite_integral_proof (x : ℝ) (h1 : x ≠ -2) (h2 : x ≠ 2) :
  let f := fun (x : ℝ) => (x^3 - 6*x^2 + 11*x - 10) / ((x+2)*(x-2)^3)
  let F := fun (x : ℝ) => Real.log (abs (x+2)) + 1 / (2*(x-2)^2)
  deriv F x = f x := by sorry

end NUMINAMATH_CALUDE_indefinite_integral_proof_l3470_347020


namespace NUMINAMATH_CALUDE_arthur_walk_distance_l3470_347025

/-- The number of blocks Arthur walks east -/
def blocks_east : ℕ := 8

/-- The number of blocks Arthur walks north -/
def blocks_north : ℕ := 15

/-- The number of blocks Arthur walks west -/
def blocks_west : ℕ := 3

/-- The length of each block in miles -/
def block_length : ℚ := 1/2

/-- The total distance Arthur walks in miles -/
def total_distance : ℚ := (blocks_east + blocks_north + blocks_west : ℚ) * block_length

theorem arthur_walk_distance : total_distance = 13 := by
  sorry

end NUMINAMATH_CALUDE_arthur_walk_distance_l3470_347025


namespace NUMINAMATH_CALUDE_distinct_three_digit_count_base_6_l3470_347039

/-- The number of three-digit numbers with distinct digits in base b -/
def distinct_three_digit_count (b : ℕ) : ℕ := (b - 1)^2 * (b - 2)

/-- Theorem: In base 6, there are exactly 100 three-digit numbers with distinct digits -/
theorem distinct_three_digit_count_base_6 : distinct_three_digit_count 6 = 100 := by
  sorry

#eval distinct_three_digit_count 6  -- This should evaluate to 100

end NUMINAMATH_CALUDE_distinct_three_digit_count_base_6_l3470_347039


namespace NUMINAMATH_CALUDE_number_equation_solution_l3470_347075

theorem number_equation_solution : ∃ (x : ℝ), x + 3 * x = 20 ∧ x = 5 := by sorry

end NUMINAMATH_CALUDE_number_equation_solution_l3470_347075


namespace NUMINAMATH_CALUDE_last_digit_of_total_edge_count_l3470_347090

/-- Represents an 8x8 chessboard -/
def Chessboard := Fin 8 × Fin 8

/-- Represents a 1x2 domino piece -/
def Domino := Σ' (i : Fin 8) (j : Fin 7), Unit

/-- A tiling of the chessboard with dominos -/
def Tiling := Chessboard → Option Domino

/-- The number of valid tilings of the chessboard -/
def numTilings : ℕ := 12988816

/-- An edge of the chessboard -/
inductive Edge
| horizontal (i : Fin 9) (j : Fin 8) : Edge
| vertical (i : Fin 8) (j : Fin 9) : Edge

/-- The number of tilings that include a given edge -/
def edgeCount (e : Edge) : ℕ := sorry

/-- The sum of edgeCount for all edges -/
def totalEdgeCount : ℕ := sorry

/-- Theorem: The last digit of totalEdgeCount is 4 -/
theorem last_digit_of_total_edge_count :
  totalEdgeCount % 10 = 4 := by sorry

end NUMINAMATH_CALUDE_last_digit_of_total_edge_count_l3470_347090


namespace NUMINAMATH_CALUDE_cut_cylinder_volume_l3470_347006

/-- Represents a right cylinder with a vertical planar cut -/
structure CutCylinder where
  height : ℝ
  baseRadius : ℝ
  cutArea : ℝ

/-- The volume of the larger piece of a cut cylinder -/
def largerPieceVolume (c : CutCylinder) : ℝ := sorry

theorem cut_cylinder_volume 
  (c : CutCylinder) 
  (h_height : c.height = 20)
  (h_radius : c.baseRadius = 5)
  (h_cut_area : c.cutArea = 100 * Real.sqrt 2) :
  largerPieceVolume c = 250 + 375 * Real.pi := by sorry

end NUMINAMATH_CALUDE_cut_cylinder_volume_l3470_347006


namespace NUMINAMATH_CALUDE_frank_candies_l3470_347064

def frank_tickets_game1 : ℕ := 33
def frank_tickets_game2 : ℕ := 9
def candy_cost : ℕ := 6

theorem frank_candies : 
  (frank_tickets_game1 + frank_tickets_game2) / candy_cost = 7 := by sorry

end NUMINAMATH_CALUDE_frank_candies_l3470_347064


namespace NUMINAMATH_CALUDE_power_product_equality_l3470_347057

theorem power_product_equality : 2^3 * 3 * 5^3 * 7 = 21000 := by
  sorry

end NUMINAMATH_CALUDE_power_product_equality_l3470_347057


namespace NUMINAMATH_CALUDE_nancys_contribution_is_36_l3470_347004

/-- The number of bottle caps Marilyn had initially -/
def initial_caps : ℝ := 51.0

/-- The number of bottle caps Marilyn had after Nancy's contribution -/
def final_caps : ℝ := 87.0

/-- The number of bottle caps Nancy gave to Marilyn -/
def nancys_contribution : ℝ := final_caps - initial_caps

theorem nancys_contribution_is_36 : nancys_contribution = 36 := by
  sorry

end NUMINAMATH_CALUDE_nancys_contribution_is_36_l3470_347004


namespace NUMINAMATH_CALUDE_fraction_simplification_l3470_347037

theorem fraction_simplification :
  (2 * (Real.sqrt 2 + Real.sqrt 6)) / (3 * Real.sqrt (2 + Real.sqrt 3)) = 4 / 3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l3470_347037


namespace NUMINAMATH_CALUDE_ellipse_equation_l3470_347010

/-- An ellipse with center at the origin, focus on the y-axis, eccentricity 1/2, and focal length 8 has the equation y²/64 + x²/48 = 1 -/
theorem ellipse_equation (x y : ℝ) : 
  let center := (0 : ℝ × ℝ)
  let focus_on_y_axis := true
  let eccentricity := (1 : ℝ) / 2
  let focal_length := (8 : ℝ)
  (y^2 / 64 + x^2 / 48 = 1) ↔ 
    ∃ (a b c : ℝ), 
      a > 0 ∧ b > 0 ∧
      c = focal_length / 2 ∧
      eccentricity = c / a ∧
      b^2 = a^2 - c^2 ∧
      y^2 / a^2 + x^2 / b^2 = 1 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_equation_l3470_347010


namespace NUMINAMATH_CALUDE_triangle_area_l3470_347027

theorem triangle_area (a b c : ℝ) (h_perimeter : a + b + c = 10 + 2 * Real.sqrt 7)
  (h_ratio : ∃ (k : ℝ), a = 2 * k ∧ b = 3 * k ∧ c = k * Real.sqrt 7) :
  let S := Real.sqrt ((1/4) * (c^2 * a^2 - ((c^2 + a^2 - b^2)/2)^2))
  S = 6 * Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_triangle_area_l3470_347027


namespace NUMINAMATH_CALUDE_point_A_in_second_quadrant_l3470_347060

/-- A point in the second quadrant has a negative x-coordinate and a positive y-coordinate -/
def second_quadrant (x y : ℝ) : Prop := x < 0 ∧ y > 0

/-- The x-coordinate of point A -/
def x_coord (x : ℝ) : ℝ := 6 - 2*x

/-- The y-coordinate of point A -/
def y_coord (x : ℝ) : ℝ := x - 5

/-- Theorem: If point A(6-2x, x-5) lies in the second quadrant, then x > 5 -/
theorem point_A_in_second_quadrant (x : ℝ) :
  second_quadrant (x_coord x) (y_coord x) → x > 5 := by
  sorry

end NUMINAMATH_CALUDE_point_A_in_second_quadrant_l3470_347060


namespace NUMINAMATH_CALUDE_ratio_equality_l3470_347054

theorem ratio_equality (a b c x y z : ℝ) 
  (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c) 
  (pos_x : 0 < x) (pos_y : 0 < y) (pos_z : 0 < z)
  (sum_squares_abc : a^2 + b^2 + c^2 = 49)
  (sum_squares_xyz : x^2 + y^2 + z^2 = 64)
  (dot_product : a*x + b*y + c*z = 56) :
  (a + b + c) / (x + y + z) = 7/8 := by
sorry

end NUMINAMATH_CALUDE_ratio_equality_l3470_347054


namespace NUMINAMATH_CALUDE_lucky_lacy_correct_percentage_l3470_347098

/-- The percentage of problems Lucky Lacy got correct on an algebra test -/
theorem lucky_lacy_correct_percentage :
  ∀ x : ℕ,
  x > 0 →
  let total_problems := 7 * x
  let missed_problems := 2 * x
  let correct_problems := total_problems - missed_problems
  let correct_fraction : ℚ := correct_problems / total_problems
  let correct_percentage := correct_fraction * 100
  ∃ ε > 0, abs (correct_percentage - 71.43) < ε :=
by
  sorry

end NUMINAMATH_CALUDE_lucky_lacy_correct_percentage_l3470_347098


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l3470_347071

theorem arithmetic_sequence_problem (x : ℚ) :
  let a₁ := 3 * x - 4
  let a₂ := 6 * x - 14
  let a₃ := 4 * x + 2
  let d := a₂ - a₁  -- common difference
  let a_n (n : ℕ) := a₁ + (n - 1) * d  -- general term
  ∃ n : ℕ, a_n n = 4018 ∧ n = 716 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l3470_347071


namespace NUMINAMATH_CALUDE_complement_M_in_U_l3470_347081

-- Define the universal set U
def U : Set ℝ := {x : ℝ | x > 0}

-- Define the set M
def M : Set ℝ := {x : ℝ | x > 1}

-- Define the complement of M in U
def complementMU : Set ℝ := {x : ℝ | x ∈ U ∧ x ∉ M}

-- Theorem statement
theorem complement_M_in_U :
  complementMU = {x : ℝ | 0 < x ∧ x ≤ 1} :=
by sorry

end NUMINAMATH_CALUDE_complement_M_in_U_l3470_347081


namespace NUMINAMATH_CALUDE_area_of_special_quadrilateral_l3470_347097

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- Represents a triangle -/
structure Triangle :=
  (a b c : Point)

/-- Represents a quadrilateral -/
structure Quadrilateral :=
  (a b c d : Point)

/-- Calculates the area of a triangle -/
noncomputable def triangleArea (t : Triangle) : ℝ := sorry

/-- Calculates the area of a quadrilateral -/
noncomputable def quadrilateralArea (q : Quadrilateral) : ℝ := sorry

/-- Main theorem -/
theorem area_of_special_quadrilateral 
  (mainTriangle : Triangle) 
  (smallTriangles : Fin 4 → Triangle)
  (quadrilaterals : Fin 3 → Quadrilateral)
  (h1 : ∀ i, triangleArea (smallTriangles i) = 1)
  (h2 : quadrilateralArea (quadrilaterals 0) = quadrilateralArea (quadrilaterals 1))
  (h3 : quadrilateralArea (quadrilaterals 1) = quadrilateralArea (quadrilaterals 2)) :
  quadrilateralArea (quadrilaterals 0) = 1 + Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_area_of_special_quadrilateral_l3470_347097


namespace NUMINAMATH_CALUDE_inequality_abc_l3470_347070

theorem inequality_abc (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (habc : a * b * c = 1) :
  (a - 1 + 1 / b) * (b - 1 + 1 / c) * (c - 1 + 1 / a) ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_abc_l3470_347070


namespace NUMINAMATH_CALUDE_least_k_cube_divisible_by_168_l3470_347030

theorem least_k_cube_divisible_by_168 :
  ∀ k : ℕ, k > 0 → k^3 % 168 = 0 → k ≥ 42 :=
by sorry

end NUMINAMATH_CALUDE_least_k_cube_divisible_by_168_l3470_347030


namespace NUMINAMATH_CALUDE_student_council_choices_l3470_347080

/-- Represents the composition of the student council -/
structure StudentCouncil where
  freshmen : Nat
  sophomores : Nat
  juniors : Nat

/-- The given student council composition -/
def council : StudentCouncil := ⟨6, 5, 4⟩

/-- Number of ways to choose one person as president -/
def choosePresident (sc : StudentCouncil) : Nat :=
  sc.freshmen + sc.sophomores + sc.juniors

/-- Number of ways to choose one person from each grade -/
def chooseOneFromEach (sc : StudentCouncil) : Nat :=
  sc.freshmen * sc.sophomores * sc.juniors

/-- Number of ways to choose two people from different grades -/
def chooseTwoFromDifferent (sc : StudentCouncil) : Nat :=
  sc.freshmen * sc.sophomores +
  sc.freshmen * sc.juniors +
  sc.sophomores * sc.juniors

theorem student_council_choices :
  choosePresident council = 15 ∧
  chooseOneFromEach council = 120 ∧
  chooseTwoFromDifferent council = 74 := by
  sorry

end NUMINAMATH_CALUDE_student_council_choices_l3470_347080


namespace NUMINAMATH_CALUDE_dot_product_problem_l3470_347086

theorem dot_product_problem (a b : ℝ × ℝ) : 
  a = (2, 1) → a - b = (-1, 2) → a • b = 5 := by
  sorry

end NUMINAMATH_CALUDE_dot_product_problem_l3470_347086


namespace NUMINAMATH_CALUDE_path_area_and_cost_l3470_347059

/-- Calculates the area of a rectangular path around a field -/
def path_area (field_length field_width path_width : ℝ) : ℝ :=
  (field_length + 2 * path_width) * (field_width + 2 * path_width) - field_length * field_width

/-- Calculates the cost of constructing a path given its area and cost per unit area -/
def construction_cost (path_area cost_per_unit : ℝ) : ℝ :=
  path_area * cost_per_unit

theorem path_area_and_cost (field_length field_width path_width cost_per_unit : ℝ) 
  (h1 : field_length = 85)
  (h2 : field_width = 55)
  (h3 : path_width = 2.5)
  (h4 : cost_per_unit = 2) : 
  path_area field_length field_width path_width = 725 ∧ 
  construction_cost (path_area field_length field_width path_width) cost_per_unit = 1450 := by
  sorry

#eval path_area 85 55 2.5
#eval construction_cost (path_area 85 55 2.5) 2

end NUMINAMATH_CALUDE_path_area_and_cost_l3470_347059


namespace NUMINAMATH_CALUDE_number_of_girls_l3470_347012

def number_of_boys : ℕ := 5
def committee_size : ℕ := 4
def boys_in_committee : ℕ := 2
def girls_in_committee : ℕ := 2
def total_committees : ℕ := 150

def choose (n k : ℕ) : ℕ := (Nat.factorial n) / ((Nat.factorial k) * (Nat.factorial (n - k)))

theorem number_of_girls : 
  ∃ g : ℕ, 
    choose number_of_boys boys_in_committee * choose g girls_in_committee = total_committees ∧ 
    g = 6 :=
sorry

end NUMINAMATH_CALUDE_number_of_girls_l3470_347012


namespace NUMINAMATH_CALUDE_carlos_summer_reading_l3470_347056

/-- Carlos' summer reading challenge -/
theorem carlos_summer_reading 
  (july_books august_books total_goal : ℕ) 
  (h1 : july_books = 28)
  (h2 : august_books = 30)
  (h3 : total_goal = 100) :
  total_goal - (july_books + august_books) = 42 := by
  sorry

end NUMINAMATH_CALUDE_carlos_summer_reading_l3470_347056


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_l3470_347042

theorem partial_fraction_decomposition (C D : ℚ) :
  (∀ x : ℚ, x ≠ 7 ∧ x ≠ -2 →
    (5 * x - 3) / (x^2 - 5*x - 14) = C / (x - 7) + D / (x + 2)) →
  C = 32/9 ∧ D = 13/9 := by
sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_l3470_347042


namespace NUMINAMATH_CALUDE_additional_distance_for_average_speed_l3470_347076

theorem additional_distance_for_average_speed
  (initial_distance : ℝ)
  (initial_speed : ℝ)
  (increased_speed : ℝ)
  (target_average_speed : ℝ)
  (h1 : initial_distance = 15)
  (h2 : initial_speed = 30)
  (h3 : increased_speed = 55)
  (h4 : target_average_speed = 50) :
  ∃ (additional_distance : ℝ),
    (initial_distance + additional_distance) / ((initial_distance / initial_speed) + (additional_distance / increased_speed)) = target_average_speed ∧
    additional_distance = 110 :=
by sorry

end NUMINAMATH_CALUDE_additional_distance_for_average_speed_l3470_347076


namespace NUMINAMATH_CALUDE_computer_additions_per_hour_l3470_347061

/-- The number of additions a computer can perform per second -/
def additions_per_second : ℕ := 10000

/-- The number of seconds in one hour -/
def seconds_per_hour : ℕ := 3600

/-- The number of additions a computer can perform in one hour -/
def additions_per_hour : ℕ := additions_per_second * seconds_per_hour

/-- Theorem stating that the computer performs 36 million additions in one hour -/
theorem computer_additions_per_hour : 
  additions_per_hour = 36000000 := by sorry

end NUMINAMATH_CALUDE_computer_additions_per_hour_l3470_347061


namespace NUMINAMATH_CALUDE_competition_problem_l3470_347017

theorem competition_problem : ((7^2 - 3^2)^4) = 2560000 := by
  sorry

end NUMINAMATH_CALUDE_competition_problem_l3470_347017


namespace NUMINAMATH_CALUDE_sum_of_numbers_l3470_347062

theorem sum_of_numbers : (3 : ℚ) / 25 + (1 : ℚ) / 5 + 55.21 = 55.53 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_numbers_l3470_347062


namespace NUMINAMATH_CALUDE_q_implies_k_range_p_or_q_and_not_p_and_q_implies_k_range_l3470_347050

-- Define proposition p
def p (k : ℝ) : Prop := ∀ x : ℝ, x^2 - k*x + 2*k + 5 ≥ 0

-- Define proposition q
def q (k : ℝ) : Prop := ∃ a b : ℝ, a > 0 ∧ b < 0 ∧ a = 4 - k ∧ b = 1 - k

-- Theorem 1
theorem q_implies_k_range (k : ℝ) : q k → 1 < k ∧ k < 4 := by sorry

-- Theorem 2
theorem p_or_q_and_not_p_and_q_implies_k_range (k : ℝ) : 
  (p k ∨ q k) ∧ ¬(p k ∧ q k) → (-2 ≤ k ∧ k ≤ 1) ∨ (4 ≤ k ∧ k ≤ 10) := by sorry

end NUMINAMATH_CALUDE_q_implies_k_range_p_or_q_and_not_p_and_q_implies_k_range_l3470_347050


namespace NUMINAMATH_CALUDE_cubic_sum_theorem_l3470_347014

theorem cubic_sum_theorem (p q r : ℝ) 
  (sum_eq : p + q + r = 4)
  (sum_prod_eq : p * q + p * r + q * r = 6)
  (prod_eq : p * q * r = -8) : 
  p^3 + q^3 + r^3 = 8 := by
sorry

end NUMINAMATH_CALUDE_cubic_sum_theorem_l3470_347014


namespace NUMINAMATH_CALUDE_fish_to_buy_l3470_347044

def current_fish : ℕ := 212
def desired_total : ℕ := 280

theorem fish_to_buy : desired_total - current_fish = 68 := by sorry

end NUMINAMATH_CALUDE_fish_to_buy_l3470_347044


namespace NUMINAMATH_CALUDE_fourteenSidedFigureArea_l3470_347068

/-- A point in 2D space represented by its x and y coordinates -/
structure Point where
  x : ℝ
  y : ℝ

/-- The vertices of the fourteen-sided figure -/
def vertices : List Point := [
  ⟨1, 2⟩, ⟨1, 3⟩, ⟨2, 4⟩, ⟨3, 5⟩, ⟨4, 6⟩, ⟨5, 5⟩, ⟨6, 5⟩,
  ⟨7, 4⟩, ⟨7, 3⟩, ⟨6, 2⟩, ⟨5, 1⟩, ⟨4, 1⟩, ⟨3, 1⟩, ⟨2, 2⟩
]

/-- Calculate the area of a polygon given its vertices -/
def polygonArea (vertices : List Point) : ℝ :=
  sorry -- Implement the calculation of polygon area

/-- Theorem stating that the area of the fourteen-sided figure is 14 square centimeters -/
theorem fourteenSidedFigureArea : polygonArea vertices = 14 := by
  sorry

end NUMINAMATH_CALUDE_fourteenSidedFigureArea_l3470_347068


namespace NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l3470_347083

theorem expression_simplification_and_evaluation :
  ∀ a : ℚ, a + 1 ≠ 0 → a + 2 ≠ 0 →
  (a + 1 - (5 + 2*a) / (a + 1)) / ((a^2 + 4*a + 4) / (a + 1)) = (a - 2) / (a + 2) ∧
  (let simplified := (a - 2) / (a + 2);
   a = -3 → simplified = 5) :=
by sorry

end NUMINAMATH_CALUDE_expression_simplification_and_evaluation_l3470_347083


namespace NUMINAMATH_CALUDE_nested_subtract_201_l3470_347045

/-- Recursive function to represent nested subtractions -/
def nestedSubtract (x : ℝ) : ℕ → ℝ
  | 0 => x - 1
  | n + 1 => x - nestedSubtract x n

/-- Theorem stating that the nested subtraction equals 1 iff x = 201 -/
theorem nested_subtract_201 (x : ℝ) :
  nestedSubtract x 199 = 1 ↔ x = 201 := by
  sorry

#check nested_subtract_201

end NUMINAMATH_CALUDE_nested_subtract_201_l3470_347045


namespace NUMINAMATH_CALUDE_M_mod_1500_l3470_347029

/-- A sequence of positive integers whose binary representation has exactly 9 ones -/
def T : Nat → Nat := sorry

/-- The 1500th number in the sequence T -/
def M : Nat := T 1500

/-- The remainder when M is divided by 1500 -/
theorem M_mod_1500 : M % 1500 = 500 := by sorry

end NUMINAMATH_CALUDE_M_mod_1500_l3470_347029


namespace NUMINAMATH_CALUDE_bus_seat_difference_l3470_347096

/-- Represents the seating configuration of a bus --/
structure BusSeating where
  left_seats : ℕ
  right_seats : ℕ
  back_seat_capacity : ℕ
  regular_seat_capacity : ℕ
  total_capacity : ℕ

/-- Theorem about the difference in seats between left and right sides of the bus --/
theorem bus_seat_difference (bus : BusSeating) : 
  bus.left_seats = 15 →
  bus.regular_seat_capacity = 3 →
  bus.back_seat_capacity = 8 →
  bus.total_capacity = 89 →
  bus.left_seats > bus.right_seats →
  bus.left_seats - bus.right_seats = 3 := by
  sorry

#check bus_seat_difference

end NUMINAMATH_CALUDE_bus_seat_difference_l3470_347096
