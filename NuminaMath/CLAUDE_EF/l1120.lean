import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_evaluate_expression_l1120_112039

-- Define variables
variable (x y : ℝ)

-- Define P and Q as per the conditions
def P (x y : ℝ) : ℝ := 2 * x + y
def Q (x y : ℝ) : ℝ := x - 2 * y

-- State the theorem
theorem evaluate_expression (x y : ℝ) :
  (P x y + Q x y) / (P x y - Q x y) - (P x y - Q x y) / (P x y + Q x y) = 
  8 * (x^2 - 2*x*y - y^2) / ((x + 3*y) * (3*x - y)) :=
by
  -- Proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_evaluate_expression_l1120_112039


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_g_l1120_112022

-- Define the function f with domain [-1, 4]
def f : ℝ → Set ℝ := sorry

-- Define the domain of f
def dom_f : Set ℝ := Set.Icc (-1) 4

-- Define the function g(x) = f(2x-1)
def g (x : ℝ) : Set ℝ := f (2 * x - 1)

-- Theorem statement
theorem domain_of_g : 
  {x : ℝ | g x = f (2 * x - 1)} = Set.Icc 0 (5/2) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_g_l1120_112022


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cricket_average_score_l1120_112032

theorem cricket_average_score (total_matches : ℕ) (first_matches : ℕ) (last_matches : ℕ)
  (avg_first : ℚ) (avg_all : ℚ) (avg_last : ℚ) :
  total_matches = first_matches + last_matches →
  total_matches = 5 →
  first_matches = 2 →
  avg_first = 40 →
  avg_all = 22 →
  (avg_first * first_matches + avg_last * last_matches) / total_matches = avg_all →
  avg_last = 10 :=
by
  intro h1 h2 h3 h4 h5 h6
  have h7 : avg_last = (avg_all * total_matches - avg_first * first_matches) / last_matches := by
    sorry
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cricket_average_score_l1120_112032


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_common_ratio_l1120_112004

/-- A geometric sequence is a sequence where each term after the first is found by multiplying the previous term by a fixed, non-zero number called the common ratio. -/
def GeometricSequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = q * a n

/-- The sum of the first n terms of a geometric sequence. -/
noncomputable def GeometricSum (a : ℕ → ℝ) (q : ℝ) (n : ℕ) : ℝ :=
  if q = 1 then n * a 0 else a 0 * (1 - q^n) / (1 - q)

/-- For a geometric sequence where the sum of the first 3 terms is equal to 3 times the first term,
    the common ratio is either -2 or 1. -/
theorem geometric_sequence_common_ratio (a : ℕ → ℝ) (q : ℝ) :
  GeometricSequence a q →
  GeometricSum a q 3 = 3 * a 0 →
  q = -2 ∨ q = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_common_ratio_l1120_112004


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pipe_renovation_theorem_l1120_112066

/-- Represents the pipe network renovation project -/
structure PipeRenovation where
  totalLength : ℝ
  efficiencyIncrease : ℝ
  daysAheadOfSchedule : ℕ
  initialConstructionDays : ℕ
  maxTotalDays : ℕ

/-- Calculates the actual daily renovation length -/
noncomputable def actualDailyLength (pr : PipeRenovation) : ℝ :=
  pr.totalLength / ((pr.totalLength / (pr.efficiencyIncrease * (pr.totalLength / (pr.totalLength / pr.efficiencyIncrease - pr.daysAheadOfSchedule)))) + pr.daysAheadOfSchedule)

/-- Calculates the additional daily length required to complete the project within the maximum total days -/
noncomputable def additionalDailyLength (pr : PipeRenovation) (actualDaily : ℝ) : ℝ :=
  (pr.totalLength - actualDaily * pr.initialConstructionDays) / (pr.maxTotalDays - pr.initialConstructionDays) - actualDaily

/-- Theorem stating the correct actual daily renovation length and minimum additional daily length required -/
theorem pipe_renovation_theorem (pr : PipeRenovation) 
  (h1 : pr.totalLength = 3600)
  (h2 : pr.efficiencyIncrease = 1.2)
  (h3 : pr.daysAheadOfSchedule = 10)
  (h4 : pr.initialConstructionDays = 20)
  (h5 : pr.maxTotalDays = 40) :
  actualDailyLength pr = 72 ∧ 
  additionalDailyLength pr (actualDailyLength pr) ≥ 36 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_pipe_renovation_theorem_l1120_112066


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_question_one_question_two_l1120_112009

-- Define the sets A, B, and C
def A (a : ℝ) : Set ℝ := {x | x^2 - a*x + a^2 - 19 = 0}
def B : Set ℝ := {x | x^2 - 5*x + 6 = 0}
def C : Set ℝ := {x | x^2 + 2*x - 8 = 0}

-- Theorem for question 1
theorem question_one (a : ℝ) : A a ∩ B = A a ∪ B → a = 5 := by sorry

-- Theorem for question 2
theorem question_two (a : ℝ) : (A a ∩ B).Nonempty ∧ (A a ∩ C = ∅) → a = -2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_question_one_question_two_l1120_112009


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_symmetric_function_l1120_112090

noncomputable def f (x a b : ℝ) : ℝ := (1 - 1/4 * x^2) * (x^2 + a*x + b)

theorem max_value_of_symmetric_function (a b : ℝ) :
  (∀ x : ℝ, f x a b = f (-2-x) a b) →  -- Symmetry condition
  (∃ x_max : ℝ, ∀ x : ℝ, f x a b ≤ f x_max a b ∧ f x_max a b = 4) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_symmetric_function_l1120_112090


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_inequality_l1120_112026

theorem power_inequality (x y : ℝ) (h : x > y) : (2 : ℝ)^x + (2 : ℝ)^(-y) > 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_inequality_l1120_112026


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1120_112038

-- Define the function f(x) = x · arcsin(x)
noncomputable def f (x : ℝ) : ℝ := x * Real.arcsin x

-- State the theorem
theorem f_properties :
  (∀ x, f x = f (-x)) ∧ 
  (∀ x, f x ≤ π / 2) ∧
  (∃ x, f x = π / 2) ∧
  (∀ x, f x ≥ 0) ∧
  (∃ x, f x = 0) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1120_112038


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_ratio_l1120_112002

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a parabola of the form x² = 4y -/
structure Parabola where
  focus : Point

/-- Represents a line of the form ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Calculate the distance between two points -/
noncomputable def distance (p1 p2 : Point) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2)

/-- Theorem: The ratio of distances from intersection points to focus -/
theorem intersection_distance_ratio
  (p : Parabola)
  (l : Line)
  (A B : Point)
  (h1 : p.focus = Point.mk 0 1)
  (h2 : l.a = 1 ∧ l.b = -Real.sqrt 3 ∧ l.c = Real.sqrt 3)
  (h3 : A.x^2 = 4 * A.y ∧ B.x^2 = 4 * B.y)  -- A and B are on the parabola
  (h4 : l.a * A.x + l.b * A.y + l.c = 0 ∧ l.a * B.x + l.b * B.y + l.c = 0)  -- A and B are on the line
  (h5 : distance A p.focus > distance B p.focus)
  : distance A p.focus / distance B p.focus = 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_distance_ratio_l1120_112002


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_thirteen_exists_l1120_112048

def S : Finset ℕ := {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12}

theorem sum_thirteen_exists (A : Finset ℕ) (h1 : A ⊆ S) (h2 : A.card = 7) :
  ∃ x y, x ∈ A ∧ y ∈ A ∧ x ≠ y ∧ x + y = 13 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_thirteen_exists_l1120_112048


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_sum_l1120_112077

/-- The parabola y^2 = 8x -/
def Parabola : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.2^2 = 8 * p.1}

/-- Point A -/
def A : ℝ × ℝ := (1, 0)

/-- Point B -/
def B : ℝ × ℝ := (7, 6)

/-- Distance between two points -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

theorem min_distance_sum :
  ∀ P ∈ Parabola, distance A P + distance B P ≥ 8 ∧
  ∃ P ∈ Parabola, distance A P + distance B P = 8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_sum_l1120_112077


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_opal_initial_winnings_l1120_112027

/-- Represents the initial amount Opal won in dollars -/
def initial_winnings : ℝ := sorry

/-- Represents the profit rate on the second bet -/
def profit_rate : ℝ := 0.60

/-- The total amount Opal puts into savings -/
def total_savings : ℝ := 90

/-- Theorem stating that Opal's initial winnings were $100 -/
theorem opal_initial_winnings : 
  (initial_winnings / 2) + ((initial_winnings / 2) * (1 + profit_rate) / 2) = total_savings →
  initial_winnings = 100 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_opal_initial_winnings_l1120_112027


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monochromatic_arithmetic_progression_exists_l1120_112020

/-- A coloring of the set {1, 2, 3, 4, 5, 6, 7, 8, 9} using two colors -/
def Coloring := Fin 9 → Bool

/-- Checks if three numbers form an arithmetic progression -/
def isArithmeticProgression (a b c : Fin 9) : Prop :=
  (b.val : ℤ) - (a.val : ℤ) = (c.val : ℤ) - (b.val : ℤ)

/-- Main theorem: Any two-coloring of {1, 2, 3, 4, 5, 6, 7, 8, 9} contains a monochromatic arithmetic progression of length 3 -/
theorem monochromatic_arithmetic_progression_exists (c : Coloring) : 
  ∃ (a b d : Fin 9), a < b ∧ b < d ∧ 
    isArithmeticProgression a b d ∧ 
    (c a = c b ∧ c b = c d) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_monochromatic_arithmetic_progression_exists_l1120_112020


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_perfect_square_to_550_l1120_112015

theorem closest_perfect_square_to_550 : 
  ∀ n : ℤ, n ≠ 23 → n * n ≠ 529 → |550 - (23 * 23)| ≤ |550 - (n * n)| :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_perfect_square_to_550_l1120_112015


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_group_size_l1120_112000

theorem largest_group_size : ∃ x : ℕ, x = 37 ∧ 
  (∀ y : ℕ, y > x → 
    ¬(∃ a b c : ℕ, 
      a + b + c = y ∧
      (a = Nat.floor (y / 2) ∨ a = Nat.ceil (y / 2)) ∧
      (b = Nat.floor (y / 3) ∨ b = Nat.ceil (y / 3)) ∧
      (c = Nat.floor (y / 5) ∨ c = Nat.ceil (y / 5)))) :=
by sorry

#check largest_group_size

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_group_size_l1120_112000


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_girls_proof_l1120_112029

/-- The total number of students in the class -/
def total_students : ℕ := 20

/-- Represents the number of girls in the class -/
def num_girls : ℕ → Prop := λ _ => True

/-- Represents the constraint that no three boys have lists with the same number of girls -/
def unique_list_constraint (d : ℕ) : Prop :=
  total_students - d ≤ 2 * (d + 1)

/-- The minimum number of girls satisfying the constraint -/
def min_girls : ℕ := 6

theorem min_girls_proof :
  (∀ d, d < min_girls → ¬(unique_list_constraint d)) ∧
  unique_list_constraint min_girls := by
  sorry

#check min_girls_proof

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_girls_proof_l1120_112029


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_focal_length_with_triangle_area_l1120_112069

/-- Represents a hyperbola with parameters a and b -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_pos_a : a > 0
  h_pos_b : b > 0

/-- The area of triangle ODE formed by the origin and intersections of x = a with asymptotes -/
def triangle_area (h : Hyperbola) : ℝ := h.a * h.b

/-- The focal length of the hyperbola -/
noncomputable def focal_length (h : Hyperbola) : ℝ := 2 * Real.sqrt (h.a^2 + h.b^2)

/-- Theorem stating the minimum focal length given the triangle area constraint -/
theorem min_focal_length_with_triangle_area (h : Hyperbola) 
  (h_area : triangle_area h = 8) : 
  ∃ (min_focal_length : ℝ), min_focal_length = 8 ∧ 
  ∀ (h' : Hyperbola), triangle_area h' = 8 → focal_length h' ≥ min_focal_length :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_focal_length_with_triangle_area_l1120_112069


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cookies_fit_without_touching_l1120_112054

/-
Define the problem setup:
- Square baking sheet with side length 8 inches
- 6 circular cookies, each with diameter 3 inches
-/

def baking_sheet_side : ℝ := 8
def cookie_diameter : ℝ := 3
def num_cookies : ℕ := 6

/-
Define a function to calculate the distance between two points
-/
noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ :=
  Real.sqrt ((x1 - x2)^2 + (y1 - y2)^2)

/-
Define a configuration of 6 points within the 8x8 square
representing the centers of the cookies
-/
def cookie_centers : List (ℝ × ℝ) :=
  [(1.5, 1.5), (1.5, 4), (1.5, 6.5),
   (6.5, 1.5), (6.5, 4), (6.5, 6.5)]

/-
Theorem: The 6 cookies can be placed on the baking sheet without touching
-/
theorem cookies_fit_without_touching :
  (∀ (p1 p2 : ℝ × ℝ), p1 ∈ cookie_centers → p2 ∈ cookie_centers → p1 ≠ p2 →
    distance p1.fst p1.snd p2.fst p2.snd ≥ cookie_diameter) ∧
  (∀ (p : ℝ × ℝ), p ∈ cookie_centers →
    0 ≤ p.fst ∧ p.fst ≤ baking_sheet_side ∧ 0 ≤ p.snd ∧ p.snd ≤ baking_sheet_side) :=
by sorry

#check cookies_fit_without_touching

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cookies_fit_without_touching_l1120_112054


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l1120_112061

-- Define the function f(x) = -x + 1/x
noncomputable def f (x : ℝ) : ℝ := -x + 1/x

-- Define the interval [-2, -1/3]
def interval : Set ℝ := {x | -2 ≤ x ∧ x ≤ -1/3}

-- State the theorem
theorem max_value_of_f :
  ∃ (max : ℝ), max = 3/2 ∧ ∀ (x : ℝ), x ∈ interval → f x ≤ max := by
  -- Proof goes here
  sorry

#check max_value_of_f

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_f_l1120_112061


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_book_price_change_l1120_112047

theorem book_price_change (P : ℝ) (h : P > 0) : 
  P * (1 - 0.15) * (1 + 0.10) = P * 0.935 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_book_price_change_l1120_112047


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l1120_112018

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := Real.sqrt (x - 1) / (x - 3)

-- Define the domain of f
def domain_f : Set ℝ := {x | x ≥ 1 ∧ x ≠ 3}

-- Theorem stating that the domain of f is [1,3) ∪ (3,+∞)
theorem domain_of_f :
  domain_f = Set.Icc 1 3 ∪ Set.Ioi 3 :=
by
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l1120_112018


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_consecutive_tame_pairs_l1120_112016

-- Define the operation of summing the squares of digits
def sum_of_squares_of_digits (n : ℕ) : ℕ := sorry

-- Define what it means for a number to be tame
def is_tame (n : ℕ) : Prop :=
  ∃ k : ℕ, (Nat.iterate sum_of_squares_of_digits k n) = 1

-- Theorem statement
theorem infinitely_many_consecutive_tame_pairs :
  ∀ m : ℕ, ∃ n : ℕ, n > m ∧ is_tame n ∧ is_tame (n + 1) :=
by
  sorry

#check infinitely_many_consecutive_tame_pairs

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_consecutive_tame_pairs_l1120_112016


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_number_proof_l1120_112093

noncomputable def z : ℂ := (-3 + Complex.I) / Complex.I

theorem complex_number_proof :
  (z = 1 + 3 * Complex.I) ∧
  ((2 - Complex.I) - z = 1 - 4 * Complex.I) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_number_proof_l1120_112093


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_harry_hike_length_l1120_112030

/-- Represents Harry's hike with given conditions -/
structure HarryHike where
  initial_water : ℚ
  duration : ℚ
  remaining_water : ℚ
  leak_rate : ℚ
  last_mile_drink : ℚ
  first_part_drink_rate : ℚ

/-- Calculates the length of Harry's hike based on given conditions -/
def hike_length (h : HarryHike) : ℚ :=
  let total_water_used := h.initial_water - h.remaining_water
  let water_leaked := h.leak_rate * h.duration
  let first_part_water := total_water_used - water_leaked - h.last_mile_drink
  let first_part_miles := first_part_water / h.first_part_drink_rate
  first_part_miles + 1

/-- Theorem stating that Harry's hike length is 4 miles -/
theorem harry_hike_length :
  let h : HarryHike := {
    initial_water := 6
    duration := 2
    remaining_water := 1
    leak_rate := 1
    last_mile_drink := 1
    first_part_drink_rate := 2/3
  }
  hike_length h = 4 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_harry_hike_length_l1120_112030


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_odd_function_solution_l1120_112081

-- Problem 1
theorem inequality_solution (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  (a > 1 → ∀ x, a^(2*x-1) > a^(x+2) ↔ x > 3) ∧
  (0 < a ∧ a < 1 → ∀ x, a^(2*x-1) > a^(x+2) ↔ x < 3) := by sorry

-- Problem 2
noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then Real.sqrt x + 1
  else if x < 0 then -Real.sqrt (-x) - 1
  else 0

theorem odd_function_solution :
  (∀ x, f (-x) = -f x) ∧
  (∀ x > 0, f x = Real.sqrt x + 1) ∧
  (∀ x < 0, f x = -Real.sqrt (-x) - 1) ∧
  (f 0 = 0) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_odd_function_solution_l1120_112081


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_max_value_in_interval_l1120_112053

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (Real.sin x + Real.cos x)^2 + 2 * (Real.cos x)^2 - 2

-- Theorem for the smallest positive period
theorem smallest_positive_period :
  ∃ (p : ℝ), p > 0 ∧ (∀ (x : ℝ), f (x + p) = f x) ∧
  (∀ (q : ℝ), q > 0 ∧ (∀ (x : ℝ), f (x + q) = f x) → p ≤ q) ∧
  p = Real.pi := by sorry

-- Theorem for the maximum value in the given interval
theorem max_value_in_interval :
  ∃ (M : ℝ), (∀ (x : ℝ), x ∈ Set.Icc (Real.pi / 4) (3 * Real.pi / 4) → f x ≤ M) ∧
  (∃ (x : ℝ), x ∈ Set.Icc (Real.pi / 4) (3 * Real.pi / 4) ∧ f x = M) ∧
  M = 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_period_max_value_in_interval_l1120_112053


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_count_valid_numbers_l1120_112096

def is_valid_number (n : ℕ) : Bool :=
  1000 ≤ n ∧ n < 10000 ∧ n % 27 = 0 ∧ (n / 1000 + (n / 100 % 10) + (n / 10 % 10) + (n % 10)) % 27 = 0

theorem count_valid_numbers : 
  (Finset.filter (fun n => is_valid_number n = true) (Finset.range 10000)).card = 75 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_count_valid_numbers_l1120_112096


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_white_flowers_count_l1120_112042

/-- Given a set of flowers with specified colors, calculate the number of white flowers. -/
theorem white_flowers_count (total : ℕ) (red : ℕ) (blue_percentage : ℚ) 
  (h_total : total = 10)
  (h_red : red = 4)
  (h_blue_percentage : blue_percentage = 40 / 100) : 
  total - red - (blue_percentage * ↑total).floor = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_white_flowers_count_l1120_112042


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_items_is_ten_l1120_112079

/-- Represents the problem of maximizing item purchases with given constraints. -/
def MaxItemsPurchase (totalMoney sandwichCost drinkCost : ℚ) : ℕ :=
  let maxSandwiches := (totalMoney / sandwichCost).floor.toNat
  let remainingMoney := totalMoney - maxSandwiches * sandwichCost
  let drinks := (remainingMoney / drinkCost).floor.toNat
  maxSandwiches + drinks

/-- Theorem stating that given the specific conditions, the maximum number of items purchased is 10. -/
theorem max_items_is_ten :
  MaxItemsPurchase 40 5 1.5 = 10 := by
  sorry

#eval MaxItemsPurchase 40 5 1.5

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_items_is_ten_l1120_112079


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rationalize_fourth_root_diff_l1120_112013

-- Define the fourth root function
noncomputable def fourth_root (x : ℝ) : ℝ := x ^ (1/4)

-- Define the original expression
noncomputable def original_expr : ℝ := 1 / (fourth_root 4 - fourth_root 2)

-- Define the rationalized form
noncomputable def rationalized_form (X Y Z W D : ℕ) : ℝ :=
  (fourth_root X + fourth_root Y + fourth_root Z + fourth_root W) / D

-- State the theorem
theorem rationalize_fourth_root_diff :
  ∃ X Y Z W D : ℕ,
    rationalized_form X Y Z W D = original_expr ∧
    X + Y + Z + W + D = 134 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rationalize_fourth_root_diff_l1120_112013


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_jelly_bean_difference_l1120_112057

/-- The number of jelly beans Napoleon has -/
def napoleon_beans : ℕ := 17

/-- The number of jelly beans Mikey has -/
def mikey_beans : ℕ := 19

/-- The difference between Sedrich's and Napoleon's jelly beans -/
def difference : ℕ := 4

/-- The number of jelly beans Sedrich has -/
def sedrich_beans : ℕ := napoleon_beans + difference

theorem jelly_bean_difference : difference = 4 := by
  -- We'll use the given equation
  have h : 2 * (napoleon_beans + sedrich_beans) = 4 * mikey_beans := by
    -- Substitute the values and perform the calculation
    calc
      2 * (napoleon_beans + sedrich_beans) = 2 * (17 + (17 + 4)) := by rfl
      _ = 2 * 38 := by rfl
      _ = 76 := by rfl
      _ = 4 * 19 := by rfl
      _ = 4 * mikey_beans := by rfl
  
  -- The theorem is true by definition, so we can use rfl
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_jelly_bean_difference_l1120_112057


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_weekend_coffee_cups_l1120_112049

/-- Proves that the number of coffee cups brewed over the weekend is 120 --/
theorem weekend_coffee_cups : ℕ := by
  let cups_per_hour : ℕ := 10
  let hours_per_day : ℕ := 5
  let weekdays : ℕ := 5
  let total_cups_per_week : ℕ := 370
  let weekday_cups : ℕ := cups_per_hour * hours_per_day * weekdays
  let weekend_cups : ℕ := total_cups_per_week - weekday_cups
  have h : weekend_cups = 120 := by sorry
  exact weekend_cups

end NUMINAMATH_CALUDE_ERRORFEEDBACK_weekend_coffee_cups_l1120_112049


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_condition_iff_has_min_l1120_112059

/-- The function f(x) = x + a/x -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x + a / x

/-- The condition that a is between 1 and 16 -/
def a_condition (a : ℝ) : Prop := 1 < a ∧ a < 16

/-- The property that f has a minimum value in the interval (1, 4) -/
def has_min_in_interval (a : ℝ) : Prop :=
  ∃ x, 1 < x ∧ x < 4 ∧ ∀ y, 1 < y ∧ y < 4 → f a x ≤ f a y

/-- Theorem stating that a_condition is necessary and sufficient for has_min_in_interval -/
theorem a_condition_iff_has_min :
  ∀ a : ℝ, a_condition a ↔ has_min_in_interval a := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_condition_iff_has_min_l1120_112059


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bee_flight_distance_l1120_112082

/-- Represents the complex plane where the bee flies -/
noncomputable def Ξ : ℂ := Complex.exp (Real.pi * Complex.I / 4)

/-- The position of the bee after n steps -/
noncomputable def bee_position : ℕ → ℂ
  | 0 => 0
  | 1 => 2
  | n+2 => bee_position (n+1) + (2*(n+2)) * (Ξ^(n+1))

/-- The theorem stating the distance of the bee from the origin after 1024 steps -/
theorem bee_flight_distance : Complex.abs (bee_position 1024) = 2049 * Real.sqrt 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_bee_flight_distance_l1120_112082


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_more_uniform_scores_smaller_variance_l1120_112028

/-- Represents the variance of a set of scores -/
structure ScoreVariance where
  value : ℝ
  nonneg : 0 ≤ value

/-- Represents a class of students with their score variance -/
structure StudentClass where
  name : String
  variance : ScoreVariance

/-- Defines what it means for one class to have more uniform scores than another -/
def has_more_uniform_scores (c1 c2 : StudentClass) : Prop :=
  c1.variance.value < c2.variance.value

/-- Given two classes with their score variances, proves which class has more uniform scores -/
theorem more_uniform_scores_smaller_variance 
  (class_A class_B : StudentClass)
  (h_A : class_A.name = "A" ∧ class_A.variance.value = 13.2)
  (h_B : class_B.name = "B" ∧ class_B.variance.value = 26.26) :
  has_more_uniform_scores class_A class_B := by
  unfold has_more_uniform_scores
  rw [h_A.2, h_B.2]
  norm_num
  
#check more_uniform_scores_smaller_variance

end NUMINAMATH_CALUDE_ERRORFEEDBACK_more_uniform_scores_smaller_variance_l1120_112028


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l1120_112056

noncomputable def f (x : ℝ) : ℝ := (1/2) * (Real.cos x)^2 + (Real.sqrt 3 / 2) * Real.sin x * Real.cos x + 2

theorem f_range :
  (∀ y, y ∈ Set.range f → 2 ≤ y ∧ y ≤ 11/4) ∧
  (∃ x₁ x₂, x₁ ∈ Set.Icc (-π/6) (π/4) ∧ x₂ ∈ Set.Icc (-π/6) (π/4) ∧ f x₁ = 2 ∧ f x₂ = 11/4) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l1120_112056


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_theta_value_l1120_112033

theorem tan_theta_value (θ : Real) (z : ℂ) 
  (h1 : z = Complex.mk (Real.sin θ - 3/5) (Real.cos θ - 4/5))
  (h2 : Complex.re z = 0)
  (h3 : Complex.im z ≠ 0) : 
  Real.tan θ = -3/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_theta_value_l1120_112033


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_theorem_l1120_112095

/-- The function f(x) defined as |x-a| + |x+b| -/
noncomputable def f (a b x : ℝ) : ℝ := |x - a| + |x + b|

/-- The expression we want to minimize -/
noncomputable def g (a b : ℝ) : ℝ := a^2 / b + b^2 / a

theorem min_value_theorem (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∀ x, f a b x ≥ 3) → (∃ x, f a b x = 3) → 
  (∀ a' b', a' > 0 → b' > 0 → g a' b' ≥ 3) ∧ (g a b = 3) := by
  sorry

#check min_value_theorem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_theorem_l1120_112095


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_b_2023_value_b_general_form_l1120_112064

-- Define the sequence b_n
def b : ℕ → ℚ
  | 0 => 2  -- We define b_0 as 2 to match b_1 in the original problem
  | 1 => 4/9
  | (n+2) => (b n * b (n+1)) / (3 * b n - b (n+1))

-- State the theorem
theorem b_2023_value : b 2023 = 8/8092 := by
  sorry

-- Additional theorem to prove the general form
theorem b_general_form (n : ℕ) : n ≥ 1 → b n = 8 / (4 * n) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_b_2023_value_b_general_form_l1120_112064


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_double_angle_relation_l1120_112070

/-- Given an angle x where tan x = b/a and tan 2x = b/(2a+b), 
    prove that the least positive value of x equals tan^(-1) (b/a) -/
theorem tan_double_angle_relation (a b : ℝ) (x : ℝ) 
  (h1 : Real.tan x = b / a) 
  (h2 : Real.tan (2 * x) = b / (2 * a + b)) :
  x = Real.arctan (b / a) ∧ x > 0 ∧ ∀ y, (y > 0 ∧ Real.tan y = b / a) → x ≤ y :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_double_angle_relation_l1120_112070


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pizza_earnings_theorem_l1120_112035

/-- Represents the types of pizzas --/
inductive PizzaType
| Margherita
| Pepperoni
| VeggieSupreme

/-- Represents the pricing structure for a pizza type --/
structure PizzaPricing where
  slicePrice : ℚ
  wholePrice : ℚ

/-- Calculates the total price for a given number of slices --/
def sliceTotal (pricing : PizzaPricing) (slices : ℕ) : ℚ :=
  pricing.slicePrice * slices

/-- Calculates the total price for a given number of whole pizzas --/
def wholeTotal (pricing : PizzaPricing) (wholes : ℕ) : ℚ :=
  pricing.wholePrice * wholes

/-- Applies a discount to a given price --/
def applyDiscount (price : ℚ) (discountPercent : ℚ) : ℚ :=
  price * (1 - discountPercent / 100)

/-- Theorem stating that the total earnings are equal to $264.50 --/
theorem pizza_earnings_theorem 
  (pricings : PizzaType → PizzaPricing)
  (margheritaSlices pepperoniSlices veggieSupremeWholes margheritaWholes : ℕ)
  (discountPercent : ℚ) :
  (pricings PizzaType.Margherita).slicePrice = 3 →
  (pricings PizzaType.Pepperoni).slicePrice = 4 →
  (pricings PizzaType.VeggieSupreme).wholePrice = 22 →
  (pricings PizzaType.Margherita).wholePrice = 15 →
  margheritaSlices = 24 →
  pepperoniSlices = 16 →
  veggieSupremeWholes = 4 →
  margheritaWholes = 3 →
  discountPercent = 10 →
  sliceTotal (pricings PizzaType.Margherita) margheritaSlices +
  sliceTotal (pricings PizzaType.Pepperoni) pepperoniSlices +
  wholeTotal (pricings PizzaType.VeggieSupreme) veggieSupremeWholes +
  applyDiscount (wholeTotal (pricings PizzaType.Margherita) margheritaWholes) discountPercent =
  264.50 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pizza_earnings_theorem_l1120_112035


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_two_points_closer_than_radius_l1120_112065

-- Define a circular disk
def CircularDisk : Set (ℝ × ℝ) := {p | p.1^2 + p.2^2 ≤ 1}

-- Define a set of 8 points on the disk
structure PointsOnDisk where
  points : Finset (ℝ × ℝ)
  subset : points.toSet ⊆ CircularDisk
  card : points.card = 8

-- Theorem statement
theorem exists_two_points_closer_than_radius (S : PointsOnDisk) :
  ∃ p q : ℝ × ℝ, p ∈ S.points ∧ q ∈ S.points ∧ p ≠ q ∧ 
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2) < 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_two_points_closer_than_radius_l1120_112065


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_sides_formula_limit_expected_sides_is_four_rectangle_expected_sides_close_to_four_l1120_112044

/-- Represents the number of sides of the initial polygon -/
def n : ℕ := 4

/-- Represents the number of cuts made -/
def k : ℕ := 3600  -- Assuming one cut per second for an hour

/-- The expected number of sides after k cuts -/
def expected_sides (n k : ℕ) : ℚ :=
  (n + 4*k : ℚ) / (k + 1 : ℚ)

/-- The limit of the expected number of sides as k approaches infinity -/
def limit_expected_sides (n : ℕ) : ℚ := 4

/-- Theorem stating the expected number of sides after k cuts -/
theorem expected_sides_formula (n k : ℕ) :
  expected_sides n k = (n + 4*k : ℚ) / (k + 1 : ℚ) :=
by
  rfl  -- reflexivity

/-- Theorem stating that the limit of expected sides approaches 4 as k approaches infinity -/
theorem limit_expected_sides_is_four (n : ℕ) :
  ∀ ε > 0, ∃ K, ∀ k ≥ K, |expected_sides n k - 4| < ε :=
by
  sorry

/-- The expected number of sides for a rectangle after an hour of cutting -/
def rectangle_expected_sides : ℚ :=
  expected_sides n k

/-- Theorem stating that the expected number of sides for a rectangle after an hour is close to 4 -/
theorem rectangle_expected_sides_close_to_four :
  |rectangle_expected_sides - 4| < 0.01 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expected_sides_formula_limit_expected_sides_is_four_rectangle_expected_sides_close_to_four_l1120_112044


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_conic_section_constant_ratio_l1120_112092

-- Define a conic section (ellipse, hyperbola, or parabola)
structure ConicSection where
  -- The set of points that form the conic section
  points : Set (ℝ × ℝ)
  -- The fixed point (focus)
  focus : ℝ × ℝ
  -- The fixed line (directrix)
  directrix : ℝ → ℝ
  -- The eccentricity (constant ratio)
  eccentricity : ℝ

-- Define the distance between two points
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Define the distance from a point to a line
noncomputable def distanceToLine (p : ℝ × ℝ) (line : ℝ → ℝ) : ℝ :=
  |p.2 - line p.1|

-- Theorem stating that for any conic section, the ratio of distances is constant
theorem conic_section_constant_ratio (c : ConicSection) :
  ∀ p ∈ c.points, (distance p c.focus) / (distanceToLine p c.directrix) = c.eccentricity := by
  sorry

-- Definitions for specific conic sections
def Ellipse := {c : ConicSection // c.eccentricity < 1}
def Hyperbola := {c : ConicSection // c.eccentricity > 1}
def Parabola := {c : ConicSection // c.eccentricity = 1}

end NUMINAMATH_CALUDE_ERRORFEEDBACK_conic_section_constant_ratio_l1120_112092


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_bounds_l1120_112072

/-- The function g defined for positive real numbers -/
noncomputable def g (x y z : ℝ) : ℝ := x^2 / (x^2 + y^2) + y^2 / (y^2 + z^2) + z^2 / (z^2 + x^2)

/-- Theorem stating the bounds of g for positive real inputs -/
theorem g_bounds (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  3/2 ≤ g x y z ∧ g x y z ≤ 1.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_bounds_l1120_112072


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hidden_number_determination_l1120_112086

/-- Represents a card with two numbers -/
structure Card where
  front : ℕ
  back : ℕ

/-- The set of all cards -/
def card_set (n : ℕ) : Set Card :=
  {c | ∃ i, 0 < i ∧ i ≤ n ∧ c.front = i - 1 ∧ c.back = i}

/-- The set of numbers shown -/
def shown_numbers (shown : Set Card) (sides : Card → ℕ) : Set ℕ :=
  {n | ∃ c ∈ shown, sides c = n}

/-- Predicate for determining if the hidden number can be deduced -/
def can_determine_hidden (n : ℕ) (shown : Set Card) (sides : Card → ℕ) (k : ℕ) : Prop :=
  (∀ m, k ≤ m ∧ m ≤ n → m ∈ shown_numbers shown sides) ∨
  (∀ m, 0 ≤ m ∧ m ≤ k → m ∈ shown_numbers shown sides) ∨
  (∃ l, l ≠ k ∧
    (∀ m, (k < m ∧ m < l) ∨ (l < m ∧ m < k) → m ∈ shown_numbers shown sides) ∧
    (∃ c1 c2, c1 ∈ shown ∧ c2 ∈ shown ∧ c1 ≠ c2 ∧ sides c1 = l ∧ sides c2 = l))

theorem hidden_number_determination (n : ℕ) (shown : Set Card) (sides : Card → ℕ) (k : ℕ) :
  k ∈ shown_numbers shown sides →
  (∃ c ∈ card_set n, sides c = k) →
  can_determine_hidden n shown sides k ↔
    (∃! m, m ≠ k ∧ ∃ c ∈ card_set n, c ∈ shown ∧ sides c = k ∧ (c.front = m ∨ c.back = m)) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hidden_number_determination_l1120_112086


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_table_properties_l1120_112007

/-- Represents the infinite table of integers as described in the problem -/
def table_term (m n : ℕ) : ℤ :=
  2 * (Int.ofNat m) * (Int.ofNat n) + (Int.ofNat m) + (Int.ofNat n)

/-- Theorem stating the properties of the table -/
theorem table_properties :
  (∀ m n : ℕ, table_term m n = 2 * (Int.ofNat m) * (Int.ofNat n) + (Int.ofNat m) + (Int.ofNat n)) ∧
  (table_term 6 153 = 1995 ∧ table_term 153 6 = 1995) ∧
  (∀ m n : ℕ, table_term m n ≠ 1994) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_table_properties_l1120_112007


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_triangle_area_l1120_112008

/-- The value of 'a' in the equation y = a ln(x) that satisfies the given conditions -/
theorem tangent_triangle_area (a : ℝ) (h1 : a > 0) : 
  (∃ f : ℝ → ℝ, f = λ x ↦ a * Real.log x) →
  (∃ t : ℝ → ℝ, t = λ x ↦ a * (x - 1)) →
  (∃ A B : ℝ × ℝ, A = (1, 0) ∧ B = (0, -a)) →
  ((1/2) * 1 * a = 4) →
  a = 8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_triangle_area_l1120_112008


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_through_point_l1120_112074

-- Define the power function as noncomputable
noncomputable def powerFunction (a : ℝ) : ℝ → ℝ := fun x ↦ x ^ a

-- State the theorem
theorem power_function_through_point (a : ℝ) :
  powerFunction a 2 = Real.sqrt 2 → a = 1/2 := by
  intro h
  -- The proof steps would go here, but we'll use sorry for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_through_point_l1120_112074


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_consecutive_cube_sums_to_squares_l1120_112051

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m^2

def sum_of_cubes (start finish : ℕ) : ℕ :=
  List.range (finish - start + 1) |>.map (fun i => (start + i)^3) |>.sum

theorem smallest_consecutive_cube_sums_to_squares :
  (∀ start : ℕ, start > 1 → start < 14 →
    ¬(is_perfect_square (sum_of_cubes start (start + 3)))) ∧
  (is_perfect_square (sum_of_cubes 14 25)) ∧
  (is_perfect_square (sum_of_cubes 25 29)) ∧
  (∀ start finish : ℕ, start > 1 → finish - start ≥ 3 →
    (start < 14 ∨ (start = 14 ∧ finish < 25) ∨ (start > 14 ∧ start < 25) ∨ (start = 25 ∧ finish < 29)) →
    ¬(is_perfect_square (sum_of_cubes start finish))) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_consecutive_cube_sums_to_squares_l1120_112051


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezium_other_side_length_l1120_112031

/-- The area of a trapezium given the lengths of its parallel sides and the distance between them. -/
noncomputable def trapeziumArea (a b h : ℝ) : ℝ := (a + b) * h / 2

/-- Theorem: Given a trapezium with one parallel side of 20 cm, a distance of 10 cm between parallel sides,
    and an area of 150 square cm, the length of the other parallel side is 10 cm. -/
theorem trapezium_other_side_length :
  ∀ x : ℝ,
  trapeziumArea 20 x 10 = 150 →
  x = 10 := by
  intro x h
  unfold trapeziumArea at h
  -- The proof steps would go here, but we'll use sorry for now
  sorry

#check trapezium_other_side_length

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trapezium_other_side_length_l1120_112031


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_number_with_special_property_l1120_112014

theorem smallest_number_with_special_property : ∃ N : ℕ,
  (N = 153846) ∧
  (N % 10 = 6) ∧
  (∀ k : ℕ, k < N → k % 10 = 6 → 4 * k ≠ (6 * 10^(Nat.digits 10 k).length + k / 10)) ∧
  (4 * N = 6 * 10^(Nat.digits 10 N).length + N / 10) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_number_with_special_property_l1120_112014


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_profit_is_8_yuan_l1120_112003

/-- Represents the price of a calculator in yuan -/
noncomputable def calculator_price : ℚ := 64

/-- Represents the profit percentage for the first calculator -/
noncomputable def profit_percentage : ℚ := 60 / 100

/-- Represents the loss percentage for the second calculator -/
noncomputable def loss_percentage : ℚ := 20 / 100

/-- Calculates the purchase price of a calculator given its selling price and profit/loss percentage -/
noncomputable def purchase_price (selling_price : ℚ) (percentage : ℚ) (is_profit : Bool) : ℚ :=
  if is_profit then
    selling_price / (1 + percentage)
  else
    selling_price / (1 - percentage)

/-- Theorem stating that the total profit from selling two calculators is 8 yuan -/
theorem total_profit_is_8_yuan :
  let purchase_price1 := purchase_price calculator_price profit_percentage true
  let purchase_price2 := purchase_price calculator_price loss_percentage false
  let total_revenue := 2 * calculator_price
  let total_cost := purchase_price1 + purchase_price2
  total_revenue - total_cost = 8 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_profit_is_8_yuan_l1120_112003


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_circle_is_cartesian_circle_l1120_112097

/-- A point in polar coordinates -/
structure PolarPoint where
  r : ℝ
  θ : ℝ

/-- Conversion from polar to Cartesian coordinates -/
noncomputable def polarToCartesian (p : PolarPoint) : ℝ × ℝ :=
  (p.r * Real.cos p.θ, p.r * Real.sin p.θ)

/-- The set of points satisfying r = 2 in polar coordinates -/
def polarCircle : Set PolarPoint :=
  {p : PolarPoint | p.r = 2}

/-- The set of points on a circle with radius 2 centered at the origin in Cartesian coordinates -/
def cartesianCircle : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 + p.2^2 = 4}

theorem polar_circle_is_cartesian_circle :
  {polarToCartesian p | p ∈ polarCircle} = cartesianCircle := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_polar_circle_is_cartesian_circle_l1120_112097


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_digits_l1120_112073

def a : Nat := 3659893456789325678
def b : Nat := 342973489379256

def num_digits (n : Nat) : Nat :=
  if n = 0 then 1 else Nat.size n

theorem product_digits :
  num_digits (a * b) = 34 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_digits_l1120_112073


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shooting_probabilities_l1120_112024

def prob_A : ℚ := 1/2
def prob_B : ℚ := 2/3
def num_trials : ℕ := 3

theorem shooting_probabilities :
  let at_least_two_B := (Nat.choose num_trials 2 * prob_B^2 * (1 - prob_B) + Nat.choose num_trials 3 * prob_B^3 : ℚ)
  let exactly_two_more_B := 
    (Nat.choose num_trials 2 * prob_B^2 * (1 - prob_B) * Nat.choose num_trials 0 * prob_A^3 +
     Nat.choose num_trials 3 * prob_B^3 * Nat.choose num_trials 1 * prob_A^2 * (1 - prob_A) : ℚ)
  at_least_two_B = 20/27 ∧ exactly_two_more_B = 1/6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_shooting_probabilities_l1120_112024


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l1120_112075

-- Define the function
noncomputable def f (x : ℝ) : ℝ := x - Real.sqrt (1 - 4*x)

-- State the theorem
theorem f_range :
  ∀ y : ℝ, (∃ x : ℝ, x ≤ (1/4 : ℝ) ∧ f x = y) ↔ y ≤ (1/4 : ℝ) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_range_l1120_112075


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_and_max_area_l1120_112043

/-- Define an ellipse C with given properties -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_ab : a > b
  h_b_pos : b > 0
  h_point : b = 1
  h_eccentricity : (a^2 - b^2) / a^2 = 1/2

/-- Define the line intersecting the ellipse -/
def intersecting_line (k : ℝ) (x : ℝ) : ℝ := k * x + 1

/-- The equation of the ellipse C -/
def ellipse_equation (x y : ℝ) : Prop :=
  x^2 / 2 + y^2 = 1

/-- The area of triangle OAB -/
noncomputable def triangle_area (k : ℝ) : ℝ :=
  |k| / ((1/2) + k^2)

/-- Main theorem stating the properties of the ellipse and the maximum area -/
theorem ellipse_properties_and_max_area (C : Ellipse) :
  (∀ x y, ellipse_equation x y ↔ x^2 / C.a^2 + y^2 / C.b^2 = 1) ∧
  (∃ k, triangle_area k = Real.sqrt 2 / 2) ∧
  (∀ k, triangle_area k ≤ Real.sqrt 2 / 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_and_max_area_l1120_112043


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_transformation_l1120_112098

theorem inequality_transformation (a b c : ℝ) 
  (h : ∀ x : ℝ, x ∈ Set.Icc (-1) 1 → |a * x^2 + b * x + c| ≤ 1) :
  ∀ x : ℝ, x ∈ Set.Icc (-1) 1 → |c * x^2 + b * x + a| ≤ 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_transformation_l1120_112098


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_construction_theorem_l1120_112021

/-- A triangle with a median and two circumcircle radii -/
structure TriangleWithMedianAndCircumcircles where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ
  F : ℝ × ℝ
  s_c : ℝ
  r_1 : ℝ
  r_2 : ℝ
  is_median : F = ((A.1 + B.1) / 2, (A.2 + B.2) / 2)
  median_length : Real.sqrt ((C.1 - F.1)^2 + (C.2 - F.2)^2) = s_c
  acf_circumradius : ∃ O₁ : ℝ × ℝ, Real.sqrt ((A.1 - O₁.1)^2 + (A.2 - O₁.2)^2) = r_1 ∧
                                   Real.sqrt ((C.1 - O₁.1)^2 + (C.2 - O₁.2)^2) = r_1 ∧
                                   Real.sqrt ((F.1 - O₁.1)^2 + (F.2 - O₁.2)^2) = r_1
  bcf_circumradius : ∃ O₂ : ℝ × ℝ, Real.sqrt ((B.1 - O₂.1)^2 + (B.2 - O₂.2)^2) = r_2 ∧
                                   Real.sqrt ((C.1 - O₂.1)^2 + (C.2 - O₂.2)^2) = r_2 ∧
                                   Real.sqrt ((F.1 - O₂.1)^2 + (F.2 - O₂.2)^2) = r_2

/-- There exists a triangle with the given median length and circumcircle radii -/
theorem triangle_construction_theorem (s_c r_1 r_2 : ℝ) (h1 : r_1 ≥ s_c / 2) (h2 : r_2 > s_c / 2) :
  ∃ t : TriangleWithMedianAndCircumcircles, t.s_c = s_c ∧ t.r_1 = r_1 ∧ t.r_2 = r_2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_construction_theorem_l1120_112021


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tunnel_length_l1120_112017

/-- Calculates the length of a tunnel given train speed, entry and exit times, and train length -/
theorem tunnel_length
  (train_speed : ℝ)
  (entry_time exit_time : ℕ)
  (train_length : ℝ)
  (h1 : train_speed = 80)
  (h2 : entry_time = 5 * 60 + 12)  -- 5:12 am in minutes
  (h3 : exit_time = 5 * 60 + 18)   -- 5:18 am in minutes
  (h4 : train_length = 1) :
  train_speed * ((exit_time - entry_time : ℝ) / 60) - train_length = 7 := by
  sorry

#check tunnel_length

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tunnel_length_l1120_112017


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sandbox_volume_calculation_l1120_112076

/-- Represents the dimensions of a trapezoidal sandbox -/
structure SandboxDimensions where
  length : ℝ
  maxWidth : ℝ
  minWidth : ℝ
  depth : ℝ

/-- Calculates the volume of a trapezoidal sandbox -/
noncomputable def sandboxVolume (d : SandboxDimensions) : ℝ :=
  (d.length * (d.maxWidth + d.minWidth) / 2) * d.depth

/-- Theorem stating the volume of the specific sandbox -/
theorem sandbox_volume_calculation :
  let dimensions : SandboxDimensions := {
    length := 312,
    maxWidth := 146,
    minWidth := 85,
    depth := 56
  }
  sandboxVolume dimensions = 2018016 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sandbox_volume_calculation_l1120_112076


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_propositions_truth_l1120_112025

-- Define the propositions
def proposition1 : Prop := ∀ m : ℝ, (∀ x : ℝ, x^2 + 2*x - m ≠ 0) → m ≤ -1

def proposition2 : Prop := (∀ x : ℝ, x = 1 → x^2 - 3*x + 2 = 0) ∧
                           ¬(∀ x : ℝ, x^2 - 3*x + 2 = 0 → x = 1)

-- Define auxiliary concepts
class Quadrilateral where
  -- Add any necessary fields or methods here

def DiagonalsEqual (q : Quadrilateral) : Prop :=
  sorry -- Define this properly based on your requirements

def IsRectangle (q : Quadrilateral) : Prop :=
  sorry -- Define this properly based on your requirements

def proposition3 : Prop := ∀ q : Quadrilateral, DiagonalsEqual q → IsRectangle q

def proposition4 : Prop := ∀ x y : ℝ, x = 0 ∧ y = 0 → x^2 + y^2 = 0

-- Theorem statement
theorem propositions_truth : 
  proposition1 ∧ proposition2 ∧ ¬proposition3 ∧ proposition4 := by
  sorry -- The proof goes here

end NUMINAMATH_CALUDE_ERRORFEEDBACK_propositions_truth_l1120_112025


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_commute_additional_time_l1120_112062

/-- Proves that the additional time spent on walking to/from stations and waiting for the train is 0.5 minutes -/
theorem train_commute_additional_time 
  (distance_to_work : ℝ) 
  (walking_speed : ℝ) 
  (train_speed : ℝ) 
  (walking_time_difference : ℝ) 
  (h1 : distance_to_work = 1.5)
  (h2 : walking_speed = 3)
  (h3 : train_speed = 20)
  (h4 : walking_time_difference = 25)
  : (distance_to_work / walking_speed * 60) = 
    (distance_to_work / train_speed * 60 + walking_time_difference + 0.5) := by
  sorry

#check train_commute_additional_time

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_commute_additional_time_l1120_112062


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_length_DE_l1120_112071

noncomputable section

structure Triangle :=
  (A B C : ℝ × ℝ)

noncomputable def Triangle.AB (t : Triangle) : ℝ := Real.sqrt ((t.A.1 - t.B.1)^2 + (t.A.2 - t.B.2)^2)
noncomputable def Triangle.BC (t : Triangle) : ℝ := Real.sqrt ((t.B.1 - t.C.1)^2 + (t.B.2 - t.C.2)^2)
noncomputable def Triangle.AC (t : Triangle) : ℝ := Real.sqrt ((t.A.1 - t.C.1)^2 + (t.A.2 - t.C.2)^2)

def Point := ℝ × ℝ

def parallel (p1 p2 p3 p4 : Point) : Prop :=
  (p2.2 - p1.2) * (p4.1 - p3.1) = (p2.1 - p1.1) * (p4.2 - p3.2)

def ratio (p1 p2 p3 : Point) (r : ℝ) : Prop :=
  r * (p2.1 - p1.1) = (p3.1 - p2.1) ∧ r * (p2.2 - p1.2) = (p3.2 - p2.2)

theorem length_DE (t : Triangle) (D E : Point) :
  t.AB = 24 →
  t.BC = 26 →
  t.AC = 28 →
  ratio t.A D t.B 3 →
  ratio t.A E t.C 3 →
  parallel D E t.B t.C →
  Real.sqrt ((D.1 - E.1)^2 + (D.2 - E.2)^2) = 19.5 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_length_DE_l1120_112071


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_g_l1120_112040

noncomputable def f (x : ℝ) : ℝ := 3 * Real.sin (x / 2) - 2 * Real.cos (x / 2)

noncomputable def g (x : ℝ) : ℝ := f (x - Real.pi / 3)

theorem max_value_of_g (θ : ℝ) (h : ∀ x, g x ≤ g θ) : 
  Real.cos (θ + Real.pi / 6) = 12 / 13 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_g_l1120_112040


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_stratified_sampling_female_athletes_l1120_112050

theorem stratified_sampling_female_athletes 
  (total_athletes : ℕ) 
  (female_athletes : ℕ) 
  (male_athletes : ℕ) 
  (sample_size : ℕ) 
  (h1 : total_athletes = female_athletes + male_athletes)
  (h2 : total_athletes = 77)
  (h3 : female_athletes = 33)
  (h4 : male_athletes = 44)
  (h5 : sample_size = 14) :
  Int.floor ((female_athletes : ℚ) / total_athletes * sample_size) = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_stratified_sampling_female_athletes_l1120_112050


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_nSn_l1120_112094

noncomputable def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

noncomputable def sum_arithmetic_sequence (a : ℕ → ℝ) (n : ℕ) : ℝ :=
  (n : ℝ) * (a 1 + a n) / 2

noncomputable def S (a : ℕ → ℝ) (n : ℕ) : ℝ := sum_arithmetic_sequence a n

theorem min_value_nSn (a : ℕ → ℝ) :
  arithmetic_sequence a →
  S a 10 = 0 →
  S a 15 = 25 →
  ∃ n : ℕ, ∀ m : ℕ, (m : ℝ) * S a m ≥ (n : ℝ) * S a n ∧ (n : ℝ) * S a n = -20 :=
by
  sorry

#check min_value_nSn

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_nSn_l1120_112094


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_percentage_male_advanced_degrees_l1120_112078

theorem percentage_male_advanced_degrees 
  (total_male : ℕ) 
  (total_female : ℕ) 
  (female_advanced_percentage : ℚ) 
  (prob_advanced_or_female : ℚ) :
  total_male = 300 →
  total_female = 150 →
  female_advanced_percentage = 2/5 →
  prob_advanced_or_female = 2/5 →
  let total_employees := total_male + total_female
  let prob_female := total_female / total_employees
  let prob_male_advanced := λ (x : ℚ) ↦ (x * total_male) / total_employees
  ∃ x : ℚ, 
    x = 3/20 ∧ 
    prob_female + prob_male_advanced x = prob_advanced_or_female := by
  sorry

#eval "Theorem statement compiled successfully"

end NUMINAMATH_CALUDE_ERRORFEEDBACK_percentage_male_advanced_degrees_l1120_112078


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_logarithmic_expression_equals_negative_one_l1120_112041

-- Define the expression
noncomputable def logarithmic_expression : ℝ :=
  2 * (Real.log 2 / Real.log 3) - (Real.log (32 / 9) / Real.log 3) + (Real.log 8 / Real.log 3) - (5 ^ (Real.log 3 / Real.log 5))

-- State the theorem
theorem logarithmic_expression_equals_negative_one :
  logarithmic_expression = -1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_logarithmic_expression_equals_negative_one_l1120_112041


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_distance_l1120_112045

-- Define the circle
def is_on_circle (x y : ℝ) : Prop := (x - 4)^2 + (y - 3)^2 = 9

-- Define the parabola
def is_on_parabola (x y : ℝ) : Prop := y^2 = -8*x

-- Define the distance between two points
noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ := Real.sqrt ((x1 - x2)^2 + (y1 - y2)^2)

-- Theorem statement
theorem smallest_distance :
  ∃ (d : ℝ), d = 2 ∧
  ∀ (x1 y1 x2 y2 : ℝ),
    is_on_circle x1 y1 → is_on_parabola x2 y2 →
    distance x1 y1 x2 y2 ≥ d :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_distance_l1120_112045


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_logarithmic_expression_equality_l1120_112034

theorem logarithmic_expression_equality : 
  2 * (Real.log 8 / Real.log 2) + (Real.log 0.01 / Real.log 10) - (Real.log (1/8) / Real.log 2) + (0.01 : Real)^((-1/2) : Real) = 25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_logarithmic_expression_equality_l1120_112034


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_properties_l1120_112001

/-- The parabola and related geometric constructions -/
structure ParabolaConfig where
  p : ℕ
  hp : Nat.Prime p
  hp2 : p ≠ 2

/-- The locus of point R -/
def Locus (config : ParabolaConfig) : Set (ℝ × ℝ) :=
  {point : ℝ × ℝ | 4 * point.2^2 = config.p * (point.1 - config.p) ∧ point.2 ≠ 0}

/-- Main theorem about the locus -/
theorem locus_properties (config : ParabolaConfig) :
  -- The locus contains infinitely many integer points
  (∃ f : ℕ → ℤ × ℤ, Function.Injective f ∧ ∀ n, (↑(f n).1, ↑(f n).2) ∈ Locus config) ∧
  -- The distance from any integer point on the locus to the origin is not an integer
  (∀ x y : ℤ, (↑x, ↑y) ∈ Locus config → ¬∃ m : ℤ, x^2 + y^2 = m^2) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_properties_l1120_112001


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_such_function_exists_l1120_112080

theorem no_such_function_exists :
  ¬∃ (f : ℕ → ℕ), ∀ (n : ℕ), n ≥ 2 → f (f (n - 1)) = f (n + 1) - f (n) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_such_function_exists_l1120_112080


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_properties_l1120_112085

/-- Represents a geometric sequence with first term a₁ and common ratio q -/
structure GeometricSequence where
  a₁ : ℝ
  q : ℝ

/-- The n-th term of a geometric sequence -/
noncomputable def nthTerm (g : GeometricSequence) (n : ℕ) : ℝ :=
  g.a₁ * g.q ^ (n - 1)

/-- The sum of the first n terms of a geometric sequence -/
noncomputable def sumFirstNTerms (g : GeometricSequence) (n : ℕ) : ℝ :=
  g.a₁ * (1 - g.q^n) / (1 - g.q)

theorem geometric_sequence_properties (g : GeometricSequence) 
    (h1 : g.a₁ = 6)
    (h2 : nthTerm g 2 = 12) :
  (∀ n : ℕ, nthTerm g n = 3 * 2^n) ∧
  (∀ n : ℕ, sumFirstNTerms g n = 3 * 2^(n+1) - 6) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_properties_l1120_112085


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_problem_solution_l1120_112011

/-- Simple interest calculation --/
noncomputable def simple_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * rate * time / 100

/-- Compound interest calculation --/
noncomputable def compound_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * ((1 + rate / 100) ^ time - 1)

/-- Theorem stating the solution to the investment problem --/
theorem investment_problem_solution :
  ∃ (x y : ℝ),
    simple_interest x y 2 = 800 ∧
    compound_interest x y 2 = 820 ∧
    x = 8000 := by
  sorry

#check investment_problem_solution

end NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_problem_solution_l1120_112011


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_from_sin_second_quadrant_l1120_112019

theorem cos_from_sin_second_quadrant (α : ℝ) (h1 : α ∈ Set.Ioo (π / 2) π) (h2 : Real.sin α = 5 / 13) : 
  Real.cos α = -12 / 13 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_from_sin_second_quadrant_l1120_112019


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_savings_amount_is_correct_l1120_112063

/-- The amount Collin needs to put into savings in US dollars -/
noncomputable def savings_amount : ℚ :=
  let lightweight_value : ℚ := 15/100
  let medium_value : ℚ := 25/100
  let heavyweight_value : ℚ := 35/100
  let home_lightweight : ℕ := 12
  let grandparents_medium : ℕ := 3 * home_lightweight
  let neighbor_heavyweight : ℕ := 46
  let dad_total : ℕ := 250
  let dad_lightweight_ratio : ℚ := 1/2
  let dad_medium_ratio : ℚ := 3/10
  let dad_heavyweight_ratio : ℚ := 1/5
  let exchange_rate : ℚ := 6/5
  let total_euros : ℚ := 
    lightweight_value * (home_lightweight + dad_lightweight_ratio * dad_total) +
    medium_value * (grandparents_medium + dad_medium_ratio * dad_total) +
    heavyweight_value * (neighbor_heavyweight + dad_heavyweight_ratio * dad_total)
  (total_euros * exchange_rate) / 2

/-- Theorem stating that the calculated savings amount is correct -/
theorem savings_amount_is_correct : savings_amount = 2457/50 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_savings_amount_is_correct_l1120_112063


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_paint_cost_per_quart_l1120_112046

def paint_coverage : ℝ := 60
def cube_edge : ℝ := 10
def total_paint_cost : ℝ := 32

def cube_surface_area (edge : ℝ) : ℝ := 6 * edge^2

theorem paint_cost_per_quart :
  (total_paint_cost / (cube_surface_area cube_edge / paint_coverage)) = 3.20 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_paint_cost_per_quart_l1120_112046


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_tangent_length_l1120_112091

/-- Two circles in a 2D plane -/
structure TwoCircles where
  C₁ : (ℝ × ℝ) → Prop
  C₂ : (ℝ × ℝ) → Prop

/-- The given circles from the problem -/
def given_circles : TwoCircles where
  C₁ := λ (x, y) ↦ (x - 12)^2 + y^2 = 49
  C₂ := λ (x, y) ↦ (x + 18)^2 + y^2 = 64

/-- A line segment between two points -/
noncomputable def LineSegment (P Q : ℝ × ℝ) : ℝ := Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)

/-- Predicate to check if a line segment PQ is tangent to both circles -/
def is_tangent_to_both (circles : TwoCircles) (P Q : ℝ × ℝ) : Prop :=
  circles.C₁ P ∧ circles.C₂ Q ∧ 
  ∀ R, circles.C₁ R → LineSegment P R ≥ LineSegment P Q ∧
  ∀ S, circles.C₂ S → LineSegment Q S ≥ LineSegment P Q

/-- The main theorem -/
theorem shortest_tangent_length (circles : TwoCircles) :
  circles = given_circles →
  ∃ P Q : ℝ × ℝ, is_tangent_to_both circles P Q ∧
    ∀ R S : ℝ × ℝ, is_tangent_to_both circles R S →
      LineSegment P Q ≤ LineSegment R S ∧
      LineSegment P Q = 20 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_shortest_tangent_length_l1120_112091


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_segment_endpoint_l1120_112089

/-- The y-coordinate of the endpoint of a line segment -/
noncomputable def endpoint_y (x1 y1 x2 : ℝ) (length : ℝ) : ℝ :=
  y1 + Real.sqrt (length ^ 2 - (x2 - x1) ^ 2)

/-- Theorem: The y-coordinate of the endpoint of a line segment -/
theorem line_segment_endpoint
  (x1 y1 x2 : ℝ)
  (length : ℝ)
  (h1 : x1 = 1)
  (h2 : y1 = 3)
  (h3 : x2 = 7)
  (h4 : length = 13)
  (h5 : endpoint_y x1 y1 x2 length > 0) :
  endpoint_y x1 y1 x2 length = 3 + Real.sqrt 133 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_segment_endpoint_l1120_112089


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_two_thirds_pi_minus_two_alpha_l1120_112010

theorem cos_two_thirds_pi_minus_two_alpha (α : ℝ) : 
  Real.sin (π / 6 + α) = 1 / 3 → Real.cos (2 * π / 3 - 2 * α) = -7 / 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_two_thirds_pi_minus_two_alpha_l1120_112010


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_p_with_product_triple_l1120_112084

def T (p : ℕ) : Set ℕ := {n : ℕ | 5 ≤ n ∧ n ≤ p}

def hasProductTriple (S : Set ℕ) : Prop :=
  ∃ a b c, a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ a * b = c

theorem smallest_p_with_product_triple :
  ∀ p ≥ 5, (∀ A B : Set ℕ, A ∪ B = T p → A ∩ B = ∅ → hasProductTriple A ∨ hasProductTriple B) ↔ p ≥ 625 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_p_with_product_triple_l1120_112084


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_l_properties_l1120_112087

noncomputable def line_l (k : ℝ) (x y : ℝ) : Prop := k * x - y + 1 + 2 * k = 0

noncomputable def triangle_area (k : ℝ) : ℝ := 4 * k + 1 / k + 4

theorem line_l_properties :
  ∀ k : ℝ,
  (∀ x y : ℝ, line_l k x y → x = -2 ∧ y = 1) ∧
  (∀ S : ℝ, S = triangle_area k → S ≥ 4) ∧
  (∃ k₀ : ℝ, triangle_area k₀ = 4 ∧ ∀ x y : ℝ, line_l k₀ x y ↔ x - 2 * y + 4 = 0) :=
by
  sorry

#check line_l_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_l_properties_l1120_112087


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_always_positive_implies_a_gt_four_l1120_112083

/-- The function f(x) defined in the problem -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (9 : ℝ)^x - 2 * (3 : ℝ)^x + a - 3

/-- Theorem stating that if f(x) is always positive, then a > 4 -/
theorem f_always_positive_implies_a_gt_four (a : ℝ) :
  (∀ x : ℝ, f a x > 0) → a > 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_always_positive_implies_a_gt_four_l1120_112083


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_ghcd_l1120_112006

/-- Represents a trapezoid ABCD with parallel sides AB and CD -/
structure Trapezoid where
  ab : ℝ
  cd : ℝ
  altitude : ℝ

/-- Represents the quadrilateral GHCD within the trapezoid ABCD -/
noncomputable def TrapezoidGHCD (t : Trapezoid) : ℝ :=
  let gh := (t.ab + t.cd) / 2
  let altitude_ghcd := t.altitude / 2
  (gh + t.cd) * altitude_ghcd / 2

/-- The main theorem stating the area of GHCD in the given trapezoid -/
theorem area_of_ghcd (t : Trapezoid) (h_ab : t.ab = 10) (h_cd : t.cd = 24) (h_altitude : t.altitude = 15) :
  TrapezoidGHCD t = 153.75 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_ghcd_l1120_112006


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_trapezoid_area_ratio_l1120_112052

/-- The ratio of areas between a small equilateral triangle and the trapezoid formed by cutting it from a larger equilateral triangle -/
theorem triangle_trapezoid_area_ratio : 
  ∀ (large_side small_side : ℝ),
  large_side = 12 →
  small_side = 6 →
  let large_area := (Real.sqrt 3 / 4) * large_side^2
  let small_area := (Real.sqrt 3 / 4) * small_side^2
  let trapezoid_area := large_area - small_area
  small_area / trapezoid_area = 1 / 3 :=
by
  intros large_side small_side h_large h_small
  -- The proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_trapezoid_area_ratio_l1120_112052


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_parallel_implies_a_equals_one_l1120_112012

open Real

/-- The function f(x) = (x + a) ln x -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (x + a) * log x

/-- The derivative of f(x) -/
noncomputable def f_derivative (a : ℝ) (x : ℝ) : ℝ := log x + (x + a) / x

theorem tangent_line_parallel_implies_a_equals_one (a : ℝ) :
  (f_derivative a 1 = 2) → a = 1 := by
  sorry

#check tangent_line_parallel_implies_a_equals_one

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_parallel_implies_a_equals_one_l1120_112012


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_factorial_24_digit_X_l1120_112058

theorem factorial_24_digit_X : ∃ X : ℕ, 
  X < 10 ∧ 
  (24 : ℕ).factorial = 620448401733239439360000 + X * 1000 ∧ 
  X = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_factorial_24_digit_X_l1120_112058


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_roots_outside_interval_l1120_112060

theorem roots_outside_interval (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  ∀ x : ℝ, (a^x + a^(-x) = 2*a) → x ∉ Set.Icc (-1 : ℝ) 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_roots_outside_interval_l1120_112060


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_wise_number_2022_l1120_112068

/-- A positive integer that can be expressed as the difference of the squares of two positive integers -/
def WiseNumber (n : ℕ) : Prop :=
  ∃ a b : ℕ, a > 0 ∧ b > 0 ∧ n = a^2 - b^2

/-- The sequence of wise numbers in ascending order -/
def WiseNumberSequence : ℕ → ℕ := sorry

/-- The 2022nd wise number -/
def WiseNumber2022 : ℕ := WiseNumberSequence 2022

theorem wise_number_2022 : WiseNumber2022 = 2699 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_wise_number_2022_l1120_112068


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_right_triangle_height_radius_difference_l1120_112037

-- Define necessary structures and functions
structure Point where
  x : ℝ
  y : ℝ

structure Triangle where
  A : Point
  B : Point
  C : Point

def IsoscelesRightTriangle (t : Triangle) : Prop := sorry

def InscribedCircle (center : Point) (radius : ℝ) (t : Triangle) : Prop := sorry

def TriangleHeight (vertex : Point) (base1 : Point) (base2 : Point) : ℝ := sorry

theorem isosceles_right_triangle_height_radius_difference 
  (X Y Z : Point) (O : Point) (r : ℝ) :
  let t := Triangle.mk X Y Z
  IsoscelesRightTriangle t →
  InscribedCircle O r t →
  r = 1/4 →
  let h := TriangleHeight Y X Z
  |h - r| = (Real.sqrt 2 - 1) / 4 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_right_triangle_height_radius_difference_l1120_112037


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_calculation_l1120_112088

/-- Calculates the length of a train given its speed in km/hr and the time it takes to cross a pole -/
noncomputable def train_length (speed_km_hr : ℝ) (time_s : ℝ) : ℝ :=
  (speed_km_hr * 1000 / 3600) * time_s

/-- Theorem stating that a train with a speed of 120 km/hr crossing a pole in 4 seconds has a length of approximately 133.32 meters -/
theorem train_length_calculation :
  let speed := (120 : ℝ)
  let time := (4 : ℝ)
  abs (train_length speed time - 133.32) < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_calculation_l1120_112088


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_factors_of_m_l1120_112036

def m : ℕ := 2^3 * 3^2 * 5^4 * 10^5

theorem number_of_factors_of_m : Finset.card (Finset.filter (· ∣ m) (Finset.range (m + 1))) = 270 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_number_of_factors_of_m_l1120_112036


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_store_credit_percentage_l1120_112005

/-- Calculates the percentage of credit given for a trade-in towards a purchase. -/
noncomputable def trade_in_credit_percentage (super_nintendo_value : ℚ) (cash_paid : ℚ) (change_received : ℚ) 
  (game_value : ℚ) (nes_price : ℚ) : ℚ :=
  let effective_cash_paid := cash_paid - change_received
  let total_value_provided := effective_cash_paid + game_value
  let credit_received := nes_price - total_value_provided
  (credit_received / super_nintendo_value) * 100

/-- The store gives 40% of the Super Nintendo's value as credit. -/
theorem store_credit_percentage :
  trade_in_credit_percentage 150 80 10 30 160 = 40 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_store_credit_percentage_l1120_112005


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_for_extreme_points_l1120_112023

-- Define the function g
noncomputable def g (a : ℝ) (x : ℝ) : ℝ := x^2 + Real.log (x + a)

-- State the theorem
theorem inequality_for_extreme_points (a : ℝ) (x₁ x₂ : ℝ) :
  -- Assume g has two extreme points x₁ and x₂
  (∃ (ε : ℝ), ε > 0 ∧ 
    (∀ x ∈ Set.Ioo (x₁ - ε) x₁, g a x < g a x₁) ∧
    (∀ x ∈ Set.Ioo x₁ (x₁ + ε), g a x < g a x₁) ∧
    (∀ x ∈ Set.Ioo (x₂ - ε) x₂, g a x < g a x₂) ∧
    (∀ x ∈ Set.Ioo x₂ (x₂ + ε), g a x < g a x₂)) →
  -- Then the inequality holds
  (g a x₁ + g a x₂) / 2 > g a ((x₁ + x₂) / 2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_for_extreme_points_l1120_112023


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_M_N_l1120_112067

def M : Set ℤ := {-1, 0, 1, 2, 3}
def N : Set ℤ := {x : ℤ | x^2 - 2*x > 0}

theorem intersection_M_N : M ∩ N = {-1, 3} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_M_N_l1120_112067


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l1120_112099

-- Define the vector operation
def vector_op (a b : ℝ × ℝ) : ℝ × ℝ := (a.1 * b.1, a.2 * b.2)

-- Define the vectors and constants
noncomputable def m : ℝ × ℝ := (2, 1/2)
noncomputable def n : ℝ × ℝ := (Real.pi/3, 0)

-- Define the functions for points P and Q
noncomputable def P (x' : ℝ) : ℝ × ℝ := (x', Real.sin x')
noncomputable def Q (x' : ℝ) : ℝ × ℝ := vector_op m (P x') + n

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := (Q ((x - Real.pi/3)/2)).2

-- Theorem statement
theorem range_of_f :
  Set.range f = Set.Icc (-1/2 : ℝ) (1/2 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_l1120_112099


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_cos_equal_x_squared_l1120_112055

theorem smallest_cos_equal_x_squared (x : ℝ) : x > 0 ∧ Real.cos x = Real.cos (x^2) →
  ∀ y, y > 0 ∧ Real.cos y = Real.cos (y^2) → x ≤ y → x = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_cos_equal_x_squared_l1120_112055
