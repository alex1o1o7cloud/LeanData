import Mathlib

namespace NUMINAMATH_CALUDE_hyperbola_equation_l1848_184863

/-- A hyperbola with center at the origin, focus at (-√5, 0), and a point P such that
    the midpoint of PF₁ is (0, 2) has the equation x² - y²/4 = 1 -/
theorem hyperbola_equation (P : ℝ × ℝ) : 
  let F₁ : ℝ × ℝ := (-Real.sqrt 5, 0)
  let midpoint : ℝ × ℝ := (0, 2)
  (P.1^2 / 1^2 - P.2^2 / 4^2 = 1) ∧ 
  ((P.1 + F₁.1) / 2 = midpoint.1 ∧ (P.2 + F₁.2) / 2 = midpoint.2) →
  ∀ x y : ℝ, x^2 - y^2/4 = 1 ↔ (x^2 / 1^2 - y^2 / 4^2 = 1) := by
  sorry

#check hyperbola_equation

end NUMINAMATH_CALUDE_hyperbola_equation_l1848_184863


namespace NUMINAMATH_CALUDE_negative_64_to_four_thirds_l1848_184860

theorem negative_64_to_four_thirds : (-64 : ℝ) ^ (4/3) = 256 := by sorry

end NUMINAMATH_CALUDE_negative_64_to_four_thirds_l1848_184860


namespace NUMINAMATH_CALUDE_triangle_properties_l1848_184871

-- Define the triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the conditions
def triangle_conditions (t : Triangle) : Prop :=
  t.a + t.b = 5 ∧
  t.c = Real.sqrt 7 ∧
  4 * (Real.sin ((t.A + t.B) / 2))^2 - Real.cos (2 * t.C) = 7/2 ∧
  t.A + t.B + t.C = Real.pi

-- State the theorem
theorem triangle_properties (t : Triangle) 
  (h : triangle_conditions t) : 
  t.C = Real.pi / 3 ∧ 
  (1/2) * t.a * t.b * Real.sin t.C = (3 * Real.sqrt 3) / 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l1848_184871


namespace NUMINAMATH_CALUDE_geometric_sequence_seventh_term_l1848_184819

/-- A geometric sequence is a sequence where the ratio of successive terms is constant. -/
def IsGeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

/-- Given a geometric sequence a with a₁ = -16 and a₄ = 8, prove that a₇ = -4 -/
theorem geometric_sequence_seventh_term
  (a : ℕ → ℝ)
  (h_geom : IsGeometricSequence a)
  (h_a1 : a 1 = -16)
  (h_a4 : a 4 = 8) :
  a 7 = -4 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_seventh_term_l1848_184819


namespace NUMINAMATH_CALUDE_tmall_double_eleven_sales_scientific_notation_l1848_184816

theorem tmall_double_eleven_sales_scientific_notation :
  let billion : ℕ := 10^9
  let sales : ℕ := 2684 * billion
  ∃ (a : ℝ) (n : ℤ), 1 ≤ a ∧ a < 10 ∧ (a * 10^n : ℝ) = sales ∧ a = 2.684 ∧ n = 11 :=
by sorry

end NUMINAMATH_CALUDE_tmall_double_eleven_sales_scientific_notation_l1848_184816


namespace NUMINAMATH_CALUDE_basketball_score_proof_l1848_184840

theorem basketball_score_proof :
  ∀ (S : ℕ) (x : ℕ),
    S > 0 →
    S % 4 = 0 →
    S % 7 = 0 →
    S / 4 + 2 * S / 7 + 15 + x = S →
    x ≤ 14 →
    x = 11 :=
by
  sorry

end NUMINAMATH_CALUDE_basketball_score_proof_l1848_184840


namespace NUMINAMATH_CALUDE_sum_of_roots_l1848_184802

theorem sum_of_roots (p q r s : ℝ) : 
  p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ q ≠ r ∧ q ≠ s ∧ r ≠ s →
  (∀ x : ℝ, x^2 - 14*p*x - 15*q = 0 ↔ x = r ∨ x = s) →
  (∀ x : ℝ, x^2 - 14*r*x - 15*s = 0 ↔ x = p ∨ x = q) →
  p + q + r + s = 3150 := by
sorry

end NUMINAMATH_CALUDE_sum_of_roots_l1848_184802


namespace NUMINAMATH_CALUDE_circle_tangent_properties_l1848_184841

-- Define the circle M
def circle_M (x y : ℝ) : Prop := (x + 1)^2 + y^2 = 2

-- Define the line l
def line_l (x y : ℝ) : Prop := x - y - 3 = 0

-- Define a point P on line l
def point_P (x y : ℝ) : Prop := line_l x y

-- Define tangent lines PA and PB
def tangent_PA (xa ya xp yp : ℝ) : Prop := 
  circle_M xa ya ∧ point_P xp yp ∧ (xa - xp) * (xa + 1) + (ya - yp) * ya = 0

def tangent_PB (xb yb xp yp : ℝ) : Prop := 
  circle_M xb yb ∧ point_P xp yp ∧ (xb - xp) * (xb + 1) + (yb - yp) * yb = 0

-- Theorem statement
theorem circle_tangent_properties :
  ∃ (min_area : ℝ) (chord_length : ℝ) (fixed_point : ℝ × ℝ),
    (min_area = 2 * Real.sqrt 3) ∧
    (chord_length = Real.sqrt 6) ∧
    (fixed_point = (-1/2, -1/2)) ∧
    (∀ xa ya xb yb xp yp : ℝ,
      tangent_PA xa ya xp yp →
      tangent_PB xb yb xp yp →
      -- 1. Minimum area of quadrilateral PAMB
      (xa - xp)^2 + (ya - yp)^2 + (xb - xp)^2 + (yb - yp)^2 ≥ min_area^2 ∧
      -- 2. Length of chord AB when |PA| is shortest
      ((xa - xp)^2 + (ya - yp)^2 = (xb - xp)^2 + (yb - yp)^2 →
        (xa - xb)^2 + (ya - yb)^2 = chord_length^2) ∧
      -- 3. Line AB passes through the fixed point
      (ya - yb) * (fixed_point.1 - xa) = (xa - xb) * (fixed_point.2 - ya)) :=
sorry

end NUMINAMATH_CALUDE_circle_tangent_properties_l1848_184841


namespace NUMINAMATH_CALUDE_min_value_fraction_sum_l1848_184874

theorem min_value_fraction_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 5) :
  1 / (a + 1) + 1 / (b + 2) ≥ 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_min_value_fraction_sum_l1848_184874


namespace NUMINAMATH_CALUDE_room_width_calculation_l1848_184845

/-- Given a rectangular room with length 5 meters, prove that its width is 4.75 meters
    when the cost of paving is 900 per square meter and the total cost is 21375. -/
theorem room_width_calculation (length : ℝ) (cost_per_sqm : ℝ) (total_cost : ℝ) :
  length = 5 →
  cost_per_sqm = 900 →
  total_cost = 21375 →
  (total_cost / cost_per_sqm) / length = 4.75 := by
  sorry

#eval (21375 / 900) / 5

end NUMINAMATH_CALUDE_room_width_calculation_l1848_184845


namespace NUMINAMATH_CALUDE_rachel_brownies_l1848_184865

/-- Rachel's brownie problem -/
theorem rachel_brownies (total : ℕ) (brought_to_school : ℕ) (left_at_home : ℕ) : 
  total = 40 → brought_to_school = 16 → left_at_home = total - brought_to_school →
  left_at_home = 24 := by
  sorry

end NUMINAMATH_CALUDE_rachel_brownies_l1848_184865


namespace NUMINAMATH_CALUDE_impossible_three_quadratics_with_two_roots_l1848_184800

theorem impossible_three_quadratics_with_two_roots :
  ¬ ∃ (a b c : ℝ),
    (∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧ a * x₁^2 + b * x₁ + c = 0 ∧ a * x₂^2 + b * x₂ + c = 0) ∧
    (∃ (y₁ y₂ : ℝ), y₁ ≠ y₂ ∧ c * y₁^2 + a * y₁ + b = 0 ∧ c * y₂^2 + a * y₂ + b = 0) ∧
    (∃ (z₁ z₂ : ℝ), z₁ ≠ z₂ ∧ b * z₁^2 + c * z₁ + a = 0 ∧ b * z₂^2 + c * z₂ + a = 0) :=
by sorry

end NUMINAMATH_CALUDE_impossible_three_quadratics_with_two_roots_l1848_184800


namespace NUMINAMATH_CALUDE_worker_travel_time_l1848_184842

theorem worker_travel_time (normal_speed : ℝ) (slower_speed : ℝ) (usual_time : ℝ) (delay : ℝ) :
  slower_speed = (5 / 6) * normal_speed →
  delay = 12 →
  slower_speed * (usual_time + delay) = normal_speed * usual_time →
  usual_time = 60 := by
sorry

end NUMINAMATH_CALUDE_worker_travel_time_l1848_184842


namespace NUMINAMATH_CALUDE_sequence_general_term_l1848_184899

/-- Given sequences {a_n} and {b_n} with initial conditions and recurrence relations,
    prove the general term formula for {b_n}. -/
theorem sequence_general_term
  (p q r : ℝ)
  (h_q_pos : q > 0)
  (h_p_gt_r : p > r)
  (h_r_pos : r > 0)
  (a : ℕ → ℝ)
  (b : ℕ → ℝ)
  (h_a_init : a 1 = p)
  (h_b_init : b 1 = q)
  (h_a_rec : ∀ n : ℕ, n ≥ 2 → a n = p * a (n - 1))
  (h_b_rec : ∀ n : ℕ, n ≥ 2 → b n = q * a (n - 1) + r * b (n - 1)) :
  ∀ n : ℕ, n ≥ 1 → b n = (q * (p^n - r^n)) / (p - r) :=
by sorry

end NUMINAMATH_CALUDE_sequence_general_term_l1848_184899


namespace NUMINAMATH_CALUDE_probability_theorem_l1848_184818

/-- A permutation of the first n natural numbers -/
def Permutation (n : ℕ) := Fin n → Fin n

/-- The property that a permutation satisfies iₖ ≥ k - 3 for all k -/
def SatisfiesInequality (n : ℕ) (p : Permutation n) : Prop :=
  ∀ k : Fin n, (p k : ℕ) + 1 ≥ k.val - 2

/-- The number of permutations satisfying the inequality -/
def CountSatisfyingPermutations (n : ℕ) : ℕ :=
  (4 ^ (n - 3)) * 6

/-- The probability theorem -/
theorem probability_theorem (n : ℕ) (h : n > 3) :
  (CountSatisfyingPermutations n : ℚ) / (Nat.factorial n) =
  (↑(4 ^ (n - 3) * 6) : ℚ) / (Nat.factorial n) := by
  sorry


end NUMINAMATH_CALUDE_probability_theorem_l1848_184818


namespace NUMINAMATH_CALUDE_correct_selection_ways_l1848_184877

/-- The number of university graduates --/
def total_graduates : ℕ := 10

/-- The number of graduates to be selected --/
def selected_graduates : ℕ := 3

/-- The function that calculates the number of ways to select graduates --/
def selection_ways (total : ℕ) (select : ℕ) (at_least_AB : Bool) (exclude_C : Bool) : ℕ := sorry

/-- The theorem stating the correct number of selection ways --/
theorem correct_selection_ways : 
  selection_ways total_graduates selected_graduates true true = 49 := by sorry

end NUMINAMATH_CALUDE_correct_selection_ways_l1848_184877


namespace NUMINAMATH_CALUDE_power_of_256_l1848_184822

theorem power_of_256 : (256 : ℝ) ^ (5/4) = 1024 := by
  have h : (256 : ℝ) = 2^8 := by sorry
  sorry

end NUMINAMATH_CALUDE_power_of_256_l1848_184822


namespace NUMINAMATH_CALUDE_first_row_seats_theorem_l1848_184867

/-- Represents a theater with a specific seating arrangement. -/
structure Theater where
  rows : ℕ
  seatIncrement : ℕ
  totalSeats : ℕ

/-- Calculates the number of seats in the first row of the theater. -/
def firstRowSeats (t : Theater) : ℚ :=
  (t.totalSeats / 10 - 76) / 2

/-- Theorem stating the relationship between the total seats and the number of seats in the first row. -/
theorem first_row_seats_theorem (t : Theater) 
    (h1 : t.rows = 20)
    (h2 : t.seatIncrement = 4)
    (h3 : t.totalSeats = 10 * (firstRowSeats t * 2 + 76)) : 
  firstRowSeats t = (t.totalSeats / 10 - 76) / 2 := by
  sorry

end NUMINAMATH_CALUDE_first_row_seats_theorem_l1848_184867


namespace NUMINAMATH_CALUDE_max_monotone_interval_l1848_184844

theorem max_monotone_interval (f : ℝ → ℝ) (h : f = λ x => Real.sin (Real.pi * x - Real.pi / 6)) :
  (∃ m : ℝ, m = 2/3 ∧ 
   (∀ x₁ x₂ : ℝ, 0 ≤ x₁ ∧ x₁ < x₂ ∧ x₂ ≤ m → f x₁ < f x₂) ∧
   (∀ m' : ℝ, m' > m → ∃ x₁ x₂ : ℝ, 0 ≤ x₁ ∧ x₁ < x₂ ∧ x₂ ≤ m' ∧ f x₁ ≥ f x₂)) :=
sorry

end NUMINAMATH_CALUDE_max_monotone_interval_l1848_184844


namespace NUMINAMATH_CALUDE_meters_to_cm_conversion_l1848_184869

/-- Conversion factor from meters to centimeters -/
def meters_to_cm : ℝ := 100

/-- Proves that 3.5 meters is equal to 350 centimeters -/
theorem meters_to_cm_conversion : 3.5 * meters_to_cm = 350 := by
  sorry

end NUMINAMATH_CALUDE_meters_to_cm_conversion_l1848_184869


namespace NUMINAMATH_CALUDE_hyperbola_real_axis_length_l1848_184875

/-- The hyperbola equation -/
def hyperbola_eq (x y : ℝ) : Prop := 2 * x^2 - y^2 = 8

/-- The length of the real axis of the hyperbola -/
def real_axis_length : ℝ := 4

/-- Theorem: The length of the real axis of the hyperbola 2x^2 - y^2 = 8 is 4 -/
theorem hyperbola_real_axis_length :
  ∀ x y : ℝ, hyperbola_eq x y → real_axis_length = 4 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_real_axis_length_l1848_184875


namespace NUMINAMATH_CALUDE_equal_intercept_line_equation_l1848_184807

/-- A line passing through (1,2) with equal intercepts on both axes -/
structure EqualInterceptLine where
  -- The slope of the line
  k : ℝ
  -- The line passes through (1,2)
  point_condition : 2 = k * (1 - 1) + 2
  -- The line has equal intercepts on both axes
  equal_intercepts : 2 - k = 1 - 2 / k

/-- The equation of the line is either x+y-3=0 or 2x-y=0 -/
theorem equal_intercept_line_equation (l : EqualInterceptLine) :
  (∀ x y, x + y - 3 = 0 ↔ y = l.k * (x - 1) + 2) ∨
  (∀ x y, 2 * x - y = 0 ↔ y = l.k * (x - 1) + 2) :=
sorry

end NUMINAMATH_CALUDE_equal_intercept_line_equation_l1848_184807


namespace NUMINAMATH_CALUDE_sum_and_reciprocal_sum_l1848_184811

theorem sum_and_reciprocal_sum (x : ℝ) (h : x > 0) (h_sum_squares : x^2 + (1/x)^2 = 23) : 
  x + (1/x) = 5 := by sorry

end NUMINAMATH_CALUDE_sum_and_reciprocal_sum_l1848_184811


namespace NUMINAMATH_CALUDE_complex_distance_problem_l1848_184854

theorem complex_distance_problem (α : ℂ) (h1 : α ≠ 1) 
  (h2 : Complex.abs (α^3 - 1) = 3 * Complex.abs (α - 1))
  (h3 : Complex.abs (α^6 - 1) = 5 * Complex.abs (α - 1)) :
  α = Complex.I * Real.sqrt 3 ∨ α = -Complex.I * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_complex_distance_problem_l1848_184854


namespace NUMINAMATH_CALUDE_alcohol_concentration_in_mixture_l1848_184882

structure Vessel where
  capacity : ℝ
  alcoholConcentration : ℝ

def totalAlcohol (vessels : List Vessel) : ℝ :=
  vessels.foldl (fun acc v => acc + v.capacity * v.alcoholConcentration) 0

def largeContainerCapacity : ℝ := 25

theorem alcohol_concentration_in_mixture 
  (vessels : List Vessel)
  (h1 : vessels = [
    ⟨2, 0.3⟩, 
    ⟨6, 0.4⟩, 
    ⟨4, 0.25⟩, 
    ⟨3, 0.35⟩, 
    ⟨5, 0.2⟩
  ]) :
  (totalAlcohol vessels) / largeContainerCapacity = 0.242 := by
  sorry

end NUMINAMATH_CALUDE_alcohol_concentration_in_mixture_l1848_184882


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_divisibility_l1848_184850

def arithmetic_sequence_sum (a₁ : ℕ) (d : ℕ) (aₙ : ℕ) : ℕ :=
  let n := (aₙ - a₁) / d + 1
  n * (a₁ + aₙ) / 2

theorem arithmetic_sequence_sum_divisibility :
  (arithmetic_sequence_sum 1 8 313) % 8 = 0 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_divisibility_l1848_184850


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_l1848_184838

theorem partial_fraction_decomposition :
  ∃ (A B C : ℚ), A = -1/2 ∧ B = 5/2 ∧ C = -5 ∧
  ∀ (x : ℚ), x ≠ 0 → x^2 ≠ 2 →
  (2*x^2 - 5*x + 1) / (x^3 - 2*x) = A / x + (B*x + C) / (x^2 - 2) := by
  sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_l1848_184838


namespace NUMINAMATH_CALUDE_range_of_f_l1848_184839

def f (x : ℤ) : ℤ := x + 1

def domain : Set ℤ := {-1, 1, 2}

theorem range_of_f : 
  {y : ℤ | ∃ x ∈ domain, f x = y} = {0, 2, 3} := by sorry

end NUMINAMATH_CALUDE_range_of_f_l1848_184839


namespace NUMINAMATH_CALUDE_sharp_four_times_100_l1848_184868

-- Define the # function
def sharp (N : ℝ) : ℝ := 0.7 * N + 5

-- State the theorem
theorem sharp_four_times_100 : sharp (sharp (sharp (sharp 100))) = 36.675 := by
  sorry

end NUMINAMATH_CALUDE_sharp_four_times_100_l1848_184868


namespace NUMINAMATH_CALUDE_susan_correct_percentage_l1848_184884

theorem susan_correct_percentage (y : ℝ) (h : y ≠ 0) :
  let total_questions : ℝ := 8 * y
  let unattempted_questions : ℝ := 2 * y + 3
  let correct_questions : ℝ := total_questions - unattempted_questions
  let percentage_correct : ℝ := (correct_questions / total_questions) * 100
  percentage_correct = 75 * (2 * y - 1) / y :=
by sorry

end NUMINAMATH_CALUDE_susan_correct_percentage_l1848_184884


namespace NUMINAMATH_CALUDE_senior_japanese_fraction_l1848_184888

theorem senior_japanese_fraction (j : ℝ) (s : ℝ) (x : ℝ) :
  s = 2 * j →                     -- Senior class is twice the size of junior class
  (1 / 3) * (j + s) = (3 / 4) * j + x * s →  -- 1/3 of all students equals 3/4 of juniors plus x fraction of seniors
  x = 1 / 8 :=                    -- Fraction of seniors studying Japanese
by sorry

end NUMINAMATH_CALUDE_senior_japanese_fraction_l1848_184888


namespace NUMINAMATH_CALUDE_quadratic_non_real_roots_l1848_184848

/-- A quadratic equation of the form x^2 - 8x + c has non-real roots if and only if c > 16 -/
theorem quadratic_non_real_roots (c : ℝ) : 
  (∀ x : ℂ, x^2 - 8*x + c = 0 → x.im ≠ 0) ↔ c > 16 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_non_real_roots_l1848_184848


namespace NUMINAMATH_CALUDE_x_plus_y_value_l1848_184834

theorem x_plus_y_value (x y : ℝ) 
  (eq1 : x + Real.cos y = 3000)
  (eq2 : x + 3000 * Real.sin y = 2999)
  (y_range : 0 ≤ y ∧ y ≤ Real.pi / 2) :
  x + y = 2999 := by
sorry

end NUMINAMATH_CALUDE_x_plus_y_value_l1848_184834


namespace NUMINAMATH_CALUDE_billys_age_l1848_184889

theorem billys_age (billy joe : ℕ) 
  (h1 : billy = 2 * joe) 
  (h2 : billy + joe = 45) : 
  billy = 30 := by
sorry

end NUMINAMATH_CALUDE_billys_age_l1848_184889


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l1848_184857

theorem sqrt_equation_solution :
  ∃ x : ℝ, x = 6 ∧ Real.sqrt (4 + 9 + x^2) = 7 :=
by sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l1848_184857


namespace NUMINAMATH_CALUDE_add_8035_seconds_to_8am_l1848_184878

/-- Represents time in hours, minutes, and seconds -/
structure Time where
  hours : Nat
  minutes : Nat
  seconds : Nat
  deriving Repr

/-- Adds seconds to a given time -/
def addSeconds (t : Time) (s : Nat) : Time :=
  sorry

/-- The starting time (8:00:00 AM) -/
def startTime : Time :=
  { hours := 8, minutes := 0, seconds := 0 }

/-- The number of seconds to add -/
def secondsToAdd : Nat := 8035

/-- The expected end time (10:13:35) -/
def endTime : Time :=
  { hours := 10, minutes := 13, seconds := 35 }

theorem add_8035_seconds_to_8am :
  addSeconds startTime secondsToAdd = endTime := by
  sorry

end NUMINAMATH_CALUDE_add_8035_seconds_to_8am_l1848_184878


namespace NUMINAMATH_CALUDE_smallest_t_is_five_l1848_184809

-- Define the triangle sides
def a : ℝ := 7.5
def b : ℝ := 12

-- Define the property that t is the smallest whole number satisfying the triangle inequality
def is_smallest_valid_t (t : ℕ) : Prop :=
  (t : ℝ) + a > b ∧ 
  (t : ℝ) + b > a ∧ 
  a + b > (t : ℝ) ∧
  ∀ k : ℕ, k < t → ¬((k : ℝ) + a > b ∧ (k : ℝ) + b > a ∧ a + b > (k : ℝ))

-- Theorem statement
theorem smallest_t_is_five : is_smallest_valid_t 5 := by
  sorry

end NUMINAMATH_CALUDE_smallest_t_is_five_l1848_184809


namespace NUMINAMATH_CALUDE_sum_of_reciprocals_is_six_l1848_184862

theorem sum_of_reciprocals_is_six (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) 
  (h : x + y = 6 * x * y) : 
  1 / x + 1 / y = 6 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_reciprocals_is_six_l1848_184862


namespace NUMINAMATH_CALUDE_peach_count_l1848_184843

/-- Given a basket of peaches with red and green peaches, calculate the total number of peaches -/
def total_peaches (red_peaches green_peaches : ℕ) : ℕ :=
  red_peaches + green_peaches

/-- Theorem: Given 1 basket with 4 red peaches and 6 green peaches, the total number of peaches is 10 -/
theorem peach_count : total_peaches 4 6 = 10 := by
  sorry

end NUMINAMATH_CALUDE_peach_count_l1848_184843


namespace NUMINAMATH_CALUDE_cubic_repeated_roots_l1848_184880

theorem cubic_repeated_roots (p q : ℝ) :
  (∃ (x : ℝ), (x^3 + p*x + q = 0 ∧ ∃ (y : ℝ), y ≠ x ∧ y^3 + p*y + q = 0 ∧ (∀ (z : ℝ), z^3 + p*z + q = 0 → z = x ∨ z = y))) ↔
  (q^2 / 4 + p^3 / 27 = 0) := by
sorry

end NUMINAMATH_CALUDE_cubic_repeated_roots_l1848_184880


namespace NUMINAMATH_CALUDE_system_solution_l1848_184864

theorem system_solution :
  ∀ x y z : ℝ,
  x + y + z = 13 →
  x^2 + y^2 + z^2 = 61 →
  x*y + x*z = 2*y*z →
  ((x = 4 ∧ y = 3 ∧ z = 6) ∨ (x = 4 ∧ y = 6 ∧ z = 3)) :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l1848_184864


namespace NUMINAMATH_CALUDE_symmetric_difference_of_A_and_B_l1848_184897

-- Define the sets A and B
def A : Set ℝ := {y | y > 0}
def B : Set ℝ := {y | y ≤ 2}

-- Define the symmetric difference operation
def symmetricDifference (M N : Set ℝ) : Set ℝ :=
  (M \ N) ∪ (N \ M)

-- State the theorem
theorem symmetric_difference_of_A_and_B :
  symmetricDifference A B = {y | y ≤ 0 ∨ y > 2} := by sorry

end NUMINAMATH_CALUDE_symmetric_difference_of_A_and_B_l1848_184897


namespace NUMINAMATH_CALUDE_correct_total_amount_paid_l1848_184815

/-- Calculates the total amount paid for fruits with discounts --/
def totalAmountPaid (
  peachPrice peachCount peachDiscountThreshold peachDiscount : ℚ)
  (applePrice appleCount appleDiscountThreshold appleDiscount : ℚ)
  (orangePrice orangeCount orangeDiscountThreshold orangeDiscount : ℚ)
  (grapefruitPrice grapefruitCount grapefruitDiscountThreshold grapefruitDiscount : ℚ)
  (bundleDiscountThreshold1 bundleDiscountThreshold2 bundleDiscountThreshold3 bundleDiscount : ℚ) : ℚ :=
  let peachTotal := peachPrice * peachCount
  let appleTotal := applePrice * appleCount
  let orangeTotal := orangePrice * orangeCount
  let grapefruitTotal := grapefruitPrice * grapefruitCount
  let peachDiscountTimes := (peachTotal / peachDiscountThreshold).floor
  let appleDiscountTimes := (appleTotal / appleDiscountThreshold).floor
  let orangeDiscountTimes := (orangeTotal / orangeDiscountThreshold).floor
  let grapefruitDiscountTimes := (grapefruitTotal / grapefruitDiscountThreshold).floor
  let totalBeforeDiscount := peachTotal + appleTotal + orangeTotal + grapefruitTotal
  let individualDiscounts := peachDiscountTimes * peachDiscount + 
                             appleDiscountTimes * appleDiscount + 
                             orangeDiscountTimes * orangeDiscount + 
                             grapefruitDiscountTimes * grapefruitDiscount
  let bundleDiscountApplied := if peachCount ≥ bundleDiscountThreshold1 ∧ 
                                  appleCount ≥ bundleDiscountThreshold2 ∧ 
                                  orangeCount ≥ bundleDiscountThreshold3 
                               then bundleDiscount else 0
  totalBeforeDiscount - individualDiscounts - bundleDiscountApplied

theorem correct_total_amount_paid : 
  totalAmountPaid 0.4 400 10 2 0.6 150 15 3 0.5 200 7 1.5 1 80 20 4 100 50 100 10 = 333 := by
  sorry


end NUMINAMATH_CALUDE_correct_total_amount_paid_l1848_184815


namespace NUMINAMATH_CALUDE_function_shift_l1848_184820

-- Define the function f
variable (f : ℝ → ℝ)

-- State the theorem
theorem function_shift (h : f 0 = 2) : f (-1 + 1) = 2 := by
  sorry

end NUMINAMATH_CALUDE_function_shift_l1848_184820


namespace NUMINAMATH_CALUDE_parallelogram_xy_product_l1848_184810

/-- A parallelogram with side lengths specified by parameters -/
structure Parallelogram (x y : ℝ) where
  ef : ℝ
  fg : ℝ
  gh : ℝ
  he : ℝ
  ef_eq : ef = 42
  fg_eq : fg = 4 * y^2
  gh_eq : gh = 3 * x + 6
  he_eq : he = 32
  opposite_sides_equal : ef = gh ∧ fg = he

/-- The product of x and y in the specified parallelogram is 24√2 -/
theorem parallelogram_xy_product (x y : ℝ) (p : Parallelogram x y) :
  x * y = 24 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_xy_product_l1848_184810


namespace NUMINAMATH_CALUDE_unknown_number_proof_l1848_184814

theorem unknown_number_proof (n : ℝ) (h : (12 : ℝ) * n^4 / 432 = 36) : n = 6 := by
  sorry

end NUMINAMATH_CALUDE_unknown_number_proof_l1848_184814


namespace NUMINAMATH_CALUDE_a_plus_b_value_l1848_184876

theorem a_plus_b_value (a b : ℝ) (h1 : a^2 = 16) (h2 : |b| = 3) (h3 : a + b < 0) :
  a + b = -1 ∨ a + b = -7 :=
by sorry

end NUMINAMATH_CALUDE_a_plus_b_value_l1848_184876


namespace NUMINAMATH_CALUDE_intersection_empty_implies_a_range_l1848_184892

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {x : ℝ | |x - 2| ≤ a}
def B : Set ℝ := {x : ℝ | x^2 - 5*x + 4 ≥ 0}

-- State the theorem
theorem intersection_empty_implies_a_range (a : ℝ) :
  A a ∩ B = ∅ → a ∈ Set.Iio 1 := by
  sorry

end NUMINAMATH_CALUDE_intersection_empty_implies_a_range_l1848_184892


namespace NUMINAMATH_CALUDE_glass_bowls_problem_l1848_184886

/-- The number of glass bowls initially bought -/
def initial_bowls : ℕ := 316

/-- The cost price per bowl in rupees -/
def cost_price : ℚ := 12

/-- The selling price per bowl in rupees -/
def selling_price : ℚ := 15

/-- The number of bowls sold -/
def bowls_sold : ℕ := 102

/-- The percentage gain -/
def percentage_gain : ℚ := 8050847457627118 / 1000000000000000

theorem glass_bowls_problem :
  initial_bowls = 316 ∧
  (bowls_sold : ℚ) * (selling_price - cost_price) / (initial_bowls * cost_price) = percentage_gain / 100 := by
  sorry


end NUMINAMATH_CALUDE_glass_bowls_problem_l1848_184886


namespace NUMINAMATH_CALUDE_elvis_song_writing_time_l1848_184826

/-- Given Elvis's album production parameters, prove the time to write each song. -/
theorem elvis_song_writing_time
  (total_songs : ℕ)
  (studio_time_hours : ℕ)
  (recording_time_per_song : ℕ)
  (total_editing_time : ℕ)
  (h1 : total_songs = 15)
  (h2 : studio_time_hours = 7)
  (h3 : recording_time_per_song = 18)
  (h4 : total_editing_time = 45) :
  (studio_time_hours * 60 - recording_time_per_song * total_songs - total_editing_time) / total_songs = 7 :=
by sorry

end NUMINAMATH_CALUDE_elvis_song_writing_time_l1848_184826


namespace NUMINAMATH_CALUDE_hundredths_place_is_zero_l1848_184879

def number : ℚ := 317.502

theorem hundredths_place_is_zero : 
  (number * 100 % 10).floor = 0 := by sorry

end NUMINAMATH_CALUDE_hundredths_place_is_zero_l1848_184879


namespace NUMINAMATH_CALUDE_inequality_proof_l1848_184829

theorem inequality_proof (a b c : ℝ) 
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_sum_squares : a^2 + b^2 + c^2 = 3) : 
  1/(1+a*b) + 1/(1+b*c) + 1/(1+c*a) ≥ 3/2 := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l1848_184829


namespace NUMINAMATH_CALUDE_right_triangle_circle_diameters_sum_l1848_184806

/-- For a right-angled triangle with legs a and b, hypotenuse c,
    radius R of the circumscribed circle, and radius r of the inscribed circle,
    the sum of the diameters of the circumscribed and inscribed circles
    is equal to the sum of the legs. -/
theorem right_triangle_circle_diameters_sum (a b c R r : ℝ) :
  (a > 0) → (b > 0) → (c > 0) → (R > 0) → (r > 0) →
  (a^2 + b^2 = c^2) →  -- Pythagorean theorem for right triangle
  (c = 2 * R) →        -- Diameter of circumscribed circle equals hypotenuse
  (r = (a + b - c) / 2) →  -- Formula for inscribed circle radius in right triangle
  (2 * R + 2 * r = a + b) := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_circle_diameters_sum_l1848_184806


namespace NUMINAMATH_CALUDE_hall_volume_l1848_184833

/-- The volume of a rectangular hall with given dimensions and area constraint -/
theorem hall_volume (length width : ℝ) (h : length = 15 ∧ width = 12) 
  (area_constraint : 2 * (length * width) = 2 * (length + width) * ((2 * length * width) / (2 * (length + width)))) :
  length * width * ((2 * length * width) / (2 * (length + width))) = 8004 := by
sorry

end NUMINAMATH_CALUDE_hall_volume_l1848_184833


namespace NUMINAMATH_CALUDE_equation_solution_l1848_184885

theorem equation_solution : 
  Real.sqrt (1 + Real.sqrt (2 + Real.sqrt 49)) = (1 + Real.sqrt 49) ^ (1/3) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1848_184885


namespace NUMINAMATH_CALUDE_only_point_0_neg2_satisfies_l1848_184847

def point_satisfies_inequalities (x y : ℝ) : Prop :=
  x + y - 1 < 0 ∧ x - y + 1 > 0

theorem only_point_0_neg2_satisfies : 
  ¬(point_satisfies_inequalities 0 2) ∧
  ¬(point_satisfies_inequalities (-2) 0) ∧
  point_satisfies_inequalities 0 (-2) ∧
  ¬(point_satisfies_inequalities 2 0) :=
sorry

end NUMINAMATH_CALUDE_only_point_0_neg2_satisfies_l1848_184847


namespace NUMINAMATH_CALUDE_round_table_seating_l1848_184887

theorem round_table_seating (W M : ℕ) : 
  W = 19 → 
  M = 16 → 
  (7 : ℕ) + 12 = W → 
  (3 : ℕ) * 12 = 3 * W - 3 * M → 
  W + M = 35 := by
sorry

end NUMINAMATH_CALUDE_round_table_seating_l1848_184887


namespace NUMINAMATH_CALUDE_all_lines_pass_through_point_common_point_is_neg_two_two_l1848_184890

/-- A line in the form ax + by = c -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Predicate to check if a, b, c form an arithmetic progression with common difference 3d -/
def is_ap (l : Line) (d : ℝ) : Prop :=
  l.b = l.a + 3 * d ∧ l.c = l.a + 6 * d

/-- Check if a point (x, y) lies on a line -/
def point_on_line (l : Line) (x y : ℝ) : Prop :=
  l.a * x + l.b * y = l.c

/-- Theorem stating that all lines satisfying the arithmetic progression condition pass through (-2, 2) -/
theorem all_lines_pass_through_point (l : Line) (d : ℝ) :
  is_ap l d → point_on_line l (-2) 2 := by
  sorry

/-- Main theorem proving the common point is (-2, 2) -/
theorem common_point_is_neg_two_two :
  ∃ (x y : ℝ), ∀ (l : Line) (d : ℝ), is_ap l d → point_on_line l x y ∧ x = -2 ∧ y = 2 := by
  sorry

end NUMINAMATH_CALUDE_all_lines_pass_through_point_common_point_is_neg_two_two_l1848_184890


namespace NUMINAMATH_CALUDE_excursion_existence_l1848_184881

theorem excursion_existence (S : Finset Nat) (E : Finset (Finset Nat)) 
  (h1 : S.card = 20) 
  (h2 : ∀ e ∈ E, e.card > 0) 
  (h3 : ∀ e ∈ E, e ⊆ S) :
  ∃ e ∈ E, ∀ s ∈ e, (E.filter (λ f => s ∈ f)).card ≥ E.card / 20 := by
sorry


end NUMINAMATH_CALUDE_excursion_existence_l1848_184881


namespace NUMINAMATH_CALUDE_allen_age_difference_l1848_184831

/-- Allen's age problem -/
theorem allen_age_difference :
  ∀ (allen_age mother_age : ℕ),
  mother_age = 30 →
  allen_age + 3 + mother_age + 3 = 41 →
  mother_age - allen_age = 25 :=
by sorry

end NUMINAMATH_CALUDE_allen_age_difference_l1848_184831


namespace NUMINAMATH_CALUDE_max_handshakes_l1848_184830

def number_of_people : ℕ := 30

theorem max_handshakes (n : ℕ) (h : n = number_of_people) : 
  Nat.choose n 2 = 435 := by
  sorry

end NUMINAMATH_CALUDE_max_handshakes_l1848_184830


namespace NUMINAMATH_CALUDE_airplane_seats_theorem_l1848_184883

/-- Represents the total number of seats on an airplane -/
def total_seats : ℕ := 360

/-- Represents the number of First Class seats -/
def first_class_seats : ℕ := 36

/-- Represents the fraction of total seats in Business Class -/
def business_class_fraction : ℚ := 3/10

/-- Represents the fraction of total seats in Economy -/
def economy_fraction : ℚ := 6/10

theorem airplane_seats_theorem :
  (first_class_seats : ℚ) + 
  (business_class_fraction * total_seats) + 
  (economy_fraction * total_seats) = total_seats := by
  sorry

end NUMINAMATH_CALUDE_airplane_seats_theorem_l1848_184883


namespace NUMINAMATH_CALUDE_five_digit_multiple_of_nine_l1848_184851

theorem five_digit_multiple_of_nine : ∃ (n : ℕ), n = 45675 ∧ n % 9 = 0 := by
  sorry

end NUMINAMATH_CALUDE_five_digit_multiple_of_nine_l1848_184851


namespace NUMINAMATH_CALUDE_max_angle_C_in_triangle_l1848_184827

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    if a² + b² = 2c², then the maximum value of angle C is π/3 -/
theorem max_angle_C_in_triangle (a b c : ℝ) (h : a^2 + b^2 = 2*c^2) :
  ∃ (A B C : ℝ), 
    0 < A ∧ 0 < B ∧ 0 < C ∧ 
    A + B + C = π ∧
    c = Real.sqrt (a^2 + b^2 - 2*a*b*Real.cos C) ∧
    C ≤ π/3 ∧
    (C = π/3 → a = b) := by
  sorry

end NUMINAMATH_CALUDE_max_angle_C_in_triangle_l1848_184827


namespace NUMINAMATH_CALUDE_prob_both_odd_is_one_sixth_l1848_184849

/-- The set of numbers to draw from -/
def S : Finset ℕ := {1, 2, 3, 4}

/-- A function to determine if a number is odd -/
def isOdd (n : ℕ) : Bool := n % 2 = 1

/-- The set of all possible pairs of numbers drawn without replacement -/
def allPairs : Finset (ℕ × ℕ) := S.product S |>.filter (fun (a, b) => a ≠ b)

/-- The set of pairs where both numbers are odd -/
def oddPairs : Finset (ℕ × ℕ) := allPairs.filter (fun (a, b) => isOdd a ∧ isOdd b)

/-- The probability of drawing two odd numbers without replacement -/
def probBothOdd : ℚ := (oddPairs.card : ℚ) / (allPairs.card : ℚ)

theorem prob_both_odd_is_one_sixth : probBothOdd = 1 / 6 := by
  sorry

end NUMINAMATH_CALUDE_prob_both_odd_is_one_sixth_l1848_184849


namespace NUMINAMATH_CALUDE_conditional_equivalence_l1848_184832

theorem conditional_equivalence (R S : Prop) :
  (¬R → S) ↔ (¬S → R) := by sorry

end NUMINAMATH_CALUDE_conditional_equivalence_l1848_184832


namespace NUMINAMATH_CALUDE_cubic_as_difference_of_squares_l1848_184856

theorem cubic_as_difference_of_squares (a : ℕ) :
  a^3 = (a * (a + 1) / 2)^2 - (a * (a - 1) / 2)^2 := by
  sorry

end NUMINAMATH_CALUDE_cubic_as_difference_of_squares_l1848_184856


namespace NUMINAMATH_CALUDE_seventeen_to_fourteen_greater_than_thirtyone_to_eleven_l1848_184805

theorem seventeen_to_fourteen_greater_than_thirtyone_to_eleven :
  (17 : ℝ)^14 > (31 : ℝ)^11 := by sorry

end NUMINAMATH_CALUDE_seventeen_to_fourteen_greater_than_thirtyone_to_eleven_l1848_184805


namespace NUMINAMATH_CALUDE_locus_of_nine_point_center_on_BC_l1848_184824

/-- Triangle ABC with fixed vertices B and C -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ := (-1, 0)
  C : ℝ × ℝ := (1, 0)

/-- The nine-point center of a triangle -/
def ninePointCenter (t : Triangle) : ℝ × ℝ := sorry

/-- Check if a point is on a line segment -/
def isOnSegment (p : ℝ × ℝ) (a : ℝ × ℝ) (b : ℝ × ℝ) : Prop := sorry

/-- The locus of point A -/
def locusOfA (x y : ℝ) : Prop := x^2 - y^2 = 1

theorem locus_of_nine_point_center_on_BC (t : Triangle) :
  isOnSegment (ninePointCenter t) t.B t.C ↔ locusOfA t.A.1 t.A.2 := by sorry

end NUMINAMATH_CALUDE_locus_of_nine_point_center_on_BC_l1848_184824


namespace NUMINAMATH_CALUDE_intersection_m_complement_n_l1848_184821

def M : Set ℝ := {x | -1 < x ∧ x < 2}
def N : Set ℝ := {x | x ≥ 1}

theorem intersection_m_complement_n : M ∩ (Set.univ \ N) = Set.Ioo (-1) 1 := by
  sorry

end NUMINAMATH_CALUDE_intersection_m_complement_n_l1848_184821


namespace NUMINAMATH_CALUDE_bus_trip_difference_l1848_184813

def bus_trip (initial : ℕ) 
             (stop1_off stop1_on : ℕ) 
             (stop2_off stop2_on : ℕ) 
             (stop3_off stop3_on : ℕ) 
             (stop4_off stop4_on : ℕ) : ℕ :=
  let after_stop1 := initial - stop1_off + stop1_on
  let after_stop2 := after_stop1 - stop2_off + stop2_on
  let after_stop3 := after_stop2 - stop3_off + stop3_on
  let final := after_stop3 - stop4_off + stop4_on
  initial - final

theorem bus_trip_difference :
  bus_trip 41 12 5 7 10 14 3 9 6 = 18 := by
  sorry

end NUMINAMATH_CALUDE_bus_trip_difference_l1848_184813


namespace NUMINAMATH_CALUDE_water_bottle_distribution_l1848_184870

theorem water_bottle_distribution (initial_bottles : ℕ) (drunk_bottles : ℕ) (num_friends : ℕ) : 
  initial_bottles = 120 → 
  drunk_bottles = 15 → 
  num_friends = 5 → 
  (initial_bottles - drunk_bottles) / (num_friends + 1) = 17 := by
sorry

end NUMINAMATH_CALUDE_water_bottle_distribution_l1848_184870


namespace NUMINAMATH_CALUDE_range_of_a_l1848_184801

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, x ∈ Set.Icc 1 2 → x^2 - a*x > 0) → 
  a < 1 := by
sorry

end NUMINAMATH_CALUDE_range_of_a_l1848_184801


namespace NUMINAMATH_CALUDE_dress_savings_theorem_l1848_184898

/-- Calculates the number of weeks needed to save for a dress -/
def weeks_to_save (dress_cost : ℕ) (initial_savings : ℕ) (weekly_allowance : ℕ) (weekly_spending : ℕ) : ℕ :=
  let additional_money_needed := dress_cost - initial_savings
  let weekly_savings := weekly_allowance - weekly_spending
  (additional_money_needed + weekly_savings - 1) / weekly_savings

theorem dress_savings_theorem (dress_cost : ℕ) (initial_savings : ℕ) (weekly_allowance : ℕ) (weekly_spending : ℕ)
  (h1 : dress_cost = 80)
  (h2 : initial_savings = 20)
  (h3 : weekly_allowance = 30)
  (h4 : weekly_spending = 10) :
  weeks_to_save dress_cost initial_savings weekly_allowance weekly_spending = 3 := by
  sorry

end NUMINAMATH_CALUDE_dress_savings_theorem_l1848_184898


namespace NUMINAMATH_CALUDE_power_multiplication_zero_power_distribute_and_simplify_negative_power_and_division_l1848_184836

-- Define variables
variable (a m : ℝ)
variable (π : ℝ)

-- Theorem statements
theorem power_multiplication : a^2 * a^3 = a^5 := by sorry

theorem zero_power : (3.142 - π)^0 = 1 := by sorry

theorem distribute_and_simplify : 2*a*(a^2 - 1) = 2*a^3 - 2*a := by sorry

theorem negative_power_and_division : (-m^3)^2 / m^4 = m^2 := by sorry

end NUMINAMATH_CALUDE_power_multiplication_zero_power_distribute_and_simplify_negative_power_and_division_l1848_184836


namespace NUMINAMATH_CALUDE_total_canoes_april_l1848_184846

def canoe_production (initial : ℕ) (months : ℕ) : ℕ :=
  if months = 0 then 0
  else initial * (3^months - 1) / 2

theorem total_canoes_april : canoe_production 5 4 = 200 := by
  sorry

end NUMINAMATH_CALUDE_total_canoes_april_l1848_184846


namespace NUMINAMATH_CALUDE_star_four_five_l1848_184858

def star (a b : ℤ) : ℤ := (a + 2*b) * (a - 2*b)

theorem star_four_five : star 4 5 = -84 := by
  sorry

end NUMINAMATH_CALUDE_star_four_five_l1848_184858


namespace NUMINAMATH_CALUDE_sweet_cookies_eaten_indeterminate_l1848_184866

def initial_salty_cookies : ℕ := 26
def initial_sweet_cookies : ℕ := 17
def salty_cookies_eaten : ℕ := 9
def salty_cookies_left : ℕ := 17

theorem sweet_cookies_eaten_indeterminate :
  ∀ (sweet_cookies_eaten : ℕ),
    sweet_cookies_eaten ≤ initial_sweet_cookies →
    salty_cookies_left = initial_salty_cookies - salty_cookies_eaten →
    ∃ (sweet_cookies_eaten' : ℕ),
      sweet_cookies_eaten' ≠ sweet_cookies_eaten ∧
      sweet_cookies_eaten' ≤ initial_sweet_cookies :=
by sorry

end NUMINAMATH_CALUDE_sweet_cookies_eaten_indeterminate_l1848_184866


namespace NUMINAMATH_CALUDE_nested_sqrt_fourth_power_l1848_184808

theorem nested_sqrt_fourth_power :
  (Real.sqrt (1 + Real.sqrt (1 + Real.sqrt (1 + Real.sqrt 1))))^4 = 6 + 2 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_nested_sqrt_fourth_power_l1848_184808


namespace NUMINAMATH_CALUDE_smallest_prime_factor_in_C_l1848_184894

def C : Finset Nat := {67, 71, 73, 76, 85}

theorem smallest_prime_factor_in_C : 
  ∃ (n : Nat), n ∈ C ∧ (∀ m ∈ C, ∀ p q : Nat, Prime p → Prime q → p ∣ n → q ∣ m → p ≤ q) ∧ n = 76 := by
  sorry

end NUMINAMATH_CALUDE_smallest_prime_factor_in_C_l1848_184894


namespace NUMINAMATH_CALUDE_min_value_of_x2_plus_2y2_l1848_184872

theorem min_value_of_x2_plus_2y2 (x y : ℝ) (h : x^2 - 2*x*y + 2*y^2 = 2) :
  ∃ (m : ℝ), m = 4 - 2*Real.sqrt 2 ∧ ∀ (a b : ℝ), a^2 - 2*a*b + 2*b^2 = 2 → x^2 + 2*y^2 ≥ m :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_x2_plus_2y2_l1848_184872


namespace NUMINAMATH_CALUDE_fraction_equality_l1848_184804

theorem fraction_equality (x y : ℝ) (h : y / x = 1 / 2) : y / (x + y) = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l1848_184804


namespace NUMINAMATH_CALUDE_intersection_A_B_union_A_complement_B_l1848_184891

-- Define the sets A and B
def A : Set ℝ := {x | 2 * x + 1 < 5}
def B : Set ℝ := {x | x^2 - x - 2 < 0}

-- Theorem for A ∩ B
theorem intersection_A_B : A ∩ B = {x : ℝ | -1 < x ∧ x < 2} := by sorry

-- Theorem for A ∪ (ℝ \ B)
theorem union_A_complement_B : A ∪ (Set.univ \ B) = Set.univ := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_union_A_complement_B_l1848_184891


namespace NUMINAMATH_CALUDE_problem_solution_l1848_184835

theorem problem_solution (x y z : ℝ) 
  (h1 : x + y = 5) 
  (h2 : z^2 = x*y + y - 9) : 
  x + 2*y + 3*z = 8 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l1848_184835


namespace NUMINAMATH_CALUDE_dog_grouping_theorem_l1848_184817

/-- The number of ways to divide 12 dogs into groups of 4, 6, and 2,
    with Fluffy in the 4-dog group and Nipper in the 6-dog group -/
def dog_grouping_ways : ℕ := 2520

/-- The total number of dogs -/
def total_dogs : ℕ := 12

/-- The size of the first group (including Fluffy) -/
def group1_size : ℕ := 4

/-- The size of the second group (including Nipper) -/
def group2_size : ℕ := 6

/-- The size of the third group -/
def group3_size : ℕ := 2

theorem dog_grouping_theorem :
  dog_grouping_ways =
    Nat.choose (total_dogs - 2) (group1_size - 1) *
    Nat.choose (total_dogs - group1_size - 1) (group2_size - 1) :=
by sorry

end NUMINAMATH_CALUDE_dog_grouping_theorem_l1848_184817


namespace NUMINAMATH_CALUDE_driver_net_pay_rate_l1848_184803

/-- Calculate the net rate of pay for a driver --/
theorem driver_net_pay_rate (travel_time : ℝ) (speed : ℝ) (fuel_efficiency : ℝ) 
  (payment_rate : ℝ) (gasoline_cost : ℝ) :
  travel_time = 3 ∧ 
  speed = 45 ∧ 
  fuel_efficiency = 36 ∧ 
  payment_rate = 0.60 ∧ 
  gasoline_cost = 2.50 → 
  (payment_rate * speed * travel_time - 
   (speed * travel_time / fuel_efficiency) * gasoline_cost) / travel_time = 23.875 := by
  sorry

end NUMINAMATH_CALUDE_driver_net_pay_rate_l1848_184803


namespace NUMINAMATH_CALUDE_min_value_re_z4_over_re_z4_l1848_184861

theorem min_value_re_z4_over_re_z4 (z : ℂ) (h : (z.re : ℝ) ≠ 0) :
  (z^4).re / (z.re^4 : ℝ) ≥ -8 := by sorry

end NUMINAMATH_CALUDE_min_value_re_z4_over_re_z4_l1848_184861


namespace NUMINAMATH_CALUDE_inequality_proof_l1848_184893

theorem inequality_proof (a₁ a₂ b₁ b₂ : ℝ) 
  (h1 : 0 < a₁) (h2 : 0 < a₂) (h3 : a₁ > a₂) (h4 : b₁ ≥ a₁) (h5 : b₁ * b₂ ≥ a₁ * a₂) : 
  b₁ + b₂ ≥ a₁ + a₂ := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1848_184893


namespace NUMINAMATH_CALUDE_power_sum_theorem_l1848_184828

theorem power_sum_theorem (a : ℝ) (m n : ℕ) (h1 : a^m = 2) (h2 : a^n = 4) : a^(m+n) = 8 := by
  sorry

end NUMINAMATH_CALUDE_power_sum_theorem_l1848_184828


namespace NUMINAMATH_CALUDE_perpendicular_plane_condition_l1848_184823

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the perpendicular relation between planes and between a line and a plane
variable (perp_planes : Plane → Plane → Prop)
variable (perp_line_plane : Line → Plane → Prop)

-- Define the subset relation between a line and a plane
variable (subset : Line → Plane → Prop)

-- State the theorem
theorem perpendicular_plane_condition
  (α β : Plane) (l : Line)
  (h_diff : α ≠ β)
  (h_subset : subset l α) :
  (perp_line_plane l β → perp_planes α β) ∧
  ¬(perp_planes α β → perp_line_plane l β) :=
sorry

end NUMINAMATH_CALUDE_perpendicular_plane_condition_l1848_184823


namespace NUMINAMATH_CALUDE_third_number_tenth_row_l1848_184852

/-- The sum of the first n natural numbers -/
def triangular_number (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The number of integers in the first n rows of the triangular array -/
def numbers_in_rows (n : ℕ) : ℕ := triangular_number (n - 1)

/-- The kth number from the left on the nth row of the triangular array -/
def number_at_position (n : ℕ) (k : ℕ) : ℕ := 
  numbers_in_rows n + k

theorem third_number_tenth_row : 
  number_at_position 10 3 = 48 := by sorry

end NUMINAMATH_CALUDE_third_number_tenth_row_l1848_184852


namespace NUMINAMATH_CALUDE_triangle_inequality_cube_l1848_184812

theorem triangle_inequality_cube (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (htriangle : a + b > c ∧ b + c > a ∧ c + a > b) : a^3 + b^3 + 3*a*b*c > c^3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_cube_l1848_184812


namespace NUMINAMATH_CALUDE_solution_set_implies_sum_l1848_184853

/-- Given that the solution set of ax^2 - bx + 2 < 0 is {x | 1 < x < 2}, prove that a + b = -2 -/
theorem solution_set_implies_sum (a b : ℝ) : 
  (∀ x : ℝ, ax^2 - b*x + 2 < 0 ↔ 1 < x ∧ x < 2) → 
  a + b = -2 := by
  sorry

end NUMINAMATH_CALUDE_solution_set_implies_sum_l1848_184853


namespace NUMINAMATH_CALUDE_painting_wall_percentage_l1848_184825

/-- Calculates the percentage of a wall taken up by a painting -/
theorem painting_wall_percentage 
  (painting_width : ℝ) 
  (painting_height : ℝ) 
  (wall_width : ℝ) 
  (wall_height : ℝ) 
  (h1 : painting_width = 2) 
  (h2 : painting_height = 4) 
  (h3 : wall_width = 5) 
  (h4 : wall_height = 10) : 
  (painting_width * painting_height) / (wall_width * wall_height) * 100 = 16 := by
  sorry

#check painting_wall_percentage

end NUMINAMATH_CALUDE_painting_wall_percentage_l1848_184825


namespace NUMINAMATH_CALUDE_quadratic_discriminant_l1848_184873

/-- The discriminant of a quadratic equation ax^2 + bx + c is b^2 - 4ac -/
def discriminant (a b c : ℝ) : ℝ := b^2 - 4*a*c

/-- The quadratic equation 5x^2 - 9x + 2 has discriminant 41 -/
theorem quadratic_discriminant : discriminant 5 (-9) 2 = 41 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_discriminant_l1848_184873


namespace NUMINAMATH_CALUDE_bus_interval_is_30_minutes_l1848_184859

/-- Represents a bus station schedule -/
structure BusSchedule where
  operatingHoursPerDay : ℕ
  operatingDays : ℕ
  totalBuses : ℕ

/-- Calculates the time interval between bus departures in minutes -/
def calculateInterval (schedule : BusSchedule) : ℕ :=
  let minutesPerDay := schedule.operatingHoursPerDay * 60
  let busesPerDay := schedule.totalBuses / schedule.operatingDays
  minutesPerDay / busesPerDay

/-- Theorem: The time interval between bus departures is 30 minutes -/
theorem bus_interval_is_30_minutes (schedule : BusSchedule) 
    (h1 : schedule.operatingHoursPerDay = 12)
    (h2 : schedule.operatingDays = 5)
    (h3 : schedule.totalBuses = 120) :
  calculateInterval schedule = 30 := by
  sorry

#eval calculateInterval { operatingHoursPerDay := 12, operatingDays := 5, totalBuses := 120 }

end NUMINAMATH_CALUDE_bus_interval_is_30_minutes_l1848_184859


namespace NUMINAMATH_CALUDE_common_factor_proof_l1848_184895

theorem common_factor_proof (m : ℝ) : ∃ (k₁ k₂ : ℝ), 
  m^2 - 4 = (m - 2) * k₁ ∧ m^2 - 4*m + 4 = (m - 2) * k₂ := by
  sorry

end NUMINAMATH_CALUDE_common_factor_proof_l1848_184895


namespace NUMINAMATH_CALUDE_pauls_initial_pens_l1848_184855

theorem pauls_initial_pens (initial_books : ℕ) (books_left : ℕ) (pens_left : ℕ) (books_sold : ℕ) :
  initial_books = 108 →
  books_left = 66 →
  pens_left = 59 →
  books_sold = 42 →
  initial_books - books_left = books_sold →
  ∃ (initial_pens : ℕ), initial_pens = 101 ∧ initial_pens - books_sold = pens_left :=
by sorry

end NUMINAMATH_CALUDE_pauls_initial_pens_l1848_184855


namespace NUMINAMATH_CALUDE_car_speed_problem_l1848_184837

theorem car_speed_problem (distance : ℝ) (original_time : ℝ) (new_time_factor : ℝ) :
  distance = 270 →
  original_time = 6 →
  new_time_factor = 3 / 2 →
  let new_time := new_time_factor * original_time
  let new_speed := distance / new_time
  new_speed = 30 := by
sorry

end NUMINAMATH_CALUDE_car_speed_problem_l1848_184837


namespace NUMINAMATH_CALUDE_z_plus_inv_z_ellipse_l1848_184896

theorem z_plus_inv_z_ellipse (z : ℂ) (h : Complex.abs z = 2) :
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧
  ∀ (x y : ℝ), (z + z⁻¹ = x + y * I) →
  (x^2 / a^2 + y^2 / b^2 = 1) :=
sorry

end NUMINAMATH_CALUDE_z_plus_inv_z_ellipse_l1848_184896
