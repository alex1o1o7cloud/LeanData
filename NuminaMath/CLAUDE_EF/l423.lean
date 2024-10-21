import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_ratio_on_circle_l423_42321

-- Define the circle
def circleEquation (x y : ℤ) : Prop := x^2 + y^2 = 64

-- Define a point on the circle
structure PointOnCircle where
  x : ℤ
  y : ℤ
  on_circle : circleEquation x y

-- Define the distance between two points
noncomputable def distance (p1 p2 : PointOnCircle) : ℝ :=
  Real.sqrt ((p1.x - p2.x)^2 + (p1.y - p2.y)^2 : ℝ)

-- Define the irrationality of a number
def isIrrational (r : ℝ) : Prop := ¬∃ (q : ℚ), r = q

-- Theorem statement
theorem max_ratio_on_circle
  (A B C D E F : PointOnCircle)
  (distinct : A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ A ≠ E ∧ A ≠ F ∧
              B ≠ C ∧ B ≠ D ∧ B ≠ E ∧ B ≠ F ∧
              C ≠ D ∧ C ≠ E ∧ C ≠ F ∧
              D ≠ E ∧ D ≠ F ∧
              E ≠ F)
  (irrational_distances : isIrrational (distance A B) ∧
                          isIrrational (distance C D) ∧
                          isIrrational (distance E F)) :
  (∀ A' B' C' D' E' F' : PointOnCircle,
    (distance A' B' * distance C' D') / distance E' F' ≤ 10 * Real.sqrt 2) ∧
  (∃ A' B' C' D' E' F' : PointOnCircle,
    (distance A' B' * distance C' D') / distance E' F' = 10 * Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_ratio_on_circle_l423_42321


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_equals_one_neither_sufficient_nor_necessary_l423_42396

noncomputable def slope_l1 (a : ℝ) : ℝ := -a / 2
noncomputable def slope_l2 (a : ℝ) : ℝ := -1 / (a + 1)

def lines_parallel (a : ℝ) : Prop := slope_l1 a = slope_l2 a

theorem a_equals_one_neither_sufficient_nor_necessary :
  ¬(∀ a : ℝ, a = 1 → lines_parallel a) ∧
  ¬(∀ a : ℝ, lines_parallel a → a = 1) := by
  sorry

#check a_equals_one_neither_sufficient_nor_necessary

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_equals_one_neither_sufficient_nor_necessary_l423_42396


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_people_is_six_l423_42331

/-- Represents the number of associate professors -/
def A : ℕ := sorry

/-- Represents the number of assistant professors -/
def B : ℕ := sorry

/-- The total number of pencils brought to the meeting -/
def total_pencils : ℕ := 7

/-- The total number of charts brought to the meeting -/
def total_charts : ℕ := 11

/-- The number of pencils each associate professor brings -/
def associate_pencils : ℕ := 2

/-- The number of charts each associate professor brings -/
def associate_charts : ℕ := 1

/-- The number of pencils each assistant professor brings -/
def assistant_pencils : ℕ := 1

/-- The number of charts each assistant professor brings -/
def assistant_charts : ℕ := 2

/-- Theorem stating that the total number of people present is 6 -/
theorem total_people_is_six :
  A * associate_pencils + B * assistant_pencils = total_pencils ∧
  A * associate_charts + B * assistant_charts = total_charts →
  A + B = 6 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_people_is_six_l423_42331


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_and_max_value_and_smallest_k_l423_42341

noncomputable def f (x : ℝ) : ℝ := (2/3) * x^3 - x

noncomputable def f' (x : ℝ) : ℝ := 2 * x^2 - 1

theorem tangent_line_and_max_value_and_smallest_k :
  -- The tangent line at (1, -1/3) has inclination angle θ where tan(θ) = 1
  f' 1 = 1 ∧
  f 1 = -1/3 ∧
  -- The maximum value of f(x) on [-1, 3] is 15
  (∀ x ∈ Set.Icc (-1 : ℝ) 3, f x ≤ 15) ∧
  f 3 = 15 ∧
  -- The smallest positive integer k such that f(x) ≤ k - 1995 for all x ∈ [-1, 3] is 2010
  (∀ x ∈ Set.Icc (-1 : ℝ) 3, f x ≤ 2010 - 1995) ∧
  (∀ k : ℕ, k < 2010 → ∃ x ∈ Set.Icc (-1 : ℝ) 3, f x > k - 1995) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_and_max_value_and_smallest_k_l423_42341


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integer_root_bound_l423_42355

theorem integer_root_bound (n : ℕ) (P : Polynomial ℤ) (x : ℤ) : 
  P.degree = n →
  P.eval x = 0 →
  ∃ M : ℕ, (∀ i, |P.coeff i| ≤ M) ∧ |x| ≤ M :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integer_root_bound_l423_42355


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_two_zeros_implies_b_negative_l423_42365

-- Define the function f
noncomputable def f (x a b : ℝ) : ℝ := (x - 1) * Real.log x - a * x + a + b

-- State the theorem
theorem two_zeros_implies_b_negative :
  (∀ a : ℝ, ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f x₁ a b = 0 ∧ f x₂ a b = 0) →
  b < 0 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_two_zeros_implies_b_negative_l423_42365


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_digits_of_3_20_times_7_15_l423_42378

/-- The number of digits of a positive real number in base 10 -/
noncomputable def numDigits (x : ℝ) : ℕ :=
  Nat.floor (Real.log x / Real.log 10) + 1

/-- The theorem stating that 3^20 * 7^15 has 23 digits in base 10 -/
theorem digits_of_3_20_times_7_15 :
  numDigits ((3 : ℝ)^20 * 7^15) = 23 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_digits_of_3_20_times_7_15_l423_42378


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_symmetry_theorem_l423_42313

-- Define the circle
def my_circle (a : ℝ) (x y : ℝ) : Prop := x^2 + y^2 = 2*a*x

-- Define the line
def my_line (x y : ℝ) : Prop := y = 2*x + 1

-- Define symmetry about a line
def symmetric_about_line (A B : ℝ × ℝ) (l : ℝ → ℝ → Prop) : Prop :=
  ∃ (M : ℝ × ℝ), l M.1 M.2 ∧ 
    (A.1 + B.1) / 2 = M.1 ∧ 
    (A.2 + B.2) / 2 = M.2

theorem circle_symmetry_theorem :
  ∀ (a : ℝ) (A B : ℝ × ℝ),
    my_circle a A.1 A.2 →
    my_circle a B.1 B.2 →
    symmetric_about_line A B my_line →
    a = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_symmetry_theorem_l423_42313


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_needle_intersection_probability_l423_42327

/-- The probability of a needle intersecting parallel lines -/
theorem needle_intersection_probability
  (a l : ℝ) 
  (h_positive_a : 0 < a)
  (h_positive_l : 0 < l)
  (h_l_lt_a : l < a) :
  let probability := (2 * l) / (π * a)
  ∃ (P : ℝ), P = probability ∧ 
    P = (2 * l) / (π * a) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_needle_intersection_probability_l423_42327


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_function_value_l423_42379

-- Define the function g(x) = 2^(x-2)
noncomputable def g (x : ℝ) : ℝ := 2^(x-2)

-- Define the property of f being symmetric to g about y = x
def symmetric_about_y_eq_x (f g : ℝ → ℝ) : Prop :=
  ∀ x y, f x = y ↔ g y = x

-- State the theorem
theorem symmetric_function_value (f : ℝ → ℝ) 
  (h : symmetric_about_y_eq_x f g) : f 8 = 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetric_function_value_l423_42379


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l423_42349

noncomputable def a (x : ℝ) : ℝ × ℝ := (Real.cos x, -1/2)
noncomputable def b (x : ℝ) : ℝ × ℝ := (Real.sqrt 3 * Real.sin x, Real.cos (2*x))

noncomputable def f (x : ℝ) : ℝ := (a x).1 * (b x).1 + (a x).2 * (b x).2

theorem f_properties :
  (∃ T : ℝ, T > 0 ∧ (∀ x, f (x + T) = f x) ∧ (∀ S, 0 < S → S < T → ∃ y, f (y + S) ≠ f y)) ∧
  (∀ x ∈ Set.Icc 0 (Real.pi/2), f x ≤ 1) ∧
  (∀ x ∈ Set.Icc 0 (Real.pi/2), f x ≥ -1/2) ∧
  (∃ x ∈ Set.Icc 0 (Real.pi/2), f x = 1) ∧
  (∃ x ∈ Set.Icc 0 (Real.pi/2), f x = -1/2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l423_42349


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_forecast_is_predict_verb_l423_42317

/-- A type representing words in the English language -/
def Word : Type := String

/-- A predicate that determines if a word is a verb -/
def IsVerb : Word → Prop := sorry

/-- A predicate that determines if a word means "predict; announce in advance" -/
def MeansPredict : Word → Prop := sorry

/-- The word "forecast" -/
def forecast : Word := "forecast"

/-- Theorem stating that "forecast" is a verb that means "predict; announce in advance" -/
theorem forecast_is_predict_verb : IsVerb forecast ∧ MeansPredict forecast := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_forecast_is_predict_verb_l423_42317


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisible_by_three_subset_l423_42342

theorem divisible_by_three_subset (a b c d e : ℤ) : 
  ∃ (x y z : ℤ), x ∈ ({a, b, c, d, e} : Set ℤ) ∧ 
                  y ∈ ({a, b, c, d, e} : Set ℤ) ∧ 
                  z ∈ ({a, b, c, d, e} : Set ℤ) ∧ 
                  x ≠ y ∧ y ≠ z ∧ x ≠ z ∧
                  (x + y + z) % 3 = 0 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisible_by_three_subset_l423_42342


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_intersection_points_l423_42356

/-- A type representing a point in a plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A type representing a line in a plane -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Function to create a line through two points -/
def Line.through (p1 p2 : Point) : Line :=
  { a := p2.y - p1.y,
    b := p1.x - p2.x,
    c := p2.x * p1.y - p1.x * p2.y }

/-- Function to check if two lines are perpendicular -/
def Line.perpendicular (l1 l2 : Line) : Prop :=
  l1.a * l2.a + l1.b * l2.b = 0

/-- Function to check if a line contains a point -/
def Line.contains (l : Line) (p : Point) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- A configuration of 5 points in a plane -/
structure Configuration where
  points : Fin 5 → Point
  not_parallel : ∀ i j k l : Fin 5, i ≠ j → k ≠ l → i ≠ k → j ≠ l → 
    Line.through (points i) (points j) ≠ Line.through (points k) (points l)
  not_perpendicular : ∀ i j k l : Fin 5, i ≠ j → k ≠ l → 
    ¬Line.perpendicular (Line.through (points i) (points j)) (Line.through (points k) (points l))
  not_coincident : ∀ i j k : Fin 5, i ≠ j → i ≠ k → j ≠ k → 
    ¬Line.contains (Line.through (points i) (points j)) (points k)

/-- The number of intersection points of perpendicular lines -/
def num_intersection_points (c : Configuration) : ℕ := sorry

/-- The theorem stating the maximum number of intersection points -/
theorem max_intersection_points (c : Configuration) : 
  num_intersection_points c = 310 := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_intersection_points_l423_42356


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_M_intersect_N_equals_closed_interval_l423_42361

-- Define the universal set U as ℝ
def U : Set ℝ := Set.univ

-- Define set M
def M : Set ℝ := {x : ℝ | x^2 + x - 2 > 0}

-- Define set N
def N : Set ℝ := {x : ℝ | (1/2 : ℝ)^(x-1) ≥ 2}

-- State the theorem
theorem complement_M_intersect_N_equals_closed_interval :
  (Set.compl M ∩ N) = Set.Icc (-2) 0 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_M_intersect_N_equals_closed_interval_l423_42361


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_next_number_is_131_l423_42344

def mySequence : List Nat := [12, 13, 15, 17, 111, 113, 117, 119, 123, 129]

def next_number (seq : List Nat) : Nat :=
  match seq.reverse with
  | [] => 0  -- Empty list case
  | [x] => x + 2  -- Single element case
  | x :: y :: _ => 
    if x - y = 6 then x + 2  -- If last difference was 6, add 2
    else if x - y = 2 then x + 4  -- If last difference was 2, add 4
    else x + 2  -- Default case, add 2

theorem next_number_is_131 : next_number mySequence = 131 := by
  rw [next_number, mySequence]
  rfl

#eval next_number mySequence

end NUMINAMATH_CALUDE_ERRORFEEDBACK_next_number_is_131_l423_42344


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_slope_at_one_l423_42386

-- Define the function f(x) = (2x - 1) / e^x
noncomputable def f (x : ℝ) : ℝ := (2 * x - 1) / Real.exp x

-- State the theorem
theorem tangent_slope_at_one :
  HasDerivAt f (1 / Real.exp 1) 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_slope_at_one_l423_42386


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_a_and_b_l423_42300

-- Define the vector space
variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

-- Define the vectors
variable (e₁ e₂ : V)

-- Define the conditions
variable (h₁ : ‖e₁‖ = 1)
variable (h₂ : ‖e₂‖ = 1)
variable (h₃ : Real.cos (60 * π / 180) = inner e₁ e₂)

-- Define vectors a and b
def a : V := 2 • e₁ + e₂
def b : V := -3 • e₁ + 2 • e₂

-- State the theorem
theorem angle_between_a_and_b :
  Real.arccos (inner (a e₁ e₂) (b e₁ e₂) / (‖a e₁ e₂‖ * ‖b e₁ e₂‖)) = 120 * π / 180 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_between_a_and_b_l423_42300


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_count_theorem_l423_42357

def floor_division_list : List ℕ := (List.range 1000).map (fun n => (((n + 1)^2 : ℚ) / 500).floor.toNat)

theorem distinct_count_theorem : 
  (floor_division_list.eraseDups).length = 876 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_count_theorem_l423_42357


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_zero_quadratic_l423_42306

/-- A quadratic function f(x) = x^2 - 2x + b has a unique zero point in the interval (2,4) if and only if b is in the open interval (-8, 0) -/
theorem unique_zero_quadratic (b : ℝ) :
  (∃! x, x ∈ Set.Ioo 2 4 ∧ x^2 - 2*x + b = 0) ↔ b ∈ Set.Ioo (-8) 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_zero_quadratic_l423_42306


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_i_squared_imaginary_power_product_l423_42337

-- Define the imaginary unit i
noncomputable def i : ℂ := Complex.I

-- Define the property of i
theorem i_squared : i^2 = -1 := by
  rw [i, Complex.I_sq]

-- State the theorem
theorem imaginary_power_product : i^17 * i^44 = i := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_i_squared_imaginary_power_product_l423_42337


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_joey_age_digit_sum_l423_42398

/-- Represents a person with an age -/
structure Person where
  age : ℕ

/-- Checks if a number is prime -/
def isPrime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

/-- Calculates the sum of digits of a natural number -/
def sumOfDigits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sumOfDigits (n / 10)

/-- Theorem statement -/
theorem joey_age_digit_sum :
  ∀ (joey chloe max : Person),
    joey.age = chloe.age + 2 →
    max.age = 2 →
    isPrime chloe.age →
    (∀ k : ℕ, k < chloe.age → ¬isPrime k) →
    (∃ n : ℕ, (joey.age + n) % (max.age + n) = 0) →
    ∃ n : ℕ, sumOfDigits (joey.age + n) = 5 ∧ (joey.age + n) % (max.age + n) = 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_joey_age_digit_sum_l423_42398


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_implies_positive_derivative_l423_42389

theorem tangent_line_implies_positive_derivative (f : ℝ → ℝ) (x : ℝ) 
  (h : HasDerivAt f 2 x) : 
  HasDerivAt f (2 : ℝ) x ∧ (2 : ℝ) > 0 := by
  constructor
  · exact h
  · exact two_pos


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_implies_positive_derivative_l423_42389


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_vector_l423_42363

def a : ℝ × ℝ := (-1, -1)
def b : ℝ × ℝ := (-3, 4)

theorem projection_vector :
  let proj := ((5 * (a.1 * b.1 + a.2 * b.2)) / (b.1^2 + b.2^2)) • b
  proj = (3/5, -4/5) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_vector_l423_42363


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_dice_probability_l423_42377

theorem dice_probability : 
  let n : ℕ := 4  -- number of dice
  let s : ℕ := 6  -- number of sides on each die
  let f : ℕ := 4  -- number of favorable outcomes per die (2, 3, 4, 5)
  let probability_no_one_or_six : ℚ := (f / s) ^ n
  probability_no_one_or_six = 16 / 81 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_dice_probability_l423_42377


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_expression_l423_42366

theorem tan_alpha_expression (α : Real) (h : Real.tan α = 3) :
  (Real.sin (α - Real.pi) + Real.cos (Real.pi - α)) / (Real.sin (Real.pi / 2 - α) + Real.cos (Real.pi / 2 + α)) = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_alpha_expression_l423_42366


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_from_intersection_to_l₃_l423_42359

-- Define the lines
def l₁ (x y : ℝ) : Prop := x - 2*y + 4 = 0
def l₂ (x y : ℝ) : Prop := x + y - 2 = 0
def l₃ (x y : ℝ) : Prop := 3*x - 4*y + 5 = 0

-- Define the intersection point
def intersection_point (P : ℝ × ℝ) : Prop :=
  l₁ P.1 P.2 ∧ l₂ P.1 P.2

-- Define the distance function from a point to a line
noncomputable def distance_to_line (P : ℝ × ℝ) : ℝ :=
  |3 * P.1 - 4 * P.2 + 5| / Real.sqrt (3^2 + (-4)^2)

-- Theorem statement
theorem distance_from_intersection_to_l₃ :
  ∃ P : ℝ × ℝ, intersection_point P ∧ distance_to_line P = 3/5 := by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_from_intersection_to_l₃_l423_42359


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_tangent_line_to_circle_l423_42376

/-- The value of c for which there is only one tangent line to the circle (x-1)^2 + (y-1)^2 = 9 passing through the point (1, c) -/
theorem unique_tangent_line_to_circle (c : ℝ) : 
  c > 0 → 
  (∃! l : Set (ℝ × ℝ), 
    (∀ p : ℝ × ℝ, p ∈ l → (p.1 - 1)^2 + (p.2 - 1)^2 = 9) ∧
    (1, c) ∈ l ∧
    (∀ p : ℝ × ℝ, p ∈ l → (p.1 - 1)^2 + (p.2 - 1)^2 ≤ 9)) →
  c = 4 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_tangent_line_to_circle_l423_42376


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_variance_with_zero_and_one_l423_42373

variable (n : ℕ)

def is_valid_set (s : Finset ℝ) : Prop :=
  s.card = n ∧ 0 ∈ s ∧ 1 ∈ s

noncomputable def variance (s : Finset ℝ) : ℝ :=
  let mean := (s.sum id) / s.card
  (s.sum (fun x => (x - mean)^2)) / s.card

theorem min_variance_with_zero_and_one (n : ℕ) (h : n ≥ 2) :
  ∃ (s : Finset ℝ), is_valid_set n s ∧
    (∀ (t : Finset ℝ), is_valid_set n t → variance t ≥ variance s) ∧
    variance s = 1 / (2 * n) ∧
    (∀ x ∈ s, x = 0 ∨ x = 1 ∨ x = 1/2) :=
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_variance_with_zero_and_one_l423_42373


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_perfect_square_to_342_l423_42380

theorem closest_perfect_square_to_342 :
  ∀ n : ℕ, n ≠ 0 → n^2 ≠ 324 → |342 - 324| ≤ |342 - (n^2 : ℤ)| :=
by
  intro n h1 h2
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_perfect_square_to_342_l423_42380


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_platform_length_is_800_l423_42388

/-- Calculates the length of a platform given train specifications -/
noncomputable def platformLength (trainLength : ℝ) (timePlatform : ℝ) (timePole : ℝ) : ℝ :=
  trainLength * (timePlatform / timePole - 1)

/-- Theorem: The platform length is 800 meters given the specified conditions -/
theorem platform_length_is_800 :
  platformLength 500 65 25 = 800 := by
  -- Unfold the definition of platformLength
  unfold platformLength
  -- Perform the calculation
  norm_num
  -- QED

end NUMINAMATH_CALUDE_ERRORFEEDBACK_platform_length_is_800_l423_42388


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_non_monotonic_implies_a_gt_4_l423_42322

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := 2 * x^2 - a * x + Real.log x

-- Define the derivative of f
noncomputable def f_deriv (a : ℝ) (x : ℝ) : ℝ := 4 * x - a + 1 / x

-- State the theorem
theorem non_monotonic_implies_a_gt_4 :
  ∀ a : ℝ, (∃ x y : ℝ, x > 0 ∧ y > 0 ∧ x < y ∧
    ((f_deriv a x < 0 ∧ f_deriv a y > 0) ∨
     (f_deriv a x > 0 ∧ f_deriv a y < 0))) →
  a > 4 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_non_monotonic_implies_a_gt_4_l423_42322


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_points_of_cosine_function_l423_42328

-- Define the function f as noncomputable due to Real.cos
noncomputable def f (ω φ : ℝ) (x : ℝ) : ℝ := Real.cos (ω * x + φ)

-- State the theorem
theorem zero_points_of_cosine_function 
  (ω φ : ℝ) 
  (h_ω_pos : ω > 0) 
  (h_φ_bound : |φ| < π / 2) 
  (h_period : ∀ x, f ω φ (x + π / ω) = f ω φ x) 
  (h_symmetry : ∀ x, f ω φ (-π / 12 + x) = f ω φ (-π / 12 - x)) :
  ∃! (zeros : Finset ℝ), 
    zeros.card = 8 ∧ 
    (∀ x ∈ zeros, -2 * π ≤ x ∧ x ≤ 2 * π) ∧
    (∀ x ∈ zeros, f ω φ x = 0) ∧
    (∀ x, -2 * π ≤ x ∧ x ≤ 2 * π → f ω φ x = 0 → x ∈ zeros) :=
by
  sorry  -- Skip the proof for now


end NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_points_of_cosine_function_l423_42328


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_intersect_l423_42304

/-- Circle O₁ with equation x² + y² - 2x = 0 -/
def circle_O₁ (x y : ℝ) : Prop := x^2 + y^2 - 2*x = 0

/-- Circle O₂ with equation x² + y² - 6y = 0 -/
def circle_O₂ (x y : ℝ) : Prop := x^2 + y^2 - 6*y = 0

/-- Center of circle O₁ -/
def center_O₁ : ℝ × ℝ := (1, 0)

/-- Center of circle O₂ -/
def center_O₂ : ℝ × ℝ := (0, 3)

/-- Radius of circle O₁ -/
def radius_O₁ : ℝ := 1

/-- Radius of circle O₂ -/
def radius_O₂ : ℝ := 3

/-- Distance between centers of O₁ and O₂ -/
noncomputable def center_distance : ℝ := Real.sqrt 10

/-- Theorem stating that circles O₁ and O₂ are intersecting -/
theorem circles_intersect : 
  radius_O₂ - radius_O₁ < center_distance ∧ 
  center_distance < radius_O₁ + radius_O₂ := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circles_intersect_l423_42304


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_wang_shopping_payments_l423_42352

/-- Represents the discount policy of the shopping mall -/
noncomputable def discount (amount : ℝ) : ℝ :=
  if amount ≤ 100 then 0
  else if amount ≤ 300 then 0.1 * amount
  else 0.1 * 300 + 0.2 * (amount - 300)

/-- Calculates the actual payment after discount -/
noncomputable def actualPayment (amount : ℝ) : ℝ :=
  amount - discount amount

theorem wang_shopping_payments (x y : ℝ) 
  (h1 : x > 100 ∧ x ≤ 300) 
  (h2 : y > 300)
  (h3 : actualPayment x + actualPayment y + 19 = actualPayment (x + y))
  (h4 : x + y - (actualPayment x + actualPayment y) = 67) :
  actualPayment x = 171 ∧ actualPayment y = 342 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_wang_shopping_payments_l423_42352


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_determinant_solution_l423_42375

/-- Given a ≠ 0 and b = 0, prove that x = -a/4 is a solution to the determinant equation. -/
theorem determinant_solution (a : ℝ) (ha : a ≠ 0) :
  let x : ℝ := -a/4
  let matrix : Matrix (Fin 3) (Fin 3) ℝ := !![x + a, 0, x; 0, x + a, 0; x, x, x + a]
  Matrix.det matrix = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_determinant_solution_l423_42375


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_perimeter_l423_42312

-- Define the square side length
def square_side : ℝ := 75

-- Define the heights of the three rectangles
variable (x y z : ℝ)

-- Define the condition that the heights sum to the square side
def heights_sum (x y z : ℝ) : Prop := x + y + z = square_side

-- Define the perimeters of the three rectangles
def P₁ (x : ℝ) : ℝ := 2 * (x + square_side)
def P₂ (y : ℝ) : ℝ := 2 * (y + square_side)
def P₃ (z : ℝ) : ℝ := 2 * (z + square_side)

-- Define the condition that one perimeter is half the sum of the other two
def perimeter_condition (x y z : ℝ) : Prop := P₁ x = (P₂ y + P₃ z) / 2

-- Theorem statement
theorem rectangle_perimeter (x y z : ℝ) : 
  heights_sum x y z → perimeter_condition x y z → P₁ x = 200 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_perimeter_l423_42312


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_Matthew_hotdogs_l423_42308

/-- The number of hotdogs Matthew needs to cook --/
noncomputable def hotdogs_needed : ℕ :=
  let ella_emma := (2.5 + 2.5 : ℝ)
  let luke := (2.5 : ℝ) ^ 2
  let michael := (7 : ℝ)
  let hunter := 1.25 * ella_emma
  let zoe := (0.6 : ℝ)
  let uncle_steve := 1.5 * (ella_emma + luke)
  ⌈(ella_emma + luke + michael + hunter + zoe + uncle_steve : ℝ)⌉.toNat

/-- Theorem stating the minimum number of whole hotdogs Matthew needs to cook --/
theorem Matthew_hotdogs : hotdogs_needed = 47 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_Matthew_hotdogs_l423_42308


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_radius_in_regular_hexagon_l423_42372

-- Define the regular hexagon
def regular_hexagon (side_length : ℝ) : Set (ℝ × ℝ) := sorry

-- Define the circle
def circle_set (center : ℝ × ℝ) (radius : ℝ) : Set (ℝ × ℝ) := sorry

-- Define the vertices of the hexagon
def A : ℝ × ℝ := sorry
def F : ℝ × ℝ := sorry

-- Define the side CD
def CD : Set (ℝ × ℝ) := sorry

-- Theorem statement
theorem circle_radius_in_regular_hexagon 
  (h : regular_hexagon 10)
  (c : ∃ (center : ℝ × ℝ) (radius : ℝ), 
    circle_set center radius ∩ {A, F} = {A, F} ∧ 
    ∃ (p : ℝ × ℝ), p ∈ CD ∧ p ∈ circle_set center radius ∧
    ∀ (q : ℝ × ℝ), q ∈ CD → q ≠ p → q ∉ circle_set center radius) :
  ∃ (center : ℝ × ℝ), circle_set center 20 ∩ {A, F} = {A, F} ∧ 
    ∃ (p : ℝ × ℝ), p ∈ CD ∧ p ∈ circle_set center 20 ∧
    ∀ (q : ℝ × ℝ), q ∈ CD → q ≠ p → q ∉ circle_set center 20 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_radius_in_regular_hexagon_l423_42372


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_locust_doubling_time_l423_42395

/-- The doubling time of the locust population -/
noncomputable def doubling_time : ℝ := 2

/-- The initial population of locusts 4 hours ago -/
noncomputable def initial_population : ℝ := 1000

/-- The population after t hours, given the doubling time h -/
noncomputable def population (t : ℝ) (h : ℝ) : ℝ := initial_population * (2 ^ (t / h))

/-- The theorem stating that the doubling time is 2 hours -/
theorem locust_doubling_time :
  ∃ (h : ℝ), h > 0 ∧ h = doubling_time ∧ 
  population (10 + 4) h > 128000 ∧
  ∀ (t : ℝ), population t h = initial_population * (2 ^ (t / h)) := by
  sorry

#check locust_doubling_time

end NUMINAMATH_CALUDE_ERRORFEEDBACK_locust_doubling_time_l423_42395


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_phi_is_cone_l423_42315

/-- Spherical coordinates -/
structure SphericalCoord where
  ρ : ℝ
  θ : ℝ
  φ : ℝ
  ρ_nonneg : 0 ≤ ρ
  θ_range : 0 ≤ θ ∧ θ < 2 * Real.pi
  φ_range : 0 ≤ φ ∧ φ ≤ Real.pi

/-- The set of points in spherical coordinates satisfying φ = c -/
def ConstantPhiSet (c : ℝ) : Set SphericalCoord :=
  {p : SphericalCoord | p.φ = c ∧ 0 ≤ c ∧ c ≤ Real.pi}

/-- Definition of a cone in spherical coordinates -/
def IsCone (S : Set SphericalCoord) : Prop :=
  ∃ c : ℝ, 0 ≤ c ∧ c ≤ Real.pi ∧ S = ConstantPhiSet c

/-- Theorem: The set of points satisfying φ = c forms a cone -/
theorem constant_phi_is_cone (c : ℝ) (h : 0 ≤ c ∧ c ≤ Real.pi) :
  IsCone (ConstantPhiSet c) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_phi_is_cone_l423_42315


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_implies_b_value_l423_42314

noncomputable def vector_projection (v u : ℝ × ℝ) : ℝ × ℝ :=
  let dot_product := v.1 * u.1 + v.2 * u.2
  let norm_squared := u.1 * u.1 + u.2 * u.2
  (dot_product / norm_squared) • u

theorem projection_implies_b_value (b : ℝ) :
  let v : ℝ × ℝ := (-3, b)
  let u : ℝ × ℝ := (3, 2)
  vector_projection v u = (-5/13 : ℝ) • u → b = 2 := by
  intro h
  -- Proof steps would go here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_implies_b_value_l423_42314


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_angles_cosine_l423_42347

theorem chord_angles_cosine (α β : Real) : 
  -- Conditions
  6^2 + 8^2 = 10^2 →  -- Right triangle formed by chord endpoints and center
  α + 2*β < π →       -- Given condition
  α/2 + β = π/2 →     -- Right angle in the triangle
  Real.sin β = 6/10 →      -- Sine of β in the right triangle
  Real.cos β = 8/10 →      -- Cosine of β in the right triangle
  Real.cos (α + 2*β) = 0 → -- α + 2β forms a right angle
  -- Conclusion
  Real.cos α = 24/25 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_angles_cosine_l423_42347


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_formula_l423_42301

def mySequence (n : ℕ+) : ℚ := 1 / n.val

theorem sequence_formula : ∀ (n : ℕ+), mySequence n = 1 / n.val := by
  intro n
  rfl

#eval mySequence 1
#eval mySequence 2
#eval mySequence 3

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_formula_l423_42301


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_weight_ratio_problem_l423_42318

theorem weight_ratio_problem (ram_weight shyam_weight : ℝ) 
  (h1 : ram_weight + shyam_weight = 72)
  (h2 : 1.1 * ram_weight + 1.22 * shyam_weight = 82.8) :
  ram_weight / shyam_weight = 7 / 5 := by
  sorry

#check weight_ratio_problem

end NUMINAMATH_CALUDE_ERRORFEEDBACK_weight_ratio_problem_l423_42318


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_adult_ticket_is_five_l423_42353

/-- Calculates the cost of an adult ticket given total sales, total tickets, child tickets, and child ticket price. -/
noncomputable def adult_ticket_cost (total_sales : ℝ) (total_tickets : ℕ) (child_tickets : ℕ) (child_price : ℝ) : ℝ :=
  (total_sales - (child_tickets : ℝ) * child_price) / ((total_tickets : ℝ) - (child_tickets : ℝ))

/-- Theorem stating that the adult ticket cost is 5 dollars given the problem conditions. -/
theorem adult_ticket_is_five :
  adult_ticket_cost 178 42 16 3 = 5 := by
  -- Unfold the definition of adult_ticket_cost
  unfold adult_ticket_cost
  -- Simplify the expression
  simp
  -- The proof is completed
  norm_num
  -- If norm_num doesn't complete the proof, you can use sorry
  -- sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_adult_ticket_is_five_l423_42353


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_extrema_l423_42371

-- Define the function f(x)
noncomputable def f (x : ℝ) : ℝ := 3 * Real.sin (x / 2 + Real.pi / 6) + 3

-- Define the interval
def I : Set ℝ := Set.Icc (Real.pi / 3) (4 * Real.pi / 3)

-- State the theorem
theorem f_extrema :
  ∃ (max_x min_x : ℝ),
    max_x ∈ I ∧ min_x ∈ I ∧
    (∀ x ∈ I, f x ≤ f max_x) ∧
    (∀ x ∈ I, f x ≥ f min_x) ∧
    f max_x = 6 ∧
    f min_x = 9/2 ∧
    max_x = 2*Real.pi/3 ∧
    min_x = 4*Real.pi/3 := by
  sorry

#check f_extrema

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_extrema_l423_42371


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_speed_minimizes_cost_l423_42333

/-- Represents the fuel cost per unit time as a function of speed -/
noncomputable def fuel_cost (v : ℝ) : ℝ := (1/200) * v^3

/-- Represents the total cost per kilometer as a function of speed -/
noncomputable def cost_per_km (v : ℝ) : ℝ := fuel_cost v / v + 270 / v

/-- The optimal speed that minimizes the cost per kilometer -/
def optimal_speed : ℝ := 30

/-- The minimum cost per kilometer -/
def min_cost : ℝ := 13.5

theorem optimal_speed_minimizes_cost :
  (∀ v : ℝ, v > 0 → cost_per_km v ≥ min_cost) ∧
  cost_per_km optimal_speed = min_cost := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_speed_minimizes_cost_l423_42333


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_log_function_l423_42381

/-- Given a logarithmic function y = log_a(x) where a > 0 and a ≠ 1,
    if the graph passes through the point (2, 1/2),
    then its inverse function is y = 4^x -/
theorem inverse_log_function (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  (∃ f : ℝ → ℝ, ∀ x > 0, f x = Real.log x / Real.log a) →
  (Real.log 2 / Real.log a = 1/2) →
  (∃ g : ℝ → ℝ, ∀ x, g x = a^x ∧ a = 4) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_log_function_l423_42381


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_A_is_referee_round_11_loser_of_round_10_is_A_l423_42383

/-- Represents a person in the competition -/
inductive Person : Type
  | A
  | B
  | C

/-- Represents the referee for a given round -/
def referee : ℕ → Person := sorry

/-- Total number of rounds in the competition -/
def total_rounds : ℕ := 25

/-- Number of rounds A is the referee -/
def A_referee_rounds : ℕ := 13

/-- Number of rounds B is the referee -/
def B_referee_rounds : ℕ := 4

/-- Number of rounds C is the referee -/
def C_referee_rounds : ℕ := 8

/-- The referee changes every round -/
axiom referee_changes (n : ℕ) : n < total_rounds → referee n ≠ referee (n + 1)

/-- The sum of referee rounds for all persons equals the total rounds -/
axiom sum_referee_rounds : A_referee_rounds + B_referee_rounds + C_referee_rounds = total_rounds

/-- A is the referee for the 11th round -/
theorem A_is_referee_round_11 : referee 11 = Person.A := by sorry

/-- The loser of the 10th round is A -/
theorem loser_of_round_10_is_A : referee 11 = Person.A → Person.A = sorry := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_A_is_referee_round_11_loser_of_round_10_is_A_l423_42383


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_through_point_l423_42316

/-- A power function that passes through the point (4, 2) -/
noncomputable def f (α : ℝ) : ℝ → ℝ := fun x ↦ x ^ α

theorem power_function_through_point (α : ℝ) (h : f α 4 = 2) : f α 16 = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_through_point_l423_42316


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_range_l423_42392

theorem equation_solution_range (a : ℝ) :
  (∃ x : ℝ, Real.sin x ^ 2 + 3 * a ^ 2 * Real.cos x - 2 * a ^ 2 * (3 * a - 2) - 1 = 0) →
  -1/2 ≤ a ∧ a ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_range_l423_42392


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_hyperbola_circle_l423_42332

/-- Given a hyperbola and a circle, prove the length of the chord formed by their intersection --/
theorem chord_length_hyperbola_circle (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let hyperbola := fun x y ↦ x^2 / a^2 - y^2 / b^2 = 1
  let asymptote := fun x y ↦ y = 2 * x
  let circle := fun x y ↦ (x - 2)^2 + y^2 = 16
  let chord_length := 16 * Real.sqrt 5 / 5
  (∃ x y, hyperbola x y ∧ asymptote x y ∧ asymptote 3 6) →
  chord_length = Real.sqrt (4 * (16 - (4 / Real.sqrt 5)^2)) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_chord_length_hyperbola_circle_l423_42332


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_books_sold_on_thursday_l423_42397

/-- Calculates the number of books sold on Thursday given the total stock,
    sales on other days, and the percentage of unsold books. -/
theorem books_sold_on_thursday
  (total_stock : ℕ)
  (monday_sales tuesday_sales wednesday_sales friday_sales : ℕ)
  (unsold_percentage : ℚ)
  (h1 : total_stock = 1400)
  (h2 : monday_sales = 62)
  (h3 : tuesday_sales = 62)
  (h4 : wednesday_sales = 60)
  (h5 : friday_sales = 40)
  (h6 : unsold_percentage = 80.57142857142857 / 100) :
  ∃ (thursday_sales : ℕ),
    thursday_sales = total_stock -
      (Int.toNat ⌊(unsold_percentage : ℚ) * total_stock⌋) -
      (monday_sales + tuesday_sales + wednesday_sales + friday_sales) ∧
    thursday_sales = 48 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_books_sold_on_thursday_l423_42397


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integer_differences_l423_42338

theorem integer_differences (S : Finset ℕ) : 
  (S.card = 55) → 
  (∀ n, n ∈ S → 1 ≤ n ∧ n ≤ 99) → 
  (∃ a b, a ∈ S ∧ b ∈ S ∧ a ≠ b ∧ (a - b = 9 ∨ a - b = 10 ∨ a - b = 12 ∨ a - b = 13)) ∧
  (∃ T : Finset ℕ, T.card = 55 ∧ (∀ n, n ∈ T → 1 ≤ n ∧ n ≤ 99) ∧ (∀ a b, a ∈ T → b ∈ T → a ≠ b → a - b ≠ 11)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integer_differences_l423_42338


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_trig_expression_l423_42394

theorem max_value_trig_expression (θ1 θ2 θ3 θ4 θ5 : ℝ) :
  Real.cos θ1 * Real.sin θ2 + 2 * Real.cos θ2 * Real.sin θ3 + 3 * Real.cos θ3 * Real.sin θ4 + 
  4 * Real.cos θ4 * Real.sin θ5 + 5 * Real.cos θ5 * Real.sin θ1 ≤ 15 / 2 ∧
  ∃ θ1' θ2' θ3' θ4' θ5' : ℝ,
    Real.cos θ1' * Real.sin θ2' + 2 * Real.cos θ2' * Real.sin θ3' + 3 * Real.cos θ3' * Real.sin θ4' + 
    4 * Real.cos θ4' * Real.sin θ5' + 5 * Real.cos θ5' * Real.sin θ1' = 15 / 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_trig_expression_l423_42394


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_surface_area_of_circumscribed_sphere_theorem_l423_42335

-- Define the tetrahedron P-ABC
structure Tetrahedron where
  A : ℝ × ℝ × ℝ
  B : ℝ × ℝ × ℝ
  C : ℝ × ℝ × ℝ
  P : ℝ × ℝ × ℝ

-- Define auxiliary functions
noncomputable def distance (a b : ℝ × ℝ × ℝ) : ℝ := sorry
noncomputable def dihedral_angle (p a b c : ℝ × ℝ × ℝ) : ℝ := sorry
noncomputable def surface_area_of_circumscribed_sphere (t : Tetrahedron) : ℝ := sorry

-- Define the conditions
def is_equilateral_triangle (t : Tetrahedron) : Prop :=
  let d := Real.sqrt 3 * 2
  distance t.A t.B = d ∧ distance t.B t.C = d ∧ distance t.C t.A = d

def has_specific_edges (t : Tetrahedron) : Prop :=
  distance t.P t.B = Real.sqrt 5 ∧ distance t.P t.C = Real.sqrt 5

def has_specific_dihedral_angle (t : Tetrahedron) : Prop :=
  dihedral_angle t.P t.B t.C t.A = Real.pi / 4

-- Define the theorem
theorem surface_area_of_circumscribed_sphere_theorem (t : Tetrahedron) :
  is_equilateral_triangle t →
  has_specific_edges t →
  has_specific_dihedral_angle t →
  surface_area_of_circumscribed_sphere t = 25 * Real.pi := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_surface_area_of_circumscribed_sphere_theorem_l423_42335


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_non_parallel_sides_lengths_l423_42399

/-- A trapezoid with the given properties -/
structure Trapezoid (a b : ℝ) where
  -- Ensures a and b are positive
  ha : 0 < a
  hb : 0 < b
  -- BD is perpendicular to AD and BC
  perp_diagonal : True
  -- Sum of acute angles A and C is 90°
  angle_sum : True

/-- The lengths of non-parallel sides of the trapezoid -/
noncomputable def non_parallel_sides (t : Trapezoid a b) : ℝ × ℝ :=
  (Real.sqrt (b * (a + b)), Real.sqrt (a * (a + b)))

/-- Theorem stating the lengths of non-parallel sides -/
theorem non_parallel_sides_lengths
  (a b : ℝ) (t : Trapezoid a b) :
  non_parallel_sides t = (Real.sqrt (b * (a + b)), Real.sqrt (a * (a + b))) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_non_parallel_sides_lengths_l423_42399


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_projections_l423_42339

/-- Represents a tetrahedron with specific projection properties -/
structure Tetrahedron where
  -- One projection is a trapezoid with area 1
  trapezoid_projection_area : ℝ := 1

/-- Predicate to check if a projection is a square -/
def is_square (area : ℝ) : Prop := ∃ (side : ℝ), side > 0 ∧ side * side = area

/-- Theorem stating the properties of the tetrahedron's projections -/
theorem tetrahedron_projections (T : Tetrahedron) :
  -- No projection can be a square with area 1
  (∀ (projection_area : ℝ), projection_area = 1 → ¬ is_square projection_area) ∧
  -- There exists a projection that is a square with area 1/2019
  (∃ (projection_area : ℝ), projection_area = 1/2019 ∧ is_square projection_area) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tetrahedron_projections_l423_42339


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_in_third_quadrant_implies_angle_in_second_quadrant_l423_42346

-- Define the point P
noncomputable def P (θ : Real) : (Real × Real) := (Real.sin θ * Real.cos θ, 2 * Real.cos θ)

-- Define the condition for a point to be in the third quadrant
def inThirdQuadrant (p : Real × Real) : Prop :=
  p.1 < 0 ∧ p.2 < 0

-- Define the condition for an angle to be in the second quadrant
def inSecondQuadrant (θ : Real) : Prop :=
  Real.sin θ > 0 ∧ Real.cos θ < 0

-- Theorem statement
theorem point_in_third_quadrant_implies_angle_in_second_quadrant (θ : Real) :
  inThirdQuadrant (P θ) → inSecondQuadrant θ :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_in_third_quadrant_implies_angle_in_second_quadrant_l423_42346


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complete_coverage_l423_42360

/-- The minimum number of circles required to cover a larger circle -/
def min_covering_circles : ℕ := 7

/-- The radius of the larger circle -/
noncomputable def R : ℝ := 1  -- Assigning a value to R for concreteness

/-- The radius of the smaller circles -/
noncomputable def small_radius : ℝ := R / 2

/-- A circle is represented by its center and radius -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- The larger circle to be covered -/
noncomputable def large_circle : Circle := { center := (0, 0), radius := R }

/-- A list of smaller circles that cover the larger circle -/
noncomputable def covering_circles : List Circle := sorry

theorem complete_coverage : 
  (covering_circles.length = min_covering_circles) ∧ 
  (∀ p : ℝ × ℝ, ‖p - large_circle.center‖ ≤ large_circle.radius → 
    ∃ c ∈ covering_circles, ‖p - c.center‖ ≤ c.radius) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complete_coverage_l423_42360


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_number_of_bottles_l423_42325

/-- The smallest number of bottles of milk Christine should purchase -/
def smallest_number_of_bottles : ℕ :=
  let required_fl_oz : ℚ := 60
  let bottle_size_ml : ℚ := 250
  let fl_oz_per_liter : ℚ := 32
  let ml_per_liter : ℚ := 1000
  
  let required_ml : ℚ := required_fl_oz * ml_per_liter / fl_oz_per_liter
  let bottles_needed : ℚ := required_ml / bottle_size_ml
  
  (bottles_needed.ceil).toNat

theorem correct_number_of_bottles :
  smallest_number_of_bottles = 8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_correct_number_of_bottles_l423_42325


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_sum_of_bi_l423_42303

def powerTower (a : List ℕ) : ℕ :=
  match a with
  | [] => 1
  | x::xs => x^(powerTower xs)

def congruent_mod (a b m : ℕ) : Prop := a % m = b % m

theorem smallest_sum_of_bi (n : ℕ) (h : n = 2017) :
  ∃ (b : List ℕ),
    (∀ (a : List ℕ), a.length = n → (∀ x ∈ a, x > n) →
      ∀ i, i ∈ List.range n →
        congruent_mod
          (powerTower (a.take i ++ [a.get! i + b.get! i] ++ a.drop (i+1)))
          (powerTower a)
          n) →
    b.length = n →
    (∀ x ∈ b, x > 0) →
    b.sum = 4983 ∧ 
    (∀ (c : List ℕ), c.length = n → (∀ x ∈ c, x > 0) →
      (∀ (a : List ℕ), a.length = n → (∀ x ∈ a, x > n) →
        ∀ i, i ∈ List.range n →
          congruent_mod
            (powerTower (a.take i ++ [a.get! i + c.get! i] ++ a.drop (i+1)))
            (powerTower a)
            n) →
      c.sum ≥ 4983) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_sum_of_bi_l423_42303


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_translation_problem_l423_42364

-- Define a translation of the complex plane
def translation (w : ℂ) : ℂ → ℂ := fun z ↦ z + w

-- State the theorem
theorem translation_problem :
  ∃ w : ℂ, translation w (1 - 3*I) = 4 + 2*I ∧ translation w (6 - 4*I) = 9 + I :=
by
  -- We'll use w = 3 + 5*I as found in the solution
  use 3 + 5*I
  constructor
  · -- Prove the first part: translation w (1 - 3*I) = 4 + 2*I
    simp [translation]
    ring
  · -- Prove the second part: translation w (6 - 4*I) = 9 + I
    simp [translation]
    ring


end NUMINAMATH_CALUDE_ERRORFEEDBACK_translation_problem_l423_42364


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_numbers_divisibility_l423_42324

theorem odd_numbers_divisibility (n : ℕ+) 
  (selected : Finset ℕ) 
  (h_size : selected.card = 2^(2*n.val - 1) + 1) 
  (h_odd : ∀ x ∈ selected, Odd x) 
  (h_interval : ∀ x ∈ selected, 2^(2*n.val) < x ∧ x < 2^(3*n.val)) :
  ∃ a b, a ∈ selected ∧ b ∈ selected ∧ a ≠ b ∧ ¬(a ∣ b^2) ∧ ¬(b ∣ a^2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_numbers_divisibility_l423_42324


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_ab_equals_six_l423_42343

noncomputable def f (a b x : ℝ) : ℝ := a * Real.tan (b * x)

theorem product_ab_equals_six (a b : ℝ) (h1 : f a b (π/8) = 3) (h2 : ∀ x, f a b (x + π/(2*b)) = f a b x) : a * b = 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_ab_equals_six_l423_42343


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_line_intersection_ratio_l423_42311

/-- Parabola with equation y^2 = 2px, where p > 0 -/
structure Parabola where
  p : ℝ
  h_p_pos : p > 0

/-- Line with equation √3x - y - (√3p)/2 = 0 -/
structure Line (parabola : Parabola) where
  x : ℝ
  y : ℝ
  h_line : Real.sqrt 3 * x - y - (Real.sqrt 3 * parabola.p) / 2 = 0

/-- Point on the parabola -/
structure ParabolaPoint (parabola : Parabola) where
  x : ℝ
  y : ℝ
  h_on_parabola : y^2 = 2 * parabola.p * x

/-- The focus of the parabola -/
noncomputable def focus (parabola : Parabola) : ℝ × ℝ := (parabola.p / 2, 0)

/-- Theorem stating the ratio of distances -/
theorem parabola_line_intersection_ratio (parabola : Parabola) 
    (l : Line parabola) (A B : ParabolaPoint parabola) :
    A.x > 0 → A.y > 0 → B.x > 0 → B.y < 0 →
    (A.x - parabola.p / 2)^2 + A.y^2 = 9 * ((B.x - parabola.p / 2)^2 + B.y^2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_line_intersection_ratio_l423_42311


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_overall_profit_l423_42382

/-- Calculates the overall profit from selling a grinder and a mobile phone --/
theorem overall_profit 
  (grinder_cost : ℝ) 
  (mobile_cost : ℝ) 
  (grinder_loss_percent : ℝ) 
  (mobile_profit_percent : ℝ) 
  (h1 : grinder_cost = 15000)
  (h2 : mobile_cost = 8000)
  (h3 : grinder_loss_percent = 2)
  (h4 : mobile_profit_percent = 10) :
  let grinder_sell := grinder_cost * (1 - grinder_loss_percent / 100)
  let mobile_sell := mobile_cost * (1 + mobile_profit_percent / 100)
  let total_cost := grinder_cost + mobile_cost
  let total_sell := grinder_sell + mobile_sell
  total_sell - total_cost = 500 := by
  sorry

#check overall_profit

end NUMINAMATH_CALUDE_ERRORFEEDBACK_overall_profit_l423_42382


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coin_distribution_theorem_l423_42334

/-- Represents the number of coins at each point on the circle. -/
def Configuration (n : ℕ) := Fin n → ℕ

/-- The initial configuration where each point Aᵢ has i coins. -/
def initial_config (n : ℕ) : Configuration n := λ i ↦ i.val + 1

/-- The target configuration where each point Aᵢ has n+1-i coins. -/
def target_config (n : ℕ) : Configuration n := λ i ↦ n + 1 - (i.val + 1)

/-- Represents a single move in the game. -/
inductive Move (n : ℕ)
  | take_two_give_adjacent : Fin n → Fin n → Move n

/-- Applies a move to a configuration. -/
def apply_move (n : ℕ) (c : Configuration n) (m : Move n) : Configuration n :=
  sorry

/-- Checks if two configurations are equal. -/
def config_eq (n : ℕ) (c1 c2 : Configuration n) : Prop :=
  ∀ i : Fin n, c1 i = c2 i

/-- Main theorem: It's possible to reach the target configuration
    if and only if n is divisible by 4. -/
theorem coin_distribution_theorem (n : ℕ) :
  (∃ (moves : List (Move n)), config_eq n
    (moves.foldl (apply_move n) (initial_config n))
    (target_config n)) ↔
  ∃ m : ℕ, n = 4 * m := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_coin_distribution_theorem_l423_42334


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_companion_vector_l423_42310

noncomputable def companion_vector (a b : ℝ) : ℝ × ℝ := (a, b)

noncomputable def g (x : ℝ) : ℝ := Real.sqrt 3 * Real.sin (x - Real.pi) - Real.sin (3 * Real.pi / 2 - x)

theorem g_companion_vector :
  ∃ (a b : ℝ), companion_vector a b = (-Real.sqrt 3, 1) ∧
  ∀ x, g x = a * Real.sin x + b * Real.cos x := by
  sorry

-- Additional theorems for parts (II) and (III) can be added here

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_companion_vector_l423_42310


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_covering_circle_radius_l423_42368

/-- Given a triangle ABC with largest side length a, the radius ρ of the smallest circle 
    covering the triangle satisfies: a/2 ≤ ρ ≤ a/√3 -/
theorem smallest_covering_circle_radius (A B C : EuclideanSpace ℝ (Fin 2)) 
  (a : ℝ) (h_a : a = max (dist A B) (max (dist B C) (dist C A))) :
  ∃ ρ : ℝ, (a / 2 ≤ ρ ∧ ρ ≤ a / Real.sqrt 3) ∧
    ∀ (center : EuclideanSpace ℝ (Fin 2)) (r : ℝ), 
      (dist center A ≤ r ∧ dist center B ≤ r ∧ dist center C ≤ r) → r ≥ ρ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_covering_circle_radius_l423_42368


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l423_42307

noncomputable def f (x : ℝ) := 1 / x - Real.sqrt (x - 1)

theorem domain_of_f :
  {x : ℝ | x ≠ 0 ∧ x - 1 ≥ 0} = {x : ℝ | x ≥ 1} := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l423_42307


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_students_is_43_l423_42354

-- Define the number of girls as a natural number
def girls : ℕ := sorry

-- Define the number of boys as 3 more than the number of girls
def boys : ℕ := girls + 3

-- Define the total number of students
def total_students : ℕ := girls + boys

-- Define the number of candies Mr. Marvel started with
def initial_candies : ℕ := 484

-- Define the number of leftover candies
def leftover_candies : ℕ := 4

-- Define the number of distributed candies
def distributed_candies : ℕ := initial_candies - leftover_candies

-- Theorem stating that the total number of students is 43
theorem total_students_is_43 : total_students = 43 :=
by
  sorry

#check total_students_is_43

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_students_is_43_l423_42354


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_AB_l423_42369

-- Define the curves
noncomputable def C₁ (θ : ℝ) : ℝ × ℝ := (1 + Real.cos θ, Real.sin θ)

noncomputable def C₂ (θ : ℝ) : ℝ := Real.sin θ + Real.cos θ

noncomputable def C₃ : ℝ := Real.pi / 6

-- Define the intersection points
noncomputable def A : ℝ × ℝ := C₁ C₃

noncomputable def B : ℝ := C₂ C₃

-- Theorem to prove
theorem distance_AB : Real.sqrt ((A.1 - B * Real.cos C₃)^2 + (A.2 - B * Real.sin C₃)^2) = (Real.sqrt 3 - 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_AB_l423_42369


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_line_through_point_l423_42305

/-- Given a line L1 with equation x + 2y - 1 = 0 and a point P (1, 3),
    prove that the line L2 with equation 2x - y + 1 = 0 passes through P
    and is perpendicular to L1. -/
theorem perpendicular_line_through_point (x y : ℝ) :
  let L1 : ℝ → ℝ → Prop := λ x y ↦ x + 2 * y - 1 = 0
  let L2 : ℝ → ℝ → Prop := λ x y ↦ 2 * x - y + 1 = 0
  let P : ℝ × ℝ := (1, 3)
  (L2 P.1 P.2) ∧ 
  (∀ (x1 y1 x2 y2 : ℝ), L1 x1 y1 → L1 x2 y2 → x1 ≠ x2 →
    (y2 - y1) * ((y2 - 1) / 2 - (y1 - 1) / 2) = -(x2 - x1) * (y2 - y1)) :=
by sorry

/-- Helper function to choose an x-coordinate for a given y on L2 -/
noncomputable def L2_choose_x (y : ℝ) : ℝ := (y - 1) / 2

end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_line_through_point_l423_42305


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_line_l423_42330

/-- Given a line √k x + 4y = 10 that forms a triangle with the x-axis and y-axis,
    where k is a positive real number and t is the area of the triangle,
    prove that k = (25 / (2t))² -/
theorem triangle_area_line (k t : ℝ) (h1 : k > 0) (h2 : t > 0) :
  (∃ (x y : ℝ), Real.sqrt k * x + 4 * y = 10 ∧ 
   x ≥ 0 ∧ y ≥ 0 ∧
   t = (1/2) * x * y) →
  k = (25 / (2 * t))^ 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_area_line_l423_42330


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_sin_equation_range_l423_42351

theorem cos_sin_equation_range (m : ℝ) : 
  (∃ x : ℝ, Real.cos x * Real.cos x + Real.sin x + m = 0) → -5/4 ≤ m ∧ m ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_sin_equation_range_l423_42351


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_glasses_displayed_l423_42348

/-- Represents the number of glasses in Damien's collection --/
def glass_collection (tall wide narrow : ℕ) : ℕ := tall + wide + narrow

/-- Capacity of the tall cupboard --/
def tall_capacity : ℕ := 20

/-- Capacity of the wide cupboard --/
def wide_capacity : ℕ := 2 * tall_capacity

/-- Original capacity of the narrow cupboard --/
def narrow_original_capacity : ℕ := 15

/-- Number of shelves in the narrow cupboard --/
def narrow_shelves : ℕ := 3

/-- Number of functional shelves in the narrow cupboard after one breaks --/
def narrow_functional_shelves : ℕ := narrow_shelves - 1

/-- The theorem stating the total number of glasses displayed --/
theorem total_glasses_displayed :
  glass_collection tall_capacity wide_capacity 
    (narrow_original_capacity / narrow_shelves * narrow_functional_shelves) = 70 := by
  -- Expand the definitions
  unfold glass_collection
  unfold tall_capacity
  unfold wide_capacity
  unfold narrow_original_capacity
  unfold narrow_shelves
  unfold narrow_functional_shelves
  -- Perform the calculations
  norm_num
  -- The proof is complete
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_glasses_displayed_l423_42348


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_x_axis_is_2sqrt3_div_3_l423_42329

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 - y^2/2 = 1

-- Define the foci
def foci (F1 F2 : ℝ × ℝ) : Prop := 
  let c := Real.sqrt 3
  F1 = (c, 0) ∧ F2 = (-c, 0)

-- Define a point on the hyperbola
def point_on_hyperbola (M : ℝ × ℝ) : Prop :=
  hyperbola M.1 M.2

-- Define the perpendicularity condition
def perpendicular_vectors (M F1 F2 : ℝ × ℝ) : Prop :=
  let MF1 := (F1.1 - M.1, F1.2 - M.2)
  let MF2 := (F2.1 - M.1, F2.2 - M.2)
  MF1.1 * MF2.1 + MF1.2 * MF2.2 = 0

-- Define the distance to x-axis
noncomputable def distance_to_x_axis (M : ℝ × ℝ) : ℝ := abs M.2

-- Main theorem
theorem distance_to_x_axis_is_2sqrt3_div_3 
  (M : ℝ × ℝ) (F1 F2 : ℝ × ℝ) :
  foci F1 F2 →
  point_on_hyperbola M →
  perpendicular_vectors M F1 F2 →
  distance_to_x_axis M = 2 * Real.sqrt 3 / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_x_axis_is_2sqrt3_div_3_l423_42329


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_origin_distance_under_dilation_l423_42362

/-- A dilation in the plane -/
structure MyDilation where
  -- Center of dilation
  center : ℝ × ℝ
  -- Scale factor
  scale : ℝ

/-- A circle in the plane -/
structure MyCircle where
  -- Center of the circle
  center : ℝ × ℝ
  -- Radius of the circle
  radius : ℝ

def origin : ℝ × ℝ := (0, 0)

/-- The distance between two points in the plane -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

/-- Theorem: Distance moved by origin under specific dilation -/
theorem origin_distance_under_dilation 
  (c1 c2 : MyCircle) 
  (h1 : c1.center = (3, 3) ∧ c1.radius = 3)
  (h2 : c2.center = (8, 9) ∧ c2.radius = 5)
  (d : MyDilation)
  (h3 : d.scale * c1.radius = c2.radius)
  (h4 : d.scale * distance d.center c1.center = distance d.center c2.center) :
  distance origin (d.scale • origin - d.center + d.center) = 3.75 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_origin_distance_under_dilation_l423_42362


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_wealth_ratio_X_to_Y_l423_42340

/-- Represents the wealth distribution in a country -/
structure Country where
  population : ℝ
  wealth : ℝ
  wealthDistribution : ℝ → ℝ

/-- The world's total population -/
def P : ℝ := 1

/-- The world's total wealth -/
def W : ℝ := 1

/-- Country X -/
noncomputable def X : Country := {
  population := 0.4 * P
  wealth := 0.5 * W
  wealthDistribution := λ _ => 0.5 * W / (0.4 * P)
}

/-- Country Y -/
noncomputable def Y : Country := {
  population := 0.2 * P
  wealth := 0.5 * W
  wealthDistribution := λ p =>
    if p ≤ 0.1 then (0.9 * 0.5 * W) / (0.1 * 0.2 * P)
    else (0.1 * 0.5 * W) / (0.9 * 0.2 * P)
}

/-- Average wealth per citizen in a country -/
noncomputable def averageWealth (c : Country) : ℝ :=
  c.wealth / c.population

/-- Theorem stating the ratio of average wealth between X and Y -/
theorem wealth_ratio_X_to_Y :
  averageWealth X / averageWealth Y = 10 / 19 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_wealth_ratio_X_to_Y_l423_42340


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_property_l423_42323

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the focus of the parabola
def focus : ℝ × ℝ := (1, 0)

-- Define points A and B on the parabola
def pointOnParabola (p : ℝ × ℝ) : Prop :=
  parabola p.1 p.2

-- Define that A and B are on opposite sides of x-axis
def oppositeSides (A B : ℝ × ℝ) : Prop :=
  A.2 * B.2 < 0

-- Define the dot product condition
def dotProductCondition (A B : ℝ × ℝ) : Prop :=
  A.1 * B.1 + A.2 * B.2 = -4

-- Define the fixed point T
def T : ℝ × ℝ := (2, 0)

-- Define that a line passes through a point
def passesThrough (A B p : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, p = (A.1 + t * (B.1 - A.1), A.2 + t * (B.2 - A.2))

-- Define the perpendicular line through T
def perpendicularLine (A B M N : ℝ × ℝ) : Prop :=
  (M.1 - T.1) * (B.1 - A.1) + (M.2 - T.2) * (B.2 - A.2) = 0 ∧
  (N.1 - T.1) * (B.1 - A.1) + (N.2 - T.2) * (B.2 - A.2) = 0 ∧
  pointOnParabola M ∧ pointOnParabola N

-- Define the area of quadrilateral AMBN
noncomputable def areaAMBN (A B M N : ℝ × ℝ) : ℝ :=
  abs ((A.1 - M.1) * (B.2 - M.2) - (A.2 - M.2) * (B.1 - M.1)) / 2 +
  abs ((B.1 - N.1) * (A.2 - N.2) - (B.2 - N.2) * (A.1 - N.1)) / 2

-- The main theorem
theorem parabola_property :
  ∀ A B : ℝ × ℝ,
  pointOnParabola A ∧ pointOnParabola B ∧
  oppositeSides A B ∧
  dotProductCondition A B →
  (passesThrough A B T ∧
   ∀ M N : ℝ × ℝ,
   perpendicularLine A B M N →
   areaAMBN A B M N ≥ 48 ∧
   ∃ M' N' : ℝ × ℝ, perpendicularLine A B M' N' ∧ areaAMBN A B M' N' = 48) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_property_l423_42323


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_base_side_length_l423_42309

/-- A structure representing a right pyramid with a square base -/
structure RightPyramid where
  base_side_length : ℝ
  slant_height : ℝ

theorem pyramid_base_side_length 
  (pyramid : RightPyramid) 
  (lateral_face_area : ℝ) : 
  lateral_face_area = 144 → 
  pyramid.slant_height = 24 → 
  pyramid.base_side_length = 12 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_base_side_length_l423_42309


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_square_plot_area_l423_42358

-- Define the price per foot of the fence
noncomputable def price_per_foot : ℚ := 56

-- Define the total cost of the fence
noncomputable def total_cost : ℚ := 3808

-- Define the function to calculate the area of a square given its perimeter
noncomputable def square_area_from_perimeter (perimeter : ℚ) : ℚ :=
  (perimeter / 4) ^ 2

-- Theorem statement
theorem square_plot_area :
  square_area_from_perimeter (total_cost / price_per_foot) = 289 := by
  -- Convert rational numbers to real numbers for calculation
  have h1 : (total_cost / price_per_foot : ℝ) = 68 := by sorry
  have h2 : (289 : ℚ) = 289 := by rfl
  
  -- Prove the equality
  calc square_area_from_perimeter (total_cost / price_per_foot)
    = (68 / 4) ^ 2 := by sorry
  _ = 17 ^ 2 := by sorry
  _ = 289 := by sorry
  _ = 289 := h2.symm


end NUMINAMATH_CALUDE_ERRORFEEDBACK_square_plot_area_l423_42358


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_n_for_property_P_2023_l423_42302

/-- A set has property P(m) if for any m different binary subsets,
    there exists a set B satisfying certain conditions. -/
def has_property_P (A : Finset ℕ) (m : ℕ) : Prop :=
  ∀ (binary_subsets : Finset (Finset ℕ)),
    binary_subsets.card = m →
    (∀ S ∈ binary_subsets, S ⊆ A ∧ S.card = 2) →
    ∃ B : Finset ℕ,
      B ⊆ A ∧
      B.card = m ∧
      ∀ S ∈ binary_subsets, (B ∩ S).card ≤ 1

/-- The minimum value of n for which the set A = {1, 2, ..., n}
    has property P(2023) is 4045. -/
theorem min_n_for_property_P_2023 :
  has_property_P (Finset.range 4045) 2023 ∧
  ∀ n < 4045, ¬(has_property_P (Finset.range n) 2023) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_n_for_property_P_2023_l423_42302


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_maximum_value_l423_42319

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := x + 1/x - 2

-- State the theorem
theorem f_maximum_value :
  (∀ x < 0, f x ≤ -4) ∧ (∃ x < 0, f x = -4) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_maximum_value_l423_42319


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_wire_volume_approx_l423_42320

/-- The volume of a cylindrical wire in cubic centimeters -/
noncomputable def wire_volume (diameter_mm : ℝ) (length_m : ℝ) : ℝ :=
  let radius_cm := diameter_mm / 20
  let length_cm := length_m * 100
  Real.pi * (radius_cm ^ 2) * length_cm

/-- Theorem stating that the volume of the wire with given dimensions is approximately 66.048 cubic centimeters -/
theorem wire_volume_approx :
  ∃ ε > 0, |wire_volume 1 84.03380995252074 - 66.048| < ε :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_wire_volume_approx_l423_42320


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_circle_intersection_l423_42350

/-- Hyperbola C with given parameters -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_a_pos : a > 0
  h_b_pos : b > 0

/-- Circle with center (2, 3) and radius 1 -/
def my_circle (x y : ℝ) : Prop :=
  (x - 2)^2 + (y - 3)^2 = 1

/-- Asymptotic line to the hyperbola -/
def asymptotic_line (x y : ℝ) : Prop :=
  y = 2 * x ∨ y = -2 * x

/-- Theorem statement -/
theorem hyperbola_circle_intersection (C : Hyperbola) 
  (h_eccentricity : C.a^2 + C.b^2 = 5 * C.a^2)
  (A B : ℝ × ℝ) 
  (h_A : my_circle A.1 A.2 ∧ asymptotic_line A.1 A.2)
  (h_B : my_circle B.1 B.2 ∧ asymptotic_line B.1 B.2) :
  ‖A - B‖ = 4 * Real.sqrt 5 / 5 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_circle_intersection_l423_42350


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_proof_l423_42393

-- Define a triangle with side lengths and angles
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the theorem
theorem isosceles_triangle_proof (t : Triangle) 
  (h : t.a * Real.cos t.B = t.b * Real.cos t.A) : 
  t.a = t.b ∧ t.A = t.B := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_isosceles_triangle_proof_l423_42393


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_zero_l423_42384

/-- The sum of an arithmetic sequence with n terms -/
noncomputable def arithmeticSum (a1 : ℝ) (d : ℝ) (n : ℕ) : ℝ :=
  (n : ℝ) / 2 * (2 * a1 + (n - 1 : ℝ) * d)

/-- Theorem: For an arithmetic sequence with a1 = 35, d = -2, and sum = 0, n = 36 -/
theorem arithmetic_sequence_sum_zero (n : ℕ) :
  arithmeticSum 35 (-2) n = 0 → n = 36 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_sum_zero_l423_42384


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_irreducibility_l423_42385

theorem polynomial_irreducibility (a : ℤ) (n : ℕ) (p : ℕ) (h_prime : Nat.Prime p) (h_bound : p > |a| + 1) :
  Irreducible (Polynomial.monomial n 1 + Polynomial.monomial 1 (a : ℚ) + Polynomial.C (p : ℚ)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polynomial_irreducibility_l423_42385


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_axis_of_symmetry_squared_form_axis_of_symmetry_x_plus_two_squared_correct_answer_is_A_l423_42390

/-- The axis of symmetry for a quadratic function ax² + bx + c is at x = -b/(2a) -/
noncomputable def axis_of_symmetry (a b : ℝ) : ℝ := -b / (2 * a)

/-- A quadratic function in the form (x + h)² has its axis of symmetry at x = -h -/
theorem axis_of_symmetry_squared_form (h : ℝ) :
  axis_of_symmetry 1 (2 * h) = -h := by sorry

/-- The quadratic function y = (x+2)² has its axis of symmetry at x = -2 -/
theorem axis_of_symmetry_x_plus_two_squared :
  axis_of_symmetry 1 4 = -2 := by sorry

/-- The correct answer is option A: y = (x+2)² -/
theorem correct_answer_is_A : 
  ∃ f : ℝ → ℝ, f = (λ x => (x + 2)^2) ∧ axis_of_symmetry 1 4 = -2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_axis_of_symmetry_squared_form_axis_of_symmetry_x_plus_two_squared_correct_answer_is_A_l423_42390


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_alpha_value_l423_42326

theorem cos_alpha_value (α : ℝ) (h1 : α ∈ Set.Ioo (π / 2) π) (h2 : Real.sin (π - α) = 4 / 5) : 
  Real.cos α = -3 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_alpha_value_l423_42326


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_max_ratio_sum_l423_42387

/-- In a triangle ABC, given that the altitude on side BC is half the length of side a,
    prove that the expression c/b + b/c reaches its maximum value when angle A is π/4. -/
theorem triangle_max_ratio_sum (a b c : ℝ) (A : ℝ) :
  (∃ h : ℝ, h = a / 2 ∧ h * b = a * a / 2) →  -- altitude on BC is a/2
  (∀ B C : ℝ, A + B + C = π) →  -- sum of angles in a triangle
  (a^2 = b^2 + c^2 - 2*b*c*Real.cos A) →  -- cosine theorem
  (∀ x : ℝ, c/b + b/c ≤ x) →  -- c/b + b/c has a maximum value
  A = π/4 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_max_ratio_sum_l423_42387


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_pi_half_plus_A_l423_42345

theorem sin_pi_half_plus_A (A : ℝ) : 
  Real.cos (π + A) = -1/2 → Real.sin (π/2 + A) = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_pi_half_plus_A_l423_42345


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_in_interval_l423_42374

noncomputable def f (x : ℝ) := Real.log x - 2 / x

theorem zero_in_interval :
  ∃ c ∈ Set.Ioo 2 3, f c = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_in_interval_l423_42374


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_calculation_l423_42336

/-- Calculates the length of a train given its speed, the speed of a person walking in the same direction, and the time it takes for the train to cross the person. -/
noncomputable def trainLength (trainSpeed manSpeed : ℝ) (crossingTime : ℝ) : ℝ :=
  let relativeSpeed := trainSpeed - manSpeed
  let relativeSpeedMS := relativeSpeed * (5/18)
  relativeSpeedMS * crossingTime

/-- Theorem stating that given the specified conditions, the length of the train is approximately 699.944 meters. -/
theorem train_length_calculation (trainSpeed manSpeed crossingTime : ℝ)
  (h1 : trainSpeed = 63)
  (h2 : manSpeed = 3)
  (h3 : crossingTime = 41.9966402687785) :
  ∃ ε > 0, |trainLength trainSpeed manSpeed crossingTime - 699.944| < ε :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_length_calculation_l423_42336


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_on_inverse_proportion_graph_l423_42367

noncomputable def inverse_proportion (x : ℝ) : ℝ := 6 / x

def point : ℝ × ℝ := (2, 3)

theorem point_on_inverse_proportion_graph :
  inverse_proportion point.1 = point.2 := by
  simp [inverse_proportion, point]
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_on_inverse_proportion_graph_l423_42367


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sum_abc_l423_42391

/-- Representation of an n-digit number with all digits equal to d -/
def digitRepeat (n : ℕ) (d : ℕ) : ℕ :=
  d * (10^n - 1) / 9

/-- The condition that must be satisfied for at least two values of n -/
def satisfiesCondition (a b c : ℕ) (n : ℕ) : Prop :=
  digitRepeat (2 * n) c - digitRepeat n b = 2 * (digitRepeat n a)^2

/-- There exist at least two positive integers n that satisfy the condition -/
def existsTwoN (a b c : ℕ) : Prop :=
  ∃ n m : ℕ, n ≠ m ∧ n > 0 ∧ m > 0 ∧ satisfiesCondition a b c n ∧ satisfiesCondition a b c m

/-- a, b, and c are nonzero digits -/
def validDigits (a b c : ℕ) : Prop :=
  0 < a ∧ a < 10 ∧ 0 < b ∧ b < 10 ∧ 0 < c ∧ c < 10

theorem max_sum_abc :
  ∀ a b c : ℕ, validDigits a b c → existsTwoN a b c → a + b + c ≤ 18 :=
by
  sorry

#check max_sum_abc

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sum_abc_l423_42391


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_special_roots_l423_42370

/-- Given a quadratic function f(x) = x^2 + px + q where f(0) and f(1) are its roots, 
    prove that f(6) = 31. -/
theorem quadratic_special_roots (p q : ℝ) : 
  let f : ℝ → ℝ := λ x ↦ x^2 + p*x + q
  (f 0 = 0 ∨ f 1 = 0) ∧ (f 0 * f 1 = 0) → f 6 = 31 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_special_roots_l423_42370
