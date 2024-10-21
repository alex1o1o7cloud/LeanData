import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_select_four_from_ten_for_three_tasks_l460_46011

/-- The number of ways to select and assign people to tasks -/
def select_and_assign (total : ℕ) (selected : ℕ) (task_a : ℕ) : ℕ :=
  Nat.choose total selected * Nat.choose selected task_a * Nat.factorial (selected - task_a)

/-- Theorem stating the number of ways to select 4 people from 10 and assign them to 3 tasks -/
theorem select_four_from_ten_for_three_tasks :
  select_and_assign 10 4 2 = 2520 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_select_four_from_ten_for_three_tasks_l460_46011


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l460_46057

noncomputable def f (x : ℝ) := -Real.cos x + Real.cos (Real.pi / 2 - x)

theorem f_properties :
  (∃ (x : ℝ), x ∈ Set.Icc 0 Real.pi ∧ f x = Real.sqrt 2 ∧ x = 3 * Real.pi / 4) ∧
  (∃ (x : ℝ), x ∈ Set.Icc 0 Real.pi ∧ f x = -Real.sqrt 2 ∧ x = 0) ∧
  (∀ (x : ℝ), x ∈ Set.Ioo 0 (Real.pi / 6) → Real.sin (2 * x) = 1 / 3 → f x = -Real.sqrt 6 / 3) ∧
  (∀ (x : ℝ), x ∈ Set.Icc 0 Real.pi → f x ≤ Real.sqrt 2) ∧
  (∀ (x : ℝ), x ∈ Set.Icc 0 Real.pi → f x ≥ -Real.sqrt 2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l460_46057


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_plus_y_not_two_pi_thirds_l460_46052

theorem x_plus_y_not_two_pi_thirds 
  (x y : ℝ) 
  (hx : x ∈ Set.Ioo 0 (π/2)) 
  (hy : y ∈ Set.Ioo 0 (π/2)) 
  (h : Real.sin (2*x) = 6 * Real.tan (x-y) * Real.cos (2*x)) : 
  x + y ≠ 2*π/3 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_plus_y_not_two_pi_thirds_l460_46052


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_equation_unique_k_exists_l460_46091

noncomputable section

-- Define the fixed points and the moving point
def F₁ : ℝ × ℝ := (-Real.sqrt 2, 0)
def F₂ : ℝ × ℝ := (Real.sqrt 2, 0)
def P : ℝ × ℝ → ℝ := λ p => Real.sqrt ((p.1 - F₁.1)^2 + (p.2 - F₁.2)^2) + 
                             Real.sqrt ((p.1 - F₂.1)^2 + (p.2 - F₂.2)^2)

end noncomputable section

-- Define the trajectory equation
def C (x y : ℝ) : Prop := x^2/3 + y^2 = 1

-- Define the line equation
def L (k : ℝ) (x y : ℝ) : Prop := y = k * x + 2

-- Define the fixed point E
def E : ℝ × ℝ := (-1, 0)

-- Theorem statements
theorem trajectory_equation : 
  ∀ x y : ℝ, P (x, y) = 2 * Real.sqrt 3 ↔ C x y := by sorry

theorem unique_k_exists : 
  ∃! k : ℝ, k ≠ 0 ∧ 
  (∃ A B : ℝ × ℝ, C A.1 A.2 ∧ C B.1 B.2 ∧ L k A.1 A.2 ∧ L k B.1 B.2 ∧ 
   (A.1 - E.1) * (B.1 - E.1) + (A.2 - E.2) * (B.2 - E.2) = 0) ∧
  k = 7/6 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_equation_unique_k_exists_l460_46091


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_journey_average_efficiency_l460_46015

noncomputable def round_trip_distance : ℝ := 300
noncomputable def sedan_efficiency : ℝ := 25
noncomputable def minivan_normal_efficiency : ℝ := 20
noncomputable def minivan_efficiency_decrease : ℝ := 5

noncomputable def average_fuel_efficiency : ℝ :=
  round_trip_distance / ((round_trip_distance / 2 / sedan_efficiency) + 
  (round_trip_distance / 2 / (minivan_normal_efficiency - minivan_efficiency_decrease)))

theorem journey_average_efficiency :
  average_fuel_efficiency = 18.75 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_journey_average_efficiency_l460_46015


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_last_locker_opened_l460_46010

/-- Represents the state of a locker (open or closed) -/
inductive LockerState
  | Open
  | Closed

/-- Represents the pattern of opening lockers -/
def openingPattern (n : Nat) : List Nat :=
  sorry

/-- The number of lockers in the hall -/
def numLockers : Nat := 1024

/-- Theorem stating that the last locker opened is number 342 -/
theorem last_locker_opened (lockers : Fin numLockers → LockerState) :
  (∀ i : Fin numLockers, lockers i = LockerState.Closed) →
  (∀ i : Fin numLockers, i.val ∈ openingPattern numLockers → lockers i = LockerState.Open) →
  (∃! i : Fin numLockers, lockers i = LockerState.Closed ∧ i.val = 342) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_last_locker_opened_l460_46010


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_rectangle_with_area_540_l460_46024

/-- Represents a rectangle with width and height --/
structure Rectangle where
  width : ℝ
  height : ℝ

/-- Calculates the area of a rectangle --/
def Rectangle.area (r : Rectangle) : ℝ := r.width * r.height

/-- Calculates the perimeter of a rectangle --/
def Rectangle.perimeter (r : Rectangle) : ℝ := 2 * (r.width + r.height)

/-- A square with side length 42 divided into four non-overlapping rectangles with equal perimeters --/
structure DividedSquare where
  side_length : ℝ
  rectangles : Fin 4 → Rectangle
  h_side_length : side_length = 42
  h_non_overlapping : ∀ i j, i ≠ j → (rectangles i).area ≠ (rectangles j).area
  h_equal_perimeters : ∀ i j, (rectangles i).perimeter = (rectangles j).perimeter

/-- One of the rectangles in the divided square has an area of 540 --/
theorem exists_rectangle_with_area_540 (s : DividedSquare) :
  ∃ i, (s.rectangles i).area = 540 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_rectangle_with_area_540_l460_46024


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_sum_max_value_l460_46078

theorem cos_sum_max_value :
  ∃ (max_value : ℝ), (∀ (x y : ℝ), Real.cos x - Real.cos y = 1/4 → Real.cos (x + y) ≤ max_value) ∧ max_value = 31/32 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_sum_max_value_l460_46078


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_school_student_count_l460_46023

/-- The number of students in the circle -/
def students_in_circle : ℕ := sorry

/-- The position of Marcos in Jonas's count -/
def marcos_jonas_count : ℕ := 37

/-- The position of Marcos in Amanda's count -/
def marcos_amanda_count : ℕ := 15

/-- The position of Nair in Jonas's count -/
def nair_jonas_count : ℕ := 3

/-- The position of Nair in Amanda's count -/
def nair_amanda_count : ℕ := 201

/-- The number of students not in the circle (Jonas and Amanda) -/
def students_outside_circle : ℕ := 2

theorem school_student_count : 
  students_in_circle + students_outside_circle = 222 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_school_student_count_l460_46023


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_triples_l460_46002

theorem existence_of_triples : ∃ S : Finset (ℕ × ℕ × ℕ), 
  (∀ (a b c : ℕ), (a, b, c) ∈ S → a^15 + b^15 = c^16) ∧
  S.card > 1000 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_triples_l460_46002


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_even_function_monotone_decreasing_theorem_l460_46083

def is_even_function (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

def monotone_decreasing_on (f : ℝ → ℝ) (s : Set ℝ) : Prop :=
  ∀ {x y}, x ∈ s → y ∈ s → x ≤ y → f y ≤ f x

noncomputable def log_base (b : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log b

theorem even_function_monotone_decreasing_theorem
  (f : ℝ → ℝ)
  (h_even : is_even_function f)
  (h_monotone : monotone_decreasing_on f (Set.Ici 0))
  (h_half : f (1/2) = 0) :
  {x : ℝ | f (log_base (1/4) x) < 0} = Set.union (Set.Ioo 0 (1/2)) (Set.Ioi 2) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_even_function_monotone_decreasing_theorem_l460_46083


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_birth_day_theorem_l460_46072

/-- The number of days in a week -/
def daysInWeek : Nat := 7

/-- The number of years between the birth and the anniversary -/
def anniversaryYears : Nat := 150

/-- The day of the week of the anniversary, where 0 is Sunday, 1 is Monday, etc. -/
def anniversaryDay : Fin daysInWeek := ⟨2, by decide⟩  -- Tuesday

/-- The number of leap years in the 150-year period -/
def leapYears : Nat := 36

/-- The number of regular years in the 150-year period -/
def regularYears : Nat := anniversaryYears - leapYears

theorem birth_day_theorem :
  (anniversaryDay - (regularYears + 2 * leapYears : Nat)) % daysInWeek = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_birth_day_theorem_l460_46072


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_phi_pi_half_sufficient_not_necessary_l460_46066

-- Define the function f as noncomputable due to dependency on Real.cos
noncomputable def f (φ : ℝ) (x : ℝ) : ℝ := Real.cos (2 * x + φ)

-- Define what it means for a function to be odd
def is_odd (g : ℝ → ℝ) : Prop := ∀ x, g (-x) = -g x

-- State the theorem
theorem phi_pi_half_sufficient_not_necessary :
  (∃ φ : ℝ, φ ≠ π/2 ∧ is_odd (f φ)) ∧
  (∀ φ : ℝ, φ = π/2 → is_odd (f φ)) := by
  sorry

-- You can add more lemmas or theorems here if needed

end NUMINAMATH_CALUDE_ERRORFEEDBACK_phi_pi_half_sufficient_not_necessary_l460_46066


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_area_l460_46047

noncomputable def f (x : ℝ) := 5 + Real.sqrt (4 - x)

noncomputable def tangent_line (x₀ : ℝ) (x : ℝ) : ℝ :=
  -1 / (2 * Real.sqrt (4 - x₀)) * (x - x₀) + f x₀

noncomputable def area (x₀ : ℝ) : ℝ :=
  14 * ((tangent_line x₀ (-26) + tangent_line x₀ 2) / 2)

theorem smallest_area :
  ∃ (x₀ : ℝ), -26 ≤ x₀ ∧ x₀ ≤ 2 ∧
  (∀ (y : ℝ), -26 ≤ y ∧ y ≤ 2 → area x₀ ≤ area y) ∧
  area x₀ = 504 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_area_l460_46047


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_xy_length_l460_46058

/-- Given a right triangle XYZ with medians and area satisfying certain conditions,
    prove that the length of XY is 2√14 -/
theorem right_triangle_xy_length 
  (X Y Z : ℝ × ℝ) -- Points in 2D plane
  (right_angle : (Y.1 - X.1) * (Z.1 - X.1) + (Y.2 - X.2) * (Z.2 - X.2) = 0) -- Right angle at X
  (median_x_length : ((Z.1 - X.1)^2 + (Z.2 - X.2)^2 + 
                     ((Y.1 - Z.1)/2)^2 + ((Y.2 - Z.2)/2)^2) = 25)
  (median_y_length : ((Z.1 - Y.1)^2 + (Z.2 - Y.2)^2 + 
                     ((X.1 - Z.1)/2)^2 + ((X.2 - Z.2)/2)^2) = 45)
  (triangle_area : (1/2) * abs ((Y.1 - X.1) * (Z.2 - X.2) - (Z.1 - X.1) * (Y.2 - X.2)) = 30)
  : ((Y.1 - X.1)^2 + (Y.2 - X.2)^2) = 56 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_right_triangle_xy_length_l460_46058


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identity_l460_46062

theorem trigonometric_identity (α : ℝ) :
  2 * (Real.sin α + 1 / Real.sin α)^2 + 2 * (Real.cos α + 1 / Real.cos α)^2 =
  14 + 2 * Real.tan α^2 + 2 * (1 / Real.tan α)^2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trigonometric_identity_l460_46062


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_square_with_five_lattice_points_l460_46042

/-- A square on a coordinate plane --/
structure Square where
  center : ℝ × ℝ
  side_length : ℝ
  angle : ℝ

/-- A lattice point is a point with integer coordinates --/
def is_lattice_point (p : ℝ × ℝ) : Prop :=
  Int.floor p.1 = p.1 ∧ Int.floor p.2 = p.2

/-- Count the number of lattice points inside a square --/
noncomputable def count_interior_lattice_points (s : Square) : ℕ :=
  sorry

/-- The theorem stating the approximate area of the largest square with exactly 5 interior lattice points --/
theorem largest_square_with_five_lattice_points :
  ∃ (s : Square),
    count_interior_lattice_points s = 5 ∧
    ∀ (s' : Square),
      count_interior_lattice_points s' = 5 →
      s'.side_length^2 ≤ s.side_length^2 ∧
      |s.side_length^2 - 10.47| < 0.01 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_square_with_five_lattice_points_l460_46042


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_multiples_of_15_between_20_and_200_l460_46054

theorem multiples_of_15_between_20_and_200 : 
  Finset.card (Finset.filter (λ n => 15 ∣ n ∧ 20 < n ∧ n ≤ 200) (Finset.range 201)) = 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_multiples_of_15_between_20_and_200_l460_46054


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_congruence_l460_46084

def sequence_a : ℕ → ℕ
  | 0 => 2  -- Adding this case to cover Nat.zero
  | 1 => 2
  | n + 1 => 2^(sequence_a n)

theorem sequence_congruence (n : ℕ) (h : n ≥ 2) :
  sequence_a n ≡ sequence_a (n - 1) [MOD n] := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_congruence_l460_46084


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_properties_l460_46093

-- Define the hyperbola E
noncomputable def hyperbola (b : ℝ) (x y : ℝ) : Prop := x^2 / 2 - y^2 / b^2 = 1

-- Define the line l
def line (m : ℝ) (x y : ℝ) : Prop := y = x + m

-- Define the eccentricity of a hyperbola
noncomputable def eccentricity (a b : ℝ) : ℝ := Real.sqrt (1 + b^2 / a^2)

-- Define the condition for l being parallel to an asymptote of E
noncomputable def parallel_to_asymptote (b : ℝ) : Prop := b / Real.sqrt 2 = 1

-- Define the condition for P and Q on l intersecting E
def intersection_condition (c : ℝ) (y₁ y₂ : ℝ) : Prop :=
  y₁ + y₂ = -2 * c * 7 / 5 ∧ y₁ * y₂ = (7 * c^2 - 14) / 5

-- Define the condition FP = (1/5)FQ
def focus_condition (y₁ y₂ : ℝ) : Prop := y₁ = (1/5) * y₂

theorem hyperbola_properties (b : ℝ) (h₁ : b > 0) (h₂ : parallel_to_asymptote b) :
  eccentricity (Real.sqrt 2) b = Real.sqrt 2 ∧
  (∃ (c : ℝ), ∃ (y₁ y₂ : ℝ),
    intersection_condition c y₁ y₂ ∧
    focus_condition y₁ y₂ →
    b^2 = 7) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_properties_l460_46093


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_increasing_and_inequality_l460_46077

-- Define the function f
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := (m^2 - m - 1) * x^(-5*m - 1)

-- State the theorem
theorem power_function_increasing_and_inequality 
  (m : ℝ)
  (h1 : ∀ x > 0, Monotone (f m))
  (h2 : ∃ a b : ℝ, ∀ x > 0, f m x = a * x^b) :
  (m = -1) ∧ 
  (∀ x : ℝ, f (-1) (x - 2) > 16 ↔ x > 4 ∨ x < 0) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_function_increasing_and_inequality_l460_46077


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_equals_g_l460_46037

noncomputable def f (x : ℝ) := Real.rpow x (1/3) * Real.rpow x (1/3) * Real.rpow x (1/3)
def g (x : ℝ) := x

theorem f_equals_g : ∀ x : ℝ, f x = g x := by
  intro x
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_equals_g_l460_46037


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_weighted_average_germination_rate_is_28_25_percent_l460_46048

/-- Represents a plot in the agricultural experiment -/
structure Plot where
  seeds : ℕ
  germination_rate : ℚ

/-- Calculates the number of germinated seeds in a plot -/
def germinated_seeds (p : Plot) : ℚ := p.seeds * p.germination_rate

/-- The plots in the experiment -/
def plots : List Plot := [
  { seeds := 300, germination_rate := 25 / 100 },
  { seeds := 200, germination_rate := 40 / 100 },
  { seeds := 500, germination_rate := 30 / 100 },
  { seeds := 400, germination_rate := 35 / 100 },
  { seeds := 600, germination_rate := 20 / 100 }
]

/-- Calculates the total number of seeds planted -/
def total_seeds : ℕ := (plots.map (λ p => p.seeds)).sum

/-- Calculates the total number of germinated seeds -/
noncomputable def total_germinated : ℚ := (plots.map germinated_seeds).sum

/-- Calculates the weighted average germination rate -/
noncomputable def weighted_average_germination_rate : ℚ := total_germinated / total_seeds

/-- Theorem stating that the weighted average germination rate is 28.25% -/
theorem weighted_average_germination_rate_is_28_25_percent :
  weighted_average_germination_rate = 2825 / 10000 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_weighted_average_germination_rate_is_28_25_percent_l460_46048


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_prime_divisors_l460_46092

/-- A sequence of positive integers satisfying the given conditions -/
def SequenceA (a : ℕ+) : ℕ → ℕ+ := sorry

/-- The condition that the sequence is strictly increasing and bounded by a -/
axiom sequence_condition (a : ℕ+) (n : ℕ) : 
  (SequenceA a n).val < (SequenceA a (n + 1)).val ∧ 
  (SequenceA a (n + 1)).val ≤ (SequenceA a n).val + a.val

/-- The theorem stating that infinitely many primes divide at least one term of the sequence -/
theorem infinitely_many_prime_divisors (a : ℕ+) :
  ∀ (N : ℕ), ∃ (p : ℕ) (n : ℕ), p > N ∧ Nat.Prime p ∧ p ∣ (SequenceA a n).val := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_prime_divisors_l460_46092


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_remaining_milk_theorem_l460_46098

/-- Calculates the remaining milk after all sales and production --/
def remaining_milk (
  cows goats sheep : ℕ)
  (morning_cow morning_goat morning_sheep : ℝ)
  (evening_cow evening_goat evening_sheep : ℝ)
  (morning_spoil evening_spoil : ℝ)
  (cheese_production : ℝ)
  (ice_cream_buy : ℝ)
  (cheese_shop_buy : ℝ)
  (previous_leftover : ℝ) : ℝ :=
  let morning_total := cows * morning_cow + goats * morning_goat + sheep * morning_sheep
  let evening_total := cows * evening_cow + goats * evening_goat + sheep * evening_sheep
  let morning_after_spoil := morning_total * (1 - morning_spoil)
  let morning_after_cheese := morning_after_spoil * (1 - cheese_production)
  let morning_after_ice_cream := morning_after_cheese * (1 - ice_cream_buy)
  let evening_after_spoil := evening_total * (1 - evening_spoil)
  let evening_after_cheese_shop := evening_after_spoil * (1 - cheese_shop_buy)
  morning_after_ice_cream + evening_after_cheese_shop + previous_leftover

/-- Theorem stating that the remaining milk is approximately 44.7735 gallons --/
theorem remaining_milk_theorem :
  ∃ ε > 0, |remaining_milk 5 4 10 13 0.5 0.25 14 0.6 0.2 0.1 0.05 0.15 0.7 0.8 15 - 44.7735| < ε := by
  sorry

#eval remaining_milk 5 4 10 13 0.5 0.25 14 0.6 0.2 0.1 0.05 0.15 0.7 0.8 15

end NUMINAMATH_CALUDE_ERRORFEEDBACK_remaining_milk_theorem_l460_46098


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_arrangements_count_l460_46043

def digits : List ℕ := [4, 5, 5, 0, 2, 2]

def is_valid_arrangement (arrangement : List ℕ) : Bool :=
  arrangement.length = 6 && arrangement.head? ≠ some 0

def count_valid_arrangements : ℕ :=
  (digits.permutations.filter is_valid_arrangement).length

theorem valid_arrangements_count : count_valid_arrangements = 150 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_arrangements_count_l460_46043


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_through_point_equal_distances_l460_46097

/-- A line in 2D space represented by its equation ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- The distance between a point and a line -/
noncomputable def distance_point_to_line (p : Point) (l : Line) : ℝ :=
  abs (l.a * p.x + l.b * p.y + l.c) / Real.sqrt (l.a ^ 2 + l.b ^ 2)

/-- Check if a point lies on a line -/
def point_on_line (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

theorem line_through_point_equal_distances (P A B : Point) :
  (P.x = 1 ∧ P.y = 2 ∧ A.x = 2 ∧ A.y = 3 ∧ B.x = 0 ∧ B.y = -5) →
  ∃ l : Line,
    (point_on_line P l) ∧
    (distance_point_to_line A l = distance_point_to_line B l) ∧
    ((l.a = 4 ∧ l.b = -1 ∧ l.c = -2) ∨ (l.a = 1 ∧ l.b = 0 ∧ l.c = -1)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_through_point_equal_distances_l460_46097


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_value_implies_a_l460_46039

noncomputable def f (x a : ℝ) : ℝ := 3^x + a / (3^x + 1)

theorem minimum_value_implies_a (a : ℝ) (h1 : a > 0) :
  (∀ x : ℝ, f x a ≥ 5) ∧ (∃ x : ℝ, f x a = 5) → a = 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_minimum_value_implies_a_l460_46039


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_quadrilateral_ABEF_l460_46033

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the focus
def focus : ℝ × ℝ := (1, 0)

-- Define the line passing through the focus at 60°
noncomputable def line_through_focus (x y : ℝ) : Prop :=
  y = Real.sqrt 3 * (x - 1)

-- Define point A as the intersection of the line and the parabola
noncomputable def point_A : ℝ × ℝ := (3, 2*Real.sqrt 3)

-- Define point B as the foot of the perpendicular from A to the directrix
def point_B : ℝ × ℝ := (3, -2)

-- Define point E as the intersection of the directrix and x-axis
def point_E : ℝ × ℝ := (-1, 0)

-- State the theorem
theorem area_of_quadrilateral_ABEF :
  parabola (point_A.1) (point_A.2) →
  line_through_focus (point_A.1) (point_A.2) →
  (point_B.2 - point_E.2) / (point_B.1 - point_E.1) = 0 →
  let area := ((point_E.1 - focus.1 + point_B.1 - point_A.1) * 
               (point_A.2 - point_B.2)) / 2
  area = 6 * Real.sqrt 3 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_quadrilateral_ABEF_l460_46033


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_point_on_line_distance_l460_46056

/-- The distance from a point (x, y) to a line Ax + By + C = 0 --/
noncomputable def distance_point_to_line (x y A B C : ℝ) : ℝ :=
  abs (A * x + B * y + C) / Real.sqrt (A^2 + B^2)

/-- Theorem: If a point P(a, 0) on the x-axis has distance 6 from the line 3x - 4y + 6 = 0,
    then a = 8 or a = -12 --/
theorem point_on_line_distance (a : ℝ) :
  distance_point_to_line a 0 3 (-4) 6 = 6 → a = 8 ∨ a = -12 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_point_on_line_distance_l460_46056


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_point_locus_l460_46070

/-- Given two points A(a, 0) and B(b, 0) where 0 < a < b, and two lines passing through these points
    that intersect the parabola y^2 = x at four points lying on a common circle,
    the x-coordinate of the intersection point P of these lines is (a + b) / 2. -/
theorem intersection_point_locus (a b : ℝ) (ha : 0 < a) (hab : a < b) :
  ∃ (k : ℝ),
  let l := λ x y : ℝ ↦ y = k * (x - a)
  let m := λ x y : ℝ ↦ y = -k * (x - b)
  let parabola := λ x y : ℝ ↦ y^2 = x
  let P := λ x y : ℝ ↦ l x y ∧ m x y
  let intersections := {p : ℝ × ℝ | (l p.1 p.2 ∨ m p.1 p.2) ∧ parabola p.1 p.2}
  (∃ (c : ℝ × ℝ) (r : ℝ), ∀ p ∈ intersections, (p.1 - c.1)^2 + (p.2 - c.2)^2 = r^2) →
  ∃ (x y : ℝ), P x y ∧ x = (a + b) / 2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_point_locus_l460_46070


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_3x_equation_l460_46028

theorem tan_3x_equation (a b : ℝ) (x : ℝ) (h1 : Real.tan x = a / b) (h2 : Real.tan (3 * x) = b / (2 * a + b)) :
  ∃ k : ℝ, k = 1 / 4 ∧ x = Real.arctan k ∧ x > 0 ∧ ∀ y, y > 0 → y = Real.arctan (1 / 4) → x ≤ y := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_3x_equation_l460_46028


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_acme_vowel_soup_sequences_l460_46096

def vowel_count : Fin 5 → ℕ
  | 0 => 6  -- A
  | 1 => 6  -- E
  | 2 => 6  -- I
  | 3 => 4  -- O
  | 4 => 4  -- U

def total_vowels : ℕ := Finset.sum (Finset.univ : Finset (Fin 5)) vowel_count

theorem acme_vowel_soup_sequences :
  (total_vowels ^ 5 : ℕ) = 11881376 := by
  have h1 : total_vowels = 26 := by
    rfl
  have h2 : 26 ^ 5 = 11881376 := by
    rfl
  rw [h1, h2]

#eval total_vowels -- This will print 26
#eval total_vowels ^ 5 -- This will print 11881376

end NUMINAMATH_CALUDE_ERRORFEEDBACK_acme_vowel_soup_sequences_l460_46096


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_theorem_l460_46075

/-- Definition of the ellipse C -/
def ellipse_C (a b x y : ℝ) : Prop :=
  x^2 / a^2 + y^2 / b^2 = 1 ∧ a > b ∧ b > 0

/-- C passes through the point (√3, -√3/2) -/
def passes_through (a b : ℝ) : Prop :=
  ellipse_C a b (Real.sqrt 3) (-Real.sqrt 3 / 2)

/-- Eccentricity of C is 1/2 -/
def eccentricity (a b : ℝ) : Prop :=
  Real.sqrt (a^2 - b^2) / a = 1 / 2

/-- Definition of S (sum of areas of triangles QF₁O and PF₁R) -/
noncomputable def S (a b : ℝ) (P Q : ℝ × ℝ) : ℝ :=
  sorry  -- Definition of S would go here

/-- The main theorem -/
theorem ellipse_theorem (a b : ℝ) :
  (ellipse_C a b (Real.sqrt 3) (-Real.sqrt 3 / 2)) ∧ 
  passes_through a b ∧ 
  eccentricity a b →
  (a = 2 ∧ b = Real.sqrt 3) ∧
  (∀ P Q : ℝ × ℝ, ellipse_C a b P.1 P.2 → ellipse_C a b Q.1 Q.2 → S a b P Q ≤ 3/2) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_theorem_l460_46075


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cleaning_time_proof_l460_46085

/-- The time it takes for all three people to clean the room together -/
noncomputable def combined_cleaning_time (lisa_time kay_time ben_time : ℝ) : ℝ :=
  1 / (1 / lisa_time + 1 / kay_time + 1 / ben_time)

/-- Theorem stating that given the individual cleaning times, 
    the combined cleaning time is 48/13 hours -/
theorem cleaning_time_proof :
  combined_cleaning_time 8 12 16 = 48 / 13 := by
  -- Proof steps would go here
  sorry

/-- Approximate the combined cleaning time -/
def approx_combined_cleaning_time : ℚ :=
  48 / 13

#eval approx_combined_cleaning_time

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cleaning_time_proof_l460_46085


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_z_l460_46076

theorem smallest_positive_z (x z : ℝ) : 
  Real.sin x = 0 → 
  Real.sin (x + z) = 1/2 → 
  z > 0 → 
  (∀ w, w > 0 → Real.sin x = 0 → Real.sin (x + w) = 1/2 → z ≤ w) → 
  z = π/6 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_z_l460_46076


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distribute_items_to_boxes_l460_46095

theorem distribute_items_to_boxes (n : ℕ) (box1 box2 box3 : ℕ) : 
  n = 9 → box1 = 3 → box2 = 2 → box3 = 4 →
  (Nat.choose n box1) * (Nat.choose (n - box1) box2) * (Nat.choose (n - box1 - box2) box3) * Nat.factorial 3 = 7560 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distribute_items_to_boxes_l460_46095


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_range_on_interval_l460_46005

theorem cosine_range_on_interval :
  ∀ y ∈ Set.Icc (-1/2 : Real) 1,
    ∃ x ∈ Set.Icc (-π/6 : Real) (2*π/3),
      Real.cos x = y ∧
      ∀ x' ∈ Set.Icc (-π/6 : Real) (2*π/3),
        -1/2 ≤ Real.cos x' ∧ Real.cos x' ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_range_on_interval_l460_46005


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_is_60_degrees_l460_46071

-- Define a triangle with sides a, b, c
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  pos_a : 0 < a
  pos_b : 0 < b
  pos_c : 0 < c
  triangle_ineq : a + b > c ∧ b + c > a ∧ c + a > b

-- Define the given condition
def satisfies_condition (t : Triangle) : Prop :=
  (t.a + t.b + t.c) * (t.a + t.b - t.c) = 3 * t.a * t.b

-- Define the angle opposite to side c
noncomputable def angle_C (t : Triangle) : ℝ :=
  Real.arccos ((t.a^2 + t.b^2 - t.c^2) / (2 * t.a * t.b))

-- Theorem statement
theorem angle_is_60_degrees (t : Triangle) (h : satisfies_condition t) :
  angle_C t = π / 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_is_60_degrees_l460_46071


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_base_r_l460_46022

-- Define the base r
def r : ℕ := sorry

-- Define x in base r
def x : ℕ := r^3 + r^2 + r + 1

-- Define the digits of x^2 in base r
def a : ℕ := sorry
def b : ℕ := sorry
def c : ℕ := sorry
def d : ℕ := sorry

-- Axioms based on the problem conditions
axiom r_even : Even r
axiom r_greater_than_9 : r > 9
axiom x_squared_palindrome : x^2 = a*r^7 + b*r^6 + c*r^5 + d*r^4 + d*r^3 + c*r^2 + b*r + a
axiom sum_second_third_digit : b + c = 24

-- Theorem to prove
theorem unique_base_r : r = 26 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_base_r_l460_46022


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_segment_areas_l460_46009

noncomputable section

def Point := ℝ × ℝ

structure Circle where
  center : Point
  radius : ℝ

def onCircle (p : Point) (c : Circle) : Prop :=
  let (x, y) := p
  let (cx, cy) := c.center
  (x - cx)^2 + (y - cy)^2 = c.radius^2

def perpendicularToDiameter (p : Point) (c : Circle) : Prop :=
  let (x, y) := p
  let (cx, cy) := c.center
  (x - cx) * c.radius = 0

theorem circle_segment_areas (diameter : ℝ) (C : Point) 
  (h1 : diameter = 10)
  (c : Circle)
  (h2 : onCircle C c)
  (h3 : perpendicularToDiameter C c) : 
  let r := diameter / 2
  let circleArea := π * r^2
  let triangleArea := r^2 / 2
  let segmentAreas := circleArea - triangleArea
  segmentAreas = 25 * π - 25 / 2 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_segment_areas_l460_46009


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_l460_46088

-- Define the triangle
variable (a b c : ℝ) (A B C : ℝ)

-- Define the conditions
def condition1 (a b c A B C : ℝ) : Prop := a / c = Real.cos B + Real.sqrt 3 * Real.cos C
def condition2 (a b c A B C : ℝ) : Prop := Real.sqrt 3 * a / c = (a - Real.sqrt 3 * Real.cos A) / Real.cos C

-- Define the area function
noncomputable def area (a b c : ℝ) : ℝ := 
  Real.sqrt (1/4 * (a^2 * c^2 - ((a^2 + c^2 - b^2) / 2)^2))

-- State the theorem
theorem max_area :
  ∃ (max_area : ℝ), max_area = (9 * Real.sqrt 3) / 4 ∧ 
  ∀ (a b c A B C : ℝ), 
    condition1 a b c A B C → condition2 a b c A B C → 
    area a b c ≤ max_area := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_area_l460_46088


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_MN_l460_46059

noncomputable section

/-- Line l in polar coordinates -/
def line_l (θ : Real) : Real → Prop :=
  fun ρ ↦ ρ * (Real.sin θ + Real.cos θ) = 2

/-- Curve C in polar coordinates -/
def curve_C (θ : Real) : Real → Prop :=
  fun ρ ↦ ρ = 4 * Real.cos θ

/-- Intersection point of ray θ = π/4 with line l -/
noncomputable def point_M : Real :=
  2 / (Real.sin (Real.pi / 4) + Real.cos (Real.pi / 4))

/-- Intersection point of ray θ = π/4 with curve C -/
noncomputable def point_N : Real :=
  4 * Real.cos (Real.pi / 4)

theorem distance_MN :
  |point_N - point_M| = Real.sqrt 2 := by sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_MN_l460_46059


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mens_wages_l460_46073

/-- Represents the wage of a single person -/
structure Wage where
  amount : ℚ
  units : String

/-- Represents a group of people with their total wage -/
structure PersonGroup where
  count : ℕ
  wage : Wage

theorem mens_wages (men women boys : PersonGroup) : 
  men.count = 5 →
  men.wage = women.wage →
  women.wage = boys.wage →
  boys.count = 8 →
  (men.count : ℚ) * men.wage.amount + 
  (women.count : ℚ) * women.wage.amount + 
  (boys.count : ℚ) * boys.wage.amount = 210 →
  (men.count : ℚ) * men.wage.amount = 105 :=
by sorry

#check mens_wages

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mens_wages_l460_46073


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_repeating_decimal_subtraction_l460_46007

theorem repeating_decimal_subtraction :
  (246 : ℚ) / 999 - 135 / 999 - 9753 / 9999 = -8647897 / 9989001 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_repeating_decimal_subtraction_l460_46007


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_moles_NaOH_is_one_l460_46025

-- Define the molar mass of H2O
def molar_mass_H2O : ℝ := 18.015

-- Define the mass of H2O produced
def mass_H2O : ℝ := 18

-- Define the number of moles of HCl
def moles_HCl : ℝ := 1

-- Define the reaction equation
def reaction_equation (moles_NaOH : ℝ) : Prop := moles_NaOH = moles_HCl

-- Define the relationship between mass and moles of H2O
def mass_mole_relation (moles_H2O : ℝ) : Prop := moles_H2O * molar_mass_H2O = mass_H2O

-- Theorem: The number of moles of NaOH is 1
theorem moles_NaOH_is_one : 
  ∃ (moles_NaOH : ℝ), moles_NaOH = 1 ∧ 
    reaction_equation moles_NaOH ∧ 
    mass_mole_relation moles_NaOH := by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_moles_NaOH_is_one_l460_46025


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_sum_is_pi_over_two_l460_46063

theorem angle_sum_is_pi_over_two 
  (α β : ℝ) 
  (h1 : 0 < α ∧ α < π/2) 
  (h2 : 0 < β ∧ β < π/2) 
  (h3 : (Real.sin α)^4 / (Real.cos β)^2 + (Real.cos α)^4 / (Real.sin β)^2 = 1) : 
  α + β = π/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_angle_sum_is_pi_over_two_l460_46063


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_range_l460_46017

/-- Line l with parametric equation x = 3 + t*cos(α), y = 2 + t*sin(α) -/
def line_l (α : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | ∃ t : ℝ, p.1 = 3 + t * Real.cos α ∧ p.2 = 2 + t * Real.sin α}

/-- Circle C with equation x^2 + y^2 - 2x = 0 -/
def circle_C : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 + p.2^2 - 2*p.1 = 0}

/-- Point M(3,2) -/
def point_M : ℝ × ℝ := (3, 2)

/-- The intersection points of line l and circle C -/
def intersection_points (α : ℝ) : Set (ℝ × ℝ) :=
  line_l α ∩ circle_C

theorem intersection_range (α : ℝ) :
  point_M ∈ line_l α →
  (∃ A B : ℝ × ℝ, A ∈ intersection_points α ∧ B ∈ intersection_points α ∧ A ≠ B) →
  ∃ sum : ℝ, 2 * Real.sqrt 7 / 7 < sum ∧ sum ≤ 4 * Real.sqrt 2 / 7 ∧
    sum = 1 / dist point_M A + 1 / dist point_M B :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_range_l460_46017


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_is_equilateral_l460_46068

/-- Helper function to calculate the area of a triangle given three points -/
def area_triangle (p1 p2 p3 : ℝ × ℝ) : ℝ :=
  sorry  -- Implementation details omitted for brevity

/-- Given a triangle ABC with circumcircle O, prove it's equilateral under certain conditions -/
theorem triangle_is_equilateral (A B C : ℝ) (a b c : ℝ) (O : ℝ × ℝ) :
  -- Conditions
  (a * c * Real.sin A + 4 * Real.sin C = 4 * c * Real.sin A) →  -- Given equation
  (Real.sqrt 3 / 3 = area_triangle O (B, 0) (C, 0)) →  -- Area of triangle OBC
  (b + c = 4) →  -- Sum of two sides
  -- Conclusions
  (a = 2 ∧ b = 2 ∧ c = 2) :=  -- Triangle is equilateral
by
  sorry  -- Proof details omitted for brevity

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_is_equilateral_l460_46068


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_points_and_isosceles_triangle_l460_46026

noncomputable def line_l (x y : ℝ) : Prop := 2 * x + y + 4 = 0

noncomputable def circle_C (x y : ℝ) : Prop := x^2 + y^2 + 2*x - 4*y + 1 = 0

noncomputable def point_A : ℝ × ℝ := (-11/5, 2/5)
noncomputable def point_B : ℝ × ℝ := (-3, 2)

def on_x_axis (p : ℝ × ℝ) : Prop := p.2 = 0

noncomputable def is_isosceles_ABD (d : ℝ × ℝ) : Prop :=
  (d.1 - point_A.1)^2 + (d.2 - point_A.2)^2 = (d.1 - point_B.1)^2 + (d.2 - point_B.2)^2 ∨
  (point_A.1 - d.1)^2 + (point_A.2 - d.2)^2 = (point_A.1 - point_B.1)^2 + (point_A.2 - point_B.2)^2 ∨
  (point_B.1 - d.1)^2 + (point_B.2 - d.2)^2 = (point_A.1 - point_B.1)^2 + (point_A.2 - point_B.2)^2

theorem intersection_points_and_isosceles_triangle :
  ∃ (d : ℝ × ℝ),
    on_x_axis d ∧
    is_isosceles_ABD d ∧
    (d = (-5, 0) ∨ d = (-11/5 + 2*Real.sqrt 19/5, 0) ∨ d = (-11/5 - 2*Real.sqrt 19/5, 0)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_points_and_isosceles_triangle_l460_46026


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_digit_sum_equation_proof_l460_46086

theorem digit_sum_equation_proof : ∃ (a b : ℕ), 
  a ≠ b ∧ 
  a ≤ 9 ∧ 
  b ≤ 9 ∧ 
  15.2 + 1.52 + 0.15 * (a : ℝ) + (b : ℝ) + 0.128 = 20 ∧ 
  a + b = 5 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_digit_sum_equation_proof_l460_46086


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_four_quadrilateral_types_have_equidistant_point_l460_46035

-- Define the types of quadrilaterals
inductive QuadrilateralType
  | Square
  | RectangleNotSquare
  | RhombusNotSquare
  | ParallelogramNotRectangleOrRhombus
  | IsoscelesTrapezoidNotParallelogram
  | KiteWithPerpendicularBisector

-- Define a function to check if a quadrilateral type has an equidistant point
def hasEquidistantPoint (q : QuadrilateralType) : Bool :=
  match q with
  | QuadrilateralType.Square => true
  | QuadrilateralType.RectangleNotSquare => true
  | QuadrilateralType.RhombusNotSquare => false
  | QuadrilateralType.ParallelogramNotRectangleOrRhombus => false
  | QuadrilateralType.IsoscelesTrapezoidNotParallelogram => true
  | QuadrilateralType.KiteWithPerpendicularBisector => true

-- Theorem stating that exactly 4 out of 6 quadrilateral types have an equidistant point
theorem four_quadrilateral_types_have_equidistant_point :
  (List.filter hasEquidistantPoint [QuadrilateralType.Square, QuadrilateralType.RectangleNotSquare,
    QuadrilateralType.RhombusNotSquare, QuadrilateralType.ParallelogramNotRectangleOrRhombus,
    QuadrilateralType.IsoscelesTrapezoidNotParallelogram, QuadrilateralType.KiteWithPerpendicularBisector]).length = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_four_quadrilateral_types_have_equidistant_point_l460_46035


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_lines_count_l460_46050

-- Define the circle
noncomputable def my_circle (x y : ℝ) : Prop := x^2 + (y - 2 * Real.sqrt 2)^2 = 4

-- Define a line with equal intercepts on both axes
def line_equal_intercepts (m b : ℝ) : Prop := ∃ a : ℝ, (a ≠ 0 ∧ b = a ∧ m = -1) ∨ (m = 1 ∧ b = 0)

-- Define tangency condition
def is_tangent (m b : ℝ) : Prop :=
  ∀ x y : ℝ, my_circle x y → (y = m * x + b → (x = y ∨ x = -y ∨ y = -x + 4 * Real.sqrt 2))

-- Theorem statement
theorem tangent_lines_count :
  ∃! (lines : Finset (ℝ × ℝ)),
    lines.card = 3 ∧
    ∀ m b : ℝ, (m, b) ∈ lines ↔ (line_equal_intercepts m b ∧ is_tangent m b) :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_lines_count_l460_46050


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_cos_2x_over_cos_plus_sin_l460_46090

open Real MeasureTheory

theorem integral_cos_2x_over_cos_plus_sin :
  (∫ x in Set.Icc 0 (π/4), (cos (2*x)) / (cos x + sin x)) = Real.sqrt 2 - 1 :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_cos_2x_over_cos_plus_sin_l460_46090


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_peters_age_l460_46008

/-- Peter's current age -/
def p : ℕ := sorry

/-- Mary's current age -/
def m : ℕ := sorry

/-- The sum of Peter and Mary's current ages is 52 -/
axiom sum_of_ages : p + m = 52

/-- Mary's current age is equal to Peter's age when Mary was half as old as Peter is now -/
axiom age_relationship : m = p / 2 + (p - m)

/-- Peter's age is 30 -/
theorem peters_age : p = 30 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_peters_age_l460_46008


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_2019_equals_neg_cos_l460_46001

open Real

-- Define the sequence of functions
noncomputable def f : ℕ → (ℝ → ℝ)
  | 0 => fun _ => 0  -- Add a base case for 0
  | 1 => cos
  | n + 1 => deriv (f n)

-- State the theorem
theorem f_2019_equals_neg_cos : f 2019 = fun x => -cos x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_2019_equals_neg_cos_l460_46001


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_identity_l460_46065

theorem sin_cos_identity (t : ℝ) : 
  (Real.sin (2 * t))^3 + (Real.cos (2 * t))^3 + (1/2) * Real.sin (4 * t) = 1 ↔ 
  (∃ k : ℤ, t = π * ↑k) ∨ (∃ n : ℤ, t = (π/4) * (4 * ↑n + 1)) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_cos_identity_l460_46065


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_line_exists_l460_46004

/-- A line in the xy-plane is defined by its x-intercept and y-intercept -/
structure Line where
  x_intercept : ℝ
  y_intercept : ℝ

/-- Checks if a number is prime -/
def isPrime (n : ℕ) : Prop := Nat.Prime n

/-- Checks if a number is composite -/
def isComposite (n : ℕ) : Prop := n > 1 ∧ ¬Nat.Prime n

/-- Checks if a line passes through the point (5,4) -/
def passesThrough (l : Line) : Prop :=
  5 / l.x_intercept + 4 / l.y_intercept = 1

/-- The main theorem -/
theorem unique_line_exists :
  ∃! l : Line,
    l.x_intercept > 0 ∧
    isPrime (Int.toNat ⌈l.x_intercept⌉) ∧
    isComposite (Int.toNat ⌈l.y_intercept⌉) ∧
    passesThrough l :=
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_line_exists_l460_46004


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circumradius_of_specific_quadrilateral_l460_46060

structure Quadrilateral where
  AB : ℝ
  BC : ℝ
  CD : ℝ
  DA : ℝ
  angleBCD : ℝ

noncomputable def circumradius (q : Quadrilateral) : ℝ :=
  Real.sqrt ((q.AB * q.BC * q.CD * q.DA) / (8 * Real.sqrt (12 * 10 * 14 * 8)))

theorem circumradius_of_specific_quadrilateral :
  ∀ q : Quadrilateral,
    q.AB = 10 ∧
    q.BC = 12 ∧
    q.CD = 8 ∧
    q.DA = 14 ∧
    q.angleBCD = 90 →
    circumradius q = Real.sqrt 210 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circumradius_of_specific_quadrilateral_l460_46060


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_roots_of_unity_and_quadratic_l460_46051

def is_root_of_unity (z : ℂ) : Prop :=
  ∃ n : ℕ, n > 0 ∧ z ^ n = 1

def is_root_of_quadratic (z : ℂ) : Prop :=
  ∃ a b : ℤ, z^2 + a*z + b = 0

theorem roots_of_unity_and_quadratic :
  ∃! (S : Finset ℂ), S.card = 4 ∧ 
  (∀ z ∈ S, is_root_of_unity z ∧ is_root_of_quadratic z) ∧
  (∀ z : ℂ, is_root_of_unity z ∧ is_root_of_quadratic z → z ∈ S) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_roots_of_unity_and_quadratic_l460_46051


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l460_46030

/-- A hyperbola with given asymptotes -/
structure Hyperbola where
  -- The asymptotes are defined implicitly by these equations
  asymptote1 : ℝ → ℝ → Prop := fun x y => x + y = 0
  asymptote2 : ℝ → ℝ → Prop := fun x y => x - y = 0

/-- The eccentricity of a hyperbola -/
noncomputable def eccentricity (h : Hyperbola) : ℝ := Real.sqrt 2

/-- Theorem stating that the eccentricity of the given hyperbola is √2 -/
theorem hyperbola_eccentricity (h : Hyperbola) : eccentricity h = Real.sqrt 2 := by
  -- Unfold the definition of eccentricity
  unfold eccentricity
  -- The result follows directly from the definition
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_l460_46030


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_two_extreme_points_iff_a_in_range_l460_46036

/-- The function f(x) defined on positive real numbers --/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x * (Real.log x - 2 * a * x)

/-- The derivative of f(x) --/
noncomputable def f' (a : ℝ) (x : ℝ) : ℝ := Real.log x + 1 - 4 * a * x

/-- The condition for f(x) to have two extreme points --/
def has_two_extreme_points (a : ℝ) : Prop :=
  ∃ x₁ x₂ : ℝ, 0 < x₁ ∧ 0 < x₂ ∧ x₁ ≠ x₂ ∧ f' a x₁ = 0 ∧ f' a x₂ = 0

/-- The theorem stating the range of a for which f(x) has two extreme points --/
theorem f_two_extreme_points_iff_a_in_range :
  ∀ a : ℝ, has_two_extreme_points a ↔ 0 < a ∧ a < 1/4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_two_extreme_points_iff_a_in_range_l460_46036


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_to_q_l460_46074

/-- The curve in polar coordinates on which point P moves -/
noncomputable def curve (ρ θ : ℝ) : Prop := ρ^2 * Real.cos θ - 2*ρ = 0

/-- The x-coordinate of the point Q -/
noncomputable def q_x : ℝ := 1/2

/-- The y-coordinate of the point Q -/
noncomputable def q_y : ℝ := Real.sqrt 3 / 2

/-- The minimum distance from a point on the curve to Q -/
noncomputable def min_distance : ℝ := 3/2

theorem min_distance_to_q :
  ∀ (x y : ℝ), (∃ (ρ θ : ℝ), curve ρ θ ∧ x = ρ * Real.cos θ ∧ y = ρ * Real.sin θ) →
  Real.sqrt ((x - q_x)^2 + (y - q_y)^2) ≥ min_distance := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_to_q_l460_46074


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_of_centers_l460_46014

-- Define the given circle k
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the given point P
def Point : Type := ℝ × ℝ

-- Define the locus of centers
inductive Locus
  | Hyperbola : Point → Point → ℝ → Locus
  | Line : Point → Point → Locus
  | Ellipse : Point → Point → ℝ → Locus

-- Define the position of P relative to k
inductive PointPosition
  | Outside
  | On
  | Inside

-- Function to determine the locus based on the point's position
def determineLocus (k : Circle) (P : Point) (pos : PointPosition) : Locus :=
  match pos with
  | PointPosition.Outside => Locus.Hyperbola k.center P k.radius
  | PointPosition.On => Locus.Line k.center P
  | PointPosition.Inside => Locus.Ellipse k.center P k.radius

-- Helper functions (not implemented, just for context)
noncomputable def set_of_points_on_locus : Locus → Set Point := sorry
noncomputable def circle_touches : Circle → Circle → Prop := sorry
noncomputable def points_on_circle : Circle → Set Point := sorry
noncomputable def distance : Point → Point → ℝ := sorry

-- Theorem statement
theorem locus_of_centers 
  (k : Circle) (P : Point) (pos : PointPosition) :
  ∃ (X : Point), 
    (X ∈ set_of_points_on_locus (determineLocus k P pos)) ∧
    (circle_touches k (Circle.mk X (distance X P))) ∧
    (P ∈ points_on_circle (Circle.mk X (distance X P))) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_of_centers_l460_46014


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_passing_pole_time_l460_46041

/-- The time (in seconds) it takes for a train to pass an electric pole -/
noncomputable def train_passing_time (train_length : ℝ) (train_speed_kmph : ℝ) : ℝ :=
  train_length / (train_speed_kmph * 1000 / 3600)

/-- Theorem: A train of length 120 meters traveling at 60 kmph takes approximately 7.20 seconds to pass an electric pole -/
theorem train_passing_pole_time :
  ∃ ε > 0, |train_passing_time 120 60 - 7.20| < ε :=
by
  -- The proof is omitted for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_passing_pole_time_l460_46041


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_double_angle_second_quadrant_l460_46046

theorem tan_double_angle_second_quadrant (α : Real) :
  (π/2 < α) ∧ (α < π) →  -- α is in the second quadrant
  Real.sin (π + α) = -3/5 →
  Real.tan (2*α) = -24/7 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_double_angle_second_quadrant_l460_46046


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_roots_is_eighteen_l460_46067

/-- A function satisfying the symmetry property f(3 + x) = f(3 - x) -/
def SymmetricAboutThree (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (3 + x) = f (3 - x)

/-- The property that a function has exactly 6 distinct real roots -/
def HasSixDistinctRoots (f : ℝ → ℝ) : Prop :=
  ∃ (r₁ r₂ r₃ r₄ r₅ r₆ : ℝ), (∀ x : ℝ, f x = 0 ↔ x = r₁ ∨ x = r₂ ∨ x = r₃ ∨ x = r₄ ∨ x = r₅ ∨ x = r₆) ∧
    r₁ ≠ r₂ ∧ r₁ ≠ r₃ ∧ r₁ ≠ r₄ ∧ r₁ ≠ r₅ ∧ r₁ ≠ r₆ ∧
    r₂ ≠ r₃ ∧ r₂ ≠ r₄ ∧ r₂ ≠ r₅ ∧ r₂ ≠ r₆ ∧
    r₃ ≠ r₄ ∧ r₃ ≠ r₅ ∧ r₃ ≠ r₆ ∧
    r₄ ≠ r₅ ∧ r₄ ≠ r₆ ∧
    r₅ ≠ r₆

theorem sum_of_roots_is_eighteen (f : ℝ → ℝ) 
    (h₁ : SymmetricAboutThree f) (h₂ : HasSixDistinctRoots f) : 
    ∃ (r₁ r₂ r₃ r₄ r₅ r₆ : ℝ), (∀ x : ℝ, f x = 0 ↔ x = r₁ ∨ x = r₂ ∨ x = r₃ ∨ x = r₄ ∨ x = r₅ ∨ x = r₆) ∧
    r₁ + r₂ + r₃ + r₄ + r₅ + r₆ = 18 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_roots_is_eighteen_l460_46067


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pendulum_proportionality_l460_46099

-- Define the variables and their types
variable (t l g : ℝ)

-- Define the relationship between t, l, and g
def pendulum_period (t l g : ℝ) : Prop := t = 2 * Real.pi * Real.sqrt (l / g)

-- Define proportionality
def proportional (x y : ℝ → ℝ) : Prop := ∃ k : ℝ, ∀ z : ℝ, x z = k * y z

-- Theorem statement
theorem pendulum_proportionality (h : pendulum_period t l g) :
  proportional (λ z ↦ z) (λ z ↦ Real.sqrt z) :=
by
  -- The proof is omitted for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_pendulum_proportionality_l460_46099


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_whole_number_above_sum_l460_46018

theorem smallest_whole_number_above_sum : ℕ := by
  let sum : ℚ := (2 + 1/4) + (3 + 1/5) + (4 + 1/6) + (5 + 1/7)
  let smallest_whole_number := Int.ceil sum
  have : smallest_whole_number = 15 := by sorry
  exact 15

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_whole_number_above_sum_l460_46018


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_nice_numbers_l460_46064

def IsNice (d : ℕ+) : Prop :=
  ∀ (x y : ℕ+), (d ∣ ((x + y)^5 - x^5 - y^5)) ↔ (d ∣ ((x + y)^7 - x^7 - y^7))

theorem infinitely_many_nice_numbers : ∃ (S : Set ℕ+), Set.Infinite S ∧ (∀ d, d ∈ S → IsNice d) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_infinitely_many_nice_numbers_l460_46064


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_product_l460_46016

theorem binomial_product : (Nat.choose 10 3) * (Nat.choose 8 3) = 6720 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_product_l460_46016


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_aliceProcessTerminates_l460_46061

/-- Represents a single step in Alice's process -/
inductive Step
  | incrementY : Step  -- represents (x, y) → (y+1, x)
  | decrementX : Step  -- represents (x, y) → (x-1, x)

/-- The sequence of positive integers -/
def Sequence := List Nat

/-- Applies a single step to adjacent elements in the sequence -/
def applyStep (s : Sequence) (i : Nat) (step : Step) : Sequence := sorry

/-- The sum of i * a[i] for all elements in the sequence -/
def weightedSum (s : Sequence) : Nat := sorry

/-- The maximum element in the sequence -/
def maxElement (s : Sequence) : Nat := sorry

/-- Theorem stating that Alice's process must terminate -/
theorem aliceProcessTerminates (s : Sequence) : 
  ∃ n : Nat, ∀ steps : List (Nat × Step), steps.length > n → 
    ¬(∃ s' : Sequence, s' = (steps.foldl (λ acc (i, step) => applyStep acc i step) s) ∧ 
      (∀ i < s'.length - 1, s'.get? i > s'.get? (i+1) → i < s.length - 1 ∧ s.get? i > s.get? (i+1))) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_aliceProcessTerminates_l460_46061


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_sqrt3_sin_plus_cos_l460_46032

theorem min_value_sqrt3_sin_plus_cos :
  (∀ x : ℝ, Real.sqrt 3 * Real.sin x + Real.cos x ≥ -2) ∧ 
  (∃ x : ℝ, Real.sqrt 3 * Real.sin x + Real.cos x = -2) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_sqrt3_sin_plus_cos_l460_46032


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_power_of_two_first_fifty_even_integers_l460_46013

/-- The product of the first n positive even integers -/
def productFirstEvenIntegers (n : ℕ) : ℕ :=
  List.range n |>.map (fun i => 2 * (i + 1)) |>.prod

/-- The largest power of 2 that divides n -/
def largestPowerOfTwo (n : ℕ) : ℕ :=
  Nat.factors n |>.filter (Eq 2) |>.length

theorem largest_power_of_two_first_fifty_even_integers :
  largestPowerOfTwo (productFirstEvenIntegers 50) = 97 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_power_of_two_first_fifty_even_integers_l460_46013


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_triangle_l460_46044

/-- Represents the relation of knowing between people -/
def Knows (n : ℕ) := Fin n → Fin n → Bool

/-- The property that each person knows at least ⌊n/2⌋ others -/
def EachKnowsMany (n : ℕ) (knows : Knows n) : Prop :=
  ∀ i : Fin n, (Finset.univ.filter (fun j => knows i j)).card ≥ n / 2

/-- The property that for any group of ⌊n/2⌋ people, either two of them know each other,
    or among the remaining people, two of them know each other -/
def GroupProperty (n : ℕ) (knows : Knows n) : Prop :=
  ∀ s : Finset (Fin n), s.card = n / 2 →
    (∃ i j, i ∈ s ∧ j ∈ s ∧ i ≠ j ∧ knows i j) ∨
    (∃ i j, i ∈ (Finset.univ \ s) ∧ j ∈ (Finset.univ \ s) ∧ i ≠ j ∧ knows i j)

/-- The existence of a triangle (three people who mutually know each other) -/
def HasTriangle (n : ℕ) (knows : Knows n) : Prop :=
  ∃ i j k : Fin n, i ≠ j ∧ j ≠ k ∧ i ≠ k ∧ knows i j ∧ knows j k ∧ knows i k

/-- The main theorem: under the given conditions, there must be a triangle -/
theorem existence_of_triangle (n : ℕ) (h : n ≥ 6) (knows : Knows n)
    (h_many : EachKnowsMany n knows) (h_group : GroupProperty n knows) :
    HasTriangle n knows := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_existence_of_triangle_l460_46044


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_xi_in_right_triangle_l460_46069

open Real

/-- In a right triangle, given an acute angle α where tan α = ∜3, 
    ξ is the angle between the median and angle bisector drawn from α. -/
theorem tan_xi_in_right_triangle (α ξ : ℝ) (h : tan α = (3 : ℝ) ^ (1/4)) :
  tan ξ = ((3 : ℝ) ^ (1/4) / (1 - sqrt 3) - (3 : ℝ) ^ (1/8)) / 
          (1 + ((3 : ℝ) ^ (1/4) / (1 - sqrt 3)) * (3 : ℝ) ^ (1/8)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tan_xi_in_right_triangle_l460_46069


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bf_length_l460_46079

-- Define the quadrilateral ABCD
structure Quadrilateral :=
  (A B C D : ℝ × ℝ)

-- Define the additional points E and F
structure QuadrilateralWithEF extends Quadrilateral :=
  (E F : ℝ × ℝ)

-- Define the properties of the quadrilateral
def isValidQuadrilateral (q : QuadrilateralWithEF) : Prop :=
  -- Right angles at A and C
  (q.A.1 = q.D.1 ∧ q.A.2 = q.B.2) ∧
  (q.C.1 = q.B.1 ∧ q.C.2 = q.D.2) ∧
  -- E and F are on AC
  (∃ t₁ t₂ : ℝ, 0 ≤ t₁ ∧ t₁ ≤ 1 ∧ 0 ≤ t₂ ∧ t₂ ≤ 1 ∧
    q.E = (q.A.1 + t₁ * (q.C.1 - q.A.1), q.A.2 + t₁ * (q.C.2 - q.A.2)) ∧
    q.F = (q.A.1 + t₂ * (q.C.1 - q.A.1), q.A.2 + t₂ * (q.C.2 - q.A.2))) ∧
  -- DE and BF are perpendicular to AC
  ((q.D.2 - q.E.2) * (q.C.1 - q.A.1) = (q.D.1 - q.E.1) * (q.C.2 - q.A.2)) ∧
  ((q.B.2 - q.F.2) * (q.C.1 - q.A.1) = (q.B.1 - q.F.1) * (q.C.2 - q.A.2))

-- Define the given lengths
def givenLengths (q : QuadrilateralWithEF) : Prop :=
  ((q.E.1 - q.A.1)^2 + (q.E.2 - q.A.2)^2)^(1/2 : ℝ) = 4 ∧
  ((q.E.1 - q.D.1)^2 + (q.E.2 - q.D.2)^2)^(1/2 : ℝ) = 5 ∧
  ((q.C.1 - q.E.1)^2 + (q.C.2 - q.E.2)^2)^(1/2 : ℝ) = 6

-- Theorem statement
theorem bf_length (q : QuadrilateralWithEF) 
  (h1 : isValidQuadrilateral q) (h2 : givenLengths q) :
  ((q.B.1 - q.F.1)^2 + (q.B.2 - q.F.2)^2)^(1/2 : ℝ) = 1000/245 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_bf_length_l460_46079


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_total_discounted_price_l460_46082

/-- Calculates the discounted price based on the given discount policy -/
noncomputable def discountedPrice (price : ℝ) : ℝ :=
  if price ≤ 200 then price
  else if price ≤ 500 then price * 0.9
  else 500 * 0.9 + (price - 500) * 0.7

/-- The price of product A -/
def priceA : ℝ := 100

/-- The price of product B -/
def priceB : ℝ := 450

/-- Theorem stating that the total discounted price for products A and B is 520 yuan -/
theorem total_discounted_price :
  discountedPrice (priceA + priceB) = 520 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_total_discounted_price_l460_46082


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_values_for_two_subsets_l460_46000

def A (a : ℝ) : Set ℝ := {x | a * x^2 + 2 * x + a = 0}

theorem a_values_for_two_subsets :
  ∀ a : ℝ, (∃! s : Set (Set ℝ), s = {∅, A a}) → a ∈ ({0, 1, -1} : Set ℝ) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_values_for_two_subsets_l460_46000


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l460_46089

-- Define the function f(x) = ln x - x²
noncomputable def f (x : ℝ) := Real.log x - x^2

-- State the theorem
theorem f_properties :
  -- The domain of f is (0, +∞)
  ∀ x > 0,
  -- 1. The maximum value of f(x) occurs at x = √2/2
  (∀ y > 0, f (Real.sqrt 2 / 2) ≥ f y) ∧
  -- 2. f(x) is increasing on the interval (1/2, √2/2)
  (∀ y z, 1/2 < y ∧ y < z ∧ z < Real.sqrt 2 / 2 → f y < f z) ∧
  -- 3. f(x) < e^x - x² - 2 for all x in (0, +∞)
  f x < Real.exp x - x^2 - 2 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l460_46089


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_degree_three_polynomial_l460_46040

/-- Given two polynomials f and g, prove that f + cg has degree 3 when c = -3/5 -/
theorem degree_three_polynomial : 
  let f : Polynomial ℝ := 2 - 15*X + 4*X^2 - 5*X^3 + 6*X^4
  let g : Polynomial ℝ := 5 - 3*X - 7*X^3 + 10*X^4
  let c : ℝ := -3/5
  let h : Polynomial ℝ := f + c • g
  Polynomial.degree h = 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_degree_three_polynomial_l460_46040


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l460_46081

theorem problem_solution : 
  (Real.sqrt 3 + 2 * Real.sqrt 2) - (3 * Real.sqrt 3 + Real.sqrt 2) = -2 * Real.sqrt 3 + Real.sqrt 2 ∧
  Real.sqrt 2 * (Real.sqrt 2 + 1 / Real.sqrt 2) - |2 - Real.sqrt 6| = 5 - Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l460_46081


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_angle_solution_l460_46087

noncomputable def m (x : ℝ) : ℝ × ℝ := (Real.sqrt 3, x)
noncomputable def n : ℝ × ℝ := (1, Real.sqrt 3)

theorem vector_angle_solution (x : ℝ) :
  let m := m x
  let n := n
  let angle := π / 6
  let dot_product := m.1 * n.1 + m.2 * n.2
  let magnitude_m := Real.sqrt (m.1 ^ 2 + m.2 ^ 2)
  let magnitude_n := Real.sqrt (n.1 ^ 2 + n.2 ^ 2)
  dot_product = magnitude_m * magnitude_n * Real.cos angle →
  x = 1 := by
  sorry

#check vector_angle_solution

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_angle_solution_l460_46087


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ordering_of_p_q_r_l460_46021

-- Define the constants
noncomputable def a : ℝ := Real.cos 1
noncomputable def p : ℝ := Real.log (1/2) / Real.log a
noncomputable def q : ℝ := a^(1/2)
noncomputable def r : ℝ := (1/2)^a

-- State the theorem
theorem ordering_of_p_q_r : r < q ∧ q < p := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ordering_of_p_q_r_l460_46021


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_theorem_l460_46038

-- Define the function f
def f (x : ℝ) : ℝ := (x - 1)^2

-- Define the inverse function f_inv
noncomputable def f_inv (x : ℝ) : ℝ := -Real.sqrt x + 1

-- State the theorem
theorem inverse_function_theorem :
  (∀ x ≤ 0, f x = (x - 1)^2) →
  (∀ x ≥ 1, f_inv x = -Real.sqrt x + 1) →
  (∀ x ≤ 0, f_inv (f x) = x) ∧
  (∀ x ≥ 1, f (f_inv x) = x) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_function_theorem_l460_46038


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_spherical_coord_negated_y_l460_46049

-- Define the types for rectangular and spherical coordinates
def RectCoord := ℝ × ℝ × ℝ
def SphCoord := ℝ × ℝ × ℝ

-- Define the conversion functions (we won't implement them, just declare their existence)
noncomputable def rect_to_sph : RectCoord → SphCoord := sorry
noncomputable def sph_to_rect : SphCoord → RectCoord := sorry

-- Define the theorem
theorem spherical_coord_negated_y (P : RectCoord) :
  let (x, y, z) := P
  let P' : RectCoord := (x, -y, z)
  rect_to_sph P = (3, 5*π/6, π/3) →
  rect_to_sph P' = (3, 7*π/6, π/3) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_spherical_coord_negated_y_l460_46049


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l460_46029

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

theorem triangle_properties (t : Triangle) 
  (h_area : t.a^2 / (3 * Real.sin t.A) = t.a * t.c * Real.sin t.B / 2) :
  /- Part 1 -/
  Real.sin t.B * Real.sin t.C = 2/3 ∧
  /- Part 2 -/
  (6 * Real.cos t.B * Real.cos t.C = 1 ∧ t.a = 3) →
  t.a + t.b + t.c = 3 + Real.sqrt 33 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_properties_l460_46029


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_central_angle_is_pi_over_three_l460_46053

-- Define the circle
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 = 4

-- Define the line
def line_eq (x y : ℝ) : Prop := Real.sqrt 3 * x + y - 2 * Real.sqrt 3 = 0

-- Define the distance from the center of the circle to the line
noncomputable def distance_center_to_line : ℝ := Real.sqrt 3

-- Define the radius of the circle
def radius : ℝ := 2

-- Theorem statement
theorem central_angle_is_pi_over_three :
  ∀ x y : ℝ, circle_eq x y → line_eq x y →
  Real.arccos (distance_center_to_line / radius) = π / 3 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_central_angle_is_pi_over_three_l460_46053


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_angle_exists_l460_46034

open Real

-- Define the equation
def angle_equation (φ : ℝ) : Prop :=
  cos φ = sin (60 * π / 180) + sin (48 * π / 180) - 
          cos (12 * π / 180) - sin (10 * π / 180)

-- State the theorem
theorem smallest_positive_angle_exists : 
  ∃ φ : ℝ, 
    φ > 0 ∧ 
    φ < 2 * π ∧
    angle_equation φ ∧
    (∀ ψ : ℝ, ψ > 0 ∧ ψ < φ → ¬angle_equation ψ) ∧
    abs (φ - 81 * π / 180) < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_angle_exists_l460_46034


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_equation_solutions_l460_46031

theorem quadratic_equation_solutions (a b c : ℝ) (ha : a ≠ 0) :
  ∃ (S : Set ℝ), (∀ x : ℝ, x ∈ S ↔ a * x^2 + b * x + c = 0) ∧ Set.Finite S ∧ Set.ncard S ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_equation_solutions_l460_46031


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_fixed_point_l460_46003

-- Define the ellipse C
noncomputable def C (x y : ℝ) : Prop := x^2 / 4 + y^2 = 1

-- Define the vertices of the ellipse
def A : ℝ × ℝ := (0, 1)
def B : ℝ × ℝ := (0, -1)

-- Define a point M
def M (t : ℝ) : ℝ × ℝ := (t, 2)

-- Define the intersection points P and Q
noncomputable def P (t : ℝ) : ℝ × ℝ := (-8*t / (t^2 + 4), (t^2 - 4) / (t^2 + 4))
noncomputable def Q (t : ℝ) : ℝ × ℝ := (24*t / (t^2 + 36), (36 - t^2) / (t^2 + 36))

-- The theorem to prove
theorem ellipse_fixed_point (t : ℝ) (h : t ≠ 0) :
  ∃ (k : ℝ), (P t).1 * k + (P t).2 * (1 - k) = 0 ∧
             (Q t).1 * k + (Q t).2 * (1 - k) = 1/2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_fixed_point_l460_46003


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l460_46020

def arithmetic_sequence : List ℚ := [4, 6, 13, 27, 50, 84]

def is_arithmetic_progression_of_order (seq : List ℚ) (k : ℕ) : Prop :=
  ∃ (f : ℕ → ℚ), (∀ n, n < seq.length → seq.get! n = f (n + 1)) ∧
  (∃ (a b c d : ℚ), ∀ n, f n = (a * n^3 + b * n^2 + c * n + d) / 6)

def nth_term (n : ℕ) : ℚ := (2 * n^3 + 3 * n^2 - 11 * n + 30) / 6

theorem sequence_properties :
  is_arithmetic_progression_of_order arithmetic_sequence 3 ∧
  ∀ n, n ≤ arithmetic_sequence.length → arithmetic_sequence.get? (n - 1) = some (nth_term n) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_properties_l460_46020


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_power_sum_l460_46012

theorem complex_power_sum (z : ℂ) (h : z + z⁻¹ = Real.sqrt 2) : z^8 + (z^8)⁻¹ = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_power_sum_l460_46012


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l460_46094

theorem problem_solution :
  ∀ (a b c d m n : ℝ),
    (a = -b) →
    (c * d = 1) →
    (m = Int.natAbs (-1)) →
    (|n| = n ∧ n⁻¹ = n) →
    (m = -1 ∧ n = 1 ∧ a + b - c * d + m - n = -3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l460_46094


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_b_value_l460_46055

/-- Two lines are parallel if and only if their slopes are equal -/
axiom parallel_lines_equal_slopes (m1 m2 : ℝ) : m1 = m2 ↔ m1 = m2

/-- The problem statement -/
theorem parallel_lines_b_value : 
  ∀ (b : ℝ), 3 = (b + 9) → b = -6 := by
  intro b hyp
  -- Proof steps would go here
  sorry

#check parallel_lines_b_value

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parallel_lines_b_value_l460_46055


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l460_46080

noncomputable def f (x : ℝ) : ℝ := (x - 2) ^ (1/3) + Real.sqrt (10 - x)

theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = Set.Iic 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l460_46080


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_heat_released_formula_l460_46019

/-- Represents an electrical circuit with a resistor and capacitor in series, connected to a galvanic cell. -/
structure Circuit where
  C : ℝ  -- Capacitance
  ε : ℝ  -- Electromotive force (EMF)

/-- Calculates the heat released in the resistor during capacitor charging. -/
noncomputable def heat_released (circuit : Circuit) : ℝ :=
  (1/2) * circuit.C * (circuit.ε^2)

/-- Theorem: The heat released in the resistor during capacitor charging is (1/2) C ε^2. -/
theorem heat_released_formula (circuit : Circuit) :
  heat_released circuit = (1/2) * circuit.C * (circuit.ε^2) := by
  -- Unfold the definition of heat_released
  unfold heat_released
  -- The equality follows directly from the definition
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_heat_released_formula_l460_46019


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_plane_existence_and_uniqueness_l460_46027

-- Define a ray as a point (origin) and a direction vector
structure Ray where
  origin : EuclideanSpace ℝ (Fin 3)
  direction : EuclideanSpace ℝ (Fin 3)

-- Define a plane as a point and a normal vector
structure Plane where
  point : EuclideanSpace ℝ (Fin 3)
  normal : EuclideanSpace ℝ (Fin 3)

-- Function to check if a ray is symmetric to another ray with respect to a plane
def isSymmetric (r1 r2 : Ray) (p : Plane) : Prop := sorry

-- Function to check if a ray forms congruent angles with two given rays
def formsCongruentAngles (r : Ray) (r1 r2 : Ray) : Prop := sorry

-- Function to check if a ray lies on a plane
def rayOnPlane (r : Ray) (p : Plane) : Prop := sorry

theorem symmetry_plane_existence_and_uniqueness 
  (r1 r2 : Ray) (h : r1.origin = r2.origin ∧ r1 ≠ r2) : 
  ∃! p : Plane,
    (isSymmetric r1 r2 p) ∧ 
    (∀ r : Ray, r.origin = r1.origin → 
      (formsCongruentAngles r r1 r2 ↔ rayOnPlane r p)) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_symmetry_plane_existence_and_uniqueness_l460_46027


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_coefficient_value_l460_46045

theorem min_coefficient_value (a b Box : ℤ) : 
  (a ≠ b ∧ a ≠ Box ∧ b ≠ Box) →
  ((∀ x, (a * x + b) * (b * x + a) = 34 * x^2 + Box * x + 34) →
  (∀ k : ℤ, (∃ a' b' : ℤ, (a' ≠ b' ∧ a' ≠ k ∧ b' ≠ k) ∧
    (∀ x, (a' * x + b') * (b' * x + a') = 34 * x^2 + k * x + 34)) →
    k ≥ Box)) →
  Box = 293 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_coefficient_value_l460_46045


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fairy_tale_island_theorem_l460_46006

/-- Represents the types of inhabitants in the fairy tale island. -/
inductive Inhabitant
  | Elf
  | Dwarf
  | Centaur

/-- Represents a county on the fairy tale island. -/
structure County where
  inhabitants : Inhabitant

/-- Represents the state of the fairy tale island. -/
structure IslandState where
  counties : List County

/-- Initial state of the island with three counties. -/
def initial_state : IslandState :=
  { counties := [
    { inhabitants := Inhabitant.Elf },
    { inhabitants := Inhabitant.Dwarf },
    { inhabitants := Inhabitant.Centaur }
  ] }

/-- Transformation rule for the first year. -/
def transform_year1 (state : IslandState) : IslandState :=
  { counties := state.counties.bind (fun c =>
    match c.inhabitants with
    | Inhabitant.Elf => [c]
    | _ => [c, c, c]
  ) }

/-- Transformation rule for the second year. -/
def transform_year2 (state : IslandState) : IslandState :=
  { counties := state.counties.bind (fun c =>
    match c.inhabitants with
    | Inhabitant.Dwarf => [c]
    | _ => [c, c, c, c]
  ) }

/-- Transformation rule for the third year. -/
def transform_year3 (state : IslandState) : IslandState :=
  { counties := state.counties.bind (fun c =>
    match c.inhabitants with
    | Inhabitant.Centaur => [c]
    | _ => [c, c, c, c, c, c]
  ) }

/-- Final state after all transformations. -/
def final_state : IslandState :=
  transform_year3 (transform_year2 (transform_year1 initial_state))

theorem fairy_tale_island_theorem :
  final_state.counties.length = 54 := by
  sorry

#eval final_state.counties.length

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fairy_tale_island_theorem_l460_46006
