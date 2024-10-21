import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_segment_length_bound_l370_37066

-- Define the polygon and points
variable (P : Set (ℝ × ℝ))
variable (S T A B C : ℝ × ℝ)

-- Define the properties
variable (convex_P : ConvexSet P)
variable (boundary_points : S ∈ frontier P ∧ T ∈ frontier P ∧ A ∈ frontier P ∧ B ∈ frontier P ∧ C ∈ frontier P)
variable (perimeter_division : ArcLength P S T = 2 * ArcLength P T S)
variable (distinct_points : A ≠ B ∧ B ≠ C ∧ C ≠ A)

-- Define the movement of points A, B, C
variable (t : ℝ)
variable (move_A : ℝ → ℝ × ℝ)
variable (move_B : ℝ → ℝ × ℝ)
variable (move_C : ℝ → ℝ × ℝ)
variable (equal_speed : ∀ t, dist (move_A t) (move_A 0) = dist (move_B t) (move_B 0) ∧
                               dist (move_B t) (move_B 0) = dist (move_C t) (move_C 0))

-- The theorem to prove
theorem min_segment_length_bound :
  ∃ t, min (dist (move_A t) (move_B t)) (min (dist (move_B t) (move_C t)) (dist (move_C t) (move_A t))) ≤ dist S T := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_segment_length_bound_l370_37066


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_seating_arrangements_l370_37053

/-- Represents a person in the seating arrangement -/
inductive Person : Type
| Alice : Person
| Bob : Person
| Carla : Person
| Derek : Person
| Emily : Person
deriving DecidableEq

/-- Represents a seating arrangement as a list of people -/
def SeatingArrangement := List Person

/-- Checks if two people are adjacent in a circular seating arrangement -/
def are_adjacent (p1 p2 : Person) (arrangement : SeatingArrangement) : Prop :=
  ∃ i, (arrangement.get? i = some p1 ∧ arrangement.get? ((i + 1) % arrangement.length) = some p2) ∨
       (arrangement.get? i = some p2 ∧ arrangement.get? ((i + 1) % arrangement.length) = some p1)

/-- Checks if a seating arrangement is valid according to the given conditions -/
def is_valid_arrangement (arrangement : SeatingArrangement) : Prop :=
  arrangement.length = 5 ∧
  arrangement.toFinset = {Person.Alice, Person.Bob, Person.Carla, Person.Derek, Person.Emily} ∧
  ¬(are_adjacent Person.Alice Person.Bob arrangement) ∧
  ¬(are_adjacent Person.Alice Person.Carla arrangement) ∧
  ¬(are_adjacent Person.Derek Person.Emily arrangement)

/-- The main theorem: there are exactly 4 valid seating arrangements -/
theorem valid_seating_arrangements :
  ∃! (arrangements : List SeatingArrangement),
    arrangements.length = 4 ∧
    (∀ arr, arr ∈ arrangements ↔ is_valid_arrangement arr) := by
  sorry

#check valid_seating_arrangements

end NUMINAMATH_CALUDE_ERRORFEEDBACK_valid_seating_arrangements_l370_37053


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l370_37085

/-- The function f as defined in the problem -/
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := (1/4) * x^2 + m*x - 3/4

/-- The sequence a_n -/
def a : ℕ → ℝ
  | n => 2 * n + 1

/-- The sequence b_n -/
noncomputable def b : ℕ → ℝ
  | n => (1 / (a n + 1))^2

/-- The sum S_n -/
noncomputable def S (n : ℕ) : ℝ := f (1/2) (a n)

/-- The sum T_n -/
noncomputable def T : ℕ → ℝ
  | 0 => 0
  | n + 1 => T n + b (n + 1)

theorem problem_solution :
  (∀ (α β : ℝ), f (1/2) (Real.sin α) ≤ 0 ∧ f (1/2) (2 + Real.cos β) ≥ 0) →
  (∀ n, 0 < a n) →
  (∀ n, Real.sqrt (b n) = 1 / (a n + 1)) →
  (∀ n, a n = 2 * n + 1) ∧
  (∀ n, T n < 1/6) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l370_37085


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_b_alone_time_l370_37015

/-- The time it takes for worker a to finish the work alone -/
noncomputable def Ta : ℝ := sorry

/-- The time it takes for worker b to finish the work alone -/
noncomputable def Tb : ℝ := sorry

/-- The time it takes for worker c to finish the work alone -/
noncomputable def Tc : ℝ := sorry

/-- a takes twice as much time as b -/
axiom a_twice_b : Ta = 2 * Tb

/-- a takes thrice as much time as c -/
axiom a_thrice_c : Ta = 3 * Tc

/-- a, b, and c working together can finish the work in 3 days -/
axiom combined_work_rate : 1 / Ta + 1 / Tb + 1 / Tc = 1 / 3

/-- a and b working together on the first day complete 20% of the work -/
axiom ab_first_day : 1 / Ta + 1 / Tb = 0.2 / 1

/-- b and c working together on the second day complete 25% of the remaining work -/
axiom bc_second_day : 1 / Tb + 1 / Tc = 0.25 * 0.8 / 1

theorem b_alone_time : Tb = 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_b_alone_time_l370_37015


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_monotone_increasing_interval_l370_37080

/-- The function f(x) = 4lnx - (1/2)x^2 + 3x -/
noncomputable def f (x : ℝ) : ℝ := 4 * Real.log x - (1/2) * x^2 + 3 * x

/-- Theorem: If f(x) is monotonically increasing on [a, a+1], then a ∈ (0, 3] -/
theorem monotone_increasing_interval (a : ℝ) : 
  (∀ x y, x ∈ Set.Icc a (a + 1) → y ∈ Set.Icc a (a + 1) → x ≤ y → f x ≤ f y) → 
  0 < a ∧ a ≤ 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_monotone_increasing_interval_l370_37080


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_properties_of_f_2_minus_x_l370_37097

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the domain of f
def domain_f : Set ℝ := Set.Ioo (-10) 10

-- f is an even function
axiom f_even : ∀ x, f x = f (-x)

-- (2, 6) is an increasing interval of f
axiom f_increasing_on_2_6 : StrictMonoOn f (Set.Ioo 2 6)

-- Main theorem
theorem properties_of_f_2_minus_x :
  (StrictMonoOn (fun x => f (2 - x)) (Set.Ioo 4 8)) ∧
  (∀ x, f (2 - x) = f (2 - (4 - x))) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_properties_of_f_2_minus_x_l370_37097


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_divisible_by_prime_l370_37068

/-- Sequence definition -/
def x (a b c : ℤ) : ℕ → ℤ
| 0 => 4
| 1 => 0
| 2 => 2 * c
| 3 => 3 * b
| (n + 3) => a * x a b c (n - 1) + b * x a b c n + c * x a b c (n + 1)

/-- Main theorem -/
theorem x_divisible_by_prime (a b c : ℤ) (h_b_odd : Odd b) :
  ∀ (m : ℕ) (p : ℕ), Nat.Prime p → (p : ℤ) ∣ x a b c (p ^ m) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_divisible_by_prime_l370_37068


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_of_P_l370_37019

-- Define the circle
def circle_equation (x y : ℝ) : Prop := x^2 + y^2 = 4

-- Define point A
def point_A : ℝ × ℝ := (8, 0)

-- Define the line passing through A and B
def line_AB (x y : ℝ) : Prop := ∃ (t : ℝ), x = 8 + t ∧ y = t

-- Define point B as the intersection of the line and the circle
noncomputable def point_B : ℝ × ℝ := sorry

-- Define point P as the midpoint of AB
def point_P (x y : ℝ) : Prop :=
  x = (8 + point_B.1) / 2 ∧ y = point_B.2 / 2

-- Theorem statement
theorem trajectory_of_P :
  ∀ x y : ℝ, point_P x y → (x - 4)^2 + y^2 = 1 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_of_P_l370_37019


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_and_max_distance_l370_37038

/-- Represents an ellipse with semi-major axis a and semi-minor axis b -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_pos : 0 < b ∧ b < a

/-- Represents a circle with radius r -/
structure Circle where
  r : ℝ
  h_pos : 0 < r

theorem ellipse_properties_and_max_distance (C : Ellipse) (O : Circle)
    (h_ecc : C.a / Real.sqrt (C.a^2 - C.b^2) = 2 / Real.sqrt 3)
    (h_minor : C.b = 1)
    (h_unit_circle : O.r = 1) :
    (∃ (x y : ℝ), x^2 / 4 + y^2 = 1 ↔ x^2 / C.a^2 + y^2 / C.b^2 = 1) ∧
    (∃ (M : ℝ × ℝ), ∀ (A B : ℝ × ℝ),
      (∃ (m t : ℝ), (∀ x y, x = m * y + t → x^2 + y^2 = 1) ∧
                    A.1^2 / C.a^2 + A.2^2 / C.b^2 = 1 ∧
                    B.1^2 / C.a^2 + B.2^2 / C.b^2 = 1 ∧
                    M = ((A.1 + B.1) / 2, (A.2 + B.2) / 2)) →
      M.1^2 + M.2^2 ≤ 25/16) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_properties_and_max_distance_l370_37038


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_and_inequality_l370_37029

noncomputable def f (x : ℝ) : ℝ := Real.exp x
def g (k : ℝ) (x : ℝ) : ℝ := k * x + 1
noncomputable def h (k : ℝ) (x : ℝ) : ℝ := f x - g k x

theorem tangent_and_inequality (k : ℝ) :
  (∃ t : ℝ, f t = g k t ∧ (deriv f) t = k) ∧
  (∀ m : ℤ, (∀ x : ℝ, x > 0 → (m - x) * (deriv (h k)) x < x + 1) → m ≤ 2) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_and_inequality_l370_37029


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_A_intersect_B_l370_37036

def A : Set ℝ := {y : ℝ | ∃ x : ℝ, y = x^2}
def B : Set ℝ := {-2, -1, 1, 2}

theorem complement_A_intersect_B : 
  (Set.compl A) ∩ B = {-2, -1} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complement_A_intersect_B_l370_37036


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_factorial_sum_of_squares_l370_37078

theorem factorial_sum_of_squares (n : ℕ) (a b : ℕ) :
  n < 14 →
  (Nat.factorial n = a^2 + b^2) ↔ 
  ((n = 2 ∧ a = 1 ∧ b = 1) ∨ (n = 6 ∧ a = 24 ∧ b = 12)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_factorial_sum_of_squares_l370_37078


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_hexagon_with_sectors_l370_37059

/-- The area of the shaded region in a regular hexagon with circular sectors --/
theorem shaded_area_hexagon_with_sectors (side_length : ℝ) (sector_radius : ℝ) 
  (h1 : side_length = 6)
  (h2 : sector_radius = 3) : ℝ := by
  -- Define the shaded area
  let shaded_area : ℝ := 54 * Real.sqrt 3 - 18 * Real.pi
  
  -- Proof steps (to be filled in)
  -- Step 1: Calculate the area of the hexagon
  -- Step 2: Calculate the area of the circular sectors
  -- Step 3: Calculate the area of the shaded region
  
  -- For now, we'll use sorry to skip the proof
  sorry

#check shaded_area_hexagon_with_sectors

end NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_hexagon_with_sectors_l370_37059


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_min_difference_l370_37074

open Real

noncomputable def f (x : ℝ) := 3 - sin x - 2 * (cos x)^2

theorem f_max_min_difference :
  let a := π / 6
  let b := 7 * π / 6
  ∃ (max min : ℝ), 
    (∀ x ∈ Set.Icc a b, f x ≤ max) ∧ 
    (∃ x ∈ Set.Icc a b, f x = max) ∧
    (∀ x ∈ Set.Icc a b, min ≤ f x) ∧ 
    (∃ x ∈ Set.Icc a b, f x = min) ∧
    max - min = 9 / 8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_min_difference_l370_37074


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_A_B_l370_37083

def A : Set ℕ := {1, 2, 3, 4}
def B : Set ℕ := {x : ℕ | 1 < x ∧ x < 5}

theorem intersection_A_B : A ∩ B = {2, 3, 4} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_A_B_l370_37083


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_of_selecting_A_group_size_positive_l370_37058

/-- The probability of selecting a specific person from a group using a fair selection method -/
def probability_of_selection (total_people : ℕ) : ℚ :=
  1 / total_people

/-- The number of people in the group -/
def group_size : ℕ := 4

/-- Theorem: The probability of selecting person A from a group of 4 people is 1/4 -/
theorem probability_of_selecting_A :
  probability_of_selection group_size = 1 / 4 := by
  -- Unfold the definition of probability_of_selection
  unfold probability_of_selection
  -- Simplify the fraction
  simp
  -- Check that the result is correct
  rfl

/-- Proof that group_size is positive -/
theorem group_size_positive : 0 < group_size := by
  -- Unfold the definition of group_size
  unfold group_size
  -- Use norm_num to prove that 0 < 4
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_of_selecting_A_group_size_positive_l370_37058


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_diagonal_small_triangle_l370_37017

/-- A convex hexagon in a 2D plane -/
structure ConvexHexagon where
  vertices : Finset (ℝ × ℝ)
  convex : Convex ℝ (convexHull ℝ (vertices.toSet))
  card_eq : vertices.card = 6

/-- The area of a polygon -/
noncomputable def area (polygon : Set (ℝ × ℝ)) : ℝ := sorry

/-- A diagonal of a hexagon -/
def diagonal (h : ConvexHexagon) : Set (ℝ × ℝ × ℝ × ℝ) := sorry

/-- The triangle cut off by a diagonal -/
def cutOffTriangle (h : ConvexHexagon) (d : ℝ × ℝ × ℝ × ℝ) : Set (ℝ × ℝ) := sorry

theorem hexagon_diagonal_small_triangle (h : ConvexHexagon) :
  ∃ d ∈ diagonal h, 
    area (cutOffTriangle h d) ≤ (1/6 : ℝ) * area (convexHull ℝ h.vertices.toSet) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagon_diagonal_small_triangle_l370_37017


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_esther_commute_distance_l370_37033

/-- The distance Esther drives to work -/
def distance_to_work : ℝ := sorry

/-- The speed Esther drives to work in the morning (in miles per hour) -/
def morning_speed : ℝ := 45

/-- The speed Esther drives back home in the evening (in miles per hour) -/
def evening_speed : ℝ := 30

/-- The total time Esther spends commuting (in hours) -/
def total_commute_time : ℝ := 1

/-- Theorem stating that given the conditions, the distance Esther drives to work is 18 miles -/
theorem esther_commute_distance :
  (distance_to_work / morning_speed) + (distance_to_work / evening_speed) = total_commute_time →
  distance_to_work = 18 :=
by
  intro h
  -- The proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_esther_commute_distance_l370_37033


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_remainders_is_93_l370_37054

/-- Represents a positive integer with four consecutive digits in increasing order -/
def ConsecutiveDigitNumber (m : Nat) : Nat :=
  1000 * m + 100 * (m + 1) + 10 * (m + 2) + (m + 3)

/-- The set of valid smallest digits for ConsecutiveDigitNumber -/
def ValidSmallestDigits : Finset Nat :=
  {0, 1, 2, 3, 4, 5, 6}

/-- The sum of remainders when dividing ConsecutiveDigitNumber by 41 for all valid smallest digits -/
def SumOfRemainders : Nat :=
  (Finset.sum ValidSmallestDigits fun m => (ConsecutiveDigitNumber m) % 41)

theorem sum_of_remainders_is_93 : SumOfRemainders = 93 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_remainders_is_93_l370_37054


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_heartbeats_count_l370_37076

/-- Represents the blood pressure function over time. -/
noncomputable def blood_pressure (t : ℝ) : ℝ := 24 * Real.sin (160 * Real.pi * t) + 110

/-- Calculates the number of heartbeats per minute based on the blood pressure function. -/
noncomputable def heartbeats_per_minute : ℝ := 160 * Real.pi / (2 * Real.pi)

/-- Theorem stating that the number of heartbeats per minute is 80. -/
theorem heartbeats_count : heartbeats_per_minute = 80 := by
  -- Unfold the definition of heartbeats_per_minute
  unfold heartbeats_per_minute
  -- Simplify the expression
  simp [Real.pi]
  -- The proof is completed with sorry for now
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_heartbeats_count_l370_37076


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_root_sum_zero_l370_37034

theorem cubic_root_sum_zero (a b c : ℝ) : 
  (∃ x y z : ℝ, x^3 + a*x^2 + b*x + c = 0 ∧ 
               y^3 + a*y^2 + b*y + c = 0 ∧ 
               z^3 + a*z^2 + b*z + c = 0 ∧ 
               y + z = 0 ∧ x ≠ y ∧ x ≠ z) ↔ 
  a * b = c := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_root_sum_zero_l370_37034


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_coloring_theorem_l370_37009

/-- A point on a sphere --/
structure SpherePoint where
  x : ℝ
  y : ℝ
  z : ℝ
  on_sphere : x^2 + y^2 + z^2 = 16

/-- Color of a point --/
inductive Color
  | Red
  | Blue
  | Green
  | Yellow

/-- A coloring of the sphere --/
def SphereColoring := SpherePoint → Color

/-- Euclidean distance between two points --/
noncomputable def distance (p q : SpherePoint) : ℝ :=
  Real.sqrt ((p.x - q.x)^2 + (p.y - q.y)^2 + (p.z - q.z)^2)

theorem sphere_coloring_theorem (coloring : SphereColoring) :
  ∃ (p q : SpherePoint), coloring p = coloring q ∧
    (distance p q = 4 * Real.sqrt 3 ∨ distance p q = 2 * Real.sqrt 6) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sphere_coloring_theorem_l370_37009


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l370_37035

noncomputable def f (x : ℝ) : ℝ := Real.log x + x + 1

theorem tangent_line_equation (x₀ : ℝ) (h₁ : x₀ > 0) :
  (deriv f x₀ = 2) → 
  ∃ y₀ : ℝ, f x₀ = y₀ ∧ (λ x ↦ 2*x) = (λ x ↦ deriv f x₀ * (x - x₀) + y₀) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l370_37035


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_digit_difference_theorem_l370_37087

def digits : List Nat := [7, 3, 1, 4]

def largest_number (digits : List Nat) : Nat :=
  (digits.toArray.qsort (· > ·)).toList.foldl (fun acc d => acc * 10 + d) 0

def smallest_number (digits : List Nat) : Nat :=
  (digits.toArray.qsort (· < ·)).toList.foldl (fun acc d => acc * 10 + d) 0

theorem digit_difference_theorem :
  largest_number digits - smallest_number digits = 6084 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_digit_difference_theorem_l370_37087


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intervals_of_increase_range_of_m_l370_37077

-- Define the vectors and function
noncomputable def a (x : ℝ) : ℝ × ℝ := (2 * Real.cos x, 1)
noncomputable def b (x m : ℝ) : ℝ × ℝ := (Real.cos x, Real.sqrt 3 * Real.sin (2 * x) + m)
noncomputable def f (x m : ℝ) : ℝ := (a x).1 * (b x m).1 + (a x).2 * (b x m).2 - 1

-- Theorem for part (1)
theorem intervals_of_increase (x : ℝ) (h : x ∈ Set.Icc 0 Real.pi) :
  (MonotoneOn f (Set.Icc 0 (Real.pi / 6))) ∧
  (MonotoneOn f (Set.Icc ((2 * Real.pi) / 3) Real.pi)) :=
sorry

-- Theorem for part (2)
theorem range_of_m (x m : ℝ) (h1 : x ∈ Set.Icc 0 (Real.pi / 6)) 
  (h2 : ∀ x ∈ Set.Icc 0 (Real.pi / 6), -4 ≤ f x m ∧ f x m ≤ 4) :
  m ∈ Set.Icc (-5) 2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intervals_of_increase_range_of_m_l370_37077


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vertical_asymptote_at_negative_five_l370_37065

-- Define the function as noncomputable
noncomputable def f (x : ℝ) : ℝ := (x^2 + 5*x + 6) / (x + 5)

-- Theorem statement
theorem vertical_asymptote_at_negative_five :
  ∃ (ε : ℝ), ε > 0 ∧ ∀ (δ : ℝ), 0 < δ ∧ δ < ε →
    (∀ (x : ℝ), 0 < |x - (-5)| ∧ |x - (-5)| < δ → |f x| > 1/ε) :=
by
  sorry

#check vertical_asymptote_at_negative_five

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vertical_asymptote_at_negative_five_l370_37065


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_perpendicular_l370_37047

noncomputable def f (x : ℝ) : ℝ := Real.exp x + (2*x - 5) / (x^2 + 1)

noncomputable def f' (x : ℝ) : ℝ := Real.exp x + (-2*x^2 + 10*x + 2) / (x^2 + 1)^2

theorem tangent_line_perpendicular (m : ℝ) : 
  (f' 0 = 3) →  -- The derivative of f at x = 0 is 3
  (∀ x y, x - m*y + 4 = 0 → y = -(1/m)*x - 4/m) → -- The equation of the perpendicular line
  (f' 0 * (-1/m) = -1) → -- Perpendicularity condition
  m = -3 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_perpendicular_l370_37047


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_phi_value_l370_37001

open Real

-- Define the function f
noncomputable def f (φ : Real) (x : Real) : Real := sin (2 * (x + π/8) + φ)

-- State the theorem
theorem phi_value (φ : Real) (h1 : 0 < φ) (h2 : φ < π) :
  (f φ 0 = 0) → φ = 3*π/4 := by
  intro h
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_phi_value_l370_37001


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_angle_measure_l370_37084

theorem quadrilateral_angle_measure (P Q R S : ℝ) : 
  P = 3 * Q ∧ P = 4 * R ∧ P = 6 * S ∧ P + Q + R + S = 360 → 
  abs (P - 206) < 1 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadrilateral_angle_measure_l370_37084


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_height_formula_l370_37000

/-- Regular triangular pyramid with given base side length and angle between slant height and lateral face -/
structure RegularTriangularPyramid where
  base_side : ℝ
  slant_angle : ℝ

/-- Height of a regular triangular pyramid -/
noncomputable def pyramid_height (p : RegularTriangularPyramid) : ℝ :=
  p.base_side * Real.sqrt 6 / 6

theorem pyramid_height_formula (a : ℝ) (h_pos : a > 0) :
  let p : RegularTriangularPyramid := ⟨a, π/4⟩
  pyramid_height p = a * Real.sqrt 6 / 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pyramid_height_formula_l370_37000


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_T_5_equals_2525_l370_37011

/-- Given a real number y such that y + 1/y = 5, Tₘ is defined as yᵐ + 1/yᵐ -/
noncomputable def T (y : ℝ) (m : ℕ) : ℝ := y^m + (1/y)^m

/-- The main theorem stating that T₅ = 2525 -/
theorem T_5_equals_2525 (y : ℝ) (h : y + 1/y = 5) : T y 5 = 2525 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_T_5_equals_2525_l370_37011


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bridge_length_calculation_l370_37048

/-- Calculates the length of a bridge given the walking speed and time to cross. -/
noncomputable def bridge_length (speed : ℝ) (time : ℝ) : ℝ :=
  speed * (time / 60)

/-- Theorem: A man walking at 5 km/hr crosses a bridge in 15 minutes. 
    The length of the bridge is 1250 meters. -/
theorem bridge_length_calculation :
  bridge_length 5 15 * 1000 = 1250 := by
  -- Unfold the definition of bridge_length
  unfold bridge_length
  -- Simplify the arithmetic
  simp [mul_assoc, mul_comm, mul_div_assoc]
  -- The proof is complete
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_bridge_length_calculation_l370_37048


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_3_sufficient_not_necessary_l370_37005

-- Define the condition for a line to be tangent to the circle
def is_tangent (k : ℝ) : Prop :=
  (k^2 + 1 = 4)

-- Define the given condition
noncomputable def given_k : ℝ := Real.sqrt 3

-- Theorem statement
theorem sqrt_3_sufficient_not_necessary :
  (is_tangent given_k) ∧ 
  (∃ k : ℝ, is_tangent k ∧ k ≠ given_k) := by
  sorry

#check sqrt_3_sufficient_not_necessary

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_3_sufficient_not_necessary_l370_37005


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_2α_value_l370_37079

theorem cos_2α_value (α : ℝ) (h1 : 0 < α ∧ α < π / 2) (h2 : Real.sin (α - π / 4) = 3 / 5) :
  Real.cos (2 * α) = -24 / 25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_2α_value_l370_37079


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_projection_l370_37094

noncomputable def vector_a : ℝ × ℝ := (Real.sqrt 3 / 2, 1 / 2)

theorem vector_projection (b : ℝ × ℝ) 
  (h1 : ‖b‖ = 2)
  (h2 : ‖2 • vector_a - b‖ = Real.sqrt 6) :
  let a := vector_a
  (a.1 * b.1 + a.2 * b.2 = 1 / 2) ∧ 
  ((a.1 * (a.1 * b.1 + a.2 * b.2) / (a.1^2 + a.2^2), 
    a.2 * (a.1 * b.1 + a.2 * b.2) / (a.1^2 + a.2^2)) = (Real.sqrt 3 / 4, 1 / 4)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_projection_l370_37094


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_small_factorable_substring_l370_37028

/-- A natural number is small factorable if it cannot be represented as the product
    of more than four integer factors, each greater than one. -/
def SmallFactorable (n : ℕ) : Prop :=
  ∀ (a b c d e : ℕ), a > 1 ∧ b > 1 ∧ c > 1 ∧ d > 1 ∧ e > 1 → n ≠ a * b * c * d * e

/-- Given seven consecutive three-digit numbers, returns all possible
    six-digit substrings formed by consecutive digits. -/
def SixDigitSubstrings (n : ℕ) : Set ℕ :=
  { m : ℕ | ∃ (i : ℕ), i ≥ 0 ∧ i ≤ 15 ∧ m = (n / 10^i) % 10^6 }

/-- The main theorem: given any seven consecutive three-digit numbers written consecutively,
    there exists a six-digit substring that is small factorable. -/
theorem exists_small_factorable_substring (n : ℕ) (h : n ≥ 100000000 ∧ n < 1000000000) :
  ∃ m ∈ SixDigitSubstrings n, SmallFactorable m := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_small_factorable_substring_l370_37028


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_is_odd_l370_37095

noncomputable def g (x : ℝ) : ℝ := (3^x - 1) / (3^x + 1)

theorem g_is_odd : ∀ x : ℝ, g (-x) = -g x := by
  intro x
  simp [g]
  -- The proof steps would go here, but we'll use sorry for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_is_odd_l370_37095


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rope_release_time_is_ten_minutes_l370_37062

/-- Represents the balloon's ascent and descent rates, and the highest elevation reached --/
structure BalloonData where
  ascent_rate : ℕ
  descent_rate : ℕ
  max_elevation : ℕ

/-- Calculates the time the rope was released for the first time --/
def rope_release_time (data : BalloonData) : ℕ :=
  let total_ascent_time : ℕ := 30
  let total_ascent : ℕ := data.ascent_rate * total_ascent_time
  let descent : ℕ := total_ascent - data.max_elevation
  descent / data.descent_rate

/-- Theorem stating that the rope release time is 10 minutes --/
theorem rope_release_time_is_ten_minutes 
  (data : BalloonData) 
  (h1 : data.ascent_rate = 50)
  (h2 : data.descent_rate = 10)
  (h3 : data.max_elevation = 1400) : 
  rope_release_time data = 10 := by
  sorry

#eval rope_release_time { ascent_rate := 50, descent_rate := 10, max_elevation := 1400 }

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rope_release_time_is_ten_minutes_l370_37062


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_cardinality_A_l370_37014

-- Define the function f
def f (x : ℝ) : ℝ := x^2

-- Define set B
def B : Set ℝ := {0, 1, 4}

-- Define set A
def A : Set ℝ := {x : ℝ | f x ∈ B}

-- Lemma to show A is finite
lemma A_finite : Set.Finite A := by sorry

-- Theorem statement
theorem max_cardinality_A : Finset.card (Set.Finite.toFinset A_finite) = 5 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_cardinality_A_l370_37014


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_p_val_6_l370_37082

/-- A monic quintic polynomial with specific values at x = 1, 2, 3, 4, 5 -/
def p : ℝ → ℝ := sorry

/-- p is a monic quintic polynomial -/
axiom p_monic_quintic : ∃ a b c d : ℝ, ∀ x, p x = x^5 + a*x^4 + b*x^3 + c*x^2 + d*x + (p 0)

/-- Values of p at x = 1, 2, 3, 4, 5 -/
axiom p_val_1 : p 1 = 2
axiom p_val_2 : p 2 = 5
axiom p_val_3 : p 3 = 10
axiom p_val_4 : p 4 = 17
axiom p_val_5 : p 5 = 26

/-- Theorem: p(6) = 163 -/
theorem p_val_6 : p 6 = 163 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_p_val_6_l370_37082


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_on_parabola_l370_37055

/-- The parabola y = -1/2x^2 -/
noncomputable def parabola (x : ℝ) : ℝ := -1/2 * x^2

/-- A point on the parabola -/
structure PointOnParabola where
  x : ℝ
  y : ℝ
  on_parabola : y = parabola x

/-- The distance between two points -/
noncomputable def distance (p q : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)

/-- Theorem: If P and Q are points on y = -1/2x^2 forming an equilateral triangle POQ with the origin,
    then the side length of POQ is 4√3 -/
theorem equilateral_triangle_on_parabola (P Q : PointOnParabola) :
  let O : ℝ × ℝ := (0, 0)
  distance (P.x, P.y) O = distance (Q.x, Q.y) O ∧
  distance (P.x, P.y) O = distance (P.x, P.y) (Q.x, Q.y) →
  distance (P.x, P.y) (Q.x, Q.y) = 4 * Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equilateral_triangle_on_parabola_l370_37055


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_odd_function_cos_symmetry_tan_cos_relation_l370_37030

-- Statement 1
theorem sin_odd_function (k : ℤ) : 
  ∀ x : ℝ, Real.sin (↑k * Real.pi - x) = -Real.sin (-(↑k * Real.pi - x)) :=
sorry

-- Statement 2
theorem cos_symmetry :
  ∀ x : ℝ, Real.cos (2 * (x + (-2 * Real.pi / 3)) + Real.pi / 3) = Real.cos (2 * ((-2 * Real.pi / 3) - x) + Real.pi / 3) :=
sorry

-- Statement 3
theorem tan_cos_relation :
  ∀ x : ℝ, Real.tan (Real.pi - x) = 2 → Real.cos x ^ 2 = 1/5 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_odd_function_cos_symmetry_tan_cos_relation_l370_37030


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_x_l370_37050

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if -1 ≤ x ∧ x ≤ 1 then
    3 * (1 - 2^x) / (2^x + 1)
  else
    -1/4 * (x^3 + 3*x)

-- State the theorem
theorem range_of_x :
  (∀ x m, m ∈ Set.Icc (-3) 2 → f (m*x - 1) + f x > 0) →
  {x : ℝ | f x = f x} ⊆ Set.Ioo (-1/2) (1/3) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_x_l370_37050


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_conversion_l370_37023

/-- Converts kilometers per hour to meters per second -/
noncomputable def kmph_to_ms (speed_kmph : ℝ) : ℝ :=
  speed_kmph * 1000 / 3600

theorem train_speed_conversion :
  kmph_to_ms 162 = 45 := by
  -- Unfold the definition of kmph_to_ms
  unfold kmph_to_ms
  -- Simplify the arithmetic expression
  simp [mul_div_assoc]
  -- Check that the equality holds
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_conversion_l370_37023


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_f_equals_four_plus_three_pi_over_twelve_l370_37041

noncomputable def f (x : ℝ) : ℝ :=
  if -1 ≤ x ∧ x ≤ 0 then (x + 1)^2
  else if 0 < x ∧ x ≤ 1 then Real.sqrt (1 - x^2)
  else 0

theorem integral_f_equals_four_plus_three_pi_over_twelve :
  ∫ x in Set.Icc (-1) 1, f x = (4 + 3 * Real.pi) / 12 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_f_equals_four_plus_three_pi_over_twelve_l370_37041


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_l_max_distance_line_l_min_area_l370_37046

/-- Line l passing through point P(1, 1) with equation (a+1)x + y - 2 - a = 0 -/
def line_l (a : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (a + 1) * p.1 + p.2 - 2 - a = 0}

/-- Point A -/
def point_A : ℝ × ℝ := (5, 3)

/-- Distance from a point to a line -/
noncomputable def distance_to_line (p : ℝ × ℝ) (l : Set (ℝ × ℝ)) : ℝ :=
  sorry

/-- Area of triangle OMN where M and N are intersections of line l with x and y axes -/
noncomputable def area_triangle_OMN (a : ℝ) : ℝ :=
  sorry

theorem line_l_max_distance :
  ∃ a : ℝ, ∀ b : ℝ, distance_to_line point_A (line_l a) ≥ distance_to_line point_A (line_l b) →
  line_l a = {p : ℝ × ℝ | 2 * p.1 + p.2 - 3 = 0} := by
  sorry

theorem line_l_min_area :
  ∃ a : ℝ, ∀ b : ℝ, area_triangle_OMN a ≤ area_triangle_OMN b →
  line_l a = {p : ℝ × ℝ | p.1 + p.2 - 2 = 0} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_l_max_distance_line_l_min_area_l370_37046


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_solution_sum_of_divisors_l370_37099

/-- Sum of positive divisors of n -/
def sum_of_divisors (n : ℕ) : ℕ := sorry

/-- Theorem: No integer i between 1 and 10000 satisfies f(i) = 1 + 2√i + i -/
theorem no_solution_sum_of_divisors :
  ¬∃ i : ℕ, 1 ≤ i ∧ i ≤ 10000 ∧ sum_of_divisors i = 1 + 2 * Nat.sqrt i + i := by
  sorry

#check no_solution_sum_of_divisors

end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_solution_sum_of_divisors_l370_37099


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_value_l370_37039

/-- The function f(x) = 4/(x-3) + x for x < 3 -/
noncomputable def f (x : ℝ) : ℝ := 4 / (x - 3) + x

/-- Theorem stating that the maximum value of f(x) for x < 3 is -1 -/
theorem f_max_value :
  (∀ x < 3, f x ≤ -1) ∧ (∃ x < 3, f x = -1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_value_l370_37039


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l370_37086

noncomputable def f (x : ℝ) : ℝ := 1 / Real.sqrt (Real.log (4 * x - 3) / Real.log 0.5)

theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = {x : ℝ | 3/4 < x ∧ x < 1} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l370_37086


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_magical_room_tiling_l370_37067

theorem magical_room_tiling (room_length room_width tile_length tile_width : ℝ) 
  (h_room_length : room_length = 8)
  (h_room_width : room_width = 12)
  (h_tile_length : tile_length = 1.5)
  (h_tile_width : tile_width = 2) :
  (room_length * room_width) / (tile_length * tile_width) = 32 := by
  sorry

#check magical_room_tiling

end NUMINAMATH_CALUDE_ERRORFEEDBACK_magical_room_tiling_l370_37067


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_johns_gym_routine_ratio_l370_37026

/-- Represents John's gym routine --/
structure GymRoutine where
  visits_per_week : ℕ
  weightlifting_time_per_visit : ℚ
  total_time_per_week : ℚ

/-- Calculates the ratio of warm-up and cardio time to weightlifting time --/
def warmup_cardio_to_weightlifting_ratio (routine : GymRoutine) : ℚ :=
  let total_weightlifting_time := routine.visits_per_week * routine.weightlifting_time_per_visit
  let total_warmup_cardio_time := routine.total_time_per_week - total_weightlifting_time
  let warmup_cardio_time_per_visit := total_warmup_cardio_time / routine.visits_per_week
  warmup_cardio_time_per_visit / routine.weightlifting_time_per_visit

theorem johns_gym_routine_ratio :
  let routine : GymRoutine := {
    visits_per_week := 3,
    weightlifting_time_per_visit := 1,
    total_time_per_week := 4
  }
  warmup_cardio_to_weightlifting_ratio routine = 1 / 3 := by
  sorry

#eval warmup_cardio_to_weightlifting_ratio {
  visits_per_week := 3,
  weightlifting_time_per_visit := 1,
  total_time_per_week := 4
}

end NUMINAMATH_CALUDE_ERRORFEEDBACK_johns_gym_routine_ratio_l370_37026


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_positive_solution_l370_37069

/-- Recursive definition of the left-hand side of the equation --/
noncomputable def leftSide (x : ℝ) : ℝ → ℝ
  | y => (x + y) ^ (1 / x)

/-- Recursive definition of the right-hand side of the equation --/
noncomputable def rightSide (x : ℝ) : ℝ → ℝ
  | y => (x * y) ^ (1 / x)

/-- The main theorem stating that 2 is the unique positive solution --/
theorem unique_positive_solution :
  ∃! (x : ℝ), x > 0 ∧ leftSide x (leftSide x (leftSide x (leftSide x 0))) = 
                     rightSide x (rightSide x (rightSide x (rightSide x 1))) ∧ 
               x = 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_positive_solution_l370_37069


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_period_and_g_max_l370_37044

noncomputable def f (x : ℝ) : ℝ := Real.sin (x + Real.pi/2) * Real.cos (x - Real.pi/3)

noncomputable def g (x : ℝ) : ℝ := 2 * Real.sin (2*x + Real.pi/6)

theorem f_period_and_g_max :
  (∃ (p : ℝ), p > 0 ∧ (∀ (x : ℝ), f (x + p) = f x) ∧
    (∀ (q : ℝ), q > 0 ∧ (∀ (x : ℝ), f (x + q) = f x) → p ≤ q)) ∧
  (∀ (x : ℝ), x ∈ Set.Icc 0 (Real.pi/3) → g x ≤ 2) ∧
  (∃ (x : ℝ), x ∈ Set.Icc 0 (Real.pi/3) ∧ g x = 2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_period_and_g_max_l370_37044


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_seventh_term_l370_37073

/-- Given a geometric sequence of positive integers where the first term is 3
    and the fifth term is 243, the seventh term is 2187. -/
theorem geometric_sequence_seventh_term :
  ∀ (a : ℕ → ℕ),
  (∀ n, a (n + 1) = a n * (a 1)) →  -- Geometric sequence property
  a 0 = 3 →                         -- First term is 3
  a 4 = 243 →                       -- Fifth term is 243
  a 6 = 2187                        -- Seventh term is 2187
:= by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_geometric_sequence_seventh_term_l370_37073


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_path_length_of_quarter_circle_rotation_l370_37052

noncomputable section

-- Define the quarter-circle
def quarter_circle (center : ℝ × ℝ) (radius : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - center.1)^2 + (p.2 - center.2)^2 = radius^2 ∧ 
               p.1 ≥ center.1 ∧ p.2 ≥ center.2}

-- Define the path length of a point during a full rotation
def path_length_full_rotation (radius : ℝ) : ℝ :=
  3 * (Real.pi / 2) * radius

-- Theorem statement
theorem path_length_of_quarter_circle_rotation 
  (B : ℝ × ℝ) -- Center of the quarter-circle
  (C : ℝ × ℝ) -- Point on the quarter-circle
  (h1 : C ∈ quarter_circle B (4 / Real.pi)) -- C is on the quarter-circle with radius 4/π
  (h2 : (C.1 - B.1)^2 + (C.2 - B.2)^2 = (4 / Real.pi)^2) -- BC = 4/π
  : path_length_full_rotation (4 / Real.pi) = 6 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_path_length_of_quarter_circle_rotation_l370_37052


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_height_l370_37022

/-- The height of the first pole in inches -/
noncomputable def h₁ : ℝ := 30

/-- The height of the second pole in inches -/
noncomputable def h₂ : ℝ := 50

/-- The distance between the poles in inches -/
noncomputable def d : ℝ := 150

/-- The slope of the line from the top of the first pole to the foot of the second pole -/
noncomputable def m₁ : ℝ := (0 - h₁) / d

/-- The slope of the line from the top of the second pole to the foot of the first pole -/
noncomputable def m₂ : ℝ := (0 - h₂) / (-d)

/-- The x-coordinate of the intersection point -/
noncomputable def x : ℝ := (h₁ - 0) / (m₂ - m₁)

/-- The y-coordinate (height) of the intersection point -/
noncomputable def y : ℝ := m₁ * x + h₁

theorem intersection_height : y = 18.75 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_height_l370_37022


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_growth_l370_37045

/-- The amount of money after compound interest is applied -/
def compound_interest (principal : ℝ) (rate : ℝ) (time : ℕ) : ℝ :=
  principal * (1 + rate) ^ time

/-- The problem statement -/
theorem investment_growth :
  let principal : ℝ := 8000
  let rate : ℝ := 0.08
  let time : ℕ := 3
  let result := compound_interest principal rate time
  ⌊result⌋₊ = 10078 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_growth_l370_37045


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_most_likely_event_l370_37090

def roll_die : Finset ℕ := {1, 2, 3, 4, 5, 6}

def P (A : Finset ℕ) : ℚ :=
  (A.card : ℚ) / (roll_die.card : ℚ)

theorem most_likely_event :
  let A := roll_die.filter (λ x => x > 2)
  let B := roll_die.filter (λ x => x = 4 ∨ x = 5)
  let C := roll_die.filter (λ x => x % 2 = 0)
  let D := roll_die.filter (λ x => x < 3)
  let E := roll_die.filter (λ x => x = 3)
  P A = 2/3 ∧
  P A > P B ∧
  P A > P C ∧
  P A > P D ∧
  P A > P E :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_most_likely_event_l370_37090


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_point_to_line_example_l370_37027

/-- The distance from a point to a line in 2D space -/
noncomputable def distance_point_to_line (x₀ y₀ a b c : ℝ) : ℝ :=
  |a * x₀ + b * y₀ + c| / Real.sqrt (a^2 + b^2)

/-- Theorem: The distance from the point (1, -2) to the line x - y = 1 is √2 -/
theorem distance_point_to_line_example : 
  distance_point_to_line 1 (-2) 1 (-1) 1 = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_point_to_line_example_l370_37027


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_eleven_is_110_l370_37032

/-- An arithmetic sequence with specific conditions -/
structure ArithmeticSequence where
  a : ℕ → ℚ
  is_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)
  third_term : a 3 = 4
  sum_fourth_ninth : a 4 + a 9 = 22

/-- The sum of the first n terms of an arithmetic sequence -/
def sum_n (seq : ArithmeticSequence) (n : ℕ) : ℚ :=
  (n : ℚ) * (seq.a 1 + seq.a n) / 2

/-- The main theorem -/
theorem sum_eleven_is_110 (seq : ArithmeticSequence) : sum_n seq 11 = 110 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_eleven_is_110_l370_37032


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nine_nines_cubed_zeros_l370_37013

/-- The number of trailing zeros in (10^9 - 1)^3 -/
def trailing_zeros : ℕ := 9

/-- The base number represented as 10^9 - 1 -/
def base_number : ℕ := 10^9 - 1

/-- Function to count trailing zeros -/
def count_trailing_zeros (n : ℕ) : ℕ :=
  if n = 0 then 0
  else if n % 10 = 0 then 1 + count_trailing_zeros (n / 10)
  else 0

theorem nine_nines_cubed_zeros : 
  count_trailing_zeros (base_number ^ 3) = trailing_zeros := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_nine_nines_cubed_zeros_l370_37013


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_final_milk_concentration_l370_37008

noncomputable def initial_mixture_volume : ℝ := 100
noncomputable def initial_milk_volume : ℝ := 36
noncomputable def removed_volume : ℝ := 50

noncomputable def milk_concentration_after_replacement (initial_milk : ℝ) (total_volume : ℝ) (removed_volume : ℝ) : ℝ :=
  (initial_milk - (initial_milk / total_volume) * removed_volume) / total_volume

theorem final_milk_concentration :
  let first_replacement := milk_concentration_after_replacement initial_milk_volume initial_mixture_volume removed_volume
  let second_replacement := milk_concentration_after_replacement (first_replacement * initial_mixture_volume) initial_mixture_volume removed_volume
  second_replacement = 0.09 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_final_milk_concentration_l370_37008


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_2012_l370_37012

def mySequence (n : ℕ+) : ℤ := (-1)^(n.val + 1 : ℕ) * n.val

theorem sequence_2012 : mySequence 2012 = -2012 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_2012_l370_37012


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_covering_inequality_example_covering_inequality_range_covering_inequality_system_l370_37092

-- Define covering inequality
def is_covering_inequality (f g : ℝ → Prop) : Prop :=
  ∀ x, g x → f x

-- Theorem 1
theorem covering_inequality_example :
  is_covering_inequality (λ x ↦ x < -1) (λ x ↦ x < -3) := by sorry

-- Theorem 2
theorem covering_inequality_range (m : ℝ) :
  is_covering_inequality (λ x ↦ x < -2) (λ x ↦ -x + 4*m > 0) →
  m ≤ -1/2 := by sorry

-- Theorem 3
theorem covering_inequality_system (a : ℝ) :
  is_covering_inequality 
    (λ x ↦ 1 ≤ x ∧ x ≤ 6)
    (λ x ↦ 2*a - x > 1 ∧ 2*x + 5 > 3*a) →
  7/3 ≤ a ∧ a ≤ 7/2 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_covering_inequality_example_covering_inequality_range_covering_inequality_system_l370_37092


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_power_inequality_l370_37007

theorem power_inequality (a b : ℝ) 
  (h1 : (1/3 : ℝ) < (1/3 : ℝ)^b) 
  (h2 : (1/3 : ℝ)^b < (1/3 : ℝ)^a) 
  (h3 : (1/3 : ℝ)^a < 1) : 
  a^b < a^a ∧ a^a < b^a := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_power_inequality_l370_37007


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l370_37042

-- Define the vectors and function
noncomputable def a (x : ℝ) : ℝ × ℝ := (Real.sin (2 * x), 2 * Real.cos x)
noncomputable def b (x : ℝ) : ℝ × ℝ := (Real.sqrt 3, Real.cos x)
noncomputable def f (x : ℝ) : ℝ := (a x).1 * (b x).1 + (a x).2 * (b x).2 - 1

-- Define the theorem
theorem problem_solution :
  (∃ (p : ℝ), p > 0 ∧ ∀ (x : ℝ), f (x + p) = f x ∧ ∀ (q : ℝ), q > 0 ∧ (∀ (x : ℝ), f (x + q) = f x) → p ≤ q) ∧
  (∃ (m : ℝ), ∀ (x : ℝ), f x ≥ m ∧ ∃ (y : ℝ), f y = m) ∧
  (∀ (A : ℝ), f (A / 4) = Real.sqrt 3 →
    ∀ (a b c : ℝ), a = 2 * Real.sqrt 13 ∧ b = 8 →
      (c = 2 ∨ c = 6)) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l370_37042


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_series_divergence_1_l370_37091

theorem series_divergence_1 (a : ℕ → ℝ) (h : ∀ n, a n = 2 * n / (n^2 + 1)) :
  ¬ (Summable a) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_series_divergence_1_l370_37091


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_equals_expected_l370_37024

/-- The probability density function for the given random variable X -/
noncomputable def P (x : ℝ) : ℝ :=
  if x > 0 then Real.exp (-x) else 0

/-- The probability that X falls within the interval (1,3) -/
noncomputable def probability : ℝ :=
  ∫ x in Set.Ioo 1 3, P x

theorem probability_equals_expected : probability = (Real.exp 2 - 1) / Real.exp 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_equals_expected_l370_37024


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_max_sum_l370_37057

noncomputable def arithmeticSequence (a₁ : ℝ) (d : ℝ) : ℕ → ℝ := λ n => a₁ + (n - 1 : ℝ) * d

noncomputable def sumOfArithmeticSequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ :=
  n * (2 * a₁ + (n - 1 : ℝ) * d) / 2

theorem arithmetic_sequence_max_sum :
  let a₁ := 25
  let a₄ := 16
  let d := (a₄ - a₁) / 3
  let aₙ := arithmeticSequence a₁ d
  let Sₙ := λ n => sumOfArithmeticSequence a₁ d n
  ∃ (n : ℕ), n = 9 ∧ Sₙ n = 117 ∧ ∀ (m : ℕ), Sₙ m ≤ Sₙ n :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_arithmetic_sequence_max_sum_l370_37057


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_two_implies_b_over_a_sqrt_three_l370_37043

/-- The eccentricity of a hyperbola with equation x²/a² - y²/b² = 1 -/
noncomputable def eccentricity (a b : ℝ) : ℝ := Real.sqrt (1 + b^2 / a^2)

theorem hyperbola_eccentricity_two_implies_b_over_a_sqrt_three 
  (a b : ℝ) (ha : a > 0) (hb : b > 0) (he : eccentricity a b = 2) :
  b / a = Real.sqrt 3 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hyperbola_eccentricity_two_implies_b_over_a_sqrt_three_l370_37043


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pi_sixth_sufficient_not_necessary_for_sin_half_l370_37020

theorem pi_sixth_sufficient_not_necessary_for_sin_half :
  (∀ θ : ℝ, θ = π / 6 → Real.sin θ = 1 / 2) ∧ 
  (∃ θ : ℝ, θ ≠ π / 6 ∧ Real.sin θ = 1 / 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pi_sixth_sufficient_not_necessary_for_sin_half_l370_37020


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_is_lowest_degree_polynomial_l370_37003

/-- The function f(x) as defined in the problem -/
def f (x : ℝ) : ℝ := x^3 + 3*x^2 + 5*x + 7

/-- The function g(x) that we need to prove is correct -/
def g (x : ℝ) : ℝ := 12*x^2 - 19*x + 25

/-- Theorem stating that g(x) is the polynomial of lowest degree satisfying the given conditions -/
theorem g_is_lowest_degree_polynomial :
  (f 3 = g 3) ∧
  (f (3 - Real.sqrt 3) = g (3 - Real.sqrt 3)) ∧
  (f (3 + Real.sqrt 3) = g (3 + Real.sqrt 3)) ∧
  (∀ h : Polynomial ℝ,
    ((h.eval 3 = f 3) ∧
     (h.eval (3 - Real.sqrt 3) = f (3 - Real.sqrt 3)) ∧
     (h.eval (3 + Real.sqrt 3) = f (3 + Real.sqrt 3))) →
    (h.degree ≥ 2)) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_is_lowest_degree_polynomial_l370_37003


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solutions_count_3x_plus_2y_equals_1001_l370_37037

theorem solutions_count_3x_plus_2y_equals_1001 :
  let solution_set := {(x, y) : ℕ × ℕ | 3 * x + 2 * y = 1001 ∧ x > 0 ∧ y > 0}
  Finset.card (Finset.filter (λ (p : ℕ × ℕ) => 3 * p.1 + 2 * p.2 = 1001 ∧ p.1 > 0 ∧ p.2 > 0) (Finset.range 1002 ×ˢ Finset.range 1002)) = 167 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solutions_count_3x_plus_2y_equals_1001_l370_37037


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_t_l370_37061

/-- The function representing the y-coordinate of the midpoint of MN -/
noncomputable def t (m : ℝ) : ℝ := (1/2) * ((2 - m) * Real.exp m + m * Real.exp (-m))

/-- The theorem stating the maximum value of t for m > 0 -/
theorem max_value_of_t :
  ∃ (max_t : ℝ), max_t = (1/2) * (Real.exp 1 + Real.exp (-1)) ∧
  ∀ (m : ℝ), m > 0 → t m ≤ max_t := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_t_l370_37061


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_evaluation_l370_37063

theorem expression_evaluation : ∃ ε > 0, 
  abs (((0.82^3) - (0.1^3)) / ((0.82^2) + 0.082 + (0.1^2) + Real.log 0.82 / Real.log 5 - Real.sin (π/4))^2 - 126.229) < ε ∧ ε < 0.001 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_expression_evaluation_l370_37063


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_property_l370_37004

-- Define the triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real
  angleSum : A + B + C = Real.pi
  sideOpposite : a > 0 ∧ b > 0 ∧ c > 0

-- Define the given conditions
def givenConditions (t : Triangle) : Prop :=
  t.a > t.b ∧ t.b = 2 ∧ 9 * (Real.sin (t.A - t.B))^2 + (Real.cos t.C)^2 = 1

-- State the theorem
theorem triangle_property (t : Triangle) (h : givenConditions t) :
  3 * t.a^2 - t.c^2 = 12 ∧
  (∃ (S : Real), S = (1/2) * t.b * t.c * Real.sin t.A ∧
    ∀ (S' : Real), S' = (1/2) * t.b * t.c * Real.sin t.A → S' ≤ 3) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_property_l370_37004


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_value_at_one_l370_37088

-- Define the function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a * Real.log x + x

-- State the theorem
theorem extreme_value_at_one (a : ℝ) :
  (∃ (ε : ℝ), ε > 0 ∧ ∀ x, x > 0 ∧ |x - 1| < ε → f a x ≤ f a 1) →
  a = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_extreme_value_at_one_l370_37088


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_non_straight_figure_with_straight_projections_l370_37051

-- Define the concept of a spatial figure
structure SpatialFigure where
  -- Add necessary properties here
  mk :: -- Constructor

-- Define the concept of a plane
structure Plane where
  -- Add necessary properties here
  mk ::

-- Define the concept of a point
structure Point where
  -- Add necessary properties here
  mk ::

-- Define the concept of projection
def projection (figure : SpatialFigure) (plane : Plane) : Set Point :=
  sorry

-- Define what it means for two planes to intersect
def intersecting (p1 p2 : Plane) : Prop :=
  sorry

-- Define what it means for a set of points to form a straight line
def isStraightLine (s : Set Point) : Prop :=
  sorry

-- Define what it means for a spatial figure to be a straight line
def isStraightLineFigure (figure : SpatialFigure) : Prop :=
  sorry

-- Theorem statement
theorem exists_non_straight_figure_with_straight_projections :
  ∃ (figure : SpatialFigure) (p1 p2 : Plane),
    intersecting p1 p2 ∧
    isStraightLine (projection figure p1) ∧
    isStraightLine (projection figure p2) ∧
    ¬(isStraightLineFigure figure) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_exists_non_straight_figure_with_straight_projections_l370_37051


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_budget_allocation_theorem_transportation_percentage_l370_37096

/-- Represents the budget allocation of a company -/
structure BudgetAllocation where
  salaries : ℚ
  research_development : ℚ
  utilities : ℚ
  supplies : ℚ
  transportation : ℚ
  equipment : ℚ

/-- Theorem stating the correct budget allocation given the conditions -/
theorem budget_allocation_theorem (b : BudgetAllocation) :
  b.salaries = 60 ∧
  b.research_development = 9 ∧
  b.utilities = 5 ∧
  b.supplies = 2 ∧
  b.transportation = 20 ∧
  b.salaries + b.research_development + b.utilities + b.supplies + b.transportation + b.equipment = 100 →
  b.equipment = 4 := by
  sorry

/-- Function to calculate the percentage of a circle given the degrees -/
def degrees_to_percentage (degrees : ℚ) : ℚ :=
  degrees * 100 / 360

/-- Theorem stating that 72 degrees represents 20% of the budget -/
theorem transportation_percentage :
  degrees_to_percentage 72 = 20 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_budget_allocation_theorem_transportation_percentage_l370_37096


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_in_interval_l370_37021

noncomputable def f (x : ℝ) : ℝ := Real.log ((3 * x / 2) - (2 / x))

theorem zero_in_interval :
  (∀ x y, 0 < x ∧ x < y → f x < f y) →
  f 1 < 0 →
  f 2 > 0 →
  ∃! x, 1 < x ∧ x < 2 ∧ f x = 0 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_zero_in_interval_l370_37021


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_no_intersection_l370_37010

-- Define the line and circle equations
def line_eq (x y : ℝ) : Prop := 3 * x + 4 * y = 12
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 = 4

-- Theorem statement
theorem no_intersection :
  ¬∃ (x y : ℝ), line_eq x y ∧ circle_eq x y := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_no_intersection_l370_37010


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_area_triangles_l370_37025

/-- Helper function to calculate the area of a triangle given its vertices -/
def area_triangle (A B C : ℚ × ℚ) : ℚ :=
  (1/2) * abs ((B.1 - A.1) * (C.2 - A.2) - (C.1 - A.1) * (B.2 - A.2))

/-- Given a triangle PQR with vertices P(-1,2), Q(5,4), and R(4,-3),
    if S(m,n) is the point inside the triangle such that triangles PQS, PRS, and QRS have equal areas,
    then 10m + n = 83/3 -/
theorem equal_area_triangles (m n : ℚ) :
  let P : ℚ × ℚ := (-1, 2)
  let Q : ℚ × ℚ := (5, 4)
  let R : ℚ × ℚ := (4, -3)
  let S : ℚ × ℚ := (m, n)
  (S.1 > -1 ∧ S.1 < 5 ∧ S.2 > -3 ∧ S.2 < 4) →  -- S is inside the triangle
  (area_triangle P Q S = area_triangle P R S) ∧
  (area_triangle P R S = area_triangle Q R S) →
  10 * m + n = 83 / 3 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_area_triangles_l370_37025


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_A_equals_negative_b_over_a_l370_37093

-- Define the variables a and b
variable (a b : ℝ)

-- Define the equation
noncomputable def A : ℝ → ℝ → ℝ := λ a b =>
  Real.tan (7 * Real.pi / 4 + (1 / 2) * Real.arccos (2 * a / b)) + 
  Real.tan (7 * Real.pi / 4 - (1 / 2) * Real.arccos (2 * a / b))

-- State the theorem
theorem A_equals_negative_b_over_a (a b : ℝ) (h : a ≠ 0) : A a b = -b / a :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_A_equals_negative_b_over_a_l370_37093


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_count_l370_37018

theorem unique_solution_count :
  ∃! (S : Finset ℤ), 
    (∀ b ∈ S, b > 0) ∧ 
    (∀ b ∈ S, ∀ x : ℤ, x > 0 → 
      ((3 * x > 4 * x - 5 ∧ 5 * x - b > -9) ↔ x = 2)) ∧
    S.card = 5 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_unique_solution_count_l370_37018


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_fraction_equality_l370_37071

open Complex

theorem complex_fraction_equality (x : ℂ) (h : abs x ≠ 1) :
  let z₁ := 1 - x^2 + x * I * (Real.sqrt 3)
  let z₂ := 1 - x^2 - x * I * (Real.sqrt 3)
  (z₁ * z₂) / (1 - x^6) = 1 / (1 - x^2) := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_fraction_equality_l370_37071


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_kats_average_training_time_l370_37040

/-- Represents Kat's weekly training schedule --/
structure TrainingSchedule where
  strength_hours : ℚ
  strength_sessions : ℚ
  strength_missed : ℚ
  boxing_hours : ℚ
  boxing_sessions : ℚ
  boxing_missed : ℚ
  cardio_hours : ℚ
  cardio_sessions : ℚ
  flexibility_hours : ℚ
  flexibility_sessions : ℚ
  interval_hours : ℚ
  interval_sessions : ℚ

/-- Calculates the average weekly training hours --/
def average_weekly_hours (schedule : TrainingSchedule) : ℚ :=
  let strength_avg := (schedule.strength_sessions - schedule.strength_missed / 2) * schedule.strength_hours
  let boxing_avg := (schedule.boxing_sessions - schedule.boxing_missed / 2) * schedule.boxing_hours
  let cardio_total := schedule.cardio_hours * schedule.cardio_sessions
  let flexibility_total := schedule.flexibility_hours * schedule.flexibility_sessions
  let interval_total := schedule.interval_hours * schedule.interval_sessions
  strength_avg + boxing_avg + cardio_total + flexibility_total + interval_total

/-- Kat's actual training schedule --/
def kats_schedule : TrainingSchedule :=
  { strength_hours := 1
  , strength_sessions := 3
  , strength_missed := 1
  , boxing_hours := 3/2
  , boxing_sessions := 4
  , boxing_missed := 1
  , cardio_hours := 1/2
  , cardio_sessions := 2
  , flexibility_hours := 3/4
  , flexibility_sessions := 1
  , interval_hours := 5/4
  , interval_sessions := 1
  }

theorem kats_average_training_time :
  average_weekly_hours kats_schedule = 43/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_kats_average_training_time_l370_37040


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_james_book_spending_fraction_l370_37056

noncomputable def weekly_allowance : ℚ := 10
def weeks_saved : ℕ := 4
noncomputable def video_game_fraction : ℚ := 1/2
noncomputable def money_left : ℚ := 15

theorem james_book_spending_fraction :
  let total_savings := weekly_allowance * weeks_saved
  let after_video_game := total_savings * (1 - video_game_fraction)
  let book_cost := after_video_game - money_left
  book_cost / after_video_game = 1/4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_james_book_spending_fraction_l370_37056


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_vertex_focus_distance_l370_37031

/-- Represents a parabola with equation x^2 = 4y -/
structure Parabola where
  equation : ∀ x y : ℝ, x^2 = 4*y

/-- The vertex of the parabola -/
def vertex : ℝ × ℝ := (0, 4)

/-- The focus of the parabola -/
def focus : ℝ × ℝ := (0, 1)

/-- Distance between two points in 2D space -/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem parabola_vertex_focus_distance :
  distance vertex focus = 3 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_vertex_focus_distance_l370_37031


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_magnitude_of_vector_sum_l370_37016

/-- Given two plane vectors a and b, prove that |a + 2b| = √26 -/
theorem magnitude_of_vector_sum (a b : ℝ × ℝ) 
  (h1 : Real.cos (Real.arccos (-Real.sqrt 2 / 2)) = -Real.sqrt 2 / 2)  -- angle between a and b is 3π/4
  (h2 : Real.sqrt (a.1^2 + a.2^2) = Real.sqrt 2)                       -- |a| = √2
  (h3 : Real.sqrt (b.1^2 + b.2^2) = 3)                                 -- |b| = 3
  : Real.sqrt ((a.1 + 2*b.1)^2 + (a.2 + 2*b.2)^2) = Real.sqrt 26 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_magnitude_of_vector_sum_l370_37016


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_circles_arrangement_max_circles_arrangement_proof_l370_37081

-- Define a circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the given unit circle K
def K : Circle := { center := (0, 0), radius := 1 }

-- Define a function to check if two circles are tangent
def are_tangent (c1 c2 : Circle) : Prop :=
  let (x1, y1) := c1.center
  let (x2, y2) := c2.center
  (x2 - x1)^2 + (y2 - y1)^2 = (c1.radius + c2.radius)^2

-- Define a function to check if two circles intersect
def intersect (c1 c2 : Circle) : Prop :=
  let (x1, y1) := c1.center
  let (x2, y2) := c2.center
  (x2 - x1)^2 + (y2 - y1)^2 < (c1.radius + c2.radius)^2

-- Define a function to check if a circle contains the center of another circle
def contains_center (c1 c2 : Circle) : Prop :=
  let (x1, y1) := c1.center
  let (x2, y2) := c2.center
  (x2 - x1)^2 + (y2 - y1)^2 < c1.radius^2

-- Main theorem
theorem max_circles_arrangement (circles : List Circle) : Prop :=
  (∀ c, c ∈ circles → c.radius = 1) ∧
  (∀ c, c ∈ circles → are_tangent K c) ∧
  (∀ c1 c2, c1 ∈ circles → c2 ∈ circles → c1 ≠ c2 → ¬(intersect c1 c2)) ∧
  (∀ c1 c2, c1 ∈ circles → c2 ∈ circles → c1 ≠ c2 → ¬(contains_center c1 c2)) ∧
  (∀ c, c ∈ circles → ¬(contains_center c K)) ∧
  (∀ c, c ∈ circles → intersect K c) →
  circles.length ≤ 18

-- Proof
theorem max_circles_arrangement_proof : ∃ circles : List Circle, 
  max_circles_arrangement circles ∧ circles.length = 18 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_circles_arrangement_max_circles_arrangement_proof_l370_37081


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_k_minimizes_distance_l370_37089

/-- The value of k that minimizes the sum of distances AC + BC -/
noncomputable def optimal_k : ℝ := 15 / 7

/-- Point A coordinates -/
def A : ℝ × ℝ := (5, 5)

/-- Point B coordinates -/
def B : ℝ × ℝ := (2, 1)

/-- Point C coordinates -/
def C (k : ℝ) : ℝ × ℝ := (0, k)

/-- Distance between two points -/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

/-- Sum of distances AC + BC -/
noncomputable def total_distance (k : ℝ) : ℝ :=
  distance A (C k) + distance B (C k)

/-- Theorem stating that optimal_k minimizes the total distance -/
theorem optimal_k_minimizes_distance :
  ∀ k : ℝ, total_distance optimal_k ≤ total_distance k := by
  sorry

#check optimal_k_minimizes_distance

end NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_k_minimizes_distance_l370_37089


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sum_first_three_l370_37002

theorem max_sum_first_three (x : Fin 7 → ℕ) 
  (h_order : ∀ i j, i < j → x i < x j)
  (h_sum : (Finset.sum Finset.univ x) = 159) :
  (∃ y : Fin 7 → ℕ, 
    (∀ i j, i < j → y i < y j) ∧ 
    ((Finset.sum Finset.univ y) = 159) ∧ 
    (y 0 + y 1 + y 2 = 61) ∧
    (∀ z : Fin 7 → ℕ, 
      (∀ i j, i < j → z i < z j) → 
      ((Finset.sum Finset.univ z) = 159) → 
      (z 0 + z 1 + z 2 ≤ 61))) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_sum_first_three_l370_37002


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_radius_is_eight_l370_37072

/-- Represents a circle with an intersecting diameter and chord -/
structure IntersectingCircle where
  /-- Length of the first part of the chord -/
  chord_part1 : ℝ
  /-- Length of the second part of the chord -/
  chord_part2 : ℝ
  /-- Length of one part of the diameter -/
  diameter_part : ℝ

/-- Calculates the radius of the circle given the lengths of chord parts and one diameter part -/
noncomputable def calculateRadius (circle : IntersectingCircle) : ℝ :=
  (circle.diameter_part + (circle.chord_part1 * circle.chord_part2 / circle.diameter_part)) / 2

theorem radius_is_eight (circle : IntersectingCircle) 
    (h1 : circle.chord_part1 = 3)
    (h2 : circle.chord_part2 = 5)
    (h3 : circle.diameter_part = 1) :
  calculateRadius circle = 8 := by
  sorry

-- Remove the #eval statement as it's not computable

end NUMINAMATH_CALUDE_ERRORFEEDBACK_radius_is_eight_l370_37072


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_solution_tan_sec_equation_l370_37070

open Real

theorem smallest_positive_solution_tan_sec_equation :
  ∃ x : ℝ, x > 0 ∧ 
    (∀ y : ℝ, y > 0 → tan (3*y) + tan (5*y) = 1 / cos (5*y) + 1 → x ≤ y) ∧
    tan (3*x) + tan (5*x) = 1 / cos (5*x) + 1 ∧
    x = π/4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_positive_solution_tan_sec_equation_l370_37070


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_right_angle_condition_l370_37006

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    if a*cos(B) + b*cos(A) = c*sin(A), then angle A is a right angle. -/
theorem triangle_right_angle_condition (a b c A B C : ℝ) : 
  0 < A → A < π →
  0 < B → B < π →
  0 < C → C < π →
  A + B + C = π →
  a > 0 → b > 0 → c > 0 →
  a * Real.cos B + b * Real.cos A = c * Real.sin A →
  A = π / 2 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_right_angle_condition_l370_37006


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_semicircle_inscribed_triangle_radius_l370_37075

/-- A triangle inscribed in a semicircle with two vertices on the semicircle 
    and the third on the diameter has a radius equal to half the longest side. -/
theorem semicircle_inscribed_triangle_radius 
  (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) 
  (htri : c^2 = a^2 + b^2) : 
  let R := c / 2
  ∃ (center : ℝ × ℝ) (A B C : ℝ × ℝ),
    (‖A - center‖ = R) ∧ 
    (‖B - center‖ = R) ∧
    (C.1 - center.1 = 0 ∨ C.1 - center.1 = 2*R) ∧
    (‖B - C‖ = a) ∧ 
    (‖A - C‖ = b) ∧ 
    (‖A - B‖ = c) :=
by
  sorry

#check semicircle_inscribed_triangle_radius

end NUMINAMATH_CALUDE_ERRORFEEDBACK_semicircle_inscribed_triangle_radius_l370_37075


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_of_point_M_l370_37049

-- Define the circle and points
def Circle : Type := ℝ × ℝ → Prop
def Point : Type := ℝ × ℝ

-- Define the given points and conditions
def isOnCircle (c : Circle) (p : Point) : Prop := sorry
def isOnSegment (p q r : Point) : Prop := sorry
def distance (p q : Point) : ℝ := sorry

-- Define the theorem
theorem locus_of_point_M (c : Circle) (A B P M C D : Point) :
  isOnCircle c A ∧ 
  isOnCircle c B ∧ 
  isOnCircle c P ∧
  ((isOnSegment P A M ∧ distance A M = distance M P + distance P B) ∨
   (isOnSegment P B M ∧ distance A P + distance M P = distance P B)) ∧
  (distance C D = 2 * (distance A C)) ∧
  (distance A C = distance C B) ∧
  (distance A D = distance D B) →
  (isOnCircle (λ p ↦ distance p A = distance p D) M) ∨
  (isOnCircle (λ p ↦ distance p A = distance p C) M) ∨
  (isOnCircle (λ p ↦ distance p B = distance p C) M) ∨
  (isOnCircle (λ p ↦ distance p B = distance p D) M) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_locus_of_point_M_l370_37049


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_power_function_l370_37098

noncomputable def f (x : ℝ) : ℝ := x^(-2 : ℤ)

theorem domain_of_power_function :
  {x : ℝ | f x ≠ 0} = {x : ℝ | x ≠ 0} :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_power_function_l370_37098


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_intercepts_not_three_and_negative_three_l370_37064

-- Define the quadratic function
noncomputable def f (x : ℝ) : ℝ := -1/2 * (x - 1)^2 + 2

-- Theorem statement
theorem x_intercepts_not_three_and_negative_three :
  ¬(f 3 = 0 ∧ f (-3) = 0) :=
by
  -- We use 'sorry' to skip the proof for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_intercepts_not_three_and_negative_three_l370_37064


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_discount_percentage_calculation_l370_37060

/-- Calculate the discount percentage given cost price, markup percentage, and selling price -/
theorem discount_percentage_calculation
  (cost_price : ℝ)
  (markup_percentage : ℝ)
  (selling_price : ℝ)
  (h1 : cost_price = 540)
  (h2 : markup_percentage = 15)
  (h3 : selling_price = 460) :
  let marked_price := cost_price * (1 + markup_percentage / 100)
  let discount := marked_price - selling_price
  let discount_percentage := (discount / marked_price) * 100
  abs (discount_percentage - 25.92) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_discount_percentage_calculation_l370_37060
