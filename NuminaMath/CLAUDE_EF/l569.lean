import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_PA_l569_56951

/-- The curve C -/
def curve_C (x y : ℝ) : Prop := x^2 / 4 + y^2 / 9 = 1

/-- The line l -/
def line_l (x y : ℝ) : Prop := 2 * x + y - 6 = 0

/-- Angle between the drawn line and line l -/
noncomputable def angle : ℝ := Real.pi / 6  -- 30 degrees in radians

/-- The maximum distance between P and A -/
theorem max_distance_PA (P : ℝ × ℝ) (A : ℝ × ℝ) :
  curve_C P.1 P.2 →
  line_l A.1 A.2 →
  ∃ (θ : ℝ), Real.cos θ * (A.1 - P.1) + Real.sin θ * (A.2 - P.2) = 
             Real.cos angle * (A.1 - P.1) + Real.sin angle * (A.2 - P.2) →
  ∀ (Q : ℝ × ℝ), curve_C Q.1 Q.2 →
    ∀ (B : ℝ × ℝ), line_l B.1 B.2 →
      ∃ (φ : ℝ), Real.cos φ * (B.1 - Q.1) + Real.sin φ * (B.2 - Q.2) = 
                  Real.cos angle * (B.1 - Q.1) + Real.sin angle * (B.2 - Q.2) →
        Real.sqrt ((A.1 - P.1)^2 + (A.2 - P.2)^2) ≤ 22 * Real.sqrt 5 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_PA_l569_56951


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_ab_value_l569_56968

theorem triangle_ab_value (A B C a b c : ℝ) : 
  0 < A ∧ 0 < B ∧ 0 < C ∧ 
  A + B + C = π ∧
  0 < a ∧ 0 < b ∧ 0 < c ∧
  Real.sin A * Real.sin B + Real.sin C * Real.sin C = Real.sin A * Real.sin A + Real.sin B * Real.sin B ∧
  (1/2) * a * b * Real.sin C = Real.sqrt 3 →
  a * b = 4 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_ab_value_l569_56968


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_yoann_anagrams_l569_56965

/-- The number of distinct anagrams of a word with 5 letters, where one letter is repeated twice -/
def distinct_anagrams (word : String) : ℕ :=
  if word.length = 5 ∧ (word.toList.count (word.get ⟨0⟩) = 2 ∨
                        word.toList.count (word.get ⟨1⟩) = 2 ∨
                        word.toList.count (word.get ⟨2⟩) = 2 ∨
                        word.toList.count (word.get ⟨3⟩) = 2 ∨
                        word.toList.count (word.get ⟨4⟩) = 2)
  then 60
  else 0

theorem yoann_anagrams :
  distinct_anagrams "YOANN" = 60 :=
by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_yoann_anagrams_l569_56965


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_not_monotone_increasing_l569_56977

noncomputable def f (x : ℝ) : ℝ := 3 * Real.sin (2 * x - Real.pi / 3)

theorem f_not_monotone_increasing :
  ¬ (∀ x y : ℝ, -Real.pi/12 < x ∧ x < y ∧ y < 7*Real.pi/12 → f x < f y) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_not_monotone_increasing_l569_56977


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integer_pairs_satisfying_equation_l569_56939

theorem integer_pairs_satisfying_equation :
  ∃! (s : Finset (ℤ × ℤ)), 
    (∀ (x y : ℤ), (x, y) ∈ s ↔ x^2020 + y^2 = 2*y) ∧ 
    Finset.card s = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integer_pairs_satisfying_equation_l569_56939


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_histogram_rectangle_area_histogram_total_area_l569_56946

/-- Represents a group in a frequency distribution histogram -/
structure HistogramGroup where
  interval : ℝ
  rate : ℝ

/-- The area of a rectangle in a frequency distribution histogram -/
noncomputable def rectangleArea (group : HistogramGroup) : ℝ :=
  group.interval * (group.rate / group.interval)

/-- Theorem: The area of each rectangle is equal to the rate of the corresponding group -/
theorem histogram_rectangle_area (group : HistogramGroup) :
  rectangleArea group = group.rate := by
  sorry

/-- The sum of all rectangle areas in a histogram is 1 -/
theorem histogram_total_area (groups : List HistogramGroup) :
  (groups.map rectangleArea).sum = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_histogram_rectangle_area_histogram_total_area_l569_56946


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mark_sprint_time_l569_56943

/-- The time Mark sprinted, given his distance and speed -/
noncomputable def sprint_time (distance : ℝ) (speed : ℝ) : ℝ :=
  distance / speed

/-- Theorem: Mark sprinted for 4 hours -/
theorem mark_sprint_time : sprint_time 24 6 = 4 := by
  -- Unfold the definition of sprint_time
  unfold sprint_time
  -- Simplify the division
  norm_num


end NUMINAMATH_CALUDE_ERRORFEEDBACK_mark_sprint_time_l569_56943


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_calories_in_350g_lemonade_l569_56992

/-- Represents the composition of Marco's lemonade --/
structure Lemonade where
  lemon_juice : ℚ
  sugar : ℚ
  water : ℚ
  mint : ℚ

/-- Calculates the total weight of the lemonade --/
def total_weight (l : Lemonade) : ℚ :=
  l.lemon_juice + l.sugar + l.water + l.mint

/-- Calculates the total calories in the lemonade --/
def total_calories (l : Lemonade) : ℚ :=
  (l.lemon_juice * 30 / 100) + (l.sugar * 400 / 100) + (l.mint * 7 / 10)

/-- Marco's lemonade recipe --/
def marcos_lemonade : Lemonade where
  lemon_juice := 150
  sugar := 200
  water := 300
  mint := 50

/-- Theorem: 350g of Marco's lemonade contains 440 calories --/
theorem calories_in_350g_lemonade :
  (350 / total_weight marcos_lemonade) * total_calories marcos_lemonade = 440 := by
  -- Proof steps would go here
  sorry

#eval (350 / total_weight marcos_lemonade) * total_calories marcos_lemonade

end NUMINAMATH_CALUDE_ERRORFEEDBACK_calories_in_350g_lemonade_l569_56992


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_polynomial_b_value_l569_56967

/-- A cubic polynomial function -/
def g (a b c d : ℝ) (x : ℝ) : ℝ := a * x^3 + b * x^2 + c * x + d

theorem cubic_polynomial_b_value
  (a b c d : ℝ) :
  g a b c d (-2) = 0 →
  g a b c d 1 = 0 →
  g a b c d 0 = 3 →
  (deriv (g a b c d)) 1 = 0 →
  b = 0 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cubic_polynomial_b_value_l569_56967


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_2sqrt2_l569_56980

-- Define the functions
noncomputable def f_A (x : ℝ) : ℝ := Real.sqrt (x^2 + 3) + 2 / Real.sqrt (x^2 + 3)
noncomputable def f_B (x : ℝ) : ℝ := Real.sin x + 2 / Real.sin x
noncomputable def f_C (x : ℝ) : ℝ := |x| + 2 / |x|
noncomputable def f_D (x : ℝ) : ℝ := Real.log x / Real.log 10 + 2 / (Real.log x / Real.log 10)

-- State the theorem
theorem min_value_2sqrt2 :
  (∀ x, f_A x > 2 * Real.sqrt 2) ∧
  (∀ x, 0 < x → x < π → f_B x > 2 * Real.sqrt 2) ∧
  (∃ x, f_C x = 2 * Real.sqrt 2) ∧
  (∀ x, x ≠ 0 → f_C x ≥ 2 * Real.sqrt 2) ∧
  (∃ x, x > 0 ∧ x ≠ 1 ∧ f_D x < 2 * Real.sqrt 2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_2sqrt2_l569_56980


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_weight_NH4Cl_ideal_gas_l569_56998

noncomputable section

-- Constants
def R : ℝ := 0.0821  -- Ideal gas constant in L·atm/(mol·K)
def T : ℝ := 1500    -- Temperature in Kelvin
def P_mmHg : ℝ := 1200  -- Pressure in mmHg
def n : ℝ := 8       -- Number of moles
def molar_mass_NH4Cl : ℝ := 53.50  -- Molar mass of NH4Cl in g/mol

-- Conversion factor
def mmHg_to_atm : ℝ := 1 / 760

-- Theorem
theorem weight_NH4Cl_ideal_gas : 
  let P_atm := P_mmHg * mmHg_to_atm
  let V := n * R * T / P_atm
  n * molar_mass_NH4Cl = 428 := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_weight_NH4Cl_ideal_gas_l569_56998


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l569_56988

-- Define the functions
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := (m + 1)^2 * x^(m^2 - m - 4)
def g (n : ℝ) (x : ℝ) : ℝ := 2 * x + n
noncomputable def F (m k : ℝ) (x : ℝ) : ℝ := f m x - k * x + (1 - k) * (1 + k)

-- State the theorem
theorem problem_solution :
  (∀ x > 0, Monotone (f m)) →
  m = -2 ∧
  (∀ x ∈ Set.Icc (-1) 3, g n x ∈ Set.image (f (-2)) (Set.Icc (-1) 3)) →
  n ∈ Set.Icc 2 3 ∧
  (∃ x₀ ∈ Set.Icc 0 2, ∀ x ∈ Set.Icc 0 2, F (-2) k x₀ ≤ F (-2) k x) →
  F (-2) k x₀ = -2 →
  k = -Real.sqrt 3 ∨ k = 2 * Real.sqrt 15 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_problem_solution_l569_56988


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_lloyd_worked_ten_point_five_hours_l569_56979

/-- Calculates the number of hours worked given the parameters of Lloyd's work schedule and earnings --/
noncomputable def hours_worked (regular_hours : ℝ) (regular_rate : ℝ) (overtime_multiplier : ℝ) (total_earnings : ℝ) : ℝ :=
  let regular_earnings := regular_hours * regular_rate
  let overtime_rate := regular_rate * overtime_multiplier
  let overtime_earnings := total_earnings - regular_earnings
  let overtime_hours := overtime_earnings / overtime_rate
  regular_hours + overtime_hours

/-- Theorem stating that Lloyd worked 10.5 hours given the specific conditions --/
theorem lloyd_worked_ten_point_five_hours :
  hours_worked 7.5 4.5 2.5 67.5 = 10.5 := by
  sorry

-- Remove the #eval statement as it's not computable
-- #eval hours_worked 7.5 4.5 2.5 67.5

end NUMINAMATH_CALUDE_ERRORFEEDBACK_lloyd_worked_ten_point_five_hours_l569_56979


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_on_interval_l569_56934

-- Define the function f(x) = ln|x-1|
noncomputable def f (x : ℝ) : ℝ := Real.log (abs (x - 1))

-- Theorem statement
theorem f_increasing_on_interval :
  StrictMonoOn f (Set.Ioi 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_increasing_on_interval_l569_56934


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l569_56991

noncomputable def f (x : ℝ) : ℝ := Real.sqrt (x + 2) / (x - 2)

theorem domain_of_f : 
  {x : ℝ | ∃ y, f x = y} = {x : ℝ | x ≥ -2 ∧ x ≠ 2} :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_domain_of_f_l569_56991


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_AB_equals_n_times_diameter_l569_56970

-- Define the original circle
structure OriginalCircle where
  center : ℝ × ℝ
  radius : ℝ

-- Define a point on the diameter AB
structure PointP where
  x : ℝ
  y : ℝ

-- Define the primary circles
structure PrimaryCircle where
  center : ℝ × ℝ
  radius : ℝ

-- Define the iteratively constructed circles
structure IterativeCircle where
  center : ℝ × ℝ
  radius : ℝ
  n : ℕ

-- Provide an instance of Inhabited for IterativeCircle
instance : Inhabited IterativeCircle where
  default := { center := (0, 0), radius := 1, n := 0 }

-- Define the construction process
def constructCircles (original : OriginalCircle) (p : PointP) (primary1 primary2 : PrimaryCircle) : 
  ℕ → IterativeCircle → IterativeCircle :=
  sorry

-- Helper functions (not implemented, just for context)
noncomputable def distance_to_line : (ℝ × ℝ) → (ℝ → ℝ → ℝ) → ℝ :=
  sorry

noncomputable def line_AB : OriginalCircle → (ℝ → ℝ → ℝ) :=
  sorry

-- Theorem statement
theorem distance_to_AB_equals_n_times_diameter 
  (original : OriginalCircle) (p : PointP) (primary1 primary2 : PrimaryCircle) (n : ℕ) :
  let nthCircle := constructCircles original p primary1 primary2 n default
  distance_to_line nthCircle.center (line_AB original) = n * (2 * nthCircle.radius) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_to_AB_equals_n_times_diameter_l569_56970


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_max_min_real_values_l569_56996

theorem sum_of_max_min_real_values (a b : ℂ) 
  (h1 : a^2 + b^2 = 7)
  (h2 : a^3 + b^3 = 10) : 
  ∃ (m n : ℝ), 
    (∀ x : ℝ, x = Complex.re (a + b) → x ≤ m ∧ x ≥ n) ∧ 
    (∃ y z : ℝ, y = Complex.re (a + b) ∧ y = m ∧ z = Complex.re (a + b) ∧ z = n) ∧
    m + n = -1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_of_max_min_real_values_l569_56996


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fixed_point_of_f_l569_56914

-- Define the function f(x) = a^(x-1) + 2
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.rpow a (x - 1) + 2

-- Theorem statement
theorem fixed_point_of_f (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  f a 1 = 3 ∧ ∀ x : ℝ, f a x = x → x = 1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fixed_point_of_f_l569_56914


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_type_characterization_l569_56923

/-- Defines the type of a triangle --/
inductive TriangleType
  | Acute
  | Right
  | Obtuse

/-- Determines the type of a triangle given its sides and circumradius --/
noncomputable def triangleType (a b c R : ℝ) : TriangleType :=
  let expr := a^2 + b^2 + c^2 - 8*R^2
  if expr > 0 then TriangleType.Acute
  else if expr = 0 then TriangleType.Right
  else TriangleType.Obtuse

/-- Theorem stating the relationship between triangle type and the expression --/
theorem triangle_type_characterization 
  (a b c R : ℝ) 
  (h_triangle : a > 0 ∧ b > 0 ∧ c > 0)
  (h_circumradius : R > 0) :
  (triangleType a b c R = TriangleType.Acute ↔ a^2 + b^2 + c^2 - 8*R^2 > 0) ∧
  (triangleType a b c R = TriangleType.Right ↔ a^2 + b^2 + c^2 - 8*R^2 = 0) ∧
  (triangleType a b c R = TriangleType.Obtuse ↔ a^2 + b^2 + c^2 - 8*R^2 < 0) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_type_characterization_l569_56923


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_x_greater_than_y_l569_56901

theorem x_greater_than_y (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a ≠ b) : 
  a^3 + b^3 > a^2 * b + a * b^2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_x_greater_than_y_l569_56901


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_accommodation_arrangements_l569_56919

/-- Represents the number of adult men -/
def num_adults : ℕ := 4

/-- Represents the number of little boys -/
def num_boys : ℕ := 2

/-- Represents the capacity of the triple room -/
def triple_room_capacity : ℕ := 3

/-- Represents the capacity of the double room -/
def double_room_capacity : ℕ := 2

/-- Represents the capacity of the single room -/
def single_room_capacity : ℕ := 1

/-- Represents the total number of rooms -/
def total_rooms : ℕ := 3

/-- Represents a room type -/
inductive Room
| Triple
| Double
| Single

/-- Function to get the number of boys in a room -/
def num_boys_in (room : Room) : ℕ := sorry

/-- Function to get the number of adults in a room -/
def num_adults_in (room : Room) : ℕ := sorry

/-- Function to get the total number of people in a room -/
def num_people_in (room : Room) : ℕ := num_boys_in room + num_adults_in room

/-- Represents the constraint that boys cannot stay alone -/
def boys_not_alone : Prop := ∀ room : Room, (num_boys_in room > 0) → (num_adults_in room > 0)

/-- Represents the constraint that all rooms must be occupied -/
def all_rooms_occupied : Prop := ∀ room : Room, num_people_in room > 0

/-- Calculates the number of different accommodation arrangements -/
def num_arrangements : ℕ := sorry

theorem accommodation_arrangements :
  boys_not_alone ∧ all_rooms_occupied → num_arrangements = 36 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_accommodation_arrangements_l569_56919


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_only_EventA_is_random_l569_56952

-- Define the events as propositions
def EventA : Prop := ∃ x : ℕ, x > 0 ∧ x < 100 -- Representing a random lottery outcome
def EventB : Prop := True -- Always true, representing a certain event
def EventC : Prop := False -- Always false, representing an impossible event
def EventD : Prop := False -- Always false, representing an impossible event

-- Define what it means for an event to be random
def isRandomEvent (e : Prop) : Prop := ¬(e ↔ True) ∧ ¬(e ↔ False)

-- Theorem statement
theorem only_EventA_is_random :
  isRandomEvent EventA ∧
  ¬isRandomEvent EventB ∧
  ¬isRandomEvent EventC ∧
  ¬isRandomEvent EventD :=
by
  sorry -- Skipping the proof for now


end NUMINAMATH_CALUDE_ERRORFEEDBACK_only_EventA_is_random_l569_56952


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_dot_product_l569_56995

/-- Ellipse E with equation x^2/4 + y^2/2 = 1 -/
def Ellipse (x y : ℝ) : Prop := x^2/4 + y^2/2 = 1

/-- Point M on the line x = 2 -/
def M (y : ℝ) : ℝ × ℝ := (2, y)

/-- Point P is the intersection of OM and the ellipse E -/
noncomputable def P (y : ℝ) : ℝ × ℝ := 
  let m := y/2
  let x := (2*m^2 - 4)/(m^2 + 2)
  let y := (4*m)/(m^2 + 2)
  (x, y)

/-- The dot product of OM and OP is constant -/
theorem constant_dot_product (y : ℝ) : 
  let m := M y
  let p := P y
  (m.1 * p.1 + m.2 * p.2 : ℝ) = 4 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_constant_dot_product_l569_56995


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_bill_main_project_time_l569_56913

structure DaySchedule where
  workHours : ℕ
  napTime : ℕ
  breakTime : ℕ
  smallTaskTime : ℕ

def mainProjectTime (schedule : DaySchedule) : ℕ :=
  (schedule.workHours - schedule.napTime - schedule.breakTime - schedule.smallTaskTime).max 0

def totalMainProjectTime (schedules : List DaySchedule) : ℕ :=
  (schedules.map mainProjectTime).sum

theorem bill_main_project_time :
  let day1 : DaySchedule := ⟨10, 3, 2, 0⟩
  let day2 : DaySchedule := ⟨8, 5, 2, 3⟩
  let day3 : DaySchedule := ⟨12, 3, 3, 6⟩
  let day4 : DaySchedule := ⟨6, 3, 1, 0⟩
  let schedules := [day1, day2, day3, day4]
  totalMainProjectTime schedules = 8 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_bill_main_project_time_l569_56913


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_third_intersection_point_l569_56915

-- Define the circle
def my_circle (x y : ℝ) : Prop :=
  (x - 5)^2 + y^2 = 25

-- State the theorem
theorem third_intersection_point : 
  ∃ (x : ℝ), x ≠ 0 ∧ x ≠ 10 ∧ my_circle x 0 ∧ x = 10 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_third_intersection_point_l569_56915


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_symmetry_implies_a_equals_3_l569_56962

/-- The function f(x) = (a-x)/(x-a-1) -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (a - x) / (x - a - 1)

/-- The inverse function of f -/
noncomputable def f_inv (a : ℝ) : ℝ → ℝ := Function.invFun (f a)

/-- The property that the graph of f_inv is symmetric with respect to (-1,4) -/
def inverse_symmetric (a : ℝ) : Prop :=
  ∀ x y, f_inv a x = y ↔ f_inv a (8 - x) = 8 - y

theorem inverse_symmetry_implies_a_equals_3 :
  ∀ a : ℝ, inverse_symmetric a → a = 3 := by
  sorry

#check inverse_symmetry_implies_a_equals_3

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_symmetry_implies_a_equals_3_l569_56962


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_P0_l569_56945

/-- A monic quadratic polynomial -/
def MonicQuadratic (d e : ℤ) : ℤ → ℤ := λ x ↦ x^2 + d*x + e

theorem min_value_of_P0 :
  ∃ min_e : ℤ, 
    min_e > 0 ∧ 
    (∀ d e a : ℤ, 
      let P := MonicQuadratic d e
      (a ≠ 20 ∧ a ≠ 22) →
      (a * P a = 20 * P 20) →
      (20 * P 20 = 22 * P 22) →
      P 0 ≥ min_e) ∧
    (∀ e' : ℤ, e' > 0 ∧ e' < min_e → 
      ¬∃ d' a' : ℤ,
        let P' := MonicQuadratic d' e'
        (a' ≠ 20 ∧ a' ≠ 22) ∧
        (a' * P' a' = 20 * P' 20) ∧ 
        (20 * P' 20 = 22 * P' 22)) ∧
    min_e = 20 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_P0_l569_56945


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_special_triangle_angles_l569_56921

/-- A triangle with special properties -/
structure SpecialTriangle where
  -- The sides of the triangle
  a : ℝ
  b : ℝ
  c : ℝ
  -- a is the shortest side, c is the longest side
  a_shortest : a ≤ b ∧ a ≤ c
  c_longest : c ≥ a ∧ c ≥ b
  -- The distance between circumcenter and orthocenter
  oh_distance : ℝ
  -- This distance is half the longest side and equal to the shortest side
  oh_property : oh_distance = c / 2 ∧ oh_distance = a
  -- The angles of the triangle in radians
  angle_A : ℝ
  angle_B : ℝ
  angle_C : ℝ
  -- The angles form a triangle
  angle_sum : angle_A + angle_B + angle_C = Real.pi

/-- The main theorem about the special triangle -/
theorem special_triangle_angles (t : SpecialTriangle) :
  t.angle_A = Real.pi/2 ∧ t.angle_B = Real.pi/3 ∧ t.angle_C = Real.pi/6 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_special_triangle_angles_l569_56921


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_horizontal_asymptote_of_f_l569_56978

/-- The rational function for which we want to find the horizontal asymptote -/
noncomputable def f (x : ℝ) : ℝ :=
  (15 * x^4 + 3 * x^3 + 7 * x^2 + 6 * x + 2) / (5 * x^4 + x^3 + 4 * x^2 + 2 * x + 1)

/-- Theorem stating that the horizontal asymptote of f occurs at y = 3 -/
theorem horizontal_asymptote_of_f :
  ∀ ε > 0, ∃ M, ∀ x, |x| > M → |f x - 3| < ε :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_horizontal_asymptote_of_f_l569_56978


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_picasso_prints_consecutive_probability_l569_56937

/-- The probability of 4 Picasso prints being placed consecutively when 12 pieces of art
    (including the 4 Picasso prints) are hung in a random order. -/
theorem picasso_prints_consecutive_probability (total_pieces : ℕ) (picasso_prints : ℕ) 
  (h1 : total_pieces = 12) (h2 : picasso_prints = 4) : 
  (Nat.factorial (total_pieces - picasso_prints + 1) * Nat.factorial picasso_prints) / 
  Nat.factorial total_pieces = 1 / 55 := by
  sorry

#eval (Nat.factorial 9 * Nat.factorial 4) / Nat.factorial 12

end NUMINAMATH_CALUDE_ERRORFEEDBACK_picasso_prints_consecutive_probability_l569_56937


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l569_56963

-- Define the triangle ABC
def Triangle (a b c : ℝ) := true

-- State the theorem
theorem triangle_problem (a b c : ℝ) (A B C : ℝ) :
  Triangle a b c →
  b = 2 →
  Real.cos C = 3/4 →
  (1/2) * a * b * Real.sin C = Real.sqrt 7/4 →
  (a = 1 ∧ Real.sin (2*A) = 5*Real.sqrt 7/16) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_problem_l569_56963


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_perfect_square_l569_56926

def tiles : Finset ℕ := Finset.range 10
def die : Finset ℕ := Finset.range 8

def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, n = m * m

-- We need to make this function computable
def is_perfect_square_bool (n : ℕ) : Bool :=
  (Nat.sqrt n) ^ 2 == n

theorem probability_perfect_square :
  (Finset.filter (λ (pair : ℕ × ℕ) => is_perfect_square_bool ((pair.1 + 1) * (pair.2 + 1))) (tiles.product die)).card /
  (tiles.card * die.card : ℚ) = 3 / 20 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_perfect_square_l569_56926


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_valid_order_l569_56976

def cost : ℕ := 1994

def does_not_contain_digits (n : ℕ) (digits : List ℕ) : Prop :=
  ∀ d ∈ digits, ¬ (n.repr.any (· == d.repr.front))

def valid_order (n : ℕ) : Prop :=
  does_not_contain_digits (cost * n) [0, 7, 8, 9]

theorem smallest_valid_order :
  (∀ m : ℕ, m < 56 → ¬ valid_order m) ∧ valid_order 56 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_valid_order_l569_56976


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_M_N_l569_56966

def M : Set ℤ := {-2, -1, 0, 1, 2}

def N : Set ℤ := {x : ℤ | (x : ℝ)^2 - (x : ℝ) - 6 ≥ 0}

theorem intersection_M_N : M ∩ N = {-2} := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_M_N_l569_56966


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_factorial_product_perfect_square_l569_56924

def is_perfect_square (n : ℕ) : Prop :=
  ∃ m : ℕ, m * m = n

def factorial_product (n : ℕ) : ℕ :=
  (List.range (2 * n)).map (fun i => Nat.factorial (i + 1)) |>.prod

theorem factorial_product_perfect_square (n : ℕ) :
  is_perfect_square (factorial_product n / Nat.factorial (n + 1)) ↔
  (∃ k : ℕ, n = 4 * k * (k + 1)) ∨ (∃ k : ℕ, n = 2 * k * k - 1) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_factorial_product_perfect_square_l569_56924


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_graph_transformation_l569_56903

-- Define the original function
noncomputable def f (x : ℝ) : ℝ := 3^x

-- Define the transformed function
noncomputable def g (x : ℝ) : ℝ := 9 * 3^x + 5

-- Theorem statement
theorem graph_transformation :
  ∀ x : ℝ, g x = f (x + 2) + 5 :=
by
  intro x
  simp [f, g]
  ring_nf
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_graph_transformation_l569_56903


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_2alpha_plus_beta_and_beta_l569_56958

theorem sin_2alpha_plus_beta_and_beta (α β : Real) 
  (h1 : 0 < α ∧ α < Real.pi/2) 
  (h2 : 0 < β ∧ β < Real.pi/2)
  (h3 : Real.sin α = (4 * Real.sqrt 3) / 7)
  (h4 : Real.cos (α + β) = -11/14) : 
  Real.sin (2*α + β) = -(39 * Real.sqrt 3) / 98 ∧ β = Real.pi/3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_2alpha_plus_beta_and_beta_l569_56958


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_base_10_to_base_6_256_l569_56999

/-- Converts a number from base 10 to base 6 --/
def toBase6 (n : ℕ) : ℕ :=
  let rec aux : ℕ → ℕ → ℕ
  | 0, acc => acc
  | m+1, acc => aux (m / 6) (acc * 10 + m % 6)
  aux n 0

theorem base_10_to_base_6_256 :
  toBase6 256 = 704 := by
  sorry

#eval toBase6 256

end NUMINAMATH_CALUDE_ERRORFEEDBACK_base_10_to_base_6_256_l569_56999


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_for_triangle_area_l569_56957

noncomputable def triangle_area (n : ℕ) : ℝ :=
  (1 / 2) * ((2 * n^2 - n - 12)^2 + 12)

theorem smallest_n_for_triangle_area : 
  (∀ k : ℕ, k > 0 ∧ k < 5 → triangle_area k ≤ 1000) ∧ 
  triangle_area 5 > 1000 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_smallest_n_for_triangle_area_l569_56957


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_approximation_l569_56925

/-- The probability of success in a single trial -/
noncomputable def p : ℝ := 0.6

/-- The number of trials in each experiment -/
def n : ℕ := 3

/-- The number of successes we're interested in -/
def k : ℕ := 2

/-- The total number of experiments conducted -/
def total_experiments : ℕ := 20

/-- The number of experiments with exactly k successes -/
def successful_experiments : ℕ := 10

/-- The theoretical probability of exactly k successes in n trials -/
noncomputable def theoretical_prob : ℝ := Nat.choose n k * p ^ k * (1 - p) ^ (n - k)

/-- The empirical probability based on the experiments -/
noncomputable def empirical_prob : ℝ := successful_experiments / total_experiments

/-- Theorem stating that the empirical probability is approximately equal to the theoretical probability -/
theorem probability_approximation : 
  abs (empirical_prob - theoretical_prob) < 0.01 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_approximation_l569_56925


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cart_wheel_circumferences_l569_56986

/-- The circumference of the front wheels of a cart -/
def front_circumference : ℝ := sorry

/-- The circumference of the rear wheels of a cart -/
def rear_circumference : ℝ := sorry

/-- The distance traveled by the cart -/
def distance : ℝ := 120

/-- The difference in revolutions between front and rear wheels -/
def revolution_difference : ℝ := 6

/-- The difference in revolutions after increasing wheel sizes -/
def new_revolution_difference : ℝ := 4

/-- Theorem stating the circumferences of the cart wheels -/
theorem cart_wheel_circumferences :
  (distance / front_circumference - distance / rear_circumference = revolution_difference) ∧
  (4/5 * (distance / front_circumference) - 5/6 * (distance / rear_circumference) = new_revolution_difference) →
  front_circumference = 4 ∧ rear_circumference = 5 := by
  sorry

#check cart_wheel_circumferences

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cart_wheel_circumferences_l569_56986


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_power_eight_minus_sine_power_eight_l569_56997

theorem cosine_power_eight_minus_sine_power_eight (α : ℝ) (m : ℝ) 
  (h : Real.cos (2 * α) = m) : 
  Real.cos α ^ 8 - Real.sin α ^ 8 = m * (1 + m^2) / 2 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_power_eight_minus_sine_power_eight_l569_56997


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_colored_lights_at_9_35_20_l569_56974

/-- Represents a time with hours, minutes, and seconds -/
structure Time where
  hours : ℕ
  minutes : ℕ
  seconds : ℕ

/-- Calculates the angle of the hour hand at a given time -/
noncomputable def hourHandAngle (t : Time) : ℝ :=
  (t.hours % 12 : ℝ) * 30 + (t.minutes : ℝ) * 0.5 + (t.seconds : ℝ) * (1 / 120)

/-- Calculates the angle of the minute hand at a given time -/
noncomputable def minuteHandAngle (t : Time) : ℝ :=
  (t.minutes : ℝ) * 6 + (t.seconds : ℝ) * 0.1

/-- Calculates the acute angle between the hour and minute hands -/
noncomputable def acuteAngleBetweenHands (t : Time) : ℝ :=
  min (abs (hourHandAngle t - minuteHandAngle t)) (360 - abs (hourHandAngle t - minuteHandAngle t))

/-- Counts the number of colored lights within the acute angle -/
noncomputable def coloredLightsCount (angle : ℝ) : ℕ :=
  Int.toNat ⌊angle / 6⌋

/-- Theorem: At 9:35:20 PM, there are 12 colored lights within the acute angle formed by the clock hands -/
theorem colored_lights_at_9_35_20 :
  coloredLightsCount (acuteAngleBetweenHands ⟨21, 35, 20⟩) = 12 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_colored_lights_at_9_35_20_l569_56974


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_theorem_l569_56955

-- Define the triangle ABC
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  a : ℝ
  b : ℝ
  c : ℝ
  D : ℝ × ℝ -- Midpoint of AB

-- Define the properties of the triangle
def triangle_properties (t : Triangle) : Prop :=
  -- √3 * b * cos(A) - a * sin(B) = 0
  Real.sqrt 3 * t.b * Real.cos t.A - t.a * Real.sin t.B = 0 ∧
  -- D is the midpoint of AB (implied by the problem)
  -- AC = 2
  2 = Real.sqrt (t.a^2 + t.b^2 - 2*t.a*t.b*Real.cos t.C) ∧
  -- CD = 2√3
  2 * Real.sqrt 3 = Real.sqrt ((t.a/2)^2 + t.c^2 - t.a*t.c*Real.cos t.B)

-- Theorem statement
theorem triangle_theorem (t : Triangle) 
  (h : triangle_properties t) : 
  t.A = Real.pi / 3 ∧ t.a = 2 * Real.sqrt 13 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_theorem_l569_56955


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_sphere_cone_ratio_l569_56940

/-- A truncated right circular cone with an inscribed sphere -/
structure TruncatedConeWithSphere where
  R : ℝ  -- radius of the larger base
  r : ℝ  -- radius of the smaller base
  s : ℝ  -- radius of the inscribed sphere
  h : ℝ  -- height of the truncated cone

/-- The volume of a sphere -/
noncomputable def sphereVolume (radius : ℝ) : ℝ := (4 / 3) * Real.pi * radius^3

/-- The volume of a truncated cone -/
noncomputable def truncatedConeVolume (c : TruncatedConeWithSphere) : ℝ :=
  (1 / 3) * Real.pi * c.h * (c.R^2 + c.r^2 + c.R * c.r)

/-- Theorem: If a sphere is inscribed in a truncated right circular cone and 
    the volume of the truncated cone is three times that of the sphere, 
    then the ratio of the radius of the bottom base to the radius of the top base is 2 -/
theorem inscribed_sphere_cone_ratio 
  (c : TruncatedConeWithSphere) 
  (h1 : c.s = Real.sqrt (c.R * c.r))  -- sphere is inscribed
  (h2 : truncatedConeVolume c = 3 * sphereVolume c.s)  -- volume relation
  : c.R / c.r = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_sphere_cone_ratio_l569_56940


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fifty_million_seven_zeros_l569_56916

/-- Represents a number in the decimal system -/
def DecimalNumber := List Nat

/-- Converts a DecimalNumber to its numeric value -/
def to_value (n : DecimalNumber) : Nat :=
  n.foldl (fun acc d => acc * 10 + d) 0

/-- Checks if a DecimalNumber represents fifty million seven -/
def is_fifty_million_seven (n : DecimalNumber) : Prop :=
  to_value n = 50000007

/-- Checks if a DecimalNumber starts with 5 and ends with 7 -/
def starts_5_ends_7 (n : DecimalNumber) : Prop :=
  n.head? = some 5 ∧ n.getLast? = some 7

/-- Counts the number of zeros between the first and last digit -/
def count_zeros (n : DecimalNumber) : Nat :=
  (n.tail.dropLast.filter (· = 0)).length

theorem fifty_million_seven_zeros :
  ∀ n : DecimalNumber,
    is_fifty_million_seven n ∧ starts_5_ends_7 n →
    count_zeros n = 7 :=
by
  sorry

#eval to_value [5, 0, 0, 0, 0, 0, 0, 0, 7]

end NUMINAMATH_CALUDE_ERRORFEEDBACK_fifty_million_seven_zeros_l569_56916


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_theorem_l569_56933

noncomputable def given_ellipse (x y : ℝ) : Prop := x^2/4 + y^2/9 = 1

noncomputable def foci : ℝ × ℝ := (0, Real.sqrt 5)

def point_P : ℝ × ℝ := (2, -3)

noncomputable def eccentricity : ℝ := Real.sqrt 5 / 5

def minor_axis_length : ℝ := 4

def ellipse1 (x y : ℝ) : Prop := x^2/10 + y^2/15 = 1
def ellipse2 (x y : ℝ) : Prop := x^2/5 + y^2/4 = 1
def ellipse3 (x y : ℝ) : Prop := x^2/4 + y^2/5 = 1

theorem ellipse_theorem :
  ∃ (e : (ℝ → ℝ → Prop)), 
    (∀ (x y : ℝ), e x y → (x - foci.1)^2 + (y - foci.2)^2 + (x - foci.1)^2 + (y + foci.2)^2 = 4 * (x^2 + y^2)) ∧
    e point_P.1 point_P.2 ∧
    (∃ (a b : ℝ), a > b ∧ a^2 - b^2 = (eccentricity * a)^2 ∧ b = minor_axis_length / 2) ∧
    (e = ellipse1 ∨ e = ellipse2 ∨ e = ellipse3) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_ellipse_theorem_l569_56933


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_probability_binomial_l569_56912

-- Define the binomial distribution
def binomial_distribution (n : ℕ) (p : ℝ) (k : ℕ) : ℝ := 
  (Nat.choose n k) * p^k * (1 - p)^(n - k)

-- Define the problem statement
theorem max_probability_binomial (X : ℕ → ℝ) :
  (∀ k, X k = binomial_distribution 8 (1/2) k) →
  (∃ k : ℕ, k ≤ 8 ∧ ∀ j : ℕ, j ≤ 8 → X k ≥ X j) →
  (∃ k : ℕ, k = 4 ∧ ∀ j : ℕ, j ≤ 8 → X k ≥ X j) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_probability_binomial_l569_56912


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_point_l569_56927

-- Define the circle
def is_on_circle (x y : ℝ) : Prop := (x - 2)^2 + (y + 3)^2 = 2

-- Define the fixed point A
def A : ℝ × ℝ := (0, -5)

-- Define the distance function
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

-- Theorem statement
theorem max_distance_point :
  ∀ P : ℝ × ℝ, is_on_circle P.1 P.2 →
    (∀ Q : ℝ × ℝ, is_on_circle Q.1 Q.2 → distance A Q ≤ distance A P) →
    P = (3, -2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_distance_point_l569_56927


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_prime_divisor_of_24_squared_plus_75_squared_l569_56971

theorem largest_prime_divisor_of_24_squared_plus_75_squared : 
  (∃ p : ℕ, Nat.Prime p ∧ p ∣ (24^2 + 75^2) ∧ ∀ q : ℕ, Nat.Prime q → q ∣ (24^2 + 75^2) → q ≤ p) ∧ 
  Nat.Prime 53 ∧ 
  53 ∣ (24^2 + 75^2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_largest_prime_divisor_of_24_squared_plus_75_squared_l569_56971


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_inequality_solution_l569_56989

-- Define the quadratic function
def f (a x : ℝ) : ℝ := a * x^2 - (a + 1) * x + 1

-- Define the solution set
def solution_set (a : ℝ) : Set ℝ :=
  if a = 0 then Set.Ioi 1
  else if a < 0 then Set.Iio (1/a) ∪ Set.Ioi 1
  else if 0 < a ∧ a < 1 then Set.Ioo 1 (1/a)
  else if a > 1 then Set.Ioo (1/a) 1
  else ∅

-- Theorem statement
theorem quadratic_inequality_solution (a : ℝ) :
  {x : ℝ | f a x < 0} = solution_set a :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_inequality_solution_l569_56989


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_difference_formula_l569_56935

theorem cos_difference_formula (α β : ℝ) 
  (h1 : Real.cos α + Real.cos β = 1/2) 
  (h2 : Real.sin α + Real.sin β = Real.sqrt 3 / 2) : 
  Real.cos (α - β) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cos_difference_formula_l569_56935


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_sum_theorem_l569_56993

theorem sqrt_sum_theorem : Real.sqrt 25 + (64 : ℝ) ^ (1/3) - Real.sqrt ((-2)^2) + Real.sqrt 4 = 9 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sqrt_sum_theorem_l569_56993


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_kindergarten_cards_l569_56985

/-- Represents the number of children who can form a specific word -/
structure WordCount where
  mama : ℕ
  nyana : ℕ
  manya : ℕ

/-- The main theorem about kindergarten card distribution -/
theorem kindergarten_cards (wc : WordCount) :
  wc.mama + wc.nyana - wc.manya = 10 :=
by sorry

/-- The specific instance from the problem -/
def problem_instance : WordCount :=
  { mama := 20
  , nyana := 30
  , manya := 40 }

/-- Verify the theorem for the problem instance -/
example : kindergarten_cards problem_instance = rfl :=
by rfl

end NUMINAMATH_CALUDE_ERRORFEEDBACK_kindergarten_cards_l569_56985


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_new_trip_duration_l569_56960

/-- Calculates the new trip duration given initial conditions and changes -/
theorem new_trip_duration 
  (initial_duration : ℝ) 
  (initial_speed : ℝ) 
  (new_speed : ℝ) 
  (rest_stop_duration : ℝ) 
  (h1 : initial_duration = 6)
  (h2 : initial_speed = 80)
  (h3 : new_speed = 40)
  (h4 : rest_stop_duration = 0.5) :
  (initial_duration * initial_speed) / new_speed + rest_stop_duration = 12.50 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_new_trip_duration_l569_56960


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_correct_l569_56936

/-- The maximum value of f(x) = -x^2 - 2x on [t, t+1] -/
noncomputable def max_value (t : ℝ) : ℝ :=
  if t < -2 then -(t + 2)^2 + 1
  else if t ≤ -1 then 1
  else -(t + 1)^2 + 1

/-- The function f(x) = -x^2 - 2x -/
def f (x : ℝ) : ℝ := -x^2 - 2*x

theorem max_value_correct (t : ℝ) :
  ∀ x ∈ Set.Icc t (t + 1), f x ≤ max_value t := by
  sorry

#check max_value_correct

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_correct_l569_56936


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_properties_l569_56918

/-- A random variable following a binomial distribution B(n, p) -/
structure BinomialRV (n : ℕ) (p : ℝ) where
  X : ℝ → ℝ

/-- The expected value of a binomial random variable -/
def expectedValue (X : BinomialRV n p) : ℝ := n * p

/-- The variance of a binomial random variable -/
def variance (X : BinomialRV n p) : ℝ := n * p * (1 - p)

/-- A linear transformation of a random variable -/
def linearTransform (X : BinomialRV n p) (a b : ℝ) : ℝ → ℝ :=
  fun ω => a * X.X ω + b

theorem binomial_properties (X : BinomialRV 10 (3/5)) 
  (Y : ℝ → ℝ) (h : Y = linearTransform X 5 (-2)) :
  expectedValue X = 6 ∧ 
  variance X = 12/5 ∧ 
  (∀ ω, Y ω = 5 * X.X ω - 2) ∧
  (∀ ω, (Y ω - 28)^2 = 25 * (X.X ω - 6)^2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_binomial_properties_l569_56918


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_polygon_ABHFGD_l569_56908

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a square -/
structure Square where
  topLeft : Point
  sideLength : ℝ

/-- The area of a polygon ABHFGD formed by two squares -/
def areaABHFGD (square1 square2 : Square) (h : Point) : ℝ :=
  sorry

theorem area_of_polygon_ABHFGD :
  ∀ (a b c d e f g h : Point),
    let square1 := Square.mk a 3
    let square2 := Square.mk e 5
    d = square1.topLeft →
    d = square2.topLeft →
    h.x = (b.x + c.x) / 2 →
    h.y = (b.y + c.y) / 2 →
    h.x = (e.x + f.x) / 2 →
    h.y = (e.y + f.y) / 2 →
    areaABHFGD square1 square2 h = 25.5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_area_of_polygon_ABHFGD_l569_56908


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_car_fuel_efficiency_l569_56990

/-- Proves that a car can travel approximately 64.01 kilometers on a liter of fuel given specific conditions -/
theorem car_fuel_efficiency (speed : ℝ) (time : ℝ) (fuel_consumed_gallons : ℝ) 
  (gallons_to_liters : ℝ) (miles_to_km : ℝ) :
  speed = 104 →
  time = 5.7 →
  fuel_consumed_gallons = 3.9 →
  gallons_to_liters = 3.8 →
  miles_to_km = 1.6 →
  ∃ (distance_per_liter : ℝ),
    abs (distance_per_liter - 64.01) < 0.01 ∧ 
    distance_per_liter = (speed * time * miles_to_km) / (fuel_consumed_gallons * gallons_to_liters) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_car_fuel_efficiency_l569_56990


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_brendas_journey_distance_l569_56917

/-- The distance between two points in a 2D plane -/
noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ :=
  Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)

/-- The total distance of Brenda's journey -/
noncomputable def total_distance : ℝ :=
  distance (-4) 5 0 0 + distance 0 0 5 (-4)

/-- Theorem stating that the total distance of Brenda's journey is 2√41 -/
theorem brendas_journey_distance :
  total_distance = 2 * Real.sqrt 41 := by
  -- Expand the definition of total_distance
  unfold total_distance
  -- Expand the definition of distance
  unfold distance
  -- Simplify the expression
  simp
  -- The proof is complete
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_brendas_journey_distance_l569_56917


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_caroline_lassis_l569_56900

/-- The ratio of lassis to usable mangoes -/
def lassi_mango_ratio : ℚ := 11 / 3

/-- The percentage of usable mangoes in a batch -/
def usable_mango_percentage : ℚ := 85 / 100

/-- The number of mangoes Caroline has -/
def total_mangoes : ℕ := 18

/-- Calculate the number of lassis Caroline can make -/
def lassis_from_mangoes (ratio : ℚ) (usable_percent : ℚ) (total : ℕ) : ℕ :=
  (ratio * (usable_percent * total : ℚ)).floor.toNat

/-- Theorem stating that Caroline can make 55 lassis from 18 mangoes -/
theorem caroline_lassis : 
  lassis_from_mangoes lassi_mango_ratio usable_mango_percentage total_mangoes = 55 := by
  sorry

#eval lassis_from_mangoes lassi_mango_ratio usable_mango_percentage total_mangoes

end NUMINAMATH_CALUDE_ERRORFEEDBACK_caroline_lassis_l569_56900


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_one_equals_three_l569_56944

/-- A function f defined as f(x) = a*sin(x) - b*tan(x) + 4*cos(π/3) -/
noncomputable def f (a b : ℝ) (x : ℝ) : ℝ := a * Real.sin x - b * Real.tan x + 4 * Real.cos (Real.pi / 3)

/-- Theorem stating that if f(-1) = 1, then f(1) = 3 -/
theorem f_one_equals_three (a b : ℝ) (h : f a b (-1) = 1) : f a b 1 = 3 := by
  sorry

#check f_one_equals_three

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_one_equals_three_l569_56944


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_sin_c_l569_56905

theorem triangle_sin_c (A B C : ℝ) (a b c : ℝ) : 
  B = 2 * π / 3 → b = 3 * c → Real.sin C = Real.sqrt 3 / 6 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_sin_c_l569_56905


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_job_completion_time_l569_56982

/-- The time taken for two workers to complete a job together, given their individual completion times -/
theorem job_completion_time (a_time b_time : ℝ) (ha : a_time > 0) (hb : b_time > 0) :
  (a_time = 20 ∧ b_time = 15) →
  1 / (1 / a_time + 1 / b_time) = 60 / 7 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_job_completion_time_l569_56982


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_highway_additional_employees_l569_56983

/-- Parameters for the highway construction problem -/
structure HighwayParams where
  initialMen : ℕ
  totalLength : ℕ
  initialDays : ℕ
  initialHours : ℕ
  completedDays : ℕ
  completedFraction : ℚ
  newHours : ℕ
  totalManHours : ℕ

/-- Calculate the additional employees needed to complete the highway on time -/
def additionalEmployees (params : HighwayParams) : ℕ :=
  let remainingDays := params.initialDays - params.completedDays
  let remainingFraction := 1 - params.completedFraction
  let remainingManHours := (params.totalManHours : ℚ) * remainingFraction
  let requiredMen := (remainingManHours / (remainingDays * params.newHours : ℚ)).ceil.toNat
  (requiredMen - params.initialMen).max 0

/-- Theorem stating that 60 additional employees are needed -/
theorem highway_additional_employees :
  let params : HighwayParams := {
    initialMen := 100
    totalLength := 2
    initialDays := 50
    initialHours := 8
    completedDays := 25
    completedFraction := 1/3
    newHours := 10
    totalManHours := 60000
  }
  additionalEmployees params = 60 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_highway_additional_employees_l569_56983


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_f_positive_range_m_for_all_real_l569_56929

-- Define the function f
def f (x m : ℝ) : ℝ := |x + 1| + |x - 2| - m

-- Theorem 1: Solution set of f(x) > 0 when m = 5
theorem solution_set_f_positive (x : ℝ) :
  f x 5 > 0 ↔ x ∈ Set.Ioi (-2) ∪ Set.Iio 3 := by
  sorry

-- Theorem 2: Range of m when solution set of f(x) ≥ 0 is ℝ
theorem range_m_for_all_real (m : ℝ) :
  (∀ x : ℝ, f x m ≥ 0) → m ≤ 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_f_positive_range_m_for_all_real_l569_56929


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_point_C_coordinates_l569_56987

noncomputable section

-- Define the points
def A : ℝ × ℝ := (-2, 4)
def B : ℝ × ℝ := (3, -1)
def D : ℝ × ℝ := (0, 2)

-- Define the ratio in which AD divides BC
def ratio : ℝ := 1 / 2

-- Theorem statement
theorem triangle_point_C_coordinates :
  let C : ℝ × ℝ := ((2 * B.1 + A.1) / 3, (2 * B.2 + A.2) / 3)
  (C.1 = 4/3 ∧ C.2 = 2/3) ∧
  -- AD is perpendicular to BC
  (C.2 - B.2) * (A.1 - D.1) + (C.1 - B.1) * (D.2 - A.2) = 0 ∧
  -- AD divides BC in the ratio 1:2
  (D.1 - B.1) / (C.1 - D.1) = ratio ∧
  (D.2 - B.2) / (C.2 - D.2) = ratio := by
  sorry

end

end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_point_C_coordinates_l569_56987


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_sum_formula_l569_56994

def f (n : ℕ) : ℕ → ℚ := λ k => 2^(3*k + 1)

theorem f_sum_formula (n : ℕ) : 
  (Finset.range (n+1)).sum (f n) = (2 / 7) * (8^(n+4) - 1) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_sum_formula_l569_56994


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_with_special_sum_of_divisors_l569_56904

def is_prime (p : ℕ) : Prop := Nat.Prime p

def sum_of_divisors (n : ℕ) : ℕ := (Nat.divisors n).sum id

theorem prime_with_special_sum_of_divisors :
  ∀ s : ℕ, 2 ≤ s → s ≤ 10 →
    ∀ p : ℕ, is_prime p →
      (∃ n : ℕ, sum_of_divisors p = n^s) →
        p ∈ ({3, 7, 31, 127} : Finset ℕ) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_prime_with_special_sum_of_divisors_l569_56904


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_area_implies_centroid_l569_56910

/-- Triangle PQR with vertices P(2,1), Q(-4,-3), and R(3,-2) -/
def triangle_PQR : Set (ℝ × ℝ) :=
  {(2, 1), (-4, -3), (3, -2)}

/-- Point S(x,y) inside triangle PQR -/
def point_S (x y : ℝ) : ℝ × ℝ := (x, y)

/-- Area of a triangle given three points -/
noncomputable def area_triangle (A B C : ℝ × ℝ) : ℝ :=
  let (x1, y1) := A
  let (x2, y2) := B
  let (x3, y3) := C
  (1/2) * abs ((x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2)))

/-- Condition that S divides PQR into three equal area triangles -/
def equal_area_condition (S : ℝ × ℝ) : Prop :=
  ∃ (A : ℝ), 
    area_triangle S (2, 1) (-4, -3) = A ∧
    area_triangle S (2, 1) (3, -2) = A ∧
    area_triangle S (-4, -3) (3, -2) = A

/-- The theorem to be proved -/
theorem equal_area_implies_centroid (x y : ℝ) :
  equal_area_condition (point_S x y) →
  10 * x + y = 2 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_equal_area_implies_centroid_l569_56910


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_ceiling_floor_l569_56938

theorem max_value_ceiling_floor (a b : ℕ+) :
  ∃ (lambda : ℝ), Irrational lambda ∧
  ∀ (mu : ℝ), Irrational mu →
    a * ⌈b * lambda⌉ - b * ⌊a * lambda⌋ ≥ a * ⌈b * mu⌉ - b * ⌊a * mu⌋ ∧
    a * ⌈b * lambda⌉ - b * ⌊a * lambda⌋ = a + b - Nat.gcd a b :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_ceiling_floor_l569_56938


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_steps_is_14_l569_56956

/-- Represents a room in the building --/
inductive Room
| Outside
| One
| Two
| Three
| Four
| Five
| Six
| Seven

/-- Represents a door between two rooms --/
def Door := Room × Room

/-- A building configuration --/
structure Building where
  doors : List Door

/-- Checks if a path is valid for the given building --/
def isValidPath (b : Building) (path : List Room) : Prop :=
  sorry

/-- Counts the number of steps in a path --/
def countSteps (path : List Room) : Nat :=
  path.length - 1

/-- Theorem: The maximum number of steps required to enter room 1, 
    reach the treasure in room 7, and return outside is 14 --/
theorem max_steps_is_14 (b : Building) : 
  (∃ path : List Room, 
    path.head? = some Room.Outside ∧ 
    path.getLast? = some Room.Outside ∧ 
    Room.Seven ∈ path ∧
    isValidPath b path) →
  (∀ path : List Room, 
    path.head? = some Room.Outside ∧ 
    path.getLast? = some Room.Outside ∧ 
    Room.Seven ∈ path ∧
    isValidPath b path →
    countSteps path ≤ 14) ∧
  (∃ path : List Room, 
    path.head? = some Room.Outside ∧ 
    path.getLast? = some Room.Outside ∧ 
    Room.Seven ∈ path ∧
    isValidPath b path ∧
    countSteps path = 14) :=
  by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_steps_is_14_l569_56956


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_fudge_sale_amount_l569_56911

/-- Represents the sale of various candy items in a store --/
structure CandySale where
  fudgePrice : ℝ
  trufflePrice : ℝ
  pretzelPrice : ℝ
  truffleCount : ℕ
  pretzelCount : ℕ
  totalRevenue : ℝ

/-- Calculates the number of pounds of fudge sold given a CandySale --/
noncomputable def fudgePoundsSold (sale : CandySale) : ℝ :=
  (sale.totalRevenue - sale.trufflePrice * sale.truffleCount - sale.pretzelPrice * sale.pretzelCount) / sale.fudgePrice

/-- Theorem stating that given the specific conditions, the store sold 20 pounds of fudge --/
theorem fudge_sale_amount : 
  ∀ (sale : CandySale), 
    sale.fudgePrice = 2.5 →
    sale.trufflePrice = 1.5 →
    sale.pretzelPrice = 2 →
    sale.truffleCount = 60 →
    sale.pretzelCount = 36 →
    sale.totalRevenue = 212 →
    fudgePoundsSold sale = 20 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_fudge_sale_amount_l569_56911


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_graph_shift_l569_56975

theorem sin_graph_shift (x : ℝ) :
  Real.sin (2*x - 2*π/3) = Real.sin (2*(x - π/3)) :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sin_graph_shift_l569_56975


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_and_min_distance_l569_56984

-- Define the triangle ABC
def B : ℝ × ℝ := (-1, 0)
def C : ℝ × ℝ := (1, 0)

-- Define the condition |AB| + |AC| = 4
def triangle_condition (A : ℝ × ℝ) : Prop :=
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) +
  Real.sqrt ((A.1 - C.1)^2 + (A.2 - C.2)^2) = 4

-- Define the trajectory M
def trajectory_M (x y : ℝ) : Prop :=
  x^2/4 + y^2/3 = 1 ∧ y ≠ 0

-- Define a point P on trajectory M
def P_on_M (P : ℝ × ℝ) : Prop :=
  trajectory_M P.1 P.2

-- Define the center O₁ of the circumcircle of △PBC
noncomputable def O₁ (P : ℝ × ℝ) : ℝ × ℝ :=
  let midBC := ((B.1 + C.1)/2, (B.2 + C.2)/2)
  let midPB := ((P.1 + B.1)/2, (P.2 + B.2)/2)
  let slopeBC := (C.2 - B.2)/(C.1 - B.1)
  let slopePB := (P.2 - B.2)/(P.1 - B.1)
  let perpSlopeBC := -1/slopeBC
  let perpSlopePB := -1/slopePB
  ((midBC.2 - midPB.2 + perpSlopeBC*midBC.1 - perpSlopePB*midPB.1)/(perpSlopeBC - perpSlopePB),
   midBC.2 + perpSlopeBC*((midBC.2 - midPB.2 + perpSlopeBC*midBC.1 - perpSlopePB*midPB.1)/(perpSlopeBC - perpSlopePB) - midBC.1))

-- Theorem statement
theorem trajectory_and_min_distance :
  (∀ A : ℝ × ℝ, triangle_condition A → trajectory_M A.1 A.2) ∧
  (∃ d : ℝ, d = Real.sqrt 3 / 3 ∧
    ∀ P : ℝ × ℝ, P_on_M P →
      d ≤ |(O₁ P).2| ∧
      (∃ Q : ℝ × ℝ, P_on_M Q ∧ |(O₁ Q).2| = d)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_and_min_distance_l569_56984


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_centroid_inequality_l569_56964

-- Define a triangle
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define the centroid of a triangle
noncomputable def centroid (t : Triangle) : ℝ × ℝ :=
  ((t.A.1 + t.B.1 + t.C.1) / 3, (t.A.2 + t.B.2 + t.C.2) / 3)

-- Define a line passing through a point
structure Line where
  point : ℝ × ℝ
  direction : ℝ × ℝ

-- Define the intersection of a line with a line segment
noncomputable def intersect (l : Line) (p q : ℝ × ℝ) : ℝ × ℝ := sorry

-- Define the distance between two points
noncomputable def distance (p q : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem centroid_inequality (t : Triangle) (l : Line) :
  let G := centroid t
  let E := intersect l t.A t.B
  let F := intersect l t.A t.C
  l.point = G →
  distance E G ≤ 2 * distance G F := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_centroid_inequality_l569_56964


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_onto_same_vector_l569_56906

noncomputable def v1 : ℝ × ℝ := (3, 2)
noncomputable def v2 : ℝ × ℝ := (1, 4)
noncomputable def q : ℝ × ℝ := (11/2, 9/2)

theorem projection_onto_same_vector :
  ∃ (v : ℝ × ℝ) (k1 k2 : ℝ),
    v1 - k1 • v = q ∧
    v2 - k2 • v = q ∧
    (v1 - q) • v = 0 ∧
    (v2 - q) • v = 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_onto_same_vector_l569_56906


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_l569_56961

theorem equation_solution (x : ℝ) : 
  Real.sqrt 2 * (Real.sin x + Real.cos x) = Real.tan x + (Real.cos x / Real.sin x) ↔ 
  ∃ l : ℤ, x = π / 4 + 2 * l * π :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_l569_56961


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_inequality_l569_56909

theorem solution_set_inequality : 
  Set.Ioo (-2 : ℝ) 3 = {x : ℝ | (x - 3) * (x + 2) < 0} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_set_inequality_l569_56909


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_correct_l569_56948

/-- The trajectory of points whose sum of distances to A(0,0) and B(3,4) equals 5 -/
def trajectory (x y : ℝ) : Prop :=
  4 * x - 3 * y = 0 ∧ 0 ≤ x ∧ x ≤ 3

/-- Point A -/
def A : ℝ × ℝ := (0, 0)

/-- Point B -/
def B : ℝ × ℝ := (3, 4)

/-- Distance between two points -/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

/-- Theorem stating that the trajectory function correctly describes the points
    whose sum of distances to A and B equals 5 -/
theorem trajectory_correct (x y : ℝ) :
  trajectory x y ↔ distance (x, y) A + distance (x, y) B = 5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_trajectory_correct_l569_56948


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_set_function_domain_l569_56973

-- Problem 1
theorem inequality_solution_set (x : ℝ) :
  -x^2 + 4*x + 5 < 0 ↔ x < -1 ∨ x > 5 := by sorry

-- Problem 2
noncomputable def f (x : ℝ) : ℝ := Real.sqrt ((x - 1) / (x + 2)) + 5

theorem function_domain (x : ℝ) :
  x ∈ Set.univ \ {-2} ∧ (x - 1) / (x + 2) ≥ 0 ↔ x < -2 ∨ x ≥ 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inequality_solution_set_function_domain_l569_56973


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_mean_proportional_234_104_l569_56930

/-- The mean proportional between two numbers -/
noncomputable def mean_proportional (a b : ℝ) : ℝ := Real.sqrt (a * b)

/-- Theorem: The mean proportional between 234 and 104 is 156 -/
theorem mean_proportional_234_104 : mean_proportional 234 104 = 156 := by
  -- Unfold the definition of mean_proportional
  unfold mean_proportional
  -- Simplify the expression under the square root
  simp [Real.sqrt_mul]
  -- The proof steps would go here, but we'll use sorry for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_mean_proportional_234_104_l569_56930


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_smallest_five_digit_in_pascal_l569_56953

/-- Definition of PascalTriangle (simplified for this problem) -/
def PascalTriangle : Set ℕ := {n : ℕ | n > 0}

/-- Pascal's triangle contains every positive integer -/
axiom pascal_contains_all_positive : ∀ n : ℕ, n > 0 → n ∈ PascalTriangle

/-- 10000 is the smallest five-digit number -/
axiom smallest_five_digit : (∀ n : ℕ, n < 10000 → n < 10000) ∧ 10000 ≥ 10000

/-- 10001 is the second smallest five-digit number in Pascal's triangle -/
theorem second_smallest_five_digit_in_pascal : 
  ∃! n : ℕ, n ∈ PascalTriangle ∧ n > 10000 ∧ (∀ m : ℕ, m ∈ PascalTriangle → m > 10000 → m ≥ n) ∧ n = 10001 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_smallest_five_digit_in_pascal_l569_56953


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_diagonal_length_perimeter_l569_56931

-- Define the rectangle
noncomputable def rectangle_width : ℝ := 40 * Real.sqrt 3
noncomputable def rectangle_height : ℝ := 30 * Real.sqrt 3

-- Theorem for the diagonal length
theorem diagonal_length :
  Real.sqrt (rectangle_width ^ 2 + rectangle_height ^ 2) = 50 * Real.sqrt 3 := by
  sorry

-- Theorem for the perimeter
theorem perimeter :
  2 * (rectangle_width + rectangle_height) = 140 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_diagonal_length_perimeter_l569_56931


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_pairs_l569_56941

-- Define the floor function
noncomputable def floor (x : ℝ) : ℤ := Int.floor x

-- Define the equation
def equation (a b x y : ℝ) : Prop :=
  floor (a*x + b*y) + floor (b*x + a*y) = floor ((a + b) * (x + y))

-- State the theorem
theorem solution_pairs :
  ∀ a b : ℝ, (∀ x y : ℝ, equation a b x y) ↔ ((a = 1 ∧ b = 1) ∨ (a = 0 ∧ b = 0)) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_solution_pairs_l569_56941


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_T_residue_l569_56972

def T : ℤ := (List.range 2026).foldr (fun i acc =>
  if i % 4 < 2 then acc + (i + 1) else acc - (i + 1)) 0

theorem T_residue : T ≡ 2026 [ZMOD 2027] := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_T_residue_l569_56972


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_to_zero_neg_four_fifths_closest_l569_56902

noncomputable def values : List ℝ := [-1, 5/4, 1^2, -4/5, 0.9]

theorem closest_to_zero (values : List ℝ) : 
  ∃ x ∈ values, ∀ y ∈ values, |x| ≤ |y| := by
  sorry

theorem neg_four_fifths_closest : 
  ∃ x ∈ values, (x = -4/5 ∧ ∀ y ∈ values, |x| ≤ |y|) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_closest_to_zero_neg_four_fifths_closest_l569_56902


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_equation_l569_56922

/-- Represents a parabola with equation y² = ax -/
structure Parabola where
  a : ℝ

/-- Represents a line with a given slope -/
structure Line where
  slope : ℝ

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- The focus of a parabola -/
noncomputable def focus (p : Parabola) : Point :=
  { x := p.a / 4, y := 0 }

/-- The area of a triangle given three points -/
noncomputable def triangleArea (p1 p2 p3 : Point) : ℝ :=
  sorry

/-- Checks if a line passes through a point -/
def Line.passesThrough (l : Line) (p : Point) : Prop :=
  sorry

/-- Checks if a line intersects the y-axis at a given point -/
def Line.intersectsYAxisAt (l : Line) (p : Point) : Prop :=
  sorry

theorem parabola_equation (p : Parabola) (l : Line) (A : Point) :
  l.slope = 2 →
  l.passesThrough (focus p) →
  l.intersectsYAxisAt A →
  triangleArea { x := 0, y := 0 } A (focus p) = 4 →
  p.a = 8 ∨ p.a = -8 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_parabola_equation_l569_56922


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_center_radius_sum_l569_56981

/-- The equation of circle C is x^2 + 12y + 57 = -y^2 - 10x -/
def circle_equation (x y : ℝ) : Prop := x^2 + 12*y + 57 = -y^2 - 10*x

/-- (a, b) is the center of circle C with radius r -/
def is_center (a b r : ℝ) : Prop := ∀ x y : ℝ, circle_equation x y ↔ (x - a)^2 + (y - b)^2 = r^2

/-- r is the radius of circle C -/
def is_radius (r : ℝ) : Prop := ∃ a b : ℝ, is_center a b r ∧ r > 0

theorem circle_center_radius_sum :
  ∃ a b r : ℝ, is_center a b r ∧ is_radius r ∧ a + b + r = -9 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_center_radius_sum_l569_56981


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_lambda_over_m_l569_56942

theorem range_of_lambda_over_m (lambda m alpha : ℝ) 
  (a : ℝ × ℝ) (b : ℝ × ℝ)
  (ha : a = (lambda + 2, lambda^2 - Real.cos alpha ^ 2))
  (hb : b = (m, m / 2 + Real.sin alpha))
  (heq : a = 2 • b) :
  ∃ k : ℝ, k = lambda / m ∧ -6 ≤ k ∧ k ≤ 1 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_lambda_over_m_l569_56942


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_meeting_distance_l569_56932

/-- The distance traveled by Train A when it meets Train B -/
noncomputable def distance_train_A_at_meeting (total_distance : ℝ) (time_A : ℝ) (time_B : ℝ) : ℝ :=
  let speed_A := total_distance / time_A
  let speed_B := total_distance / time_B
  let time_to_meet := total_distance / (speed_A + speed_B)
  speed_A * time_to_meet

/-- Theorem stating that the distance traveled by Train A when it meets Train B is approximately 50 miles -/
theorem train_meeting_distance :
  let total_distance : ℝ := 125
  let time_A : ℝ := 12
  let time_B : ℝ := 8
  ∃ ε > 0, |distance_train_A_at_meeting total_distance time_A time_B - 50| < ε :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_meeting_distance_l569_56932


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_sum_equality_l569_56947

theorem distinct_sum_equality (S : Finset ℕ) : 
  S.card = 13 → (∀ x, x ∈ S → 1 ≤ x ∧ x ≤ 37) → 
  ∃ a b c d, a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ d ∈ S ∧
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d ∧ a + b = c + d :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distinct_sum_equality_l569_56947


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_alice_walks_25km_l569_56907

/-- The distance Alice walks before meeting Bob -/
noncomputable def alice_distance (total_distance : ℝ) (alice_speed bob_speed : ℝ) (head_start : ℝ) : ℝ :=
  let meeting_time := (total_distance - alice_speed * head_start) / (alice_speed + bob_speed)
  alice_speed * (meeting_time + head_start)

/-- Theorem stating that Alice walks 25 km before meeting Bob -/
theorem alice_walks_25km :
  alice_distance 41 5 4 1 = 25 := by
  -- Unfold the definition of alice_distance
  unfold alice_distance
  -- Simplify the expression
  simp
  -- Perform numerical calculations
  norm_num
  -- Close the proof
  done

end NUMINAMATH_CALUDE_ERRORFEEDBACK_alice_walks_25km_l569_56907


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_ratio_l569_56949

theorem triangle_ratio (a b c A B C : Real) :
  (0 < a) ∧ (0 < b) ∧ (0 < c) ∧
  (0 < A) ∧ (A < π) ∧ (0 < B) ∧ (B < π) ∧ (0 < C) ∧ (C < π) ∧
  (A + B + C = π) ∧
  (a = b * Real.sin A / Real.sin B) ∧ (b = c * Real.sin B / Real.sin C) ∧ (c = a * Real.sin C / Real.sin A) ∧
  (a + b * Real.cos C = Real.cos B + 4 * Real.cos C) ∧
  (A = π / 6) →
  (b / c = Real.sqrt 3 / 2) ∨ (b / c = 2) :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_triangle_ratio_l569_56949


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_equation_negative_root_m_range_l569_56920

theorem quadratic_equation_negative_root_m_range :
  ∀ m : ℝ,
  (∃ x : ℝ, x < 0 ∧ m * x^2 + 2 * x + 1 = 0) →
  m ≤ 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_quadratic_equation_negative_root_m_range_l569_56920


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_tangent_point_l569_56950

/-- 
Given two curves C₁: y = e^x and C₂: y = (x+a)², 
if they have the same tangent line at their intersection point,
then a = 2 - ln 4
-/
theorem intersection_tangent_point (a : ℝ) : 
  (∃ x₀ : ℝ, 
    Real.exp x₀ = (x₀ + a)^2 ∧ 
    Real.exp x₀ = 2 * (x₀ + a)) → 
  a = 2 - Real.log 4 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_intersection_tangent_point_l569_56950


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_max_f_equals_one_l569_56928

open Real

-- Define the function f(x, y) = |x^2 - xy + 1|
noncomputable def f (x y : ℝ) : ℝ := abs (x^2 - x*y + 1)

-- Define the maximum of f(x, y) over the interval [0, 2]
noncomputable def max_f (y : ℝ) : ℝ := 
  ⨆ (x : ℝ) (h : 0 ≤ x ∧ x ≤ 2), f x y

-- State the theorem
theorem min_max_f_equals_one : 
  ⨅ (y : ℝ), max_f y = 1 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_max_f_equals_one_l569_56928


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_functional_equation_solution_l569_56959

/-- A function satisfying the given functional equation -/
def FunctionalEquation (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x > 0 → y > 0 → x * f y - y * f x = f (x * y)

/-- The main theorem stating that f(100) = 0 for any function satisfying the functional equation -/
theorem functional_equation_solution (f : ℝ → ℝ) (h : FunctionalEquation f) : f 100 = 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_functional_equation_solution_l569_56959


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sunny_lead_theorem_l569_56954

/-- Represents a runner in the race -/
structure Runner where
  speed : ℝ

/-- Represents the race scenario -/
structure RaceScenario where
  h : ℝ  -- race distance
  d : ℝ  -- Sunny's lead in the first race
  sunny : Runner
  windy : Runner

/-- Defines the conditions of the race scenario -/
def valid_race_scenario (race : RaceScenario) : Prop :=
  race.h > 0 ∧ race.d > 0 ∧
  race.sunny.speed > 0 ∧ race.windy.speed > 0 ∧
  race.h / race.sunny.speed = (race.h - race.d) / race.windy.speed

/-- Calculates Sunny's lead at the end of the second race -/
noncomputable def sunny_lead_second_race (race : RaceScenario) : ℝ :=
  (race.d^2 + 2*race.d^2 - 2*race.h*race.d) / race.h

/-- Theorem stating that Sunny's lead at the end of the second race
    is (d^2 + 2d^2 - 2hd) / h meters -/
theorem sunny_lead_theorem (race : RaceScenario) 
  (h_valid : valid_race_scenario race) : 
  sunny_lead_second_race race = (race.d^2 + 2*race.d^2 - 2*race.h*race.d) / race.h :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sunny_lead_theorem_l569_56954


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_solution_set_g_minimum_greater_than_f_l569_56969

-- Define the functions f and g
noncomputable def f (x : ℝ) : ℝ := |x - 2| - |x + 1|

noncomputable def g (a x : ℝ) : ℝ := (a * x^2 - x + 1) / x

-- State the theorems to be proved
theorem f_inequality_solution_set :
  {x : ℝ | f x > 1} = {x : ℝ | x < 0} := by sorry

theorem g_minimum_greater_than_f (a : ℝ) :
  (a > 0) → (∀ x > 0, ∃ m, (∀ y > 0, g a y ≥ m) ∧ (g a x = m) ∧ (m > f x)) ↔ a ≥ 1 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_inequality_solution_set_g_minimum_greater_than_f_l569_56969
