import Mathlib

namespace NUMINAMATH_CALUDE_hiker_first_day_distance_l732_73258

/-- A hiker's three-day journey --/
def HikersJourney (h : ℝ) : Prop :=
  let d1 := 3 * h  -- Distance on day 1
  let d2 := 4 * (h - 1)  -- Distance on day 2
  let d3 := 5 * 6  -- Distance on day 3
  d1 + d2 + d3 = 68  -- Total distance

/-- The hiker walked 18 miles on the first day --/
theorem hiker_first_day_distance :
  ∃ h : ℝ, HikersJourney h ∧ 3 * h = 18 :=
sorry

end NUMINAMATH_CALUDE_hiker_first_day_distance_l732_73258


namespace NUMINAMATH_CALUDE_a_in_M_necessary_not_sufficient_for_a_in_N_l732_73245

def M : Set ℝ := {x | 0 < x ∧ x ≤ 3}
def N : Set ℝ := {x | 0 < x ∧ x ≤ 2}

theorem a_in_M_necessary_not_sufficient_for_a_in_N :
  (∀ a, a ∈ N → a ∈ M) ∧ (∃ a, a ∈ M ∧ a ∉ N) :=
sorry

end NUMINAMATH_CALUDE_a_in_M_necessary_not_sufficient_for_a_in_N_l732_73245


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l732_73288

def A : Set ℝ := {x | |x| ≤ 2}
def B : Set ℝ := {x | x^2 - 1 ≥ 0}

theorem intersection_of_A_and_B : 
  A ∩ B = {x : ℝ | -2 ≤ x ∧ x ≤ -1 ∨ 1 ≤ x ∧ x ≤ 2} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l732_73288


namespace NUMINAMATH_CALUDE_carpenter_logs_needed_l732_73296

/-- A carpenter building a house needs additional logs. -/
theorem carpenter_logs_needed
  (total_woodblocks_needed : ℕ)
  (logs_available : ℕ)
  (woodblocks_per_log : ℕ)
  (h1 : total_woodblocks_needed = 80)
  (h2 : logs_available = 8)
  (h3 : woodblocks_per_log = 5) :
  total_woodblocks_needed - logs_available * woodblocks_per_log = 8 * woodblocks_per_log :=
by sorry

end NUMINAMATH_CALUDE_carpenter_logs_needed_l732_73296


namespace NUMINAMATH_CALUDE_crayon_difference_l732_73223

/-- Given an initial number of crayons, the number given away, and the number lost,
    prove that the difference between lost and given away is their subtraction. -/
theorem crayon_difference (initial given lost : ℕ) : lost - given = lost - given := by
  sorry

end NUMINAMATH_CALUDE_crayon_difference_l732_73223


namespace NUMINAMATH_CALUDE_rational_sum_theorem_l732_73207

theorem rational_sum_theorem (a₁ a₂ a₃ a₄ : ℚ) : 
  ({a₁ * a₂, a₁ * a₃, a₁ * a₄, a₂ * a₃, a₂ * a₄, a₃ * a₄} : Finset ℚ) = 
    {-24, -2, -3/2, -1/8, 1, 3} → 
  a₁ + a₂ + a₃ + a₄ = 9/4 ∨ a₁ + a₂ + a₃ + a₄ = -9/4 := by
sorry

end NUMINAMATH_CALUDE_rational_sum_theorem_l732_73207


namespace NUMINAMATH_CALUDE_probability_one_from_each_group_l732_73278

theorem probability_one_from_each_group :
  ∀ (total : ℕ) (group1 : ℕ) (group2 : ℕ),
    total = group1 + group2 →
    group1 > 0 →
    group2 > 0 →
    (group1 : ℚ) / total * group2 / (total - 1) +
    (group2 : ℚ) / total * group1 / (total - 1) = 5 / 9 :=
by
  sorry

end NUMINAMATH_CALUDE_probability_one_from_each_group_l732_73278


namespace NUMINAMATH_CALUDE_polynomial_without_xy_term_l732_73227

theorem polynomial_without_xy_term (k : ℝ) : 
  (∀ x y : ℝ, x^2 - 3*k*x*y - 3*y^2 + 6*x*y - 8 = x^2 - 3*y^2 - 8) ↔ k = 2 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_without_xy_term_l732_73227


namespace NUMINAMATH_CALUDE_logarithm_exponent_equality_special_case_2019_l732_73222

theorem logarithm_exponent_equality (x : ℝ) (hx : x > 1) : 
  x^(Real.log (Real.log x)) = (Real.log x)^(Real.log x) :=
by
  sorry

-- The main theorem
theorem special_case_2019 : 
  2019^(Real.log (Real.log 2019)) - (Real.log 2019)^(Real.log 2019) = 0 :=
by
  sorry

end NUMINAMATH_CALUDE_logarithm_exponent_equality_special_case_2019_l732_73222


namespace NUMINAMATH_CALUDE_dodecahedron_outer_rectangle_property_l732_73283

/-- Regular dodecahedron with side length a -/
structure RegularDodecahedron (a : ℝ) where
  side_length : a > 0

/-- Point on a line outside a face of the dodecahedron -/
structure OuterPoint (a m : ℝ) where
  distance : m > 0

/-- Rectangle formed by four outer points -/
structure OuterRectangle (a m : ℝ) where
  A : OuterPoint a m
  B : OuterPoint a m
  C : OuterPoint a m
  D : OuterPoint a m

theorem dodecahedron_outer_rectangle_property 
  (a m : ℝ) 
  (d : RegularDodecahedron a) 
  (r : OuterRectangle a m) : 
  ∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ 
  y / x = (1 + Real.sqrt 5) / 2 := by
  sorry

end NUMINAMATH_CALUDE_dodecahedron_outer_rectangle_property_l732_73283


namespace NUMINAMATH_CALUDE_line_tangent_to_log_curve_l732_73259

/-- A line y = x + 1 is tangent to the curve y = ln(x + a) if and only if a = 2 -/
theorem line_tangent_to_log_curve (a : ℝ) : 
  (∃ x : ℝ, x + 1 = Real.log (x + a) ∧ 
   ∀ y : ℝ, y ≠ x → y + 1 ≠ Real.log (y + a) ∧
   (1 : ℝ) = 1 / (x + a)) ↔ 
  a = 2 := by
  sorry

end NUMINAMATH_CALUDE_line_tangent_to_log_curve_l732_73259


namespace NUMINAMATH_CALUDE_number_of_teachers_l732_73264

/-- Represents the number of students at Queen Middle School -/
def total_students : ℕ := 1500

/-- Represents the number of classes each student takes per day -/
def classes_per_student : ℕ := 6

/-- Represents the number of classes each teacher teaches -/
def classes_per_teacher : ℕ := 5

/-- Represents the number of students in each class -/
def students_per_class : ℕ := 25

/-- Represents the number of teachers in each class -/
def teachers_per_class : ℕ := 1

/-- Theorem stating that the number of teachers at Queen Middle School is 72 -/
theorem number_of_teachers : 
  (total_students * classes_per_student) / students_per_class / classes_per_teacher = 72 := by
  sorry

end NUMINAMATH_CALUDE_number_of_teachers_l732_73264


namespace NUMINAMATH_CALUDE_only_earth_revolves_certain_l732_73215

-- Define the type for events
inductive Event
| earth_revolves : Event
| shooter_hits_bullseye : Event
| three_suns_appear : Event
| red_light_encounter : Event

-- Define the property of being a certain event
def is_certain_event (e : Event) : Prop :=
  match e with
  | Event.earth_revolves => True
  | _ => False

-- Theorem stating that only the Earth revolving is a certain event
theorem only_earth_revolves_certain :
  ∀ e : Event, is_certain_event e ↔ e = Event.earth_revolves :=
sorry

end NUMINAMATH_CALUDE_only_earth_revolves_certain_l732_73215


namespace NUMINAMATH_CALUDE_pizza_slices_per_person_l732_73205

theorem pizza_slices_per_person 
  (total_slices : Nat) 
  (people : Nat) 
  (slices_left : Nat) 
  (h1 : total_slices = 16) 
  (h2 : people = 6) 
  (h3 : slices_left = 4) 
  (h4 : people > 0) : 
  (total_slices - slices_left) / people = 2 := by
sorry

end NUMINAMATH_CALUDE_pizza_slices_per_person_l732_73205


namespace NUMINAMATH_CALUDE_range_of_a_l732_73270

-- Define the propositions p and q
def p (a : ℝ) : Prop := ∀ x, a^x > 1 ↔ x < 0

def q (a : ℝ) : Prop := ∀ x, a*x^2 - x + a > 0

-- Define the theorem
theorem range_of_a (a : ℝ) :
  (p a ∨ q a) ∧ ¬(p a ∧ q a) →
  (0 < a ∧ a ≤ 1/2) ∨ a ≥ 1 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l732_73270


namespace NUMINAMATH_CALUDE_books_after_donation_l732_73287

theorem books_after_donation (boris_initial : ℕ) (cameron_initial : ℕ) 
  (boris_donation_fraction : ℚ) (cameron_donation_fraction : ℚ)
  (h1 : boris_initial = 24)
  (h2 : cameron_initial = 30)
  (h3 : boris_donation_fraction = 1/4)
  (h4 : cameron_donation_fraction = 1/3) :
  (boris_initial - boris_initial * boris_donation_fraction).floor +
  (cameron_initial - cameron_initial * cameron_donation_fraction).floor = 38 := by
sorry

end NUMINAMATH_CALUDE_books_after_donation_l732_73287


namespace NUMINAMATH_CALUDE_compound_interest_principal_exists_l732_73263

theorem compound_interest_principal_exists : ∃ (P r : ℝ), 
  P > 0 ∧ r > 0 ∧ 
  P * (1 + r)^2 = 8800 ∧ 
  P * (1 + r)^3 = 9261 := by
sorry

end NUMINAMATH_CALUDE_compound_interest_principal_exists_l732_73263


namespace NUMINAMATH_CALUDE_triangle_properties_l732_73210

noncomputable section

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

/-- The theorem statement -/
theorem triangle_properties (t : Triangle) 
  (h1 : 2 * Real.cos t.A * Real.cos t.C + 1 = 2 * Real.sin t.A * Real.sin t.C)
  (h2 : t.a + t.c = 3 * Real.sqrt 3 / 2)
  (h3 : t.b = Real.sqrt 3) :
  t.B = π / 3 ∧ 
  (1/2 * t.a * t.c * Real.sin t.B) = 5 * Real.sqrt 3 / 16 := by
  sorry

end

end NUMINAMATH_CALUDE_triangle_properties_l732_73210


namespace NUMINAMATH_CALUDE_quadratic_roots_sum_l732_73217

theorem quadratic_roots_sum (m n : ℝ) : 
  (m^2 + 5*m - 2023 = 0) → (n^2 + 5*n - 2023 = 0) → m^2 + 7*m + 2*n = 2013 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_roots_sum_l732_73217


namespace NUMINAMATH_CALUDE_tan_equation_solution_l732_73228

open Set Real

-- Define the set of angles that satisfy the conditions
def solution_set : Set ℝ := {x | 0 ≤ x ∧ x < π ∧ tan (4 * x - π / 4) = 1}

-- State the theorem
theorem tan_equation_solution :
  solution_set = {π/8, 3*π/8, 5*π/8, 7*π/8} := by sorry

end NUMINAMATH_CALUDE_tan_equation_solution_l732_73228


namespace NUMINAMATH_CALUDE_range_of_a_l732_73267

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, x^2 - 4*a*x + 3*a^2 < 0 → |x - 3| > 1) ∧ 
  (∃ x : ℝ, |x - 3| > 1 ∧ x^2 - 4*a*x + 3*a^2 ≥ 0) ∧
  (a > 0) →
  a ≥ 4 ∨ (0 < a ∧ a ≤ 2/3) :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l732_73267


namespace NUMINAMATH_CALUDE_smallest_n_for_divisibility_l732_73281

/-- Given a positive odd number m, find the smallest natural number n 
    such that 2^1989 divides m^n - 1 -/
theorem smallest_n_for_divisibility (m : ℕ) (h_m_pos : 0 < m) (h_m_odd : Odd m) :
  ∃ (k : ℕ), ∃ (n : ℕ),
    (∀ (i : ℕ), i ≤ k → m % (2^i) = 1) ∧
    (m % (2^(k+1)) ≠ 1) ∧
    (n = 2^(1989 - k)) ∧
    (2^1989 ∣ m^n - 1) ∧
    (∀ (j : ℕ), j < n → ¬(2^1989 ∣ m^j - 1)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_for_divisibility_l732_73281


namespace NUMINAMATH_CALUDE_ellipse_x_intercept_l732_73200

/-- Definition of an ellipse with given foci and a point on it -/
def is_ellipse (f1 f2 p : ℝ × ℝ) : Prop :=
  let d1 := Real.sqrt ((p.1 - f1.1)^2 + (p.2 - f1.2)^2)
  let d2 := Real.sqrt ((p.2 - f2.1)^2 + (p.2 - f2.2)^2)
  let c := Real.sqrt ((f1.1 - f2.1)^2 + (f1.2 - f2.2)^2)
  d1 + d2 = 2 * Real.sqrt ((c/2)^2 + ((d1 - d2)/2)^2)

/-- The ellipse intersects the x-axis at (0, 0) -/
def intersects_origin (f1 f2 : ℝ × ℝ) : Prop :=
  is_ellipse f1 f2 (0, 0)

/-- Theorem: For an ellipse with foci at (0, 3) and (4, 0) that intersects
    the x-axis at (0, 0), the other x-intercept is at (56/11, 0) -/
theorem ellipse_x_intercept :
  let f1 : ℝ × ℝ := (0, 3)
  let f2 : ℝ × ℝ := (4, 0)
  intersects_origin f1 f2 →
  is_ellipse f1 f2 (56/11, 0) ∧
  ∀ x : ℝ, x ≠ 0 ∧ x ≠ 56/11 → ¬is_ellipse f1 f2 (x, 0) := by
  sorry

end NUMINAMATH_CALUDE_ellipse_x_intercept_l732_73200


namespace NUMINAMATH_CALUDE_world_grain_ratio_2010_l732_73276

theorem world_grain_ratio_2010 : 
  let supply : ℕ := 1800000
  let demand : ℕ := 2400000
  (supply : ℚ) / demand = 3 / 4 := by sorry

end NUMINAMATH_CALUDE_world_grain_ratio_2010_l732_73276


namespace NUMINAMATH_CALUDE_sheets_in_stack_l732_73253

/-- Given that 400 sheets of paper are 4 centimeters thick, 
    prove that a 14-inch high stack contains 3556 sheets. -/
theorem sheets_in_stack (sheets_in_4cm : ℕ) (thickness_4cm : ℝ) 
  (stack_height_inches : ℝ) (cm_per_inch : ℝ) :
  sheets_in_4cm = 400 →
  thickness_4cm = 4 →
  stack_height_inches = 14 →
  cm_per_inch = 2.54 →
  (stack_height_inches * cm_per_inch) / (thickness_4cm / sheets_in_4cm) = 3556 := by
  sorry

end NUMINAMATH_CALUDE_sheets_in_stack_l732_73253


namespace NUMINAMATH_CALUDE_arithmetic_sequence_fifth_term_l732_73257

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_fifth_term
  (a : ℕ → ℝ)
  (h_arith : ArithmeticSequence a)
  (h_sum : a 2 + a 4 = 16)
  (h_first : a 1 = 1) :
  a 5 = 15 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_fifth_term_l732_73257


namespace NUMINAMATH_CALUDE_farm_animals_l732_73204

theorem farm_animals (total_animals : ℕ) (total_legs : ℕ) 
  (h1 : total_animals = 8)
  (h2 : total_legs = 24) :
  ∃ (ducks dogs : ℕ),
    ducks + dogs = total_animals ∧
    2 * ducks + 4 * dogs = total_legs ∧
    ducks = 4 := by
  sorry

end NUMINAMATH_CALUDE_farm_animals_l732_73204


namespace NUMINAMATH_CALUDE_consecutive_even_squares_divisibility_l732_73241

theorem consecutive_even_squares_divisibility (n : ℤ) : 
  ∃ (k : ℤ), (4 * n ^ 2 + (4 * n ^ 2 + 8 * n + 4) + (4 * n ^ 2 + 16 * n + 16)) = 4 * k ∧
  ∃ (m : ℤ), (4 * n ^ 2 + (4 * n ^ 2 + 8 * n + 4) + (4 * n ^ 2 + 16 * n + 16)) ≠ 7 * m :=
by sorry

end NUMINAMATH_CALUDE_consecutive_even_squares_divisibility_l732_73241


namespace NUMINAMATH_CALUDE_room_equation_l732_73289

/-- 
Theorem: For a positive integer x representing the number of rooms, 
if accommodating 6 people per room leaves exactly one room vacant, 
and accommodating 5 people per room leaves 4 people unaccommodated, 
then the equation 6(x-1) = 5x + 4 holds true.
-/
theorem room_equation (x : ℕ+) 
  (h1 : 6 * (x - 1) = 6 * x - 6)  -- With 6 people per room, one room is vacant
  (h2 : 5 * x + 4 = 6 * x - 6)    -- With 5 people per room, 4 people are unaccommodated
  : 6 * (x - 1) = 5 * x + 4 := by
  sorry


end NUMINAMATH_CALUDE_room_equation_l732_73289


namespace NUMINAMATH_CALUDE_equation_represents_point_l732_73266

/-- The equation x^2 + 36y^2 - 12x - 72y + 36 = 0 represents a single point (6, 1) in the xy-plane -/
theorem equation_represents_point :
  ∀ x y : ℝ, x^2 + 36*y^2 - 12*x - 72*y + 36 = 0 ↔ x = 6 ∧ y = 1 := by
  sorry

end NUMINAMATH_CALUDE_equation_represents_point_l732_73266


namespace NUMINAMATH_CALUDE_quadrilateral_area_is_77_over_6_l732_73262

/-- Represents the arrangement of three squares -/
structure SquareArrangement where
  small_side : ℝ
  medium_side : ℝ
  large_side : ℝ
  coplanar : Prop
  side_by_side : Prop

/-- Calculates the area of the quadrilateral formed in the square arrangement -/
def quadrilateral_area (arr : SquareArrangement) : ℝ :=
  sorry

/-- The main theorem stating that the quadrilateral area is 77/6 -/
theorem quadrilateral_area_is_77_over_6 (arr : SquareArrangement) 
  (h1 : arr.small_side = 3)
  (h2 : arr.medium_side = 5)
  (h3 : arr.large_side = 7)
  (h4 : arr.coplanar)
  (h5 : arr.side_by_side) :
  quadrilateral_area arr = 77 / 6 :=
sorry

end NUMINAMATH_CALUDE_quadrilateral_area_is_77_over_6_l732_73262


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l732_73249

theorem quadratic_equation_solution : ∃ x₁ x₂ : ℝ, 
  (x₁ = 2 + Real.sqrt 7 ∧ x₁^2 - 4*x₁ + 7 = 10) ∧
  (x₂ = 2 - Real.sqrt 7 ∧ x₂^2 - 4*x₂ + 7 = 10) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l732_73249


namespace NUMINAMATH_CALUDE_nine_possible_values_for_D_l732_73277

-- Define the type for digits (0-9)
def Digit := Fin 10

-- Define the addition equation
def addition_equation (A B C D : Digit) : Prop :=
  10000 * A.val + 1000 * B.val + 100 * A.val + 10 * C.val + A.val +
  10000 * C.val + 1000 * A.val + 100 * D.val + 10 * A.val + B.val =
  10000 * D.val + 1000 * C.val + 100 * D.val + 10 * D.val + D.val

-- Define the distinct digits condition
def distinct_digits (A B C D : Digit) : Prop :=
  A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ B ≠ C ∧ B ≠ D ∧ C ≠ D

-- Theorem statement
theorem nine_possible_values_for_D :
  ∃ (s : Finset Digit),
    s.card = 9 ∧
    (∀ D ∈ s, ∃ A B C, distinct_digits A B C D ∧ addition_equation A B C D) ∧
    (∀ D, D ∉ s → ¬∃ A B C, distinct_digits A B C D ∧ addition_equation A B C D) :=
sorry

end NUMINAMATH_CALUDE_nine_possible_values_for_D_l732_73277


namespace NUMINAMATH_CALUDE_school_fee_calculation_l732_73201

def mother_contribution : ℕ := 2 * 100 + 1 * 50 + 5 * 20 + 3 * 10 + 4 * 5
def father_contribution : ℕ := 3 * 100 + 4 * 50 + 2 * 20 + 1 * 10 + 6 * 5

theorem school_fee_calculation : mother_contribution + father_contribution = 980 := by
  sorry

end NUMINAMATH_CALUDE_school_fee_calculation_l732_73201


namespace NUMINAMATH_CALUDE_beach_visitors_beach_visitors_proof_l732_73295

theorem beach_visitors (initial_people : ℕ) (people_left : ℕ) (total_if_stayed : ℕ) : ℕ :=
  let total_before_leaving := total_if_stayed + people_left
  total_before_leaving - initial_people

#check beach_visitors 3 40 63 = 100

/- Proof
theorem beach_visitors_proof :
  beach_visitors 3 40 63 = 100 := by
  sorry
-/

end NUMINAMATH_CALUDE_beach_visitors_beach_visitors_proof_l732_73295


namespace NUMINAMATH_CALUDE_complex_cube_equality_l732_73252

theorem complex_cube_equality (c d : ℝ) (h : d > 0) :
  (c + d * Complex.I) ^ 3 = (c - d * Complex.I) ^ 3 ↔ d / c = Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_complex_cube_equality_l732_73252


namespace NUMINAMATH_CALUDE_books_difference_alicia_ian_l732_73246

/-- Represents a student in the book reading contest -/
structure Student where
  name : String
  booksRead : Nat

/-- Represents the book reading contest -/
structure BookReadingContest where
  students : Finset Student
  alicia : Student
  ian : Student
  aliciaMostBooks : ∀ s ∈ students, s.booksRead ≤ alicia.booksRead
  ianFewestBooks : ∀ s ∈ students, ian.booksRead ≤ s.booksRead
  aliciaInContest : alicia ∈ students
  ianInContest : ian ∈ students
  contestSize : students.card = 8
  aliciaBooksRead : alicia.booksRead = 8
  ianBooksRead : ian.booksRead = 1

/-- The difference in books read between Alicia and Ian is 7 -/
theorem books_difference_alicia_ian (contest : BookReadingContest) :
  contest.alicia.booksRead - contest.ian.booksRead = 7 := by
  sorry

end NUMINAMATH_CALUDE_books_difference_alicia_ian_l732_73246


namespace NUMINAMATH_CALUDE_bullet_speed_difference_l732_73242

/-- The speed of the horse in feet per second -/
def horse_speed : ℝ := 20

/-- The speed of the bullet in feet per second -/
def bullet_speed : ℝ := 400

/-- The difference in bullet speed when fired in the same direction as the horse versus the opposite direction -/
def speed_difference : ℝ := (bullet_speed + horse_speed) - (bullet_speed - horse_speed)

theorem bullet_speed_difference :
  speed_difference = 40 :=
by sorry

end NUMINAMATH_CALUDE_bullet_speed_difference_l732_73242


namespace NUMINAMATH_CALUDE_square_mod_four_l732_73286

theorem square_mod_four (n : ℤ) : (n^2) % 4 = 0 ∨ (n^2) % 4 = 1 := by
  sorry

end NUMINAMATH_CALUDE_square_mod_four_l732_73286


namespace NUMINAMATH_CALUDE_x_gt_3_sufficient_not_necessary_for_x_sq_gt_4_l732_73269

theorem x_gt_3_sufficient_not_necessary_for_x_sq_gt_4 :
  (∀ x : ℝ, x > 3 → x^2 > 4) ∧
  ¬(∀ x : ℝ, x^2 > 4 → x > 3) :=
by sorry

end NUMINAMATH_CALUDE_x_gt_3_sufficient_not_necessary_for_x_sq_gt_4_l732_73269


namespace NUMINAMATH_CALUDE_vasya_meeting_time_l732_73275

/-- Represents time in hours and minutes -/
structure Time where
  hours : Nat
  minutes : Nat
  h_valid : hours < 24
  m_valid : minutes < 60

/-- Calculates the difference in minutes between two times -/
def timeDifference (t1 t2 : Time) : Int :=
  (t1.hours * 60 + t1.minutes) - (t2.hours * 60 + t2.minutes)

theorem vasya_meeting_time :
  let normalArrival : Time := ⟨18, 0, by norm_num, by norm_num⟩
  let earlyArrival : Time := ⟨17, 0, by norm_num, by norm_num⟩
  let meetingTime : Time := ⟨17, 50, by norm_num, by norm_num⟩
  let normalHomeArrival : Time := ⟨19, 0, by norm_num, by norm_num⟩  -- Assuming normal home arrival is at 19:00
  let earlyHomeArrival : Time := ⟨18, 40, by norm_num, by norm_num⟩  -- 20 minutes earlier than normal

  -- Vasya arrives 1 hour early
  timeDifference normalArrival earlyArrival = 60 →
  -- They arrive home 20 minutes earlier than usual
  timeDifference normalHomeArrival earlyHomeArrival = 20 →
  -- The meeting time is 10 minutes before the normal arrival time
  timeDifference normalArrival meetingTime = 10 →
  -- The meeting time is 50 minutes after the early arrival time
  timeDifference meetingTime earlyArrival = 50 →
  meetingTime = ⟨17, 50, by norm_num, by norm_num⟩ :=
by
  sorry


end NUMINAMATH_CALUDE_vasya_meeting_time_l732_73275


namespace NUMINAMATH_CALUDE_ornament_shop_profit_maximization_l732_73212

/-- Ornament shop profit maximization problem -/
theorem ornament_shop_profit_maximization :
  ∀ (cost_A cost_B selling_price total_quantity : ℕ) 
    (min_B max_B_ratio discount_threshold discount_rate : ℕ),
  cost_A = 1400 →
  cost_B = 630 →
  cost_A = 2 * cost_B →
  selling_price = 15 →
  total_quantity = 600 →
  min_B = 390 →
  max_B_ratio = 4 →
  discount_threshold = 150 →
  discount_rate = 40 →
  ∃ (quantity_A quantity_B profit : ℕ),
    quantity_A + quantity_B = total_quantity ∧
    quantity_B ≥ min_B ∧
    quantity_B ≤ max_B_ratio * quantity_A ∧
    quantity_A = 210 ∧
    quantity_B = 390 ∧
    profit = 3630 ∧
    (∀ (other_quantity_A other_quantity_B other_profit : ℕ),
      other_quantity_A + other_quantity_B = total_quantity →
      other_quantity_B ≥ min_B →
      other_quantity_B ≤ max_B_ratio * other_quantity_A →
      other_profit ≤ profit) :=
by sorry

end NUMINAMATH_CALUDE_ornament_shop_profit_maximization_l732_73212


namespace NUMINAMATH_CALUDE_no_real_solution_ffx_l732_73247

/-- A second-degree polynomial function -/
def SecondDegreePolynomial (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, ∀ x, f x = a * x^2 + b * x + c

/-- No real solution for f(x) = x -/
def NoRealSolutionForFX (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f x ≠ x

/-- No real solution for f(f(x)) = x -/
def NoRealSolutionForFFX (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (f x) ≠ x

theorem no_real_solution_ffx 
  (f : ℝ → ℝ) 
  (h1 : SecondDegreePolynomial f) 
  (h2 : NoRealSolutionForFX f) : 
  NoRealSolutionForFFX f :=
sorry

end NUMINAMATH_CALUDE_no_real_solution_ffx_l732_73247


namespace NUMINAMATH_CALUDE_fraction_simplification_l732_73230

theorem fraction_simplification (y b : ℝ) : 
  (y + 2) / 4 + (5 - 4*y + b) / 3 = (-13*y + 4*b + 26) / 12 := by
sorry

end NUMINAMATH_CALUDE_fraction_simplification_l732_73230


namespace NUMINAMATH_CALUDE_consecutive_integers_square_sum_l732_73243

theorem consecutive_integers_square_sum (x : ℕ) : 
  x > 0 ∧ x * (x + 1) = 812 → x^2 + (x + 1)^2 = 1625 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_integers_square_sum_l732_73243


namespace NUMINAMATH_CALUDE_midpoint_segment_length_is_three_l732_73256

/-- A trapezoid with specific properties -/
structure Trapezoid where
  /-- The sum of the two base angles is 90° -/
  base_angles_sum : ℝ
  /-- The length of the upper base -/
  upper_base : ℝ
  /-- The length of the lower base -/
  lower_base : ℝ
  /-- The sum of the two base angles is 90° -/
  base_angles_sum_eq : base_angles_sum = 90
  /-- The length of the upper base is 5 -/
  upper_base_eq : upper_base = 5
  /-- The length of the lower base is 11 -/
  lower_base_eq : lower_base = 11

/-- The length of the segment connecting the midpoints of the two bases -/
def midpoint_segment_length (t : Trapezoid) : ℝ := 3

/-- Theorem: The length of the segment connecting the midpoints of the two bases is 3 -/
theorem midpoint_segment_length_is_three (t : Trapezoid) :
  midpoint_segment_length t = 3 := by
  sorry

#check midpoint_segment_length_is_three

end NUMINAMATH_CALUDE_midpoint_segment_length_is_three_l732_73256


namespace NUMINAMATH_CALUDE_equation_solution_l732_73229

theorem equation_solution :
  ∃ x : ℝ, (3 / (x - 2) = 2 / (x - 1)) ∧ (x = -1) :=
by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l732_73229


namespace NUMINAMATH_CALUDE_polynomial_simplification_l732_73297

theorem polynomial_simplification (q : ℝ) :
  (4 * q^4 + 5 * q^3 - 7 * q + 8) + (3 - 9 * q^3 + 5 * q^2 - 2 * q) =
  4 * q^4 - 4 * q^3 + 5 * q^2 - 9 * q + 11 :=
by sorry

end NUMINAMATH_CALUDE_polynomial_simplification_l732_73297


namespace NUMINAMATH_CALUDE_triangle_formation_l732_73216

/-- Triangle Inequality Theorem: A triangle can be formed if the sum of the lengths of any two sides
    is greater than the length of the remaining side. -/
def can_form_triangle (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

/-- Given three line segments with lengths 4cm, 5cm, and 6cm, they can form a triangle. -/
theorem triangle_formation : can_form_triangle 4 5 6 := by
  sorry

end NUMINAMATH_CALUDE_triangle_formation_l732_73216


namespace NUMINAMATH_CALUDE_first_player_wins_l732_73272

/-- Represents a digit (0-9) -/
def Digit : Type := Fin 10

/-- Represents an operation (addition or multiplication) -/
inductive Operation
| add : Operation
| mul : Operation

/-- Represents a game state -/
structure GameState :=
  (digits : List Digit)
  (operations : List Operation)

/-- Represents a game move -/
structure Move :=
  (digit : Digit)
  (operation : Option Operation)

/-- Evaluates the final result of the game -/
def evaluateGame (state : GameState) : ℕ :=
  sorry

/-- Checks if a number is even -/
def isEven (n : ℕ) : Prop :=
  ∃ k, n = 2 * k

/-- Theorem: The first player can always win with optimal play -/
theorem first_player_wins :
  ∀ (initial_digit : Digit),
    isEven initial_digit.val →
    ∃ (strategy : List Move),
      ∀ (opponent_moves : List Move),
        let final_state := sorry
        isEven (evaluateGame final_state) :=
by sorry

end NUMINAMATH_CALUDE_first_player_wins_l732_73272


namespace NUMINAMATH_CALUDE_bus_passengers_l732_73248

theorem bus_passengers (men women : ℕ) : 
  women = men / 2 →
  men - 16 = women + 8 →
  men + women = 72 :=
by
  sorry

end NUMINAMATH_CALUDE_bus_passengers_l732_73248


namespace NUMINAMATH_CALUDE_jays_family_percentage_l732_73280

theorem jays_family_percentage (total_guests : ℕ) (female_percentage : ℚ) (jays_family_females : ℕ) : 
  total_guests = 240 → 
  female_percentage = 60 / 100 → 
  jays_family_females = 72 → 
  (jays_family_females : ℚ) / (female_percentage * total_guests) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_jays_family_percentage_l732_73280


namespace NUMINAMATH_CALUDE_not_sufficient_nor_necessary_l732_73292

theorem not_sufficient_nor_necessary (p q : Prop) :
  ¬(((p ∧ q) → ¬p) ∧ (¬p → (p ∧ q))) := by sorry

end NUMINAMATH_CALUDE_not_sufficient_nor_necessary_l732_73292


namespace NUMINAMATH_CALUDE_stating_angle_edge_to_face_special_case_l732_73203

/-- Represents a trihedral angle with vertex A and edges AB, AC, and AD -/
structure TrihedralAngle where
  BAC : ℝ  -- Angle between AB and AC
  CAD : ℝ  -- Angle between AC and AD
  BAD : ℝ  -- Angle between AB and AD

/-- 
Calculates the angle between edge AB and face ACD in a trihedral angle
given the measures of angles BAC, CAD, and BAD
-/
def angleEdgeToFace (t : TrihedralAngle) : ℝ :=
  sorry

/-- 
Theorem stating that for a trihedral angle with BAC = 45°, CAD = 90°, and BAD = 60°,
the angle between edge AB and face ACD is 30°
-/
theorem angle_edge_to_face_special_case :
  let t : TrihedralAngle := { BAC := Real.pi / 4, CAD := Real.pi / 2, BAD := Real.pi / 3 }
  angleEdgeToFace t = Real.pi / 6 := by
  sorry

end NUMINAMATH_CALUDE_stating_angle_edge_to_face_special_case_l732_73203


namespace NUMINAMATH_CALUDE_complex_number_location_l732_73211

theorem complex_number_location :
  let z : ℂ := (1 - Complex.I) / (1 + Complex.I)
  let w : ℂ := z / (1 + Complex.I)
  (w.re < 0) ∧ (w.im < 0) := by
  sorry

end NUMINAMATH_CALUDE_complex_number_location_l732_73211


namespace NUMINAMATH_CALUDE_min_perimeter_of_rectangle_l732_73290

theorem min_perimeter_of_rectangle (l w : ℕ) : 
  l * w = 50 → 2 * (l + w) ≥ 30 := by
  sorry

end NUMINAMATH_CALUDE_min_perimeter_of_rectangle_l732_73290


namespace NUMINAMATH_CALUDE_thirty_people_handshakes_l732_73293

/-- The number of handshakes in a group of n people where each person shakes hands
    with every other person exactly once. -/
def handshakes (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem stating that in a group of 30 people, the total number of handshakes is 435. -/
theorem thirty_people_handshakes :
  handshakes 30 = 435 := by sorry

end NUMINAMATH_CALUDE_thirty_people_handshakes_l732_73293


namespace NUMINAMATH_CALUDE_point_on_angle_bisector_l732_73254

/-- 
Given a point M with coordinates (3n-2, 2n+7) that lies on the angle bisector 
of the second and fourth quadrants, prove that n = -1.
-/
theorem point_on_angle_bisector (n : ℝ) : 
  (∃ M : ℝ × ℝ, M.1 = 3*n - 2 ∧ M.2 = 2*n + 7 ∧ 
   M.1 + M.2 = 0) → n = -1 := by
sorry

end NUMINAMATH_CALUDE_point_on_angle_bisector_l732_73254


namespace NUMINAMATH_CALUDE_simplify_power_of_power_l732_73235

theorem simplify_power_of_power (x : ℝ) : (5 * x^2)^4 = 625 * x^8 := by
  sorry

end NUMINAMATH_CALUDE_simplify_power_of_power_l732_73235


namespace NUMINAMATH_CALUDE_percent_relation_l732_73261

theorem percent_relation (x y z : ℝ) (h1 : 0.45 * z = 0.39 * y) (h2 : z = 0.65 * x) :
  y = 0.75 * x := by
  sorry

end NUMINAMATH_CALUDE_percent_relation_l732_73261


namespace NUMINAMATH_CALUDE_tyler_saltwater_animals_l732_73226

/-- Represents the number of aquariums of each type -/
structure AquariumCounts where
  typeA : ℕ
  typeB : ℕ
  typeC : ℕ

/-- Represents the number of animals in each type of aquarium -/
structure AquariumAnimals where
  typeA : ℕ
  typeB : ℕ
  typeC : ℕ

/-- Calculates the total number of saltwater animals -/
def totalSaltwaterAnimals (counts : AquariumCounts) (animals : AquariumAnimals) : ℕ :=
  counts.typeA * animals.typeA + counts.typeB * animals.typeB + counts.typeC * animals.typeC

/-- Tyler's aquarium setup -/
def tylerAquariums : AquariumCounts :=
  { typeA := 10
    typeB := 14
    typeC := 6 }

/-- Number of animals in each type of Tyler's aquariums -/
def tylerAnimals : AquariumAnimals :=
  { typeA := 12 * 4  -- 12 corals with 4 animals each
    typeB := 18 + 10 -- 18 large fish and 10 small fish
    typeC := 25 + 20 -- 25 invertebrates and 20 small fish
  }

theorem tyler_saltwater_animals :
  totalSaltwaterAnimals tylerAquariums tylerAnimals = 1142 := by
  sorry

end NUMINAMATH_CALUDE_tyler_saltwater_animals_l732_73226


namespace NUMINAMATH_CALUDE_shoe_pairs_count_l732_73250

theorem shoe_pairs_count (total_shoes : ℕ) (prob_same_color : ℚ) : 
  total_shoes = 16 → 
  prob_same_color = 1 / 15 → 
  (total_shoes / 2 : ℕ) = 8 := by
sorry

end NUMINAMATH_CALUDE_shoe_pairs_count_l732_73250


namespace NUMINAMATH_CALUDE_birds_on_fence_l732_73233

theorem birds_on_fence (initial_birds : ℝ) (birds_flew_away : ℝ) (remaining_birds : ℝ) : 
  initial_birds = 12.0 → birds_flew_away = 8.0 → remaining_birds = initial_birds - birds_flew_away → remaining_birds = 4.0 := by
  sorry

end NUMINAMATH_CALUDE_birds_on_fence_l732_73233


namespace NUMINAMATH_CALUDE_range_of_expressions_l732_73273

-- Define variables a and b with given constraints
theorem range_of_expressions (a b : ℝ) (ha : 1 < a ∧ a < 4) (hb : 2 < b ∧ b < 8) :
  -- (1) Range of a/b
  (1/8 : ℝ) < a/b ∧ a/b < 2 ∧
  -- (2) Range of 2a + 3b
  8 < 2*a + 3*b ∧ 2*a + 3*b < 32 ∧
  -- (3) Range of a - b
  -7 < a - b ∧ a - b < 2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_expressions_l732_73273


namespace NUMINAMATH_CALUDE_simplify_expression_l732_73221

theorem simplify_expression (p q r : ℝ) 
  (hp : p ≠ 7) (hq : q ≠ 8) (hr : r ≠ 9) : 
  (p - 7) / (9 - r) * (q - 8) / (7 - p) * (r - 9) / (8 - q) = -1 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l732_73221


namespace NUMINAMATH_CALUDE_building_height_l732_73274

/-- Given a flagpole and a building casting shadows under similar conditions,
    prove that the height of the building is 20 meters. -/
theorem building_height
  (flagpole_height : ℝ)
  (flagpole_shadow : ℝ)
  (building_shadow : ℝ)
  (h_flagpole_height : flagpole_height = 18)
  (h_flagpole_shadow : flagpole_shadow = 45)
  (h_building_shadow : building_shadow = 50)
  : (flagpole_height / flagpole_shadow) * building_shadow = 20 :=
by sorry

end NUMINAMATH_CALUDE_building_height_l732_73274


namespace NUMINAMATH_CALUDE_max_distance_origin_to_line_l732_73240

/-- Given a line l with equation ax + by + c = 0, where a, b, and c form an arithmetic sequence,
    the maximum distance from the origin O(0,0) to the line l is √5. -/
theorem max_distance_origin_to_line (a b c : ℝ) :
  (∃ d : ℝ, a - b = b - c) →  -- a, b, c form an arithmetic sequence
  (∃ x y : ℝ, a * x + b * y + c = 0) →  -- line equation exists
  (∃ d : ℝ, ∀ x y : ℝ, a * x + b * y + c = 0 → d ≥ Real.sqrt (x^2 + y^2)) →  -- distance definition
  (∃ d : ℝ, ∀ x y : ℝ, a * x + b * y + c = 0 → d ≤ Real.sqrt 5) →  -- upper bound
  (∃ x y : ℝ, a * x + b * y + c = 0 ∧ Real.sqrt (x^2 + y^2) = Real.sqrt 5)  -- maximum distance achieved
  := by sorry

end NUMINAMATH_CALUDE_max_distance_origin_to_line_l732_73240


namespace NUMINAMATH_CALUDE_distance_is_600_km_l732_73209

/-- The distance between the starting points of two persons traveling towards each other -/
def distance_between_starting_points (speed1 speed2 : ℝ) (travel_time : ℝ) : ℝ :=
  (speed1 + speed2) * travel_time

/-- Theorem stating that the distance between starting points is 600 km -/
theorem distance_is_600_km (speed1 speed2 travel_time : ℝ) 
  (h1 : speed1 = 70)
  (h2 : speed2 = 80)
  (h3 : travel_time = 4) :
  distance_between_starting_points speed1 speed2 travel_time = 600 := by
  sorry

#check distance_is_600_km

end NUMINAMATH_CALUDE_distance_is_600_km_l732_73209


namespace NUMINAMATH_CALUDE_vector_subtraction_l732_73260

theorem vector_subtraction (a b : ℝ × ℝ) : 
  a = (3, 5) → b = (-2, 1) → a - 2 • b = (7, 3) := by
  sorry

end NUMINAMATH_CALUDE_vector_subtraction_l732_73260


namespace NUMINAMATH_CALUDE_smallest_d_value_l732_73244

theorem smallest_d_value (a b c d e : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) (he : e > 0)
  (h : ∀ x : ℝ, (x + a) * (x + b) * (x + c) = x^3 + 3*d*x^2 + 3*x + e^3) :
  d ≥ 1 ∧ ∃ (a₀ b₀ c₀ e₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ c₀ > 0 ∧ e₀ > 0 ∧
    (∀ x : ℝ, (x + a₀) * (x + b₀) * (x + c₀) = x^3 + 3*x^2 + 3*x + e₀^3) := by
  sorry

#check smallest_d_value

end NUMINAMATH_CALUDE_smallest_d_value_l732_73244


namespace NUMINAMATH_CALUDE_norm_photos_difference_l732_73231

-- Define the number of photos taken by each photographer
variable (L M N : ℕ)

-- Define the conditions from the problem
def condition1 (L M N : ℕ) : Prop := L + M = M + N - 60
def condition2 (L N : ℕ) : Prop := ∃ X, N = 2 * L + X
def condition3 (N : ℕ) : Prop := N = 110

-- State the theorem
theorem norm_photos_difference (L M N : ℕ) 
  (h1 : condition1 L M N) (h2 : condition2 L N) (h3 : condition3 N) : 
  ∃ X, N = 2 * L + X ∧ X = 110 - 2 * L :=
sorry

end NUMINAMATH_CALUDE_norm_photos_difference_l732_73231


namespace NUMINAMATH_CALUDE_smallest_integer_in_set_l732_73237

theorem smallest_integer_in_set (n : ℤ) : 
  (n + 6 ≤ 2 * ((7 * n + 21) / 7)) → n ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_smallest_integer_in_set_l732_73237


namespace NUMINAMATH_CALUDE_danielle_age_l732_73208

/-- Given the ages of Anna, Ben, Carlos, and Danielle, prove that Danielle is 22 years old. -/
theorem danielle_age (anna ben carlos danielle : ℕ)
  (h1 : anna = ben - 4)
  (h2 : ben = carlos + 3)
  (h3 : danielle = carlos + 6)
  (h4 : anna = 15) :
  danielle = 22 := by
sorry

end NUMINAMATH_CALUDE_danielle_age_l732_73208


namespace NUMINAMATH_CALUDE_trumpet_to_running_ratio_l732_73238

/-- Proves that the ratio of time spent practicing trumpet to time spent running is 2:1 -/
theorem trumpet_to_running_ratio 
  (basketball_time : ℕ) 
  (trumpet_time : ℕ) 
  (h1 : basketball_time = 10)
  (h2 : trumpet_time = 40) :
  (trumpet_time : ℚ) / (2 * basketball_time) = 2 / 1 :=
by sorry

end NUMINAMATH_CALUDE_trumpet_to_running_ratio_l732_73238


namespace NUMINAMATH_CALUDE_planar_figure_division_l732_73268

/-- A planar figure with diameter 1 -/
structure PlanarFigure where
  diam : ℝ
  diam_eq_one : diam = 1

/-- The minimum diameter of n parts that a planar figure can be divided into -/
noncomputable def δ₂ (n : ℕ) (F : PlanarFigure) : ℝ := sorry

/-- Main theorem about division of planar figures -/
theorem planar_figure_division (F : PlanarFigure) : 
  (δ₂ 3 F ≤ Real.sqrt 3 / 2) ∧ 
  (δ₂ 4 F ≤ Real.sqrt 2 / 2) ∧ 
  (δ₂ 7 F ≤ 1 / 2) := by sorry

end NUMINAMATH_CALUDE_planar_figure_division_l732_73268


namespace NUMINAMATH_CALUDE_total_amount_spent_l732_73239

def meal_prices : List Float := [12, 15, 10, 18, 20]
def ice_cream_prices : List Float := [2, 3, 3, 4, 4]
def tip_percentage : Float := 0.15
def tax_percentage : Float := 0.08

theorem total_amount_spent :
  let total_meal_cost := meal_prices.sum
  let total_ice_cream_cost := ice_cream_prices.sum
  let tip := tip_percentage * total_meal_cost
  let tax := tax_percentage * total_meal_cost
  total_meal_cost + total_ice_cream_cost + tip + tax = 108.25 := by
sorry

end NUMINAMATH_CALUDE_total_amount_spent_l732_73239


namespace NUMINAMATH_CALUDE_inequality_proof_l732_73294

theorem inequality_proof (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x < y) :
  x + Real.sqrt (y^2 + 2) < y + Real.sqrt (x^2 + 2) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l732_73294


namespace NUMINAMATH_CALUDE_people_reached_in_day_l732_73220

/-- The number of people reached after n hours of message spreading -/
def people_reached (n : ℕ) : ℕ :=
  2^(n+1) - 1

/-- Theorem stating the number of people reached in 24 hours -/
theorem people_reached_in_day : people_reached 24 = 2^24 - 1 := by
  sorry

#eval people_reached 24

end NUMINAMATH_CALUDE_people_reached_in_day_l732_73220


namespace NUMINAMATH_CALUDE_video_game_expenditure_l732_73265

theorem video_game_expenditure (total : ℝ) (books toys snacks : ℝ) : 
  total = 45 →
  books = (1/4) * total →
  toys = (1/3) * total →
  snacks = (2/9) * total →
  total - (books + toys + snacks) = 8.75 :=
by sorry

end NUMINAMATH_CALUDE_video_game_expenditure_l732_73265


namespace NUMINAMATH_CALUDE_exponent_addition_l732_73299

theorem exponent_addition (a : ℝ) : a^3 * a^4 = a^7 := by
  sorry

end NUMINAMATH_CALUDE_exponent_addition_l732_73299


namespace NUMINAMATH_CALUDE_simplify_expression_l732_73232

theorem simplify_expression :
  (3 * (Real.sqrt 3 + Real.sqrt 5)) / (4 * Real.sqrt (3 + Real.sqrt 4)) =
  (3 * Real.sqrt 15 + 3 * Real.sqrt 5) / 20 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l732_73232


namespace NUMINAMATH_CALUDE_perpendicular_line_plane_condition_l732_73291

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular relation between lines and between a line and a plane
variable (perp_line : Line → Line → Prop)
variable (perp_plane : Line → Plane → Prop)

-- Define the subset relation between a line and a plane
variable (subset : Line → Plane → Prop)

-- State the theorem
theorem perpendicular_line_plane_condition 
  (a l : Line) (α : Plane) (h_subset : subset a α) :
  (perp_plane l α → perp_line l a) ∧ 
  ∃ (l' : Line), perp_line l' a ∧ ¬perp_plane l' α :=
sorry

end NUMINAMATH_CALUDE_perpendicular_line_plane_condition_l732_73291


namespace NUMINAMATH_CALUDE_smallest_number_proof_l732_73206

theorem smallest_number_proof (x y z : ℝ) : 
  y = 2 * x →
  z = 4 * y →
  (x + y + z) / 3 = 165 →
  x = 45 := by
sorry

end NUMINAMATH_CALUDE_smallest_number_proof_l732_73206


namespace NUMINAMATH_CALUDE_slope_equals_half_implies_y_eleven_l732_73218

/-- Given two points P and Q in a coordinate plane, if the slope of the line through P and Q is 1/2, then the y-coordinate of Q is 11. -/
theorem slope_equals_half_implies_y_eleven (x₁ y₁ x₂ y₂ : ℝ) : 
  x₁ = -3 → y₁ = 7 → x₂ = 5 → 
  (y₂ - y₁) / (x₂ - x₁) = 1/2 →
  y₂ = 11 := by
  sorry

#check slope_equals_half_implies_y_eleven

end NUMINAMATH_CALUDE_slope_equals_half_implies_y_eleven_l732_73218


namespace NUMINAMATH_CALUDE_range_of_a_l732_73224

theorem range_of_a (a : ℝ) : 
  (∀ x y : ℝ, x > 0 → y > 0 → x + y + 3 = x * y → 
    (x + y)^2 - a * (x + y) + 1 ≥ 0) ↔ 
  a ≤ 37 / 6 :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l732_73224


namespace NUMINAMATH_CALUDE_math_city_intersections_l732_73234

/-- Represents a city with a number of straight, non-parallel streets -/
structure City where
  num_streets : ℕ
  streets_straight : Bool
  streets_non_parallel : Bool

/-- Calculates the maximum number of intersections in a city -/
def max_intersections (city : City) : ℕ :=
  (city.num_streets * (city.num_streets - 1)) / 2

/-- Theorem: A city with 10 straight, non-parallel streets has 45 intersections -/
theorem math_city_intersections :
  ∀ (c : City), c.num_streets = 10 ∧ c.streets_straight ∧ c.streets_non_parallel →
  max_intersections c = 45 := by
  sorry

end NUMINAMATH_CALUDE_math_city_intersections_l732_73234


namespace NUMINAMATH_CALUDE_limit_proof_l732_73298

theorem limit_proof (ε : ℝ) (hε : ε > 0) :
  ∃ δ : ℝ, δ > 0 ∧
  ∀ x : ℝ, 0 < |x - 11| ∧ |x - 11| < δ →
    |(2 * x^2 - 21 * x - 11) / (x - 11) - 23| < ε := by
  sorry

end NUMINAMATH_CALUDE_limit_proof_l732_73298


namespace NUMINAMATH_CALUDE_convex_polygon_angle_theorem_l732_73213

theorem convex_polygon_angle_theorem (n : ℕ) (x : ℝ) :
  n ≥ 3 →
  x > 0 →
  x < 180 →
  (n : ℝ) * 180 - 3 * x = 3330 + 180 * 2 →
  x = 54 := by
  sorry

end NUMINAMATH_CALUDE_convex_polygon_angle_theorem_l732_73213


namespace NUMINAMATH_CALUDE_function_identity_l732_73219

theorem function_identity (f : ℝ → ℝ) 
  (h : ∀ a x : ℝ, a < x ∧ x < a + 100 → a ≤ f x ∧ f x ≤ a + 100) : 
  ∀ x : ℝ, f x = x := by
  sorry

end NUMINAMATH_CALUDE_function_identity_l732_73219


namespace NUMINAMATH_CALUDE_sum_of_divisors_3600_l732_73255

/-- Sum of divisors function -/
def sum_of_divisors (n : ℕ) : ℕ := sorry

/-- Theorem: If the sum of divisors of 2^i * 3^j * 5^k is 3600, then i + j + k = 7 -/
theorem sum_of_divisors_3600 (i j k : ℕ) : 
  sum_of_divisors (2^i * 3^j * 5^k) = 3600 → i + j + k = 7 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_divisors_3600_l732_73255


namespace NUMINAMATH_CALUDE_two_times_three_plus_two_times_three_l732_73279

theorem two_times_three_plus_two_times_three : 2 * 3 + 2 * 3 = 12 := by
  sorry

end NUMINAMATH_CALUDE_two_times_three_plus_two_times_three_l732_73279


namespace NUMINAMATH_CALUDE_infinitely_many_m_with_1000_nonzero_bits_l732_73251

def count_nonzero_bits (m : ℕ) : ℕ :=
  (m.bits.filter (· ≠ 0)).length

theorem infinitely_many_m_with_1000_nonzero_bits :
  ∀ n : ℕ, ∃ m : ℕ, m > n ∧ count_nonzero_bits m = 1000 :=
by sorry

end NUMINAMATH_CALUDE_infinitely_many_m_with_1000_nonzero_bits_l732_73251


namespace NUMINAMATH_CALUDE_factorial_ratio_l732_73225

-- Define the factorial operation
def factorial : Nat → Nat
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

-- State the theorem
theorem factorial_ratio : (factorial 50) / (factorial 48) = 2450 := by
  sorry

end NUMINAMATH_CALUDE_factorial_ratio_l732_73225


namespace NUMINAMATH_CALUDE_sum_lower_bound_l732_73236

theorem sum_lower_bound (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (h : a * b = a + b + 3) :
  6 ≤ a + b := by
  sorry

end NUMINAMATH_CALUDE_sum_lower_bound_l732_73236


namespace NUMINAMATH_CALUDE_tan_thirteen_pi_sixths_l732_73285

theorem tan_thirteen_pi_sixths : Real.tan (13 * π / 6) = 1 / Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_tan_thirteen_pi_sixths_l732_73285


namespace NUMINAMATH_CALUDE_medicine_parts_for_child_l732_73282

/-- Calculates the number of equal parts a medicine dose should be divided into -/
def medicine_parts (weight : ℕ) (dosage_per_kg : ℕ) (mg_per_part : ℕ) : ℕ :=
  (weight * dosage_per_kg * 1000) / mg_per_part

/-- Theorem: For a 30 kg child, with 5 ml/kg dosage and 50 mg parts, the dose divides into 3000 parts -/
theorem medicine_parts_for_child : medicine_parts 30 5 50 = 3000 := by
  sorry

end NUMINAMATH_CALUDE_medicine_parts_for_child_l732_73282


namespace NUMINAMATH_CALUDE_smallest_positive_multiple_of_45_l732_73284

theorem smallest_positive_multiple_of_45 :
  ∀ n : ℕ, n > 0 ∧ 45 ∣ n → n ≥ 45 :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_positive_multiple_of_45_l732_73284


namespace NUMINAMATH_CALUDE_college_sports_participation_l732_73271

/-- The total number of students who play at least one sport (cricket or basketball) -/
def total_students (cricket_players basketball_players both_players : ℕ) : ℕ :=
  cricket_players + basketball_players - both_players

/-- Theorem stating the total number of students playing at least one sport -/
theorem college_sports_participation : 
  total_students 500 600 220 = 880 := by
  sorry

end NUMINAMATH_CALUDE_college_sports_participation_l732_73271


namespace NUMINAMATH_CALUDE_alex_coin_distribution_l732_73214

/-- The minimum number of additional coins needed for distribution -/
def min_additional_coins (friends : ℕ) (initial_coins : ℕ) : ℕ :=
  let required_coins := friends * (friends + 1) / 2
  if required_coins > initial_coins then
    required_coins - initial_coins
  else
    0

/-- Theorem stating the minimum number of additional coins needed -/
theorem alex_coin_distribution (friends : ℕ) (initial_coins : ℕ)
    (h1 : friends = 15)
    (h2 : initial_coins = 95) :
    min_additional_coins friends initial_coins = 25 := by
  sorry

end NUMINAMATH_CALUDE_alex_coin_distribution_l732_73214


namespace NUMINAMATH_CALUDE_blueberry_earnings_relationship_l732_73202

/-- Represents the relationship between blueberry picking amount and earnings --/
def blueberry_earnings (x : ℝ) : ℝ × ℝ :=
  let y₁ := 60 + 30 * 0.6 * x
  let y₂ := 10 * 30 + 30 * 0.5 * (x - 10)
  (y₁, y₂)

/-- Theorem stating the relationship between y₁, y₂, and x when x > 10 --/
theorem blueberry_earnings_relationship (x : ℝ) (h : x > 10) :
  let (y₁, y₂) := blueberry_earnings x
  y₁ = 60 + 18 * x ∧ y₂ = 150 + 15 * x :=
by sorry

end NUMINAMATH_CALUDE_blueberry_earnings_relationship_l732_73202
