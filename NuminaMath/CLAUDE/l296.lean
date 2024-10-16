import Mathlib

namespace NUMINAMATH_CALUDE_gross_profit_calculation_l296_29682

theorem gross_profit_calculation (sales_price : ℝ) (gross_profit_percentage : ℝ) :
  sales_price = 91 →
  gross_profit_percentage = 1.6 →
  ∃ (cost : ℝ) (gross_profit : ℝ),
    cost > 0 ∧
    gross_profit = gross_profit_percentage * cost ∧
    sales_price = cost + gross_profit ∧
    gross_profit = 56 :=
by sorry

end NUMINAMATH_CALUDE_gross_profit_calculation_l296_29682


namespace NUMINAMATH_CALUDE_total_cups_sold_l296_29615

def plastic_cups : ℕ := 284
def ceramic_cups : ℕ := 284

theorem total_cups_sold : plastic_cups + ceramic_cups = 568 := by
  sorry

end NUMINAMATH_CALUDE_total_cups_sold_l296_29615


namespace NUMINAMATH_CALUDE_broken_line_length_formula_l296_29640

/-- Given an acute angle α and a point A₁ on one of its sides, we repeatedly drop perpendiculars
    to form an infinite broken line. This function represents the length of that line. -/
noncomputable def broken_line_length (α : Real) (m : Real) : Real :=
  m / (1 - Real.cos α)

/-- Theorem stating that the length of the infinite broken line formed by repeatedly dropping
    perpendiculars in an acute angle is equal to m / (1 - cos(α)), where m is the length of
    the first perpendicular and α is the magnitude of the angle. -/
theorem broken_line_length_formula (α : Real) (m : Real) 
    (h_acute : 0 < α ∧ α < Real.pi / 2) 
    (h_positive : m > 0) : 
  broken_line_length α m = m / (1 - Real.cos α) := by
  sorry

end NUMINAMATH_CALUDE_broken_line_length_formula_l296_29640


namespace NUMINAMATH_CALUDE_smallest_positive_real_l296_29625

theorem smallest_positive_real : ∃ (x : ℝ), x > 0 ∧ x + 1 > 1 * x ∧ ∀ (y : ℝ), y > 0 ∧ y + 1 > 1 * y → x ≤ y :=
sorry

end NUMINAMATH_CALUDE_smallest_positive_real_l296_29625


namespace NUMINAMATH_CALUDE_parabola_symmetry_l296_29611

/-- Given two parabolas, prove that they are symmetrical about the x-axis -/
theorem parabola_symmetry (x : ℝ) : 
  let f (x : ℝ) := (x - 1)^2 + 3
  let g (x : ℝ) := -(x - 1)^2 - 3
  ∀ x, f x = -g x := by
  sorry

end NUMINAMATH_CALUDE_parabola_symmetry_l296_29611


namespace NUMINAMATH_CALUDE_angle_in_fourth_quadrant_l296_29626

def angle : Int := -1120

def is_coterminal (a b : Int) : Prop :=
  ∃ k : Int, a = b + k * 360

def in_fourth_quadrant (a : Int) : Prop :=
  ∃ b : Int, is_coterminal a b ∧ 270 ≤ b ∧ b < 360

theorem angle_in_fourth_quadrant : in_fourth_quadrant angle := by
  sorry

end NUMINAMATH_CALUDE_angle_in_fourth_quadrant_l296_29626


namespace NUMINAMATH_CALUDE_sin_sum_special_angles_l296_29604

theorem sin_sum_special_angles : 
  Real.sin (Real.arcsin (4/5) + Real.arctan (Real.sqrt 3)) = (2 + 3 * Real.sqrt 3) / 10 := by
  sorry

end NUMINAMATH_CALUDE_sin_sum_special_angles_l296_29604


namespace NUMINAMATH_CALUDE_yoongis_pets_l296_29603

theorem yoongis_pets (dogs : ℕ) (cats : ℕ) : dogs = 5 → cats = 2 → dogs + cats = 7 := by
  sorry

end NUMINAMATH_CALUDE_yoongis_pets_l296_29603


namespace NUMINAMATH_CALUDE_exists_subset_with_unique_sum_representation_l296_29698

-- Define the property for the subset X
def has_unique_sum_representation (X : Set ℤ) : Prop :=
  ∀ n : ℤ, ∃! (p : ℤ × ℤ), p.1 ∈ X ∧ p.2 ∈ X ∧ p.1 + 2 * p.2 = n

-- Theorem statement
theorem exists_subset_with_unique_sum_representation :
  ∃ X : Set ℤ, has_unique_sum_representation X :=
sorry

end NUMINAMATH_CALUDE_exists_subset_with_unique_sum_representation_l296_29698


namespace NUMINAMATH_CALUDE_hyperbolic_identity_l296_29643

theorem hyperbolic_identity (θ : ℝ) (h : Real.cosh θ = 5) :
  (Real.sinh θ - 2)^2 + 25 / (Real.sinh θ - 2)^2 = 287/10 + 4 * Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_hyperbolic_identity_l296_29643


namespace NUMINAMATH_CALUDE_circle_radius_tripled_area_l296_29649

theorem circle_radius_tripled_area (r : ℝ) : r > 0 →
  (π * (r + 3)^2 = 3 * π * r^2) → r = (3 * (1 + Real.sqrt 3)) / 2 := by
  sorry

end NUMINAMATH_CALUDE_circle_radius_tripled_area_l296_29649


namespace NUMINAMATH_CALUDE_tuesday_max_available_l296_29692

-- Define the days of the week
inductive Day
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday

-- Define the people
inductive Person
  | Anna
  | Bill
  | Carl
  | Dana
  | Evan

-- Define a function to represent availability
def isAvailable (p : Person) (d : Day) : Bool :=
  match p, d with
  | Person.Anna, Day.Monday => false
  | Person.Anna, Day.Tuesday => true
  | Person.Anna, Day.Wednesday => false
  | Person.Anna, Day.Thursday => true
  | Person.Anna, Day.Friday => false
  | Person.Bill, Day.Monday => true
  | Person.Bill, Day.Tuesday => false
  | Person.Bill, Day.Wednesday => true
  | Person.Bill, Day.Thursday => false
  | Person.Bill, Day.Friday => true
  | Person.Carl, Day.Monday => false
  | Person.Carl, Day.Tuesday => false
  | Person.Carl, Day.Wednesday => true
  | Person.Carl, Day.Thursday => true
  | Person.Carl, Day.Friday => false
  | Person.Dana, Day.Monday => true
  | Person.Dana, Day.Tuesday => true
  | Person.Dana, Day.Wednesday => false
  | Person.Dana, Day.Thursday => false
  | Person.Dana, Day.Friday => true
  | Person.Evan, Day.Monday => false
  | Person.Evan, Day.Tuesday => true
  | Person.Evan, Day.Wednesday => false
  | Person.Evan, Day.Thursday => true
  | Person.Evan, Day.Friday => true

-- Count available people for a given day
def countAvailable (d : Day) : Nat :=
  List.length (List.filter (fun p => isAvailable p d) [Person.Anna, Person.Bill, Person.Carl, Person.Dana, Person.Evan])

-- Theorem: Tuesday has the maximum number of available people
theorem tuesday_max_available :
  ∀ d : Day, countAvailable Day.Tuesday ≥ countAvailable d :=
by
  sorry

end NUMINAMATH_CALUDE_tuesday_max_available_l296_29692


namespace NUMINAMATH_CALUDE_solve_scooter_problem_l296_29651

def scooter_problem (C : ℝ) (repair_percentage : ℝ) (profit_percentage : ℝ) (profit : ℝ) : Prop :=
  let repair_cost := repair_percentage * C
  let selling_price := (1 + profit_percentage) * C
  selling_price - C = profit ∧ 
  repair_cost = 550

theorem solve_scooter_problem :
  ∃ C : ℝ, scooter_problem C 0.1 0.2 1100 :=
sorry

end NUMINAMATH_CALUDE_solve_scooter_problem_l296_29651


namespace NUMINAMATH_CALUDE_sum_of_digits_6_11_l296_29663

def tens_digit (n : ℕ) : ℕ := (n / 10) % 10

def ones_digit (n : ℕ) : ℕ := n % 10

theorem sum_of_digits_6_11 : 
  tens_digit (6^11) + ones_digit (6^11) = 13 := by sorry

end NUMINAMATH_CALUDE_sum_of_digits_6_11_l296_29663


namespace NUMINAMATH_CALUDE_chords_from_nine_points_l296_29642

/-- The number of different chords that can be drawn by connecting two points 
    out of n points on the circumference of a circle -/
def num_chords (n : ℕ) : ℕ := n.choose 2

/-- Theorem stating that the number of chords from 9 points is 36 -/
theorem chords_from_nine_points : num_chords 9 = 36 := by
  sorry

end NUMINAMATH_CALUDE_chords_from_nine_points_l296_29642


namespace NUMINAMATH_CALUDE_hcl_formed_l296_29674

-- Define the chemical reaction
structure Reaction where
  ch4 : ℕ
  cl2 : ℕ
  ccl4 : ℕ
  hcl : ℕ

-- Define the balanced equation
def balanced_equation (r : Reaction) : Prop :=
  r.ch4 = 1 ∧ r.cl2 = 4 ∧ r.ccl4 = 1 ∧ r.hcl = 4

-- Define the given amounts of reactants
def given_reactants (r : Reaction) : Prop :=
  r.ch4 = 1 ∧ r.cl2 = 4

-- Theorem: Given the reactants and balanced equation, prove that 4 moles of HCl are formed
theorem hcl_formed (r : Reaction) 
  (h1 : balanced_equation r) 
  (h2 : given_reactants r) : 
  r.hcl = 4 := by
  sorry


end NUMINAMATH_CALUDE_hcl_formed_l296_29674


namespace NUMINAMATH_CALUDE_subset_condition_disjoint_condition_l296_29675

-- Define the sets A and S
def A : Set ℝ := {x : ℝ | -1 ≤ x ∧ x ≤ 7}
def S (k : ℝ) : Set ℝ := {x : ℝ | k + 1 ≤ x ∧ x ≤ 2*k - 1}

-- Theorem for condition (1)
theorem subset_condition (k : ℝ) : A ⊇ S k ↔ k ≤ 4 := by sorry

-- Theorem for condition (2)
theorem disjoint_condition (k : ℝ) : A ∩ S k = ∅ ↔ k < 2 ∨ k > 6 := by sorry

end NUMINAMATH_CALUDE_subset_condition_disjoint_condition_l296_29675


namespace NUMINAMATH_CALUDE_books_on_shelf_l296_29683

/-- The number of books remaining on a shelf after some are removed. -/
def booksRemaining (initial : ℝ) (removed : ℝ) : ℝ :=
  initial - removed

theorem books_on_shelf (initial : ℝ) (removed : ℝ) :
  initial ≥ removed →
  booksRemaining initial removed = initial - removed :=
by
  sorry

end NUMINAMATH_CALUDE_books_on_shelf_l296_29683


namespace NUMINAMATH_CALUDE_cube_cutting_theorem_l296_29629

/-- A plane in 3D space --/
structure Plane where
  normal : ℝ × ℝ × ℝ
  distance : ℝ

/-- A part of a cube resulting from cuts --/
structure CubePart where
  points : Set (ℝ × ℝ × ℝ)

/-- Function to calculate the maximum distance between any two points in a set --/
def maxDistance (s : Set (ℝ × ℝ × ℝ)) : ℝ := sorry

/-- Function to cut a unit cube with given planes --/
def cutCube (planes : List Plane) : List CubePart := sorry

theorem cube_cutting_theorem :
  (∃ (planes : List Plane), planes.length = 4 ∧ 
    ∀ part ∈ cutCube planes, maxDistance part.points < 4/5) ∧
  (¬ ∃ (planes : List Plane), planes.length = 4 ∧ 
    ∀ part ∈ cutCube planes, maxDistance part.points < 4/7) := by
  sorry

end NUMINAMATH_CALUDE_cube_cutting_theorem_l296_29629


namespace NUMINAMATH_CALUDE_inequality_proof_l296_29666

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (habc : a * b * c = 1) :
  (a^3 / (a^3 + 2*b^2)) + (b^3 / (b^3 + 2*c^2)) + (c^3 / (c^3 + 2*a^2)) ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l296_29666


namespace NUMINAMATH_CALUDE_prism_surface_area_l296_29647

/-- A rectangular prism formed by unit cubes -/
structure RectangularPrism where
  length : ℕ
  width : ℕ
  height : ℕ

/-- The volume of a rectangular prism -/
def volume (p : RectangularPrism) : ℕ :=
  p.length * p.width * p.height

/-- The surface area of a rectangular prism -/
def surfaceArea (p : RectangularPrism) : ℕ :=
  2 * (p.length * p.width + p.width * p.height + p.height * p.length)

/-- The number of unpainted cubes in a prism -/
def unpaintedCubes (p : RectangularPrism) : ℕ :=
  (p.length - 2) * (p.width - 2) * (p.height - 2)

theorem prism_surface_area :
  ∃ (p : RectangularPrism),
    volume p = 120 ∧
    unpaintedCubes p = 24 ∧
    surfaceArea p = 148 := by
  sorry

end NUMINAMATH_CALUDE_prism_surface_area_l296_29647


namespace NUMINAMATH_CALUDE_child_ticket_cost_l296_29612

/-- Proves that the cost of a child ticket is 1 dollar given the conditions of the problem -/
theorem child_ticket_cost
  (adult_ticket_cost : ℕ)
  (total_attendees : ℕ)
  (total_revenue : ℕ)
  (child_attendees : ℕ)
  (h1 : adult_ticket_cost = 8)
  (h2 : total_attendees = 22)
  (h3 : total_revenue = 50)
  (h4 : child_attendees = 18) :
  let adult_attendees : ℕ := total_attendees - child_attendees
  let child_ticket_cost : ℚ := (total_revenue - adult_ticket_cost * adult_attendees) / child_attendees
  child_ticket_cost = 1 := by sorry

end NUMINAMATH_CALUDE_child_ticket_cost_l296_29612


namespace NUMINAMATH_CALUDE_seashells_given_to_sam_l296_29662

/-- Given that Joan found a certain number of seashells and has some left after giving some to Sam,
    prove that the number of seashells given to Sam is the difference between the initial and remaining amounts. -/
theorem seashells_given_to_sam 
  (initial : ℕ) 
  (remaining : ℕ) 
  (h1 : initial = 70) 
  (h2 : remaining = 27) 
  (h3 : remaining < initial) : 
  initial - remaining = 43 := by
sorry

end NUMINAMATH_CALUDE_seashells_given_to_sam_l296_29662


namespace NUMINAMATH_CALUDE_marble_jar_count_l296_29636

theorem marble_jar_count (r : ℚ) : 
  let b := r / 1.3
  let y := 1.5 * r
  r + b + y = 42.5 * r / 13 := by sorry

end NUMINAMATH_CALUDE_marble_jar_count_l296_29636


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l296_29622

-- Define the function f
def f (x : ℝ) : ℝ := x^2

-- Define the set B
def B : Set ℝ := {1, 2}

-- Theorem statement
theorem intersection_of_A_and_B 
  (A : Set ℝ) 
  (h1 : ∀ y ∈ B, ∃ x ∈ A, f x = y) :
  (A ∩ B = ∅) ∨ (A ∩ B = {1}) :=
sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l296_29622


namespace NUMINAMATH_CALUDE_sandy_age_l296_29614

theorem sandy_age :
  ∀ (S M J : ℕ),
    S = M - 14 →
    J = S + 6 →
    9 * S = 7 * M →
    6 * S = 5 * J →
    S = 49 :=
by
  sorry

end NUMINAMATH_CALUDE_sandy_age_l296_29614


namespace NUMINAMATH_CALUDE_gw_to_w_conversion_l296_29631

/-- Conversion factor from gigawatts to watts -/
def gw_to_w : ℝ := 1000000000

/-- The newly installed capacity in gigawatts -/
def installed_capacity : ℝ := 125

/-- Theorem stating that 125 gigawatts is equal to 1.25 × 10^11 watts -/
theorem gw_to_w_conversion :
  installed_capacity * gw_to_w = 1.25 * (10 : ℝ) ^ 11 := by sorry

end NUMINAMATH_CALUDE_gw_to_w_conversion_l296_29631


namespace NUMINAMATH_CALUDE_alcohol_mixture_theorem_alcohol_mixture_validity_l296_29628

/-- Proves that adding the calculated amount of alcohol results in the desired concentration -/
theorem alcohol_mixture_theorem (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c ≥ 0) (hd : d ≥ 0) 
  (hac : a ≠ c) (had : a ≠ d) (hcd : c ≠ d) :
  let x := b * (d - c) / (a - d)
  (b * c + x * a) / (b + x) = d :=
by sorry

/-- Proves that the solution is valid when d is between a and c -/
theorem alcohol_mixture_validity (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c ≥ 0) (hd : d ≥ 0) 
  (hac : a ≠ c) (had : a ≠ d) (hcd : c ≠ d) :
  (min a c < d ∧ d < max a c) → 
  let x := b * (d - c) / (a - d)
  x > 0 :=
by sorry

end NUMINAMATH_CALUDE_alcohol_mixture_theorem_alcohol_mixture_validity_l296_29628


namespace NUMINAMATH_CALUDE_distance_sum_is_18_l296_29694

/-- Given three points A, B, and D in a plane, prove that the sum of distances AD and BD is 18 -/
theorem distance_sum_is_18 (A B D : ℝ × ℝ) : 
  A = (16, 0) → B = (1, 1) → D = (4, 5) → 
  Real.sqrt ((A.1 - D.1)^2 + (A.2 - D.2)^2) + Real.sqrt ((B.1 - D.1)^2 + (B.2 - D.2)^2) = 18 := by
  sorry

end NUMINAMATH_CALUDE_distance_sum_is_18_l296_29694


namespace NUMINAMATH_CALUDE_max_value_of_E_l296_29686

theorem max_value_of_E (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h : a^5 + b^5 = a^3 + b^3) : 
  ∃ (M : ℝ), M = 1 ∧ ∀ x y, x > 0 → y > 0 → x^5 + y^5 = x^3 + y^3 → 
  x^2 - x*y + y^2 ≤ M :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_E_l296_29686


namespace NUMINAMATH_CALUDE_product_of_five_consecutive_integers_divisibility_l296_29665

theorem product_of_five_consecutive_integers_divisibility 
  (m : ℤ) 
  (k : ℤ) 
  (h1 : m = k * (k + 1) * (k + 2) * (k + 3) * (k + 4)) 
  (h2 : 11 ∣ m) : 
  (10 ∣ m) ∧ (22 ∣ m) ∧ (33 ∣ m) ∧ (55 ∣ m) ∧ ¬(∀ m, 66 ∣ m) :=
by sorry

end NUMINAMATH_CALUDE_product_of_five_consecutive_integers_divisibility_l296_29665


namespace NUMINAMATH_CALUDE_max_n_is_five_l296_29635

/-- A regular n-sided prism with lateral edge length equal to base side length -/
structure RegularPrism (n : ℕ) where
  -- n is the number of sides of the base
  -- lateral_edge is the length of the lateral edge
  -- base_side is the length of a side of the base
  (n_ge_3 : n ≥ 3)
  (lateral_edge : ℝ)
  (base_side : ℝ)
  (lateral_eq_base : lateral_edge = base_side)

/-- The maximum value of n for which a RegularPrism can exist -/
def max_n_regular_prism : ℕ := 5

/-- Theorem stating that 5 is the maximum value of n for which a RegularPrism can exist -/
theorem max_n_is_five :
  ∀ n : ℕ, n > max_n_regular_prism → ¬ (∃ p : RegularPrism n, True) :=
sorry

end NUMINAMATH_CALUDE_max_n_is_five_l296_29635


namespace NUMINAMATH_CALUDE_seashells_total_l296_29681

theorem seashells_total (sam_shells mary_shells : ℕ) 
  (h1 : sam_shells = 18) (h2 : mary_shells = 47) : 
  sam_shells + mary_shells = 65 := by
  sorry

end NUMINAMATH_CALUDE_seashells_total_l296_29681


namespace NUMINAMATH_CALUDE_integral_tan_cos_equality_l296_29652

open Real MeasureTheory Interval

theorem integral_tan_cos_equality : 
  ∫ x in (-1 : ℝ)..1, (tan x)^11 + (cos x)^21 = 2 * ∫ x in (0 : ℝ)..1, (cos x)^21 := by
  sorry

end NUMINAMATH_CALUDE_integral_tan_cos_equality_l296_29652


namespace NUMINAMATH_CALUDE_today_is_wednesday_l296_29699

-- Define the days of the week
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

def nextDay (d : DayOfWeek) : DayOfWeek :=
  match d with
  | DayOfWeek.Sunday => DayOfWeek.Monday
  | DayOfWeek.Monday => DayOfWeek.Tuesday
  | DayOfWeek.Tuesday => DayOfWeek.Wednesday
  | DayOfWeek.Wednesday => DayOfWeek.Thursday
  | DayOfWeek.Thursday => DayOfWeek.Friday
  | DayOfWeek.Friday => DayOfWeek.Saturday
  | DayOfWeek.Saturday => DayOfWeek.Sunday

def prevDay (d : DayOfWeek) : DayOfWeek :=
  match d with
  | DayOfWeek.Sunday => DayOfWeek.Saturday
  | DayOfWeek.Monday => DayOfWeek.Sunday
  | DayOfWeek.Tuesday => DayOfWeek.Monday
  | DayOfWeek.Wednesday => DayOfWeek.Tuesday
  | DayOfWeek.Thursday => DayOfWeek.Wednesday
  | DayOfWeek.Friday => DayOfWeek.Thursday
  | DayOfWeek.Saturday => DayOfWeek.Friday

def dayAfterTomorrow (d : DayOfWeek) : DayOfWeek := nextDay (nextDay d)

def distanceToSunday (d : DayOfWeek) : Nat :=
  match d with
  | DayOfWeek.Sunday => 0
  | DayOfWeek.Monday => 6
  | DayOfWeek.Tuesday => 5
  | DayOfWeek.Wednesday => 4
  | DayOfWeek.Thursday => 3
  | DayOfWeek.Friday => 2
  | DayOfWeek.Saturday => 1

theorem today_is_wednesday :
  ∃ (today : DayOfWeek),
    (dayAfterTomorrow today = prevDay today) ∧
    (distanceToSunday today = distanceToSunday (prevDay (nextDay today))) ∧
    (today = DayOfWeek.Wednesday) := by
  sorry


end NUMINAMATH_CALUDE_today_is_wednesday_l296_29699


namespace NUMINAMATH_CALUDE_square_perimeter_l296_29684

theorem square_perimeter (rectangle_perimeter : ℝ) (square_side : ℝ) : 
  (rectangle_perimeter + 4 * square_side) - rectangle_perimeter = 17 →
  4 * square_side = 34 := by
  sorry

end NUMINAMATH_CALUDE_square_perimeter_l296_29684


namespace NUMINAMATH_CALUDE_runners_meeting_point_l296_29653

/-- Represents the meeting point on the circular track -/
inductive MeetingPoint
| S -- Boundary of A and D
| A
| B
| C
| D

/-- Represents a runner on the circular track -/
structure Runner where
  startPosition : ℝ -- Position in meters from point S
  direction : Bool -- true for counterclockwise, false for clockwise
  distanceRun : ℝ -- Total distance run in meters

/-- Theorem stating where Alice and Bob meet on the circular track -/
theorem runners_meeting_point 
  (trackCircumference : ℝ)
  (alice : Runner)
  (bob : Runner)
  (h1 : trackCircumference = 60)
  (h2 : alice.startPosition = 0)
  (h3 : alice.direction = true)
  (h4 : alice.distanceRun = 7200)
  (h5 : bob.startPosition = 30)
  (h6 : bob.direction = false)
  (h7 : bob.distanceRun = alice.distanceRun) :
  MeetingPoint.S = 
    (let aliceFinalPosition := alice.startPosition + (alice.distanceRun % trackCircumference)
     let bobFinalPosition := bob.startPosition - (bob.distanceRun % trackCircumference)
     if aliceFinalPosition = bobFinalPosition 
     then MeetingPoint.S 
     else MeetingPoint.A) :=
by sorry

end NUMINAMATH_CALUDE_runners_meeting_point_l296_29653


namespace NUMINAMATH_CALUDE_square_area_error_l296_29658

theorem square_area_error (a : ℝ) (h : a > 0) : 
  let measured_side := a * 1.05
  let actual_area := a ^ 2
  let calculated_area := measured_side ^ 2
  let area_error := (calculated_area - actual_area) / actual_area
  area_error = 0.1025 := by sorry

end NUMINAMATH_CALUDE_square_area_error_l296_29658


namespace NUMINAMATH_CALUDE_arithmetic_sqrt_of_nine_l296_29641

-- Define the arithmetic square root function
noncomputable def arithmetic_sqrt (x : ℝ) : ℝ :=
  Real.sqrt x

-- Theorem statement
theorem arithmetic_sqrt_of_nine : arithmetic_sqrt 9 = 3 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sqrt_of_nine_l296_29641


namespace NUMINAMATH_CALUDE_min_value_sum_reciprocals_l296_29676

theorem min_value_sum_reciprocals (p q r s t u : ℝ) 
  (pos_p : p > 0) (pos_q : q > 0) (pos_r : r > 0) 
  (pos_s : s > 0) (pos_t : t > 0) (pos_u : u > 0)
  (sum_eq_10 : p + q + r + s + t + u = 10) : 
  (1/p + 9/q + 4/r + 1/s + 16/t + 25/u) ≥ 25.6 := by
  sorry

end NUMINAMATH_CALUDE_min_value_sum_reciprocals_l296_29676


namespace NUMINAMATH_CALUDE_min_value_theorem_equality_exists_l296_29650

theorem min_value_theorem (x : ℝ) (h : x > 0) : x^2 + 12*x + 108/x^4 ≥ 42 := by
  sorry

theorem equality_exists : ∃ x : ℝ, x > 0 ∧ x^2 + 12*x + 108/x^4 = 42 := by
  sorry

end NUMINAMATH_CALUDE_min_value_theorem_equality_exists_l296_29650


namespace NUMINAMATH_CALUDE_complement_of_P_P_subset_Q_range_P_inter_Q_eq_Q_range_final_range_of_m_l296_29630

-- Define sets P and Q
def P : Set ℝ := {x | -2 ≤ x ∧ x ≤ 10}
def Q (m : ℝ) : Set ℝ := {x | 1 - m ≤ x ∧ x ≤ 1 + m}

-- Theorem for the complement of P
theorem complement_of_P : (Set.univ \ P) = {x | x < -2 ∨ x > 10} := by sorry

-- Theorem for P being a subset of Q
theorem P_subset_Q_range (m : ℝ) : P ⊆ Q m ↔ m ≥ 9 := by sorry

-- Theorem for intersection of P and Q equals Q
theorem P_inter_Q_eq_Q_range (m : ℝ) : P ∩ Q m = Q m ↔ m ≤ 9 := by sorry

-- Theorem for the final range of m satisfying both conditions
theorem final_range_of_m : 
  {m : ℝ | P ⊆ Q m ∧ P ∩ Q m = Q m} = {m : ℝ | 9 ≤ m ∧ m ≤ 9} := by sorry

end NUMINAMATH_CALUDE_complement_of_P_P_subset_Q_range_P_inter_Q_eq_Q_range_final_range_of_m_l296_29630


namespace NUMINAMATH_CALUDE_trig_identity_l296_29609

theorem trig_identity (α : Real) (h : Real.sin (α - π / 12) = 1 / 3) :
  Real.cos (α + 17 * π / 12) = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_trig_identity_l296_29609


namespace NUMINAMATH_CALUDE_common_tangents_count_l296_29656

/-- Two circles in a plane -/
structure CirclePair :=
  (c1 c2 : Set ℝ × ℝ)
  (r1 r2 : ℝ)
  (h_unequal : r1 ≠ r2)

/-- The number of common tangents for a pair of circles -/
def num_common_tangents (cp : CirclePair) : ℕ := sorry

/-- Theorem stating that the number of common tangents for unequal circles is always 0, 1, 2, 3, or 4 -/
theorem common_tangents_count (cp : CirclePair) :
  num_common_tangents cp = 0 ∨
  num_common_tangents cp = 1 ∨
  num_common_tangents cp = 2 ∨
  num_common_tangents cp = 3 ∨
  num_common_tangents cp = 4 :=
sorry

end NUMINAMATH_CALUDE_common_tangents_count_l296_29656


namespace NUMINAMATH_CALUDE_max_value_of_four_numbers_l296_29621

theorem max_value_of_four_numbers
  (a b c d : ℝ)
  (h_positive : 0 < d ∧ 0 < c ∧ 0 < b ∧ 0 < a)
  (h_order : d ≤ c ∧ c ≤ b ∧ b ≤ a)
  (h_sum : a + b + c + d = 4)
  (h_sum_squares : a^2 + b^2 + c^2 + d^2 = 8) :
  a ≤ 1 + Real.sqrt 3 ∧ ∃ (a₀ b₀ c₀ d₀ : ℝ),
    0 < d₀ ∧ d₀ ≤ c₀ ∧ c₀ ≤ b₀ ∧ b₀ ≤ a₀ ∧
    a₀ + b₀ + c₀ + d₀ = 4 ∧
    a₀^2 + b₀^2 + c₀^2 + d₀^2 = 8 ∧
    a₀ = 1 + Real.sqrt 3 :=
by
  sorry

end NUMINAMATH_CALUDE_max_value_of_four_numbers_l296_29621


namespace NUMINAMATH_CALUDE_wire_poles_problem_l296_29695

theorem wire_poles_problem (wire_length : ℝ) (distance_increase : ℝ) : 
  wire_length = 5000 →
  distance_increase = 1.25 →
  ∃ (n : ℕ), 
    n > 1 ∧
    wire_length / (n - 1 : ℝ) + distance_increase = wire_length / (n - 2 : ℝ) ∧
    n = 65 := by
  sorry

end NUMINAMATH_CALUDE_wire_poles_problem_l296_29695


namespace NUMINAMATH_CALUDE_consecutive_integers_divisible_by_three_l296_29607

theorem consecutive_integers_divisible_by_three (a b c d e : ℕ) : 
  (70 < a) ∧ (a < 100) ∧
  (b = a + 1) ∧ (c = b + 1) ∧ (d = c + 1) ∧ (e = d + 1) ∧
  (a % 3 = 0) ∧ (b % 3 = 0) ∧ (c % 3 = 0) ∧ (d % 3 = 0) ∧ (e % 3 = 0) →
  e = 84 := by
sorry

end NUMINAMATH_CALUDE_consecutive_integers_divisible_by_three_l296_29607


namespace NUMINAMATH_CALUDE_two_true_statements_l296_29632

theorem two_true_statements 
  (x y a b : ℝ) 
  (h_nonzero : x ≠ 0 ∧ y ≠ 0 ∧ a ≠ 0 ∧ b ≠ 0) 
  (h_x_lt_a : x < a) 
  (h_y_lt_b : y < b) 
  (h_positive : x > 0 ∧ y > 0 ∧ a > 0 ∧ b > 0) : 
  ∃! n : ℕ, n = 2 ∧ n = (
    (if x + y < a + b then 1 else 0) +
    (if x - y < a - b then 1 else 0) +
    (if x * y < a * b then 1 else 0) +
    (if (x / y < a / b → x / y < a / b) then 1 else 0)
  ) := by sorry

end NUMINAMATH_CALUDE_two_true_statements_l296_29632


namespace NUMINAMATH_CALUDE_number_of_sets_l296_29613

/-- The number of flowers in each set -/
def flowers_per_set : ℕ := 90

/-- The total number of flowers bought -/
def total_flowers : ℕ := 270

/-- Theorem: The number of sets of flowers bought is 3 -/
theorem number_of_sets : total_flowers / flowers_per_set = 3 := by
  sorry

end NUMINAMATH_CALUDE_number_of_sets_l296_29613


namespace NUMINAMATH_CALUDE_johns_money_proof_l296_29690

/-- Calculates John's initial amount of money given his purchases and remaining money -/
def johns_initial_money (roast_cost vegetables_cost remaining_money : ℕ) : ℕ :=
  roast_cost + vegetables_cost + remaining_money

theorem johns_money_proof (roast_cost vegetables_cost remaining_money : ℕ) 
  (h1 : roast_cost = 17)
  (h2 : vegetables_cost = 11)
  (h3 : remaining_money = 72) :
  johns_initial_money roast_cost vegetables_cost remaining_money = 100 := by
  sorry

end NUMINAMATH_CALUDE_johns_money_proof_l296_29690


namespace NUMINAMATH_CALUDE_complex_number_location_l296_29624

theorem complex_number_location :
  let z : ℂ := (-2 + 3*I) / (3 - 4*I)
  (z.re < 0 ∧ z.im > 0) := by sorry

end NUMINAMATH_CALUDE_complex_number_location_l296_29624


namespace NUMINAMATH_CALUDE_will_toy_purchase_l296_29600

theorem will_toy_purchase (initial_amount : ℕ) (spent_amount : ℕ) (toy_cost : ℕ) : 
  initial_amount = 83 → spent_amount = 47 → toy_cost = 4 →
  (initial_amount - spent_amount) / toy_cost = 9 := by
sorry

end NUMINAMATH_CALUDE_will_toy_purchase_l296_29600


namespace NUMINAMATH_CALUDE_division_problem_l296_29667

theorem division_problem (a b : ℕ) : 
  a > 0 ∧ b > 0 ∧ 
  a = 13 * b + 6 ∧ 
  a + b + 13 + 6 = 137 → 
  a = 110 ∧ b = 8 := by
sorry

end NUMINAMATH_CALUDE_division_problem_l296_29667


namespace NUMINAMATH_CALUDE_produce_worth_l296_29697

theorem produce_worth (asparagus_bundles : ℕ) (asparagus_price : ℚ)
                      (grape_boxes : ℕ) (grape_price : ℚ)
                      (apple_count : ℕ) (apple_price : ℚ) :
  asparagus_bundles = 60 ∧ asparagus_price = 3 ∧
  grape_boxes = 40 ∧ grape_price = 5/2 ∧
  apple_count = 700 ∧ apple_price = 1/2 →
  asparagus_bundles * asparagus_price +
  grape_boxes * grape_price +
  apple_count * apple_price = 630 :=
by sorry

end NUMINAMATH_CALUDE_produce_worth_l296_29697


namespace NUMINAMATH_CALUDE_range_of_m_in_fractional_equation_l296_29680

/-- Given a fractional equation with a non-negative solution, prove the range of m -/
theorem range_of_m_in_fractional_equation (m : ℝ) : 
  (∃ x : ℝ, x ≥ 0 ∧ m / (x - 2) + 1 = x / (2 - x)) → 
  m ≤ 2 ∧ m ≠ -2 := by
  sorry

end NUMINAMATH_CALUDE_range_of_m_in_fractional_equation_l296_29680


namespace NUMINAMATH_CALUDE_nine_b_value_l296_29618

theorem nine_b_value (a b : ℚ) (h1 : 8 * a + 3 * b = 0) (h2 : b - 3 = a) : 9 * b = 216 / 11 := by
  sorry

end NUMINAMATH_CALUDE_nine_b_value_l296_29618


namespace NUMINAMATH_CALUDE_coffee_shop_run_time_l296_29645

/-- Represents the time in minutes to run a given distance at a constant pace -/
def runTime (distance : ℝ) (pace : ℝ) : ℝ := distance * pace

theorem coffee_shop_run_time :
  let parkDistance : ℝ := 5
  let parkTime : ℝ := 30
  let coffeeShopDistance : ℝ := 2
  let pace : ℝ := parkTime / parkDistance
  runTime coffeeShopDistance pace = 12 := by sorry

end NUMINAMATH_CALUDE_coffee_shop_run_time_l296_29645


namespace NUMINAMATH_CALUDE_sam_reading_speed_l296_29655

/-- Proves that given Dustin can read 75 pages in an hour and reads 34 more pages than Sam in 40 minutes, Sam can read 72 pages in an hour. -/
theorem sam_reading_speed (dustin_pages_per_hour : ℕ) (extra_pages : ℕ) : 
  dustin_pages_per_hour = 75 → 
  dustin_pages_per_hour * (40 : ℚ) / 60 - extra_pages = 
    (72 : ℚ) * (40 : ℚ) / 60 → 
  extra_pages = 34 →
  72 = 72 := by sorry

end NUMINAMATH_CALUDE_sam_reading_speed_l296_29655


namespace NUMINAMATH_CALUDE_triangle_angles_l296_29691

theorem triangle_angles (a b c : ℝ) (A B C : ℝ) : 
  0 < A ∧ 0 < B ∧ 0 < C ∧ 
  A + B + C = π ∧
  a = 2 * b * Real.cos C ∧
  Real.sin A * Real.sin (B / 2 + C) = Real.sin C * (Real.sin (B / 2) + Real.sin A) →
  A = 5 * π / 9 ∧ B = 2 * π / 9 ∧ C = 2 * π / 9 := by
sorry

end NUMINAMATH_CALUDE_triangle_angles_l296_29691


namespace NUMINAMATH_CALUDE_exists_multiple_in_ascending_sequence_l296_29679

/-- Definition of an ascending sequence -/
def IsAscending (a : ℕ → ℕ) : Prop :=
  ∀ n, a n < a (n + 1) ∧ a (2 * n) = 2 * a n

/-- Theorem: For any ascending sequence of positive integers and prime p > a₁,
    there exists a term in the sequence divisible by p -/
theorem exists_multiple_in_ascending_sequence
    (a : ℕ → ℕ)
    (h_ascending : IsAscending a)
    (h_positive : ∀ n, a n > 0)
    (p : ℕ)
    (h_prime : Nat.Prime p)
    (h_p_gt_a1 : p > a 1) :
    ∃ n, p ∣ a n := by
  sorry

end NUMINAMATH_CALUDE_exists_multiple_in_ascending_sequence_l296_29679


namespace NUMINAMATH_CALUDE_pizza_buffet_l296_29608

theorem pizza_buffet (A B C : ℕ) (h1 : ∃ x : ℕ, A = x * B) 
  (h2 : B * 8 = C) (h3 : A + B + C = 360) : 
  ∃ x : ℕ, A = 351 * B := by
  sorry

end NUMINAMATH_CALUDE_pizza_buffet_l296_29608


namespace NUMINAMATH_CALUDE_intersection_kth_element_l296_29693

-- Define set A
def A : Set ℕ := {n | ∃ m : ℕ, n = m * (m + 1)}

-- Define set B
def B : Set ℕ := {n | ∃ m : ℕ, n = 3 * m - 1}

-- Define the intersection of A and B
def A_intersect_B : Set ℕ := A ∩ B

-- Define the kth element of the intersection
def a (k : ℕ) : ℕ := 9 * k^2 - 9 * k + 2

-- Theorem statement
theorem intersection_kth_element (k : ℕ) : 
  a k ∈ A_intersect_B ∧ 
  (∀ n ∈ A_intersect_B, n < a k → 
    ∃ j < k, n = a j) ∧
  (∀ n ∈ A_intersect_B, n ≠ a k → 
    ∃ j ≠ k, n = a j) :=
sorry

end NUMINAMATH_CALUDE_intersection_kth_element_l296_29693


namespace NUMINAMATH_CALUDE_function_determination_l296_29605

/-- Given a function f(x) = x³ - ax + b where x ∈ ℝ, 
    and the tangent line to f(x) at (1, f(1)) is 2x - y + 3 = 0,
    prove that f(x) = x³ - x + 5 -/
theorem function_determination (a b : ℝ) :
  (∀ x : ℝ, ∃ f : ℝ → ℝ, f x = x^3 - a*x + b) →
  (∃ f : ℝ → ℝ, ∀ x : ℝ, f x = x^3 - a*x + b ∧ 
    (2 * 1 - f 1 + 3 = 0) ∧
    (∀ x : ℝ, (2 * x - f x + 3 = 0) → x = 1)) →
  (∀ x : ℝ, ∃ f : ℝ → ℝ, f x = x^3 - x + 5) :=
by sorry

end NUMINAMATH_CALUDE_function_determination_l296_29605


namespace NUMINAMATH_CALUDE_x_plus_2y_equals_100_l296_29610

theorem x_plus_2y_equals_100 (x y : ℝ) (h1 : y = 25) (h2 : x = 50) : x + 2*y = 100 := by
  sorry

end NUMINAMATH_CALUDE_x_plus_2y_equals_100_l296_29610


namespace NUMINAMATH_CALUDE_second_derivative_y_l296_29602

noncomputable def y (x : ℝ) : ℝ := x^2 * Real.log (1 + Real.sin x)

theorem second_derivative_y (x : ℝ) :
  (deriv (deriv y)) x = 2 * Real.log (1 + Real.sin x) + (4 * x * Real.cos x - x^2) / (1 + Real.sin x) :=
by sorry

end NUMINAMATH_CALUDE_second_derivative_y_l296_29602


namespace NUMINAMATH_CALUDE_conjugate_complex_magnitude_l296_29633

theorem conjugate_complex_magnitude (α β : ℂ) : 
  (∃ (x y : ℝ), α = x + y * Complex.I ∧ β = x - y * Complex.I) →  -- conjugate complex numbers
  (∃ (r : ℝ), α / β^3 = r) →  -- α/β³ is real
  Complex.abs (α - β) = 4 →  -- |α - β| = 4
  Complex.abs α = 4 * Real.sqrt 3 / 3 :=  -- |α| = 4√3/3
by sorry

end NUMINAMATH_CALUDE_conjugate_complex_magnitude_l296_29633


namespace NUMINAMATH_CALUDE_mans_age_twice_sons_l296_29654

/-- 
Proves that the number of years until a man's age is twice his son's age is 2,
given that the man is 24 years older than his son and the son's present age is 22 years.
-/
theorem mans_age_twice_sons (
  son_age : ℕ) (man_age : ℕ) (years : ℕ) : 
  son_age = 22 →
  man_age = son_age + 24 →
  man_age + years = 2 * (son_age + years) →
  years = 2 :=
by sorry

end NUMINAMATH_CALUDE_mans_age_twice_sons_l296_29654


namespace NUMINAMATH_CALUDE_new_acute_angle_l296_29616

theorem new_acute_angle (initial_angle : ℝ) (net_rotation : ℝ) : 
  initial_angle = 60 → net_rotation = 90 → 
  (180 - (initial_angle + net_rotation)) % 180 = 30 := by
sorry

end NUMINAMATH_CALUDE_new_acute_angle_l296_29616


namespace NUMINAMATH_CALUDE_ned_good_games_l296_29670

/-- The number of good games Ned ended up with -/
def good_games (games_from_friend : ℕ) (games_from_garage_sale : ℕ) (non_working_games : ℕ) : ℕ :=
  games_from_friend + games_from_garage_sale - non_working_games

/-- Proof that Ned ended up with 3 good games -/
theorem ned_good_games :
  good_games 50 27 74 = 3 := by
  sorry

end NUMINAMATH_CALUDE_ned_good_games_l296_29670


namespace NUMINAMATH_CALUDE_probability_product_one_five_dice_l296_29687

def standard_die := Finset.range 6

def probability_of_one (n : ℕ) : ℚ :=
  (1 : ℚ) / 6

theorem probability_product_one_five_dice :
  (probability_of_one 5)^5 = 1 / 7776 := by
  sorry

end NUMINAMATH_CALUDE_probability_product_one_five_dice_l296_29687


namespace NUMINAMATH_CALUDE_additional_triangles_for_hexagon_l296_29678

/-- The number of vertices in a hexagon -/
def hexagon_vertices : ℕ := 6

/-- The number of sides in a hexagon -/
def hexagon_sides : ℕ := 6

/-- The number of triangles in the original shape -/
def original_triangles : ℕ := 36

/-- The number of additional triangles needed for each vertex -/
def triangles_per_vertex : ℕ := 2

/-- The number of additional triangles needed for each side -/
def triangles_per_side : ℕ := 1

/-- The theorem stating the smallest number of additional triangles needed -/
theorem additional_triangles_for_hexagon :
  hexagon_vertices * triangles_per_vertex + hexagon_sides * triangles_per_side = 18 :=
sorry

end NUMINAMATH_CALUDE_additional_triangles_for_hexagon_l296_29678


namespace NUMINAMATH_CALUDE_boat_speed_theorem_l296_29606

/-- Represents the speed of a boat in a stream -/
structure BoatInStream where
  boatSpeed : ℝ  -- Speed of the boat in still water
  streamSpeed : ℝ  -- Speed of the stream

/-- Calculates the effective speed of the boat -/
def BoatInStream.effectiveSpeed (b : BoatInStream) (upstream : Bool) : ℝ :=
  if upstream then b.boatSpeed - b.streamSpeed else b.boatSpeed + b.streamSpeed

/-- Theorem: If the time taken to row upstream is twice the time taken to row downstream
    for the same distance, and the stream speed is 24, then the boat speed in still water is 72 -/
theorem boat_speed_theorem (b : BoatInStream) (distance : ℝ) 
    (h1 : b.streamSpeed = 24)
    (h2 : distance / b.effectiveSpeed true = 2 * (distance / b.effectiveSpeed false)) :
    b.boatSpeed = 72 := by
  sorry

#check boat_speed_theorem

end NUMINAMATH_CALUDE_boat_speed_theorem_l296_29606


namespace NUMINAMATH_CALUDE_store_pants_price_l296_29648

theorem store_pants_price (selling_price : ℝ) (price_difference : ℝ) (store_price : ℝ) : 
  selling_price = 34 →
  price_difference = 8 →
  store_price = selling_price - price_difference →
  store_price = 26 := by
sorry

end NUMINAMATH_CALUDE_store_pants_price_l296_29648


namespace NUMINAMATH_CALUDE_spring_length_theorem_l296_29671

/-- Represents the relationship between spring length and attached mass -/
def spring_length (x : ℝ) : ℝ :=
  0.3 * x + 6

/-- Theorem stating the relationship between spring length and attached mass -/
theorem spring_length_theorem (x : ℝ) :
  let initial_length : ℝ := 6
  let extension_rate : ℝ := 0.3
  spring_length x = initial_length + extension_rate * x :=
by
  sorry

#check spring_length_theorem

end NUMINAMATH_CALUDE_spring_length_theorem_l296_29671


namespace NUMINAMATH_CALUDE_line_equation_given_ellipse_midpoint_l296_29673

/-- The equation of a line that intersects an ellipse, given the midpoint of the intersection -/
theorem line_equation_given_ellipse_midpoint (x y : ℝ → ℝ) (l : Set (ℝ × ℝ)) :
  (∀ t, (x t)^2 / 36 + (y t)^2 / 9 = 1) →  -- Ellipse equation
  (∃ t₁ t₂, (x t₁, y t₁) ∈ l ∧ (x t₂, y t₂) ∈ l ∧ t₁ ≠ t₂) →  -- Line intersects ellipse at two points
  ((x t₁ + x t₂) / 2 = 4 ∧ (y t₁ + y t₂) / 2 = 2) →  -- Midpoint is (4,2)
  (∀ p, p ∈ l ↔ p.1 + 2 * p.2 - 8 = 0) :=  -- Line equation
by sorry

end NUMINAMATH_CALUDE_line_equation_given_ellipse_midpoint_l296_29673


namespace NUMINAMATH_CALUDE_dog_weight_problem_l296_29620

theorem dog_weight_problem (x y : ℝ) :
  -- Define the weights of the dogs
  let w₂ : ℝ := 31
  let w₃ : ℝ := 35
  let w₄ : ℝ := 33
  let w₅ : ℝ := y
  -- The average of the first 4 dogs equals the average of all 5 dogs
  (x + w₂ + w₃ + w₄) / 4 = (x + w₂ + w₃ + w₄ + w₅) / 5 →
  -- The weight of the fifth dog is 31 pounds
  y = 31 →
  -- The weight of the first dog is 25 pounds
  x = 25 := by
sorry

end NUMINAMATH_CALUDE_dog_weight_problem_l296_29620


namespace NUMINAMATH_CALUDE_parallel_lines_j_value_l296_29660

/-- Given two points on a line and another line equation, find the j-coordinate of the second point -/
theorem parallel_lines_j_value :
  let line1_point1 : ℝ × ℝ := (5, -6)
  let line1_point2 : ℝ × ℝ := (j, 29)
  let line2_slope : ℝ := 3 / 2
  let line2_equation (x y : ℝ) := 3 * x - 2 * y = 15
  ∀ j : ℝ,
    (line1_point2.2 - line1_point1.2) / (line1_point2.1 - line1_point1.1) = line2_slope →
    j = 85 / 3 :=
by
  sorry

end NUMINAMATH_CALUDE_parallel_lines_j_value_l296_29660


namespace NUMINAMATH_CALUDE_sum_of_squares_quadratic_roots_sum_of_squares_specific_quadratic_l296_29696

theorem sum_of_squares_quadratic_roots (a b c : ℝ) (h : a ≠ 0) :
  let r₁ := (-b + Real.sqrt (b^2 - 4*a*c)) / (2*a)
  let r₂ := (-b - Real.sqrt (b^2 - 4*a*c)) / (2*a)
  a * x^2 + b * x + c = 0 → r₁^2 + r₂^2 = (b^2 - 2*a*c) / a^2 :=
by sorry

theorem sum_of_squares_specific_quadratic :
  let r₁ := (16 + Real.sqrt 256) / 2
  let r₂ := (16 - Real.sqrt 256) / 2
  x^2 - 16*x + 4 = 0 → r₁^2 + r₂^2 = 248 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_squares_quadratic_roots_sum_of_squares_specific_quadratic_l296_29696


namespace NUMINAMATH_CALUDE_josh_shopping_cost_l296_29664

def film_cost : ℕ := 5
def book_cost : ℕ := 4
def cd_cost : ℕ := 3

def num_films : ℕ := 9
def num_books : ℕ := 4
def num_cds : ℕ := 6

theorem josh_shopping_cost : 
  (num_films * film_cost + num_books * book_cost + num_cds * cd_cost) = 79 := by
  sorry

end NUMINAMATH_CALUDE_josh_shopping_cost_l296_29664


namespace NUMINAMATH_CALUDE_average_of_three_numbers_l296_29672

theorem average_of_three_numbers (y : ℝ) : (14 + 23 + y) / 3 = 21 → y = 26 := by
  sorry

end NUMINAMATH_CALUDE_average_of_three_numbers_l296_29672


namespace NUMINAMATH_CALUDE_remainder_problem_l296_29688

theorem remainder_problem (n : ℕ) : n % 44 = 0 ∧ n / 44 = 432 → n % 34 = 20 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l296_29688


namespace NUMINAMATH_CALUDE_geometric_sequence_product_l296_29617

/-- A geometric sequence with positive terms where a₁ and a₉₉ are roots of x² - 10x + 16 = 0 -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  (∀ n, a n > 0) ∧ 
  (∃ r, ∀ n, a (n + 1) = r * a n) ∧
  (a 1 * a 99 = 16) ∧
  (a 1 + a 99 = 10)

/-- The product of specific terms in the geometric sequence equals 64 -/
theorem geometric_sequence_product (a : ℕ → ℝ) (h : GeometricSequence a) :
  a 20 * a 50 * a 80 = 64 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_product_l296_29617


namespace NUMINAMATH_CALUDE_x_wins_in_six_moves_l296_29619

/-- Represents a position on the infinite grid -/
structure Position :=
  (x : ℤ) (y : ℤ)

/-- Represents a player in the game -/
inductive Player
  | X
  | O

/-- Represents the state of the game -/
structure GameState :=
  (moves : List (Player × Position))
  (currentPlayer : Player)

/-- Checks if a given list of positions forms a winning line -/
def isWinningLine (line : List Position) : Bool :=
  sorry

/-- Checks if the current game state is a win for the given player -/
def isWin (state : GameState) (player : Player) : Bool :=
  sorry

/-- Represents a strategy for playing the game -/
def Strategy := GameState → Position

/-- The theorem stating that X has a winning strategy in at most 6 moves -/
theorem x_wins_in_six_moves :
  ∃ (strategy : Strategy),
    ∀ (opponent_strategy : Strategy),
      ∃ (final_state : GameState),
        (final_state.moves.length ≤ 6) ∧
        (isWin final_state Player.X) :=
  sorry

end NUMINAMATH_CALUDE_x_wins_in_six_moves_l296_29619


namespace NUMINAMATH_CALUDE_intersection_theorem_l296_29637

-- Define the four lines
def line1 (x y : ℚ) : Prop := 2 * y - 3 * x = 4
def line2 (x y : ℚ) : Prop := x + 3 * y = 3
def line3 (x y : ℚ) : Prop := 6 * x - 4 * y = 2
def line4 (x y : ℚ) : Prop := 5 * x - 15 * y = 15

-- Define the set of intersection points
def intersection_points : Set (ℚ × ℚ) :=
  {(18/11, 13/11), (21/11, 8/11)}

-- Define a function to check if a point lies on at least two lines
def on_at_least_two_lines (p : ℚ × ℚ) : Prop :=
  let (x, y) := p
  (line1 x y ∧ line2 x y) ∨ (line1 x y ∧ line3 x y) ∨ (line1 x y ∧ line4 x y) ∨
  (line2 x y ∧ line3 x y) ∨ (line2 x y ∧ line4 x y) ∨ (line3 x y ∧ line4 x y)

-- Theorem statement
theorem intersection_theorem :
  {p : ℚ × ℚ | on_at_least_two_lines p} = intersection_points := by sorry

end NUMINAMATH_CALUDE_intersection_theorem_l296_29637


namespace NUMINAMATH_CALUDE_license_plate_combinations_l296_29644

def num_consonants : ℕ := 21
def num_vowels : ℕ := 6
def num_digits : ℕ := 10
def num_special_chars : ℕ := 3

theorem license_plate_combinations : 
  num_consonants * num_vowels * num_consonants * num_digits * num_special_chars = 79380 :=
by sorry

end NUMINAMATH_CALUDE_license_plate_combinations_l296_29644


namespace NUMINAMATH_CALUDE_extreme_value_implies_b_l296_29668

/-- Given a function f(x) = x³ + ax² + bx + a² where a and b are real numbers,
    if f(x) has an extreme value of 10 at x = 1, then b = -11 -/
theorem extreme_value_implies_b (a b : ℝ) : 
  let f := fun (x : ℝ) => x^3 + a*x^2 + b*x + a^2
  (∃ (ε : ℝ), ∀ (x : ℝ), x ≠ 1 → |x - 1| < ε → f x ≤ f 1) ∧ 
  (f 1 = 10) →
  b = -11 := by sorry

end NUMINAMATH_CALUDE_extreme_value_implies_b_l296_29668


namespace NUMINAMATH_CALUDE_correct_operation_l296_29601

theorem correct_operation (a : ℝ) : 3 * a^3 - 2 * a^3 = a^3 := by
  sorry

end NUMINAMATH_CALUDE_correct_operation_l296_29601


namespace NUMINAMATH_CALUDE_female_students_count_l296_29623

/-- Given a class with n total students and m male students, 
    prove that the number of female students is n - m. -/
theorem female_students_count (n m : ℕ) : ℕ :=
  n - m

#check female_students_count

end NUMINAMATH_CALUDE_female_students_count_l296_29623


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l296_29661

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_property (a : ℕ → ℝ) :
  arithmetic_sequence a →
  a 4 + a 6 + a 8 + a 10 + a 12 = 120 →
  2 * a 10 - a 12 = 24 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l296_29661


namespace NUMINAMATH_CALUDE_positive_expression_l296_29685

theorem positive_expression (x y z : ℝ) 
  (hx : 0 < x ∧ x < 1) 
  (hy : -2 < y ∧ y < 0) 
  (hz : 0 < z ∧ z < 1) : 
  x + y^2 > 0 := by
  sorry

end NUMINAMATH_CALUDE_positive_expression_l296_29685


namespace NUMINAMATH_CALUDE_tomatoes_for_family_of_eight_l296_29677

/-- The number of tomatoes needed to feed a family for a single meal -/
def tomatoes_needed (slices_per_tomato : ℕ) (slices_per_person : ℕ) (family_size : ℕ) : ℕ :=
  (slices_per_person * family_size) / slices_per_tomato

theorem tomatoes_for_family_of_eight :
  tomatoes_needed 8 20 8 = 20 := by
  sorry

end NUMINAMATH_CALUDE_tomatoes_for_family_of_eight_l296_29677


namespace NUMINAMATH_CALUDE_f_zero_equals_one_l296_29634

theorem f_zero_equals_one (f : ℝ → ℝ) : 
  (∀ x, f x = Real.sin x + Real.exp x) → f 0 = 1 := by
  sorry

end NUMINAMATH_CALUDE_f_zero_equals_one_l296_29634


namespace NUMINAMATH_CALUDE_smallest_sum_of_identical_numbers_l296_29657

theorem smallest_sum_of_identical_numbers : ∃ (a b c : ℕ), 
  (6036 = 2010 * a) ∧ 
  (6036 = 2012 * b) ∧ 
  (6036 = 2013 * c) ∧ 
  (∀ (n : ℕ) (x y z : ℕ), 
    n > 0 ∧ n < 6036 → 
    ¬(n = 2010 * x ∧ n = 2012 * y ∧ n = 2013 * z)) :=
by sorry

end NUMINAMATH_CALUDE_smallest_sum_of_identical_numbers_l296_29657


namespace NUMINAMATH_CALUDE_remaining_payment_proof_l296_29639

/-- Given a deposit percentage and deposit amount, calculates the remaining amount to be paid -/
def remaining_payment (deposit_percentage : ℚ) (deposit_amount : ℚ) : ℚ :=
  (deposit_amount / deposit_percentage) - deposit_amount

/-- Proves that the remaining amount to be paid is 990, given a 10% deposit of 110 -/
theorem remaining_payment_proof : 
  remaining_payment (1/10) 110 = 990 := by
  sorry

end NUMINAMATH_CALUDE_remaining_payment_proof_l296_29639


namespace NUMINAMATH_CALUDE_antelopes_count_l296_29669

/-- Represents the count of animals on a safari --/
structure SafariCount where
  antelopes : ℕ
  rabbits : ℕ
  hyenas : ℕ
  wild_dogs : ℕ
  leopards : ℕ

/-- Conditions for the safari animal count --/
def safari_conditions (count : SafariCount) : Prop :=
  count.rabbits = count.antelopes + 34 ∧
  count.hyenas = count.antelopes + count.rabbits - 42 ∧
  count.wild_dogs = count.hyenas + 50 ∧
  count.leopards * 2 = count.rabbits ∧
  count.antelopes + count.rabbits + count.hyenas + count.wild_dogs + count.leopards = 605

/-- The theorem stating that the number of antelopes is 80 --/
theorem antelopes_count (count : SafariCount) :
  safari_conditions count → count.antelopes = 80 := by
  sorry

end NUMINAMATH_CALUDE_antelopes_count_l296_29669


namespace NUMINAMATH_CALUDE_max_inscribed_equilateral_triangle_area_l296_29659

/-- The maximum area of an equilateral triangle inscribed in a 12 by 15 rectangle -/
theorem max_inscribed_equilateral_triangle_area :
  ∃ (A : ℝ), A = 48 * Real.sqrt 3 ∧
  ∀ (s : ℝ), s > 0 →
    s * Real.sqrt 3 / 2 ≤ 12 →
    s ≤ 15 →
    s * s * Real.sqrt 3 / 4 ≤ A :=
by sorry

end NUMINAMATH_CALUDE_max_inscribed_equilateral_triangle_area_l296_29659


namespace NUMINAMATH_CALUDE_train_length_l296_29689

/-- Calculates the length of a train given its speed, the speed of a bus moving in the opposite direction, and the time it takes for the train to pass the bus. -/
theorem train_length (train_speed : ℝ) (bus_speed : ℝ) (passing_time : ℝ) :
  train_speed = 90 →
  bus_speed = 60 →
  passing_time = 5.279577633789296 →
  let relative_speed := (train_speed + bus_speed) * (5 / 18)
  let train_length := relative_speed * passing_time
  train_length = 41.663147 := by
  sorry

end NUMINAMATH_CALUDE_train_length_l296_29689


namespace NUMINAMATH_CALUDE_john_gave_one_third_l296_29646

/-- The fraction of burritos John gave to his friend -/
def fraction_given_away (boxes : ℕ) (burritos_per_box : ℕ) (days : ℕ) (burritos_per_day : ℕ) (burritos_left : ℕ) : ℚ :=
  let total_bought := boxes * burritos_per_box
  let total_eaten := days * burritos_per_day
  let total_before_eating := total_eaten + burritos_left
  let given_away := total_bought - total_before_eating
  given_away / total_bought

/-- Theorem stating that John gave away 1/3 of the burritos -/
theorem john_gave_one_third :
  fraction_given_away 3 20 10 3 10 = 1 / 3 := by
  sorry


end NUMINAMATH_CALUDE_john_gave_one_third_l296_29646


namespace NUMINAMATH_CALUDE_exam_score_calculation_l296_29627

theorem exam_score_calculation (total_questions : ℕ) (correct_score : ℕ) (total_score : ℕ) (correct_answers : ℕ) :
  total_questions = 75 →
  correct_score = 4 →
  total_score = 125 →
  correct_answers = 40 →
  (total_questions - correct_answers) * (correct_score - (correct_score * correct_answers - total_score) / (total_questions - correct_answers)) = total_score :=
by sorry

end NUMINAMATH_CALUDE_exam_score_calculation_l296_29627


namespace NUMINAMATH_CALUDE_proposition_b_is_true_l296_29638

theorem proposition_b_is_true : 3 > 4 ∨ 3 < 4 := by
  sorry

end NUMINAMATH_CALUDE_proposition_b_is_true_l296_29638
