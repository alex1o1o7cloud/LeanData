import Mathlib

namespace NUMINAMATH_CALUDE_intersection_P_Q_l2727_272754

def P : Set ℝ := {0, 1, 2, 3}
def Q : Set ℝ := {x : ℝ | |x| < 2}

theorem intersection_P_Q : P ∩ Q = {0, 1} := by sorry

end NUMINAMATH_CALUDE_intersection_P_Q_l2727_272754


namespace NUMINAMATH_CALUDE_arithmetic_calculations_l2727_272786

theorem arithmetic_calculations : 
  (12 - (-18) + (-7) - 15 = 8) ∧ 
  (-1^4 + (-2)^3 * (-1/2) - |(-1-5)| = -3) := by sorry

end NUMINAMATH_CALUDE_arithmetic_calculations_l2727_272786


namespace NUMINAMATH_CALUDE_circle_tangent_properties_l2727_272714

-- Define the circle C
def circle_C (a r : ℝ) := {(x, y) : ℝ × ℝ | (x - 2)^2 + (y - a)^2 = r^2}

-- Define the tangent line
def tangent_line := {(x, y) : ℝ × ℝ | x + 2*y - 7 = 0}

-- Define the condition that the line is tangent to the circle at (3, 2)
def is_tangent (a r : ℝ) : Prop :=
  (3, 2) ∈ circle_C a r ∧ (3, 2) ∈ tangent_line ∧
  ∀ (x y : ℝ), (x, y) ∈ circle_C a r ∩ tangent_line → (x, y) = (3, 2)

-- Theorem statement
theorem circle_tangent_properties (a r : ℝ) (h : is_tangent a r) :
  a = 0 ∧ (-1, -1) ∉ circle_C a r :=
sorry

end NUMINAMATH_CALUDE_circle_tangent_properties_l2727_272714


namespace NUMINAMATH_CALUDE_real_condition_pure_imaginary_condition_fourth_quadrant_condition_l2727_272708

-- Define the complex number z as a function of a
def z (a : ℝ) : ℂ := Complex.mk (a^2 - 2*a - 3) (a^2 + a - 12)

-- (I) z is a real number iff a = -4 or a = 3
theorem real_condition (a : ℝ) : z a = Complex.mk (z a).re 0 ↔ a = -4 ∨ a = 3 := by sorry

-- (II) z is a pure imaginary number iff a = -1
theorem pure_imaginary_condition (a : ℝ) : z a = Complex.mk 0 (z a).im ∧ (z a).im ≠ 0 ↔ a = -1 := by sorry

-- (III) z is in the fourth quadrant iff -4 < a < -1
theorem fourth_quadrant_condition (a : ℝ) : (z a).re > 0 ∧ (z a).im < 0 ↔ -4 < a ∧ a < -1 := by sorry

end NUMINAMATH_CALUDE_real_condition_pure_imaginary_condition_fourth_quadrant_condition_l2727_272708


namespace NUMINAMATH_CALUDE_ages_sum_l2727_272787

theorem ages_sum (a b c : ℕ+) : 
  a = b ∧ a > c ∧ a * b * c = 162 → a + b + c = 20 := by
sorry

end NUMINAMATH_CALUDE_ages_sum_l2727_272787


namespace NUMINAMATH_CALUDE_tangent_circles_radii_l2727_272747

/-- Given a sequence of six circles tangent to each other and two parallel lines,
    where the radii form a geometric sequence, prove that if the smallest radius
    is 5 and the largest is 20, then the radius of the third circle is 5 * 2^(2/5). -/
theorem tangent_circles_radii (r : Fin 6 → ℝ) : 
  (∀ i : Fin 5, r i > 0) →  -- All radii are positive
  (∀ i : Fin 5, r i < r i.succ) →  -- Radii are in increasing order
  (∀ i j : Fin 5, i < j → r j / r i = r (j+1) / r (j : Fin 6)) →  -- Geometric sequence
  r 0 = 5 →  -- Smallest radius
  r 5 = 20 →  -- Largest radius
  r 2 = 5 * 2^(2/5) := by
sorry

end NUMINAMATH_CALUDE_tangent_circles_radii_l2727_272747


namespace NUMINAMATH_CALUDE_constant_term_product_l2727_272765

theorem constant_term_product (x : ℝ) : 
  (x^4 + x^2 + 7) * (2*x^5 + 3*x^3 + 10) = 70 + x * (2*x^8 + 3*x^6 + 20*x^4 + 3*x^7 + 10*x^5 + 7*2*x^5 + 7*3*x^3) :=
by sorry

end NUMINAMATH_CALUDE_constant_term_product_l2727_272765


namespace NUMINAMATH_CALUDE_binary_multiplication_addition_l2727_272748

/-- Converts a binary number represented as a list of bits to its decimal equivalent -/
def binaryToDecimal (bits : List Bool) : Nat :=
  bits.foldl (fun acc b => 2 * acc + if b then 1 else 0) 0

/-- Represents a binary number as a list of bits (least significant bit first) -/
def Binary := List Bool

def binary_11011 : Binary := [true, true, false, true, true]
def binary_111 : Binary := [true, true, true]
def binary_1010 : Binary := [false, true, false, true]
def binary_11000111 : Binary := [true, true, true, false, false, false, true, true]

theorem binary_multiplication_addition :
  (binaryToDecimal binary_11011 * binaryToDecimal binary_111 + binaryToDecimal binary_1010) =
  binaryToDecimal binary_11000111 := by
  sorry

end NUMINAMATH_CALUDE_binary_multiplication_addition_l2727_272748


namespace NUMINAMATH_CALUDE_min_value_of_exponential_sum_l2727_272769

theorem min_value_of_exponential_sum (a b : ℝ) (h : a + b = 2) :
  ∃ m : ℝ, m = 6 ∧ ∀ x y : ℝ, x + y = 2 → 3^x + 3^y ≥ m :=
sorry

end NUMINAMATH_CALUDE_min_value_of_exponential_sum_l2727_272769


namespace NUMINAMATH_CALUDE_third_concert_highest_attendance_l2727_272706

/-- Represents a concert with its attendance and early departure numbers -/
structure Concert where
  attendance : ℕ
  early_departure : ℕ

/-- Calculates the number of people who remained until the end of the concert -/
def remaining_attendance (c : Concert) : ℕ :=
  c.attendance - c.early_departure

/-- The three concerts attended -/
def concert1 : Concert := { attendance := 65899, early_departure := 375 }
def concert2 : Concert := { attendance := 65899 + 119, early_departure := 498 }
def concert3 : Concert := { attendance := 80453, early_departure := 612 }

theorem third_concert_highest_attendance :
  remaining_attendance concert3 > remaining_attendance concert1 ∧
  remaining_attendance concert3 > remaining_attendance concert2 :=
by sorry

end NUMINAMATH_CALUDE_third_concert_highest_attendance_l2727_272706


namespace NUMINAMATH_CALUDE_not_balanced_numbers_l2727_272783

/-- Definition of balanced numbers with respect to l -/
def balanced (a b : ℝ) : Prop := a + b = 2

/-- Given equation -/
axiom equation : ∃ m : ℝ, (Real.sqrt 3 + m) * (Real.sqrt 3 - 1) = 2

/-- Theorem to prove -/
theorem not_balanced_numbers : ¬∃ m : ℝ, 
  (Real.sqrt 3 + m) * (Real.sqrt 3 - 1) = 2 ∧ 
  balanced (m + Real.sqrt 3) (2 - Real.sqrt 3) := by
  sorry

end NUMINAMATH_CALUDE_not_balanced_numbers_l2727_272783


namespace NUMINAMATH_CALUDE_largest_angle_measure_l2727_272790

def ConvexPentagon (a b c d e : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ e > 0 ∧
  b = a + 1 ∧ c = b + 1 ∧ d = c + 1 ∧ e = d + 1 ∧
  a + b + c + d + e = 540

theorem largest_angle_measure (a b c d e : ℝ) :
  ConvexPentagon a b c d e →
  c - 3 = a →
  e = 110 := by
  sorry

end NUMINAMATH_CALUDE_largest_angle_measure_l2727_272790


namespace NUMINAMATH_CALUDE_inverse_proportion_ordering_l2727_272793

theorem inverse_proportion_ordering (y₁ y₂ y₃ : ℝ) :
  y₁ = 7 / (-3) ∧ y₂ = 7 / (-1) ∧ y₃ = 7 / 2 →
  y₂ < y₁ ∧ y₁ < y₃ := by
  sorry

end NUMINAMATH_CALUDE_inverse_proportion_ordering_l2727_272793


namespace NUMINAMATH_CALUDE_square_sum_geq_linear_l2727_272738

theorem square_sum_geq_linear (a b : ℝ) : a^2 + b^2 ≥ 2*a - 2*b - 2 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_geq_linear_l2727_272738


namespace NUMINAMATH_CALUDE_initial_shells_amount_l2727_272742

/-- The amount of shells initially in Jovana's bucket -/
def initial_shells : ℕ := sorry

/-- The amount of shells added to fill the bucket -/
def added_shells : ℕ := 12

/-- The total amount of shells after filling the bucket -/
def total_shells : ℕ := 17

/-- Theorem stating that the initial amount of shells is 5 pounds -/
theorem initial_shells_amount : initial_shells = 5 := by
  sorry

end NUMINAMATH_CALUDE_initial_shells_amount_l2727_272742


namespace NUMINAMATH_CALUDE_walking_problem_l2727_272736

theorem walking_problem (distance : ℝ) (initial_meeting_time : ℝ) 
  (speed_ratio : ℝ) (h1 : distance = 100) (h2 : initial_meeting_time = 3) 
  (h3 : speed_ratio = 4) : 
  ∃ (speed_A speed_B : ℝ) (meeting_times : List ℝ),
    speed_A = 80 / 3 ∧ 
    speed_B = 20 / 3 ∧
    speed_A = speed_ratio * speed_B ∧
    initial_meeting_time * (speed_A + speed_B) = distance ∧
    meeting_times = [3, 5, 9, 15] ∧
    (∀ t ∈ meeting_times, 
      (t ≤ distance / speed_B) ∧ 
      (∃ n : ℕ, t * speed_B = 2 * n * distance - t * speed_A ∨ 
               t * speed_B = (2 * n + 1) * distance - (distance - t * speed_A))) :=
by sorry

end NUMINAMATH_CALUDE_walking_problem_l2727_272736


namespace NUMINAMATH_CALUDE_sams_sandwich_count_l2727_272732

/-- Represents the number of different types for each sandwich component -/
structure SandwichOptions where
  bread : Nat
  meat : Nat
  cheese : Nat

/-- Calculates the number of sandwiches Sam can order given the options and restrictions -/
def samsSandwichOptions (options : SandwichOptions) : Nat :=
  options.bread * options.meat * options.cheese - 
  options.bread - 
  options.cheese - 
  options.bread

/-- The theorem stating the number of sandwich options for Sam -/
theorem sams_sandwich_count :
  samsSandwichOptions ⟨5, 7, 6⟩ = 194 := by
  sorry

#eval samsSandwichOptions ⟨5, 7, 6⟩

end NUMINAMATH_CALUDE_sams_sandwich_count_l2727_272732


namespace NUMINAMATH_CALUDE_cost_of_flour_for_cakes_claire_cake_flour_cost_l2727_272718

/-- The cost of flour for making cakes -/
theorem cost_of_flour_for_cakes (num_cakes : ℕ) (packages_per_cake : ℕ) (cost_per_package : ℕ) : 
  num_cakes * packages_per_cake * cost_per_package = num_cakes * (packages_per_cake * cost_per_package) :=
by sorry

/-- Claire's cake flour cost calculation -/
theorem claire_cake_flour_cost : 2 * (2 * 3) = 12 :=
by sorry

end NUMINAMATH_CALUDE_cost_of_flour_for_cakes_claire_cake_flour_cost_l2727_272718


namespace NUMINAMATH_CALUDE_fd_length_l2727_272766

-- Define the triangle and arc
def Triangle (A B C : ℝ × ℝ) : Prop :=
  ∃ (r : ℝ), r = 20 ∧ 
  (A.1 - B.1)^2 + (A.2 - B.2)^2 = r^2 ∧
  (C.1 - B.1)^2 + (C.2 - B.2)^2 = r^2 ∧
  (A.1 - C.1)^2 + (A.2 - C.2)^2 = r^2

-- Define the semicircle
def Semicircle (A B D : ℝ × ℝ) : Prop :=
  ∃ (O : ℝ × ℝ), O = ((A.1 + B.1)/2, (A.2 + B.2)/2) ∧
  (D.1 - O.1)^2 + (D.2 - O.2)^2 = ((A.1 - B.1)^2 + (A.2 - B.2)^2) / 4

-- Define the tangent point
def Tangent (C D O : ℝ × ℝ) : Prop :=
  (C.1 - D.1) * (D.1 - O.1) + (C.2 - D.2) * (D.2 - O.2) = 0

-- Define the intersection point
def Intersect (C D F B : ℝ × ℝ) : Prop :=
  ∃ (t : ℝ), 0 < t ∧ t < 1 ∧
  F = (C.1 + t*(D.1 - C.1), C.2 + t*(D.2 - C.2)) ∧
  (F.1 - B.1)^2 + (F.2 - B.2)^2 = 20^2

-- Main theorem
theorem fd_length (A B C D F : ℝ × ℝ) :
  Triangle A B C →
  Semicircle A B D →
  Tangent C D ((A.1 + B.1)/2, (A.2 + B.2)/2) →
  Intersect C D F B →
  (F.1 - D.1)^2 + (F.2 - D.2)^2 = 16 := by
  sorry

end NUMINAMATH_CALUDE_fd_length_l2727_272766


namespace NUMINAMATH_CALUDE_min_distance_squared_l2727_272760

/-- The minimum squared distance from a point M(x,y,z) to N(1,1,1), 
    given specific conditions on x, y, and z -/
theorem min_distance_squared (x y z : ℝ) : 
  (∃ r : ℝ, y = x * r ∧ z = y * r) →  -- geometric progression condition
  (y * z = (x * y + x * z) / 2) →    -- arithmetic progression condition
  (z ≥ 1) →                          -- z ≥ 1 condition
  (x ≠ y ∧ y ≠ z ∧ x ≠ z) →          -- distinctness condition
  18 ≤ (x - 1)^2 + (y - 1)^2 + (z - 1)^2 :=
by sorry

end NUMINAMATH_CALUDE_min_distance_squared_l2727_272760


namespace NUMINAMATH_CALUDE_spaceship_total_distance_l2727_272784

/-- The total distance traveled by a spaceship between Earth and various planets -/
theorem spaceship_total_distance (d_earth_x d_x_y d_y_z d_z_w d_w_earth : ℝ) 
  (h1 : d_earth_x = 3.37)
  (h2 : d_x_y = 1.57)
  (h3 : d_y_z = 2.19)
  (h4 : d_z_w = 4.27)
  (h5 : d_w_earth = 1.89) :
  d_earth_x + d_x_y + d_y_z + d_z_w + d_w_earth = 13.29 := by
  sorry

end NUMINAMATH_CALUDE_spaceship_total_distance_l2727_272784


namespace NUMINAMATH_CALUDE_system_solution_l2727_272795

def solution_set : Set (ℝ × ℝ) :=
  {(-1/Real.sqrt 10, 3/Real.sqrt 10), (-1/Real.sqrt 10, -3/Real.sqrt 10),
   (1/Real.sqrt 10, 3/Real.sqrt 10), (1/Real.sqrt 10, -3/Real.sqrt 10)}

def satisfies_system (p : ℝ × ℝ) : Prop :=
  let (x, y) := p
  x^2 + y^2 ≤ 1 ∧
  x^4 - 18*x^2*y^2 + 81*y^4 - 20*x^2 - 180*y^2 + 100 = 0

theorem system_solution :
  {p : ℝ × ℝ | satisfies_system p} = solution_set :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l2727_272795


namespace NUMINAMATH_CALUDE_xiaodong_election_l2727_272741

theorem xiaodong_election (V : ℝ) (h : V > 0) : 
  let votes_needed := (3/4 : ℝ) * V
  let votes_calculated := (2/3 : ℝ) * V
  let votes_obtained := (5/6 : ℝ) * votes_calculated
  let votes_remaining := V - votes_calculated
  let additional_votes_needed := votes_needed - votes_obtained
  (additional_votes_needed / votes_remaining) = (7/12 : ℝ) := by
sorry

end NUMINAMATH_CALUDE_xiaodong_election_l2727_272741


namespace NUMINAMATH_CALUDE_seven_non_similar_triangles_l2727_272792

/-- Represents a point in a 2D plane -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- Represents a triangle -/
structure Triangle :=
  (A : Point) (B : Point) (C : Point)

/-- Represents an altitude of a triangle -/
structure Altitude :=
  (base : Point) (top : Point)

/-- Checks if a triangle is acute-angled -/
def isAcuteAngled (t : Triangle) : Prop :=
  sorry

/-- Checks if all sides of a triangle are unequal -/
def hasUnequalSides (t : Triangle) : Prop :=
  sorry

/-- Checks if three lines intersect at a single point -/
def intersectAtPoint (a b c : Altitude) (H : Point) : Prop :=
  sorry

/-- Counts the number of non-similar triangle types in the figure -/
def countNonSimilarTriangles (t : Triangle) (AD BE CF : Altitude) (H : Point) : ℕ :=
  sorry

/-- The main theorem -/
theorem seven_non_similar_triangles 
  (ABC : Triangle) 
  (AD BE CF : Altitude) 
  (H : Point) 
  (h1 : isAcuteAngled ABC) 
  (h2 : hasUnequalSides ABC)
  (h3 : intersectAtPoint AD BE CF H) :
  countNonSimilarTriangles ABC AD BE CF H = 7 :=
sorry

end NUMINAMATH_CALUDE_seven_non_similar_triangles_l2727_272792


namespace NUMINAMATH_CALUDE_correct_average_weight_l2727_272731

theorem correct_average_weight (n : ℕ) (initial_avg : ℚ) (misread_weight : ℚ) (correct_weight : ℚ) :
  n = 20 →
  initial_avg = 58.4 →
  misread_weight = 56 →
  correct_weight = 66 →
  (n : ℚ) * initial_avg + (correct_weight - misread_weight) = n * 58.9 := by
  sorry

end NUMINAMATH_CALUDE_correct_average_weight_l2727_272731


namespace NUMINAMATH_CALUDE_cone_height_from_circular_sector_l2727_272735

/-- The height of a cone formed from a sector of a circular sheet -/
theorem cone_height_from_circular_sector (r : ℝ) (n : ℕ) (h : n > 0) : 
  let base_radius := r * Real.pi / (2 * n)
  let slant_height := r
  let height := Real.sqrt (slant_height^2 - base_radius^2)
  (r = 10 ∧ n = 4) → height = Real.sqrt 93.75 := by
  sorry

end NUMINAMATH_CALUDE_cone_height_from_circular_sector_l2727_272735


namespace NUMINAMATH_CALUDE_same_color_probability_is_59_225_l2727_272700

/-- Represents a 30-sided die with colored sides -/
structure ColoredDie :=
  (blue : Nat)
  (yellow : Nat)
  (green : Nat)
  (purple : Nat)
  (total : Nat)
  (side_sum : blue + yellow + green + purple = total)

/-- The probability of two dice showing the same color -/
def same_color_probability (d : ColoredDie) : Rat :=
  let blue_prob := (d.blue * d.blue : Rat) / (d.total * d.total)
  let yellow_prob := (d.yellow * d.yellow : Rat) / (d.total * d.total)
  let green_prob := (d.green * d.green : Rat) / (d.total * d.total)
  let purple_prob := (d.purple * d.purple : Rat) / (d.total * d.total)
  blue_prob + yellow_prob + green_prob + purple_prob

/-- The specific 30-sided die described in the problem -/
def problem_die : ColoredDie :=
  { blue := 6
    yellow := 8
    green := 10
    purple := 6
    total := 30
    side_sum := by norm_num }

/-- Theorem stating the probability of two problem dice showing the same color -/
theorem same_color_probability_is_59_225 :
  same_color_probability problem_die = 59 / 225 := by
  sorry


end NUMINAMATH_CALUDE_same_color_probability_is_59_225_l2727_272700


namespace NUMINAMATH_CALUDE_college_students_count_l2727_272794

theorem college_students_count (boys girls : ℕ) (h1 : boys * 5 = girls * 8) (h2 : girls = 175) :
  boys + girls = 455 := by
  sorry

end NUMINAMATH_CALUDE_college_students_count_l2727_272794


namespace NUMINAMATH_CALUDE_mark_sugar_intake_excess_l2727_272723

/-- Represents the calorie content and sugar information for a soft drink -/
structure SoftDrink where
  totalCalories : ℕ
  sugarPercentage : ℚ

/-- Represents the sugar content of a candy bar -/
structure CandyBar where
  sugarCalories : ℕ

theorem mark_sugar_intake_excess (drink : SoftDrink) (bar : CandyBar) 
    (h1 : drink.totalCalories = 2500)
    (h2 : drink.sugarPercentage = 5 / 100)
    (h3 : bar.sugarCalories = 25)
    (h4 : (drink.totalCalories : ℚ) * drink.sugarPercentage + 7 * bar.sugarCalories = 300)
    (h5 : (300 : ℚ) / 150 - 1 = 1) : 
    (300 : ℚ) / 150 - 1 = 1 := by sorry

end NUMINAMATH_CALUDE_mark_sugar_intake_excess_l2727_272723


namespace NUMINAMATH_CALUDE_unique_satisfying_function_l2727_272759

/-- A function f : [1, +∞) → [1, +∞) satisfying the given conditions -/
def SatisfyingFunction (f : ℝ → ℝ) : Prop :=
  (∀ x ≥ 1, f x ≥ 1) ∧
  (∀ x ≥ 1, f x ≤ 2 * (x + 1)) ∧
  (∀ x ≥ 1, f (x + 1) = (1 / x) * ((f x)^2 - 1))

/-- The theorem stating that x + 1 is the unique function satisfying the conditions -/
theorem unique_satisfying_function :
  ∃! f : ℝ → ℝ, SatisfyingFunction f ∧ ∀ x ≥ 1, f x = x + 1 :=
sorry

end NUMINAMATH_CALUDE_unique_satisfying_function_l2727_272759


namespace NUMINAMATH_CALUDE_people_in_line_l2727_272709

theorem people_in_line (initial_people : ℕ) (additional_people : ℕ) : 
  initial_people = 61 → additional_people = 22 → initial_people + additional_people = 83 := by
  sorry

end NUMINAMATH_CALUDE_people_in_line_l2727_272709


namespace NUMINAMATH_CALUDE_smallest_positive_z_l2727_272746

open Real

theorem smallest_positive_z (x z : ℝ) (h1 : cos x = 0) (h2 : cos (x + z) = 1/2) :
  ∃ (z_min : ℝ), z_min = π/6 ∧ z_min > 0 ∧ ∀ (z' : ℝ), z' > 0 → cos x = 0 → cos (x + z') = 1/2 → z' ≥ z_min :=
sorry

end NUMINAMATH_CALUDE_smallest_positive_z_l2727_272746


namespace NUMINAMATH_CALUDE_unique_four_digit_reverse_l2727_272770

/-- Reverses the digits of a four-digit number -/
def reverse_digits (n : ℕ) : ℕ :=
  let a := n / 1000
  let b := (n / 100) % 10
  let c := (n / 10) % 10
  let d := n % 10
  d * 1000 + c * 100 + b * 10 + a

/-- Checks if a number is a four-digit number -/
def is_four_digit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n < 10000

theorem unique_four_digit_reverse : ∃! n : ℕ, is_four_digit n ∧ 4 * n = reverse_digits n :=
  sorry

end NUMINAMATH_CALUDE_unique_four_digit_reverse_l2727_272770


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l2727_272755

theorem imaginary_part_of_complex_fraction :
  let i : ℂ := Complex.I
  let z : ℂ := (1 - i) / (1 + i)
  Complex.im z = -1 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l2727_272755


namespace NUMINAMATH_CALUDE_axis_of_symmetry_parabola_l2727_272749

/-- The axis of symmetry for the parabola y² = -8x is the line x = 2 -/
theorem axis_of_symmetry_parabola (x y : ℝ) : 
  y^2 = -8*x → (x = 2 ↔ ∀ y', y'^2 = -8*x → y'^2 = y^2) :=
by sorry

end NUMINAMATH_CALUDE_axis_of_symmetry_parabola_l2727_272749


namespace NUMINAMATH_CALUDE_vector_operation_result_l2727_272774

def v1 : Fin 3 → ℝ := ![-3, 2, -1]
def v2 : Fin 3 → ℝ := ![1, 10, -2]
def scalar : ℝ := 2

theorem vector_operation_result :
  scalar • v1 + v2 = ![(-5 : ℝ), 14, -4] := by sorry

end NUMINAMATH_CALUDE_vector_operation_result_l2727_272774


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l2727_272739

theorem quadratic_inequality_solution_set :
  {x : ℝ | -x^2 + 5*x + 6 > 0} = {x : ℝ | -1 < x ∧ x < 6} := by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l2727_272739


namespace NUMINAMATH_CALUDE_windows_preference_l2727_272763

theorem windows_preference (total : ℕ) (mac_pref : ℕ) (no_pref : ℕ) : 
  total = 210 →
  mac_pref = 60 →
  no_pref = 90 →
  ∃ (windows_pref : ℕ),
    windows_pref = total - mac_pref - (mac_pref / 3) - no_pref ∧
    windows_pref = 40 := by
  sorry

end NUMINAMATH_CALUDE_windows_preference_l2727_272763


namespace NUMINAMATH_CALUDE_squirrel_mushroom_theorem_l2727_272719

theorem squirrel_mushroom_theorem (N : ℝ) (h : N > 0) :
  let initial_porcini := 0.85 * N
  let initial_saffron := 0.15 * N
  let eaten (x : ℝ) := x
  let remaining_porcini (x : ℝ) := initial_porcini - eaten x
  let remaining_total (x : ℝ) := N - eaten x
  let final_saffron_ratio (x : ℝ) := initial_saffron / remaining_total x
  ∃ x : ℝ, x ≥ 0 ∧ x ≤ initial_porcini ∧ final_saffron_ratio x = 0.3 ∧ eaten x / N = 1/2 :=
by
  sorry

end NUMINAMATH_CALUDE_squirrel_mushroom_theorem_l2727_272719


namespace NUMINAMATH_CALUDE_alternating_sequence_sum_l2727_272757

def arithmetic_sequence_sum (a₁ : ℤ) (d : ℤ) (n : ℕ) : ℤ :=
  n * (2 * a₁ + (n - 1) * d) / 2

def alternating_sum (a₁ : ℤ) (d : ℤ) (n : ℕ) : ℤ :=
  arithmetic_sequence_sum a₁ (2 * d) ((n + 1) / 2) -
  arithmetic_sequence_sum (a₁ + d) (2 * d) (n / 2)

theorem alternating_sequence_sum :
  alternating_sum 2 3 19 = 29 := by
  sorry

end NUMINAMATH_CALUDE_alternating_sequence_sum_l2727_272757


namespace NUMINAMATH_CALUDE_average_after_addition_l2727_272775

theorem average_after_addition (numbers : List ℝ) (target_avg : ℝ) : 
  numbers = [6, 16, 8, 12, 21] → target_avg = 17 →
  ∃ x : ℝ, (numbers.sum + x) / (numbers.length + 1 : ℝ) = target_avg ∧ x = 39 := by
sorry

end NUMINAMATH_CALUDE_average_after_addition_l2727_272775


namespace NUMINAMATH_CALUDE_plywood_area_conservation_l2727_272778

theorem plywood_area_conservation (A W : ℝ) (h : A > 0 ∧ W > 0) :
  let L : ℝ := A / W
  let L' : ℝ := A / (2 * W)
  A = W * L ∧ A = (2 * W) * L' := by sorry

end NUMINAMATH_CALUDE_plywood_area_conservation_l2727_272778


namespace NUMINAMATH_CALUDE_cylinder_radius_proof_l2727_272768

/-- The radius of a cylinder with specific properties -/
def cylinder_radius : ℝ := 12

/-- The original height of the cylinder -/
def original_height : ℝ := 4

/-- The increase in radius or height -/
def increase : ℝ := 8

theorem cylinder_radius_proof :
  (cylinder_radius + increase)^2 * original_height = 
  cylinder_radius^2 * (original_height + increase) :=
by sorry

end NUMINAMATH_CALUDE_cylinder_radius_proof_l2727_272768


namespace NUMINAMATH_CALUDE_seventh_grade_class_size_l2727_272743

theorem seventh_grade_class_size (girls boys : ℕ) : 
  girls * 3 + boys = 24 → 
  boys / 3 = 6 → 
  girls + boys = 24 :=
by
  sorry

end NUMINAMATH_CALUDE_seventh_grade_class_size_l2727_272743


namespace NUMINAMATH_CALUDE_counterexample_exists_l2727_272779

-- Define the types for points, lines, and planes
variable (Point Line Plane : Type)

-- Define the relation for a point being on a line or in a plane
variable (on_line : Point → Line → Prop)
variable (in_plane : Point → Plane → Prop)

-- Define the relation for a line being a subset of a plane
variable (line_subset_plane : Line → Plane → Prop)

-- Theorem statement
theorem counterexample_exists (l : Line) (α : Plane) (A : Point) 
  (h1 : ¬ line_subset_plane l α) 
  (h2 : on_line A l) :
  ¬ (∀ A, on_line A l → ¬ in_plane A α) :=
by sorry

end NUMINAMATH_CALUDE_counterexample_exists_l2727_272779


namespace NUMINAMATH_CALUDE_inequality_proof_l2727_272725

theorem inequality_proof (x y z : ℝ) (h1 : x ≥ y) (h2 : y ≥ z) (h3 : z > 0) :
  (x^2 * y) / (y + z) + (y^2 * z) / (z + x) + (z^2 * x) / (x + y) ≥ (1/2) * (x^2 + y^2 + z^2) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2727_272725


namespace NUMINAMATH_CALUDE_cookies_in_fridge_l2727_272712

theorem cookies_in_fridge (total cookies_to_tim cookies_to_mike : ℕ) 
  (h1 : total = 512)
  (h2 : cookies_to_tim = 30)
  (h3 : cookies_to_mike = 45)
  (h4 : cookies_to_anna = 3 * cookies_to_tim) :
  total - (cookies_to_tim + cookies_to_mike + cookies_to_anna) = 347 :=
by sorry

end NUMINAMATH_CALUDE_cookies_in_fridge_l2727_272712


namespace NUMINAMATH_CALUDE_complex_fraction_simplification_l2727_272773

theorem complex_fraction_simplification :
  let a := 6 + 7 / 2015
  let b := 4 + 5 / 2016
  let c := 7 + 2008 / 2015
  let d := 2 + 2011 / 2016
  let expression := a * b - c * d - 7 * (7 / 2015)
  expression = 5 / 144 := by sorry

end NUMINAMATH_CALUDE_complex_fraction_simplification_l2727_272773


namespace NUMINAMATH_CALUDE_purely_imaginary_complex_number_l2727_272796

theorem purely_imaginary_complex_number (a : ℝ) : 
  (Complex.mk (a^2 - 2*a - 3) (a + 1)).im ≠ 0 ∧ 
  (Complex.mk (a^2 - 2*a - 3) (a + 1)).re = 0 → 
  a = 3 := by
  sorry

end NUMINAMATH_CALUDE_purely_imaginary_complex_number_l2727_272796


namespace NUMINAMATH_CALUDE_two_digit_sum_ten_l2727_272733

/-- A two-digit number is a natural number between 10 and 99, inclusive. -/
def TwoDigitNumber (n : ℕ) : Prop :=
  10 ≤ n ∧ n ≤ 99

/-- The digit sum of a natural number is the sum of its digits. -/
def DigitSum (n : ℕ) : ℕ :=
  (n / 10) + (n % 10)

/-- There are exactly 9 two-digit numbers whose digits sum to 10. -/
theorem two_digit_sum_ten :
  ∃! (s : Finset ℕ), (∀ n ∈ s, TwoDigitNumber n ∧ DigitSum n = 10) ∧ s.card = 9 := by
sorry

end NUMINAMATH_CALUDE_two_digit_sum_ten_l2727_272733


namespace NUMINAMATH_CALUDE_quadratic_two_distinct_roots_l2727_272726

theorem quadratic_two_distinct_roots (m : ℝ) : 
  ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁^2 + m*x₁ - 8 = 0 ∧ x₂^2 + m*x₂ - 8 = 0 :=
by
  sorry


end NUMINAMATH_CALUDE_quadratic_two_distinct_roots_l2727_272726


namespace NUMINAMATH_CALUDE_fractional_sum_zero_l2727_272781

theorem fractional_sum_zero (a b c k : ℝ) (h1 : a ≠ b) (h2 : b ≠ c) (h3 : c ≠ a) (h4 : k ≠ 0) 
  (h5 : a / (b - c) + b / (c - a) + c / (a - b) = 0) :
  a / (k * (b - c)^2) + b / (k * (c - a)^2) + c / (k * (a - b)^2) = 0 := by
  sorry

end NUMINAMATH_CALUDE_fractional_sum_zero_l2727_272781


namespace NUMINAMATH_CALUDE_alices_lawn_area_l2727_272711

/-- Represents a rectangular lawn with fence posts -/
structure Lawn :=
  (total_posts : ℕ)
  (post_spacing : ℕ)
  (long_side_posts : ℕ)
  (short_side_posts : ℕ)

/-- Calculates the area of the lawn given its specifications -/
def lawn_area (l : Lawn) : ℕ :=
  (l.post_spacing * (l.short_side_posts - 1)) * (l.post_spacing * (l.long_side_posts - 1))

/-- Theorem stating the area of Alice's lawn -/
theorem alices_lawn_area :
  ∀ (l : Lawn),
  l.total_posts = 24 →
  l.post_spacing = 5 →
  l.long_side_posts = 3 * l.short_side_posts →
  2 * (l.long_side_posts + l.short_side_posts - 2) = l.total_posts →
  lawn_area l = 825 := by
  sorry

#check alices_lawn_area

end NUMINAMATH_CALUDE_alices_lawn_area_l2727_272711


namespace NUMINAMATH_CALUDE_michael_watermelon_weight_l2727_272751

/-- The weight of Michael's watermelon in pounds -/
def michael_watermelon : ℝ := 8

/-- The weight of Clay's watermelon in pounds -/
def clay_watermelon : ℝ := 3 * michael_watermelon

/-- The weight of John's watermelon in pounds -/
def john_watermelon : ℝ := 12

theorem michael_watermelon_weight :
  michael_watermelon = 8 ∧
  clay_watermelon = 3 * michael_watermelon ∧
  john_watermelon = clay_watermelon / 2 ∧
  john_watermelon = 12 := by
  sorry

end NUMINAMATH_CALUDE_michael_watermelon_weight_l2727_272751


namespace NUMINAMATH_CALUDE_parabola_focal_line_properties_l2727_272756

/-- Parabola structure -/
structure Parabola where
  p : ℝ
  pos_p : p > 0

/-- Point on a parabola -/
structure ParabolaPoint (para : Parabola) where
  x : ℝ
  y : ℝ
  on_parabola : y^2 = 2 * para.p * x

/-- Line through focus intersecting parabola -/
structure FocalLine (para : Parabola) where
  A : ParabolaPoint para
  B : ParabolaPoint para

/-- Theorem statement -/
theorem parabola_focal_line_properties (para : Parabola) (l : FocalLine para) :
  ∃ (N : ℝ × ℝ) (P : ℝ × ℝ),
    -- 1. FN = 1/2 * AB
    (N.1 - para.p/2)^2 + N.2^2 = (1/2)^2 * ((l.A.x - l.B.x)^2 + (l.A.y - l.B.y)^2) ∧
    -- 2. Trajectory of P
    P.1 + para.p/2 = 0 := by
  sorry

end NUMINAMATH_CALUDE_parabola_focal_line_properties_l2727_272756


namespace NUMINAMATH_CALUDE_complex_expression_equality_l2727_272753

theorem complex_expression_equality : 
  Real.sqrt (4/9) - Real.sqrt ((-2)^4) + (19/27 - 1)^(1/3) - (-1)^2017 = -5/3 := by
  sorry

end NUMINAMATH_CALUDE_complex_expression_equality_l2727_272753


namespace NUMINAMATH_CALUDE_cos_210_degrees_l2727_272737

theorem cos_210_degrees : Real.cos (210 * π / 180) = -Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_210_degrees_l2727_272737


namespace NUMINAMATH_CALUDE_problem_1_problem_2_l2727_272797

-- Problem 1
theorem problem_1 : -9 / 3 + (1 / 2 - 2 / 3) * 12 - |(-4)^3| = -69 := by sorry

-- Problem 2
theorem problem_2 (a b : ℝ) : 2 * (a^2 + 2*b^2) - 3 * (2*a^2 - b^2) = -4*a^2 + 7*b^2 := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_l2727_272797


namespace NUMINAMATH_CALUDE_recipe_scaling_l2727_272791

def original_flour : ℚ := 20/3

theorem recipe_scaling :
  let scaled_flour : ℚ := (1/3) * original_flour
  let scaled_sugar : ℚ := (1/2) * scaled_flour
  scaled_flour = 20/9 ∧ scaled_sugar = 10/9 := by sorry

end NUMINAMATH_CALUDE_recipe_scaling_l2727_272791


namespace NUMINAMATH_CALUDE_shelter_adoption_percentage_l2727_272720

def initial_dogs : ℕ := 80
def returned_dogs : ℕ := 5
def final_dogs : ℕ := 53

def adoption_percentage : ℚ := 40

theorem shelter_adoption_percentage :
  (initial_dogs - (initial_dogs * adoption_percentage / 100) + returned_dogs : ℚ) = final_dogs :=
sorry

end NUMINAMATH_CALUDE_shelter_adoption_percentage_l2727_272720


namespace NUMINAMATH_CALUDE_complex_simplification_l2727_272799

theorem complex_simplification :
  (-5 + 3*I : ℂ) - (2 - 7*I) + (1 + 2*I) * (4 - 3*I) = 3 + 15*I :=
by sorry

end NUMINAMATH_CALUDE_complex_simplification_l2727_272799


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l2727_272702

theorem quadratic_inequality_range (a : ℝ) : 
  (¬ ∃ x : ℝ, x^2 + 2*a*x + a ≤ 0) ↔ (0 < a ∧ a < 1) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l2727_272702


namespace NUMINAMATH_CALUDE_solve_for_y_l2727_272777

theorem solve_for_y (x y : ℚ) (h1 : x = 202) (h2 : x^3 * y - 4 * x^2 * y + 2 * x * y = 808080) : y = 1/10 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_y_l2727_272777


namespace NUMINAMATH_CALUDE_sum_of_variables_l2727_272710

theorem sum_of_variables (a b c : ℚ) 
  (eq1 : b + c = 15 - 4*a)
  (eq2 : a + c = -18 - 4*b)
  (eq3 : a + b = 10 - 4*c) : 
  2*a + 2*b + 2*c = 7/3 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_variables_l2727_272710


namespace NUMINAMATH_CALUDE_inverse_proportion_inequality_l2727_272722

/-- Given two points on the inverse proportion function y = -4/x, 
    if the x-coordinate of the first point is negative and 
    the x-coordinate of the second point is positive, 
    then the y-coordinate of the first point is greater than 
    the y-coordinate of the second point. -/
theorem inverse_proportion_inequality (x₁ x₂ y₁ y₂ : ℝ) : 
  x₁ < 0 → 0 < x₂ → y₁ = -4 / x₁ → y₂ = -4 / x₂ → y₁ > y₂ := by
  sorry

end NUMINAMATH_CALUDE_inverse_proportion_inequality_l2727_272722


namespace NUMINAMATH_CALUDE_five_students_three_locations_l2727_272724

/-- The number of ways for a given number of students to choose from a given number of locations. -/
def num_ways (num_students : ℕ) (num_locations : ℕ) : ℕ := num_locations ^ num_students

/-- Theorem: Five students choosing from three locations results in 243 different ways. -/
theorem five_students_three_locations : num_ways 5 3 = 243 := by
  sorry

end NUMINAMATH_CALUDE_five_students_three_locations_l2727_272724


namespace NUMINAMATH_CALUDE_cos_sin_225_degrees_l2727_272752

theorem cos_sin_225_degrees :
  Real.cos (225 * π / 180) = -Real.sqrt 2 / 2 ∧
  Real.sin (225 * π / 180) = -Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_sin_225_degrees_l2727_272752


namespace NUMINAMATH_CALUDE_cone_surface_area_l2727_272721

/-- The surface area of a cone formed from a 270-degree sector of a circle with radius 20, divided by π, is 525. -/
theorem cone_surface_area (r : ℝ) (θ : ℝ) : 
  r = 20 → θ = 270 → (π * r^2 + π * r * (2 * π * r * θ / 360) / (2 * π)) / π = 525 := by
  sorry

end NUMINAMATH_CALUDE_cone_surface_area_l2727_272721


namespace NUMINAMATH_CALUDE_D_72_equals_81_l2727_272762

/-- D(n) represents the number of ways to write a positive integer n as a product of integers greater than 1, where the order matters. -/
def D (n : ℕ+) : ℕ :=
  sorry

/-- Theorem stating that D(72) = 81 -/
theorem D_72_equals_81 : D 72 = 81 := by
  sorry

end NUMINAMATH_CALUDE_D_72_equals_81_l2727_272762


namespace NUMINAMATH_CALUDE_cubic_function_properties_l2727_272703

-- Define the function f
def f (a b c : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + b*x + c

-- State the theorem
theorem cubic_function_properties (a b c x₀ : ℝ) 
  (h1 : f a b c (-1) = 0)
  (h2 : f a b c 1 = 0)
  (h3 : f a b c x₀ = 0)
  (h4 : 2 < x₀) (h5 : x₀ < 3) :
  (a + c = 0) ∧ (2 < c ∧ c < 3) ∧ (4*a + 2*b + c < -8) := by
  sorry


end NUMINAMATH_CALUDE_cubic_function_properties_l2727_272703


namespace NUMINAMATH_CALUDE_count_denominators_repeating_decimal_l2727_272761

/-- The number of different possible denominators for the fraction representation of a repeating decimal 0.ab̅ in lowest terms, where a and b are digits. -/
theorem count_denominators_repeating_decimal : ∃ (n : ℕ), n = 6 ∧ n = (Finset.image (λ (p : ℕ × ℕ) => (Nat.lcm 99 (10 * p.1 + p.2) / (10 * p.1 + p.2)).gcd 99) (Finset.filter (λ (p : ℕ × ℕ) => p.1 < 10 ∧ p.2 < 10) (Finset.product (Finset.range 10) (Finset.range 10)))).card := by
  sorry

end NUMINAMATH_CALUDE_count_denominators_repeating_decimal_l2727_272761


namespace NUMINAMATH_CALUDE_triangle_sum_theorem_l2727_272788

noncomputable def triangle_sum (AB AC BC CX₁ : ℝ) : ℝ :=
  let M := BC / 2
  let NC := (5 / 13) * CX₁
  let X₁C := Real.sqrt (CX₁^2 - NC^2)
  let BN := BC - NC
  let X₁B := Real.sqrt (BN^2 + X₁C^2)
  let X₂X₁ := X₁B * (16 / 63)
  let ratio := 1 - (X₁B * (65 / 63) / AB)
  (X₁B + X₂X₁) / (1 - ratio)

theorem triangle_sum_theorem (AB AC BC CX₁ : ℝ) 
  (h1 : AB = 182) (h2 : AC = 182) (h3 : BC = 140) (h4 : CX₁ = 130) :
  triangle_sum AB AC BC CX₁ = 1106 / 5 :=
by sorry

end NUMINAMATH_CALUDE_triangle_sum_theorem_l2727_272788


namespace NUMINAMATH_CALUDE_three_tangent_lines_l2727_272764

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 8*x

-- Define a line passing through (0,2)
def line_through_point (m b : ℝ) (x y : ℝ) : Prop := y = m*x + b ∧ 2 = b

-- Define the condition for a line to intersect the parabola at exactly one point
def intersects_once (m b : ℝ) : Prop :=
  ∃! x y, parabola x y ∧ line_through_point m b x y

-- The main theorem
theorem three_tangent_lines :
  ∃ L1 L2 L3 : ℝ × ℝ,
    L1 ≠ L2 ∧ L1 ≠ L3 ∧ L2 ≠ L3 ∧
    (∀ m b, intersects_once m b ↔ (m, b) = L1 ∨ (m, b) = L2 ∨ (m, b) = L3) :=
sorry

end NUMINAMATH_CALUDE_three_tangent_lines_l2727_272764


namespace NUMINAMATH_CALUDE_bounded_harmonic_constant_l2727_272728

/-- A function f: ℤ² → ℝ is harmonic if it satisfies the discrete Laplace equation -/
def Harmonic (f : ℤ × ℤ → ℝ) : Prop :=
  ∀ x y, f (x + 1, y) + f (x - 1, y) + f (x, y + 1) + f (x, y - 1) = 4 * f (x, y)

/-- A function f: ℤ² → ℝ is bounded if there exists a positive constant M such that |f(x, y)| ≤ M for all (x, y) in ℤ² -/
def Bounded (f : ℤ × ℤ → ℝ) : Prop :=
  ∃ M > 0, ∀ x y, |f (x, y)| ≤ M

/-- If a function f: ℤ² → ℝ is both harmonic and bounded, then it is constant -/
theorem bounded_harmonic_constant (f : ℤ × ℤ → ℝ) (hf_harmonic : Harmonic f) (hf_bounded : Bounded f) :
  ∃ c : ℝ, ∀ x y, f (x, y) = c :=
sorry

end NUMINAMATH_CALUDE_bounded_harmonic_constant_l2727_272728


namespace NUMINAMATH_CALUDE_probability_two_qualified_products_l2727_272745

theorem probability_two_qualified_products (total : ℕ) (qualified : ℕ) (unqualified : ℕ) 
  (h1 : total = qualified + unqualified)
  (h2 : total = 10)
  (h3 : qualified = 8)
  (h4 : unqualified = 2) :
  let p := (qualified - 1) / (total - 1)
  p = 7 / 11 := by
sorry

end NUMINAMATH_CALUDE_probability_two_qualified_products_l2727_272745


namespace NUMINAMATH_CALUDE_students_taking_no_subjects_l2727_272758

/-- Represents the number of students in various subject combinations --/
structure ScienceClub where
  total : ℕ
  math : ℕ
  physics : ℕ
  chemistry : ℕ
  math_physics : ℕ
  physics_chemistry : ℕ
  math_chemistry : ℕ
  all_three : ℕ

/-- Theorem stating the number of students taking no subjects --/
theorem students_taking_no_subjects (club : ScienceClub)
  (h_total : club.total = 150)
  (h_math : club.math = 85)
  (h_physics : club.physics = 63)
  (h_chemistry : club.chemistry = 40)
  (h_math_physics : club.math_physics = 20)
  (h_physics_chemistry : club.physics_chemistry = 15)
  (h_math_chemistry : club.math_chemistry = 10)
  (h_all_three : club.all_three = 5) :
  club.total - (club.math + club.physics + club.chemistry
    - club.math_physics - club.physics_chemistry - club.math_chemistry
    + club.all_three) = 2 := by
  sorry

#check students_taking_no_subjects

end NUMINAMATH_CALUDE_students_taking_no_subjects_l2727_272758


namespace NUMINAMATH_CALUDE_coin_fraction_missing_l2727_272727

theorem coin_fraction_missing (x : ℚ) : x > 0 →
  let lost := (1 / 3 : ℚ) * x
  let found := (3 / 4 : ℚ) * lost
  let remaining := x - lost + found
  x - remaining = (1 / 12 : ℚ) * x := by
  sorry

end NUMINAMATH_CALUDE_coin_fraction_missing_l2727_272727


namespace NUMINAMATH_CALUDE_solution_system_l2727_272798

theorem solution_system (x y : ℝ) 
  (eq1 : x + y = 10) 
  (eq2 : x / y = 7 / 3) : 
  x = 7 ∧ y = 3 := by
sorry

end NUMINAMATH_CALUDE_solution_system_l2727_272798


namespace NUMINAMATH_CALUDE_find_number_l2727_272716

theorem find_number : ∃ x : ℝ, ((x * 0.5 + 26.1) / 0.4) - 35 = 35 := by
  use 3.8
  sorry

end NUMINAMATH_CALUDE_find_number_l2727_272716


namespace NUMINAMATH_CALUDE_combined_mixture_indeterminate_l2727_272707

structure TrailMix where
  nuts : ℝ
  dried_fruit : ℝ
  chocolate_chips : ℝ
  pretzels : ℝ
  granola : ℝ
  sum_to_one : nuts + dried_fruit + chocolate_chips + pretzels + granola = 1

def sue_mix : TrailMix := {
  nuts := 0.3,
  dried_fruit := 0.7,
  chocolate_chips := 0,
  pretzels := 0,
  granola := 0,
  sum_to_one := by norm_num
}

def jane_mix : TrailMix := {
  nuts := 0.6,
  dried_fruit := 0,
  chocolate_chips := 0.3,
  pretzels := 0.1,
  granola := 0,
  sum_to_one := by norm_num
}

def tom_mix : TrailMix := {
  nuts := 0.4,
  dried_fruit := 0.5,
  chocolate_chips := 0,
  pretzels := 0,
  granola := 0.1,
  sum_to_one := by norm_num
}

theorem combined_mixture_indeterminate 
  (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (h_sum : a + b + c = 1) 
  (h_nuts : a * sue_mix.nuts + b * jane_mix.nuts + c * tom_mix.nuts = 0.45) :
  ∃ (x y : ℝ), 
    x ≠ y ∧ 
    (a * sue_mix.dried_fruit + b * jane_mix.dried_fruit + c * tom_mix.dried_fruit = x) ∧
    (a * sue_mix.dried_fruit + b * jane_mix.dried_fruit + c * tom_mix.dried_fruit = y) :=
sorry

end NUMINAMATH_CALUDE_combined_mixture_indeterminate_l2727_272707


namespace NUMINAMATH_CALUDE_perfect_square_problem_l2727_272776

theorem perfect_square_problem :
  (∃ x : ℝ, 6^2024 = x^2) ∧
  (∀ y : ℝ, 7^2025 ≠ y^2) ∧
  (∃ z : ℝ, 8^2026 = z^2) ∧
  (∃ w : ℝ, 9^2027 = w^2) ∧
  (∃ v : ℝ, 10^2028 = v^2) := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_problem_l2727_272776


namespace NUMINAMATH_CALUDE_pamphlets_total_l2727_272782

/-- Calculates the total number of pamphlets printed by Mike and Leo -/
def total_pamphlets (mike_initial_speed : ℕ) (mike_initial_hours : ℕ) (mike_final_hours : ℕ) : ℕ :=
  let mike_initial_pamphlets := mike_initial_speed * mike_initial_hours
  let mike_final_speed := mike_initial_speed / 3
  let mike_final_pamphlets := mike_final_speed * mike_final_hours
  let leo_hours := mike_initial_hours / 3
  let leo_speed := mike_initial_speed * 2
  let leo_pamphlets := leo_speed * leo_hours
  mike_initial_pamphlets + mike_final_pamphlets + leo_pamphlets

/-- Theorem stating that the total number of pamphlets printed is 9400 -/
theorem pamphlets_total : total_pamphlets 600 9 2 = 9400 := by
  sorry

end NUMINAMATH_CALUDE_pamphlets_total_l2727_272782


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l2727_272750

theorem quadratic_inequality_range (a : ℝ) : 
  (∀ x : ℝ, x^2 + 2*a*x + a > 0) ↔ (0 < a ∧ a < 1) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l2727_272750


namespace NUMINAMATH_CALUDE_smallest_next_divisor_after_437_l2727_272734

theorem smallest_next_divisor_after_437 (m : ℕ) (h1 : 10000 ≤ m ∧ m ≤ 99999) 
  (h2 : Odd m) (h3 : 437 ∣ m) :
  ∃ (d : ℕ), d ∣ m ∧ 437 < d ∧ d ≤ 874 ∧ ∀ (x : ℕ), x ∣ m → 437 < x → x ≥ d :=
by sorry

end NUMINAMATH_CALUDE_smallest_next_divisor_after_437_l2727_272734


namespace NUMINAMATH_CALUDE_sphere_surface_area_with_inscribed_cube_l2727_272713

theorem sphere_surface_area_with_inscribed_cube (cube_surface_area : ℝ) 
  (h : cube_surface_area = 54) : 
  ∃ (sphere_surface_area : ℝ), sphere_surface_area = 27 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_sphere_surface_area_with_inscribed_cube_l2727_272713


namespace NUMINAMATH_CALUDE_sqrt_sum_eq_sum_iff_two_zero_l2727_272789

theorem sqrt_sum_eq_sum_iff_two_zero (a b c : ℝ) (ha : a ≥ 0) (hb : b ≥ 0) (hc : c ≥ 0) :
  Real.sqrt (a^2 + b^2 + c^2) = a + b + c ↔ (a = 0 ∧ b = 0) ∨ (a = 0 ∧ c = 0) ∨ (b = 0 ∧ c = 0) :=
by sorry

end NUMINAMATH_CALUDE_sqrt_sum_eq_sum_iff_two_zero_l2727_272789


namespace NUMINAMATH_CALUDE_third_degree_polynomial_property_l2727_272767

/-- A third-degree polynomial with real coefficients -/
def ThirdDegreePolynomial := ℝ → ℝ

/-- The property that |f(x)| = 15 for x ∈ {2, 4, 5, 6, 8, 9} -/
def HasAbsoluteValue15 (f : ThirdDegreePolynomial) : Prop :=
  ∀ x ∈ ({2, 4, 5, 6, 8, 9} : Set ℝ), |f x| = 15

theorem third_degree_polynomial_property (f : ThirdDegreePolynomial) 
  (h : HasAbsoluteValue15 f) : |f 0| = 135 := by
  sorry

end NUMINAMATH_CALUDE_third_degree_polynomial_property_l2727_272767


namespace NUMINAMATH_CALUDE_simplify_fraction_product_l2727_272744

theorem simplify_fraction_product : 4 * (15 / 5) * (25 / -75) = -4 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_product_l2727_272744


namespace NUMINAMATH_CALUDE_tv_sales_decrease_l2727_272740

theorem tv_sales_decrease (original_price original_quantity : ℝ) 
  (h_price_increase : ℝ) (h_revenue_increase : ℝ) :
  original_price > 0 →
  original_quantity > 0 →
  h_price_increase = 0.6 →
  h_revenue_increase = 0.28 →
  let new_price := original_price * (1 + h_price_increase)
  let new_revenue := (1 + h_revenue_increase) * (original_price * original_quantity)
  let sales_decrease := 1 - (new_revenue / (new_price * original_quantity))
  sales_decrease = 0.2 := by
sorry

end NUMINAMATH_CALUDE_tv_sales_decrease_l2727_272740


namespace NUMINAMATH_CALUDE_student_selection_l2727_272785

theorem student_selection (n : ℕ) (h : n = 30) : 
  (Nat.choose n 2 = 435) ∧ (Nat.choose n 3 = 4060) := by
  sorry

#check student_selection

end NUMINAMATH_CALUDE_student_selection_l2727_272785


namespace NUMINAMATH_CALUDE_rectangles_in_5x5_grid_l2727_272780

/-- The number of dots in each row and column of the square array -/
def gridSize : Nat := 5

/-- The number of different rectangles that can be formed in the grid -/
def numRectangles : Nat := (gridSize.choose 2) * (gridSize.choose 2)

/-- Theorem stating the number of rectangles in a 5x5 grid -/
theorem rectangles_in_5x5_grid : numRectangles = 100 := by sorry

end NUMINAMATH_CALUDE_rectangles_in_5x5_grid_l2727_272780


namespace NUMINAMATH_CALUDE_polynomial_product_equals_difference_of_cubes_l2727_272772

theorem polynomial_product_equals_difference_of_cubes (x : ℝ) :
  (x^4 + 30*x^2 + 225) * (x^2 - 15) = x^6 - 3375 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_product_equals_difference_of_cubes_l2727_272772


namespace NUMINAMATH_CALUDE_units_digit_17_35_l2727_272701

theorem units_digit_17_35 : 17^35 % 10 = 3 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_17_35_l2727_272701


namespace NUMINAMATH_CALUDE_floor_length_percentage_l2727_272715

/-- Proves that for a rectangular floor with given length and area, 
    the percentage by which the length is more than the breadth is 200% -/
theorem floor_length_percentage (length : ℝ) (area : ℝ) :
  length = 19.595917942265423 →
  area = 128 →
  let breadth := area / length
  ((length - breadth) / breadth) * 100 = 200 := by sorry

end NUMINAMATH_CALUDE_floor_length_percentage_l2727_272715


namespace NUMINAMATH_CALUDE_simplify_complex_fraction_l2727_272729

theorem simplify_complex_fraction (b : ℝ) (h : b ≠ 2) :
  2 - (1 / (1 + b / (2 - b))) = 1 + b / 2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_complex_fraction_l2727_272729


namespace NUMINAMATH_CALUDE_triangle_abc_theorem_l2727_272717

noncomputable section

variables {a b c : ℝ} {A B C : ℝ} {O P : ℝ × ℝ}

def triangle_abc (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b > c ∧ b + c > a ∧ c + a > b

def angle_condition (a b c A B C : ℝ) : Prop :=
  a * Real.sin A + a * Real.sin C * Real.cos B + b * Real.sin C * Real.cos A = 
  b * Real.sin B + c * Real.sin A

def acute_triangle (A B C : ℝ) : Prop :=
  0 < A ∧ A < Real.pi/2 ∧ 0 < B ∧ B < Real.pi/2 ∧ 0 < C ∧ C < Real.pi/2

def circumradius (a b c : ℝ) : ℝ :=
  (a * b * c) / (4 * Real.sqrt ((a + b + c) * (-a + b + c) * (a - b + c) * (a + b - c)))

def dot_product (v w : ℝ × ℝ) : ℝ :=
  v.1 * w.1 + v.2 * w.2

theorem triangle_abc_theorem (a b c A B C : ℝ) (O P : ℝ × ℝ) :
  triangle_abc a b c →
  angle_condition a b c A B C →
  (a = 2 → acute_triangle A B C → 
    3 + Real.sqrt 3 < a + b + c ∧ a + b + c < 6 + 2 * Real.sqrt 3) →
  (b^2 = a*c → circumradius a b c = 2 → 
    -2 ≤ dot_product (P.1 - O.1, P.2 - O.2) (P.1 - O.1 - a, P.2 - O.2) ∧
    dot_product (P.1 - O.1, P.2 - O.2) (P.1 - O.1 - a, P.2 - O.2) ≤ 6) →
  B = Real.pi / 3 := by sorry

end

end NUMINAMATH_CALUDE_triangle_abc_theorem_l2727_272717


namespace NUMINAMATH_CALUDE_equation_solution_l2727_272771

theorem equation_solution : 
  ∃ x : ℚ, (3 + 2*x) / (1 + 2*x) - (5 + 2*x) / (7 + 2*x) = 1 - (4*x^2 - 2) / (7 + 16*x + 4*x^2) ∧ x = 7/8 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2727_272771


namespace NUMINAMATH_CALUDE_gwen_gave_away_seven_games_l2727_272705

/-- The number of games Gwen gave away -/
def games_given_away (initial : ℕ) (remaining : ℕ) : ℕ :=
  initial - remaining

/-- Proof that Gwen gave away 7 games -/
theorem gwen_gave_away_seven_games :
  let initial := 98
  let remaining := 91
  games_given_away initial remaining = 7 := by
  sorry

end NUMINAMATH_CALUDE_gwen_gave_away_seven_games_l2727_272705


namespace NUMINAMATH_CALUDE_tetrahedron_properties_l2727_272730

/-- A right prism with vertices A, B, C, A₁, B₁, C₁ -/
structure RightPrism (V : Type*) [NormedAddCommGroup V] [InnerProductSpace ℝ V] :=
  (A B C A₁ B₁ C₁ : V)
  (is_right_prism : sorry)

/-- Points P and P₁ on edges BB₁ and CC₁ respectively -/
structure PrismWithPoints (V : Type*) [NormedAddCommGroup V] [InnerProductSpace ℝ V] extends RightPrism V :=
  (P P₁ : V)
  (P_on_BB₁ : sorry)
  (P₁_on_CC₁ : sorry)
  (ratio_condition : sorry)

/-- The dihedral angle between two planes -/
def dihedral_angle (V : Type*) [NormedAddCommGroup V] [InnerProductSpace ℝ V] 
  (plane1 plane2 : Set V) : ℝ := sorry

/-- Theorem stating the properties of the tetrahedron AA₁PP₁ -/
theorem tetrahedron_properties 
  {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]
  (prism : PrismWithPoints V) :
  let A := prism.A
  let A₁ := prism.A₁
  let P := prism.P
  let P₁ := prism.P₁
  (dihedral_angle V {A, P₁, P} {A, A₁, P} = π / 2) ∧ 
  (dihedral_angle V {A₁, P, P₁} {A₁, A, P} = π / 2) ∧
  (dihedral_angle V {A, P, P₁} {A, A₁, P} + 
   dihedral_angle V {A, P, P₁} {A₁, P, P₁} + 
   dihedral_angle V {A₁, P₁, P} {A, A₁, P₁} = π) := by
  sorry

end NUMINAMATH_CALUDE_tetrahedron_properties_l2727_272730


namespace NUMINAMATH_CALUDE_rectangular_field_area_l2727_272704

theorem rectangular_field_area (width : ℝ) (length : ℝ) (perimeter : ℝ) (area : ℝ) : 
  width > 0 →
  length > 0 →
  width = length / 3 →
  perimeter = 2 * (width + length) →
  perimeter = 90 →
  area = width * length →
  area = 379.6875 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_field_area_l2727_272704
