import Mathlib

namespace NUMINAMATH_CALUDE_perpendicular_line_to_plane_l1224_122467

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the perpendicular relation between planes
variable (perp_planes : Plane → Plane → Prop)

-- Define the perpendicular relation between a line and a plane
variable (perp_line_plane : Line → Plane → Prop)

-- Define the perpendicular relation between lines
variable (perp_lines : Line → Line → Prop)

-- Define the relation for a line being contained in a plane
variable (line_in_plane : Line → Plane → Prop)

-- Define the intersection of two planes
variable (plane_intersection : Plane → Plane → Line)

-- Theorem statement
theorem perpendicular_line_to_plane 
  (α β : Plane) (l m : Line) 
  (h1 : perp_planes α β)
  (h2 : plane_intersection α β = l)
  (h3 : line_in_plane m α)
  (h4 : perp_lines m l) :
  perp_line_plane m β :=
sorry

end NUMINAMATH_CALUDE_perpendicular_line_to_plane_l1224_122467


namespace NUMINAMATH_CALUDE_line_ellipse_intersection_l1224_122409

/-- The line y = k(x-1) + 1 intersects the ellipse (x^2 / 9) + (y^2 / 4) = 1 for any real k -/
theorem line_ellipse_intersection (k : ℝ) :
  ∃ (x y : ℝ), y = k * (x - 1) + 1 ∧ (x^2 / 9) + (y^2 / 4) = 1 := by
  sorry

end NUMINAMATH_CALUDE_line_ellipse_intersection_l1224_122409


namespace NUMINAMATH_CALUDE_smallest_primes_satisfying_conditions_l1224_122400

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

theorem smallest_primes_satisfying_conditions (p q : ℕ) :
  is_prime p ∧ is_prime q ∧ is_prime (p * q + 1) ∧ p - q > 40 →
  p = 53 ∧ q = 2 :=
sorry

end NUMINAMATH_CALUDE_smallest_primes_satisfying_conditions_l1224_122400


namespace NUMINAMATH_CALUDE_exists_distinct_subsequences_l1224_122453

/-- A binary sequence is a function from ℕ to Bool -/
def BinarySequence := ℕ → Bool

/-- Cyclic index function to wrap around the sequence -/
def cyclicIndex (len : ℕ) (i : ℕ) : ℕ :=
  i % len

/-- Check if all n-length subsequences in a sequence of length 2^n are distinct -/
def allSubsequencesDistinct (n : ℕ) (seq : BinarySequence) : Prop :=
  ∀ i j, i < 2^n → j < 2^n → i ≠ j →
    (∃ k, k < n ∧ seq (cyclicIndex (2^n) (i + k)) ≠ seq (cyclicIndex (2^n) (j + k)))

/-- Main theorem: For any positive n, there exists a binary sequence of length 2^n
    where all n-length subsequences are distinct when considered cyclically -/
theorem exists_distinct_subsequences (n : ℕ) (hn : n > 0) :
  ∃ seq : BinarySequence, allSubsequencesDistinct n seq :=
sorry

end NUMINAMATH_CALUDE_exists_distinct_subsequences_l1224_122453


namespace NUMINAMATH_CALUDE_solve_equation_l1224_122422

theorem solve_equation (x : ℚ) : (4/7 : ℚ) * (1/5 : ℚ) * x = 12 → x = 105 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l1224_122422


namespace NUMINAMATH_CALUDE_smallest_divisible_k_l1224_122457

/-- The polynomial p(z) = z^12 + z^11 + z^8 + z^7 + z^6 + z^3 + 1 -/
def p (z : ℂ) : ℂ := z^12 + z^11 + z^8 + z^7 + z^6 + z^3 + 1

/-- The function f(k) = z^k - 1 -/
def f (k : ℕ) (z : ℂ) : ℂ := z^k - 1

/-- Theorem stating that 120 is the smallest positive integer k such that p(z) divides f(k)(z) -/
theorem smallest_divisible_k : 
  (∀ z : ℂ, p z ∣ f 120 z) ∧ 
  (∀ k : ℕ, k < 120 → ∃ z : ℂ, ¬(p z ∣ f k z)) :=
sorry

end NUMINAMATH_CALUDE_smallest_divisible_k_l1224_122457


namespace NUMINAMATH_CALUDE_largest_difference_l1224_122499

def A : ℕ := 3 * 2010^2011
def B : ℕ := 2010^2011
def C : ℕ := 2009 * 2010^2010
def D : ℕ := 3 * 2010^2010
def E : ℕ := 2010^2010
def F : ℕ := 2010^2009

theorem largest_difference : 
  (A - B > B - C) ∧ 
  (A - B > C - D) ∧ 
  (A - B > D - E) ∧ 
  (A - B > E - F) := by sorry

end NUMINAMATH_CALUDE_largest_difference_l1224_122499


namespace NUMINAMATH_CALUDE_conditions_necessary_not_sufficient_l1224_122438

theorem conditions_necessary_not_sufficient :
  (∀ x y : ℝ, (2 < x ∧ x < 3 ∧ 0 < y ∧ y < 1) → (2 < x + y ∧ x + y < 4 ∧ 0 < x * y ∧ x * y < 3)) ∧
  (∃ x y : ℝ, (2 < x + y ∧ x + y < 4 ∧ 0 < x * y ∧ x * y < 3) ∧ ¬(2 < x ∧ x < 3 ∧ 0 < y ∧ y < 1)) :=
by sorry

end NUMINAMATH_CALUDE_conditions_necessary_not_sufficient_l1224_122438


namespace NUMINAMATH_CALUDE_parabola_tangent_to_line_l1224_122494

/-- A parabola y = ax^2 + 8 is tangent to the line y = 2x + 3 if and only if a = 1/5 -/
theorem parabola_tangent_to_line (a : ℝ) : 
  (∃! x : ℝ, a * x^2 + 8 = 2 * x + 3) ↔ a = 1/5 := by
  sorry

end NUMINAMATH_CALUDE_parabola_tangent_to_line_l1224_122494


namespace NUMINAMATH_CALUDE_mango_rate_per_kg_l1224_122437

/-- The rate per kg of mangoes given the purchase details -/
theorem mango_rate_per_kg
  (grape_kg : ℕ)
  (grape_rate : ℕ)
  (mango_kg : ℕ)
  (total_paid : ℕ)
  (h1 : grape_kg = 8)
  (h2 : grape_rate = 70)
  (h3 : mango_kg = 9)
  (h4 : total_paid = 1055)
  : (total_paid - grape_kg * grape_rate) / mango_kg = 55 := by
  sorry

end NUMINAMATH_CALUDE_mango_rate_per_kg_l1224_122437


namespace NUMINAMATH_CALUDE_frog_hop_probability_l1224_122448

/-- Represents a position on the 4x4 grid -/
structure Position :=
  (x : Fin 4)
  (y : Fin 4)

/-- Defines whether a position is on the edge of the grid -/
def isEdge (p : Position) : Bool :=
  p.x = 0 || p.x = 3 || p.y = 0 || p.y = 3

/-- Represents a single hop in one of the four cardinal directions -/
inductive Direction
  | Up
  | Down
  | Left
  | Right

/-- Applies a hop in the given direction, wrapping around if necessary -/
def hop (p : Position) (d : Direction) : Position :=
  match d with
  | Direction.Up => ⟨p.x - 1, p.y⟩
  | Direction.Down => ⟨p.x + 1, p.y⟩
  | Direction.Left => ⟨p.x, p.y - 1⟩
  | Direction.Right => ⟨p.x, p.y + 1⟩

/-- The probability of ending on an edge after three hops -/
def probEndOnEdge (start : Position) : ℚ :=
  sorry

theorem frog_hop_probability :
  probEndOnEdge ⟨1, 1⟩ = 37 / 64 := by sorry

end NUMINAMATH_CALUDE_frog_hop_probability_l1224_122448


namespace NUMINAMATH_CALUDE_race_time_difference_l1224_122416

def race_length : ℝ := 15
def malcolm_speed : ℝ := 6
def joshua_speed : ℝ := 7

theorem race_time_difference : 
  let malcolm_time := race_length * malcolm_speed
  let joshua_time := race_length * joshua_speed
  joshua_time - malcolm_time = 15 := by
sorry

end NUMINAMATH_CALUDE_race_time_difference_l1224_122416


namespace NUMINAMATH_CALUDE_equation_holds_iff_k_equals_neg_15_l1224_122493

theorem equation_holds_iff_k_equals_neg_15 :
  (∀ x : ℝ, -x^2 - (k + 9)*x - 8 = -(x - 2)*(x - 4)) ↔ k = -15 :=
by sorry

end NUMINAMATH_CALUDE_equation_holds_iff_k_equals_neg_15_l1224_122493


namespace NUMINAMATH_CALUDE_sphere_radius_relation_l1224_122402

/-- Given two spheres, one with radius 5 cm and another with 3 times its volume,
    prove that the radius of the larger sphere is 5 * (3^(1/3)) cm. -/
theorem sphere_radius_relation :
  ∀ (r : ℝ),
  (4 / 3 * Real.pi * r^3 = 3 * (4 / 3 * Real.pi * 5^3)) →
  r = 5 * (3^(1 / 3)) :=
by sorry

end NUMINAMATH_CALUDE_sphere_radius_relation_l1224_122402


namespace NUMINAMATH_CALUDE_carpool_commute_days_l1224_122458

/-- Proves that the number of commuting days per week is 5 given the carpool conditions --/
theorem carpool_commute_days : 
  let total_commute : ℝ := 21 -- miles one way
  let gas_cost : ℝ := 2.5 -- $/gallon
  let car_efficiency : ℝ := 30 -- miles/gallon
  let weeks_per_month : ℕ := 4
  let individual_payment : ℝ := 14 -- $ per month
  let num_friends : ℕ := 5
  
  -- Calculate the number of commuting days per week
  let commute_days : ℝ := 
    (individual_payment * num_friends) / 
    (gas_cost * (2 * total_commute / car_efficiency) * weeks_per_month)
  
  commute_days = 5 := by
  sorry

end NUMINAMATH_CALUDE_carpool_commute_days_l1224_122458


namespace NUMINAMATH_CALUDE_problem_solution_l1224_122470

theorem problem_solution (X : ℝ) : 
  (213 * 16 = 3408) → 
  ((213 * 16) + (1.6 * 2.13) = X) → 
  (X - (5/2) * 1.25 = 3408.283) := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l1224_122470


namespace NUMINAMATH_CALUDE_number_of_divisors_32_l1224_122412

theorem number_of_divisors_32 : Finset.card (Nat.divisors 32) = 6 := by
  sorry

end NUMINAMATH_CALUDE_number_of_divisors_32_l1224_122412


namespace NUMINAMATH_CALUDE_fraction_inequality_range_l1224_122401

theorem fraction_inequality_range (m : ℝ) : 
  (∀ x : ℝ, (3 * x^2 + 2 * x + 2) / (x^2 + x + 1) ≥ m) ↔ m ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_inequality_range_l1224_122401


namespace NUMINAMATH_CALUDE_cube_root_four_solves_equation_l1224_122468

theorem cube_root_four_solves_equation :
  let x : ℝ := (4 : ℝ) ^ (1/3)
  x^3 - ⌊x⌋ = 3 := by sorry

end NUMINAMATH_CALUDE_cube_root_four_solves_equation_l1224_122468


namespace NUMINAMATH_CALUDE_event_attendance_l1224_122462

/-- Given an event with a total of 42 people where the number of children is twice the number of adults,
    prove that the number of children is 28. -/
theorem event_attendance (total : ℕ) (adults : ℕ) (children : ℕ)
    (h1 : total = 42)
    (h2 : total = adults + children)
    (h3 : children = 2 * adults) :
    children = 28 := by
  sorry

end NUMINAMATH_CALUDE_event_attendance_l1224_122462


namespace NUMINAMATH_CALUDE_solve_ticket_problem_l1224_122486

/-- Represents the cost of tickets and number of students for two teachers. -/
structure TicketInfo where
  student_price : ℕ
  adult_price : ℕ
  kadrnozka_students : ℕ
  hnizdo_students : ℕ

/-- Checks if the given TicketInfo satisfies all the problem conditions. -/
def satisfies_conditions (info : TicketInfo) : Prop :=
  info.adult_price > info.student_price ∧
  info.adult_price ≤ 2 * info.student_price ∧
  info.student_price * info.kadrnozka_students + info.adult_price = 994 ∧
  info.hnizdo_students = info.kadrnozka_students + 3 ∧
  info.student_price * info.hnizdo_students + info.adult_price = 1120

/-- Theorem stating the solution to the problem. -/
theorem solve_ticket_problem :
  ∃ (info : TicketInfo), satisfies_conditions info ∧ 
    info.hnizdo_students = 25 ∧ info.adult_price = 70 :=
by
  sorry


end NUMINAMATH_CALUDE_solve_ticket_problem_l1224_122486


namespace NUMINAMATH_CALUDE_max_cube_volume_from_sheet_l1224_122455

/-- Given a rectangular sheet of dimensions 60 cm by 25 cm, 
    prove that the maximum volume of a cube that can be constructed from this sheet is 3375 cm³. -/
theorem max_cube_volume_from_sheet (sheet_length : ℝ) (sheet_width : ℝ) 
  (h_length : sheet_length = 60) (h_width : sheet_width = 25) :
  ∃ (cube_edge : ℝ), 
    cube_edge > 0 ∧
    6 * cube_edge^2 ≤ sheet_length * sheet_width ∧
    ∀ (other_edge : ℝ), 
      other_edge > 0 → 
      6 * other_edge^2 ≤ sheet_length * sheet_width → 
      other_edge^3 ≤ cube_edge^3 ∧
    cube_edge^3 = 3375 := by
  sorry

end NUMINAMATH_CALUDE_max_cube_volume_from_sheet_l1224_122455


namespace NUMINAMATH_CALUDE_jasons_commute_distance_l1224_122466

/-- Jason's commute to work problem -/
theorem jasons_commute_distance : ∀ (d1 d2 d3 d4 d5 : ℝ),
  d1 = 6 →                           -- Distance between first and second store
  d2 = d1 + (2/3 * d1) →             -- Distance between second and third store
  d3 = 4 →                           -- Distance from house to first store
  d4 = 4 →                           -- Distance from last store to work
  d5 = d1 + d2 + d3 + d4 →           -- Total commute distance
  d5 = 24 := by sorry

end NUMINAMATH_CALUDE_jasons_commute_distance_l1224_122466


namespace NUMINAMATH_CALUDE_max_smaller_cuboids_l1224_122450

/-- Represents the dimensions of a cuboid -/
structure CuboidDimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Calculates the volume of a cuboid given its dimensions -/
def cuboidVolume (d : CuboidDimensions) : ℕ :=
  d.length * d.width * d.height

/-- The dimensions of the smaller cuboid -/
def smallCuboid : CuboidDimensions :=
  { length := 6, width := 4, height := 3 }

/-- The dimensions of the larger cuboid -/
def largeCuboid : CuboidDimensions :=
  { length := 18, width := 15, height := 2 }

/-- Theorem stating the maximum number of whole smaller cuboids that can be formed -/
theorem max_smaller_cuboids :
  (cuboidVolume largeCuboid) / (cuboidVolume smallCuboid) = 7 :=
sorry

end NUMINAMATH_CALUDE_max_smaller_cuboids_l1224_122450


namespace NUMINAMATH_CALUDE_no_solution_condition_l1224_122435

theorem no_solution_condition (a : ℝ) : 
  (∀ x : ℝ, ¬((a^2 * x + 2*a) / (a*x - 2 + a^2) ≥ 0 ∧ a*x + a > 5/4)) ↔ 
  (a ≤ -1/2 ∨ a = 0) :=
sorry

end NUMINAMATH_CALUDE_no_solution_condition_l1224_122435


namespace NUMINAMATH_CALUDE_angle_measures_l1224_122478

-- Define a triangle ABC
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the properties of the triangle
def valid_triangle (t : Triangle) : Prop :=
  t.A > 0 ∧ t.B > 0 ∧ t.C > 0 ∧ t.A + t.B + t.C = 180

-- Define the ratio condition
def ratio_condition (t : Triangle) : Prop :=
  t.A / t.B = 1 / 2 ∧ t.B / t.C = 2 / 3

-- Theorem statement
theorem angle_measures (t : Triangle) 
  (h1 : valid_triangle t) (h2 : ratio_condition t) : 
  t.A = 30 ∧ t.B = 60 ∧ t.C = 90 :=
sorry

end NUMINAMATH_CALUDE_angle_measures_l1224_122478


namespace NUMINAMATH_CALUDE_book_purchase_problem_l1224_122419

/-- Proves that given the conditions of the book purchase problem, the number of math books is 53. -/
theorem book_purchase_problem (total_books : ℕ) (math_cost history_cost total_price : ℚ) 
  (h_total : total_books = 90)
  (h_math_cost : math_cost = 4)
  (h_history_cost : history_cost = 5)
  (h_total_price : total_price = 397) :
  ∃ (math_books : ℕ), 
    math_books = 53 ∧ 
    math_books ≤ total_books ∧
    ∃ (history_books : ℕ),
      history_books = total_books - math_books ∧
      math_cost * math_books + history_cost * history_books = total_price := by
sorry

end NUMINAMATH_CALUDE_book_purchase_problem_l1224_122419


namespace NUMINAMATH_CALUDE_dress_discount_price_l1224_122487

theorem dress_discount_price (original_price discount_percentage : ℝ) 
  (h1 : original_price = 50)
  (h2 : discount_percentage = 30) : 
  original_price * (1 - discount_percentage / 100) = 35 := by
  sorry

end NUMINAMATH_CALUDE_dress_discount_price_l1224_122487


namespace NUMINAMATH_CALUDE_paper_string_area_l1224_122484

/-- The area of a paper string made from overlapping square sheets -/
theorem paper_string_area
  (num_sheets : ℕ)
  (sheet_side : ℝ)
  (overlap : ℝ)
  (h_num_sheets : num_sheets = 6)
  (h_sheet_side : sheet_side = 30)
  (h_overlap : overlap = 7) :
  (sheet_side + (num_sheets - 1) * (sheet_side - overlap)) * sheet_side = 4350 :=
sorry

end NUMINAMATH_CALUDE_paper_string_area_l1224_122484


namespace NUMINAMATH_CALUDE_tan_alpha_values_l1224_122452

theorem tan_alpha_values (α : Real) (h : 2 * Real.sin (2 * α) = 1 - Real.cos (2 * α)) :
  Real.tan α = 2 ∨ Real.tan α = 0 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_values_l1224_122452


namespace NUMINAMATH_CALUDE_two_cubic_feet_to_cubic_inches_l1224_122441

-- Define the conversion factor
def inches_per_foot : ℕ := 12

-- Define the volume conversion
def cubic_inches_per_cubic_foot : ℕ := inches_per_foot ^ 3

-- Theorem statement
theorem two_cubic_feet_to_cubic_inches :
  2 * cubic_inches_per_cubic_foot = 3456 := by
  sorry

end NUMINAMATH_CALUDE_two_cubic_feet_to_cubic_inches_l1224_122441


namespace NUMINAMATH_CALUDE_average_monthly_growth_rate_l1224_122480

theorem average_monthly_growth_rate 
  (initial_sales : ℝ) 
  (final_sales : ℝ) 
  (months : ℕ) 
  (h1 : initial_sales = 5000)
  (h2 : final_sales = 7200)
  (h3 : months = 2) :
  ∃ (rate : ℝ), 
    rate = 1/5 ∧ 
    initial_sales * (1 + rate) ^ months = final_sales :=
by sorry

end NUMINAMATH_CALUDE_average_monthly_growth_rate_l1224_122480


namespace NUMINAMATH_CALUDE_sum_of_radii_l1224_122461

noncomputable section

-- Define the circle radius
def R : ℝ := 5

-- Define the ratios of the sectors
def ratio1 : ℝ := 1
def ratio2 : ℝ := 2
def ratio3 : ℝ := 3

-- Define the base radii of the cones
def r₁ : ℝ := (ratio1 / (ratio1 + ratio2 + ratio3)) * R
def r₂ : ℝ := (ratio2 / (ratio1 + ratio2 + ratio3)) * R
def r₃ : ℝ := (ratio3 / (ratio1 + ratio2 + ratio3)) * R

theorem sum_of_radii : r₁ + r₂ + r₃ = R := by
  sorry

end NUMINAMATH_CALUDE_sum_of_radii_l1224_122461


namespace NUMINAMATH_CALUDE_total_guitars_count_l1224_122414

/-- The number of guitars owned by Davey -/
def daveys_guitars : ℕ := 18

/-- The number of guitars owned by Barbeck -/
def barbecks_guitars : ℕ := daveys_guitars / 3

/-- The number of guitars owned by Steve -/
def steves_guitars : ℕ := barbecks_guitars / 2

/-- The total number of guitars -/
def total_guitars : ℕ := daveys_guitars + barbecks_guitars + steves_guitars

theorem total_guitars_count : total_guitars = 27 := by
  sorry

end NUMINAMATH_CALUDE_total_guitars_count_l1224_122414


namespace NUMINAMATH_CALUDE_negation_of_conditional_l1224_122413

theorem negation_of_conditional (a : ℝ) :
  ¬(a > 0 → a^2 > 0) ↔ (a ≤ 0 → a^2 ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_conditional_l1224_122413


namespace NUMINAMATH_CALUDE_amanda_hiking_trip_l1224_122447

/-- Represents Amanda's hiking trip -/
def hiking_trip (total_distance : ℚ) : Prop :=
  let first_segment := total_distance / 4
  let forest_segment := 25
  let mountain_segment := total_distance / 6
  let plain_segment := 2 * forest_segment
  first_segment + forest_segment + mountain_segment + plain_segment = total_distance

theorem amanda_hiking_trip :
  ∃ (total_distance : ℚ), hiking_trip total_distance ∧ total_distance = 900 / 7 := by
  sorry

end NUMINAMATH_CALUDE_amanda_hiking_trip_l1224_122447


namespace NUMINAMATH_CALUDE_even_sum_probability_l1224_122415

/-- Represents a wheel with a given number of even and odd sections -/
structure Wheel where
  total : ℕ
  even : ℕ
  odd : ℕ
  h1 : even + odd = total
  h2 : 0 < total

/-- The probability of getting an even number on a wheel -/
def prob_even (w : Wheel) : ℚ :=
  w.even / w.total

/-- The probability of getting an odd number on a wheel -/
def prob_odd (w : Wheel) : ℚ :=
  w.odd / w.total

/-- Wheel A with 2 even and 3 odd sections -/
def wheel_a : Wheel :=
  { total := 5
  , even := 2
  , odd := 3
  , h1 := by simp
  , h2 := by simp }

/-- Wheel B with 1 even and 1 odd section -/
def wheel_b : Wheel :=
  { total := 2
  , even := 1
  , odd := 1
  , h1 := by simp
  , h2 := by simp }

/-- The probability of getting an even sum when spinning both wheels -/
def prob_even_sum (a b : Wheel) : ℚ :=
  prob_even a * prob_even b + prob_odd a * prob_odd b

theorem even_sum_probability :
  prob_even_sum wheel_a wheel_b = 1/2 := by
  sorry


end NUMINAMATH_CALUDE_even_sum_probability_l1224_122415


namespace NUMINAMATH_CALUDE_smallest_digit_sum_of_sum_l1224_122485

/-- A three-digit positive integer -/
def ThreeDigitInt := {n : ℕ // 100 ≤ n ∧ n < 1000}

/-- Extracts digits from a natural number -/
def digits (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
  let rec extract (m : ℕ) (acc : List ℕ) : List ℕ :=
    if m = 0 then acc else extract (m / 10) ((m % 10) :: acc)
  extract n []

/-- Sum of digits of a natural number -/
def digitSum (n : ℕ) : ℕ := (digits n).sum

/-- All digits in two numbers are different -/
def allDigitsDifferent (a b : ThreeDigitInt) : Prop :=
  (digits a.val ++ digits b.val).Nodup

theorem smallest_digit_sum_of_sum (a b : ThreeDigitInt) 
  (h : allDigitsDifferent a b) : 
  ∃ (S : ℕ), S = a.val + b.val ∧ 1000 ≤ S ∧ S < 10000 ∧ 
  (∀ (T : ℕ), T = a.val + b.val → 1000 ≤ T ∧ T < 10000 → digitSum S ≤ digitSum T) ∧
  digitSum S = 8 :=
sorry

end NUMINAMATH_CALUDE_smallest_digit_sum_of_sum_l1224_122485


namespace NUMINAMATH_CALUDE_system_solution_l1224_122427

theorem system_solution : ∃ (x y : ℚ), 
  (3 * x - 4 * y = -7) ∧ 
  (4 * x - 3 * y = 5) ∧ 
  (x = 41 / 7) ∧ 
  (y = 43 / 7) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l1224_122427


namespace NUMINAMATH_CALUDE_equation_solution_implies_m_range_l1224_122417

theorem equation_solution_implies_m_range :
  ∀ m : ℝ,
  (∃ x : ℝ, 2^(2*x) + (m^2 - 2*m - 5)*2^x + 1 = 0) →
  m ∈ Set.Icc (-1 : ℝ) 3 := by
sorry

end NUMINAMATH_CALUDE_equation_solution_implies_m_range_l1224_122417


namespace NUMINAMATH_CALUDE_stratified_sampling_high_school_l1224_122476

theorem stratified_sampling_high_school
  (total_students : ℕ)
  (freshmen : ℕ)
  (sophomores : ℕ)
  (sample_size : ℕ)
  (h_total : total_students = 950)
  (h_freshmen : freshmen = 350)
  (h_sophomores : sophomores = 400)
  (h_sample : sample_size = 190) :
  let juniors := total_students - freshmen - sophomores
  let sample_ratio := sample_size / total_students
  let freshmen_sample := (sample_ratio * freshmen : ℚ).num
  let sophomores_sample := (sample_ratio * sophomores : ℚ).num
  let juniors_sample := (sample_ratio * juniors : ℚ).num
  (freshmen_sample, sophomores_sample, juniors_sample) = (70, 80, 40) := by
sorry

end NUMINAMATH_CALUDE_stratified_sampling_high_school_l1224_122476


namespace NUMINAMATH_CALUDE_min_floor_sum_l1224_122407

theorem min_floor_sum (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  ∃ (a₀ b₀ c₀ : ℝ) (ha₀ : a₀ > 0) (hb₀ : b₀ > 0) (hc₀ : c₀ > 0),
    (⌊(2*a₀+b₀)/c₀⌋ + ⌊(b₀+2*c₀)/a₀⌋ + ⌊(c₀+2*a₀)/b₀⌋ = 9) ∧
    (∀ (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0),
      ⌊(2*x+y)/z⌋ + ⌊(y+2*z)/x⌋ + ⌊(z+2*x)/y⌋ ≥ 9) :=
by sorry

end NUMINAMATH_CALUDE_min_floor_sum_l1224_122407


namespace NUMINAMATH_CALUDE_sqrt_4_equals_plus_minus_2_cube_root_negative_8_over_27_equals_negative_2_over_3_sqrt_diff_equals_point_1_abs_sqrt_2_minus_1_equals_sqrt_2_minus_1_l1224_122465

-- 1. Prove that ±√4 = ±2
theorem sqrt_4_equals_plus_minus_2 : ∀ x : ℝ, x^2 = 4 ↔ x = 2 ∨ x = -2 := by sorry

-- 2. Prove that ∛(-8/27) = -2/3
theorem cube_root_negative_8_over_27_equals_negative_2_over_3 : 
  ((-8/27 : ℝ) ^ (1/3 : ℝ)) = -2/3 := by sorry

-- 3. Prove that √0.09 - √0.04 = 0.1
theorem sqrt_diff_equals_point_1 : 
  Real.sqrt 0.09 - Real.sqrt 0.04 = 0.1 := by sorry

-- 4. Prove that |√2 - 1| = √2 - 1
theorem abs_sqrt_2_minus_1_equals_sqrt_2_minus_1 : 
  |Real.sqrt 2 - 1| = Real.sqrt 2 - 1 := by sorry

end NUMINAMATH_CALUDE_sqrt_4_equals_plus_minus_2_cube_root_negative_8_over_27_equals_negative_2_over_3_sqrt_diff_equals_point_1_abs_sqrt_2_minus_1_equals_sqrt_2_minus_1_l1224_122465


namespace NUMINAMATH_CALUDE_sum_of_reciprocals_l1224_122411

theorem sum_of_reciprocals (x y : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (h : x + y = x * y - 1) :
  1 / x + 1 / y = 1 - 1 / (x * y) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_reciprocals_l1224_122411


namespace NUMINAMATH_CALUDE_sqrt_2m_minus_n_equals_sqrt_2_l1224_122495

theorem sqrt_2m_minus_n_equals_sqrt_2 (m n : ℝ) : 
  (2 * m + n = 8 ∧ 2 * n - m = 1) → Real.sqrt (2 * m - n) = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_2m_minus_n_equals_sqrt_2_l1224_122495


namespace NUMINAMATH_CALUDE_janes_profit_is_correct_l1224_122440

/-- Farm data -/
structure FarmData where
  chickenCount : ℕ
  duckCount : ℕ
  quailCount : ℕ
  chickenEggsPerWeek : ℕ
  duckEggsPerWeek : ℕ
  quailEggsPerWeek : ℕ
  chickenEggPrice : ℚ
  duckEggPrice : ℚ
  quailEggPrice : ℚ
  chickenFeedCost : ℚ
  duckFeedCost : ℚ
  quailFeedCost : ℚ

/-- Sales data for a week -/
structure WeeklySales where
  chickenEggsSoldPercent : ℚ
  duckEggsSoldPercent : ℚ
  quailEggsSoldPercent : ℚ

def calculateProfit (farm : FarmData) (sales : List WeeklySales) : ℚ :=
  sorry

def janesFarm : FarmData := {
  chickenCount := 10,
  duckCount := 8,
  quailCount := 12,
  chickenEggsPerWeek := 6,
  duckEggsPerWeek := 4,
  quailEggsPerWeek := 10,
  chickenEggPrice := 2 / 12,
  duckEggPrice := 3 / 12,
  quailEggPrice := 4 / 12,
  chickenFeedCost := 1 / 2,
  duckFeedCost := 3 / 4,
  quailFeedCost := 3 / 5
}

def janesSales : List WeeklySales := [
  { chickenEggsSoldPercent := 1, duckEggsSoldPercent := 1, quailEggsSoldPercent := 1/2 },
  { chickenEggsSoldPercent := 1, duckEggsSoldPercent := 3/4, quailEggsSoldPercent := 1 },
  { chickenEggsSoldPercent := 0, duckEggsSoldPercent := 1, quailEggsSoldPercent := 1 }
]

theorem janes_profit_is_correct :
  calculateProfit janesFarm janesSales = 876 / 10 := by
  sorry

end NUMINAMATH_CALUDE_janes_profit_is_correct_l1224_122440


namespace NUMINAMATH_CALUDE_jamie_father_burns_500_calories_l1224_122456

/-- The number of calories in a pound of body fat -/
def calories_per_pound : ℕ := 3500

/-- The number of pounds Jamie's father wants to lose -/
def pounds_to_lose : ℕ := 5

/-- The number of days it takes Jamie's father to burn off the weight -/
def days_to_burn : ℕ := 35

/-- The number of calories Jamie's father eats per day -/
def calories_eaten_daily : ℕ := 2000

/-- The number of calories Jamie's father burns daily through light exercise -/
def calories_burned_daily : ℕ := (pounds_to_lose * calories_per_pound) / days_to_burn

theorem jamie_father_burns_500_calories :
  calories_burned_daily = 500 :=
sorry

end NUMINAMATH_CALUDE_jamie_father_burns_500_calories_l1224_122456


namespace NUMINAMATH_CALUDE_baseball_league_games_l1224_122483

theorem baseball_league_games (N M : ℕ) : 
  (N > 2 * M) → 
  (M > 4) → 
  (4 * N + 5 * M = 94) → 
  (4 * N = 64) := by
sorry

end NUMINAMATH_CALUDE_baseball_league_games_l1224_122483


namespace NUMINAMATH_CALUDE_exponent_division_l1224_122479

theorem exponent_division (a : ℝ) : a^7 / a^4 = a^3 := by
  sorry

end NUMINAMATH_CALUDE_exponent_division_l1224_122479


namespace NUMINAMATH_CALUDE_lines_properties_l1224_122497

-- Define the lines l₁ and l₂
def l₁ (a x y : ℝ) : Prop := a * x - y + 1 = 0
def l₂ (a x y : ℝ) : Prop := x + a * y + 1 = 0

-- Theorem statement
theorem lines_properties (a : ℝ) :
  -- 1. Perpendicularity condition
  (a ≠ 0 → (∀ x₁ y₁ x₂ y₂ : ℝ, l₁ a x₁ y₁ ∧ l₂ a x₂ y₂ → (x₂ - x₁) * (y₂ - y₁) = -1)) ∧
  -- 2. Fixed points condition
  (l₁ a 0 1 ∧ l₂ a (-1) 0) ∧
  -- 3. Maximum distance condition
  (∀ x y : ℝ, l₁ a x y ∧ l₂ a x y → x^2 + y^2 ≤ 2) :=
by sorry

end NUMINAMATH_CALUDE_lines_properties_l1224_122497


namespace NUMINAMATH_CALUDE_candy_problem_l1224_122444

/-- The number of candies left in Shelly's bowl before her friend came over -/
def initial_candies : ℕ := 63

/-- The number of candies Shelly's friend brought -/
def friend_candies : ℕ := 2 * initial_candies

/-- The total number of candies after the friend's contribution -/
def total_candies : ℕ := initial_candies + friend_candies

/-- The number of candies Shelly's friend had after eating 10 -/
def friend_final_candies : ℕ := 85

theorem candy_problem :
  initial_candies = 63 ∧
  friend_candies = 2 * initial_candies ∧
  total_candies = initial_candies + friend_candies ∧
  friend_final_candies + 10 = total_candies / 2 :=
sorry

end NUMINAMATH_CALUDE_candy_problem_l1224_122444


namespace NUMINAMATH_CALUDE_michelle_oranges_l1224_122420

theorem michelle_oranges :
  ∀ (total : ℕ),
  (total / 3 : ℚ) + 5 + 7 = total →
  total = 18 :=
by
  sorry

end NUMINAMATH_CALUDE_michelle_oranges_l1224_122420


namespace NUMINAMATH_CALUDE_fixed_point_of_parabolas_l1224_122433

/-- The function f_m that defines the family of parabolas -/
def f_m (m : ℝ) (x : ℝ) : ℝ := (m^2 + m + 1) * x^2 - 2 * (m^2 + 1) * x + m^2 - m + 1

/-- Theorem stating that (1, 0) is the fixed common point of all parabolas -/
theorem fixed_point_of_parabolas :
  ∀ m : ℝ, f_m m 1 = 1 := by sorry

end NUMINAMATH_CALUDE_fixed_point_of_parabolas_l1224_122433


namespace NUMINAMATH_CALUDE_m_xor_n_equals_target_l1224_122471

-- Define the custom set operation ⊗
def setXor (A B : Set ℝ) : Set ℝ := (A ∪ B) \ (A ∩ B)

-- Define sets M and N
def M : Set ℝ := {x | |x| < 2}
def N : Set ℝ := {x | x^2 - 4*x + 3 < 0}

-- State the theorem
theorem m_xor_n_equals_target : 
  setXor M N = {x | -2 < x ∧ x ≤ 1 ∨ 2 ≤ x ∧ x < 3} := by sorry

end NUMINAMATH_CALUDE_m_xor_n_equals_target_l1224_122471


namespace NUMINAMATH_CALUDE_stable_performance_lower_variance_l1224_122482

/-- Represents an athlete's shooting performance -/
structure Athlete where
  average_score : ℝ
  variance : ℝ
  sessions : ℕ

/-- Defines stability of performance based on variance -/
def more_stable (a b : Athlete) : Prop :=
  a.variance < b.variance

theorem stable_performance_lower_variance 
  (a b : Athlete) 
  (h1 : a.average_score = b.average_score) 
  (h2 : a.sessions = b.sessions) 
  (h3 : a.sessions > 0) 
  (h4 : a.variance < b.variance) : 
  more_stable a b :=
sorry

end NUMINAMATH_CALUDE_stable_performance_lower_variance_l1224_122482


namespace NUMINAMATH_CALUDE_cardinals_second_inning_l1224_122492

def cubs_third_inning : ℕ := 2
def cubs_fifth_inning : ℕ := 1
def cubs_eighth_inning : ℕ := 2
def cardinals_fifth_inning : ℕ := 1
def cubs_advantage : ℕ := 3

def cubs_total : ℕ := cubs_third_inning + cubs_fifth_inning + cubs_eighth_inning
def cardinals_total : ℕ := cubs_total - cubs_advantage

theorem cardinals_second_inning : cardinals_total - cardinals_fifth_inning = 1 := by
  sorry

end NUMINAMATH_CALUDE_cardinals_second_inning_l1224_122492


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l1224_122434

theorem quadratic_inequality_solution_set 
  (a b c : ℝ) 
  (h : Set.Ioo (-1 : ℝ) 2 = {x : ℝ | a * x^2 + b * x + c > 0}) :
  {x : ℝ | c * x^2 + b * x + a > 0} = Set.Ioi (1/2 : ℝ) ∪ Set.Iic (-1 : ℝ) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l1224_122434


namespace NUMINAMATH_CALUDE_card_problem_l1224_122473

/-- Given the number of cards for Brenda, Janet, and Mara, calculate the certain number -/
def certainNumber (brenda : ℕ) (janet : ℕ) (mara : ℕ) : ℕ :=
  mara + 40

theorem card_problem (brenda : ℕ) :
  let janet := brenda + 9
  let mara := 2 * janet
  brenda + janet + mara = 211 →
  certainNumber brenda janet mara = 150 := by
  sorry

#check card_problem

end NUMINAMATH_CALUDE_card_problem_l1224_122473


namespace NUMINAMATH_CALUDE_rectangle_area_l1224_122418

theorem rectangle_area (L W : ℝ) (h1 : L + W = 7) (h2 : L^2 + W^2 = 25) : L * W = 12 := by
  sorry

#check rectangle_area

end NUMINAMATH_CALUDE_rectangle_area_l1224_122418


namespace NUMINAMATH_CALUDE_luke_money_calculation_l1224_122469

theorem luke_money_calculation (initial_amount spent_amount received_amount : ℕ) : 
  initial_amount = 48 → spent_amount = 11 → received_amount = 21 →
  initial_amount - spent_amount + received_amount = 58 := by
sorry

end NUMINAMATH_CALUDE_luke_money_calculation_l1224_122469


namespace NUMINAMATH_CALUDE_production_line_b_units_l1224_122421

theorem production_line_b_units (total : ℕ) (a b c : ℕ) : 
  total = 16800 →
  total = a + b + c →
  b - a = c - b →
  b = 5600 := by
  sorry

end NUMINAMATH_CALUDE_production_line_b_units_l1224_122421


namespace NUMINAMATH_CALUDE_biggest_number_in_ratio_l1224_122451

theorem biggest_number_in_ratio (a b c d : ℕ) : 
  a + b + c + d = 1344 →
  2 * b = 3 * a →
  4 * a = 2 * c →
  5 * a = 2 * d →
  d ≤ 480 ∧ (∃ (x : ℕ), d = 480) :=
by sorry

end NUMINAMATH_CALUDE_biggest_number_in_ratio_l1224_122451


namespace NUMINAMATH_CALUDE_min_value_expression_l1224_122410

theorem min_value_expression (a b : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : a + b = 2) :
  2 / (a + 3 * b) + 1 / (a - b) ≥ (3 + 2 * Real.sqrt 2) / 4 := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l1224_122410


namespace NUMINAMATH_CALUDE_f_inequality_l1224_122424

noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 0 then x + 1 else 2^x

theorem f_inequality (x : ℝ) : f x + f (x - 1/2) > 1 ↔ x > -1/4 := by
  sorry

end NUMINAMATH_CALUDE_f_inequality_l1224_122424


namespace NUMINAMATH_CALUDE_pure_imaginary_condition_l1224_122463

theorem pure_imaginary_condition (a : ℝ) : 
  (Complex.I + 1) * (Complex.I * a + 2) = Complex.I * (Complex.I * b + c) → a = 2 := by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_condition_l1224_122463


namespace NUMINAMATH_CALUDE_bird_percentage_problem_l1224_122464

theorem bird_percentage_problem :
  ∀ (total : ℝ) (sparrows pigeons crows parrots : ℝ),
    sparrows = 0.4 * total →
    pigeons = 0.2 * total →
    crows = 0.15 * total →
    parrots = total - (sparrows + pigeons + crows) →
    (crows / (total - pigeons)) * 100 = 18.75 := by
  sorry

end NUMINAMATH_CALUDE_bird_percentage_problem_l1224_122464


namespace NUMINAMATH_CALUDE_probability_no_player_wins_all_is_11_16_l1224_122439

def num_players : Nat := 5

def num_games : Nat := (num_players * (num_players - 1)) / 2

def probability_no_player_wins_all : Rat :=
  1 - (num_players * (1 / 2 ^ (num_players - 1))) / (2 ^ num_games)

theorem probability_no_player_wins_all_is_11_16 :
  probability_no_player_wins_all = 11 / 16 := by
  sorry

end NUMINAMATH_CALUDE_probability_no_player_wins_all_is_11_16_l1224_122439


namespace NUMINAMATH_CALUDE_new_line_equation_l1224_122489

/-- Given a line y = mx + b, proves that a new line with half the slope
    and triple the y-intercept has the equation y = (m/2)x + 3b -/
theorem new_line_equation (m b : ℝ) :
  let original_line := fun x => m * x + b
  let new_line := fun x => (m / 2) * x + 3 * b
  ∀ x, new_line x = (m / 2) * x + 3 * b :=
by sorry

end NUMINAMATH_CALUDE_new_line_equation_l1224_122489


namespace NUMINAMATH_CALUDE_largest_angle_in_special_triangle_l1224_122404

theorem largest_angle_in_special_triangle : ∀ (a b c : ℝ),
  -- Two angles sum to 4/3 of a right angle
  a + b = 4/3 * 90 →
  -- One angle is 40° larger than the other
  b = a + 40 →
  -- All angles are non-negative
  0 ≤ a ∧ 0 ≤ b ∧ 0 ≤ c →
  -- Sum of angles in a triangle is 180°
  a + b + c = 180 →
  -- The largest angle is 80°
  max a (max b c) = 80 := by
  sorry

end NUMINAMATH_CALUDE_largest_angle_in_special_triangle_l1224_122404


namespace NUMINAMATH_CALUDE_ten_digit_divisible_by_11_exists_l1224_122491

def is_valid_number (n : ℕ) : Prop :=
  (n ≥ 1000000000 ∧ n < 10000000000) ∧
  (∀ d : Fin 10, ∃! p : Fin 10, (n / (10 ^ p.val) % 10) = d) ∧
  n % 11 = 0

theorem ten_digit_divisible_by_11_exists : ∃ n : ℕ, is_valid_number n :=
sorry

end NUMINAMATH_CALUDE_ten_digit_divisible_by_11_exists_l1224_122491


namespace NUMINAMATH_CALUDE_equal_goldfish_theorem_l1224_122426

/-- Number of months for Brent and Gretel to have the same number of goldfish -/
def equal_goldfish_months : ℕ := 8

/-- Brent's initial number of goldfish -/
def brent_initial : ℕ := 3

/-- Gretel's initial number of goldfish -/
def gretel_initial : ℕ := 243

/-- Brent's goldfish growth rate per month -/
def brent_growth_rate : ℝ := 3

/-- Gretel's goldfish growth rate per month -/
def gretel_growth_rate : ℝ := 1.5

/-- Brent's number of goldfish after n months -/
def brent_goldfish (n : ℕ) : ℝ := brent_initial * brent_growth_rate ^ n

/-- Gretel's number of goldfish after n months -/
def gretel_goldfish (n : ℕ) : ℝ := gretel_initial * gretel_growth_rate ^ n

/-- Theorem stating that Brent and Gretel have the same number of goldfish after equal_goldfish_months -/
theorem equal_goldfish_theorem : 
  brent_goldfish equal_goldfish_months = gretel_goldfish equal_goldfish_months :=
sorry

end NUMINAMATH_CALUDE_equal_goldfish_theorem_l1224_122426


namespace NUMINAMATH_CALUDE_sum_of_first_50_even_integers_l1224_122443

theorem sum_of_first_50_even_integers (sum_odd : ℕ) : 
  sum_odd = 50^2 → 
  (Finset.sum (Finset.range 50) (λ i => 2*i + 2) = sum_odd + 50) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_first_50_even_integers_l1224_122443


namespace NUMINAMATH_CALUDE_pizza_solution_l1224_122490

/-- Represents the number of pizza slices with different topping combinations -/
structure PizzaToppings where
  total : ℕ
  ham : ℕ
  pineapple : ℕ
  jalapeno : ℕ
  all_three : ℕ
  ham_only : ℕ
  pineapple_only : ℕ
  jalapeno_only : ℕ
  ham_pineapple : ℕ
  ham_jalapeno : ℕ
  pineapple_jalapeno : ℕ

/-- The pizza topping problem -/
def pizza_problem (p : PizzaToppings) : Prop :=
  p.total = 24 ∧
  p.ham = 15 ∧
  p.pineapple = 10 ∧
  p.jalapeno = 14 ∧
  p.all_three = p.jalapeno_only ∧
  p.total = p.ham_only + p.pineapple_only + p.jalapeno_only + 
            p.ham_pineapple + p.ham_jalapeno + p.pineapple_jalapeno + p.all_three ∧
  p.ham = p.ham_only + p.ham_pineapple + p.ham_jalapeno + p.all_three ∧
  p.pineapple = p.pineapple_only + p.ham_pineapple + p.pineapple_jalapeno + p.all_three ∧
  p.jalapeno = p.jalapeno_only + p.ham_jalapeno + p.pineapple_jalapeno + p.all_three

theorem pizza_solution (p : PizzaToppings) (h : pizza_problem p) : p.all_three = 5 := by
  sorry

end NUMINAMATH_CALUDE_pizza_solution_l1224_122490


namespace NUMINAMATH_CALUDE_correct_oranges_to_put_back_l1224_122405

/-- The number of oranges Mary must put back to achieve the desired average price -/
def oranges_to_put_back (apple_price orange_price : ℚ) (total_fruit : ℕ) (initial_avg_price desired_avg_price : ℚ) : ℕ :=
  sorry

theorem correct_oranges_to_put_back :
  oranges_to_put_back (40/100) (60/100) 10 (54/100) (50/100) = 4 := by
  sorry

end NUMINAMATH_CALUDE_correct_oranges_to_put_back_l1224_122405


namespace NUMINAMATH_CALUDE_circle_tangent_to_line_l1224_122472

/-- A circle with center (1,1) that is tangent to the line x + y = 4 -/
def TangentCircle : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - 1)^2 + (p.2 - 1)^2 = 2}

/-- The line x + y = 4 -/
def TangentLine : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 + p.2 = 4}

theorem circle_tangent_to_line :
  (∃ (p : ℝ × ℝ), p ∈ TangentCircle ∧ p ∈ TangentLine) ∧
  (∀ (p : ℝ × ℝ), p ∈ TangentCircle → p ∈ TangentLine → 
    ∀ (q : ℝ × ℝ), q ∈ TangentCircle → q = p ∨ q ∉ TangentLine) :=
sorry

end NUMINAMATH_CALUDE_circle_tangent_to_line_l1224_122472


namespace NUMINAMATH_CALUDE_range_of_H_l1224_122436

-- Define the function H
def H (x : ℝ) : ℝ := |x + 2| - |x - 2|

-- State the theorem about the range of H
theorem range_of_H :
  ∀ y : ℝ, (∃ x : ℝ, H x = y) ↔ -4 ≤ y ∧ y ≤ 4 :=
sorry

end NUMINAMATH_CALUDE_range_of_H_l1224_122436


namespace NUMINAMATH_CALUDE_delta_implies_sigma_l1224_122477

-- Define the type for pairs of real numbers
def Pair := ℝ × ℝ

-- Define equality for pairs
def pair_eq (p q : Pair) : Prop := p.1 = q.1 ∧ p.2 = q.2

-- Define the Ä operation
def op_delta (p q : Pair) : Pair :=
  (p.1 * q.1 + p.2 * q.2, p.2 * q.1 - p.1 * q.2)

-- Define the Å operation
def op_sigma (p q : Pair) : Pair :=
  (p.1 + q.1, p.2 + q.2)

-- State the theorem
theorem delta_implies_sigma :
  ∀ x y : ℝ, pair_eq (op_delta (3, 4) (x, y)) (11, -2) →
  pair_eq (op_sigma (3, 4) (x, y)) (4, 6) := by
  sorry

end NUMINAMATH_CALUDE_delta_implies_sigma_l1224_122477


namespace NUMINAMATH_CALUDE_meaningful_expression_range_l1224_122428

theorem meaningful_expression_range (x : ℝ) :
  (∃ y : ℝ, y = (Real.sqrt (x + 2)) / (x - 1)) ↔ x ≥ -2 ∧ x ≠ 1 := by
  sorry

end NUMINAMATH_CALUDE_meaningful_expression_range_l1224_122428


namespace NUMINAMATH_CALUDE_second_year_sample_size_l1224_122429

/-- Represents the number of students to be sampled from each year group -/
structure SampleSize where
  first_year : ℕ
  second_year : ℕ
  third_year : ℕ
  fourth_year : ℕ

/-- Calculates the sample size for stratified sampling -/
def stratified_sample (total_students : ℕ) (sample_size : ℕ) (ratio : List ℕ) : SampleSize :=
  sorry

/-- Theorem stating the correct number of second-year students to be sampled -/
theorem second_year_sample_size :
  let total_students : ℕ := 5000
  let sample_size : ℕ := 260
  let ratio : List ℕ := [5, 4, 3, 1]
  let result := stratified_sample total_students sample_size ratio
  result.second_year = 80 := by sorry

end NUMINAMATH_CALUDE_second_year_sample_size_l1224_122429


namespace NUMINAMATH_CALUDE_cone_base_circumference_l1224_122498

/-- The circumference of the base of a right circular cone formed by removing a 180° sector from a circle with radius 6 inches is equal to 6π inches. -/
theorem cone_base_circumference (r : ℝ) (θ : ℝ) : 
  r = 6 → θ = 180 → 2 * π * r * (θ / 360) = 6 * π := by sorry

end NUMINAMATH_CALUDE_cone_base_circumference_l1224_122498


namespace NUMINAMATH_CALUDE_exists_n_no_rational_roots_l1224_122442

/-- A quadratic trinomial with real coefficients -/
structure QuadraticTrinomial where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The value of a quadratic trinomial at a given x -/
def QuadraticTrinomial.eval (p : QuadraticTrinomial) (x : ℝ) : ℝ :=
  p.a * x^2 + p.b * x + p.c

/-- Theorem: For any quadratic trinomial with real coefficients, 
    there exists a positive integer n such that p(x) = 1/n has no rational roots -/
theorem exists_n_no_rational_roots (p : QuadraticTrinomial) : 
  ∃ n : ℕ+, ¬∃ q : ℚ, p.eval q = (1 : ℝ) / n := by
  sorry

end NUMINAMATH_CALUDE_exists_n_no_rational_roots_l1224_122442


namespace NUMINAMATH_CALUDE_derivative_sum_at_points_l1224_122446

-- Define the function f
def f (m : ℝ) (x : ℝ) : ℝ := x^3 + m*x^2 + 2*x + 5

-- Define the derivative of f
def f' (m : ℝ) (x : ℝ) : ℝ := 3*x^2 + 2*m*x + 2

-- Theorem statement
theorem derivative_sum_at_points (m : ℝ) : f' m 2 + f' m (-2) = 28 := by
  sorry

end NUMINAMATH_CALUDE_derivative_sum_at_points_l1224_122446


namespace NUMINAMATH_CALUDE_miles_driven_l1224_122475

/-- Calculates the number of miles driven given car rental costs and total expenses --/
theorem miles_driven (rental_cost gas_needed gas_price per_mile_charge total_cost : ℚ) : 
  rental_cost = 150 →
  gas_needed = 8 →
  gas_price = 3.5 →
  per_mile_charge = 0.5 →
  total_cost = 338 →
  (total_cost - (rental_cost + gas_needed * gas_price)) / per_mile_charge = 320 := by
  sorry


end NUMINAMATH_CALUDE_miles_driven_l1224_122475


namespace NUMINAMATH_CALUDE_absolute_value_not_always_greater_than_negative_l1224_122408

theorem absolute_value_not_always_greater_than_negative : ∃ a : ℝ, |a| ≤ -a := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_not_always_greater_than_negative_l1224_122408


namespace NUMINAMATH_CALUDE_recurring_larger_than_finite_l1224_122459

def recurring_decimal : ℚ := 1 + 3/10 + 5/100 + 42/10000 + 5/1000 * (1/9)
def finite_decimal : ℚ := 1 + 3/10 + 5/100 + 4/1000 + 2/10000

theorem recurring_larger_than_finite : recurring_decimal > finite_decimal := by
  sorry

end NUMINAMATH_CALUDE_recurring_larger_than_finite_l1224_122459


namespace NUMINAMATH_CALUDE_bike_ride_time_l1224_122403

theorem bike_ride_time (distance_to_julia : ℝ) (time_to_julia : ℝ) (distance_to_bernard : ℝ) :
  distance_to_julia = 2 →
  time_to_julia = 8 →
  distance_to_bernard = 5 →
  (distance_to_bernard / distance_to_julia) * time_to_julia = 20 :=
by sorry

end NUMINAMATH_CALUDE_bike_ride_time_l1224_122403


namespace NUMINAMATH_CALUDE_problem_solution_l1224_122488

theorem problem_solution (x n : ℝ) (h1 : x = 40) (h2 : ((x / 4) * 5) + n - 12 = 48) : n = 10 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1224_122488


namespace NUMINAMATH_CALUDE_michaels_pie_order_cost_l1224_122406

/-- Calculate the total cost of fruit for Michael's pie order --/
theorem michaels_pie_order_cost :
  let peach_pies := 8
  let apple_pies := 6
  let blueberry_pies := 5
  let mixed_fruit_pies := 3
  let peach_per_pie := 4
  let apple_per_pie := 3
  let blueberry_per_pie := 3.5
  let mixed_fruit_per_pie := 3
  let apple_price := 1.25
  let blueberry_price := 0.90
  let peach_price := 2.50
  let mixed_fruit_per_type := mixed_fruit_per_pie / 3

  let total_peaches := peach_pies * peach_per_pie + mixed_fruit_pies * mixed_fruit_per_type
  let total_apples := apple_pies * apple_per_pie + mixed_fruit_pies * mixed_fruit_per_type
  let total_blueberries := blueberry_pies * blueberry_per_pie + mixed_fruit_pies * mixed_fruit_per_type

  let peach_cost := total_peaches * peach_price
  let apple_cost := total_apples * apple_price
  let blueberry_cost := total_blueberries * blueberry_price

  let total_cost := peach_cost + apple_cost + blueberry_cost

  total_cost = 132.20 := by
    sorry

end NUMINAMATH_CALUDE_michaels_pie_order_cost_l1224_122406


namespace NUMINAMATH_CALUDE_middle_number_values_l1224_122432

/-- Represents a three-layer product pyramid --/
structure ProductPyramid where
  bottom_left : ℕ+
  bottom_middle : ℕ+
  bottom_right : ℕ+

/-- Calculates the top number of the pyramid --/
def top_number (p : ProductPyramid) : ℕ :=
  (p.bottom_left * p.bottom_middle) * (p.bottom_middle * p.bottom_right)

/-- Theorem stating the possible values for the middle number --/
theorem middle_number_values (p : ProductPyramid) :
  top_number p = 90 → p.bottom_middle = 1 ∨ p.bottom_middle = 3 := by
  sorry

#check middle_number_values

end NUMINAMATH_CALUDE_middle_number_values_l1224_122432


namespace NUMINAMATH_CALUDE_binomial_probability_problem_l1224_122474

/-- A random variable following a binomial distribution -/
structure BinomialRV where
  n : ℕ
  p : ℝ
  hp : 0 ≤ p ∧ p ≤ 1

/-- Probability of a binomial random variable being greater than or equal to k -/
noncomputable def prob_ge (X : BinomialRV) (k : ℕ) : ℝ := sorry

theorem binomial_probability_problem (ξ η : BinomialRV)
  (hξ : ξ.n = 2)
  (hη : η.n = 4)
  (hp : ξ.p = η.p)
  (hprob : prob_ge ξ 1 = 5/9) :
  prob_ge η 2 = 11/27 := by sorry

end NUMINAMATH_CALUDE_binomial_probability_problem_l1224_122474


namespace NUMINAMATH_CALUDE_percentage_of_female_dogs_l1224_122445

theorem percentage_of_female_dogs (total_dogs : ℕ) (birth_ratio : ℚ) (puppies_per_birth : ℕ) (total_puppies : ℕ) :
  total_dogs = 40 →
  birth_ratio = 3 / 4 →
  puppies_per_birth = 10 →
  total_puppies = 180 →
  (↑total_puppies : ℚ) = (birth_ratio * puppies_per_birth * (60 / 100 * total_dogs)) →
  60 = (100 * (total_puppies / (birth_ratio * puppies_per_birth * total_dogs))) := by
  sorry

end NUMINAMATH_CALUDE_percentage_of_female_dogs_l1224_122445


namespace NUMINAMATH_CALUDE_circle_center_and_radius_l1224_122431

/-- A circle in the 2D plane. -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- The equation of a circle. -/
def Circle.equation (c : Circle) (x y : ℝ) : Prop :=
  (x - c.center.1)^2 + (y - c.center.2)^2 = c.radius^2

theorem circle_center_and_radius :
  ∃ (c : Circle), (∀ x y : ℝ, c.equation x y ↔ (x - 1)^2 + y^2 = 1) ∧
                  c.center = (1, 0) ∧
                  c.radius = 1 := by
  sorry


end NUMINAMATH_CALUDE_circle_center_and_radius_l1224_122431


namespace NUMINAMATH_CALUDE_chip_drawing_probability_l1224_122423

/-- The number of tan chips in the bag -/
def num_tan : ℕ := 4

/-- The number of pink chips in the bag -/
def num_pink : ℕ := 3

/-- The number of violet chips in the bag -/
def num_violet : ℕ := 5

/-- The number of green chips in the bag -/
def num_green : ℕ := 2

/-- The total number of chips in the bag -/
def total_chips : ℕ := num_tan + num_pink + num_violet + num_green

/-- The probability of drawing the chips in the specified arrangement -/
def probability : ℚ := (num_tan.factorial * num_pink.factorial * num_violet.factorial * 6) / total_chips.factorial

theorem chip_drawing_probability : probability = 1440 / total_chips.factorial :=
sorry

end NUMINAMATH_CALUDE_chip_drawing_probability_l1224_122423


namespace NUMINAMATH_CALUDE_sum_squares_equality_l1224_122430

theorem sum_squares_equality (N : ℕ) : 
  (1^2 + 2^2 + 3^2 + 4^2) / 4 = (2000^2 + 2001^2 + 2002^2 + 2003^2) / N → N = 2134 := by
  sorry

end NUMINAMATH_CALUDE_sum_squares_equality_l1224_122430


namespace NUMINAMATH_CALUDE_circle_area_from_diameter_endpoints_l1224_122481

/-- Given two points C and D as endpoints of a diameter of a circle,
    calculate the area of the circle. -/
theorem circle_area_from_diameter_endpoints
  (C D : ℝ × ℝ) -- C and D are points in the real plane
  (h : C = (-2, 3) ∧ D = (4, -1)) -- C and D have specific coordinates
  : (π * ((C.1 - D.1)^2 + (C.2 - D.2)^2) / 4) = 13 * π := by
  sorry


end NUMINAMATH_CALUDE_circle_area_from_diameter_endpoints_l1224_122481


namespace NUMINAMATH_CALUDE_range_of_m_l1224_122454

/-- The set A -/
def A : Set ℝ := {x | |x - 2| ≤ 4}

/-- The set B parameterized by m -/
def B (m : ℝ) : Set ℝ := {x | (x - 1 - m) * (x - 1 + m) ≤ 0}

/-- The proposition that ¬p is a necessary but not sufficient condition for ¬q -/
def not_p_necessary_not_sufficient_for_not_q (m : ℝ) : Prop :=
  (∀ x, x ∉ B m → x ∉ A) ∧ ∃ x, x ∉ B m ∧ x ∈ A

/-- The theorem stating the range of m -/
theorem range_of_m :
  ∀ m : ℝ, m > 0 ∧ not_p_necessary_not_sufficient_for_not_q m ↔ m ≥ 5 :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l1224_122454


namespace NUMINAMATH_CALUDE_water_tower_problem_l1224_122460

theorem water_tower_problem (total_capacity : ℕ) (n1 n2 n3 n4 n5 : ℕ) :
  total_capacity = 2700 →
  n1 = 300 →
  n2 = 2 * n1 →
  n3 = n2 + 100 →
  n4 = 3 * n1 →
  n5 = n3 / 2 →
  n1 + n2 + n3 + n4 + n5 > total_capacity :=
by sorry

end NUMINAMATH_CALUDE_water_tower_problem_l1224_122460


namespace NUMINAMATH_CALUDE_total_cupcakes_is_52_l1224_122496

/-- Represents the number of cupcakes ordered by the mum -/
def total_cupcakes : ℕ := 52

/-- Represents the number of vegan cupcakes -/
def vegan_cupcakes : ℕ := 24

/-- Represents the number of non-vegan cupcakes containing gluten -/
def non_vegan_gluten_cupcakes : ℕ := 28

/-- States that half of all cupcakes are gluten-free -/
axiom half_gluten_free : total_cupcakes / 2 = total_cupcakes - (vegan_cupcakes / 2 + non_vegan_gluten_cupcakes)

/-- States that half of vegan cupcakes are gluten-free -/
axiom half_vegan_gluten_free : vegan_cupcakes / 2 = total_cupcakes / 2 - non_vegan_gluten_cupcakes

/-- Theorem: The total number of cupcakes is 52 -/
theorem total_cupcakes_is_52 : total_cupcakes = 52 := by
  sorry

end NUMINAMATH_CALUDE_total_cupcakes_is_52_l1224_122496


namespace NUMINAMATH_CALUDE_exists_noncommuting_matrix_exp_l1224_122425

open Matrix

/-- Definition of matrix exponential -/
def matrix_exp (M : Matrix (Fin 2) (Fin 2) ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  1 + M + (1/2) • (M * M) + (1/6) • (M * M * M) + sorry

/-- Theorem: There exist 2x2 matrices A and B such that exp(A+B) ≠ exp(A)exp(B) -/
theorem exists_noncommuting_matrix_exp :
  ∃ (A B : Matrix (Fin 2) (Fin 2) ℝ), matrix_exp (A + B) ≠ matrix_exp A * matrix_exp B :=
sorry

end NUMINAMATH_CALUDE_exists_noncommuting_matrix_exp_l1224_122425


namespace NUMINAMATH_CALUDE_book_pages_theorem_l1224_122449

theorem book_pages_theorem (total_pages : ℚ) (read_pages : ℚ) 
  (h1 : read_pages = 3 / 7 * total_pages) : 
  ∃ (remaining_pages : ℚ),
    remaining_pages = 4 / 7 * total_pages ∧ 
    read_pages = 3 / 4 * remaining_pages := by
  sorry

end NUMINAMATH_CALUDE_book_pages_theorem_l1224_122449
