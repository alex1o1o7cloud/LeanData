import Mathlib

namespace NUMINAMATH_CALUDE_smallest_addition_for_divisibility_l1394_139447

theorem smallest_addition_for_divisibility : ∃! x : ℕ, 
  (x ≤ 2374) ∧ (1275890 + x) % 2375 = 0 ∧ 
  ∀ y : ℕ, y < x → (1275890 + y) % 2375 ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_smallest_addition_for_divisibility_l1394_139447


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l1394_139491

theorem right_triangle_hypotenuse (a b c : ℕ) : 
  a^2 + b^2 = c^2 →  -- Pythagorean theorem
  a = 1994 →         -- One cathetus is 1994
  c = 994010         -- Hypotenuse is 994010
  := by sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l1394_139491


namespace NUMINAMATH_CALUDE_inequality_solution_set_l1394_139495

theorem inequality_solution_set (x : ℝ) : -2 * x + 3 < 0 ↔ x > 3/2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l1394_139495


namespace NUMINAMATH_CALUDE_probability_less_than_20_l1394_139429

theorem probability_less_than_20 (total : ℕ) (over_30 : ℕ) (h1 : total = 150) (h2 : over_30 = 90) :
  (total - over_30 : ℚ) / total = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_probability_less_than_20_l1394_139429


namespace NUMINAMATH_CALUDE_inequality_and_minimum_value_l1394_139482

variables (a b x : ℝ)

def f (a b x : ℝ) : ℝ := |2*x - a^4 + (1 - 6*a^2*b^2 - b^4)| + 2*|x - (2*a^3*b + 2*a*b^3 - 1)|

theorem inequality_and_minimum_value :
  (a^4 + 6*a^2*b^2 + b^4 ≥ 4*a*b*(a^2 + b^2)) ∧
  (∀ x, f a b x ≥ 1) := by
  sorry

end NUMINAMATH_CALUDE_inequality_and_minimum_value_l1394_139482


namespace NUMINAMATH_CALUDE_sqrt_ab_equals_18_l1394_139427

theorem sqrt_ab_equals_18 (a b : ℝ) : 
  a = Real.log 9 / Real.log 4 → 
  b = 108 * (Real.log 8 / Real.log 3) → 
  Real.sqrt (a * b) = 18 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_ab_equals_18_l1394_139427


namespace NUMINAMATH_CALUDE_tangent_line_to_circle_l1394_139414

/-- A line is tangent to a circle if and only if the distance from the center of the circle
    to the line is equal to the radius of the circle. -/
axiom tangent_line_distance_eq_radius {a b c : ℝ} {x₀ y₀ r : ℝ} :
  (∀ x y, (x - x₀)^2 + (y - y₀)^2 = r^2 → a*x + b*y + c = 0 → 
    (x - x₀)^2 + (y - y₀)^2 = r^2 ∧ a*x + b*y + c = 0) ↔ 
  |a*x₀ + b*y₀ + c| / Real.sqrt (a^2 + b^2) = r

/-- The theorem to be proved -/
theorem tangent_line_to_circle (m : ℝ) (h_pos : m > 0) 
  (h_tangent : ∀ x y, (x - 3)^2 + (y - 4)^2 = 4 → 3*x - 4*y - m = 0 → 
    (x - 3)^2 + (y - 4)^2 = 4 ∧ 3*x - 4*y - m = 0) : 
  m = 3 := by
sorry

end NUMINAMATH_CALUDE_tangent_line_to_circle_l1394_139414


namespace NUMINAMATH_CALUDE_x_squared_mod_25_l1394_139477

theorem x_squared_mod_25 (x : ℤ) 
  (h1 : 5 * x ≡ 10 [ZMOD 25]) 
  (h2 : 4 * x ≡ 21 [ZMOD 25]) : 
  x^2 ≡ 21 [ZMOD 25] := by
  sorry

end NUMINAMATH_CALUDE_x_squared_mod_25_l1394_139477


namespace NUMINAMATH_CALUDE_special_sequence_sum_2017_l1394_139415

/-- A sequence with special properties -/
def SpecialSequence (a : ℕ → ℝ) (S : ℕ → ℝ) : Prop :=
  a 1 = 1 ∧ ∀ n : ℕ, n ≥ 1 → S (n + 1) - S n = 3^n / a n

/-- The sum of the first 2017 terms of the special sequence -/
theorem special_sequence_sum_2017 (a : ℕ → ℝ) (S : ℕ → ℝ) 
    (h : SpecialSequence a S) : S 2017 = 3^1009 - 2 := by
  sorry

end NUMINAMATH_CALUDE_special_sequence_sum_2017_l1394_139415


namespace NUMINAMATH_CALUDE_sqrt_seven_simplification_l1394_139422

theorem sqrt_seven_simplification : 3 * Real.sqrt 7 - Real.sqrt 7 = 2 * Real.sqrt 7 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_seven_simplification_l1394_139422


namespace NUMINAMATH_CALUDE_remaining_money_l1394_139459

/-- Calculates the remaining money after purchases and discount --/
theorem remaining_money (initial_amount purchases discount_rate : ℚ) : 
  initial_amount = 10 ∧ 
  purchases = 3 + 2 + 1.5 + 0.75 ∧ 
  discount_rate = 0.05 → 
  initial_amount - (purchases - purchases * discount_rate) = 311/100 := by
  sorry

end NUMINAMATH_CALUDE_remaining_money_l1394_139459


namespace NUMINAMATH_CALUDE_sum_of_digits_next_l1394_139475

/-- Sum of digits function -/
def S (n : ℕ) : ℕ := sorry

/-- Theorem: For a positive integer n where S(n) = 1274, S(n+1) = 1239 -/
theorem sum_of_digits_next (n : ℕ) (h : S n = 1274) : S (n + 1) = 1239 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_digits_next_l1394_139475


namespace NUMINAMATH_CALUDE_min_distance_sum_l1394_139478

/-- A line in 2D space passing through (1,4) and intersecting positive x and y axes -/
structure IntersectingLine where
  a : ℝ
  b : ℝ
  a_pos : a > 0
  b_pos : b > 0
  passes_through_point : 1 / a + 4 / b = 1

/-- The sum of distances from origin to intersection points is at least 9 -/
theorem min_distance_sum (l : IntersectingLine) :
  l.a + l.b ≥ 9 := by
  sorry

#check min_distance_sum

end NUMINAMATH_CALUDE_min_distance_sum_l1394_139478


namespace NUMINAMATH_CALUDE_blue_pill_cost_l1394_139428

/-- Represents the cost of pills for a 21-day regimen --/
structure PillCost where
  blue : ℝ
  yellow : ℝ
  total : ℝ
  h1 : blue = yellow + 3
  h2 : 21 * (blue + yellow) = total

/-- The theorem stating the cost of a blue pill given the conditions --/
theorem blue_pill_cost (pc : PillCost) (h : pc.total = 882) : pc.blue = 22.5 := by
  sorry

end NUMINAMATH_CALUDE_blue_pill_cost_l1394_139428


namespace NUMINAMATH_CALUDE_bug_total_distance_l1394_139430

def bug_journey (start end1 end2 end3 : ℝ) : ℝ :=
  |end1 - start| + |end2 - end1| + |end3 - end2|

theorem bug_total_distance :
  bug_journey 0 4 (-3) 7 = 21 := by
  sorry

end NUMINAMATH_CALUDE_bug_total_distance_l1394_139430


namespace NUMINAMATH_CALUDE_imaginary_part_sum_of_fractions_l1394_139458

theorem imaginary_part_sum_of_fractions :
  Complex.im (1 / (Complex.ofReal (-2) + Complex.I) + 1 / (Complex.ofReal 1 - 2 * Complex.I)) = 1 / 5 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_sum_of_fractions_l1394_139458


namespace NUMINAMATH_CALUDE_min_value_theorem_l1394_139485

-- Define the condition for a, b, c
def satisfies_condition (a b c : ℝ) : Prop :=
  ∀ x y : ℝ, x + 2*y - 3 ≤ a*x + b*y + c ∧ a*x + b*y + c ≤ x + 2*y + 3

-- State the theorem
theorem min_value_theorem :
  ∃ (a b c : ℝ), satisfies_condition a b c ∧
  (∀ (a' b' c' : ℝ), satisfies_condition a' b' c' → a + 2*b - 3*c ≤ a' + 2*b' - 3*c') ∧
  a + 2*b - 3*c = -2 :=
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l1394_139485


namespace NUMINAMATH_CALUDE_P_on_xoz_plane_l1394_139400

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- The xoz plane in 3D space -/
def xoz_plane : Set Point3D :=
  {p : Point3D | p.y = 0}

/-- The given point P -/
def P : Point3D := ⟨-2, 0, 3⟩

/-- Theorem: Point P lies on the xoz plane -/
theorem P_on_xoz_plane : P ∈ xoz_plane := by
  sorry


end NUMINAMATH_CALUDE_P_on_xoz_plane_l1394_139400


namespace NUMINAMATH_CALUDE_f_is_generalized_distance_l1394_139433

def generalizedDistance (f : ℝ → ℝ → ℝ) : Prop :=
  (∀ x y, f x y ≥ 0 ∧ (f x y = 0 ↔ x = 0 ∧ y = 0)) ∧
  (∀ x y, f x y = f y x) ∧
  (∀ x y z, f x y ≤ f x z + f z y)

def f (x y : ℝ) : ℝ := x^2 + y^2

theorem f_is_generalized_distance : generalizedDistance f := by sorry

end NUMINAMATH_CALUDE_f_is_generalized_distance_l1394_139433


namespace NUMINAMATH_CALUDE_abs_gt_one_necessary_not_sufficient_for_lt_neg_two_l1394_139498

theorem abs_gt_one_necessary_not_sufficient_for_lt_neg_two (x : ℝ) :
  (∀ x, x < -2 → |x| > 1) ∧ 
  (∃ x, |x| > 1 ∧ ¬(x < -2)) :=
by sorry

end NUMINAMATH_CALUDE_abs_gt_one_necessary_not_sufficient_for_lt_neg_two_l1394_139498


namespace NUMINAMATH_CALUDE_unoccupied_area_formula_l1394_139496

/-- The area of a rectangle not occupied by a hole and a square -/
def unoccupied_area (x : ℝ) : ℝ :=
  let large_rect := (2*x + 9) * (x + 6)
  let hole := (x - 1) * (2*x - 5)
  let square := (x + 3)^2
  large_rect - hole - square

/-- Theorem stating the unoccupied area in terms of x -/
theorem unoccupied_area_formula (x : ℝ) :
  unoccupied_area x = -x^2 + 22*x + 40 := by
  sorry

end NUMINAMATH_CALUDE_unoccupied_area_formula_l1394_139496


namespace NUMINAMATH_CALUDE_minimize_y_l1394_139460

variable (a b c x : ℝ)

def y (x : ℝ) := (x - a)^2 + (x - b)^2 + 2*c*x

theorem minimize_y :
  ∃ (x_min : ℝ), ∀ (x : ℝ), y x ≥ y x_min ∧ x_min = (a + b - c) / 2 :=
sorry

end NUMINAMATH_CALUDE_minimize_y_l1394_139460


namespace NUMINAMATH_CALUDE_expression_simplification_l1394_139439

theorem expression_simplification (a b : ℝ) (h : (a + 2)^2 + |b - 1| = 0) :
  (3 * a^2 * b - a * b^2) - (1/2) * (a^2 * b - (2 * a * b^2 - 4)) + 1 = 9 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l1394_139439


namespace NUMINAMATH_CALUDE_find_number_l1394_139401

theorem find_number : ∃ x : ℝ, 3 * (2 * x + 5) = 105 := by
  sorry

end NUMINAMATH_CALUDE_find_number_l1394_139401


namespace NUMINAMATH_CALUDE_frame_ratio_l1394_139416

theorem frame_ratio (x : ℝ) (h : x > 0) : 
  (20 + 2*x) * (30 + 6*x) - 20 * 30 = 20 * 30 →
  (20 + 2*x) / (30 + 6*x) = 1/2 := by
sorry

end NUMINAMATH_CALUDE_frame_ratio_l1394_139416


namespace NUMINAMATH_CALUDE_cylinder_minus_cones_volume_l1394_139490

/-- The volume of a cylinder minus two congruent cones -/
theorem cylinder_minus_cones_volume 
  (r : ℝ) 
  (h_cone : ℝ) 
  (h_cyl : ℝ) 
  (h_r : r = 10) 
  (h_cone_height : h_cone = 15) 
  (h_cyl_height : h_cyl = 30) : 
  π * r^2 * h_cyl - 2 * (1/3 * π * r^2 * h_cone) = 2000 * π := by
  sorry

end NUMINAMATH_CALUDE_cylinder_minus_cones_volume_l1394_139490


namespace NUMINAMATH_CALUDE_full_price_revenue_l1394_139462

/-- Represents the price of a full-price ticket -/
def full_price : ℝ := sorry

/-- Represents the number of full-price tickets sold -/
def full_price_tickets : ℕ := sorry

/-- Represents the number of discounted tickets sold -/
def discounted_tickets : ℕ := sorry

/-- The total number of tickets sold is 160 -/
axiom total_tickets : full_price_tickets + discounted_tickets = 160

/-- The total revenue is $2400 -/
axiom total_revenue : full_price * full_price_tickets + (full_price / 3) * discounted_tickets = 2400

/-- Theorem stating that the revenue from full-price tickets is $400 -/
theorem full_price_revenue : full_price * full_price_tickets = 400 := by sorry

end NUMINAMATH_CALUDE_full_price_revenue_l1394_139462


namespace NUMINAMATH_CALUDE_no_solution_equation_one_solutions_equation_two_l1394_139446

-- Problem 1
theorem no_solution_equation_one : 
  ¬ ∃ x : ℝ, (1 / (x - 2) + 2 = (1 - x) / (2 - x)) ∧ (x ≠ 2) :=
sorry

-- Problem 2
theorem solutions_equation_two :
  ∀ x : ℝ, (x - 4)^2 = 4*(2*x + 1)^2 ↔ x = 2/5 ∨ x = -2 :=
sorry

end NUMINAMATH_CALUDE_no_solution_equation_one_solutions_equation_two_l1394_139446


namespace NUMINAMATH_CALUDE_proportionality_problem_l1394_139409

/-- Given that x is directly proportional to y^4 and y is inversely proportional to z^2,
    prove that x = 1/16 when z = 32, given that x = 4 when z = 8. -/
theorem proportionality_problem (x y z : ℝ) (k₁ k₂ : ℝ) 
    (h₁ : x = k₁ * y^4)
    (h₂ : y * z^2 = k₂)
    (h₃ : x = 4 ∧ z = 8 → k₁ * k₂^4 = 67108864) :
    z = 32 → x = 1/16 := by
  sorry

end NUMINAMATH_CALUDE_proportionality_problem_l1394_139409


namespace NUMINAMATH_CALUDE_arc_length_45_degrees_l1394_139479

theorem arc_length_45_degrees (circle_circumference : Real) (central_angle : Real) (arc_length : Real) : 
  circle_circumference = 72 →
  central_angle = 45 →
  arc_length = circle_circumference * (central_angle / 360) →
  arc_length = 9 :=
by sorry

end NUMINAMATH_CALUDE_arc_length_45_degrees_l1394_139479


namespace NUMINAMATH_CALUDE_mila_trip_distance_l1394_139424

/-- Represents the details of Mila's trip -/
structure MilaTrip where
  /-- Miles per gallon of Mila's car -/
  mpg : ℝ
  /-- Capacity of Mila's gas tank in gallons -/
  tankCapacity : ℝ
  /-- Miles driven in the first leg of the trip -/
  firstLegMiles : ℝ
  /-- Gallons of gas refueled -/
  refueledGallons : ℝ
  /-- Fraction of tank full upon arrival -/
  finalTankFraction : ℝ

/-- Calculates the total distance of Mila's trip -/
def totalDistance (trip : MilaTrip) : ℝ :=
  trip.firstLegMiles + (trip.tankCapacity - trip.finalTankFraction * trip.tankCapacity) * trip.mpg

/-- Theorem stating that Mila's total trip distance is 826 miles -/
theorem mila_trip_distance :
  ∀ (trip : MilaTrip),
    trip.mpg = 40 ∧
    trip.tankCapacity = 16 ∧
    trip.firstLegMiles = 400 ∧
    trip.refueledGallons = 10 ∧
    trip.finalTankFraction = 1/3 →
    totalDistance trip = 826 := by
  sorry

end NUMINAMATH_CALUDE_mila_trip_distance_l1394_139424


namespace NUMINAMATH_CALUDE_area_of_triangle_formed_by_tangent_points_l1394_139484

/-- Represents a circle with a center point and a radius -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Checks if two circles are externally tangent -/
def are_externally_tangent (c1 c2 : Circle) : Prop :=
  let (x1, y1) := c1.center
  let (x2, y2) := c2.center
  (x1 - x2)^2 + (y1 - y2)^2 = (c1.radius + c2.radius)^2

/-- Checks if a circle is tangent to the x-axis -/
def is_tangent_to_x_axis (c : Circle) : Prop :=
  let (_, y) := c.center
  y = c.radius

/-- The main theorem -/
theorem area_of_triangle_formed_by_tangent_points : 
  ∀ (c1 c2 c3 : Circle),
  c1.radius = 1 ∧ c2.radius = 3 ∧ c3.radius = 5 →
  are_externally_tangent c1 c2 ∧ 
  are_externally_tangent c2 c3 ∧ 
  are_externally_tangent c1 c3 →
  is_tangent_to_x_axis c1 ∧ 
  is_tangent_to_x_axis c2 ∧ 
  is_tangent_to_x_axis c3 →
  let (x1, _) := c1.center
  let (x2, _) := c2.center
  let (x3, _) := c3.center
  (1/2) * (|x2 - x1| + |x3 - x2| + |x3 - x1|) * (c3.radius - c1.radius) = 6 :=
by sorry

end NUMINAMATH_CALUDE_area_of_triangle_formed_by_tangent_points_l1394_139484


namespace NUMINAMATH_CALUDE_min_value_xyz_plus_2sum_l1394_139464

theorem min_value_xyz_plus_2sum (x y z : ℝ) 
  (hx : |x| ≥ 2) (hy : |y| ≥ 2) (hz : |z| ≥ 2) : 
  |x * y * z + 2 * (x + y + z)| ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_min_value_xyz_plus_2sum_l1394_139464


namespace NUMINAMATH_CALUDE_original_number_problem_l1394_139470

theorem original_number_problem (x : ℝ) : 3 * (2 * x + 9) = 69 → x = 7 := by
  sorry

end NUMINAMATH_CALUDE_original_number_problem_l1394_139470


namespace NUMINAMATH_CALUDE_geometric_sequence_formula_l1394_139465

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_formula 
  (a : ℕ → ℝ) 
  (h_positive : ∀ n, a n > 0)
  (h_diff : |a 2 - a 3| = 14)
  (h_product : a 1 * a 2 * a 3 = 343)
  (h_geometric : geometric_sequence a) :
  ∃ q : ℝ, q = 3 ∧ ∀ n : ℕ, a n = 7 * q^(n - 2) :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_formula_l1394_139465


namespace NUMINAMATH_CALUDE_percentage_sum_l1394_139402

theorem percentage_sum (A B C : ℝ) : 
  (0.45 * A = 270) → 
  (0.35 * B = 210) → 
  (0.25 * C = 150) → 
  (0.75 * A + 0.65 * B + 0.45 * C = 1110) := by
sorry

end NUMINAMATH_CALUDE_percentage_sum_l1394_139402


namespace NUMINAMATH_CALUDE_x_less_than_2_necessary_not_sufficient_l1394_139455

theorem x_less_than_2_necessary_not_sufficient :
  (∃ x : ℝ, x^2 < 4 ∧ ¬(x < 2)) ∧
  (∀ x : ℝ, x^2 < 4 → x < 2) :=
by sorry

end NUMINAMATH_CALUDE_x_less_than_2_necessary_not_sufficient_l1394_139455


namespace NUMINAMATH_CALUDE_boy_escapes_l1394_139436

/-- Represents the square pool -/
structure Pool :=
  (side_length : ℝ)
  (boy_position : ℝ × ℝ)
  (teacher_position : ℝ × ℝ)

/-- Represents the speeds of the boy and teacher -/
structure Speeds :=
  (boy_swim : ℝ)
  (boy_run : ℝ)
  (teacher_run : ℝ)

/-- Checks if the boy can escape given the pool configuration and speeds -/
def can_escape (p : Pool) (s : Speeds) : Prop :=
  p.side_length = 2 ∧
  p.boy_position = (0, 0) ∧
  p.teacher_position = (1, 1) ∧
  s.boy_swim = s.teacher_run / 3 ∧
  s.boy_run > s.teacher_run

theorem boy_escapes (p : Pool) (s : Speeds) :
  can_escape p s → true :=
sorry

end NUMINAMATH_CALUDE_boy_escapes_l1394_139436


namespace NUMINAMATH_CALUDE_point_P_coordinates_and_PQ_length_l1394_139413

def point_P (n : ℝ) : ℝ × ℝ := (n + 3, 2 - 3*n)

def fourth_quadrant (p : ℝ × ℝ) : Prop :=
  p.1 > 0 ∧ p.2 < 0

def distance_to_x_axis (p : ℝ × ℝ) : ℝ :=
  |p.2|

def distance_to_y_axis (p : ℝ × ℝ) : ℝ :=
  |p.1|

def point_Q (n : ℝ) : ℝ × ℝ := (n, -4)

def parallel_to_x_axis (p q : ℝ × ℝ) : Prop :=
  p.2 = q.2

theorem point_P_coordinates_and_PQ_length :
  ∃ n : ℝ,
    let p := point_P n
    let q := point_Q n
    fourth_quadrant p ∧
    distance_to_x_axis p = distance_to_y_axis p + 1 ∧
    parallel_to_x_axis p q ∧
    p = (6, -7) ∧
    |p.1 - q.1| = 3 :=
by sorry

end NUMINAMATH_CALUDE_point_P_coordinates_and_PQ_length_l1394_139413


namespace NUMINAMATH_CALUDE_right_triangle_perimeter_equals_area_l1394_139463

theorem right_triangle_perimeter_equals_area :
  ∀ a b c : ℕ,
  a > 0 ∧ b > 0 ∧ c > 0 →
  a^2 + b^2 = c^2 →
  a + b + c = (a * b) / 2 →
  ((a = 5 ∧ b = 12 ∧ c = 13) ∨
   (a = 12 ∧ b = 5 ∧ c = 13) ∨
   (a = 6 ∧ b = 8 ∧ c = 10) ∨
   (a = 8 ∧ b = 6 ∧ c = 10)) :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_perimeter_equals_area_l1394_139463


namespace NUMINAMATH_CALUDE_shortest_altitude_of_triangle_l1394_139426

/-- Given a triangle with sides 9, 12, and 15, the shortest altitude has length 7.2 -/
theorem shortest_altitude_of_triangle (a b c h : ℝ) : 
  a = 9 → b = 12 → c = 15 → 
  a^2 + b^2 = c^2 →
  h * c = 2 * (a * b / 2) →
  h = 7.2 := by sorry

end NUMINAMATH_CALUDE_shortest_altitude_of_triangle_l1394_139426


namespace NUMINAMATH_CALUDE_children_still_hiding_l1394_139440

theorem children_still_hiding (total : ℕ) (found : ℕ) (seeker : ℕ) : 
  total = 16 → found = 6 → seeker = 1 → total - found - seeker = 9 := by
sorry

end NUMINAMATH_CALUDE_children_still_hiding_l1394_139440


namespace NUMINAMATH_CALUDE_curtain_length_for_given_room_l1394_139469

/-- Calculates the required curtain length in inches given the room height in feet and additional material in inches. -/
def curtain_length (room_height_feet : ℕ) (additional_inches : ℕ) : ℕ :=
  room_height_feet * 12 + additional_inches

/-- Theorem stating that for a room height of 8 feet and 5 inches of additional material, the required curtain length is 101 inches. -/
theorem curtain_length_for_given_room : curtain_length 8 5 = 101 := by
  sorry

end NUMINAMATH_CALUDE_curtain_length_for_given_room_l1394_139469


namespace NUMINAMATH_CALUDE_human_genome_project_satisfies_conditions_l1394_139404

/-- Represents a scientific plan --/
structure ScientificPlan where
  name : String
  launchYear : Nat
  participatingCountries : List String
  isMajorPlan : Bool

/-- The Human Genome Project --/
def humanGenomeProject : ScientificPlan := {
  name := "Human Genome Project",
  launchYear := 1990,
  participatingCountries := ["United States", "United Kingdom", "France", "Germany", "Japan", "China"],
  isMajorPlan := true
}

/-- The Manhattan Project --/
def manhattanProject : ScientificPlan := {
  name := "Manhattan Project",
  launchYear := 1942,
  participatingCountries := ["United States", "United Kingdom", "Canada"],
  isMajorPlan := true
}

/-- The Apollo Program --/
def apolloProgram : ScientificPlan := {
  name := "Apollo Program",
  launchYear := 1961,
  participatingCountries := ["United States"],
  isMajorPlan := true
}

/-- The set of "Three Major Scientific Plans" --/
def threeMajorPlans : List ScientificPlan := [humanGenomeProject, manhattanProject, apolloProgram]

/-- Theorem stating that the Human Genome Project satisfies all conditions --/
theorem human_genome_project_satisfies_conditions :
  humanGenomeProject.launchYear = 1990 ∧
  humanGenomeProject.participatingCountries = ["United States", "United Kingdom", "France", "Germany", "Japan", "China"] ∧
  humanGenomeProject ∈ threeMajorPlans := by
  sorry


end NUMINAMATH_CALUDE_human_genome_project_satisfies_conditions_l1394_139404


namespace NUMINAMATH_CALUDE_second_polygon_sides_l1394_139405

/-- Given two regular polygons with the same perimeter, where the first has 38 sides
    and a side length twice that of the second, prove the second has 76 sides. -/
theorem second_polygon_sides (s : ℝ) (n : ℕ) :
  s > 0 →
  38 * (2 * s) = n * s →
  n = 76 := by
  sorry

end NUMINAMATH_CALUDE_second_polygon_sides_l1394_139405


namespace NUMINAMATH_CALUDE_restaurant_menu_fraction_l1394_139431

theorem restaurant_menu_fraction (total_dishes : ℕ) 
  (h1 : 6 = (1 / 3 : ℚ) * total_dishes)
  (h2 : 4 ≤ 6) : 
  (2 : ℚ) / total_dishes = 1 / 9 := by sorry

end NUMINAMATH_CALUDE_restaurant_menu_fraction_l1394_139431


namespace NUMINAMATH_CALUDE_toenail_size_ratio_l1394_139425

/-- Represents the capacity of the jar in terms of regular toenails -/
def jar_capacity : ℕ := 100

/-- Represents the number of big toenails in the jar -/
def big_toenails : ℕ := 20

/-- Represents the number of regular toenails initially in the jar -/
def regular_toenails : ℕ := 40

/-- Represents the additional regular toenails that can fit in the jar -/
def additional_regular_toenails : ℕ := 20

/-- Represents the ratio of the size of a big toenail to a regular toenail -/
def big_to_regular_ratio : ℚ := 2

theorem toenail_size_ratio :
  (jar_capacity - regular_toenails - additional_regular_toenails) / big_toenails = big_to_regular_ratio :=
sorry

end NUMINAMATH_CALUDE_toenail_size_ratio_l1394_139425


namespace NUMINAMATH_CALUDE_gcd_of_powers_of_47_l1394_139432

theorem gcd_of_powers_of_47 :
  Nat.Prime 47 →
  Nat.gcd (47^5 + 1) (47^5 + 47^3 + 47 + 1) = 1 := by
sorry

end NUMINAMATH_CALUDE_gcd_of_powers_of_47_l1394_139432


namespace NUMINAMATH_CALUDE_function_increasing_l1394_139412

theorem function_increasing (f : ℝ → ℝ) 
  (h : ∀ x₁ x₂ : ℝ, x₁ ≠ x₂ → x₁ * f x₁ + x₂ * f x₂ > x₁ * f x₂ + x₂ * f x₁) : 
  StrictMono f := by
  sorry

end NUMINAMATH_CALUDE_function_increasing_l1394_139412


namespace NUMINAMATH_CALUDE_michaels_lap_time_l1394_139467

/-- Race on a circular track -/
structure RaceTrack where
  length : ℝ
  donovan_lap_time : ℝ
  michael_laps_to_pass : ℕ

/-- Given race conditions, prove Michael's lap time -/
theorem michaels_lap_time (race : RaceTrack)
  (h1 : race.length = 300)
  (h2 : race.donovan_lap_time = 45)
  (h3 : race.michael_laps_to_pass = 9) :
  ∃ t : ℝ, t = 50 ∧ t * race.michael_laps_to_pass = (race.michael_laps_to_pass + 1) * race.donovan_lap_time :=
by sorry

end NUMINAMATH_CALUDE_michaels_lap_time_l1394_139467


namespace NUMINAMATH_CALUDE_intersection_nonempty_iff_p_less_than_one_l1394_139421

theorem intersection_nonempty_iff_p_less_than_one (p : ℝ) :
  let M : Set ℝ := {x | x ≤ 1}
  let N : Set ℝ := {x | x > p}
  (M ∩ N).Nonempty ↔ p < 1 := by
  sorry

end NUMINAMATH_CALUDE_intersection_nonempty_iff_p_less_than_one_l1394_139421


namespace NUMINAMATH_CALUDE_integral_inequality_l1394_139449

theorem integral_inequality (m : ℕ+) : 
  0 ≤ ∫ x in (0:ℝ)..1, (x + 1 - Real.sqrt (x^2 + 2*x * Real.cos (2*Real.pi / (2*(m:ℝ) + 1)) + 1)) ∧
  ∫ x in (0:ℝ)..1, (x + 1 - Real.sqrt (x^2 + 2*x * Real.cos (2*Real.pi / (2*(m:ℝ) + 1)) + 1)) ≤ 1 :=
by sorry

end NUMINAMATH_CALUDE_integral_inequality_l1394_139449


namespace NUMINAMATH_CALUDE_words_with_consonant_count_l1394_139461

/-- The set of all letters available --/
def letters : Finset Char := {'A', 'B', 'C', 'D', 'E', 'F'}

/-- The set of consonants --/
def consonants : Finset Char := {'B', 'C', 'D', 'F'}

/-- The set of vowels --/
def vowels : Finset Char := {'A', 'E'}

/-- The length of words we're considering --/
def word_length : Nat := 5

/-- A function that returns the number of words with at least one consonant --/
def words_with_consonant : Nat :=
  letters.card ^ word_length - vowels.card ^ word_length

theorem words_with_consonant_count :
  words_with_consonant = 7744 := by sorry

end NUMINAMATH_CALUDE_words_with_consonant_count_l1394_139461


namespace NUMINAMATH_CALUDE_negative_reals_sup_and_max_l1394_139445

-- Define the set of negative real numbers
def NegativeReals : Set ℝ := {x | x < 0}

-- Theorem statement
theorem negative_reals_sup_and_max :
  (∃ s : ℝ, IsLUB NegativeReals s) ∧
  (¬∃ m : ℝ, m ∈ NegativeReals ∧ ∀ x ∈ NegativeReals, x ≤ m) :=
by sorry

end NUMINAMATH_CALUDE_negative_reals_sup_and_max_l1394_139445


namespace NUMINAMATH_CALUDE_specific_arrangement_double_coverage_l1394_139474

/-- Represents a rectangle on a grid -/
structure Rectangle where
  width : ℕ
  height : ℕ

/-- Represents the arrangement of rectangles on the grid -/
structure Arrangement where
  rectangles : List Rectangle
  -- Additional properties to describe the specific arrangement could be added here

/-- Counts the number of cells covered by exactly two rectangles in the given arrangement -/
def countDoublyCoveredCells (arr : Arrangement) : ℕ :=
  sorry -- Implementation details would go here

/-- The main theorem stating that for the specific arrangement of three 4x6 rectangles,
    the number of cells covered by exactly two rectangles is 14 -/
theorem specific_arrangement_double_coverage :
  ∃ (arr : Arrangement),
    (arr.rectangles.length = 3) ∧
    (∀ r ∈ arr.rectangles, r.width = 4 ∧ r.height = 6) ∧
    (countDoublyCoveredCells arr = 14) := by
  sorry


end NUMINAMATH_CALUDE_specific_arrangement_double_coverage_l1394_139474


namespace NUMINAMATH_CALUDE_min_chopsticks_for_different_colors_l1394_139411

/-- Represents the number of pairs of chopsticks for each color -/
def pairs_per_color : ℕ := 4

/-- Represents the total number of colors -/
def total_colors : ℕ := 3

/-- Represents the total number of chopsticks -/
def total_chopsticks : ℕ := pairs_per_color * total_colors * 2

/-- 
Theorem: Given 12 pairs of chopsticks in 3 different colors (4 pairs each), 
the minimum number of chopsticks that must be taken out to guarantee 
two pairs of different colors is 11.
-/
theorem min_chopsticks_for_different_colors : ℕ := by
  sorry

end NUMINAMATH_CALUDE_min_chopsticks_for_different_colors_l1394_139411


namespace NUMINAMATH_CALUDE_polynomial_roots_l1394_139417

theorem polynomial_roots : 
  let p : ℝ → ℝ := fun x ↦ x^4 - 3*x^3 + 3*x^2 - x - 6
  ∀ x : ℝ, p x = 0 ↔ x = -1 ∨ x = 1 ∨ x = 2 ∨ x = 3 :=
by sorry

end NUMINAMATH_CALUDE_polynomial_roots_l1394_139417


namespace NUMINAMATH_CALUDE_monomial_combination_l1394_139486

theorem monomial_combination (a b : ℝ) (x y : ℤ) : 
  (∃ (k : ℝ), ∃ (m n : ℤ), 3 * a^(7*x) * b^(y+7) = k * a^m * b^n ∧ 
                           -7 * a^(2-4*y) * b^(2*x) = k * a^m * b^n) → 
  x + y = -1 := by sorry

end NUMINAMATH_CALUDE_monomial_combination_l1394_139486


namespace NUMINAMATH_CALUDE_second_bag_weight_is_10_l1394_139493

/-- The weight of the second bag of dog food Elise bought -/
def second_bag_weight (initial_weight first_bag_weight final_weight : ℕ) : ℕ :=
  final_weight - (initial_weight + first_bag_weight)

theorem second_bag_weight_is_10 :
  second_bag_weight 15 15 40 = 10 := by
  sorry

end NUMINAMATH_CALUDE_second_bag_weight_is_10_l1394_139493


namespace NUMINAMATH_CALUDE_hyperbola_theorem_l1394_139471

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 - y^2 = 2

-- Define the foci
def F1 : ℝ × ℝ := (-2, 0)
def F2 : ℝ × ℝ := (2, 0)

-- Define the origin
def O : ℝ × ℝ := (0, 0)

-- Define vector addition
def vec_add (v1 v2 : ℝ × ℝ) : ℝ × ℝ := (v1.1 + v2.1, v1.2 + v2.2)

-- Define vector from a point to another
def vec_from_to (p1 p2 : ℝ × ℝ) : ℝ × ℝ := (p2.1 - p1.1, p2.2 - p1.2)

-- Define the condition for point M
def M_condition (M A B : ℝ × ℝ) : Prop :=
  vec_from_to F1 M = vec_add (vec_add (vec_from_to F1 A) (vec_from_to F1 B)) (vec_from_to F1 O)

-- Define the dot product
def dot_product (v1 v2 : ℝ × ℝ) : ℝ := v1.1 * v2.1 + v1.2 * v2.2

-- Theorem statement
theorem hyperbola_theorem (A B M : ℝ × ℝ) 
  (hA : hyperbola A.1 A.2) 
  (hB : hyperbola B.1 B.2) 
  (hM : M_condition M A B) :
  -- 1. The locus of M is (x-6)^2 - y^2 = 4
  ((M.1 - 6)^2 - M.2^2 = 4) ∧
  -- 2. There exists a fixed point C(1, 0) such that CA · CB is constant
  (∃ (C : ℝ × ℝ), C = (1, 0) ∧ 
    dot_product (vec_from_to C A) (vec_from_to C B) = -1) :=
sorry

end NUMINAMATH_CALUDE_hyperbola_theorem_l1394_139471


namespace NUMINAMATH_CALUDE_zoo_field_trip_buses_l1394_139451

theorem zoo_field_trip_buses (fifth_graders sixth_graders seventh_graders : ℕ)
  (teachers_per_grade parents_per_grade : ℕ) (seats_per_bus : ℕ)
  (h1 : fifth_graders = 109)
  (h2 : sixth_graders = 115)
  (h3 : seventh_graders = 118)
  (h4 : teachers_per_grade = 4)
  (h5 : parents_per_grade = 2)
  (h6 : seats_per_bus = 72) :
  (fifth_graders + sixth_graders + seventh_graders +
   3 * (teachers_per_grade + parents_per_grade) + seats_per_bus - 1) / seats_per_bus = 5 :=
by sorry

end NUMINAMATH_CALUDE_zoo_field_trip_buses_l1394_139451


namespace NUMINAMATH_CALUDE_repetend_of_four_seventeenths_l1394_139497

/-- The decimal representation of 4/17 has a 6-digit repetend of 235294 -/
theorem repetend_of_four_seventeenths : ∃ (a b : ℕ), 
  (4 : ℚ) / 17 = (a : ℚ) / 999999 + (b : ℚ) / (999999 * 1000000) ∧ 
  a = 235294 ∧ 
  b < 999999 := by sorry

end NUMINAMATH_CALUDE_repetend_of_four_seventeenths_l1394_139497


namespace NUMINAMATH_CALUDE_metaphase_mitosis_observable_l1394_139403

/-- Represents the types of cell division that can occur in testis --/
inductive CellDivisionType
| Mitosis
| Meiosis

/-- Represents the phases of mitosis --/
inductive MitosisPhase
| Prophase
| Metaphase
| Anaphase
| Telophase

/-- Represents a cell in a testis slice --/
structure TestisCell where
  divisionType : CellDivisionType
  phase : Option MitosisPhase

/-- Represents a locust testis slice --/
structure LocustTestisSlice where
  cells : List TestisCell

/-- Condition: Both meiosis and mitosis can occur in the testis --/
def testisCanUndergoMitosisAndMeiosis (slice : LocustTestisSlice) : Prop :=
  ∃ (c1 c2 : TestisCell), c1 ∈ slice.cells ∧ c2 ∈ slice.cells ∧
    c1.divisionType = CellDivisionType.Mitosis ∧
    c2.divisionType = CellDivisionType.Meiosis

/-- Theorem: In locust testis slices, cells in the metaphase of mitosis can be observed --/
theorem metaphase_mitosis_observable (slice : LocustTestisSlice) 
  (h : testisCanUndergoMitosisAndMeiosis slice) :
  ∃ (c : TestisCell), c ∈ slice.cells ∧ 
    c.divisionType = CellDivisionType.Mitosis ∧
    c.phase = some MitosisPhase.Metaphase :=
  sorry

end NUMINAMATH_CALUDE_metaphase_mitosis_observable_l1394_139403


namespace NUMINAMATH_CALUDE_sum_of_cubes_mod_6_l1394_139494

theorem sum_of_cubes_mod_6 (h : ∀ n : ℕ, n^3 % 6 = n % 6) :
  (Finset.sum (Finset.range 150) (fun i => (i + 1)^3)) % 6 = 5 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_cubes_mod_6_l1394_139494


namespace NUMINAMATH_CALUDE_x_one_value_l1394_139423

theorem x_one_value (x₁ x₂ x₃ : ℝ) 
  (h_order : 0 ≤ x₃ ∧ x₃ ≤ x₂ ∧ x₂ ≤ x₁ ∧ x₁ ≤ 1) 
  (h_sum : (1 - x₁)^2 + (x₁ - x₂)^2 + (x₂ - x₃)^2 + x₃^2 = 1/3) : 
  x₁ = 2/3 := by
  sorry

end NUMINAMATH_CALUDE_x_one_value_l1394_139423


namespace NUMINAMATH_CALUDE_thirtieth_triangular_number_l1394_139438

/-- Definition of triangular numbers -/
def triangular_number (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The 30th triangular number is 465 -/
theorem thirtieth_triangular_number : triangular_number 30 = 465 := by
  sorry

end NUMINAMATH_CALUDE_thirtieth_triangular_number_l1394_139438


namespace NUMINAMATH_CALUDE_unique_integer_solution_l1394_139489

theorem unique_integer_solution (x y : ℤ) : 
  ({2 * x, x + y} : Set ℤ) = {7, 4} → x = 2 ∧ y = 5 := by
  sorry

end NUMINAMATH_CALUDE_unique_integer_solution_l1394_139489


namespace NUMINAMATH_CALUDE_sum_of_pyramid_edges_l1394_139473

/-- Represents a pyramid structure -/
structure Pyramid where
  vertices : ℕ

/-- The number of edges in a pyramid -/
def Pyramid.edges (p : Pyramid) : ℕ := 2 * p.vertices - 2

/-- Theorem: For three pyramids with a total of 40 vertices, the sum of their edges is 74 -/
theorem sum_of_pyramid_edges (a b c : Pyramid) 
  (h : a.vertices + b.vertices + c.vertices = 40) : 
  a.edges + b.edges + c.edges = 74 := by
  sorry


end NUMINAMATH_CALUDE_sum_of_pyramid_edges_l1394_139473


namespace NUMINAMATH_CALUDE_percentage_parents_without_full_time_jobs_l1394_139437

theorem percentage_parents_without_full_time_jobs :
  ∀ (total_parents : ℕ) (mothers fathers : ℕ),
    mothers + fathers = total_parents →
    mothers = (2 / 5 : ℚ) * total_parents →
    (3 / 4 : ℚ) * mothers + (9 / 10 : ℚ) * fathers = (21 / 25 : ℚ) * total_parents :=
by
  sorry

end NUMINAMATH_CALUDE_percentage_parents_without_full_time_jobs_l1394_139437


namespace NUMINAMATH_CALUDE_train_distance_proof_l1394_139480

/-- The initial distance between two trains -/
def initial_distance : ℝ := 13

/-- The speed of Train A in miles per hour -/
def speed_A : ℝ := 37

/-- The speed of Train B in miles per hour -/
def speed_B : ℝ := 43

/-- The time it takes for Train B to overtake and be ahead of Train A, in hours -/
def overtake_time : ℝ := 5

/-- The distance Train B is ahead of Train A after overtaking, in miles -/
def ahead_distance : ℝ := 17

theorem train_distance_proof :
  initial_distance = (speed_B - speed_A) * overtake_time - ahead_distance :=
by sorry

end NUMINAMATH_CALUDE_train_distance_proof_l1394_139480


namespace NUMINAMATH_CALUDE_ali_seashells_left_l1394_139419

/-- The number of seashells Ali has left after giving some away and selling half --/
def seashells_left (initial : ℕ) (given_to_friends : ℕ) (given_to_brothers : ℕ) : ℕ :=
  let remaining_after_giving := initial - given_to_friends - given_to_brothers
  remaining_after_giving - remaining_after_giving / 2

/-- Theorem stating that Ali has 55 seashells left --/
theorem ali_seashells_left : seashells_left 180 40 30 = 55 := by
  sorry

end NUMINAMATH_CALUDE_ali_seashells_left_l1394_139419


namespace NUMINAMATH_CALUDE_product_remainder_l1394_139450

theorem product_remainder (x : ℕ) :
  (1274 * x * 1277 * 1285) % 12 = 6 → x % 12 = 1 := by
  sorry

end NUMINAMATH_CALUDE_product_remainder_l1394_139450


namespace NUMINAMATH_CALUDE_two_students_same_type_l1394_139456

-- Define the types of books
inductive BookType
  | History
  | Literature
  | Science

-- Define a type for a pair of books
def BookPair := BookType × BookType

-- Define the set of all possible book pairs
def allBookPairs : Finset BookPair :=
  sorry

-- Define the number of students
def numStudents : Nat := 7

-- Theorem statement
theorem two_students_same_type :
  ∃ (s₁ s₂ : Fin numStudents) (bp : BookPair),
    s₁ ≠ s₂ ∧ 
    (∀ (s : Fin numStudents), ∃ (bp : BookPair), bp ∈ allBookPairs) ∧
    (∃ (f : Fin numStudents → BookPair), f s₁ = bp ∧ f s₂ = bp) :=
  sorry

end NUMINAMATH_CALUDE_two_students_same_type_l1394_139456


namespace NUMINAMATH_CALUDE_jakes_snake_length_l1394_139441

/-- Given two snakes where one is 12 inches longer than the other,
    and their combined length is 70 inches, prove that the longer snake is 41 inches long. -/
theorem jakes_snake_length (penny_snake : ℕ) (jake_snake : ℕ)
  (h1 : jake_snake = penny_snake + 12)
  (h2 : penny_snake + jake_snake = 70) :
  jake_snake = 41 :=
by sorry

end NUMINAMATH_CALUDE_jakes_snake_length_l1394_139441


namespace NUMINAMATH_CALUDE_B_power_15_minus_3_power_14_l1394_139468

def B : Matrix (Fin 2) (Fin 2) ℝ := !![3, 4; 0, 2]

theorem B_power_15_minus_3_power_14 :
  B^15 - 3 • B^14 = !![0, 4; 0, -1] := by sorry

end NUMINAMATH_CALUDE_B_power_15_minus_3_power_14_l1394_139468


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_l1394_139410

theorem min_value_reciprocal_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 3 * a + 2 * b = 2) :
  (1 / a + 1 / b) ≥ (5 + 2 * Real.sqrt 6) / 2 ∧
  ∃ (a₀ b₀ : ℝ), a₀ > 0 ∧ b₀ > 0 ∧ 3 * a₀ + 2 * b₀ = 2 ∧ 1 / a₀ + 1 / b₀ = (5 + 2 * Real.sqrt 6) / 2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_l1394_139410


namespace NUMINAMATH_CALUDE_right_triangle_third_side_l1394_139406

theorem right_triangle_third_side (a b c : ℝ) : 
  a = 3 ∧ b = 5 ∧ (c^2 = a^2 + b^2 ∨ b^2 = a^2 + c^2) → c = Real.sqrt 34 ∨ c = 4 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_third_side_l1394_139406


namespace NUMINAMATH_CALUDE_jason_earnings_l1394_139457

/-- Calculates Jason's total earnings for the week given his work hours and rates --/
theorem jason_earnings (after_school_rate : ℝ) (saturday_rate : ℝ) (total_hours : ℝ) (saturday_hours : ℝ) :
  after_school_rate = 4 ∧ 
  saturday_rate = 6 ∧ 
  total_hours = 18 ∧ 
  saturday_hours = 8 →
  (total_hours - saturday_hours) * after_school_rate + saturday_hours * saturday_rate = 88 := by
  sorry

end NUMINAMATH_CALUDE_jason_earnings_l1394_139457


namespace NUMINAMATH_CALUDE_inscribed_circle_chord_length_l1394_139453

theorem inscribed_circle_chord_length (a b : Real) (h1 : a > 0) (h2 : b > 0) (h3 : a^2 + b^2 = 1) :
  let r := (a + b - 1) / 2
  let chord_length := Real.sqrt (1 - 2 * r^2)
  chord_length = Real.sqrt 2 / 2 := by sorry

end NUMINAMATH_CALUDE_inscribed_circle_chord_length_l1394_139453


namespace NUMINAMATH_CALUDE_find_x_l1394_139407

theorem find_x : ∃ x : ℚ, (3 * x + 4) / 6 = 15 ∧ x = 86 / 3 := by
  sorry

end NUMINAMATH_CALUDE_find_x_l1394_139407


namespace NUMINAMATH_CALUDE_company_picnic_attendance_l1394_139452

theorem company_picnic_attendance (men_attendance : Real) (women_attendance : Real) 
  (total_attendance : Real) :
  men_attendance = 0.2 →
  women_attendance = 0.4 →
  total_attendance = 0.30000000000000004 →
  ∃ (men_percentage : Real),
    men_percentage * men_attendance + (1 - men_percentage) * women_attendance = total_attendance ∧
    men_percentage = 0.5 := by
  sorry

end NUMINAMATH_CALUDE_company_picnic_attendance_l1394_139452


namespace NUMINAMATH_CALUDE_angle_between_vectors_l1394_139435

def a : ℝ × ℝ := (2, -1)

theorem angle_between_vectors (b : ℝ × ℝ) (θ : ℝ) 
  (h1 : ‖b‖ = 2 * Real.sqrt 5)
  (h2 : (a.1 + b.1) * a.1 + (a.2 + b.2) * a.2 = 10)
  (h3 : θ = Real.arccos ((a.1 * b.1 + a.2 * b.2) / (‖a‖ * ‖b‖)))
  : θ = π / 3 := by
  sorry

end NUMINAMATH_CALUDE_angle_between_vectors_l1394_139435


namespace NUMINAMATH_CALUDE_intersection_implies_union_l1394_139408

-- Define the sets M and N
def M (a : ℝ) : Set ℝ := {x | (x - a) * (x - 3) = 0}
def N : Set ℝ := {x | (x - 4) * (x - 1) = 0}

-- State the theorem
theorem intersection_implies_union (a : ℝ) : 
  (M a ∩ N ≠ ∅) → (M a ∪ N = {1, 3, 4}) := by
  sorry

end NUMINAMATH_CALUDE_intersection_implies_union_l1394_139408


namespace NUMINAMATH_CALUDE_tip_percentage_calculation_l1394_139472

theorem tip_percentage_calculation (total_bill : ℝ) (sales_tax_rate : ℝ) (food_price : ℝ) : 
  total_bill = 211.20 ∧ 
  sales_tax_rate = 0.10 ∧ 
  food_price = 160 → 
  (total_bill - food_price * (1 + sales_tax_rate)) / (food_price * (1 + sales_tax_rate)) = 0.20 := by
  sorry

end NUMINAMATH_CALUDE_tip_percentage_calculation_l1394_139472


namespace NUMINAMATH_CALUDE_larger_solution_quadratic_l1394_139487

theorem larger_solution_quadratic (x : ℝ) :
  x^2 - 9*x - 22 = 0 → x ≤ 11 ∧ (∃ y, y^2 - 9*y - 22 = 0 ∧ y ≠ x) :=
by
  sorry

end NUMINAMATH_CALUDE_larger_solution_quadratic_l1394_139487


namespace NUMINAMATH_CALUDE_set_intersection_problem_l1394_139448

theorem set_intersection_problem :
  let M : Set ℝ := {x | x^2 - x = 0}
  let N : Set ℝ := {-1, 0}
  M ∩ N = {0} := by
sorry

end NUMINAMATH_CALUDE_set_intersection_problem_l1394_139448


namespace NUMINAMATH_CALUDE_distance_sum_on_unit_circle_l1394_139488

theorem distance_sum_on_unit_circle (a b : ℝ) (h : a^2 + b^2 = 1) :
  a^4 + b^4 + ((a - b)^4 / 4) + ((a + b)^4 / 4) = 3/2 := by sorry

end NUMINAMATH_CALUDE_distance_sum_on_unit_circle_l1394_139488


namespace NUMINAMATH_CALUDE_overtime_hours_example_l1394_139444

/-- Represents a worker's pay structure and hours worked -/
structure WorkerPay where
  ordinaryRate : ℚ  -- Rate for ordinary hours in dollars
  overtimeRate : ℚ  -- Rate for overtime hours in dollars
  totalHours : ℕ    -- Total hours worked
  totalPay : ℚ      -- Total pay received in dollars

/-- Calculates the number of overtime hours worked -/
def overtimeHours (w : WorkerPay) : ℕ :=
  sorry

/-- Theorem stating that given the specific conditions, the overtime hours are 8 -/
theorem overtime_hours_example :
  let w : WorkerPay := {
    ordinaryRate := 60/100,  -- 60 cents
    overtimeRate := 90/100,  -- 90 cents
    totalHours := 50,
    totalPay := 3240/100     -- $32.40
  }
  overtimeHours w = 8 := by sorry

end NUMINAMATH_CALUDE_overtime_hours_example_l1394_139444


namespace NUMINAMATH_CALUDE_final_black_goats_count_l1394_139442

theorem final_black_goats_count (total : ℕ) (initial_black : ℕ) (new_black : ℕ) :
  total = 93 →
  initial_black = 66 →
  new_black = 21 →
  initial_black ≤ total →
  let initial_white := total - initial_black
  let new_total_black := initial_black + new_black
  let deaths := min initial_white new_total_black
  new_total_black - deaths = 60 :=
by
  sorry

end NUMINAMATH_CALUDE_final_black_goats_count_l1394_139442


namespace NUMINAMATH_CALUDE_emily_calculation_l1394_139454

theorem emily_calculation (a b c : ℝ) 
  (h1 : a - (2*b - c) = 15) 
  (h2 : a - 2*b - c = 5) : 
  a - 2*b = 10 := by sorry

end NUMINAMATH_CALUDE_emily_calculation_l1394_139454


namespace NUMINAMATH_CALUDE_distance_to_origin_l1394_139499

/-- The distance between point P(3,1) and the origin (0,0) in the Cartesian coordinate system is √10. -/
theorem distance_to_origin : Real.sqrt ((3 : ℝ) ^ 2 + (1 : ℝ) ^ 2) = Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_distance_to_origin_l1394_139499


namespace NUMINAMATH_CALUDE_sun_rise_set_differences_l1394_139483

/-- Represents a geographical location with latitude and longitude -/
structure Location where
  latitude : Real
  longitude : Real

/-- Calculates the time difference of sunrise between two locations given a solar declination -/
def sunriseTimeDifference (loc1 loc2 : Location) (solarDeclination : Real) : Real :=
  sorry

/-- Calculates the time difference of sunset between two locations given a solar declination -/
def sunsetTimeDifference (loc1 loc2 : Location) (solarDeclination : Real) : Real :=
  sorry

def szeged : Location := { latitude := 46.25, longitude := 20.1667 }
def nyiregyhaza : Location := { latitude := 47.9667, longitude := 21.75 }
def winterSolsticeDeclination : Real := -23.5

theorem sun_rise_set_differences (ε : Real) :
  (ε > 0) →
  (∃ d : Real, abs (d - winterSolsticeDeclination) < ε ∧
    sunriseTimeDifference szeged nyiregyhaza d > 0) ∧
  (∃ d : Real, sunsetTimeDifference szeged nyiregyhaza d < 0) :=
sorry

end NUMINAMATH_CALUDE_sun_rise_set_differences_l1394_139483


namespace NUMINAMATH_CALUDE_smallest_a_value_l1394_139434

/-- Represents a parabola with equation y = ax^2 + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The theorem stating the smallest possible value of a for the given parabola -/
theorem smallest_a_value (p : Parabola) 
  (vertex_x : p.a * (3/4)^2 + p.b * (3/4) + p.c = -25/16) 
  (vertex_y : -p.b / (2 * p.a) = 3/4)
  (a_positive : p.a > 0)
  (sum_integer : ∃ n : ℤ, p.a + p.b + p.c = n) :
  9 ≤ p.a ∧ ∀ a' : ℝ, 0 < a' ∧ a' < 9 → 
    ¬∃ (b' c' : ℝ) (n : ℤ), 
      a' * (3/4)^2 + b' * (3/4) + c' = -25/16 ∧
      -b' / (2 * a') = 3/4 ∧
      a' + b' + c' = n := by
  sorry

end NUMINAMATH_CALUDE_smallest_a_value_l1394_139434


namespace NUMINAMATH_CALUDE_value_of_P_l1394_139492

theorem value_of_P : ∃ P : ℚ, (3/4 : ℚ) * (1/9 : ℚ) * P = (1/4 : ℚ) * (1/8 : ℚ) * 160 ∧ P = 60 := by
  sorry

end NUMINAMATH_CALUDE_value_of_P_l1394_139492


namespace NUMINAMATH_CALUDE_team_selection_ways_l1394_139481

def total_boys : ℕ := 10
def total_girls : ℕ := 12
def team_size : ℕ := 8
def required_boys : ℕ := 5
def required_girls : ℕ := 3

theorem team_selection_ways :
  (Nat.choose total_boys required_boys) * (Nat.choose total_girls required_girls) = 55440 :=
by sorry

end NUMINAMATH_CALUDE_team_selection_ways_l1394_139481


namespace NUMINAMATH_CALUDE_range_of_a_l1394_139466

open Set Real

theorem range_of_a (a : ℝ) : 
  (∀ x ∈ Icc 1 2, 3 * x^2 - a ≥ 0) ∧ 
  (∃ x : ℝ, x^2 + 2*a*x + 2 - a = 0) →
  a ≤ -2 ∨ (1 ≤ a ∧ a ≤ 3) :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l1394_139466


namespace NUMINAMATH_CALUDE_fixed_point_of_logarithmic_function_l1394_139476

theorem fixed_point_of_logarithmic_function (a : ℝ) (ha : a > 0) (ha' : a ≠ 1) :
  let f : ℝ → ℝ := λ x ↦ 1 + Real.log x / Real.log a
  f 1 = 1 := by
  sorry

end NUMINAMATH_CALUDE_fixed_point_of_logarithmic_function_l1394_139476


namespace NUMINAMATH_CALUDE_unique_k_solution_l1394_139443

def g (n : ℤ) : ℤ :=
  if n % 2 = 0 then n + 5 else (n + 1) / 2

theorem unique_k_solution (k : ℤ) (h1 : k % 2 = 0) (h2 : g (g (g k)) = 61) : k = 236 := by
  sorry

end NUMINAMATH_CALUDE_unique_k_solution_l1394_139443


namespace NUMINAMATH_CALUDE_sand_in_partial_bag_l1394_139418

theorem sand_in_partial_bag (total_sand : ℝ) (bag_capacity : ℝ) (h1 : total_sand = 1254.75) (h2 : bag_capacity = 73.5) :
  total_sand - (bag_capacity * ⌊total_sand / bag_capacity⌋) = 5.25 := by
  sorry

end NUMINAMATH_CALUDE_sand_in_partial_bag_l1394_139418


namespace NUMINAMATH_CALUDE_integral_cos_plus_exp_l1394_139420

theorem integral_cos_plus_exp (π : Real) : ∫ x in -π..0, (Real.cos x + Real.exp x) = 1 - 1 / Real.exp π := by
  sorry

end NUMINAMATH_CALUDE_integral_cos_plus_exp_l1394_139420
