import Mathlib

namespace NUMINAMATH_CALUDE_g_8_equals_1036_l2595_259523

def g (x : ℝ) : ℝ := 3*x^4 - 22*x^3 + 37*x^2 - 28*x - 84

theorem g_8_equals_1036 : g 8 = 1036 := by
  sorry

end NUMINAMATH_CALUDE_g_8_equals_1036_l2595_259523


namespace NUMINAMATH_CALUDE_m_equals_two_sufficient_not_necessary_l2595_259508

-- Define sets A and B
def A (m : ℝ) : Set ℝ := {1, m^2}
def B : Set ℝ := {2, 4}

-- Define the property we want to prove
def property (m : ℝ) : Prop := A m ∩ B = {4}

-- Theorem statement
theorem m_equals_two_sufficient_not_necessary :
  (∀ m : ℝ, m = 2 → property m) ∧
  (∃ m : ℝ, m ≠ 2 ∧ property m) :=
sorry

end NUMINAMATH_CALUDE_m_equals_two_sufficient_not_necessary_l2595_259508


namespace NUMINAMATH_CALUDE_complex_number_equality_l2595_259503

theorem complex_number_equality : ∀ z : ℂ, z = 1 - 2*I → z = -I := by
  sorry

end NUMINAMATH_CALUDE_complex_number_equality_l2595_259503


namespace NUMINAMATH_CALUDE_finite_good_numbers_not_divisible_by_l2595_259563

/-- τ(n) is the number of positive divisors of n -/
def tau (n : ℕ+) : ℕ := sorry

/-- n is a good number if τ(m) < τ(n) for all m < n -/
def is_good (n : ℕ+) : Prop :=
  ∀ m : ℕ+, m < n → tau m < tau n

/-- The set of good numbers not divisible by k is finite -/
theorem finite_good_numbers_not_divisible_by (k : ℕ+) :
  {n : ℕ+ | is_good n ∧ ¬k ∣ n}.Finite := by sorry

end NUMINAMATH_CALUDE_finite_good_numbers_not_divisible_by_l2595_259563


namespace NUMINAMATH_CALUDE_inequality_solution_set_l2595_259589

theorem inequality_solution_set (a : ℝ) :
  (∀ x, (x - a) * (x + a - 1) > 0 ↔ 
    (a = 1/2 ∧ x ≠ 1/2) ∨
    (a < 1/2 ∧ (x > 1 - a ∨ x < a)) ∨
    (a > 1/2 ∧ (x > a ∨ x < 1 - a))) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l2595_259589


namespace NUMINAMATH_CALUDE_triangle_area_implies_ab_value_l2595_259575

theorem triangle_area_implies_ab_value (a b : ℝ) : 
  a > 0 → b > 0 → 
  (1/2 * (12/a) * (12/b) = 9) → 
  a * b = 8 := by
sorry

end NUMINAMATH_CALUDE_triangle_area_implies_ab_value_l2595_259575


namespace NUMINAMATH_CALUDE_clearview_soccer_league_members_l2595_259571

/-- Represents the Clearview Soccer League --/
structure SoccerLeague where
  sockPrice : ℕ
  tshirtPriceIncrease : ℕ
  hatPrice : ℕ
  totalExpenditure : ℕ

/-- Calculates the number of members in the league --/
def calculateMembers (league : SoccerLeague) : ℕ :=
  let tshirtPrice := league.sockPrice + league.tshirtPriceIncrease
  let memberCost := 2 * (league.sockPrice + tshirtPrice + league.hatPrice)
  league.totalExpenditure / memberCost

/-- Theorem stating the number of members in the Clearview Soccer League --/
theorem clearview_soccer_league_members :
  let league := SoccerLeague.mk 3 7 2 3516
  calculateMembers league = 117 := by
  sorry

end NUMINAMATH_CALUDE_clearview_soccer_league_members_l2595_259571


namespace NUMINAMATH_CALUDE_complex_fraction_equality_l2595_259584

theorem complex_fraction_equality : (3 - I) / (1 - I) = 2 + I := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_equality_l2595_259584


namespace NUMINAMATH_CALUDE_intersection_implies_t_equals_two_l2595_259517

theorem intersection_implies_t_equals_two (t : ℝ) : 
  let M : Set ℝ := {1, t^2}
  let N : Set ℝ := {-2, t+2}
  (M ∩ N).Nonempty → t = 2 := by
sorry

end NUMINAMATH_CALUDE_intersection_implies_t_equals_two_l2595_259517


namespace NUMINAMATH_CALUDE_locus_is_S_l2595_259572

/-- A point moving along a line with constant velocity -/
structure MovingPoint where
  line : Set ℝ × ℝ  -- Represents a line in 2D space
  velocity : ℝ

/-- The locus of lines XX' -/
def locus (X X' : MovingPoint) : Set (Set ℝ × ℝ) := sorry

/-- The specific set S that represents the correct locus -/
def S : Set (Set ℝ × ℝ) := sorry

/-- Theorem stating that the locus of lines XX' is the set S -/
theorem locus_is_S (X X' : MovingPoint) (h : X.velocity ≠ X'.velocity) :
  locus X X' = S := by sorry

end NUMINAMATH_CALUDE_locus_is_S_l2595_259572


namespace NUMINAMATH_CALUDE_cos_double_alpha_l2595_259598

theorem cos_double_alpha (α : ℝ) : 
  (Real.cos α)^2 + (Real.sqrt 2 / 2)^2 = (Real.sqrt 3 / 2)^2 → 
  Real.cos (2 * α) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_cos_double_alpha_l2595_259598


namespace NUMINAMATH_CALUDE_probability_cover_both_clubs_l2595_259532

/-- The probability of selecting two students that cover both clubs -/
theorem probability_cover_both_clubs 
  (total_students : Nat) 
  (robotics_members : Nat) 
  (science_members : Nat) 
  (h1 : total_students = 30)
  (h2 : robotics_members = 22)
  (h3 : science_members = 24) :
  (Nat.choose total_students 2 - (Nat.choose (robotics_members + science_members - total_students) 2 + 
   Nat.choose (robotics_members - (robotics_members + science_members - total_students)) 2 + 
   Nat.choose (science_members - (robotics_members + science_members - total_students)) 2)) / 
   Nat.choose total_students 2 = 392 / 435 := by
sorry

end NUMINAMATH_CALUDE_probability_cover_both_clubs_l2595_259532


namespace NUMINAMATH_CALUDE_candy_distribution_l2595_259551

theorem candy_distribution (n : ℕ) : n > 0 → (100 % n = 0) → (99 % n = 0) → n = 11 := by
  sorry

end NUMINAMATH_CALUDE_candy_distribution_l2595_259551


namespace NUMINAMATH_CALUDE_hanas_stamp_collection_value_l2595_259514

theorem hanas_stamp_collection_value :
  ∀ (total_value : ℚ),
    (4 / 7 : ℚ) * total_value +  -- Amount sold at garage sale
    (1 / 3 : ℚ) * ((3 / 7 : ℚ) * total_value) = 28 →  -- Amount sold at auction
    total_value = 196 := by
  sorry

end NUMINAMATH_CALUDE_hanas_stamp_collection_value_l2595_259514


namespace NUMINAMATH_CALUDE_three_prime_divisors_of_eight_power_minus_one_l2595_259586

theorem three_prime_divisors_of_eight_power_minus_one (n : ℕ) :
  let x := 8^n - 1
  (∃ p q r : ℕ, Nat.Prime p ∧ Nat.Prime q ∧ Nat.Prime r ∧ p ≠ q ∧ q ≠ r ∧ p ≠ r ∧
    x = p * q * r) →
  (31 ∣ x) →
  x = 32767 := by
sorry

end NUMINAMATH_CALUDE_three_prime_divisors_of_eight_power_minus_one_l2595_259586


namespace NUMINAMATH_CALUDE_quadrant_I_solution_l2595_259518

theorem quadrant_I_solution (c : ℝ) : 
  (∃ x y : ℝ, x - y = 2 ∧ c * x + y = 3 ∧ x > 0 ∧ y > 0) ↔ -1 < c ∧ c < 3/2 :=
by sorry

end NUMINAMATH_CALUDE_quadrant_I_solution_l2595_259518


namespace NUMINAMATH_CALUDE_correct_average_proof_l2595_259527

/-- The number of students in the class -/
def num_students : ℕ := 60

/-- The incorrect average marks -/
def incorrect_average : ℚ := 82

/-- Reema's correct mark -/
def reema_correct : ℕ := 78

/-- Reema's incorrect mark -/
def reema_incorrect : ℕ := 68

/-- Mark's correct mark -/
def mark_correct : ℕ := 95

/-- Mark's incorrect mark -/
def mark_incorrect : ℕ := 91

/-- Jenny's correct mark -/
def jenny_correct : ℕ := 84

/-- Jenny's incorrect mark -/
def jenny_incorrect : ℕ := 74

/-- The correct average marks -/
def correct_average : ℚ := 82.40

theorem correct_average_proof :
  let incorrect_total := (incorrect_average * num_students : ℚ)
  let mark_difference := (reema_correct - reema_incorrect) + (mark_correct - mark_incorrect) + (jenny_correct - jenny_incorrect)
  let correct_total := incorrect_total + mark_difference
  (correct_total / num_students : ℚ) = correct_average := by sorry

end NUMINAMATH_CALUDE_correct_average_proof_l2595_259527


namespace NUMINAMATH_CALUDE_percentage_of_x_l2595_259540

theorem percentage_of_x (x y : ℝ) (h1 : x / y = 4) (h2 : y ≠ 0) : (2 * x - y) / x = 175 / 100 := by
  sorry

end NUMINAMATH_CALUDE_percentage_of_x_l2595_259540


namespace NUMINAMATH_CALUDE_share_division_l2595_259562

theorem share_division (total : ℚ) (a b c : ℚ) 
  (h_total : total = 585)
  (h_equal : 4 * a = 6 * b ∧ 6 * b = 3 * c)
  (h_sum : a + b + c = total) :
  c = 260 := by
  sorry

end NUMINAMATH_CALUDE_share_division_l2595_259562


namespace NUMINAMATH_CALUDE_portias_school_students_l2595_259511

theorem portias_school_students (portia_students lara_students : ℕ) : 
  portia_students = 2 * lara_students →
  portia_students + lara_students = 3000 →
  portia_students = 2000 := by
  sorry

end NUMINAMATH_CALUDE_portias_school_students_l2595_259511


namespace NUMINAMATH_CALUDE_abc_sum_mod_7_l2595_259565

theorem abc_sum_mod_7 (a b c : ℕ) : 
  0 < a ∧ a < 7 ∧ 
  0 < b ∧ b < 7 ∧ 
  0 < c ∧ c < 7 ∧ 
  (a * b * c) % 7 = 1 ∧ 
  (5 * c) % 7 = 2 ∧ 
  (6 * b) % 7 = (3 + b) % 7 → 
  (a + b + c) % 7 = 4 := by
sorry

end NUMINAMATH_CALUDE_abc_sum_mod_7_l2595_259565


namespace NUMINAMATH_CALUDE_cost_price_percentage_l2595_259559

-- Define the profit percent
def profit_percent : ℝ := 25

-- Define the relationship between selling price (SP) and cost price (CP)
def selling_price_relation (CP SP : ℝ) : Prop :=
  SP = CP * (1 + profit_percent / 100)

-- Theorem statement
theorem cost_price_percentage (CP SP : ℝ) :
  selling_price_relation CP SP →
  CP / SP * 100 = 80 := by
sorry

end NUMINAMATH_CALUDE_cost_price_percentage_l2595_259559


namespace NUMINAMATH_CALUDE_number_of_students_l2595_259516

theorem number_of_students (possible_outcomes : ℕ) (total_results : ℕ) : 
  possible_outcomes = 3 → total_results = 59049 → 
  ∃ n : ℕ, possible_outcomes ^ n = total_results ∧ n = 10 :=
by sorry

end NUMINAMATH_CALUDE_number_of_students_l2595_259516


namespace NUMINAMATH_CALUDE_rose_bundle_price_l2595_259599

theorem rose_bundle_price (rose_price : ℕ) (total_roses : ℕ) (num_bundles : ℕ) 
  (h1 : rose_price = 500)
  (h2 : total_roses = 200)
  (h3 : num_bundles = 25) :
  (rose_price * total_roses) / num_bundles = 4000 :=
by sorry

end NUMINAMATH_CALUDE_rose_bundle_price_l2595_259599


namespace NUMINAMATH_CALUDE_range_of_a_l2595_259513

/-- Line l: 3x + 4y + a = 0 -/
def line_l (a : ℝ) (x y : ℝ) : Prop := 3*x + 4*y + a = 0

/-- Circle C: (x-2)² + y² = 2 -/
def circle_C (x y : ℝ) : Prop := (x - 2)^2 + y^2 = 2

/-- Point M is on line l -/
def M_on_line_l (a : ℝ) (M : ℝ × ℝ) : Prop := line_l a M.1 M.2

/-- Tangent condition: �angle PMQ = 90° -/
def tangent_condition (M : ℝ × ℝ) : Prop := 
  ∃ (P Q : ℝ × ℝ), circle_C P.1 P.2 ∧ circle_C Q.1 Q.2 ∧ 
    (M.1 - P.1) * (M.1 - Q.1) + (M.2 - P.2) * (M.2 - Q.2) = 0

/-- Main theorem -/
theorem range_of_a (a : ℝ) : 
  (∃ M : ℝ × ℝ, M_on_line_l a M ∧ tangent_condition M) → 
  -16 ≤ a ∧ a ≤ 4 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l2595_259513


namespace NUMINAMATH_CALUDE_inscribed_circle_radius_rhombus_l2595_259539

/-- The radius of a circle inscribed in a rhombus with given diagonals -/
theorem inscribed_circle_radius_rhombus (d1 d2 : ℝ) (h1 : d1 = 14) (h2 : d2 = 30) :
  let a := Real.sqrt ((d1 / 2) ^ 2 + (d2 / 2) ^ 2)
  let r := (d1 * d2) / (8 * a)
  r = 105 / (2 * Real.sqrt 274) := by
  sorry

end NUMINAMATH_CALUDE_inscribed_circle_radius_rhombus_l2595_259539


namespace NUMINAMATH_CALUDE_number_of_divisors_of_60_l2595_259521

theorem number_of_divisors_of_60 : Nat.card {d : Nat | d > 0 ∧ 60 % d = 0} = 12 := by
  sorry

end NUMINAMATH_CALUDE_number_of_divisors_of_60_l2595_259521


namespace NUMINAMATH_CALUDE_square_sum_equals_48_l2595_259542

theorem square_sum_equals_48 (x y : ℝ) (h1 : x - 2*y = 4) (h2 : x*y = 8) : x^2 + 4*y^2 = 48 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_equals_48_l2595_259542


namespace NUMINAMATH_CALUDE_latest_departure_time_correct_l2595_259541

/-- Represents time in 24-hour format -/
structure Time where
  hours : Nat
  minutes : Nat
  valid : hours < 24 ∧ minutes < 60

/-- Calculates the difference between two times in minutes -/
def timeDiffMinutes (t1 t2 : Time) : Nat :=
  (t1.hours - t2.hours) * 60 + (t1.minutes - t2.minutes)

/-- The flight departure time -/
def flightTime : Time := { hours := 20, minutes := 0, valid := by simp }

/-- The recommended check-in time in minutes -/
def checkInTime : Nat := 120

/-- The time needed to drive to the airport in minutes -/
def driveTime : Nat := 45

/-- The time needed to park and reach the terminal in minutes -/
def parkAndWalkTime : Nat := 15

/-- The latest time they can leave their house -/
def latestDepartureTime : Time := { hours := 17, minutes := 0, valid := by simp }

theorem latest_departure_time_correct :
  timeDiffMinutes flightTime latestDepartureTime = checkInTime + driveTime + parkAndWalkTime :=
sorry

end NUMINAMATH_CALUDE_latest_departure_time_correct_l2595_259541


namespace NUMINAMATH_CALUDE_cube_third_times_eighth_equals_one_over_216_l2595_259507

theorem cube_third_times_eighth_equals_one_over_216 :
  (1 / 3 : ℚ)^3 * (1 / 8 : ℚ) = 1 / 216 := by sorry

end NUMINAMATH_CALUDE_cube_third_times_eighth_equals_one_over_216_l2595_259507


namespace NUMINAMATH_CALUDE_equation_solution_l2595_259590

/-- Given an equation y = a + b/x where a and b are constants, 
    if y = 3 when x = 2 and y = 2 when x = 4, then a + b = 5 -/
theorem equation_solution (a b : ℝ) : 
  (3 = a + b / 2) → (2 = a + b / 4) → (a + b = 5) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2595_259590


namespace NUMINAMATH_CALUDE_distinct_sums_count_l2595_259531

def S : Finset ℕ := {2, 5, 8, 11, 14, 17, 20}

def fourDistinctSum (s : Finset ℕ) : Finset ℕ :=
  (s.powerset.filter (λ t => t.card = 4)).image (λ t => t.sum id)

theorem distinct_sums_count : (fourDistinctSum S).card = 12 := by
  sorry

end NUMINAMATH_CALUDE_distinct_sums_count_l2595_259531


namespace NUMINAMATH_CALUDE_cubic_equation_root_l2595_259500

theorem cubic_equation_root (c d : ℚ) : 
  (3 + Real.sqrt 5 : ℝ)^3 + c * (3 + Real.sqrt 5)^2 + d * (3 + Real.sqrt 5) + 20 = 0 → 
  d = -26 := by
sorry

end NUMINAMATH_CALUDE_cubic_equation_root_l2595_259500


namespace NUMINAMATH_CALUDE_banana_orange_equivalence_l2595_259588

/-- The cost of fruits at Lola's Fruit Stand -/
structure FruitCost where
  banana_apple_ratio : ℚ  -- 4 bananas = 3 apples
  apple_orange_ratio : ℚ  -- 9 apples = 5 oranges

/-- The theorem stating the relationship between bananas and oranges -/
theorem banana_orange_equivalence (fc : FruitCost) 
  (h1 : fc.banana_apple_ratio = 4 / 3)
  (h2 : fc.apple_orange_ratio = 9 / 5) : 
  24 * (fc.apple_orange_ratio * fc.banana_apple_ratio) = 10 := by
  sorry

#check banana_orange_equivalence

end NUMINAMATH_CALUDE_banana_orange_equivalence_l2595_259588


namespace NUMINAMATH_CALUDE_cos_squared_plus_sin_double_l2595_259549

open Real

theorem cos_squared_plus_sin_double (α : ℝ) (h : tan α = 2) : 
  cos α ^ 2 + sin (2 * α) = 1 := by
  sorry

end NUMINAMATH_CALUDE_cos_squared_plus_sin_double_l2595_259549


namespace NUMINAMATH_CALUDE_square_has_most_symmetry_axes_l2595_259593

/-- The number of axes of symmetry for a square -/
def square_symmetry_axes : ℕ := 4

/-- The number of axes of symmetry for an equilateral triangle -/
def equilateral_triangle_symmetry_axes : ℕ := 3

/-- The number of axes of symmetry for an isosceles triangle -/
def isosceles_triangle_symmetry_axes : ℕ := 1

/-- The number of axes of symmetry for an isosceles trapezoid -/
def isosceles_trapezoid_symmetry_axes : ℕ := 1

/-- The shape with the most axes of symmetry -/
def shape_with_most_symmetry_axes : ℕ := square_symmetry_axes

theorem square_has_most_symmetry_axes :
  shape_with_most_symmetry_axes = square_symmetry_axes ∧
  shape_with_most_symmetry_axes > equilateral_triangle_symmetry_axes ∧
  shape_with_most_symmetry_axes > isosceles_triangle_symmetry_axes ∧
  shape_with_most_symmetry_axes > isosceles_trapezoid_symmetry_axes :=
by sorry

end NUMINAMATH_CALUDE_square_has_most_symmetry_axes_l2595_259593


namespace NUMINAMATH_CALUDE_daily_servings_sold_l2595_259522

theorem daily_servings_sold (cost profit_A profit_B revenue total_profit : ℚ)
  (h1 : cost = 14)
  (h2 : profit_A = 20)
  (h3 : profit_B = 18)
  (h4 : revenue = 1120)
  (h5 : total_profit = 280) :
  ∃ (x y : ℚ), x + y = 60 ∧ 
    profit_A * x + profit_B * y = revenue ∧
    (profit_A - cost) * x + (profit_B - cost) * y = total_profit :=
by sorry

end NUMINAMATH_CALUDE_daily_servings_sold_l2595_259522


namespace NUMINAMATH_CALUDE_solution_value_l2595_259554

theorem solution_value (x a : ℝ) : x = 2 ∧ 2*x + 3*a = 10 → a = 2 := by
  sorry

end NUMINAMATH_CALUDE_solution_value_l2595_259554


namespace NUMINAMATH_CALUDE_exists_solution_in_interval_l2595_259533

theorem exists_solution_in_interval : 
  ∃ z : ℝ, -10 ≤ z ∧ z ≤ 10 ∧ Real.exp (2 * z) = (z - 2) / (z + 2) := by
  sorry

end NUMINAMATH_CALUDE_exists_solution_in_interval_l2595_259533


namespace NUMINAMATH_CALUDE_geometric_sequence_tenth_term_l2595_259556

theorem geometric_sequence_tenth_term : 
  ∀ (a : ℚ) (r : ℚ),
    a = 5 →
    a * r = 20 / 3 →
    a * r^9 = 1310720 / 19683 :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_tenth_term_l2595_259556


namespace NUMINAMATH_CALUDE_lines_parallel_in_plane_l2595_259574

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (contained_in : Line → Plane → Prop)
variable (parallel : Line → Plane → Prop)
variable (coplanar : Line → Line → Plane → Prop)
variable (parallel_lines : Line → Line → Prop)

-- State the theorem
theorem lines_parallel_in_plane 
  (m n : Line) (α β : Plane) :
  m ≠ n →  -- m and n are distinct
  α ≠ β →  -- α and β are distinct
  contained_in m α →
  parallel n α →
  coplanar m n β →
  parallel_lines m n :=
sorry

end NUMINAMATH_CALUDE_lines_parallel_in_plane_l2595_259574


namespace NUMINAMATH_CALUDE_locus_of_T_l2595_259528

-- Define the ellipse
def ellipse (x y : ℝ) : Prop := x^2 / 4 + y^2 = 1

-- Define point P
def P : ℝ × ℝ := (1, 0)

-- Define vertices A and B
def A : ℝ × ℝ := (-2, 0)
def B : ℝ × ℝ := (2, 0)

-- Define a point on the ellipse that is not A or B
def M (x y : ℝ) : Prop := ellipse x y ∧ (x, y) ≠ A ∧ (x, y) ≠ B

-- Define point N as the intersection of MP and the ellipse
def N (x y : ℝ) : Prop := 
  M x y → ∃ t : ℝ, ellipse (x + t * (1 - x)) (y + t * (-y)) ∧ t ≠ 0

-- Define point T as the intersection of AM and BN
def T (x y : ℝ) : Prop :=
  ∃ (xm ym : ℝ), M xm ym ∧
  ∃ (xn yn : ℝ), N xn yn ∧
  (y / (x + 2) = ym / (xm + 2)) ∧
  (y / (x - 2) = yn / (xn - 2))

-- Theorem statement
theorem locus_of_T : ∀ x y : ℝ, T x y → y ≠ 0 → x = 4 := by sorry

end NUMINAMATH_CALUDE_locus_of_T_l2595_259528


namespace NUMINAMATH_CALUDE_width_of_sum_l2595_259569

/-- A convex curve in a 2D plane -/
structure ConvexCurve where
  -- Add necessary fields here
  
/-- The width of a convex curve in a given direction -/
def width (K : ConvexCurve) (direction : ℝ × ℝ) : ℝ :=
  sorry

/-- The sum of two convex curves -/
def curve_sum (K₁ K₂ : ConvexCurve) : ConvexCurve :=
  sorry

/-- Theorem: The width of the sum of two convex curves is the sum of their individual widths -/
theorem width_of_sum (K₁ K₂ : ConvexCurve) (direction : ℝ × ℝ) :
  width (curve_sum K₁ K₂) direction = width K₁ direction + width K₂ direction :=
sorry

end NUMINAMATH_CALUDE_width_of_sum_l2595_259569


namespace NUMINAMATH_CALUDE_square_floor_theorem_l2595_259525

/-- Represents a square floor tiled with congruent square tiles -/
structure SquareFloor where
  side_length : ℕ

/-- The number of black tiles on the diagonals of a square floor -/
def black_tiles (floor : SquareFloor) : ℕ :=
  2 * floor.side_length - 1

/-- The total number of tiles on a square floor -/
def total_tiles (floor : SquareFloor) : ℕ :=
  floor.side_length * floor.side_length

/-- Theorem: If a square floor has 75 black tiles on its diagonals, 
    then the total number of tiles is 1444 -/
theorem square_floor_theorem :
  ∃ (floor : SquareFloor), black_tiles floor = 75 ∧ total_tiles floor = 1444 :=
by
  sorry

end NUMINAMATH_CALUDE_square_floor_theorem_l2595_259525


namespace NUMINAMATH_CALUDE_line_L_equation_l2595_259570

-- Define the lines l₁, l₂, and l₃
def l₁ (x y : ℝ) : Prop := 3 * x + 2 * y - 1 = 0
def l₂ (x y : ℝ) : Prop := 5 * x + 2 * y + 1 = 0
def l₃ (x y : ℝ) : Prop := 3 * x - 5 * y + 6 = 0

-- Define the intersection point of l₁ and l₂
def intersection_point : ℝ × ℝ := (-1, 2)

-- Define line L
def L (x y : ℝ) : Prop := 5 * x + 3 * y - 1 = 0

-- Theorem statement
theorem line_L_equation :
  (∀ x y : ℝ, L x y ↔ 
    (x = intersection_point.1 ∧ y = intersection_point.2 ∨
    ∃ t : ℝ, x = intersection_point.1 + t ∧ y = intersection_point.2 - (5/3) * t)) ∧
  (∀ x y : ℝ, L x y → l₃ x y → 
    (x - intersection_point.1) * 3 + (y - intersection_point.2) * (-5) = 0) :=
by sorry


end NUMINAMATH_CALUDE_line_L_equation_l2595_259570


namespace NUMINAMATH_CALUDE_weather_forecast_inaccuracy_l2595_259501

theorem weather_forecast_inaccuracy (p_a p_b : ℝ) 
  (h_a : p_a = 0.9) 
  (h_b : p_b = 0.6) 
  (h_independent : True) -- Representing independence
  : (1 - p_a) * (1 - p_b) = 0.04 := by
  sorry

end NUMINAMATH_CALUDE_weather_forecast_inaccuracy_l2595_259501


namespace NUMINAMATH_CALUDE_A_intersect_B_eq_open_interval_l2595_259529

-- Define set A
def A : Set ℝ := {x | |x - 1| < 2}

-- Define set B
def B : Set ℝ := {x | Real.log x / Real.log 2 > Real.log x / Real.log 3}

-- Theorem statement
theorem A_intersect_B_eq_open_interval :
  A ∩ B = {x | 1 < x ∧ x < 3} := by sorry

end NUMINAMATH_CALUDE_A_intersect_B_eq_open_interval_l2595_259529


namespace NUMINAMATH_CALUDE_estimate_sqrt_expression_l2595_259519

theorem estimate_sqrt_expression :
  6 < (Real.sqrt 54 + 2 * Real.sqrt 3) * Real.sqrt (1/3) ∧
  (Real.sqrt 54 + 2 * Real.sqrt 3) * Real.sqrt (1/3) < 7 :=
by sorry

end NUMINAMATH_CALUDE_estimate_sqrt_expression_l2595_259519


namespace NUMINAMATH_CALUDE_smallest_yellow_marbles_l2595_259553

theorem smallest_yellow_marbles (total : ℕ) (blue red green yellow : ℕ) : 
  blue = total / 5 →
  red = 2 * green →
  green = 10 →
  blue + red + green + yellow = total →
  yellow ≥ 10 ∧ ∀ y : ℕ, y < 10 → ¬(
    ∃ t : ℕ, t / 5 + 2 * 10 + 10 + y = t ∧ 
    t / 5 + 2 * 10 + 10 + y = blue + red + green + y
  ) :=
by sorry

end NUMINAMATH_CALUDE_smallest_yellow_marbles_l2595_259553


namespace NUMINAMATH_CALUDE_sequence_sum_theorem_l2595_259585

/-- The sum of the sequence 1+2-3-4+5+6-7-8+...+2017+2018-2019-2020 -/
def sequenceSum : ℤ := -2020

/-- The last term in the sequence -/
def lastTerm : ℕ := 2020

/-- The number of complete groups of four in the sequence -/
def groupCount : ℕ := lastTerm / 4

/-- The sum of each group of four terms in the sequence -/
def groupSum : ℤ := -4

theorem sequence_sum_theorem :
  sequenceSum = groupCount * groupSum :=
by sorry

end NUMINAMATH_CALUDE_sequence_sum_theorem_l2595_259585


namespace NUMINAMATH_CALUDE_cookies_per_package_l2595_259557

theorem cookies_per_package
  (num_friends : ℕ)
  (num_packages : ℕ)
  (cookies_per_child : ℕ)
  (h1 : num_friends = 4)
  (h2 : num_packages = 3)
  (h3 : cookies_per_child = 15) :
  (num_friends + 1) * cookies_per_child / num_packages = 25 := by
  sorry

end NUMINAMATH_CALUDE_cookies_per_package_l2595_259557


namespace NUMINAMATH_CALUDE_initial_mean_calculation_l2595_259524

theorem initial_mean_calculation (n : ℕ) (M initial_wrong corrected_value new_mean : ℝ) :
  n = 50 ∧
  initial_wrong = 23 ∧
  corrected_value = 30 ∧
  new_mean = 36.5 ∧
  (n : ℝ) * new_mean = (n : ℝ) * M + (corrected_value - initial_wrong) →
  M = 36.36 := by
sorry

end NUMINAMATH_CALUDE_initial_mean_calculation_l2595_259524


namespace NUMINAMATH_CALUDE_always_negative_monotone_decreasing_l2595_259512

/-- The function f(x) = kx^2 - 2x + 4k -/
def f (k : ℝ) (x : ℝ) : ℝ := k * x^2 - 2 * x + 4 * k

/-- Theorem 1: f(x) is always less than zero on ℝ iff k < -1/2 -/
theorem always_negative (k : ℝ) : (∀ x : ℝ, f k x < 0) ↔ k < -1/2 := by sorry

/-- Theorem 2: f(x) is monotonically decreasing on [2, 4] iff k ≤ 1/4 -/
theorem monotone_decreasing (k : ℝ) : 
  (∀ x y : ℝ, 2 ≤ x ∧ x < y ∧ y ≤ 4 → f k x > f k y) ↔ k ≤ 1/4 := by sorry

end NUMINAMATH_CALUDE_always_negative_monotone_decreasing_l2595_259512


namespace NUMINAMATH_CALUDE_coin_game_probability_l2595_259573

/-- Represents a player in the coin game -/
inductive Player := | Abby | Bernardo | Carl | Debra

/-- Represents a ball color in the game -/
inductive BallColor := | Green | Red | Blue

/-- The number of rounds in the game -/
def numRounds : Nat := 5

/-- The number of coins each player starts with -/
def initialCoins : Nat := 5

/-- The number of balls of each color in the urn -/
def ballCounts : Fin 3 → Nat
  | 0 => 2  -- Green
  | 1 => 2  -- Red
  | 2 => 1  -- Blue

/-- Represents the state of the game after each round -/
structure GameState where
  coins : Player → Nat
  round : Nat

/-- Represents a single round of the game -/
def gameRound (state : GameState) : GameState := sorry

/-- The probability of a specific outcome in a single round -/
def roundProbability (outcome : Player → BallColor) : Rat := sorry

/-- The probability of returning to the initial state after all rounds -/
def finalProbability : Rat := sorry

/-- The main theorem stating the probability of each player having 5 coins at the end -/
theorem coin_game_probability : finalProbability = 64 / 15625 := sorry

end NUMINAMATH_CALUDE_coin_game_probability_l2595_259573


namespace NUMINAMATH_CALUDE_min_perimeter_isosceles_triangles_l2595_259510

/-- Represents an isosceles triangle with integer side lengths -/
structure IsoscelesTriangle where
  leg : ℕ
  base : ℕ

/-- Calculates the perimeter of an isosceles triangle -/
def perimeter (t : IsoscelesTriangle) : ℕ := 2 * t.leg + t.base

/-- Calculates the area of an isosceles triangle -/
noncomputable def area (t : IsoscelesTriangle) : ℝ :=
  (t.base : ℝ) * (Real.sqrt ((t.leg : ℝ)^2 - ((t.base : ℝ)/2)^2)) / 2

/-- Theorem: The minimum perimeter of two noncongruent integer-sided isosceles triangles
    with the same perimeter, same area, and bases in the ratio 8:7 is 676 -/
theorem min_perimeter_isosceles_triangles :
  ∃ (t₁ t₂ : IsoscelesTriangle),
    t₁ ≠ t₂ ∧
    perimeter t₁ = perimeter t₂ ∧
    area t₁ = area t₂ ∧
    8 * t₁.base = 7 * t₂.base ∧
    (∀ (s₁ s₂ : IsoscelesTriangle),
      s₁ ≠ s₂ →
      perimeter s₁ = perimeter s₂ →
      area s₁ = area s₂ →
      8 * s₁.base = 7 * s₂.base →
      perimeter t₁ ≤ perimeter s₁) ∧
    perimeter t₁ = 676 :=
sorry

end NUMINAMATH_CALUDE_min_perimeter_isosceles_triangles_l2595_259510


namespace NUMINAMATH_CALUDE_troy_straw_distribution_l2595_259535

/-- Given the conditions of Troy's straw distribution problem, prove that
    the number of straws fed to adult pigs is 120. -/
theorem troy_straw_distribution
  (total_straws : ℕ)
  (num_piglets : ℕ)
  (straws_per_piglet : ℕ)
  (h1 : total_straws = 300)
  (h2 : num_piglets = 20)
  (h3 : straws_per_piglet = 6)
  (h4 : ∃ (x : ℕ), x + x ≤ total_straws ∧ x = num_piglets * straws_per_piglet) :
  ∃ (x : ℕ), x = 120 ∧ x + x ≤ total_straws ∧ x = num_piglets * straws_per_piglet :=
sorry

end NUMINAMATH_CALUDE_troy_straw_distribution_l2595_259535


namespace NUMINAMATH_CALUDE_third_consecutive_odd_integer_l2595_259597

/-- Given three consecutive odd integers where 3 times the first is 3 more than twice the third, 
    prove that the third integer is 15. -/
theorem third_consecutive_odd_integer (x : ℤ) : 
  (∃ y z : ℤ, 
    y = x + 2 ∧ 
    z = x + 4 ∧ 
    Odd x ∧ 
    Odd y ∧ 
    Odd z ∧ 
    3 * x = 2 * z + 3) →
  x + 4 = 15 := by
sorry

end NUMINAMATH_CALUDE_third_consecutive_odd_integer_l2595_259597


namespace NUMINAMATH_CALUDE_max_distance_on_circle_common_chord_equation_three_common_tangents_l2595_259505

-- Define the circles
def circle_C (x y : ℝ) : Prop := x^2 + y^2 = 4
def circle_C1 (x y : ℝ) : Prop := x^2 + y^2 + 4*x - 4*y = 0
def circle_C2 (x y : ℝ) : Prop := x^2 + y^2 + 2*x - 12 = 0
def circle_C3 (x y : ℝ) : Prop := x^2 + y^2 + 2*x = 0
def circle_C4 (x y : ℝ) : Prop := x^2 + y^2 - 4*x - 8*y + 4 = 0

-- Theorem 1
theorem max_distance_on_circle :
  ∀ x₁ y₁ : ℝ, circle_C x₁ y₁ → (∀ x y : ℝ, circle_C x y → (x - 1)^2 + (y - 2*Real.sqrt 2)^2 ≤ (x₁ - 1)^2 + (y₁ - 2*Real.sqrt 2)^2) →
  (x₁ - 1)^2 + (y₁ - 2*Real.sqrt 2)^2 = 25 :=
sorry

-- Theorem 2
theorem common_chord_equation :
  ∀ x y : ℝ, (circle_C1 x y ∧ circle_C2 x y) → x - 2*y + 6 = 0 :=
sorry

-- Theorem 3
theorem three_common_tangents :
  ∃! n : ℕ, n = 3 ∧ 
  (∀ l : ℝ → ℝ → Prop, (∀ x y : ℝ, (circle_C3 x y → l x y) ∧ (circle_C4 x y → l x y)) →
  (∃ x₁ y₁ x₂ y₂ : ℝ, x₁ ≠ x₂ ∧ y₁ ≠ y₂ ∧ l x₁ y₁ ∧ l x₂ y₂)) →
  n = 3 :=
sorry

end NUMINAMATH_CALUDE_max_distance_on_circle_common_chord_equation_three_common_tangents_l2595_259505


namespace NUMINAMATH_CALUDE_largest_of_eight_consecutive_integers_l2595_259594

theorem largest_of_eight_consecutive_integers (n : ℕ) 
  (h1 : n > 0) 
  (h2 : n + (n+1) + (n+2) + (n+3) + (n+4) + (n+5) + (n+6) + (n+7) = 4328) : 
  n + 7 = 544 := by
  sorry

end NUMINAMATH_CALUDE_largest_of_eight_consecutive_integers_l2595_259594


namespace NUMINAMATH_CALUDE_discounted_good_price_l2595_259568

/-- The price of a good after applying successive discounts -/
def discounted_price (initial_price : ℝ) : ℝ :=
  initial_price * 0.75 * 0.85 * 0.90 * 0.93

theorem discounted_good_price (P : ℝ) :
  discounted_price P = 6600 → P = 11118.75 := by
  sorry

end NUMINAMATH_CALUDE_discounted_good_price_l2595_259568


namespace NUMINAMATH_CALUDE_average_speed_calculation_l2595_259580

def initial_reading : ℕ := 2552
def final_reading : ℕ := 2772
def total_time : ℕ := 9

theorem average_speed_calculation :
  (final_reading - initial_reading : ℚ) / total_time = 220 / 9 := by sorry

end NUMINAMATH_CALUDE_average_speed_calculation_l2595_259580


namespace NUMINAMATH_CALUDE_equation_solution_l2595_259595

-- Define the equation
def equation (m : ℝ) (x : ℝ) : Prop :=
  (2*m - 6) * x^(|m| - 2) = m^2

-- State the theorem
theorem equation_solution :
  ∃ (m : ℝ), ∀ (x : ℝ), equation m x ↔ x = -3/4 :=
sorry

end NUMINAMATH_CALUDE_equation_solution_l2595_259595


namespace NUMINAMATH_CALUDE_square_circle_area_ratio_l2595_259560

theorem square_circle_area_ratio (a r : ℝ) (h : a > 0) (k : r > 0) : 
  4 * a = 2 * 2 * Real.pi * r → a^2 / (Real.pi * r^2) = Real.pi := by
  sorry

end NUMINAMATH_CALUDE_square_circle_area_ratio_l2595_259560


namespace NUMINAMATH_CALUDE_program_cost_calculation_l2595_259526

/-- Calculates the total cost for running a computer program -/
theorem program_cost_calculation (program_time_seconds : ℝ) : 
  let milliseconds_per_second : ℝ := 1000
  let overhead_cost : ℝ := 1.07
  let cost_per_millisecond : ℝ := 0.023
  let tape_mounting_cost : ℝ := 5.35
  let program_time_milliseconds : ℝ := program_time_seconds * milliseconds_per_second
  let computer_time_cost : ℝ := program_time_milliseconds * cost_per_millisecond
  let total_cost : ℝ := overhead_cost + computer_time_cost + tape_mounting_cost
  program_time_seconds = 1.5 → total_cost = 40.92 := by
  sorry

end NUMINAMATH_CALUDE_program_cost_calculation_l2595_259526


namespace NUMINAMATH_CALUDE_als_original_investment_l2595_259576

-- Define the original investment amounts
variable (a b c d : ℝ)

-- Define the conditions
axiom total_investment : a + b + c + d = 1200
axiom different_amounts : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d
axiom final_total : (a - 150) + 3*b + 3*c + 2*d = 1800

-- Theorem to prove
theorem als_original_investment : a = 825 := by
  sorry

end NUMINAMATH_CALUDE_als_original_investment_l2595_259576


namespace NUMINAMATH_CALUDE_divisibility_theorem_l2595_259520

theorem divisibility_theorem (a b c : ℕ) (h1 : ∀ (p : ℕ), Nat.Prime p → c % (p^2) ≠ 0) 
  (h2 : (a^2) ∣ (b^2 * c)) : a ∣ b := by
  sorry

end NUMINAMATH_CALUDE_divisibility_theorem_l2595_259520


namespace NUMINAMATH_CALUDE_complex_fraction_calculation_l2595_259566

theorem complex_fraction_calculation : 
  27 * ((2 + 2/3) - (3 + 1/4)) / ((1 + 1/2) + (2 + 1/5)) = -(4 + 43/74) := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_calculation_l2595_259566


namespace NUMINAMATH_CALUDE_fourth_place_votes_l2595_259545

theorem fourth_place_votes (total_votes : ℕ) (num_candidates : ℕ) 
  (difference1 : ℕ) (difference2 : ℕ) (difference3 : ℕ) : 
  total_votes = 979 →
  num_candidates = 4 →
  difference1 = 53 →
  difference2 = 79 →
  difference3 = 105 →
  ∃ (winner_votes : ℕ),
    winner_votes - difference1 + 
    winner_votes - difference2 + 
    winner_votes - difference3 + 
    winner_votes = total_votes ∧
    winner_votes - difference3 = 199 :=
by
  sorry

end NUMINAMATH_CALUDE_fourth_place_votes_l2595_259545


namespace NUMINAMATH_CALUDE_quadratic_always_nonnegative_l2595_259534

theorem quadratic_always_nonnegative (a : ℝ) :
  (∀ x : ℝ, x^2 + (a - 2) * x + (1/4 : ℝ) ≥ 0) ↔ 1 ≤ a ∧ a ≤ 3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_always_nonnegative_l2595_259534


namespace NUMINAMATH_CALUDE_not_prime_29n_plus_11_l2595_259546

theorem not_prime_29n_plus_11 (n : ℕ+) 
  (h1 : ∃ x : ℕ, 3 * n + 1 = x^2) 
  (h2 : ∃ y : ℕ, 10 * n + 1 = y^2) : 
  ¬ Nat.Prime (29 * n + 11) := by
sorry

end NUMINAMATH_CALUDE_not_prime_29n_plus_11_l2595_259546


namespace NUMINAMATH_CALUDE_blue_fish_count_l2595_259567

theorem blue_fish_count (total_fish goldfish : ℕ) (h1 : total_fish = 22) (h2 : goldfish = 15) :
  total_fish - goldfish = 7 := by
  sorry

end NUMINAMATH_CALUDE_blue_fish_count_l2595_259567


namespace NUMINAMATH_CALUDE_division_of_25_by_4_l2595_259536

theorem division_of_25_by_4 : ∃ (q r : ℕ), 25 = 4 * q + r ∧ r < 4 ∧ q = 6 := by
  sorry

end NUMINAMATH_CALUDE_division_of_25_by_4_l2595_259536


namespace NUMINAMATH_CALUDE_exactly_one_divisible_l2595_259555

theorem exactly_one_divisible (p a b c d : ℕ) : 
  Prime p → 
  p % 2 = 1 →
  0 < a → a < p →
  0 < b → b < p →
  0 < c → c < p →
  0 < d → d < p →
  p ∣ (a^2 + b^2) →
  p ∣ (c^2 + d^2) →
  (p ∣ (a*c + b*d) ∧ ¬(p ∣ (a*d + b*c))) ∨ (¬(p ∣ (a*c + b*d)) ∧ p ∣ (a*d + b*c)) :=
by sorry

end NUMINAMATH_CALUDE_exactly_one_divisible_l2595_259555


namespace NUMINAMATH_CALUDE_rhombus_area_l2595_259543

/-- The area of a rhombus with side length 4 and an interior angle of 45 degrees is 8√2 -/
theorem rhombus_area (s : ℝ) (θ : ℝ) (h1 : s = 4) (h2 : θ = π / 4) :
  s * s * Real.sin θ = 8 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_rhombus_area_l2595_259543


namespace NUMINAMATH_CALUDE_inequality_solution_l2595_259561

theorem inequality_solution (x : ℝ) : 
  (x / (x + 1) + (x - 3) / (2 * x) ≥ 4) ↔ (x ∈ Set.Icc (-3) (-1/5)) :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_l2595_259561


namespace NUMINAMATH_CALUDE_max_odd_integers_with_even_product_l2595_259509

theorem max_odd_integers_with_even_product (integers : Finset ℕ) 
  (h1 : integers.card = 7)
  (h2 : ∀ n ∈ integers, n > 0)
  (h3 : Even (integers.prod id)) :
  { odd_count : ℕ // odd_count ≤ 6 ∧ 
    ∃ (odd_subset : Finset ℕ), 
      odd_subset ⊆ integers ∧ 
      odd_subset.card = odd_count ∧ 
      ∀ n ∈ odd_subset, Odd n } :=
by sorry

end NUMINAMATH_CALUDE_max_odd_integers_with_even_product_l2595_259509


namespace NUMINAMATH_CALUDE_new_ratio_is_7_to_5_l2595_259502

/-- Represents the ratio of toddlers to infants -/
structure Ratio :=
  (toddlers : ℕ)
  (infants : ℕ)

def initial_ratio : Ratio := ⟨7, 3⟩
def toddler_count : ℕ := 42
def new_infants : ℕ := 12

def calculate_new_ratio (r : Ratio) (t : ℕ) (n : ℕ) : Ratio :=
  let initial_infants := t * r.infants / r.toddlers
  ⟨t, initial_infants + n⟩

theorem new_ratio_is_7_to_5 :
  let new_ratio := calculate_new_ratio initial_ratio toddler_count new_infants
  ∃ (k : ℕ), k > 0 ∧ new_ratio.toddlers = 7 * k ∧ new_ratio.infants = 5 * k :=
sorry

end NUMINAMATH_CALUDE_new_ratio_is_7_to_5_l2595_259502


namespace NUMINAMATH_CALUDE_triangle_ratio_proof_l2595_259548

theorem triangle_ratio_proof (A B C : Real) (a b c : Real) :
  -- Triangle ABC exists
  0 < a ∧ 0 < b ∧ 0 < c →
  -- Angle A is 60°
  A = Real.pi / 3 →
  -- Side b is 1
  b = 1 →
  -- Area of triangle is √3/2
  (1/2) * b * c * Real.sin A = Real.sqrt 3 / 2 →
  -- The sum of angles in a triangle is π
  A + B + C = Real.pi →
  -- Triangle inequality
  a < b + c ∧ b < a + c ∧ c < a + b →
  -- Prove that the expression equals 2
  (a + b + c) / (Real.sin A + Real.sin B + Real.sin C) = 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_ratio_proof_l2595_259548


namespace NUMINAMATH_CALUDE_race_head_start_l2595_259537

theorem race_head_start (v_a v_b L : ℝ) (h : v_a = (16 / 15) * v_b) :
  (L / v_a = (L - (L / 16)) / v_b) → (L / 16 : ℝ) = L - (L - (L / 16)) := by
  sorry

end NUMINAMATH_CALUDE_race_head_start_l2595_259537


namespace NUMINAMATH_CALUDE_child_ticket_cost_l2595_259506

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
  (total_revenue - (total_attendees - child_attendees) * adult_ticket_cost) / child_attendees = 1 :=
by sorry

end NUMINAMATH_CALUDE_child_ticket_cost_l2595_259506


namespace NUMINAMATH_CALUDE_max_value_z_l2595_259504

/-- The maximum value of z = x - 2y subject to constraints -/
theorem max_value_z (x y : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) (hxy : 2 * x + y ≤ 2) :
  ∃ (max_z : ℝ), max_z = 1 ∧ ∀ (z : ℝ), z = x - 2 * y → z ≤ max_z :=
by sorry

end NUMINAMATH_CALUDE_max_value_z_l2595_259504


namespace NUMINAMATH_CALUDE_amanda_quizzes_l2595_259544

/-- The number of quizzes Amanda has taken so far -/
def n : ℕ := sorry

/-- Amanda's average score on quizzes taken so far (as a percentage) -/
def current_average : ℚ := 92

/-- The required score on the final quiz to get an A (as a percentage) -/
def final_quiz_score : ℚ := 97

/-- The required average score over all quizzes to get an A (as a percentage) -/
def required_average : ℚ := 93

/-- The total number of quizzes including the final quiz -/
def total_quizzes : ℕ := 5

theorem amanda_quizzes : 
  n * current_average + final_quiz_score = required_average * total_quizzes ∧ n = 4 := by sorry

end NUMINAMATH_CALUDE_amanda_quizzes_l2595_259544


namespace NUMINAMATH_CALUDE_complex_simplification_and_multiplication_l2595_259582

theorem complex_simplification_and_multiplication :
  3 * ((4 - 3*Complex.I) - (2 + 5*Complex.I)) = 6 - 24*Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_simplification_and_multiplication_l2595_259582


namespace NUMINAMATH_CALUDE_trigonometric_ratio_proof_l2595_259547

theorem trigonometric_ratio_proof (α : Real) 
  (h : ∃ (x y : Real), x = 3/5 ∧ y = 4/5 ∧ x^2 + y^2 = 1 ∧ x = Real.cos α ∧ y = Real.sin α) : 
  (Real.cos (2*α)) / (1 + Real.sin (2*α)) = -1/7 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_ratio_proof_l2595_259547


namespace NUMINAMATH_CALUDE_ripe_orange_harvest_l2595_259587

/-- The number of days of harvest -/
def harvest_days : ℕ := 73

/-- The number of sacks of ripe oranges harvested per day -/
def daily_ripe_harvest : ℕ := 5

/-- The total number of sacks of ripe oranges harvested over the entire period -/
def total_ripe_harvest : ℕ := harvest_days * daily_ripe_harvest

theorem ripe_orange_harvest :
  total_ripe_harvest = 365 := by sorry

end NUMINAMATH_CALUDE_ripe_orange_harvest_l2595_259587


namespace NUMINAMATH_CALUDE_tangent_slope_at_point_A_l2595_259538

-- Define the function f(x) = x^2 + 3x
def f (x : ℝ) : ℝ := x^2 + 3*x

-- State the theorem
theorem tangent_slope_at_point_A :
  -- The derivative of f at x = 1 is equal to 5
  (deriv f) 1 = 5 :=
sorry

end NUMINAMATH_CALUDE_tangent_slope_at_point_A_l2595_259538


namespace NUMINAMATH_CALUDE_kevins_toads_l2595_259550

/-- The number of toads in Kevin's shoebox -/
def num_toads : ℕ := 8

/-- The number of worms each toad is fed daily -/
def worms_per_toad : ℕ := 3

/-- The time (in minutes) it takes Kevin to find each worm -/
def minutes_per_worm : ℕ := 15

/-- The time (in hours) it takes Kevin to find enough worms for all toads -/
def total_hours : ℕ := 6

/-- Theorem stating that the number of toads is 8 given the conditions -/
theorem kevins_toads : 
  num_toads = (total_hours * 60) / minutes_per_worm / worms_per_toad :=
by sorry

end NUMINAMATH_CALUDE_kevins_toads_l2595_259550


namespace NUMINAMATH_CALUDE_lucas_chocolate_problem_l2595_259577

theorem lucas_chocolate_problem (total_students : ℕ) 
  (candy_per_student : ℕ) 
  (h1 : total_students * candy_per_student = 40) 
  (h2 : (total_students - 3) * candy_per_student = 28) :
  candy_per_student = 4 := by
  sorry

end NUMINAMATH_CALUDE_lucas_chocolate_problem_l2595_259577


namespace NUMINAMATH_CALUDE_set_properties_l2595_259579

-- Define set A
def A : Set Int := {x | ∃ m n : Int, x = m^2 - n^2}

-- Define set B
def B : Set Int := {x | ∃ k : Int, x = 2*k + 1}

-- Theorem statement
theorem set_properties :
  (8 ∈ A ∧ 9 ∈ A ∧ 10 ∉ A) ∧
  (∀ x, x ∈ B → x ∈ A) ∧
  (∃ x, x ∈ A ∧ x ∉ B) ∧
  (∀ x, x ∈ A ∧ Even x ↔ ∃ k : Int, x = 4*k) :=
by sorry

end NUMINAMATH_CALUDE_set_properties_l2595_259579


namespace NUMINAMATH_CALUDE_selection_ways_eq_756_l2595_259564

/-- The number of ways to select 5 people from a group of 12, 
    where at most 2 out of 3 specific people can be selected -/
def selection_ways : ℕ :=
  Nat.choose 9 5 + 
  (Nat.choose 3 1 * Nat.choose 9 4) + 
  (Nat.choose 3 2 * Nat.choose 9 3)

/-- Theorem stating that the number of selection ways is 756 -/
theorem selection_ways_eq_756 : selection_ways = 756 := by
  sorry

end NUMINAMATH_CALUDE_selection_ways_eq_756_l2595_259564


namespace NUMINAMATH_CALUDE_inequality_solution_l2595_259583

theorem inequality_solution (n : Int) :
  n ∈ ({-1, 0, 1, 2, 3} : Set Int) →
  ((-1/2 : ℚ)^n > (-1/5 : ℚ)^n) ↔ (n = -1 ∨ n = 2) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l2595_259583


namespace NUMINAMATH_CALUDE_sum_of_consecutive_odd_numbers_l2595_259578

theorem sum_of_consecutive_odd_numbers : 
  let odd_numbers := [997, 999, 1001, 1003, 1005]
  (List.sum odd_numbers) = 5100 - 95 := by
sorry

end NUMINAMATH_CALUDE_sum_of_consecutive_odd_numbers_l2595_259578


namespace NUMINAMATH_CALUDE_cross_quadrilateral_area_l2595_259581

/-- Given two rectangles ABCD and EFGH forming a cross shape, 
    prove that the area of quadrilateral AFCH is 52.5 -/
theorem cross_quadrilateral_area 
  (AB BC EF FG : ℝ) 
  (h_AB : AB = 9) 
  (h_BC : BC = 5) 
  (h_EF : EF = 3) 
  (h_FG : FG = 10) : 
  Real.sqrt ((AB * FG / 2 + BC * EF / 2) ^ 2 + (AB * BC + EF * FG - BC * EF) ^ 2) = 52.5 := by
  sorry

end NUMINAMATH_CALUDE_cross_quadrilateral_area_l2595_259581


namespace NUMINAMATH_CALUDE_tangent_line_equation_l2595_259515

-- Define the parabola function
def f (x : ℝ) : ℝ := 2 * x^2

-- Define the slope of the line parallel to 4x - y + 3 = 0
def m : ℝ := 4

-- Theorem statement
theorem tangent_line_equation :
  ∃ (x₀ y₀ : ℝ),
    -- The point (x₀, y₀) is on the parabola
    y₀ = f x₀ ∧
    -- The slope of the tangent line at (x₀, y₀) is m
    (deriv f) x₀ = m ∧
    -- The equation of the tangent line is 4x - y - 2 = 0
    ∀ (x y : ℝ), y - y₀ = m * (x - x₀) ↔ 4 * x - y - 2 = 0 :=
sorry

end NUMINAMATH_CALUDE_tangent_line_equation_l2595_259515


namespace NUMINAMATH_CALUDE_reference_city_hospitals_l2595_259591

/-- The number of hospitals in the reference city -/
def reference_hospitals : ℕ := sorry

/-- The number of stores in the reference city -/
def reference_stores : ℕ := 2000

/-- The number of schools in the reference city -/
def reference_schools : ℕ := 200

/-- The number of police stations in the reference city -/
def reference_police : ℕ := 20

/-- The total number of buildings in the new city -/
def new_city_total : ℕ := 2175

theorem reference_city_hospitals :
  reference_stores / 2 + 2 * reference_hospitals + (reference_schools - 50) + (reference_police + 5) = new_city_total →
  reference_hospitals = 500 := by
  sorry

end NUMINAMATH_CALUDE_reference_city_hospitals_l2595_259591


namespace NUMINAMATH_CALUDE_complex_equation_solution_l2595_259552

theorem complex_equation_solution (z : ℂ) (m n : ℝ) : 
  (Complex.abs (1 - z) + z = 10 - 3 * Complex.I) →
  (z = 5 - 3 * Complex.I) ∧
  (z^2 + m * z + n = 1 - 3 * Complex.I) →
  (m = 14 ∧ n = -103) := by
sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l2595_259552


namespace NUMINAMATH_CALUDE_problem_solution_l2595_259592

theorem problem_solution (x y : ℝ) 
  (eq1 : x + Real.sin y = 2010)
  (eq2 : x + 2010 * Real.cos y + 3 * Real.sin y = 2005)
  (h : 0 ≤ y ∧ y ≤ Real.pi) :
  x + y = 2009 + Real.pi / 2 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l2595_259592


namespace NUMINAMATH_CALUDE_smallest_solution_and_ratio_l2595_259558

theorem smallest_solution_and_ratio (x : ℝ) (a b c d : ℤ) : 
  (7 * x / 8 - 1 = 4 / x) →
  (x = (a + b * Real.sqrt c) / d) →
  (x ≥ (4 - 4 * Real.sqrt 15) / 7) →
  (x = (4 - 4 * Real.sqrt 15) / 7 → a * c * d / b = -105) :=
by sorry

end NUMINAMATH_CALUDE_smallest_solution_and_ratio_l2595_259558


namespace NUMINAMATH_CALUDE_product_of_roots_l2595_259530

theorem product_of_roots (x : ℝ) : 
  (24 * x^2 + 36 * x - 648 = 0) → 
  (∃ r₁ r₂ : ℝ, (24 * r₁^2 + 36 * r₁ - 648 = 0) ∧ 
                (24 * r₂^2 + 36 * r₂ - 648 = 0) ∧ 
                (r₁ * r₂ = -27)) := by
  sorry

end NUMINAMATH_CALUDE_product_of_roots_l2595_259530


namespace NUMINAMATH_CALUDE_ten_factorial_mod_thirteen_l2595_259596

def factorial (n : ℕ) : ℕ := Nat.factorial n

theorem ten_factorial_mod_thirteen : factorial 10 % 13 = 6 := by sorry

end NUMINAMATH_CALUDE_ten_factorial_mod_thirteen_l2595_259596
