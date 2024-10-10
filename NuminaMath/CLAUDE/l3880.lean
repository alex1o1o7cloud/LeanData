import Mathlib

namespace lawrence_county_summer_break_l3880_388043

/-- The number of kids who stay home during summer break in Lawrence county -/
def kids_stay_home (total_kids : ℕ) (kids_at_camp : ℕ) : ℕ :=
  total_kids - kids_at_camp

/-- Proof that 907,611 kids stay home during summer break in Lawrence county -/
theorem lawrence_county_summer_break :
  kids_stay_home 1363293 455682 = 907611 := by
sorry

end lawrence_county_summer_break_l3880_388043


namespace odometer_reading_l3880_388024

theorem odometer_reading (initial_reading lunch_reading total_distance : ℝ) :
  lunch_reading - initial_reading = 372.0 →
  total_distance = 584.3 →
  initial_reading = 212.3 :=
by sorry

end odometer_reading_l3880_388024


namespace min_S_and_max_m_l3880_388056

-- Define the function S
def S (x : ℝ) : ℝ := |x - 2| + |x - 4|

-- State the theorem
theorem min_S_and_max_m :
  (∃ (min_S : ℝ), ∀ x : ℝ, S x ≥ min_S ∧ ∃ x₀ : ℝ, S x₀ = min_S) ∧
  (∃ (max_m : ℝ), (∀ x y : ℝ, S x ≥ max_m * (-y^2 + 2*y)) ∧
    ∀ m : ℝ, (∀ x y : ℝ, S x ≥ m * (-y^2 + 2*y)) → m ≤ max_m) ∧
  (∀ x : ℝ, S x ≥ 2) ∧
  (∀ x y : ℝ, S x ≥ 2 * (-y^2 + 2*y)) :=
by sorry

end min_S_and_max_m_l3880_388056


namespace triangle_side_length_l3880_388066

theorem triangle_side_length (A B C : ℝ) (a b c : ℝ) :
  a = 2 →
  b = Real.sqrt 7 →
  B = π / 3 →  -- 60° in radians
  b^2 = a^2 + c^2 - 2*a*c*Real.cos B →
  c = 3 := by
sorry

end triangle_side_length_l3880_388066


namespace negation_equivalence_l3880_388065

theorem negation_equivalence :
  (¬ ∀ x : ℝ, x^2 - x + 3 > 0) ↔ (∃ x : ℝ, x^2 - x + 3 ≤ 0) := by
  sorry

end negation_equivalence_l3880_388065


namespace salary_comparison_l3880_388020

/-- Proves that given the salaries of A, B, and C are in the ratio of 1 : 2 : 3, 
    and the sum of B and C's salaries is 6000, 
    the percentage by which C's salary exceeds A's salary is 200%. -/
theorem salary_comparison (sa sb sc : ℝ) : 
  sa > 0 → sb > 0 → sc > 0 → 
  sb / sa = 2 → sc / sa = 3 → 
  sb + sc = 6000 → 
  (sc - sa) / sa * 100 = 200 := by
sorry

end salary_comparison_l3880_388020


namespace unique_k_for_prime_roots_l3880_388049

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n

def quadratic_roots (a b c : ℤ) : Set ℤ :=
  {x : ℤ | a * x^2 + b * x + c = 0}

theorem unique_k_for_prime_roots : 
  ∃! k : ℤ, ∀ x ∈ quadratic_roots 1 (-76) k, is_prime (x.natAbs) :=
sorry

end unique_k_for_prime_roots_l3880_388049


namespace prime_sqrt_sum_integer_implies_equal_l3880_388054

theorem prime_sqrt_sum_integer_implies_equal (p q : ℕ) : 
  Prime p → Prime q → 
  ∃ (z : ℤ), (Int.sqrt (p^2 + 7*p*q + q^2) + Int.sqrt (p^2 + 14*p*q + q^2) = z) → 
  p = q :=
by sorry

end prime_sqrt_sum_integer_implies_equal_l3880_388054


namespace standing_students_count_l3880_388029

/-- Given a school meeting with the following conditions:
  * total_attendees: The total number of attendees at the meeting
  * seated_students: The number of seated students
  * seated_teachers: The number of seated teachers

  This theorem proves that the number of standing students is equal to 25.
-/
theorem standing_students_count
  (total_attendees : Nat)
  (seated_students : Nat)
  (seated_teachers : Nat)
  (h1 : total_attendees = 355)
  (h2 : seated_students = 300)
  (h3 : seated_teachers = 30) :
  total_attendees - (seated_students + seated_teachers) = 25 := by
  sorry

#check standing_students_count

end standing_students_count_l3880_388029


namespace product_equals_888888_l3880_388019

theorem product_equals_888888 : 143 * 21 * 4 * 37 * 2 = 888888 := by
  sorry

end product_equals_888888_l3880_388019


namespace prime_square_mod_180_l3880_388037

theorem prime_square_mod_180 (p : ℕ) (h_prime : Nat.Prime p) (h_gt5 : p > 5) :
  ∃ (r₁ r₂ : ℕ), r₁ ≠ r₂ ∧ 
  (∀ (r : ℕ), p^2 % 180 = r → (r = r₁ ∨ r = r₂)) :=
sorry

end prime_square_mod_180_l3880_388037


namespace total_highlighters_l3880_388071

theorem total_highlighters (pink : ℕ) (yellow : ℕ) (blue : ℕ)
  (h1 : pink = 3)
  (h2 : yellow = 7)
  (h3 : blue = 5) :
  pink + yellow + blue = 15 := by
  sorry

end total_highlighters_l3880_388071


namespace quadratic_decreasing_implies_a_range_l3880_388072

/-- A quadratic function f(x) = x^2 + 2(a-1)x + 2 -/
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + 2*(a-1)*x + 2

/-- The interval (-∞, 4] -/
def interval : Set ℝ := Set.Iic 4

theorem quadratic_decreasing_implies_a_range (a : ℝ) :
  (∀ x ∈ interval, StrictMonoOn (f a) interval) → a < -5 :=
sorry

end quadratic_decreasing_implies_a_range_l3880_388072


namespace constant_reciprocal_sum_parabola_l3880_388052

/-- Theorem: Constant Reciprocal Sum of Squared Distances on Parabola
  Given a point P(a,0) on the x-axis and a line through P intersecting
  the parabola y^2 = 8x at points A and B, if the sum of reciprocals of
  squared distances 1/|AP^2| + 1/|BP^2| is constant for all such lines,
  then a = 4. -/
theorem constant_reciprocal_sum_parabola (a : ℝ) : 
  (∀ m : ℝ, ∃ A B : ℝ × ℝ, 
    (A.2)^2 = 8 * A.1 ∧ 
    (B.2)^2 = 8 * B.1 ∧ 
    A.1 = m * A.2 + a ∧ 
    B.1 = m * B.2 + a ∧
    (∃ k : ℝ, ∀ m : ℝ, 
      1 / ((A.1 - a)^2 + (A.2)^2) + 1 / ((B.1 - a)^2 + (B.2)^2) = k)) →
  a = 4 := by
sorry

end constant_reciprocal_sum_parabola_l3880_388052


namespace ellipse_and_line_intersection_l3880_388098

-- Define the ellipse C
def ellipse_C (x y a b : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1

-- Define the line l
def line_l (x y k : ℝ) : Prop := y = k * (x - 1)

theorem ellipse_and_line_intersection :
  ∀ (a b : ℝ), a > b ∧ b > 0 →
  ellipse_C (Real.sqrt 2) 1 a b →
  (∃ (x : ℝ), x > 0 ∧ ellipse_C x 0 a b ∧ x^2 = 2) →
  (∀ (k : ℝ), k > 0 →
    (∃ (x₁ y₁ x₂ y₂ : ℝ),
      ellipse_C x₁ y₁ a b ∧
      ellipse_C x₂ y₂ a b ∧
      line_l x₁ y₁ k ∧
      line_l x₂ y₂ k ∧
      x₂ - 1 = -x₁ ∧
      y₂ = -k - y₁) →
    k = Real.sqrt 2 / 2 ∧
    ∃ (x₁ y₁ x₂ y₂ : ℝ),
      ellipse_C x₁ y₁ a b ∧
      ellipse_C x₂ y₂ a b ∧
      line_l x₁ y₁ k ∧
      line_l x₂ y₂ k ∧
      Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2) = Real.sqrt 42 / 2) →
  a^2 = 4 ∧ b^2 = 2 :=
by sorry

end ellipse_and_line_intersection_l3880_388098


namespace triangle_area_l3880_388092

-- Define the triangle PQR
def Triangle (P Q R : ℝ × ℝ) : Prop :=
  -- Right triangle condition
  (Q.1 - P.1) * (R.1 - P.1) + (Q.2 - P.2) * (R.2 - P.2) = 0 ∧
  -- Angle Q = 60°
  (R.1 - Q.1)^2 + (R.2 - Q.2)^2 = (P.1 - Q.1)^2 + (P.2 - Q.2)^2 ∧
  -- Angle R = 30°
  4 * ((P.1 - R.1)^2 + (P.2 - R.2)^2) = (Q.1 - R.1)^2 + (Q.2 - R.2)^2 ∧
  -- QR = 12
  (Q.1 - R.1)^2 + (Q.2 - R.2)^2 = 144

-- Theorem statement
theorem triangle_area (P Q R : ℝ × ℝ) (h : Triangle P Q R) :
  let area := abs ((Q.1 - P.1) * (R.2 - P.2) - (R.1 - P.1) * (Q.2 - P.2)) / 2
  area = 18 * Real.sqrt 3 := by
  sorry

end triangle_area_l3880_388092


namespace polynomial_evaluation_l3880_388045

theorem polynomial_evaluation (f : ℝ → ℝ) :
  (∀ x, f (x^2 + 2) = x^4 + 6*x^2 + 4) →
  (∀ x, f (x^2 - 2) = x^4 - 2*x^2 - 4) :=
by sorry

end polynomial_evaluation_l3880_388045


namespace faster_train_speed_l3880_388069

theorem faster_train_speed 
  (distance : ℝ) 
  (time : ℝ) 
  (slower_speed : ℝ) 
  (h1 : distance = 536) 
  (h2 : time = 4) 
  (h3 : slower_speed = 60) :
  ∃ faster_speed : ℝ, 
    faster_speed = distance / time - slower_speed ∧ 
    faster_speed = 74 :=
by sorry

end faster_train_speed_l3880_388069


namespace prob_same_fee_prob_sum_fee_4_prob_sum_fee_6_l3880_388086

/-- Represents the rental time bracket for a bike rental -/
inductive RentalTime
  | WithinTwo
  | TwoToThree
  | ThreeToFour

/-- Calculates the rental fee based on the rental time -/
def rentalFee (time : RentalTime) : ℕ :=
  match time with
  | RentalTime.WithinTwo => 0
  | RentalTime.TwoToThree => 2
  | RentalTime.ThreeToFour => 4

/-- Represents the probability distribution for a person's rental time -/
structure RentalDistribution where
  withinTwo : ℚ
  twoToThree : ℚ
  threeToFour : ℚ
  sum_to_one : withinTwo + twoToThree + threeToFour = 1

/-- The rental distribution for person A -/
def distA : RentalDistribution :=
  { withinTwo := 1/4
  , twoToThree := 1/2
  , threeToFour := 1/4
  , sum_to_one := by norm_num }

/-- The rental distribution for person B -/
def distB : RentalDistribution :=
  { withinTwo := 1/2
  , twoToThree := 1/4
  , threeToFour := 1/4
  , sum_to_one := by norm_num }

/-- Theorem stating the probability that A and B pay the same fee -/
theorem prob_same_fee : 
  distA.withinTwo * distB.withinTwo + 
  distA.twoToThree * distB.twoToThree + 
  distA.threeToFour * distB.threeToFour = 5/16 := by sorry

/-- Theorem stating the probability that the sum of fees is 4 -/
theorem prob_sum_fee_4 :
  distA.withinTwo * distB.threeToFour + 
  distB.withinTwo * distA.threeToFour + 
  distA.twoToThree * distB.twoToThree = 5/16 := by sorry

/-- Theorem stating the probability that the sum of fees is 6 -/
theorem prob_sum_fee_6 :
  distA.twoToThree * distB.threeToFour + 
  distB.twoToThree * distA.threeToFour = 3/16 := by sorry

end prob_same_fee_prob_sum_fee_4_prob_sum_fee_6_l3880_388086


namespace base_seven_sum_l3880_388048

/-- Given A, B, and C are non-zero distinct digits in base 7 satisfying the equation, prove B + C = 6 in base 7 -/
theorem base_seven_sum (A B C : ℕ) : 
  A ≠ 0 ∧ B ≠ 0 ∧ C ≠ 0 ∧ 
  A < 7 ∧ B < 7 ∧ C < 7 ∧
  A ≠ B ∧ B ≠ C ∧ A ≠ C ∧
  (7^2 * A + 7 * B + C) + (7^2 * B + 7 * C + A) + (7^2 * C + 7 * A + B) = 7^3 * A + 7^2 * A + 7 * A →
  B + C = 6 :=
by sorry

end base_seven_sum_l3880_388048


namespace min_distance_parabola_circle_l3880_388042

/-- The minimum distance between a point on the parabola y^2 = x and a point on the circle (x-3)^2 + y^2 = 1 is (√11)/2 - 1 -/
theorem min_distance_parabola_circle :
  let parabola := {p : ℝ × ℝ | p.2^2 = p.1}
  let circle := {p : ℝ × ℝ | (p.1 - 3)^2 + p.2^2 = 1}
  ∃ (d : ℝ), d = Real.sqrt 11 / 2 - 1 ∧
    ∀ (m : ℝ × ℝ) (n : ℝ × ℝ), m ∈ parabola → n ∈ circle →
      Real.sqrt ((m.1 - n.1)^2 + (m.2 - n.2)^2) ≥ d :=
by sorry

end min_distance_parabola_circle_l3880_388042


namespace range_of_x_plus_y_min_distance_intersection_l3880_388040

-- Define the curve C
def on_curve_C (x y : ℝ) : Prop := x^2 = 4*y

-- Define the line l
def on_line_l (x y t α : ℝ) : Prop := x = t * Real.cos α ∧ y = 1 + t * Real.sin α

-- Theorem 1: Range of x+y
theorem range_of_x_plus_y (x y : ℝ) (h : on_curve_C x y) : x + y ≥ -1 := by
  sorry

-- Theorem 2: Minimum distance between intersection points
theorem min_distance_intersection (α : ℝ) : 
  ∃ (A B : ℝ × ℝ), 
    (on_curve_C A.1 A.2 ∧ ∃ t, on_line_l A.1 A.2 t α) ∧ 
    (on_curve_C B.1 B.2 ∧ ∃ t, on_line_l B.1 B.2 t α) ∧
    ∀ (P Q : ℝ × ℝ), 
      (on_curve_C P.1 P.2 ∧ ∃ t, on_line_l P.1 P.2 t α) →
      (on_curve_C Q.1 Q.2 ∧ ∃ t, on_line_l Q.1 Q.2 t α) →
      Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) ≤ Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2) ∧
      Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 4 := by
  sorry

end range_of_x_plus_y_min_distance_intersection_l3880_388040


namespace unique_number_exists_l3880_388076

theorem unique_number_exists : ∃! x : ℝ, x > 0 ∧ 100000 * x = 5 * (1 / x) := by
  sorry

end unique_number_exists_l3880_388076


namespace area_of_triangle_PQR_prove_area_of_triangle_PQR_l3880_388070

/-- Given two lines intersecting at point P(2,8), where one line has a slope of 3
    and the other has a slope of -1, and Q and R are the x-intercepts of these lines respectively,
    the area of triangle PQR is 128/3. -/
theorem area_of_triangle_PQR : ℝ → Prop :=
  fun area =>
    let P : ℝ × ℝ := (2, 8)
    let slope1 : ℝ := 3
    let slope2 : ℝ := -1
    let line1 := fun x => slope1 * (x - P.1) + P.2
    let line2 := fun x => slope2 * (x - P.1) + P.2
    let Q : ℝ × ℝ := (-(line1 0) / slope1, 0)
    let R : ℝ × ℝ := (-(line2 0) / slope2, 0)
    area = 128 / 3 ∧
    area = (1 / 2) * (R.1 - Q.1) * P.2

/-- Proof of the theorem -/
theorem prove_area_of_triangle_PQR : area_of_triangle_PQR (128 / 3) := by
  sorry

end area_of_triangle_PQR_prove_area_of_triangle_PQR_l3880_388070


namespace complement_of_28_45_l3880_388087

-- Define a type for angles in degrees and minutes
structure Angle :=
  (degrees : ℕ)
  (minutes : ℕ)

-- Define the complement of an angle
def complement (α : Angle) : Angle :=
  let totalMinutes := 90 * 60 - (α.degrees * 60 + α.minutes)
  ⟨totalMinutes / 60, totalMinutes % 60⟩

-- State the theorem
theorem complement_of_28_45 :
  let α : Angle := ⟨28, 45⟩
  complement α = ⟨61, 15⟩ := by
  sorry


end complement_of_28_45_l3880_388087


namespace trains_at_initial_positions_l3880_388073

/-- Represents a metro line with a given cycle time -/
structure MetroLine where
  cycletime : ℕ

/-- Represents a metro system with multiple lines -/
structure MetroSystem where
  lines : List MetroLine

/-- Checks if all trains return to their initial positions after a given time -/
def allTrainsAtInitialPositions (system : MetroSystem) (time : ℕ) : Prop :=
  ∀ line ∈ system.lines, time % line.cycletime = 0

/-- The metro system of city N -/
def cityNMetro : MetroSystem :=
  { lines := [
      { cycletime := 14 },  -- Red line
      { cycletime := 16 },  -- Blue line
      { cycletime := 18 }   -- Green line
    ]
  }

/-- Theorem: After 2016 minutes, all trains in city N's metro system will be at their initial positions -/
theorem trains_at_initial_positions :
  allTrainsAtInitialPositions cityNMetro 2016 :=
by
  sorry


end trains_at_initial_positions_l3880_388073


namespace evaluate_expression_l3880_388022

theorem evaluate_expression : 2 - (-3) * 2 - 4 - (-5) * 3 - 6 = 13 := by
  sorry

end evaluate_expression_l3880_388022


namespace batsman_average_l3880_388095

theorem batsman_average (previous_total : ℕ) (previous_average : ℚ) : 
  previous_total = 16 * previous_average ∧
  (previous_total + 65 : ℚ) / 17 = previous_average + 3 →
  (previous_total + 65 : ℚ) / 17 = 17 :=
by sorry

end batsman_average_l3880_388095


namespace quadratic_unique_solution_l3880_388010

theorem quadratic_unique_solution (a c : ℝ) : 
  (∃! x, a * x^2 + 15 * x + c = 0) → 
  (a + c = 24) → 
  (a < c) → 
  (a = (24 - Real.sqrt 351) / 2 ∧ c = (24 + Real.sqrt 351) / 2) :=
by sorry

end quadratic_unique_solution_l3880_388010


namespace money_lasts_four_weeks_l3880_388000

def total_earnings : ℕ := 27
def weekly_expenses : ℕ := 6

theorem money_lasts_four_weeks :
  (total_earnings / weekly_expenses : ℕ) = 4 :=
by sorry

end money_lasts_four_weeks_l3880_388000


namespace function_passes_through_point_two_two_l3880_388068

theorem function_passes_through_point_two_two 
  (a : ℝ) (ha : a > 0) (ha_ne_one : a ≠ 1) :
  let f := fun (x : ℝ) => a^(x - 2) + 1
  f 2 = 2 := by
sorry

end function_passes_through_point_two_two_l3880_388068


namespace derivative_f_l3880_388053

noncomputable def f (x : ℝ) := x * Real.sin x + Real.cos x

theorem derivative_f :
  deriv f = fun x ↦ x * Real.cos x := by sorry

end derivative_f_l3880_388053


namespace fencing_calculation_l3880_388085

/-- Calculates the fencing required for a rectangular field -/
theorem fencing_calculation (area : ℝ) (uncovered_side : ℝ) : 
  area = 600 → uncovered_side = 30 → 
  ∃ (width : ℝ), 
    area = uncovered_side * width ∧ 
    2 * width + uncovered_side = 70 :=
by sorry

end fencing_calculation_l3880_388085


namespace min_value_xyz_l3880_388097

theorem min_value_xyz (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (hsum : x + y + z = 3) : (x + y) / (x * y * z) ≥ 16 / 9 := by
  sorry

end min_value_xyz_l3880_388097


namespace sine_cosine_inequality_l3880_388002

theorem sine_cosine_inequality (x : ℝ) (n : ℕ) :
  (Real.sin (2 * x))^n + ((Real.sin x)^n - (Real.cos x)^n)^2 ≤ 1 := by sorry

end sine_cosine_inequality_l3880_388002


namespace probability_prime_sum_two_dice_l3880_388030

-- Define the number of sides on each die
def dice_sides : ℕ := 8

-- Define the set of possible prime sums
def prime_sums : Set ℕ := {2, 3, 5, 7, 11, 13}

-- Define a function to count favorable outcomes
def count_favorable_outcomes : ℕ := 29

-- Define the total number of possible outcomes
def total_outcomes : ℕ := dice_sides * dice_sides

-- Theorem statement
theorem probability_prime_sum_two_dice :
  (count_favorable_outcomes : ℚ) / total_outcomes = 29 / 64 :=
sorry

end probability_prime_sum_two_dice_l3880_388030


namespace a_range_l3880_388077

theorem a_range (a : ℝ) : 
  (∃ x : ℝ, x ∈ Set.Icc (-1) 1 ∧ a^2 * x^2 + a * x - 2 = 0) →
  (∃! x : ℝ, x^2 + 2*a*x + 2*a ≤ 0) →
  ¬((∃ x : ℝ, x ∈ Set.Icc (-1) 1 ∧ a^2 * x^2 + a * x - 2 = 0) ∧ 
    (∃! x : ℝ, x^2 + 2*a*x + 2*a ≤ 0)) →
  a ∈ Set.union (Set.Ioo (-1) 0) (Set.Ioo 0 1) :=
by sorry

end a_range_l3880_388077


namespace other_endpoint_of_line_segment_l3880_388059

/-- Given a line segment with midpoint (3, 1) and one endpoint (7, -3), prove that the other endpoint is (-1, 5) --/
theorem other_endpoint_of_line_segment (midpoint endpoint1 endpoint2 : ℝ × ℝ) : 
  midpoint = (3, 1) → endpoint1 = (7, -3) → 
  (midpoint.1 = (endpoint1.1 + endpoint2.1) / 2 ∧ 
   midpoint.2 = (endpoint1.2 + endpoint2.2) / 2) →
  endpoint2 = (-1, 5) := by
sorry

end other_endpoint_of_line_segment_l3880_388059


namespace ticket_price_possibilities_l3880_388017

theorem ticket_price_possibilities : ∃ (n : ℕ), n = (Nat.divisors 90 ∩ Nat.divisors 150).card ∧ n = 8 := by
  sorry

end ticket_price_possibilities_l3880_388017


namespace a₃_eq_10_l3880_388088

/-- The coefficient a₃ in the expansion of x^5 as a polynomial in (1+x) -/
def a₃ : ℝ := 10

/-- The function f(x) = x^5 -/
def f (x : ℝ) : ℝ := x^5

/-- The expansion of f(x) in terms of (1+x) -/
def f_expansion (x a₀ a₁ a₂ a₃ a₄ a₅ : ℝ) : ℝ :=
  a₀ + a₁*(1+x) + a₂*(1+x)^2 + a₃*(1+x)^3 + a₄*(1+x)^4 + a₅*(1+x)^5

/-- Theorem stating that a₃ = 10 in the expansion of x^5 -/
theorem a₃_eq_10 :
  ∀ x a₀ a₁ a₂ a₄ a₅ : ℝ, f x = f_expansion x a₀ a₁ a₂ a₃ a₄ a₅ → a₃ = 10 :=
by sorry

end a₃_eq_10_l3880_388088


namespace prob_non_matching_is_five_sixths_l3880_388038

/-- Represents the possible colors for shorts -/
inductive ShortsColor
| Black
| Gold
| Blue

/-- Represents the possible colors for jerseys -/
inductive JerseyColor
| White
| Gold

/-- The total number of possible color combinations -/
def total_combinations : ℕ := 6

/-- The number of non-matching color combinations -/
def non_matching_combinations : ℕ := 5

/-- Probability of selecting a non-matching color combination -/
def prob_non_matching : ℚ := non_matching_combinations / total_combinations

theorem prob_non_matching_is_five_sixths :
  prob_non_matching = 5 / 6 := by sorry

end prob_non_matching_is_five_sixths_l3880_388038


namespace cone_height_from_cylinder_l3880_388034

/-- Given a cylinder and cones with specified dimensions, prove the height of the cones. -/
theorem cone_height_from_cylinder (cylinder_radius cylinder_height cone_radius : ℝ) 
  (num_cones : ℕ) (h_cylinder_radius : cylinder_radius = 12) 
  (h_cylinder_height : cylinder_height = 10) (h_cone_radius : cone_radius = 4) 
  (h_num_cones : num_cones = 135) : 
  ∃ (cone_height : ℝ), 
    cone_height = 2 ∧ 
    (π * cylinder_radius^2 * cylinder_height = 
     num_cones * (1/3 * π * cone_radius^2 * cone_height)) := by
  sorry


end cone_height_from_cylinder_l3880_388034


namespace arithmetic_sequence_formula_l3880_388015

/-- An arithmetic sequence with specific properties -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  (∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d) ∧ 
  a 3 = 3 ∧
  a 1 + a 2 = 0

/-- The general term of the sequence -/
def GeneralTerm (n : ℕ) : ℝ := 2 * n - 3

theorem arithmetic_sequence_formula (a : ℕ → ℝ) :
  ArithmeticSequence a → (∀ n : ℕ, a n = GeneralTerm n) := by
  sorry

end arithmetic_sequence_formula_l3880_388015


namespace right_triangle_hypotenuse_l3880_388032

theorem right_triangle_hypotenuse (x : ℝ) :
  x > 0 ∧
  (1/2 * x * (2*x - 1) = 72) →
  Real.sqrt (x^2 + (2*x - 1)^2) = Real.sqrt 370 := by
  sorry

end right_triangle_hypotenuse_l3880_388032


namespace reciprocal_problem_l3880_388080

theorem reciprocal_problem (x : ℝ) (h : 8 * x - 6 = 10) : 200 * (1 / x) = 100 := by
  sorry

end reciprocal_problem_l3880_388080


namespace translation_complex_plane_l3880_388018

/-- A translation in the complex plane that takes 1 + 3i to 5 + 7i also takes 2 - i to 6 + 3i -/
theorem translation_complex_plane : 
  ∀ (f : ℂ → ℂ), 
  (∀ z : ℂ, ∃ w : ℂ, f z = z + w) → -- f is a translation
  (f (1 + 3*I) = 5 + 7*I) →         -- f takes 1 + 3i to 5 + 7i
  (f (2 - I) = 6 + 3*I) :=          -- f takes 2 - i to 6 + 3i
by sorry

end translation_complex_plane_l3880_388018


namespace find_n_l3880_388050

theorem find_n (n : ℕ) : lcm n 16 = 48 → gcd n 16 = 4 → n = 12 := by
  sorry

end find_n_l3880_388050


namespace tan_alpha_two_expressions_l3880_388055

theorem tan_alpha_two_expressions (α : Real) (h : Real.tan α = 2) :
  (2 * Real.sin α + Real.cos α) / (Real.sin α - 3 * Real.cos α) = -5 ∧
  Real.sin α * (Real.sin α + Real.cos α) = 6/5 := by
  sorry

end tan_alpha_two_expressions_l3880_388055


namespace hot_sauce_serving_size_l3880_388006

/-- Calculates the number of ounces per serving of hot sauce -/
theorem hot_sauce_serving_size (servings_per_day : ℕ) (quart_size : ℕ) (container_reduction : ℕ) (days_lasting : ℕ) :
  servings_per_day = 3 →
  quart_size = 32 →
  container_reduction = 2 →
  days_lasting = 20 →
  (quart_size - container_reduction : ℚ) / (servings_per_day * days_lasting) = 1/2 := by
  sorry

end hot_sauce_serving_size_l3880_388006


namespace distribution_four_to_three_l3880_388035

/-- The number of ways to distribute n distinct objects into k distinct groups,
    where each group must contain at least one object -/
def distributionCount (n k : ℕ) : ℕ := sorry

/-- The number of ways to choose r items from n items -/
def choose (n r : ℕ) : ℕ := sorry

theorem distribution_four_to_three :
  distributionCount 4 3 = 30 := by
  sorry

end distribution_four_to_three_l3880_388035


namespace expression_evaluation_l3880_388089

theorem expression_evaluation : 
  |5 - 8 * (3 - 12)^2| - |5 - 11| + Real.sqrt 16 + Real.sin (π / 2) = 642 := by
  sorry

end expression_evaluation_l3880_388089


namespace students_walking_distance_l3880_388007

/-- The problem of finding the distance students need to walk --/
theorem students_walking_distance 
  (teacher_speed : ℝ) 
  (teacher_initial_distance : ℝ)
  (student1_initial_distance : ℝ)
  (student2_initial_distance : ℝ)
  (student3_initial_distance : ℝ)
  (h1 : teacher_speed = 1.5)
  (h2 : teacher_initial_distance = 235)
  (h3 : student1_initial_distance = 87)
  (h4 : student2_initial_distance = 59)
  (h5 : student3_initial_distance = 26) :
  ∃ x : ℝ, 
    x = 42 ∧ 
    teacher_initial_distance - teacher_speed * x = 
      (student1_initial_distance - x) + 
      (student2_initial_distance - x) + 
      (student3_initial_distance - x) := by
  sorry

end students_walking_distance_l3880_388007


namespace polynomial_division_remainder_l3880_388011

theorem polynomial_division_remainder (k : ℚ) : 
  ∃! k, ∃ q : Polynomial ℚ, 
    3 * X^3 + k * X^2 - 8 * X + 52 = (3 * X + 4) * q + 7 := by
  sorry

end polynomial_division_remainder_l3880_388011


namespace total_matches_played_l3880_388090

theorem total_matches_played (average_all : ℝ) (average_first_six : ℝ) (average_last_four : ℝ)
  (h1 : average_all = 38.9)
  (h2 : average_first_six = 41)
  (h3 : average_last_four = 35.75) :
  ∃ n : ℕ, n = 10 ∧ average_all * n = average_first_six * 6 + average_last_four * 4 :=
by sorry

end total_matches_played_l3880_388090


namespace rectangular_enclosure_fence_posts_l3880_388047

/-- Calculates the number of fence posts required for a rectangular enclosure --/
def fencePostsRequired (length width postSpacing : ℕ) : ℕ :=
  2 * (length / postSpacing + width / postSpacing) + 4

/-- Proves that the minimum number of fence posts for the given dimensions is 30 --/
theorem rectangular_enclosure_fence_posts :
  fencePostsRequired 72 48 8 = 30 := by
  sorry

end rectangular_enclosure_fence_posts_l3880_388047


namespace branch_A_more_profitable_l3880_388084

/-- Represents a branch of the factory -/
inductive Branch
| A
| B

/-- Represents a grade of the product -/
inductive Grade
| A
| B
| C
| D

/-- Returns the processing fee for a given grade -/
def processingFee (g : Grade) : Int :=
  match g with
  | Grade.A => 90
  | Grade.B => 50
  | Grade.C => 20
  | Grade.D => -50

/-- Returns the processing cost for a given branch -/
def processingCost (b : Branch) : Int :=
  match b with
  | Branch.A => 25
  | Branch.B => 20

/-- Returns the frequency of a grade for a given branch -/
def frequency (b : Branch) (g : Grade) : Rat :=
  match b, g with
  | Branch.A, Grade.A => 40 / 100
  | Branch.A, Grade.B => 20 / 100
  | Branch.A, Grade.C => 20 / 100
  | Branch.A, Grade.D => 20 / 100
  | Branch.B, Grade.A => 28 / 100
  | Branch.B, Grade.B => 17 / 100
  | Branch.B, Grade.C => 34 / 100
  | Branch.B, Grade.D => 21 / 100

/-- Calculates the average profit for a given branch -/
def averageProfit (b : Branch) : Rat :=
  (processingFee Grade.A - processingCost b) * frequency b Grade.A +
  (processingFee Grade.B - processingCost b) * frequency b Grade.B +
  (processingFee Grade.C - processingCost b) * frequency b Grade.C +
  (processingFee Grade.D - processingCost b) * frequency b Grade.D

/-- Theorem stating that Branch A has higher average profit than Branch B -/
theorem branch_A_more_profitable : averageProfit Branch.A > averageProfit Branch.B := by
  sorry


end branch_A_more_profitable_l3880_388084


namespace revenue_change_after_price_and_sales_change_l3880_388023

theorem revenue_change_after_price_and_sales_change 
  (original_price original_quantity : ℝ) 
  (price_increase_percent : ℝ) 
  (sales_decrease_percent : ℝ) : 
  price_increase_percent = 60 → 
  sales_decrease_percent = 35 → 
  let new_price := original_price * (1 + price_increase_percent / 100)
  let new_quantity := original_quantity * (1 - sales_decrease_percent / 100)
  let original_revenue := original_price * original_quantity
  let new_revenue := new_price * new_quantity
  (new_revenue - original_revenue) / original_revenue * 100 = 4 := by
sorry

end revenue_change_after_price_and_sales_change_l3880_388023


namespace pencil_ratio_l3880_388075

theorem pencil_ratio (jeanine_initial : ℕ) (clare : ℕ) : 
  jeanine_initial = 18 →
  clare = jeanine_initial * 2 / 3 - 3 →
  clare.gcd jeanine_initial = clare →
  clare / (clare.gcd jeanine_initial) = 1 ∧ 
  jeanine_initial / (clare.gcd jeanine_initial) = 2 := by
sorry

end pencil_ratio_l3880_388075


namespace external_tangent_y_intercept_l3880_388099

-- Define the circles
def circle1_center : ℝ × ℝ := (1, 3)
def circle1_radius : ℝ := 3
def circle2_center : ℝ × ℝ := (10, 6)
def circle2_radius : ℝ := 7

-- Define the tangent line equation
def tangent_line (m b : ℝ) (x : ℝ) : ℝ := m * x + b

-- State the theorem
theorem external_tangent_y_intercept :
  ∃ (m b : ℝ), m > 0 ∧
  (∀ (x : ℝ), tangent_line m b x = m * x + b) ∧
  b = 9/4 := by
  sorry

end external_tangent_y_intercept_l3880_388099


namespace line_parallel_to_y_axis_l3880_388021

/-- A line passing through the point (-1, 3) and parallel to the y-axis has the equation x = -1 -/
theorem line_parallel_to_y_axis (line : Set (ℝ × ℝ)) : 
  ((-1, 3) ∈ line) → 
  (∀ (x y₁ y₂ : ℝ), ((x, y₁) ∈ line ∧ (x, y₂) ∈ line) → y₁ = y₂) →
  (line = {p : ℝ × ℝ | p.1 = -1}) :=
by sorry

end line_parallel_to_y_axis_l3880_388021


namespace solution_set_inequality_range_of_m_l3880_388094

-- Define the function f(x) = |x-2|
def f (x : ℝ) : ℝ := |x - 2|

-- Theorem for part 1
theorem solution_set_inequality (x : ℝ) :
  f x + f (2 * x + 1) ≥ 6 ↔ x ≤ -1 ∨ x ≥ 3 :=
sorry

-- Theorem for part 2
theorem range_of_m (a b m : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  (∀ x : ℝ, f (x - m) - f (-x) ≤ 4 / a + 1 / b) →
  -13 ≤ m ∧ m ≤ 5 :=
sorry

end solution_set_inequality_range_of_m_l3880_388094


namespace ramanujan_hardy_game_l3880_388081

theorem ramanujan_hardy_game (h r : ℂ) : 
  h * r = 32 - 8 * I ∧ h = 5 + 3 * I → r = 4 - 4 * I := by
  sorry

end ramanujan_hardy_game_l3880_388081


namespace triangle_is_obtuse_l3880_388064

theorem triangle_is_obtuse (a b c : ℝ) (ha : a = 4) (hb : b = 6) (hc : c = 8) :
  a^2 + b^2 < c^2 := by
  sorry

end triangle_is_obtuse_l3880_388064


namespace stating_standard_representation_of_point_l3880_388012

/-- 
Given a point in spherical coordinates (ρ, θ, φ), this function returns its standard representation
where 0 ≤ θ < 2π and 0 ≤ φ ≤ π.
-/
def standardSphericalRepresentation (ρ θ φ : Real) : Real × Real × Real :=
  sorry

/-- 
Theorem stating that the standard representation of the point (5, 3π/5, 9π/5) 
in spherical coordinates is (5, 8π/5, π/5).
-/
theorem standard_representation_of_point : 
  standardSphericalRepresentation 5 (3 * Real.pi / 5) (9 * Real.pi / 5) = 
    (5, 8 * Real.pi / 5, Real.pi / 5) := by
  sorry

end stating_standard_representation_of_point_l3880_388012


namespace function_inequality_implies_non_negative_l3880_388013

theorem function_inequality_implies_non_negative 
  (f : ℝ → ℝ) 
  (h : ∀ (x y : ℝ), f (x * y) + f (y - x) ≥ f (y + x)) : 
  ∀ (x : ℝ), f x ≥ 0 := by
sorry

end function_inequality_implies_non_negative_l3880_388013


namespace triangle_angle_B_l3880_388067

theorem triangle_angle_B (a b c : ℝ) (A B C : ℝ) :
  b = 50 * Real.sqrt 6 →
  c = 150 →
  C = π / 3 →
  b / Real.sin B = c / Real.sin C →
  B < C →
  B = π / 4 :=
sorry

end triangle_angle_B_l3880_388067


namespace greatest_power_of_three_l3880_388078

def p : ℕ := (List.range 34).foldl (· * ·) 1

theorem greatest_power_of_three (k : ℕ) : k ≤ 16 ↔ (3^k : ℕ) ∣ p := by
  sorry

end greatest_power_of_three_l3880_388078


namespace students_in_both_clubs_is_40_l3880_388041

/-- The number of students in both photography and science clubs -/
def students_in_both_clubs (total : ℕ) (photo : ℕ) (science : ℕ) (either : ℕ) : ℕ :=
  photo + science - either

/-- Theorem: Given the conditions from the problem, prove that there are 40 students in both clubs -/
theorem students_in_both_clubs_is_40 :
  students_in_both_clubs 300 120 140 220 = 40 := by
  sorry

end students_in_both_clubs_is_40_l3880_388041


namespace power_product_result_l3880_388039

theorem power_product_result : (-8)^20 * (1/4)^31 = 1/4 := by
  sorry

end power_product_result_l3880_388039


namespace largest_x_value_l3880_388063

theorem largest_x_value (x y : ℤ) : 
  (1/4 : ℚ) < (x : ℚ)/7 ∧ (x : ℚ)/7 < 2/3 ∧ x + y = 10 →
  x ≤ 4 ∧ (∃ (z : ℤ), (1/4 : ℚ) < (z : ℚ)/7 ∧ (z : ℚ)/7 < 2/3 ∧ z + (10 - z) = 10 ∧ z = 4) :=
by sorry

end largest_x_value_l3880_388063


namespace rectangle_ratio_theorem_l3880_388026

/-- Represents the configuration of rectangles around a square -/
structure RectangleConfiguration where
  inner_square_side : ℝ
  rectangle_short_side : ℝ
  rectangle_long_side : ℝ

/-- The theorem statement -/
theorem rectangle_ratio_theorem (config : RectangleConfiguration) :
  (config.inner_square_side > 0) →
  (config.rectangle_short_side > 0) →
  (config.rectangle_long_side > 0) →
  (config.inner_square_side + 2 * config.rectangle_short_side = 3 * config.inner_square_side) →
  (config.rectangle_long_side + config.rectangle_short_side = 3 * config.inner_square_side) →
  (config.rectangle_long_side / config.rectangle_short_side = 2) :=
by sorry

end rectangle_ratio_theorem_l3880_388026


namespace sqrt_increasing_l3880_388079

/-- The square root function is increasing on the non-negative real numbers. -/
theorem sqrt_increasing : ∀ x₁ x₂ : ℝ, 0 ≤ x₁ → 0 ≤ x₂ → x₁ < x₂ → Real.sqrt x₁ < Real.sqrt x₂ := by
  sorry

end sqrt_increasing_l3880_388079


namespace power_three_mod_eleven_l3880_388014

theorem power_three_mod_eleven : 3^2040 % 11 = 1 := by
  sorry

end power_three_mod_eleven_l3880_388014


namespace infinite_series_sum_l3880_388062

open Real

noncomputable def series_sum : ℝ := ∑' k : ℕ, (k : ℝ)^2 / 3^k

theorem infinite_series_sum : series_sum = 1/2 := by
  sorry

end infinite_series_sum_l3880_388062


namespace handshake_theorem_l3880_388061

/-- Represents a gathering of people where each person shakes hands with a fixed number of others. -/
structure Gathering where
  num_people : ℕ
  handshakes_per_person : ℕ

/-- Calculates the total number of handshakes in a gathering. -/
def total_handshakes (g : Gathering) : ℕ :=
  g.num_people * g.handshakes_per_person / 2

/-- Theorem stating that in a gathering of 30 people where each person shakes hands with 3 others,
    the total number of handshakes is 45. -/
theorem handshake_theorem (g : Gathering) (h1 : g.num_people = 30) (h2 : g.handshakes_per_person = 3) :
  total_handshakes g = 45 := by
  sorry

#eval total_handshakes ⟨30, 3⟩

end handshake_theorem_l3880_388061


namespace birds_and_storks_on_fence_l3880_388058

theorem birds_and_storks_on_fence (initial_birds : ℕ) (initial_storks : ℕ) (additional_storks : ℕ) : 
  initial_birds = 3 → initial_storks = 4 → additional_storks = 6 →
  initial_birds + initial_storks + additional_storks = 13 := by
sorry

end birds_and_storks_on_fence_l3880_388058


namespace remainder_problem_l3880_388031

theorem remainder_problem (N : ℤ) : 
  (∃ k : ℤ, N = 97 * k + 37) → N % 19 = 1 := by sorry

end remainder_problem_l3880_388031


namespace rectangle_diagonal_shortcut_l3880_388093

theorem rectangle_diagonal_shortcut (x y : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : x ≤ y) :
  x + y - Real.sqrt (x^2 + y^2) = (1/3) * y → x/y = 5/12 := by
  sorry

end rectangle_diagonal_shortcut_l3880_388093


namespace square_roots_of_625_l3880_388027

theorem square_roots_of_625 :
  (∃ x : ℝ, x > 0 ∧ x^2 = 625 ∧ x = 25) ∧
  (∀ x : ℝ, x^2 = 625 ↔ x = 25 ∨ x = -25) := by
  sorry

end square_roots_of_625_l3880_388027


namespace total_spent_l3880_388036

-- Define the amounts spent by each person
variable (A B C : ℝ)

-- Define the relationships between spending amounts
axiom alice_bella : A = (13/10) * B
axiom clara_bella : C = (4/5) * B
axiom alice_clara : A = C + 15

-- Theorem to prove
theorem total_spent : A + B + C = 93 := by
  sorry

end total_spent_l3880_388036


namespace women_picnic_attendance_l3880_388083

/-- Represents the percentage of employees in a company -/
structure CompanyPercentage where
  total : Real
  men : Real
  women : Real
  menAttended : Real
  womenAttended : Real
  totalAttended : Real

/-- Conditions for the company picnic attendance problem -/
def picnicConditions (c : CompanyPercentage) : Prop :=
  c.total = 100 ∧
  c.men = 50 ∧
  c.women = 50 ∧
  c.menAttended = 20 * c.men / 100 ∧
  c.totalAttended = 30.000000000000004 ∧
  c.womenAttended = c.totalAttended - c.menAttended

/-- Theorem stating that 40% of women attended the picnic -/
theorem women_picnic_attendance (c : CompanyPercentage) 
  (h : picnicConditions c) : c.womenAttended / c.women * 100 = 40 := by
  sorry


end women_picnic_attendance_l3880_388083


namespace spread_diluted_ecoli_correct_l3880_388051

/-- Represents different biological experimental procedures -/
inductive ExperimentalProcedure
  | SpreadDilutedEColi
  | IntroduceSterileAir
  | InoculateSoilLeachate
  | UseOpenRoseFlowers

/-- Represents the outcome of an experimental procedure -/
inductive ExperimentOutcome
  | Success
  | Failure

/-- Function that determines the outcome of a given experimental procedure -/
def experimentResult (procedure : ExperimentalProcedure) : ExperimentOutcome :=
  match procedure with
  | ExperimentalProcedure.SpreadDilutedEColi => ExperimentOutcome.Success
  | _ => ExperimentOutcome.Failure

/-- Theorem stating that spreading diluted E. coli culture is the correct method -/
theorem spread_diluted_ecoli_correct :
  ∀ (procedure : ExperimentalProcedure),
    experimentResult procedure = ExperimentOutcome.Success ↔
    procedure = ExperimentalProcedure.SpreadDilutedEColi :=
by
  sorry

#check spread_diluted_ecoli_correct

end spread_diluted_ecoli_correct_l3880_388051


namespace constant_product_rule_l3880_388057

theorem constant_product_rule (a b k : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : k > 0) :
  a * b = (k * a) * (b / k) :=
by sorry

end constant_product_rule_l3880_388057


namespace quadratic_roots_relation_l3880_388003

theorem quadratic_roots_relation (m n : ℝ) (r₁ r₂ : ℝ) (p q : ℝ) : 
  r₁^2 - 2*m*r₁ + n = 0 →
  r₂^2 - 2*m*r₂ + n = 0 →
  r₁^4 + p*r₁^4 + q = 0 →
  r₂^4 + p*r₂^4 + q = 0 →
  r₁ + r₂ = 2*m - 3 →
  p = -(2*m - 3)^4 + 4*n*(2*m - 3)^2 - 2*n^2 := by
sorry

end quadratic_roots_relation_l3880_388003


namespace count_prime_pairs_sum_50_l3880_388025

def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

def primePairSum50 (p q : ℕ) : Prop := isPrime p ∧ isPrime q ∧ p + q = 50

theorem count_prime_pairs_sum_50 : 
  ∃! (count : ℕ), ∃ (pairs : Finset (ℕ × ℕ)), 
    pairs.card = count ∧ 
    (∀ (p q : ℕ), (p, q) ∈ pairs ↔ primePairSum50 p q ∧ p ≤ q) ∧
    count = 4 :=
sorry

end count_prime_pairs_sum_50_l3880_388025


namespace trash_can_problem_l3880_388016

theorem trash_can_problem (x y a b : ℝ) : 
  3 * x + 4 * y = 580 →
  6 * x + 5 * y = 860 →
  a + b = 200 →
  60 * a + 100 * b ≤ 15000 →
  (x = 60 ∧ y = 100) ∧ a ≥ 125 := by sorry

end trash_can_problem_l3880_388016


namespace jessys_reading_plan_l3880_388005

/-- Jessy's reading plan problem -/
theorem jessys_reading_plan (total_pages : ℕ) (days : ℕ) (pages_per_session : ℕ) (additional_pages : ℕ)
  (h1 : total_pages = 140)
  (h2 : days = 7)
  (h3 : pages_per_session = 6)
  (h4 : additional_pages = 2) :
  ∃ (sessions : ℕ), sessions * pages_per_session * days + additional_pages * days = total_pages ∧ sessions = 3 :=
by sorry

end jessys_reading_plan_l3880_388005


namespace equal_volume_equal_capacity_container2_capacity_l3880_388074

/-- Represents a rectangular container -/
structure Container where
  height : ℝ
  width : ℝ
  length : ℝ
  capacity : ℝ

/-- Calculates the volume of a container -/
def volume (c : Container) : ℝ := c.height * c.width * c.length

/-- Theorem: Two containers with the same volume have the same capacity -/
theorem equal_volume_equal_capacity (c1 c2 : Container) 
  (h_volume : volume c1 = volume c2) 
  (h_capacity : c1.capacity = 80) : 
  c2.capacity = 80 := by
  sorry

/-- The first container -/
def container1 : Container := {
  height := 2,
  width := 3,
  length := 10,
  capacity := 80
}

/-- The second container -/
def container2 : Container := {
  height := 1,
  width := 3,
  length := 20,
  capacity := 80  -- We'll prove this
}

/-- Proof that container2 can hold 80 grams -/
theorem container2_capacity : container2.capacity = 80 := by
  apply equal_volume_equal_capacity container1 container2
  · -- Prove that volumes are equal
    simp [volume, container1, container2]
    -- 2 * 3 * 10 = 1 * 3 * 20
    ring
  · -- Show that container1's capacity is 80
    rfl

#check container2_capacity

end equal_volume_equal_capacity_container2_capacity_l3880_388074


namespace system_solution_l3880_388008

theorem system_solution (a b c d x y z u : ℝ) : 
  (a^3 * x + a^2 * y + a * z + u = 0) →
  (b^3 * x + b^2 * y + b * z + u = 0) →
  (c^3 * x + c^2 * y + c * z + u = 0) →
  (d^3 * x + d^2 * y + d * z + u = 1) →
  (x = 1 / ((d-a)*(d-b)*(d-c))) →
  (y = -(a+b+c) / ((d-a)*(d-b)*(d-c))) →
  (z = (a*b + b*c + c*a) / ((d-a)*(d-b)*(d-c))) →
  (u = -(a*b*c) / ((d-a)*(d-b)*(d-c))) →
  (a ≠ d) → (b ≠ d) → (c ≠ d) →
  (a^3 * x + a^2 * y + a * z + u = 0) ∧
  (b^3 * x + b^2 * y + b * z + u = 0) ∧
  (c^3 * x + c^2 * y + c * z + u = 0) ∧
  (d^3 * x + d^2 * y + d * z + u = 1) := by
  sorry

end system_solution_l3880_388008


namespace circle_packing_line_division_l3880_388001

/-- A circle in the coordinate plane --/
structure Circle where
  center : ℝ × ℝ
  diameter : ℝ

/-- The region formed by the union of circular regions --/
def Region (circles : List Circle) : Set (ℝ × ℝ) := sorry

/-- A line in the coordinate plane --/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Check if a line divides a region into two equal areas --/
def dividesEquallyArea (l : Line) (r : Set (ℝ × ℝ)) : Prop := sorry

/-- Express a line in the form ax = by + c --/
def lineToStandardForm (l : Line) : ℕ × ℕ × ℕ := sorry

/-- The greatest common divisor of three natural numbers --/
def gcd3 (a b c : ℕ) : ℕ := sorry

theorem circle_packing_line_division :
  ∀ (circles : List Circle) (l : Line),
    circles.length = 6 ∧
    (∀ c ∈ circles, c.diameter = 2 ∧ c.center.1 > 0 ∧ c.center.2 > 0) ∧
    l.slope = 2 ∧
    dividesEquallyArea l (Region circles) →
    let (a, b, c) := lineToStandardForm l
    gcd3 a b c = 1 →
    a^2 + b^2 + c^2 = 6 := by
  sorry

end circle_packing_line_division_l3880_388001


namespace geometric_sequence_ratio_l3880_388044

/-- A geometric sequence {a_n} satisfying certain conditions -/
structure GeometricSequence where
  a : ℕ → ℝ
  is_geometric : ∀ n : ℕ, a (n + 1) / a n = a 2 / a 1
  condition1 : a 5 * a 8 = 6
  condition2 : a 3 + a 10 = 5

/-- The ratio of a_20 to a_13 in the geometric sequence is either 3/2 or 2/3 -/
theorem geometric_sequence_ratio (seq : GeometricSequence) :
  seq.a 20 / seq.a 13 = 3/2 ∨ seq.a 20 / seq.a 13 = 2/3 :=
by sorry

end geometric_sequence_ratio_l3880_388044


namespace keith_score_l3880_388028

theorem keith_score (keith larry danny : ℕ) 
  (larry_score : larry = 3 * keith)
  (danny_score : danny = larry + 5)
  (total_score : keith + larry + danny = 26) :
  keith = 3 := by
sorry

end keith_score_l3880_388028


namespace fabian_walking_speed_l3880_388091

theorem fabian_walking_speed (initial_hours : ℕ) (additional_hours : ℕ) (total_distance : ℕ) :
  initial_hours = 3 →
  additional_hours = 3 →
  total_distance = 30 →
  (initial_hours + additional_hours) * (total_distance / (initial_hours + additional_hours)) = total_distance :=
by sorry

end fabian_walking_speed_l3880_388091


namespace parabola_y_axis_intersection_l3880_388009

/-- The parabola y = x^2 - 4 intersects the y-axis at the point (0, -4) -/
theorem parabola_y_axis_intersection :
  let f : ℝ → ℝ := λ x ↦ x^2 - 4
  ∃! p : ℝ × ℝ, p.1 = 0 ∧ p.2 = f p.1 ∧ p = (0, -4) :=
by sorry

end parabola_y_axis_intersection_l3880_388009


namespace cyclic_inequality_l3880_388004

theorem cyclic_inequality (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  x / (y * z) + y / (z * x) + z / (x * y) ≥ 1 / x + 1 / y + 1 / z := by
  sorry

end cyclic_inequality_l3880_388004


namespace rajan_share_is_2400_l3880_388096

/-- Calculates the share of profit for a partner in a business based on investments and durations. -/
def calculate_share (rajan_investment : ℕ) (rajan_duration : ℕ) 
                    (rakesh_investment : ℕ) (rakesh_duration : ℕ)
                    (mukesh_investment : ℕ) (mukesh_duration : ℕ)
                    (total_profit : ℕ) : ℕ :=
  let rajan_product := rajan_investment * rajan_duration
  let rakesh_product := rakesh_investment * rakesh_duration
  let mukesh_product := mukesh_investment * mukesh_duration
  let total_product := rajan_product + rakesh_product + mukesh_product
  (rajan_product * total_profit) / total_product

/-- Theorem stating that Rajan's share of the profit is 2400 given the specified investments and durations. -/
theorem rajan_share_is_2400 :
  calculate_share 20000 12 25000 4 15000 8 4600 = 2400 := by
  sorry

end rajan_share_is_2400_l3880_388096


namespace magic_trick_always_succeeds_l3880_388033

/-- Represents a box in the magic trick setup -/
structure Box :=
  (index : Fin 13)

/-- Represents the state of the magic trick setup -/
structure MagicTrickSetup :=
  (boxes : Fin 13 → Box)
  (coin_boxes : Fin 2 → Box)
  (opened_box : Box)

/-- Represents the magician's strategy -/
structure MagicianStrategy :=
  (choose_boxes : MagicTrickSetup → Fin 4 → Box)

/-- Predicate to check if a strategy is successful -/
def is_successful_strategy (strategy : MagicianStrategy) : Prop :=
  ∀ (setup : MagicTrickSetup),
    ∃ (i j : Fin 4),
      strategy.choose_boxes setup i = setup.coin_boxes 0 ∧
      strategy.choose_boxes setup j = setup.coin_boxes 1

theorem magic_trick_always_succeeds :
  ∃ (strategy : MagicianStrategy), is_successful_strategy strategy := by
  sorry

end magic_trick_always_succeeds_l3880_388033


namespace quadratic_factorization_sum_l3880_388060

theorem quadratic_factorization_sum (d e f : ℤ) :
  (∀ x : ℝ, x^2 + 17*x + 72 = (x + d)*(x + e)) ∧
  (∀ x : ℝ, x^2 - 15*x + 54 = (x - e)*(x - f)) →
  d + e + f = 23 := by
sorry

end quadratic_factorization_sum_l3880_388060


namespace cubic_of_99999_l3880_388046

theorem cubic_of_99999 :
  let N : ℕ := 99999
  N^3 = 999970000299999 := by
sorry

end cubic_of_99999_l3880_388046


namespace probability_x_plus_y_le_5_l3880_388082

-- Define the rectangle
def rectangle : Set (ℝ × ℝ) :=
  {p | 0 ≤ p.1 ∧ p.1 ≤ 4 ∧ 0 ≤ p.2 ∧ p.2 ≤ 8}

-- Define the region where x + y ≤ 5
def region : Set (ℝ × ℝ) :=
  {p ∈ rectangle | p.1 + p.2 ≤ 5}

-- Define the measure (area) of the rectangle
noncomputable def rectangleArea : ℝ := 32

-- Define the measure (area) of the region
noncomputable def regionArea : ℝ := 12

-- Theorem statement
theorem probability_x_plus_y_le_5 :
  (regionArea / rectangleArea : ℝ) = 3/8 :=
sorry

end probability_x_plus_y_le_5_l3880_388082
