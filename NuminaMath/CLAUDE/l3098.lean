import Mathlib

namespace expand_product_l3098_309897

theorem expand_product (x : ℝ) : 2 * (x - 3) * (x + 6) = 2 * x^2 + 6 * x - 36 := by
  sorry

end expand_product_l3098_309897


namespace partition_equivalence_l3098_309853

/-- Represents a partition of a positive integer -/
def Partition (n : ℕ) := Multiset ℕ

/-- The number of representations of n as a sum of distinct positive integers -/
def distinctSum (n : ℕ) : ℕ := sorry

/-- The number of representations of n as a sum of positive odd integers -/
def oddSum (n : ℕ) : ℕ := sorry

/-- The number of representations of n as a sum of positive integers, 
    where no term is repeated more than k-1 times -/
def limitedRepetitionSum (n k : ℕ) : ℕ := sorry

/-- The number of representations of n as a sum of positive integers, 
    where no term is divisible by k -/
def notDivisibleSum (n k : ℕ) : ℕ := sorry

/-- Main theorem stating the equality of representations -/
theorem partition_equivalence (n : ℕ) : 
  (∀ k : ℕ, k > 0 → limitedRepetitionSum n k = notDivisibleSum n k) ∧ 
  distinctSum n = oddSum n := by sorry

end partition_equivalence_l3098_309853


namespace binary_to_quaternary_conversion_l3098_309805

/-- Converts a binary (base 2) number to its decimal (base 10) representation -/
def binary_to_decimal (b : List Bool) : ℕ :=
  b.enum.foldl (fun acc (i, bit) => acc + if bit then 2^i else 0) 0

/-- Converts a decimal (base 10) number to its quaternary (base 4) representation -/
def decimal_to_quaternary (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
    let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc else aux (m / 4) ((m % 4) :: acc)
    aux n []

theorem binary_to_quaternary_conversion :
  let binary : List Bool := [true, false, true, false, true, true, false, true]
  let decimal : ℕ := binary_to_decimal binary
  let quaternary : List ℕ := decimal_to_quaternary decimal
  quaternary = [2, 2, 1, 3] :=
by sorry

end binary_to_quaternary_conversion_l3098_309805


namespace florist_initial_roses_l3098_309891

/-- Represents the number of roses picked in the first round -/
def first_pick : ℝ := 16.0

/-- Represents the number of roses picked in the second round -/
def second_pick : ℝ := 19.0

/-- Represents the total number of roses after all picking -/
def total_roses : ℕ := 72

/-- Calculates the initial number of roses the florist had -/
def initial_roses : ℝ := total_roses - (first_pick + second_pick)

/-- Theorem stating that the initial number of roses was 37 -/
theorem florist_initial_roses : initial_roses = 37 := by sorry

end florist_initial_roses_l3098_309891


namespace extremum_implies_zero_derivative_l3098_309855

-- Define a real-valued function
variable (f : ℝ → ℝ)

-- Define what it means for a point to be an extremum
def is_extremum (f : ℝ → ℝ) (x : ℝ) : Prop :=
  ∀ y, |y - x| < 1 → f y ≤ f x ∨ f y ≥ f x

-- State the theorem
theorem extremum_implies_zero_derivative (f : ℝ → ℝ) (x : ℝ) 
  (h1 : Differentiable ℝ f) 
  (h2 : is_extremum f x) : 
  deriv f x = 0 :=
sorry

end extremum_implies_zero_derivative_l3098_309855


namespace angle_conversion_l3098_309825

theorem angle_conversion (angle : Real) : ∃ (α k : Real), 
  angle * (π / 180) = α + 2 * k * π ∧ 
  0 ≤ α ∧ α < 2 * π ∧ 
  α = 7 * π / 4 ∧
  k = -10 := by
  sorry

end angle_conversion_l3098_309825


namespace square_nonnegative_l3098_309856

theorem square_nonnegative (x : ℚ) : x^2 ≥ 0 := by
  sorry

end square_nonnegative_l3098_309856


namespace max_c_value_l3098_309877

theorem max_c_value (c : ℝ) : 
  (∀ x y : ℝ, x > y ∧ y > 0 → x^2 - 2*y^2 ≤ c*x*(y-x)) → 
  c ≤ 2*Real.sqrt 2 - 4 :=
by sorry

end max_c_value_l3098_309877


namespace min_horseshoed_ponies_fraction_l3098_309826

/-- A ranch with horses and ponies -/
structure Ranch where
  horses : ℕ
  ponies : ℕ
  horseshoed_ponies : ℕ
  iceland_horseshoed_ponies : ℕ

/-- The conditions of the ranch problem -/
def ranch_conditions (r : Ranch) : Prop :=
  r.horses = r.ponies + 4 ∧
  r.horses + r.ponies ≥ 40 ∧
  r.iceland_horseshoed_ponies = (2 * r.horseshoed_ponies) / 3

/-- The theorem stating the minimum fraction of ponies with horseshoes -/
theorem min_horseshoed_ponies_fraction (r : Ranch) : 
  ranch_conditions r → r.horseshoed_ponies * 12 ≤ r.ponies := by
  sorry

#check min_horseshoed_ponies_fraction

end min_horseshoed_ponies_fraction_l3098_309826


namespace stick_length_4_forms_triangle_stick_length_1_cannot_form_triangle_stick_length_2_cannot_form_triangle_stick_length_3_cannot_form_triangle_l3098_309814

/-- Triangle inequality check function -/
def triangle_inequality (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

/-- Theorem: A stick of length 4 can form a triangle with sticks of lengths 3 and 6 -/
theorem stick_length_4_forms_triangle :
  triangle_inequality 3 6 4 :=
sorry

/-- Theorem: A stick of length 1 cannot form a triangle with sticks of lengths 3 and 6 -/
theorem stick_length_1_cannot_form_triangle :
  ¬ triangle_inequality 3 6 1 :=
sorry

/-- Theorem: A stick of length 2 cannot form a triangle with sticks of lengths 3 and 6 -/
theorem stick_length_2_cannot_form_triangle :
  ¬ triangle_inequality 3 6 2 :=
sorry

/-- Theorem: A stick of length 3 cannot form a triangle with sticks of lengths 3 and 6 -/
theorem stick_length_3_cannot_form_triangle :
  ¬ triangle_inequality 3 6 3 :=
sorry

end stick_length_4_forms_triangle_stick_length_1_cannot_form_triangle_stick_length_2_cannot_form_triangle_stick_length_3_cannot_form_triangle_l3098_309814


namespace terminal_side_angle_l3098_309893

/-- Given a point P(-4,3) on the terminal side of angle θ, prove that 2sin θ + cos θ = 2/5 -/
theorem terminal_side_angle (θ : ℝ) : 
  let P : ℝ × ℝ := (-4, 3)
  (P.1 = -4 ∧ P.2 = 3) →  -- Point P(-4,3)
  (P.1 = Real.cos θ * Real.sqrt (P.1^2 + P.2^2) ∧ 
   P.2 = Real.sin θ * Real.sqrt (P.1^2 + P.2^2)) →  -- P is on the terminal side of θ
  2 * Real.sin θ + Real.cos θ = 2/5 := by
  sorry

end terminal_side_angle_l3098_309893


namespace billiard_ball_weight_l3098_309865

/-- Given a box containing 6 equally weighted billiard balls, where the total weight
    of the box with balls is 1.82 kg and the empty box weighs 0.5 kg,
    prove that the weight of one billiard ball is 0.22 kg. -/
theorem billiard_ball_weight
  (num_balls : ℕ)
  (total_weight : ℝ)
  (empty_box_weight : ℝ)
  (h1 : num_balls = 6)
  (h2 : total_weight = 1.82)
  (h3 : empty_box_weight = 0.5) :
  (total_weight - empty_box_weight) / num_balls = 0.22 := by
  sorry

#eval (1.82 - 0.5) / 6

end billiard_ball_weight_l3098_309865


namespace second_to_third_quadrant_l3098_309830

-- Define a point in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Define the quadrants
def isSecondQuadrant (p : Point2D) : Prop := p.x < 0 ∧ p.y > 0
def isThirdQuadrant (p : Point2D) : Prop := p.x < 0 ∧ p.y < 0

-- Define the transformation from P to Q
def transformPtoQ (p : Point2D) : Point2D :=
  { x := -p.y, y := p.x }

-- Theorem statement
theorem second_to_third_quadrant (a b : ℝ) :
  let p := Point2D.mk a b
  let q := transformPtoQ p
  isSecondQuadrant p → isThirdQuadrant q := by
  sorry

end second_to_third_quadrant_l3098_309830


namespace last_digit_of_largest_known_prime_l3098_309885

theorem last_digit_of_largest_known_prime (n : ℕ) : n = 216091 →
  (2^n - 1) % 10 = 7 := by sorry

end last_digit_of_largest_known_prime_l3098_309885


namespace first_half_total_score_l3098_309827

/-- Represents the score of a team in a basketball game -/
structure Score where
  quarter1 : ℚ
  quarter2 : ℚ
  quarter3 : ℚ
  quarter4 : ℚ

/-- The Eagles' score -/
def eagles : Score :=
  { quarter1 := 1/2,
    quarter2 := 1/2 * 2,
    quarter3 := 1/2 * 2^2,
    quarter4 := 1/2 * 2^3 }

/-- The Tigers' score -/
def tigers : Score :=
  { quarter1 := 5,
    quarter2 := 5,
    quarter3 := 5,
    quarter4 := 5 }

/-- Total score for a team -/
def totalScore (s : Score) : ℚ :=
  s.quarter1 + s.quarter2 + s.quarter3 + s.quarter4

/-- First half score for a team -/
def firstHalfScore (s : Score) : ℚ :=
  s.quarter1 + s.quarter2

/-- Theorem stating the total first half score -/
theorem first_half_total_score :
  ⌈firstHalfScore eagles⌉ + ⌈firstHalfScore tigers⌉ = 19 ∧
  eagles.quarter1 = tigers.quarter1 ∧
  totalScore eagles = totalScore tigers + 2 ∧
  totalScore eagles ≤ 100 ∧
  totalScore tigers ≤ 100 :=
sorry


end first_half_total_score_l3098_309827


namespace right_triangle_hypotenuse_segment_ratio_l3098_309864

theorem right_triangle_hypotenuse_segment_ratio 
  (A B C D : ℝ × ℝ) 
  (right_angle : (B.1 - A.1) * (C.1 - A.1) + (B.2 - A.2) * (C.2 - A.2) = 0) 
  (leg_ratio : Real.sqrt ((B.1 - C.1)^2 + (B.2 - C.2)^2) = 
               (1/2) * Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)) 
  (D_on_AC : ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ 
             D = (t * A.1 + (1 - t) * C.1, t * A.2 + (1 - t) * C.2)) 
  (D_perpendicular : (B.1 - D.1) * (C.1 - A.1) + (B.2 - D.2) * (C.2 - A.2) = 0) : 
  Real.sqrt ((C.1 - D.1)^2 + (C.2 - D.2)^2) = 
  4 * Real.sqrt ((A.1 - D.1)^2 + (A.2 - D.2)^2) := by
sorry

end right_triangle_hypotenuse_segment_ratio_l3098_309864


namespace two_consecutive_increases_l3098_309828

theorem two_consecutive_increases (initial : ℝ) (increase1 : ℝ) (increase2 : ℝ) : 
  let after_first_increase := initial * (1 + increase1 / 100)
  let final_number := after_first_increase * (1 + increase2 / 100)
  initial = 1256 ∧ increase1 = 325 ∧ increase2 = 147 → final_number = 6000.54 := by
sorry

end two_consecutive_increases_l3098_309828


namespace holly_weekly_pill_count_l3098_309832

/-- Calculates the total number of pills Holly takes in a week -/
def total_weekly_pills : ℕ :=
  let insulin_daily := 2
  let bp_daily := 3
  let anticonvulsant_daily := 2 * bp_daily
  let calcium_every_other_day := 3 * insulin_daily
  let vitamin_d_twice_weekly := 4
  let multivitamin_thrice_weekly := 1
  let anxiety_sunday := 3 * bp_daily

  7 * insulin_daily + 
  7 * bp_daily + 
  7 * anticonvulsant_daily + 
  (7 / 2) * calcium_every_other_day +
  2 * vitamin_d_twice_weekly + 
  3 * multivitamin_thrice_weekly + 
  anxiety_sunday

theorem holly_weekly_pill_count : total_weekly_pills = 118 := by
  sorry

end holly_weekly_pill_count_l3098_309832


namespace dress_price_ratio_l3098_309896

theorem dress_price_ratio (marked_price : ℝ) (marked_price_pos : marked_price > 0) : 
  let discount_rate : ℝ := 1/4
  let selling_price : ℝ := marked_price * (1 - discount_rate)
  let cost_price : ℝ := 2/3 * selling_price
  cost_price / marked_price = 1/2 := by
sorry

end dress_price_ratio_l3098_309896


namespace plane_from_three_points_l3098_309868

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- A plane in 3D space -/
structure Plane3D where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- Three points are non-collinear if they do not lie on the same line -/
def NonCollinear (p1 p2 p3 : Point3D) : Prop :=
  ¬∃ (t : ℝ), p3 = Point3D.mk (p1.x + t * (p2.x - p1.x)) (p1.y + t * (p2.y - p1.y)) (p1.z + t * (p2.z - p1.z))

/-- A plane is uniquely determined by three non-collinear points -/
theorem plane_from_three_points (p1 p2 p3 : Point3D) (h : NonCollinear p1 p2 p3) :
  ∃! (plane : Plane3D), (plane.a * p1.x + plane.b * p1.y + plane.c * p1.z + plane.d = 0) ∧
                        (plane.a * p2.x + plane.b * p2.y + plane.c * p2.z + plane.d = 0) ∧
                        (plane.a * p3.x + plane.b * p3.y + plane.c * p3.z + plane.d = 0) :=
by sorry

end plane_from_three_points_l3098_309868


namespace probability_of_selection_for_student_survey_l3098_309819

/-- Represents a simple random sampling without replacement -/
structure SimpleRandomSampling where
  population : ℕ
  sample_size : ℕ
  h_sample_size_le_population : sample_size ≤ population

/-- The probability of a specific item being selected in a simple random sampling without replacement -/
def probability_of_selection (srs : SimpleRandomSampling) : ℚ :=
  srs.sample_size / srs.population

theorem probability_of_selection_for_student_survey :
  let srs : SimpleRandomSampling := {
    population := 303,
    sample_size := 50,
    h_sample_size_le_population := by sorry
  }
  probability_of_selection srs = 50 / 303 := by sorry

end probability_of_selection_for_student_survey_l3098_309819


namespace initial_value_problem_l3098_309802

theorem initial_value_problem (x : ℤ) : x + 335 = 456 * (x + 335) / 456 → x = 121 :=
by sorry

end initial_value_problem_l3098_309802


namespace min_product_of_three_exists_min_product_l3098_309831

def S : Set Int := {-10, -8, -5, -3, 0, 4, 6}

theorem min_product_of_three (a b c : Int) (ha : a ∈ S) (hb : b ∈ S) (hc : c ∈ S) 
  (hab : a ≠ b) (hbc : b ≠ c) (hca : c ≠ a) :
  a * b * c ≥ -240 :=
sorry

theorem exists_min_product :
  ∃ (a b c : Int), a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ a ≠ b ∧ b ≠ c ∧ c ≠ a ∧ a * b * c = -240 :=
sorry

end min_product_of_three_exists_min_product_l3098_309831


namespace pizza_sharing_l3098_309816

theorem pizza_sharing (total_slices : ℕ) (buzz_ratio waiter_ratio : ℕ) 
  (h1 : total_slices = 78)
  (h2 : buzz_ratio = 5)
  (h3 : waiter_ratio = 8) :
  waiter_ratio * (total_slices / (buzz_ratio + waiter_ratio)) - 20 = 28 :=
by sorry

end pizza_sharing_l3098_309816


namespace officer_average_salary_l3098_309886

/-- Proves that the average salary of officers is 440 Rs/month -/
theorem officer_average_salary
  (total_avg : ℝ)
  (non_officer_avg : ℝ)
  (officer_count : ℕ)
  (non_officer_count : ℕ)
  (h_total_avg : total_avg = 120)
  (h_non_officer_avg : non_officer_avg = 110)
  (h_officer_count : officer_count = 15)
  (h_non_officer_count : non_officer_count = 480) :
  (total_avg * (officer_count + non_officer_count) - non_officer_avg * non_officer_count) / officer_count = 440 :=
by sorry

end officer_average_salary_l3098_309886


namespace bike_ride_distance_l3098_309858

/-- Calculates the total distance of a 3-hour bike ride given the conditions -/
theorem bike_ride_distance (second_hour : ℝ) 
  (h1 : second_hour = 12)
  (h2 : second_hour = 1.2 * (second_hour / 1.2))
  (h3 : second_hour * 1.25 = 15) : 
  (second_hour / 1.2) + second_hour + (second_hour * 1.25) = 37 := by
  sorry

end bike_ride_distance_l3098_309858


namespace prob_different_subjects_is_one_sixth_l3098_309811

/-- The number of subjects available for selection -/
def num_subjects : ℕ := 4

/-- The number of subjects each student selects -/
def subjects_per_student : ℕ := 2

/-- The total number of possible subject selection combinations for one student -/
def total_combinations : ℕ := (num_subjects.choose subjects_per_student)

/-- The total number of possible events (combinations for both students) -/
def total_events : ℕ := total_combinations * total_combinations

/-- The number of events where both students select different subjects -/
def different_subjects_events : ℕ := total_combinations * ((num_subjects - subjects_per_student).choose subjects_per_student)

/-- The probability that the two students select different subjects -/
def prob_different_subjects : ℚ := different_subjects_events / total_events

theorem prob_different_subjects_is_one_sixth : 
  prob_different_subjects = 1 / 6 := by sorry

end prob_different_subjects_is_one_sixth_l3098_309811


namespace trig_equation_solution_l3098_309849

open Real

theorem trig_equation_solution (n : ℤ) : 
  let x : ℝ := π / 6 * (3 * ↑n + 1)
  tan (2 * x) * sin (2 * x) - 3 * sqrt 3 * (1 / tan (2 * x)) * cos (2 * x) = 0 := by
  sorry

end trig_equation_solution_l3098_309849


namespace museum_entrance_cost_l3098_309874

/-- The total cost of entrance tickets for a group of students and teachers -/
def total_cost (num_students : ℕ) (num_teachers : ℕ) (ticket_price : ℕ) : ℕ :=
  (num_students + num_teachers) * ticket_price

/-- Theorem: The total cost for 20 students and 3 teachers with $5 tickets is $115 -/
theorem museum_entrance_cost :
  total_cost 20 3 5 = 115 := by
  sorry

end museum_entrance_cost_l3098_309874


namespace spinner_probability_l3098_309852

theorem spinner_probability (P : Finset (Fin 4) → ℚ) 
  (h_total : P {0, 1, 2, 3} = 1)
  (h_A : P {0} = 1/4)
  (h_B : P {1} = 1/3)
  (h_D : P {3} = 1/6) :
  P {2} = 1/4 := by
  sorry

end spinner_probability_l3098_309852


namespace solve_inequality_l3098_309804

theorem solve_inequality (x : ℝ) : 
  (x + 5) / 2 - 1 < (3 * x + 2) / 2 ↔ x > 1 :=
by sorry

end solve_inequality_l3098_309804


namespace prob_three_consecutive_in_ten_l3098_309829

/-- The number of ways to arrange n items -/
def arrangements (n : ℕ) : ℕ := n.factorial

/-- The number of ways to arrange n items with a block of k consecutive items -/
def arrangements_with_block (n k : ℕ) : ℕ := (n - k + 1) * k.factorial * (n - k).factorial

/-- The probability of k specific items being consecutive in a random arrangement of n items -/
def prob_consecutive (n k : ℕ) : ℚ :=
  (arrangements_with_block n k : ℚ) / (arrangements n : ℚ)

theorem prob_three_consecutive_in_ten :
  prob_consecutive 10 3 = 1 / 15 := by sorry

end prob_three_consecutive_in_ten_l3098_309829


namespace election_votes_calculation_l3098_309880

/-- The total number of votes in the election -/
def total_votes : ℕ := 560000

/-- The percentage of valid votes that candidate A received -/
def candidate_A_percentage : ℚ := 65 / 100

/-- The percentage of invalid votes -/
def invalid_votes_percentage : ℚ := 15 / 100

/-- The number of valid votes for candidate A -/
def candidate_A_valid_votes : ℕ := 309400

theorem election_votes_calculation :
  (1 - invalid_votes_percentage) * candidate_A_percentage * total_votes = candidate_A_valid_votes :=
by sorry

end election_votes_calculation_l3098_309880


namespace billy_sleep_problem_l3098_309854

theorem billy_sleep_problem (x : ℝ) : 
  x + (x + 2) + (x + 2) / 2 + 3 * ((x + 2) / 2) = 30 → x = 6 := by
sorry

end billy_sleep_problem_l3098_309854


namespace jesse_stamp_ratio_l3098_309847

theorem jesse_stamp_ratio :
  let total_stamps : ℕ := 444
  let european_stamps : ℕ := 333
  let asian_stamps : ℕ := total_stamps - european_stamps
  (european_stamps : ℚ) / (asian_stamps : ℚ) = 3 / 1 :=
by sorry

end jesse_stamp_ratio_l3098_309847


namespace broken_line_intersections_l3098_309838

/-- A broken line is represented as a list of points in the plane -/
def BrokenLine := List (Real × Real)

/-- The length of a broken line -/
def length (bl : BrokenLine) : Real :=
  sorry

/-- Checks if a broken line is inside the unit square -/
def isInsideUnitSquare (bl : BrokenLine) : Prop :=
  sorry

/-- Counts the number of intersections between a broken line and a line parallel to the x-axis -/
def intersectionsWithHorizontalLine (bl : BrokenLine) (y : Real) : Nat :=
  sorry

/-- Counts the number of intersections between a broken line and a line parallel to the y-axis -/
def intersectionsWithVerticalLine (bl : BrokenLine) (x : Real) : Nat :=
  sorry

/-- The main theorem -/
theorem broken_line_intersections (bl : BrokenLine) 
  (h1 : length bl = 1000)
  (h2 : isInsideUnitSquare bl) :
  (∃ y : Real, y ∈ Set.Icc 0 1 ∧ intersectionsWithHorizontalLine bl y ≥ 500) ∨
  (∃ x : Real, x ∈ Set.Icc 0 1 ∧ intersectionsWithVerticalLine bl x ≥ 500) :=
sorry

end broken_line_intersections_l3098_309838


namespace two_intersecting_lines_l3098_309835

/-- A parabola defined by the equation y^2 = 8x -/
def parabola (x y : ℝ) : Prop := y^2 = 8*x

/-- A point (2,4) that lies on the parabola -/
def point : ℝ × ℝ := (2, 4)

/-- A function that returns the number of lines intersecting the parabola at exactly one point -/
def num_intersecting_lines : ℕ := 2

/-- Theorem stating that there are exactly two lines intersecting the parabola at one point -/
theorem two_intersecting_lines :
  parabola point.1 point.2 ∧ num_intersecting_lines = 2 :=
sorry

end two_intersecting_lines_l3098_309835


namespace b_current_age_l3098_309822

/-- Given two people A and B, proves that B's current age is 39 years
    under the given conditions. -/
theorem b_current_age (a b : ℕ) : 
  (a + 10 = 2 * (b - 10)) →  -- A's age in 10 years equals twice B's age from 10 years ago
  (a = b + 9) →              -- A is currently 9 years older than B
  b = 39 :=                  -- B's current age is 39 years
by sorry

end b_current_age_l3098_309822


namespace total_unbroken_seashells_is_17_l3098_309867

/-- The number of unbroken seashells Tom found over three days -/
def total_unbroken_seashells : ℕ :=
  let day1_total := 7
  let day1_broken := 4
  let day2_total := 12
  let day2_broken := 5
  let day3_total := 15
  let day3_broken := 8
  (day1_total - day1_broken) + (day2_total - day2_broken) + (day3_total - day3_broken)

/-- Theorem stating that the total number of unbroken seashells is 17 -/
theorem total_unbroken_seashells_is_17 : total_unbroken_seashells = 17 := by
  sorry

end total_unbroken_seashells_is_17_l3098_309867


namespace root_sum_theorem_l3098_309887

/-- The equation from the original problem -/
def equation (x : ℝ) : Prop :=
  1/x + 1/(x + 3) - 1/(x + 6) - 1/(x + 9) - 1/(x + 12) - 1/(x + 15) + 1/(x + 18) + 1/(x + 21) = 0

/-- Definition of the root form -/
def root_form (a b c d : ℝ) (x : ℝ) : Prop :=
  (x = -a + Real.sqrt (b + c * Real.sqrt d)) ∨ (x = -a + Real.sqrt (b - c * Real.sqrt d)) ∨
  (x = -a - Real.sqrt (b + c * Real.sqrt d)) ∨ (x = -a - Real.sqrt (b - c * Real.sqrt d))

/-- d is not divisible by the square of a prime -/
def not_divisible_by_prime_square (d : ℕ) : Prop :=
  ∀ p : ℕ, Prime p → ¬(p^2 ∣ d)

theorem root_sum_theorem (a b c : ℝ) (d : ℕ) :
  (∃ x : ℝ, equation x ∧ root_form a b c (d : ℝ) x) →
  not_divisible_by_prime_square d →
  a + b + c + d = 57.5 := by
  sorry

end root_sum_theorem_l3098_309887


namespace unique_integral_solution_l3098_309846

theorem unique_integral_solution (x y z n : ℤ) 
  (eq1 : x * y + y * z + z * x = 3 * n^2 - 1)
  (eq2 : x + y + z = 3 * n)
  (h1 : x ≥ y)
  (h2 : y ≥ z) :
  x = n + 1 ∧ y = n ∧ z = n - 1 :=
by sorry

end unique_integral_solution_l3098_309846


namespace quadratic_inequality_requires_conditional_branch_l3098_309813

/-- Represents an algorithm --/
inductive Algorithm
  | ProductOfTwoNumbers
  | DistancePointToLine
  | QuadraticInequality
  | TrapezoidArea

/-- Determines if an algorithm requires a conditional branch structure --/
def requires_conditional_branch (a : Algorithm) : Prop :=
  match a with
  | Algorithm.QuadraticInequality => True
  | _ => False

/-- Theorem stating that only solving a quadratic inequality requires a conditional branch structure --/
theorem quadratic_inequality_requires_conditional_branch :
  ∀ (a : Algorithm), requires_conditional_branch a ↔ a = Algorithm.QuadraticInequality :=
by sorry

end quadratic_inequality_requires_conditional_branch_l3098_309813


namespace extremum_point_implies_inequality_non_negative_function_implies_m_range_l3098_309850

noncomputable section

variable (m : ℝ)
def f (x : ℝ) : ℝ := Real.exp (x + m) - Real.log x

def a : ℝ := Real.exp (1 / Real.exp 1)

theorem extremum_point_implies_inequality :
  (∃ (m : ℝ), f m 1 = 0 ∧ (∀ (x : ℝ), x > 0 → f m x ≥ f m 1)) →
  ∀ (x : ℝ), x > 0 → Real.exp x - Real.exp 1 * Real.log x ≥ Real.exp 1 :=
sorry

theorem non_negative_function_implies_m_range :
  (∃ (x₀ : ℝ), x₀ > 0 ∧ (∀ (x : ℝ), x > 0 → f m x ≥ f m x₀)) →
  (∀ (x : ℝ), x > 0 → f m x ≥ 0) →
  m ≥ -a - Real.log a :=
sorry

end

end extremum_point_implies_inequality_non_negative_function_implies_m_range_l3098_309850


namespace zeros_of_f_l3098_309821

-- Define the function f(x)
def f (x : ℝ) : ℝ := x^3 - 2*x^2 - x + 2

-- Theorem statement
theorem zeros_of_f :
  ∃ (a b c : ℝ), (a = -1 ∧ b = 1 ∧ c = 2) ∧
  (∀ x : ℝ, f x = 0 ↔ x = a ∨ x = b ∨ x = c) :=
sorry

end zeros_of_f_l3098_309821


namespace new_arithmetic_mean_l3098_309895

def original_set_size : ℕ := 60
def original_mean : ℚ := 42
def removed_numbers : List ℚ := [50, 60, 70]

theorem new_arithmetic_mean :
  let original_sum : ℚ := original_mean * original_set_size
  let removed_sum : ℚ := removed_numbers.sum
  let new_sum : ℚ := original_sum - removed_sum
  let new_set_size : ℕ := original_set_size - removed_numbers.length
  (new_sum / new_set_size : ℚ) = 41 := by sorry

end new_arithmetic_mean_l3098_309895


namespace prob_even_card_l3098_309836

/-- The probability of drawing a card with an even number from a set of cards -/
theorem prob_even_card (total_cards : ℕ) (even_cards : ℕ) 
  (h1 : total_cards = 6) 
  (h2 : even_cards = 3) : 
  (even_cards : ℚ) / total_cards = 1 / 2 := by
  sorry

#check prob_even_card

end prob_even_card_l3098_309836


namespace parallel_vectors_m_value_l3098_309883

def vector_a (m : ℝ) : ℝ × ℝ := (3, m)
def vector_b : ℝ × ℝ := (2, -4)

def parallel (v w : ℝ × ℝ) : Prop :=
  ∃ (k : ℝ), v = (k * w.1, k * w.2)

theorem parallel_vectors_m_value :
  ∀ m : ℝ, parallel (vector_a m) vector_b → m = -6 :=
by sorry

end parallel_vectors_m_value_l3098_309883


namespace problem_solution_l3098_309871

theorem problem_solution (x y : ℝ) 
  (hx : x = Real.sqrt 5 + Real.sqrt 3) 
  (hy : y = Real.sqrt 5 - Real.sqrt 3) : 
  (x^2 + 2*x*y + y^2) / (x^2 - y^2) = Real.sqrt 15 / 3 ∧ 
  Real.sqrt (x^2 + y^2 - 3) = Real.sqrt 13 ∨ 
  Real.sqrt (x^2 + y^2 - 3) = -Real.sqrt 13 := by
  sorry

end problem_solution_l3098_309871


namespace clock_adjustment_theorem_l3098_309810

/-- Represents the gain of the clock in minutes per day -/
def clock_gain : ℚ := 13/4

/-- Represents the number of days between May 1st 10 A.M. and May 10th 2 P.M. -/
def days : ℚ := 9 + 4/24

/-- Calculates the adjustment needed for the clock -/
def adjustment (gain : ℚ) (time : ℚ) : ℚ := gain * time

/-- Theorem stating that the adjustment is approximately 29.8 minutes -/
theorem clock_adjustment_theorem :
  ∃ ε > 0, abs (adjustment clock_gain days - 29.8) < ε :=
sorry

end clock_adjustment_theorem_l3098_309810


namespace sum_of_a_and_b_l3098_309875

theorem sum_of_a_and_b (a b : ℝ) 
  (h1 : (a + Real.sqrt b) + (a - Real.sqrt b) = -6)
  (h2 : (a + Real.sqrt b) * (a - Real.sqrt b) = 4) :
  a + b = 2 := by
sorry

end sum_of_a_and_b_l3098_309875


namespace smallest_n_perfect_square_and_cube_l3098_309820

theorem smallest_n_perfect_square_and_cube : ∃ (n : ℕ), 
  (n > 0) ∧ 
  (∃ (k : ℕ), 5 * n = k^2) ∧ 
  (∃ (m : ℕ), 4 * n = m^3) ∧
  (∀ (x : ℕ), x > 0 → (∃ (y : ℕ), 5 * x = y^2) → (∃ (z : ℕ), 4 * x = z^3) → x ≥ n) ∧
  n = 625000 :=
by sorry

end smallest_n_perfect_square_and_cube_l3098_309820


namespace rectangles_count_l3098_309882

/-- The number of rectangles in an n×n square grid -/
def rectangles_in_square (n : ℕ) : ℕ := (n * (n + 1) / 2) ^ 2

/-- The number of rectangles in the given arrangement of three n×n square grids -/
def rectangles_in_three_squares (n : ℕ) : ℕ := 
  n^2 * (2*n + 1)^2 - n^4 - n^3*(n + 1) - (n * (n + 1) / 2)^2

theorem rectangles_count (n : ℕ) (h : n > 0) : 
  (rectangles_in_square n = (n * (n + 1) / 2) ^ 2) ∧ 
  (rectangles_in_three_squares n = n^2 * (2*n + 1)^2 - n^4 - n^3*(n + 1) - (n * (n + 1) / 2)^2) := by
  sorry

end rectangles_count_l3098_309882


namespace latoya_card_credit_l3098_309872

/-- Calculates the remaining credit on a prepaid phone card after a call -/
def remaining_credit (initial_value : ℚ) (cost_per_minute : ℚ) (call_duration : ℕ) : ℚ :=
  initial_value - (cost_per_minute * call_duration)

/-- Theorem stating the remaining credit on Latoya's prepaid phone card -/
theorem latoya_card_credit :
  let initial_value : ℚ := 30
  let cost_per_minute : ℚ := 16 / 100
  let call_duration : ℕ := 22
  remaining_credit initial_value cost_per_minute call_duration = 2648 / 100 := by
sorry

end latoya_card_credit_l3098_309872


namespace hyperbola_eccentricity_range_l3098_309818

/-- The eccentricity of a hyperbola defined by x²/(1+m) - y²/(1-m) = 1 with m > 0 is between 1 and √2 -/
theorem hyperbola_eccentricity_range (m : ℝ) (h1 : m > 0) :
  let C := {(x, y) : ℝ × ℝ | x^2 / (1 + m) - y^2 / (1 - m) = 1}
  let e := Real.sqrt 2 / Real.sqrt (1 + m)
  1 < e ∧ e < Real.sqrt 2 := by
sorry

end hyperbola_eccentricity_range_l3098_309818


namespace f_iter_has_two_roots_l3098_309848

def f (x : ℝ) : ℝ := x^2 + 2018*x + 1

def f_iter (n : ℕ) : ℝ → ℝ :=
  match n with
  | 0 => id
  | n+1 => f ∘ f_iter n

theorem f_iter_has_two_roots (n : ℕ+) : ∃ (x y : ℝ), x ≠ y ∧ f_iter n x = 0 ∧ f_iter n y = 0 := by
  sorry

end f_iter_has_two_roots_l3098_309848


namespace triangle_properties_l3098_309889

structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  sinA : ℝ
  sinB : ℝ
  sinC : ℝ
  cosC : ℝ

def is_valid_triangle (t : Triangle) : Prop :=
  t.sinA > 0 ∧ t.sinB > 0 ∧ t.sinC > 0 ∧
  t.a > 0 ∧ t.b > 0 ∧ t.c > 0 ∧
  t.a + t.b > t.c ∧ t.b + t.c > t.a ∧ t.c + t.a > t.b

theorem triangle_properties (t : Triangle) 
  (h_valid : is_valid_triangle t)
  (h_arith_seq : 2 * t.sinB = t.sinA + t.sinC)
  (h_cosC : t.cosC = 1/3) :
  (t.b / t.a = 10/9) ∧ 
  (t.c = 11 → t.a * t.b * t.sinC / 2 = 30 * Real.sqrt 2) :=
sorry

end triangle_properties_l3098_309889


namespace set_operations_l3098_309841

-- Define the sets A and B
def A : Set ℝ := {x | -5 ≤ x ∧ x ≤ 3}
def B : Set ℝ := {x | x < -2 ∨ x > 4}

-- State the theorem
theorem set_operations :
  (A ∩ B = {x | -5 ≤ x ∧ x < -2}) ∧
  ((Aᶜ ∪ Bᶜ) = {x | x < -5 ∨ x ≥ -2}) := by
  sorry

end set_operations_l3098_309841


namespace rockets_won_38_games_l3098_309892

/-- Represents the number of wins for each team -/
structure TeamWins where
  sharks : ℕ
  dolphins : ℕ
  rockets : ℕ
  wolves : ℕ
  comets : ℕ

/-- The set of possible win numbers -/
def winNumbers : Finset ℕ := {28, 33, 38, 43}

/-- The conditions of the problem -/
def validTeamWins (tw : TeamWins) : Prop :=
  tw.sharks > tw.dolphins ∧
  tw.rockets > tw.wolves ∧
  tw.comets > tw.rockets ∧
  tw.wolves > 25 ∧
  tw.sharks ∈ winNumbers ∧
  tw.dolphins ∈ winNumbers ∧
  tw.rockets ∈ winNumbers ∧
  tw.wolves ∈ winNumbers ∧
  tw.comets ∈ winNumbers

/-- Theorem: Given the conditions, the Rockets won 38 games -/
theorem rockets_won_38_games (tw : TeamWins) (h : validTeamWins tw) : tw.rockets = 38 := by
  sorry

end rockets_won_38_games_l3098_309892


namespace solution_set_when_a_is_3_range_of_a_for_empty_solution_l3098_309899

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |x - a| + |x - 1|

-- Part I
theorem solution_set_when_a_is_3 :
  {x : ℝ | f 3 x ≤ 4} = Set.Icc 0 4 := by sorry

-- Part II
theorem range_of_a_for_empty_solution :
  {a : ℝ | ∀ x, f a x ≥ 2} = Set.Iic (-1) ∪ Set.Ici 3 := by sorry

end solution_set_when_a_is_3_range_of_a_for_empty_solution_l3098_309899


namespace marble_pairs_l3098_309803

-- Define the set of marbles
def Marble : Type := 
  Sum (Fin 1) (Sum (Fin 1) (Sum (Fin 1) (Sum (Fin 3) (Fin 2))))

-- Define the function to count distinct pairs
def countDistinctPairs (s : Finset Marble) : ℕ := sorry

-- State the theorem
theorem marble_pairs : 
  let s : Finset Marble := sorry
  countDistinctPairs s = 12 := by sorry

end marble_pairs_l3098_309803


namespace midpoint_slope_l3098_309801

/-- The slope of the line containing the midpoints of two specific line segments is 1.5 -/
theorem midpoint_slope : 
  let midpoint1 := ((0 + 8) / 2, (0 + 6) / 2)
  let midpoint2 := ((5 + 5) / 2, (0 + 9) / 2)
  let slope := (midpoint2.2 - midpoint1.2) / (midpoint2.1 - midpoint1.1)
  slope = 1.5 := by sorry

end midpoint_slope_l3098_309801


namespace always_positive_l3098_309815

theorem always_positive (x y : ℝ) : x^2 - 4*x + y^2 + 13 > 0 := by
  sorry

end always_positive_l3098_309815


namespace circle_radius_largest_radius_l3098_309837

/-- A circle tangent to both x and y axes with center (r,r) passing through (9,2) has radius 17 or 5 -/
theorem circle_radius (r : ℝ) : 
  (r > 0) → 
  ((9 - r)^2 + (2 - r)^2 = r^2) → 
  (r = 17 ∨ r = 5) :=
by sorry

/-- The largest possible radius of a circle tangent to both x and y axes and passing through (9,2) is 17 -/
theorem largest_radius : 
  ∃ (r : ℝ), (r > 0) ∧ 
  ((9 - r)^2 + (2 - r)^2 = r^2) ∧ 
  (∀ (s : ℝ), (s > 0) ∧ ((9 - s)^2 + (2 - s)^2 = s^2) → s ≤ r) ∧
  r = 17 :=
by sorry

end circle_radius_largest_radius_l3098_309837


namespace triangle_sum_in_closed_shape_l3098_309845

theorem triangle_sum_in_closed_shape (n : ℕ) (C : ℝ) : 
  n > 0 → C = 3 * 360 - 180 := by
  sorry

end triangle_sum_in_closed_shape_l3098_309845


namespace petes_ten_dollar_bills_l3098_309879

theorem petes_ten_dollar_bills (
  total_owed : ℕ)
  (twenty_dollar_bills : ℕ)
  (bottle_refund : ℚ)
  (bottles_to_return : ℕ)
  (h1 : total_owed = 90)
  (h2 : twenty_dollar_bills = 2)
  (h3 : bottle_refund = 1/2)
  (h4 : bottles_to_return = 20)
  : ∃ (ten_dollar_bills : ℕ),
    ten_dollar_bills = 4 ∧
    20 * twenty_dollar_bills + 10 * ten_dollar_bills + (bottle_refund * bottles_to_return) = total_owed :=
by sorry

end petes_ten_dollar_bills_l3098_309879


namespace function_equality_proof_l3098_309839

theorem function_equality_proof (f : ℝ → ℝ) 
  (h₁ : ∀ x, x > 0 → f x > 0)
  (h₂ : ∀ x₁ x₂, x₁ > 0 → x₂ > 0 → 
    |x₁ * f x₂ - x₂ * f x₁| = (f x₁ + f x₂) * (x₂ - x₁)) :
  ∃ c : ℝ, c > 0 ∧ ∀ x, x > 0 → f x = c / x :=
sorry

end function_equality_proof_l3098_309839


namespace magnitude_z_l3098_309862

open Complex

theorem magnitude_z (w z : ℂ) (h1 : w * z = 15 - 20 * I) (h2 : abs w = Real.sqrt 29) :
  abs z = (25 * Real.sqrt 29) / 29 := by
  sorry

end magnitude_z_l3098_309862


namespace not_divides_power_plus_one_l3098_309861

theorem not_divides_power_plus_one (n : ℕ) (h : n > 1) : ¬(2^n ∣ 3^n + 1) := by
  sorry

end not_divides_power_plus_one_l3098_309861


namespace gift_distribution_count_l3098_309842

/-- The number of bags of gifts -/
def num_bags : ℕ := 5

/-- The number of elderly people -/
def num_people : ℕ := 4

/-- The number of ways to distribute consecutive pairs -/
def consecutive_pairs : ℕ := 4

/-- The number of ways to arrange the remaining bags -/
def remaining_arrangements : ℕ := 24  -- This is A_4^4

/-- The total number of distribution methods -/
def total_distributions : ℕ := consecutive_pairs * remaining_arrangements

theorem gift_distribution_count :
  total_distributions = 96 :=
sorry

end gift_distribution_count_l3098_309842


namespace quadratic_inequality_range_l3098_309823

theorem quadratic_inequality_range (m : ℝ) : 
  (¬ ∀ x : ℝ, x^2 + 3*x + m > 0) ↔ m ≤ 9/4 :=
by sorry

end quadratic_inequality_range_l3098_309823


namespace smallest_product_of_factors_l3098_309866

def is_factor (a b : ℕ) : Prop := b % a = 0

theorem smallest_product_of_factors (x y : ℕ) : 
  x ≠ y →
  is_factor x 48 →
  is_factor y 48 →
  (Even x ∨ Even y) →
  ¬(is_factor (x * y) 48) →
  ∀ (a b : ℕ), a ≠ b ∧ is_factor a 48 ∧ is_factor b 48 ∧ (Even a ∨ Even b) ∧ ¬(is_factor (a * b) 48) →
  x * y ≤ a * b →
  x * y = 32 :=
sorry

end smallest_product_of_factors_l3098_309866


namespace f_derivative_positive_at_midpoint_l3098_309898

open Real

/-- The function f(x) = x^2 + 2x - a(ln x + x) -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x^2 + 2*x - a*(log x + x)

/-- The derivative of f(x) -/
noncomputable def f_derivative (a : ℝ) (x : ℝ) : ℝ := 2*x + 2 - a*(1/x + 1)

theorem f_derivative_positive_at_midpoint (a c : ℝ) (x₁ x₂ : ℝ) 
  (hx₁ : f a x₁ = c) (hx₂ : f a x₂ = c) (hne : x₁ ≠ x₂) :
  f_derivative a ((x₁ + x₂) / 2) > 0 :=
sorry

end f_derivative_positive_at_midpoint_l3098_309898


namespace mary_travel_time_l3098_309873

/-- The total time Mary spends from calling the Uber to the plane being ready for takeoff -/
def total_time (uber_to_house : ℕ) (bag_check : ℕ) (wait_for_boarding : ℕ) : ℕ :=
  let uber_to_airport := 5 * uber_to_house
  let security := 3 * bag_check
  let wait_for_takeoff := 2 * wait_for_boarding
  uber_to_house + uber_to_airport + bag_check + security + wait_for_boarding + wait_for_takeoff

/-- The theorem stating that Mary's total travel preparation time is 3 hours -/
theorem mary_travel_time :
  total_time 10 15 20 = 180 :=
sorry

end mary_travel_time_l3098_309873


namespace rabbit_travel_time_l3098_309894

/-- Proves that a rabbit running at a constant speed of 6 miles per hour will take 20 minutes to travel 2 miles. -/
theorem rabbit_travel_time :
  let rabbit_speed : ℝ := 6 -- miles per hour
  let distance : ℝ := 2 -- miles
  let time_in_hours : ℝ := distance / rabbit_speed
  let time_in_minutes : ℝ := time_in_hours * 60
  time_in_minutes = 20 := by sorry

end rabbit_travel_time_l3098_309894


namespace first_place_percentage_l3098_309834

/-- 
Given a pot of money where:
- 8 people each contribute $5
- Third place gets $4
- Second and third place split the remaining money after first place
Prove that first place gets 80% of the total money
-/
theorem first_place_percentage (total_people : Nat) (contribution : ℕ) (third_place_prize : ℕ) :
  total_people = 8 →
  contribution = 5 →
  third_place_prize = 4 →
  (((total_people * contribution - 2 * third_place_prize) : ℚ) / (total_people * contribution)) = 4/5 := by
  sorry

#check first_place_percentage

end first_place_percentage_l3098_309834


namespace triangle_inequality_l3098_309890

theorem triangle_inequality (a b c : ℝ) : 
  (0 < a ∧ 0 < b ∧ 0 < c) → (a + b > c ∧ b + c > a ∧ c + a > b) → 
  (3 = a ∧ 7 = b) → 4 < c ∧ c < 10 :=
by sorry

end triangle_inequality_l3098_309890


namespace girl_scout_cookies_l3098_309878

theorem girl_scout_cookies (boxes_per_case : ℕ) (boxes_sold : ℕ) (unpacked_boxes : ℕ) :
  boxes_per_case = 12 →
  boxes_sold > 0 →
  unpacked_boxes = 7 →
  ∃ n : ℕ, boxes_sold = 12 * n + 7 :=
by sorry

end girl_scout_cookies_l3098_309878


namespace center_trajectory_of_circle_family_l3098_309876

-- Define the family of circles
def circle_family (t x y : ℝ) : Prop :=
  x^2 + y^2 - 4*t*x - 2*t*y + 3*t^2 - 4 = 0

-- Define the trajectory of centers
def center_trajectory (t x y : ℝ) : Prop :=
  x = 2*t ∧ y = t

-- Theorem statement
theorem center_trajectory_of_circle_family :
  ∀ t : ℝ, ∃ x y : ℝ,
    circle_family t x y ↔ center_trajectory t x y :=
sorry

end center_trajectory_of_circle_family_l3098_309876


namespace scott_total_oranges_l3098_309824

/-- The number of boxes Scott has for oranges. -/
def num_boxes : ℕ := 8

/-- The number of oranges that must be in each box. -/
def oranges_per_box : ℕ := 7

/-- Theorem stating that Scott has 56 oranges in total. -/
theorem scott_total_oranges : num_boxes * oranges_per_box = 56 := by
  sorry

end scott_total_oranges_l3098_309824


namespace inequality_solution_l3098_309851

-- Define the real-valued functions f and g
variable (f g : ℝ → ℝ)

-- Define the conditions
axiom f_odd : ∀ x, f (-x) = -f x
axiom g_even : ∀ x, g (-x) = g x
axiom g_never_zero : ∀ x, g x ≠ 0
axiom condition_neg : ∀ x, x < 0 → f x * g x - f x * (deriv g x) > 0
axiom f_3_eq_0 : f 3 = 0

-- Define the solution set
def solution_set : Set ℝ := {x | x < -3 ∨ (0 < x ∧ x < 3)}

-- State the theorem
theorem inequality_solution :
  {x : ℝ | f x * g x < 0} = solution_set :=
sorry

end inequality_solution_l3098_309851


namespace minimum_employees_needed_l3098_309840

theorem minimum_employees_needed (forest_employees : ℕ) (marine_employees : ℕ) (both_employees : ℕ)
  (h1 : forest_employees = 95)
  (h2 : marine_employees = 80)
  (h3 : both_employees = 35)
  (h4 : both_employees ≤ forest_employees ∧ both_employees ≤ marine_employees) :
  forest_employees + marine_employees - both_employees = 140 :=
by sorry

end minimum_employees_needed_l3098_309840


namespace trajectory_of_P_l3098_309884

/-- The trajectory of point P given two fixed points F₁ and F₂ -/
theorem trajectory_of_P (F₁ F₂ P : ℝ × ℝ) : 
  F₁ = (-1, 0) →
  F₂ = (1, 0) →
  (dist P F₁ + dist P F₂) / 2 = dist F₁ F₂ / 2 →
  ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ P = (1 - t) • F₁ + t • F₂ :=
sorry

#check trajectory_of_P

end trajectory_of_P_l3098_309884


namespace train_crossing_time_l3098_309869

/-- Proves that a train with given length and speed takes the calculated time to cross a pole -/
theorem train_crossing_time (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) :
  train_length = 350 →
  train_speed_kmh = 60 →
  crossing_time = 21 →
  crossing_time = train_length / (train_speed_kmh * 1000 / 3600) := by
  sorry

#check train_crossing_time

end train_crossing_time_l3098_309869


namespace point_A_in_third_quadrant_l3098_309809

/-- The point A with coordinates (sin 2018°, tan 117°) is in the third quadrant -/
theorem point_A_in_third_quadrant :
  let x : ℝ := Real.sin (2018 * π / 180)
  let y : ℝ := Real.tan (117 * π / 180)
  x < 0 ∧ y < 0 := by sorry

end point_A_in_third_quadrant_l3098_309809


namespace quadratic_roots_and_equation_l3098_309859

theorem quadratic_roots_and_equation (x₁ x₂ a : ℝ) : 
  (x₁^2 + 4*x₁ - 3 = 0) →
  (x₂^2 + 4*x₂ - 3 = 0) →
  (2*x₁*(x₂^2 + 3*x₂ - 3) + a = 2) →
  (a = -4) := by
sorry

end quadratic_roots_and_equation_l3098_309859


namespace age_difference_l3098_309843

theorem age_difference (a b c d : ℕ) 
  (eq1 : a + b = b + c + 12)
  (eq2 : b + d = c + d + 8)
  (eq3 : d = a + 5) :
  c = a - 12 := by
sorry

end age_difference_l3098_309843


namespace least_number_with_given_remainders_l3098_309817

theorem least_number_with_given_remainders :
  ∃ (n : ℕ), n > 1 ∧ n % 25 = 1 ∧ n % 7 = 1 ∧
  ∀ (m : ℕ), m > 1 → m % 25 = 1 → m % 7 = 1 → n ≤ m :=
by
  use 176
  sorry

end least_number_with_given_remainders_l3098_309817


namespace hyperbola_other_asymptote_l3098_309888

/-- A hyperbola with given properties -/
structure Hyperbola where
  /-- One asymptote of the hyperbola -/
  asymptote1 : ℝ → ℝ
  /-- x-coordinate of the foci -/
  foci_x : ℝ
  /-- Condition that the first asymptote is y = 2x -/
  h_asymptote1 : ∀ x, asymptote1 x = 2 * x
  /-- Condition that the foci x-coordinate is 4 -/
  h_foci_x : foci_x = 4

/-- The other asymptote of the hyperbola -/
def other_asymptote (h : Hyperbola) : ℝ → ℝ :=
  fun x ↦ -2 * x + 16

/-- Theorem stating the equation of the other asymptote -/
theorem hyperbola_other_asymptote (h : Hyperbola) :
  other_asymptote h = fun x ↦ -2 * x + 16 := by
  sorry

end hyperbola_other_asymptote_l3098_309888


namespace apartment_rent_theorem_l3098_309806

/-- Calculates the total rent paid over a period of time with different monthly rates -/
def totalRent (months1 : ℕ) (rate1 : ℕ) (months2 : ℕ) (rate2 : ℕ) : ℕ :=
  months1 * rate1 + months2 * rate2

theorem apartment_rent_theorem :
  totalRent 36 300 24 350 = 19200 := by
  sorry

end apartment_rent_theorem_l3098_309806


namespace random_walk_exits_lawn_l3098_309833

/-- A random walk on a 2D plane -/
def RandomWalk2D := ℕ → ℝ × ℝ

/-- The origin (starting point) of the random walk -/
def origin : ℝ × ℝ := (0, 0)

/-- The radius of the circular lawn -/
def lawn_radius : ℝ := 100

/-- The length of each step in the random walk -/
def step_length : ℝ := 1

/-- The expected distance from the origin after n steps in a 2D random walk -/
noncomputable def expected_distance (n : ℕ) : ℝ := Real.sqrt (n : ℝ)

/-- Theorem: For a sufficiently large number of steps, the expected distance 
    from the origin in a 2D random walk exceeds the lawn radius -/
theorem random_walk_exits_lawn :
  ∃ N : ℕ, ∀ n : ℕ, n ≥ N → expected_distance n > lawn_radius :=
sorry

end random_walk_exits_lawn_l3098_309833


namespace sine_cosine_inequality_l3098_309808

theorem sine_cosine_inequality (c : ℝ) :
  (∀ x : ℝ, 3 * Real.sin x - 4 * Real.cos x + c > 0) ↔ c > 5 := by
  sorry

end sine_cosine_inequality_l3098_309808


namespace work_ratio_l3098_309860

/-- Given that A can finish a work in 18 days and A and B working together can finish 1/6 of the work in a day, 
    prove that the ratio of time taken by B to A to finish the work is 1:2 -/
theorem work_ratio (a_time : ℝ) (combined_rate : ℝ) 
  (ha : a_time = 18)
  (hc : combined_rate = 1/6) : 
  (a_time / 2) / a_time = 1/2 := by
  sorry

end work_ratio_l3098_309860


namespace arithmetic_sequence_ratio_l3098_309807

/-- An arithmetic sequence with a non-zero common difference -/
def ArithmeticSequence (a : ℕ → ℚ) (d : ℚ) : Prop :=
  d ≠ 0 ∧ ∀ n, a (n + 1) = a n + d

/-- Three terms of a sequence form a geometric sequence -/
def FormGeometricSequence (a : ℕ → ℚ) (i j k : ℕ) : Prop :=
  (a j) ^ 2 = a i * a k

theorem arithmetic_sequence_ratio (a : ℕ → ℚ) (d : ℚ) :
  ArithmeticSequence a d →
  FormGeometricSequence a 2 3 9 →
  (a 4 + a 5 + a 6) / (a 2 + a 3 + a 4) = 8 / 3 := by
  sorry

#check arithmetic_sequence_ratio

end arithmetic_sequence_ratio_l3098_309807


namespace grsl_team_count_grsl_solution_l3098_309870

/-- Represents the number of teams in each group of the Greater Regional Soccer League -/
def n : ℕ := sorry

/-- The total number of games played in the league -/
def total_games : ℕ := 56

/-- The number of inter-group games played by each team in Group A -/
def inter_group_games_per_team : ℕ := 2

theorem grsl_team_count :
  n * (n - 1) + 2 * n = total_games :=
sorry

theorem grsl_solution :
  n = 7 :=
sorry

end grsl_team_count_grsl_solution_l3098_309870


namespace distance_AF_l3098_309812

-- Define the parabola
def Parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the focus of the parabola
def Focus : ℝ × ℝ := (1, 0)

-- Define a point on the parabola
structure PointOnParabola where
  x : ℝ
  y : ℝ
  on_parabola : Parabola x y

-- Define the property of the midpoint
def MidpointProperty (A : PointOnParabola) : Prop :=
  (A.x + Focus.1) / 2 = 2

-- Theorem statement
theorem distance_AF (A : PointOnParabola) 
  (h : MidpointProperty A) : 
  Real.sqrt ((A.x - Focus.1)^2 + (A.y - Focus.2)^2) = 4 := by
  sorry

end distance_AF_l3098_309812


namespace missy_dog_yells_l3098_309800

/-- The number of times Missy yells at her dogs -/
def total_yells : ℕ := 60

/-- The ratio of yells at the stubborn dog to yells at the obedient dog -/
def stubborn_to_obedient_ratio : ℕ := 4

/-- The number of times Missy yells at the obedient dog -/
def obedient_dog_yells : ℕ := 12

theorem missy_dog_yells :
  obedient_dog_yells * (stubborn_to_obedient_ratio + 1) = total_yells :=
sorry

end missy_dog_yells_l3098_309800


namespace new_students_weight_l3098_309844

theorem new_students_weight (initial_count : ℕ) (replaced_weight1 replaced_weight2 avg_decrease : ℝ) :
  initial_count = 8 →
  replaced_weight1 = 85 →
  replaced_weight2 = 96 →
  avg_decrease = 7.5 →
  (initial_count : ℝ) * avg_decrease = (replaced_weight1 + replaced_weight2) - (new_student_weight1 + new_student_weight2) →
  new_student_weight1 + new_student_weight2 = 121 :=
by
  sorry

#check new_students_weight

end new_students_weight_l3098_309844


namespace equal_weight_implies_all_genuine_l3098_309863

/-- Represents a coin, which can be either genuine or counterfeit. -/
inductive Coin
| genuine
| counterfeit

/-- The total number of coins. -/
def total_coins : ℕ := 12

/-- The number of genuine coins. -/
def genuine_coins : ℕ := 9

/-- The number of counterfeit coins. -/
def counterfeit_coins : ℕ := 3

/-- A function that returns the weight of a coin. -/
def weight : Coin → ℝ
| Coin.genuine => 1
| Coin.counterfeit => 2  -- Counterfeit coins are heavier

/-- A type representing a selection of coins. -/
def CoinSelection := Fin 6 → Coin

/-- The property that all coins in a selection are genuine. -/
def all_genuine (selection : CoinSelection) : Prop :=
  ∀ i, selection i = Coin.genuine

/-- The property that the weights of two sets of coins are equal. -/
def weights_equal (selection : CoinSelection) : Prop :=
  (weight (selection 0) + weight (selection 1) + weight (selection 2)) =
  (weight (selection 3) + weight (selection 4) + weight (selection 5))

/-- The main theorem to be proved. -/
theorem equal_weight_implies_all_genuine :
  ∀ (selection : CoinSelection),
  weights_equal selection → all_genuine selection :=
by sorry

end equal_weight_implies_all_genuine_l3098_309863


namespace book_sales_theorem_l3098_309857

def monday_sales : ℕ := 15

def tuesday_sales : ℕ := 2 * monday_sales

def wednesday_sales : ℕ := tuesday_sales + (tuesday_sales / 2)

def thursday_sales : ℕ := wednesday_sales + (wednesday_sales / 2)

def friday_expected_sales : ℕ := thursday_sales + (thursday_sales / 2)

def friday_actual_sales : ℕ := friday_expected_sales + (friday_expected_sales / 4)

def saturday_sales : ℕ := (friday_expected_sales * 7) / 10

def total_sales : ℕ := monday_sales + tuesday_sales + wednesday_sales + thursday_sales + friday_actual_sales + saturday_sales

theorem book_sales_theorem : total_sales = 357 := by
  sorry

end book_sales_theorem_l3098_309857


namespace triangle_problem_l3098_309881

theorem triangle_problem (a b c : ℝ) (A B C : ℝ) :
  -- Triangle conditions
  a > 0 ∧ b > 0 ∧ c > 0 ∧
  A > 0 ∧ B > 0 ∧ C > 0 ∧
  A + B + C = π ∧
  -- Given equation
  c * Real.cos B + (Real.sqrt 3 / 3) * b * Real.sin C - a = 0 ∧
  -- Given side length
  c = 3 ∧
  -- Given area
  (1/2) * a * b * Real.sin C = (3 * Real.sqrt 3) / 4 →
  -- Conclusions
  C = π/3 ∧ a + b = 3 * Real.sqrt 2 := by
sorry


end triangle_problem_l3098_309881
