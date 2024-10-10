import Mathlib

namespace all_truth_probability_l974_97457

def alice_truth_prob : ℝ := 0.7
def bob_truth_prob : ℝ := 0.6
def carol_truth_prob : ℝ := 0.8
def david_truth_prob : ℝ := 0.5

theorem all_truth_probability :
  alice_truth_prob * bob_truth_prob * carol_truth_prob * david_truth_prob = 0.168 := by
  sorry

end all_truth_probability_l974_97457


namespace quadratic_solution_property_l974_97413

theorem quadratic_solution_property :
  ∀ p q : ℝ,
  (5 * p^2 - 20 * p + 15 = 0) →
  (5 * q^2 - 20 * q + 15 = 0) →
  p ≠ q →
  (p * q - 3)^2 = 0 :=
by sorry

end quadratic_solution_property_l974_97413


namespace yoga_time_calculation_l974_97410

/-- Represents the ratio of time spent on different activities -/
structure ActivityRatio :=
  (swimming : ℕ)
  (running : ℕ)
  (gym : ℕ)
  (biking : ℕ)
  (yoga : ℕ)

/-- Calculates the time spent on yoga given the activity ratio and biking time -/
def yoga_time (ratio : ActivityRatio) (biking_time : ℕ) : ℕ :=
  (biking_time * ratio.yoga) / ratio.biking

/-- Theorem stating that given the specific activity ratio and 30 minutes of biking, 
    the time spent on yoga is 24 minutes -/
theorem yoga_time_calculation :
  let ratio : ActivityRatio := {
    swimming := 1,
    running := 2,
    gym := 3,
    biking := 5,
    yoga := 4
  }
  let biking_time : ℕ := 30
  yoga_time ratio biking_time = 24 := by sorry

end yoga_time_calculation_l974_97410


namespace election_winner_percentage_l974_97420

theorem election_winner_percentage (total_votes : ℕ) (winning_margin : ℕ) :
  total_votes = 900 →
  winning_margin = 360 →
  ∃ (winning_percentage : ℚ),
    winning_percentage = 70 / 100 ∧
    (winning_percentage * total_votes : ℚ) - ((1 - winning_percentage) * total_votes : ℚ) = winning_margin :=
by sorry

end election_winner_percentage_l974_97420


namespace circle_chord_intersection_theorem_l974_97451

noncomputable def circle_chord_intersection_problem 
  (O : ℝ × ℝ) 
  (A B C D P : ℝ × ℝ) 
  (radius : ℝ) 
  (chord_AB_length : ℝ) 
  (chord_CD_length : ℝ) 
  (midpoint_distance : ℝ) : Prop :=
  let midpoint_AB := ((A.1 + B.1) / 2, (A.2 + B.2) / 2)
  let midpoint_CD := ((C.1 + D.1) / 2, (C.2 + D.2) / 2)
  radius = 25 ∧
  chord_AB_length = 30 ∧
  chord_CD_length = 14 ∧
  midpoint_distance = 12 ∧
  (∀ X : ℝ × ℝ, (X.1 - O.1)^2 + (X.2 - O.2)^2 = radius^2 → 
    ((X = A ∨ X = B ∨ X = C ∨ X = D) ∨ 
     ((X.1 - A.1)^2 + (X.2 - A.2)^2) * ((X.1 - B.1)^2 + (X.2 - B.2)^2) > chord_AB_length^2 ∧
     ((X.1 - C.1)^2 + (X.2 - C.2)^2) * ((X.1 - D.1)^2 + (X.2 - D.2)^2) > chord_CD_length^2)) ∧
  (P.1 - A.1) * (B.1 - A.1) + (P.2 - A.2) * (B.2 - A.2) = 
    (P.1 - B.1) * (A.1 - B.1) + (P.2 - B.2) * (A.2 - B.2) ∧
  (P.1 - C.1) * (D.1 - C.1) + (P.2 - C.2) * (D.2 - C.2) = 
    (P.1 - D.1) * (C.1 - D.1) + (P.2 - D.2) * (C.2 - D.2) ∧
  (midpoint_AB.1 - midpoint_CD.1)^2 + (midpoint_AB.2 - midpoint_CD.2)^2 = midpoint_distance^2 →
  (P.1 - O.1)^2 + (P.2 - O.2)^2 = 4050 / 7

theorem circle_chord_intersection_theorem 
  (O : ℝ × ℝ) 
  (A B C D P : ℝ × ℝ) 
  (radius : ℝ) 
  (chord_AB_length : ℝ) 
  (chord_CD_length : ℝ) 
  (midpoint_distance : ℝ) :
  circle_chord_intersection_problem O A B C D P radius chord_AB_length chord_CD_length midpoint_distance :=
by sorry

end circle_chord_intersection_theorem_l974_97451


namespace blocks_standing_final_value_l974_97465

/-- The number of blocks left standing in the final tower -/
def blocks_standing_final (first_stack : ℕ) (second_stack_diff : ℕ) (final_stack_diff : ℕ) 
  (blocks_standing_second : ℕ) (total_fallen : ℕ) : ℕ :=
  let second_stack := first_stack + second_stack_diff
  let final_stack := second_stack + final_stack_diff
  let fallen_first := first_stack
  let fallen_second := second_stack - blocks_standing_second
  let fallen_final := total_fallen - fallen_first - fallen_second
  final_stack - fallen_final

theorem blocks_standing_final_value :
  blocks_standing_final 7 5 7 2 33 = 3 := by sorry

end blocks_standing_final_value_l974_97465


namespace tiles_crossed_specific_floor_l974_97453

/-- Represents a rectangular floor -/
structure Floor :=
  (width : ℕ) (length : ℕ)

/-- Represents a rectangular tile -/
structure Tile :=
  (width : ℕ) (length : ℕ)

/-- Counts the number of tiles crossed by a diagonal line on a floor -/
def tilesCrossedByDiagonal (f : Floor) (t : Tile) : ℕ :=
  f.width + f.length - Nat.gcd f.width f.length

theorem tiles_crossed_specific_floor :
  let floor := Floor.mk 12 19
  let tile := Tile.mk 1 2
  tilesCrossedByDiagonal floor tile = 30 := by
  sorry

#eval tilesCrossedByDiagonal (Floor.mk 12 19) (Tile.mk 1 2)

end tiles_crossed_specific_floor_l974_97453


namespace circle_center_sum_l974_97458

theorem circle_center_sum (x y : ℝ) : 
  (x^2 + y^2 = 4*x - 6*y + 9) → (x + y = -1) := by
sorry

end circle_center_sum_l974_97458


namespace plane_line_propositions_l974_97462

/-- A plane in 3D space -/
structure Plane

/-- A line in 3D space -/
structure Line

/-- Two lines are parallel -/
def parallel (l1 l2 : Line) : Prop :=
  sorry

/-- A line is parallel to a plane -/
def parallel_to_plane (l : Line) (p : Plane) : Prop :=
  sorry

/-- A line is perpendicular to a plane -/
def perpendicular_to_plane (l : Line) (p : Plane) : Prop :=
  sorry

/-- Two lines are perpendicular -/
def perpendicular (l1 l2 : Line) : Prop :=
  sorry

/-- Two lines are skew -/
def skew (l1 l2 : Line) : Prop :=
  sorry

/-- A line intersects a plane -/
def intersects_plane (l : Line) (p : Plane) : Prop :=
  sorry

/-- The angles formed by two lines with a plane are equal -/
def equal_angles_with_plane (l1 l2 : Line) (p : Plane) : Prop :=
  sorry

theorem plane_line_propositions (α : Plane) (m n : Line) :
  (∃! prop : Prop, prop = true ∧
    (prop = (parallel m n → equal_angles_with_plane m n α) ∨
     prop = (parallel_to_plane m α → parallel_to_plane n α → parallel m n) ∨
     prop = (perpendicular_to_plane m α → perpendicular m n → parallel_to_plane n α) ∨
     prop = (skew m n → parallel_to_plane m α → intersects_plane n α))) :=
  sorry

end plane_line_propositions_l974_97462


namespace linear_function_not_in_fourth_quadrant_l974_97403

theorem linear_function_not_in_fourth_quadrant (b : ℝ) (h : b ≥ 0) :
  ∀ x y : ℝ, y = 2 * x + b → ¬(x > 0 ∧ y < 0) :=
by
  sorry

end linear_function_not_in_fourth_quadrant_l974_97403


namespace product_absolute_value_l974_97449

theorem product_absolute_value (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0)
  (hdistinct : x ≠ y ∧ y ≠ z ∧ x ≠ z)
  (heq : x + 2 / y = y + 2 / z ∧ y + 2 / z = z + 2 / x) :
  |x * y * z| = 2 * Real.sqrt 2 := by
sorry

end product_absolute_value_l974_97449


namespace k_value_max_value_on_interval_l974_97476

-- Define the function f(x) with parameter k
def f (k : ℝ) (x : ℝ) : ℝ := x^2 + (2*k - 3)*x + k^2 - 7

-- Theorem 1: k = 3 given the zeros of f(x)
theorem k_value (k : ℝ) : (f k (-1) = 0 ∧ f k (-2) = 0) → k = 3 := by sorry

-- Define the specific function f(x) = x^2 + 3x + 2
def f_specific (x : ℝ) : ℝ := x^2 + 3*x + 2

-- Theorem 2: Maximum value of f_specific on [-2, 2] is 12
theorem max_value_on_interval : 
  ∀ x : ℝ, x ∈ Set.Icc (-2) 2 → f_specific x ≤ 12 ∧ ∃ y ∈ Set.Icc (-2) 2, f_specific y = 12 := by sorry

end k_value_max_value_on_interval_l974_97476


namespace quadratic_inequality_range_l974_97411

/-- Given that the inequality ax^2 + x + 1 < 0 has a non-empty solution set for x,
    prove that the range of a is a < 1/4 -/
theorem quadratic_inequality_range (a : ℝ) : 
  (∃ x : ℝ, a * x^2 + x + 1 < 0) → a < (1/4 : ℝ) :=
by sorry

end quadratic_inequality_range_l974_97411


namespace close_interval_for_f_and_g_l974_97438

-- Define the functions f and g
def f (x : ℝ) : ℝ := x^2 - 3*x + 4
def g (x : ℝ) : ℝ := 2*x - 3

-- Define the property of being "close functions" on an interval
def are_close_functions (f g : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x, a ≤ x ∧ x ≤ b → |f x - g x| ≤ 1

-- State the theorem
theorem close_interval_for_f_and_g :
  are_close_functions f g 2 3 ∧
  ∀ a b, a < 2 ∨ b > 3 → ¬(are_close_functions f g a b) :=
sorry

end close_interval_for_f_and_g_l974_97438


namespace reverse_digits_sum_l974_97467

/-- Two-digit integer -/
def TwoDigitInt (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

theorem reverse_digits_sum (x y : ℕ) (hx : TwoDigitInt x) (hy : TwoDigitInt y)
  (h_reverse : y = 10 * (x % 10) + (x / 10))
  (a b : ℕ) (hx_digits : x = 10 * a + b)
  (hab : a - b = 3)
  (m : ℕ) (hm : x^2 - y^2 = m^2) :
  x + y + m = 178 := by
sorry

end reverse_digits_sum_l974_97467


namespace identity_function_l974_97472

theorem identity_function (f : ℕ → ℕ) (h : ∀ n : ℕ, f (n + 1) > f (f n)) : 
  ∀ n : ℕ, f n = n := by
sorry

end identity_function_l974_97472


namespace distance_to_park_is_correct_l974_97461

/-- The distance from point A to the forest amusement park -/
def distance_to_park : ℕ := 2370

/-- The rabbit's starting time in minutes after midnight -/
def rabbit_start : ℕ := 9 * 60

/-- The turtle's starting time in minutes after midnight -/
def turtle_start : ℕ := 6 * 60 + 40

/-- The rabbit's speed in meters per minute -/
def rabbit_speed : ℕ := 40

/-- The turtle's speed in meters per minute -/
def turtle_speed : ℕ := 10

/-- The rabbit's jumping time in minutes -/
def rabbit_jump_time : ℕ := 3

/-- The rabbit's resting time in minutes -/
def rabbit_rest_time : ℕ := 2

/-- The time difference between rabbit and turtle arrival in seconds -/
def arrival_time_diff : ℕ := 15

theorem distance_to_park_is_correct : 
  ∀ (t : ℕ), 
  t * turtle_speed = distance_to_park ∧ 
  t = (rabbit_start - turtle_start) + 
      (distance_to_park - (rabbit_start - turtle_start) * turtle_speed) / 
      (rabbit_speed * rabbit_jump_time / (rabbit_jump_time + rabbit_rest_time) - turtle_speed) + 
      arrival_time_diff / 60 :=
sorry

end distance_to_park_is_correct_l974_97461


namespace smallest_integer_with_remainders_l974_97470

theorem smallest_integer_with_remainders : ∃ (x : ℕ), 
  (x > 0) ∧ 
  (x % 5 = 2) ∧ 
  (x % 7 = 3) ∧ 
  (∀ y : ℕ, y > 0 ∧ y % 5 = 2 ∧ y % 7 = 3 → x ≤ y) ∧
  (x = 17) := by
sorry

end smallest_integer_with_remainders_l974_97470


namespace digit_150_of_one_thirteenth_l974_97431

def decimal_representation (n : ℕ) : ℚ → ℕ := sorry

theorem digit_150_of_one_thirteenth : decimal_representation 150 (1/13) = 3 := by
  sorry

end digit_150_of_one_thirteenth_l974_97431


namespace tangent_intersection_of_specific_circles_l974_97498

/-- A circle in a 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- The x-coordinate of the intersection point of a line tangent to two circles -/
def tangentIntersection (c1 c2 : Circle) : ℝ :=
  sorry

theorem tangent_intersection_of_specific_circles :
  let c1 : Circle := { center := (0, 0), radius := 3 }
  let c2 : Circle := { center := (18, 0), radius := 8 }
  tangentIntersection c1 c2 = 54 / 11 := by
  sorry

end tangent_intersection_of_specific_circles_l974_97498


namespace first_courier_speed_l974_97421

/-- The speed of the first courier in km/h -/
def v : ℝ := 30

/-- The distance between cities A and B in km -/
def distance : ℝ := 120

/-- The speed of the second courier in km/h -/
def speed_second : ℝ := 50

/-- The time delay of the second courier in hours -/
def delay : ℝ := 1

theorem first_courier_speed :
  (distance / v = (3 * speed_second) / (v - speed_second)) ∧
  (v > 0) ∧ (v < speed_second) := by
  sorry

#check first_courier_speed

end first_courier_speed_l974_97421


namespace katrina_lunch_sales_l974_97493

/-- The number of cookies sold during the lunch rush -/
def lunch_rush_sales (initial : ℕ) (morning_dozens : ℕ) (afternoon : ℕ) (remaining : ℕ) : ℕ :=
  initial - (morning_dozens * 12) - afternoon - remaining

/-- Proof that Katrina sold 57 cookies during the lunch rush -/
theorem katrina_lunch_sales :
  lunch_rush_sales 120 3 16 11 = 57 := by
  sorry

end katrina_lunch_sales_l974_97493


namespace correct_mixture_ratio_l974_97464

/-- Represents a salt solution with a given concentration and amount -/
structure SaltSolution :=
  (concentration : ℚ)
  (amount : ℚ)

/-- Represents a mixture of two salt solutions -/
def mix (s1 s2 : SaltSolution) (r1 r2 : ℚ) : SaltSolution :=
  { concentration := (s1.concentration * r1 + s2.concentration * r2) / (r1 + r2),
    amount := r1 + r2 }

theorem correct_mixture_ratio :
  let solutionA : SaltSolution := ⟨2/5, 30⟩
  let solutionB : SaltSolution := ⟨4/5, 60⟩
  let mixedSolution := mix solutionA solutionB 3 1
  mixedSolution.concentration = 1/2 ∧ mixedSolution.amount = 50 :=
by sorry


end correct_mixture_ratio_l974_97464


namespace eighteenth_replacement_november_l974_97486

/-- Represents months of the year -/
inductive Month
| January | February | March | April | May | June
| July | August | September | October | November | December

/-- Converts a number of months since January to a Month -/
def monthsToMonth (n : ℕ) : Month :=
  match n % 12 with
  | 0 => Month.December
  | 1 => Month.January
  | 2 => Month.February
  | 3 => Month.March
  | 4 => Month.April
  | 5 => Month.May
  | 6 => Month.June
  | 7 => Month.July
  | 8 => Month.August
  | 9 => Month.September
  | 10 => Month.October
  | _ => Month.November

/-- The month of the nth battery replacement, given replacements occur every 7 months starting from January -/
def batteryReplacementMonth (n : ℕ) : Month :=
  monthsToMonth (7 * (n - 1) + 1)

theorem eighteenth_replacement_november :
  batteryReplacementMonth 18 = Month.November := by
  sorry

end eighteenth_replacement_november_l974_97486


namespace shortest_tangent_length_l974_97406

-- Define the circles
def C1 (x y : ℝ) : Prop := (x - 12)^2 + y^2 = 49
def C2 (x y : ℝ) : Prop := (x + 18)^2 + y^2 = 64

-- Define the tangent line segment
def is_tangent (P Q : ℝ × ℝ) : Prop :=
  C1 P.1 P.2 ∧ C2 Q.1 Q.2 ∧ 
  ∀ R : ℝ × ℝ, (C1 R.1 R.2 ∨ C2 R.1 R.2) → 
    Real.sqrt ((R.1 - P.1)^2 + (R.2 - P.2)^2) + Real.sqrt ((R.1 - Q.1)^2 + (R.2 - Q.2)^2) ≥ 
    Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)

-- Theorem statement
theorem shortest_tangent_length :
  ∃ P Q : ℝ × ℝ, is_tangent P Q ∧ 
    Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2) = 15 ∧
    ∀ P' Q' : ℝ × ℝ, is_tangent P' Q' → 
      Real.sqrt ((P'.1 - Q'.1)^2 + (P'.2 - Q'.2)^2) ≥ 15 :=
sorry

end shortest_tangent_length_l974_97406


namespace domain_of_f_l974_97423

noncomputable def f (x : ℝ) : ℝ := (x + 1) / (x^2 + 5*x + 6)

theorem domain_of_f :
  {x : ℝ | ∃ y, f x = y} = {x : ℝ | x ≠ -3 ∧ x ≠ -2} :=
by sorry

end domain_of_f_l974_97423


namespace chord_length_dot_product_value_l974_97400

-- Define the line l
def line_l (x y : ℝ) : Prop := 3 * x + y - 6 = 0

-- Define the circle C
def circle_C (x y : ℝ) : Prop := (x - 0)^2 + (y - 1)^2 = 5

-- Define point P
def point_P : ℝ × ℝ := (0, -2)

-- Theorem for the length of the chord
theorem chord_length :
  ∃ (A B : ℝ × ℝ),
    line_l A.1 A.2 ∧ line_l B.1 B.2 ∧
    circle_C A.1 A.2 ∧ circle_C B.1 B.2 ∧
    (A.1 - B.1)^2 + (A.2 - B.2)^2 = 10 :=
sorry

-- Theorem for the dot product
theorem dot_product_value :
  ∀ (A B : ℝ × ℝ),
    circle_C A.1 A.2 ∧ circle_C B.1 B.2 ∧
    A ≠ B ∧
    ∃ (t : ℝ), A.1 = point_P.1 + t * (A.1 - point_P.1) ∧
                A.2 = point_P.2 + t * (A.2 - point_P.2) ∧
                B.1 = point_P.1 + t * (B.1 - point_P.1) ∧
                B.2 = point_P.2 + t * (B.2 - point_P.2) →
    ((A.1 - point_P.1) * (B.1 - point_P.1) + (A.2 - point_P.2) * (B.2 - point_P.2))^2 = 16 :=
sorry

end chord_length_dot_product_value_l974_97400


namespace parallel_vectors_m_value_l974_97402

/-- Two vectors are parallel if their cross product is zero -/
def are_parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

theorem parallel_vectors_m_value :
  let a : ℝ × ℝ := (1, 3)
  let b : ℝ × ℝ := (m, 1)
  are_parallel a b → m = 1/3 := by
  sorry

end parallel_vectors_m_value_l974_97402


namespace earnings_difference_proof_l974_97437

/-- Calculates the difference in annual earnings between two jobs --/
def annual_earnings_difference (
  new_wage : ℕ
  ) (new_hours : ℕ
  ) (old_wage : ℕ
  ) (old_hours : ℕ
  ) (weeks_per_year : ℕ
  ) : ℕ :=
  (new_wage * new_hours * weeks_per_year) - (old_wage * old_hours * weeks_per_year)

/-- Proves that the difference in annual earnings is $20,800 --/
theorem earnings_difference_proof :
  annual_earnings_difference 20 40 16 25 52 = 20800 := by
  sorry

end earnings_difference_proof_l974_97437


namespace intersection_A_B_l974_97447

def A : Set ℝ := {x | 0 ≤ x ∧ x ≤ 4}
def B : Set ℝ := {x | x^2 + 4*x ≤ 0}

theorem intersection_A_B : A ∩ B = {0} := by sorry

end intersection_A_B_l974_97447


namespace pizza_slices_left_l974_97445

theorem pizza_slices_left (total_slices : Nat) (john_slices : Nat) (sam_multiplier : Nat) : 
  total_slices = 12 → 
  john_slices = 3 → 
  sam_multiplier = 2 → 
  total_slices - (john_slices + sam_multiplier * john_slices) = 3 := by
  sorry

end pizza_slices_left_l974_97445


namespace zach_rental_cost_l974_97440

/-- Calculates the total cost of a car rental given the base cost, cost per mile, and miles driven. -/
def rental_cost (base_cost : ℚ) (cost_per_mile : ℚ) (miles_driven : ℚ) : ℚ :=
  base_cost + cost_per_mile * miles_driven

/-- Proves that the total cost of Zach's car rental is $832. -/
theorem zach_rental_cost :
  let base_cost : ℚ := 150
  let cost_per_mile : ℚ := 1/2
  let monday_miles : ℚ := 620
  let thursday_miles : ℚ := 744
  let total_miles : ℚ := monday_miles + thursday_miles
  rental_cost base_cost cost_per_mile total_miles = 832 := by
  sorry

end zach_rental_cost_l974_97440


namespace square_sum_implies_sum_l974_97489

theorem square_sum_implies_sum (x : ℝ) (h : x > 0) :
  Real.sqrt x + (Real.sqrt x)⁻¹ = 3 → x + x⁻¹ = 7 := by
  sorry

end square_sum_implies_sum_l974_97489


namespace solve_system_of_equations_l974_97434

theorem solve_system_of_equations (a b : ℝ) 
  (eq1 : 3 * a + 2 = 2) 
  (eq2 : 2 * b - 3 * a = 4) : 
  b = 2 := by
sorry

end solve_system_of_equations_l974_97434


namespace sundae_probability_l974_97483

def ice_cream_flavors : ℕ := 3
def syrup_types : ℕ := 2
def topping_options : ℕ := 3

def total_combinations : ℕ := ice_cream_flavors * syrup_types * topping_options

def specific_combination : ℕ := 1

theorem sundae_probability :
  (specific_combination : ℚ) / total_combinations = 1 / 18 := by sorry

end sundae_probability_l974_97483


namespace target_same_type_as_reference_l974_97495

/-- Represents a monomial term with variables x and y -/
structure Monomial :=
  (x_exp : ℕ)
  (y_exp : ℕ)

/-- Determines if two monomials are of the same type -/
def same_type (m1 m2 : Monomial) : Prop :=
  m1.x_exp = m2.x_exp ∧ m1.y_exp = m2.y_exp

/-- The reference monomial 3x²y -/
def reference : Monomial :=
  ⟨2, 1⟩

/-- The monomial -yx² -/
def target : Monomial :=
  ⟨2, 1⟩

theorem target_same_type_as_reference : same_type target reference :=
  sorry

end target_same_type_as_reference_l974_97495


namespace quadratic_discriminant_l974_97456

/-- The discriminant of a quadratic equation ax^2 + bx + c is b^2 - 4ac -/
def discriminant (a b c : ℝ) : ℝ := b^2 - 4*a*c

/-- The quadratic equation 4x^2 - 6x + 9 has discriminant -108 -/
theorem quadratic_discriminant :
  discriminant 4 (-6) 9 = -108 := by
  sorry

end quadratic_discriminant_l974_97456


namespace polynomial_divisibility_l974_97455

/-- A polynomial of degree 5 with coefficients in ℝ -/
def polynomial (p q : ℝ) (x : ℝ) : ℝ :=
  x^5 - x^4 + x^3 - p*x^2 + q*x + 9

/-- The condition that the polynomial is divisible by (x + 3)(x - 2) -/
def is_divisible (p q : ℝ) : Prop :=
  ∀ x : ℝ, (x + 3 = 0 ∨ x - 2 = 0) → polynomial p q x = 0

/-- The main theorem stating that if the polynomial is divisible by (x + 3)(x - 2),
    then p = -130.5 and q = -277.5 -/
theorem polynomial_divisibility (p q : ℝ) :
  is_divisible p q → p = -130.5 ∧ q = -277.5 := by
  sorry

end polynomial_divisibility_l974_97455


namespace triangle_point_distance_l974_97405

-- Define the triangle and points
def Triangle (A B C : ℝ × ℝ) : Prop :=
  let d (p q : ℝ × ℝ) := Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)
  d A B = 4 ∧ d B C = 5 ∧ d C A = 6

def OnRay (A B D : ℝ × ℝ) : Prop :=
  ∃ t : ℝ, t > 1 ∧ D = (A.1 + t * (B.1 - A.1), A.2 + t * (B.2 - A.2))

def CircumCircle (A B C P : ℝ × ℝ) : Prop :=
  let d (p q : ℝ × ℝ) := Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)
  d A P = d B P ∧ d B P = d C P

-- State the theorem
theorem triangle_point_distance (A B C D E F : ℝ × ℝ) :
  Triangle A B C →
  OnRay A B D →
  OnRay A B E →
  CircumCircle A C D F →
  CircumCircle E B C F →
  F ≠ C →
  let d (p q : ℝ × ℝ) := Real.sqrt ((p.1 - q.1)^2 + (p.2 - q.2)^2)
  d D F = 2 →
  d E F = 7 →
  d B E = (5 + 21 * Real.sqrt 2) / 4 :=
by sorry

end triangle_point_distance_l974_97405


namespace sqrt_equation_solution_l974_97478

theorem sqrt_equation_solution (y : ℝ) : 
  Real.sqrt (2 + Real.sqrt (3 * y - 4)) = Real.sqrt 7 → y = 29 / 3 := by
  sorry

end sqrt_equation_solution_l974_97478


namespace natalie_portion_ratio_l974_97482

def total_amount : ℝ := 10000

def third_person_amount : ℝ := 2000

def second_person_percentage : ℝ := 0.6

theorem natalie_portion_ratio (first_person_amount : ℝ) 
  (h1 : third_person_amount = total_amount - first_person_amount - second_person_percentage * (total_amount - first_person_amount))
  (h2 : first_person_amount > 0)
  (h3 : first_person_amount < total_amount) :
  first_person_amount / total_amount = 1 / 2 := by
  sorry

end natalie_portion_ratio_l974_97482


namespace sphere_properties_l974_97459

/-- Proves surface area and volume of a sphere with diameter 10 inches -/
theorem sphere_properties :
  let d : ℝ := 10  -- diameter
  let r : ℝ := d / 2  -- radius
  ∀ (S V : ℝ),  -- surface area and volume
  S = 4 * Real.pi * r^2 →
  V = (4/3) * Real.pi * r^3 →
  S = 100 * Real.pi ∧ V = (500/3) * Real.pi :=
by sorry

end sphere_properties_l974_97459


namespace complex_fraction_equality_l974_97444

theorem complex_fraction_equality : 
  2013 * (5.7 * 4.2 + (21/5) * 4.3) / ((14/73) * 15 + (5/73) * 177 + 656) = 126 := by
  sorry

end complex_fraction_equality_l974_97444


namespace milkshake_leftover_l974_97441

/-- Calculates the amount of milk left over after making milkshakes -/
theorem milkshake_leftover (milk_per_shake ice_cream_per_shake total_milk total_ice_cream : ℕ) :
  milk_per_shake = 4 →
  ice_cream_per_shake = 12 →
  total_milk = 72 →
  total_ice_cream = 192 →
  total_milk - (total_ice_cream / ice_cream_per_shake * milk_per_shake) = 8 := by
  sorry

#check milkshake_leftover

end milkshake_leftover_l974_97441


namespace sum_of_interior_angles_is_180_l974_97412

-- Define a triangle in Euclidean space
def Triangle : Type := ℝ × ℝ × ℝ

-- Define the function that calculates the sum of interior angles of a triangle
def sum_of_interior_angles (t : Triangle) : ℝ := sorry

-- Theorem stating that the sum of interior angles of any triangle is 180°
theorem sum_of_interior_angles_is_180 (t : Triangle) :
  sum_of_interior_angles t = 180 := by sorry

end sum_of_interior_angles_is_180_l974_97412


namespace bug_position_after_2023_jumps_l974_97488

/-- Represents the points on the circle -/
inductive Point
| one | two | three | four | five | six | seven

/-- Calculates the next point based on the jumping rules -/
def nextPoint (p : Point) : Point :=
  match p with
  | Point.one => Point.four
  | Point.two => Point.five
  | Point.three => Point.five
  | Point.four => Point.seven
  | Point.five => Point.one
  | Point.six => Point.two
  | Point.seven => Point.three

/-- Calculates the point after n jumps -/
def jumpNTimes (start : Point) (n : ℕ) : Point :=
  match n with
  | 0 => start
  | n + 1 => nextPoint (jumpNTimes start n)

theorem bug_position_after_2023_jumps :
  jumpNTimes Point.seven 2023 = Point.one :=
sorry

end bug_position_after_2023_jumps_l974_97488


namespace matrix_power_four_l974_97443

def A : Matrix (Fin 2) (Fin 2) ℤ := !![1, -1; 1, 0]

theorem matrix_power_four :
  A ^ 4 = !![(-1 : ℤ), 1; -1, 0] := by sorry

end matrix_power_four_l974_97443


namespace peanut_price_in_mixed_nuts_l974_97416

/-- Calculates the price per pound of peanuts in a mixed nut blend --/
theorem peanut_price_in_mixed_nuts
  (total_weight : ℝ)
  (mixed_price : ℝ)
  (cashew_weight : ℝ)
  (cashew_price : ℝ)
  (h1 : total_weight = 100)
  (h2 : mixed_price = 2.5)
  (h3 : cashew_weight = 60)
  (h4 : cashew_price = 4) :
  (total_weight * mixed_price - cashew_weight * cashew_price) / (total_weight - cashew_weight) = 0.25 := by
  sorry

end peanut_price_in_mixed_nuts_l974_97416


namespace ship_departure_theorem_l974_97426

/-- Represents the total transit time for a cargo shipment -/
def total_transit_time (navigation_time customs_time delivery_time : ℕ) : ℕ :=
  navigation_time + customs_time + delivery_time

/-- Calculates the departure date given the expected arrival and total transit time -/
def departure_date (days_until_arrival total_transit : ℕ) : ℕ :=
  days_until_arrival + total_transit

/-- Theorem: Given the specified conditions, the ship should have departed 34 days ago -/
theorem ship_departure_theorem (navigation_time customs_time delivery_time days_until_arrival : ℕ)
  (h1 : navigation_time = 21)
  (h2 : customs_time = 4)
  (h3 : delivery_time = 7)
  (h4 : days_until_arrival = 2) :
  departure_date days_until_arrival (total_transit_time navigation_time customs_time delivery_time) = 34 := by
  sorry

#check ship_departure_theorem

end ship_departure_theorem_l974_97426


namespace quadrilateral_side_length_l974_97417

/-- Given a quadrilateral ABCD with specific side lengths and angles, prove that AD = √7 -/
theorem quadrilateral_side_length (A B C D : ℝ × ℝ) : 
  let AB := Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2)
  let BC := Real.sqrt ((B.1 - C.1)^2 + (B.2 - C.2)^2)
  let CD := Real.sqrt ((C.1 - D.1)^2 + (C.2 - D.2)^2)
  let AD := Real.sqrt ((A.1 - D.1)^2 + (A.2 - D.2)^2)
  let angle_ABC := Real.arccos ((AB^2 + BC^2 - (A.1 - C.1)^2 - (A.2 - C.2)^2) / (2 * AB * BC))
  let angle_BCD := Real.arccos ((BC^2 + CD^2 - (B.1 - D.1)^2 - (B.2 - D.2)^2) / (2 * BC * CD))
  AB = 1 ∧ BC = 2 ∧ CD = Real.sqrt 3 ∧ angle_ABC = 2 * Real.pi / 3 ∧ angle_BCD = Real.pi / 2 →
  AD = Real.sqrt 7 := by
sorry


end quadrilateral_side_length_l974_97417


namespace horseshoe_selling_price_l974_97469

/-- Proves that the selling price per set of horseshoes is $50 given the specified conditions. -/
theorem horseshoe_selling_price
  (initial_outlay : ℕ)
  (cost_per_set : ℕ)
  (num_sets : ℕ)
  (profit : ℕ)
  (h1 : initial_outlay = 10000)
  (h2 : cost_per_set = 20)
  (h3 : num_sets = 500)
  (h4 : profit = 5000) :
  ∃ (selling_price : ℕ),
    selling_price * num_sets = initial_outlay + cost_per_set * num_sets + profit ∧
    selling_price = 50 :=
by sorry

end horseshoe_selling_price_l974_97469


namespace min_value_problem_l974_97432

theorem min_value_problem (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x * y^3 = 16/9) :
  3 * x + y ≥ 8/3 ∧ ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ x₀ * y₀^3 = 16/9 ∧ 3 * x₀ + y₀ = 8/3 := by
  sorry

end min_value_problem_l974_97432


namespace factorial_gcd_property_l974_97430

theorem factorial_gcd_property (m n : ℕ) (h : m > n) :
  Nat.gcd (Nat.factorial n) (Nat.factorial m) = Nat.factorial n := by
  sorry

end factorial_gcd_property_l974_97430


namespace scientific_notation_of_0_00001_l974_97427

/-- Scientific notation representation of a real number -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ |coefficient| ∧ |coefficient| < 10

/-- Convert a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem scientific_notation_of_0_00001 :
  toScientificNotation 0.00001 = ScientificNotation.mk 1 (-5) sorry :=
  sorry

end scientific_notation_of_0_00001_l974_97427


namespace age_of_replaced_man_l974_97463

theorem age_of_replaced_man
  (n : ℕ)
  (initial_avg : ℝ)
  (new_avg : ℝ)
  (age_increase : ℝ)
  (replaced_man1_age : ℝ)
  (women_avg_age : ℝ)
  (h1 : n = 7)
  (h2 : new_avg = initial_avg + age_increase)
  (h3 : age_increase = 4)
  (h4 : replaced_man1_age = 30)
  (h5 : women_avg_age = 42)
  : ∃ (replaced_man2_age : ℝ),
    n * new_avg = n * initial_avg - replaced_man1_age - replaced_man2_age + 2 * women_avg_age
    ∧ replaced_man2_age = 26 :=
by sorry

end age_of_replaced_man_l974_97463


namespace collinear_points_k_value_l974_97439

/-- Given vectors OA, OB, OC in ℝ², prove that if A, B, C are collinear, then the x-coordinate of OA is 6. -/
theorem collinear_points_k_value (OA OB OC : ℝ × ℝ) :
  OA.1 = k ∧ OA.2 = 11 ∧
  OB = (4, 5) ∧
  OC = (5, 8) ∧
  ∃ (t : ℝ), (OC.1 - OB.1, OC.2 - OB.2) = t • (OB.1 - OA.1, OB.2 - OA.2) →
  k = 6 := by
sorry

end collinear_points_k_value_l974_97439


namespace scientific_notation_of_40_9_billion_l974_97409

theorem scientific_notation_of_40_9_billion :
  (40.9 : ℝ) * 1000000000 = 4.09 * (10 : ℝ)^9 := by
  sorry

end scientific_notation_of_40_9_billion_l974_97409


namespace prism_with_21_edges_has_14_vertices_l974_97450

/-- A prism is a polyhedron with two congruent and parallel bases -/
structure Prism where
  base_edges : ℕ
  total_edges : ℕ

/-- The number of edges in a prism is three times the number of edges in its base -/
axiom prism_edge_count (p : Prism) : p.total_edges = 3 * p.base_edges

/-- The number of vertices in a prism is twice the number of edges in its base -/
def prism_vertex_count (p : Prism) : ℕ := 2 * p.base_edges

/-- Theorem: A prism with 21 edges has 14 vertices -/
theorem prism_with_21_edges_has_14_vertices (p : Prism) (h : p.total_edges = 21) : 
  prism_vertex_count p = 14 := by
  sorry

end prism_with_21_edges_has_14_vertices_l974_97450


namespace simplify_expression_l974_97422

theorem simplify_expression (x y : ℝ) : 3*x + 2*y + 4*x + 5*y + 7 = 7*x + 7*y + 7 := by
  sorry

end simplify_expression_l974_97422


namespace smallest_disguisable_triangle_two_sides_perfect_squares_l974_97480

/-- A triangle with integer side lengths a, b, and c is disguisable if there exists a similar triangle
    with side lengths d, a, b where d ≥ a ≥ b > c -/
def IsDisguisableTriangle (a b c : ℕ) : Prop :=
  ∃ d : ℚ, d ≥ a ∧ a ≥ b ∧ b > c ∧ (d : ℚ) / a = (a : ℚ) / b ∧ (a : ℚ) / b = (b : ℚ) / c

/-- The perimeter of a triangle with side lengths a, b, and c -/
def Perimeter (a b c : ℕ) : ℕ := a + b + c

/-- A number is a perfect square if it's equal to some integer squared -/
def IsPerfectSquare (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

theorem smallest_disguisable_triangle :
  ∀ a b c : ℕ, IsDisguisableTriangle a b c →
    Perimeter a b c ≥ 19 ∧
    (Perimeter a b c = 19 → (a, b, c) = (9, 6, 4)) :=
sorry

theorem two_sides_perfect_squares :
  ∀ a b c : ℕ, IsDisguisableTriangle a b c →
    (∀ k : ℕ, k < a → ¬IsDisguisableTriangle k (k * b / a) (k * c / a)) →
    (IsPerfectSquare a ∧ IsPerfectSquare c) ∨
    (IsPerfectSquare a ∧ IsPerfectSquare b) ∨
    (IsPerfectSquare b ∧ IsPerfectSquare c) :=
sorry

end smallest_disguisable_triangle_two_sides_perfect_squares_l974_97480


namespace intersection_of_A_and_B_l974_97435

def A : Set ℝ := {x | x + 2 = 0}
def B : Set ℝ := {x | x^2 - 4 = 0}

theorem intersection_of_A_and_B : A ∩ B = {-2} := by sorry

end intersection_of_A_and_B_l974_97435


namespace limit_at_one_l974_97424

noncomputable def f (x : ℝ) : ℝ := (5/3) * x - Real.log (2*x + 1)

theorem limit_at_one (ε : ℝ) (hε : ε > 0) :
  ∃ δ > 0, ∀ Δx : ℝ, 0 < |Δx| ∧ |Δx| < δ →
    |(f (1 + Δx) - f 1) / Δx - 1| < ε :=
sorry

end limit_at_one_l974_97424


namespace current_trees_count_l974_97473

/-- The number of dogwood trees to be planted today -/
def trees_planted_today : ℕ := 41

/-- The number of dogwood trees to be planted tomorrow -/
def trees_planted_tomorrow : ℕ := 20

/-- The total number of dogwood trees after planting -/
def total_trees_after_planting : ℕ := 100

/-- The number of dogwood trees currently in the park -/
def current_trees : ℕ := total_trees_after_planting - trees_planted_today - trees_planted_tomorrow

theorem current_trees_count : current_trees = 39 := by
  sorry

end current_trees_count_l974_97473


namespace equal_reading_time_l974_97471

/-- Represents the reading scenario in Mrs. Reed's English class -/
structure ReadingScenario where
  total_pages : ℕ
  mia_speed : ℕ  -- seconds per page
  leo_speed : ℕ  -- seconds per page
  mia_pages : ℕ

/-- The specific reading scenario from the problem -/
def problem_scenario : ReadingScenario :=
  { total_pages := 840
  , mia_speed := 60
  , leo_speed := 40
  , mia_pages := 336 }

/-- Calculates the total reading time for a given number of pages and reading speed -/
def reading_time (pages : ℕ) (speed : ℕ) : ℕ := pages * speed

/-- Theorem stating that Mia and Leo spend equal time reading in the given scenario -/
theorem equal_reading_time (s : ReadingScenario) (h : s = problem_scenario) :
  reading_time s.mia_pages s.mia_speed = reading_time (s.total_pages - s.mia_pages) s.leo_speed := by
  sorry

#check equal_reading_time

end equal_reading_time_l974_97471


namespace product_expansion_sum_l974_97460

theorem product_expansion_sum (a b c d : ℝ) :
  (∀ x, (4*x^2 - 6*x + 5) * (8 - 3*x) = a*x^3 + b*x^2 + c*x + d) →
  8*a + 4*b + 2*c + d = 18 := by
sorry

end product_expansion_sum_l974_97460


namespace green_sweets_count_l974_97407

/-- Given the number of blue and yellow sweets, and the total number of sweets,
    calculate the number of green sweets. -/
theorem green_sweets_count 
  (blue_sweets : ℕ) 
  (yellow_sweets : ℕ) 
  (total_sweets : ℕ) 
  (h1 : blue_sweets = 310) 
  (h2 : yellow_sweets = 502) 
  (h3 : total_sweets = 1024) : 
  total_sweets - (blue_sweets + yellow_sweets) = 212 := by
sorry

#eval 1024 - (310 + 502)  -- This should output 212

end green_sweets_count_l974_97407


namespace totalCost_equals_64_l974_97401

-- Define the side length of each square
def squareSide : ℝ := 4

-- Define the number of squares
def numSquares : ℕ := 4

-- Define the areas of overlap
def centralOverlap : ℕ := 1
def tripleOverlap : ℕ := 6
def doubleOverlap : ℕ := 12
def singleArea : ℕ := 18

-- Define the cost function
def costFunction (overlappingSquares : ℕ) : ℕ := overlappingSquares

-- Theorem statement
theorem totalCost_equals_64 :
  (centralOverlap * costFunction numSquares) +
  (tripleOverlap * costFunction 3) +
  (doubleOverlap * costFunction 2) +
  (singleArea * costFunction 1) = 64 := by
  sorry

end totalCost_equals_64_l974_97401


namespace egg_sale_remainder_l974_97436

theorem egg_sale_remainder : (53 + 65 + 26) % 15 = 9 := by sorry

end egg_sale_remainder_l974_97436


namespace fraction_zero_implies_x_negative_one_l974_97491

theorem fraction_zero_implies_x_negative_one (x : ℝ) :
  (1 - x^2) / (x - 1) = 0 ∧ x ≠ 1 → x = -1 := by
  sorry

end fraction_zero_implies_x_negative_one_l974_97491


namespace matchstick_length_theorem_l974_97433

/-- Represents a figure made of matchsticks -/
structure MatchstickFigure where
  smallSquareCount : ℕ
  largeSquareCount : ℕ
  totalArea : ℝ

/-- Calculates the total length of matchsticks used in the figure -/
def totalMatchstickLength (figure : MatchstickFigure) : ℝ :=
  sorry

/-- Theorem stating the total length of matchsticks in the given figure -/
theorem matchstick_length_theorem (figure : MatchstickFigure) 
  (h1 : figure.smallSquareCount = 8)
  (h2 : figure.largeSquareCount = 1)
  (h3 : figure.totalArea = 300) :
  totalMatchstickLength figure = 140 := by
  sorry

end matchstick_length_theorem_l974_97433


namespace largest_inscribed_square_side_length_largest_inscribed_square_side_length_proof_l974_97477

/-- The side length of the largest square that can be inscribed in a square with side length 12,
    given two congruent equilateral triangles are inscribed as described in the problem. -/
theorem largest_inscribed_square_side_length : ℝ :=
  let outer_square_side : ℝ := 12
  let triangle_side : ℝ := 4 * Real.sqrt 6
  6 - Real.sqrt 6

/-- Proof that the calculated side length is correct -/
theorem largest_inscribed_square_side_length_proof :
  largest_inscribed_square_side_length = 6 - Real.sqrt 6 := by
  sorry

end largest_inscribed_square_side_length_largest_inscribed_square_side_length_proof_l974_97477


namespace crocus_to_daffodil_ratio_l974_97414

/-- Represents the number of flower bulbs of each type planted by Jane. -/
structure FlowerBulbs where
  tulips : ℕ
  irises : ℕ
  daffodils : ℕ
  crocus : ℕ

/-- Calculates the total earnings from planting flower bulbs. -/
def earnings (bulbs : FlowerBulbs) : ℚ :=
  0.5 * (bulbs.tulips + bulbs.irises + bulbs.daffodils + bulbs.crocus)

/-- Proves that given the conditions, the ratio of crocus bulbs to daffodil bulbs is 3:1. -/
theorem crocus_to_daffodil_ratio 
  (bulbs : FlowerBulbs)
  (h1 : bulbs.tulips = 20)
  (h2 : bulbs.irises = bulbs.tulips / 2)
  (h3 : bulbs.daffodils = 30)
  (h4 : earnings bulbs = 75) :
  bulbs.crocus / bulbs.daffodils = 3 := by
  sorry


end crocus_to_daffodil_ratio_l974_97414


namespace subtracted_value_l974_97479

theorem subtracted_value (x y : ℤ) (h1 : x = 120) (h2 : 2 * x - y = 102) : y = 138 := by
  sorry

end subtracted_value_l974_97479


namespace set_inclusion_equivalence_l974_97496

theorem set_inclusion_equivalence (a : ℤ) : 
  let A := {x : ℝ | 2 * a + 1 ≤ x ∧ x ≤ 3 * a - 5}
  let B := {x : ℝ | 3 ≤ x ∧ x ≤ 32}
  (A ⊆ A ∩ B ∧ A.Nonempty) ↔ (6 ≤ a ∧ a ≤ 9) :=
by sorry

end set_inclusion_equivalence_l974_97496


namespace prob_two_target_rolls_l974_97429

/-- The number of sides on each die -/
def num_sides : ℕ := 7

/-- The sum we're aiming for -/
def target_sum : ℕ := 8

/-- The set of all possible outcomes when rolling two dice -/
def all_outcomes : Finset (ℕ × ℕ) :=
  Finset.product (Finset.range num_sides) (Finset.range num_sides)

/-- The set of outcomes that sum to the target -/
def favorable_outcomes : Finset (ℕ × ℕ) :=
  all_outcomes.filter (fun (a, b) => a + b + 2 = target_sum)

/-- The probability of rolling the target sum once -/
def prob_target : ℚ :=
  (favorable_outcomes.card : ℚ) / (all_outcomes.card : ℚ)

theorem prob_two_target_rolls : prob_target * prob_target = 1 / 49 := by
  sorry


end prob_two_target_rolls_l974_97429


namespace jose_bottle_caps_l974_97490

theorem jose_bottle_caps (initial : ℕ) (received : ℕ) (total : ℕ) : 
  initial = 7 → received = 2 → total = initial + received → total = 9 := by
  sorry

end jose_bottle_caps_l974_97490


namespace greatest_whole_number_inequality_l974_97481

theorem greatest_whole_number_inequality (x : ℤ) : 
  (∀ y : ℤ, y > x → ¬(6*y - 5 < 7 - 3*y)) → 
  (6*x - 5 < 7 - 3*x) → 
  x = 1 := by
sorry

end greatest_whole_number_inequality_l974_97481


namespace greeting_cards_group_size_l974_97419

theorem greeting_cards_group_size (n : ℕ) : 
  n * (n - 1) = 72 → n = 9 := by
  sorry

end greeting_cards_group_size_l974_97419


namespace power_function_through_point_l974_97499

-- Define a power function
def isPowerFunction (f : ℝ → ℝ) : Prop :=
  ∃ α : ℝ, ∀ x > 0, f x = x ^ α

-- State the theorem
theorem power_function_through_point (f : ℝ → ℝ) :
  isPowerFunction f → f 2 = Real.sqrt 2 → ∀ x > 0, f x = x ^ (1/2) := by
  sorry

end power_function_through_point_l974_97499


namespace notebooks_given_to_mike_l974_97408

theorem notebooks_given_to_mike (jack_original : ℕ) (gerald : ℕ) (to_paula : ℕ) (jack_final : ℕ) : 
  jack_original = gerald + 13 →
  gerald = 8 →
  to_paula = 5 →
  jack_final = 10 →
  jack_original - to_paula - jack_final = 6 := by
sorry

end notebooks_given_to_mike_l974_97408


namespace perpendicular_vectors_k_l974_97485

-- Define the vectors a and b
def a : Fin 2 → ℝ := ![1, 2]
def b : Fin 2 → ℝ := ![-3, 2]

-- Define the dot product of two 2D vectors
def dot_product (u v : Fin 2 → ℝ) : ℝ := (u 0) * (v 0) + (u 1) * (v 1)

-- Define the perpendicularity condition
def is_perpendicular (u v : Fin 2 → ℝ) : Prop := dot_product u v = 0

-- State the theorem
theorem perpendicular_vectors_k (k : ℝ) :
  is_perpendicular 
    (fun i => k * (a i) + (b i)) 
    (fun i => (a i) - 3 * (b i)) 
  → k = 19 := by sorry

end perpendicular_vectors_k_l974_97485


namespace special_gp_ratio_equation_special_gp_ratio_approx_l974_97442

/-- A geometric progression with positive terms where any term is equal to the sum of the next three following terms -/
structure SpecialGP where
  a : ℝ
  r : ℝ
  a_pos : a > 0
  r_pos : r > 0
  sum_property : ∀ n : ℕ, a * r^n = a * r^(n+1) + a * r^(n+2) + a * r^(n+3)

/-- The common ratio of a SpecialGP satisfies a cubic equation -/
theorem special_gp_ratio_equation (gp : SpecialGP) :
  gp.r^3 + gp.r^2 + gp.r - 1 = 0 := by
  sorry

/-- The solution to the cubic equation is approximately 0.5437 -/
theorem special_gp_ratio_approx (gp : SpecialGP) :
  ∃ ε > 0, |gp.r - 0.5437| < ε := by
  sorry

end special_gp_ratio_equation_special_gp_ratio_approx_l974_97442


namespace hyperbola_equation_l974_97446

/-- Given a hyperbola with equation x²/a² - y²/b² = 1 where a > 0 and b > 0,
    if one of its asymptotes is parallel to the line x + 3y + 2√5 = 0
    and one of its foci lies on this line, then a² = 18 and b² = 2 -/
theorem hyperbola_equation (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (∃ (x y : ℝ), x^2 / a^2 - y^2 / b^2 = 1) →
  (-b / a = -1 / 3) →
  (∃ (x : ℝ), x + 3 * 0 + 2 * Real.sqrt 5 = 0 ∧ x^2 = 4 * 5) →
  a^2 = 18 ∧ b^2 = 2 := by
  sorry

end hyperbola_equation_l974_97446


namespace study_time_for_desired_average_l974_97487

/-- Represents the relationship between study time and test score -/
structure StudyRelationship where
  time : ℝ
  score : ℝ
  k : ℝ
  inverse_prop : score * time = k

/-- Represents two tests with their study times and scores -/
structure TwoTests where
  test1 : StudyRelationship
  test2 : StudyRelationship
  avg_score : ℝ
  avg_constraint : (test1.score + test2.score) / 2 = avg_score

/-- The main theorem to prove -/
theorem study_time_for_desired_average (tests : TwoTests) :
  tests.test1.time = 6 ∧
  tests.test1.score = 80 ∧
  tests.avg_score = 85 →
  tests.test2.time = 16 / 3 :=
by sorry

end study_time_for_desired_average_l974_97487


namespace cone_lateral_surface_area_l974_97474

/-- The lateral surface area of a cone with base radius 3 and slant height 4 is 12π. -/
theorem cone_lateral_surface_area :
  ∀ (r l : ℝ), r = 3 → l = 4 → π * r * l = 12 * π :=
by
  sorry

end cone_lateral_surface_area_l974_97474


namespace union_of_sets_l974_97468

theorem union_of_sets (A B : Set ℤ) (hA : A = {0, 1}) (hB : B = {-1, 1}) :
  A ∪ B = {-1, 0, 1} := by
  sorry

end union_of_sets_l974_97468


namespace cubic_root_implies_h_value_l974_97448

theorem cubic_root_implies_h_value :
  ∀ h : ℝ, ((-3 : ℝ)^3 + h * (-3) - 18 = 0) → h = -15 := by
  sorry

end cubic_root_implies_h_value_l974_97448


namespace two_distinct_roots_range_l974_97492

theorem two_distinct_roots_range (a : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ 
    x^4 - 2*a*x^2 - x + a^2 - a = 0 ∧ 
    y^4 - 2*a*y^2 - y + a^2 - a = 0 ∧
    (∀ z : ℝ, z^4 - 2*a*z^2 - z + a^2 - a = 0 → z = x ∨ z = y)) →
  a > -1/4 ∧ a < 3/4 :=
sorry

end two_distinct_roots_range_l974_97492


namespace eighteen_gon_symmetry_sum_l974_97475

/-- A regular polygon with n sides -/
structure RegularPolygon (n : ℕ) where
  sides : n > 2

/-- The number of lines of symmetry in a regular polygon -/
def linesOfSymmetry (p : RegularPolygon n) : ℕ := n

/-- The smallest positive angle (in degrees) for rotational symmetry of a regular polygon -/
def rotationalSymmetryAngle (p : RegularPolygon n) : ℚ :=
  360 / n

theorem eighteen_gon_symmetry_sum :
  ∀ (p : RegularPolygon 18),
    (linesOfSymmetry p : ℚ) + rotationalSymmetryAngle p = 38 := by
  sorry

end eighteen_gon_symmetry_sum_l974_97475


namespace ice_cream_ratio_l974_97494

theorem ice_cream_ratio (sunday pints : ℕ) (k : ℕ) : 
  sunday = 4 →
  let monday := k * sunday
  let tuesday := monday / 3
  let wednesday := tuesday / 2
  18 = sunday + monday + tuesday - wednesday →
  monday / sunday = 3 := by sorry

end ice_cream_ratio_l974_97494


namespace g_negative_one_eq_three_l974_97452

/-- A polynomial function of degree 9 -/
noncomputable def g (d e f : ℝ) (x : ℝ) : ℝ := d * x^9 - e * x^5 + f * x + 1

/-- Theorem: If g(1) = -1, then g(-1) = 3 -/
theorem g_negative_one_eq_three {d e f : ℝ} (h : g d e f 1 = -1) : g d e f (-1) = 3 := by
  sorry

end g_negative_one_eq_three_l974_97452


namespace triangle_altitude_slopes_l974_97466

/-- Given a triangle ABC with vertices A(-1,0), B(1,1), and C(0,2),
    prove that the slopes of the altitudes on sides AB, AC, and BC
    are -2, -1/2, and 1 respectively. -/
theorem triangle_altitude_slopes :
  let A : ℝ × ℝ := (-1, 0)
  let B : ℝ × ℝ := (1, 1)
  let C : ℝ × ℝ := (0, 2)
  let slope (P Q : ℝ × ℝ) : ℝ := (Q.2 - P.2) / (Q.1 - P.1)
  let perpendicular_slope (m : ℝ) : ℝ := -1 / m
  (perpendicular_slope (slope A B) = -2) ∧
  (perpendicular_slope (slope A C) = -1/2) ∧
  (perpendicular_slope (slope B C) = 1) :=
by sorry

end triangle_altitude_slopes_l974_97466


namespace distance_covered_l974_97404

theorem distance_covered (time_minutes : ℝ) (speed_km_per_hour : ℝ) :
  time_minutes = 24 →
  speed_km_per_hour = 10 →
  (time_minutes / 60) * speed_km_per_hour = 4 :=
by sorry

end distance_covered_l974_97404


namespace xiao_ming_calculation_l974_97425

theorem xiao_ming_calculation (a : ℚ) : 
  (37 + 31 * a = 37 + 31 + a) → (a = 31 / 30) := by
  sorry

end xiao_ming_calculation_l974_97425


namespace factorization_example_l974_97418

theorem factorization_example : ∀ x : ℝ, x^2 - 1 = (x + 1) * (x - 1) := by
  sorry

end factorization_example_l974_97418


namespace tangent_parallel_points_l974_97484

-- Define the curve
def f (x : ℝ) : ℝ := x^3 + x - 2

-- Define the derivative of the curve
def f' (x : ℝ) : ℝ := 3*x^2 + 1

-- Theorem statement
theorem tangent_parallel_points :
  ∀ x y : ℝ, f x = y ∧ f' x = 4 ↔ (x = -1 ∧ y = -4) ∨ (x = 1 ∧ y = 0) := by
  sorry

end tangent_parallel_points_l974_97484


namespace max_volume_container_l974_97428

/-- Represents the dimensions of a rectangular container --/
structure ContainerDimensions where
  length : Real
  width : Real
  height : Real

/-- Calculates the volume of a rectangular container --/
def volume (d : ContainerDimensions) : Real :=
  d.length * d.width * d.height

/-- Represents the constraint of the total length of the steel bar --/
def totalLength (d : ContainerDimensions) : Real :=
  2 * (d.length + d.width) + 4 * d.height

/-- Theorem stating the maximum volume and corresponding height --/
theorem max_volume_container :
  ∃ (d : ContainerDimensions),
    totalLength d = 14.8 ∧
    d.length = d.width + 0.5 ∧
    d.height = 1.2 ∧
    volume d = 2.2 ∧
    ∀ (d' : ContainerDimensions),
      totalLength d' = 14.8 ∧ d'.length = d'.width + 0.5 →
      volume d' ≤ volume d :=
by sorry

end max_volume_container_l974_97428


namespace hanna_roses_to_friends_l974_97454

/-- Calculates the number of roses Hanna gives to her friends --/
def roses_given_to_friends (total_money : ℚ) (rose_price : ℚ) 
  (jenna_fraction : ℚ) (imma_fraction : ℚ) : ℚ :=
  let total_roses := total_money / rose_price
  let jenna_roses := jenna_fraction * total_roses
  let imma_roses := imma_fraction * total_roses
  jenna_roses + imma_roses

/-- Theorem stating the number of roses Hanna gives to her friends --/
theorem hanna_roses_to_friends : 
  roses_given_to_friends 300 2 (1/3) (1/2) = 125 := by
  sorry

end hanna_roses_to_friends_l974_97454


namespace red_square_density_l974_97415

/-- A standard rectangle is a rectangle on the coordinate plane with vertices at integer points and edges parallel to the coordinate axes. -/
def StandardRectangle (w h : ℕ) : Prop := sorry

/-- A unit square is a standard rectangle with an area of 1. -/
def UnitSquare : Prop := StandardRectangle 1 1

/-- A coloring of unit squares on the coordinate plane. -/
def Coloring := ℕ → ℕ → Bool

/-- The number of red squares in a standard rectangle. -/
def RedSquares (c : Coloring) (x y w h : ℕ) : ℕ := sorry

theorem red_square_density (a b : ℕ) (h1 : 0 < a) (h2 : a < b) (h3 : b < 2 * a) 
  (c : Coloring) (h4 : ∀ x y, RedSquares c x y a b + RedSquares c x y b a > 0) :
  ∀ N : ℕ, ∃ x y : ℕ, RedSquares c x y N N ≥ N * N * N / (N - 1) := by sorry

end red_square_density_l974_97415


namespace lcm_of_24_36_40_l974_97497

theorem lcm_of_24_36_40 : Nat.lcm (Nat.lcm 24 36) 40 = 360 := by
  sorry

end lcm_of_24_36_40_l974_97497
