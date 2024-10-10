import Mathlib

namespace geometric_progression_fourth_term_l1286_128600

/-- Given a geometric progression with the first three terms, find the fourth term -/
theorem geometric_progression_fourth_term 
  (a : ℝ) 
  (r : ℝ) 
  (h1 : a = 5^(1/3 : ℝ)) 
  (h2 : a * r = 5^(1/5 : ℝ)) 
  (h3 : a * r^2 = 5^(1/15 : ℝ)) : 
  a * r^3 = 5^(-1/15 : ℝ) := by
sorry

end geometric_progression_fourth_term_l1286_128600


namespace seller_took_weight_l1286_128601

/-- Given 10 weights with masses n, n+1, ..., n+9, if the sum of 9 of these weights is 1457,
    then the missing weight is 158. -/
theorem seller_took_weight (n : ℕ) (x : ℕ) (h1 : x ≤ 9) 
    (h2 : (10 * n + 45) - (n + x) = 1457) : n + x = 158 := by
  sorry

end seller_took_weight_l1286_128601


namespace points_collinear_l1286_128628

/-- Three points are collinear if the slope between any two pairs of points is equal. -/
def collinear (p1 p2 p3 : ℝ × ℝ) : Prop :=
  let (x1, y1) := p1
  let (x2, y2) := p2
  let (x3, y3) := p3
  (y2 - y1) * (x3 - x2) = (y3 - y2) * (x2 - x1)

/-- The three given points are collinear. -/
theorem points_collinear : collinear (2, 5) (-6, -3) (-4, -1) := by
  sorry


end points_collinear_l1286_128628


namespace orange_apple_weight_equivalence_l1286_128626

/-- Given that 7 oranges weigh the same as 5 apples, prove that 28 oranges weigh the same as 20 apples. -/
theorem orange_apple_weight_equivalence :
  ∀ (orange_weight apple_weight : ℝ),
    orange_weight > 0 →
    apple_weight > 0 →
    7 * orange_weight = 5 * apple_weight →
    28 * orange_weight = 20 * apple_weight :=
by sorry

end orange_apple_weight_equivalence_l1286_128626


namespace camping_cost_equalization_l1286_128672

theorem camping_cost_equalization 
  (X Y Z : ℝ) 
  (h_order : X < Y ∧ Y < Z) :
  let total_cost := X + Y + Z
  let equal_share := total_cost / 3
  (equal_share - X) = (Y + Z - 2 * X) / 3 := by
sorry

end camping_cost_equalization_l1286_128672


namespace cats_adopted_count_l1286_128674

/-- The cost to get a cat ready for adoption -/
def cat_cost : ℕ := 50

/-- The cost to get an adult dog ready for adoption -/
def adult_dog_cost : ℕ := 100

/-- The cost to get a puppy ready for adoption -/
def puppy_cost : ℕ := 150

/-- The number of adult dogs adopted -/
def adult_dogs_adopted : ℕ := 3

/-- The number of puppies adopted -/
def puppies_adopted : ℕ := 2

/-- The total cost for all adopted animals -/
def total_cost : ℕ := 700

/-- Theorem stating that the number of cats adopted is 2 -/
theorem cats_adopted_count : 
  ∃ (c : ℕ), c * cat_cost + adult_dogs_adopted * adult_dog_cost + puppies_adopted * puppy_cost = total_cost ∧ c = 2 :=
by sorry

end cats_adopted_count_l1286_128674


namespace alice_plate_stacking_l1286_128639

theorem alice_plate_stacking (initial_plates : ℕ) (first_addition : ℕ) (total_plates : ℕ) : 
  initial_plates = 27 → 
  first_addition = 37 → 
  total_plates = 83 → 
  total_plates - (initial_plates + first_addition) = 19 := by
sorry

end alice_plate_stacking_l1286_128639


namespace integer_sum_problem_l1286_128620

theorem integer_sum_problem (x y : ℤ) : 
  x > 0 → y > 0 → x - y = 8 → x * y = 120 → x + y = 4 * Real.sqrt 34 := by
  sorry

end integer_sum_problem_l1286_128620


namespace statistical_relationships_properties_l1286_128675

-- Define the basic concepts
def FunctionalRelationship : Type := Unit
def DeterministicRelationship : Type := Unit
def Correlation : Type := Unit
def NonDeterministicRelationship : Type := Unit
def RegressionAnalysis : Type := Unit
def StatisticalAnalysisMethod : Type := Unit
def TwoVariables : Type := Unit

-- Define the properties
def isDeterministic (r : FunctionalRelationship) : Prop := sorry
def isNonDeterministic (c : Correlation) : Prop := sorry
def isUsedFor (m : StatisticalAnalysisMethod) (v : TwoVariables) (c : Correlation) : Prop := sorry

-- Theorem to prove
theorem statistical_relationships_properties :
  (∀ (r : FunctionalRelationship), isDeterministic r) ∧
  (∀ (c : Correlation), isNonDeterministic c) ∧
  (∃ (m : RegressionAnalysis) (v : TwoVariables) (c : Correlation), 
    isUsedFor m v c) :=
by sorry

end statistical_relationships_properties_l1286_128675


namespace ufo_convention_attendees_l1286_128648

theorem ufo_convention_attendees (total : ℕ) (male : ℕ) 
  (h1 : total = 120) 
  (h2 : male = 62) 
  (h3 : male > total - male) : 
  male - (total - male) = 4 := by
  sorry

end ufo_convention_attendees_l1286_128648


namespace go_stones_problem_l1286_128615

theorem go_stones_problem (x : ℕ) (h1 : (x / 7 + 40) * 5 = 555) (h2 : x ≥ 55) : x - 55 = 442 := by
  sorry

end go_stones_problem_l1286_128615


namespace elevator_floors_l1286_128608

/-- The number of floors the elevator needs to move down. -/
def total_floors : ℕ := sorry

/-- The time taken for the first half of the floors (in minutes). -/
def first_half_time : ℕ := 15

/-- The time taken per floor for the next 5 floors (in minutes). -/
def middle_time_per_floor : ℕ := 5

/-- The number of floors in the middle section. -/
def middle_floors : ℕ := 5

/-- The time taken per floor for the final 5 floors (in minutes). -/
def final_time_per_floor : ℕ := 16

/-- The number of floors in the final section. -/
def final_floors : ℕ := 5

/-- The total time taken to reach the bottom (in minutes). -/
def total_time : ℕ := 120

theorem elevator_floors :
  first_half_time + 
  (middle_time_per_floor * middle_floors) + 
  (final_time_per_floor * final_floors) = total_time ∧
  total_floors = (total_floors / 2) + middle_floors + final_floors ∧
  total_floors = 20 := by sorry

end elevator_floors_l1286_128608


namespace min_square_difference_of_roots_l1286_128676

theorem min_square_difference_of_roots (α β b : ℝ) : 
  α^2 + 2*b*α + b = 1 → β^2 + 2*b*β + b = 1 → 
  ∀ γ δ c : ℝ, (γ^2 + 2*c*γ + c = 1 ∧ δ^2 + 2*c*δ + c = 1) → 
  (α - β)^2 ≥ 3 ∧ (∃ e : ℝ, (α - β)^2 = 3) :=
by sorry

end min_square_difference_of_roots_l1286_128676


namespace special_triangle_area_special_triangle_area_is_48_l1286_128650

/-- A triangle with two sides of length 10 and 12, and a median to the third side of length 5 -/
structure SpecialTriangle where
  side1 : ℝ
  side2 : ℝ
  median : ℝ
  h_side1 : side1 = 10
  h_side2 : side2 = 12
  h_median : median = 5

/-- The area of a SpecialTriangle is 48 -/
theorem special_triangle_area (t : SpecialTriangle) : ℝ :=
  48

/-- The area of a SpecialTriangle is indeed 48 -/
theorem special_triangle_area_is_48 (t : SpecialTriangle) :
  special_triangle_area t = 48 := by
  sorry

end special_triangle_area_special_triangle_area_is_48_l1286_128650


namespace square_perimeter_l1286_128647

theorem square_perimeter (rectangle_width : ℝ) (rectangle_length : ℝ) 
  (h1 : rectangle_length = 4 * rectangle_width) 
  (h2 : 28 * rectangle_width = 56) : 
  4 * (rectangle_width + rectangle_length) = 32 := by
  sorry

end square_perimeter_l1286_128647


namespace episode_length_l1286_128635

/-- Given a TV mini series with 6 episodes and a total watching time of 5 hours,
    prove that the length of each episode is 50 minutes. -/
theorem episode_length (num_episodes : ℕ) (total_time : ℕ) : 
  num_episodes = 6 → total_time = 5 * 60 → total_time / num_episodes = 50 := by
  sorry

end episode_length_l1286_128635


namespace zach_stadium_goal_l1286_128655

/-- The number of stadiums Zach wants to visit --/
def num_stadiums : ℕ := 30

/-- The cost per stadium in dollars --/
def cost_per_stadium : ℕ := 900

/-- Zach's yearly savings in dollars --/
def yearly_savings : ℕ := 1500

/-- The number of years to accomplish the goal --/
def years_to_goal : ℕ := 18

/-- Theorem stating that the number of stadiums Zach wants to visit is 30 --/
theorem zach_stadium_goal :
  num_stadiums = (yearly_savings * years_to_goal) / cost_per_stadium :=
by sorry

end zach_stadium_goal_l1286_128655


namespace pizza_portion_eaten_l1286_128698

theorem pizza_portion_eaten (total_slices : ℕ) (slices_left : ℕ) :
  total_slices = 16 → slices_left = 4 →
  (total_slices - slices_left : ℚ) / total_slices = 3/4 := by
  sorry

end pizza_portion_eaten_l1286_128698


namespace speed_conversion_l1286_128686

/-- Conversion factor from m/s to km/h -/
def mps_to_kmph : ℝ := 3.6

/-- Given speed in meters per second -/
def speed_mps : ℝ := 20.0016

/-- Speed in kilometers per hour to be proven -/
def speed_kmph : ℝ := 72.00576

/-- Theorem stating that the given speed in km/h is equivalent to the speed in m/s -/
theorem speed_conversion : speed_kmph = speed_mps * mps_to_kmph := by
  sorry

end speed_conversion_l1286_128686


namespace rationalize_denominator_l1286_128693

theorem rationalize_denominator :
  (Real.sqrt 12 + Real.sqrt 5) / (Real.sqrt 3 + Real.sqrt 5) = (Real.sqrt 15 - 1) / 2 := by
  sorry

end rationalize_denominator_l1286_128693


namespace intersection_of_A_and_B_l1286_128642

-- Define the sets A and B
def A : Set ℝ := {x | x^2 - 4 < 0}
def B : Set ℝ := {x | x < 0}

-- State the theorem
theorem intersection_of_A_and_B :
  A ∩ B = {x : ℝ | -2 < x ∧ x < 0} := by sorry

end intersection_of_A_and_B_l1286_128642


namespace system_solution_l1286_128685

theorem system_solution (x y z : ℝ) : 
  (x * y * z) / (x + y) = 6/5 ∧ 
  (x * y * z) / (y + z) = 2 ∧ 
  (x * y * z) / (z + x) = 3/2 →
  ((x = 3 ∧ y = 2 ∧ z = 1) ∨ (x = -3 ∧ y = -2 ∧ z = -1)) :=
by sorry

end system_solution_l1286_128685


namespace max_value_of_expression_l1286_128684

theorem max_value_of_expression (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  let A := (a^4 + b^4 + c^4) / ((a + b + c)^4 - 80*(a*b*c)^(4/3))
  A ≤ 3 ∧ ∃ (x : ℝ), x > 0 ∧ 
    let A' := (x^4 + x^4 + x^4) / ((x + x + x)^4 - 80*(x*x*x)^(4/3))
    A' = 3 :=
by sorry

end max_value_of_expression_l1286_128684


namespace equilateral_iff_rhombus_l1286_128683

-- Define a parallelogram
structure Parallelogram :=
  (sides : Fin 4 → ℝ)
  (is_parallelogram : sides 0 = sides 2 ∧ sides 1 = sides 3)

-- Define an equilateral parallelogram
def is_equilateral (p : Parallelogram) : Prop :=
  p.sides 0 = p.sides 1 ∧ p.sides 1 = p.sides 2 ∧ p.sides 2 = p.sides 3

-- Define a rhombus
def is_rhombus (p : Parallelogram) : Prop :=
  p.sides 0 = p.sides 1 ∧ p.sides 1 = p.sides 2 ∧ p.sides 2 = p.sides 3

-- Theorem: A parallelogram is equilateral if and only if it is a rhombus
theorem equilateral_iff_rhombus (p : Parallelogram) :
  is_equilateral p ↔ is_rhombus p :=
sorry

end equilateral_iff_rhombus_l1286_128683


namespace range_of_m2_plus_n2_l1286_128696

-- Define a decreasing function on ℝ
def DecreasingFunction (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x > f y

-- Define the theorem
theorem range_of_m2_plus_n2
  (f : ℝ → ℝ)
  (h_decreasing : DecreasingFunction f)
  (h_inequality : ∀ m n : ℝ, f (n^2 - 10*n - 15) ≥ f (12 - m^2 + 24*m)) :
  ∀ m n : ℝ, 0 ≤ m^2 + n^2 ∧ m^2 + n^2 ≤ 729 :=
sorry

end range_of_m2_plus_n2_l1286_128696


namespace spongebob_burger_price_l1286_128695

/-- The price of a burger in Spongebob's shop -/
def burger_price : ℝ := 2

/-- The number of burgers sold -/
def burgers_sold : ℕ := 30

/-- The number of large fries sold -/
def fries_sold : ℕ := 12

/-- The price of each large fries -/
def fries_price : ℝ := 1.5

/-- The total earnings for the day -/
def total_earnings : ℝ := 78

theorem spongebob_burger_price :
  burger_price * burgers_sold + fries_price * fries_sold = total_earnings :=
by sorry

end spongebob_burger_price_l1286_128695


namespace square_root_of_2m_minus_n_is_2_l1286_128622

theorem square_root_of_2m_minus_n_is_2 
  (m n : ℝ) 
  (eq1 : m * 2 + n * 1 = 8) 
  (eq2 : n * 2 - m * 1 = 1) : 
  Real.sqrt (2 * m - n) = 2 := by
  sorry

end square_root_of_2m_minus_n_is_2_l1286_128622


namespace not_5x_representation_l1286_128678

-- Define the expressions
def expr_A (x : ℝ) : ℝ := 5 * x
def expr_B (x : ℝ) : ℝ := x^5
def expr_C (x : ℝ) : ℝ := x + x + x + x + x

-- Theorem stating that B is not equal to 5x, while A and C are
theorem not_5x_representation (x : ℝ) : 
  expr_A x = 5 * x ∧ expr_C x = 5 * x ∧ expr_B x ≠ 5 * x :=
sorry

end not_5x_representation_l1286_128678


namespace consecutive_products_divisibility_l1286_128679

theorem consecutive_products_divisibility (a : ℤ) :
  ∃ k : ℤ, a * (a + 1) + (a + 1) * (a + 2) + (a + 2) * (a + 3) + a * (a + 3) + 1 = 12 * k :=
by sorry

end consecutive_products_divisibility_l1286_128679


namespace trivia_team_groups_l1286_128613

theorem trivia_team_groups (total_students : ℕ) (not_picked : ℕ) (students_per_group : ℕ) :
  total_students = 64 →
  not_picked = 36 →
  students_per_group = 7 →
  (total_students - not_picked) / students_per_group = 4 :=
by
  sorry

end trivia_team_groups_l1286_128613


namespace timeDifference_div_by_40_l1286_128607

/-- Represents time in days, hours, minutes, and seconds -/
structure Time where
  days : ℕ
  hours : ℕ
  minutes : ℕ
  seconds : ℕ

/-- Converts Time to its numerical representation (ignoring punctuation) -/
def Time.toNumerical (t : Time) : ℕ :=
  10^6 * t.days + 10^4 * t.hours + 100 * t.minutes + t.seconds

/-- Converts Time to total seconds -/
def Time.toSeconds (t : Time) : ℕ :=
  86400 * t.days + 3600 * t.hours + 60 * t.minutes + t.seconds

/-- The difference between numerical representation and total seconds -/
def timeDifference (t : Time) : ℤ :=
  (t.toNumerical : ℤ) - (t.toSeconds : ℤ)

/-- Theorem: 40 always divides the time difference -/
theorem timeDifference_div_by_40 (t : Time) : 
  (40 : ℤ) ∣ timeDifference t := by
  sorry

end timeDifference_div_by_40_l1286_128607


namespace sum_of_repeating_decimals_l1286_128671

/-- The sum of the repeating decimals 0.4̄ and 0.26̄ is equal to 70/99 -/
theorem sum_of_repeating_decimals : 
  (∃ (x y : ℚ), x = 4/9 ∧ y = 26/99 ∧ x + y = 70/99) := by
  sorry

end sum_of_repeating_decimals_l1286_128671


namespace system_solution_l1286_128694

theorem system_solution (a b c x y z : ℝ) : 
  (a * x + (a - b) * y + (a - c) * z = a^2 + (b - c)^2) ∧
  ((b - a) * x + b * y + (b - c) * z = b^2 + (c - a)^2) ∧
  ((c - a) * x + (c - b) * y + c * z = c^2 + (a - b)^2) →
  (x = b + c - a ∧ y = c + a - b ∧ z = a + b - c) :=
by sorry

end system_solution_l1286_128694


namespace arithmetic_sequence_100th_term_nth_term_is_298_implies_n_is_100_l1286_128667

/-- An arithmetic sequence with first term 1 and common difference 3 -/
def arithmetic_sequence (n : ℕ) : ℤ :=
  1 + 3 * (n - 1)

/-- Theorem stating that the 100th term of the sequence is 298 -/
theorem arithmetic_sequence_100th_term :
  arithmetic_sequence 100 = 298 :=
sorry

/-- Theorem proving that when the nth term is 298, n must be 100 -/
theorem nth_term_is_298_implies_n_is_100 (n : ℕ) :
  arithmetic_sequence n = 298 → n = 100 :=
sorry

end arithmetic_sequence_100th_term_nth_term_is_298_implies_n_is_100_l1286_128667


namespace largest_divisor_of_consecutive_odd_product_l1286_128621

theorem largest_divisor_of_consecutive_odd_product :
  ∀ n : ℕ, n > 0 →
  ∃ m : ℕ, m > 0 ∧
  (∀ k : ℕ, k > m → ¬(k ∣ ((2*n - 1) * (2*n + 1) * (2*n + 3)))) ∧
  (3 ∣ ((2*n - 1) * (2*n + 1) * (2*n + 3))) :=
by sorry

end largest_divisor_of_consecutive_odd_product_l1286_128621


namespace submerged_sphere_segment_height_l1286_128603

/-- 
Theorem: For a homogeneous spherical segment of radius r floating in water, 
the height of the submerged portion m is equal to r/2 * (3 - √5) when it 
submerges up to the edge of its base spherical cap.
-/
theorem submerged_sphere_segment_height 
  (r : ℝ) -- radius of the sphere
  (h_pos : r > 0) -- assumption that radius is positive
  : ∃ m : ℝ, 
    -- m is the height of the submerged portion
    -- Volume of spherical sector
    (2 * π * m^3 / 3 = 
    -- Volume of submerged spherical segment
    π * m^2 * (3*r - m) / 3) ∧ 
    -- m is less than r (physical constraint)
    m < r ∧ 
    -- m equals the derived formula
    m = r/2 * (3 - Real.sqrt 5) := by
  sorry

end submerged_sphere_segment_height_l1286_128603


namespace principal_proof_l1286_128697

/-- The principal amount that satisfies the given conditions -/
def principal_amount : ℝ := by sorry

theorem principal_proof :
  let R : ℝ := 0.05  -- Interest rate (5% per annum)
  let T : ℝ := 10    -- Time period in years
  let P : ℝ := principal_amount
  let I : ℝ := P * R * T  -- Interest calculation
  (P - I = P - 3100) →  -- Interest is 3100 less than principal
  P = 6200 := by sorry

end principal_proof_l1286_128697


namespace factor_3x_squared_minus_75_l1286_128673

theorem factor_3x_squared_minus_75 (x : ℝ) : 3 * x^2 - 75 = 3 * (x + 5) * (x - 5) := by
  sorry

end factor_3x_squared_minus_75_l1286_128673


namespace tan_ratio_max_tan_A_l1286_128646

-- Define a triangle ABC
structure Triangle :=
  (a b c : ℝ)
  (A B C : ℝ)

-- Define the conditions
def triangle_condition (t : Triangle) : Prop :=
  t.b^2 + 3*t.a^2 = t.c^2

-- Theorem 1: tan(C) / tan(B) = -2
theorem tan_ratio (t : Triangle) (h : triangle_condition t) :
  Real.tan t.C / Real.tan t.B = -2 := by sorry

-- Theorem 2: Maximum value of tan(A) is √2/4
theorem max_tan_A (t : Triangle) (h : triangle_condition t) :
  ∃ (max_tan_A : ℝ), (∀ (t' : Triangle), triangle_condition t' → Real.tan t'.A ≤ max_tan_A) ∧ max_tan_A = Real.sqrt 2 / 4 := by sorry

end tan_ratio_max_tan_A_l1286_128646


namespace workshop_workers_correct_l1286_128611

/-- The number of workers in a workshop with given salary conditions -/
def workshop_workers : ℕ :=
  let average_salary : ℚ := 750
  let technician_count : ℕ := 5
  let technician_salary : ℚ := 900
  let non_technician_salary : ℚ := 700
  20

/-- Proof that the number of workers in the workshop is correct -/
theorem workshop_workers_correct :
  let average_salary : ℚ := 750
  let technician_count : ℕ := 5
  let technician_salary : ℚ := 900
  let non_technician_salary : ℚ := 700
  let total_workers := workshop_workers
  (average_salary * total_workers : ℚ) =
    technician_salary * technician_count +
    non_technician_salary * (total_workers - technician_count) :=
by
  sorry

#eval workshop_workers

end workshop_workers_correct_l1286_128611


namespace smallest_part_of_proportional_division_l1286_128649

theorem smallest_part_of_proportional_division (total : ℕ) (a b c : ℕ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  total = 90 ∧ a = 3 ∧ b = 5 ∧ c = 7 →
  ∃ x : ℚ, x > 0 ∧ total = a * x + b * x + c * x ∧ min (a * x) (min (b * x) (c * x)) = 18 :=
by sorry

end smallest_part_of_proportional_division_l1286_128649


namespace linear_combination_of_reals_with_rational_products_l1286_128660

theorem linear_combination_of_reals_with_rational_products 
  (a b c : ℝ) 
  (hab : ∃ (q : ℚ), a * b = q) 
  (hbc : ∃ (q : ℚ), b * c = q) 
  (hca : ∃ (q : ℚ), c * a = q) 
  (hnz : a ≠ 0 ∨ b ≠ 0 ∨ c ≠ 0) : 
  ∃ (x y z : ℤ), a * (x : ℝ) + b * (y : ℝ) + c * (z : ℝ) = 0 := by
sorry

end linear_combination_of_reals_with_rational_products_l1286_128660


namespace line_segment_endpoint_l1286_128614

theorem line_segment_endpoint (x : ℝ) : 
  x > 0 → 
  ((4 - 1)^2 + (x - 3)^2 : ℝ) = 5^2 → 
  x = 7 := by
sorry

end line_segment_endpoint_l1286_128614


namespace max_value_of_product_sum_l1286_128641

theorem max_value_of_product_sum (a b c d : ℝ) : 
  a ≥ 0 → b ≥ 0 → c ≥ 0 → d ≥ 0 → a + b + c + d = 200 → 
  a * b + b * c + c * d ≤ 2500 := by
sorry

end max_value_of_product_sum_l1286_128641


namespace complex_division_simplification_l1286_128638

theorem complex_division_simplification : 
  let i : ℂ := Complex.I
  (2 - 3 * i) / (1 + i) = -1/2 - 5/2 * i := by sorry

end complex_division_simplification_l1286_128638


namespace polynomial_root_sum_l1286_128665

theorem polynomial_root_sum (a b c d e : ℤ) : 
  let g : ℝ → ℝ := λ x => x^5 + a*x^4 + b*x^3 + c*x^2 + d*x + e
  (∀ r : ℝ, g r = 0 → ∃ k : ℤ, r = -k ∧ k > 0) →
  a + b + c + d + e = 3403 →
  e = 9240 := by
sorry

end polynomial_root_sum_l1286_128665


namespace absolute_difference_l1286_128677

theorem absolute_difference (a x : ℝ) (h1 : a < 0) (h2 : |a| * x ≤ a) : 
  |x + 1| - |x - 2| = -3 := by
sorry

end absolute_difference_l1286_128677


namespace two_x_equals_y_l1286_128630

theorem two_x_equals_y (x y : ℝ) 
  (h1 : (x + y) / 3 = 1) 
  (h2 : x + 2*y = 5) : 
  2*x = y := by sorry

end two_x_equals_y_l1286_128630


namespace cos_alpha_value_l1286_128629

/-- Given an angle α in the second quadrant, if the slope of the line 2x + (tan α)y + 1 = 0 is 8/3, 
    then cos α = -4/5 -/
theorem cos_alpha_value (α : Real) : 
  (π/2 < α ∧ α < π) →  -- α is in the second quadrant
  (-(2 : Real) / Real.tan α = 8/3) →  -- slope of the line
  Real.cos α = -4/5 := by
sorry

end cos_alpha_value_l1286_128629


namespace nathan_tokens_used_l1286_128663

/-- The number of tokens Nathan used at the arcade -/
def tokens_used (air_hockey_games basketball_games tokens_per_game : ℕ) : ℕ :=
  (air_hockey_games + basketball_games) * tokens_per_game

/-- Theorem: Nathan used 18 tokens at the arcade -/
theorem nathan_tokens_used :
  tokens_used 2 4 3 = 18 := by
  sorry

end nathan_tokens_used_l1286_128663


namespace cube_volume_from_surface_area_l1286_128631

theorem cube_volume_from_surface_area (surface_area : ℝ) (volume : ℝ) :
  surface_area = 150 →
  (6 : ℝ) * (volume ^ (1/3 : ℝ))^2 = surface_area →
  volume = 125 := by
sorry

end cube_volume_from_surface_area_l1286_128631


namespace combination_number_identity_l1286_128618

theorem combination_number_identity (n r : ℕ) (h1 : n > r) (h2 : r ≥ 1) :
  Nat.choose n r = (n / r) * Nat.choose (n - 1) (r - 1) := by
  sorry

end combination_number_identity_l1286_128618


namespace parabola_directrix_l1286_128651

/-- Given a parabola with equation y = 4x^2 - 6, its directrix has equation y = -97/16 -/
theorem parabola_directrix (x y : ℝ) :
  y = 4 * x^2 - 6 → ∃ (k : ℝ), k = -97/16 ∧ (∀ (x₀ y₀ : ℝ), y₀ = 4 * x₀^2 - 6 → y₀ - k = (x₀ - 0)^2 + (y₀ - (k + 1/4))^2) :=
by sorry

end parabola_directrix_l1286_128651


namespace triangle_ABC_angle_proof_l1286_128612

def triangle_ABC_angle (A B C : ℝ × ℝ) : Prop :=
  let BA : ℝ × ℝ := (Real.sqrt 3, 1)
  let BC : ℝ × ℝ := (0, 1)
  let AB : ℝ × ℝ := (-BA.1, -BA.2)
  let angle := Real.arccos (AB.1 * BC.1 + AB.2 * BC.2) / 
               (Real.sqrt (AB.1^2 + AB.2^2) * Real.sqrt (BC.1^2 + BC.2^2))
  angle = 2 * Real.pi / 3

theorem triangle_ABC_angle_proof (A B C : ℝ × ℝ) : 
  triangle_ABC_angle A B C := by sorry

end triangle_ABC_angle_proof_l1286_128612


namespace arrangements_count_l1286_128688

/-- The number of ways to arrange four people in a row with one person not at the ends -/
def arrangements_with_restriction : ℕ :=
  let total_people : ℕ := 4
  let restricted_person : ℕ := 1
  let unrestricted_people : ℕ := total_people - restricted_person
  let unrestricted_arrangements : ℕ := Nat.factorial unrestricted_people
  let valid_positions : ℕ := unrestricted_people - 1
  unrestricted_arrangements * valid_positions

theorem arrangements_count :
  arrangements_with_restriction = 12 :=
sorry

end arrangements_count_l1286_128688


namespace avery_wall_time_l1286_128658

/-- The time it takes Avery to build the wall alone -/
def avery_time : ℝ := 4

/-- The time it takes Tom to build the wall alone -/
def tom_time : ℝ := 2

/-- The additional time Tom needs to finish the wall after working with Avery for 1 hour -/
def tom_additional_time : ℝ := 0.5

theorem avery_wall_time : 
  (1 / avery_time + 1 / tom_time) + tom_additional_time / tom_time = 1 := by sorry

end avery_wall_time_l1286_128658


namespace existence_of_polynomials_l1286_128606

theorem existence_of_polynomials : ∃ (p q : Polynomial ℤ),
  (∃ (i j : ℕ), (abs (p.coeff i) > 2015) ∧ (abs (q.coeff j) > 2015)) ∧
  (∀ k : ℕ, abs ((p * q).coeff k) ≤ 1) := by
  sorry

end existence_of_polynomials_l1286_128606


namespace adjacent_sum_theorem_l1286_128636

/-- Represents a 3x3 table with numbers from 1 to 9 -/
def Table := Fin 3 → Fin 3 → Fin 9

/-- Checks if a table contains each number from 1 to 9 exactly once -/
def isValidTable (t : Table) : Prop :=
  ∀ n : Fin 9, ∃! (i j : Fin 3), t i j = n

/-- Checks if the table has 1, 2, 3, and 4 in the correct positions -/
def hasCorrectCorners (t : Table) : Prop :=
  t 0 0 = 0 ∧ t 2 0 = 1 ∧ t 0 2 = 2 ∧ t 2 2 = 3

/-- Returns the sum of adjacent numbers to the given position -/
def adjacentSum (t : Table) (i j : Fin 3) : Nat :=
  (if i > 0 then (t (i-1) j).val + 1 else 0) +
  (if i < 2 then (t (i+1) j).val + 1 else 0) +
  (if j > 0 then (t i (j-1)).val + 1 else 0) +
  (if j < 2 then (t i (j+1)).val + 1 else 0)

/-- The main theorem to prove -/
theorem adjacent_sum_theorem (t : Table) 
  (valid : isValidTable t) 
  (corners : hasCorrectCorners t) 
  (sum_5 : ∃ i j : Fin 3, t i j = 4 ∧ adjacentSum t i j = 9) :
  ∃ i j : Fin 3, t i j = 5 ∧ adjacentSum t i j = 29 := by
  sorry

end adjacent_sum_theorem_l1286_128636


namespace perpendicular_vectors_l1286_128625

def a : ℝ × ℝ := (2, 1)
def b (x : ℝ) : ℝ × ℝ := (3, x)

theorem perpendicular_vectors (x : ℝ) : 
  (a.1 * (b x).1 + a.2 * (b x).2 = 0) → x = -6 := by
  sorry

end perpendicular_vectors_l1286_128625


namespace abs_equation_solution_difference_l1286_128691

theorem abs_equation_solution_difference : ∃ x₁ x₂ : ℝ, 
  (|x₁ - 4| = 12 ∧ |x₂ - 4| = 12 ∧ x₁ ≠ x₂) ∧ |x₁ - x₂| = 24 :=
by
  sorry

end abs_equation_solution_difference_l1286_128691


namespace modulus_of_z_l1286_128643

theorem modulus_of_z (z : ℂ) (h : (z - Complex.I) * Complex.I = 1 + Complex.I) : Complex.abs z = 1 := by
  sorry

end modulus_of_z_l1286_128643


namespace arrange_40555_l1286_128689

def digit_arrangements (n : ℕ) : ℕ := 
  if n = 40555 then 12 else 0

theorem arrange_40555 :
  digit_arrangements 40555 = 12 ∧
  (∀ x : ℕ, x ≠ 40555 → digit_arrangements x = 0) :=
sorry

end arrange_40555_l1286_128689


namespace quadratic_inequality_solution_set_l1286_128690

theorem quadratic_inequality_solution_set (a : ℝ) : 
  (∀ x : ℝ, x^2 - a*x < 0 ↔ 0 < x ∧ x < 1) → a = 1 := by
  sorry

end quadratic_inequality_solution_set_l1286_128690


namespace initial_girls_count_l1286_128692

theorem initial_girls_count (b g : ℕ) : 
  (3 * (g - 18) = b) →
  (4 * (b - 36) = g - 18) →
  g = 31 :=
by sorry

end initial_girls_count_l1286_128692


namespace vote_change_theorem_l1286_128666

/-- Represents the voting results of an assembly --/
structure VotingResults where
  total_members : ℕ
  initial_for : ℕ
  initial_against : ℕ
  revote_for : ℕ
  revote_against : ℕ

/-- Theorem about the change in votes for a resolution --/
theorem vote_change_theorem (v : VotingResults) : 
  v.total_members = 500 →
  v.initial_for + v.initial_against = v.total_members →
  v.revote_for + v.revote_against = v.total_members →
  v.initial_against > v.initial_for →
  v.revote_for > v.revote_against →
  (v.revote_for - v.revote_against) = 3 * (v.initial_against - v.initial_for) →
  v.revote_for = (7 * v.initial_against) / 6 →
  v.revote_for - v.initial_for = 90 := by
  sorry


end vote_change_theorem_l1286_128666


namespace fine_calculation_l1286_128662

/-- Calculates the fine for inappropriate items in the recycling bin -/
def calculate_fine (weeks : ℕ) (trash_bin_cost : ℚ) (recycling_bin_cost : ℚ) 
  (trash_bins : ℕ) (recycling_bins : ℕ) (discount_percent : ℚ) (total_bill : ℚ) : ℚ := 
  let weekly_cost := trash_bin_cost * trash_bins + recycling_bin_cost * recycling_bins
  let monthly_cost := weekly_cost * weeks
  let discount := discount_percent * monthly_cost
  let discounted_cost := monthly_cost - discount
  total_bill - discounted_cost

theorem fine_calculation :
  calculate_fine 4 10 5 2 1 (18/100) 102 = 20 := by
  sorry

end fine_calculation_l1286_128662


namespace liter_milliliter_comparison_l1286_128670

theorem liter_milliliter_comparison : ¬(1000 < 9000 / 1000) := by
  sorry

end liter_milliliter_comparison_l1286_128670


namespace intersection_nonempty_iff_m_leq_neg_one_l1286_128653

/-- Sets A and B are defined as follows:
    A = {(x, y) | y = x^2 + mx + 2}
    B = {(x, y) | x - y + 1 = 0 and 0 ≤ x ≤ 2}
    This theorem states that A ∩ B is non-empty if and only if m ≤ -1 -/
theorem intersection_nonempty_iff_m_leq_neg_one (m : ℝ) :
  (∃ x y : ℝ, y = x^2 + m*x + 2 ∧ x - y + 1 = 0 ∧ 0 ≤ x ∧ x ≤ 2) ↔ m ≤ -1 := by
  sorry

end intersection_nonempty_iff_m_leq_neg_one_l1286_128653


namespace farm_area_is_1200_l1286_128652

/-- Represents a rectangular farm with fencing on one long side, one short side, and the diagonal -/
structure RectangularFarm where
  short_side : ℝ
  long_side : ℝ
  diagonal : ℝ
  fencing_cost_per_meter : ℝ
  total_fencing_cost : ℝ

/-- Calculates the area of a rectangular farm -/
def farm_area (farm : RectangularFarm) : ℝ :=
  farm.short_side * farm.long_side

/-- Calculates the total length of fencing required -/
def total_fencing_length (farm : RectangularFarm) : ℝ :=
  farm.short_side + farm.long_side + farm.diagonal

/-- The main theorem: If a rectangular farm satisfies the given conditions, its area is 1200 square meters -/
theorem farm_area_is_1200 (farm : RectangularFarm) 
    (h1 : farm.short_side = 30)
    (h2 : farm.fencing_cost_per_meter = 13)
    (h3 : farm.total_fencing_cost = 1560)
    (h4 : farm.total_fencing_cost = total_fencing_length farm * farm.fencing_cost_per_meter)
    (h5 : farm.diagonal^2 = farm.long_side^2 + farm.short_side^2) :
    farm_area farm = 1200 := by
  sorry


end farm_area_is_1200_l1286_128652


namespace calculate_expression_l1286_128661

theorem calculate_expression : -1^2023 - (-2)^3 - (-2) * (-3) = 1 := by
  sorry

end calculate_expression_l1286_128661


namespace class_notification_problem_l1286_128605

theorem class_notification_problem (n : ℕ) : 
  (1 + n + n^2 = 43) ↔ (n = 6) :=
by sorry

end class_notification_problem_l1286_128605


namespace milk_needed_for_cookies_l1286_128680

/-- The number of cookies in a dozen -/
def dozen : ℕ := 12

/-- The number of cookies that can be baked with 10 half-gallons of milk -/
def cookies_per_ten_halfgallons : ℕ := 40

/-- The number of half-gallons of milk needed for 40 cookies -/
def milk_for_forty_cookies : ℕ := 10

/-- The number of dozens of cookies to be baked -/
def dozens_to_bake : ℕ := 200

theorem milk_needed_for_cookies : 
  (dozens_to_bake * dozen * milk_for_forty_cookies) / cookies_per_ten_halfgallons = 600 := by
  sorry

end milk_needed_for_cookies_l1286_128680


namespace chess_game_probability_l1286_128637

theorem chess_game_probability (prob_draw prob_B_win : ℚ) 
  (h1 : prob_draw = 1/2) 
  (h2 : prob_B_win = 1/3) : 
  1 - prob_draw - prob_B_win = 1/6 := by
sorry

end chess_game_probability_l1286_128637


namespace expression_evaluation_l1286_128633

theorem expression_evaluation (x y z : ℤ) (hx : x = 25) (hy : y = 33) (hz : z = 7) :
  (x - (y - z)) - ((x - y) - z) = 14 := by
  sorry

end expression_evaluation_l1286_128633


namespace polygon_sides_l1286_128617

theorem polygon_sides (sum_angles : ℕ) (h1 : sum_angles = 1980) : ∃ n : ℕ, n = 13 ∧ sum_angles = 180 * (n - 2) := by
  sorry

end polygon_sides_l1286_128617


namespace compare_logarithms_and_sqrt_l1286_128610

theorem compare_logarithms_and_sqrt : 
  let a := 2 * Real.log (21/20)
  let b := Real.log (11/10)
  let c := Real.sqrt 1.2 - 1
  c < a ∧ a < b :=
by sorry

end compare_logarithms_and_sqrt_l1286_128610


namespace oak_willow_difference_l1286_128602

theorem oak_willow_difference (total_trees : ℕ) (willows : ℕ) 
  (h1 : total_trees = 83) (h2 : willows = 36) : total_trees - willows - willows = 11 := by
  sorry

end oak_willow_difference_l1286_128602


namespace geometric_sequence_property_l1286_128609

/-- Given a geometric sequence {a_n} with specific properties, prove a_7 = -2 -/
theorem geometric_sequence_property (a : ℕ → ℝ) :
  (∃ (q : ℝ), ∀ (n : ℕ), a (n + 1) = a n * q) →  -- geometric sequence condition
  a 2 * a 4 * a 5 = a 3 * a 6 →                   -- given condition
  a 9 * a 10 = -8 →                               -- given condition
  a 7 = -2 :=
by sorry

end geometric_sequence_property_l1286_128609


namespace berry_count_l1286_128657

theorem berry_count (total : ℕ) (raspberries blackberries blueberries : ℕ) : 
  total = 42 →
  raspberries = total / 2 →
  blackberries = total / 3 →
  total = raspberries + blackberries + blueberries →
  blueberries = 7 := by
sorry

end berry_count_l1286_128657


namespace orange_juice_distribution_l1286_128681

theorem orange_juice_distribution (pitcher_capacity : ℝ) (h : pitcher_capacity > 0) :
  let juice_amount : ℝ := (5 / 8) * pitcher_capacity
  let num_cups : ℕ := 4
  let juice_per_cup : ℝ := juice_amount / num_cups
  (juice_per_cup / pitcher_capacity) * 100 = 15.625 := by
sorry

end orange_juice_distribution_l1286_128681


namespace prob_same_length_is_17_35_l1286_128632

/-- The number of sides in a regular hexagon -/
def num_sides : ℕ := 6

/-- The number of diagonals in a regular hexagon -/
def num_diagonals : ℕ := 9

/-- The set of all sides and diagonals of a regular hexagon -/
def S : Finset ℕ := Finset.range (num_sides + num_diagonals)

/-- The probability of selecting two segments of the same length from S -/
def prob_same_length : ℚ :=
  (Nat.choose num_sides 2 + Nat.choose num_diagonals 2) / Nat.choose S.card 2

theorem prob_same_length_is_17_35 : prob_same_length = 17 / 35 := by
  sorry

end prob_same_length_is_17_35_l1286_128632


namespace fruit_salad_ratio_l1286_128624

def total_salads : ℕ := 600
def alaya_salads : ℕ := 200

theorem fruit_salad_ratio :
  let angel_salads := total_salads - alaya_salads
  (angel_salads : ℚ) / alaya_salads = 2 := by
  sorry

end fruit_salad_ratio_l1286_128624


namespace jasons_books_l1286_128634

/-- Given that Keith has 20 books and together with Jason they have 41 books,
    prove that Jason has 21 books. -/
theorem jasons_books (keith_books : ℕ) (total_books : ℕ) (h1 : keith_books = 20) (h2 : total_books = 41) :
  total_books - keith_books = 21 := by
  sorry

end jasons_books_l1286_128634


namespace equation_solution_l1286_128687

theorem equation_solution : ∃ (S : Set ℝ), S = {x : ℝ | (3*x + 6) / (x^2 + 5*x + 6) = (3 - x) / (x - 2) ∧ x ≠ 2 ∧ x ≠ -2} ∧ S = {3, -3} := by
  sorry

end equation_solution_l1286_128687


namespace crushers_win_probability_l1286_128627

theorem crushers_win_probability (n : ℕ) (p : ℚ) (h1 : n = 6) (h2 : p = 4/5) :
  p^n = 4096/15625 := by
  sorry

end crushers_win_probability_l1286_128627


namespace tyson_race_time_l1286_128669

/-- Calculates the total time Tyson spent in races given his swimming speeds and race details. -/
theorem tyson_race_time (lake_speed ocean_speed : ℝ) (total_races : ℕ) (race_distance : ℝ) : 
  lake_speed = 3 →
  ocean_speed = 2.5 →
  total_races = 10 →
  race_distance = 3 →
  (total_races / 2 : ℝ) * race_distance / lake_speed + 
  (total_races / 2 : ℝ) * race_distance / ocean_speed = 11 := by
  sorry


end tyson_race_time_l1286_128669


namespace decreasing_f_implies_a_geq_2_l1286_128619

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - 2*a*x - 3

-- State the theorem
theorem decreasing_f_implies_a_geq_2 :
  ∀ a : ℝ, (∀ x y : ℝ, -8 < x ∧ x < y ∧ y < 2 → f a x > f a y) → a ≥ 2 := by
  sorry

end decreasing_f_implies_a_geq_2_l1286_128619


namespace exists_same_color_transformation_l1286_128604

/-- Represents the color of a square on the chessboard -/
inductive Color
| Black
| White

/-- Represents a 16x16 chessboard -/
def Chessboard := Fin 16 → Fin 16 → Color

/-- Initial chessboard with alternating colors -/
def initialChessboard : Chessboard :=
  fun i j => if (i.val + j.val) % 2 = 0 then Color.Black else Color.White

/-- Apply operation A to the chessboard at position (i, j) -/
def applyOperationA (board : Chessboard) (i j : Fin 16) : Chessboard :=
  fun x y =>
    if x = i || y = j then
      match board x y with
      | Color.Black => Color.White
      | Color.White => Color.Black
    else
      board x y

/-- Check if all squares on the chessboard have the same color -/
def allSameColor (board : Chessboard) : Prop :=
  ∀ i j : Fin 16, board i j = board 0 0

/-- Theorem: There exists a sequence of operations A that transforms all squares to the same color -/
theorem exists_same_color_transformation :
  ∃ (operations : List (Fin 16 × Fin 16)),
    allSameColor (operations.foldl (fun b (i, j) => applyOperationA b i j) initialChessboard) :=
  sorry

end exists_same_color_transformation_l1286_128604


namespace sum_of_composite_function_l1286_128616

def p (x : ℝ) : ℝ := |x| - 3

def q (x : ℝ) : ℝ := -x^2

def x_values : List ℝ := [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5]

theorem sum_of_composite_function :
  (x_values.map (fun x => q (p x))).sum = -29 := by sorry

end sum_of_composite_function_l1286_128616


namespace savings_fraction_proof_l1286_128682

def total_savings : ℕ := 180000
def ppf_savings : ℕ := 72000

theorem savings_fraction_proof :
  let nsc_savings : ℕ := total_savings - ppf_savings
  let fraction : ℚ := (1/3 : ℚ) * nsc_savings / ppf_savings
  fraction = (1/2 : ℚ) := by sorry

end savings_fraction_proof_l1286_128682


namespace reunion_handshakes_l1286_128699

/-- Calculates the number of handshakes in a group --/
def handshakes (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Represents the reunion scenario --/
structure Reunion :=
  (total_boys : ℕ)
  (left_handed_boys : ℕ)
  (h_left_handed_le_total : left_handed_boys ≤ total_boys)

/-- Calculates the total number of handshakes at the reunion --/
def total_handshakes (r : Reunion) : ℕ :=
  handshakes r.left_handed_boys + handshakes (r.total_boys - r.left_handed_boys)

/-- Theorem stating that the total number of handshakes is 34 for the given scenario --/
theorem reunion_handshakes :
  ∀ (r : Reunion), r.total_boys = 12 → r.left_handed_boys = 4 → total_handshakes r = 34 :=
by
  sorry


end reunion_handshakes_l1286_128699


namespace inscribed_square_distances_l1286_128644

/-- A circle with radius 5 containing an inscribed square -/
structure InscribedSquareCircle where
  radius : ℝ
  radius_eq : radius = 5

/-- A point on the circumference of the circle -/
structure CircumferencePoint (c : InscribedSquareCircle) where
  point : ℝ × ℝ
  on_circle : (point.1 - c.radius)^2 + point.2^2 = c.radius^2

/-- Vertices of the inscribed square -/
def square_vertices (c : InscribedSquareCircle) : Fin 4 → ℝ × ℝ := sorry

/-- Distance between two points -/
def distance (p1 p2 : ℝ × ℝ) : ℝ := sorry

/-- Theorem stating the distances from a point on the circumference to the square vertices -/
theorem inscribed_square_distances
  (c : InscribedSquareCircle)
  (m : CircumferencePoint c)
  (h : distance m.point (square_vertices c 0) = 6) :
  ∃ (perm : Fin 3 → Fin 3),
    distance m.point (square_vertices c 1) = 8 ∧
    distance m.point (square_vertices c 2) = Real.sqrt 2 ∧
    distance m.point (square_vertices c 3) = 7 * Real.sqrt 2 := by
  sorry

end inscribed_square_distances_l1286_128644


namespace quadratic_root_relation_l1286_128656

/-- For a quadratic equation ax^2 + bx + c = 0, if one root is three times the other,
    then the coefficients a, b, and c satisfy the relationship 3b^2 = 16ac. -/
theorem quadratic_root_relation (a b c : ℝ) (h₁ : a ≠ 0) :
  (∃ x y : ℝ, x ≠ y ∧ a * x^2 + b * x + c = 0 ∧ a * y^2 + b * y + c = 0 ∧ y = 3 * x) →
  3 * b^2 = 16 * a * c :=
by sorry

end quadratic_root_relation_l1286_128656


namespace ellipse_param_sum_l1286_128664

/-- An ellipse with given properties -/
structure Ellipse where
  F₁ : ℝ × ℝ
  F₂ : ℝ × ℝ
  constant_sum : ℝ
  tangent_slope : ℝ

/-- The standard form parameters of an ellipse -/
structure EllipseParams where
  h : ℝ
  k : ℝ
  a : ℝ
  b : ℝ

/-- Theorem stating the sum of h, k, a, and b for the given ellipse -/
theorem ellipse_param_sum (e : Ellipse) (p : EllipseParams) : 
  e.F₁ = (-1, 1) → 
  e.F₂ = (5, 1) → 
  e.constant_sum = 10 → 
  e.tangent_slope = 1 → 
  p.h + p.k + p.a + p.b = 12 := by
  sorry

end ellipse_param_sum_l1286_128664


namespace car_distance_problem_l1286_128668

/-- Proves that Car X travels 105 miles from when Car Y starts until both cars stop -/
theorem car_distance_problem (speed_x speed_y : ℝ) (head_start : ℝ) (distance : ℝ) : 
  speed_x = 35 →
  speed_y = 49 →
  head_start = 1.2 →
  distance = speed_x * (head_start + (distance - speed_x * head_start) / (speed_y - speed_x)) →
  distance - speed_x * head_start = 105 := by
  sorry

#check car_distance_problem

end car_distance_problem_l1286_128668


namespace incorrect_transformation_l1286_128645

theorem incorrect_transformation (a b : ℝ) (h1 : a ≠ 0) (h2 : b ≠ 0) (h3 : a / 2 = b / 3) :
  ¬(2 * a = 3 * b) := by
  sorry

end incorrect_transformation_l1286_128645


namespace inequality_solution_set_l1286_128640

theorem inequality_solution_set : 
  {x : ℝ | x * (x - 1) * (x - 2) > 0} = {x : ℝ | (0 < x ∧ x < 1) ∨ x > 2} := by
sorry

end inequality_solution_set_l1286_128640


namespace other_endpoint_coordinates_l1286_128659

/-- Given a line segment with midpoint (3, 0) and one endpoint at (7, -4), 
    prove that the other endpoint is at (-1, 4) -/
theorem other_endpoint_coordinates :
  ∀ (A B : ℝ × ℝ),
    (A.1 + B.1) / 2 = 3 ∧
    (A.2 + B.2) / 2 = 0 ∧
    A = (7, -4) →
    B = (-1, 4) := by
  sorry

end other_endpoint_coordinates_l1286_128659


namespace croissant_resting_time_l1286_128623

theorem croissant_resting_time (fold_count : ℕ) (fold_time : ℕ) (mixing_time : ℕ) (baking_time : ℕ) (total_time : ℕ) :
  fold_count = 4 →
  fold_time = 5 →
  mixing_time = 10 →
  baking_time = 30 →
  total_time = 6 * 60 →
  (total_time - (mixing_time + fold_count * fold_time + baking_time)) / fold_count = 75 := by
  sorry

end croissant_resting_time_l1286_128623


namespace parabola_coefficients_l1286_128654

/-- A parabola passing through (1, 1) with a tangent line of slope 1 at (2, -1) has coefficients a = 3, b = -11, and c = 9. -/
theorem parabola_coefficients : 
  ∀ (a b c : ℝ), 
  (a * 1^2 + b * 1 + c = 1) →  -- Passes through (1, 1)
  (a * 2^2 + b * 2 + c = -1) →  -- Passes through (2, -1)
  (2 * a * 2 + b = 1) →  -- Slope of tangent line at (2, -1) is 1
  (a = 3 ∧ b = -11 ∧ c = 9) := by
sorry

end parabola_coefficients_l1286_128654
