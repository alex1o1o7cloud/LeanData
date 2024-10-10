import Mathlib

namespace seating_theorem_l1164_116434

/-- Number of seats in a row -/
def total_seats : ℕ := 7

/-- Number of people to be seated -/
def people_to_seat : ℕ := 4

/-- Number of adjacent empty seats required -/
def adjacent_empty_seats : ℕ := 2

/-- Function to calculate the number of seating arrangements -/
def seating_arrangements (total_seats : ℕ) (people_to_seat : ℕ) (adjacent_empty_seats : ℕ) : ℕ :=
  sorry

/-- Theorem stating the number of seating arrangements -/
theorem seating_theorem :
  seating_arrangements total_seats people_to_seat adjacent_empty_seats = 336 :=
sorry

end seating_theorem_l1164_116434


namespace cornelias_current_age_l1164_116400

/-- Proves Cornelia's current age given the conditions of the problem -/
theorem cornelias_current_age (kilee_current_age : ℕ) (cornelia_future_age kilee_future_age : ℕ) :
  kilee_current_age = 20 →
  kilee_future_age = kilee_current_age + 10 →
  cornelia_future_age = 3 * kilee_future_age →
  cornelia_future_age - 10 = 80 :=
by sorry

end cornelias_current_age_l1164_116400


namespace root_ordering_implies_a_range_l1164_116407

/-- Given two quadratic equations and an ordering of their roots, 
    prove the range of the coefficient a. -/
theorem root_ordering_implies_a_range 
  (a b : ℝ) 
  (x₁ x₂ x₃ x₄ : ℝ) 
  (h₁ : a * x₁^2 + b * x₁ + 1 = 0)
  (h₂ : a * x₂^2 + b * x₂ + 1 = 0)
  (h₃ : a^2 * x₃^2 + b * x₃ + 1 = 0)
  (h₄ : a^2 * x₄^2 + b * x₄ + 1 = 0)
  (h_order : x₃ < x₁ ∧ x₁ < x₂ ∧ x₂ < x₄) : 
  0 < a ∧ a < 1 := by
  sorry

end root_ordering_implies_a_range_l1164_116407


namespace total_savings_together_l1164_116401

/-- Regular price of a window -/
def regular_price : ℝ := 120

/-- Calculate the number of windows to pay for given the number of windows bought -/
def windows_to_pay_for (n : ℕ) : ℕ :=
  n - n / 6

/-- Calculate the price with the special deal (free 6th window) -/
def price_with_deal (n : ℕ) : ℝ :=
  (windows_to_pay_for n : ℝ) * regular_price

/-- Apply the additional 5% discount for purchases over 10 windows -/
def apply_discount (price : ℝ) (n : ℕ) : ℝ :=
  if n > 10 then price * 0.95 else price

/-- Calculate the final price after all discounts -/
def final_price (n : ℕ) : ℝ :=
  apply_discount (price_with_deal n) n

/-- Nina's number of windows -/
def nina_windows : ℕ := 9

/-- Carl's number of windows -/
def carl_windows : ℕ := 11

/-- Theorem: The total savings when Nina and Carl buy windows together is $348 -/
theorem total_savings_together : 
  (nina_windows + carl_windows) * regular_price - final_price (nina_windows + carl_windows) = 348 := by
  sorry

end total_savings_together_l1164_116401


namespace triangles_count_l1164_116482

/-- The number of triangles that can be made from a wire -/
def triangles_from_wire (original_length : ℕ) (remaining_length : ℕ) (triangle_wire_length : ℕ) : ℕ :=
  (original_length - remaining_length) / triangle_wire_length

/-- Theorem: Given the specified wire lengths, 24 triangles can be made -/
theorem triangles_count : triangles_from_wire 84 12 3 = 24 := by
  sorry

end triangles_count_l1164_116482


namespace at_most_one_obtuse_angle_l1164_116446

-- Define a triangle
def Triangle : Type := Unit

-- Define an angle in a triangle
def Angle (t : Triangle) : Type := Unit

-- Define if an angle is obtuse
def IsObtuse (t : Triangle) (a : Angle t) : Prop := sorry

-- State the theorem
theorem at_most_one_obtuse_angle (t : Triangle) :
  ¬∃ (a b : Angle t), a ≠ b ∧ IsObtuse t a ∧ IsObtuse t b :=
sorry

end at_most_one_obtuse_angle_l1164_116446


namespace arithmetic_sequence_sum_property_l1164_116439

/-- An arithmetic sequence with its sum function -/
structure ArithmeticSequence where
  S : ℕ → ℝ  -- Sum function

/-- Theorem: For an arithmetic sequence, if S_30 = S_60, then S_90 = 0 -/
theorem arithmetic_sequence_sum_property (seq : ArithmeticSequence) 
  (h : seq.S 30 = seq.S 60) : seq.S 90 = 0 := by
  sorry


end arithmetic_sequence_sum_property_l1164_116439


namespace opposite_of_negative_2023_l1164_116495

theorem opposite_of_negative_2023 : -(Int.neg 2023) = 2023 := by
  sorry

end opposite_of_negative_2023_l1164_116495


namespace problem_statement_l1164_116478

theorem problem_statement :
  (∀ x : ℝ, x^2 - 3*x + 1 = 0 → x^3 + 1/x^3 - 3 = 15) ∧
  (∀ x a b c : ℝ, a = 1/20*x + 20 ∧ b = 1/20*x + 19 ∧ c = 1/20*x + 21 →
    a^2 + b^2 + c^2 - a*b - b*c - a*c = 3) :=
by
  sorry

end problem_statement_l1164_116478


namespace max_value_of_f_l1164_116489

def f (x : ℝ) : ℝ := -4 * x^2 + 10 * x

theorem max_value_of_f :
  ∃ (max : ℝ), max = 25/4 ∧ ∀ (x : ℝ), f x ≤ max :=
sorry

end max_value_of_f_l1164_116489


namespace janet_dresses_pockets_l1164_116488

theorem janet_dresses_pockets :
  -- Total number of dresses
  ∀ total_dresses : ℕ,
  -- Number of dresses with pockets
  ∀ dresses_with_pockets : ℕ,
  -- Number of dresses with 2 pockets
  ∀ dresses_with_two_pockets : ℕ,
  -- Total number of pockets
  ∀ total_pockets : ℕ,
  -- Conditions
  total_dresses = 24 →
  dresses_with_pockets = total_dresses / 2 →
  dresses_with_two_pockets = dresses_with_pockets / 3 →
  total_pockets = 32 →
  -- Conclusion
  (total_pockets - 2 * dresses_with_two_pockets) / (dresses_with_pockets - dresses_with_two_pockets) = 3 :=
by sorry

end janet_dresses_pockets_l1164_116488


namespace sum_possible_side_lengths_is_330_l1164_116498

/-- Represents a convex quadrilateral with specific properties -/
structure ConvexQuadrilateral where
  EF : ℝ
  angleE : ℝ
  sidesArithmeticProgression : Bool
  EFisMaxLength : Bool
  EFparallelGH : Bool

/-- Calculates the sum of all possible values for the length of one of the other sides -/
def sumPossibleSideLengths (q : ConvexQuadrilateral) : ℝ :=
  sorry

/-- Theorem stating the sum of all possible values for the length of one of the other sides is 330 -/
theorem sum_possible_side_lengths_is_330 (q : ConvexQuadrilateral) :
  q.EF = 24 ∧ q.angleE = 45 ∧ q.sidesArithmeticProgression ∧ q.EFisMaxLength ∧ q.EFparallelGH →
  sumPossibleSideLengths q = 330 :=
by sorry

end sum_possible_side_lengths_is_330_l1164_116498


namespace john_computer_cost_l1164_116437

/-- The total cost of a computer after upgrades -/
def total_cost (initial_cost old_video_card old_memory old_processor new_video_card new_memory new_processor : ℕ) : ℕ :=
  initial_cost + new_video_card + new_memory + new_processor - old_video_card - old_memory - old_processor

/-- Theorem: The total cost of John's computer after upgrades is $2500 -/
theorem john_computer_cost : 
  total_cost 2000 300 100 150 500 200 350 = 2500 := by
  sorry

end john_computer_cost_l1164_116437


namespace batsman_average_after_31st_inning_l1164_116438

/-- Represents a batsman's cricket statistics -/
structure Batsman where
  innings : ℕ
  totalRuns : ℕ
  average : ℚ

/-- Calculates the new average after an innings -/
def newAverage (b : Batsman) (runsScored : ℕ) : ℚ :=
  (b.totalRuns + runsScored : ℚ) / (b.innings + 1 : ℚ)

theorem batsman_average_after_31st_inning 
  (b : Batsman)
  (h1 : b.innings = 30)
  (h2 : newAverage b 105 = b.average + 3) :
  newAverage b 105 = 15 := by
  sorry

#check batsman_average_after_31st_inning

end batsman_average_after_31st_inning_l1164_116438


namespace bernard_luke_age_problem_l1164_116453

/-- Given that in 8 years, Mr. Bernard will be 3 times as old as Luke is now,
    prove that 10 years less than their average current age is 2 * L - 14,
    where L is Luke's current age. -/
theorem bernard_luke_age_problem (L : ℕ) : 
  (L + ((3 * L) - 8)) / 2 - 10 = 2 * L - 14 := by
  sorry

end bernard_luke_age_problem_l1164_116453


namespace class_average_problem_l1164_116429

theorem class_average_problem (x : ℝ) : 
  0.15 * x + 0.50 * 78 + 0.35 * 63 = 76.05 → x = 100 := by
  sorry

end class_average_problem_l1164_116429


namespace reflection_in_fourth_quadrant_l1164_116432

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Defines the second quadrant -/
def secondQuadrant (p : Point) : Prop :=
  p.x < 0 ∧ p.y > 0

/-- Defines the fourth quadrant -/
def fourthQuadrant (p : Point) : Prop :=
  p.x > 0 ∧ p.y < 0

/-- Reflects a point across the y-axis -/
def reflectYAxis (p : Point) : Point :=
  { x := -p.x, y := p.y }

/-- Theorem stating that if P is in the second quadrant, 
    then the reflection of Q across the y-axis is in the fourth quadrant -/
theorem reflection_in_fourth_quadrant (a b : ℝ) :
  let p : Point := { x := a, y := b }
  let q : Point := { x := a - 1, y := -b }
  secondQuadrant p → fourthQuadrant (reflectYAxis q) := by
  sorry


end reflection_in_fourth_quadrant_l1164_116432


namespace problem_solution_l1164_116416

theorem problem_solution (x y : ℝ) : 
  (0.5 * x = 0.05 * 500 - 20) ∧ 
  (0.3 * y = 0.25 * x + 10) → 
  (x = 10 ∧ y = 125/3) := by
sorry

end problem_solution_l1164_116416


namespace work_completion_time_proportional_aarti_triple_work_time_l1164_116423

/-- If a person can complete a piece of work in a given number of days,
    then the time required to complete a multiple of that work is proportional. -/
theorem work_completion_time_proportional
  (days_for_single_work : ℕ) (work_multiple : ℕ) :
  let days_for_multiple_work := days_for_single_work * work_multiple
  days_for_multiple_work = days_for_single_work * work_multiple :=
by sorry

/-- Aarti's work completion time for triple work -/
theorem aarti_triple_work_time :
  let days_for_single_work := 6
  let work_multiple := 3
  let days_for_triple_work := days_for_single_work * work_multiple
  days_for_triple_work = 18 :=
by sorry

end work_completion_time_proportional_aarti_triple_work_time_l1164_116423


namespace area_perimeter_ratio_l1164_116451

/-- The side length of the square -/
def square_side : ℝ := 5

/-- The side length of the equilateral triangle -/
def triangle_side : ℝ := 6

/-- The area of a square given its side length -/
def square_area (side : ℝ) : ℝ := side * side

/-- The perimeter of an equilateral triangle given its side length -/
def equilateral_triangle_perimeter (side : ℝ) : ℝ := 3 * side

/-- Theorem stating the ratio of the square's area to the triangle's perimeter -/
theorem area_perimeter_ratio :
  (square_area square_side) / (equilateral_triangle_perimeter triangle_side) = 25 / 18 := by
  sorry


end area_perimeter_ratio_l1164_116451


namespace range_of_x_for_proposition_l1164_116410

theorem range_of_x_for_proposition (x : ℝ) : 
  (∃ a : ℝ, a ∈ Set.Icc 1 3 ∧ a * x^2 + (a - 2) * x - 2 > 0) ↔ 
  x < -1 ∨ x > 2/3 :=
sorry

end range_of_x_for_proposition_l1164_116410


namespace zero_in_interval_l1164_116499

theorem zero_in_interval (a b : ℝ) (ha : a > 1) (hb : 0 < b) (hb' : b < 1) :
  ∃ x : ℝ, -1 < x ∧ x < 0 ∧ a^x + x - b = 0 := by
  sorry

end zero_in_interval_l1164_116499


namespace moon_speed_conversion_l1164_116455

/-- The number of seconds in an hour -/
def seconds_per_hour : ℕ := 3600

/-- The moon's speed in kilometers per second -/
def moon_speed_km_per_second : ℝ := 1.02

/-- Converts speed from kilometers per second to kilometers per hour -/
def km_per_second_to_km_per_hour (speed_km_per_second : ℝ) : ℝ :=
  speed_km_per_second * seconds_per_hour

theorem moon_speed_conversion :
  km_per_second_to_km_per_hour moon_speed_km_per_second = 3672 := by
  sorry

end moon_speed_conversion_l1164_116455


namespace quadratic_equation_properties_l1164_116487

theorem quadratic_equation_properties (k : ℝ) :
  let f : ℝ → ℝ := λ x ↦ x^2 + 3*x + k - 2
  (∃ x₁ x₂ : ℝ, f x₁ = 0 ∧ f x₂ = 0 ∧ x₁ ≠ x₂) →
  (k ≤ 17/4 ∧
   (∀ x₁ x₂ : ℝ, f x₁ = 0 → f x₂ = 0 → x₁ ≠ x₂ → (x₁ - 1)*(x₂ - 1) = -1 → k = -3)) :=
by sorry

end quadratic_equation_properties_l1164_116487


namespace no_divisibility_by_1955_l1164_116418

theorem no_divisibility_by_1955 : ∀ n : ℤ, ¬(1955 ∣ (n^2 + n + 1)) := by
  sorry

end no_divisibility_by_1955_l1164_116418


namespace tom_car_lease_cost_l1164_116463

/-- Calculates the total yearly cost for Tom's car lease -/
theorem tom_car_lease_cost :
  let miles_per_week : ℕ := 4 * 50 + 3 * 100
  let cost_per_mile : ℚ := 1 / 10
  let weekly_fee : ℕ := 100
  let weeks_per_year : ℕ := 52
  (miles_per_week : ℚ) * cost_per_mile * weeks_per_year + (weekly_fee : ℚ) * weeks_per_year = 7800 := by
  sorry

end tom_car_lease_cost_l1164_116463


namespace quadratic_inequality_condition_l1164_116493

theorem quadratic_inequality_condition (a : ℝ) :
  (a ≥ 0 → ∃ x : ℝ, a * x^2 + x + 1 ≥ 0) ∧
  (∃ a : ℝ, a < 0 ∧ ∃ x : ℝ, a * x^2 + x + 1 ≥ 0) :=
by sorry

end quadratic_inequality_condition_l1164_116493


namespace MON_is_right_angle_l1164_116456

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 2*x

-- Define point E
def E : ℝ × ℝ := (2, 2)

-- Define the line l
def line_l (x y : ℝ) : Prop := ∃ k, y = k*(x - 2)

-- Define that l passes through (2,0)
axiom l_through_2_0 : line_l 2 0

-- Define points A and B on the parabola and line l
def A : ℝ × ℝ := sorry
def B : ℝ × ℝ := sorry
axiom A_on_parabola : parabola A.1 A.2
axiom B_on_parabola : parabola B.1 B.2
axiom A_on_l : line_l A.1 A.2
axiom B_on_l : line_l B.1 B.2
axiom A_not_E : A ≠ E
axiom B_not_E : B ≠ E

-- Define points M and N
def M : ℝ × ℝ := sorry
def N : ℝ × ℝ := sorry
axiom M_on_EA : ∃ t, M = (1 - t) • E + t • A
axiom N_on_EB : ∃ t, N = (1 - t) • E + t • B
axiom M_on_x_neg2 : M.1 = -2
axiom N_on_x_neg2 : N.1 = -2

-- Define the origin O
def O : ℝ × ℝ := (0, 0)

-- Theorem to prove
theorem MON_is_right_angle : 
  let OM := M - O
  let ON := N - O
  OM.1 * ON.1 + OM.2 * ON.2 = 0 := by sorry

end MON_is_right_angle_l1164_116456


namespace fourth_sphere_radius_l1164_116459

/-- Given four spheres where each touches the other three, and three of them have radius R,
    the radius of the fourth sphere is R/3. -/
theorem fourth_sphere_radius (R : ℝ) (R_pos : R > 0) : ℝ :=
  let fourth_radius := R / 3
  fourth_radius

#check fourth_sphere_radius

end fourth_sphere_radius_l1164_116459


namespace dessert_preference_l1164_116421

theorem dessert_preference (total : ℕ) (apple : ℕ) (chocolate : ℕ) (neither : ℕ)
  (h1 : total = 50)
  (h2 : apple = 22)
  (h3 : chocolate = 20)
  (h4 : neither = 15) :
  apple + chocolate - (total - neither) = 7 :=
by sorry

end dessert_preference_l1164_116421


namespace log_equation_solution_l1164_116441

theorem log_equation_solution :
  ∃! (x : ℝ), x > 0 ∧ 2 * Real.log x = Real.log (x + 12) :=
by
  -- The unique solution is x = 4
  use 4
  sorry

end log_equation_solution_l1164_116441


namespace regression_line_change_l1164_116427

/-- Represents a linear regression equation of the form y = a + bx -/
structure RegressionLine where
  a : ℝ
  b : ℝ

/-- Calculates the change in y when x increases by 1 unit -/
def change_in_y (line : RegressionLine) : ℝ := -line.b

/-- Theorem: For the given regression line, when x increases by 1 unit, y decreases by 1.5 units -/
theorem regression_line_change (line : RegressionLine) 
  (h1 : line.a = 2) 
  (h2 : line.b = -1.5) : 
  change_in_y line = -1.5 := by
  sorry

end regression_line_change_l1164_116427


namespace intersection_M_complement_N_l1164_116474

open Set Real

def M : Set ℝ := {x | -1 < x ∧ x < 2}
def N : Set ℝ := {x | ∃ y, y = Real.sqrt (x - 1)}

theorem intersection_M_complement_N : M ∩ (𝒰 \ N) = Ioo (-1) 1 := by
  sorry

end intersection_M_complement_N_l1164_116474


namespace centroid_of_equal_areas_l1164_116466

/-- A point in a 2D plane -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- A triangle defined by three points -/
structure Triangle :=
  (A : Point)
  (B : Point)
  (C : Point)

/-- Check if a point is inside a triangle -/
def isInside (M : Point) (T : Triangle) : Prop :=
  sorry

/-- Calculate the area of a triangle -/
def triangleArea (A B C : Point) : ℝ :=
  sorry

/-- Check if three triangles have equal areas -/
def equalAreas (T1 T2 T3 : Triangle) : Prop :=
  triangleArea T1.A T1.B T1.C = triangleArea T2.A T2.B T2.C ∧
  triangleArea T2.A T2.B T2.C = triangleArea T3.A T3.B T3.C

/-- Check if a point is the centroid of a triangle -/
def isCentroid (M : Point) (T : Triangle) : Prop :=
  sorry

theorem centroid_of_equal_areas (ABC : Triangle) (M : Point) 
  (h1 : isInside M ABC)
  (h2 : equalAreas (Triangle.mk M ABC.A ABC.B) (Triangle.mk M ABC.A ABC.C) (Triangle.mk M ABC.B ABC.C)) :
  isCentroid M ABC :=
sorry

end centroid_of_equal_areas_l1164_116466


namespace parallelepiped_properties_l1164_116464

/-- A rectangular parallelepiped with an inscribed sphere -/
structure Parallelepiped :=
  (k : ℝ)  -- Ratio of parallelepiped volume to sphere volume
  (h : k > 0)  -- k is positive

/-- Theorem about the angles and permissible values of k for a parallelepiped with an inscribed sphere -/
theorem parallelepiped_properties (p : Parallelepiped) :
  let α := Real.arcsin (6 / (Real.pi * p.k))
  ∃ (angle1 angle2 : ℝ),
    (angle1 = α ∧ angle2 = Real.pi - α) ∧  -- Angles at the base
    p.k ≥ 6 / Real.pi :=  -- Permissible values of k
by sorry

end parallelepiped_properties_l1164_116464


namespace fence_perimeter_is_200_l1164_116471

/-- A square field enclosed by evenly spaced triangular posts -/
structure FenceSetup where
  total_posts : ℕ
  post_width : ℝ
  gap_width : ℝ

/-- Calculate the outer perimeter of the fence setup -/
def outer_perimeter (f : FenceSetup) : ℝ :=
  let posts_per_side := f.total_posts / 4
  let gaps_per_side := posts_per_side - 1
  let side_length := posts_per_side * f.post_width + gaps_per_side * f.gap_width
  4 * side_length

/-- Theorem: The outer perimeter of the given fence setup is 200 feet -/
theorem fence_perimeter_is_200 : 
  outer_perimeter ⟨36, 2, 4⟩ = 200 := by sorry

end fence_perimeter_is_200_l1164_116471


namespace platform_length_l1164_116449

/-- Given a train of length 300 meters that takes 36 seconds to cross a platform
    and 18 seconds to cross a signal pole, the length of the platform is 300 meters. -/
theorem platform_length (train_length : ℝ) (platform_crossing_time : ℝ) (pole_crossing_time : ℝ)
  (h1 : train_length = 300)
  (h2 : platform_crossing_time = 36)
  (h3 : pole_crossing_time = 18) :
  let train_speed := train_length / pole_crossing_time
  let platform_length := train_speed * platform_crossing_time - train_length
  platform_length = 300 := by
  sorry

end platform_length_l1164_116449


namespace boys_average_age_l1164_116411

/-- Proves that the average age of boys is 12 years given the school statistics -/
theorem boys_average_age (total_students : ℕ) (girls : ℕ) (girls_avg_age : ℝ) (school_avg_age : ℝ) :
  total_students = 652 →
  girls = 163 →
  girls_avg_age = 11 →
  school_avg_age = 11.75 →
  let boys := total_students - girls
  let boys_total_age := school_avg_age * total_students - girls_avg_age * girls
  boys_total_age / boys = 12 := by
sorry


end boys_average_age_l1164_116411


namespace cylinder_heights_sum_l1164_116414

theorem cylinder_heights_sum (p₁ p₂ p₃ : ℝ) 
  (h₁ : p₁ = 6) 
  (h₂ : p₂ = 9) 
  (h₃ : p₃ = 11) : 
  p₁ + p₂ + p₃ = 26 := by
  sorry

end cylinder_heights_sum_l1164_116414


namespace integral_of_derivative_scaled_l1164_116420

theorem integral_of_derivative_scaled (f : ℝ → ℝ) (a b : ℝ) (hf : Differentiable ℝ f) (hab : a < b) :
  ∫ x in a..b, (deriv f (3 * x)) = (1 / 3) * (f (3 * b) - f (3 * a)) := by
  sorry

end integral_of_derivative_scaled_l1164_116420


namespace product_purely_imaginary_l1164_116497

theorem product_purely_imaginary (x : ℝ) : 
  (∃ y : ℝ, (x + 2*I) * ((x + 1) + 3*I) * ((x + 2) + 4*I) = y*I) ↔ x = 1 :=
by sorry

end product_purely_imaginary_l1164_116497


namespace solve_for_c_l1164_116443

theorem solve_for_c (m a b c : ℝ) (h : m = (c * b * a) / (a - c)) :
  c = (m * a) / (m + b * a) := by
  sorry

end solve_for_c_l1164_116443


namespace ones_digit_8_pow_32_l1164_116422

/-- The ones digit of a natural number -/
def ones_digit (n : ℕ) : ℕ := n % 10

/-- The ones digit of 8^n for any natural number n -/
def ones_digit_8_pow (n : ℕ) : ℕ :=
  match n % 4 with
  | 0 => 6
  | 1 => 8
  | 2 => 4
  | 3 => 2
  | _ => 0  -- This case is unreachable, but needed for exhaustiveness

theorem ones_digit_8_pow_32 :
  ones_digit (8^32) = 6 := by
  sorry

end ones_digit_8_pow_32_l1164_116422


namespace john_used_one_nickel_l1164_116473

/-- Calculates the number of nickels used in a purchase, given the number of quarters and dimes used, the cost of the item, and the change received. -/
def nickels_used (quarters : ℕ) (dimes : ℕ) (cost : ℕ) (change : ℕ) : ℕ :=
  let quarter_value := 25
  let dime_value := 10
  let nickel_value := 5
  let total_paid := cost + change
  let paid_without_nickels := quarters * quarter_value + dimes * dime_value
  (total_paid - paid_without_nickels) / nickel_value

theorem john_used_one_nickel :
  nickels_used 4 3 131 4 = 1 := by
  sorry

end john_used_one_nickel_l1164_116473


namespace socks_thrown_away_l1164_116406

theorem socks_thrown_away (initial_socks : ℕ) (new_socks : ℕ) (final_socks : ℕ) : 
  initial_socks = 33 → new_socks = 13 → final_socks = 27 → 
  initial_socks - (final_socks - new_socks) = 19 := by
sorry

end socks_thrown_away_l1164_116406


namespace normal_level_short_gallons_needed_after_evaporation_l1164_116430

/-- Represents a water reservoir with given properties -/
structure Reservoir where
  current_level : ℝ
  normal_level : ℝ
  total_capacity : ℝ
  evaporation_rate : ℝ
  current_is_twice_normal : current_level = 2 * normal_level
  current_is_75_percent : current_level = 0.75 * total_capacity
  h_current_level : current_level = 30
  h_evaporation_rate : evaporation_rate = 0.1

/-- The normal level is 25 million gallons short of total capacity -/
theorem normal_level_short (r : Reservoir) :
  r.total_capacity - r.normal_level = 25 :=
sorry

/-- After evaporation, 13 million gallons are needed to reach total capacity -/
theorem gallons_needed_after_evaporation (r : Reservoir) :
  r.total_capacity - (r.current_level - r.evaporation_rate * r.current_level) = 13 :=
sorry

end normal_level_short_gallons_needed_after_evaporation_l1164_116430


namespace prob_at_least_one_to_museum_l1164_116465

/-- The probability that at least one of two independent events occurs -/
def prob_at_least_one (p₁ p₂ : ℝ) : ℝ := 1 - (1 - p₁) * (1 - p₂)

/-- The probability that at least one of two people goes to the museum -/
theorem prob_at_least_one_to_museum (p_a p_b : ℝ) 
  (h_a : p_a = 0.8) 
  (h_b : p_b = 0.7) : 
  prob_at_least_one p_a p_b = 0.94 := by
  sorry

end prob_at_least_one_to_museum_l1164_116465


namespace max_grid_mean_l1164_116491

def Grid := Fin 3 → Fin 3 → ℕ

def valid_grid (g : Grid) : Prop :=
  (∀ i j, g i j ∈ Finset.range 9) ∧
  (∀ n, n ∈ Finset.range 9 → ∃ i j, g i j = n)

def circle_mean (g : Grid) (i j : Fin 2) : ℚ :=
  (g i j + g i (j+1) + g (i+1) j + g (i+1) (j+1)) / 4

def grid_mean (g : Grid) : ℚ :=
  (circle_mean g 0 0 + circle_mean g 0 1 + circle_mean g 1 0 + circle_mean g 1 1) / 4

theorem max_grid_mean :
  ∀ g : Grid, valid_grid g → grid_mean g ≤ 5.8125 :=
sorry

end max_grid_mean_l1164_116491


namespace max_victory_margin_l1164_116457

/-- Represents the vote count for a candidate in a specific time period -/
structure VoteCount where
  first_two_hours : ℕ
  last_two_hours : ℕ

/-- Represents the election results -/
structure ElectionResult where
  petya : VoteCount
  vasya : VoteCount

def total_votes (result : ElectionResult) : ℕ :=
  result.petya.first_two_hours + result.petya.last_two_hours +
  result.vasya.first_two_hours + result.vasya.last_two_hours

def petya_total (result : ElectionResult) : ℕ :=
  result.petya.first_two_hours + result.petya.last_two_hours

def vasya_total (result : ElectionResult) : ℕ :=
  result.vasya.first_two_hours + result.vasya.last_two_hours

def is_valid_result (result : ElectionResult) : Prop :=
  total_votes result = 27 ∧
  result.petya.first_two_hours = result.vasya.first_two_hours + 9 ∧
  result.vasya.last_two_hours = result.petya.last_two_hours + 9 ∧
  petya_total result > vasya_total result

def victory_margin (result : ElectionResult) : ℕ :=
  petya_total result - vasya_total result

theorem max_victory_margin :
  ∀ result : ElectionResult,
    is_valid_result result →
    victory_margin result ≤ 9 :=
by
  sorry

#check max_victory_margin

end max_victory_margin_l1164_116457


namespace bottom_level_legos_l1164_116454

/-- Represents a 3-level pyramid with decreasing lego sides -/
structure LegoPyramid where
  bottom : ℕ  -- Number of legos per side on the bottom level
  mid : ℕ     -- Number of legos per side on the middle level
  top : ℕ     -- Number of legos per side on the top level

/-- Calculates the total number of legos in the pyramid -/
def totalLegos (p : LegoPyramid) : ℕ :=
  p.bottom ^ 2 + p.mid ^ 2 + p.top ^ 2

/-- Theorem: The bottom level of a 3-level pyramid with 110 total legos has 7 legos per side -/
theorem bottom_level_legos :
  ∃ (p : LegoPyramid),
    p.mid = p.bottom - 1 ∧
    p.top = p.bottom - 2 ∧
    totalLegos p = 110 ∧
    p.bottom = 7 :=
by
  sorry

end bottom_level_legos_l1164_116454


namespace complex_number_problem_l1164_116496

theorem complex_number_problem (z₁ z₂ : ℂ) : 
  ((z₁ - 2) * (1 + Complex.I) = 1 - Complex.I) →
  z₂.im = 2 →
  (z₁ * z₂).im = 0 →
  z₂ = 4 + 2 * Complex.I ∧ Complex.abs z₂ = 2 * Real.sqrt 5 := by
  sorry

end complex_number_problem_l1164_116496


namespace chad_video_games_earnings_l1164_116433

/-- Chad's earnings and savings problem -/
theorem chad_video_games_earnings
  (savings_rate : ℚ)
  (mowing_earnings : ℚ)
  (birthday_earnings : ℚ)
  (odd_jobs_earnings : ℚ)
  (total_savings : ℚ)
  (h1 : savings_rate = 40 / 100)
  (h2 : mowing_earnings = 600)
  (h3 : birthday_earnings = 250)
  (h4 : odd_jobs_earnings = 150)
  (h5 : total_savings = 460) :
  let total_earnings := total_savings / savings_rate
  let known_earnings := mowing_earnings + birthday_earnings + odd_jobs_earnings
  total_earnings - known_earnings = 150 := by
sorry

end chad_video_games_earnings_l1164_116433


namespace isabella_hair_growth_l1164_116470

def current_hair_length : ℕ := 18
def hair_growth : ℕ := 4

theorem isabella_hair_growth :
  current_hair_length + hair_growth = 22 :=
by sorry

end isabella_hair_growth_l1164_116470


namespace factorization_equality_l1164_116469

-- Define the equality we want to prove
theorem factorization_equality (a : ℝ) : a^2 - 2*a + 1 = (a - 1)^2 := by
  sorry

end factorization_equality_l1164_116469


namespace smallest_AAB_l1164_116468

/-- Represents a two-digit number --/
def TwoDigitNumber (a b : Nat) : Prop :=
  a ≥ 1 ∧ a ≤ 9 ∧ b ≥ 0 ∧ b ≤ 9

/-- Represents a three-digit number --/
def ThreeDigitNumber (a b : Nat) : Prop :=
  a ≥ 1 ∧ a ≤ 9 ∧ b ≥ 0 ∧ b ≤ 9

/-- The value of a two-digit number AB --/
def ValueAB (a b : Nat) : Nat :=
  10 * a + b

/-- The value of a three-digit number AAB --/
def ValueAAB (a b : Nat) : Nat :=
  100 * a + 10 * a + b

theorem smallest_AAB :
  ∀ a b : Nat,
    TwoDigitNumber a b →
    ThreeDigitNumber a b →
    a ≠ b →
    8 * (ValueAB a b) = ValueAAB a b →
    ∀ x y : Nat,
      TwoDigitNumber x y →
      ThreeDigitNumber x y →
      x ≠ y →
      8 * (ValueAB x y) = ValueAAB x y →
      ValueAAB a b ≤ ValueAAB x y →
    ValueAAB a b = 224 :=
by sorry

end smallest_AAB_l1164_116468


namespace intersection_line_slope_l1164_116412

/-- Given two circles in the xy-plane, this theorem proves that the slope of the line 
    passing through their intersection points is -2/3. -/
theorem intersection_line_slope (x y : ℝ) : 
  (x^2 + y^2 - 6*x + 4*y - 12 = 0) →
  (x^2 + y^2 - 10*x - 2*y + 22 = 0) →
  ∃ (m : ℝ), m = -2/3 ∧ 
  ∀ (x₁ y₁ x₂ y₂ : ℝ), 
    (x₁^2 + y₁^2 - 6*x₁ + 4*y₁ - 12 = 0) →
    (x₁^2 + y₁^2 - 10*x₁ - 2*y₁ + 22 = 0) →
    (x₂^2 + y₂^2 - 6*x₂ + 4*y₂ - 12 = 0) →
    (x₂^2 + y₂^2 - 10*x₂ - 2*y₂ + 22 = 0) →
    x₁ ≠ x₂ →
    (y₂ - y₁) / (x₂ - x₁) = m :=
by sorry

end intersection_line_slope_l1164_116412


namespace polygon_with_five_triangles_l1164_116403

/-- A polygon with n sides -/
structure Polygon (n : ℕ) where
  -- We don't need to define the structure, just declare it

/-- The number of triangles formed when drawing diagonals from a single vertex -/
def triangles_from_vertex (n : ℕ) : ℕ := n - 2

/-- Theorem: If the diagonals from the same vertex of an n-sided polygon
    exactly divide the polygon into 5 triangles, then n = 7 -/
theorem polygon_with_five_triangles (n : ℕ) :
  triangles_from_vertex n = 5 → n = 7 := by
  sorry


end polygon_with_five_triangles_l1164_116403


namespace max_shoe_pairs_l1164_116460

theorem max_shoe_pairs (initial_pairs : ℕ) (lost_shoes : ℕ) (max_remaining_pairs : ℕ) : 
  initial_pairs = 23 → lost_shoes = 9 → max_remaining_pairs = 14 →
  max_remaining_pairs = initial_pairs - lost_shoes / 2 := by
  sorry

end max_shoe_pairs_l1164_116460


namespace greatest_prime_factor_of_4_pow_17_minus_2_pow_29_l1164_116419

theorem greatest_prime_factor_of_4_pow_17_minus_2_pow_29 : 
  ∃ (p : ℕ), p.Prime ∧ p = 31 ∧ 
  (∀ q : ℕ, q.Prime → q ∣ (4^17 - 2^29) → q ≤ p) :=
sorry

end greatest_prime_factor_of_4_pow_17_minus_2_pow_29_l1164_116419


namespace a_minus_b_value_l1164_116472

theorem a_minus_b_value (a b : ℝ) 
  (ha : |a| = 4)
  (hb : |b| = 2)
  (hab : |a + b| = -(a + b)) :
  a - b = -2 ∨ a - b = -6 := by
sorry

end a_minus_b_value_l1164_116472


namespace last_passenger_correct_seat_prob_l1164_116479

/-- Represents a bus with n seats and n passengers -/
structure Bus (n : ℕ) where
  seats : Fin n → Passenger
  tickets : Fin n → Seat

/-- Represents a passenger -/
inductive Passenger
| scientist
| regular (id : ℕ)

/-- Represents a seat -/
def Seat := ℕ

/-- The seating process for the bus -/
def seatingProcess (b : Bus n) : Bus n := sorry

/-- The probability that the last passenger sits in their assigned seat -/
def lastPassengerInCorrectSeat (b : Bus n) : ℚ := sorry

/-- Theorem stating that the probability of the last passenger sitting in their assigned seat is 1/2 -/
theorem last_passenger_correct_seat_prob (n : ℕ) (b : Bus n) :
  lastPassengerInCorrectSeat (seatingProcess b) = 1 / 2 := by sorry

end last_passenger_correct_seat_prob_l1164_116479


namespace rectangle_area_l1164_116409

theorem rectangle_area (w : ℝ) (h1 : w > 0) : 
  let l := 2 * w
  let square_side := 1
  let rectangle_perimeter := 2 * l + 2 * w
  let square_perimeter := 4 * square_side
  rectangle_perimeter = square_perimeter → l * w = 8 / 9 := by
sorry

end rectangle_area_l1164_116409


namespace unknown_number_in_average_l1164_116477

theorem unknown_number_in_average (x : ℝ) : 
  (20 + 40 + 60) / 3 = (10 + 50 + x) / 3 + 5 → x = 45 := by
  sorry

end unknown_number_in_average_l1164_116477


namespace equation_describes_parabola_l1164_116417

-- Define the equation
def equation (x y : ℝ) : Prop := |y + 5| = Real.sqrt ((x - 2)^2 + y^2)

-- Define what it means for an equation to describe a parabola
def describes_parabola (eq : (ℝ → ℝ → Prop)) : Prop :=
  ∃ a b c d : ℝ, a ≠ 0 ∧ ∀ x y : ℝ, eq x y ↔ y = a * x^2 + b * x + c ∨ x = a * y^2 + b * y + d

-- Theorem statement
theorem equation_describes_parabola : describes_parabola equation := by sorry

end equation_describes_parabola_l1164_116417


namespace stick_cutting_l1164_116484

theorem stick_cutting (short_length long_length : ℝ) : 
  long_length = short_length + 18 →
  short_length + long_length = 30 →
  long_length / short_length = 4 :=
by
  sorry

end stick_cutting_l1164_116484


namespace win_sector_area_l1164_116462

theorem win_sector_area (r : ℝ) (p : ℝ) (h1 : r = 8) (h2 : p = 3/8) :
  p * π * r^2 = 24 * π := by
  sorry

end win_sector_area_l1164_116462


namespace inequality_implication_l1164_116405

theorem inequality_implication (a b : ℝ) : -2 * a + 1 < -2 * b + 1 → a > b := by
  sorry

end inequality_implication_l1164_116405


namespace store_shirts_sold_l1164_116485

theorem store_shirts_sold (num_jeans : ℕ) (shirt_price : ℕ) (total_earnings : ℕ) :
  num_jeans = 10 ∧ 
  shirt_price = 10 ∧ 
  total_earnings = 400 →
  ∃ (num_shirts : ℕ), 
    num_shirts * shirt_price + num_jeans * (2 * shirt_price) = total_earnings ∧
    num_shirts = 20 :=
by sorry

end store_shirts_sold_l1164_116485


namespace max_integer_solution_inequality_system_l1164_116480

theorem max_integer_solution_inequality_system :
  ∀ x : ℤ, (3 * x - 1 < x + 1 ∧ 2 * (2 * x - 1) ≤ 5 * x + 1) →
  x ≤ 0 :=
by sorry

end max_integer_solution_inequality_system_l1164_116480


namespace triangle_is_right_angled_l1164_116475

/-- A triangle with angles satisfying specific ratios is right-angled -/
theorem triangle_is_right_angled (angle1 angle2 angle3 : ℝ) : 
  angle1 + angle2 + angle3 = 180 →
  angle1 = 3 * angle2 →
  angle3 = 2 * angle2 →
  angle1 = 90 := by
sorry

end triangle_is_right_angled_l1164_116475


namespace tobias_driveways_shoveled_tobias_driveways_shoveled_proof_l1164_116404

theorem tobias_driveways_shoveled : ℕ → ℕ → ℕ → ℕ → ℕ → ℕ → ℕ → ℕ → ℕ → ℕ → Prop :=
  fun shoe_cost saving_months allowance lawn_charge shovel_charge wage change hours_worked lawns_mowed driveways_shoveled =>
    shoe_cost = 95 ∧
    saving_months = 3 ∧
    allowance = 5 ∧
    lawn_charge = 15 ∧
    shovel_charge = 7 ∧
    wage = 8 ∧
    change = 15 ∧
    hours_worked = 10 ∧
    lawns_mowed = 4 →
    driveways_shoveled = 6

theorem tobias_driveways_shoveled_proof : tobias_driveways_shoveled 95 3 5 15 7 8 15 10 4 6 := by
  sorry

end tobias_driveways_shoveled_tobias_driveways_shoveled_proof_l1164_116404


namespace hyperbola_eccentricity_l1164_116444

/-- The eccentricity of a hyperbola with equation x²/a² - y²/b² = 1,
    given that one of its asymptotes passes through the point (2, √21) -/
theorem hyperbola_eccentricity (a b : ℝ) (h : a ≠ 0) (k : b ≠ 0) :
  (∃ (x y : ℝ), x = 2 ∧ y = Real.sqrt 21 ∧ y = (b / a) * x) →
  Real.sqrt (1 + (b / a)^2) = 5/2 :=
sorry

end hyperbola_eccentricity_l1164_116444


namespace number_problem_l1164_116431

theorem number_problem (N : ℝ) : 
  (1/8 : ℝ) * (3/5 : ℝ) * (4/7 : ℝ) * (5/11 : ℝ) * N - (1/9 : ℝ) * (2/3 : ℝ) * (3/4 : ℝ) * (5/8 : ℝ) * N = 30 → 
  (75/100 : ℝ) * N = -1476 := by
sorry

end number_problem_l1164_116431


namespace quadratic_inequality_always_negative_l1164_116428

theorem quadratic_inequality_always_negative : ∀ x : ℝ, -6 * x^2 + 2 * x - 8 < 0 := by
  sorry

end quadratic_inequality_always_negative_l1164_116428


namespace perpendicular_vectors_x_value_l1164_116445

theorem perpendicular_vectors_x_value (x : ℝ) : 
  let a : Fin 3 → ℝ := ![2, -1, x]
  let b : Fin 3 → ℝ := ![3, 2, -1]
  (∀ i : Fin 3, (a i) * (b i) = 0) → x = 4 := by
  sorry

end perpendicular_vectors_x_value_l1164_116445


namespace total_fireworks_count_l1164_116492

/-- The number of boxes Koby has -/
def koby_boxes : ℕ := 2

/-- The number of sparklers in each of Koby's boxes -/
def koby_sparklers_per_box : ℕ := 3

/-- The number of whistlers in each of Koby's boxes -/
def koby_whistlers_per_box : ℕ := 5

/-- The number of boxes Cherie has -/
def cherie_boxes : ℕ := 1

/-- The number of sparklers in Cherie's box -/
def cherie_sparklers : ℕ := 8

/-- The number of whistlers in Cherie's box -/
def cherie_whistlers : ℕ := 9

/-- The total number of fireworks Koby and Cherie have -/
def total_fireworks : ℕ := 
  koby_boxes * (koby_sparklers_per_box + koby_whistlers_per_box) +
  cherie_boxes * (cherie_sparklers + cherie_whistlers)

theorem total_fireworks_count : total_fireworks = 33 := by
  sorry

end total_fireworks_count_l1164_116492


namespace rectangular_prism_to_cube_l1164_116424

theorem rectangular_prism_to_cube (a b c : ℝ) (h1 : a = 8) (h2 : b = 8) (h3 : c = 27) :
  ∃ s : ℝ, s^3 = a * b * c ∧ s = 12 := by
  sorry

end rectangular_prism_to_cube_l1164_116424


namespace upper_side_length_l1164_116408

/-- Represents a trapezoid with given properties -/
structure Trapezoid where
  lower_side : ℝ
  upper_side : ℝ
  height : ℝ
  area : ℝ
  upper_shorter : upper_side = lower_side - 6
  height_value : height = 8
  area_value : area = 72
  area_formula : area = (lower_side + upper_side) / 2 * height

/-- Theorem: The length of the upper side of the trapezoid is 6 cm -/
theorem upper_side_length (t : Trapezoid) : t.upper_side = 6 := by
  sorry

end upper_side_length_l1164_116408


namespace divisor_problem_l1164_116486

theorem divisor_problem (n m : ℕ) (h1 : n = 3830) (h2 : m = 5) : 
  (∃ d : ℕ, d > 0 ∧ (n - m) % d = 0 ∧ 
   ∀ k < m, ¬((n - k) % d = 0)) → 
  (n - m) % 15 = 0 ∧ 15 > 0 ∧ 
  ∀ k < m, ¬((n - k) % 15 = 0) :=
sorry

end divisor_problem_l1164_116486


namespace min_area_AOB_l1164_116413

noncomputable section

-- Define the hyperbola C₁
def C₁ (a : ℝ) (x y : ℝ) : Prop := x^2 / a^2 - y^2 / (2 * a^2) = 1 ∧ a > 0

-- Define the parabola C₂
def C₂ (a : ℝ) (x y : ℝ) : Prop := y^2 = -4 * Real.sqrt 3 * a * x

-- Define the focus F₁
def F₁ (a : ℝ) : ℝ × ℝ := (-Real.sqrt 3 * a, 0)

-- Define a chord AB of C₂ passing through F₁
def chord_AB (a k : ℝ) (x y : ℝ) : Prop :=
  y = k * (x + Real.sqrt 3 * a) ∧ C₂ a x y

-- Define the area of triangle AOB
def area_AOB (a k : ℝ) : ℝ := 6 * a^2 * Real.sqrt (1 + 1 / k^2)

-- Main theorem
theorem min_area_AOB (a : ℝ) :
  (∃ k : ℝ, ∀ k' : ℝ, area_AOB a k ≤ area_AOB a k') ∧
  (∃ x : ℝ, x = -Real.sqrt 3 * a ∧ 
    ∀ k : ℝ, area_AOB a k ≥ 6 * a^2) :=
sorry

end

end min_area_AOB_l1164_116413


namespace f_properties_l1164_116440

noncomputable def f (x : ℝ) : ℝ := x^2 * Real.log x

theorem f_properties :
  (∀ x > 1, f x > 0) ∧
  (∀ x, 0 < x → x < 1 → f x < 0) ∧
  (Set.range f = Set.Ici (-1 / (2 * Real.exp 1))) ∧
  (∀ x > 0, f x ≥ x - 1) :=
sorry

end f_properties_l1164_116440


namespace tangent_and_perpendicular_l1164_116461

-- Define the curve f(x)
def f (x : ℝ) : ℝ := x^3 + 3*x^2 - 1

-- Define the line perpendicular to the given line
def perp_line (x y : ℝ) : Prop := 2*x - 6*y + 1 = 0

-- Define the tangent line
def tangent_line (x y : ℝ) : Prop := 3*x + y + 2 = 0

-- Define the theorem
theorem tangent_and_perpendicular :
  ∃ (x₀ y₀ : ℝ),
    -- The point (x₀, y₀) is on the curve
    f x₀ = y₀ ∧
    -- The tangent line passes through (x₀, y₀)
    tangent_line x₀ y₀ ∧
    -- The tangent line is perpendicular to the given line
    (∀ (x y : ℝ), perp_line x y → 
      (y - y₀) = -(1/3) * (x - x₀)) ∧
    -- The slope of the tangent line at (x₀, y₀) is the derivative of f at x₀
    (3*x₀^2 + 6*x₀ = -3) :=
sorry

end tangent_and_perpendicular_l1164_116461


namespace rose_painting_time_l1164_116415

/-- Time to paint a lily in minutes -/
def lily_time : ℕ := 5

/-- Time to paint an orchid in minutes -/
def orchid_time : ℕ := 3

/-- Time to paint a vine in minutes -/
def vine_time : ℕ := 2

/-- Total time spent painting in minutes -/
def total_time : ℕ := 213

/-- Number of lilies painted -/
def lily_count : ℕ := 17

/-- Number of roses painted -/
def rose_count : ℕ := 10

/-- Number of orchids painted -/
def orchid_count : ℕ := 6

/-- Number of vines painted -/
def vine_count : ℕ := 20

/-- Time to paint a rose in minutes -/
def rose_time : ℕ := 7

theorem rose_painting_time : 
  lily_count * lily_time + rose_count * rose_time + orchid_count * orchid_time + vine_count * vine_time = total_time := by
  sorry

end rose_painting_time_l1164_116415


namespace inverse_sum_property_l1164_116425

-- Define a function f with domain ℝ
variable (f : ℝ → ℝ)

-- Define the inverse function g of f
variable (g : ℝ → ℝ)

-- Define the symmetry condition for f
def symmetric_about_neg_one_zero (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f ((-1) - x) = f ((-1) + x)

-- Define the inverse relationship between f and g
def inverse_functions (f g : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, g (f x) = x ∧ f (g x) = x

-- Theorem statement
theorem inverse_sum_property
  (h_sym : symmetric_about_neg_one_zero f)
  (h_inv : inverse_functions f g)
  (x₁ x₂ : ℝ)
  (h_sum : x₁ + x₂ = 0) :
  g x₁ + g x₂ = -2 := by
  sorry

end inverse_sum_property_l1164_116425


namespace helmet_store_theorem_l1164_116435

structure HelmetStore where
  wholesale_price_A : ℕ
  wholesale_price_B : ℕ
  day1_sales_A : ℕ
  day1_sales_B : ℕ
  day1_total : ℕ
  day2_sales_A : ℕ
  day2_sales_B : ℕ
  day2_total : ℕ
  budget : ℕ
  total_helmets : ℕ
  profit_target : ℕ

def selling_prices (store : HelmetStore) : ℕ × ℕ :=
  -- Placeholder for the function to calculate selling prices
  (0, 0)

def can_achieve_profit (store : HelmetStore) (prices : ℕ × ℕ) : Prop :=
  -- Placeholder for the function to check if profit target can be achieved
  false

theorem helmet_store_theorem (store : HelmetStore) 
  (h1 : store.wholesale_price_A = 40)
  (h2 : store.wholesale_price_B = 30)
  (h3 : store.day1_sales_A = 10)
  (h4 : store.day1_sales_B = 15)
  (h5 : store.day1_total = 1150)
  (h6 : store.day2_sales_A = 6)
  (h7 : store.day2_sales_B = 12)
  (h8 : store.day2_total = 810)
  (h9 : store.budget = 3400)
  (h10 : store.total_helmets = 100)
  (h11 : store.profit_target = 1300) :
  let prices := selling_prices store
  prices = (55, 40) ∧ ¬(can_achieve_profit store prices) := by
  sorry

end helmet_store_theorem_l1164_116435


namespace wedge_volume_l1164_116481

/-- The volume of a wedge cut from a cylindrical log --/
theorem wedge_volume (d : ℝ) (θ : ℝ) (h : θ = 60) :
  let r := d / 2
  let cylinder_volume := π * r^2 * d
  let wedge_volume := cylinder_volume * θ / 360
  d = 16 → wedge_volume = 341 * π :=
by sorry

end wedge_volume_l1164_116481


namespace product_digit_sum_l1164_116476

def first_number : ℕ := 404040404040404040404040404040404040404040404040404040404040404040404040404040404040404040404040404040404040404
def second_number : ℕ := 707070707070707070707070707070707070707070707070707070707070707070707070707070707070707070707070707070707070707

theorem product_digit_sum : 
  let product := first_number * second_number
  let thousands_digit := (product / 1000) % 10
  let units_digit := product % 10
  thousands_digit + units_digit = 13 := by sorry

end product_digit_sum_l1164_116476


namespace geometric_sequence_a5_l1164_116483

/-- A geometric sequence with a_3 = 1 and a_7 = 9 -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  (∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n) ∧ 
  a 3 = 1 ∧ 
  a 7 = 9

theorem geometric_sequence_a5 (a : ℕ → ℝ) (h : geometric_sequence a) : 
  a 5 = 3 := by
  sorry

end geometric_sequence_a5_l1164_116483


namespace power_of_power_l1164_116467

theorem power_of_power (a : ℝ) : (a^2)^3 = a^6 := by
  sorry

end power_of_power_l1164_116467


namespace circle_m_equation_l1164_116448

/-- A circle M with center on the negative x-axis and radius 4, tangent to the line 3x + 4y + 4 = 0 -/
structure CircleM where
  /-- The x-coordinate of the center of the circle -/
  a : ℝ
  /-- The center is on the negative x-axis -/
  h_negative : a < 0
  /-- The radius of the circle is 4 -/
  radius : ℝ := 4
  /-- The line 3x + 4y + 4 = 0 is tangent to the circle -/
  h_tangent : |3 * a + 4| / Real.sqrt (3^2 + 4^2) = radius

/-- The equation of circle M is (x+8)² + y² = 16 -/
theorem circle_m_equation (m : CircleM) : 
  ∀ x y : ℝ, (x - m.a)^2 + y^2 = m.radius^2 ↔ (x + 8)^2 + y^2 = 16 :=
sorry

end circle_m_equation_l1164_116448


namespace correct_calculation_l1164_116494

theorem correct_calculation (x : ℝ) (h : x * 3 = 18) : x / 3 = 2 := by
  sorry

end correct_calculation_l1164_116494


namespace exponential_inequality_l1164_116447

theorem exponential_inequality (x : ℝ) : (2 : ℝ) ^ (2 * x - 7) > (2 : ℝ) ^ (4 * x - 1) ↔ x < -3 := by
  sorry

end exponential_inequality_l1164_116447


namespace min_value_expression_l1164_116452

theorem min_value_expression (a b : ℝ) (hb : b ≠ 0) :
  a^2 + b^2 + a/b + 1/b^2 ≥ Real.sqrt 3 ∧
  ∃ (a₀ b₀ : ℝ) (hb₀ : b₀ ≠ 0), a₀^2 + b₀^2 + a₀/b₀ + 1/b₀^2 = Real.sqrt 3 :=
by sorry

end min_value_expression_l1164_116452


namespace calculation_proof_l1164_116450

theorem calculation_proof : 2⁻¹ + Real.sqrt 16 - (3 - Real.sqrt 3)^0 + |Real.sqrt 2 - 1/2| = 3 + Real.sqrt 2 := by
  sorry

end calculation_proof_l1164_116450


namespace triangle_area_l1164_116402

/-- The area of the triangle formed by the intersection of two lines and the y-axis --/
theorem triangle_area (line1 line2 : ℝ → ℝ) : 
  line1 = (λ x => 3 * x - 6) →
  line2 = (λ x => -4 * x + 24) →
  let x_intersect := (30 : ℝ) / 7
  let y_intersect := (48 : ℝ) / 7
  let base := 30
  let height := x_intersect
  (1 / 2 : ℝ) * base * height = 450 / 7 := by
sorry

end triangle_area_l1164_116402


namespace horse_speed_l1164_116458

/-- Given a square field with area 1600 km^2 and a horse that takes 10 hours to run around it,
    the speed of the horse is 16 km/h. -/
theorem horse_speed (field_area : ℝ) (run_time : ℝ) (horse_speed : ℝ) : 
  field_area = 1600 → run_time = 10 → horse_speed = (4 * Real.sqrt field_area) / run_time → 
  horse_speed = 16 := by sorry

end horse_speed_l1164_116458


namespace ratio_of_two_numbers_l1164_116490

theorem ratio_of_two_numbers (a b : ℝ) (h1 : a > 0) (h2 : b > 0) 
  (h3 : a + b = 44) (h4 : a - b = 20) : a / b = 8 / 3 := by
  sorry

end ratio_of_two_numbers_l1164_116490


namespace max_min_product_l1164_116426

theorem max_min_product (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (sum_eq : a + b + c = 10) (prod_sum_eq : a * b + b * c + c * a = 25) :
  ∃ (m : ℝ), m = min (a * b) (min (b * c) (c * a)) ∧ m ≤ 25 / 9 ∧
  ∃ (a' b' c' : ℝ), a' > 0 ∧ b' > 0 ∧ c' > 0 ∧
    a' + b' + c' = 10 ∧ a' * b' + b' * c' + c' * a' = 25 ∧
    min (a' * b') (min (b' * c') (c' * a')) = 25 / 9 :=
by sorry

end max_min_product_l1164_116426


namespace large_planks_count_l1164_116442

theorem large_planks_count (nails_per_plank : ℕ) (additional_nails : ℕ) (total_nails : ℕ) :
  nails_per_plank = 17 →
  additional_nails = 8 →
  total_nails = 229 →
  ∃ (x : ℕ), x * nails_per_plank + additional_nails = total_nails ∧ x = 13 :=
by sorry

end large_planks_count_l1164_116442


namespace notebook_discount_rate_l1164_116436

/-- The maximum discount rate that can be applied to a notebook while maintaining a minimum profit margin. -/
theorem notebook_discount_rate (cost : ℝ) (original_price : ℝ) (min_profit_margin : ℝ) :
  cost = 6 →
  original_price = 9 →
  min_profit_margin = 0.05 →
  ∃ (max_discount : ℝ), 
    max_discount = 0.7 ∧ 
    ∀ (discount : ℝ), 
      discount ≤ max_discount →
      (original_price * (1 - discount) - cost) / cost ≥ min_profit_margin :=
by sorry

end notebook_discount_rate_l1164_116436
