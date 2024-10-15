import Mathlib

namespace NUMINAMATH_CALUDE_gift_cost_proof_l406_40628

theorem gift_cost_proof (initial_friends : Nat) (dropped_out : Nat) (share_increase : Int) :
  initial_friends = 10 →
  dropped_out = 4 →
  share_increase = 8 →
  ∃ (cost : Int),
    cost > 0 ∧
    (cost / (initial_friends - dropped_out : Int) = cost / initial_friends + share_increase) ∧
    cost = 120 := by
  sorry

end NUMINAMATH_CALUDE_gift_cost_proof_l406_40628


namespace NUMINAMATH_CALUDE_probability_sum_multiple_of_three_l406_40684

def die_faces : ℕ := 6

def total_outcomes : ℕ := die_faces * die_faces

def is_multiple_of_three (n : ℕ) : Prop := ∃ k, n = 3 * k

def favorable_outcomes : ℕ := 12

theorem probability_sum_multiple_of_three :
  (favorable_outcomes : ℚ) / total_outcomes = 1 / 3 :=
sorry

end NUMINAMATH_CALUDE_probability_sum_multiple_of_three_l406_40684


namespace NUMINAMATH_CALUDE_final_value_calculation_l406_40645

def initial_value : ℝ := 1500

def first_increase (x : ℝ) : ℝ := x * 1.20

def second_decrease (x : ℝ) : ℝ := x * 0.85

def third_increase (x : ℝ) : ℝ := x * 1.10

theorem final_value_calculation :
  third_increase (second_decrease (first_increase initial_value)) = 1683 := by
  sorry

end NUMINAMATH_CALUDE_final_value_calculation_l406_40645


namespace NUMINAMATH_CALUDE_base_k_representation_l406_40697

theorem base_k_representation (k : ℕ) (hk : k > 0) : 
  (8 : ℚ) / 63 = (k + 5 : ℚ) / (k^2 - 1) → k = 17 := by
  sorry

end NUMINAMATH_CALUDE_base_k_representation_l406_40697


namespace NUMINAMATH_CALUDE_similar_triangles_segment_length_l406_40648

/-- Two triangles are similar -/
structure SimilarTriangles (T1 T2 : Type) :=
  (sim : T1 → T2 → Prop)

/-- Triangle GHI -/
structure TriangleGHI :=
  (G H I : ℝ)

/-- Triangle XYZ -/
structure TriangleXYZ :=
  (X Y Z : ℝ)

/-- The problem statement -/
theorem similar_triangles_segment_length 
  (tri_GHI : TriangleGHI) 
  (tri_XYZ : TriangleXYZ) 
  (sim : SimilarTriangles TriangleGHI TriangleXYZ) 
  (h_sim : sim.sim tri_GHI tri_XYZ)
  (h_GH : tri_GHI.H - tri_GHI.G = 8)
  (h_HI : tri_GHI.I - tri_GHI.H = 16)
  (h_YZ : tri_XYZ.Z - tri_XYZ.Y = 24) :
  tri_XYZ.Y - tri_XYZ.X = 12 := by
  sorry

end NUMINAMATH_CALUDE_similar_triangles_segment_length_l406_40648


namespace NUMINAMATH_CALUDE_company_employee_reduction_l406_40694

theorem company_employee_reduction (reduction_percentage : ℝ) (final_employees : ℕ) : 
  reduction_percentage = 14 →
  final_employees = 195 →
  round (final_employees / (1 - reduction_percentage / 100)) = 227 :=
by sorry

end NUMINAMATH_CALUDE_company_employee_reduction_l406_40694


namespace NUMINAMATH_CALUDE_twelve_tone_equal_temperament_l406_40689

theorem twelve_tone_equal_temperament (a : ℕ → ℝ) :
  (∀ n : ℕ, 1 ≤ n → n ≤ 13 → a n > 0) →
  (∀ n : ℕ, 1 < n → n ≤ 13 → a n / a (n-1) = a 2 / a 1) →
  a 1 = 1 →
  a 13 = 2 →
  a 5 = 2^(1/3) :=
sorry

end NUMINAMATH_CALUDE_twelve_tone_equal_temperament_l406_40689


namespace NUMINAMATH_CALUDE_fraction_domain_l406_40635

theorem fraction_domain (x : ℝ) : (∃ y : ℝ, y = 1 / (x - 1)) → x ≠ 1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_domain_l406_40635


namespace NUMINAMATH_CALUDE_first_group_size_correct_l406_40623

/-- The number of beavers in the first group -/
def first_group_size : ℕ := 20

/-- The time taken by the first group to build the dam (in hours) -/
def first_group_time : ℕ := 3

/-- The number of beavers in the second group -/
def second_group_size : ℕ := 12

/-- The time taken by the second group to build the dam (in hours) -/
def second_group_time : ℕ := 5

/-- Theorem stating that the first group size is correct -/
theorem first_group_size_correct : 
  first_group_size * first_group_time = second_group_size * second_group_time :=
by sorry

end NUMINAMATH_CALUDE_first_group_size_correct_l406_40623


namespace NUMINAMATH_CALUDE_m_range_l406_40651

def A : Set ℝ := {x | x < 1}
def B : Set ℝ := {x | x * (x - 1) > 6}
def C (m : ℝ) : Set ℝ := {x | -1 + m < x ∧ x < 2 * m}

theorem m_range (m : ℝ) : 
  (C m).Nonempty ∧ C m ⊆ (A ∩ (Set.univ \ B)) → -1 < m ∧ m ≤ 1/2 := by
  sorry

end NUMINAMATH_CALUDE_m_range_l406_40651


namespace NUMINAMATH_CALUDE_blue_parrots_count_l406_40699

theorem blue_parrots_count (total : ℕ) (red_fraction green_fraction : ℚ) : 
  total = 120 →
  red_fraction = 2/3 →
  green_fraction = 1/6 →
  (total : ℚ) * (1 - (red_fraction + green_fraction)) = 20 := by
  sorry

end NUMINAMATH_CALUDE_blue_parrots_count_l406_40699


namespace NUMINAMATH_CALUDE_inscribed_rectangle_area_coefficient_l406_40676

/-- Triangle with side lengths --/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Rectangle inscribed in a triangle --/
structure InscribedRectangle where
  base : ℝ

/-- The area of the inscribed rectangle as a function of its base --/
def rectangleArea (t : Triangle) (r : InscribedRectangle) : ℝ → ℝ :=
  fun ω => α * ω - β * ω^2
  where
    α : ℝ := sorry
    β : ℝ := sorry

theorem inscribed_rectangle_area_coefficient (t : Triangle) (r : InscribedRectangle) :
  t.a = 15 ∧ t.b = 34 ∧ t.c = 21 →
  ∃ α β : ℝ, (∀ ω : ℝ, rectangleArea t r ω = α * ω - β * ω^2) ∧ β = 5/41 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_rectangle_area_coefficient_l406_40676


namespace NUMINAMATH_CALUDE_gap_height_from_wire_extension_l406_40654

/-- Given a sphere of radius R and a wire wrapped around its equator,
    if the wire's length is increased by L, the resulting gap height h
    between the sphere and the wire is given by h = L / (2π). -/
theorem gap_height_from_wire_extension (R L : ℝ) (h : ℝ) 
    (hR : R > 0) (hL : L > 0) : 
    2 * π * (R + h) = 2 * π * R + L → h = L / (2 * π) := by
  sorry

end NUMINAMATH_CALUDE_gap_height_from_wire_extension_l406_40654


namespace NUMINAMATH_CALUDE_fib_even_iff_index_div_three_l406_40670

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | n + 2 => fib (n + 1) + fib n

/-- Theorem: A Fibonacci number is even if and only if its index is divisible by 3 -/
theorem fib_even_iff_index_div_three (n : ℕ) : Even (fib n) ↔ 3 ∣ n := by sorry

end NUMINAMATH_CALUDE_fib_even_iff_index_div_three_l406_40670


namespace NUMINAMATH_CALUDE_no_positive_solutions_l406_40664

theorem no_positive_solutions : ¬∃ (a b c d : ℝ), 
  a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧
  a * d^2 + b * d - c = 0 ∧
  Real.sqrt a * d + Real.sqrt b * Real.sqrt d - Real.sqrt c = 0 := by
  sorry

end NUMINAMATH_CALUDE_no_positive_solutions_l406_40664


namespace NUMINAMATH_CALUDE_leticia_dish_cost_is_10_l406_40693

/-- The cost of Leticia's dish -/
def leticia_dish_cost : ℝ := 10

/-- The cost of Scarlett's dish -/
def scarlett_dish_cost : ℝ := 13

/-- The cost of Percy's dish -/
def percy_dish_cost : ℝ := 17

/-- The tip rate -/
def tip_rate : ℝ := 0.10

/-- The total tip amount -/
def total_tip : ℝ := 4

/-- Theorem stating that Leticia's dish costs $10 given the conditions -/
theorem leticia_dish_cost_is_10 :
  leticia_dish_cost = 10 :=
by sorry

end NUMINAMATH_CALUDE_leticia_dish_cost_is_10_l406_40693


namespace NUMINAMATH_CALUDE_solution_equation_l406_40661

theorem solution_equation (x : ℝ) : (5 * 12) / (x / 3) + 80 = 81 ↔ x = 180 :=
by sorry

end NUMINAMATH_CALUDE_solution_equation_l406_40661


namespace NUMINAMATH_CALUDE_cos_squared_minus_sin_squared_l406_40680

theorem cos_squared_minus_sin_squared (α : Real) :
  (∃ (x y : Real), x = 1 ∧ y = 2 ∧ y / x = Real.tan α) →
  Real.cos α ^ 2 - Real.sin α ^ 2 = -3/5 := by
  sorry

end NUMINAMATH_CALUDE_cos_squared_minus_sin_squared_l406_40680


namespace NUMINAMATH_CALUDE_bird_count_l406_40640

theorem bird_count (swallows bluebirds cardinals : ℕ) : 
  swallows = 2 →
  swallows * 2 = bluebirds →
  cardinals = bluebirds * 3 →
  swallows + bluebirds + cardinals = 18 := by
sorry

end NUMINAMATH_CALUDE_bird_count_l406_40640


namespace NUMINAMATH_CALUDE_no_solution_absolute_value_equation_l406_40673

theorem no_solution_absolute_value_equation :
  ∀ x : ℝ, |-3 * x| + 5 ≠ 0 := by
sorry

end NUMINAMATH_CALUDE_no_solution_absolute_value_equation_l406_40673


namespace NUMINAMATH_CALUDE_cricket_team_average_age_l406_40604

theorem cricket_team_average_age 
  (team_size : ℕ) 
  (average_age : ℝ) 
  (wicket_keeper_age_diff : ℝ) 
  (remaining_average_diff : ℝ) :
  team_size = 11 →
  average_age = 29 →
  wicket_keeper_age_diff = 3 →
  remaining_average_diff = 1 →
  ∃ (captain_age : ℝ),
    team_size * average_age = 
      (team_size - 2) * (average_age - remaining_average_diff) + 
      captain_age + 
      (average_age + wicket_keeper_age_diff) :=
by sorry

end NUMINAMATH_CALUDE_cricket_team_average_age_l406_40604


namespace NUMINAMATH_CALUDE_remainder_3_102_mod_101_l406_40634

theorem remainder_3_102_mod_101 (h : Nat.Prime 101) : 3^102 ≡ 9 [MOD 101] := by
  sorry

end NUMINAMATH_CALUDE_remainder_3_102_mod_101_l406_40634


namespace NUMINAMATH_CALUDE_angle_R_measure_l406_40602

/-- A hexagon with specific angle properties -/
structure Hexagon where
  F : ℝ  -- Angle F in degrees
  I : ℝ  -- Angle I in degrees
  G : ℝ  -- Angle G in degrees
  U : ℝ  -- Angle U in degrees
  R : ℝ  -- Angle R in degrees
  E : ℝ  -- Angle E in degrees
  sum_angles : F + I + G + U + R + E = 720  -- Sum of angles in a hexagon
  supplementary : G + U = 180  -- G and U are supplementary
  congruent : F = I ∧ I = R ∧ R = E  -- F, I, R, and E are congruent

/-- The measure of angle R in the specific hexagon is 135 degrees -/
theorem angle_R_measure (h : Hexagon) : h.R = 135 := by
  sorry

end NUMINAMATH_CALUDE_angle_R_measure_l406_40602


namespace NUMINAMATH_CALUDE_elevator_weight_problem_l406_40672

theorem elevator_weight_problem (initial_people : ℕ) (initial_avg_weight : ℝ) 
  (new_avg_weight : ℝ) (h1 : initial_people = 6) 
  (h2 : initial_avg_weight = 152) (h3 : new_avg_weight = 151) :
  let total_initial_weight := initial_people * initial_avg_weight
  let total_new_weight := (initial_people + 1) * new_avg_weight
  let seventh_person_weight := total_new_weight - total_initial_weight
  seventh_person_weight = 145 := by
  sorry

end NUMINAMATH_CALUDE_elevator_weight_problem_l406_40672


namespace NUMINAMATH_CALUDE_line_intercepts_l406_40638

/-- The equation of the line -/
def line_equation (x y : ℝ) : Prop := 3 * x - y + 6 = 0

/-- The y-intercept of the line -/
def y_intercept : ℝ := 6

/-- The x-intercept of the line -/
def x_intercept : ℝ := -2

/-- Theorem stating that the y-intercept and x-intercept are correct for the given line equation -/
theorem line_intercepts :
  line_equation 0 y_intercept ∧ line_equation x_intercept 0 :=
sorry

end NUMINAMATH_CALUDE_line_intercepts_l406_40638


namespace NUMINAMATH_CALUDE_no_double_application_successor_function_l406_40682

theorem no_double_application_successor_function :
  ¬ ∃ (f : ℕ → ℕ), ∀ (n : ℕ), f (f n) = n + 1 := by
  sorry

end NUMINAMATH_CALUDE_no_double_application_successor_function_l406_40682


namespace NUMINAMATH_CALUDE_frog_climb_time_l406_40647

/-- Represents the frog's climbing problem -/
structure FrogClimb where
  well_depth : ℝ
  climb_distance : ℝ
  slip_distance : ℝ
  slip_time_ratio : ℝ
  time_to_near_top : ℝ

/-- Calculates the total time for the frog to climb to the top of the well -/
def total_climb_time (f : FrogClimb) : ℝ :=
  sorry

/-- Theorem stating that the total climb time is 20 minutes -/
theorem frog_climb_time (f : FrogClimb) 
  (h1 : f.well_depth = 12)
  (h2 : f.climb_distance = 3)
  (h3 : f.slip_distance = 1)
  (h4 : f.slip_time_ratio = 1/3)
  (h5 : f.time_to_near_top = 17) :
  total_climb_time f = 20 := by
  sorry

end NUMINAMATH_CALUDE_frog_climb_time_l406_40647


namespace NUMINAMATH_CALUDE_divisible_by_eight_inductive_step_l406_40622

theorem divisible_by_eight (n : ℕ) :
  ∃ m : ℤ, 3^(4*n+1) + 5^(2*n+1) = 8 * m :=
by
  sorry

theorem inductive_step (k : ℕ) :
  3^(4*(k+1)+1) + 5^(2*(k+1)+1) = 56 * 3^(4*k+1) + 25 * (3^(4*k+1) + 5^(2*k+1)) :=
by
  sorry

end NUMINAMATH_CALUDE_divisible_by_eight_inductive_step_l406_40622


namespace NUMINAMATH_CALUDE_alex_not_jogging_probability_l406_40691

theorem alex_not_jogging_probability (p : ℚ) 
  (h : p = 5/8) : 1 - p = 3/8 := by
  sorry

end NUMINAMATH_CALUDE_alex_not_jogging_probability_l406_40691


namespace NUMINAMATH_CALUDE_parallelogram_base_l406_40655

/-- Given a parallelogram with area 180 square centimeters and height 10 cm, its base is 18 cm. -/
theorem parallelogram_base (area height base : ℝ) : 
  area = 180 ∧ height = 10 ∧ area = base * height → base = 18 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_base_l406_40655


namespace NUMINAMATH_CALUDE_max_value_of_sum_l406_40660

theorem max_value_of_sum (a b c : ℝ) (h : a^2 + b^2 + c^2 = 4) :
  ∃ (max : ℝ), max = 10 * Real.sqrt 2 ∧ ∀ (x y z : ℝ), x^2 + y^2 + z^2 = 4 → 3*x + 4*y + 5*z ≤ max :=
sorry

end NUMINAMATH_CALUDE_max_value_of_sum_l406_40660


namespace NUMINAMATH_CALUDE_percentage_problem_l406_40631

theorem percentage_problem (x : ℝ) (h1 : x > 0) (h2 : (x / 100) * x = 9) : x = 30 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l406_40631


namespace NUMINAMATH_CALUDE_sandbox_width_l406_40603

/-- A rectangular sandbox with perimeter 30 feet and length twice the width has a width of 5 feet. -/
theorem sandbox_width :
  ∀ (width length : ℝ),
  width > 0 →
  length > 0 →
  length = 2 * width →
  2 * length + 2 * width = 30 →
  width = 5 := by
sorry

end NUMINAMATH_CALUDE_sandbox_width_l406_40603


namespace NUMINAMATH_CALUDE_problem_solution_l406_40609

theorem problem_solution : 101^4 - 4 * 101^3 + 6 * 101^2 - 4 * 101 + 1 = 100000000 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l406_40609


namespace NUMINAMATH_CALUDE_inscribed_rectangle_pc_length_l406_40695

-- Define the triangle ABC
structure EquilateralTriangle where
  side : ℝ
  side_positive : side > 0

-- Define the rectangle PQRS
structure InscribedRectangle (t : EquilateralTriangle) where
  ps : ℝ
  pq : ℝ
  ps_positive : ps > 0
  pq_positive : pq > 0
  on_sides : ps ≤ t.side ∧ pq ≤ t.side
  is_rectangle : pq = Real.sqrt 3 * ps
  area : ps * pq = 28 * Real.sqrt 3

-- Define the theorem
theorem inscribed_rectangle_pc_length 
  (t : EquilateralTriangle) 
  (r : InscribedRectangle t) : 
  ∃ (pc : ℝ), pc = 2 * Real.sqrt 7 ∧ 
  pc^2 = t.side^2 + (t.side - r.ps)^2 - 2 * t.side * (t.side - r.ps) * Real.cos (π/3) :=
sorry

end NUMINAMATH_CALUDE_inscribed_rectangle_pc_length_l406_40695


namespace NUMINAMATH_CALUDE_initial_points_count_l406_40669

/-- Represents the number of points after performing the point-adding operation n times -/
def pointsAfterOperations (k : ℕ) (n : ℕ) : ℕ :=
  match n with
  | 0 => k
  | n + 1 => 2 * (pointsAfterOperations k n) - 1

/-- The theorem stating that if 101 points result after two operations, then 26 points were initially marked -/
theorem initial_points_count : 
  ∀ k : ℕ, pointsAfterOperations k 2 = 101 → k = 26 := by
  sorry

end NUMINAMATH_CALUDE_initial_points_count_l406_40669


namespace NUMINAMATH_CALUDE_merry_go_round_revolutions_l406_40632

/-- The number of revolutions needed for the second horse to cover the same distance as the first horse on a merry-go-round -/
theorem merry_go_round_revolutions (r1 r2 n1 : ℝ) (hr1 : r1 = 30) (hr2 : r2 = 10) (hn1 : n1 = 40) :
  let d1 := 2 * Real.pi * r1 * n1
  let n2 := d1 / (2 * Real.pi * r2)
  n2 = 120 := by sorry

end NUMINAMATH_CALUDE_merry_go_round_revolutions_l406_40632


namespace NUMINAMATH_CALUDE_two_thousand_twentieth_digit_l406_40685

/-- Represents the decimal number formed by concatenating integers from 1 to 1000 -/
def x : ℚ := sorry

/-- Returns the nth digit after the decimal point in the number x -/
def nth_digit (n : ℕ) : ℕ := sorry

/-- The 2020th digit after the decimal point in x is 7 -/
theorem two_thousand_twentieth_digit : nth_digit 2020 = 7 := by sorry

end NUMINAMATH_CALUDE_two_thousand_twentieth_digit_l406_40685


namespace NUMINAMATH_CALUDE_arithmetic_expression_equality_l406_40675

theorem arithmetic_expression_equality : 12 - 7 * (-32) + 16 / (-4) = 232 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_expression_equality_l406_40675


namespace NUMINAMATH_CALUDE_cube_root_of_456533_l406_40677

theorem cube_root_of_456533 (z : ℤ) :
  z^3 = 456533 → z = 77 := by
  sorry

end NUMINAMATH_CALUDE_cube_root_of_456533_l406_40677


namespace NUMINAMATH_CALUDE_intersection_point_l406_40690

/-- Two lines in a 2D plane -/
structure TwoLines where
  line1 : ℝ → ℝ → Prop
  line2 : ℝ → ℝ → Prop

/-- The given two lines -/
def givenLines : TwoLines where
  line1 := fun x y => x - y = 0
  line2 := fun x y => 3 * x + 2 * y - 5 = 0

/-- Theorem: The point (1, 1) is the unique intersection of the given lines -/
theorem intersection_point (l : TwoLines := givenLines) :
  (∃! p : ℝ × ℝ, l.line1 p.1 p.2 ∧ l.line2 p.1 p.2) ∧
  (l.line1 1 1 ∧ l.line2 1 1) :=
sorry

end NUMINAMATH_CALUDE_intersection_point_l406_40690


namespace NUMINAMATH_CALUDE_garden_operations_result_l406_40612

/-- Represents the quantities of vegetables in the garden -/
structure VegetableQuantities where
  tomatoes : ℕ
  potatoes : ℕ
  cucumbers : ℕ
  cabbages : ℕ

/-- Calculates the final quantities of vegetables after operations -/
def final_quantities (initial : VegetableQuantities) 
  (picked_tomatoes picked_potatoes picked_cabbages : ℕ)
  (new_cucumber_plants new_cabbage_plants : ℕ)
  (cucumber_yield cabbage_yield : ℕ) : VegetableQuantities :=
  { tomatoes := initial.tomatoes - picked_tomatoes,
    potatoes := initial.potatoes - picked_potatoes,
    cucumbers := initial.cucumbers + new_cucumber_plants * cucumber_yield,
    cabbages := initial.cabbages - picked_cabbages + new_cabbage_plants * cabbage_yield }

theorem garden_operations_result :
  let initial := VegetableQuantities.mk 500 400 300 100
  let final := final_quantities initial 325 270 50 200 80 2 3
  final.tomatoes = 175 ∧ 
  final.potatoes = 130 ∧ 
  final.cucumbers = 700 ∧ 
  final.cabbages = 290 := by
  sorry

end NUMINAMATH_CALUDE_garden_operations_result_l406_40612


namespace NUMINAMATH_CALUDE_problem_solution_l406_40687

theorem problem_solution (m n : ℝ) 
  (h1 : m^2 + m*n = -3) 
  (h2 : n^2 - 3*m*n = 18) : 
  m^2 + 4*m*n - n^2 = -21 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l406_40687


namespace NUMINAMATH_CALUDE_rectangular_solid_surface_area_l406_40666

/-- The surface area of a rectangular solid with edge lengths 2, 3, and 4 is 52 -/
theorem rectangular_solid_surface_area : 
  let a : ℝ := 4
  let b : ℝ := 3
  let h : ℝ := 2
  2 * (a * b + b * h + a * h) = 52 := by sorry

end NUMINAMATH_CALUDE_rectangular_solid_surface_area_l406_40666


namespace NUMINAMATH_CALUDE_exponential_inequality_l406_40624

theorem exponential_inequality (a b : ℝ) (h1 : a * b ≠ 0) (h2 : a < b) : 
  (2 : ℝ) ^ a < (2 : ℝ) ^ b := by
  sorry

end NUMINAMATH_CALUDE_exponential_inequality_l406_40624


namespace NUMINAMATH_CALUDE_rectangle_width_length_ratio_l406_40607

/-- Given a rectangle with length 12 and perimeter 36, prove that the ratio of its width to its length is 1:2 -/
theorem rectangle_width_length_ratio (w : ℝ) : 
  w > 0 → -- width is positive
  12 > 0 → -- length is positive
  2 * w + 2 * 12 = 36 → -- perimeter formula
  w / 12 = 1 / 2 := by sorry

end NUMINAMATH_CALUDE_rectangle_width_length_ratio_l406_40607


namespace NUMINAMATH_CALUDE_divisibility_of_integer_part_l406_40605

theorem divisibility_of_integer_part (m : ℕ) 
  (h_odd : m % 2 = 1) 
  (h_not_div_3 : m % 3 ≠ 0) : 
  ∃ k : ℤ, (4^m : ℝ) - (2 + Real.sqrt 2)^m = k + (112 : ℝ) * ↑(⌊((4^m : ℝ) - (2 + Real.sqrt 2)^m) / 112⌋) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_of_integer_part_l406_40605


namespace NUMINAMATH_CALUDE_lattice_points_on_hyperbola_l406_40606

theorem lattice_points_on_hyperbola : 
  ∃! (points : Finset (ℤ × ℤ)), 
    points.card = 4 ∧ 
    ∀ (x y : ℤ), (x, y) ∈ points ↔ x^2 - y^2 = 61 := by
  sorry

end NUMINAMATH_CALUDE_lattice_points_on_hyperbola_l406_40606


namespace NUMINAMATH_CALUDE_lcm_of_330_and_210_l406_40610

theorem lcm_of_330_and_210 (hcf : ℕ) (a b lcm : ℕ) : 
  hcf = 30 → a = 330 → b = 210 → lcm = Nat.lcm a b → lcm = 2310 := by
sorry

end NUMINAMATH_CALUDE_lcm_of_330_and_210_l406_40610


namespace NUMINAMATH_CALUDE_dvd_pack_cost_l406_40636

/-- Given that 6 packs of DVDs can be bought with 120 dollars, 
    prove that each pack costs 20 dollars. -/
theorem dvd_pack_cost (total_cost : ℕ) (num_packs : ℕ) (cost_per_pack : ℕ) 
    (h1 : total_cost = 120) 
    (h2 : num_packs = 6) 
    (h3 : total_cost = num_packs * cost_per_pack) : 
  cost_per_pack = 20 := by
  sorry

end NUMINAMATH_CALUDE_dvd_pack_cost_l406_40636


namespace NUMINAMATH_CALUDE_angle_between_vectors_l406_40698

/-- Given two vectors a and b in ℝ², where a = (3, 0) and a + 2b = (1, 2√3),
    prove that the angle between a and b is 120°. -/
theorem angle_between_vectors (b : ℝ × ℝ) : 
  let a : ℝ × ℝ := (3, 0)
  (a.1 + 2 * b.1, a.2 + 2 * b.2) = (1, 2 * Real.sqrt 3) →
  Real.arccos ((a.1 * b.1 + a.2 * b.2) / 
    (Real.sqrt (a.1^2 + a.2^2) * Real.sqrt (b.1^2 + b.2^2))) = 2 * π / 3 := by
  sorry

end NUMINAMATH_CALUDE_angle_between_vectors_l406_40698


namespace NUMINAMATH_CALUDE_digit_product_equation_l406_40696

def digit_product (k : ℕ) : ℕ :=
  if k = 0 then 0
  else if k < 10 then k
  else (k % 10) * digit_product (k / 10)

theorem digit_product_equation : 
  ∀ k : ℕ, k > 0 → (digit_product k = (25 * k) / 8 - 211) ↔ (k = 72 ∨ k = 88) :=
sorry

end NUMINAMATH_CALUDE_digit_product_equation_l406_40696


namespace NUMINAMATH_CALUDE_fruit_arrangement_count_l406_40613

theorem fruit_arrangement_count : 
  let total_fruits : ℕ := 10
  let apple_count : ℕ := 4
  let orange_count : ℕ := 3
  let banana_count : ℕ := 2
  let grape_count : ℕ := 1
  apple_count + orange_count + banana_count + grape_count = total_fruits →
  (Nat.factorial total_fruits) / 
  ((Nat.factorial apple_count) * (Nat.factorial orange_count) * 
   (Nat.factorial banana_count) * (Nat.factorial grape_count)) = 12600 := by
sorry

end NUMINAMATH_CALUDE_fruit_arrangement_count_l406_40613


namespace NUMINAMATH_CALUDE_friend_jogging_time_l406_40681

/-- Proves that if a person completes a route in 3 hours, and another person travels at twice the speed of the first person, then the second person will complete the same route in 90 minutes. -/
theorem friend_jogging_time (my_time : ℝ) (friend_speed : ℝ) (my_speed : ℝ) :
  my_time = 3 →
  friend_speed = 2 * my_speed →
  friend_speed * (90 / 60) = my_speed * my_time :=
by
  sorry

#check friend_jogging_time

end NUMINAMATH_CALUDE_friend_jogging_time_l406_40681


namespace NUMINAMATH_CALUDE_circle_radius_l406_40678

theorem circle_radius (x y : ℝ) :
  (16 * x^2 + 32 * x + 16 * y^2 - 48 * y + 76 = 0) →
  ∃ (center_x center_y : ℝ), 
    (x - center_x)^2 + (y - center_y)^2 = 3/2 :=
by sorry

end NUMINAMATH_CALUDE_circle_radius_l406_40678


namespace NUMINAMATH_CALUDE_walking_speed_calculation_l406_40621

/-- Given a person who runs at 8 km/hr and covers a total distance of 16 km
    (half walking, half running) in 3 hours, prove that the walking speed is 4 km/hr. -/
theorem walking_speed_calculation (running_speed : ℝ) (total_distance : ℝ) (total_time : ℝ) :
  running_speed = 8 →
  total_distance = 16 →
  total_time = 3 →
  ∃ (walking_speed : ℝ),
    walking_speed * (total_distance / 2 / walking_speed) +
    running_speed * (total_distance / 2 / running_speed) = total_time ∧
    walking_speed = 4 :=
by sorry

end NUMINAMATH_CALUDE_walking_speed_calculation_l406_40621


namespace NUMINAMATH_CALUDE_unique_function_divisibility_l406_40644

theorem unique_function_divisibility 
  (f : ℕ+ → ℕ+) 
  (h : ∀ (m n : ℕ+), (m^2 + f n) ∣ (m * f m + n)) : 
  ∀ (n : ℕ+), f n = n :=
sorry

end NUMINAMATH_CALUDE_unique_function_divisibility_l406_40644


namespace NUMINAMATH_CALUDE_fifth_term_of_arithmetic_sequence_l406_40619

def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

theorem fifth_term_of_arithmetic_sequence 
  (a : ℕ → ℤ) 
  (h_arith : arithmetic_sequence a) 
  (h_first : a 1 = 6) 
  (h_third : a 3 = 2) : 
  a 5 = -2 := by
sorry

end NUMINAMATH_CALUDE_fifth_term_of_arithmetic_sequence_l406_40619


namespace NUMINAMATH_CALUDE_polynomial_division_remainder_l406_40692

theorem polynomial_division_remainder : 
  let p (x : ℚ) := x^4 - 4*x^3 + 13*x^2 - 14*x + 4
  let d (x : ℚ) := x^2 - 3*x + 13/3
  let q (x : ℚ) := x^2 - x + 10/3
  let r (x : ℚ) := 2*x + 16/9
  ∀ x, p x = d x * q x + r x := by
sorry

end NUMINAMATH_CALUDE_polynomial_division_remainder_l406_40692


namespace NUMINAMATH_CALUDE_min_tan_product_acute_triangle_l406_40688

theorem min_tan_product_acute_triangle (A B C : ℝ) (h_acute : 0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π) 
  (h_sin : Real.sin A = 3 * Real.sin B * Real.sin C) : 
  (∀ A' B' C', 0 < A' ∧ 0 < B' ∧ 0 < C' ∧ A' + B' + C' = π → 
    Real.sin A' = 3 * Real.sin B' * Real.sin C' → 
    12 ≤ Real.tan A' * Real.tan B' * Real.tan C') ∧
  (∃ A₀ B₀ C₀, 0 < A₀ ∧ 0 < B₀ ∧ 0 < C₀ ∧ A₀ + B₀ + C₀ = π ∧
    Real.sin A₀ = 3 * Real.sin B₀ * Real.sin C₀ ∧
    Real.tan A₀ * Real.tan B₀ * Real.tan C₀ = 12) := by
  sorry

end NUMINAMATH_CALUDE_min_tan_product_acute_triangle_l406_40688


namespace NUMINAMATH_CALUDE_polynomial_factorization_l406_40616

theorem polynomial_factorization :
  ∀ x : ℝ, x^15 + x^7 + 1 = (x^2 + x + 1) * (x^13 - x^12 + x^10 - x^9 + x^7 - x^6 + x^4 - x^3 + x - 1) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l406_40616


namespace NUMINAMATH_CALUDE_b_200_equals_179101_l406_40627

/-- Sequence a_n defined as n(n+1)/2 -/
def a (n : ℕ) : ℕ := n * (n + 1) / 2

/-- Predicate to check if a number is not divisible by 3 -/
def not_divisible_by_three (n : ℕ) : Prop := n % 3 ≠ 0

/-- Sequence b_n derived from a_n by removing terms divisible by 3 and rearranging -/
def b (n : ℕ) : ℕ := a (3 * n - 2)

/-- Theorem stating that the 200th term of sequence b_n is 179101 -/
theorem b_200_equals_179101 : b 200 = 179101 := by sorry

end NUMINAMATH_CALUDE_b_200_equals_179101_l406_40627


namespace NUMINAMATH_CALUDE_unique_intersection_implies_line_equation_l406_40683

/-- Given a line y = mx + b passing through (2, 3), prove that if there exists exactly one k
    where x = k intersects y = x^2 - 4x + 4 and y = mx + b at points 6 units apart,
    then m = -6 and b = 15 -/
theorem unique_intersection_implies_line_equation 
  (m b : ℝ) 
  (passes_through : 3 = 2 * m + b) 
  (h : ∃! k : ℝ, ∃ y₁ y₂ : ℝ, 
    y₁ = k^2 - 4*k + 4 ∧ 
    y₂ = m*k + b ∧ 
    (y₁ - y₂)^2 = 36) : 
  m = -6 ∧ b = 15 := by
sorry

end NUMINAMATH_CALUDE_unique_intersection_implies_line_equation_l406_40683


namespace NUMINAMATH_CALUDE_polygon_interior_angles_sum_l406_40618

theorem polygon_interior_angles_sum (n : ℕ) :
  (180 * (n - 2) = 1260) →
  (180 * ((n + 3) - 2) = 1800) := by
  sorry

end NUMINAMATH_CALUDE_polygon_interior_angles_sum_l406_40618


namespace NUMINAMATH_CALUDE_rational_powers_imply_rational_irrational_with_rational_powers_l406_40662

-- Define rationality for real numbers
def IsRational (x : ℝ) : Prop := ∃ (q : ℚ), x = q

-- Part a
theorem rational_powers_imply_rational (x : ℝ) 
  (h1 : IsRational (x^7)) (h2 : IsRational (x^12)) : 
  IsRational x := by sorry

-- Part b
theorem irrational_with_rational_powers : 
  ∃ (x : ℝ), IsRational (x^9) ∧ IsRational (x^12) ∧ ¬IsRational x := by sorry

end NUMINAMATH_CALUDE_rational_powers_imply_rational_irrational_with_rational_powers_l406_40662


namespace NUMINAMATH_CALUDE_inequality_solution_implies_a_range_l406_40615

theorem inequality_solution_implies_a_range (a : ℝ) : 
  (∃ x : ℝ, 1 < x ∧ x < 4 ∧ 2 * x^2 - 8 * x - 4 - a > 0) → a < -4 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_implies_a_range_l406_40615


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l406_40608

-- Define the quadratic function
def f (m : ℝ) (x : ℝ) : ℝ := x^2 + m*x + 9

-- Define the condition for no real roots
def has_no_real_roots (m : ℝ) : Prop := ∀ x : ℝ, f m x ≠ 0

-- Define the given interval
def p (m : ℝ) : Prop := -6 ≤ m ∧ m ≤ 6

-- Theorem statement
theorem sufficient_not_necessary :
  (∀ m : ℝ, p m → has_no_real_roots m) ∧
  ¬(∀ m : ℝ, has_no_real_roots m → p m) :=
sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l406_40608


namespace NUMINAMATH_CALUDE_square_cube_relation_l406_40649

theorem square_cube_relation (a b : ℝ) :
  ¬(∀ a b : ℝ, a^2 > b^2 → a^3 > b^3) ∧
  ¬(∀ a b : ℝ, a^3 > b^3 → a^2 > b^2) :=
sorry

end NUMINAMATH_CALUDE_square_cube_relation_l406_40649


namespace NUMINAMATH_CALUDE_split_sum_equals_capacity_l406_40674

/-- The capacity of a pile with n stones -/
def capacity (n : ℕ) : ℕ := n * (n - 1) / 2

/-- The sum of products obtained by splitting n stones -/
def split_sum (n : ℕ) : ℕ := sorry

theorem split_sum_equals_capacity :
  split_sum 2019 = capacity 2019 :=
sorry

end NUMINAMATH_CALUDE_split_sum_equals_capacity_l406_40674


namespace NUMINAMATH_CALUDE_hyperbola_focus_l406_40614

-- Define the hyperbola equation
def hyperbola_eq (x y : ℝ) : Prop :=
  x^2 - 4*y^2 - 6*x + 24*y - 11 = 0

-- Define the foci coordinates
def focus_coord (x y : ℝ) : Prop :=
  (x = 3 ∧ y = 3 + 2 * Real.sqrt 5) ∨ (x = 3 ∧ y = 3 - 2 * Real.sqrt 5)

-- Theorem statement
theorem hyperbola_focus :
  ∃ (x y : ℝ), hyperbola_eq x y ∧ focus_coord x y :=
sorry

end NUMINAMATH_CALUDE_hyperbola_focus_l406_40614


namespace NUMINAMATH_CALUDE_min_value_expression_l406_40600

theorem min_value_expression (a b : ℝ) (h1 : a > b) (h2 : b > 0) :
  a^2 + 1/(a*b) + 1/(a*(a-b)) ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l406_40600


namespace NUMINAMATH_CALUDE_sum_of_digits_power_two_l406_40611

/-- Sum of digits function -/
def s (n : ℕ) : ℕ := sorry

/-- Main theorem -/
theorem sum_of_digits_power_two : 
  (∀ n : ℕ, (n - s n) % 9 = 0) → 
  (2^2009 < 10^672) → 
  s (s (s (2^2009))) = 5 := by sorry

end NUMINAMATH_CALUDE_sum_of_digits_power_two_l406_40611


namespace NUMINAMATH_CALUDE_quadratic_completion_l406_40639

theorem quadratic_completion (y : ℝ) : ∃ b : ℝ, y^2 + 14*y + 60 = (y + b)^2 + 11 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_completion_l406_40639


namespace NUMINAMATH_CALUDE_z_gets_30_paisa_l406_40686

/-- Represents the division of money among three parties -/
structure MoneyDivision where
  total : ℚ
  y_share : ℚ
  y_rate : ℚ

/-- Calculates the share of z given a money division -/
def z_share (md : MoneyDivision) : ℚ :=
  md.total - md.y_share - (md.y_share / md.y_rate)

/-- Calculates the rate at which z receives money compared to x -/
def z_rate (md : MoneyDivision) : ℚ :=
  (z_share md) / (md.y_share / md.y_rate)

/-- Theorem stating that z gets 30 paisa for each rupee x gets -/
theorem z_gets_30_paisa (md : MoneyDivision) 
  (h1 : md.total = 105)
  (h2 : md.y_share = 27)
  (h3 : md.y_rate = 45/100) : 
  z_rate md = 30/100 := by
  sorry

end NUMINAMATH_CALUDE_z_gets_30_paisa_l406_40686


namespace NUMINAMATH_CALUDE_shorter_diagonal_length_l406_40637

/-- Represents a rhombus with given properties -/
structure Rhombus where
  area : ℝ
  diagonal_ratio : ℝ × ℝ
  diagonal_side_relation : Bool

/-- Theorem: In a rhombus with area 150 and diagonal ratio 5:3, the shorter diagonal is 6√5 -/
theorem shorter_diagonal_length (R : Rhombus) 
    (h_area : R.area = 150)
    (h_ratio : R.diagonal_ratio = (5, 3))
    (h_relation : R.diagonal_side_relation = true) : 
  ∃ (d : ℝ), d = 6 * Real.sqrt 5 ∧ d = min (5 * (2 * Real.sqrt 5)) (3 * (2 * Real.sqrt 5)) :=
by sorry

end NUMINAMATH_CALUDE_shorter_diagonal_length_l406_40637


namespace NUMINAMATH_CALUDE_problem_statement_l406_40668

theorem problem_statement (t : ℝ) :
  let x := 3 - 1.5 * t
  let y := 3 * t + 4
  x = 6 → y = -2 := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l406_40668


namespace NUMINAMATH_CALUDE_polynomial_coefficient_sum_l406_40642

theorem polynomial_coefficient_sum (a₀ a₁ a₂ a₃ : ℝ) :
  (∀ x : ℝ, (5 * x + 4)^3 = a₀ + a₁ * x + a₂ * x^2 + a₃ * x^3) →
  (a₀ + a₂) - (a₁ + a₃) = -1 := by
sorry

end NUMINAMATH_CALUDE_polynomial_coefficient_sum_l406_40642


namespace NUMINAMATH_CALUDE_complex_number_location_l406_40656

open Complex

theorem complex_number_location (z : ℂ) (h : z / (1 + I) = 2 - I) :
  0 < z.re ∧ 0 < z.im :=
by sorry

end NUMINAMATH_CALUDE_complex_number_location_l406_40656


namespace NUMINAMATH_CALUDE_max_area_specific_quadrilateral_l406_40643

/-- A convex quadrilateral with given side lengths -/
structure ConvexQuadrilateral where
  ab : ℝ
  bc : ℝ
  cd : ℝ
  da : ℝ
  convex : ab > 0 ∧ bc > 0 ∧ cd > 0 ∧ da > 0

/-- The area of a convex quadrilateral -/
def area (q : ConvexQuadrilateral) : ℝ :=
  sorry

/-- Theorem: Maximum area of a specific convex quadrilateral -/
theorem max_area_specific_quadrilateral :
  ∃ (q : ConvexQuadrilateral),
    q.ab = 2 ∧ q.bc = 4 ∧ q.cd = 5 ∧ q.da = 3 ∧
    ∀ (q' : ConvexQuadrilateral),
      q'.ab = 2 → q'.bc = 4 → q'.cd = 5 → q'.da = 3 →
      area q' ≤ 2 * Real.sqrt 30 :=
  sorry

end NUMINAMATH_CALUDE_max_area_specific_quadrilateral_l406_40643


namespace NUMINAMATH_CALUDE_necklace_cost_l406_40665

/-- The cost of Scarlet's necklace given her savings and expenses -/
theorem necklace_cost (savings : ℕ) (earrings_cost : ℕ) (remaining : ℕ) : 
  savings = 80 → earrings_cost = 23 → remaining = 9 → 
  savings - earrings_cost - remaining = 48 := by
  sorry

end NUMINAMATH_CALUDE_necklace_cost_l406_40665


namespace NUMINAMATH_CALUDE_residual_plot_vertical_axis_l406_40671

/-- Represents a residual plot in regression analysis -/
structure ResidualPlot where
  verticalAxis : Set ℝ
  horizontalAxis : Set ℝ

/-- Definition of a residual in regression analysis -/
def Residual : Type := ℝ

/-- Theorem stating that the vertical axis of a residual plot represents residuals -/
theorem residual_plot_vertical_axis (plot : ResidualPlot) : 
  plot.verticalAxis = Set.range (λ r : Residual => r) := by
  sorry

end NUMINAMATH_CALUDE_residual_plot_vertical_axis_l406_40671


namespace NUMINAMATH_CALUDE_product_of_fifth_and_eighth_l406_40630

/-- A geometric progression with terms a_n -/
def geometric_progression (a : ℕ → ℝ) : Prop :=
  ∃ (r : ℝ) (a₁ : ℝ), ∀ n : ℕ, a n = a₁ * r^(n - 1)

/-- The 3rd and 10th terms are roots of x^2 - 3x - 5 = 0 -/
def roots_condition (a : ℕ → ℝ) : Prop :=
  (a 3)^2 - 3*(a 3) - 5 = 0 ∧ (a 10)^2 - 3*(a 10) - 5 = 0

theorem product_of_fifth_and_eighth (a : ℕ → ℝ) 
  (h1 : geometric_progression a) (h2 : roots_condition a) : 
  a 5 * a 8 = -5 := by
  sorry

end NUMINAMATH_CALUDE_product_of_fifth_and_eighth_l406_40630


namespace NUMINAMATH_CALUDE_integral_cos_squared_sin_l406_40625

theorem integral_cos_squared_sin (x : Real) :
  deriv (fun x => -Real.cos x ^ 3 / 3) x = Real.cos x ^ 2 * Real.sin x := by
  sorry

end NUMINAMATH_CALUDE_integral_cos_squared_sin_l406_40625


namespace NUMINAMATH_CALUDE_x_plus_y_equals_ten_l406_40653

theorem x_plus_y_equals_ten (x y : ℝ) 
  (hx : x + Real.log x / Real.log 10 = 10) 
  (hy : y + 10^y = 10) : 
  x + y = 10 := by
sorry

end NUMINAMATH_CALUDE_x_plus_y_equals_ten_l406_40653


namespace NUMINAMATH_CALUDE_operation_property_l406_40679

theorem operation_property (h : ℕ → ℝ) (k : ℝ) (n : ℕ) 
  (h_def : ∀ m l : ℕ, h (m + l) = h m * h l) 
  (h_2 : h 2 = k) 
  (k_nonzero : k ≠ 0) : 
  h (2 * n) * h 2024 = k^(n + 1012) := by
  sorry

end NUMINAMATH_CALUDE_operation_property_l406_40679


namespace NUMINAMATH_CALUDE_clara_number_problem_l406_40658

theorem clara_number_problem (x : ℝ) : 2 * x + 3 = 23 → x = 10 := by
  sorry

end NUMINAMATH_CALUDE_clara_number_problem_l406_40658


namespace NUMINAMATH_CALUDE_least_subtrahend_for_divisibility_specific_case_l406_40652

theorem least_subtrahend_for_divisibility (n : Nat) (d : Nat) (h : d > 0) :
  ∃ (k : Nat), k < d ∧ (n - k) % d = 0 ∧ ∀ (m : Nat), m < k → (n - m) % d ≠ 0 :=
by
  sorry

theorem specific_case : 
  ∃ (k : Nat), k < 47 ∧ (929 - k) % 47 = 0 ∧ ∀ (m : Nat), m < k → (929 - m) % 47 ≠ 0 ∧ k = 44 :=
by
  sorry

end NUMINAMATH_CALUDE_least_subtrahend_for_divisibility_specific_case_l406_40652


namespace NUMINAMATH_CALUDE_find_b_l406_40629

-- Define the inverse relationship between a² and √b
def inverse_relation (a b : ℝ) : Prop := ∃ k : ℝ, a^2 * Real.sqrt b = k

-- Define the conditions
def condition1 : ℝ := 3
def condition2 : ℝ := 36

-- Define the target equation
def target_equation (a b : ℝ) : Prop := a * b = 54

-- Theorem statement
theorem find_b :
  ∀ a b : ℝ,
  inverse_relation a b →
  inverse_relation condition1 condition2 →
  target_equation a b →
  b = 18 * (4^(1/3)) :=
by
  sorry

end NUMINAMATH_CALUDE_find_b_l406_40629


namespace NUMINAMATH_CALUDE_polynomial_never_equals_33_l406_40617

theorem polynomial_never_equals_33 (x y : ℤ) : 
  x^5 + 3*x^4*y - 5*x^3*y^2 - 15*x^2*y^3 + 4*x*y^4 + 12*y^5 ≠ 33 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_never_equals_33_l406_40617


namespace NUMINAMATH_CALUDE_road_building_divisibility_l406_40601

/-- Represents the number of ways to build roads between n cities with the given constraints -/
def T (n : ℕ) : ℕ :=
  sorry  -- Definition of T_n based on the problem constraints

/-- The main theorem to be proved -/
theorem road_building_divisibility (n : ℕ) (h : n > 1) :
  (n % 2 = 1 → n ∣ T n) ∧ (n % 2 = 0 → (n / 2) ∣ T n) :=
by sorry

end NUMINAMATH_CALUDE_road_building_divisibility_l406_40601


namespace NUMINAMATH_CALUDE_both_companies_participate_both_will_participate_l406_40657

/-- Represents a company in country A --/
structure Company where
  expectedIncome : ℝ
  investmentCost : ℝ

/-- The market conditions for the new technology development --/
structure MarketConditions where
  V : ℝ  -- Income if developed alone
  α : ℝ  -- Probability of success
  IC : ℝ  -- Investment cost
  h1 : 0 < α
  h2 : α < 1

/-- Calculate the expected income for a company when both participate --/
def expectedIncomeBothParticipate (m : MarketConditions) : ℝ :=
  m.α * (1 - m.α) * m.V + 0.5 * m.α^2 * m.V

/-- Theorem: Condition for both companies to participate --/
theorem both_companies_participate (m : MarketConditions) :
  expectedIncomeBothParticipate m - m.IC ≥ 0 ↔
  m.α * (1 - m.α) * m.V + 0.5 * m.α^2 * m.V - m.IC ≥ 0 := by
  sorry

/-- Function to determine if a company will participate --/
def willParticipate (c : Company) (m : MarketConditions) : Prop :=
  c.expectedIncome - c.investmentCost ≥ 0

/-- Theorem: Both companies will participate if the condition is met --/
theorem both_will_participate (c1 c2 : Company) (m : MarketConditions)
  (h : expectedIncomeBothParticipate m - m.IC ≥ 0) :
  willParticipate c1 m ∧ willParticipate c2 m := by
  sorry

end NUMINAMATH_CALUDE_both_companies_participate_both_will_participate_l406_40657


namespace NUMINAMATH_CALUDE_bombardment_percentage_approx_10_percent_l406_40663

def initial_population : ℕ := 8515
def final_population : ℕ := 6514
def departure_rate : ℚ := 15 / 100

def bombardment_percentage : ℚ :=
  (initial_population - final_population) / initial_population * 100

theorem bombardment_percentage_approx_10_percent :
  ∃ (ε : ℚ), ε > 0 ∧ ε < 1 ∧ 
  abs (bombardment_percentage - 10) < ε ∧
  final_population = 
    initial_population * (1 - bombardment_percentage / 100) * (1 - departure_rate) :=
by sorry

end NUMINAMATH_CALUDE_bombardment_percentage_approx_10_percent_l406_40663


namespace NUMINAMATH_CALUDE_greater_number_problem_l406_40650

theorem greater_number_problem (x y : ℝ) (sum_eq : x + y = 40) (diff_eq : x - y = 10) :
  max x y = 25 := by sorry

end NUMINAMATH_CALUDE_greater_number_problem_l406_40650


namespace NUMINAMATH_CALUDE_exists_thirteen_cubes_l406_40626

/-- Represents a 4x4 board with cube stacks -/
def Board := Fin 4 → Fin 4 → ℕ

/-- Predicate to check if the board configuration is valid -/
def valid_board (b : Board) : Prop :=
  ∀ n : Fin 8, ∃! (i j k l : Fin 4), 
    b i j = n + 1 ∧ b k l = n + 1 ∧ (i ≠ k ∨ j ≠ l)

/-- Theorem stating that there exists a pair of cells with 13 cubes total -/
theorem exists_thirteen_cubes (b : Board) (h : valid_board b) : 
  ∃ (i j k l : Fin 4), b i j + b k l = 13 :=
sorry

end NUMINAMATH_CALUDE_exists_thirteen_cubes_l406_40626


namespace NUMINAMATH_CALUDE_hockey_tournament_games_l406_40620

/-- The number of teams in the hockey league --/
def num_teams : ℕ := 7

/-- The number of times each team plays against every other team --/
def games_per_matchup : ℕ := 4

/-- The total number of games played in the tournament --/
def total_games : ℕ := num_teams * (num_teams - 1) / 2 * games_per_matchup

theorem hockey_tournament_games :
  total_games = 84 :=
by sorry

end NUMINAMATH_CALUDE_hockey_tournament_games_l406_40620


namespace NUMINAMATH_CALUDE_min_value_expression_l406_40667

theorem min_value_expression (m n : ℝ) (h : m - n^2 = 1) :
  ∃ (min : ℝ), min = 4 ∧ ∀ (x y : ℝ), x - y^2 = 1 → x^2 + 2*y^2 + 4*x - 1 ≥ min :=
sorry

end NUMINAMATH_CALUDE_min_value_expression_l406_40667


namespace NUMINAMATH_CALUDE_concentric_circles_area_l406_40641

theorem concentric_circles_area (r₁ : ℝ) (chord_length : ℝ) (h₁ : r₁ = 50) (h₂ : chord_length = 120) : 
  let r₂ := Real.sqrt (r₁^2 + (chord_length/2)^2)
  π * (r₂^2 - r₁^2) = 3600 * π := by
sorry

end NUMINAMATH_CALUDE_concentric_circles_area_l406_40641


namespace NUMINAMATH_CALUDE_connie_marbles_proof_l406_40659

/-- The number of marbles Connie started with -/
def initial_marbles : ℕ := 776

/-- The number of marbles Connie gave to Juan -/
def marbles_given : ℕ := 183

/-- The number of marbles Connie has left -/
def remaining_marbles : ℕ := initial_marbles - marbles_given

theorem connie_marbles_proof : remaining_marbles = 593 := by
  sorry

end NUMINAMATH_CALUDE_connie_marbles_proof_l406_40659


namespace NUMINAMATH_CALUDE_carnival_tickets_l406_40633

/-- Calculates the total number of tickets used at a carnival -/
theorem carnival_tickets (ferris_wheel_rides bumper_car_rides roller_coaster_rides teacup_rides : ℕ)
                         (ferris_wheel_cost bumper_car_cost roller_coaster_cost teacup_cost : ℕ) :
  ferris_wheel_rides * ferris_wheel_cost +
  bumper_car_rides * bumper_car_cost +
  roller_coaster_rides * roller_coaster_cost +
  teacup_rides * teacup_cost = 105 :=
by
  -- Assuming ferris_wheel_rides = 7, bumper_car_rides = 3, roller_coaster_rides = 4, teacup_rides = 5
  -- and ferris_wheel_cost = 5, bumper_car_cost = 6, roller_coaster_cost = 8, teacup_cost = 4
  sorry


end NUMINAMATH_CALUDE_carnival_tickets_l406_40633


namespace NUMINAMATH_CALUDE_teal_color_survey_l406_40646

theorem teal_color_survey (total : ℕ) (green : ℕ) (both : ℕ) (neither : ℕ) 
  (h_total : total = 120)
  (h_green : green = 70)
  (h_both : both = 35)
  (h_neither : neither = 20) :
  ∃ blue : ℕ, blue = 65 ∧ 
    blue + green - both + neither = total :=
by sorry

end NUMINAMATH_CALUDE_teal_color_survey_l406_40646
