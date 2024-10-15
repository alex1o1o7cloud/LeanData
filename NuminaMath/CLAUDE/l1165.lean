import Mathlib

namespace NUMINAMATH_CALUDE_root_difference_of_equation_l1165_116513

theorem root_difference_of_equation : ∃ a b : ℝ,
  (∀ x : ℝ, (6 * x - 18) / (x^2 + 3 * x - 28) = x + 3 ↔ x = a ∨ x = b) ∧
  a > b ∧
  a - b = 2 := by
sorry

end NUMINAMATH_CALUDE_root_difference_of_equation_l1165_116513


namespace NUMINAMATH_CALUDE_lines_perpendicular_to_plane_are_parallel_l1165_116560

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular and parallel relations
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Line → Line → Prop)

-- State the theorem
theorem lines_perpendicular_to_plane_are_parallel 
  (m n : Line) (α : Plane) :
  perpendicular m α → perpendicular n α → parallel m n :=
sorry

end NUMINAMATH_CALUDE_lines_perpendicular_to_plane_are_parallel_l1165_116560


namespace NUMINAMATH_CALUDE_hole_filling_proof_l1165_116522

/-- The amount of water initially in the hole -/
def initial_water : ℕ := 676

/-- The additional amount of water needed to fill the hole -/
def additional_water : ℕ := 147

/-- The total amount of water needed to fill the hole -/
def total_water : ℕ := initial_water + additional_water

theorem hole_filling_proof : total_water = 823 := by
  sorry

end NUMINAMATH_CALUDE_hole_filling_proof_l1165_116522


namespace NUMINAMATH_CALUDE_elective_schemes_count_l1165_116562

/-- The number of courses offered -/
def total_courses : ℕ := 10

/-- The number of courses that can't be chosen together -/
def conflicting_courses : ℕ := 3

/-- The number of courses each student must choose -/
def courses_to_choose : ℕ := 3

/-- The number of different elective schemes -/
def num_elective_schemes : ℕ := 98

theorem elective_schemes_count :
  (total_courses = 10) →
  (conflicting_courses = 3) →
  (courses_to_choose = 3) →
  (num_elective_schemes = Nat.choose (total_courses - conflicting_courses) courses_to_choose +
                          conflicting_courses * Nat.choose (total_courses - conflicting_courses) (courses_to_choose - 1)) :=
by sorry

end NUMINAMATH_CALUDE_elective_schemes_count_l1165_116562


namespace NUMINAMATH_CALUDE_line_equation_specific_l1165_116591

/-- The equation of a line with given slope and y-intercept -/
def line_equation (slope : ℝ) (y_intercept : ℝ) : ℝ → ℝ := λ x => slope * x + y_intercept

/-- Theorem: The equation of a line with slope 2 and y-intercept 1 is y = 2x + 1 -/
theorem line_equation_specific : line_equation 2 1 = λ x => 2 * x + 1 := by
  sorry

end NUMINAMATH_CALUDE_line_equation_specific_l1165_116591


namespace NUMINAMATH_CALUDE_circle_triangle_count_l1165_116559

/-- The number of points on the circle's circumference -/
def n : ℕ := 10

/-- The total number of triangles that can be formed from n points -/
def total_triangles (n : ℕ) : ℕ := n.choose 3

/-- The number of triangles with consecutive vertices -/
def consecutive_triangles (n : ℕ) : ℕ := n

/-- The number of valid triangles (no consecutive vertices) -/
def valid_triangles (n : ℕ) : ℕ := total_triangles n - consecutive_triangles n

theorem circle_triangle_count :
  valid_triangles n = 110 :=
sorry

end NUMINAMATH_CALUDE_circle_triangle_count_l1165_116559


namespace NUMINAMATH_CALUDE_bowling_ball_weight_proof_l1165_116529

/-- The weight of one bowling ball in pounds -/
def bowling_ball_weight : ℝ := 20

/-- The weight of one canoe in pounds -/
def canoe_weight : ℝ := 30

theorem bowling_ball_weight_proof :
  (9 * bowling_ball_weight = 6 * canoe_weight) ∧
  (4 * canoe_weight = 120) →
  bowling_ball_weight = 20 := by
sorry

end NUMINAMATH_CALUDE_bowling_ball_weight_proof_l1165_116529


namespace NUMINAMATH_CALUDE_grain_spilled_calculation_l1165_116577

/-- Calculates the amount of grain spilled into the water -/
def grain_spilled (original : ℕ) (remaining : ℕ) : ℕ :=
  original - remaining

/-- Theorem: The amount of grain spilled is the difference between original and remaining -/
theorem grain_spilled_calculation (original remaining : ℕ) 
  (h1 : original = 50870)
  (h2 : remaining = 918) :
  grain_spilled original remaining = 49952 := by
  sorry

#eval grain_spilled 50870 918

end NUMINAMATH_CALUDE_grain_spilled_calculation_l1165_116577


namespace NUMINAMATH_CALUDE_brownie_pieces_count_l1165_116550

/-- Represents the dimensions of a rectangular object -/
structure Dimensions where
  length : ℕ
  width : ℕ

/-- Calculates the area of a rectangular object given its dimensions -/
def area (d : Dimensions) : ℕ := d.length * d.width

/-- Represents a rectangular pan of brownies -/
structure BrowniePan where
  panDimensions : Dimensions
  pieceDimensions : Dimensions

/-- Calculates the number of brownie pieces that can be cut from the pan -/
def numberOfPieces (pan : BrowniePan) : ℕ :=
  (area pan.panDimensions) / (area pan.pieceDimensions)

theorem brownie_pieces_count :
  let pan : BrowniePan := {
    panDimensions := { length := 24, width := 15 },
    pieceDimensions := { length := 3, width := 2 }
  }
  numberOfPieces pan = 60 := by sorry

end NUMINAMATH_CALUDE_brownie_pieces_count_l1165_116550


namespace NUMINAMATH_CALUDE_data_transmission_time_l1165_116544

/-- Represents the number of blocks of data to be sent -/
def num_blocks : ℕ := 100

/-- Represents the number of chunks in each block -/
def chunks_per_block : ℕ := 450

/-- Represents the transmission rate in chunks per second -/
def transmission_rate : ℕ := 200

/-- Represents the number of seconds in an hour -/
def seconds_per_hour : ℕ := 3600

/-- Theorem stating that the time to send the data is 0.0625 hours -/
theorem data_transmission_time :
  (num_blocks * chunks_per_block : ℚ) / transmission_rate / seconds_per_hour = 0.0625 := by
  sorry

end NUMINAMATH_CALUDE_data_transmission_time_l1165_116544


namespace NUMINAMATH_CALUDE_tea_bags_count_l1165_116569

/-- Represents the number of tea bags in a box -/
def n : ℕ := sorry

/-- Represents the number of cups Natasha made -/
def natasha_cups : ℕ := 41

/-- Represents the number of cups Inna made -/
def inna_cups : ℕ := 58

/-- The number of cups made from Natasha's box is between 2n and 3n -/
axiom natasha_range : 2 * n ≤ natasha_cups ∧ natasha_cups ≤ 3 * n

/-- The number of cups made from Inna's box is between 2n and 3n -/
axiom inna_range : 2 * n ≤ inna_cups ∧ inna_cups ≤ 3 * n

/-- The number of tea bags in the box is 20 -/
theorem tea_bags_count : n = 20 := by sorry

end NUMINAMATH_CALUDE_tea_bags_count_l1165_116569


namespace NUMINAMATH_CALUDE_negation_existence_quadratic_l1165_116566

theorem negation_existence_quadratic (a : ℝ) :
  (¬ ∃ x : ℝ, x^2 + 2*a*x + a ≤ 0) ↔ (∀ x : ℝ, x^2 + 2*a*x + a > 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_existence_quadratic_l1165_116566


namespace NUMINAMATH_CALUDE_reciprocal_proof_l1165_116504

theorem reciprocal_proof (a b : ℝ) 
  (h_pos_a : a > 0) 
  (h_pos_b : b > 0) 
  (h_diff : a ≠ b) 
  (h_eq : 1 / (1 + a) + 1 / (1 + b) = 2 / (1 + Real.sqrt (a * b))) : 
  a * b = 1 := by
sorry

end NUMINAMATH_CALUDE_reciprocal_proof_l1165_116504


namespace NUMINAMATH_CALUDE_max_x_given_lcm_l1165_116597

theorem max_x_given_lcm (x : ℕ) : 
  (Nat.lcm x (Nat.lcm 15 21) = 105) → x ≤ 105 :=
by sorry

end NUMINAMATH_CALUDE_max_x_given_lcm_l1165_116597


namespace NUMINAMATH_CALUDE_circle_intersection_equation_l1165_116501

noncomputable def circle_equation (t : ℝ) (x y : ℝ) : Prop :=
  (x - t)^2 + (y - 2/t)^2 = t^2 + (2/t)^2

theorem circle_intersection_equation :
  ∀ t : ℝ,
  t ≠ 0 →
  circle_equation t 0 0 →
  (∃ a : ℝ, a ≠ 0 ∧ circle_equation t a 0) →
  (∃ b : ℝ, b ≠ 0 ∧ circle_equation t 0 b) →
  (∀ x y : ℝ, 2*x + y = 4 → circle_equation t x y → 
    ∃ m n : ℝ, circle_equation t m n ∧ 2*m + n = 4 ∧ m^2 + n^2 = x^2 + y^2) →
  circle_equation 2 x y ∧ (x - 2)^2 + (y - 1)^2 = 5 :=
sorry

end NUMINAMATH_CALUDE_circle_intersection_equation_l1165_116501


namespace NUMINAMATH_CALUDE_two_in_M_l1165_116519

-- Define the universal set U
def U : Set Nat := {1, 2, 3, 4, 5}

-- Define the complement of M in U
def complementM : Set Nat := {1, 3}

-- Theorem to prove
theorem two_in_M : 2 ∈ (U \ complementM) := by
  sorry

end NUMINAMATH_CALUDE_two_in_M_l1165_116519


namespace NUMINAMATH_CALUDE_partnership_profit_l1165_116555

/-- Given two partners A and B in a business partnership, this theorem proves
    that the total profit is 7 times B's profit under certain conditions. -/
theorem partnership_profit
  (investment_A investment_B : ℝ)
  (period_A period_B : ℝ)
  (profit_B : ℝ)
  (h1 : investment_A = 3 * investment_B)
  (h2 : period_A = 2 * period_B)
  (h3 : profit_B = investment_B * period_B)
  : investment_A * period_A + investment_B * period_B = 7 * profit_B :=
by sorry

end NUMINAMATH_CALUDE_partnership_profit_l1165_116555


namespace NUMINAMATH_CALUDE_michelle_crayons_l1165_116573

/-- The number of crayons Michelle has -/
def total_crayons (num_boxes : ℕ) (crayons_per_box : ℕ) : ℕ :=
  num_boxes * crayons_per_box

/-- Proof that Michelle has 35 crayons -/
theorem michelle_crayons : total_crayons 7 5 = 35 := by
  sorry

end NUMINAMATH_CALUDE_michelle_crayons_l1165_116573


namespace NUMINAMATH_CALUDE_consecutive_even_product_divisible_l1165_116575

theorem consecutive_even_product_divisible (n : ℕ) : 
  ∃ k : ℕ, (2*n) * (2*n + 2) * (2*n + 4) * (2*n + 6) = 240 * k := by
  sorry

end NUMINAMATH_CALUDE_consecutive_even_product_divisible_l1165_116575


namespace NUMINAMATH_CALUDE_problem_solution_l1165_116531

theorem problem_solution : ∃ x : ℝ, (0.2 * 30 = 0.25 * x + 2) ∧ x = 16 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l1165_116531


namespace NUMINAMATH_CALUDE_sin_45_degrees_l1165_116520

theorem sin_45_degrees : Real.sin (π / 4) = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_45_degrees_l1165_116520


namespace NUMINAMATH_CALUDE_brendan_grass_cutting_l1165_116534

/-- Proves that Brendan can cut 84 yards of grass in a week with his new lawnmower -/
theorem brendan_grass_cutting (initial_capacity : ℕ) (increase_percentage : ℚ) (days_in_week : ℕ) :
  initial_capacity = 8 →
  increase_percentage = 1/2 →
  days_in_week = 7 →
  (initial_capacity + initial_capacity * increase_percentage) * days_in_week = 84 :=
by sorry

end NUMINAMATH_CALUDE_brendan_grass_cutting_l1165_116534


namespace NUMINAMATH_CALUDE_grasshopper_jumps_l1165_116506

-- Define a circle
def Circle := ℝ × ℝ → Prop

-- Define a point
def Point := ℝ × ℝ

-- Define a segment
def Segment := Point × Point

-- Define the property of being inside a circle
def InsideCircle (c : Circle) (p : Point) : Prop := sorry

-- Define the property of being on the boundary of a circle
def OnCircleBoundary (c : Circle) (p : Point) : Prop := sorry

-- Define the property of two segments not intersecting
def DoNotIntersect (s1 s2 : Segment) : Prop := sorry

-- Define the property of a point being reachable from another point
def Reachable (c : Circle) (points_inside : List Point) (points_boundary : List Point) (p q : Point) : Prop := sorry

theorem grasshopper_jumps 
  (c : Circle) 
  (n : ℕ) 
  (points_inside : List Point) 
  (points_boundary : List Point) 
  (h1 : points_inside.length = n) 
  (h2 : points_boundary.length = n)
  (h3 : ∀ p ∈ points_inside, InsideCircle c p)
  (h4 : ∀ p ∈ points_boundary, OnCircleBoundary c p)
  (h5 : ∀ (i j : Fin n), i ≠ j → 
    DoNotIntersect (points_inside[i], points_boundary[i]) (points_inside[j], points_boundary[j]))
  : ∀ (p q : Point), p ∈ points_inside → q ∈ points_inside → Reachable c points_inside points_boundary p q :=
sorry

end NUMINAMATH_CALUDE_grasshopper_jumps_l1165_116506


namespace NUMINAMATH_CALUDE_replacement_philosophy_in_lines_one_and_three_l1165_116509

/-- Represents a line of poetry -/
inductive PoeticLine
| EndlessFalling
| SpringRiver
| NewLeaves
| Waterfall

/-- Checks if a poetic line contains the philosophy of new things replacing old ones -/
def containsReplacementPhilosophy (line : PoeticLine) : Prop :=
  match line with
  | PoeticLine.EndlessFalling => True
  | PoeticLine.SpringRiver => False
  | PoeticLine.NewLeaves => True
  | PoeticLine.Waterfall => False

/-- The theorem stating that only lines ① and ③ contain the replacement philosophy -/
theorem replacement_philosophy_in_lines_one_and_three :
  (∀ line : PoeticLine, containsReplacementPhilosophy line ↔
    (line = PoeticLine.EndlessFalling ∨ line = PoeticLine.NewLeaves)) :=
by sorry

end NUMINAMATH_CALUDE_replacement_philosophy_in_lines_one_and_three_l1165_116509


namespace NUMINAMATH_CALUDE_sum_of_squares_of_reciprocals_l1165_116536

theorem sum_of_squares_of_reciprocals (x y : ℝ) 
  (sum_eq : x + y = 12) 
  (product_eq : x * y = 32) : 
  (1 / x)^2 + (1 / y)^2 = 5 / 64 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_of_reciprocals_l1165_116536


namespace NUMINAMATH_CALUDE_prime_squared_plus_two_prime_l1165_116535

theorem prime_squared_plus_two_prime (p : ℕ) : 
  Prime p → Prime (p^2 + 2) → p = 3 := by sorry

end NUMINAMATH_CALUDE_prime_squared_plus_two_prime_l1165_116535


namespace NUMINAMATH_CALUDE_two_numbers_with_110_divisors_and_nine_zeros_sum_l1165_116526

/-- A number ends with 9 zeros if it's divisible by 10^9 -/
def ends_with_nine_zeros (n : ℕ) : Prop := n % (10^9) = 0

/-- Count the number of divisors of a natural number -/
def count_divisors (n : ℕ) : ℕ := (Finset.filter (· ∣ n) (Finset.range (n + 1))).card

/-- The main theorem -/
theorem two_numbers_with_110_divisors_and_nine_zeros_sum :
  ∃ (a b : ℕ), a ≠ b ∧
                ends_with_nine_zeros a ∧
                ends_with_nine_zeros b ∧
                count_divisors a = 110 ∧
                count_divisors b = 110 ∧
                a + b = 7000000000 := by sorry

end NUMINAMATH_CALUDE_two_numbers_with_110_divisors_and_nine_zeros_sum_l1165_116526


namespace NUMINAMATH_CALUDE_complex_number_modulus_l1165_116593

theorem complex_number_modulus (z : ℂ) : z = (1 - Complex.I) / (1 + Complex.I) → Complex.abs z = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_modulus_l1165_116593


namespace NUMINAMATH_CALUDE_range_of_f_l1165_116523

-- Define the function
def f (x : ℝ) : ℝ := -x^2 + 2*x + 3

-- Define the domain
def domain : Set ℝ := {x | 0 ≤ x ∧ x ≤ 3}

-- State the theorem
theorem range_of_f :
  {y | ∃ x ∈ domain, f x = y} = {y | 0 ≤ y ∧ y ≤ 4} := by sorry

end NUMINAMATH_CALUDE_range_of_f_l1165_116523


namespace NUMINAMATH_CALUDE_bobs_raise_l1165_116542

/-- Calculates the raise per hour given the following conditions:
  * Bob works 40 hours per week
  * His housing benefit is reduced by $60 per month
  * He earns $5 more per week after the changes
-/
theorem bobs_raise (hours_per_week : ℕ) (benefit_reduction_per_month : ℚ) (extra_earnings_per_week : ℚ) :
  hours_per_week = 40 →
  benefit_reduction_per_month = 60 →
  extra_earnings_per_week = 5 →
  ∃ (raise_per_hour : ℚ), 
    raise_per_hour * hours_per_week - (benefit_reduction_per_month / 4) + extra_earnings_per_week = 0 ∧
    raise_per_hour = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_bobs_raise_l1165_116542


namespace NUMINAMATH_CALUDE_units_digit_of_27_times_46_l1165_116554

theorem units_digit_of_27_times_46 : (27 * 46) % 10 = 2 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_of_27_times_46_l1165_116554


namespace NUMINAMATH_CALUDE_symmetry_of_shifted_functions_l1165_116525

-- Define a function f from reals to reals
variable (f : ℝ → ℝ)

-- Define the property of symmetry about x = -1
def symmetric_about_neg_one (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x + 1) = y ↔ f (-x - 1) = y

-- State the theorem
theorem symmetry_of_shifted_functions :
  symmetric_about_neg_one f := by sorry

end NUMINAMATH_CALUDE_symmetry_of_shifted_functions_l1165_116525


namespace NUMINAMATH_CALUDE_quadratic_factorization_sum_l1165_116540

theorem quadratic_factorization_sum (a b c : ℤ) : 
  (∀ x, x^2 + 9*x + 18 = (x + a) * (x + b)) →
  (∀ x, x^2 + 19*x + 90 = (x + b) * (x + c)) →
  a + b + c = 22 := by
sorry

end NUMINAMATH_CALUDE_quadratic_factorization_sum_l1165_116540


namespace NUMINAMATH_CALUDE_constant_term_zero_implies_a_equals_six_l1165_116581

theorem constant_term_zero_implies_a_equals_six (a : ℝ) : 
  (∃ b c : ℝ, ∀ x : ℝ, (a + 2) * x^2 + b * x + (a - 6) = 0) → a = 6 := by
  sorry

end NUMINAMATH_CALUDE_constant_term_zero_implies_a_equals_six_l1165_116581


namespace NUMINAMATH_CALUDE_unique_x_with_rational_sums_l1165_116518

theorem unique_x_with_rational_sums (x : ℝ) :
  (∃ a : ℚ, x + Real.sqrt 3 = a) →
  (∃ b : ℚ, x^2 + Real.sqrt 3 = b) →
  x = 1/2 - Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_unique_x_with_rational_sums_l1165_116518


namespace NUMINAMATH_CALUDE_min_distance_point_to_circle_l1165_116511

/-- The minimum distance from a point to a circle -/
theorem min_distance_point_to_circle (x y : ℝ) :
  (x - 1)^2 + y^2 = 4 →  -- Circle equation
  ∃ (d : ℝ), d = Real.sqrt 18 - 2 ∧ 
  ∀ (px py : ℝ), (px + 2)^2 + (py + 3)^2 ≥ d^2 := by
  sorry

end NUMINAMATH_CALUDE_min_distance_point_to_circle_l1165_116511


namespace NUMINAMATH_CALUDE_polynomial_expansion_l1165_116549

theorem polynomial_expansion (x : ℝ) : 
  (5 * x^3 + 7) * (3 * x + 4) = 15 * x^4 + 20 * x^3 + 21 * x + 28 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_expansion_l1165_116549


namespace NUMINAMATH_CALUDE_multiple_of_seven_problem_l1165_116561

theorem multiple_of_seven_problem (start : Nat) (count : Nat) (result : Nat) : 
  start = 21 → count = 47 → result = 329 → 
  ∃ (n : Nat), result = start + 7 * (count - 1) ∧ result % 7 = 0 := by
  sorry

end NUMINAMATH_CALUDE_multiple_of_seven_problem_l1165_116561


namespace NUMINAMATH_CALUDE_complex_arg_range_l1165_116556

theorem complex_arg_range (z : ℂ) (h : Complex.abs (2 * z + 1 / z) = 1) :
  ∃ k : ℤ, k ∈ ({0, 1} : Set ℤ) ∧
    k * Real.pi + Real.pi / 2 - (1 / 2) * Real.arccos (3 / 4) ≤ Complex.arg z ∧
    Complex.arg z ≤ k * Real.pi + Real.pi / 2 + (1 / 2) * Real.arccos (3 / 4) :=
by sorry

end NUMINAMATH_CALUDE_complex_arg_range_l1165_116556


namespace NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_problem_4_l1165_116515

-- Problem 1
theorem problem_1 (a : ℝ) : -2 * a^3 * 3 * a^2 = -6 * a^5 := by sorry

-- Problem 2
theorem problem_2 (m : ℝ) (hm : m ≠ 0) : m^4 * (m^2)^3 / m^8 = m^2 := by sorry

-- Problem 3
theorem problem_3 (x : ℝ) : (-2*x - 1) * (2*x - 1) = 1 - 4*x^2 := by sorry

-- Problem 4
theorem problem_4 (x : ℝ) : (-3*x + 2)^2 = 9*x^2 - 12*x + 4 := by sorry

end NUMINAMATH_CALUDE_problem_1_problem_2_problem_3_problem_4_l1165_116515


namespace NUMINAMATH_CALUDE_prob_more_ones_than_fives_five_dice_l1165_116532

-- Define the number of dice
def num_dice : ℕ := 5

-- Define the number of sides on each die
def num_sides : ℕ := 6

-- Define a function to calculate the probability
def prob_more_ones_than_fives (n : ℕ) (s : ℕ) : ℚ :=
  190 / (s^n : ℚ)

-- Theorem statement
theorem prob_more_ones_than_fives_five_dice : 
  prob_more_ones_than_fives num_dice num_sides = 190 / 7776 := by
  sorry


end NUMINAMATH_CALUDE_prob_more_ones_than_fives_five_dice_l1165_116532


namespace NUMINAMATH_CALUDE_survey_results_l1165_116564

/-- Represents the survey results for a subject -/
structure SubjectSurvey where
  yes : Nat
  no : Nat
  unsure : Nat

/-- The main theorem about the survey results -/
theorem survey_results 
  (total_students : Nat)
  (subject_m : SubjectSurvey)
  (subject_r : SubjectSurvey)
  (yes_only_m : Nat)
  (h1 : total_students = 800)
  (h2 : subject_m.yes = 500)
  (h3 : subject_m.no = 200)
  (h4 : subject_m.unsure = 100)
  (h5 : subject_r.yes = 400)
  (h6 : subject_r.no = 100)
  (h7 : subject_r.unsure = 300)
  (h8 : yes_only_m = 150)
  (h9 : subject_m.yes + subject_m.no + subject_m.unsure = total_students)
  (h10 : subject_r.yes + subject_r.no + subject_r.unsure = total_students) :
  total_students - (subject_m.yes + subject_r.yes - yes_only_m) = 400 := by
  sorry

end NUMINAMATH_CALUDE_survey_results_l1165_116564


namespace NUMINAMATH_CALUDE_card_covers_at_least_twelve_squares_l1165_116580

/-- Represents a square card with a given side length -/
structure Card where
  side_length : ℝ

/-- Represents a checkerboard with squares of a given side length -/
structure Checkerboard where
  square_side_length : ℝ

/-- Calculates the maximum number of squares that can be covered by a card on a checkerboard -/
def max_squares_covered (card : Card) (board : Checkerboard) : ℕ :=
  sorry

/-- Theorem stating that a 1.5-inch square card can cover at least 12 one-inch squares on a checkerboard -/
theorem card_covers_at_least_twelve_squares :
  ∀ (card : Card) (board : Checkerboard),
    card.side_length = 1.5 ∧ board.square_side_length = 1 →
    max_squares_covered card board ≥ 12 :=
  sorry

end NUMINAMATH_CALUDE_card_covers_at_least_twelve_squares_l1165_116580


namespace NUMINAMATH_CALUDE_max_quadratic_solution_power_l1165_116583

/-- Given positive integers a, b, c that are powers of k, and r is the unique real solution
    to ax^2 - bx + c = 0 where r < 100, prove that the maximum possible value of r is 64 -/
theorem max_quadratic_solution_power (k a b c : ℕ+) (r : ℝ) :
  (∃ m n p : ℕ, a = k ^ m ∧ b = k ^ n ∧ c = k ^ p) →
  (∀ x : ℝ, a * x^2 - b * x + c = 0 ↔ x = r) →
  r < 100 →
  r ≤ 64 :=
sorry

end NUMINAMATH_CALUDE_max_quadratic_solution_power_l1165_116583


namespace NUMINAMATH_CALUDE_circle_inscribed_angles_sum_l1165_116537

theorem circle_inscribed_angles_sum (n : ℕ) (x y : ℝ) : 
  n = 18 →
  x = 3 * (360 / n) / 2 →
  y = 5 * (360 / n) / 2 →
  x + y = 80 := by
  sorry

end NUMINAMATH_CALUDE_circle_inscribed_angles_sum_l1165_116537


namespace NUMINAMATH_CALUDE_car_speed_problem_l1165_116505

/-- Proves that car R's speed is 50 mph given the problem conditions -/
theorem car_speed_problem (distance : ℝ) (time_diff : ℝ) (speed_diff : ℝ) :
  distance = 600 →
  time_diff = 2 →
  speed_diff = 10 →
  (distance / (distance / 50 - time_diff) = 50 + speed_diff) →
  50 = distance / (distance / 50) :=
by sorry

end NUMINAMATH_CALUDE_car_speed_problem_l1165_116505


namespace NUMINAMATH_CALUDE_school_track_length_l1165_116553

/-- Given that 200 steps correspond to 100 meters and 800 steps were walked along a track,
    the length of the track is 400 meters. -/
theorem school_track_length (steps_per_hundred_meters : ℕ) (track_steps : ℕ) : 
  steps_per_hundred_meters = 200 →
  track_steps = 800 →
  (100 : ℝ) / steps_per_hundred_meters * track_steps = 400 := by
  sorry

end NUMINAMATH_CALUDE_school_track_length_l1165_116553


namespace NUMINAMATH_CALUDE_problem_solution_l1165_116541

theorem problem_solution (w y z x : ℕ) 
  (hw : w = 50)
  (hz : z = 2 * w + 3)
  (hy : y = z + 5)
  (hx : x = 2 * y + 4) :
  x = 220 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l1165_116541


namespace NUMINAMATH_CALUDE_sum_of_roots_eq_eight_l1165_116557

theorem sum_of_roots_eq_eight : 
  let f : ℝ → ℝ := λ x => (x - 4)^2 - 16
  ∃ x₁ x₂ : ℝ, f x₁ = 0 ∧ f x₂ = 0 ∧ x₁ + x₂ = 8 :=
by
  sorry

end NUMINAMATH_CALUDE_sum_of_roots_eq_eight_l1165_116557


namespace NUMINAMATH_CALUDE_percentage_of_360_is_180_l1165_116572

theorem percentage_of_360_is_180 : 
  let whole : ℝ := 360
  let part : ℝ := 180
  let percentage : ℝ := (part / whole) * 100
  percentage = 50 := by sorry

end NUMINAMATH_CALUDE_percentage_of_360_is_180_l1165_116572


namespace NUMINAMATH_CALUDE_bottles_per_box_l1165_116538

theorem bottles_per_box 
  (num_boxes : ℕ) 
  (bottle_capacity : ℚ) 
  (fill_ratio : ℚ) 
  (total_water : ℚ) :
  num_boxes = 10 →
  bottle_capacity = 12 →
  fill_ratio = 3/4 →
  total_water = 4500 →
  (total_water / (bottle_capacity * fill_ratio)) / num_boxes = 50 :=
by sorry

end NUMINAMATH_CALUDE_bottles_per_box_l1165_116538


namespace NUMINAMATH_CALUDE_total_people_in_program_l1165_116596

theorem total_people_in_program (parents : Nat) (pupils : Nat) 
  (h1 : parents = 105) (h2 : pupils = 698) : 
  parents + pupils = 803 := by
  sorry

end NUMINAMATH_CALUDE_total_people_in_program_l1165_116596


namespace NUMINAMATH_CALUDE_percentage_calculation_l1165_116563

theorem percentage_calculation (first_number second_number : ℝ) 
  (h1 : first_number = 110)
  (h2 : second_number = 22) :
  (second_number / first_number) * 100 = 20 := by
  sorry

end NUMINAMATH_CALUDE_percentage_calculation_l1165_116563


namespace NUMINAMATH_CALUDE_money_distribution_theorem_l1165_116565

/-- Represents the money distribution problem --/
structure MoneyDistribution where
  total : ℚ
  first_share : ℚ
  second_share : ℚ
  third_share : ℚ

/-- Checks if the given distribution satisfies the initial conditions --/
def valid_initial_distribution (d : MoneyDistribution) : Prop :=
  d.first_share = d.total / 2 ∧
  d.second_share = d.total / 3 ∧
  d.third_share = d.total / 6

/-- Calculates the amount each person saves --/
def savings (d : MoneyDistribution) : (ℚ × ℚ × ℚ) :=
  (d.first_share / 2, d.second_share / 3, d.third_share / 6)

/-- Calculates the total amount saved --/
def total_savings (d : MoneyDistribution) : ℚ :=
  let (s1, s2, s3) := savings d
  s1 + s2 + s3

/-- Checks if the final distribution is equal for all three people --/
def equal_final_distribution (d : MoneyDistribution) : Prop :=
  let total_saved := total_savings d
  d.first_share + total_saved / 3 =
  d.second_share + total_saved / 3 ∧
  d.second_share + total_saved / 3 =
  d.third_share + total_saved / 3

/-- The main theorem stating the existence of a valid solution --/
theorem money_distribution_theorem :
  ∃ (d : MoneyDistribution),
    valid_initial_distribution d ∧
    equal_final_distribution d ∧
    d.first_share = 23.5 ∧
    d.second_share = 15 + 2/3 ∧
    d.third_share = 7 + 5/6 :=
sorry

end NUMINAMATH_CALUDE_money_distribution_theorem_l1165_116565


namespace NUMINAMATH_CALUDE_acute_angles_tangent_sum_l1165_116508

theorem acute_angles_tangent_sum (α β : Real) : 
  0 < α ∧ α < π/2 →
  0 < β ∧ β < π/2 →
  (1 + Real.tan α) * (1 + Real.tan β) = 2 →
  α + β = π/4 := by
sorry

end NUMINAMATH_CALUDE_acute_angles_tangent_sum_l1165_116508


namespace NUMINAMATH_CALUDE_sum_calculation_l1165_116514

theorem sum_calculation : 3 * 501 + 2 * 501 + 4 * 501 + 500 = 5009 := by
  sorry

end NUMINAMATH_CALUDE_sum_calculation_l1165_116514


namespace NUMINAMATH_CALUDE_circle_in_rectangle_l1165_116590

theorem circle_in_rectangle (rectangle_side : Real) (circle_area : Real) : 
  rectangle_side = 14 →
  circle_area = 153.93804002589985 →
  (circle_area = π * (rectangle_side / 2)^2) →
  rectangle_side = 14 :=
by
  sorry

end NUMINAMATH_CALUDE_circle_in_rectangle_l1165_116590


namespace NUMINAMATH_CALUDE_square_areas_equality_l1165_116587

theorem square_areas_equality (a : ℝ) :
  let M := a^2 + (a+3)^2 + (a+5)^2 + (a+6)^2
  let N := (a+1)^2 + (a+2)^2 + (a+4)^2 + (a+7)^2
  M = N := by sorry

end NUMINAMATH_CALUDE_square_areas_equality_l1165_116587


namespace NUMINAMATH_CALUDE_union_of_M_and_N_l1165_116543

def M : Set ℝ := {-1, 1, 2}
def N : Set ℝ := {x : ℝ | x^2 - 5*x + 4 = 0}

theorem union_of_M_and_N : M ∪ N = {-1, 1, 2, 4} := by
  sorry

end NUMINAMATH_CALUDE_union_of_M_and_N_l1165_116543


namespace NUMINAMATH_CALUDE_coefficient_x3_is_negative_five_l1165_116503

-- Define the binomial coefficient
def binomial (n k : ℕ) : ℕ := sorry

-- Define the coefficient of x^3 in the expansion of (x+1)(x-2)^3
def coefficient_x3 : ℤ :=
  (1 * (binomial 3 0)) + (-2 * (binomial 3 1))

-- Theorem statement
theorem coefficient_x3_is_negative_five :
  coefficient_x3 = -5 := by sorry

end NUMINAMATH_CALUDE_coefficient_x3_is_negative_five_l1165_116503


namespace NUMINAMATH_CALUDE_possible_to_end_with_80_19_specific_sequence_leads_to_80_19_l1165_116517

/-- Represents the result of a shot --/
inductive ShotResult
  | Success
  | Miss

/-- Applies the effect of a shot to the current amount --/
def applyShot (amount : ℝ) (result : ShotResult) : ℝ :=
  match result with
  | ShotResult.Success => amount * 1.1
  | ShotResult.Miss => amount * 0.9

/-- Theorem stating that it's possible to end up with 80.19 rubles --/
theorem possible_to_end_with_80_19 : ∃ (shots : List ShotResult), 
  shots.foldl applyShot 100 = 80.19 := by
  sorry

/-- Proof that the specific sequence of shots leads to 80.19 rubles --/
theorem specific_sequence_leads_to_80_19 : 
  [ShotResult.Miss, ShotResult.Miss, ShotResult.Miss, ShotResult.Success].foldl applyShot 100 = 80.19 := by
  sorry

end NUMINAMATH_CALUDE_possible_to_end_with_80_19_specific_sequence_leads_to_80_19_l1165_116517


namespace NUMINAMATH_CALUDE_distance_to_other_focus_l1165_116579

/-- Definition of the ellipse -/
def is_on_ellipse (x y : ℝ) : Prop :=
  x^2 / 16 + y^2 / 12 = 1

/-- Distance from a point to one focus is 3 -/
def distance_to_one_focus (x y : ℝ) : Prop :=
  ∃ (fx fy : ℝ), (x - fx)^2 + (y - fy)^2 = 3^2

/-- Theorem: If a point is on the ellipse and its distance to one focus is 3,
    then its distance to the other focus is 5 -/
theorem distance_to_other_focus
  (x y : ℝ)
  (h1 : is_on_ellipse x y)
  (h2 : distance_to_one_focus x y) :
  ∃ (gx gy : ℝ), (x - gx)^2 + (y - gy)^2 = 5^2 :=
sorry

end NUMINAMATH_CALUDE_distance_to_other_focus_l1165_116579


namespace NUMINAMATH_CALUDE_max_c_trees_l1165_116524

/-- Represents the types of scenic trees -/
inductive TreeType
| A
| B
| C

/-- The price of a tree given its type -/
def price (t : TreeType) : ℕ :=
  match t with
  | TreeType.A => 200
  | TreeType.B => 200
  | TreeType.C => 300

/-- The total budget for purchasing trees -/
def total_budget : ℕ := 220120

/-- The total number of trees to be purchased -/
def total_trees : ℕ := 1000

/-- Theorem stating the maximum number of C-type trees that can be purchased -/
theorem max_c_trees :
  (∃ (a b c : ℕ), a + b + c = total_trees ∧
                   a * price TreeType.A + b * price TreeType.B + c * price TreeType.C ≤ total_budget) →
  (∀ (a b c : ℕ), a + b + c = total_trees →
                   a * price TreeType.A + b * price TreeType.B + c * price TreeType.C ≤ total_budget →
                   c ≤ 201) ∧
  (∃ (a b : ℕ), a + b + 201 = total_trees ∧
                 a * price TreeType.A + b * price TreeType.B + 201 * price TreeType.C ≤ total_budget) :=
by sorry


end NUMINAMATH_CALUDE_max_c_trees_l1165_116524


namespace NUMINAMATH_CALUDE_robotics_club_enrollment_l1165_116500

theorem robotics_club_enrollment (total : ℕ) (cs : ℕ) (elec : ℕ) (both : ℕ)
  (h1 : total = 80)
  (h2 : cs = 50)
  (h3 : elec = 35)
  (h4 : both = 25) :
  total - (cs + elec - both) = 20 :=
by sorry

end NUMINAMATH_CALUDE_robotics_club_enrollment_l1165_116500


namespace NUMINAMATH_CALUDE_gcd_143_144_l1165_116533

theorem gcd_143_144 : Nat.gcd 143 144 = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_143_144_l1165_116533


namespace NUMINAMATH_CALUDE_sqrt_sum_equals_four_sqrt_three_l1165_116551

theorem sqrt_sum_equals_four_sqrt_three :
  Real.sqrt (16 - 8 * Real.sqrt 3) + Real.sqrt (16 + 8 * Real.sqrt 3) = 4 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_equals_four_sqrt_three_l1165_116551


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_sets_l1165_116598

theorem quadratic_inequality_solution_sets
  (a b c : ℝ)
  (h : Set.Ioo (-1/3 : ℝ) 2 = {x | a * x^2 + b * x + c > 0}) :
  {x : ℝ | c * x^2 + b * x + a < 0} = Set.Ioo (-3 : ℝ) (1/2) := by
    sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_sets_l1165_116598


namespace NUMINAMATH_CALUDE_andrey_gifts_l1165_116586

theorem andrey_gifts :
  ∃ (n a : ℕ), 
    n > 0 ∧ 
    a > 0 ∧ 
    n * (n - 2) = a * (n - 1) + 16 ∧ 
    n = 18 := by
  sorry

end NUMINAMATH_CALUDE_andrey_gifts_l1165_116586


namespace NUMINAMATH_CALUDE_study_time_for_average_score_l1165_116576

/-- Represents the relationship between study time and test score -/
structure StudyScoreRelation where
  studyTime : ℝ
  score : ℝ
  ratio : ℝ
  direct_relation : ratio = score / studyTime

/-- The problem setup and solution -/
theorem study_time_for_average_score
  (first_exam : StudyScoreRelation)
  (h_first_exam : first_exam.studyTime = 3 ∧ first_exam.score = 60)
  (target_average : ℝ)
  (h_target_average : target_average = 75)
  : ∃ (second_exam : StudyScoreRelation),
    second_exam.ratio = first_exam.ratio ∧
    (first_exam.score + second_exam.score) / 2 = target_average ∧
    second_exam.studyTime = 4.5 := by
  sorry

end NUMINAMATH_CALUDE_study_time_for_average_score_l1165_116576


namespace NUMINAMATH_CALUDE_exam_score_calculation_l1165_116594

/-- Given an examination with the following conditions:
  - Total number of questions is 120
  - Each correct answer scores 3 marks
  - Each wrong answer loses 1 mark
  - The total score is 180 marks
  This theorem proves that the number of correctly answered questions is 75. -/
theorem exam_score_calculation (total_questions : ℕ) (correct_score wrong_score total_score : ℤ) 
  (h1 : total_questions = 120)
  (h2 : correct_score = 3)
  (h3 : wrong_score = -1)
  (h4 : total_score = 180) :
  ∃ (correct_answers : ℕ), 
    correct_answers * correct_score + (total_questions - correct_answers) * wrong_score = total_score ∧ 
    correct_answers = 75 := by
  sorry

end NUMINAMATH_CALUDE_exam_score_calculation_l1165_116594


namespace NUMINAMATH_CALUDE_smallest_positive_multiple_of_45_l1165_116502

theorem smallest_positive_multiple_of_45 : 
  ∀ n : ℕ, n > 0 ∧ 45 ∣ n → n ≥ 45 :=
by
  sorry

end NUMINAMATH_CALUDE_smallest_positive_multiple_of_45_l1165_116502


namespace NUMINAMATH_CALUDE_physics_marks_l1165_116588

theorem physics_marks (P C M : ℝ) 
  (avg_all : (P + C + M) / 3 = 60)
  (avg_pm : (P + M) / 2 = 90)
  (avg_pc : (P + C) / 2 = 70) :
  P = 140 := by
sorry

end NUMINAMATH_CALUDE_physics_marks_l1165_116588


namespace NUMINAMATH_CALUDE_second_smallest_coprime_to_210_l1165_116567

def is_relatively_prime (a b : ℕ) : Prop := Nat.gcd a b = 1

theorem second_smallest_coprime_to_210 :
  ∃ (x : ℕ), x > 1 ∧ 
  is_relatively_prime x 210 ∧
  (∃ (y : ℕ), y > 1 ∧ y < x ∧ is_relatively_prime y 210) ∧
  (∀ (z : ℕ), z > 1 ∧ z < x ∧ is_relatively_prime z 210 → z = 11) ∧
  x = 13 := by
sorry

end NUMINAMATH_CALUDE_second_smallest_coprime_to_210_l1165_116567


namespace NUMINAMATH_CALUDE_candle_lighting_time_l1165_116546

/-- The time (in minutes) when the candles are lit before 5 PM -/
def lighting_time : ℝ := 218

/-- The length of time (in minutes) it takes for the first candle to burn out completely -/
def burn_time_1 : ℝ := 240

/-- The length of time (in minutes) it takes for the second candle to burn out completely -/
def burn_time_2 : ℝ := 300

/-- The ratio of the length of the longer stub to the shorter stub at 5 PM -/
def stub_ratio : ℝ := 3

theorem candle_lighting_time :
  (burn_time_2 - lighting_time) / burn_time_2 = stub_ratio * ((burn_time_1 - lighting_time) / burn_time_1) :=
sorry

end NUMINAMATH_CALUDE_candle_lighting_time_l1165_116546


namespace NUMINAMATH_CALUDE_pens_purchased_l1165_116510

theorem pens_purchased (total_cost : ℝ) (num_pencils : ℕ) (pencil_price : ℝ) (pen_price : ℝ)
  (h1 : total_cost = 690)
  (h2 : num_pencils = 75)
  (h3 : pencil_price = 2)
  (h4 : pen_price = 18) :
  (total_cost - num_pencils * pencil_price) / pen_price = 30 := by
  sorry

end NUMINAMATH_CALUDE_pens_purchased_l1165_116510


namespace NUMINAMATH_CALUDE_jellybeans_count_l1165_116585

/-- The number of jellybeans in a bag with black, green, and orange beans -/
def total_jellybeans (black green orange : ℕ) : ℕ := black + green + orange

/-- Theorem: Given the conditions, the total number of jellybeans is 27 -/
theorem jellybeans_count :
  ∀ (black green orange : ℕ),
  black = 8 →
  green = black + 2 →
  orange = green - 1 →
  total_jellybeans black green orange = 27 := by
sorry

end NUMINAMATH_CALUDE_jellybeans_count_l1165_116585


namespace NUMINAMATH_CALUDE_count_integers_with_repeated_digits_is_156_l1165_116527

/-- A function that counts the number of positive three-digit integers less than 700 
    with at least two digits that are the same. -/
def count_integers_with_repeated_digits : ℕ :=
  sorry

/-- The theorem stating that the count of integers with the given properties is 156. -/
theorem count_integers_with_repeated_digits_is_156 : 
  count_integers_with_repeated_digits = 156 := by
  sorry

end NUMINAMATH_CALUDE_count_integers_with_repeated_digits_is_156_l1165_116527


namespace NUMINAMATH_CALUDE_two_books_different_genres_count_l1165_116574

/-- Represents the number of books in each genre -/
def booksPerGenre : ℕ := 3

/-- Represents the number of genres -/
def numberOfGenres : ℕ := 4

/-- Represents the number of genres to choose -/
def genresToChoose : ℕ := 2

/-- Calculates the number of ways to choose two books of different genres -/
def chooseTwoBooksOfDifferentGenres : ℕ :=
  Nat.choose numberOfGenres genresToChoose * booksPerGenre * booksPerGenre

theorem two_books_different_genres_count :
  chooseTwoBooksOfDifferentGenres = 54 := by
  sorry

end NUMINAMATH_CALUDE_two_books_different_genres_count_l1165_116574


namespace NUMINAMATH_CALUDE_proportion_inequality_l1165_116558

theorem proportion_inequality (a b c : ℝ) (h : a / b = b / c) : a^2 + c^2 ≥ 2 * b^2 := by
  sorry

end NUMINAMATH_CALUDE_proportion_inequality_l1165_116558


namespace NUMINAMATH_CALUDE_women_work_hours_l1165_116521

/-- Given work completed by men and women under specific conditions, prove that women worked 6 hours per day. -/
theorem women_work_hours (men : ℕ) (women : ℕ) (men_days : ℕ) (women_days : ℕ) (men_hours : ℕ) 
  (h_men : men = 15)
  (h_women : women = 21)
  (h_men_days : men_days = 21)
  (h_women_days : women_days = 30)
  (h_men_hours : men_hours = 8)
  (h_work_rate : (3 : ℚ) / women = (2 : ℚ) / men) :
  ∃ women_hours : ℚ, women_hours = 6 ∧ 
    (men * men_days * men_hours : ℚ) = (women * women_days * women_hours) :=
sorry

end NUMINAMATH_CALUDE_women_work_hours_l1165_116521


namespace NUMINAMATH_CALUDE_quadratic_function_range_l1165_116545

/-- The function f(x) = x^2 + mx - 1 -/
def f (m : ℝ) (x : ℝ) : ℝ := x^2 + m*x - 1

theorem quadratic_function_range (m : ℝ) :
  (∀ x ∈ Set.Icc m (m + 1), f m x < 0) ↔ m ∈ Set.Ioo (- Real.sqrt 2 / 2) 0 :=
sorry

end NUMINAMATH_CALUDE_quadratic_function_range_l1165_116545


namespace NUMINAMATH_CALUDE_contrapositive_theorem_l1165_116582

theorem contrapositive_theorem (a b c : ℝ) :
  (abc = 0 → a = 0 ∨ b = 0 ∨ c = 0) ↔ (a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 → abc ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_contrapositive_theorem_l1165_116582


namespace NUMINAMATH_CALUDE_cubic_factorization_l1165_116512

theorem cubic_factorization (a : ℝ) : a^3 - 25*a = a*(a+5)*(a-5) := by sorry

end NUMINAMATH_CALUDE_cubic_factorization_l1165_116512


namespace NUMINAMATH_CALUDE_polar_to_rectangular_conversion_l1165_116589

theorem polar_to_rectangular_conversion :
  let r : ℝ := 4
  let θ : ℝ := π / 3
  let x : ℝ := r * Real.cos θ
  let y : ℝ := r * Real.sin θ
  (x, y) = (2, 2 * Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_polar_to_rectangular_conversion_l1165_116589


namespace NUMINAMATH_CALUDE_solve_lindas_savings_l1165_116552

def lindas_savings_problem (savings : ℚ) : Prop :=
  let furniture_fraction : ℚ := 3 / 5
  let tv_fraction : ℚ := 1 - furniture_fraction
  let tv_cost : ℚ := 400
  tv_fraction * savings = tv_cost ∧ savings = 1000

theorem solve_lindas_savings : ∃ (savings : ℚ), lindas_savings_problem savings :=
sorry

end NUMINAMATH_CALUDE_solve_lindas_savings_l1165_116552


namespace NUMINAMATH_CALUDE_square_sum_roots_l1165_116592

theorem square_sum_roots (a b c : ℝ) (h : a ≠ 0) : 
  let roots_sum := -b / a
  (x^2 + b*x + c = 0) → roots_sum^2 = 36 :=
by
  sorry

end NUMINAMATH_CALUDE_square_sum_roots_l1165_116592


namespace NUMINAMATH_CALUDE_consecutive_negative_integers_sum_l1165_116595

theorem consecutive_negative_integers_sum (n : ℤ) : 
  n < 0 ∧ n * (n + 1) = 3080 → n + (n + 1) = -111 := by
  sorry

end NUMINAMATH_CALUDE_consecutive_negative_integers_sum_l1165_116595


namespace NUMINAMATH_CALUDE_sum_remainder_l1165_116507

theorem sum_remainder (a b c d e : ℕ) 
  (ha : a % 13 = 3)
  (hb : b % 13 = 5)
  (hc : c % 13 = 7)
  (hd : d % 13 = 9)
  (he : e % 13 = 11)
  (hsquare : ∃ k : ℕ, c = k * k) :
  (a + b + c + d + e) % 13 = 9 := by
  sorry

end NUMINAMATH_CALUDE_sum_remainder_l1165_116507


namespace NUMINAMATH_CALUDE_lcm_24_30_40_l1165_116584

theorem lcm_24_30_40 : Nat.lcm (Nat.lcm 24 30) 40 = 120 := by sorry

end NUMINAMATH_CALUDE_lcm_24_30_40_l1165_116584


namespace NUMINAMATH_CALUDE_ratio_evaluation_l1165_116530

theorem ratio_evaluation : (2^2005 * 3^2003) / 6^2004 = 2/3 := by sorry

end NUMINAMATH_CALUDE_ratio_evaluation_l1165_116530


namespace NUMINAMATH_CALUDE_root_in_interval_implies_m_range_l1165_116516

-- Define the function f(x) = x^3 - 3x
def f (x : ℝ) : ℝ := x^3 - 3*x

-- Define the theorem
theorem root_in_interval_implies_m_range :
  ∀ m : ℝ, (∃ x : ℝ, x ∈ Set.Icc 0 2 ∧ f x + m = 0) → m ∈ Set.Icc (-2) 2 := by
  sorry

end NUMINAMATH_CALUDE_root_in_interval_implies_m_range_l1165_116516


namespace NUMINAMATH_CALUDE_park_area_l1165_116568

/-- The area of a rectangular park with perimeter 120 feet and length three times the width is 675 square feet. -/
theorem park_area (length width : ℝ) : 
  (2 * length + 2 * width = 120) →
  (length = 3 * width) →
  (length * width = 675) :=
by
  sorry

end NUMINAMATH_CALUDE_park_area_l1165_116568


namespace NUMINAMATH_CALUDE_complex_sum_modulus_l1165_116578

theorem complex_sum_modulus : 
  Complex.abs (1/5 - (2/5)*Complex.I) + Complex.abs (3/5 + (4/5)*Complex.I) = (1 + Real.sqrt 5) / Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_complex_sum_modulus_l1165_116578


namespace NUMINAMATH_CALUDE_individuals_from_c_is_twenty_l1165_116570

/-- Represents the ratio of individuals in strata A, B, and C -/
structure StrataRatio :=
  (a : ℕ)
  (b : ℕ)
  (c : ℕ)

/-- Calculates the number of individuals to be drawn from stratum C -/
def individualsFromC (ratio : StrataRatio) (sampleSize : ℕ) : ℕ :=
  (ratio.c * sampleSize) / (ratio.a + ratio.b + ratio.c)

/-- Theorem: Given the specified ratio and sample size, 20 individuals should be drawn from C -/
theorem individuals_from_c_is_twenty :
  let ratio := StrataRatio.mk 5 3 2
  let sampleSize := 100
  individualsFromC ratio sampleSize = 20 := by
  sorry

end NUMINAMATH_CALUDE_individuals_from_c_is_twenty_l1165_116570


namespace NUMINAMATH_CALUDE_incenter_position_l1165_116599

-- Define a triangle PQR
structure Triangle where
  P : ℝ × ℝ
  Q : ℝ × ℝ
  R : ℝ × ℝ

-- Define the side lengths
def side_lengths (t : Triangle) : ℝ × ℝ × ℝ :=
  (11, 5, 8)

-- Define the incenter of a triangle
def incenter (t : Triangle) : ℝ × ℝ := sorry

-- Theorem stating the position of the incenter
theorem incenter_position (t : Triangle) :
  let (p, q, r) := side_lengths t
  let J := incenter t
  J = (11/24 * t.P.1 + 5/24 * t.Q.1 + 8/24 * t.R.1,
       11/24 * t.P.2 + 5/24 * t.Q.2 + 8/24 * t.R.2) :=
by sorry

end NUMINAMATH_CALUDE_incenter_position_l1165_116599


namespace NUMINAMATH_CALUDE_set_intersection_problem_l1165_116571

theorem set_intersection_problem (M N : Set ℤ) : 
  M = {-1, 0, 1} → N = {0, 1, 2} → M ∩ N = {0, 1} := by
  sorry

end NUMINAMATH_CALUDE_set_intersection_problem_l1165_116571


namespace NUMINAMATH_CALUDE_roger_tray_trips_l1165_116528

/-- Calculates the number of trips needed to carry a given number of trays -/
def trips_needed (trays_per_trip : ℕ) (total_trays : ℕ) : ℕ :=
  (total_trays + trays_per_trip - 1) / trays_per_trip

/-- Proves that 3 trips are needed to carry 12 trays when 4 trays can be carried per trip -/
theorem roger_tray_trips : trips_needed 4 12 = 3 := by
  sorry

end NUMINAMATH_CALUDE_roger_tray_trips_l1165_116528


namespace NUMINAMATH_CALUDE_first_term_of_special_arithmetic_sequence_l1165_116548

def arithmetic_sequence (a : ℕ → ℝ) := ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)

theorem first_term_of_special_arithmetic_sequence (a : ℕ → ℝ) :
  arithmetic_sequence a →
  (∀ n, a (n + 1) > a n) →
  a 1 + a 2 + a 3 = 12 →
  a 1 * a 2 * a 3 = 48 →
  a 1 = 2 := by
sorry

end NUMINAMATH_CALUDE_first_term_of_special_arithmetic_sequence_l1165_116548


namespace NUMINAMATH_CALUDE_ceiling_product_equation_solution_l1165_116539

theorem ceiling_product_equation_solution :
  ∃! (x : ℝ), ⌈x⌉ * x = 225 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_ceiling_product_equation_solution_l1165_116539


namespace NUMINAMATH_CALUDE_solve_for_t_l1165_116547

theorem solve_for_t (s t : ℚ) (eq1 : 8 * s + 6 * t = 160) (eq2 : s = t + 3) : t = 68 / 7 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_t_l1165_116547
