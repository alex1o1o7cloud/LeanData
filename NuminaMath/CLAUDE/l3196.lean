import Mathlib

namespace NUMINAMATH_CALUDE_tripled_base_doubled_exponent_l3196_319614

theorem tripled_base_doubled_exponent 
  (a b x : ℝ) 
  (hb : b ≠ 0) 
  (hr : (3 * a) ^ (2 * b) = a ^ b * x ^ b) : 
  x = 9 * a := by sorry

end NUMINAMATH_CALUDE_tripled_base_doubled_exponent_l3196_319614


namespace NUMINAMATH_CALUDE_f_properties_l3196_319650

-- Define the function f(x)
def f (x : ℝ) : ℝ := 3 * x^3 - 9 * x + 5

-- Define the interval
def interval : Set ℝ := Set.Icc (-3) 3

-- State the theorem
theorem f_properties :
  -- Monotonicity properties
  (∀ x y, x < y ∧ x < -1 → f x < f y) ∧
  (∀ x y, x < y ∧ 1 < x → f x < f y) ∧
  (∀ x y, -1 < x ∧ x < y ∧ y < 1 → f x > f y) ∧
  -- Maximum and minimum values
  (∃ x ∈ interval, ∀ y ∈ interval, f y ≤ f x ∧ f x = 59) ∧
  (∃ x ∈ interval, ∀ y ∈ interval, f y ≥ f x ∧ f x = -49) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l3196_319650


namespace NUMINAMATH_CALUDE_smallest_integer_above_sqrt3_plus_sqrt2_to_8th_l3196_319602

theorem smallest_integer_above_sqrt3_plus_sqrt2_to_8th (x : ℝ) : 
  x = (Real.sqrt 3 + Real.sqrt 2)^8 → 
  ∀ n : ℤ, (n : ℝ) > x → n ≥ 5360 ∧ 5360 > x :=
by sorry

end NUMINAMATH_CALUDE_smallest_integer_above_sqrt3_plus_sqrt2_to_8th_l3196_319602


namespace NUMINAMATH_CALUDE_cylinder_volume_change_l3196_319640

/-- Theorem: Tripling the radius and doubling the height of a cylinder increases its volume by a factor of 18. -/
theorem cylinder_volume_change (r h V : ℝ) (hV : V = π * r^2 * h) :
  π * (3*r)^2 * (2*h) = 18 * V := by sorry

end NUMINAMATH_CALUDE_cylinder_volume_change_l3196_319640


namespace NUMINAMATH_CALUDE_playground_fence_posts_l3196_319607

/-- Calculates the minimum number of fence posts needed for a rectangular playground. -/
def min_fence_posts (length width post_spacing : ℕ) : ℕ :=
  let long_side := max length width
  let short_side := min length width
  let long_side_posts := long_side / post_spacing + 1
  let short_side_posts := short_side / post_spacing + 1
  long_side_posts + 2 * (short_side_posts - 1)

/-- Theorem stating the minimum number of fence posts needed for the given playground. -/
theorem playground_fence_posts :
  min_fence_posts 100 50 10 = 21 :=
by
  sorry

#eval min_fence_posts 100 50 10

end NUMINAMATH_CALUDE_playground_fence_posts_l3196_319607


namespace NUMINAMATH_CALUDE_intersection_distance_sum_l3196_319670

noncomputable section

-- Define the curve C
def curve_C (x y : ℝ) : Prop := x^2 / 4 + y^2 = 1

-- Define the line l
def line_l (x y : ℝ) : Prop := x - y + 1 = 0

-- Define point P
def point_P : ℝ × ℝ := (0, 1)

-- Define the distance function
def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

theorem intersection_distance_sum :
  ∃ (A B : ℝ × ℝ),
    curve_C A.1 A.2 ∧ curve_C B.1 B.2 ∧
    line_l A.1 A.2 ∧ line_l B.1 B.2 ∧
    A ≠ B ∧
    distance point_P A + distance point_P B = 8 * Real.sqrt 2 / 5 :=
sorry

end

end NUMINAMATH_CALUDE_intersection_distance_sum_l3196_319670


namespace NUMINAMATH_CALUDE_rectangle_area_l3196_319613

theorem rectangle_area (d : ℝ) (w : ℝ) (h : w > 0) : 
  d^2 = 10 * w^2 → 3 * w^2 = 3 * d^2 / 10 :=
by
  sorry

#check rectangle_area

end NUMINAMATH_CALUDE_rectangle_area_l3196_319613


namespace NUMINAMATH_CALUDE_raft_sticks_total_l3196_319621

/-- The number of sticks needed for Simon's raft -/
def simon_sticks : ℕ := 36

/-- The number of sticks needed for Gerry's raft -/
def gerry_sticks : ℕ := (2 * simon_sticks) / 3

/-- The number of sticks needed for Micky's raft -/
def micky_sticks : ℕ := simon_sticks + gerry_sticks + 9

/-- The total number of sticks needed for all three rafts -/
def total_sticks : ℕ := simon_sticks + gerry_sticks + micky_sticks

theorem raft_sticks_total : total_sticks = 129 := by
  sorry

end NUMINAMATH_CALUDE_raft_sticks_total_l3196_319621


namespace NUMINAMATH_CALUDE_opposite_numbers_proof_l3196_319697

theorem opposite_numbers_proof : 
  (-(5^2) = -((5^2))) ∧ ((5^2) = (-5)^2) → 
  (-(5^2) = -(((-5)^2))) ∧ (-(5^2) ≠ (-5)^2) := by
sorry

end NUMINAMATH_CALUDE_opposite_numbers_proof_l3196_319697


namespace NUMINAMATH_CALUDE_mark_distance_before_turning_l3196_319699

/-- Proves that Mark walked 7.5 miles before turning around -/
theorem mark_distance_before_turning (chris_speed : ℝ) (school_distance : ℝ) 
  (mark_extra_time : ℝ) (h1 : chris_speed = 3) (h2 : school_distance = 9) 
  (h3 : mark_extra_time = 2) : 
  let chris_time := school_distance / chris_speed
  let mark_time := chris_time + mark_extra_time
  let mark_total_distance := chris_speed * mark_time
  mark_total_distance / 2 = 7.5 := by
  sorry

end NUMINAMATH_CALUDE_mark_distance_before_turning_l3196_319699


namespace NUMINAMATH_CALUDE_min_value_of_reciprocal_sum_l3196_319665

theorem min_value_of_reciprocal_sum (t q a b : ℝ) : 
  (∀ x, x^2 - t*x + q = 0 ↔ x = a ∨ x = b) →
  a + b = a^2 + b^2 →
  a + b = a^3 + b^3 →
  a + b = a^4 + b^4 →
  ∃ (min : ℝ), min = 128 * Real.sqrt 3 / 45 ∧ 
    ∀ (t' q' a' b' : ℝ), 
      (∀ x, x^2 - t'*x + q' = 0 ↔ x = a' ∨ x = b') →
      a' + b' = a'^2 + b'^2 →
      a' + b' = a'^3 + b'^3 →
      a' + b' = a'^4 + b'^4 →
      1/a'^5 + 1/b'^5 ≥ min :=
sorry

end NUMINAMATH_CALUDE_min_value_of_reciprocal_sum_l3196_319665


namespace NUMINAMATH_CALUDE_cricket_team_age_difference_l3196_319619

/-- Represents a cricket team with its age-related properties -/
structure CricketTeam where
  total_members : ℕ
  team_average_age : ℝ
  wicket_keeper_age_difference : ℝ
  remaining_players_average_age : ℝ

/-- Theorem stating the difference between the team's average age and the remaining players' average age -/
theorem cricket_team_age_difference (team : CricketTeam)
  (h1 : team.total_members = 11)
  (h2 : team.team_average_age = 28)
  (h3 : team.wicket_keeper_age_difference = 3)
  (h4 : team.remaining_players_average_age = 25) :
  team.team_average_age - team.remaining_players_average_age = 3 := by
  sorry

end NUMINAMATH_CALUDE_cricket_team_age_difference_l3196_319619


namespace NUMINAMATH_CALUDE_tangent_line_sum_l3196_319666

/-- Given a function f: ℝ → ℝ with a tangent line at x = 2 
    described by the equation 2x + y - 3 = 0,
    prove that f(2) + f'(2) = -3 -/
theorem tangent_line_sum (f : ℝ → ℝ) (hf : Differentiable ℝ f) 
  (h_tangent : ∀ x y, y = f x → 2 * x + y - 3 = 0 → x = 2) :
  f 2 + deriv f 2 = -3 := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_sum_l3196_319666


namespace NUMINAMATH_CALUDE_line_difference_l3196_319657

theorem line_difference (line_length : ℝ) (h : line_length = 80) :
  (0.75 - 0.4) * line_length = 28 :=
by sorry

end NUMINAMATH_CALUDE_line_difference_l3196_319657


namespace NUMINAMATH_CALUDE_fifteen_tomorrow_l3196_319683

/-- Represents the fishing schedule in a coastal village -/
structure FishingSchedule where
  daily : ℕ           -- Number of people fishing daily
  everyOther : ℕ      -- Number of people fishing every other day
  everyThree : ℕ      -- Number of people fishing every three days
  yesterday : ℕ       -- Number of people who fished yesterday
  today : ℕ           -- Number of people fishing today

/-- Calculates the number of people who will fish tomorrow given a fishing schedule -/
def tomorrowsFishers (schedule : FishingSchedule) : ℕ :=
  sorry

/-- Theorem stating that given the specific fishing schedule, 15 people will fish tomorrow -/
theorem fifteen_tomorrow (schedule : FishingSchedule) 
  (h1 : schedule.daily = 7)
  (h2 : schedule.everyOther = 8)
  (h3 : schedule.everyThree = 3)
  (h4 : schedule.yesterday = 12)
  (h5 : schedule.today = 10) :
  tomorrowsFishers schedule = 15 := by
  sorry

end NUMINAMATH_CALUDE_fifteen_tomorrow_l3196_319683


namespace NUMINAMATH_CALUDE_parabola_hyperbola_disjunction_l3196_319656

-- Define the propositions
def p : Prop := ∀ y : ℝ, (∃ x : ℝ, x = 4 * y^2) → (∃ x : ℝ, x = 1)

def q : Prop := ∃ x y : ℝ, (x^2 / 4 - y^2 / 5 = -1) ∧ (x = 0 ∧ y = 3)

-- Theorem to prove
theorem parabola_hyperbola_disjunction : p ∨ q := by sorry

end NUMINAMATH_CALUDE_parabola_hyperbola_disjunction_l3196_319656


namespace NUMINAMATH_CALUDE_mirabel_candy_distribution_l3196_319671

theorem mirabel_candy_distribution :
  ∃ (k : ℕ), k = 2 ∧ 
  (∀ (j : ℕ), j < k → ¬∃ (n : ℕ), 10 ≤ n ∧ n < 20 ∧ (47 - j) % n = 0) ∧
  (∃ (n : ℕ), 10 ≤ n ∧ n < 20 ∧ (47 - k) % n = 0) :=
by sorry

end NUMINAMATH_CALUDE_mirabel_candy_distribution_l3196_319671


namespace NUMINAMATH_CALUDE_building_windows_l3196_319695

/-- The number of windows already installed -/
def installed_windows : ℕ := 6

/-- The time it takes to install one window (in hours) -/
def hours_per_window : ℕ := 5

/-- The time it will take to install the remaining windows (in hours) -/
def remaining_hours : ℕ := 20

/-- The total number of windows needed for the building -/
def total_windows : ℕ := installed_windows + remaining_hours / hours_per_window

theorem building_windows : total_windows = 10 := by
  sorry

end NUMINAMATH_CALUDE_building_windows_l3196_319695


namespace NUMINAMATH_CALUDE_sample_correlation_coefficient_range_l3196_319620

/-- The sample correlation coefficient -/
def sample_correlation_coefficient (X Y : List ℝ) : ℝ := sorry

/-- Theorem: The sample correlation coefficient is in the closed interval [-1, 1] -/
theorem sample_correlation_coefficient_range 
  (X Y : List ℝ) : 
  ∃ r : ℝ, sample_correlation_coefficient X Y = r ∧ r ∈ Set.Icc (-1 : ℝ) 1 :=
sorry

end NUMINAMATH_CALUDE_sample_correlation_coefficient_range_l3196_319620


namespace NUMINAMATH_CALUDE_sum_of_squares_l3196_319646

theorem sum_of_squares (x y : ℝ) (h1 : x + y = 12) (h2 : x * y = 50) : x^2 + y^2 = 44 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_l3196_319646


namespace NUMINAMATH_CALUDE_polygon_sides_count_l3196_319675

theorem polygon_sides_count (n : ℕ) (k : ℕ) (r : ℚ) : 
  k = n * (n - 3) / 2 →
  k = r * n →
  r = 3 / 2 →
  n = 6 := by
sorry

end NUMINAMATH_CALUDE_polygon_sides_count_l3196_319675


namespace NUMINAMATH_CALUDE_worker_count_proof_l3196_319652

/-- The number of workers who raised money by equal contribution -/
def number_of_workers : ℕ := 1200

/-- The total contribution in rupees -/
def total_contribution : ℕ := 300000

/-- The increased total contribution if each worker contributed 50 rupees extra -/
def increased_contribution : ℕ := 360000

/-- The extra amount each worker would contribute in the increased scenario -/
def extra_contribution : ℕ := 50

theorem worker_count_proof :
  (number_of_workers * (total_contribution / number_of_workers) = total_contribution) ∧
  (number_of_workers * (total_contribution / number_of_workers + extra_contribution) = increased_contribution) :=
sorry

end NUMINAMATH_CALUDE_worker_count_proof_l3196_319652


namespace NUMINAMATH_CALUDE_max_q_minus_r_for_1057_l3196_319647

theorem max_q_minus_r_for_1057 :
  ∃ (q r : ℕ+), 1057 = 23 * q + r ∧ 
  ∀ (q' r' : ℕ+), 1057 = 23 * q' + r' → q' - r' ≤ q - r ∧ q - r = 23 := by
sorry

end NUMINAMATH_CALUDE_max_q_minus_r_for_1057_l3196_319647


namespace NUMINAMATH_CALUDE_direct_proportion_through_point_l3196_319658

/-- A direct proportion function passing through (2, -1) -/
def f (x : ℝ) : ℝ := sorry

/-- The point (2, -1) lies on the graph of f -/
axiom point_on_graph : f 2 = -1

/-- f is a direct proportion function -/
axiom direct_proportion (x : ℝ) : ∃ k : ℝ, f x = k * x

theorem direct_proportion_through_point :
  ∀ x : ℝ, f x = -1/2 * x := by sorry

end NUMINAMATH_CALUDE_direct_proportion_through_point_l3196_319658


namespace NUMINAMATH_CALUDE_cell_phone_bill_is_45_l3196_319610

/-- Calculates the total cell phone bill based on given parameters --/
def calculate_bill (fixed_charge : ℚ) (daytime_rate : ℚ) (evening_rate : ℚ) 
                   (free_evening_minutes : ℕ) (daytime_minutes : ℕ) (evening_minutes : ℕ) : ℚ :=
  let daytime_cost := daytime_rate * daytime_minutes
  let chargeable_evening_minutes := max (evening_minutes - free_evening_minutes) 0
  let evening_cost := evening_rate * chargeable_evening_minutes
  fixed_charge + daytime_cost + evening_cost

/-- Theorem stating that the cell phone bill is $45 given the specified conditions --/
theorem cell_phone_bill_is_45 :
  calculate_bill 20 0.1 0.05 200 200 300 = 45 := by
  sorry


end NUMINAMATH_CALUDE_cell_phone_bill_is_45_l3196_319610


namespace NUMINAMATH_CALUDE_intersection_complement_equality_l3196_319631

def U : Set Int := Set.univ

def A : Set Int := {-1, 1, 3, 5, 7, 9}

def B : Set Int := {-1, 5, 7}

theorem intersection_complement_equality :
  A ∩ (U \ B) = {1, 3, 9} := by sorry

end NUMINAMATH_CALUDE_intersection_complement_equality_l3196_319631


namespace NUMINAMATH_CALUDE_intersection_M_N_l3196_319684

-- Define the sets M and N
def M : Set ℝ := Set.univ
def N : Set ℝ := {x | 2 * x - x^2 > 0}

-- State the theorem
theorem intersection_M_N : M ∩ N = Set.Ioo 0 2 := by
  sorry

end NUMINAMATH_CALUDE_intersection_M_N_l3196_319684


namespace NUMINAMATH_CALUDE_bob_distance_from_start_l3196_319678

/-- Regular hexagon with side length 3 km -/
structure RegularHexagon where
  side_length : ℝ
  is_regular : side_length = 3

/-- Position after walking along the perimeter -/
def position_after_walk (h : RegularHexagon) (distance : ℝ) : ℝ × ℝ :=
  sorry

/-- Distance between two points -/
def distance_between_points (p1 p2 : ℝ × ℝ) : ℝ :=
  sorry

theorem bob_distance_from_start (h : RegularHexagon) :
  let start_point := (0, 0)
  let end_point := position_after_walk h 7
  distance_between_points start_point end_point = 2 := by
  sorry

end NUMINAMATH_CALUDE_bob_distance_from_start_l3196_319678


namespace NUMINAMATH_CALUDE_checkerboard_coverage_three_by_five_uncoverable_l3196_319642

/-- Represents a checkerboard -/
structure Checkerboard where
  rows : ℕ
  cols : ℕ

/-- A domino covers exactly two squares -/
def domino_size : ℕ := 2

/-- The total number of squares on a checkerboard -/
def total_squares (board : Checkerboard) : ℕ :=
  board.rows * board.cols

/-- A checkerboard can be covered by dominoes if its total squares is even -/
def can_be_covered_by_dominoes (board : Checkerboard) : Prop :=
  total_squares board % domino_size = 0

/-- Theorem: A checkerboard can be covered by dominoes iff its total squares is even -/
theorem checkerboard_coverage (board : Checkerboard) :
  can_be_covered_by_dominoes board ↔ Even (total_squares board) := by sorry

/-- The 3x5 checkerboard cannot be covered by dominoes -/
theorem three_by_five_uncoverable :
  ¬ can_be_covered_by_dominoes ⟨3, 5⟩ := by sorry

end NUMINAMATH_CALUDE_checkerboard_coverage_three_by_five_uncoverable_l3196_319642


namespace NUMINAMATH_CALUDE_factor_expression_l3196_319682

theorem factor_expression (x : ℝ) : x * (x + 3) + 2 * (x + 3) = (x + 2) * (x + 3) := by
  sorry

end NUMINAMATH_CALUDE_factor_expression_l3196_319682


namespace NUMINAMATH_CALUDE_change_received_l3196_319686

def skirt_price : ℕ := 13
def skirt_count : ℕ := 2
def blouse_price : ℕ := 6
def blouse_count : ℕ := 3
def amount_paid : ℕ := 100

def total_cost : ℕ := skirt_price * skirt_count + blouse_price * blouse_count

theorem change_received : amount_paid - total_cost = 56 := by
  sorry

end NUMINAMATH_CALUDE_change_received_l3196_319686


namespace NUMINAMATH_CALUDE_flowmaster_pump_l3196_319672

/-- The FlowMaster pump problem -/
theorem flowmaster_pump (pump_rate : ℝ) (time : ℝ) (h1 : pump_rate = 600) (h2 : time = 0.5) :
  pump_rate * time = 300 := by
  sorry

end NUMINAMATH_CALUDE_flowmaster_pump_l3196_319672


namespace NUMINAMATH_CALUDE_polynomial_roots_imply_composite_sum_of_squares_l3196_319615

/-- A polynomial with integer coefficients -/
def IntPolynomial (p q : ℤ) : ℝ → ℝ := fun x ↦ x^2 + p*x + q + 1

/-- Definition of a composite number -/
def IsComposite (n : ℕ) : Prop := ∃ a b : ℕ, a > 1 ∧ b > 1 ∧ a * b = n

theorem polynomial_roots_imply_composite_sum_of_squares (p q : ℤ) :
  (∃ a b : ℕ, a > 0 ∧ b > 0 ∧ (IntPolynomial p q a = 0) ∧ (IntPolynomial p q b = 0)) →
  IsComposite (Int.natAbs (p^2 + q^2)) :=
sorry

end NUMINAMATH_CALUDE_polynomial_roots_imply_composite_sum_of_squares_l3196_319615


namespace NUMINAMATH_CALUDE_janeth_round_balloon_bags_l3196_319606

/-- The number of balloons in each bag of round balloons -/
def round_balloons_per_bag : ℕ := 20

/-- The number of bags of long balloons bought -/
def long_balloon_bags : ℕ := 4

/-- The number of balloons in each bag of long balloons -/
def long_balloons_per_bag : ℕ := 30

/-- The number of round balloons that burst -/
def burst_balloons : ℕ := 5

/-- The total number of balloons left -/
def total_balloons_left : ℕ := 215

/-- Theorem stating that Janeth bought 5 bags of round balloons -/
theorem janeth_round_balloon_bags : ℕ := by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_janeth_round_balloon_bags_l3196_319606


namespace NUMINAMATH_CALUDE_solve_equation_l3196_319653

theorem solve_equation (b : ℚ) (h : b + b/4 = 10/4) : b = 2 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l3196_319653


namespace NUMINAMATH_CALUDE_mask_digit_correct_l3196_319674

/-- Represents the four masks in the problem -/
inductive Mask
| elephant
| mouse
| pig
| panda

/-- Associates each mask with a digit -/
def mask_digit : Mask → Nat
| Mask.elephant => 6
| Mask.mouse => 4
| Mask.pig => 8
| Mask.panda => 1

/-- The theorem to be proved -/
theorem mask_digit_correct :
  (mask_digit Mask.elephant) * (mask_digit Mask.elephant) = 36 ∧
  (mask_digit Mask.mouse) * (mask_digit Mask.mouse) = 16 ∧
  (mask_digit Mask.pig) * (mask_digit Mask.pig) = 64 ∧
  (mask_digit Mask.panda) * (mask_digit Mask.panda) = 1 ∧
  (∀ m1 m2 : Mask, m1 ≠ m2 → mask_digit m1 ≠ mask_digit m2) :=
by sorry

#check mask_digit_correct

end NUMINAMATH_CALUDE_mask_digit_correct_l3196_319674


namespace NUMINAMATH_CALUDE_twice_a_plus_one_nonnegative_l3196_319676

theorem twice_a_plus_one_nonnegative (a : ℝ) : (2 * a + 1 ≥ 0) ↔ (∀ x : ℝ, x = 2 * a + 1 → x ≥ 0) := by sorry

end NUMINAMATH_CALUDE_twice_a_plus_one_nonnegative_l3196_319676


namespace NUMINAMATH_CALUDE_cyclic_sum_inequality_l3196_319632

theorem cyclic_sum_inequality (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  x / Real.sqrt (y + z) + y / Real.sqrt (z + x) + z / Real.sqrt (x + y) ≥ 
  Real.sqrt (3 / 2) * Real.sqrt (x + y + z) := by
  sorry

end NUMINAMATH_CALUDE_cyclic_sum_inequality_l3196_319632


namespace NUMINAMATH_CALUDE_four_integer_solutions_l3196_319634

def satisfies_equation (a : ℤ) : Prop :=
  |2 * a + 7| + |2 * a - 1| = 8

theorem four_integer_solutions :
  ∃ (S : Finset ℤ), (∀ a ∈ S, satisfies_equation a) ∧ 
                    (∀ a : ℤ, satisfies_equation a → a ∈ S) ∧
                    Finset.card S = 4 :=
sorry

end NUMINAMATH_CALUDE_four_integer_solutions_l3196_319634


namespace NUMINAMATH_CALUDE_friend_fruit_consumption_l3196_319600

/-- Given three friends who ate a total of 128 ounces of fruit, 
    where one friend ate 8 ounces and another ate 96 ounces,
    prove that the third friend ate 24 ounces. -/
theorem friend_fruit_consumption 
  (total : ℕ) 
  (friend1 : ℕ) 
  (friend2 : ℕ) 
  (h1 : total = 128)
  (h2 : friend1 = 8)
  (h3 : friend2 = 96) :
  total - friend1 - friend2 = 24 :=
by sorry

end NUMINAMATH_CALUDE_friend_fruit_consumption_l3196_319600


namespace NUMINAMATH_CALUDE_angle_between_perpendicular_lines_to_dihedral_angle_l3196_319635

-- Define the dihedral angle
def dihedral_angle (α l β : Plane) : ℝ := sorry

-- Define perpendicularity between a line and a plane
def perpendicular (m : Line) (α : Plane) : Prop := sorry

-- Define the angle between two lines
def angle_between_lines (m n : Line) : ℝ := sorry

-- Define skew lines
def skew_lines (m n : Line) : Prop := sorry

theorem angle_between_perpendicular_lines_to_dihedral_angle 
  (α l β : Plane) (m n : Line) :
  dihedral_angle α l β = 60 ∧ 
  skew_lines m n ∧
  perpendicular m α ∧
  perpendicular n β →
  angle_between_lines m n = 60 := by sorry

end NUMINAMATH_CALUDE_angle_between_perpendicular_lines_to_dihedral_angle_l3196_319635


namespace NUMINAMATH_CALUDE_min_reciprocal_sum_l3196_319661

theorem min_reciprocal_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : Real.log (x + y) = 0) :
  (1 / x + 1 / y) ≥ 4 ∧ ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ Real.log (a + b) = 0 ∧ 1 / a + 1 / b = 4 := by
  sorry

end NUMINAMATH_CALUDE_min_reciprocal_sum_l3196_319661


namespace NUMINAMATH_CALUDE_largest_multiple_six_negation_greater_than_neg_150_l3196_319605

theorem largest_multiple_six_negation_greater_than_neg_150 :
  (∀ n : ℤ, n % 6 = 0 ∧ -n > -150 → n ≤ 144) ∧
  144 % 6 = 0 ∧ -144 > -150 :=
by sorry

end NUMINAMATH_CALUDE_largest_multiple_six_negation_greater_than_neg_150_l3196_319605


namespace NUMINAMATH_CALUDE_certain_number_is_fourteen_l3196_319617

/-- A certain number multiplied by d is the square of an integer -/
def is_square_multiple (n : ℕ) (d : ℕ) : Prop :=
  ∃ m : ℕ, n * d = m^2

/-- d is the smallest positive integer satisfying the condition -/
def is_smallest_d (n : ℕ) (d : ℕ) : Prop :=
  is_square_multiple n d ∧ ∀ k < d, ¬(is_square_multiple n k)

theorem certain_number_is_fourteen (d : ℕ) (h1 : d = 14) 
  (h2 : ∃ n : ℕ, is_smallest_d n d) : 
  ∃ n : ℕ, is_smallest_d n d ∧ n = 14 :=
sorry

end NUMINAMATH_CALUDE_certain_number_is_fourteen_l3196_319617


namespace NUMINAMATH_CALUDE_second_train_length_correct_l3196_319664

/-- Calculates the length of a train given the length of another train, their speeds, and the time they take to cross each other when moving in opposite directions. -/
def calculate_train_length (length_train1 : ℝ) (speed_train1 : ℝ) (speed_train2 : ℝ) (time_to_cross : ℝ) : ℝ :=
  let relative_speed := speed_train1 + speed_train2
  let total_distance := relative_speed * time_to_cross
  total_distance - length_train1

/-- Theorem stating that the calculated length of the second train is correct given the problem conditions. -/
theorem second_train_length_correct (length_train1 : ℝ) (speed_train1 : ℝ) (speed_train2 : ℝ) (time_to_cross : ℝ)
  (h1 : length_train1 = 110)
  (h2 : speed_train1 = 60 * 1000 / 3600)
  (h3 : speed_train2 = 40 * 1000 / 3600)
  (h4 : time_to_cross = 9.719222462203025) :
  let length_train2 := calculate_train_length length_train1 speed_train1 speed_train2 time_to_cross
  ∃ ε > 0, |length_train2 - 159.98| < ε :=
sorry

end NUMINAMATH_CALUDE_second_train_length_correct_l3196_319664


namespace NUMINAMATH_CALUDE_difference_of_squares_601_597_l3196_319696

theorem difference_of_squares_601_597 : 601^2 - 597^2 = 4792 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_601_597_l3196_319696


namespace NUMINAMATH_CALUDE_planes_parallel_from_skew_lines_l3196_319690

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the relations
variable (subset : Line → Plane → Prop)
variable (parallel : Line → Plane → Prop)
variable (plane_parallel : Plane → Plane → Prop)
variable (skew : Line → Line → Prop)

-- State the theorem
theorem planes_parallel_from_skew_lines 
  (α β : Plane) (l m : Line) :
  skew l m →
  subset l α →
  parallel l β →
  subset m β →
  parallel m α →
  plane_parallel α β :=
sorry

end NUMINAMATH_CALUDE_planes_parallel_from_skew_lines_l3196_319690


namespace NUMINAMATH_CALUDE_f_lower_bound_f_condition_equivalent_l3196_319636

-- Define the function f
def f (x a : ℝ) : ℝ := |x - a^2| + |x + 2*a + 3|

-- Theorem 1: For all real x and a, f(x) ≥ 2
theorem f_lower_bound (x a : ℝ) : f x a ≥ 2 := by sorry

-- Theorem 2: f(-3/2) < 3 is equivalent to -1 < a < 0
theorem f_condition_equivalent (a : ℝ) : 
  (f (-3/2) a < 3) ↔ (-1 < a ∧ a < 0) := by sorry

end NUMINAMATH_CALUDE_f_lower_bound_f_condition_equivalent_l3196_319636


namespace NUMINAMATH_CALUDE_cone_to_cylinder_volume_ratio_l3196_319637

/-- 
Given a cylinder and a cone with the same radius, where the cone's height is one-third of the cylinder's height,
prove that the ratio of the cone's volume to the cylinder's volume is 1/9.
-/
theorem cone_to_cylinder_volume_ratio (r h : ℝ) (hr : r > 0) (hh : h > 0) : 
  (1 / 3 * π * r^2 * (h / 3)) / (π * r^2 * h) = 1 / 9 := by
  sorry

end NUMINAMATH_CALUDE_cone_to_cylinder_volume_ratio_l3196_319637


namespace NUMINAMATH_CALUDE_octagon_area_theorem_l3196_319604

/-- The area of a regular octagon inscribed in a square with perimeter 144 cm,
    where each side of the square is trisected by the vertices of the octagon. -/
def inscribedOctagonArea : ℝ := 1008

/-- The perimeter of the square. -/
def squarePerimeter : ℝ := 144

/-- A side of the square is trisected by the vertices of the octagon. -/
def isTrisected (s : ℝ) : Prop := ∃ p : ℝ, s = 3 * p

theorem octagon_area_theorem (s : ℝ) (h1 : s * 4 = squarePerimeter) (h2 : isTrisected s) :
  inscribedOctagonArea = s^2 - 4 * (s/3)^2 :=
sorry

end NUMINAMATH_CALUDE_octagon_area_theorem_l3196_319604


namespace NUMINAMATH_CALUDE_smallest_sum_of_reciprocals_l3196_319651

theorem smallest_sum_of_reciprocals (x y : ℕ+) (h1 : x ≠ y) (h2 : (1 : ℚ) / x + (1 : ℚ) / y = (1 : ℚ) / 15) :
  ∃ (a b : ℕ+), a ≠ b ∧ (1 : ℚ) / a + (1 : ℚ) / b = (1 : ℚ) / 15 ∧ a + b ≤ x + y ∧ a + b = 64 :=
sorry

end NUMINAMATH_CALUDE_smallest_sum_of_reciprocals_l3196_319651


namespace NUMINAMATH_CALUDE_intersection_implies_sum_of_translations_l3196_319677

/-- Given two functions f and g that intersect at points (1,7) and (9,1),
    prove that the sum of their x-axis translation parameters is 10 -/
theorem intersection_implies_sum_of_translations (a b c d : ℝ) :
  (∀ x, -2 * |x - a| + b = 2 * |x - c| + d ↔ (x = 1 ∧ -2 * |x - a| + b = 7) ∨ (x = 9 ∧ -2 * |x - a| + b = 1)) →
  a + c = 10 := by
sorry

end NUMINAMATH_CALUDE_intersection_implies_sum_of_translations_l3196_319677


namespace NUMINAMATH_CALUDE_ascetics_equal_distance_l3196_319649

theorem ascetics_equal_distance (h m : ℝ) (h_pos : h > 0) (m_pos : m > 0) :
  ∃ x : ℝ, x > 0 ∧ 
  (x + (((x + h)^2 + (m * h)^2).sqrt) = h + m * h) ∧
  x = (h * m) / (m + 2) := by
sorry

end NUMINAMATH_CALUDE_ascetics_equal_distance_l3196_319649


namespace NUMINAMATH_CALUDE_gumball_probability_l3196_319601

theorem gumball_probability (blue_prob : ℚ) : 
  blue_prob ^ 2 = 16 / 49 → (1 : ℚ) - blue_prob = 3 / 7 := by
  sorry

end NUMINAMATH_CALUDE_gumball_probability_l3196_319601


namespace NUMINAMATH_CALUDE_grocer_banana_purchase_l3196_319645

/-- Calculates the number of pounds of bananas purchased by a grocer given the purchase price, selling price, and total profit. -/
theorem grocer_banana_purchase
  (purchase_price : ℚ)
  (purchase_quantity : ℚ)
  (selling_price : ℚ)
  (selling_quantity : ℚ)
  (total_profit : ℚ)
  (h1 : purchase_price / purchase_quantity = 0.50 / 3)
  (h2 : selling_price / selling_quantity = 1.00 / 4)
  (h3 : total_profit = 9.00) :
  ∃ (pounds : ℚ), pounds = 108 ∧ 
    pounds * (selling_price / selling_quantity - purchase_price / purchase_quantity) = total_profit :=
by sorry

end NUMINAMATH_CALUDE_grocer_banana_purchase_l3196_319645


namespace NUMINAMATH_CALUDE_circle_square_area_difference_l3196_319626

/-- The difference between the area of a circle with diameter 10 inches
    and the area of a square with diagonal 10 inches is approximately 28.5 square inches. -/
theorem circle_square_area_difference :
  let square_diagonal : ℝ := 10
  let circle_diameter : ℝ := 10
  let square_area : ℝ := (square_diagonal ^ 2) / 2
  let circle_area : ℝ := π * ((circle_diameter / 2) ^ 2)
  ∃ ε > 0, ε < 0.1 ∧ |circle_area - square_area - 28.5| < ε :=
by
  sorry


end NUMINAMATH_CALUDE_circle_square_area_difference_l3196_319626


namespace NUMINAMATH_CALUDE_solution_set_abs_x_times_one_minus_two_x_l3196_319660

theorem solution_set_abs_x_times_one_minus_two_x (x : ℝ) :
  (|x| * (1 - 2*x) > 0) ↔ (x < 0 ∨ (x > 0 ∧ x < 1/2)) := by sorry

end NUMINAMATH_CALUDE_solution_set_abs_x_times_one_minus_two_x_l3196_319660


namespace NUMINAMATH_CALUDE_geometric_series_r_value_l3196_319654

theorem geometric_series_r_value (a r : ℝ) (h1 : a ≠ 0) (h2 : |r| < 1) :
  (a / (1 - r) = 20) →
  (a * r / (1 - r^2) = 8) →
  r = 2/3 := by
sorry

end NUMINAMATH_CALUDE_geometric_series_r_value_l3196_319654


namespace NUMINAMATH_CALUDE_lists_with_high_number_l3196_319648

def total_balls : ℕ := 15
def draws : ℕ := 4
def threshold : ℕ := 10

theorem lists_with_high_number (total_balls draws threshold : ℕ) :
  total_balls = 15 ∧ draws = 4 ∧ threshold = 10 →
  (total_balls ^ draws) - (threshold ^ draws) = 40625 := by
  sorry

end NUMINAMATH_CALUDE_lists_with_high_number_l3196_319648


namespace NUMINAMATH_CALUDE_sum_of_digits_of_a_l3196_319618

def a : ℕ := (10^10) - 47

/-- The sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

theorem sum_of_digits_of_a : sum_of_digits a = 81 := by sorry

end NUMINAMATH_CALUDE_sum_of_digits_of_a_l3196_319618


namespace NUMINAMATH_CALUDE_geometric_sequence_problem_l3196_319668

/-- A geometric sequence with common ratio greater than 1 -/
structure GeometricSequence where
  a : ℕ → ℝ
  q : ℝ
  h_positive : q > 1
  h_geometric : ∀ n : ℕ, a (n + 1) = q * a n

/-- The theorem to be proved -/
theorem geometric_sequence_problem (seq : GeometricSequence)
    (h1 : seq.a 3 * seq.a 7 = 72)
    (h2 : seq.a 2 + seq.a 8 = 27) :
  seq.a 12 = 96 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_problem_l3196_319668


namespace NUMINAMATH_CALUDE_vacuum_time_difference_l3196_319639

/-- Given vacuuming times, proves the difference between upstairs time and twice downstairs time -/
theorem vacuum_time_difference (total_time upstairs_time downstairs_time : ℕ) 
  (h1 : total_time = 38)
  (h2 : upstairs_time = 27)
  (h3 : total_time = upstairs_time + downstairs_time)
  (h4 : upstairs_time > 2 * downstairs_time) :
  upstairs_time - 2 * downstairs_time = 5 := by
  sorry


end NUMINAMATH_CALUDE_vacuum_time_difference_l3196_319639


namespace NUMINAMATH_CALUDE_parallel_vectors_m_value_l3196_319692

/-- Given two vectors a and b in ℝ², prove that if they are parallel and
    a = (1, 2) and b = (1-m, 2m-4), then m = 3/2 -/
theorem parallel_vectors_m_value (m : ℝ) :
  let a : Fin 2 → ℝ := ![1, 2]
  let b : Fin 2 → ℝ := ![1 - m, 2 * m - 4]
  (∃ (k : ℝ), k ≠ 0 ∧ b = k • a) →
  m = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_m_value_l3196_319692


namespace NUMINAMATH_CALUDE_tank_fill_time_with_leak_l3196_319663

def pump_rate : ℚ := 1 / 6
def leak_rate : ℚ := 1 / 12

theorem tank_fill_time_with_leak :
  let net_fill_rate := pump_rate - leak_rate
  (1 : ℚ) / net_fill_rate = 12 := by sorry

end NUMINAMATH_CALUDE_tank_fill_time_with_leak_l3196_319663


namespace NUMINAMATH_CALUDE_function_composition_equality_l3196_319655

theorem function_composition_equality (m n p q c : ℝ) :
  let f := fun (x : ℝ) => m * x + n + c
  let g := fun (x : ℝ) => p * x + q + c
  (∀ x, f (g x) = g (f x)) ↔ n * (1 - p) - q * (1 - m) + c * (m - p) = 0 := by
  sorry

end NUMINAMATH_CALUDE_function_composition_equality_l3196_319655


namespace NUMINAMATH_CALUDE_trailing_zeroes_500_factorial_l3196_319624

/-- The number of trailing zeroes in n! -/
def trailingZeroes (n : ℕ) : ℕ :=
  (n / 5) + (n / 25) + (n / 125)

/-- Theorem: The number of trailing zeroes in 500! is 124 -/
theorem trailing_zeroes_500_factorial :
  trailingZeroes 500 = 124 := by
  sorry

end NUMINAMATH_CALUDE_trailing_zeroes_500_factorial_l3196_319624


namespace NUMINAMATH_CALUDE_percentage_difference_l3196_319623

theorem percentage_difference : (0.8 * 40) - ((4 / 5) * 15) = 20 := by
  sorry

end NUMINAMATH_CALUDE_percentage_difference_l3196_319623


namespace NUMINAMATH_CALUDE_system_solution_l3196_319644

theorem system_solution : ∃ (x y : ℝ), 
  (x = 4 + 2 * Real.sqrt 3 ∧ y = 12 + 6 * Real.sqrt 3) ∧
  (1 - 12 / (3 * x + y) = 2 / Real.sqrt x) ∧
  (1 + 12 / (3 * x + y) = 6 / Real.sqrt y) := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l3196_319644


namespace NUMINAMATH_CALUDE_inequality_solution_l3196_319622

-- Define the inequality
def inequality (a : ℝ) (x : ℝ) : Prop := (a * x) / (x - 1) < 1

-- Define the solution set
def solution_set (a : ℝ) : Set ℝ := {x | x < 1 ∨ x > 3}

-- Theorem statement
theorem inequality_solution (a : ℝ) :
  (∀ x, x ∈ solution_set a ↔ inequality a x) → a = 2/3 := by
  sorry


end NUMINAMATH_CALUDE_inequality_solution_l3196_319622


namespace NUMINAMATH_CALUDE_work_completion_men_count_l3196_319643

/-- Given a work that can be completed by M men in 20 days, 
    or by (M - 4) men in 25 days, prove that M = 16. -/
theorem work_completion_men_count :
  ∀ (M : ℕ) (W : ℝ),
  (M : ℝ) * (W / 20) = (M - 4 : ℝ) * (W / 25) →
  M = 16 :=
by sorry

end NUMINAMATH_CALUDE_work_completion_men_count_l3196_319643


namespace NUMINAMATH_CALUDE_workshop_output_comparison_l3196_319679

/-- Represents the monthly increase factor for a workshop -/
structure WorkshopGrowth where
  fixed_amount : ℝ
  percentage : ℝ

/-- Theorem statement for workshop output comparison -/
theorem workshop_output_comparison 
  (growth_A growth_B : WorkshopGrowth)
  (h_initial_equal : growth_A.fixed_amount = growth_B.percentage) -- Initial outputs are equal
  (h_equal_after_7 : 1 + 6 * growth_A.fixed_amount = (1 + growth_B.percentage) ^ 6) -- Equal after 7 months
  : 1 + 3 * growth_A.fixed_amount > (1 + growth_B.percentage) ^ 3 := by
  sorry

end NUMINAMATH_CALUDE_workshop_output_comparison_l3196_319679


namespace NUMINAMATH_CALUDE_f_even_and_increasing_l3196_319693

def f (x : ℝ) := x^2 + 1

theorem f_even_and_increasing : 
  (∀ x : ℝ, f x = f (-x)) ∧ 
  (∀ x y : ℝ, 0 < x → x < y → f x < f y) :=
sorry

end NUMINAMATH_CALUDE_f_even_and_increasing_l3196_319693


namespace NUMINAMATH_CALUDE_blocks_left_in_second_tower_is_two_l3196_319629

/-- The number of blocks left standing in the second tower --/
def blocks_left_in_second_tower (first_stack_height : ℕ) 
                                (second_stack_diff : ℕ) 
                                (third_stack_diff : ℕ) 
                                (blocks_left_in_third : ℕ) 
                                (total_fallen : ℕ) : ℕ :=
  let second_stack_height := first_stack_height + second_stack_diff
  let third_stack_height := second_stack_height + third_stack_diff
  let total_blocks := first_stack_height + second_stack_height + third_stack_height
  let fallen_from_first := first_stack_height
  let fallen_from_third := third_stack_height - blocks_left_in_third
  let fallen_from_second := total_fallen - fallen_from_first - fallen_from_third
  second_stack_height - fallen_from_second

theorem blocks_left_in_second_tower_is_two :
  blocks_left_in_second_tower 7 5 7 3 33 = 2 := by
  sorry

end NUMINAMATH_CALUDE_blocks_left_in_second_tower_is_two_l3196_319629


namespace NUMINAMATH_CALUDE_binomial_18_choose_7_l3196_319659

theorem binomial_18_choose_7 : Nat.choose 18 7 = 31824 := by
  sorry

end NUMINAMATH_CALUDE_binomial_18_choose_7_l3196_319659


namespace NUMINAMATH_CALUDE_seashells_to_find_l3196_319638

def current_seashells : ℕ := 19
def target_seashells : ℕ := 25

theorem seashells_to_find : target_seashells - current_seashells = 6 := by
  sorry

end NUMINAMATH_CALUDE_seashells_to_find_l3196_319638


namespace NUMINAMATH_CALUDE_inequality_proof_l3196_319627

theorem inequality_proof (x : ℝ) (hx : x > 0) : x^2 + 1/(4*x) ≥ 3/4 ∧ Real.sqrt 3 - 1 < 3/4 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3196_319627


namespace NUMINAMATH_CALUDE_max_points_for_top_teams_l3196_319616

/-- Represents a football tournament with the given rules --/
structure FootballTournament where
  num_teams : ℕ
  num_top_teams : ℕ
  points_for_win : ℕ
  points_for_draw : ℕ

/-- The maximum possible points that can be achieved by the top teams --/
def max_points (t : FootballTournament) : ℕ :=
  let internal_games := t.num_top_teams.choose 2
  let external_games := t.num_top_teams * (t.num_teams - t.num_top_teams)
  internal_games * t.points_for_win + external_games * t.points_for_win

/-- The theorem stating the maximum integer N for which at least 6 teams can score N or more points --/
theorem max_points_for_top_teams (t : FootballTournament) 
  (h1 : t.num_teams = 15)
  (h2 : t.num_top_teams = 6)
  (h3 : t.points_for_win = 3)
  (h4 : t.points_for_draw = 1) :
  ∃ (N : ℕ), N = 34 ∧ 
  (∀ (M : ℕ), (M : ℝ) * t.num_top_teams ≤ max_points t → M ≤ N) ∧
  (N : ℝ) * t.num_top_teams ≤ max_points t :=
by sorry

end NUMINAMATH_CALUDE_max_points_for_top_teams_l3196_319616


namespace NUMINAMATH_CALUDE_det_A_eq_22_l3196_319608

def A : Matrix (Fin 2) (Fin 2) ℤ := !![7, -5; -4, 6]

theorem det_A_eq_22 : A.det = 22 := by
  sorry

end NUMINAMATH_CALUDE_det_A_eq_22_l3196_319608


namespace NUMINAMATH_CALUDE_max_value_trig_expression_l3196_319673

theorem max_value_trig_expression :
  ∀ θ φ : ℝ, 0 ≤ θ ∧ θ ≤ π/2 → 0 ≤ φ ∧ φ ≤ π/2 →
  3 * Real.sin θ * Real.cos φ + 2 * (Real.sin φ)^2 ≤ 5 :=
by sorry

end NUMINAMATH_CALUDE_max_value_trig_expression_l3196_319673


namespace NUMINAMATH_CALUDE_blue_whale_tongue_weight_l3196_319611

-- Define the weight of one ton in pounds
def ton_in_pounds : ℕ := 2000

-- Define the weight of a blue whale's tongue in tons
def blue_whale_tongue_tons : ℕ := 3

-- Theorem: The weight of a blue whale's tongue in pounds
theorem blue_whale_tongue_weight :
  blue_whale_tongue_tons * ton_in_pounds = 6000 := by
  sorry

end NUMINAMATH_CALUDE_blue_whale_tongue_weight_l3196_319611


namespace NUMINAMATH_CALUDE_dawn_at_6am_l3196_319680

/-- Represents the time of dawn in hours before noon -/
def dawn_time : ℝ := 6

/-- Represents the time (in hours after noon) when the first pedestrian arrives at B -/
def arrival_time_B : ℝ := 4

/-- Represents the time (in hours after noon) when the second pedestrian arrives at A -/
def arrival_time_A : ℝ := 9

/-- The theorem states that given the conditions of the problem, dawn occurred at 6 AM -/
theorem dawn_at_6am :
  dawn_time * arrival_time_B = arrival_time_A * dawn_time ∧
  dawn_time + arrival_time_B + dawn_time + arrival_time_A = 24 →
  dawn_time = 6 := by
  sorry

end NUMINAMATH_CALUDE_dawn_at_6am_l3196_319680


namespace NUMINAMATH_CALUDE_negation_of_implication_l3196_319681

theorem negation_of_implication (x : ℝ) :
  ¬(x > 2 → x > 1) ↔ (x ≤ 2 → x ≤ 1) := by sorry

end NUMINAMATH_CALUDE_negation_of_implication_l3196_319681


namespace NUMINAMATH_CALUDE_cube_displacement_l3196_319603

/-- The volume of water displaced by a cube in a cylindrical barrel -/
theorem cube_displacement (cube_side : ℝ) (barrel_radius barrel_height : ℝ) 
  (h_cube_side : cube_side = 6)
  (h_barrel_radius : barrel_radius = 5)
  (h_barrel_height : barrel_height = 12)
  (h_fully_submerged : cube_side ≤ barrel_height) :
  cube_side ^ 3 = 216 := by
  sorry

end NUMINAMATH_CALUDE_cube_displacement_l3196_319603


namespace NUMINAMATH_CALUDE_lottery_theorem_l3196_319669

-- Define the lottery parameters
def total_numbers : ℕ := 90
def numbers_drawn : ℕ := 5
def numbers_played : ℕ := 7
def group_size : ℕ := 10

-- Define the ticket prices and payouts
def ticket_cost : ℕ := 60
def payout_three_match : ℕ := 7000
def payout_two_match : ℕ := 300

-- Define the probability of drawing 3 out of 7 specific numbers
def probability_three_match : ℚ := 119105 / 43949268

-- Define the profit per person
def profit_per_person : ℕ := 4434

-- Theorem statement
theorem lottery_theorem :
  (probability_three_match = 119105 / 43949268) ∧
  (profit_per_person = 4434) := by
  sorry


end NUMINAMATH_CALUDE_lottery_theorem_l3196_319669


namespace NUMINAMATH_CALUDE_sum_of_last_two_digits_of_series_l3196_319698

def fibonacci_factorial_series := [2, 3, 5, 8, 13, 21]

def last_two_digits (n : ℕ) : ℕ := n % 100

def factorial (n : ℕ) : ℕ := if n = 0 then 1 else n * factorial (n - 1)

theorem sum_of_last_two_digits_of_series : 
  (fibonacci_factorial_series.map (λ x => last_two_digits (factorial x))).sum = 8 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_last_two_digits_of_series_l3196_319698


namespace NUMINAMATH_CALUDE_opposite_of_negative_two_l3196_319687

theorem opposite_of_negative_two : -((-2 : ℤ)) = 2 := by sorry

end NUMINAMATH_CALUDE_opposite_of_negative_two_l3196_319687


namespace NUMINAMATH_CALUDE_star_problem_l3196_319612

def star (x y : ℕ) : ℕ := x^2 + y

theorem star_problem : (3^(star 5 7)) ^ 2 + 4^(star 4 6) = 3^64 + 4^22 := by
  sorry

end NUMINAMATH_CALUDE_star_problem_l3196_319612


namespace NUMINAMATH_CALUDE_M_inequality_l3196_319662

/-- The number of h-subsets with property P_k(X) in a set X of size n -/
def M (n k h : ℕ) : ℕ := sorry

/-- Theorem stating the inequality for M(n,k,h) -/
theorem M_inequality (n k h : ℕ) :
  (n.choose h) / (k.choose h) ≤ M n k h ∧ M n k h ≤ (n - k + h).choose h :=
sorry

end NUMINAMATH_CALUDE_M_inequality_l3196_319662


namespace NUMINAMATH_CALUDE_complex_equation_sum_l3196_319641

theorem complex_equation_sum (a b : ℝ) : 
  (a - Complex.I = 2 + b * Complex.I) → (a + b = 1) := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_sum_l3196_319641


namespace NUMINAMATH_CALUDE_unique_zero_implies_a_gt_one_l3196_319633

/-- A function f(x) = 2ax^2 - x - 1 has only one zero in the interval (0, 1) -/
def has_unique_zero_in_interval (a : ℝ) : Prop :=
  ∃! x : ℝ, 0 < x ∧ x < 1 ∧ 2 * a * x^2 - x - 1 = 0

/-- If f(x) = 2ax^2 - x - 1 has only one zero in the interval (0, 1), then a > 1 -/
theorem unique_zero_implies_a_gt_one :
  ∀ a : ℝ, has_unique_zero_in_interval a → a > 1 :=
by sorry

end NUMINAMATH_CALUDE_unique_zero_implies_a_gt_one_l3196_319633


namespace NUMINAMATH_CALUDE_order_of_values_l3196_319688

theorem order_of_values : 
  let a := Real.sin (60 * π / 180)
  let b := Real.sqrt (5 / 9)
  let c := π / 2014
  a > b ∧ b > c := by sorry

end NUMINAMATH_CALUDE_order_of_values_l3196_319688


namespace NUMINAMATH_CALUDE_cos_equality_for_n_l3196_319630

theorem cos_equality_for_n (n : ℤ) : ∃ n, 0 ≤ n ∧ n ≤ 180 ∧ Real.cos (n * π / 180) = Real.cos (259 * π / 180) ∧ n = 101 := by
  sorry

end NUMINAMATH_CALUDE_cos_equality_for_n_l3196_319630


namespace NUMINAMATH_CALUDE_x_value_proof_l3196_319628

theorem x_value_proof (x : ℝ) (h : 3*x - 4*x + 7*x = 180) : x = 30 := by
  sorry

end NUMINAMATH_CALUDE_x_value_proof_l3196_319628


namespace NUMINAMATH_CALUDE_cow_profit_is_600_l3196_319689

/-- Calculates the profit from selling a cow given the purchase price, daily food cost,
    health care cost, number of days kept, and selling price. -/
def cowProfit (purchasePrice foodCostPerDay healthCareCost numDays sellingPrice : ℕ) : ℕ :=
  sellingPrice - (purchasePrice + foodCostPerDay * numDays + healthCareCost)

/-- Theorem stating that the profit from selling the cow under given conditions is $600. -/
theorem cow_profit_is_600 :
  cowProfit 600 20 500 40 2500 = 600 := by
  sorry

#eval cowProfit 600 20 500 40 2500

end NUMINAMATH_CALUDE_cow_profit_is_600_l3196_319689


namespace NUMINAMATH_CALUDE_triangle_cosine_theorem_l3196_319625

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    and area S, prove that when 6S = a²sin A + b²sin B and (a+b)/c is maximized,
    cos C = 7/9 -/
theorem triangle_cosine_theorem (a b c : ℝ) (A B C : ℝ) (S : ℝ) 
  (h1 : 0 < a ∧ 0 < b ∧ 0 < c)
  (h2 : 0 < A ∧ A < π ∧ 0 < B ∧ B < π ∧ 0 < C ∧ C < π)
  (h3 : A + B + C = π)
  (h4 : S = (1/2) * a * b * Real.sin C)
  (h5 : 6 * S = a^2 * Real.sin A + b^2 * Real.sin B)
  (h6 : ∀ (x y z : ℝ), (x + y) / z ≤ (a + b) / c) :
  Real.cos C = 7/9 :=
sorry

end NUMINAMATH_CALUDE_triangle_cosine_theorem_l3196_319625


namespace NUMINAMATH_CALUDE_m_range_l3196_319694

/-- Given a real number a where 0 < a < 1, and m is a real number. -/
def a_condition (a : ℝ) : Prop := 0 < a ∧ a < 1

/-- The solution set of ax^2 - ax - 2a^2 > 1 is (-a, 2a) -/
def inequality_solution_set (a : ℝ) : Prop :=
  ∀ x, a * x^2 - a * x - 2 * a^2 > 1 ↔ -a < x ∧ x < 2*a

/-- The domain of f(x) = sqrt((1/a)^(x^2 + 2mx - m) - 1) is ℝ -/
def function_domain (a m : ℝ) : Prop :=
  ∀ x, (1/a)^(x^2 + 2*m*x - m) - 1 ≥ 0

/-- The main theorem stating that given the conditions, the range of m is [-1, 0] -/
theorem m_range (a m : ℝ) :
  a_condition a →
  inequality_solution_set a →
  function_domain a m →
  -1 ≤ m ∧ m ≤ 0 :=
by sorry

end NUMINAMATH_CALUDE_m_range_l3196_319694


namespace NUMINAMATH_CALUDE_no_fixed_point_implies_no_double_fixed_point_no_intersection_implies_no_double_intersection_l3196_319685

-- Part (a)
theorem no_fixed_point_implies_no_double_fixed_point
  (f : ℝ → ℝ) (hf : Continuous f) (h : ∀ x, f x ≠ x) :
  ∀ x, f (f x) ≠ x :=
sorry

-- Part (b)
theorem no_intersection_implies_no_double_intersection
  (f g : ℝ → ℝ) (hf : Continuous f) (hg : Continuous g)
  (h_comm : ∀ x, f (g x) = g (f x)) (h_neq : ∀ x, f x ≠ g x) :
  ∀ x, f (f x) ≠ g (g x) :=
sorry

end NUMINAMATH_CALUDE_no_fixed_point_implies_no_double_fixed_point_no_intersection_implies_no_double_intersection_l3196_319685


namespace NUMINAMATH_CALUDE_inequality_proof_l3196_319667

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h : x * y * z ≥ 1) :
  (x^5 - x^2) / (x^5 + y^2 + z^2) + (y^5 - y^2) / (y^5 + x^2 + z^2) + (z^5 - z^2) / (z^5 + x^2 + y^2) ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3196_319667


namespace NUMINAMATH_CALUDE_complement_of_M_l3196_319609

-- Define the universal set U
def U : Set ℝ := {x | 1 ≤ x ∧ x ≤ 5}

-- Define the set M
def M : Set ℝ := {1}

-- Theorem statement
theorem complement_of_M (x : ℝ) : x ∈ (U \ M) ↔ 1 < x ∧ x ≤ 5 := by sorry

end NUMINAMATH_CALUDE_complement_of_M_l3196_319609


namespace NUMINAMATH_CALUDE_even_multiples_sum_difference_l3196_319691

theorem even_multiples_sum_difference : 
  let n : ℕ := 2025
  let even_sum : ℕ := n * (2 + 2 * n)
  let multiples_of_three_sum : ℕ := n * (3 + 3 * n)
  (even_sum : ℤ) - (multiples_of_three_sum : ℤ) = -2052155 := by
  sorry

end NUMINAMATH_CALUDE_even_multiples_sum_difference_l3196_319691
