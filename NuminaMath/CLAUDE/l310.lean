import Mathlib

namespace NUMINAMATH_CALUDE_sequence_a_increasing_l310_31064

def a (n : ℕ) : ℚ := (n - 1 : ℚ) / (n + 1 : ℚ)

theorem sequence_a_increasing : ∀ n ≥ 2, a n < a (n + 1) := by
  sorry

end NUMINAMATH_CALUDE_sequence_a_increasing_l310_31064


namespace NUMINAMATH_CALUDE_num_pentagons_from_circle_points_l310_31000

/-- The number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := Nat.choose n k

/-- The number of distinct points on the circle -/
def num_points : ℕ := 15

/-- The number of vertices in a pentagon -/
def pentagon_vertices : ℕ := 5

/-- Theorem: The number of different convex pentagons formed by selecting 5 points
    from 15 distinct points on the circumference of a circle is 3003 -/
theorem num_pentagons_from_circle_points :
  choose num_points pentagon_vertices = 3003 := by sorry

end NUMINAMATH_CALUDE_num_pentagons_from_circle_points_l310_31000


namespace NUMINAMATH_CALUDE_rahul_savings_l310_31099

/-- Rahul's savings problem -/
theorem rahul_savings (nsc ppf : ℕ) : 
  (3 * nsc = 2 * ppf) →  -- One-third of NSC equals one-half of PPF
  (nsc + ppf = 180000) → -- Total savings
  (ppf = 72000) :=       -- PPF savings to prove
by sorry

end NUMINAMATH_CALUDE_rahul_savings_l310_31099


namespace NUMINAMATH_CALUDE_second_meeting_time_is_four_minutes_l310_31096

/-- Represents a swimmer in the pool --/
structure Swimmer where
  speed : ℝ
  startPosition : ℝ

/-- Represents the swimming pool scenario --/
structure PoolScenario where
  poolLength : ℝ
  swimmer1 : Swimmer
  swimmer2 : Swimmer
  firstMeetingTime : ℝ
  firstMeetingPosition : ℝ

/-- Calculates the time of the second meeting --/
def secondMeetingTime (scenario : PoolScenario) : ℝ :=
  sorry

/-- Theorem stating that the second meeting occurs 4 minutes after starting --/
theorem second_meeting_time_is_four_minutes (scenario : PoolScenario) 
    (h1 : scenario.poolLength = 50)
    (h2 : scenario.swimmer1.startPosition = 0)
    (h3 : scenario.swimmer2.startPosition = 50)
    (h4 : scenario.firstMeetingTime = 2)
    (h5 : scenario.firstMeetingPosition = 20) :
    secondMeetingTime scenario = 4 := by
  sorry

end NUMINAMATH_CALUDE_second_meeting_time_is_four_minutes_l310_31096


namespace NUMINAMATH_CALUDE_unique_a_value_l310_31069

theorem unique_a_value (a : ℝ) : 
  (∃! (x₁ x₂ x₃ : ℝ), x₁ ≠ x₂ ∧ x₂ ≠ x₃ ∧ x₁ ≠ x₃ ∧ 
    x₁ ∈ Set.Icc (0 : ℝ) (2 * Real.pi) ∧ 
    x₂ ∈ Set.Icc (0 : ℝ) (2 * Real.pi) ∧ 
    x₃ ∈ Set.Icc (0 : ℝ) (2 * Real.pi) ∧
    Real.sin x₁ + Real.sqrt 3 * Real.cos x₁ = a ∧
    Real.sin x₂ + Real.sqrt 3 * Real.cos x₂ = a ∧
    Real.sin x₃ + Real.sqrt 3 * Real.cos x₃ = a) ↔ 
  a = Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_unique_a_value_l310_31069


namespace NUMINAMATH_CALUDE_max_valid_n_l310_31091

def S : Set ℕ := {n | ∃ x y : ℕ, n = x * y * (x + y)}

def valid (a n : ℕ) : Prop :=
  ∀ k : ℕ, 1 ≤ k ∧ k ≤ n → (a + 2^k) ∈ S

theorem max_valid_n :
  ∃ a : ℕ, valid a 3 ∧ ∀ n : ℕ, valid a n → n ≤ 3 :=
sorry

end NUMINAMATH_CALUDE_max_valid_n_l310_31091


namespace NUMINAMATH_CALUDE_coins_per_pile_l310_31033

theorem coins_per_pile (total_piles : ℕ) (total_coins : ℕ) (h1 : total_piles = 10) (h2 : total_coins = 30) :
  ∃ (coins_per_pile : ℕ), coins_per_pile * total_piles = total_coins ∧ coins_per_pile = 3 :=
sorry

end NUMINAMATH_CALUDE_coins_per_pile_l310_31033


namespace NUMINAMATH_CALUDE_min_value_f_min_value_sum_squares_l310_31050

-- Define the function f(x)
def f (x : ℝ) : ℝ := |x - 2| + |x + 3|

-- Theorem for the minimum value of f(x)
theorem min_value_f : 
  ∃ m : ℝ, (∀ x : ℝ, f x ≥ m) ∧ (∃ x : ℝ, f x = m) ∧ m = 5 :=
sorry

-- Theorem for the minimum value of a^2 + 2b^2 + 3c^2
theorem min_value_sum_squares :
  ∃ m : ℝ, m = 15/2 ∧
  (∀ a b c : ℝ, a + 2*b + c = 5 → a^2 + 2*b^2 + 3*c^2 ≥ m) ∧
  (∃ a b c : ℝ, a + 2*b + c = 5 ∧ a^2 + 2*b^2 + 3*c^2 = m) :=
sorry

end NUMINAMATH_CALUDE_min_value_f_min_value_sum_squares_l310_31050


namespace NUMINAMATH_CALUDE_triangle_side_length_l310_31010

theorem triangle_side_length (A B C : ℝ) (AB BC AC : ℝ) : 
  BC = 1 → 
  A = π / 3 → 
  Real.sin B = 2 * Real.sin C → 
  AB = Real.sqrt 3 := by sorry

end NUMINAMATH_CALUDE_triangle_side_length_l310_31010


namespace NUMINAMATH_CALUDE_root_in_interval_l310_31030

-- Define the function f(x) = x³ - x - 3
def f (x : ℝ) : ℝ := x^3 - x - 3

-- State the theorem
theorem root_in_interval :
  ∃ (c : ℝ), c ∈ Set.Icc 1 2 ∧ f c = 0 := by
  sorry

end NUMINAMATH_CALUDE_root_in_interval_l310_31030


namespace NUMINAMATH_CALUDE_box_two_three_neg_one_l310_31040

-- Define the box operation for integers a, b, and c
def box (a b c : ℤ) : ℚ := (a ^ b : ℚ) - (b ^ c : ℚ) + (c ^ a : ℚ)

-- Theorem statement
theorem box_two_three_neg_one : box 2 3 (-1) = 26 / 3 := by
  sorry

end NUMINAMATH_CALUDE_box_two_three_neg_one_l310_31040


namespace NUMINAMATH_CALUDE_changhee_semester_average_l310_31046

/-- Calculates the average score for a semester given midterm and final exam scores and subject counts. -/
def semesterAverage (midtermAvg : ℚ) (midtermSubjects : ℕ) (finalAvg : ℚ) (finalSubjects : ℕ) : ℚ :=
  (midtermAvg * midtermSubjects + finalAvg * finalSubjects) / (midtermSubjects + finalSubjects)

/-- Proves that Changhee's semester average is 83.5 given the exam scores and subject counts. -/
theorem changhee_semester_average :
  semesterAverage 83.1 10 84 8 = 83.5 := by
  sorry

#eval semesterAverage 83.1 10 84 8

end NUMINAMATH_CALUDE_changhee_semester_average_l310_31046


namespace NUMINAMATH_CALUDE_team_score_proof_l310_31024

def team_size : ℕ := 15
def absent_members : ℕ := 5
def present_members : ℕ := team_size - absent_members
def scores : List ℕ := [4, 6, 2, 8, 3, 5, 10, 3, 7]

theorem team_score_proof :
  present_members = scores.length ∧ scores.sum = 48 := by sorry

end NUMINAMATH_CALUDE_team_score_proof_l310_31024


namespace NUMINAMATH_CALUDE_profit_difference_example_l310_31031

/-- Given a total profit and a ratio of division between two parties,
    calculates the difference in their profit shares. -/
def profit_difference (total_profit : ℚ) (ratio_x : ℚ) (ratio_y : ℚ) : ℚ :=
  let total_ratio := ratio_x + ratio_y
  let share_x := (ratio_x / total_ratio) * total_profit
  let share_y := (ratio_y / total_ratio) * total_profit
  share_x - share_y

/-- Theorem stating that for a total profit of 1000 and a ratio of 1/2 : 1/3,
    the difference in profit shares is 200. -/
theorem profit_difference_example :
  profit_difference 1000 (1/2) (1/3) = 200 := by
  sorry

end NUMINAMATH_CALUDE_profit_difference_example_l310_31031


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l310_31016

theorem sqrt_equation_solution :
  ∃! z : ℝ, Real.sqrt (10 + 3 * z) = 8 := by sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l310_31016


namespace NUMINAMATH_CALUDE_median_and_perpendicular_bisector_equations_l310_31053

/-- Given three points in a plane, prove the equations of the median and perpendicular bisector of a side -/
theorem median_and_perpendicular_bisector_equations 
  (A B C : ℝ × ℝ) 
  (hA : A = (1, 2)) 
  (hB : B = (-1, 4)) 
  (hC : C = (5, 2)) : 
  (∃ (m b : ℝ), ∀ x y : ℝ, (y = m * x + b) ↔ (y - 3 = -1 * (x - 0))) ∧ 
  (∃ (m b : ℝ), ∀ x y : ℝ, (y = m * x + b) ↔ (y = x + 3)) := by
  sorry


end NUMINAMATH_CALUDE_median_and_perpendicular_bisector_equations_l310_31053


namespace NUMINAMATH_CALUDE_bridge_crossing_time_l310_31035

/-- Proves that a man walking at 10 km/hr takes 15 minutes to cross a 2500-meter bridge -/
theorem bridge_crossing_time :
  let walking_speed : ℝ := 10  -- km/hr
  let bridge_length : ℝ := 2.5  -- km (2500 meters)
  let crossing_time : ℝ := bridge_length / walking_speed * 60  -- minutes
  crossing_time = 15 := by sorry

end NUMINAMATH_CALUDE_bridge_crossing_time_l310_31035


namespace NUMINAMATH_CALUDE_f_at_seven_l310_31082

def f (x : ℝ) : ℝ := 7*x^5 + 12*x^4 - 5*x^3 - 6*x^2 + 3*x - 5

theorem f_at_seven : f 7 = 144468 := by
  sorry

end NUMINAMATH_CALUDE_f_at_seven_l310_31082


namespace NUMINAMATH_CALUDE_angle_D_measure_l310_31090

/-- An isosceles triangle with specific angle relationships -/
structure IsoscelesTriangle where
  /-- Measure of angle E in degrees -/
  angleE : ℝ
  /-- The triangle is isosceles with angle D congruent to angle F -/
  isIsosceles : True
  /-- The measure of angle F is three times the measure of angle E -/
  angleFRelation : True

/-- The measure of angle D in the isosceles triangle -/
def measureAngleD (t : IsoscelesTriangle) : ℝ :=
  3 * t.angleE

/-- Theorem: The measure of angle D is approximately 77 degrees -/
theorem angle_D_measure (t : IsoscelesTriangle) :
  ‖measureAngleD t - 77‖ < 1 := by
  sorry

#check angle_D_measure

end NUMINAMATH_CALUDE_angle_D_measure_l310_31090


namespace NUMINAMATH_CALUDE_largest_product_of_three_l310_31062

def S : Set ℤ := {-3, -2, 4, 5}

theorem largest_product_of_three (a b c : ℤ) :
  a ∈ S → b ∈ S → c ∈ S → a ≠ b → b ≠ c → a ≠ c →
  ∀ x y z : ℤ, x ∈ S → y ∈ S → z ∈ S → x ≠ y → y ≠ z → x ≠ z →
  a * b * c ≤ 30 ∧ (∃ p q r : ℤ, p ∈ S ∧ q ∈ S ∧ r ∈ S ∧ p ≠ q ∧ q ≠ r ∧ p ≠ r ∧ p * q * r = 30) :=
by
  sorry

end NUMINAMATH_CALUDE_largest_product_of_three_l310_31062


namespace NUMINAMATH_CALUDE_special_line_equation_l310_31047

/-- A line passing through (-10, 10) with x-intercept four times y-intercept -/
structure SpecialLine where
  -- The line passes through (-10, 10)
  passes_through : (Int × Int)
  -- The x-intercept is four times the y-intercept
  x_intercept_relation : ℝ → ℝ → Prop

/-- The equation of the special line -/
def line_equation (L : SpecialLine) : Set (ℝ × ℝ) :=
  {(x, y) | x + y = 0 ∨ x + 4*y - 30 = 0}

/-- Theorem stating that the equation of the special line is correct -/
theorem special_line_equation (L : SpecialLine) 
  (h1 : L.passes_through = (-10, 10))
  (h2 : L.x_intercept_relation = λ x y => x = 4*y) :
  ∀ (x y : ℝ), (x, y) ∈ line_equation L ↔ 
    ((x + y = 0) ∨ (x + 4*y - 30 = 0)) :=
by
  sorry


end NUMINAMATH_CALUDE_special_line_equation_l310_31047


namespace NUMINAMATH_CALUDE_cells_after_one_week_l310_31061

/-- The number of cells after n days, given that each cell divides into three new cells every day -/
def num_cells (n : ℕ) : ℕ := 3^n

/-- Theorem: After 7 days, there will be 2187 cells -/
theorem cells_after_one_week : num_cells 7 = 2187 := by
  sorry

end NUMINAMATH_CALUDE_cells_after_one_week_l310_31061


namespace NUMINAMATH_CALUDE_prob_no_eight_correct_l310_31042

/-- The probability of selecting a number from 1 to 10000 that doesn't contain the digit 8 -/
def prob_no_eight : ℚ :=
  (9^4 : ℚ) / 10000

/-- Theorem stating that the probability of selecting a number from 1 to 10000
    that doesn't contain the digit 8 is equal to (9^4) / 10000 -/
theorem prob_no_eight_correct :
  prob_no_eight = (9^4 : ℚ) / 10000 := by sorry

end NUMINAMATH_CALUDE_prob_no_eight_correct_l310_31042


namespace NUMINAMATH_CALUDE_total_spent_is_12_30_l310_31029

/-- The cost of the football Alyssa bought -/
def football_cost : ℚ := 571/100

/-- The cost of the marbles Alyssa bought -/
def marbles_cost : ℚ := 659/100

/-- The total amount Alyssa spent on toys -/
def total_spent : ℚ := football_cost + marbles_cost

/-- Theorem stating that the total amount Alyssa spent on toys is $12.30 -/
theorem total_spent_is_12_30 : total_spent = 1230/100 := by
  sorry

end NUMINAMATH_CALUDE_total_spent_is_12_30_l310_31029


namespace NUMINAMATH_CALUDE_parabola_properties_l310_31003

/-- Parabola with equation y = ax(x-6) + 1 where a ≠ 0 -/
def parabola (a : ℝ) (x : ℝ) : ℝ := a * x * (x - 6) + 1

theorem parabola_properties (a : ℝ) (h : a ≠ 0) :
  /- Point (0,1) lies on the parabola -/
  (parabola a 0 = 1) ∧
  /- If the distance from the vertex to the x-axis is 5, then a = 2/3 or a = -4/9 -/
  (∃ (x : ℝ), (∀ y : ℝ, parabola a y ≥ parabola a x) →
    |parabola a x| = 5 → (a = 2/3 ∨ a = -4/9)) ∧
  /- If the length of the segment formed by the intersection of the parabola with the x-axis
     is less than or equal to 4, then 1/9 < a ≤ 1/5 -/
  ((∃ x₁ x₂ : ℝ, x₁ < x₂ ∧ parabola a x₁ = 0 ∧ parabola a x₂ = 0 ∧ x₂ - x₁ ≤ 4) →
    1/9 < a ∧ a ≤ 1/5) :=
by sorry

end NUMINAMATH_CALUDE_parabola_properties_l310_31003


namespace NUMINAMATH_CALUDE_equation_solution_l310_31057

theorem equation_solution : 
  let y : ℚ := 6/7
  ∀ (y : ℚ), y ≠ -2 ∧ y ≠ -1 →
  (7*y) / ((y+2)*(y+1)) - 4 / ((y+2)*(y+1)) = 2 / ((y+2)*(y+1)) →
  y = 6/7 := by
sorry

end NUMINAMATH_CALUDE_equation_solution_l310_31057


namespace NUMINAMATH_CALUDE_pastries_sum_is_147_l310_31020

/-- The total number of pastries made by Lola, Lulu, and Lila -/
def total_pastries (lola_cupcakes lola_poptarts lola_pies lola_eclairs
                    lulu_cupcakes lulu_poptarts lulu_pies lulu_eclairs
                    lila_cupcakes lila_poptarts lila_pies lila_eclairs : ℕ) : ℕ :=
  lola_cupcakes + lola_poptarts + lola_pies + lola_eclairs +
  lulu_cupcakes + lulu_poptarts + lulu_pies + lulu_eclairs +
  lila_cupcakes + lila_poptarts + lila_pies + lila_eclairs

/-- Theorem stating that the total number of pastries is 147 -/
theorem pastries_sum_is_147 :
  total_pastries 13 10 8 6 16 12 14 9 22 15 10 12 = 147 := by
  sorry

end NUMINAMATH_CALUDE_pastries_sum_is_147_l310_31020


namespace NUMINAMATH_CALUDE_sequence_general_term_1_l310_31065

theorem sequence_general_term_1 (n : ℕ) (S : ℕ → ℤ) (a : ℕ → ℤ) 
  (h : ∀ n, S n = 2 * n^2 - 3 * n + 2) :
  (a 1 = 1 ∧ ∀ n ≥ 2, a n = 4 * n - 5) ↔ 
  (∀ n, n ≥ 1 → a n = S n - S (n-1)) :=
sorry


end NUMINAMATH_CALUDE_sequence_general_term_1_l310_31065


namespace NUMINAMATH_CALUDE_sum_of_xyz_l310_31087

/-- An arithmetic sequence with six terms where the first term is 4 and the last term is 31 -/
def arithmetic_sequence (x y z : ℝ) : Prop :=
  let d := (31 - 4) / 5
  (y - x = d) ∧ (16 - y = d) ∧ (z - 16 = d)

/-- The theorem stating that the sum of x, y, and z in the given arithmetic sequence is 45.6 -/
theorem sum_of_xyz (x y z : ℝ) (h : arithmetic_sequence x y z) : x + y + z = 45.6 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_xyz_l310_31087


namespace NUMINAMATH_CALUDE_fabian_shopping_cost_l310_31006

/-- The cost of Fabian's shopping trip -/
def shopping_cost (apple_price : ℝ) (walnut_price : ℝ) (apple_quantity : ℝ) (sugar_quantity : ℝ) (walnut_quantity : ℝ) : ℝ :=
  let sugar_price := apple_price - 1
  apple_price * apple_quantity + sugar_price * sugar_quantity + walnut_price * walnut_quantity

/-- Theorem: The total cost of Fabian's shopping is $16 -/
theorem fabian_shopping_cost : 
  shopping_cost 2 6 5 3 0.5 = 16 := by
  sorry

end NUMINAMATH_CALUDE_fabian_shopping_cost_l310_31006


namespace NUMINAMATH_CALUDE_delaney_missed_bus_l310_31073

/-- Represents time in hours and minutes -/
structure Time where
  hours : Nat
  minutes : Nat

/-- Calculates the difference between two times in minutes -/
def timeDifference (t1 t2 : Time) : Nat :=
  (t1.hours * 60 + t1.minutes) - (t2.hours * 60 + t2.minutes)

/-- Adds minutes to a given time -/
def addMinutes (t : Time) (m : Nat) : Time :=
  let totalMinutes := t.hours * 60 + t.minutes + m
  { hours := totalMinutes / 60, minutes := totalMinutes % 60 }

theorem delaney_missed_bus (busLeaveTime : Time) (travelTime : Nat) (leftHomeTime : Time) :
  busLeaveTime = { hours := 8, minutes := 0 } →
  travelTime = 30 →
  leftHomeTime = { hours := 7, minutes := 50 } →
  timeDifference (addMinutes leftHomeTime travelTime) busLeaveTime = 20 := by
  sorry

end NUMINAMATH_CALUDE_delaney_missed_bus_l310_31073


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l310_31077

theorem quadratic_inequality_range (x : ℝ) (h : x^2 - 3*x + 2 < 0) :
  ∃ y ∈ Set.Ioo (-0.25 : ℝ) 0, y = x^2 - 3*x + 2 :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l310_31077


namespace NUMINAMATH_CALUDE_propositions_true_l310_31056

-- Definition of reciprocals
def are_reciprocals (x y : ℝ) : Prop := x * y = 1

-- Definition of real roots for a quadratic equation
def has_real_roots (b : ℝ) : Prop := ∃ x : ℝ, x^2 - 2*b*x + b^2 + b = 0

theorem propositions_true : 
  (∀ x y : ℝ, are_reciprocals x y → x * y = 1) ∧
  (∀ b : ℝ, ¬(has_real_roots b) → b > -1) ∧
  (∃ α β : ℝ, Real.sin (α + β) = Real.sin α + Real.sin β) :=
by sorry

end NUMINAMATH_CALUDE_propositions_true_l310_31056


namespace NUMINAMATH_CALUDE_polar_to_cartesian_coordinates_l310_31072

theorem polar_to_cartesian_coordinates :
  let r : ℝ := 2
  let θ : ℝ := π / 3
  let x : ℝ := r * Real.cos θ
  let y : ℝ := r * Real.sin θ
  (x = 1 ∧ y = Real.sqrt 3) :=
by sorry

end NUMINAMATH_CALUDE_polar_to_cartesian_coordinates_l310_31072


namespace NUMINAMATH_CALUDE_additional_cans_needed_l310_31078

def martha_cans : ℕ := 90
def diego_cans : ℕ := martha_cans / 2 + 10
def leah_cans : ℕ := martha_cans / 3 - 5

def martha_aluminum : ℕ := (martha_cans * 70) / 100
def diego_aluminum : ℕ := (diego_cans * 50) / 100
def leah_aluminum : ℕ := (leah_cans * 80) / 100

def total_needed : ℕ := 200

theorem additional_cans_needed :
  total_needed - (martha_aluminum + diego_aluminum + leah_aluminum) = 90 :=
by sorry

end NUMINAMATH_CALUDE_additional_cans_needed_l310_31078


namespace NUMINAMATH_CALUDE_parabola_directrix_l310_31041

/-- Given a parabola with equation y² = 2x, its directrix has the equation x = -1/2 -/
theorem parabola_directrix (x y : ℝ) : y^2 = 2*x → (∃ (p : ℝ), p > 0 ∧ y^2 = 4*p*x ∧ x = -1/2) := by
  sorry

end NUMINAMATH_CALUDE_parabola_directrix_l310_31041


namespace NUMINAMATH_CALUDE_find_g_of_x_l310_31044

/-- Given that 4x^4 + 2x^2 - 7x + 3 + g(x) = 5x^3 - 8x^2 + 4x - 1,
    prove that g(x) = -4x^4 + 5x^3 - 10x^2 + 11x - 4 -/
theorem find_g_of_x (g : ℝ → ℝ) :
  (∀ x : ℝ, 4 * x^4 + 2 * x^2 - 7 * x + 3 + g x = 5 * x^3 - 8 * x^2 + 4 * x - 1) →
  (∀ x : ℝ, g x = -4 * x^4 + 5 * x^3 - 10 * x^2 + 11 * x - 4) :=
by sorry

end NUMINAMATH_CALUDE_find_g_of_x_l310_31044


namespace NUMINAMATH_CALUDE_gumball_difference_l310_31043

theorem gumball_difference (carolyn lew amanda tom : ℕ) (x : ℕ) :
  carolyn = 17 →
  lew = 12 →
  amanda = 24 →
  tom = 8 →
  14 ≤ (carolyn + lew + amanda + tom + x) / 7 →
  (carolyn + lew + amanda + tom + x) / 7 ≤ 32 →
  ∃ (x_min x_max : ℕ), x_min ≤ x ∧ x ≤ x_max ∧ x_max - x_min = 126 :=
by sorry

end NUMINAMATH_CALUDE_gumball_difference_l310_31043


namespace NUMINAMATH_CALUDE_simultaneous_inequality_condition_l310_31060

theorem simultaneous_inequality_condition (a : ℝ) : 
  (∃ x₀ : ℝ, x₀^2 - a*x₀ + a + 3 < 0 ∧ a*x₀ - 2*a < 0) ↔ a > 7 := by
  sorry

end NUMINAMATH_CALUDE_simultaneous_inequality_condition_l310_31060


namespace NUMINAMATH_CALUDE_group_size_proof_l310_31067

def group_size (adult_meal_cost : ℕ) (total_cost : ℕ) (num_kids : ℕ) : ℕ :=
  (total_cost / adult_meal_cost) + num_kids

theorem group_size_proof :
  group_size 2 14 2 = 9 := by
  sorry

end NUMINAMATH_CALUDE_group_size_proof_l310_31067


namespace NUMINAMATH_CALUDE_feathers_needed_for_wings_l310_31032

theorem feathers_needed_for_wings 
  (feathers_per_set : ℕ) 
  (num_sets : ℕ) 
  (charlie_feathers : ℕ) 
  (susan_feathers : ℕ) :
  feathers_per_set = 900 →
  num_sets = 2 →
  charlie_feathers = 387 →
  susan_feathers = 250 →
  feathers_per_set * num_sets - (charlie_feathers + susan_feathers) = 1163 :=
by sorry

end NUMINAMATH_CALUDE_feathers_needed_for_wings_l310_31032


namespace NUMINAMATH_CALUDE_no_adjacent_standing_probability_l310_31048

-- Define the number of people
def n : ℕ := 10

-- Define a function to represent the recursive relation
def a : ℕ → ℕ
| 0 => 1
| 1 => 2
| m + 2 => a (m + 1) + a m

-- Define the probability
def probability : ℚ := a n / 2^n

-- Theorem statement
theorem no_adjacent_standing_probability : probability = 123 / 1024 := by
  sorry

end NUMINAMATH_CALUDE_no_adjacent_standing_probability_l310_31048


namespace NUMINAMATH_CALUDE_value_equals_eleven_l310_31055

theorem value_equals_eleven (number : ℝ) (value : ℝ) : 
  number = 10 → 
  (number / 2) + 6 = value → 
  value = 11 := by
sorry

end NUMINAMATH_CALUDE_value_equals_eleven_l310_31055


namespace NUMINAMATH_CALUDE_line_circle_relationship_l310_31084

/-- The line equation -/
def line_equation (k x y : ℝ) : Prop :=
  (3*k + 2) * x - k * y - 2 = 0

/-- The circle equation -/
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 - 2*x - 2*y - 2 = 0

/-- The theorem stating the positional relationship between the line and the circle -/
theorem line_circle_relationship :
  ∀ k : ℝ, ∃ x y : ℝ, 
    (line_equation k x y ∧ circle_equation x y) ∨ 
    (∃ x₀ y₀ : ℝ, line_equation k x₀ y₀ ∧ circle_equation x₀ y₀ ∧ 
      ∀ x y : ℝ, line_equation k x y ∧ circle_equation x y → (x, y) = (x₀, y₀)) :=
by sorry

end NUMINAMATH_CALUDE_line_circle_relationship_l310_31084


namespace NUMINAMATH_CALUDE_algae_coverage_day_18_and_19_l310_31004

/-- Represents the coverage of algae on the pond on a given day -/
def algaeCoverage (day : ℕ) : ℚ :=
  (1 : ℚ) / 3^(20 - day)

/-- The problem statement -/
theorem algae_coverage_day_18_and_19 :
  algaeCoverage 18 < (1 : ℚ) / 4 ∧ (1 : ℚ) / 4 < algaeCoverage 19 := by
  sorry

#eval algaeCoverage 18  -- Expected: 1/9
#eval algaeCoverage 19  -- Expected: 1/3

end NUMINAMATH_CALUDE_algae_coverage_day_18_and_19_l310_31004


namespace NUMINAMATH_CALUDE_arithmetic_expression_result_l310_31009

theorem arithmetic_expression_result : 3 + 15 / 3 - 2^3 = 0 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_expression_result_l310_31009


namespace NUMINAMATH_CALUDE_f_derivative_at_zero_l310_31026

noncomputable def f (x : ℝ) : ℝ := (x^2 + 2*x) / Real.exp x

theorem f_derivative_at_zero : 
  deriv f 0 = 2 := by sorry

end NUMINAMATH_CALUDE_f_derivative_at_zero_l310_31026


namespace NUMINAMATH_CALUDE_number_of_children_l310_31068

/-- Given that each child has 8 crayons and there are 56 crayons in total,
    prove that the number of children is 7. -/
theorem number_of_children (crayons_per_child : ℕ) (total_crayons : ℕ) 
  (h1 : crayons_per_child = 8) (h2 : total_crayons = 56) :
  total_crayons / crayons_per_child = 7 := by
  sorry

end NUMINAMATH_CALUDE_number_of_children_l310_31068


namespace NUMINAMATH_CALUDE_arithmetic_sequence_length_l310_31005

theorem arithmetic_sequence_length
  (a₁ : ℤ)
  (aₙ : ℤ)
  (d : ℤ)
  (h1 : a₁ = -3)
  (h2 : aₙ = 45)
  (h3 : d = 4)
  (h4 : aₙ = a₁ + (n - 1) * d) :
  n = 13 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_length_l310_31005


namespace NUMINAMATH_CALUDE_specific_trade_profit_l310_31012

/-- Represents a trading scenario for baseball cards -/
structure CardTrade where
  card_given_value : ℝ
  cards_given_count : ℕ
  card_received_value : ℝ

/-- Calculates the profit from a card trade -/
def trade_profit (trade : CardTrade) : ℝ :=
  trade.card_received_value - (trade.card_given_value * trade.cards_given_count)

/-- Theorem stating that the specific trade results in a $5 profit -/
theorem specific_trade_profit :
  let trade : CardTrade := {
    card_given_value := 8,
    cards_given_count := 2,
    card_received_value := 21
  }
  trade_profit trade = 5 := by sorry

end NUMINAMATH_CALUDE_specific_trade_profit_l310_31012


namespace NUMINAMATH_CALUDE_three_fifths_of_negative_twelve_sevenths_l310_31079

theorem three_fifths_of_negative_twelve_sevenths :
  (3 : ℚ) / 5 * (-12 : ℚ) / 7 = -36 / 35 := by sorry

end NUMINAMATH_CALUDE_three_fifths_of_negative_twelve_sevenths_l310_31079


namespace NUMINAMATH_CALUDE_shipping_cost_proof_l310_31095

/-- Calculates the total shipping cost for fish -/
def total_shipping_cost (total_weight : ℕ) (crate_weight : ℕ) (cost_per_crate : ℚ) (surcharge_per_crate : ℚ) (flat_fee : ℚ) : ℚ :=
  let num_crates : ℕ := total_weight / crate_weight
  let crate_total_cost : ℚ := (cost_per_crate + surcharge_per_crate) * num_crates
  crate_total_cost + flat_fee

/-- Proves that the total shipping cost for the given conditions is $46.00 -/
theorem shipping_cost_proof :
  total_shipping_cost 540 30 (3/2) (1/2) 10 = 46 := by
  sorry

end NUMINAMATH_CALUDE_shipping_cost_proof_l310_31095


namespace NUMINAMATH_CALUDE_empty_bottle_weight_l310_31051

/-- Given a full bottle of sesame oil weighing 3.4 kg and the same bottle weighing 2.98 kg
    after using 1/5 of the oil, the weight of the empty bottle is 1.3 kg. -/
theorem empty_bottle_weight (full_weight : ℝ) (partial_weight : ℝ) (empty_weight : ℝ) : 
  full_weight = 3.4 →
  partial_weight = 2.98 →
  full_weight = empty_weight + (5/4) * (partial_weight - empty_weight) →
  empty_weight = 1.3 := by
  sorry

#check empty_bottle_weight

end NUMINAMATH_CALUDE_empty_bottle_weight_l310_31051


namespace NUMINAMATH_CALUDE_arithmetic_seq_properties_l310_31070

/-- An arithmetic sequence with a_1 = 1 and a_3 - a_2 = 1 -/
def arithmetic_seq (n : ℕ) : ℕ := n

/-- Sum of the first n terms of the arithmetic sequence -/
def arithmetic_seq_sum (n : ℕ) : ℕ := n * (n + 1) / 2

theorem arithmetic_seq_properties :
  let a := arithmetic_seq
  let S := arithmetic_seq_sum
  (∀ n : ℕ, n ≥ 1 → a n = n) ∧
  (∀ n : ℕ, n ≥ 1 → S n = n * (n + 1) / 2) :=
by
  sorry

#check arithmetic_seq_properties

end NUMINAMATH_CALUDE_arithmetic_seq_properties_l310_31070


namespace NUMINAMATH_CALUDE_not_zero_necessary_not_sufficient_for_positive_l310_31049

theorem not_zero_necessary_not_sufficient_for_positive (x : ℝ) :
  (∃ x, x ≠ 0 ∧ x ≤ 0) ∧ (∀ x, x > 0 → x ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_not_zero_necessary_not_sufficient_for_positive_l310_31049


namespace NUMINAMATH_CALUDE_f_1000_is_even_l310_31039

/-- A function that satisfies the given functional equation -/
def SatisfiesEquation (f : ℕ → ℕ) : Prop :=
  ∀ n : ℕ, n > 0 → (f^[f n] n) = n^2 / (f (f n))

/-- Theorem stating that f(1000) is even for any function satisfying the equation -/
theorem f_1000_is_even (f : ℕ → ℕ) (h : SatisfiesEquation f) : 
  ∃ k : ℕ, f 1000 = 2 * k :=
sorry

end NUMINAMATH_CALUDE_f_1000_is_even_l310_31039


namespace NUMINAMATH_CALUDE_wall_painting_theorem_l310_31045

theorem wall_painting_theorem (heidi_time tim_time total_time : ℚ) 
  (h1 : heidi_time = 45)
  (h2 : tim_time = 30)
  (h3 : total_time = 9) :
  let heidi_rate : ℚ := 1 / heidi_time
  let tim_rate : ℚ := 1 / tim_time
  let combined_rate : ℚ := heidi_rate + tim_rate
  (combined_rate * total_time) = 1/2 := by sorry

end NUMINAMATH_CALUDE_wall_painting_theorem_l310_31045


namespace NUMINAMATH_CALUDE_jane_sunflower_seeds_l310_31017

/-- Calculates the total number of sunflower seeds given the number of cans and seeds per can. -/
def total_seeds (num_cans : ℕ) (seeds_per_can : ℕ) : ℕ :=
  num_cans * seeds_per_can

/-- Theorem stating that 9 cans with 6 seeds each results in 54 total seeds. -/
theorem jane_sunflower_seeds :
  total_seeds 9 6 = 54 := by
  sorry

end NUMINAMATH_CALUDE_jane_sunflower_seeds_l310_31017


namespace NUMINAMATH_CALUDE_student_sister_weight_l310_31097

/-- The combined weight of a student and his sister, given specific conditions --/
theorem student_sister_weight (student_weight sister_weight : ℝ) : 
  student_weight = 79 →
  student_weight - 5 = 2 * sister_weight →
  student_weight + sister_weight = 116 := by
sorry

end NUMINAMATH_CALUDE_student_sister_weight_l310_31097


namespace NUMINAMATH_CALUDE_solution_sets_l310_31011

def f (a x : ℝ) := x^2 - (a - 1) * x - a

theorem solution_sets (a : ℝ) :
  (a = 2 → {x : ℝ | f 2 x < 0} = {x : ℝ | -1 < x ∧ x < 2}) ∧
  (a > -1 → {x : ℝ | f a x > 0} = {x : ℝ | x < -1 ∨ x > a}) ∧
  (a = -1 → {x : ℝ | f (-1) x > 0} = {x : ℝ | x < -1 ∨ x > -1}) ∧
  (a < -1 → {x : ℝ | f a x > 0} = {x : ℝ | x < a ∨ x > -1}) :=
by sorry

end NUMINAMATH_CALUDE_solution_sets_l310_31011


namespace NUMINAMATH_CALUDE_age_problem_l310_31059

theorem age_problem (a b c : ℕ) : 
  a = b + 2 →
  b = 2 * c →
  a + b + c = 22 →
  b = 8 := by sorry

end NUMINAMATH_CALUDE_age_problem_l310_31059


namespace NUMINAMATH_CALUDE_greatest_x_with_lcm_l310_31054

theorem greatest_x_with_lcm (x : ℕ) : 
  (∃ (y : ℕ), y = Nat.lcm x (Nat.lcm 8 12) ∧ y = 120) → 
  x ≤ 120 ∧ ∃ (z : ℕ), z = 120 ∧ Nat.lcm z (Nat.lcm 8 12) = 120 :=
by sorry

end NUMINAMATH_CALUDE_greatest_x_with_lcm_l310_31054


namespace NUMINAMATH_CALUDE_find_x1_l310_31058

theorem find_x1 (x1 x2 x3 : ℝ) 
  (h1 : 0 ≤ x3 ∧ x3 ≤ x2 ∧ x2 ≤ x1 ∧ x1 ≤ 1)
  (h2 : (1 - x1)^2 + (x1 - x2)^2 + (x2 - x3)^2 + x3^2 = 3/4) :
  x1 = 3 * Real.sqrt 3 / 8 := by
  sorry

end NUMINAMATH_CALUDE_find_x1_l310_31058


namespace NUMINAMATH_CALUDE_cubic_equation_solution_l310_31085

theorem cubic_equation_solution 
  (a b c d : ℝ) 
  (h1 : a * d = b * c) 
  (h2 : a * d ≠ 0) 
  (h3 : b * d < 0) :
  let x1 := -b / a
  let x2 := Real.sqrt (-d / b)
  let x3 := -Real.sqrt (-d / b)
  (a * x1^3 + b * x1^2 + c * x1 + d = 0) ∧
  (a * x2^3 + b * x2^2 + c * x2 + d = 0) ∧
  (a * x3^3 + b * x3^2 + c * x3 + d = 0) :=
by sorry

end NUMINAMATH_CALUDE_cubic_equation_solution_l310_31085


namespace NUMINAMATH_CALUDE_jefferson_high_school_groups_l310_31007

/-- Represents the number of students in exactly two groups -/
def students_in_two_groups (total_students : ℕ) (orchestra : ℕ) (band : ℕ) (chorus : ℕ) (in_any_group : ℕ) : ℕ :=
  orchestra + band + chorus - in_any_group

/-- Theorem: Given the conditions from Jefferson High School, 
    the number of students in exactly two groups is 130 -/
theorem jefferson_high_school_groups : 
  students_in_two_groups 500 120 190 220 400 = 130 := by
  sorry

end NUMINAMATH_CALUDE_jefferson_high_school_groups_l310_31007


namespace NUMINAMATH_CALUDE_thirty_two_distributions_l310_31075

/-- Represents a knockout tournament with 6 players. -/
structure Tournament :=
  (players : Fin 6 → ℕ)

/-- The number of possible outcomes for each match. -/
def match_outcomes : ℕ := 2

/-- The number of rounds in the tournament. -/
def num_rounds : ℕ := 5

/-- Calculates the total number of possible prize distribution orders. -/
def prize_distributions (t : Tournament) : ℕ :=
  match_outcomes ^ num_rounds

/-- Theorem stating that there are 32 possible prize distribution orders. -/
theorem thirty_two_distributions (t : Tournament) :
  prize_distributions t = 32 := by
  sorry

end NUMINAMATH_CALUDE_thirty_two_distributions_l310_31075


namespace NUMINAMATH_CALUDE_sanity_determination_question_exists_l310_31094

/-- Represents the sanity state of a guest -/
inductive Sanity
| Sane
| Insane

/-- Represents possible answers to a question -/
inductive Answer
| Yes
| Ball

/-- A function representing how a guest answers a question based on their sanity -/
def guest_answer (s : Sanity) : Answer :=
  match s with
  | Sanity.Sane => Answer.Ball
  | Sanity.Insane => Answer.Yes

/-- The theorem stating that there exists a question that can determine a guest's sanity -/
theorem sanity_determination_question_exists :
  ∃ (question : Sanity → Answer),
    (∀ s : Sanity, question s = guest_answer s) ∧
    (∀ s₁ s₂ : Sanity, question s₁ = question s₂ → s₁ = s₂) :=
by sorry

end NUMINAMATH_CALUDE_sanity_determination_question_exists_l310_31094


namespace NUMINAMATH_CALUDE_largest_five_digit_with_product_7_l310_31028

/-- The product of the first 7 positive integers -/
def product_7 : ℕ := 7 * 6 * 5 * 4 * 3 * 2 * 1

/-- A function to check if a number is a five-digit integer -/
def is_five_digit (n : ℕ) : Prop := 10000 ≤ n ∧ n ≤ 99999

/-- A function to calculate the product of digits of a number -/
def digit_product (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) * digit_product (n / 10)

/-- The theorem stating that 98752 is the largest five-digit integer
    whose digits have a product equal to (7)(6)(5)(4)(3)(2)(1) -/
theorem largest_five_digit_with_product_7 :
  (is_five_digit 98752) ∧ 
  (digit_product 98752 = product_7) ∧ 
  (∀ n : ℕ, is_five_digit n → digit_product n = product_7 → n ≤ 98752) :=
by sorry

end NUMINAMATH_CALUDE_largest_five_digit_with_product_7_l310_31028


namespace NUMINAMATH_CALUDE_milk_calculation_l310_31074

theorem milk_calculation (initial : ℚ) (given : ℚ) (received : ℚ) :
  initial = 5 →
  given = 18 / 4 →
  received = 7 / 4 →
  initial - given + received = 9 / 4 := by
sorry

end NUMINAMATH_CALUDE_milk_calculation_l310_31074


namespace NUMINAMATH_CALUDE_number_problem_l310_31002

theorem number_problem : ∃ x : ℝ, (0.3 * x = 0.6 * 50 + 30) ∧ (x = 200) := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l310_31002


namespace NUMINAMATH_CALUDE_range_of_e_l310_31025

theorem range_of_e (a b c d e : ℝ) 
  (sum_eq : a + b + c + d + e = 8) 
  (sum_squares_eq : a^2 + b^2 + c^2 + d^2 + e^2 = 16) : 
  0 ≤ e ∧ e ≤ 16/5 := by
  sorry

end NUMINAMATH_CALUDE_range_of_e_l310_31025


namespace NUMINAMATH_CALUDE_triangular_pizza_area_l310_31052

theorem triangular_pizza_area :
  ∀ (base height hypotenuse : ℝ),
  base = 9 →
  hypotenuse = 15 →
  base ^ 2 + height ^ 2 = hypotenuse ^ 2 →
  (base * height) / 2 = 54 :=
by
  sorry

end NUMINAMATH_CALUDE_triangular_pizza_area_l310_31052


namespace NUMINAMATH_CALUDE_commission_breakpoint_l310_31092

/-- Proves that for a sale of $800, if the commission is 20% of the first $X plus 25% of the remainder, 
    and the total commission is 21.875% of the sale, then X = $500. -/
theorem commission_breakpoint (X : ℝ) : 
  let total_sale := 800
  let commission_rate_1 := 0.20
  let commission_rate_2 := 0.25
  let total_commission_rate := 0.21875
  commission_rate_1 * X + commission_rate_2 * (total_sale - X) = total_commission_rate * total_sale →
  X = 500 :=
by sorry

end NUMINAMATH_CALUDE_commission_breakpoint_l310_31092


namespace NUMINAMATH_CALUDE_hyperbola_eccentricity_l310_31036

/-- Given a hyperbola with semi-major axis a and semi-minor axis b, 
    a point P on its right branch, F as its right focus, 
    and M on the line x = -a²/c, where c is the focal distance,
    prove that if OP = OF + OM and OP ⋅ FM = 0, then the eccentricity is 2 -/
theorem hyperbola_eccentricity (a b c : ℝ) (P F M O : ℝ × ℝ) : 
  a > 0 → b > 0 → c > 0 →
  (P.1^2 / a^2) - (P.2^2 / b^2) = 1 →  -- P is on the right branch of the hyperbola
  P.1 > 0 →  -- P is on the right branch
  F = (c, 0) →  -- F is the right focus
  M.1 = -a^2 / c →  -- M is on the line x = -a²/c
  P.1 - O.1 = F.1 - O.1 + M.1 - O.1 →  -- OP = OF + OM (x-component)
  P.2 - O.2 = F.2 - O.2 + M.2 - O.2 →  -- OP = OF + OM (y-component)
  (P.1 - F.1) * (M.1 - F.1) + (P.2 - F.2) * (M.2 - F.2) = 0 →  -- OP ⋅ FM = 0
  c / a = 2 :=  -- eccentricity is 2
by sorry

end NUMINAMATH_CALUDE_hyperbola_eccentricity_l310_31036


namespace NUMINAMATH_CALUDE_set_A_representation_l310_31093

def A : Set (ℤ × ℤ) := {p | p.1^2 = p.2 + 1 ∧ |p.1| < 2}

theorem set_A_representation : A = {(-1, 0), (0, -1), (1, 0)} := by
  sorry

end NUMINAMATH_CALUDE_set_A_representation_l310_31093


namespace NUMINAMATH_CALUDE_ivanov_family_problem_l310_31063

/-- The Ivanov family problem -/
theorem ivanov_family_problem (father mother daughter : ℕ) : 
  father + mother + daughter = 74 →  -- Current sum of ages
  father + mother + daughter - 30 = 47 →  -- Sum of ages 10 years ago
  mother - 26 = daughter →  -- Mother's age at daughter's birth
  mother = 33 := by
  sorry

end NUMINAMATH_CALUDE_ivanov_family_problem_l310_31063


namespace NUMINAMATH_CALUDE_range_of_x_l310_31021

theorem range_of_x (x : ℝ) : Real.sqrt ((x - 3)^2) = x - 3 → x ≥ 3 := by sorry

end NUMINAMATH_CALUDE_range_of_x_l310_31021


namespace NUMINAMATH_CALUDE_rectangle_tiling_tiling_count_l310_31013

/-- A piece is a shape that can be used to tile a rectangle -/
structure Piece where
  shape : Set (ℕ × ℕ)

/-- A tiling is a way to cover a rectangle with pieces -/
def Tiling (m n : ℕ) (pieces : Finset Piece) :=
  Set (ℕ × ℕ × Piece)

/-- The number of ways to tile a 5 × 2k rectangle with 2k pieces -/
def TilingCount (k : ℕ) : ℕ := sorry

/-- The main theorem -/
theorem rectangle_tiling (n : ℕ) (pieces : Finset Piece) :
  (∃ (t : Tiling 5 n pieces), pieces.card = n) → Even n :=
sorry

/-- The counting theorem -/
theorem tiling_count (k : ℕ) :
  k ≥ 3 → TilingCount k > 2 * 3^(k - 1) :=
sorry

end NUMINAMATH_CALUDE_rectangle_tiling_tiling_count_l310_31013


namespace NUMINAMATH_CALUDE_johns_number_l310_31027

def is_two_digit_prime (n : ℕ) : Prop := 10 ≤ n ∧ n < 100 ∧ Nat.Prime n

theorem johns_number 
  (t m j d : ℕ) 
  (h1 : is_two_digit_prime t)
  (h2 : is_two_digit_prime m)
  (h3 : is_two_digit_prime j)
  (h4 : is_two_digit_prime d)
  (h5 : t ≠ m ∧ t ≠ j ∧ t ≠ d ∧ m ≠ j ∧ m ≠ d ∧ j ≠ d)
  (h6 : t + j = 26)
  (h7 : m + d = 32)
  (h8 : j + d = 34)
  (h9 : t + d = 36) : 
  j = 13 := by sorry

end NUMINAMATH_CALUDE_johns_number_l310_31027


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l310_31081

/-- A quadratic function with vertex (-3, 4) passing through (1, 2) -/
def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

theorem sum_of_coefficients (a b c : ℝ) :
  (∀ x, f a b c x = a * x^2 + b * x + c) →
  f a b c (-3) = 4 →
  (∀ h : ℝ, f a b c (-3 + h) = f a b c (-3 - h)) →
  f a b c 1 = 2 →
  a + b + c = 3 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l310_31081


namespace NUMINAMATH_CALUDE_stratified_sampling_third_year_count_l310_31083

theorem stratified_sampling_third_year_count 
  (total_students : ℕ) 
  (third_year_students : ℕ) 
  (sample_size : ℕ) 
  (h1 : total_students = 900) 
  (h2 : third_year_students = 400) 
  (h3 : sample_size = 45) :
  (third_year_students * sample_size) / total_students = 20 := by
  sorry

end NUMINAMATH_CALUDE_stratified_sampling_third_year_count_l310_31083


namespace NUMINAMATH_CALUDE_power_multiplication_calculate_3000_power_l310_31037

theorem power_multiplication (a : ℕ) (m n : ℕ) :
  a * (a ^ n) = a ^ (n + 1) :=
by sorry

theorem calculate_3000_power :
  3000 * (3000 ^ 1999) = 3000 ^ 2000 :=
by sorry

end NUMINAMATH_CALUDE_power_multiplication_calculate_3000_power_l310_31037


namespace NUMINAMATH_CALUDE_log_power_sum_l310_31034

theorem log_power_sum (c d : ℝ) (hc : c = Real.log 16) (hd : d = Real.log 25) :
  (9 : ℝ) ^ (c / d) + (4 : ℝ) ^ (d / c) = 4421 / 625 := by
  sorry

end NUMINAMATH_CALUDE_log_power_sum_l310_31034


namespace NUMINAMATH_CALUDE_arithmetic_sequence_formula_l310_31066

def arithmetic_sequence (a₁ : ℤ) (d : ℤ) (n : ℕ) : ℤ := a₁ + (n - 1) * d

theorem arithmetic_sequence_formula :
  ∀ n : ℕ, n ≥ 1 → arithmetic_sequence (-3) 4 n = 4*n - 7 :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_formula_l310_31066


namespace NUMINAMATH_CALUDE_value_of_a_l310_31080

theorem value_of_a (M : Set ℝ) (a : ℝ) : 
  M = {0, 1, a + 1} → -1 ∈ M → a = -2 := by
  sorry

end NUMINAMATH_CALUDE_value_of_a_l310_31080


namespace NUMINAMATH_CALUDE_minor_premise_is_proposition1_l310_31038

-- Define the propositions
def proposition1 : Prop := 0 < (1/2 : ℝ) ∧ (1/2 : ℝ) < 1
def proposition2 (a : ℝ) : Prop := ∀ x y : ℝ, x < y → a^y < a^x
def proposition3 : Prop := ∀ a : ℝ, 0 < a ∧ a < 1 → (∀ x y : ℝ, x < y → a^y < a^x)

-- Define the syllogism structure
structure Syllogism :=
  (major_premise : Prop)
  (minor_premise : Prop)
  (conclusion : Prop)

-- Theorem statement
theorem minor_premise_is_proposition1 :
  ∃ s : Syllogism, s.major_premise = proposition3 ∧
                   s.minor_premise = proposition1 ∧
                   s.conclusion = proposition2 (1/2) :=
sorry

end NUMINAMATH_CALUDE_minor_premise_is_proposition1_l310_31038


namespace NUMINAMATH_CALUDE_linear_function_composition_l310_31019

theorem linear_function_composition (a b : ℝ) : 
  (∀ x : ℝ, (3 * (a * x + b) - 6) = 4 * x + 7) → 
  a + b = 17 / 3 := by
sorry

end NUMINAMATH_CALUDE_linear_function_composition_l310_31019


namespace NUMINAMATH_CALUDE_line_slope_intercept_product_l310_31001

/-- For a line y = mx + b with negative slope m and positive y-intercept b, 
    the product mb satisfies -1 < mb < 0 -/
theorem line_slope_intercept_product (m b : ℝ) (h1 : m < 0) (h2 : b > 0) : 
  -1 < m * b ∧ m * b < 0 := by
  sorry

end NUMINAMATH_CALUDE_line_slope_intercept_product_l310_31001


namespace NUMINAMATH_CALUDE_rectangular_garden_diagonal_ratio_l310_31089

theorem rectangular_garden_diagonal_ratio (b : ℝ) (h : b > 0) :
  let a := 3 * b
  let diagonal := Real.sqrt (a^2 + b^2)
  let perimeter := 2 * (a + b)
  diagonal / perimeter = Real.sqrt 10 / 8 ∧ perimeter - diagonal = b :=
by
  sorry

end NUMINAMATH_CALUDE_rectangular_garden_diagonal_ratio_l310_31089


namespace NUMINAMATH_CALUDE_solution_set_l310_31023

def system_solution (x y : ℝ) : Prop :=
  5 * x^2 + 14 * x * y + 10 * y^2 = 17 ∧ 4 * x^2 + 10 * x * y + 6 * y^2 = 8

theorem solution_set : 
  {p : ℝ × ℝ | system_solution p.1 p.2} = {(-1, 2), (11, -7), (-11, 7), (1, -2)} := by
sorry

end NUMINAMATH_CALUDE_solution_set_l310_31023


namespace NUMINAMATH_CALUDE_twenty_fifth_in_base5_l310_31008

/-- Converts a natural number to its representation in base 5 --/
def toBase5 (n : ℕ) : List ℕ :=
  if n = 0 then [0]
  else
    let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc
      else aux (m / 5) ((m % 5) :: acc)
    aux n []

/-- Checks if a list of digits represents a valid number in base 5 --/
def isValidBase5 (l : List ℕ) : Prop :=
  l.all (· < 5)

/-- Converts a list of base 5 digits to a natural number --/
def fromBase5 (l : List ℕ) : ℕ :=
  l.foldl (fun acc d => 5 * acc + d) 0

theorem twenty_fifth_in_base5 :
  ∃ (l : List ℕ), isValidBase5 l ∧ fromBase5 l = 25 ∧ l = [1, 0, 0] :=
sorry

end NUMINAMATH_CALUDE_twenty_fifth_in_base5_l310_31008


namespace NUMINAMATH_CALUDE_linear_function_composition_l310_31086

/-- A linear function is a function of the form f(x) = kx + b for some constants k and b. -/
def LinearFunction (f : ℝ → ℝ) : Prop :=
  ∃ k b : ℝ, ∀ x, f x = k * x + b

/-- The main theorem: if f is a linear function satisfying f(f(x)) = 4x + 6,
    then f(x) = 2x + 2 or f(x) = -2x - 6. -/
theorem linear_function_composition (f : ℝ → ℝ) :
  LinearFunction f → (∀ x, f (f x) = 4 * x + 6) →
  (∀ x, f x = 2 * x + 2) ∨ (∀ x, f x = -2 * x - 6) := by
  sorry

end NUMINAMATH_CALUDE_linear_function_composition_l310_31086


namespace NUMINAMATH_CALUDE_sqrt_D_irrational_l310_31098

theorem sqrt_D_irrational (x : ℝ) : 
  ∀ (y : ℝ), y ^ 2 ≠ 3 * (2 * x) ^ 2 + 3 * (2 * x + 1) ^ 2 + (4 * x + 1) ^ 2 :=
sorry

end NUMINAMATH_CALUDE_sqrt_D_irrational_l310_31098


namespace NUMINAMATH_CALUDE_det_A_zero_for_n_gt_five_l310_31076

def A (n : ℕ) : Matrix (Fin n) (Fin n) ℤ :=
  λ i j => (i.val^j.val + j.val^i.val) % 3

theorem det_A_zero_for_n_gt_five (n : ℕ) (h : n > 5) :
  Matrix.det (A n) = 0 :=
sorry

end NUMINAMATH_CALUDE_det_A_zero_for_n_gt_five_l310_31076


namespace NUMINAMATH_CALUDE_f_increasing_on_interval_l310_31014

noncomputable def f (x : ℝ) : ℝ := 2 * (Real.log x / Real.log (1/2))^2 - 2 * (Real.log x / Real.log (1/2)) + 1

theorem f_increasing_on_interval :
  StrictMonoOn f { x : ℝ | x ≥ Real.sqrt 2 / 2 } :=
sorry

end NUMINAMATH_CALUDE_f_increasing_on_interval_l310_31014


namespace NUMINAMATH_CALUDE_distance_between_intersection_points_max_sum_on_C₂_l310_31018

-- Define the curves C₁ and C₂ in polar coordinates
def C₁ (ρ θ : ℝ) : Prop := ρ * (Real.sin θ + Real.cos θ) = 1
def C₂ (ρ θ : ℝ) : Prop := ρ = 2 * Real.cos θ - 4 * Real.sin θ

-- Define the intersection points A and B
def intersection_points (A B : ℝ × ℝ) : Prop :=
  ∃ (ρ₁ θ₁ ρ₂ θ₂ : ℝ),
    C₁ ρ₁ θ₁ ∧ C₂ ρ₁ θ₁ ∧
    C₁ ρ₂ θ₂ ∧ C₂ ρ₂ θ₂ ∧
    A = (ρ₁ * Real.cos θ₁, ρ₁ * Real.sin θ₁) ∧
    B = (ρ₂ * Real.cos θ₂, ρ₂ * Real.sin θ₂) ∧
    A ≠ B

-- Theorem 1: Distance between intersection points
theorem distance_between_intersection_points
  (A B : ℝ × ℝ) (h : intersection_points A B) :
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 2 * Real.sqrt 3 :=
sorry

-- Define a point on C₂ in Cartesian coordinates
def point_on_C₂ (x y : ℝ) : Prop :=
  ∃ (ρ θ : ℝ), C₂ ρ θ ∧ x = ρ * Real.cos θ ∧ y = ρ * Real.sin θ

-- Theorem 2: Maximum value of x + y for points on C₂
theorem max_sum_on_C₂ :
  ∃ (M : ℝ), M = Real.sqrt 10 - 1 ∧
  (∀ x y, point_on_C₂ x y → x + y ≤ M) ∧
  (∃ x y, point_on_C₂ x y ∧ x + y = M) :=
sorry

end NUMINAMATH_CALUDE_distance_between_intersection_points_max_sum_on_C₂_l310_31018


namespace NUMINAMATH_CALUDE_circles_internally_tangent_l310_31015

/-- Represents a circle in 2D space --/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Determines if two circles are internally tangent --/
def are_internally_tangent (c1 c2 : Circle) : Prop :=
  let (x1, y1) := c1.center
  let (x2, y2) := c2.center
  let distance := Real.sqrt ((x2 - x1)^2 + (y2 - y1)^2)
  distance = abs (c2.radius - c1.radius)

/-- The first circle: x^2 + y^2 - 2x = 0 --/
def circle1 : Circle :=
  { center := (1, 0), radius := 1 }

/-- The second circle: x^2 + y^2 - 2x - 6y - 6 = 0 --/
def circle2 : Circle :=
  { center := (1, 3), radius := 4 }

/-- Theorem stating that the two given circles are internally tangent --/
theorem circles_internally_tangent : are_internally_tangent circle1 circle2 := by
  sorry

end NUMINAMATH_CALUDE_circles_internally_tangent_l310_31015


namespace NUMINAMATH_CALUDE_right_triangle_sine_cosine_l310_31088

theorem right_triangle_sine_cosine (P Q R : Real) (h1 : 3 * Real.sin P = 4 * Real.cos P) :
  Real.sin P = 4/5 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_sine_cosine_l310_31088


namespace NUMINAMATH_CALUDE_total_dolls_count_l310_31022

/-- The number of dolls owned by grandmother -/
def grandmother_dolls : ℕ := 50

/-- The number of dolls owned by Rene's sister -/
def sister_dolls : ℕ := grandmother_dolls + 2

/-- The number of dolls owned by Rene -/
def rene_dolls : ℕ := 3 * sister_dolls

/-- The total number of dolls owned by Rene, her sister, and their grandmother -/
def total_dolls : ℕ := rene_dolls + sister_dolls + grandmother_dolls

theorem total_dolls_count : total_dolls = 258 := by
  sorry

end NUMINAMATH_CALUDE_total_dolls_count_l310_31022


namespace NUMINAMATH_CALUDE_bridge_length_l310_31071

/-- The length of a bridge given train specifications and crossing time -/
theorem bridge_length (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) :
  train_length = 110 →
  train_speed_kmh = 45 →
  crossing_time = 30 →
  (train_speed_kmh * 1000 / 3600 * crossing_time) - train_length = 265 :=
by sorry

end NUMINAMATH_CALUDE_bridge_length_l310_31071
