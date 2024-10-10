import Mathlib

namespace seed_flower_probability_l2203_220323

theorem seed_flower_probability : ∀ (total_seeds small_seeds large_seeds : ℕ)
  (p_small_to_small p_large_to_large : ℝ),
  total_seeds = small_seeds + large_seeds →
  0 ≤ p_small_to_small ∧ p_small_to_small ≤ 1 →
  0 ≤ p_large_to_large ∧ p_large_to_large ≤ 1 →
  total_seeds = 10 →
  small_seeds = 6 →
  large_seeds = 4 →
  p_small_to_small = 0.9 →
  p_large_to_large = 0.8 →
  (small_seeds : ℝ) / (total_seeds : ℝ) * p_small_to_small +
  (large_seeds : ℝ) / (total_seeds : ℝ) * (1 - p_large_to_large) = 0.62 := by
  sorry

end seed_flower_probability_l2203_220323


namespace no_solution_equation_l2203_220306

theorem no_solution_equation : 
  ¬∃ (x : ℝ), x - 9 / (x - 4) = 4 - 9 / (x - 4) := by
  sorry

end no_solution_equation_l2203_220306


namespace f_odd_when_c_zero_f_unique_root_when_b_zero_c_pos_f_symmetric_about_0_c_f_more_than_two_roots_l2203_220353

-- Define the function f
def f (b c x : ℝ) : ℝ := x * |x| + b * x + c

-- Theorem 1: When c = 0, f(-x) = -f(x) for all x
theorem f_odd_when_c_zero (b : ℝ) :
  ∀ x, f b 0 (-x) = -f b 0 x := by sorry

-- Theorem 2: When b = 0 and c > 0, f(x) = 0 has exactly one real root
theorem f_unique_root_when_b_zero_c_pos (c : ℝ) (hc : c > 0) :
  ∃! x, f 0 c x = 0 := by sorry

-- Theorem 3: The graph of y = f(x) is symmetric about (0, c)
theorem f_symmetric_about_0_c (b c : ℝ) :
  ∀ x, f b c (-x) + f b c x = 2 * c := by sorry

-- Theorem 4: There exists a case where f(x) = 0 has more than two real roots
theorem f_more_than_two_roots :
  ∃ b c, ∃ x y z, x ≠ y ∧ y ≠ z ∧ x ≠ z ∧
    f b c x = 0 ∧ f b c y = 0 ∧ f b c z = 0 := by sorry

end f_odd_when_c_zero_f_unique_root_when_b_zero_c_pos_f_symmetric_about_0_c_f_more_than_two_roots_l2203_220353


namespace grid_arithmetic_progression_l2203_220313

def is_arithmetic_progression (a b c : ℚ) : Prop :=
  b - a = c - b

theorem grid_arithmetic_progression :
  ∀ x : ℚ,
  let pos_3_4 := 2*x - 103
  let pos_1_4 := 251 - 2*x
  let pos_1_3 := 2/3*(51 - 2*x)
  let pos_3_3 := x
  (is_arithmetic_progression pos_1_3 pos_3_3 pos_3_4 ∧
   is_arithmetic_progression pos_1_4 pos_3_3 pos_3_4) →
  x = 60 := by
sorry

end grid_arithmetic_progression_l2203_220313


namespace apple_distribution_l2203_220336

theorem apple_distribution (x : ℕ) (total_apples : ℕ) : 
  (total_apples = 3 * x + 8) →
  (total_apples > 5 * (x - 1) ∧ total_apples < 5 * x) →
  ((x = 5 ∧ total_apples = 23) ∨ (x = 6 ∧ total_apples = 26)) :=
by sorry

end apple_distribution_l2203_220336


namespace class_fund_solution_l2203_220322

/-- Represents the number of bills in a class fund -/
structure ClassFund where
  bills_10 : ℕ
  bills_20 : ℕ

/-- Calculates the total amount in the fund -/
def total_amount (fund : ClassFund) : ℕ :=
  10 * fund.bills_10 + 20 * fund.bills_20

theorem class_fund_solution :
  ∃ (fund : ClassFund),
    total_amount fund = 120 ∧
    fund.bills_10 = 2 * fund.bills_20 ∧
    fund.bills_20 = 3 := by
  sorry

end class_fund_solution_l2203_220322


namespace suzy_books_wednesday_morning_l2203_220349

/-- The number of books Suzy had at the end of Friday -/
def friday_end : ℕ := 80

/-- The number of books returned on Friday -/
def friday_returned : ℕ := 7

/-- The number of books checked out on Thursday -/
def thursday_checked_out : ℕ := 5

/-- The number of books returned on Thursday -/
def thursday_returned : ℕ := 23

/-- The number of books checked out on Wednesday -/
def wednesday_checked_out : ℕ := 43

/-- The number of books Suzy had on Wednesday morning -/
def wednesday_morning : ℕ := friday_end + friday_returned + thursday_checked_out - thursday_returned + wednesday_checked_out

theorem suzy_books_wednesday_morning : wednesday_morning = 98 := by
  sorry

end suzy_books_wednesday_morning_l2203_220349


namespace one_intersection_iff_tangent_l2203_220310

-- Define a line
def Line : Type := sorry

-- Define a conic curve
def ConicCurve : Type := sorry

-- Define the property of having only one intersection point
def hasOneIntersectionPoint (l : Line) (c : ConicCurve) : Prop := sorry

-- Define the property of being tangent
def isTangent (l : Line) (c : ConicCurve) : Prop := sorry

-- Theorem stating that having one intersection point is both sufficient and necessary for being tangent
theorem one_intersection_iff_tangent (l : Line) (c : ConicCurve) : 
  hasOneIntersectionPoint l c ↔ isTangent l c := by sorry

end one_intersection_iff_tangent_l2203_220310


namespace trajectory_and_minimum_distance_l2203_220311

-- Define the points M and N
def M : ℝ × ℝ := (4, 0)
def N : ℝ × ℝ := (1, 0)

-- Define the line l
def l (x y : ℝ) : ℝ := x + 2*y - 12

-- Define the condition for point P
def P_condition (x y : ℝ) : Prop :=
  let MP := (x - M.1, y - M.2)
  let MN := (N.1 - M.1, N.2 - M.2)
  let NP := (x - N.1, y - N.2)
  MN.1 * MP.1 + MN.2 * MP.2 = 6 * Real.sqrt (NP.1^2 + NP.2^2)

-- State the theorem
theorem trajectory_and_minimum_distance :
  ∃ (Q : ℝ × ℝ),
    (∀ (x y : ℝ), P_condition x y ↔ x^2/4 + y^2/3 = 1) ∧
    Q = (1, 3/2) ∧
    (∀ (P : ℝ × ℝ), P_condition P.1 P.2 →
      |l P.1 P.2| / Real.sqrt 5 ≥ 8/5) ∧
    |l Q.1 Q.2| / Real.sqrt 5 = 8/5 :=
  sorry

end trajectory_and_minimum_distance_l2203_220311


namespace fraction_equality_solution_l2203_220328

theorem fraction_equality_solution : ∃! x : ℝ, (4 + x) / (6 + x) = (1 + x) / (2 + x) := by
  sorry

end fraction_equality_solution_l2203_220328


namespace smallest_number_of_cubes_is_80_l2203_220393

/-- Represents the dimensions of a box -/
structure BoxDimensions where
  length : Nat
  width : Nat
  depth : Nat

/-- Calculates the smallest number of identical cubes needed to fill a box completely -/
def smallestNumberOfCubes (box : BoxDimensions) : Nat :=
  let cubeSideLength := Nat.gcd (Nat.gcd box.length box.width) box.depth
  (box.length / cubeSideLength) * (box.width / cubeSideLength) * (box.depth / cubeSideLength)

/-- Theorem stating that the smallest number of cubes to fill the given box is 80 -/
theorem smallest_number_of_cubes_is_80 :
  smallestNumberOfCubes ⟨30, 48, 12⟩ = 80 := by
  sorry

#eval smallestNumberOfCubes ⟨30, 48, 12⟩

end smallest_number_of_cubes_is_80_l2203_220393


namespace triangle_properties_l2203_220333

-- Define the triangle ABC
def Triangle (A B C : ℝ) := A + B + C = Real.pi

-- Define the conditions
def ConditionOne (A B C : ℝ) := A + B = 3 * C
def ConditionTwo (A B C : ℝ) := 2 * Real.sin (A - C) = Real.sin B
def ConditionThree := 5

-- Define the height function
def Height (A B C : ℝ) : ℝ := sorry

-- Theorem statement
theorem triangle_properties (A B C : ℝ) 
  (h1 : Triangle A B C) 
  (h2 : ConditionOne A B C) 
  (h3 : ConditionTwo A B C) :
  Real.sin A = 3 * Real.sqrt 10 / 10 ∧ 
  Height A B C = 6 := by sorry

end triangle_properties_l2203_220333


namespace parabola_circle_problem_l2203_220384

/-- Parabola in the Cartesian coordinate system -/
structure Parabola where
  p : ℝ
  eq : ℝ → ℝ → Prop
  h_p_pos : p > 0
  h_eq : ∀ x y, eq x y ↔ y^2 = 2*p*x

/-- Circle in the Cartesian coordinate system -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- The setup of the problem -/
structure ParabolaCircleSetup where
  C : Parabola
  Q : Circle
  h_Q_passes_O : Q.center.1^2 + Q.center.2^2 = Q.radius^2
  h_Q_passes_F : (Q.center.1 - C.p/2)^2 + Q.center.2^2 = Q.radius^2
  h_Q_center_directrix : Q.center.1 + C.p/2 = 3/2

/-- The theorem to be proved -/
theorem parabola_circle_problem (setup : ParabolaCircleSetup) :
  -- 1. The equation of parabola C is y^2 = 4x
  setup.C.p = 2 ∧
  -- 2. For any point M(t, 4) on C and chords MD and ME with MD ⊥ ME,
  --    the line DE passes through the fixed point (8, -4)
  ∀ t : ℝ, setup.C.eq t 4 →
    ∀ D E : ℝ × ℝ, setup.C.eq D.1 D.2 → setup.C.eq E.1 E.2 →
      (t - D.1) * (t - E.1) + (4 - D.2) * (4 - E.2) = 0 →
        ∃ m : ℝ, (D.1 = m * (D.2 + 4) + 8 ∧ E.1 = m * (E.2 + 4) + 8) ∨
                 (D.1 = m * (D.2 - 4) + 4 ∧ E.1 = m * (E.2 - 4) + 4) :=
by sorry

end parabola_circle_problem_l2203_220384


namespace sqrt_equation_solutions_l2203_220395

theorem sqrt_equation_solutions :
  ∀ x : ℚ, (Real.sqrt (9 * x - 4) + 16 / Real.sqrt (9 * x - 4) = 9) ↔ (x = 68/9 ∨ x = 5/9) :=
by sorry

end sqrt_equation_solutions_l2203_220395


namespace regular_polygon_sides_l2203_220392

theorem regular_polygon_sides (interior_angle : ℝ) (n : ℕ) :
  interior_angle = 144 →
  (n : ℝ) * (180 - interior_angle) = 360 →
  n = 10 := by
  sorry

end regular_polygon_sides_l2203_220392


namespace inequality_theorem_l2203_220303

theorem inequality_theorem (a b : ℝ) (ha : a < 0) (hb : b < 0) :
  (b^2 / a + a^2 / b) ≤ (a + b) := by
  sorry

end inequality_theorem_l2203_220303


namespace local_minimum_of_f_l2203_220378

def f (x : ℝ) := x^3 - 3*x

theorem local_minimum_of_f :
  ∃ δ > 0, ∀ x, |x - 1| < δ → f x ≥ f 1 :=
sorry

end local_minimum_of_f_l2203_220378


namespace equal_angles_necessary_not_sufficient_l2203_220385

-- Define a quadrilateral
structure Quadrilateral :=
  (vertices : Fin 4 → ℝ × ℝ)

-- Define a square
def is_square (q : Quadrilateral) : Prop :=
  sorry -- Definition of a square

-- Define the property of having equal interior angles
def has_equal_interior_angles (q : Quadrilateral) : Prop :=
  sorry -- Definition of equal interior angles

theorem equal_angles_necessary_not_sufficient :
  (∀ q : Quadrilateral, is_square q → has_equal_interior_angles q) ∧
  (∃ q : Quadrilateral, has_equal_interior_angles q ∧ ¬is_square q) :=
sorry

end equal_angles_necessary_not_sufficient_l2203_220385


namespace range_of_a_l2203_220308

def proposition_p (a : ℝ) : Prop :=
  ∀ x ∈ Set.Icc 1 2, x^2 - a ≥ 0

def proposition_q (a : ℝ) : Prop :=
  ∃ x : ℝ, x^2 + 2*a*x + 2 - a = 0

theorem range_of_a : 
  ∀ a : ℝ, proposition_p a ∧ proposition_q a → a ≤ -2 ∨ a = 1 :=
by sorry

end range_of_a_l2203_220308


namespace trig_system_relation_l2203_220361

/-- Given a system of trigonometric equations, prove the relationship between a, b, and c -/
theorem trig_system_relation (x y a b c : ℝ) 
  (h1 : Real.sin x + Real.sin y = 2 * a)
  (h2 : Real.cos x + Real.cos y = 2 * b)
  (h3 : Real.tan x + Real.tan y = 2 * c) :
  a * (b + a * c) = c * (a^2 + b^2)^2 := by
  sorry

end trig_system_relation_l2203_220361


namespace log_sum_equals_three_l2203_220335

-- Define the logarithm base 10 function
noncomputable def log10 (x : ℝ) : ℝ := Real.log x / Real.log 10

-- State the theorem
theorem log_sum_equals_three : log10 5 + log10 2 + 2 = 3 := by
  sorry

end log_sum_equals_three_l2203_220335


namespace solve_equation_l2203_220344

theorem solve_equation (x : ℚ) : 
  3 - 1 / (3 - 2 * x) = 2 / 3 * (1 / (3 - 2 * x)) → x = 2 / 3 := by
  sorry

end solve_equation_l2203_220344


namespace fraction_inequality_solution_set_l2203_220317

theorem fraction_inequality_solution_set (x : ℝ) :
  (x + 1) / (x + 2) < 0 ↔ -2 < x ∧ x < -1 :=
sorry

end fraction_inequality_solution_set_l2203_220317


namespace total_staff_weekdays_and_weekends_l2203_220316

def weekday_chefs : ℕ := 16
def weekday_waiters : ℕ := 16
def weekday_busboys : ℕ := 10
def weekday_hostesses : ℕ := 5

def weekend_additional_chefs : ℕ := 5
def weekend_additional_hostesses : ℕ := 2

def chef_leave_percentage : ℚ := 25 / 100
def waiter_leave_percentage : ℚ := 20 / 100
def busboy_leave_percentage : ℚ := 30 / 100
def hostess_leave_percentage : ℚ := 15 / 100

theorem total_staff_weekdays_and_weekends :
  let weekday_chefs_left := weekday_chefs - Int.floor (chef_leave_percentage * weekday_chefs)
  let weekday_waiters_left := weekday_waiters - Int.floor (waiter_leave_percentage * weekday_waiters)
  let weekday_busboys_left := weekday_busboys - Int.floor (busboy_leave_percentage * weekday_busboys)
  let weekday_hostesses_left := weekday_hostesses - Int.floor (hostess_leave_percentage * weekday_hostesses)
  
  let weekday_total := weekday_chefs_left + weekday_waiters_left + weekday_busboys_left + weekday_hostesses_left
  
  let weekend_chefs := weekday_chefs + weekend_additional_chefs
  let weekend_waiters := weekday_waiters_left
  let weekend_busboys := weekday_busboys_left
  let weekend_hostesses := weekday_hostesses + weekend_additional_hostesses
  
  let weekend_total := weekend_chefs + weekend_waiters + weekend_busboys + weekend_hostesses
  
  weekday_total + weekend_total = 84 := by
    sorry

end total_staff_weekdays_and_weekends_l2203_220316


namespace binomial_coefficient_equality_l2203_220370

theorem binomial_coefficient_equality (x : ℕ) : 
  Nat.choose 28 x = Nat.choose 28 (2 * x - 1) → x = 1 := by
  sorry

end binomial_coefficient_equality_l2203_220370


namespace bat_survey_result_l2203_220374

theorem bat_survey_result (total : ℕ) 
  (blind_percent : ℚ) (deaf_percent : ℚ) (deaf_count : ℕ) 
  (h1 : blind_percent = 784/1000) 
  (h2 : deaf_percent = 532/1000) 
  (h3 : deaf_count = 33) : total = 79 :=
by
  sorry

end bat_survey_result_l2203_220374


namespace smallest_norm_v_l2203_220364

open Real
open Vector

/-- Given a vector v such that ||v + (-2, 4)|| = 10, the smallest possible value of ||v|| is 10 - 2√5 -/
theorem smallest_norm_v (v : ℝ × ℝ) (h : ‖v + (-2, 4)‖ = 10) :
  ∀ w : ℝ × ℝ, ‖w + (-2, 4)‖ = 10 → ‖v‖ ≤ ‖w‖ ∧ ‖v‖ = 10 - 2 * Real.sqrt 5 :=
sorry

end smallest_norm_v_l2203_220364


namespace baker_weekday_hours_l2203_220340

/-- Represents the baker's baking schedule and output --/
structure BakingSchedule where
  loavesPerHourPerOven : ℕ
  numOvens : ℕ
  weekendHoursPerDay : ℕ
  totalLoavesIn3Weeks : ℕ

/-- Calculates the number of hours the baker bakes from Monday to Friday each week --/
def weekdayHoursPerWeek (schedule : BakingSchedule) : ℕ :=
  let loavesPerHour := schedule.loavesPerHourPerOven * schedule.numOvens
  let weekendHours := schedule.weekendHoursPerDay * 2  -- 2 weekend days
  let weekendLoavesPerWeek := loavesPerHour * weekendHours
  let weekdayLoavesIn3Weeks := schedule.totalLoavesIn3Weeks - (weekendLoavesPerWeek * 3)
  weekdayLoavesIn3Weeks / (loavesPerHour * 3)

/-- Theorem stating that given the baker's schedule, they bake for 25 hours on weekdays --/
theorem baker_weekday_hours (schedule : BakingSchedule)
  (h1 : schedule.loavesPerHourPerOven = 5)
  (h2 : schedule.numOvens = 4)
  (h3 : schedule.weekendHoursPerDay = 2)
  (h4 : schedule.totalLoavesIn3Weeks = 1740) :
  weekdayHoursPerWeek schedule = 25 := by
  sorry


end baker_weekday_hours_l2203_220340


namespace father_son_age_difference_l2203_220346

/-- Represents the ages of a father and son pair -/
structure FatherSonAges where
  father : ℕ
  son : ℕ

/-- The conditions of the problem -/
def satisfiesConditions (ages : FatherSonAges) : Prop :=
  ages.father = 44 ∧
  (ages.father + 4 = 2 * (ages.son + 4) + 20)

/-- The theorem to be proved -/
theorem father_son_age_difference (ages : FatherSonAges) 
  (h : satisfiesConditions ages) : 
  ages.father - 4 * ages.son = 4 := by
  sorry

end father_son_age_difference_l2203_220346


namespace delegation_selection_l2203_220302

theorem delegation_selection (n k : ℕ) (h1 : n = 12) (h2 : k = 3) : 
  Nat.choose n k = 220 := by
  sorry

end delegation_selection_l2203_220302


namespace john_reaches_floor_pushups_in_12_weeks_l2203_220305

/-- Represents the number of days John trains per week -/
def training_days_per_week : ℕ := 5

/-- Represents the number of push-up variations John needs to progress through -/
def num_variations : ℕ := 4

/-- Represents the number of reps John needs to reach before progressing to the next variation -/
def reps_to_progress : ℕ := 20

/-- Calculates the total number of days it takes John to progress through all variations -/
def total_training_days : ℕ := (num_variations - 1) * reps_to_progress

/-- Calculates the number of weeks it takes John to reach floor push-ups -/
def weeks_to_floor_pushups : ℕ := total_training_days / training_days_per_week

/-- Theorem stating that it takes John 12 weeks to reach floor push-ups -/
theorem john_reaches_floor_pushups_in_12_weeks : weeks_to_floor_pushups = 12 := by
  sorry

end john_reaches_floor_pushups_in_12_weeks_l2203_220305


namespace proposition_equivalence_l2203_220389

theorem proposition_equivalence (a : ℝ) : 
  (∀ x ∈ Set.Icc 1 3, x^2 - a ≤ 0) ↔ a ≥ 9 := by sorry

end proposition_equivalence_l2203_220389


namespace restaurant_theorem_l2203_220391

def restaurant_problem (expenditures : List ℝ) : Prop :=
  let n := 6
  let avg := (List.sum (List.take n expenditures)) / n
  let g_spent := avg - 5
  let h_spent := 2 * (avg - g_spent)
  let total_spent := (List.sum expenditures) + g_spent + h_spent
  expenditures.length = 8 ∧
  List.take n expenditures = [13, 17, 9, 15, 11, 20] ∧
  total_spent = 104.17

theorem restaurant_theorem (expenditures : List ℝ) :
  restaurant_problem expenditures :=
sorry

end restaurant_theorem_l2203_220391


namespace min_value_inequality_l2203_220383

theorem min_value_inequality (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (x + y + z) * ((1 / (x + y)) + (1 / (y + z)) + (1 / (z + x))) ≥ 9 / 2 := by
  sorry

end min_value_inequality_l2203_220383


namespace A_in_second_quadrant_l2203_220350

/-- A point in a 2D Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Definition of the second quadrant -/
def second_quadrant (p : Point) : Prop :=
  p.x < 0 ∧ p.y > 0

/-- The point A with coordinates (-3, 4) -/
def A : Point :=
  { x := -3, y := 4 }

/-- Theorem stating that point A is in the second quadrant -/
theorem A_in_second_quadrant : second_quadrant A := by
  sorry

end A_in_second_quadrant_l2203_220350


namespace smallest_z_magnitude_l2203_220329

theorem smallest_z_magnitude (z : ℂ) (h : Complex.abs (z - 9) + Complex.abs (z - 4 * Complex.I) = 15) :
  ∃ (w : ℂ), Complex.abs w ≤ Complex.abs z ∧ Complex.abs w = 36 / Real.sqrt 97 :=
sorry

end smallest_z_magnitude_l2203_220329


namespace cubic_factorization_l2203_220360

theorem cubic_factorization (x : ℝ) : x^3 - 9*x = x*(x+3)*(x-3) := by
  sorry

end cubic_factorization_l2203_220360


namespace dihedral_angle_distance_l2203_220330

/-- Given a dihedral angle φ and a point A on one of its faces with distance a from the edge,
    the distance from A to the plane of the other face is a * sin(φ). -/
theorem dihedral_angle_distance (φ : ℝ) (a : ℝ) :
  let distance_to_edge := a
  let distance_to_other_face := a * Real.sin φ
  distance_to_other_face = distance_to_edge * Real.sin φ := by
  sorry

end dihedral_angle_distance_l2203_220330


namespace set_equality_l2203_220397

-- Define the universal set I
def I : Set Nat := {1, 2, 3, 4, 5, 6}

-- Define set M
def M : Set Nat := {3, 4, 5}

-- Define set N
def N : Set Nat := {1, 3, 6}

-- Define the set we want to prove equal to (C_I M) ∩ (C_I N)
def target_set : Set Nat := {2, 7}

-- Theorem statement
theorem set_equality : 
  target_set = (I \ M) ∩ (I \ N) := by sorry

end set_equality_l2203_220397


namespace roots_sum_quotient_and_reciprocal_l2203_220363

theorem roots_sum_quotient_and_reciprocal (a b : ℝ) : 
  (a^2 + 10*a + 5 = 0) → 
  (b^2 + 10*b + 5 = 0) → 
  (a ≠ 0) → 
  (b ≠ 0) → 
  a/b + b/a = 18 := by sorry

end roots_sum_quotient_and_reciprocal_l2203_220363


namespace power_of_two_equality_l2203_220373

theorem power_of_two_equality (M : ℕ) : (32^3) * (16^3) = 2^M → M = 27 := by
  sorry

end power_of_two_equality_l2203_220373


namespace pairing_theorem_l2203_220358

/-- The number of ways to pair 2n points on a circle with n non-intersecting chords -/
def pairings (n : ℕ) : ℚ :=
  1 / (n + 1 : ℚ) * (Nat.choose (2 * n) n : ℚ)

/-- Theorem stating that the number of ways to pair 2n points on a circle
    with n non-intersecting chords is equal to (1 / (n+1)) * binomial(2n, n) -/
theorem pairing_theorem (n : ℕ) (h : n ≥ 1) :
  pairings n = 1 / (n + 1 : ℚ) * (Nat.choose (2 * n) n : ℚ) := by
  sorry

end pairing_theorem_l2203_220358


namespace orphan_house_donation_percentage_l2203_220365

def total_income : ℝ := 400000
def children_percentage : ℝ := 0.2
def num_children : ℕ := 3
def wife_percentage : ℝ := 0.25
def remaining_amount : ℝ := 60000
def final_amount : ℝ := 40000

theorem orphan_house_donation_percentage :
  let children_share := children_percentage * num_children * total_income
  let wife_share := wife_percentage * total_income
  let donation_amount := remaining_amount - final_amount
  (donation_amount / remaining_amount) * 100 = 100/3 :=
by sorry

end orphan_house_donation_percentage_l2203_220365


namespace decimal_to_fraction_l2203_220343

theorem decimal_to_fraction :
  (3.75 : ℚ) = 15 / 4 := by sorry

end decimal_to_fraction_l2203_220343


namespace quadratic_roots_condition_l2203_220351

theorem quadratic_roots_condition (m : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ x^2 + x - m = 0 ∧ y^2 + y - m = 0) → m > -1/4 := by
  sorry

end quadratic_roots_condition_l2203_220351


namespace jackpot_probability_6_45_100_l2203_220348

/-- Represents the lottery "6 out of 45" -/
structure Lottery :=
  (total_numbers : Nat)
  (numbers_to_choose : Nat)

/-- Represents a player's bet in the lottery -/
structure Bet :=
  (number_of_tickets : Nat)

/-- Calculate the number of combinations for choosing k items from n items -/
def choose (n k : Nat) : Nat :=
  Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

/-- Calculate the probability of hitting the jackpot -/
def jackpot_probability (l : Lottery) (b : Bet) : ℚ :=
  b.number_of_tickets / (choose l.total_numbers l.numbers_to_choose)

/-- Theorem: The probability of hitting the jackpot in a "6 out of 45" lottery with 100 unique tickets -/
theorem jackpot_probability_6_45_100 :
  let l : Lottery := ⟨45, 6⟩
  let b : Bet := ⟨100⟩
  jackpot_probability l b = 100 / 8145060 := by sorry

end jackpot_probability_6_45_100_l2203_220348


namespace checkout_lane_shoppers_l2203_220369

theorem checkout_lane_shoppers (total_shoppers : ℕ) (avoid_fraction : ℚ) : 
  total_shoppers = 480 →
  avoid_fraction = 5/8 →
  total_shoppers - (total_shoppers * avoid_fraction).floor = 180 :=
by
  sorry

end checkout_lane_shoppers_l2203_220369


namespace inverse_proportion_l2203_220309

/-- Given that x is inversely proportional to y, prove that if x = 5 when y = 15, 
    then x = 5/3 when y = 45 -/
theorem inverse_proportion (x y : ℝ) (k : ℝ) (h1 : x * y = k) 
    (h2 : 5 * 15 = k) : 
  5 / 3 * 45 = k := by
  sorry

end inverse_proportion_l2203_220309


namespace repeating_decimal_equals_fraction_l2203_220355

/-- The repeating decimal 0.37268̄ expressed as a fraction -/
def repeating_decimal : ℚ := 371896 / 99900

/-- The decimal representation of 0.37268̄ -/
def decimal_representation : ℚ := 37 / 100 + 268 / 99900

theorem repeating_decimal_equals_fraction : 
  repeating_decimal = decimal_representation := by sorry

end repeating_decimal_equals_fraction_l2203_220355


namespace willey_farm_problem_l2203_220331

/-- The Willey Farm Collective Problem -/
theorem willey_farm_problem (total_land : ℝ) (corn_cost : ℝ) (wheat_cost : ℝ) (available_capital : ℝ)
  (h1 : total_land = 4500)
  (h2 : corn_cost = 42)
  (h3 : wheat_cost = 35)
  (h4 : available_capital = 165200) :
  ∃ (wheat_acres : ℝ), wheat_acres = 3400 ∧
    wheat_acres ≥ 0 ∧
    wheat_acres ≤ total_land ∧
    ∃ (corn_acres : ℝ), corn_acres ≥ 0 ∧
      corn_acres + wheat_acres = total_land ∧
      corn_cost * corn_acres + wheat_cost * wheat_acres = available_capital :=
by sorry

end willey_farm_problem_l2203_220331


namespace softball_team_ratio_l2203_220376

/-- Represents a co-ed softball team with different skill levels -/
structure SoftballTeam where
  beginnerMen : ℕ
  beginnerWomen : ℕ
  intermediateMen : ℕ
  intermediateWomen : ℕ
  advancedMen : ℕ
  advancedWomen : ℕ

/-- Theorem stating the ratio of men to women on the softball team -/
theorem softball_team_ratio (team : SoftballTeam) : 
  team.beginnerMen = 2 ∧ 
  team.beginnerWomen = 4 ∧
  team.intermediateMen = 3 ∧
  team.intermediateWomen = 5 ∧
  team.advancedMen = 1 ∧
  team.advancedWomen = 3 →
  (team.beginnerMen + team.intermediateMen + team.advancedMen) * 2 = 
  (team.beginnerWomen + team.intermediateWomen + team.advancedWomen) := by
  sorry

#check softball_team_ratio

end softball_team_ratio_l2203_220376


namespace direct_proportion_increasing_iff_m_gt_two_l2203_220366

/-- A direct proportion function y = (m-2)x where y increases as x increases -/
def direct_proportion_increasing (m : ℝ) : Prop :=
  ∀ x₁ x₂ : ℝ, x₁ < x₂ → (m - 2) * x₁ < (m - 2) * x₂

theorem direct_proportion_increasing_iff_m_gt_two :
  ∀ m : ℝ, direct_proportion_increasing m ↔ m > 2 :=
by sorry

end direct_proportion_increasing_iff_m_gt_two_l2203_220366


namespace quadratic_max_value_l2203_220354

theorem quadratic_max_value :
  let f : ℝ → ℝ := fun z ↦ -4 * z^2 + 20 * z - 6
  ∃ (max : ℝ), max = 19 ∧ ∀ z, f z ≤ max :=
by sorry

end quadratic_max_value_l2203_220354


namespace triangle_properties_l2203_220372

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  A : ℝ
  B : ℝ
  C : ℝ
  a : ℝ
  b : ℝ
  c : ℝ

/-- The theorem states properties of a specific triangle -/
theorem triangle_properties (t : Triangle) 
  (h1 : 2 * (Real.cos ((t.C - t.A) / 2))^2 * Real.cos t.A - 
        Real.sin (t.C - t.A) * Real.sin t.A + 
        Real.cos (t.B + t.C) = 1/3)
  (h2 : t.c = 2 * Real.sqrt 2) : 
  Real.sin t.C = (2 * Real.sqrt 2) / 3 ∧ 
  ∃ (max_area : ℝ), max_area = 2 * Real.sqrt 2 ∧ 
    ∀ (area : ℝ), area = 1/2 * t.a * t.b * Real.sin t.C → area ≤ max_area :=
by sorry

end triangle_properties_l2203_220372


namespace count_perfect_square_factors_450_l2203_220368

/-- The number of perfect square factors of 450 -/
def perfect_square_factors_of_450 : ℕ :=
  (Finset.filter (fun n => n^2 ∣ 450) (Finset.range (450 + 1))).card

/-- Theorem: The number of perfect square factors of 450 is 4 -/
theorem count_perfect_square_factors_450 : perfect_square_factors_of_450 = 4 := by
  sorry

end count_perfect_square_factors_450_l2203_220368


namespace corn_spacing_theorem_l2203_220362

/-- Calculates the space required for each seed in a row of corn. -/
def space_per_seed (row_length_feet : ℕ) (seeds_per_row : ℕ) : ℕ :=
  (row_length_feet * 12) / seeds_per_row

/-- Theorem: Given a row length of 120 feet and 80 seeds per row, 
    the space required for each seed is 18 inches. -/
theorem corn_spacing_theorem : space_per_seed 120 80 = 18 := by
  sorry

end corn_spacing_theorem_l2203_220362


namespace complement_of_union_is_four_l2203_220399

def U : Set Nat := {1, 2, 3, 4}
def A : Set Nat := {1, 2}
def B : Set Nat := {2, 3}

theorem complement_of_union_is_four :
  (U \ (A ∪ B)) = {4} := by
  sorry

end complement_of_union_is_four_l2203_220399


namespace sqrt_meaningful_range_l2203_220318

theorem sqrt_meaningful_range (x : ℝ) : 
  (∃ y : ℝ, y ^ 2 = 2 - x) ↔ x ≤ 2 := by sorry

end sqrt_meaningful_range_l2203_220318


namespace function_bounds_l2203_220307

-- Define the functions F and G
def F (a b c x : ℝ) : ℝ := a * x^2 + b * x + c
def G (a b c x : ℝ) : ℝ := c * x^2 + b * x + a

-- State the theorem
theorem function_bounds (a b c : ℝ) 
  (h1 : |F a b c 0| ≤ 1)
  (h2 : |F a b c 1| ≤ 1)
  (h3 : |F a b c (-1)| ≤ 1) :
  (∀ x : ℝ, |x| ≤ 1 → |F a b c x| ≤ 5/4) ∧
  (∀ x : ℝ, |x| ≤ 1 → |G a b c x| ≤ 2) :=
by sorry

end function_bounds_l2203_220307


namespace average_weight_group_B_proof_l2203_220388

/-- The average weight of additional friends in Group B -/
def average_weight_group_B : ℝ := 141

theorem average_weight_group_B_proof
  (initial_group : ℕ) (additional_group : ℕ) (group_A : ℕ) (group_B : ℕ)
  (avg_weight_increase : ℝ) (avg_weight_gain_A : ℝ) (final_avg_weight : ℝ)
  (h1 : initial_group = 50)
  (h2 : additional_group = 40)
  (h3 : group_A = 20)
  (h4 : group_B = 20)
  (h5 : avg_weight_increase = 12)
  (h6 : avg_weight_gain_A = 15)
  (h7 : final_avg_weight = 46)
  (h8 : additional_group = group_A + group_B) :
  average_weight_group_B = 141 := by
  sorry

end average_weight_group_B_proof_l2203_220388


namespace rectangle_equal_diagonals_converse_is_false_contrapositive_is_true_l2203_220315

-- Define a quadrilateral
structure Quadrilateral :=
  (vertices : Fin 4 → ℝ × ℝ)

-- Define a rectangle
def is_rectangle (q : Quadrilateral) : Prop := sorry

-- Define equal diagonals
def has_equal_diagonals (q : Quadrilateral) : Prop := sorry

-- Theorem for the original proposition
theorem rectangle_equal_diagonals (q : Quadrilateral) :
  is_rectangle q → has_equal_diagonals q := sorry

-- Theorem for the converse (which is false)
theorem converse_is_false : ¬(∀ q : Quadrilateral, has_equal_diagonals q → is_rectangle q) := sorry

-- Theorem for the contrapositive (which is true)
theorem contrapositive_is_true :
  ∀ q : Quadrilateral, ¬has_equal_diagonals q → ¬is_rectangle q := sorry

end rectangle_equal_diagonals_converse_is_false_contrapositive_is_true_l2203_220315


namespace plains_total_area_l2203_220387

/-- The total area of two plains given their individual areas -/
def total_area (area_A area_B : ℝ) : ℝ := area_A + area_B

/-- Theorem: Given the conditions, the total area of both plains is 350 square miles -/
theorem plains_total_area :
  ∀ (area_A area_B : ℝ),
  area_B = 200 →
  area_A = area_B - 50 →
  total_area area_A area_B = 350 := by
sorry

end plains_total_area_l2203_220387


namespace perpendicular_vectors_k_value_l2203_220301

-- Define the vectors i and j
def i : ℝ × ℝ := (1, 0)
def j : ℝ × ℝ := (0, 1)

-- Define vectors a and b
def a : ℝ × ℝ := (2 * i.1 + 3 * j.1, 2 * i.2 + 3 * j.2)
def b (k : ℝ) : ℝ × ℝ := (k * i.1 - 4 * j.1, k * i.2 - 4 * j.2)

-- Define the dot product for 2D vectors
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

-- Define perpendicularity
def perpendicular (v w : ℝ × ℝ) : Prop := dot_product v w = 0

-- Theorem statement
theorem perpendicular_vectors_k_value :
  ∃ k : ℝ, perpendicular a (b k) ∧ k = 6 :=
sorry

end perpendicular_vectors_k_value_l2203_220301


namespace fourth_term_binomial_expansion_l2203_220342

theorem fourth_term_binomial_expansion 
  (a x : ℝ) (hx : x ≠ 0) :
  let binomial := (a / x^2 + x^2 / a)
  let fourth_term := Nat.choose 7 3 * (a / x^2)^(7 - 3) * (x^2 / a)^3
  fourth_term = 35 * a / x^2 := by
  sorry

end fourth_term_binomial_expansion_l2203_220342


namespace company_male_employees_l2203_220314

theorem company_male_employees (m f : ℕ) : 
  m / f = 7 / 8 →
  (m + 3) / f = 8 / 9 →
  m = 189 := by
sorry

end company_male_employees_l2203_220314


namespace bananas_per_box_l2203_220381

/-- Given 40 bananas and 10 boxes, prove that the number of bananas per box is 4. -/
theorem bananas_per_box (total_bananas : ℕ) (total_boxes : ℕ) (h1 : total_bananas = 40) (h2 : total_boxes = 10) :
  total_bananas / total_boxes = 4 := by
  sorry

end bananas_per_box_l2203_220381


namespace definite_integral_proofs_l2203_220320

theorem definite_integral_proofs :
  (∫ x in (0:ℝ)..1, x^2 - x) = -1/6 ∧
  (∫ x in (1:ℝ)..3, |x - 2|) = 2 ∧
  (∫ x in (0:ℝ)..1, Real.sqrt (1 - x^2)) = π/4 := by
  sorry

end definite_integral_proofs_l2203_220320


namespace center_coordinate_sum_l2203_220382

/-- The equation of a circle in the xy-plane -/
def CircleEquation (x y : ℝ) : Prop :=
  x^2 + y^2 = 4*x + 12*y - 39

/-- The center of a circle given by its equation -/
def CenterOfCircle (h k : ℝ) : Prop :=
  ∀ x y : ℝ, CircleEquation x y ↔ (x - h)^2 + (y - k)^2 = 1

/-- Theorem: The sum of coordinates of the center of the given circle is 8 -/
theorem center_coordinate_sum :
  ∃ h k : ℝ, CenterOfCircle h k ∧ h + k = 8 := by sorry

end center_coordinate_sum_l2203_220382


namespace unique_g_50_18_l2203_220327

def divisor_count (n : ℕ) : ℕ := (Nat.divisors n).card

def g₁ (n : ℕ) : ℕ := 3 * divisor_count n

def g (j n : ℕ) : ℕ :=
  match j with
  | 0 => n
  | j+1 => g₁ (g j n)

theorem unique_g_50_18 :
  ∃! (n : ℕ), n ≤ 25 ∧ g 50 n = 18 := by sorry

end unique_g_50_18_l2203_220327


namespace shoe_box_problem_l2203_220339

theorem shoe_box_problem (n : ℕ) (pairs : ℕ) (prob : ℚ) : 
  pairs = 7 →
  prob = 1 / 13 →
  prob = (pairs : ℚ) / (n.choose 2) →
  n = 14 :=
by sorry

end shoe_box_problem_l2203_220339


namespace john_drinks_42_quarts_per_week_l2203_220345

/-- The number of quarts John drinks in a week -/
def quarts_per_week (gallons_per_day : ℚ) (days_per_week : ℕ) (quarts_per_gallon : ℕ) : ℚ :=
  gallons_per_day * days_per_week * quarts_per_gallon

/-- Proof that John drinks 42 quarts of water in a week -/
theorem john_drinks_42_quarts_per_week :
  quarts_per_week (3/2) 7 4 = 42 := by
  sorry

end john_drinks_42_quarts_per_week_l2203_220345


namespace complement_union_problem_l2203_220390

def U : Finset Nat := {1, 2, 3, 4, 5}
def A : Finset Nat := {1, 3, 4}
def B : Finset Nat := {2, 4}

theorem complement_union_problem :
  (U \ A) ∪ B = {2, 4, 5} := by sorry

end complement_union_problem_l2203_220390


namespace annika_hiking_distance_l2203_220324

/-- Annika's hiking problem -/
theorem annika_hiking_distance 
  (rate : ℝ) -- Hiking rate in minutes per kilometer
  (initial_distance : ℝ) -- Initial distance hiked east in kilometers
  (total_time : ℝ) -- Total time available in minutes
  (h_rate : rate = 10) -- Hiking rate is 10 minutes per kilometer
  (h_initial : initial_distance = 2.5) -- Initial distance is 2.5 kilometers
  (h_time : total_time = 45) -- Total available time is 45 minutes
  : ∃ (total_east : ℝ), total_east = 3.5 ∧ 
    2 * (total_east - initial_distance) * rate + initial_distance * rate = total_time :=
by sorry

end annika_hiking_distance_l2203_220324


namespace parabola_midpoint_distance_squared_l2203_220332

/-- Given a parabola y = 3x^2 + 6x - 2 and two points C and D on it with the origin as their midpoint,
    the square of the distance between C and D is 740/3. -/
theorem parabola_midpoint_distance_squared :
  ∀ (C D : ℝ × ℝ),
  (∃ (x y : ℝ), C = (x, y) ∧ y = 3 * x^2 + 6 * x - 2) →
  (∃ (x y : ℝ), D = (x, y) ∧ y = 3 * x^2 + 6 * x - 2) →
  (0 : ℝ × ℝ) = ((C.1 + D.1) / 2, (C.2 + D.2) / 2) →
  (C.1 - D.1)^2 + (C.2 - D.2)^2 = 740 / 3 :=
by sorry

end parabola_midpoint_distance_squared_l2203_220332


namespace min_value_P_l2203_220352

theorem min_value_P (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : x^2 + y^2 + 1/x + 1/y = 27/4) : 
  ∀ (P : ℝ), P = 15/x - 3/(4*y) → P ≥ 6 :=
by sorry

end min_value_P_l2203_220352


namespace unique_positive_root_implies_a_less_than_neg_one_l2203_220396

/-- Given two functions f and g, if their difference has a unique positive root,
    then the parameter a in f must be less than -1 -/
theorem unique_positive_root_implies_a_less_than_neg_one
  (f g : ℝ → ℝ)
  (h : ∀ x : ℝ, f x = 2 * a * x^3 + 3)
  (k : ∀ x : ℝ, g x = 3 * x^2 + 2)
  (unique_root : ∃! x₀ : ℝ, x₀ > 0 ∧ f x₀ = g x₀) :
  a < -1 :=
sorry

end unique_positive_root_implies_a_less_than_neg_one_l2203_220396


namespace solve_for_y_l2203_220386

theorem solve_for_y (x y : ℝ) (h1 : x^2 + 3*x + 7 = y - 5) (h2 : x = -4) : y = 16 := by
  sorry

end solve_for_y_l2203_220386


namespace regular_hexagon_side_length_l2203_220337

/-- A regular hexagon with opposite sides 18 inches apart has side length 12√3 inches -/
theorem regular_hexagon_side_length (h : RegularHexagon) 
  (opposite_sides_distance : ℝ) (side_length : ℝ) : 
  opposite_sides_distance = 18 → side_length = 12 * Real.sqrt 3 := by
  sorry

#check regular_hexagon_side_length

end regular_hexagon_side_length_l2203_220337


namespace quadratic_equation_coefficients_l2203_220319

/-- Given a quadratic equation 3x² = -2x + 5, prove that it can be rewritten
    in the general form ax² + bx + c = 0 with specific coefficients. -/
theorem quadratic_equation_coefficients :
  ∃ (a b c : ℝ), 
    (∀ x, 3 * x^2 = -2 * x + 5) →
    (∀ x, a * x^2 + b * x + c = 0) ∧
    a = 3 ∧ b = 2 ∧ c = -5 := by
  sorry

end quadratic_equation_coefficients_l2203_220319


namespace least_possible_BC_l2203_220375

theorem least_possible_BC (AB AC DC BD BC : ℕ) : 
  AB = 7 → 
  AC = 15 → 
  DC = 11 → 
  BD = 25 → 
  BC > AC - AB → 
  BC > BD - DC → 
  BC ≥ 14 ∧ ∀ n : ℕ, (n ≥ 14 → n ≥ BC) → BC = 14 :=
by
  sorry

end least_possible_BC_l2203_220375


namespace quadratic_roots_theorem_l2203_220312

theorem quadratic_roots_theorem (k : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
    (∀ x : ℝ, x^2 + 2*k*x + k^2 = x + 1 ↔ x = x₁ ∨ x = x₂) ∧
    (3*x₁ - x₂)*(x₁ - 3*x₂) = 19) →
  k = 0 ∨ k = -3 := by
sorry

end quadratic_roots_theorem_l2203_220312


namespace direct_proportion_implies_m_zero_l2203_220347

/-- A function f is a direct proportion function if there exists a constant k such that f x = k * x for all x -/
def is_direct_proportion (f : ℝ → ℝ) : Prop :=
  ∃ k : ℝ, ∀ x : ℝ, f x = k * x

/-- The function y = -2x + m -/
def f (m : ℝ) (x : ℝ) : ℝ := -2 * x + m

theorem direct_proportion_implies_m_zero (m : ℝ) :
  is_direct_proportion (f m) → m = 0 := by
  sorry

end direct_proportion_implies_m_zero_l2203_220347


namespace P_no_real_roots_l2203_220398

/-- Recursive definition of the polynomial sequence P_n(x) -/
def P : ℕ → ℝ → ℝ
  | 0, x => 1
  | n + 1, x => x^(11 * (n + 1)) - P n x

/-- Theorem stating that P_n(x) has no real roots for all n ≥ 0 -/
theorem P_no_real_roots : ∀ (n : ℕ) (x : ℝ), P n x ≠ 0 := by
  sorry

end P_no_real_roots_l2203_220398


namespace hyperbolas_same_asymptotes_l2203_220394

/-- Given two hyperbolas x²/9 - y²/16 = 1 and y²/25 - x²/M = 1,
    prove that M = 225/16 for the hyperbolas to have the same asymptotes -/
theorem hyperbolas_same_asymptotes :
  ∀ M : ℝ,
  (∀ x y : ℝ, x^2 / 9 - y^2 / 16 = 1 ↔ y^2 / 25 - x^2 / M = 1) →
  (∀ k : ℝ, (∃ x y : ℝ, y = k * x ∧ x^2 / 9 - y^2 / 16 = 1) ↔
            (∃ x y : ℝ, y = k * x ∧ y^2 / 25 - x^2 / M = 1)) →
  M = 225 / 16 :=
by sorry

end hyperbolas_same_asymptotes_l2203_220394


namespace B_power_48_l2203_220341

def B : Matrix (Fin 3) (Fin 3) ℤ := !![0, 0, 0; 0, 0, 2; 0, -2, 0]

theorem B_power_48 : 
  B^48 = !![0, 0, 0; 0, 16^12, 0; 0, 0, 16^12] := by sorry

end B_power_48_l2203_220341


namespace trigonometric_identity_l2203_220334

theorem trigonometric_identity (a b : ℝ) (θ : ℝ) (h : a > 0) (k : b > 0) 
  (eq : (Real.sin θ)^6 / a + (Real.cos θ)^6 / b = 1 / (a + b)) :
  (Real.sin θ)^12 / a^2 + (Real.cos θ)^12 / b^2 = (a^4 + b^4) / (a + b)^6 := by
  sorry

end trigonometric_identity_l2203_220334


namespace units_digit_of_7_power_2024_l2203_220367

theorem units_digit_of_7_power_2024 : 7^2024 ≡ 1 [ZMOD 10] := by
  sorry

end units_digit_of_7_power_2024_l2203_220367


namespace domino_rearrangement_l2203_220357

/-- Represents a chessboard -/
structure Chessboard :=
  (size : Nat)
  (is_covered : Bool)
  (empty_corner : Nat × Nat)

/-- Represents a domino -/
structure Domino :=
  (length : Nat)
  (width : Nat)

/-- Checks if a given position is a corner of the chessboard -/
def is_corner (board : Chessboard) (pos : Nat × Nat) : Prop :=
  (pos.1 = 1 ∨ pos.1 = board.size) ∧ (pos.2 = 1 ∨ pos.2 = board.size)

/-- Main theorem statement -/
theorem domino_rearrangement 
  (board : Chessboard) 
  (domino : Domino) 
  (h1 : board.size = 9)
  (h2 : domino.length = 1 ∧ domino.width = 2)
  (h3 : board.is_covered = true)
  (h4 : is_corner board board.empty_corner) :
  ∀ (corner : Nat × Nat), is_corner board corner → 
  ∃ (new_board : Chessboard), 
    new_board.size = board.size ∧ 
    new_board.is_covered = true ∧ 
    new_board.empty_corner = corner :=
sorry

end domino_rearrangement_l2203_220357


namespace distribute_seven_books_four_friends_l2203_220321

/-- The number of ways to distribute n identical books among k friends, 
    where each friend must have at least one book -/
def distribute_books (n k : ℕ) : ℕ := sorry

/-- Theorem: Distributing 7 books among 4 friends results in 34 ways -/
theorem distribute_seven_books_four_friends : 
  distribute_books 7 4 = 34 := by sorry

end distribute_seven_books_four_friends_l2203_220321


namespace distance_to_place_l2203_220356

/-- Proves that the distance to a place is 144 km given the rowing speed, current speed, and total round trip time. -/
theorem distance_to_place (rowing_speed : ℝ) (current_speed : ℝ) (total_time : ℝ) :
  rowing_speed = 10 →
  current_speed = 2 →
  total_time = 30 →
  (total_time * (rowing_speed + current_speed) * (rowing_speed - current_speed)) / (2 * rowing_speed) = 144 := by
sorry

end distance_to_place_l2203_220356


namespace interval_necessary_not_sufficient_l2203_220359

theorem interval_necessary_not_sufficient :
  ¬(∀ x : ℝ, -1 ≤ x ∧ x ≤ 5 ↔ (x - 5) * (x + 1) < 0) ∧
  (∀ x : ℝ, (x - 5) * (x + 1) < 0 → -1 ≤ x ∧ x ≤ 5) :=
by sorry

end interval_necessary_not_sufficient_l2203_220359


namespace floor_sqrt_23_squared_l2203_220304

theorem floor_sqrt_23_squared : ⌊Real.sqrt 23⌋^2 = 16 := by sorry

end floor_sqrt_23_squared_l2203_220304


namespace name_calculation_result_l2203_220300

/-- Represents the alphabetical position of a letter (A=1, B=2, ..., Z=26) -/
def alphabeticalPosition (c : Char) : Nat :=
  match c with
  | 'A' => 1 | 'B' => 2 | 'C' => 3 | 'D' => 4 | 'E' => 5
  | 'F' => 6 | 'G' => 7 | 'H' => 8 | 'I' => 9 | 'J' => 10
  | 'K' => 11 | 'L' => 12 | 'M' => 13 | 'N' => 14 | 'O' => 15
  | 'P' => 16 | 'Q' => 17 | 'R' => 18 | 'S' => 19 | 'T' => 20
  | 'U' => 21 | 'V' => 22 | 'W' => 23 | 'X' => 24 | 'Y' => 25
  | 'Z' => 26
  | _ => 0

theorem name_calculation_result :
  let elida := "ELIDA"
  let adrianna := "ADRIANNA"
  let belinda := "BELINDA"

  let elida_sum := (elida.data.map alphabeticalPosition).sum
  let adrianna_sum := (adrianna.data.map alphabeticalPosition).sum
  let belinda_sum := (belinda.data.map alphabeticalPosition).sum

  let total_sum := elida_sum + adrianna_sum + belinda_sum
  let average := total_sum / 3

  elida.length = 5 →
  adrianna.length = 2 * elida.length - 2 →
  (average * 3 : ℕ) - elida_sum = 109 := by
  sorry

#check name_calculation_result

end name_calculation_result_l2203_220300


namespace complex_modulus_problem_l2203_220325

theorem complex_modulus_problem (a : ℝ) (h1 : a > 0) (h2 : Complex.abs (a + Complex.I) = 2) :
  a = Real.sqrt 3 := by
  sorry

end complex_modulus_problem_l2203_220325


namespace inequality_proof_l2203_220379

theorem inequality_proof (x : ℝ) (h1 : (3/2 : ℝ) ≤ x) (h2 : x ≤ 5) :
  2 * Real.sqrt (x + 1) + Real.sqrt (2 * x - 3) + Real.sqrt (15 - 3 * x) < 2 * Real.sqrt 19 := by
  sorry

end inequality_proof_l2203_220379


namespace absolute_value_complex_l2203_220377

theorem absolute_value_complex : Complex.abs (-1 + (2/3) * Complex.I) = Real.sqrt 13 / 3 := by
  sorry

end absolute_value_complex_l2203_220377


namespace opposite_face_is_U_l2203_220380

-- Define the faces of the cube
inductive Face : Type
  | P | Q | R | S | T | U

-- Define the property of being adjacent in the net
def adjacent_in_net : Face → Face → Prop :=
  sorry

-- Define the property of being opposite in the cube
def opposite_in_cube : Face → Face → Prop :=
  sorry

-- State the theorem
theorem opposite_face_is_U :
  (adjacent_in_net Face.P Face.Q) →
  (adjacent_in_net Face.P Face.R) →
  (adjacent_in_net Face.P Face.S) →
  (¬adjacent_in_net Face.P Face.T ∨ ¬adjacent_in_net Face.P Face.U) →
  opposite_in_cube Face.P Face.U :=
by
  sorry

end opposite_face_is_U_l2203_220380


namespace max_squares_covered_two_inch_card_l2203_220371

/-- Represents a square card -/
structure Card where
  side_length : ℝ

/-- Represents a checkerboard square -/
structure BoardSquare where
  side_length : ℝ

/-- Calculates the maximum number of board squares that can be covered by a card -/
def max_squares_covered (card : Card) (board_square : BoardSquare) : ℕ :=
  sorry

/-- Theorem stating the maximum number of squares covered by a 2-inch card on a board of 1-inch squares -/
theorem max_squares_covered_two_inch_card :
  let card := Card.mk 2
  let board_square := BoardSquare.mk 1
  max_squares_covered card board_square = 16 := by
    sorry

end max_squares_covered_two_inch_card_l2203_220371


namespace trapezoid_area_l2203_220338

-- Define the rectangle ABCD
def Rectangle (A B C D : Point) : Prop := sorry

-- Define the trapezoid EFBA
def Trapezoid (E F B A : Point) : Prop := sorry

-- Define the area function
def area (shape : Set Point) : ℝ := sorry

-- Define the points
variable (A B C D E F : Point)

-- State the theorem
theorem trapezoid_area 
  (h1 : Rectangle A B C D) 
  (h2 : area {A, B, C, D} = 20) 
  (h3 : Trapezoid E F B A) : 
  area {E, F, B, A} = 14 := by sorry

end trapezoid_area_l2203_220338


namespace ratio_of_segments_l2203_220326

/-- Given four points A, B, C, and D on a line in that order, with AB = 2, BC = 5, and AD = 14,
    prove that the ratio of AC to BD is 7/12. -/
theorem ratio_of_segments (A B C D : ℝ) : 
  (A < B) → (B < C) → (C < D) → 
  (B - A = 2) → (C - B = 5) → (D - A = 14) →
  (C - A) / (D - B) = 7 / 12 := by
sorry

end ratio_of_segments_l2203_220326
