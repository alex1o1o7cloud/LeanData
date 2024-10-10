import Mathlib

namespace five_twelve_thirteen_pythagorean_l1515_151586

/-- Definition of a Pythagorean triple -/
def is_pythagorean_triple (a b c : ℕ) : Prop :=
  a * a + b * b = c * c

/-- Proof that (5, 12, 13) is a Pythagorean triple -/
theorem five_twelve_thirteen_pythagorean : is_pythagorean_triple 5 12 13 := by
  sorry

end five_twelve_thirteen_pythagorean_l1515_151586


namespace M_values_l1515_151584

theorem M_values (a b : ℚ) (h : a * b ≠ 0) :
  let M := (2 * abs a) / a + (3 * b) / abs b
  M = 1 ∨ M = -1 ∨ M = 5 ∨ M = -5 :=
sorry

end M_values_l1515_151584


namespace soup_problem_solution_l1515_151581

/-- Represents the number of people a can of soup can feed -/
structure CanCapacity where
  adults : Nat
  children : Nat

/-- Represents the problem setup -/
structure SoupProblem where
  capacity : CanCapacity
  totalCans : Nat
  childrenFed : Nat

/-- Calculates the number of adults that can be fed with the remaining soup -/
def remainingAdults (problem : SoupProblem) : Nat :=
  let cansUsedForChildren := problem.childrenFed / problem.capacity.children
  let remainingCans := problem.totalCans - cansUsedForChildren
  remainingCans * problem.capacity.adults

/-- Proves that given the problem conditions, 16 adults can be fed with the remaining soup -/
theorem soup_problem_solution (problem : SoupProblem) 
  (h1 : problem.capacity = ⟨4, 6⟩) 
  (h2 : problem.totalCans = 8) 
  (h3 : problem.childrenFed = 24) : 
  remainingAdults problem = 16 := by
  sorry

#eval remainingAdults ⟨⟨4, 6⟩, 8, 24⟩

end soup_problem_solution_l1515_151581


namespace range_of_a_l1515_151523

-- Define the sets A and B
def A : Set ℝ := {x | x ≤ 1 ∨ x ≥ 3}
def B (a : ℝ) : Set ℝ := {x | a ≤ x ∧ x ≤ a + 1}

-- State the theorem
theorem range_of_a (a : ℝ) : (A ∩ B a = B a) → (a ≤ 0 ∨ a ≥ 3) := by
  sorry

end range_of_a_l1515_151523


namespace basketball_tournament_l1515_151597

/-- The number of teams in the basketball tournament --/
def n : ℕ := 12

/-- The total number of matches played in the tournament --/
def total_matches (n : ℕ) : ℕ := n * (n - 1)

/-- The total number of points distributed in the tournament --/
def total_points (n : ℕ) : ℕ := 2 * total_matches n

/-- The number of teams scoring 24 points --/
def a (n : ℕ) : ℤ := n * (n - 1) - 11 * n + 33

/-- The number of teams scoring 22 points --/
def b (n : ℕ) : ℤ := -n^2 + 12 * n - 36

theorem basketball_tournament :
  (∃ (winner : ℕ) (last1 last2 : ℕ),
    winner = 26 ∧ 
    last1 = 20 ∧ 
    last2 = 20 ∧ 
    winner + last1 + last2 + 24 * (a n) + 22 * (b n) = total_points n) ∧
  a n ≥ 0 ∧
  b n ≥ 0 ∧
  a n + b n = n - 3 :=
by sorry

end basketball_tournament_l1515_151597


namespace first_fun_friday_is_april_28_l1515_151521

/-- Represents a date in a calendar year -/
structure Date where
  month : Nat
  day : Nat

/-- Represents a day of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Returns true if the given date is a Friday -/
def isFriday (d : Date) (startDate : Date) (startDay : DayOfWeek) : Bool :=
  sorry

/-- Returns true if the given month has five Fridays -/
def hasFiveFridays (month : Nat) (year : Nat) (startDate : Date) (startDay : DayOfWeek) : Bool :=
  sorry

/-- Returns the date of the first Fun Friday after the given start date -/
def firstFunFriday (startDate : Date) (startDay : DayOfWeek) : Date :=
  sorry

theorem first_fun_friday_is_april_28 :
  let fiscalYearStart : Date := ⟨3, 1⟩
  let fiscalYearStartDay : DayOfWeek := DayOfWeek.Wednesday
  firstFunFriday fiscalYearStart fiscalYearStartDay = ⟨4, 28⟩ := by
  sorry

end first_fun_friday_is_april_28_l1515_151521


namespace maxine_purchase_l1515_151582

theorem maxine_purchase (x y z : ℕ) : 
  x + y + z = 40 ∧ 
  50 * x + 400 * y + 500 * z = 10000 →
  x = 40 ∧ y = 0 ∧ z = 0 := by
sorry

end maxine_purchase_l1515_151582


namespace floor_product_eq_sum_iff_in_solution_set_l1515_151520

/-- The floor function -/
noncomputable def floor (x : ℝ) : ℤ := ⌊x⌋

/-- The solution set for the equation [x] · [y] = x + y -/
def solution_set : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 = 2 ∧ p.2 = 2) ∨ (2 ≤ p.1 ∧ p.1 < 4 ∧ p.1 ≠ 3 ∧ p.2 = 6 - p.1)}

theorem floor_product_eq_sum_iff_in_solution_set (x y : ℝ) (hx : 0 < x) (hy : 0 < y) :
  (floor x) * (floor y) = x + y ↔ (x, y) ∈ solution_set := by sorry

end floor_product_eq_sum_iff_in_solution_set_l1515_151520


namespace divisibility_of_factorial_plus_one_l1515_151529

theorem divisibility_of_factorial_plus_one (p : ℕ) : 
  (Nat.Prime p → p ∣ (Nat.factorial (p - 1) + 1)) ∧
  (¬Nat.Prime p → ¬(p ∣ (Nat.factorial (p - 1) + 1))) :=
by sorry

end divisibility_of_factorial_plus_one_l1515_151529


namespace unique_intersection_l1515_151576

/-- The coefficient of x^2 in the quadratic equation -/
def b : ℚ := 49 / 16

/-- The quadratic function -/
def f (x : ℝ) : ℝ := b * x^2 + 5 * x + 2

/-- The linear function -/
def g (x : ℝ) : ℝ := -2 * x - 2

/-- The difference between the quadratic and linear functions -/
def h (x : ℝ) : ℝ := f x - g x

theorem unique_intersection :
  ∃! x, h x = 0 :=
sorry

end unique_intersection_l1515_151576


namespace sum_remainder_by_eight_l1515_151568

theorem sum_remainder_by_eight (n : ℤ) : (8 - n + (n + 5)) % 8 = 5 := by
  sorry

end sum_remainder_by_eight_l1515_151568


namespace exam_score_calculation_l1515_151560

theorem exam_score_calculation 
  (total_questions : ℕ) 
  (correct_answers : ℕ) 
  (total_score : ℕ) 
  (wrong_answer_penalty : ℕ) :
  total_questions = 75 →
  correct_answers = 40 →
  total_score = 125 →
  wrong_answer_penalty = 1 →
  ∃ (score_per_correct : ℕ),
    score_per_correct * correct_answers - 
    wrong_answer_penalty * (total_questions - correct_answers) = total_score ∧
    score_per_correct = 4 := by
  sorry

end exam_score_calculation_l1515_151560


namespace trisha_take_home_pay_l1515_151553

/-- Calculates the annual take-home pay for an hourly worker. -/
def annual_take_home_pay (hourly_rate : ℚ) (hours_per_week : ℕ) (weeks_per_year : ℕ) (withholding_rate : ℚ) : ℚ :=
  let gross_pay := hourly_rate * hours_per_week * weeks_per_year
  let withholding := withholding_rate * gross_pay
  gross_pay - withholding

/-- Proves that Trisha's annual take-home pay is $24,960 given the specified conditions. -/
theorem trisha_take_home_pay :
  annual_take_home_pay 15 40 52 (1/5) = 24960 := by
  sorry

#eval annual_take_home_pay 15 40 52 (1/5)

end trisha_take_home_pay_l1515_151553


namespace intersection_A_complement_B_l1515_151532

open Set

def A : Set ℝ := {x | x^2 - 2*x < 0}
def B : Set ℝ := {x | x > 1}

theorem intersection_A_complement_B : A ∩ (Bᶜ) = {x : ℝ | 0 < x ∧ x ≤ 1} := by sorry

end intersection_A_complement_B_l1515_151532


namespace one_carton_per_case_l1515_151509

/-- The number of cartons in a case -/
def cartons_per_case : ℕ := 1

/-- The number of boxes in each carton -/
def boxes_per_carton : ℕ := 1

/-- The number of paper clips in each box -/
def clips_per_box : ℕ := 500

/-- The total number of paper clips in two cases -/
def total_clips : ℕ := 1000

/-- Theorem stating that there is exactly one carton in a case -/
theorem one_carton_per_case :
  (∀ b : ℕ, b > 0 → 2 * cartons_per_case * b * clips_per_box = total_clips) →
  cartons_per_case = 1 :=
by sorry

end one_carton_per_case_l1515_151509


namespace expected_value_unfair_coin_l1515_151501

/-- The expected value of an unfair coin flip -/
theorem expected_value_unfair_coin : 
  let p_heads : ℚ := 2/3
  let p_tails : ℚ := 1/3
  let gain_heads : ℚ := 5
  let loss_tails : ℚ := -9
  p_heads * gain_heads + p_tails * loss_tails = 1/3 := by
  sorry

end expected_value_unfair_coin_l1515_151501


namespace smallest_start_for_five_odd_squares_l1515_151599

theorem smallest_start_for_five_odd_squares : 
  ∃ (n : ℕ), n > 0 ∧ 
  (∃ (s : Finset ℕ), s.card = 5 ∧ 
    (∀ m ∈ s, n ≤ m ∧ m ≤ 100 ∧ Odd m ∧ ∃ k : ℕ, m = k^2) ∧
    (∀ m : ℕ, n ≤ m ∧ m ≤ 100 ∧ Odd m ∧ (∃ k : ℕ, m = k^2) → m ∈ s)) ∧
  (∀ n' : ℕ, 0 < n' ∧ n' < n → 
    ¬∃ (s : Finset ℕ), s.card = 5 ∧ 
      (∀ m ∈ s, n' ≤ m ∧ m ≤ 100 ∧ Odd m ∧ ∃ k : ℕ, m = k^2) ∧
      (∀ m : ℕ, n' ≤ m ∧ m ≤ 100 ∧ Odd m ∧ (∃ k : ℕ, m = k^2) → m ∈ s)) :=
by
  sorry

end smallest_start_for_five_odd_squares_l1515_151599


namespace smallest_valid_arrangement_l1515_151556

def is_valid_arrangement (n : ℕ) : Prop :=
  (12 ∣ n) ∧ 
  (Finset.card (Finset.filter (λ d => d ∣ n) (Finset.range (n + 1))) = 13) ∧
  (∀ k : ℕ, 1 ≤ k → k ≤ 13 → ∃ m : ℕ, k ≤ m ∧ m < n ∧ m ∣ n)

theorem smallest_valid_arrangement :
  ∃ n : ℕ, is_valid_arrangement n ∧ ∀ m : ℕ, m < n → ¬is_valid_arrangement m :=
by sorry

end smallest_valid_arrangement_l1515_151556


namespace double_symmetry_quadratic_l1515_151569

/-- Given a quadratic function f(x) = ax^2 + bx + c, 
    this function returns the quadratic function 
    that results from applying y-axis symmetry 
    followed by x-axis symmetry -/
def double_symmetry (a b c : ℝ) : ℝ → ℝ := 
  fun x => -a * x^2 + b * x - c

/-- Theorem stating that the double symmetry operation 
    on a quadratic function results in the expected 
    transformed function -/
theorem double_symmetry_quadratic (a b c : ℝ) (ha : a ≠ 0) :
  ∀ x, double_symmetry a b c x = -(a * x^2 + b * x + c) :=
by
  sorry

#check double_symmetry_quadratic

end double_symmetry_quadratic_l1515_151569


namespace inequality_solution_set_l1515_151536

theorem inequality_solution_set (a b : ℝ) : 
  (∀ x, (a * x) / (x - 1) < 1 ↔ (x < b ∨ x > 3)) → 
  ((3 * a) / (3 - 1) = 1) → 
  a - b = -1/3 := by
sorry

end inequality_solution_set_l1515_151536


namespace equation_solution_l1515_151528

theorem equation_solution : 
  ∃ x : ℝ, (1 / (x + 5) + 1 / (x + 3) = 1 / (x + 6) + 1 / (x + 2)) ∧ (x = -4) := by
  sorry

end equation_solution_l1515_151528


namespace add_1857_minutes_to_noon_l1515_151591

/-- Represents a time of day --/
structure TimeOfDay where
  hours : Nat
  minutes : Nat
  is_pm : Bool

/-- Adds minutes to a given time --/
def addMinutes (t : TimeOfDay) (m : Nat) : TimeOfDay :=
  sorry

/-- Checks if two times are equal --/
def timeEqual (t1 t2 : TimeOfDay) : Prop :=
  t1.hours = t2.hours ∧ t1.minutes = t2.minutes ∧ t1.is_pm = t2.is_pm

theorem add_1857_minutes_to_noon :
  let noon := TimeOfDay.mk 12 0 true
  let result := TimeOfDay.mk 6 57 false
  timeEqual (addMinutes noon 1857) result := by
  sorry

end add_1857_minutes_to_noon_l1515_151591


namespace wendy_bags_proof_l1515_151596

/-- The number of points Wendy earns per bag of cans recycled -/
def points_per_bag : ℕ := 5

/-- The number of bags Wendy didn't recycle -/
def unrecycled_bags : ℕ := 2

/-- The total points Wendy would earn if she recycled all but 2 bags -/
def total_points : ℕ := 45

/-- The initial number of bags Wendy had -/
def initial_bags : ℕ := 11

theorem wendy_bags_proof :
  points_per_bag * (initial_bags - unrecycled_bags) = total_points :=
by sorry

end wendy_bags_proof_l1515_151596


namespace equation_solutions_l1515_151508

theorem equation_solutions :
  ∀ x : ℝ, (Real.sqrt ((3 + 2 * Real.sqrt 2) ^ x) + Real.sqrt ((3 - 2 * Real.sqrt 2) ^ x) = 6) ↔ (x = 2 ∨ x = -2) := by
  sorry

end equation_solutions_l1515_151508


namespace parametric_to_ordinary_equation_l1515_151530

noncomputable def x (θ : Real) : Real := |Real.sin (θ / 2) + Real.cos (θ / 2)|
noncomputable def y (θ : Real) : Real := 1 + Real.sin θ

theorem parametric_to_ordinary_equation :
  ∀ θ : Real, 0 ≤ θ ∧ θ < 2 * Real.pi →
  ∃ x_val y_val : Real,
    x θ = x_val ∧
    y θ = y_val ∧
    x_val ^ 2 = y_val ∧
    0 ≤ x_val ∧ x_val ≤ Real.sqrt 2 ∧
    0 ≤ y_val ∧ y_val ≤ 2 :=
by sorry

end parametric_to_ordinary_equation_l1515_151530


namespace deepak_investment_l1515_151551

/-- Proves that Deepak's investment is 15000 given the conditions of the business problem -/
theorem deepak_investment (total_profit : ℝ) (anand_investment : ℝ) (deepak_profit : ℝ) 
  (h1 : total_profit = 13800)
  (h2 : anand_investment = 22500)
  (h3 : deepak_profit = 5400) :
  ∃ deepak_investment : ℝ, 
    deepak_investment = 15000 ∧ 
    deepak_profit / total_profit = deepak_investment / (anand_investment + deepak_investment) :=
by
  sorry


end deepak_investment_l1515_151551


namespace instantaneous_velocity_at_4_seconds_l1515_151504

-- Define the equation of motion
def s (t : ℝ) : ℝ := 1 - t + t^2

-- Define the velocity function as the derivative of s
def v (t : ℝ) : ℝ := 2 * t - 1

-- Theorem statement
theorem instantaneous_velocity_at_4_seconds :
  v 4 = 7 := by
  sorry

end instantaneous_velocity_at_4_seconds_l1515_151504


namespace square_inequality_l1515_151534

theorem square_inequality (a b : ℝ) : a > |b| → a^2 > b^2 := by
  sorry

end square_inequality_l1515_151534


namespace total_winter_clothing_l1515_151594

-- Define a structure for a box of winter clothing
structure WinterClothingBox where
  scarves : Nat
  mittens : Nat
  hats : Nat

-- Define the contents of each box
def box1 : WinterClothingBox := ⟨2, 3, 1⟩
def box2 : WinterClothingBox := ⟨4, 2, 2⟩
def box3 : WinterClothingBox := ⟨1, 5, 3⟩
def box4 : WinterClothingBox := ⟨3, 4, 1⟩
def box5 : WinterClothingBox := ⟨5, 3, 2⟩
def box6 : WinterClothingBox := ⟨2, 6, 0⟩
def box7 : WinterClothingBox := ⟨4, 1, 3⟩
def box8 : WinterClothingBox := ⟨3, 2, 4⟩
def box9 : WinterClothingBox := ⟨1, 4, 5⟩

-- Define a function to count items in a box
def countItems (box : WinterClothingBox) : Nat :=
  box.scarves + box.mittens + box.hats

-- Theorem statement
theorem total_winter_clothing :
  countItems box1 + countItems box2 + countItems box3 +
  countItems box4 + countItems box5 + countItems box6 +
  countItems box7 + countItems box8 + countItems box9 = 76 := by
  sorry

end total_winter_clothing_l1515_151594


namespace trapezoid_median_length_l1515_151506

/-- Given a triangle and a trapezoid with the same height, prove that the median of the trapezoid is 18 inches when the triangle's base is 36 inches and their areas are equal. -/
theorem trapezoid_median_length (h : ℝ) (h_pos : h > 0) : 
  let triangle_base : ℝ := 36
  let triangle_area : ℝ := (1 / 2) * triangle_base * h
  let trapezoid_median : ℝ := triangle_area / h
  trapezoid_median = 18 := by sorry

end trapezoid_median_length_l1515_151506


namespace arithmetic_sequence_problem_l1515_151542

/-- An arithmetic sequence of integers -/
def arithmeticSeq (b : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, b (n + 1) = b n + d

/-- An increasing sequence -/
def increasingSeq (b : ℕ → ℤ) : Prop :=
  ∀ n m : ℕ, n < m → b n < b m

theorem arithmetic_sequence_problem (b : ℕ → ℤ) 
    (h_arith : arithmeticSeq b)
    (h_incr : increasingSeq b)
    (h_prod : b 4 * b 5 = 30) : 
  b 3 * b 6 = 28 := by
  sorry

end arithmetic_sequence_problem_l1515_151542


namespace total_rainfall_l1515_151500

def rainfall_problem (first_week : ℝ) (second_week : ℝ) : Prop :=
  (second_week = 1.5 * first_week) ∧
  (second_week = 15) ∧
  (first_week + second_week = 25)

theorem total_rainfall : ∃ (first_week second_week : ℝ), 
  rainfall_problem first_week second_week :=
by
  sorry

end total_rainfall_l1515_151500


namespace distinct_permutations_l1515_151502

def word1 := "NONNA"
def word2 := "MATHEMATICS"

def count_letter (w : String) (c : Char) : Nat :=
  w.toList.filter (· == c) |>.length

theorem distinct_permutations :
  (Nat.factorial 5 / Nat.factorial (count_letter word1 'N')) = 20 ∧
  (Nat.factorial 10 / (Nat.factorial (count_letter word2 'M') *
                       Nat.factorial (count_letter word2 'A') *
                       Nat.factorial (count_letter word2 'T'))) = 151200 := by
  sorry

#check distinct_permutations

end distinct_permutations_l1515_151502


namespace sine_characterization_l1515_151533

def IsPeriodic (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f (x + p) = f x

def IsSymmetricAbout (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x, f (2 * a - x) = f x

def IsIncreasingOn (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x → x < y → y ≤ b → f x < f y

theorem sine_characterization (f : ℝ → ℝ) 
  (h1 : IsPeriodic f π)
  (h2 : IsSymmetricAbout f (π/3))
  (h3 : IsIncreasingOn f (-π/6) (π/3)) :
  ∀ x, f x = Real.sin (2*x - π/6) := by
sorry

end sine_characterization_l1515_151533


namespace distance_P_to_x_axis_l1515_151543

/-- The distance from a point to the x-axis in a Cartesian coordinate system -/
def distanceToXAxis (y : ℝ) : ℝ := |y|

/-- Point P in the Cartesian coordinate system -/
def P : ℝ × ℝ := (4, -3)

/-- Theorem: The distance from point P(4, -3) to the x-axis is 3 -/
theorem distance_P_to_x_axis :
  distanceToXAxis P.2 = 3 := by
  sorry

end distance_P_to_x_axis_l1515_151543


namespace specific_ellipse_foci_distance_l1515_151595

/-- An ellipse with axes parallel to the coordinate axes -/
structure ParallelAxisEllipse where
  /-- The point where the ellipse is tangent to the x-axis -/
  x_tangent : ℝ × ℝ
  /-- The point where the ellipse is tangent to the y-axis -/
  y_tangent : ℝ × ℝ

/-- The distance between the foci of an ellipse -/
def foci_distance (e : ParallelAxisEllipse) : ℝ := sorry

/-- Theorem stating the distance between foci for a specific ellipse -/
theorem specific_ellipse_foci_distance :
  ∃ (e : ParallelAxisEllipse),
    e.x_tangent = (5, 0) ∧
    e.y_tangent = (0, 2) ∧
    foci_distance e = 2 * Real.sqrt 21 :=
  sorry

end specific_ellipse_foci_distance_l1515_151595


namespace track_width_l1515_151549

-- Define the radii of the two circles
variable (r₁ r₂ : ℝ)

-- Define the condition that the circles are concentric and r₁ > r₂
variable (h₁ : r₁ > r₂)

-- Define the condition that the difference in circumferences is 16π
variable (h₂ : 2 * Real.pi * r₁ - 2 * Real.pi * r₂ = 16 * Real.pi)

-- Theorem statement
theorem track_width (r₁ r₂ : ℝ) (h₁ : r₁ > r₂) (h₂ : 2 * Real.pi * r₁ - 2 * Real.pi * r₂ = 16 * Real.pi) :
  r₁ - r₂ = 8 := by
  sorry

end track_width_l1515_151549


namespace boys_count_is_sixty_l1515_151505

/-- Represents the number of boys in a group of 3 students -/
inductive GroupComposition
  | ThreeGirls
  | TwoGirlsOneBoy
  | OneGirlTwoBoys
  | ThreeBoys

/-- Represents the distribution of groups -/
structure GroupDistribution where
  total_groups : Nat
  one_boy_groups : Nat
  at_least_two_boys_groups : Nat
  three_boys_groups : Nat
  three_girls_groups : Nat

/-- Calculates the total number of boys given a group distribution -/
def count_boys (gd : GroupDistribution) : Nat :=
  gd.one_boy_groups + 2 * (gd.at_least_two_boys_groups - gd.three_boys_groups) + 3 * gd.three_boys_groups

/-- The main theorem to be proved -/
theorem boys_count_is_sixty (gd : GroupDistribution) 
  (h1 : gd.total_groups = 35)
  (h2 : gd.one_boy_groups = 10)
  (h3 : gd.at_least_two_boys_groups = 19)
  (h4 : gd.three_boys_groups = 2 * gd.three_girls_groups)
  (h5 : gd.three_girls_groups = gd.total_groups - gd.one_boy_groups - gd.at_least_two_boys_groups) :
  count_boys gd = 60 := by
  sorry

end boys_count_is_sixty_l1515_151505


namespace divisibility_by_3_and_2_l1515_151537

theorem divisibility_by_3_and_2 (n : ℕ) : 
  (3 ∣ n) → (2 ∣ n) → (6 ∣ n) := by
  sorry

end divisibility_by_3_and_2_l1515_151537


namespace central_cell_value_l1515_151545

theorem central_cell_value (a b c d e f g h i : ℝ) 
  (row_products : a * b * c = 10 ∧ d * e * f = 10 ∧ g * h * i = 10)
  (col_products : a * d * g = 10 ∧ b * e * h = 10 ∧ c * f * i = 10)
  (square_products : a * b * d * e = 3 ∧ b * c * e * f = 3 ∧ d * e * g * h = 3 ∧ e * f * h * i = 3) :
  e = 0.00081 := by
  sorry

end central_cell_value_l1515_151545


namespace inscribed_quadrilateral_equal_orthocenter_quadrilateral_l1515_151510

/-- A point in the Euclidean plane -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- A quadrilateral defined by four points -/
structure Quadrilateral :=
  (A : Point) (B : Point) (C : Point) (D : Point)

/-- Definition of an inscribed quadrilateral -/
def isInscribed (q : Quadrilateral) : Prop :=
  sorry

/-- Definition of an orthocenter of a triangle -/
def isOrthocenter (H : Point) (A B C : Point) : Prop :=
  sorry

/-- Definition of equality between quadrilaterals -/
def quadrilateralEqual (q1 q2 : Quadrilateral) : Prop :=
  sorry

/-- The main theorem -/
theorem inscribed_quadrilateral_equal_orthocenter_quadrilateral 
  (A₁ A₂ A₃ A₄ H₁ H₂ H₃ H₄ : Point) :
  isInscribed (Quadrilateral.mk A₁ A₂ A₃ A₄) →
  isOrthocenter H₁ A₂ A₃ A₄ →
  isOrthocenter H₂ A₁ A₃ A₄ →
  isOrthocenter H₃ A₁ A₂ A₄ →
  isOrthocenter H₄ A₁ A₂ A₃ →
  quadrilateralEqual 
    (Quadrilateral.mk A₁ A₂ A₃ A₄) 
    (Quadrilateral.mk H₁ H₂ H₃ H₄) :=
by
  sorry

end inscribed_quadrilateral_equal_orthocenter_quadrilateral_l1515_151510


namespace union_and_complement_when_a_is_one_intersection_equals_b_iff_a_in_range_l1515_151558

-- Define sets A and B
def A : Set ℝ := {x | x^2 - 2*x - 3 ≤ 0}
def B (a : ℝ) : Set ℝ := {x | a - 1 ≤ x ∧ x ≤ 2*a + 1}

-- Theorem for part 1
theorem union_and_complement_when_a_is_one :
  (A ∪ B 1 = {x | -1 ≤ x ∧ x ≤ 3}) ∧
  (Set.univ \ B 1 = {x | x < 0 ∨ x > 3}) := by sorry

-- Theorem for part 2
theorem intersection_equals_b_iff_a_in_range :
  ∀ a : ℝ, A ∩ B a = B a ↔ a ∈ Set.Ioi (-2) ∪ Set.Icc 0 1 := by sorry

end union_and_complement_when_a_is_one_intersection_equals_b_iff_a_in_range_l1515_151558


namespace min_value_sum_of_squares_l1515_151539

theorem min_value_sum_of_squares (x y z : ℝ) 
  (pos_x : 0 < x) (pos_y : 0 < y) (pos_z : 0 < z) 
  (sum_condition : x + y + z = 9) : 
  (x^2 + y^2)/(x + y) + (x^2 + z^2)/(x + z) + (y^2 + z^2)/(y + z) ≥ 9 := by
  sorry

end min_value_sum_of_squares_l1515_151539


namespace johnson_carter_tie_l1515_151574

/-- Represents the months of the baseball season --/
inductive Month
| March
| April
| May
| June
| July
| August

/-- Represents a player's home run data --/
structure PlayerData where
  monthly_hrs : Month → ℕ

def johnson_data : PlayerData :=
  ⟨λ m => match m with
    | Month.March => 2
    | Month.April => 12
    | Month.May => 18
    | Month.June => 0
    | Month.July => 0
    | Month.August => 12⟩

def carter_data : PlayerData :=
  ⟨λ m => match m with
    | Month.March => 0
    | Month.April => 4
    | Month.May => 8
    | Month.June => 22
    | Month.July => 10
    | Month.August => 0⟩

def total_hrs (player : PlayerData) : ℕ :=
  (player.monthly_hrs Month.March) +
  (player.monthly_hrs Month.April) +
  (player.monthly_hrs Month.May) +
  (player.monthly_hrs Month.June) +
  (player.monthly_hrs Month.July) +
  (player.monthly_hrs Month.August)

theorem johnson_carter_tie :
  total_hrs johnson_data = total_hrs carter_data :=
by sorry

end johnson_carter_tie_l1515_151574


namespace nabla_calculation_l1515_151565

def nabla (a b : ℕ) : ℕ := 3 + b^a

theorem nabla_calculation : nabla (nabla 2 3) 2 = 4099 := by
  sorry

end nabla_calculation_l1515_151565


namespace weekend_to_weekday_ratio_is_three_to_one_l1515_151535

/-- The number of episodes watched on a weekday -/
def weekday_episodes : ℕ := 8

/-- The total number of episodes watched in a week -/
def total_episodes : ℕ := 88

/-- The number of weekdays in a week -/
def weekdays : ℕ := 5

/-- The number of weekend days in a week -/
def weekend_days : ℕ := 2

/-- The ratio of episodes watched on a weekend day to episodes watched on a weekday -/
def weekend_to_weekday_ratio : ℚ :=
  (total_episodes - weekday_episodes * weekdays) / (weekend_days * weekday_episodes)

theorem weekend_to_weekday_ratio_is_three_to_one :
  weekend_to_weekday_ratio = 3 := by sorry

end weekend_to_weekday_ratio_is_three_to_one_l1515_151535


namespace hyperbola_eccentricity_l1515_151571

/-- Represents a hyperbola with semi-major axis a and semi-minor axis b -/
structure Hyperbola where
  a : ℝ
  b : ℝ
  h_pos_a : a > 0
  h_pos_b : b > 0

/-- The eccentricity of a hyperbola -/
def eccentricity (h : Hyperbola) : ℝ := sorry

/-- The right focus of the hyperbola -/
def right_focus (h : Hyperbola) : ℝ × ℝ := sorry

/-- The left vertex of the hyperbola -/
def left_vertex (h : Hyperbola) : ℝ × ℝ := sorry

/-- Predicate to check if a point lies on the circle with diameter between two other points -/
def lies_on_circle_diameter (p q r : ℝ × ℝ) : Prop := sorry

theorem hyperbola_eccentricity (h : Hyperbola) : 
  lies_on_circle_diameter (0, h.b) (left_vertex h) (right_focus h) →
  eccentricity h = (Real.sqrt 5 + 1) / 2 := by sorry

end hyperbola_eccentricity_l1515_151571


namespace sams_age_five_years_ago_l1515_151567

/-- Proves Sam's age 5 years ago given the conditions about John, Sam, and Ted's ages --/
theorem sams_age_five_years_ago (sam_current_age : ℕ) : 
  -- John is 3 times as old as Sam
  (3 * sam_current_age = 3 * sam_current_age) →
  -- In 15 years, John will be twice as old as Sam
  (3 * sam_current_age + 15 = 2 * (sam_current_age + 15)) →
  -- Ted is 5 years younger than Sam
  (sam_current_age - 5 = sam_current_age - 5) →
  -- In 15 years, Ted will be three-fourths the age of Sam
  ((sam_current_age - 5 + 15) * 4 = (sam_current_age + 15) * 3) →
  -- Sam's age 5 years ago was 10
  sam_current_age - 5 = 10 := by
  sorry

end sams_age_five_years_ago_l1515_151567


namespace license_plate_combinations_license_plate_count_l1515_151546

theorem license_plate_combinations : ℕ :=
  let letter_choices : ℕ := 26
  let digit_choices : ℕ := 10
  let letter_positions : ℕ := 3
  let digit_positions : ℕ := 4
  letter_choices ^ letter_positions * digit_choices ^ digit_positions

theorem license_plate_count : license_plate_combinations = 175760000 := by
  sorry

end license_plate_combinations_license_plate_count_l1515_151546


namespace inequality_solution_set_l1515_151583

theorem inequality_solution_set (m : ℝ) :
  let S := {x : ℝ | x^2 + (m - 1) * x - m > 0}
  (m = -1 → S = {x : ℝ | x ≠ 1}) ∧
  (m > -1 → S = {x : ℝ | x < -m ∨ x > 1}) ∧
  (m < -1 → S = {x : ℝ | x < 1 ∨ x > -m}) :=
by sorry

end inequality_solution_set_l1515_151583


namespace intersection_A_B_l1515_151559

def A : Set ℝ := {x | x^2 - 2*x - 3 < 0}

def B : Set ℝ := {-1, 0, 1, 2, 3}

theorem intersection_A_B : A ∩ B = {0, 1, 2} := by sorry

end intersection_A_B_l1515_151559


namespace ash_cloud_ratio_l1515_151575

/-- Given a volcanic eruption where ashes are shot into the sky, this theorem proves
    the ratio of the ash cloud's diameter to the eruption height. -/
theorem ash_cloud_ratio (eruption_height : ℝ) (cloud_radius : ℝ) 
    (h1 : eruption_height = 300)
    (h2 : cloud_radius = 2700) : 
    (2 * cloud_radius) / eruption_height = 18 := by
  sorry

end ash_cloud_ratio_l1515_151575


namespace imoProof_l1515_151541

theorem imoProof (d : ℕ) (h1 : d ≠ 2) (h2 : d ≠ 5) (h3 : d ≠ 13) (h4 : d > 0) : 
  ∃ (a b : ℕ), a ∈ ({2, 5, 13, d} : Set ℕ) ∧ 
               b ∈ ({2, 5, 13, d} : Set ℕ) ∧ 
               a ≠ b ∧ 
               ¬∃ (k : ℕ), a * b - 1 = k * k :=
by sorry

end imoProof_l1515_151541


namespace polynomial_division_l1515_151527

theorem polynomial_division (x : ℂ) : 
  ∃! (a : ℤ), ∃ (p : ℂ → ℂ), (x^2 - x + a) * p x = x^15 + x^2 + 100 := by
  sorry

end polynomial_division_l1515_151527


namespace incorrect_copy_difference_l1515_151547

theorem incorrect_copy_difference (square : ℝ) : 
  let x := 4 * (square - 3)
  let y := 4 * square - 3
  x - y = -9 := by sorry

end incorrect_copy_difference_l1515_151547


namespace similar_triangles_leg_sum_l1515_151517

theorem similar_triangles_leg_sum (area_small area_large hyp_small : ℝ) 
  (h1 : area_small = 10)
  (h2 : area_large = 250)
  (h3 : hyp_small = 13)
  (h4 : area_small > 0)
  (h5 : area_large > 0)
  (h6 : hyp_small > 0) :
  ∃ (leg1_small leg2_small leg1_large leg2_large : ℝ),
    leg1_small^2 + leg2_small^2 = hyp_small^2 ∧
    leg1_small * leg2_small / 2 = area_small ∧
    leg1_large^2 + leg2_large^2 = (hyp_small * (area_large / area_small).sqrt)^2 ∧
    leg1_large * leg2_large / 2 = area_large ∧
    leg1_large + leg2_large = 35 := by
sorry


end similar_triangles_leg_sum_l1515_151517


namespace lines_parallel_iff_same_slope_diff_intercept_l1515_151526

/-- Two lines in the form y = kx + l are parallel if and only if 
    they have the same slope but different y-intercepts -/
theorem lines_parallel_iff_same_slope_diff_intercept 
  (k₁ k₂ l₁ l₂ : ℝ) : 
  (∀ x y : ℝ, y = k₁ * x + l₁ ↔ y = k₂ * x + l₂) ↔ 
  (k₁ = k₂ ∧ l₁ ≠ l₂) :=
by sorry

end lines_parallel_iff_same_slope_diff_intercept_l1515_151526


namespace matilda_age_l1515_151598

/-- Given the ages of Louis, Jerica, and Matilda, prove Matilda's age -/
theorem matilda_age (louis_age jerica_age matilda_age : ℕ) : 
  louis_age = 14 →
  jerica_age = 2 * louis_age →
  matilda_age = jerica_age + 7 →
  matilda_age = 35 := by
sorry

end matilda_age_l1515_151598


namespace pyramid_volume_l1515_151578

/-- The volume of a pyramid with a square base and given dimensions -/
theorem pyramid_volume (base_side : ℝ) (edge_length : ℝ) (h : base_side = 10 ∧ edge_length = 17) :
  (1 / 3 : ℝ) * base_side ^ 2 * Real.sqrt (edge_length ^ 2 - (base_side ^ 2 / 2)) = 
    (100 * Real.sqrt 239) / 3 := by
  sorry

end pyramid_volume_l1515_151578


namespace sum_even_numbers_1_to_200_l1515_151555

/- Define the sum of even numbers from 1 to n -/
def sumEvenNumbers (n : ℕ) : ℕ :=
  (n / 2) * (2 + n)

/- Theorem statement -/
theorem sum_even_numbers_1_to_200 : sumEvenNumbers 200 = 10100 := by
  sorry

end sum_even_numbers_1_to_200_l1515_151555


namespace symmetric_point_about_x_axis_l1515_151564

/-- Given a point P with coordinates (-1, 2), its symmetric point about the x-axis has coordinates (-1, -2) -/
theorem symmetric_point_about_x_axis :
  let P : ℝ × ℝ := (-1, 2)
  let symmetric_point (p : ℝ × ℝ) : ℝ × ℝ := (p.1, -p.2)
  symmetric_point P = (-1, -2) := by sorry

end symmetric_point_about_x_axis_l1515_151564


namespace min_trips_for_field_trip_l1515_151585

/-- The minimum number of trips required to transport all students -/
def min_trips (total_students : ℕ) (num_buses : ℕ) (bus_capacity : ℕ) : ℕ :=
  (total_students + num_buses * bus_capacity - 1) / (num_buses * bus_capacity)

theorem min_trips_for_field_trip :
  min_trips 520 5 45 = 3 := by
  sorry

end min_trips_for_field_trip_l1515_151585


namespace chess_tournament_participants_l1515_151590

theorem chess_tournament_participants (n : ℕ) (h : n > 0) : 
  (n * (n - 1)) / 2 = 120 → n = 16 := by sorry

end chess_tournament_participants_l1515_151590


namespace digit_sum_problem_l1515_151563

/-- Given six unique digits from 2 to 7, prove that if their sums along specific lines total 66, then B must be 4. -/
theorem digit_sum_problem (A B C D E F : ℕ) : 
  A ∈ ({2, 3, 4, 5, 6, 7} : Set ℕ) →
  B ∈ ({2, 3, 4, 5, 6, 7} : Set ℕ) →
  C ∈ ({2, 3, 4, 5, 6, 7} : Set ℕ) →
  D ∈ ({2, 3, 4, 5, 6, 7} : Set ℕ) →
  E ∈ ({2, 3, 4, 5, 6, 7} : Set ℕ) →
  F ∈ ({2, 3, 4, 5, 6, 7} : Set ℕ) →
  A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ A ≠ E ∧ A ≠ F ∧
  B ≠ C ∧ B ≠ D ∧ B ≠ E ∧ B ≠ F ∧
  C ≠ D ∧ C ≠ E ∧ C ≠ F ∧
  D ≠ E ∧ D ≠ F ∧
  E ≠ F →
  (A + B + C) + (A + B + E + F) + (C + D + E) + (B + D + F) + (C + F) = 66 →
  B = 4 := by
sorry

end digit_sum_problem_l1515_151563


namespace bed_frame_cost_l1515_151592

theorem bed_frame_cost (bed_price : ℝ) (total_price : ℝ) (discount_rate : ℝ) (final_price : ℝ) :
  bed_price = 10 * total_price →
  discount_rate = 0.2 →
  final_price = (1 - discount_rate) * (bed_price + total_price) →
  final_price = 660 →
  total_price = 75 := by
sorry

end bed_frame_cost_l1515_151592


namespace multiset_permutations_eq_1680_l1515_151514

/-- The number of permutations of a multiset with 9 elements, where there are 3 elements of each of 3 types -/
def multiset_permutations : ℕ :=
  Nat.factorial 9 / (Nat.factorial 3 * Nat.factorial 3 * Nat.factorial 3)

/-- Theorem stating that the number of permutations of the described multiset is 1680 -/
theorem multiset_permutations_eq_1680 : multiset_permutations = 1680 := by
  sorry

end multiset_permutations_eq_1680_l1515_151514


namespace lana_morning_muffins_l1515_151507

/-- Proves that Lana sold 12 muffins in the morning given the conditions of the bake sale -/
theorem lana_morning_muffins (total_goal : ℕ) (afternoon_sales : ℕ) (remaining : ℕ) 
  (h1 : total_goal = 20)
  (h2 : afternoon_sales = 4)
  (h3 : remaining = 4) :
  total_goal = afternoon_sales + remaining + 12 := by
  sorry

end lana_morning_muffins_l1515_151507


namespace workers_combined_rate_l1515_151548

/-- The fraction of a job two workers can complete together in one day, 
    given their individual completion times. -/
def combined_work_rate (time_a time_b : ℚ) : ℚ :=
  1 / time_a + 1 / time_b

/-- Theorem: Two workers, where one takes 18 days and the other takes half that time,
    can complete 1/6 of the job in one day when working together. -/
theorem workers_combined_rate : 
  combined_work_rate 18 9 = 1/6 := by
  sorry

end workers_combined_rate_l1515_151548


namespace even_function_sum_l1515_151538

def f (a b x : ℝ) : ℝ := a * x^2 + b * x

theorem even_function_sum (a b : ℝ) :
  (∀ x ∈ Set.Icc (a - 1) (2 * a), f a b x = f a b (-x)) →
  a + b = 1/3 :=
by sorry

end even_function_sum_l1515_151538


namespace intersection_properties_y₁_gt_y₂_l1515_151572

/-- The quadratic function y₁ -/
def y₁ (b : ℝ) (x : ℝ) : ℝ := x^2 + b*x - 3

/-- The linear function y₂ -/
def y₂ (x : ℝ) : ℝ := x + 1

/-- Theorem stating the properties of the intersection points and the resulting quadratic function -/
theorem intersection_properties :
  ∀ b m : ℝ,
  (y₁ b (-1) = y₂ (-1)) →
  (y₁ b 4 = y₂ 4) →
  (y₁ b (-1) = 0) →
  (y₁ b 4 = m) →
  (b = -2 ∧ m = 5) :=
sorry

/-- Theorem stating when y₁ > y₂ -/
theorem y₁_gt_y₂ :
  ∀ x : ℝ,
  (y₁ (-2) x > y₂ x) ↔ (x < -1 ∨ x > 4) :=
sorry

end intersection_properties_y₁_gt_y₂_l1515_151572


namespace horse_cloth_problem_l1515_151544

/-- Represents the system of equations for the horse and cloth problem -/
def horse_cloth_system (m n : ℚ) : Prop :=
  m + n = 100 ∧ 3 * m + n / 3 = 100

/-- The horse and cloth problem statement -/
theorem horse_cloth_problem :
  ∃ m n : ℚ, 
    m ≥ 0 ∧ n ≥ 0 ∧  -- Ensuring non-negative numbers of horses
    horse_cloth_system m n :=
by sorry

end horse_cloth_problem_l1515_151544


namespace integer_solutions_quadratic_equation_l1515_151552

theorem integer_solutions_quadratic_equation :
  ∀ x y : ℤ, x + y = x^2 - x*y + y^2 ↔ 
    (x = 0 ∧ y = 0) ∨ 
    (x = 0 ∧ y = 1) ∨ 
    (x = 1 ∧ y = 0) ∨ 
    (x = 1 ∧ y = 2) ∨ 
    (x = 2 ∧ y = 1) ∨ 
    (x = 2 ∧ y = 2) := by
  sorry

end integer_solutions_quadratic_equation_l1515_151552


namespace derivative_fifth_root_cube_l1515_151580

theorem derivative_fifth_root_cube (x : ℝ) (h : x ≠ 0) :
  deriv (λ x => x^(3/5)) x = 3 / (5 * x^(2/5)) :=
sorry

end derivative_fifth_root_cube_l1515_151580


namespace binary_representation_of_37_l1515_151531

/-- Converts a natural number to its binary representation as a list of booleans -/
def toBinary (n : ℕ) : List Bool :=
  if n = 0 then [false] else
  let rec toBinaryAux (m : ℕ) : List Bool :=
    if m = 0 then [] else (m % 2 = 1) :: toBinaryAux (m / 2)
  toBinaryAux n

/-- The binary representation of 37 -/
def binary37 : List Bool := [true, false, true, false, false, true]

/-- Theorem stating that the binary representation of 37 is [true, false, true, false, false, true] -/
theorem binary_representation_of_37 : toBinary 37 = binary37 := by
  sorry

end binary_representation_of_37_l1515_151531


namespace amys_tickets_proof_l1515_151577

/-- The total number of tickets Amy has after buying more at the fair -/
def amys_total_tickets (initial : ℕ) (additional : ℕ) : ℕ :=
  initial + additional

/-- Theorem stating that Amy's total tickets is 54 given her initial and additional tickets -/
theorem amys_tickets_proof :
  amys_total_tickets 33 21 = 54 := by
  sorry

end amys_tickets_proof_l1515_151577


namespace length_PQ_is_4_l1515_151561

-- Define the semicircle (C)
def semicircle (x y : ℝ) : Prop :=
  (x - 1)^2 + y^2 = 1 ∧ 0 ≤ y ∧ y ≤ 1

-- Define the polar equation of line (l)
def line_l (ρ θ : ℝ) : Prop :=
  ρ * (Real.sin θ + Real.sqrt 3 * Real.cos θ) = 5 * Real.sqrt 3

-- Define the ray OM
def ray_OM (θ : ℝ) : Prop :=
  θ = Real.pi / 3

-- Define the point P as the intersection of semicircle (C) and ray OM
def point_P (ρ θ : ℝ) : Prop :=
  ρ = 2 * Real.cos θ ∧ ray_OM θ

-- Define the point Q as the intersection of line (l) and ray OM
def point_Q (ρ θ : ℝ) : Prop :=
  line_l ρ θ ∧ ray_OM θ

-- Theorem statement
theorem length_PQ_is_4 :
  ∀ (ρ_P θ_P ρ_Q θ_Q : ℝ),
    point_P ρ_P θ_P →
    point_Q ρ_Q θ_Q →
    |ρ_P - ρ_Q| = 4 :=
sorry

end length_PQ_is_4_l1515_151561


namespace cube_volume_from_lateral_surface_area_l1515_151522

theorem cube_volume_from_lateral_surface_area :
  ∀ (lateral_surface_area : ℝ) (volume : ℝ),
  lateral_surface_area = 100 →
  volume = (lateral_surface_area / 4) ^ (3/2) →
  volume = 125 := by
sorry

end cube_volume_from_lateral_surface_area_l1515_151522


namespace perpendicular_line_equation_l1515_151525

-- Define a point in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Define a line in the form ax + by + c = 0
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define the problem
theorem perpendicular_line_equation (A B C : ℝ) (P₀ : Point2D) :
  let L₁ : Line := { a := A, b := B, c := C }
  let L₂ : Line := { a := B, b := -A, c := -B * P₀.x + A * P₀.y }
  (∀ (x y : ℝ), A * x + B * y + C = 0 → L₁.a * x + L₁.b * y + L₁.c = 0) →
  (L₂.a * P₀.x + L₂.b * P₀.y + L₂.c = 0) →
  (∀ (x y : ℝ), B * x - A * y - B * P₀.x + A * P₀.y = 0 ↔ L₂.a * x + L₂.b * y + L₂.c = 0) :=
by sorry

end perpendicular_line_equation_l1515_151525


namespace perpendicular_line_equation_l1515_151587

/-- A line in 2D space represented by the equation ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if a point lies on a line -/
def pointOnLine (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- Check if two lines are perpendicular -/
def perpendicular (l1 l2 : Line) : Prop :=
  l1.a * l2.a + l1.b * l2.b = 0

theorem perpendicular_line_equation (given_line : Line) (point : Point) :
  given_line.a = 2 ∧ given_line.b = -3 ∧ given_line.c = 4 ∧
  point.x = -2 ∧ point.y = -3 →
  ∃ (l : Line), 
    pointOnLine point l ∧ 
    perpendicular l given_line ∧
    l.a = 3 ∧ l.b = 2 ∧ l.c = 12 := by
  sorry

end perpendicular_line_equation_l1515_151587


namespace number_of_female_democrats_l1515_151570

/-- Given a meeting with male and female participants, prove the number of female Democrats --/
theorem number_of_female_democrats 
  (total_participants : ℕ) 
  (female_participants : ℕ) 
  (male_participants : ℕ) 
  (h1 : total_participants = 780)
  (h2 : female_participants + male_participants = total_participants)
  (h3 : 2 * (total_participants / 3) = female_participants / 2 + male_participants / 4) :
  female_participants / 2 = 130 := by
  sorry

end number_of_female_democrats_l1515_151570


namespace fair_coin_probability_l1515_151589

theorem fair_coin_probability : 
  let n : ℕ := 8  -- number of coin tosses
  let p : ℚ := 1/2  -- probability of heads for a fair coin
  let favorable_outcomes : ℕ := (n.choose 2) + (n.choose 3) + (n.choose 4)
  let total_outcomes : ℕ := 2^n
  (favorable_outcomes : ℚ) / total_outcomes = 77/128 := by sorry

end fair_coin_probability_l1515_151589


namespace sum_of_four_solution_values_l1515_151573

-- Define the polynomial function f
noncomputable def f (x : ℝ) : ℝ := 
  (x - 5) * (x - 3) * (x - 1) * (x + 1) * (x + 3) * (x + 5) / 315 - 3.4

-- Define the property of having exactly 4 solutions
def has_four_solutions (c : ℝ) : Prop :=
  ∃ (x₁ x₂ x₃ x₄ : ℝ), 
    (f x₁ = c ∧ f x₂ = c ∧ f x₃ = c ∧ f x₄ = c) ∧
    (∀ x, f x = c → x = x₁ ∨ x = x₂ ∨ x = x₃ ∨ x = x₄)

-- Theorem statement
theorem sum_of_four_solution_values :
  ∃ (a b : ℤ), has_four_solutions (a : ℝ) ∧ has_four_solutions (b : ℝ) ∧ a + b = -7 :=
sorry

end sum_of_four_solution_values_l1515_151573


namespace dandelion_game_strategy_l1515_151519

/-- The dandelion mowing game -/
def has_winning_strategy (m n : ℕ+) : Prop :=
  (m.val + n.val) % 2 = 1 ∨ min m.val n.val = 1

theorem dandelion_game_strategy (m n : ℕ+) :
  has_winning_strategy m n ↔ (m.val + n.val) % 2 = 1 ∨ min m.val n.val = 1 := by
  sorry

end dandelion_game_strategy_l1515_151519


namespace book_set_cost_l1515_151511

/-- The cost of a book set given lawn mowing parameters -/
theorem book_set_cost 
  (charge_rate : ℚ)
  (lawn_length : ℕ)
  (lawn_width : ℕ)
  (lawns_mowed : ℕ)
  (additional_area : ℕ)
  (h1 : charge_rate = 1 / 10)
  (h2 : lawn_length = 20)
  (h3 : lawn_width = 15)
  (h4 : lawns_mowed = 3)
  (h5 : additional_area = 600) :
  (lawn_length * lawn_width * lawns_mowed + additional_area) * charge_rate = 150 := by
  sorry

#check book_set_cost

end book_set_cost_l1515_151511


namespace arithmetic_sequence_problem_l1515_151557

/-- An arithmetic sequence is a sequence where the difference between any two consecutive terms is constant. -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Given an arithmetic sequence where the 4th term is 23 and the 6th term is 47, the 8th term is 71. -/
theorem arithmetic_sequence_problem (a : ℕ → ℝ) (h : ArithmeticSequence a) 
    (h4 : a 4 = 23) (h6 : a 6 = 47) : a 8 = 71 := by
  sorry

end arithmetic_sequence_problem_l1515_151557


namespace unique_integer_with_16_divisors_l1515_151513

def hasSixteenDivisors (n : ℕ) : Prop :=
  (Finset.filter (· ∣ n) (Finset.range (n + 1))).card = 16

def divisorsOrdered (n : ℕ) : Prop :=
  ∀ i j, 1 ≤ i ∧ i < j ∧ j ≤ 16 → (Finset.filter (· ∣ n) (Finset.range (n + 1))).toList.nthLe i sorry <
    (Finset.filter (· ∣ n) (Finset.range (n + 1))).toList.nthLe j sorry

def divisorProperty (n : ℕ) : Prop :=
  let divisors := (Finset.filter (· ∣ n) (Finset.range (n + 1))).toList
  let d₂ := divisors.nthLe 1 sorry
  let d₄ := divisors.nthLe 3 sorry
  let d₅ := divisors.nthLe 4 sorry
  let d₆ := divisors.nthLe 5 sorry
  divisors.nthLe (d₅ - 1) sorry = (d₂ + d₄) * d₆

theorem unique_integer_with_16_divisors :
  ∃! n : ℕ, n > 0 ∧ hasSixteenDivisors n ∧ divisorsOrdered n ∧ divisorProperty n ∧ n = 2002 :=
sorry

end unique_integer_with_16_divisors_l1515_151513


namespace birds_meeting_time_l1515_151588

/-- The time taken for two birds flying in opposite directions to meet -/
theorem birds_meeting_time 
  (duck_time : ℝ) 
  (goose_time : ℝ) 
  (duck_time_positive : duck_time > 0)
  (goose_time_positive : goose_time > 0) :
  ∃ x : ℝ, x > 0 ∧ (1 / duck_time + 1 / goose_time) * x = 1 :=
sorry

end birds_meeting_time_l1515_151588


namespace distance_is_8_sqrt2_div_3_l1515_151593

/-- Two lines l₁ and l₂ in the plane -/
structure ParallelLines where
  a : ℝ
  l₁ : ℝ × ℝ → Prop
  l₂ : ℝ × ℝ → Prop
  l₁_eq : ∀ x y, l₁ (x, y) ↔ x + a * y + 6 = 0
  l₂_eq : ∀ x y, l₂ (x, y) ↔ (a - 2) * x + 3 * y + 2 * a = 0
  parallel : ∃ k : ℝ, k ≠ 0 ∧ ∀ x y, l₁ (x, y) ↔ l₂ (k * x, k * y)

/-- The distance between two parallel lines -/
def distance (lines : ParallelLines) : ℝ := sorry

/-- Theorem: The distance between the parallel lines is 8√2/3 -/
theorem distance_is_8_sqrt2_div_3 (lines : ParallelLines) :
  distance lines = 8 * Real.sqrt 2 / 3 := by sorry

end distance_is_8_sqrt2_div_3_l1515_151593


namespace problem_statement_l1515_151515

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := ((x + 1)^2 + a * Real.sin x) / (x^2 + 1) + 3

theorem problem_statement (a : ℝ) :
  f a (Real.log (Real.log 5 / Real.log 2)) = 5 →
  f a (Real.log (Real.log 2 / Real.log 5)) = 3 := by
sorry

end problem_statement_l1515_151515


namespace total_students_proof_l1515_151503

def school_problem (n : ℕ) (largest_class : ℕ) (diff : ℕ) : ℕ :=
  let class_sizes := List.range n |>.map (fun i => largest_class - i * diff)
  class_sizes.sum

theorem total_students_proof :
  school_problem 5 25 2 = 105 := by
  sorry

end total_students_proof_l1515_151503


namespace trajectory_of_point_on_moving_segment_l1515_151516

/-- The trajectory of a point M on a moving line segment AB -/
theorem trajectory_of_point_on_moving_segment (A B M : ℝ × ℝ) 
  (h_length : (A.1 - B.1)^2 + (A.2 - B.2)^2 = 4)
  (h_A_on_x : A.2 = 0)
  (h_B_on_y : B.1 = 0)
  (h_M_on_AB : ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ M = (1 - t) • A + t • B)
  (h_ratio : ∃ k : ℝ, k > 0 ∧ 
    (M.1 - A.1)^2 + (M.2 - A.2)^2 = k^2 * ((B.1 - M.1)^2 + (B.2 - M.2)^2) ∧
    k = 1/2) :
  9 * M.1^2 + 36 * M.2^2 = 16 := by
  sorry

end trajectory_of_point_on_moving_segment_l1515_151516


namespace wire_length_ratio_l1515_151540

theorem wire_length_ratio (edge_length : ℕ) (wire_pieces : ℕ) (wire_length : ℕ) : 
  edge_length = wire_length ∧ wire_pieces = 12 →
  (wire_pieces * wire_length) / (edge_length^3 * 12) = 1 / 36 := by
  sorry

end wire_length_ratio_l1515_151540


namespace linear_regression_coefficient_l1515_151518

theorem linear_regression_coefficient
  (x : Fin 4 → ℝ)
  (y : Fin 4 → ℝ)
  (h_x : x = ![6, 8, 10, 12])
  (h_y : y = ![6, 5, 3, 2])
  (a : ℝ)
  (h_reg : ∀ i, y i = a * x i + 10.3) :
  a = -0.7 := by
sorry

end linear_regression_coefficient_l1515_151518


namespace sqrt_2_plus_x_real_range_l1515_151524

theorem sqrt_2_plus_x_real_range (x : ℝ) : 
  (∃ y : ℝ, y^2 = 2 + x) ↔ x ≥ -2 :=
by sorry

end sqrt_2_plus_x_real_range_l1515_151524


namespace power_mod_seventeen_l1515_151562

theorem power_mod_seventeen : 2^2023 % 17 = 4 := by
  sorry

end power_mod_seventeen_l1515_151562


namespace system_solutions_l1515_151566

-- Define the logarithm base 4
noncomputable def log4 (x : ℝ) : ℝ := Real.log x / Real.log 4

-- Define the system of equations
def system (x y : ℝ) : Prop :=
  x + y - 20 = 0 ∧ log4 x + log4 y = 1 + log4 9

-- Theorem stating the solutions
theorem system_solutions :
  ∃ (x y : ℝ), system x y ∧ ((x = 18 ∧ y = 2) ∨ (x = 2 ∧ y = 18)) :=
sorry

end system_solutions_l1515_151566


namespace floor_sqrt_99_l1515_151550

theorem floor_sqrt_99 : ⌊Real.sqrt 99⌋ = 9 := by
  sorry

end floor_sqrt_99_l1515_151550


namespace max_valid_sequence_length_l1515_151554

def is_valid_sequence (seq : List Nat) : Prop :=
  (∀ i, i + 2 < seq.length → (seq[i]! + seq[i+1]! + seq[i+2]!) % 2 = 0) ∧
  (∀ i, i + 3 < seq.length → (seq[i]! + seq[i+1]! + seq[i+2]! + seq[i+3]!) % 2 = 1)

theorem max_valid_sequence_length :
  (∃ (seq : List Nat), is_valid_sequence seq ∧ seq.length = 5) ∧
  (∀ (seq : List Nat), is_valid_sequence seq → seq.length ≤ 5) := by
  sorry

end max_valid_sequence_length_l1515_151554


namespace last_four_digits_of_5_to_15000_l1515_151512

theorem last_four_digits_of_5_to_15000 (h : 5^500 ≡ 1 [ZMOD 1250]) :
  5^15000 ≡ 1 [ZMOD 1250] := by
  sorry

end last_four_digits_of_5_to_15000_l1515_151512


namespace bowtie_equation_solution_l1515_151579

-- Define the operation
noncomputable def bowtie (c x : ℝ) : ℝ := c + Real.sqrt (x + Real.sqrt (x + Real.sqrt (x + Real.sqrt x)))

-- State the theorem
theorem bowtie_equation_solution :
  ∃ x : ℝ, bowtie 5 x = 11 ∧ x = 30 := by sorry

end bowtie_equation_solution_l1515_151579
