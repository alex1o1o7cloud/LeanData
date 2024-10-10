import Mathlib

namespace trigonometric_inequality_l1036_103666

theorem trigonometric_inequality (a b A G : ℝ) : 
  a = Real.sin (π / 3) →
  b = Real.cos (π / 3) →
  A = (a + b) / 2 →
  G = Real.sqrt (a * b) →
  b < G ∧ G < A ∧ A < a :=
by sorry

end trigonometric_inequality_l1036_103666


namespace circles_intersect_iff_l1036_103620

/-- Two circles C1 and C2 in the plane -/
structure TwoCircles where
  /-- The parameter a for the second circle -/
  a : ℝ
  /-- a is positive -/
  a_pos : a > 0

/-- The condition for the two circles to intersect -/
def intersect (c : TwoCircles) : Prop :=
  3 < c.a ∧ c.a < 5

/-- Theorem stating the necessary and sufficient condition for the circles to intersect -/
theorem circles_intersect_iff (c : TwoCircles) :
  (∃ (x y : ℝ), x^2 + (y-1)^2 = 1 ∧ (x-c.a)^2 + (y-1)^2 = 16) ↔ intersect c := by
  sorry

end circles_intersect_iff_l1036_103620


namespace proportional_function_value_l1036_103618

/-- A function f is proportional if it can be written as f(x) = kx for some constant k -/
def IsProportional (f : ℝ → ℝ) : Prop :=
  ∃ k : ℝ, ∀ x : ℝ, f x = k * x

/-- The function f(x) = (m-2)x + m^2 - 4 -/
def f (m : ℝ) (x : ℝ) : ℝ := (m - 2) * x + m^2 - 4

theorem proportional_function_value :
  ∀ m : ℝ, IsProportional (f m) → f m (-2) = 8 := by
  sorry

end proportional_function_value_l1036_103618


namespace line_equation_proof_l1036_103698

/-- A line that passes through a point and intersects both axes -/
structure IntersectingLine where
  -- The point through which the line passes
  P : ℝ × ℝ
  -- The point where the line intersects the x-axis
  A : ℝ × ℝ
  -- The point where the line intersects the y-axis
  B : ℝ × ℝ
  -- Ensure A is on the x-axis
  hA : A.2 = 0
  -- Ensure B is on the y-axis
  hB : B.1 = 0
  -- Ensure P is the midpoint of AB
  hP : P = ((A.1 + B.1) / 2, (A.2 + B.2) / 2)

/-- The equation of the line is 3x - 2y + 24 = 0 -/
def lineEquation (l : IntersectingLine) : Prop :=
  ∀ x y : ℝ, (x, y) ∈ {p : ℝ × ℝ | 3 * p.1 - 2 * p.2 + 24 = 0} ↔ 
    ∃ t : ℝ, (x, y) = (1 - t) • l.A + t • l.B

/-- The main theorem -/
theorem line_equation_proof (l : IntersectingLine) (h : l.P = (-4, 6)) : 
  lineEquation l := by sorry

end line_equation_proof_l1036_103698


namespace triangle_and_function_properties_l1036_103617

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    and vectors m and n that are parallel, prove the angle B and properties of function f. -/
theorem triangle_and_function_properties
  (a b c : ℝ)
  (A B C : ℝ)
  (m : ℝ × ℝ)
  (n : ℝ × ℝ)
  (ω : ℝ)
  (h_triangle : 0 < A ∧ 0 < B ∧ 0 < C ∧ A + B + C = π)
  (h_sides : a > 0 ∧ b > 0 ∧ c > 0)
  (h_m : m = (b, 2*a - c))
  (h_n : n = (Real.cos B, Real.cos C))
  (h_parallel : ∃ (k : ℝ), m = k • n)
  (h_ω : ω > 0)
  (f : ℝ → ℝ)
  (h_f : f = λ x => Real.cos (ω * x - π/6) + Real.sin (ω * x))
  (h_period : ∀ x, f (x + π) = f x) :
  (B = π/3) ∧
  (∃ x₀ ∈ Set.Icc 0 (π/2), ∀ x ∈ Set.Icc 0 (π/2), f x ≤ f x₀ ∧ f x₀ = Real.sqrt 3) ∧
  (∃ x₁ ∈ Set.Icc 0 (π/2), ∀ x ∈ Set.Icc 0 (π/2), f x₁ ≤ f x ∧ f x₁ = -Real.sqrt 3 / 2) :=
by sorry

end triangle_and_function_properties_l1036_103617


namespace perpendicular_implies_cos_value_triangle_implies_f_range_l1036_103614

noncomputable section

-- Define the vectors m and n
def m (x : ℝ) : Fin 2 → ℝ := ![Real.sqrt 3 * Real.sin (x/4), 1]
def n (x : ℝ) : Fin 2 → ℝ := ![Real.cos (x/4), Real.cos (x/4)^2]

-- Define the dot product
def dot_product (v w : Fin 2 → ℝ) : ℝ := (v 0) * (w 0) + (v 1) * (w 1)

-- Define perpendicularity
def perpendicular (v w : Fin 2 → ℝ) : Prop := dot_product v w = 0

-- Define the function f
def f (x : ℝ) : ℝ := dot_product (m x) (n x)

-- Theorem 1
theorem perpendicular_implies_cos_value (x : ℝ) :
  perpendicular (m x) (n x) → Real.cos (2 * Real.pi / 3 - x) = -1/2 := by sorry

-- Theorem 2
theorem triangle_implies_f_range (A B C a b c : ℝ) :
  A + B + C = Real.pi →
  (2 * a - c) * Real.cos B = b * Real.cos C →
  0 < A →
  A < 2 * Real.pi / 3 →
  ∃ (y : ℝ), 1 < f A ∧ f A < 3/2 := by sorry

end

end perpendicular_implies_cos_value_triangle_implies_f_range_l1036_103614


namespace larger_number_problem_l1036_103685

theorem larger_number_problem (smaller larger : ℕ) : 
  larger - smaller = 1365 →
  larger = 6 * smaller + 35 →
  larger = 1631 := by
  sorry

end larger_number_problem_l1036_103685


namespace remainder_problem_l1036_103664

theorem remainder_problem (n : ℤ) (h : n % 7 = 3) : (4 * n - 9) % 7 = 3 := by
  sorry

end remainder_problem_l1036_103664


namespace inequality_system_solution_l1036_103619

theorem inequality_system_solution (a : ℝ) : 
  (∀ x : ℝ, x > 1 ↔ (x - 1 > 0 ∧ 2*x - a > 0)) →
  a ≤ 2 :=
by sorry

end inequality_system_solution_l1036_103619


namespace average_increase_is_three_l1036_103697

/-- Represents a batsman's performance over multiple innings -/
structure BatsmanPerformance where
  innings : Nat
  totalRuns : Nat
  averageRuns : Rat

/-- Calculates the increase in average runs after a new inning -/
def averageIncrease (prev : BatsmanPerformance) (newRuns : Nat) (newAverage : Rat) : Rat :=
  newAverage - prev.averageRuns

/-- Theorem: The increase in the batsman's average is 3 runs -/
theorem average_increase_is_three 
  (prev : BatsmanPerformance) 
  (h1 : prev.innings = 16) 
  (h2 : (prev.totalRuns + 87 : Rat) / 17 = 39) : 
  averageIncrease prev 87 39 = 3 := by
sorry

end average_increase_is_three_l1036_103697


namespace fraction_non_negative_l1036_103612

theorem fraction_non_negative (x : ℝ) : (x + 7) / (x^2 + 2*x + 8) ≥ 0 ↔ x ≥ -7 := by
  sorry

end fraction_non_negative_l1036_103612


namespace quadratic_real_roots_l1036_103641

theorem quadratic_real_roots (m : ℝ) :
  (∃ x : ℝ, x^2 + 4*x + m + 5 = 0) ↔ m ≤ -1 := by
  sorry

end quadratic_real_roots_l1036_103641


namespace cos_five_pi_thirds_plus_two_alpha_l1036_103686

theorem cos_five_pi_thirds_plus_two_alpha (α : ℝ) 
  (h : Real.cos (π/6 - α) = 2/3) : 
  Real.cos (5*π/3 + 2*α) = -1/9 := by
  sorry

end cos_five_pi_thirds_plus_two_alpha_l1036_103686


namespace thirteenth_result_l1036_103611

theorem thirteenth_result (total_count : Nat) (total_average : ℝ) 
  (first_twelve_average : ℝ) (last_twelve_average : ℝ) :
  total_count = 25 →
  total_average = 20 →
  first_twelve_average = 14 →
  last_twelve_average = 17 →
  (12 * first_twelve_average + 12 * last_twelve_average + 
    (total_count * total_average - 12 * first_twelve_average - 12 * last_twelve_average)) / 1 = 128 := by
  sorry

#check thirteenth_result

end thirteenth_result_l1036_103611


namespace triangular_prism_no_body_diagonal_l1036_103687

-- Define what a prism is
structure Prism where
  base : Type
  has_base_diagonal : Bool

-- Define the property of having a body diagonal
def has_body_diagonal (p : Prism) : Bool := p.has_base_diagonal

-- Define specific types of prisms
def triangular_prism : Prism := { base := Unit, has_base_diagonal := false }

-- Theorem statement
theorem triangular_prism_no_body_diagonal : 
  ¬(has_body_diagonal triangular_prism) := by sorry

end triangular_prism_no_body_diagonal_l1036_103687


namespace polynomial_division_remainder_l1036_103671

theorem polynomial_division_remainder : ∃ (q r : Polynomial ℝ),
  X^5 + 4 = (X - 3)^2 * q + r ∧ 
  r = 331 * X - 746 ∧
  r.degree < 2 := by
  sorry

end polynomial_division_remainder_l1036_103671


namespace expression_evaluation_l1036_103696

theorem expression_evaluation (a b : ℤ) (h1 : a = 3) (h2 : b = 2) : 
  (a^2 + a*b + b^2)^2 - (a^2 - a*b + b^2)^2 = 72 := by
  sorry

end expression_evaluation_l1036_103696


namespace cubic_function_equality_l1036_103642

/-- Given two cubic functions f and g, prove that f(g(x)) = g(f(x)) for all x if and only if d = ±a -/
theorem cubic_function_equality (a b c d e f : ℝ) :
  (∀ x : ℝ, (a * (d * x^3 + e * x + f)^3 + b * (d * x^3 + e * x + f) + c) = 
            (d * (a * x^3 + b * x + c)^3 + e * (a * x^3 + b * x + c) + f)) ↔ 
  (d = a ∨ d = -a) := by
  sorry

end cubic_function_equality_l1036_103642


namespace work_completion_time_l1036_103650

theorem work_completion_time (work_rate_individual : ℝ) (total_work : ℝ) : 
  work_rate_individual > 0 → total_work > 0 →
  (total_work / work_rate_individual = 50) →
  (total_work / (2 * work_rate_individual) = 25) :=
by sorry

end work_completion_time_l1036_103650


namespace joe_original_cans_l1036_103643

/-- Represents the number of rooms that can be painted with a given number of paint cans -/
def rooms_paintable (cans : ℕ) : ℕ := sorry

/-- The number of rooms Joe could initially paint -/
def initial_rooms : ℕ := 40

/-- The number of rooms Joe could paint after losing cans -/
def remaining_rooms : ℕ := 32

/-- The number of cans Joe lost -/
def lost_cans : ℕ := 2

theorem joe_original_cans :
  ∃ (original_cans : ℕ),
    rooms_paintable original_cans = initial_rooms ∧
    rooms_paintable (original_cans - lost_cans) = remaining_rooms ∧
    original_cans = 10 := by sorry

end joe_original_cans_l1036_103643


namespace seating_theorem_l1036_103604

/-- Represents a group of people seated around a round table -/
structure SeatingArrangement where
  num_men : ℕ
  num_women : ℕ

/-- A man is satisfied if at least one woman is sitting next to him -/
def is_satisfied (s : SeatingArrangement) : Prop :=
  ∃ (p : ℝ), p = 1 - (s.num_men - 1) / (s.num_men + s.num_women - 1) * (s.num_men - 2) / (s.num_men + s.num_women - 2)

/-- The probability of a specific man being satisfied -/
def satisfaction_probability (s : SeatingArrangement) : ℚ :=
  25 / 33

/-- The expected number of satisfied men -/
def expected_satisfied_men (s : SeatingArrangement) : ℚ :=
  (s.num_men : ℚ) * (satisfaction_probability s)

/-- The main theorem about the seating arrangement -/
theorem seating_theorem (s : SeatingArrangement) 
  (h1 : s.num_men = 50) 
  (h2 : s.num_women = 50) : 
  is_satisfied s ∧ 
  satisfaction_probability s = 25 / 33 ∧ 
  expected_satisfied_men s = 1250 / 33 := by
  sorry


end seating_theorem_l1036_103604


namespace circle_configuration_radius_l1036_103634

/-- Given a configuration of three circles C, D, and E, prove that the radius of circle D is 4√15 - 14 -/
theorem circle_configuration_radius (C D E : ℝ → ℝ → Prop) (A B F : ℝ × ℝ) :
  (∀ x y, C x y ↔ (x - 0)^2 + (y - 0)^2 = 4) →  -- Circle C with radius 2 centered at origin
  (∃ x y, C x y ∧ D x y) →  -- D is internally tangent to C
  (∃ x y, C x y ∧ E x y) →  -- E is tangent to C
  (∃ x y, D x y ∧ E x y) →  -- E is externally tangent to D
  (∃ t, 0 ≤ t ∧ t ≤ 1 ∧ F = (2*t - 1, 0) ∧ E (2*t - 1) 0) →  -- E is tangent to AB at F
  (∀ x y z w, D x y ∧ E z w → (x - z)^2 + (y - w)^2 = (3*r)^2 - r^2) →  -- Radius of D is 3 times radius of E
  (∃ r_D, ∀ x y, D x y ↔ (x - 0)^2 + (y - 0)^2 = r_D^2 ∧ r_D = 4*Real.sqrt 15 - 14) :=
sorry

end circle_configuration_radius_l1036_103634


namespace equal_money_after_transfer_l1036_103665

/-- Represents the amount of gold coins each merchant has -/
structure Merchants where
  foma : ℤ
  ierema : ℤ
  yuliy : ℤ

/-- The conditions of the problem -/
def satisfies_conditions (m : Merchants) : Prop :=
  (m.ierema + 70 = m.yuliy) ∧ (m.foma - 40 = m.yuliy)

/-- The theorem to prove -/
theorem equal_money_after_transfer (m : Merchants) 
  (h : satisfies_conditions m) : 
  m.foma - 55 = m.ierema + 55 := by
  sorry

#check equal_money_after_transfer

end equal_money_after_transfer_l1036_103665


namespace set_B_representation_l1036_103660

-- Define the function f
def f (a b x : ℝ) : ℝ := x^2 - a*x + b

-- Define set A
def A (a b : ℝ) : Set ℝ := {x | f a b x - x = 0}

-- Define set B
def B (a b : ℝ) : Set ℝ := {x | f a b x - a*x = 0}

-- State the theorem
theorem set_B_representation (a b : ℝ) : 
  A a b = {1, -3} → B a b = {-2 - Real.sqrt 7, -2 + Real.sqrt 7} := by
  sorry

end set_B_representation_l1036_103660


namespace seniors_in_three_sports_l1036_103667

theorem seniors_in_three_sports 
  (total_seniors : ℕ) 
  (football : ℕ) 
  (baseball : ℕ) 
  (football_lacrosse : ℕ) 
  (baseball_football : ℕ) 
  (baseball_lacrosse : ℕ) 
  (h1 : total_seniors = 85)
  (h2 : football = 74)
  (h3 : baseball = 26)
  (h4 : football_lacrosse = 17)
  (h5 : baseball_football = 18)
  (h6 : baseball_lacrosse = 13)
  : ∃ (n : ℕ), n = 11 ∧ 
    total_seniors = football + baseball + 2*n - baseball_football - football_lacrosse - baseball_lacrosse + n :=
by sorry

end seniors_in_three_sports_l1036_103667


namespace anoop_joining_time_l1036_103609

/-- Proves that Anoop joined after 6 months given the investment conditions -/
theorem anoop_joining_time (arjun_investment anoop_investment : ℕ) 
  (total_months : ℕ) (x : ℕ) :
  arjun_investment = 20000 →
  anoop_investment = 40000 →
  total_months = 12 →
  arjun_investment * total_months = anoop_investment * (total_months - x) →
  x = 6 := by
  sorry

end anoop_joining_time_l1036_103609


namespace parallelogram_double_reflection_l1036_103672

-- Define the parallelogram vertices
def A : ℝ × ℝ := (3, 6)
def B : ℝ × ℝ := (5, 10)
def C : ℝ × ℝ := (7, 6)
def D : ℝ × ℝ := (5, 2)

-- Define the reflection functions
def reflect_x_axis (p : ℝ × ℝ) : ℝ × ℝ := (p.1, -p.2)

def reflect_y_eq_x_plus_2 (p : ℝ × ℝ) : ℝ × ℝ :=
  let p' := (p.1, p.2 - 2)  -- Translate down by 2
  let p'' := (p'.2, p'.1)   -- Reflect over y = x
  (p''.1, p''.2 + 2)        -- Translate up by 2

-- Theorem statement
theorem parallelogram_double_reflection :
  reflect_y_eq_x_plus_2 (reflect_x_axis D) = (-4, 7) := by
  sorry


end parallelogram_double_reflection_l1036_103672


namespace arctan_sum_special_case_l1036_103638

theorem arctan_sum_special_case (a b : ℝ) :
  a = 2/3 →
  (a + 1) * (b + 1) = 3 →
  Real.arctan a + Real.arctan b = π / 2 := by
  sorry

end arctan_sum_special_case_l1036_103638


namespace divisible_by_two_and_three_l1036_103691

theorem divisible_by_two_and_three (n : ℕ) : 
  (∃ (k : ℕ), k = 33 ∧ k = (n.div 6).succ) ↔ n = 204 :=
by sorry

end divisible_by_two_and_three_l1036_103691


namespace sqrt_equation_solution_l1036_103668

theorem sqrt_equation_solution (y : ℝ) : 
  Real.sqrt (1 + Real.sqrt (2 * y - 3)) = Real.sqrt 6 → y = 14 := by
sorry

end sqrt_equation_solution_l1036_103668


namespace children_on_airplane_l1036_103629

/-- Proves that the number of children on an airplane is 20 given specific conditions --/
theorem children_on_airplane (total_passengers : ℕ) (num_men : ℕ) :
  total_passengers = 80 →
  num_men = 30 →
  ∃ (num_women num_children : ℕ),
    num_women = num_men ∧
    num_children = total_passengers - (num_men + num_women) ∧
    num_children = 20 := by
  sorry

end children_on_airplane_l1036_103629


namespace unique_congruent_integer_l1036_103627

theorem unique_congruent_integer (h : ∃ m : ℤ, 10 ≤ m ∧ m ≤ 15 ∧ m ≡ 9433 [ZMOD 7]) :
  ∃! m : ℤ, 10 ≤ m ∧ m ≤ 15 ∧ m ≡ 9433 [ZMOD 7] ∧ m = 14 :=
by sorry

end unique_congruent_integer_l1036_103627


namespace probability_of_humanities_is_two_thirds_l1036_103677

/-- Represents a school subject -/
inductive Subject
| Mathematics
| Chinese
| Politics
| Geography
| English
| History
| PhysicalEducation

/-- Represents the time of day for a class -/
inductive TimeOfDay
| Morning
| Afternoon

/-- Defines whether a subject is considered a humanities subject -/
def isHumanities (s : Subject) : Bool :=
  match s with
  | Subject.Politics | Subject.History | Subject.Geography => true
  | _ => false

/-- Returns the list of subjects for a given time of day -/
def subjectsForTime (t : TimeOfDay) : List Subject :=
  match t with
  | TimeOfDay.Morning => [Subject.Mathematics, Subject.Chinese, Subject.Politics, Subject.Geography]
  | TimeOfDay.Afternoon => [Subject.English, Subject.History, Subject.PhysicalEducation]

/-- Calculates the probability of selecting at least one humanities class -/
def probabilityOfHumanities : ℚ :=
  let morningSubjects := subjectsForTime TimeOfDay.Morning
  let afternoonSubjects := subjectsForTime TimeOfDay.Afternoon
  let totalCombinations := morningSubjects.length * afternoonSubjects.length
  let humanitiesCombinations := 
    (morningSubjects.filter isHumanities).length * afternoonSubjects.length +
    (morningSubjects.filter (not ∘ isHumanities)).length * (afternoonSubjects.filter isHumanities).length
  humanitiesCombinations / totalCombinations

theorem probability_of_humanities_is_two_thirds :
  probabilityOfHumanities = 2 / 3 := by
  sorry

end probability_of_humanities_is_two_thirds_l1036_103677


namespace quadratic_inequality_solution_sets_l1036_103682

theorem quadratic_inequality_solution_sets 
  (a b : ℝ) 
  (h : Set.Ioo 2 3 = {x : ℝ | x^2 + a*x + b < 0}) :
  {x : ℝ | b*x^2 + a*x + 1 > 0} = Set.Iic (1/3) ∪ Set.Ioi (1/2) :=
sorry

end quadratic_inequality_solution_sets_l1036_103682


namespace arithmetic_triangle_b_range_l1036_103644

/-- A triangle with side lengths forming an arithmetic sequence --/
structure ArithmeticTriangle where
  a : ℝ
  b : ℝ
  c : ℝ
  is_arithmetic : ∃ d : ℝ, a = b - d ∧ c = b + d
  sum_of_squares : a^2 + b^2 + c^2 = 21
  triangle_inequality : a + b > c ∧ b + c > a ∧ c + a > b

/-- The range of possible values for the middle term of the arithmetic sequence --/
theorem arithmetic_triangle_b_range (t : ArithmeticTriangle) :
  t.b ∈ Set.Ioo (Real.sqrt 6) (Real.sqrt 7) ∪ {Real.sqrt 7} :=
sorry

end arithmetic_triangle_b_range_l1036_103644


namespace cubic_gt_27_implies_abs_gt_3_but_not_conversely_l1036_103639

theorem cubic_gt_27_implies_abs_gt_3_but_not_conversely :
  (∀ x : ℝ, x^3 > 27 → |x| > 3) ∧
  (∃ x : ℝ, |x| > 3 ∧ x^3 ≤ 27) :=
by sorry

end cubic_gt_27_implies_abs_gt_3_but_not_conversely_l1036_103639


namespace min_distance_circle_line_l1036_103600

/-- The minimum distance between a point on the circle (x + 1)² + y² = 1 
    and a point on the line 3x + 4y + 13 = 0 is equal to 1. -/
theorem min_distance_circle_line : 
  ∃ (d : ℝ), d = 1 ∧ 
  ∀ (x₁ y₁ x₂ y₂ : ℝ), 
    ((x₁ + 1)^2 + y₁^2 = 1) →
    (3*x₂ + 4*y₂ + 13 = 0) →
    ((x₁ - x₂)^2 + (y₁ - y₂)^2)^(1/2) ≥ d :=
by sorry

end min_distance_circle_line_l1036_103600


namespace root_in_interval_l1036_103680

theorem root_in_interval :
  ∃ x₀ ∈ Set.Ioo (1/2 : ℝ) 1, Real.exp x₀ = 3 - 2 * x₀ := by
  sorry

end root_in_interval_l1036_103680


namespace smallest_integer_satisfying_inequality_seven_satisfies_inequality_seven_is_smallest_l1036_103631

theorem smallest_integer_satisfying_inequality :
  ∀ n : ℤ, n^2 - 15*n + 56 ≤ 0 → n ≥ 7 :=
by
  sorry

theorem seven_satisfies_inequality :
  7^2 - 15*7 + 56 ≤ 0 :=
by
  sorry

theorem seven_is_smallest :
  ∀ n : ℤ, n < 7 → n^2 - 15*n + 56 > 0 :=
by
  sorry

end smallest_integer_satisfying_inequality_seven_satisfies_inequality_seven_is_smallest_l1036_103631


namespace square_of_five_equals_twentyfive_l1036_103663

theorem square_of_five_equals_twentyfive : (5 : ℕ)^2 = 25 := by
  sorry

end square_of_five_equals_twentyfive_l1036_103663


namespace power_fraction_simplification_l1036_103655

theorem power_fraction_simplification :
  (3^2016 + 3^2014) / (3^2016 - 3^2014) = 5/4 := by
  sorry

end power_fraction_simplification_l1036_103655


namespace special_arrangements_count_l1036_103649

/-- The number of ways to arrange guests in a special circular formation. -/
def specialArrangements (n : ℕ) : ℕ :=
  (3 * n).factorial

/-- The main theorem stating that the number of special arrangements is (3n)! -/
theorem special_arrangements_count (n : ℕ) :
  specialArrangements n = (3 * n).factorial := by
  sorry

end special_arrangements_count_l1036_103649


namespace runners_meeting_time_l1036_103640

/-- Represents a runner with their lap time and start time offset -/
structure Runner where
  lap_time : ℕ
  start_offset : ℕ

/-- Calculates the earliest meeting time for multiple runners -/
def earliest_meeting_time (runners : List Runner) : ℕ :=
  sorry

/-- The main theorem stating the earliest meeting time for the given runners -/
theorem runners_meeting_time :
  let ben := Runner.mk 5 0
  let emily := Runner.mk 8 2
  let nick := Runner.mk 9 4
  earliest_meeting_time [ben, emily, nick] = 360 :=
sorry

end runners_meeting_time_l1036_103640


namespace intersection_M_N_l1036_103624

def M : Set ℝ := {x : ℝ | 0 < x ∧ x < 4}
def N : Set ℝ := {x : ℝ | 1/3 ≤ x ∧ x ≤ 5}

theorem intersection_M_N : M ∩ N = {x : ℝ | 1/3 ≤ x ∧ x < 4} := by sorry

end intersection_M_N_l1036_103624


namespace x_equals_seven_l1036_103661

theorem x_equals_seven (x y : ℝ) (hx : x > 0) (hy : y > 0)
  (h : 7 * x^3 + 14 * x^2 * y = x^4 + 2 * x^3 * y) : x = 7 := by
  sorry

end x_equals_seven_l1036_103661


namespace sum_of_coefficients_for_factored_form_l1036_103605

theorem sum_of_coefficients_for_factored_form : ∃ (a b c d e f : ℤ),
  (2401 : ℤ) * x^4 + 16 = (a * x + b) * (c * x^3 + d * x^2 + e * x + f) ∧
  a + b + c + d + e + f = 274 :=
by sorry

end sum_of_coefficients_for_factored_form_l1036_103605


namespace jacob_pencils_l1036_103654

theorem jacob_pencils (total : ℕ) (zain_monday : ℕ) (zain_tuesday : ℕ) : 
  total = 21 →
  zain_monday + zain_tuesday + (2 * zain_monday + zain_tuesday) / 3 = total →
  (2 * zain_monday + zain_tuesday) / 3 = 8 :=
by sorry

end jacob_pencils_l1036_103654


namespace intersection_A_B_intersection_complements_A_B_l1036_103693

-- Define the universal set U
def U : Set Nat := {x | 1 ≤ x ∧ x ≤ 10}

-- Define sets A and B
def A : Set Nat := {1, 2, 3, 5, 8}
def B : Set Nat := {1, 3, 5, 7, 9}

-- Theorem for the intersection of A and B
theorem intersection_A_B : A ∩ B = {1, 3, 5} := by sorry

-- Theorem for the intersection of complements of A and B
theorem intersection_complements_A_B : (U \ A) ∩ (U \ B) = {4, 6, 10} := by sorry

end intersection_A_B_intersection_complements_A_B_l1036_103693


namespace quadrilateral_iff_interior_exterior_sum_equal_l1036_103690

/-- A polygon has 4 sides if and only if the sum of its interior angles is equal to the sum of its exterior angles. -/
theorem quadrilateral_iff_interior_exterior_sum_equal (n : ℕ) : n = 4 ↔ (n - 2) * 180 = 360 := by
  sorry

end quadrilateral_iff_interior_exterior_sum_equal_l1036_103690


namespace real_roots_of_polynomial_l1036_103601

def polynomial (x : ℝ) : ℝ := x^5 - 3*x^4 + 3*x^3 - x^2 - 4*x + 4

theorem real_roots_of_polynomial :
  ∀ x : ℝ, polynomial x = 0 ↔ x = 2 ∨ x = -Real.sqrt 2 ∨ x = Real.sqrt 2 :=
by sorry

end real_roots_of_polynomial_l1036_103601


namespace five_apples_ten_oranges_baskets_l1036_103623

/-- Represents the number of different fruit baskets that can be made -/
def fruitBaskets (apples oranges : ℕ) : ℕ :=
  (apples + 1) * (oranges + 1) - 1

/-- Theorem stating that the number of different fruit baskets
    with 5 apples and 10 oranges is 65 -/
theorem five_apples_ten_oranges_baskets :
  fruitBaskets 5 10 = 65 := by
  sorry

end five_apples_ten_oranges_baskets_l1036_103623


namespace problem_solution_l1036_103610

theorem problem_solution (x y : ℝ) 
  (h1 : x / 2 + 5 = 11) 
  (h2 : Real.sqrt y = x) : 
  x = 12 ∧ y = 144 := by
sorry

end problem_solution_l1036_103610


namespace multiple_with_specific_remainder_l1036_103658

theorem multiple_with_specific_remainder (x : ℕ) (hx : x > 0) 
  (hx_rem : x % 9 = 5) : 
  (∃ k : ℕ, k > 0 ∧ (k * x) % 9 = 2) ∧ 
  (∀ k : ℕ, k > 0 → (k * x) % 9 = 2 → k ≥ 4) :=
sorry

end multiple_with_specific_remainder_l1036_103658


namespace quadratic_roots_value_l1036_103645

theorem quadratic_roots_value (x₁ x₂ m : ℝ) : 
  (∀ x, x^2 - 8*x + m = 0 ↔ x = x₁ ∨ x = x₂) →
  x₁ = 3*x₂ →
  m = 12 :=
by sorry

end quadratic_roots_value_l1036_103645


namespace initial_puppies_count_l1036_103678

/-- The number of puppies Alyssa gave away -/
def puppies_given_away : ℕ := 7

/-- The number of puppies Alyssa has now -/
def puppies_remaining : ℕ := 5

/-- The initial number of puppies Alyssa had -/
def initial_puppies : ℕ := puppies_given_away + puppies_remaining

theorem initial_puppies_count : initial_puppies = 12 := by
  sorry

end initial_puppies_count_l1036_103678


namespace triangle_side_ratio_l1036_103648

-- Define a triangle ABC
structure Triangle where
  A : Real
  B : Real
  C : Real
  a : Real
  b : Real
  c : Real

-- Define the theorem
theorem triangle_side_ratio (t : Triangle) : 
  (t.A : Real) / (t.B : Real) = 1 / 2 ∧ 
  (t.B : Real) / (t.C : Real) = 2 / 3 → 
  t.a / t.b = 1 / Real.sqrt 3 ∧ 
  t.b / t.c = Real.sqrt 3 / 2 := by
sorry

end triangle_side_ratio_l1036_103648


namespace perfect_square_trinomial_l1036_103679

theorem perfect_square_trinomial (a : ℝ) : 
  (∃ b : ℝ, ∀ x : ℝ, x^2 + a*x + 81 = (x + b)^2) → (a = 18 ∨ a = -18) := by
  sorry

end perfect_square_trinomial_l1036_103679


namespace stratified_sample_school_b_l1036_103652

/-- Represents the number of students in each school -/
structure SchoolPopulation where
  a : ℕ  -- Number of students in school A
  b : ℕ  -- Number of students in school B
  c : ℕ  -- Number of students in school C

/-- The total number of students across all three schools -/
def total_students : ℕ := 1500

/-- The size of the sample to be drawn -/
def sample_size : ℕ := 120

/-- Checks if the given school population forms an arithmetic sequence -/
def is_arithmetic_sequence (pop : SchoolPopulation) : Prop :=
  pop.b - pop.a = pop.c - pop.b

/-- Checks if the given school population sums to the total number of students -/
def is_valid_population (pop : SchoolPopulation) : Prop :=
  pop.a + pop.b + pop.c = total_students

/-- Calculates the number of students to be sampled from a given school -/
def stratified_sample_size (school_size : ℕ) : ℕ :=
  school_size * sample_size / total_students

/-- The main theorem: proves that the number of students to be sampled from school B is 40 -/
theorem stratified_sample_school_b :
  ∀ pop : SchoolPopulation,
  is_arithmetic_sequence pop →
  is_valid_population pop →
  stratified_sample_size pop.b = 40 := by
  sorry

end stratified_sample_school_b_l1036_103652


namespace max_square_partitions_l1036_103673

/-- Represents the dimensions of the rectangular field -/
structure FieldDimensions where
  width : ℕ
  length : ℕ

/-- Represents the available internal fencing -/
def availableFencing : ℕ := 2100

/-- Calculates the number of square partitions given the side length of each square -/
def numPartitions (field : FieldDimensions) (squareSide : ℕ) : ℕ :=
  (field.width / squareSide) * (field.length / squareSide)

/-- Calculates the required internal fencing for given partitions -/
def requiredFencing (field : FieldDimensions) (squareSide : ℕ) : ℕ :=
  (field.width / squareSide - 1) * field.length + 
  (field.length / squareSide - 1) * field.width

/-- Theorem stating the maximum number of square partitions -/
theorem max_square_partitions (field : FieldDimensions) 
  (h1 : field.width = 30) 
  (h2 : field.length = 45) : 
  (∃ (squareSide : ℕ), 
    numPartitions field squareSide = 75 ∧ 
    requiredFencing field squareSide ≤ availableFencing ∧
    ∀ (otherSide : ℕ), 
      requiredFencing field otherSide ≤ availableFencing → 
      numPartitions field otherSide ≤ 75) :=
  sorry

#check max_square_partitions

end max_square_partitions_l1036_103673


namespace max_pencils_buyable_l1036_103699

def total_money : ℚ := 36
def pencil_cost : ℚ := 1.80
def pen_cost : ℚ := 2.60
def num_pens : ℕ := 9

theorem max_pencils_buyable :
  ∃ (num_pencils : ℕ),
    (num_pencils * pencil_cost + num_pens * pen_cost ≤ total_money) ∧
    ((num_pencils + num_pens) % 3 = 0) ∧
    (∀ (n : ℕ), n > num_pencils →
      (n * pencil_cost + num_pens * pen_cost > total_money ∨
       (n + num_pens) % 3 ≠ 0)) ∧
    num_pencils = 6 :=
by sorry

end max_pencils_buyable_l1036_103699


namespace restaurant_additional_hamburgers_l1036_103695

/-- The number of additional hamburgers made by a restaurant -/
def additional_hamburgers (initial : ℝ) (final : ℝ) : ℝ :=
  final - initial

/-- Proof that the restaurant made 3 additional hamburgers -/
theorem restaurant_additional_hamburgers : 
  let initial_hamburgers : ℝ := 9.0
  let final_hamburgers : ℝ := 12.0
  additional_hamburgers initial_hamburgers final_hamburgers = 3 := by
sorry

end restaurant_additional_hamburgers_l1036_103695


namespace ceiling_sqrt_200_l1036_103653

theorem ceiling_sqrt_200 : ⌈Real.sqrt 200⌉ = 15 := by
  sorry

end ceiling_sqrt_200_l1036_103653


namespace cubic_root_equation_l1036_103625

theorem cubic_root_equation (y : ℝ) : 
  (1 + Real.sqrt (2 * y - 3)) ^ (1/3 : ℝ) = 3 → y = 339.5 := by
  sorry

end cubic_root_equation_l1036_103625


namespace max_teams_in_tournament_l1036_103651

/-- The number of players in each team -/
def players_per_team : ℕ := 3

/-- The maximum number of games that can be played in the tournament -/
def max_games : ℕ := 200

/-- The number of games played between two teams -/
def games_between_teams : ℕ := players_per_team * players_per_team

/-- The function to calculate the total number of games for a given number of teams -/
def total_games (n : ℕ) : ℕ := games_between_teams * (n * (n - 1) / 2)

/-- The theorem stating the maximum number of teams that can participate -/
theorem max_teams_in_tournament : 
  ∃ (n : ℕ), n > 0 ∧ total_games n ≤ max_games ∧ ∀ m : ℕ, m > n → total_games m > max_games :=
by sorry

end max_teams_in_tournament_l1036_103651


namespace fraction_multiplication_l1036_103689

theorem fraction_multiplication (a b : ℝ) (h : a ≠ b) : 
  (3*a * 3*b) / (3*a - 3*b) = 3 * (a*b / (a - b)) := by
sorry

end fraction_multiplication_l1036_103689


namespace simplify_exponents_l1036_103628

theorem simplify_exponents (t : ℝ) : (t^4 * t^5) * (t^2)^2 = t^13 := by
  sorry

end simplify_exponents_l1036_103628


namespace second_equation_value_l1036_103622

theorem second_equation_value (x y : ℝ) 
  (eq1 : 2 * x + y = 26) 
  (eq2 : (x + y) / 3 = 4) : 
  x + 2 * y = 10 := by
sorry

end second_equation_value_l1036_103622


namespace max_intersection_points_l1036_103630

/-- Represents a line in the plane -/
structure Line :=
  (id : ℕ)

/-- The set of all lines -/
def all_lines : Finset Line := sorry

/-- The set of lines that are parallel to each other -/
def parallel_lines : Finset Line := sorry

/-- The set of lines that pass through point B -/
def point_b_lines : Finset Line := sorry

/-- A point of intersection between two lines -/
structure IntersectionPoint :=
  (l1 : Line)
  (l2 : Line)

/-- The set of all intersection points -/
def intersection_points : Finset IntersectionPoint := sorry

theorem max_intersection_points :
  (∀ l ∈ all_lines, l.id ≤ 150) →
  (∀ l ∈ all_lines, ∀ m ∈ all_lines, l ≠ m → l.id ≠ m.id) →
  (Finset.card all_lines = 150) →
  (∀ n : ℕ, n > 0 → parallel_lines.card = 100) →
  (∀ n : ℕ, n > 0 → point_b_lines.card = 50) →
  (∀ l ∈ parallel_lines, ∀ m ∈ parallel_lines, l ≠ m → ¬∃ p : IntersectionPoint, p.l1 = l ∧ p.l2 = m) →
  (∀ l ∈ point_b_lines, ∀ m ∈ point_b_lines, l ≠ m → ∃! p : IntersectionPoint, p.l1 = l ∧ p.l2 = m) →
  (∀ l ∈ parallel_lines, ∀ m ∈ point_b_lines, ∃! p : IntersectionPoint, p.l1 = l ∧ p.l2 = m) →
  Finset.card intersection_points = 5001 :=
by sorry

end max_intersection_points_l1036_103630


namespace solve_for_a_l1036_103692

theorem solve_for_a (m d b a : ℝ) (h1 : m = (d * a * b) / (a - b)) (h2 : m ≠ d * b) :
  a = (m * b) / (m - d * b) := by
  sorry

end solve_for_a_l1036_103692


namespace game_ends_after_28_rounds_l1036_103662

/-- Represents the state of the game at any given round -/
structure GameState where
  x : Nat
  y : Nat
  z : Nat

/-- Represents the rules of the token redistribution game -/
def redistributeTokens (state : GameState) : GameState :=
  sorry

/-- Determines if the game has ended (i.e., if any player has run out of tokens) -/
def gameEnded (state : GameState) : Bool :=
  sorry

/-- Counts the number of rounds until the game ends -/
def countRounds (state : GameState) : Nat :=
  sorry

/-- Theorem stating that the game ends after 28 rounds -/
theorem game_ends_after_28_rounds :
  countRounds (GameState.mk 18 15 12) = 28 := by
  sorry

end game_ends_after_28_rounds_l1036_103662


namespace bouquet_cost_45_l1036_103637

/-- The cost of a bouquet of lilies, given the number of lilies -/
def bouquet_cost (n : ℕ) : ℚ :=
  30 * (n : ℚ) / 18

theorem bouquet_cost_45 : bouquet_cost 45 = 75 := by
  sorry

end bouquet_cost_45_l1036_103637


namespace sum_of_first_six_primes_mod_seventh_prime_l1036_103659

theorem sum_of_first_six_primes_mod_seventh_prime : 
  let sum_first_six_primes := 41
  let seventh_prime := 17
  sum_first_six_primes % seventh_prime = 7 := by
sorry

end sum_of_first_six_primes_mod_seventh_prime_l1036_103659


namespace parallelogram_area_l1036_103647

/-- The area of a parallelogram, given the area of a triangle formed by its diagonal -/
theorem parallelogram_area (triangle_area : ℝ) (h : triangle_area = 64) : 
  2 * triangle_area = 128 := by
  sorry

end parallelogram_area_l1036_103647


namespace flag_paint_cost_l1036_103681

-- Define constants
def flag_width : Real := 3.5
def flag_height : Real := 2.5
def paint_cost_per_quart : Real := 4
def paint_coverage_per_quart : Real := 4
def sq_ft_per_sq_m : Real := 10.7639

-- Define the theorem
theorem flag_paint_cost : 
  let flag_area := flag_width * flag_height
  let total_area := 2 * flag_area
  let total_area_sq_ft := total_area * sq_ft_per_sq_m
  let quarts_needed := ⌈total_area_sq_ft / paint_coverage_per_quart⌉
  let total_cost := quarts_needed * paint_cost_per_quart
  total_cost = 192 := by
  sorry


end flag_paint_cost_l1036_103681


namespace crate_middle_dimension_l1036_103646

/-- Represents the dimensions of a crate -/
structure CrateDimensions where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Represents a right circular cylinder -/
structure Cylinder where
  radius : ℝ
  height : ℝ

/-- Checks if a cylinder fits upright in a crate -/
def cylinderFitsUpright (crate : CrateDimensions) (cylinder : Cylinder) : Prop :=
  (cylinder.radius * 2 ≤ crate.length ∧ cylinder.height ≤ crate.width) ∨
  (cylinder.radius * 2 ≤ crate.length ∧ cylinder.height ≤ crate.height) ∨
  (cylinder.radius * 2 ≤ crate.width ∧ cylinder.height ≤ crate.length) ∨
  (cylinder.radius * 2 ≤ crate.width ∧ cylinder.height ≤ crate.height) ∨
  (cylinder.radius * 2 ≤ crate.height ∧ cylinder.height ≤ crate.length) ∨
  (cylinder.radius * 2 ≤ crate.height ∧ cylinder.height ≤ crate.width)

theorem crate_middle_dimension (x : ℝ) :
  let crate := CrateDimensions.mk 5 x 12
  let cylinder := Cylinder.mk 5 12
  cylinderFitsUpright crate cylinder → x = 10 := by
  sorry

end crate_middle_dimension_l1036_103646


namespace meeting_selection_ways_l1036_103632

/-- The number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := Nat.choose n k

/-- The total number of managers -/
def total_managers : ℕ := 7

/-- The number of managers needed for the meeting -/
def meeting_size : ℕ := 4

/-- The number of managers who cannot attend together -/
def incompatible_managers : ℕ := 2

/-- The number of ways to select managers for the meeting -/
def select_managers : ℕ :=
  choose (total_managers - incompatible_managers) meeting_size +
  incompatible_managers * choose (total_managers - 1) (meeting_size - 1)

theorem meeting_selection_ways :
  select_managers = 25 := by sorry

end meeting_selection_ways_l1036_103632


namespace smallest_difference_l1036_103613

def Digits : Finset Nat := {0, 3, 4, 7, 8}

def isValidArrangement (a b c d e : Nat) : Prop :=
  a ∈ Digits ∧ b ∈ Digits ∧ c ∈ Digits ∧ d ∈ Digits ∧ e ∈ Digits ∧
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧
  b ≠ c ∧ b ≠ d ∧ b ≠ e ∧
  c ≠ d ∧ c ≠ e ∧
  d ≠ e ∧
  a ≠ 0

def difference (a b c d e : Nat) : Nat :=
  (100 * a + 10 * b + c) - (10 * d + e)

theorem smallest_difference :
  ∀ a b c d e,
    isValidArrangement a b c d e →
    difference a b c d e ≥ 339 :=
by sorry

end smallest_difference_l1036_103613


namespace lcm_hcf_problem_l1036_103657

theorem lcm_hcf_problem (A B : ℕ+) : 
  Nat.lcm A B = 2310 →
  Nat.gcd A B = 30 →
  A = 210 →
  B = 330 := by
sorry

end lcm_hcf_problem_l1036_103657


namespace sum_after_100_operations_l1036_103635

def initial_sequence : List ℕ := [2, 11, 8, 9]

def operation (seq : List ℤ) : List ℤ :=
  seq ++ (seq.zip (seq.tail!)).map (fun (a, b) => b - a)

def sum_after_n_operations (n : ℕ) : ℤ :=
  30 + 7 * n

theorem sum_after_100_operations :
  sum_after_n_operations 100 = 730 :=
sorry

end sum_after_100_operations_l1036_103635


namespace coin_probability_l1036_103633

theorem coin_probability (p q : ℝ) : 
  q = 1 - p →
  (Nat.choose 10 5 : ℝ) * p^5 * q^5 = (Nat.choose 10 6 : ℝ) * p^6 * q^4 →
  p = 6/11 := by
sorry

end coin_probability_l1036_103633


namespace algebraic_expression_value_l1036_103636

theorem algebraic_expression_value (x y : ℝ) (h : x + y = 2) :
  (1/2) * x^2 + x * y + (1/2) * y^2 = 2 := by
  sorry

end algebraic_expression_value_l1036_103636


namespace linear_increase_l1036_103683

/-- A linear function f(x) = 5x - 3 -/
def f (x : ℝ) : ℝ := 5 * x - 3

/-- Theorem: For a linear function f(x) = 5x - 3, 
    if x₁ < x₂, then f(x₁) < f(x₂) -/
theorem linear_increase (x₁ x₂ : ℝ) (h : x₁ < x₂) : f x₁ < f x₂ := by
  sorry

end linear_increase_l1036_103683


namespace parabola_focus_distance_l1036_103615

/-- The value of p for a parabola y^2 = 2px where the distance between (-2, 3) and the focus is 5 -/
theorem parabola_focus_distance (p : ℝ) : 
  p > 0 → -- Condition that p is positive
  let focus : ℝ × ℝ := (p/2, 0) -- Definition of focus for parabola y^2 = 2px
  (((-2 : ℝ) - p/2)^2 + 3^2).sqrt = 5 → -- Distance formula between (-2, 3) and focus is 5
  p = 4 := by sorry

end parabola_focus_distance_l1036_103615


namespace complex_distance_range_l1036_103656

theorem complex_distance_range (z : ℂ) (h : Complex.abs z = 2) :
  ∃ (x : ℝ), 0 ≤ x ∧ x ≤ 4 ∧ Complex.abs (1 + Complex.I * Real.sqrt 3 + z) = x :=
sorry

end complex_distance_range_l1036_103656


namespace eliza_numbers_l1036_103674

theorem eliza_numbers (a b : ℤ) (h1 : 2 * a + 3 * b = 110) (h2 : a = 32 ∨ b = 32) : 
  (a = 7 ∧ b = 32) ∨ (a = 32 ∧ b = 7) := by
sorry

end eliza_numbers_l1036_103674


namespace diameter_eq_hypotenuse_l1036_103616

/-- Triangle PQR with sides PQ = 15, QR = 36, and RP = 39 -/
structure RightTriangle where
  PQ : ℝ
  QR : ℝ
  RP : ℝ
  pq_eq : PQ = 15
  qr_eq : QR = 36
  rp_eq : RP = 39
  right_angle : PQ^2 + QR^2 = RP^2

/-- The diameter of the circumscribed circle of a right triangle is equal to the length of its hypotenuse -/
theorem diameter_eq_hypotenuse (t : RightTriangle) : 
  2 * (t.RP / 2) = t.RP := by sorry

end diameter_eq_hypotenuse_l1036_103616


namespace betty_boxes_l1036_103621

theorem betty_boxes (total_oranges : ℕ) (oranges_per_box : ℕ) (boxes : ℕ) : 
  total_oranges = 24 → 
  oranges_per_box = 8 → 
  total_oranges = boxes * oranges_per_box → 
  boxes = 3 := by
sorry

end betty_boxes_l1036_103621


namespace f_strictly_decreasing_on_interval_l1036_103669

-- Define the function f
def f (x : ℝ) : ℝ := x^3 + 3*x^2 + 2

-- State the theorem
theorem f_strictly_decreasing_on_interval :
  ∀ x ∈ Set.Ioo (-2 : ℝ) 0, StrictMonoOn f (Set.Ioo (-2 : ℝ) 0) := by
  sorry

end f_strictly_decreasing_on_interval_l1036_103669


namespace range_of_a_l1036_103606

theorem range_of_a (p q : Prop) 
  (hp : p ↔ ∀ x : ℝ, x > 0 → x + 1/x > a)
  (hq : q ↔ ∃ x₀ : ℝ, x₀^2 - 2*a*x₀ + 1 ≤ 0)
  (hnq : ¬¬q)
  (hpq : ¬(p ∧ q)) :
  a ≥ 2 := by sorry

end range_of_a_l1036_103606


namespace congruence_problem_l1036_103684

theorem congruence_problem (x : ℤ) : 
  (3 * x + 8) % 17 = 3 → (2 * x + 14) % 17 = 5 := by
  sorry

end congruence_problem_l1036_103684


namespace officers_selection_count_l1036_103670

/-- Represents the number of ways to choose officers from a club. -/
def choose_officers (total_members boys girls : ℕ) : ℕ :=
  2 * (boys * (boys - 1) * (boys - 2))

/-- Theorem stating the number of ways to choose officers under given conditions. -/
theorem officers_selection_count :
  let total_members : ℕ := 24
  let boys : ℕ := 12
  let girls : ℕ := 12
  choose_officers total_members boys girls = 2640 := by
  sorry

#eval choose_officers 24 12 12

end officers_selection_count_l1036_103670


namespace quadratic_roots_imply_k_l1036_103603

theorem quadratic_roots_imply_k (k : ℝ) : 
  (∀ x : ℂ, 8 * x^2 + 4 * x + k = 0 ↔ x = (-4 + Complex.I * Real.sqrt 380) / 16 ∨ x = (-4 - Complex.I * Real.sqrt 380) / 16) →
  k = 12.375 := by
sorry

end quadratic_roots_imply_k_l1036_103603


namespace unique_birth_date_l1036_103676

/-- Represents a date in the 20th century -/
structure Date where
  day : Nat
  month : Nat
  year : Nat
  h1 : 1 ≤ day ∧ day ≤ 31
  h2 : 1 ≤ month ∧ month ≤ 12
  h3 : 1900 ≤ year ∧ year ≤ 1999

def date_to_number (d : Date) : Nat :=
  d.day * 10000 + d.month * 100 + (d.year - 1900)

/-- The birth dates of two friends satisfy the given conditions -/
def valid_birth_dates (d1 d2 : Date) : Prop :=
  d1.month = d2.month ∧
  d1.year = d2.year ∧
  d2.day = d1.day + 7 ∧
  date_to_number d2 = 6 * date_to_number d1

theorem unique_birth_date :
  ∃! d : Date, ∃ d2 : Date, valid_birth_dates d d2 ∧ d.day = 1 ∧ d.month = 4 ∧ d.year = 1900 :=
by sorry

end unique_birth_date_l1036_103676


namespace eleven_distinct_points_l1036_103688

/-- Represents a circular track with a cyclist and pedestrian -/
structure Track where
  length : ℝ
  pedestrian_speed : ℝ
  cyclist_speed : ℝ
  (cyclist_faster : cyclist_speed = pedestrian_speed * 1.55)
  (positive_speed : pedestrian_speed > 0)

/-- Calculates the number of distinct overtaking points on the track -/
def distinct_overtaking_points (track : Track) : ℕ :=
  sorry

/-- Theorem stating that there are 11 distinct overtaking points -/
theorem eleven_distinct_points (track : Track) :
  distinct_overtaking_points track = 11 := by
  sorry

end eleven_distinct_points_l1036_103688


namespace ackermann_3_2_l1036_103608

def A : ℕ → ℕ → ℕ
| 0, n => n + 1
| m + 1, 0 => A m 1
| m + 1, n + 1 => A m (A (m + 1) n)

theorem ackermann_3_2 : A 3 2 = 11 := by sorry

end ackermann_3_2_l1036_103608


namespace y_intercept_of_parallel_line_l1036_103602

/-- A line in the plane can be represented by its slope and a point it passes through. -/
structure Line where
  slope : ℝ
  point : ℝ × ℝ

/-- The y-intercept of a line is the y-coordinate of the point where the line intersects the y-axis. -/
def y_intercept (l : Line) : ℝ :=
  l.point.2 - l.slope * l.point.1

/-- Two lines are parallel if and only if they have the same slope. -/
def parallel (l1 l2 : Line) : Prop :=
  l1.slope = l2.slope

theorem y_intercept_of_parallel_line (b : Line) :
  parallel b { slope := 3, point := (0, 2) } →
  b.point = (5, 10) →
  y_intercept b = -5 := by
  sorry

#check y_intercept_of_parallel_line

end y_intercept_of_parallel_line_l1036_103602


namespace ratio_sum_max_l1036_103607

theorem ratio_sum_max (a b : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : a / b = 3 / 4) (h4 : a + b = 21) : 
  max a b = 12 := by
  sorry

end ratio_sum_max_l1036_103607


namespace positive_numbers_inequalities_l1036_103694

theorem positive_numbers_inequalities (a b c : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a + b + c = 1 → (1 - a) * (1 - b) * (1 - c) ≥ 8 * a * b * c) ∧
  (∃ r : ℝ, r > 0 ∧ b = a * r ∧ c = b * r → a^2 + b^2 + c^2 > (a - b + c)^2) := by
  sorry

end positive_numbers_inequalities_l1036_103694


namespace alphabet_theorem_l1036_103626

theorem alphabet_theorem (total : ℕ) (both : ℕ) (line_only : ℕ) 
  (h1 : total = 76) 
  (h2 : both = 20) 
  (h3 : line_only = 46) 
  (h4 : total = both + line_only + (total - (both + line_only))) :
  total - (both + line_only) = 30 := by
  sorry

end alphabet_theorem_l1036_103626


namespace final_sequence_values_l1036_103675

/-- The number of elements in the initial sequence -/
def n : ℕ := 2022

/-- Function to calculate the new value for a given position after one iteration -/
def newValue (i : ℕ) : ℕ := i^2 + 1

/-- The number of iterations required to reduce the sequence to two numbers -/
def iterations : ℕ := (n - 2) / 2

/-- The final two numbers in the sequence after all iterations -/
def finalPair : (ℕ × ℕ) := (newValue (n/2) + iterations, newValue (n/2 + 1) + iterations)

/-- Theorem stating the final two numbers in the sequence -/
theorem final_sequence_values :
  finalPair = (1023131, 1025154) := by sorry

end final_sequence_values_l1036_103675
