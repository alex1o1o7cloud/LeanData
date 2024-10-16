import Mathlib

namespace NUMINAMATH_CALUDE_value_of_x_l15_1540

theorem value_of_x : ∀ (x y z w u : ℤ),
  x = y + 12 →
  y = z + 15 →
  z = w + 25 →
  w = u + 10 →
  u = 95 →
  x = 157 := by
sorry

end NUMINAMATH_CALUDE_value_of_x_l15_1540


namespace NUMINAMATH_CALUDE_sin_has_P_pi_property_P4_central_sym_monotone_P0_P3_implies_periodic_l15_1576

-- Definition of P(a) property
def has_P_property (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x, ∃ a, f (x + a) = f (-x)

-- Statement 1
theorem sin_has_P_pi_property : has_P_property Real.sin π :=
  sorry

-- Definition of central symmetry about a point
def centrally_symmetric (f : ℝ → ℝ) (p : ℝ × ℝ) : Prop :=
  ∀ x, f (2 * p.1 - x) = 2 * p.2 - f x

-- Definition of monotonically decreasing on an interval
def monotone_decreasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f y < f x

-- Definition of monotonically increasing on an interval
def monotone_increasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x < f y

-- Statement 3
theorem P4_central_sym_monotone (f : ℝ → ℝ) 
  (h1 : has_P_property f 4)
  (h2 : centrally_symmetric f (1, 0))
  (h3 : ∃ ε > 0, monotone_decreasing_on f (-1-ε) (-1+ε)) :
  monotone_decreasing_on f (-2) (-1) ∧ monotone_increasing_on f 1 2 :=
  sorry

-- Definition of periodic function
def periodic (f : ℝ → ℝ) : Prop :=
  ∃ p ≠ 0, ∀ x, f (x + p) = f x

-- Statement 4
theorem P0_P3_implies_periodic (f g : ℝ → ℝ)
  (h1 : f ≠ 0)
  (h2 : has_P_property f 0)
  (h3 : has_P_property f 3)
  (h4 : ∀ x₁ x₂, |f x₁ - f x₂| ≥ |g x₁ - g x₂|) :
  periodic g :=
  sorry

end NUMINAMATH_CALUDE_sin_has_P_pi_property_P4_central_sym_monotone_P0_P3_implies_periodic_l15_1576


namespace NUMINAMATH_CALUDE_azure_valley_skirts_l15_1519

/-- The number of skirts in Purple Valley -/
def purple_skirts : ℕ := 10

/-- The ratio of skirts in Purple Valley to Seafoam Valley -/
def purple_to_seafoam_ratio : ℚ := 1 / 4

/-- The ratio of skirts in Seafoam Valley to Azure Valley -/
def seafoam_to_azure_ratio : ℚ := 2 / 3

/-- The number of skirts in Azure Valley -/
def azure_skirts : ℕ := 60

theorem azure_valley_skirts :
  azure_skirts = (purple_skirts : ℚ) / (purple_to_seafoam_ratio * seafoam_to_azure_ratio) := by
  sorry

end NUMINAMATH_CALUDE_azure_valley_skirts_l15_1519


namespace NUMINAMATH_CALUDE_cost_per_box_l15_1523

/-- The cost per box for packaging the fine arts collection --/
theorem cost_per_box (box_length box_width box_height : ℝ)
  (total_volume min_total_cost : ℝ) :
  box_length = 20 ∧ box_width = 20 ∧ box_height = 15 ∧
  total_volume = 3060000 ∧ min_total_cost = 357 →
  (min_total_cost / (total_volume / (box_length * box_width * box_height))) = 0.70 := by
  sorry

end NUMINAMATH_CALUDE_cost_per_box_l15_1523


namespace NUMINAMATH_CALUDE_vacation_speed_problem_l15_1575

theorem vacation_speed_problem (distance1 distance2 time_diff : ℝ) 
  (h1 : distance1 = 100)
  (h2 : distance2 = 175)
  (h3 : time_diff = 3)
  (h4 : distance2 / speed = distance1 / speed + time_diff)
  (speed : ℝ) :
  speed = 25 := by
sorry

end NUMINAMATH_CALUDE_vacation_speed_problem_l15_1575


namespace NUMINAMATH_CALUDE_shaded_area_is_zero_l15_1501

/-- Rectangle JKLM with given dimensions and points -/
structure Rectangle where
  J : ℝ × ℝ
  K : ℝ × ℝ
  L : ℝ × ℝ
  M : ℝ × ℝ
  C : ℝ × ℝ
  B : ℝ × ℝ
  E : ℝ × ℝ
  F : ℝ × ℝ

/-- The conditions of the rectangle as given in the problem -/
def rectangle_conditions (r : Rectangle) : Prop :=
  r.J = (0, 0) ∧
  r.K = (4, 0) ∧
  r.L = (4, 5) ∧
  r.M = (0, 5) ∧
  r.C = (1.5, 5) ∧
  r.B = (4, 4) ∧
  r.E = r.J ∧
  r.F = r.M

/-- The area of the shaded region formed by the intersection of CF and BE -/
def shaded_area (r : Rectangle) : ℝ := sorry

/-- Theorem stating that the shaded area is 0 -/
theorem shaded_area_is_zero (r : Rectangle) (h : rectangle_conditions r) : 
  shaded_area r = 0 := by sorry

end NUMINAMATH_CALUDE_shaded_area_is_zero_l15_1501


namespace NUMINAMATH_CALUDE_people_who_left_l15_1542

theorem people_who_left (initial_people : ℕ) (remaining_people : ℕ) : 
  initial_people = 11 → remaining_people = 5 → initial_people - remaining_people = 6 := by
  sorry

end NUMINAMATH_CALUDE_people_who_left_l15_1542


namespace NUMINAMATH_CALUDE_daily_reading_goal_l15_1560

def sunday_pages : ℕ := 43
def monday_pages : ℕ := 65
def tuesday_pages : ℕ := 28
def wednesday_pages : ℕ := 0
def thursday_pages : ℕ := 70
def friday_pages : ℕ := 56
def saturday_pages : ℕ := 88

def total_pages : ℕ := sunday_pages + monday_pages + tuesday_pages + wednesday_pages + thursday_pages + friday_pages + saturday_pages

def days_in_week : ℕ := 7

theorem daily_reading_goal :
  (total_pages : ℚ) / days_in_week = 50 := by sorry

end NUMINAMATH_CALUDE_daily_reading_goal_l15_1560


namespace NUMINAMATH_CALUDE_five_twelve_thirteen_pythagorean_triple_l15_1599

/-- A Pythagorean triple is a set of three positive integers (a, b, c) that satisfy a² + b² = c² --/
def is_pythagorean_triple (a b c : ℕ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a^2 + b^2 = c^2

/-- The set (5, 12, 13) is a Pythagorean triple --/
theorem five_twelve_thirteen_pythagorean_triple :
  is_pythagorean_triple 5 12 13 := by
  sorry

end NUMINAMATH_CALUDE_five_twelve_thirteen_pythagorean_triple_l15_1599


namespace NUMINAMATH_CALUDE_sister_packs_l15_1531

def total_packs : ℕ := 13
def emily_packs : ℕ := 6

theorem sister_packs : total_packs - emily_packs = 7 := by
  sorry

end NUMINAMATH_CALUDE_sister_packs_l15_1531


namespace NUMINAMATH_CALUDE_line_parabola_properties_l15_1552

/-- A line represented by the equation y = ax + b -/
structure Line where
  a : ℝ
  b : ℝ

/-- A parabola represented by the equation y = ax^2 + b -/
structure Parabola where
  a : ℝ
  b : ℝ

/-- The theorem stating the properties of the line and parabola -/
theorem line_parabola_properties (l : Line) (p : Parabola)
    (h1 : l.a > 0)
    (h2 : p.a > 0)
    (h3 : l.b = p.b) :
    (∃ x y : ℝ, x = 0 ∧ y = l.b ∧ y = l.a * x + l.b ∧ y = p.a * x^2 + p.b) ∧
    (∀ x₁ x₂ : ℝ, x₁ < x₂ → l.a * x₁ + l.b < l.a * x₂ + l.b) ∧
    (∀ x₁ x₂ : ℝ, x₁ < x₂ → p.a * x₁^2 + p.b < p.a * x₂^2 + p.b) :=
  sorry


end NUMINAMATH_CALUDE_line_parabola_properties_l15_1552


namespace NUMINAMATH_CALUDE_cricket_team_age_problem_l15_1524

/-- Represents the age difference between the wicket keeper and the team average -/
def wicket_keeper_age_difference (team_size : ℕ) (team_average_age : ℕ) 
  (known_member_age : ℕ) (remaining_average_age : ℕ) : ℕ :=
  let total_age := team_size * team_average_age
  let remaining_total_age := (team_size - 2) * remaining_average_age
  let wicket_keeper_age := total_age - known_member_age - remaining_total_age
  wicket_keeper_age - team_average_age

theorem cricket_team_age_problem :
  wicket_keeper_age_difference 11 22 25 21 = 6 := by
  sorry

end NUMINAMATH_CALUDE_cricket_team_age_problem_l15_1524


namespace NUMINAMATH_CALUDE_toothpicks_12th_stage_l15_1588

/-- The number of toothpicks in the nth stage of the pattern -/
def toothpicks (n : ℕ) : ℕ := 3 * n

/-- Theorem: The 12th stage of the pattern contains 36 toothpicks -/
theorem toothpicks_12th_stage : toothpicks 12 = 36 := by
  sorry

end NUMINAMATH_CALUDE_toothpicks_12th_stage_l15_1588


namespace NUMINAMATH_CALUDE_schedule_theorem_l15_1572

/-- The number of periods in a day -/
def num_periods : ℕ := 7

/-- The number of courses to be scheduled -/
def num_courses : ℕ := 4

/-- Calculates the number of ways to schedule distinct courses in non-consecutive periods -/
def schedule_ways (periods : ℕ) (courses : ℕ) : ℕ :=
  (Nat.choose (periods - courses + 1) courses) * (Nat.factorial courses)

/-- Theorem stating that there are 1680 ways to schedule 4 distinct courses in a 7-period day
    with no two courses in consecutive periods -/
theorem schedule_theorem : 
  schedule_ways num_periods num_courses = 1680 := by
  sorry

end NUMINAMATH_CALUDE_schedule_theorem_l15_1572


namespace NUMINAMATH_CALUDE_tax_deduction_for_jacob_l15_1598

/-- Calculates the local tax deduction in cents given an hourly wage in dollars and a tax rate percentage. -/
def localTaxDeduction (hourlyWage : ℚ) (taxRate : ℚ) : ℚ :=
  hourlyWage * 100 * (taxRate / 100)

/-- Theorem stating that for an hourly wage of $25 and a 2% tax rate, the local tax deduction is 50 cents. -/
theorem tax_deduction_for_jacob :
  localTaxDeduction 25 2 = 50 := by
  sorry

end NUMINAMATH_CALUDE_tax_deduction_for_jacob_l15_1598


namespace NUMINAMATH_CALUDE_lines_parallel_to_same_line_are_parallel_l15_1580

-- Define a type for lines in space
variable (Line : Type)

-- Define a relation for parallel lines
variable (parallel : Line → Line → Prop)

-- Axiom: If two lines are parallel to the same line, they are parallel to each other
axiom parallel_transitivity :
  ∀ (l1 l2 l3 : Line), parallel l1 l3 → parallel l2 l3 → parallel l1 l2

-- Theorem: Two lines parallel to the same line are parallel to each other
theorem lines_parallel_to_same_line_are_parallel
  (l1 l2 l3 : Line) (h1 : parallel l1 l3) (h2 : parallel l2 l3) :
  parallel l1 l2 :=
sorry

end NUMINAMATH_CALUDE_lines_parallel_to_same_line_are_parallel_l15_1580


namespace NUMINAMATH_CALUDE_largest_number_with_nine_factors_l15_1593

/-- A function that returns the number of positive factors of a natural number -/
def num_factors (n : ℕ) : ℕ := sorry

/-- A function that checks if a number is less than 150 -/
def less_than_150 (n : ℕ) : Prop := n < 150

/-- The theorem stating that 100 is the largest number less than 150 with exactly 9 factors -/
theorem largest_number_with_nine_factors :
  (∀ m : ℕ, less_than_150 m → num_factors m = 9 → m ≤ 100) ∧
  (less_than_150 100 ∧ num_factors 100 = 9) :=
sorry

end NUMINAMATH_CALUDE_largest_number_with_nine_factors_l15_1593


namespace NUMINAMATH_CALUDE_binomial_expansion_coefficient_l15_1584

theorem binomial_expansion_coefficient (n : ℕ) : 
  (9 : ℕ) * (n.choose 2) = 54 → n = 4 := by
  sorry

end NUMINAMATH_CALUDE_binomial_expansion_coefficient_l15_1584


namespace NUMINAMATH_CALUDE_cricket_average_l15_1517

theorem cricket_average (innings : ℕ) (next_runs : ℕ) (increase : ℕ) (current_average : ℕ) : 
  innings = 12 → 
  next_runs = 178 → 
  increase = 10 → 
  (innings * current_average + next_runs) / (innings + 1) = current_average + increase →
  current_average = 48 := by
sorry

end NUMINAMATH_CALUDE_cricket_average_l15_1517


namespace NUMINAMATH_CALUDE_min_dihedral_angle_cube_l15_1532

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a plane in 3D space -/
structure Plane3D where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- A unit cube with vertices ABCD-A₁B₁C₁D₁ -/
structure UnitCube where
  A : Point3D
  B : Point3D
  C : Point3D
  D : Point3D
  A₁ : Point3D
  B₁ : Point3D
  C₁ : Point3D
  D₁ : Point3D

/-- A point P on edge AB of the cube -/
def P (cube : UnitCube) (t : ℝ) : Point3D :=
  { x := cube.A.x + t * (cube.B.x - cube.A.x),
    y := cube.A.y + t * (cube.B.y - cube.A.y),
    z := cube.A.z + t * (cube.B.z - cube.A.z) }

/-- The dihedral angle between two planes -/
def dihedralAngle (plane1 : Plane3D) (plane2 : Plane3D) : ℝ := sorry

/-- The plane PDB₁ -/
def planePDB₁ (cube : UnitCube) (p : Point3D) : Plane3D := sorry

/-- The plane ADD₁A₁ -/
def planeADD₁A₁ (cube : UnitCube) : Plane3D := sorry

theorem min_dihedral_angle_cube (cube : UnitCube) :
  ∃ (t : ℝ), t ∈ Set.Icc 0 1 ∧
    ∀ (s : ℝ), s ∈ Set.Icc 0 1 →
      dihedralAngle (planePDB₁ cube (P cube t)) (planeADD₁A₁ cube) ≤
      dihedralAngle (planePDB₁ cube (P cube s)) (planeADD₁A₁ cube) ∧
    dihedralAngle (planePDB₁ cube (P cube t)) (planeADD₁A₁ cube) = Real.arctan (Real.sqrt 2 / 2) := by
  sorry


end NUMINAMATH_CALUDE_min_dihedral_angle_cube_l15_1532


namespace NUMINAMATH_CALUDE_two_numbers_with_sum_or_diff_divisible_by_1000_l15_1567

theorem two_numbers_with_sum_or_diff_divisible_by_1000 (S : Finset ℕ) (h : S.card = 502) :
  ∃ (a b : ℕ), a ∈ S ∧ b ∈ S ∧ a ≠ b ∧ (1000 ∣ (a - b) ∨ 1000 ∣ (a + b)) := by
  sorry

end NUMINAMATH_CALUDE_two_numbers_with_sum_or_diff_divisible_by_1000_l15_1567


namespace NUMINAMATH_CALUDE_perpendicular_implies_parallel_skew_perpendicular_parallel_implies_perpendicular_l15_1565

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (perpendicular : Line → Plane → Prop)
variable (parallel : Line → Plane → Prop)
variable (plane_parallel : Plane → Plane → Prop)
variable (plane_perpendicular : Plane → Plane → Prop)
variable (line_parallel : Line → Line → Prop)
variable (skew : Line → Line → Prop)

-- Define non-coincidence for lines and planes
variable (non_coincident_lines : Line → Line → Prop)
variable (non_coincident_planes : Plane → Plane → Prop)

variable (m n : Line)
variable (α β γ : Plane)

-- Axioms
axiom non_coincident_mn : non_coincident_lines m n
axiom non_coincident_αβ : non_coincident_planes α β
axiom non_coincident_βγ : non_coincident_planes β γ
axiom non_coincident_αγ : non_coincident_planes α γ

-- Theorem 1
theorem perpendicular_implies_parallel 
  (h1 : perpendicular m α) (h2 : perpendicular m β) : 
  plane_parallel α β :=
sorry

-- Theorem 2
theorem skew_perpendicular_parallel_implies_perpendicular 
  (h1 : skew m n)
  (h2 : perpendicular m α) (h3 : parallel m β)
  (h4 : perpendicular n β) (h5 : parallel n α) :
  plane_perpendicular α β :=
sorry

end NUMINAMATH_CALUDE_perpendicular_implies_parallel_skew_perpendicular_parallel_implies_perpendicular_l15_1565


namespace NUMINAMATH_CALUDE_prob_two_defective_out_of_three_l15_1594

/-- The probability of selecting exactly 2 defective items out of 3 randomly chosen items
    from a set of 100 products containing 10 defective items. -/
theorem prob_two_defective_out_of_three (total_products : ℕ) (defective_items : ℕ) 
    (selected_items : ℕ) (h1 : total_products = 100) (h2 : defective_items = 10) 
    (h3 : selected_items = 3) :
  (Nat.choose defective_items 2 * Nat.choose (total_products - defective_items) 1) / 
  Nat.choose total_products selected_items = 27 / 1078 := by
  sorry

end NUMINAMATH_CALUDE_prob_two_defective_out_of_three_l15_1594


namespace NUMINAMATH_CALUDE_cubic_sum_theorem_l15_1521

theorem cubic_sum_theorem (a b c : ℝ) 
  (h1 : a^3 + b^3 + c^3 = 3*a*b*c)
  (h2 : a^3 + b^3 + c^3 = 6)
  (h3 : a^2 + b^2 + c^2 = 8) :
  a*b/(a+b) + b*c/(b+c) + c*a/(c+a) = -8 :=
sorry

end NUMINAMATH_CALUDE_cubic_sum_theorem_l15_1521


namespace NUMINAMATH_CALUDE_base_4_last_digit_379_l15_1503

def base_4_last_digit (n : ℕ) : ℕ :=
  n % 4

theorem base_4_last_digit_379 : base_4_last_digit 379 = 3 := by
  sorry

end NUMINAMATH_CALUDE_base_4_last_digit_379_l15_1503


namespace NUMINAMATH_CALUDE_length_of_AB_prime_l15_1525

-- Define the points
def A : ℝ × ℝ := (0, 10)
def B : ℝ × ℝ := (0, 15)
def C : ℝ × ℝ := (3, 9)

-- Define the condition for A' and B' to be on the line y = x
def on_diagonal (p : ℝ × ℝ) : Prop := p.1 = p.2

-- Define the condition for AA' and BB' to intersect at C
def intersect_at_C (A' B' : ℝ × ℝ) : Prop :=
  ∃ t₁ t₂ : ℝ,
    C = (A.1 + t₁ * (A'.1 - A.1), A.2 + t₁ * (A'.2 - A.2)) ∧
    C = (B.1 + t₂ * (B'.1 - B.1), B.2 + t₂ * (B'.2 - B.2))

-- State the theorem
theorem length_of_AB_prime : 
  ∃ A' B' : ℝ × ℝ,
    on_diagonal A' ∧ 
    on_diagonal B' ∧ 
    intersect_at_C A' B' ∧ 
    Real.sqrt ((A'.1 - B'.1)^2 + (A'.2 - B'.2)^2) = 2.5 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_length_of_AB_prime_l15_1525


namespace NUMINAMATH_CALUDE_fudge_ratio_is_one_to_three_l15_1530

-- Define the amount of fudge eaten by each person in ounces
def tomas_fudge : ℚ := 1.5 * 16
def boris_fudge : ℚ := 2 * 16
def total_fudge : ℚ := 64

-- Define Katya's fudge as the remaining amount
def katya_fudge : ℚ := total_fudge - tomas_fudge - boris_fudge

-- Define the ratio of Katya's fudge to Tomas's fudge
def fudge_ratio : ℚ × ℚ := (katya_fudge, tomas_fudge)

-- Theorem to prove
theorem fudge_ratio_is_one_to_three : fudge_ratio = (1, 3) := by
  sorry

end NUMINAMATH_CALUDE_fudge_ratio_is_one_to_three_l15_1530


namespace NUMINAMATH_CALUDE_stickers_per_page_l15_1579

theorem stickers_per_page (total_pages : ℕ) (total_stickers : ℕ) (stickers_per_page : ℕ) : 
  total_pages = 22 → 
  total_stickers = 220 → 
  total_stickers = total_pages * stickers_per_page → 
  stickers_per_page = 10 := by
sorry

end NUMINAMATH_CALUDE_stickers_per_page_l15_1579


namespace NUMINAMATH_CALUDE_sum_of_square_roots_lower_bound_l15_1500

theorem sum_of_square_roots_lower_bound (a b c d e : ℝ) 
  (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (hd : 0 < d) (he : 0 < e) : 
  Real.sqrt (a / (b + c + d + e)) + 
  Real.sqrt (b / (a + c + d + e)) + 
  Real.sqrt (c / (a + b + d + e)) + 
  Real.sqrt (d / (a + b + c + e)) + 
  Real.sqrt (e / (a + b + c + d)) ≥ 2 := by
sorry

end NUMINAMATH_CALUDE_sum_of_square_roots_lower_bound_l15_1500


namespace NUMINAMATH_CALUDE_floor_plus_s_eq_15_4_l15_1555

theorem floor_plus_s_eq_15_4 (s : ℝ) : 
  (⌊s⌋ : ℝ) + s = 15.4 → s = 7.4 := by
sorry

end NUMINAMATH_CALUDE_floor_plus_s_eq_15_4_l15_1555


namespace NUMINAMATH_CALUDE_correct_answers_for_given_score_l15_1596

/-- Represents a test with a scoring system and a student's performance. -/
structure Test where
  total_questions : ℕ
  correct_answers : ℕ
  score : ℤ

/-- Calculates the score based on correct and incorrect answers. -/
def calculate_score (test : Test) : ℤ :=
  (test.correct_answers : ℤ) - 2 * ((test.total_questions - test.correct_answers) : ℤ)

theorem correct_answers_for_given_score (test : Test) :
  test.total_questions = 100 ∧
  test.score = 64 ∧
  calculate_score test = test.score →
  test.correct_answers = 88 := by
  sorry

end NUMINAMATH_CALUDE_correct_answers_for_given_score_l15_1596


namespace NUMINAMATH_CALUDE_general_trigonometric_equation_l15_1566

theorem general_trigonometric_equation (θ : Real) : 
  Real.sin θ ^ 2 + Real.cos (θ + Real.pi / 6) ^ 2 + Real.sin θ * Real.cos (θ + Real.pi / 6) = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_general_trigonometric_equation_l15_1566


namespace NUMINAMATH_CALUDE_sixteen_black_squares_with_odd_numbers_l15_1597

/-- Represents a square on the chessboard -/
structure Square where
  row : Nat
  col : Nat
  number : Nat
  isBlack : Bool

/-- Represents a chessboard -/
def Chessboard := List Square

/-- Creates a standard 8x8 chessboard with alternating black and white squares,
    numbered from 1 to 64 left to right and top to bottom, with 1 on a black square -/
def createStandardChessboard : Chessboard := sorry

/-- Counts the number of black squares containing odd numbers on the chessboard -/
def countBlackSquaresWithOddNumbers (board : Chessboard) : Nat := sorry

/-- Theorem stating that there are exactly 16 black squares containing odd numbers
    on a standard 8x8 chessboard -/
theorem sixteen_black_squares_with_odd_numbers :
  ∀ (board : Chessboard),
    board = createStandardChessboard →
    countBlackSquaresWithOddNumbers board = 16 := by
  sorry

end NUMINAMATH_CALUDE_sixteen_black_squares_with_odd_numbers_l15_1597


namespace NUMINAMATH_CALUDE_track_width_l15_1518

theorem track_width (r₁ r₂ : ℝ) (h : 2 * Real.pi * r₁ - 2 * Real.pi * r₂ = 16 * Real.pi) : 
  r₁ - r₂ = 8 := by
sorry

end NUMINAMATH_CALUDE_track_width_l15_1518


namespace NUMINAMATH_CALUDE_part_one_part_two_l15_1571

-- Define the sets A and B
def A (a : ℝ) : Set ℝ := {x | a - 1 < x ∧ x < a + 1}
def B : Set ℝ := {x | x^2 - 4*x + 3 ≥ 0}

-- Theorem for part I
theorem part_one (a : ℝ) : (A a ∩ B = ∅) ∧ (A a ∪ B = Set.univ) → a = 2 := by
  sorry

-- Theorem for part II
theorem part_two (a : ℝ) : (∀ x, x ∈ A a → x ∈ B) → a ≤ 0 ∨ a ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_part_one_part_two_l15_1571


namespace NUMINAMATH_CALUDE_even_function_symmetry_l15_1527

/-- Definition of an even function -/
def EvenFunction (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f x = f (-x)

/-- Definition of an odd function -/
def OddFunction (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f x = -f (-x)

/-- The main theorem stating that only the third proposition is correct -/
theorem even_function_symmetry :
  (¬ ∀ f : ℝ → ℝ, EvenFunction f → ∃ y : ℝ, f 0 = y) ∧
  (¬ ∀ f : ℝ → ℝ, OddFunction f → f 0 = 0) ∧
  (∀ f : ℝ → ℝ, EvenFunction f → ∀ x : ℝ, f x = f (-x)) ∧
  (¬ ∀ f : ℝ → ℝ, (EvenFunction f ∧ OddFunction f) → ∀ x : ℝ, f x = 0) :=
sorry

end NUMINAMATH_CALUDE_even_function_symmetry_l15_1527


namespace NUMINAMATH_CALUDE_calculate_expression_l15_1535

theorem calculate_expression : (2 * Real.sqrt 48 - 3 * Real.sqrt (1/3)) / Real.sqrt 6 = 7 * Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l15_1535


namespace NUMINAMATH_CALUDE_square_area_from_vertices_l15_1546

/-- The area of a square with adjacent vertices at (1,5) and (4,-2) is 58 -/
theorem square_area_from_vertices : 
  let p1 : ℝ × ℝ := (1, 5)
  let p2 : ℝ × ℝ := (4, -2)
  let side_length := Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2)
  let area := side_length^2
  area = 58 := by sorry

end NUMINAMATH_CALUDE_square_area_from_vertices_l15_1546


namespace NUMINAMATH_CALUDE_successive_discounts_theorem_l15_1569

/-- The original price of the gadget -/
def original_price : ℝ := 350.00

/-- The first discount rate -/
def first_discount : ℝ := 0.10

/-- The second discount rate -/
def second_discount : ℝ := 0.12

/-- The final sale price as a percentage of the original price -/
def final_sale_percentage : ℝ := 0.792

theorem successive_discounts_theorem :
  let price_after_first_discount := original_price * (1 - first_discount)
  let final_price := price_after_first_discount * (1 - second_discount)
  (final_price / original_price) = final_sale_percentage := by sorry

end NUMINAMATH_CALUDE_successive_discounts_theorem_l15_1569


namespace NUMINAMATH_CALUDE_total_matches_played_l15_1506

theorem total_matches_played (home_wins rival_wins home_draws rival_draws : ℕ) : 
  home_wins = 3 →
  rival_wins = 2 * home_wins →
  home_draws = 4 →
  rival_draws = 4 →
  home_wins + rival_wins + home_draws + rival_draws = 17 :=
by
  sorry

end NUMINAMATH_CALUDE_total_matches_played_l15_1506


namespace NUMINAMATH_CALUDE_y_values_from_x_equation_l15_1551

theorem y_values_from_x_equation (x : ℝ) :
  x^2 + 5 * (x / (x - 3))^2 = 50 →
  ∃ y : ℝ, y = (x - 3)^2 * (x + 4) / (2*x - 5) ∧
    (∃ k : ℝ, (k = 5 + Real.sqrt 55 ∨ k = 5 - Real.sqrt 55 ∨
               k = 3 + 2 * Real.sqrt 6 ∨ k = 3 - 2 * Real.sqrt 6) ∧
              y = (k - 3)^2 * (k + 4) / (2*k - 5)) :=
by sorry

end NUMINAMATH_CALUDE_y_values_from_x_equation_l15_1551


namespace NUMINAMATH_CALUDE_max_profit_at_150_l15_1544

/-- Represents the total revenue function for the workshop --/
noncomputable def H (x : ℝ) : ℝ :=
  if 0 ≤ x ∧ x ≤ 200 then 400 * x - x^2 else 40000

/-- Represents the total cost function for the workshop --/
def total_cost (x : ℝ) : ℝ := 7500 + 100 * x

/-- Represents the profit function for the workshop --/
noncomputable def profit (x : ℝ) : ℝ := H x - total_cost x

/-- Theorem stating the maximum profit and corresponding production volume --/
theorem max_profit_at_150 :
  (∃ (x : ℝ), ∀ (y : ℝ), profit y ≤ profit x) ∧
  (∀ (x : ℝ), profit x ≤ 15000) ∧
  profit 150 = 15000 :=
sorry

end NUMINAMATH_CALUDE_max_profit_at_150_l15_1544


namespace NUMINAMATH_CALUDE_sqrt_27_minus_3_sqrt_one_third_l15_1534

theorem sqrt_27_minus_3_sqrt_one_third : 
  Real.sqrt 27 - 3 * Real.sqrt (1/3) = 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_27_minus_3_sqrt_one_third_l15_1534


namespace NUMINAMATH_CALUDE_product_of_roots_l15_1528

theorem product_of_roots (x : ℂ) :
  (2 * x^3 - 3 * x^2 - 10 * x + 18 = 0) →
  (∃ r₁ r₂ r₃ : ℂ, (x - r₁) * (x - r₂) * (x - r₃) = 2 * x^3 - 3 * x^2 - 10 * x + 18 ∧ r₁ * r₂ * r₃ = -9) :=
by sorry

end NUMINAMATH_CALUDE_product_of_roots_l15_1528


namespace NUMINAMATH_CALUDE_silver_dollar_problem_l15_1536

/-- The problem of calculating the total value of silver dollars -/
theorem silver_dollar_problem (x y : ℕ) : 
  -- Mr. Ha owns x silver dollars, which is 2/3 of Mr. Phung's amount
  x = (2 * y) / 3 →
  -- Mr. Phung has y silver dollars, which is 16 more than Mr. Chiu's amount
  y = 56 + 16 →
  -- The total value of all silver dollars is $483.75
  (x + y + 56 + (((x + y + 56) * 120) / 100)) * (5 / 4) = 96750 / 200 := by
  sorry

end NUMINAMATH_CALUDE_silver_dollar_problem_l15_1536


namespace NUMINAMATH_CALUDE_error_clock_correct_time_fraction_l15_1564

/-- Represents a 24-hour digital clock with a minute display error -/
structure ErrorClock where
  /-- The number of hours in a day -/
  hours_per_day : ℕ
  /-- The number of minutes in an hour -/
  minutes_per_hour : ℕ
  /-- The number of minutes with display error per hour -/
  error_minutes_per_hour : ℕ

/-- The fraction of the day the clock shows the correct time -/
def correct_time_fraction (clock : ErrorClock) : ℚ :=
  (clock.hours_per_day * (clock.minutes_per_hour - clock.error_minutes_per_hour)) /
  (clock.hours_per_day * clock.minutes_per_hour)

/-- Theorem stating the correct time fraction for the given clock -/
theorem error_clock_correct_time_fraction :
  let clock : ErrorClock := {
    hours_per_day := 24,
    minutes_per_hour := 60,
    error_minutes_per_hour := 1
  }
  correct_time_fraction clock = 59 / 60 := by
  sorry

end NUMINAMATH_CALUDE_error_clock_correct_time_fraction_l15_1564


namespace NUMINAMATH_CALUDE_probability_consecutive_numbers_l15_1550

/-- The number of balls -/
def n : ℕ := 6

/-- The number of balls to be drawn -/
def k : ℕ := 3

/-- The total number of ways to draw k balls from n balls -/
def total_ways : ℕ := Nat.choose n k

/-- The number of ways to draw k balls with exactly two consecutive numbers -/
def consecutive_ways : ℕ := 12

/-- The probability of drawing exactly two balls with consecutive numbers -/
def probability : ℚ := consecutive_ways / total_ways

theorem probability_consecutive_numbers : probability = 3/5 := by sorry

end NUMINAMATH_CALUDE_probability_consecutive_numbers_l15_1550


namespace NUMINAMATH_CALUDE_max_factors_of_power_l15_1510

def is_power_of_two_primes (b : ℕ) : Prop :=
  ∃ p q k l : ℕ, p.Prime ∧ q.Prime ∧ p ≠ q ∧ b = p^k * q^l

theorem max_factors_of_power (b n : ℕ) : 
  b > 0 → n > 0 → b ≤ 20 → n ≤ 20 → is_power_of_two_primes b →
  (∃ k : ℕ, k ≤ b^n ∧ (∀ m : ℕ, m ≤ b^n → Nat.card (Nat.divisors m) ≤ Nat.card (Nat.divisors k))) →
  Nat.card (Nat.divisors (b^n)) ≤ 441 :=
sorry

end NUMINAMATH_CALUDE_max_factors_of_power_l15_1510


namespace NUMINAMATH_CALUDE_largest_positive_integer_for_binary_operation_l15_1570

theorem largest_positive_integer_for_binary_operation
  (x : ℝ) (h : x > -8) :
  (∀ n : ℕ+, n - 5 * n < x) ∧
  (∀ m : ℕ+, m > 2 → ¬(m - 5 * m < x)) :=
sorry

end NUMINAMATH_CALUDE_largest_positive_integer_for_binary_operation_l15_1570


namespace NUMINAMATH_CALUDE_hundredth_decimal_is_9_l15_1513

/-- The decimal expansion of 10/11 -/
def decimal_expansion_10_11 : ℕ → ℕ := 
  fun n => if n % 2 = 0 then 0 else 9

/-- The 100th decimal digit in the expansion of 10/11 -/
def hundredth_decimal : ℕ := decimal_expansion_10_11 100

theorem hundredth_decimal_is_9 : hundredth_decimal = 9 := by sorry

end NUMINAMATH_CALUDE_hundredth_decimal_is_9_l15_1513


namespace NUMINAMATH_CALUDE_school_robe_cost_l15_1520

/-- Calculates the cost of robes based on the given pricing tiers --/
def robeCost (n : ℕ) : ℚ :=
  if n ≤ 10 then 3 * n
  else if n ≤ 20 then 2.5 * n
  else 2 * n

/-- Calculates the total cost including alterations, customization, and sales tax --/
def totalCost (singers : ℕ) (existingRobes : ℕ) (alterationCost : ℚ) (customizationCost : ℚ) (salesTax : ℚ) : ℚ :=
  let neededRobes := singers - existingRobes
  let baseCost := robeCost neededRobes
  let additionalCost := (alterationCost + customizationCost) * neededRobes
  let subtotal := baseCost + additionalCost
  subtotal * (1 + salesTax)

theorem school_robe_cost :
  totalCost 30 12 1.5 0.75 0.08 = 92.34 :=
sorry

end NUMINAMATH_CALUDE_school_robe_cost_l15_1520


namespace NUMINAMATH_CALUDE_students_in_cars_l15_1512

theorem students_in_cars (total_students : ℕ) (num_buses : ℕ) (students_per_bus : ℕ) :
  total_students = 396 →
  num_buses = 7 →
  students_per_bus = 56 →
  total_students - (num_buses * students_per_bus) = 4 :=
by sorry

end NUMINAMATH_CALUDE_students_in_cars_l15_1512


namespace NUMINAMATH_CALUDE_mod_eight_power_difference_l15_1556

theorem mod_eight_power_difference : (47^2023 - 22^2023) % 8 = 1 := by
  sorry

end NUMINAMATH_CALUDE_mod_eight_power_difference_l15_1556


namespace NUMINAMATH_CALUDE_king_queen_ages_l15_1577

theorem king_queen_ages : ∃ (K Q : ℕ),
  -- The king is twice as old as the queen was when the king was as old as the queen is now
  K = 2 * (Q - (K - Q)) ∧
  -- When the queen is as old as the king is now, their combined ages will be 63 years
  Q + (K - (K - Q)) + K = 63 ∧
  -- The king's age is 28 and the queen's age is 21
  K = 28 ∧ Q = 21 := by
sorry

end NUMINAMATH_CALUDE_king_queen_ages_l15_1577


namespace NUMINAMATH_CALUDE_power_four_times_power_four_l15_1505

theorem power_four_times_power_four (x : ℝ) : x^4 * x^4 = x^8 := by
  sorry

end NUMINAMATH_CALUDE_power_four_times_power_four_l15_1505


namespace NUMINAMATH_CALUDE_half_abs_diff_squares_plus_five_l15_1573

theorem half_abs_diff_squares_plus_five : 
  (|20^2 - 12^2| / 2 : ℝ) + 5 = 133 := by sorry

end NUMINAMATH_CALUDE_half_abs_diff_squares_plus_five_l15_1573


namespace NUMINAMATH_CALUDE_motel_billing_solution_l15_1557

/-- Represents the motel's billing system -/
structure MotelBilling where
  flatFee : ℝ  -- Flat fee for the first night
  nightlyRate : ℝ  -- Fixed rate for subsequent nights

/-- Calculates the total cost for a stay -/
def totalCost (billing : MotelBilling) (nights : ℕ) : ℝ :=
  billing.flatFee + billing.nightlyRate * (nights - 1 : ℝ) -
    if nights > 4 then 25 else 0

/-- The motel billing system satisfies the given conditions -/
theorem motel_billing_solution :
  ∃ (billing : MotelBilling),
    totalCost billing 4 = 215 ∧
    totalCost billing 7 = 360 ∧
    billing.flatFee = 45 := by
  sorry


end NUMINAMATH_CALUDE_motel_billing_solution_l15_1557


namespace NUMINAMATH_CALUDE_shirt_discount_l15_1545

/-- Given a shirt with an original price and a sale price, calculate the discount amount. -/
def discount (original_price sale_price : ℕ) : ℕ :=
  original_price - sale_price

/-- Theorem stating that for a shirt with an original price of $22 and a sale price of $16, 
    the discount amount is $6. -/
theorem shirt_discount :
  let original_price : ℕ := 22
  let sale_price : ℕ := 16
  discount original_price sale_price = 6 := by
sorry

end NUMINAMATH_CALUDE_shirt_discount_l15_1545


namespace NUMINAMATH_CALUDE_complex_fraction_equality_l15_1583

theorem complex_fraction_equality : ((-1 : ℂ) + 3*I) / (1 + I) = 1 + 2*I := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_equality_l15_1583


namespace NUMINAMATH_CALUDE_domain_shift_l15_1548

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the domain of f
def domain_f : Set ℝ := Set.Icc 0 1

-- State the theorem
theorem domain_shift :
  (∀ x, f x ≠ 0 → x ∈ domain_f) →
  (∀ x, f (x + 2) ≠ 0 → x ∈ Set.Icc (-2) (-1)) :=
sorry

end NUMINAMATH_CALUDE_domain_shift_l15_1548


namespace NUMINAMATH_CALUDE_circle_area_from_circumference_l15_1547

/-- Given a circle with circumference 36 cm, its area is 324/π square centimeters. -/
theorem circle_area_from_circumference :
  ∀ (r : ℝ), 2 * π * r = 36 → π * r^2 = 324 / π := by
  sorry

end NUMINAMATH_CALUDE_circle_area_from_circumference_l15_1547


namespace NUMINAMATH_CALUDE_q_squared_minus_one_div_24_l15_1522

/-- The largest prime number with 2023 digits -/
def q : ℕ := sorry

/-- q is prime -/
axiom q_prime : Nat.Prime q

/-- q has 2023 digits -/
axiom q_digits : q ≥ 10^2022 ∧ q < 10^2023

/-- q is the largest prime with 2023 digits -/
axiom q_largest : ∀ p, Nat.Prime p → (p ≥ 10^2022 ∧ p < 10^2023) → p ≤ q

/-- The theorem to be proved -/
theorem q_squared_minus_one_div_24 : 24 ∣ (q^2 - 1) := by sorry

end NUMINAMATH_CALUDE_q_squared_minus_one_div_24_l15_1522


namespace NUMINAMATH_CALUDE_quadratic_rewrite_sum_l15_1533

theorem quadratic_rewrite_sum (a b c : ℤ) : 
  (49 : ℤ) * x^2 + 70 * x - 121 = 0 ↔ (a * x + b)^2 = c ∧ 
  a > 0 ∧ 
  a + b + c = -134 := by
sorry

end NUMINAMATH_CALUDE_quadratic_rewrite_sum_l15_1533


namespace NUMINAMATH_CALUDE_percentage_of_cat_owners_l15_1559

/-- The percentage of students who own cats in a school survey -/
theorem percentage_of_cat_owners (total_students : ℕ) (cat_owners : ℕ) 
  (h1 : total_students = 500) 
  (h2 : cat_owners = 75) : 
  (cat_owners : ℝ) / total_students * 100 = 15 := by
  sorry

end NUMINAMATH_CALUDE_percentage_of_cat_owners_l15_1559


namespace NUMINAMATH_CALUDE_sum_of_integers_l15_1595

theorem sum_of_integers (x y : ℕ+) (h1 : x.val - y.val = 8) (h2 : x.val * y.val = 56) :
  x.val + y.val = 12 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_integers_l15_1595


namespace NUMINAMATH_CALUDE_max_choir_members_satisfies_conditions_max_choir_members_is_maximum_l15_1561

/-- The maximum number of choir members satisfying the given conditions -/
def max_choir_members : ℕ := 54

/-- Predicate to check if a number satisfies the square formation condition -/
def satisfies_square_condition (n : ℕ) : Prop :=
  ∃ x : ℕ, n = x^2 + 11

/-- Predicate to check if a number satisfies the rectangle formation condition -/
def satisfies_rectangle_condition (n : ℕ) : Prop :=
  ∃ y : ℕ, n = y * (y + 3)

/-- Theorem stating that max_choir_members satisfies both conditions -/
theorem max_choir_members_satisfies_conditions :
  satisfies_square_condition max_choir_members ∧
  satisfies_rectangle_condition max_choir_members :=
by sorry

/-- Theorem stating that max_choir_members is the maximum number satisfying both conditions -/
theorem max_choir_members_is_maximum :
  ∀ n : ℕ, 
    satisfies_square_condition n ∧ 
    satisfies_rectangle_condition n → 
    n ≤ max_choir_members :=
by sorry

end NUMINAMATH_CALUDE_max_choir_members_satisfies_conditions_max_choir_members_is_maximum_l15_1561


namespace NUMINAMATH_CALUDE_month_with_conditions_has_30_days_l15_1538

/-- Represents the days of the week -/
inductive DayOfWeek
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

/-- Represents a month with its number of days and day counts -/
structure Month where
  days : Nat
  dayCounts : DayOfWeek → Nat

/-- Definition of a valid month -/
def validMonth (m : Month) : Prop :=
  (m.days ≥ 28 ∧ m.days ≤ 31) ∧
  (∀ d : DayOfWeek, m.dayCounts d = 4 ∨ m.dayCounts d = 5)

/-- The condition of more Mondays than Tuesdays -/
def moreMondays (m : Month) : Prop :=
  m.dayCounts DayOfWeek.Monday > m.dayCounts DayOfWeek.Tuesday

/-- The condition of fewer Saturdays than Sundays -/
def fewerSaturdays (m : Month) : Prop :=
  m.dayCounts DayOfWeek.Saturday < m.dayCounts DayOfWeek.Sunday

theorem month_with_conditions_has_30_days (m : Month) 
  (hValid : validMonth m) 
  (hMondays : moreMondays m) 
  (hSaturdays : fewerSaturdays m) : 
  m.days = 30 := by
  sorry

end NUMINAMATH_CALUDE_month_with_conditions_has_30_days_l15_1538


namespace NUMINAMATH_CALUDE_brick_length_l15_1526

/-- The surface area of a rectangular prism -/
def surface_area (l w h : ℝ) : ℝ := 2 * (l * w + l * h + w * h)

/-- Theorem: Given a rectangular prism with width 4, height 3, and surface area 164, its length is 10 -/
theorem brick_length (w h SA : ℝ) (hw : w = 4) (hh : h = 3) (hSA : SA = 164) :
  ∃ l : ℝ, surface_area l w h = SA ∧ l = 10 :=
sorry

end NUMINAMATH_CALUDE_brick_length_l15_1526


namespace NUMINAMATH_CALUDE_inequality_proof_l15_1589

theorem inequality_proof (p : ℝ) (h1 : 18 * p < 10) (h2 : p > 0.5) : 0.5 < p ∧ p < 5/9 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l15_1589


namespace NUMINAMATH_CALUDE_cube_root_of_sum_l15_1585

theorem cube_root_of_sum (x y : ℝ) : 
  (Real.sqrt (x - 1) + (y + 2)^2 = 0) → 
  (x + y)^(1/3 : ℝ) = -1 := by
sorry

end NUMINAMATH_CALUDE_cube_root_of_sum_l15_1585


namespace NUMINAMATH_CALUDE_cruise_ship_problem_l15_1504

/-- Cruise ship problem -/
theorem cruise_ship_problem 
  (distance : ℝ) 
  (x : ℝ) 
  (k : ℝ) 
  (h1 : distance = 5)
  (h2 : 20 ≤ x ∧ x ≤ 50)
  (h3 : 1/15 ≤ k ∧ k ≤ 1/5)
  (h4 : x/40 - k = 5/8) :
  (∃ (x_range : Set ℝ), x_range = {x | 20 ≤ x ∧ x ≤ 40} ∧ 
    ∀ y ∈ x_range, y/40 - k + 1/y ≤ 9/10) ∧
  (∀ y : ℝ, 20 ≤ y ∧ y ≤ 50 →
    (1/15 ≤ k ∧ k < 1/10 → 
      5/y * (y/40 - k + 1/y) ≥ (1 - 10*k^2) / 8) ∧
    (1/10 ≤ k ∧ k ≤ 1/5 → 
      5/y * (y/40 - k + 1/y) ≥ (11 - 20*k) / 80)) :=
sorry

end NUMINAMATH_CALUDE_cruise_ship_problem_l15_1504


namespace NUMINAMATH_CALUDE_expression_evaluation_l15_1590

theorem expression_evaluation (a : ℚ) (h : a = 4/3) : (4*a^2 - 12*a + 9)*(3*a - 4) = 0 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l15_1590


namespace NUMINAMATH_CALUDE_triangle_side_ratio_l15_1514

/-- Given a triangle ABC with sides a ≤ b ≤ c, prove that the ratio of the lengths of the sides satisfies 2b² = a² + c² -/
theorem triangle_side_ratio (a b c : ℝ) (h1 : 0 < a) (h2 : 0 < b) (h3 : 0 < c) 
  (h4 : a ≤ b) (h5 : b ≤ c) : 2 * b^2 = a^2 + c^2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_ratio_l15_1514


namespace NUMINAMATH_CALUDE_paint_usage_l15_1562

theorem paint_usage (mary_paint mike_paint sun_paint total_paint : ℝ) 
  (h1 : mike_paint = mary_paint + 2)
  (h2 : sun_paint = 5)
  (h3 : total_paint = 13)
  (h4 : mary_paint + mike_paint + sun_paint = total_paint) :
  mary_paint = 3 := by
sorry

end NUMINAMATH_CALUDE_paint_usage_l15_1562


namespace NUMINAMATH_CALUDE_tangent_property_reasoning_l15_1574

-- Define the types of geometric objects
inductive GeometricObject
| Circle
| Line
| Sphere
| Plane

-- Define the types of reasoning
inductive ReasoningType
| Inductive
| Deductive
| Analogical
| Transitive

-- Define the property of perpendicularity for 2D and 3D cases
def isPerpendicular (obj1 obj2 : GeometricObject) : Prop :=
  match obj1, obj2 with
  | GeometricObject.Line, GeometricObject.Line => true
  | GeometricObject.Line, GeometricObject.Plane => true
  | _, _ => false

-- Define the tangent property for 2D case
def tangentProperty2D (circle : GeometricObject) (tangentLine : GeometricObject) (centerToTangentLine : GeometricObject) : Prop :=
  circle = GeometricObject.Circle ∧
  tangentLine = GeometricObject.Line ∧
  centerToTangentLine = GeometricObject.Line ∧
  isPerpendicular tangentLine centerToTangentLine

-- Define the tangent property for 3D case
def tangentProperty3D (sphere : GeometricObject) (tangentPlane : GeometricObject) (centerToTangentLine : GeometricObject) : Prop :=
  sphere = GeometricObject.Sphere ∧
  tangentPlane = GeometricObject.Plane ∧
  centerToTangentLine = GeometricObject.Line ∧
  isPerpendicular tangentPlane centerToTangentLine

-- Theorem statement
theorem tangent_property_reasoning :
  (∃ (circle tangentLine centerToTangentLine : GeometricObject),
    tangentProperty2D circle tangentLine centerToTangentLine) →
  (∃ (sphere tangentPlane centerToTangentLine : GeometricObject),
    tangentProperty3D sphere tangentPlane centerToTangentLine) →
  (∀ (r : ReasoningType), r = ReasoningType.Analogical) :=
by sorry

end NUMINAMATH_CALUDE_tangent_property_reasoning_l15_1574


namespace NUMINAMATH_CALUDE_train_length_calculation_l15_1537

/-- Calculates the length of a train given the speeds of two trains, the time they take to cross each other, and the length of the other train. -/
theorem train_length_calculation (v1 v2 : ℝ) (t cross_time : ℝ) (l2 : ℝ) :
  v1 = 60 →
  v2 = 40 →
  cross_time = 12.239020878329734 →
  l2 = 200 →
  (v1 + v2) * 1000 / 3600 * cross_time - l2 = 140 :=
by sorry

end NUMINAMATH_CALUDE_train_length_calculation_l15_1537


namespace NUMINAMATH_CALUDE_x_plus_y_equals_two_l15_1581

theorem x_plus_y_equals_two (x y : ℝ) (h : |x - 6| + (y + 4)^2 = 0) : x + y = 2 := by
  sorry

end NUMINAMATH_CALUDE_x_plus_y_equals_two_l15_1581


namespace NUMINAMATH_CALUDE_edwards_initial_spending_l15_1508

/-- Given Edward's initial balance, additional spending, and final balance,
    prove the amount he spent initially. -/
theorem edwards_initial_spending
  (initial_balance : ℕ)
  (additional_spending : ℕ)
  (final_balance : ℕ)
  (h1 : initial_balance = 34)
  (h2 : additional_spending = 8)
  (h3 : final_balance = 17)
  : initial_balance - additional_spending - final_balance = 9 := by
  sorry

#check edwards_initial_spending

end NUMINAMATH_CALUDE_edwards_initial_spending_l15_1508


namespace NUMINAMATH_CALUDE_marie_erasers_l15_1591

theorem marie_erasers (initial : ℕ) (lost : ℕ) (final : ℕ) : 
  initial = 95 → lost = 42 → final = initial - lost → final = 53 := by
sorry

end NUMINAMATH_CALUDE_marie_erasers_l15_1591


namespace NUMINAMATH_CALUDE_bart_earnings_l15_1558

/-- Calculates the total earnings for Bart's survey work over five days --/
theorem bart_earnings (
  monday_rate : ℚ)
  (monday_questions : ℕ)
  (monday_surveys : ℕ)
  (tuesday_rate : ℚ)
  (tuesday_questions : ℕ)
  (tuesday_surveys : ℕ)
  (wednesday_rate : ℚ)
  (wednesday_questions : ℕ)
  (wednesday_surveys : ℕ)
  (thursday_rate : ℚ)
  (thursday_questions : ℕ)
  (thursday_surveys : ℕ)
  (friday_rate : ℚ)
  (friday_questions : ℕ)
  (friday_surveys : ℕ)
  (h1 : monday_rate = 20/100)
  (h2 : monday_questions = 10)
  (h3 : monday_surveys = 3)
  (h4 : tuesday_rate = 25/100)
  (h5 : tuesday_questions = 12)
  (h6 : tuesday_surveys = 4)
  (h7 : wednesday_rate = 10/100)
  (h8 : wednesday_questions = 15)
  (h9 : wednesday_surveys = 5)
  (h10 : thursday_rate = 15/100)
  (h11 : thursday_questions = 8)
  (h12 : thursday_surveys = 6)
  (h13 : friday_rate = 30/100)
  (h14 : friday_questions = 20)
  (h15 : friday_surveys = 2) :
  monday_rate * monday_questions * monday_surveys +
  tuesday_rate * tuesday_questions * tuesday_surveys +
  wednesday_rate * wednesday_questions * wednesday_surveys +
  thursday_rate * thursday_questions * thursday_surveys +
  friday_rate * friday_questions * friday_surveys = 447/10 := by
  sorry

end NUMINAMATH_CALUDE_bart_earnings_l15_1558


namespace NUMINAMATH_CALUDE_equal_intercept_line_correct_l15_1553

/-- A line passing through point (2, 3) with equal intercepts on both axes -/
def equal_intercept_line (x y : ℝ) : Prop :=
  x + y - 5 = 0

theorem equal_intercept_line_correct :
  -- The line passes through (2, 3)
  equal_intercept_line 2 3 ∧
  -- The line has equal intercepts on both axes
  ∃ a : ℝ, a ≠ 0 ∧ equal_intercept_line a 0 ∧ equal_intercept_line 0 a :=
by
  sorry

end NUMINAMATH_CALUDE_equal_intercept_line_correct_l15_1553


namespace NUMINAMATH_CALUDE_coin_problem_l15_1502

/-- Given a total sum in paise, the number of 20 paise coins, and that the remaining sum is made up of 25 paise coins, calculate the total number of coins. -/
def total_coins (total_sum : ℕ) (coins_20 : ℕ) : ℕ :=
  let sum_20 := coins_20 * 20
  let sum_25 := total_sum - sum_20
  let coins_25 := sum_25 / 25
  coins_20 + coins_25

/-- Theorem stating that given the specific conditions, the total number of coins is 334. -/
theorem coin_problem : total_coins 7100 250 = 334 := by
  sorry

end NUMINAMATH_CALUDE_coin_problem_l15_1502


namespace NUMINAMATH_CALUDE_green_ducks_percentage_l15_1592

theorem green_ducks_percentage (smaller_pond : ℕ) (larger_pond : ℕ) 
  (larger_pond_green_percent : ℝ) (total_green_percent : ℝ) :
  smaller_pond = 30 →
  larger_pond = 50 →
  larger_pond_green_percent = 12 →
  total_green_percent = 15 →
  (smaller_pond_green_percent : ℝ) * smaller_pond / 100 + 
    larger_pond_green_percent * larger_pond / 100 = 
    total_green_percent * (smaller_pond + larger_pond) / 100 →
  smaller_pond_green_percent = 20 :=
by
  sorry

end NUMINAMATH_CALUDE_green_ducks_percentage_l15_1592


namespace NUMINAMATH_CALUDE_square_sum_divisibility_problem_l15_1515

theorem square_sum_divisibility_problem :
  ∃ a b : ℕ, a^2 + b^2 = 2018 ∧ 7 ∣ (a + b) ∧
  ((a = 43 ∧ b = 13) ∨ (a = 13 ∧ b = 43)) ∧
  (∀ x y : ℕ, x^2 + y^2 = 2018 ∧ 7 ∣ (x + y) → (x = 43 ∧ y = 13) ∨ (x = 13 ∧ y = 43)) :=
by sorry

end NUMINAMATH_CALUDE_square_sum_divisibility_problem_l15_1515


namespace NUMINAMATH_CALUDE_unique_three_digit_number_l15_1516

/-- Represents a three-digit number -/
structure ThreeDigitNumber where
  value : ℕ
  is_three_digit : 100 ≤ value ∧ value < 1000

/-- Returns the two-digit number formed by removing the first digit -/
def remove_first_digit (n : ThreeDigitNumber) : ℕ :=
  n.value % 100

/-- Checks if a three-digit number satisfies the division condition -/
def satisfies_division_condition (n : ThreeDigitNumber) : Prop :=
  let two_digit := remove_first_digit n
  n.value / two_digit = 8 ∧ n.value % two_digit = 6

theorem unique_three_digit_number :
  ∃! n : ThreeDigitNumber, satisfies_division_condition n ∧ n.value = 342 :=
sorry

end NUMINAMATH_CALUDE_unique_three_digit_number_l15_1516


namespace NUMINAMATH_CALUDE_lyra_remaining_budget_l15_1586

/-- Calculates the remaining budget after Lyra's purchases --/
theorem lyra_remaining_budget (budget : ℝ) (chicken_price : ℝ) (beef_price : ℝ) (beef_weight : ℝ)
  (soup_price : ℝ) (soup_cans : ℕ) (milk_price : ℝ) (milk_discount : ℝ) :
  budget = 80 →
  chicken_price = 12 →
  beef_price = 3 →
  beef_weight = 4.5 →
  soup_price = 2 →
  soup_cans = 3 →
  milk_price = 4 →
  milk_discount = 0.1 →
  budget - (chicken_price + beef_price * beef_weight + 
    (soup_price * ↑soup_cans / 2) + milk_price * (1 - milk_discount)) = 47.9 := by
  sorry

#eval (80 : ℚ) - (12 + 3 * (9/2) + (2 * 3 / 2) + 4 * (1 - 1/10))

end NUMINAMATH_CALUDE_lyra_remaining_budget_l15_1586


namespace NUMINAMATH_CALUDE_rectangle_length_proof_l15_1582

theorem rectangle_length_proof (width area : ℝ) (h1 : width = 3 * Real.sqrt 2) (h2 : area = 18 * Real.sqrt 6) :
  area / width = 6 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_length_proof_l15_1582


namespace NUMINAMATH_CALUDE_digit_reversal_l15_1554

theorem digit_reversal (n : ℕ) : 
  let B := n^2 + 1
  (n^2 * (n^2 + 2)^2 = 1 * B^3 + 0 * B^2 + (B - 2) * B + (B - 1)) ∧
  (n^4 * (n^2 + 2)^2 = 1 * B^3 + (B - 2) * B^2 + 0 * B + (B - 1)) := by
sorry

end NUMINAMATH_CALUDE_digit_reversal_l15_1554


namespace NUMINAMATH_CALUDE_fraction_equality_l15_1563

theorem fraction_equality (a : ℕ+) : (a : ℚ) / ((a : ℚ) + 36) = 775 / 1000 → a = 124 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l15_1563


namespace NUMINAMATH_CALUDE_intersection_slope_range_l15_1509

/-- Given two points P and Q in the Cartesian plane, and a linear function y = kx - 1
    that intersects the extension of line segment PQ (excluding Q),
    prove that the range of k is (1/3, 3/2). -/
theorem intersection_slope_range (P Q : ℝ × ℝ) (k : ℝ) : 
  P = (-1, 1) →
  Q = (2, 2) →
  (∃ x y : ℝ, y = k * x - 1 ∧ (y - 1) / (x + 1) = (2 - 1) / (2 + 1) ∧ (x, y) ≠ Q) →
  1/3 < k ∧ k < 3/2 :=
by sorry

end NUMINAMATH_CALUDE_intersection_slope_range_l15_1509


namespace NUMINAMATH_CALUDE_max_moves_l15_1529

def S : Set (ℕ × ℕ) := {p | p.1 ≥ 1 ∧ p.1 ≤ 2022 ∧ p.2 ≥ 1 ∧ p.2 ≤ 2022}

def isGoodRectangle (r : (ℕ × ℕ) × (ℕ × ℕ)) : Prop :=
  let ((x1, y1), (x2, y2)) := r
  x1 ≥ 1 ∧ x1 ≤ 2022 ∧ y1 ≥ 1 ∧ y1 ≤ 2022 ∧
  x2 ≥ 1 ∧ x2 ≤ 2022 ∧ y2 ≥ 1 ∧ y2 ≤ 2022 ∧
  x1 < x2 ∧ y1 < y2

def Move := (ℕ × ℕ) × (ℕ × ℕ)

def isValidMove (m : Move) : Prop := isGoodRectangle m

theorem max_moves : ∃ (moves : List Move), 
  (∀ m ∈ moves, isValidMove m) ∧ 
  moves.length = 1011^4 ∧ 
  (∀ (other_moves : List Move), (∀ m ∈ other_moves, isValidMove m) → other_moves.length ≤ 1011^4) := by
  sorry

end NUMINAMATH_CALUDE_max_moves_l15_1529


namespace NUMINAMATH_CALUDE_sequence_bound_l15_1511

theorem sequence_bound (a : ℕ → ℝ) (c : ℝ) 
  (h1 : ∀ i : ℕ, i > 0 → 0 ≤ a i ∧ a i ≤ c)
  (h2 : ∀ i j : ℕ, i > 0 → j > 0 → i ≠ j → |a i - a j| ≥ 1 / (i + j)) :
  c ≥ 1 := by
sorry

end NUMINAMATH_CALUDE_sequence_bound_l15_1511


namespace NUMINAMATH_CALUDE_parabola_symmetry_transform_l15_1539

/-- Given a parabola with equation y = -2(x+1)^2 + 3, prove that its transformation
    by symmetry about the line y = 1 results in the equation y = 2(x+1)^2 - 1. -/
theorem parabola_symmetry_transform (x y : ℝ) :
  (y = -2 * (x + 1)^2 + 3) →
  (∃ (y' : ℝ), y' = 2 * (x + 1)^2 - 1 ∧ 
    (∀ (p q : ℝ × ℝ), (p.2 = -2 * (p.1 + 1)^2 + 3 ∧ q.2 = y') → 
      (p.1 = q.1 ∧ p.2 + q.2 = 2))) :=
by sorry

end NUMINAMATH_CALUDE_parabola_symmetry_transform_l15_1539


namespace NUMINAMATH_CALUDE_number_difference_l15_1549

theorem number_difference (x y : ℝ) (h1 : x + y = 30) (h2 : x * y = 216) : 
  |x - y| = 6 := by sorry

end NUMINAMATH_CALUDE_number_difference_l15_1549


namespace NUMINAMATH_CALUDE_benny_march_savings_l15_1578

/-- The amount of money Benny added to his piggy bank in January -/
def january_amount : ℕ := 19

/-- The amount of money Benny added to his piggy bank in February -/
def february_amount : ℕ := 19

/-- The total amount of money in Benny's piggy bank by the end of March -/
def march_total : ℕ := 46

/-- The amount of money Benny added to his piggy bank in March -/
def march_amount : ℕ := march_total - (january_amount + february_amount)

/-- Proof that Benny added $8 to his piggy bank in March -/
theorem benny_march_savings : march_amount = 8 := by
  sorry

end NUMINAMATH_CALUDE_benny_march_savings_l15_1578


namespace NUMINAMATH_CALUDE_davids_english_marks_l15_1568

theorem davids_english_marks :
  let math_marks : ℕ := 65
  let physics_marks : ℕ := 82
  let chemistry_marks : ℕ := 67
  let biology_marks : ℕ := 85
  let average_marks : ℕ := 78
  let num_subjects : ℕ := 5
  let total_marks : ℕ := average_marks * num_subjects
  let known_marks : ℕ := math_marks + physics_marks + chemistry_marks + biology_marks
  let english_marks : ℕ := total_marks - known_marks
  english_marks = 91 := by
sorry

end NUMINAMATH_CALUDE_davids_english_marks_l15_1568


namespace NUMINAMATH_CALUDE_mysoon_ornament_collection_l15_1541

/-- The number of ornaments in Mysoon's collection -/
def total_ornaments : ℕ := 20

/-- The number of handmade ornaments -/
def handmade_ornaments : ℕ := total_ornaments / 6 + 10

/-- The number of handmade antique ornaments -/
def handmade_antique_ornaments : ℕ := total_ornaments / 3

theorem mysoon_ornament_collection :
  (handmade_ornaments = total_ornaments / 6 + 10) ∧
  (handmade_antique_ornaments = handmade_ornaments / 2) ∧
  (handmade_antique_ornaments = total_ornaments / 3) →
  total_ornaments = 20 := by
sorry

end NUMINAMATH_CALUDE_mysoon_ornament_collection_l15_1541


namespace NUMINAMATH_CALUDE_sum_of_squares_l15_1507

theorem sum_of_squares (x y z : ℤ) 
  (sum_eq : x + y + z = 3) 
  (sum_cubes_eq : x^3 + y^3 + z^3 = 3) : 
  x^2 + y^2 + z^2 = 3 ∨ x^2 + y^2 + z^2 = 57 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_l15_1507


namespace NUMINAMATH_CALUDE_propositions_p_and_q_l15_1587

theorem propositions_p_and_q : 
  (∃ a b : ℝ, a > b ∧ 1/a > 1/b) ∧ 
  (∀ x : ℝ, Real.sin x + Real.cos x < 3/2) := by
  sorry

end NUMINAMATH_CALUDE_propositions_p_and_q_l15_1587


namespace NUMINAMATH_CALUDE_angle_bisector_quadrilateral_sum_l15_1543

/-- Given a convex quadrilateral ABCD with angles α, β, γ, δ, 
    and its angle bisectors intersecting to form quadrilateral HIJE,
    the sum of opposite angles HIJ and JEH in HIJE is 180°. -/
theorem angle_bisector_quadrilateral_sum (α β γ δ : Real) 
  (h_convex : α > 0 ∧ β > 0 ∧ γ > 0 ∧ δ > 0)
  (h_sum : α + β + γ + δ = 360) : 
  (α/2 + β/2) + (γ/2 + δ/2) = 180 := by
  sorry

end NUMINAMATH_CALUDE_angle_bisector_quadrilateral_sum_l15_1543
