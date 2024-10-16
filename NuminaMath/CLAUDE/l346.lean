import Mathlib

namespace NUMINAMATH_CALUDE_subtract_negative_l346_34635

theorem subtract_negative : -3 - 1 = -4 := by
  sorry

end NUMINAMATH_CALUDE_subtract_negative_l346_34635


namespace NUMINAMATH_CALUDE_distance_to_y_axis_l346_34690

def point_A (x : ℝ) : ℝ × ℝ := (x - 4, 2 * x + 3)

theorem distance_to_y_axis (x : ℝ) : 
  (|x - 4| = 1) ↔ (x = 5 ∨ x = 3) :=
sorry

end NUMINAMATH_CALUDE_distance_to_y_axis_l346_34690


namespace NUMINAMATH_CALUDE_magic_square_solution_l346_34675

/-- Represents a 3x3 magic square -/
structure MagicSquare :=
  (a11 a12 a13 a21 a22 a23 a31 a32 a33 : ℤ)

/-- The sum of any row, column, or diagonal in a magic square is constant -/
def MagicSquare.isMagic (s : MagicSquare) : Prop :=
  let sum := s.a11 + s.a12 + s.a13
  sum = s.a21 + s.a22 + s.a23 ∧
  sum = s.a31 + s.a32 + s.a33 ∧
  sum = s.a11 + s.a21 + s.a31 ∧
  sum = s.a12 + s.a22 + s.a32 ∧
  sum = s.a13 + s.a23 + s.a33 ∧
  sum = s.a11 + s.a22 + s.a33 ∧
  sum = s.a13 + s.a22 + s.a31

theorem magic_square_solution :
  ∀ (s : MagicSquare),
    s.isMagic →
    s.a11 = s.a11 ∧
    s.a12 = 23 ∧
    s.a13 = 84 ∧
    s.a21 = 3 →
    s.a11 = 175 := by
  sorry

#check magic_square_solution

end NUMINAMATH_CALUDE_magic_square_solution_l346_34675


namespace NUMINAMATH_CALUDE_line_problem_l346_34692

/-- A line in the xy-plane defined by y = 2x + 4 -/
def line (x : ℝ) : ℝ := 2 * x + 4

/-- Point P on the x-axis -/
def P (p : ℝ) : ℝ × ℝ := (p, 0)

/-- Point R where the line intersects the y-axis -/
def R : ℝ × ℝ := (0, line 0)

/-- Point Q where the line intersects the vertical line through P -/
def Q (p : ℝ) : ℝ × ℝ := (p, line p)

/-- Area of the quadrilateral OPQR -/
def area_OPQR (p : ℝ) : ℝ := p * (p + 4)

theorem line_problem (p : ℝ) (h : p > 0) :
  R.2 = 4 ∧
  Q p = (p, 2 * p + 4) ∧
  area_OPQR p = p * (p + 4) ∧
  (p = 8 → area_OPQR p = 96) ∧
  (area_OPQR p = 77 → p = 7) := by
  sorry

end NUMINAMATH_CALUDE_line_problem_l346_34692


namespace NUMINAMATH_CALUDE_vector_propositions_correctness_l346_34668

variable {V : Type*} [AddCommGroup V]

-- Define vectors as differences between points
def vec (A B : V) : V := B - A

-- State the theorem
theorem vector_propositions_correctness :
  ∃ (A B C : V),
    (vec A B + vec B A = 0) ∧
    (vec A B + vec B C = vec A C) ∧
    ¬(vec A B - vec A C = vec B C) ∧
    ¬(0 • vec A B = 0) :=
by sorry

end NUMINAMATH_CALUDE_vector_propositions_correctness_l346_34668


namespace NUMINAMATH_CALUDE_tax_free_amount_satisfies_equation_l346_34623

/-- The tax-free amount for goods purchased in country B -/
def tax_free_amount : ℝ :=
  -- We define the tax-free amount, but don't provide its value
  -- as it needs to be proved
  sorry

/-- The total value of goods purchased -/
def total_value : ℝ := 1720

/-- The tax rate as a decimal -/
def tax_rate : ℝ := 0.11

/-- The amount of tax paid -/
def tax_paid : ℝ := 123.2

/-- Theorem stating that the tax-free amount satisfies the given equation -/
theorem tax_free_amount_satisfies_equation :
  tax_rate * (total_value - tax_free_amount) = tax_paid := by
  sorry

end NUMINAMATH_CALUDE_tax_free_amount_satisfies_equation_l346_34623


namespace NUMINAMATH_CALUDE_possible_m_values_l346_34604

def A : Set ℝ := {x | x^2 - 9*x - 10 = 0}
def B (m : ℝ) : Set ℝ := {x | m*x + 1 = 0}

theorem possible_m_values :
  {m : ℝ | A ∪ B m = A} = {0, 1, -(1/10)} := by sorry

end NUMINAMATH_CALUDE_possible_m_values_l346_34604


namespace NUMINAMATH_CALUDE_foster_farms_donation_l346_34615

-- Define the number of dressed chickens donated by Foster Farms
variable (C : ℕ)

-- Define the donations of each company
def foster_farms := C
def american_summits := 2 * C
def hormel := 3 * C
def boudin_butchers := C
def del_monte := 2 * C - 30

-- Define the total number of food items
def total_items := 375

-- Theorem statement
theorem foster_farms_donation :
  foster_farms + american_summits + hormel + boudin_butchers + del_monte = total_items →
  C = 45 := by
  sorry

end NUMINAMATH_CALUDE_foster_farms_donation_l346_34615


namespace NUMINAMATH_CALUDE_plate_on_square_table_l346_34626

/-- Given a square table with a round plate, if the distances from the edge of the plate
    to three edges of the table are 10 cm, 63 cm, and 20 cm respectively, then the distance
    from the edge of the plate to the fourth edge of the table is 53 cm. -/
theorem plate_on_square_table (d1 d2 d3 d4 : ℝ) (h1 : d1 = 10) (h2 : d2 = 63) (h3 : d3 = 20)
    (h_square : d1 + d2 = d3 + d4) : d4 = 53 := by
  sorry

end NUMINAMATH_CALUDE_plate_on_square_table_l346_34626


namespace NUMINAMATH_CALUDE_b_grazing_months_l346_34642

/-- Represents the number of months b put his oxen for grazing -/
def b_months : ℕ := sorry

/-- Represents the total rent of the pasture in rupees -/
def total_rent : ℕ := 140

/-- Represents c's share of the rent in rupees -/
def c_share : ℕ := 36

/-- Calculates the total oxen-months for all three people -/
def total_oxen_months : ℕ := 70 + 12 * b_months + 45

theorem b_grazing_months :
  (c_share : ℚ) / total_rent = 45 / total_oxen_months → b_months = 5 := by sorry

end NUMINAMATH_CALUDE_b_grazing_months_l346_34642


namespace NUMINAMATH_CALUDE_stability_comparison_l346_34688

/-- Represents an athlete's shooting performance -/
structure Athlete where
  average_score : ℝ
  variance : ℝ
  variance_nonneg : 0 ≤ variance

/-- Defines when one athlete's performance is more stable than another's -/
def more_stable (a b : Athlete) : Prop :=
  a.average_score = b.average_score ∧ a.variance < b.variance

theorem stability_comparison (a b : Athlete) 
  (h : a.variance < b.variance) (h_avg : a.average_score = b.average_score) : 
  more_stable a b := by
  sorry

#check stability_comparison

end NUMINAMATH_CALUDE_stability_comparison_l346_34688


namespace NUMINAMATH_CALUDE_opposite_lateral_angle_is_90_l346_34670

/-- A regular quadrangular pyramid -/
structure RegularQuadrangularPyramid where
  /-- The angle between a lateral face and the base plane -/
  lateral_base_angle : ℝ
  /-- The angle between a lateral face and the base plane is 45° -/
  angle_is_45 : lateral_base_angle = 45

/-- The angle between opposite lateral faces of the pyramid -/
def opposite_lateral_angle (p : RegularQuadrangularPyramid) : ℝ := sorry

/-- Theorem: In a regular quadrangular pyramid where the lateral face forms a 45° angle 
    with the base plane, the angle between opposite lateral faces is 90° -/
theorem opposite_lateral_angle_is_90 (p : RegularQuadrangularPyramid) :
  opposite_lateral_angle p = 90 := by sorry

end NUMINAMATH_CALUDE_opposite_lateral_angle_is_90_l346_34670


namespace NUMINAMATH_CALUDE_work_completion_time_l346_34630

theorem work_completion_time (a b : ℝ) 
  (h1 : a + b = 1 / 16)  -- A and B together finish in 16 days
  (h2 : a = 1 / 32)      -- A alone finishes in 32 days
  : 1 / b = 32 :=        -- B alone finishes in 32 days
by sorry

end NUMINAMATH_CALUDE_work_completion_time_l346_34630


namespace NUMINAMATH_CALUDE_simplify_radical_product_l346_34601

theorem simplify_radical_product : 
  (3 * 5) ^ (1/3) * (5^2 * 3^4) ^ (1/2) = 15 := by
  sorry

end NUMINAMATH_CALUDE_simplify_radical_product_l346_34601


namespace NUMINAMATH_CALUDE_tomorrow_is_saturday_l346_34654

-- Define the days of the week
inductive Day : Type
  | Sunday : Day
  | Monday : Day
  | Tuesday : Day
  | Wednesday : Day
  | Thursday : Day
  | Friday : Day
  | Saturday : Day

-- Define a function to get the next day
def nextDay (d : Day) : Day :=
  match d with
  | Day.Sunday => Day.Monday
  | Day.Monday => Day.Tuesday
  | Day.Tuesday => Day.Wednesday
  | Day.Wednesday => Day.Thursday
  | Day.Thursday => Day.Friday
  | Day.Friday => Day.Saturday
  | Day.Saturday => Day.Sunday

-- Define a function to add days
def addDays (d : Day) (n : Nat) : Day :=
  match n with
  | 0 => d
  | Nat.succ k => nextDay (addDays d k)

-- Define the given condition
def dayBeforeYesterday : Day := Day.Wednesday

-- Theorem to prove
theorem tomorrow_is_saturday : 
  addDays (nextDay (nextDay dayBeforeYesterday)) 1 = Day.Saturday :=
sorry

end NUMINAMATH_CALUDE_tomorrow_is_saturday_l346_34654


namespace NUMINAMATH_CALUDE_max_value_M_l346_34605

theorem max_value_M (x y z w : ℝ) (h : x + y + z + w = 1) :
  let M := x*w + 2*y*w + 3*x*y + 3*z*w + 4*x*z + 5*y*z
  ∃ (x₀ y₀ z₀ w₀ : ℝ), x₀ + y₀ + z₀ + w₀ = 1 ∧
    (∀ x y z w, x + y + z + w = 1 →
      x*w + 2*y*w + 3*x*y + 3*z*w + 4*x*z + 5*y*z ≤
      x₀*w₀ + 2*y₀*w₀ + 3*x₀*y₀ + 3*z₀*w₀ + 4*x₀*z₀ + 5*y₀*z₀) ∧
    x₀*w₀ + 2*y₀*w₀ + 3*x₀*y₀ + 3*z₀*w₀ + 4*x₀*z₀ + 5*y₀*z₀ = 3/2 :=
by sorry

end NUMINAMATH_CALUDE_max_value_M_l346_34605


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l346_34625

/-- Given that the solution set of ax^2 - bx - 1 > 0 is {x | -1/2 < x < -1/3},
    prove that the solution set of x^2 - bx - a ≥ 0 is {x | x ≥ 3 ∨ x ≤ 2} -/
theorem quadratic_inequality_solution_set 
  (a b : ℝ) 
  (h : ∀ x : ℝ, ax^2 - b*x - 1 > 0 ↔ -1/2 < x ∧ x < -1/3) :
  ∀ x : ℝ, x^2 - b*x - a ≥ 0 ↔ x ≥ 3 ∨ x ≤ 2 :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l346_34625


namespace NUMINAMATH_CALUDE_holly_401k_contribution_l346_34620

/-- Calculates the total contribution to Holly's 401k after 1 year -/
def total_contribution (paychecks_per_year : ℕ) (contribution_per_paycheck : ℚ) (company_match_percentage : ℚ) : ℚ :=
  let employee_contribution := paychecks_per_year * contribution_per_paycheck
  let company_contribution := employee_contribution * company_match_percentage
  employee_contribution + company_contribution

/-- Theorem stating that Holly's total 401k contribution after 1 year is $2,756.00 -/
theorem holly_401k_contribution :
  total_contribution 26 100 (6 / 100) = 2756 :=
by sorry

end NUMINAMATH_CALUDE_holly_401k_contribution_l346_34620


namespace NUMINAMATH_CALUDE_range_of_2alpha_minus_beta_l346_34624

theorem range_of_2alpha_minus_beta (α β : ℝ) 
  (h1 : π < α + β ∧ α + β < (4*π)/3)
  (h2 : -π < α - β ∧ α - β < -π/3) :
  ∀ x, (-π < x ∧ x < π/6) ↔ ∃ α' β', 
    (π < α' + β' ∧ α' + β' < (4*π)/3) ∧
    (-π < α' - β' ∧ α' - β' < -π/3) ∧
    x = 2*α' - β' :=
by sorry

end NUMINAMATH_CALUDE_range_of_2alpha_minus_beta_l346_34624


namespace NUMINAMATH_CALUDE_solution_using_determinants_l346_34640

/-- Definition of 2x2 determinant -/
def det2x2 (a b c d : ℝ) : ℝ := a * d - b * c

/-- System of equations -/
def equation1 (x y : ℝ) : Prop := 2 * x - y = 1
def equation2 (x y : ℝ) : Prop := 3 * x + 2 * y = 11

/-- Determinants for the system -/
def D : ℝ := det2x2 2 (-1) 3 2
def D_x : ℝ := det2x2 1 (-1) 11 2
def D_y : ℝ := det2x2 2 1 3 11

/-- Theorem: Solution of the system using determinant method -/
theorem solution_using_determinants :
  ∃ (x y : ℝ), equation1 x y ∧ equation2 x y ∧ x = D_x / D ∧ y = D_y / D :=
sorry

end NUMINAMATH_CALUDE_solution_using_determinants_l346_34640


namespace NUMINAMATH_CALUDE_thirtieth_triangular_number_l346_34693

/-- The nth triangular number -/
def triangular_number (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The 30th triangular number is 465 -/
theorem thirtieth_triangular_number : triangular_number 30 = 465 := by
  sorry

end NUMINAMATH_CALUDE_thirtieth_triangular_number_l346_34693


namespace NUMINAMATH_CALUDE_collinear_points_sum_l346_34665

/-- Three points in 3D space are collinear if they lie on the same straight line. -/
def collinear (p1 p2 p3 : ℝ × ℝ × ℝ) : Prop :=
  ∃ t : ℝ, p2 = p1 + t • (p3 - p1) ∨ p1 = p2 + t • (p3 - p2)

theorem collinear_points_sum (a b : ℝ) :
  collinear (2, a, b) (a, 3, b) (a, b, 4) → a + b = 6 := by
  sorry

end NUMINAMATH_CALUDE_collinear_points_sum_l346_34665


namespace NUMINAMATH_CALUDE_night_day_crew_loading_ratio_l346_34644

theorem night_day_crew_loading_ratio 
  (day_crew : ℕ) 
  (night_crew : ℕ) 
  (total_boxes : ℝ) 
  (h1 : night_crew = (3 / 4 : ℝ) * day_crew)
  (h2 : (0.64 : ℝ) * total_boxes = day_crew * (total_boxes / day_crew)) :
  (total_boxes - 0.64 * total_boxes) / night_crew = 
  (3 / 4 : ℝ) * (0.64 * total_boxes / day_crew) := by
  sorry

end NUMINAMATH_CALUDE_night_day_crew_loading_ratio_l346_34644


namespace NUMINAMATH_CALUDE_unique_products_count_l346_34613

def set_a : Finset ℕ := {2, 3, 5, 7, 11}
def set_b : Finset ℕ := {2, 4, 6, 19}

theorem unique_products_count : 
  Finset.card ((set_a.product set_b).image (λ (x : ℕ × ℕ) => x.1 * x.2)) = 19 := by
  sorry

end NUMINAMATH_CALUDE_unique_products_count_l346_34613


namespace NUMINAMATH_CALUDE_recipe_total_cups_l346_34602

-- Define the ratio of ingredients
def butter_ratio : ℚ := 2
def flour_ratio : ℚ := 5
def sugar_ratio : ℚ := 3

-- Define the amount of sugar used
def sugar_cups : ℚ := 9

-- Theorem statement
theorem recipe_total_cups : 
  let total_ratio := butter_ratio + flour_ratio + sugar_ratio
  let scale_factor := sugar_cups / sugar_ratio
  let total_cups := scale_factor * total_ratio
  total_cups = 30 := by sorry

end NUMINAMATH_CALUDE_recipe_total_cups_l346_34602


namespace NUMINAMATH_CALUDE_diagonal_intersection_coincides_l346_34666

-- Define a circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define a point
structure Point where
  x : ℝ
  y : ℝ

-- Define a quadrilateral
structure Quadrilateral where
  A : Point
  B : Point
  C : Point
  D : Point

-- Function to check if a quadrilateral is inscribed in a circle
def isInscribed (q : Quadrilateral) (c : Circle) : Prop := sorry

-- Function to check if a point is on a circle
def isOnCircle (p : Point) (c : Circle) : Prop := sorry

-- Function to check if a line is tangent to a circle at a point
def isTangent (p1 p2 : Point) (c : Circle) (p : Point) : Prop := sorry

-- Function to find the intersection point of two lines
def intersectionPoint (p1 p2 p3 p4 : Point) : Point := sorry

-- Main theorem
theorem diagonal_intersection_coincides 
  (c : Circle) 
  (ABCD : Quadrilateral) 
  (E F G K : Point) :
  isInscribed ABCD c →
  isOnCircle E c ∧ isOnCircle F c ∧ isOnCircle G c ∧ isOnCircle K c →
  isTangent ABCD.A ABCD.B c E →
  isTangent ABCD.B ABCD.C c F →
  isTangent ABCD.C ABCD.D c G →
  isTangent ABCD.D ABCD.A c K →
  intersectionPoint ABCD.A ABCD.C ABCD.B ABCD.D = intersectionPoint E G F K := by
  sorry

end NUMINAMATH_CALUDE_diagonal_intersection_coincides_l346_34666


namespace NUMINAMATH_CALUDE_company_profit_calculation_l346_34629

/-- Given a company's total annual profit and the difference between first and second half profits,
    prove that the second half profit is as calculated. -/
theorem company_profit_calculation (total_profit second_half_profit : ℚ) : 
  total_profit = 3635000 →
  second_half_profit + 2750000 + second_half_profit = total_profit →
  second_half_profit = 442500 := by
  sorry

end NUMINAMATH_CALUDE_company_profit_calculation_l346_34629


namespace NUMINAMATH_CALUDE_max_value_cos_sin_l346_34691

theorem max_value_cos_sin (x : Real) : 3 * Real.cos x + Real.sin x ≤ Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_max_value_cos_sin_l346_34691


namespace NUMINAMATH_CALUDE_urn_theorem_l346_34647

/-- Represents the state of the urn -/
structure UrnState where
  white : ℕ
  black : ℕ

/-- Represents a marble replacement rule -/
inductive Rule
  | rule1
  | rule2
  | rule3
  | rule4
  | rule5

/-- Applies a rule to the current urn state -/
def applyRule (state : UrnState) (rule : Rule) : UrnState :=
  match rule with
  | Rule.rule1 => UrnState.mk (state.white - 4) (state.black + 2)
  | Rule.rule2 => UrnState.mk (state.white - 1) (state.black)
  | Rule.rule3 => UrnState.mk (state.white - 1) (state.black)
  | Rule.rule4 => UrnState.mk (state.white + 1) (state.black - 3)
  | Rule.rule5 => UrnState.mk (state.white) (state.black - 1)

/-- Checks if the total number of marbles is even -/
def isEvenTotal (state : UrnState) : Prop :=
  Even (state.white + state.black)

/-- Checks if a given state is reachable from the initial state -/
def isReachable (initial : UrnState) (final : UrnState) : Prop :=
  ∃ (rules : List Rule), final = rules.foldl applyRule initial ∧ 
    (∀ (intermediate : UrnState), intermediate ∈ rules.scanl applyRule initial → isEvenTotal intermediate)

/-- The main theorem to prove -/
theorem urn_theorem (initial : UrnState) : 
  initial.white = 150 ∧ initial.black = 50 →
  (isReachable initial (UrnState.mk 78 72) ∨ isReachable initial (UrnState.mk 126 24)) :=
sorry


end NUMINAMATH_CALUDE_urn_theorem_l346_34647


namespace NUMINAMATH_CALUDE_total_bonus_calculation_l346_34610

def senior_bonus : ℕ := 1900
def junior_bonus : ℕ := 3100

theorem total_bonus_calculation : senior_bonus + junior_bonus = 5000 := by
  sorry

end NUMINAMATH_CALUDE_total_bonus_calculation_l346_34610


namespace NUMINAMATH_CALUDE_combination_equation_solution_l346_34622

theorem combination_equation_solution (x : ℕ) : 
  (Nat.choose 34 (2*x) = Nat.choose 34 (4*x - 8)) → (x = 4 ∨ x = 7) := by
  sorry

end NUMINAMATH_CALUDE_combination_equation_solution_l346_34622


namespace NUMINAMATH_CALUDE_mans_wage_to_womans_wage_ratio_l346_34684

/-- Prove that the ratio of a man's daily wage to a woman's daily wage is 4:1 -/
theorem mans_wage_to_womans_wage_ratio :
  ∀ (man_wage woman_wage : ℚ),
  (∃ k : ℚ, man_wage = k * woman_wage) →  -- Man's wage is some multiple of woman's wage
  (8 * 25 * man_wage = 14400) →           -- 8 men working for 25 days earn Rs. 14400
  (40 * 30 * woman_wage = 21600) →        -- 40 women working for 30 days earn Rs. 21600
  man_wage / woman_wage = 4 / 1 := by
sorry

end NUMINAMATH_CALUDE_mans_wage_to_womans_wage_ratio_l346_34684


namespace NUMINAMATH_CALUDE_largest_fraction_of_consecutive_evens_l346_34662

theorem largest_fraction_of_consecutive_evens (a b c d : ℕ) : 
  2 < a → a < b → b < c → c < d → 
  Even a → Even b → Even c → Even d →
  (b = a + 2) → (c = b + 2) → (d = c + 2) →
  (c + d) / (b + a) > (b + c) / (a + d) ∧
  (c + d) / (b + a) > (a + d) / (c + b) ∧
  (c + d) / (b + a) > (a + c) / (b + d) ∧
  (c + d) / (b + a) > (b + d) / (c + a) := by
  sorry

end NUMINAMATH_CALUDE_largest_fraction_of_consecutive_evens_l346_34662


namespace NUMINAMATH_CALUDE_nine_digit_rearrangement_l346_34655

def is_coprime (a b : ℕ) : Prop := Nat.gcd a b = 1

def last_digit (n : ℕ) : ℕ := n % 10

def move_last_to_first (n : ℕ) : ℕ :=
  (last_digit n) * 10^8 + (n / 10)

theorem nine_digit_rearrangement (B : ℕ) (A : ℕ) :
  (B > 666666666) →
  (B < 1000000000) →
  (is_coprime B 24) →
  (A = move_last_to_first B) →
  (166666667 ≤ A) ∧ (A ≤ 999999998) :=
sorry

end NUMINAMATH_CALUDE_nine_digit_rearrangement_l346_34655


namespace NUMINAMATH_CALUDE_inequality_proof_l346_34699

theorem inequality_proof (a b c d e f : ℝ) 
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0 ∧ d > 0 ∧ e > 0 ∧ f > 0) 
  (h_cond : |Real.sqrt (a * b) - Real.sqrt (c * d)| ≤ 2) :
  (e / a + b / e) * (e / c + d / e) ≥ (f / a - b) * (d - f / c) := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l346_34699


namespace NUMINAMATH_CALUDE_probability_P_equals_1_plus_i_l346_34634

/-- The set of vertices of a regular hexagon in the complex plane -/
def V : Set ℂ := {1, -1, Complex.I, -Complex.I, (1/2) + (Real.sqrt 3 / 2) * Complex.I, -(1/2) - (Real.sqrt 3 / 2) * Complex.I}

/-- The number of elements chosen from V -/
def n : ℕ := 10

/-- The product of n randomly chosen elements from V -/
noncomputable def P : ℂ := sorry

/-- The probability that P equals 1 + i -/
noncomputable def prob_P_equals_1_plus_i : ℝ := sorry

/-- Theorem stating the probability of P equaling 1 + i -/
theorem probability_P_equals_1_plus_i : prob_P_equals_1_plus_i = 120 / 24649 := by sorry

end NUMINAMATH_CALUDE_probability_P_equals_1_plus_i_l346_34634


namespace NUMINAMATH_CALUDE_jake_has_seven_peaches_l346_34600

-- Define the number of peaches and apples for Steven and Jake
def steven_peaches : ℕ := 19
def steven_apples : ℕ := 14
def jake_peaches : ℕ := steven_peaches - 12
def jake_apples : ℕ := steven_apples + 79

-- Theorem to prove
theorem jake_has_seven_peaches : jake_peaches = 7 := by
  sorry

end NUMINAMATH_CALUDE_jake_has_seven_peaches_l346_34600


namespace NUMINAMATH_CALUDE_mrs_white_orchard_yield_l346_34695

/-- Represents the dimensions and crop yields of Mrs. White's orchard -/
structure Orchard where
  length_paces : ℕ
  width_paces : ℕ
  feet_per_pace : ℕ
  tomato_yield_per_sqft : ℚ
  cucumber_yield_per_sqft : ℚ

/-- Calculates the expected crop yield from the orchard -/
def expected_yield (o : Orchard) : ℚ :=
  let area_sqft := (o.length_paces * o.feet_per_pace) * (o.width_paces * o.feet_per_pace)
  let half_area_sqft := area_sqft / 2
  let tomato_yield := half_area_sqft * o.tomato_yield_per_sqft
  let cucumber_yield := half_area_sqft * o.cucumber_yield_per_sqft
  tomato_yield + cucumber_yield

/-- Mrs. White's orchard -/
def mrs_white_orchard : Orchard :=
  { length_paces := 10
  , width_paces := 30
  , feet_per_pace := 3
  , tomato_yield_per_sqft := 3/4
  , cucumber_yield_per_sqft := 2/5 }

theorem mrs_white_orchard_yield :
  expected_yield mrs_white_orchard = 1552.5 := by
  sorry

end NUMINAMATH_CALUDE_mrs_white_orchard_yield_l346_34695


namespace NUMINAMATH_CALUDE_inequality_proof_l346_34661

theorem inequality_proof (x y z : ℝ) 
  (non_neg_x : x ≥ 0) (non_neg_y : y ≥ 0) (non_neg_z : z ≥ 0)
  (sum_of_squares : x^2 + y^2 + z^2 = 1) :
  1 ≤ (x / (1 + y*z)) + (y / (1 + z*x)) + (z / (1 + x*y)) ∧
  (x / (1 + y*z)) + (y / (1 + z*x)) + (z / (1 + x*y)) ≤ Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l346_34661


namespace NUMINAMATH_CALUDE_inequality_proof_l346_34672

theorem inequality_proof (a b c d : ℝ) 
  (nonzero : a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0)
  (distinct : a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d)
  (sum_condition : a/b + b/c + c/d + d/a = 4)
  (product_condition : a*c = b*d) :
  (a/c + b/d + c/a + d/b ≤ -12) ∧
  (∀ k : ℝ, (∀ a' b' c' d' : ℝ, 
    a'/b' + b'/c' + c'/d' + d'/a' = 4 → 
    a'*c' = b'*d' → 
    a'/c' + b'/d' + c'/a' + d'/b' ≤ k) → 
  k ≤ -12) :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l346_34672


namespace NUMINAMATH_CALUDE_find_divisor_l346_34632

theorem find_divisor (dividend : ℕ) (quotient : ℕ) (remainder : ℕ) (divisor : ℕ) :
  dividend = 161 →
  quotient = 10 →
  remainder = 1 →
  dividend = divisor * quotient + remainder →
  divisor = 16 := by
sorry

end NUMINAMATH_CALUDE_find_divisor_l346_34632


namespace NUMINAMATH_CALUDE_contradiction_proof_l346_34681

theorem contradiction_proof (a b c : ℝ) 
  (h1 : a + b + c > 0) 
  (h2 : a * b + b * c + a * c > 0) 
  (h3 : a * b * c > 0) : 
  ¬(a > 0 ∧ b > 0 ∧ c > 0) ↔ (a ≤ 0 ∨ b ≤ 0 ∨ c ≤ 0) := by
sorry

end NUMINAMATH_CALUDE_contradiction_proof_l346_34681


namespace NUMINAMATH_CALUDE_sixteen_power_divided_by_four_l346_34650

theorem sixteen_power_divided_by_four (n : ℕ) : n = 16^2023 → n/4 = 4^4045 := by
  sorry

end NUMINAMATH_CALUDE_sixteen_power_divided_by_four_l346_34650


namespace NUMINAMATH_CALUDE_basketball_game_scores_l346_34685

/-- Represents the scores of a team in a basketball game -/
structure TeamScores where
  q1 : ℕ
  q2 : ℕ
  q3 : ℕ
  q4 : ℕ

/-- Calculates the total score of a team -/
def totalScore (scores : TeamScores) : ℕ :=
  scores.q1 + scores.q2 + scores.q3 + scores.q4

/-- Calculates the score for the first half -/
def firstHalfScore (scores : TeamScores) : ℕ :=
  scores.q1 + scores.q2

/-- Calculates the score for the second half -/
def secondHalfScore (scores : TeamScores) : ℕ :=
  scores.q3 + scores.q4

/-- Checks if the scores form an increasing geometric sequence -/
def isGeometricSequence (scores : TeamScores) : Prop :=
  ∃ r : ℚ, r > 1 ∧ 
    scores.q2 = scores.q1 * r ∧
    scores.q3 = scores.q2 * r ∧
    scores.q4 = scores.q3 * r

/-- Checks if the scores form an increasing arithmetic sequence -/
def isArithmeticSequence (scores : TeamScores) : Prop :=
  ∃ d : ℕ, d > 0 ∧
    scores.q2 = scores.q1 + d ∧
    scores.q3 = scores.q2 + d ∧
    scores.q4 = scores.q3 + d

theorem basketball_game_scores 
  (eagles : TeamScores) (tigers : TeamScores) 
  (h1 : isGeometricSequence eagles)
  (h2 : isArithmeticSequence tigers)
  (h3 : firstHalfScore eagles = firstHalfScore tigers)
  (h4 : totalScore eagles = totalScore tigers + 2)
  (h5 : totalScore eagles ≤ 100)
  (h6 : totalScore tigers ≤ 100) :
  secondHalfScore eagles + secondHalfScore tigers = 116 := by
  sorry

end NUMINAMATH_CALUDE_basketball_game_scores_l346_34685


namespace NUMINAMATH_CALUDE_divisibility_implication_l346_34698

theorem divisibility_implication (a b : ℤ) : 
  (31 ∣ (6 * a + 11 * b)) → (31 ∣ (a + 7 * b)) := by
sorry

end NUMINAMATH_CALUDE_divisibility_implication_l346_34698


namespace NUMINAMATH_CALUDE_y_completion_time_l346_34648

/-- A worker's rate is defined as the fraction of work they can complete in one day -/
def worker_rate (days_to_complete : ℚ) : ℚ := 1 / days_to_complete

/-- The time taken to complete a given fraction of work at a given rate -/
def time_for_fraction (fraction : ℚ) (rate : ℚ) : ℚ := fraction / rate

theorem y_completion_time (x_total_days : ℚ) (x_worked_days : ℚ) (y_completion_days : ℚ) : 
  x_total_days = 40 → x_worked_days = 8 → y_completion_days = 28 →
  time_for_fraction 1 (worker_rate (time_for_fraction 1 
    (worker_rate y_completion_days / (1 - x_worked_days * worker_rate x_total_days)))) = 35 := by
  sorry

end NUMINAMATH_CALUDE_y_completion_time_l346_34648


namespace NUMINAMATH_CALUDE_exists_monochromatic_isosceles_right_triangle_l346_34627

-- Define a type for colors
inductive Color
| Red
| Blue

-- Define a type for points in the plane
structure Point where
  x : ℝ
  y : ℝ

-- Define a coloring function
def coloring : Point → Color := sorry

-- Define an isosceles right triangle
def isIsoscelesRightTriangle (a b c : Point) : Prop := sorry

-- Theorem statement
theorem exists_monochromatic_isosceles_right_triangle :
  ∃ a b c : Point, isIsoscelesRightTriangle a b c ∧ 
    coloring a = coloring b ∧ coloring b = coloring c := by sorry

end NUMINAMATH_CALUDE_exists_monochromatic_isosceles_right_triangle_l346_34627


namespace NUMINAMATH_CALUDE_percentage_decrease_l346_34628

/-- Proves that for an original number of 40, if the difference between its value 
    increased by 25% and its value decreased by x% is 22, then x = 30. -/
theorem percentage_decrease (x : ℝ) : 
  (40 + 0.25 * 40) - (40 - 0.01 * x * 40) = 22 → x = 30 := by
  sorry

end NUMINAMATH_CALUDE_percentage_decrease_l346_34628


namespace NUMINAMATH_CALUDE_shaded_area_rectangle_with_quarter_circles_l346_34658

/-- The area of the shaded region in a rectangle with quarter circles in each corner -/
theorem shaded_area_rectangle_with_quarter_circles
  (length : ℝ) (width : ℝ) (radius : ℝ)
  (h_length : length = 12)
  (h_width : width = 8)
  (h_radius : radius = 4) :
  length * width - π * radius^2 = 96 - 16 * π :=
by sorry

end NUMINAMATH_CALUDE_shaded_area_rectangle_with_quarter_circles_l346_34658


namespace NUMINAMATH_CALUDE_area_of_quadrilateral_l346_34649

/-- A quadrilateral with specific properties -/
structure Quadrilateral :=
  (EF HG EH FG : ℕ)
  (right_angle_F : EF ^ 2 + FG ^ 2 = 25)
  (right_angle_H : EH ^ 2 + HG ^ 2 = 25)
  (different_sides : ∃ (a b : ℕ), (a ≠ b) ∧ ((a = EF ∧ b = FG) ∨ (a = EH ∧ b = HG) ∨ (a = EF ∧ b = HG) ∨ (a = EH ∧ b = FG)))

/-- The area of the quadrilateral EFGH is 12 -/
theorem area_of_quadrilateral (q : Quadrilateral) : (q.EF * q.FG + q.EH * q.HG) / 2 = 12 :=
sorry

end NUMINAMATH_CALUDE_area_of_quadrilateral_l346_34649


namespace NUMINAMATH_CALUDE_problem_solution_l346_34617

-- Define the conversion rate from paise to rupees
def paise_to_rupees (paise : ℚ) : ℚ := paise / 100

-- Define the given conditions
def condition_a (a : ℚ) : Prop := 0.005 * a = paise_to_rupees 80
def condition_b (b : ℚ) : Prop := 0.0025 * b = paise_to_rupees 60
def condition_c (a b c : ℚ) : Prop := c = 0.5 * a - 0.1 * b

-- Theorem statement
theorem problem_solution (a b c : ℚ) 
  (ha : condition_a a) 
  (hb : condition_b b) 
  (hc : condition_c a b c) : 
  a = 160 ∧ b = 240 ∧ c = 56 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l346_34617


namespace NUMINAMATH_CALUDE_book_pages_digits_unique_book_pages_l346_34636

/-- Given a natural number n, calculate the total number of digits used to number pages from 1 to n. -/
def totalDigits (n : ℕ) : ℕ :=
  let singleDigits := min n 9
  let doubleDigits := max (min n 99 - 9) 0
  let tripleDigits := max (n - 99) 0
  singleDigits + 2 * doubleDigits + 3 * tripleDigits

/-- Theorem stating that a book with 266 pages requires exactly 690 digits to number all its pages. -/
theorem book_pages_digits : totalDigits 266 = 690 := by
  sorry

/-- Theorem stating that 266 is the unique number of pages that requires exactly 690 digits. -/
theorem unique_book_pages (n : ℕ) : totalDigits n = 690 → n = 266 := by
  sorry

end NUMINAMATH_CALUDE_book_pages_digits_unique_book_pages_l346_34636


namespace NUMINAMATH_CALUDE_no_geometric_sequence_sin_angles_l346_34671

theorem no_geometric_sequence_sin_angles :
  ¬∃ a : Real, 0 < a ∧ a < 2 * Real.pi ∧
  ∃ r : Real, (Real.sin (2 * a) = r * Real.sin a) ∧
             (Real.sin (3 * a) = r * Real.sin (2 * a)) := by
  sorry

end NUMINAMATH_CALUDE_no_geometric_sequence_sin_angles_l346_34671


namespace NUMINAMATH_CALUDE_evaluate_expression_l346_34656

theorem evaluate_expression : -(16 / 4 * 11 - 70 + 5^2) = 1 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l346_34656


namespace NUMINAMATH_CALUDE_min_length_intersection_l346_34616

theorem min_length_intersection (m n : ℝ) : 
  let M := {x : ℝ | m ≤ x ∧ x ≤ m + 3/4}
  let N := {x : ℝ | n - 1/3 ≤ x ∧ x ≤ n}
  ∀ x ∈ M, 0 ≤ x ∧ x ≤ 1 →
  ∀ x ∈ N, 0 ≤ x ∧ x ≤ 1 →
  ∃ a b : ℝ, M ∩ N ⊆ {x : ℝ | a ≤ x ∧ x ≤ b} ∧
            ∀ c d : ℝ, M ∩ N ⊆ {x : ℝ | c ≤ x ∧ x ≤ d} →
            b - a ≤ d - c ∧
            b - a = 1/12 :=
by sorry

end NUMINAMATH_CALUDE_min_length_intersection_l346_34616


namespace NUMINAMATH_CALUDE_largest_root_range_l346_34674

def polynomial (x b₃ b₂ b₁ b₀ : ℝ) : ℝ := x^4 + b₃*x^3 + b₂*x^2 + b₁*x + b₀

def is_valid_coefficient (b : ℝ) : Prop := abs b < 3

theorem largest_root_range :
  ∃ s : ℝ, 3 < s ∧ s < 4 ∧
  (∀ b₃ b₂ b₁ b₀ : ℝ, is_valid_coefficient b₃ → is_valid_coefficient b₂ →
    is_valid_coefficient b₁ → is_valid_coefficient b₀ →
    (∀ x : ℝ, x > s → polynomial x b₃ b₂ b₁ b₀ ≠ 0)) ∧
  (∃ b₃ b₂ b₁ b₀ : ℝ, is_valid_coefficient b₃ ∧ is_valid_coefficient b₂ ∧
    is_valid_coefficient b₁ ∧ is_valid_coefficient b₀ ∧
    polynomial s b₃ b₂ b₁ b₀ = 0) :=
by sorry

end NUMINAMATH_CALUDE_largest_root_range_l346_34674


namespace NUMINAMATH_CALUDE_sledding_problem_l346_34611

/-- Sledding problem -/
theorem sledding_problem (mary_hill_length : ℝ) (mary_speed : ℝ) (ann_speed : ℝ) (time_difference : ℝ)
  (h1 : mary_hill_length = 630)
  (h2 : mary_speed = 90)
  (h3 : ann_speed = 40)
  (h4 : time_difference = 13) :
  let mary_time := mary_hill_length / mary_speed
  let ann_time := mary_time + time_difference
  ann_speed * ann_time = 800 :=
by sorry

end NUMINAMATH_CALUDE_sledding_problem_l346_34611


namespace NUMINAMATH_CALUDE_cost_per_box_l346_34652

/-- Calculates the cost per box for packaging a fine arts collection --/
theorem cost_per_box (box_length box_width box_height : ℝ)
  (total_volume min_total_cost : ℝ) :
  box_length = 20 ∧ box_width = 20 ∧ box_height = 12 ∧
  total_volume = 2160000 ∧ min_total_cost = 180 →
  (min_total_cost / (total_volume / (box_length * box_width * box_height))) = 0.4 := by
  sorry

end NUMINAMATH_CALUDE_cost_per_box_l346_34652


namespace NUMINAMATH_CALUDE_subset_condition_l346_34673

def A : Set ℝ := {x | 3*x + 6 > 0 ∧ 2*x - 10 < 0}

def B (m : ℝ) : Set ℝ := {x | m + 1 ≤ x ∧ x ≤ 2*m - 1}

theorem subset_condition (m : ℝ) : B m ⊆ A ↔ m < 3 := by sorry

end NUMINAMATH_CALUDE_subset_condition_l346_34673


namespace NUMINAMATH_CALUDE_sally_peach_cost_l346_34608

-- Define the given amounts
def total_spent : ℚ := 23.86
def cherry_cost : ℚ := 11.54

-- Define the amount spent on peaches after coupon
def peach_cost : ℚ := total_spent - cherry_cost

-- Theorem to prove
theorem sally_peach_cost : peach_cost = 12.32 := by
  sorry

end NUMINAMATH_CALUDE_sally_peach_cost_l346_34608


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l346_34639

/-- An arithmetic sequence is a sequence where the difference between
    any two consecutive terms is constant. -/
def is_arithmetic_sequence (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℚ) :
  is_arithmetic_sequence a → a 2 = 5 → a 5 = 33 → a 3 + a 4 = 38 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l346_34639


namespace NUMINAMATH_CALUDE_final_round_probability_final_round_probability_value_l346_34612

def GuessTheCard : Type := Unit

def tournament (n : ℕ) (rounds : ℕ) (win_prob : ℚ) : Type := Unit

theorem final_round_probability
  (n : ℕ)
  (rounds : ℕ)
  (win_prob : ℚ)
  (h1 : n = 16)
  (h2 : rounds = 4)
  (h3 : win_prob = 1 / 2)
  (h4 : ∀ (game : GuessTheCard), ∃! (winner : Unit), true)
  : ℚ :=
by
  sorry

#check final_round_probability

theorem final_round_probability_value
  (n : ℕ)
  (rounds : ℕ)
  (win_prob : ℚ)
  (h1 : n = 16)
  (h2 : rounds = 4)
  (h3 : win_prob = 1 / 2)
  (h4 : ∀ (game : GuessTheCard), ∃! (winner : Unit), true)
  : final_round_probability n rounds win_prob h1 h2 h3 h4 = 1 / 64 :=
by
  sorry

end NUMINAMATH_CALUDE_final_round_probability_final_round_probability_value_l346_34612


namespace NUMINAMATH_CALUDE_min_distance_to_line_l346_34677

/-- The minimum value of (x - 2)^2 + (y - 2)^2 when (x, y) lies on the line x - y - 1 = 0 -/
theorem min_distance_to_line : 
  ∃ (min : ℝ), min = (1/2 : ℝ) ∧ 
  ∀ (x y : ℝ), x - y - 1 = 0 → (x - 2)^2 + (y - 2)^2 ≥ min :=
by sorry

end NUMINAMATH_CALUDE_min_distance_to_line_l346_34677


namespace NUMINAMATH_CALUDE_cube_surface_area_increase_l346_34633

theorem cube_surface_area_increase (L : ℝ) (h : L > 0) :
  let original_area := 6 * L^2
  let new_edge_length := 1.5 * L
  let new_area := 6 * new_edge_length^2
  (new_area - original_area) / original_area * 100 = 125 :=
by sorry

end NUMINAMATH_CALUDE_cube_surface_area_increase_l346_34633


namespace NUMINAMATH_CALUDE_spider_has_eight_legs_l346_34682

/-- The number of legs a human has -/
def human_legs : ℕ := 2

/-- The number of legs a spider has -/
def spider_legs : ℕ := 2 * (2 * human_legs)

/-- Theorem stating that a spider has 8 legs -/
theorem spider_has_eight_legs : spider_legs = 8 := by
  sorry

end NUMINAMATH_CALUDE_spider_has_eight_legs_l346_34682


namespace NUMINAMATH_CALUDE_divide_decimals_l346_34651

theorem divide_decimals : (0.08 : ℚ) / (0.002 : ℚ) = 40 := by sorry

end NUMINAMATH_CALUDE_divide_decimals_l346_34651


namespace NUMINAMATH_CALUDE_a_25_mod_26_l346_34696

/-- Definition of a_n as the integer obtained by concatenating all integers from 1 to n -/
def a (n : ℕ) : ℕ := sorry

/-- Theorem stating that a_25 mod 26 = 13 -/
theorem a_25_mod_26 : a 25 % 26 = 13 := by sorry

end NUMINAMATH_CALUDE_a_25_mod_26_l346_34696


namespace NUMINAMATH_CALUDE_root_sum_fraction_l346_34603

/-- Given a, b, and c are the roots of x^3 - 8x^2 + 11x - 3 = 0,
    prove that (a/(bc - 1)) + (b/(ac - 1)) + (c/(ab - 1)) = 13 -/
theorem root_sum_fraction (a b c : ℝ) 
  (h1 : a^3 - 8*a^2 + 11*a - 3 = 0)
  (h2 : b^3 - 8*b^2 + 11*b - 3 = 0)
  (h3 : c^3 - 8*c^2 + 11*c - 3 = 0) :
  a / (b * c - 1) + b / (a * c - 1) + c / (a * b - 1) = 13 := by
  sorry

end NUMINAMATH_CALUDE_root_sum_fraction_l346_34603


namespace NUMINAMATH_CALUDE_complex_cube_root_sum_l346_34631

-- Define the complex number i
def i : ℂ := Complex.I

-- Define the theorem
theorem complex_cube_root_sum (a b : ℝ) (h : i^3 = a - b*i) : a + b = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_cube_root_sum_l346_34631


namespace NUMINAMATH_CALUDE_dime_probability_l346_34676

def coin_jar (quarter_value dime_value penny_value : ℚ)
             (total_quarter_value total_dime_value total_penny_value : ℚ) : Prop :=
  let quarter_count := total_quarter_value / quarter_value
  let dime_count := total_dime_value / dime_value
  let penny_count := total_penny_value / penny_value
  let total_coins := quarter_count + dime_count + penny_count
  dime_count / total_coins = 1 / 7

theorem dime_probability :
  coin_jar (25/100) (10/100) (1/100) (1250/100) (500/100) (250/100) := by
  sorry

end NUMINAMATH_CALUDE_dime_probability_l346_34676


namespace NUMINAMATH_CALUDE_cubic_equation_solution_l346_34694

theorem cubic_equation_solution :
  let x : ℝ := Real.rpow (19/2) (1/3) - 2
  2 * x^3 + 24 * x = 3 - 12 * x^2 := by
    sorry

end NUMINAMATH_CALUDE_cubic_equation_solution_l346_34694


namespace NUMINAMATH_CALUDE_difference_of_squares_divided_l346_34660

theorem difference_of_squares_divided : (311^2 - 297^2) / 14 = 608 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_divided_l346_34660


namespace NUMINAMATH_CALUDE_ellipse_triangle_perimeter_l346_34657

-- Define the ellipse
def Ellipse (x y : ℝ) : Prop := x^2 / 9 + y^2 / 25 = 1

-- Define the foci
def F1 : ℝ × ℝ := sorry
def F2 : ℝ × ℝ := sorry

-- Define a chord passing through F1
def Chord (A B : ℝ × ℝ) : Prop := sorry

-- Define the perimeter of a triangle
def TrianglePerimeter (A B C : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem ellipse_triangle_perimeter 
  (A B : ℝ × ℝ) 
  (h_ellipse : Ellipse A.1 A.2 ∧ Ellipse B.1 B.2)
  (h_chord : Chord A B) :
  TrianglePerimeter A B F2 = 20 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_triangle_perimeter_l346_34657


namespace NUMINAMATH_CALUDE_vector_magnitude_problem_l346_34687

/-- Given vectors a, b, and c in ℝ², prove that if a - 2b is perpendicular to c, 
    then the magnitude of b is 3√5. -/
theorem vector_magnitude_problem (a b c : ℝ × ℝ) : 
  a = (-2, 1) → 
  b.1 = k ∧ b.2 = -3 → 
  c = (1, 2) → 
  (a.1 - 2 * b.1, a.2 - 2 * b.2) • c = 0 → 
  Real.sqrt (b.1^2 + b.2^2) = 3 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_vector_magnitude_problem_l346_34687


namespace NUMINAMATH_CALUDE_min_sum_squares_l346_34645

def S : Finset Int := {-6, -4, -1, 0, 3, 5, 7, 12}

theorem min_sum_squares (p q r s t u v w : Int) 
  (h_distinct : p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ p ≠ t ∧ p ≠ u ∧ p ≠ v ∧ p ≠ w ∧
                q ≠ r ∧ q ≠ s ∧ q ≠ t ∧ q ≠ u ∧ q ≠ v ∧ q ≠ w ∧
                r ≠ s ∧ r ≠ t ∧ r ≠ u ∧ r ≠ v ∧ r ≠ w ∧
                s ≠ t ∧ s ≠ u ∧ s ≠ v ∧ s ≠ w ∧
                t ≠ u ∧ t ≠ v ∧ t ≠ w ∧
                u ≠ v ∧ u ≠ w ∧
                v ≠ w)
  (h_in_S : p ∈ S ∧ q ∈ S ∧ r ∈ S ∧ s ∈ S ∧ t ∈ S ∧ u ∈ S ∧ v ∈ S ∧ w ∈ S) :
  (p + q + r + s)^2 + (t + u + v + w)^2 ≥ 128 :=
by sorry

end NUMINAMATH_CALUDE_min_sum_squares_l346_34645


namespace NUMINAMATH_CALUDE_intersection_point_l346_34678

noncomputable section

-- Define the functions
def f (x : ℝ) : ℝ := (x^2 - 8*x + 12) / (2*x - 6)
def g (x b c d e : ℝ) : ℝ := (b*x^2 + c*x + d) / (x - e)

-- State the theorem
theorem intersection_point (b c d e : ℝ) :
  -- Conditions
  (∀ x, (2*x - 6 = 0 ↔ x - e = 0)) →  -- Same vertical asymptote
  (∃ k, ∀ x, g x b c d e = -2*x - 4 + k / (x - e)) →  -- Oblique asymptote of g
  (f (-3) = g (-3) b c d e) →  -- Intersection at x = -3
  -- Conclusion
  (∃ x y, x ≠ -3 ∧ f x = g x b c d e ∧ x = 14 ∧ y = 3) :=
by sorry

end NUMINAMATH_CALUDE_intersection_point_l346_34678


namespace NUMINAMATH_CALUDE_sum_of_roots_equals_one_l346_34637

theorem sum_of_roots_equals_one :
  ∀ x₁ x₂ : ℝ,
  (x₁ + 3) * (x₁ - 4) = 18 →
  (x₂ + 3) * (x₂ - 4) = 18 →
  x₁ + x₂ = 1 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_roots_equals_one_l346_34637


namespace NUMINAMATH_CALUDE_domain_of_composite_function_l346_34664

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the domain of f
def domain_f : Set ℝ := Set.Icc (-1) 2

-- State the theorem
theorem domain_of_composite_function :
  {x : ℝ | f (3 - 2*x) ∈ Set.range f} = Set.Icc (1/2) 2 :=
sorry

end NUMINAMATH_CALUDE_domain_of_composite_function_l346_34664


namespace NUMINAMATH_CALUDE_test_problems_l346_34614

theorem test_problems (total_problems : ℕ) (comp_points : ℕ) (word_points : ℕ) (total_points : ℕ) :
  total_problems = 30 →
  comp_points = 3 →
  word_points = 5 →
  total_points = 110 →
  ∃ (comp_count : ℕ) (word_count : ℕ),
    comp_count + word_count = total_problems ∧
    comp_count * comp_points + word_count * word_points = total_points ∧
    comp_count = 20 := by
  sorry

end NUMINAMATH_CALUDE_test_problems_l346_34614


namespace NUMINAMATH_CALUDE_place_value_ratio_l346_34697

theorem place_value_ratio : 
  let number : ℚ := 56842.7093
  let digit_8_place_value : ℚ := 1000
  let digit_7_place_value : ℚ := 0.1
  digit_8_place_value / digit_7_place_value = 10000 := by
  sorry

end NUMINAMATH_CALUDE_place_value_ratio_l346_34697


namespace NUMINAMATH_CALUDE_expression_evaluation_l346_34618

theorem expression_evaluation (x y : ℝ) (hx : x = 3) (hy : y = 5) : 
  (3 * x^4 + 2 * y^2 + 10) / 8 = 303 / 8 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l346_34618


namespace NUMINAMATH_CALUDE_birds_cannot_all_be_on_same_tree_l346_34638

/-- Represents the state of birds on trees -/
structure BirdState where
  white : Nat -- Number of birds on white trees
  black : Nat -- Number of birds on black trees

/-- A move represents two birds switching to neighboring trees -/
def move (state : BirdState) : BirdState :=
  { white := state.white, black := state.black }

theorem birds_cannot_all_be_on_same_tree :
  ∀ (n : Nat), n > 0 →
  let initial_state : BirdState := { white := 3, black := 3 }
  let final_state := (move^[n]) initial_state
  (final_state.white ≠ 0 ∧ final_state.black ≠ 6) ∧
  (final_state.white ≠ 6 ∧ final_state.black ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_birds_cannot_all_be_on_same_tree_l346_34638


namespace NUMINAMATH_CALUDE_vector_linear_combination_l346_34643

/-- Given vectors a, b, and c in ℝ², prove that c can be expressed as a linear combination of a and b -/
theorem vector_linear_combination (a b c : ℝ × ℝ) 
  (ha : a = (1, 1)) 
  (hb : b = (1, -1)) 
  (hc : c = (-1, 2)) : 
  c = (1/2 : ℝ) • a - (3/2 : ℝ) • b :=
sorry

end NUMINAMATH_CALUDE_vector_linear_combination_l346_34643


namespace NUMINAMATH_CALUDE_quadratic_root_sum_bound_l346_34669

theorem quadratic_root_sum_bound (a b : ℝ) :
  let f : ℝ → ℝ := λ x ↦ x^2 + a*x + b
  let roots := {x : ℝ | f x = 0}
  (∃ m n : ℝ, m ∈ roots ∧ n ∈ roots ∧ abs m + abs n ≤ 1) →
  -1/4 ≤ b ∧ b < 1/4 := by
sorry

end NUMINAMATH_CALUDE_quadratic_root_sum_bound_l346_34669


namespace NUMINAMATH_CALUDE_greatest_multiple_of_5_and_7_less_than_1000_l346_34659

theorem greatest_multiple_of_5_and_7_less_than_1000 :
  ∀ n : ℕ, n < 1000 → n % 5 = 0 → n % 7 = 0 → n ≤ 980 :=
by
  sorry

end NUMINAMATH_CALUDE_greatest_multiple_of_5_and_7_less_than_1000_l346_34659


namespace NUMINAMATH_CALUDE_quadratic_function_bound_l346_34653

theorem quadratic_function_bound (a b : ℝ) :
  (∃ m : ℝ, |m^2 + a*m + b| ≤ 1/4 ∧ |(m+1)^2 + a*(m+1) + b| ≤ 1/4) →
  0 ≤ a^2 - 4*b ∧ a^2 - 4*b ≤ 2 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_bound_l346_34653


namespace NUMINAMATH_CALUDE_david_shells_l346_34667

theorem david_shells (david mia ava alice : ℕ) : 
  mia = 4 * david →
  ava = mia + 20 →
  alice = ava / 2 →
  david + mia + ava + alice = 195 →
  david = 15 := by
sorry

end NUMINAMATH_CALUDE_david_shells_l346_34667


namespace NUMINAMATH_CALUDE_plane_perpendicularity_condition_l346_34683

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the properties and relations
variable (subset : Line → Plane → Prop)
variable (perpendicular_line_plane : Line → Plane → Prop)
variable (perpendicular_plane_plane : Plane → Plane → Prop)

-- State the theorem
theorem plane_perpendicularity_condition 
  (α β : Plane) (a : Line) 
  (h_subset : subset a α) :
  (∀ (a : Line), subset a α → perpendicular_line_plane a β → perpendicular_plane_plane α β) ∧ 
  (∃ (a : Line), subset a α ∧ perpendicular_plane_plane α β ∧ ¬perpendicular_line_plane a β) :=
sorry

end NUMINAMATH_CALUDE_plane_perpendicularity_condition_l346_34683


namespace NUMINAMATH_CALUDE_correct_matching_probability_l346_34680

def num_celebrities : ℕ := 3
def num_baby_photos : ℕ := 3

theorem correct_matching_probability :
  let total_arrangements := Nat.factorial num_celebrities
  let correct_arrangements := 1
  (correct_arrangements : ℚ) / total_arrangements = 1 / 6 := by
  sorry

end NUMINAMATH_CALUDE_correct_matching_probability_l346_34680


namespace NUMINAMATH_CALUDE_complex_number_equality_l346_34641

theorem complex_number_equality (z : ℂ) : z * (1 + Complex.I * Real.sqrt 3) = Complex.abs (1 + Complex.I * Real.sqrt 3) → z = 1 - Complex.I * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_equality_l346_34641


namespace NUMINAMATH_CALUDE_filter_kit_price_l346_34689

theorem filter_kit_price :
  let individual_prices : List ℝ := [12.45, 12.45, 14.05, 14.05, 11.50]
  let total_individual_price := individual_prices.sum
  let discount_percentage : ℝ := 11.03448275862069 / 100
  let kit_price := total_individual_price * (1 - discount_percentage)
  kit_price = 57.382758620689655 := by
sorry

end NUMINAMATH_CALUDE_filter_kit_price_l346_34689


namespace NUMINAMATH_CALUDE_gcd_problem_l346_34609

theorem gcd_problem (p : Nat) (h : Nat.Prime p) (hp : p = 107) :
  Nat.gcd (p^7 + 1) (p^7 + p^3 + 1) = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_problem_l346_34609


namespace NUMINAMATH_CALUDE_equality_of_abc_l346_34679

theorem equality_of_abc (a b c : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0)
  (h : a^2 * (b + c - a) = b^2 * (c + a - b) ∧ b^2 * (c + a - b) = c^2 * (a + b - c)) :
  a = b ∧ b = c := by
  sorry

end NUMINAMATH_CALUDE_equality_of_abc_l346_34679


namespace NUMINAMATH_CALUDE_xyz_squared_sum_l346_34686

theorem xyz_squared_sum (x y z : ℝ) 
  (h_arithmetic : (x + y + z) / 3 = 10)
  (h_geometric : (x * y * z) ^ (1/3 : ℝ) = 6)
  (h_harmonic : 3 / (1/x + 1/y + 1/z) = 4) :
  x^2 + y^2 + z^2 = 576 := by
sorry

end NUMINAMATH_CALUDE_xyz_squared_sum_l346_34686


namespace NUMINAMATH_CALUDE_area_fraction_to_CD_l346_34606

/-- Represents a trapezoid ABCD with specific properties -/
structure Trapezoid where
  -- AB is parallel to CD and AB < CD
  AB : ℝ
  CD : ℝ
  h_parallel : AB < CD
  -- ∠BAD = 45° and ∠ABC = 135°
  angle_BAD : ℝ
  angle_ABC : ℝ
  h_angles : angle_BAD = π/4 ∧ angle_ABC = 3*π/4
  -- AD = BC = 100 m
  AD : ℝ
  BC : ℝ
  h_sides : AD = 100 ∧ BC = 100
  -- AB = 80 m
  h_AB : AB = 80
  -- CD > 100 m
  h_CD : CD > 100

/-- The fraction of the area closer to CD than to AB is approximately 3/4 -/
theorem area_fraction_to_CD (t : Trapezoid) : 
  ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ 
  |((t.CD - t.AB) * t.AD / (2 * (t.AB + t.CD) * t.AD)) - 3/4| < ε :=
sorry

end NUMINAMATH_CALUDE_area_fraction_to_CD_l346_34606


namespace NUMINAMATH_CALUDE_exact_one_common_point_chord_length_when_m_4_l346_34663

-- Define the curve C
def curve_C (t : ℝ) : ℝ × ℝ := (4 * t^2, 4 * t)

-- Define the line l in polar form
def line_l (m : ℝ) (ρ θ : ℝ) : Prop := ρ * (4 * Real.cos θ + 3 * Real.sin θ) - m = 0

-- Theorem 1: Value of m for exactly one common point
theorem exact_one_common_point :
  ∃ (m : ℝ), m = -9/4 ∧
  (∃! (t : ℝ), ∃ (ρ θ : ℝ), curve_C t = (ρ * Real.cos θ, ρ * Real.sin θ) ∧ line_l m ρ θ) :=
sorry

-- Theorem 2: Length of chord when m = 4
theorem chord_length_when_m_4 :
  let m := 4
  ∃ (t₁ t₂ : ℝ), t₁ ≠ t₂ ∧
  (∃ (ρ₁ θ₁ ρ₂ θ₂ : ℝ), 
    curve_C t₁ = (ρ₁ * Real.cos θ₁, ρ₁ * Real.sin θ₁) ∧ 
    curve_C t₂ = (ρ₂ * Real.cos θ₂, ρ₂ * Real.sin θ₂) ∧
    line_l m ρ₁ θ₁ ∧ line_l m ρ₂ θ₂) ∧
  let (x₁, y₁) := curve_C t₁
  let (x₂, y₂) := curve_C t₂
  Real.sqrt ((x₂ - x₁)^2 + (y₂ - y₁)^2) = 25/4 :=
sorry

end NUMINAMATH_CALUDE_exact_one_common_point_chord_length_when_m_4_l346_34663


namespace NUMINAMATH_CALUDE_money_theorem_l346_34621

/-- Given the conditions on c and d, prove that c > 12.4 and d < 24 -/
theorem money_theorem (c d : ℝ) 
  (h1 : 7 * c - d > 80)
  (h2 : 4 * c + d = 44)
  (h3 : d < 2 * c) :
  c > 12.4 ∧ d < 24 := by
  sorry

end NUMINAMATH_CALUDE_money_theorem_l346_34621


namespace NUMINAMATH_CALUDE_cherry_pies_count_l346_34619

/-- Represents the types of pies --/
inductive PieType
  | Apple
  | Blueberry
  | Cherry

/-- Calculates the number of cherry pies given the total number of pies and the ratio --/
def cherry_pies (total : ℕ) (apple_ratio : ℕ) (blueberry_ratio : ℕ) (cherry_ratio : ℕ) : ℕ :=
  let ratio_sum := apple_ratio + blueberry_ratio + cherry_ratio
  let pies_per_ratio := total / ratio_sum
  cherry_ratio * pies_per_ratio

/-- Theorem stating that given 30 total pies and a 1:5:4 ratio, there are 12 cherry pies --/
theorem cherry_pies_count : cherry_pies 30 1 5 4 = 12 := by
  sorry

end NUMINAMATH_CALUDE_cherry_pies_count_l346_34619


namespace NUMINAMATH_CALUDE_jeff_scores_mean_l346_34607

theorem jeff_scores_mean : 
  let scores : List ℕ := [89, 92, 88, 95, 91, 93]
  (scores.sum : ℚ) / scores.length = 548 / 6 := by sorry

end NUMINAMATH_CALUDE_jeff_scores_mean_l346_34607


namespace NUMINAMATH_CALUDE_product_mod_five_l346_34646

theorem product_mod_five : (1234 * 5678) % 5 = 2 := by
  sorry

end NUMINAMATH_CALUDE_product_mod_five_l346_34646
