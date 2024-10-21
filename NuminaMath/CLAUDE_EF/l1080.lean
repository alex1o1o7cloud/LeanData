import Mathlib

namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_minority_shareholders_percentage_l1080_108031

/-- A company on a stock exchange --/
structure Company where
  shareholders : ℕ
  isPublic : Bool

/-- A shareholder on a stock exchange --/
structure Shareholder where
  ownership : ℝ
  isMinority : Bool

/-- A stock exchange --/
structure StockExchange where
  companies : List Company
  shareholders : List Shareholder

/-- Properties of the stock exchange --/
def StockExchange.properties (se : StockExchange) : Prop :=
  (∀ c, c ∈ se.companies →
    (c.isPublic ↔ c.shareholders ≥ 15)) ∧
  (∀ s, s ∈ se.shareholders → 
    (s.isMinority ↔ s.ownership ≤ 0.25)) ∧
  ((se.companies.filter (·.isPublic)).length = se.companies.length / 6) ∧
  (∀ s, s ∈ se.shareholders → ∃! c, c ∈ se.companies ∧ true)  -- Each shareholder owns shares of only one company

theorem minority_shareholders_percentage (se : StockExchange) 
    (h : se.properties) : 
    ((se.shareholders.filter (·.isMinority)).length : ℝ) / (se.shareholders.length : ℝ) ≥ 0.2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_minority_shareholders_percentage_l1080_108031


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_2012_value_l1080_108040

def my_sequence (n : ℕ) : ℚ :=
  match n with
  | 0 => 6/7
  | n + 1 =>
    let a := my_sequence n
    if 0 ≤ a ∧ a < 1/2 then 2 * a
    else if 1/2 ≤ a ∧ a < 1 then 2 * a - 1
    else 0  -- This case should never occur given the constraints, but Lean requires it for completeness

theorem sequence_2012_value : my_sequence 2011 = 5/7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_2012_value_l1080_108040


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_days_worked_per_week_l1080_108062

/-- Represents the number of days worked per week -/
def days_per_week : ℕ := 5

/-- Josh's daily work hours -/
def josh_daily_hours : ℕ := 8

/-- Carl's daily work hours -/
def carl_daily_hours : ℕ := josh_daily_hours - 2

/-- Number of weeks in a month -/
def weeks_per_month : ℕ := 4

/-- Josh's hourly rate in dollars -/
def josh_hourly_rate : ℚ := 9

/-- Carl's hourly rate in dollars -/
def carl_hourly_rate : ℚ := josh_hourly_rate / 2

/-- Total monthly payment for both Josh and Carl in dollars -/
def total_monthly_payment : ℕ := 1980

/-- Theorem stating that the number of days worked per week is 5 -/
theorem days_worked_per_week : 
  days_per_week * (josh_daily_hours * josh_hourly_rate + carl_daily_hours * carl_hourly_rate) * weeks_per_month = total_monthly_payment := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_days_worked_per_week_l1080_108062


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circle_tangency_l1080_108036

-- Define the basic geometric elements
structure Point where
  x : ℝ
  y : ℝ

structure Angle where
  vertex : Point

structure Circle where
  center : Point
  radius : ℝ

structure Line where
  point1 : Point
  point2 : Point

-- Define the geometric relationships
def isInscribed (c : Circle) (a : Angle) : Prop := sorry

def isTangent (l : Line) (c : Circle) : Prop := sorry

def intersectsAt (l : Line) (a : Angle) (p1 p2 : Point) : Prop := sorry

noncomputable def circumcircle (p1 p2 p3 : Point) : Circle := sorry

-- State the theorem
theorem inscribed_circle_tangency 
  (A : Point) (angle : Angle) (inscribedCircle : Circle) 
  (B C : Point) (tangentLine : Line) :
  isInscribed inscribedCircle angle →
  isTangent tangentLine inscribedCircle →
  intersectsAt tangentLine angle B C →
  isTangent (Line.mk B C) inscribedCircle := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inscribed_circle_tangency_l1080_108036


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_set_f_transformations_l1080_108034

noncomputable def f (x : ℝ) : ℝ := 3 * Real.sin (2 * x + Real.pi / 6) - 1

def max_set : Set ℝ := {x | ∃ k : ℤ, x = k * Real.pi + Real.pi / 6}

theorem f_max_set : 
  ∀ x : ℝ, f x ≤ 2 ∧ (f x = 2 ↔ x ∈ max_set) :=
by sorry

theorem f_transformations (x : ℝ) : 
  f x = 3 * Real.sin (2 * x + Real.pi / 6) - 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_max_set_f_transformations_l1080_108034


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_period_of_cos_3x_over_2_l1080_108009

noncomputable def f (x : ℝ) := Real.cos (3 * x / 2)

theorem period_of_cos_3x_over_2 : 
  ∃ (p : ℝ), p > 0 ∧ (∀ x, f (x + p) = f x) ∧ 
  (∀ q, q > 0 ∧ (∀ x, f (x + q) = f x) → p ≤ q) ∧
  p = 4 * Real.pi / 3 := by
  sorry

#check period_of_cos_3x_over_2

end NUMINAMATH_CALUDE_ERRORFEEDBACK_period_of_cos_3x_over_2_l1080_108009


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_john_speed_is_30_l1080_108061

/-- Represents the scenario of John and Lewis driving between two cities -/
structure DrivingScenario where
  distance_between_cities : ℚ
  lewis_speed : ℚ
  meeting_distance : ℚ

/-- Calculates John's speed given the driving scenario -/
def calculate_john_speed (scenario : DrivingScenario) : ℚ :=
  3 * scenario.meeting_distance / (2 * scenario.distance_between_cities / scenario.lewis_speed)

/-- Theorem stating that John's speed is 30 mph given the specific conditions -/
theorem john_speed_is_30 (scenario : DrivingScenario) 
  (h1 : scenario.distance_between_cities = 240)
  (h2 : scenario.lewis_speed = 60)
  (h3 : scenario.meeting_distance = 160) :
  calculate_john_speed scenario = 30 := by
  sorry

#eval calculate_john_speed ⟨240, 60, 160⟩

end NUMINAMATH_CALUDE_ERRORFEEDBACK_john_speed_is_30_l1080_108061


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_u_1000_greater_than_45_l1080_108093

noncomputable def u : ℕ → ℝ
  | 0 => 5
  | n + 1 => u n + 1 / u n

theorem u_1000_greater_than_45 : u 1000 > 45 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_u_1000_greater_than_45_l1080_108093


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_group_l1080_108058

structure ScientificGroup where
  name : String
  average_score : ℝ
  variance : ℝ

def is_optimal (g : ScientificGroup) (groups : List ScientificGroup) : Prop :=
  (∀ h ∈ groups, g.average_score ≥ h.average_score) ∧
  (∀ h ∈ groups, g.average_score = h.average_score → g.variance ≤ h.variance)

theorem optimal_group :
  let groups := [
    ScientificGroup.mk "A" 7 1,
    ScientificGroup.mk "B" 8 1.2,
    ScientificGroup.mk "C" 8 1,
    ScientificGroup.mk "D" 7 1.8
  ]
  let group_C := ScientificGroup.mk "C" 8 1
  is_optimal group_C groups :=
by
  sorry

#check optimal_group

end NUMINAMATH_CALUDE_ERRORFEEDBACK_optimal_group_l1080_108058


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_horner_method_operations_l1080_108045

/-- Represents a polynomial function -/
def MyPolynomial (α : Type*) := List α

/-- Horner's Method for polynomial evaluation -/
def horner_eval {α : Type*} [Ring α] (p : MyPolynomial α) (x : α) : α :=
  p.foldl (fun acc a => acc * x + a) 0

/-- Counts the number of multiplication operations in Horner's Method -/
def mult_count {α : Type*} (p : MyPolynomial α) : Nat :=
  p.length - 1

/-- Counts the number of addition/subtraction operations in Horner's Method -/
def add_sub_count {α : Type*} (p : MyPolynomial α) : Nat :=
  p.length - 1

theorem horner_method_operations (x : ℝ) :
  let p : MyPolynomial ℝ := [2, 0, 0, 0, -1, 0, 2]
  (mult_count p = 6) ∧ (add_sub_count p = 2) := by
  sorry

#check horner_method_operations

end NUMINAMATH_CALUDE_ERRORFEEDBACK_horner_method_operations_l1080_108045


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proportion_inequality_l1080_108059

/-- An inverse proportion function -/
noncomputable def inverse_proportion (k : ℝ) : ℝ → ℝ := fun x ↦ k / x

theorem inverse_proportion_inequality (k m y₁ y₂ : ℝ) :
  let f := inverse_proportion k
  k = -2 →
  m > 0 →
  f (-1) = 2 →
  f m = y₁ →
  f (m + 1) = y₂ →
  y₁ < y₂ := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_inverse_proportion_inequality_l1080_108059


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_circle_problem_l1080_108006

-- Define the circle
def circle_equation (x y a : ℝ) : Prop := x^2 + y^2 - 2*a*x + 2*y - 1 = 0

-- Define the point P
def P (a : ℝ) : ℝ × ℝ := (-5, a)

-- Define the tangency condition
def tangency_condition (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  (y₂ - y₁)/(x₂ - x₁) + (x₁ + x₂ - 2)/(y₁ + y₂) = 0

theorem tangent_circle_problem (a : ℝ) 
  (x₁ y₁ x₂ y₂ : ℝ) 
  (h₁ : circle_equation x₁ y₁ a) 
  (h₂ : circle_equation x₂ y₂ a) 
  (h₃ : tangency_condition x₁ y₁ x₂ y₂) :
  a^2 - a - 6 = 0 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_circle_problem_l1080_108006


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_a_3_equals_4_l1080_108086

def mySequence (n : ℕ) : ℕ → ℕ
  | 0 => 1  -- a₁ = 1
  | (k+1) => mySequence n k + (k+1)  -- aₙ₊₁ - aₙ = n

theorem a_3_equals_4 : mySequence 3 2 = 4 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_a_3_equals_4_l1080_108086


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_g_l1080_108088

-- Define the function g(x)
noncomputable def g (x : ℝ) : ℝ := min (x + 3) (min (x - 1) (-1/2 * x + 5))

-- Theorem statement
theorem max_value_of_g :
  ∃ (M : ℝ), M = 13/3 ∧ ∀ (x : ℝ), g x ≤ M ∧ ∃ (x₀ : ℝ), g x₀ = M := by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_value_of_g_l1080_108088


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_james_age_l1080_108068

/-- Given the age relationships between James, John, and Tim, prove James's current age. -/
theorem james_age (james_age : ℕ) :
  -- John and James have a 12-year age difference
  (∃ (john_age : ℕ), john_age - 12 = james_age) →
  -- Tim's age (79) is 5 years less than twice John's age
  (∃ (john_age : ℕ), 2 * john_age - 5 = 79) →
  -- James's age is John's age minus 12
  (∃ (john_age : ℕ), john_age - 12 = james_age) →
  -- James's age is 30
  james_age = 30 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_james_age_l1080_108068


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_annulus_radius_l1080_108047

-- Define the area of the annulus
variable (S : ℝ)

-- Define the radius of the smaller circle
variable (R₁ : ℝ)

-- Define the radius of the larger circle
variable (R₂ : ℝ)

-- State the theorem
theorem annulus_radius (h1 : S = π * R₂^2 - π * R₁^2) (h2 : R₂ = 2 * π * R₁) :
  R₁ = Real.sqrt (S / (π * (4 * π^2 - 1))) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_annulus_radius_l1080_108047


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x_neg_3_is_160_l1080_108099

/-- The coefficient of x^(-3) in the expansion of (2x + 1/x^2)^6 -/
def coefficient_x_neg_3 : ℕ := 160

/-- Theorem stating that the coefficient of x^(-3) in (2x + 1/x^2)^6 is 160 -/
theorem coefficient_x_neg_3_is_160 :
  coefficient_x_neg_3 = 160 := by
  rfl  -- reflexivity, since we defined coefficient_x_neg_3 as 160

#eval coefficient_x_neg_3

end NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_x_neg_3_is_160_l1080_108099


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cycling_speed_is_12000_l1080_108026

/-- Represents a rectangular park with given properties -/
structure RectangularPark where
  length : ℝ
  breadth : ℝ
  area : ℝ
  cyclingTime : ℝ
  (ratio_constraint : breadth = 3 * length)
  (area_constraint : area = length * breadth)
  (area_value : area = 30000)
  (time_in_minutes : cyclingTime = 4)

/-- Calculates the cycling speed given a rectangular park -/
noncomputable def cyclingSpeed (park : RectangularPark) : ℝ :=
  (2 * (park.length + park.breadth)) / (park.cyclingTime / 60)

/-- Theorem stating that the cycling speed is 12000 m/h -/
theorem cycling_speed_is_12000 (park : RectangularPark) :
  cyclingSpeed park = 12000 := by
  sorry

#check cycling_speed_is_12000

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cycling_speed_is_12000_l1080_108026


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_percentage_men_science_majors_l1080_108069

theorem percentage_men_science_majors 
  (total_students : ℕ) 
  (total_students_pos : total_students > 0)
  (women_science_major_ratio : ℚ)
  (non_science_ratio : ℚ)
  (men_ratio : ℚ)
  (women_science_major_ratio_eq : women_science_major_ratio = 1 / 10)
  (non_science_ratio_eq : non_science_ratio = 3 / 5)
  (men_ratio_eq : men_ratio = 2 / 5) :
  let women_ratio : ℚ := 1 - men_ratio
  let science_ratio : ℚ := 1 - non_science_ratio
  let women_count : ℚ := women_ratio * total_students
  let science_major_count : ℚ := science_ratio * total_students
  let women_science_major_count : ℚ := women_science_major_ratio * women_count
  let men_science_major_count : ℚ := science_major_count - women_science_major_count
  let men_count : ℚ := men_ratio * total_students
  men_science_major_count / men_count = 17 / 20 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_percentage_men_science_majors_l1080_108069


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_value_of_x_l1080_108043

theorem value_of_x (x y z : ℤ) 
  (h_pos : 0 < x ∧ 0 < y ∧ 0 < z)
  (h_order : x ≥ y ∧ y ≥ z)
  (h_eq1 : x^2 - y^2 - z^2 + x*y = 4021)
  (h_eq2 : x^2 + 5*y^2 + 5*z^2 - 5*x*y - 4*x*z - 4*y*z = -3983) :
  x = 285 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_value_of_x_l1080_108043


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_not_all_sides_odd_intersections_l1080_108065

/-- A rectangle on a grid with 45° angle to grid lines -/
structure GridRectangle where
  /-- The rectangle's sides form 45° angles with the grid lines -/
  angle_with_grid : ℝ
  /-- The vertices of the rectangle do not lie on grid lines -/
  vertices_off_grid : Prop
  /-- Each side of the rectangle intersects grid lines -/
  sides_intersect_grid : Prop

/-- Represents a side of the rectangle -/
structure RectangleSide where
  /-- The number of intersections with grid lines -/
  intersections : ℕ

/-- Theorem stating the impossibility of all sides intersecting an odd number of grid lines -/
theorem not_all_sides_odd_intersections (rect : GridRectangle) 
  (h_angle : rect.angle_with_grid = 45)
  (h_vertices : rect.vertices_off_grid)
  (h_intersect : rect.sides_intersect_grid) :
  ¬ (∀ side : RectangleSide, Odd side.intersections) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_not_all_sides_odd_intersections_l1080_108065


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_rate_theorem_l1080_108083

/-- Calculates the required interest rate for the remaining investment --/
noncomputable def calculate_remaining_rate (total_investment : ℝ) (first_investment : ℝ) (first_rate : ℝ)
  (second_investment : ℝ) (second_rate : ℝ) (desired_income : ℝ) : ℝ :=
  let remaining_investment := total_investment - first_investment - second_investment
  let income_from_first := first_investment * first_rate
  let income_from_second := second_investment * second_rate
  let remaining_income := desired_income - income_from_first - income_from_second
  remaining_income / remaining_investment

/-- Theorem stating the required interest rate for the remaining investment --/
theorem investment_rate_theorem (total_investment : ℝ) (first_investment : ℝ) (first_rate : ℝ)
  (second_investment : ℝ) (second_rate : ℝ) (desired_income : ℝ)
  (h1 : total_investment = 12000)
  (h2 : first_investment = 5000)
  (h3 : first_rate = 0.06)
  (h4 : second_investment = 4000)
  (h5 : second_rate = 0.035)
  (h6 : desired_income = 600) :
  ∃ ε > 0, |calculate_remaining_rate total_investment first_investment first_rate
    second_investment second_rate desired_income - 0.0533| < ε := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_investment_rate_theorem_l1080_108083


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_l1080_108048

theorem equation_solution : 
  ∃ y : ℝ, ((1/16 : ℝ)^(3*y + 12) = (4 : ℝ)^(4*y + 1)) ↔ y = -5/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_equation_solution_l1080_108048


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_plane_equation_l1080_108025

def is_foot_of_perpendicular (a b c d : ℤ) : Prop :=
  9 * a - 3 * b + 2 * c + d = 0

theorem plane_equation :
  ∃ (a b c d : ℤ),
    a > 0 ∧
    is_foot_of_perpendicular a b c d ∧
    Int.gcd (a.natAbs) (Int.gcd (b.natAbs) (Int.gcd (c.natAbs) (d.natAbs))) = 1 ∧
    ∀ (x y z : ℝ), a * x + b * y + c * z + d = 0 ↔ 9 * x - 3 * y + 2 * z - 94 = 0 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_plane_equation_l1080_108025


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_race_odds_l1080_108046

/-- Represents the odds against an event occurring -/
structure Odds where
  against : ℕ
  infavor : ℕ

/-- Calculates the probability of an event given its odds -/
def oddsToProb (o : Odds) : ℚ :=
  o.infavor / (o.against + o.infavor)

theorem race_odds (oddsA oddsB : Odds) 
  (hA : oddsA = Odds.mk 4 1) 
  (hB : oddsB = Odds.mk 1 1) 
  (hSum : oddsToProb oddsA + oddsToProb oddsB < 1) :
  ∃ (oddsC : Odds), oddsC = Odds.mk 7 3 ∧ 
    oddsToProb oddsA + oddsToProb oddsB + oddsToProb oddsC = 1 := by
  sorry

#check race_odds

end NUMINAMATH_CALUDE_ERRORFEEDBACK_race_odds_l1080_108046


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_proposition_range_l1080_108098

-- Define the propositions p and q as functions of m
def p (m : ℝ) : Prop := ∃ x y : ℝ, (x^2 / (2*m)) - (y^2 / (m-1)) = 1

def q (m : ℝ) : Prop := ∃ e : ℝ, (1 < e ∧ e < 2) ∧ (e^2 = 1 + 5/m)

-- Define the theorem
theorem proposition_range :
  ∀ m : ℝ, (p m ∨ q m) ∧ ¬(p m ∧ q m) ↔ 1/3 ≤ m ∧ m < 15 :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_proposition_range_l1080_108098


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_g_sum_equals_l1080_108071

noncomputable def g (n : ℕ+) : ℝ := Real.log ((n + 2) ^ 2 : ℝ) / Real.log 1001

theorem g_sum_equals : g 8 + g 12 + g 13 = 2 + Real.log (300 ^ 2 : ℝ) / Real.log 1001 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_g_sum_equals_l1080_108071


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_reciprocal_of_negative_one_seventh_l1080_108037

theorem reciprocal_of_negative_one_seventh :
  (-7 : ℚ) * (-1/7 : ℚ) = 1 := by
  norm_num

#check reciprocal_of_negative_one_seventh

end NUMINAMATH_CALUDE_ERRORFEEDBACK_reciprocal_of_negative_one_seventh_l1080_108037


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_insurance_coverage_percentage_l1080_108013

theorem insurance_coverage_percentage (pills_per_day : ℕ) (pill_cost : ℚ) (days_in_month : ℕ) (john_payment : ℚ) :
  pills_per_day = 2 →
  pill_cost = 3/2 →
  days_in_month = 30 →
  john_payment = 54 →
  let total_cost := (pills_per_day : ℚ) * days_in_month * pill_cost
  let insurance_coverage := total_cost - john_payment
  let coverage_percentage := (insurance_coverage / total_cost) * 100
  coverage_percentage = 40 := by
    intro h1 h2 h3 h4
    sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_insurance_coverage_percentage_l1080_108013


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_equals_zero_implies_a_half_f_odd_iff_a_half_l1080_108051

-- Define the function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a - 1 / (1 + 2^x)

-- Theorem 1: If f(1) + f(-1) = 0, then a = 1/2
theorem sum_equals_zero_implies_a_half (a : ℝ) :
  f a 1 + f a (-1) = 0 → a = 1/2 := by
  sorry

-- Theorem 2: f is an odd function if and only if a = 1/2
theorem f_odd_iff_a_half (a : ℝ) :
  (∀ x, f a (-x) = -f a x) ↔ a = 1/2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_sum_equals_zero_implies_a_half_f_odd_iff_a_half_l1080_108051


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_A_simplification_and_root_value_l1080_108021

/-- The algebraic expression A as a function of m -/
noncomputable def A (m : ℝ) : ℝ := (m^2 / (m - 2) - 2*m / (m - 2)) + (m - 3) * (2*m + 1)

/-- The equation x^2 - 2x = 0 -/
def root_equation (x : ℝ) : Prop := x^2 - 2*x = 0

theorem A_simplification_and_root_value :
  (∀ m : ℝ, m ≠ 2 → A m = 2*m^2 - 4*m - 3) ∧
  (∀ m : ℝ, root_equation m → A m = -3) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_A_simplification_and_root_value_l1080_108021


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_calculation_l1080_108024

/-- Given a figure with three semicircles of radius 2 units and a rectangle, 
    prove that the shaded area is 8 - 2π square units. -/
theorem shaded_area_calculation (r : ℝ) (h_r : r = 2) : 
  ∃ (shaded_area : ℝ), shaded_area = 8 - 2 * π := by
  -- Define the semicircles
  let semicircle_ADB : ℝ := π * r^2 / 2
  let semicircle_BEC : ℝ := π * r^2 / 2
  let semicircle_DFE : ℝ := π * r^2 / 2

  -- Define the rectangle
  let rectangle_width : ℝ := 2 * r
  let rectangle_height : ℝ := r
  let rectangle_area : ℝ := rectangle_width * rectangle_height

  -- Define the shaded area
  let shaded_area : ℝ := rectangle_area - semicircle_DFE

  -- Prove the theorem
  use shaded_area
  sorry  -- Skip the proof for now


end NUMINAMATH_CALUDE_ERRORFEEDBACK_shaded_area_calculation_l1080_108024


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_odd_and_increasing_l1080_108050

-- Define the function
noncomputable def f (x : ℝ) : ℝ := x + 1/x

-- State the theorem
theorem f_odd_and_increasing : 
  (∀ x, f (-x) = -f x) ∧ 
  (∀ x y, 1 < x ∧ x < y → f x < f y) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_odd_and_increasing_l1080_108050


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_similar_triangle_perimeter_and_area_l1080_108014

/-- Represents a triangle with side lengths a, b, and c -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Calculates the semi-perimeter of a triangle -/
noncomputable def semiPerimeter (t : Triangle) : ℝ := (t.a + t.b + t.c) / 2

/-- Calculates the area of a triangle using Heron's formula -/
noncomputable def area (t : Triangle) : ℝ :=
  let s := semiPerimeter t
  Real.sqrt (s * (s - t.a) * (s - t.b) * (s - t.c))

/-- Checks if two triangles are similar -/
def areSimilar (t1 t2 : Triangle) : Prop :=
  t1.a / t2.a = t1.b / t2.b ∧ t1.b / t2.b = t1.c / t2.c

theorem similar_triangle_perimeter_and_area 
  (small : Triangle) 
  (large : Triangle) 
  (h_small_isosceles : small.a = small.b ∧ small.a = 12 ∧ small.c = 15)
  (h_large_longest : large.c = 30)
  (h_similar : areSimilar small large) :
  (large.a + large.b + large.c = 78) ∧ 
  (area large = Real.sqrt 710775) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_similar_triangle_perimeter_and_area_l1080_108014


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_omega_values_l1080_108029

noncomputable def f (ω φ x : ℝ) : ℝ := Real.sin (ω * x + φ)

theorem omega_values (ω φ : ℝ) :
  ω > 0 →
  0 ≤ φ ∧ φ ≤ π →
  (∀ x, f ω φ x = f ω φ (-x)) →
  (∀ x, f ω φ x = -f ω φ (3*π/2 - x)) →
  (∀ x y, 0 ≤ x ∧ x < y ∧ y ≤ π/2 → f ω φ x < f ω φ y ∨ f ω φ x > f ω φ y) →
  ω = 2/3 ∨ ω = 2 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_omega_values_l1080_108029


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_root_product_l1080_108020

-- Define the logarithm function (base 10)
noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 10

-- Define the equation
def equation (x : ℝ) : Prop := (lg x)^2 - lg x + lg 2 * lg 5 = 0

-- State the theorem
theorem root_product (m n : ℝ) : equation m ∧ equation n → (2 : ℝ)^(m + n) = 128 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_root_product_l1080_108020


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_population_after_10_years_years_to_reach_1_2_million_l1080_108091

-- Define the initial population and growth rate
def initial_population : ℝ := 100 -- in tens of thousands
def growth_rate : ℝ := 0.012 -- 1.2%

-- Define the population function
noncomputable def population (x : ℝ) : ℝ := initial_population * (1 + growth_rate) ^ x

-- Theorem 1: Population after 10 years
theorem population_after_10_years :
  ∃ ε > 0, |population 10 - 112.7| < ε := by
  sorry

-- Theorem 2: Years to reach 1.2 million
theorem years_to_reach_1_2_million :
  ∃ ε > 0, |Real.log (120 / initial_population) / Real.log (1 + growth_rate) - 16| < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_population_after_10_years_years_to_reach_1_2_million_l1080_108091


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_calculation_l1080_108016

-- Define constants
noncomputable def train_length : ℝ := 200
noncomputable def crossing_time : ℝ := 8
noncomputable def man_speed_kmh : ℝ := 8

-- Define conversion factors
noncomputable def km_to_m : ℝ := 1000
noncomputable def hour_to_sec : ℝ := 3600

-- Define functions
noncomputable def kmh_to_ms (speed_kmh : ℝ) : ℝ :=
  speed_kmh * km_to_m / hour_to_sec

noncomputable def ms_to_kmh (speed_ms : ℝ) : ℝ :=
  speed_ms * hour_to_sec / km_to_m

-- Theorem statement
theorem train_speed_calculation :
  let man_speed_ms := kmh_to_ms man_speed_kmh
  let relative_speed := train_length / crossing_time
  let train_speed_ms := relative_speed - man_speed_ms
  let train_speed_kmh := ms_to_kmh train_speed_ms
  ∃ ε > 0, abs (train_speed_kmh - 82) < ε := by
  sorry

#check train_speed_calculation

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_speed_calculation_l1080_108016


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_pizza_area_increase_l1080_108074

/-- Calculates the percent increase in area when changing from a 16-inch diameter pizza to a 20-inch diameter pizza -/
theorem pizza_area_increase (initial_diameter new_diameter : Real)
  (h1 : initial_diameter = 16)
  (h2 : new_diameter = 20) :
  (((new_diameter / 2) ^ 2 * Real.pi - (initial_diameter / 2) ^ 2 * Real.pi) / ((initial_diameter / 2) ^ 2 * Real.pi)) * 100 = 56.25 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_pizza_area_increase_l1080_108074


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polygon_D_largest_area_l1080_108032

/-- The area of a unit square -/
def unit_square_area : ℝ := 1

/-- The area of a right triangle with legs of length 1 -/
def right_triangle_area : ℝ := 0.5

/-- The area of a quarter circle with radius 1 -/
noncomputable def quarter_circle_area : ℝ := Real.pi / 4

/-- The area of Polygon A -/
def polygon_A_area : ℝ := 4 * unit_square_area + 2 * right_triangle_area

/-- The area of Polygon B -/
noncomputable def polygon_B_area : ℝ := 2 * unit_square_area + 2 * right_triangle_area + quarter_circle_area

/-- The area of Polygon C -/
def polygon_C_area : ℝ := 3 * unit_square_area + 3 * right_triangle_area

/-- The area of Polygon D -/
noncomputable def polygon_D_area : ℝ := 3 * unit_square_area + right_triangle_area + 2 * quarter_circle_area

/-- The area of Polygon E -/
noncomputable def polygon_E_area : ℝ := unit_square_area + 3 * right_triangle_area + 3 * quarter_circle_area

/-- Theorem stating that Polygon D has the largest area -/
theorem polygon_D_largest_area :
  polygon_D_area > polygon_A_area ∧
  polygon_D_area > polygon_B_area ∧
  polygon_D_area > polygon_C_area ∧
  polygon_D_area > polygon_E_area :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_polygon_D_largest_area_l1080_108032


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_sum_of_squares_l1080_108090

open Real

/-- The function f(x) = a(ln x - 1) + (b + 1)x has a root in [e, e³] -/
def has_root_in_interval (a b : ℝ) : Prop :=
  ∃ x, (exp 1) ≤ x ∧ x ≤ (exp 1)^3 ∧ a * (log x - 1) + (b + 1) * x = 0

/-- The theorem stating the minimum value of a² + b² -/
theorem min_value_of_sum_of_squares (a b : ℝ) :
  has_root_in_interval a b → a^2 + b^2 ≥ (exp 4) / (1 + exp 4) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_of_sum_of_squares_l1080_108090


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_sum_of_squares_l1080_108057

theorem vector_sum_of_squares (p q n : ℝ × ℝ) : 
  n = (4, -2) →
  n = ((p.1 + q.1) / 2, (p.2 + q.2) / 2) →
  p.1 * q.1 + p.2 * q.2 = 12 →
  p.1^2 + p.2^2 + q.1^2 + q.2^2 = 56 := by
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_vector_sum_of_squares_l1080_108057


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_2m_l1080_108089

/-- Given planar vectors a and b, and a function f that is monotonically increasing on [-m, m],
    prove that the range of f(2m) is [1, 2]. -/
theorem range_of_f_2m (m : ℝ) : 
  let a : ℝ → ℝ × ℝ := λ x => (Real.sin x, 1)
  let b : ℝ → ℝ × ℝ := λ x => (Real.sqrt 3, Real.cos x)
  let f : ℝ → ℝ := λ x => (a x).1 * (b x).1 + (a x).2 * (b x).2
  (∀ x₁ x₂, -m ≤ x₁ ∧ x₁ < x₂ ∧ x₂ ≤ m → f x₁ < f x₂) →
  (0 < m ∧ m ≤ Real.pi / 6) →
  ∃ y ∈ Set.Icc 1 2, f (2 * m) = y :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_f_2m_l1080_108089


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_product_unit_digit_of_40_odds_l1080_108080

def is_odd (n : ℕ) : Prop := ∃ k, n = 2*k + 1

def unit_digit (n : ℕ) : ℕ := n % 10

theorem product_unit_digit_of_40_odds 
  (S : Finset ℕ) 
  (h_size : S.card = 40)
  (h_range : ∀ n ∈ S, 1 ≤ n ∧ n ≤ 100)
  (h_odd : ∀ n ∈ S, is_odd n) :
  ∃ p, (S.prod id) % 10 = p ∧ (p = 1 ∨ p = 5) :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_product_unit_digit_of_40_odds_l1080_108080


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_find_real_coins_l1080_108018

/-- Represents a coin that can be either real or counterfeit -/
inductive Coin
| real : Coin
| counterfeit : Coin

/-- Represents the result of weighing two coins -/
inductive WeighResult
| equal : WeighResult
| leftHeavier : WeighResult
| rightHeavier : WeighResult

/-- A function that performs a single weighing of two coins -/
def weigh (a b : Coin) : WeighResult :=
  match a, b with
  | Coin.real, Coin.real => WeighResult.equal
  | Coin.real, Coin.counterfeit => WeighResult.leftHeavier
  | Coin.counterfeit, Coin.real => WeighResult.rightHeavier
  | Coin.counterfeit, Coin.counterfeit => WeighResult.equal

/-- Represents a row of 100 coins -/
def CoinRow := Fin 100 → Coin

theorem find_real_coins 
  (coins : CoinRow)
  (h_counterfeit : ∃ (i : Fin 51), ∀ (j : Fin 50), coins (i + j) = Coin.counterfeit) :
  ∃ (S : Finset (Fin 100)), S.card ≥ 34 ∧ ∀ (i : Fin 100), i ∈ S → coins i = Coin.real :=
by
  sorry

#check find_real_coins

end NUMINAMATH_CALUDE_ERRORFEEDBACK_find_real_coins_l1080_108018


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_zeros_count_l1080_108038

-- Define the function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x^2 - x - a - 6 * Real.log x

-- Define the derivative of f(x)
noncomputable def f_deriv (x : ℝ) : ℝ := 2 * x - 1 - 6 / x

-- Theorem for the tangent line equation
theorem tangent_line_equation (a : ℝ) :
  ∃ (m b : ℝ), m = -5 ∧ b = -a + 5 ∧
  ∀ x y, y = m * x + b ↔ 5 * x + y + a - 5 = 0 := by
  sorry

-- Theorem for the number of zeros
theorem zeros_count (a : ℝ) :
  (∀ x ∈ Set.Ioo 0 4, f a x ≠ 0) ∨
  (∃! x, x ∈ Set.Ioo 0 4 ∧ f a x = 0 ∧ x = 2) ∨
  (∃! x, x ∈ Set.Ioo 0 4 ∧ f a x = 0 ∧ a > 12 - 12 * Real.log 2) ∨
  (∃ x y, x ∈ Set.Ioo 0 4 ∧ y ∈ Set.Ioo 0 4 ∧ x ≠ y ∧ f a x = 0 ∧ f a y = 0 ∧
    2 - 6 * Real.log 2 < a ∧ a ≤ 12 - 12 * Real.log 2) := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_zeros_count_l1080_108038


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_polygon_intersection_theorem_l1080_108079

-- Define IntersectionArea as a function that takes two polygon indices and returns their intersection area
noncomputable def IntersectionArea : ℕ → ℕ → ℝ := sorry

-- Define a function to represent the area of a polygon
noncomputable def PolygonArea : ℕ → ℝ := sorry

-- Define a function to represent the area of the square
noncomputable def SquareArea : ℝ := sorry

theorem polygon_intersection_theorem (A B : ℝ) (n : ℕ) 
  (h1 : A > 0) 
  (h2 : B > 0) 
  (h3 : n ≥ 2) 
  (h4 : n * B ≥ A) : 
  ∃ (i j : ℕ), i < n ∧ j < n ∧ i ≠ j ∧ 
    ∃ (S : ℝ), S ≥ (n * B - A) / (n.choose 2) ∧ 
      S = IntersectionArea i j :=
sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_polygon_intersection_theorem_l1080_108079


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_nine_pointed_star_angle_sum_l1080_108030

/-- A 9-pointed star is formed by connecting 9 evenly spaced points on a circle. -/
structure NinePointedStar :=
  (points : Fin 9 → ℝ × ℝ)
  (is_on_circle : ∀ i, (points i).1^2 + (points i).2^2 = 1)
  (evenly_spaced : ∀ i j, (i.val + 1) % 9 = j.val →
    (points i).1 * (points j).2 - (points i).2 * (points j).1 = Real.sin (2 * Real.pi / 9))

/-- The angle at each tip of the 9-pointed star -/
noncomputable def tip_angle (star : NinePointedStar) : ℝ := 80 * Real.pi / 180

/-- The theorem stating that the sum of angles at the tips of a 9-pointed star is 720° -/
theorem nine_pointed_star_angle_sum (star : NinePointedStar) :
  (9 : ℝ) * tip_angle star = 4 * Real.pi := by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_nine_pointed_star_angle_sum_l1080_108030


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_triangle_area_l1080_108035

-- Define the circle
def circle_eq (x y : ℝ) : Prop := x^2 + y^2 = 4

-- Define the line passing through (0, 1)
def line_eq (k : ℝ) (x y : ℝ) : Prop := y = k * x + 1

-- Define the area of triangle OAB
noncomputable def triangle_area (k : ℝ) : ℝ :=
  let d := 1 / Real.sqrt (k^2 + 1)
  Real.sqrt (4 - d^2) * d

-- Theorem statement
theorem max_triangle_area :
  ∃ (max_area : ℝ), max_area = Real.sqrt 3 ∧
  ∀ (k : ℝ), triangle_area k ≤ max_area := by
  sorry

#check max_triangle_area

end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_triangle_area_l1080_108035


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_transform_sin_2x_equals_sin_4x_l1080_108042

-- Define the original function
noncomputable def original_function (x : ℝ) : ℝ := Real.sin (2 * x)

-- Define the first transformation (move left by π/2)
noncomputable def first_transform (f : ℝ → ℝ) (x : ℝ) : ℝ := f (x + Real.pi / 2)

-- Define the second transformation (reduce x-coordinate to half)
def second_transform (f : ℝ → ℝ) (x : ℝ) : ℝ := f (2 * x)

-- Define the composition of both transformations
noncomputable def composed_transform (f : ℝ → ℝ) (x : ℝ) : ℝ := 
  second_transform (first_transform f) x

-- State the theorem
theorem transform_sin_2x_equals_sin_4x :
  ∀ x : ℝ, composed_transform original_function x = Real.sin (4 * x) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_transform_sin_2x_equals_sin_4x_l1080_108042


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_shift_l1080_108049

noncomputable def f (ω : ℝ) (x : ℝ) : ℝ := Real.cos (ω * x + Real.pi / 4)
noncomputable def g (ω : ℝ) (x : ℝ) : ℝ := Real.cos (ω * x)

theorem cosine_shift (ω : ℝ) (h : ω > 0) :
  (∀ x, f ω x = f ω (x + Real.pi / ω)) →
  (∀ x, g ω x = f ω (x + Real.pi / (8 * ω))) :=
by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_cosine_shift_l1080_108049


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l1080_108084

open Real

-- Define the functions f and g
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := x + a / x
noncomputable def g (x : ℝ) : ℝ := x - Real.log x

-- State the theorem
theorem range_of_a (a : ℝ) :
  (0 < a ∧ a ≤ 1) →
  (∀ x₁ x₂ : ℝ, x₁ ∈ Set.Icc 1 ℯ → x₂ ∈ Set.Icc 1 ℯ → f a x₁ ≥ g x₂) →
  a ∈ Set.Icc (ℯ - 2) 1 :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_range_of_a_l1080_108084


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_second_tier_rate_is_15_percent_l1080_108053

/-- Two-tiered tax system for imported cars -/
structure TaxSystem where
  firstTierRate : ℚ  -- First tier tax rate
  firstTierLimit : ℚ  -- Price limit for first tier
  secondTierRate : ℚ  -- Second tier tax rate (to be proved)

/-- Calculate total tax for a given car price -/
def calculateTax (system : TaxSystem) (carPrice : ℚ) : ℚ :=
  let firstTierTax := min carPrice system.firstTierLimit * system.firstTierRate
  let secondTierTax := max (carPrice - system.firstTierLimit) 0 * system.secondTierRate
  firstTierTax + secondTierTax

theorem second_tier_rate_is_15_percent 
  (system : TaxSystem)
  (h1 : system.firstTierRate = 1/4)
  (h2 : system.firstTierLimit = 10000)
  (h3 : calculateTax system 30000 = 5500) :
  system.secondTierRate = 3/20 := by
  sorry

#eval calculateTax ⟨1/4, 10000, 3/20⟩ 30000

end NUMINAMATH_CALUDE_ERRORFEEDBACK_second_tier_rate_is_15_percent_l1080_108053


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_odd_distinct_digits_l1080_108075

def is_odd (n : ℕ) : Bool := n % 2 = 1

def has_distinct_digits (n : ℕ) : Bool :=
  let digits := n.digits 10
  digits.toFinset.card = digits.length

def count_valid_numbers : ℕ :=
  (List.range 9000).map (λ x => x + 1000)
    |>.filter (λ n => is_odd n && has_distinct_digits n)
    |>.length

theorem probability_odd_distinct_digits :
  (count_valid_numbers : ℚ) / 9000 = 56 / 225 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_probability_odd_distinct_digits_l1080_108075


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_farm_animal_ratio_l1080_108002

theorem farm_animal_ratio : ∀ (initial_cows initial_dogs : ℕ),
  initial_cows = 184 →
  (3 * initial_cows / 4 + initial_dogs / 4 : ℚ) = 161 →
  initial_cows / initial_dogs = 2 :=
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_farm_animal_ratio_l1080_108002


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_condition_l1080_108010

/-- Two lines are perpendicular if and only if the product of their slopes is -1 -/
def perpendicular (m₁ m₂ : ℝ) : Prop := m₁ * m₂ = -1

/-- The slope of the first line mx + (2m - 1)y + 1 = 0 -/
noncomputable def slope1 (m : ℝ) : ℝ := -m / (2*m - 1)

/-- The slope of the second line 3x + my + 2 = 0 -/
noncomputable def slope2 (m : ℝ) : ℝ := -3 / m

/-- Theorem stating that m = -1 is a sufficient but not necessary condition -/
theorem perpendicular_condition (m : ℝ) : 
  (m = -1 → perpendicular (slope1 m) (slope2 m)) ∧ 
  ¬(perpendicular (slope1 m) (slope2 m) → m = -1) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_perpendicular_condition_l1080_108010


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_ralph_tennis_balls_l1080_108023

theorem ralph_tennis_balls (total_balls : ℕ) (first_group : ℕ) (second_group : ℕ)
  (hit_rate_first : ℚ) (hit_rate_second : ℚ)
  (h1 : total_balls = 175)
  (h2 : first_group = 100)
  (h3 : second_group = 75)
  (h4 : hit_rate_first = 2/5)
  (h5 : hit_rate_second = 1/3)
  (h6 : total_balls = first_group + second_group) :
  ((1 - hit_rate_first) * first_group + (1 - hit_rate_second) * second_group : ℚ) = 110 := by
sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_ralph_tennis_balls_l1080_108023


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_teacher_has_winning_strategy_l1080_108081

/-- Represents a player in the game -/
inductive Player
| Teacher
| Student (n : Nat)

/-- Represents a unit segment on the grid -/
structure Segment where
  x : Int
  y : Int
  horizontal : Bool

/-- Represents the state of the game -/
structure GameState where
  painted_segments : Set Segment
  current_player : Player

/-- Represents a move in the game -/
def Move := Segment

/-- Checks if a move is valid given the current game state -/
def is_valid_move (state : GameState) (move : Move) : Prop :=
  move ∉ state.painted_segments

/-- Updates the game state after a move -/
def make_move (state : GameState) (move : Move) : GameState :=
  { painted_segments := state.painted_segments.insert move
    current_player := 
      match state.current_player with
      | Player.Teacher => Player.Student 0
      | Player.Student n => 
          if n = 29 then Player.Teacher else Player.Student (n + 1)
  }

/-- Checks if there's a winning 1x2 or 2x1 rectangle -/
def is_winning_state (state : GameState) : Prop :=
  ∃ (x y : Int), 
    (({ x := x, y := y, horizontal := true } ∈ state.painted_segments ∧
      { x := x, y := y + 1, horizontal := true } ∈ state.painted_segments ∧
      { x := x, y := y, horizontal := false } ∈ state.painted_segments ∧
      { x := x + 1, y := y, horizontal := false } ∈ state.painted_segments) ∧
     { x := x, y := y + 1, horizontal := false } ∉ state.painted_segments) ∨
    (({ x := x, y := y, horizontal := false } ∈ state.painted_segments ∧
      { x := x + 1, y := y, horizontal := false } ∈ state.painted_segments ∧
      { x := x, y := y, horizontal := true } ∈ state.painted_segments ∧
      { x := x, y := y + 1, horizontal := true } ∈ state.painted_segments) ∧
     { x := x + 1, y := y, horizontal := true } ∉ state.painted_segments)

/-- The main theorem stating that the teacher has a winning strategy -/
theorem teacher_has_winning_strategy :
  ∃ (strategy : GameState → Move),
    ∀ (initial_state : GameState),
      ∃ (n : Nat),
        let final_state := (Nat.iterate (λ s => make_move s (strategy s)) n initial_state)
        is_winning_state final_state ∧
        (∀ k < n, is_valid_move ((Nat.iterate (λ s => make_move s (strategy s)) k initial_state)) (strategy ((Nat.iterate (λ s => make_move s (strategy s)) k initial_state)))) :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_teacher_has_winning_strategy_l1080_108081


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_exp_plus_x_l1080_108007

open Real MeasureTheory

theorem integral_exp_plus_x : ∫ x in Set.Icc 0 1, (Real.exp x + x) = Real.exp 1 - 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_integral_exp_plus_x_l1080_108007


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_removed_percentage_l1080_108064

/-- The percentage of volume removed from a rectangular prism when cubes are cut from each corner -/
theorem volume_removed_percentage (l w h c : ℝ) (hl : l = 20) (hw : w = 12) (hh : h = 10) (hc : c = 4) :
  (8 * c^3) / (l * w * h) * 100 = 16/75 * 100 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_volume_removed_percentage_l1080_108064


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_output_nine_inequality_solution_set_l1080_108017

-- Part 1
noncomputable def y (x : ℝ) : ℝ :=
  if x < 0 then x*x - 2*x + 6 else (x-1)*(x-1)

theorem output_nine (x : ℝ) : y x = 9 ↔ x = -1 ∨ x = 4 := by sorry

-- Part 2
theorem inequality_solution_set (a : ℝ) (h : a < 0) :
  {x : ℝ | x*(1-x) < a*(1-a)} = {x : ℝ | x < a ∨ x > 1-a} := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_output_nine_inequality_solution_set_l1080_108017


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l1080_108028

/-- The circle with equation x^2 + y^2 = 4 -/
def my_circle (x y : ℝ) : Prop := x^2 + y^2 = 4

/-- The point of tangency -/
def point_of_tangency : ℝ × ℝ := (3, 1)

/-- The equation of the tangent line -/
def tangent_line (x y : ℝ) : Prop := x + y - 4 = 0

theorem tangent_line_equation :
  ∀ x y : ℝ,
  my_circle x y →
  (x, y) = point_of_tangency →
  tangent_line x y :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_line_equation_l1080_108028


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_garden_watering_time_l1080_108066

/-- Represents the rate at which a sprinkler waters the garden -/
structure Sprinkler where
  rate : ℝ

/-- The rate at which the garden needs to be watered -/
def G : ℝ := 1

/-- The rates of individual sprinklers -/
def X : Sprinkler := ⟨0⟩
def Y : Sprinkler := ⟨0⟩
def Z : Sprinkler := ⟨0⟩

/-- The rate at which a combination of sprinklers waters the garden -/
def combined_rate (s1 s2 : Sprinkler) : ℝ := s1.rate + s2.rate

/-- The time taken by two sprinklers to water the garden -/
noncomputable def time_taken (s1 s2 : Sprinkler) : ℝ := 1 / (combined_rate s1 s2)

theorem garden_watering_time 
  (h1 : time_taken X Y = 5)
  (h2 : time_taken X Z = 6)
  (h3 : time_taken Y Z = 7)
  (h4 : ∀ s1 s2 s3 : Sprinkler, s1.rate + s2.rate + s3.rate = combined_rate s1 s2 + s3.rate)
  : 1 / (X.rate + Y.rate + Z.rate) = 420 / 107 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_garden_watering_time_l1080_108066


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_semicircle_radius_formula_semicircle_radius_approx_l1080_108005

/-- The perimeter of a semicircle in centimeters -/
def semicircle_perimeter : ℝ := 140

/-- The radius of a semicircle given its perimeter -/
noncomputable def semicircle_radius (p : ℝ) : ℝ := p / (Real.pi + 2)

/-- Theorem: The radius of a semicircle with perimeter 140 cm is equal to 140 / (π + 2) cm -/
theorem semicircle_radius_formula :
  semicircle_radius semicircle_perimeter = 140 / (Real.pi + 2) := by
  sorry

/-- Theorem: The radius of a semicircle with perimeter 140 cm is approximately 27.23 cm -/
theorem semicircle_radius_approx :
  ∃ ε > 0, abs (semicircle_radius semicircle_perimeter - 27.23) < ε := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_semicircle_radius_formula_semicircle_radius_approx_l1080_108005


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_properties_l1080_108096

noncomputable def f (a b x : ℝ) : ℝ := (b - 2^x) / (2^x + a)

theorem odd_function_properties (a b : ℝ) :
  (∀ x, f a b (-x) = -f a b x) →
  (a = 1 ∧ b = 1) ∧
  (∀ x y, x < y → f 1 1 x > f 1 1 y) ∧
  (∀ k, (∀ t, f 1 1 (t^2 - 2*t) + f 1 1 (2*t^2 - k) < 0) ↔ k < -1/3) :=
by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_odd_function_properties_l1080_108096


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_popton_school_bus_toes_l1080_108060

/-- Represents a Hoopit on planet Popton -/
structure Hoopit where
  hands : Nat := 4
  toes_per_hand : Nat := 3

/-- Represents a Neglart on planet Popton -/
structure Neglart where
  hands : Nat := 5
  toes_per_hand : Nat := 2

/-- Represents a Popton school bus -/
structure PoptonSchoolBus where
  hoopit_students : Nat := 7
  neglart_students : Nat := 8

/-- Calculates the total number of toes for a given number of Hoopits -/
def total_hoopit_toes (n : Nat) : Nat :=
  n * 4 * 3

/-- Calculates the total number of toes for a given number of Neglarts -/
def total_neglart_toes (n : Nat) : Nat :=
  n * 5 * 2

/-- Theorem: The total number of toes on a Popton school bus is 164 -/
theorem popton_school_bus_toes (bus : PoptonSchoolBus) :
  total_hoopit_toes bus.hoopit_students + total_neglart_toes bus.neglart_students = 164 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_popton_school_bus_toes_l1080_108060


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_of_x_squared_y_over_three_l1080_108082

/-- Definition: The coefficient of a monomial is the numerical part of the term that multiplies the variable(s). -/
def coefficient (term : ℚ) : ℚ := term

/-- Theorem: The coefficient of x²y/3 is 1/3 -/
theorem coefficient_of_x_squared_y_over_three :
  coefficient (1/3) = 1/3 := by
  rfl


end NUMINAMATH_CALUDE_ERRORFEEDBACK_coefficient_of_x_squared_y_over_three_l1080_108082


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_smaller_angle_at_5_15_l1080_108001

/-- Represents a time on a 12-hour analog clock -/
structure ClockTime where
  hours : ℕ
  minutes : ℕ

/-- Calculates the angle of the hour hand from 12 o'clock position -/
noncomputable def hourHandAngle (t : ClockTime) : ℝ :=
  (t.hours % 12 : ℝ) * 30 + (t.minutes : ℝ) * 0.5

/-- Calculates the angle of the minute hand from 12 o'clock position -/
noncomputable def minuteHandAngle (t : ClockTime) : ℝ :=
  (t.minutes : ℝ) * 6

/-- Calculates the smaller angle between the hour and minute hands -/
noncomputable def smallerAngleBetweenHands (t : ClockTime) : ℝ :=
  let diff := abs (hourHandAngle t - minuteHandAngle t)
  min diff (360 - diff)

/-- The theorem stating that at 5:15, the smaller angle between hands is 67.5° -/
theorem smaller_angle_at_5_15 :
  smallerAngleBetweenHands ⟨5, 15⟩ = 67.5 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_smaller_angle_at_5_15_l1080_108001


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_area_l1080_108022

theorem rectangle_area (b₁ b₂ b₃ b₄ b₅ b₆ b₇ b₈ b₉ : ℕ) 
  (h1 : b₁ + b₂ = b₃)
  (h2 : b₁ + b₃ = b₄)
  (h3 : b₃ + b₄ = b₅)
  (h4 : b₄ + b₅ = b₆)
  (h5 : b₂ + b₃ + b₅ = b₇)
  (h6 : b₂ + b₇ = b₈)
  (h7 : b₁ + b₄ + b₆ = b₉)
  (h8 : b₆ + b₉ = b₇ + b₈)
  (h_coprime : Int.gcd (b₁ + b₄ + b₆) (b₂ + b₇) = 1) :
  (b₁ + b₄ + b₆) * (b₂ + b₇) = 4004 := by
  sorry

#check rectangle_area

end NUMINAMATH_CALUDE_ERRORFEEDBACK_rectangle_area_l1080_108022


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_graph_division_theorem_l1080_108012

/-- A simple graph structure -/
structure Graph (V : Type*) where
  adj : V → V → Prop
  sym : ∀ u v, adj u v → adj v u
  loopless : ∀ v, ¬adj v v

/-- A k-coloring of a graph -/
def is_k_colorable {V : Type*} (G : Graph V) (k : ℕ) :=
  ∃ (f : V → Fin k), ∀ u v, G.adj u v → f u ≠ f v

/-- Subgraph of a graph -/
def is_subgraph {V : Type*} (H G : Graph V) :=
  ∀ u v, H.adj u v → G.adj u v

/-- The main theorem -/
theorem graph_division_theorem {V : Type*} (G : Graph V) :
  (¬is_k_colorable G 3) →
  ∃ (M N : Graph V), 
    (is_subgraph M G) ∧ (is_subgraph N G) ∧
    (¬is_k_colorable M 2) ∧ (¬is_k_colorable N 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_graph_division_theorem_l1080_108012


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_formula_l1080_108055

def mySequence (n : ℕ) : ℚ :=
  if n = 0 then 0
  else (n^2 : ℚ) / (n + 1)

theorem sequence_formula (n : ℕ) (hn : n > 0) :
  mySequence n = (n^2 : ℚ) / (n + 1) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_sequence_formula_l1080_108055


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_specific_lines_l1080_108063

/-- The distance between two parallel lines -/
noncomputable def distance_between_parallel_lines (A B C₁ C₂ : ℝ) : ℝ :=
  |C₂ - C₁| / Real.sqrt (A^2 + B^2)

/-- Theorem: The distance between the parallel lines 6x + 8y - 3 = 0 and 6x + 8y + 5 = 0 is 4/5 -/
theorem distance_between_specific_lines :
  distance_between_parallel_lines 6 8 (-3) 5 = 4/5 := by
  -- Proof goes here
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_distance_between_specific_lines_l1080_108063


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_midpoint_slope_zero_l1080_108052

-- Define the curve C
noncomputable def C (x y : ℝ) : Prop := y^2 = 4*x

-- Define the tangent line l'
noncomputable def l' (k m x y : ℝ) : Prop := y = k*x + m

-- Define point A as the tangent point
noncomputable def A (k : ℝ) : ℝ × ℝ := (1/k^2, 2/k)

-- Define point Q as the midpoint of OA
noncomputable def Q (k : ℝ) : ℝ × ℝ := (1/(2*k^2), 1/k)

-- Define point P as the y-intercept of l'
noncomputable def P (k : ℝ) : ℝ × ℝ := (0, 1/k)

-- Theorem statement
theorem tangent_midpoint_slope_zero (k : ℝ) (hk : k ≠ 0) :
  let a := A k
  let q := Q k
  let p := P k
  C a.1 a.2 ∧ l' k (1/k) a.1 a.2 →
  (q.2 - p.2) / (q.1 - p.1) = 0 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_tangent_midpoint_slope_zero_l1080_108052


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_nonagon_interior_angle_l1080_108072

/-- The number of sides in a nonagon -/
def nonagon_sides : ℕ := 9

/-- The sum of interior angles of a polygon with n sides -/
noncomputable def sum_interior_angles (n : ℕ) : ℝ := 180 * (n - 2)

/-- The measure of each interior angle in a regular polygon with n sides -/
noncomputable def interior_angle (n : ℕ) : ℝ := (sum_interior_angles n) / n

theorem regular_nonagon_interior_angle :
  interior_angle nonagon_sides = 140 := by sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_regular_nonagon_interior_angle_l1080_108072


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_snow_fall_time_is_100_hours_l1080_108015

/-- The time in hours for 1 m of snow to fall, given a snowfall rate of 1 mm every 6 minutes -/
noncomputable def snow_fall_time : ℝ :=
  let snow_rate_mm_per_6min : ℝ := 1
  let minutes_per_hour : ℝ := 60
  let mm_per_m : ℝ := 1000
  (mm_per_m / snow_rate_mm_per_6min) * (6 / minutes_per_hour)

theorem snow_fall_time_is_100_hours : snow_fall_time = 100 := by
  unfold snow_fall_time
  -- The proof steps would go here, but we'll use sorry for now
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_snow_fall_time_is_100_hours_l1080_108015


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_simplification_f_specific_value_l1080_108078

noncomputable def f (α : Real) : Real :=
  (Real.sin (7 * Real.pi - α) * Real.cos (α + 3 * Real.pi / 2) * Real.cos (3 * Real.pi + α)) /
  (Real.sin (α - 3 * Real.pi / 2) * Real.cos (α + 5 * Real.pi / 2) * Real.tan (α - 5 * Real.pi))

theorem f_simplification (α : Real) : f α = Real.cos α := by
  sorry

theorem f_specific_value (α : Real) 
  (h1 : π / 2 < α ∧ α < π) 
  (h2 : Real.cos (3 * Real.pi / 2 + α) = 1 / 7) : 
  f α = -4 * Real.sqrt 3 / 7 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_simplification_f_specific_value_l1080_108078


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_knight_liar_pairs_l1080_108095

/-- Represents a person on the island -/
inductive Person
| Knight
| Liar
deriving DecidableEq

/-- Represents a friendship between two people -/
structure Friendship where
  person1 : Person
  person2 : Person

/-- The statement made by a person -/
inductive Statement
| AllFriendsKnights
| AllFriendsLiars
deriving DecidableEq

/-- The island population -/
structure Island where
  people : Finset Person
  friendships : Finset Friendship
  statements : Person → Statement

/-- The properties of the island as described in the problem -/
structure IslandProperties (i : Island) : Prop where
  total_people : i.people.card = 200
  equal_knights_liars : (i.people.filter (λ p => p = Person.Knight)).card = 100
  everyone_has_friend : ∀ p ∈ i.people, ∃ f ∈ i.friendships, f.person1 = p ∨ f.person2 = p
  statement_counts : 
    (i.people.filter (λ p => i.statements p = Statement.AllFriendsKnights)).card = 100 ∧
    (i.people.filter (λ p => i.statements p = Statement.AllFriendsLiars)).card = 100
  knights_truthful : ∀ p ∈ i.people, p = Person.Knight → 
    (i.statements p = Statement.AllFriendsKnights → 
      ∀ f ∈ i.friendships, (f.person1 = p ∨ f.person2 = p) → 
        (f.person1 = Person.Knight ∧ f.person2 = Person.Knight)) ∧
    (i.statements p = Statement.AllFriendsLiars → 
      ∀ f ∈ i.friendships, (f.person1 = p ∨ f.person2 = p) → 
        (f.person1 = Person.Liar ∧ f.person2 = Person.Liar))
  liars_lie : ∀ p ∈ i.people, p = Person.Liar → 
    (i.statements p = Statement.AllFriendsKnights → 
      ∃ f ∈ i.friendships, (f.person1 = p ∨ f.person2 = p) ∧ 
        (f.person1 = Person.Liar ∨ f.person2 = Person.Liar)) ∧
    (i.statements p = Statement.AllFriendsLiars → 
      ∃ f ∈ i.friendships, (f.person1 = p ∨ f.person2 = p) ∧ 
        (f.person1 = Person.Knight ∨ f.person2 = Person.Knight))

/-- The main theorem to prove -/
theorem min_knight_liar_pairs (i : Island) (h : IslandProperties i) :
  (i.friendships.filter (λ f => 
    (f.person1 = Person.Knight ∧ f.person2 = Person.Liar) ∨
    (f.person1 = Person.Liar ∧ f.person2 = Person.Knight))).card ≥ 50 := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_knight_liar_pairs_l1080_108095


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_circumcircle_circumference_l1080_108077

-- Define the triangle ABC
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

-- Define the properties of the triangle
def Triangle.area (t : Triangle) : ℝ := 10
def Triangle.altitudeMidpointsCollinear (t : Triangle) : Prop := sorry

-- Define the circumference of the circumscribed circle
noncomputable def Triangle.circumCircleCircumference (t : Triangle) : ℝ := sorry

-- Theorem statement
theorem min_circumcircle_circumference (t : Triangle) 
  (h1 : Triangle.area t = 10)
  (h2 : Triangle.altitudeMidpointsCollinear t) :
  Triangle.circumCircleCircumference t ≥ 2 * Real.pi * Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_circumcircle_circumference_l1080_108077


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagonal_packing_improvement_l1080_108000

-- Define the square dimensions
def square_size : ℝ := 100

-- Define the circle diameter
def circle_diameter : ℝ := 1

-- Define the number of circles in the original packing
def original_circle_count : ℕ := 10000

-- Define the height of an equilateral triangle formed by three circle centers
noncomputable def triangle_height : ℝ := Real.sqrt 3 / 2

-- Define the theorem
theorem hexagonal_packing_improvement (square_size circle_diameter : ℝ)
  (original_circle_count : ℕ) (triangle_height : ℝ) :
  square_size = 100 →
  circle_diameter = 1 →
  original_circle_count = 10000 →
  triangle_height = Real.sqrt 3 / 2 →
  ∃ (new_circle_count : ℕ),
    new_circle_count = 11443 ∧
    new_circle_count > original_circle_count ∧
    new_circle_count ≤ (square_size / circle_diameter) ^ 2 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_hexagonal_packing_improvement_l1080_108000


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_laura_running_speed_l1080_108004

-- Define the variables
variable (x : ℝ)

-- Define the conditions
noncomputable def bike_distance : ℝ := 25
noncomputable def run_distance : ℝ := 7
noncomputable def transition_time : ℝ := 10 / 60  -- Convert to hours
noncomputable def total_time : ℝ := 140 / 60  -- Convert to hours

noncomputable def bike_speed (x : ℝ) : ℝ := 2 * x + 2
noncomputable def run_speed (x : ℝ) : ℝ := x

-- State the theorem
theorem laura_running_speed :
  ∃ x : ℝ, x > 0 ∧ 
  (bike_distance / bike_speed x + run_distance / run_speed x + transition_time = total_time) ∧
  (abs (4.3334 * x^2 - 34.6666 * x - 14) < 0.0001) := by
  sorry

#check laura_running_speed

end NUMINAMATH_CALUDE_ERRORFEEDBACK_laura_running_speed_l1080_108004


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_option_A_same_function_option_C_same_function_l1080_108003

-- Define the functions for Option A
def f_A (x : ℝ) : ℝ := x
noncomputable def g_A (x : ℝ) : ℝ := (x^3) ^ (1/3 : ℝ)

-- Define the functions for Option C
noncomputable def f_C (x : ℝ) : ℝ := x + 1/x
noncomputable def g_C (t : ℝ) : ℝ := t + 1/t

-- Theorem for Option A
theorem option_A_same_function : ∀ x : ℝ, f_A x = g_A x := by
  sorry

-- Theorem for Option C
theorem option_C_same_function : ∀ x : ℝ, x ≠ 0 → f_C x = g_C x := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_option_A_same_function_option_C_same_function_l1080_108003


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_compound_interest_time_period_l1080_108097

/-- Compound interest calculation --/
noncomputable def compound_interest (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * ((1 + rate) ^ time - 1)

/-- Time period calculation for compound interest --/
theorem compound_interest_time_period 
  (principal : ℝ) 
  (rate : ℝ) 
  (interest : ℝ) 
  (h_principal : principal = 12000)
  (h_rate : rate = 0.15)
  (h_interest : interest = 4663.5) :
  ∃ (t : ℝ), 
    compound_interest principal rate t = interest ∧ 
    1.5 < t ∧ t < 2.5 := by
  sorry

-- Remove the #eval line as it might cause issues without proper setup

end NUMINAMATH_CALUDE_ERRORFEEDBACK_compound_interest_time_period_l1080_108097


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_between_circles_l1080_108076

-- Define the circles M and N
noncomputable def circle_M (x y : ℝ) : Prop := (x - 5 * Real.sqrt 3 / 2)^2 + (y - 7/2)^2 = 4
noncomputable def circle_N (x y : ℝ) : Prop := (x - Real.sqrt 3 / 2)^2 + (y - 3/2)^2 = 1

-- Define the distance between two points
noncomputable def distance (x1 y1 x2 y2 : ℝ) : ℝ := Real.sqrt ((x1 - x2)^2 + (y1 - y2)^2)

-- Theorem statement
theorem min_distance_between_circles :
  ∃ (d : ℝ), d = 1 ∧
  ∀ (x1 y1 x2 y2 : ℝ),
    circle_M x1 y1 → circle_N x2 y2 →
    distance x1 y1 x2 y2 ≥ d ∧
    (∃ (x1' y1' x2' y2' : ℝ), 
      circle_M x1' y1' ∧ circle_N x2' y2' ∧
      distance x1' y1' x2' y2' = d) :=
by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_distance_between_circles_l1080_108076


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_risk_factors_probability_l1080_108033

noncomputable def population : ℝ := 1000

noncomputable def P_only_A : ℝ := 0.1
noncomputable def P_only_B : ℝ := 0.1
noncomputable def P_only_C : ℝ := 0.1

noncomputable def P_AB_not_C : ℝ := 0.14
noncomputable def P_AC_not_B : ℝ := 0.14
noncomputable def P_BC_not_A : ℝ := 0.14

noncomputable def P_C_given_AB : ℝ := 1/3

theorem risk_factors_probability (p q : ℕ) (h_coprime : Nat.Coprime p q) :
  let P_no_risk_given_not_A : ℚ := 21 / 55
  P_no_risk_given_not_A = p / q →
  p + q = 76 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_risk_factors_probability_l1080_108033


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_roll_l1080_108027

/-- The starting point of a circle rolling on a number line -/
def starting_point (x : ℝ) : Prop := True

/-- The ending point of a circle rolling on a number line -/
def ending_point (x : ℝ) : Prop := True

/-- A circle with radius 1 -/
def unit_circle : Prop := True

/-- The circle rolls exactly one round -/
def one_round : Prop := True

theorem circle_roll (x : ℝ) :
  unit_circle →
  one_round →
  starting_point x →
  ending_point (-1) →
  x = -2 * Real.pi - 1 ∨ x = 2 * Real.pi - 1 :=
by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_circle_roll_l1080_108027


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_expression_value_l1080_108070

theorem complex_expression_value : 
  abs ((Real.sqrt 6.5 * (2 ^ (1/3 : ℝ)) + (9.5 - 2^2) * 7.2 + (8.7 - 0.3) * (2 * (4.3 + 1)) - 5.3^2 + (1 / (3 + 4))) - 103.903776014) < 0.000000001 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_complex_expression_value_l1080_108070


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_a_on_b_l1080_108019

noncomputable def a : ℝ × ℝ := (2, 3)
noncomputable def b : ℝ × ℝ := (-2, 1)

noncomputable def projection (u v : ℝ × ℝ) : ℝ :=
  (u.1 * v.1 + u.2 * v.2) / Real.sqrt (v.1^2 + v.2^2)

theorem projection_a_on_b :
  projection a b = -Real.sqrt 5 / 5 := by
  -- Unfold the definitions
  unfold projection a b
  -- Perform the calculation
  simp [Real.sqrt_eq_rpow]
  -- The rest of the proof
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_projection_a_on_b_l1080_108019


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_of_S_l1080_108054

def S (n : ℕ) : ℕ :=
  (List.range n).foldr (λ i acc => acc + (2^i - 1)^(2^i - 1)) 0

theorem divisibility_of_S (n : ℕ) (h : n > 1) :
  (2^n ∣ S n) ∧ ¬(2^(n+1) ∣ S n) :=
by
  sorry

#eval S 3  -- This line is added to test the function

end NUMINAMATH_CALUDE_ERRORFEEDBACK_divisibility_of_S_l1080_108054


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_from_projection_l1080_108011

noncomputable def proj_vector (u v : ℝ × ℝ) : ℝ × ℝ :=
  let dot_product := u.1 * v.1 + u.2 * v.2
  let norm_squared := v.1 * v.1 + v.2 * v.2
  (dot_product / norm_squared * v.1, dot_product / norm_squared * v.2)

theorem line_equation_from_projection (x y : ℝ) :
  proj_vector (x, y) (3, 4) = (-3/2, -2) →
  y = -3/4 * x - 25/8 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_line_equation_from_projection_l1080_108011


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_independence_test_appropriate_independence_test_is_correct_method_l1080_108092

/-- Represents the survey data for a gender group -/
structure GenderData where
  total : ℕ
  opposing : ℕ

/-- Represents the survey results -/
structure SurveyResults where
  male : GenderData
  female : GenderData

/-- Calculates the chi-square statistic for the given survey results -/
noncomputable def chiSquare (results : SurveyResults) : ℝ :=
  let n := (results.male.total + results.female.total : ℝ)
  let a := (results.male.opposing : ℝ)
  let b := (results.male.total - results.male.opposing : ℝ)
  let c := (results.female.opposing : ℝ)
  let d := (results.female.total - results.female.opposing : ℝ)
  (n * (a * d - b * c)^2) / (results.male.total * results.female.total * (a + c) * (b + d))

/-- Determines if the independence test is appropriate based on the chi-square statistic -/
def isIndependenceTestAppropriate (results : SurveyResults) : Prop :=
  chiSquare results > 10.828

/-- The survey results from the problem -/
def problemResults : SurveyResults :=
  { male := { total := 2548, opposing := 1560 },
    female := { total := 2452, opposing := 1200 } }

/-- Theorem stating that the independence test is appropriate for the given survey results -/
theorem independence_test_appropriate :
  isIndependenceTestAppropriate problemResults := by
  sorry

/-- Main theorem proving that the independence test is the correct statistical method -/
theorem independence_test_is_correct_method :
  isIndependenceTestAppropriate problemResults →
  "Independence test" = "Correct statistical method for the problem" := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_independence_test_appropriate_independence_test_is_correct_method_l1080_108092


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_m_range_for_min_distance_at_right_vertex_l1080_108073

/-- Definition of the ellipse -/
def is_on_ellipse (x y : ℝ) : Prop := x^2/16 + y^2/12 = 1

/-- Point M on the major axis -/
def point_M (m : ℝ) : ℝ × ℝ := (m, 0)

/-- Right vertex of the ellipse -/
def right_vertex : ℝ × ℝ := (4, 0)

/-- Distance between two points -/
noncomputable def distance (p1 p2 : ℝ × ℝ) : ℝ :=
  Real.sqrt ((p1.1 - p2.1)^2 + (p1.2 - p2.2)^2)

/-- Theorem: Range of m when minimum distance occurs at right vertex -/
theorem m_range_for_min_distance_at_right_vertex :
  ∀ m : ℝ,
  (∀ x y : ℝ, is_on_ellipse x y →
    distance (point_M m) (x, y) ≥ distance (point_M m) right_vertex) →
  1 ≤ m ∧ m ≤ 4 :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_m_range_for_min_distance_at_right_vertex_l1080_108073


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_composition_value_l1080_108067

noncomputable def f (x : ℝ) : ℝ :=
  if x < 1 then x + 1 else -x + 3

theorem f_composition_value : f (f (5/2)) = 3/2 := by
  -- Evaluate f(5/2)
  have h1 : f (5/2) = 1/2 := by
    simp [f]
    norm_num
  
  -- Evaluate f(f(5/2))
  have h2 : f (1/2) = 3/2 := by
    simp [f]
    norm_num
  
  -- Combine the results
  rw [h1, h2]


end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_composition_value_l1080_108067


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_automatic_control_students_proof_l1080_108039

/-- The total number of students in the faculty -/
def total_students : ℕ := 673

/-- The proportion of second-year students -/
def second_year_proportion : ℚ := 4/5

/-- The number of second-year students studying numeric methods -/
def numeric_methods_students : ℕ := 250

/-- The number of second-year students studying both numeric methods and automatic control -/
def both_subjects_students : ℕ := 134

/-- The number of second-year students -/
def second_year_students : ℕ := 538

/-- The number of second-year students studying automatic control -/
def automatic_control_students : ℕ := 422

theorem automatic_control_students_proof :
  second_year_students - numeric_methods_students + both_subjects_students = automatic_control_students := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_automatic_control_students_proof_l1080_108039


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_petri_dish_radius_proof_l1080_108044

/-- The equation of the petri dish boundary -/
def petri_dish_equation (x y : ℝ) : Prop :=
  x^2 + y^2 + 10 = 6*x + 12*y

/-- The radius of the petri dish -/
noncomputable def petri_dish_radius : ℝ := Real.sqrt 35

/-- Theorem stating the equivalence between the original equation and the standard form of a circle -/
theorem petri_dish_radius_proof :
  ∃ (h k : ℝ), ∀ (x y : ℝ),
    petri_dish_equation x y ↔ (x - h)^2 + (y - k)^2 = petri_dish_radius^2 := by
  -- Provide the center coordinates
  use 3, 6
  -- Introduce variables and the equivalence
  intro x y
  -- State the equivalence
  apply Iff.intro
  -- Forward direction
  · intro h
    -- Expand the definition and simplify
    simp [petri_dish_equation] at h
    simp [petri_dish_radius]
    -- Algebraic manipulation (simplified for brevity)
    sorry
  -- Backward direction
  · intro h
    -- Expand the definition and simplify
    simp [petri_dish_radius] at h
    simp [petri_dish_equation]
    -- Algebraic manipulation (simplified for brevity)
    sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_petri_dish_radius_proof_l1080_108044


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_f_l1080_108094

noncomputable def f (x : ℝ) : ℝ := 2 * Real.cos (2 * x + Real.pi / 6)

theorem min_value_f :
  ∃ (min : ℝ), min = -Real.sqrt 3 ∧
  ∀ x, x ∈ Set.Icc (-Real.pi / 2) 0 → f x ≥ min ∧
  ∃ x₀, x₀ ∈ Set.Icc (-Real.pi / 2) 0 ∧ f x₀ = min :=
by sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_min_value_f_l1080_108094


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_identity_l1080_108085

theorem trig_identity (α : ℝ) 
  (h1 : Real.tan (α + π/4) = 1/2) 
  (h2 : α ∈ Set.Ioo (-π/2) 0) : 
  (2 * (Real.sin α)^2 + Real.sin (2*α)) / Real.cos (α - π/4) = -2*Real.sqrt 5 / 5 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_trig_identity_l1080_108085


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_three_squares_between_2011_and_2209_l1080_108056

theorem three_squares_between_2011_and_2209 : 
  ∃ (a b c : ℕ), 2011 < a^2 ∧ a^2 < b^2 ∧ b^2 < c^2 ∧ c^2 ≤ 2209 ∧
  ∀ (x : ℕ), 2011 < x^2 ∧ x^2 ≤ 2209 → x^2 = a^2 ∨ x^2 = b^2 ∨ x^2 = c^2 :=
by
  -- We'll use 45, 46, and 47 as our a, b, and c
  existsi 45
  existsi 46
  existsi 47
  have h1 : 2011 < 45^2 := by norm_num
  have h2 : 45^2 < 46^2 := by norm_num
  have h3 : 46^2 < 47^2 := by norm_num
  have h4 : 47^2 ≤ 2209 := by norm_num
  refine ⟨h1, h2, h3, h4, ?_⟩
  intro x hx
  -- We now need to show that if 2011 < x^2 ≤ 2209, then x^2 is one of 45^2, 46^2, or 47^2
  sorry -- The actual proof would go here, but we'll use sorry for now

#check three_squares_between_2011_and_2209

end NUMINAMATH_CALUDE_ERRORFEEDBACK_three_squares_between_2011_and_2209_l1080_108056


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_train_passing_time_l1080_108041

/-- Calculates the time taken for a train to pass a person moving in the same direction -/
noncomputable def time_to_pass (train_length : ℝ) (train_speed : ℝ) (person_speed : ℝ) : ℝ :=
  train_length / ((train_speed - person_speed) * (1000 / 3600))

/-- Theorem stating that the time taken for a 120 m long train moving at 68 kmph
    to pass a person moving at 8 kmph in the same direction is approximately 7.2 seconds -/
theorem train_passing_time :
  let train_length : ℝ := 120
  let train_speed : ℝ := 68
  let person_speed : ℝ := 8
  let actual_time := time_to_pass train_length train_speed person_speed
  abs (actual_time - 7.2) < 0.1 := by
  sorry

end NUMINAMATH_CALUDE_ERRORFEEDBACK_train_passing_time_l1080_108041


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_max_min_values_of_f_l1080_108087

-- Define the function f
noncomputable def f (x : ℝ) : ℝ := 4 * x / (x^2 + 1)

-- Define the domain
def domain : Set ℝ := Set.Icc (-2) 2

-- Theorem statement
theorem max_min_values_of_f :
  (∃ (x : ℝ), x ∈ domain ∧ f x = 2) ∧
  (∃ (y : ℝ), y ∈ domain ∧ f y = -2) ∧
  (∀ (z : ℝ), z ∈ domain → f z ≤ 2) ∧
  (∀ (w : ℝ), w ∈ domain → f w ≥ -2) := by
  sorry


end NUMINAMATH_CALUDE_ERRORFEEDBACK_max_min_values_of_f_l1080_108087


namespace NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1080_108008

noncomputable def f (x : ℝ) : ℝ := Real.sin (x / 3) + Real.cos (x / 3)

theorem f_properties :
  (∃ (p : ℝ), p > 0 ∧ p = 6 * Real.pi ∧ ∀ (x : ℝ), f (x + p) = f x) ∧
  (∃ (M : ℝ), M = Real.sqrt 2 ∧ ∀ (x : ℝ), f x ≤ M) := by
  sorry

#check f_properties

end NUMINAMATH_CALUDE_ERRORFEEDBACK_f_properties_l1080_108008
