import Mathlib

namespace NUMINAMATH_CALUDE_simplify_trig_expression_l2565_256598

theorem simplify_trig_expression :
  (Real.cos (5 * π / 180))^2 - (Real.sin (5 * π / 180))^2 =
  2 * Real.sin (40 * π / 180) * Real.cos (40 * π / 180) := by
  sorry

end NUMINAMATH_CALUDE_simplify_trig_expression_l2565_256598


namespace NUMINAMATH_CALUDE_inequality_proof_l2565_256556

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (h_sum : x + y + z ≥ 1) : 
  (x * Real.sqrt x) / (y + z) + (y * Real.sqrt y) / (z + x) + (z * Real.sqrt z) / (x + y) ≥ Real.sqrt 3 / 2 := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l2565_256556


namespace NUMINAMATH_CALUDE_collinear_vectors_l2565_256554

variable {V : Type*} [AddCommGroup V] [Module ℝ V]
variable (a b : V)

theorem collinear_vectors (h1 : ¬ Collinear ℝ ({0, a, b} : Set V))
    (h2 : Collinear ℝ ({0, 2 • a + k • b, a - b} : Set V)) :
  k = -2 := by
  sorry

end NUMINAMATH_CALUDE_collinear_vectors_l2565_256554


namespace NUMINAMATH_CALUDE_hexagram_arrangement_count_l2565_256543

/-- A hexagram is a regular six-pointed star with 12 points of intersection -/
structure Hexagram :=
  (points : Fin 12 → α)

/-- The number of symmetries of a hexagram (rotations and reflections) -/
def hexagram_symmetries : ℕ := 12

/-- The number of distinct arrangements of 12 unique objects on a hexagram,
    considering rotations and reflections as equivalent -/
def distinct_hexagram_arrangements : ℕ := Nat.factorial 12 / hexagram_symmetries

theorem hexagram_arrangement_count :
  distinct_hexagram_arrangements = 39916800 := by sorry

end NUMINAMATH_CALUDE_hexagram_arrangement_count_l2565_256543


namespace NUMINAMATH_CALUDE_john_daily_gallons_l2565_256581

-- Define the conversion rate from quarts to gallons
def quarts_per_gallon : ℚ := 4

-- Define the number of days in a week
def days_per_week : ℚ := 7

-- Define John's weekly water consumption in quarts
def john_weekly_quarts : ℚ := 42

-- Theorem to prove
theorem john_daily_gallons : 
  john_weekly_quarts / quarts_per_gallon / days_per_week = 1.5 := by
  sorry

end NUMINAMATH_CALUDE_john_daily_gallons_l2565_256581


namespace NUMINAMATH_CALUDE_unique_three_digit_number_l2565_256544

theorem unique_three_digit_number : ∃! n : ℕ, 
  100 ≤ n ∧ n < 1000 ∧  -- three-digit number
  n % 35 = 0 ∧          -- multiple of 35
  (n / 100 + (n / 10) % 10 + n % 10 = 15) ∧  -- sum of digits is 15
  n = 735 := by
sorry

end NUMINAMATH_CALUDE_unique_three_digit_number_l2565_256544


namespace NUMINAMATH_CALUDE_no_reassignment_possible_l2565_256519

/-- Represents a classroom with rows and columns of chairs -/
structure Classroom where
  rows : Nat
  columns : Nat

/-- Represents the total number of chairs in the classroom -/
def Classroom.totalChairs (c : Classroom) : Nat :=
  c.rows * c.columns

/-- Represents the number of occupied chairs -/
def Classroom.occupiedChairs (c : Classroom) (students : Nat) : Nat :=
  students

/-- Represents whether a reassignment is possible -/
def isReassignmentPossible (c : Classroom) (students : Nat) : Prop :=
  ∃ (redChairs blackChairs : Nat),
    redChairs + blackChairs = c.totalChairs - 1 ∧
    redChairs = students ∧
    blackChairs > redChairs

theorem no_reassignment_possible (c : Classroom) (students : Nat) :
  c.rows = 5 →
  c.columns = 7 →
  students = 34 →
  ¬ isReassignmentPossible c students :=
sorry

end NUMINAMATH_CALUDE_no_reassignment_possible_l2565_256519


namespace NUMINAMATH_CALUDE_max_segments_theorem_l2565_256578

/-- Represents an equilateral triangle divided into smaller equilateral triangles --/
structure DividedTriangle where
  n : ℕ  -- number of parts each side is divided into

/-- The maximum number of segments that can be marked without forming a complete smaller triangle --/
def max_marked_segments (t : DividedTriangle) : ℕ := t.n * (t.n + 1)

/-- Theorem stating the maximum number of segments that can be marked --/
theorem max_segments_theorem (t : DividedTriangle) :
  max_marked_segments t = t.n * (t.n + 1) :=
by sorry

end NUMINAMATH_CALUDE_max_segments_theorem_l2565_256578


namespace NUMINAMATH_CALUDE_euler_function_gcd_l2565_256504

open Nat

theorem euler_function_gcd (m n : ℕ) (h : φ (5^m - 1) = 5^n - 1) : (m.gcd n) > 1 := by
  sorry

end NUMINAMATH_CALUDE_euler_function_gcd_l2565_256504


namespace NUMINAMATH_CALUDE_arithmetic_expression_equality_l2565_256555

theorem arithmetic_expression_equality : (4 * 12) - (4 + 12) = 32 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_expression_equality_l2565_256555


namespace NUMINAMATH_CALUDE_area_of_triangle_OBA_l2565_256558

/-- Given two points A and B in polar coordinates, prove that the area of triangle OBA is 6 --/
theorem area_of_triangle_OBA (A B : ℝ × ℝ) (h_A : A = (3, π/3)) (h_B : B = (4, π/6)) : 
  let O : ℝ × ℝ := (0, 0)
  let area := (1/2) * (A.1 * B.1) * Real.sin (B.2 - A.2)
  area = 6 := by sorry

end NUMINAMATH_CALUDE_area_of_triangle_OBA_l2565_256558


namespace NUMINAMATH_CALUDE_breakfast_omelet_eggs_l2565_256525

/-- The number of eggs Gus ate in total -/
def total_eggs : ℕ := 6

/-- The number of eggs in Gus's lunch -/
def lunch_eggs : ℕ := 3

/-- The number of eggs in Gus's dinner -/
def dinner_eggs : ℕ := 1

/-- The number of eggs in Gus's breakfast omelet -/
def breakfast_eggs : ℕ := total_eggs - lunch_eggs - dinner_eggs

theorem breakfast_omelet_eggs :
  breakfast_eggs = 2 := by
  sorry

end NUMINAMATH_CALUDE_breakfast_omelet_eggs_l2565_256525


namespace NUMINAMATH_CALUDE_dairy_farmer_june_income_l2565_256513

/-- Calculates the total income for a dairy farmer in June -/
theorem dairy_farmer_june_income 
  (daily_production : ℕ) 
  (price_per_gallon : ℚ) 
  (days_in_june : ℕ) 
  (h1 : daily_production = 200)
  (h2 : price_per_gallon = 355/100)
  (h3 : days_in_june = 30) :
  daily_production * days_in_june * price_per_gallon = 21300 := by
sorry

end NUMINAMATH_CALUDE_dairy_farmer_june_income_l2565_256513


namespace NUMINAMATH_CALUDE_exactly_one_false_proposition_l2565_256535

theorem exactly_one_false_proposition :
  let prop1 := (∀ x : ℝ, (x ^ 2 - 3 * x + 2 ≠ 0) → (x ≠ 1)) ↔ (∀ x : ℝ, (x ≠ 1) → (x ^ 2 - 3 * x + 2 ≠ 0))
  let prop2 := (∀ x : ℝ, x > 2 → x ^ 2 - 3 * x + 2 > 0) ∧ (∃ x : ℝ, x ≤ 2 ∧ x ^ 2 - 3 * x + 2 > 0)
  let prop3 := ∀ p q : Prop, (p ∧ q → False) → (p → False) ∧ (q → False)
  let prop4 := (∃ x : ℝ, x ^ 2 + x + 1 < 0) ↔ ¬(∀ x : ℝ, x ^ 2 + x + 1 ≥ 0)
  ∃! i : Fin 4, ¬(match i with
    | 0 => prop1
    | 1 => prop2
    | 2 => prop3
    | 3 => prop4) :=
by
  sorry

end NUMINAMATH_CALUDE_exactly_one_false_proposition_l2565_256535


namespace NUMINAMATH_CALUDE_area_of_fourth_square_l2565_256567

/-- Given two right triangles PQR and PRS sharing a common hypotenuse PR,
    where the squares on PQ, QR, and RS have areas 25, 49, and 64 square units respectively,
    prove that the area of the square on PS is 10 square units. -/
theorem area_of_fourth_square (P Q R S : ℝ × ℝ) : 
  (P.1 - Q.1)^2 + (P.2 - Q.2)^2 = 25 →
  (Q.1 - R.1)^2 + (Q.2 - R.2)^2 = 49 →
  (R.1 - S.1)^2 + (R.2 - S.2)^2 = 64 →
  (P.1 - S.1)^2 + (P.2 - S.2)^2 = 10 := by
  sorry


end NUMINAMATH_CALUDE_area_of_fourth_square_l2565_256567


namespace NUMINAMATH_CALUDE_min_area_rectangle_l2565_256521

/-- A rectangle with even integer dimensions and perimeter 120 has a minimum area of 116 -/
theorem min_area_rectangle (l w : ℕ) : 
  Even l → Even w → 
  2 * (l + w) = 120 → 
  ∀ a : ℕ, (Even a.sqrt ∧ Even (60 - a.sqrt) ∧ a = a.sqrt * (60 - a.sqrt)) → 
  116 ≤ a := by
sorry

end NUMINAMATH_CALUDE_min_area_rectangle_l2565_256521


namespace NUMINAMATH_CALUDE_survey_probability_l2565_256585

theorem survey_probability : 
  let n : ℕ := 14  -- Total number of questions
  let k : ℕ := 10  -- Number of correct answers
  let m : ℕ := 4   -- Number of possible answers per question
  (n.choose k * (m - 1)^(n - k)) / m^n = 1001 * 3^4 / 4^14 := by
  sorry

end NUMINAMATH_CALUDE_survey_probability_l2565_256585


namespace NUMINAMATH_CALUDE_arithmetic_calculations_l2565_256576

theorem arithmetic_calculations :
  (-(1/8 : ℚ) + 3/4 - (-(1/4)) - 5/8 = 1/4) ∧
  (-3^2 + 5 * (-6) - (-4)^2 / (-8) = -37) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculations_l2565_256576


namespace NUMINAMATH_CALUDE_mens_wages_l2565_256503

theorem mens_wages (total_earnings : ℝ) (num_men : ℕ) (num_boys : ℕ)
  (h_total : total_earnings = 432)
  (h_men : num_men = 15)
  (h_boys : num_boys = 12)
  (h_equal_earnings : ∃ (num_women : ℕ), num_men * (total_earnings / (num_men + num_women + num_boys)) = 
                                         num_women * (total_earnings / (num_men + num_women + num_boys)) ∧
                                         num_women * (total_earnings / (num_men + num_women + num_boys)) = 
                                         num_boys * (total_earnings / (num_men + num_women + num_boys))) :
  num_men * (total_earnings / (num_men + num_men + num_men)) = 144 := by
  sorry

end NUMINAMATH_CALUDE_mens_wages_l2565_256503


namespace NUMINAMATH_CALUDE_five_mondays_in_march_l2565_256517

/-- Represents days of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Represents a leap year with five Sundays in February -/
structure LeapYearWithFiveSundaysInFebruary :=
  (isLeapYear : Bool)
  (februaryHasFiveSundays : Bool)

/-- Function to get the next day of the week -/
def nextDay (d : DayOfWeek) : DayOfWeek :=
  match d with
  | DayOfWeek.Sunday => DayOfWeek.Monday
  | DayOfWeek.Monday => DayOfWeek.Tuesday
  | DayOfWeek.Tuesday => DayOfWeek.Wednesday
  | DayOfWeek.Wednesday => DayOfWeek.Thursday
  | DayOfWeek.Thursday => DayOfWeek.Friday
  | DayOfWeek.Friday => DayOfWeek.Saturday
  | DayOfWeek.Saturday => DayOfWeek.Sunday

theorem five_mondays_in_march 
  (year : LeapYearWithFiveSundaysInFebruary) : 
  ∃ (mondayCount : Nat), mondayCount = 5 ∧ 
  (∀ (d : DayOfWeek), d ≠ DayOfWeek.Monday → 
    ∃ (otherCount : Nat), otherCount < 5) :=
by sorry

end NUMINAMATH_CALUDE_five_mondays_in_march_l2565_256517


namespace NUMINAMATH_CALUDE_forgotten_item_distance_l2565_256527

/-- Calculates the total distance walked when a person forgets an item halfway to school -/
def total_distance_walked (home_to_school : ℕ) : ℕ :=
  let halfway := home_to_school / 2
  halfway + halfway + home_to_school

/-- Proves that the total distance walked is 1500 meters given the conditions -/
theorem forgotten_item_distance :
  total_distance_walked 750 = 1500 := by
  sorry

#eval total_distance_walked 750

end NUMINAMATH_CALUDE_forgotten_item_distance_l2565_256527


namespace NUMINAMATH_CALUDE_roses_to_mother_l2565_256557

def roses_problem (total_roses grandmother_roses sister_roses kept_roses : ℕ) : ℕ :=
  total_roses - (grandmother_roses + sister_roses + kept_roses)

theorem roses_to_mother :
  roses_problem 20 9 4 1 = 6 := by
  sorry

end NUMINAMATH_CALUDE_roses_to_mother_l2565_256557


namespace NUMINAMATH_CALUDE_function_properties_l2565_256563

noncomputable section

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := (a * x^2 + 4) / x

-- State the theorem
theorem function_properties (a : ℝ) :
  f a 1 = 5 →
  (a = 1 ∧
   (∀ x : ℝ, x ≠ 0 → f a (-x) = -(f a x)) ∧
   (∀ x₁ x₂ : ℝ, 2 ≤ x₁ → x₁ < x₂ → f a x₁ < f a x₂)) :=
by sorry

end

end NUMINAMATH_CALUDE_function_properties_l2565_256563


namespace NUMINAMATH_CALUDE_students_in_other_communities_l2565_256564

theorem students_in_other_communities 
  (total_students : ℕ) 
  (muslim_percent hindu_percent sikh_percent : ℚ) :
  total_students = 1520 →
  muslim_percent = 41/100 →
  hindu_percent = 32/100 →
  sikh_percent = 12/100 →
  (total_students : ℚ) * (1 - (muslim_percent + hindu_percent + sikh_percent)) = 228 := by
  sorry

end NUMINAMATH_CALUDE_students_in_other_communities_l2565_256564


namespace NUMINAMATH_CALUDE_two_sqrt_six_lt_five_l2565_256522

theorem two_sqrt_six_lt_five : 2 * Real.sqrt 6 < 5 := by
  sorry

end NUMINAMATH_CALUDE_two_sqrt_six_lt_five_l2565_256522


namespace NUMINAMATH_CALUDE_wheat_mixture_problem_arun_wheat_problem_l2565_256575

/-- Calculates the rate of the second wheat purchase given the conditions of Arun's wheat mixture problem -/
theorem wheat_mixture_problem (first_quantity : ℝ) (first_rate : ℝ) (second_quantity : ℝ) (selling_rate : ℝ) (profit_percentage : ℝ) : ℝ :=
  let total_quantity := first_quantity + second_quantity
  let first_cost := first_quantity * first_rate
  let total_selling_price := total_quantity * selling_rate
  let total_cost := total_selling_price / (1 + profit_percentage / 100)
  (total_cost - first_cost) / second_quantity

/-- The rate of the second wheat purchase in Arun's problem is 14.25 -/
theorem arun_wheat_problem : 
  wheat_mixture_problem 30 11.50 20 15.75 25 = 14.25 := by
  sorry

end NUMINAMATH_CALUDE_wheat_mixture_problem_arun_wheat_problem_l2565_256575


namespace NUMINAMATH_CALUDE_readers_overlap_l2565_256502

theorem readers_overlap (total : ℕ) (science_fiction : ℕ) (literary : ℕ) 
  (h1 : total = 650) 
  (h2 : science_fiction = 250) 
  (h3 : literary = 550) : 
  total = science_fiction + literary - 150 := by
  sorry

#check readers_overlap

end NUMINAMATH_CALUDE_readers_overlap_l2565_256502


namespace NUMINAMATH_CALUDE_imaginary_part_of_z_l2565_256528

theorem imaginary_part_of_z (x y : ℝ) (h : (1 + Complex.I) * x + (1 - Complex.I) * y = 2) :
  let z : ℂ := (x + Complex.I) / (y - Complex.I)
  Complex.im z = 1 := by sorry

end NUMINAMATH_CALUDE_imaginary_part_of_z_l2565_256528


namespace NUMINAMATH_CALUDE_solution_set_theorem_g_zero_range_l2565_256559

-- Define the function f
def f (x a : ℝ) : ℝ := |x - 1| + |x + a|

-- Define the function g
def g (x a : ℝ) : ℝ := f x a - |3 + a|

-- Theorem for the solution set of |x-1| + |x+3| > 6
theorem solution_set_theorem :
  {x : ℝ | |x - 1| + |x + 3| > 6} = {x | x < -4} ∪ {x | -3 < x ∧ x < 1} ∪ {x | x > 2} :=
sorry

-- Theorem for the range of a when g has a zero
theorem g_zero_range (a : ℝ) :
  (∃ x, g x a = 0) ↔ a ≥ -2 :=
sorry

end NUMINAMATH_CALUDE_solution_set_theorem_g_zero_range_l2565_256559


namespace NUMINAMATH_CALUDE_three_cones_problem_l2565_256506

/-- A cone with vertex A -/
structure Cone (A : Point) where
  vertex_angle : ℝ

/-- A plane passing through a point -/
structure Plane (A : Point)

/-- Three cones touching each other externally -/
def touching_cones (A : Point) (c1 c2 c3 : Cone A) : Prop :=
  sorry

/-- Two cones are identical -/
def identical_cones (c1 c2 : Cone A) : Prop :=
  c1.vertex_angle = c2.vertex_angle

/-- A cone touches a plane -/
def cone_touches_plane (c : Cone A) (p : Plane A) : Prop :=
  sorry

/-- A cone lies on one side of a plane -/
def cone_on_one_side (c : Cone A) (p : Plane A) : Prop :=
  sorry

theorem three_cones_problem (A : Point) (c1 c2 c3 : Cone A) (p : Plane A) :
  touching_cones A c1 c2 c3 →
  identical_cones c1 c2 →
  c3.vertex_angle = π / 2 →
  cone_touches_plane c1 p →
  cone_touches_plane c2 p →
  cone_touches_plane c3 p →
  cone_on_one_side c1 p →
  cone_on_one_side c2 p →
  cone_on_one_side c3 p →
  c1.vertex_angle = 2 * Real.arctan (4 / 5) :=
sorry

end NUMINAMATH_CALUDE_three_cones_problem_l2565_256506


namespace NUMINAMATH_CALUDE_cricket_team_age_difference_l2565_256574

theorem cricket_team_age_difference (team_size : ℕ) (captain_age : ℕ) (team_avg_age : ℕ) :
  team_size = 11 →
  captain_age = 26 →
  team_avg_age = 23 →
  ∃ (wicket_keeper_age : ℕ),
    wicket_keeper_age > captain_age ∧
    (team_avg_age * team_size - captain_age - wicket_keeper_age) / (team_size - 2) + 1 = team_avg_age ∧
    wicket_keeper_age - captain_age = 3 :=
by sorry

end NUMINAMATH_CALUDE_cricket_team_age_difference_l2565_256574


namespace NUMINAMATH_CALUDE_last_digit_of_one_over_three_to_fifteen_l2565_256510

theorem last_digit_of_one_over_three_to_fifteen (n : ℕ) : 
  n = 15 → (1 : ℚ) / 3^n % 10 = 7 := by
  sorry

end NUMINAMATH_CALUDE_last_digit_of_one_over_three_to_fifteen_l2565_256510


namespace NUMINAMATH_CALUDE_thankYouCards_count_l2565_256592

/-- Represents the number of items to be mailed --/
structure MailItems where
  thankYouCards : ℕ
  bills : ℕ
  rebates : ℕ
  jobApplications : ℕ

/-- Calculates the total number of stamps required --/
def totalStamps (items : MailItems) : ℕ :=
  items.thankYouCards + items.bills + 1 + items.rebates + items.jobApplications

/-- Theorem stating the number of thank you cards --/
theorem thankYouCards_count (items : MailItems) : 
  items.bills = 2 ∧ 
  items.rebates = items.bills + 3 ∧ 
  items.jobApplications = 2 * items.rebates ∧
  totalStamps items = 21 →
  items.thankYouCards = 3 := by
  sorry

end NUMINAMATH_CALUDE_thankYouCards_count_l2565_256592


namespace NUMINAMATH_CALUDE_inscribed_circle_area_ratio_l2565_256533

theorem inscribed_circle_area_ratio (α : Real) (h : 0 < α ∧ α < π / 2) :
  let rhombus_area (a : Real) := a^2 * Real.sin α
  let circle_area (r : Real) := π * r^2
  let inscribed_circle_radius (a : Real) := (a * Real.sin α) / 2
  ∀ a > 0, circle_area (inscribed_circle_radius a) / rhombus_area a = (π / 4) * Real.sin α :=
sorry

end NUMINAMATH_CALUDE_inscribed_circle_area_ratio_l2565_256533


namespace NUMINAMATH_CALUDE_F_is_even_l2565_256566

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the property that f(-x) + f(x) = 0 for all x
axiom f_property : ∀ x, f (-x) + f x = 0

-- Define F(x) = |f(x)|
def F (x : ℝ) : ℝ := |f x|

-- Theorem statement
theorem F_is_even : ∀ x, F x = F (-x) := by sorry

end NUMINAMATH_CALUDE_F_is_even_l2565_256566


namespace NUMINAMATH_CALUDE_flower_position_l2565_256568

/-- Represents the number of students in the circle -/
def n : ℕ := 7

/-- Represents the number of times the drum is beaten -/
def k : ℕ := 50

/-- Function to calculate the final position after k rotations in a circle of n elements -/
def finalPosition (n k : ℕ) : ℕ := 
  (k % n) + 1

theorem flower_position : 
  finalPosition n k = 2 := by sorry

end NUMINAMATH_CALUDE_flower_position_l2565_256568


namespace NUMINAMATH_CALUDE_cyclic_inequality_l2565_256511

theorem cyclic_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a^3 / (a^2 + a*b + b^2)) + (b^3 / (b^2 + b*c + c^2)) + (c^3 / (c^2 + a*c + a^2)) ≥ (a + b + c) / 3 := by
  sorry

end NUMINAMATH_CALUDE_cyclic_inequality_l2565_256511


namespace NUMINAMATH_CALUDE_roxy_garden_plants_l2565_256520

/-- Calculates the total number of plants in Roxy's garden after buying and giving away plants -/
def total_plants_remaining (initial_flowering : ℕ) (bought_flowering : ℕ) (bought_fruiting : ℕ) 
  (given_away_flowering : ℕ) (given_away_fruiting : ℕ) : ℕ :=
  let initial_fruiting := 2 * initial_flowering
  let final_flowering := initial_flowering + bought_flowering - given_away_flowering
  let final_fruiting := initial_fruiting + bought_fruiting - given_away_fruiting
  final_flowering + final_fruiting

/-- Theorem stating that the total number of plants remaining in Roxy's garden is 21 -/
theorem roxy_garden_plants : 
  total_plants_remaining 7 3 2 1 4 = 21 := by
  sorry

end NUMINAMATH_CALUDE_roxy_garden_plants_l2565_256520


namespace NUMINAMATH_CALUDE_line_parallel_to_plane_theorem_l2565_256508

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relationships between lines and planes
variable (parallel_line_plane : Line → Plane → Prop)
variable (parallel_lines : Line → Line → Prop)
variable (inside_plane : Line → Plane → Prop)
variable (skew_lines : Line → Line → Prop)

-- State the theorem
theorem line_parallel_to_plane_theorem 
  (a b : Line) (α : Plane) 
  (h1 : parallel_line_plane a α) 
  (h2 : inside_plane b α) :
  parallel_lines a b ∨ skew_lines a b :=
sorry

end NUMINAMATH_CALUDE_line_parallel_to_plane_theorem_l2565_256508


namespace NUMINAMATH_CALUDE_terrell_lifting_equivalence_l2565_256562

/-- The number of times Terrell lifts the 40-pound weight -/
def original_lifts : ℕ := 12

/-- The weight of the original weight in pounds -/
def original_weight : ℕ := 40

/-- The weight of the new weight in pounds -/
def new_weight : ℕ := 30

/-- The total weight lifted with the original weight -/
def total_weight : ℕ := original_weight * original_lifts

/-- The number of times Terrell must lift the new weight to achieve the same total weight -/
def new_lifts : ℕ := total_weight / new_weight

theorem terrell_lifting_equivalence :
  new_lifts = 16 :=
sorry

end NUMINAMATH_CALUDE_terrell_lifting_equivalence_l2565_256562


namespace NUMINAMATH_CALUDE_heathers_remaining_blocks_l2565_256550

theorem heathers_remaining_blocks
  (initial_blocks : ℕ)
  (shared_with_jose : ℕ)
  (shared_with_emily : ℕ)
  (h1 : initial_blocks = 86)
  (h2 : shared_with_jose = 41)
  (h3 : shared_with_emily = 15) :
  initial_blocks - (shared_with_jose + shared_with_emily) = 30 :=
by sorry

end NUMINAMATH_CALUDE_heathers_remaining_blocks_l2565_256550


namespace NUMINAMATH_CALUDE_multiple_properties_l2565_256542

theorem multiple_properties (x y : ℤ) 
  (hx : ∃ m : ℤ, x = 6 * m) 
  (hy : ∃ n : ℤ, y = 9 * n) : 
  (∃ k : ℤ, x - y = 3 * k) ∧ 
  (∃ a b : ℤ, (∃ m : ℤ, a = 6 * m) ∧ (∃ n : ℤ, b = 9 * n) ∧ (∃ l : ℤ, a - b = 9 * l)) :=
by sorry

end NUMINAMATH_CALUDE_multiple_properties_l2565_256542


namespace NUMINAMATH_CALUDE_remainder_4536_div_32_l2565_256523

theorem remainder_4536_div_32 : 4536 % 32 = 24 := by
  sorry

end NUMINAMATH_CALUDE_remainder_4536_div_32_l2565_256523


namespace NUMINAMATH_CALUDE_distribution_count_l2565_256530

/-- Represents a distribution of tickets to people -/
structure TicketDistribution where
  /-- The number of tickets -/
  num_tickets : Nat
  /-- The number of people -/
  num_people : Nat
  /-- Condition that each person receives at least one ticket -/
  at_least_one_ticket : num_tickets ≥ num_people
  /-- Condition that the number of tickets is 5 -/
  five_tickets : num_tickets = 5
  /-- Condition that the number of people is 4 -/
  four_people : num_people = 4

/-- Counts the number of valid distributions -/
def count_distributions (d : TicketDistribution) : Nat :=
  -- The actual implementation is not provided here
  sorry

/-- Theorem stating that the number of valid distributions is 96 -/
theorem distribution_count (d : TicketDistribution) : count_distributions d = 96 := by
  sorry

end NUMINAMATH_CALUDE_distribution_count_l2565_256530


namespace NUMINAMATH_CALUDE_circle_equation_l2565_256594

/-- The equation of a circle with center (h, k) and radius r is (x - h)² + (y - k)² = r² -/
theorem circle_equation (x y : ℝ) : 
  let h : ℝ := 1
  let k : ℝ := 2
  let r : ℝ := 5
  (x - h)^2 + (y - k)^2 = r^2 := by sorry

end NUMINAMATH_CALUDE_circle_equation_l2565_256594


namespace NUMINAMATH_CALUDE_parabola_properties_l2565_256531

-- Define the parabola equation
def parabola_equation (x : ℝ) : ℝ := -3 * x^2 + 18 * x - 22

-- Define the vertex of the parabola
def vertex : ℝ × ℝ := (3, 5)

-- Define a point that the parabola passes through
def point : ℝ × ℝ := (2, 2)

-- Theorem statement
theorem parabola_properties :
  -- The parabola passes through the given point
  parabola_equation point.1 = point.2 ∧
  -- The vertex of the parabola is at (3, 5)
  (∀ x, parabola_equation x ≤ parabola_equation vertex.1) ∧
  -- The axis of symmetry is vertical (x = 3)
  (∀ x, parabola_equation (2 * vertex.1 - x) = parabola_equation x) :=
by sorry

end NUMINAMATH_CALUDE_parabola_properties_l2565_256531


namespace NUMINAMATH_CALUDE_box_height_minimum_l2565_256577

theorem box_height_minimum (x : ℝ) : 
  x > 0 →                           -- side length is positive
  2 * x^2 + 4 * x * (2 * x) ≥ 120 → -- surface area is at least 120
  2 * x ≥ 4 * Real.sqrt 3 :=        -- height (2x) is at least 4√3
by
  sorry

end NUMINAMATH_CALUDE_box_height_minimum_l2565_256577


namespace NUMINAMATH_CALUDE_no_real_a_with_unique_solution_l2565_256590

-- Define the function f(x) = x^2 + ax + 2a
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + a*x + 2*a

-- Define the property that |f(x)| ≤ 5 has exactly one solution
def has_unique_solution (a : ℝ) : Prop :=
  ∃! x : ℝ, |f a x| ≤ 5

-- Theorem statement
theorem no_real_a_with_unique_solution :
  ¬∃ a : ℝ, has_unique_solution a :=
sorry

end NUMINAMATH_CALUDE_no_real_a_with_unique_solution_l2565_256590


namespace NUMINAMATH_CALUDE_three_number_product_l2565_256512

theorem three_number_product (a b c : ℝ) 
  (sum_eq : a + b + c = 30)
  (first_eq : a = 2 * (b + c))
  (second_eq : b = 5 * c) :
  a * b * c = 2500 / 9 := by
sorry

end NUMINAMATH_CALUDE_three_number_product_l2565_256512


namespace NUMINAMATH_CALUDE_inequality_solution_l2565_256538

theorem inequality_solution (x : ℝ) : 
  (6*x^2 + 18*x - 64) / ((3*x - 2)*(x + 5)) < 2 ↔ -5 < x ∧ x < 2/3 := by sorry

end NUMINAMATH_CALUDE_inequality_solution_l2565_256538


namespace NUMINAMATH_CALUDE_income_calculation_l2565_256539

theorem income_calculation (income expenditure savings : ℕ) : 
  income = 7 * expenditure / 6 →
  savings = income - expenditure →
  savings = 2000 →
  income = 14000 := by
sorry

end NUMINAMATH_CALUDE_income_calculation_l2565_256539


namespace NUMINAMATH_CALUDE_customers_left_l2565_256547

theorem customers_left (initial : Nat) (remaining : Nat) : initial - remaining = 11 :=
  by
  -- Proof goes here
  sorry

end NUMINAMATH_CALUDE_customers_left_l2565_256547


namespace NUMINAMATH_CALUDE_substance_volume_weight_relation_l2565_256500

/-- Given a substance where volume is directly proportional to weight,
    prove that if 48 cubic inches weigh 112 ounces,
    then 63 ounces will have a volume of 27 cubic inches. -/
theorem substance_volume_weight_relation 
  (k : ℚ) -- Constant of proportionality
  (h1 : 48 = k * 112) -- 48 cubic inches weigh 112 ounces
  : k * 63 = 27 := by
  sorry

end NUMINAMATH_CALUDE_substance_volume_weight_relation_l2565_256500


namespace NUMINAMATH_CALUDE_power_inequality_l2565_256548

theorem power_inequality (n : ℕ) (h : n > 2) : n^(n+1) > (n+1)^n := by
  sorry

end NUMINAMATH_CALUDE_power_inequality_l2565_256548


namespace NUMINAMATH_CALUDE_compound_interest_rate_l2565_256509

/-- Given a principal amount, time period, and final amount, 
    calculate the annual interest rate for compound interest. -/
theorem compound_interest_rate 
  (principal : ℝ) 
  (time : ℝ) 
  (final_amount : ℝ) 
  (h1 : principal = 8000) 
  (h2 : time = 2) 
  (h3 : final_amount = 8820) : 
  ∃ (rate : ℝ), 
    final_amount = principal * (1 + rate) ^ time ∧ 
    rate = 0.05 := by
  sorry

end NUMINAMATH_CALUDE_compound_interest_rate_l2565_256509


namespace NUMINAMATH_CALUDE_extremum_implies_a_equals_one_l2565_256589

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^3 - 3*x*a

-- State the theorem
theorem extremum_implies_a_equals_one (a : ℝ) : 
  (∃ (ε : ℝ), ε > 0 ∧ ∀ (x : ℝ), x ≠ 1 ∧ |x - 1| < ε → f a x ≤ f a 1) →
  a = 1 := by
  sorry

end NUMINAMATH_CALUDE_extremum_implies_a_equals_one_l2565_256589


namespace NUMINAMATH_CALUDE_sufficient_condition_for_collinearity_l2565_256536

/-- Two vectors are collinear if one is a scalar multiple of the other -/
def collinear (a b : ℝ × ℝ) : Prop :=
  ∃ k : ℝ, a.1 * b.2 = k * a.2 * b.1

theorem sufficient_condition_for_collinearity (x : ℝ) :
  let a : ℝ × ℝ := (1, 2 - x)
  let b : ℝ × ℝ := (2 + x, 3)
  b = (1, 3) → collinear a b :=
by
  sorry

end NUMINAMATH_CALUDE_sufficient_condition_for_collinearity_l2565_256536


namespace NUMINAMATH_CALUDE_quadratic_range_theorem_l2565_256560

/-- A quadratic function passing through specific points -/
def quadratic_function (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

/-- The range of a quadratic function within a specific interval -/
def range_in_interval (f : ℝ → ℝ) (l u : ℝ) : Set ℝ :=
  {y | ∃ x, l < x ∧ x < u ∧ f x = y}

theorem quadratic_range_theorem (a b c : ℝ) (h : a ≠ 0) :
  quadratic_function a b c (-1) = -5 →
  quadratic_function a b c 0 = -8 →
  quadratic_function a b c 1 = -9 →
  quadratic_function a b c 3 = -5 →
  quadratic_function a b c 5 = 7 →
  range_in_interval (quadratic_function a b c) 0 5 = {y | -9 ≤ y ∧ y < 7} := by
  sorry

end NUMINAMATH_CALUDE_quadratic_range_theorem_l2565_256560


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l2565_256580

/-- Given an arithmetic sequence {a_n} with sum of first n terms S_n,
    prove that the common difference d equals 2 when (S_3 / 3) - (S_2 / 2) = 1 -/
theorem arithmetic_sequence_common_difference 
  (a : ℕ → ℝ) 
  (S : ℕ → ℝ) 
  (h_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)) 
  (h_sum : ∀ n, S n = (n / 2) * (2 * a 1 + (n - 1) * (a 2 - a 1))) 
  (h_condition : S 3 / 3 - S 2 / 2 = 1) :
  a 2 - a 1 = 2 :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l2565_256580


namespace NUMINAMATH_CALUDE_max_regions_50_lines_20_parallel_l2565_256570

/-- The maximum number of regions created by n lines in a plane -/
def max_regions (n : ℕ) : ℕ :=
  n * (n + 1) / 2 + 1

/-- The number of additional regions created by m parallel lines intersecting n non-parallel lines -/
def parallel_regions (m n : ℕ) : ℕ :=
  m * (n + 1)

/-- The maximum number of regions created by n lines in a plane, where p of them are parallel -/
def max_regions_with_parallel (n p : ℕ) : ℕ :=
  max_regions (n - p) + parallel_regions p (n - p)

theorem max_regions_50_lines_20_parallel :
  max_regions_with_parallel 50 20 = 1086 := by
  sorry

end NUMINAMATH_CALUDE_max_regions_50_lines_20_parallel_l2565_256570


namespace NUMINAMATH_CALUDE_petya_wins_l2565_256526

/-- Represents the game board -/
def Board (n : ℕ) := Fin n → Fin n → Bool

/-- Represents a position on the board -/
structure Position (n : ℕ) where
  row : Fin n
  col : Fin n

/-- Represents a player in the game -/
inductive Player
  | Petya
  | Vasya

/-- Represents the game state -/
structure GameState (n : ℕ) where
  board : Board n
  rook_position : Position n
  current_player : Player

/-- Checks if a move is valid -/
def is_valid_move (n : ℕ) (state : GameState n) (new_pos : Position n) : Bool :=
  sorry

/-- Applies a move to the game state -/
def apply_move (n : ℕ) (state : GameState n) (new_pos : Position n) : GameState n :=
  sorry

/-- Checks if the game is over -/
def is_game_over (n : ℕ) (state : GameState n) : Bool :=
  sorry

/-- The main theorem stating Petya has a winning strategy -/
theorem petya_wins (n : ℕ) (h : n ≥ 2) :
  ∃ (strategy : GameState n → Position n),
    ∀ (game : GameState n),
      game.current_player = Player.Petya →
      ¬(is_game_over n game) →
      is_valid_move n game (strategy game) ∧
      (∀ (vasya_move : Position n),
        is_valid_move n (apply_move n game (strategy game)) vasya_move →
        ∃ (petya_next_move : Position n),
          is_valid_move n (apply_move n (apply_move n game (strategy game)) vasya_move) petya_next_move) :=
sorry

end NUMINAMATH_CALUDE_petya_wins_l2565_256526


namespace NUMINAMATH_CALUDE_arithmetic_geometric_ratio_l2565_256579

/-- Given an arithmetic sequence with non-zero common difference,
    if a_5, a_9, and a_15 form a geometric sequence,
    then a_15 / a_9 = 3/2 -/
theorem arithmetic_geometric_ratio 
  (a : ℕ → ℝ) 
  (d : ℝ) 
  (h_arith : ∀ n, a (n + 1) = a n + d) 
  (h_d_nonzero : d ≠ 0)
  (h_geom : (a 9) ^ 2 = (a 5) * (a 15)) :
  a 15 / a 9 = 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_ratio_l2565_256579


namespace NUMINAMATH_CALUDE_sin_negative_nineteen_sixths_pi_l2565_256507

theorem sin_negative_nineteen_sixths_pi : 
  Real.sin (-19/6 * Real.pi) = 1/2 := by sorry

end NUMINAMATH_CALUDE_sin_negative_nineteen_sixths_pi_l2565_256507


namespace NUMINAMATH_CALUDE_line_tangent_to_parabola_l2565_256549

/-- A line is tangent to a parabola if and only if the resulting quadratic equation has a double root -/
axiom tangent_condition (a b c : ℝ) : 
  b^2 - 4*a*c = 0 ↔ ∃ x, a*x^2 + b*x + c = 0 ∧ ∀ y, a*y^2 + b*y + c = 0 → y = x

/-- The problem statement -/
theorem line_tangent_to_parabola (k : ℝ) :
  (∀ x y : ℝ, y^2 = 32*x → (4*x + 6*y + k = 0 ↔ 
    ∃! t, 4*t + 6*(32*t)^(1/2) + k = 0 ∨ 4*t + 6*(-32*t)^(1/2) + k = 0)) →
  k = 72 := by
  sorry

end NUMINAMATH_CALUDE_line_tangent_to_parabola_l2565_256549


namespace NUMINAMATH_CALUDE_max_sin_theta_is_one_l2565_256596

theorem max_sin_theta_is_one (a b : ℝ) (h_nonzero : a ≠ 0 ∧ b ≠ 0) :
  (∃ θ : ℝ, a * Real.sin θ + b * Real.cos θ ≥ 0 ∧ a * Real.cos θ - b * Real.sin θ ≥ 0) →
  (∃ θ : ℝ, a * Real.sin θ + b * Real.cos θ ≥ 0 ∧ a * Real.cos θ - b * Real.sin θ ≥ 0 ∧
    ∀ φ : ℝ, (a * Real.sin φ + b * Real.cos φ ≥ 0 ∧ a * Real.cos φ - b * Real.sin φ ≥ 0) →
      Real.sin θ ≥ Real.sin φ) →
  (∃ θ : ℝ, a * Real.sin θ + b * Real.cos θ ≥ 0 ∧ a * Real.cos θ - b * Real.sin θ ≥ 0 ∧ Real.sin θ = 1) :=
sorry

end NUMINAMATH_CALUDE_max_sin_theta_is_one_l2565_256596


namespace NUMINAMATH_CALUDE_lunks_for_apples_l2565_256529

/-- The number of lunks that can be traded for a given number of kunks -/
def lunks_per_kunks : ℚ := 4 / 2

/-- The number of kunks that can be traded for a given number of apples -/
def kunks_per_apples : ℚ := 3 / 5

/-- The number of apples we want to purchase -/
def target_apples : ℕ := 20

/-- Theorem: The number of lunks needed to purchase 20 apples is 24 -/
theorem lunks_for_apples : 
  (target_apples : ℚ) * kunks_per_apples * lunks_per_kunks = 24 := by sorry

end NUMINAMATH_CALUDE_lunks_for_apples_l2565_256529


namespace NUMINAMATH_CALUDE_two_numbers_difference_l2565_256532

theorem two_numbers_difference (x y : ℝ) (h1 : x + y = 10) (h2 : x^2 - y^2 = 20) : 
  |x - y| = 2 := by
sorry

end NUMINAMATH_CALUDE_two_numbers_difference_l2565_256532


namespace NUMINAMATH_CALUDE_non_empty_proper_subsets_of_A_l2565_256524

def A : Set ℕ := {2, 3}

theorem non_empty_proper_subsets_of_A :
  {s : Set ℕ | s ⊆ A ∧ s ≠ ∅ ∧ s ≠ A} = {{2}, {3}} := by sorry

end NUMINAMATH_CALUDE_non_empty_proper_subsets_of_A_l2565_256524


namespace NUMINAMATH_CALUDE_sheep_to_cow_ratio_is_ten_to_one_l2565_256587

/-- Represents the farm owned by Mr. Reyansh -/
structure Farm where
  num_cows : ℕ
  cow_water_daily : ℕ
  sheep_water_ratio : ℚ
  total_water_weekly : ℕ

/-- Calculates the ratio of sheep to cows on the farm -/
def sheep_to_cow_ratio (f : Farm) : ℚ :=
  let cow_water_weekly := f.num_cows * f.cow_water_daily * 7
  let sheep_water_weekly := f.total_water_weekly - cow_water_weekly
  let sheep_water_daily := sheep_water_weekly / 7
  let num_sheep := sheep_water_daily / (f.cow_water_daily * f.sheep_water_ratio)
  num_sheep / f.num_cows

/-- Theorem stating that the ratio of sheep to cows is 10:1 -/
theorem sheep_to_cow_ratio_is_ten_to_one (f : Farm) 
    (h1 : f.num_cows = 40)
    (h2 : f.cow_water_daily = 80)
    (h3 : f.sheep_water_ratio = 1/4)
    (h4 : f.total_water_weekly = 78400) :
  sheep_to_cow_ratio f = 10 := by
  sorry

#eval sheep_to_cow_ratio { num_cows := 40, cow_water_daily := 80, sheep_water_ratio := 1/4, total_water_weekly := 78400 }

end NUMINAMATH_CALUDE_sheep_to_cow_ratio_is_ten_to_one_l2565_256587


namespace NUMINAMATH_CALUDE_toy_poodle_height_l2565_256514

/-- The height of the toy poodle given the heights of other poodle types -/
theorem toy_poodle_height (h_standard : ℕ) (h_mini : ℕ) (h_toy : ℕ)
  (standard_mini : h_standard = h_mini + 8)
  (mini_toy : h_mini = h_toy + 6)
  (standard_height : h_standard = 28) :
  h_toy = 14 := by
  sorry

end NUMINAMATH_CALUDE_toy_poodle_height_l2565_256514


namespace NUMINAMATH_CALUDE_team_selection_count_l2565_256534

def total_athletes : ℕ := 10
def veteran_players : ℕ := 2
def new_players : ℕ := 8
def team_size : ℕ := 3
def excluded_new_player : ℕ := 1

theorem team_selection_count :
  (Nat.choose veteran_players 1 * Nat.choose (new_players - excluded_new_player) 2) +
  (Nat.choose (new_players - excluded_new_player) team_size) = 77 := by
  sorry

end NUMINAMATH_CALUDE_team_selection_count_l2565_256534


namespace NUMINAMATH_CALUDE_factorize_xm_minus_xn_l2565_256597

theorem factorize_xm_minus_xn (x m n : ℝ) : x * m - x * n = x * (m - n) := by
  sorry

end NUMINAMATH_CALUDE_factorize_xm_minus_xn_l2565_256597


namespace NUMINAMATH_CALUDE_complementary_angles_ratio_l2565_256586

/-- Two complementary angles in a ratio of 5:4 have the larger angle measuring 50 degrees -/
theorem complementary_angles_ratio (a b : ℝ) : 
  a + b = 90 →  -- angles are complementary
  a / b = 5 / 4 →  -- ratio of angles is 5:4
  max a b = 50 :=  -- larger angle measures 50 degrees
by sorry

end NUMINAMATH_CALUDE_complementary_angles_ratio_l2565_256586


namespace NUMINAMATH_CALUDE_intersection_complement_when_m_3_find_m_for_given_intersection_l2565_256540

-- Define set A
def A : Set ℝ := {x | |x - 2| < 3}

-- Define set B (parameterized by m)
def B (m : ℝ) : Set ℝ := {x | x^2 - 2*x - m < 0}

-- Theorem 1
theorem intersection_complement_when_m_3 :
  A ∩ (Set.univ \ B 3) = {x | 3 ≤ x ∧ x < 5} :=
sorry

-- Theorem 2
theorem find_m_for_given_intersection :
  A ∩ B 8 = {x | -1 < x ∧ x < 4} :=
sorry

end NUMINAMATH_CALUDE_intersection_complement_when_m_3_find_m_for_given_intersection_l2565_256540


namespace NUMINAMATH_CALUDE_max_ratio_squared_l2565_256537

theorem max_ratio_squared (c d x y : ℝ) (hc : c > 0) (hd : d > 0) (hcd : c ≥ d)
  (heq : c^2 + y^2 = d^2 + x^2 ∧ d^2 + x^2 = (c - x)^2 + (d - y)^2)
  (hx : 0 ≤ x ∧ x < c) (hy : 0 ≤ y ∧ y < d) :
  (c / d)^2 ≤ 4/3 :=
sorry

end NUMINAMATH_CALUDE_max_ratio_squared_l2565_256537


namespace NUMINAMATH_CALUDE_obtuse_angle_range_l2565_256515

/-- Two vectors form an obtuse angle if their dot product is negative -/
def obtuse_angle (a b : ℝ × ℝ) : Prop :=
  a.1 * b.1 + a.2 * b.2 < 0

/-- The theorem stating the range of m for which vectors a and b form an obtuse angle -/
theorem obtuse_angle_range :
  ∀ m : ℝ, obtuse_angle (-2, 3) (1, m) ↔ m < 2/3 ∧ m ≠ -3/2 := by
  sorry

end NUMINAMATH_CALUDE_obtuse_angle_range_l2565_256515


namespace NUMINAMATH_CALUDE_karen_cake_days_l2565_256595

/-- The number of school days in a week -/
def school_days : ℕ := 5

/-- The number of days Karen packs ham sandwiches -/
def ham_days : ℕ := 3

/-- The probability of packing a ham sandwich and cake on the same day -/
def ham_cake_prob : ℚ := 12 / 100

/-- The number of days Karen packs a piece of cake -/
def cake_days : ℕ := sorry

theorem karen_cake_days :
  (ham_days : ℚ) / school_days * cake_days / school_days = ham_cake_prob →
  cake_days = 1 := by sorry

end NUMINAMATH_CALUDE_karen_cake_days_l2565_256595


namespace NUMINAMATH_CALUDE_max_l_pieces_theorem_max_l_pieces_5x10_max_l_pieces_5x9_l2565_256551

/-- Represents an L-shaped piece consisting of 3 cells -/
structure LPiece where
  size : Nat
  size_eq : size = 3

/-- Represents a rectangular grid -/
structure Grid where
  rows : Nat
  cols : Nat

/-- Calculates the maximum number of L-shaped pieces that can be cut from a grid -/
def maxLPieces (g : Grid) (l : LPiece) : Nat :=
  (g.rows * g.cols) / l.size

/-- Theorem: The maximum number of L-shaped pieces in a grid is the floor of total cells divided by piece size -/
theorem max_l_pieces_theorem (g : Grid) (l : LPiece) :
  maxLPieces g l = ⌊(g.rows * g.cols : ℚ) / l.size⌋ :=
sorry

/-- Corollary: For a 5x10 grid, the maximum number of L-shaped pieces is 16 -/
theorem max_l_pieces_5x10 :
  maxLPieces { rows := 5, cols := 10 } { size := 3, size_eq := rfl } = 16 :=
sorry

/-- Corollary: For a 5x9 grid, the maximum number of L-shaped pieces is 15 -/
theorem max_l_pieces_5x9 :
  maxLPieces { rows := 5, cols := 9 } { size := 3, size_eq := rfl } = 15 :=
sorry

end NUMINAMATH_CALUDE_max_l_pieces_theorem_max_l_pieces_5x10_max_l_pieces_5x9_l2565_256551


namespace NUMINAMATH_CALUDE_tutor_schedule_lcm_l2565_256591

theorem tutor_schedule_lcm : Nat.lcm 5 (Nat.lcm 6 (Nat.lcm 10 8)) = 120 := by
  sorry

end NUMINAMATH_CALUDE_tutor_schedule_lcm_l2565_256591


namespace NUMINAMATH_CALUDE_necklace_price_l2565_256541

theorem necklace_price (bracelet_price earring_price ensemble_price : ℕ)
                       (necklaces bracelets earrings ensembles : ℕ)
                       (total_revenue : ℕ) :
  bracelet_price = 15 →
  earring_price = 10 →
  ensemble_price = 45 →
  necklaces = 5 →
  bracelets = 10 →
  earrings = 20 →
  ensembles = 2 →
  total_revenue = 565 →
  ∃ (necklace_price : ℕ),
    necklace_price = 25 ∧
    necklace_price * necklaces + bracelet_price * bracelets + 
    earring_price * earrings + ensemble_price * ensembles = total_revenue :=
by
  sorry

end NUMINAMATH_CALUDE_necklace_price_l2565_256541


namespace NUMINAMATH_CALUDE_unique_conjugate_pair_l2565_256561

/-- A quadratic trinomial function -/
def QuadraticTrinomial (a b c : ℝ) : ℝ → ℝ := fun x ↦ a * x^2 + b * x + c

/-- Conjugate numbers for a function -/
def Conjugate (f : ℝ → ℝ) (x y : ℝ) : Prop := f x = y ∧ f y = x

theorem unique_conjugate_pair (a b c : ℝ) (x y : ℝ) :
  x ≠ y →
  let f := QuadraticTrinomial a b c
  Conjugate f x y →
  ∀ u v : ℝ, Conjugate f u v → (u = x ∧ v = y) ∨ (u = y ∧ v = x) := by
  sorry

end NUMINAMATH_CALUDE_unique_conjugate_pair_l2565_256561


namespace NUMINAMATH_CALUDE_smallest_N_l2565_256546

theorem smallest_N (k : ℕ) (hk : k ≥ 1) :
  let N := 2 * k^3 + 3 * k^2 + k
  ∀ (a : Fin (2 * k + 1) → ℕ),
    (∀ i, a i ≥ 1) →
    (∀ i j, i ≠ j → a i ≠ a j) →
    (Finset.sum Finset.univ a ≥ N) →
    (∀ s : Finset (Fin (2 * k + 1)), s.card = k → Finset.sum s a ≤ N / 2) →
    ∀ M : ℕ, M < N →
      ¬∃ (b : Fin (2 * k + 1) → ℕ),
        (∀ i, b i ≥ 1) ∧
        (∀ i j, i ≠ j → b i ≠ b j) ∧
        (Finset.sum Finset.univ b ≥ M) ∧
        (∀ s : Finset (Fin (2 * k + 1)), s.card = k → Finset.sum s b ≤ M / 2) :=
by sorry

end NUMINAMATH_CALUDE_smallest_N_l2565_256546


namespace NUMINAMATH_CALUDE_first_term_of_arithmetic_sequence_l2565_256552

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem first_term_of_arithmetic_sequence
  (a : ℕ → ℝ)
  (h_arith : arithmetic_sequence a)
  (h_a3 : a 3 = 3)
  (h_d : ∃ d : ℝ, d = 2 ∧ ∀ n : ℕ, a (n + 1) = a n + d) :
  a 1 = -1 := by
sorry

end NUMINAMATH_CALUDE_first_term_of_arithmetic_sequence_l2565_256552


namespace NUMINAMATH_CALUDE_u_2008_eq_225_l2565_256516

/-- Defines the sequence u_n as described in the problem -/
def u : ℕ → ℕ := sorry

/-- The 2008th term of the sequence is 225 -/
theorem u_2008_eq_225 : u 2008 = 225 := by sorry

end NUMINAMATH_CALUDE_u_2008_eq_225_l2565_256516


namespace NUMINAMATH_CALUDE_older_brother_running_distance_l2565_256599

/-- The running speed of the older brother in meters per minute -/
def older_brother_speed : ℝ := 110

/-- The running speed of the younger brother in meters per minute -/
def younger_brother_speed : ℝ := 80

/-- The additional time the younger brother runs in minutes -/
def additional_time : ℝ := 30

/-- The additional distance the younger brother runs in meters -/
def additional_distance : ℝ := 900

/-- The distance run by the older brother in meters -/
def older_brother_distance : ℝ := 5500

theorem older_brother_running_distance :
  ∃ (t : ℝ), 
    t > 0 ∧
    (t + additional_time) * younger_brother_speed = t * older_brother_speed + additional_distance ∧
    t * older_brother_speed = older_brother_distance :=
by sorry

end NUMINAMATH_CALUDE_older_brother_running_distance_l2565_256599


namespace NUMINAMATH_CALUDE_candy_bar_ratio_l2565_256588

/-- Proves the ratio of candy bars given the second time to the first time -/
theorem candy_bar_ratio (initial_bars : ℕ) (initial_given : ℕ) (bought_bars : ℕ) (kept_bars : ℕ) :
  initial_bars = 7 →
  initial_given = 3 →
  bought_bars = 30 →
  kept_bars = 22 →
  ∃ (second_given : ℕ), 
    second_given = initial_bars + bought_bars - kept_bars - initial_given ∧
    second_given = 4 * initial_given :=
by sorry

end NUMINAMATH_CALUDE_candy_bar_ratio_l2565_256588


namespace NUMINAMATH_CALUDE_triangle_theorem_triangle_max_sum_l2565_256571

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ
  positive_sides : 0 < a ∧ 0 < b ∧ 0 < c
  positive_angles : 0 < A ∧ 0 < B ∧ 0 < C
  angle_sum : A + B + C = π

theorem triangle_theorem (t : Triangle) (h : 2 * t.c - t.a = 2 * t.b * Real.cos t.A) :
  t.B = π / 3 := by sorry

theorem triangle_max_sum (t : Triangle) 
  (h1 : 2 * t.c - t.a = 2 * t.b * Real.cos t.A) 
  (h2 : t.b = 2 * Real.sqrt 3) :
  (∀ (s : Triangle), s.a + s.c ≤ 4 * Real.sqrt 3) ∧ 
  (∃ (s : Triangle), s.a + s.c = 4 * Real.sqrt 3) := by sorry

end NUMINAMATH_CALUDE_triangle_theorem_triangle_max_sum_l2565_256571


namespace NUMINAMATH_CALUDE_area_of_triangle_DEF_l2565_256505

-- Define the square PQRS
def PQRS_area : ℝ := 36

-- Define the side length of smaller squares
def small_square_side : ℝ := 2

-- Define the triangle DEF
structure Triangle_DEF where
  DE : ℝ
  DF : ℝ
  EF : ℝ

-- Define the folding property
def folds_to_center (t : Triangle_DEF) (s : ℝ) : Prop :=
  t.DE = t.DF ∧ t.DE = s / 2 + 2 * small_square_side

-- Main theorem
theorem area_of_triangle_DEF (t : Triangle_DEF) (s : ℝ) :
  s^2 = PQRS_area →
  folds_to_center t s →
  t.EF = s - 2 * small_square_side →
  (1/2) * t.EF * t.DE = 10 := by sorry

end NUMINAMATH_CALUDE_area_of_triangle_DEF_l2565_256505


namespace NUMINAMATH_CALUDE_equation_solutions_l2565_256584

theorem equation_solutions : 
  let f (x : ℝ) := 1 / (x^2 + 10*x - 8) + 1 / (x^2 + 3*x - 8) + 1 / (x^2 - 12*x - 8)
  ∀ x : ℝ, f x = 0 ↔ x = 1 ∨ x = -19 ∨ x = (5 + Real.sqrt 57) / 2 ∨ x = (5 - Real.sqrt 57) / 2 :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l2565_256584


namespace NUMINAMATH_CALUDE_integer_sum_l2565_256545

theorem integer_sum (x y : ℕ+) (h1 : x.val - y.val = 8) (h2 : x.val * y.val = 288) : 
  x.val + y.val = 35 := by
sorry

end NUMINAMATH_CALUDE_integer_sum_l2565_256545


namespace NUMINAMATH_CALUDE_clock_time_after_2016_hours_l2565_256501

theorem clock_time_after_2016_hours (current_time : ℕ) (hours_passed : ℕ) : 
  current_time = 7 → hours_passed = 2016 → (current_time + hours_passed) % 12 = 7 := by
  sorry

end NUMINAMATH_CALUDE_clock_time_after_2016_hours_l2565_256501


namespace NUMINAMATH_CALUDE_composite_expression_l2565_256569

theorem composite_expression (n : ℕ) (h : n ≥ 2) :
  ∃ (a b : ℕ), a > 1 ∧ b > 1 ∧ 3^(2*n+1) - 2^(2*n+1) - 6^n = a * b := by
  sorry

end NUMINAMATH_CALUDE_composite_expression_l2565_256569


namespace NUMINAMATH_CALUDE_work_completion_fraction_l2565_256572

theorem work_completion_fraction (x_days y_days z_days total_days : ℕ) 
  (hx : x_days = 14) 
  (hy : y_days = 20) 
  (hz : z_days = 25) 
  (ht : total_days = 5) : 
  (total_days : ℚ) * ((1 : ℚ) / x_days + (1 : ℚ) / y_days + (1 : ℚ) / z_days) = 113 / 140 := by
  sorry

end NUMINAMATH_CALUDE_work_completion_fraction_l2565_256572


namespace NUMINAMATH_CALUDE_complement_of_A_l2565_256573

def A : Set ℝ := {x | |x - 1| ≤ 2}

theorem complement_of_A :
  Aᶜ = {x : ℝ | x < -1 ∨ x > 3} :=
by sorry

end NUMINAMATH_CALUDE_complement_of_A_l2565_256573


namespace NUMINAMATH_CALUDE_papaya_tree_first_year_growth_l2565_256583

/-- The growth pattern of a papaya tree over 5 years -/
def PapayaTreeGrowth (first_year_growth : ℝ) : ℝ :=
  let second_year := 1.5 * first_year_growth
  let third_year := 1.5 * second_year
  let fourth_year := 2 * third_year
  let fifth_year := 0.5 * fourth_year
  first_year_growth + second_year + third_year + fourth_year + fifth_year

/-- Theorem stating that if a papaya tree grows to 23 feet in 5 years following the given pattern, 
    it must have grown 2 feet in the first year -/
theorem papaya_tree_first_year_growth :
  ∃ (x : ℝ), PapayaTreeGrowth x = 23 → x = 2 :=
sorry

end NUMINAMATH_CALUDE_papaya_tree_first_year_growth_l2565_256583


namespace NUMINAMATH_CALUDE_gcd_divides_n_plus_two_l2565_256565

theorem gcd_divides_n_plus_two (a b n : ℤ) 
  (h_coprime : Nat.Coprime a.natAbs b.natAbs) 
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) : 
  ∃ k : ℤ, k * Int.gcd (a^2 + b^2 - n*a*b) (a + b) = n + 2 := by
  sorry

end NUMINAMATH_CALUDE_gcd_divides_n_plus_two_l2565_256565


namespace NUMINAMATH_CALUDE_age_problem_l2565_256593

theorem age_problem (a b c d : ℕ) : 
  a = b + 2 →
  b = 2 * c →
  d = c / 2 →
  a + b + c + d = 39 →
  b = 14 := by
sorry

end NUMINAMATH_CALUDE_age_problem_l2565_256593


namespace NUMINAMATH_CALUDE_stock_annual_return_l2565_256553

/-- Calculates the annual return percentage given initial price and price increase -/
def annual_return_percentage (initial_price price_increase : ℚ) : ℚ :=
  (price_increase / initial_price) * 100

/-- Theorem: The annual return percentage for a stock with initial price 8000 and price increase 400 is 5% -/
theorem stock_annual_return :
  let initial_price : ℚ := 8000
  let price_increase : ℚ := 400
  annual_return_percentage initial_price price_increase = 5 := by
  sorry

#eval annual_return_percentage 8000 400

end NUMINAMATH_CALUDE_stock_annual_return_l2565_256553


namespace NUMINAMATH_CALUDE_problem_statement_l2565_256582

theorem problem_statement (a b c : ℝ) 
  (ha : a < 0) (hb : b < 0) (hc1 : 0 ≤ c) (hc2 : c < -b) :
  (a + b < b + c) ∧ (c / a < 1) := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l2565_256582


namespace NUMINAMATH_CALUDE_goldbach_for_given_numbers_l2565_256518

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

def goldbach_for_number (n : ℕ) : Prop :=
  ∃ p q : ℕ, is_prime p ∧ is_prime q ∧ n = p + q

theorem goldbach_for_given_numbers :
  goldbach_for_number 102 ∧
  goldbach_for_number 144 ∧
  goldbach_for_number 178 ∧
  goldbach_for_number 200 :=
sorry

end NUMINAMATH_CALUDE_goldbach_for_given_numbers_l2565_256518
