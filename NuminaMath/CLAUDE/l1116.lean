import Mathlib

namespace bennys_total_work_hours_l1116_111687

/-- Calculates the total hours worked given the hours per day and number of days -/
def total_hours (hours_per_day : ℕ) (days : ℕ) : ℕ :=
  hours_per_day * days

/-- Theorem: Benny's total work hours -/
theorem bennys_total_work_hours :
  let hours_per_day : ℕ := 3
  let days : ℕ := 6
  total_hours hours_per_day days = 18 := by
  sorry

end bennys_total_work_hours_l1116_111687


namespace stock_price_calculation_l1116_111647

theorem stock_price_calculation (closing_price : ℝ) (percent_increase : ℝ) (opening_price : ℝ) : 
  closing_price = 16 → 
  percent_increase = 6.666666666666665 → 
  closing_price = opening_price * (1 + percent_increase / 100) →
  opening_price = 15 := by
sorry

end stock_price_calculation_l1116_111647


namespace additional_amount_proof_l1116_111629

theorem additional_amount_proof (n : ℕ) (h : n = 3) : 7 * n - 3 * n = 12 := by
  sorry

end additional_amount_proof_l1116_111629


namespace expression_meaning_l1116_111631

theorem expression_meaning (a : ℝ) : 2 * (a - 3)^2 = 2 * (a - 3) * (a - 3) := by
  sorry

end expression_meaning_l1116_111631


namespace leah_bird_feeding_l1116_111652

/-- The number of weeks Leah can feed her birds without going back to the store -/
def weeks_of_feed (boxes_bought : ℕ) (boxes_in_pantry : ℕ) (parrot_consumption : ℕ) (cockatiel_consumption : ℕ) (grams_per_box : ℕ) : ℕ :=
  let total_boxes := boxes_bought + boxes_in_pantry
  let total_grams := total_boxes * grams_per_box
  let weekly_consumption := parrot_consumption + cockatiel_consumption
  total_grams / weekly_consumption

/-- Theorem stating that Leah can feed her birds for 12 weeks -/
theorem leah_bird_feeding :
  weeks_of_feed 3 5 100 50 225 = 12 := by
sorry

end leah_bird_feeding_l1116_111652


namespace swimmer_speed_is_5_l1116_111619

/-- Represents the speed of a swimmer in still water and the speed of the stream. -/
structure SwimmingScenario where
  swimmer_speed : ℝ
  stream_speed : ℝ

/-- Calculates the effective speed when swimming downstream. -/
def downstream_speed (s : SwimmingScenario) : ℝ :=
  s.swimmer_speed + s.stream_speed

/-- Calculates the effective speed when swimming upstream. -/
def upstream_speed (s : SwimmingScenario) : ℝ :=
  s.swimmer_speed - s.stream_speed

/-- Theorem stating that given the conditions, the swimmer's speed in still water is 5 km/h. -/
theorem swimmer_speed_is_5 (s : SwimmingScenario) 
    (h_downstream : downstream_speed s * 6 = 54)
    (h_upstream : upstream_speed s * 6 = 6) : 
    s.swimmer_speed = 5 := by
  sorry


end swimmer_speed_is_5_l1116_111619


namespace min_students_satisfying_conditions_l1116_111670

/-- Represents the number of students in a classroom. -/
structure Classroom where
  boys : ℕ
  girls : ℕ

/-- Checks if the given classroom satisfies all conditions. -/
def satisfiesConditions (c : Classroom) : Prop :=
  ∃ (passed_boys passed_girls : ℕ),
    passed_boys = passed_girls ∧
    passed_boys = (3 * c.boys) / 5 ∧
    passed_girls = (2 * c.girls) / 3 ∧
    (c.boys + c.girls) % 10 = 0

/-- The theorem stating the minimum number of students satisfying all conditions. -/
theorem min_students_satisfying_conditions :
  ∀ c : Classroom, satisfiesConditions c →
    ∀ c' : Classroom, satisfiesConditions c' →
      c.boys + c.girls ≤ c'.boys + c'.girls →
        c.boys + c.girls = 38 := by
  sorry

#check min_students_satisfying_conditions

end min_students_satisfying_conditions_l1116_111670


namespace incenter_inside_BOH_l1116_111699

/-- Triangle type with vertices A, B, C -/
structure Triangle where
  A : ℝ × ℝ
  B : ℝ × ℝ
  C : ℝ × ℝ

/-- Angle measure of a triangle -/
def angle (t : Triangle) (v : ℝ × ℝ) : ℝ := sorry

/-- Incenter of a triangle -/
def incenter (t : Triangle) : ℝ × ℝ := sorry

/-- Circumcenter of a triangle -/
def circumcenter (t : Triangle) : ℝ × ℝ := sorry

/-- Orthocenter of a triangle -/
def orthocenter (t : Triangle) : ℝ × ℝ := sorry

/-- Check if a point is inside a triangle -/
def is_inside (p : ℝ × ℝ) (t : Triangle) : Prop := sorry

/-- Main theorem: Incenter lies inside the triangle formed by circumcenter, vertex B, and orthocenter -/
theorem incenter_inside_BOH (t : Triangle) 
  (h1 : angle t t.C > angle t t.B)
  (h2 : angle t t.B > angle t t.A) : 
  is_inside (incenter t) (Triangle.mk (circumcenter t) t.B (orthocenter t)) := by
  sorry

end incenter_inside_BOH_l1116_111699


namespace converse_and_inverse_false_l1116_111614

-- Define the types
def Quadrilateral : Type := sorry
def Rhombus : Type := sorry
def Parallelogram : Type := sorry

-- Define the properties
def is_rhombus : Quadrilateral → Prop := sorry
def is_parallelogram : Quadrilateral → Prop := sorry

-- Given statement
axiom rhombus_is_parallelogram : ∀ q : Quadrilateral, is_rhombus q → is_parallelogram q

-- Theorem to prove
theorem converse_and_inverse_false : 
  (∃ q : Quadrilateral, is_parallelogram q ∧ ¬is_rhombus q) ∧ 
  (∃ q : Quadrilateral, ¬is_rhombus q ∧ is_parallelogram q) := by
  sorry

end converse_and_inverse_false_l1116_111614


namespace trigonometric_identities_l1116_111661

open Real

theorem trigonometric_identities (α : ℝ) (h : tan α = 2) : 
  ((2 * sin α - 3 * cos α) / (4 * sin α - 9 * cos α) = -1) ∧ 
  (4 * sin α ^ 2 - 3 * sin α * cos α - 5 * cos α ^ 2 = 1) := by sorry

end trigonometric_identities_l1116_111661


namespace equation_solution_l1116_111634

theorem equation_solution (a : ℤ) : 
  (∃ x : ℤ, x > 0 ∧ a * x + 3 = 4 * x + 1) ↔ (a = 2 ∨ a = 3) :=
by sorry

end equation_solution_l1116_111634


namespace hamilton_marching_band_max_members_l1116_111646

theorem hamilton_marching_band_max_members :
  ∃ (m : ℕ),
    (30 * m) % 31 = 5 ∧
    30 * m < 1500 ∧
    ∀ (n : ℕ), (30 * n) % 31 = 5 ∧ 30 * n < 1500 → 30 * n ≤ 30 * m :=
by sorry

end hamilton_marching_band_max_members_l1116_111646


namespace expensive_feed_cost_l1116_111638

/-- Prove that the cost per pound of the more expensive dog feed is $0.53 --/
theorem expensive_feed_cost 
  (total_mix : ℝ) 
  (target_cost : ℝ) 
  (cheap_feed_cost : ℝ) 
  (cheap_feed_amount : ℝ) 
  (h1 : total_mix = 35)
  (h2 : target_cost = 0.36)
  (h3 : cheap_feed_cost = 0.18)
  (h4 : cheap_feed_amount = 17)
  : ∃ expensive_feed_cost : ℝ, 
    expensive_feed_cost = 0.53 ∧
    expensive_feed_cost * (total_mix - cheap_feed_amount) + 
    cheap_feed_cost * cheap_feed_amount = 
    target_cost * total_mix := by
  sorry

end expensive_feed_cost_l1116_111638


namespace geni_phone_expense_l1116_111613

/-- Represents a telephone plan with fixed fee, free minutes, and per-minute rate -/
structure TelephonePlan where
  fixedFee : ℝ
  freeMinutes : ℕ
  ratePerMinute : ℝ

/-- Calculates the bill for a given usage in minutes -/
def calculateBill (plan : TelephonePlan) (usageMinutes : ℕ) : ℝ :=
  plan.fixedFee + max 0 (usageMinutes - plan.freeMinutes) * plan.ratePerMinute

/-- Converts hours and minutes to total minutes -/
def toMinutes (hours : ℕ) (minutes : ℕ) : ℕ := hours * 60 + minutes

theorem geni_phone_expense :
  let plan : TelephonePlan := { fixedFee := 18, freeMinutes := 600, ratePerMinute := 0.03 }
  let januaryUsage : ℕ := toMinutes 15 17
  let februaryUsage : ℕ := toMinutes 9 55
  calculateBill plan januaryUsage + calculateBill plan februaryUsage = 45.51 := by
  sorry

end geni_phone_expense_l1116_111613


namespace deal_or_no_deal_probability_l1116_111656

/-- The set of values in the Deal or No Deal game -/
def deal_values : Finset ℕ := {1, 5, 10, 25, 50, 75, 100, 200, 300, 400, 500, 750, 1000, 5000, 10000, 25000, 50000, 75000, 100000, 200000, 300000, 400000, 500000, 750000, 1000000}

/-- The number of boxes in the game -/
def total_boxes : ℕ := 26

/-- The threshold value for high-value boxes -/
def threshold : ℕ := 200000

/-- The set of high-value boxes -/
def high_value_boxes : Finset ℕ := deal_values.filter (λ x => x ≥ threshold)

/-- The number of boxes to eliminate -/
def boxes_to_eliminate : ℕ := 14

theorem deal_or_no_deal_probability :
  (total_boxes - boxes_to_eliminate) / 2 = high_value_boxes.card ∧
  (total_boxes - boxes_to_eliminate) % 2 = 0 :=
sorry

#eval deal_values.card
#eval total_boxes
#eval high_value_boxes
#eval boxes_to_eliminate

end deal_or_no_deal_probability_l1116_111656


namespace equation_solution_l1116_111674

theorem equation_solution (x : ℝ) (h1 : x ≠ 1) (h2 : x ≠ 0) :
  (2 / (x - 1) - (x + 2) / (x^2 - x) = 0) ↔ x = 2 := by
  sorry

end equation_solution_l1116_111674


namespace log_243_between_consecutive_integers_l1116_111655

theorem log_243_between_consecutive_integers (a b : ℤ) :
  (a : ℝ) < Real.log 243 / Real.log 5 ∧
  Real.log 243 / Real.log 5 < (b : ℝ) ∧
  b = a + 1 →
  a + b = 7 := by sorry

end log_243_between_consecutive_integers_l1116_111655


namespace categorical_variables_correct_l1116_111676

-- Define the type for variables
inductive Variable
  | Smoking
  | Gender
  | ReligiousBelief
  | Nationality

-- Define a function to check if a variable is categorical
def isCategorical (v : Variable) : Prop :=
  match v with
  | Variable.Smoking => False
  | _ => True

-- Define the set of all variables
def allVariables : Set Variable :=
  {Variable.Smoking, Variable.Gender, Variable.ReligiousBelief, Variable.Nationality}

-- Define the set of categorical variables
def categoricalVariables : Set Variable :=
  {v ∈ allVariables | isCategorical v}

-- Theorem statement
theorem categorical_variables_correct :
  categoricalVariables = {Variable.Gender, Variable.ReligiousBelief, Variable.Nationality} :=
by sorry

end categorical_variables_correct_l1116_111676


namespace number_of_divisors_180_l1116_111625

theorem number_of_divisors_180 : Nat.card {d : ℕ | d > 0 ∧ 180 % d = 0} = 18 := by
  sorry

end number_of_divisors_180_l1116_111625


namespace base_8_addition_l1116_111659

/-- Addition in base 8 -/
def add_base_8 (a b : ℕ) : ℕ := 
  (a + b) % 8

/-- Conversion from base 8 to base 10 -/
def base_8_to_10 (n : ℕ) : ℕ := 
  (n / 10) * 8 + (n % 10)

theorem base_8_addition : 
  add_base_8 (base_8_to_10 5) (base_8_to_10 13) = base_8_to_10 20 := by
  sorry

end base_8_addition_l1116_111659


namespace inequality_system_solution_l1116_111698

theorem inequality_system_solution :
  ∀ x : ℝ, (2 * x + 1 < 5 ∧ 2 - x ≤ 1) ↔ (1 ≤ x ∧ x < 2) := by sorry

end inequality_system_solution_l1116_111698


namespace probability_A_wins_after_four_games_l1116_111658

def probability_A_wins : ℚ := 3 / 5
def probability_B_wins : ℚ := 2 / 5
def number_of_games : ℕ := 4
def number_of_wins_needed : ℕ := 3

theorem probability_A_wins_after_four_games :
  (Nat.choose number_of_games number_of_wins_needed : ℚ) * 
  probability_A_wins ^ number_of_wins_needed * 
  probability_B_wins ^ (number_of_games - number_of_wins_needed) =
  (Nat.choose number_of_games number_of_wins_needed : ℚ) * 
  (3 / 5) ^ 3 * (2 / 5) := by
  sorry

end probability_A_wins_after_four_games_l1116_111658


namespace no_convex_equal_sided_all_obtuse_polygon_l1116_111609

/-- A polygon is represented as a list of points in 2D space -/
def Polygon := List (Real × Real)

/-- A polygon is convex if for any three consecutive vertices, the turn is always in the same direction -/
def is_convex (p : Polygon) : Prop := sorry

/-- All sides of a polygon have equal length -/
def has_equal_sides (p : Polygon) : Prop := sorry

/-- Three points form an obtuse triangle if one of its angles is greater than 90 degrees -/
def is_obtuse_triangle (a b c : Real × Real) : Prop := sorry

/-- Any three vertices of the polygon form an obtuse triangle -/
def all_triangles_obtuse (p : Polygon) : Prop := sorry

theorem no_convex_equal_sided_all_obtuse_polygon :
  ¬∃ (p : Polygon), is_convex p ∧ has_equal_sides p ∧ all_triangles_obtuse p := by
  sorry

end no_convex_equal_sided_all_obtuse_polygon_l1116_111609


namespace male_students_count_l1116_111626

/-- Given a school with a total of 1200 students, where a sample of 200 students
    contains 85 females, prove that the number of male students in the school is 690. -/
theorem male_students_count (total : ℕ) (sample : ℕ) (females_in_sample : ℕ)
    (h1 : total = 1200)
    (h2 : sample = 200)
    (h3 : females_in_sample = 85) :
    total - (females_in_sample * (total / sample)) = 690 := by
  sorry

end male_students_count_l1116_111626


namespace simplify_trig_expression_l1116_111666

theorem simplify_trig_expression (α : Real) (h : 270 * π / 180 < α ∧ α < 360 * π / 180) :
  Real.sqrt ((1/2) + (1/2) * Real.sqrt ((1/2) + (1/2) * Real.cos (2 * α))) = -Real.cos (α / 2) := by
  sorry

end simplify_trig_expression_l1116_111666


namespace tims_sock_drawer_probability_l1116_111691

/-- Represents the number of socks of each color in the drawer -/
structure SockDrawer where
  gray : ℕ
  white : ℕ
  black : ℕ

/-- Calculates the probability of picking a matching pair of socks -/
def probabilityOfMatchingPair (drawer : SockDrawer) : ℚ :=
  let totalSocks := drawer.gray + drawer.white + drawer.black
  let totalPairs := (totalSocks * (totalSocks - 1)) / 2
  let matchingPairs := (drawer.gray * (drawer.gray - 1) + 
                        drawer.white * (drawer.white - 1) + 
                        drawer.black * (drawer.black - 1)) / 2
  matchingPairs / totalPairs

/-- Theorem stating that the probability of picking a matching pair 
    from Tim's sock drawer is 1/3 -/
theorem tims_sock_drawer_probability : 
  probabilityOfMatchingPair ⟨12, 10, 6⟩ = 1/3 := by
  sorry

end tims_sock_drawer_probability_l1116_111691


namespace root_in_interval_l1116_111695

noncomputable def f (x : ℝ) : ℝ := 4 - 4*x - Real.exp x

theorem root_in_interval :
  ∃ x ∈ Set.Ioo 0 1, f x = 0 :=
by
  have h1 : Continuous f := sorry
  have h2 : StrictMono (fun x => -f x) := sorry
  have h3 : f 0 > 0 := sorry
  have h4 : f 1 < 0 := sorry
  sorry

end root_in_interval_l1116_111695


namespace complete_square_result_l1116_111611

/-- Given a quadratic equation x^2 + 6x - 3 = 0, prove that when completing the square, 
    the resulting equation (x + a)^2 = b has b = 12 -/
theorem complete_square_result (x : ℝ) : 
  (∃ a b : ℝ, x^2 + 6*x - 3 = 0 ↔ (x + a)^2 = b) → 
  (∃ a : ℝ, (x + a)^2 = 12) :=
by sorry

end complete_square_result_l1116_111611


namespace total_players_on_ground_l1116_111667

theorem total_players_on_ground (cricket_players hockey_players football_players softball_players : ℕ) 
  (h1 : cricket_players = 10)
  (h2 : hockey_players = 12)
  (h3 : football_players = 16)
  (h4 : softball_players = 13) :
  cricket_players + hockey_players + football_players + softball_players = 51 := by
  sorry

end total_players_on_ground_l1116_111667


namespace set_intersection_equality_l1116_111640

def A : Set ℝ := {x | x ≤ 2 * x + 1 ∧ 2 * x + 1 ≤ 5}
def B : Set ℝ := {x | 0 < x ∧ x ≤ 3}

theorem set_intersection_equality : A ∩ B = {x | 0 < x ∧ x ≤ 2} := by sorry

end set_intersection_equality_l1116_111640


namespace fencing_cost_is_2210_l1116_111686

/-- Calculates the total cost of fencing a rectangular plot -/
def total_fencing_cost (width : ℝ) (perimeter : ℝ) (cost_per_meter : ℝ) : ℝ :=
  perimeter * cost_per_meter

/-- Theorem: The total cost of fencing the rectangular plot is 2210 -/
theorem fencing_cost_is_2210 :
  ∃ (width : ℝ),
    let length := width + 10
    let perimeter := 2 * (length + width)
    perimeter = 340 ∧
    total_fencing_cost width perimeter 6.5 = 2210 := by
  sorry

end fencing_cost_is_2210_l1116_111686


namespace polynomial_root_ratio_l1116_111671

theorem polynomial_root_ratio (a b c d e : ℝ) (h1 : a ≠ 0) :
  (∀ x : ℝ, a*x^4 + b*x^3 + c*x^2 + d*x + e = 0 ↔ x = 5 ∨ x = -3 ∨ x = 2 ∨ x = (-(b+d)/a - 5 - (-3) - 2)) →
  (b + d) / a = -12496 / 3173 := by
sorry

end polynomial_root_ratio_l1116_111671


namespace average_salary_combined_l1116_111601

theorem average_salary_combined (num_supervisors : ℕ) (num_laborers : ℕ) 
  (avg_salary_supervisors : ℚ) (avg_salary_laborers : ℚ) :
  num_supervisors = 6 →
  num_laborers = 42 →
  avg_salary_supervisors = 2450 →
  avg_salary_laborers = 950 →
  let total_salary := num_supervisors * avg_salary_supervisors + num_laborers * avg_salary_laborers
  let total_workers := num_supervisors + num_laborers
  (total_salary / total_workers : ℚ) = 1137.5 := by
sorry

end average_salary_combined_l1116_111601


namespace final_salary_calculation_l1116_111633

/-- Calculates the final salary after two salary changes --/
theorem final_salary_calculation (initial_salary : ℝ) (first_year_raise : ℝ) (second_year_cut : ℝ) :
  initial_salary = 10 →
  first_year_raise = 0.2 →
  second_year_cut = 0.75 →
  initial_salary * (1 + first_year_raise) * second_year_cut = 9 := by
  sorry

end final_salary_calculation_l1116_111633


namespace middle_number_is_four_l1116_111645

/-- Represents a triple of positive integers -/
structure Triple where
  left : Nat
  middle : Nat
  right : Nat
  left_pos : 0 < left
  middle_pos : 0 < middle
  right_pos : 0 < right

/-- Checks if a triple satisfies the problem conditions -/
def validTriple (t : Triple) : Prop :=
  t.left < t.middle ∧ t.middle < t.right ∧ t.left + t.middle + t.right = 15

/-- Casey cannot determine the other two numbers -/
def caseyUncertain (t : Triple) : Prop :=
  ∃ t' : Triple, t'.left = t.left ∧ validTriple t' ∧ t' ≠ t

/-- Tracy cannot determine the other two numbers -/
def tracyUncertain (t : Triple) : Prop :=
  ∃ t' : Triple, t'.right = t.right ∧ validTriple t' ∧ t' ≠ t

/-- Stacy cannot determine the other two numbers -/
def stacyUncertain (t : Triple) : Prop :=
  ∃ t' : Triple, t'.middle = t.middle ∧ validTriple t' ∧ t' ≠ t

/-- The main theorem stating that the middle number must be 4 -/
theorem middle_number_is_four :
  ∀ t : Triple,
    validTriple t →
    caseyUncertain t →
    tracyUncertain t →
    stacyUncertain t →
    t.middle = 4 := by
  sorry


end middle_number_is_four_l1116_111645


namespace dihedral_angle_of_inscribed_spheres_l1116_111612

theorem dihedral_angle_of_inscribed_spheres (r R : ℝ) (θ : ℝ) : 
  r > 0 → 
  R = 3 * r → 
  (R + r) * Real.cos θ = (R + r) * (1/2) → 
  Real.cos (θ) = 1/3 := by
sorry

end dihedral_angle_of_inscribed_spheres_l1116_111612


namespace reading_time_for_18_pages_l1116_111678

-- Define the reading rate (pages per minute)
def reading_rate : ℚ := 4 / 2

-- Define the number of pages to read
def pages_to_read : ℕ := 18

-- Theorem: It takes 9 minutes to read 18 pages at the given rate
theorem reading_time_for_18_pages :
  (pages_to_read : ℚ) / reading_rate = 9 := by
  sorry

end reading_time_for_18_pages_l1116_111678


namespace line_relations_l1116_111697

-- Define the concept of a line in 3D space
variable (Line : Type)

-- Define the parallel relation between two lines
variable (parallel : Line → Line → Prop)

-- Define the perpendicular relation between two lines
variable (perpendicular : Line → Line → Prop)

-- State the theorem
theorem line_relations (a b c : Line) :
  parallel a b → perpendicular a c → perpendicular b c := by
  sorry

end line_relations_l1116_111697


namespace cubic_coefficient_b_value_l1116_111651

/-- A cubic function passing through specific points -/
def g (a b c d : ℝ) (x : ℝ) : ℝ := a * x^3 + b * x^2 + c * x + d

/-- Theorem stating that for a cubic function passing through (2,0), (-1,0), and (1,4), b = 6 -/
theorem cubic_coefficient_b_value 
  (a b c d : ℝ) 
  (h1 : g a b c d 2 = 0)
  (h2 : g a b c d (-1) = 0)
  (h3 : g a b c d 1 = 4) :
  b = 6 := by
  sorry

end cubic_coefficient_b_value_l1116_111651


namespace special_circle_equation_l1116_111653

/-- A circle with center on the line 2x - y - 3 = 0 passing through (5, 2) and (3, -2) -/
def special_circle (a : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - a)^2 + (p.2 - (2*a - 3))^2 = ((5 - a)^2 + (2 - (2*a - 3))^2)}

/-- The equation of the circle is (x-2)^2 + (y-1)^2 = 10 -/
theorem special_circle_equation :
  ∃ a : ℝ, special_circle a = {p : ℝ × ℝ | (p.1 - 2)^2 + (p.2 - 1)^2 = 10} ∧
    (5, 2) ∈ special_circle a ∧ (3, -2) ∈ special_circle a :=
by
  sorry

end special_circle_equation_l1116_111653


namespace probability_inside_circle_l1116_111683

def is_inside_circle (x y : ℕ) : Prop := x^2 + y^2 < 9

def favorable_outcomes : ℕ := 4

def total_outcomes : ℕ := 36

theorem probability_inside_circle :
  (favorable_outcomes : ℚ) / total_outcomes = 1 / 9 :=
sorry

end probability_inside_circle_l1116_111683


namespace prism_pyramid_sum_l1116_111669

/-- A pentagonal prism with a pyramid attached to one of its pentagonal faces -/
structure PrismWithPyramid where
  prism_faces : Nat
  prism_vertices : Nat
  prism_edges : Nat
  pyramid_faces : Nat
  pyramid_vertex : Nat
  pyramid_edges : Nat

/-- The total number of exterior elements (faces, vertices, edges) of the combined solid -/
def total_elements (solid : PrismWithPyramid) : Nat :=
  (solid.prism_faces - 1 + solid.pyramid_faces) +
  (solid.prism_vertices + solid.pyramid_vertex) +
  (solid.prism_edges + solid.pyramid_edges)

/-- Theorem stating that the sum of exterior faces, vertices, and edges is 42 -/
theorem prism_pyramid_sum (solid : PrismWithPyramid)
  (h1 : solid.prism_faces = 7)
  (h2 : solid.prism_vertices = 10)
  (h3 : solid.prism_edges = 15)
  (h4 : solid.pyramid_faces = 5)
  (h5 : solid.pyramid_vertex = 1)
  (h6 : solid.pyramid_edges = 5) :
  total_elements solid = 42 := by
  sorry

end prism_pyramid_sum_l1116_111669


namespace sufficient_condition_l1116_111673

theorem sufficient_condition (θ P₁ P₂ : Prop) 
  (h1 : P₁ → θ) 
  (h2 : P₂ → P₁) : 
  P₂ → θ := by
sorry

end sufficient_condition_l1116_111673


namespace triangle_sine_property_l1116_111637

theorem triangle_sine_property (A B C : ℝ) (h : 3 * Real.sin B ^ 2 + 7 * Real.sin C ^ 2 = 2 * Real.sin A * Real.sin B * Real.sin C + 2 * Real.sin A ^ 2) :
  Real.sin (A + π / 4) = -Real.sqrt 10 / 10 := by
  sorry

end triangle_sine_property_l1116_111637


namespace right_triangle_hypotenuse_l1116_111604

/-- Given a right triangle PQR with legs PQ and PR, points M and N on PQ and PR respectively
    such that PM:MQ = PN:NQ = 1:3, QN = 18, and MR = 30, prove that QR = 8√18 -/
theorem right_triangle_hypotenuse (P Q R M N : ℝ × ℝ) 
  (h_right : (Q.1 - P.1) * (R.1 - P.1) + (Q.2 - P.2) * (R.2 - P.2) = 0)
  (h_M_on_PQ : ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ M = (t * P.1 + (1 - t) * Q.1, t * P.2 + (1 - t) * Q.2))
  (h_N_on_PR : ∃ s : ℝ, 0 ≤ s ∧ s ≤ 1 ∧ N = (s * P.1 + (1 - s) * R.1, s * P.2 + (1 - s) * R.2))
  (h_ratio_M : dist P M = 1/4 * dist P Q)
  (h_ratio_N : dist P N = 1/4 * dist P R)
  (h_QN : dist Q N = 18)
  (h_MR : dist M R = 30) :
  dist Q R = 8 * Real.sqrt 18 := by
  sorry

end right_triangle_hypotenuse_l1116_111604


namespace lemonade_water_amount_solution_is_correct_l1116_111657

/-- Represents the recipe for lemonade --/
structure LemonadeRecipe where
  water : ℝ
  sugar : ℝ
  lemon_juice : ℝ

/-- Checks if the recipe satisfies the given ratios --/
def is_valid_recipe (r : LemonadeRecipe) : Prop :=
  r.water = 5 * r.sugar ∧ r.sugar = 3 * r.lemon_juice

/-- The main theorem: given the ratios and lemon juice amount, prove the water amount --/
theorem lemonade_water_amount (r : LemonadeRecipe) 
  (h1 : is_valid_recipe r) (h2 : r.lemon_juice = 5) : r.water = 75 := by
  sorry

/-- Proof that our solution is correct --/
theorem solution_is_correct : ∃ r : LemonadeRecipe, 
  is_valid_recipe r ∧ r.lemon_juice = 5 ∧ r.water = 75 := by
  sorry

end lemonade_water_amount_solution_is_correct_l1116_111657


namespace simplify_inverse_sum_l1116_111693

theorem simplify_inverse_sum (k x y : ℝ) (hk : k ≠ 0) (hx : x ≠ 0) (hy : y ≠ 0) :
  (k * x⁻¹ + k * y⁻¹)⁻¹ = (x * y) / (k * (x + y)) := by
  sorry

end simplify_inverse_sum_l1116_111693


namespace max_sum_with_reciprocal_constraint_l1116_111608

theorem max_sum_with_reciprocal_constraint (a b : ℝ) 
  (ha : a > 0) (hb : b > 0) (h_eq : 1/a + 9/b = 1) : 
  a + b ≤ 16 := by
sorry

end max_sum_with_reciprocal_constraint_l1116_111608


namespace circles_internally_tangent_l1116_111692

/-- Two circles are internally tangent if the distance between their centers
    is equal to the difference of their radii -/
def internally_tangent (r₁ r₂ d : ℝ) : Prop :=
  d = abs (r₁ - r₂)

/-- Given two circles with radii 5 cm and 3 cm, with centers 2 cm apart,
    prove that they are internally tangent -/
theorem circles_internally_tangent :
  let r₁ : ℝ := 5  -- radius of larger circle
  let r₂ : ℝ := 3  -- radius of smaller circle
  let d  : ℝ := 2  -- distance between centers
  internally_tangent r₁ r₂ d := by
  sorry

end circles_internally_tangent_l1116_111692


namespace inequality_theorem_l1116_111696

theorem inequality_theorem (a b c x y z : ℝ) 
  (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (hx : x ≥ 0) (hy : y ≥ 0) (hz : z ≥ 0)
  (h1 : x + a * b * y ≤ a * (y + z))
  (h2 : y + b * c * z ≤ b * (z + x))
  (h3 : z + c * a * x ≤ c * (x + y)) :
  (x = 0 ∧ y = 0 ∧ z = 0) ∨ (a = 1 ∧ b = 1 ∧ c = 1) :=
by sorry

end inequality_theorem_l1116_111696


namespace sufficient_not_necessary_condition_l1116_111642

/-- Represents an ellipse with foci on the x-axis -/
def is_ellipse_x_axis (m : ℝ) : Prop :=
  m^2 - 1 > 3

/-- The condition m^2 > 5 is sufficient but not necessary for the equation to represent an ellipse with foci on the x-axis -/
theorem sufficient_not_necessary_condition :
  (∀ m : ℝ, m^2 > 5 → is_ellipse_x_axis m) ∧
  (∃ m : ℝ, m^2 ≤ 5 ∧ is_ellipse_x_axis m) :=
sorry

end sufficient_not_necessary_condition_l1116_111642


namespace fibonacci_last_four_zeros_exist_l1116_111617

def fibonacci : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | n + 2 => fibonacci n + fibonacci (n + 1)

def last_four_digits (n : ℕ) : ℕ := n % 10000

theorem fibonacci_last_four_zeros_exist :
  ∃ n, n < 100000001 ∧ last_four_digits (fibonacci n) = 0 := by
sorry

end fibonacci_last_four_zeros_exist_l1116_111617


namespace carries_tshirt_purchase_l1116_111616

/-- The cost of a single t-shirt in dollars -/
def cost_per_shirt : ℝ := 9.95

/-- The number of t-shirts Carrie bought -/
def num_shirts : ℕ := 25

/-- The total cost of Carrie's t-shirt purchase -/
def total_cost : ℝ := cost_per_shirt * (num_shirts : ℝ)

theorem carries_tshirt_purchase :
  total_cost = 248.75 := by sorry

end carries_tshirt_purchase_l1116_111616


namespace find_number_l1116_111649

theorem find_number : ∃! x : ℝ, 7 * x + 21.28 = 50.68 := by
  sorry

end find_number_l1116_111649


namespace complement_union_A_B_range_of_m_when_B_subset_A_l1116_111677

-- Define sets A and B
def A : Set ℝ := {x | -1 ≤ x ∧ x ≤ 2}
def B (m : ℝ) : Set ℝ := {x | m ≤ x ∧ x ≤ m + 1}

-- Theorem 1
theorem complement_union_A_B : 
  (A ∪ B (-2))ᶜ = {x | x < -2 ∨ x > 2} := by sorry

-- Theorem 2
theorem range_of_m_when_B_subset_A : 
  ∀ m : ℝ, B m ⊆ A ↔ -1 ≤ m ∧ m ≤ 1 := by sorry

end complement_union_A_B_range_of_m_when_B_subset_A_l1116_111677


namespace simplify_and_evaluate_l1116_111636

theorem simplify_and_evaluate (x : ℤ) 
  (h1 : -1 ≤ x ∧ x ≤ 1) 
  (h2 : x ≠ 0) 
  (h3 : x ≠ 1) : 
  ((((x^2 - 1) / (x^2 - 2*x + 1) + 1 / (1 - x)) : ℚ) / (x^2 : ℚ) * (x - 1)) = -1 :=
by sorry

end simplify_and_evaluate_l1116_111636


namespace triangle_trig_inequality_l1116_111632

theorem triangle_trig_inequality (A B C : ℝ) (h : A + B + C = π) :
  Real.sin A + Real.cos B * Real.cos C ≤ (1 + Real.sqrt 5) / 2 := by
  sorry

end triangle_trig_inequality_l1116_111632


namespace irreducible_fraction_l1116_111628

theorem irreducible_fraction (n : ℕ+) : 
  (Nat.gcd (3 * n + 1) (5 * n + 2) = 1) := by sorry

end irreducible_fraction_l1116_111628


namespace right_triangle_perimeter_l1116_111624

theorem right_triangle_perimeter (a b c : ℝ) : 
  a > 0 → b > 0 → c > 0 →
  a * b / 2 = 150 →
  a = 30 →
  c^2 = a^2 + b^2 →
  a + b + c = 40 + 10 * Real.sqrt 10 := by
sorry

end right_triangle_perimeter_l1116_111624


namespace emily_subtraction_l1116_111639

theorem emily_subtraction (h : 51^2 = 50^2 + 101) : 50^2 - 49^2 = 99 := by
  sorry

end emily_subtraction_l1116_111639


namespace area_change_possibilities_l1116_111675

/-- Represents the change in area of a rectangle when one side is increased by 3 cm
    and the other is decreased by 3 cm. --/
def areaChange (a b : ℝ) : ℝ := 3 * (a - b - 3)

/-- Theorem stating that the area change can be positive, negative, or zero. --/
theorem area_change_possibilities (a b : ℝ) :
  ∃ (x y z : ℝ), x > 0 ∧ y < 0 ∧ z = 0 ∧
  (areaChange x b = z ∨ areaChange a x = z) ∧
  (areaChange y b > 0 ∨ areaChange a y > 0) ∧
  (areaChange z b < 0 ∨ areaChange a z < 0) := by
  sorry

end area_change_possibilities_l1116_111675


namespace length_of_AB_is_two_l1116_111663

-- Define the points A and B
def A (a : ℝ) : ℝ × ℝ := (3, a + 3)
def B (a : ℝ) : ℝ × ℝ := (a, 4)

-- Define the condition that AB is parallel to the x-axis
def parallel_to_x_axis (a : ℝ) : Prop :=
  (A a).2 = (B a).2

-- Define the length of segment AB
def length_AB (a : ℝ) : ℝ :=
  |((A a).1 - (B a).1)|

-- Theorem statement
theorem length_of_AB_is_two (a : ℝ) :
  parallel_to_x_axis a → length_AB a = 2 := by
  sorry

end length_of_AB_is_two_l1116_111663


namespace extremum_at_two_min_value_of_sum_l1116_111685

/-- The function f(x) = -x³ + ax² - 4 -/
def f (a : ℝ) (x : ℝ) : ℝ := -x^3 + a*x^2 - 4

/-- The derivative of f(x) -/
def f_deriv (a : ℝ) (x : ℝ) : ℝ := -3*x^2 + 2*a*x

theorem extremum_at_two (a : ℝ) : f_deriv a 2 = 0 ↔ a = 3 := by sorry

theorem min_value_of_sum (m n : ℝ) (hm : m ∈ Set.Icc (-1 : ℝ) 1) (hn : n ∈ Set.Icc (-1 : ℝ) 1) :
  ∃ (a : ℝ), f_deriv a 2 = 0 ∧ f a m + f_deriv a n ≥ -13 ∧
  ∃ (m' n' : ℝ), m' ∈ Set.Icc (-1 : ℝ) 1 ∧ n' ∈ Set.Icc (-1 : ℝ) 1 ∧ f a m' + f_deriv a n' = -13 := by sorry

end extremum_at_two_min_value_of_sum_l1116_111685


namespace kamals_math_marks_l1116_111665

def english_marks : ℕ := 66
def physics_marks : ℕ := 77
def chemistry_marks : ℕ := 62
def biology_marks : ℕ := 75
def average_marks : ℚ := 69
def total_subjects : ℕ := 5

theorem kamals_math_marks :
  let total_marks := average_marks * total_subjects
  let known_marks_sum := english_marks + physics_marks + chemistry_marks + biology_marks
  let math_marks := total_marks - known_marks_sum
  math_marks = 65 := by sorry

end kamals_math_marks_l1116_111665


namespace science_book_pages_l1116_111602

/-- Given a history book, novel, and science book, prove that the science book has 600 pages. -/
theorem science_book_pages
  (history_book novel science_book : ℕ) -- Define the books as natural numbers
  (h1 : novel = history_book / 2) -- The novel has half as many pages as the history book
  (h2 : science_book = 4 * novel) -- The science book has 4 times the amount of pages as the novel
  (h3 : history_book = 300) -- The history book has 300 pages
  : science_book = 600 := by
  sorry

end science_book_pages_l1116_111602


namespace joint_purchase_effectiveness_l1116_111679

/-- Represents the benefits of joint purchases -/
structure JointPurchaseBenefits where
  cost_savings : ℝ
  quality_assessment : ℝ
  community_trust : ℝ

/-- Represents the drawbacks of joint purchases -/
structure JointPurchaseDrawbacks where
  transaction_costs : ℝ
  organizational_efforts : ℝ
  convenience_issues : ℝ
  potential_disputes : ℝ

/-- Represents the characteristics of a group making joint purchases -/
structure PurchaseGroup where
  size : ℕ
  is_localized : Bool

/-- Calculates the total benefit of joint purchases for a group -/
def calculate_total_benefit (benefits : JointPurchaseBenefits) (group : PurchaseGroup) : ℝ :=
  benefits.cost_savings + benefits.quality_assessment + benefits.community_trust

/-- Calculates the total drawback of joint purchases for a group -/
def calculate_total_drawback (drawbacks : JointPurchaseDrawbacks) (group : PurchaseGroup) : ℝ :=
  drawbacks.transaction_costs + drawbacks.organizational_efforts + drawbacks.convenience_issues + drawbacks.potential_disputes

/-- Theorem stating that joint purchases are beneficial for large groups but not for small, localized groups -/
theorem joint_purchase_effectiveness (benefits : JointPurchaseBenefits) (drawbacks : JointPurchaseDrawbacks) :
  ∀ (group : PurchaseGroup),
    (group.size > 100 → calculate_total_benefit benefits group > calculate_total_drawback drawbacks group) ∧
    (group.size ≤ 100 ∧ group.is_localized → calculate_total_benefit benefits group ≤ calculate_total_drawback drawbacks group) :=
by sorry

end joint_purchase_effectiveness_l1116_111679


namespace quadratic_property_l1116_111680

/-- A quadratic function with a real coefficient b -/
def f (b : ℝ) (x : ℝ) : ℝ := x^2 + b*x + 3

/-- The range of f is [0, +∞) -/
def has_nonnegative_range (b : ℝ) : Prop :=
  ∀ y, (∃ x, f b x = y) → y ≥ 0

/-- The solution set of f(x) < c is an open interval of length 8 -/
def has_solution_interval_of_length_eight (b c : ℝ) : Prop :=
  ∃ m, ∀ x, f b x < c ↔ m - 8 < x ∧ x < m

theorem quadratic_property (b : ℝ) (h1 : has_nonnegative_range b) 
  (h2 : has_solution_interval_of_length_eight b c) : c = 16 :=
sorry

end quadratic_property_l1116_111680


namespace unique_root_of_R_l1116_111690

/-- Represents a quadratic trinomial ax^2 + bx + c -/
structure QuadraticTrinomial where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Given a quadratic trinomial P, construct R by adding P to the trinomial formed by swapping P's a and c -/
def constructR (P : QuadraticTrinomial) : QuadraticTrinomial :=
  { a := P.a + P.c
  , b := 2 * P.b
  , c := P.a + P.c }

theorem unique_root_of_R (P : QuadraticTrinomial) :
  let R := constructR P
  (∃! x : ℝ, R.a * x^2 + R.b * x + R.c = 0) →
  (∃ x : ℝ, x = -2 ∨ x = 2 ∧ R.a * x^2 + R.b * x + R.c = 0) :=
by sorry

end unique_root_of_R_l1116_111690


namespace fraction_operation_result_l1116_111600

theorem fraction_operation_result (x : ℝ) : 
  x = 2.5 → ((x / (1 / 2)) * x) / ((x * (1 / 2)) / x) = 25 := by
  sorry

end fraction_operation_result_l1116_111600


namespace difference_of_squares_l1116_111623

theorem difference_of_squares (x : ℝ) : x^2 - 1 = (x + 1) * (x - 1) := by
  sorry

end difference_of_squares_l1116_111623


namespace arithmetic_sequence_sum_l1116_111630

/-- Given an arithmetic sequence, prove that if the sum of the first n terms is 54
    and the sum of the first 2n terms is 72, then the sum of the first 3n terms is 78. -/
theorem arithmetic_sequence_sum (n : ℕ) (S : ℕ → ℝ) : 
  S n = 54 → S (2*n) = 72 → S (3*n) = 78 := by sorry

end arithmetic_sequence_sum_l1116_111630


namespace quadratic_inequality_solution_set_l1116_111660

theorem quadratic_inequality_solution_set :
  {x : ℝ | x^2 + x - 6 < 0} = Set.Ioo (-3 : ℝ) 2 := by
  sorry

end quadratic_inequality_solution_set_l1116_111660


namespace chess_group_games_l1116_111654

/-- Represents a chess group with alternating even-odd opponent play --/
structure ChessGroup where
  total_players : ℕ
  even_players : ℕ
  odd_players : ℕ
  alternating_play : Bool

/-- Calculates the total number of games played in the chess group --/
def total_games (cg : ChessGroup) : ℕ :=
  (cg.total_players * cg.even_players) / 2

/-- Theorem stating the total number of games played in a specific chess group setup --/
theorem chess_group_games :
  ∀ (cg : ChessGroup),
    cg.total_players = 12 ∧
    cg.even_players = 6 ∧
    cg.odd_players = 6 ∧
    cg.alternating_play = true →
    total_games cg = 36 := by
  sorry

end chess_group_games_l1116_111654


namespace quadratic_polynomial_with_complex_root_l1116_111622

theorem quadratic_polynomial_with_complex_root :
  ∃ (a b c : ℝ), 
    (∀ x : ℂ, (3 : ℂ) * x^2 + (a : ℂ) * x + (b : ℂ) = 0 ↔ x = 4 + 2*I ∨ x = 4 - 2*I) ∧
    c = 3 ∧
    a = -24 ∧
    b = 60 :=
sorry

end quadratic_polynomial_with_complex_root_l1116_111622


namespace parabola_focus_distance_l1116_111681

/-- Theorem: For a parabola y² = ax (a > 0) with a point P(3/2, y₀) on it,
    if the distance from P to the focus is 2, then a = 2. -/
theorem parabola_focus_distance (a : ℝ) (y₀ : ℝ) :
  a > 0 →
  y₀^2 = a * (3/2) →
  2 = (|3/2 - a/4| + |y₀|) →
  a = 2 := by
  sorry

end parabola_focus_distance_l1116_111681


namespace parallelogram_intersection_ratio_l1116_111627

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- Represents a parallelogram -/
structure Parallelogram :=
  (W : Point) (X : Point) (Y : Point) (Z : Point)

/-- The theorem statement -/
theorem parallelogram_intersection_ratio 
  (WXYZ : Parallelogram) 
  (M : Point) (N : Point) (P : Point) :
  (∃ t : ℝ, t ∈ Set.Icc 0 1 ∧ M = Point.mk ((1 - t) * WXYZ.W.x + t * WXYZ.Z.x) ((1 - t) * WXYZ.W.y + t * WXYZ.Z.y) ∧ t = 3/100) →
  (∃ s : ℝ, s ∈ Set.Icc 0 1 ∧ N = Point.mk ((1 - s) * WXYZ.W.x + s * WXYZ.Y.x) ((1 - s) * WXYZ.W.y + s * WXYZ.Y.y) ∧ s = 3/251) →
  (∃ r : ℝ, r ∈ Set.Icc 0 1 ∧ P = Point.mk ((1 - r) * WXYZ.W.x + r * WXYZ.Y.x) ((1 - r) * WXYZ.W.y + r * WXYZ.Y.y)) →
  (∃ q : ℝ, q ∈ Set.Icc 0 1 ∧ P = Point.mk ((1 - q) * M.x + q * N.x) ((1 - q) * M.y + q * N.y)) →
  (WXYZ.Y.x - WXYZ.W.x) / (P.x - WXYZ.W.x) = 2 ∧ (WXYZ.Y.y - WXYZ.W.y) / (P.y - WXYZ.W.y) = 2 :=
by sorry

end parallelogram_intersection_ratio_l1116_111627


namespace robert_books_read_l1116_111605

def reading_speed : ℕ := 120
def book_length : ℕ := 360
def available_time : ℕ := 8

def books_read (speed pages time : ℕ) : ℕ :=
  (speed * time) / pages

theorem robert_books_read :
  books_read reading_speed book_length available_time = 2 := by
  sorry

end robert_books_read_l1116_111605


namespace students_in_chemistry_or_physics_not_both_l1116_111664

theorem students_in_chemistry_or_physics_not_both (total_chemistry : ℕ) (both : ℕ) (only_physics : ℕ)
  (h1 : both = 15)
  (h2 : total_chemistry = 30)
  (h3 : only_physics = 12) :
  total_chemistry - both + only_physics = 27 :=
by sorry

end students_in_chemistry_or_physics_not_both_l1116_111664


namespace rectangular_solid_diagonal_l1116_111648

theorem rectangular_solid_diagonal (x y z : ℝ) : 
  (2 * (x * y + y * z + z * x) = 62) →
  (4 * (x + y + z) = 48) →
  (x^2 + y^2 + z^2 : ℝ) = 82 :=
by sorry

end rectangular_solid_diagonal_l1116_111648


namespace farm_heads_count_l1116_111606

/-- Represents a farm with hens and cows -/
structure Farm where
  hens : ℕ
  cows : ℕ

/-- The total number of feet on the farm -/
def totalFeet (f : Farm) : ℕ := 2 * f.hens + 4 * f.cows

/-- The total number of heads (animals) on the farm -/
def totalHeads (f : Farm) : ℕ := f.hens + f.cows

/-- Theorem: Given a farm with 24 hens and 144 total feet, the total number of heads is 48 -/
theorem farm_heads_count (f : Farm) 
  (hen_count : f.hens = 24) 
  (feet_count : totalFeet f = 144) : 
  totalHeads f = 48 := by
  sorry


end farm_heads_count_l1116_111606


namespace andy_cookies_count_l1116_111621

/-- The number of cookies Andy ate -/
def andy_ate : Nat := 3

/-- The number of cookies Andy gave to his brother -/
def brother_cookies : Nat := 5

/-- The number of players in Andy's basketball team -/
def team_size : Nat := 8

/-- The sequence of cookies taken by each team member -/
def team_sequence (n : Nat) : Nat := 2 * n - 1

/-- The total number of cookies Andy had at the start -/
def total_cookies : Nat := andy_ate + brother_cookies + (Finset.sum (Finset.range team_size) team_sequence)

theorem andy_cookies_count : total_cookies = 72 := by sorry

end andy_cookies_count_l1116_111621


namespace chicken_bucket_price_l1116_111650

/-- Represents the price of a chicken bucket with sides -/
def bucket_price (people_per_bucket : ℕ) (total_people : ℕ) (total_cost : ℕ) : ℚ :=
  total_cost / (total_people / people_per_bucket)

/-- Proves that the price of each chicken bucket with sides is $12 -/
theorem chicken_bucket_price :
  bucket_price 6 36 72 = 12 := by
  sorry

end chicken_bucket_price_l1116_111650


namespace linear_function_increasing_l1116_111643

/-- A linear function y = mx + b where m = k - 2 and b = 3 -/
def linearFunction (k : ℝ) (x : ℝ) : ℝ := (k - 2) * x + 3

/-- The property that y increases as x increases -/
def increasingFunction (f : ℝ → ℝ) : Prop :=
  ∀ x₁ x₂ : ℝ, x₁ < x₂ → f x₁ < f x₂

theorem linear_function_increasing (k : ℝ) :
  increasingFunction (linearFunction k) → k > 2 := by
  sorry

end linear_function_increasing_l1116_111643


namespace candy_distribution_l1116_111662

theorem candy_distribution (total : Nat) (friends : Nat) (h1 : total = 17) (h2 : friends = 5) :
  total % friends = 2 := by
  sorry

end candy_distribution_l1116_111662


namespace smallest_n_for_324_l1116_111618

/-- A geometric sequence (b_n) with given first three terms -/
def geometric_sequence (b : ℕ → ℝ) : Prop :=
  b 1 = 2 ∧ b 2 = 6 ∧ b 3 = 18 ∧ ∀ n : ℕ, n ≥ 1 → b (n + 1) / b n = b 2 / b 1

/-- The smallest n for which b_n = 324 in the given geometric sequence is 5 -/
theorem smallest_n_for_324 (b : ℕ → ℝ) (h : geometric_sequence b) :
  (∃ n : ℕ, b n = 324) ∧ (∀ m : ℕ, b m = 324 → m ≥ 5) :=
sorry

end smallest_n_for_324_l1116_111618


namespace complex_magnitude_equation_l1116_111603

theorem complex_magnitude_equation (n : ℝ) : 
  n > 0 ∧ Complex.abs (3 + n * Complex.I) = 3 * Real.sqrt 10 → n = 9 := by
  sorry

end complex_magnitude_equation_l1116_111603


namespace more_diamonds_than_rubies_l1116_111620

theorem more_diamonds_than_rubies (diamonds : ℕ) (rubies : ℕ) 
  (h1 : diamonds = 421) (h2 : rubies = 377) : 
  diamonds - rubies = 44 := by sorry

end more_diamonds_than_rubies_l1116_111620


namespace max_silver_tokens_l1116_111615

/-- Represents the state of tokens --/
structure TokenState :=
  (red : ℕ)
  (blue : ℕ)
  (silver : ℕ)

/-- Represents an exchange at a booth --/
inductive Exchange
  | RedToSilver
  | BlueToSilver

/-- Applies an exchange to the current state --/
def applyExchange (state : TokenState) (ex : Exchange) : TokenState :=
  match ex with
  | Exchange.RedToSilver => 
      if state.red ≥ 3 then
        TokenState.mk (state.red - 3) (state.blue + 2) (state.silver + 1)
      else
        state
  | Exchange.BlueToSilver => 
      if state.blue ≥ 4 then
        TokenState.mk (state.red + 2) (state.blue - 4) (state.silver + 1)
      else
        state

/-- Checks if any exchange is possible --/
def canExchange (state : TokenState) : Bool :=
  state.red ≥ 3 ∨ state.blue ≥ 4

/-- Theorem: The maximum number of silver tokens Alex can obtain is 131 --/
theorem max_silver_tokens : 
  ∃ (exchanges : List Exchange), 
    let finalState := exchanges.foldl applyExchange (TokenState.mk 100 100 0)
    ¬(canExchange finalState) ∧ finalState.silver = 131 := by
  sorry

end max_silver_tokens_l1116_111615


namespace inverse_sum_mod_31_l1116_111610

theorem inverse_sum_mod_31 : ∃ (a b : ℤ), (5 * a) % 31 = 1 ∧ (5^2 * b) % 31 = 1 ∧ (a + b) % 31 = 26 := by
  sorry

end inverse_sum_mod_31_l1116_111610


namespace cube_root_21600_l1116_111672

theorem cube_root_21600 : ∃ (a b : ℕ), a > 0 ∧ b > 0 ∧ (∀ (a' b' : ℕ), a' > 0 → b' > 0 → a'^3 * b' = 21600 → b ≤ b') ∧ a^3 * b = 21600 ∧ a + b = 106 := by
  sorry

end cube_root_21600_l1116_111672


namespace problem_statement_l1116_111607

theorem problem_statement (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = a * b) :
  ((a - 1) * (b - 1) = 1) ∧
  (∀ a b, a > 0 → b > 0 → a + b = a * b → a + 4 * b ≥ 9) ∧
  (∀ a b, a > 0 → b > 0 → a + b = a * b → 1 / a^2 + 2 / b^2 ≥ 2 / 3) :=
by sorry

end problem_statement_l1116_111607


namespace unique_quadratic_solution_l1116_111635

/-- Given a quadratic equation ax^2 + 16x + c = 0 with exactly one solution,
    where a + c = 25 and a < c, prove that a = 3 and c = 22 -/
theorem unique_quadratic_solution (a c : ℝ) : 
  (∃! x, a * x^2 + 16 * x + c = 0) →  -- exactly one solution
  (a + c = 25) →                      -- sum condition
  (a < c) →                           -- order condition
  (a = 3 ∧ c = 22) :=                 -- conclusion
by sorry

end unique_quadratic_solution_l1116_111635


namespace increase_in_average_age_l1116_111694

/-- Calculates the increase in average age when two men in a group are replaced -/
theorem increase_in_average_age
  (n : ℕ) -- Total number of men
  (age1 age2 : ℕ) -- Ages of the two men being replaced
  (new_avg : ℚ) -- Average age of the two new men
  (h1 : n = 15)
  (h2 : age1 = 21)
  (h3 : age2 = 23)
  (h4 : new_avg = 37) :
  (2 * new_avg - (age1 + age2 : ℚ)) / n = 2 := by sorry

end increase_in_average_age_l1116_111694


namespace geometric_progression_first_term_l1116_111682

theorem geometric_progression_first_term (S a r : ℝ) : 
  S = 10 → 
  a + a * r = 6 → 
  a = 2 * r → 
  (a = -1 + Real.sqrt 13 ∨ a = -1 - Real.sqrt 13) :=
by sorry

end geometric_progression_first_term_l1116_111682


namespace largest_non_representable_l1116_111689

def is_representable (n : ℕ) : Prop :=
  ∃ (a b c : ℕ), n = 6 * a + 9 * b + 20 * c

theorem largest_non_representable : 
  (∀ m : ℕ, m > 43 → is_representable m) ∧ 
  ¬(is_representable 43) := by sorry

end largest_non_representable_l1116_111689


namespace composite_fraction_theorem_l1116_111641

def first_eight_composites : List Nat := [4, 6, 8, 9, 10, 12, 14, 15]
def next_eight_composites : List Nat := [16, 18, 20, 21, 22, 24, 25, 26]
def first_prime : Nat := 2
def second_prime : Nat := 3

theorem composite_fraction_theorem :
  let numerator := (List.prod first_eight_composites + first_prime)
  let denominator := (List.prod next_eight_composites + second_prime)
  (numerator : ℚ) / denominator = 
    (4 * 6 * 8 * 9 * 10 * 12 * 14 * 15 + 2 : ℚ) / 
    (16 * 18 * 20 * 21 * 22 * 24 * 25 * 26 + 3) := by
  sorry

end composite_fraction_theorem_l1116_111641


namespace correct_quotient_l1116_111644

theorem correct_quotient (N : ℕ) (h1 : N % 21 = 0) (h2 : N / 12 = 63) : N / 21 = 36 := by
  sorry

end correct_quotient_l1116_111644


namespace det_A_l1116_111684

/-- The matrix A as described in the problem -/
def A (n : ℕ) : Matrix (Fin n) (Fin n) ℚ :=
  λ i j => 1 / (min i.val j.val + 1 : ℚ)

/-- The theorem stating the determinant of matrix A -/
theorem det_A (n : ℕ) : 
  Matrix.det (A n) = (-1 : ℚ)^(n-1) / ((Nat.factorial (n-1)) * (Nat.factorial n)) := by
  sorry

end det_A_l1116_111684


namespace gumdrop_replacement_l1116_111688

theorem gumdrop_replacement (blue_percent : Real) (brown_percent : Real) 
  (red_percent : Real) (yellow_percent : Real) (green_count : Nat) :
  blue_percent = 0.3 →
  brown_percent = 0.2 →
  red_percent = 0.15 →
  yellow_percent = 0.1 →
  green_count = 30 →
  let total := green_count / (1 - (blue_percent + brown_percent + red_percent + yellow_percent))
  let blue_count := blue_percent * total
  let brown_count := brown_percent * total
  let new_brown_count := brown_count + blue_count / 2
  new_brown_count = 42 := by
  sorry

end gumdrop_replacement_l1116_111688


namespace perfect_square_digit_sum_l1116_111668

def sum_of_digits (n : ℕ) : ℕ := sorry

theorem perfect_square_digit_sum :
  (¬ ∃ n : ℕ, ∃ m : ℕ, n = m^2 ∧ sum_of_digits n = 20) ∧
  (∃ n : ℕ, ∃ m : ℕ, n = m^2 ∧ sum_of_digits n = 10) := by sorry

end perfect_square_digit_sum_l1116_111668
