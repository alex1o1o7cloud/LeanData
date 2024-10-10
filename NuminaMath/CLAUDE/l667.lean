import Mathlib

namespace phones_sold_is_four_l667_66742

/-- Calculates the total number of cell phones sold given the initial and final counts
    of Samsung phones and iPhones, as well as the number of damaged/defective phones. -/
def total_phones_sold (initial_samsung : ℕ) (final_samsung : ℕ) (initial_iphone : ℕ) 
                      (final_iphone : ℕ) (damaged_samsung : ℕ) (defective_iphone : ℕ) : ℕ :=
  (initial_samsung - final_samsung - damaged_samsung) + 
  (initial_iphone - final_iphone - defective_iphone)

/-- Theorem stating that the total number of cell phones sold is 4 given the specific
    initial and final counts, and the number of damaged/defective phones. -/
theorem phones_sold_is_four : 
  total_phones_sold 14 10 8 5 2 1 = 4 := by
  sorry

end phones_sold_is_four_l667_66742


namespace one_fourth_of_six_times_eight_l667_66702

theorem one_fourth_of_six_times_eight : (1 / 4 : ℚ) * (6 * 8) = 12 := by
  sorry

end one_fourth_of_six_times_eight_l667_66702


namespace tangent_length_is_six_l667_66760

/-- A circle passing through three points -/
structure Circle3Points where
  p1 : ℝ × ℝ
  p2 : ℝ × ℝ
  p3 : ℝ × ℝ

/-- The length of the tangent from a point to a circle -/
def tangentLength (origin : ℝ × ℝ) (circle : Circle3Points) : ℝ :=
  sorry

/-- Theorem: The length of the tangent from the origin to the specific circle is 6 -/
theorem tangent_length_is_six : 
  let origin : ℝ × ℝ := (0, 0)
  let circle : Circle3Points := { 
    p1 := (2, 3),
    p2 := (4, 6),
    p3 := (6, 15)
  }
  tangentLength origin circle = 6 := by
  sorry

end tangent_length_is_six_l667_66760


namespace ellipse_b_plus_k_l667_66732

/-- Definition of an ellipse with given foci and a point on the curve -/
def Ellipse (f1 f2 p : ℝ × ℝ) :=
  ∃ (a b h k : ℝ), a > 0 ∧ b > 0 ∧
    (p.1 - h)^2 / a^2 + (p.2 - k)^2 / b^2 = 1 ∧
    (f1.1 - h)^2 / a^2 + (f1.2 - k)^2 / b^2 = 1 ∧
    (f2.1 - h)^2 / a^2 + (f2.2 - k)^2 / b^2 = 1

/-- Theorem stating the sum of b and k for the given ellipse -/
theorem ellipse_b_plus_k :
  ∀ (a b h k : ℝ),
    Ellipse (2, 3) (2, 7) (6, 5) →
    a > 0 →
    b > 0 →
    (6 - h)^2 / a^2 + (5 - k)^2 / b^2 = 1 →
    b + k = 4 * Real.sqrt 5 + 5 := by
  sorry

end ellipse_b_plus_k_l667_66732


namespace f_decreasing_implies_a_range_l667_66787

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then (3 * a - 1) * x + 4 * a else Real.log x / Real.log a

theorem f_decreasing_implies_a_range (a : ℝ) :
  (∀ x₁ x₂ : ℝ, x₁ < x₂ → f a x₁ > f a x₂) →
  (a ≥ 1/7 ∧ a < 1/3) :=
sorry

end f_decreasing_implies_a_range_l667_66787


namespace race_distance_l667_66734

theorem race_distance (total_length : Real) (part1 : Real) (part2 : Real) (part3 : Real)
  (h1 : total_length = 74.5)
  (h2 : part1 = 15.5)
  (h3 : part2 = 21.5)
  (h4 : part3 = 21.5) :
  total_length - (part1 + part2 + part3) = 16 := by
  sorry

end race_distance_l667_66734


namespace equation_roots_l667_66796

theorem equation_roots : 
  {x : ℝ | Real.sqrt (x^2) + 3 * x⁻¹ = 4} = {3, -3, 1, -1} :=
by sorry

end equation_roots_l667_66796


namespace divisors_of_power_difference_l667_66749

theorem divisors_of_power_difference (n : ℕ) :
  n = 11^60 - 17^24 →
  ∃ (d : ℕ), d ≥ 120 ∧ (∀ (x : ℕ), x ∣ n → x > 0 → x ≤ d) :=
by sorry

end divisors_of_power_difference_l667_66749


namespace point_in_intersection_l667_66748

def U : Set (ℝ × ℝ) := Set.univ

def A (m : ℝ) : Set (ℝ × ℝ) := U \ {p : ℝ × ℝ | p.1 + p.2 > m}

def B (n : ℝ) : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1^2 + p.2^2 ≤ n}

def C_U (S : Set (ℝ × ℝ)) : Set (ℝ × ℝ) := U \ S

theorem point_in_intersection (m n : ℝ) :
  (1, 2) ∈ (C_U (A m) ∩ B n) ↔ m ≥ 3 ∧ n ≥ 5 := by sorry

end point_in_intersection_l667_66748


namespace janes_blouses_l667_66755

theorem janes_blouses (skirt_price : ℕ) (blouse_price : ℕ) (num_skirts : ℕ) (total_paid : ℕ) (change : ℕ) : 
  skirt_price = 13 →
  blouse_price = 6 →
  num_skirts = 2 →
  total_paid = 100 →
  change = 56 →
  (total_paid - change - (num_skirts * skirt_price)) / blouse_price = 3 :=
by sorry

end janes_blouses_l667_66755


namespace find_m_l667_66704

def factorial (n : ℕ) : ℕ := Nat.factorial n

theorem find_m : ∃ m : ℕ, m * factorial m + 2 * factorial m = 5040 ∧ m = 5 := by
  sorry

end find_m_l667_66704


namespace z_sixth_power_l667_66729

theorem z_sixth_power (z : ℂ) : z = (-Real.sqrt 5 + Complex.I) / 2 → z^6 = -1 := by
  sorry

end z_sixth_power_l667_66729


namespace cube_surface_area_l667_66775

/-- Given a cube with volume x^3, its surface area is 6x^2 -/
theorem cube_surface_area (x : ℝ) (h : x > 0) :
  (6 : ℝ) * x^2 = 6 * (x^3)^((2:ℝ)/3) := by
  sorry

end cube_surface_area_l667_66775


namespace fish_sample_properties_l667_66761

/-- Represents the mass categories of fish -/
inductive MassCategory
  | Mass1 : MassCategory
  | Mass2 : MassCategory
  | Mass3 : MassCategory
  | Mass4 : MassCategory

/-- Maps mass categories to their actual mass values -/
def massValue : MassCategory → Float
  | MassCategory.Mass1 => 1.0
  | MassCategory.Mass2 => 1.2
  | MassCategory.Mass3 => 1.5
  | MassCategory.Mass4 => 1.8

/-- Represents the frequency of each mass category -/
def frequency : MassCategory → Nat
  | MassCategory.Mass1 => 4
  | MassCategory.Mass2 => 5
  | MassCategory.Mass3 => 8
  | MassCategory.Mass4 => 3

/-- The total number of fish in the sample -/
def sampleSize : Nat := 20

/-- The number of marked fish recaptured -/
def markedRecaptured : Nat := 2

/-- The total number of fish recaptured -/
def totalRecaptured : Nat := 100

/-- Theorem stating the properties of the fish sample -/
theorem fish_sample_properties :
  (∃ median : Float, median = 1.5) ∧
  (∃ mean : Float, mean = 1.37) ∧
  (∃ totalMass : Float, totalMass = 1370) := by
  sorry

end fish_sample_properties_l667_66761


namespace train_crossing_time_l667_66789

/-- Proves that the time taken for a train to cross a bridge is 20 seconds -/
theorem train_crossing_time (bridge_length : ℝ) (train_length : ℝ) (train_speed : ℝ) :
  bridge_length = 180 →
  train_length = 120 →
  train_speed = 15 →
  (bridge_length + train_length) / train_speed = 20 :=
by sorry

end train_crossing_time_l667_66789


namespace fourth_grid_shaded_fraction_initial_shaded_squares_shaded_squares_arithmetic_l667_66785

/-- Represents the number of shaded squares in the nth grid -/
def shaded_squares (n : ℕ) : ℕ := 2 * n - 1

/-- Represents the total number of squares in the nth grid -/
def total_squares (n : ℕ) : ℕ := n ^ 2

/-- The main theorem stating the fraction of shaded squares in the fourth grid -/
theorem fourth_grid_shaded_fraction :
  (shaded_squares 4 : ℚ) / (total_squares 4 : ℚ) = 7 / 16 := by
  sorry

/-- Verifies that the first three grids have 1, 3, and 5 shaded squares respectively -/
theorem initial_shaded_squares :
  shaded_squares 1 = 1 ∧ shaded_squares 2 = 3 ∧ shaded_squares 3 = 5 := by
  sorry

/-- Verifies that the sequence of shaded squares is arithmetic -/
theorem shaded_squares_arithmetic :
  ∀ n : ℕ, shaded_squares (n + 1) - shaded_squares n = 
           shaded_squares (n + 2) - shaded_squares (n + 1) := by
  sorry

end fourth_grid_shaded_fraction_initial_shaded_squares_shaded_squares_arithmetic_l667_66785


namespace quadrilateral_perimeter_l667_66741

-- Define the quadrilateral ABCD
structure Quadrilateral :=
  (A B C D : ℝ × ℝ)

-- Define the perimeter function
def perimeter (q : Quadrilateral) : ℝ := sorry

-- Define the perpendicular function
def perpendicular (v1 v2 : ℝ × ℝ) : Prop := sorry

-- Define the distance function
def distance (p1 p2 : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem quadrilateral_perimeter 
  (ABCD : Quadrilateral)
  (perp_AB_BC : perpendicular (ABCD.B - ABCD.A) (ABCD.C - ABCD.B))
  (perp_DC_BC : perpendicular (ABCD.C - ABCD.D) (ABCD.C - ABCD.B))
  (AB_length : distance ABCD.A ABCD.B = 15)
  (DC_length : distance ABCD.D ABCD.C = 6)
  (BC_length : distance ABCD.B ABCD.C = 10)
  (AB_eq_AD : distance ABCD.A ABCD.B = distance ABCD.A ABCD.D) :
  perimeter ABCD = 31 + Real.sqrt 181 := by
  sorry

end quadrilateral_perimeter_l667_66741


namespace investment_growth_l667_66709

/-- The annual interest rate as a decimal -/
def interest_rate : ℝ := 0.10

/-- The number of years the investment grows -/
def years : ℕ := 4

/-- The initial investment amount -/
def initial_investment : ℝ := 300

/-- The final value after compounding -/
def final_value : ℝ := 439.23

/-- Theorem stating that the initial investment grows to the final value 
    when compounded annually at the given interest rate for the specified number of years -/
theorem investment_growth :
  initial_investment * (1 + interest_rate) ^ years = final_value := by
  sorry

end investment_growth_l667_66709


namespace quadratic_common_root_theorem_l667_66700

theorem quadratic_common_root_theorem (a b : ℕ+) :
  (∃ x : ℝ, (a - 1 : ℝ) * x^2 - (a^2 + 2 : ℝ) * x + (a^2 + 2*a : ℝ) = 0 ∧
             (b - 1 : ℝ) * x^2 - (b^2 + 2 : ℝ) * x + (b^2 + 2*b : ℝ) = 0) →
  (a^(b : ℕ) + b^(a : ℕ) : ℝ) / (a^(-(b : ℤ)) + b^(-(a : ℤ)) : ℝ) = 256 :=
by sorry

end quadratic_common_root_theorem_l667_66700


namespace shortest_chord_equation_l667_66774

/-- Circle C -/
def circle_C (x y : ℝ) : Prop := x^2 + y^2 - 2*x - 24 = 0

/-- Line l -/
def line_l (x y k : ℝ) : Prop := y = k*(x - 2) - 1

/-- Line AB -/
def line_AB (x y : ℝ) : Prop := x - y - 3 = 0

/-- The theorem statement -/
theorem shortest_chord_equation (k : ℝ) :
  (∃ A B : ℝ × ℝ, 
    (circle_C A.1 A.2 ∧ circle_C B.1 B.2) ∧ 
    (line_l A.1 A.2 k ∧ line_l B.1 B.2 k) ∧
    (∀ P Q : ℝ × ℝ, circle_C P.1 P.2 ∧ circle_C Q.1 Q.2 ∧ 
      line_l P.1 P.2 k ∧ line_l Q.1 Q.2 k →
      (A.1 - B.1)^2 + (A.2 - B.2)^2 ≤ (P.1 - Q.1)^2 + (P.2 - Q.2)^2)) →
  (∀ x y : ℝ, line_AB x y ↔ (circle_C x y ∧ line_l x y k)) :=
sorry

end shortest_chord_equation_l667_66774


namespace trig_identity_l667_66799

theorem trig_identity (α : ℝ) : 
  -Real.sin α + Real.sqrt 3 * Real.cos α = 2 * Real.sin (α + 2 * Real.pi / 3) :=
by sorry

end trig_identity_l667_66799


namespace milk_volume_calculation_l667_66772

def milk_volumes : List ℝ := [2.35, 1.75, 0.9, 0.75, 0.5, 0.325, 0.25]

theorem milk_volume_calculation :
  let total_volume := milk_volumes.sum
  let average_volume := total_volume / milk_volumes.length
  total_volume = 6.825 ∧ average_volume = 0.975 := by sorry

end milk_volume_calculation_l667_66772


namespace unique_intersection_point_l667_66763

-- Define the function f
def f (x : ℝ) : ℝ := x^3 - 9*x^2 + 27*x - 14

-- State the theorem
theorem unique_intersection_point :
  ∃! p : ℝ × ℝ, 
    (∀ x, f x = (p.2) ↔ x = p.1) ∧ 
    p.1 = p.2 ∧ 
    p = (2, 2) := by
  sorry

end unique_intersection_point_l667_66763


namespace max_rectangular_pen_area_l667_66721

/-- Given 50 feet of fencing with 5 feet used for a non-enclosing gate,
    the maximum area of a rectangular pen enclosed by the remaining fencing
    is 126.5625 square feet. -/
theorem max_rectangular_pen_area : 
  ∀ (width height : ℝ),
    width > 0 → height > 0 →
    width + height = (50 - 5) / 2 →
    width * height ≤ 126.5625 :=
by sorry

end max_rectangular_pen_area_l667_66721


namespace king_probability_l667_66782

/-- Custom deck of cards -/
structure Deck :=
  (cards : Nat)
  (suits : Nat)
  (ranks : Nat)
  (one_card_per_rank_suit : cards = suits * ranks)

/-- Probability of drawing a specific rank -/
def prob_draw_rank (d : Deck) (rank_count : Nat) : ℚ :=
  rank_count / d.cards

theorem king_probability (d : Deck) (h1 : d.cards = 65) (h2 : d.suits = 5) (h3 : d.ranks = 13) :
  prob_draw_rank d d.suits = 1 / 13 := by
  sorry

end king_probability_l667_66782


namespace rachel_total_steps_l667_66713

/-- Represents a landmark with its stair information -/
structure Landmark where
  name : String
  flightsUp : Nat
  flightsDown : Nat
  stepsPerFlight : Nat

/-- Calculates the total steps for a single landmark -/
def stepsForLandmark (l : Landmark) : Nat :=
  (l.flightsUp + l.flightsDown) * l.stepsPerFlight

/-- The list of landmarks Rachel visited -/
def landmarks : List Landmark := [
  { name := "Eiffel Tower", flightsUp := 347, flightsDown := 216, stepsPerFlight := 10 },
  { name := "Notre-Dame Cathedral", flightsUp := 178, flightsDown := 165, stepsPerFlight := 12 },
  { name := "Leaning Tower of Pisa", flightsUp := 294, flightsDown := 172, stepsPerFlight := 8 },
  { name := "Colosseum", flightsUp := 122, flightsDown := 93, stepsPerFlight := 15 },
  { name := "Sagrada Familia", flightsUp := 267, flightsDown := 251, stepsPerFlight := 11 },
  { name := "Park Güell", flightsUp := 134, flightsDown := 104, stepsPerFlight := 9 }
]

/-- Calculates the total steps for all landmarks -/
def totalSteps : Nat :=
  landmarks.map stepsForLandmark |>.sum

theorem rachel_total_steps :
  totalSteps = 24539 := by
  sorry

end rachel_total_steps_l667_66713


namespace carrot_cost_correct_l667_66723

/-- Represents the cost of carrots for all students -/
def carrot_cost : ℚ := 185

/-- Represents the number of third grade classes -/
def third_grade_classes : ℕ := 5

/-- Represents the number of students in each third grade class -/
def third_grade_students_per_class : ℕ := 30

/-- Represents the number of fourth grade classes -/
def fourth_grade_classes : ℕ := 4

/-- Represents the number of students in each fourth grade class -/
def fourth_grade_students_per_class : ℕ := 28

/-- Represents the number of fifth grade classes -/
def fifth_grade_classes : ℕ := 4

/-- Represents the number of students in each fifth grade class -/
def fifth_grade_students_per_class : ℕ := 27

/-- Represents the cost of a hamburger -/
def hamburger_cost : ℚ := 21/10

/-- Represents the cost of a cookie -/
def cookie_cost : ℚ := 1/5

/-- Represents the total cost of lunch for all students -/
def total_lunch_cost : ℚ := 1036

/-- Theorem stating that the cost of carrots is correct given the conditions -/
theorem carrot_cost_correct : 
  let total_students := third_grade_classes * third_grade_students_per_class + 
                        fourth_grade_classes * fourth_grade_students_per_class + 
                        fifth_grade_classes * fifth_grade_students_per_class
  total_lunch_cost = total_students * (hamburger_cost + cookie_cost) + carrot_cost :=
by sorry

end carrot_cost_correct_l667_66723


namespace lakers_win_probability_l667_66766

/-- The probability of a team winning a single game in the NBA finals -/
def win_prob : ℚ := 1/4

/-- The number of wins needed to win the NBA finals -/
def wins_needed : ℕ := 4

/-- The total number of games in a 7-game series -/
def total_games : ℕ := 7

/-- The probability of the Lakers winning the NBA finals in exactly 7 games -/
def lakers_win_in_seven : ℚ := 135/4096

theorem lakers_win_probability :
  lakers_win_in_seven = (Nat.choose 6 3 : ℚ) * win_prob^3 * (1 - win_prob)^3 * win_prob :=
by sorry

end lakers_win_probability_l667_66766


namespace camel_height_is_28_feet_l667_66705

/-- The height of a hare in inches -/
def hare_height : ℕ := 14

/-- The factor by which a camel is taller than a hare -/
def camel_height_factor : ℕ := 24

/-- The number of inches in a foot -/
def inches_per_foot : ℕ := 12

/-- Calculates the height of a camel in feet -/
def camel_height_in_feet : ℕ :=
  (hare_height * camel_height_factor) / inches_per_foot

/-- Theorem stating that the camel's height is 28 feet -/
theorem camel_height_is_28_feet : camel_height_in_feet = 28 := by
  sorry

end camel_height_is_28_feet_l667_66705


namespace unique_K_value_l667_66738

theorem unique_K_value : ∃! K : ℕ, 
  (∃ Z : ℕ, 1000 < Z ∧ Z < 8000 ∧ K > 2 ∧ Z = K * K^2) ∧ 
  (∃ a b : ℕ, K^3 = a^2 ∧ K^3 = b^3) ∧
  K = 16 :=
sorry

end unique_K_value_l667_66738


namespace income_increase_is_fifty_percent_l667_66794

/-- Represents the financial situation of a person over two years -/
structure FinancialData where
  income1 : ℝ
  savingsRate1 : ℝ
  incomeIncrease : ℝ

/-- The conditions of the problem -/
def problemConditions (d : FinancialData) : Prop :=
  d.savingsRate1 = 0.5 ∧
  d.income1 > 0 ∧
  d.incomeIncrease > 0 ∧
  let savings1 := d.savingsRate1 * d.income1
  let expenditure1 := d.income1 - savings1
  let income2 := d.income1 * (1 + d.incomeIncrease)
  let savings2 := 2 * savings1
  let expenditure2 := income2 - savings2
  expenditure1 + expenditure2 = 2 * expenditure1

/-- The theorem stating that under the given conditions, 
    the income increase in the second year is 50% -/
theorem income_increase_is_fifty_percent (d : FinancialData) :
  problemConditions d → d.incomeIncrease = 0.5 := by
  sorry

end income_increase_is_fifty_percent_l667_66794


namespace parallel_lines_condition_l667_66746

theorem parallel_lines_condition (a₁ b₁ c₁ a₂ b₂ c₂ : ℝ) 
  (h₁ : a₁^2 + b₁^2 ≠ 0) (h₂ : a₂^2 + b₂^2 ≠ 0) :
  ¬(∀ (x y : ℝ), (a₁*x + b₁*y + c₁ = 0 ↔ a₂*x + b₂*y + c₂ = 0) ↔ 
    (a₁*b₂ - a₂*b₁ ≠ 0)) :=
sorry

end parallel_lines_condition_l667_66746


namespace problem_statement_l667_66754

theorem problem_statement (a b c d : ℕ+) 
  (h1 : a^3 = b^2) 
  (h2 : c^4 = d^3) 
  (h3 : c - a = 31) : 
  d - b = 229 := by
  sorry

end problem_statement_l667_66754


namespace train_length_l667_66750

/-- The length of a train given its speed and time to cross a pole -/
theorem train_length (speed_kmph : ℝ) (crossing_time : ℝ) : 
  speed_kmph = 18 → crossing_time = 5 → 
  (speed_kmph * 1000 / 3600) * crossing_time = 25 := by sorry

end train_length_l667_66750


namespace parallel_lines_in_parallel_planes_parallel_line_to_intersecting_planes_l667_66752

-- Define the basic types
variable (Point : Type) (Line : Type) (Plane : Type)

-- Define the relationships between geometric objects
variable (perpendicular : Line → Line → Prop)
variable (perpendicular_plane : Plane → Plane → Prop)
variable (perpendicular_line_plane : Line → Plane → Prop)
variable (parallel : Line → Line → Prop)
variable (parallel_plane : Plane → Plane → Prop)
variable (parallel_line_plane : Line → Plane → Prop)
variable (intersect : Plane → Plane → Line → Prop)
variable (in_plane : Line → Plane → Prop)

-- Theorem for proposition 2
theorem parallel_lines_in_parallel_planes
  (α β γ : Plane) (m n : Line) :
  parallel_plane α β →
  intersect α γ m →
  intersect β γ n →
  parallel m n :=
sorry

-- Theorem for proposition 4
theorem parallel_line_to_intersecting_planes
  (α β : Plane) (m n : Line) :
  intersect α β m →
  parallel m n →
  ¬in_plane n α →
  ¬in_plane n β →
  parallel_line_plane n α ∧ parallel_line_plane n β :=
sorry

end parallel_lines_in_parallel_planes_parallel_line_to_intersecting_planes_l667_66752


namespace two_p_plus_q_l667_66743

theorem two_p_plus_q (p q : ℚ) (h : p / q = 3 / 5) : 2 * p + q = (11 / 5) * q := by
  sorry

end two_p_plus_q_l667_66743


namespace linear_function_k_value_l667_66765

/-- Given a linear function y = kx + 1 passing through the point (-1, 0), prove that k = 1 -/
theorem linear_function_k_value (k : ℝ) : 
  (∀ x y : ℝ, y = k * x + 1) → -- The function is linear with equation y = kx + 1
  (0 = k * (-1) + 1) →         -- The graph passes through the point (-1, 0)
  k = 1                        -- Conclusion: k equals 1
:= by sorry

end linear_function_k_value_l667_66765


namespace students_interested_in_both_sports_and_music_l667_66769

/-- Given a class with the following properties:
  * There are 55 students in total
  * 43 students are sports enthusiasts
  * 34 students are music enthusiasts
  * 4 students are neither interested in sports nor music
  Prove that 26 students are interested in both sports and music -/
theorem students_interested_in_both_sports_and_music 
  (total : ℕ) (sports : ℕ) (music : ℕ) (neither : ℕ) 
  (h_total : total = 55)
  (h_sports : sports = 43)
  (h_music : music = 34)
  (h_neither : neither = 4) :
  sports + music - (total - neither) = 26 := by
  sorry

end students_interested_in_both_sports_and_music_l667_66769


namespace average_price_rahim_l667_66783

/-- Represents a book purchase from a shop -/
structure BookPurchase where
  quantity : ℕ
  totalPrice : ℕ

/-- Calculates the average price per book given a list of book purchases -/
def averagePrice (purchases : List BookPurchase) : ℚ :=
  let totalBooks := purchases.map (fun p => p.quantity) |>.sum
  let totalCost := purchases.map (fun p => p.totalPrice) |>.sum
  (totalCost : ℚ) / (totalBooks : ℚ)

theorem average_price_rahim (purchases : List BookPurchase) 
  (h1 : purchases = [
    ⟨40, 600⟩,  -- Shop A
    ⟨20, 240⟩,  -- Shop B
    ⟨15, 180⟩,  -- Shop C
    ⟨25, 325⟩   -- Shop D
  ]) : 
  averagePrice purchases = 1345 / 100 := by
  sorry

#eval (1345 : ℚ) / 100  -- To verify the result is indeed 13.45

end average_price_rahim_l667_66783


namespace fruit_basket_problem_l667_66792

theorem fruit_basket_problem :
  Nat.gcd (Nat.gcd 15 9) 18 = 3 := by
  sorry

end fruit_basket_problem_l667_66792


namespace function_value_at_five_l667_66745

/-- Given a function g : ℝ → ℝ satisfying g(x) + 3g(1-x) = 4x^2 - 1 for all x,
    prove that g(5) = 11.25 -/
theorem function_value_at_five (g : ℝ → ℝ) 
    (h : ∀ x, g x + 3 * g (1 - x) = 4 * x^2 - 1) : 
    g 5 = 11.25 := by
  sorry

end function_value_at_five_l667_66745


namespace angle_rotation_and_trig_identity_l667_66776

theorem angle_rotation_and_trig_identity 
  (initial_angle : Real) 
  (rotations : Nat) 
  (α : Real) 
  (h1 : initial_angle = 30 * Real.pi / 180)
  (h2 : rotations = 3)
  (h3 : Real.sin (-Real.pi/2 - α) = -1/3)
  (h4 : Real.tan α < 0) :
  (initial_angle + rotations * 2 * Real.pi) * 180 / Real.pi = 1110 ∧ 
  Real.cos (3 * Real.pi / 2 + α) = -2 * Real.sqrt 2 / 3 := by
  sorry

end angle_rotation_and_trig_identity_l667_66776


namespace tank_capacity_l667_66731

theorem tank_capacity (initial_fullness : Rat) (final_fullness : Rat) (added_water : Rat) :
  initial_fullness = 1/4 →
  final_fullness = 2/3 →
  added_water = 120 →
  (final_fullness - initial_fullness) * (added_water / (final_fullness - initial_fullness)) = 288 :=
by
  sorry

#check tank_capacity

end tank_capacity_l667_66731


namespace sequence_range_l667_66751

theorem sequence_range (a : ℝ) : 
  (∀ n : ℕ+, (fun n => if n < 6 then (1/2 - a) * n + 1 else a^(n - 5)) n > 
             (fun n => if n < 6 then (1/2 - a) * n + 1 else a^(n - 5)) (n + 1)) → 
  (1/2 < a ∧ a < 7/12) := by
  sorry

end sequence_range_l667_66751


namespace polynomial_value_at_n_plus_one_l667_66779

theorem polynomial_value_at_n_plus_one (n : ℕ) (P : Polynomial ℝ) 
  (h_degree : P.degree ≤ n) 
  (h_values : ∀ k : ℕ, k ≤ n → P.eval (k : ℝ) = k / (k + 1)) :
  P.eval ((n + 1 : ℕ) : ℝ) = (n + 1 + (-1)^(n + 1)) / (n + 2) := by
  sorry

end polynomial_value_at_n_plus_one_l667_66779


namespace quadratic_expansion_sum_l667_66706

theorem quadratic_expansion_sum (a b : ℝ) : 
  (∀ x : ℝ, x^2 + 4*x + 3 = (x - 1)^2 + a*(x - 1) + b) → a + b = 14 := by
  sorry

end quadratic_expansion_sum_l667_66706


namespace line_vector_to_slope_intercept_l667_66710

/-- Given a line in vector form, proves that its slope-intercept form has specific m and b values -/
theorem line_vector_to_slope_intercept :
  let vector_form := fun (x y : ℝ) => -3 * (x - 5) + 2 * (y + 1) = 0
  let slope_intercept_form := fun (x y : ℝ) => y = (3/2) * x - 17/2
  (∀ x y, vector_form x y ↔ slope_intercept_form x y) := by
  sorry

end line_vector_to_slope_intercept_l667_66710


namespace correct_statements_count_l667_66739

-- Define a structure to represent a statement
structure GeometricStatement :=
  (id : Nat)
  (content : String)
  (isCorrect : Bool)

-- Define the four statements
def statement1 : GeometricStatement :=
  { id := 1
  , content := "The prism with the least number of faces has 6 vertices"
  , isCorrect := true }

def statement2 : GeometricStatement :=
  { id := 2
  , content := "A frustum is the middle part of a cone cut by two parallel planes"
  , isCorrect := false }

def statement3 : GeometricStatement :=
  { id := 3
  , content := "A plane passing through the vertex of a cone cuts the cone into a section that is an isosceles triangle"
  , isCorrect := true }

def statement4 : GeometricStatement :=
  { id := 4
  , content := "Equal angles remain equal in perspective drawings"
  , isCorrect := false }

-- Define the list of all statements
def allStatements : List GeometricStatement :=
  [statement1, statement2, statement3, statement4]

-- Theorem to prove
theorem correct_statements_count :
  (allStatements.filter (·.isCorrect)).length = 2 := by
  sorry

end correct_statements_count_l667_66739


namespace perimeter_of_triangle_from_unit_square_l667_66793

/-- Represents a triangle formed from a unit square --/
structure TriangleFromUnitSquare where
  /-- The base of the isosceles triangle --/
  base : ℝ
  /-- One leg of the isosceles triangle --/
  leg : ℝ
  /-- The triangle is isosceles --/
  isIsosceles : base = 2 * leg
  /-- The base is formed by two sides of the unit square --/
  baseFromSquare : base = Real.sqrt 2
  /-- Each leg is formed by one side of the unit square --/
  legFromSquare : leg = Real.sqrt 2 / 2

/-- The perimeter of the triangle formed from a unit square is 2√2 --/
theorem perimeter_of_triangle_from_unit_square (t : TriangleFromUnitSquare) :
  t.base + 2 * t.leg = 2 * Real.sqrt 2 := by
  sorry

end perimeter_of_triangle_from_unit_square_l667_66793


namespace sin_alpha_minus_cos_alpha_l667_66703

theorem sin_alpha_minus_cos_alpha (α : Real) (h : Real.tan α = -3/4) :
  Real.sin α * (Real.sin α - Real.cos α) = 21/25 := by
  sorry

end sin_alpha_minus_cos_alpha_l667_66703


namespace chocolates_on_square_perimeter_l667_66795

/-- The number of chocolates on one side of the square -/
def chocolates_per_side : ℕ := 6

/-- The number of sides in a square -/
def sides_of_square : ℕ := 4

/-- The number of corners in a square -/
def corners_of_square : ℕ := 4

/-- The total number of chocolates around the perimeter of the square -/
def chocolates_on_perimeter : ℕ := chocolates_per_side * sides_of_square - corners_of_square

theorem chocolates_on_square_perimeter : chocolates_on_perimeter = 20 := by
  sorry

end chocolates_on_square_perimeter_l667_66795


namespace area1_is_linear_area2_is_quadratic_l667_66797

-- Define the rectangles
def rectangle1 (x : ℝ) : ℝ × ℝ := (10 - x, 5)
def rectangle2 (x : ℝ) : ℝ × ℝ := (30 + x, 20 + x)

-- Define the area functions
def area1 (x : ℝ) : ℝ := (rectangle1 x).1 * (rectangle1 x).2
def area2 (x : ℝ) : ℝ := (rectangle2 x).1 * (rectangle2 x).2

-- Theorem statements
theorem area1_is_linear : ∃ (m b : ℝ), ∀ x, area1 x = m * x + b := by sorry

theorem area2_is_quadratic : ∃ (a b c : ℝ), a ≠ 0 ∧ (∀ x, area2 x = a * x^2 + b * x + c) := by sorry

end area1_is_linear_area2_is_quadratic_l667_66797


namespace dumbbell_weight_problem_l667_66756

theorem dumbbell_weight_problem (total_weight : ℝ) (first_pair_weight : ℝ) (third_pair_weight : ℝ) 
  (h1 : total_weight = 32)
  (h2 : first_pair_weight = 3)
  (h3 : third_pair_weight = 8) :
  total_weight - 2 * first_pair_weight - 2 * third_pair_weight = 10 := by
  sorry

end dumbbell_weight_problem_l667_66756


namespace fair_distribution_theorem_l667_66790

/-- Represents the outcome of a chess game -/
inductive GameOutcome
  | AWin
  | BWin

/-- Represents the state of the chess competition -/
structure ChessCompetition where
  total_games : Nat
  games_played : Nat
  a_wins : Nat
  b_wins : Nat
  prize_money : Nat

/-- Calculates the probability of player A winning the competition -/
def probability_a_wins (comp : ChessCompetition) : Rat :=
  sorry

/-- Calculates the fair distribution of prize money -/
def fair_distribution (comp : ChessCompetition) : Nat × Nat :=
  sorry

/-- Theorem stating the fair distribution of prize money -/
theorem fair_distribution_theorem (comp : ChessCompetition) 
  (h1 : comp.total_games = 7)
  (h2 : comp.games_played = 5)
  (h3 : comp.a_wins = 3)
  (h4 : comp.b_wins = 2)
  (h5 : comp.prize_money = 10000) :
  fair_distribution comp = (7500, 2500) :=
sorry

end fair_distribution_theorem_l667_66790


namespace equation_solutions_l667_66773

def has_different_divisors (a b : ℤ) : Prop :=
  ∃ d : ℤ, (d ∣ a ∧ ¬(d ∣ b)) ∨ (d ∣ b ∧ ¬(d ∣ a))

theorem equation_solutions :
  ∀ a b : ℤ, has_different_divisors a b → a^2 + a = b^3 + b →
  ((a = 1 ∧ b = 1) ∨ (a = -2 ∧ b = 1) ∨ (a = 5 ∧ b = 3)) :=
sorry

end equation_solutions_l667_66773


namespace magnitude_of_complex_fraction_l667_66725

theorem magnitude_of_complex_fraction (i : ℂ) (h : i^2 = -1) :
  Complex.abs ((1 - i) / (2 * i + 1)) = Real.sqrt 10 / 5 := by
  sorry

end magnitude_of_complex_fraction_l667_66725


namespace smallest_coinciding_triangle_l667_66768

/-- Represents the type of isosceles triangle -/
inductive TriangleType
  | Acute
  | Right

/-- Returns the vertex angle of a triangle based on its type -/
def vertexAngle (t : TriangleType) : ℕ :=
  match t with
  | TriangleType.Acute => 30
  | TriangleType.Right => 90

/-- Returns the type of the n-th triangle in the sequence -/
def nthTriangleType (n : ℕ) : TriangleType :=
  if n % 3 = 0 then TriangleType.Right else TriangleType.Acute

/-- Calculates the sum of vertex angles for the first n triangles -/
def sumOfAngles (n : ℕ) : ℕ :=
  List.range n |> List.map (fun i => vertexAngle (nthTriangleType (i + 1))) |> List.sum

/-- The main theorem to prove -/
theorem smallest_coinciding_triangle : 
  (∀ k < 23, sumOfAngles k % 360 ≠ 0) ∧ sumOfAngles 23 % 360 = 0 := by
  sorry


end smallest_coinciding_triangle_l667_66768


namespace equation_solutions_l667_66728

def is_solution (X Y Z : ℕ) : Prop :=
  X^Y + Y^Z = X * Y * Z

theorem equation_solutions :
  ∀ X Y Z : ℕ,
    is_solution X Y Z ↔
      (X = 1 ∧ Y = 1 ∧ Z = 2) ∨
      (X = 2 ∧ Y = 2 ∧ Z = 2) ∨
      (X = 2 ∧ Y = 2 ∧ Z = 3) ∨
      (X = 4 ∧ Y = 2 ∧ Z = 3) ∨
      (X = 4 ∧ Y = 2 ∧ Z = 4) :=
by sorry

end equation_solutions_l667_66728


namespace percentage_problem_l667_66798

theorem percentage_problem (x : ℝ) : 0.25 * x = 0.15 * 1500 - 30 → x = 780 := by
  sorry

end percentage_problem_l667_66798


namespace min_value_reciprocal_sum_l667_66718

theorem min_value_reciprocal_sum (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y = 2) :
  1/x + 1/y ≥ 2 ∧ (1/x + 1/y = 2 ↔ x = 1 ∧ y = 1) := by
  sorry

end min_value_reciprocal_sum_l667_66718


namespace distinct_triangles_in_grid_l667_66791

/-- The number of points in a 3 x 2 grid -/
def total_points : ℕ := 6

/-- The number of points needed to form a triangle -/
def points_per_triangle : ℕ := 3

/-- The number of rows in the grid -/
def num_rows : ℕ := 3

/-- Function to calculate combinations -/
def choose (n k : ℕ) : ℕ :=
  if k > n then 0
  else (List.range n).foldl (λ acc i => acc * (n - i) / (i + 1)) 1

/-- The number of degenerate cases (collinear points in rows) -/
def degenerate_cases : ℕ := num_rows

/-- Theorem: The number of distinct triangles in a 3 x 2 grid is 17 -/
theorem distinct_triangles_in_grid :
  choose total_points points_per_triangle - degenerate_cases = 17 := by
  sorry


end distinct_triangles_in_grid_l667_66791


namespace ellipse_on_y_axis_l667_66722

/-- Given real numbers m and n where m > n > 0, the equation mx² + ny² = 1 represents an ellipse with foci on the y-axis -/
theorem ellipse_on_y_axis (m n : ℝ) (h1 : m > n) (h2 : n > 0) :
  ∃ (a b : ℝ), a > b ∧ b > 0 ∧ 
  (∀ (x y : ℝ), m * x^2 + n * y^2 = 1 ↔ (x^2 / a^2) + (y^2 / b^2) = 1) ∧
  (∃ (c : ℝ), c > 0 ∧ c^2 = a^2 - b^2) :=
sorry

end ellipse_on_y_axis_l667_66722


namespace double_march_earnings_cars_l667_66708

/-- Represents the earnings of a car salesman -/
structure CarSalesmanEarnings where
  baseSalary : ℕ
  commissionPerCar : ℕ
  marchEarnings : ℕ

/-- Calculates the number of cars needed to be sold to reach a target earning -/
def carsNeededForTarget (e : CarSalesmanEarnings) (targetEarnings : ℕ) : ℕ :=
  ((targetEarnings - e.baseSalary) + e.commissionPerCar - 1) / e.commissionPerCar

/-- Theorem: The number of cars needed to double March earnings is 15 -/
theorem double_march_earnings_cars (e : CarSalesmanEarnings) 
    (h1 : e.baseSalary = 1000)
    (h2 : e.commissionPerCar = 200)
    (h3 : e.marchEarnings = 2000) : 
    carsNeededForTarget e (2 * e.marchEarnings) = 15 := by
  sorry

end double_march_earnings_cars_l667_66708


namespace horner_v₂_value_l667_66788

def horner_polynomial (x : ℝ) : ℝ := x^6 - 5*x^5 + 6*x^4 - 3*x^3 + 1.8*x^2 + 0.35*x + 2

def v₀ : ℝ := 1

def v₁ (x : ℝ) : ℝ := v₀ * x - 5

def v₂ (x : ℝ) : ℝ := v₁ x * x + 6

theorem horner_v₂_value :
  v₂ (-1) = 12 :=
by sorry

end horner_v₂_value_l667_66788


namespace angle_triple_complement_l667_66771

theorem angle_triple_complement (x : ℝ) : 
  (x = 3 * (90 - x)) → x = 67.5 := by
  sorry

end angle_triple_complement_l667_66771


namespace trains_crossing_time_l667_66714

/-- Proves the time taken for two trains to cross each other -/
theorem trains_crossing_time
  (length : ℝ)
  (time1 time2 : ℝ)
  (h1 : length = 120)
  (h2 : time1 = 5)
  (h3 : time2 = 15)
  : (2 * length) / ((length / time1) + (length / time2)) = 7.5 := by
  sorry

end trains_crossing_time_l667_66714


namespace motorcyclist_cyclist_problem_l667_66735

/-- The distance between two points A and B, given the conditions of the problem -/
def distance_AB : ℝ := 20

theorem motorcyclist_cyclist_problem (x : ℝ) 
  (h1 : x > 0) -- Ensure distance is positive
  (h2 : x - 4 > 0) -- Ensure meeting point is between A and B
  (h3 : (x - 4) / 4 = x / (x - 15)) -- Ratio of speeds equation
  : x = distance_AB := by
  sorry

end motorcyclist_cyclist_problem_l667_66735


namespace max_balloons_with_promotion_orvin_max_balloons_l667_66719

/-- The maximum number of balloons that can be bought given a promotion --/
theorem max_balloons_with_promotion (full_price_balloons : ℕ) : ℕ :=
  let discounted_sets := (full_price_balloons * 2) / 3
  discounted_sets * 2

/-- Proof that given the conditions, the maximum number of balloons Orvin can buy is 52 --/
theorem orvin_max_balloons : max_balloons_with_promotion 40 = 52 := by
  sorry

end max_balloons_with_promotion_orvin_max_balloons_l667_66719


namespace pet_shop_limbs_l667_66786

/-- The total number of legs and arms in the pet shop -/
def total_limbs : ℕ :=
  4 * 2 +  -- birds
  6 * 4 +  -- dogs
  5 * 0 +  -- snakes
  2 * 8 +  -- spiders
  3 * 4 +  -- horses
  7 * 4 +  -- rabbits
  2 * 8 +  -- octopuses
  8 * 6 +  -- ants
  1 * 12   -- unique creature

/-- Theorem stating that the total number of legs and arms in the pet shop is 164 -/
theorem pet_shop_limbs : total_limbs = 164 := by
  sorry

end pet_shop_limbs_l667_66786


namespace chess_tournament_games_l667_66737

theorem chess_tournament_games (n : ℕ) (h : n = 8) : 
  n * (n - 1) = 56 ∧ 2 * (n * (n - 1)) = 112 := by
  sorry

#check chess_tournament_games

end chess_tournament_games_l667_66737


namespace vector_subtraction_scalar_multiplication_l667_66730

theorem vector_subtraction_scalar_multiplication :
  let v1 : Fin 2 → ℝ := ![3, -8]
  let v2 : Fin 2 → ℝ := ![-2, 6]
  let scalar : ℝ := 5
  v1 - scalar • v2 = ![13, -38] := by
  sorry

end vector_subtraction_scalar_multiplication_l667_66730


namespace birdseed_mix_l667_66780

/-- Given two brands of birdseed and their composition, prove the percentage of sunflower in Brand A -/
theorem birdseed_mix (x : ℝ) : 
  (0.4 + x / 100 = 1) →  -- Brand A composition
  (0.65 + 0.35 = 1) →  -- Brand B composition
  (0.6 * x / 100 + 0.4 * 0.35 = 0.5) →  -- Mix composition
  x = 60 := by sorry

end birdseed_mix_l667_66780


namespace orange_harvest_l667_66770

theorem orange_harvest (days : ℕ) (total_sacks : ℕ) (sacks_per_day : ℕ) : 
  days = 6 → total_sacks = 498 → sacks_per_day * days = total_sacks → sacks_per_day = 83 := by
  sorry

end orange_harvest_l667_66770


namespace factorization_3a_squared_minus_3_l667_66744

theorem factorization_3a_squared_minus_3 (a : ℝ) : 3 * a^2 - 3 = 3 * (a - 1) * (a + 1) := by
  sorry

end factorization_3a_squared_minus_3_l667_66744


namespace circle_equation_l667_66726

/-- The standard equation of a circle with center (-3, 4) and radius √5 -/
theorem circle_equation (x y : ℝ) :
  let center : ℝ × ℝ := (-3, 4)
  let radius : ℝ := Real.sqrt 5
  (x + 3)^2 + (y - 4)^2 = 5 ↔
    ((x - center.1)^2 + (y - center.2)^2 = radius^2) :=
by sorry

end circle_equation_l667_66726


namespace quadratic_zeros_imply_a_range_l667_66740

/-- A quadratic function f(x) = x^2 - 2ax + 4 with parameter a -/
def f (a : ℝ) (x : ℝ) : ℝ := x^2 - 2*a*x + 4

/-- The property that f has two zeros in the interval (1, +∞) -/
def has_two_zeros_after_one (a : ℝ) : Prop :=
  ∃ x y, 1 < x ∧ x < y ∧ f a x = 0 ∧ f a y = 0

/-- If f(x) = x^2 - 2ax + 4 has two zeros in (1, +∞), then 2 < a < 5/2 -/
theorem quadratic_zeros_imply_a_range (a : ℝ) : 
  has_two_zeros_after_one a → 2 < a ∧ a < 5/2 := by
  sorry

end quadratic_zeros_imply_a_range_l667_66740


namespace pulley_centers_distance_l667_66747

/-- Distance between centers of two pulleys -/
theorem pulley_centers_distance 
  (r1 : ℝ) (r2 : ℝ) (contact_distance : ℝ)
  (h1 : r1 = 10)
  (h2 : r2 = 6)
  (h3 : contact_distance = 30) :
  ∃ (center_distance : ℝ), 
    center_distance = 2 * Real.sqrt 229 := by
  sorry

end pulley_centers_distance_l667_66747


namespace sqrt2_similarity_l667_66716

-- Define similarity for quadratic surds
def similar_quadratic_surds (a b : ℝ) : Prop :=
  ∃ (r : ℚ), r ≠ 0 ∧ a = r * b

-- Theorem statement
theorem sqrt2_similarity (r : ℚ) (h : r ≠ 0) :
  similar_quadratic_surds (r * Real.sqrt 2) (Real.sqrt 2) :=
sorry

end sqrt2_similarity_l667_66716


namespace closest_integer_to_cube_root_200_l667_66707

theorem closest_integer_to_cube_root_200 : 
  ∀ n : ℤ, |n - (200 : ℝ)^(1/3)| ≥ |6 - (200 : ℝ)^(1/3)| := by
  sorry

end closest_integer_to_cube_root_200_l667_66707


namespace arithmetic_computation_l667_66757

theorem arithmetic_computation : -10 * 3 - (-4 * -2) + (-12 * -4) / 2 = -14 := by
  sorry

end arithmetic_computation_l667_66757


namespace sarah_marriage_age_l667_66717

def game_prediction (name_length : ℕ) (current_age : ℕ) : ℕ :=
  name_length + 2 * current_age

theorem sarah_marriage_age :
  game_prediction 5 9 = 23 := by
  sorry

end sarah_marriage_age_l667_66717


namespace blue_cube_problem_l667_66759

theorem blue_cube_problem (n : ℕ) (h : n > 0) :
  (6 * n^2 : ℚ) / (6 * n^3 : ℚ) = 1 / 3 → n = 3 :=
by sorry

end blue_cube_problem_l667_66759


namespace mountaineer_arrangements_l667_66753

theorem mountaineer_arrangements (total : ℕ) (familiar : ℕ) (groups : ℕ) (familiar_per_group : ℕ) :
  total = 10 →
  familiar = 4 →
  groups = 2 →
  familiar_per_group = 2 →
  (familiar.choose familiar_per_group) * ((total - familiar).choose familiar_per_group) * groups = 120 :=
by sorry

end mountaineer_arrangements_l667_66753


namespace lg_graph_property_l667_66727

-- Define the logarithm function (base 10)
noncomputable def lg (x : ℝ) : ℝ := Real.log x / Real.log 10

-- Define what it means for a point to be on the graph of y = lg x
def on_lg_graph (p : ℝ × ℝ) : Prop := p.2 = lg p.1

-- Theorem statement
theorem lg_graph_property (a b : ℝ) (h1 : on_lg_graph (a, b)) (h2 : a ≠ 1) :
  on_lg_graph (a^2, 2*b) :=
sorry

end lg_graph_property_l667_66727


namespace l_shape_area_l667_66733

/-- Represents the dimensions of a rectangle -/
structure Rectangle where
  length : ℝ
  width : ℝ

/-- Calculates the area of a rectangle -/
def area (r : Rectangle) : ℝ := r.length * r.width

/-- The large rectangle -/
def large_rectangle : Rectangle := { length := 10, width := 6 }

/-- The small rectangle to be subtracted -/
def small_rectangle : Rectangle := { length := 4, width := 3 }

/-- The number of small rectangles to be subtracted -/
def num_small_rectangles : ℕ := 2

/-- Theorem: The area of the L-shape is 36 square units -/
theorem l_shape_area : 
  area large_rectangle - num_small_rectangles * area small_rectangle = 36 := by
  sorry

end l_shape_area_l667_66733


namespace obtuse_triangle_consecutive_sides_l667_66762

/-- An obtuse triangle with consecutive natural number side lengths has sides 2, 3, and 4 -/
theorem obtuse_triangle_consecutive_sides : 
  ∀ (a b c : ℕ), 
  (a < b) ∧ (b < c) ∧  -- consecutive
  (c = a + 2) ∧        -- consecutive
  (c^2 > a^2 + b^2) →  -- obtuse (by law of cosines)
  a = 2 ∧ b = 3 ∧ c = 4 := by
sorry

end obtuse_triangle_consecutive_sides_l667_66762


namespace base_7_units_digit_of_sum_l667_66764

theorem base_7_units_digit_of_sum (a b : ℕ) (ha : a = 156) (hb : b = 97) :
  (a + b) % 7 = 1 := by
  sorry

end base_7_units_digit_of_sum_l667_66764


namespace both_normal_l667_66715

-- Define a type for people
inductive Person : Type
| MrA : Person
| MrsA : Person

-- Define what it means to be normal
def normal (p : Person) : Prop := True

-- Define the statement made by each person
def statement (p : Person) : Prop :=
  match p with
  | Person.MrA => normal Person.MrsA
  | Person.MrsA => normal Person.MrA

-- Theorem: There exists a consistent interpretation where both are normal
theorem both_normal :
  ∃ (interp : Person → Prop),
    (∀ p, interp p ↔ normal p) ∧
    (∀ p, interp p → statement p) :=
sorry

end both_normal_l667_66715


namespace initial_order_is_60_l667_66777

/-- Represents the cog production scenario with two production rates and an overall average --/
def CogProduction (initial_rate : ℝ) (increased_rate : ℝ) (additional_cogs : ℝ) (average_output : ℝ) : Prop :=
  ∃ (initial_order : ℝ),
    initial_order > 0 ∧
    (initial_order + additional_cogs) / (initial_order / initial_rate + additional_cogs / increased_rate) = average_output

/-- Theorem stating that given the specific production rates and average, the initial order is 60 cogs --/
theorem initial_order_is_60 :
  CogProduction 15 60 60 24 → ∃ (x : ℝ), x = 60 ∧ CogProduction 15 60 60 24 := by
  sorry

#check initial_order_is_60

end initial_order_is_60_l667_66777


namespace extreme_value_condition_l667_66712

-- Define the function f
def f (a b : ℝ) (x : ℝ) : ℝ := x^3 + a*x^2 + b*x + a^2

-- State the theorem
theorem extreme_value_condition (a b : ℝ) :
  (f a b 1 = 4) ∧ (∃ ε > 0, ∀ x ∈ Set.Ioo (1 - ε) (1 + ε), f a b x ≤ f a b 1) →
  a * b = -27 ∨ a * b = -2 :=
by sorry

end extreme_value_condition_l667_66712


namespace platform_length_l667_66784

/-- Calculates the length of a platform given train parameters --/
theorem platform_length 
  (train_length : ℝ) 
  (train_speed_kmph : ℝ) 
  (crossing_time : ℝ) : 
  train_length = 175 →
  train_speed_kmph = 36 →
  crossing_time = 40 →
  (train_speed_kmph * 1000 / 3600 * crossing_time) - train_length = 225 := by
  sorry

#check platform_length

end platform_length_l667_66784


namespace projection_closed_l667_66711

open Set
open Topology

-- Define the projection function
def proj_y (p : ℝ × ℝ) : ℝ := p.2

-- State the theorem
theorem projection_closed {a b : ℝ} {S : Set (ℝ × ℝ)} 
  (hS : IsClosed S) 
  (hSub : S ⊆ {p : ℝ × ℝ | a < p.1 ∧ p.1 < b}) :
  IsClosed (proj_y '' S) := by
  sorry

end projection_closed_l667_66711


namespace talent_show_proof_l667_66778

theorem talent_show_proof (total : ℕ) (cant_sing cant_dance cant_act : ℕ) : 
  total = 120 →
  cant_sing = 50 →
  cant_dance = 75 →
  cant_act = 35 →
  let can_sing := total - cant_sing
  let can_dance := total - cant_dance
  let can_act := total - cant_act
  let two_talents := can_sing + can_dance + can_act - total
  two_talents = 80 := by
sorry

end talent_show_proof_l667_66778


namespace batsman_highest_score_l667_66736

theorem batsman_highest_score 
  (total_innings : ℕ) 
  (average : ℚ) 
  (score_difference : ℕ) 
  (average_excluding_extremes : ℚ) 
  (h : total_innings = 46) 
  (h1 : average = 60) 
  (h2 : score_difference = 190) 
  (h3 : average_excluding_extremes = 58) : 
  ∃ (highest_score lowest_score : ℕ), 
    highest_score - lowest_score = score_difference ∧ 
    (total_innings : ℚ) * average = (total_innings - 2 : ℚ) * average_excluding_extremes + highest_score + lowest_score ∧
    highest_score = 199 :=
by sorry

end batsman_highest_score_l667_66736


namespace cubic_polynomial_value_l667_66758

/-- The given polynomial h -/
def h (x : ℝ) : ℝ := x^3 - x^2 + x - 1

/-- The roots of h -/
def roots_h : Set ℝ := {x | h x = 0}

/-- The theorem statement -/
theorem cubic_polynomial_value (p : ℝ → ℝ) :
  (∃ a b c : ℝ, roots_h = {a, b, c}) →  -- h has three distinct roots
  (∀ x, x ∈ roots_h → x^3 ∈ {y | p y = 0}) →  -- roots of p are cubes of roots of h
  (∀ x, p (p x) = p (p (p x))) →  -- p is a cubic polynomial
  p 1 = 2 →  -- given condition
  p 8 = 1008 := by  -- conclusion to prove
sorry


end cubic_polynomial_value_l667_66758


namespace calculation_proof_l667_66767

theorem calculation_proof :
  ((125 + 17) * 8 = 1136) ∧ ((458 - (85 + 28)) / 23 = 15) := by
  sorry

end calculation_proof_l667_66767


namespace sum_has_even_digit_l667_66724

def reverse_number (n : List Nat) : List Nat :=
  n.reverse

def sum_digits (n m : List Nat) : List Nat :=
  sorry

theorem sum_has_even_digit (n : List Nat) (h : n.length = 17) :
  ∃ (d : Nat), d ∈ sum_digits n (reverse_number n) ∧ Even d :=
sorry

end sum_has_even_digit_l667_66724


namespace tangent_line_fixed_point_l667_66720

/-- The function f(x) = x^2 + mx + 1 -/
def f (m : ℝ) (x : ℝ) : ℝ := x^2 + m*x + 1

/-- The derivative of f(x) -/
def f_derivative (m : ℝ) (x : ℝ) : ℝ := 2*x + m

/-- Theorem: The tangent line to f(x) at x = 2 passes through (0, -3) for all m -/
theorem tangent_line_fixed_point (m : ℝ) : 
  let x₀ : ℝ := 2
  let y₀ : ℝ := f m x₀
  let slope : ℝ := f_derivative m x₀
  ∃ (k : ℝ), k * slope = y₀ + 3 ∧ k * (-1) = x₀ := by
  sorry


end tangent_line_fixed_point_l667_66720


namespace intersection_A_B_l667_66701

def A : Set ℤ := {1, 2, 3}

def B : Set ℤ := {x | x * (x + 1) * (x - 2) < 0}

theorem intersection_A_B : A ∩ B = {1} := by sorry

end intersection_A_B_l667_66701


namespace power_difference_l667_66781

theorem power_difference (m n : ℕ) (h1 : 2^m = 32) (h2 : 3^n = 81) : 5^(m-n) = 5 := by
  sorry

end power_difference_l667_66781
