import Mathlib

namespace NUMINAMATH_CALUDE_lucille_house_height_difference_l2007_200799

/-- Proves that Lucille's house is 9.32 feet shorter than the average height of all houses. -/
theorem lucille_house_height_difference :
  let lucille_height : ℝ := 80
  let neighbor1_height : ℝ := 70.5
  let neighbor2_height : ℝ := 99.3
  let neighbor3_height : ℝ := 84.2
  let neighbor4_height : ℝ := 112.6
  let total_height : ℝ := lucille_height + neighbor1_height + neighbor2_height + neighbor3_height + neighbor4_height
  let average_height : ℝ := total_height / 5
  average_height - lucille_height = 9.32 := by
  sorry

#eval (80 + 70.5 + 99.3 + 84.2 + 112.6) / 5 - 80

end NUMINAMATH_CALUDE_lucille_house_height_difference_l2007_200799


namespace NUMINAMATH_CALUDE_consecutive_even_numbers_sum_l2007_200748

theorem consecutive_even_numbers_sum (x : ℤ) : 
  (x % 2 = 0) →  -- x is even
  (x + (x + 2) + (x + 4) = x + 18) →  -- sum condition
  (x + 4 = 10)  -- largest number is 10
  := by sorry

end NUMINAMATH_CALUDE_consecutive_even_numbers_sum_l2007_200748


namespace NUMINAMATH_CALUDE_sequence_properties_l2007_200754

/-- Sequence a_n is a first-degree function of n -/
def a (n : ℕ) : ℝ := 2 * n + 1

/-- Sequence b_n is composed of a_2, a_4, a_6, a_8, ... -/
def b (n : ℕ) : ℝ := a (2 * n)

theorem sequence_properties :
  (a 1 = 3) ∧ 
  (a 10 = 21) ∧ 
  (∀ n : ℕ, a_2009 = 4019) ∧
  (∀ n : ℕ, b n = 4 * n + 1) :=
sorry

end NUMINAMATH_CALUDE_sequence_properties_l2007_200754


namespace NUMINAMATH_CALUDE_prob_two_defective_shipment_l2007_200737

/-- The probability of selecting two defective smartphones from a shipment -/
def prob_two_defective (total : ℕ) (defective : ℕ) : ℚ :=
  (defective : ℚ) / total * ((defective - 1) : ℚ) / (total - 1)

/-- Theorem: The probability of selecting two defective smartphones from a 
    shipment of 250 smartphones, of which 76 are defective, is equal to 
    (76/250) * (75/249) -/
theorem prob_two_defective_shipment : 
  prob_two_defective 250 76 = 76 / 250 * 75 / 249 := by
  sorry

#eval prob_two_defective 250 76

end NUMINAMATH_CALUDE_prob_two_defective_shipment_l2007_200737


namespace NUMINAMATH_CALUDE_blue_or_green_probability_l2007_200749

/-- A die with colored faces -/
structure ColoredDie where
  sides : ℕ
  red : ℕ
  yellow : ℕ
  blue : ℕ
  green : ℕ
  all_faces : sides = red + yellow + blue + green

/-- The probability of an event -/
def probability (favorable : ℕ) (total : ℕ) : ℚ :=
  favorable / total

theorem blue_or_green_probability (d : ColoredDie)
    (h : d.sides = 10 ∧ d.red = 5 ∧ d.yellow = 3 ∧ d.blue = 1 ∧ d.green = 1) :
    probability (d.blue + d.green) d.sides = 1/5 := by
  sorry

end NUMINAMATH_CALUDE_blue_or_green_probability_l2007_200749


namespace NUMINAMATH_CALUDE_parabola_coefficients_from_vertex_and_point_l2007_200742

/-- Represents a parabola of the form y = ax^2 + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The y-coordinate for a given x on the parabola -/
def Parabola.y (p : Parabola) (x : ℝ) : ℝ :=
  p.a * x^2 + p.b * x + p.c

theorem parabola_coefficients_from_vertex_and_point
  (p : Parabola)
  (vertex_x vertex_y : ℝ)
  (point_x point_y : ℝ)
  (h_vertex : p.y vertex_x = vertex_y)
  (h_point : p.y point_x = point_y)
  (h_vertex_x : vertex_x = 4)
  (h_vertex_y : vertex_y = 3)
  (h_point_x : point_x = 2)
  (h_point_y : point_y = 1) :
  p.a = -1/2 ∧ p.b = 4 ∧ p.c = -5 := by
  sorry

end NUMINAMATH_CALUDE_parabola_coefficients_from_vertex_and_point_l2007_200742


namespace NUMINAMATH_CALUDE_cos_2pi_3_plus_2alpha_l2007_200766

theorem cos_2pi_3_plus_2alpha (α : ℝ) (h : Real.sin (π / 6 - α) = 1 / 4) : 
  Real.cos (2 * π / 3 + 2 * α) = -7 / 8 := by
  sorry

end NUMINAMATH_CALUDE_cos_2pi_3_plus_2alpha_l2007_200766


namespace NUMINAMATH_CALUDE_valid_routes_count_l2007_200725

-- Define the cities
inductive City : Type
| P | Q | R | S | T | U

-- Define the roads
inductive Road : Type
| PQ | PS | PT | PU | QR | QS | RS | RT | SU | UT

-- Define a route as a list of roads
def Route := List Road

-- Function to check if a route is valid
def isValidRoute (r : Route) : Bool :=
  -- Implementation details omitted
  sorry

-- Function to count valid routes
def countValidRoutes : Nat :=
  -- Implementation details omitted
  sorry

-- Theorem statement
theorem valid_routes_count :
  countValidRoutes = 15 :=
sorry

end NUMINAMATH_CALUDE_valid_routes_count_l2007_200725


namespace NUMINAMATH_CALUDE_sally_buttons_count_l2007_200778

/-- The number of buttons needed for all clothing items Sally sews over three days -/
def total_buttons : ℕ :=
  let shirt_buttons := 5
  let pants_buttons := 3
  let jacket_buttons := 10
  let monday := 4 * shirt_buttons + 2 * pants_buttons + 1 * jacket_buttons
  let tuesday := 3 * shirt_buttons + 1 * pants_buttons + 2 * jacket_buttons
  let wednesday := 2 * shirt_buttons + 3 * pants_buttons + 1 * jacket_buttons
  monday + tuesday + wednesday

/-- Theorem stating that the total number of buttons Sally needs is 103 -/
theorem sally_buttons_count : total_buttons = 103 := by
  sorry

end NUMINAMATH_CALUDE_sally_buttons_count_l2007_200778


namespace NUMINAMATH_CALUDE_complex_purely_imaginary_l2007_200771

theorem complex_purely_imaginary (m : ℝ) : 
  let z : ℂ := m + 2*I
  (∃ (y : ℝ), (2 + I) * z = y * I) → m = 1 := by
sorry

end NUMINAMATH_CALUDE_complex_purely_imaginary_l2007_200771


namespace NUMINAMATH_CALUDE_right_triangle_max_ratio_l2007_200751

theorem right_triangle_max_ratio (a b c : ℝ) (h_right : a^2 + b^2 = c^2) (h_a : a = 3) :
  (∀ x y z, x^2 + y^2 = z^2 → x = 3 → (x^2 + y^2 + z^2) / z^2 ≤ 2) ∧
  (∃ x y z, x^2 + y^2 = z^2 ∧ x = 3 ∧ (x^2 + y^2 + z^2) / z^2 = 2) :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_max_ratio_l2007_200751


namespace NUMINAMATH_CALUDE_round_trip_time_l2007_200701

/-- Calculates the total time for a round trip between two towns given the speeds and initial travel time. -/
theorem round_trip_time (speed_to_b : ℝ) (speed_to_a : ℝ) (time_to_b : ℝ) : 
  speed_to_b > 0 → speed_to_a > 0 → time_to_b > 0 →
  speed_to_b = 100 → speed_to_a = 150 → time_to_b = 3 →
  time_to_b + (speed_to_b * time_to_b) / speed_to_a = 5 := by
  sorry

#check round_trip_time

end NUMINAMATH_CALUDE_round_trip_time_l2007_200701


namespace NUMINAMATH_CALUDE_special_sequence_values_l2007_200790

/-- An increasing sequence of natural numbers satisfying a_{a_k} = 3k for any k. -/
def SpecialSequence (a : ℕ → ℕ) : Prop :=
  (∀ n m, n < m → a n < a m) ∧ 
  (∀ k, a (a k) = 3 * k)

theorem special_sequence_values (a : ℕ → ℕ) (h : SpecialSequence a) :
  a 100 = 181 ∧ a 1983 = 3762 := by
  sorry

end NUMINAMATH_CALUDE_special_sequence_values_l2007_200790


namespace NUMINAMATH_CALUDE_sum_reciprocal_lower_bound_l2007_200710

theorem sum_reciprocal_lower_bound (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a + b ≤ 4) :
  1/a + 1/b ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_sum_reciprocal_lower_bound_l2007_200710


namespace NUMINAMATH_CALUDE_equivalent_workout_l2007_200706

/-- Represents the weight of a single dumbbell in pounds -/
def dumbbell_weight : ℕ → ℕ
| 0 => 15
| 1 => 20
| _ => 0

/-- Calculates the total weight lifted given the dumbbell type and number of repetitions -/
def total_weight (dumbbell_type : ℕ) (repetitions : ℕ) : ℕ :=
  2 * dumbbell_weight dumbbell_type * repetitions

/-- Proves that lifting two 15-pound weights 16 times is equivalent to lifting two 20-pound weights 12 times -/
theorem equivalent_workout : total_weight 0 16 = total_weight 1 12 := by
  sorry

end NUMINAMATH_CALUDE_equivalent_workout_l2007_200706


namespace NUMINAMATH_CALUDE_min_value_expression_l2007_200765

theorem min_value_expression (a b : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : a + b = 2) :
  2 / (a + 3 * b) + 1 / (a - b) ≥ (3 + 2 * Real.sqrt 2) / 4 := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l2007_200765


namespace NUMINAMATH_CALUDE_cafeteria_apples_l2007_200721

/-- The number of apples handed out to students -/
def apples_handed_out : ℕ := 8

/-- The number of apples needed for each pie -/
def apples_per_pie : ℕ := 9

/-- The number of pies that could be made with the remaining apples -/
def pies_made : ℕ := 6

/-- The initial number of apples in the cafeteria -/
def initial_apples : ℕ := 62

theorem cafeteria_apples :
  initial_apples = apples_handed_out + apples_per_pie * pies_made :=
by sorry

end NUMINAMATH_CALUDE_cafeteria_apples_l2007_200721


namespace NUMINAMATH_CALUDE_probability_one_authentic_one_defective_l2007_200776

def total_products : ℕ := 5
def authentic_products : ℕ := 4
def defective_products : ℕ := 1

theorem probability_one_authentic_one_defective :
  (authentic_products * defective_products : ℚ) / (total_products.choose 2) = 2 / 5 := by
sorry

end NUMINAMATH_CALUDE_probability_one_authentic_one_defective_l2007_200776


namespace NUMINAMATH_CALUDE_number_puzzle_l2007_200777

theorem number_puzzle : ∃ x : ℝ, 3 * (2 * x + 13) = 93 := by
  sorry

end NUMINAMATH_CALUDE_number_puzzle_l2007_200777


namespace NUMINAMATH_CALUDE_triangle_perimeter_l2007_200750

theorem triangle_perimeter : ∀ (a b c : ℝ),
  a = 3 ∧ b = 6 ∧ 
  (c^2 - 6*c + 8 = 0) ∧
  (a + b > c) ∧ (b + c > a) ∧ (c + a > b) →
  a + b + c = 13 := by
sorry

end NUMINAMATH_CALUDE_triangle_perimeter_l2007_200750


namespace NUMINAMATH_CALUDE_josanna_minimum_score_l2007_200726

def current_scores : List ℕ := [75, 85, 65, 95, 70]
def increase_amount : ℕ := 10

def minimum_next_score (scores : List ℕ) (increase : ℕ) : ℕ :=
  let current_sum := scores.sum
  let current_count := scores.length
  let current_avg := current_sum / current_count
  let target_avg := current_avg + increase
  let total_count := current_count + 1
  target_avg * total_count - current_sum

theorem josanna_minimum_score :
  minimum_next_score current_scores increase_amount = 138 := by
  sorry

end NUMINAMATH_CALUDE_josanna_minimum_score_l2007_200726


namespace NUMINAMATH_CALUDE_min_value_expression_l2007_200745

theorem min_value_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (a + 1/b) * (a + 1/b - 1009) + (b + 1/a) * (b + 1/a - 1009) ≥ -509004.5 := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l2007_200745


namespace NUMINAMATH_CALUDE_sum_equals_336_l2007_200770

theorem sum_equals_336 : 237 + 45 + 36 + 18 = 336 := by
  sorry

end NUMINAMATH_CALUDE_sum_equals_336_l2007_200770


namespace NUMINAMATH_CALUDE_max_shelves_with_five_books_together_l2007_200767

/-- Given 1300 books and k shelves, this theorem states that 18 is the largest value of k
    for which there will always be at least 5 books on the same shelf
    before and after any rearrangement. -/
theorem max_shelves_with_five_books_together (k : ℕ) : 
  (∀ (arrangement₁ arrangement₂ : Fin k → Fin 1300 → Prop), 
    (∀ b, ∃! s, arrangement₁ s b) → 
    (∀ b, ∃! s, arrangement₂ s b) → 
    (∃ s : Fin k, ∃ (books : Finset (Fin 1300)), 
      books.card = 5 ∧ 
      (∀ b ∈ books, arrangement₁ s b ∧ arrangement₂ s b))) ↔ 
  k ≤ 18 :=
sorry

end NUMINAMATH_CALUDE_max_shelves_with_five_books_together_l2007_200767


namespace NUMINAMATH_CALUDE_parking_garage_spots_per_level_l2007_200732

theorem parking_garage_spots_per_level :
  -- Define the number of levels in the parking garage
  let num_levels : ℕ := 4

  -- Define the number of open spots on each level
  let open_spots_level1 : ℕ := 58
  let open_spots_level2 : ℕ := open_spots_level1 + 2
  let open_spots_level3 : ℕ := open_spots_level2 + 5
  let open_spots_level4 : ℕ := 31

  -- Define the total number of full spots
  let full_spots : ℕ := 186

  -- Calculate the total number of open spots
  let total_open_spots : ℕ := open_spots_level1 + open_spots_level2 + open_spots_level3 + open_spots_level4

  -- Calculate the total number of spots
  let total_spots : ℕ := total_open_spots + full_spots

  -- The number of spots per level
  (total_spots / num_levels : ℕ) = 100 := by
  sorry

end NUMINAMATH_CALUDE_parking_garage_spots_per_level_l2007_200732


namespace NUMINAMATH_CALUDE_running_speed_is_six_l2007_200704

/-- Calculates the running speed given swimming speed and average speed -/
def calculate_running_speed (swimming_speed average_speed : ℝ) : ℝ :=
  2 * average_speed - swimming_speed

/-- Proves that given a swimming speed of 1 mph and an average speed of 3.5 mph
    for equal time spent swimming and running, the running speed is 6 mph -/
theorem running_speed_is_six :
  let swimming_speed : ℝ := 1
  let average_speed : ℝ := 3.5
  calculate_running_speed swimming_speed average_speed = 6 := by
  sorry

#eval calculate_running_speed 1 3.5

end NUMINAMATH_CALUDE_running_speed_is_six_l2007_200704


namespace NUMINAMATH_CALUDE_expression_simplification_l2007_200772

theorem expression_simplification :
  (2^2 - 1) * (3^2 - 1) * (4^2 - 1) * (5^2 - 1) / 
  ((2 * 3) * (3 * 4) * (4 * 5) * (5 * 6)) = 1/5 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l2007_200772


namespace NUMINAMATH_CALUDE_alpha_value_l2007_200733

theorem alpha_value (α : ℝ) :
  (6 * Real.sqrt 3) / (3 * Real.sqrt 2 + 2 * Real.sqrt 3) = 3 * Real.sqrt α - 6 →
  α = 6 :=
by sorry

end NUMINAMATH_CALUDE_alpha_value_l2007_200733


namespace NUMINAMATH_CALUDE_orthocenter_property_l2007_200796

-- Define a triangle ABC
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define the orthocenter of a triangle
def orthocenter (t : Triangle) : ℝ × ℝ := sorry

-- Define the cosine of the sum of two angles
def cos_sum_angles (α β : ℝ) : ℝ := sorry

-- Define the measure of an angle
def angle_measure (p q r : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem orthocenter_property (t : Triangle) :
  let O := orthocenter t
  (angle_measure t.A t.B t.C > π / 2) →  -- Angle A is obtuse
  (dist O t.A = dist t.B t.C) →  -- AO = BC
  (cos_sum_angles (angle_measure O t.B t.C) (angle_measure O t.C t.B) = -Real.sqrt 2 / 2) :=
by sorry

end NUMINAMATH_CALUDE_orthocenter_property_l2007_200796


namespace NUMINAMATH_CALUDE_solution_set1_correct_solution_set2_correct_l2007_200786

open Set

-- Define the solution sets
def solution_set1 : Set ℝ := Iic (-3) ∪ Ici 1
def solution_set2 : Set ℝ := Ico (-3) 1 ∪ Ioc 3 7

-- Define the inequalities
def inequality1 (x : ℝ) : Prop := (4 - x) / (x^2 + x + 1) ≤ 1
def inequality2 (x : ℝ) : Prop := 1 < |x - 2| ∧ |x - 2| ≤ 5

-- Theorem statements
theorem solution_set1_correct :
  ∀ x : ℝ, x ∈ solution_set1 ↔ inequality1 x :=
sorry

theorem solution_set2_correct :
  ∀ x : ℝ, x ∈ solution_set2 ↔ inequality2 x :=
sorry

end NUMINAMATH_CALUDE_solution_set1_correct_solution_set2_correct_l2007_200786


namespace NUMINAMATH_CALUDE_alton_daily_earnings_l2007_200731

/-- Calculates daily earnings given weekly rent, weekly profit, and number of workdays --/
def daily_earnings (weekly_rent : ℚ) (weekly_profit : ℚ) (workdays : ℕ) : ℚ :=
  (weekly_rent + weekly_profit) / workdays

/-- Proves that given the specified conditions, daily earnings are $11.20 --/
theorem alton_daily_earnings :
  let weekly_rent : ℚ := 20
  let weekly_profit : ℚ := 36
  let workdays : ℕ := 5
  daily_earnings weekly_rent weekly_profit workdays = 11.2 := by
sorry

end NUMINAMATH_CALUDE_alton_daily_earnings_l2007_200731


namespace NUMINAMATH_CALUDE_cubic_equation_ratio_l2007_200789

/-- Given a cubic equation ax³ + bx² + cx + d = 0 with roots 1, 2, and 3,
    prove that c/d = -11/6 -/
theorem cubic_equation_ratio (a b c d : ℝ) : 
  (∀ x : ℝ, a * x^3 + b * x^2 + c * x + d = 0 ↔ x = 1 ∨ x = 2 ∨ x = 3) →
  c / d = -11 / 6 := by
  sorry

end NUMINAMATH_CALUDE_cubic_equation_ratio_l2007_200789


namespace NUMINAMATH_CALUDE_vowel_initials_probability_l2007_200702

/-- Represents the set of possible initials --/
def Initials : Type := Char

/-- The set of all possible initials --/
def all_initials : Finset Initials := sorry

/-- The set of vowel initials --/
def vowel_initials : Finset Initials := sorry

/-- The number of students in the class --/
def class_size : ℕ := 30

/-- No two students have the same initials --/
axiom unique_initials : class_size ≤ Finset.card all_initials

/-- The probability of picking a student with vowel initials --/
def vowel_probability : ℚ := (Finset.card vowel_initials : ℚ) / (Finset.card all_initials : ℚ)

/-- Main theorem: The probability of picking a student with vowel initials is 5/26 --/
theorem vowel_initials_probability : vowel_probability = 5 / 26 := by sorry

end NUMINAMATH_CALUDE_vowel_initials_probability_l2007_200702


namespace NUMINAMATH_CALUDE_tournament_committee_count_l2007_200779

/-- Number of teams in the frisbee association -/
def num_teams : ℕ := 6

/-- Number of members in each team -/
def team_size : ℕ := 8

/-- Number of members selected from the host team -/
def host_select : ℕ := 3

/-- Number of members selected from each regular non-host team -/
def nonhost_select : ℕ := 2

/-- Number of members selected from the special non-host team -/
def special_nonhost_select : ℕ := 3

/-- Total number of possible tournament committees -/
def total_committees : ℕ := 11568055296

theorem tournament_committee_count :
  (num_teams) *
  (team_size.choose host_select) *
  ((team_size.choose nonhost_select) ^ (num_teams - 2)) *
  (team_size.choose special_nonhost_select) =
  total_committees :=
sorry

end NUMINAMATH_CALUDE_tournament_committee_count_l2007_200779


namespace NUMINAMATH_CALUDE_unique_integer_prime_expressions_l2007_200753

theorem unique_integer_prime_expressions : ∃! n : ℤ, 
  Nat.Prime (Int.natAbs (n^3 - 4*n^2 + 3*n - 35)) ∧ 
  Nat.Prime (Int.natAbs (n^2 + 4*n + 8)) ∧ 
  n = 5 := by sorry

end NUMINAMATH_CALUDE_unique_integer_prime_expressions_l2007_200753


namespace NUMINAMATH_CALUDE_planes_perpendicular_l2007_200785

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (parallel : Line → Line → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (contains : Plane → Line → Prop)
variable (perpendicularPlanes : Plane → Plane → Prop)

-- State the theorem
theorem planes_perpendicular 
  (m l : Line) (α β : Plane) 
  (h1 : m ≠ l) 
  (h2 : α ≠ β) 
  (h3 : parallel m l) 
  (h4 : perpendicular l β) 
  (h5 : contains α m) : 
  perpendicularPlanes α β :=
sorry

end NUMINAMATH_CALUDE_planes_perpendicular_l2007_200785


namespace NUMINAMATH_CALUDE_consecutive_integers_sum_l2007_200700

theorem consecutive_integers_sum (a b c : ℤ) : 
  (b = a + 1) → 
  (c = b + 1) → 
  (a * b * c = 990) → 
  ((a + 2) + (b + 2) + (c + 2) = 36) := by
sorry

end NUMINAMATH_CALUDE_consecutive_integers_sum_l2007_200700


namespace NUMINAMATH_CALUDE_choir_robe_cost_l2007_200782

/-- Calculates the cost of buying additional robes for a school choir. -/
theorem choir_robe_cost (total_robes : ℕ) (existing_robes : ℕ) (cost_per_robe : ℕ) : 
  total_robes = 30 → existing_robes = 12 → cost_per_robe = 2 → 
  (total_robes - existing_robes) * cost_per_robe = 36 :=
by
  sorry

#check choir_robe_cost

end NUMINAMATH_CALUDE_choir_robe_cost_l2007_200782


namespace NUMINAMATH_CALUDE_candidate_votes_l2007_200761

theorem candidate_votes (total_votes : ℕ) (invalid_percent : ℚ) (candidate_percent : ℚ) :
  total_votes = 560000 →
  invalid_percent = 15 / 100 →
  candidate_percent = 70 / 100 →
  ⌊(1 - invalid_percent) * candidate_percent * total_votes⌋ = 333200 :=
by sorry

end NUMINAMATH_CALUDE_candidate_votes_l2007_200761


namespace NUMINAMATH_CALUDE_difference_of_squares_153_147_l2007_200773

theorem difference_of_squares_153_147 : 153^2 - 147^2 = 1800 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_153_147_l2007_200773


namespace NUMINAMATH_CALUDE_line_ellipse_intersection_l2007_200764

/-- The line y = k(x-1) + 1 intersects the ellipse (x^2 / 9) + (y^2 / 4) = 1 for any real k -/
theorem line_ellipse_intersection (k : ℝ) :
  ∃ (x y : ℝ), y = k * (x - 1) + 1 ∧ (x^2 / 9) + (y^2 / 4) = 1 := by
  sorry

end NUMINAMATH_CALUDE_line_ellipse_intersection_l2007_200764


namespace NUMINAMATH_CALUDE_greatest_difference_of_unit_digits_l2007_200705

def is_multiple_of_four (n : ℕ) : Prop := ∃ k : ℕ, n = 4 * k

def three_digit_72X (n : ℕ) : Prop := ∃ x : ℕ, n = 720 + x ∧ x < 10

def possible_unit_digit (x : ℕ) : Prop :=
  ∃ n : ℕ, three_digit_72X n ∧ is_multiple_of_four n ∧ n % 10 = x

theorem greatest_difference_of_unit_digits :
  (∃ x y : ℕ, possible_unit_digit x ∧ possible_unit_digit y ∧ x - y = 8) ∧
  (∀ a b : ℕ, possible_unit_digit a → possible_unit_digit b → a - b ≤ 8) :=
sorry

end NUMINAMATH_CALUDE_greatest_difference_of_unit_digits_l2007_200705


namespace NUMINAMATH_CALUDE_rebecca_earrings_gemstones_l2007_200728

/-- Calculates the number of gemstones needed for a given number of earring sets -/
def gemstones_needed (num_sets : ℕ) : ℕ :=
  let magnets_per_earring : ℕ := 2
  let buttons_per_earring : ℕ := magnets_per_earring / 2
  let gemstones_per_earring : ℕ := buttons_per_earring * 3
  let earrings_per_set : ℕ := 2
  num_sets * earrings_per_set * gemstones_per_earring

theorem rebecca_earrings_gemstones :
  gemstones_needed 4 = 24 := by
  sorry

end NUMINAMATH_CALUDE_rebecca_earrings_gemstones_l2007_200728


namespace NUMINAMATH_CALUDE_equation_with_two_variables_degree_one_is_linear_l2007_200784

/-- Definition of a linear equation in two variables -/
def is_linear_equation_in_two_variables (f : ℝ → ℝ → ℝ) : Prop :=
  ∃ (a b : ℝ) (c : ℝ), ∀ (x y : ℝ), f x y = a * x + b * y + c

/-- Theorem stating that an equation with two variables and terms of degree 1 is a linear equation in two variables -/
theorem equation_with_two_variables_degree_one_is_linear 
  (f : ℝ → ℝ → ℝ) 
  (h1 : ∃ (x y : ℝ), f x y ≠ f 0 0) -- Condition: contains two variables
  (h2 : ∀ (x y : ℝ), ∃ (a b : ℝ) (c : ℝ), f x y = a * x + b * y + c) -- Condition: terms with variables are of degree 1
  : is_linear_equation_in_two_variables f :=
sorry

end NUMINAMATH_CALUDE_equation_with_two_variables_degree_one_is_linear_l2007_200784


namespace NUMINAMATH_CALUDE_problem_solving_probability_l2007_200795

theorem problem_solving_probability (prob_a prob_b : ℝ) 
  (h_a : prob_a = 1/2)
  (h_b : prob_b = 1/3) :
  (1 - prob_a) * (1 - prob_b) = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_problem_solving_probability_l2007_200795


namespace NUMINAMATH_CALUDE_color_film_fraction_l2007_200758

/-- Given a film festival selection process, prove the fraction of color films in the selection. -/
theorem color_film_fraction (x y : ℚ) (h1 : x > 0) (h2 : y > 0) : 
  let total_bw : ℚ := 40 * x
  let total_color : ℚ := 10 * y
  let selected_bw : ℚ := (y / x) * (total_bw / 100)
  let selected_color : ℚ := total_color
  let total_selected : ℚ := selected_bw + selected_color
  (selected_color / total_selected) = 25 / 26 := by
  sorry

end NUMINAMATH_CALUDE_color_film_fraction_l2007_200758


namespace NUMINAMATH_CALUDE_dress_discount_price_l2007_200717

theorem dress_discount_price (original_price discount_percentage : ℝ) 
  (h1 : original_price = 50)
  (h2 : discount_percentage = 30) : 
  original_price * (1 - discount_percentage / 100) = 35 := by
  sorry

end NUMINAMATH_CALUDE_dress_discount_price_l2007_200717


namespace NUMINAMATH_CALUDE_zero_location_l2007_200780

theorem zero_location (x y : ℝ) (h : x^5 < y^8 ∧ y^8 < y^3 ∧ y^3 < x^6) :
  x^5 < 0 ∧ 0 < y^8 := by
  sorry

end NUMINAMATH_CALUDE_zero_location_l2007_200780


namespace NUMINAMATH_CALUDE_vector_magnitude_solution_l2007_200794

/-- Given a vector a = (5, x) with magnitude 9, prove that x = 2√14 or x = -2√14 -/
theorem vector_magnitude_solution (x : ℝ) : 
  let a : ℝ × ℝ := (5, x)
  (‖a‖ = 9) → (x = 2 * Real.sqrt 14 ∨ x = -2 * Real.sqrt 14) := by
  sorry

end NUMINAMATH_CALUDE_vector_magnitude_solution_l2007_200794


namespace NUMINAMATH_CALUDE_total_distance_traveled_l2007_200730

def trip_duration : ℕ := 12
def speed1 : ℕ := 70
def time1 : ℕ := 3
def speed2 : ℕ := 80
def time2 : ℕ := 4
def speed3 : ℕ := 65
def time3 : ℕ := 3
def speed4 : ℕ := 90
def time4 : ℕ := 2

theorem total_distance_traveled :
  speed1 * time1 + speed2 * time2 + speed3 * time3 + speed4 * time4 = 905 :=
by
  sorry

#check total_distance_traveled

end NUMINAMATH_CALUDE_total_distance_traveled_l2007_200730


namespace NUMINAMATH_CALUDE_mollys_current_age_l2007_200757

/-- Represents the ages of Sandy and Molly -/
structure Ages where
  sandy : ℕ
  molly : ℕ

/-- The ratio of Sandy's age to Molly's age is 4:3 -/
def age_ratio (ages : Ages) : Prop :=
  4 * ages.molly = 3 * ages.sandy

/-- Sandy will be 42 years old in 6 years -/
def sandy_future_age (ages : Ages) : Prop :=
  ages.sandy + 6 = 42

theorem mollys_current_age (ages : Ages) :
  age_ratio ages → sandy_future_age ages → ages.molly = 27 := by
  sorry

end NUMINAMATH_CALUDE_mollys_current_age_l2007_200757


namespace NUMINAMATH_CALUDE_discount_amount_l2007_200738

theorem discount_amount (t_shirt_price backpack_price cap_price total_before_discount total_after_discount : ℕ) : 
  t_shirt_price = 30 →
  backpack_price = 10 →
  cap_price = 5 →
  total_before_discount = t_shirt_price + backpack_price + cap_price →
  total_after_discount = 43 →
  total_before_discount - total_after_discount = 2 := by
sorry

end NUMINAMATH_CALUDE_discount_amount_l2007_200738


namespace NUMINAMATH_CALUDE_chord_intersection_probability_is_one_twelfth_l2007_200708

/-- The number of points evenly spaced around the circle -/
def n : ℕ := 2020

/-- The probability that two randomly chosen chords intersect -/
def chord_intersection_probability : ℚ := 1 / 12

/-- Theorem stating that the probability of two randomly chosen chords intersecting is 1/12 -/
theorem chord_intersection_probability_is_one_twelfth :
  chord_intersection_probability = 1 / 12 := by sorry

end NUMINAMATH_CALUDE_chord_intersection_probability_is_one_twelfth_l2007_200708


namespace NUMINAMATH_CALUDE_skating_time_calculation_l2007_200713

/-- The number of days Gage skated for each duration -/
def days_per_duration : ℕ := 4

/-- The duration of skating in minutes for the first set of days -/
def duration1 : ℕ := 80

/-- The duration of skating in minutes for the second set of days -/
def duration2 : ℕ := 105

/-- The desired average skating time in minutes per day -/
def desired_average : ℕ := 100

/-- The total number of days, including the day to be calculated -/
def total_days : ℕ := 2 * days_per_duration + 1

/-- The required skating time on the last day to achieve the desired average -/
def required_time : ℕ := 160

theorem skating_time_calculation :
  (days_per_duration * duration1 + days_per_duration * duration2 + required_time) / total_days = desired_average := by
  sorry

end NUMINAMATH_CALUDE_skating_time_calculation_l2007_200713


namespace NUMINAMATH_CALUDE_soybean_price_l2007_200724

/-- Proves that the price of soybean is 20.5 given the conditions of the mixture problem -/
theorem soybean_price (peas_price : ℝ) (mixture_price : ℝ) (ratio : ℝ) :
  peas_price = 16 →
  ratio = 2 →
  mixture_price = 19 →
  (peas_price + ratio * (20.5 : ℝ)) / (1 + ratio) = mixture_price := by
sorry

end NUMINAMATH_CALUDE_soybean_price_l2007_200724


namespace NUMINAMATH_CALUDE_money_division_l2007_200756

theorem money_division (total : ℕ) (p q r : ℕ) : 
  p + q + r = total →
  3 * q = 7 * p →
  3 * r = 4 * q →
  q - p = 2800 →
  r - q = 3500 :=
by sorry

end NUMINAMATH_CALUDE_money_division_l2007_200756


namespace NUMINAMATH_CALUDE_debby_vacation_pictures_l2007_200709

/-- The number of pictures Debby took at the zoo -/
def zoo_pictures : ℕ := 24

/-- The number of pictures Debby took at the museum -/
def museum_pictures : ℕ := 12

/-- The number of pictures Debby deleted -/
def deleted_pictures : ℕ := 14

/-- The total number of pictures Debby took during her vacation -/
def total_pictures : ℕ := zoo_pictures + museum_pictures

/-- The number of pictures Debby still has from her vacation -/
def remaining_pictures : ℕ := total_pictures - deleted_pictures

theorem debby_vacation_pictures : remaining_pictures = 22 := by
  sorry

end NUMINAMATH_CALUDE_debby_vacation_pictures_l2007_200709


namespace NUMINAMATH_CALUDE_least_froods_to_drop_l2007_200763

def sum_first_n (n : ℕ) : ℕ := n * (n + 1) / 2

theorem least_froods_to_drop : ∃ n : ℕ, n > 0 ∧ sum_first_n n > 15 * n ∧ ∀ m : ℕ, m > 0 → m < n → sum_first_n m ≤ 15 * m :=
  sorry

end NUMINAMATH_CALUDE_least_froods_to_drop_l2007_200763


namespace NUMINAMATH_CALUDE_class_average_weight_l2007_200715

/-- Given two sections A and B in a class, prove that the average weight of the whole class is 38 kg -/
theorem class_average_weight (students_A : ℕ) (students_B : ℕ) (avg_weight_A : ℝ) (avg_weight_B : ℝ)
  (h1 : students_A = 30)
  (h2 : students_B = 20)
  (h3 : avg_weight_A = 40)
  (h4 : avg_weight_B = 35) :
  (students_A * avg_weight_A + students_B * avg_weight_B) / (students_A + students_B) = 38 := by
  sorry

end NUMINAMATH_CALUDE_class_average_weight_l2007_200715


namespace NUMINAMATH_CALUDE_unique_base_perfect_square_l2007_200769

theorem unique_base_perfect_square : 
  ∃! n : ℕ, 5 ≤ n ∧ n ≤ 15 ∧ ∃ m : ℕ, 2 * n^2 + 3 * n + 2 = m^2 := by
  sorry

end NUMINAMATH_CALUDE_unique_base_perfect_square_l2007_200769


namespace NUMINAMATH_CALUDE_cookie_cost_cookie_cost_is_65_l2007_200759

/-- The cost of a package of cookies, given the amount Diane has and the additional amount she needs. -/
theorem cookie_cost (diane_has : ℕ) (diane_needs : ℕ) : ℕ :=
  diane_has + diane_needs

/-- Proof that the cost of the cookies is 65 cents. -/
theorem cookie_cost_is_65 : cookie_cost 27 38 = 65 := by
  sorry

end NUMINAMATH_CALUDE_cookie_cost_cookie_cost_is_65_l2007_200759


namespace NUMINAMATH_CALUDE_rectangle_side_ratio_l2007_200768

/-- Represents the configuration of rectangles around a square -/
structure RectangleSquareArrangement where
  t : ℝ  -- Side length of the inner square
  a : ℝ  -- Shorter side of each rectangle
  b : ℝ  -- Longer side of each rectangle
  h_positive_t : t > 0
  h_positive_a : a > 0
  h_positive_b : b > 0
  h_outer_square : t + 2*a = t + b  -- Outer square side length condition
  h_area_ratio : (t + 2*a)^2 = 3 * t^2  -- Area ratio condition

/-- The theorem stating the ratio of rectangle sides -/
theorem rectangle_side_ratio 
  (arrange : RectangleSquareArrangement) : arrange.b / arrange.a = 2 := by
  sorry


end NUMINAMATH_CALUDE_rectangle_side_ratio_l2007_200768


namespace NUMINAMATH_CALUDE_water_depth_ratio_l2007_200783

theorem water_depth_ratio (dean_height : ℝ) (water_depth_difference : ℝ) :
  dean_height = 9 →
  water_depth_difference = 81 →
  (dean_height + water_depth_difference) / dean_height = 10 :=
by
  sorry

end NUMINAMATH_CALUDE_water_depth_ratio_l2007_200783


namespace NUMINAMATH_CALUDE_fruit_salad_count_l2007_200739

def total_fruit_salads (alaya_salads : ℕ) (angel_multiplier : ℕ) : ℕ :=
  alaya_salads + angel_multiplier * alaya_salads

theorem fruit_salad_count :
  total_fruit_salads 200 2 = 600 :=
by sorry

end NUMINAMATH_CALUDE_fruit_salad_count_l2007_200739


namespace NUMINAMATH_CALUDE_geometric_sequence_minimum_value_l2007_200740

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, q > 0 ∧ ∀ n : ℕ, a (n + 1) = q * a n

theorem geometric_sequence_minimum_value
  (a : ℕ → ℝ)
  (h_geom : is_geometric_sequence a)
  (h_positive : ∀ n, a n > 0)
  (h_exists : ∃ m n : ℕ, Real.sqrt (a m * a n) = 4 * a 1)
  (h_relation : a 6 = a 5 + 2 * a 4) :
  ∃ m n : ℕ, 1 / m + 4 / n = 3 / 2 ∧
    ∀ k l : ℕ, k > 0 ∧ l > 0 → 1 / k + 4 / l ≥ 3 / 2 :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_minimum_value_l2007_200740


namespace NUMINAMATH_CALUDE_sequence_properties_l2007_200760

def a (n : ℕ+) : ℚ := (9 * n^2 - 9 * n + 2) / (9 * n^2 - 1)

theorem sequence_properties :
  (a 10 = 28 / 31) ∧
  (∀ n : ℕ+, a n ≠ 99 / 100) ∧
  (∀ n : ℕ+, 0 < a n ∧ a n < 1) ∧
  (∃! n : ℕ+, 1 / 3 < a n ∧ a n < 2 / 3) :=
by sorry

end NUMINAMATH_CALUDE_sequence_properties_l2007_200760


namespace NUMINAMATH_CALUDE_vector_equality_implies_norm_equality_l2007_200791

variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

theorem vector_equality_implies_norm_equality (a b : V) (h : a ≠ 0 ∧ b ≠ 0) :
  a + 2 • b = 0 → ‖a - b‖ = ‖a‖ + ‖b‖ :=
sorry

end NUMINAMATH_CALUDE_vector_equality_implies_norm_equality_l2007_200791


namespace NUMINAMATH_CALUDE_sum_reciprocals_bound_l2007_200723

theorem sum_reciprocals_bound (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (sum_one : a + b + c = 1) :
  ∃ S : Set ℝ, S = { x | x ≥ 9 ∧ ∃ (a' b' c' : ℝ), a' > 0 ∧ b' > 0 ∧ c' > 0 ∧ 
    a' + b' + c' = 1 ∧ x = 1/a' + 1/b' + 1/c' } :=
by sorry

end NUMINAMATH_CALUDE_sum_reciprocals_bound_l2007_200723


namespace NUMINAMATH_CALUDE_arithmetic_mean_not_less_than_harmonic_mean_l2007_200775

theorem arithmetic_mean_not_less_than_harmonic_mean :
  ∃ (a b : ℝ), a > 0 ∧ b > 0 ∧ a ≠ b ∧ (a + b) / 2 ≥ 2 / (1/a + 1/b) :=
sorry

end NUMINAMATH_CALUDE_arithmetic_mean_not_less_than_harmonic_mean_l2007_200775


namespace NUMINAMATH_CALUDE_hcf_problem_l2007_200762

/-- Given two positive integers with HCF H and LCM (H * 13 * 14),
    where the larger number is 350, prove that H = 70 -/
theorem hcf_problem (a b : ℕ) (H : ℕ) (h1 : a > 0) (h2 : b > 0) (h3 : a = 350)
  (h4 : H = Nat.gcd a b) (h5 : Nat.lcm a b = H * 13 * 14) : H = 70 := by
  sorry

end NUMINAMATH_CALUDE_hcf_problem_l2007_200762


namespace NUMINAMATH_CALUDE_rectangular_garden_perimeter_l2007_200722

theorem rectangular_garden_perimeter 
  (x y : ℝ) 
  (diagonal_squared : x^2 + y^2 = 900)
  (area : x * y = 240) : 
  2 * (x + y) = 4 * Real.sqrt 345 := by
sorry

end NUMINAMATH_CALUDE_rectangular_garden_perimeter_l2007_200722


namespace NUMINAMATH_CALUDE_zigzag_angle_in_rectangle_l2007_200719

theorem zigzag_angle_in_rectangle (angle1 angle2 angle3 angle4 : ℝ) 
  (h1 : angle1 = 10)
  (h2 : angle2 = 14)
  (h3 : angle3 = 26)
  (h4 : angle4 = 33) :
  ∃ θ : ℝ, θ = 11 ∧ 
  (90 - angle1) + (90 - angle3) + θ = 180 ∧
  (180 - (90 - angle1) - angle2) + (180 - (90 - angle3) - angle4) + θ = 180 :=
by sorry

end NUMINAMATH_CALUDE_zigzag_angle_in_rectangle_l2007_200719


namespace NUMINAMATH_CALUDE_min_modulus_complex_l2007_200792

theorem min_modulus_complex (z : ℂ) : 
  (∃ x : ℝ, x^2 - 2*z*x + (3/4 : ℂ) + Complex.I = 0) → Complex.abs z ≥ 1 ∧ ∃ z₀ : ℂ, Complex.abs z₀ = 1 ∧ 
  (∃ x : ℝ, x^2 - 2*z₀*x + (3/4 : ℂ) + Complex.I = 0) := by
sorry

end NUMINAMATH_CALUDE_min_modulus_complex_l2007_200792


namespace NUMINAMATH_CALUDE_count_eight_to_800_l2007_200744

/-- Count of digit 8 in a single number -/
def count_eight (n : ℕ) : ℕ := sorry

/-- Sum of count_eight for all numbers from 1 to n -/
def sum_count_eight (n : ℕ) : ℕ := sorry

/-- The count of the digit 8 in all integers from 1 to 800 is 161 -/
theorem count_eight_to_800 : sum_count_eight 800 = 161 := by sorry

end NUMINAMATH_CALUDE_count_eight_to_800_l2007_200744


namespace NUMINAMATH_CALUDE_not_prime_polynomial_l2007_200746

theorem not_prime_polynomial (x y : ℤ) : 
  ¬ (Nat.Prime (x^8 - x^7*y + x^6*y^2 - x^5*y^3 + x^4*y^4 - x^3*y^5 + x^2*y^6 - x*y^7 + y^8).natAbs) :=
by sorry

end NUMINAMATH_CALUDE_not_prime_polynomial_l2007_200746


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l2007_200718

theorem sqrt_equation_solution : ∃! z : ℝ, Real.sqrt (10 + 3 * z) = 8 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l2007_200718


namespace NUMINAMATH_CALUDE_necessary_but_not_sufficient_l2007_200788

-- Define the condition |a-1| + |a| ≤ 1
def condition (a : ℝ) : Prop := abs (a - 1) + abs a ≤ 1

-- Define the property that y = a^x is decreasing on ℝ
def is_decreasing (a : ℝ) : Prop := ∀ x y : ℝ, x < y → a^x > a^y

theorem necessary_but_not_sufficient :
  (∀ a : ℝ, is_decreasing a → condition a) ∧
  (∃ a : ℝ, condition a ∧ ¬is_decreasing a) :=
sorry

end NUMINAMATH_CALUDE_necessary_but_not_sufficient_l2007_200788


namespace NUMINAMATH_CALUDE_reciprocal_power_l2007_200793

theorem reciprocal_power (a : ℚ) : 
  (a ≠ 0 ∧ a = 1 / a) → a^2014 = (1 : ℕ) :=
by
  sorry

end NUMINAMATH_CALUDE_reciprocal_power_l2007_200793


namespace NUMINAMATH_CALUDE_dice_probability_l2007_200727

/-- The number of sides on each die -/
def numSides : ℕ := 8

/-- The number of dice rolled -/
def numDice : ℕ := 8

/-- The number of sides showing numbers less than or equal to 4 -/
def favorableOutcomes : ℕ := 4

/-- The number of dice required to show numbers less than or equal to 4 -/
def requiredSuccesses : ℕ := 4

/-- The probability of rolling a number less than or equal to 4 on a single die -/
def singleDieProbability : ℚ := favorableOutcomes / numSides

theorem dice_probability :
  Nat.choose numDice requiredSuccesses *
  singleDieProbability ^ requiredSuccesses *
  (1 - singleDieProbability) ^ (numDice - requiredSuccesses) =
  35 / 128 :=
sorry

end NUMINAMATH_CALUDE_dice_probability_l2007_200727


namespace NUMINAMATH_CALUDE_cubic_root_sum_l2007_200711

theorem cubic_root_sum (u v w : ℝ) : 
  u^3 - 6*u^2 + 11*u - 6 = 0 →
  v^3 - 6*v^2 + 11*v - 6 = 0 →
  w^3 - 6*w^2 + 11*w - 6 = 0 →
  u * v / w + v * w / u + w * u / v = 49 / 6 := by
sorry

end NUMINAMATH_CALUDE_cubic_root_sum_l2007_200711


namespace NUMINAMATH_CALUDE_max_d_value_l2007_200707

def a (n : ℕ+) : ℕ := 100 + n^2 + 3*n

def d (n : ℕ+) : ℕ := Nat.gcd (a n) (a (n + 1))

theorem max_d_value :
  ∃ (m : ℕ+), d m = 13 ∧ ∀ (n : ℕ+), d n ≤ 13 :=
sorry

end NUMINAMATH_CALUDE_max_d_value_l2007_200707


namespace NUMINAMATH_CALUDE_bus_distance_l2007_200781

theorem bus_distance (total_distance : ℝ) (plane_fraction : ℝ) (train_bus_ratio : ℝ)
  (h1 : total_distance = 900)
  (h2 : plane_fraction = 1 / 3)
  (h3 : train_bus_ratio = 2 / 3) :
  let plane_distance := total_distance * plane_fraction
  let bus_distance := (total_distance - plane_distance) / (1 + train_bus_ratio)
  bus_distance = 360 := by
sorry

end NUMINAMATH_CALUDE_bus_distance_l2007_200781


namespace NUMINAMATH_CALUDE_village_population_equality_l2007_200741

-- Define the initial populations and known rate of decrease
def population_X : ℕ := 70000
def population_Y : ℕ := 42000
def decrease_rate_X : ℕ := 1200
def years : ℕ := 14

-- Define the unknown rate of increase for Village Y
def increase_rate_Y : ℕ := sorry

-- Theorem statement
theorem village_population_equality :
  population_X - years * decrease_rate_X = population_Y + years * increase_rate_Y ∧
  increase_rate_Y = 800 := by sorry

end NUMINAMATH_CALUDE_village_population_equality_l2007_200741


namespace NUMINAMATH_CALUDE_circle_area_through_points_l2007_200798

/-- The area of a circle with center P(2, -5) passing through Q(-7, 6) is 202π. -/
theorem circle_area_through_points :
  let P : ℝ × ℝ := (2, -5)
  let Q : ℝ × ℝ := (-7, 6)
  let r : ℝ := Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2)
  (π * r^2) = 202 * π := by sorry

end NUMINAMATH_CALUDE_circle_area_through_points_l2007_200798


namespace NUMINAMATH_CALUDE_solve_ticket_problem_l2007_200716

/-- Represents the cost of tickets and number of students for two teachers. -/
structure TicketInfo where
  student_price : ℕ
  adult_price : ℕ
  kadrnozka_students : ℕ
  hnizdo_students : ℕ

/-- Checks if the given TicketInfo satisfies all the problem conditions. -/
def satisfies_conditions (info : TicketInfo) : Prop :=
  info.adult_price > info.student_price ∧
  info.adult_price ≤ 2 * info.student_price ∧
  info.student_price * info.kadrnozka_students + info.adult_price = 994 ∧
  info.hnizdo_students = info.kadrnozka_students + 3 ∧
  info.student_price * info.hnizdo_students + info.adult_price = 1120

/-- Theorem stating the solution to the problem. -/
theorem solve_ticket_problem :
  ∃ (info : TicketInfo), satisfies_conditions info ∧ 
    info.hnizdo_students = 25 ∧ info.adult_price = 70 :=
by
  sorry


end NUMINAMATH_CALUDE_solve_ticket_problem_l2007_200716


namespace NUMINAMATH_CALUDE_x_value_proof_l2007_200774

theorem x_value_proof (x y : ℝ) (hx : x ≠ 0) (h1 : x/3 = y^2) (h2 : x/6 = 3*y) : x = 108 := by
  sorry

end NUMINAMATH_CALUDE_x_value_proof_l2007_200774


namespace NUMINAMATH_CALUDE_team_formation_count_l2007_200752

theorem team_formation_count (n : ℕ) (k : ℕ) (h1 : n = 30) (h2 : k = 3) :
  (Nat.choose (n - 1) (k - 1)) = 406 :=
by sorry

end NUMINAMATH_CALUDE_team_formation_count_l2007_200752


namespace NUMINAMATH_CALUDE_sqrt_difference_of_squares_l2007_200797

theorem sqrt_difference_of_squares : 
  (Real.sqrt 2023 + Real.sqrt 23) * (Real.sqrt 2023 - Real.sqrt 23) = 2000 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_difference_of_squares_l2007_200797


namespace NUMINAMATH_CALUDE_power_sum_inequality_l2007_200747

theorem power_sum_inequality (A B : ℝ) (n : ℕ+) (hA : A ≥ 0) (hB : B ≥ 0) :
  (A + B) ^ (n : ℕ) ≤ 2 ^ (n - 1 : ℕ) * (A ^ (n : ℕ) + B ^ (n : ℕ)) := by
  sorry

end NUMINAMATH_CALUDE_power_sum_inequality_l2007_200747


namespace NUMINAMATH_CALUDE_jons_payment_per_visit_l2007_200720

/-- Represents the payment structure for Jon's website -/
structure WebsitePayment where
  visits_per_hour : ℕ
  hours_per_day : ℕ
  days_per_month : ℕ
  monthly_revenue : ℚ

/-- Calculates the payment per visit given the website payment structure -/
def payment_per_visit (wp : WebsitePayment) : ℚ :=
  wp.monthly_revenue / (wp.visits_per_hour * wp.hours_per_day * wp.days_per_month)

/-- Theorem stating that Jon's payment per visit is $0.10 -/
theorem jons_payment_per_visit :
  let wp : WebsitePayment := {
    visits_per_hour := 50,
    hours_per_day := 24,
    days_per_month := 30,
    monthly_revenue := 3600
  }
  payment_per_visit wp = 1/10 := by
  sorry

end NUMINAMATH_CALUDE_jons_payment_per_visit_l2007_200720


namespace NUMINAMATH_CALUDE_exists_m_divisible_by_1988_l2007_200734

-- Define the function f
def f (x : ℤ) : ℤ := 3 * x + 2

-- Define the k-fold composition of f
def f_comp (k : ℕ) : ℤ → ℤ :=
  match k with
  | 0 => id
  | n + 1 => f ∘ (f_comp n)

-- State the theorem
theorem exists_m_divisible_by_1988 :
  ∃ m : ℕ+, (1988 : ℤ) ∣ (f_comp 100 m.val) :=
sorry

end NUMINAMATH_CALUDE_exists_m_divisible_by_1988_l2007_200734


namespace NUMINAMATH_CALUDE_coefficient_of_x3y5_l2007_200735

/-- The coefficient of x^3y^5 in the expansion of (2/3x - y/3)^8 -/
def coefficient : ℚ := -448/6561

/-- The binomial expansion of (a + b)^n -/
def binomial_expansion (a b : ℚ) (n : ℕ) (k : ℕ) : ℚ := 
  (n.choose k) * (a^(n-k)) * (b^k)

theorem coefficient_of_x3y5 :
  coefficient = binomial_expansion (2/3) (-1/3) 8 5 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_of_x3y5_l2007_200735


namespace NUMINAMATH_CALUDE_triangle_area_l2007_200736

/-- The area of a triangle composed of two right-angled triangles -/
theorem triangle_area (base1 height1 base2 height2 : ℝ) 
  (h1 : base1 = 1) (h2 : height1 = 1) 
  (h3 : base2 = 2) (h4 : height2 = 1) : 
  (1/2 * base1 * height1) + (1/2 * base2 * height2) = (3/2 : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_l2007_200736


namespace NUMINAMATH_CALUDE_sixth_member_income_l2007_200743

theorem sixth_member_income
  (family_size : ℕ)
  (average_income : ℕ)
  (income1 income2 income3 income4 income5 : ℕ)
  (h1 : family_size = 6)
  (h2 : average_income = 12000)
  (h3 : income1 = 11000)
  (h4 : income2 = 15000)
  (h5 : income3 = 10000)
  (h6 : income4 = 9000)
  (h7 : income5 = 13000) :
  average_income * family_size - (income1 + income2 + income3 + income4 + income5) = 14000 := by
  sorry

end NUMINAMATH_CALUDE_sixth_member_income_l2007_200743


namespace NUMINAMATH_CALUDE_T_value_for_K_9_l2007_200714

-- Define the equation T = 4hK + 2
def T (h K : ℝ) : ℝ := 4 * h * K + 2

-- State the theorem
theorem T_value_for_K_9 (h : ℝ) :
  (T h 7 = 58) → (T h 9 = 74) := by
  sorry

end NUMINAMATH_CALUDE_T_value_for_K_9_l2007_200714


namespace NUMINAMATH_CALUDE_equation_solution_l2007_200703

theorem equation_solution :
  ∃ x : ℚ, x - 1/2 = 7/8 - 2/3 ∧ x = 17/24 := by sorry

end NUMINAMATH_CALUDE_equation_solution_l2007_200703


namespace NUMINAMATH_CALUDE_gas_price_calculation_l2007_200755

theorem gas_price_calculation (expected_cash : ℝ) : 
  (12 * (expected_cash / 12) = 10 * (expected_cash / 12 + 0.3)) →
  expected_cash / 12 + 0.3 = 1.8 := by
  sorry

end NUMINAMATH_CALUDE_gas_price_calculation_l2007_200755


namespace NUMINAMATH_CALUDE_special_polynomial_f_one_l2007_200729

/-- A polynomial function satisfying a specific equation -/
def SpecialPolynomial (f : ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ,
    (∀ x : ℝ, f x = a * x^2 + b * x + c) ∧
    (∀ x : ℝ, x ≠ 0 → f (x - 1) + f x + f (x + 1) = (f x)^2 / (2027 * x))

/-- The theorem stating that for a special polynomial, f(1) must equal 6081 -/
theorem special_polynomial_f_one (f : ℝ → ℝ) (hf : SpecialPolynomial f) : f 1 = 6081 := by
  sorry

end NUMINAMATH_CALUDE_special_polynomial_f_one_l2007_200729


namespace NUMINAMATH_CALUDE_remaining_food_feeds_children_l2007_200787

/-- Represents the amount of food required for one adult. -/
def adult_meal : ℚ := 1

/-- Represents the amount of food required for one child. -/
def child_meal : ℚ := 7/9

/-- Represents the total amount of food available. -/
def total_food : ℚ := 70 * adult_meal

/-- Theorem stating that if 35 adults have their meal, the remaining food can feed 45 children. -/
theorem remaining_food_feeds_children : 
  total_food - 35 * adult_meal = 45 * child_meal := by
  sorry

#check remaining_food_feeds_children

end NUMINAMATH_CALUDE_remaining_food_feeds_children_l2007_200787


namespace NUMINAMATH_CALUDE_group_size_proof_l2007_200712

theorem group_size_proof (n : ℕ) (f m : ℕ) : 
  f = 8 → 
  m + f = n → 
  (n - f : ℚ) / n - (n - m : ℚ) / n = 36 / 100 → 
  n = 25 := by
sorry

end NUMINAMATH_CALUDE_group_size_proof_l2007_200712
