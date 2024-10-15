import Mathlib

namespace NUMINAMATH_CALUDE_mung_bean_germination_l86_8681

theorem mung_bean_germination 
  (germination_rate : ℝ) 
  (total_seeds : ℝ) 
  (h1 : germination_rate = 0.971) 
  (h2 : total_seeds = 1000) : 
  total_seeds * (1 - germination_rate) = 29 := by
sorry

end NUMINAMATH_CALUDE_mung_bean_germination_l86_8681


namespace NUMINAMATH_CALUDE_equilateral_triangle_area_l86_8654

/-- The area of an equilateral triangle, given specific internal perpendiculars -/
theorem equilateral_triangle_area (a b c : ℝ) (h : a = 2 ∧ b = 3 ∧ c = 4) : 
  ∃ (side : ℝ), 
    side > 0 ∧ 
    (a + b + c) * side / 2 = side * (side * Real.sqrt 3 / 2) / 2 ∧
    side * (side * Real.sqrt 3 / 2) / 2 = 27 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_equilateral_triangle_area_l86_8654


namespace NUMINAMATH_CALUDE_acute_angles_trigonometry_l86_8676

open Real

theorem acute_angles_trigonometry (α β : ℝ) 
  (h_acute_α : 0 < α ∧ α < π / 2)
  (h_acute_β : 0 < β ∧ β < π / 2)
  (h_tan_α : tan α = 2)
  (h_sin_diff : sin (α - β) = -sqrt 10 / 10) :
  sin (2 * α) = 4 / 5 ∧ tan (α + β) = -9 / 13 := by
sorry


end NUMINAMATH_CALUDE_acute_angles_trigonometry_l86_8676


namespace NUMINAMATH_CALUDE_replacement_cost_20_gyms_l86_8697

/-- The cost to replace all cardio machines in multiple gyms -/
def total_replacement_cost (num_gyms : ℕ) (bike_cost : ℕ) : ℕ :=
  let treadmill_cost : ℕ := (3 * bike_cost) / 2
  let elliptical_cost : ℕ := 2 * treadmill_cost
  let gym_cost : ℕ := 10 * bike_cost + 5 * treadmill_cost + 5 * elliptical_cost
  num_gyms * gym_cost

/-- Theorem stating the total replacement cost for 20 gyms -/
theorem replacement_cost_20_gyms :
  total_replacement_cost 20 700 = 455000 := by
  sorry

end NUMINAMATH_CALUDE_replacement_cost_20_gyms_l86_8697


namespace NUMINAMATH_CALUDE_smallest_three_digit_pq2r_l86_8699

theorem smallest_three_digit_pq2r : ∃ (p q r : ℕ), 
  Nat.Prime p ∧ Nat.Prime q ∧ Nat.Prime r ∧ 
  p ≠ q ∧ p ≠ r ∧ q ≠ r ∧
  126 = p * q^2 * r ∧
  (∀ (x p' q' r' : ℕ), 
    100 ≤ x ∧ x < 126 →
    Nat.Prime p' → Nat.Prime q' → Nat.Prime r' →
    p' ≠ q' → p' ≠ r' → q' ≠ r' →
    x ≠ p' * q'^2 * r') :=
by sorry

end NUMINAMATH_CALUDE_smallest_three_digit_pq2r_l86_8699


namespace NUMINAMATH_CALUDE_find_g_value_l86_8604

theorem find_g_value (x g : ℝ) (h1 : x = 0.3) 
  (h2 : (10 * x + 2) / 4 - (3 * x - 6) / 18 = (g * x + 4) / 3) : g = 2 := by
  sorry

end NUMINAMATH_CALUDE_find_g_value_l86_8604


namespace NUMINAMATH_CALUDE_unique_pair_l86_8605

def is_valid_pair (a b : ℕ) : Prop :=
  a ≥ 60 ∧ a < 70 ∧ b ≥ 60 ∧ b < 70 ∧ 
  a % 10 ≠ 6 ∧ b % 10 ≠ 6 ∧
  a * b = (10 * (a % 10) + 6) * (10 * (b % 10) + 6)

theorem unique_pair : 
  ∀ a b : ℕ, is_valid_pair a b → ((a = 69 ∧ b = 64) ∨ (a = 64 ∧ b = 69)) :=
by sorry

end NUMINAMATH_CALUDE_unique_pair_l86_8605


namespace NUMINAMATH_CALUDE_throwers_count_l86_8653

/-- Represents a football team with throwers and non-throwers (left-handed and right-handed) -/
structure FootballTeam where
  total_players : ℕ
  throwers : ℕ
  left_handed : ℕ
  right_handed : ℕ

/-- Conditions for the football team -/
def valid_team (team : FootballTeam) : Prop :=
  team.total_players = 70 ∧
  team.throwers > 0 ∧
  team.throwers + team.left_handed + team.right_handed = team.total_players ∧
  team.left_handed = (team.total_players - team.throwers) / 3 ∧
  team.right_handed = team.throwers + 2 * team.left_handed ∧
  team.throwers + team.right_handed = 60

/-- Theorem stating that a valid team has 40 throwers -/
theorem throwers_count (team : FootballTeam) (h : valid_team team) : team.throwers = 40 := by
  sorry

end NUMINAMATH_CALUDE_throwers_count_l86_8653


namespace NUMINAMATH_CALUDE_set_union_problem_l86_8675

theorem set_union_problem (a b : ℝ) :
  let M : Set ℝ := {a, b}
  let N : Set ℝ := {a + 1, 3}
  M ∩ N = {2} →
  M ∪ N = {1, 2, 3} := by
sorry

end NUMINAMATH_CALUDE_set_union_problem_l86_8675


namespace NUMINAMATH_CALUDE_min_teams_for_highest_score_fewer_wins_l86_8607

/-- Represents a soccer team --/
structure Team :=
  (id : ℕ)
  (wins : ℕ)
  (draws : ℕ)
  (losses : ℕ)

/-- Calculates the score of a team --/
def score (t : Team) : ℕ := 2 * t.wins + t.draws

/-- Represents a soccer tournament --/
structure Tournament :=
  (teams : List Team)
  (numTeams : ℕ)
  (allPlayedAgainstEachOther : Bool)

/-- Checks if a team has the highest score in the tournament --/
def hasHighestScore (t : Team) (tournament : Tournament) : Prop :=
  ∀ other : Team, other ∈ tournament.teams → score t ≥ score other

/-- Checks if a team has fewer wins than all other teams --/
def hasFewerWins (t : Team) (tournament : Tournament) : Prop :=
  ∀ other : Team, other ∈ tournament.teams → other.id ≠ t.id → t.wins < other.wins

theorem min_teams_for_highest_score_fewer_wins (n : ℕ) :
  (∃ tournament : Tournament,
    tournament.numTeams = n ∧
    tournament.allPlayedAgainstEachOther = true ∧
    (∃ t : Team, t ∈ tournament.teams ∧ 
      hasHighestScore t tournament ∧
      hasFewerWins t tournament)) →
  n ≥ 6 :=
sorry

end NUMINAMATH_CALUDE_min_teams_for_highest_score_fewer_wins_l86_8607


namespace NUMINAMATH_CALUDE_sanxingdui_jinsha_visitor_l86_8616

/-- Represents the four people in the problem -/
inductive Person : Type
  | A | B | C | D

/-- Represents the two archaeological sites -/
inductive Site : Type
  | Sanxingdui
  | Jinsha

/-- Predicate to represent if a person visited a site -/
def visited (p : Person) (s : Site) : Prop := sorry

/-- Predicate to represent if a person is telling the truth -/
def telling_truth (p : Person) : Prop := sorry

theorem sanxingdui_jinsha_visitor :
  (∃! p : Person, ∀ s : Site, visited p s) →
  (∃! p : Person, ¬telling_truth p) →
  (¬visited Person.A Site.Sanxingdui ∧ ¬visited Person.A Site.Jinsha) →
  (visited Person.B Site.Sanxingdui ↔ visited Person.A Site.Sanxingdui) →
  (visited Person.C Site.Jinsha ↔ visited Person.B Site.Jinsha) →
  (∀ s : Site, visited Person.D s → ¬visited Person.B s) →
  (∀ s : Site, visited Person.C s) :=
sorry

end NUMINAMATH_CALUDE_sanxingdui_jinsha_visitor_l86_8616


namespace NUMINAMATH_CALUDE_cubic_root_ratio_l86_8602

theorem cubic_root_ratio (p q r s : ℝ) (h : p ≠ 0) :
  (∀ x, p * x^3 + q * x^2 + r * x + s = 0 ↔ x = -1 ∨ x = 3 ∨ x = 4) →
  r / s = -5 / 12 := by
sorry

end NUMINAMATH_CALUDE_cubic_root_ratio_l86_8602


namespace NUMINAMATH_CALUDE_students_not_picked_l86_8655

theorem students_not_picked (total_students : ℕ) (num_groups : ℕ) (students_per_group : ℕ) 
  (h1 : total_students = 58)
  (h2 : num_groups = 8)
  (h3 : students_per_group = 6) : 
  total_students - (num_groups * students_per_group) = 10 := by
  sorry

end NUMINAMATH_CALUDE_students_not_picked_l86_8655


namespace NUMINAMATH_CALUDE_triangle_side_length_l86_8635

/-- Represents a triangle with given side lengths -/
structure Triangle where
  side1 : ℝ
  side2 : ℝ
  side3 : ℝ

/-- Calculates the perimeter of a triangle -/
def Triangle.perimeter (t : Triangle) : ℝ :=
  t.side1 + t.side2 + t.side3

theorem triangle_side_length 
  (t : Triangle) 
  (h_perimeter : t.perimeter = 160) 
  (h_side1 : t.side1 = 40) 
  (h_side3 : t.side3 = 70) : 
  t.side2 = 50 := by
sorry

end NUMINAMATH_CALUDE_triangle_side_length_l86_8635


namespace NUMINAMATH_CALUDE_workshop_analysis_l86_8683

/-- Workshop worker information -/
structure Workshop where
  total_workers : ℕ
  avg_salary : ℝ
  technicians : ℕ
  technician_salary : ℝ
  managers : ℕ
  manager_salary : ℝ
  assistant_salary : ℝ

/-- Theorem about workshop workers and salaries -/
theorem workshop_analysis (w : Workshop)
  (h_total : w.total_workers = 20)
  (h_avg : w.avg_salary = 8000)
  (h_tech : w.technicians = 7)
  (h_tech_salary : w.technician_salary = 12000)
  (h_man : w.managers = 5)
  (h_man_salary : w.manager_salary = 15000)
  (h_assist_salary : w.assistant_salary = 6000) :
  let assistants := w.total_workers - w.technicians - w.managers
  let tech_man_total := w.technicians * w.technician_salary + w.managers * w.manager_salary
  let tech_man_avg := tech_man_total / (w.technicians + w.managers : ℝ)
  assistants = 8 ∧ tech_man_avg = 13250 := by
  sorry


end NUMINAMATH_CALUDE_workshop_analysis_l86_8683


namespace NUMINAMATH_CALUDE_tangent_line_to_parabola_l86_8696

/-- A line is tangent to a parabola if and only if the discriminant of the resulting quadratic equation is zero. -/
axiom tangent_iff_discriminant_zero (a b c : ℝ) :
  (∃ x y : ℝ, y = a*x + b ∧ y^2 = c*x) →
  (∀ x y : ℝ, y = a*x + b → y^2 = c*x → (a*x + b)^2 = c*x) →
  b^2 = a*c

/-- The main theorem: if y = 3x + c is tangent to y^2 = 12x, then c = 1 -/
theorem tangent_line_to_parabola (c : ℝ) :
  (∃ x y : ℝ, y = 3*x + c ∧ y^2 = 12*x) →
  (∀ x y : ℝ, y = 3*x + c → y^2 = 12*x → (3*x + c)^2 = 12*x) →
  c = 1 := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_to_parabola_l86_8696


namespace NUMINAMATH_CALUDE_quadratic_inequality_bc_l86_8632

/-- Given a quadratic inequality x^2 + bx + c ≤ 0 with solution set [-2, 5], 
    prove that bc = 30 -/
theorem quadratic_inequality_bc (b c : ℝ) : 
  (∀ x, x^2 + b*x + c ≤ 0 ↔ -2 ≤ x ∧ x ≤ 5) → b*c = 30 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_bc_l86_8632


namespace NUMINAMATH_CALUDE_difference_of_squares_650_550_l86_8673

theorem difference_of_squares_650_550 : 650^2 - 550^2 = 120000 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_650_550_l86_8673


namespace NUMINAMATH_CALUDE_prob_no_adjacent_standing_ten_people_l86_8615

/-- Represents the number of valid arrangements for n people where no two adjacent people are standing. -/
def validArrangements : ℕ → ℕ
  | 0 => 1
  | 1 => 2
  | 2 => 3
  | n + 3 => validArrangements (n + 1) + validArrangements (n + 2)

/-- The number of people seated around the table. -/
def numPeople : ℕ := 10

/-- The total number of possible outcomes when flipping n fair coins. -/
def totalOutcomes (n : ℕ) : ℕ := 2^n

/-- The probability of no two adjacent people standing when n people flip fair coins. -/
def noAdjacentStandingProb (n : ℕ) : ℚ :=
  validArrangements n / totalOutcomes n

theorem prob_no_adjacent_standing_ten_people :
  noAdjacentStandingProb numPeople = 123 / 1024 := by
  sorry

#eval noAdjacentStandingProb numPeople

end NUMINAMATH_CALUDE_prob_no_adjacent_standing_ten_people_l86_8615


namespace NUMINAMATH_CALUDE_initial_depth_is_40_l86_8636

/-- Represents the work done by a group of workers digging to a certain depth -/
structure DiggingWork where
  workers : ℕ  -- number of workers
  hours   : ℕ  -- hours worked per day
  depth   : ℝ  -- depth dug in meters

/-- The theorem stating that given the initial and final conditions, the initial depth is 40 meters -/
theorem initial_depth_is_40 (initial final : DiggingWork) 
  (h1 : initial.workers = 45)
  (h2 : initial.hours = 8)
  (h3 : final.workers = initial.workers + 30)
  (h4 : final.hours = 6)
  (h5 : final.depth = 50)
  (h6 : initial.workers * initial.hours * initial.depth = final.workers * final.hours * final.depth) :
  initial.depth = 40 := by
  sorry

#check initial_depth_is_40

end NUMINAMATH_CALUDE_initial_depth_is_40_l86_8636


namespace NUMINAMATH_CALUDE_line_through_points_l86_8601

/-- A line passing through two points (1,2) and (5,14) has equation y = ax + b. This theorem proves that a - b = 4. -/
theorem line_through_points (a b : ℝ) : 
  (2 = a * 1 + b) → (14 = a * 5 + b) → a - b = 4 := by
  sorry

end NUMINAMATH_CALUDE_line_through_points_l86_8601


namespace NUMINAMATH_CALUDE_multiplicative_inverse_201_mod_299_l86_8665

theorem multiplicative_inverse_201_mod_299 :
  ∃! x : ℕ, x < 299 ∧ (201 * x) % 299 = 1 :=
by
  use 180
  sorry

end NUMINAMATH_CALUDE_multiplicative_inverse_201_mod_299_l86_8665


namespace NUMINAMATH_CALUDE_rachels_pizza_consumption_l86_8640

theorem rachels_pizza_consumption 
  (total_pizza : ℕ) 
  (bellas_pizza : ℕ) 
  (h1 : total_pizza = 952) 
  (h2 : bellas_pizza = 354) : 
  total_pizza - bellas_pizza = 598 := by
sorry

end NUMINAMATH_CALUDE_rachels_pizza_consumption_l86_8640


namespace NUMINAMATH_CALUDE_illumination_configurations_count_l86_8612

/-- The number of different ways to illuminate n traffic lights, each with three possible states. -/
def illumination_configurations (n : ℕ) : ℕ := 3^n

/-- Theorem stating that the number of different ways to illuminate n traffic lights,
    each with three possible states, is 3^n. -/
theorem illumination_configurations_count (n : ℕ) :
  illumination_configurations n = 3^n :=
by sorry

end NUMINAMATH_CALUDE_illumination_configurations_count_l86_8612


namespace NUMINAMATH_CALUDE_decreasing_linear_function_l86_8679

theorem decreasing_linear_function (x1 x2 : ℝ) (h : x2 > x1) : -6 * x2 < -6 * x1 := by
  sorry

end NUMINAMATH_CALUDE_decreasing_linear_function_l86_8679


namespace NUMINAMATH_CALUDE_hall_people_count_l86_8698

theorem hall_people_count (total_desks : ℕ) (occupied_desks : ℕ) (people : ℕ) : 
  total_desks = 72 →
  occupied_desks = 60 →
  people * 4 = occupied_desks * 5 →
  total_desks - occupied_desks = 12 →
  people = 75 := by
sorry

end NUMINAMATH_CALUDE_hall_people_count_l86_8698


namespace NUMINAMATH_CALUDE_inequality_proof_l86_8625

theorem inequality_proof (a b c k : ℕ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hk : k ≥ 1) :
  (a^(k+1) / b^k : ℚ) + (b^(k+1) / c^k : ℚ) + (c^(k+1) / a^k : ℚ) ≥ 
  (a^k / b^(k-1) : ℚ) + (b^k / c^(k-1) : ℚ) + (c^k / a^(k-1) : ℚ) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l86_8625


namespace NUMINAMATH_CALUDE_max_value_implies_t_equals_one_l86_8622

def f (t : ℝ) (x : ℝ) : ℝ := |x^2 - 2*x - t|

theorem max_value_implies_t_equals_one (t : ℝ) :
  (∀ x ∈ Set.Icc 0 3, f t x ≤ 2) ∧
  (∃ x ∈ Set.Icc 0 3, f t x = 2) →
  t = 1 := by
sorry

end NUMINAMATH_CALUDE_max_value_implies_t_equals_one_l86_8622


namespace NUMINAMATH_CALUDE_count_integers_eq_1278_l86_8618

/-- Recursive function to calculate the number of n-digit sequences with no consecutive 1's -/
def a : ℕ → ℕ
  | 0 => 1
  | 1 => 2
  | (n + 2) => a (n + 1) + a n

/-- The number of 12-digit positive integers with all digits either 1 or 2 and exactly two consecutive 1's -/
def count_integers : ℕ := 2 * a 10 + 9 * (2 * a 9)

/-- Theorem stating that the count of such integers is 1278 -/
theorem count_integers_eq_1278 : count_integers = 1278 := by sorry

end NUMINAMATH_CALUDE_count_integers_eq_1278_l86_8618


namespace NUMINAMATH_CALUDE_intersection_range_l86_8685

-- Define the points P and Q
def P : ℝ × ℝ := (-1, 1)
def Q : ℝ × ℝ := (2, 2)

-- Define the line equation
def line_equation (m : ℝ) (x y : ℝ) : Prop := x + m * y + m = 0

-- Define the line PQ
def line_PQ (x y : ℝ) : Prop := y = x + 2

-- Theorem statement
theorem intersection_range :
  ∀ m : ℝ, 
  (∃ x y : ℝ, x > 2 ∧ line_equation m x y ∧ line_PQ x y) ↔ 
  -3 < m ∧ m < 0 :=
sorry

end NUMINAMATH_CALUDE_intersection_range_l86_8685


namespace NUMINAMATH_CALUDE_whitney_max_sets_l86_8646

/-- Represents the number of items Whitney has --/
structure ItemCounts where
  tshirts : ℕ
  buttons : ℕ
  stickers : ℕ
  keychains : ℕ

/-- Represents the requirements for each set --/
structure SetRequirements where
  tshirts : ℕ
  buttonToStickerRatio : ℕ
  keychains : ℕ

/-- Calculates the maximum number of sets that can be made --/
def maxSets (items : ItemCounts) (reqs : SetRequirements) : ℕ :=
  min (items.tshirts / reqs.tshirts)
    (min (items.buttons / reqs.buttonToStickerRatio)
      (min (items.stickers)
        (items.keychains / reqs.keychains)))

/-- Theorem stating that the maximum number of sets Whitney can make is 7 --/
theorem whitney_max_sets :
  let items := ItemCounts.mk 7 36 15 21
  let reqs := SetRequirements.mk 1 4 3
  maxSets items reqs = 7 := by
  sorry


end NUMINAMATH_CALUDE_whitney_max_sets_l86_8646


namespace NUMINAMATH_CALUDE_quadratic_root_shift_l86_8670

theorem quadratic_root_shift (p q : ℝ) (x₁ x₂ : ℝ) : 
  (x₁^2 + p*x₁ + q = 0) ∧ (x₂^2 + p*x₂ + q = 0) →
  ((x₁ + 1)^2 + (p - 2)*(x₁ + 1) + (q - p + 1) = 0) ∧
  ((x₂ + 1)^2 + (p - 2)*(x₂ + 1) + (q - p + 1) = 0) := by
sorry

end NUMINAMATH_CALUDE_quadratic_root_shift_l86_8670


namespace NUMINAMATH_CALUDE_jane_final_score_l86_8603

/-- Calculates the final score in a card game --/
def final_score (rounds : ℕ) (points_per_win : ℕ) (points_lost : ℕ) : ℕ :=
  rounds * points_per_win - points_lost

/-- Theorem: Jane's final score in the card game --/
theorem jane_final_score :
  let rounds : ℕ := 8
  let points_per_win : ℕ := 10
  let points_lost : ℕ := 20
  final_score rounds points_per_win points_lost = 60 := by
  sorry


end NUMINAMATH_CALUDE_jane_final_score_l86_8603


namespace NUMINAMATH_CALUDE_find_number_l86_8628

theorem find_number : ∃! x : ℤ, x - 254 + 329 = 695 ∧ x = 620 := by sorry

end NUMINAMATH_CALUDE_find_number_l86_8628


namespace NUMINAMATH_CALUDE_quadratic_minimum_l86_8695

/-- The quadratic function f(x) = 3x^2 - 8x + 7 -/
def f (x : ℝ) : ℝ := 3 * x^2 - 8 * x + 7

/-- The derivative of f(x) -/
def f' (x : ℝ) : ℝ := 6 * x - 8

theorem quadratic_minimum :
  ∃ (x_min : ℝ), x_min = 4/3 ∧ ∀ (x : ℝ), f x ≥ f x_min :=
by
  sorry


end NUMINAMATH_CALUDE_quadratic_minimum_l86_8695


namespace NUMINAMATH_CALUDE_apple_expense_calculation_l86_8619

/-- Proves that the amount spent on apples is the difference between the total amount and the sum of other expenses and remaining money. -/
theorem apple_expense_calculation (total amount_oranges amount_candy amount_left : ℕ) 
  (h1 : total = 95)
  (h2 : amount_oranges = 14)
  (h3 : amount_candy = 6)
  (h4 : amount_left = 50) :
  total - (amount_oranges + amount_candy + amount_left) = 25 :=
by sorry

end NUMINAMATH_CALUDE_apple_expense_calculation_l86_8619


namespace NUMINAMATH_CALUDE_cubic_equation_solution_l86_8659

theorem cubic_equation_solution :
  ∃! x : ℝ, (2010 + x)^3 = -x^3 ∧ x = -1005 := by sorry

end NUMINAMATH_CALUDE_cubic_equation_solution_l86_8659


namespace NUMINAMATH_CALUDE_optimal_plan_l86_8691

/-- Represents the cost and quantity of new energy vehicles --/
structure VehiclePlan where
  costA : ℝ  -- Cost of A-type car in million yuan
  costB : ℝ  -- Cost of B-type car in million yuan
  quantA : ℕ -- Quantity of A-type cars
  quantB : ℕ -- Quantity of B-type cars

/-- Conditions for the vehicle purchase plan --/
def satisfiesConditions (plan : VehiclePlan) : Prop :=
  3 * plan.costA + plan.costB = 85 ∧
  2 * plan.costA + 4 * plan.costB = 140 ∧
  plan.quantA + plan.quantB = 15 ∧
  plan.quantA ≤ 2 * plan.quantB

/-- Total cost of the vehicle purchase plan --/
def totalCost (plan : VehiclePlan) : ℝ :=
  plan.costA * plan.quantA + plan.costB * plan.quantB

/-- Theorem stating the most cost-effective plan --/
theorem optimal_plan :
  ∃ (plan : VehiclePlan),
    satisfiesConditions plan ∧
    plan.costA = 20 ∧
    plan.costB = 25 ∧
    plan.quantA = 10 ∧
    plan.quantB = 5 ∧
    totalCost plan = 325 ∧
    (∀ (otherPlan : VehiclePlan),
      satisfiesConditions otherPlan →
      totalCost otherPlan ≥ totalCost plan) :=
by
  sorry


end NUMINAMATH_CALUDE_optimal_plan_l86_8691


namespace NUMINAMATH_CALUDE_expression_factorization_l86_8674

theorem expression_factorization (b : ℝ) :
  (8 * b^3 + 120 * b^2 - 14) - (9 * b^3 - 2 * b^2 + 14) = -1 * (b^3 - 122 * b^2 + 28) := by
  sorry

end NUMINAMATH_CALUDE_expression_factorization_l86_8674


namespace NUMINAMATH_CALUDE_cos_alpha_terminal_point_l86_8631

/-- Given a point P(-12, 5) on the terminal side of angle α, prove that cos α = -12/13 -/
theorem cos_alpha_terminal_point (α : Real) :
  let P : Real × Real := (-12, 5)
  (P.1 = -12 ∧ P.2 = 5) → -- Point P is (-12, 5)
  (P.1 = -12 * Real.cos α ∧ P.2 = -12 * Real.sin α) → -- P is on the terminal side of α
  Real.cos α = -12/13 := by
sorry

end NUMINAMATH_CALUDE_cos_alpha_terminal_point_l86_8631


namespace NUMINAMATH_CALUDE_lcm_hcf_problem_l86_8611

/-- Given two positive integers with specific LCM and HCF, prove that if one number is 210, the other is 517 -/
theorem lcm_hcf_problem (A B : ℕ+) (h1 : Nat.lcm A B = 2310) (h2 : Nat.gcd A B = 47) (h3 : A = 210) : B = 517 := by
  sorry

end NUMINAMATH_CALUDE_lcm_hcf_problem_l86_8611


namespace NUMINAMATH_CALUDE_solve_equation_l86_8609

theorem solve_equation (n : ℤ) (h : 8 + 6 = n + 8) : n = 6 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l86_8609


namespace NUMINAMATH_CALUDE_arithmetic_equality_l86_8652

theorem arithmetic_equality : 202 - 101 + 9 = 110 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_equality_l86_8652


namespace NUMINAMATH_CALUDE_arithmetic_calculation_l86_8656

theorem arithmetic_calculation : 
  4 * ((3.6 * 0.48 * 2.50) / (0.12 * 0.09 * 0.5)) = 3200 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculation_l86_8656


namespace NUMINAMATH_CALUDE_increasing_on_open_interval_l86_8684

open Real

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the derivative of f
variable (f' : ℝ → ℝ)

-- Assume f' is the derivative of f
variable (h : ∀ x, HasDerivAt f (f' x) x)

-- Theorem statement
theorem increasing_on_open_interval
  (h1 : ∀ x ∈ Set.Ioo 4 5, f' x > 0) :
  StrictMonoOn f (Set.Ioo 4 5) :=
sorry

end NUMINAMATH_CALUDE_increasing_on_open_interval_l86_8684


namespace NUMINAMATH_CALUDE_competition_participants_l86_8688

theorem competition_participants (freshmen : ℕ) (sophomores : ℕ) : 
  freshmen = 8 → sophomores = 5 * freshmen → freshmen + sophomores = 48 := by
sorry

end NUMINAMATH_CALUDE_competition_participants_l86_8688


namespace NUMINAMATH_CALUDE_min_value_problem_l86_8658

theorem min_value_problem (x y : ℝ) :
  (abs y ≤ 1) →
  (2 * x + y = 1) →
  (∀ x' y' : ℝ, abs y' ≤ 1 → 2 * x' + y' = 1 → 2 * x'^2 + 16 * x' + 3 * y'^2 ≥ 3) :=
by sorry

end NUMINAMATH_CALUDE_min_value_problem_l86_8658


namespace NUMINAMATH_CALUDE_min_sum_of_reciprocal_equation_l86_8680

theorem min_sum_of_reciprocal_equation : 
  ∃ (x y z : ℕ+), 
    (1 : ℝ) / x + 4 / y + 9 / z = 1 ∧ 
    x + y + z = 36 ∧ 
    ∀ (a b c : ℕ+), (1 : ℝ) / a + 4 / b + 9 / c = 1 → a + b + c ≥ 36 := by
  sorry

end NUMINAMATH_CALUDE_min_sum_of_reciprocal_equation_l86_8680


namespace NUMINAMATH_CALUDE_no_valid_score_l86_8639

/-- Represents a player in the hockey match -/
inductive Player
| Anton
| Ilya
| Sergey

/-- Represents the statements made by each player -/
def Statement : Type := Player → ℕ

/-- The statements made by Anton -/
def AntonStatement : Statement :=
  fun p => match p with
  | Player.Anton => 3
  | Player.Ilya => 1
  | Player.Sergey => 0

/-- The statements made by Ilya -/
def IlyaStatement : Statement :=
  fun p => match p with
  | Player.Anton => 0
  | Player.Ilya => 4
  | Player.Sergey => 5

/-- The statements made by Sergey -/
def SergeyStatement : Statement :=
  fun p => match p with
  | Player.Anton => 2
  | Player.Ilya => 0
  | Player.Sergey => 6

/-- Checks if a given score satisfies the conditions -/
def satisfiesConditions (score : Player → ℕ) : Prop :=
  (score Player.Anton + score Player.Ilya + score Player.Sergey = 10) ∧
  (∃ (p : Player), AntonStatement p = score p) ∧
  (∃ (p : Player), AntonStatement p ≠ score p) ∧
  (∃ (p : Player), IlyaStatement p = score p) ∧
  (∃ (p : Player), IlyaStatement p ≠ score p) ∧
  (∃ (p : Player), SergeyStatement p = score p) ∧
  (∃ (p : Player), SergeyStatement p ≠ score p)

/-- Theorem stating that no score satisfies all conditions -/
theorem no_valid_score : ¬∃ (score : Player → ℕ), satisfiesConditions score := by
  sorry


end NUMINAMATH_CALUDE_no_valid_score_l86_8639


namespace NUMINAMATH_CALUDE_find_certain_number_l86_8613

theorem find_certain_number : ∃ x : ℤ, x - 5 = 4 ∧ x = 9 := by
  sorry

end NUMINAMATH_CALUDE_find_certain_number_l86_8613


namespace NUMINAMATH_CALUDE_correct_stratified_sampling_l86_8637

/-- Represents the types of land in the farm --/
inductive LandType
  | Flat
  | Ditch
  | Sloped

/-- Represents the farm's land distribution --/
def farm : LandType → ℕ
  | LandType.Flat => 150
  | LandType.Ditch => 30
  | LandType.Sloped => 90

/-- Total acreage of the farm --/
def totalAcres : ℕ := farm LandType.Flat + farm LandType.Ditch + farm LandType.Sloped

/-- Sample size for the study --/
def sampleSize : ℕ := 18

/-- Calculates the sample size for each land type --/
def stratifiedSample (t : LandType) : ℕ :=
  (farm t * sampleSize) / totalAcres

/-- Theorem stating the correct stratified sampling for each land type --/
theorem correct_stratified_sampling :
  stratifiedSample LandType.Flat = 10 ∧
  stratifiedSample LandType.Ditch = 2 ∧
  stratifiedSample LandType.Sloped = 6 := by
  sorry


end NUMINAMATH_CALUDE_correct_stratified_sampling_l86_8637


namespace NUMINAMATH_CALUDE_negative_square_cubed_l86_8624

theorem negative_square_cubed (a : ℝ) : (-a^2)^3 = -a^6 := by
  sorry

end NUMINAMATH_CALUDE_negative_square_cubed_l86_8624


namespace NUMINAMATH_CALUDE_stating_max_fully_connected_is_N_minus_1_l86_8671

/-- Represents a network of computers. -/
structure Network where
  N : ℕ
  not_fully_connected : ∃ (node : Fin N), ∃ (other : Fin N), node ≠ other
  N_gt_3 : N > 3

/-- The maximum number of fully connected nodes in the network. -/
def max_fully_connected (net : Network) : ℕ := net.N - 1

/-- 
Theorem stating that the maximum number of fully connected nodes 
in a network with the given conditions is N-1.
-/
theorem max_fully_connected_is_N_minus_1 (net : Network) : 
  max_fully_connected net = net.N - 1 := by
  sorry


end NUMINAMATH_CALUDE_stating_max_fully_connected_is_N_minus_1_l86_8671


namespace NUMINAMATH_CALUDE_circle_area_tripled_l86_8606

theorem circle_area_tripled (r m : ℝ) (h : r > 0) (h' : m > 0) : 
  π * (r + m)^2 = 3 * (π * r^2) → r = (m * (1 + Real.sqrt 3)) / 2 := by
  sorry

end NUMINAMATH_CALUDE_circle_area_tripled_l86_8606


namespace NUMINAMATH_CALUDE_simplest_form_l86_8678

theorem simplest_form (a b : ℝ) (h : a ≠ b ∧ a ≠ -b) : 
  ¬∃ (f g : ℝ → ℝ → ℝ), ∀ (x y : ℝ), 
    (x^2 + y^2) / (x^2 - y^2) = f x y / g x y ∧ 
    (f x y ≠ x^2 + y^2 ∨ g x y ≠ x^2 - y^2) :=
sorry

end NUMINAMATH_CALUDE_simplest_form_l86_8678


namespace NUMINAMATH_CALUDE_solve_x_solve_y_solve_pqr_l86_8669

-- Define the structure of the diagram for parts (a) and (b)
structure Diagram :=
  (top_left : ℤ)
  (top_right : ℤ)
  (bottom : ℤ)
  (top_sum : ℤ)
  (left_sum : ℤ)
  (right_sum : ℤ)

-- Define the diagram for part (a)
def diagram_a : Diagram :=
  { top_left := 9,  -- This is derived from the given information
    top_right := 4,
    bottom := 1,    -- This is derived from the given information
    top_sum := 13,
    left_sum := 10,
    right_sum := 5  -- This is x, which we need to prove
  }

-- Define the diagram for part (b)
def diagram_b : Diagram :=
  { top_left := 24,  -- This is 3w, where w = 8
    top_right := 24, -- This is also 3w
    bottom := 8,     -- This is w
    top_sum := 48,
    left_sum := 32,  -- This is y, which we need to prove
    right_sum := 32  -- This is also y
  }

-- Theorem for part (a)
theorem solve_x (d : Diagram) : 
  d.top_left + d.top_right = d.top_sum ∧
  d.top_left + d.bottom = d.left_sum ∧
  d.bottom + d.top_right = d.right_sum →
  d.right_sum = 5 :=
sorry

-- Theorem for part (b)
theorem solve_y (d : Diagram) :
  d.top_left = d.top_right ∧
  d.top_left = 3 * d.bottom ∧
  d.top_left + d.top_right = d.top_sum ∧
  d.top_left + d.bottom = d.left_sum ∧
  d.left_sum = d.right_sum →
  d.left_sum = 32 :=
sorry

-- Theorem for part (c)
theorem solve_pqr (p q r : ℤ) :
  p + r = 3 ∧
  p + q = 18 ∧
  q + r = 13 →
  p = 4 ∧ q = 14 ∧ r = -1 :=
sorry

end NUMINAMATH_CALUDE_solve_x_solve_y_solve_pqr_l86_8669


namespace NUMINAMATH_CALUDE_car_speed_time_relation_l86_8687

/-- Represents a car with its speed and travel time -/
structure Car where
  speed : ℝ
  time : ℝ

/-- Theorem stating that if Car O travels at three times the speed of Car P for the same distance,
    then Car O's travel time is one-third of Car P's travel time -/
theorem car_speed_time_relation (p o : Car) (distance : ℝ) :
  o.speed = 3 * p.speed →
  distance = p.speed * p.time →
  distance = o.speed * o.time →
  o.time = p.time / 3 := by
  sorry


end NUMINAMATH_CALUDE_car_speed_time_relation_l86_8687


namespace NUMINAMATH_CALUDE_product_c_remaining_amount_l86_8617

/-- Calculate the remaining amount to be paid for a product -/
def remaining_amount (cost deposit discount_rate tax_rate : ℝ) : ℝ :=
  let discounted_price := cost * (1 - discount_rate)
  let total_price := discounted_price * (1 + tax_rate)
  total_price - deposit

/-- Theorem: The remaining amount to be paid for Product C is $3,610 -/
theorem product_c_remaining_amount :
  remaining_amount 3800 380 0 0.05 = 3610 := by
  sorry

end NUMINAMATH_CALUDE_product_c_remaining_amount_l86_8617


namespace NUMINAMATH_CALUDE_statement_b_false_statement_c_false_l86_8626

-- Define the ⋆ operation
def star (x y : ℝ) : ℝ := |x - y + 3|

-- Statement B is false
theorem statement_b_false :
  ¬ (∀ x y : ℝ, 3 * (star x y) = star (3 * x + 3) (3 * y + 3)) :=
sorry

-- Statement C is false
theorem statement_c_false :
  ¬ (∀ x : ℝ, star x (-3) = x) :=
sorry

end NUMINAMATH_CALUDE_statement_b_false_statement_c_false_l86_8626


namespace NUMINAMATH_CALUDE_complex_modulus_three_fourths_minus_three_i_l86_8621

theorem complex_modulus_three_fourths_minus_three_i :
  Complex.abs (3/4 - 3*I) = (3 * Real.sqrt 17) / 4 := by
  sorry

end NUMINAMATH_CALUDE_complex_modulus_three_fourths_minus_three_i_l86_8621


namespace NUMINAMATH_CALUDE_increasing_quadratic_function_m_bound_l86_8627

/-- Given that f(x) = -x^2 + mx is an increasing function on (-∞, 1], prove that m ≥ 2 -/
theorem increasing_quadratic_function_m_bound 
  (f : ℝ → ℝ) 
  (m : ℝ) 
  (h1 : ∀ x, f x = -x^2 + m*x) 
  (h2 : ∀ x y, x < y → x ≤ 1 → y ≤ 1 → f x < f y) : 
  m ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_increasing_quadratic_function_m_bound_l86_8627


namespace NUMINAMATH_CALUDE_vector_ratio_theorem_l86_8682

variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

theorem vector_ratio_theorem (a b : V) (ha : a ≠ 0) (hb : b ≠ 0) 
  (h1 : ‖a - b‖ = ‖a + 2 • b‖) 
  (h2 : inner a b / (‖a‖ * ‖b‖) = -1/4) : 
  ‖a‖ / ‖b‖ = 2 := by
  sorry

end NUMINAMATH_CALUDE_vector_ratio_theorem_l86_8682


namespace NUMINAMATH_CALUDE_f_neither_odd_nor_even_l86_8651

-- Define the function f
def f (x : ℝ) : ℝ := x^2 - 2

-- Define the domain of f
def domain : Set ℝ := Set.Ioc (-5) 5

-- Theorem statement
theorem f_neither_odd_nor_even :
  ¬(∀ x ∈ domain, f x = -f (-x)) ∧ ¬(∀ x ∈ domain, f x = f (-x)) :=
sorry

end NUMINAMATH_CALUDE_f_neither_odd_nor_even_l86_8651


namespace NUMINAMATH_CALUDE_four_white_possible_l86_8677

/-- Represents the state of the urn -/
structure UrnState :=
  (white : ℕ)
  (black : ℕ)

/-- Represents the possible operations on the urn -/
inductive Operation
  | removeFourBlackAddTwoBlack
  | removeThreeBlackOneWhiteAddOneBlackOneWhite
  | removeOneBlackThreeWhiteAddTwoWhite
  | removeFourWhiteAddTwoWhiteOneBlack

/-- Applies an operation to the urn state -/
def applyOperation (state : UrnState) (op : Operation) : UrnState :=
  match op with
  | Operation.removeFourBlackAddTwoBlack => 
      ⟨state.white, state.black - 2⟩
  | Operation.removeThreeBlackOneWhiteAddOneBlackOneWhite => 
      ⟨state.white, state.black - 2⟩
  | Operation.removeOneBlackThreeWhiteAddTwoWhite => 
      ⟨state.white - 1, state.black - 1⟩
  | Operation.removeFourWhiteAddTwoWhiteOneBlack => 
      ⟨state.white - 2, state.black + 1⟩

/-- The theorem to be proved -/
theorem four_white_possible : 
  ∃ (ops : List Operation), 
    let final_state := ops.foldl applyOperation ⟨150, 150⟩
    final_state.white = 4 :=
sorry

end NUMINAMATH_CALUDE_four_white_possible_l86_8677


namespace NUMINAMATH_CALUDE_rightmost_three_digits_of_7_to_1993_l86_8667

theorem rightmost_three_digits_of_7_to_1993 : 7^1993 % 1000 = 407 := by
  sorry

end NUMINAMATH_CALUDE_rightmost_three_digits_of_7_to_1993_l86_8667


namespace NUMINAMATH_CALUDE_system_solution_l86_8672

def solution_set : Set (ℝ × ℝ) := {(3, 2)}

theorem system_solution :
  {(x, y) : ℝ × ℝ | x + y = 5 ∧ x - y = 1} = solution_set :=
by sorry

end NUMINAMATH_CALUDE_system_solution_l86_8672


namespace NUMINAMATH_CALUDE_factorization_problem_1_factorization_problem_2_l86_8600

-- Problem 1
theorem factorization_problem_1 (x y : ℝ) :
  x * y^2 - 4 * x = x * (y + 2) * (y - 2) := by sorry

-- Problem 2
theorem factorization_problem_2 (x y : ℝ) :
  3 * x^2 - 12 * x * y + 12 * y^2 = 3 * (x - 2 * y)^2 := by sorry

end NUMINAMATH_CALUDE_factorization_problem_1_factorization_problem_2_l86_8600


namespace NUMINAMATH_CALUDE_power_equality_l86_8643

theorem power_equality (a b : ℝ) (h : (a - 2)^2 + |b + 1| = 0) : b^a = 1 := by
  sorry

end NUMINAMATH_CALUDE_power_equality_l86_8643


namespace NUMINAMATH_CALUDE_rachel_envelope_stuffing_l86_8693

/-- Rachel's envelope stuffing problem -/
theorem rachel_envelope_stuffing 
  (total_hours : ℕ) 
  (total_envelopes : ℕ) 
  (first_hour_envelopes : ℕ) 
  (h1 : total_hours = 8) 
  (h2 : total_envelopes = 1500) 
  (h3 : first_hour_envelopes = 135) :
  ∃ (second_hour_envelopes : ℕ),
    second_hour_envelopes = 195 ∧ 
    (total_envelopes - first_hour_envelopes - second_hour_envelopes) / (total_hours - 2) = 
    (total_envelopes - first_hour_envelopes - second_hour_envelopes) / (total_hours - 2) :=
by sorry


end NUMINAMATH_CALUDE_rachel_envelope_stuffing_l86_8693


namespace NUMINAMATH_CALUDE_b_and_d_know_grades_l86_8634

-- Define the grade types
inductive Grade
| Excellent
| Good

-- Define the students
inductive Student
| A
| B
| C
| D

-- Function to represent the actual grade of a student
def actualGrade : Student → Grade := sorry

-- Function to represent what grades a student can see
def canSee : Student → Student → Prop := sorry

-- Theorem statement
theorem b_and_d_know_grades :
  -- There are 2 excellent grades and 2 good grades
  (∃ (s1 s2 s3 s4 : Student), s1 ≠ s2 ∧ s1 ≠ s3 ∧ s1 ≠ s4 ∧ s2 ≠ s3 ∧ s2 ≠ s4 ∧ s3 ≠ s4 ∧
    actualGrade s1 = Grade.Excellent ∧ actualGrade s2 = Grade.Excellent ∧
    actualGrade s3 = Grade.Good ∧ actualGrade s4 = Grade.Good) →
  -- A, B, and C can see each other's grades
  (canSee Student.A Student.B ∧ canSee Student.A Student.C ∧
   canSee Student.B Student.A ∧ canSee Student.B Student.C ∧
   canSee Student.C Student.A ∧ canSee Student.C Student.B) →
  -- B and C can see each other's grades
  (canSee Student.B Student.C ∧ canSee Student.C Student.B) →
  -- D and A can see each other's grades
  (canSee Student.D Student.A ∧ canSee Student.A Student.D) →
  -- A doesn't know their own grade after seeing B and C's grades
  (∃ (g1 g2 : Grade), g1 ≠ g2 ∧
    ((actualGrade Student.B = g1 ∧ actualGrade Student.C = g2) ∨
     (actualGrade Student.B = g2 ∧ actualGrade Student.C = g1))) →
  -- B and D can know their own grades
  (∃ (gb gd : Grade),
    (actualGrade Student.B = gb ∧ ∀ g, actualGrade Student.B = g → g = gb) ∧
    (actualGrade Student.D = gd ∧ ∀ g, actualGrade Student.D = g → g = gd))
  := by sorry

end NUMINAMATH_CALUDE_b_and_d_know_grades_l86_8634


namespace NUMINAMATH_CALUDE_probability_of_white_ball_l86_8645

-- Define the number of red and white balls
def num_red_balls : ℕ := 3
def num_white_balls : ℕ := 2

-- Define the total number of balls
def total_balls : ℕ := num_red_balls + num_white_balls

-- Define the probability of drawing a white ball
def prob_white_ball : ℚ := num_white_balls / total_balls

-- Theorem statement
theorem probability_of_white_ball :
  prob_white_ball = 2 / 5 := by sorry

end NUMINAMATH_CALUDE_probability_of_white_ball_l86_8645


namespace NUMINAMATH_CALUDE_impossibleCubePlacement_l86_8610

/-- A type representing the vertices of a cube --/
inductive CubeVertex
| v1 | v2 | v3 | v4 | v5 | v6 | v7 | v8

/-- A function type representing a placement of numbers on the cube vertices --/
def CubePlacement := CubeVertex → Nat

/-- Predicate to check if two vertices are adjacent on a cube --/
def adjacent : CubeVertex → CubeVertex → Prop :=
  sorry

/-- Predicate to check if a number is in the valid range and not divisible by 13 --/
def validNumber (n : Nat) : Prop :=
  1 ≤ n ∧ n ≤ 245 ∧ n % 13 ≠ 0

/-- Predicate to check if two numbers have a common divisor greater than 1 --/
def hasCommonDivisor (a b : Nat) : Prop :=
  ∃ (d : Nat), d > 1 ∧ d ∣ a ∧ d ∣ b

theorem impossibleCubePlacement :
  ¬∃ (p : CubePlacement),
    (∀ v, validNumber (p v)) ∧
    (∀ v1 v2, v1 ≠ v2 → p v1 ≠ p v2) ∧
    (∀ v1 v2, adjacent v1 v2 → hasCommonDivisor (p v1) (p v2)) ∧
    (∀ v1 v2, ¬adjacent v1 v2 → ¬hasCommonDivisor (p v1) (p v2)) :=
by
  sorry


end NUMINAMATH_CALUDE_impossibleCubePlacement_l86_8610


namespace NUMINAMATH_CALUDE_square_of_negative_product_l86_8664

theorem square_of_negative_product (a b : ℝ) : (-a^2 * b)^2 = a^4 * b^2 := by
  sorry

end NUMINAMATH_CALUDE_square_of_negative_product_l86_8664


namespace NUMINAMATH_CALUDE_halfway_fraction_l86_8620

theorem halfway_fraction (a b : ℚ) (ha : a = 3/4) (hb : b = 5/6) :
  (a + b) / 2 = 19/24 := by
  sorry

end NUMINAMATH_CALUDE_halfway_fraction_l86_8620


namespace NUMINAMATH_CALUDE_expected_abs_difference_10_days_l86_8629

/-- Represents the outcome of a single day --/
inductive DailyOutcome
| CatWins
| FoxWins
| BothLose

/-- Probability distribution for daily outcomes --/
def dailyProbability (outcome : DailyOutcome) : ℝ :=
  match outcome with
  | DailyOutcome.CatWins => 0.25
  | DailyOutcome.FoxWins => 0.25
  | DailyOutcome.BothLose => 0.5

/-- Expected value of the absolute difference in wealth after n days --/
def expectedAbsDifference (n : ℕ) : ℝ :=
  sorry

/-- Theorem stating the expected absolute difference after 10 days is 1 --/
theorem expected_abs_difference_10_days :
  expectedAbsDifference 10 = 1 :=
sorry

end NUMINAMATH_CALUDE_expected_abs_difference_10_days_l86_8629


namespace NUMINAMATH_CALUDE_cube_volume_problem_l86_8633

theorem cube_volume_problem : ∃ (a : ℕ), 
  (a > 0) ∧ 
  (a^3 - ((a + 2) * (a - 2) * (a + 3)) = 7) ∧ 
  (a^3 = 27) := by
  sorry

end NUMINAMATH_CALUDE_cube_volume_problem_l86_8633


namespace NUMINAMATH_CALUDE_haploid_12_pairs_implies_tetraploid_l86_8630

/-- Represents the ploidy level of a plant -/
inductive Ploidy
  | Diploid
  | Triploid
  | Tetraploid
  | Hexaploid

/-- Represents a potato plant -/
structure PotatoPlant where
  ploidy : Ploidy

/-- Represents a haploid plant derived from anther culture -/
structure HaploidPlant where
  chromosomePairs : Nat

/-- Function to determine the ploidy of the original plant based on the haploid plant's chromosome pairs -/
def determinePloidy (haploid : HaploidPlant) : Ploidy :=
  if haploid.chromosomePairs = 12 then Ploidy.Tetraploid else Ploidy.Diploid

/-- Theorem stating that if a haploid plant derived from anther culture forms 12 chromosome pairs,
    then the original potato plant is tetraploid -/
theorem haploid_12_pairs_implies_tetraploid (haploid : HaploidPlant) (original : PotatoPlant) :
  haploid.chromosomePairs = 12 → original.ploidy = Ploidy.Tetraploid :=
by
  sorry


end NUMINAMATH_CALUDE_haploid_12_pairs_implies_tetraploid_l86_8630


namespace NUMINAMATH_CALUDE_complex_number_sum_l86_8650

theorem complex_number_sum (a b : ℝ) : 
  (Complex.I : ℂ)^5 * (Complex.I - 1) = Complex.mk a b → a + b = -2 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_sum_l86_8650


namespace NUMINAMATH_CALUDE_third_grade_sample_size_l86_8649

/-- Calculates the number of students to be sampled from a specific grade in stratified sampling -/
def stratified_sample_size (total_students : ℕ) (sample_size : ℕ) (grade_students : ℕ) : ℕ :=
  (grade_students * sample_size) / total_students

/-- Theorem: In a stratified sampling of 65 students from a high school with 1300 total students,
    where 500 students are in the third grade, the number of students to be sampled from the
    third grade is 25. -/
theorem third_grade_sample_size :
  stratified_sample_size 1300 65 500 = 25 := by
  sorry

#eval stratified_sample_size 1300 65 500

end NUMINAMATH_CALUDE_third_grade_sample_size_l86_8649


namespace NUMINAMATH_CALUDE_logarithm_simplification_l86_8660

theorem logarithm_simplification :
  (Real.log 2 / Real.log 6)^2 + (Real.log 2 / Real.log 6) * (Real.log 3 / Real.log 6) +
  2 * (Real.log 3 / Real.log 6) - 6^(Real.log 2 / Real.log 6) = -(Real.log 2 / Real.log 6) := by
  sorry

end NUMINAMATH_CALUDE_logarithm_simplification_l86_8660


namespace NUMINAMATH_CALUDE_farmer_cows_distribution_l86_8663

theorem farmer_cows_distribution (total : ℕ) : 
  (total : ℚ) / 3 + (total : ℚ) / 6 + (total : ℚ) / 8 + 15 = total → total = 40 := by
  sorry

end NUMINAMATH_CALUDE_farmer_cows_distribution_l86_8663


namespace NUMINAMATH_CALUDE_curve_self_intersection_l86_8638

/-- The x-coordinate of a point on the curve as a function of t -/
def x (t : ℝ) : ℝ := t^2 - 3

/-- The y-coordinate of a point on the curve as a function of t -/
def y (t : ℝ) : ℝ := t^3 - 6*t + 4

/-- The curve intersects itself if there exist two distinct real numbers that yield the same point -/
def self_intersection (a b : ℝ) : Prop :=
  a ≠ b ∧ x a = x b ∧ y a = y b

theorem curve_self_intersection :
  ∃ a b : ℝ, self_intersection a b ∧ x a = 3 ∧ y a = 4 :=
sorry

end NUMINAMATH_CALUDE_curve_self_intersection_l86_8638


namespace NUMINAMATH_CALUDE_charlie_extra_cost_l86_8648

/-- Charlie's cell phone plan and usage details -/
structure CellPhonePlan where
  included_data : ℕ
  extra_cost_per_gb : ℕ
  week1_usage : ℕ
  week2_usage : ℕ
  week3_usage : ℕ
  week4_usage : ℕ

/-- Calculate the extra cost for Charlie's cell phone usage -/
def calculate_extra_cost (plan : CellPhonePlan) : ℕ :=
  let total_usage := plan.week1_usage + plan.week2_usage + plan.week3_usage + plan.week4_usage
  let over_limit := if total_usage > plan.included_data then total_usage - plan.included_data else 0
  over_limit * plan.extra_cost_per_gb

/-- Theorem: Charlie's extra cost is $120.00 -/
theorem charlie_extra_cost :
  let charlie_plan : CellPhonePlan := {
    included_data := 8,
    extra_cost_per_gb := 10,
    week1_usage := 2,
    week2_usage := 3,
    week3_usage := 5,
    week4_usage := 10
  }
  calculate_extra_cost charlie_plan = 120 := by
  sorry

end NUMINAMATH_CALUDE_charlie_extra_cost_l86_8648


namespace NUMINAMATH_CALUDE_next_term_is_512x4_l86_8689

def geometric_sequence (x : ℝ) : ℕ → ℝ
  | 0 => 2
  | 1 => 8 * x
  | 2 => 32 * x^2
  | 3 => 128 * x^3
  | (n + 4) => geometric_sequence x n

theorem next_term_is_512x4 (x : ℝ) : geometric_sequence x 4 = 512 * x^4 := by
  sorry

end NUMINAMATH_CALUDE_next_term_is_512x4_l86_8689


namespace NUMINAMATH_CALUDE_average_side_length_of_squares_l86_8668

theorem average_side_length_of_squares (a₁ a₂ a₃ : ℝ) 
  (h₁ : a₁ = 25) (h₂ : a₂ = 64) (h₃ : a₃ = 144) : 
  (Real.sqrt a₁ + Real.sqrt a₂ + Real.sqrt a₃) / 3 = 25 / 3 := by
  sorry

end NUMINAMATH_CALUDE_average_side_length_of_squares_l86_8668


namespace NUMINAMATH_CALUDE_contrapositive_example_l86_8666

theorem contrapositive_example : 
  (∀ x : ℝ, x > 2 → x > 1) ↔ (∀ x : ℝ, x ≤ 1 → x ≤ 2) := by
  sorry

end NUMINAMATH_CALUDE_contrapositive_example_l86_8666


namespace NUMINAMATH_CALUDE_sum_cubes_minus_product_l86_8662

theorem sum_cubes_minus_product (a b c d : ℝ) 
  (sum_eq : a + b + c + d = 15)
  (sum_products_eq : a*b + a*c + a*d + b*c + b*d + c*d = 40) :
  a^3 + b^3 + c^3 + d^3 - 3*a*b*c*d = 1695 := by
  sorry

end NUMINAMATH_CALUDE_sum_cubes_minus_product_l86_8662


namespace NUMINAMATH_CALUDE_slips_with_three_l86_8661

/-- Given a bag with 15 slips, each having either 3 or 9, prove that if the expected value
    of a randomly drawn slip is 5, then 10 slips have 3 on them. -/
theorem slips_with_three (total : ℕ) (value_a value_b : ℕ) (expected : ℚ) : 
  total = 15 →
  value_a = 3 →
  value_b = 9 →
  expected = 5 →
  ∃ (count_a : ℕ), 
    count_a ≤ total ∧
    (count_a : ℚ) / total * value_a + (total - count_a : ℚ) / total * value_b = expected ∧
    count_a = 10 :=
by sorry

end NUMINAMATH_CALUDE_slips_with_three_l86_8661


namespace NUMINAMATH_CALUDE_sum_and_sum_squares_bound_equality_conditions_l86_8641

theorem sum_and_sum_squares_bound (a b c : ℝ) 
  (h1 : a ≥ -1) (h2 : b ≥ -1) (h3 : c ≥ -1)
  (h4 : a^3 + b^3 + c^3 = 1) :
  a + b + c + a^2 + b^2 + c^2 ≤ 4 := by
  sorry

theorem equality_conditions (a b c : ℝ) 
  (h1 : a ≥ -1) (h2 : b ≥ -1) (h3 : c ≥ -1)
  (h4 : a^3 + b^3 + c^3 = 1) :
  (a + b + c + a^2 + b^2 + c^2 = 4) ↔ 
  ((a, b, c) = (1, 1, -1) ∨ (a, b, c) = (1, -1, 1) ∨ (a, b, c) = (-1, 1, 1)) := by
  sorry

end NUMINAMATH_CALUDE_sum_and_sum_squares_bound_equality_conditions_l86_8641


namespace NUMINAMATH_CALUDE_video_game_lives_l86_8644

theorem video_game_lives (initial_lives next_level_lives total_lives : ℝ) 
  (h1 : initial_lives = 43.0)
  (h2 : next_level_lives = 27.0)
  (h3 : total_lives = 84) :
  ∃ hard_part_lives : ℝ, 
    hard_part_lives = 14.0 ∧ 
    initial_lives + hard_part_lives + next_level_lives = total_lives :=
by sorry

end NUMINAMATH_CALUDE_video_game_lives_l86_8644


namespace NUMINAMATH_CALUDE_juniper_bones_theorem_l86_8608

/-- Represents the number of bones Juniper has -/
def juniper_bones (initial : ℕ) (x : ℕ) (y : ℕ) : ℕ :=
  initial + x - y

theorem juniper_bones_theorem (x : ℕ) (y : ℕ) :
  juniper_bones 4 x y = 8 - y :=
by
  sorry

#check juniper_bones_theorem

end NUMINAMATH_CALUDE_juniper_bones_theorem_l86_8608


namespace NUMINAMATH_CALUDE_wedge_volume_l86_8692

/-- The volume of a wedge cut from a cylindrical log --/
theorem wedge_volume (d : ℝ) (θ : ℝ) (h : d = 16 ∧ θ = 30 * π / 180) : 
  let r := d / 2
  let v := (r^2 * d * π) / 4
  v = 256 * π := by sorry

end NUMINAMATH_CALUDE_wedge_volume_l86_8692


namespace NUMINAMATH_CALUDE_hyperbola_properties_l86_8642

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x^2 / 4 - y^2 / 12 = 1

-- Define eccentricity
def eccentricity (e : ℝ) : Prop := e = 2

-- Define asymptotes
def asymptotes (x y : ℝ) : Prop := y = Real.sqrt 3 * x ∨ y = -Real.sqrt 3 * x

theorem hyperbola_properties :
  ∃ (e : ℝ), eccentricity e ∧
  ∀ (x y : ℝ), hyperbola x y → asymptotes x y :=
sorry

end NUMINAMATH_CALUDE_hyperbola_properties_l86_8642


namespace NUMINAMATH_CALUDE_cube_root_of_negative_eight_l86_8623

theorem cube_root_of_negative_eight (x : ℝ) : x^3 = -8 ↔ x = -2 := by
  sorry

end NUMINAMATH_CALUDE_cube_root_of_negative_eight_l86_8623


namespace NUMINAMATH_CALUDE_sqrt_difference_l86_8694

theorem sqrt_difference (a b : ℝ) (ha : a < 0) (hb : b < 0) (hab : a - b = 6) :
  Real.sqrt (a^2) - Real.sqrt (b^2) = -6 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_difference_l86_8694


namespace NUMINAMATH_CALUDE_parallel_line_through_point_l86_8690

/-- A line parallel to y = 1/2x - 1 passing through (0, 3) has equation y = 1/2x + 3 -/
theorem parallel_line_through_point (k b : ℝ) : 
  (∀ x y : ℝ, y = k * x + b) →  -- The line has equation y = kx + b
  k = 1/2 →                    -- The line is parallel to y = 1/2x - 1
  3 = b →                      -- The line passes through (0, 3)
  ∀ x y : ℝ, y = 1/2 * x + 3   -- The equation of the line is y = 1/2x + 3
:= by sorry

end NUMINAMATH_CALUDE_parallel_line_through_point_l86_8690


namespace NUMINAMATH_CALUDE_smallest_total_books_l86_8657

/-- Represents the number of books for each subject -/
structure BookCounts where
  physics : ℕ
  chemistry : ℕ
  biology : ℕ

/-- Checks if the given book counts satisfy the required ratios -/
def satisfiesRatios (books : BookCounts) : Prop :=
  3 * books.chemistry = 2 * books.physics ∧
  4 * books.biology = 3 * books.chemistry

/-- Calculates the total number of books -/
def totalBooks (books : BookCounts) : ℕ :=
  books.physics + books.chemistry + books.biology

/-- Theorem stating the smallest possible total number of books -/
theorem smallest_total_books :
  ∃ (books : BookCounts),
    satisfiesRatios books ∧
    totalBooks books > 3000 ∧
    ∀ (other : BookCounts),
      satisfiesRatios other → totalBooks other > 3000 →
      totalBooks books ≤ totalBooks other :=
sorry

end NUMINAMATH_CALUDE_smallest_total_books_l86_8657


namespace NUMINAMATH_CALUDE_min_ab_for_line_through_point_l86_8647

/-- Given a line equation (x/a) + (y/b) = 1 where a > 0 and b > 0,
    and the line passes through the point (1,1),
    the minimum value of ab is 4. -/
theorem min_ab_for_line_through_point (a b : ℝ) : 
  a > 0 → b > 0 → (1 / a + 1 / b = 1) → (∀ x y : ℝ, x / a + y / b = 1 → (x, y) = (1, 1)) → 
  ∀ c d : ℝ, c > 0 → d > 0 → (1 / c + 1 / d = 1) → c * d ≥ 4 := by
  sorry

#check min_ab_for_line_through_point

end NUMINAMATH_CALUDE_min_ab_for_line_through_point_l86_8647


namespace NUMINAMATH_CALUDE_word_count_between_czyeb_and_xceda_l86_8686

/-- Represents the set of available letters --/
inductive Letter : Type
  | A | B | C | D | E | X | Y | Z

/-- A word is a list of 5 letters --/
def Word := List Letter

/-- Convert a letter to its corresponding digit in base 8 --/
def letterToDigit (l : Letter) : Nat :=
  match l with
  | Letter.A => 0
  | Letter.B => 1
  | Letter.C => 2
  | Letter.D => 3
  | Letter.E => 4
  | Letter.X => 5
  | Letter.Y => 6
  | Letter.Z => 7

/-- Convert a word to its corresponding number in base 8 --/
def wordToNumber (w : Word) : Nat :=
  w.foldl (fun acc l => acc * 8 + letterToDigit l) 0

/-- The word CZYEB --/
def czyeb : Word := [Letter.C, Letter.Z, Letter.Y, Letter.E, Letter.B]

/-- The word XCEDA --/
def xceda : Word := [Letter.X, Letter.C, Letter.E, Letter.D, Letter.A]

/-- The theorem to be proved --/
theorem word_count_between_czyeb_and_xceda :
  (wordToNumber xceda) - (wordToNumber czyeb) - 1 = 9590 := by
  sorry

end NUMINAMATH_CALUDE_word_count_between_czyeb_and_xceda_l86_8686


namespace NUMINAMATH_CALUDE_f1_properties_f2_properties_f3_properties_f4_properties_l86_8614

-- Function 1: y = 4 - x^2 for |x| ≤ 2
def f1 (x : ℝ) := 4 - x^2

-- Function 2: y = 0.5(x^2 + x|x| + 4)
def f2 (x : ℝ) := 0.5 * (x^2 + x * |x| + 4)

-- Function 3: y = (x^3 - x) / |x|
noncomputable def f3 (x : ℝ) := (x^3 - x) / |x|

-- Function 4: y = (x - 2)|x|
def f4 (x : ℝ) := (x - 2) * |x|

-- Theorem for function 1
theorem f1_properties (x : ℝ) (h : |x| ≤ 2) :
  f1 x ≤ 4 ∧ f1 0 = 4 ∧ f1 2 = f1 (-2) := by sorry

-- Theorem for function 2
theorem f2_properties (x : ℝ) :
  (x ≥ 0 → f2 x = x^2 + 2) ∧ (x < 0 → f2 x = 2) := by sorry

-- Theorem for function 3
theorem f3_properties (x : ℝ) (h : x ≠ 0) :
  (x > 0 → f3 x = x^2 - 1) ∧ (x < 0 → f3 x = -x^2 + 1) := by sorry

-- Theorem for function 4
theorem f4_properties (x : ℝ) :
  (x ≥ 0 → f4 x = x^2 - 2*x) ∧ (x < 0 → f4 x = -x^2 + 2*x) := by sorry

end NUMINAMATH_CALUDE_f1_properties_f2_properties_f3_properties_f4_properties_l86_8614
