import Mathlib

namespace complex_number_in_fourth_quadrant_l601_60121

def complex_number_quadrant (z : ℂ) : Prop :=
  Real.sign (z.re) = 1 ∧ Real.sign (z.im) = -1

theorem complex_number_in_fourth_quadrant :
  let z : ℂ := (3 + 2 * Complex.I) / Complex.I
  complex_number_quadrant z :=
by sorry

end complex_number_in_fourth_quadrant_l601_60121


namespace negation_equivalence_l601_60114

theorem negation_equivalence (x : ℝ) :
  ¬(2 < x ∧ x < 5 → x^2 - 7*x + 10 < 0) ↔
  (x ≤ 2 ∨ x ≥ 5 → x^2 - 7*x + 10 ≥ 0) :=
by sorry

end negation_equivalence_l601_60114


namespace perpendicular_vectors_x_value_l601_60177

/-- Given two 2D vectors a and b, where a = (x-1, 2) and b = (2, 1),
    if a is perpendicular to b, then x = 0. -/
theorem perpendicular_vectors_x_value (x : ℝ) :
  let a : ℝ × ℝ := (x - 1, 2)
  let b : ℝ × ℝ := (2, 1)
  (a.1 * b.1 + a.2 * b.2 = 0) → x = 0 := by
sorry

end perpendicular_vectors_x_value_l601_60177


namespace apples_remaining_l601_60118

/-- The number of apples left after picking and eating -/
def applesLeft (mikeApples nancyApples keithApples : Float) : Float :=
  mikeApples + nancyApples - keithApples

theorem apples_remaining :
  applesLeft 7.0 3.0 6.0 = 4.0 := by
  sorry

#eval applesLeft 7.0 3.0 6.0

end apples_remaining_l601_60118


namespace regular_survey_rate_l601_60128

/-- Proves that the regular rate for completing a survey is 10 given the specified conditions. -/
theorem regular_survey_rate (total_surveys : ℕ) (cellphone_surveys : ℕ) (total_earnings : ℚ) :
  total_surveys = 100 →
  cellphone_surveys = 60 →
  total_earnings = 1180 →
  ∃ (regular_rate : ℚ),
    regular_rate * (total_surveys - cellphone_surveys) +
    (regular_rate * 1.3) * cellphone_surveys = total_earnings ∧
    regular_rate = 10 := by
  sorry

end regular_survey_rate_l601_60128


namespace collinear_vectors_x_value_l601_60184

/-- Two vectors are collinear if their cross product is zero -/
def collinear (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 - a.2 * b.1 = 0

/-- Given vectors a and b, if they are collinear, then x = 3 -/
theorem collinear_vectors_x_value :
  let a : ℝ × ℝ := (1, -5)
  let b : ℝ × ℝ := (x - 1, -10)
  collinear a b → x = 3 := by
    sorry


end collinear_vectors_x_value_l601_60184


namespace charles_speed_l601_60182

/-- Charles' stroll scenario -/
def charles_stroll (distance : ℝ) (time : ℝ) : Prop :=
  distance = 6 ∧ time = 2 ∧ distance / time = 3

theorem charles_speed : ∃ (distance time : ℝ), charles_stroll distance time :=
  sorry

end charles_speed_l601_60182


namespace last_two_digits_squared_l601_60169

theorem last_two_digits_squared (n : ℤ) : 
  (n * 402 * 503 * 604 * 646 * 547 * 448 * 349) ^ 2 % 100 = 76 := by
  sorry

end last_two_digits_squared_l601_60169


namespace remainder_2503_div_28_l601_60155

theorem remainder_2503_div_28 : 2503 % 28 = 11 := by
  sorry

end remainder_2503_div_28_l601_60155


namespace line_slope_is_two_l601_60131

/-- Given a line ax + y - 4 = 0 passing through the point (-1, 2), prove that its slope is 2 -/
theorem line_slope_is_two (a : ℝ) : 
  (a * (-1) + 2 - 4 = 0) → -- Line passes through (-1, 2)
  (∃ m b : ℝ, ∀ x y : ℝ, a * x + y - 4 = 0 ↔ y = m * x + b) → -- Line can be written in slope-intercept form
  (∃ m : ℝ, ∀ x y : ℝ, a * x + y - 4 = 0 ↔ y = m * x + 4) → -- Specific y-intercept is 4
  (∃ m : ℝ, ∀ x y : ℝ, a * x + y - 4 = 0 ↔ y = 2 * x + 4) -- Slope is 2
  := by sorry

end line_slope_is_two_l601_60131


namespace stating_saucepan_capacity_l601_60191

/-- Represents a cylindrical saucepan with a volume scale in cups. -/
structure Saucepan where
  capacity : ℝ
  partialFill : ℝ
  partialVolume : ℝ

/-- 
Theorem stating that a saucepan's capacity is 125 cups when 28% of it contains 35 cups.
-/
theorem saucepan_capacity (s : Saucepan) 
  (h1 : s.partialFill = 0.28)
  (h2 : s.partialVolume = 35) :
  s.capacity = 125 := by
  sorry

#check saucepan_capacity

end stating_saucepan_capacity_l601_60191


namespace project_completion_proof_l601_60144

/-- Represents the number of days to complete the project -/
def project_completion_time : ℕ := 11

/-- Represents A's completion rate per day -/
def rate_A : ℚ := 1 / 20

/-- Represents B's initial completion rate per day -/
def rate_B : ℚ := 1 / 30

/-- Represents C's completion rate per day -/
def rate_C : ℚ := 1 / 40

/-- Represents B's doubled completion rate -/
def rate_B_doubled : ℚ := 2 * rate_B

/-- Represents the time A quits before project completion -/
def time_A_quits_before : ℕ := 10

/-- Theorem stating that the project will be completed in 11 days -/
theorem project_completion_proof :
  let total_work : ℚ := 1
  let combined_rate : ℚ := rate_A + rate_B + rate_C
  let final_rate : ℚ := rate_B_doubled + rate_C
  (project_completion_time - time_A_quits_before) * combined_rate +
  time_A_quits_before * final_rate = total_work :=
by sorry


end project_completion_proof_l601_60144


namespace budgets_equal_after_6_years_l601_60164

def initial_budget_Q : ℕ := 540000
def initial_budget_V : ℕ := 780000
def annual_increase_Q : ℕ := 30000
def annual_decrease_V : ℕ := 10000

def budget_Q (years : ℕ) : ℕ := initial_budget_Q + annual_increase_Q * years
def budget_V (years : ℕ) : ℕ := initial_budget_V - annual_decrease_V * years

theorem budgets_equal_after_6_years :
  ∃ (years : ℕ), years = 6 ∧ budget_Q years = budget_V years :=
sorry

end budgets_equal_after_6_years_l601_60164


namespace blue_sequins_count_l601_60171

/-- The number of blue sequins in each row of Jane's costume. -/
def blue_sequins_per_row : ℕ := 
  let total_sequins : ℕ := 162
  let blue_rows : ℕ := 6
  let purple_rows : ℕ := 5
  let purple_per_row : ℕ := 12
  let green_rows : ℕ := 9
  let green_per_row : ℕ := 6
  (total_sequins - purple_rows * purple_per_row - green_rows * green_per_row) / blue_rows

theorem blue_sequins_count : blue_sequins_per_row = 8 := by
  sorry

end blue_sequins_count_l601_60171


namespace min_max_tan_sum_l601_60190

/-- The system of equations -/
def system (x y z : ℝ) : Prop :=
  (Real.tan x)^3 + (Real.tan y)^3 + (Real.tan z)^3 = 36 ∧
  (Real.tan x)^2 + (Real.tan y)^2 + (Real.tan z)^2 = 14 ∧
  ((Real.tan x)^2 + Real.tan y) * (Real.tan x + Real.tan z) * (Real.tan y + Real.tan z) = 60

/-- The theorem to prove -/
theorem min_max_tan_sum (x y z : ℝ) :
  system x y z →
  ∃ (min_tan max_tan : ℝ),
    (∀ w, system x w z → Real.tan x ≤ max_tan ∧ min_tan ≤ Real.tan x) ∧
    min_tan + max_tan = 4 :=
sorry

end min_max_tan_sum_l601_60190


namespace sara_letters_problem_l601_60162

theorem sara_letters_problem (january february march total : ℕ) :
  february = 9 →
  march = 3 * january →
  total = january + february + march →
  total = 33 →
  january = 6 :=
by sorry

end sara_letters_problem_l601_60162


namespace only_valid_solutions_l601_60163

/-- A pair of natural numbers (m, n) is a valid solution if both n^2 + 4m and m^2 + 5n are perfect squares. -/
def is_valid_solution (m n : ℕ) : Prop :=
  ∃ (a b : ℕ), n^2 + 4*m = a^2 ∧ m^2 + 5*n = b^2

/-- The set of all valid solutions. -/
def valid_solutions : Set (ℕ × ℕ) :=
  {p | is_valid_solution p.1 p.2}

/-- The theorem stating that the only valid solutions are (2,1), (22,9), and (9,8). -/
theorem only_valid_solutions :
  valid_solutions = {(2, 1), (22, 9), (9, 8)} :=
by sorry

end only_valid_solutions_l601_60163


namespace linda_purchase_theorem_l601_60109

/-- Represents the number of items at each price point -/
structure ItemCounts where
  cents50 : ℕ
  dollars2 : ℕ
  dollars4 : ℕ

/-- Calculates the total cost in cents given the item counts -/
def totalCost (items : ItemCounts) : ℕ :=
  50 * items.cents50 + 200 * items.dollars2 + 400 * items.dollars4

/-- Theorem stating that given the conditions, Linda bought 40 50-cent items -/
theorem linda_purchase_theorem (items : ItemCounts) : 
  (items.cents50 + items.dollars2 + items.dollars4 = 50) →
  (totalCost items = 5000) →
  (items.cents50 = 40) := by
  sorry

#eval totalCost { cents50 := 40, dollars2 := 4, dollars4 := 6 }

end linda_purchase_theorem_l601_60109


namespace intersection_A_B_l601_60174

def A : Set ℝ := {x | ∃ y, y = Real.log x}
def B : Set ℝ := {-2, -1, 1, 2}

theorem intersection_A_B : A ∩ B = {1, 2} := by sorry

end intersection_A_B_l601_60174


namespace seating_arrangements_eq_twelve_l601_60147

/-- The number of ways to arrange 4 people in a row of 4 seats, 
    where 2 specific people must sit next to each other. -/
def seating_arrangements : ℕ := 12

/-- Theorem stating that the number of seating arrangements is 12. -/
theorem seating_arrangements_eq_twelve : seating_arrangements = 12 := by
  sorry

end seating_arrangements_eq_twelve_l601_60147


namespace total_credit_hours_l601_60197

/-- Represents the number of credit hours for a course -/
structure CreditHours where
  hours : ℕ

/-- Represents a college course -/
structure Course where
  credits : CreditHours

def standard_course : Course :=
  { credits := { hours := 3 } }

def advanced_course : Course :=
  { credits := { hours := 4 } }

def max_courses : ℕ := 40
def max_semesters : ℕ := 4
def max_courses_per_semester : ℕ := 5
def max_advanced_courses : ℕ := 2

def sid_courses : ℕ := 4 * max_courses
def sid_advanced_courses : ℕ := 2 * max_advanced_courses

theorem total_credit_hours : 
  (max_courses - max_advanced_courses) * standard_course.credits.hours +
  max_advanced_courses * advanced_course.credits.hours +
  (sid_courses - sid_advanced_courses) * standard_course.credits.hours +
  sid_advanced_courses * advanced_course.credits.hours = 606 := by
  sorry


end total_credit_hours_l601_60197


namespace peter_winning_strategy_l601_60138

open Set

/-- Represents a point on a plane -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- Represents a triangle defined by three points -/
structure Triangle :=
  (a : Point)
  (b : Point)
  (c : Point)

/-- Represents a color (red or blue) -/
inductive Color
  | Red
  | Blue

/-- Function to check if two triangles are similar -/
def are_similar (t1 t2 : Triangle) : Prop :=
  sorry

/-- Function to check if all points in a set have the same color -/
def all_same_color (points : Set Point) (coloring : Point → Color) : Prop :=
  sorry

/-- Theorem stating that two points are sufficient for Peter's winning strategy -/
theorem peter_winning_strategy (original : Triangle) :
  ∃ (p1 p2 : Point), ∀ (coloring : Point → Color),
    ∃ (t : Triangle), are_similar t original ∧
      all_same_color {t.a, t.b, t.c} coloring :=
sorry

end peter_winning_strategy_l601_60138


namespace tim_extra_running_days_l601_60160

def extra_running_days (original_days : ℕ) (hours_per_day : ℕ) (total_hours : ℕ) : ℕ :=
  (total_hours / hours_per_day) - original_days

theorem tim_extra_running_days :
  extra_running_days 3 2 10 = 2 := by
  sorry

end tim_extra_running_days_l601_60160


namespace sequence_with_geometric_differences_l601_60180

def geometric_difference_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n : ℕ, n ≥ 1 → a n - a (n - 1) = 2 * (a (n - 1) - a (n - 2))

theorem sequence_with_geometric_differences 
  (a : ℕ → ℝ) 
  (h1 : a 1 = 1) 
  (h2 : geometric_difference_sequence a) :
  ∀ n : ℕ, n ≥ 1 → a n = 2^n - 1 := by
sorry

end sequence_with_geometric_differences_l601_60180


namespace no_two_distinct_roots_ellipse_slope_product_constant_l601_60102

-- Statement for ①
theorem no_two_distinct_roots (f : ℝ → ℝ) (h : Monotone f) :
  ¬∃ k : ℝ, ∃ x y : ℝ, x ≠ y ∧ f x + k = 0 ∧ f y + k = 0 :=
sorry

-- Statement for ④
theorem ellipse_slope_product_constant (a b : ℝ) (h : a > b) (h' : b > 0) :
  ∃ c : ℝ, ∀ m n : ℝ, 
    (b^2 * m^2 + a^2 * n^2 = a^2 * b^2) →
    (n / (m + a)) * (n / (m - a)) = c :=
sorry

end no_two_distinct_roots_ellipse_slope_product_constant_l601_60102


namespace three_couples_arrangement_l601_60126

/-- The number of arrangements for three couples standing in a row -/
def couple_arrangements : ℕ := 48

/-- The number of ways to arrange three distinct units in a row -/
def unit_arrangements : ℕ := 6

/-- The number of internal arrangements for each couple -/
def internal_arrangements : ℕ := 2

/-- Theorem: The number of different arrangements for three couples standing in a row,
    where each couple must stand next to each other, is equal to 48. -/
theorem three_couples_arrangement :
  couple_arrangements = unit_arrangements * internal_arrangements^3 :=
by sorry

end three_couples_arrangement_l601_60126


namespace four_thirds_of_number_is_36_l601_60129

theorem four_thirds_of_number_is_36 (x : ℚ) : (4 : ℚ) / 3 * x = 36 → x = 27 := by
  sorry

end four_thirds_of_number_is_36_l601_60129


namespace odd_integers_divisibility_l601_60181

theorem odd_integers_divisibility (a b : ℕ) : 
  Odd a → Odd b → a > 0 → b > 0 → (2 * a * b + 1) ∣ (a^2 + b^2 + 1) → a = b := by
  sorry

end odd_integers_divisibility_l601_60181


namespace perfect_cube_units_digits_l601_60175

theorem perfect_cube_units_digits : ∀ d : Fin 10, ∃ n : ℤ, (n ^ 3 : ℤ) % 10 = d.val :=
sorry

end perfect_cube_units_digits_l601_60175


namespace largest_n_divisible_by_seven_l601_60139

def expression (n : ℕ) : ℤ :=
  9 * (n - 3)^7 - 2 * n^3 + 15 * n - 33

theorem largest_n_divisible_by_seven :
  ∃ (n : ℕ), n = 149998 ∧
  n < 150000 ∧
  expression n % 7 = 0 ∧
  ∀ (m : ℕ), m < 150000 → m > n → expression m % 7 ≠ 0 :=
sorry

end largest_n_divisible_by_seven_l601_60139


namespace expression_evaluation_l601_60100

theorem expression_evaluation (x : ℚ) (h : x = -4) : 
  (1 - 4 / (x + 3)) / ((x^2 - 1) / (x^2 + 6*x + 9)) = 1/3 := by
  sorry

end expression_evaluation_l601_60100


namespace cafeteria_apples_l601_60134

def apples_handed_out (initial_apples : ℕ) (pies_made : ℕ) (apples_per_pie : ℕ) : ℕ :=
  initial_apples - pies_made * apples_per_pie

theorem cafeteria_apples 
  (initial_apples : ℕ) 
  (pies_made : ℕ) 
  (apples_per_pie : ℕ) 
  (h1 : initial_apples = 50) 
  (h2 : pies_made = 9) 
  (h3 : apples_per_pie = 5) :
  apples_handed_out initial_apples pies_made apples_per_pie = 5 := by
sorry

end cafeteria_apples_l601_60134


namespace min_intersection_points_2000_l601_60195

/-- Represents a collection of congruent circles on a plane -/
structure CircleCollection where
  n : ℕ
  no_tangent : Bool
  meets_two : Bool

/-- The minimum number of intersection points for a given collection of circles -/
def min_intersection_points (c : CircleCollection) : ℕ :=
  2 * (c.n - 2) + 1

/-- Theorem: For 2000 circles satisfying the given conditions, 
    the minimum number of intersection points is 3997 -/
theorem min_intersection_points_2000 :
  ∀ (c : CircleCollection), 
    c.n = 2000 ∧ c.no_tangent ∧ c.meets_two → 
    min_intersection_points c = 3997 := by
  sorry

#eval min_intersection_points ⟨2000, true, true⟩

end min_intersection_points_2000_l601_60195


namespace M_subset_N_l601_60185

def M : Set ℝ := {-1, 1}
def N : Set ℝ := {x | (1 / x) < 2}

theorem M_subset_N : M ⊆ N := by
  sorry

end M_subset_N_l601_60185


namespace subset_implies_a_equals_one_l601_60119

-- Define the set A
def A : Set ℝ := {-1, 0, 2}

-- Define the set B as a function of a
def B (a : ℝ) : Set ℝ := {2^a}

-- Theorem statement
theorem subset_implies_a_equals_one (a : ℝ) (h : B a ⊆ A) : a = 1 := by
  sorry

end subset_implies_a_equals_one_l601_60119


namespace smallest_two_digit_k_for_45k_perfect_square_l601_60156

/-- A number is a perfect square if it has an integer square root -/
def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, m * m = n

/-- A number is a two-digit positive integer if it's between 10 and 99 inclusive -/
def is_two_digit_positive (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

theorem smallest_two_digit_k_for_45k_perfect_square :
  ∀ k : ℕ, is_two_digit_positive k → is_perfect_square (45 * k) → k ≥ 20 :=
by sorry

end smallest_two_digit_k_for_45k_perfect_square_l601_60156


namespace matrix_zero_product_implies_zero_multiplier_l601_60149

theorem matrix_zero_product_implies_zero_multiplier 
  (A B : Matrix (Fin 3) (Fin 3) ℂ) 
  (hB : B ≠ 0) 
  (hAB : A * B = 0) : 
  ∃ D : Matrix (Fin 3) (Fin 3) ℂ, D ≠ 0 ∧ A * D = 0 ∧ D * A = 0 := by
  sorry

end matrix_zero_product_implies_zero_multiplier_l601_60149


namespace smallest_natural_number_satisfying_conditions_l601_60193

theorem smallest_natural_number_satisfying_conditions : 
  ∃ (n : ℕ), n = 37 ∧ 
  (∃ (k : ℕ), n + 13 = 5 * k) ∧ 
  (∃ (m : ℕ), n - 13 = 6 * m) ∧
  (∀ (x : ℕ), x < n → ¬((∃ (k : ℕ), x + 13 = 5 * k) ∧ (∃ (m : ℕ), x - 13 = 6 * m))) :=
by sorry

end smallest_natural_number_satisfying_conditions_l601_60193


namespace jeremy_speed_l601_60176

/-- Given a distance of 20 kilometers and a time of 10 hours, prove that the speed is 2 kilometers per hour. -/
theorem jeremy_speed (distance : ℝ) (time : ℝ) (h1 : distance = 20) (h2 : time = 10) :
  distance / time = 2 := by
  sorry

end jeremy_speed_l601_60176


namespace plane_perpendicularity_l601_60154

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (parallel : Line → Plane → Prop)
variable (perpendicular : Line → Plane → Prop)
variable (perpendicularPlanes : Plane → Plane → Prop)

-- State the theorem
theorem plane_perpendicularity 
  (a b : Line) 
  (α β γ : Plane) 
  (h1 : a ≠ b) 
  (h2 : α ≠ β ∧ β ≠ γ ∧ α ≠ γ) 
  (h3 : parallel a α) 
  (h4 : perpendicular a β) : 
  perpendicularPlanes α β :=
sorry

end plane_perpendicularity_l601_60154


namespace prob_double_is_one_seventh_l601_60103

/-- Represents a domino set with integers from 0 to 12 -/
def DominoSet : Type := Unit

/-- The number of integers in the domino set -/
def num_integers : ℕ := 13

/-- The total number of domino tiles in the set -/
def total_tiles (ds : DominoSet) : ℕ := (num_integers * (num_integers + 1)) / 2

/-- The number of double tiles in the set -/
def num_doubles (ds : DominoSet) : ℕ := num_integers

/-- The probability of randomly selecting a double from the domino set -/
def prob_double (ds : DominoSet) : ℚ := (num_doubles ds : ℚ) / (total_tiles ds : ℚ)

theorem prob_double_is_one_seventh (ds : DominoSet) :
  prob_double ds = 1 / 7 := by
  sorry

end prob_double_is_one_seventh_l601_60103


namespace fraction_simplification_l601_60145

-- Define the statement
theorem fraction_simplification : (36 ^ 40) / (72 ^ 20) = 18 ^ 20 := by
  sorry

end fraction_simplification_l601_60145


namespace jackie_has_ten_apples_l601_60161

/-- The number of apples Adam has -/
def adam_apples : ℕ := 9

/-- The number of apples Jackie has -/
def jackie_apples : ℕ := adam_apples + 1

/-- Theorem: Jackie has 10 apples -/
theorem jackie_has_ten_apples : jackie_apples = 10 := by
  sorry

end jackie_has_ten_apples_l601_60161


namespace prob_first_class_correct_l601_60117

/-- Represents the two types of items -/
inductive ItemClass
| First
| Second

/-- Represents the two trucks -/
inductive Truck
| A
| B

/-- The total number of items -/
def totalItems : Nat := 10

/-- The number of items in each truck -/
def truckItems : Truck → ItemClass → Nat
| Truck.A, ItemClass.First => 2
| Truck.A, ItemClass.Second => 2
| Truck.B, ItemClass.First => 4
| Truck.B, ItemClass.Second => 2

/-- The number of broken items per truck -/
def brokenItemsPerTruck : Nat := 1

/-- The number of remaining items after breakage -/
def remainingItems : Nat := totalItems - 2 * brokenItemsPerTruck

/-- The probability of selecting a first-class item from the remaining items -/
def probFirstClass : Rat := 29 / 48

theorem prob_first_class_correct :
  probFirstClass = 29 / 48 := by sorry

end prob_first_class_correct_l601_60117


namespace geometric_series_ratio_l601_60137

theorem geometric_series_ratio (a r : ℝ) (h : r ≠ 1) :
  (a * r^4 / (1 - r)) = (a / (1 - r)) / 81 → r = 1/3 := by
  sorry

end geometric_series_ratio_l601_60137


namespace tangent_line_at_zero_f_positive_range_l601_60188

noncomputable section

variables (a : ℝ)

-- Define the function f
def f (x : ℝ) : ℝ := (a * x + 1) * Real.exp x - (a + 1) * x - 1

-- Theorem 1: The tangent line at (0, f(0)) is y = 0
theorem tangent_line_at_zero (a : ℝ) : 
  ∃ (m b : ℝ), ∀ x, m * x + b = 0 ∧ 
  (∀ ε > 0, ∃ δ > 0, ∀ h, abs h < δ → abs (f a (0 + h) - f a 0 - m * h) ≤ ε * abs h) :=
sorry

-- Theorem 2: For f(x) > 0 to always hold when x > 0, a must be in [0,+∞)
theorem f_positive_range (a : ℝ) :
  (∀ x > 0, f a x > 0) ↔ a ≥ 0 :=
sorry

end tangent_line_at_zero_f_positive_range_l601_60188


namespace favorite_subject_count_l601_60130

theorem favorite_subject_count (total : ℕ) (math_fraction : ℚ) (english_fraction : ℚ)
  (science_fraction : ℚ) (h_total : total = 30) (h_math : math_fraction = 1/5)
  (h_english : english_fraction = 1/3) (h_science : science_fraction = 1/7) :
  total - (total * math_fraction).floor - (total * english_fraction).floor -
  ((total - (total * math_fraction).floor - (total * english_fraction).floor) * science_fraction).floor = 12 :=
by sorry

end favorite_subject_count_l601_60130


namespace gcd_360_504_l601_60170

theorem gcd_360_504 : Nat.gcd 360 504 = 72 := by
  sorry

end gcd_360_504_l601_60170


namespace bicycle_price_after_discounts_l601_60123

/-- Calculates the final price of a bicycle after two consecutive discounts. -/
def final_price (original_price : ℝ) (discount1 : ℝ) (discount2 : ℝ) : ℝ :=
  original_price * (1 - discount1) * (1 - discount2)

/-- Theorem stating that a $200 bicycle, after a 40% discount followed by a 25% discount, costs $90. -/
theorem bicycle_price_after_discounts :
  final_price 200 0.4 0.25 = 90 := by
  sorry

#eval final_price 200 0.4 0.25

end bicycle_price_after_discounts_l601_60123


namespace negation_of_universal_proposition_l601_60146

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^2 ≠ x) ↔ (∃ x : ℝ, x^2 = x) := by sorry

end negation_of_universal_proposition_l601_60146


namespace unique_prime_pair_l601_60132

theorem unique_prime_pair : ∃! (p q : ℕ), 
  Nat.Prime p ∧ Nat.Prime q ∧ Nat.Prime (p^2 + 2*p*q^2 + 1) :=
by
  sorry

end unique_prime_pair_l601_60132


namespace left_square_side_length_l601_60108

/-- Given three squares with specific side length relationships, prove the left square's side length --/
theorem left_square_side_length (x : ℝ) : 
  x > 0 ∧ 
  x + (x + 17) + (x + 11) = 52 → 
  x = 8 := by
  sorry

end left_square_side_length_l601_60108


namespace farm_pets_after_changes_l601_60150

/-- Calculates the total number of pets after changes to a farm's pet population -/
theorem farm_pets_after_changes 
  (initial_dogs : ℕ) 
  (initial_fish : ℕ) 
  (initial_cats : ℕ) 
  (dogs_left : ℕ) 
  (rabbits_added : ℕ) 
  (h_initial_dogs : initial_dogs = 43)
  (h_initial_fish : initial_fish = 72)
  (h_initial_cats : initial_cats = 34)
  (h_dogs_left : dogs_left = 5)
  (h_rabbits_added : rabbits_added = 10) :
  initial_dogs - dogs_left + 2 * initial_fish + initial_cats + rabbits_added = 226 := by
  sorry

end farm_pets_after_changes_l601_60150


namespace equation_solutions_l601_60189

theorem equation_solutions :
  ∀ x : ℝ, x * (x + 1) = 12 ↔ x = -4 ∨ x = 3 := by
sorry

end equation_solutions_l601_60189


namespace pentagon_cannot_tile_l601_60148

-- Define the regular polygons we're considering
inductive RegularPolygon
  | EquilateralTriangle
  | Square
  | Pentagon
  | Hexagon

-- Function to calculate the interior angle of a regular polygon
def interiorAngle (p : RegularPolygon) : ℚ :=
  match p with
  | RegularPolygon.EquilateralTriangle => 60
  | RegularPolygon.Square => 90
  | RegularPolygon.Pentagon => 108
  | RegularPolygon.Hexagon => 120

-- Define what it means for a shape to be able to tile a plane
def canTilePlane (p : RegularPolygon) : Prop :=
  ∃ (n : ℕ), n * interiorAngle p = 360

-- Theorem stating that only the pentagon cannot tile the plane
theorem pentagon_cannot_tile :
  ∀ p : RegularPolygon,
    ¬(canTilePlane p) ↔ p = RegularPolygon.Pentagon :=
by sorry

end pentagon_cannot_tile_l601_60148


namespace cube_vertical_faces_same_color_prob_l601_60158

/-- Represents the probability of painting a face blue -/
def blue_prob : ℚ := 1/3

/-- Represents the probability of painting a face red -/
def red_prob : ℚ := 2/3

/-- Represents the number of faces on a cube -/
def num_faces : ℕ := 6

/-- Represents the number of vertical faces when a cube is placed on a horizontal surface -/
def num_vertical_faces : ℕ := 4

/-- Calculates the probability of all faces being the same color -/
def all_same_color_prob : ℚ := red_prob^num_faces + blue_prob^num_faces

/-- Calculates the probability of vertical faces being one color and top/bottom being another -/
def mixed_color_prob : ℚ := 3 * (red_prob^num_vertical_faces * blue_prob^(num_faces - num_vertical_faces) +
                                 blue_prob^num_vertical_faces * red_prob^(num_faces - num_vertical_faces))

/-- The main theorem stating the probability of the cube having all four vertical faces
    the same color when placed on a horizontal surface -/
theorem cube_vertical_faces_same_color_prob :
  all_same_color_prob + mixed_color_prob = 789/6561 := by sorry

end cube_vertical_faces_same_color_prob_l601_60158


namespace bernoulli_inequality_l601_60153

theorem bernoulli_inequality (n : ℕ) (x : ℝ) (h : x ≥ -1) :
  1 + n * x ≤ (1 + x)^n := by sorry

end bernoulli_inequality_l601_60153


namespace unique_solution_for_equation_l601_60105

theorem unique_solution_for_equation : ∃! (x y z : ℕ),
  (x < 10 ∧ y < 10 ∧ z < 10) ∧
  (10 * x + 5 < 100) ∧
  (300 ≤ 300 + 10 * y + z) ∧
  (300 + 10 * y + z < 400) ∧
  ((10 * x + 5) * (300 + 10 * y + z) = 7850) ∧
  x = 2 ∧ y = 1 ∧ z = 4 := by
sorry

end unique_solution_for_equation_l601_60105


namespace a_2017_equals_2_l601_60166

def sequence_a : ℕ → ℚ
  | 0 => 2
  | n + 1 => (sequence_a n - 1) / (sequence_a n + 1)

theorem a_2017_equals_2 : sequence_a 2016 = 2 := by
  sorry

end a_2017_equals_2_l601_60166


namespace estate_division_percentage_l601_60122

/-- Represents the estate division problem --/
structure EstateDivision where
  amount₁ : ℝ  -- Amount received by the first person
  range : ℝ    -- Smallest possible range between highest and lowest amounts
  percentage : ℝ -- Percentage stipulation

/-- The estate division problem satisfies the given conditions --/
def valid_division (e : EstateDivision) : Prop :=
  e.amount₁ = 20000 ∧ 
  e.range = 10000 ∧ 
  0 < e.percentage ∧ 
  e.percentage < 100

/-- The theorem stating that the percentage stipulation is 25% --/
theorem estate_division_percentage (e : EstateDivision) 
  (h : valid_division e) : e.percentage = 25 := by
  sorry

end estate_division_percentage_l601_60122


namespace equation_solution_l601_60194

theorem equation_solution : ∃ x : ℝ, x > 0 ∧ 4 * x^(1/3) - 2 * (x / x^(2/3)) = 7 + x^(1/3) ∧ x = 343 := by
  sorry

end equation_solution_l601_60194


namespace line_through_points_l601_60112

/-- The equation of a line passing through two points (x₁, y₁) and (x₂, y₂) is
    (y - y₁) / (y₂ - y₁) = (x - x₁) / (x₂ - x₁) -/
def line_equation (x₁ y₁ x₂ y₂ : ℝ) (x y : ℝ) : Prop :=
  (y - y₁) * (x₂ - x₁) = (x - x₁) * (y₂ - y₁)

theorem line_through_points :
  ∀ x y : ℝ, line_equation 2 (-2) (-2) 6 x y ↔ 2 * x + y - 2 = 0 := by sorry

end line_through_points_l601_60112


namespace triangle_abc_area_l601_60136

/-- Triangle ABC with vertices A(0,0), B(1,7), and C(0,8) has an area of 28 square units -/
theorem triangle_abc_area : 
  let A : ℝ × ℝ := (0, 0)
  let B : ℝ × ℝ := (1, 7)
  let C : ℝ × ℝ := (0, 8)
  let triangle_area := (1/2) * |(C.2 - A.2)| * |(B.1 - A.1)|
  triangle_area = 28 := by
  sorry

end triangle_abc_area_l601_60136


namespace discriminant_less_than_negative_one_l601_60173

/-- A quadratic function that doesn't intersect with y = x and y = -x -/
structure NonIntersectingQuadratic where
  a : ℝ
  b : ℝ
  c : ℝ
  h1 : ∀ x : ℝ, a * x^2 + b * x + c ≠ x
  h2 : ∀ x : ℝ, a * x^2 + b * x + c ≠ -x

/-- The discriminant of a quadratic function is less than -1 -/
theorem discriminant_less_than_negative_one (f : NonIntersectingQuadratic) :
  |f.b^2 - 4 * f.a * f.c| > 1 := by
  sorry

end discriminant_less_than_negative_one_l601_60173


namespace polygon_sides_l601_60198

theorem polygon_sides (n : ℕ) (h : (n - 2) * 180 = 4 * 360) : n = 10 := by
  sorry

end polygon_sides_l601_60198


namespace speed_calculation_l601_60179

/-- 
Given a speed v, if increasing the speed by 21 miles per hour reduces the time by 1/3, 
then v must be 42 miles per hour.
-/
theorem speed_calculation (v : ℝ) : 
  (v * 1 = (v + 21) * (2/3)) → v = 42 := by
  sorry

end speed_calculation_l601_60179


namespace product_from_lcm_hcf_l601_60168

theorem product_from_lcm_hcf (a b : ℕ+) 
  (h_lcm : Nat.lcm a b = 72) 
  (h_hcf : Nat.gcd a b = 6) : 
  a * b = 432 := by
  sorry

end product_from_lcm_hcf_l601_60168


namespace base_is_twelve_l601_60187

/-- Represents a number system with a given base -/
structure NumberSystem where
  base : ℕ
  base_gt_5 : base > 5

/-- Converts a number from base b to decimal -/
def to_decimal (n : ℕ) (b : ℕ) : ℕ :=
  (n / 10) * b + (n % 10)

/-- Theorem: In a number system where the square of 24 is 554, the base of the system is 12 -/
theorem base_is_twelve (ns : NumberSystem) 
  (h : (to_decimal 24 ns.base)^2 = to_decimal 554 ns.base) : 
  ns.base = 12 := by
  sorry


end base_is_twelve_l601_60187


namespace a_plus_b_eq_neg_one_l601_60125

-- Define the sets A and B
def A (a b : ℝ) : Set ℝ := {1, a, b}
def B (a : ℝ) : Set ℝ := {a, a^2, a*a}

-- State the theorem
theorem a_plus_b_eq_neg_one (a b : ℝ) : A a b = B a → a + b = -1 := by
  sorry

end a_plus_b_eq_neg_one_l601_60125


namespace joshua_journey_l601_60186

/-- Proves that given a journey where half the distance is traveled at 12 km/h and 
    the other half at 8 km/h, with a total journey time of 50 minutes, 
    the distance traveled in the second half (jogging) is 4 km. -/
theorem joshua_journey (total_time : ℝ) (speed1 speed2 : ℝ) (h1 : total_time = 50 / 60) 
  (h2 : speed1 = 12) (h3 : speed2 = 8) : 
  let d := (total_time * speed1 * speed2) / (speed1 + speed2)
  d = 4 := by sorry

end joshua_journey_l601_60186


namespace isosceles_triangle_perimeter_l601_60110

/-- Given real numbers x and y satisfying |x-4| + √(y-10) = 0,
    prove that the perimeter of an isosceles triangle with side lengths x, y, and y is 24. -/
theorem isosceles_triangle_perimeter (x y : ℝ) 
  (h : |x - 4| + Real.sqrt (y - 10) = 0) : 
  x + y + y = 24 := by
  sorry

end isosceles_triangle_perimeter_l601_60110


namespace square_of_product_plus_one_l601_60133

theorem square_of_product_plus_one :
  24 * 25 * 26 * 27 + 1 = (24^2 + 3 * 24 + 1)^2 := by
  sorry

end square_of_product_plus_one_l601_60133


namespace marco_trading_cards_l601_60107

theorem marco_trading_cards (x : ℚ) : 
  (2 / 15 : ℚ) * x = 850 → x = 6375 := by
  sorry

end marco_trading_cards_l601_60107


namespace worker_a_time_l601_60116

theorem worker_a_time (worker_b_time worker_ab_time : ℝ) 
  (hb : worker_b_time = 15)
  (hab : worker_ab_time = 20 / 3) : 
  ∃ worker_a_time : ℝ, 
    worker_a_time = 12 ∧ 
    1 / worker_a_time + 1 / worker_b_time = 1 / worker_ab_time :=
by sorry

end worker_a_time_l601_60116


namespace circular_garden_radius_l601_60135

theorem circular_garden_radius (r : ℝ) (h : r > 0) : 2 * π * r = (1 / 8) * π * r^2 → r = 16 := by
  sorry

end circular_garden_radius_l601_60135


namespace ceiling_neg_sqrt_fraction_l601_60151

theorem ceiling_neg_sqrt_fraction : ⌈-Real.sqrt (36 / 9)⌉ = -2 := by sorry

end ceiling_neg_sqrt_fraction_l601_60151


namespace harolds_leftover_money_l601_60152

/-- Harold's financial situation --/
def harolds_finances (income rent car_payment groceries : ℚ) : Prop :=
  let utilities := car_payment / 2
  let total_expenses := rent + car_payment + utilities + groceries
  let remaining := income - total_expenses
  let retirement := remaining / 2
  let left_after_retirement := remaining - retirement
  income = 2500 ∧ 
  rent = 700 ∧ 
  car_payment = 300 ∧ 
  groceries = 50 ∧ 
  left_after_retirement = 650

theorem harolds_leftover_money :
  ∃ (income rent car_payment groceries : ℚ),
    harolds_finances income rent car_payment groceries :=
sorry

end harolds_leftover_money_l601_60152


namespace roger_step_goal_time_l601_60124

/-- Represents the number of steps Roger can walk in 30 minutes -/
def steps_per_30_min : ℕ := 2000

/-- Represents Roger's daily step goal -/
def daily_goal : ℕ := 10000

/-- Represents the time in minutes it takes Roger to reach his daily goal -/
def time_to_reach_goal : ℕ := 150

/-- Theorem stating that the time required for Roger to reach his daily goal is 150 minutes -/
theorem roger_step_goal_time : 
  (daily_goal / steps_per_30_min) * 30 = time_to_reach_goal :=
by sorry

end roger_step_goal_time_l601_60124


namespace intersection_of_A_and_B_l601_60120

def A : Set Int := {-1, 1, 2, 4}
def B : Set Int := {-1, 0, 2}

theorem intersection_of_A_and_B : A ∩ B = {-1, 2} := by
  sorry

end intersection_of_A_and_B_l601_60120


namespace sum_of_seven_angles_l601_60143

-- Define the angles
variable (angle1 angle2 angle3 angle4 angle5 angle6 angle7 angle8 angle9 angle10 : ℝ)

-- State the theorem
theorem sum_of_seven_angles :
  (angle5 + angle6 + angle7 + angle8 = 360) →
  (angle2 + angle3 + angle4 + (180 - angle9) = 360) →
  (angle9 = angle10) →
  (angle8 = angle10 + angle1) →
  (angle1 + angle2 + angle3 + angle4 + angle5 + angle6 + angle7 = 540) :=
by sorry

end sum_of_seven_angles_l601_60143


namespace max_distance_between_sine_cosine_graphs_l601_60199

theorem max_distance_between_sine_cosine_graphs : 
  ∃ (C : ℝ), C = 4 ∧ ∀ m : ℝ, |2 * Real.sin m - 2 * Real.sqrt 3 * Real.cos m| ≤ C ∧ 
  ∃ m : ℝ, |2 * Real.sin m - 2 * Real.sqrt 3 * Real.cos m| = C :=
sorry

end max_distance_between_sine_cosine_graphs_l601_60199


namespace intersection_empty_implies_t_geq_one_l601_60192

theorem intersection_empty_implies_t_geq_one (t : ℝ) : 
  let A : Set ℝ := {-1, 0, 1}
  let B : Set ℝ := {x | x > t}
  A ∩ B = ∅ → t ≥ 1 := by
sorry

end intersection_empty_implies_t_geq_one_l601_60192


namespace fibonacci_geometric_sequence_l601_60104

-- Define the Fibonacci sequence
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | n + 2 => fib n + fib (n + 1)

-- Define a predicate for geometric sequence
def is_geometric (a b c : ℕ) : Prop :=
  ∃ r : ℚ, r ≠ 0 ∧ fib b = r * fib a ∧ fib c = r * fib b

theorem fibonacci_geometric_sequence :
  ∀ a b c : ℕ,
    is_geometric a b c →
    a + b + c = 3000 →
    a = 999 := by
  sorry

end fibonacci_geometric_sequence_l601_60104


namespace min_value_product_l601_60127

theorem min_value_product (a b c x y z : ℝ) 
  (non_neg : a ≥ 0 ∧ b ≥ 0 ∧ c ≥ 0 ∧ x ≥ 0 ∧ y ≥ 0 ∧ z ≥ 0)
  (sum_abc : a + b + c = 1)
  (sum_xyz : x + y + z = 1) :
  (a - x^2) * (b - y^2) * (c - z^2) ≥ -1/4 :=
sorry

end min_value_product_l601_60127


namespace total_fishes_is_32_l601_60159

/-- The total number of fishes caught by Melanie and Tom -/
def total_fishes (melanie_trout : ℕ) (tom_salmon_multiplier : ℕ) : ℕ :=
  melanie_trout + tom_salmon_multiplier * melanie_trout

/-- Proof that the total number of fishes caught is 32 -/
theorem total_fishes_is_32 : total_fishes 8 3 = 32 := by
  sorry

end total_fishes_is_32_l601_60159


namespace complement_of_A_l601_60167

def I : Set ℕ := {1, 2, 3, 4, 5, 6}
def A : Set ℕ := {1, 3, 5}

theorem complement_of_A : (I \ A) = {2, 4, 6} := by sorry

end complement_of_A_l601_60167


namespace remainder_r_15_minus_1_l601_60101

theorem remainder_r_15_minus_1 (r : ℝ) : (r^15 - 1) % (r - 1) = 0 := by
  sorry

end remainder_r_15_minus_1_l601_60101


namespace min_value_of_y_l601_60172

theorem min_value_of_y (a x : ℝ) (h1 : 0 < a) (h2 : a < 15) (h3 : a ≤ x) (h4 : x ≤ 15) :
  let y := |x - a| + |x - 15| + |x - (a + 15)|
  ∃ (min_y : ℝ), min_y = 15 ∧ ∀ z, a ≤ z ∧ z ≤ 15 → y ≤ |z - a| + |z - 15| + |z - (a + 15)| :=
by
  sorry

end min_value_of_y_l601_60172


namespace number_times_one_fourth_squared_equals_four_cubed_l601_60157

theorem number_times_one_fourth_squared_equals_four_cubed (x : ℝ) : 
  x * (1/4)^2 = 4^3 ↔ x = 1024 := by
  sorry

end number_times_one_fourth_squared_equals_four_cubed_l601_60157


namespace candy_distribution_l601_60141

theorem candy_distribution (total_candies : ℕ) (total_children : ℕ) (lollipops_per_boy : ℕ) :
  total_candies = 90 →
  total_children = 40 →
  lollipops_per_boy = 3 →
  ∃ (num_boys num_girls : ℕ) (candy_canes_per_girl : ℕ),
    num_boys + num_girls = total_children ∧
    num_boys * lollipops_per_boy = total_candies / 3 ∧
    num_girls * candy_canes_per_girl = total_candies * 2 / 3 ∧
    candy_canes_per_girl = 2 :=
by
  sorry

end candy_distribution_l601_60141


namespace circle_symmetry_l601_60140

-- Define the original circle
def original_circle (x y : ℝ) : Prop := (x - 2)^2 + (y + 1)^2 = 4

-- Define the line of symmetry
def symmetry_line (x y : ℝ) : Prop := y = x + 1

-- Define the symmetric circle
def symmetric_circle (x y : ℝ) : Prop := (x + 2)^2 + (y - 3)^2 = 4

-- Theorem statement
theorem circle_symmetry :
  ∀ (x y : ℝ), 
  (∃ (x₀ y₀ : ℝ), original_circle x₀ y₀ ∧ 
   symmetry_line ((x + x₀) / 2) ((y + y₀) / 2) ∧
   (y - y₀) = -(x - x₀)) →
  symmetric_circle x y :=
by sorry

end circle_symmetry_l601_60140


namespace thursday_to_wednesday_ratio_l601_60178

/-- Represents the number of laundry loads washed on each day of the week --/
structure LaundryWeek where
  wednesday : ℕ
  thursday : ℕ
  friday : ℕ
  saturday : ℕ

/-- Defines the conditions for Vincent's laundry week --/
def vincentLaundryWeek (w : LaundryWeek) : Prop :=
  w.wednesday = 6 ∧
  w.friday = w.thursday / 2 ∧
  w.saturday = w.wednesday / 3 ∧
  w.wednesday + w.thursday + w.friday + w.saturday = 26

/-- Theorem stating that the ratio of loads washed on Thursday to Wednesday is 2:1 --/
theorem thursday_to_wednesday_ratio (w : LaundryWeek) 
  (h : vincentLaundryWeek w) : w.thursday = 2 * w.wednesday := by
  sorry

end thursday_to_wednesday_ratio_l601_60178


namespace fraction_difference_equals_eight_sqrt_three_l601_60113

theorem fraction_difference_equals_eight_sqrt_three :
  let a : ℝ := 2 + Real.sqrt 3
  let b : ℝ := 2 - Real.sqrt 3
  (a / b) - (b / a) = 8 * Real.sqrt 3 := by
  sorry

end fraction_difference_equals_eight_sqrt_three_l601_60113


namespace largest_inscribed_circle_circumference_l601_60196

theorem largest_inscribed_circle_circumference (square_side : ℝ) (h : square_side = 12) :
  let circle_radius := square_side / 2
  2 * Real.pi * circle_radius = 12 * Real.pi :=
by sorry

end largest_inscribed_circle_circumference_l601_60196


namespace linear_equation_implies_mn_zero_l601_60106

/-- If x^(m+n) + 5y^(m-n+2) = 8 is a linear equation in x and y, then mn = 0 -/
theorem linear_equation_implies_mn_zero (m n : ℤ) : 
  (∃ a b c : ℝ, a ≠ 0 ∧ b ≠ 0 ∧ ∀ x y : ℝ, x^(m+n) + 5*y^(m-n+2) = a*x + b*y + c) → 
  m * n = 0 := by
sorry

end linear_equation_implies_mn_zero_l601_60106


namespace complement_of_A_l601_60142

def A : Set ℝ := {x | |x - 1| > 2}

theorem complement_of_A : 
  (Set.univ \ A) = {x : ℝ | -1 ≤ x ∧ x ≤ 3} := by sorry

end complement_of_A_l601_60142


namespace simplify_expression_l601_60165

theorem simplify_expression (w x : ℝ) : 
  3*w + 5*w + 7*w + 9*w + 11*w + 13*x + 15 = 35*w + 13*x + 15 := by
  sorry

end simplify_expression_l601_60165


namespace linear_function_theorem_l601_60111

/-- A linear function f(x) = ax + b -/
def f (a b : ℝ) (x : ℝ) : ℝ := a * x + b

/-- The derivative of f with respect to x -/
def f_derivative (a : ℝ) : ℝ := a

theorem linear_function_theorem (a b : ℝ) :
  f a b 1 = 2 ∧ f_derivative a = 2 → f a b 2 = 4 := by
  sorry

end linear_function_theorem_l601_60111


namespace platform_completion_time_l601_60115

/-- Represents the number of days required to complete a portion of a project given a number of workers -/
def days_to_complete (workers : ℕ) (portion : ℚ) : ℚ :=
  sorry

theorem platform_completion_time :
  let initial_workers : ℕ := 90
  let initial_days : ℕ := 6
  let initial_portion : ℚ := 1/2
  let remaining_workers : ℕ := 60
  let remaining_portion : ℚ := 1/2
  days_to_complete initial_workers initial_portion = initial_days →
  days_to_complete remaining_workers remaining_portion = 9 :=
by sorry

end platform_completion_time_l601_60115


namespace group_collection_proof_l601_60183

/-- Calculates the total amount collected by a group of students, where each student
    contributes as many paise as there are members in the group. -/
def total_amount_collected (num_members : ℕ) : ℚ :=
  (num_members * num_members : ℚ) / 100

/-- Proves that a group of 85 students, each contributing as many paise as there are members,
    will collect a total of 72.25 rupees. -/
theorem group_collection_proof :
  total_amount_collected 85 = 72.25 := by
  sorry

end group_collection_proof_l601_60183
