import Mathlib

namespace NUMINAMATH_CALUDE_monkey_climb_l3290_329091

theorem monkey_climb (tree_height : ℝ) (climb_rate : ℝ) (total_time : ℕ) 
  (h1 : tree_height = 19)
  (h2 : climb_rate = 3)
  (h3 : total_time = 17) :
  ∃ (slip_back : ℝ), 
    slip_back = 2 ∧ 
    (total_time - 1 : ℝ) * (climb_rate - slip_back) + climb_rate = tree_height :=
by sorry

end NUMINAMATH_CALUDE_monkey_climb_l3290_329091


namespace NUMINAMATH_CALUDE_g_of_6_l3290_329093

def g (x : ℝ) : ℝ := 3*x^4 - 20*x^3 + 37*x^2 - 18*x - 80

theorem g_of_6 : g 6 = 712 := by
  sorry

end NUMINAMATH_CALUDE_g_of_6_l3290_329093


namespace NUMINAMATH_CALUDE_negation_of_forall_positive_l3290_329016

theorem negation_of_forall_positive (S : Set ℚ) :
  (¬ ∀ x ∈ S, 2 * x + 1 > 0) ↔ (∃ x ∈ S, 2 * x + 1 ≤ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_forall_positive_l3290_329016


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l3290_329092

def arithmetic_sequence (a₁ : ℚ) (d : ℚ) (n : ℕ) : ℚ := a₁ + (n - 1 : ℚ) * d

def sum_arithmetic_sequence (a₁ : ℚ) (aₙ : ℚ) (n : ℕ) : ℚ := n * (a₁ + aₙ) / 2

theorem arithmetic_sequence_problem (a₁ a₂ a₅ aₙ : ℚ) (n : ℕ) :
  a₁ = 1/3 →
  a₂ + a₅ = 4 →
  aₙ = 33 →
  (∃ d : ℚ, ∀ k : ℕ, arithmetic_sequence a₁ d k = a₁ + (k - 1 : ℚ) * d) →
  n = 50 ∧ sum_arithmetic_sequence a₁ aₙ n = 850 :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l3290_329092


namespace NUMINAMATH_CALUDE_book_club_boys_count_l3290_329081

theorem book_club_boys_count (total_members attendees : ℕ) 
  (h_total : total_members = 30)
  (h_attendees : attendees = 18)
  (h_all_boys_attended : ∃ boys girls : ℕ, 
    boys + girls = total_members ∧
    boys + (girls / 3) = attendees) : 
  ∃ boys : ℕ, boys = 12 ∧ ∃ girls : ℕ, boys + girls = total_members :=
sorry

end NUMINAMATH_CALUDE_book_club_boys_count_l3290_329081


namespace NUMINAMATH_CALUDE_first_group_work_days_l3290_329005

/-- Represents the daily work units done by a person -/
@[ext] structure WorkUnit where
  value : ℚ

/-- Represents a group of workers -/
structure WorkGroup where
  men : ℕ
  boys : ℕ

/-- Calculates the total work done by a group in a given number of days -/
def totalWork (g : WorkGroup) (manUnit boyUnit : WorkUnit) (days : ℚ) : ℚ :=
  (g.men : ℚ) * manUnit.value * days + (g.boys : ℚ) * boyUnit.value * days

theorem first_group_work_days : 
  let manUnit : WorkUnit := ⟨2⟩
  let boyUnit : WorkUnit := ⟨1⟩
  let firstGroup : WorkGroup := ⟨12, 16⟩
  let secondGroup : WorkGroup := ⟨13, 24⟩
  let secondGroupDays : ℚ := 4
  totalWork firstGroup manUnit boyUnit 5 = totalWork secondGroup manUnit boyUnit secondGroupDays := by
  sorry

end NUMINAMATH_CALUDE_first_group_work_days_l3290_329005


namespace NUMINAMATH_CALUDE_friend_consumption_l3290_329095

def total_people : ℕ := 8
def pizzas : ℕ := 5
def slices_per_pizza : ℕ := 8
def pasta_bowls : ℕ := 2
def garlic_breads : ℕ := 12

def ron_scott_pizza : ℕ := 10
def mark_pizza : ℕ := 2
def sam_pizza : ℕ := 4

def ron_scott_pasta_percent : ℚ := 40 / 100
def ron_scott_mark_garlic_percent : ℚ := 25 / 100

theorem friend_consumption :
  let remaining_friends := total_people - 4
  let remaining_pizza := pizzas * slices_per_pizza - (ron_scott_pizza + mark_pizza + sam_pizza)
  let remaining_pasta_percent := 1 - ron_scott_pasta_percent
  let remaining_garlic_percent := 1 - ron_scott_mark_garlic_percent
  (remaining_pizza / remaining_friends = 6) ∧
  (remaining_pasta_percent / (total_people - 2) = 10 / 100) ∧
  (remaining_garlic_percent * garlic_breads / (total_people - 3) = 1.8) := by
  sorry

end NUMINAMATH_CALUDE_friend_consumption_l3290_329095


namespace NUMINAMATH_CALUDE_disjoint_quadratic_sets_l3290_329002

theorem disjoint_quadratic_sets (A B : ℤ) : 
  ∃ C : ℤ, (∀ x y : ℤ, x^2 + A*x + B ≠ 2*y^2 + 2*y + C) := by
  sorry

end NUMINAMATH_CALUDE_disjoint_quadratic_sets_l3290_329002


namespace NUMINAMATH_CALUDE_sum_of_two_primes_24_l3290_329006

theorem sum_of_two_primes_24 : ∃ p q : ℕ, Nat.Prime p ∧ Nat.Prime q ∧ p + q = 24 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_two_primes_24_l3290_329006


namespace NUMINAMATH_CALUDE_sqrt_inequality_l3290_329034

theorem sqrt_inequality (a : ℝ) (h : a ≥ 3) :
  Real.sqrt a - Real.sqrt (a - 2) < Real.sqrt (a - 1) - Real.sqrt (a - 3) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_inequality_l3290_329034


namespace NUMINAMATH_CALUDE_expression_evaluation_l3290_329069

theorem expression_evaluation : 
  (4 * 6) / (12 * 15) * (5 * 12 * 15^2) / (2 * 6 * 5) = 2.5 := by
sorry

end NUMINAMATH_CALUDE_expression_evaluation_l3290_329069


namespace NUMINAMATH_CALUDE_largest_two_digit_divisor_l3290_329084

def a : ℕ := 2^5 * 3^3 * 5^2 * 7

theorem largest_two_digit_divisor :
  (∀ d : ℕ, d > 96 → d < 100 → ¬(d ∣ a)) ∧ (96 ∣ a) := by sorry

end NUMINAMATH_CALUDE_largest_two_digit_divisor_l3290_329084


namespace NUMINAMATH_CALUDE_fraction_denominator_problem_l3290_329004

theorem fraction_denominator_problem (y : ℝ) (x : ℝ) (h1 : y > 0) 
  (h2 : (y / 20) + (3 * y / x) = 0.35 * y) : x = 10 := by
  sorry

end NUMINAMATH_CALUDE_fraction_denominator_problem_l3290_329004


namespace NUMINAMATH_CALUDE_bakery_doughnuts_l3290_329073

/-- The number of doughnuts in each box -/
def doughnuts_per_box : ℕ := 10

/-- The number of boxes sold -/
def boxes_sold : ℕ := 27

/-- The number of doughnuts given away -/
def doughnuts_given_away : ℕ := 30

/-- The total number of doughnuts made by the bakery -/
def total_doughnuts : ℕ := doughnuts_per_box * boxes_sold + doughnuts_given_away

theorem bakery_doughnuts : total_doughnuts = 300 := by
  sorry

end NUMINAMATH_CALUDE_bakery_doughnuts_l3290_329073


namespace NUMINAMATH_CALUDE_taxi_average_speed_l3290_329007

/-- The average speed of a taxi that travels 100 kilometers in 1 hour and 15 minutes is 80 kilometers per hour. -/
theorem taxi_average_speed :
  let distance : ℝ := 100 -- distance in kilometers
  let time : ℝ := 1.25 -- time in hours (1 hour and 15 minutes = 1.25 hours)
  let average_speed := distance / time
  average_speed = 80 := by sorry

end NUMINAMATH_CALUDE_taxi_average_speed_l3290_329007


namespace NUMINAMATH_CALUDE_problem_2023_l3290_329011

theorem problem_2023 : (2023^2 - 2023) / 2023 = 2022 := by
  sorry

end NUMINAMATH_CALUDE_problem_2023_l3290_329011


namespace NUMINAMATH_CALUDE_max_distance_ellipse_circle_l3290_329037

/-- The maximum distance between a point on the ellipse x²/9 + y² = 1
    and a point on the circle (x-4)² + y² = 1 is 8 -/
theorem max_distance_ellipse_circle : 
  ∃ (max_dist : ℝ),
    max_dist = 8 ∧
    ∀ (P Q : ℝ × ℝ),
      (P.1^2 / 9 + P.2^2 = 1) →
      ((Q.1 - 4)^2 + Q.2^2 = 1) →
      Real.sqrt ((P.1 - Q.1)^2 + (P.2 - Q.2)^2) ≤ max_dist :=
by sorry

end NUMINAMATH_CALUDE_max_distance_ellipse_circle_l3290_329037


namespace NUMINAMATH_CALUDE_total_notes_count_l3290_329012

/-- Proves that given a total amount of 480 rupees in equal numbers of one-rupee, five-rupee, and ten-rupee notes, the total number of notes is 90. -/
theorem total_notes_count (total_amount : ℕ) (note_count : ℕ) : 
  total_amount = 480 →
  note_count * 1 + note_count * 5 + note_count * 10 = total_amount →
  3 * note_count = 90 :=
by
  sorry

#check total_notes_count

end NUMINAMATH_CALUDE_total_notes_count_l3290_329012


namespace NUMINAMATH_CALUDE_cube_occupation_percentage_l3290_329070

/-- Represents the dimensions of a rectangular box in inches -/
structure BoxDimensions where
  length : ℚ
  width : ℚ
  height : ℚ

/-- Represents the side length of a cube in inches -/
def CubeSideLength : ℚ := 3

/-- The dimensions of the given box -/
def givenBox : BoxDimensions := ⟨6, 5, 10⟩

/-- Calculates the volume of a box given its dimensions -/
def boxVolume (box : BoxDimensions) : ℚ :=
  box.length * box.width * box.height

/-- Calculates the largest dimensions that can be filled with cubes -/
def largestFillableDimensions (box : BoxDimensions) (cubeSize : ℚ) : BoxDimensions :=
  ⟨
    (box.length / cubeSize).floor * cubeSize,
    (box.width / cubeSize).floor * cubeSize,
    (box.height / cubeSize).floor * cubeSize
  ⟩

/-- Calculates the percentage of the box volume occupied by cubes -/
def percentageOccupied (box : BoxDimensions) (cubeSize : ℚ) : ℚ :=
  let fillableBox := largestFillableDimensions box cubeSize
  (boxVolume fillableBox) / (boxVolume box) * 100

theorem cube_occupation_percentage :
  percentageOccupied givenBox CubeSideLength = 54 := by
  sorry

end NUMINAMATH_CALUDE_cube_occupation_percentage_l3290_329070


namespace NUMINAMATH_CALUDE_max_area_prime_sides_l3290_329008

/-- A prime number is a natural number greater than 1 that has no positive divisors other than 1 and itself. -/
def isPrime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ m : ℕ, m > 0 → m < n → n % m ≠ 0

/-- The perimeter of the rectangle is 40 meters. -/
def perimeter : ℕ := 40

/-- The theorem stating that the maximum area of a rectangular enclosure with prime side lengths and a perimeter of 40 meters is 91 square meters. -/
theorem max_area_prime_sides : 
  ∀ l w : ℕ, 
    isPrime l → 
    isPrime w → 
    l + w = perimeter / 2 → 
    l * w ≤ 91 :=
sorry

end NUMINAMATH_CALUDE_max_area_prime_sides_l3290_329008


namespace NUMINAMATH_CALUDE_tangent_line_circle_l3290_329097

theorem tangent_line_circle (R : ℝ) : 
  R > 0 → 
  (∃ x y : ℝ, x + y = 2 * R ∧ (x - 1)^2 + y^2 = R ∧ 
    ∀ x' y' : ℝ, x' + y' = 2 * R → (x' - 1)^2 + y'^2 ≥ R) →
  R = (3 + Real.sqrt 5) / 4 ∨ R = (3 - Real.sqrt 5) / 4 := by
sorry

end NUMINAMATH_CALUDE_tangent_line_circle_l3290_329097


namespace NUMINAMATH_CALUDE_scout_troop_profit_is_480_l3290_329051

/-- Calculates the profit of a scout troop selling candy bars -/
def scout_troop_profit (total_bars : ℕ) (cost_per_six : ℚ) (discount_rate : ℚ)
  (price_first_tier : ℚ) (price_second_tier : ℚ) (first_tier_limit : ℕ) : ℚ :=
  let cost_per_bar := cost_per_six / 6
  let total_cost := total_bars * cost_per_bar
  let discounted_cost := total_cost * (1 - discount_rate)
  let revenue_first_tier := min first_tier_limit total_bars * price_first_tier
  let revenue_second_tier := max 0 (total_bars - first_tier_limit) * price_second_tier
  let total_revenue := revenue_first_tier + revenue_second_tier
  total_revenue - discounted_cost

theorem scout_troop_profit_is_480 :
  scout_troop_profit 1200 3 (5/100) 1 (3/4) 600 = 480 := by
  sorry

end NUMINAMATH_CALUDE_scout_troop_profit_is_480_l3290_329051


namespace NUMINAMATH_CALUDE_last_passenger_probability_l3290_329058

/-- Represents a bus with n seats and n passengers -/
structure Bus (n : ℕ) where
  seats : Fin n → Passenger
  tickets : Fin n → Passenger

/-- Represents a passenger -/
inductive Passenger
| scientist
| regular (i : ℕ)

/-- The seating strategy for passengers -/
def seatingStrategy (b : Bus n) : Bus n := sorry

/-- The probability that the last passenger sits in their assigned seat -/
def lastPassengerProbability (n : ℕ) : ℚ :=
  if n < 2 then 0 else 1/2

/-- Theorem stating that the probability of the last passenger sitting in their assigned seat is 1/2 for n ≥ 2 -/
theorem last_passenger_probability (n : ℕ) (h : n ≥ 2) :
  lastPassengerProbability n = 1/2 := by sorry

end NUMINAMATH_CALUDE_last_passenger_probability_l3290_329058


namespace NUMINAMATH_CALUDE_corrected_mean_l3290_329000

theorem corrected_mean (n : ℕ) (initial_mean : ℚ) (incorrect_value correct_value : ℚ) :
  n = 50 →
  initial_mean = 36 →
  incorrect_value = 23 →
  correct_value = 48 →
  let total_sum := n * initial_mean
  let corrected_sum := total_sum - incorrect_value + correct_value
  corrected_sum / n = 36.5 := by
  sorry

end NUMINAMATH_CALUDE_corrected_mean_l3290_329000


namespace NUMINAMATH_CALUDE_lcm_of_20_45_75_l3290_329043

theorem lcm_of_20_45_75 : Nat.lcm 20 (Nat.lcm 45 75) = 900 := by
  sorry

end NUMINAMATH_CALUDE_lcm_of_20_45_75_l3290_329043


namespace NUMINAMATH_CALUDE_rice_yields_variance_l3290_329072

def rice_yields : List ℝ := [9.8, 9.9, 10.1, 10, 10.2]

theorem rice_yields_variance : 
  let n : ℕ := rice_yields.length
  let mean : ℝ := rice_yields.sum / n
  let variance : ℝ := (rice_yields.map (fun x => (x - mean)^2)).sum / n
  variance = 0.02 := by sorry

end NUMINAMATH_CALUDE_rice_yields_variance_l3290_329072


namespace NUMINAMATH_CALUDE_molecular_weight_AlOH3_l3290_329062

/-- The atomic weight of Aluminum in g/mol -/
def atomic_weight_Al : ℝ := 26.98

/-- The atomic weight of Oxygen in g/mol -/
def atomic_weight_O : ℝ := 16.00

/-- The atomic weight of Hydrogen in g/mol -/
def atomic_weight_H : ℝ := 1.01

/-- The molecular weight of a compound in g/mol -/
def molecular_weight (n_Al n_O n_H : ℕ) : ℝ :=
  n_Al * atomic_weight_Al + n_O * atomic_weight_O + n_H * atomic_weight_H

/-- The molecular weight of Al(OH)3 is 78.01 g/mol -/
theorem molecular_weight_AlOH3 :
  molecular_weight 1 3 3 = 78.01 := by sorry

end NUMINAMATH_CALUDE_molecular_weight_AlOH3_l3290_329062


namespace NUMINAMATH_CALUDE_yellow_candy_percentage_l3290_329049

theorem yellow_candy_percentage :
  ∀ (r b y : ℝ),
  r + b + y = 1 →
  y = 1.14 * b →
  r = 0.86 * b →
  y = 0.38 :=
by
  sorry

end NUMINAMATH_CALUDE_yellow_candy_percentage_l3290_329049


namespace NUMINAMATH_CALUDE_original_salary_calculation_l3290_329050

/-- Proves that if a salary S is increased by 2% to result in €10,200, then S equals €10,000. -/
theorem original_salary_calculation (S : ℝ) : S * 1.02 = 10200 → S = 10000 := by
  sorry

end NUMINAMATH_CALUDE_original_salary_calculation_l3290_329050


namespace NUMINAMATH_CALUDE_inequality_proof_l3290_329088

theorem inequality_proof (x : ℝ) : (x - 5) / ((x - 3)^2 + 1) < 0 ↔ x < 5 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3290_329088


namespace NUMINAMATH_CALUDE_D_144_l3290_329071

/-- 
D(n) represents the number of ways to write a positive integer n as a product of 
integers greater than 1, where the order of factors matters.
-/
def D (n : ℕ+) : ℕ := sorry

/-- The main theorem stating that D(144) = 41 -/
theorem D_144 : D 144 = 41 := by sorry

end NUMINAMATH_CALUDE_D_144_l3290_329071


namespace NUMINAMATH_CALUDE_lucky_larry_coincidence_l3290_329074

theorem lucky_larry_coincidence : ∃ e : ℝ, 
  let a : ℝ := 5
  let b : ℝ := 3
  let c : ℝ := 4
  let d : ℝ := 2
  (a + b - c + d - e) = (a + (b - (c + (d - e)))) := by
  sorry

end NUMINAMATH_CALUDE_lucky_larry_coincidence_l3290_329074


namespace NUMINAMATH_CALUDE_average_score_is_490_l3290_329080

-- Define the maximum score
def max_score : ℕ := 700

-- Define the number of students
def num_students : ℕ := 4

-- Define the scores as percentages
def gibi_percent : ℕ := 59
def jigi_percent : ℕ := 55
def mike_percent : ℕ := 99
def lizzy_percent : ℕ := 67

-- Define a function to calculate the actual score from a percentage
def calculate_score (percent : ℕ) : ℕ :=
  (percent * max_score) / 100

-- Theorem to prove
theorem average_score_is_490 : 
  (calculate_score gibi_percent + calculate_score jigi_percent + 
   calculate_score mike_percent + calculate_score lizzy_percent) / num_students = 490 :=
by sorry

end NUMINAMATH_CALUDE_average_score_is_490_l3290_329080


namespace NUMINAMATH_CALUDE_figure_x_value_l3290_329068

/-- Given a figure composed of two squares, a right triangle, and a rectangle,
    where:
    - The right triangle has legs measuring 3x and 4x
    - One square has a side length of 4x
    - Another square has a side length of 6x
    - The rectangle has length 3x and width x
    - The total area of the figure is 1100 square inches
    Prove that the value of x is √(1100/61) -/
theorem figure_x_value :
  ∀ x : ℝ,
  (4*x)^2 + (6*x)^2 + (1/2 * 3*x * 4*x) + (3*x * x) = 1100 →
  x = Real.sqrt (1100 / 61) :=
by sorry

end NUMINAMATH_CALUDE_figure_x_value_l3290_329068


namespace NUMINAMATH_CALUDE_fraction_equality_l3290_329040

theorem fraction_equality (a b : ℝ) (h : a / b = 2 / 3) : a / (a + b) = 2 / 5 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l3290_329040


namespace NUMINAMATH_CALUDE_no_seven_edge_polyhedron_l3290_329022

/-- A polyhedron in three-dimensional space. -/
structure Polyhedron where
  vertices : ℕ
  edges : ℕ
  faces : ℕ
  euler : vertices - edges + faces = 2
  min_degree : edges * 2 ≥ vertices * 3

/-- Theorem stating that no polyhedron can have exactly seven edges. -/
theorem no_seven_edge_polyhedron :
  ¬∃ (p : Polyhedron), p.edges = 7 := by
  sorry

end NUMINAMATH_CALUDE_no_seven_edge_polyhedron_l3290_329022


namespace NUMINAMATH_CALUDE_impossible_all_white_l3290_329048

-- Define the grid as a function from coordinates to colors
def Grid := Fin 8 → Fin 8 → Bool

-- Define the initial grid configuration
def initial_grid : Grid :=
  fun i j => (i = 0 ∧ j = 0) ∨ (i = 0 ∧ j = 7) ∨ (i = 7 ∧ j = 0) ∨ (i = 7 ∧ j = 7)

-- Define a row flip operation
def flip_row (g : Grid) (row : Fin 8) : Grid :=
  fun i j => if i = row then !g i j else g i j

-- Define a column flip operation
def flip_column (g : Grid) (col : Fin 8) : Grid :=
  fun i j => if j = col then !g i j else g i j

-- Define a predicate for an all-white grid
def all_white (g : Grid) : Prop :=
  ∀ i j, g i j = false

-- Theorem: It's impossible to achieve an all-white configuration
theorem impossible_all_white :
  ¬ ∃ (flips : List (Sum (Fin 8) (Fin 8))),
    all_white (flips.foldl (fun g flip => 
      match flip with
      | Sum.inl row => flip_row g row
      | Sum.inr col => flip_column g col
    ) initial_grid) :=
  sorry


end NUMINAMATH_CALUDE_impossible_all_white_l3290_329048


namespace NUMINAMATH_CALUDE_tylenol_interval_l3290_329075

-- Define the problem parameters
def total_hours : ℝ := 12
def tablet_mg : ℝ := 500
def tablets_per_dose : ℝ := 2
def total_grams : ℝ := 3

-- Define the theorem
theorem tylenol_interval :
  let total_mg : ℝ := total_grams * 1000
  let total_tablets : ℝ := total_mg / tablet_mg
  let intervals : ℝ := total_tablets - 1
  total_hours / intervals = 2.4 := by sorry

end NUMINAMATH_CALUDE_tylenol_interval_l3290_329075


namespace NUMINAMATH_CALUDE_hotdog_price_l3290_329030

/-- The cost of a hamburger -/
def hamburger_cost : ℝ := sorry

/-- The cost of a hot dog -/
def hotdog_cost : ℝ := sorry

/-- First day's purchase equation -/
axiom first_day : 3 * hamburger_cost + 4 * hotdog_cost = 10

/-- Second day's purchase equation -/
axiom second_day : 2 * hamburger_cost + 3 * hotdog_cost = 7

/-- Theorem stating that a hot dog costs 1 dollar -/
theorem hotdog_price : hotdog_cost = 1 := by sorry

end NUMINAMATH_CALUDE_hotdog_price_l3290_329030


namespace NUMINAMATH_CALUDE_right_triangle_altitude_relation_l3290_329025

theorem right_triangle_altitude_relation (a b c x : ℝ) 
  (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) (h4 : x > 0)
  (h5 : c^2 = a^2 + b^2)  -- Pythagorean theorem
  (h6 : a * b = c * x)    -- Area relation
  : 1 / x^2 = 1 / a^2 + 1 / b^2 := by
  sorry

#check right_triangle_altitude_relation

end NUMINAMATH_CALUDE_right_triangle_altitude_relation_l3290_329025


namespace NUMINAMATH_CALUDE_money_difference_l3290_329027

def derek_initial : ℕ := 40
def derek_expense1 : ℕ := 14
def derek_expense2 : ℕ := 11
def derek_expense3 : ℕ := 5
def dave_initial : ℕ := 50
def dave_expense : ℕ := 7

theorem money_difference :
  dave_initial - dave_expense - (derek_initial - derek_expense1 - derek_expense2 - derek_expense3) = 33 := by
  sorry

end NUMINAMATH_CALUDE_money_difference_l3290_329027


namespace NUMINAMATH_CALUDE_circle_center_coordinate_difference_l3290_329064

/-- Given two points that are the endpoints of a circle's diameter,
    calculate the difference between the x and y coordinates of the center. -/
theorem circle_center_coordinate_difference
  (p1 : ℝ × ℝ) (p2 : ℝ × ℝ)
  (h1 : p1 = (10, -6))
  (h2 : p2 = (-2, 2))
  : (p1.1 + p2.1) / 2 - (p1.2 + p2.2) / 2 = 6 := by
  sorry

end NUMINAMATH_CALUDE_circle_center_coordinate_difference_l3290_329064


namespace NUMINAMATH_CALUDE_discount_problem_l3290_329076

/-- The total cost after discount for a given number of toys, cost per toy, and discount percentage. -/
def totalCostAfterDiscount (numToys : ℕ) (costPerToy : ℚ) (discountPercentage : ℚ) : ℚ :=
  let totalCost := numToys * costPerToy
  let discountAmount := totalCost * (discountPercentage / 100)
  totalCost - discountAmount

/-- Theorem stating that the total cost after a 20% discount for 5 toys costing $3 each is $12. -/
theorem discount_problem : totalCostAfterDiscount 5 3 20 = 12 := by
  sorry

end NUMINAMATH_CALUDE_discount_problem_l3290_329076


namespace NUMINAMATH_CALUDE_original_curve_equation_l3290_329053

/-- Given a curve C in a Cartesian coordinate system that undergoes a stretching transformation,
    this theorem proves the equation of the original curve C. -/
theorem original_curve_equation
  (C : Set (ℝ × ℝ)) -- The original curve C
  (stretching : ℝ × ℝ → ℝ × ℝ) -- The stretching transformation
  (h_stretching : ∀ (x y : ℝ), stretching (x, y) = (3 * x, y)) -- Definition of the stretching
  (h_transformed : ∀ (x y : ℝ), (x, y) ∈ C → x^2 + 9*y^2 = 9) -- Equation of the transformed curve
  : ∀ (x y : ℝ), (x, y) ∈ C ↔ x^2 + y^2 = 1 := by sorry

end NUMINAMATH_CALUDE_original_curve_equation_l3290_329053


namespace NUMINAMATH_CALUDE_diagonals_29_sided_polygon_l3290_329023

/-- The number of diagonals in a convex polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- Theorem: A convex polygon with 29 sides has 377 diagonals -/
theorem diagonals_29_sided_polygon : num_diagonals 29 = 377 := by
  sorry

end NUMINAMATH_CALUDE_diagonals_29_sided_polygon_l3290_329023


namespace NUMINAMATH_CALUDE_sum_of_digits_M_l3290_329046

/-- The third smallest positive integer divisible by all integers less than 9 -/
def M : ℕ := sorry

/-- M is divisible by all positive integers less than 9 -/
axiom M_divisible (n : ℕ) (h : n > 0 ∧ n < 9) : M % n = 0

/-- M is the third smallest such integer -/
axiom M_third_smallest :
  ∀ k : ℕ, k > 0 ∧ k < M → (∀ n : ℕ, n > 0 ∧ n < 9 → k % n = 0) →
  ∃ j : ℕ, j > 0 ∧ j < M ∧ j ≠ k ∧ (∀ n : ℕ, n > 0 ∧ n < 9 → j % n = 0)

/-- Sum of digits function -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- The main theorem to prove -/
theorem sum_of_digits_M : sum_of_digits M = 9 := sorry

end NUMINAMATH_CALUDE_sum_of_digits_M_l3290_329046


namespace NUMINAMATH_CALUDE_quadratic_factorization_l3290_329017

theorem quadratic_factorization (x : ℝ) : 
  x^2 - 5*x + 3 = (x + 5/2 + Real.sqrt 13/2) * (x + 5/2 - Real.sqrt 13/2) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l3290_329017


namespace NUMINAMATH_CALUDE_kim_average_unchanged_l3290_329054

def kim_scores : List ℝ := [92, 86, 95, 89, 93]

theorem kim_average_unchanged (scores := kim_scores) :
  let first_three_avg := (scores.take 3).sum / 3
  let all_five_avg := scores.sum / 5
  all_five_avg - first_three_avg = 0 := by
sorry

end NUMINAMATH_CALUDE_kim_average_unchanged_l3290_329054


namespace NUMINAMATH_CALUDE_area_of_pentagon_l3290_329059

-- Define the points and lengths
structure Triangle :=
  (A B C : ℝ × ℝ)

def AB : ℝ := 5
def BC : ℝ := 3
def BD : ℝ := 3
def EC : ℝ := 1
def FD : ℝ := 2

-- Define the triangles
def triangleABC : Triangle := sorry
def triangleABD : Triangle := sorry

-- Define that ABC and ABD are right triangles
axiom ABC_right : triangleABC.C.1^2 + triangleABC.C.2^2 = AB^2
axiom ABD_right : triangleABD.C.1^2 + triangleABD.C.2^2 = AB^2

-- Define that C and D are on opposite sides of AB
axiom C_D_opposite : triangleABC.C.2 * triangleABD.C.2 < 0

-- Define points E and F
def E : ℝ × ℝ := sorry
def F : ℝ × ℝ := sorry

-- Define that E is on AC and F is on AD
axiom E_on_AC : ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ E = (t * triangleABC.A.1 + (1 - t) * triangleABC.C.1, t * triangleABC.A.2 + (1 - t) * triangleABC.C.2)
axiom F_on_AD : ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ F = (t * triangleABD.A.1 + (1 - t) * triangleABD.C.1, t * triangleABD.A.2 + (1 - t) * triangleABD.C.2)

-- Define the area of a polygon
def area (polygon : List (ℝ × ℝ)) : ℝ := sorry

-- Theorem to prove
theorem area_of_pentagon :
  area [E, C, B, D, F] = 303 / 25 := sorry

end NUMINAMATH_CALUDE_area_of_pentagon_l3290_329059


namespace NUMINAMATH_CALUDE_multiplication_value_proof_l3290_329029

theorem multiplication_value_proof (n r : ℚ) (hn : n = 9) (hr : r = 18) :
  ∃ x : ℚ, (n / 6) * x = r ∧ x = 12 := by
  sorry

end NUMINAMATH_CALUDE_multiplication_value_proof_l3290_329029


namespace NUMINAMATH_CALUDE_triangle_inequality_l3290_329047

/-- Given an acute triangle ABC with circumradius 1, 
    prove that the sum of the ratios of each side to (1 - sine of its opposite angle) 
    is greater than or equal to 18 + 12√3 -/
theorem triangle_inequality (A B C : ℝ) (a b c : ℝ) : 
  0 < A ∧ 0 < B ∧ 0 < C ∧ 
  A + B + C = π ∧
  A < π/2 ∧ B < π/2 ∧ C < π/2 ∧
  a = 2 * Real.sin A ∧
  b = 2 * Real.sin B ∧
  c = 2 * Real.sin C →
  (a / (1 - Real.sin A)) + (b / (1 - Real.sin B)) + (c / (1 - Real.sin C)) ≥ 18 + 12 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_l3290_329047


namespace NUMINAMATH_CALUDE_specific_trapezoid_area_l3290_329067

/-- A trapezoid with an inscribed circle -/
structure InscribedCircleTrapezoid where
  /-- The length of the shorter base of the trapezoid -/
  shorterBase : ℝ
  /-- The length of the longer segment of the non-parallel side divided by the point of tangency -/
  longerSegment : ℝ
  /-- The length of the shorter segment of the non-parallel side divided by the point of tangency -/
  shorterSegment : ℝ
  /-- The shorter base is positive -/
  shorterBase_pos : 0 < shorterBase
  /-- The longer segment is positive -/
  longerSegment_pos : 0 < longerSegment
  /-- The shorter segment is positive -/
  shorterSegment_pos : 0 < shorterSegment

/-- The area of the trapezoid -/
def area (t : InscribedCircleTrapezoid) : ℝ := sorry

/-- Theorem stating that the area of the specific trapezoid is 198 -/
theorem specific_trapezoid_area :
  ∀ t : InscribedCircleTrapezoid,
  t.shorterBase = 6 ∧ t.longerSegment = 9 ∧ t.shorterSegment = 4 →
  area t = 198 := by
  sorry

end NUMINAMATH_CALUDE_specific_trapezoid_area_l3290_329067


namespace NUMINAMATH_CALUDE_complex_imaginary_problem_l3290_329014

/-- A complex number is purely imaginary if its real part is zero -/
def isPurelyImaginary (z : ℂ) : Prop := z.re = 0

theorem complex_imaginary_problem (z : ℂ) 
  (h1 : isPurelyImaginary z) 
  (h2 : isPurelyImaginary ((z + 1)^2 - 2*I)) : 
  z = -I := by sorry

end NUMINAMATH_CALUDE_complex_imaginary_problem_l3290_329014


namespace NUMINAMATH_CALUDE_quadratic_inequality_l3290_329086

theorem quadratic_inequality (x : ℝ) : x^2 + 7*x + 6 < 0 ↔ -6 < x ∧ x < -1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_l3290_329086


namespace NUMINAMATH_CALUDE_danny_collection_difference_l3290_329036

/-- Represents Danny's collection of bottle caps and wrappers --/
structure Collection where
  park_caps : ℕ
  park_wrappers : ℕ
  beach_caps : ℕ
  beach_wrappers : ℕ
  forest_caps : ℕ
  forest_wrappers : ℕ
  previous_caps : ℕ
  previous_wrappers : ℕ

/-- Calculates the total number of bottle caps in the collection --/
def total_caps (c : Collection) : ℕ :=
  c.park_caps + c.beach_caps + c.forest_caps + c.previous_caps

/-- Calculates the total number of wrappers in the collection --/
def total_wrappers (c : Collection) : ℕ :=
  c.park_wrappers + c.beach_wrappers + c.forest_wrappers + c.previous_wrappers

/-- Theorem stating the difference between bottle caps and wrappers in Danny's collection --/
theorem danny_collection_difference :
  ∀ (c : Collection),
  c.park_caps = 58 →
  c.park_wrappers = 25 →
  c.beach_caps = 34 →
  c.beach_wrappers = 15 →
  c.forest_caps = 21 →
  c.forest_wrappers = 32 →
  c.previous_caps = 12 →
  c.previous_wrappers = 11 →
  total_caps c - total_wrappers c = 42 := by
  sorry

end NUMINAMATH_CALUDE_danny_collection_difference_l3290_329036


namespace NUMINAMATH_CALUDE_count_valid_pairs_l3290_329082

def is_valid_pair (x y : ℕ) : Prop :=
  Nat.Prime x ∧ Nat.Prime y ∧ x ≠ y ∧ (621 * x * y) % (x + y) = 0

theorem count_valid_pairs :
  ∃! (pairs : Finset (ℕ × ℕ)), 
    (∀ p ∈ pairs, is_valid_pair p.1 p.2) ∧ 
    (∀ x y, is_valid_pair x y → (x, y) ∈ pairs) ∧
    pairs.card = 6 := by
  sorry

end NUMINAMATH_CALUDE_count_valid_pairs_l3290_329082


namespace NUMINAMATH_CALUDE_equation_one_solutions_l3290_329078

theorem equation_one_solutions (x : ℝ) :
  3 * (x - 1)^2 = 12 ↔ x = 3 ∨ x = -1 := by sorry

end NUMINAMATH_CALUDE_equation_one_solutions_l3290_329078


namespace NUMINAMATH_CALUDE_value_of_a_l3290_329020

/-- Given a function f(x) = ax³ + 3x² - 6 where f'(-1) = 4, prove that a = 10/3 -/
theorem value_of_a (f : ℝ → ℝ) (a : ℝ) 
  (h1 : ∀ x, f x = a * x^3 + 3 * x^2 - 6)
  (h2 : deriv f (-1) = 4) : 
  a = 10/3 := by
  sorry

end NUMINAMATH_CALUDE_value_of_a_l3290_329020


namespace NUMINAMATH_CALUDE_sam_yellow_marbles_l3290_329099

/-- The number of yellow marbles Sam has after receiving more from Joan -/
def total_yellow_marbles (initial : Real) (received : Real) : Real :=
  initial + received

/-- Theorem stating that Sam now has 111.0 yellow marbles -/
theorem sam_yellow_marbles :
  total_yellow_marbles 86.0 25.0 = 111.0 := by
  sorry

end NUMINAMATH_CALUDE_sam_yellow_marbles_l3290_329099


namespace NUMINAMATH_CALUDE_parallelogram_iff_opposite_sides_equal_l3290_329038

structure Quadrilateral where
  vertices : Fin 4 → ℝ × ℝ

def opposite_sides_equal (q : Quadrilateral) : Prop :=
  (q.vertices 0 = q.vertices 2) ∧ (q.vertices 1 = q.vertices 3)

def is_parallelogram (q : Quadrilateral) : Prop :=
  ∃ (v : ℝ × ℝ), (q.vertices 1 - q.vertices 0 = v) ∧ (q.vertices 2 - q.vertices 3 = v) ∧
                 (q.vertices 3 - q.vertices 0 = v) ∧ (q.vertices 2 - q.vertices 1 = v)

theorem parallelogram_iff_opposite_sides_equal (q : Quadrilateral) :
  is_parallelogram q ↔ opposite_sides_equal q := by sorry

end NUMINAMATH_CALUDE_parallelogram_iff_opposite_sides_equal_l3290_329038


namespace NUMINAMATH_CALUDE_range_of_a_l3290_329065

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, 0 ≤ x ∧ x ≤ 2 → a < -x^2 + 2*x) → 
  (∀ y : ℝ, y < 0 → ∃ x : ℝ, 0 ≤ x ∧ x ≤ 2 ∧ y < -x^2 + 2*x) ∧ 
  (∀ z : ℝ, z ≥ 0 → ∃ w : ℝ, 0 ≤ w ∧ w ≤ 2 ∧ z ≥ -w^2 + 2*w) :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l3290_329065


namespace NUMINAMATH_CALUDE_solution_is_three_fourths_l3290_329063

/-- The sum of the series given the value of x -/
def seriesSum (x : ℝ) : ℝ := 1 + 4*x + 8*x^2 + 12*x^3 + 16*x^4 + 20*x^5 + 24*x^6 + 28*x^7 + 32*x^8 + 36*x^9 + 40*x^10

/-- The theorem stating that 3/4 is the solution to the equation -/
theorem solution_is_three_fourths :
  ∃ (x : ℝ), x = 3/4 ∧ seriesSum x = 76 ∧ abs x < 1 := by
  sorry

end NUMINAMATH_CALUDE_solution_is_three_fourths_l3290_329063


namespace NUMINAMATH_CALUDE_campers_rowing_difference_l3290_329009

theorem campers_rowing_difference (morning_campers afternoon_campers evening_campers : ℕ) 
  (h1 : morning_campers = 44)
  (h2 : afternoon_campers = 39)
  (h3 : evening_campers = 31) :
  morning_campers - afternoon_campers = 5 := by
  sorry

end NUMINAMATH_CALUDE_campers_rowing_difference_l3290_329009


namespace NUMINAMATH_CALUDE_cos_eight_alpha_in_right_triangle_l3290_329085

theorem cos_eight_alpha_in_right_triangle (α : ℝ) :
  (∃ (a b c : ℝ), a = 1 ∧ b = Real.sqrt 2 ∧ c^2 = a^2 + b^2 ∧ 
   α = Real.arccos (b / c) ∧ 0 < α ∧ α < π/4) →
  Real.cos (8 * α) = 17/81 := by
sorry

end NUMINAMATH_CALUDE_cos_eight_alpha_in_right_triangle_l3290_329085


namespace NUMINAMATH_CALUDE_inscribed_circle_radius_for_given_triangle_l3290_329003

/-- Represents a triangle with an inscribed circle -/
structure TriangleWithInscribedCircle where
  /-- Distance from vertex A to side BC -/
  h_a : ℝ
  /-- Sum of distances from B to AC and from C to AB -/
  h_b_plus_h_c : ℝ
  /-- Radius of the inscribed circle -/
  r : ℝ
  /-- The radius satisfies the relationship with heights -/
  radius_height_relation : 1 / r = 1 / h_a + 2 / h_b_plus_h_c

/-- The theorem stating the radius of the inscribed circle for the given triangle -/
theorem inscribed_circle_radius_for_given_triangle :
  ∀ (t : TriangleWithInscribedCircle),
    t.h_a = 100 ∧ t.h_b_plus_h_c = 300 →
    t.r = 300 / 7 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_circle_radius_for_given_triangle_l3290_329003


namespace NUMINAMATH_CALUDE_tank_plastering_cost_l3290_329021

/-- Calculates the cost of plastering a rectangular tank's walls and bottom -/
def plasteringCost (length width depth rate : ℝ) : ℝ :=
  let wallArea := 2 * (length * depth + width * depth)
  let bottomArea := length * width
  let totalArea := wallArea + bottomArea
  totalArea * rate

/-- Theorem stating the cost of plastering the given tank -/
theorem tank_plastering_cost :
  plasteringCost 25 12 6 0.75 = 558 := by
  sorry

#eval plasteringCost 25 12 6 0.75

end NUMINAMATH_CALUDE_tank_plastering_cost_l3290_329021


namespace NUMINAMATH_CALUDE_trigonometric_equation_solution_l3290_329026

theorem trigonometric_equation_solution (t : ℝ) : 
  (16 * Real.sin (t / 2) - 25 * Real.cos (t / 2) ≠ 0) →
  (40 * (Real.sin (t / 2) ^ 3 - Real.cos (t / 2) ^ 3) / 
   (16 * Real.sin (t / 2) - 25 * Real.cos (t / 2)) = Real.sin t) ↔
  (∃ k : ℤ, t = 2 * Real.arctan (4 / 5) + 2 * Real.pi * ↑k) := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_equation_solution_l3290_329026


namespace NUMINAMATH_CALUDE_repeating_decimal_to_fraction_l3290_329018

theorem repeating_decimal_to_fraction :
  ∃ (x : ℚ), x = 0.7333333333333333 ∧ x = 11 / 15 := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_to_fraction_l3290_329018


namespace NUMINAMATH_CALUDE_total_population_avalon_l3290_329098

theorem total_population_avalon (num_towns : ℕ) (avg_lower avg_upper : ℝ) :
  num_towns = 25 →
  5400 ≤ avg_lower →
  avg_upper ≤ 5700 →
  avg_lower ≤ (avg_lower + avg_upper) / 2 →
  (avg_lower + avg_upper) / 2 ≤ avg_upper →
  ∃ (total_population : ℝ),
    total_population = num_towns * ((avg_lower + avg_upper) / 2) ∧
    total_population = 138750 :=
by sorry

end NUMINAMATH_CALUDE_total_population_avalon_l3290_329098


namespace NUMINAMATH_CALUDE_soda_cost_calculation_l3290_329057

theorem soda_cost_calculation (total_cost sandwich_cost : ℚ) 
  (num_sandwiches num_sodas : ℕ) :
  total_cost = 6.46 →
  sandwich_cost = 1.49 →
  num_sandwiches = 2 →
  num_sodas = 4 →
  (total_cost - (↑num_sandwiches * sandwich_cost)) / ↑num_sodas = 0.87 := by
  sorry

end NUMINAMATH_CALUDE_soda_cost_calculation_l3290_329057


namespace NUMINAMATH_CALUDE_line_intersection_l3290_329019

theorem line_intersection :
  ∃! p : ℝ × ℝ, 
    (p.2 = -3 * p.1 + 1) ∧ 
    (p.2 + 1 = 15 * p.1) ∧ 
    p = (1/9, 2/3) := by
  sorry

end NUMINAMATH_CALUDE_line_intersection_l3290_329019


namespace NUMINAMATH_CALUDE_bob_water_usage_percentage_l3290_329032

/-- Represents a farmer with their crop acreages -/
structure Farmer where
  corn_acres : ℝ
  cotton_acres : ℝ
  bean_acres : ℝ

/-- Represents water requirements for different crops -/
structure WaterRequirements where
  corn_gallons_per_acre : ℝ
  cotton_gallons_per_acre : ℝ
  bean_gallons_per_acre : ℝ

/-- Calculates the total water usage for a farmer -/
def water_usage (f : Farmer) (w : WaterRequirements) : ℝ :=
  f.corn_acres * w.corn_gallons_per_acre +
  f.cotton_acres * w.cotton_gallons_per_acre +
  f.bean_acres * w.bean_gallons_per_acre

/-- Theorem: The percentage of total water used by Farmer Bob is 36% -/
theorem bob_water_usage_percentage
  (bob : Farmer)
  (brenda : Farmer)
  (bernie : Farmer)
  (water_req : WaterRequirements)
  (h1 : bob.corn_acres = 3 ∧ bob.cotton_acres = 9 ∧ bob.bean_acres = 12)
  (h2 : brenda.corn_acres = 6 ∧ brenda.cotton_acres = 7 ∧ brenda.bean_acres = 14)
  (h3 : bernie.corn_acres = 2 ∧ bernie.cotton_acres = 12 ∧ bernie.bean_acres = 0)
  (h4 : water_req.corn_gallons_per_acre = 20)
  (h5 : water_req.cotton_gallons_per_acre = 80)
  (h6 : water_req.bean_gallons_per_acre = 2 * water_req.corn_gallons_per_acre) :
  (water_usage bob water_req) / (water_usage bob water_req + water_usage brenda water_req + water_usage bernie water_req) = 0.36 := by
  sorry


end NUMINAMATH_CALUDE_bob_water_usage_percentage_l3290_329032


namespace NUMINAMATH_CALUDE_elizabeth_study_time_l3290_329090

/-- Given that Elizabeth studied for a total of 60 minutes, including 35 minutes for math,
    prove that she studied for 25 minutes for science. -/
theorem elizabeth_study_time (total_time math_time science_time : ℕ) : 
  total_time = 60 ∧ math_time = 35 ∧ total_time = math_time + science_time →
  science_time = 25 := by
sorry

end NUMINAMATH_CALUDE_elizabeth_study_time_l3290_329090


namespace NUMINAMATH_CALUDE_cousin_payment_l3290_329096

def friend_payment : ℕ := 5
def brother_payment : ℕ := 8
def total_days : ℕ := 7
def total_amount : ℕ := 119

theorem cousin_payment (cousin_pay : ℕ) : 
  (friend_payment * total_days + brother_payment * total_days + cousin_pay * total_days = total_amount) →
  cousin_pay = 4 := by
sorry

end NUMINAMATH_CALUDE_cousin_payment_l3290_329096


namespace NUMINAMATH_CALUDE_magical_gate_diameter_l3290_329052

theorem magical_gate_diameter :
  let circle_equation := fun (x y : ℝ) => x^2 + y^2 + 2*x - 4*y + 3 = 0
  ∃ (center : ℝ × ℝ) (radius : ℝ),
    (∀ x y, circle_equation x y ↔ (x - center.1)^2 + (y - center.2)^2 = radius^2) ∧
    2 * radius = 2 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_magical_gate_diameter_l3290_329052


namespace NUMINAMATH_CALUDE_solution_value_l3290_329089

theorem solution_value (k : ℝ) : (2 * 3 - k + 1 = 0) → k = 7 := by
  sorry

end NUMINAMATH_CALUDE_solution_value_l3290_329089


namespace NUMINAMATH_CALUDE_composite_expression_l3290_329024

theorem composite_expression (a b : ℕ) : 
  ∃ (p q : ℕ), p > 1 ∧ q > 1 ∧ 4*a^2 + 4*a*b + 4*a + 2*b + 1 = p * q :=
by sorry

end NUMINAMATH_CALUDE_composite_expression_l3290_329024


namespace NUMINAMATH_CALUDE_sqrt_450_simplified_l3290_329094

theorem sqrt_450_simplified : Real.sqrt 450 = 15 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_450_simplified_l3290_329094


namespace NUMINAMATH_CALUDE_greatest_difference_l3290_329031

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

def is_perfect_square (n : ℕ) : Prop := ∃ m : ℕ, n = m * m

theorem greatest_difference (x y : ℕ) 
  (hx_lower : 6 < x) (hx_upper : x < 10)
  (hy_lower : 10 < y) (hy_upper : y < 17)
  (hx_prime : is_prime x)
  (hy_square : is_perfect_square y) :
  (∀ x' y' : ℕ, 
    6 < x' → x' < 10 → 10 < y' → y' < 17 → 
    is_prime x' → is_perfect_square y' → 
    y' - x' ≤ y - x) ∧
  y - x = 9 :=
sorry

end NUMINAMATH_CALUDE_greatest_difference_l3290_329031


namespace NUMINAMATH_CALUDE_x_value_l3290_329045

theorem x_value : ∃ x : ℝ, 3 * x = (26 - x) + 18 ∧ x = 11 := by sorry

end NUMINAMATH_CALUDE_x_value_l3290_329045


namespace NUMINAMATH_CALUDE_arithmetic_sequence_property_l3290_329039

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_property (a : ℕ → ℝ) :
  (is_arithmetic_sequence a → a 1 + a 3 = 2 * a 2) ∧
  (∃ a : ℕ → ℝ, a 1 + a 3 = 2 * a 2 ∧ ¬is_arithmetic_sequence a) :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_property_l3290_329039


namespace NUMINAMATH_CALUDE_coordinate_sum_of_point_B_l3290_329066

/-- Given points A(0, 0) and B(x, 5) where the slope of AB is 3/4,
    prove that the sum of x- and y-coordinates of B is 35/3 -/
theorem coordinate_sum_of_point_B (x : ℚ) : 
  let A : ℚ × ℚ := (0, 0)
  let B : ℚ × ℚ := (x, 5)
  let slope : ℚ := (B.2 - A.2) / (B.1 - A.1)
  slope = 3/4 → x + 5 = 35/3 := by
  sorry

end NUMINAMATH_CALUDE_coordinate_sum_of_point_B_l3290_329066


namespace NUMINAMATH_CALUDE_no_equal_consecutive_digit_sums_l3290_329001

def sum_of_digits (n : ℕ) : ℕ :=
  (Nat.digits 10 n).sum

def S (n : ℕ) : ℕ :=
  sum_of_digits (2^n)

theorem no_equal_consecutive_digit_sums :
  ∀ n : ℕ, n > 0 → S (n + 1) ≠ S n :=
sorry

end NUMINAMATH_CALUDE_no_equal_consecutive_digit_sums_l3290_329001


namespace NUMINAMATH_CALUDE_cylinder_volume_ratio_l3290_329042

/-- Theorem: For two cylinders with given properties, the ratio of their volumes is 3/2 -/
theorem cylinder_volume_ratio (S₁ S₂ V₁ V₂ : ℝ) (h₁ : S₁ / S₂ = 9 / 4) (h₂ : S₁ > 0) (h₃ : S₂ > 0) : 
  ∃ (R r H h : ℝ), 
    R > 0 ∧ r > 0 ∧ H > 0 ∧ h > 0 ∧
    S₁ = π * R^2 ∧
    S₂ = π * r^2 ∧
    V₁ = π * R^2 * H ∧
    V₂ = π * r^2 * h ∧
    2 * π * R * H = 2 * π * r * h →
    V₁ / V₂ = 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cylinder_volume_ratio_l3290_329042


namespace NUMINAMATH_CALUDE_bryson_shoes_pairs_l3290_329077

/-- Given that Bryson has a total of 4 new shoes and a pair of shoes consists of 2 shoes,
    prove that the number of pairs of shoes he bought is 2. -/
theorem bryson_shoes_pairs : 
  ∀ (total_shoes : ℕ) (shoes_per_pair : ℕ),
    total_shoes = 4 →
    shoes_per_pair = 2 →
    total_shoes / shoes_per_pair = 2 := by
  sorry

end NUMINAMATH_CALUDE_bryson_shoes_pairs_l3290_329077


namespace NUMINAMATH_CALUDE_geometric_sequence_general_term_l3290_329028

/-- A geometric sequence with given third and tenth terms -/
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  (∃ q : ℝ, ∀ n : ℕ, a (n + 1) = q * a n) ∧ 
  a 3 = 3 ∧ 
  a 10 = 384

/-- The general term of the geometric sequence -/
def general_term (n : ℕ) : ℝ := 3 * 2^(n - 3)

/-- Theorem stating that the general term is correct for the given geometric sequence -/
theorem geometric_sequence_general_term (a : ℕ → ℝ) :
  geometric_sequence a → (∀ n : ℕ, a n = general_term n) :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_general_term_l3290_329028


namespace NUMINAMATH_CALUDE_shirts_not_washed_l3290_329056

theorem shirts_not_washed (short_sleeve : ℕ) (long_sleeve : ℕ) (washed : ℕ) : 
  short_sleeve = 9 → long_sleeve = 27 → washed = 20 → 
  short_sleeve + long_sleeve - washed = 16 := by
  sorry

end NUMINAMATH_CALUDE_shirts_not_washed_l3290_329056


namespace NUMINAMATH_CALUDE_symmetry_about_a_periodicity_l3290_329079

variable (f : ℝ → ℝ)
variable (a b : ℝ)

axiom a_nonzero : a ≠ 0
axiom b_diff_a : b ≠ a
axiom f_symmetry : ∀ x, f (a + x) = f (a - x)

theorem symmetry_about_a : ∀ x, f x = f (2*a - x) := by sorry

axiom symmetry_about_b : ∀ x, f x = f (2*b - x)

theorem periodicity : ∃ p : ℝ, p ≠ 0 ∧ ∀ x, f (x + p) = f x := by sorry

end NUMINAMATH_CALUDE_symmetry_about_a_periodicity_l3290_329079


namespace NUMINAMATH_CALUDE_investment_income_is_648_l3290_329013

/-- Calculates the annual income from a stock investment given the total investment,
    share face value, quoted price, and dividend rate. -/
def annual_income (total_investment : ℚ) (face_value : ℚ) (quoted_price : ℚ) (dividend_rate : ℚ) : ℚ :=
  let num_shares := total_investment / quoted_price
  let dividend_per_share := (dividend_rate / 100) * face_value
  num_shares * dividend_per_share

/-- Theorem stating that the annual income for the given investment scenario is 648. -/
theorem investment_income_is_648 :
  annual_income 4455 10 8.25 12 = 648 := by
  sorry

end NUMINAMATH_CALUDE_investment_income_is_648_l3290_329013


namespace NUMINAMATH_CALUDE_not_in_first_quadrant_l3290_329033

def linear_function (x : ℝ) : ℝ := -3 * x - 2

theorem not_in_first_quadrant :
  ∀ x y : ℝ, y = linear_function x → ¬(x > 0 ∧ y > 0) := by
  sorry

end NUMINAMATH_CALUDE_not_in_first_quadrant_l3290_329033


namespace NUMINAMATH_CALUDE_josh_marbles_l3290_329087

theorem josh_marbles (initial : ℕ) (lost : ℕ) (difference : ℕ) (found : ℕ) : 
  initial = 15 →
  lost = 23 →
  difference = 14 →
  lost = found + difference →
  found = 9 := by
sorry

end NUMINAMATH_CALUDE_josh_marbles_l3290_329087


namespace NUMINAMATH_CALUDE_inequality_holds_l3290_329010

theorem inequality_holds (a : ℝ) (h : a ≥ 7/2) : 
  ∀ θ : ℝ, 0 ≤ θ ∧ θ ≤ π/2 → 
    ∀ x : ℝ, (x + 3 + 2 * Real.sin θ * Real.cos θ)^2 + 
              (x + a * Real.sin θ + a * Real.cos θ)^2 ≥ 1/8 := by
  sorry

end NUMINAMATH_CALUDE_inequality_holds_l3290_329010


namespace NUMINAMATH_CALUDE_right_triangle_sets_l3290_329015

def is_right_triangle (a b c : ℕ) : Prop :=
  a^2 + b^2 = c^2 ∨ a^2 + c^2 = b^2 ∨ b^2 + c^2 = a^2

theorem right_triangle_sets :
  is_right_triangle 3 4 5 ∧
  ¬is_right_triangle 2 3 4 ∧
  ¬is_right_triangle 4 6 7 ∧
  ¬is_right_triangle 5 11 12 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_sets_l3290_329015


namespace NUMINAMATH_CALUDE_divisibility_sequence_l3290_329035

theorem divisibility_sequence (a : ℕ) : ∃ n : ℕ, ∀ k : ℕ, a ∣ (n^(n^k) + 1) := by
  sorry

end NUMINAMATH_CALUDE_divisibility_sequence_l3290_329035


namespace NUMINAMATH_CALUDE_percentage_problem_l3290_329044

theorem percentage_problem : ∃ p : ℝ, p > 0 ∧ p < 100 ∧ (p / 100) * 30 = (25 / 100) * 16 + 2 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l3290_329044


namespace NUMINAMATH_CALUDE_basic_structures_correct_l3290_329060

/-- The set of basic structures of an algorithm -/
def BasicStructures : Set String :=
  {"Sequential structure", "Conditional structure", "Loop structure"}

/-- The correct answer option -/
def CorrectAnswer : Set String :=
  {"Sequential structure", "Conditional structure", "Loop structure"}

/-- Theorem stating that the basic structures of an algorithm are correctly defined -/
theorem basic_structures_correct : BasicStructures = CorrectAnswer := by
  sorry

end NUMINAMATH_CALUDE_basic_structures_correct_l3290_329060


namespace NUMINAMATH_CALUDE_sum_reciprocals_l3290_329083

theorem sum_reciprocals (a b c d : ℝ) (ω : ℂ) 
  (ha : a ≠ -1) (hb : b ≠ -1) (hc : c ≠ -1) (hd : d ≠ -1)
  (hω1 : ω^4 = 1) (hω2 : ω ≠ 1)
  (h : 1 / (a + ω) + 1 / (b + ω) + 1 / (c + ω) + 1 / (d + ω) = 4 / ω^2) :
  1 / (a + 1) + 1 / (b + 1) + 1 / (c + 1) + 1 / (d + 1) = 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_reciprocals_l3290_329083


namespace NUMINAMATH_CALUDE_sqrt_sum_inequality_l3290_329055

theorem sqrt_sum_inequality (x y α : ℝ) 
  (h : Real.sqrt (1 + x) + Real.sqrt (1 + y) = 2 * Real.sqrt (1 + α)) : 
  x + y ≥ 2 * α := by
  sorry

end NUMINAMATH_CALUDE_sqrt_sum_inequality_l3290_329055


namespace NUMINAMATH_CALUDE_inverse_cube_root_relation_l3290_329061

/-- Given that z varies inversely as ∛w, prove that w = 1 when z = 6, 
    given that z = 3 when w = 8. -/
theorem inverse_cube_root_relation (z w : ℝ) (k : ℝ) 
  (h1 : ∀ w z, z * (w ^ (1/3 : ℝ)) = k)
  (h2 : 3 * (8 ^ (1/3 : ℝ)) = k)
  (h3 : 6 * (w ^ (1/3 : ℝ)) = k) : 
  w = 1 := by sorry

end NUMINAMATH_CALUDE_inverse_cube_root_relation_l3290_329061


namespace NUMINAMATH_CALUDE_equation_two_solutions_l3290_329041

/-- The equation has exactly two distinct solutions when k < -3/8 -/
theorem equation_two_solutions (k : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
    (x₁ - 3) / (k * x₁ + 2) = 2 * x₁ ∧ 
    (x₂ - 3) / (k * x₂ + 2) = 2 * x₂ ∧
    (∀ x : ℝ, (x - 3) / (k * x + 2) = 2 * x → x = x₁ ∨ x = x₂)) ↔ 
  k < -3/8 :=
sorry

end NUMINAMATH_CALUDE_equation_two_solutions_l3290_329041
