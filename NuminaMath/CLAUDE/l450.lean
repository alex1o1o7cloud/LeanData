import Mathlib

namespace NUMINAMATH_CALUDE_total_players_on_ground_l450_45033

theorem total_players_on_ground (cricket hockey football softball : ℕ) : 
  cricket = 15 → hockey = 12 → football = 13 → softball = 15 →
  cricket + hockey + football + softball = 55 := by
  sorry

end NUMINAMATH_CALUDE_total_players_on_ground_l450_45033


namespace NUMINAMATH_CALUDE_initial_water_ratio_l450_45068

/-- Proves that the ratio of initial water to tank capacity is 1:2 given the specified conditions --/
theorem initial_water_ratio (tank_capacity : ℝ) (inflow_rate : ℝ) (outflow_rate1 : ℝ) (outflow_rate2 : ℝ) (fill_time : ℝ) :
  tank_capacity = 6000 →
  inflow_rate = 500 →
  outflow_rate1 = 250 →
  outflow_rate2 = 1000 / 6 →
  fill_time = 36 →
  (tank_capacity - (inflow_rate - outflow_rate1 - outflow_rate2) * fill_time) / tank_capacity = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_initial_water_ratio_l450_45068


namespace NUMINAMATH_CALUDE_mrs_hilt_carnival_tickets_cost_l450_45012

/-- Represents the cost and quantity of carnival tickets --/
structure CarnivalTickets where
  kids_usual_cost : ℚ
  kids_usual_quantity : ℕ
  adults_usual_cost : ℚ
  adults_usual_quantity : ℕ
  kids_deal_cost : ℚ
  kids_deal_quantity : ℕ
  adults_deal_cost : ℚ
  adults_deal_quantity : ℕ
  kids_bought : ℕ
  adults_bought : ℕ

/-- Calculates the total cost of carnival tickets --/
def total_cost (tickets : CarnivalTickets) : ℚ :=
  let kids_deal_used := tickets.kids_bought / tickets.kids_deal_quantity
  let kids_usual_used := tickets.kids_bought % tickets.kids_deal_quantity / tickets.kids_usual_quantity
  let adults_deal_used := tickets.adults_bought / tickets.adults_deal_quantity
  let adults_usual_used := tickets.adults_bought % tickets.adults_deal_quantity / tickets.adults_usual_quantity
  kids_deal_used * tickets.kids_deal_cost +
  kids_usual_used * tickets.kids_usual_cost +
  adults_deal_used * tickets.adults_deal_cost +
  adults_usual_used * tickets.adults_usual_cost

/-- Theorem: The total cost of Mrs. Hilt's carnival tickets is $15 --/
theorem mrs_hilt_carnival_tickets_cost :
  let tickets : CarnivalTickets := {
    kids_usual_cost := 1/4,
    kids_usual_quantity := 4,
    adults_usual_cost := 2/3,
    adults_usual_quantity := 3,
    kids_deal_cost := 4,
    kids_deal_quantity := 20,
    adults_deal_cost := 8,
    adults_deal_quantity := 15,
    kids_bought := 24,
    adults_bought := 18
  }
  total_cost tickets = 15 := by sorry


end NUMINAMATH_CALUDE_mrs_hilt_carnival_tickets_cost_l450_45012


namespace NUMINAMATH_CALUDE_tournament_scheduling_correct_l450_45002

/-- Represents a team in the tournament --/
inductive Team (n : ℕ) where
  | num : Fin n → Team n
  | inf : Team n

/-- A match between two teams --/
structure Match (n : ℕ) where
  team1 : Team n
  team2 : Team n

/-- A round in the tournament --/
def Round (n : ℕ) := List (Match n)

/-- Generate the next round based on the current round --/
def nextRound (n : ℕ) (current : Round n) : Round n :=
  sorry

/-- Check if a round is valid (each team plays exactly once) --/
def isValidRound (n : ℕ) (round : Round n) : Prop :=
  sorry

/-- Check if two teams have played against each other --/
def havePlayedAgainst (n : ℕ) (team1 team2 : Team n) (rounds : List (Round n)) : Prop :=
  sorry

/-- The main theorem: tournament scheduling is correct --/
theorem tournament_scheduling_correct (n : ℕ) (h : n > 1) :
  ∃ (rounds : List (Round n)),
    (rounds.length = n - 1) ∧
    (∀ r ∈ rounds, isValidRound n r) ∧
    (∀ t1 t2 : Team n, t1 ≠ t2 → havePlayedAgainst n t1 t2 rounds) :=
  sorry

end NUMINAMATH_CALUDE_tournament_scheduling_correct_l450_45002


namespace NUMINAMATH_CALUDE_circle_and_line_properties_l450_45006

-- Define the circle C
def circle_C (x y : ℝ) : Prop :=
  ∃ (t : ℝ), (x + 3)^2 + (y + 2)^2 = 25 ∧ t - y + 1 = 0

-- Define the line m
def line_m (x y : ℝ) : Prop :=
  y = (5/12) * x + 43/12

theorem circle_and_line_properties :
  -- Circle C passes through (0,2) and (2,-2)
  circle_C 0 2 ∧ circle_C 2 (-2) ∧
  -- Line m passes through (1,4)
  line_m 1 4 ∧
  -- The chord length of the intersection between circle C and line m is 6
  (∃ (x₁ y₁ x₂ y₂ : ℝ),
    circle_C x₁ y₁ ∧ circle_C x₂ y₂ ∧
    line_m x₁ y₁ ∧ line_m x₂ y₂ ∧
    (x₁ - x₂)^2 + (y₁ - y₂)^2 = 36) :=
by
  sorry

#check circle_and_line_properties

end NUMINAMATH_CALUDE_circle_and_line_properties_l450_45006


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l450_45001

/-- An arithmetic sequence -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- Theorem: For an arithmetic sequence, if a₁ + a₉ = 8, then a₂ + a₈ = 8 -/
theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  arithmetic_sequence a → a 1 + a 9 = 8 → a 2 + a 8 = 8 := by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l450_45001


namespace NUMINAMATH_CALUDE_ron_multiplication_mistake_l450_45075

theorem ron_multiplication_mistake (a b : ℕ) : 
  10 ≤ a ∧ a < 100 →  -- a is a two-digit number
  b < 10 →            -- b is a single-digit number
  a * (b + 10) = 190 →
  a * b = 0 := by
sorry

end NUMINAMATH_CALUDE_ron_multiplication_mistake_l450_45075


namespace NUMINAMATH_CALUDE_fencing_cost_proof_l450_45020

/-- Calculates the total cost of fencing a rectangular plot -/
def total_fencing_cost (length breadth cost_per_meter : ℝ) : ℝ :=
  2 * (length + breadth) * cost_per_meter

/-- Proves that the total cost of fencing the given rectangular plot is 5300 -/
theorem fencing_cost_proof (length breadth cost_per_meter : ℝ) 
  (h1 : length = 64)
  (h2 : breadth = length - 28)
  (h3 : cost_per_meter = 26.5) :
  total_fencing_cost length breadth cost_per_meter = 5300 := by
  sorry

#eval total_fencing_cost 64 36 26.5

end NUMINAMATH_CALUDE_fencing_cost_proof_l450_45020


namespace NUMINAMATH_CALUDE_hockey_games_played_total_games_played_l450_45079

/-- Calculates the total number of hockey games played in a season -/
theorem hockey_games_played 
  (season_duration : ℕ) 
  (games_per_month : ℕ) 
  (cancelled_games : ℕ) 
  (postponed_games : ℕ) : ℕ :=
  season_duration * games_per_month - cancelled_games

/-- Proves that the total number of hockey games played is 172 -/
theorem total_games_played : 
  hockey_games_played 14 13 10 5 = 172 := by
  sorry

end NUMINAMATH_CALUDE_hockey_games_played_total_games_played_l450_45079


namespace NUMINAMATH_CALUDE_lattice_points_on_quadratic_l450_45010

def is_lattice_point (x y : ℤ) : Prop :=
  y = (x^2 / 10) - (x / 10) + 9 / 5 ∧ y ≤ abs x

theorem lattice_points_on_quadratic :
  ∀ x y : ℤ, is_lattice_point x y ↔ 
    (x = 2 ∧ y = 2) ∨ 
    (x = 4 ∧ y = 3) ∨ 
    (x = 7 ∧ y = 6) ∨ 
    (x = 9 ∧ y = 9) ∨ 
    (x = -6 ∧ y = 6) ∨ 
    (x = -3 ∧ y = 3) :=
by sorry


end NUMINAMATH_CALUDE_lattice_points_on_quadratic_l450_45010


namespace NUMINAMATH_CALUDE_largest_number_value_l450_45077

theorem largest_number_value (a b c : ℝ) (h1 : a < b) (h2 : b < c)
  (h3 : a + b + c = 100) (h4 : c = b + 10) (h5 : b = a + 5) : c = 125/3 := by
  sorry

end NUMINAMATH_CALUDE_largest_number_value_l450_45077


namespace NUMINAMATH_CALUDE_relationship_abc_l450_45094

theorem relationship_abc : 
  let a : ℝ := 1 + Real.sqrt 7
  let b : ℝ := Real.sqrt 3 + Real.sqrt 5
  let c : ℝ := 4
  c > b ∧ b > a := by sorry

end NUMINAMATH_CALUDE_relationship_abc_l450_45094


namespace NUMINAMATH_CALUDE_line_vector_proof_l450_45099

def line_vector (t : ℝ) : ℝ × ℝ := sorry

theorem line_vector_proof :
  (line_vector 1 = (2, 5) ∧ line_vector 4 = (5, -7)) →
  line_vector (-3) = (-2, 21) := by sorry

end NUMINAMATH_CALUDE_line_vector_proof_l450_45099


namespace NUMINAMATH_CALUDE_distance_sum_squares_l450_45059

theorem distance_sum_squares (z : ℂ) (h : Complex.abs (z - (3 - 3*I)) = 3) :
  let z' := 1 + I
  let z'' := 5 - 5*I  -- reflection of z' about 3 - 3i
  (Complex.abs (z - z'))^2 + (Complex.abs (z - z''))^2 = 101 := by
  sorry

end NUMINAMATH_CALUDE_distance_sum_squares_l450_45059


namespace NUMINAMATH_CALUDE_point_A_x_range_l450_45064

-- Define the circle M
def circle_M (x y : ℝ) : Prop := (x - 1)^2 + (y - 1)^2 = 4

-- Define the line l
def line_l (x y : ℝ) : Prop := x + y - 6 = 0

-- Define a point on the circle
def point_on_circle (x y : ℝ) : Prop := circle_M x y

-- Define a point on the line
def point_on_line (x y : ℝ) : Prop := line_l x y

-- Define the angle between three points
def angle (A B C : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem point_A_x_range :
  ∀ (A B C : ℝ × ℝ),
    point_on_line A.1 A.2 →
    point_on_circle B.1 B.2 →
    point_on_circle C.1 C.2 →
    angle A B C = 60 →
    1 ≤ A.1 ∧ A.1 ≤ 5 :=
by sorry

end NUMINAMATH_CALUDE_point_A_x_range_l450_45064


namespace NUMINAMATH_CALUDE_linear_function_properties_l450_45086

-- Define the linear function
def f (x : ℝ) : ℝ := x + 2

-- Theorem stating the properties of the function
theorem linear_function_properties :
  (f 1 = 3) ∧ 
  (f (-2) = 0) ∧ 
  (∃ x > 2, f x ≥ 4) ∧
  (∀ x y, f x = y → (x > 0 → y > 0)) :=
by sorry

#check linear_function_properties

end NUMINAMATH_CALUDE_linear_function_properties_l450_45086


namespace NUMINAMATH_CALUDE_hall_length_l450_45081

theorem hall_length (hall_breadth : ℝ) (stone_length stone_width : ℝ) (num_stones : ℕ) :
  hall_breadth = 15 →
  stone_length = 0.3 →
  stone_width = 0.5 →
  num_stones = 3600 →
  (hall_breadth * (num_stones * stone_length * stone_width / hall_breadth)) = 36 := by
  sorry

end NUMINAMATH_CALUDE_hall_length_l450_45081


namespace NUMINAMATH_CALUDE_original_number_is_point_three_l450_45045

theorem original_number_is_point_three : 
  ∃ x : ℝ, (10 * x = x + 2.7) ∧ (x = 0.3) := by
  sorry

end NUMINAMATH_CALUDE_original_number_is_point_three_l450_45045


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_l450_45097

theorem min_value_reciprocal_sum (m n : ℝ) (h1 : m + n = 1) (h2 : m * n > 0) :
  1/m + 1/n ≥ 4 := by
sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_l450_45097


namespace NUMINAMATH_CALUDE_sales_volume_equation_l450_45030

def daily_sales_volume (x : ℝ) : ℝ := -x + 38

theorem sales_volume_equation :
  (∀ x y : ℝ, y = daily_sales_volume x → (x = 13 → y = 25) ∧ (x = 18 → y = 20)) ∧
  (daily_sales_volume 13 = 25) ∧
  (daily_sales_volume 18 = 20) :=
by sorry

end NUMINAMATH_CALUDE_sales_volume_equation_l450_45030


namespace NUMINAMATH_CALUDE_arccos_gt_arctan_iff_l450_45072

theorem arccos_gt_arctan_iff (x : ℝ) : Real.arccos x > Real.arctan x ↔ x ∈ Set.Ici (-1) ∩ Set.Iio 1 := by
  sorry

end NUMINAMATH_CALUDE_arccos_gt_arctan_iff_l450_45072


namespace NUMINAMATH_CALUDE_inequality_solution_l450_45050

open Set

theorem inequality_solution (x : ℝ) : 
  3 * x - 2 < (x + 2)^2 ∧ (x + 2)^2 < 9 * x - 8 ↔ x ∈ Ioo 3 4 :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_l450_45050


namespace NUMINAMATH_CALUDE_curve_is_hyperbola_l450_45028

/-- The curve defined by r = 1 / (1 - sin θ) is a hyperbola -/
theorem curve_is_hyperbola (θ : ℝ) (r : ℝ) :
  r = 1 / (1 - Real.sin θ) → ∃ (a b c d e f : ℝ), 
    a ≠ 0 ∧ c ≠ 0 ∧ a * c < 0 ∧
    ∀ (x y : ℝ), a * x^2 + b * x * y + c * y^2 + d * x + e * y + f = 0 := by
  sorry

end NUMINAMATH_CALUDE_curve_is_hyperbola_l450_45028


namespace NUMINAMATH_CALUDE_exam_results_l450_45000

theorem exam_results (failed_hindi : ℝ) (failed_english : ℝ) (failed_both : ℝ)
  (h1 : failed_hindi = 20)
  (h2 : failed_english = 70)
  (h3 : failed_both = 10) :
  100 - (failed_hindi + failed_english - failed_both) = 20 := by
  sorry

end NUMINAMATH_CALUDE_exam_results_l450_45000


namespace NUMINAMATH_CALUDE_sin_210_degrees_l450_45090

theorem sin_210_degrees : Real.sin (210 * π / 180) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_sin_210_degrees_l450_45090


namespace NUMINAMATH_CALUDE_simplify_expressions_l450_45087

theorem simplify_expressions (x y a b : ℝ) :
  ((-3 * x + y) + (4 * x - 3 * y) = x - 2 * y) ∧
  (2 * a - (3 * b - 5 * a - (2 * a - 7 * b)) = 9 * a - 10 * b) := by
  sorry

end NUMINAMATH_CALUDE_simplify_expressions_l450_45087


namespace NUMINAMATH_CALUDE_ripe_apples_theorem_l450_45053

-- Define the universe of discourse
variable (Basket : Type)
-- Define the property of being ripe
variable (isRipe : Basket → Prop)

-- Define the statement "All apples in this basket are ripe" is false
axiom not_all_ripe : ¬(∀ (apple : Basket), isRipe apple)

-- Theorem to prove
theorem ripe_apples_theorem :
  (∃ (apple : Basket), ¬(isRipe apple)) ∧
  (¬(∀ (apple : Basket), isRipe apple)) := by
  sorry

end NUMINAMATH_CALUDE_ripe_apples_theorem_l450_45053


namespace NUMINAMATH_CALUDE_library_repacking_l450_45025

theorem library_repacking (total_books : Nat) (initial_boxes : Nat) (new_box_size : Nat) 
  (h1 : total_books = 1870)
  (h2 : initial_boxes = 55)
  (h3 : new_box_size = 36) :
  total_books % new_box_size = 34 := by
  sorry

end NUMINAMATH_CALUDE_library_repacking_l450_45025


namespace NUMINAMATH_CALUDE_initial_water_percentage_l450_45073

/-- Given a container with capacity 100 liters, prove that the initial percentage
    of water is 30% if adding 45 liters makes it 3/4 full. -/
theorem initial_water_percentage
  (capacity : ℝ)
  (added_water : ℝ)
  (final_fraction : ℝ)
  (h1 : capacity = 100)
  (h2 : added_water = 45)
  (h3 : final_fraction = 3/4)
  (h4 : (initial_percentage / 100) * capacity + added_water = final_fraction * capacity) :
  initial_percentage = 30 :=
sorry

#check initial_water_percentage

end NUMINAMATH_CALUDE_initial_water_percentage_l450_45073


namespace NUMINAMATH_CALUDE_max_value_of_a_l450_45034

theorem max_value_of_a (a b c : ℝ) (sum_zero : a + b + c = 0) (sum_squares_six : a^2 + b^2 + c^2 = 6) :
  ∀ x : ℝ, (∃ y z : ℝ, x + y + z = 0 ∧ x^2 + y^2 + z^2 = 6) → x ≤ 2 :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_a_l450_45034


namespace NUMINAMATH_CALUDE_equilateral_triangle_cd_product_l450_45040

/-- Given an equilateral triangle with vertices at (0,0), (c,15), and (d,47),
    the product cd equals 1216√3/9 -/
theorem equilateral_triangle_cd_product (c d : ℝ) : 
  (∀ (z : ℂ), z ^ 3 = 1 ∧ z ≠ 1 → (c + 15 * I) * z = d + 47 * I) →
  c * d = 1216 * Real.sqrt 3 / 9 := by
  sorry

end NUMINAMATH_CALUDE_equilateral_triangle_cd_product_l450_45040


namespace NUMINAMATH_CALUDE_ninety_percent_of_600_equals_fifty_percent_of_x_l450_45060

theorem ninety_percent_of_600_equals_fifty_percent_of_x (x : ℝ) :
  (90 / 100) * 600 = (50 / 100) * x → x = 1080 := by
sorry

end NUMINAMATH_CALUDE_ninety_percent_of_600_equals_fifty_percent_of_x_l450_45060


namespace NUMINAMATH_CALUDE_distance_between_cities_l450_45070

/-- Represents the travel scenario of two cars between two cities -/
structure TravelScenario where
  v : ℝ  -- Speed of car A in km/min
  x : ℝ  -- Total travel time for car A in minutes
  d : ℝ  -- Distance between the two cities in km

/-- Conditions of the travel scenario -/
def travel_conditions (s : TravelScenario) : Prop :=
  -- Both cars travel the same distance in first 5 minutes
  -- Car B's speed reduces to 2/5 of original after 5 minutes
  -- Car B arrives 15 minutes after car A
  (5 * s.v - 25) / 2 = s.x - 5 + 15 ∧
  -- If failure occurred 4 km farther, B would arrive 10 minutes after A
  25 - 10 / s.v = 20 - 4 / s.v ∧
  -- Total distance is speed multiplied by time
  s.d = s.v * s.x

/-- The main theorem stating the distance between the cities -/
theorem distance_between_cities :
  ∀ s : TravelScenario, travel_conditions s → s.d = 18 :=
by sorry

end NUMINAMATH_CALUDE_distance_between_cities_l450_45070


namespace NUMINAMATH_CALUDE_a1_lt_a3_neither_sufficient_nor_necessary_for_a2_lt_a4_l450_45078

/-- A geometric sequence with first term a and common ratio q -/
def geometric_sequence (a q : ℝ) : ℕ → ℝ := fun n => a * q^(n-1)

theorem a1_lt_a3_neither_sufficient_nor_necessary_for_a2_lt_a4 :
  ∃ (a q : ℝ), 
    let seq := geometric_sequence a q
    (seq 1 < seq 3 ∧ seq 2 ≥ seq 4) ∧
    (seq 2 < seq 4 ∧ seq 1 ≥ seq 3) :=
sorry

end NUMINAMATH_CALUDE_a1_lt_a3_neither_sufficient_nor_necessary_for_a2_lt_a4_l450_45078


namespace NUMINAMATH_CALUDE_raw_materials_cost_l450_45049

/-- The total amount Kanul had --/
def total : ℝ := 5714.29

/-- The amount spent on machinery --/
def machinery : ℝ := 1000

/-- The percentage of total amount kept as cash --/
def cash_percentage : ℝ := 0.30

/-- The amount spent on raw materials --/
def raw_materials : ℝ := total - machinery - (cash_percentage * total)

/-- Theorem stating that the amount spent on raw materials is approximately $3000.00 --/
theorem raw_materials_cost : ∃ (ε : ℝ), ε > 0 ∧ ε < 0.01 ∧ |raw_materials - 3000| < ε := by
  sorry

end NUMINAMATH_CALUDE_raw_materials_cost_l450_45049


namespace NUMINAMATH_CALUDE_cos_2017_pi_thirds_l450_45043

theorem cos_2017_pi_thirds : Real.cos (2017 * Real.pi / 3) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_2017_pi_thirds_l450_45043


namespace NUMINAMATH_CALUDE_range_of_a_l450_45003

/-- The function f(x) = a^x - x - a has two zeros -/
def has_two_zeros (f : ℝ → ℝ) : Prop :=
  ∃ x₁ x₂, x₁ ≠ x₂ ∧ f x₁ = 0 ∧ f x₂ = 0

/-- The main theorem -/
theorem range_of_a (a : ℝ) (h₁ : a > 0) (h₂ : a ≠ 1) :
  has_two_zeros (fun x => a^x - x - a) → a > 1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l450_45003


namespace NUMINAMATH_CALUDE_fourth_term_of_geometric_sequence_l450_45007

def geometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = a n * r

theorem fourth_term_of_geometric_sequence (a : ℕ → ℝ) :
  geometricSequence a → a 1 = 2 → a 2 = 4 → a 4 = 16 := by
  sorry

end NUMINAMATH_CALUDE_fourth_term_of_geometric_sequence_l450_45007


namespace NUMINAMATH_CALUDE_johns_sleep_theorem_l450_45031

theorem johns_sleep_theorem :
  let days_in_week : ℕ := 7
  let short_sleep_days : ℕ := 2
  let short_sleep_hours : ℝ := 3
  let recommended_sleep : ℝ := 8
  let sleep_percentage : ℝ := 0.6

  let normal_sleep_days : ℕ := days_in_week - short_sleep_days
  let normal_sleep_hours : ℝ := sleep_percentage * recommended_sleep
  
  let total_sleep : ℝ := 
    (short_sleep_days : ℝ) * short_sleep_hours + 
    (normal_sleep_days : ℝ) * normal_sleep_hours

  total_sleep = 30 := by sorry

end NUMINAMATH_CALUDE_johns_sleep_theorem_l450_45031


namespace NUMINAMATH_CALUDE_ben_win_probability_l450_45008

theorem ben_win_probability (p_lose : ℚ) (h1 : p_lose = 3/7) (h2 : p_lose + p_win = 1) : p_win = 4/7 := by
  sorry

end NUMINAMATH_CALUDE_ben_win_probability_l450_45008


namespace NUMINAMATH_CALUDE_difference_of_squares_factorization_l450_45013

theorem difference_of_squares_factorization (a b p q : ℝ) : 
  (∃ x y, -a^2 + 9 = (x + y) * (x - y)) ∧ 
  (¬∃ x y, -a^2 - b^2 = (x + y) * (x - y)) ∧ 
  (¬∃ x y, p^2 - (-q^2) = (x + y) * (x - y)) ∧ 
  (¬∃ x y, a^2 - b^3 = (x + y) * (x - y)) :=
by sorry

end NUMINAMATH_CALUDE_difference_of_squares_factorization_l450_45013


namespace NUMINAMATH_CALUDE_age_problem_l450_45021

theorem age_problem (parent_age son_age : ℕ) : 
  parent_age = 3 * son_age ∧ 
  parent_age + 5 = (5/2) * (son_age + 5) →
  parent_age = 45 ∧ son_age = 15 := by
sorry

end NUMINAMATH_CALUDE_age_problem_l450_45021


namespace NUMINAMATH_CALUDE_chess_game_draw_probability_l450_45041

theorem chess_game_draw_probability 
  (p_a_wins : ℝ) 
  (p_a_not_lose : ℝ) 
  (h1 : p_a_wins = 0.4) 
  (h2 : p_a_not_lose = 0.9) : 
  p_a_not_lose - p_a_wins = 0.5 := by
  sorry

end NUMINAMATH_CALUDE_chess_game_draw_probability_l450_45041


namespace NUMINAMATH_CALUDE_quadratic_equation_completing_square_l450_45009

theorem quadratic_equation_completing_square (x : ℝ) : 
  x^2 - 4*x + 3 = 0 → (x - 2)^2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_completing_square_l450_45009


namespace NUMINAMATH_CALUDE_lumberjack_chopped_25_trees_l450_45092

/-- Represents the lumberjack's work -/
structure LumberjackWork where
  logs_per_tree : ℕ
  firewood_per_log : ℕ
  total_firewood : ℕ

/-- Calculates the number of trees chopped based on the lumberjack's work -/
def trees_chopped (work : LumberjackWork) : ℕ :=
  work.total_firewood / (work.logs_per_tree * work.firewood_per_log)

/-- Theorem stating that given the specific conditions, the lumberjack chopped 25 trees -/
theorem lumberjack_chopped_25_trees :
  let work := LumberjackWork.mk 4 5 500
  trees_chopped work = 25 := by
  sorry

#eval trees_chopped (LumberjackWork.mk 4 5 500)

end NUMINAMATH_CALUDE_lumberjack_chopped_25_trees_l450_45092


namespace NUMINAMATH_CALUDE_greatest_six_digit_divisible_l450_45024

def is_six_digit (n : ℕ) : Prop := 100000 ≤ n ∧ n ≤ 999999

theorem greatest_six_digit_divisible (n : ℕ) : 
  is_six_digit n ∧ 
  21 ∣ n ∧ 
  35 ∣ n ∧ 
  66 ∣ n ∧ 
  110 ∣ n ∧ 
  143 ∣ n → 
  n ≤ 990990 :=
sorry

end NUMINAMATH_CALUDE_greatest_six_digit_divisible_l450_45024


namespace NUMINAMATH_CALUDE_prize_probabilities_l450_45089

/-- Represents the number of balls of each color in a box -/
structure BallBox where
  red : Nat
  white : Nat

/-- Calculates the probability of drawing a specific number of red balls from two boxes -/
def probability_draw_red (box_a box_b : BallBox) (red_count : Nat) : Rat :=
  sorry

/-- The first box containing 4 red balls and 6 white balls -/
def box_a : BallBox := { red := 4, white := 6 }

/-- The second box containing 5 red balls and 5 white balls -/
def box_b : BallBox := { red := 5, white := 5 }

theorem prize_probabilities :
  probability_draw_red box_a box_b 4 = 4 / 135 ∧
  probability_draw_red box_a box_b 3 = 26 / 135 ∧
  (1 - probability_draw_red box_a box_b 0) = 75 / 81 :=
sorry

end NUMINAMATH_CALUDE_prize_probabilities_l450_45089


namespace NUMINAMATH_CALUDE_bridge_length_calculation_l450_45023

/-- Calculates the length of a bridge given train parameters -/
theorem bridge_length_calculation (train_length : ℝ) (train_speed_kmh : ℝ) (crossing_time : ℝ) :
  train_length = 120 →
  train_speed_kmh = 45 →
  crossing_time = 30 →
  255 = (train_speed_kmh * 1000 / 3600 * crossing_time) - train_length :=
by sorry

end NUMINAMATH_CALUDE_bridge_length_calculation_l450_45023


namespace NUMINAMATH_CALUDE_special_pyramid_volume_l450_45096

/-- Represents a pyramid with an equilateral triangle base and isosceles right triangle lateral faces -/
structure SpecialPyramid where
  base_side_length : ℝ
  is_equilateral_base : base_side_length > 0
  is_isosceles_right_lateral : True

/-- Calculates the volume of the special pyramid -/
def volume (p : SpecialPyramid) : ℝ :=
  sorry

/-- Theorem stating that the volume of the special pyramid with base side length 2 is √2/3 -/
theorem special_pyramid_volume :
  ∀ (p : SpecialPyramid), p.base_side_length = 2 → volume p = Real.sqrt 2 / 3 :=
  sorry

end NUMINAMATH_CALUDE_special_pyramid_volume_l450_45096


namespace NUMINAMATH_CALUDE_distinct_colorings_l450_45005

/-- The number of disks in the circle -/
def n : ℕ := 8

/-- The number of blue disks -/
def blue : ℕ := 4

/-- The number of red disks -/
def red : ℕ := 3

/-- The number of green disks -/
def green : ℕ := 1

/-- The number of rotational symmetries -/
def rotations : ℕ := 4

/-- The number of reflection symmetries -/
def reflections : ℕ := 4

/-- The total number of symmetries -/
def total_symmetries : ℕ := rotations + reflections + 1

/-- The number of colorings fixed by identity -/
def fixed_by_identity : ℕ := (n.choose blue) * ((n - blue).choose red)

/-- The number of colorings fixed by each reflection -/
def fixed_by_reflection : ℕ := 6

/-- The number of colorings fixed by each rotation (other than identity) -/
def fixed_by_rotation : ℕ := 0

/-- Theorem: The number of distinct colorings is 38 -/
theorem distinct_colorings : 
  (fixed_by_identity + reflections * fixed_by_reflection + (rotations - 1) * fixed_by_rotation) / total_symmetries = 38 := by
  sorry

end NUMINAMATH_CALUDE_distinct_colorings_l450_45005


namespace NUMINAMATH_CALUDE_remaining_calculation_l450_45037

def calculate_remaining (income : ℝ) : ℝ :=
  let after_rent := income * (1 - 0.15)
  let after_education := after_rent * (1 - 0.15)
  let after_misc := after_education * (1 - 0.10)
  let after_medical := after_misc * (1 - 0.15)
  after_medical

theorem remaining_calculation (income : ℝ) :
  income = 10037.77 →
  calculate_remaining income = 5547.999951125 := by
  sorry

end NUMINAMATH_CALUDE_remaining_calculation_l450_45037


namespace NUMINAMATH_CALUDE_problem_statement_l450_45014

theorem problem_statement (a : ℝ) 
  (A : Set ℝ) (hA : A = {0, 2, a^2})
  (B : Set ℝ) (hB : B = {1, a})
  (hUnion : A ∪ B = {0, 1, 2, 4}) : a = 2 := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l450_45014


namespace NUMINAMATH_CALUDE_hockey_league_teams_l450_45026

/-- The number of teams in a hockey league --/
def num_teams : ℕ := 18

/-- The number of times each team faces every other team --/
def games_per_pair : ℕ := 10

/-- The total number of games played in the season --/
def total_games : ℕ := 1530

/-- Theorem: Given the conditions, the number of teams in the league is 18 --/
theorem hockey_league_teams :
  (num_teams * (num_teams - 1) * games_per_pair) / 2 = total_games :=
sorry

end NUMINAMATH_CALUDE_hockey_league_teams_l450_45026


namespace NUMINAMATH_CALUDE_sum_of_squares_roots_l450_45042

theorem sum_of_squares_roots (x : ℝ) :
  Real.sqrt (64 - x^2) - Real.sqrt (36 - x^2) = 4 →
  Real.sqrt (64 - x^2) + Real.sqrt (36 - x^2) = 7 :=
by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_roots_l450_45042


namespace NUMINAMATH_CALUDE_new_person_weight_l450_45061

/-- Given a group of 15 people, proves that if replacing a person weighing 45 kg 
    with a new person increases the average weight by 8 kg, 
    then the new person weighs 165 kg. -/
theorem new_person_weight (initial_count : ℕ) (weight_increase : ℝ) 
  (replaced_weight : ℝ) (new_weight : ℝ) : 
  initial_count = 15 → 
  weight_increase = 8 → 
  replaced_weight = 45 → 
  new_weight = initial_count * weight_increase + replaced_weight → 
  new_weight = 165 := by
  sorry

end NUMINAMATH_CALUDE_new_person_weight_l450_45061


namespace NUMINAMATH_CALUDE_f_properties_l450_45016

noncomputable def f (x : ℝ) : ℝ := x^2 * Real.exp x

theorem f_properties :
  (∀ x ∈ Set.Ioo (-2 : ℝ) 0, ∀ y ∈ Set.Ioo (-2 : ℝ) 0, x < y → f x > f y) ∧
  (∃ x₁ x₂, x₁ ≠ x₂ ∧ f x₁ = x₁ - 2012 ∧ f x₂ = x₂ - 2012 ∧
    ∀ x, x ≠ x₁ ∧ x ≠ x₂ → f x ≠ x - 2012) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l450_45016


namespace NUMINAMATH_CALUDE_worker_efficiency_l450_45063

/-- Given two workers p and q, where p can complete a work in 26 days,
    and p and q together can complete the same work in 16 days,
    prove that p is approximately 1.442% more efficient than q. -/
theorem worker_efficiency (p q : ℝ) (h1 : p > 0) (h2 : q > 0) 
  (h3 : p = 1 / 26) (h4 : p + q = 1 / 16) : 
  ∃ ε > 0, |((p - q) / q) * 100 - 1.442| < ε :=
sorry

end NUMINAMATH_CALUDE_worker_efficiency_l450_45063


namespace NUMINAMATH_CALUDE_hyperbola_asymptotes_separate_from_circle_l450_45066

/-- The hyperbola with equation x^2 - my^2 and eccentricity 3 has asymptotes that are separate from the circle (  )x^2 + y^2 = 7 -/
theorem hyperbola_asymptotes_separate_from_circle 
  (m : ℝ) 
  (hyperbola : ℝ → ℝ → Prop) 
  (circle : ℝ → ℝ → Prop) 
  (eccentricity : ℝ) :
  (∀ x y, hyperbola x y ↔ x^2 - m*y^2 = 1) →
  (∀ x y, circle x y ↔ x^2 + y^2 = 7) →
  eccentricity = 3 →
  ∃ d : ℝ, d > Real.sqrt 7 ∧ 
    (∀ x y, y = 2*Real.sqrt 2*x ∨ y = -2*Real.sqrt 2*x → 
      d ≤ Real.sqrt ((x - 3)^2 + y^2)) :=
by sorry


end NUMINAMATH_CALUDE_hyperbola_asymptotes_separate_from_circle_l450_45066


namespace NUMINAMATH_CALUDE_sum_of_two_elements_equals_power_of_two_l450_45083

def M : Set ℕ := {m : ℕ | ∃ n : ℕ, m = n * (n + 1)}

theorem sum_of_two_elements_equals_power_of_two :
  ∃ n : ℕ, n * (n - 1) ∈ M ∧ n * (n + 1) ∈ M ∧ n * (n - 1) + n * (n + 1) = 2^2021 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_two_elements_equals_power_of_two_l450_45083


namespace NUMINAMATH_CALUDE_square_roots_problem_l450_45051

theorem square_roots_problem (x : ℝ) (n : ℝ) (h1 : n > 0) 
  (h2 : x + 1 = Real.sqrt n) (h3 : x - 5 = Real.sqrt n) : n = 9 := by
  sorry

end NUMINAMATH_CALUDE_square_roots_problem_l450_45051


namespace NUMINAMATH_CALUDE_arm_wrestling_tournament_rounds_l450_45095

/-- Represents the rules and structure of the arm wrestling tournament -/
structure Tournament :=
  (num_athletes : Nat)
  (point_diff_limit : Nat)
  (extra_point_rule : Bool)

/-- Calculates the minimum number of rounds required for a tournament -/
def min_rounds (t : Tournament) : Nat :=
  sorry

/-- The main theorem stating that a tournament with 510 athletes requires at least 9 rounds -/
theorem arm_wrestling_tournament_rounds :
  ∀ (t : Tournament),
    t.num_athletes = 510 ∧
    t.point_diff_limit = 1 ∧
    t.extra_point_rule = true →
    min_rounds t ≥ 9 :=
by sorry

end NUMINAMATH_CALUDE_arm_wrestling_tournament_rounds_l450_45095


namespace NUMINAMATH_CALUDE_election_vote_difference_l450_45035

theorem election_vote_difference (total_votes : ℕ) (candidate_percentage : ℚ) : 
  total_votes = 7600 → 
  candidate_percentage = 35/100 → 
  (total_votes : ℚ) * (1 - candidate_percentage) - (total_votes : ℚ) * candidate_percentage = 2280 := by
sorry

end NUMINAMATH_CALUDE_election_vote_difference_l450_45035


namespace NUMINAMATH_CALUDE_pencil_length_l450_45082

theorem pencil_length : ∀ (L : ℝ),
  (L / 8 : ℝ) +  -- Black part
  ((7 * L / 8) / 2 : ℝ) +  -- White part
  (7 / 2 : ℝ) = L →  -- Blue part
  L = 16 := by
sorry

end NUMINAMATH_CALUDE_pencil_length_l450_45082


namespace NUMINAMATH_CALUDE_candy_distribution_theorem_l450_45091

/-- Represents the candy distribution pattern -/
def candy_distribution (n : ℕ) (k : ℕ) : ℕ :=
  (k * (k + 1) / 2) % n

/-- Predicate to check if all children receive candy -/
def all_children_receive_candy (n : ℕ) : Prop :=
  ∀ a : ℕ, ∃ k : ℕ, candy_distribution n k = a % n

/-- Theorem: All children receive candy iff n is a power of 2 -/
theorem candy_distribution_theorem (n : ℕ) :
  all_children_receive_candy n ↔ ∃ k : ℕ, n = 2^k :=
sorry

end NUMINAMATH_CALUDE_candy_distribution_theorem_l450_45091


namespace NUMINAMATH_CALUDE_alcohol_solution_proof_l450_45011

/-- Proves that adding 1.8 litres of pure alcohol to a 6-litre solution
    that is 35% alcohol results in a 50% alcohol solution -/
theorem alcohol_solution_proof :
  let initial_volume : ℝ := 6
  let initial_percentage : ℝ := 0.35
  let target_percentage : ℝ := 0.5
  let added_alcohol : ℝ := 1.8
  let final_volume := initial_volume + added_alcohol
  let initial_alcohol := initial_volume * initial_percentage
  let final_alcohol := initial_alcohol + added_alcohol
  (final_alcohol / final_volume) = target_percentage := by sorry

end NUMINAMATH_CALUDE_alcohol_solution_proof_l450_45011


namespace NUMINAMATH_CALUDE_triangle_problem_l450_45069

-- Define the triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- Define the theorem
theorem triangle_problem (t : Triangle) 
  (h1 : (t.a + t.c) / (t.a + t.b) = (t.b - t.a) / t.c)
  (h2 : t.b = Real.sqrt 7)
  (h3 : Real.sin t.C = 2 * Real.sin t.A) :
  t.B = 2 * π / 3 ∧ min t.a (min t.b t.c) = 1 := by
  sorry


end NUMINAMATH_CALUDE_triangle_problem_l450_45069


namespace NUMINAMATH_CALUDE_number_problem_l450_45048

theorem number_problem : ∃ x : ℝ, 0.50 * x = 0.30 * 50 + 13 ∧ x = 56 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l450_45048


namespace NUMINAMATH_CALUDE_sqrt_abs_sum_zero_implies_power_l450_45017

theorem sqrt_abs_sum_zero_implies_power (m n : ℝ) :
  Real.sqrt (m - 2) + |n + 3| = 0 → (m + n)^2023 = -1 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_abs_sum_zero_implies_power_l450_45017


namespace NUMINAMATH_CALUDE_min_value_of_expression_l450_45076

/-- Given a function f(x) = x² + 2√a x - b + 1, where a and b are positive real numbers,
    and f(x) has only one zero, the minimum value of 1/a + 2a/(b+1) is 5/2. -/
theorem min_value_of_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  (∃! x, x^2 + 2 * Real.sqrt a * x - b + 1 = 0) → 
  (∀ a' b', a' > 0 → b' > 0 → 
    (∃! x, x^2 + 2 * Real.sqrt a' * x - b' + 1 = 0) → 
    1 / a + 2 * a / (b + 1) ≤ 1 / a' + 2 * a' / (b' + 1)) ∧
  (∃ a₀ b₀, a₀ > 0 ∧ b₀ > 0 ∧ 
    (∃! x, x^2 + 2 * Real.sqrt a₀ * x - b₀ + 1 = 0) ∧
    1 / a₀ + 2 * a₀ / (b₀ + 1) = 5 / 2) :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l450_45076


namespace NUMINAMATH_CALUDE_complex_fraction_equality_l450_45074

theorem complex_fraction_equality : 1 + 1 / (2 + 1 / (2 + 2)) = 13 / 9 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_equality_l450_45074


namespace NUMINAMATH_CALUDE_farmer_cow_count_farmer_has_52_cows_l450_45080

/-- The number of cows a farmer has, given milk production data. -/
theorem farmer_cow_count : ℕ :=
  let milk_per_cow_per_day : ℕ := 5
  let days_per_week : ℕ := 7
  let total_milk_per_week : ℕ := 1820
  let milk_per_cow_per_week : ℕ := milk_per_cow_per_day * days_per_week
  total_milk_per_week / milk_per_cow_per_week

/-- Proof that the farmer has 52 cows. -/
theorem farmer_has_52_cows : farmer_cow_count = 52 := by
  sorry

end NUMINAMATH_CALUDE_farmer_cow_count_farmer_has_52_cows_l450_45080


namespace NUMINAMATH_CALUDE_unique_element_implies_a_value_l450_45047

-- Define the set A
def A (a : ℝ) : Set ℝ := {x | a * x^2 + 2 * x + 1 = 0}

-- State the theorem
theorem unique_element_implies_a_value (a : ℝ) :
  (∃! x, x ∈ A a) → a = 0 ∨ a = 1 := by
  sorry

end NUMINAMATH_CALUDE_unique_element_implies_a_value_l450_45047


namespace NUMINAMATH_CALUDE_geometric_series_sum_l450_45057

theorem geometric_series_sum (a b : ℝ) (h : ∑' n, a / b^n = 7) :
  ∑' n, a / (a + 2*b)^n = (7*(b-1)) / (9*b-8) := by
  sorry

end NUMINAMATH_CALUDE_geometric_series_sum_l450_45057


namespace NUMINAMATH_CALUDE_canoe_upstream_speed_l450_45054

/-- 
Given a canoe that rows downstream at 12 km/hr and a stream with a speed of 4.5 km/hr,
prove that the speed of the canoe when rowing upstream is 3 km/hr.
-/
theorem canoe_upstream_speed : 
  ∀ (downstream_speed stream_speed : ℝ),
  downstream_speed = 12 →
  stream_speed = 4.5 →
  (downstream_speed - 2 * stream_speed) = 3 :=
by sorry

end NUMINAMATH_CALUDE_canoe_upstream_speed_l450_45054


namespace NUMINAMATH_CALUDE_square_land_side_length_l450_45036

theorem square_land_side_length (area : ℝ) (h : area = Real.sqrt 625) :
  ∃ (side : ℝ), side * side = area ∧ side = 25 := by
  sorry

end NUMINAMATH_CALUDE_square_land_side_length_l450_45036


namespace NUMINAMATH_CALUDE_exponential_function_point_l450_45044

theorem exponential_function_point (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  (fun x => a^x) 2 = 9 → a = 3 := by
  sorry

end NUMINAMATH_CALUDE_exponential_function_point_l450_45044


namespace NUMINAMATH_CALUDE_min_product_of_primes_l450_45038

theorem min_product_of_primes (x y z : Nat) : 
  Nat.Prime x ∧ Nat.Prime y ∧ Nat.Prime z ∧
  Odd x ∧ Odd y ∧ Odd z ∧
  (y^5 + 1) % x = 0 ∧
  (z^5 + 1) % y = 0 ∧
  (x^5 + 1) % z = 0 →
  ∀ a b c : Nat, 
    Nat.Prime a ∧ Nat.Prime b ∧ Nat.Prime c ∧
    Odd a ∧ Odd b ∧ Odd c ∧
    (b^5 + 1) % a = 0 ∧
    (c^5 + 1) % b = 0 ∧
    (a^5 + 1) % c = 0 →
    x * y * z ≤ a * b * c ∧
    x * y * z = 2013 := by
sorry

end NUMINAMATH_CALUDE_min_product_of_primes_l450_45038


namespace NUMINAMATH_CALUDE_central_cell_value_l450_45056

/-- A 3x3 table of real numbers -/
structure Table :=
  (a b c d e f g h i : ℝ)

/-- The conditions for the table -/
def satisfies_conditions (t : Table) : Prop :=
  t.a * t.b * t.c = 10 ∧
  t.d * t.e * t.f = 10 ∧
  t.g * t.h * t.i = 10 ∧
  t.a * t.d * t.g = 10 ∧
  t.b * t.e * t.h = 10 ∧
  t.c * t.f * t.i = 10 ∧
  t.a * t.b * t.d * t.e = 3 ∧
  t.b * t.c * t.e * t.f = 3 ∧
  t.d * t.e * t.g * t.h = 3 ∧
  t.e * t.f * t.h * t.i = 3

/-- The theorem statement -/
theorem central_cell_value (t : Table) (h : satisfies_conditions t) : t.e = 0.00081 := by
  sorry

end NUMINAMATH_CALUDE_central_cell_value_l450_45056


namespace NUMINAMATH_CALUDE_tan_alpha_value_l450_45046

theorem tan_alpha_value (α : Real) (h : 3 * Real.sin α + 4 * Real.cos α = 5) : 
  Real.tan α = 3/4 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_value_l450_45046


namespace NUMINAMATH_CALUDE_ratio_a_to_b_l450_45052

theorem ratio_a_to_b (a b : ℚ) (h : 2 * a = 3 * b) : 
  ∃ (k : ℚ), k > 0 ∧ a = (3 * k) ∧ b = (2 * k) := by
  sorry

end NUMINAMATH_CALUDE_ratio_a_to_b_l450_45052


namespace NUMINAMATH_CALUDE_triangle_sum_equals_58_l450_45085

/-- The triangle operation that takes three numbers and returns the sum of their squares -/
def triangle (a b c : ℝ) : ℝ := a^2 + b^2 + c^2

/-- Theorem stating that the sum of triangle(2,3,6) and triangle(1,2,2) equals 58 -/
theorem triangle_sum_equals_58 : triangle 2 3 6 + triangle 1 2 2 = 58 := by
  sorry

end NUMINAMATH_CALUDE_triangle_sum_equals_58_l450_45085


namespace NUMINAMATH_CALUDE_goals_scored_over_two_days_l450_45004

/-- The total number of goals scored by Gina and Tom over two days -/
def total_goals (gina_day1 gina_day2 tom_day1 tom_day2 : ℕ) : ℕ :=
  gina_day1 + gina_day2 + tom_day1 + tom_day2

/-- Theorem stating the total number of goals scored by Gina and Tom over two days -/
theorem goals_scored_over_two_days :
  ∃ (gina_day1 gina_day2 tom_day1 tom_day2 : ℕ),
    gina_day1 = 2 ∧
    tom_day1 = gina_day1 + 3 ∧
    tom_day2 = 6 ∧
    gina_day2 = tom_day2 - 2 ∧
    total_goals gina_day1 gina_day2 tom_day1 tom_day2 = 17 :=
by
  sorry

end NUMINAMATH_CALUDE_goals_scored_over_two_days_l450_45004


namespace NUMINAMATH_CALUDE_neon_signs_blink_together_l450_45067

theorem neon_signs_blink_together (a b c d : ℕ) 
  (ha : a = 7) (hb : b = 11) (hc : c = 13) (hd : d = 17) : 
  Nat.lcm a (Nat.lcm b (Nat.lcm c d)) = 17017 := by
  sorry

end NUMINAMATH_CALUDE_neon_signs_blink_together_l450_45067


namespace NUMINAMATH_CALUDE_sock_selection_theorem_l450_45062

theorem sock_selection_theorem :
  let n : ℕ := 7  -- Total number of socks
  let k : ℕ := 4  -- Number of socks to choose
  Nat.choose n k = 35 := by
  sorry

end NUMINAMATH_CALUDE_sock_selection_theorem_l450_45062


namespace NUMINAMATH_CALUDE_arthur_walk_distance_l450_45027

/-- Represents the distance walked in a single direction -/
structure DirectionalWalk where
  blocks : ℕ
  direction : String

/-- Calculates the total distance walked given a list of directional walks and the length of each block in miles -/
def totalDistance (walks : List DirectionalWalk) (blockLength : ℚ) : ℚ :=
  (walks.map (·.blocks)).sum * blockLength

theorem arthur_walk_distance :
  let eastWalk : DirectionalWalk := ⟨8, "east"⟩
  let northWalk : DirectionalWalk := ⟨10, "north"⟩
  let westWalk : DirectionalWalk := ⟨3, "west"⟩
  let walks : List DirectionalWalk := [eastWalk, northWalk, westWalk]
  let blockLength : ℚ := 1/3
  totalDistance walks blockLength = 7 := by
  sorry

end NUMINAMATH_CALUDE_arthur_walk_distance_l450_45027


namespace NUMINAMATH_CALUDE_drummer_tosses_six_sets_l450_45018

/-- Calculates the number of drum stick sets tossed to the audience after each show -/
def drumSticksTossedPerShow (setsPerShow : ℕ) (totalNights : ℕ) (totalSetsUsed : ℕ) : ℕ :=
  ((totalSetsUsed - setsPerShow * totalNights) / totalNights)

/-- Theorem: Given the conditions, the drummer tosses 6 sets of drum sticks after each show -/
theorem drummer_tosses_six_sets :
  drumSticksTossedPerShow 5 30 330 = 6 := by
  sorry

end NUMINAMATH_CALUDE_drummer_tosses_six_sets_l450_45018


namespace NUMINAMATH_CALUDE_graphing_calculator_theorem_l450_45065

/-- Represents the number of students who brought graphing calculators -/
def graphing_calculator_count : ℕ := 10

/-- Represents the total number of boys in the class -/
def total_boys : ℕ := 20

/-- Represents the total number of girls in the class -/
def total_girls : ℕ := 18

/-- Represents the number of students who brought scientific calculators -/
def scientific_calculator_count : ℕ := 30

/-- Represents the number of girls who brought scientific calculators -/
def girls_with_scientific_calculators : ℕ := 15

theorem graphing_calculator_theorem :
  graphing_calculator_count = 10 ∧
  total_boys + total_girls = scientific_calculator_count + graphing_calculator_count :=
by sorry

end NUMINAMATH_CALUDE_graphing_calculator_theorem_l450_45065


namespace NUMINAMATH_CALUDE_jose_profit_share_l450_45015

/-- Calculates the share of profit for an investor in a partnership --/
def calculate_profit_share (investment : ℕ) (months : ℕ) (total_profit : ℕ) (total_capital_months : ℕ) : ℕ :=
  (investment * months * total_profit) / total_capital_months

theorem jose_profit_share :
  let tom_investment : ℕ := 30000
  let tom_months : ℕ := 12
  let jose_investment : ℕ := 45000
  let jose_months : ℕ := 10
  let total_profit : ℕ := 36000
  let total_capital_months : ℕ := tom_investment * tom_months + jose_investment * jose_months
  
  calculate_profit_share jose_investment jose_months total_profit total_capital_months = 20000 := by
  sorry


end NUMINAMATH_CALUDE_jose_profit_share_l450_45015


namespace NUMINAMATH_CALUDE_room_width_calculation_l450_45019

/-- Given a rectangular room with length 12 feet and width w feet, 
    with a carpet placed leaving a 2-foot wide border all around, 
    if the area of the border is 72 square feet, 
    then the width of the room is 10 feet. -/
theorem room_width_calculation (w : ℝ) : 
  w > 0 →  -- width is positive
  12 * w - 8 * (w - 4) = 72 →  -- area of border is 72 sq ft
  w = 10 := by
sorry

end NUMINAMATH_CALUDE_room_width_calculation_l450_45019


namespace NUMINAMATH_CALUDE_range_of_m_l450_45071

theorem range_of_m (f g : ℝ → ℝ) (m : ℝ) :
  (∀ x, f x = x - 2) →
  (∀ x, g x = x^2 - 2*m*x + 4) →
  (∀ x₁ ∈ Set.Icc 1 2, ∃ x₂ ∈ Set.Icc 4 5, g x₁ = f x₂) →
  m ∈ Set.Icc (5/4) (Real.sqrt 2) :=
by sorry

end NUMINAMATH_CALUDE_range_of_m_l450_45071


namespace NUMINAMATH_CALUDE_train_speed_problem_l450_45098

/-- Proves that given two trains traveling towards each other from cities 100 miles apart,
    with one train traveling at 45 mph, if they meet after 1.33333333333 hours,
    then the speed of the other train must be 30 mph. -/
theorem train_speed_problem (distance : ℝ) (speed_train1 : ℝ) (time : ℝ) (speed_train2 : ℝ) : 
  distance = 100 →
  speed_train1 = 45 →
  time = 1.33333333333 →
  distance = speed_train1 * time + speed_train2 * time →
  speed_train2 = 30 := by
sorry

end NUMINAMATH_CALUDE_train_speed_problem_l450_45098


namespace NUMINAMATH_CALUDE_extreme_values_imply_b_l450_45039

/-- A cubic function with parameters a and b -/
def f (a b x : ℝ) : ℝ := 2 * x^3 + 3 * a * x^2 + 3 * b * x

/-- The derivative of f with respect to x -/
def f' (a b x : ℝ) : ℝ := 6 * x^2 + 6 * a * x + 3 * b

theorem extreme_values_imply_b (a b : ℝ) :
  (f' a b 1 = 0) → (f' a b 2 = 0) → b = 4 := by
  sorry

end NUMINAMATH_CALUDE_extreme_values_imply_b_l450_45039


namespace NUMINAMATH_CALUDE_road_trip_duration_l450_45029

theorem road_trip_duration (family_size : ℕ) (water_per_person_per_hour : ℚ) 
  (total_water_bottles : ℕ) (h : ℕ) : 
  family_size = 4 → 
  water_per_person_per_hour = 1/2 → 
  total_water_bottles = 32 → 
  (2 * h : ℚ) * (family_size : ℚ) * water_per_person_per_hour = total_water_bottles → 
  h = 8 := by
sorry

end NUMINAMATH_CALUDE_road_trip_duration_l450_45029


namespace NUMINAMATH_CALUDE_eighteenth_power_digits_l450_45058

/-- The function that returns the list of digits in the decimal representation of a natural number -/
def digits (n : ℕ) : List ℕ :=
  sorry

/-- Theorem stating that 18 is the positive integer whose sixth power's decimal representation
    consists of the digits 0, 1, 2, 2, 2, 3, 4, 4 -/
theorem eighteenth_power_digits :
  ∃! (n : ℕ), n > 0 ∧ digits (n^6) = [3, 4, 0, 1, 2, 2, 2, 4] ∧ n = 18 :=
sorry

end NUMINAMATH_CALUDE_eighteenth_power_digits_l450_45058


namespace NUMINAMATH_CALUDE_tan_sum_pi_fourth_l450_45032

theorem tan_sum_pi_fourth (θ : Real) (h : Real.tan θ = 1/3) : 
  Real.tan (θ + π/4) = 2 := by
  sorry

end NUMINAMATH_CALUDE_tan_sum_pi_fourth_l450_45032


namespace NUMINAMATH_CALUDE_ralph_sock_purchase_l450_45093

/-- Represents the number of pairs of socks at each price point -/
structure SockPurchase where
  two_dollar : ℕ
  three_dollar : ℕ
  five_dollar : ℕ

/-- Checks if the purchase satisfies the given conditions -/
def is_valid_purchase (p : SockPurchase) : Prop :=
  p.two_dollar + p.three_dollar + p.five_dollar = 15 ∧
  2 * p.two_dollar + 3 * p.three_dollar + 5 * p.five_dollar = 36 ∧
  p.two_dollar ≥ 1 ∧ p.three_dollar ≥ 1 ∧ p.five_dollar ≥ 1

/-- The theorem to be proved -/
theorem ralph_sock_purchase :
  ∃ (p : SockPurchase), is_valid_purchase p ∧ p.two_dollar = 11 :=
sorry

end NUMINAMATH_CALUDE_ralph_sock_purchase_l450_45093


namespace NUMINAMATH_CALUDE_stating_currency_exchange_problem_l450_45022

/-- Represents the exchange rate from U.S. dollars to Canadian dollars -/
def exchange_rate : ℚ := 12 / 8

/-- Represents the amount spent in Canadian dollars -/
def amount_spent : ℕ := 72

/-- 
Theorem stating that if a person exchanges m U.S. dollars to Canadian dollars
at the given exchange rate, spends the specified amount, and is left with m
Canadian dollars, then m must equal 144.
-/
theorem currency_exchange_problem (m : ℕ) :
  (m : ℚ) * exchange_rate - amount_spent = m →
  m = 144 :=
by sorry

end NUMINAMATH_CALUDE_stating_currency_exchange_problem_l450_45022


namespace NUMINAMATH_CALUDE_horner_v2_equals_22_l450_45055

/-- Horner's method for polynomial evaluation -/
def horner (coeffs : List ℝ) (x : ℝ) : ℝ :=
  coeffs.foldl (fun acc a => x * acc + a) 0

/-- The polynomial f(x) = x^6 + 6x^4 + 9x^2 + 208 -/
def f (x : ℝ) : ℝ := x^6 + 6*x^4 + 9*x^2 + 208

/-- The coefficients of f(x) in descending order of degree -/
def f_coeffs : List ℝ := [1, 0, 6, 0, 9, 0, 208]

/-- Theorem: v₂ = 22 when evaluating f(x) at x = -4 using Horner's method -/
theorem horner_v2_equals_22 :
  let x := -4
  let v₀ := 208
  let v₁ := x * v₀ + 0
  let v₂ := x * v₁ + 9
  v₂ = 22 := by sorry

end NUMINAMATH_CALUDE_horner_v2_equals_22_l450_45055


namespace NUMINAMATH_CALUDE_savanna_safari_snake_ratio_l450_45084

-- Define the number of animals in Safari National Park
def safari_lions : ℕ := 100
def safari_snakes : ℕ := safari_lions / 2
def safari_giraffes : ℕ := safari_snakes - 10

-- Define the number of animals in Savanna National Park
def savanna_lions : ℕ := 2 * safari_lions
def savanna_giraffes : ℕ := safari_giraffes + 20

-- Define the total number of animals in Savanna National Park
def savanna_total : ℕ := 410

-- Define the ratio of snakes in Savanna to Safari
def snake_ratio : ℚ := 3

theorem savanna_safari_snake_ratio :
  snake_ratio = (savanna_total - savanna_lions - savanna_giraffes) / safari_snakes := by
  sorry

end NUMINAMATH_CALUDE_savanna_safari_snake_ratio_l450_45084


namespace NUMINAMATH_CALUDE_sum_of_integers_l450_45088

theorem sum_of_integers (a b : ℕ+) (h1 : a - b = 6) (h2 : a * b = 272) : a + b = 34 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_integers_l450_45088
