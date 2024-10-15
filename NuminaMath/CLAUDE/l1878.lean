import Mathlib

namespace NUMINAMATH_CALUDE_cube_surface_area_increase_l1878_187846

theorem cube_surface_area_increase :
  ∀ (s : ℝ), s > 0 →
  let original_surface_area := 6 * s^2
  let new_edge_length := 1.3 * s
  let new_surface_area := 6 * new_edge_length^2
  (new_surface_area - original_surface_area) / original_surface_area = 0.69 :=
by sorry

end NUMINAMATH_CALUDE_cube_surface_area_increase_l1878_187846


namespace NUMINAMATH_CALUDE_butterflies_let_go_l1878_187804

theorem butterflies_let_go (original : ℕ) (left : ℕ) (h1 : original = 93) (h2 : left = 82) :
  original - left = 11 := by
  sorry

end NUMINAMATH_CALUDE_butterflies_let_go_l1878_187804


namespace NUMINAMATH_CALUDE_division_problem_l1878_187876

theorem division_problem (n : ℤ) : 
  (n / 6 = 124 ∧ n % 6 = 4) → ((n + 24) / 8 : ℚ) = 96.5 := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l1878_187876


namespace NUMINAMATH_CALUDE_larger_number_proof_l1878_187875

theorem larger_number_proof (a b : ℕ+) : 
  (Nat.gcd a b = 40) →
  (Nat.lcm a b = 6600) →
  ((a = 40 * 11 ∧ b = 40 * 15) ∨ (a = 40 * 15 ∧ b = 40 * 11)) →
  max a b = 600 := by
sorry

end NUMINAMATH_CALUDE_larger_number_proof_l1878_187875


namespace NUMINAMATH_CALUDE_sum_of_abc_l1878_187871

theorem sum_of_abc (a b c : ℝ) 
  (eq1 : a^2 + 6*b = -17)
  (eq2 : b^2 + 8*c = -23)
  (eq3 : c^2 + 2*a = 14) :
  a + b + c = -8 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_abc_l1878_187871


namespace NUMINAMATH_CALUDE_smallest_degree_for_horizontal_asymptote_l1878_187853

/-- The numerator of our rational function -/
def f (x : ℝ) : ℝ := 5 * x^7 + 2 * x^4 - 7

/-- A proposition stating that a rational function has a horizontal asymptote -/
def has_horizontal_asymptote (num den : ℝ → ℝ) : Prop :=
  ∃ (L : ℝ), ∀ ε > 0, ∃ M, ∀ x, |x| > M → |num x / den x - L| < ε

/-- The main theorem: the smallest possible degree of p(x) is 7 -/
theorem smallest_degree_for_horizontal_asymptote :
  ∀ p : ℝ → ℝ, has_horizontal_asymptote f p → (∃ n : ℕ, ∀ x, p x = x^n) → 
  (∀ m : ℕ, (∃ x, p x = x^m) → m ≥ 7) :=
sorry

end NUMINAMATH_CALUDE_smallest_degree_for_horizontal_asymptote_l1878_187853


namespace NUMINAMATH_CALUDE_no_five_coprime_two_digit_composites_l1878_187834

theorem no_five_coprime_two_digit_composites : 
  ¬ ∃ (a b c d e : ℕ), 
    (10 ≤ a ∧ a < 100 ∧ ¬ Nat.Prime a) ∧
    (10 ≤ b ∧ b < 100 ∧ ¬ Nat.Prime b) ∧
    (10 ≤ c ∧ c < 100 ∧ ¬ Nat.Prime c) ∧
    (10 ≤ d ∧ d < 100 ∧ ¬ Nat.Prime d) ∧
    (10 ≤ e ∧ e < 100 ∧ ¬ Nat.Prime e) ∧
    (Nat.Coprime a b ∧ Nat.Coprime a c ∧ Nat.Coprime a d ∧ Nat.Coprime a e ∧
     Nat.Coprime b c ∧ Nat.Coprime b d ∧ Nat.Coprime b e ∧
     Nat.Coprime c d ∧ Nat.Coprime c e ∧
     Nat.Coprime d e) :=
by
  sorry


end NUMINAMATH_CALUDE_no_five_coprime_two_digit_composites_l1878_187834


namespace NUMINAMATH_CALUDE_arithmetic_sequence_inequality_l1878_187817

/-- An arithmetic sequence with positive terms and non-zero common difference -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  d : ℝ
  h1 : ∀ n, a n > 0
  h2 : d ≠ 0
  h3 : ∀ n, a (n + 1) = a n + d

theorem arithmetic_sequence_inequality (seq : ArithmeticSequence) : 
  seq.a 1 * seq.a 8 < seq.a 4 * seq.a 5 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_inequality_l1878_187817


namespace NUMINAMATH_CALUDE_not_in_fourth_quadrant_l1878_187861

-- Define a point in 2D space
structure Point2D where
  x : ℝ
  y : ℝ

-- Define the fourth quadrant
def fourth_quadrant (p : Point2D) : Prop :=
  p.x > 0 ∧ p.y < 0

-- Theorem statement
theorem not_in_fourth_quadrant (m : ℝ) :
  ¬(fourth_quadrant ⟨m - 2, m + 1⟩) := by
  sorry

end NUMINAMATH_CALUDE_not_in_fourth_quadrant_l1878_187861


namespace NUMINAMATH_CALUDE_borrowing_period_is_one_year_l1878_187839

-- Define the problem parameters
def initial_amount : ℕ := 5000
def borrowing_rate : ℚ := 4 / 100
def lending_rate : ℚ := 6 / 100
def gain_per_year : ℕ := 100

-- Define the function to calculate interest
def calculate_interest (amount : ℕ) (rate : ℚ) : ℚ :=
  (amount : ℚ) * rate

-- Define the function to calculate the gain
def calculate_gain (amount : ℕ) (borrow_rate lending_rate : ℚ) : ℚ :=
  calculate_interest amount lending_rate - calculate_interest amount borrow_rate

-- Theorem statement
theorem borrowing_period_is_one_year :
  calculate_gain initial_amount borrowing_rate lending_rate = gain_per_year := by
  sorry

end NUMINAMATH_CALUDE_borrowing_period_is_one_year_l1878_187839


namespace NUMINAMATH_CALUDE_factory_works_four_days_l1878_187868

/-- Represents a toy factory with weekly production and daily production rates. -/
structure ToyFactory where
  weekly_production : ℕ
  daily_production : ℕ

/-- Calculates the number of working days per week for a given toy factory. -/
def working_days (factory : ToyFactory) : ℕ :=
  factory.weekly_production / factory.daily_production

/-- Theorem: The toy factory works 4 days per week. -/
theorem factory_works_four_days :
  let factory : ToyFactory := { weekly_production := 6000, daily_production := 1500 }
  working_days factory = 4 := by
  sorry

#eval working_days { weekly_production := 6000, daily_production := 1500 }

end NUMINAMATH_CALUDE_factory_works_four_days_l1878_187868


namespace NUMINAMATH_CALUDE_catfish_dinner_price_l1878_187821

/-- The price of a catfish dinner at River Joe's Seafood Diner -/
def catfish_price : ℚ := 6

/-- The price of a popcorn shrimp dinner at River Joe's Seafood Diner -/
def popcorn_shrimp_price : ℚ := 7/2

/-- The total number of orders filled -/
def total_orders : ℕ := 26

/-- The number of popcorn shrimp orders sold -/
def popcorn_shrimp_orders : ℕ := 9

/-- The total revenue collected -/
def total_revenue : ℚ := 267/2

theorem catfish_dinner_price :
  catfish_price * (total_orders - popcorn_shrimp_orders) + 
  popcorn_shrimp_price * popcorn_shrimp_orders = total_revenue :=
by sorry

end NUMINAMATH_CALUDE_catfish_dinner_price_l1878_187821


namespace NUMINAMATH_CALUDE_expand_product_l1878_187880

theorem expand_product (x : ℝ) (h : x ≠ 0) :
  2/5 * (5/x + 10*x^2) = 2/x + 4*x^2 := by
  sorry

end NUMINAMATH_CALUDE_expand_product_l1878_187880


namespace NUMINAMATH_CALUDE_at_least_one_not_less_than_two_l1878_187884

theorem at_least_one_not_less_than_two (a b : ℕ) (h : a + b ≥ 3) : max a b ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_at_least_one_not_less_than_two_l1878_187884


namespace NUMINAMATH_CALUDE_no_consistent_values_l1878_187893

theorem no_consistent_values : ¬∃ (A B C D : ℤ), 
  B = 59 ∧ 
  C = 27 ∧ 
  D = 31 ∧ 
  (4701 % A = 0) ∧ 
  A = B * C + D :=
sorry

end NUMINAMATH_CALUDE_no_consistent_values_l1878_187893


namespace NUMINAMATH_CALUDE_triangular_number_difference_l1878_187872

def triangular_number (n : ℕ) : ℕ := n * (n + 1) / 2

theorem triangular_number_difference : 
  triangular_number 30 - triangular_number 28 = 59 := by
sorry

end NUMINAMATH_CALUDE_triangular_number_difference_l1878_187872


namespace NUMINAMATH_CALUDE_no_rational_solutions_l1878_187892

theorem no_rational_solutions (m : ℕ+) : ¬∃ (x : ℚ), m * x^2 + 40 * x + m = 0 := by
  sorry

end NUMINAMATH_CALUDE_no_rational_solutions_l1878_187892


namespace NUMINAMATH_CALUDE_total_distance_is_9km_l1878_187801

/-- Represents the travel plans from the city bus station to Tianbo Mountain -/
inductive TravelPlan
| BusOnly
| BikeOnly
| BikeThenBus
| BusThenBike

/-- Represents the journey from the city bus station to Tianbo Mountain -/
structure Journey where
  distance_to_hehua : ℝ
  distance_from_hehua : ℝ
  bus_speed : ℝ
  bike_speed : ℝ
  bus_stop_time : ℝ

/-- The actual journey based on the problem description -/
def actual_journey : Journey where
  distance_to_hehua := 6
  distance_from_hehua := 3
  bus_speed := 24
  bike_speed := 16
  bus_stop_time := 0.5

/-- Theorem stating that the total distance is 9 km -/
theorem total_distance_is_9km (j : Journey) :
  j.distance_to_hehua + j.distance_from_hehua = 9 ∧
  j.distance_to_hehua = 6 ∧
  j.distance_from_hehua = 3 ∧
  j.bus_speed = 24 ∧
  j.bike_speed = 16 ∧
  j.bus_stop_time = 0.5 ∧
  (j.distance_to_hehua + j.distance_from_hehua) / j.bus_speed + j.bus_stop_time =
    (j.distance_to_hehua + j.distance_from_hehua + 1) / j.bike_speed ∧
  j.distance_to_hehua / j.bus_speed = 4 / j.bike_speed ∧
  (j.distance_to_hehua / j.bus_speed + j.bus_stop_time + j.distance_from_hehua / j.bike_speed) =
    ((j.distance_to_hehua + j.distance_from_hehua) / j.bus_speed + j.bus_stop_time - 0.25) :=
by sorry

#check total_distance_is_9km actual_journey

end NUMINAMATH_CALUDE_total_distance_is_9km_l1878_187801


namespace NUMINAMATH_CALUDE_non_monotonic_derivative_range_l1878_187825

open Real

theorem non_monotonic_derivative_range (f : ℝ → ℝ) (k : ℝ) :
  (∀ x, deriv f x = exp x + k^2 / exp x - 1 / k) →
  (¬ Monotone f) →
  0 < k ∧ k < sqrt 2 / 2 :=
sorry

end NUMINAMATH_CALUDE_non_monotonic_derivative_range_l1878_187825


namespace NUMINAMATH_CALUDE_fraction_spent_l1878_187866

def borrowed_brother : ℕ := 20
def borrowed_father : ℕ := 40
def borrowed_mother : ℕ := 30
def gift_grandmother : ℕ := 70
def savings : ℕ := 100
def remaining : ℕ := 65

def total_amount : ℕ := borrowed_brother + borrowed_father + borrowed_mother + gift_grandmother + savings

theorem fraction_spent (h : total_amount - remaining = 195) :
  (total_amount - remaining : ℚ) / total_amount = 3 / 4 := by
  sorry

end NUMINAMATH_CALUDE_fraction_spent_l1878_187866


namespace NUMINAMATH_CALUDE_garden_furniture_cost_ratio_l1878_187803

/-- Given a garden table and bench with a combined cost of 750 and the bench costing 250,
    prove that the ratio of the table's cost to the bench's cost is 2:1. -/
theorem garden_furniture_cost_ratio :
  ∀ (table_cost bench_cost : ℝ),
    bench_cost = 250 →
    table_cost + bench_cost = 750 →
    ∃ (n : ℕ), table_cost = n * bench_cost →
    table_cost / bench_cost = 2 := by
  sorry

end NUMINAMATH_CALUDE_garden_furniture_cost_ratio_l1878_187803


namespace NUMINAMATH_CALUDE_lcm_18_24_l1878_187800

theorem lcm_18_24 : Nat.lcm 18 24 = 72 := by
  sorry

end NUMINAMATH_CALUDE_lcm_18_24_l1878_187800


namespace NUMINAMATH_CALUDE_amount_in_paise_l1878_187810

theorem amount_in_paise : 
  let a : ℝ := 130
  let percentage : ℝ := 0.5
  let amount_in_rupees : ℝ := (percentage / 100) * a
  let paise_per_rupee : ℝ := 100
  (percentage / 100 * a) * paise_per_rupee = 65 := by
  sorry

end NUMINAMATH_CALUDE_amount_in_paise_l1878_187810


namespace NUMINAMATH_CALUDE_smallest_angle_for_trig_equation_l1878_187802

theorem smallest_angle_for_trig_equation : 
  ∃ y : ℝ, y > 0 ∧ 
  (∀ z : ℝ, z > 0 → 6 * Real.sin z * Real.cos z ^ 3 - 6 * Real.sin z ^ 3 * Real.cos z = 3/2 → y ≤ z) ∧
  6 * Real.sin y * Real.cos y ^ 3 - 6 * Real.sin y ^ 3 * Real.cos y = 3/2 ∧
  y = 7.5 * π / 180 := by
  sorry

end NUMINAMATH_CALUDE_smallest_angle_for_trig_equation_l1878_187802


namespace NUMINAMATH_CALUDE_unique_solution_quadratic_equation_l1878_187855

theorem unique_solution_quadratic_equation (m n : ℤ) :
  m^2 - 2*m*n + 2*n^2 - 4*n + 4 = 0 → m = 2 ∧ n = 2 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_quadratic_equation_l1878_187855


namespace NUMINAMATH_CALUDE_triangle_inequality_l1878_187867

-- Define a triangle structure
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  α : ℝ
  positive_sides : 0 < a ∧ 0 < b ∧ 0 < c
  angle_range : 0 < α ∧ α < π
  cosine_rule : 2 * b * c * Real.cos α = b^2 + c^2 - a^2

-- State the theorem
theorem triangle_inequality (t : Triangle) : 
  (2 * t.b * t.c * Real.cos t.α) / (t.b + t.c) < t.b + t.c - t.a ∧ 
  t.b + t.c - t.a < (2 * t.b * t.c) / t.a :=
sorry

end NUMINAMATH_CALUDE_triangle_inequality_l1878_187867


namespace NUMINAMATH_CALUDE_largest_number_on_board_l1878_187806

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

def ends_in_four (n : ℕ) : Prop := n % 10 = 4

def set_of_interest : Set ℕ := {n | is_two_digit n ∧ n % 6 = 0 ∧ ends_in_four n}

theorem largest_number_on_board : 
  ∃ (m : ℕ), m ∈ set_of_interest ∧ ∀ (n : ℕ), n ∈ set_of_interest → n ≤ m ∧ m = 84 :=
sorry

end NUMINAMATH_CALUDE_largest_number_on_board_l1878_187806


namespace NUMINAMATH_CALUDE_geometric_sequence_minimum_l1878_187822

-- Define a geometric sequence
def geometric_sequence (a₁ : ℝ) (q : ℝ) (n : ℕ) : ℝ := a₁ * q ^ (n - 1)

-- Define the problem statement
theorem geometric_sequence_minimum (a₁ : ℝ) (q : ℝ) :
  (a₁ > 0) →
  (q > 0) →
  (geometric_sequence a₁ q 2017 = geometric_sequence a₁ q 2016 + 2 * geometric_sequence a₁ q 2015) →
  (∃ m n : ℕ, (geometric_sequence a₁ q m) * (geometric_sequence a₁ q n) = 16 * a₁^2) →
  (∃ m n : ℕ, ∀ k l : ℕ, 4/k + 1/l ≥ 4/m + 1/n ∧ 4/m + 1/n = 3/2) :=
by sorry

end NUMINAMATH_CALUDE_geometric_sequence_minimum_l1878_187822


namespace NUMINAMATH_CALUDE_spice_difference_l1878_187894

def cinnamon : ℝ := 0.6666666666666666
def nutmeg : ℝ := 0.5
def ginger : ℝ := 0.4444444444444444

def total_difference : ℝ := |cinnamon - nutmeg| + |nutmeg - ginger| + |cinnamon - ginger|

theorem spice_difference : total_difference = 0.4444444444444444 := by sorry

end NUMINAMATH_CALUDE_spice_difference_l1878_187894


namespace NUMINAMATH_CALUDE_magic_money_box_theorem_l1878_187827

/-- Represents the state of the magic money box on a given day -/
structure BoxState :=
  (day : Nat)
  (value : Nat)

/-- Calculates the next day's value based on the current state and added coins -/
def nextDayValue (state : BoxState) (added : Nat) : Nat :=
  (state.value * (state.day + 2) + added)

/-- Simulates the magic money box for a week -/
def simulateWeek : Nat :=
  let monday := BoxState.mk 0 2
  let tuesday := BoxState.mk 1 (nextDayValue monday 5)
  let wednesday := BoxState.mk 2 (nextDayValue tuesday 10)
  let thursday := BoxState.mk 3 (nextDayValue wednesday 25)
  let friday := BoxState.mk 4 (nextDayValue thursday 50)
  let saturday := BoxState.mk 5 (nextDayValue friday 0)
  let sunday := BoxState.mk 6 (nextDayValue saturday 0)
  sunday.value

theorem magic_money_box_theorem : simulateWeek = 142240 := by
  sorry

end NUMINAMATH_CALUDE_magic_money_box_theorem_l1878_187827


namespace NUMINAMATH_CALUDE_total_spent_is_413_06_l1878_187805

/-- Calculates the total amount spent including sales tax -/
def total_spent (speakers_cost cd_player_cost tires_cost tax_rate : ℝ) : ℝ :=
  let subtotal := speakers_cost + cd_player_cost + tires_cost
  let tax := subtotal * tax_rate
  subtotal + tax

/-- Theorem stating that the total amount spent is $413.06 -/
theorem total_spent_is_413_06 :
  total_spent 136.01 139.38 112.46 0.065 = 413.06 := by
  sorry

end NUMINAMATH_CALUDE_total_spent_is_413_06_l1878_187805


namespace NUMINAMATH_CALUDE_pythagorean_side_divisible_by_five_l1878_187896

theorem pythagorean_side_divisible_by_five (a b c : ℕ+) (h : a^2 + b^2 = c^2) :
  5 ∣ a ∨ 5 ∣ b ∨ 5 ∣ c := by
sorry

end NUMINAMATH_CALUDE_pythagorean_side_divisible_by_five_l1878_187896


namespace NUMINAMATH_CALUDE_tissue_count_after_use_l1878_187823

def initial_tissue_count : ℕ := 97
def used_tissue_count : ℕ := 4

theorem tissue_count_after_use :
  initial_tissue_count - used_tissue_count = 93 := by
  sorry

end NUMINAMATH_CALUDE_tissue_count_after_use_l1878_187823


namespace NUMINAMATH_CALUDE_max_value_3m_4n_l1878_187860

theorem max_value_3m_4n (m n : ℕ+) : 
  (m.val * (m.val + 1) + n.val^2 = 1987) → 
  (∀ k l : ℕ+, k.val * (k.val + 1) + l.val^2 = 1987 → 3 * k.val + 4 * l.val ≤ 3 * m.val + 4 * n.val) →
  3 * m.val + 4 * n.val = 221 :=
by sorry

end NUMINAMATH_CALUDE_max_value_3m_4n_l1878_187860


namespace NUMINAMATH_CALUDE_hash_eight_two_l1878_187847

-- Define the # operation
def hash (a b : ℝ) : ℝ := (a + b)^3 * (a - b)

-- Theorem statement
theorem hash_eight_two : hash 8 2 = 6000 := by
  sorry

end NUMINAMATH_CALUDE_hash_eight_two_l1878_187847


namespace NUMINAMATH_CALUDE_jack_and_jill_games_l1878_187879

/-- A game between Jack and Jill -/
structure Game where
  winner : Bool  -- true if Jack wins, false if Jill wins

/-- The score of a player in a single game -/
def score (g : Game) (isJack : Bool) : Nat :=
  if g.winner == isJack then 2 else 1

/-- The total score of a player across multiple games -/
def totalScore (games : List Game) (isJack : Bool) : Nat :=
  (games.map (fun g => score g isJack)).sum

theorem jack_and_jill_games 
  (games : List Game) 
  (h1 : games.length > 0)
  (h2 : (games.filter (fun g => g.winner)).length = 4)  -- Jack won 4 games
  (h3 : totalScore games false = 10)  -- Jill's final score is 10
  : games.length = 7 := by
  sorry


end NUMINAMATH_CALUDE_jack_and_jill_games_l1878_187879


namespace NUMINAMATH_CALUDE_square_of_negative_x_plus_one_l1878_187886

theorem square_of_negative_x_plus_one (x : ℝ) : (-x - 1)^2 = x^2 + 2*x + 1 := by
  sorry

end NUMINAMATH_CALUDE_square_of_negative_x_plus_one_l1878_187886


namespace NUMINAMATH_CALUDE_line_inclination_angle_l1878_187812

/-- The inclination angle of a line given by parametric equations -/
def inclinationAngle (x y : ℝ → ℝ) : ℝ := sorry

/-- Cosine of 20 degrees -/
def cos20 : ℝ := sorry

/-- Sine of 20 degrees -/
def sin20 : ℝ := sorry

theorem line_inclination_angle :
  let x : ℝ → ℝ := λ t => -t * cos20
  let y : ℝ → ℝ := λ t => 3 + t * sin20
  inclinationAngle x y = 160 * π / 180 := by sorry

end NUMINAMATH_CALUDE_line_inclination_angle_l1878_187812


namespace NUMINAMATH_CALUDE_range_of_a_l1878_187841

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then (1 - a) * x + 3 else Real.log x - 2 * a

theorem range_of_a (a : ℝ) :
  (∀ y : ℝ, ∃ x : ℝ, f a x = y) → -4 ≤ a ∧ a < 1 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l1878_187841


namespace NUMINAMATH_CALUDE_distribute_graduates_eq_90_l1878_187874

/-- The number of ways to evenly distribute 6 graduates to 3 schools -/
def distribute_graduates : ℕ :=
  Nat.choose 6 2 * Nat.choose 4 2 * Nat.choose 2 2

/-- Theorem stating that the number of ways to distribute graduates is 90 -/
theorem distribute_graduates_eq_90 : distribute_graduates = 90 := by
  sorry

end NUMINAMATH_CALUDE_distribute_graduates_eq_90_l1878_187874


namespace NUMINAMATH_CALUDE_license_plate_count_l1878_187877

def digit_choices : ℕ := 10
def letter_choices : ℕ := 26
def num_digits : ℕ := 5
def num_letters : ℕ := 3
def num_slots : ℕ := num_digits + 1

def choose (n k : ℕ) : ℕ := Nat.choose n k

theorem license_plate_count : 
  (digit_choices ^ num_digits) * (letter_choices ^ num_letters) * (choose num_slots num_letters) = 35152000000 := by
  sorry

end NUMINAMATH_CALUDE_license_plate_count_l1878_187877


namespace NUMINAMATH_CALUDE_triangle_angle_sum_l1878_187899

theorem triangle_angle_sum (A B C : ℝ) (h1 : A = 75) (h2 : B = 40) : C = 65 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_sum_l1878_187899


namespace NUMINAMATH_CALUDE_correct_operation_l1878_187869

theorem correct_operation (a b : ℝ) : 3*a + (a - 3*b) = 4*a - 3*b := by sorry

end NUMINAMATH_CALUDE_correct_operation_l1878_187869


namespace NUMINAMATH_CALUDE_triangle_sin_A_l1878_187897

theorem triangle_sin_A (A B C : Real) (a b c : Real) :
  -- Triangle ABC with sides a, b, c opposite to angles A, B, C
  (a > 0) → (b > 0) → (c > 0) →
  (A > 0) → (B > 0) → (C > 0) →
  (A + B + C = Real.pi) →
  -- Given conditions
  (a = 2) →
  (b = 3) →
  (Real.tan B = 3) →
  -- Law of Sines (assumed as part of triangle definition)
  (a / Real.sin A = b / Real.sin B) →
  -- Conclusion
  Real.sin A = Real.sqrt 10 / 5 := by
sorry

end NUMINAMATH_CALUDE_triangle_sin_A_l1878_187897


namespace NUMINAMATH_CALUDE_photocopy_cost_l1878_187898

/-- The cost of a single photocopy --/
def C : ℝ := sorry

/-- The discount rate for large orders --/
def discount_rate : ℝ := 0.25

/-- The number of copies in a large order --/
def large_order : ℕ := 160

/-- The total cost savings when placing a large order --/
def total_savings : ℝ := 0.80

theorem photocopy_cost :
  C = 0.02 :=
by sorry

end NUMINAMATH_CALUDE_photocopy_cost_l1878_187898


namespace NUMINAMATH_CALUDE_first_angle_measure_l1878_187858

theorem first_angle_measure (a b c : ℝ) : 
  a + b + c = 180 →  -- sum of angles in a triangle is 180 degrees
  b = 3 * a →        -- second angle is three times the first
  c = 2 * a - 12 →   -- third angle is 12 degrees less than twice the first
  a = 32 :=          -- prove that the first angle is 32 degrees
by sorry

end NUMINAMATH_CALUDE_first_angle_measure_l1878_187858


namespace NUMINAMATH_CALUDE_ab_value_l1878_187824

theorem ab_value (a b : ℝ) (h : (a - 2)^2 + |b + 3| = 0) : a * b = -6 := by
  sorry

end NUMINAMATH_CALUDE_ab_value_l1878_187824


namespace NUMINAMATH_CALUDE_sum_of_digits_l1878_187807

/-- Given a three-digit number of the form 4a4, where 'a' is a single digit,
    we add 258 to it to get a three-digit number of the form 7b2,
    where 'b' is also a single digit. If 7b2 is divisible by 3,
    then a + b = 4. -/
theorem sum_of_digits (a b : Nat) : 
  (a ≥ 0 ∧ a ≤ 9) →  -- 'a' is a single digit
  (b ≥ 0 ∧ b ≤ 9) →  -- 'b' is a single digit
  (400 + 10 * a + 4) + 258 = 700 + 10 * b + 2 →  -- 4a4 + 258 = 7b2
  (700 + 10 * b + 2) % 3 = 0 →  -- 7b2 is divisible by 3
  a + b = 4 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_digits_l1878_187807


namespace NUMINAMATH_CALUDE_last_two_digits_of_2005_power_1989_l1878_187811

theorem last_two_digits_of_2005_power_1989 :
  (2005^1989) % 100 = 25 := by
  sorry

end NUMINAMATH_CALUDE_last_two_digits_of_2005_power_1989_l1878_187811


namespace NUMINAMATH_CALUDE_relationship_xyz_l1878_187895

theorem relationship_xyz (a : ℝ) (x y z : ℝ) 
  (h1 : 0 < a) (h2 : a < 1) 
  (hx : x = a^a) (hy : y = a) (hz : z = Real.log a / Real.log a) : 
  z > x ∧ x > y := by sorry

end NUMINAMATH_CALUDE_relationship_xyz_l1878_187895


namespace NUMINAMATH_CALUDE_distance_between_cities_l1878_187857

theorem distance_between_cities (v1 v2 t_diff : ℝ) (h1 : v1 = 60) (h2 : v2 = 70) (h3 : t_diff = 0.25) :
  let t := (v2 * t_diff) / (v2 - v1)
  v1 * t = 105 :=
by sorry

end NUMINAMATH_CALUDE_distance_between_cities_l1878_187857


namespace NUMINAMATH_CALUDE_bowl_glass_pairings_l1878_187814

/-- The number of bowls -/
def num_bowls : ℕ := 5

/-- The number of glasses -/
def num_glasses : ℕ := 5

/-- The number of unique bowl colors -/
def num_bowl_colors : ℕ := 5

/-- The number of unique glass colors -/
def num_glass_colors : ℕ := 3

/-- The number of red glasses -/
def num_red_glasses : ℕ := 2

/-- The number of blue glasses -/
def num_blue_glasses : ℕ := 2

/-- The number of yellow glasses -/
def num_yellow_glasses : ℕ := 1

theorem bowl_glass_pairings :
  num_bowls * num_glasses = 25 :=
sorry

end NUMINAMATH_CALUDE_bowl_glass_pairings_l1878_187814


namespace NUMINAMATH_CALUDE_sequence_properties_l1878_187816

-- Define the arithmetic sequence a_n and its sum S_n
def a (n : ℕ) : ℝ := sorry
def S (n : ℕ) : ℝ := sorry

-- Define the geometric sequence b_n and its sum T_n
def b (n : ℕ) : ℝ := sorry
def T (n : ℕ) : ℝ := sorry

-- State the theorem
theorem sequence_properties :
  (a 3 = 5 ∧ S 3 = 9) →
  (∀ n : ℕ, a n = 2 * n - 1) ∧
  (∃ q : ℝ, q > 0 ∧ b 3 = a 5 ∧ T 3 = 13 ∧
    ∀ n : ℕ, T n = (3^n - 1) / 2) :=
by sorry

end NUMINAMATH_CALUDE_sequence_properties_l1878_187816


namespace NUMINAMATH_CALUDE_unique_496_consecutive_sum_l1878_187829

/-- Represents a sequence of consecutive positive integers -/
structure ConsecutiveSequence where
  start : Nat
  length : Nat

/-- Checks if a ConsecutiveSequence sums to the target value -/
def sumTo (seq : ConsecutiveSequence) (target : Nat) : Prop :=
  seq.length * seq.start + seq.length * (seq.length - 1) / 2 = target

/-- Checks if a ConsecutiveSequence is valid (length ≥ 2) -/
def isValid (seq : ConsecutiveSequence) : Prop :=
  seq.length ≥ 2

theorem unique_496_consecutive_sum :
  ∃! seq : ConsecutiveSequence, isValid seq ∧ sumTo seq 496 :=
sorry

end NUMINAMATH_CALUDE_unique_496_consecutive_sum_l1878_187829


namespace NUMINAMATH_CALUDE_joyce_basketball_shots_l1878_187881

theorem joyce_basketball_shots (initial_shots initial_made next_shots : ℕ) 
  (initial_average new_average : ℚ) : 
  initial_shots = 40 →
  initial_made = 15 →
  next_shots = 15 →
  initial_average = 375/1000 →
  new_average = 45/100 →
  ∃ (next_made : ℕ), 
    next_made = 10 ∧ 
    (initial_made + next_made : ℚ) / (initial_shots + next_shots) = new_average :=
by sorry

end NUMINAMATH_CALUDE_joyce_basketball_shots_l1878_187881


namespace NUMINAMATH_CALUDE_cara_seating_arrangements_l1878_187830

/-- The number of people in the circular arrangement -/
def total_people : ℕ := 8

/-- The number of friends excluding Cara and Mark -/
def remaining_friends : ℕ := total_people - 2

/-- The number of different pairs Cara could be sitting between -/
def possible_pairs : ℕ := remaining_friends

theorem cara_seating_arrangements :
  possible_pairs = 6 :=
sorry

end NUMINAMATH_CALUDE_cara_seating_arrangements_l1878_187830


namespace NUMINAMATH_CALUDE_toy_bridge_weight_l1878_187819

theorem toy_bridge_weight (total_weight : ℕ) (num_full_cans : ℕ) (soda_weight : ℕ) (empty_can_weight : ℕ) :
  total_weight = 88 →
  num_full_cans = 6 →
  soda_weight = 12 →
  empty_can_weight = 2 →
  (num_full_cans * (soda_weight + empty_can_weight) + (total_weight - num_full_cans * (soda_weight + empty_can_weight))) / empty_can_weight = 2 :=
by sorry

end NUMINAMATH_CALUDE_toy_bridge_weight_l1878_187819


namespace NUMINAMATH_CALUDE_plane_equation_proof_l1878_187820

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a plane equation in the form Ax + By + Cz + D = 0 -/
structure PlaneEquation where
  A : ℤ
  B : ℤ
  C : ℤ
  D : ℤ

/-- Check if a point satisfies a plane equation -/
def satisfiesPlaneEquation (p : Point3D) (eq : PlaneEquation) : Prop :=
  eq.A * p.x + eq.B * p.y + eq.C * p.z + eq.D = 0

/-- Check if two plane equations are parallel -/
def areParallelPlanes (eq1 eq2 : PlaneEquation) : Prop :=
  ∃ (k : ℚ), k ≠ 0 ∧ eq1.A = k * eq2.A ∧ eq1.B = k * eq2.B ∧ eq1.C = k * eq2.C

theorem plane_equation_proof : 
  let givenPoint : Point3D := ⟨2, -3, 1⟩
  let givenPlane : PlaneEquation := ⟨3, -2, 1, -5⟩
  let resultPlane : PlaneEquation := ⟨3, -2, 1, -13⟩
  satisfiesPlaneEquation givenPoint resultPlane ∧ 
  areParallelPlanes resultPlane givenPlane ∧
  resultPlane.A > 0 ∧
  Nat.gcd (Nat.gcd (Nat.gcd (Int.natAbs resultPlane.A) (Int.natAbs resultPlane.B)) 
                   (Int.natAbs resultPlane.C)) 
          (Int.natAbs resultPlane.D) = 1 :=
by
  sorry

end NUMINAMATH_CALUDE_plane_equation_proof_l1878_187820


namespace NUMINAMATH_CALUDE_system_solution_l1878_187849

theorem system_solution (x y m : ℝ) : 
  (2 * x + y = 4) → 
  (x + 2 * y = m) → 
  (x + y = 1) → 
  m = -1 := by
sorry

end NUMINAMATH_CALUDE_system_solution_l1878_187849


namespace NUMINAMATH_CALUDE_sqrt_abs_sum_zero_implies_sum_power_l1878_187831

theorem sqrt_abs_sum_zero_implies_sum_power (a b : ℝ) : 
  (Real.sqrt (a + 2) + |b - 1| = 0) → ((a + b)^2023 = -1) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_abs_sum_zero_implies_sum_power_l1878_187831


namespace NUMINAMATH_CALUDE_parabola_from_circles_l1878_187856

-- Define the two circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 - 4 = 0
def circle2 (x y : ℝ) : Prop := x^2 + y^2 + y - 3 = 0

-- Define the directrix
def directrix (y : ℝ) : Prop := y = -1

-- Define the parabola
def parabola (x y : ℝ) : Prop := x^2 = 4*y

theorem parabola_from_circles :
  ∀ (x y : ℝ),
  (∃ (x₁ y₁ x₂ y₂ : ℝ), circle1 x₁ y₁ ∧ circle2 x₂ y₂ ∧ directrix y₁ ∧ directrix y₂) →
  parabola x y :=
by sorry

end NUMINAMATH_CALUDE_parabola_from_circles_l1878_187856


namespace NUMINAMATH_CALUDE_xy_sum_over_three_l1878_187887

theorem xy_sum_over_three (x y : ℚ) 
  (eq1 : 2 * x + y = 6) 
  (eq2 : x + 2 * y = 5) : 
  (x + y) / 3 = 11 / 9 := by
  sorry

end NUMINAMATH_CALUDE_xy_sum_over_three_l1878_187887


namespace NUMINAMATH_CALUDE_fourth_test_score_for_average_l1878_187885

def test1 : ℕ := 80
def test2 : ℕ := 70
def test3 : ℕ := 90
def test4 : ℕ := 100
def targetAverage : ℕ := 85

theorem fourth_test_score_for_average :
  (test1 + test2 + test3 + test4) / 4 = targetAverage :=
sorry

end NUMINAMATH_CALUDE_fourth_test_score_for_average_l1878_187885


namespace NUMINAMATH_CALUDE_no_finite_moves_to_fill_board_l1878_187818

-- Define the chessboard as a type
def Chessboard := ℤ × ℤ

-- Define the set A
def A : Set Chessboard :=
  {p | 100 ∣ p.1 ∧ 100 ∣ p.2}

-- Define a king's move
def is_valid_move (start finish : Chessboard) : Prop :=
  (start = finish) ∨
  (abs (start.1 - finish.1) ≤ 1 ∧ abs (start.2 - finish.2) ≤ 1)

-- Define the initial configuration of kings
def initial_kings : Set Chessboard :=
  {p | p ∉ A}

-- Define the state after k moves
def state_after_moves (k : ℕ) : Set Chessboard → Set Chessboard :=
  sorry

-- The main theorem
theorem no_finite_moves_to_fill_board :
  ¬ ∃ (k : ℕ), (state_after_moves k initial_kings) = Set.univ :=
sorry

end NUMINAMATH_CALUDE_no_finite_moves_to_fill_board_l1878_187818


namespace NUMINAMATH_CALUDE_yellow_raisins_cups_l1878_187832

theorem yellow_raisins_cups (total_raisins : Real) (black_raisins : Real) (yellow_raisins : Real) :
  total_raisins = 0.7 →
  black_raisins = 0.4 →
  total_raisins = yellow_raisins + black_raisins →
  yellow_raisins = 0.3 := by
  sorry

end NUMINAMATH_CALUDE_yellow_raisins_cups_l1878_187832


namespace NUMINAMATH_CALUDE_calculate_premium_rate_l1878_187837

/-- Calculates the premium rate for shares given the investment details --/
theorem calculate_premium_rate (investment total_dividend face_value dividend_rate : ℚ)
  (h1 : investment = 14400)
  (h2 : total_dividend = 600)
  (h3 : face_value = 100)
  (h4 : dividend_rate = 5 / 100) :
  ∃ premium_rate : ℚ,
    premium_rate = 20 ∧
    (investment / (face_value + premium_rate)) * (face_value * dividend_rate) = total_dividend :=
by sorry

end NUMINAMATH_CALUDE_calculate_premium_rate_l1878_187837


namespace NUMINAMATH_CALUDE_system_of_inequalities_l1878_187809

theorem system_of_inequalities (x : ℝ) : 
  (x + 1 < 5 ∧ (2 * x - 1) / 3 ≥ 1) ↔ 2 ≤ x ∧ x < 4 := by
  sorry

end NUMINAMATH_CALUDE_system_of_inequalities_l1878_187809


namespace NUMINAMATH_CALUDE_not_always_preservable_flight_relations_l1878_187813

/-- Represents a city in the country -/
structure City where
  id : Nat

/-- Represents the flight guide for the country -/
structure FlightGuide where
  cities : Finset City
  has_direct_flight : City → City → Bool

/-- Represents a permutation of city IDs -/
def CityPermutation := Nat → Nat

/-- Theorem stating that it's not always possible to maintain flight relations after swapping city numbers -/
theorem not_always_preservable_flight_relations :
  ∃ (fg : FlightGuide) (m n : City),
    m ∈ fg.cities → n ∈ fg.cities → m ≠ n →
    ¬∀ (p : CityPermutation),
      (∀ c : City, c ∈ fg.cities → p (c.id) ≠ c.id → (c = m ∨ c = n)) →
      (p m.id = n.id ∧ p n.id = m.id) →
      (∀ c1 c2 : City, c1 ∈ fg.cities → c2 ∈ fg.cities →
        fg.has_direct_flight c1 c2 = fg.has_direct_flight
          ⟨p c1.id⟩ ⟨p c2.id⟩) :=
sorry

end NUMINAMATH_CALUDE_not_always_preservable_flight_relations_l1878_187813


namespace NUMINAMATH_CALUDE_waiter_section_proof_l1878_187843

/-- Calculates the number of customers who left a waiter's section. -/
def customers_who_left (initial_customers : ℕ) (remaining_tables : ℕ) (people_per_table : ℕ) : ℕ :=
  initial_customers - (remaining_tables * people_per_table)

/-- Proves that 17 customers left the waiter's section given the initial conditions. -/
theorem waiter_section_proof :
  customers_who_left 62 5 9 = 17 := by
  sorry

end NUMINAMATH_CALUDE_waiter_section_proof_l1878_187843


namespace NUMINAMATH_CALUDE_quadratic_discriminant_l1878_187826

/-- The discriminant of the quadratic equation 2x^2 + (2 + 1/2)x + 1/2 is 9/4 -/
theorem quadratic_discriminant : 
  let a : ℚ := 2
  let b : ℚ := 5/2
  let c : ℚ := 1/2
  let discriminant := b^2 - 4*a*c
  discriminant = 9/4 := by
sorry

end NUMINAMATH_CALUDE_quadratic_discriminant_l1878_187826


namespace NUMINAMATH_CALUDE_square_19_on_top_l1878_187850

/-- Represents a position on the 9x9 grid -/
structure Position :=
  (row : Fin 9)
  (col : Fin 9)

/-- Represents the state of the grid after folding -/
structure FoldedGrid :=
  (top_square : Nat)

/-- Defines the initial 9x9 grid -/
def initial_grid : List (List Nat) :=
  List.range 9 |> List.map (fun i => List.range 9 |> List.map (fun j => i * 9 + j + 1))

/-- Performs the sequence of folds on the grid -/
def fold_grid (grid : List (List Nat)) : FoldedGrid :=
  sorry

/-- The main theorem stating that square 19 is on top after folding -/
theorem square_19_on_top :
  (fold_grid initial_grid).top_square = 19 := by sorry

end NUMINAMATH_CALUDE_square_19_on_top_l1878_187850


namespace NUMINAMATH_CALUDE_apple_pear_ratio_l1878_187854

/-- Proves that the ratio of initial apples to initial pears is 2:1 given the conditions --/
theorem apple_pear_ratio (initial_pears initial_oranges : ℕ) 
  (fruits_given_away fruits_left : ℕ) : 
  initial_pears = 10 →
  initial_oranges = 20 →
  fruits_given_away = 2 →
  fruits_left = 44 →
  ∃ (initial_apples : ℕ), 
    initial_apples - fruits_given_away + 
    (initial_pears - fruits_given_away) + 
    (initial_oranges - fruits_given_away) = fruits_left ∧
    initial_apples / initial_pears = 2 := by
  sorry

end NUMINAMATH_CALUDE_apple_pear_ratio_l1878_187854


namespace NUMINAMATH_CALUDE_tinas_hourly_wage_l1878_187888

/-- Represents Tina's work schedule and pay structure -/
structure WorkSchedule where
  regularHours : ℕ := 8
  overtimeRate : ℚ := 3/2
  daysWorked : ℕ := 5
  hoursPerDay : ℕ := 10
  totalPay : ℚ := 990

/-- Calculates Tina's hourly wage based on her work schedule -/
def calculateHourlyWage (schedule : WorkSchedule) : ℚ :=
  let regularHoursPerWeek := schedule.regularHours * schedule.daysWorked
  let overtimeHoursPerWeek := (schedule.hoursPerDay - schedule.regularHours) * schedule.daysWorked
  let totalHoursEquivalent := regularHoursPerWeek + overtimeHoursPerWeek * schedule.overtimeRate
  schedule.totalPay / totalHoursEquivalent

/-- Theorem stating that Tina's hourly wage is $18 -/
theorem tinas_hourly_wage (schedule : WorkSchedule) : 
  calculateHourlyWage schedule = 18 := by
  sorry

#eval calculateHourlyWage {} -- Should output 18

end NUMINAMATH_CALUDE_tinas_hourly_wage_l1878_187888


namespace NUMINAMATH_CALUDE_berry_problem_l1878_187844

/-- Proves that given the conditions in the berry problem, Steve started with 8.5 berries and Amanda started with 3.5 berries. -/
theorem berry_problem (stacy_initial : ℝ) (steve_takes : ℝ) (amanda_takes : ℝ) (amanda_more : ℝ)
  (h1 : stacy_initial = 32)
  (h2 : steve_takes = 4)
  (h3 : amanda_takes = 3.25)
  (h4 : amanda_more = 5.75)
  (h5 : steve_takes + (stacy_initial / 2 - 7.5) = stacy_initial / 2 - 7.5 + steve_takes - amanda_takes + amanda_more) :
  (stacy_initial / 2 - 7.5 = 8.5) ∧ (stacy_initial / 2 - 7.5 + steve_takes - amanda_takes - amanda_more = 3.5) :=
by sorry

end NUMINAMATH_CALUDE_berry_problem_l1878_187844


namespace NUMINAMATH_CALUDE_selection_with_condition_l1878_187808

/-- The number of ways to choose k items from n items -/
def choose (n k : ℕ) : ℕ := sorry

/-- The total number of students -/
def total_students : ℕ := 10

/-- The number of students to be selected -/
def selected_students : ℕ := 4

/-- The number of students excluding the two specific students -/
def remaining_students : ℕ := total_students - 2

theorem selection_with_condition :
  (choose total_students selected_students) - (choose remaining_students selected_students) = 140 := by
  sorry

end NUMINAMATH_CALUDE_selection_with_condition_l1878_187808


namespace NUMINAMATH_CALUDE_rafael_monday_hours_l1878_187852

/-- Represents the number of hours Rafael worked on Monday -/
def monday_hours : ℕ := sorry

/-- Represents the number of hours Rafael worked on Tuesday -/
def tuesday_hours : ℕ := 8

/-- Represents the number of hours Rafael has left to work in the week -/
def remaining_hours : ℕ := 20

/-- Represents Rafael's hourly pay rate in dollars -/
def hourly_rate : ℕ := 20

/-- Represents Rafael's total earnings for the week in dollars -/
def total_earnings : ℕ := 760

/-- Theorem stating that Rafael worked 10 hours on Monday -/
theorem rafael_monday_hours :
  monday_hours = 10 :=
by sorry

end NUMINAMATH_CALUDE_rafael_monday_hours_l1878_187852


namespace NUMINAMATH_CALUDE_f_not_in_quadrant_II_l1878_187882

-- Define the linear function
def f (x : ℝ) : ℝ := 3 * x - 4

-- Define Quadrant II
def in_quadrant_II (x y : ℝ) : Prop := x < 0 ∧ y > 0

-- Theorem: The function f does not pass through Quadrant II
theorem f_not_in_quadrant_II :
  ∀ x : ℝ, ¬(in_quadrant_II x (f x)) :=
by
  sorry

end NUMINAMATH_CALUDE_f_not_in_quadrant_II_l1878_187882


namespace NUMINAMATH_CALUDE_intersection_orthogonality_l1878_187848

/-- The line equation -/
def line (x y : ℝ) : Prop := y = 2 * Real.sqrt 2 * (x - 1)

/-- The parabola equation -/
def parabola (x y : ℝ) : Prop := y^2 = 4 * x

/-- Point A satisfies both line and parabola equations -/
def point_A (x y : ℝ) : Prop := line x y ∧ parabola x y

/-- Point B satisfies both line and parabola equations -/
def point_B (x y : ℝ) : Prop := line x y ∧ parabola x y

/-- Point M has coordinates (-1, m) -/
def point_M (m : ℝ) : ℝ × ℝ := (-1, m)

/-- The dot product of vectors MA and MB is zero -/
def orthogonal_condition (x_a y_a x_b y_b m : ℝ) : Prop :=
  (x_a + 1) * (x_b + 1) + (y_a - m) * (y_b - m) = 0

theorem intersection_orthogonality (x_a y_a x_b y_b m : ℝ) :
  point_A x_a y_a →
  point_B x_b y_b →
  orthogonal_condition x_a y_a x_b y_b m →
  m = Real.sqrt 2 / 2 := by sorry

end NUMINAMATH_CALUDE_intersection_orthogonality_l1878_187848


namespace NUMINAMATH_CALUDE_max_consecutive_odds_is_five_l1878_187845

/-- Represents a natural number as a list of its digits -/
def Digits := List Nat

/-- Returns the largest digit in a number -/
def largestDigit (n : Digits) : Nat :=
  n.foldl max 0

/-- Adds the largest digit to the number -/
def addLargestDigit (n : Digits) : Digits :=
  sorry

/-- Checks if a number is odd -/
def isOdd (n : Digits) : Bool :=
  sorry

/-- Generates the sequence of numbers following the given rule -/
def generateSequence (start : Digits) : List Digits :=
  sorry

/-- Counts the maximum number of consecutive odd numbers in a list -/
def maxConsecutiveOdds (seq : List Digits) : Nat :=
  sorry

/-- The main theorem stating that the maximum number of consecutive odd numbers is 5 -/
theorem max_consecutive_odds_is_five :
  ∀ start : Digits, maxConsecutiveOdds (generateSequence start) ≤ 5 ∧
  ∃ start : Digits, maxConsecutiveOdds (generateSequence start) = 5 :=
sorry

end NUMINAMATH_CALUDE_max_consecutive_odds_is_five_l1878_187845


namespace NUMINAMATH_CALUDE_expression_simplification_l1878_187873

theorem expression_simplification (a x y : ℝ) : 
  ((-2*a)^6*(-3*a^3) + (2*a^2)^3 / (1 / ((-2)^2 * 3^2 * (x*y)^3))) = 192*a^9 + 288*a^6*(x*y)^3 ∧
  |-(1/8)| + π^3 + (-(1/2)^3 - (1/3)^2) = π^3 - 1/72 :=
by sorry

end NUMINAMATH_CALUDE_expression_simplification_l1878_187873


namespace NUMINAMATH_CALUDE_compound_interest_rate_interest_rate_problem_l1878_187862

/-- Compound interest calculation --/
theorem compound_interest_rate (P A : ℝ) (t n : ℕ) (h : A = P * (1 + 0.25)^(n * t)) :
  ∃ (r : ℝ), A = P * (1 + r)^(n * t) ∧ r = 0.25 :=
by sorry

/-- Problem-specific theorem --/
theorem interest_rate_problem (P A : ℝ) (t n : ℕ) 
  (h_P : P = 1200)
  (h_A : A = 2488.32)
  (h_t : t = 4)
  (h_n : n = 1) :
  ∃ (r : ℝ), A = P * (1 + r)^(n * t) ∧ r = 0.25 :=
by sorry

end NUMINAMATH_CALUDE_compound_interest_rate_interest_rate_problem_l1878_187862


namespace NUMINAMATH_CALUDE_circumcenter_rational_coords_l1878_187835

/-- Given a triangle with rational coordinates, the center of its circumscribed circle has rational coordinates. -/
theorem circumcenter_rational_coords (a₁ a₂ a₃ b₁ b₂ b₃ : ℚ) :
  ∃ (x y : ℚ), 
    (x - a₁)^2 + (y - b₁)^2 = (x - a₂)^2 + (y - b₂)^2 ∧
    (x - a₁)^2 + (y - b₁)^2 = (x - a₃)^2 + (y - b₃)^2 :=
by sorry

end NUMINAMATH_CALUDE_circumcenter_rational_coords_l1878_187835


namespace NUMINAMATH_CALUDE_profit_maximizing_price_l1878_187838

/-- Represents the profit function for a product -/
def profit_function (initial_price initial_volume : ℝ) (price_increase : ℝ) : ℝ → ℝ :=
  λ x => (initial_price + x - 80) * (initial_volume - 20 * x)

/-- Theorem stating that the profit-maximizing price is 95 yuan -/
theorem profit_maximizing_price :
  let initial_price : ℝ := 90
  let initial_volume : ℝ := 400
  let price_increase : ℝ := 1
  let profit := profit_function initial_price initial_volume price_increase
  ∃ (max_price : ℝ), max_price = 95 ∧
    ∀ (x : ℝ), profit x ≤ profit (max_price - initial_price) :=
by sorry

end NUMINAMATH_CALUDE_profit_maximizing_price_l1878_187838


namespace NUMINAMATH_CALUDE_sqrt_five_identity_l1878_187840

theorem sqrt_five_identity (m n a b c d : ℝ) :
  m + n * Real.sqrt 5 = (a + b * Real.sqrt 5) * (c + d * Real.sqrt 5) →
  m - n * Real.sqrt 5 = (a - b * Real.sqrt 5) * (c - d * Real.sqrt 5) :=
by sorry

end NUMINAMATH_CALUDE_sqrt_five_identity_l1878_187840


namespace NUMINAMATH_CALUDE_cos_240_degrees_l1878_187859

theorem cos_240_degrees : Real.cos (240 * π / 180) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_cos_240_degrees_l1878_187859


namespace NUMINAMATH_CALUDE_dawn_cd_count_l1878_187883

theorem dawn_cd_count (dawn kristine : ℕ) 
  (h1 : kristine = dawn + 7)
  (h2 : dawn + kristine = 27) : 
  dawn = 10 := by
sorry

end NUMINAMATH_CALUDE_dawn_cd_count_l1878_187883


namespace NUMINAMATH_CALUDE_hyperbola_properties_l1878_187833

/-- Given a hyperbola C with the equation (x²/a²) - (y²/b²) = 1, where a > 0 and b > 0,
    real axis length 4√2, and eccentricity √6/2, prove the following statements. -/
theorem hyperbola_properties (a b : ℝ) (ha : a > 0) (hb : b > 0) 
  (h_real_axis : 2 * a = 4 * Real.sqrt 2)
  (h_eccentricity : (Real.sqrt (a^2 + b^2)) / a = Real.sqrt 6 / 2) :
  /- 1. The standard equation is x²/8 - y²/4 = 1 -/
  (a^2 = 8 ∧ b^2 = 4) ∧ 
  /- 2. The locus equation of the midpoint Q of AP, where A(3,0) and P is any point on C,
        is ((2x - 3)²/8) - y² = 1 -/
  (∀ x y : ℝ, ((2*x - 3)^2 / 8) - y^2 = 1 ↔ 
    ∃ px py : ℝ, (px^2 / a^2) - (py^2 / b^2) = 1 ∧ x = (px + 3) / 2 ∧ y = py / 2) ∧
  /- 3. The minimum value of |AP| is 3 - 2√2 -/
  (∀ px py : ℝ, (px^2 / a^2) - (py^2 / b^2) = 1 → 
    Real.sqrt ((px - 3)^2 + py^2) ≥ 3 - 2 * Real.sqrt 2) ∧
  (∃ px py : ℝ, (px^2 / a^2) - (py^2 / b^2) = 1 ∧ 
    Real.sqrt ((px - 3)^2 + py^2) = 3 - 2 * Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_properties_l1878_187833


namespace NUMINAMATH_CALUDE_perpendicular_vectors_m_value_l1878_187878

/-- Given two points A and B in a plane, and vectors a and b, 
    prove that if a is perpendicular to b, then m = 1. -/
theorem perpendicular_vectors_m_value 
  (A B : ℝ × ℝ) 
  (h_A : A = (0, 2)) 
  (h_B : B = (3, -1)) 
  (a b : ℝ × ℝ) 
  (h_a : a = B - A) 
  (h_b : b = (1, m)) 
  (h_perp : a.1 * b.1 + a.2 * b.2 = 0) : 
  m = 1 := by
sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_m_value_l1878_187878


namespace NUMINAMATH_CALUDE_problem_solution_l1878_187870

noncomputable def f (a : ℝ) : ℝ → ℝ := fun x ↦ -1/2 + a / (3^x + 1)

theorem problem_solution (a : ℝ) (h_odd : ∀ x, f a x = -(f a (-x))) :
  (a = 1) ∧
  (∀ x y, x < y → f a x > f a y) ∧
  (∀ m, (∃ t ∈ Set.Ioo 1 2, f a (-2*t^2 + t + 1) + f a (t^2 - 2*m*t) ≤ 0) → m < 1/2) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l1878_187870


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l1878_187851

theorem quadratic_inequality_solution (a b : ℝ) :
  (∀ x, ax^2 + bx + 1 > 0 ↔ -1 < x ∧ x < 1/3) →
  a * b = 6 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l1878_187851


namespace NUMINAMATH_CALUDE_binary_101101_to_octal_l1878_187828

def binary_to_octal (b : ℕ) : ℕ := sorry

theorem binary_101101_to_octal :
  binary_to_octal 0b101101 = 0o55 := by sorry

end NUMINAMATH_CALUDE_binary_101101_to_octal_l1878_187828


namespace NUMINAMATH_CALUDE_square_root_fourth_power_l1878_187889

theorem square_root_fourth_power (x : ℝ) : (Real.sqrt x)^4 = 256 → x = 16 := by
  sorry

end NUMINAMATH_CALUDE_square_root_fourth_power_l1878_187889


namespace NUMINAMATH_CALUDE_car_features_l1878_187815

theorem car_features (total : ℕ) (steering : ℕ) (windows : ℕ) (both : ℕ) :
  total = 65 →
  steering = 45 →
  windows = 25 →
  both = 17 →
  total - (steering + windows - both) = 12 := by
sorry

end NUMINAMATH_CALUDE_car_features_l1878_187815


namespace NUMINAMATH_CALUDE_sin_690_degrees_l1878_187842

theorem sin_690_degrees : Real.sin (690 * π / 180) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_sin_690_degrees_l1878_187842


namespace NUMINAMATH_CALUDE_income_calculation_l1878_187865

-- Define the total income
def total_income : ℝ := sorry

-- Define the percentage given to children
def children_percentage : ℝ := 0.2 * 3

-- Define the percentage given to wife
def wife_percentage : ℝ := 0.3

-- Define the remaining percentage after giving to children and wife
def remaining_percentage : ℝ := 1 - children_percentage - wife_percentage

-- Define the percentage donated to orphan house
def orphan_house_percentage : ℝ := 0.05

-- Define the final amount left
def final_amount : ℝ := 40000

-- Theorem to prove
theorem income_calculation : 
  ∃ (total_income : ℝ),
    total_income > 0 ∧
    final_amount = total_income * remaining_percentage * (1 - orphan_house_percentage) ∧
    (total_income ≥ 421052) ∧ (total_income ≤ 421053) :=
by sorry

end NUMINAMATH_CALUDE_income_calculation_l1878_187865


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l1878_187863

theorem quadratic_inequality_range (a : ℝ) : 
  (¬ ∀ x : ℝ, 4 * x^2 + (a - 2) * x + 1 > 0) ↔ 
  (a ≤ -2 ∨ a ≥ 6) := by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l1878_187863


namespace NUMINAMATH_CALUDE_min_translation_for_even_sine_l1878_187836

theorem min_translation_for_even_sine (f : ℝ → ℝ) (m : ℝ) :
  (∀ x, f x = Real.sin (3 * x + π / 4)) →
  m > 0 →
  (∀ x, f (x + m) = f (-x - m)) →
  m ≥ π / 12 ∧ ∃ m₀ > 0, m₀ < m → ¬(∀ x, f (x + m₀) = f (-x - m₀)) :=
by sorry

end NUMINAMATH_CALUDE_min_translation_for_even_sine_l1878_187836


namespace NUMINAMATH_CALUDE_mans_age_twice_sons_l1878_187864

/-- 
Given a man and his son, where:
- The man is currently 30 years older than his son
- The son's present age is 28 years

This theorem proves that it will take 2 years for the man's age 
to be twice his son's age.
-/
theorem mans_age_twice_sons (
  son_age : ℕ) 
  (man_age : ℕ) 
  (h1 : son_age = 28) 
  (h2 : man_age = son_age + 30) : 
  ∃ (years : ℕ), years = 2 ∧ man_age + years = 2 * (son_age + years) :=
sorry

end NUMINAMATH_CALUDE_mans_age_twice_sons_l1878_187864


namespace NUMINAMATH_CALUDE_classroom_ratio_l1878_187890

theorem classroom_ratio (total_students : ℕ) (boys : ℕ) (girls : ℕ) : 
  total_students = 30 → boys = 20 → girls = total_students - boys → 
  (girls : ℚ) / (boys : ℚ) = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_classroom_ratio_l1878_187890


namespace NUMINAMATH_CALUDE_cookie_sales_proof_l1878_187891

/-- The number of homes in Neighborhood A -/
def homes_a : ℕ := 10

/-- The number of boxes each home in Neighborhood A buys -/
def boxes_per_home_a : ℕ := 2

/-- The number of boxes each home in Neighborhood B buys -/
def boxes_per_home_b : ℕ := 5

/-- The cost of each box of cookies in dollars -/
def cost_per_box : ℕ := 2

/-- The total sales in dollars from the better neighborhood -/
def better_sales : ℕ := 50

/-- The number of homes in Neighborhood B -/
def homes_b : ℕ := 5

theorem cookie_sales_proof : 
  homes_b * boxes_per_home_b * cost_per_box = better_sales ∧
  homes_b * boxes_per_home_b * cost_per_box > homes_a * boxes_per_home_a * cost_per_box :=
by sorry

end NUMINAMATH_CALUDE_cookie_sales_proof_l1878_187891
