import Mathlib

namespace most_suitable_sampling_plan_l3925_392552

/-- Represents a production line in the factory -/
structure ProductionLine where
  boxes_per_day : ℕ
  deriving Repr

/-- Represents the factory with its production lines -/
structure Factory where
  production_lines : List ProductionLine
  deriving Repr

/-- Represents a sampling plan -/
inductive SamplingPlan
  | RandomOneFromAll
  | LastFromEach
  | RandomOneFromEach
  | AllFromOne
  deriving Repr

/-- Defines what makes a sampling plan suitable -/
def is_suitable_plan (factory : Factory) (plan : SamplingPlan) : Prop :=
  plan = SamplingPlan.RandomOneFromEach

/-- The theorem stating that randomly selecting one box from each production line is the most suitable sampling plan -/
theorem most_suitable_sampling_plan (factory : Factory) 
  (h1 : factory.production_lines.length = 5)
  (h2 : ∀ line ∈ factory.production_lines, line.boxes_per_day = 20) :
  is_suitable_plan factory SamplingPlan.RandomOneFromEach :=
by
  sorry

#check most_suitable_sampling_plan

end most_suitable_sampling_plan_l3925_392552


namespace least_number_of_trees_l3925_392580

theorem least_number_of_trees : ∃ n : ℕ, 
  (∀ m : ℕ, m < n → (m % 7 ≠ 0 ∨ m % 6 ≠ 0 ∨ m % 4 ≠ 0)) ∧
  n % 7 = 0 ∧ n % 6 = 0 ∧ n % 4 = 0 ∧
  n = 84 := by
sorry

end least_number_of_trees_l3925_392580


namespace monotonic_cubic_implies_m_range_l3925_392524

/-- A function f : ℝ → ℝ is monotonic if it is either monotonically increasing or monotonically decreasing. -/
def Monotonic (f : ℝ → ℝ) : Prop :=
  (∀ x y, x ≤ y → f x ≤ f y) ∨ (∀ x y, x ≤ y → f y ≤ f x)

/-- The main theorem: if f(x) = x^3 + x^2 + mx + 1 is monotonic on ℝ, then m ≥ 1/3. -/
theorem monotonic_cubic_implies_m_range (m : ℝ) :
  Monotonic (fun x => x^3 + x^2 + m*x + 1) → m ≥ 1/3 := by
  sorry

end monotonic_cubic_implies_m_range_l3925_392524


namespace no_reverse_equal_base6_l3925_392509

/-- Function to reverse the digits of a natural number in base 10 --/
def reverseDigits (n : ℕ) : ℕ := sorry

/-- Function to convert a natural number to its base 6 representation --/
def toBase6 (n : ℕ) : ℕ := sorry

/-- Theorem stating that no natural number greater than 5 has its reversed decimal representation equal to its base 6 representation --/
theorem no_reverse_equal_base6 :
  ∀ n : ℕ, n > 5 → reverseDigits n ≠ toBase6 n :=
sorry

end no_reverse_equal_base6_l3925_392509


namespace penny_problem_l3925_392594

theorem penny_problem (initial_pennies : ℕ) (older_pennies : ℕ) (removal_percentage : ℚ) :
  initial_pennies = 200 →
  older_pennies = 30 →
  removal_percentage = 1/5 →
  initial_pennies - older_pennies - Int.floor ((initial_pennies - older_pennies : ℚ) * removal_percentage) = 136 :=
by sorry

end penny_problem_l3925_392594


namespace movie_ticket_cost_l3925_392517

/-- Proves that the cost of each movie ticket is $10.62 --/
theorem movie_ticket_cost (ticket_count : ℕ) (rental_cost movie_cost total_cost : ℚ) :
  ticket_count = 2 →
  rental_cost = 1.59 →
  movie_cost = 13.95 →
  total_cost = 36.78 →
  ∃ (ticket_price : ℚ), 
    ticket_price * ticket_count + rental_cost + movie_cost = total_cost ∧
    ticket_price = 10.62 := by
  sorry

end movie_ticket_cost_l3925_392517


namespace presidency_meeting_arrangements_l3925_392514

/-- The number of schools in the club -/
def num_schools : ℕ := 3

/-- The number of members from each school -/
def members_per_school : ℕ := 6

/-- The number of representatives sent by the host school -/
def host_representatives : ℕ := 3

/-- The number of representatives sent by each non-host school -/
def non_host_representatives : ℕ := 1

/-- The total number of possible ways to arrange the presidency meeting -/
def total_arrangements : ℕ := 2160

theorem presidency_meeting_arrangements :
  (num_schools * (members_per_school.choose host_representatives) *
   (members_per_school.choose non_host_representatives) *
   (members_per_school.choose non_host_representatives)) = total_arrangements :=
by sorry

end presidency_meeting_arrangements_l3925_392514


namespace exists_cycle_l3925_392512

structure Team :=
  (id : Nat)

structure Tournament :=
  (teams : Finset Team)
  (score : Team → Nat)
  (beats : Team → Team → Prop)
  (round_robin : ∀ t1 t2 : Team, t1 ≠ t2 → (beats t1 t2 ∨ beats t2 t1))

theorem exists_cycle (t : Tournament) 
  (h : ∃ t1 t2 : Team, t1 ∈ t.teams ∧ t2 ∈ t.teams ∧ t1 ≠ t2 ∧ t.score t1 = t.score t2) :
  ∃ A B C : Team, A ∈ t.teams ∧ B ∈ t.teams ∧ C ∈ t.teams ∧ 
    t.beats A B ∧ t.beats B C ∧ t.beats C A :=
sorry

end exists_cycle_l3925_392512


namespace divisor_sum_product_2016_l3925_392563

def sum_odd_divisors (n : ℕ) : ℕ := sorry

def sum_even_divisors (n : ℕ) : ℕ := sorry

theorem divisor_sum_product_2016 (n : ℕ) :
  n % 2 = 0 →
  (sum_odd_divisors n) * (sum_even_divisors n) = 2016 ↔ n = 192 ∨ n = 88 := by
  sorry

end divisor_sum_product_2016_l3925_392563


namespace cricket_bat_profit_percentage_l3925_392598

/-- Calculates the profit percentage of a middleman in a series of transactions -/
theorem cricket_bat_profit_percentage 
  (a_cost : ℝ) 
  (a_profit_percent : ℝ) 
  (c_price : ℝ) 
  (h1 : a_cost = 152)
  (h2 : a_profit_percent = 20)
  (h3 : c_price = 228) :
  let a_sell := a_cost * (1 + a_profit_percent / 100)
  let b_profit := c_price - a_sell
  let b_profit_percent := (b_profit / a_sell) * 100
  b_profit_percent = 25 := by
sorry


end cricket_bat_profit_percentage_l3925_392598


namespace fraction_value_at_three_l3925_392585

theorem fraction_value_at_three : 
  let x : ℝ := 3
  (x^8 + 18*x^4 + 81) / (x^4 + 9) = 90 := by sorry

end fraction_value_at_three_l3925_392585


namespace necessary_but_not_sufficient_condition_l3925_392584

theorem necessary_but_not_sufficient_condition :
  ∃ (x : ℝ), (x < 0 ∨ x > 2) ∧ (2*x^2 - 5*x - 3 < 0) ∧
  ∀ (y : ℝ), (2*y^2 - 5*y - 3 ≥ 0) → (y < 0 ∨ y > 2) :=
by sorry

end necessary_but_not_sufficient_condition_l3925_392584


namespace scientific_notation_of_number_l3925_392531

def number : ℕ := 97070000000

theorem scientific_notation_of_number :
  (9.707 : ℝ) * (10 : ℝ) ^ (10 : ℕ) = number := by sorry

end scientific_notation_of_number_l3925_392531


namespace total_cans_stored_l3925_392530

/-- Represents a closet with its storage capacity -/
structure Closet where
  cansPerRow : Nat
  rowsPerShelf : Nat
  numShelves : Nat

/-- Calculates the total number of cans that can be stored in a closet -/
def cansInCloset (c : Closet) : Nat :=
  c.cansPerRow * c.rowsPerShelf * c.numShelves

/-- The first closet in Jack's emergency bunker -/
def closet1 : Closet := ⟨12, 4, 10⟩

/-- The second closet in Jack's emergency bunker -/
def closet2 : Closet := ⟨15, 5, 8⟩

/-- Theorem stating the total number of cans Jack can store -/
theorem total_cans_stored : cansInCloset closet1 + cansInCloset closet2 = 1080 := by
  sorry

end total_cans_stored_l3925_392530


namespace divisor_of_power_of_four_l3925_392591

theorem divisor_of_power_of_four (a : ℕ) (d : ℕ) (h1 : a > 0) (h2 : 2 ∣ a) :
  let p := 4^a
  (p % d = 6) → d = 22 := by
sorry

end divisor_of_power_of_four_l3925_392591


namespace concatenated_not_palindromic_l3925_392510

/-- Represents the concatenation of integers from 1 to n as a natural number -/
def concatenatedNumber (n : ℕ) : ℕ := sorry

/-- Checks if a natural number is palindromic -/
def isPalindromic (num : ℕ) : Prop := sorry

/-- Theorem stating that the concatenated number is not palindromic for n > 1 -/
theorem concatenated_not_palindromic (n : ℕ) (h : n > 1) : 
  ¬(isPalindromic (concatenatedNumber n)) := by sorry

end concatenated_not_palindromic_l3925_392510


namespace pastry_price_is_18_l3925_392571

/-- The price of a pastry satisfies the given conditions -/
theorem pastry_price_is_18 
  (usual_pastries : ℕ) 
  (usual_bread : ℕ) 
  (today_pastries : ℕ) 
  (today_bread : ℕ) 
  (bread_price : ℕ) 
  (daily_difference : ℕ) 
  (h1 : usual_pastries = 20) 
  (h2 : usual_bread = 10) 
  (h3 : today_pastries = 14) 
  (h4 : today_bread = 25) 
  (h5 : bread_price = 4) 
  (h6 : daily_difference = 48) : 
  ∃ (pastry_price : ℕ), 
    pastry_price = 18 ∧ 
    usual_pastries * pastry_price + usual_bread * bread_price - 
    (today_pastries * pastry_price + today_bread * bread_price) = daily_difference :=
by sorry

end pastry_price_is_18_l3925_392571


namespace exp_13pi_over_2_equals_i_l3925_392549

-- Define Euler's formula
axiom euler_formula (θ : ℝ) : Complex.exp (θ * Complex.I) = Complex.cos θ + Complex.I * Complex.sin θ

-- State the theorem
theorem exp_13pi_over_2_equals_i : Complex.exp ((13 * Real.pi / 2) * Complex.I) = Complex.I := by sorry

end exp_13pi_over_2_equals_i_l3925_392549


namespace last_ball_is_white_l3925_392536

/-- Represents the color of a ball -/
inductive BallColor
| White
| Black

/-- Represents the state of the box -/
structure BoxState where
  white : Nat
  black : Nat

/-- The process of drawing and replacing balls -/
def process (state : BoxState) : BoxState :=
  sorry

/-- Predicate to check if the process has terminated (only one ball left) -/
def isTerminated (state : BoxState) : Prop :=
  state.white + state.black = 1

/-- Theorem stating that the last ball is white given an odd initial number of white balls -/
theorem last_ball_is_white 
  (initial_white : Nat) 
  (initial_black : Nat) 
  (h_odd : Odd initial_white) :
  ∃ (final_state : BoxState), 
    (∃ (n : Nat), final_state = (process^[n] ⟨initial_white, initial_black⟩)) ∧ 
    isTerminated final_state ∧ 
    final_state.white = 1 :=
  sorry

end last_ball_is_white_l3925_392536


namespace usable_seats_in_section_C_l3925_392519

def x : ℝ := 60 + 3 * 80
def y : ℝ := 3 * x + 20
def z : ℝ := 2 * y - 30.5

theorem usable_seats_in_section_C : z = 1809.5 := by
  sorry

end usable_seats_in_section_C_l3925_392519


namespace unique_rational_root_l3925_392537

def f (x : ℚ) : ℚ := 6 * x^5 - 4 * x^4 - 16 * x^3 + 8 * x^2 + 4 * x - 3

theorem unique_rational_root :
  ∀ x : ℚ, f x = 0 ↔ x = 1/2 := by
sorry

end unique_rational_root_l3925_392537


namespace orange_ribbons_l3925_392528

theorem orange_ribbons (total : ℕ) (yellow purple orange silver : ℕ) : 
  yellow = total / 4 →
  purple = total / 3 →
  orange = total / 8 →
  silver = 45 →
  yellow + purple + orange + silver = total →
  orange = 19 := by
sorry

end orange_ribbons_l3925_392528


namespace alternating_odd_sum_equals_21_l3925_392583

/-- Calculates the sum of the alternating series of odd numbers from 1 to 41 -/
def alternating_odd_sum : ℤ :=
  let n := 20  -- Number of pairs (1-3), (5-7), etc.
  41 - 2 * n

/-- The sum of the series 1-3+5-7+9-11+13-...-39+41 equals 21 -/
theorem alternating_odd_sum_equals_21 : alternating_odd_sum = 21 := by
  sorry

#eval alternating_odd_sum  -- To check the result

end alternating_odd_sum_equals_21_l3925_392583


namespace not_perfect_square_8p_plus_1_l3925_392550

theorem not_perfect_square_8p_plus_1 (p : ℕ) (h_prime : Nat.Prime p) (h_gt_3 : p > 3) :
  ¬ ∃ n : ℕ, 8 * p + 1 = (2 * n + 1)^2 := by
sorry

end not_perfect_square_8p_plus_1_l3925_392550


namespace factorization_x_squared_minus_xy_l3925_392568

theorem factorization_x_squared_minus_xy (x y : ℝ) : x^2 - x*y = x*(x - y) := by
  sorry

end factorization_x_squared_minus_xy_l3925_392568


namespace lara_likes_one_last_digit_l3925_392562

theorem lara_likes_one_last_digit :
  ∃! d : ℕ, d < 10 ∧ ∀ n : ℕ, (n % 3 = 0 ∧ n % 5 = 0) → n % 10 = d :=
sorry

end lara_likes_one_last_digit_l3925_392562


namespace parking_lot_wheels_l3925_392505

/-- Represents the number of wheels for each vehicle type -/
structure VehicleWheels where
  car : Nat
  bike : Nat
  truck : Nat
  bus : Nat

/-- Represents the count of each vehicle type in the parking lot -/
structure VehicleCount where
  cars : Nat
  bikes : Nat
  trucks : Nat
  buses : Nat

/-- Calculates the total number of wheels in the parking lot -/
def totalWheels (wheels : VehicleWheels) (count : VehicleCount) : Nat :=
  wheels.car * count.cars +
  wheels.bike * count.bikes +
  wheels.truck * count.trucks +
  wheels.bus * count.buses

/-- Theorem: The total number of wheels in the parking lot is 156 -/
theorem parking_lot_wheels :
  let wheels : VehicleWheels := ⟨4, 2, 8, 6⟩
  let count : VehicleCount := ⟨14, 10, 7, 4⟩
  totalWheels wheels count = 156 := by
  sorry

#check parking_lot_wheels

end parking_lot_wheels_l3925_392505


namespace total_erasers_l3925_392500

/-- Given an initial number of erasers and a number of erasers added, 
    the total number of erasers is equal to the sum of the initial number and the added number. -/
theorem total_erasers (initial_erasers added_erasers : ℕ) :
  initial_erasers + added_erasers = initial_erasers + added_erasers :=
by sorry

end total_erasers_l3925_392500


namespace paco_sweet_cookies_left_l3925_392599

/-- The number of sweet cookies Paco has left after eating some -/
def sweet_cookies_left (initial : ℕ) (eaten : ℕ) : ℕ :=
  initial - eaten

/-- Theorem: Paco has 19 sweet cookies left -/
theorem paco_sweet_cookies_left : 
  sweet_cookies_left 34 15 = 19 := by
  sorry

end paco_sweet_cookies_left_l3925_392599


namespace polynomial_simplification_l3925_392538

theorem polynomial_simplification (x : ℝ) :
  (14 * x^12 + 8 * x^9 + 3 * x^8) + (2 * x^14 - x^12 + 2 * x^9 + 5 * x^5 + 7 * x^2 + 6) =
  2 * x^14 + 13 * x^12 + 10 * x^9 + 3 * x^8 + 5 * x^5 + 7 * x^2 + 6 := by
  sorry

end polynomial_simplification_l3925_392538


namespace line_divides_area_in_half_l3925_392546

/-- A point in the 2D plane -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- The L-shaped region defined by its vertices -/
def LShapedRegion : List Point2D := [
  ⟨0, 0⟩, ⟨0, 4⟩, ⟨4, 4⟩, ⟨4, 2⟩, ⟨7, 2⟩, ⟨7, 0⟩
]

/-- Calculate the area of a polygon given its vertices -/
def polygonArea (vertices : List Point2D) : ℝ :=
  sorry

/-- Calculate the area of a polygon formed by the origin and a line intersecting the L-shaped region -/
def areaAboveLine (slope : ℝ) : ℝ :=
  sorry

/-- The theorem stating that the line with slope 1/9 divides the L-shaped region in half -/
theorem line_divides_area_in_half :
  let totalArea := polygonArea LShapedRegion
  let slope := 1 / 9
  areaAboveLine slope = totalArea / 2 := by
  sorry

end line_divides_area_in_half_l3925_392546


namespace train_length_problem_l3925_392520

/-- Given a bridge length, train speed, and time to pass over the bridge, 
    calculate the length of the train. -/
def train_length (bridge_length : ℝ) (train_speed : ℝ) (time_to_pass : ℝ) : ℝ :=
  train_speed * time_to_pass - bridge_length

/-- Theorem stating that under the given conditions, the train length is 400 meters. -/
theorem train_length_problem : 
  let bridge_length : ℝ := 2800
  let train_speed : ℝ := 800
  let time_to_pass : ℝ := 4
  train_length bridge_length train_speed time_to_pass = 400 := by
  sorry

end train_length_problem_l3925_392520


namespace special_hexagon_angle_sum_l3925_392545

/-- A hexagon with specific angle properties -/
structure SpecialHexagon where
  /-- Exterior angle measuring 40° -/
  ext_angle : ℝ
  /-- First interior angle measuring 45° -/
  int_angle1 : ℝ
  /-- Second interior angle measuring 80° -/
  int_angle2 : ℝ
  /-- Third interior angle -/
  int_angle3 : ℝ
  /-- Fourth interior angle -/
  int_angle4 : ℝ
  /-- Condition: Exterior angle is 40° -/
  h1 : ext_angle = 40
  /-- Condition: First interior angle is 45° -/
  h2 : int_angle1 = 45
  /-- Condition: Second interior angle is 80° -/
  h3 : int_angle2 = 80
  /-- Sum of interior angles of a hexagon is 720° -/
  h4 : int_angle1 + int_angle2 + int_angle3 + int_angle4 + (180 - ext_angle) + 90 = 720

/-- The sum of the third and fourth interior angles is 15° -/
theorem special_hexagon_angle_sum (h : SpecialHexagon) : h.int_angle3 + h.int_angle4 = 15 := by
  sorry

end special_hexagon_angle_sum_l3925_392545


namespace cubic_polynomials_constant_term_l3925_392516

/-- Given two cubic polynomials with specific root relationships, 
    this theorem states the possible values for the constant term of the first polynomial. -/
theorem cubic_polynomials_constant_term (c d : ℝ) (u v : ℝ) : 
  (∃ w : ℝ, u^3 + c*u + d = 0 ∧ v^3 + c*v + d = 0 ∧ w^3 + c*w + d = 0) →
  (∃ w : ℝ, (u+2)^3 + c*(u+2) + (d-120) = 0 ∧ 
            (v-5)^3 + c*(v-5) + (d-120) = 0 ∧ 
             w^3 + c*w + (d-120) = 0) →
  d = 396 ∨ d = 8 := by
sorry

end cubic_polynomials_constant_term_l3925_392516


namespace age_difference_l3925_392596

theorem age_difference (a b : ℕ) (h1 : b = 38) (h2 : a + 10 = 2 * (b - 10)) : a - b = 8 := by
  sorry

end age_difference_l3925_392596


namespace inequality_solution_l3925_392529

theorem inequality_solution (x : ℝ) :
  x + 1 > 0 →
  x + 1 - Real.sqrt (x + 1) ≠ 0 →
  (x^2 / ((x + 1 - Real.sqrt (x + 1))^2) < (x^2 + 3*x + 18) / (x + 1)^2) ↔ 
  (-1 < x ∧ x < 3) := by
sorry

end inequality_solution_l3925_392529


namespace orthogonal_vectors_x_value_l3925_392581

def vector_a (x : ℝ) : Fin 2 → ℝ := ![x, 2]
def vector_b : Fin 2 → ℝ := ![2, -1]

def orthogonal (u v : Fin 2 → ℝ) : Prop :=
  (u 0) * (v 0) + (u 1) * (v 1) = 0

theorem orthogonal_vectors_x_value :
  ∃ x : ℝ, orthogonal (vector_a x) vector_b ∧ x = 1 := by
  sorry

end orthogonal_vectors_x_value_l3925_392581


namespace shaded_area_in_circle_configuration_l3925_392548

/-- The area of the shaded region in a circle configuration --/
theorem shaded_area_in_circle_configuration (R : ℝ) (h : R = 9) :
  let r : ℝ := R / 2
  let larger_circle_area : ℝ := π * R^2
  let smaller_circle_area : ℝ := π * r^2
  let total_smaller_circles_area : ℝ := 3 * smaller_circle_area
  let shaded_area : ℝ := larger_circle_area - total_smaller_circles_area
  shaded_area = 20.25 * π :=
by sorry


end shaded_area_in_circle_configuration_l3925_392548


namespace gcd_of_specific_numbers_l3925_392588

theorem gcd_of_specific_numbers : 
  let m : ℕ := 55555555
  let n : ℕ := 111111111
  Nat.gcd m n = 1 := by sorry

end gcd_of_specific_numbers_l3925_392588


namespace expression_evaluation_l3925_392578

theorem expression_evaluation (x : ℝ) : 
  x * (x * (x * (3 - x) - 5) + 12) + 2 = -x^4 + 3*x^3 - 5*x^2 + 12*x + 2 := by
  sorry

end expression_evaluation_l3925_392578


namespace binomial_floor_divisibility_l3925_392597

theorem binomial_floor_divisibility (p n : ℕ) (hp : Nat.Prime p) :
  p ∣ (Nat.choose n p - n / p) := by
  sorry

end binomial_floor_divisibility_l3925_392597


namespace largest_divisor_of_even_square_diff_l3925_392569

theorem largest_divisor_of_even_square_diff (m n : ℤ) : 
  Even m → Even n → n < m → 
  (∃ (k : ℤ), ∀ (a b : ℤ), (Even a ∧ Even b ∧ b < a) → 
    k ∣ (a^2 - b^2) ∧ 
    (∀ (l : ℤ), (∀ (x y : ℤ), (Even x ∧ Even y ∧ y < x) → l ∣ (x^2 - y^2)) → l ≤ k)) → 
  (∃ (k : ℤ), ∀ (a b : ℤ), (Even a ∧ Even b ∧ b < a) → 
    k ∣ (a^2 - b^2) ∧ 
    (∀ (l : ℤ), (∀ (x y : ℤ), (Even x ∧ Even y ∧ y < x) → l ∣ (x^2 - y^2)) → l ≤ k)) ∧ 
  (∃ (k : ℤ), ∀ (a b : ℤ), (Even a ∧ Even b ∧ b < a) → 
    k ∣ (a^2 - b^2) ∧ 
    (∀ (l : ℤ), (∀ (x y : ℤ), (Even x ∧ Even y ∧ y < x) → l ∣ (x^2 - y^2)) → l ≤ k)) → k = 4 :=
by sorry


end largest_divisor_of_even_square_diff_l3925_392569


namespace right_triangle_hypotenuse_l3925_392556

/-- A right triangle with perimeter 60 and area 120 has a hypotenuse of length 26. -/
theorem right_triangle_hypotenuse : ∀ (a b c : ℝ),
  a > 0 → b > 0 → c > 0 →
  a^2 + b^2 = c^2 →
  a + b + c = 60 →
  (1/2) * a * b = 120 →
  c = 26 := by
  sorry

end right_triangle_hypotenuse_l3925_392556


namespace incorrect_average_calculation_l3925_392573

theorem incorrect_average_calculation (n : ℕ) (correct_num incorrect_num : ℚ) (correct_avg : ℚ) :
  n = 10 ∧ 
  correct_num = 86 ∧ 
  incorrect_num = 26 ∧ 
  correct_avg = 26 →
  (n * correct_avg - correct_num + incorrect_num) / n = 20 := by
  sorry

end incorrect_average_calculation_l3925_392573


namespace square_sum_equals_twenty_l3925_392559

theorem square_sum_equals_twenty (x y : ℝ) 
  (h1 : (x + y)^2 = 36) (h2 : x * y = 8) : x^2 + y^2 = 20 := by
  sorry

end square_sum_equals_twenty_l3925_392559


namespace sixth_root_square_equation_solution_l3925_392590

theorem sixth_root_square_equation_solution :
  ∃ (x : ℝ), x > 0 ∧ (((x * (x^4)^(1/2))^(1/6))^2 = 4) ∧ x = 4 := by
sorry

end sixth_root_square_equation_solution_l3925_392590


namespace not_decreasing_if_f0_lt_f4_l3925_392542

def IsDecreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, x < y → f x ≥ f y

theorem not_decreasing_if_f0_lt_f4 (f : ℝ → ℝ) (h : f 0 < f 4) : ¬ IsDecreasing f := by
  sorry

end not_decreasing_if_f0_lt_f4_l3925_392542


namespace money_division_l3925_392535

/-- The problem of dividing money among three children -/
theorem money_division (anusha babu esha : ℕ) : 
  12 * anusha = 8 * babu ∧ 
  8 * babu = 6 * esha ∧ 
  anusha = 84 → 
  anusha + babu + esha = 378 := by
  sorry


end money_division_l3925_392535


namespace tan_ratio_problem_l3925_392587

theorem tan_ratio_problem (x : Real) (h : Real.tan (x + π / 4) = 2) :
  (Real.tan x) / (Real.tan (2 * x)) = 4 / 9 := by
  sorry

end tan_ratio_problem_l3925_392587


namespace charge_account_interest_l3925_392586

/-- Calculates the total amount owed after one year with simple interest -/
def total_amount_owed (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal * (1 + rate * time)

/-- Proves that the total amount owed after one year is $38.15 -/
theorem charge_account_interest :
  let principal : ℝ := 35
  let rate : ℝ := 0.09
  let time : ℝ := 1
  total_amount_owed principal rate time = 38.15 := by
sorry

end charge_account_interest_l3925_392586


namespace tooth_extraction_cost_l3925_392577

def cleaning_cost : ℕ := 70
def filling_cost : ℕ := 120
def root_canal_cost : ℕ := 400
def dental_crown_cost : ℕ := 600

def total_known_costs : ℕ := cleaning_cost + 3 * filling_cost + root_canal_cost + dental_crown_cost

def total_bill : ℕ := 9 * root_canal_cost

theorem tooth_extraction_cost : 
  total_bill - total_known_costs = 2170 := by sorry

end tooth_extraction_cost_l3925_392577


namespace students_per_table_unchanged_l3925_392539

theorem students_per_table_unchanged (tables : ℝ) (initial_students_per_table : ℝ) 
  (h1 : tables = 34.0) 
  (h2 : initial_students_per_table = 6.0) : 
  (tables * initial_students_per_table) / tables = initial_students_per_table := by
  sorry

#check students_per_table_unchanged

end students_per_table_unchanged_l3925_392539


namespace number_of_divisors_3003_l3925_392567

theorem number_of_divisors_3003 : Nat.card (Nat.divisors 3003) = 16 := by
  sorry

end number_of_divisors_3003_l3925_392567


namespace angle_difference_range_l3925_392508

/-- Given an acute angle and the absolute difference between this angle and its supplementary angle -/
theorem angle_difference_range (x α : Real) : 
  (0 < x) → (x < 90) → (α = |180 - 2*x|) → (0 < α ∧ α < 180) := by sorry

end angle_difference_range_l3925_392508


namespace remainder_97_pow_25_mod_50_l3925_392564

theorem remainder_97_pow_25_mod_50 : 97^25 % 50 = 7 := by
  sorry

end remainder_97_pow_25_mod_50_l3925_392564


namespace intercept_sum_zero_line_equation_l3925_392574

/-- A line passing through a point with sum of intercepts equal to zero -/
structure InterceptSumZeroLine where
  /-- The slope of the line -/
  slope : ℝ
  /-- The y-intercept of the line -/
  y_intercept : ℝ
  /-- The line passes through the point (1,4) -/
  passes_through_point : slope + y_intercept = 4
  /-- The sum of x and y intercepts is zero -/
  sum_of_intercepts_zero : (- y_intercept / slope) + y_intercept = 0

/-- The equation of the line is either 4x-y=0 or x-y+3=0 -/
theorem intercept_sum_zero_line_equation (line : InterceptSumZeroLine) :
  (line.slope = 4 ∧ line.y_intercept = 0) ∨ (line.slope = 1 ∧ line.y_intercept = 3) :=
sorry

end intercept_sum_zero_line_equation_l3925_392574


namespace ellipse_equation_l3925_392557

/-- Given an ellipse C with equation x²/a² + y²/b² = 1, where a > b > 0, and the major axis is √2
    times the minor axis, if the line y = -x + 1 intersects the ellipse at points A and B such that
    the length of chord AB is 4√5/3, then the equation of the ellipse is x²/4 + y²/2 = 1. -/
theorem ellipse_equation (a b : ℝ) (h1 : a > b) (h2 : b > 0) 
    (h3 : a^2 = 2 * b^2) 
    (h4 : ∃ (x1 y1 x2 y2 : ℝ), 
      x1^2/a^2 + y1^2/b^2 = 1 ∧ 
      x2^2/a^2 + y2^2/b^2 = 1 ∧
      y1 = -x1 + 1 ∧ 
      y2 = -x2 + 1 ∧ 
      (x2 - x1)^2 + (y2 - y1)^2 = (4*Real.sqrt 5/3)^2) :
  ∀ x y : ℝ, x^2/4 + y^2/2 = 1 ↔ x^2/a^2 + y^2/b^2 = 1 :=
sorry

end ellipse_equation_l3925_392557


namespace distance_between_foci_l3925_392589

-- Define the ellipse equation
def ellipse_equation (x y : ℝ) : Prop :=
  Real.sqrt ((x - 4)^2 + (y - 5)^2) + Real.sqrt ((x + 6)^2 + (y - 9)^2) = 25

-- Define the foci
def focus1 : ℝ × ℝ := (4, 5)
def focus2 : ℝ × ℝ := (-6, 9)

-- Theorem: The distance between the foci of the ellipse is 2√29
theorem distance_between_foci :
  let (x1, y1) := focus1
  let (x2, y2) := focus2
  Real.sqrt ((x1 - x2)^2 + (y1 - y2)^2) = 2 * Real.sqrt 29 :=
by sorry

end distance_between_foci_l3925_392589


namespace weights_division_l3925_392553

theorem weights_division (n : ℕ) (h : n ≥ 3) :
  (∃ (a b c : Finset ℕ), a ∩ b = ∅ ∧ a ∩ c = ∅ ∧ b ∩ c = ∅ ∧
    a ∪ b ∪ c = Finset.range (n + 1) \ {0} ∧
    (a.sum id = b.sum id) ∧ (b.sum id = c.sum id)) ↔
  (∃ k : ℕ, (n = 3 * k + 2 ∨ n = 3 * k + 3) ∧ k > 0) :=
by sorry

end weights_division_l3925_392553


namespace train_cars_count_l3925_392576

theorem train_cars_count (cars_per_15s : ℕ) (total_time : ℕ) : 
  cars_per_15s = 10 → total_time = 210 → (total_time * cars_per_15s) / 15 = 140 := by
  sorry

end train_cars_count_l3925_392576


namespace difference_positive_inequality_l3925_392593

theorem difference_positive_inequality (x : ℝ) :
  (1 / 3 * x - x > 0) ↔ (-2 / 3 * x > 0) := by sorry

end difference_positive_inequality_l3925_392593


namespace inverse_proportion_constant_l3925_392526

/-- Given an inverse proportion function y = k/x passing through the point (-2, -3), prove that k = 6 -/
theorem inverse_proportion_constant (k : ℝ) : 
  (∃ f : ℝ → ℝ, (∀ x ≠ 0, f x = k / x) ∧ f (-2) = -3) → k = 6 := by
  sorry

end inverse_proportion_constant_l3925_392526


namespace parabola_coefficients_l3925_392561

/-- A parabola with equation y = ax^2 + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The vertex of a parabola -/
def vertex (p : Parabola) : ℝ × ℝ := sorry

/-- Whether a parabola has a vertical axis of symmetry -/
def has_vertical_axis_of_symmetry (p : Parabola) : Prop := sorry

/-- Whether a point (x, y) lies on the parabola -/
def contains_point (p : Parabola) (x y : ℝ) : Prop := 
  y = p.a * x^2 + p.b * x + p.c

theorem parabola_coefficients :
  ∀ p : Parabola,
  vertex p = (2, -1) →
  has_vertical_axis_of_symmetry p →
  contains_point p 0 7 →
  (p.a, p.b, p.c) = (2, -8, 7) := by
  sorry

end parabola_coefficients_l3925_392561


namespace negation_of_universal_proposition_l3925_392502

theorem negation_of_universal_proposition (P Q : Prop) :
  (P ↔ ∀ x : ℤ, x < 1) →
  (Q ↔ ∃ x : ℤ, x ≥ 1) →
  (¬P ↔ Q) := by
  sorry

end negation_of_universal_proposition_l3925_392502


namespace angle_bisector_length_l3925_392560

-- Define the triangle ABC
def triangle_ABC (A B C : ℝ × ℝ) : Prop :=
  dist A B = 5 ∧ dist B C = 7 ∧ dist A C = 8

-- Define the angle bisector BD
def is_angle_bisector (A B C D : ℝ × ℝ) : Prop :=
  ∃ (x y : ℝ), x / y = 5 / 7 ∧ x + y = 8 ∧ 
  dist A D = x ∧ dist D C = y

-- Main theorem
theorem angle_bisector_length 
  (A B C D : ℝ × ℝ) 
  (h1 : triangle_ABC A B C) 
  (h2 : is_angle_bisector A B C D) 
  (h3 : ∃ (k : ℝ), dist B D = k * Real.sqrt 3) : 
  ∃ (k : ℝ), dist B D = k * Real.sqrt 3 ∧ k = 5 / 3 :=
sorry

end angle_bisector_length_l3925_392560


namespace replacement_cost_100_movies_l3925_392543

/-- The cost to replace VHS movies with DVDs -/
def replacement_cost (num_movies : ℕ) (vhs_trade_value : ℚ) (dvd_cost : ℚ) : ℚ :=
  num_movies * dvd_cost - num_movies * vhs_trade_value

/-- Theorem: The cost to replace 100 VHS movies with DVDs is $800 -/
theorem replacement_cost_100_movies :
  replacement_cost 100 2 10 = 800 := by
  sorry

end replacement_cost_100_movies_l3925_392543


namespace function_passes_through_point_l3925_392504

theorem function_passes_through_point (a : ℝ) (ha : a > 0) (hna : a ≠ 1) :
  let f : ℝ → ℝ := fun x ↦ a^(x - 1)
  f 1 = 1 := by
  sorry

end function_passes_through_point_l3925_392504


namespace quadratic_equal_real_roots_l3925_392527

theorem quadratic_equal_real_roots
  (a c : ℝ)
  (h_disc_zero : (4 * Real.sqrt 2) ^ 2 - 4 * a * c = 0)
  : ∃ x : ℝ, (a * x ^ 2 - 4 * x * Real.sqrt 2 + c = 0) ∧
    (∀ y : ℝ, a * y ^ 2 - 4 * y * Real.sqrt 2 + c = 0 → y = x) :=
by sorry

end quadratic_equal_real_roots_l3925_392527


namespace quadratic_function_theorem_l3925_392575

/-- The quadratic function f(x) = ax^2 + bx -/
def f (a b : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x

/-- The function g(x) = f(x) - x -/
def g (a b : ℝ) (x : ℝ) : ℝ := f a b x - x

theorem quadratic_function_theorem (a b : ℝ) (ha : a ≠ 0) :
  (∀ x, f a b (1 - x) = f a b (1 + x)) →
  (∃! x, g a b x = 0) →
  (f a b = fun x ↦ -1/2 * x^2 + x) ∧
  (∀ x ∈ Set.Icc (-4 : ℝ) 0, f a b x ∈ Set.Icc (-12 : ℝ) 0) ∧
  (∀ y ∈ Set.Icc (-12 : ℝ) 0, ∃ x ∈ Set.Icc (-4 : ℝ) 0, f a b x = y) :=
sorry

end quadratic_function_theorem_l3925_392575


namespace sum_of_digits_511_base2_l3925_392533

/-- The sum of the digits in the base-2 representation of 511₁₀ is 9. -/
theorem sum_of_digits_511_base2 : 
  (List.range 9).sum = (Nat.digits 2 511).sum := by sorry

end sum_of_digits_511_base2_l3925_392533


namespace old_edition_pages_l3925_392513

theorem old_edition_pages (new_edition : ℕ) (old_edition : ℕ) 
  (h1 : new_edition = 450) 
  (h2 : new_edition = 2 * old_edition - 230) : old_edition = 340 := by
  sorry

end old_edition_pages_l3925_392513


namespace winnie_keeps_six_l3925_392525

/-- The number of cherry lollipops Winnie has -/
def cherry : ℕ := 60

/-- The number of wintergreen lollipops Winnie has -/
def wintergreen : ℕ := 135

/-- The number of grape lollipops Winnie has -/
def grape : ℕ := 5

/-- The number of shrimp cocktail lollipops Winnie has -/
def shrimp : ℕ := 250

/-- The number of Winnie's friends -/
def friends : ℕ := 12

/-- The total number of lollipops Winnie has -/
def total : ℕ := cherry + wintergreen + grape + shrimp

/-- The number of lollipops Winnie keeps for herself -/
def kept : ℕ := total % friends

theorem winnie_keeps_six : kept = 6 := by
  sorry

end winnie_keeps_six_l3925_392525


namespace unique_factorial_sum_l3925_392507

theorem unique_factorial_sum (n : ℕ) : 2 * n * n.factorial + n.factorial = 2520 ↔ n = 10 := by
  sorry

end unique_factorial_sum_l3925_392507


namespace rectangular_box_surface_area_l3925_392532

theorem rectangular_box_surface_area 
  (x y z : ℝ) 
  (h1 : 4 * x + 4 * y + 4 * z = 160) 
  (h2 : Real.sqrt (x^2 + y^2 + z^2) = 25) : 
  2 * (x * y + y * z + z * x) = 975 := by
  sorry

end rectangular_box_surface_area_l3925_392532


namespace units_digit_of_4_pow_8_cubed_l3925_392555

theorem units_digit_of_4_pow_8_cubed (n : ℕ) : n = 4^(8^3) → n % 10 = 6 := by
  sorry

end units_digit_of_4_pow_8_cubed_l3925_392555


namespace intercepted_line_with_midpoint_at_origin_l3925_392541

/-- Given two lines l₁ and l₂, prove that the line x + 6y = 0 is intercepted by both lines
    and has its midpoint at the origin. -/
theorem intercepted_line_with_midpoint_at_origin :
  let l₁ : ℝ → ℝ → Prop := λ x y ↦ 4*x + y + 6 = 0
  let l₂ : ℝ → ℝ → Prop := λ x y ↦ 3*x - 5*y - 6 = 0
  let intercepted_line : ℝ → ℝ → Prop := λ x y ↦ x + 6*y = 0
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    l₁ x₁ y₁ ∧ l₂ x₂ y₂ ∧
    intercepted_line x₁ y₁ ∧ intercepted_line x₂ y₂ ∧
    (x₁ + x₂) / 2 = 0 ∧ (y₁ + y₂) / 2 = 0 :=
by sorry

end intercepted_line_with_midpoint_at_origin_l3925_392541


namespace rhombus_area_l3925_392551

/-- The area of a rhombus with side length 4 and a 45-degree angle between adjacent sides is 8√2 -/
theorem rhombus_area (side : ℝ) (angle : ℝ) : 
  side = 4 → angle = π / 4 → 
  let area := side * side * Real.sin angle
  area = 8 * Real.sqrt 2 := by sorry

end rhombus_area_l3925_392551


namespace correct_sales_growth_equation_l3925_392534

/-- Represents the growth of new energy vehicle sales over two months -/
def sales_growth (initial_sales : ℝ) (final_sales : ℝ) (growth_rate : ℝ) : Prop :=
  initial_sales * (1 + growth_rate)^2 = final_sales

/-- Theorem stating that the given equation correctly represents the sales growth -/
theorem correct_sales_growth_equation :
  ∃ x : ℝ, sales_growth 33.2 54.6 x :=
sorry

end correct_sales_growth_equation_l3925_392534


namespace least_cubes_from_cuboid_l3925_392503

/-- Given a cuboidal block with dimensions 6 cm x 9 cm x 12 cm,
    prove that the least possible number of equal cubes that can be cut from this block is 24. -/
theorem least_cubes_from_cuboid (length width height : ℕ) 
  (h_length : length = 6)
  (h_width : width = 9)
  (h_height : height = 12) :
  (∃ (cube_side : ℕ), 
    cube_side > 0 ∧
    length % cube_side = 0 ∧
    width % cube_side = 0 ∧
    height % cube_side = 0 ∧
    (length * width * height) / (cube_side ^ 3) = 24 ∧
    ∀ (other_side : ℕ), other_side > cube_side →
      ¬(length % other_side = 0 ∧
        width % other_side = 0 ∧
        height % other_side = 0)) :=
by sorry

end least_cubes_from_cuboid_l3925_392503


namespace additional_blurays_is_six_l3925_392540

/-- Represents the movie collection and purchase scenario -/
structure MovieCollection where
  initialDVDRatio : Nat
  initialBluRayRatio : Nat
  newDVDRatio : Nat
  newBluRayRatio : Nat
  totalInitialMovies : Nat

/-- Calculates the number of additional Blu-ray movies purchased -/
def additionalBluRays (mc : MovieCollection) : Nat :=
  let initialX := mc.totalInitialMovies / (mc.initialDVDRatio + mc.initialBluRayRatio)
  let initialDVD := mc.initialDVDRatio * initialX
  let initialBluRay := mc.initialBluRayRatio * initialX
  ((initialDVD * mc.newBluRayRatio) - (initialBluRay * mc.newDVDRatio)) / mc.newDVDRatio

/-- Theorem stating that the number of additional Blu-ray movies purchased is 6 -/
theorem additional_blurays_is_six (mc : MovieCollection) 
  (h1 : mc.initialDVDRatio = 7)
  (h2 : mc.initialBluRayRatio = 2)
  (h3 : mc.newDVDRatio = 13)
  (h4 : mc.newBluRayRatio = 4)
  (h5 : mc.totalInitialMovies = 351) :
  additionalBluRays mc = 6 := by
  sorry

#eval additionalBluRays { initialDVDRatio := 7, initialBluRayRatio := 2, 
                          newDVDRatio := 13, newBluRayRatio := 4, 
                          totalInitialMovies := 351 }

end additional_blurays_is_six_l3925_392540


namespace sqrt_sum_squares_l3925_392522

theorem sqrt_sum_squares : Real.sqrt (3^2) + (Real.sqrt 2)^2 = 5 := by
  sorry

end sqrt_sum_squares_l3925_392522


namespace class_selection_ways_l3925_392566

def total_classes : ℕ := 10
def advanced_classes : ℕ := 6
def intro_classes : ℕ := 4
def classes_to_choose : ℕ := 5
def min_advanced : ℕ := 3

theorem class_selection_ways :
  (Nat.choose advanced_classes 3 * Nat.choose intro_classes 2) +
  (Nat.choose advanced_classes 4 * Nat.choose intro_classes 1) +
  (Nat.choose advanced_classes 5) = 186 := by
  sorry

end class_selection_ways_l3925_392566


namespace identity_element_is_negative_four_l3925_392582

-- Define the operation ⊕
def circplus (a b : ℝ) : ℝ := a + b + 4

-- Define the property of being an identity element for ⊕
def is_identity (e : ℝ) : Prop :=
  ∀ a : ℝ, circplus e a = a

-- Theorem statement
theorem identity_element_is_negative_four :
  ∃ e : ℝ, is_identity e ∧ e = -4 := by
  sorry

end identity_element_is_negative_four_l3925_392582


namespace brookes_social_studies_problems_l3925_392572

/-- Calculates the number of social studies problems in Brooke's homework -/
theorem brookes_social_studies_problems :
  ∀ (math_problems science_problems : ℕ)
    (math_time social_studies_time science_time total_time : ℚ),
  math_problems = 15 →
  science_problems = 10 →
  math_time = 2 →
  social_studies_time = 1/2 →
  science_time = 3/2 →
  total_time = 48 →
  ∃ (social_studies_problems : ℕ),
    social_studies_problems = 6 ∧
    (math_problems : ℚ) * math_time +
    (social_studies_problems : ℚ) * social_studies_time +
    (science_problems : ℚ) * science_time = total_time :=
by sorry

end brookes_social_studies_problems_l3925_392572


namespace difference_equals_1011_l3925_392511

/-- The sum of consecutive odd numbers from 1 to 2021 -/
def sum_odd : ℕ := (2021 + 1) / 2 ^ 2

/-- The sum of consecutive even numbers from 2 to 2020 -/
def sum_even : ℕ := (2020 / 2) * (2020 / 2 + 1)

/-- The difference between the sum of odd numbers and the sum of even numbers -/
def difference : ℕ := sum_odd - sum_even

theorem difference_equals_1011 : difference = 1011 := by
  sorry

end difference_equals_1011_l3925_392511


namespace square_area_ratio_l3925_392518

theorem square_area_ratio (x : ℝ) (h : x > 0) :
  (x^2) / ((3*x)^2) = 1/9 := by
  sorry

end square_area_ratio_l3925_392518


namespace variance_range_best_for_stability_l3925_392565

/-- Represents a set of exam scores -/
def ExamScores := List ℝ

/-- Calculates the variance of a list of numbers -/
def variance (scores : ExamScores) : ℝ := sorry

/-- Calculates the range of a list of numbers -/
def range (scores : ExamScores) : ℝ := sorry

/-- Calculates the mean of a list of numbers -/
def mean (scores : ExamScores) : ℝ := sorry

/-- Calculates the median of a list of numbers -/
def median (scores : ExamScores) : ℝ := sorry

/-- Calculates the mode of a list of numbers -/
def mode (scores : ExamScores) : ℝ := sorry

/-- Measures how well a statistic represents the stability of scores -/
def stabilityMeasure (f : ExamScores → ℝ) : ℝ := sorry

theorem variance_range_best_for_stability (scores : ExamScores) 
  (h : scores.length = 5) :
  (stabilityMeasure variance > stabilityMeasure mean) ∧
  (stabilityMeasure variance > stabilityMeasure median) ∧
  (stabilityMeasure variance > stabilityMeasure mode) ∧
  (stabilityMeasure range > stabilityMeasure mean) ∧
  (stabilityMeasure range > stabilityMeasure median) ∧
  (stabilityMeasure range > stabilityMeasure mode) := by
  sorry

end variance_range_best_for_stability_l3925_392565


namespace complement_of_union_is_singleton_one_l3925_392554

-- Define the universal set I
def I : Set Nat := {0, 1, 2, 3}

-- Define set M
def M : Set Nat := {0, 2}

-- Define set N
def N : Set Nat := {0, 2, 3}

-- Theorem statement
theorem complement_of_union_is_singleton_one :
  (M ∪ N)ᶜ = {1} := by sorry

end complement_of_union_is_singleton_one_l3925_392554


namespace max_teams_in_tournament_l3925_392547

/-- The number of players in each team -/
def players_per_team : ℕ := 3

/-- The maximum number of games that can be played in the tournament -/
def max_games : ℕ := 150

/-- The number of games played between two teams -/
def games_between_teams : ℕ := players_per_team * players_per_team

/-- The maximum number of team pairs that can play within the game limit -/
def max_team_pairs : ℕ := max_games / games_between_teams

/-- The function to calculate the number of unique pairs of teams -/
def team_pairs (n : ℕ) : ℕ := n * (n - 1) / 2

/-- The theorem stating the maximum number of teams that can participate -/
theorem max_teams_in_tournament : 
  ∃ (n : ℕ), n > 0 ∧ team_pairs n ≤ max_team_pairs ∧ 
  ∀ (m : ℕ), m > n → team_pairs m > max_team_pairs :=
sorry

end max_teams_in_tournament_l3925_392547


namespace fraction_equation_solution_l3925_392515

theorem fraction_equation_solution (x y : ℝ) 
  (hx1 : x ≠ 0) (hx2 : x ≠ 2) (hy1 : y ≠ 0) (hy2 : y ≠ 3) :
  3 / x + 2 / y = 5 / 6 → x = 18 * y / (5 * y - 12) :=
by sorry

end fraction_equation_solution_l3925_392515


namespace T_is_four_sided_polygon_l3925_392544

-- Define the set T
def T (a : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | let x := p.1; let y := p.2;
    a ≤ x ∧ x ≤ 3*a ∧
    a ≤ y ∧ y ≤ 3*a ∧
    x + y ≥ 2*a ∧
    x + 2*a ≥ y ∧
    y + 2*a ≥ x ∧
    x + y ≤ 4*a}

-- Theorem statement
theorem T_is_four_sided_polygon (a : ℝ) (h : a > 0) :
  ∃ (v1 v2 v3 v4 : ℝ × ℝ),
    v1 ∈ T a ∧ v2 ∈ T a ∧ v3 ∈ T a ∧ v4 ∈ T a ∧
    (∀ p ∈ T a, p = v1 ∨ p = v2 ∨ p = v3 ∨ p = v4 ∨
      (∃ t : ℝ, 0 < t ∧ t < 1 ∧
        (p = (1 - t) • v1 + t • v2 ∨
         p = (1 - t) • v2 + t • v3 ∨
         p = (1 - t) • v3 + t • v4 ∨
         p = (1 - t) • v4 + t • v1))) :=
by sorry

end T_is_four_sided_polygon_l3925_392544


namespace lcm_of_ratio_and_sum_l3925_392579

theorem lcm_of_ratio_and_sum (a b : ℕ+) : 
  (a : ℚ) / b = 2 / 3 → a + b = 30 → Nat.lcm a b = 18 := by
  sorry

end lcm_of_ratio_and_sum_l3925_392579


namespace other_x_intercept_of_quadratic_l3925_392521

/-- A quadratic function f(x) = ax^2 + bx + c with vertex (4, 10) and one x-intercept at (1, 0) -/
def QuadraticFunction (a b c : ℝ) : ℝ → ℝ := λ x ↦ a * x^2 + b * x + c

theorem other_x_intercept_of_quadratic 
  (a b c : ℝ) 
  (h_vertex : QuadraticFunction a b c 4 = 10 ∧ (∀ x, QuadraticFunction a b c x ≥ 10 ∨ QuadraticFunction a b c x ≤ 10))
  (h_intercept : QuadraticFunction a b c 1 = 0) :
  ∃ x, x ≠ 1 ∧ QuadraticFunction a b c x = 0 ∧ x = 7 := by
sorry

end other_x_intercept_of_quadratic_l3925_392521


namespace triangle_area_is_twelve_l3925_392501

/-- The area of a triangle formed by the x-axis, y-axis, and the line 3x + 2y = 12 -/
def triangleArea : ℝ := 12

/-- The equation of the line bounding the triangle -/
def lineEquation (x y : ℝ) : Prop := 3 * x + 2 * y = 12

theorem triangle_area_is_twelve :
  ∃ (x₁ y₁ x₂ y₂ : ℝ),
    x₁ > 0 ∧ y₂ > 0 ∧
    lineEquation x₁ 0 ∧
    lineEquation 0 y₂ ∧
    (1/2 : ℝ) * x₁ * y₂ = triangleArea :=
by sorry

end triangle_area_is_twelve_l3925_392501


namespace algebraic_expression_symmetry_l3925_392595

theorem algebraic_expression_symmetry (a b : ℝ) : 
  (a * 3^3 + b * 3 - 5 = 20) → (a * (-3)^3 + b * (-3) - 5 = -30) := by
  sorry

end algebraic_expression_symmetry_l3925_392595


namespace range_of_a_l3925_392558

theorem range_of_a (a : ℝ) : (∀ x > 0, 4 * a > x^2 - x^3) → a > 1/27 := by sorry

end range_of_a_l3925_392558


namespace carltons_outfits_l3925_392570

/-- The number of unique outfit combinations for Carlton -/
def unique_outfit_combinations (button_up_shirts : ℕ) : ℕ :=
  let sweater_vests := 3 * button_up_shirts
  let ties := 2 * sweater_vests
  let shoes := 4 * ties
  let socks := 6 * shoes
  button_up_shirts * sweater_vests * ties * shoes * socks

/-- Theorem stating that Carlton's unique outfit combinations equal 77,760,000 -/
theorem carltons_outfits :
  unique_outfit_combinations 5 = 77760000 := by
  sorry

end carltons_outfits_l3925_392570


namespace sum_mod_nine_l3925_392506

theorem sum_mod_nine : 
  (2 + 33 + 444 + 5555 + 66666 + 777777 + 8888888 + 99999999) % 9 = 6 := by
  sorry

end sum_mod_nine_l3925_392506


namespace improved_running_distance_l3925_392523

/-- Proves that if a person can run 40 yards in 5 seconds and improves their speed by 40%, 
    they can run 112 yards in 10 seconds. -/
theorem improved_running_distance 
  (initial_distance : ℝ) 
  (initial_time : ℝ) 
  (improvement_percentage : ℝ) 
  (new_time : ℝ) :
  initial_distance = 40 ∧ 
  initial_time = 5 ∧ 
  improvement_percentage = 40 ∧ 
  new_time = 10 →
  (initial_distance + initial_distance * (improvement_percentage / 100)) * (new_time / initial_time) = 112 :=
by sorry

end improved_running_distance_l3925_392523


namespace arabella_dance_time_l3925_392592

/-- The time Arabella spends learning three dance steps -/
def dance_learning_time (first_step : ℕ) : ℕ :=
  let second_step := first_step / 2
  let third_step := first_step + second_step
  first_step + second_step + third_step

/-- Proof that Arabella spends 90 minutes learning three dance steps -/
theorem arabella_dance_time :
  dance_learning_time 30 = 90 := by
  sorry

end arabella_dance_time_l3925_392592
