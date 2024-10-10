import Mathlib

namespace linear_regression_average_increase_l3259_325974

/- Define a linear regression model -/
def LinearRegression (x y : ℝ → ℝ) (a b : ℝ) :=
  ∀ t, y t = b * x t + a

/- Define the average increase in y when x increases by 1 unit -/
def AverageIncrease (x y : ℝ → ℝ) (b : ℝ) :=
  ∀ t, y (t + 1) - y t = b

/- Theorem: In a linear regression model, when x increases by 1 unit,
   y increases by b units on average -/
theorem linear_regression_average_increase
  (x y : ℝ → ℝ) (a b : ℝ)
  (h : LinearRegression x y a b) :
  AverageIncrease x y b :=
by sorry

end linear_regression_average_increase_l3259_325974


namespace total_entertainment_hours_l3259_325961

/-- Represents the hours spent on an activity for each day of the week -/
structure WeeklyHours :=
  (monday : ℕ) (tuesday : ℕ) (wednesday : ℕ) (thursday : ℕ)
  (friday : ℕ) (saturday : ℕ) (sunday : ℕ)

/-- Calculates the total hours spent on an activity throughout the week -/
def totalHours (hours : WeeklyHours) : ℕ :=
  hours.monday + hours.tuesday + hours.wednesday + hours.thursday +
  hours.friday + hours.saturday + hours.sunday

/-- Haley's TV watching hours -/
def tvHours : WeeklyHours :=
  { monday := 0, tuesday := 2, wednesday := 0, thursday := 4,
    friday := 0, saturday := 6, sunday := 3 }

/-- Haley's video game playing hours -/
def gameHours : WeeklyHours :=
  { monday := 3, tuesday := 0, wednesday := 5, thursday := 0,
    friday := 1, saturday := 0, sunday := 0 }

theorem total_entertainment_hours :
  totalHours tvHours + totalHours gameHours = 24 := by
  sorry

end total_entertainment_hours_l3259_325961


namespace marks_garden_l3259_325900

theorem marks_garden (yellow purple green : ℕ) : 
  purple = yellow + (yellow * 4 / 5) →
  green = (yellow + purple) / 4 →
  yellow + purple + green = 35 →
  yellow = 10 := by
sorry

end marks_garden_l3259_325900


namespace lake_distance_difference_l3259_325912

/-- The difference between the circumference and diameter of a circular lake -/
theorem lake_distance_difference (diameter : ℝ) (pi : ℝ) 
  (h1 : diameter = 2)
  (h2 : pi = 3.14) : 
  2 * pi * (diameter / 2) - diameter = 4.28 := by
  sorry

end lake_distance_difference_l3259_325912


namespace max_value_of_z_l3259_325914

-- Define the system of inequalities
def system (x y : ℝ) : Prop :=
  x + y ≤ 4 ∧ y - 2*x + 2 ≤ 0 ∧ y ≥ 0

-- Define z as a function of x and y
def z (x y : ℝ) : ℝ := x + 2*y

-- Theorem statement
theorem max_value_of_z :
  ∃ (x y : ℝ), system x y ∧ z x y = 6 ∧
  ∀ (x' y' : ℝ), system x' y' → z x' y' ≤ 6 := by
sorry

end max_value_of_z_l3259_325914


namespace coefficient_x4_l3259_325967

/-- The coefficient of x^4 in the simplified form of 5(x^4 - 3x^2) + 3(2x^3 - x^4 + 4x^6) - (6x^2 - 2x^4) is 4 -/
theorem coefficient_x4 (x : ℝ) : 
  let expr := 5*(x^4 - 3*x^2) + 3*(2*x^3 - x^4 + 4*x^6) - (6*x^2 - 2*x^4)
  ∃ (a b c d e : ℝ), expr = 4*x^4 + a*x^6 + b*x^3 + c*x^2 + d*x + e :=
by sorry

end coefficient_x4_l3259_325967


namespace pythagorean_triple_identification_l3259_325976

def is_pythagorean_triple (a b c : ℕ) : Prop :=
  a * a + b * b = c * c

theorem pythagorean_triple_identification :
  ¬(is_pythagorean_triple 3 4 5) ∧
  ¬(is_pythagorean_triple 3 4 7) ∧
  ¬(is_pythagorean_triple 0 1 1) ∧
  is_pythagorean_triple 9 12 15 :=
by sorry

end pythagorean_triple_identification_l3259_325976


namespace trouser_sale_price_l3259_325969

theorem trouser_sale_price 
  (original_price : ℝ) 
  (discount_percentage : ℝ) 
  (h1 : original_price = 100)
  (h2 : discount_percentage = 80) : 
  original_price * (1 - discount_percentage / 100) = 20 := by
  sorry

end trouser_sale_price_l3259_325969


namespace square_of_binomial_l3259_325999

theorem square_of_binomial (x : ℝ) : (7 - Real.sqrt (x^2 - 33))^2 = x^2 - 14 * Real.sqrt (x^2 - 33) + 16 := by
  sorry

end square_of_binomial_l3259_325999


namespace observation_mean_invariance_l3259_325937

theorem observation_mean_invariance (n : ℕ) (h : n > 0) :
  let original_mean : ℚ := 200
  let decrement : ℚ := 6
  let new_mean : ℚ := 194
  n * original_mean - n * decrement = n * new_mean :=
by
  sorry

end observation_mean_invariance_l3259_325937


namespace average_milk_per_container_l3259_325903

-- Define the number of containers and their respective capacities
def containers_1_5 : ℕ := 6
def containers_0_67 : ℕ := 4
def containers_0_875 : ℕ := 5
def containers_2_33 : ℕ := 3
def containers_1_25 : ℕ := 2

def capacity_1_5 : ℚ := 3/2
def capacity_0_67 : ℚ := 67/100
def capacity_0_875 : ℚ := 875/1000
def capacity_2_33 : ℚ := 233/100
def capacity_1_25 : ℚ := 5/4

-- Define the total number of containers
def total_containers : ℕ := containers_1_5 + containers_0_67 + containers_0_875 + containers_2_33 + containers_1_25

-- Define the total amount of milk sold
def total_milk : ℚ := containers_1_5 * capacity_1_5 + containers_0_67 * capacity_0_67 + 
                      containers_0_875 * capacity_0_875 + containers_2_33 * capacity_2_33 + 
                      containers_1_25 * capacity_1_25

-- Theorem: The average amount of milk sold per container is 1.27725 liters
theorem average_milk_per_container : total_milk / total_containers = 127725 / 100000 := by
  sorry

end average_milk_per_container_l3259_325903


namespace quadratic_roots_sum_abs_l3259_325919

theorem quadratic_roots_sum_abs (p : ℝ) : 
  (∃ x y : ℝ, x^2 + p*x - 6 = 0 ∧ y^2 + p*y - 6 = 0 ∧ |x| + |y| = 5) → 
  (p = 1 ∨ p = -1) := by
  sorry

end quadratic_roots_sum_abs_l3259_325919


namespace astronaut_revolutions_l3259_325987

/-- Represents a circle with a given radius -/
structure Circle where
  radius : ℝ

/-- Represents the configuration of the three circles -/
structure CircleConfiguration where
  c₁ : Circle
  c₂ : Circle
  c₃ : Circle
  n : ℕ

/-- Defines the conditions of the problem -/
def ValidConfiguration (config : CircleConfiguration) : Prop :=
  config.n > 2 ∧
  config.c₁.radius = config.n * config.c₃.radius ∧
  config.c₂.radius = 2 * config.c₃.radius

/-- Calculates the number of revolutions of c₃ relative to the ground -/
noncomputable def revolutions (config : CircleConfiguration) : ℝ :=
  config.n - 1

/-- The main theorem to be proved -/
theorem astronaut_revolutions 
  (config : CircleConfiguration) 
  (h : ValidConfiguration config) :
  revolutions config = config.n - 1 := by
  sorry

end astronaut_revolutions_l3259_325987


namespace rectangular_solid_surface_area_l3259_325980

/-- A rectangular solid with prime edge lengths and volume 385 has surface area 334 -/
theorem rectangular_solid_surface_area : 
  ∀ (l w h : ℕ), 
    Prime l → Prime w → Prime h →
    l * w * h = 385 →
    2 * (l * w + l * h + w * h) = 334 := by
  sorry

end rectangular_solid_surface_area_l3259_325980


namespace zachary_crunches_count_l3259_325951

/-- The number of push-ups and crunches done by David and Zachary -/
def gym_class (david_pushups david_crunches zachary_pushups zachary_crunches : ℕ) : Prop :=
  (david_pushups = zachary_pushups + 40) ∧ 
  (zachary_crunches = david_crunches + 17) ∧
  (david_crunches = 45) ∧
  (zachary_pushups = 34)

theorem zachary_crunches_count :
  ∀ (david_pushups david_crunches zachary_pushups zachary_crunches : ℕ),
  gym_class david_pushups david_crunches zachary_pushups zachary_crunches →
  zachary_crunches = 62 :=
by
  sorry

end zachary_crunches_count_l3259_325951


namespace one_correct_proposition_l3259_325908

theorem one_correct_proposition : 
  (∃! n : Nat, n = 1 ∧ 
    ((∀ a b : ℝ, a > abs b → a^2 > b^2) ∧ 
     ¬(∀ a b c d : ℝ, a > b ∧ c > d → a - c > b - d) ∧
     ¬(∀ a b c d : ℝ, a > b ∧ c > d → a * c > b * d) ∧
     ¬(∀ a b c : ℝ, a > b ∧ b > 0 → c / a > c / b))) :=
by sorry

end one_correct_proposition_l3259_325908


namespace exists_special_sequence_l3259_325994

/-- A sequence of natural numbers -/
def IncreasingSequence := ℕ → ℕ

/-- Property that the sequence is strictly increasing -/
def IsStrictlyIncreasing (a : IncreasingSequence) : Prop :=
  a 0 = 0 ∧ ∀ n : ℕ, a (n + 1) > a n

/-- Property that every natural number is the sum of two sequence terms -/
def HasAllSums (a : IncreasingSequence) : Prop :=
  ∀ k : ℕ, ∃ i j : ℕ, k = a i + a j

/-- Property that each term is greater than n²/16 -/
def SatisfiesLowerBound (a : IncreasingSequence) : Prop :=
  ∀ n : ℕ, n > 0 → a n > (n^2 : ℚ) / 16

/-- The main theorem stating the existence of a sequence satisfying all conditions -/
theorem exists_special_sequence :
  ∃ a : IncreasingSequence, 
    IsStrictlyIncreasing a ∧ 
    HasAllSums a ∧ 
    SatisfiesLowerBound a := by
  sorry

end exists_special_sequence_l3259_325994


namespace bal_puzzle_l3259_325957

/-- Represents the possible meanings of the word "bal" -/
inductive BalMeaning
  | Yes
  | No

/-- Represents the possible types of inhabitants -/
inductive InhabitantType
  | Human
  | Zombie

/-- Represents the response to a yes/no question -/
def Response := BalMeaning

/-- Models the behavior of an inhabitant based on their type -/
def inhabitantBehavior (t : InhabitantType) (actual : BalMeaning) (response : Response) : Prop :=
  match t with
  | InhabitantType.Human => response = actual
  | InhabitantType.Zombie => response ≠ actual

/-- The main theorem capturing the essence of the problem -/
theorem bal_puzzle (response : Response) :
  (∀ meaning : BalMeaning, ∃ t : InhabitantType, inhabitantBehavior t meaning response) ∧
  (∃! t : InhabitantType, ∀ meaning : BalMeaning, inhabitantBehavior t meaning response) :=
by sorry

end bal_puzzle_l3259_325957


namespace min_n_for_geometric_sum_l3259_325964

theorem min_n_for_geometric_sum (n : ℕ) : 
  (∀ k : ℕ, k < n → (2^(k+1) - 1) ≤ 128) ∧ 
  (2^(n+1) - 1) > 128 → 
  n = 7 := by
sorry

end min_n_for_geometric_sum_l3259_325964


namespace sqrt_2x_minus_3_equals_1_l3259_325913

theorem sqrt_2x_minus_3_equals_1 (x : ℝ) (h : x = 2) : Real.sqrt (2 * x - 3) = 1 := by
  sorry

end sqrt_2x_minus_3_equals_1_l3259_325913


namespace ones_digit_product_seven_consecutive_l3259_325910

theorem ones_digit_product_seven_consecutive (k : ℕ) (h1 : k > 0) (h2 : k % 5 = 1) : 
  (((k * (k + 1) * (k + 2) * (k + 3) * (k + 4) * (k + 5) * (k + 6)) % 10) = 0) := by
  sorry

end ones_digit_product_seven_consecutive_l3259_325910


namespace earl_went_up_seven_floors_l3259_325932

/-- Represents the number of floors in the building -/
def total_floors : ℕ := 20

/-- Represents Earl's initial floor -/
def initial_floor : ℕ := 1

/-- Represents the number of floors Earl goes up initially -/
def first_up : ℕ := 5

/-- Represents the number of floors Earl goes down -/
def down : ℕ := 2

/-- Represents the number of floors Earl is away from the top after his final movement -/
def floors_from_top : ℕ := 9

/-- Calculates the number of floors Earl went up the second time -/
def second_up : ℕ := total_floors - floors_from_top - (initial_floor + first_up - down)

/-- Theorem stating that Earl went up 7 floors the second time -/
theorem earl_went_up_seven_floors : second_up = 7 := by sorry

end earl_went_up_seven_floors_l3259_325932


namespace mary_pokemon_cards_l3259_325918

theorem mary_pokemon_cards (x : ℕ) : 
  x + 23 - 6 = 56 → x = 39 := by
sorry

end mary_pokemon_cards_l3259_325918


namespace trivia_team_size_l3259_325905

theorem trivia_team_size (absent_members : ℕ) (points_per_member : ℕ) (total_points : ℕ) :
  absent_members = 7 →
  points_per_member = 5 →
  total_points = 35 →
  absent_members + (total_points / points_per_member) = 14 := by
sorry

end trivia_team_size_l3259_325905


namespace line_perp_two_planes_implies_parallel_l3259_325947

/-- A line in 3D space -/
structure Line3D where
  -- Add necessary fields here
  
/-- A plane in 3D space -/
structure Plane3D where
  -- Add necessary fields here

/-- Perpendicularity between a line and a plane -/
def perpendicular (l : Line3D) (p : Plane3D) : Prop :=
  sorry

/-- Parallelism between two planes -/
def parallel (p1 p2 : Plane3D) : Prop :=
  sorry

/-- Theorem: If a line is perpendicular to two different planes, then the planes are parallel -/
theorem line_perp_two_planes_implies_parallel 
  (l : Line3D) (α β : Plane3D) 
  (h_diff : α ≠ β) 
  (h_perp_α : perpendicular l α) 
  (h_perp_β : perpendicular l β) : 
  parallel α β :=
sorry

end line_perp_two_planes_implies_parallel_l3259_325947


namespace binomial_12_11_l3259_325977

theorem binomial_12_11 : Nat.choose 12 11 = 12 := by
  sorry

end binomial_12_11_l3259_325977


namespace no_common_root_l3259_325992

theorem no_common_root (a b c d : ℝ) (h : 0 < a ∧ a < b ∧ b < c ∧ c < d) :
  ∀ x : ℝ, (x^2 + b*x + c = 0) → (x^2 + a*x + d = 0) → False :=
by sorry

end no_common_root_l3259_325992


namespace digit_multiplication_sum_l3259_325990

theorem digit_multiplication_sum (p q : ℕ) : 
  p < 10 → q < 10 → (40 + p) * (10 * q + 5) = 190 → p + q = 4 := by
  sorry

end digit_multiplication_sum_l3259_325990


namespace line_through_point_parallel_to_line_l3259_325909

/-- A line passing through a point and parallel to another line -/
theorem line_through_point_parallel_to_line :
  let point : ℝ × ℝ := (2, 1)
  let parallel_line (x y : ℝ) := 2 * x - 3 * y + 1 = 0
  let target_line (x y : ℝ) := 2 * x - 3 * y - 1 = 0
  (∀ x y : ℝ, parallel_line x y ↔ y = 2/3 * x + 1/3) →
  (target_line point.1 point.2) ∧
  (∀ x y : ℝ, target_line x y ↔ y = 2/3 * x + 1/3) :=
by sorry

end line_through_point_parallel_to_line_l3259_325909


namespace bucket_capacity_reduction_l3259_325971

/-- Given a tank that requires different numbers of buckets to fill based on bucket capacity,
    this theorem proves the relationship between the original and reduced bucket capacities. -/
theorem bucket_capacity_reduction (original_buckets reduced_buckets : ℕ) 
  (h1 : original_buckets = 10)
  (h2 : reduced_buckets = 25)
  : (original_buckets : ℚ) / reduced_buckets = 2 / 5 :=
by sorry

end bucket_capacity_reduction_l3259_325971


namespace arithmetic_calculations_l3259_325988

theorem arithmetic_calculations :
  ((-24) - (-15) + (-1) + (-15) = -25) ∧
  ((-27) / (3/2) * (2/3) = -12) := by
  sorry

end arithmetic_calculations_l3259_325988


namespace craig_commission_l3259_325966

/-- Calculates the total commission for Craig's appliance sales. -/
def total_commission (
  refrigerator_base : ℝ)
  (refrigerator_rate : ℝ)
  (washing_machine_base : ℝ)
  (washing_machine_rate : ℝ)
  (oven_base : ℝ)
  (oven_rate : ℝ)
  (refrigerator_count : ℕ)
  (refrigerator_total_price : ℝ)
  (washing_machine_count : ℕ)
  (washing_machine_total_price : ℝ)
  (oven_count : ℕ)
  (oven_total_price : ℝ) : ℝ :=
  (refrigerator_count * (refrigerator_base + refrigerator_rate * refrigerator_total_price)) +
  (washing_machine_count * (washing_machine_base + washing_machine_rate * washing_machine_total_price)) +
  (oven_count * (oven_base + oven_rate * oven_total_price))

/-- Craig's total commission for the week is $5620.20. -/
theorem craig_commission :
  total_commission 75 0.08 50 0.10 60 0.12 3 5280 4 2140 5 4620 = 5620.20 := by
  sorry

end craig_commission_l3259_325966


namespace age_problem_l3259_325927

/-- Given three people a, b, and c, where:
  - a is two years older than b
  - b is twice as old as c
  - The total of their ages is 52
  Prove that b is 20 years old. -/
theorem age_problem (a b c : ℕ) 
  (h1 : a = b + 2)
  (h2 : b = 2 * c)
  (h3 : a + b + c = 52) :
  b = 20 := by
  sorry

end age_problem_l3259_325927


namespace vector_expression_l3259_325920

/-- Given vectors in ℝ², prove that c = 3a + 2b -/
theorem vector_expression (a b c : ℝ × ℝ) : 
  a = (1, -1) → b = (-1, 2) → c = (1, 1) → c = 3 • a + 2 • b := by sorry

end vector_expression_l3259_325920


namespace gathering_dancers_l3259_325991

theorem gathering_dancers (men : ℕ) (women : ℕ) : 
  men = 15 →
  men * 4 = women * 3 →
  women = 20 := by
sorry

end gathering_dancers_l3259_325991


namespace total_selection_methods_l3259_325972

-- Define the number of candidate schools
def total_schools : ℕ := 8

-- Define the number of schools to be selected
def selected_schools : ℕ := 4

-- Define the number of schools for session A
def schools_in_session_A : ℕ := 2

-- Define the number of remaining sessions (B and C)
def remaining_sessions : ℕ := 2

-- Theorem to prove
theorem total_selection_methods :
  (total_schools.choose selected_schools) *
  (selected_schools.choose schools_in_session_A) *
  (remaining_sessions!) = 840 := by
  sorry

end total_selection_methods_l3259_325972


namespace four_integers_problem_l3259_325921

theorem four_integers_problem (x y z u n : ℤ) :
  x + y + z + u = 36 →
  x + n = y - n ∧ y - n = z * n ∧ z * n = u / n →
  n = 1 ∧ x = 8 ∧ y = 10 ∧ z = 9 ∧ u = 9 :=
by sorry

end four_integers_problem_l3259_325921


namespace sum_of_digits_of_special_number_l3259_325901

/-- The least 6-digit number -/
def least_six_digit : ℕ := 100000

/-- Function to check if a number is 6-digit -/
def is_six_digit (n : ℕ) : Prop := n ≥ least_six_digit ∧ n < 1000000

/-- Function to calculate the sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sum_of_digits (n / 10)

/-- The main theorem -/
theorem sum_of_digits_of_special_number :
  ∃ n : ℕ,
    is_six_digit n ∧
    n % 4 = 2 ∧
    n % 610 = 2 ∧
    n % 15 = 2 ∧
    (∀ m : ℕ, m < n → ¬(is_six_digit m ∧ m % 4 = 2 ∧ m % 610 = 2 ∧ m % 15 = 2)) ∧
    sum_of_digits n = 17 :=
by sorry

end sum_of_digits_of_special_number_l3259_325901


namespace square_roots_problem_l3259_325981

theorem square_roots_problem (a m : ℝ) : 
  ((2 - m)^2 = a ∧ (2*m + 1)^2 = a) → a = 25 := by
  sorry

end square_roots_problem_l3259_325981


namespace primeDivisorsOf50FactorialIs15_l3259_325930

/-- The number of prime divisors of 50! -/
def primeDivisorsOf50Factorial : ℕ :=
  (List.range 51).filter (fun n => n.Prime && n > 1) |>.length

/-- Theorem: The number of prime divisors of 50! is 15 -/
theorem primeDivisorsOf50FactorialIs15 : primeDivisorsOf50Factorial = 15 := by
  sorry

end primeDivisorsOf50FactorialIs15_l3259_325930


namespace ball_hit_ground_time_l3259_325911

/-- The time at which a ball hits the ground when thrown upward -/
theorem ball_hit_ground_time : ∃ t : ℚ, t = 10/7 ∧ -4.9 * t^2 + 3.5 * t + 5 = 0 := by
  sorry

end ball_hit_ground_time_l3259_325911


namespace p_investment_l3259_325931

/-- Given that Q invested 15000 and the profit is divided in the ratio 5:1, prove that P's investment is 75000 --/
theorem p_investment (q_investment : ℕ) (profit_ratio_p profit_ratio_q : ℕ) :
  q_investment = 15000 →
  profit_ratio_p = 5 →
  profit_ratio_q = 1 →
  profit_ratio_p * q_investment = profit_ratio_q * 75000 :=
by sorry

end p_investment_l3259_325931


namespace festival_attendance_l3259_325959

theorem festival_attendance (total : ℕ) (day1 : ℕ) (day2 : ℕ) (day3 : ℕ) 
  (h_total : total = 2700)
  (h_day2 : day2 = day1 / 2)
  (h_day3 : day3 = 3 * day1)
  (h_sum : day1 + day2 + day3 = total) :
  day2 = 300 := by
sorry

end festival_attendance_l3259_325959


namespace max_sum_squared_distances_l3259_325985

theorem max_sum_squared_distances (z : ℂ) (h : Complex.abs (z - (3 - 3*I)) = 4) :
  (∃ (max_val : ℝ), max_val = 15 + 24 * (1.5 / Real.sqrt (1.5^2 + 1)) - 16 * (1 / Real.sqrt (1.5^2 + 1)) ∧
   ∀ (w : ℂ), Complex.abs (w - (3 - 3*I)) = 4 →
     Complex.abs (w - (2 + I))^2 + Complex.abs (w - (6 - 2*I))^2 ≤ max_val) :=
by sorry

end max_sum_squared_distances_l3259_325985


namespace tank_overflow_time_l3259_325993

/-- Represents the time it takes for a pipe to fill the tank -/
structure PipeRate where
  fill_time : ℝ
  fill_time_pos : fill_time > 0

/-- Represents the state of the tank filling process -/
structure TankFilling where
  overflow_time : ℝ
  pipe_a : PipeRate
  pipe_b : PipeRate
  pipe_b_close_time : ℝ

/-- The main theorem stating when the tank will overflow -/
theorem tank_overflow_time (tf : TankFilling) 
  (h1 : tf.pipe_a.fill_time = 2)
  (h2 : tf.pipe_b.fill_time = 1)
  (h3 : tf.pipe_b_close_time = tf.overflow_time - 0.5)
  (h4 : tf.overflow_time > 0) :
  tf.overflow_time = 1 := by
  sorry

#check tank_overflow_time

end tank_overflow_time_l3259_325993


namespace negative_integers_abs_leq_four_l3259_325946

theorem negative_integers_abs_leq_four :
  {x : ℤ | x < 0 ∧ |x| ≤ 4} = {-1, -2, -3, -4} := by sorry

end negative_integers_abs_leq_four_l3259_325946


namespace min_value_expression_l3259_325945

/-- The minimum value of a specific expression given certain constraints -/
theorem min_value_expression (x y z w : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (hw : w > 0) 
  (h_prod : x * y * z * w = 3) :
  x^2 + 4*x*y + 9*y^2 + 6*y*z + 8*z^2 + 3*x*w + 4*w^2 ≥ 81.25 ∧ 
  ∃ (x₀ y₀ z₀ w₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ z₀ > 0 ∧ w₀ > 0 ∧ 
    x₀ * y₀ * z₀ * w₀ = 3 ∧
    x₀^2 + 4*x₀*y₀ + 9*y₀^2 + 6*y₀*z₀ + 8*z₀^2 + 3*x₀*w₀ + 4*w₀^2 = 81.25 :=
by sorry

end min_value_expression_l3259_325945


namespace log_not_always_decreasing_l3259_325915

-- Define the logarithmic function
noncomputable def log (a : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log a

-- State the theorem
theorem log_not_always_decreasing :
  ¬ (∀ (a : ℝ), a > 0 → a ≠ 1 → 
    (∀ (x₁ x₂ : ℝ), x₁ > 0 → x₂ > 0 → x₁ < x₂ → log a x₁ > log a x₂)) :=
by sorry

end log_not_always_decreasing_l3259_325915


namespace coffee_table_price_l3259_325965

theorem coffee_table_price 
  (sofa_price : ℕ) 
  (armchair_price : ℕ) 
  (num_armchairs : ℕ) 
  (total_invoice : ℕ) 
  (h1 : sofa_price = 1250)
  (h2 : armchair_price = 425)
  (h3 : num_armchairs = 2)
  (h4 : total_invoice = 2430) :
  total_invoice - (sofa_price + num_armchairs * armchair_price) = 330 := by
sorry

end coffee_table_price_l3259_325965


namespace chenny_initial_candies_l3259_325982

/-- The number of friends Chenny has -/
def num_friends : ℕ := 7

/-- The number of candies each friend should receive -/
def candies_per_friend : ℕ := 2

/-- The number of additional candies Chenny needs to buy -/
def additional_candies : ℕ := 4

/-- Chenny's initial number of candies -/
def initial_candies : ℕ := num_friends * candies_per_friend - additional_candies

theorem chenny_initial_candies : initial_candies = 10 := by
  sorry

end chenny_initial_candies_l3259_325982


namespace angle_not_sharing_terminal_side_l3259_325923

def same_terminal_side (a b : ℝ) : Prop :=
  ∃ k : ℤ, a = b + k * 360

theorem angle_not_sharing_terminal_side :
  ¬(same_terminal_side 680 (-750)) ∧
  (same_terminal_side 330 (-750)) ∧
  (same_terminal_side (-30) (-750)) ∧
  (same_terminal_side (-1110) (-750)) :=
sorry

end angle_not_sharing_terminal_side_l3259_325923


namespace school_gender_difference_l3259_325954

theorem school_gender_difference (girls boys : ℕ) 
  (h1 : girls = 34) 
  (h2 : boys = 841) : 
  boys - girls = 807 := by
  sorry

end school_gender_difference_l3259_325954


namespace acute_angle_through_point_l3259_325944

theorem acute_angle_through_point (α : Real) : 
  α > 0 ∧ α < Real.pi/2 →
  (∃ (r : Real), r > 0 ∧ r * (Real.cos α) = Real.cos (40 * Real.pi/180) + 1 ∧ 
                            r * (Real.sin α) = Real.sin (40 * Real.pi/180)) →
  α = 20 * Real.pi/180 := by
sorry

end acute_angle_through_point_l3259_325944


namespace three_quarters_of_fifteen_fifths_minus_half_l3259_325996

theorem three_quarters_of_fifteen_fifths_minus_half (x : ℚ) : x = (3 / 4) * (15 / 5) - (1 / 2) → x = 7 / 4 := by
  sorry

end three_quarters_of_fifteen_fifths_minus_half_l3259_325996


namespace arg_z1_div_z2_l3259_325948

theorem arg_z1_div_z2 (z₁ z₂ : ℂ) (h1 : Complex.abs z₁ = 1) (h2 : Complex.abs z₂ = 1) (h3 : z₂ - z₁ = -1) :
  Complex.arg (z₁ / z₂) = π / 3 ∨ Complex.arg (z₁ / z₂) = 5 * π / 3 := by
  sorry

end arg_z1_div_z2_l3259_325948


namespace son_age_is_eleven_l3259_325989

/-- Represents the ages of a mother and son -/
structure FamilyAges where
  son : ℕ
  mother : ℕ

/-- The conditions of the age problem -/
def AgeProblemConditions (ages : FamilyAges) : Prop :=
  (ages.son + ages.mother = 55) ∧ 
  (ages.son - 3 + ages.mother - 3 = 49) ∧
  (ages.mother = 4 * ages.son)

/-- The theorem stating that under the given conditions, the son's age is 11 -/
theorem son_age_is_eleven (ages : FamilyAges) 
  (h : AgeProblemConditions ages) : ages.son = 11 := by
  sorry

end son_age_is_eleven_l3259_325989


namespace odd_multiple_of_nine_is_multiple_of_three_l3259_325916

theorem odd_multiple_of_nine_is_multiple_of_three :
  (∀ n : ℕ, 9 ∣ n → 3 ∣ n) →
  ∀ k : ℕ, Odd k → 9 ∣ k → 3 ∣ k :=
by
  sorry

end odd_multiple_of_nine_is_multiple_of_three_l3259_325916


namespace expression_simplification_l3259_325942

theorem expression_simplification : 120 * (120 - 12) - (120 * 120 - 12) = -1428 := by
  sorry

end expression_simplification_l3259_325942


namespace count_words_with_vowels_l3259_325984

/-- The number of letters in the alphabet -/
def alphabet_size : ℕ := 7

/-- The number of vowels in the alphabet -/
def vowel_count : ℕ := 2

/-- The length of the words we're considering -/
def word_length : ℕ := 5

/-- The number of 5-letter words with at least one vowel -/
def words_with_vowels : ℕ := alphabet_size ^ word_length - (alphabet_size - vowel_count) ^ word_length

theorem count_words_with_vowels :
  words_with_vowels = 13682 :=
sorry

end count_words_with_vowels_l3259_325984


namespace decimal_multiplication_l3259_325922

theorem decimal_multiplication (h : 213 * 16 = 3408) : 0.16 * 2.13 = 0.3408 := by
  sorry

end decimal_multiplication_l3259_325922


namespace max_sum_on_circle_l3259_325925

theorem max_sum_on_circle (x y : ℕ) : x^2 + y^2 = 64 → x + y ≤ 8 := by
  sorry

end max_sum_on_circle_l3259_325925


namespace problem_solution_l3259_325940

theorem problem_solution (x : ℝ) (h_pos : x > 0) 
  (h_eq : Real.sqrt (12 * x) * Real.sqrt (5 * x) * Real.sqrt (7 * x) * Real.sqrt (21 * x) = 42) : 
  x = Real.sqrt (21 / 47) := by
  sorry

end problem_solution_l3259_325940


namespace probability_of_winning_all_games_l3259_325978

def number_of_games : ℕ := 6
def probability_of_winning_single_game : ℚ := 3/5

theorem probability_of_winning_all_games :
  (probability_of_winning_single_game ^ number_of_games : ℚ) = 729/15625 := by
  sorry

end probability_of_winning_all_games_l3259_325978


namespace quadratic_even_iff_b_zero_l3259_325926

/-- A quadratic function f(x) = ax^2 + bx + c -/
def quadratic (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

/-- Definition of an even function -/
def is_even (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

theorem quadratic_even_iff_b_zero (a b c : ℝ) :
  is_even (quadratic a b c) ↔ b = 0 := by sorry

end quadratic_even_iff_b_zero_l3259_325926


namespace mn_positive_necessary_not_sufficient_necessity_not_sufficient_l3259_325956

/-- Defines an ellipse in terms of its equation coefficients -/
def is_ellipse (m n : ℝ) : Prop :=
  m > 0 ∧ n > 0 ∧ m ≠ n

/-- The main theorem stating that mn > 0 is necessary but not sufficient for mx^2 + ny^2 = 1 to be an ellipse -/
theorem mn_positive_necessary_not_sufficient :
  (∀ m n : ℝ, is_ellipse m n → m * n > 0) ∧
  (∃ m n : ℝ, m * n > 0 ∧ ¬is_ellipse m n) :=
sorry

/-- Proving necessity: if mx^2 + ny^2 = 1 is an ellipse, then mn > 0 -/
theorem necessity (m n : ℝ) (h : is_ellipse m n) : m * n > 0 :=
sorry

/-- Proving not sufficient: there exist m and n where mn > 0 but mx^2 + ny^2 = 1 is not an ellipse -/
theorem not_sufficient : ∃ m n : ℝ, m * n > 0 ∧ ¬is_ellipse m n :=
sorry

end mn_positive_necessary_not_sufficient_necessity_not_sufficient_l3259_325956


namespace equation_condition_l3259_325997

theorem equation_condition (a b c : ℕ+) (hb : b < 12) (hc : c < 12) :
  (12 * a + b) * (12 * a + c) = 144 * a * (a + 1) + b * c ↔ b + c = 12 := by
  sorry

end equation_condition_l3259_325997


namespace geometric_series_common_ratio_l3259_325917

/-- The common ratio of an infinite geometric series with given first term and sum -/
theorem geometric_series_common_ratio 
  (a : ℝ) 
  (S : ℝ) 
  (h1 : a = 400) 
  (h2 : S = 2500) 
  (h3 : a > 0) 
  (h4 : S > a) : 
  ∃ (r : ℝ), r = 21 / 25 ∧ S = a / (1 - r) := by
sorry

end geometric_series_common_ratio_l3259_325917


namespace simple_interest_problem_l3259_325928

theorem simple_interest_problem (P r : ℝ) 
  (h1 : P * (1 + 0.02 * r) = 600)
  (h2 : P * (1 + 0.07 * r) = 850) : 
  P = 500 := by
sorry

end simple_interest_problem_l3259_325928


namespace logan_hair_length_l3259_325933

/-- Given information about hair lengths of Kate, Emily, and Logan, prove Logan's hair length. -/
theorem logan_hair_length (kate_length emily_length logan_length : ℝ) 
  (h1 : kate_length = 7)
  (h2 : kate_length = emily_length / 2)
  (h3 : emily_length = logan_length + 6) :
  logan_length = 8 := by
  sorry

end logan_hair_length_l3259_325933


namespace cctv_systematic_sampling_group_size_l3259_325949

/-- Calculates the group size for systematic sampling -/
def systematicSamplingGroupSize (totalViewers : ℕ) (selectedViewers : ℕ) : ℕ :=
  totalViewers / selectedViewers

/-- Theorem: The group size for selecting 10 lucky viewers from 10000 viewers using systematic sampling is 1000 -/
theorem cctv_systematic_sampling_group_size :
  systematicSamplingGroupSize 10000 10 = 1000 := by
  sorry

end cctv_systematic_sampling_group_size_l3259_325949


namespace square_area_13m_l3259_325904

/-- The area of a square with side length 13 meters is 169 square meters. -/
theorem square_area_13m (side_length : ℝ) (h : side_length = 13) :
  side_length * side_length = 169 := by
  sorry

end square_area_13m_l3259_325904


namespace intersection_point_of_lines_l3259_325924

theorem intersection_point_of_lines (x y : ℚ) : 
  (8 * x - 5 * y = 10) ∧ (3 * x + 2 * y = 1) ↔ (x = 25/31 ∧ y = -22/31) :=
by sorry

end intersection_point_of_lines_l3259_325924


namespace plates_per_meal_l3259_325986

theorem plates_per_meal (guests : ℕ) (people : ℕ) (meals_per_day : ℕ) (days : ℕ) (total_plates : ℕ) :
  guests = 5 →
  people = guests + 1 →
  meals_per_day = 3 →
  days = 4 →
  total_plates = 144 →
  (total_plates / (people * meals_per_day * days) : ℚ) = 2 := by
  sorry

end plates_per_meal_l3259_325986


namespace g_of_3_eq_3_l3259_325983

/-- The function g is defined as g(x) = x^2 - 2x for all real x. -/
def g (x : ℝ) : ℝ := x^2 - 2*x

/-- Theorem: The value of g(3) is 3. -/
theorem g_of_3_eq_3 : g 3 = 3 := by
  sorry

end g_of_3_eq_3_l3259_325983


namespace initial_bucket_capacity_is_5_l3259_325962

/-- The capacity of the initially filled bucket -/
def initial_bucket_capacity : ℝ := 5

/-- The capacity of the small bucket -/
def small_bucket_capacity : ℝ := 3

/-- The capacity of the large bucket -/
def large_bucket_capacity : ℝ := 6

/-- The amount of additional water the large bucket can hold -/
def additional_capacity : ℝ := 4

theorem initial_bucket_capacity_is_5 :
  initial_bucket_capacity = small_bucket_capacity + (large_bucket_capacity - additional_capacity) :=
by
  sorry

#check initial_bucket_capacity_is_5

end initial_bucket_capacity_is_5_l3259_325962


namespace geometric_sequence_product_l3259_325939

-- Define a geometric sequence
def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

-- State the theorem
theorem geometric_sequence_product (a : ℕ → ℝ) :
  is_geometric_sequence a →
  a 3 * a 4 * a 5 = 3 →
  a 6 * a 7 * a 8 = 21 →
  a 9 * a 10 * a 11 = 147 := by
  sorry


end geometric_sequence_product_l3259_325939


namespace sophomore_count_l3259_325929

theorem sophomore_count (total : ℕ) (soph_percent : ℚ) (junior_percent : ℚ) :
  total = 36 →
  soph_percent = 1/5 →
  junior_percent = 3/20 →
  ∃ (soph junior : ℕ),
    soph + junior = total ∧
    soph_percent * soph = junior_percent * junior ∧
    soph = 16 :=
by sorry

end sophomore_count_l3259_325929


namespace smallest_root_property_l3259_325973

-- Define the quadratic equation
def quadratic_equation (x : ℝ) : Prop := x^2 - 9*x - 10 = 0

-- Define a as the smallest root
def a : ℝ := sorry

-- State the properties of a
axiom a_is_root : quadratic_equation a
axiom a_is_smallest : ∀ x, quadratic_equation x → a ≤ x

-- Theorem to prove
theorem smallest_root_property : a^4 - 909*a = 910 := by sorry

end smallest_root_property_l3259_325973


namespace regular_decagon_interior_angle_l3259_325975

/-- The measure of each interior angle of a regular decagon is 144 degrees. -/
theorem regular_decagon_interior_angle : ℝ := by
  -- Define the number of sides of a decagon
  let n : ℕ := 10

  -- Define the sum of interior angles formula
  let sum_of_interior_angles (sides : ℕ) : ℝ := (sides - 2) * 180

  -- Calculate the sum of interior angles for a decagon
  let total_angle_sum : ℝ := sum_of_interior_angles n

  -- Calculate the measure of one interior angle
  let interior_angle : ℝ := total_angle_sum / n

  -- Prove that the interior angle is 144 degrees
  sorry

end regular_decagon_interior_angle_l3259_325975


namespace quadratic_equation_equivalence_l3259_325968

theorem quadratic_equation_equivalence :
  ∀ x : ℝ, x^2 - 2*(3*x - 2) + (x + 1) = 0 ↔ x^2 - 5*x + 5 = 0 := by
  sorry

end quadratic_equation_equivalence_l3259_325968


namespace perpendicular_lines_from_perpendicular_planes_l3259_325943

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the perpendicular relation for planes and lines
variable (perp_plane : Plane → Plane → Prop)
variable (perp_line_plane : Line → Plane → Prop)
variable (perp_line : Line → Line → Prop)

-- State the theorem
theorem perpendicular_lines_from_perpendicular_planes
  (α β : Plane) (a b : Line)
  (h1 : perp_plane α β)
  (h2 : perp_line_plane a α)
  (h3 : perp_line_plane b β) :
  perp_line a b :=
sorry

end perpendicular_lines_from_perpendicular_planes_l3259_325943


namespace cube_edge_length_proof_l3259_325970

-- Define the vessel dimensions
def vessel_length : ℝ := 20
def vessel_width : ℝ := 15
def water_level_rise : ℝ := 3.3333333333333335

-- Define the cube's edge length
def cube_edge_length : ℝ := 10

-- Theorem statement
theorem cube_edge_length_proof :
  let vessel_base_area := vessel_length * vessel_width
  let water_volume_displaced := vessel_base_area * water_level_rise
  water_volume_displaced = cube_edge_length ^ 3 := by
  sorry

end cube_edge_length_proof_l3259_325970


namespace third_month_sales_l3259_325907

def sales_1 : ℕ := 6435
def sales_2 : ℕ := 6927
def sales_4 : ℕ := 7230
def sales_5 : ℕ := 6562
def sales_6 : ℕ := 6191
def average_sale : ℕ := 6700
def num_months : ℕ := 6

theorem third_month_sales :
  ∃ (sales_3 : ℕ),
    sales_3 = average_sale * num_months - (sales_1 + sales_2 + sales_4 + sales_5 + sales_6) ∧
    sales_3 = 6855 := by
  sorry

end third_month_sales_l3259_325907


namespace two_interviewers_passing_l3259_325935

def number_of_interviewers : ℕ := 5
def interviewers_to_choose : ℕ := 2

theorem two_interviewers_passing :
  Nat.choose number_of_interviewers interviewers_to_choose = 10 := by
  sorry

end two_interviewers_passing_l3259_325935


namespace union_complement_equals_l3259_325979

-- Define the universal set U
def U : Finset Nat := {1, 2, 3, 4, 5}

-- Define set M
def M : Finset Nat := {1, 4}

-- Define set N
def N : Finset Nat := {2, 5}

-- Theorem statement
theorem union_complement_equals : N ∪ (U \ M) = {2, 3, 5} := by
  sorry

end union_complement_equals_l3259_325979


namespace triangle_inequality_theorem_triangle_equality_theorem_l3259_325941

/-- A triangle with sides a, b, c and area S -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  S : ℝ
  positive_sides : 0 < a ∧ 0 < b ∧ 0 < c
  triangle_inequality : a + b > c ∧ b + c > a ∧ c + a > b
  area_positive : 0 < S

/-- The inequality holds for all triangles -/
theorem triangle_inequality_theorem (t : Triangle) :
  t.a^2 + t.b^2 + t.c^2 ≥ 4 * t.S * Real.sqrt 3 :=
sorry

/-- The equality holds if and only if the triangle is equilateral -/
theorem triangle_equality_theorem (t : Triangle) :
  t.a^2 + t.b^2 + t.c^2 = 4 * t.S * Real.sqrt 3 ↔ t.a = t.b ∧ t.b = t.c :=
sorry

end triangle_inequality_theorem_triangle_equality_theorem_l3259_325941


namespace common_volume_theorem_l3259_325953

/-- Represents a triangular pyramid with a point O on the segment connecting
    the vertex with the intersection point of base medians -/
structure TriangularPyramid where
  volume : ℝ
  ratio : ℝ

/-- Calculates the volume of the common part of the original pyramid
    and its symmetric counterpart with respect to point O -/
noncomputable def commonVolume (pyramid : TriangularPyramid) : ℝ :=
  if pyramid.ratio = 1 then 2 * pyramid.volume / 9
  else if pyramid.ratio = 3 then pyramid.volume / 2
  else if pyramid.ratio = 2 then 110 * pyramid.volume / 243
  else if pyramid.ratio = 4 then 12 * pyramid.volume / 25
  else 0  -- undefined for other ratios

theorem common_volume_theorem (pyramid : TriangularPyramid) :
  (pyramid.ratio = 1 → commonVolume pyramid = 2 * pyramid.volume / 9) ∧
  (pyramid.ratio = 3 → commonVolume pyramid = pyramid.volume / 2) ∧
  (pyramid.ratio = 2 → commonVolume pyramid = 110 * pyramid.volume / 243) ∧
  (pyramid.ratio = 4 → commonVolume pyramid = 12 * pyramid.volume / 25) :=
by sorry

end common_volume_theorem_l3259_325953


namespace pauls_savings_l3259_325955

/-- Paul's initial savings in dollars -/
def initial_savings : ℕ := sorry

/-- Cost of one toy in dollars -/
def toy_cost : ℕ := 5

/-- Number of toys Paul wants to buy -/
def num_toys : ℕ := 2

/-- Additional money Paul receives in dollars -/
def additional_money : ℕ := 7

theorem pauls_savings :
  initial_savings = 3 ∧
  initial_savings + additional_money = num_toys * toy_cost :=
by sorry

end pauls_savings_l3259_325955


namespace no_base_for_172_four_digit_odd_final_l3259_325963

theorem no_base_for_172_four_digit_odd_final (b : ℕ) : ¬ (
  (b ^ 3 ≤ 172 ∧ 172 < b ^ 4) ∧  -- four-digit number condition
  (172 % b % 2 = 1)              -- odd final digit condition
) := by
  sorry

end no_base_for_172_four_digit_odd_final_l3259_325963


namespace square_area_quadrupled_l3259_325938

theorem square_area_quadrupled (a : ℝ) (h : a > 0) :
  (2 * a)^2 = 4 * a^2 := by sorry

end square_area_quadrupled_l3259_325938


namespace multiply_106_94_l3259_325936

theorem multiply_106_94 : 106 * 94 = 9964 := by
  sorry

end multiply_106_94_l3259_325936


namespace inscribed_box_radius_l3259_325934

/-- Given a rectangular box Q inscribed in a sphere of radius r,
    if the surface area of Q is 672 and the sum of the lengths of its 12 edges is 168,
    then r = √273 -/
theorem inscribed_box_radius (a b c r : ℝ) : 
  a > 0 → b > 0 → c > 0 → r > 0 →
  2 * (a * b + b * c + a * c) = 672 →
  4 * (a + b + c) = 168 →
  (2 * r) ^ 2 = a ^ 2 + b ^ 2 + c ^ 2 →
  r = Real.sqrt 273 := by
  sorry

end inscribed_box_radius_l3259_325934


namespace negation_of_proposition_l3259_325906

theorem negation_of_proposition :
  (¬ ∀ x : ℝ, x > 0 → x^2 + x ≥ 0) ↔ (∃ x₀ : ℝ, x₀ > 0 ∧ x₀^2 + x₀ < 0) :=
by sorry

end negation_of_proposition_l3259_325906


namespace quadratic_equation_1_l3259_325995

theorem quadratic_equation_1 : ∃ x₁ x₂ : ℝ, x₁ = 6 ∧ x₂ = -1 ∧ x₁^2 - 5*x₁ - 6 = 0 ∧ x₂^2 - 5*x₂ - 6 = 0 := by
  sorry

end quadratic_equation_1_l3259_325995


namespace final_b_value_l3259_325960

def program_execution (a b c : Int) : Int :=
  let a' := b
  let b' := c
  b'

theorem final_b_value :
  ∀ (a b c : Int),
  a = 3 →
  b = -5 →
  c = 8 →
  program_execution a b c = 8 := by
  sorry

end final_b_value_l3259_325960


namespace subset_of_square_eq_self_l3259_325998

theorem subset_of_square_eq_self : {1} ⊆ {x : ℝ | x^2 = x} := by sorry

end subset_of_square_eq_self_l3259_325998


namespace smallest_slope_tangent_line_l3259_325952

-- Define the curve
def f (x : ℝ) : ℝ := x^3 + 3*x^2 + 6*x - 1

-- Define the derivative of the curve
def f' (x : ℝ) : ℝ := 3*x^2 + 6*x + 6

-- Theorem statement
theorem smallest_slope_tangent_line :
  ∃ (x₀ y₀ : ℝ),
    (∀ x, f' x₀ ≤ f' x) ∧
    y₀ = f x₀ ∧
    (∀ x y, y = f x → 3*x - y - 2 = 0 ∨ 3*x - y - 2 > 0) ∧
    3*x₀ - y₀ - 2 = 0 :=
sorry

end smallest_slope_tangent_line_l3259_325952


namespace initial_birds_count_l3259_325958

def birds_problem (initial_birds : ℕ) (landed_birds : ℕ) (total_birds : ℕ) : Prop :=
  initial_birds + landed_birds = total_birds

theorem initial_birds_count : ∃ (initial_birds : ℕ), 
  birds_problem initial_birds 8 20 ∧ initial_birds = 12 := by
  sorry

end initial_birds_count_l3259_325958


namespace negation_of_product_nonzero_implies_factors_nonzero_l3259_325902

theorem negation_of_product_nonzero_implies_factors_nonzero :
  (¬(∀ a b : ℝ, a * b ≠ 0 → a ≠ 0 ∧ b ≠ 0)) ↔
  (∀ a b : ℝ, a * b = 0 → a = 0 ∨ b = 0) :=
by sorry

end negation_of_product_nonzero_implies_factors_nonzero_l3259_325902


namespace cannot_form_square_l3259_325950

/-- Represents the number of sticks of each length -/
structure StickCounts where
  one_cm : Nat
  two_cm : Nat
  three_cm : Nat
  four_cm : Nat

/-- Calculates the total perimeter from the given stick counts -/
def totalPerimeter (counts : StickCounts) : Nat :=
  counts.one_cm * 1 + counts.two_cm * 2 + counts.three_cm * 3 + counts.four_cm * 4

/-- Checks if it's possible to form a square with the given stick counts -/
def canFormSquare (counts : StickCounts) : Prop :=
  ∃ (side : Nat), side > 0 ∧ 4 * side = totalPerimeter counts

/-- The given stick counts -/
def givenSticks : StickCounts :=
  { one_cm := 6
  , two_cm := 3
  , three_cm := 6
  , four_cm := 5
  }

/-- Theorem stating it's impossible to form a square with the given sticks -/
theorem cannot_form_square : ¬ canFormSquare givenSticks := by
  sorry

end cannot_form_square_l3259_325950
