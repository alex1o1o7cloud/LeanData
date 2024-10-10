import Mathlib

namespace find_k_l193_19396

theorem find_k (f g : ℝ → ℝ) (k : ℝ) : 
  (∀ x, f x = 7 * x^3 - 1/x + 5) →
  (∀ x, g x = x^3 - k) →
  f 3 - g 3 = 5 →
  k = -485/3 := by sorry

end find_k_l193_19396


namespace vidyas_age_l193_19389

theorem vidyas_age (vidya_age : ℕ) (mother_age : ℕ) : 
  mother_age = 3 * vidya_age + 5 →
  mother_age = 44 →
  vidya_age = 13 := by
sorry

end vidyas_age_l193_19389


namespace radians_to_degrees_l193_19335

theorem radians_to_degrees (π : ℝ) (h : π > 0) :
  (8 * π / 5) * (180 / π) = 288 := by
  sorry

end radians_to_degrees_l193_19335


namespace two_objects_ten_recipients_l193_19383

/-- The number of ways to distribute two distinct objects among a given number of recipients. -/
def distributionWays (recipients : ℕ) : ℕ := recipients * recipients

/-- Theorem: The number of ways to distribute two distinct objects among ten recipients is 100. -/
theorem two_objects_ten_recipients :
  distributionWays 10 = 100 := by
  sorry

end two_objects_ten_recipients_l193_19383


namespace investment_rate_proof_l193_19364

/-- Proves that the required interest rate for the remaining investment is 6.4% --/
theorem investment_rate_proof (total_investment : ℝ) (first_investment : ℝ) (second_investment : ℝ)
  (first_rate : ℝ) (second_rate : ℝ) (desired_income : ℝ)
  (h1 : total_investment = 10000)
  (h2 : first_investment = 4000)
  (h3 : second_investment = 3500)
  (h4 : first_rate = 0.05)
  (h5 : second_rate = 0.04)
  (h6 : desired_income = 500) :
  (desired_income - (first_investment * first_rate + second_investment * second_rate)) / 
  (total_investment - first_investment - second_investment) = 0.064 := by
sorry

end investment_rate_proof_l193_19364


namespace remainder_of_3_600_mod_17_l193_19363

theorem remainder_of_3_600_mod_17 : 3^600 % 17 = 9 := by
  sorry

end remainder_of_3_600_mod_17_l193_19363


namespace gcd_special_numbers_l193_19351

theorem gcd_special_numbers : Nat.gcd (2^2010 - 3) (2^2001 - 3) = 1533 := by
  sorry

end gcd_special_numbers_l193_19351


namespace fifteenth_thirty_seventh_215th_digit_l193_19381

def decimal_representation (n d : ℕ) : List ℕ := sorry

def nth_digit (n : ℕ) (l : List ℕ) : ℕ := sorry

theorem fifteenth_thirty_seventh_215th_digit :
  let rep := decimal_representation 15 37
  nth_digit 215 rep = 0 := by sorry

end fifteenth_thirty_seventh_215th_digit_l193_19381


namespace age_ratio_dan_james_l193_19399

theorem age_ratio_dan_james : 
  ∀ (dan_future_age james_age : ℕ),
    dan_future_age = 28 →
    james_age = 20 →
    ∃ (dan_age : ℕ),
      dan_age + 4 = dan_future_age ∧
      dan_age * 5 = james_age * 6 := by
sorry

end age_ratio_dan_james_l193_19399


namespace path_count_theorem_l193_19398

/-- Represents a grid with width and height -/
structure Grid :=
  (width : ℕ)
  (height : ℕ)

/-- Calculates the number of paths in a grid with the given constraints -/
def count_paths (g : Grid) : ℕ :=
  Nat.choose (g.width + g.height - 1) g.height -
  Nat.choose (g.width + g.height - 2) g.height +
  Nat.choose (g.width + g.height - 3) g.height

/-- The problem statement -/
theorem path_count_theorem (g : Grid) (h1 : g.width = 7) (h2 : g.height = 6) :
  count_paths g = 1254 := by
  sorry

end path_count_theorem_l193_19398


namespace cos_sin_75_product_equality_l193_19342

theorem cos_sin_75_product_equality : 
  (Real.cos (75 * π / 180) + Real.sin (75 * π / 180)) * 
  (Real.cos (75 * π / 180) - Real.sin (75 * π / 180)) = 
  - (Real.sqrt 3) / 2 := by sorry

end cos_sin_75_product_equality_l193_19342


namespace birds_on_fence_l193_19300

/-- Given an initial number of birds and a final total number of birds,
    calculate the number of birds that joined. -/
def birds_joined (initial : ℕ) (final : ℕ) : ℕ :=
  final - initial

/-- Theorem stating that given 2 initial birds and 6 final birds,
    the number of birds that joined is 4. -/
theorem birds_on_fence : birds_joined 2 6 = 4 := by
  sorry

end birds_on_fence_l193_19300


namespace total_pencils_specific_pencil_case_l193_19359

/-- Given an initial number of pencils and a number of pencils added, 
    the total number of pencils is equal to the sum of the initial number and the added number. -/
theorem total_pencils (initial : ℕ) (added : ℕ) : 
  initial + added = initial + added :=
by sorry

/-- In the specific case of 2 initial pencils and 3 added pencils, the total is 5. -/
theorem specific_pencil_case : 
  2 + 3 = 5 :=
by sorry

end total_pencils_specific_pencil_case_l193_19359


namespace complex_modulus_range_l193_19320

theorem complex_modulus_range (a : ℝ) (z : ℂ) (h1 : 0 < a) (h2 : a < 2) (h3 : z = a + Complex.I) :
  1 < Complex.abs z ∧ Complex.abs z < Real.sqrt 5 := by
  sorry

end complex_modulus_range_l193_19320


namespace roots_cubic_expression_l193_19336

theorem roots_cubic_expression (γ δ : ℝ) : 
  (γ^2 - 3*γ + 2 = 0) → 
  (δ^2 - 3*δ + 2 = 0) → 
  8*γ^3 - 6*δ^2 = 48 := by
sorry

end roots_cubic_expression_l193_19336


namespace largest_divisor_of_consecutive_odd_product_l193_19323

theorem largest_divisor_of_consecutive_odd_product (n : ℕ) (h : Even n) (h' : n > 0) :
  ∃ (k : ℕ), k = 15 ∧ 
  (∀ m : ℕ, m > k → ¬(m ∣ (n+1)*(n+3)*(n+5)*(n+7)*(n+9)*(n+11)*(n+13))) ∧
  (k ∣ (n+1)*(n+3)*(n+5)*(n+7)*(n+9)*(n+11)*(n+13)) :=
by sorry

end largest_divisor_of_consecutive_odd_product_l193_19323


namespace arithmetic_sequence_common_ratio_l193_19360

def arithmetic_sequence (a : ℕ → ℚ) : Prop :=
  ∃ (d : ℚ), ∀ (n : ℕ), a (n + 1) = a n + d

theorem arithmetic_sequence_common_ratio
  (a : ℕ → ℚ)
  (h_arithmetic : arithmetic_sequence a)
  (h_sum1 : a 1 + a 3 = 10)
  (h_sum2 : a 4 + a 6 = 5/4) :
  ∃ (q : ℚ), q = 1/2 ∧ ∀ (n : ℕ), a (n + 1) = a n * q :=
sorry

end arithmetic_sequence_common_ratio_l193_19360


namespace complex_cube_theorem_l193_19355

theorem complex_cube_theorem (z : ℂ) (h1 : Complex.abs (z - 2) = 2) (h2 : Complex.abs z = 2) : 
  z^3 = -8 := by sorry

end complex_cube_theorem_l193_19355


namespace polar_to_rectangular_l193_19332

theorem polar_to_rectangular :
  let r : ℝ := 3 * Real.sqrt 2
  let θ : ℝ := π / 4
  let x : ℝ := r * Real.cos θ
  let y : ℝ := r * Real.sin θ
  (x = 3 ∧ y = 3) := by sorry

end polar_to_rectangular_l193_19332


namespace consecutive_color_draws_probability_l193_19316

def blue_chips : ℕ := 4
def green_chips : ℕ := 3
def red_chips : ℕ := 5
def total_chips : ℕ := blue_chips + green_chips + red_chips

def probability_consecutive_color_draws : ℚ :=
  (Nat.factorial 3 * Nat.factorial blue_chips * Nat.factorial green_chips * Nat.factorial red_chips) /
  Nat.factorial total_chips

theorem consecutive_color_draws_probability :
  probability_consecutive_color_draws = 1 / 4620 :=
by sorry

end consecutive_color_draws_probability_l193_19316


namespace chord_length_of_perpendicular_bisector_l193_19302

/-- 
Given a circle with radius 15 units and a chord that is the perpendicular bisector of a radius,
prove that the length of this chord is 26 units.
-/
theorem chord_length_of_perpendicular_bisector (r : ℝ) (chord_length : ℝ) : 
  r = 15 → 
  chord_length = 2 * Real.sqrt (r^2 - (r/2)^2) → 
  chord_length = 26 := by
  sorry

end chord_length_of_perpendicular_bisector_l193_19302


namespace equation_solution_l193_19311

theorem equation_solution : ∃ x : ℝ, x ≠ 0 ∧ 9 - 3 / (1 / x) + 3 = 3 → x = 3 := by
  sorry

end equation_solution_l193_19311


namespace smallest_solution_quartic_equation_l193_19384

theorem smallest_solution_quartic_equation :
  ∃ (x : ℝ), x^4 - 34*x^2 + 225 = 0 ∧ 
  (∀ (y : ℝ), y^4 - 34*y^2 + 225 = 0 → x ≤ y) ∧
  x = -5 :=
by sorry

end smallest_solution_quartic_equation_l193_19384


namespace melanie_selling_four_gumballs_l193_19350

/-- The number of gumballs Melanie is selling -/
def num_gumballs : ℕ := 32 / 8

/-- The price of each gumball in cents -/
def price_per_gumball : ℕ := 8

/-- The total amount Melanie gets from selling gumballs in cents -/
def total_amount : ℕ := 32

/-- Theorem stating that Melanie is selling 4 gumballs -/
theorem melanie_selling_four_gumballs :
  num_gumballs = 4 :=
by sorry

end melanie_selling_four_gumballs_l193_19350


namespace five_balls_four_boxes_l193_19328

/-- The number of ways to put n distinguishable balls into k indistinguishable boxes -/
def ballsInBoxes (n k : ℕ) : ℕ :=
  sorry

/-- Theorem: There are 61 ways to put 5 distinguishable balls into 4 indistinguishable boxes -/
theorem five_balls_four_boxes : ballsInBoxes 5 4 = 61 := by
  sorry

end five_balls_four_boxes_l193_19328


namespace fractional_inequality_solution_set_l193_19301

theorem fractional_inequality_solution_set :
  {x : ℝ | (x + 2) / (x - 1) > 0} = {x : ℝ | x > 1 ∨ x < -2} := by sorry

end fractional_inequality_solution_set_l193_19301


namespace constant_variance_properties_l193_19327

/-- A sequence is constant variance if the sequence of its squares is arithmetic -/
def ConstantVariance (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 2)^2 - a (n + 1)^2 = a (n + 1)^2 - a n^2

/-- A sequence is constant if all its terms are equal -/
def ConstantSequence (a : ℕ → ℝ) : Prop :=
  ∃ c : ℝ, ∀ n : ℕ, a n = c

/-- A sequence is arithmetic if the difference between consecutive terms is constant -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

theorem constant_variance_properties (a : ℕ → ℝ) :
  (ConstantSequence a → ConstantVariance a) ∧
  (ConstantVariance a → ArithmeticSequence (λ n => (a n)^2)) ∧
  (ConstantVariance a → ConstantVariance (λ n => a (2*n))) :=
sorry

end constant_variance_properties_l193_19327


namespace sum_of_cubes_l193_19365

theorem sum_of_cubes (a b : ℝ) (h1 : a + b = 4) (h2 : a * b = -5) : a^3 + b^3 = 124 := by
  sorry

end sum_of_cubes_l193_19365


namespace lower_profit_percentage_l193_19361

/-- Proves that given an article with a cost price of $800, if the profit at 18% is $72 more than the profit at another percentage, then that other percentage is 9%. -/
theorem lower_profit_percentage (cost_price : ℝ) (higher_percentage lower_percentage : ℝ) : 
  cost_price = 800 →
  higher_percentage = 18 →
  (higher_percentage / 100) * cost_price = (lower_percentage / 100) * cost_price + 72 →
  lower_percentage = 9 := by
  sorry

end lower_profit_percentage_l193_19361


namespace quadratic_roots_product_l193_19343

theorem quadratic_roots_product (a b : ℝ) : 
  (3 * a^2 + 9 * a - 21 = 0) → 
  (3 * b^2 + 9 * b - 21 = 0) → 
  (3*a - 4) * (5*b - 6) = -27 := by
sorry

end quadratic_roots_product_l193_19343


namespace isabella_currency_exchange_l193_19339

theorem isabella_currency_exchange (d : ℕ) : 
  (11 * d / 8 : ℚ) - 80 = d →
  (d / 100 + (d / 10) % 10 + d % 10 : ℕ) = 6 :=
by sorry

end isabella_currency_exchange_l193_19339


namespace path_area_and_cost_calculation_l193_19346

/-- Calculates the area of a rectangular path around a field -/
def path_area (field_length field_width path_width : ℝ) : ℝ :=
  (field_length + 2 * path_width) * (field_width + 2 * path_width) - field_length * field_width

/-- Calculates the cost of constructing a path given its area and cost per unit area -/
def construction_cost (path_area cost_per_unit : ℝ) : ℝ :=
  path_area * cost_per_unit

theorem path_area_and_cost_calculation 
  (field_length : ℝ) 
  (field_width : ℝ) 
  (path_width : ℝ) 
  (cost_per_unit : ℝ)
  (h1 : field_length = 75)
  (h2 : field_width = 55)
  (h3 : path_width = 2.8)
  (h4 : cost_per_unit = 2) :
  path_area field_length field_width path_width = 759.36 ∧ 
  construction_cost (path_area field_length field_width path_width) cost_per_unit = 1518.72 := by
  sorry

#eval path_area 75 55 2.8
#eval construction_cost (path_area 75 55 2.8) 2

end path_area_and_cost_calculation_l193_19346


namespace solution_range_l193_19333

-- Define the system of inequalities
def system (x m : ℝ) : Prop :=
  (6 - 3*(x + 1) < x - 9) ∧ 
  (x - m > -1) ∧ 
  (x > 3)

-- Theorem statement
theorem solution_range (m : ℝ) : 
  (∀ x, system x m → x > 3) → m ≤ 4 := by
  sorry

end solution_range_l193_19333


namespace train_length_calculation_l193_19349

/-- Calculates the length of a train given its speed, the speed of a person walking in the same direction, and the time it takes for the train to pass the person completely. -/
theorem train_length_calculation (train_speed : ℝ) (man_speed : ℝ) (time_to_cross : ℝ) : 
  train_speed = 63 →
  man_speed = 3 →
  time_to_cross = 29.997600191984642 →
  (train_speed - man_speed) * time_to_cross * (1000 / 3600) = 500 := by
  sorry

#check train_length_calculation

end train_length_calculation_l193_19349


namespace ninas_money_l193_19340

theorem ninas_money (x : ℚ) 
  (h1 : 10 * x = 14 * (x - 1)) : 10 * x = 35 :=
by sorry

end ninas_money_l193_19340


namespace max_value_sqrt_sum_l193_19348

theorem max_value_sqrt_sum (x y z : ℝ) (h1 : x ≥ 0) (h2 : y ≥ 0) (h3 : z ≥ 0) (h4 : x + y + z = 7) :
  Real.sqrt (3 * x + 2) + Real.sqrt (3 * y + 2) + Real.sqrt (3 * z + 2) ≤ 9 := by
  sorry

end max_value_sqrt_sum_l193_19348


namespace geometric_sequence_common_ratio_l193_19388

/-- Given a geometric sequence {a_n} with a₃ = 6 and S₃ = 18,
    prove that the common ratio q is either 1 or -1/2. -/
theorem geometric_sequence_common_ratio 
  (a : ℕ → ℝ) 
  (h_geometric : ∀ n : ℕ, a (n + 1) = a n * q) 
  (h_a3 : a 3 = 6)
  (h_S3 : a 1 + a 2 + a 3 = 18) :
  q = 1 ∨ q = -1/2 := by sorry

end geometric_sequence_common_ratio_l193_19388


namespace katie_game_difference_l193_19391

theorem katie_game_difference : 
  ∀ (katie_new_games katie_old_games friends_new_games : ℕ),
  katie_new_games = 57 →
  katie_old_games = 39 →
  friends_new_games = 34 →
  katie_new_games + katie_old_games - friends_new_games = 62 := by
sorry

end katie_game_difference_l193_19391


namespace problem_solution_l193_19379

theorem problem_solution (x y : ℝ) (h1 : x > 0) (h2 : y > 0)
  (h3 : 8 * x^2 + 16 * x * y = x^3 + 3 * x^2 * y) (h4 : y = 2 * x) :
  x = 40 / 7 := by
sorry

end problem_solution_l193_19379


namespace laura_age_l193_19370

theorem laura_age :
  ∃ (L : ℕ), 
    L > 0 ∧
    L < 100 ∧
    (L - 1) % 8 = 0 ∧
    (L + 1) % 7 = 0 ∧
    (∃ (A : ℕ), 
      A > L ∧
      A < 100 ∧
      (A - 1) % 8 = 0 ∧
      (A + 1) % 7 = 0) →
    L = 41 := by
  sorry

end laura_age_l193_19370


namespace class_composition_l193_19377

theorem class_composition (boys_score girls_score class_average : ℝ) 
  (boys_score_val : boys_score = 80)
  (girls_score_val : girls_score = 90)
  (class_average_val : class_average = 86) :
  let boys_percentage : ℝ := 40
  let girls_percentage : ℝ := 100 - boys_percentage
  class_average = (boys_percentage * boys_score + girls_percentage * girls_score) / 100 :=
by sorry

end class_composition_l193_19377


namespace solve_system_of_equations_l193_19324

theorem solve_system_of_equations (a b m : ℤ) 
  (eq1 : a - b = 6)
  (eq2 : 2 * a + b = m)
  (opposite : a + b = 0) : m = 3 := by
  sorry

end solve_system_of_equations_l193_19324


namespace train_speed_l193_19344

-- Define the train's parameters
def train_length : Real := 240  -- in meters
def crossing_time : Real := 16  -- in seconds

-- Define the conversion factor from m/s to km/h
def mps_to_kmh : Real := 3.6

-- Theorem statement
theorem train_speed :
  let speed_mps := train_length / crossing_time
  let speed_kmh := speed_mps * mps_to_kmh
  speed_kmh = 54 := by
  sorry

end train_speed_l193_19344


namespace brad_siblings_product_l193_19376

/-- A family structure with a focus on two siblings -/
structure Family :=
  (total_sisters : ℕ)
  (total_brothers : ℕ)
  (sarah_sisters : ℕ)
  (sarah_brothers : ℕ)

/-- The number of sisters and brothers that Brad has -/
def brad_siblings (f : Family) : ℕ × ℕ :=
  (f.total_sisters, f.total_brothers - 1)

/-- The theorem stating the product of Brad's siblings -/
theorem brad_siblings_product (f : Family) 
  (h1 : f.sarah_sisters = 4)
  (h2 : f.sarah_brothers = 7)
  (h3 : f.total_sisters = f.sarah_sisters + 1)
  (h4 : f.total_brothers = f.sarah_brothers + 1) :
  (brad_siblings f).1 * (brad_siblings f).2 = 35 := by
  sorry

end brad_siblings_product_l193_19376


namespace digital_earth_function_l193_19380

/-- Represents the functions of a system --/
inductive SystemFunction
| InformationProcessing
| GeographicInformationManagement
| InformationIntegrationAndDisplay
| SpatialPositioning

/-- Represents different systems --/
inductive System
| DigitalEarth
| RemoteSensing
| GeographicInformationSystem
| GlobalPositioningSystem

/-- Defines the function of a given system --/
def system_function : System → SystemFunction
| System.DigitalEarth => SystemFunction.InformationIntegrationAndDisplay
| System.RemoteSensing => SystemFunction.InformationProcessing
| System.GeographicInformationSystem => SystemFunction.GeographicInformationManagement
| System.GlobalPositioningSystem => SystemFunction.SpatialPositioning

/-- Theorem: The function of Digital Earth is Information Integration and Display --/
theorem digital_earth_function :
  system_function System.DigitalEarth = SystemFunction.InformationIntegrationAndDisplay := by
  sorry

end digital_earth_function_l193_19380


namespace consecutive_cube_diff_square_l193_19308

theorem consecutive_cube_diff_square (x : ℤ) :
  ∃ y : ℤ, (x + 1)^3 - x^3 = y^2 →
  ∃ a b : ℤ, y = a^2 + b^2 ∧ b = a + 1 := by
sorry

end consecutive_cube_diff_square_l193_19308


namespace sum_of_first_seven_primes_mod_eighth_prime_l193_19347

def first_eight_primes : List Nat := [2, 3, 5, 7, 11, 13, 17, 19]

theorem sum_of_first_seven_primes_mod_eighth_prime :
  (List.sum (List.take 7 first_eight_primes)) % (List.get! first_eight_primes 7) = 1 := by
  sorry

end sum_of_first_seven_primes_mod_eighth_prime_l193_19347


namespace exists_k_for_1001_free_ends_l193_19313

/-- Represents the number of free ends after k iterations of extending segments -/
def num_free_ends (k : ℕ) : ℕ := 1 + 4 * k

/-- Theorem stating that there exists a number of iterations that results in 1001 free ends -/
theorem exists_k_for_1001_free_ends : ∃ k : ℕ, num_free_ends k = 1001 := by
  sorry

end exists_k_for_1001_free_ends_l193_19313


namespace misery_ratio_bound_l193_19325

/-- Represents a room with its total load -/
structure Room where
  load : ℝ
  load_positive : load > 0

/-- Represents a student with their download request -/
structure Student where
  bits : ℝ
  bits_positive : bits > 0

/-- Calculates the displeasure of a student in a given room -/
def displeasure (s : Student) (r : Room) : ℝ := s.bits * r.load

/-- Calculates the total misery for a given configuration -/
def misery (students : List Student) (rooms : List Room) (assignment : Student → Room) : ℝ :=
  (students.map (fun s => displeasure s (assignment s))).sum

/-- Defines a balanced configuration -/
def is_balanced (students : List Student) (rooms : List Room) (assignment : Student → Room) : Prop :=
  ∀ s : Student, ∀ r : Room, displeasure s (assignment s) ≤ displeasure s r

theorem misery_ratio_bound 
  (students : List Student) 
  (rooms : List Room) 
  (balanced_assignment : Student → Room)
  (other_assignment : Student → Room)
  (h_balanced : is_balanced students rooms balanced_assignment) :
  let M1 := misery students rooms balanced_assignment
  let M2 := misery students rooms other_assignment
  M1 / M2 ≤ 9 / 8 := by
  sorry

end misery_ratio_bound_l193_19325


namespace parallelogram_properties_l193_19310

/-- A parallelogram with given vertices -/
structure Parallelogram where
  v1 : ℝ × ℝ
  v2 : ℝ × ℝ
  v3 : ℝ × ℝ
  v4 : ℝ × ℝ
  opposite : v1 = v2

/-- The fourth vertex and diagonal intersection of a specific parallelogram -/
theorem parallelogram_properties (p : Parallelogram) 
  (h1 : p.v1 = (2, -3))
  (h2 : p.v2 = (8, 5))
  (h3 : p.v3 = (5, 0)) :
  p.v4 = (5, 2) ∧ 
  (((p.v1.1 + p.v2.1) / 2, (p.v1.2 + p.v2.2) / 2) : ℝ × ℝ) = (5, 1) := by
  sorry

end parallelogram_properties_l193_19310


namespace max_four_digit_product_of_primes_l193_19387

def is_prime (p : ℕ) : Prop := Nat.Prime p

theorem max_four_digit_product_of_primes :
  ∃ (n x y : ℕ),
    n = x * y * (10 * x + y) ∧
    is_prime x ∧
    is_prime y ∧
    is_prime (10 * x + y) ∧
    x < 5 ∧
    y < 5 ∧
    x ≠ y ∧
    1000 ≤ n ∧
    n < 10000 ∧
    (∀ (m x' y' : ℕ),
      m = x' * y' * (10 * x' + y') →
      is_prime x' →
      is_prime y' →
      is_prime (10 * x' + y') →
      x' < 5 →
      y' < 5 →
      x' ≠ y' →
      1000 ≤ m →
      m < 10000 →
      m ≤ n) ∧
    n = 138 :=
by sorry

end max_four_digit_product_of_primes_l193_19387


namespace water_surface_scientific_notation_correct_l193_19322

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ
  is_valid : 1 ≤ coefficient ∧ coefficient < 10

/-- The area of water surface in China in km² -/
def water_surface_area : ℕ := 370000

/-- The scientific notation representation of the water surface area -/
def water_surface_scientific : ScientificNotation :=
  { coefficient := 3.7
    exponent := 5
    is_valid := by sorry }

/-- Theorem stating that the water surface area is correctly represented in scientific notation -/
theorem water_surface_scientific_notation_correct :
  (water_surface_scientific.coefficient * (10 : ℝ) ^ water_surface_scientific.exponent) = water_surface_area := by
  sorry

end water_surface_scientific_notation_correct_l193_19322


namespace fraction_calculation_l193_19353

theorem fraction_calculation : (1 / 4 : ℚ) * (1 / 3 : ℚ) * (1 / 6 : ℚ) * 144 + (1 / 2 : ℚ) = (5 / 2 : ℚ) := by
  sorry

end fraction_calculation_l193_19353


namespace max_b_minus_a_l193_19331

theorem max_b_minus_a (a b : ℝ) : 
  a < 0 → 
  (∀ x : ℝ, (x^2 + 2017*a)*(x + 2016*b) ≥ 0) → 
  b - a ≤ 2017 :=
by sorry

end max_b_minus_a_l193_19331


namespace square_root_sum_l193_19393

theorem square_root_sum (x : ℝ) :
  (Real.sqrt (64 - x^2) - Real.sqrt (16 - x^2) = 4) →
  (Real.sqrt (64 - x^2) + Real.sqrt (16 - x^2) = 12) :=
by sorry

end square_root_sum_l193_19393


namespace guitar_price_theorem_l193_19337

theorem guitar_price_theorem (hendricks_price : ℝ) (discount_percentage : ℝ) (gerald_price : ℝ) : 
  hendricks_price = 200 →
  discount_percentage = 20 →
  hendricks_price = gerald_price * (1 - discount_percentage / 100) →
  gerald_price = 250 :=
by sorry

end guitar_price_theorem_l193_19337


namespace tangent_intersection_x_coordinate_l193_19371

/-- A circle in a 2D plane -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- Point where a line intersects the x-axis -/
def XAxisIntersection : ℝ → ℝ × ℝ := λ x ↦ (x, 0)

/-- Theorem: Tangent line intersection for two specific circles -/
theorem tangent_intersection_x_coordinate :
  let c1 : Circle := { center := (0, 0), radius := 3 }
  let c2 : Circle := { center := (12, 0), radius := 5 }
  ∃ (x : ℝ), x > 0 ∧ 
    ∃ (l : Set (ℝ × ℝ)), 
      (XAxisIntersection x ∈ l) ∧ 
      (∃ p1 ∈ l, (p1.1 - c1.center.1)^2 + (p1.2 - c1.center.2)^2 = c1.radius^2) ∧
      (∃ p2 ∈ l, (p2.1 - c2.center.1)^2 + (p2.2 - c2.center.2)^2 = c2.radius^2) ∧
      x = 9/2 :=
by
  sorry

end tangent_intersection_x_coordinate_l193_19371


namespace P_superset_Q_l193_19394

-- Define the sets P and Q
def P : Set ℝ := {x | x ≥ 5}
def Q : Set ℝ := {x | 5 ≤ x ∧ x ≤ 7}

-- Theorem statement
theorem P_superset_Q : Q ⊆ P := by
  sorry

end P_superset_Q_l193_19394


namespace root_difference_zero_l193_19392

theorem root_difference_zero : ∃ (r : ℝ), 
  (∀ x : ℝ, x^2 + 20*x + 75 = -25 ↔ x = r) ∧ 
  (abs (r - r) = 0) :=
by sorry

end root_difference_zero_l193_19392


namespace sum_of_x_solutions_is_zero_l193_19378

theorem sum_of_x_solutions_is_zero :
  ∀ x₁ x₂ : ℝ,
  (∃ y : ℝ, y = 7 ∧ x₁^2 + y^2 = 100 ∧ x₂^2 + y^2 = 100) →
  x₁ + x₂ = 0 :=
by sorry

end sum_of_x_solutions_is_zero_l193_19378


namespace chessboard_placements_l193_19386

/-- Represents a standard 8x8 chessboard -/
def Chessboard := Fin 8 × Fin 8

/-- Represents the different types of chess pieces -/
inductive ChessPiece
| Rook
| King
| Bishop
| Knight
| Queen

/-- Returns true if two pieces of the given type at the given positions do not attack each other -/
def not_attacking (piece : ChessPiece) (pos1 pos2 : Chessboard) : Prop := sorry

/-- Counts the number of ways to place two identical pieces on the chessboard without attacking each other -/
def count_placements (piece : ChessPiece) : ℕ := sorry

theorem chessboard_placements :
  (count_placements ChessPiece.Rook = 1568) ∧
  (count_placements ChessPiece.King = 1806) ∧
  (count_placements ChessPiece.Bishop = 1736) ∧
  (count_placements ChessPiece.Knight = 1848) ∧
  (count_placements ChessPiece.Queen = 1288) := by sorry

end chessboard_placements_l193_19386


namespace city_population_l193_19375

/-- Represents the population distribution of a city -/
structure CityPopulation where
  total : ℕ
  under18 : ℕ
  between18and65 : ℕ
  over65 : ℕ
  belowPovertyLine : ℕ
  middleClass : ℕ
  wealthy : ℕ
  menUnder18 : ℕ
  womenUnder18 : ℕ

/-- Theorem stating the total population of the city given the conditions -/
theorem city_population (c : CityPopulation) : c.total = 500000 :=
  by
  have h1 : c.under18 = c.total / 4 := sorry
  have h2 : c.between18and65 = c.total * 11 / 20 := sorry
  have h3 : c.over65 = c.total / 5 := sorry
  have h4 : c.belowPovertyLine = c.total * 3 / 20 := sorry
  have h5 : c.middleClass = c.total * 13 / 20 := sorry
  have h6 : c.wealthy = c.total / 5 := sorry
  have h7 : c.menUnder18 = c.under18 * 3 / 5 := sorry
  have h8 : c.womenUnder18 = c.under18 * 2 / 5 := sorry
  have h9 : c.wealthy * 1 / 5 = 20000 := sorry
  sorry

#check city_population

end city_population_l193_19375


namespace positive_difference_of_roots_l193_19329

theorem positive_difference_of_roots : ∃ (r₁ r₂ : ℝ),
  (r₁^2 - 5*r₁ - 26) / (r₁ + 5) = 3*r₁ + 8 ∧
  (r₂^2 - 5*r₂ - 26) / (r₂ + 5) = 3*r₂ + 8 ∧
  r₁ ≠ r₂ ∧
  |r₁ - r₂| = 8 :=
by sorry

end positive_difference_of_roots_l193_19329


namespace ice_cream_consumption_l193_19318

theorem ice_cream_consumption (friday_amount saturday_amount : Real) 
  (h1 : friday_amount = 3.25) 
  (h2 : saturday_amount = 0.25) : 
  friday_amount + saturday_amount = 3.5 := by
  sorry

end ice_cream_consumption_l193_19318


namespace quadrilateral_perimeter_l193_19338

/-- A quadrilateral ABCD with specific properties -/
structure Quadrilateral :=
  (A B C D : ℝ × ℝ)
  (perpendicular : (B.1 - A.1) * (C.1 - B.1) + (B.2 - A.2) * (C.2 - B.2) = 0)
  (parallel : (D.1 - C.1) * (B.2 - A.2) = (D.2 - C.2) * (B.1 - A.1))
  (AB_length : Real.sqrt ((B.1 - A.1)^2 + (B.2 - A.2)^2) = 7)
  (DC_length : Real.sqrt ((D.1 - C.1)^2 + (D.2 - C.2)^2) = 6)
  (BC_length : Real.sqrt ((C.1 - B.1)^2 + (C.2 - B.2)^2) = 10)

/-- The perimeter of the quadrilateral ABCD is 35.2 cm -/
theorem quadrilateral_perimeter (q : Quadrilateral) :
  Real.sqrt ((q.B.1 - q.A.1)^2 + (q.B.2 - q.A.2)^2) +
  Real.sqrt ((q.C.1 - q.B.1)^2 + (q.C.2 - q.B.2)^2) +
  Real.sqrt ((q.D.1 - q.C.1)^2 + (q.D.2 - q.C.2)^2) +
  Real.sqrt ((q.A.1 - q.D.1)^2 + (q.A.2 - q.D.2)^2) = 35.2 := by
  sorry

end quadrilateral_perimeter_l193_19338


namespace two_digit_reverse_sum_square_l193_19362

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n < 100

def reverse_digits (n : ℕ) : ℕ :=
  (n % 10) * 10 + (n / 10)

def is_perfect_square (n : ℕ) : Prop :=
  ∃ k : ℕ, k * k = n

def satisfies_condition (n : ℕ) : Prop :=
  is_two_digit n ∧ is_perfect_square (n + reverse_digits n)

theorem two_digit_reverse_sum_square :
  {n : ℕ | satisfies_condition n} = {29, 38, 47, 56, 65, 74, 83, 92} :=
by sorry

end two_digit_reverse_sum_square_l193_19362


namespace probability_second_genuine_given_first_genuine_l193_19345

def total_items : ℕ := 10
def genuine_items : ℕ := 6
def defective_items : ℕ := 4

theorem probability_second_genuine_given_first_genuine :
  let first_genuine : ℝ := genuine_items / total_items
  let second_genuine : ℝ := (genuine_items - 1) / (total_items - 1)
  let both_genuine : ℝ := first_genuine * second_genuine
  both_genuine / first_genuine = 5 / 9 :=
by sorry

end probability_second_genuine_given_first_genuine_l193_19345


namespace product_equals_one_l193_19303

theorem product_equals_one (x y z : ℝ) 
  (eq1 : x + 1/y = 4)
  (eq2 : y + 1/z = 1)
  (eq3 : z + 1/x = 7/3) :
  x * y * z = 1 := by
sorry

end product_equals_one_l193_19303


namespace hyperbola_asymptote_l193_19367

/-- Given a hyperbola with equation x^2 - y^2/b^2 = 1 where b > 0,
    if one of its asymptotes is y = 3x, then b = 3. -/
theorem hyperbola_asymptote (b : ℝ) (hb : b > 0) : 
  (∀ x y : ℝ, x^2 - y^2/b^2 = 1) → 
  (∃ x y : ℝ, y = 3*x ∧ x^2 - y^2/b^2 = 1) → 
  b = 3 := by
sorry

end hyperbola_asymptote_l193_19367


namespace oliver_games_l193_19357

def number_of_games (initial_money : ℕ) (money_spent : ℕ) (game_cost : ℕ) : ℕ :=
  (initial_money - money_spent) / game_cost

theorem oliver_games : number_of_games 35 7 4 = 7 := by
  sorry

end oliver_games_l193_19357


namespace bottle_cost_l193_19305

theorem bottle_cost (total : ℕ) (wine_extra : ℕ) (h1 : total = 30) (h2 : wine_extra = 26) : 
  ∃ (bottle : ℕ), bottle + (bottle + wine_extra) = total ∧ bottle = 2 := by
  sorry

end bottle_cost_l193_19305


namespace hyperbola_circle_range_l193_19352

theorem hyperbola_circle_range (a : ℝ) : 
  let P := (a > 1 ∨ a < -3)
  let Q := (-1 < a ∧ a < 3)
  (¬(P ∧ Q) ∧ ¬(¬Q)) → (-1 < a ∧ a ≤ 1) := by
  sorry

end hyperbola_circle_range_l193_19352


namespace floor_product_theorem_l193_19358

theorem floor_product_theorem :
  ∃ (x : ℝ), x > 0 ∧ (↑⌊x⌋ : ℝ) * x = 90 → x = 10 := by
  sorry

end floor_product_theorem_l193_19358


namespace maria_age_l193_19354

theorem maria_age (maria ann : ℕ) : 
  maria = ann - 3 →
  maria - 4 = (ann - 4) / 2 →
  maria = 7 := by sorry

end maria_age_l193_19354


namespace bell_rings_theorem_l193_19395

/-- Represents the number of times a bell rings for a single class -/
def bell_rings_per_class : ℕ := 2

/-- Represents the total number of classes in a day -/
def total_classes : ℕ := 5

/-- Represents the current class number (1-indexed) -/
def current_class : ℕ := 5

/-- Calculates the total number of bell rings up to and including the current class -/
def total_bell_rings (completed_classes : ℕ) (current_class : ℕ) : ℕ :=
  completed_classes * bell_rings_per_class + 1

/-- Theorem: Given 5 classes where the bell rings twice for each completed class 
    and once for the current class (Music), the total number of bell rings is 9 -/
theorem bell_rings_theorem : 
  total_bell_rings (current_class - 1) current_class = 9 := by
  sorry

end bell_rings_theorem_l193_19395


namespace geometric_sequence_sixth_term_l193_19385

/-- Given a geometric sequence with first term 243 and eighth term 32,
    the sixth term of the sequence is 1. -/
theorem geometric_sequence_sixth_term :
  ∀ (a : ℝ) (r : ℝ),
    a = 243 →
    a * r^7 = 32 →
    a * r^5 = 1 :=
by
  sorry

end geometric_sequence_sixth_term_l193_19385


namespace shortest_distance_moving_points_l193_19314

/-- The shortest distance between two points moving along perpendicular edges of a square -/
theorem shortest_distance_moving_points (side_length : ℝ) (v1 v2 : ℝ) 
  (h1 : side_length = 10)
  (h2 : v1 = 30 / 100)
  (h3 : v2 = 40 / 100) :
  ∃ t : ℝ, ∃ x y : ℝ,
    x = v1 * t ∧
    y = v2 * t ∧
    ∀ s : ℝ, (v1 * s - side_length)^2 + (v2 * s)^2 ≥ x^2 + y^2 ∧
    Real.sqrt (x^2 + y^2) = 8 :=
by sorry

end shortest_distance_moving_points_l193_19314


namespace golden_ratio_bounds_l193_19369

theorem golden_ratio_bounds : ∃ x : ℝ, x = Real.sqrt (1 + x) ∧ 1 < x ∧ x < 2 := by
  sorry

end golden_ratio_bounds_l193_19369


namespace reach_64_from_2_cannot_reach_2_2011_from_2_l193_19341

def cube (x : ℚ) : ℚ := x^3

def div_by_8 (x : ℚ) : ℚ := x / 8

inductive Operation
| Cube
| DivBy8

def apply_operation (x : ℚ) (op : Operation) : ℚ :=
  match op with
  | Operation.Cube => cube x
  | Operation.DivBy8 => div_by_8 x

def can_reach (start : ℚ) (target : ℚ) : Prop :=
  ∃ (ops : List Operation), target = ops.foldl apply_operation start

theorem reach_64_from_2 : can_reach 2 64 := by sorry

theorem cannot_reach_2_2011_from_2 : ¬ can_reach 2 (2^2011) := by sorry

end reach_64_from_2_cannot_reach_2_2011_from_2_l193_19341


namespace range_of_x_l193_19326

theorem range_of_x (a b x : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) :
  (∀ a b, a ≠ 0 → b ≠ 0 → |2*a - b| + |a + b| ≥ |a| * (|x - 1| + |x + 1|)) →
  x ∈ Set.Icc (-3/2) (3/2) :=
by sorry

end range_of_x_l193_19326


namespace square_diagonal_property_l193_19306

/-- Represents a point in 2D space -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- Represents a square -/
structure Square :=
  (p q r s : Point)

/-- The small square PQRS is contained in the big square -/
def small_square_in_big_square (small : Square) (big : Square) : Prop := sorry

/-- Point A lies on the extension of PQ -/
def point_on_extension (p q a : Point) : Prop := sorry

/-- Points A, B, C, D lie on the sides of the big square in order -/
def points_on_sides (a b c d : Point) (big : Square) : Prop := sorry

/-- Two line segments are equal -/
def segments_equal (p1 q1 p2 q2 : Point) : Prop := sorry

/-- Two line segments are perpendicular -/
def segments_perpendicular (p1 q1 p2 q2 : Point) : Prop := sorry

theorem square_diagonal_property (small big : Square) (a b c d : Point) :
  small_square_in_big_square small big →
  point_on_extension small.p small.q a →
  point_on_extension small.q small.r b →
  point_on_extension small.r small.s c →
  point_on_extension small.s small.p d →
  points_on_sides a b c d big →
  segments_equal a c b d ∧ segments_perpendicular a c b d := by
  sorry

end square_diagonal_property_l193_19306


namespace inequality_proof_l193_19304

theorem inequality_proof (x y z : ℝ) 
  (h1 : 0 < z) (h2 : z < y) (h3 : y < x) (h4 : x < π/2) : 
  (π/2) + 2 * Real.sin x * Real.cos y + 2 * Real.sin y * Real.cos z > 
  Real.sin (2*x) + Real.sin (2*y) + Real.sin (2*z) := by
  sorry

end inequality_proof_l193_19304


namespace system_implication_l193_19366

variable {X : Type*} [LinearOrder X]
variable (f g : X → ℝ)

theorem system_implication :
  (∀ x, f x > 0 ∧ g x > 0) →
  (∀ x, f x > 0 ∧ f x + g x > 0) :=
by sorry

example : ∃ f g : ℝ → ℝ, 
  (∃ x, f x > 0 ∧ f x + g x > 0) ∧
  ¬(∀ x, f x > 0 ∧ g x > 0) :=
by sorry

end system_implication_l193_19366


namespace pie_chart_most_suitable_l193_19317

-- Define the characteristics of the data
structure DataCharacteristics where
  partsOfWhole : Bool
  categorical : Bool
  compareProportions : Bool

-- Define the types of statistical graphs
inductive StatisticalGraph
  | PieChart
  | BarGraph
  | LineGraph
  | Histogram

-- Define the suitability of a graph for given data characteristics
def isSuitable (graph : StatisticalGraph) (data : DataCharacteristics) : Prop :=
  match graph with
  | StatisticalGraph.PieChart => data.partsOfWhole ∧ data.categorical ∧ data.compareProportions
  | _ => False

-- Theorem statement
theorem pie_chart_most_suitable (data : DataCharacteristics) 
  (h1 : data.partsOfWhole = true) 
  (h2 : data.categorical = true) 
  (h3 : data.compareProportions = true) :
  ∀ (graph : StatisticalGraph), 
    isSuitable graph data → graph = StatisticalGraph.PieChart := by
  sorry

end pie_chart_most_suitable_l193_19317


namespace race_head_start_l193_19368

/-- Given two runners A and B, where A's speed is 21/19 times B's speed,
    the head start fraction that A should give B for a dead heat is 2/21 of the race length. -/
theorem race_head_start (speed_a speed_b length head_start : ℝ) :
  speed_a = (21 / 19) * speed_b →
  length > 0 →
  head_start > 0 →
  length / speed_a = (length - head_start) / speed_b →
  head_start / length = 2 / 21 := by
sorry

end race_head_start_l193_19368


namespace rectangle_area_with_inscribed_circle_l193_19334

theorem rectangle_area_with_inscribed_circle (r : ℝ) (ratio : ℝ) : 
  r = 7 → ratio = 3 → 
  let width := 2 * r
  let length := ratio * width
  width * length = 588 := by
sorry

end rectangle_area_with_inscribed_circle_l193_19334


namespace blue_pill_cost_proof_l193_19315

/-- The cost of one blue pill in dollars -/
def blue_pill_cost : ℚ := 11

/-- The number of days in the treatment period -/
def days : ℕ := 21

/-- The daily discount in dollars after the first week -/
def daily_discount : ℚ := 2

/-- The number of days with discount -/
def discount_days : ℕ := 14

/-- The total cost without discount for the entire period -/
def total_cost_without_discount : ℚ := 735

/-- The number of blue pills taken daily -/
def daily_blue_pills : ℕ := 2

/-- The number of orange pills taken daily -/
def daily_orange_pills : ℕ := 1

/-- The cost difference between orange and blue pills in dollars -/
def orange_blue_cost_difference : ℚ := 2

theorem blue_pill_cost_proof :
  blue_pill_cost * (daily_blue_pills * days + daily_orange_pills * days) +
  orange_blue_cost_difference * (daily_orange_pills * days) -
  daily_discount * discount_days = total_cost_without_discount - daily_discount * discount_days :=
by sorry

end blue_pill_cost_proof_l193_19315


namespace taehyung_has_most_points_l193_19397

def yoongi_points : ℕ := 7
def jungkook_points : ℕ := 6
def yuna_points : ℕ := 9
def yoojung_points : ℕ := 8
def taehyung_points : ℕ := 10

theorem taehyung_has_most_points :
  taehyung_points ≥ yoongi_points ∧
  taehyung_points ≥ jungkook_points ∧
  taehyung_points ≥ yuna_points ∧
  taehyung_points ≥ yoojung_points :=
by sorry

end taehyung_has_most_points_l193_19397


namespace no_real_solutions_cubic_equation_l193_19374

theorem no_real_solutions_cubic_equation :
  ∀ x : ℝ, x^3 + 2*(x+1)^3 + 3*(x+2)^3 ≠ 6*(x+4)^3 := by
  sorry

end no_real_solutions_cubic_equation_l193_19374


namespace production_normality_l193_19330

-- Define the parameters of the normal distribution
def μ : ℝ := 8.0
def σ : ℝ := 0.15

-- Define the 3-sigma range
def lower_bound : ℝ := μ - 3 * σ
def upper_bound : ℝ := μ + 3 * σ

-- Define the observed diameters
def morning_diameter : ℝ := 7.9
def afternoon_diameter : ℝ := 7.5

-- Define what it means for a production to be normal
def is_normal (x : ℝ) : Prop := lower_bound ≤ x ∧ x ≤ upper_bound

-- Theorem statement
theorem production_normality :
  is_normal morning_diameter ∧ ¬is_normal afternoon_diameter :=
sorry

end production_normality_l193_19330


namespace simplify_expression_l193_19356

theorem simplify_expression (x y : ℝ) (h : x ≠ 0) :
  y * (x⁻¹ - 2) = (y * (1 - 2*x)) / x := by sorry

end simplify_expression_l193_19356


namespace fair_ride_cost_l193_19307

/-- Represents the fair entrance and ride costs --/
structure FairCosts where
  under18Fee : ℚ
  adultFeeIncrease : ℚ
  totalSpent : ℚ
  numRides : ℕ
  numUnder18 : ℕ
  numAdults : ℕ

/-- Calculates the cost per ride given the fair costs --/
def costPerRide (costs : FairCosts) : ℚ :=
  let adultFee := costs.under18Fee * (1 + costs.adultFeeIncrease)
  let totalEntrance := costs.under18Fee * costs.numUnder18 + adultFee * costs.numAdults
  let totalRideCost := costs.totalSpent - totalEntrance
  totalRideCost / costs.numRides

/-- Theorem stating that the cost per ride is $0.50 given the problem conditions --/
theorem fair_ride_cost :
  let costs : FairCosts := {
    under18Fee := 5,
    adultFeeIncrease := 1/5,
    totalSpent := 41/2,
    numRides := 9,
    numUnder18 := 2,
    numAdults := 1
  }
  costPerRide costs = 1/2 := by sorry


end fair_ride_cost_l193_19307


namespace no_consecutive_squares_l193_19373

def t (n : ℕ) : ℕ := (Nat.divisors n).card

def a : ℕ → ℕ
  | 0 => 1  -- Arbitrary starting value
  | n + 1 => a n + 2 * t n

theorem no_consecutive_squares (n k : ℕ) :
  a n = k^2 → ¬∃ m : ℕ, a (n + 1) = (k + m)^2 :=
by sorry

end no_consecutive_squares_l193_19373


namespace modular_inverse_of_4_mod_21_l193_19321

theorem modular_inverse_of_4_mod_21 : ∃ x : ℕ, x ≤ 20 ∧ (4 * x) % 21 = 1 :=
by
  use 16
  sorry

end modular_inverse_of_4_mod_21_l193_19321


namespace new_person_weight_l193_19309

theorem new_person_weight (initial_count : ℕ) (initial_weight : ℝ) (weight_increase : ℝ) (replaced_weight : ℝ) :
  initial_count = 8 →
  replaced_weight = 40 →
  weight_increase = 6 →
  (initial_count : ℝ) * weight_increase + replaced_weight = 88 :=
by
  sorry

end new_person_weight_l193_19309


namespace binomial_12_6_l193_19382

theorem binomial_12_6 : Nat.choose 12 6 = 1848 := by
  sorry

end binomial_12_6_l193_19382


namespace pizza_slice_volume_l193_19390

/-- The volume of a pizza slice -/
theorem pizza_slice_volume (thickness : ℝ) (diameter : ℝ) (num_slices : ℕ) :
  thickness = 1/2 →
  diameter = 16 →
  num_slices = 16 →
  (π * (diameter/2)^2 * thickness) / num_slices = 2 * π := by
  sorry

#check pizza_slice_volume

end pizza_slice_volume_l193_19390


namespace similar_triangle_perimeter_l193_19372

/-- Represents a triangle with side lengths a, b, and c -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Checks if a triangle is isosceles -/
def Triangle.isIsosceles (t : Triangle) : Prop :=
  t.a = t.b ∨ t.b = t.c ∨ t.c = t.a

/-- Calculates the perimeter of a triangle -/
def Triangle.perimeter (t : Triangle) : ℝ :=
  t.a + t.b + t.c

/-- Checks if two triangles are similar -/
def Triangle.isSimilar (t1 t2 : Triangle) : Prop :=
  ∃ k : ℝ, k > 0 ∧ 
    t2.a = k * t1.a ∧
    t2.b = k * t1.b ∧
    t2.c = k * t1.c

theorem similar_triangle_perimeter (t1 t2 : Triangle) :
  t1.isIsosceles ∧
  t1.a = 30 ∧ t1.b = 30 ∧ t1.c = 15 ∧
  t2.isSimilar t1 ∧
  min t2.a (min t2.b t2.c) = 45 →
  t2.perimeter = 225 := by
  sorry

end similar_triangle_perimeter_l193_19372


namespace leading_coeff_of_polynomial_l193_19319

/-- Given a polynomial f such that f(x + 1) - f(x) = 8x^2 + 6x + 4 for all real x,
    the leading coefficient of f is 8/3 -/
theorem leading_coeff_of_polynomial (f : ℝ → ℝ) :
  (∀ x : ℝ, f (x + 1) - f x = 8 * x^2 + 6 * x + 4) →
  ∃ (a b c d : ℝ), (∀ x : ℝ, f x = (8/3) * x^3 + a * x^2 + b * x + c) ∧ a ≠ (8/3) :=
by sorry

end leading_coeff_of_polynomial_l193_19319


namespace problem_solution_l193_19312

theorem problem_solution :
  ∀ (A B C : ℝ) (a n b c : ℕ) (d : ℕ+),
    A^2 + B^2 + C^2 = 3 →
    A * B + B * C + C * A = 3 →
    a = A^2 →
    29 * n + 42 * b = a →
    5 < b →
    b < 10 →
    (Real.sqrt 3 - Real.sqrt 5 + Real.sqrt 7) / (Real.sqrt 3 + Real.sqrt 5 + Real.sqrt 7) = 
      (c * Real.sqrt 21 - 18 * Real.sqrt 15 - 2 * Real.sqrt 35 + b) / 59 →
    d = (Nat.factors c).length →
    a = 1 ∧ b = 9 ∧ c = 20 ∧ d = 6 := by
  sorry


end problem_solution_l193_19312
