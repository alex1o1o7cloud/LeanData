import Mathlib

namespace screws_per_section_l1423_142324

def initial_screws : ℕ := 8
def buy_multiplier : ℕ := 2
def num_sections : ℕ := 4

theorem screws_per_section :
  (initial_screws + initial_screws * buy_multiplier) / num_sections = 6 := by
  sorry

end screws_per_section_l1423_142324


namespace floor_ceiling_sum_l1423_142304

theorem floor_ceiling_sum : ⌊(-3.67 : ℝ)⌋ + ⌈(34.2 : ℝ)⌉ = 31 := by
  sorry

end floor_ceiling_sum_l1423_142304


namespace company_x_employees_l1423_142350

theorem company_x_employees (full_time : ℕ) (worked_year : ℕ) (neither : ℕ) (both : ℕ) :
  full_time = 80 →
  worked_year = 100 →
  neither = 20 →
  both = 30 →
  full_time + worked_year - both + neither = 170 := by
  sorry

end company_x_employees_l1423_142350


namespace villager_A_motorcycle_fraction_l1423_142384

/-- Represents the scenario of two villagers and a motorcycle traveling to a station -/
structure TravelScenario where
  totalDistance : ℝ := 1
  walkingSpeed : ℝ
  motorcycleSpeed : ℝ
  simultaneousArrival : Prop

/-- The main theorem stating the fraction of journey villager A travels by motorcycle -/
theorem villager_A_motorcycle_fraction (scenario : TravelScenario) 
  (h1 : scenario.motorcycleSpeed = 9 * scenario.walkingSpeed)
  (h2 : scenario.simultaneousArrival) : 
  ∃ (x : ℝ), x = 5/6 ∧ x * scenario.totalDistance = scenario.totalDistance - scenario.walkingSpeed / scenario.motorcycleSpeed * scenario.totalDistance :=
by sorry

end villager_A_motorcycle_fraction_l1423_142384


namespace decimal_sum_to_fraction_l1423_142360

theorem decimal_sum_to_fraction :
  (0.3 : ℚ) + 0.04 + 0.005 + 0.0006 + 0.00007 = 34567 / 100000 := by
  sorry

end decimal_sum_to_fraction_l1423_142360


namespace solution_equality_l1423_142375

theorem solution_equality (x y : ℝ) : 
  |x + y - 2| + (2 * x - 3 * y + 5)^2 = 0 → 
  ((x = 1 ∧ y = 9) ∨ (x = 5 ∧ y = 5)) := by
sorry

end solution_equality_l1423_142375


namespace sweets_neither_red_nor_green_l1423_142391

theorem sweets_neither_red_nor_green 
  (total : ℕ) (red : ℕ) (green : ℕ) 
  (h1 : total = 285) 
  (h2 : red = 49) 
  (h3 : green = 59) : 
  total - (red + green) = 177 := by
  sorry

end sweets_neither_red_nor_green_l1423_142391


namespace morgan_pens_l1423_142393

theorem morgan_pens (total red blue : ℕ) (h1 : total = 168) (h2 : red = 65) (h3 : blue = 45) :
  total - red - blue = 58 := by
  sorry

end morgan_pens_l1423_142393


namespace unique_positive_zero_implies_a_negative_l1423_142349

/-- The function f(x) = ax³ + 3x² + 1 -/
def f (a : ℝ) (x : ℝ) : ℝ := a * x^3 + 3 * x^2 + 1

/-- Theorem: If f(x) has a unique zero point x₀ > 0, then a ∈ (-∞, 0) -/
theorem unique_positive_zero_implies_a_negative
  (a : ℝ)
  (h_unique : ∃! x₀ : ℝ, f a x₀ = 0)
  (h_positive : ∃ x₀ : ℝ, f a x₀ = 0 ∧ x₀ > 0) :
  a < 0 :=
sorry

end unique_positive_zero_implies_a_negative_l1423_142349


namespace prize_money_calculation_l1423_142387

def total_amount : ℕ := 300
def paintings_sold : ℕ := 3
def price_per_painting : ℕ := 50

theorem prize_money_calculation :
  total_amount - (paintings_sold * price_per_painting) = 150 := by
  sorry

end prize_money_calculation_l1423_142387


namespace config_7_3_1_wins_for_second_player_l1423_142329

/-- Represents the nim-value of a wall of bricks. -/
def nimValue (n : ℕ) : ℕ :=
  match n with
  | 1 => 1
  | 2 => 2
  | 3 => 3
  | 4 => 1
  | 5 => 4
  | 6 => 3
  | 7 => 2
  | _ => 0  -- Default case, though not used in this problem

/-- Calculates the nim-sum (XOR) of a list of natural numbers. -/
def nimSum : List ℕ → ℕ
  | [] => 0
  | (x::xs) => x ^^^ (nimSum xs)

/-- Represents a configuration of walls in the game. -/
structure GameConfig where
  walls : List ℕ

/-- Determines if a given game configuration is a winning position for the second player. -/
def isWinningForSecondPlayer (config : GameConfig) : Prop :=
  nimSum (config.walls.map nimValue) = 0

/-- The theorem stating that the configuration (7, 3, 1) is a winning position for the second player. -/
theorem config_7_3_1_wins_for_second_player :
  isWinningForSecondPlayer ⟨[7, 3, 1]⟩ := by sorry

end config_7_3_1_wins_for_second_player_l1423_142329


namespace rational_smallest_abs_value_and_monomial_degree_l1423_142300

-- Define the concept of absolute value for rational numbers
def abs_rat (q : ℚ) : ℚ := max q (-q)

-- Define the degree of a monomial
def monomial_degree (a b c : ℕ) : ℕ := a + b + c

theorem rational_smallest_abs_value_and_monomial_degree :
  (∀ q : ℚ, abs_rat q ≥ 0) ∧
  (∀ q : ℚ, abs_rat q = 0 ↔ q = 0) ∧
  (monomial_degree 2 1 0 = 3) :=
sorry

end rational_smallest_abs_value_and_monomial_degree_l1423_142300


namespace factorial_sum_l1423_142361

theorem factorial_sum : 7 * Nat.factorial 7 + 5 * Nat.factorial 5 + 6 * Nat.factorial 6 = 40200 := by
  sorry

end factorial_sum_l1423_142361


namespace geometric_progression_cubed_sum_l1423_142365

theorem geometric_progression_cubed_sum
  (b s : ℝ) (h1 : -1 < s) (h2 : s < 1) :
  let series := fun n => b^3 * s^(3*n)
  (∑' n, series n) = b^3 / (1 - s^3) :=
sorry

end geometric_progression_cubed_sum_l1423_142365


namespace ball_count_proof_l1423_142344

/-- Theorem: Given a bag of balls with specific color counts and probability,
    prove the total number of balls. -/
theorem ball_count_proof
  (white green yellow red purple : ℕ)
  (prob_not_red_or_purple : ℚ)
  (h1 : white = 50)
  (h2 : green = 20)
  (h3 : yellow = 10)
  (h4 : red = 17)
  (h5 : purple = 3)
  (h6 : prob_not_red_or_purple = 4/5) :
  white + green + yellow + red + purple = 100 := by
  sorry

end ball_count_proof_l1423_142344


namespace anthony_pizza_fraction_l1423_142376

theorem anthony_pizza_fraction (total_slices : ℕ) (whole_slice : ℚ) (shared_slice : ℚ) : 
  total_slices = 16 → 
  whole_slice = 1 / total_slices → 
  shared_slice = 1 / (2 * total_slices) → 
  whole_slice + 2 * shared_slice = 1 / 8 := by
  sorry

end anthony_pizza_fraction_l1423_142376


namespace ratio_of_sum_and_difference_l1423_142309

theorem ratio_of_sum_and_difference (x y : ℝ) : 
  x > 0 → y > 0 → x > y → x + y = 7 * (x - y) → x / y = 4 / 3 := by
  sorry

end ratio_of_sum_and_difference_l1423_142309


namespace product_of_digits_less_than_number_l1423_142378

def digits (n : ℕ) : List ℕ :=
  if n < 10 then [n] else (n % 10) :: digits (n / 10)

def digit_product (n : ℕ) : ℕ :=
  (digits n).prod

theorem product_of_digits_less_than_number (N : ℕ) (h : N > 9) :
  digit_product N < N :=
sorry

end product_of_digits_less_than_number_l1423_142378


namespace sum_digits_base_seven_999_l1423_142312

/-- Represents a number in base 7 as a list of digits (least significant digit first) -/
def BaseSevenRepresentation := List Nat

/-- Converts a natural number to its base 7 representation -/
def toBaseSeven (n : Nat) : BaseSevenRepresentation :=
  sorry

/-- Computes the sum of digits in a base 7 representation -/
def sumDigitsBaseSeven (rep : BaseSevenRepresentation) : Nat :=
  sorry

theorem sum_digits_base_seven_999 :
  sumDigitsBaseSeven (toBaseSeven 999) = 15 := by
  sorry

end sum_digits_base_seven_999_l1423_142312


namespace rational_identity_product_l1423_142326

theorem rational_identity_product (M₁ M₂ : ℝ) :
  (∀ x : ℝ, x ≠ 2 ∧ x ≠ 3 → (42 * x - 51) / (x^2 - 5*x + 6) = M₁ / (x - 2) + M₂ / (x - 3)) →
  M₁ * M₂ = -2981.25 := by sorry

end rational_identity_product_l1423_142326


namespace particle_and_account_max_l1423_142301

-- Define the elevation function
def elevation (t : ℝ) : ℝ := 150 * t - 15 * t^2 + 50

-- Define the account balance function
def accountBalance (t : ℝ) : ℝ := 1000 * (1 + 0.05 * t)

-- Theorem statement
theorem particle_and_account_max (t : ℝ) :
  (∀ s : ℝ, elevation s ≤ elevation t) →
  elevation t = 425 ∧ accountBalance (t / 12) = 1020.83 := by
  sorry


end particle_and_account_max_l1423_142301


namespace quadrilateral_division_l1423_142321

/-- A triangle is a type representing a geometric triangle. -/
structure Triangle

/-- A quadrilateral is a type representing a geometric quadrilateral. -/
structure Quadrilateral

/-- Represents the concept of dividing a shape into equal parts. -/
def can_be_divided_into (n : ℕ) (T : Type) : Prop :=
  ∃ (parts : Fin n → T), ∀ i j : Fin n, i ≠ j → parts i ≠ parts j

/-- Any triangle can be divided into 4 equal triangles. -/
axiom triangle_division : can_be_divided_into 4 Triangle

/-- The main theorem: there exists a quadrilateral that can be divided into 7 equal triangles. -/
theorem quadrilateral_division : ∃ (Q : Quadrilateral), can_be_divided_into 7 Triangle :=
sorry

end quadrilateral_division_l1423_142321


namespace kris_herbert_age_difference_l1423_142333

/-- The age difference between two people --/
def age_difference (age1 : ℕ) (age2 : ℕ) : ℕ := 
  if age1 ≥ age2 then age1 - age2 else age2 - age1

/-- Theorem: The age difference between Kris and Herbert is 10 years --/
theorem kris_herbert_age_difference : 
  let kris_age : ℕ := 24
  let herbert_age_next_year : ℕ := 15
  let herbert_age : ℕ := herbert_age_next_year - 1
  age_difference kris_age herbert_age = 10 := by
  sorry

end kris_herbert_age_difference_l1423_142333


namespace variance_of_scores_l1423_142395

def scores : List ℝ := [87, 90, 90, 91, 91, 94, 94]

theorem variance_of_scores : 
  let n : ℕ := scores.length
  let mean : ℝ := scores.sum / n
  let variance : ℝ := (scores.map (λ x => (x - mean)^2)).sum / n
  variance = 36/7 := by
  sorry

end variance_of_scores_l1423_142395


namespace equation_positive_root_implies_m_equals_one_l1423_142318

-- Define the equation
def equation (x m : ℝ) : Prop :=
  (x - 4) / (x - 3) - m - 4 = m / (3 - x)

-- Define the theorem
theorem equation_positive_root_implies_m_equals_one :
  (∃ x : ℝ, x > 0 ∧ equation x m) → m = 1 := by
  sorry

end equation_positive_root_implies_m_equals_one_l1423_142318


namespace find_correct_divisor_l1423_142320

theorem find_correct_divisor (X D : ℕ) (h1 : X % D = 0) (h2 : X / (D + 12) = 70) (h3 : X / D = 40) : D = 28 := by
  sorry

end find_correct_divisor_l1423_142320


namespace normal_distribution_probability_l1423_142337

/-- A random variable following a normal distribution with mean 1 and standard deviation σ -/
def ξ (σ : ℝ) : Type := Unit

/-- The probability that ξ is less than a given value -/
def P_less_than (σ : ℝ) (x : ℝ) : ℝ := sorry

/-- The theorem stating that if P(ξ < 0) = 0.3, then P(ξ < 2) = 0.7 for ξ ~ N(1, σ²) -/
theorem normal_distribution_probability (σ : ℝ) (h : P_less_than σ 0 = 0.3) :
  P_less_than σ 2 = 0.7 := by sorry

end normal_distribution_probability_l1423_142337


namespace vector_properties_l1423_142397

def a : ℝ × ℝ := (2, 1)
def b : ℝ × ℝ := (1, -1)
def c (m n : ℝ) : ℝ × ℝ := (m - 2, n)

variable (m n : ℝ)
variable (hm : m > 0)
variable (hn : n > 0)

theorem vector_properties :
  (a.1 * b.1 + a.2 * b.2 = 1) ∧
  ((a.1 - b.1) * (c m n).1 + (a.2 - b.2) * (c m n).2 = 0 → m + 2*n = 2) := by
  sorry

end vector_properties_l1423_142397


namespace triangle_altitude_bound_l1423_142328

/-- For any triangle with perimeter 2, there exists at least one altitude that is less than or equal to 1/√3. -/
theorem triangle_altitude_bound (a b c : ℝ) (h_perimeter : a + b + c = 2) (h_positive : 0 < a ∧ 0 < b ∧ 0 < c) :
  ∃ h : ℝ, h ≤ 1 / Real.sqrt 3 ∧ (h = 2 * Real.sqrt ((1 - a) * (1 - b) * (1 - c)) / a ∨
                                  h = 2 * Real.sqrt ((1 - a) * (1 - b) * (1 - c)) / b ∨
                                  h = 2 * Real.sqrt ((1 - a) * (1 - b) * (1 - c)) / c) :=
by sorry


end triangle_altitude_bound_l1423_142328


namespace probability_two_defective_approx_l1423_142331

/-- The probability of selecting two defective smartphones from a shipment -/
def probability_two_defective (total : ℕ) (defective : ℕ) : ℚ :=
  (defective : ℚ) / total * ((defective - 1) : ℚ) / (total - 1)

/-- Theorem stating the probability of selecting two defective smartphones -/
theorem probability_two_defective_approx :
  ∃ (ε : ℚ), ε > 0 ∧ ε < 1/1000 ∧ 
  abs (probability_two_defective 220 84 - 1447/10000) < ε :=
sorry

end probability_two_defective_approx_l1423_142331


namespace quadratic_solution_l1423_142398

theorem quadratic_solution : ∃ x : ℚ, x > 0 ∧ 5 * x^2 + 9 * x - 18 = 0 ∧ x = 6/5 := by
  sorry

end quadratic_solution_l1423_142398


namespace rectangle_perimeter_l1423_142314

/-- Given a rectangle with one side of length 18 and the sum of its area and perimeter
    equal to 2016, prove that its perimeter is 234. -/
theorem rectangle_perimeter (a : ℝ) : 
  a > 0 → 
  18 * a + 2 * (18 + a) = 2016 → 
  2 * (18 + a) = 234 :=
by
  sorry

end rectangle_perimeter_l1423_142314


namespace solve_equation_l1423_142388

theorem solve_equation (x : ℝ) (n : ℝ) (h1 : 5 / (n + 1 / x) = 1) (h2 : x = 1) : n = 4 := by
  sorry

end solve_equation_l1423_142388


namespace dot_product_range_l1423_142356

-- Define the triangle ABC
structure Triangle :=
  (A B C : ℝ × ℝ)
  (angle_BAC : Real)
  (length_AB : Real)
  (length_AC : Real)

-- Define a point D on side BC
def PointOnBC (triangle : Triangle) := 
  {D : ℝ × ℝ | ∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ D = (1 - t) • triangle.B + t • triangle.C}

-- Define the dot product of two 2D vectors
def dot_product (v w : ℝ × ℝ) : ℝ :=
  v.1 * w.1 + v.2 * w.2

-- Theorem statement
theorem dot_product_range (triangle : Triangle) 
  (h1 : triangle.angle_BAC = 2*π/3)  -- 120° in radians
  (h2 : triangle.length_AB = 2)
  (h3 : triangle.length_AC = 1) :
  ∀ D ∈ PointOnBC triangle, 
    -5 ≤ dot_product (D - triangle.A) (triangle.C - triangle.B) ∧ 
    dot_product (D - triangle.A) (triangle.C - triangle.B) ≤ 0 :=
sorry

end dot_product_range_l1423_142356


namespace parallel_lines_intersection_problem_solution_l1423_142394

/-- Given two sets of parallel lines intersecting each other, 
    calculate the number of lines in the second set based on 
    the number of parallelograms formed -/
theorem parallel_lines_intersection (first_set : ℕ) (parallelograms : ℕ) 
  (h1 : first_set = 5) 
  (h2 : parallelograms = 280) : 
  ∃ (second_set : ℕ), second_set * (first_set - 1) = parallelograms := by
  sorry

/-- The specific case for the given problem -/
theorem problem_solution : 
  ∃ (second_set : ℕ), second_set * 4 = 280 ∧ second_set = 71 := by
  sorry

end parallel_lines_intersection_problem_solution_l1423_142394


namespace quadratic_inequality_l1423_142353

def f (x : ℝ) : ℝ := (x - 2)^2 + 1

theorem quadratic_inequality : f 2 < f 3 ∧ f 3 < f 0 := by
  sorry

end quadratic_inequality_l1423_142353


namespace x_squared_in_set_l1423_142346

theorem x_squared_in_set (x : ℝ) : x^2 ∈ ({0, 1, x} : Set ℝ) → x = -1 := by
  sorry

end x_squared_in_set_l1423_142346


namespace car_speed_problem_l1423_142352

theorem car_speed_problem (d : ℝ) (v : ℝ) (h1 : d > 0) (h2 : v > 0) :
  let t := d / v
  let return_time := 2 * t
  let total_distance := 2 * d
  let total_time := t + return_time
  (total_distance / total_time = 30) → v = 45 := by
  sorry

end car_speed_problem_l1423_142352


namespace gcd_15225_20335_35475_l1423_142369

theorem gcd_15225_20335_35475 : Nat.gcd 15225 (Nat.gcd 20335 35475) = 5 := by
  sorry

end gcd_15225_20335_35475_l1423_142369


namespace first_12_average_l1423_142371

theorem first_12_average (total_count : Nat) (total_average : ℝ) (last_12_average : ℝ) (result_13 : ℝ) :
  total_count = 25 →
  total_average = 18 →
  last_12_average = 20 →
  result_13 = 90 →
  (((total_count : ℝ) * total_average - 12 * last_12_average - result_13) / 12 : ℝ) = 10 := by
  sorry

end first_12_average_l1423_142371


namespace power_of_seven_mod_eight_l1423_142340

theorem power_of_seven_mod_eight : 7^123 % 8 = 7 := by
  sorry

end power_of_seven_mod_eight_l1423_142340


namespace library_books_count_l1423_142364

theorem library_books_count (old_books : ℕ) 
  (h1 : old_books + 300 + 400 - 200 = 1000) : old_books = 500 := by
  sorry

end library_books_count_l1423_142364


namespace factorial_equation_solution_l1423_142380

theorem factorial_equation_solution : ∃ (n : ℕ), n > 0 ∧ (n + 1).factorial + (n + 3).factorial = n.factorial * 1190 ∧ n = 8 := by
  sorry

end factorial_equation_solution_l1423_142380


namespace expression_simplification_l1423_142358

theorem expression_simplification :
  (((1 + 2 + 3 + 6) / 3) + ((3 * 6 + 9) / 4)) = 43 / 4 := by
  sorry

end expression_simplification_l1423_142358


namespace complex_power_abs_one_l1423_142396

theorem complex_power_abs_one : 
  Complex.abs ((1/2 : ℂ) + (Complex.I * (Real.sqrt 3)/2))^8 = 1 := by
  sorry

end complex_power_abs_one_l1423_142396


namespace frog_vertical_side_probability_l1423_142313

/-- Represents a point in the 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents the square area -/
def Square : Set Point := {p | 0 ≤ p.x ∧ p.x ≤ 5 ∧ 0 ≤ p.y ∧ p.y ≤ 5}

/-- Represents a vertical side of the square -/
def VerticalSide : Set Point := {p | p.x = 0 ∨ p.x = 5}

/-- Represents a single jump of the frog -/
def Jump (p : Point) : Set Point :=
  {q | (q.x = p.x ∧ (q.y = p.y + 1 ∨ q.y = p.y - 1)) ∨
       (q.y = p.y ∧ (q.x = p.x + 1 ∨ q.x = p.x - 1))}

/-- The probability of ending on a vertical side given the starting point -/
noncomputable def ProbVerticalSide (p : Point) : ℝ := sorry

/-- The theorem stating the probability of ending on a vertical side is 1/2 -/
theorem frog_vertical_side_probability :
  ProbVerticalSide ⟨2, 2⟩ = 1/2 := by sorry

end frog_vertical_side_probability_l1423_142313


namespace perpendicular_vectors_l1423_142302

/-- Given two vectors a and b in ℝ², where a is perpendicular to (a + b), prove that the y-coordinate of b is -3. -/
theorem perpendicular_vectors (a b : ℝ × ℝ) (h : a = (2, 1)) (h' : b.1 = -1) 
  (h'' : (a.1 * (a.1 + b.1) + a.2 * (a.2 + b.2) : ℝ) = 0) : 
  b.2 = -3 := by
  sorry

end perpendicular_vectors_l1423_142302


namespace roses_to_grandmother_l1423_142343

/-- Given that Ian had a certain number of roses and distributed them in a specific way,
    this theorem proves how many roses he gave to his grandmother. -/
theorem roses_to_grandmother (total : ℕ) (to_mother : ℕ) (to_sister : ℕ) (kept : ℕ) 
    (h1 : total = 20)
    (h2 : to_mother = 6)
    (h3 : to_sister = 4)
    (h4 : kept = 1) :
    total - (to_mother + to_sister + kept) = 9 := by
  sorry

end roses_to_grandmother_l1423_142343


namespace sum_of_a_and_b_l1423_142385

-- Define the solution set
def solution_set : Set ℝ := Set.Ioi 4 ∪ Set.Iic 1

-- Define the inequality
def inequality (a b x : ℝ) : Prop := (x - a) / (x - b) > 0

theorem sum_of_a_and_b (a b : ℝ) :
  (∀ x, x ∈ solution_set ↔ inequality a b x) →
  a + b = 5 := by sorry

end sum_of_a_and_b_l1423_142385


namespace max_product_fg_l1423_142377

-- Define the functions f and g
def f : ℝ → ℝ := sorry
def g : ℝ → ℝ := sorry

-- State the conditions on the ranges of f and g
axiom f_range : ∀ x, -3 ≤ f x ∧ f x ≤ 4
axiom g_range : ∀ x, -3 ≤ g x ∧ g x ≤ 2

-- Theorem stating the maximum value of f(x) · g(x)
theorem max_product_fg : 
  ∃ x : ℝ, ∀ y : ℝ, f y * g y ≤ f x * g x ∧ f x * g x = 12 :=
sorry

end max_product_fg_l1423_142377


namespace initial_conditions_squares_in_figure_100_l1423_142372

/-- The number of squares in figure n -/
def f (n : ℕ) : ℕ := 3 * n^2 + 2 * n + 1

/-- The sequence satisfies the given initial conditions -/
theorem initial_conditions :
  f 0 = 1 ∧ f 1 = 6 ∧ f 2 = 17 ∧ f 3 = 34 := by sorry

/-- The number of squares in figure 100 is 30201 -/
theorem squares_in_figure_100 :
  f 100 = 30201 := by sorry

end initial_conditions_squares_in_figure_100_l1423_142372


namespace dividend_divisor_quotient_l1423_142310

theorem dividend_divisor_quotient (x y z : ℚ) :
  x = y * z + 15 ∧ y = 25 ∧ 3 * x - 4 * y + 2 * z = 0 →
  x = 230 / 7 ∧ z = 5 / 7 := by
sorry

end dividend_divisor_quotient_l1423_142310


namespace bunny_teddy_ratio_l1423_142342

def initial_teddies : ℕ := 5
def koala_bears : ℕ := 1
def additional_teddies_per_bunny : ℕ := 2
def total_mascots : ℕ := 51

def bunnies : ℕ := (total_mascots - initial_teddies - koala_bears) / (additional_teddies_per_bunny + 1)

theorem bunny_teddy_ratio :
  bunnies / initial_teddies = 3 ∧ bunnies % initial_teddies = 0 := by
  sorry

end bunny_teddy_ratio_l1423_142342


namespace u_eq_complement_a_union_b_l1423_142382

/-- The universal set U -/
def U : Finset Nat := {1, 2, 3, 4, 5, 7}

/-- Set A -/
def A : Finset Nat := {4, 7}

/-- Set B -/
def B : Finset Nat := {1, 3, 4, 7}

/-- Theorem stating that U is equal to the union of the complement of A in U and B -/
theorem u_eq_complement_a_union_b : U = (U \ A) ∪ B := by sorry

end u_eq_complement_a_union_b_l1423_142382


namespace log_weight_l1423_142368

theorem log_weight (log_length : ℕ) (weight_per_foot : ℕ) (cut_pieces : ℕ) : 
  log_length = 20 → 
  weight_per_foot = 150 → 
  cut_pieces = 2 → 
  (log_length / cut_pieces) * weight_per_foot = 1500 :=
by sorry

end log_weight_l1423_142368


namespace movement_increases_dimension_l1423_142363

/-- Dimension of geometric objects -/
inductive GeometricDimension
  | point
  | line
  | surface
  deriving Repr

/-- Function that returns the dimension of the object formed by moving an object of a given dimension -/
def dimensionAfterMovement (d : GeometricDimension) : GeometricDimension :=
  match d with
  | GeometricDimension.point => GeometricDimension.line
  | GeometricDimension.line => GeometricDimension.surface
  | GeometricDimension.surface => GeometricDimension.surface

/-- Theorem stating that moving a point forms a line and moving a line forms a surface -/
theorem movement_increases_dimension :
  (dimensionAfterMovement GeometricDimension.point = GeometricDimension.line) ∧
  (dimensionAfterMovement GeometricDimension.line = GeometricDimension.surface) :=
by sorry

end movement_increases_dimension_l1423_142363


namespace intersection_M_N_l1423_142306

-- Define the sets M and N
def M : Set ℝ := { x | x < 1 }
def N : Set ℝ := { x | 0 < x ∧ x < 2 }

-- State the theorem
theorem intersection_M_N : M ∩ N = { x | 0 < x ∧ x < 1 } := by sorry

end intersection_M_N_l1423_142306


namespace find_y_value_l1423_142307

theorem find_y_value (x y z : ℚ) : 
  x + y + z = 150 → x + 8 = y - 8 → x + 8 = 4 * z → y = 224 / 3 := by
  sorry

end find_y_value_l1423_142307


namespace total_houses_l1423_142332

theorem total_houses (dogs cats both : ℕ) 
  (h_dogs : dogs = 40)
  (h_cats : cats = 30)
  (h_both : both = 10) :
  dogs + cats - both = 60 := by
  sorry

end total_houses_l1423_142332


namespace perpendicular_condition_l1423_142379

/-- Represents a line in the form Ax + By + C = 0 -/
structure Line where
  A : ℝ
  B : ℝ
  C : ℝ

/-- Check if two lines are perpendicular -/
def are_perpendicular (l1 l2 : Line) : Prop :=
  l1.A * l2.A + l1.B * l2.B = 0

/-- The first line: 4x - (a+1)y + 9 = 0 -/
def line1 (a : ℝ) : Line :=
  { A := 4, B := -(a+1), C := 9 }

/-- The second line: (a^2-1)x - ay + 6 = 0 -/
def line2 (a : ℝ) : Line :=
  { A := a^2-1, B := -a, C := 6 }

/-- Statement: a = -1 is a sufficient but not necessary condition for the lines to be perpendicular -/
theorem perpendicular_condition :
  (∀ a : ℝ, a = -1 → are_perpendicular (line1 a) (line2 a)) ∧
  (∃ a : ℝ, a ≠ -1 ∧ are_perpendicular (line1 a) (line2 a)) := by
  sorry

end perpendicular_condition_l1423_142379


namespace complex_equation_solution_l1423_142381

theorem complex_equation_solution (n : ℝ) : 
  (1 : ℂ) / (1 + Complex.I) = (1 : ℂ) / 2 - n * Complex.I → n = 1 / 2 := by
sorry

end complex_equation_solution_l1423_142381


namespace tom_missed_no_games_l1423_142327

/-- The number of hockey games Tom missed this year -/
def games_missed_this_year (games_this_year games_last_year total_games : ℕ) : ℕ :=
  total_games - (games_this_year + games_last_year)

/-- Theorem: Tom missed 0 hockey games this year -/
theorem tom_missed_no_games :
  games_missed_this_year 4 9 13 = 0 := by
  sorry

end tom_missed_no_games_l1423_142327


namespace equation_solution_l1423_142311

theorem equation_solution : ∃! x : ℝ, x ≠ 1 ∧ (3 * x + 6) / (x^2 + 5 * x - 6) = (3 - x) / (x - 1) := by
  sorry

end equation_solution_l1423_142311


namespace solution_set_for_a_eq_1_range_of_a_for_f_always_greater_than_1_l1423_142362

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := |2 * x + 4| + |x - a|

-- Theorem for part I
theorem solution_set_for_a_eq_1 :
  {x : ℝ | f 1 x ≤ 5} = {x : ℝ | -8/3 ≤ x ∧ x ≤ 0} := by sorry

-- Theorem for part II
theorem range_of_a_for_f_always_greater_than_1 :
  ∀ a : ℝ, (∀ x : ℝ, f a x > 1) ↔ (a < -3 ∨ a > -1) := by sorry

end solution_set_for_a_eq_1_range_of_a_for_f_always_greater_than_1_l1423_142362


namespace deepak_age_l1423_142392

theorem deepak_age (rahul_age deepak_age : ℕ) : 
  (rahul_age : ℚ) / deepak_age = 4 / 3 →
  rahul_age + 6 = 18 →
  deepak_age = 9 := by
sorry

end deepak_age_l1423_142392


namespace man_daily_wage_l1423_142316

/-- The daily wage of a man -/
def M : ℝ := sorry

/-- The daily wage of a woman -/
def W : ℝ := sorry

/-- The total wages of 24 men and 16 women per day -/
def total_wages : ℝ := 11600

/-- The number of men -/
def num_men : ℕ := 24

/-- The number of women -/
def num_women : ℕ := 16

/-- The wages of 24 men and 16 women amount to Rs. 11600 per day -/
axiom wage_equation : num_men * M + num_women * W = total_wages

/-- Half the number of men and 37 women earn the same amount per day -/
axiom half_men_equation : (num_men / 2) * M + 37 * W = total_wages

theorem man_daily_wage : M = 350 := by sorry

end man_daily_wage_l1423_142316


namespace ellipse_eccentricity_l1423_142383

/-- The eccentricity of an ellipse with equation x²/4 + y²/9 = 1 is √5/3 -/
theorem ellipse_eccentricity : 
  let a : ℝ := 3
  let b : ℝ := 2
  let c : ℝ := Real.sqrt (a^2 - b^2)
  let e : ℝ := c / a
  e = Real.sqrt 5 / 3 := by sorry

end ellipse_eccentricity_l1423_142383


namespace projection_property_l1423_142305

def projection (v : ℝ × ℝ) : ℝ × ℝ := sorry

theorem projection_property :
  let p := projection
  p (3, 3) = (27/5, 9/5) →
  p (1, -1) = (3/5, 1/5) := by sorry

end projection_property_l1423_142305


namespace tangent_slope_sin_pi_over_four_l1423_142334

theorem tangent_slope_sin_pi_over_four :
  let f : ℝ → ℝ := fun x ↦ Real.sin x
  deriv f (π / 4) = Real.sqrt 2 / 2 := by
sorry

end tangent_slope_sin_pi_over_four_l1423_142334


namespace matrix_addition_theorem_l1423_142345

/-- Given matrices A and B, prove that C = 2A + B is equal to the expected result. -/
theorem matrix_addition_theorem (A B : Matrix (Fin 2) (Fin 2) ℤ) : 
  A = !![2, 1; 3, 4] → 
  B = !![0, -5; -1, 6] → 
  2 • A + B = !![4, -3; 5, 14] := by
  sorry

end matrix_addition_theorem_l1423_142345


namespace transformed_system_solution_l1423_142317

theorem transformed_system_solution 
  (a₁ a₂ b₁ b₂ c₁ c₂ : ℝ) 
  (h₁ : a₁ * 6 + b₁ * 3 = c₁)
  (h₂ : a₂ * 6 + b₂ * 3 = c₂) :
  (4 * a₁ * 22 + 3 * b₁ * 33 = 11 * c₁) ∧ 
  (4 * a₂ * 22 + 3 * b₂ * 33 = 11 * c₂) := by
sorry

end transformed_system_solution_l1423_142317


namespace cos_squared_difference_equals_sqrt3_over_2_l1423_142348

theorem cos_squared_difference_equals_sqrt3_over_2 :
  (Real.cos (π / 12))^2 - (Real.cos (5 * π / 12))^2 = Real.sqrt 3 / 2 := by
  sorry

end cos_squared_difference_equals_sqrt3_over_2_l1423_142348


namespace john_basketball_shots_l1423_142325

theorem john_basketball_shots 
  (initial_shots : ℕ) 
  (initial_percentage : ℚ) 
  (additional_shots : ℕ) 
  (new_percentage : ℚ) 
  (h1 : initial_shots = 30)
  (h2 : initial_percentage = 2/5)
  (h3 : additional_shots = 10)
  (h4 : new_percentage = 11/25) :
  (new_percentage * (initial_shots + additional_shots)).floor - 
  (initial_percentage * initial_shots).floor = 6 := by
sorry

end john_basketball_shots_l1423_142325


namespace consecutive_integers_average_l1423_142366

theorem consecutive_integers_average (a c : ℤ) : 
  (∀ k ∈ Finset.range 7, k + a > 0) →  -- Positive integers condition
  c = (Finset.sum (Finset.range 7) (λ k => k + a)) / 7 →  -- Average condition
  (Finset.sum (Finset.range 7) (λ k => k + c)) / 7 = a + 6 := by
  sorry

end consecutive_integers_average_l1423_142366


namespace complex_expression_equals_eight_l1423_142315

theorem complex_expression_equals_eight :
  (1 / Real.sqrt 0.04) + ((1 / Real.sqrt 27) ^ (1/3)) + 
  ((Real.sqrt 2 + 1)⁻¹) - (2 ^ (1/2)) + ((-2) ^ 0) = 8 := by
  sorry

end complex_expression_equals_eight_l1423_142315


namespace trajectory_equation_l1423_142355

/-- The trajectory of a point P(x,y) satisfying a specific condition with respect to fixed points M and N -/
theorem trajectory_equation (x y : ℝ) : 
  let M : ℝ × ℝ := (-2, 0)
  let N : ℝ × ℝ := (2, 0)
  let P : ℝ × ℝ := (x, y)
  let MN : ℝ × ℝ := (N.1 - M.1, N.2 - M.2)
  let MP : ℝ × ℝ := (P.1 - M.1, P.2 - M.2)
  let NP : ℝ × ℝ := (P.1 - N.1, P.2 - N.2)
  ‖MN‖ * ‖MP‖ + MN.1 * NP.1 + MN.2 * NP.2 = 0 →
  y^2 = -8*x := by
sorry


end trajectory_equation_l1423_142355


namespace share_in_ratio_l1423_142390

theorem share_in_ratio (total : ℕ) (ratio1 ratio2 ratio3 : ℕ) (h1 : total = 4320) (h2 : ratio1 = 2) (h3 : ratio2 = 4) (h4 : ratio3 = 6) :
  let sum_ratio := ratio1 + ratio2 + ratio3
  let share1 := total * ratio1 / sum_ratio
  share1 = 720 := by
sorry

end share_in_ratio_l1423_142390


namespace average_marks_l1423_142357

theorem average_marks (n : ℕ) (avg_five : ℝ) (sixth_mark : ℝ) (h1 : n = 6) (h2 : avg_five = 74) (h3 : sixth_mark = 62) :
  ((avg_five * (n - 1) + sixth_mark) / n : ℝ) = 72 := by
  sorry

end average_marks_l1423_142357


namespace ellipse_symmetric_points_m_range_l1423_142335

/-- An ellipse centered at the origin with right focus at (1,0) and one vertex at (0,√3) -/
def Ellipse : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1^2 / 4 + p.2^2 / 3 = 1}

/-- Two points are symmetric about the line y = x + m -/
def SymmetricAboutLine (p q : ℝ × ℝ) (m : ℝ) : Prop :=
  ∃ (t : ℝ), p.1 + q.1 = 2 * t ∧ p.2 + q.2 = 2 * (t + m)

theorem ellipse_symmetric_points_m_range :
  ∀ (m : ℝ),
    (∃ (p q : ℝ × ℝ), p ∈ Ellipse ∧ q ∈ Ellipse ∧ p ≠ q ∧ SymmetricAboutLine p q m) →
    -Real.sqrt 7 / 7 < m ∧ m < Real.sqrt 7 / 7 :=
by sorry

end ellipse_symmetric_points_m_range_l1423_142335


namespace valid_license_plates_count_l1423_142351

/-- Represents a license plate with 4 characters -/
structure LicensePlate :=
  (first : Char) (second : Char) (third : Nat) (fourth : Nat)

/-- Checks if a character is a letter (A-Z) -/
def isLetter (c : Char) : Bool :=
  'A' ≤ c ∧ c ≤ 'Z'

/-- Checks if a number is a single digit (0-9) -/
def isDigit (n : Nat) : Bool :=
  n < 10

/-- Checks if a license plate satisfies all conditions -/
def isValidLicensePlate (plate : LicensePlate) : Prop :=
  isLetter plate.first ∧
  isLetter plate.second ∧
  isDigit plate.third ∧
  isDigit plate.fourth ∧
  plate.third = plate.fourth ∧
  (plate.first.toNat = plate.third ∨ plate.second.toNat = plate.third)

/-- The number of valid license plates -/
def numValidLicensePlates : Nat :=
  (26 * 26) * 10

theorem valid_license_plates_count :
  numValidLicensePlates = 6760 :=
by sorry

end valid_license_plates_count_l1423_142351


namespace equation_solution_l1423_142308

theorem equation_solution : 
  ∃ (x₁ x₂ : ℝ), x₁ > 0 ∧ x₂ > 0 ∧ 
  x₁ = (40 + Real.sqrt 1636) / 2 ∧ 
  x₂ = (-20 + Real.sqrt 388) / 2 ∧
  ∀ (x : ℝ), x > 0 → 
  ((3 / 5) * (2 * x^2 - 2) = (x^2 - 40*x - 8) * (x^2 + 20*x + 4)) ↔ 
  (x = x₁ ∨ x = x₂) :=
by sorry

end equation_solution_l1423_142308


namespace circle_equation_k_value_l1423_142347

theorem circle_equation_k_value (k : ℝ) : 
  (∀ x y : ℝ, x^2 + 6*x + y^2 + 8*y - k = 0 ↔ (x + 3)^2 + (y + 4)^2 = 10^2) → 
  k = 75 := by
sorry

end circle_equation_k_value_l1423_142347


namespace prob_one_day_both_first_class_conditional_prob_both_first_class_expected_daily_profit_l1423_142323

-- Define the probabilities and costs
def p_first_class : ℝ := 0.5
def p_second_class : ℝ := 0.4
def cost_per_unit : ℝ := 2000
def price_first_class : ℝ := 10000
def price_second_class : ℝ := 8000
def loss_substandard : ℝ := 1000

-- Define the probability of both products being first-class in one day
def p_both_first_class : ℝ := p_first_class * p_first_class

-- Define the probability of exactly one first-class product in a day
def p_one_first_class : ℝ := 2 * p_first_class * (1 - p_first_class)

-- Define the daily profit function
def daily_profit (n_first n_second n_substandard : ℕ) : ℝ :=
  n_first * (price_first_class - cost_per_unit) +
  n_second * (price_second_class - cost_per_unit) +
  n_substandard * (-cost_per_unit - loss_substandard)

-- Theorem 1: Probability of exactly one day with both first-class products in three days
theorem prob_one_day_both_first_class :
  (3 : ℝ) * p_both_first_class * (1 - p_both_first_class)^2 = 27/64 := by sorry

-- Theorem 2: Conditional probability of both products being first-class given one is first-class
theorem conditional_prob_both_first_class :
  p_both_first_class / (p_both_first_class + p_one_first_class) = 1/3 := by sorry

-- Theorem 3: Expected daily profit
theorem expected_daily_profit :
  p_both_first_class * daily_profit 2 0 0 +
  p_one_first_class * daily_profit 1 1 0 +
  (p_second_class * p_second_class) * daily_profit 0 2 0 +
  (2 * p_first_class * (1 - p_first_class - p_second_class)) * daily_profit 1 0 1 +
  (2 * p_second_class * (1 - p_first_class - p_second_class)) * daily_profit 0 1 1 +
  ((1 - p_first_class - p_second_class) * (1 - p_first_class - p_second_class)) * daily_profit 0 0 2 = 12200 := by sorry

end prob_one_day_both_first_class_conditional_prob_both_first_class_expected_daily_profit_l1423_142323


namespace team_a_win_probabilities_l1423_142367

/-- Probability of Team A winning a single game -/
def p_a : ℝ := 0.6

/-- Probability of Team B winning a single game -/
def p_b : ℝ := 0.4

/-- Sum of probabilities for a single game is 1 -/
axiom prob_sum : p_a + p_b = 1

/-- Probability of Team A winning in a best-of-three format -/
def p_a_bo3 : ℝ := p_a^2 + 2 * p_a^2 * p_b

/-- Probability of Team A winning in a best-of-five format -/
def p_a_bo5 : ℝ := p_a^3 + 3 * p_a^3 * p_b + 6 * p_a^3 * p_b^2

/-- Theorem: Probabilities of Team A winning in best-of-three and best-of-five formats -/
theorem team_a_win_probabilities : 
  p_a_bo3 = 0.648 ∧ p_a_bo5 = 0.68256 :=
sorry

end team_a_win_probabilities_l1423_142367


namespace pillow_average_price_l1423_142319

/-- Given 4 pillows with an average cost of $5 and an additional pillow costing $10,
    prove that the average price of all 5 pillows is $6 -/
theorem pillow_average_price (n : ℕ) (avg_cost : ℚ) (additional_cost : ℚ) :
  n = 4 ∧ avg_cost = 5 ∧ additional_cost = 10 →
  ((n : ℚ) * avg_cost + additional_cost) / ((n : ℚ) + 1) = 6 := by
  sorry

end pillow_average_price_l1423_142319


namespace jerry_shelf_difference_l1423_142373

def shelf_difference (initial_action_figures : ℕ) (initial_books : ℕ) (added_books : ℕ) : ℕ :=
  initial_action_figures - (initial_books + added_books)

theorem jerry_shelf_difference :
  shelf_difference 7 2 4 = 1 := by
  sorry

end jerry_shelf_difference_l1423_142373


namespace middle_integer_of_consecutive_evens_l1423_142322

theorem middle_integer_of_consecutive_evens (n : ℕ) : 
  n > 0 ∧ n < 10 ∧ n % 2 = 0 ∧
  (n - 2) > 0 ∧ (n + 2) < 10 ∧
  (n - 2) + n + (n + 2) = ((n - 2) * n * (n + 2)) / 8 →
  n = 4 := by
sorry

end middle_integer_of_consecutive_evens_l1423_142322


namespace max_angle_B_in_arithmetic_sequence_triangle_l1423_142338

theorem max_angle_B_in_arithmetic_sequence_triangle :
  ∀ (a b c : ℝ) (A B C : ℝ),
    0 < a ∧ 0 < b ∧ 0 < c →
    0 < A ∧ 0 < B ∧ 0 < C →
    A + B + C = π →
    b^2 = a * c →  -- arithmetic sequence condition
    B ≤ π / 3 :=
by sorry

end max_angle_B_in_arithmetic_sequence_triangle_l1423_142338


namespace solution_set_inequality_l1423_142354

theorem solution_set_inequality (x : ℝ) :
  (x + 2) * (1 - x) > 0 ↔ -2 < x ∧ x < 1 := by sorry

end solution_set_inequality_l1423_142354


namespace same_number_of_digits_l1423_142359

/-- 
Given a natural number n, if k is the number of digits in 1974^n,
then 1974^n + 2^n < 10^k.
This implies that 1974^n and 1974^n + 2^n have the same number of digits.
-/
theorem same_number_of_digits (n : ℕ) : 
  let k := (Nat.log 10 (1974^n) + 1)
  1974^n + 2^n < 10^k := by
  sorry

end same_number_of_digits_l1423_142359


namespace xiaofang_english_score_l1423_142374

/-- Represents the scores of four subjects -/
structure Scores where
  chinese : ℝ
  math : ℝ
  english : ℝ
  science : ℝ

/-- The average score of four subjects is 88 -/
def avg_four (s : Scores) : Prop :=
  (s.chinese + s.math + s.english + s.science) / 4 = 88

/-- The average score of the first two subjects is 93 -/
def avg_first_two (s : Scores) : Prop :=
  (s.chinese + s.math) / 2 = 93

/-- The average score of the last three subjects is 87 -/
def avg_last_three (s : Scores) : Prop :=
  (s.math + s.english + s.science) / 3 = 87

/-- Xiaofang's English test score is 95 -/
theorem xiaofang_english_score (s : Scores) 
  (h1 : avg_four s) (h2 : avg_first_two s) (h3 : avg_last_three s) : 
  s.english = 95 := by
  sorry

end xiaofang_english_score_l1423_142374


namespace terminating_decimal_count_l1423_142303

theorem terminating_decimal_count : 
  (Finset.filter (fun n : ℕ => n % 13 = 0) (Finset.range 543)).card = 41 := by
  sorry

end terminating_decimal_count_l1423_142303


namespace three_four_five_triangle_l1423_142330

/-- Triangle inequality theorem: the sum of the lengths of any two sides of a triangle
    must be greater than the length of the remaining side. -/
def satisfies_triangle_inequality (a b c : ℝ) : Prop :=
  a + b > c ∧ b + c > a ∧ c + a > b

/-- A function to check if three given lengths can form a triangle. -/
def can_form_triangle (a b c : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ satisfies_triangle_inequality a b c

/-- Theorem stating that the lengths 3, 4, and 5 can form a triangle. -/
theorem three_four_five_triangle :
  can_form_triangle 3 4 5 := by
  sorry


end three_four_five_triangle_l1423_142330


namespace hyperbola_focal_coordinates_l1423_142389

/-- The hyperbola equation -/
def hyperbola_equation (x y : ℝ) : Prop := x^2 / 16 - y^2 / 9 = 1

/-- The focal coordinates of the hyperbola -/
def focal_coordinates : Set (ℝ × ℝ) := {(-5, 0), (5, 0)}

/-- Theorem: The focal coordinates of the hyperbola x^2/16 - y^2/9 = 1 are (-5, 0) and (5, 0) -/
theorem hyperbola_focal_coordinates :
  ∀ (x y : ℝ), hyperbola_equation x y → (x, y) ∈ focal_coordinates :=
sorry

end hyperbola_focal_coordinates_l1423_142389


namespace backpack_price_relationship_l1423_142336

/-- Represents the relationship between backpack purchases and prices -/
theorem backpack_price_relationship (x : ℝ) 
  (h1 : x > 0) -- Ensure x is positive for division
  (h2 : 810 > 0) -- Total spent on type A is positive
  (h3 : 600 > 0) -- Total spent on type B is positive
  (h4 : x + 20 > 0) -- Ensure denominator is positive
  : 810 / (x + 20) = (600 / x) * (1 - 10 / 100) :=
by sorry

end backpack_price_relationship_l1423_142336


namespace fraction_equality_l1423_142386

theorem fraction_equality (a b : ℝ) (h : a / b = 4 / 3) :
  (3 * a + 2 * b) / (3 * a - 2 * b) = 3 := by
  sorry

end fraction_equality_l1423_142386


namespace savings_equality_l1423_142341

theorem savings_equality (your_initial : ℕ) (friend_initial : ℕ) (friend_weekly : ℕ) (weeks : ℕ) 
  (h1 : your_initial = 160)
  (h2 : friend_initial = 210)
  (h3 : friend_weekly = 5)
  (h4 : weeks = 25) :
  ∃ your_weekly : ℕ, 
    your_initial + weeks * your_weekly = friend_initial + weeks * friend_weekly ∧ 
    your_weekly = 7 := by
  sorry

end savings_equality_l1423_142341


namespace coefficient_d_nonzero_l1423_142339

/-- A polynomial of degree 5 -/
def P (a b c d e : ℝ) (x : ℝ) : ℝ := x^5 + a*x^4 + b*x^3 + c*x^2 + d*x + e

/-- The statement that P has five distinct x-intercepts -/
def has_five_distinct_roots (a b c d e : ℝ) : Prop :=
  ∃ (r₁ r₂ r₃ r₄ r₅ : ℝ), (r₁ ≠ r₂ ∧ r₁ ≠ r₃ ∧ r₁ ≠ r₄ ∧ r₁ ≠ r₅ ∧
                           r₂ ≠ r₃ ∧ r₂ ≠ r₄ ∧ r₂ ≠ r₅ ∧
                           r₃ ≠ r₄ ∧ r₃ ≠ r₅ ∧
                           r₄ ≠ r₅) ∧
                          (∀ x : ℝ, P a b c d e x = 0 ↔ x = r₁ ∨ x = r₂ ∨ x = r₃ ∨ x = r₄ ∨ x = r₅)

theorem coefficient_d_nonzero (a b c d e : ℝ) 
  (h1 : has_five_distinct_roots a b c d e)
  (h2 : P a b c d e 0 = 0) : -- One root is at (0,0)
  d ≠ 0 := by
  sorry

end coefficient_d_nonzero_l1423_142339


namespace cards_kept_away_is_seven_l1423_142370

/-- The number of cards in a standard deck -/
def standard_deck : ℕ := 52

/-- The number of cards used in the game -/
def cards_used : ℕ := 45

/-- The number of cards kept away -/
def cards_kept_away : ℕ := standard_deck - cards_used

theorem cards_kept_away_is_seven : cards_kept_away = 7 := by
  sorry

end cards_kept_away_is_seven_l1423_142370


namespace cos_270_degrees_l1423_142399

theorem cos_270_degrees : Real.cos (270 * π / 180) = 0 := by sorry

end cos_270_degrees_l1423_142399
