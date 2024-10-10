import Mathlib

namespace factor_polynomial_l1393_139378

theorem factor_polynomial (x y : ℝ) : -(2*x - y) * (2*x + y) = -4*x^2 + y^2 := by
  sorry

end factor_polynomial_l1393_139378


namespace num_workers_is_500_l1393_139367

/-- The number of workers who raised money by equal contribution -/
def num_workers : ℕ := sorry

/-- The original contribution amount per worker in rupees -/
def contribution_per_worker : ℕ := sorry

/-- The total contribution is 300,000 rupees -/
axiom total_contribution : num_workers * contribution_per_worker = 300000

/-- If each worker contributed 50 rupees extra, the total would be 325,000 rupees -/
axiom total_with_extra : num_workers * (contribution_per_worker + 50) = 325000

/-- Theorem: The number of workers is 500 -/
theorem num_workers_is_500 : num_workers = 500 := by sorry

end num_workers_is_500_l1393_139367


namespace unique_496_consecutive_sum_l1393_139365

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

end unique_496_consecutive_sum_l1393_139365


namespace ab_value_l1393_139339

theorem ab_value (a b : ℝ) (h : (a - 2)^2 + |b + 3| = 0) : a * b = -6 := by
  sorry

end ab_value_l1393_139339


namespace bus_driver_max_regular_hours_l1393_139342

/-- Represents the problem of finding the maximum regular hours for a bus driver -/
theorem bus_driver_max_regular_hours 
  (regular_rate : ℝ) 
  (overtime_rate_factor : ℝ) 
  (total_compensation : ℝ) 
  (total_hours : ℝ) 
  (h1 : regular_rate = 16)
  (h2 : overtime_rate_factor = 1.75)
  (h3 : total_compensation = 1116)
  (h4 : total_hours = 57) :
  ∃ (max_regular_hours : ℝ),
    max_regular_hours * regular_rate + 
    (total_hours - max_regular_hours) * (regular_rate * overtime_rate_factor) = 
    total_compensation ∧ 
    max_regular_hours = 40 :=
by sorry

end bus_driver_max_regular_hours_l1393_139342


namespace toy_bridge_weight_l1393_139311

theorem toy_bridge_weight (total_weight : ℕ) (num_full_cans : ℕ) (soda_weight : ℕ) (empty_can_weight : ℕ) :
  total_weight = 88 →
  num_full_cans = 6 →
  soda_weight = 12 →
  empty_can_weight = 2 →
  (num_full_cans * (soda_weight + empty_can_weight) + (total_weight - num_full_cans * (soda_weight + empty_can_weight))) / empty_can_weight = 2 :=
by sorry

end toy_bridge_weight_l1393_139311


namespace silver_division_representation_l1393_139324

/-- Represents the problem of dividing silver among guests -/
structure SilverDivision where
  guests : ℕ      -- number of guests
  silver : ℕ      -- total amount of silver in taels

/-- The conditions of the silver division problem are satisfied -/
def satisfiesConditions (sd : SilverDivision) : Prop :=
  (7 * sd.guests = sd.silver - 4) ∧ 
  (9 * sd.guests = sd.silver + 8)

/-- The system of equations correctly represents the silver division problem -/
theorem silver_division_representation (sd : SilverDivision) : 
  satisfiesConditions sd ↔ 
  (∃ x y : ℕ, 
    sd.guests = x ∧ 
    sd.silver = y ∧ 
    7 * x = y - 4 ∧ 
    9 * x = y + 8) :=
sorry

end silver_division_representation_l1393_139324


namespace division_problem_l1393_139363

theorem division_problem (n : ℤ) : 
  (n / 6 = 124 ∧ n % 6 = 4) → ((n + 24) / 8 : ℚ) = 96.5 := by
  sorry

end division_problem_l1393_139363


namespace hash_eight_two_l1393_139364

-- Define the # operation
def hash (a b : ℝ) : ℝ := (a + b)^3 * (a - b)

-- Theorem statement
theorem hash_eight_two : hash 8 2 = 6000 := by
  sorry

end hash_eight_two_l1393_139364


namespace joyce_basketball_shots_l1393_139328

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

end joyce_basketball_shots_l1393_139328


namespace larger_number_proof_l1393_139361

theorem larger_number_proof (a b : ℕ+) : 
  (Nat.gcd a b = 40) →
  (Nat.lcm a b = 6600) →
  ((a = 40 * 11 ∧ b = 40 * 15) ∨ (a = 40 * 15 ∧ b = 40 * 11)) →
  max a b = 600 := by
sorry

end larger_number_proof_l1393_139361


namespace no_rational_solutions_l1393_139356

theorem no_rational_solutions (m : ℕ+) : ¬∃ (x : ℚ), m * x^2 + 40 * x + m = 0 := by
  sorry

end no_rational_solutions_l1393_139356


namespace election_winner_percentage_l1393_139349

theorem election_winner_percentage : 
  ∀ (total_votes : ℕ) (winner_votes loser_votes : ℕ),
  winner_votes = 864 →
  winner_votes - loser_votes = 288 →
  total_votes = winner_votes + loser_votes →
  (winner_votes : ℚ) / (total_votes : ℚ) = 3/5 :=
by sorry

end election_winner_percentage_l1393_139349


namespace plane_equation_proof_l1393_139312

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

end plane_equation_proof_l1393_139312


namespace spice_difference_l1393_139304

def cinnamon : ℝ := 0.6666666666666666
def nutmeg : ℝ := 0.5
def ginger : ℝ := 0.4444444444444444

def total_difference : ℝ := |cinnamon - nutmeg| + |nutmeg - ginger| + |cinnamon - ginger|

theorem spice_difference : total_difference = 0.4444444444444444 := by sorry

end spice_difference_l1393_139304


namespace g_of_5_equals_15_l1393_139370

-- Define the function g
def g (x : ℝ) : ℝ := x^2 - 2*x

-- State the theorem
theorem g_of_5_equals_15 : g 5 = 15 := by sorry

end g_of_5_equals_15_l1393_139370


namespace complement_of_union_A_B_l1393_139380

-- Define the sets A and B
def A : Set ℝ := {x | ∃ y, y = Real.log x}
def B : Set ℝ := {x | -7 < 2 + 3*x ∧ 2 + 3*x < 5}

-- State the theorem
theorem complement_of_union_A_B :
  (Set.univ : Set ℝ) \ (A ∪ B) = {x | x ≤ -3} := by sorry

end complement_of_union_A_B_l1393_139380


namespace cos_240_degrees_l1393_139357

theorem cos_240_degrees : Real.cos (240 * π / 180) = -1/2 := by
  sorry

end cos_240_degrees_l1393_139357


namespace fraction_of_pet_owners_l1393_139387

/-- Proves that the fraction of freshmen and sophomores who own a pet is 1/5 -/
theorem fraction_of_pet_owners (total_students : ℕ) (freshmen_sophomores : ℕ) (no_pet : ℕ) :
  total_students = 400 →
  freshmen_sophomores = total_students / 2 →
  no_pet = 160 →
  (freshmen_sophomores - no_pet) / freshmen_sophomores = 1 / 5 := by
  sorry

end fraction_of_pet_owners_l1393_139387


namespace circle_center_radius_sum_l1393_139379

/-- Given a circle C with equation x^2 + 12y + 57 = -y^2 - 10x, 
    prove that the sum of its center coordinates and radius is -9 -/
theorem circle_center_radius_sum (x y : ℝ) :
  (∃ (a b r : ℝ), 
    (∀ x y : ℝ, x^2 + 12*y + 57 = -y^2 - 10*x ↔ (x - a)^2 + (y - b)^2 = r^2) →
    a + b + r = -9) := by
  sorry

end circle_center_radius_sum_l1393_139379


namespace comparison_theorem_l1393_139377

theorem comparison_theorem :
  (-3/4 : ℚ) > -4/5 ∧ -(-3) > -|(-3)| := by sorry

end comparison_theorem_l1393_139377


namespace arithmetic_sequence_problem_l1393_139396

/-- An arithmetic sequence is a sequence where the difference between any two consecutive terms is constant. -/
def ArithmeticSequence (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, a (n + 1) - a n = d

/-- Given an arithmetic sequence a where a₄ = 1 and a₇ + a₉ = 16, prove that a₁₂ = 15 -/
theorem arithmetic_sequence_problem (a : ℕ → ℚ) 
    (h_arith : ArithmeticSequence a) 
    (h_a4 : a 4 = 1) 
    (h_sum : a 7 + a 9 = 16) : 
  a 12 = 15 := by
sorry


end arithmetic_sequence_problem_l1393_139396


namespace rafael_monday_hours_l1393_139321

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

end rafael_monday_hours_l1393_139321


namespace min_x_prime_factorization_sum_l1393_139369

theorem min_x_prime_factorization_sum (x y : ℕ+) (h : 5 * x^7 = 13 * y^11) :
  ∃ (a b c d : ℕ),
    (∀ x' : ℕ+, 5 * x'^7 = 13 * y^11 → x' ≥ x) →
    x = a^c * b^d ∧
    Prime a ∧ Prime b ∧
    a + b + c + d = 62 := by
  sorry

end min_x_prime_factorization_sum_l1393_139369


namespace distribute_graduates_eq_90_l1393_139360

/-- The number of ways to evenly distribute 6 graduates to 3 schools -/
def distribute_graduates : ℕ :=
  Nat.choose 6 2 * Nat.choose 4 2 * Nat.choose 2 2

/-- Theorem stating that the number of ways to distribute graduates is 90 -/
theorem distribute_graduates_eq_90 : distribute_graduates = 90 := by
  sorry

end distribute_graduates_eq_90_l1393_139360


namespace calculate_premium_rate_l1393_139301

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

end calculate_premium_rate_l1393_139301


namespace car_features_l1393_139317

theorem car_features (total : ℕ) (steering : ℕ) (windows : ℕ) (both : ℕ) :
  total = 65 →
  steering = 45 →
  windows = 25 →
  both = 17 →
  total - (steering + windows - both) = 12 := by
sorry

end car_features_l1393_139317


namespace smallest_special_number_proof_l1393_139325

/-- A function that returns true if a natural number uses exactly four different digits -/
def uses_four_different_digits (n : ℕ) : Prop :=
  (Finset.card (Finset.image (λ d => d % 10) (Finset.range 4))) = 4

/-- The smallest natural number greater than 3429 that uses exactly four different digits -/
def smallest_special_number : ℕ := 3450

theorem smallest_special_number_proof :
  smallest_special_number > 3429 ∧
  uses_four_different_digits smallest_special_number ∧
  ∀ n : ℕ, n > 3429 ∧ n < smallest_special_number → ¬(uses_four_different_digits n) :=
sorry

end smallest_special_number_proof_l1393_139325


namespace solve_system_l1393_139371

theorem solve_system (a b : ℝ) (h1 : 3 * a + 4 * b = 2) (h2 : a = 2 * b - 3) : 5 * b = 5.5 := by
  sorry

end solve_system_l1393_139371


namespace f_not_in_quadrant_II_l1393_139329

-- Define the linear function
def f (x : ℝ) : ℝ := 3 * x - 4

-- Define Quadrant II
def in_quadrant_II (x y : ℝ) : Prop := x < 0 ∧ y > 0

-- Theorem: The function f does not pass through Quadrant II
theorem f_not_in_quadrant_II :
  ∀ x : ℝ, ¬(in_quadrant_II x (f x)) :=
by
  sorry

end f_not_in_quadrant_II_l1393_139329


namespace population_in_scientific_notation_l1393_139385

/-- Represents a number in scientific notation -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ

/-- Converts a real number to scientific notation -/
def toScientificNotation (x : ℝ) : ScientificNotation :=
  sorry

theorem population_in_scientific_notation :
  let population : ℝ := 4.6e9
  toScientificNotation population = ScientificNotation.mk 4.6 9 := by
  sorry

end population_in_scientific_notation_l1393_139385


namespace parabola_vertex_l1393_139376

/-- A quadratic function f(x) = 2x^2 + px + q with roots at -6 and 4 -/
def f (p q : ℝ) (x : ℝ) : ℝ := 2 * x^2 + p * x + q

theorem parabola_vertex (p q : ℝ) :
  (∀ x ∈ Set.Icc (-6 : ℝ) 4, f p q x ≥ 0) ∧
  (∀ x ∉ Set.Icc (-6 : ℝ) 4, f p q x < 0) →
  ∃ vertex : ℝ × ℝ, vertex = (-1, -50) ∧
    ∀ x : ℝ, f p q x ≥ f p q (-1) :=
by sorry

end parabola_vertex_l1393_139376


namespace hyperbola_properties_l1393_139344

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

end hyperbola_properties_l1393_139344


namespace fourth_term_value_l1393_139394

/-- A geometric sequence with the given properties -/
structure GeometricSequence where
  a : ℕ → ℝ
  is_geometric : ∀ n : ℕ, a (n + 1) = a n * a 2
  first_term : a 1 = 9
  fifth_term : a 5 = a 3 * (a 4)^2

/-- The fourth term of the geometric sequence is ± 1/3 -/
theorem fourth_term_value (seq : GeometricSequence) : 
  seq.a 4 = 1/3 ∨ seq.a 4 = -1/3 := by
  sorry

end fourth_term_value_l1393_139394


namespace sqrt_abs_sum_zero_implies_sum_power_l1393_139333

theorem sqrt_abs_sum_zero_implies_sum_power (a b : ℝ) : 
  (Real.sqrt (a + 2) + |b - 1| = 0) → ((a + b)^2023 = -1) := by
  sorry

end sqrt_abs_sum_zero_implies_sum_power_l1393_139333


namespace quadratic_inequality_solution_l1393_139320

theorem quadratic_inequality_solution (a b : ℝ) :
  (∀ x, ax^2 + bx + 1 > 0 ↔ -1 < x ∧ x < 1/3) →
  a * b = 6 := by
  sorry

end quadratic_inequality_solution_l1393_139320


namespace no_five_coprime_two_digit_composites_l1393_139345

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


end no_five_coprime_two_digit_composites_l1393_139345


namespace profit_maximizing_price_l1393_139302

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

end profit_maximizing_price_l1393_139302


namespace expand_product_l1393_139327

theorem expand_product (x : ℝ) (h : x ≠ 0) :
  2/5 * (5/x + 10*x^2) = 2/x + 4*x^2 := by
  sorry

end expand_product_l1393_139327


namespace sqrt_ax_cube_l1393_139319

theorem sqrt_ax_cube (a x : ℝ) (ha : a < 0) : 
  Real.sqrt (a * x^3) = -x * Real.sqrt (a * x) :=
sorry

end sqrt_ax_cube_l1393_139319


namespace jimmy_lodging_expenses_l1393_139374

/-- Jimmy's lodging expenses during vacation -/
theorem jimmy_lodging_expenses :
  let hostel_nights : ℕ := 3
  let hostel_rate : ℕ := 15
  let cabin_nights : ℕ := 2
  let cabin_total_rate : ℕ := 45
  let cabin_friends : ℕ := 2
  
  let hostel_cost := hostel_nights * hostel_rate
  let cabin_cost := cabin_nights * (cabin_total_rate / (cabin_friends + 1))
  
  hostel_cost + cabin_cost = 75 := by sorry

end jimmy_lodging_expenses_l1393_139374


namespace sequence_properties_l1393_139318

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

end sequence_properties_l1393_139318


namespace sqrt_five_identity_l1393_139313

theorem sqrt_five_identity (m n a b c d : ℝ) :
  m + n * Real.sqrt 5 = (a + b * Real.sqrt 5) * (c + d * Real.sqrt 5) →
  m - n * Real.sqrt 5 = (a - b * Real.sqrt 5) * (c - d * Real.sqrt 5) :=
by sorry

end sqrt_five_identity_l1393_139313


namespace milkman_profit_is_51_l1393_139306

/-- Represents the milkman's problem --/
structure MilkmanProblem where
  total_milk : ℝ
  total_water : ℝ
  first_mixture_milk : ℝ
  first_mixture_water : ℝ
  second_mixture_water : ℝ
  pure_milk_cost : ℝ
  first_mixture_price : ℝ
  second_mixture_price : ℝ

/-- Calculate the total profit for the milkman --/
def calculate_profit (p : MilkmanProblem) : ℝ :=
  let second_mixture_milk := p.total_milk - p.first_mixture_milk
  let first_mixture_volume := p.first_mixture_milk + p.first_mixture_water
  let second_mixture_volume := second_mixture_milk + p.second_mixture_water
  let total_cost := p.pure_milk_cost * p.total_milk
  let total_revenue := p.first_mixture_price * first_mixture_volume + 
                       p.second_mixture_price * second_mixture_volume
  total_revenue - total_cost

/-- Theorem stating that the milkman's profit is 51 --/
theorem milkman_profit_is_51 : 
  let p : MilkmanProblem := {
    total_milk := 50,
    total_water := 15,
    first_mixture_milk := 30,
    first_mixture_water := 8,
    second_mixture_water := 7,
    pure_milk_cost := 20,
    first_mixture_price := 17,
    second_mixture_price := 15
  }
  calculate_profit p = 51 := by sorry


end milkman_profit_is_51_l1393_139306


namespace original_denominator_proof_l1393_139384

theorem original_denominator_proof : 
  ∀ d : ℚ, (5 : ℚ) / (d + 4) = (1 : ℚ) / 3 → d = 11 := by
  sorry

end original_denominator_proof_l1393_139384


namespace first_angle_measure_l1393_139331

theorem first_angle_measure (a b c : ℝ) : 
  a + b + c = 180 →  -- sum of angles in a triangle is 180 degrees
  b = 3 * a →        -- second angle is three times the first
  c = 2 * a - 12 →   -- third angle is 12 degrees less than twice the first
  a = 32 :=          -- prove that the first angle is 32 degrees
by sorry

end first_angle_measure_l1393_139331


namespace tinas_hourly_wage_l1393_139353

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

end tinas_hourly_wage_l1393_139353


namespace triangle_max_perimeter_l1393_139390

theorem triangle_max_perimeter :
  ∀ x : ℕ,
    x > 0 →
    x < 18 →
    x + 2*x > 18 →
    x + 2*x + 18 ≤ 69 :=
by sorry

end triangle_max_perimeter_l1393_139390


namespace count_integer_values_is_ten_l1393_139348

/-- The number of integer values of n for which 8000 * (2/5)^n is an integer --/
def count_integer_values : ℕ := 10

/-- Predicate to check if a given number is an integer --/
def is_integer (x : ℚ) : Prop := ∃ (k : ℤ), x = k

/-- The main theorem stating that there are exactly 10 integer values of n
    for which 8000 * (2/5)^n is an integer --/
theorem count_integer_values_is_ten :
  (∃! (s : Finset ℤ), s.card = count_integer_values ∧
    ∀ n : ℤ, n ∈ s ↔ is_integer (8000 * (2/5)^n)) :=
sorry

end count_integer_values_is_ten_l1393_139348


namespace points_in_circle_l1393_139393

theorem points_in_circle (points : Finset (ℝ × ℝ)) : 
  (points.card = 51) →
  (∀ p ∈ points, p.1 ∈ Set.Icc (0 : ℝ) 1 ∧ p.2 ∈ Set.Icc (0 : ℝ) 1) →
  ∃ c : ℝ × ℝ, ∃ s : Finset (ℝ × ℝ), 
    s ⊆ points ∧ 
    s.card = 3 ∧ 
    (∀ p ∈ s, (p.1 - c.1)^2 + (p.2 - c.2)^2 ≤ (1/7)^2) :=
by sorry

end points_in_circle_l1393_139393


namespace triangle_angle_sum_l1393_139341

theorem triangle_angle_sum (A B C : ℝ) (h1 : A = 75) (h2 : B = 40) : C = 65 := by
  sorry

end triangle_angle_sum_l1393_139341


namespace election_votes_theorem_l1393_139383

theorem election_votes_theorem :
  ∀ (total_votes : ℕ) (valid_votes : ℕ) (candidate1_votes : ℕ) (candidate2_votes : ℕ),
    valid_votes = (80 * total_votes) / 100 →
    candidate1_votes = (55 * valid_votes) / 100 →
    candidate2_votes = 2700 →
    candidate1_votes + candidate2_votes = valid_votes →
    total_votes = 7500 := by
  sorry

end election_votes_theorem_l1393_139383


namespace triangular_number_difference_l1393_139316

def triangular_number (n : ℕ) : ℕ := n * (n + 1) / 2

theorem triangular_number_difference : 
  triangular_number 30 - triangular_number 28 = 59 := by
sorry

end triangular_number_difference_l1393_139316


namespace fraction_spent_l1393_139343

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

end fraction_spent_l1393_139343


namespace cara_seating_arrangements_l1393_139332

/-- The number of people in the circular arrangement -/
def total_people : ℕ := 8

/-- The number of friends excluding Cara and Mark -/
def remaining_friends : ℕ := total_people - 2

/-- The number of different pairs Cara could be sitting between -/
def possible_pairs : ℕ := remaining_friends

theorem cara_seating_arrangements :
  possible_pairs = 6 :=
sorry

end cara_seating_arrangements_l1393_139332


namespace max_value_3m_4n_l1393_139358

theorem max_value_3m_4n (m n : ℕ+) : 
  (m.val * (m.val + 1) + n.val^2 = 1987) → 
  (∀ k l : ℕ+, k.val * (k.val + 1) + l.val^2 = 1987 → 3 * k.val + 4 * l.val ≤ 3 * m.val + 4 * n.val) →
  3 * m.val + 4 * n.val = 221 :=
by sorry

end max_value_3m_4n_l1393_139358


namespace complex_power_approximation_l1393_139307

/-- Prove that (3 * cos(30°) + 3i * sin(30°))^8 is approximately equal to -3281 - 3281i * √3 -/
theorem complex_power_approximation :
  ∃ (ε : ℝ), ε > 0 ∧ ε < 1 ∧ 
  Complex.abs ((3 * Complex.cos (30 * π / 180) + 3 * Complex.I * Complex.sin (30 * π / 180))^8 - 
               (-3281 - 3281 * Complex.I * Real.sqrt 3)) < ε :=
by sorry

end complex_power_approximation_l1393_139307


namespace all_drawings_fit_three_notebooks_l1393_139398

/-- Proves that all drawings fit in three notebooks after reorganization --/
theorem all_drawings_fit_three_notebooks 
  (initial_notebooks : Nat) 
  (pages_per_notebook : Nat) 
  (initial_drawings_per_page : Nat) 
  (new_drawings_per_page : Nat) 
  (h1 : initial_notebooks = 5)
  (h2 : pages_per_notebook = 60)
  (h3 : initial_drawings_per_page = 8)
  (h4 : new_drawings_per_page = 15) :
  (initial_notebooks * pages_per_notebook * initial_drawings_per_page) ≤ 
  (3 * pages_per_notebook * new_drawings_per_page) := by
  sorry

#check all_drawings_fit_three_notebooks

end all_drawings_fit_three_notebooks_l1393_139398


namespace f_not_increasing_l1393_139388

-- Define the function
def f (x : ℝ) : ℝ := |3 - x|

-- State the theorem
theorem f_not_increasing :
  ¬(∀ x y : ℝ, 0 ≤ x ∧ x < y → f x ≤ f y) :=
sorry

end f_not_increasing_l1393_139388


namespace expression_simplification_l1393_139340

theorem expression_simplification (a x y : ℝ) : 
  ((-2*a)^6*(-3*a^3) + (2*a^2)^3 / (1 / ((-2)^2 * 3^2 * (x*y)^3))) = 192*a^9 + 288*a^6*(x*y)^3 ∧
  |-(1/8)| + π^3 + (-(1/2)^3 - (1/3)^2) = π^3 - 1/72 :=
by sorry

end expression_simplification_l1393_139340


namespace min_translation_for_even_sine_l1393_139300

theorem min_translation_for_even_sine (f : ℝ → ℝ) (m : ℝ) :
  (∀ x, f x = Real.sin (3 * x + π / 4)) →
  m > 0 →
  (∀ x, f (x + m) = f (-x - m)) →
  m ≥ π / 12 ∧ ∃ m₀ > 0, m₀ < m → ¬(∀ x, f (x + m₀) = f (-x - m₀)) :=
by sorry

end min_translation_for_even_sine_l1393_139300


namespace borrowing_period_is_one_year_l1393_139362

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

end borrowing_period_is_one_year_l1393_139362


namespace train_average_speed_l1393_139386

/-- 
Given a train that travels two distances in two time periods, 
this theorem proves that its average speed is the total distance divided by the total time.
-/
theorem train_average_speed 
  (distance1 : ℝ) (time1 : ℝ) (distance2 : ℝ) (time2 : ℝ) 
  (h1 : distance1 = 325) 
  (h2 : time1 = 3.5)
  (h3 : distance2 = 470)
  (h4 : time2 = 4) :
  (distance1 + distance2) / (time1 + time2) = 106 := by
sorry

end train_average_speed_l1393_139386


namespace pythagorean_side_divisible_by_five_l1393_139309

theorem pythagorean_side_divisible_by_five (a b c : ℕ+) (h : a^2 + b^2 = c^2) :
  5 ∣ a ∨ 5 ∣ b ∨ 5 ∣ c := by
sorry

end pythagorean_side_divisible_by_five_l1393_139309


namespace relationship_xyz_l1393_139308

theorem relationship_xyz (a : ℝ) (x y z : ℝ) 
  (h1 : 0 < a) (h2 : a < 1) 
  (hx : x = a^a) (hy : y = a) (hz : z = Real.log a / Real.log a) : 
  z > x ∧ x > y := by sorry

end relationship_xyz_l1393_139308


namespace correct_operation_l1393_139335

theorem correct_operation (a b : ℝ) : 3*a + (a - 3*b) = 4*a - 3*b := by sorry

end correct_operation_l1393_139335


namespace pencils_in_drawer_l1393_139368

/-- The number of pencils initially in the drawer -/
def initial_pencils : ℕ := 72 - 45

/-- The number of pencils Nancy added to the drawer -/
def added_pencils : ℕ := 45

/-- The total number of pencils in the drawer after Nancy added more -/
def total_pencils : ℕ := 72

theorem pencils_in_drawer :
  initial_pencils + added_pencils = total_pencils :=
by sorry

end pencils_in_drawer_l1393_139368


namespace log_sum_greater_than_two_l1393_139375

theorem log_sum_greater_than_two (x y a : ℝ) (m : ℝ) 
  (h1 : 0 < x) (h2 : x < y) (h3 : y < a) (h4 : a < 1)
  (h5 : m = Real.log x / Real.log a + Real.log y / Real.log a) : 
  m > 2 := by
sorry

end log_sum_greater_than_two_l1393_139375


namespace yellow_raisins_cups_l1393_139334

theorem yellow_raisins_cups (total_raisins : Real) (black_raisins : Real) (yellow_raisins : Real) :
  total_raisins = 0.7 →
  black_raisins = 0.4 →
  total_raisins = yellow_raisins + black_raisins →
  yellow_raisins = 0.3 := by
  sorry

end yellow_raisins_cups_l1393_139334


namespace complex_magnitude_equals_sqrt15_l1393_139389

theorem complex_magnitude_equals_sqrt15 (s : ℝ) :
  Complex.abs (-3 + s * Complex.I) = 3 * Real.sqrt 5 → s = 6 := by
  sorry

end complex_magnitude_equals_sqrt15_l1393_139389


namespace negative_abs_negative_three_l1393_139397

theorem negative_abs_negative_three : -|-3| = -3 := by
  sorry

end negative_abs_negative_three_l1393_139397


namespace sin_690_degrees_l1393_139310

theorem sin_690_degrees : Real.sin (690 * π / 180) = -1/2 := by
  sorry

end sin_690_degrees_l1393_139310


namespace waiter_section_proof_l1393_139330

/-- Calculates the number of customers who left a waiter's section. -/
def customers_who_left (initial_customers : ℕ) (remaining_tables : ℕ) (people_per_table : ℕ) : ℕ :=
  initial_customers - (remaining_tables * people_per_table)

/-- Proves that 17 customers left the waiter's section given the initial conditions. -/
theorem waiter_section_proof :
  customers_who_left 62 5 9 = 17 := by
  sorry

end waiter_section_proof_l1393_139330


namespace no_consistent_values_l1393_139303

theorem no_consistent_values : ¬∃ (A B C D : ℤ), 
  B = 59 ∧ 
  C = 27 ∧ 
  D = 31 ∧ 
  (4701 % A = 0) ∧ 
  A = B * C + D :=
sorry

end no_consistent_values_l1393_139303


namespace bowl_glass_pairings_l1393_139351

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

end bowl_glass_pairings_l1393_139351


namespace license_plate_count_l1393_139346

def digit_choices : ℕ := 10
def letter_choices : ℕ := 26
def num_digits : ℕ := 5
def num_letters : ℕ := 3
def num_slots : ℕ := num_digits + 1

def choose (n k : ℕ) : ℕ := Nat.choose n k

theorem license_plate_count : 
  (digit_choices ^ num_digits) * (letter_choices ^ num_letters) * (choose num_slots num_letters) = 35152000000 := by
  sorry

end license_plate_count_l1393_139346


namespace perpendicular_vectors_m_value_l1393_139315

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

end perpendicular_vectors_m_value_l1393_139315


namespace xy_sum_over_three_l1393_139352

theorem xy_sum_over_three (x y : ℚ) 
  (eq1 : 2 * x + y = 6) 
  (eq2 : x + 2 * y = 5) : 
  (x + y) / 3 = 11 / 9 := by
  sorry

end xy_sum_over_three_l1393_139352


namespace concert_tickets_sold_l1393_139399

theorem concert_tickets_sold (cost_A cost_B total_tickets total_revenue : ℚ)
  (h1 : cost_A = 8)
  (h2 : cost_B = 4.25)
  (h3 : total_tickets = 4500)
  (h4 : total_revenue = 30000)
  : ∃ (tickets_A tickets_B : ℚ),
    tickets_A + tickets_B = total_tickets ∧
    cost_A * tickets_A + cost_B * tickets_B = total_revenue ∧
    tickets_A = 2900 := by
  sorry

end concert_tickets_sold_l1393_139399


namespace perpendicular_bisector_of_line_segment_l1393_139382

-- Define the line segment
def line_segment (x y : ℝ) : Prop := x - 2*y + 1 = 0 ∧ -1 ≤ x ∧ x ≤ 3

-- Define the perpendicular bisector
def perpendicular_bisector (x y : ℝ) : Prop := 2*x - y - 1 = 0

-- Theorem statement
theorem perpendicular_bisector_of_line_segment :
  ∀ x y : ℝ, line_segment x y →
  ∃ x' y' : ℝ, perpendicular_bisector x' y' ∧
  (x' = (x + (-1))/2 ∧ y' = (y + 0)/2) ∧
  (2*x' - y' - 1 = 0) :=
sorry

end perpendicular_bisector_of_line_segment_l1393_139382


namespace jack_and_jill_games_l1393_139366

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


end jack_and_jill_games_l1393_139366


namespace bicycle_installation_problem_l1393_139323

/-- The number of bicycles a skilled worker can install per day -/
def x : ℕ := sorry

/-- The number of bicycles a new worker can install per day -/
def y : ℕ := sorry

/-- The number of skilled workers -/
def a : ℕ := sorry

/-- The number of new workers -/
def b : ℕ := sorry

/-- Theorem stating the conditions and expected results -/
theorem bicycle_installation_problem :
  (2 * x + 3 * y = 44) ∧
  (4 * x = 5 * y) ∧
  (25 * (a * x + b * y) = 3500) →
  (x = 10 ∧ y = 8) ∧
  ((a = 2 ∧ b = 15) ∨ (a = 6 ∧ b = 10) ∨ (a = 10 ∧ b = 5)) :=
by sorry

end bicycle_installation_problem_l1393_139323


namespace catfish_dinner_price_l1393_139305

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

end catfish_dinner_price_l1393_139305


namespace square_19_on_top_l1393_139359

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

end square_19_on_top_l1393_139359


namespace rabbit_carrot_problem_l1393_139373

theorem rabbit_carrot_problem :
  ∀ (rabbit_holes hamster_holes : ℕ),
    rabbit_holes = hamster_holes - 3 →
    4 * rabbit_holes = 5 * hamster_holes →
    4 * rabbit_holes = 36 :=
by
  sorry

end rabbit_carrot_problem_l1393_139373


namespace probability_A_rolls_correct_l1393_139392

/-- The probability that player A rolls on the n-th turn in a dice game with the following rules:
  - A and B take turns rolling a die, with A going first.
  - If A rolls a 1, A continues to roll; otherwise, it's B's turn.
  - If B rolls a 3, B continues to roll; otherwise, it's A's turn. -/
def probability_A_rolls (n : ℕ) : ℚ :=
  1/2 - 1/3 * (-2/3)^(n-2)

/-- Theorem stating that the probability A rolls on the n-th turn is given by the formula -/
theorem probability_A_rolls_correct (n : ℕ) :
  probability_A_rolls n = 1/2 - 1/3 * (-2/3)^(n-2) := by
  sorry

end probability_A_rolls_correct_l1393_139392


namespace cubic_polynomial_property_l1393_139381

/-- The cubic polynomial whose roots we're interested in -/
def f (x : ℝ) : ℝ := x^3 + 3*x^2 + 5*x + 7

/-- Theorem stating the properties of the cubic polynomial P and its value at 0 -/
theorem cubic_polynomial_property (P : ℝ → ℝ) (a b c : ℝ) 
  (hf : f a = 0 ∧ f b = 0 ∧ f c = 0)
  (hPa : P a = b + c)
  (hPb : P b = c + a)
  (hPc : P c = a + b)
  (hPsum : P (a + b + c) = -16) :
  P 0 = 25 := by
  sorry

end cubic_polynomial_property_l1393_139381


namespace sum_of_abc_l1393_139337

theorem sum_of_abc (a b c : ℝ) 
  (eq1 : a^2 + 6*b = -17)
  (eq2 : b^2 + 8*c = -23)
  (eq3 : c^2 + 2*a = 14) :
  a + b + c = -8 := by
  sorry

end sum_of_abc_l1393_139337


namespace bridge_length_calculation_bridge_length_approx_248_30_l1393_139391

/-- Calculates the length of a bridge given train and wind conditions --/
theorem bridge_length_calculation (train_length : ℝ) (train_speed_kmh : ℝ) 
  (deceleration : ℝ) (headwind_speed_kmh : ℝ) : ℝ :=
  let train_speed_ms := train_speed_kmh * 1000 / 3600
  let headwind_speed_ms := headwind_speed_kmh * 1000 / 3600
  let effective_speed := train_speed_ms - headwind_speed_ms
  let deceleration_distance := effective_speed^2 / (2 * deceleration)
  deceleration_distance + train_length

/-- The bridge length is approximately 248.30 meters --/
theorem bridge_length_approx_248_30 :
  ∃ ε > 0, |bridge_length_calculation 200 60 2 10 - 248.30| < ε :=
by
  sorry

end bridge_length_calculation_bridge_length_approx_248_30_l1393_139391


namespace problem_solution_l1393_139336

noncomputable def f (a : ℝ) : ℝ → ℝ := fun x ↦ -1/2 + a / (3^x + 1)

theorem problem_solution (a : ℝ) (h_odd : ∀ x, f a x = -(f a (-x))) :
  (a = 1) ∧
  (∀ x y, x < y → f a x > f a y) ∧
  (∀ m, (∃ t ∈ Set.Ioo 1 2, f a (-2*t^2 + t + 1) + f a (t^2 - 2*m*t) ≤ 0) → m < 1/2) :=
by sorry

end problem_solution_l1393_139336


namespace tissue_count_after_use_l1393_139338

def initial_tissue_count : ℕ := 97
def used_tissue_count : ℕ := 4

theorem tissue_count_after_use :
  initial_tissue_count - used_tissue_count = 93 := by
  sorry

end tissue_count_after_use_l1393_139338


namespace binary_101101_to_octal_l1393_139347

def binary_to_octal (b : ℕ) : ℕ := sorry

theorem binary_101101_to_octal :
  binary_to_octal 0b101101 = 0o55 := by sorry

end binary_101101_to_octal_l1393_139347


namespace mans_age_twice_sons_l1393_139354

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

end mans_age_twice_sons_l1393_139354


namespace range_of_a_l1393_139314

noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then (1 - a) * x + 3 else Real.log x - 2 * a

theorem range_of_a (a : ℝ) :
  (∀ y : ℝ, ∃ x : ℝ, f a x = y) → -4 ≤ a ∧ a < 1 :=
by sorry

end range_of_a_l1393_139314


namespace smallest_degree_for_horizontal_asymptote_l1393_139322

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

end smallest_degree_for_horizontal_asymptote_l1393_139322


namespace no_function_satisfies_condition_l1393_139350

-- Define the function g
def g : ℝ → ℝ := sorry

-- Properties of g
axiom g_integer : ∀ n : ℤ, g n = (-1) ^ n
axiom g_affine : ∀ n : ℤ, ∀ x : ℝ, n ≤ x → x ≤ n + 1 → 
  ∃ a b : ℝ, ∀ y : ℝ, n ≤ y → y ≤ n + 1 → g y = a * y + b

-- Theorem statement
theorem no_function_satisfies_condition : 
  ¬ ∃ f : ℝ → ℝ, ∀ x y : ℝ, f (x + f y) = f x + g y := by sorry

end no_function_satisfies_condition_l1393_139350


namespace ship_departure_theorem_l1393_139395

/-- Represents the travel times and expected delivery for a cargo shipment --/
structure CargoShipment where
  travelDays : ℕ        -- Days for ship travel
  customsDays : ℕ       -- Days for customs processing
  deliveryDays : ℕ      -- Days from port to warehouse
  expectedArrival : ℕ   -- Days until expected arrival at warehouse

/-- Calculates the number of days ago the ship should have departed --/
def departureDays (shipment : CargoShipment) : ℕ :=
  shipment.travelDays + shipment.customsDays + shipment.deliveryDays - shipment.expectedArrival

/-- Theorem stating that for the given conditions, the ship should have departed 30 days ago --/
theorem ship_departure_theorem (shipment : CargoShipment) 
  (h1 : shipment.travelDays = 21)
  (h2 : shipment.customsDays = 4)
  (h3 : shipment.deliveryDays = 7)
  (h4 : shipment.expectedArrival = 2) :
  departureDays shipment = 30 := by
  sorry

#eval departureDays { travelDays := 21, customsDays := 4, deliveryDays := 7, expectedArrival := 2 }

end ship_departure_theorem_l1393_139395


namespace cookie_sales_proof_l1393_139355

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

end cookie_sales_proof_l1393_139355


namespace playground_area_l1393_139372

theorem playground_area (w l : ℚ) (h1 : l = 2 * w + 30) (h2 : 2 * (l + w) = 700) : w * l = 233600 / 9 := by
  sorry

end playground_area_l1393_139372


namespace max_consecutive_odds_is_five_l1393_139326

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

end max_consecutive_odds_is_five_l1393_139326
