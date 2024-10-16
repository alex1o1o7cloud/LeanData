import Mathlib

namespace NUMINAMATH_CALUDE_subcommittees_count_l3631_363120

/-- The number of different three-person sub-committees from a seven-person committee -/
def subcommittees : ℕ := Nat.choose 7 3

/-- Theorem stating that the number of subcommittees is 35 -/
theorem subcommittees_count : subcommittees = 35 := by
  sorry

end NUMINAMATH_CALUDE_subcommittees_count_l3631_363120


namespace NUMINAMATH_CALUDE_milkman_A_grazing_period_l3631_363199

/-- Represents the rental arrangement for a pasture shared by four milkmen. -/
structure PastureRental where
  /-- Number of cows grazed by milkman A -/
  cows_A : ℕ
  /-- Number of months milkman A grazed his cows (to be determined) -/
  months_A : ℕ
  /-- Number of cows grazed by milkman B -/
  cows_B : ℕ
  /-- Number of months milkman B grazed his cows -/
  months_B : ℕ
  /-- Number of cows grazed by milkman C -/
  cows_C : ℕ
  /-- Number of months milkman C grazed his cows -/
  months_C : ℕ
  /-- Number of cows grazed by milkman D -/
  cows_D : ℕ
  /-- Number of months milkman D grazed his cows -/
  months_D : ℕ
  /-- A's share of the rent in Rupees -/
  share_A : ℕ
  /-- Total rent of the field in Rupees -/
  total_rent : ℕ

/-- Theorem stating that given the conditions of the pasture rental,
    milkman A grazed his cows for 3 months. -/
theorem milkman_A_grazing_period (r : PastureRental)
  (h1 : r.cows_A = 24)
  (h2 : r.cows_B = 10)
  (h3 : r.months_B = 5)
  (h4 : r.cows_C = 35)
  (h5 : r.months_C = 4)
  (h6 : r.cows_D = 21)
  (h7 : r.months_D = 3)
  (h8 : r.share_A = 1440)
  (h9 : r.total_rent = 6500) :
  r.months_A = 3 := by
  sorry

end NUMINAMATH_CALUDE_milkman_A_grazing_period_l3631_363199


namespace NUMINAMATH_CALUDE_problem_distribution_l3631_363109

def distribute_problems (n : ℕ) (k : ℕ) : ℕ :=
  Nat.choose n k * Nat.factorial k

theorem problem_distribution :
  distribute_problems 9 7 = 181440 := by
  sorry

end NUMINAMATH_CALUDE_problem_distribution_l3631_363109


namespace NUMINAMATH_CALUDE_factorize_quadratic_xy_value_l3631_363184

-- Problem 1
theorem factorize_quadratic (x : ℝ) : 
  x^2 - 120*x + 3456 = (x - 48) * (x - 72) := by sorry

-- Problem 2
theorem xy_value (x y : ℝ) : 
  x^2 + y^2 + 8*x - 12*y + 52 = 0 → x*y = -24 := by sorry

end NUMINAMATH_CALUDE_factorize_quadratic_xy_value_l3631_363184


namespace NUMINAMATH_CALUDE_small_circles_radius_l3631_363194

theorem small_circles_radius (R : ℝ) (r : ℝ) : 
  R = 10 → 3 * (2 * r) = 2 * R → r = 10 / 3 :=
by sorry

end NUMINAMATH_CALUDE_small_circles_radius_l3631_363194


namespace NUMINAMATH_CALUDE_grand_forest_trail_length_l3631_363139

/-- Represents the length of Jamie's hike on the Grand Forest Trail -/
def GrandForestTrail : Type :=
  { hike : Vector ℝ 5 // 
    hike.get 0 + hike.get 1 + hike.get 2 = 42 ∧
    (hike.get 1 + hike.get 2) / 2 = 15 ∧
    hike.get 3 + hike.get 4 = 40 ∧
    hike.get 0 + hike.get 3 = 36 }

/-- The total length of the Grand Forest Trail is 82 miles -/
theorem grand_forest_trail_length (hike : GrandForestTrail) :
  hike.val.get 0 + hike.val.get 1 + hike.val.get 2 + hike.val.get 3 + hike.val.get 4 = 82 :=
by sorry

end NUMINAMATH_CALUDE_grand_forest_trail_length_l3631_363139


namespace NUMINAMATH_CALUDE_stratified_sampling_sample_size_l3631_363136

theorem stratified_sampling_sample_size 
  (total_employees : ℕ) 
  (young_employees : ℕ) 
  (young_in_sample : ℕ) 
  (h1 : total_employees = 750) 
  (h2 : young_employees = 350) 
  (h3 : young_in_sample = 7) : 
  (young_in_sample * total_employees) / young_employees = 15 := by
  sorry

end NUMINAMATH_CALUDE_stratified_sampling_sample_size_l3631_363136


namespace NUMINAMATH_CALUDE_total_rain_time_l3631_363160

def rain_duration_day1 : ℕ := 10

def rain_duration_day2 (d1 : ℕ) : ℕ := d1 + 2

def rain_duration_day3 (d2 : ℕ) : ℕ := 2 * d2

def total_rain_duration (d1 d2 d3 : ℕ) : ℕ := d1 + d2 + d3

theorem total_rain_time :
  total_rain_duration rain_duration_day1 
    (rain_duration_day2 rain_duration_day1) 
    (rain_duration_day3 (rain_duration_day2 rain_duration_day1)) = 46 := by
  sorry

end NUMINAMATH_CALUDE_total_rain_time_l3631_363160


namespace NUMINAMATH_CALUDE_quadratic_roots_problem_l3631_363125

theorem quadratic_roots_problem (k : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ 
    (∀ x : ℝ, x^2 + 2*k*x + k^2 = x + 1 ↔ x = x₁ ∨ x = x₂) ∧
    (3*x₁ - x₂)*(x₁ - 3*x₂) = 19) →
  k = 0 ∨ k = -3 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_problem_l3631_363125


namespace NUMINAMATH_CALUDE_R_value_at_7_l3631_363149

/-- The function that defines R in terms of S and h -/
def R (S h : ℝ) : ℝ := h * S + 2 * S - 6

/-- The theorem stating that if R = 28 when S = 5, then R = 41 when S = 7 -/
theorem R_value_at_7 (h : ℝ) (h_condition : R 5 h = 28) : R 7 h = 41 := by
  sorry

end NUMINAMATH_CALUDE_R_value_at_7_l3631_363149


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l3631_363182

/-- A geometric sequence with its sum sequence -/
structure GeometricSequence where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- The sum sequence
  geom : ∀ n : ℕ, a (n + 1) = a n * (a 1 / a 0)  -- Geometric property
  sum : ∀ n : ℕ, S n = (a 0 * (1 - (a 1 / a 0)^n)) / (1 - (a 1 / a 0))  -- Sum formula

/-- Theorem: If S_4 / S_2 = 3 for a geometric sequence, then 2a_2 - a_4 = 0 -/
theorem geometric_sequence_property (seq : GeometricSequence) 
  (h : seq.S 4 / seq.S 2 = 3) : 2 * seq.a 2 - seq.a 4 = 0 := by
  sorry


end NUMINAMATH_CALUDE_geometric_sequence_property_l3631_363182


namespace NUMINAMATH_CALUDE_hyperbola_sum_l3631_363131

/-- Represents a hyperbola with center (h, k) and parameters a and b -/
structure Hyperbola where
  h : ℝ
  k : ℝ
  a : ℝ
  b : ℝ

/-- The equation of the hyperbola -/
def hyperbola_equation (hyp : Hyperbola) (x y : ℝ) : Prop :=
  (x - hyp.h)^2 / hyp.a^2 - (y - hyp.k)^2 / hyp.b^2 = 1

theorem hyperbola_sum (hyp : Hyperbola) 
  (center : hyp.h = -3 ∧ hyp.k = 1)
  (vertex_distance : 2 * hyp.a = 8)
  (foci_distance : Real.sqrt (hyp.a^2 + hyp.b^2) = 5) :
  hyp.h + hyp.k + hyp.a + hyp.b = 5 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_sum_l3631_363131


namespace NUMINAMATH_CALUDE_regular_triangular_pyramid_volume_l3631_363161

/-- The volume of a regular triangular pyramid -/
theorem regular_triangular_pyramid_volume 
  (a b γ : ℝ) 
  (h_a : a > 0) 
  (h_b : b > 0) 
  (h_γ : 0 < γ ∧ γ < π) : 
  ∃ V : ℝ, V = (1/3) * (a^2 * Real.sqrt 3 / 4) * 
    Real.sqrt (b^2 - (a * Real.sqrt 3 / (2 * Real.cos (γ/2)))^2) := by
  sorry

end NUMINAMATH_CALUDE_regular_triangular_pyramid_volume_l3631_363161


namespace NUMINAMATH_CALUDE_integer_pair_divisibility_l3631_363151

theorem integer_pair_divisibility (m n : ℤ) : 
  m > 1 ∧ n > 1 → (m * n - 1 ∣ n^3 - 1) ↔ (m = n^2 ∨ n = m^2) :=
by sorry

end NUMINAMATH_CALUDE_integer_pair_divisibility_l3631_363151


namespace NUMINAMATH_CALUDE_inequality_problem_l3631_363152

theorem inequality_problem (x y : ℝ) 
  (h1 : 2 * x - 3 * y > 2 * x) 
  (h2 : 2 * x + 3 * y < 3 * y) : 
  x < 0 ∧ y < 0 := by
sorry

end NUMINAMATH_CALUDE_inequality_problem_l3631_363152


namespace NUMINAMATH_CALUDE_parabola_equation_l3631_363186

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a parabola with equation y^2 = 2px -/
structure Parabola where
  p : ℝ
  h_p_pos : p > 0

/-- Represents a line passing through two points -/
structure Line where
  a : Point
  b : Point

theorem parabola_equation (c : Parabola) (l : Line) :
  let f := Point.mk (c.p / 2) 0  -- Focus of the parabola
  let m := Point.mk 3 2  -- Midpoint of AB
  (l.a.y ^ 2 = 2 * c.p * l.a.x) ∧  -- A is on the parabola
  (l.b.y ^ 2 = 2 * c.p * l.b.x) ∧  -- B is on the parabola
  ((l.a.x + l.b.x) / 2 = m.x) ∧  -- M is the midpoint of AB (x-coordinate)
  ((l.a.y + l.b.y) / 2 = m.y) ∧  -- M is the midpoint of AB (y-coordinate)
  (f.x - l.a.x) * (l.b.y - l.a.y) = (f.y - l.a.y) * (l.b.x - l.a.x)  -- L passes through F
  →
  c.p = 2 ∨ c.p = 4 :=
by sorry

end NUMINAMATH_CALUDE_parabola_equation_l3631_363186


namespace NUMINAMATH_CALUDE_quadratic_function_range_l3631_363144

/-- The function f(x) = x^2 + mx - 1 -/
def f (m : ℝ) (x : ℝ) : ℝ := x^2 + m*x - 1

theorem quadratic_function_range (m : ℝ) :
  (∀ x ∈ Set.Icc m (m + 1), f m x < 0) ↔ m ∈ Set.Ioo (- Real.sqrt 2 / 2) 0 :=
sorry

end NUMINAMATH_CALUDE_quadratic_function_range_l3631_363144


namespace NUMINAMATH_CALUDE_tennis_balls_first_set_l3631_363148

theorem tennis_balls_first_set :
  ∀ (total_balls first_set second_set : ℕ),
    total_balls = 175 →
    second_set = 75 →
    first_set + second_set = total_balls →
    (2 : ℚ) / 5 * first_set + (1 : ℚ) / 3 * second_set + 110 = total_balls →
    first_set = 100 := by
  sorry

end NUMINAMATH_CALUDE_tennis_balls_first_set_l3631_363148


namespace NUMINAMATH_CALUDE_kelly_snacks_weight_l3631_363128

theorem kelly_snacks_weight (peanuts raisins total : Real) : 
  peanuts = 0.1 → raisins = 0.4 → total = peanuts + raisins → total = 0.5 := by
  sorry

end NUMINAMATH_CALUDE_kelly_snacks_weight_l3631_363128


namespace NUMINAMATH_CALUDE_total_cost_15_pencils_9_notebooks_l3631_363124

/-- The cost of a pencil in dollars -/
def pencil_cost : ℝ := sorry

/-- The cost of a notebook in dollars -/
def notebook_cost : ℝ := sorry

/-- The first given condition: 8 pencils and 5 notebooks cost $3.90 -/
axiom first_condition : 8 * pencil_cost + 5 * notebook_cost = 3.90

/-- The second given condition: 6 pencils and 4 notebooks cost $2.96 -/
axiom second_condition : 6 * pencil_cost + 4 * notebook_cost = 2.96

/-- The theorem to be proved -/
theorem total_cost_15_pencils_9_notebooks : 
  15 * pencil_cost + 9 * notebook_cost = 7.26 := by sorry

end NUMINAMATH_CALUDE_total_cost_15_pencils_9_notebooks_l3631_363124


namespace NUMINAMATH_CALUDE_hyperbola_equation_l3631_363196

theorem hyperbola_equation (f1 f2 : ℝ × ℝ) (p : ℝ × ℝ) :
  f1 = (0, 5) →
  f2 = (0, -5) →
  p = (2, 3 * Real.sqrt 5 / 2) →
  ∃ (a b : ℝ),
    a^2 = 9 ∧
    b^2 = 16 ∧
    ∀ (x y : ℝ),
      (y^2 / a^2) - (x^2 / b^2) = 1 ↔
      (x - f1.1)^2 + (y - f1.2)^2 - ((x - f2.1)^2 + (y - f2.2)^2) = 4 * a^2 ∧
      (p.1 - f1.1)^2 + (p.2 - f1.2)^2 - ((p.1 - f2.1)^2 + (p.2 - f2.2)^2) = 4 * a^2 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_equation_l3631_363196


namespace NUMINAMATH_CALUDE_fixed_point_exponential_l3631_363118

/-- The function f(x) = a^(x-1) + 4 always passes through the point (1, 5) for any a > 0 and a ≠ 1 -/
theorem fixed_point_exponential (a : ℝ) (ha : a > 0) (hna : a ≠ 1) :
  let f : ℝ → ℝ := λ x ↦ a^(x-1) + 4
  f 1 = 5 := by sorry

end NUMINAMATH_CALUDE_fixed_point_exponential_l3631_363118


namespace NUMINAMATH_CALUDE_students_per_class_is_twenty_l3631_363112

/-- Represents a school with teachers, a principal, classes, and students. -/
structure School where
  teachers : ℕ
  principal : ℕ
  classes : ℕ
  total_people : ℕ
  students_per_class : ℕ

/-- Theorem stating that in a school with given parameters, there are 20 students in each class. -/
theorem students_per_class_is_twenty (school : School)
  (h1 : school.teachers = 48)
  (h2 : school.principal = 1)
  (h3 : school.classes = 15)
  (h4 : school.total_people = 349)
  (h5 : school.total_people = school.teachers + school.principal + school.classes * school.students_per_class) :
  school.students_per_class = 20 := by
  sorry

end NUMINAMATH_CALUDE_students_per_class_is_twenty_l3631_363112


namespace NUMINAMATH_CALUDE_factor_polynomial_l3631_363108

theorem factor_polynomial (x : ℝ) : 72 * x^5 - 90 * x^9 = -18 * x^5 * (5 * x^4 - 4) := by
  sorry

end NUMINAMATH_CALUDE_factor_polynomial_l3631_363108


namespace NUMINAMATH_CALUDE_three_digit_primes_with_digit_product_189_l3631_363106

def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

def digit_product (n : ℕ) : ℕ :=
  (n / 100) * ((n / 10) % 10) * (n % 10)

def target_set : Set ℕ := {379, 397, 739, 937}

theorem three_digit_primes_with_digit_product_189 :
  ∀ n : ℕ, is_three_digit n ∧ Nat.Prime n ∧ digit_product n = 189 ↔ n ∈ target_set :=
sorry

end NUMINAMATH_CALUDE_three_digit_primes_with_digit_product_189_l3631_363106


namespace NUMINAMATH_CALUDE_matrix_product_R_S_l3631_363133

def R : Matrix (Fin 3) (Fin 3) ℝ := !![0, -1, 0; 1, 0, 0; 0, 0, 1]

def S (b c d : ℝ) : Matrix (Fin 3) (Fin 3) ℝ := !![d^2, d*c, d*b; d*c, c^2, c*b; d*b, c*b, b^2]

theorem matrix_product_R_S (b c d : ℝ) :
  R * S b c d = !![-(d*c), -(c^2), -(c*b); d^2, d*c, d*b; d*b, c*b, b^2] := by
  sorry

end NUMINAMATH_CALUDE_matrix_product_R_S_l3631_363133


namespace NUMINAMATH_CALUDE_seventh_grade_percentage_l3631_363140

theorem seventh_grade_percentage 
  (seventh_graders : ℕ) 
  (sixth_graders : ℕ) 
  (sixth_grade_percentage : ℚ) :
  seventh_graders = 64 →
  sixth_graders = 76 →
  sixth_grade_percentage = 38/100 →
  (↑seventh_graders : ℚ) / (↑sixth_graders / sixth_grade_percentage) = 32/100 :=
by sorry

end NUMINAMATH_CALUDE_seventh_grade_percentage_l3631_363140


namespace NUMINAMATH_CALUDE_lowest_energy_point_min_energy_at_two_l3631_363100

/-- Represents the energy function for an athlete during a 4-hour training session. -/
noncomputable def Q (t : ℝ) : ℝ :=
  if 0 < t ∧ t ≤ 1 then
    10000 - 3600 * t
  else if 1 < t ∧ t ≤ 4 then
    400 + 1200 * t + 4800 / t
  else
    0

/-- Theorem stating that the athlete's energy reaches its lowest point at t = 2 hours with a value of 5200kJ. -/
theorem lowest_energy_point :
  ∀ t : ℝ, 0 < t ∧ t ≤ 4 → Q t ≥ 5200 ∧ Q 2 = 5200 := by sorry

/-- Corollary stating that the minimum energy occurs at t = 2. -/
theorem min_energy_at_two :
  ∀ t : ℝ, 0 < t ∧ t ≤ 4 → Q 2 ≤ Q t := by sorry

end NUMINAMATH_CALUDE_lowest_energy_point_min_energy_at_two_l3631_363100


namespace NUMINAMATH_CALUDE_anime_watching_problem_l3631_363177

/-- The number of days from today to April 1, 2023 (exclusive) -/
def days_to_april_1 : ℕ := sorry

/-- The total number of episodes in the anime series -/
def total_episodes : ℕ := sorry

/-- Theorem stating the solution to the anime watching problem -/
theorem anime_watching_problem :
  (total_episodes - 2 * days_to_april_1 = 215) ∧
  (total_episodes - 5 * days_to_april_1 = 50) →
  (days_to_april_1 = 55 ∧ total_episodes = 325) :=
by sorry

end NUMINAMATH_CALUDE_anime_watching_problem_l3631_363177


namespace NUMINAMATH_CALUDE_ten_player_tournament_rounds_l3631_363195

/-- The number of rounds needed for a round-robin tennis tournament -/
def roundsNeeded (players : ℕ) (courts : ℕ) : ℕ :=
  (players * (players - 1) / 2 + courts - 1) / courts

/-- Theorem: A 10-player round-robin tournament on 5 courts needs 9 rounds -/
theorem ten_player_tournament_rounds :
  roundsNeeded 10 5 = 9 := by
  sorry

end NUMINAMATH_CALUDE_ten_player_tournament_rounds_l3631_363195


namespace NUMINAMATH_CALUDE_total_money_l3631_363183

theorem total_money (john alice bob : ℚ) (h1 : john = 5/8) (h2 : alice = 7/20) (h3 : bob = 1/4) :
  john + alice + bob = 1.225 := by
  sorry

end NUMINAMATH_CALUDE_total_money_l3631_363183


namespace NUMINAMATH_CALUDE_photo_arrangement_probability_photo_arrangement_probability_is_one_twentieth_l3631_363178

/-- The probability that in a group of six students with distinct heights,
    arranged in two rows of three each, every student in the back row
    is taller than every student in the front row. -/
theorem photo_arrangement_probability : ℚ :=
  let n_students : ℕ := 6
  let n_per_row : ℕ := 3
  let total_arrangements : ℕ := n_students.factorial
  let favorable_arrangements : ℕ := (n_per_row.factorial) * (n_per_row.factorial)
  favorable_arrangements / total_arrangements

/-- Proof that the probability is 1/20 -/
theorem photo_arrangement_probability_is_one_twentieth :
  photo_arrangement_probability = 1 / 20 := by
  sorry

end NUMINAMATH_CALUDE_photo_arrangement_probability_photo_arrangement_probability_is_one_twentieth_l3631_363178


namespace NUMINAMATH_CALUDE_fraction_calculation_l3631_363135

theorem fraction_calculation : (1/2 - 1/3) / (3/4 + 1/8) = 4/21 := by
  sorry

end NUMINAMATH_CALUDE_fraction_calculation_l3631_363135


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l3631_363155

def f (k : ℝ) (x : ℝ) : ℝ := x^2 - k*x - 1

theorem quadratic_function_properties (k : ℝ) :
  (∀ x₁ x₂ : ℝ, x₁ > x₂ ∧ x₂ ≥ 1 → f 2 x₁ > f 2 x₂) ∧
  (∀ x : ℝ, x ∈ Set.Icc 1 4 →
    (k ≥ 8 → f k x ≥ 16 - 4*k) ∧
    (k ≤ 2 → f k x ≥ -k) ∧
    (2 ≤ k ∧ k ≤ 8 → f k x ≥ -1)) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_function_properties_l3631_363155


namespace NUMINAMATH_CALUDE_max_cars_in_parking_lot_l3631_363193

/-- Represents a parking lot configuration -/
structure ParkingLot :=
  (grid : Fin 7 → Fin 7 → Bool)
  (gate : Fin 7 × Fin 7)

/-- Checks if a car can exit from its position -/
def canExit (lot : ParkingLot) (pos : Fin 7 × Fin 7) : Prop :=
  sorry

/-- Counts the number of cars in the parking lot -/
def carCount (lot : ParkingLot) : Nat :=
  sorry

/-- Checks if the parking lot configuration is valid -/
def isValidConfig (lot : ParkingLot) : Prop :=
  ∀ pos, lot.grid pos.1 pos.2 → canExit lot pos

/-- The main theorem stating the maximum number of cars that can be parked -/
theorem max_cars_in_parking_lot :
  ∃ (lot : ParkingLot), isValidConfig lot ∧ carCount lot = 28 ∧
  ∀ (other : ParkingLot), isValidConfig other → carCount other ≤ 28 :=
sorry

end NUMINAMATH_CALUDE_max_cars_in_parking_lot_l3631_363193


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l3631_363172

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, 2 * x^4 - x^2 + 1 < 0) ↔ (∃ x : ℝ, 2 * x^4 - x^2 + 1 ≥ 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l3631_363172


namespace NUMINAMATH_CALUDE_perpendicular_lines_a_value_l3631_363147

/-- Given a line l passing through points (a-2, -1) and (-a-2, 1), 
    and perpendicular to the line 2x+3y+1=0, prove that a = -2/3 -/
theorem perpendicular_lines_a_value (a : ℝ) : 
  let l : Set (ℝ × ℝ) := {p | ∃ t : ℝ, p = (a - 2 + t*(-2*a), -1 + t*2)}
  let slope_l := (1 - (-1)) / ((-a - 2) - (a - 2))
  let slope_other := -2 / 3
  (∀ p ∈ l, 2 * p.1 + 3 * p.2 + 1 ≠ 0) → 
  (slope_l * slope_other = -1) →
  a = -2/3 := by
sorry

end NUMINAMATH_CALUDE_perpendicular_lines_a_value_l3631_363147


namespace NUMINAMATH_CALUDE_pauls_weekly_spending_l3631_363188

/-- Given Paul's earnings and the duration the money lasted, calculate his weekly spending. -/
theorem pauls_weekly_spending (lawn_mowing : ℕ) (weed_eating : ℕ) (weeks : ℕ) 
  (h1 : lawn_mowing = 44)
  (h2 : weed_eating = 28)
  (h3 : weeks = 8)
  (h4 : weeks > 0) :
  (lawn_mowing + weed_eating) / weeks = 9 := by
  sorry

#check pauls_weekly_spending

end NUMINAMATH_CALUDE_pauls_weekly_spending_l3631_363188


namespace NUMINAMATH_CALUDE_binomial_coefficient_problem_l3631_363119

theorem binomial_coefficient_problem (m : ℕ+) 
  (a b : ℕ) 
  (ha : a = Nat.choose (2 * m) m)
  (hb : b = Nat.choose (2 * m + 1) m)
  (h_eq : 13 * a = 7 * b) : 
  m = 6 := by
sorry

end NUMINAMATH_CALUDE_binomial_coefficient_problem_l3631_363119


namespace NUMINAMATH_CALUDE_negative_a_squared_times_a_fourth_l3631_363189

theorem negative_a_squared_times_a_fourth (a : ℝ) : (-a)^2 * a^4 = a^6 := by
  sorry

end NUMINAMATH_CALUDE_negative_a_squared_times_a_fourth_l3631_363189


namespace NUMINAMATH_CALUDE_m_value_l3631_363174

def f (m : ℝ) (x : ℝ) : ℝ := x^3 - 3*x + m

theorem m_value (m : ℝ) :
  (∃ (max min : ℝ), 
    (∀ x ∈ Set.Icc (-3) 0, f m x ≤ max) ∧
    (∃ x ∈ Set.Icc (-3) 0, f m x = max) ∧
    (∀ x ∈ Set.Icc (-3) 0, min ≤ f m x) ∧
    (∃ x ∈ Set.Icc (-3) 0, f m x = min) ∧
    max + min = -1) →
  m = 7.5 := by
sorry

end NUMINAMATH_CALUDE_m_value_l3631_363174


namespace NUMINAMATH_CALUDE_ratio_of_sums_l3631_363165

/-- An arithmetic sequence with common difference d, first term 8d, and sum of first n terms S_n -/
structure ArithmeticSequence (d : ℝ) where
  a : ℕ → ℝ
  S : ℕ → ℝ
  h1 : d ≠ 0
  h2 : ∀ n, a (n + 1) = a n + d
  h3 : a 1 = 8 * d
  h4 : ∀ n, S n = n * (2 * a 1 + (n - 1) * d) / 2

/-- The ratio of 7S_5 to 5S_7 is 10/11 for the given arithmetic sequence -/
theorem ratio_of_sums (d : ℝ) (seq : ArithmeticSequence d) :
  7 * seq.S 5 / (5 * seq.S 7) = 10 / 11 :=
sorry

end NUMINAMATH_CALUDE_ratio_of_sums_l3631_363165


namespace NUMINAMATH_CALUDE_extreme_value_negative_a_one_zero_positive_a_l3631_363168

noncomputable section

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := a * Real.log x + (1/2) * x^2 - (a + 1) * x

-- Theorem for the case when a < 0
theorem extreme_value_negative_a (a : ℝ) (ha : a < 0) :
  (∃ x : ℝ, ∀ y : ℝ, f a y ≥ f a x) ∧
  (∀ x : ℝ, f a x ≥ -a - 1/2) ∧
  (¬∃ x : ℝ, ∀ y : ℝ, f a y ≤ f a x) :=
sorry

-- Theorem for the case when a > 0
theorem one_zero_positive_a (a : ℝ) (ha : a > 0) :
  ∃! x : ℝ, f a x = 0 :=
sorry

end

end NUMINAMATH_CALUDE_extreme_value_negative_a_one_zero_positive_a_l3631_363168


namespace NUMINAMATH_CALUDE_students_in_both_competitions_l3631_363157

/-- The number of students who participated in both competitions A and B -/
def students_in_both (total students_A students_B : ℕ) : ℕ :=
  students_A + students_B - total

theorem students_in_both_competitions 
  (total : ℕ) (students_A : ℕ) (students_B : ℕ)
  (h_total : total = 55)
  (h_A : students_A = 38)
  (h_B : students_B = 42)
  (h_all_participated : total ≤ students_A + students_B) :
  students_in_both total students_A students_B = 25 := by
  sorry

#eval students_in_both 55 38 42  -- Should output 25

end NUMINAMATH_CALUDE_students_in_both_competitions_l3631_363157


namespace NUMINAMATH_CALUDE_dice_surface_area_l3631_363137

/-- The surface area of a cube with edge length 11 cm is 726 cm^2. -/
theorem dice_surface_area (edge_length : ℝ) (h : edge_length = 11) :
  6 * edge_length^2 = 726 := by
  sorry

end NUMINAMATH_CALUDE_dice_surface_area_l3631_363137


namespace NUMINAMATH_CALUDE_additional_cars_needed_l3631_363197

def current_cars : ℕ := 37
def cars_per_row : ℕ := 8

theorem additional_cars_needed : 
  ∃ (n : ℕ), 
    (n > 0) ∧ 
    (current_cars + n) % cars_per_row = 0 ∧
    ∀ (m : ℕ), m < n → (current_cars + m) % cars_per_row ≠ 0 :=
by
  sorry

end NUMINAMATH_CALUDE_additional_cars_needed_l3631_363197


namespace NUMINAMATH_CALUDE_quadratic_properties_l3631_363153

/-- A quadratic function passing through (-3, 0) with axis of symmetry x = -1 -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ
  a_nonzero : a ≠ 0
  passes_through_minus_three : a * (-3)^2 + b * (-3) + c = 0
  axis_of_symmetry : -b / (2 * a) = -1

/-- Properties of the quadratic function -/
theorem quadratic_properties (f : QuadraticFunction) :
  (f.a + f.b + f.c = 0) ∧
  (2 * f.c + 3 * f.b = 0) ∧
  (∀ k : ℝ, k > 0 → ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧
    f.a * x₁^2 + f.b * x₁ + f.c = k * (x₁ + 1) ∧
    f.a * x₂^2 + f.b * x₂ + f.c = k * (x₂ + 1)) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_properties_l3631_363153


namespace NUMINAMATH_CALUDE_prime_pairs_dividing_sum_of_powers_l3631_363110

theorem prime_pairs_dividing_sum_of_powers (p q : ℕ) : 
  Prime p → Prime q → (p * q ∣ 2^p + 2^q) → 
  ((p = 2 ∧ q = 2) ∨ (p = 2 ∧ q = 3) ∨ (p = 3 ∧ q = 2)) := by
  sorry

end NUMINAMATH_CALUDE_prime_pairs_dividing_sum_of_powers_l3631_363110


namespace NUMINAMATH_CALUDE_garden_area_l3631_363170

/-- A rectangular garden with perimeter 36 feet and one side 10 feet has an area of 80 square feet. -/
theorem garden_area (perimeter : ℝ) (side : ℝ) (h1 : perimeter = 36) (h2 : side = 10) :
  let other_side := (perimeter - 2 * side) / 2
  side * other_side = 80 :=
by sorry

end NUMINAMATH_CALUDE_garden_area_l3631_363170


namespace NUMINAMATH_CALUDE_carries_payment_is_94_l3631_363127

/-- The amount Carrie pays for clothes at the mall -/
def carries_payment (num_shirts num_pants num_jackets shirt_cost pant_cost jacket_cost : ℕ) : ℕ :=
  let total_cost := num_shirts * shirt_cost + num_pants * pant_cost + num_jackets * jacket_cost
  total_cost / 2

/-- Theorem: Carrie pays $94 for the clothes -/
theorem carries_payment_is_94 : carries_payment 4 2 2 8 18 60 = 94 := by
  sorry

#eval carries_payment 4 2 2 8 18 60

end NUMINAMATH_CALUDE_carries_payment_is_94_l3631_363127


namespace NUMINAMATH_CALUDE_remaining_steps_l3631_363175

theorem remaining_steps (total : ℕ) (climbed : ℕ) (h1 : total = 96) (h2 : climbed = 74) :
  total - climbed = 22 := by
  sorry

end NUMINAMATH_CALUDE_remaining_steps_l3631_363175


namespace NUMINAMATH_CALUDE_sum_equals_zero_l3631_363115

/-- The number of numbers satisfying: "there is no other number whose absolute value is equal to the absolute value of a" -/
def a : ℕ := sorry

/-- The number of numbers satisfying: "there is no other number whose square is equal to the square of b" -/
def b : ℕ := sorry

/-- The number of numbers satisfying: "there is no other number that, when multiplied by c, results in a product greater than 1" -/
def c : ℕ := sorry

theorem sum_equals_zero : a + b + c = 0 := by sorry

end NUMINAMATH_CALUDE_sum_equals_zero_l3631_363115


namespace NUMINAMATH_CALUDE_cloth_cost_calculation_l3631_363162

/-- The total cost of cloth given the length and price per metre -/
def total_cost (length : ℝ) (price_per_metre : ℝ) : ℝ :=
  length * price_per_metre

/-- Theorem: The total cost of 9.25 m of cloth at $45 per metre is $416.25 -/
theorem cloth_cost_calculation :
  total_cost 9.25 45 = 416.25 := by
  sorry

end NUMINAMATH_CALUDE_cloth_cost_calculation_l3631_363162


namespace NUMINAMATH_CALUDE_price_of_49_dozens_l3631_363101

/-- Calculates the price of a given number of dozens of apples at a new price -/
def price_of_apples (initial_price : ℝ) (new_price : ℝ) (dozens : ℕ) : ℝ :=
  dozens * new_price

/-- Theorem: The price of 49 dozens of apples at the new price is 49 times the new price -/
theorem price_of_49_dozens 
  (initial_price : ℝ) 
  (new_price : ℝ) 
  (h1 : initial_price = 1517.25)
  (h2 : new_price = 2499) :
  price_of_apples initial_price new_price 49 = 49 * new_price :=
by sorry

end NUMINAMATH_CALUDE_price_of_49_dozens_l3631_363101


namespace NUMINAMATH_CALUDE_average_after_removal_l3631_363141

theorem average_after_removal (numbers : Finset ℝ) (sum : ℝ) :
  Finset.card numbers = 10 →
  sum = Finset.sum numbers id →
  sum / 10 = 85 →
  72 ∈ numbers →
  78 ∈ numbers →
  ((sum - 72 - 78) / 8) = 87.5 :=
sorry

end NUMINAMATH_CALUDE_average_after_removal_l3631_363141


namespace NUMINAMATH_CALUDE_cow_value_increase_is_600_l3631_363166

/-- Calculates the increase in a cow's value after weight gain -/
def cow_value_increase (initial_weight : ℝ) (weight_increase_factor : ℝ) (price_per_pound : ℝ) : ℝ :=
  (initial_weight * weight_increase_factor * price_per_pound) - (initial_weight * price_per_pound)

/-- Theorem stating that the increase in the cow's value is $600 -/
theorem cow_value_increase_is_600 :
  cow_value_increase 400 1.5 3 = 600 := by
  sorry

end NUMINAMATH_CALUDE_cow_value_increase_is_600_l3631_363166


namespace NUMINAMATH_CALUDE_housewife_money_l3631_363117

theorem housewife_money (initial_money : ℚ) : 
  (1 - 2/3) * initial_money = 50 → initial_money = 150 := by
  sorry

end NUMINAMATH_CALUDE_housewife_money_l3631_363117


namespace NUMINAMATH_CALUDE_dwarf_attire_comparison_l3631_363143

/-- Represents the problem of comparing dwarf groups based on their attire. -/
theorem dwarf_attire_comparison :
  let total_dwarves : ℕ := 25
  let dwarves_without_hats : ℕ := 12
  let barefoot_dwarves : ℕ := 5
  let dwarves_with_hats := total_dwarves - dwarves_without_hats
  let dwarves_with_shoes := total_dwarves - barefoot_dwarves
  let dwarves_with_shoes_no_hat := dwarves_with_shoes - dwarves_with_hats
  dwarves_with_hats = dwarves_with_shoes_no_hat + 6 :=
by
  sorry


end NUMINAMATH_CALUDE_dwarf_attire_comparison_l3631_363143


namespace NUMINAMATH_CALUDE_function_properties_l3631_363138

-- Define the function f(x) with parameter a
def f (a : ℝ) (x : ℝ) : ℝ := x^3 + (4-a)*x^2 - 15*x + a

-- Define the derivative of f(x) with respect to x
def f' (a : ℝ) (x : ℝ) : ℝ := 3*x^2 + 2*(4-a)*x - 15

theorem function_properties :
  -- Part 1: When f(0) = -2, a = -2
  (∀ a : ℝ, f a 0 = -2 → a = -2) ∧
  
  -- Part 2: The minimum value of f(x) when a = -2 is -10
  (∃ x : ℝ, f (-2) x = -10 ∧ ∀ y : ℝ, f (-2) y ≥ -10) ∧
  
  -- Part 3: The maximum value of a for which f'(x) ≤ 0 on (-1, 1) is 10
  (∀ a : ℝ, (∀ x : ℝ, -1 < x ∧ x < 1 → f' a x ≤ 0) → a ≤ 10) ∧
  (∃ a : ℝ, a = 10 ∧ ∀ x : ℝ, -1 < x ∧ x < 1 → f' a x ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_function_properties_l3631_363138


namespace NUMINAMATH_CALUDE_max_stamps_purchasable_l3631_363181

/-- Given a stamp price of 28 cents and a budget of 3600 cents,
    the maximum number of stamps that can be purchased is 128. -/
theorem max_stamps_purchasable (stamp_price : ℕ) (budget : ℕ) :
  stamp_price = 28 → budget = 3600 → 
  (∃ (n : ℕ), n * stamp_price ≤ budget ∧ 
    ∀ (m : ℕ), m * stamp_price ≤ budget → m ≤ n) →
  (∃ (max_stamps : ℕ), max_stamps = 128) :=
by sorry

end NUMINAMATH_CALUDE_max_stamps_purchasable_l3631_363181


namespace NUMINAMATH_CALUDE_estate_distribution_theorem_l3631_363158

/-- Represents the estate distribution problem -/
structure EstateProblem where
  num_beneficiaries : Nat
  min_ratio : Real
  known_amount : Real

/-- Calculates the smallest possible range between the highest and lowest amounts -/
def smallest_range (problem : EstateProblem) : Real :=
  sorry

/-- The theorem stating the smallest possible range for the given problem -/
theorem estate_distribution_theorem (problem : EstateProblem) 
  (h1 : problem.num_beneficiaries = 8)
  (h2 : problem.min_ratio = 1.4)
  (h3 : problem.known_amount = 80000) :
  smallest_range problem = 72412 := by
  sorry

end NUMINAMATH_CALUDE_estate_distribution_theorem_l3631_363158


namespace NUMINAMATH_CALUDE_rakesh_cash_calculation_l3631_363173

/-- Calculates the cash in hand after fixed deposit and grocery expenses --/
def cash_in_hand (salary : ℚ) (fixed_deposit_rate : ℚ) (grocery_rate : ℚ) : ℚ :=
  let fixed_deposit := salary * fixed_deposit_rate
  let remaining := salary - fixed_deposit
  let groceries := remaining * grocery_rate
  remaining - groceries

/-- Proves that given the conditions, the cash in hand is 2380 --/
theorem rakesh_cash_calculation :
  cash_in_hand 4000 (15/100) (30/100) = 2380 := by
  sorry

end NUMINAMATH_CALUDE_rakesh_cash_calculation_l3631_363173


namespace NUMINAMATH_CALUDE_new_year_cards_cost_l3631_363114

def card_price_1 : ℚ := 10 / 100
def card_price_2 : ℚ := 15 / 100
def card_price_3 : ℚ := 25 / 100
def card_price_4 : ℚ := 40 / 100

def total_cards : ℕ := 30

theorem new_year_cards_cost (q1 q2 q3 q4 : ℕ) 
  (h1 : q1 + q2 + q3 + q4 = total_cards)
  (h2 : (q1 = 5 ∧ q2 = 5) ∨ (q1 = 5 ∧ q3 = 5) ∨ (q1 = 5 ∧ q4 = 5) ∨ 
        (q2 = 5 ∧ q3 = 5) ∨ (q2 = 5 ∧ q4 = 5) ∨ (q3 = 5 ∧ q4 = 5))
  (h3 : (q1 = 10 ∧ q2 = 10) ∨ (q1 = 10 ∧ q3 = 10) ∨ (q1 = 10 ∧ q4 = 10) ∨ 
        (q2 = 10 ∧ q3 = 10) ∨ (q2 = 10 ∧ q4 = 10) ∨ (q3 = 10 ∧ q4 = 10))
  (h4 : ∃ (n : ℕ), q1 * card_price_1 + q2 * card_price_2 + q3 * card_price_3 + q4 * card_price_4 = n) :
  q1 * card_price_1 + q2 * card_price_2 + q3 * card_price_3 + q4 * card_price_4 = 7 := by
sorry


end NUMINAMATH_CALUDE_new_year_cards_cost_l3631_363114


namespace NUMINAMATH_CALUDE_first_term_of_geometric_sequence_l3631_363146

/-- A geometric sequence is a sequence where each term after the first is found by multiplying the previous term by a fixed, non-zero number called the common ratio. -/
def IsGeometricSequence (s : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ ∀ n : ℕ, s (n + 1) = s n * r

/-- Given a geometric sequence where the 6th term is 32 and the 7th term is 64, the first term is 1. -/
theorem first_term_of_geometric_sequence
  (s : ℕ → ℝ)
  (h_geometric : IsGeometricSequence s)
  (h_6th : s 6 = 32)
  (h_7th : s 7 = 64) :
  s 1 = 1 := by
  sorry


end NUMINAMATH_CALUDE_first_term_of_geometric_sequence_l3631_363146


namespace NUMINAMATH_CALUDE_area_after_reflection_l3631_363179

/-- Right triangle ABC with given side lengths -/
structure RightTriangle where
  AB : ℝ
  BC : ℝ
  right_angle : AB > 0 ∧ BC > 0

/-- Points after reflection -/
structure ReflectedPoints where
  A' : ℝ × ℝ
  B' : ℝ × ℝ
  C' : ℝ × ℝ

/-- Function to perform reflections -/
def reflect (t : RightTriangle) : ReflectedPoints := sorry

/-- Calculate area of triangle A'B'C' -/
def area_A'B'C' (p : ReflectedPoints) : ℝ := sorry

/-- Main theorem -/
theorem area_after_reflection (t : RightTriangle) 
  (h1 : t.AB = 5)
  (h2 : t.BC = 12) : 
  area_A'B'C' (reflect t) = 17.5 := by sorry

end NUMINAMATH_CALUDE_area_after_reflection_l3631_363179


namespace NUMINAMATH_CALUDE_exists_integer_sqrt_20n_is_integer_l3631_363167

theorem exists_integer_sqrt_20n_is_integer : ∃ n : ℤ, ∃ m : ℤ, 20 * n = m^2 := by
  sorry

end NUMINAMATH_CALUDE_exists_integer_sqrt_20n_is_integer_l3631_363167


namespace NUMINAMATH_CALUDE_lunch_cost_theorem_l3631_363111

/-- Calculates the total cost of lunches for a field trip --/
def total_lunch_cost (total_people : ℕ) (extra_lunches : ℕ) (vegetarian : ℕ) (gluten_free : ℕ) 
  (nut_free : ℕ) (halal : ℕ) (veg_and_gf : ℕ) (regular_cost : ℕ) (special_cost : ℕ) 
  (veg_gf_cost : ℕ) : ℕ :=
  let total_lunches := total_people + extra_lunches
  let regular_lunches := total_lunches - (vegetarian + gluten_free + nut_free + halal - veg_and_gf)
  let regular_total := regular_lunches * regular_cost
  let vegetarian_total := (vegetarian - veg_and_gf) * special_cost
  let gluten_free_total := gluten_free * special_cost
  let nut_free_total := nut_free * special_cost
  let halal_total := halal * special_cost
  let veg_gf_total := veg_and_gf * veg_gf_cost
  regular_total + vegetarian_total + gluten_free_total + nut_free_total + halal_total + veg_gf_total

theorem lunch_cost_theorem :
  total_lunch_cost 41 3 10 5 3 4 2 7 8 9 = 346 := by
  sorry

#eval total_lunch_cost 41 3 10 5 3 4 2 7 8 9

end NUMINAMATH_CALUDE_lunch_cost_theorem_l3631_363111


namespace NUMINAMATH_CALUDE_price_per_shirt_is_35_l3631_363171

/-- Calculates the price per shirt given the following parameters:
    * num_employees: number of employees
    * shirts_per_employee: number of shirts made per employee per day
    * hours_per_shift: number of hours in a shift
    * hourly_wage: hourly wage per employee
    * per_shirt_wage: additional wage per shirt made
    * nonemployee_expenses: daily nonemployee expenses
    * daily_profit: target daily profit
-/
def price_per_shirt (
  num_employees : ℕ
) (shirts_per_employee : ℕ
) (hours_per_shift : ℕ
) (hourly_wage : ℚ
) (per_shirt_wage : ℚ
) (nonemployee_expenses : ℚ
) (daily_profit : ℚ
) : ℚ :=
  let total_shirts := num_employees * shirts_per_employee
  let total_wages := num_employees * hours_per_shift * hourly_wage + total_shirts * per_shirt_wage
  let total_expenses := total_wages + nonemployee_expenses
  let total_revenue := daily_profit + total_expenses
  total_revenue / total_shirts

theorem price_per_shirt_is_35 :
  price_per_shirt 20 20 8 12 5 1000 9080 = 35 := by
  sorry

#eval price_per_shirt 20 20 8 12 5 1000 9080

end NUMINAMATH_CALUDE_price_per_shirt_is_35_l3631_363171


namespace NUMINAMATH_CALUDE_product_of_fractions_and_powers_of_three_l3631_363132

theorem product_of_fractions_and_powers_of_three (x : ℚ) : 
  x = (1/3 : ℚ) * 9 * (1/27 : ℚ) * 81 * (1/243 : ℚ) * 729 * (1/2187 : ℚ) * 6561 * (1/19683 : ℚ) * 59049 → 
  x = 243 := by
sorry

end NUMINAMATH_CALUDE_product_of_fractions_and_powers_of_three_l3631_363132


namespace NUMINAMATH_CALUDE_parallel_planes_from_parallel_intersecting_lines_parallel_planes_transitive_l3631_363121

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the relationships between planes and lines
variable (parallel : Plane → Plane → Prop)
variable (intersect : Line → Line → Prop)
variable (in_plane : Line → Plane → Prop)
variable (parallel_lines : Line → Line → Prop)

-- Theorem 1
theorem parallel_planes_from_parallel_intersecting_lines 
  (α β : Plane) (l1 l2 m1 m2 : Line) :
  in_plane l1 α → in_plane l2 α → intersect l1 l2 →
  in_plane m1 β → in_plane m2 β → intersect m1 m2 →
  parallel_lines l1 m1 → parallel_lines l2 m2 →
  parallel α β :=
sorry

-- Theorem 2
theorem parallel_planes_transitive (α β γ : Plane) :
  parallel α β → parallel β γ → parallel α γ :=
sorry

end NUMINAMATH_CALUDE_parallel_planes_from_parallel_intersecting_lines_parallel_planes_transitive_l3631_363121


namespace NUMINAMATH_CALUDE_basketball_score_proof_l3631_363150

theorem basketball_score_proof (total : ℕ) 
  (hA : total / 4 = total / 4)  -- Player A scored 1/4 of total
  (hB : (total * 2) / 7 = (total * 2) / 7)  -- Player B scored 2/7 of total
  (hC : 15 ≤ total)  -- Player C scored 15 points
  (hRemaining : ∀ i : Fin 7, (total - (total / 4 + (total * 2) / 7 + 15)) / 7 ≤ 2)  -- Remaining players scored no more than 2 points each
  : total - (total / 4 + (total * 2) / 7 + 15) = 13 :=
by sorry

end NUMINAMATH_CALUDE_basketball_score_proof_l3631_363150


namespace NUMINAMATH_CALUDE_student_average_less_than_true_average_l3631_363176

theorem student_average_less_than_true_average 
  (x y w : ℝ) (h : x > y ∧ y > w) : 
  ((x + y) / 2 + w) / 2 < (x + y + w) / 3 := by
sorry

end NUMINAMATH_CALUDE_student_average_less_than_true_average_l3631_363176


namespace NUMINAMATH_CALUDE_geometric_sequence_solution_l3631_363123

noncomputable def f (x : ℝ) : ℝ :=
  if x ≥ 1 then x * Real.log x else Real.log x / x

def is_geometric_sequence (a : ℕ → ℝ) (r : ℝ) : Prop :=
  ∀ n, a (n + 1) = r * a n

theorem geometric_sequence_solution (a : ℕ → ℝ) (r : ℝ) :
  is_geometric_sequence a r →
  r > 0 →
  a 3 * a 4 * a 5 = 1 →
  f (a 1) + f (a 2) + f (a 3) + f (a 4) + f (a 5) + f (a 6) = 2 * a 1 →
  a 1 = Real.exp 2 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_solution_l3631_363123


namespace NUMINAMATH_CALUDE_tan_3x_increasing_interval_l3631_363190

theorem tan_3x_increasing_interval (m : ℝ) : 
  (∀ x₁ x₂, m < x₁ ∧ x₁ < x₂ ∧ x₂ < π/6 → Real.tan (3*x₁) < Real.tan (3*x₂)) → 
  m ∈ Set.Icc (-π/6) (π/6) := by
sorry

end NUMINAMATH_CALUDE_tan_3x_increasing_interval_l3631_363190


namespace NUMINAMATH_CALUDE_gcd_of_quadratic_and_linear_l3631_363191

theorem gcd_of_quadratic_and_linear (b : ℤ) (h : ∃ k : ℤ, b = 1620 * k) :
  Nat.gcd (Int.natAbs (b^2 + 11*b + 36)) (Int.natAbs (b + 6)) = 6 := by
  sorry

end NUMINAMATH_CALUDE_gcd_of_quadratic_and_linear_l3631_363191


namespace NUMINAMATH_CALUDE_fraction_equation_solution_l3631_363104

theorem fraction_equation_solution (n : ℚ) :
  (1 : ℚ) / (n + 1) + (2 : ℚ) / (n + 1) + n / (n + 1) = 4 → n = -1/3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equation_solution_l3631_363104


namespace NUMINAMATH_CALUDE_probability_outside_circle_l3631_363185

/-- A die roll outcome is a natural number between 1 and 6 -/
def DieRoll : Type := {n : ℕ // 1 ≤ n ∧ n ≤ 6}

/-- A point P is defined by two die roll outcomes -/
structure Point where
  m : DieRoll
  n : DieRoll

/-- A point P(m,n) is outside the circle if m^2 + n^2 > 25 -/
def isOutsideCircle (p : Point) : Prop :=
  (p.m.val ^ 2 + p.n.val ^ 2 : ℚ) > 25

/-- The total number of possible outcomes when rolling a die twice -/
def totalOutcomes : ℕ := 36

/-- The number of outcomes resulting in a point outside the circle -/
def favorableOutcomes : ℕ := 11

/-- The main theorem: probability of a point being outside the circle -/
theorem probability_outside_circle :
  (favorableOutcomes : ℚ) / totalOutcomes = 11 / 36 := by sorry

end NUMINAMATH_CALUDE_probability_outside_circle_l3631_363185


namespace NUMINAMATH_CALUDE_ratio_composition_l3631_363169

theorem ratio_composition (a b c : ℚ) 
  (h1 : a / b = 2 / 3) 
  (h2 : b / c = 1 / 5) : 
  a / c = 2 / 15 := by
  sorry

end NUMINAMATH_CALUDE_ratio_composition_l3631_363169


namespace NUMINAMATH_CALUDE_partial_fraction_decomposition_l3631_363116

theorem partial_fraction_decomposition :
  ∃ (C D : ℚ), C = 32/9 ∧ D = 13/9 ∧
  ∀ x : ℚ, x ≠ 7 ∧ x ≠ -2 →
    (5*x - 3) / (x^2 - 5*x - 14) = C / (x - 7) + D / (x + 2) :=
by sorry

end NUMINAMATH_CALUDE_partial_fraction_decomposition_l3631_363116


namespace NUMINAMATH_CALUDE_existence_of_counterexample_l3631_363129

theorem existence_of_counterexample :
  ∃ (a b c : ℝ), 
    a > 0 ∧ b > 0 ∧ c > 0 ∧ 
    a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
    b / a ≥ (b + c) / (a + c) :=
by sorry

end NUMINAMATH_CALUDE_existence_of_counterexample_l3631_363129


namespace NUMINAMATH_CALUDE_smallest_n_for_equations_l3631_363180

theorem smallest_n_for_equations :
  (∃ (n : ℕ), 
    (∀ (m : ℕ), m < n → ¬∃ (x y : ℕ), x * (x + m) = y^2) ∧
    (∃ (x y : ℕ), x * (x + n) = y^2) ∧
    n = 3) ∧
  (∃ (n : ℕ),
    (∀ (m : ℕ), m < n → ¬∃ (x y : ℕ), x * (x + m) = y^3) ∧
    (∃ (x y : ℕ), x * (x + n) = y^3) ∧
    n = 2) :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_for_equations_l3631_363180


namespace NUMINAMATH_CALUDE_problem_surface_area_l3631_363122

/-- Represents a solid block formed by unit cubes -/
structure SolidBlock where
  base_width : ℕ
  base_length : ℕ
  base_height : ℕ
  top_cubes : ℕ

/-- Calculates the surface area of a SolidBlock -/
def surface_area (block : SolidBlock) : ℕ :=
  sorry

/-- The specific solid block described in the problem -/
def problem_block : SolidBlock :=
  { base_width := 3
  , base_length := 2
  , base_height := 2
  , top_cubes := 2 }

theorem problem_surface_area : surface_area problem_block = 42 := by
  sorry

end NUMINAMATH_CALUDE_problem_surface_area_l3631_363122


namespace NUMINAMATH_CALUDE_refrigerator_savings_l3631_363134

/-- Calculates the savings from switching to a more energy-efficient refrigerator -/
theorem refrigerator_savings 
  (old_cost : ℝ) 
  (new_cost : ℝ) 
  (days : ℕ) 
  (h1 : old_cost = 0.85) 
  (h2 : new_cost = 0.45) 
  (h3 : days = 30) : 
  (old_cost * days) - (new_cost * days) = 12 :=
by sorry

end NUMINAMATH_CALUDE_refrigerator_savings_l3631_363134


namespace NUMINAMATH_CALUDE_statement_correctness_l3631_363156

theorem statement_correctness : 
  (∀ a b : ℝ, a > b ∧ b > 0 → a^2 > b^2) ∧
  (∃ a b : ℝ, a^2 > b^2 ∧ ¬(a > b ∧ b > 0)) ∧
  ¬(∀ a b : ℝ, (a > b ∧ b > 0 ↔ 1/a < 1/b)) ∧
  ¬(∀ a b : ℝ, (a > b ∧ b > 0 ↔ a^3 > b^3)) :=
by sorry

end NUMINAMATH_CALUDE_statement_correctness_l3631_363156


namespace NUMINAMATH_CALUDE_candle_lighting_time_l3631_363145

/-- The time (in minutes) when the candles are lit before 5 PM -/
def lighting_time : ℝ := 218

/-- The length of time (in minutes) it takes for the first candle to burn out completely -/
def burn_time_1 : ℝ := 240

/-- The length of time (in minutes) it takes for the second candle to burn out completely -/
def burn_time_2 : ℝ := 300

/-- The ratio of the length of the longer stub to the shorter stub at 5 PM -/
def stub_ratio : ℝ := 3

theorem candle_lighting_time :
  (burn_time_2 - lighting_time) / burn_time_2 = stub_ratio * ((burn_time_1 - lighting_time) / burn_time_1) :=
sorry

end NUMINAMATH_CALUDE_candle_lighting_time_l3631_363145


namespace NUMINAMATH_CALUDE_min_value_of_roots_squared_difference_l3631_363142

theorem min_value_of_roots_squared_difference (a : ℝ) (m n : ℝ) 
  (h1 : a ≥ 1)
  (h2 : m^2 - 2*a*m + 1 = 0)
  (h3 : n^2 - 2*a*n + 1 = 0) :
  ∃ (k : ℝ), k = (m - 1)^2 + (n - 1)^2 ∧ k ≥ 0 ∧ ∀ (x : ℝ), x = (m - 1)^2 + (n - 1)^2 → x ≥ k :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_roots_squared_difference_l3631_363142


namespace NUMINAMATH_CALUDE_sqrt_53_between_consecutive_integers_l3631_363164

theorem sqrt_53_between_consecutive_integers :
  ∃ (n : ℕ), n > 0 ∧ (n : ℝ)^2 < 53 ∧ 53 < (n + 1 : ℝ)^2 ∧ n * (n + 1) = 56 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_53_between_consecutive_integers_l3631_363164


namespace NUMINAMATH_CALUDE_inequality_proof_l3631_363154

theorem inequality_proof (a b c : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0)
  (h_sum : a^3 + b^3 + c^3 = 3) :
  (1 / (a^2 + a + 1)) + (1 / (b^2 + b + 1)) + (1 / (c^2 + c + 1)) ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3631_363154


namespace NUMINAMATH_CALUDE_blood_donation_selection_l3631_363130

theorem blood_donation_selection (o a b ab : ℕ) 
  (ho : o = 18) (ha : a = 10) (hb : b = 8) (hab : ab = 3) : 
  o * a * b * ab = 4320 := by
  sorry

end NUMINAMATH_CALUDE_blood_donation_selection_l3631_363130


namespace NUMINAMATH_CALUDE_james_money_calculation_l3631_363102

/-- Given 3 bills of $20 each and $75 already in a wallet, prove that the total amount is $135 -/
theorem james_money_calculation :
  let bills_count : ℕ := 3
  let bill_value : ℕ := 20
  let initial_wallet_amount : ℕ := 75
  bills_count * bill_value + initial_wallet_amount = 135 :=
by sorry

end NUMINAMATH_CALUDE_james_money_calculation_l3631_363102


namespace NUMINAMATH_CALUDE_star_polygon_is_pyramid_net_l3631_363159

/-- Represents a star-shaped polygon constructed from two concentric circles and an inscribed regular polygon -/
structure StarPolygon where
  R : ℝ  -- Radius of the larger circle
  r : ℝ  -- Radius of the smaller circle
  n : ℕ  -- Number of sides of the inscribed regular polygon
  h : R > r  -- Condition that the larger circle's radius is greater than the smaller circle's radius

/-- Determines whether a star-shaped polygon is the net of a pyramid -/
def is_pyramid_net (s : StarPolygon) : Prop :=
  s.R > 2 * s.r

/-- Theorem stating the condition for a star-shaped polygon to be the net of a pyramid -/
theorem star_polygon_is_pyramid_net (s : StarPolygon) :
  is_pyramid_net s ↔ s.R > 2 * s.r :=
sorry

end NUMINAMATH_CALUDE_star_polygon_is_pyramid_net_l3631_363159


namespace NUMINAMATH_CALUDE_total_cost_train_and_bus_l3631_363107

/-- The cost of a train ride from town P to town Q -/
def train_cost : ℝ := 8.35

/-- The cost of a bus ride from town P to town Q -/
def bus_cost : ℝ := 1.50

/-- The difference in cost between a train ride and a bus ride -/
def cost_difference : ℝ := 6.85

theorem total_cost_train_and_bus :
  train_cost + bus_cost = 9.85 ∧
  train_cost = bus_cost + cost_difference :=
sorry

end NUMINAMATH_CALUDE_total_cost_train_and_bus_l3631_363107


namespace NUMINAMATH_CALUDE_ball_trajectory_l3631_363105

/-- A rectangle with side lengths 2a and 2b -/
structure Rectangle (α : Type*) [NormedAddCommGroup α] [NormedSpace ℝ α] where
  a : ℝ
  b : ℝ
  h : (5 : ℝ) * a = (3 : ℝ) * b

/-- The angle at which the ball is hit from corner A -/
def hitAngle (α : ℝ) := α

/-- The ball hits three different sides before reaching the center -/
def hitsThreeSides (rect : Rectangle ℝ) (α : ℝ) : Prop :=
  ∃ (p q r : ℝ × ℝ), 
    p ≠ q ∧ q ≠ r ∧ r ≠ p ∧
    (p.1 = 0 ∨ p.1 = 2*rect.a ∨ p.2 = 0 ∨ p.2 = 2*rect.b) ∧
    (q.1 = 0 ∨ q.1 = 2*rect.a ∨ q.2 = 0 ∨ q.2 = 2*rect.b) ∧
    (r.1 = 0 ∨ r.1 = 2*rect.a ∨ r.2 = 0 ∨ r.2 = 2*rect.b)

theorem ball_trajectory (rect : Rectangle ℝ) (α : ℝ) :
  hitsThreeSides rect α ↔ Real.tan α = 9/25 := by sorry

end NUMINAMATH_CALUDE_ball_trajectory_l3631_363105


namespace NUMINAMATH_CALUDE_circle_through_point_on_x_axis_l3631_363187

def circle_equation (center : ℝ × ℝ) (radius : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - center.1)^2 + (p.2 - center.2)^2 = radius^2}

theorem circle_through_point_on_x_axis 
  (center : ℝ × ℝ) 
  (h_center_on_x_axis : center.2 = 0) 
  (h_radius : radius = 1) 
  (h_through_point : (2, 1) ∈ circle_equation center radius) :
  circle_equation center radius = circle_equation (2, 0) 1 := by
sorry

end NUMINAMATH_CALUDE_circle_through_point_on_x_axis_l3631_363187


namespace NUMINAMATH_CALUDE_cos_of_tan_in_third_quadrant_l3631_363103

/-- Prove that for an angle α in the third quadrant with tan α = 4/3, cos α = -3/5 -/
theorem cos_of_tan_in_third_quadrant (α : Real) 
  (h1 : π < α ∧ α < 3*π/2)  -- α is in the third quadrant
  (h2 : Real.tan α = 4/3) : 
  Real.cos α = -3/5 := by
  sorry

end NUMINAMATH_CALUDE_cos_of_tan_in_third_quadrant_l3631_363103


namespace NUMINAMATH_CALUDE_regular_pentagon_side_length_l3631_363163

/-- A regular pentagon with a perimeter of 23.4 cm has sides of length 4.68 cm. -/
theorem regular_pentagon_side_length : 
  ∀ (p : ℝ) (s : ℝ), 
  p = 23.4 →  -- perimeter is 23.4 cm
  s = p / 5 →  -- side length is perimeter divided by 5 (number of sides in a pentagon)
  s = 4.68 := by
sorry

end NUMINAMATH_CALUDE_regular_pentagon_side_length_l3631_363163


namespace NUMINAMATH_CALUDE_integer_expression_l3631_363198

theorem integer_expression (m : ℤ) : ∃ k : ℤ, (m / 3 + m^2 / 2 + m^3 / 6 : ℚ) = k := by
  sorry

end NUMINAMATH_CALUDE_integer_expression_l3631_363198


namespace NUMINAMATH_CALUDE_absolute_value_difference_l3631_363126

theorem absolute_value_difference (a b : ℝ) : 
  (a < b → |a - b| = b - a) ∧ (a ≥ b → |a - b| = a - b) := by sorry

end NUMINAMATH_CALUDE_absolute_value_difference_l3631_363126


namespace NUMINAMATH_CALUDE_gcd_13n_plus_3_7n_plus_1_max_exists_gcd_13n_plus_3_7n_plus_1_eq_8_l3631_363192

theorem gcd_13n_plus_3_7n_plus_1_max (n : ℕ+) : Nat.gcd (13 * n + 3) (7 * n + 1) ≤ 8 :=
sorry

theorem exists_gcd_13n_plus_3_7n_plus_1_eq_8 : ∃ (n : ℕ+), Nat.gcd (13 * n + 3) (7 * n + 1) = 8 :=
sorry

end NUMINAMATH_CALUDE_gcd_13n_plus_3_7n_plus_1_max_exists_gcd_13n_plus_3_7n_plus_1_eq_8_l3631_363192


namespace NUMINAMATH_CALUDE_age_difference_proof_l3631_363113

theorem age_difference_proof (younger_age elder_age : ℕ) : 
  younger_age = 35 →
  elder_age - 15 = 2 * (younger_age - 15) →
  elder_age - younger_age = 20 :=
by
  sorry

end NUMINAMATH_CALUDE_age_difference_proof_l3631_363113
