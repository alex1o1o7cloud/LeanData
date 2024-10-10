import Mathlib

namespace fibonacci_13th_term_l223_22380

def fibonacci : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | (n + 2) => fibonacci n + fibonacci (n + 1)

theorem fibonacci_13th_term : fibonacci 6 = 13 := by
  sorry

end fibonacci_13th_term_l223_22380


namespace complex_equation_solution_l223_22326

theorem complex_equation_solution (a b : ℝ) : 
  (Complex.I : ℂ) * (Complex.I : ℂ) = -1 →
  (1 + 2 * Complex.I) / (a + b * Complex.I) = 1 + Complex.I →
  a = 3/2 ∧ b = 1/2 :=
sorry

end complex_equation_solution_l223_22326


namespace digit_sum_problem_l223_22336

theorem digit_sum_problem (x y z w : ℕ) : 
  (x ≠ y ∧ x ≠ z ∧ x ≠ w ∧ y ≠ z ∧ y ≠ w ∧ z ≠ w) →
  (x < 10 ∧ y < 10 ∧ z < 10 ∧ w < 10) →
  (100 * x + 10 * y + z) + (100 * w + 10 * z + x) = 1000 →
  z + x ≥ 10 →
  y + z < 10 →
  x + y + z + w = 19 := by
sorry

end digit_sum_problem_l223_22336


namespace rectangle_perimeter_l223_22392

/-- Given a square with perimeter 100 units divided vertically into 4 congruent rectangles,
    the perimeter of one of these rectangles is 62.5 units. -/
theorem rectangle_perimeter (s : ℝ) (h1 : s > 0) (h2 : 4 * s = 100) : 
  2 * (s + s / 4) = 62.5 := by
  sorry

end rectangle_perimeter_l223_22392


namespace min_a_plus_b_l223_22300

theorem min_a_plus_b (x y a b : ℝ) : 
  2*x - y + 2 ≥ 0 →
  8*x - y - 4 ≤ 0 →
  x ≥ 0 →
  y ≥ 0 →
  a > 0 →
  b > 0 →
  (∀ x' y', 2*x' - y' + 2 ≥ 0 → 8*x' - y' - 4 ≤ 0 → x' ≥ 0 → y' ≥ 0 → a*x' + y' ≤ 8) →
  a*x + y = 8 →
  a + b ≥ 4 :=
sorry

end min_a_plus_b_l223_22300


namespace negate_all_men_are_good_drivers_l223_22315

-- Define the universe of discourse
variable (U : Type)

-- Define predicates
variable (Man : U → Prop)
variable (GoodDriver : U → Prop)

-- Define the statements
def AllMenAreGoodDrivers : Prop := ∀ x, Man x → GoodDriver x
def AtLeastOneManIsBadDriver : Prop := ∃ x, Man x ∧ ¬GoodDriver x

-- Theorem to prove
theorem negate_all_men_are_good_drivers :
  AtLeastOneManIsBadDriver U Man GoodDriver ↔ ¬(AllMenAreGoodDrivers U Man GoodDriver) :=
sorry

end negate_all_men_are_good_drivers_l223_22315


namespace tangent_line_equation_l223_22351

/-- The equation of the tangent line to y = x^3 + 2x at (1, 3) is 5x - y - 2 = 0 -/
theorem tangent_line_equation : 
  let f (x : ℝ) := x^3 + 2*x
  let P : ℝ × ℝ := (1, 3)
  ∃ (m b : ℝ), 
    (∀ x y, y = m*x + b ↔ m*x - y + b = 0) ∧ 
    (f P.1 = P.2) ∧
    (∀ x, x ≠ P.1 → (f x - P.2) / (x - P.1) ≠ m) ∧
    m*P.1 - P.2 + b = 0 ∧
    m = 5 ∧ b = -2 := by
  sorry

end tangent_line_equation_l223_22351


namespace odd_function_properties_l223_22354

noncomputable def f (x : ℝ) : ℝ :=
  if -1 < x ∧ x < 0 then -2^(-x)
  else if x = 0 then 0
  else if 0 < x ∧ x < 1 then 2^x
  else 0

theorem odd_function_properties (a : ℝ) :
  (∀ x ∈ Set.Ioo (-1 : ℝ) 1, f (-x) = -f x) →
  (∀ x ∈ Set.Ioo (0 : ℝ) 1, f x = 2^x) →
  (∀ x ∈ Set.Ioo (-1 : ℝ) 1, f x ≤ 2*a) →
  (∀ x ∈ Set.Ioo (-1 : ℝ) 0, f x = -2^(-x)) ∧
  (f 0 = 0) ∧
  (∀ x ∈ Set.Ioo (0 : ℝ) 1, f x = 2^x) ∧
  (a ≥ 1) := by sorry

end odd_function_properties_l223_22354


namespace trapezoid_semicircle_area_l223_22319

/-- Represents a trapezoid with semicircles on each side -/
structure TrapezoidWithSemicircles where
  side1 : ℝ
  side2 : ℝ
  side3 : ℝ
  side4 : ℝ

/-- Calculates the area of the region bounded by the semicircles -/
noncomputable def boundedArea (t : TrapezoidWithSemicircles) : ℝ :=
  sorry

/-- The theorem to be proved -/
theorem trapezoid_semicircle_area 
  (t : TrapezoidWithSemicircles) 
  (h1 : t.side1 = 10) 
  (h2 : t.side2 = 10) 
  (h3 : t.side3 = 10) 
  (h4 : t.side4 = 22) : 
  boundedArea t = 128 + 60.5 * Real.pi := by
  sorry

end trapezoid_semicircle_area_l223_22319


namespace eight_pencils_l223_22352

/-- Represents Sam's pen and pencil collection -/
structure SamsCollection where
  pencils : ℕ
  blue_pens : ℕ
  black_pens : ℕ
  red_pens : ℕ

/-- The conditions of Sam's collection -/
def valid_collection (c : SamsCollection) : Prop :=
  c.black_pens = c.blue_pens + 10 ∧
  c.blue_pens = 2 * c.pencils ∧
  c.red_pens = c.pencils - 2 ∧
  c.black_pens + c.blue_pens + c.red_pens = 48

/-- Theorem stating that in a valid collection, there are 8 pencils -/
theorem eight_pencils (c : SamsCollection) (h : valid_collection c) : c.pencils = 8 := by
  sorry

end eight_pencils_l223_22352


namespace ball_count_l223_22321

theorem ball_count (white blue red : ℕ) : 
  blue = white + 12 →
  red = 2 * blue →
  white = 16 →
  white + blue + red = 100 :=
by
  sorry

end ball_count_l223_22321


namespace smallest_sum_proof_l223_22345

theorem smallest_sum_proof : 
  let sums : List ℚ := [1/3 + 1/4, 1/3 + 1/5, 1/3 + 1/6, 1/3 + 1/7, 1/3 + 1/8]
  (∀ x ∈ sums, 1/3 + 1/8 ≤ x) ∧ (1/3 + 1/8 = 11/24) := by
  sorry

end smallest_sum_proof_l223_22345


namespace complex_equation_solution_l223_22378

theorem complex_equation_solution (z : ℂ) : (1 + Complex.I) * z = 2 → z = 1 - Complex.I := by
  sorry

end complex_equation_solution_l223_22378


namespace minimum_bailing_rate_l223_22350

/-- Proves that the minimum bailing rate to reach shore without sinking is 10.75 gallons per minute --/
theorem minimum_bailing_rate 
  (distance : ℝ) 
  (intake_rate : ℝ) 
  (max_water : ℝ) 
  (rowing_speed : ℝ) 
  (h1 : distance = 2) 
  (h2 : intake_rate = 12) 
  (h3 : max_water = 50) 
  (h4 : rowing_speed = 3) : 
  ∃ (bailing_rate : ℝ), 
    bailing_rate ≥ 10.75 ∧ 
    bailing_rate < intake_rate ∧
    (distance / rowing_speed) * 60 * (intake_rate - bailing_rate) ≤ max_water :=
by sorry

end minimum_bailing_rate_l223_22350


namespace odometer_puzzle_l223_22395

theorem odometer_puzzle (a b c : ℕ) 
  (h1 : a ≥ 1) 
  (h2 : 100 ≤ a * b * c ∧ a * b * c ≤ 300)
  (h3 : 75 ∣ b)
  (h4 : (a * b * c) + b - a * b * c = b) :
  a^2 + b^2 + c^2 = 5635 := by
sorry

end odometer_puzzle_l223_22395


namespace volume_equals_target_l223_22357

/-- Represents a rectangular parallelepiped -/
structure Parallelepiped where
  length : ℝ
  width : ℝ
  height : ℝ

/-- Calculates the volume of points inside or within one unit of a parallelepiped -/
def volume_with_buffer (p : Parallelepiped) : ℝ := sorry

/-- The specific parallelepiped in the problem -/
def problem_parallelepiped : Parallelepiped :=
  { length := 2,
    width := 3,
    height := 4 }

theorem volume_equals_target : 
  volume_with_buffer problem_parallelepiped = (456 + 31 * Real.pi) / 6 := by sorry

end volume_equals_target_l223_22357


namespace inscribed_cylinder_radius_l223_22384

/-- Represents a right circular cone -/
structure Cone where
  diameter : ℝ
  altitude : ℝ

/-- Represents a right circular cylinder -/
structure Cylinder where
  radius : ℝ

/-- Represents the configuration of a cylinder inscribed in a cone -/
structure InscribedCylinder where
  cone : Cone
  cylinder : Cylinder
  height_radius_ratio : ℝ
  axes_coincide : Bool

/-- Theorem stating the radius of the inscribed cylinder -/
theorem inscribed_cylinder_radius 
  (ic : InscribedCylinder) 
  (h1 : ic.cone.diameter = 12) 
  (h2 : ic.cone.altitude = 15) 
  (h3 : ic.height_radius_ratio = 3) 
  (h4 : ic.axes_coincide = true) : 
  ic.cylinder.radius = 30 / 11 := by
  sorry

end inscribed_cylinder_radius_l223_22384


namespace fibonacci_arithmetic_sequence_l223_22329

-- Define the Fibonacci sequence
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | n + 2 => fib (n + 1) + fib n

-- Define the property of arithmetic sequence for Fibonacci numbers
def is_arithmetic_sequence (a : ℕ) : Prop :=
  fib (a + 4) = 2 * fib (a + 2) - fib a

-- Define the sum condition
def sum_condition (a : ℕ) : Prop :=
  a + (a + 2) + (a + 4) = 2500

-- Theorem statement
theorem fibonacci_arithmetic_sequence :
  ∃ a : ℕ, is_arithmetic_sequence a ∧ sum_condition a ∧ a = 831 := by
  sorry

end fibonacci_arithmetic_sequence_l223_22329


namespace intersection_P_complement_Q_l223_22324

-- Define the sets P and Q
def P : Set ℝ := {x | x^2 < 1}
def Q : Set ℝ := {x | x ≥ 0}

-- State the theorem
theorem intersection_P_complement_Q : 
  P ∩ (Set.univ \ Q) = {x : ℝ | -1 < x ∧ x < 0} := by sorry

end intersection_P_complement_Q_l223_22324


namespace pizza_toppings_combinations_l223_22377

theorem pizza_toppings_combinations : Nat.choose 7 3 = 35 := by
  sorry

end pizza_toppings_combinations_l223_22377


namespace remaining_payment_proof_l223_22347

/-- Given a deposit percentage and deposit amount, calculates the remaining amount to be paid -/
def remaining_payment (deposit_percentage : ℚ) (deposit_amount : ℚ) : ℚ :=
  (deposit_amount / deposit_percentage) - deposit_amount

/-- Proves that the remaining amount to be paid is 990, given a 10% deposit of 110 -/
theorem remaining_payment_proof : 
  remaining_payment (1/10) 110 = 990 := by
  sorry

end remaining_payment_proof_l223_22347


namespace basketball_game_probability_basketball_game_probability_is_one_l223_22358

/-- The probability that at least 4 people stay for the entire game, given that
    8 people come to a basketball game, 4 are certain to stay, and 4 have a
    1/3 probability of staying. -/
theorem basketball_game_probability : Real :=
  let total_people : ℕ := 8
  let certain_stayers : ℕ := 4
  let uncertain_stayers : ℕ := 4
  let stay_probability : Real := 1/3
  1

/-- Proof that the probability is indeed 1. -/
theorem basketball_game_probability_is_one :
  basketball_game_probability = 1 := by
  sorry

end basketball_game_probability_basketball_game_probability_is_one_l223_22358


namespace at_least_two_primes_of_form_l223_22323

theorem at_least_two_primes_of_form (n : ℕ) : ∃ (a b : ℕ), 2 ≤ a ∧ 2 ≤ b ∧ a ≠ b ∧ 
  Nat.Prime (a^3 + a + 1) ∧ Nat.Prime (b^3 + b + 1) :=
sorry

end at_least_two_primes_of_form_l223_22323


namespace sin_half_angle_second_quadrant_l223_22385

theorem sin_half_angle_second_quadrant (θ : Real) 
  (h1 : π < θ ∧ θ < 3*π/2) 
  (h2 : 25 * Real.sin θ ^ 2 + Real.sin θ - 24 = 0) : 
  Real.sin (θ/2) = 4/5 ∨ Real.sin (θ/2) = -4/5 := by
sorry

end sin_half_angle_second_quadrant_l223_22385


namespace marble_selection_probability_l223_22399

/-- The probability of selecting 2 red, 1 blue, and 1 green marble when choosing 4 marbles
    without replacement from a bag containing 3 red, 3 blue, and 3 green marbles. -/
theorem marble_selection_probability :
  let total_marbles : ℕ := 9
  let red_marbles : ℕ := 3
  let blue_marbles : ℕ := 3
  let green_marbles : ℕ := 3
  let selected_marbles : ℕ := 4
  let favorable_outcomes : ℕ := Nat.choose red_marbles 2 * Nat.choose blue_marbles 1 * Nat.choose green_marbles 1
  let total_outcomes : ℕ := Nat.choose total_marbles selected_marbles
  (favorable_outcomes : ℚ) / total_outcomes = 3 / 14 :=
by sorry

end marble_selection_probability_l223_22399


namespace parallelogram_side_length_l223_22335

theorem parallelogram_side_length (s : ℝ) : 
  s > 0 → -- side length is positive
  let angle : ℝ := 30 * π / 180 -- 30 degrees in radians
  let area : ℝ := 12 * Real.sqrt 3 -- area of the parallelogram
  s * (s * Real.sin angle) = area → -- area formula for parallelogram
  s = 2 * Real.sqrt 6 := by
sorry

end parallelogram_side_length_l223_22335


namespace square_difference_fourth_power_l223_22303

theorem square_difference_fourth_power : (7^2 - 5^2)^4 = 331776 := by
  sorry

end square_difference_fourth_power_l223_22303


namespace forty_percent_relation_l223_22371

theorem forty_percent_relation (x : ℝ) (v : ℝ) : 
  (1/4 : ℝ) * (1/3 : ℝ) * (2/5 : ℝ) * x = v → (40/100 : ℝ) * x = 12 * v := by
  sorry

end forty_percent_relation_l223_22371


namespace equal_tasks_after_transfer_l223_22306

/-- Given that Robyn has 4 tasks and Sasha has 14 tasks, prove that if Robyn takes 5 tasks from Sasha, they will have an equal number of tasks. -/
theorem equal_tasks_after_transfer (robyn_initial : Nat) (sasha_initial : Nat) (tasks_transferred : Nat) : 
  robyn_initial = 4 → 
  sasha_initial = 14 → 
  tasks_transferred = 5 → 
  (robyn_initial + tasks_transferred = sasha_initial - tasks_transferred) := by
  sorry

#check equal_tasks_after_transfer

end equal_tasks_after_transfer_l223_22306


namespace three_reflection_theorem_l223_22391

/-- A circular billiard table -/
structure BilliardTable where
  R : ℝ
  R_pos : R > 0

/-- A point on the billiard table -/
structure Point (bt : BilliardTable) where
  x : ℝ
  y : ℝ
  on_table : x^2 + y^2 ≤ bt.R^2

/-- Predicate for a valid starting point A -/
def valid_start_point (bt : BilliardTable) (A : Point bt) : Prop :=
  A.x^2 + A.y^2 > (bt.R/3)^2 ∧ A.x^2 + A.y^2 < bt.R^2

/-- Predicate for a valid reflection path -/
def valid_reflection_path (bt : BilliardTable) (A : Point bt) : Prop :=
  ∃ (B C : Point bt),
    B ≠ A ∧ C ≠ A ∧ B ≠ C ∧
    (A.x^2 + A.y^2 = B.x^2 + B.y^2) ∧
    (B.x^2 + B.y^2 = C.x^2 + C.y^2) ∧
    (C.x^2 + C.y^2 = A.x^2 + A.y^2)

theorem three_reflection_theorem (bt : BilliardTable) (A : Point bt) :
  valid_start_point bt A ↔ valid_reflection_path bt A :=
sorry

end three_reflection_theorem_l223_22391


namespace unique_number_with_three_prime_factors_l223_22328

theorem unique_number_with_three_prime_factors (x n : ℕ) : 
  x = 5^n - 1 ∧ 
  (∃ p q : ℕ, Nat.Prime p ∧ Nat.Prime q ∧ p ≠ q ∧ p ≠ 11 ∧ q ≠ 11 ∧ 
    x = 2^(Nat.log 2 x) * 11 * p * q) →
  x = 3124 :=
by sorry

end unique_number_with_three_prime_factors_l223_22328


namespace triangle_with_long_altitudes_l223_22341

theorem triangle_with_long_altitudes (a b c : ℝ) (ma mb : ℝ) (h_positive : a > 0 ∧ b > 0 ∧ c > 0)
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b)
  (h_altitudes : ma ≥ a ∧ mb ≥ b)
  (h_area : a * b * Real.sin (Real.arccos ((a^2 + b^2 - c^2) / (2 * a * b))) / 2 = a * ma / 2)
  (h_area_alt : a * b * Real.sin (Real.arccos ((a^2 + b^2 - c^2) / (2 * a * b))) / 2 = b * mb / 2) :
  a = b ∧ c^2 = 2 * a^2 :=
by sorry

end triangle_with_long_altitudes_l223_22341


namespace min_x_prime_factorization_sum_l223_22388

theorem min_x_prime_factorization_sum : ∃ (x y p q r : ℕ+) (a b c : ℕ),
  (3 : ℚ) * (x : ℚ)^7 = 5 * (y : ℚ)^11 ∧
  x = p^a * q^b * r^c ∧
  p ≠ q ∧ q ≠ r ∧ p ≠ r ∧
  (∀ (x' : ℕ+), (3 : ℚ) * (x' : ℚ)^7 = 5 * (y : ℚ)^11 → x ≤ x') →
  p + q + r + a + b + c = 24 :=
sorry

end min_x_prime_factorization_sum_l223_22388


namespace watch_selling_price_l223_22334

/-- Calculates the selling price of an item given its cost price and profit percentage. -/
def selling_price (cost_price : ℕ) (profit_percentage : ℕ) : ℕ :=
  cost_price + (cost_price * profit_percentage) / 100

/-- Proves that for a watch with a cost price of 90 rupees, 
    if the profit percentage is equal to the cost price, 
    then the selling price is 180 rupees. -/
theorem watch_selling_price : 
  let cost_price : ℕ := 90
  let profit_percentage : ℕ := 100
  selling_price cost_price profit_percentage = 180 := by
sorry


end watch_selling_price_l223_22334


namespace shoe_comparison_l223_22317

theorem shoe_comparison (bobby_shoes : ℕ) (bonny_shoes : ℕ) : 
  bobby_shoes = 27 →
  bonny_shoes = 13 →
  ∃ (becky_shoes : ℕ), 
    bobby_shoes = 3 * becky_shoes ∧
    2 * becky_shoes - bonny_shoes = 5 :=
by
  sorry

end shoe_comparison_l223_22317


namespace probability_between_C_and_D_l223_22375

/-- Given points A, B, C, D on a line segment AB where AB = 4AD and AB = 8BC,
    prove that the probability of a randomly selected point on AB
    being between C and D is 5/8. -/
theorem probability_between_C_and_D (A B C D : ℝ) : 
  A < C ∧ C < D ∧ D < B →  -- Points are in order on the line segment
  B - A = 4 * (D - A) →    -- AB = 4AD
  B - A = 8 * (C - B) →    -- AB = 8BC
  (D - C) / (B - A) = 5/8 := by sorry

end probability_between_C_and_D_l223_22375


namespace student_distribution_problem_l223_22389

/-- The number of ways to distribute n distinguishable students among k distinguishable schools,
    with each school receiving at least one student. -/
def distribute_students (n k : ℕ) : ℕ :=
  if n < k then 0
  else (k.choose 2) * k.factorial

/-- The problem statement -/
theorem student_distribution_problem :
  distribute_students 4 3 = 36 := by
  sorry

end student_distribution_problem_l223_22389


namespace binary_1101_to_base5_l223_22360

/-- Converts a binary number represented as a list of bits to its decimal equivalent -/
def binary_to_decimal (bits : List Bool) : ℕ :=
  bits.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

/-- Converts a natural number to its base-5 representation -/
def decimal_to_base5 (n : ℕ) : List ℕ :=
  if n < 5 then [n]
  else (n % 5) :: decimal_to_base5 (n / 5)

/-- The binary representation of the number we want to convert -/
def binary_1101 : List Bool := [true, false, true, true]

theorem binary_1101_to_base5 :
  decimal_to_base5 (binary_to_decimal binary_1101) = [3, 2] :=
by sorry

end binary_1101_to_base5_l223_22360


namespace sum_distinct_remainders_divided_by_13_l223_22367

def distinct_remainders (n : ℕ) : Finset ℕ :=
  (Finset.range n).image (λ i => (i + 1)^2 % 13)

theorem sum_distinct_remainders_divided_by_13 :
  (Finset.sum (distinct_remainders 12) id) / 13 = 3 :=
sorry

end sum_distinct_remainders_divided_by_13_l223_22367


namespace fermat_point_sum_l223_22379

theorem fermat_point_sum (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h1 : x^2 + y^2 + x*y = 1)
  (h2 : y^2 + z^2 + y*z = 2)
  (h3 : z^2 + x^2 + z*x = 3) :
  x + y + z = Real.sqrt (3 + Real.sqrt 6) :=
sorry

end fermat_point_sum_l223_22379


namespace quadratic_factorization_l223_22320

theorem quadratic_factorization (x : ℝ) : 4 - 4*x + x^2 = (2 - x)^2 := by
  sorry

end quadratic_factorization_l223_22320


namespace damaged_glassware_count_l223_22386

-- Define the constants from the problem
def total_glassware : ℕ := 1500
def undamaged_fee : ℚ := 5/2
def damaged_fee : ℕ := 3
def total_received : ℕ := 3618

-- Define the theorem
theorem damaged_glassware_count :
  ∃ x : ℕ, 
    x ≤ total_glassware ∧ 
    (undamaged_fee * (total_glassware - x) : ℚ) - (damaged_fee * x : ℚ) = total_received ∧
    x = 24 := by
  sorry

end damaged_glassware_count_l223_22386


namespace max_intersections_theorem_l223_22304

/-- A convex polygon in a plane -/
structure ConvexPolygon where
  sides : ℕ
  convex : Bool

/-- Represents the configuration of two convex polygons in a plane -/
structure PolygonConfiguration where
  P1 : ConvexPolygon
  P2 : ConvexPolygon
  no_common_segment : Bool
  h : P1.sides ≤ P2.sides

/-- The maximum number of intersections between two convex polygons -/
def max_intersections (config : PolygonConfiguration) : ℕ :=
  config.P1.sides * config.P2.sides

/-- Theorem stating the maximum number of intersections between two convex polygons -/
theorem max_intersections_theorem (config : PolygonConfiguration) :
  config.P1.convex ∧ config.P2.convex ∧ config.no_common_segment →
  max_intersections config = config.P1.sides * config.P2.sides :=
by sorry

end max_intersections_theorem_l223_22304


namespace unique_angle_solution_l223_22373

theorem unique_angle_solution :
  ∃! x : ℝ, 0 < x ∧ x < 180 ∧
  Real.tan ((150 - x) * π / 180) =
    (Real.sin (150 * π / 180) - Real.sin (x * π / 180)) /
    (Real.cos (150 * π / 180) - Real.cos (x * π / 180)) ∧
  x = 130 := by
sorry

end unique_angle_solution_l223_22373


namespace f_composition_negative_two_l223_22396

-- Define the piecewise function f
noncomputable def f (x : ℝ) : ℝ :=
  if x > 0 then Real.log x else (10 : ℝ) ^ x

-- State the theorem
theorem f_composition_negative_two :
  f (f (-2)) = -2 := by
  sorry

end f_composition_negative_two_l223_22396


namespace exists_range_sum_and_even_count_l223_22308

/-- Sum of integers from n to m, inclusive -/
def sum_range (n m : ℤ) : ℤ := (m - n + 1) * (n + m) / 2

/-- Number of even integers from n to m, inclusive -/
def count_even (n m : ℤ) : ℤ :=
  if (n % 2 = m % 2) then (m - n) / 2 + 1 else (m - n + 1) / 2

/-- Theorem stating the existence of a range satisfying the given conditions -/
theorem exists_range_sum_and_even_count :
  ∃ (n m : ℤ), n ≤ m ∧ sum_range n m + count_even n m = 641 :=
sorry

end exists_range_sum_and_even_count_l223_22308


namespace prism_volume_proof_l223_22394

/-- The volume of a right rectangular prism with face areas 28, 45, and 63 square centimeters -/
def prism_volume : ℝ := 282

theorem prism_volume_proof (x y z : ℝ) 
  (face1 : x * y = 28)
  (face2 : x * z = 45)
  (face3 : y * z = 63) :
  x * y * z = prism_volume := by
  sorry

end prism_volume_proof_l223_22394


namespace expected_value_special_coin_l223_22344

/-- The expected value of winnings for a special coin flip -/
theorem expected_value_special_coin : 
  let p_heads : ℚ := 2 / 5
  let p_tails : ℚ := 3 / 5
  let win_heads : ℚ := 4
  let lose_tails : ℚ := 3
  p_heads * win_heads - p_tails * lose_tails = -1 / 5 := by
sorry

end expected_value_special_coin_l223_22344


namespace longest_side_of_triangle_l223_22364

-- Define the triangle ABC
def triangle_ABC (A B C : ℝ) (a b c : ℝ) : Prop :=
  A + B + C = 180 ∧ -- Angle sum theorem
  a / Real.sin A = b / Real.sin B ∧ -- Sine rule
  a / Real.sin A = c / Real.sin C -- Sine rule

-- State the theorem
theorem longest_side_of_triangle :
  ∀ (A B C a b c : ℝ),
  triangle_ABC A B C a b c →
  B = 135 →
  C = 15 →
  a = 5 →
  b = 5 * Real.sqrt 2 :=
by sorry

end longest_side_of_triangle_l223_22364


namespace peters_exam_score_l223_22309

theorem peters_exam_score :
  ∀ (e m h : ℕ),
  e + m + h = 25 →
  2 * e + 3 * m + 5 * h = 84 →
  m % 2 = 0 →
  h % 3 = 0 →
  2 * e + 3 * (m / 2) + 5 * (h / 3) = 40 :=
by sorry

end peters_exam_score_l223_22309


namespace max_circles_in_square_l223_22343

/-- The maximum number of non-overlapping circles with radius 2 cm
    that can fit inside a square with side length 8 cm -/
def max_circles : ℕ := 4

/-- The side length of the square in cm -/
def square_side : ℝ := 8

/-- The radius of each circle in cm -/
def circle_radius : ℝ := 2

theorem max_circles_in_square :
  ∀ n : ℕ,
  (n : ℝ) * (2 * circle_radius) ≤ square_side →
  (n : ℝ) * (2 * circle_radius) > square_side - 2 * circle_radius →
  n * n = max_circles :=
by sorry

end max_circles_in_square_l223_22343


namespace delivery_driver_boxes_l223_22311

/-- Calculates the total number of boxes a delivery driver has -/
def total_boxes (num_stops : ℕ) (boxes_per_stop : ℕ) : ℕ :=
  num_stops * boxes_per_stop

/-- Proves that a delivery driver with 3 stops and 9 boxes per stop has 27 boxes in total -/
theorem delivery_driver_boxes :
  total_boxes 3 9 = 27 := by
  sorry

end delivery_driver_boxes_l223_22311


namespace cube_surface_area_l223_22339

/-- The surface area of a cube with edge length 4a is 96a² -/
theorem cube_surface_area (a : ℝ) : 
  let edge_length : ℝ := 4 * a
  let surface_area : ℝ := 6 * (edge_length ^ 2)
  surface_area = 96 * (a ^ 2) := by
  sorry

end cube_surface_area_l223_22339


namespace percentage_increase_l223_22362

theorem percentage_increase (initial : ℝ) (final : ℝ) : 
  initial = 500 → final = 650 → (final - initial) / initial * 100 = 30 := by
  sorry

end percentage_increase_l223_22362


namespace arithmetic_mean_after_removal_l223_22333

def original_set_size : ℕ := 60
def original_mean : ℚ := 42
def discarded_numbers : List ℚ := [50, 60, 70]

theorem arithmetic_mean_after_removal :
  let original_sum : ℚ := original_mean * original_set_size
  let remaining_sum : ℚ := original_sum - (discarded_numbers.sum)
  let remaining_set_size : ℕ := original_set_size - discarded_numbers.length
  (remaining_sum / remaining_set_size : ℚ) = 41 := by sorry

end arithmetic_mean_after_removal_l223_22333


namespace intersection_M_complement_N_l223_22346

def R : Set ℝ := Set.univ

def M : Set ℝ := {-1, 0, 1, 5}

def N : Set ℝ := {x : ℝ | x^2 - x - 2 ≥ 0}

theorem intersection_M_complement_N : M ∩ (R \ N) = {0, 1} := by sorry

end intersection_M_complement_N_l223_22346


namespace mold_diameter_l223_22372

/-- The diameter of a circular mold with radius 2 inches is 4 inches. -/
theorem mold_diameter (r : ℝ) (h : r = 2) : 2 * r = 4 := by
  sorry

end mold_diameter_l223_22372


namespace total_score_is_210_l223_22368

/-- Represents the test scores of three students -/
structure TestScores where
  total_questions : ℕ
  marks_per_question : ℕ
  jose_wrong_questions : ℕ
  meghan_diff : ℕ
  jose_alisson_diff : ℕ

/-- Calculates the total score for three students given their test performance -/
def calculate_total_score (scores : TestScores) : ℕ :=
  let total_marks := scores.total_questions * scores.marks_per_question
  let jose_score := total_marks - (scores.jose_wrong_questions * scores.marks_per_question)
  let meghan_score := jose_score - scores.meghan_diff
  let alisson_score := jose_score - scores.jose_alisson_diff
  jose_score + meghan_score + alisson_score

/-- Theorem stating that the total score for the three students is 210 marks -/
theorem total_score_is_210 (scores : TestScores) 
  (h1 : scores.total_questions = 50)
  (h2 : scores.marks_per_question = 2)
  (h3 : scores.jose_wrong_questions = 5)
  (h4 : scores.meghan_diff = 20)
  (h5 : scores.jose_alisson_diff = 40) :
  calculate_total_score scores = 210 := by
  sorry

end total_score_is_210_l223_22368


namespace geometric_sequence_product_l223_22376

def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

theorem geometric_sequence_product (a : ℕ → ℝ) :
  geometric_sequence a →
  (3 * a 3 ^ 2 - 11 * a 3 + 9 = 0) →
  (3 * a 9 ^ 2 - 11 * a 9 + 9 = 0) →
  (a 5 * a 6 * a 7 = 3 * Real.sqrt 3 ∨ a 5 * a 6 * a 7 = -3 * Real.sqrt 3) :=
by
  sorry


end geometric_sequence_product_l223_22376


namespace complex_magnitude_l223_22342

theorem complex_magnitude (z : ℂ) (h : Complex.I * z = -1 + Complex.I) : Complex.abs z = Real.sqrt 2 := by
  sorry

end complex_magnitude_l223_22342


namespace age_difference_l223_22353

theorem age_difference (A B : ℕ) : B = 38 → A + 10 = 2 * (B - 10) → A - B = 8 := by
  sorry

end age_difference_l223_22353


namespace diminished_value_proof_diminished_value_l223_22322

theorem diminished_value_proof (n : Nat) (divisors : List Nat) : Prop :=
  let smallest := 1013
  let value := 5
  let lcm := Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm 12 16) 18) 21) 28
  (∀ d ∈ divisors, (smallest - value) % d = 0) ∧
  (smallest = lcm + value) ∧
  (∀ m < smallest, ∃ d ∈ divisors, (m - value) % d ≠ 0)

/-- The value that needs to be diminished from 1013 to make it divisible by 12, 16, 18, 21, and 28 is 5. -/
theorem diminished_value :
  diminished_value_proof 1013 [12, 16, 18, 21, 28] :=
by sorry

end diminished_value_proof_diminished_value_l223_22322


namespace petya_strategy_works_l223_22330

/-- Represents a non-zero digit (1-9) -/
def NonZeroDigit := {n : Nat // 1 ≤ n ∧ n ≤ 9}

/-- Represents a 3-digit number -/
def ThreeDigitNumber := {n : Nat // 100 ≤ n ∧ n ≤ 999}

/-- The main theorem stating that any 12 non-zero digits can be arranged into four 3-digit numbers whose product is divisible by 9 -/
theorem petya_strategy_works (digits : Fin 12 → NonZeroDigit) : 
  ∃ (a b c d : ThreeDigitNumber), (a.val * b.val * c.val * d.val) % 9 = 0 := by
  sorry

end petya_strategy_works_l223_22330


namespace integer_triple_divisibility_l223_22310

theorem integer_triple_divisibility :
  ∀ a b c : ℤ,
    1 < a ∧ a < b ∧ b < c →
    (a - 1) * (b - 1) * (c - 1) ∣ a * b * c - 1 →
    ((a = 3 ∧ b = 5 ∧ c = 15) ∨ (a = 2 ∧ b = 4 ∧ c = 8)) :=
by sorry

end integer_triple_divisibility_l223_22310


namespace sara_gave_four_limes_l223_22359

def limes_from_sara (initial_limes final_limes : ℕ) : ℕ :=
  final_limes - initial_limes

theorem sara_gave_four_limes (initial_limes final_limes : ℕ) 
  (h1 : initial_limes = 9)
  (h2 : final_limes = 13) :
  limes_from_sara initial_limes final_limes = 4 := by
  sorry

end sara_gave_four_limes_l223_22359


namespace broken_line_length_formula_l223_22348

/-- Given an acute angle α and a point A₁ on one of its sides, we repeatedly drop perpendiculars
    to form an infinite broken line. This function represents the length of that line. -/
noncomputable def broken_line_length (α : Real) (m : Real) : Real :=
  m / (1 - Real.cos α)

/-- Theorem stating that the length of the infinite broken line formed by repeatedly dropping
    perpendiculars in an acute angle is equal to m / (1 - cos(α)), where m is the length of
    the first perpendicular and α is the magnitude of the angle. -/
theorem broken_line_length_formula (α : Real) (m : Real) 
    (h_acute : 0 < α ∧ α < Real.pi / 2) 
    (h_positive : m > 0) : 
  broken_line_length α m = m / (1 - Real.cos α) := by
  sorry

end broken_line_length_formula_l223_22348


namespace fgh_supermarkets_count_l223_22387

/-- The number of FGH supermarkets in the US -/
def us_supermarkets : ℕ := 42

/-- The number of FGH supermarkets in Canada -/
def canada_supermarkets : ℕ := us_supermarkets - 14

/-- The total number of FGH supermarkets -/
def total_supermarkets : ℕ := 70

theorem fgh_supermarkets_count :
  us_supermarkets = 42 ∧
  us_supermarkets + canada_supermarkets = total_supermarkets ∧
  us_supermarkets = canada_supermarkets + 14 := by
  sorry

end fgh_supermarkets_count_l223_22387


namespace symmetry_x_axis_correct_l223_22331

/-- A point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Symmetry with respect to the x-axis -/
def symmetricXAxis (p : Point3D) : Point3D :=
  { x := p.x, y := -p.y, z := -p.z }

theorem symmetry_x_axis_correct :
  let M : Point3D := { x := -1, y := 2, z := 1 }
  let M' : Point3D := { x := -1, y := -2, z := -1 }
  symmetricXAxis M = M' := by sorry

end symmetry_x_axis_correct_l223_22331


namespace sequence_sum_l223_22383

def geometric_sequence (a : ℕ → ℚ) := ∀ n, a (n + 1) = 2 * a n

def arithmetic_sequence (b : ℕ → ℚ) := ∃ d, ∀ n, b (n + 1) = b n + d

theorem sequence_sum (a : ℕ → ℚ) (b : ℕ → ℚ) :
  geometric_sequence a →
  a 2 * a 3 * a 4 = 27 / 64 →
  arithmetic_sequence b →
  b 7 = a 5 →
  b 3 + b 11 = 6 := by
  sorry

end sequence_sum_l223_22383


namespace solve_linear_equation_l223_22398

theorem solve_linear_equation :
  ∃ x : ℚ, 3*x - 5*x + 9*x + 4 = 289 ∧ x = 285/7 := by
  sorry

end solve_linear_equation_l223_22398


namespace product_as_difference_of_squares_l223_22325

theorem product_as_difference_of_squares (a b : ℝ) : 
  a * b = ((a + b) / 2)^2 - ((a - b) / 2)^2 := by
  sorry

end product_as_difference_of_squares_l223_22325


namespace nested_sqrt_value_l223_22397

theorem nested_sqrt_value : 
  ∃ x : ℝ, x = Real.sqrt (3 - x) ∧ x = (-1 + Real.sqrt 13) / 2 := by
sorry

end nested_sqrt_value_l223_22397


namespace point_four_units_from_origin_l223_22374

theorem point_four_units_from_origin (x : ℝ) : 
  |x| = 4 → x = 4 ∨ x = -4 := by
sorry

end point_four_units_from_origin_l223_22374


namespace decreasing_interval_of_f_decreasing_interval_is_open_interval_l223_22318

-- Define the function
def f (x : ℝ) := x^3 - 3*x

-- Define the derivative of the function
def f' (x : ℝ) := 3*x^2 - 3

-- Theorem statement
theorem decreasing_interval_of_f :
  ∀ x : ℝ, (f' x < 0) ↔ (-1 < x ∧ x < 1) :=
sorry

-- Main theorem
theorem decreasing_interval_is_open_interval :
  {x : ℝ | ∀ y : ℝ, -1 < y ∧ y < x → f y > f x} = Set.Ioo (-1) 1 :=
sorry

end decreasing_interval_of_f_decreasing_interval_is_open_interval_l223_22318


namespace period_of_symmetric_function_l223_22365

/-- A function f is symmetric about a point c if f(c + x) = f(c - x) for all x -/
def SymmetricAbout (f : ℝ → ℝ) (c : ℝ) : Prop :=
  ∀ x, f (c + x) = f (c - x)

/-- A real number p is a period of a function f if f(x + p) = f(x) for all x -/
def IsPeriod (f : ℝ → ℝ) (p : ℝ) : Prop :=
  ∀ x, f (x + p) = f x

theorem period_of_symmetric_function (f : ℝ → ℝ) (a b : ℝ) 
    (h1 : SymmetricAbout (fun x ↦ f (2 * x)) (a / 2)) 
    (h2 : SymmetricAbout (fun x ↦ f (2 * x)) (b / 2)) 
    (h3 : b > a) : 
    IsPeriod f (4 * (b - a)) := by
  sorry

end period_of_symmetric_function_l223_22365


namespace initial_number_proof_l223_22327

theorem initial_number_proof : ∃ x : ℤ, x - 10 * 2 * 5 = 10011 ∧ x = 10111 := by
  sorry

end initial_number_proof_l223_22327


namespace kth_level_associated_point_coordinates_l223_22337

/-- Definition of a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Definition of the k-th level associated point -/
def kth_level_associated_point (A : Point) (k : ℝ) : Point :=
  { x := k * A.x + A.y,
    y := A.x + k * A.y }

/-- Theorem: The k-th level associated point B of A(x,y) has coordinates (kx+y, x+ky) -/
theorem kth_level_associated_point_coordinates (A : Point) (k : ℝ) (h : k ≠ 0) :
  let B := kth_level_associated_point A k
  B.x = k * A.x + A.y ∧ B.y = A.x + k * A.y :=
by sorry

end kth_level_associated_point_coordinates_l223_22337


namespace voting_scenario_theorem_l223_22369

/-- Represents the voting scenario in a certain city -/
structure VotingScenario where
  total_voters : ℝ
  dem_percent : ℝ
  rep_percent : ℝ
  dem_for_A_percent : ℝ
  total_for_A_percent : ℝ
  rep_for_A_percent : ℝ

/-- The theorem statement for the voting scenario problem -/
theorem voting_scenario_theorem (v : VotingScenario) :
  v.dem_percent = 0.6 ∧
  v.rep_percent = 0.4 ∧
  v.dem_for_A_percent = 0.75 ∧
  v.total_for_A_percent = 0.57 →
  v.rep_for_A_percent = 0.3 := by
  sorry


end voting_scenario_theorem_l223_22369


namespace binary_1001_equals_9_l223_22332

/-- Converts a binary number represented as a list of bits (least significant bit first) to its decimal equivalent -/
def binary_to_decimal (bits : List Bool) : ℕ :=
  bits.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

/-- The binary representation of 1001₂ -/
def binary_1001 : List Bool := [true, false, false, true]

theorem binary_1001_equals_9 : binary_to_decimal binary_1001 = 9 := by
  sorry

end binary_1001_equals_9_l223_22332


namespace least_integer_with_nine_factors_l223_22390

/-- A function that returns the number of distinct positive factors of a positive integer -/
def number_of_factors (n : ℕ+) : ℕ :=
  sorry

/-- A function that checks if a number has exactly nine distinct positive factors -/
def has_nine_factors (n : ℕ+) : Prop :=
  number_of_factors n = 9

theorem least_integer_with_nine_factors :
  ∃ (n : ℕ+), has_nine_factors n ∧ ∀ (m : ℕ+), has_nine_factors m → n ≤ m :=
sorry

end least_integer_with_nine_factors_l223_22390


namespace debt_average_payment_l223_22338

/-- Prove that the average payment for a debt with specific payment structure is $442.50 -/
theorem debt_average_payment (n : ℕ) (first_payment second_payment : ℚ) : 
  n = 40 →
  first_payment = 410 →
  second_payment = first_payment + 65 →
  (n / 2 * first_payment + n / 2 * second_payment) / n = 442.5 := by
  sorry

end debt_average_payment_l223_22338


namespace full_bucket_weight_formula_l223_22316

/-- Represents the weight of a bucket with water -/
structure BucketWeight where
  twoThirdsFull : ℝ  -- Weight when 2/3 full
  halfFull : ℝ       -- Weight when 1/2 full

/-- Calculates the weight of a bucket when it's full of water -/
def fullBucketWeight (bw : BucketWeight) : ℝ :=
  3 * bw.twoThirdsFull - 2 * bw.halfFull

/-- Theorem stating that the weight of a full bucket is 3a - 2b given the weights at 2/3 and 1/2 full -/
theorem full_bucket_weight_formula (bw : BucketWeight) :
  fullBucketWeight bw = 3 * bw.twoThirdsFull - 2 * bw.halfFull := by
  sorry

end full_bucket_weight_formula_l223_22316


namespace amount_division_l223_22349

/-- Given an amount divided into 3 parts proportional to 1/2 : 2/3 : 3/4, 
    with the first part being 204, prove the total amount is 782. -/
theorem amount_division (amount : ℕ) 
  (h1 : amount > 0)
  (h2 : ∃ (a b c : ℕ), a > 0 ∧ b > 0 ∧ c > 0 ∧ 
    a * 3 * 4 = 1 * 2 * 4 ∧ 
    b * 2 * 4 = 2 * 3 * 4 ∧ 
    c * 2 * 3 = 3 * 2 * 4 ∧
    a + b + c = amount)
  (h3 : a = 204) : 
  amount = 782 := by
  sorry

end amount_division_l223_22349


namespace unique_solution_system_l223_22313

theorem unique_solution_system (x y z : ℝ) : 
  (x^2 - 23*y - 25*z = -681) ∧
  (y^2 - 21*x - 21*z = -419) ∧
  (z^2 - 19*x - 21*y = -313) ↔
  (x = 20 ∧ y = 22 ∧ z = 23) :=
by sorry

end unique_solution_system_l223_22313


namespace max_value_product_max_value_achieved_l223_22301

theorem max_value_product (x y z w : ℝ) 
  (pos_x : x > 0) (pos_y : y > 0) (pos_z : z > 0) (pos_w : w > 0)
  (sum_one : x + y + z + w = 1) :
  x^2 * y^2 * z^2 * w ≤ 64 / 823543 := by
sorry

theorem max_value_achieved (x y z w : ℝ) 
  (pos_x : x > 0) (pos_y : y > 0) (pos_z : z > 0) (pos_w : w > 0)
  (sum_one : x + y + z + w = 1) :
  ∃ x y z w, x^2 * y^2 * z^2 * w = 64 / 823543 ∧ 
             x + y + z + w = 1 ∧ 
             x > 0 ∧ y > 0 ∧ z > 0 ∧ w > 0 := by
sorry

end max_value_product_max_value_achieved_l223_22301


namespace complex_magnitude_equality_l223_22366

theorem complex_magnitude_equality (m : ℝ) (h : m > 0) :
  Complex.abs (4 + m * Complex.I) = 4 * Real.sqrt 5 → m = 8 := by
sorry

end complex_magnitude_equality_l223_22366


namespace computable_logarithms_l223_22314

def is_computable (n : ℕ) : Prop :=
  ∃ (m n p : ℕ), n = 2^m * 3^n * 5^p ∧ n ≤ 100

def computable_set : Set ℕ :=
  {n : ℕ | n ≥ 1 ∧ n ≤ 100 ∧ is_computable n}

theorem computable_logarithms :
  computable_set = {1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 15, 16, 18, 20, 24, 25, 27, 30, 32, 36, 40, 45, 48, 50, 54, 60, 64, 72, 75, 80, 81, 90, 96, 100} :=
by sorry

end computable_logarithms_l223_22314


namespace all_measurements_correct_l223_22393

-- Define a structure for measurements
structure Measurement where
  value : Float
  unit : String

-- Define the measurements
def ruler_length : Measurement := { value := 2, unit := "decimeters" }
def truck_capacity : Measurement := { value := 5, unit := "tons" }
def bus_speed : Measurement := { value := 100, unit := "kilometers" }
def book_thickness : Measurement := { value := 7, unit := "millimeters" }
def backpack_weight : Measurement := { value := 4000, unit := "grams" }

-- Define propositions for correct units
def correct_ruler_unit (m : Measurement) : Prop := m.unit = "decimeters"
def correct_truck_unit (m : Measurement) : Prop := m.unit = "tons"
def correct_bus_unit (m : Measurement) : Prop := m.unit = "kilometers"
def correct_book_unit (m : Measurement) : Prop := m.unit = "millimeters"
def correct_backpack_unit (m : Measurement) : Prop := m.unit = "grams"

-- Theorem stating that all measurements have correct units
theorem all_measurements_correct : 
  correct_ruler_unit ruler_length ∧
  correct_truck_unit truck_capacity ∧
  correct_bus_unit bus_speed ∧
  correct_book_unit book_thickness ∧
  correct_backpack_unit backpack_weight :=
by sorry


end all_measurements_correct_l223_22393


namespace congruence_solution_sum_l223_22363

theorem congruence_solution_sum (a m : ℕ) : 
  m ≥ 2 → 
  0 ≤ a → 
  a < m → 
  (∀ x : ℤ, (8 * x + 1) % 12 = 5 % 12 ↔ x % m = a % m) → 
  a + m = 5 := by
  sorry

end congruence_solution_sum_l223_22363


namespace like_terms_imply_equation_l223_22361

/-- Two monomials are like terms if their variables and corresponding exponents are the same -/
def are_like_terms (m n : ℕ) : Prop :=
  m = 3 ∧ n = 1

theorem like_terms_imply_equation (m n : ℕ) :
  are_like_terms m n → m - 2*n = 1 := by
  sorry

end like_terms_imply_equation_l223_22361


namespace egg_box_count_l223_22307

theorem egg_box_count (total_eggs : Real) (eggs_per_box : Real) (h1 : total_eggs = 3.0) (h2 : eggs_per_box = 1.5) :
  (total_eggs / eggs_per_box : Real) = 2 := by
  sorry

end egg_box_count_l223_22307


namespace weight_loss_challenge_l223_22340

/-- 
Given an initial weight W, a weight loss percentage, and a clothing weight percentage,
calculates the final measured weight loss percentage.
-/
def measured_weight_loss_percentage (initial_weight_loss : Real) (clothing_weight_percent : Real) : Real :=
  let remaining_weight_percent := 1 - initial_weight_loss
  let final_weight_percent := remaining_weight_percent * (1 + clothing_weight_percent)
  (1 - final_weight_percent) * 100

/-- 
Proves that given an initial weight loss of 15% and clothes that add 2% to the final weight,
the measured weight loss percentage at the final weigh-in is 13.3%.
-/
theorem weight_loss_challenge (ε : Real) :
  ∃ δ > 0, ∀ x, |x - 0.133| < δ → |measured_weight_loss_percentage 0.15 0.02 - x| < ε :=
sorry

end weight_loss_challenge_l223_22340


namespace function_composition_ratio_l223_22356

-- Define the functions f and g
def f (x : ℝ) : ℝ := 2 * x + 3
def g (x : ℝ) : ℝ := 3 * x - 2

-- State the theorem
theorem function_composition_ratio :
  (f (g (f 3))) / (g (f (g 3))) = 53 / 49 := by
  sorry

end function_composition_ratio_l223_22356


namespace service_charge_percentage_l223_22355

theorem service_charge_percentage (salmon_cost black_burger_cost chicken_katsu_cost : ℝ)
  (tip_percentage : ℝ) (total_paid change : ℝ) :
  salmon_cost = 40 →
  black_burger_cost = 15 →
  chicken_katsu_cost = 25 →
  tip_percentage = 0.05 →
  total_paid = 100 →
  change = 8 →
  let food_cost := salmon_cost + black_burger_cost + chicken_katsu_cost
  let tip := food_cost * tip_percentage
  let service_charge := total_paid - change - food_cost - tip
  service_charge / food_cost = 0.1 := by
sorry

end service_charge_percentage_l223_22355


namespace total_cars_produced_l223_22382

/-- The total number of cars produced in North America, Europe, and Asia is 9972. -/
theorem total_cars_produced (north_america europe asia : ℕ) 
  (h1 : north_america = 3884)
  (h2 : europe = 2871)
  (h3 : asia = 3217) : 
  north_america + europe + asia = 9972 := by
  sorry

end total_cars_produced_l223_22382


namespace correct_equation_l223_22312

theorem correct_equation (x y : ℝ) : 3 * x^2 * y - 4 * y * x^2 = -x^2 * y := by
  sorry

end correct_equation_l223_22312


namespace f_properties_l223_22381

noncomputable def f (x : ℝ) : ℝ := 2 * Real.sin (x - Real.pi) * Real.cos (Real.pi - x)

theorem f_properties :
  (∃ (T : ℝ), T > 0 ∧ (∀ (x : ℝ), f (x + T) = f x) ∧
    (∀ (T' : ℝ), T' > 0 → (∀ (x : ℝ), f (x + T') = f x) → T ≤ T') ∧ T = Real.pi) ∧
  (∀ (x : ℝ), x ∈ Set.Icc (-Real.pi/6) (Real.pi/2) → f x ≥ -Real.sqrt 3 / 2) ∧
  (∃ (x : ℝ), x ∈ Set.Icc (-Real.pi/6) (Real.pi/2) ∧ f x = -Real.sqrt 3 / 2) :=
sorry

end f_properties_l223_22381


namespace hexagonal_tile_difference_l223_22305

theorem hexagonal_tile_difference (initial_blue : ℕ) (initial_green : ℕ) (border_tiles : ℕ) : 
  initial_blue = 20 → initial_green = 15 → border_tiles = 18 →
  (initial_green + 2 * border_tiles) - initial_blue = 31 := by
sorry

end hexagonal_tile_difference_l223_22305


namespace wednesday_most_frequent_l223_22302

/-- Represents days of the week -/
inductive DayOfWeek
  | sunday
  | monday
  | tuesday
  | wednesday
  | thursday
  | friday
  | saturday

/-- Represents a date in the year 2014 -/
structure Date2014 where
  month : Nat
  day : Nat

def march_9_2014 : Date2014 := ⟨3, 9⟩

/-- The number of days in 2014 -/
def days_in_2014 : Nat := 365

/-- Function to determine the day of the week for a given date in 2014 -/
def dayOfWeek (d : Date2014) : DayOfWeek := sorry

/-- Function to count occurrences of each day of the week in 2014 -/
def countDayOccurrences (day : DayOfWeek) : Nat := sorry

/-- Theorem stating that Wednesday occurs most frequently in 2014 -/
theorem wednesday_most_frequent :
  (dayOfWeek march_9_2014 = DayOfWeek.sunday) →
  (∀ d : DayOfWeek, countDayOccurrences DayOfWeek.wednesday ≥ countDayOccurrences d) :=
by sorry

end wednesday_most_frequent_l223_22302


namespace sum_of_fifth_powers_l223_22370

theorem sum_of_fifth_powers (a b c : ℝ) 
  (sum_condition : a + b + c = 1)
  (sum_squares_condition : a^2 + b^2 + c^2 = 3)
  (sum_cubes_condition : a^3 + b^3 + c^3 = 4) :
  a^5 + b^5 + c^5 = 11/3 := by
sorry

end sum_of_fifth_powers_l223_22370
