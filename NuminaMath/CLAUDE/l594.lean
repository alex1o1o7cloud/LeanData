import Mathlib

namespace NUMINAMATH_CALUDE_delta_sports_club_ratio_l594_59424

/-- Proves that the ratio of female to male members is 2/3 given the average ages --/
theorem delta_sports_club_ratio (f m : ℕ) (hf : f > 0) (hm : m > 0) : 
  (35 : ℝ) * f + 30 * m = 32 * (f + m) → f / m = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_delta_sports_club_ratio_l594_59424


namespace NUMINAMATH_CALUDE_gcf_of_75_and_90_l594_59488

theorem gcf_of_75_and_90 : Nat.gcd 75 90 = 15 := by
  sorry

end NUMINAMATH_CALUDE_gcf_of_75_and_90_l594_59488


namespace NUMINAMATH_CALUDE_no_solutions_for_inequality_system_l594_59410

theorem no_solutions_for_inequality_system :
  ¬ ∃ (x y : ℝ), (11 * x^2 - 10 * x * y + 3 * y^2 ≤ 3) ∧ (5 * x + y ≤ -10) := by
  sorry

end NUMINAMATH_CALUDE_no_solutions_for_inequality_system_l594_59410


namespace NUMINAMATH_CALUDE_odd_count_after_ten_operations_l594_59443

/-- Represents the state of the board after n operations -/
structure BoardState (n : ℕ) where
  odd_count : ℕ  -- Number of odd numbers on the board
  total_count : ℕ  -- Total number of numbers on the board

/-- Performs one operation on the board -/
def next_state (state : BoardState n) : BoardState (n + 1) :=
  sorry

/-- Initial state of the board with 0 and 1 -/
def initial_state : BoardState 0 :=
  { odd_count := 1, total_count := 2 }

/-- The state of the board after n operations -/
def board_state (n : ℕ) : BoardState n :=
  match n with
  | 0 => initial_state
  | n + 1 => next_state (board_state n)

theorem odd_count_after_ten_operations :
  (board_state 10).odd_count = 683 :=
sorry

end NUMINAMATH_CALUDE_odd_count_after_ten_operations_l594_59443


namespace NUMINAMATH_CALUDE_danjiangkou_tourists_scientific_notation_l594_59400

/-- Converts a positive integer to scientific notation -/
def to_scientific_notation (n : ℕ) : ℚ × ℤ :=
  sorry

theorem danjiangkou_tourists_scientific_notation :
  to_scientific_notation 456000 = (4.56, 5) :=
sorry

end NUMINAMATH_CALUDE_danjiangkou_tourists_scientific_notation_l594_59400


namespace NUMINAMATH_CALUDE_babylonian_square_58_l594_59426

/-- Represents the Babylonian method of expressing squares --/
def babylonian_square (n : ℕ) : ℕ × ℕ :=
  let square := n * n
  let quotient := square / 60
  let remainder := square % 60
  if remainder = 0 then (quotient, 60) else (quotient, remainder)

/-- The theorem to be proved --/
theorem babylonian_square_58 : babylonian_square 58 = (56, 4) := by
  sorry

#eval babylonian_square 58  -- To check the result

end NUMINAMATH_CALUDE_babylonian_square_58_l594_59426


namespace NUMINAMATH_CALUDE_platform_length_l594_59466

/-- Given a train of length 300 meters that crosses a platform in 27 seconds
    and a signal pole in 18 seconds, the length of the platform is 150 meters. -/
theorem platform_length (train_length : ℝ) (platform_time : ℝ) (pole_time : ℝ) :
  train_length = 300 →
  platform_time = 27 →
  pole_time = 18 →
  (train_length * platform_time / pole_time) - train_length = 150 := by
sorry

end NUMINAMATH_CALUDE_platform_length_l594_59466


namespace NUMINAMATH_CALUDE_quadratic_two_real_roots_l594_59422

theorem quadratic_two_real_roots (k : ℝ) :
  (∃ x y : ℝ, x ≠ y ∧ k * x^2 - 6 * x - 1 = 0 ∧ k * y^2 - 6 * y - 1 = 0) ↔
  (k ≥ -9 ∧ k ≠ 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_two_real_roots_l594_59422


namespace NUMINAMATH_CALUDE_simplify_sqrt_expression_l594_59479

theorem simplify_sqrt_expression : 
  Real.sqrt 7 - Real.sqrt 28 + Real.sqrt 63 = 2 * Real.sqrt 7 := by
  sorry

end NUMINAMATH_CALUDE_simplify_sqrt_expression_l594_59479


namespace NUMINAMATH_CALUDE_geometric_sequence_problem_l594_59486

-- Define the geometric sequence
def geometric_sequence (a₁ : ℝ) (r : ℝ) (n : ℕ) : ℝ :=
  a₁ * r^(n - 1)

-- State the theorem
theorem geometric_sequence_problem (a₁ a₄ : ℝ) (m : ℤ) :
  a₁ = 2 →
  a₄ = 1/4 →
  m = -15 →
  (∃ r : ℝ, ∀ n : ℕ, geometric_sequence a₁ r n = 2^(2 - n)) →
  m = 14 := by
  sorry


end NUMINAMATH_CALUDE_geometric_sequence_problem_l594_59486


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l594_59429

theorem quadratic_inequality_solution_set 
  (a b : ℝ) 
  (h1 : ∀ x : ℝ, ax^2 + b*x + 2 > 0 ↔ -1 < x ∧ x < 2) :
  ∀ x : ℝ, 2*x^2 + b*x + a < 0 ↔ -1 < x ∧ x < 1/2 :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l594_59429


namespace NUMINAMATH_CALUDE_cone_surface_area_l594_59437

/-- The surface area of a cone with given height and base area -/
theorem cone_surface_area (h : ℝ) (base_area : ℝ) (h_pos : h > 0) (base_pos : base_area > 0) :
  let r := Real.sqrt (base_area / Real.pi)
  let l := Real.sqrt (r^2 + h^2)
  h = 4 ∧ base_area = 9 * Real.pi → Real.pi * r * l + base_area = 24 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_cone_surface_area_l594_59437


namespace NUMINAMATH_CALUDE_greatest_sum_consecutive_integers_l594_59482

theorem greatest_sum_consecutive_integers (n : ℕ) : 
  (n * (n + 1) < 500) → (∀ m : ℕ, m > n → m * (m + 1) ≥ 500) → n + (n + 1) = 43 :=
by sorry

end NUMINAMATH_CALUDE_greatest_sum_consecutive_integers_l594_59482


namespace NUMINAMATH_CALUDE_unique_number_satisfies_equation_l594_59476

theorem unique_number_satisfies_equation : ∃! x : ℝ, (60 + 12) / 3 = (x - 12) * 3 := by
  sorry

end NUMINAMATH_CALUDE_unique_number_satisfies_equation_l594_59476


namespace NUMINAMATH_CALUDE_nisos_population_meets_capacity_l594_59481

/-- Represents the state of Nisos island at a given time -/
structure NisosState where
  year : ℕ
  population : ℕ

/-- Calculates the population after a given number of 20-year periods -/
def population_after (initial_population : ℕ) (periods : ℕ) : ℕ :=
  initial_population * (4 ^ periods)

/-- Theorem: Nisos island population meets capacity limit after 60 years -/
theorem nisos_population_meets_capacity : 
  ∀ (initial_state : NisosState),
    initial_state.year = 1998 →
    initial_state.population = 100 →
    ∃ (final_state : NisosState),
      final_state.year = initial_state.year + 60 ∧
      final_state.population ≥ 7500 ∧
      final_state.population < population_after 100 4 :=
sorry

/-- The land area of Nisos island in hectares -/
def nisos_area : ℕ := 15000

/-- The land area required per person in hectares -/
def land_per_person : ℕ := 2

/-- The capacity of Nisos island -/
def nisos_capacity : ℕ := nisos_area / land_per_person

/-- The population growth factor per 20-year period -/
def growth_factor : ℕ := 4

/-- The number of 20-year periods in 60 years -/
def periods_in_60_years : ℕ := 3

end NUMINAMATH_CALUDE_nisos_population_meets_capacity_l594_59481


namespace NUMINAMATH_CALUDE_cos_30_degrees_l594_59475

theorem cos_30_degrees : Real.cos (π / 6) = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_30_degrees_l594_59475


namespace NUMINAMATH_CALUDE_complement_intersection_theorem_l594_59412

open Set

def I : Finset Nat := {1,2,3,4,5}
def A : Finset Nat := {2,3,5}
def B : Finset Nat := {1,2}

theorem complement_intersection_theorem :
  (I \ B) ∩ A = {3,5} := by sorry

end NUMINAMATH_CALUDE_complement_intersection_theorem_l594_59412


namespace NUMINAMATH_CALUDE_sum_of_solutions_l594_59496

theorem sum_of_solutions (x y : ℝ) 
  (hx : x^3 - 6*x^2 + 12*x = 13) 
  (hy : y^3 + 3*y - 3*y^2 = -4) : 
  x + y = 3 := by
sorry

end NUMINAMATH_CALUDE_sum_of_solutions_l594_59496


namespace NUMINAMATH_CALUDE_parkers_richies_ratio_l594_59499

/-- Given that Parker's share is $50 and the total shared amount is $125,
    prove that the ratio of Parker's share to Richie's share is 2:3. -/
theorem parkers_richies_ratio (parker_share : ℝ) (total_share : ℝ) :
  parker_share = 50 →
  total_share = 125 →
  parker_share < total_share →
  ∃ (a b : ℕ), a = 2 ∧ b = 3 ∧ parker_share / (total_share - parker_share) = a / b :=
by sorry

end NUMINAMATH_CALUDE_parkers_richies_ratio_l594_59499


namespace NUMINAMATH_CALUDE_remainder_1999_11_mod_8_l594_59432

theorem remainder_1999_11_mod_8 : 1999^11 % 8 = 7 := by
  sorry

end NUMINAMATH_CALUDE_remainder_1999_11_mod_8_l594_59432


namespace NUMINAMATH_CALUDE_parabolas_intersection_l594_59474

-- Define the parabolas
def parabola1 (a x y : ℝ) : Prop := y = x^2 + x + a
def parabola2 (a x y : ℝ) : Prop := x = 4*y^2 + 3*y + a

-- Define the condition of four intersection points
def has_four_intersections (a : ℝ) : Prop := ∃ x1 y1 x2 y2 x3 y3 x4 y4 : ℝ,
  (parabola1 a x1 y1 ∧ parabola2 a x1 y1) ∧
  (parabola1 a x2 y2 ∧ parabola2 a x2 y2) ∧
  (parabola1 a x3 y3 ∧ parabola2 a x3 y3) ∧
  (parabola1 a x4 y4 ∧ parabola2 a x4 y4) ∧
  (x1 ≠ x2 ∨ y1 ≠ y2) ∧ (x1 ≠ x3 ∨ y1 ≠ y3) ∧ (x1 ≠ x4 ∨ y1 ≠ y4) ∧
  (x2 ≠ x3 ∨ y2 ≠ y3) ∧ (x2 ≠ x4 ∨ y2 ≠ y4) ∧ (x3 ≠ x4 ∨ y3 ≠ y4)

-- Define the range of a
def a_range (a : ℝ) : Prop := (a < -1/2 ∨ (-1/2 < a ∧ a < -7/16))

-- Define the condition for points being concyclic
def concyclic (x1 y1 x2 y2 x3 y3 x4 y4 : ℝ) : Prop :=
  ∃ cx cy r : ℝ, 
    (x1 - cx)^2 + (y1 - cy)^2 = r^2 ∧
    (x2 - cx)^2 + (y2 - cy)^2 = r^2 ∧
    (x3 - cx)^2 + (y3 - cy)^2 = r^2 ∧
    (x4 - cx)^2 + (y4 - cy)^2 = r^2

-- The main theorem
theorem parabolas_intersection (a : ℝ) :
  has_four_intersections a →
  (a_range a ∧
   ∃ x1 y1 x2 y2 x3 y3 x4 y4 : ℝ,
     (parabola1 a x1 y1 ∧ parabola2 a x1 y1) ∧
     (parabola1 a x2 y2 ∧ parabola2 a x2 y2) ∧
     (parabola1 a x3 y3 ∧ parabola2 a x3 y3) ∧
     (parabola1 a x4 y4 ∧ parabola2 a x4 y4) ∧
     concyclic x1 y1 x2 y2 x3 y3 x4 y4 ∧
     ∃ cx cy : ℝ, cx = -3/8 ∧ cy = 1/8) :=
by sorry

end NUMINAMATH_CALUDE_parabolas_intersection_l594_59474


namespace NUMINAMATH_CALUDE_prob_both_white_one_third_l594_59454

/-- Represents a bag of balls -/
structure Bag where
  white : Nat
  yellow : Nat

/-- Calculates the probability of drawing a white ball from a bag -/
def probWhite (bag : Bag) : Rat :=
  bag.white / (bag.white + bag.yellow)

/-- The probability of drawing white balls from both bags -/
def probBothWhite (bagA bagB : Bag) : Rat :=
  probWhite bagA * probWhite bagB

theorem prob_both_white_one_third :
  let bagA : Bag := { white := 1, yellow := 1 }
  let bagB : Bag := { white := 2, yellow := 1 }
  probBothWhite bagA bagB = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_prob_both_white_one_third_l594_59454


namespace NUMINAMATH_CALUDE_plane_equation_l594_59451

/-- A plane in 3D space defined by a normal vector and a point it passes through. -/
structure Plane where
  normal : ℝ × ℝ × ℝ
  point : ℝ × ℝ × ℝ

/-- Checks if a given point lies on the plane. -/
def Plane.contains (π : Plane) (p : ℝ × ℝ × ℝ) : Prop :=
  let (nx, ny, nz) := π.normal
  let (ax, ay, az) := π.point
  let (x, y, z) := p
  nx * (x - ax) + ny * (y - ay) + nz * (z - az) = 0

/-- The main theorem stating the equation of the plane. -/
theorem plane_equation (π : Plane) (h : π.normal = (1, -1, 2) ∧ π.point = (0, 3, 1)) :
  ∀ p : ℝ × ℝ × ℝ, π.contains p ↔ p.1 - p.2.1 + 2 * p.2.2 + 1 = 0 := by
  sorry

end NUMINAMATH_CALUDE_plane_equation_l594_59451


namespace NUMINAMATH_CALUDE_max_clock_digit_sum_l594_59427

def is_valid_hour (h : ℕ) : Prop := h ≥ 0 ∧ h ≤ 23

def is_valid_minute (m : ℕ) : Prop := m ≥ 0 ∧ m ≤ 59

def digit_sum (n : ℕ) : ℕ := (n / 10) + (n % 10)

def clock_digit_sum (h m : ℕ) : ℕ := digit_sum h + digit_sum m

theorem max_clock_digit_sum :
  ∃ (h m : ℕ), is_valid_hour h ∧ is_valid_minute m ∧
  ∀ (h' m' : ℕ), is_valid_hour h' → is_valid_minute m' →
  clock_digit_sum h m ≥ clock_digit_sum h' m' ∧
  clock_digit_sum h m = 24 :=
sorry

end NUMINAMATH_CALUDE_max_clock_digit_sum_l594_59427


namespace NUMINAMATH_CALUDE_strip_length_is_14_l594_59434

/-- Represents a folded rectangular strip of paper -/
structure FoldedStrip :=
  (width : ℝ)
  (ap_length : ℝ)
  (bm_length : ℝ)

/-- Calculates the total length of the folded strip -/
def total_length (strip : FoldedStrip) : ℝ :=
  strip.ap_length + strip.width + strip.bm_length

/-- Theorem: The length of the rectangular strip is 14 cm -/
theorem strip_length_is_14 (strip : FoldedStrip) 
  (h_width : strip.width = 4)
  (h_ap : strip.ap_length = 5)
  (h_bm : strip.bm_length = 5) : 
  total_length strip = 14 :=
by
  sorry

#eval total_length { width := 4, ap_length := 5, bm_length := 5 }

end NUMINAMATH_CALUDE_strip_length_is_14_l594_59434


namespace NUMINAMATH_CALUDE_cos_seven_pi_sixths_l594_59485

theorem cos_seven_pi_sixths : Real.cos (7 * π / 6) = -Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_seven_pi_sixths_l594_59485


namespace NUMINAMATH_CALUDE_work_together_duration_l594_59456

/-- Given two workers A and B, where A can complete a job in 15 days and B in 20 days,
    this theorem proves that if they work together until 5/12 of the job is left,
    then they worked together for 5 days. -/
theorem work_together_duration (a_rate b_rate : ℚ) (work_left : ℚ) (days_worked : ℕ) :
  a_rate = 1 / 15 →
  b_rate = 1 / 20 →
  work_left = 5 / 12 →
  (a_rate + b_rate) * days_worked = 1 - work_left →
  days_worked = 5 :=
by sorry

end NUMINAMATH_CALUDE_work_together_duration_l594_59456


namespace NUMINAMATH_CALUDE_line_intersects_circle_l594_59448

theorem line_intersects_circle (a : ℝ) (h : a ≥ 0) :
  ∃ (x y : ℝ), (a * x - y + Real.sqrt 2 * a = 0) ∧ (x^2 + y^2 = 9) := by
  sorry

end NUMINAMATH_CALUDE_line_intersects_circle_l594_59448


namespace NUMINAMATH_CALUDE_initial_quarters_count_l594_59402

-- Define the problem parameters
def cents_left : ℕ := 300
def cents_spent : ℕ := 50
def cents_per_quarter : ℕ := 25

-- Theorem statement
theorem initial_quarters_count : 
  (cents_left + cents_spent) / cents_per_quarter = 14 := by
  sorry

end NUMINAMATH_CALUDE_initial_quarters_count_l594_59402


namespace NUMINAMATH_CALUDE_judes_chair_expenditure_l594_59472

/-- Proves that the amount spent on chairs is $36 given the conditions of Jude's purchase --/
theorem judes_chair_expenditure
  (table_cost : ℕ)
  (plate_set_cost : ℕ)
  (num_plate_sets : ℕ)
  (money_given : ℕ)
  (change_received : ℕ)
  (h1 : table_cost = 50)
  (h2 : plate_set_cost = 20)
  (h3 : num_plate_sets = 2)
  (h4 : money_given = 130)
  (h5 : change_received = 4) :
  money_given - change_received - (table_cost + num_plate_sets * plate_set_cost) = 36 := by
  sorry

#check judes_chair_expenditure

end NUMINAMATH_CALUDE_judes_chair_expenditure_l594_59472


namespace NUMINAMATH_CALUDE_min_value_inequality_l594_59444

theorem min_value_inequality (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (1 + b / a) * (1 + 4 * a / b) ≥ 9 := by
  sorry

end NUMINAMATH_CALUDE_min_value_inequality_l594_59444


namespace NUMINAMATH_CALUDE_percentage_without_muffin_l594_59428

theorem percentage_without_muffin (muffin yogurt fruit granola : ℝ) :
  muffin = 38 →
  yogurt = 10 →
  fruit = 27 →
  granola = 25 →
  muffin + yogurt + fruit + granola = 100 →
  100 - muffin = 62 :=
by sorry

end NUMINAMATH_CALUDE_percentage_without_muffin_l594_59428


namespace NUMINAMATH_CALUDE_parallel_vectors_k_value_l594_59408

/-- Given two vectors a and b in ℝ³, if k * a + b is parallel to 2 * a - b, then k = -2 -/
theorem parallel_vectors_k_value (a b : ℝ × ℝ × ℝ) (k : ℝ) 
    (h1 : a = (1, 1, 0)) 
    (h2 : b = (-1, 0, -2)) 
    (h_parallel : ∃ (t : ℝ), t • (k • a + b) = 2 • a - b) : 
  k = -2 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_k_value_l594_59408


namespace NUMINAMATH_CALUDE_stock_income_calculation_l594_59470

/-- Calculates the income derived from a stock investment --/
theorem stock_income_calculation
  (interest_rate : ℝ)
  (investment_amount : ℝ)
  (brokerage_rate : ℝ)
  (market_value_per_100 : ℝ)
  (h1 : interest_rate = 0.105)
  (h2 : investment_amount = 6000)
  (h3 : brokerage_rate = 0.0025)
  (h4 : market_value_per_100 = 83.08333333333334) :
  let brokerage_fee := investment_amount * brokerage_rate
  let actual_investment := investment_amount - brokerage_fee
  let num_units := actual_investment / market_value_per_100
  let face_value := num_units * 100
  let income := face_value * interest_rate
  income = 756 := by sorry

end NUMINAMATH_CALUDE_stock_income_calculation_l594_59470


namespace NUMINAMATH_CALUDE_nilpotent_matrix_square_zero_l594_59433

theorem nilpotent_matrix_square_zero 
  (B : Matrix (Fin 2) (Fin 2) ℝ) 
  (h : B ^ 4 = 0) : 
  B ^ 2 = 0 := by
sorry

end NUMINAMATH_CALUDE_nilpotent_matrix_square_zero_l594_59433


namespace NUMINAMATH_CALUDE_f_max_min_on_interval_l594_59423

open Real

-- Define the function f
def f (x : ℝ) : ℝ := x^3 - x^2 - x + 1

-- State the theorem
theorem f_max_min_on_interval :
  ∃ (max min : ℝ), 
    (∀ x ∈ Set.Icc (-1 : ℝ) 3, f x ≤ max) ∧
    (∃ x ∈ Set.Icc (-1 : ℝ) 3, f x = max) ∧
    (∀ x ∈ Set.Icc (-1 : ℝ) 3, min ≤ f x) ∧
    (∃ x ∈ Set.Icc (-1 : ℝ) 3, f x = min) ∧
    max = 16 ∧ min = 0 := by
  sorry

end NUMINAMATH_CALUDE_f_max_min_on_interval_l594_59423


namespace NUMINAMATH_CALUDE_quadratic_factorization_l594_59407

theorem quadratic_factorization (x : ℝ) : 
  x^2 - 6*x - 6 = 0 ↔ (x - 3)^2 = 15 := by sorry

end NUMINAMATH_CALUDE_quadratic_factorization_l594_59407


namespace NUMINAMATH_CALUDE_winter_clothing_count_l594_59409

theorem winter_clothing_count (num_boxes : ℕ) (scarves_per_box : ℕ) (mittens_per_box : ℕ) : 
  num_boxes = 4 → scarves_per_box = 2 → mittens_per_box = 6 →
  num_boxes * scarves_per_box + num_boxes * mittens_per_box = 32 := by
  sorry

end NUMINAMATH_CALUDE_winter_clothing_count_l594_59409


namespace NUMINAMATH_CALUDE_inscribed_cylinder_radius_l594_59484

/-- Represents a right circular cylinder inscribed in a right circular cone -/
structure InscribedCylinder where
  cone_diameter : ℝ
  cone_altitude : ℝ
  cylinder_radius : ℝ

/-- The condition that the cylinder's diameter equals its height -/
def cylinder_diameter_equals_height (c : InscribedCylinder) : Prop :=
  2 * c.cylinder_radius = 2 * c.cylinder_radius

/-- The condition that the cone's diameter is 15 -/
def cone_diameter_is_15 (c : InscribedCylinder) : Prop :=
  c.cone_diameter = 15

/-- The condition that the cone's altitude is 15 -/
def cone_altitude_is_15 (c : InscribedCylinder) : Prop :=
  c.cone_altitude = 15

/-- The main theorem: the radius of the inscribed cylinder is 15/4 -/
theorem inscribed_cylinder_radius (c : InscribedCylinder) 
  (h1 : cylinder_diameter_equals_height c)
  (h2 : cone_diameter_is_15 c)
  (h3 : cone_altitude_is_15 c) :
  c.cylinder_radius = 15 / 4 := by
  sorry


end NUMINAMATH_CALUDE_inscribed_cylinder_radius_l594_59484


namespace NUMINAMATH_CALUDE_shortest_path_length_is_28b_l594_59417

/-- Represents a 3x3 grid of blocks with side length b -/
structure Grid :=
  (b : ℝ)
  (size : ℕ := 3)

/-- The number of street segments in the grid -/
def Grid.streetSegments (g : Grid) : ℕ := 24

/-- The number of intersections with odd degree -/
def Grid.oddDegreeIntersections (g : Grid) : ℕ := 8

/-- The extra segments that need to be traversed twice -/
def Grid.extraSegments (g : Grid) : ℕ := g.oddDegreeIntersections / 2

/-- The shortest path length to pave all streets in the grid -/
def Grid.shortestPathLength (g : Grid) : ℝ :=
  (g.streetSegments + g.extraSegments) * g.b

/-- Theorem stating that the shortest path length is 28b -/
theorem shortest_path_length_is_28b (g : Grid) :
  g.shortestPathLength = 28 * g.b := by
  sorry

end NUMINAMATH_CALUDE_shortest_path_length_is_28b_l594_59417


namespace NUMINAMATH_CALUDE_positive_integer_N_equals_121_l594_59463

theorem positive_integer_N_equals_121 :
  ∃ (N : ℕ), N > 0 ∧ 33^2 * 55^2 = 15^2 * N^2 ∧ N = 121 := by
  sorry

end NUMINAMATH_CALUDE_positive_integer_N_equals_121_l594_59463


namespace NUMINAMATH_CALUDE_picture_frame_width_l594_59491

theorem picture_frame_width 
  (height : ℝ) 
  (circumference : ℝ) 
  (h_height : height = 12) 
  (h_circumference : circumference = 38) : 
  let width := (circumference - 2 * height) / 2
  width = 7 := by
sorry

end NUMINAMATH_CALUDE_picture_frame_width_l594_59491


namespace NUMINAMATH_CALUDE_least_perimeter_l594_59419

/-- Represents a triangle with two known sides and an integral third side -/
structure Triangle where
  side1 : ℕ
  side2 : ℕ
  side3 : ℕ
  is_triangle : side1 + side2 > side3 ∧ side1 + side3 > side2 ∧ side2 + side3 > side1

/-- The perimeter of a triangle -/
def perimeter (t : Triangle) : ℕ := t.side1 + t.side2 + t.side3

/-- The specific triangle from the problem -/
def problem_triangle : Triangle → Prop
  | t => t.side1 = 24 ∧ t.side2 = 51

theorem least_perimeter :
  ∀ t : Triangle, problem_triangle t →
  ∀ u : Triangle, problem_triangle u →
  perimeter t ≥ 103 ∧ (∃ v : Triangle, problem_triangle v ∧ perimeter v = 103) :=
by sorry

end NUMINAMATH_CALUDE_least_perimeter_l594_59419


namespace NUMINAMATH_CALUDE_triangle_side_length_l594_59461

theorem triangle_side_length (A B C : ℝ) (a b c : ℝ) : 
  A = π/3 → B = π/4 → b = Real.sqrt 6 - Real.sqrt 2 → 
  A + B + C = π → 
  a / Real.sin A = b / Real.sin B → 
  b / Real.sin B = c / Real.sin C →
  c = Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_triangle_side_length_l594_59461


namespace NUMINAMATH_CALUDE_homeroom_teacher_selection_count_l594_59468

/-- The number of ways to arrange k elements from n distinct elements -/
def arrangementCount (n k : ℕ) : ℕ := sorry

/-- The number of valid selection schemes for homeroom teachers -/
def validSelectionCount (maleTotalCount femaleTotalCount selectCount : ℕ) : ℕ :=
  arrangementCount (maleTotalCount + femaleTotalCount) selectCount -
  (arrangementCount maleTotalCount selectCount + arrangementCount femaleTotalCount selectCount)

theorem homeroom_teacher_selection_count :
  validSelectionCount 5 4 3 = 420 := by sorry

end NUMINAMATH_CALUDE_homeroom_teacher_selection_count_l594_59468


namespace NUMINAMATH_CALUDE_line_slope_angle_l594_59498

theorem line_slope_angle (x y : ℝ) : 
  y - Real.sqrt 3 * x + 5 = 0 → 
  ∃ α : ℝ, 0 ≤ α ∧ α < π ∧ Real.tan α = Real.sqrt 3 ∧ α = π / 3 :=
by sorry

end NUMINAMATH_CALUDE_line_slope_angle_l594_59498


namespace NUMINAMATH_CALUDE_number_relationship_l594_59441

theorem number_relationship (s l : ℕ) : 
  s + l = 124 → s = 31 → l = s + 62 := by
  sorry

end NUMINAMATH_CALUDE_number_relationship_l594_59441


namespace NUMINAMATH_CALUDE_inequality_proof_l594_59401

theorem inequality_proof (x y z : ℝ) 
  (pos_x : x > 0) (pos_y : y > 0) (pos_z : z > 0) 
  (sum_one : x + y + z = 1) : 
  (x / (y^2 + z)) + (y / (z^2 + x)) + (z / (x^2 + y)) ≥ 9/4 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l594_59401


namespace NUMINAMATH_CALUDE_function_monotonicity_l594_59462

theorem function_monotonicity (f : ℝ → ℝ) (hf : Differentiable ℝ f) 
  (h : ∀ x, (x^2 - 3*x + 2) * (deriv (deriv f) x) < 0) :
  ∀ x ∈ Set.Icc 1 2, f 1 ≤ f x ∧ f x ≤ f 2 := by sorry

end NUMINAMATH_CALUDE_function_monotonicity_l594_59462


namespace NUMINAMATH_CALUDE_cooler_capacity_increase_l594_59439

/-- Given three coolers with specific capacity relationships, prove the percentage increase from the first to the second cooler --/
theorem cooler_capacity_increase (a b c : ℝ) : 
  a = 100 → 
  b > a → 
  c = b / 2 → 
  a + b + c = 325 → 
  (b - a) / a * 100 = 50 := by
  sorry

end NUMINAMATH_CALUDE_cooler_capacity_increase_l594_59439


namespace NUMINAMATH_CALUDE_prime_squared_minus_one_divisibility_l594_59465

theorem prime_squared_minus_one_divisibility (p : ℕ) (h_prime : Nat.Prime p) (h_ge_7 : p ≥ 7) :
  (∃ q : ℕ, Nat.Prime q ∧ q ≥ 7 ∧ 40 ∣ (q^2 - 1)) ∧
  (∃ r : ℕ, Nat.Prime r ∧ r ≥ 7 ∧ ¬(40 ∣ (r^2 - 1))) :=
sorry

end NUMINAMATH_CALUDE_prime_squared_minus_one_divisibility_l594_59465


namespace NUMINAMATH_CALUDE_negation_equivalence_l594_59445

theorem negation_equivalence (a : ℝ) : 
  (¬ ∃ x : ℝ, x^2 + 2*x + a ≤ 0) ↔ (∀ x : ℝ, x^2 + 2*x + a > 0) := by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l594_59445


namespace NUMINAMATH_CALUDE_saturday_sales_77_l594_59452

/-- Represents the number of boxes sold on each day --/
structure DailySales where
  saturday : ℕ
  sunday : ℕ
  monday : ℕ
  tuesday : ℕ
  wednesday : ℕ

/-- Calculates the total sales over 5 days --/
def totalSales (sales : DailySales) : ℕ :=
  sales.saturday + sales.sunday + sales.monday + sales.tuesday + sales.wednesday

/-- Checks if the sales follow the given percentage increases --/
def followsPercentageIncreases (sales : DailySales) : Prop :=
  sales.sunday = (sales.saturday * 3) / 2 ∧
  sales.monday = (sales.sunday * 13) / 10 ∧
  sales.tuesday = (sales.monday * 6) / 5 ∧
  sales.wednesday = (sales.tuesday * 11) / 10

theorem saturday_sales_77 (sales : DailySales) :
  followsPercentageIncreases sales →
  totalSales sales = 720 →
  sales.saturday = 77 := by
  sorry


end NUMINAMATH_CALUDE_saturday_sales_77_l594_59452


namespace NUMINAMATH_CALUDE_range_of_f_l594_59416

-- Define the function f
def f (x : ℝ) : ℝ := x + |x - 2|

-- State the theorem about the range of f
theorem range_of_f :
  {y : ℝ | ∃ x : ℝ, f x = y} = Set.Ici 2 := by sorry

end NUMINAMATH_CALUDE_range_of_f_l594_59416


namespace NUMINAMATH_CALUDE_sqrt_x_plus_3_real_l594_59413

theorem sqrt_x_plus_3_real (x : ℝ) : (∃ y : ℝ, y^2 = x + 3) ↔ x ≥ -3 := by sorry

end NUMINAMATH_CALUDE_sqrt_x_plus_3_real_l594_59413


namespace NUMINAMATH_CALUDE_world_cup_matches_l594_59483

/-- The number of matches played in a group of teams where each pair plays twice -/
def number_of_matches (n : ℕ) : ℕ :=
  n * (n - 1)

/-- Theorem: In a group of 6 teams where each pair plays twice, 30 matches are played -/
theorem world_cup_matches : number_of_matches 6 = 30 := by
  sorry

end NUMINAMATH_CALUDE_world_cup_matches_l594_59483


namespace NUMINAMATH_CALUDE_pure_imaginary_fraction_l594_59449

theorem pure_imaginary_fraction (a : ℝ) : 
  (∃ b : ℝ, (a + Complex.I) / (1 - Complex.I) = Complex.I * b) → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_pure_imaginary_fraction_l594_59449


namespace NUMINAMATH_CALUDE_tyler_puppies_l594_59467

/-- The number of puppies Tyler has after a week -/
def total_puppies (total_dogs : ℕ) 
                  (dogs_with_5_5 : ℕ) 
                  (dogs_with_8 : ℕ) 
                  (puppies_per_dog_1 : ℚ) 
                  (puppies_per_dog_2 : ℕ) 
                  (puppies_per_dog_3 : ℕ) 
                  (dogs_with_extra : ℕ) 
                  (extra_puppies : ℚ) : ℚ := 
  let remaining_dogs := total_dogs - dogs_with_5_5 - dogs_with_8
  dogs_with_5_5 * puppies_per_dog_1 + 
  dogs_with_8 * puppies_per_dog_2 + 
  remaining_dogs * puppies_per_dog_3 + 
  dogs_with_extra * extra_puppies

theorem tyler_puppies : 
  total_puppies 35 15 10 (5.5) 8 6 5 (2.5) = 235 := by
  sorry

end NUMINAMATH_CALUDE_tyler_puppies_l594_59467


namespace NUMINAMATH_CALUDE_age_change_proof_l594_59460

theorem age_change_proof (n : ℕ) (A : ℝ) : 
  ((n + 1) * (A + 7) = n * A + 39) →
  ((n + 1) * (A - 1) = n * A + 15) →
  n = 2 := by
  sorry

end NUMINAMATH_CALUDE_age_change_proof_l594_59460


namespace NUMINAMATH_CALUDE_root_product_value_l594_59480

theorem root_product_value (p q : ℝ) : 
  3 * p ^ 2 + 9 * p - 21 = 0 →
  3 * q ^ 2 + 9 * q - 21 = 0 →
  (3 * p - 4) * (6 * q - 8) = -22 := by
sorry

end NUMINAMATH_CALUDE_root_product_value_l594_59480


namespace NUMINAMATH_CALUDE_range_of_m_l594_59458

theorem range_of_m (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 4 * x + y = x * y) :
  (∃ m : ℝ, x + y / 4 < m^2 + 3 * m) ↔ ∃ m : ℝ, m < -4 ∨ m > 1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_m_l594_59458


namespace NUMINAMATH_CALUDE_sum_of_digits_inequality_l594_59477

/-- Sum of digits function -/
def sum_of_digits (n : ℕ+) : ℕ :=
  sorry

/-- Theorem: For any positive integer n, s(n) ≤ 8 * s(8n) -/
theorem sum_of_digits_inequality (n : ℕ+) : sum_of_digits n ≤ 8 * sum_of_digits (8 * n) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_digits_inequality_l594_59477


namespace NUMINAMATH_CALUDE_proposition_is_true_l594_59440

theorem proposition_is_true : ∀ x : ℝ, x > 2 → Real.log (x - 1) + x^2 + 4 > 4*x := by
  sorry

end NUMINAMATH_CALUDE_proposition_is_true_l594_59440


namespace NUMINAMATH_CALUDE_total_toy_count_l594_59497

def toy_count (jerry gabriel jaxon sarah emily : ℕ) : Prop :=
  jerry = gabriel + 8 ∧
  gabriel = 2 * jaxon ∧
  jaxon = 15 ∧
  sarah = jerry - 5 ∧
  sarah = emily + 3 ∧
  emily = 2 * gabriel

theorem total_toy_count :
  ∀ jerry gabriel jaxon sarah emily : ℕ,
  toy_count jerry gabriel jaxon sarah emily →
  jerry + gabriel + jaxon + sarah + emily = 176 :=
by
  sorry

end NUMINAMATH_CALUDE_total_toy_count_l594_59497


namespace NUMINAMATH_CALUDE_M_equals_N_l594_59418

/-- Set M of integers defined as 12m + 8n + 4l where m, n, l are integers -/
def M : Set ℤ := {u : ℤ | ∃ (m n l : ℤ), u = 12*m + 8*n + 4*l}

/-- Set N of integers defined as 20p + 16q + 12r where p, q, r are integers -/
def N : Set ℤ := {u : ℤ | ∃ (p q r : ℤ), u = 20*p + 16*q + 12*r}

/-- Theorem stating that set M is equal to set N -/
theorem M_equals_N : M = N := by sorry

end NUMINAMATH_CALUDE_M_equals_N_l594_59418


namespace NUMINAMATH_CALUDE_problem_solution_l594_59490

def sequence_property (a : ℕ → ℝ) : Prop :=
  (∀ n : ℕ, n ≤ 98 → a n - 2022 * a (n + 1) + 2021 * a (n + 2) ≥ 0) ∧
  (a 99 - 2022 * a 100 + 2021 * a 1 ≥ 0) ∧
  (a 100 - 2022 * a 1 + 2021 * a 2 ≥ 0)

theorem problem_solution (a : ℕ → ℝ) (h : sequence_property a) (h10 : a 10 = 10) : 
  a 22 = 10 := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l594_59490


namespace NUMINAMATH_CALUDE_smallest_n_divisible_by_2013_l594_59471

def product_of_consecutive_evens (n : ℕ) : ℕ :=
  Finset.prod (Finset.range (n/2)) (fun i => 2 * (i + 1))

theorem smallest_n_divisible_by_2013 :
  ∀ n : ℕ, n % 2 = 0 →
    (product_of_consecutive_evens n % 2013 = 0 →
      n ≥ 122) ∧
    (n ≥ 122 →
      product_of_consecutive_evens n % 2013 = 0) :=
sorry

end NUMINAMATH_CALUDE_smallest_n_divisible_by_2013_l594_59471


namespace NUMINAMATH_CALUDE_correct_observation_value_l594_59493

theorem correct_observation_value
  (n : ℕ)
  (initial_mean : ℝ)
  (wrong_value : ℝ)
  (corrected_mean : ℝ)
  (h_n : n = 20)
  (h_initial_mean : initial_mean = 36)
  (h_wrong_value : wrong_value = 40)
  (h_corrected_mean : corrected_mean = 34.9) :
  (n : ℝ) * initial_mean - wrong_value + (n : ℝ) * corrected_mean - ((n : ℝ) * initial_mean - wrong_value) = 18 := by
  sorry

end NUMINAMATH_CALUDE_correct_observation_value_l594_59493


namespace NUMINAMATH_CALUDE_sqrt_200_equals_10_l594_59442

theorem sqrt_200_equals_10 : Real.sqrt 200 = 10 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_200_equals_10_l594_59442


namespace NUMINAMATH_CALUDE_quadratic_symmetry_l594_59430

/-- A quadratic function with specific properties -/
def p (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

/-- Theorem: For a quadratic function p(x) with axis of symmetry at x = 8.5 and p(-1) = -4, p(18) = -4 -/
theorem quadratic_symmetry (a b c : ℝ) :
  (∀ x : ℝ, p a b c (17 - x) = p a b c x) →  -- axis of symmetry at x = 8.5
  p a b c (-1) = -4 →
  p a b c 18 = -4 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_symmetry_l594_59430


namespace NUMINAMATH_CALUDE_athena_spent_14_dollars_l594_59406

/-- The total amount Athena spent on snacks for her friends -/
def total_spent (sandwich_price : ℝ) (sandwich_quantity : ℕ) (drink_price : ℝ) (drink_quantity : ℕ) : ℝ :=
  sandwich_price * sandwich_quantity + drink_price * drink_quantity

/-- Theorem stating that Athena spent $14 in total -/
theorem athena_spent_14_dollars :
  let sandwich_price : ℝ := 3
  let sandwich_quantity : ℕ := 3
  let drink_price : ℝ := 2.5
  let drink_quantity : ℕ := 2
  total_spent sandwich_price sandwich_quantity drink_price drink_quantity = 14 := by
sorry

end NUMINAMATH_CALUDE_athena_spent_14_dollars_l594_59406


namespace NUMINAMATH_CALUDE_bananas_left_l594_59404

theorem bananas_left (initial : ℕ) (eaten : ℕ) : 
  initial = 12 → eaten = 4 → initial - eaten = 8 := by
  sorry

end NUMINAMATH_CALUDE_bananas_left_l594_59404


namespace NUMINAMATH_CALUDE_f_properties_l594_59464

def f (x : ℝ) := -7 * x

theorem f_properties :
  (∀ x y : ℝ, (x > 0 ∧ f x < 0) ∨ (x < 0 ∧ f x > 0)) ∧
  f 1 = -7 ∧
  (∀ x y : ℝ, x < y → f x > f y) :=
sorry

end NUMINAMATH_CALUDE_f_properties_l594_59464


namespace NUMINAMATH_CALUDE_robins_haircut_l594_59421

theorem robins_haircut (initial_length current_length : ℕ) 
  (h1 : initial_length = 17)
  (h2 : current_length = 13) :
  initial_length - current_length = 4 := by
  sorry

end NUMINAMATH_CALUDE_robins_haircut_l594_59421


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l594_59495

theorem simplify_and_evaluate (a : ℚ) (h : a = -3/2) :
  (a + 2)^2 - (a + 1)*(a - 1) = -1 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l594_59495


namespace NUMINAMATH_CALUDE_smallest_ab_value_l594_59457

theorem smallest_ab_value (a b : ℤ) (h : (a : ℚ) / 2 + (b : ℚ) / 1009 = 1 / 2018) :
  ∃ (a₀ b₀ : ℤ), (a₀ : ℚ) / 2 + (b₀ : ℚ) / 1009 = 1 / 2018 ∧ |a₀ * b₀| = 504 ∧
    ∀ (a' b' : ℤ), (a' : ℚ) / 2 + (b' : ℚ) / 1009 = 1 / 2018 → |a' * b'| ≥ 504 :=
by sorry

end NUMINAMATH_CALUDE_smallest_ab_value_l594_59457


namespace NUMINAMATH_CALUDE_rectangle_area_l594_59436

/-- Given a rectangle with perimeter 24 and one side length x (x > 0),
    prove that its area y is equal to (12-x)x -/
theorem rectangle_area (x : ℝ) (hx : x > 0) : 
  let perimeter : ℝ := 24
  let y : ℝ := x * (perimeter / 2 - x)
  y = (12 - x) * x :=
by sorry

end NUMINAMATH_CALUDE_rectangle_area_l594_59436


namespace NUMINAMATH_CALUDE_inequality_proof_l594_59446

theorem inequality_proof (a b : ℝ) (ha : 0 < a) (hb : 0 < b) :
  a + b ≥ Real.sqrt (a * b) + Real.sqrt ((a^2 + b^2) / 2) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l594_59446


namespace NUMINAMATH_CALUDE_fraction_equality_l594_59489

theorem fraction_equality (a b : ℝ) (h : a / (a + b) = 3 / 4) : a / b = 3 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l594_59489


namespace NUMINAMATH_CALUDE_giant_kite_area_l594_59420

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Calculates the area of a kite given its four vertices -/
def kiteArea (p1 p2 p3 p4 : Point) : ℝ :=
  let base := p3.x - p1.x
  let height := p2.y - p1.y
  base * height

/-- Theorem: The area of the specified kite is 72 square inches -/
theorem giant_kite_area :
  let p1 : Point := ⟨2, 12⟩
  let p2 : Point := ⟨8, 18⟩
  let p3 : Point := ⟨14, 12⟩
  let p4 : Point := ⟨8, 2⟩
  kiteArea p1 p2 p3 p4 = 72 := by
  sorry

end NUMINAMATH_CALUDE_giant_kite_area_l594_59420


namespace NUMINAMATH_CALUDE_complex_cube_root_sum_l594_59438

theorem complex_cube_root_sum (z : ℂ) (h1 : z^3 = 1) (h2 : z ≠ 1) :
  z^103 + z^104 + z^105 + z^106 + z^107 + z^108 = 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_cube_root_sum_l594_59438


namespace NUMINAMATH_CALUDE_cubic_equation_roots_l594_59450

theorem cubic_equation_roots (x : ℝ) : ∃ (a b : ℝ),
  (x^3 - x^2 - 2*x + 1 = 0) ∧ (a^3 - a^2 - 2*a + 1 = 0) ∧ (b^3 - b^2 - 2*b + 1 = 0) ∧ (a - a*b = 1) :=
by sorry

end NUMINAMATH_CALUDE_cubic_equation_roots_l594_59450


namespace NUMINAMATH_CALUDE_sams_dimes_l594_59403

/-- Given that Sam had 9 dimes initially and received 7 more dimes from his dad,
    prove that the total number of dimes Sam has now is 16. -/
theorem sams_dimes (initial_dimes : ℕ) (received_dimes : ℕ) (total_dimes : ℕ) : 
  initial_dimes = 9 → received_dimes = 7 → total_dimes = initial_dimes + received_dimes → total_dimes = 16 := by
  sorry

end NUMINAMATH_CALUDE_sams_dimes_l594_59403


namespace NUMINAMATH_CALUDE_bird_count_2003_l594_59487

/-- The number of birds in the Weishui Development Zone over three years -/
structure BirdCount where
  year2001 : ℝ
  year2002 : ℝ
  year2003 : ℝ

/-- The conditions of the bird count problem -/
def bird_count_conditions (bc : BirdCount) : Prop :=
  bc.year2002 = 1.5 * bc.year2001 ∧ 
  bc.year2003 = 2 * bc.year2002

/-- Theorem stating that under the given conditions, the number of birds in 2003 is 3 times the number in 2001 -/
theorem bird_count_2003 (bc : BirdCount) (h : bird_count_conditions bc) : 
  bc.year2003 = 3 * bc.year2001 := by
  sorry

end NUMINAMATH_CALUDE_bird_count_2003_l594_59487


namespace NUMINAMATH_CALUDE_binary_multiplication_division_l594_59494

/-- Convert a binary string to a natural number -/
def binary_to_nat (s : String) : ℕ :=
  s.foldl (fun acc c => 2 * acc + (if c = '1' then 1 else 0)) 0

/-- Convert a natural number to a binary string -/
def nat_to_binary (n : ℕ) : String :=
  if n = 0 then "0" else
  let rec aux (m : ℕ) : String :=
    if m = 0 then "" else aux (m / 2) ++ (if m % 2 = 1 then "1" else "0")
  aux n

theorem binary_multiplication_division :
  let a := binary_to_nat "11100"
  let b := binary_to_nat "11010"
  let c := binary_to_nat "100"
  nat_to_binary ((a * b) / c) = "10100110" := by
  sorry

end NUMINAMATH_CALUDE_binary_multiplication_division_l594_59494


namespace NUMINAMATH_CALUDE_basketball_donations_l594_59415

theorem basketball_donations (total_donations : ℕ) 
  (basketball_hoops : ℕ) (pool_floats : ℕ) (footballs : ℕ) (tennis_balls : ℕ) :
  total_donations = 300 →
  basketball_hoops = 60 →
  pool_floats = 120 →
  footballs = 50 →
  tennis_balls = 40 →
  ∃ (basketballs : ℕ),
    basketballs = total_donations - (basketball_hoops + (pool_floats - pool_floats / 4) + footballs + tennis_balls) + basketball_hoops / 2 ∧
    basketballs = 90 :=
by sorry

end NUMINAMATH_CALUDE_basketball_donations_l594_59415


namespace NUMINAMATH_CALUDE_find_second_number_l594_59405

theorem find_second_number (G N : ℕ) (h1 : G = 101) (h2 : 4351 % G = 8) (h3 : N % G = 10) :
  N = 4359 := by
  sorry

end NUMINAMATH_CALUDE_find_second_number_l594_59405


namespace NUMINAMATH_CALUDE_toms_age_ratio_l594_59469

theorem toms_age_ratio (T N : ℕ) : 
  (∃ (x y z : ℕ), T = x + y + z) →  -- T is the sum of three children's ages
  (T - N = 2 * ((T - N) - 3 * N)) →  -- N years ago, Tom's age was twice the sum of his children's ages
  T / N = 5 := by
sorry

end NUMINAMATH_CALUDE_toms_age_ratio_l594_59469


namespace NUMINAMATH_CALUDE_shopkeeper_loss_percent_l594_59478

theorem shopkeeper_loss_percent 
  (profit_rate : ℝ) 
  (theft_rate : ℝ) 
  (initial_value : ℝ) 
  (profit_rate_is_10_percent : profit_rate = 0.1)
  (theft_rate_is_60_percent : theft_rate = 0.6)
  (initial_value_positive : initial_value > 0) : 
  let remaining_goods := initial_value * (1 - theft_rate)
  let final_value := remaining_goods * (1 + profit_rate)
  let loss := initial_value - final_value
  let loss_percent := (loss / initial_value) * 100
  loss_percent = 56 := by
sorry

end NUMINAMATH_CALUDE_shopkeeper_loss_percent_l594_59478


namespace NUMINAMATH_CALUDE_power_sum_of_i_l594_59459

-- Define the complex number i
def i : ℂ := Complex.I

-- State the theorem
theorem power_sum_of_i : i^20 + i^39 = 1 - i := by
  sorry

end NUMINAMATH_CALUDE_power_sum_of_i_l594_59459


namespace NUMINAMATH_CALUDE_hexadecagon_triangles_l594_59492

/-- The number of vertices in a regular hexadecagon -/
def n : ℕ := 16

/-- Represents that no three vertices of the hexadecagon are collinear -/
axiom no_collinear_vertices : True

/-- The number of triangles that can be formed using the vertices of a regular hexadecagon -/
def num_triangles : ℕ := Nat.choose n 3

theorem hexadecagon_triangles : num_triangles = 560 := by
  sorry

end NUMINAMATH_CALUDE_hexadecagon_triangles_l594_59492


namespace NUMINAMATH_CALUDE_veranda_width_l594_59473

/-- Given a rectangular room with length 19 m and width 12 m, surrounded by a veranda on all sides
    with an area of 140 m², prove that the width of the veranda is 2 m. -/
theorem veranda_width (room_length : ℝ) (room_width : ℝ) (veranda_area : ℝ) :
  room_length = 19 →
  room_width = 12 →
  veranda_area = 140 →
  ∃ (w : ℝ), w = 2 ∧
    (room_length + 2 * w) * (room_width + 2 * w) - room_length * room_width = veranda_area :=
by sorry

end NUMINAMATH_CALUDE_veranda_width_l594_59473


namespace NUMINAMATH_CALUDE_largest_divisor_with_equal_quotient_remainder_l594_59411

theorem largest_divisor_with_equal_quotient_remainder :
  ∀ (A B C : ℕ),
    (10 = A * B + C) →
    (B = C) →
    A ≤ 9 ∧
    (∃ (A' : ℕ), A' = 9 ∧ ∃ (B' C' : ℕ), 10 = A' * B' + C' ∧ B' = C') :=
by sorry

end NUMINAMATH_CALUDE_largest_divisor_with_equal_quotient_remainder_l594_59411


namespace NUMINAMATH_CALUDE_inequality_proof_l594_59455

theorem inequality_proof (x y : ℝ) (h : x^4 + y^4 ≥ 2) :
  |x^12 - y^12| + 2 * x^6 * y^6 ≥ 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l594_59455


namespace NUMINAMATH_CALUDE_mix_g_weekly_amount_l594_59435

/-- Calculates the weekly amount of Mix G birdseed needed for pigeons -/
def weekly_mix_g_amount (num_pigeons : ℕ) (daily_consumption : ℕ) (days : ℕ) : ℕ :=
  num_pigeons * daily_consumption * days

/-- Theorem stating that the weekly amount of Mix G birdseed needed is 168 grams -/
theorem mix_g_weekly_amount :
  weekly_mix_g_amount 6 4 7 = 168 :=
by sorry

end NUMINAMATH_CALUDE_mix_g_weekly_amount_l594_59435


namespace NUMINAMATH_CALUDE_seventeenth_term_is_two_l594_59414

/-- An arithmetic sequence with specific properties -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  d : ℝ      -- Common difference
  sum : ℕ → ℝ -- Sum function
  is_arithmetic : ∀ n, a (n + 1) = a n + d
  sum_formula : ∀ n, sum n = n * (a 1 + a n) / 2
  sum_13 : sum 13 = 78
  sum_7_12 : a 7 + a 12 = 10

/-- The 17th term of the arithmetic sequence is 2 -/
theorem seventeenth_term_is_two (seq : ArithmeticSequence) : seq.a 17 = 2 := by
  sorry

end NUMINAMATH_CALUDE_seventeenth_term_is_two_l594_59414


namespace NUMINAMATH_CALUDE_sum_of_rectangle_areas_l594_59453

def first_six_odd_numbers : List ℕ := [1, 3, 5, 7, 9, 11]

def rectangle_areas (base_width : ℕ) (lengths : List ℕ) : List ℕ :=
  lengths.map (λ l => base_width * l^2)

theorem sum_of_rectangle_areas :
  let base_width := 2
  let areas := rectangle_areas base_width first_six_odd_numbers
  List.sum areas = 572 := by sorry

end NUMINAMATH_CALUDE_sum_of_rectangle_areas_l594_59453


namespace NUMINAMATH_CALUDE_betty_age_l594_59447

/-- Given the ages of Albert, Mary, and Betty, prove that Betty is 4 years old. -/
theorem betty_age (albert mary betty : ℕ) 
  (h1 : albert = 2 * mary)
  (h2 : albert = 4 * betty)
  (h3 : mary = albert - 8) : 
  betty = 4 := by
sorry

end NUMINAMATH_CALUDE_betty_age_l594_59447


namespace NUMINAMATH_CALUDE_div_fraction_equality_sum_fraction_equality_l594_59425

-- Define variables
variable (a b : ℝ)

-- Assume a ≠ b and a ≠ 0 to avoid division by zero
variable (h1 : a ≠ b) (h2 : a ≠ 0)

-- Theorem 1
theorem div_fraction_equality : (4 * b^3 / a) / (2 * b / a^2) = 2 * a * b^2 := by sorry

-- Theorem 2
theorem sum_fraction_equality : a^2 / (a - b) + b^2 / (a - b) - 2 * a * b / (a - b) = a - b := by sorry

end NUMINAMATH_CALUDE_div_fraction_equality_sum_fraction_equality_l594_59425


namespace NUMINAMATH_CALUDE_gcd_consecutive_b_terms_is_one_l594_59431

def b (n : ℕ) : ℕ := 2 * n.factorial + n

theorem gcd_consecutive_b_terms_is_one (n : ℕ) : 
  Nat.gcd (b n) (b (n + 1)) = 1 := by sorry

end NUMINAMATH_CALUDE_gcd_consecutive_b_terms_is_one_l594_59431
