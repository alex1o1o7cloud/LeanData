import Mathlib

namespace NUMINAMATH_CALUDE_carriage_sharing_equation_correct_l1379_137927

/-- Represents the problem of "multiple people sharing a carriage" --/
def carriage_sharing_problem (x : ℕ) : Prop :=
  -- When 3 people share 1 carriage, 2 carriages are left empty
  (x / 3 : ℚ) + 2 = 
  -- When 2 people share 1 carriage, 9 people are left without a carriage
  ((x - 9) / 2 : ℚ)

/-- The equation (x/3) + 2 = (x-9)/2 correctly represents the carriage sharing problem --/
theorem carriage_sharing_equation_correct (x : ℕ) : 
  carriage_sharing_problem x ↔ (x / 3 : ℚ) + 2 = ((x - 9) / 2 : ℚ) :=
sorry

end NUMINAMATH_CALUDE_carriage_sharing_equation_correct_l1379_137927


namespace NUMINAMATH_CALUDE_max_pawns_2019_l1379_137979

/-- Represents a chessboard with dimensions n x n -/
structure Chessboard (n : ℕ) where
  size : ℕ := n

/-- Represents the placement of pieces on the chessboard -/
structure Placement (n : ℕ) where
  board : Chessboard n
  pawns : ℕ
  rooks : ℕ
  no_rooks_see_each_other : Bool

/-- The maximum number of pawns that can be placed -/
def max_pawns (n : ℕ) : ℕ := (n / 2) ^ 2

/-- Theorem stating the maximum number of pawns for a 2019 x 2019 chessboard -/
theorem max_pawns_2019 :
  ∃ (p : Placement 2019),
    p.pawns = max_pawns 2019 ∧
    p.rooks = p.pawns + 2019 ∧
    p.no_rooks_see_each_other = true ∧
    ∀ (q : Placement 2019),
      q.no_rooks_see_each_other = true →
      q.rooks = q.pawns + 2019 →
      q.pawns ≤ p.pawns :=
by sorry

end NUMINAMATH_CALUDE_max_pawns_2019_l1379_137979


namespace NUMINAMATH_CALUDE_compound_interest_rate_l1379_137902

theorem compound_interest_rate (P r : ℝ) (h1 : P * (1 + r / 100) ^ 2 = 3650) (h2 : P * (1 + r / 100) ^ 3 = 4015) : r = 10 := by
  sorry

end NUMINAMATH_CALUDE_compound_interest_rate_l1379_137902


namespace NUMINAMATH_CALUDE_equation_has_real_root_l1379_137943

theorem equation_has_real_root (M : ℝ) : ∃ x : ℝ, x = M^2 * (x - 1) * (x - 2) * (x - 3) := by
  sorry

end NUMINAMATH_CALUDE_equation_has_real_root_l1379_137943


namespace NUMINAMATH_CALUDE_volume_of_one_gram_l1379_137993

/-- Given a substance with the following properties:
  - The mass of 1 cubic meter of the substance is 200 kg.
  - 1 kg = 1,000 grams
  - 1 cubic meter = 1,000,000 cubic centimeters
  
  This theorem proves that the volume of 1 gram of this substance is 5 cubic centimeters. -/
theorem volume_of_one_gram (mass_per_cubic_meter : ℝ) (kg_to_g : ℝ) (cubic_meter_to_cubic_cm : ℝ) :
  mass_per_cubic_meter = 200 →
  kg_to_g = 1000 →
  cubic_meter_to_cubic_cm = 1000000 →
  (1 : ℝ) / (mass_per_cubic_meter * kg_to_g / cubic_meter_to_cubic_cm) = 5 := by
  sorry

#check volume_of_one_gram

end NUMINAMATH_CALUDE_volume_of_one_gram_l1379_137993


namespace NUMINAMATH_CALUDE_rainfall_second_week_l1379_137947

theorem rainfall_second_week (total_rainfall : ℝ) (ratio : ℝ) :
  total_rainfall = 40 →
  ratio = 1.5 →
  ∃ (first_week : ℝ) (second_week : ℝ),
    first_week + second_week = total_rainfall ∧
    second_week = ratio * first_week ∧
    second_week = 24 := by
  sorry

end NUMINAMATH_CALUDE_rainfall_second_week_l1379_137947


namespace NUMINAMATH_CALUDE_train_speed_l1379_137976

/-- The speed of a train passing a jogger --/
theorem train_speed (jogger_speed : ℝ) (initial_lead : ℝ) (train_length : ℝ) (passing_time : ℝ) : 
  jogger_speed = 9 →
  initial_lead = 240 →
  train_length = 110 →
  passing_time = 35 →
  ∃ (train_speed : ℝ), train_speed = 45 := by
  sorry

#check train_speed

end NUMINAMATH_CALUDE_train_speed_l1379_137976


namespace NUMINAMATH_CALUDE_train_speed_l1379_137959

/-- Proves that a train of given length crossing a bridge of given length in a given time has a specific speed in km/hr -/
theorem train_speed (train_length bridge_length : Real) (crossing_time : Real) : 
  train_length = 150 ∧ 
  bridge_length = 225 ∧ 
  crossing_time = 30 → 
  (train_length + bridge_length) / crossing_time * 3.6 = 45 := by
  sorry

end NUMINAMATH_CALUDE_train_speed_l1379_137959


namespace NUMINAMATH_CALUDE_prime_cube_difference_to_sum_of_squares_l1379_137926

theorem prime_cube_difference_to_sum_of_squares (p a b : ℕ) : 
  Prime p → (∃ a b : ℕ, p = a^3 - b^3) → (∃ c d : ℕ, p = c^2 + 3*d^2) := by
  sorry

end NUMINAMATH_CALUDE_prime_cube_difference_to_sum_of_squares_l1379_137926


namespace NUMINAMATH_CALUDE_isosceles_triangle_height_decreases_as_base_increases_l1379_137960

/-- Given an isosceles triangle with fixed side length and variable base length,
    the height is a decreasing function of the base length. -/
theorem isosceles_triangle_height_decreases_as_base_increases 
  (a : ℝ) (b : ℝ → ℝ) (h : ℝ → ℝ) :
  (∀ x, a > 0 ∧ b x > 0 ∧ h x > 0) →  -- Positive lengths
  (∀ x, a^2 = (h x)^2 + (b x)^2) →   -- Pythagorean theorem
  (∀ x, h x = Real.sqrt (a^2 - (b x)^2)) →  -- Height formula
  (∀ x y, x < y → b x < b y) →  -- b is increasing
  (∀ x y, x < y → h x > h y) :=  -- h is decreasing
by sorry


end NUMINAMATH_CALUDE_isosceles_triangle_height_decreases_as_base_increases_l1379_137960


namespace NUMINAMATH_CALUDE_area_at_stage_6_l1379_137970

/-- The side length of each square in inches -/
def square_side : ℕ := 4

/-- The number of squares at a given stage -/
def num_squares (stage : ℕ) : ℕ := stage

/-- The area of the rectangle at a given stage in square inches -/
def rectangle_area (stage : ℕ) : ℕ :=
  (num_squares stage) * (square_side * square_side)

/-- Theorem: The area of the rectangle at Stage 6 is 96 square inches -/
theorem area_at_stage_6 : rectangle_area 6 = 96 := by
  sorry

end NUMINAMATH_CALUDE_area_at_stage_6_l1379_137970


namespace NUMINAMATH_CALUDE_max_intersections_proof_l1379_137981

/-- The number of points on the positive x-axis -/
def num_x_points : ℕ := 12

/-- The number of points on the positive y-axis -/
def num_y_points : ℕ := 6

/-- The maximum number of intersection points in the first quadrant -/
def max_intersections : ℕ := Nat.choose num_x_points 2 * Nat.choose num_y_points 2

theorem max_intersections_proof :
  max_intersections = 990 :=
sorry

end NUMINAMATH_CALUDE_max_intersections_proof_l1379_137981


namespace NUMINAMATH_CALUDE_nick_money_value_l1379_137945

/-- The value of a nickel in cents -/
def nickel_value : ℕ := 5

/-- The value of a dime in cents -/
def dime_value : ℕ := 10

/-- The value of a quarter in cents -/
def quarter_value : ℕ := 25

/-- The number of nickels Nick has -/
def num_nickels : ℕ := 6

/-- The number of dimes Nick has -/
def num_dimes : ℕ := 2

/-- The number of quarters Nick has -/
def num_quarters : ℕ := 1

/-- The total value of Nick's coins in cents -/
def total_value : ℕ := num_nickels * nickel_value + num_dimes * dime_value + num_quarters * quarter_value

theorem nick_money_value : total_value = 75 := by
  sorry

end NUMINAMATH_CALUDE_nick_money_value_l1379_137945


namespace NUMINAMATH_CALUDE_homework_submission_negation_l1379_137922

variable (Student : Type)
variable (inClass : Student → Prop)
variable (submittedHomework : Student → Prop)

theorem homework_submission_negation :
  (¬ ∀ s : Student, inClass s → submittedHomework s) ↔
  (∃ s : Student, inClass s ∧ ¬ submittedHomework s) :=
by sorry

end NUMINAMATH_CALUDE_homework_submission_negation_l1379_137922


namespace NUMINAMATH_CALUDE_semicircle_square_properties_l1379_137955

-- Define the semicircle and inscribed square
def semicircle_with_square (a b : ℝ) : Prop :=
  ∃ (A B C D E F : ℝ × ℝ),
    -- A and B are endpoints of the diameter
    (A.1 - B.1)^2 + (A.2 - B.2)^2 = 4 * ((C.1 - D.1)^2 + (C.2 - D.2)^2) ∧
    -- CDEF is a square with side length 1
    (C.1 - D.1)^2 + (C.2 - D.2)^2 = 1 ∧
    (D.1 - E.1)^2 + (D.2 - E.2)^2 = 1 ∧
    (E.1 - F.1)^2 + (E.2 - F.2)^2 = 1 ∧
    (F.1 - C.1)^2 + (F.2 - C.2)^2 = 1 ∧
    -- AC = a and BC = b
    (A.1 - C.1)^2 + (A.2 - C.2)^2 = a^2 ∧
    (B.1 - C.1)^2 + (B.2 - C.2)^2 = b^2

-- State the theorem
theorem semicircle_square_properties (a b : ℝ) (h : semicircle_with_square a b) :
  a - b = 1 ∧ a * b = 1 ∧ a + b = Real.sqrt 5 ∧ a^2 + b^2 ≠ 5 := by
  sorry

end NUMINAMATH_CALUDE_semicircle_square_properties_l1379_137955


namespace NUMINAMATH_CALUDE_even_function_implies_a_equals_two_l1379_137930

/-- A function f: ℝ → ℝ is even if f(-x) = f(x) for all x ∈ ℝ -/
def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

/-- The function f(x) = (a+1)x^2 + (a-2)x + a^2 - a - 2 -/
def f (a : ℝ) (x : ℝ) : ℝ := (a+1)*x^2 + (a-2)*x + a^2 - a - 2

theorem even_function_implies_a_equals_two :
  ∀ a : ℝ, IsEven (f a) → a = 2 := by sorry

end NUMINAMATH_CALUDE_even_function_implies_a_equals_two_l1379_137930


namespace NUMINAMATH_CALUDE_sum_reciprocals_l1379_137963

theorem sum_reciprocals (a b c d : ℝ) (ω : ℂ) :
  a ≠ -1 → b ≠ -1 → c ≠ -1 → d ≠ -1 →
  ω^4 = 1 →
  ω ≠ 1 →
  (1 / (a + ω)) + (1 / (b + ω)) + (1 / (c + ω)) + (1 / (d + ω)) = 4 / ω →
  (1 / (a + 1)) + (1 / (b + 1)) + (1 / (c + 1)) + (1 / (d + 1)) = 1 := by
sorry

end NUMINAMATH_CALUDE_sum_reciprocals_l1379_137963


namespace NUMINAMATH_CALUDE_mean_equality_implies_sum_l1379_137949

theorem mean_equality_implies_sum (y z : ℝ) : 
  (8 + 15 + 21) / 3 = (14 + y + z) / 3 → y + z = 30 := by
  sorry

end NUMINAMATH_CALUDE_mean_equality_implies_sum_l1379_137949


namespace NUMINAMATH_CALUDE_parallel_line_through_point_l1379_137909

/-- A line in 2D space represented by the equation ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Check if a point lies on a line -/
def Point.liesOn (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

/-- Check if two lines are parallel -/
def Line.isParallelTo (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l1.b * l2.a

/-- The given line 2x - 3y + 4 = 0 -/
def givenLine : Line := { a := 2, b := -3, c := 4 }

/-- The point (-1, 2) -/
def givenPoint : Point := { x := -1, y := 2 }

/-- The line we want to prove -/
def targetLine : Line := { a := 2, b := -3, c := 8 }

theorem parallel_line_through_point :
  (targetLine.isParallelTo givenLine) ∧
  (givenPoint.liesOn targetLine) := by
  sorry

#check parallel_line_through_point

end NUMINAMATH_CALUDE_parallel_line_through_point_l1379_137909


namespace NUMINAMATH_CALUDE_soldiers_on_great_wall_count_l1379_137904

/-- The number of soldiers in beacon towers along the Great Wall --/
def soldiers_on_great_wall (wall_length : ℕ) (tower_interval : ℕ) (soldiers_per_tower : ℕ) : ℕ :=
  (wall_length / tower_interval) * soldiers_per_tower

/-- Theorem stating the number of soldiers on the Great Wall --/
theorem soldiers_on_great_wall_count :
  soldiers_on_great_wall 7300 5 2 = 2920 := by
  sorry

end NUMINAMATH_CALUDE_soldiers_on_great_wall_count_l1379_137904


namespace NUMINAMATH_CALUDE_union_of_M_and_N_l1379_137919

-- Define sets M and N
def M : Set ℝ := {x | -3 < x ∧ x < 1}
def N : Set ℝ := {x | x ≤ -3}

-- State the theorem
theorem union_of_M_and_N : M ∪ N = {x : ℝ | x < 1} := by sorry

end NUMINAMATH_CALUDE_union_of_M_and_N_l1379_137919


namespace NUMINAMATH_CALUDE_quadratic_properties_l1379_137944

theorem quadratic_properties (a b c m : ℝ) : 
  a < 0 →
  -2 < m →
  m < -1 →
  a * 1^2 + b * 1 + c = 0 →
  a * m^2 + b * m + c = 0 →
  b < 0 ∧ 
  a + b + c = 0 ∧ 
  a * (m + 1) - b + c > 0 := by
sorry

end NUMINAMATH_CALUDE_quadratic_properties_l1379_137944


namespace NUMINAMATH_CALUDE_second_round_difference_l1379_137924

/-- Bowling game results -/
structure BowlingGame where
  patrick_first : ℕ
  richard_first : ℕ
  patrick_second : ℕ
  richard_second : ℕ

/-- Conditions of the bowling game -/
def bowling_conditions (game : BowlingGame) : Prop :=
  game.patrick_first = 70 ∧
  game.richard_first = game.patrick_first + 15 ∧
  game.patrick_second = 2 * game.richard_first ∧
  game.richard_second < game.patrick_second ∧
  game.richard_first + game.richard_second = game.patrick_first + game.patrick_second + 12

/-- Theorem: The difference between Patrick's and Richard's knocked down pins in the second round is 3 -/
theorem second_round_difference (game : BowlingGame) 
  (h : bowling_conditions game) : 
  game.patrick_second - game.richard_second = 3 := by
  sorry

end NUMINAMATH_CALUDE_second_round_difference_l1379_137924


namespace NUMINAMATH_CALUDE_ice_cream_sales_l1379_137972

def daily_sales : List ℝ := [100, 92, 109, 96, 0, 96, 105]

theorem ice_cream_sales (x : ℝ) :
  let sales := daily_sales.set 4 x
  sales.length = 7 ∧ 
  sales.sum / sales.length = 100.1 →
  x = 102.7 := by
sorry

end NUMINAMATH_CALUDE_ice_cream_sales_l1379_137972


namespace NUMINAMATH_CALUDE_percentage_deposited_approx_28_percent_l1379_137965

def deposit : ℝ := 4500
def monthly_income : ℝ := 16071.42857142857

theorem percentage_deposited_approx_28_percent :
  ∃ ε > 0, ε < 0.01 ∧ |deposit / monthly_income * 100 - 28| < ε := by
  sorry

end NUMINAMATH_CALUDE_percentage_deposited_approx_28_percent_l1379_137965


namespace NUMINAMATH_CALUDE_parallel_vectors_condition_l1379_137906

variable {V : Type*} [NormedAddCommGroup V] [NormedSpace ℝ V]

theorem parallel_vectors_condition (a b : V) (ha : a ≠ 0) (hb : b ≠ 0) :
  (a + 2 • b = 0 → ∃ k : ℝ, a = k • b) ∧
  ¬(∃ k : ℝ, a = k • b → a + 2 • b = 0) :=
sorry

end NUMINAMATH_CALUDE_parallel_vectors_condition_l1379_137906


namespace NUMINAMATH_CALUDE_spiders_in_room_l1379_137928

theorem spiders_in_room (total_legs : ℕ) (legs_per_spider : ℕ) (h1 : total_legs = 32) (h2 : legs_per_spider = 8) :
  total_legs / legs_per_spider = 4 := by
sorry

end NUMINAMATH_CALUDE_spiders_in_room_l1379_137928


namespace NUMINAMATH_CALUDE_antons_offer_is_cheapest_l1379_137995

/-- Represents a shareholder in the company -/
structure Shareholder where
  name : String
  shares : Nat
  yield : Rat

/-- Represents the company and its shareholders -/
structure Company where
  totalShares : Nat
  sharePrice : Nat
  shareholders : List Shareholder

def Company.largestShareholderShares (c : Company) : Nat :=
  c.shareholders.map (·.shares) |>.maximum?.getD 0

def buySharesCost (sharePrice : Nat) (shares : Nat) (yield : Rat) : Nat :=
  Nat.ceil (sharePrice * shares * (1 + yield))

theorem antons_offer_is_cheapest (c : Company) (arina : Shareholder) : 
  c.totalShares = 300000 ∧
  c.sharePrice = 10 ∧
  arina.shares = 90001 ∧
  c.shareholders = [
    ⟨"Maxim", 104999, 1/10⟩,
    ⟨"Inga", 30000, 1/4⟩,
    ⟨"Yuri", 30000, 3/20⟩,
    ⟨"Yulia", 30000, 3/10⟩,
    ⟨"Anton", 15000, 2/5⟩
  ] →
  let requiredShares := c.largestShareholderShares - arina.shares + 1
  let antonsCost := buySharesCost c.sharePrice (c.shareholders.find? (·.name = "Anton") |>.map (·.shares) |>.getD 0) (2/5)
  ∀ s ∈ c.shareholders, s.name ≠ "Anton" →
    buySharesCost c.sharePrice s.shares s.yield ≥ antonsCost ∧
    s.shares ≥ requiredShares →
    antonsCost ≤ buySharesCost c.sharePrice s.shares s.yield :=
by sorry


end NUMINAMATH_CALUDE_antons_offer_is_cheapest_l1379_137995


namespace NUMINAMATH_CALUDE_largest_divisor_of_n4_minus_n2_l1379_137962

theorem largest_divisor_of_n4_minus_n2 (n : ℤ) : 
  ∃ (m : ℕ), m = 6 ∧ 
  (∃ (k : ℤ), n^4 - n^2 = m * k) ∧ 
  (∀ (d : ℕ), d > m → ¬∃ (j : ℤ), ∀ (n : ℤ), n^4 - n^2 = d * j) :=
sorry

end NUMINAMATH_CALUDE_largest_divisor_of_n4_minus_n2_l1379_137962


namespace NUMINAMATH_CALUDE_unspent_portion_after_transfer_l1379_137914

/-- Represents a credit card with a spending limit and balance. -/
structure CreditCard where
  limit : ℝ
  balance : ℝ

/-- Calculates the unspent portion of a credit card's limit after a balance transfer. -/
def unspentPortionAfterTransfer (gold : CreditCard) (platinum : CreditCard) : ℝ :=
  platinum.limit - (platinum.balance + gold.balance)

/-- Theorem stating the unspent portion of the platinum card's limit after transferring the gold card's balance. -/
theorem unspent_portion_after_transfer 
  (gold : CreditCard) 
  (platinum : CreditCard) 
  (h1 : gold.limit > 0)
  (h2 : platinum.limit = 2 * gold.limit)
  (h3 : gold.balance = (1/3) * gold.limit)
  (h4 : platinum.balance = (1/4) * platinum.limit) :
  unspentPortionAfterTransfer gold platinum = (7/6) * gold.limit :=
by
  sorry

#check unspent_portion_after_transfer

end NUMINAMATH_CALUDE_unspent_portion_after_transfer_l1379_137914


namespace NUMINAMATH_CALUDE_sphere_radius_from_intersection_l1379_137901

theorem sphere_radius_from_intersection (r h : ℝ) : 
  r > 0 → h > 0 → r^2 + h^2 = (r + h)^2 → r = 12 → h = 8 → r + h = 13 := by
  sorry

end NUMINAMATH_CALUDE_sphere_radius_from_intersection_l1379_137901


namespace NUMINAMATH_CALUDE_jam_eaten_for_lunch_l1379_137982

theorem jam_eaten_for_lunch (x : ℚ) : 
  (1 - x) * (1 - 1/7) = 4/7 → x = 1/21 := by
  sorry

end NUMINAMATH_CALUDE_jam_eaten_for_lunch_l1379_137982


namespace NUMINAMATH_CALUDE_equation_solution_l1379_137900

theorem equation_solution :
  ∃ x : ℝ, (x^2 + 3*x + 2) / (x^2 + 1) = x - 2 ∧ x = 4 := by
sorry

end NUMINAMATH_CALUDE_equation_solution_l1379_137900


namespace NUMINAMATH_CALUDE_equation_solutions_l1379_137948

theorem equation_solutions :
  (∃ x : ℚ, 8 * (x + 2)^3 = 27 ↔ x = -1/2) ∧
  (∃ x₁ x₂ : ℚ, 25 * (x₁ - 1)^2 = 4 ∧ 25 * (x₂ - 1)^2 = 4 ↔ x₁ = 7/5 ∧ x₂ = 3/5) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l1379_137948


namespace NUMINAMATH_CALUDE_systematic_sampling_l1379_137988

theorem systematic_sampling (total_students : Nat) (sample_size : Nat) (last_sample : Nat) 
  (h1 : total_students = 300)
  (h2 : sample_size = 60)
  (h3 : last_sample = 293) :
  ∃ (first_sample : Nat), first_sample = 3 ∧ 
  (first_sample + (sample_size - 1) * (total_students / sample_size) = last_sample) := by
  sorry

end NUMINAMATH_CALUDE_systematic_sampling_l1379_137988


namespace NUMINAMATH_CALUDE_tuna_price_is_two_l1379_137996

/-- Represents the daily catch and earnings of a fisherman -/
structure FishermanData where
  red_snappers : ℕ
  tunas : ℕ
  red_snapper_price : ℚ
  daily_earnings : ℚ

/-- Calculates the price of a Tuna given the fisherman's data -/
def tuna_price (data : FishermanData) : ℚ :=
  (data.daily_earnings - data.red_snappers * data.red_snapper_price) / data.tunas

/-- Theorem stating that the price of a Tuna is $2 given the fisherman's data -/
theorem tuna_price_is_two (data : FishermanData)
  (h1 : data.red_snappers = 8)
  (h2 : data.tunas = 14)
  (h3 : data.red_snapper_price = 3)
  (h4 : data.daily_earnings = 52) :
  tuna_price data = 2 := by
  sorry

end NUMINAMATH_CALUDE_tuna_price_is_two_l1379_137996


namespace NUMINAMATH_CALUDE_storks_on_fence_l1379_137939

/-- The number of storks that joined the birds on the fence -/
def storks_joined : ℕ := 6

/-- The initial number of birds on the fence -/
def initial_birds : ℕ := 3

/-- The number of additional birds that joined -/
def additional_birds : ℕ := 2

theorem storks_on_fence :
  storks_joined = initial_birds + additional_birds + 1 :=
by sorry

end NUMINAMATH_CALUDE_storks_on_fence_l1379_137939


namespace NUMINAMATH_CALUDE_typist_salary_problem_l1379_137908

theorem typist_salary_problem (original_salary : ℝ) : 
  (original_salary * 1.1 * 0.95 = 3135) → original_salary = 3000 := by
  sorry

end NUMINAMATH_CALUDE_typist_salary_problem_l1379_137908


namespace NUMINAMATH_CALUDE_house_rent_calculation_l1379_137978

/-- Given a person's expenditure pattern and petrol cost, calculate house rent -/
theorem house_rent_calculation (income : ℝ) (petrol_percentage : ℝ) (rent_percentage : ℝ) (petrol_cost : ℝ) : 
  petrol_percentage = 0.3 →
  rent_percentage = 0.2 →
  petrol_cost = 300 →
  petrol_cost = petrol_percentage * income →
  rent_percentage * (income - petrol_cost) = 140 :=
by sorry

end NUMINAMATH_CALUDE_house_rent_calculation_l1379_137978


namespace NUMINAMATH_CALUDE_not_or_implies_both_false_l1379_137989

theorem not_or_implies_both_false (p q : Prop) : 
  ¬(p ∨ q) → ¬p ∧ ¬q := by sorry

end NUMINAMATH_CALUDE_not_or_implies_both_false_l1379_137989


namespace NUMINAMATH_CALUDE_moon_mission_cost_share_l1379_137969

/-- Calculates the individual share of a total cost divided equally among a population -/
def individual_share (total_cost : ℕ) (population : ℕ) : ℚ :=
  (total_cost : ℚ) / (population : ℚ)

/-- Proves that the individual share of 40 billion dollars among 200 million people is 200 dollars -/
theorem moon_mission_cost_share :
  individual_share (40 * 10^9) (200 * 10^6) = 200 := by
  sorry

end NUMINAMATH_CALUDE_moon_mission_cost_share_l1379_137969


namespace NUMINAMATH_CALUDE_cube_vertex_to_plane_distance_l1379_137917

/-- The distance from the closest vertex of a cube to a plane, given specific conditions --/
theorem cube_vertex_to_plane_distance (s : ℝ) (h₁ h₂ h₃ : ℝ) : 
  s = 8 ∧ h₁ = 8 ∧ h₂ = 9 ∧ h₃ = 10 → 
  ∃ (a b c d : ℝ), 
    a^2 + b^2 + c^2 = 1 ∧
    s * a + d = h₁ ∧
    s * b + d = h₂ ∧
    s * c + d = h₃ ∧
    d = (27 - Real.sqrt 186) / 3 := by
  sorry

#check cube_vertex_to_plane_distance

end NUMINAMATH_CALUDE_cube_vertex_to_plane_distance_l1379_137917


namespace NUMINAMATH_CALUDE_quadratic_roots_property_l1379_137999

theorem quadratic_roots_property (d e : ℝ) : 
  (5 * d^2 - 4 * d - 1 = 0) → 
  (5 * e^2 - 4 * e - 1 = 0) → 
  (d - 2) * (e - 2) = 11/5 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_property_l1379_137999


namespace NUMINAMATH_CALUDE_ice_cream_flavors_l1379_137966

/-- The number of ways to distribute n indistinguishable objects into k distinguishable categories -/
def distribute (n k : ℕ) : ℕ := Nat.choose (n + k - 1) (k - 1)

/-- The number of basic flavors -/
def num_flavors : ℕ := 4

/-- The number of scoops in each new flavor -/
def num_scoops : ℕ := 5

theorem ice_cream_flavors :
  distribute num_scoops num_flavors = Nat.choose (num_scoops + num_flavors - 1) (num_flavors - 1) :=
by sorry

end NUMINAMATH_CALUDE_ice_cream_flavors_l1379_137966


namespace NUMINAMATH_CALUDE_statement_truth_condition_l1379_137916

theorem statement_truth_condition (g : ℝ → ℝ) (c d : ℝ) :
  (∀ x, g x = 4 * x + 5) →
  c > 0 →
  d > 0 →
  (∀ x, |x + 3| < d → |g x + 7| < c) ↔
  d ≤ c / 4 :=
by sorry

end NUMINAMATH_CALUDE_statement_truth_condition_l1379_137916


namespace NUMINAMATH_CALUDE_triangle_inradius_l1379_137903

/-- Given a triangle with perimeter 48 cm and area 60 cm², its inradius is 2.5 cm. -/
theorem triangle_inradius (p : ℝ) (A : ℝ) (r : ℝ) 
  (h1 : p = 48) 
  (h2 : A = 60) 
  (h3 : A = r * p / 2) : 
  r = 2.5 := by
  sorry

end NUMINAMATH_CALUDE_triangle_inradius_l1379_137903


namespace NUMINAMATH_CALUDE_annual_rent_per_square_foot_l1379_137994

-- Define the shop dimensions
def shop_length : ℝ := 20
def shop_width : ℝ := 15

-- Define the monthly rent
def monthly_rent : ℝ := 3600

-- Define the number of months in a year
def months_per_year : ℕ := 12

-- Theorem statement
theorem annual_rent_per_square_foot :
  let shop_area := shop_length * shop_width
  let annual_rent := monthly_rent * months_per_year
  annual_rent / shop_area = 144 := by
  sorry

end NUMINAMATH_CALUDE_annual_rent_per_square_foot_l1379_137994


namespace NUMINAMATH_CALUDE_tangent_slope_perpendicular_lines_l1379_137992

/-- The function f(x) = x^3 + 3ax --/
def f (a : ℝ) (x : ℝ) : ℝ := x^3 + 3*a*x

/-- The derivative of f(x) --/
def f_derivative (a : ℝ) (x : ℝ) : ℝ := 3*x^2 + 3*a

theorem tangent_slope_perpendicular_lines (a : ℝ) :
  (f_derivative a 1 = 6) ↔ ((-1 : ℝ) * (-a) = -1) :=
sorry

end NUMINAMATH_CALUDE_tangent_slope_perpendicular_lines_l1379_137992


namespace NUMINAMATH_CALUDE_boy_running_speed_l1379_137973

/-- Calculates the speed of a boy running around a square field -/
theorem boy_running_speed (side : ℝ) (time : ℝ) (speed_kmh : ℝ) : 
  side = 40 → 
  time = 64 → 
  speed_kmh = (4 * side / time) * 3.6 →
  speed_kmh = 9 := by
  sorry

#check boy_running_speed

end NUMINAMATH_CALUDE_boy_running_speed_l1379_137973


namespace NUMINAMATH_CALUDE_class_trip_problem_l1379_137952

theorem class_trip_problem (x y : ℕ) : 
  ((x + 5) * (y + 6) = x * y + 792) →
  ((x - 4) * (y + 4) = x * y - 388) →
  (x = 27 ∧ y = 120) :=
by sorry

end NUMINAMATH_CALUDE_class_trip_problem_l1379_137952


namespace NUMINAMATH_CALUDE_boat_rental_cost_sharing_l1379_137907

theorem boat_rental_cost_sharing (total_cost : ℝ) (initial_friends : ℕ) (additional_friends : ℕ) (cost_reduction : ℝ) :
  total_cost = 180 →
  initial_friends = 4 →
  additional_friends = 2 →
  cost_reduction = 15 →
  (total_cost / initial_friends) - cost_reduction = (total_cost / (initial_friends + additional_friends)) →
  total_cost / (initial_friends + additional_friends) = 30 :=
by sorry

end NUMINAMATH_CALUDE_boat_rental_cost_sharing_l1379_137907


namespace NUMINAMATH_CALUDE_emily_pastry_production_l1379_137964

/-- Emily's pastry production problem -/
theorem emily_pastry_production (p h : ℕ) : 
  p = 3 * h →
  h = 1 →
  (p - 3) * (h + 3) - p * h = 3 := by
  sorry

end NUMINAMATH_CALUDE_emily_pastry_production_l1379_137964


namespace NUMINAMATH_CALUDE_only_parallel_converse_true_l1379_137925

-- Define the basic concepts
def Line : Type := sorry
def Angle : Type := sorry
def Triangle : Type := sorry

-- Define properties and relations
def parallel (l1 l2 : Line) : Prop := sorry
def alternateInteriorAngles (a1 a2 : Angle) (l1 l2 : Line) : Prop := sorry
def isosceles (t : Triangle) : Prop := sorry
def acute (t : Triangle) : Prop := sorry
def rightAngle (a : Angle) : Prop := sorry
def correspondingAngles (a1 a2 : Angle) (t1 t2 : Triangle) : Prop := sorry
def congruent (t1 t2 : Triangle) : Prop := sorry

-- Theorem stating that only the converse of proposition B is true
theorem only_parallel_converse_true :
  (∀ t : Triangle, acute t → isosceles t) = False ∧
  (∀ l1 l2 : Line, ∀ a1 a2 : Angle, alternateInteriorAngles a1 a2 l1 l2 → parallel l1 l2) = True ∧
  (∀ t1 t2 : Triangle, ∀ a1 a2 : Angle, correspondingAngles a1 a2 t1 t2 → congruent t1 t2) = False ∧
  (∀ a1 a2 : Angle, a1 = a2 → rightAngle a1 ∧ rightAngle a2) = False :=
sorry

end NUMINAMATH_CALUDE_only_parallel_converse_true_l1379_137925


namespace NUMINAMATH_CALUDE_function_forms_theorem_l1379_137938

/-- The set of all non-negative integers -/
def S : Set ℕ := Set.univ

/-- The condition that must be satisfied by f, g, and h -/
def satisfies_condition (f g h : ℕ → ℕ) : Prop :=
  ∀ m n, f (m + n) = g m + h n + 2 * m * n

/-- The theorem stating the only possible forms of f, g, and h -/
theorem function_forms_theorem (f g h : ℕ → ℕ) 
  (h1 : satisfies_condition f g h) (h2 : g 1 = 1) (h3 : h 1 = 1) :
  ∃ a : ℕ, a ≤ 4 ∧ 
    (∀ n, f n = n^2 - a*n + 2*a) ∧
    (∀ n, g n = n^2 - a*n + a) ∧
    (∀ n, h n = n^2 - a*n + a) :=
sorry


end NUMINAMATH_CALUDE_function_forms_theorem_l1379_137938


namespace NUMINAMATH_CALUDE_no_solution_quadratic_with_constraint_l1379_137983

theorem no_solution_quadratic_with_constraint : 
  ¬ ∃ (x : ℝ), x^2 - 4*x + 4 = 0 ∧ x ≠ 2 := by
sorry

end NUMINAMATH_CALUDE_no_solution_quadratic_with_constraint_l1379_137983


namespace NUMINAMATH_CALUDE_distance_AA_l1379_137933

/-- Two unit circles intersecting at X and Y with distance 1 between them -/
structure IntersectingCircles where
  X : ℝ × ℝ
  Y : ℝ × ℝ
  dist_X_Y : dist X Y = 1

/-- Point C on one circle with tangents to the other circle -/
structure TangentPoint (ic : IntersectingCircles) where
  C : ℝ × ℝ
  on_circle : (∃ center, dist center C = 1 ∧ (center = ic.X ∨ center = ic.Y))
  A : ℝ × ℝ  -- Point where tangent CA touches the other circle
  B : ℝ × ℝ  -- Point where tangent CB touches the other circle
  is_tangent_A : ∃ center, dist center A = 1 ∧ center ≠ C
  is_tangent_B : ∃ center, dist center B = 1 ∧ center ≠ C

/-- A' is the point where CB intersects the first circle again -/
def A' (ic : IntersectingCircles) (tp : TangentPoint ic) : ℝ × ℝ :=
  sorry  -- Definition of A' based on the given conditions

/-- The main theorem to prove -/
theorem distance_AA'_is_sqrt3 (ic : IntersectingCircles) (tp : TangentPoint ic) :
  dist tp.A (A' ic tp) = Real.sqrt 3 :=
sorry

end NUMINAMATH_CALUDE_distance_AA_l1379_137933


namespace NUMINAMATH_CALUDE_power_of_product_l1379_137971

theorem power_of_product (a b : ℝ) : (a * b^3)^2 = a^2 * b^6 := by
  sorry

end NUMINAMATH_CALUDE_power_of_product_l1379_137971


namespace NUMINAMATH_CALUDE_airline_passengers_l1379_137968

/-- Given an airline where each passenger can take 8 pieces of luggage,
    and a total of 32 bags, prove that 4 people were flying. -/
theorem airline_passengers (bags_per_person : ℕ) (total_bags : ℕ) (num_people : ℕ) : 
  bags_per_person = 8 →
  total_bags = 32 →
  num_people * bags_per_person = total_bags →
  num_people = 4 := by
sorry

end NUMINAMATH_CALUDE_airline_passengers_l1379_137968


namespace NUMINAMATH_CALUDE_profit_discount_rate_l1379_137977

/-- Proves that a 20% profit on a product with a purchase price of 200 yuan and a marked price of 300 yuan is achieved by selling at 80% of the marked price. -/
theorem profit_discount_rate (purchase_price marked_price : ℝ) 
  (h_purchase : purchase_price = 200)
  (h_marked : marked_price = 300)
  (profit_rate : ℝ) (h_profit : profit_rate = 0.2)
  (discount_rate : ℝ) :
  discount_rate * marked_price = purchase_price * (1 + profit_rate) →
  discount_rate = 0.8 := by
sorry

end NUMINAMATH_CALUDE_profit_discount_rate_l1379_137977


namespace NUMINAMATH_CALUDE_possible_values_of_a_minus_b_l1379_137946

theorem possible_values_of_a_minus_b (a b : ℝ) 
  (ha : |a| = 8) 
  (hb : |b| = 6) 
  (hab : |a + b| = a + b) : 
  a - b = 2 ∨ a - b = 14 := by
sorry

end NUMINAMATH_CALUDE_possible_values_of_a_minus_b_l1379_137946


namespace NUMINAMATH_CALUDE_power_mod_remainder_l1379_137911

theorem power_mod_remainder : 6^50 % 215 = 36 := by sorry

end NUMINAMATH_CALUDE_power_mod_remainder_l1379_137911


namespace NUMINAMATH_CALUDE_final_cost_is_33_08_l1379_137974

/-- The cost of a single deck in dollars -/
def deck_cost : ℚ := 7

/-- The discount rate as a decimal -/
def discount_rate : ℚ := 0.1

/-- The sales tax rate as a decimal -/
def sales_tax_rate : ℚ := 0.05

/-- The number of decks Frank bought -/
def frank_decks : ℕ := 3

/-- The number of decks Frank's friend bought -/
def friend_decks : ℕ := 2

/-- The total cost before discount and tax -/
def total_cost : ℚ := deck_cost * (frank_decks + friend_decks)

/-- The discounted cost -/
def discounted_cost : ℚ := total_cost * (1 - discount_rate)

/-- The final cost including tax -/
def final_cost : ℚ := discounted_cost * (1 + sales_tax_rate)

/-- Theorem stating the final cost is $33.08 -/
theorem final_cost_is_33_08 : 
  ∃ (ε : ℚ), abs (final_cost - 33.08) < ε ∧ ε = 0.005 := by
  sorry

end NUMINAMATH_CALUDE_final_cost_is_33_08_l1379_137974


namespace NUMINAMATH_CALUDE_quadratic_root_implies_k_l1379_137987

theorem quadratic_root_implies_k (k : ℝ) : 
  ((k - 1) * 1^2 + k^2 - k = 0) → 
  (k - 1 ≠ 0) → 
  (k = -1) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_root_implies_k_l1379_137987


namespace NUMINAMATH_CALUDE_john_total_spend_l1379_137910

def calculate_total_spend (tshirt_price : ℝ) (tshirt_count : ℕ) (pants_price : ℝ) (pants_count : ℕ)
  (jacket_price : ℝ) (jacket_discount : ℝ) (hat_price : ℝ) (shoes_price : ℝ) (shoes_discount : ℝ)
  (clothes_tax_rate : ℝ) (shoes_tax_rate : ℝ) : ℝ :=
  let tshirt_total := tshirt_price * 2 + tshirt_price * 0.5
  let pants_total := pants_price * pants_count
  let jacket_total := jacket_price * (1 - jacket_discount)
  let shoes_total := shoes_price * (1 - shoes_discount)
  let clothes_subtotal := tshirt_total + pants_total + jacket_total + hat_price
  let total_before_tax := clothes_subtotal + shoes_total
  let clothes_tax := clothes_subtotal * clothes_tax_rate
  let shoes_tax := shoes_total * shoes_tax_rate
  total_before_tax + clothes_tax + shoes_tax

theorem john_total_spend :
  calculate_total_spend 20 3 50 2 80 0.25 15 60 0.1 0.05 0.08 = 294.57 := by
  sorry

end NUMINAMATH_CALUDE_john_total_spend_l1379_137910


namespace NUMINAMATH_CALUDE_total_spending_is_correct_l1379_137942

def lunch_cost : ℚ := 50.50
def dessert_cost : ℚ := 8.25
def beverage_cost : ℚ := 3.75
def lunch_discount : ℚ := 0.10
def dessert_tax : ℚ := 0.07
def beverage_tax : ℚ := 0.05
def lunch_tip : ℚ := 0.20
def other_tip : ℚ := 0.15

def discounted_lunch : ℚ := lunch_cost * (1 - lunch_discount)
def taxed_dessert : ℚ := dessert_cost * (1 + dessert_tax)
def taxed_beverage : ℚ := beverage_cost * (1 + beverage_tax)

def lunch_tip_amount : ℚ := discounted_lunch * lunch_tip
def other_tip_amount : ℚ := (taxed_dessert + taxed_beverage) * other_tip

def total_spending : ℚ := discounted_lunch + taxed_dessert + taxed_beverage + lunch_tip_amount + other_tip_amount

theorem total_spending_is_correct : total_spending = 69.23 := by sorry

end NUMINAMATH_CALUDE_total_spending_is_correct_l1379_137942


namespace NUMINAMATH_CALUDE_solution_set_a_2_range_of_a_l1379_137950

-- Define the function f
def f (x a : ℝ) : ℝ := |x - a^2| + |x - 2*a + 1|

-- Part 1: Solution set when a = 2
theorem solution_set_a_2 :
  {x : ℝ | f x 2 ≥ 4} = {x : ℝ | x ≤ 3/2 ∨ x ≥ 11/2} :=
sorry

-- Part 2: Range of a
theorem range_of_a :
  {a : ℝ | ∃ x, f x a ≥ 4} = {a : ℝ | a ≤ -1 ∨ a ≥ 3} :=
sorry

end NUMINAMATH_CALUDE_solution_set_a_2_range_of_a_l1379_137950


namespace NUMINAMATH_CALUDE_brand_preference_ratio_l1379_137967

theorem brand_preference_ratio (total_respondents : ℕ) (brand_x_preference : ℕ) 
  (h1 : total_respondents = 80)
  (h2 : brand_x_preference = 60)
  (h3 : brand_x_preference < total_respondents) :
  (brand_x_preference : ℚ) / (total_respondents - brand_x_preference : ℚ) = 3 / 1 := by
  sorry

end NUMINAMATH_CALUDE_brand_preference_ratio_l1379_137967


namespace NUMINAMATH_CALUDE_journey_distance_l1379_137923

/-- Prove that given a journey with specified conditions, the total distance traveled is 270 km. -/
theorem journey_distance (total_time : ℝ) (speed1 : ℝ) (speed2 : ℝ) : 
  total_time = 15 →
  speed1 = 45 →
  speed2 = 30 →
  (total_time * speed1 * speed2) / (speed1 + speed2) = 270 := by
  sorry

#check journey_distance

end NUMINAMATH_CALUDE_journey_distance_l1379_137923


namespace NUMINAMATH_CALUDE_price_increase_percentage_l1379_137961

theorem price_increase_percentage (old_price new_price : ℝ) 
  (h1 : old_price = 300)
  (h2 : new_price = 360) :
  (new_price - old_price) / old_price * 100 = 20 := by
  sorry

end NUMINAMATH_CALUDE_price_increase_percentage_l1379_137961


namespace NUMINAMATH_CALUDE_remainder_problem_l1379_137920

theorem remainder_problem (x R : ℤ) : 
  (∃ k : ℤ, x = 82 * k + R) → 
  (∃ m : ℤ, x + 17 = 41 * m + 22) → 
  R = 5 := by
sorry

end NUMINAMATH_CALUDE_remainder_problem_l1379_137920


namespace NUMINAMATH_CALUDE_manager_percentage_reduction_l1379_137990

theorem manager_percentage_reduction (total_employees : ℕ) (initial_percentage : ℚ) 
  (managers_leaving : ℕ) (target_percentage : ℚ) : 
  total_employees = 600 →
  initial_percentage = 99 / 100 →
  managers_leaving = 300 →
  target_percentage = 49 / 100 →
  (total_employees * initial_percentage - managers_leaving) / total_employees = target_percentage := by
  sorry

end NUMINAMATH_CALUDE_manager_percentage_reduction_l1379_137990


namespace NUMINAMATH_CALUDE_interview_probability_l1379_137931

theorem interview_probability (total_students : ℕ) (french_students : ℕ) (spanish_students : ℕ) (german_students : ℕ)
  (h1 : total_students = 30)
  (h2 : french_students = 22)
  (h3 : spanish_students = 25)
  (h4 : german_students = 5)
  (h5 : french_students ≤ spanish_students)
  (h6 : spanish_students ≤ total_students)
  (h7 : german_students ≤ total_students) :
  let non_french_spanish : ℕ := total_students - spanish_students
  let total_combinations : ℕ := total_students.choose 2
  let non_informative_combinations : ℕ := (non_french_spanish + (spanish_students - french_students)).choose 2
  (1 : ℚ) - (non_informative_combinations : ℚ) / (total_combinations : ℚ) = 407 / 435 := by
    sorry

end NUMINAMATH_CALUDE_interview_probability_l1379_137931


namespace NUMINAMATH_CALUDE_negation_equivalence_l1379_137984

theorem negation_equivalence (f g : ℝ → ℝ) :
  (¬ ∃ x : ℝ, f x * g x = 0) ↔ (∀ x : ℝ, f x ≠ 0 ∧ g x ≠ 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_equivalence_l1379_137984


namespace NUMINAMATH_CALUDE_solution_value_l1379_137985

theorem solution_value (x y a : ℝ) : 
  x = 1 → y = -3 → a * x - y = 1 → a = -2 := by
  sorry

end NUMINAMATH_CALUDE_solution_value_l1379_137985


namespace NUMINAMATH_CALUDE_difference_three_fifths_l1379_137935

theorem difference_three_fifths (x : ℝ) : x - (3/5) * x = 145 → x = 362.5 := by
  sorry

end NUMINAMATH_CALUDE_difference_three_fifths_l1379_137935


namespace NUMINAMATH_CALUDE_mikes_training_hours_l1379_137918

/-- Represents a day of the week -/
inductive Day
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

/-- Represents a week number -/
inductive Week
  | First
  | Second

/-- Returns true if the given day is a weekday, false otherwise -/
def isWeekday (d : Day) : Bool :=
  match d with
  | Day.Saturday | Day.Sunday => false
  | _ => true

/-- Returns the maximum training hours for a given day and week -/
def maxTrainingHours (d : Day) (w : Week) : Nat :=
  match w with
  | Week.First => if isWeekday d then 2 else 1
  | Week.Second => if isWeekday d then 3 else 2

/-- Returns true if the given day is a rest day, false otherwise -/
def isRestDay (dayNumber : Nat) : Bool :=
  dayNumber % 5 == 0

/-- Calculates the total training hours for Mike over two weeks -/
def totalTrainingHours : Nat :=
  let firstWeekHours := 12  -- 5 weekdays * 2 hours + 2 weekend days * 1 hour
  let secondWeekHours := 16 -- 4 weekdays * 3 hours + 2 weekend days * 2 hours (1 rest day)
  firstWeekHours + secondWeekHours

/-- Theorem stating that Mike's total training hours over two weeks is 28 -/
theorem mikes_training_hours : totalTrainingHours = 28 := by
  sorry

#eval totalTrainingHours  -- This should output 28

end NUMINAMATH_CALUDE_mikes_training_hours_l1379_137918


namespace NUMINAMATH_CALUDE_highest_score_can_be_less_than_15_l1379_137997

/-- Represents a team in the tournament -/
structure Team :=
  (score : ℕ)

/-- Represents the tournament -/
structure Tournament :=
  (teams : Finset Team)
  (num_teams : ℕ)
  (total_games : ℕ)
  (total_points : ℕ)

/-- The tournament satisfies the given conditions -/
def valid_tournament (t : Tournament) : Prop :=
  t.num_teams = 10 ∧
  t.total_games = t.num_teams * (t.num_teams - 1) / 2 ∧
  t.total_points = 3 * t.total_games ∧
  t.teams.card = t.num_teams

/-- The theorem to be proved -/
theorem highest_score_can_be_less_than_15 :
  ∃ (t : Tournament), valid_tournament t ∧
    (∀ team ∈ t.teams, team.score < 15) :=
  sorry

end NUMINAMATH_CALUDE_highest_score_can_be_less_than_15_l1379_137997


namespace NUMINAMATH_CALUDE_no_solutions_in_interval_l1379_137975

theorem no_solutions_in_interval (x : Real) : 
  x ∈ Set.Icc 0 (2 * Real.pi) → 
  (1 / Real.sin x + 1 / Real.cos x ≠ 4) :=
by sorry

end NUMINAMATH_CALUDE_no_solutions_in_interval_l1379_137975


namespace NUMINAMATH_CALUDE_no_roots_in_larger_interval_l1379_137951

-- Define a real-valued function
variable (f : ℝ → ℝ)

-- Define the property of having exactly one root
def has_unique_root (f : ℝ → ℝ) : Prop :=
  ∃! x, f x = 0

-- Define the property of a root being within an open interval
def root_in_interval (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∃ x, a < x ∧ x < b ∧ f x = 0

-- Theorem statement
theorem no_roots_in_larger_interval
  (h_unique : has_unique_root f)
  (h_16 : root_in_interval f 0 16)
  (h_8 : root_in_interval f 0 8)
  (h_4 : root_in_interval f 0 4)
  (h_2 : root_in_interval f 0 2) :
  ∀ x ∈ Set.Icc 2 16, f x ≠ 0 :=
sorry

end NUMINAMATH_CALUDE_no_roots_in_larger_interval_l1379_137951


namespace NUMINAMATH_CALUDE_largest_multiple_of_15_under_500_l1379_137921

theorem largest_multiple_of_15_under_500 : 
  ∀ n : ℕ, n * 15 < 500 → n * 15 ≤ 495 :=
sorry

end NUMINAMATH_CALUDE_largest_multiple_of_15_under_500_l1379_137921


namespace NUMINAMATH_CALUDE_condition_D_iff_right_triangle_l1379_137905

/-- Triangle ABC with sides a, b, c opposite to angles A, B, C respectively -/
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : Real
  B : Real
  C : Real

/-- Definition of a right triangle -/
def is_right_triangle (t : Triangle) : Prop :=
  t.A = Real.pi / 2 ∨ t.B = Real.pi / 2 ∨ t.C = Real.pi / 2

/-- The condition a² = b² - c² -/
def condition_D (t : Triangle) : Prop :=
  t.a^2 = t.b^2 - t.c^2

/-- Theorem stating that condition D is equivalent to the triangle being a right triangle -/
theorem condition_D_iff_right_triangle (t : Triangle) :
  condition_D t ↔ is_right_triangle t :=
sorry

end NUMINAMATH_CALUDE_condition_D_iff_right_triangle_l1379_137905


namespace NUMINAMATH_CALUDE_utensils_per_pack_l1379_137980

/-- Given that packs have an equal number of knives, forks, and spoons,
    and 5 packs contain 50 spoons, prove that each pack contains 30 utensils. -/
theorem utensils_per_pack (total_packs : ℕ) (total_spoons : ℕ) 
  (h1 : total_packs = 5)
  (h2 : total_spoons = 50) :
  let spoons_per_pack := total_spoons / total_packs
  let utensils_per_pack := 3 * spoons_per_pack
  utensils_per_pack = 30 := by
sorry

end NUMINAMATH_CALUDE_utensils_per_pack_l1379_137980


namespace NUMINAMATH_CALUDE_roxanne_change_l1379_137991

def lemonade_price : ℚ := 2
def lemonade_quantity : ℕ := 4

def sandwich_price : ℚ := 2.5
def sandwich_quantity : ℕ := 3

def watermelon_price : ℚ := 1.25
def watermelon_quantity : ℕ := 2

def chips_price : ℚ := 1.75
def chips_quantity : ℕ := 1

def cookie_price : ℚ := 0.75
def cookie_quantity : ℕ := 4

def pretzel_price : ℚ := 1
def pretzel_quantity : ℕ := 5

def salad_price : ℚ := 8
def salad_quantity : ℕ := 1

def payment : ℚ := 100

theorem roxanne_change :
  payment - (lemonade_price * lemonade_quantity +
             sandwich_price * sandwich_quantity +
             watermelon_price * watermelon_quantity +
             chips_price * chips_quantity +
             cookie_price * cookie_quantity +
             pretzel_price * pretzel_quantity +
             salad_price * salad_quantity) = 63.75 := by
  sorry

end NUMINAMATH_CALUDE_roxanne_change_l1379_137991


namespace NUMINAMATH_CALUDE_sequence_gcd_is_one_l1379_137957

theorem sequence_gcd_is_one (n : ℕ+) : 
  let a : ℕ+ → ℕ := fun k => 100 + 2 * k^2
  Nat.gcd (a n) (a (n + 1)) = 1 := by
sorry

end NUMINAMATH_CALUDE_sequence_gcd_is_one_l1379_137957


namespace NUMINAMATH_CALUDE_geometric_sequence_solution_l1379_137936

theorem geometric_sequence_solution (x : ℝ) :
  (1 < x ∧ x < 9 ∧ x^2 = 9) ↔ (x = 3 ∨ x = -3) := by sorry

end NUMINAMATH_CALUDE_geometric_sequence_solution_l1379_137936


namespace NUMINAMATH_CALUDE_sum_of_coefficients_zero_l1379_137986

theorem sum_of_coefficients_zero 
  (a b c d : ℝ) 
  (h : ∀ x : ℝ, (1 + x)^2 * (1 - x) = a + b*x + c*x^2 + d*x^3) : 
  a + b + c + d = 0 := by
sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_zero_l1379_137986


namespace NUMINAMATH_CALUDE_charge_with_interest_after_one_year_l1379_137941

/-- Calculates the amount owed after one year given an initial charge and simple annual interest rate -/
def amount_owed_after_one_year (initial_charge : ℝ) (interest_rate : ℝ) : ℝ :=
  initial_charge * (1 + interest_rate)

/-- Theorem stating that a $35 charge with 7% simple annual interest results in $37.45 owed after one year -/
theorem charge_with_interest_after_one_year :
  let initial_charge : ℝ := 35
  let interest_rate : ℝ := 0.07
  amount_owed_after_one_year initial_charge interest_rate = 37.45 := by
  sorry

#eval amount_owed_after_one_year 35 0.07

end NUMINAMATH_CALUDE_charge_with_interest_after_one_year_l1379_137941


namespace NUMINAMATH_CALUDE_triangle_problem_l1379_137915

-- Define the triangle ABC
structure Triangle :=
  (A B C : ℝ)  -- Angles
  (a b c : ℝ)  -- Sides opposite to angles A, B, C respectively

-- Define the given conditions
def given_conditions (t : Triangle) : Prop :=
  (1 + Real.sin t.B + Real.cos t.B) * (Real.cos (t.B / 2) - Real.sin (t.B / 2)) = 
    7 / 12 * Real.sqrt (2 + 2 * Real.cos t.B) ∧
  t.c / t.a = 2 / 3

-- Define point D on side AC such that BD = AC
def point_D (t : Triangle) (D : ℝ) : Prop :=
  0 < D ∧ D < t.c ∧ Real.sqrt (t.a^2 + D^2 - 2 * t.a * D * Real.cos t.A) = t.c

-- State the theorem
theorem triangle_problem (t : Triangle) (D : ℝ) :
  given_conditions t → point_D t D →
  Real.cos t.B = 7 / 12 ∧ D / (t.c - D) = 2 :=
by sorry

end NUMINAMATH_CALUDE_triangle_problem_l1379_137915


namespace NUMINAMATH_CALUDE_winston_remaining_cents_l1379_137958

/-- The number of cents in a quarter -/
def cents_per_quarter : ℕ := 25

/-- The number of cents in half a dollar -/
def half_dollar_cents : ℕ := 50

/-- The number of quarters Winston has -/
def winston_quarters : ℕ := 14

theorem winston_remaining_cents : 
  winston_quarters * cents_per_quarter - half_dollar_cents = 300 := by
  sorry

end NUMINAMATH_CALUDE_winston_remaining_cents_l1379_137958


namespace NUMINAMATH_CALUDE_ladybug_dots_average_l1379_137937

/-- The number of ladybugs caught on Monday -/
def monday_ladybugs : ℕ := 8

/-- The number of ladybugs caught on Tuesday -/
def tuesday_ladybugs : ℕ := 5

/-- The total number of dots on all ladybugs -/
def total_dots : ℕ := 78

/-- The average number of dots per ladybug -/
def average_dots : ℚ := total_dots / (monday_ladybugs + tuesday_ladybugs)

theorem ladybug_dots_average :
  average_dots = 6 := by sorry

end NUMINAMATH_CALUDE_ladybug_dots_average_l1379_137937


namespace NUMINAMATH_CALUDE_expression_simplification_l1379_137929

theorem expression_simplification (x : ℝ) 
  (h1 : x ≠ 2) (h2 : x ≠ 3) (h3 : x ≠ 4) (h4 : x ≠ 5) : 
  ((x^2 - 4*x + 3) / (x^2 - 6*x + 9)) / ((x^2 - 6*x + 8) / (x^2 - 8*x + 15)) = 
  (x - 1)*(x - 5) / ((x - 2)*(x - 4)) := by sorry

end NUMINAMATH_CALUDE_expression_simplification_l1379_137929


namespace NUMINAMATH_CALUDE_sin_minus_cos_sqrt_two_l1379_137932

theorem sin_minus_cos_sqrt_two (x : Real) :
  0 ≤ x ∧ x < 2 * Real.pi →
  Real.sin x - Real.cos x = Real.sqrt 2 →
  x = 3 * Real.pi / 4 := by
sorry

end NUMINAMATH_CALUDE_sin_minus_cos_sqrt_two_l1379_137932


namespace NUMINAMATH_CALUDE_geometric_sequence_problem_l1379_137934

-- Define a geometric sequence
def geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

-- State the theorem
theorem geometric_sequence_problem (a : ℕ → ℝ) :
  geometric_sequence a →
  a 1 = 2 →
  a 3 * a 5 = 4 * (a 6)^2 →
  a 3 = 1/2 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_problem_l1379_137934


namespace NUMINAMATH_CALUDE_area_trapezoid_EFBA_l1379_137912

/-- Rectangle ABCD with points E and F on side DC -/
structure RectangleWithPoints where
  /-- Length of side AB -/
  AB : ℝ
  /-- Length of side BC -/
  BC : ℝ
  /-- Length of segment DE -/
  DE : ℝ
  /-- Length of segment FC -/
  FC : ℝ
  /-- Area of rectangle ABCD -/
  area_ABCD : ℝ
  /-- AB is positive -/
  AB_pos : AB > 0
  /-- BC is positive -/
  BC_pos : BC > 0
  /-- DE is positive -/
  DE_pos : DE > 0
  /-- FC is positive -/
  FC_pos : FC > 0
  /-- Area of ABCD is product of AB and BC -/
  area_eq : area_ABCD = AB * BC
  /-- DE + EF + FC = DC = AB -/
  side_sum : DE + (AB - DE - FC) + FC = AB

/-- The area of trapezoid EFBA is 14 square units -/
theorem area_trapezoid_EFBA (r : RectangleWithPoints) (h1 : r.AB = 10) (h2 : r.BC = 2) 
    (h3 : r.DE = 2) (h4 : r.FC = 4) (h5 : r.area_ABCD = 20) : 
    r.AB * r.BC - r.DE * r.BC / 2 - r.FC * r.BC / 2 = 14 := by
  sorry

end NUMINAMATH_CALUDE_area_trapezoid_EFBA_l1379_137912


namespace NUMINAMATH_CALUDE_batsman_average_after_12th_innings_l1379_137940

/-- Calculates the new average of a batsman after 12 innings -/
def new_average (previous_total : ℕ) (new_score : ℕ) : ℚ :=
  (previous_total + new_score) / 12

/-- Represents the increase in average after the 12th innings -/
def average_increase (previous_average : ℚ) (new_average : ℚ) : ℚ :=
  new_average - previous_average

theorem batsman_average_after_12th_innings 
  (previous_total : ℕ) 
  (previous_average : ℚ) 
  (new_score : ℕ) :
  previous_total = previous_average * 11 →
  new_score = 115 →
  average_increase previous_average (new_average previous_total new_score) = 3 →
  new_average previous_total new_score = 82 :=
by sorry

end NUMINAMATH_CALUDE_batsman_average_after_12th_innings_l1379_137940


namespace NUMINAMATH_CALUDE_brick_wall_theorem_l1379_137954

/-- Calculates the total number of bricks in a wall with a given number of rows,
    where each row has one less brick than the row below it. -/
def total_bricks (rows : ℕ) (bottom_row_bricks : ℕ) : ℕ :=
  (2 * bottom_row_bricks - rows + 1) * rows / 2

/-- Theorem stating that a wall with 5 rows, 18 bricks in the bottom row,
    and each row having one less brick than the row below it,
    has a total of 80 bricks. -/
theorem brick_wall_theorem :
  total_bricks 5 18 = 80 := by
  sorry

end NUMINAMATH_CALUDE_brick_wall_theorem_l1379_137954


namespace NUMINAMATH_CALUDE_cone_height_ratio_l1379_137953

/-- Proves the ratio of new height to original height for a cone with reduced height -/
theorem cone_height_ratio (base_circumference : ℝ) (original_height : ℝ) (new_volume : ℝ) :
  base_circumference = 20 * Real.pi →
  original_height = 40 →
  new_volume = 400 * Real.pi →
  ∃ (new_height : ℝ),
    (1 / 3) * Real.pi * ((base_circumference / (2 * Real.pi)) ^ 2) * new_height = new_volume ∧
    new_height / original_height = 3 / 10 := by
  sorry


end NUMINAMATH_CALUDE_cone_height_ratio_l1379_137953


namespace NUMINAMATH_CALUDE_cityD_highest_increase_l1379_137998

structure City where
  name : String
  population1990 : ℕ
  population2000 : ℕ

def percentageIncrease (city : City) : ℚ :=
  (city.population2000 : ℚ) / (city.population1990 : ℚ)

def cityA : City := ⟨"A", 45, 60⟩
def cityB : City := ⟨"B", 65, 85⟩
def cityC : City := ⟨"C", 90, 120⟩
def cityD : City := ⟨"D", 115, 160⟩
def cityE : City := ⟨"E", 150, 200⟩
def cityF : City := ⟨"F", 130, 180⟩

def cities : List City := [cityA, cityB, cityC, cityD, cityE, cityF]

theorem cityD_highest_increase :
  ∀ city ∈ cities, percentageIncrease cityD ≥ percentageIncrease city :=
by sorry

end NUMINAMATH_CALUDE_cityD_highest_increase_l1379_137998


namespace NUMINAMATH_CALUDE_unique_satisfying_function_l1379_137913

/-- A function satisfying the given functional equations -/
def SatisfyingFunction (f : ℝ → ℝ) : Prop :=
  (∀ x, f (Real.sqrt 3 / 3 * x) = Real.sqrt 3 * f x - 2 * Real.sqrt 3 / 3 * x) ∧
  (∀ x y, y ≠ 0 → f x * f y = f (x * y) + f (x / y))

/-- The theorem stating that x + 1/x is the only function satisfying the given equations -/
theorem unique_satisfying_function :
  ∀ f : ℝ → ℝ, SatisfyingFunction f ↔ ∀ x, f x = x + 1/x :=
sorry

end NUMINAMATH_CALUDE_unique_satisfying_function_l1379_137913


namespace NUMINAMATH_CALUDE_smallest_with_20_divisors_l1379_137956

def num_divisors (n : ℕ+) : ℕ := (Nat.divisors n.val).card

theorem smallest_with_20_divisors : 
  ∃ (n : ℕ+), num_divisors n = 20 ∧ ∀ (m : ℕ+), m < n → num_divisors m ≠ 20 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_smallest_with_20_divisors_l1379_137956
