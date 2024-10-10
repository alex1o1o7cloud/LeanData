import Mathlib

namespace inequality_theorem_l354_35431

theorem inequality_theorem (a b : ℝ) : 
  (a * b > 0 → b / a + a / b ≥ 2) ∧ 
  (a + 2 * b = 1 → 3^a + 9^b ≥ 2 * Real.sqrt 3) := by
  sorry

end inequality_theorem_l354_35431


namespace circle_symmetry_line_l354_35448

/-- If a circle with equation (x-1)^2 + (y-2)^2 = 1 is symmetric about the line y = x + b, then b = 1 -/
theorem circle_symmetry_line (b : ℝ) : 
  (∀ x y : ℝ, (x - 1)^2 + (y - 2)^2 = 1 ↔ (x - 1)^2 + ((x + b) - 2)^2 = 1) → 
  b = 1 := by
  sorry

end circle_symmetry_line_l354_35448


namespace jar_water_problem_l354_35493

theorem jar_water_problem (s l w : ℝ) (hs : s > 0) (hl : l > 0) (hw : w > 0)
  (h1 : w = l / 2)  -- Larger jar is 1/2 full
  (h2 : w + w = 2 * l / 3)  -- When combined, 2/3 of larger jar is filled
  (h3 : s < l)  -- Smaller jar has less capacity
  : w = 3 * s / 4  -- Smaller jar was 3/4 full
  := by sorry

end jar_water_problem_l354_35493


namespace tangent_line_slope_range_l354_35466

open Real Set

noncomputable def f (x : ℝ) : ℝ := Real.sqrt 3 * Real.sin (2 * x) - Real.cos (2 * x)

theorem tangent_line_slope_range :
  let symmetry_axis : ℝ := π / 3
  let tangent_line (m c : ℝ) (x y : ℝ) : Prop := x + m * y + c = 0
  ∃ (c : ℝ), ∃ (x : ℝ), tangent_line m c x (f x) ↔ 
    m ∈ Iic (-1/4) ∪ Ici (1/4) :=
sorry

end tangent_line_slope_range_l354_35466


namespace division_equation_solution_l354_35434

theorem division_equation_solution :
  ∃ x : ℝ, (0.009 / x = 0.1) ∧ (x = 0.09) := by
  sorry

end division_equation_solution_l354_35434


namespace student_rank_from_left_l354_35455

/-- Given a total number of students and a student's rank from the right,
    calculate the student's rank from the left. -/
def rankFromLeft (totalStudents : ℕ) (rankFromRight : ℕ) : ℕ :=
  totalStudents - rankFromRight + 1

/-- Theorem: Given 20 students in total and a student ranked 13th from the right,
    prove that the student's rank from the left is 8th. -/
theorem student_rank_from_left :
  let totalStudents : ℕ := 20
  let rankFromRight : ℕ := 13
  rankFromLeft totalStudents rankFromRight = 8 := by
  sorry

end student_rank_from_left_l354_35455


namespace product_modulo_25_l354_35467

theorem product_modulo_25 :
  ∃ m : ℕ, 0 ≤ m ∧ m < 25 ∧ (68 * 95 * 113) % 25 = m ∧ m = 5 := by
  sorry

end product_modulo_25_l354_35467


namespace sum_of_odd_numbers_l354_35441

theorem sum_of_odd_numbers (N : ℕ) : 
  1001 + 1003 + 1005 + 1007 + 1009 = 5100 - N → N = 75 := by
  sorry

end sum_of_odd_numbers_l354_35441


namespace product_a_b_equals_27_over_8_l354_35490

theorem product_a_b_equals_27_over_8 
  (a b c : ℝ) 
  (h1 : 0 < a ∧ 0 < b ∧ 0 < c) 
  (h2 : c = 3 → a = b^2) 
  (h3 : b + c = 2*a) 
  (h4 : c = 3) 
  (h5 : b + c = b * c) : 
  a * b = 27/8 := by
sorry

end product_a_b_equals_27_over_8_l354_35490


namespace third_circle_radius_l354_35496

theorem third_circle_radius (r₁ r₂ r₃ : ℝ) : 
  r₁ = 25 → r₂ = 40 → 
  π * r₃^2 = (π * r₂^2 - π * r₁^2) / 2 →
  r₃ = 15 * Real.sqrt 13 :=
by sorry

end third_circle_radius_l354_35496


namespace overall_profit_calculation_l354_35485

/-- Calculate the overall profit from selling a refrigerator and a mobile phone -/
theorem overall_profit_calculation (refrigerator_cost mobile_cost : ℕ) 
  (refrigerator_loss_percent mobile_profit_percent : ℚ) :
  refrigerator_cost = 15000 →
  mobile_cost = 8000 →
  refrigerator_loss_percent = 5 / 100 →
  mobile_profit_percent = 10 / 100 →
  (refrigerator_cost * (1 - refrigerator_loss_percent) + 
   mobile_cost * (1 + mobile_profit_percent)) - 
  (refrigerator_cost + mobile_cost) = 50 := by
  sorry

end overall_profit_calculation_l354_35485


namespace vector_problem_l354_35449

/-- Given vectors in ℝ² -/
def a : ℝ × ℝ := (4, 2)
def b : ℝ × ℝ := (-1, 2)
def c (m : ℝ) : ℝ × ℝ := (2, m)

/-- Dot product of two 2D vectors -/
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

/-- Two vectors are parallel if one is a scalar multiple of the other -/
def parallel (v w : ℝ × ℝ) : Prop :=
  ∃ (k : ℝ), v.1 = k * w.1 ∧ v.2 = k * w.2

theorem vector_problem (m : ℝ) :
  (dot_product a (c m) < m^2 → m > 4 ∨ m < -2) ∧
  (parallel (a.1 + (c m).1, a.2 + (c m).2) b → m = -14) :=
sorry

end vector_problem_l354_35449


namespace equation_solutions_l354_35483

theorem equation_solutions :
  (∀ x : ℝ, (x - 1)^2 = 4 ↔ x = 3 ∨ x = -1) ∧
  (∀ x : ℝ, 2*x^3 = -16 ↔ x = -2) := by
  sorry

end equation_solutions_l354_35483


namespace square_floor_tiles_l354_35425

/-- Represents a square floor tiled with congruent square tiles. -/
structure SquareFloor where
  side_length : ℕ
  is_valid : side_length > 0

/-- Calculates the number of black tiles on the boundary of the floor. -/
def black_tiles (floor : SquareFloor) : ℕ :=
  4 * floor.side_length - 4

/-- Calculates the total number of tiles on the floor. -/
def total_tiles (floor : SquareFloor) : ℕ :=
  floor.side_length ^ 2

/-- Theorem stating that a square floor with 100 black boundary tiles has 676 total tiles. -/
theorem square_floor_tiles (floor : SquareFloor) :
  black_tiles floor = 100 → total_tiles floor = 676 := by
  sorry

end square_floor_tiles_l354_35425


namespace luncheon_tables_l354_35498

theorem luncheon_tables (invited : ℕ) (no_show : ℕ) (per_table : ℕ) 
  (h1 : invited = 18) 
  (h2 : no_show = 12) 
  (h3 : per_table = 3) : 
  (invited - no_show) / per_table = 2 := by
  sorry

end luncheon_tables_l354_35498


namespace prob_spade_heart_king_l354_35452

/-- Represents a standard 52-card deck -/
def StandardDeck : ℕ := 52

/-- Number of spades in a standard deck -/
def NumSpades : ℕ := 13

/-- Number of hearts in a standard deck -/
def NumHearts : ℕ := 13

/-- Number of Kings in a standard deck -/
def NumKings : ℕ := 4

/-- Probability of drawing a spade, then a heart, then a King from a standard 52-card deck -/
theorem prob_spade_heart_king :
  (NumSpades * NumHearts * NumKings) / (StandardDeck * (StandardDeck - 1) * (StandardDeck - 2)) = 17 / 3683 := by
  sorry


end prob_spade_heart_king_l354_35452


namespace proposition_q_false_l354_35486

theorem proposition_q_false (h1 : ¬(p ∧ q)) (h2 : ¬(¬p)) : ¬q := by
  sorry

end proposition_q_false_l354_35486


namespace relay_race_time_reduction_l354_35491

theorem relay_race_time_reduction (T T₁ T₂ T₃ T₄ T₅ : ℝ) 
  (total_time : T = T₁ + T₂ + T₃ + T₄ + T₅)
  (first_runner : T₁ / 2 + T₂ + T₃ + T₄ + T₅ = 0.95 * T)
  (second_runner : T₁ + T₂ / 2 + T₃ + T₄ + T₅ = 0.90 * T)
  (third_runner : T₁ + T₂ + T₃ / 2 + T₄ + T₅ = 0.88 * T)
  (fourth_runner : T₁ + T₂ + T₃ + T₄ / 2 + T₅ = 0.85 * T) :
  T₁ + T₂ + T₃ + T₄ + T₅ / 2 = 0.92 * T := by
  sorry

end relay_race_time_reduction_l354_35491


namespace simplify_fraction_l354_35468

theorem simplify_fraction (x : ℝ) (h : x = 2) : 15 * x^5 / (45 * x^3) = 4/3 := by
  sorry

end simplify_fraction_l354_35468


namespace fraction_product_square_l354_35411

theorem fraction_product_square : (8 / 9) ^ 2 * (1 / 3) ^ 2 = 64 / 729 := by
  sorry

end fraction_product_square_l354_35411


namespace third_fraction_is_two_ninths_l354_35476

-- Define a fraction type
structure Fraction where
  numerator : ℤ
  denominator : ℕ
  denominator_nonzero : denominator ≠ 0

-- Define the HCF function for fractions
def hcf_fractions (f1 f2 f3 : Fraction) : ℚ :=
  sorry

-- Theorem statement
theorem third_fraction_is_two_ninths
  (f1 : Fraction)
  (f2 : Fraction)
  (f3 : Fraction)
  (h1 : f1 = ⟨2, 3, sorry⟩)
  (h2 : f2 = ⟨4, 9, sorry⟩)
  (h3 : hcf_fractions f1 f2 f3 = 1 / 9) :
  f3 = ⟨2, 9, sorry⟩ :=
sorry

end third_fraction_is_two_ninths_l354_35476


namespace gcf_factorial_seven_eight_l354_35408

theorem gcf_factorial_seven_eight : Nat.gcd (Nat.factorial 7) (Nat.factorial 8) = Nat.factorial 7 := by
  sorry

end gcf_factorial_seven_eight_l354_35408


namespace quadratic_root_implies_a_l354_35497

theorem quadratic_root_implies_a (a : ℝ) :
  (∃ x : ℝ, x^2 - a = 0) ∧ (2^2 - a = 0) → a = 4 := by
  sorry

end quadratic_root_implies_a_l354_35497


namespace bus_seating_capacity_l354_35432

theorem bus_seating_capacity :
  let left_seats : ℕ := 15
  let right_seats : ℕ := left_seats - 3
  let people_per_seat : ℕ := 3
  let back_seat_capacity : ℕ := 7
  let total_capacity : ℕ := left_seats * people_per_seat + right_seats * people_per_seat + back_seat_capacity
  total_capacity = 88 := by
  sorry

end bus_seating_capacity_l354_35432


namespace max_value_of_f_l354_35445

-- Define the function f
def f (x : ℝ) : ℝ := 3 * x^3 - 18 * x^2 + 27 * x

-- State the theorem
theorem max_value_of_f :
  ∃ (M : ℝ), M = 12 ∧ ∀ (x : ℝ), x ≥ 0 → f x ≤ M :=
sorry

end max_value_of_f_l354_35445


namespace milk_needed_for_cookies_l354_35451

/-- The number of cups in a quart -/
def cups_per_quart : ℚ := 4

/-- The number of cookies that can be baked with 3 quarts of milk -/
def cookies_per_three_quarts : ℕ := 24

/-- The number of cookies we want to bake -/
def target_cookies : ℕ := 6

/-- The number of cups of milk needed for the target number of cookies -/
def milk_needed : ℚ := 3

theorem milk_needed_for_cookies :
  milk_needed = (target_cookies : ℚ) / cookies_per_three_quarts * (3 * cups_per_quart) :=
sorry

end milk_needed_for_cookies_l354_35451


namespace problem_solution_l354_35403

theorem problem_solution (a b : ℕ) (ha : a > 0) (hb : b > 0) : 
  Real.log b / Real.log a = 3 → b - a = 1000 → a + b = 1010 := by
  sorry

end problem_solution_l354_35403


namespace problem_1_problem_2_problem_3_problem_4_problem_5_problem_6_l354_35426

-- Problem 1
theorem problem_1 : 13 + (-24) - (-40) = 29 := by sorry

-- Problem 2
theorem problem_2 : 3 * (-2) + (-36) / 4 = -15 := by sorry

-- Problem 3
theorem problem_3 : (1 + 3/4 - 7/8 - 7/16) / (-7/8) = -1/2 := by sorry

-- Problem 4
theorem problem_4 : (-2)^3 / 4 - (10 - (-1)^10 * 2) = -10 := by sorry

-- Problem 5
theorem problem_5 (x y : ℝ) : 7*x*y + 2 - 3*x*y - 5 = 4*x*y - 3 := by sorry

-- Problem 6
theorem problem_6 (x : ℝ) : 4*x^2 - (5*x + x^2) + 6*x - 2*x^2 = x^2 + x := by sorry

end problem_1_problem_2_problem_3_problem_4_problem_5_problem_6_l354_35426


namespace students_guinea_pigs_difference_l354_35487

theorem students_guinea_pigs_difference : 
  let students_per_class : ℕ := 25
  let guinea_pigs_per_class : ℕ := 3
  let num_classes : ℕ := 6
  let total_students : ℕ := students_per_class * num_classes
  let total_guinea_pigs : ℕ := guinea_pigs_per_class * num_classes
  total_students - total_guinea_pigs = 132 :=
by
  sorry


end students_guinea_pigs_difference_l354_35487


namespace intersection_M_N_l354_35407

open Set

def M : Set ℝ := {x | x < 2017}
def N : Set ℝ := {x | 0 < x ∧ x < 1}

theorem intersection_M_N : M ∩ N = {x | 0 < x ∧ x < 1} := by sorry

end intersection_M_N_l354_35407


namespace cube_root_27_times_fourth_root_81_times_square_root_9_l354_35474

theorem cube_root_27_times_fourth_root_81_times_square_root_9 :
  (27 : ℝ) ^ (1/3) * (81 : ℝ) ^ (1/4) * (9 : ℝ) ^ (1/2) = 27 := by sorry

end cube_root_27_times_fourth_root_81_times_square_root_9_l354_35474


namespace G_equals_2F_l354_35427

noncomputable section

variable (x : ℝ)

def F (x : ℝ) : ℝ := Real.log ((1 + x) / (1 - x))

def G (x : ℝ) : ℝ := F ((x * (1 + x^2)) / (1 + x^4))

theorem G_equals_2F : G x = 2 * F x := by sorry

end G_equals_2F_l354_35427


namespace abs_p_minus_q_equals_five_l354_35430

theorem abs_p_minus_q_equals_five (p q : ℝ) (h1 : p * q = 6) (h2 : p + q = 7) : |p - q| = 5 := by
  sorry

end abs_p_minus_q_equals_five_l354_35430


namespace sqrt_product_plus_one_l354_35458

theorem sqrt_product_plus_one : 
  Real.sqrt ((41:ℝ) * 40 * 39 * 38 + 1) = 1559 := by sorry

end sqrt_product_plus_one_l354_35458


namespace problem_solution_l354_35495

theorem problem_solution (x : ℝ) 
  (h : x + Real.sqrt (x^2 - 4) + 1 / (x - Real.sqrt (x^2 - 4)) = 10) : 
  x^2 + Real.sqrt (x^4 - 4) + 1 / (x^2 + Real.sqrt (x^4 - 4)) = 841/100 := by
  sorry

end problem_solution_l354_35495


namespace final_net_worth_l354_35480

/-- Represents a person's assets --/
structure Assets where
  cash : Int
  has_house : Bool
  has_vehicle : Bool

/-- Represents a transaction between two people --/
inductive Transaction
  | sell_house (price : Int)
  | sell_vehicle (price : Int)

/-- Performs a transaction and updates the assets of both parties --/
def perform_transaction (a b : Assets) (t : Transaction) : Assets × Assets :=
  match t with
  | Transaction.sell_house price => 
    ({ cash := a.cash + price, has_house := false, has_vehicle := a.has_vehicle },
     { cash := b.cash - price, has_house := true, has_vehicle := b.has_vehicle })
  | Transaction.sell_vehicle price => 
    ({ cash := a.cash - price, has_house := a.has_house, has_vehicle := true },
     { cash := b.cash + price, has_house := b.has_house, has_vehicle := false })

/-- Calculates the net worth of a person given their assets and the values of the house and vehicle --/
def net_worth (a : Assets) (house_value vehicle_value : Int) : Int :=
  a.cash + (if a.has_house then house_value else 0) + (if a.has_vehicle then vehicle_value else 0)

/-- The main theorem stating the final net worth of Mr. A and Mr. B --/
theorem final_net_worth (initial_a initial_b : Assets) 
  (house_value vehicle_value : Int) (transactions : List Transaction) : 
  initial_a.cash = 20000 → initial_a.has_house = true → initial_a.has_vehicle = false →
  initial_b.cash = 22000 → initial_b.has_house = false → initial_b.has_vehicle = true →
  house_value = 20000 → vehicle_value = 10000 →
  transactions = [
    Transaction.sell_house 25000,
    Transaction.sell_vehicle 12000,
    Transaction.sell_house 18000,
    Transaction.sell_vehicle 9000
  ] →
  let (final_a, final_b) := transactions.foldl 
    (fun (acc : Assets × Assets) (t : Transaction) => perform_transaction acc.1 acc.2 t) 
    (initial_a, initial_b)
  net_worth final_a house_value vehicle_value = 40000 ∧ 
  net_worth final_b house_value vehicle_value = 8000 := by
  sorry


end final_net_worth_l354_35480


namespace a_10_value_l354_35473

def arithmeticSequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

theorem a_10_value (a : ℕ → ℤ) 
  (h_seq : arithmeticSequence a) 
  (h_a7 : a 7 = 4) 
  (h_a8 : a 8 = 1) : 
  a 10 = -5 := by
sorry

end a_10_value_l354_35473


namespace marble_probability_theorem_l354_35422

/-- Represents the number of marbles of each color in the bag -/
structure MarbleCounts where
  red : ℕ
  white : ℕ
  blue : ℕ
  green : ℕ

/-- Calculates the probability of drawing 4 marbles of the same color -/
def probSameColor (counts : MarbleCounts) : ℚ :=
  let total := counts.red + counts.white + counts.blue + counts.green
  let probRed := Nat.choose counts.red 4 / Nat.choose total 4
  let probWhite := Nat.choose counts.white 4 / Nat.choose total 4
  let probBlue := Nat.choose counts.blue 4 / Nat.choose total 4
  let probGreen := Nat.choose counts.green 4 / Nat.choose total 4
  probRed + probWhite + probBlue + probGreen

theorem marble_probability_theorem (counts : MarbleCounts) 
    (h : counts = ⟨6, 7, 8, 9⟩) : 
    probSameColor counts = 82 / 9135 := by
  sorry

end marble_probability_theorem_l354_35422


namespace sequence_characterization_l354_35484

theorem sequence_characterization (a : ℕ → ℝ) :
  (∀ n : ℕ, a (n + 1) / a n = (a (n + 2) - a (n + 1)) / (a (n + 1) - a n)) →
  (∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n) :=
by sorry

end sequence_characterization_l354_35484


namespace system1_solution_system2_solution_l354_35401

-- System 1
theorem system1_solution :
  ∃ (x y : ℝ), x - 2*y = 1 ∧ 4*x + 3*y = 26 ∧ x = 5 ∧ y = 2 := by
  sorry

-- System 2
theorem system2_solution :
  ∃ (x y : ℝ), 2*x + 3*y = 3 ∧ 5*x - 3*y = 18 ∧ x = 3 ∧ y = -1 := by
  sorry

end system1_solution_system2_solution_l354_35401


namespace quadratic_inequality_properties_l354_35402

-- Define the quadratic function
def f (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

-- Define the solution set
def solution_set (a b c : ℝ) : Set ℝ := {x | x < -3 ∨ x > 4}

-- State the theorem
theorem quadratic_inequality_properties
  (a b c : ℝ)
  (h : ∀ x, f a b c x > 0 ↔ x ∈ solution_set a b c) :
  a > 0 ∧
  (∀ x, c * x^2 - b * x + a < 0 ↔ x < -1/4 ∨ x > 1/3) :=
by sorry

end quadratic_inequality_properties_l354_35402


namespace shoe_price_ratio_l354_35492

theorem shoe_price_ratio (marked_price : ℝ) (marked_price_pos : marked_price > 0) : 
  let discount_rate : ℝ := 1/4
  let selling_price : ℝ := marked_price * (1 - discount_rate)
  let cost_price : ℝ := (2/3) * selling_price
  cost_price / marked_price = 1/2 := by sorry

end shoe_price_ratio_l354_35492


namespace integer_solutions_l354_35477

theorem integer_solutions (a : ℤ) : 
  (∃ b c : ℤ, ∀ x : ℤ, (x - a) * (x - 12) + 1 = (x + b) * (x + c)) ↔ 
  (a = 10 ∨ a = 14) :=
sorry

end integer_solutions_l354_35477


namespace cube_volume_ratio_l354_35417

-- Define the edge lengths in inches
def edge_length_1 : ℚ := 9
def edge_length_2 : ℚ := 3 * 12

-- Define the volume ratio function
def volume_ratio (a b : ℚ) : ℚ := (a / b) ^ 3

-- Theorem statement
theorem cube_volume_ratio : volume_ratio edge_length_1 edge_length_2 = 1 / 64 := by
  sorry

end cube_volume_ratio_l354_35417


namespace shems_earnings_proof_l354_35453

/-- Calculates Shem's earnings for a workday given Kem's hourly rate, Shem's rate multiplier, and hours worked. -/
def shems_daily_earnings (kems_hourly_rate : ℝ) (shems_rate_multiplier : ℝ) (hours_worked : ℝ) : ℝ :=
  kems_hourly_rate * shems_rate_multiplier * hours_worked

/-- Proves that Shem's earnings for an 8-hour workday is $80, given the conditions. -/
theorem shems_earnings_proof (kems_hourly_rate : ℝ) (shems_rate_multiplier : ℝ) (hours_worked : ℝ) 
    (h1 : kems_hourly_rate = 4)
    (h2 : shems_rate_multiplier = 2.5)
    (h3 : hours_worked = 8) :
    shems_daily_earnings kems_hourly_rate shems_rate_multiplier hours_worked = 80 := by
  sorry

end shems_earnings_proof_l354_35453


namespace lens_break_probability_l354_35409

def prob_break_first : ℝ := 0.3
def prob_break_second_given_not_first : ℝ := 0.4
def prob_break_third_given_not_first_two : ℝ := 0.9

theorem lens_break_probability :
  let prob_break_second := (1 - prob_break_first) * prob_break_second_given_not_first
  let prob_break_third := (1 - prob_break_first) * (1 - prob_break_second_given_not_first) * prob_break_third_given_not_first_two
  prob_break_first + prob_break_second + prob_break_third = 0.958 := by
  sorry

end lens_break_probability_l354_35409


namespace triangle_height_l354_35442

/-- Given a triangle with area 3 square meters and base 2 meters, its height is 3 meters -/
theorem triangle_height (area : ℝ) (base : ℝ) (height : ℝ) : 
  area = 3 → base = 2 → area = (base * height) / 2 → height = 3 := by
  sorry

end triangle_height_l354_35442


namespace platonic_self_coincidences_l354_35438

/-- Represents a regular polyhedron -/
structure RegularPolyhedron where
  n : ℕ  -- number of sides of each face
  F : ℕ  -- number of faces
  is_regular : n ≥ 3 ∧ F ≥ 4  -- conditions for regularity

/-- Calculates the number of self-coincidences for a regular polyhedron -/
def self_coincidences (p : RegularPolyhedron) : ℕ :=
  2 * p.n * p.F

/-- Theorem stating the number of self-coincidences for each Platonic solid -/
theorem platonic_self_coincidences :
  ∃ (tetrahedron cube octahedron dodecahedron icosahedron : RegularPolyhedron),
    (self_coincidences tetrahedron = 24) ∧
    (self_coincidences cube = 48) ∧
    (self_coincidences octahedron = 48) ∧
    (self_coincidences dodecahedron = 120) ∧
    (self_coincidences icosahedron = 120) :=
by sorry

end platonic_self_coincidences_l354_35438


namespace building_heights_l354_35443

/-- Given three buildings with specified height relationships, calculate their total height. -/
theorem building_heights (height_1 : ℝ) : 
  height_1 = 600 →
  let height_2 := 2 * height_1
  let height_3 := 3 * (height_1 + height_2)
  height_1 + height_2 + height_3 = 7200 := by
  sorry

end building_heights_l354_35443


namespace largest_prime_factor_is_13_l354_35405

def numbers : List Nat := [45, 63, 98, 121, 169]

def is_prime (n : Nat) : Prop := n > 1 ∧ ∀ m : Nat, m > 1 → m < n → ¬(n % m = 0)

def prime_factors (n : Nat) : Set Nat :=
  {p : Nat | is_prime p ∧ n % p = 0}

theorem largest_prime_factor_is_13 :
  ∃ (n : Nat), n ∈ numbers ∧ 13 ∈ prime_factors n ∧
  ∀ (m : Nat), m ∈ numbers → ∀ (p : Nat), p ∈ prime_factors m → p ≤ 13 :=
by sorry

end largest_prime_factor_is_13_l354_35405


namespace all_2k_trips_use_both_modes_all_2k_plus_1_trips_use_both_modes_l354_35428

/-- A type representing cities in a country. -/
structure City where
  id : Nat

/-- A type representing transportation modes. -/
inductive TransportMode
  | Bus
  | Flight

/-- A function that determines if two cities are directly connected by a given transport mode. -/
def directConnection (c1 c2 : City) (mode : TransportMode) : Prop :=
  sorry

/-- A proposition stating that any two cities are connected by either a direct flight or a direct bus route. -/
axiom connected_cities (c1 c2 : City) :
  directConnection c1 c2 TransportMode.Bus ∨ directConnection c1 c2 TransportMode.Flight

/-- A type representing a round trip as a list of cities. -/
def RoundTrip := List City

/-- A function that checks if a round trip uses both bus and flight. -/
def usesBothModes (trip : RoundTrip) : Prop :=
  sorry

/-- A theorem stating that all round trips touching 2k cities (k > 3) must use both bus and flight. -/
theorem all_2k_trips_use_both_modes (k : Nat) (h : k > 3) :
  ∀ (trip : RoundTrip), trip.length = 2 * k → usesBothModes trip :=
  sorry

/-- The main theorem to prove: if all round trips touching 2k cities (k > 3) must use both bus and flight,
    then all round trips touching 2k+1 cities must also use both bus and flight. -/
theorem all_2k_plus_1_trips_use_both_modes (k : Nat) (h : k > 3) :
  (∀ (trip : RoundTrip), trip.length = 2 * k → usesBothModes trip) →
  (∀ (trip : RoundTrip), trip.length = 2 * k + 1 → usesBothModes trip) :=
  sorry

end all_2k_trips_use_both_modes_all_2k_plus_1_trips_use_both_modes_l354_35428


namespace choose_4_from_10_l354_35429

theorem choose_4_from_10 : Nat.choose 10 4 = 210 := by
  sorry

end choose_4_from_10_l354_35429


namespace sum_of_roots_l354_35419

theorem sum_of_roots (p q r s : ℝ) : 
  p ≠ q ∧ p ≠ r ∧ p ≠ s ∧ q ≠ r ∧ q ≠ s ∧ r ≠ s →
  (∀ x : ℝ, x^2 - 12*p*x - 13*q = 0 ↔ x = r ∨ x = s) →
  (∀ x : ℝ, x^2 - 12*r*x - 13*s = 0 ↔ x = p ∨ x = q) →
  p + q + r + s = 2028 :=
by sorry

end sum_of_roots_l354_35419


namespace tank_capacity_l354_35413

/-- Proves that the capacity of a tank is 21600 litres given specific inlet and outlet conditions. -/
theorem tank_capacity : 
  ∀ (outlet_time inlet_rate extended_time : ℝ),
  outlet_time = 10 →
  inlet_rate = 16 * 60 →
  extended_time = outlet_time + 8 →
  ∃ (capacity : ℝ),
  capacity / outlet_time - inlet_rate = capacity / extended_time ∧
  capacity = 21600 :=
by sorry

end tank_capacity_l354_35413


namespace simplify_expression_l354_35420

theorem simplify_expression (x : ℝ) : 5 * x + 2 * (4 + x) = 7 * x + 8 := by
  sorry

end simplify_expression_l354_35420


namespace compound_molecular_weight_l354_35479

-- Define atomic weights
def atomic_weight_Ca : ℝ := 40.08
def atomic_weight_O : ℝ := 16.00
def atomic_weight_H : ℝ := 1.01

-- Define number of atoms in the compound
def num_Ca : ℕ := 1
def num_O : ℕ := 2
def num_H : ℕ := 2

-- Define the molecular weight calculation function
def molecular_weight : ℝ :=
  (num_Ca : ℝ) * atomic_weight_Ca + 
  (num_O : ℝ) * atomic_weight_O + 
  (num_H : ℝ) * atomic_weight_H

-- Theorem statement
theorem compound_molecular_weight : 
  molecular_weight = 74.10 := by sorry

end compound_molecular_weight_l354_35479


namespace apps_deletion_ways_l354_35481

-- Define the total number of applications
def total_apps : ℕ := 21

-- Define the number of applications to be deleted
def apps_to_delete : ℕ := 6

-- Define the number of special applications
def special_apps : ℕ := 6

-- Define the number of special apps to be selected
def special_apps_to_select : ℕ := 3

-- Define the number of pairs of special apps
def special_pairs : ℕ := 3

-- Theorem statement
theorem apps_deletion_ways :
  (2^special_pairs) * (Nat.choose (total_apps - special_apps) (apps_to_delete - special_apps_to_select)) = 3640 :=
sorry

end apps_deletion_ways_l354_35481


namespace range_of_a_l354_35450

-- Define the function f
def f (x : ℝ) : ℝ := -2 * x^5 - x^3 - 7 * x + 2

-- State the theorem
theorem range_of_a (a : ℝ) :
  f (a^2) + f (a - 2) > 4 → -2 < a ∧ a < 1 :=
by sorry

end range_of_a_l354_35450


namespace factorization_proof_l354_35418

theorem factorization_proof (z : ℂ) : 55 * z^17 + 121 * z^34 = 11 * z^17 * (5 + 11 * z^17) := by
  sorry

end factorization_proof_l354_35418


namespace matches_for_128_teams_l354_35406

/-- Represents a single-elimination tournament. -/
structure SingleEliminationTournament where
  num_teams : ℕ
  num_teams_positive : 0 < num_teams

/-- Calculates the number of matches required to determine a champion. -/
def matches_required (tournament : SingleEliminationTournament) : ℕ :=
  tournament.num_teams - 1

/-- Theorem: In a single-elimination tournament with 128 teams, 127 matches are required. -/
theorem matches_for_128_teams :
  let tournament := SingleEliminationTournament.mk 128 (by norm_num)
  matches_required tournament = 127 := by
  sorry

end matches_for_128_teams_l354_35406


namespace all_statements_false_l354_35457

-- Define prime and composite numbers
def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m ∣ n → m = 1 ∨ m = n
def isComposite (n : ℕ) : Prop := n > 1 ∧ ¬(isPrime n)

-- Define the four statements
def statement1 : Prop := ∀ p q : ℕ, isPrime p → isPrime q → isComposite (p + q)
def statement2 : Prop := ∀ a b : ℕ, isComposite a → isComposite b → isComposite (a + b)
def statement3 : Prop := ∀ p c : ℕ, isPrime p → isComposite c → isComposite (p + c)
def statement4 : Prop := ∀ p c : ℕ, isPrime p → isComposite c → ¬(isComposite (p + c))

-- Theorem stating that all four statements are false
theorem all_statements_false : ¬statement1 ∧ ¬statement2 ∧ ¬statement3 ∧ ¬statement4 :=
sorry

end all_statements_false_l354_35457


namespace palmer_photos_before_trip_l354_35478

def photos_before_trip (first_week : ℕ) (second_week_multiplier : ℕ) (third_fourth_week : ℕ) (total_after_trip : ℕ) : ℕ :=
  total_after_trip - (first_week + second_week_multiplier * first_week + third_fourth_week)

theorem palmer_photos_before_trip :
  photos_before_trip 50 2 80 380 = 150 :=
by sorry

end palmer_photos_before_trip_l354_35478


namespace arithmetic_sequence_proof_l354_35475

def a (n : ℕ) : ℤ := 3 * n - 5

theorem arithmetic_sequence_proof :
  (∀ n : ℕ, a (n + 1) - a n = 3) ∧
  (a 1 = -2) ∧
  (∀ n : ℕ, a (n + 1) - a n = 3) :=
by sorry

end arithmetic_sequence_proof_l354_35475


namespace rachel_picked_apples_l354_35424

/-- Represents the number of apples Rachel picked from her tree -/
def apples_picked : ℕ := 2

/-- The initial number of apples on Rachel's tree -/
def initial_apples : ℕ := 4

/-- The number of new apples that grew on the tree -/
def new_apples : ℕ := 3

/-- The final number of apples on the tree -/
def final_apples : ℕ := 5

/-- Theorem stating that the number of apples Rachel picked is correct -/
theorem rachel_picked_apples :
  initial_apples - apples_picked + new_apples = final_apples :=
by sorry

end rachel_picked_apples_l354_35424


namespace probability_both_classes_l354_35461

-- Define the total number of students
def total_students : ℕ := 40

-- Define the number of students in Mandarin
def mandarin_students : ℕ := 30

-- Define the number of students in German
def german_students : ℕ := 35

-- Define the function to calculate the number of ways to choose k items from n items
def choose (n k : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

-- Define the theorem
theorem probability_both_classes : 
  let students_both := mandarin_students + german_students - total_students
  let students_only_mandarin := mandarin_students - students_both
  let students_only_german := german_students - students_both
  let total_ways := choose total_students 2
  let ways_not_both := choose students_only_mandarin 2 + choose students_only_german 2
  (total_ways - ways_not_both) / total_ways = 145 / 156 := by
sorry

end probability_both_classes_l354_35461


namespace modified_sum_theorem_l354_35469

theorem modified_sum_theorem (S a b : ℝ) (h : a + b = S) :
  (3 * a + 4) + (2 * b + 5) = 3 * S + 9 := by
  sorry

end modified_sum_theorem_l354_35469


namespace system_solution_l354_35436

theorem system_solution (k : ℚ) : 
  (∃ x y : ℚ, x + y = 5 * k ∧ x - y = 9 * k ∧ 2 * x + 3 * y = 6) → k = 3/4 := by
  sorry

end system_solution_l354_35436


namespace divide_by_repeating_decimal_l354_35447

theorem divide_by_repeating_decimal :
  ∃ (x : ℚ), (∀ (n : ℕ), x = (3 * 10^n - 3) / (9 * 10^n)) ∧ (8 / x = 24) := by
  sorry

end divide_by_repeating_decimal_l354_35447


namespace smaller_integer_proof_l354_35435

theorem smaller_integer_proof (x y : ℤ) (h1 : x + y = -9) (h2 : y - x = 1) : x = -5 := by
  sorry

end smaller_integer_proof_l354_35435


namespace monotone_increasing_interval_l354_35494

/-- The function f(x) = (3 - x^2) * e^x is monotonically increasing on the interval (-3, 1) -/
theorem monotone_increasing_interval (x : ℝ) :
  StrictMonoOn (fun x => (3 - x^2) * Real.exp x) (Set.Ioo (-3) 1) := by
  sorry

end monotone_increasing_interval_l354_35494


namespace bobs_income_changes_l354_35462

def initial_income : ℝ := 2750
def february_increase : ℝ := 0.15
def march_decrease : ℝ := 0.10

theorem bobs_income_changes (initial : ℝ) (increase : ℝ) (decrease : ℝ) :
  initial = initial_income →
  increase = february_increase →
  decrease = march_decrease →
  initial * (1 + increase) * (1 - decrease) = 2846.25 :=
by sorry

end bobs_income_changes_l354_35462


namespace inequality_not_always_true_l354_35488

theorem inequality_not_always_true (a b : ℝ) (h1 : 0 < b) (h2 : b < a) :
  ∃ a b, 0 < b ∧ b < a ∧ ¬((1 / (a - b)) > (1 / b)) :=
sorry

end inequality_not_always_true_l354_35488


namespace derivative_f_at_pi_l354_35437

noncomputable def f (x : ℝ) : ℝ := x^2 / Real.cos x

theorem derivative_f_at_pi : 
  deriv f π = -2 * π := by sorry

end derivative_f_at_pi_l354_35437


namespace quadratic_solution_l354_35414

theorem quadratic_solution : 
  ∀ x : ℝ, x * (x - 7) = 0 ↔ x = 0 ∨ x = 7 := by sorry

end quadratic_solution_l354_35414


namespace fourth_root_of_25000000_l354_35465

theorem fourth_root_of_25000000 : (70.7 : ℝ)^4 = 25000000 := by
  sorry

end fourth_root_of_25000000_l354_35465


namespace hyperbola_asymptote_implies_a_eq_2_l354_35444

-- Define the hyperbola equation
def hyperbola_eq (x y a : ℝ) : Prop := x^2 / a^2 - y^2 / 9 = 1

-- Define the asymptote equation
def asymptote_eq (x y : ℝ) : Prop := 3 * x + 2 * y = 0

-- Theorem statement
theorem hyperbola_asymptote_implies_a_eq_2 :
  ∀ a : ℝ, a > 0 →
  (∀ x y : ℝ, hyperbola_eq x y a ↔ asymptote_eq x y) →
  a = 2 :=
sorry

end hyperbola_asymptote_implies_a_eq_2_l354_35444


namespace marble_distribution_l354_35433

/-- The minimum number of additional marbles needed and the sum of marbles for specific friends -/
theorem marble_distribution (n : Nat) (initial_marbles : Nat) 
  (h1 : n = 12) (h2 : initial_marbles = 34) : 
  let additional_marbles := (n * (n + 1)) / 2 - initial_marbles
  let third_friend := 3
  let seventh_friend := 7
  let eleventh_friend := 11
  (additional_marbles = 44) ∧ 
  (third_friend + seventh_friend + eleventh_friend = 21) := by
  sorry

end marble_distribution_l354_35433


namespace girl_multiplication_mistake_l354_35464

theorem girl_multiplication_mistake (x : ℝ) : 43 * x - 34 * x = 1215 → x = 135 := by
  sorry

end girl_multiplication_mistake_l354_35464


namespace billboard_problem_l354_35470

/-- The number of billboards to be erected -/
def num_billboards : ℕ := 200

/-- The length of the road in meters -/
def road_length : ℕ := 1100

/-- The spacing between billboards in the first scenario (in meters) -/
def spacing1 : ℚ := 5

/-- The spacing between billboards in the second scenario (in meters) -/
def spacing2 : ℚ := 11/2

/-- The number of missing billboards in the first scenario -/
def missing1 : ℕ := 21

/-- The number of missing billboards in the second scenario -/
def missing2 : ℕ := 1

theorem billboard_problem :
  (spacing1 * (num_billboards + missing1 - 1 : ℚ) = road_length) ∧
  (spacing2 * (num_billboards + missing2 - 1 : ℚ) = road_length) := by
  sorry

end billboard_problem_l354_35470


namespace equation_solution_l354_35499

theorem equation_solution : ∃ x : ℝ, 2 * x - 3 = 6 - x :=
  by
    use 3
    sorry

#check equation_solution

end equation_solution_l354_35499


namespace cuboid_surface_area_example_l354_35439

/-- The surface area of a cuboid -/
def cuboidSurfaceArea (length width height : ℝ) : ℝ :=
  2 * (length * width + length * height + width * height)

/-- Theorem: The surface area of a cuboid with length 8, width 10, and height 12 is 592 -/
theorem cuboid_surface_area_example : cuboidSurfaceArea 8 10 12 = 592 := by
  sorry

end cuboid_surface_area_example_l354_35439


namespace product_increase_by_2016_l354_35440

theorem product_increase_by_2016 : ∃ (a b c : ℕ), 
  (a - 3) * (b - 3) * (c - 3) = a * b * c + 2016 := by
sorry

end product_increase_by_2016_l354_35440


namespace parabola_focus_directrix_distance_l354_35415

/-- For a parabola with equation y^2 = 4x, the distance between its focus and directrix is 2. -/
theorem parabola_focus_directrix_distance :
  ∀ (x y : ℝ), y^2 = 4*x → ∃ (f d : ℝ × ℝ),
    (f.1 = 1 ∧ f.2 = 0) ∧  -- focus
    (d.1 = -1 ∧ ∀ t, d.2 = t) ∧  -- directrix
    (f.1 - d.1 = 2) :=
by sorry

end parabola_focus_directrix_distance_l354_35415


namespace smallest_divisor_after_subtraction_l354_35459

theorem smallest_divisor_after_subtraction (n : ℕ) (m : ℕ) (d : ℕ) : 
  n = 378461 →
  m = 5 →
  d = 47307 →
  (n - m) % d = 0 ∧
  ∀ k : ℕ, 5 < k → k < d → (n - m) % k ≠ 0 :=
by sorry

end smallest_divisor_after_subtraction_l354_35459


namespace f_positive_iff_l354_35404

def f (x : ℝ) := (x + 1) * (x - 1) * (x - 3)

theorem f_positive_iff (x : ℝ) : f x > 0 ↔ (x > -1 ∧ x < 1) ∨ x > 3 := by
  sorry

end f_positive_iff_l354_35404


namespace prob_three_out_of_five_l354_35482

def prob_single_win : ℚ := 2/3

theorem prob_three_out_of_five :
  let n : ℕ := 5
  let k : ℕ := 3
  let p : ℚ := prob_single_win
  (n.choose k) * p^k * (1-p)^(n-k) = 80/243 := by
sorry

end prob_three_out_of_five_l354_35482


namespace penguin_count_l354_35471

/-- The number of penguins in a zoo can be determined by adding the number of penguins 
    already fed and the number of penguins still to be fed. -/
theorem penguin_count (total_fish : ℕ) (fed_penguins : ℕ) (to_be_fed : ℕ) :
  total_fish ≥ fed_penguins + to_be_fed →
  fed_penguins + to_be_fed = fed_penguins + to_be_fed :=
by
  sorry

#check penguin_count

end penguin_count_l354_35471


namespace arrangement_count_correct_l354_35454

/-- The number of ways to arrange 5 volunteers and 2 elderly people in a row -/
def arrangementCount : ℕ :=
  let volunteerCount : ℕ := 5
  let elderlyCount : ℕ := 2
  let totalCount : ℕ := volunteerCount + elderlyCount
  let endPositions : ℕ := 2  -- number of end positions
  let intermediatePositions : ℕ := totalCount - endPositions - 1  -- -1 for elderly pair

  -- Choose volunteers for end positions
  let endArrangements : ℕ := volunteerCount * (volunteerCount - 1)
  
  -- Arrange remaining volunteers and elderly pair
  let intermediateArrangements : ℕ := Nat.factorial intermediatePositions
  
  -- Arrange elderly within their pair
  let elderlyArrangements : ℕ := Nat.factorial elderlyCount

  endArrangements * intermediateArrangements * elderlyArrangements

theorem arrangement_count_correct :
  arrangementCount = 960 := by
  sorry

end arrangement_count_correct_l354_35454


namespace shelves_needed_l354_35410

theorem shelves_needed (initial_books : ℝ) (added_books : ℝ) (books_per_shelf : ℝ) :
  initial_books = 46.0 →
  added_books = 10.0 →
  books_per_shelf = 4.0 →
  ((initial_books + added_books) / books_per_shelf) = 14.0 := by
  sorry

end shelves_needed_l354_35410


namespace book_purchase_problem_l354_35463

theorem book_purchase_problem (total_A total_B only_B both : ℕ) 
  (h1 : total_A = 2 * total_B)
  (h2 : both = 500)
  (h3 : both = 2 * only_B) :
  total_A - both = 1000 := by
  sorry

end book_purchase_problem_l354_35463


namespace polynomial_value_theorem_l354_35472

theorem polynomial_value_theorem (x : ℂ) (h : x^2 + x + 1 = 0) :
  x^2000 + x^1999 + x^1998 + 1000*x^1000 + 1000*x^999 + 1000*x^998 + 2000*x^3 + 2000*x^2 + 2000*x + 3000 = 3000 := by
  sorry

end polynomial_value_theorem_l354_35472


namespace acute_triangle_sine_cosine_inequality_l354_35412

theorem acute_triangle_sine_cosine_inequality (α β γ : Real) 
  (h_acute : α > 0 ∧ β > 0 ∧ γ > 0 ∧ α + β + γ = π) : 
  Real.sin α * Real.sin β * Real.sin γ > 5 * Real.cos α * Real.cos β * Real.cos γ := by
  sorry

end acute_triangle_sine_cosine_inequality_l354_35412


namespace periodic_function_theorem_l354_35423

def is_periodic (f : ℝ → ℝ) : Prop :=
  ∃ c : ℝ, c ≠ 0 ∧ ∀ x : ℝ, f (x + c) = f x

theorem periodic_function_theorem (f : ℝ → ℝ) 
  (h1 : ∀ x : ℝ, |f x| ≤ 1)
  (h2 : ∀ x : ℝ, f (x + 13/42) + f x = f (x + 1/6) + f (x + 1/7)) :
  is_periodic f :=
sorry

end periodic_function_theorem_l354_35423


namespace equation_root_conditions_l354_35421

theorem equation_root_conditions (a : ℝ) : 
  (∃ x > 0, |x| = a*x - a) ∧ 
  (∀ x < 0, |x| ≠ a*x - a) → 
  a > 1 ∨ a ≤ -1 :=
by sorry

end equation_root_conditions_l354_35421


namespace circle_equation_l354_35460

-- Define the polar coordinate system
def PolarCoordinate := ℝ × ℝ  -- (ρ, θ)

-- Define the line l
def line_l (p : PolarCoordinate) : Prop :=
  p.1 * Real.cos p.2 + p.1 * Real.sin p.2 = 2

-- Define the point M where line l intersects the polar axis
def point_M : ℝ × ℝ := (2, 0)  -- Cartesian coordinates

-- Define the circle with OM as diameter
def circle_OM (p : PolarCoordinate) : Prop :=
  p.1 = 2 * Real.cos p.2

-- Theorem statement
theorem circle_equation (p : PolarCoordinate) :
  line_l p → circle_OM p :=
sorry

end circle_equation_l354_35460


namespace residue_mod_17_l354_35400

theorem residue_mod_17 : (207 * 13 - 22 * 8 + 5) % 17 = 3 := by
  sorry

end residue_mod_17_l354_35400


namespace lindas_age_multiple_l354_35456

/-- Given:
  - Linda's age (L) is 3 more than a certain multiple (M) of Jane's age (J)
  - In five years, the sum of their ages will be 28
  - Linda's current age is 13
Prove that the multiple M is equal to 2 -/
theorem lindas_age_multiple (J L M : ℕ) : 
  L = M * J + 3 →
  L = 13 →
  L + 5 + J + 5 = 28 →
  M = 2 := by
sorry

end lindas_age_multiple_l354_35456


namespace eight_power_x_equals_one_eighth_of_two_power_thirty_l354_35416

theorem eight_power_x_equals_one_eighth_of_two_power_thirty (x : ℝ) : 
  (1/8 : ℝ) * (2^30) = 8^x → x = 9 := by
sorry

end eight_power_x_equals_one_eighth_of_two_power_thirty_l354_35416


namespace x_value_when_derivative_is_three_l354_35489

def f (x : ℝ) := x^3

theorem x_value_when_derivative_is_three (x : ℝ) (h1 : x > 0) (h2 : (deriv f) x = 3) : x = 1 := by
  sorry

end x_value_when_derivative_is_three_l354_35489


namespace quadratic_inequality_solution_set_l354_35446

theorem quadratic_inequality_solution_set (b : ℝ) :
  (∀ x : ℝ, x^2 + b*x - b - 3/4 > 0) ↔ -3 < b ∧ b < -1 := by
  sorry

end quadratic_inequality_solution_set_l354_35446
