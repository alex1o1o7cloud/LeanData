import Mathlib

namespace NUMINAMATH_CALUDE_difference_of_squares_divisible_by_eight_l1874_187450

theorem difference_of_squares_divisible_by_eight (a b : ℤ) (h : a > b) :
  ∃ k : ℤ, (2 * a + 1)^2 - (2 * b + 1)^2 = 8 * k := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_divisible_by_eight_l1874_187450


namespace NUMINAMATH_CALUDE_cut_to_square_l1874_187409

/-- Represents a shape on a checkered paper --/
structure Shape :=
  (area : ℕ)
  (has_hole : Bool)

/-- Represents a square --/
def is_square (s : Shape) : Prop :=
  ∃ (side : ℕ), s.area = side * side ∧ s.has_hole = false

/-- Represents the ability to cut a shape into two parts --/
def can_cut (s : Shape) : Prop :=
  ∃ (part1 part2 : Shape), part1.area + part2.area = s.area

/-- Represents the ability to form a square from two parts --/
def can_form_square (part1 part2 : Shape) : Prop :=
  is_square (Shape.mk (part1.area + part2.area) false)

/-- The main theorem: given a shape with a hole, it can be cut into two parts
    that can form a square --/
theorem cut_to_square (s : Shape) (h : s.has_hole = true) :
  ∃ (part1 part2 : Shape),
    can_cut s ∧
    can_form_square part1 part2 :=
sorry

end NUMINAMATH_CALUDE_cut_to_square_l1874_187409


namespace NUMINAMATH_CALUDE_thirteenth_fib_is_610_l1874_187440

/-- Fibonacci sequence -/
def fib : ℕ → ℕ
  | 0 => 0
  | 1 => 1
  | (n + 2) => fib (n + 1) + fib n

/-- The 13th Fibonacci number is 610 -/
theorem thirteenth_fib_is_610 : fib 13 = 610 := by
  sorry

end NUMINAMATH_CALUDE_thirteenth_fib_is_610_l1874_187440


namespace NUMINAMATH_CALUDE_paint_room_time_l1874_187461

/-- The time (in hours) it takes Alice to paint the room alone -/
def alice_time : ℝ := 3

/-- The time (in hours) it takes Bob to paint the room alone -/
def bob_time : ℝ := 6

/-- The duration (in hours) of the break Alice and Bob take -/
def break_time : ℝ := 2

/-- The total time (in hours) it takes Alice and Bob to paint the room together, including the break -/
def total_time : ℝ := 4

theorem paint_room_time :
  (1 / alice_time + 1 / bob_time) * (total_time - break_time) = 1 :=
sorry

end NUMINAMATH_CALUDE_paint_room_time_l1874_187461


namespace NUMINAMATH_CALUDE_no_distributive_laws_hold_l1874_187417

-- Define the # operation
def hash (a b : ℝ) : ℝ := a + 2 * b

-- Theorem stating that none of the distributive laws hold
theorem no_distributive_laws_hold :
  (∃ x y z : ℝ, hash x (y + z) ≠ hash x y + hash x z) ∧
  (∃ x y z : ℝ, x + hash y z ≠ hash (x + y) (x + z)) ∧
  (∃ x y z : ℝ, hash x (hash y z) ≠ hash (hash x y) (hash x z)) :=
by
  sorry


end NUMINAMATH_CALUDE_no_distributive_laws_hold_l1874_187417


namespace NUMINAMATH_CALUDE_remaining_payment_l1874_187471

/-- Given a product with a 10% deposit of $140, prove that the remaining amount to be paid is $1260 -/
theorem remaining_payment (deposit : ℝ) (deposit_percentage : ℝ) (full_price : ℝ) : 
  deposit = 140 ∧ 
  deposit_percentage = 0.1 ∧ 
  deposit = deposit_percentage * full_price → 
  full_price - deposit = 1260 :=
by sorry

end NUMINAMATH_CALUDE_remaining_payment_l1874_187471


namespace NUMINAMATH_CALUDE_integer_fraction_characterization_l1874_187422

theorem integer_fraction_characterization (p n : ℕ) :
  Nat.Prime p → n > 0 →
  (∃ k : ℕ, (n^p + 1 : ℕ) = k * (p^n + 1)) ↔
  ((p = 2 ∧ (n = 2 ∨ n = 4)) ∨ (p > 2 ∧ n = p)) := by
  sorry

end NUMINAMATH_CALUDE_integer_fraction_characterization_l1874_187422


namespace NUMINAMATH_CALUDE_log_inequality_l1874_187441

theorem log_inequality : 
  let m := Real.log 0.6 / Real.log 0.3
  let n := (1/2) * (Real.log 0.6 / Real.log 2)
  m + n > m * n := by sorry

end NUMINAMATH_CALUDE_log_inequality_l1874_187441


namespace NUMINAMATH_CALUDE_total_discount_calculation_l1874_187426

theorem total_discount_calculation (original_price : ℝ) (initial_discount : ℝ) (additional_discount : ℝ) :
  initial_discount = 0.5 →
  additional_discount = 0.25 →
  let sale_price := original_price * (1 - initial_discount)
  let final_price := sale_price * (1 - additional_discount)
  let total_discount := (original_price - final_price) / original_price
  total_discount = 0.625 :=
by sorry

end NUMINAMATH_CALUDE_total_discount_calculation_l1874_187426


namespace NUMINAMATH_CALUDE_same_quotient_remainder_numbers_l1874_187478

theorem same_quotient_remainder_numbers : 
  {a : ℕ | ∃ q : ℕ, 0 < q ∧ q < 6 ∧ a = 7 * q} = {7, 14, 21, 28, 35} := by
  sorry

end NUMINAMATH_CALUDE_same_quotient_remainder_numbers_l1874_187478


namespace NUMINAMATH_CALUDE_distance_to_park_is_five_l1874_187462

/-- The distance from Talia's house to the park -/
def distance_to_park : ℝ := sorry

/-- The distance from the park to the grocery store -/
def park_to_grocery : ℝ := 3

/-- The distance from the grocery store to Talia's house -/
def grocery_to_house : ℝ := 8

/-- The total distance Talia drives -/
def total_distance : ℝ := 16

theorem distance_to_park_is_five :
  distance_to_park = 5 :=
by
  have h1 : distance_to_park + park_to_grocery + grocery_to_house = total_distance := by sorry
  sorry

end NUMINAMATH_CALUDE_distance_to_park_is_five_l1874_187462


namespace NUMINAMATH_CALUDE_locus_is_ellipse_l1874_187483

-- Define the two given circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 + 6*x + 5 = 0
def circle2 (x y : ℝ) : Prop := x^2 + y^2 - 6*x - 91 = 0

-- Define the property of being externally tangent
def externally_tangent (x y : ℝ) : Prop :=
  ∃ (R : ℝ), R > 0 ∧ (x + 3)^2 + y^2 = (R + 2)^2

-- Define the property of being internally tangent
def internally_tangent (x y : ℝ) : Prop :=
  ∃ (R : ℝ), R > 0 ∧ (x - 3)^2 + y^2 = (10 - R)^2

-- Define the locus of points
def locus (x y : ℝ) : Prop :=
  externally_tangent x y ∧ internally_tangent x y

-- Theorem stating that the locus forms an ellipse
theorem locus_is_ellipse :
  ∀ (x y : ℝ), locus x y → (x + 3)^2 / 36 + y^2 / 27 = 1 :=
sorry

end NUMINAMATH_CALUDE_locus_is_ellipse_l1874_187483


namespace NUMINAMATH_CALUDE_midnight_temperature_l1874_187424

def morning_temp : ℝ := 30
def afternoon_rise : ℝ := 1
def midnight_drop : ℝ := 7

theorem midnight_temperature : 
  morning_temp + afternoon_rise - midnight_drop = 24 := by
  sorry

end NUMINAMATH_CALUDE_midnight_temperature_l1874_187424


namespace NUMINAMATH_CALUDE_buses_meet_time_l1874_187490

/-- Represents a time of day in hours and minutes -/
structure Time where
  hours : Nat
  minutes : Nat
  deriving Repr

/-- Represents a bus journey -/
structure BusJourney where
  startTime : Time
  endTime : Time
  distance : Nat
  deriving Repr

/-- The problem setup -/
def busProbe : Prop :=
  let totalDistance : Nat := 189
  let lishanToCounty : Nat := 54
  let busAToCounty : BusJourney := { startTime := { hours := 8, minutes := 30 },
                                     endTime := { hours := 9, minutes := 15 },
                                     distance := lishanToCounty }
  let busAToProvincial : BusJourney := { startTime := { hours := 9, minutes := 30 },
                                         endTime := { hours := 11, minutes := 0 },
                                         distance := totalDistance - lishanToCounty }
  let busBSpeed : Nat := 60
  let busBStartTime : Time := { hours := 8, minutes := 50 }
  
  ∃ (meetingTime : Time),
    meetingTime.hours = 10 ∧ meetingTime.minutes = 8

theorem buses_meet_time : busProbe := by
  sorry

end NUMINAMATH_CALUDE_buses_meet_time_l1874_187490


namespace NUMINAMATH_CALUDE_anna_overall_score_l1874_187438

/-- Represents a test with a number of problems and a score percentage -/
structure Test where
  problems : ℕ
  score : ℚ
  h_score_range : 0 ≤ score ∧ score ≤ 1

/-- Calculates the number of problems answered correctly in a test -/
def correctProblems (t : Test) : ℚ :=
  t.problems * t.score

/-- Theorem stating that Anna's overall score across three tests is 78% -/
theorem anna_overall_score (test1 test2 test3 : Test)
  (h1 : test1.problems = 30 ∧ test1.score = 3/4)
  (h2 : test2.problems = 50 ∧ test2.score = 17/20)
  (h3 : test3.problems = 20 ∧ test3.score = 13/20) :
  (correctProblems test1 + correctProblems test2 + correctProblems test3) /
  (test1.problems + test2.problems + test3.problems) = 39/50 := by
  sorry

end NUMINAMATH_CALUDE_anna_overall_score_l1874_187438


namespace NUMINAMATH_CALUDE_sqrt_x_div_sqrt_y_equals_five_halves_l1874_187436

theorem sqrt_x_div_sqrt_y_equals_five_halves (x y : ℝ) 
  (h : ((2/3)^2 + (1/6)^2) / ((1/2)^2 + (1/7)^2) = 28*x/(25*y)) : 
  Real.sqrt x / Real.sqrt y = 5/2 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_x_div_sqrt_y_equals_five_halves_l1874_187436


namespace NUMINAMATH_CALUDE_arithmetic_geometric_mean_problem_l1874_187485

theorem arithmetic_geometric_mean_problem (x y : ℝ) 
  (h1 : (x + y) / 2 = 20) 
  (h2 : Real.sqrt (x * y) = Real.sqrt 132) : 
  x^2 + y^2 = 1336 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_mean_problem_l1874_187485


namespace NUMINAMATH_CALUDE_max_profit_at_max_price_l1874_187482

/-- Represents the monthly sales and profit of eye-protection lamps --/
structure LampSales where
  cost_price : ℝ
  selling_price : ℝ
  monthly_sales : ℝ
  profit : ℝ

/-- The conditions and constraints of the lamp sales problem --/
def lamp_sales_constraints (s : LampSales) : Prop :=
  s.cost_price = 40 ∧
  s.selling_price ≥ s.cost_price ∧
  s.selling_price ≤ 2 * s.cost_price ∧
  s.monthly_sales = -s.selling_price + 140 ∧
  s.profit = (s.selling_price - s.cost_price) * s.monthly_sales

/-- Theorem stating that the maximum monthly profit is achieved at the highest allowed selling price --/
theorem max_profit_at_max_price (s : LampSales) :
  lamp_sales_constraints s →
  ∃ (max_s : LampSales),
    lamp_sales_constraints max_s ∧
    max_s.selling_price = 80 ∧
    max_s.profit = 2400 ∧
    ∀ (other_s : LampSales), lamp_sales_constraints other_s → other_s.profit ≤ max_s.profit :=
by sorry

end NUMINAMATH_CALUDE_max_profit_at_max_price_l1874_187482


namespace NUMINAMATH_CALUDE_simple_interest_calculation_l1874_187488

/-- Simple interest calculation -/
theorem simple_interest_calculation (interest : ℚ) (rate : ℚ) (time : ℚ) :
  interest = 4016.25 →
  rate = 1 / 100 →
  time = 3 →
  ∃ principal : ℚ, principal = 133875 ∧ interest = principal * rate * time :=
by sorry

end NUMINAMATH_CALUDE_simple_interest_calculation_l1874_187488


namespace NUMINAMATH_CALUDE_consecutive_integers_sum_30_l1874_187415

theorem consecutive_integers_sum_30 : ∃! a : ℕ, ∃ n : ℕ,
  n ≥ 3 ∧ (Finset.range n).sum (λ i => a + i) = 30 :=
by sorry

end NUMINAMATH_CALUDE_consecutive_integers_sum_30_l1874_187415


namespace NUMINAMATH_CALUDE_race_length_is_18_l1874_187498

/-- The length of a cross-country relay race with 5 members -/
def race_length : ℕ :=
  let other_members : ℕ := 4
  let other_distance : ℕ := 3
  let ralph_multiplier : ℕ := 2
  (other_members * other_distance) + (ralph_multiplier * other_distance)

/-- Theorem: The length of the race is 18 km -/
theorem race_length_is_18 : race_length = 18 := by
  sorry

end NUMINAMATH_CALUDE_race_length_is_18_l1874_187498


namespace NUMINAMATH_CALUDE_sum_of_solutions_equation_l1874_187464

theorem sum_of_solutions_equation (x₁ x₂ : ℚ) : 
  (4 * x₁ + 7 = 0 ∨ 5 * x₁ - 8 = 0) ∧
  (4 * x₂ + 7 = 0 ∨ 5 * x₂ - 8 = 0) ∧
  x₁ ≠ x₂ →
  x₁ + x₂ = -3/20 := by sorry

end NUMINAMATH_CALUDE_sum_of_solutions_equation_l1874_187464


namespace NUMINAMATH_CALUDE_polynomial_value_theorem_l1874_187476

/-- Given a polynomial ax³ + bx - 3 where a and b are constants,
    if the value of the polynomial is 15 when x = 2,
    then the value of the polynomial is -21 when x = -2. -/
theorem polynomial_value_theorem (a b : ℝ) : 
  (8 * a + 2 * b - 3 = 15) → (-8 * a - 2 * b - 3 = -21) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_value_theorem_l1874_187476


namespace NUMINAMATH_CALUDE_symmetry_wrt_x_axis_l1874_187497

/-- Given a point A(-2, 3) in a Cartesian coordinate system, 
    its symmetrical point with respect to the x-axis has coordinates (-2, -3). -/
theorem symmetry_wrt_x_axis : 
  let A : ℝ × ℝ := (-2, 3)
  let symmetrical_point : ℝ × ℝ := (-2, -3)
  (∀ (x y : ℝ), (x, y) = A → (x, -y) = symmetrical_point) :=
by sorry

end NUMINAMATH_CALUDE_symmetry_wrt_x_axis_l1874_187497


namespace NUMINAMATH_CALUDE_complement_A_intersect_B_l1874_187469

open Set Real

-- Define set A
def A : Set ℝ := {x | ∃ y, y = Real.sqrt (x - 1)}

-- Define set B
def B : Set ℝ := {x | 2 * x - x^2 > 0}

-- State the theorem
theorem complement_A_intersect_B : (𝒰 \ A) ∩ B = Ioo 0 1 := by sorry

end NUMINAMATH_CALUDE_complement_A_intersect_B_l1874_187469


namespace NUMINAMATH_CALUDE_shaded_area_calculation_l1874_187466

/-- The area of the shaded regions in a figure with two rectangles and two semicircles removed -/
theorem shaded_area_calculation (small_radius : ℝ) (large_radius : ℝ)
  (h_small : small_radius = 3)
  (h_large : large_radius = 6) :
  let small_rect_area := small_radius * (2 * small_radius)
  let large_rect_area := large_radius * (2 * large_radius)
  let small_semicircle_area := π * small_radius^2 / 2
  let large_semicircle_area := π * large_radius^2 / 2
  small_rect_area + large_rect_area - small_semicircle_area - large_semicircle_area = 90 - 45 * π / 2 :=
by sorry

end NUMINAMATH_CALUDE_shaded_area_calculation_l1874_187466


namespace NUMINAMATH_CALUDE_inverse_function_symmetry_l1874_187491

-- Define a function and its inverse
variable (f : ℝ → ℝ) (f_inv : ℝ → ℝ)

-- State that f_inv is the inverse of f
axiom inverse_relation : ∀ x, f (f_inv x) = x ∧ f_inv (f x) = x

-- Define symmetry about the line x - y = 0
def symmetric_about_x_eq_y (f g : ℝ → ℝ) : Prop :=
  ∀ x y, f x = y ↔ g y = x

-- Theorem statement
theorem inverse_function_symmetry :
  symmetric_about_x_eq_y f f_inv :=
sorry

end NUMINAMATH_CALUDE_inverse_function_symmetry_l1874_187491


namespace NUMINAMATH_CALUDE_range_of_f_l1874_187477

noncomputable def f (x : ℝ) : ℝ := Real.arcsin (Real.cos x) + Real.arccos (Real.sin x)

theorem range_of_f :
  Set.range f = Set.Icc 0 Real.pi :=
sorry

end NUMINAMATH_CALUDE_range_of_f_l1874_187477


namespace NUMINAMATH_CALUDE_dividend_calculation_l1874_187449

/-- Calculates the dividend received from an investment in shares -/
theorem dividend_calculation (investment : ℝ) (face_value : ℝ) (premium_rate : ℝ) (dividend_rate : ℝ)
  (h1 : investment = 14400)
  (h2 : face_value = 100)
  (h3 : premium_rate = 0.2)
  (h4 : dividend_rate = 0.07) :
  let share_cost := face_value * (1 + premium_rate)
  let num_shares := investment / share_cost
  let dividend_per_share := face_value * dividend_rate
  num_shares * dividend_per_share = 840 := by sorry

end NUMINAMATH_CALUDE_dividend_calculation_l1874_187449


namespace NUMINAMATH_CALUDE_inequality_implies_upper_bound_l1874_187444

theorem inequality_implies_upper_bound (a : ℝ) :
  (∀ x : ℝ, |x + 2| + |x - 3| > a) → a < 5 := by
  sorry

end NUMINAMATH_CALUDE_inequality_implies_upper_bound_l1874_187444


namespace NUMINAMATH_CALUDE_vegetable_field_division_l1874_187432

theorem vegetable_field_division (total_area : ℚ) (num_parts : ℕ) 
  (h1 : total_area = 5)
  (h2 : num_parts = 8) :
  (1 : ℚ) / num_parts = 1 / 8 ∧ total_area / num_parts = 5 / 8 := by
  sorry

end NUMINAMATH_CALUDE_vegetable_field_division_l1874_187432


namespace NUMINAMATH_CALUDE_emily_spent_twelve_dollars_l1874_187452

/-- The amount Emily spent on flowers -/
def emily_spent (price_per_flower : ℕ) (num_roses : ℕ) (num_daisies : ℕ) : ℕ :=
  price_per_flower * (num_roses + num_daisies)

/-- Theorem: Emily spent 12 dollars on flowers -/
theorem emily_spent_twelve_dollars :
  emily_spent 3 2 2 = 12 := by
  sorry

end NUMINAMATH_CALUDE_emily_spent_twelve_dollars_l1874_187452


namespace NUMINAMATH_CALUDE_function_and_range_l1874_187446

-- Define the function f
def f : ℝ → ℝ := fun x ↦ 3 * x - 2

-- Define the function g
def g : ℝ → ℝ := fun x ↦ x * f x

-- Theorem statement
theorem function_and_range :
  (∀ x : ℝ, f x + 2 * f (-x) = -3 * x - 6) →
  (∀ x : ℝ, f x = 3 * x - 2) ∧
  (Set.Icc 0 3).image g = Set.Icc (-1/3) 21 :=
by sorry

end NUMINAMATH_CALUDE_function_and_range_l1874_187446


namespace NUMINAMATH_CALUDE_chess_group_players_l1874_187451

/-- The number of players in the chess group. -/
def n : ℕ := 20

/-- The total number of games played. -/
def total_games : ℕ := 190

/-- Theorem stating that the number of players is correct given the conditions. -/
theorem chess_group_players :
  (n * (n - 1) / 2 = total_games) ∧
  (∀ m : ℕ, m ≠ n → m * (m - 1) / 2 ≠ total_games) := by
  sorry

#check chess_group_players

end NUMINAMATH_CALUDE_chess_group_players_l1874_187451


namespace NUMINAMATH_CALUDE_pump_fill_time_l1874_187414

/-- The time it takes to fill the tank with the leak present -/
def fill_time_with_leak : ℝ := 10

/-- The time it takes for the leak to empty a full tank -/
def empty_time : ℝ := 10

/-- The time it takes for the pump to fill the tank without the leak -/
def fill_time_without_leak : ℝ := 5

theorem pump_fill_time :
  (1 / fill_time_without_leak - 1 / empty_time = 1 / fill_time_with_leak) →
  fill_time_without_leak = 5 := by
  sorry

end NUMINAMATH_CALUDE_pump_fill_time_l1874_187414


namespace NUMINAMATH_CALUDE_select_president_and_vice_president_l1874_187434

/-- The number of students in the classroom --/
def num_students : ℕ := 4

/-- The number of positions to be filled (president and vice president) --/
def num_positions : ℕ := 2

/-- Theorem stating the number of ways to select a class president and vice president --/
theorem select_president_and_vice_president :
  (num_students * (num_students - 1)) = 12 := by
  sorry

end NUMINAMATH_CALUDE_select_president_and_vice_president_l1874_187434


namespace NUMINAMATH_CALUDE_beatrice_prob_five_given_win_l1874_187416

-- Define the number of players and die sides
def num_players : ℕ := 5
def num_sides : ℕ := 8

-- Define the probability of rolling a specific number
def prob_roll (n : ℕ) : ℚ := 1 / num_sides

-- Define the probability of winning for any player
def prob_win : ℚ := 1 / num_players

-- Define the probability of other players rolling less than 5
def prob_others_less_than_5 : ℚ := (4 / 8) ^ (num_players - 1)

-- Define the probability of winning with a 5 (including tie-breaks)
def prob_win_with_5 : ℚ := prob_others_less_than_5 + 369 / 2048

-- State the theorem
theorem beatrice_prob_five_given_win :
  (prob_roll 5 * prob_win_with_5) / prob_win = 115 / 1024 := by
sorry

end NUMINAMATH_CALUDE_beatrice_prob_five_given_win_l1874_187416


namespace NUMINAMATH_CALUDE_test_scores_theorem_l1874_187486

-- Define the total number of tests
def total_tests : ℕ := 13

-- Define the number of tests with scores exceeding 90
def high_score_tests : ℕ := 4

-- Define the number of tests taken by A and B
def A_tests : ℕ := 6
def B_tests : ℕ := 7

-- Define the number of excellent scores for A and B
def A_excellent : ℕ := 3
def B_excellent : ℕ := 4

-- Define the number of tests selected from A and B
def A_selected : ℕ := 4
def B_selected : ℕ := 3

-- Define the probability of selecting a test with score > 90
def prob_high_score : ℚ := high_score_tests / total_tests

-- Define the expected value of X (excellent scores when selecting 4 out of A's 6 tests)
def E_X : ℚ := 2

-- Define the expected value of Y (excellent scores when selecting 3 out of B's 7 tests)
def E_Y : ℚ := 12 / 7

theorem test_scores_theorem :
  (prob_high_score = 4 / 13) ∧
  (E_X = 2) ∧
  (E_Y = 12 / 7) ∧
  (E_X > E_Y) := by
  sorry

end NUMINAMATH_CALUDE_test_scores_theorem_l1874_187486


namespace NUMINAMATH_CALUDE_largest_a_value_l1874_187472

/-- The equation has at least one integer root -/
def has_integer_root (a : ℤ) : Prop :=
  ∃ x : ℤ, (x^2 - (a+7)*x + 7*a)^(1/3) + 3^(1/3) = 0

/-- 11 is the largest integer value of a for which the equation has at least one integer root -/
theorem largest_a_value : (has_integer_root 11 ∧ ∀ a : ℤ, a > 11 → ¬has_integer_root a) :=
sorry

end NUMINAMATH_CALUDE_largest_a_value_l1874_187472


namespace NUMINAMATH_CALUDE_existence_of_linear_bound_l1874_187492

open Real

noncomputable def f (x : ℝ) : ℝ := (x + 2 * sin x) * (2^(-x) + 1)

theorem existence_of_linear_bound :
  ∃ (a b m : ℝ), ∀ x > 0, |f x - a * x - b| ≤ m :=
sorry

end NUMINAMATH_CALUDE_existence_of_linear_bound_l1874_187492


namespace NUMINAMATH_CALUDE_number_of_goats_l1874_187413

theorem number_of_goats (total_cost cow_price goat_price : ℕ) 
  (h1 : total_cost = 1500)
  (h2 : cow_price = 400)
  (h3 : goat_price = 70) : 
  ∃ (num_goats : ℕ), total_cost = 2 * cow_price + num_goats * goat_price ∧ num_goats = 10 :=
by sorry

end NUMINAMATH_CALUDE_number_of_goats_l1874_187413


namespace NUMINAMATH_CALUDE_unique_modular_residue_l1874_187468

theorem unique_modular_residue :
  ∃! n : ℤ, 0 ≤ n ∧ n < 11 ∧ -1234 ≡ n [ZMOD 11] :=
by sorry

end NUMINAMATH_CALUDE_unique_modular_residue_l1874_187468


namespace NUMINAMATH_CALUDE_fathers_full_time_jobs_l1874_187411

theorem fathers_full_time_jobs (total_parents : ℝ) (h1 : total_parents > 0) : 
  let mothers := 0.4 * total_parents
  let fathers := 0.6 * total_parents
  let mothers_full_time := 0.9 * mothers
  let total_full_time := 0.81 * total_parents
  let fathers_full_time := total_full_time - mothers_full_time
  fathers_full_time / fathers = 3/4 := by sorry

end NUMINAMATH_CALUDE_fathers_full_time_jobs_l1874_187411


namespace NUMINAMATH_CALUDE_floor_ceil_sum_l1874_187495

theorem floor_ceil_sum : ⌊(-1.001 : ℝ)⌋ + ⌈(3.999 : ℝ)⌉ + ⌊(0.998 : ℝ)⌋ = 2 := by
  sorry

end NUMINAMATH_CALUDE_floor_ceil_sum_l1874_187495


namespace NUMINAMATH_CALUDE_digit_129_in_n_or_3n_l1874_187418

/-- Given a natural number, returns true if it contains the digit 1, 2, or 9 in its base-ten representation -/
def containsDigit129 (n : ℕ) : Prop :=
  ∃ d, d ∈ [1, 2, 9] ∧ ∃ k m, n = k * 10 + d + m * 10

theorem digit_129_in_n_or_3n (n : ℕ+) : containsDigit129 n.val ∨ containsDigit129 (3 * n.val) := by
  sorry

end NUMINAMATH_CALUDE_digit_129_in_n_or_3n_l1874_187418


namespace NUMINAMATH_CALUDE_sunday_visitors_theorem_l1874_187423

/-- Represents the average number of visitors on Sundays in a library -/
def average_sunday_visitors (
  total_days : ℕ)  -- Total number of days in the month
  (sunday_count : ℕ)  -- Number of Sundays in the month
  (non_sunday_average : ℕ)  -- Average number of visitors on non-Sundays
  (month_average : ℕ)  -- Average number of visitors per day for the entire month
  : ℕ :=
  ((month_average * total_days) - (non_sunday_average * (total_days - sunday_count))) / sunday_count

/-- Theorem stating that the average number of Sunday visitors is 510 given the problem conditions -/
theorem sunday_visitors_theorem :
  average_sunday_visitors 30 5 240 285 = 510 := by
  sorry

#eval average_sunday_visitors 30 5 240 285

end NUMINAMATH_CALUDE_sunday_visitors_theorem_l1874_187423


namespace NUMINAMATH_CALUDE_average_first_five_subjects_l1874_187447

/-- Given a student's average marks and marks in the last subject, calculate the average of the first 5 subjects -/
theorem average_first_five_subjects 
  (total_subjects : Nat) 
  (average_all : ℚ) 
  (marks_last : ℚ) 
  (h1 : total_subjects = 6) 
  (h2 : average_all = 79) 
  (h3 : marks_last = 104) : 
  (average_all * total_subjects - marks_last) / (total_subjects - 1) = 74 := by
sorry

end NUMINAMATH_CALUDE_average_first_five_subjects_l1874_187447


namespace NUMINAMATH_CALUDE_solve_equation_l1874_187484

theorem solve_equation : ∃ x : ℚ, 3 * x + 15 = (1/3) * (8 * x - 24) ∧ x = -69 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l1874_187484


namespace NUMINAMATH_CALUDE_league_face_count_l1874_187473

/-- The number of games in a single round-robin tournament with n teams -/
def roundRobinGames (n : ℕ) : ℕ := n * (n - 1) / 2

/-- The number of times each team faces another in a league -/
def faceCount (totalTeams : ℕ) (totalGames : ℕ) : ℕ :=
  totalGames / roundRobinGames totalTeams

theorem league_face_count :
  faceCount 14 455 = 5 := by sorry

end NUMINAMATH_CALUDE_league_face_count_l1874_187473


namespace NUMINAMATH_CALUDE_abs_3x_plus_5_not_positive_l1874_187433

theorem abs_3x_plus_5_not_positive (x : ℚ) : ¬(|3*x + 5| > 0) ↔ x = -5/3 := by
  sorry

end NUMINAMATH_CALUDE_abs_3x_plus_5_not_positive_l1874_187433


namespace NUMINAMATH_CALUDE_magical_red_knights_fraction_l1874_187401

theorem magical_red_knights_fraction (total : ℕ) (red : ℕ) (blue : ℕ) (magical : ℕ) 
  (h1 : red = (3 * total) / 7)
  (h2 : blue = total - red)
  (h3 : magical = total / 4)
  (h4 : ∃ (r s : ℕ), (r * blue * 3 = s * red) ∧ (r * red + r * blue = s * magical)) :
  ∃ (r s : ℕ), (r * red = s * magical) ∧ (r = 21 ∧ s = 52) :=
sorry

end NUMINAMATH_CALUDE_magical_red_knights_fraction_l1874_187401


namespace NUMINAMATH_CALUDE_expression_value_l1874_187496

theorem expression_value (x y z : ℝ) 
  (hxy : x ≠ y) (hxz : x ≠ z) (hyz : y ≠ z)
  (hx : x ≠ 2) (hy : y ≠ 3) (hz : z ≠ 5) :
  ((x - 2) / (3 - z) * (y - 3) / (5 - x) * (z - 5) / (2 - y))^2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_expression_value_l1874_187496


namespace NUMINAMATH_CALUDE_arnold_protein_consumption_l1874_187460

/-- Protein content of food items and consumption amounts -/
def collagen_protein : ℕ := 9
def protein_powder_protein : ℕ := 21
def steak_protein : ℕ := 56
def yogurt_protein : ℕ := 15
def almonds_protein : ℕ := 12

def collagen_scoops : ℕ := 1
def protein_powder_scoops : ℕ := 2
def steak_count : ℕ := 1
def yogurt_servings : ℕ := 1
def almonds_cups : ℕ := 1

/-- Total protein consumed by Arnold -/
def total_protein : ℕ :=
  collagen_protein * collagen_scoops +
  protein_powder_protein * protein_powder_scoops +
  steak_protein * steak_count +
  yogurt_protein * yogurt_servings +
  almonds_protein * almonds_cups

/-- Theorem stating that the total protein consumed is 134 grams -/
theorem arnold_protein_consumption : total_protein = 134 := by
  sorry

end NUMINAMATH_CALUDE_arnold_protein_consumption_l1874_187460


namespace NUMINAMATH_CALUDE_expansion_sum_zero_l1874_187479

theorem expansion_sum_zero (n k : ℕ) (a b : ℝ) (h1 : n ≥ 2) (h2 : a * b ≠ 0) (h3 : a = k^2 * b) (h4 : k > 0) :
  (n * (a - b)^(n-1) * (-b) + n * (n-1) / 2 * (a - b)^(n-2) * (-b)^2 = 0) →
  n = 2 * k + 1 :=
by sorry

end NUMINAMATH_CALUDE_expansion_sum_zero_l1874_187479


namespace NUMINAMATH_CALUDE_zero_acceleration_in_quadrant_IV_l1874_187412

-- Define the disk and its properties
structure Disk where
  uniform : Bool
  rolling_smoothly : Bool
  pulled_by_force : Bool

-- Define the acceleration vectors
structure Acceleration where
  tangential : ℝ × ℝ
  centripetal : ℝ × ℝ
  horizontal : ℝ × ℝ

-- Define the quadrants of the disk
inductive Quadrant
  | I
  | II
  | III
  | IV

-- Function to check if a point in a given quadrant can have zero total acceleration
def can_have_zero_acceleration (d : Disk) (q : Quadrant) (a : Acceleration) : Prop :=
  d.uniform ∧ d.rolling_smoothly ∧ d.pulled_by_force ∧
  match q with
  | Quadrant.IV => ∃ (x y : ℝ), 
      x > 0 ∧ y < 0 ∧
      a.tangential.1 + a.centripetal.1 + a.horizontal.1 = 0 ∧
      a.tangential.2 + a.centripetal.2 + a.horizontal.2 = 0
  | _ => False

-- Theorem statement
theorem zero_acceleration_in_quadrant_IV (d : Disk) (a : Acceleration) :
  d.uniform ∧ d.rolling_smoothly ∧ d.pulled_by_force →
  ∃ (q : Quadrant), can_have_zero_acceleration d q a :=
sorry

end NUMINAMATH_CALUDE_zero_acceleration_in_quadrant_IV_l1874_187412


namespace NUMINAMATH_CALUDE_city_distance_proof_l1874_187445

theorem city_distance_proof : 
  ∃ S : ℕ+, 
    (∀ x : ℕ, x ≤ S → (Nat.gcd x (S - x) = 1 ∨ Nat.gcd x (S - x) = 3 ∨ Nat.gcd x (S - x) = 13)) ∧ 
    (∀ T : ℕ+, T < S → ∃ y : ℕ, y ≤ T ∧ Nat.gcd y (T - y) ≠ 1 ∧ Nat.gcd y (T - y) ≠ 3 ∧ Nat.gcd y (T - y) ≠ 13) ∧
    S = 39 :=
by sorry

end NUMINAMATH_CALUDE_city_distance_proof_l1874_187445


namespace NUMINAMATH_CALUDE_condo_rented_units_l1874_187480

/-- Represents the number of units of each bedroom type in a condominium -/
structure CondoUnits where
  one_bedroom : ℕ
  two_bedroom : ℕ
  three_bedroom : ℕ

/-- Represents the number of rented units of each bedroom type in a condominium -/
structure RentedUnits where
  one_bedroom : ℕ
  two_bedroom : ℕ
  three_bedroom : ℕ

def total_units (c : CondoUnits) : ℕ :=
  c.one_bedroom + c.two_bedroom + c.three_bedroom

def total_rented (r : RentedUnits) : ℕ :=
  r.one_bedroom + r.two_bedroom + r.three_bedroom

theorem condo_rented_units 
  (c : CondoUnits)
  (r : RentedUnits)
  (h1 : total_units c = 1200)
  (h2 : total_rented r = 700)
  (h3 : r.one_bedroom * 3 = r.two_bedroom * 2)
  (h4 : r.one_bedroom * 2 = r.three_bedroom)
  (h5 : r.two_bedroom * 2 = c.two_bedroom)
  : c.two_bedroom - r.two_bedroom = 231 := by
  sorry

end NUMINAMATH_CALUDE_condo_rented_units_l1874_187480


namespace NUMINAMATH_CALUDE_shop_width_calculation_l1874_187494

/-- Given a shop with specified rent and dimensions, calculate its width -/
theorem shop_width_calculation (monthly_rent : ℕ) (length : ℕ) (annual_rent_per_sqft : ℕ) : 
  monthly_rent = 2244 →
  length = 22 →
  annual_rent_per_sqft = 68 →
  (monthly_rent * 12) / annual_rent_per_sqft / length = 18 := by
  sorry

#check shop_width_calculation

end NUMINAMATH_CALUDE_shop_width_calculation_l1874_187494


namespace NUMINAMATH_CALUDE_least_palindrome_addition_l1874_187457

/-- A function that checks if a natural number is a palindrome -/
def isPalindrome (n : ℕ) : Prop := sorry

/-- The starting number in our problem -/
def startNumber : ℕ := 250000

/-- The least number to be added to create a palindrome -/
def leastAddition : ℕ := 52

/-- Theorem stating that 52 is the least natural number that,
    when added to 250000, results in a palindrome -/
theorem least_palindrome_addition :
  (∀ k : ℕ, k < leastAddition → ¬isPalindrome (startNumber + k)) ∧
  isPalindrome (startNumber + leastAddition) := by sorry

end NUMINAMATH_CALUDE_least_palindrome_addition_l1874_187457


namespace NUMINAMATH_CALUDE_train_passing_time_l1874_187459

/-- Given a train of length l traveling at constant velocity v, if the time to pass a platform
    of length 3l is 4 times the time to pass a pole, then the time to pass the pole is l/v. -/
theorem train_passing_time
  (l v : ℝ) -- Length of train and velocity
  (h_pos_l : l > 0)
  (h_pos_v : v > 0)
  (t : ℝ) -- Time to pass the pole
  (T : ℝ) -- Time to pass the platform
  (h_platform_time : T = 4 * t) -- Time to pass platform is 4 times time to pass pole
  (h_platform_length : 4 * l = v * T) -- Distance-velocity-time equation for platform
  : t = l / v := by
  sorry

end NUMINAMATH_CALUDE_train_passing_time_l1874_187459


namespace NUMINAMATH_CALUDE_meet_once_l1874_187435

/-- Represents the meeting scenario between Michael and the garbage truck --/
structure MeetingScenario where
  michael_speed : ℝ
  pail_distance : ℝ
  truck_speed : ℝ
  truck_stop_time : ℝ
  initial_distance : ℝ

/-- Calculates the number of meetings between Michael and the truck --/
def number_of_meetings (scenario : MeetingScenario) : ℕ :=
  sorry

/-- The theorem stating that Michael and the truck meet exactly once --/
theorem meet_once (scenario : MeetingScenario) 
  (h1 : scenario.michael_speed = 4)
  (h2 : scenario.pail_distance = 300)
  (h3 : scenario.truck_speed = 6)
  (h4 : scenario.truck_stop_time = 20)
  (h5 : scenario.initial_distance = 300) :
  number_of_meetings scenario = 1 :=
sorry

end NUMINAMATH_CALUDE_meet_once_l1874_187435


namespace NUMINAMATH_CALUDE_midpoint_chain_l1874_187403

/-- Given a line segment XY with midpoints as described, prove that XY = 80 when XJ = 5 -/
theorem midpoint_chain (X Y G H I J : ℝ) : 
  (G = (X + Y) / 2) →  -- G is midpoint of XY
  (H = (X + G) / 2) →  -- H is midpoint of XG
  (I = (X + H) / 2) →  -- I is midpoint of XH
  (J = (X + I) / 2) →  -- J is midpoint of XI
  (J - X = 5) →        -- XJ = 5
  (Y - X = 80) :=      -- XY = 80
by sorry

end NUMINAMATH_CALUDE_midpoint_chain_l1874_187403


namespace NUMINAMATH_CALUDE_concavity_and_inflection_point_l1874_187437

-- Define the function
def f (x : ℝ) : ℝ := x^3 - 6*x^2 + 4

-- Define the second derivative of f
def f'' (x : ℝ) : ℝ := 6*x - 12

-- Theorem stating the concavity and inflection point properties
theorem concavity_and_inflection_point :
  (∀ x < 2, f'' x < 0) ∧
  (∀ x > 2, f'' x > 0) ∧
  f'' 2 = 0 ∧
  f 2 = -12 := by
  sorry

end NUMINAMATH_CALUDE_concavity_and_inflection_point_l1874_187437


namespace NUMINAMATH_CALUDE_not_divisible_by_169_l1874_187456

theorem not_divisible_by_169 (n : ℕ) : ¬(169 ∣ (n^2 + 5*n + 16)) := by
  sorry

end NUMINAMATH_CALUDE_not_divisible_by_169_l1874_187456


namespace NUMINAMATH_CALUDE_stock_investment_fractions_l1874_187430

theorem stock_investment_fractions (initial_investment : ℝ) 
  (final_value : ℝ) (f : ℝ) : 
  initial_investment = 900 →
  final_value = 1350 →
  0 ≤ f →
  f ≤ 1/2 →
  2 * (2 * f * initial_investment) + (1/2 * (1 - 2*f) * initial_investment) = final_value →
  f = 1/3 := by
  sorry

end NUMINAMATH_CALUDE_stock_investment_fractions_l1874_187430


namespace NUMINAMATH_CALUDE_doll_production_theorem_l1874_187400

/-- The number of non-defective dolls produced per day -/
def non_defective_dolls : ℕ := 4800

/-- The ratio of total dolls to non-defective dolls -/
def total_to_non_defective_ratio : ℚ := 133 / 100

/-- The total number of dolls produced per day -/
def total_dolls : ℕ := 6384

/-- Theorem stating the relationship between non-defective dolls, the ratio, and total dolls -/
theorem doll_production_theorem :
  (non_defective_dolls : ℚ) * total_to_non_defective_ratio = total_dolls := by
  sorry

end NUMINAMATH_CALUDE_doll_production_theorem_l1874_187400


namespace NUMINAMATH_CALUDE_fifteen_power_division_l1874_187499

theorem fifteen_power_division : (15 : ℕ) ^ 11 / (15 : ℕ) ^ 8 = 3375 := by
  sorry

end NUMINAMATH_CALUDE_fifteen_power_division_l1874_187499


namespace NUMINAMATH_CALUDE_circle_center_l1874_187427

/-- The center of a circle given by the equation (x-h)^2 + (y-k)^2 = r^2 is (h,k) -/
theorem circle_center (h k r : ℝ) : 
  (∀ x y : ℝ, (x - h)^2 + (y - k)^2 = r^2 ↔ ((x, y) ∈ {p : ℝ × ℝ | (p.1 - h)^2 + (p.2 - k)^2 = r^2})) → 
  (h, k) = (1, 1) → r^2 = 2 →
  (1, 1) ∈ {p : ℝ × ℝ | (p.1 - 1)^2 + (p.2 - 1)^2 = 2} :=
by sorry

end NUMINAMATH_CALUDE_circle_center_l1874_187427


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l1874_187429

theorem arithmetic_sequence_problem :
  ∀ a b c : ℤ,
  (∃ d : ℤ, b = a + d ∧ c = b + d) →  -- arithmetic sequence condition
  a + b + c = 6 →                    -- sum condition
  a * b * c = -10 →                  -- product condition
  ((a = 5 ∧ b = 2 ∧ c = -1) ∨ (a = -1 ∧ b = 2 ∧ c = 5)) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l1874_187429


namespace NUMINAMATH_CALUDE_partition_product_ratio_l1874_187453

theorem partition_product_ratio (n : ℕ) (h : n > 2) :
  ∃ (A B : Finset ℕ), 
    A ∪ B = Finset.range n ∧ 
    A ∩ B = ∅ ∧ 
    max ((A.prod id) / (B.prod id)) ((B.prod id) / (A.prod id)) ≤ (n - 1) / (n - 2) := by
  sorry

end NUMINAMATH_CALUDE_partition_product_ratio_l1874_187453


namespace NUMINAMATH_CALUDE_sticker_distribution_l1874_187407

theorem sticker_distribution (n : ℕ) (k : ℕ) (h1 : n = 10) (h2 : k = 5) :
  (Nat.choose (n + k - 1) (k - 1)) = 1001 := by
  sorry

end NUMINAMATH_CALUDE_sticker_distribution_l1874_187407


namespace NUMINAMATH_CALUDE_largest_four_digit_number_l1874_187475

/-- Represents a digit (0-9) -/
def Digit := Fin 10

/-- Calculates the result of the operation (AAA + BA) * C -/
def calculate (A B C : Digit) : ℕ :=
  (111 * A.val + 10 * B.val + A.val) * C.val

/-- Checks if three digits are all different -/
def allDifferent (A B C : Digit) : Prop :=
  A ≠ B ∧ B ≠ C ∧ A ≠ C

theorem largest_four_digit_number :
  ∃ (A B C : Digit), allDifferent A B C ∧ 
    calculate A B C = 8624 ∧
    (∀ (X Y Z : Digit), allDifferent X Y Z → calculate X Y Z ≤ 8624) :=
sorry

end NUMINAMATH_CALUDE_largest_four_digit_number_l1874_187475


namespace NUMINAMATH_CALUDE_composite_triangle_perimeter_l1874_187405

/-- A triangle composed of four smaller equilateral triangles -/
structure CompositeTriangle where
  /-- The side length of the smaller equilateral triangles -/
  small_side : ℝ
  /-- The perimeter of each smaller equilateral triangle is 9 -/
  small_perimeter : small_side * 3 = 9

/-- The perimeter of the large equilateral triangle -/
def large_perimeter (t : CompositeTriangle) : ℝ :=
  3 * (2 * t.small_side)

/-- Theorem: The perimeter of the large equilateral triangle is 18 -/
theorem composite_triangle_perimeter (t : CompositeTriangle) :
  large_perimeter t = 18 := by
  sorry

end NUMINAMATH_CALUDE_composite_triangle_perimeter_l1874_187405


namespace NUMINAMATH_CALUDE_range_of_a_l1874_187431

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, x ∈ Set.Icc (-1) 2 → a ≥ x^2 - 2*x - 1) → 
  a ≥ 2 := by
sorry

end NUMINAMATH_CALUDE_range_of_a_l1874_187431


namespace NUMINAMATH_CALUDE_concert_attendance_difference_l1874_187420

theorem concert_attendance_difference (first_concert : Nat) (second_concert : Nat)
  (h1 : first_concert = 65899)
  (h2 : second_concert = 66018) :
  second_concert - first_concert = 119 := by
  sorry

end NUMINAMATH_CALUDE_concert_attendance_difference_l1874_187420


namespace NUMINAMATH_CALUDE_max_y_coordinate_of_ellipse_l1874_187470

/-- Represents a point on a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Defines the equation of the ellipse -/
def isOnEllipse (p : Point) : Prop :=
  (p.x - 3)^2 / 49 + (p.y - 4)^2 / 25 = 1

/-- Theorem: The maximum y-coordinate of any point on the given ellipse is 9 -/
theorem max_y_coordinate_of_ellipse :
  ∀ p : Point, isOnEllipse p → p.y ≤ 9 ∧ ∃ q : Point, isOnEllipse q ∧ q.y = 9 :=
by sorry

end NUMINAMATH_CALUDE_max_y_coordinate_of_ellipse_l1874_187470


namespace NUMINAMATH_CALUDE_crane_among_chickens_is_random_l1874_187481

-- Define the type for events
inductive Event
| CoveringSky
| FumingOrifices
| StridingMeteor
| CraneAmongChickens

-- Define the property of being a random event
def isRandomEvent (e : Event) : Prop :=
  ∃ (outcome : Bool), (outcome = true ∨ outcome = false)

-- State the theorem
theorem crane_among_chickens_is_random :
  isRandomEvent Event.CraneAmongChickens :=
sorry

end NUMINAMATH_CALUDE_crane_among_chickens_is_random_l1874_187481


namespace NUMINAMATH_CALUDE_divisibility_by_29_fourth_power_l1874_187439

theorem divisibility_by_29_fourth_power (x y z : ℤ) (S : ℤ) 
  (h1 : S = x^4 + y^4 + z^4) 
  (h2 : 29 ∣ S) : 
  29^4 ∣ S := by
  sorry

end NUMINAMATH_CALUDE_divisibility_by_29_fourth_power_l1874_187439


namespace NUMINAMATH_CALUDE_square_root_equation_solution_l1874_187467

theorem square_root_equation_solution (P : ℝ) :
  Real.sqrt (3 - 2*P) + Real.sqrt (1 - 2*P) = 2 → P = 3/8 := by
  sorry

end NUMINAMATH_CALUDE_square_root_equation_solution_l1874_187467


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l1874_187410

theorem sufficient_not_necessary_condition (x : ℝ) :
  (x = 1 → x^3 = x) ∧ ¬(x^3 = x → x = 1) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_condition_l1874_187410


namespace NUMINAMATH_CALUDE_rational_terms_count_l1874_187463

/-- The number of rational terms in the expansion of (√2 + ∛3)^100 -/
def rational_terms_a : ℕ := 26

/-- The number of rational terms in the expansion of (√2 + ∜3)^300 -/
def rational_terms_b : ℕ := 13

/-- Theorem stating the number of rational terms in the expansions -/
theorem rational_terms_count :
  (rational_terms_a = 26) ∧ (rational_terms_b = 13) := by sorry

end NUMINAMATH_CALUDE_rational_terms_count_l1874_187463


namespace NUMINAMATH_CALUDE_income_comparison_l1874_187474

theorem income_comparison (juan tim mary : ℝ) 
  (h1 : tim = 0.6 * juan) 
  (h2 : mary = 0.84 * juan) : 
  (mary - tim) / tim * 100 = 40 := by
sorry

end NUMINAMATH_CALUDE_income_comparison_l1874_187474


namespace NUMINAMATH_CALUDE_exactlyOneAndTwoBlackMutuallyExclusiveAndNonOpposite_l1874_187489

/-- A pocket containing balls of two colors -/
structure Pocket where
  red : ℕ
  black : ℕ

/-- An event that can occur when selecting balls from a pocket -/
inductive Event
  | ExactlyOneBlack
  | ExactlyTwoBlack

/-- The pocket we're considering in this problem -/
def problemPocket : Pocket := { red := 2, black := 2 }

/-- Two events are mutually exclusive if they cannot occur simultaneously -/
def mutuallyExclusive (e1 e2 : Event) : Prop :=
  ¬(e1 = Event.ExactlyOneBlack ∧ e2 = Event.ExactlyTwoBlack)

/-- Two events are non-opposite if it's possible for neither to occur -/
def nonOpposite (e1 e2 : Event) : Prop :=
  ∃ (outcome : Pocket → Bool), ¬outcome problemPocket

/-- The main theorem stating that ExactlyOneBlack and ExactlyTwoBlack are mutually exclusive and non-opposite -/
theorem exactlyOneAndTwoBlackMutuallyExclusiveAndNonOpposite :
  mutuallyExclusive Event.ExactlyOneBlack Event.ExactlyTwoBlack ∧
  nonOpposite Event.ExactlyOneBlack Event.ExactlyTwoBlack :=
sorry

end NUMINAMATH_CALUDE_exactlyOneAndTwoBlackMutuallyExclusiveAndNonOpposite_l1874_187489


namespace NUMINAMATH_CALUDE_friends_bill_split_l1874_187448

-- Define the problem parameters
def num_friends : ℕ := 5
def original_bill : ℚ := 100
def discount_percentage : ℚ := 6

-- Define the theorem
theorem friends_bill_split :
  let discount := discount_percentage / 100 * original_bill
  let discounted_bill := original_bill - discount
  let individual_payment := discounted_bill / num_friends
  individual_payment = 18.8 := by sorry

end NUMINAMATH_CALUDE_friends_bill_split_l1874_187448


namespace NUMINAMATH_CALUDE_square_equation_solutions_cubic_equation_solution_l1874_187493

-- Part 1
theorem square_equation_solutions (x : ℝ) :
  (x - 2)^2 = 9 ↔ x = 5 ∨ x = -1 := by sorry

-- Part 2
theorem cubic_equation_solution (x : ℝ) :
  27 * (x + 1)^3 + 8 = 0 ↔ x = -5/3 := by sorry

end NUMINAMATH_CALUDE_square_equation_solutions_cubic_equation_solution_l1874_187493


namespace NUMINAMATH_CALUDE_min_container_cost_l1874_187406

def container_cost (a b : ℝ) : ℝ := 20 * (a * b) + 10 * 2 * (a + b)

theorem min_container_cost :
  ∀ a b : ℝ,
  a > 0 → b > 0 →
  a * b = 4 →
  container_cost a b ≥ 160 :=
by
  sorry

end NUMINAMATH_CALUDE_min_container_cost_l1874_187406


namespace NUMINAMATH_CALUDE_arithmetic_geometric_mean_inequality_l1874_187421

theorem arithmetic_geometric_mean_inequality 
  (a b c : ℝ) 
  (ha : a ≥ 0) 
  (hb : b ≥ 0) 
  (hc : c ≥ 0) : 
  (a + b + c) / 3 ≥ (a * b * c) ^ (1/3) :=
sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_mean_inequality_l1874_187421


namespace NUMINAMATH_CALUDE_fraction_evaluation_l1874_187425

theorem fraction_evaluation (a b c : ℝ) (ha : a = 2) (hb : b = 3) (hc : c = 1) :
  6 / (a + b + c) = 1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_evaluation_l1874_187425


namespace NUMINAMATH_CALUDE_common_chord_of_circles_l1874_187442

/-- Given two circles in the xy-plane, this theorem states that
    their common chord lies on a specific line. -/
theorem common_chord_of_circles (x y : ℝ) : 
  (x^2 + y^2 + 2*x = 0) ∧ (x^2 + y^2 - 4*y = 0) → (x + 2*y = 0) := by
  sorry

end NUMINAMATH_CALUDE_common_chord_of_circles_l1874_187442


namespace NUMINAMATH_CALUDE_max_value_implies_m_l1874_187404

-- Define the variables
variable (x y m : ℝ)

-- Define the function z
def z (x y : ℝ) : ℝ := x - 3 * y

-- State the theorem
theorem max_value_implies_m (h1 : y ≥ x) (h2 : x + 3 * y ≤ 4) (h3 : x ≥ m)
  (h4 : ∀ x' y', y' ≥ x' → x' + 3 * y' ≤ 4 → x' ≥ m → z x' y' ≤ 8) 
  (h5 : ∃ x' y', y' ≥ x' ∧ x' + 3 * y' ≤ 4 ∧ x' ≥ m ∧ z x' y' = 8) : m = -4 := by
  sorry

end NUMINAMATH_CALUDE_max_value_implies_m_l1874_187404


namespace NUMINAMATH_CALUDE_geometric_roots_difference_l1874_187428

theorem geometric_roots_difference (m n : ℝ) : 
  (∃ a r : ℝ, a = 1/2 ∧ r > 0 ∧ 
    (∀ x : ℝ, (x^2 - m*x + 2)*(x^2 - n*x + 2) = 0 ↔ 
      x = a ∨ x = a*r ∨ x = a*r^2 ∨ x = a*r^3)) →
  |m - n| = 3/2 := by sorry

end NUMINAMATH_CALUDE_geometric_roots_difference_l1874_187428


namespace NUMINAMATH_CALUDE_perfect_square_consecutive_base_equation_l1874_187402

theorem perfect_square_consecutive_base_equation :
  ∀ (A B : ℕ),
    (∃ n : ℕ, A = n^2) →
    B = A + 1 →
    (1 * A^2 + 2 * A + 3) + (2 * B + 1) = 5 * (A + B) →
    (A : ℝ) + B = 7 + 4 * Real.sqrt 2 :=
by
  sorry

end NUMINAMATH_CALUDE_perfect_square_consecutive_base_equation_l1874_187402


namespace NUMINAMATH_CALUDE_complex_magnitude_problem_l1874_187408

theorem complex_magnitude_problem (z : ℂ) (h : (z - Complex.I) * (1 + Complex.I) = 2 - Complex.I) :
  Complex.abs z = Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_problem_l1874_187408


namespace NUMINAMATH_CALUDE_race_outcomes_l1874_187487

theorem race_outcomes (n : ℕ) (k : ℕ) (h1 : n = 6) (h2 : k = 4) :
  (n.factorial / (n - k).factorial) = 360 := by
  sorry

end NUMINAMATH_CALUDE_race_outcomes_l1874_187487


namespace NUMINAMATH_CALUDE_ant_final_position_l1874_187455

/-- Represents a point in 2D space -/
structure Point where
  x : Int
  y : Int

/-- Represents a direction -/
inductive Direction
  | North
  | East
  | South
  | West

/-- Represents the state of the ant -/
structure AntState where
  position : Point
  direction : Direction
  moveCount : Nat
  moveDistance : Nat

/-- Function to update the ant's state after a move -/
def move (state : AntState) : AntState :=
  match state.direction with
  | Direction.North => { state with position := ⟨state.position.x, state.position.y + state.moveDistance⟩, direction := Direction.East }
  | Direction.East => { state with position := ⟨state.position.x + state.moveDistance, state.position.y⟩, direction := Direction.South }
  | Direction.South => { state with position := ⟨state.position.x, state.position.y - state.moveDistance⟩, direction := Direction.West }
  | Direction.West => { state with position := ⟨state.position.x - state.moveDistance, state.position.y⟩, direction := Direction.North }

/-- Function to perform multiple moves -/
def multiMove (initialState : AntState) (n : Nat) : AntState :=
  match n with
  | 0 => initialState
  | m + 1 => 
    let newState := move initialState
    multiMove { newState with moveCount := newState.moveCount + 1, moveDistance := newState.moveDistance + 2 } m

/-- Theorem stating the final position of the ant -/
theorem ant_final_position :
  let initialState : AntState := {
    position := ⟨10, -10⟩,
    direction := Direction.North,
    moveCount := 0,
    moveDistance := 2
  }
  let finalState := multiMove initialState 10
  finalState.position = ⟨22, 0⟩ := by
  sorry


end NUMINAMATH_CALUDE_ant_final_position_l1874_187455


namespace NUMINAMATH_CALUDE_addition_subtraction_elimination_not_factorization_l1874_187465

/-- Represents a mathematical method --/
inductive Method
  | TakingOutCommonFactor
  | CrossMultiplication
  | Formula
  | AdditionSubtractionElimination

/-- Predicate to determine if a method is a factorization method --/
def IsFactorizationMethod (m : Method) : Prop :=
  m = Method.TakingOutCommonFactor ∨ 
  m = Method.CrossMultiplication ∨ 
  m = Method.Formula

theorem addition_subtraction_elimination_not_factorization :
  ¬(IsFactorizationMethod Method.AdditionSubtractionElimination) :=
by sorry

end NUMINAMATH_CALUDE_addition_subtraction_elimination_not_factorization_l1874_187465


namespace NUMINAMATH_CALUDE_impossible_equal_sum_arrangement_l1874_187458

theorem impossible_equal_sum_arrangement : ¬∃ (a b c d e f : ℕ),
  (a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧
   b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧
   c ≠ d ∧ c ≠ e ∧ c ≠ f ∧
   d ≠ e ∧ d ≠ f ∧
   e ≠ f) ∧
  (∃ (s : ℕ), 
    a + b + c = s ∧
    a + d + e = s ∧
    b + d + f = s ∧
    c + e + f = s) :=
by sorry

end NUMINAMATH_CALUDE_impossible_equal_sum_arrangement_l1874_187458


namespace NUMINAMATH_CALUDE_blue_paint_calculation_l1874_187443

/-- Given a paint mixture with a ratio of blue to green paint and a total number of cans,
    calculate the number of cans of blue paint required. -/
def blue_paint_cans (blue_ratio green_ratio total_cans : ℕ) : ℕ :=
  (blue_ratio * total_cans) / (blue_ratio + green_ratio)

/-- Theorem stating that for a 4:3 ratio of blue to green paint and 42 total cans,
    24 cans of blue paint are required. -/
theorem blue_paint_calculation :
  blue_paint_cans 4 3 42 = 24 := by
  sorry

end NUMINAMATH_CALUDE_blue_paint_calculation_l1874_187443


namespace NUMINAMATH_CALUDE_rectangle_triangle_equal_area_l1874_187419

/-- The width of a rectangle whose area is equal to the area of a triangle with base 16 and height equal to the rectangle's length -/
theorem rectangle_triangle_equal_area (x : ℝ) (y : ℝ) 
  (h : x * y = (1/2) * 16 * x) : y = 8 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_triangle_equal_area_l1874_187419


namespace NUMINAMATH_CALUDE_max_value_expression_l1874_187454

theorem max_value_expression (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (x^2 * y^2 * z^2 * (x^2 + y^2 + z^2)) / ((x + y)^3 * (y + z)^3) ≤ 1/24 ∧
  ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧
    (a^2 * b^2 * c^2 * (a^2 + b^2 + c^2)) / ((a + b)^3 * (b + c)^3) = 1/24 :=
by sorry

end NUMINAMATH_CALUDE_max_value_expression_l1874_187454
