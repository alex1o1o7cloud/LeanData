import Mathlib

namespace NUMINAMATH_CALUDE_multiply_mixed_number_l2466_246606

theorem multiply_mixed_number : 7 * (9 + 2/5) = 65 + 4/5 := by
  sorry

end NUMINAMATH_CALUDE_multiply_mixed_number_l2466_246606


namespace NUMINAMATH_CALUDE_ribbon_gifts_l2466_246636

theorem ribbon_gifts (total_ribbon : ℕ) (ribbon_per_gift : ℕ) (remaining_ribbon : ℕ) : 
  total_ribbon = 18 ∧ ribbon_per_gift = 2 ∧ remaining_ribbon = 6 →
  (total_ribbon - remaining_ribbon) / ribbon_per_gift = 6 :=
by sorry

end NUMINAMATH_CALUDE_ribbon_gifts_l2466_246636


namespace NUMINAMATH_CALUDE_linear_function_not_in_quadrant_II_l2466_246633

/-- Represents a linear function y = mx + b -/
structure LinearFunction where
  m : ℝ  -- slope
  b : ℝ  -- y-intercept

/-- Checks if a point (x, y) is in Quadrant II -/
def isInQuadrantII (x : ℝ) (y : ℝ) : Prop :=
  x < 0 ∧ y > 0

/-- Theorem: The linear function y = 3x - 2 does not pass through Quadrant II -/
theorem linear_function_not_in_quadrant_II :
  let f : LinearFunction := { m := 3, b := -2 }
  ∀ x y : ℝ, y = f.m * x + f.b → ¬(isInQuadrantII x y) :=
by
  sorry


end NUMINAMATH_CALUDE_linear_function_not_in_quadrant_II_l2466_246633


namespace NUMINAMATH_CALUDE_min_value_of_sum_l2466_246604

theorem min_value_of_sum (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h : x * y * z = 8) :
  x + 3 * y + 5 * z ≥ 14 * (40 / 3) ^ (1 / 3) :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_sum_l2466_246604


namespace NUMINAMATH_CALUDE_second_place_score_l2466_246681

/-- Represents a player in the chess tournament -/
structure Player where
  score : ℕ

/-- Represents a chess tournament -/
structure ChessTournament where
  players : Finset Player
  secondPlace : Player
  lastFour : Finset Player

/-- The rules and conditions of the tournament -/
def TournamentRules (t : ChessTournament) : Prop :=
  -- 8 players in total
  t.players.card = 8 ∧
  -- Second place player is in the set of all players
  t.secondPlace ∈ t.players ∧
  -- Last four players are in the set of all players
  t.lastFour ⊆ t.players ∧
  -- Last four players are distinct and have 4 members
  t.lastFour.card = 4 ∧
  -- All scores are different
  ∀ p1 p2 : Player, p1 ∈ t.players → p2 ∈ t.players → p1 ≠ p2 → p1.score ≠ p2.score ∧
  -- Second place score equals sum of last four scores
  t.secondPlace.score = (t.lastFour.toList.map Player.score).sum ∧
  -- Maximum possible score is 14
  ∀ p : Player, p ∈ t.players → p.score ≤ 14

/-- The main theorem -/
theorem second_place_score (t : ChessTournament) :
  TournamentRules t → t.secondPlace.score = 12 := by sorry

end NUMINAMATH_CALUDE_second_place_score_l2466_246681


namespace NUMINAMATH_CALUDE_rectangle_area_error_percent_l2466_246611

/-- Given a rectangle with actual length L and width W, if the measured length is 1.09L
    and the measured width is 0.92W, then the error percent in the calculated area
    compared to the actual area is 0.28%. -/
theorem rectangle_area_error_percent (L W : ℝ) (L_pos : L > 0) (W_pos : W > 0) :
  let measured_length := 1.09 * L
  let measured_width := 0.92 * W
  let actual_area := L * W
  let calculated_area := measured_length * measured_width
  let error_percent := (calculated_area - actual_area) / actual_area * 100
  error_percent = 0.28 := by sorry

end NUMINAMATH_CALUDE_rectangle_area_error_percent_l2466_246611


namespace NUMINAMATH_CALUDE_new_difference_greater_than_original_l2466_246649

theorem new_difference_greater_than_original
  (x y a b : ℝ)
  (h_x_pos : x > 0)
  (h_y_pos : y > 0)
  (h_a_pos : a > 0)
  (h_b_pos : b > 0)
  (h_x_gt_y : x > y)
  (h_a_neq_b : a ≠ b) :
  (x + a) - (y - b) > x - y :=
sorry

end NUMINAMATH_CALUDE_new_difference_greater_than_original_l2466_246649


namespace NUMINAMATH_CALUDE_product_of_powers_l2466_246612

theorem product_of_powers (n : ℕ) : (500 ^ 50) * (2 ^ 100) = 10 ^ 75 := by
  sorry

end NUMINAMATH_CALUDE_product_of_powers_l2466_246612


namespace NUMINAMATH_CALUDE_problem_solution_l2466_246695

def f (x : ℝ) := |2*x - 7| + 1

def g (x : ℝ) := f x - 2*|x - 1|

theorem problem_solution :
  (∀ x : ℝ, f x ≤ x ↔ (8/3 ≤ x ∧ x ≤ 6)) ∧
  (∀ x : ℝ, g x ≥ -4) ∧
  (∀ a : ℝ, (∃ x : ℝ, g x ≤ a) ↔ a ≥ -4) := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2466_246695


namespace NUMINAMATH_CALUDE_least_positive_angle_l2466_246650

theorem least_positive_angle (x : Real) (a b : Real) : 
  Real.tan x = a / b →
  Real.tan (2 * x) = b / (a + b) →
  Real.tan (3 * x) = (a - b) / (a + b) →
  ∃ k, k = 13 / 9 ∧ x = Real.arctan k ∧ 
    ∀ y, y > 0 → Real.tan y = a / b → Real.tan (2 * y) = b / (a + b) → 
    Real.tan (3 * y) = (a - b) / (a + b) → y ≥ x :=
by sorry

end NUMINAMATH_CALUDE_least_positive_angle_l2466_246650


namespace NUMINAMATH_CALUDE_travel_ways_theorem_l2466_246624

/-- Represents the number of transportation options between two cities -/
structure TransportOptions where
  buses : Nat
  trains : Nat
  ferries : Nat

/-- The total number of ways to travel between two cities -/
def total_ways (options : TransportOptions) : Nat :=
  options.buses + options.trains + options.ferries

theorem travel_ways_theorem (ab_morning : TransportOptions) (bc_afternoon : TransportOptions)
  (h1 : ab_morning.buses = 5)
  (h2 : ab_morning.trains = 2)
  (h3 : ab_morning.ferries = 0)
  (h4 : bc_afternoon.buses = 3)
  (h5 : bc_afternoon.trains = 0)
  (h6 : bc_afternoon.ferries = 2) :
  (total_ways ab_morning) * (total_ways bc_afternoon) = 35 := by
  sorry

#check travel_ways_theorem

end NUMINAMATH_CALUDE_travel_ways_theorem_l2466_246624


namespace NUMINAMATH_CALUDE_class_artworks_l2466_246666

/-- Represents the number of artworks created by a class of students -/
def total_artworks (num_students : ℕ) (artworks_group1 : ℕ) (artworks_group2 : ℕ) : ℕ :=
  (num_students / 2) * artworks_group1 + (num_students / 2) * artworks_group2

/-- Theorem stating that a class of 10 students, where half make 3 artworks and half make 4, creates 35 artworks in total -/
theorem class_artworks : total_artworks 10 3 4 = 35 := by
  sorry

end NUMINAMATH_CALUDE_class_artworks_l2466_246666


namespace NUMINAMATH_CALUDE_ticket_sales_l2466_246696

theorem ticket_sales (total : ℕ) (full_price : ℕ) (reduced_price : ℕ) :
  total = 25200 →
  full_price = 16500 →
  full_price = 5 * reduced_price →
  reduced_price = 3300 := by
sorry

end NUMINAMATH_CALUDE_ticket_sales_l2466_246696


namespace NUMINAMATH_CALUDE_imaginary_number_theorem_l2466_246693

theorem imaginary_number_theorem (z : ℂ) :
  (∃ a : ℝ, z = a * I) →
  ((z + 2) / (1 - I)).im = 0 →
  z = -2 * I :=
by sorry

end NUMINAMATH_CALUDE_imaginary_number_theorem_l2466_246693


namespace NUMINAMATH_CALUDE_cricket_game_remaining_overs_l2466_246669

def cricket_game (total_overs : ℕ) (target_runs : ℕ) (initial_overs : ℕ) (initial_run_rate : ℚ) : Prop :=
  let runs_scored := initial_run_rate * initial_overs
  let remaining_runs := target_runs - runs_scored
  let remaining_overs := total_overs - initial_overs
  remaining_overs = 40

theorem cricket_game_remaining_overs :
  cricket_game 50 282 10 (32/10) :=
sorry

end NUMINAMATH_CALUDE_cricket_game_remaining_overs_l2466_246669


namespace NUMINAMATH_CALUDE_apples_basket_value_l2466_246615

/-- Given a total number of apples, number of baskets, and price per apple,
    calculates the value of apples in one basket. -/
def value_of_basket (total_apples : ℕ) (num_baskets : ℕ) (price_per_apple : ℕ) : ℕ :=
  (total_apples / num_baskets) * price_per_apple

/-- Theorem stating that the value of apples in one basket is 6000 won
    given the specific conditions of the problem. -/
theorem apples_basket_value :
  value_of_basket 180 6 200 = 6000 := by
  sorry

end NUMINAMATH_CALUDE_apples_basket_value_l2466_246615


namespace NUMINAMATH_CALUDE_log_problem_l2466_246697

-- Define the logarithm function
noncomputable def log : ℝ → ℝ := Real.log

-- Define the conditions and the theorem
theorem log_problem (x y : ℝ) (h1 : log (x * y^5) = 1) (h2 : log (x^3 * y) = 1) :
  log (x^2 * y^2) = 6/7 := by
  sorry

end NUMINAMATH_CALUDE_log_problem_l2466_246697


namespace NUMINAMATH_CALUDE_notebook_savings_theorem_l2466_246622

/-- Calculates the savings when buying notebooks on sale compared to regular price -/
def calculate_savings (original_price : ℚ) (regular_quantity : ℕ) (sale_quantity : ℕ) 
  (sale_discount : ℚ) (extra_discount : ℚ) : ℚ :=
  let regular_cost := original_price * regular_quantity
  let discounted_price := original_price * (1 - sale_discount)
  let sale_cost := if sale_quantity > 10
    then discounted_price * sale_quantity * (1 - extra_discount)
    else discounted_price * sale_quantity
  regular_cost - sale_cost

theorem notebook_savings_theorem : 
  let original_price : ℚ := 3
  let regular_quantity : ℕ := 8
  let sale_quantity : ℕ := 12
  let sale_discount : ℚ := 1/4
  let extra_discount : ℚ := 1/20
  calculate_savings original_price regular_quantity sale_quantity sale_discount extra_discount = 10.35 := by
  sorry

end NUMINAMATH_CALUDE_notebook_savings_theorem_l2466_246622


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l2466_246690

def M : Set ℝ := {x | x^2 < 4}
def N : Set ℝ := {x | x^2 - 2*x - 3 < 0}

theorem intersection_of_M_and_N :
  M ∩ N = {x : ℝ | -1 < x ∧ x < 2} := by sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l2466_246690


namespace NUMINAMATH_CALUDE_skittles_given_to_karen_l2466_246610

/-- The number of Skittles Pamela initially had -/
def initial_skittles : ℕ := 50

/-- The number of Skittles Pamela has now -/
def remaining_skittles : ℕ := 43

/-- The number of Skittles Pamela gave to Karen -/
def skittles_given : ℕ := initial_skittles - remaining_skittles

theorem skittles_given_to_karen : skittles_given = 7 := by
  sorry

end NUMINAMATH_CALUDE_skittles_given_to_karen_l2466_246610


namespace NUMINAMATH_CALUDE_only_D_is_simple_random_sample_l2466_246628

/-- Represents a sampling method --/
inductive SamplingMethod
| A  -- Every 1 million postcards form a lottery group
| B  -- Sample a package every 30 minutes
| C  -- Draw from different staff categories
| D  -- Select 3 out of 10 products randomly

/-- Defines what constitutes a simple random sample --/
def isSimpleRandomSample (method : SamplingMethod) : Prop :=
  match method with
  | SamplingMethod.A => false  -- Fixed interval
  | SamplingMethod.B => false  -- Fixed interval
  | SamplingMethod.C => false  -- Stratified sampling
  | SamplingMethod.D => true   -- Equal probability for each item

/-- Theorem stating that only method D is a simple random sample --/
theorem only_D_is_simple_random_sample :
  ∀ (method : SamplingMethod), isSimpleRandomSample method ↔ method = SamplingMethod.D :=
by sorry

end NUMINAMATH_CALUDE_only_D_is_simple_random_sample_l2466_246628


namespace NUMINAMATH_CALUDE_alternating_sequence_sum_l2466_246663

def alternating_sequence (n : ℕ) : ℤ := 
  if n % 2 = 0 then (n : ℤ) else -((n + 1) : ℤ)

def sequence_sum (n : ℕ) : ℤ := 
  (List.range n).map alternating_sequence |>.sum

theorem alternating_sequence_sum : sequence_sum 50 = 25 := by
  sorry

end NUMINAMATH_CALUDE_alternating_sequence_sum_l2466_246663


namespace NUMINAMATH_CALUDE_average_difference_l2466_246653

theorem average_difference (a b c : ℝ) 
  (hab : (a + b) / 2 = 80) 
  (hbc : (b + c) / 2 = 180) : 
  a - c = -200 := by sorry

end NUMINAMATH_CALUDE_average_difference_l2466_246653


namespace NUMINAMATH_CALUDE_dannys_age_l2466_246675

/-- Proves Danny's current age given Jane's age and their age relationship 19 years ago -/
theorem dannys_age (jane_age : ℕ) (h1 : jane_age = 26) 
  (h2 : ∃ (danny_age : ℕ), danny_age - 19 = 3 * (jane_age - 19)) : 
  ∃ (danny_age : ℕ), danny_age = 40 := by
  sorry

end NUMINAMATH_CALUDE_dannys_age_l2466_246675


namespace NUMINAMATH_CALUDE_same_solution_equations_l2466_246644

theorem same_solution_equations (x c : ℝ) : 
  (3 * x + 11 = 5) ∧ (c * x - 14 = -4) → c = -5 := by
  sorry

end NUMINAMATH_CALUDE_same_solution_equations_l2466_246644


namespace NUMINAMATH_CALUDE_motel_room_rate_l2466_246694

theorem motel_room_rate (total_rent : ℕ) (lower_rate : ℕ) (reduction_percentage : ℚ) 
  (num_rooms_changed : ℕ) (h1 : total_rent = 2000) (h2 : lower_rate = 40) 
  (h3 : reduction_percentage = 1/10) (h4 : num_rooms_changed = 10) : 
  ∃ (higher_rate : ℕ), 
    (∃ (num_lower_rooms num_higher_rooms : ℕ), 
      total_rent = lower_rate * num_lower_rooms + higher_rate * num_higher_rooms ∧
      total_rent - (reduction_percentage * total_rent) = 
        lower_rate * (num_lower_rooms + num_rooms_changed) + 
        higher_rate * (num_higher_rooms - num_rooms_changed)) ∧
    higher_rate = 60 := by
  sorry

end NUMINAMATH_CALUDE_motel_room_rate_l2466_246694


namespace NUMINAMATH_CALUDE_ellipse_sum_range_l2466_246699

theorem ellipse_sum_range (x y : ℝ) (h : x^2/16 + y^2/9 = 1) :
  ∃ (z : ℝ), z = x + y ∧ -5 ≤ z ∧ z ≤ 5 :=
sorry

end NUMINAMATH_CALUDE_ellipse_sum_range_l2466_246699


namespace NUMINAMATH_CALUDE_cubic_function_property_l2466_246688

theorem cubic_function_property (a b : ℝ) :
  let f : ℝ → ℝ := λ x => a * x^3 + b * x - 4
  f 2 = 6 → f (-2) = -14 := by
  sorry

end NUMINAMATH_CALUDE_cubic_function_property_l2466_246688


namespace NUMINAMATH_CALUDE_complement_P_in_U_l2466_246626

-- Define the sets U and P
def U : Set ℝ := {y | ∃ x > 1, y = Real.log x / Real.log 2}
def P : Set ℝ := {y | ∃ x > 2, y = 1 / x}

-- State the theorem
theorem complement_P_in_U : 
  (U \ P) = {y | y ∈ Set.Ici (1/2)} := by sorry

end NUMINAMATH_CALUDE_complement_P_in_U_l2466_246626


namespace NUMINAMATH_CALUDE_y_derivative_l2466_246656

open Real

noncomputable def y (x : ℝ) : ℝ := 
  (sin (tan (1/7)) * (cos (16*x))^2) / (32 * sin (32*x))

theorem y_derivative (x : ℝ) : 
  deriv y x = -sin (tan (1/7)) / (4 * (sin (16*x))^2) :=
sorry

end NUMINAMATH_CALUDE_y_derivative_l2466_246656


namespace NUMINAMATH_CALUDE_expression_simplification_l2466_246651

theorem expression_simplification (x : ℝ) :
  4 * x^3 + 5 * x + 9 - (3 * x^3 - 2 * x + 1) + 2 * x^2 - (x^2 - 4 * x - 6) =
  x^3 + x^2 + 11 * x + 14 := by
  sorry

end NUMINAMATH_CALUDE_expression_simplification_l2466_246651


namespace NUMINAMATH_CALUDE_ball_drawing_properties_l2466_246661

/-- Represents the number of balls of each color in the bag -/
structure BagContents where
  red : Nat
  white : Nat
  white_gt_3 : white > 3

/-- Represents the possible outcomes when drawing two balls -/
inductive DrawOutcome
  | SameColor
  | DifferentColors

/-- Calculates the probability of an outcome given the bag contents -/
def probability (b : BagContents) (o : DrawOutcome) : Rat :=
  sorry

/-- Calculates the probability of drawing at least one red ball -/
def probabilityAtLeastOneRed (b : BagContents) : Rat :=
  sorry

theorem ball_drawing_properties (n : Nat) (h : n > 3) :
  let b : BagContents := ⟨3, n, h⟩
  -- Events "same color" and "different colors" are mutually exclusive
  (probability b DrawOutcome.SameColor + probability b DrawOutcome.DifferentColors = 1) ∧
  -- When P(SameColor) = P(DifferentColors), P(AtLeastOneRed) = 7/12
  (probability b DrawOutcome.SameColor = probability b DrawOutcome.DifferentColors →
   probabilityAtLeastOneRed b = 7/12) :=
by
  sorry

end NUMINAMATH_CALUDE_ball_drawing_properties_l2466_246661


namespace NUMINAMATH_CALUDE_statements_about_positive_numbers_l2466_246630

theorem statements_about_positive_numbers (a b : ℝ) 
  (ha : a > 0) (hb : b > 0) (hab : a ≠ b) : 
  (a > b → a^2 > b^2) ∧ 
  ((2 * a * b) / (a + b) < (a + b) / 2) ∧ 
  ((a^2 + b^2) / 2 > ((a + b) / 2)^2) := by
  sorry

end NUMINAMATH_CALUDE_statements_about_positive_numbers_l2466_246630


namespace NUMINAMATH_CALUDE_arithmetic_mean_problem_l2466_246641

theorem arithmetic_mean_problem (original_list : List ℝ) (x y z : ℝ) :
  original_list.length = 15 →
  original_list.sum / original_list.length = 70 →
  (original_list.sum + x + y + z) / (original_list.length + 3) = 80 →
  (x + y + z) / 3 = 130 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_mean_problem_l2466_246641


namespace NUMINAMATH_CALUDE_paradise_park_capacity_l2466_246602

/-- A Ferris wheel in paradise park -/
structure FerrisWheel where
  total_seats : ℕ
  broken_seats : ℕ
  people_per_seat : ℕ

/-- The capacity of a Ferris wheel is the number of people it can hold on functioning seats -/
def FerrisWheel.capacity (fw : FerrisWheel) : ℕ :=
  (fw.total_seats - fw.broken_seats) * fw.people_per_seat

/-- The paradise park with its three Ferris wheels -/
def paradise_park : List FerrisWheel :=
  [{ total_seats := 18, broken_seats := 10, people_per_seat := 15 },
   { total_seats := 25, broken_seats := 7,  people_per_seat := 15 },
   { total_seats := 30, broken_seats := 12, people_per_seat := 15 }]

/-- The total capacity of all Ferris wheels in paradise park -/
def total_park_capacity : ℕ :=
  (paradise_park.map FerrisWheel.capacity).sum

theorem paradise_park_capacity :
  total_park_capacity = 660 := by
  sorry

end NUMINAMATH_CALUDE_paradise_park_capacity_l2466_246602


namespace NUMINAMATH_CALUDE_camp_boys_percentage_l2466_246618

theorem camp_boys_percentage (total : ℕ) (added_boys : ℕ) (girl_percentage : ℚ) : 
  total = 60 →
  added_boys = 60 →
  girl_percentage = 5 / 100 →
  (girl_percentage * (total + added_boys) : ℚ) = (total - (9 * total / 10) : ℚ) →
  (9 * total / 10 : ℚ) / total = 9 / 10 :=
by sorry

end NUMINAMATH_CALUDE_camp_boys_percentage_l2466_246618


namespace NUMINAMATH_CALUDE_angle_inequality_l2466_246647

theorem angle_inequality (x y : Real) (h1 : x ≤ 90 * Real.pi / 180) (h2 : Real.sin y = 3/4 * Real.sin x) : y > x/2 := by
  sorry

end NUMINAMATH_CALUDE_angle_inequality_l2466_246647


namespace NUMINAMATH_CALUDE_range_of_a_l2466_246646

open Real

noncomputable def f (a x : ℝ) : ℝ := exp x * (2 * x - 1) - 2 * a * x + 2 * a

theorem range_of_a (a : ℝ) :
  (a < 1) →
  (∃! (x₀ : ℤ), f a (x₀ : ℝ) < 0) →
  a ∈ Set.Icc (3 / (4 * exp 1)) (1 / 2) :=
sorry

end NUMINAMATH_CALUDE_range_of_a_l2466_246646


namespace NUMINAMATH_CALUDE_two_numbers_equation_l2466_246677

theorem two_numbers_equation (α β : ℝ) : 
  (α + β) / 2 = 8 → 
  Real.sqrt (α * β) = 15 → 
  ∃ (x : ℝ), x^2 - 16*x + 225 = 0 ∧ (x = α ∨ x = β) := by
  sorry

end NUMINAMATH_CALUDE_two_numbers_equation_l2466_246677


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l2466_246680

def A : Set ℝ := {x | -1 ≤ x ∧ x ≤ 1}
def B : Set ℝ := {-1, 0, 2}

theorem intersection_of_A_and_B : A ∩ B = {-1, 0} := by sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l2466_246680


namespace NUMINAMATH_CALUDE_binomial_coefficient_problem_l2466_246691

theorem binomial_coefficient_problem (m : ℝ) (n : ℕ) :
  (∃ k : ℕ, (Nat.choose n 3) * m^3 = 160 ∧ n = 6) → m = 2 := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_problem_l2466_246691


namespace NUMINAMATH_CALUDE_complex_number_modulus_l2466_246639

theorem complex_number_modulus (b : ℝ) : 
  let z : ℂ := (3 - b * Complex.I) / (2 + Complex.I)
  (z.re = z.im) → Complex.abs z = 3 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_modulus_l2466_246639


namespace NUMINAMATH_CALUDE_perception_arrangements_l2466_246683

def word : String := "PERCEPTION"

theorem perception_arrangements :
  (word.length = 10) →
  (word.count 'P' = 2) →
  (word.count 'E' = 2) →
  (word.count 'I' = 2) →
  (word.count 'C' = 1) →
  (word.count 'T' = 1) →
  (word.count 'O' = 1) →
  (word.count 'N' = 1) →
  (Nat.factorial 10 / (Nat.factorial 2 * Nat.factorial 2 * Nat.factorial 2) = 453600) :=
by
  sorry

end NUMINAMATH_CALUDE_perception_arrangements_l2466_246683


namespace NUMINAMATH_CALUDE_roots_and_minimum_value_l2466_246654

def f (a : ℝ) (x : ℝ) : ℝ := |x^2 - x| - a*x

theorem roots_and_minimum_value :
  (∀ x, f (1/3) x = 0 ↔ x = 0 ∨ x = 2/3 ∨ x = 4/3) ∧
  (∀ a, a ≤ -1 →
    (∀ x ∈ Set.Icc (-2) 3, f a x ≥ 
      (if a ≤ -5 then 2*a + 6 else -(a+1)^2/4)) ∧
    (∃ x ∈ Set.Icc (-2) 3, f a x = 
      (if a ≤ -5 then 2*a + 6 else -(a+1)^2/4))) := by
  sorry

end NUMINAMATH_CALUDE_roots_and_minimum_value_l2466_246654


namespace NUMINAMATH_CALUDE_average_income_b_and_c_l2466_246621

/-- Proves that given the conditions, the average monthly income of B and C is 6250 --/
theorem average_income_b_and_c (income_a income_b income_c : ℝ) : 
  (income_a + income_b) / 2 = 5050 →
  (income_a + income_c) / 2 = 5200 →
  income_a = 4000 →
  (income_b + income_c) / 2 = 6250 := by
sorry

end NUMINAMATH_CALUDE_average_income_b_and_c_l2466_246621


namespace NUMINAMATH_CALUDE_min_a_for_simplest_quadratic_root_l2466_246671

-- Define the property of being the simplest quadratic root
def is_simplest_quadratic_root (x : ℝ) : Prop :=
  ∃ (n : ℕ), x = Real.sqrt n ∧ ∀ (m : ℕ), m < n → ¬(∃ (q : ℚ), q * q = m)

-- Define the main theorem
theorem min_a_for_simplest_quadratic_root :
  ∃ (a : ℤ), (∀ (b : ℤ), is_simplest_quadratic_root (Real.sqrt (3 * b + 1)) → a ≤ b) ∧
             is_simplest_quadratic_root (Real.sqrt (3 * a + 1)) ∧
             a = 2 :=
sorry

end NUMINAMATH_CALUDE_min_a_for_simplest_quadratic_root_l2466_246671


namespace NUMINAMATH_CALUDE_sum_of_two_squares_equivalence_l2466_246629

theorem sum_of_two_squares_equivalence (x : ℤ) :
  (∃ a b : ℤ, x = a^2 + b^2) ↔ (∃ u v : ℤ, 2*x = u^2 + v^2) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_two_squares_equivalence_l2466_246629


namespace NUMINAMATH_CALUDE_circle_C_radius_l2466_246608

-- Define the circle C
def Circle_C : Set (ℝ × ℝ) := sorry

-- Define points A and B
def A : ℝ × ℝ := (-2, 0)
def B : ℝ × ℝ := (-1, 1)

-- Define the tangent line
def tangent_line (x y : ℝ) : Prop := 3 * x - 4 * y + 7 = 0

-- State that A is on the circle
axiom A_on_circle : A ∈ Circle_C

-- State that B is on the circle and the tangent line
axiom B_on_circle : B ∈ Circle_C
axiom B_on_tangent : tangent_line B.1 B.2

-- Define the radius of the circle
def radius (c : Set (ℝ × ℝ)) : ℝ := sorry

-- Theorem statement
theorem circle_C_radius : radius Circle_C = 5 := by sorry

end NUMINAMATH_CALUDE_circle_C_radius_l2466_246608


namespace NUMINAMATH_CALUDE_least_number_divisible_by_five_primes_l2466_246674

theorem least_number_divisible_by_five_primes : 
  ∃ (p₁ p₂ p₃ p₄ p₅ : ℕ), 
    Prime p₁ ∧ Prime p₂ ∧ Prime p₃ ∧ Prime p₄ ∧ Prime p₅ ∧ 
    p₁ ≠ p₂ ∧ p₁ ≠ p₃ ∧ p₁ ≠ p₄ ∧ p₁ ≠ p₅ ∧ 
    p₂ ≠ p₃ ∧ p₂ ≠ p₄ ∧ p₂ ≠ p₅ ∧ 
    p₃ ≠ p₄ ∧ p₃ ≠ p₅ ∧ 
    p₄ ≠ p₅ ∧
    2310 = p₁ * p₂ * p₃ * p₄ * p₅ ∧
    ∀ (n : ℕ), n > 0 → (∃ (q₁ q₂ q₃ q₄ q₅ : ℕ), 
      Prime q₁ ∧ Prime q₂ ∧ Prime q₃ ∧ Prime q₄ ∧ Prime q₅ ∧ 
      q₁ ≠ q₂ ∧ q₁ ≠ q₃ ∧ q₁ ≠ q₄ ∧ q₁ ≠ q₅ ∧ 
      q₂ ≠ q₃ ∧ q₂ ≠ q₄ ∧ q₂ ≠ q₅ ∧ 
      q₃ ≠ q₄ ∧ q₃ ≠ q₅ ∧ 
      q₄ ≠ q₅ ∧
      n % q₁ = 0 ∧ n % q₂ = 0 ∧ n % q₃ = 0 ∧ n % q₄ = 0 ∧ n % q₅ = 0) → 
    n ≥ 2310 := by
  sorry

end NUMINAMATH_CALUDE_least_number_divisible_by_five_primes_l2466_246674


namespace NUMINAMATH_CALUDE_smallest_prime_perimeter_isosceles_triangle_l2466_246609

-- Define what it means for a number to be prime
def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

-- Define an isosceles triangle with prime side lengths
def isoscelesTrianglePrime (a b : ℕ) : Prop :=
  isPrime a ∧ isPrime b ∧ (a + a + b > a) ∧ (a + b > a)

-- Define the perimeter of the triangle
def perimeter (a b : ℕ) : ℕ := a + a + b

-- State the theorem
theorem smallest_prime_perimeter_isosceles_triangle :
  ∀ a b : ℕ, isoscelesTrianglePrime a b → isPrime (perimeter a b) →
  perimeter a b ≥ 11 :=
sorry

end NUMINAMATH_CALUDE_smallest_prime_perimeter_isosceles_triangle_l2466_246609


namespace NUMINAMATH_CALUDE_first_digit_after_500_erasure_l2466_246616

/-- Calculates the total number of digits when writing numbers from 1 to n in sequence -/
def totalDigits (n : ℕ) : ℕ := sorry

/-- Finds the first digit after erasing a certain number of digits from the sequence -/
def firstDigitAfterErasure (totalNumbers : ℕ) (erasedDigits : ℕ) : ℕ := sorry

theorem first_digit_after_500_erasure :
  firstDigitAfterErasure 500 500 = 3 := by sorry

end NUMINAMATH_CALUDE_first_digit_after_500_erasure_l2466_246616


namespace NUMINAMATH_CALUDE_q_sufficient_not_necessary_for_p_l2466_246686

-- Define the propositions p and q
def p (x : ℝ) : Prop := |x| < 2
def q (x : ℝ) : Prop := x^2 - x - 2 < 0

-- Theorem stating that q is sufficient but not necessary for p
theorem q_sufficient_not_necessary_for_p :
  (∀ x, q x → p x) ∧ ¬(∀ x, p x → q x) := by
  sorry

end NUMINAMATH_CALUDE_q_sufficient_not_necessary_for_p_l2466_246686


namespace NUMINAMATH_CALUDE_equilateral_triangle_not_centrally_symmetric_l2466_246698

-- Define the shape type
inductive Shape
  | Parallelogram
  | LineSegment
  | EquilateralTriangle
  | Rhombus

-- Define the property of being centrally symmetric
def is_centrally_symmetric (s : Shape) : Prop :=
  match s with
  | Shape.Parallelogram => True
  | Shape.LineSegment => True
  | Shape.EquilateralTriangle => False
  | Shape.Rhombus => True

-- Theorem statement
theorem equilateral_triangle_not_centrally_symmetric :
  ∀ s : Shape, ¬(is_centrally_symmetric s) ↔ s = Shape.EquilateralTriangle :=
by sorry

end NUMINAMATH_CALUDE_equilateral_triangle_not_centrally_symmetric_l2466_246698


namespace NUMINAMATH_CALUDE_constant_sum_l2466_246607

theorem constant_sum (a b : ℝ) : 
  (∀ x : ℝ, x ≠ 0 → (a + b / x = 1 ↔ x = -1) ∧ (a + b / x = 5 ↔ x = -5)) →
  a + b = 11 := by
sorry

end NUMINAMATH_CALUDE_constant_sum_l2466_246607


namespace NUMINAMATH_CALUDE_prime_factors_of_factorial_30_l2466_246635

theorem prime_factors_of_factorial_30 : 
  (Finset.filter Nat.Prime (Finset.range 31)).card = 
  (Nat.factors 30).toFinset.card := by sorry

end NUMINAMATH_CALUDE_prime_factors_of_factorial_30_l2466_246635


namespace NUMINAMATH_CALUDE_no_integer_solution_l2466_246673

theorem no_integer_solution : ¬ ∃ (n : ℤ), (n + 15 > 20) ∧ (-3*n > -9) := by
  sorry

end NUMINAMATH_CALUDE_no_integer_solution_l2466_246673


namespace NUMINAMATH_CALUDE_regression_equation_change_l2466_246664

theorem regression_equation_change (x y : ℝ) :
  y = 3 - 5 * x →
  (3 - 5 * (x + 1)) = y - 5 := by
sorry

end NUMINAMATH_CALUDE_regression_equation_change_l2466_246664


namespace NUMINAMATH_CALUDE_find_number_l2466_246603

theorem find_number : ∃ x : ℝ, 0.5 * x = 0.2 * 650 + 190 := by
  sorry

end NUMINAMATH_CALUDE_find_number_l2466_246603


namespace NUMINAMATH_CALUDE_platform_length_l2466_246625

/-- Given a train and platform with specific properties, prove the length of the platform. -/
theorem platform_length 
  (train_length : ℝ) 
  (time_platform : ℝ) 
  (time_pole : ℝ) 
  (h1 : train_length = 300)
  (h2 : time_platform = 51)
  (h3 : time_pole = 18) :
  ∃ (platform_length : ℝ), platform_length = 550 := by
  sorry

end NUMINAMATH_CALUDE_platform_length_l2466_246625


namespace NUMINAMATH_CALUDE_seedlings_per_packet_l2466_246670

theorem seedlings_per_packet (total_seedlings : ℕ) (num_packets : ℕ) 
  (h1 : total_seedlings = 420) (h2 : num_packets = 60) :
  total_seedlings / num_packets = 7 := by
  sorry

end NUMINAMATH_CALUDE_seedlings_per_packet_l2466_246670


namespace NUMINAMATH_CALUDE_square_circle_perimeter_l2466_246684

/-- Given a square with perimeter 28 cm and a circle with radius equal to the side of the square,
    the perimeter of the circle is 14π cm. -/
theorem square_circle_perimeter : 
  ∀ (square_side circle_radius : ℝ),
    square_side * 4 = 28 →
    circle_radius = square_side →
    2 * Real.pi * circle_radius = 14 * Real.pi := by
  sorry

end NUMINAMATH_CALUDE_square_circle_perimeter_l2466_246684


namespace NUMINAMATH_CALUDE_congruence_problem_l2466_246679

theorem congruence_problem (n : ℤ) : 
  0 ≤ n ∧ n < 25 ∧ -150 ≡ n [ZMOD 25] → n = 0 := by
  sorry

end NUMINAMATH_CALUDE_congruence_problem_l2466_246679


namespace NUMINAMATH_CALUDE_same_start_end_words_count_l2466_246667

/-- The number of letters in the alphabet -/
def alphabet_size : ℕ := 26

/-- The length of the words we're considering -/
def word_length : ℕ := 5

/-- The number of freely chosen letters in each word -/
def free_choices : ℕ := word_length - 2

/-- The number of five-letter words that begin and end with the same letter -/
def same_start_end_words : ℕ := alphabet_size ^ free_choices

theorem same_start_end_words_count :
  same_start_end_words = 456976 :=
sorry

end NUMINAMATH_CALUDE_same_start_end_words_count_l2466_246667


namespace NUMINAMATH_CALUDE_solution_implies_a_value_l2466_246645

theorem solution_implies_a_value (a : ℝ) : 
  (2 * 1 + 3 * a = -1) → a = -1 := by
  sorry

end NUMINAMATH_CALUDE_solution_implies_a_value_l2466_246645


namespace NUMINAMATH_CALUDE_no_rain_probability_l2466_246619

theorem no_rain_probability (p : ℝ) (n : ℕ) (h1 : p = 2/3) (h2 : n = 5) :
  (1 - p)^n = 1/243 := by
  sorry

end NUMINAMATH_CALUDE_no_rain_probability_l2466_246619


namespace NUMINAMATH_CALUDE_recurrence_necessary_not_sufficient_l2466_246631

/-- A sequence of real numbers -/
def Sequence := ℕ → ℝ

/-- The property that a sequence satisfies a_n = 2a_{n-1} for n ≥ 2 -/
def SatisfiesRecurrence (a : Sequence) : Prop :=
  ∀ n : ℕ, n ≥ 2 → a n = 2 * a (n - 1)

/-- The property that a sequence is geometric with common ratio 2 -/
def IsGeometricSequenceWithRatio2 (a : Sequence) : Prop :=
  ∃ r : ℝ, r ≠ 0 ∧ ∀ n : ℕ, a n = r * (2 ^ n)

/-- The main theorem stating that SatisfiesRecurrence is necessary but not sufficient
    for IsGeometricSequenceWithRatio2 -/
theorem recurrence_necessary_not_sufficient :
  (∀ a : Sequence, IsGeometricSequenceWithRatio2 a → SatisfiesRecurrence a) ∧
  (∃ a : Sequence, SatisfiesRecurrence a ∧ ¬IsGeometricSequenceWithRatio2 a) :=
by sorry

end NUMINAMATH_CALUDE_recurrence_necessary_not_sufficient_l2466_246631


namespace NUMINAMATH_CALUDE_shelter_cats_l2466_246665

theorem shelter_cats (cats dogs : ℕ) : 
  (cats : ℚ) / dogs = 15 / 7 ∧ 
  cats / (dogs + 12) = 15 / 11 → 
  cats = 45 := by
sorry

end NUMINAMATH_CALUDE_shelter_cats_l2466_246665


namespace NUMINAMATH_CALUDE_triangle_perimeter_range_l2466_246687

/-- Given a triangle ABC with side AC of length 2 and satisfying the equation
    √3 tan A tan C = tan A + tan C + √3, its perimeter is in (4, 2 + 2√3) ∪ (2 + 2√3, 6] -/
theorem triangle_perimeter_range (A C : Real) (hAC : Real) :
  hAC = 2 →
  Real.sqrt 3 * Real.tan A * Real.tan C = Real.tan A + Real.tan C + Real.sqrt 3 →
  ∃ (p : Real), p ∈ Set.union (Set.Ioo 4 (2 + 2 * Real.sqrt 3)) (Set.Ioc (2 + 2 * Real.sqrt 3) 6) ∧
                p = hAC + 2 * Real.sin (A + π / 6) + 2 * Real.sin (C + π / 6) :=
by sorry

end NUMINAMATH_CALUDE_triangle_perimeter_range_l2466_246687


namespace NUMINAMATH_CALUDE_matrix_power_four_l2466_246643

def A : Matrix (Fin 2) (Fin 2) ℤ := !![2, -1; 1, 1]

theorem matrix_power_four :
  A^4 = !![0, -9; 9, -9] := by sorry

end NUMINAMATH_CALUDE_matrix_power_four_l2466_246643


namespace NUMINAMATH_CALUDE_section_B_students_l2466_246672

/-- The number of students in section A -/
def students_A : ℕ := 36

/-- The average weight of students in section A (in kg) -/
def avg_weight_A : ℚ := 40

/-- The average weight of students in section B (in kg) -/
def avg_weight_B : ℚ := 35

/-- The average weight of the whole class (in kg) -/
def avg_weight_total : ℚ := 37.25

/-- The number of students in section B -/
def students_B : ℕ := 44

theorem section_B_students : 
  (students_A : ℚ) * avg_weight_A + (students_B : ℚ) * avg_weight_B = 
  ((students_A : ℚ) + students_B) * avg_weight_total := by
  sorry

end NUMINAMATH_CALUDE_section_B_students_l2466_246672


namespace NUMINAMATH_CALUDE_fraction_of_loss_l2466_246642

/-- Given the selling price and cost price of an item, calculate the fraction of loss. -/
theorem fraction_of_loss (selling_price cost_price : ℚ) 
  (h1 : selling_price = 15)
  (h2 : cost_price = 16) :
  (cost_price - selling_price) / cost_price = 1 / 16 := by
  sorry

end NUMINAMATH_CALUDE_fraction_of_loss_l2466_246642


namespace NUMINAMATH_CALUDE_expression_factorization_l2466_246659

theorem expression_factorization (x : ℝ) : 
  (4 * x^3 + 75 * x^2 - 12) - (-5 * x^3 + 3 * x^2 - 12) = 9 * x^2 * (x + 8) := by
  sorry

end NUMINAMATH_CALUDE_expression_factorization_l2466_246659


namespace NUMINAMATH_CALUDE_boys_to_girls_ratio_l2466_246614

/-- Theorem: Ratio of boys to girls in a class --/
theorem boys_to_girls_ratio 
  (boys_avg : ℝ) 
  (girls_avg : ℝ) 
  (class_avg : ℝ) 
  (missing_scores : ℕ) 
  (missing_avg : ℝ) 
  (h1 : boys_avg = 90) 
  (h2 : girls_avg = 96) 
  (h3 : class_avg = 94) 
  (h4 : missing_scores = 3) 
  (h5 : missing_avg = 92) :
  ∃ (boys girls : ℕ), 
    boys > 0 ∧ 
    girls > 0 ∧ 
    (boys : ℝ) / girls = 1 / 5 ∧
    class_avg * (boys + girls + missing_scores : ℝ) = 
      boys_avg * boys + girls_avg * girls + missing_avg * missing_scores :=
by sorry

end NUMINAMATH_CALUDE_boys_to_girls_ratio_l2466_246614


namespace NUMINAMATH_CALUDE_dave_files_left_l2466_246617

/-- The number of files Dave has left on his phone -/
def files_left : ℕ := 24

/-- The number of apps Dave has left on his phone -/
def apps_left : ℕ := 2

/-- The difference between the number of files and apps left -/
def file_app_difference : ℕ := 22

theorem dave_files_left :
  files_left = apps_left + file_app_difference :=
by sorry

end NUMINAMATH_CALUDE_dave_files_left_l2466_246617


namespace NUMINAMATH_CALUDE_watermelon_sales_theorem_l2466_246657

/-- Calculates the total income from selling watermelons -/
def watermelon_income (weight : ℕ) (price_per_pound : ℕ) (num_watermelons : ℕ) : ℕ :=
  weight * price_per_pound * num_watermelons

/-- Proves that selling 18 watermelons of 23 pounds each at $2 per pound yields $828 -/
theorem watermelon_sales_theorem :
  watermelon_income 23 2 18 = 828 := by
  sorry

end NUMINAMATH_CALUDE_watermelon_sales_theorem_l2466_246657


namespace NUMINAMATH_CALUDE_smallest_base_perfect_square_l2466_246692

/-- 
Given a base b > 4, returns the value of 45 in base b expressed in decimal.
-/
def base_b_to_decimal (b : ℕ) : ℕ := 4 * b + 5

/-- 
Proposition: 5 is the smallest integer b > 4 for which 45_b is a perfect square.
-/
theorem smallest_base_perfect_square : 
  (∀ b : ℕ, b > 4 ∧ b < 5 → ¬ ∃ k : ℕ, base_b_to_decimal b = k ^ 2) ∧
  (∃ k : ℕ, base_b_to_decimal 5 = k ^ 2) := by
  sorry

end NUMINAMATH_CALUDE_smallest_base_perfect_square_l2466_246692


namespace NUMINAMATH_CALUDE_otimes_self_otimes_self_l2466_246620

def otimes (x y : ℝ) : ℝ := x^2 - y^2

theorem otimes_self_otimes_self (h : ℝ) : otimes h (otimes h h) = h^2 := by
  sorry

end NUMINAMATH_CALUDE_otimes_self_otimes_self_l2466_246620


namespace NUMINAMATH_CALUDE_sum_inequality_l2466_246637

theorem sum_inequality (m n : ℕ) (hm : m > 0) (hn : n > 0) : 
  n * (n + 1) / 2 ≠ m * (m + 1) := by
  sorry

end NUMINAMATH_CALUDE_sum_inequality_l2466_246637


namespace NUMINAMATH_CALUDE_inequality_proof_l2466_246668

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a / (b + c) + b / (c + a) + c / (a + b) ≥ 3 / 2) ∧
  ((a * b * c = 1) → (a^2 / (b + c) + b^2 / (c + a) + c^2 / (a + b) ≥ 3 / 2)) ∧
  ((a * b * c = 1) → (1 / (a^3 * (b + c)) + 1 / (b^3 * (c + a)) + 1 / (c^3 * (a + b)) ≥ 3 / 2)) :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l2466_246668


namespace NUMINAMATH_CALUDE_animal_shelter_count_l2466_246648

theorem animal_shelter_count : 645 + 567 + 316 + 120 = 1648 := by
  sorry

end NUMINAMATH_CALUDE_animal_shelter_count_l2466_246648


namespace NUMINAMATH_CALUDE_number_problem_l2466_246613

theorem number_problem (x : ℚ) : x - (3/5) * x = 62 ↔ x = 155 := by
  sorry

end NUMINAMATH_CALUDE_number_problem_l2466_246613


namespace NUMINAMATH_CALUDE_right_triangle_sin_c_l2466_246678

theorem right_triangle_sin_c (A B C : ℝ) (h_right_angle : A + B + C = π) 
  (h_B_90 : B = π / 2) (h_cos_A : Real.cos A = 3 / 5) : Real.sin C = 4 / 5 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_sin_c_l2466_246678


namespace NUMINAMATH_CALUDE_parallel_lines_circle_distance_l2466_246605

theorem parallel_lines_circle_distance (r : ℝ) (d : ℝ) : 
  (∃ (chord1 chord2 chord3 : ℝ),
    chord1 = 38 ∧ 
    chord2 = 38 ∧ 
    chord3 = 34 ∧
    chord1 * 38 * chord1 / 4 + (d / 2) * 38 * (d / 2) = chord1 * r^2 ∧
    chord3 * 34 * chord3 / 4 + (3 * d / 2) * 34 * (3 * d / 2) = chord3 * r^2) →
  d = 6 := by
sorry

end NUMINAMATH_CALUDE_parallel_lines_circle_distance_l2466_246605


namespace NUMINAMATH_CALUDE_greatest_three_digit_multiple_of_17_l2466_246658

theorem greatest_three_digit_multiple_of_17 : ∃ n : ℕ, 
  n ≤ 999 ∧ 
  100 ≤ n ∧ 
  n % 17 = 0 ∧ 
  ∀ m : ℕ, m ≤ 999 ∧ 100 ≤ m ∧ m % 17 = 0 → m ≤ n :=
by
  use 986
  sorry

end NUMINAMATH_CALUDE_greatest_three_digit_multiple_of_17_l2466_246658


namespace NUMINAMATH_CALUDE_john_school_year_hours_l2466_246655

/-- Calculates the number of hours John needs to work per week during the school year -/
def school_year_hours (summer_hours : ℕ) (summer_weeks : ℕ) (summer_earnings : ℕ) 
  (school_year_weeks : ℕ) (school_year_earnings : ℕ) : ℕ :=
  let summer_hourly_rate := summer_earnings / (summer_hours * summer_weeks)
  let school_year_weekly_earnings := school_year_earnings / school_year_weeks
  school_year_weekly_earnings / summer_hourly_rate

/-- Theorem stating that John needs to work 8 hours per week during the school year -/
theorem john_school_year_hours : 
  school_year_hours 40 10 4000 50 4000 = 8 := by
  sorry

end NUMINAMATH_CALUDE_john_school_year_hours_l2466_246655


namespace NUMINAMATH_CALUDE_cube_of_negative_two_x_l2466_246652

theorem cube_of_negative_two_x (x : ℝ) : (-2 * x)^3 = -8 * x^3 := by
  sorry

end NUMINAMATH_CALUDE_cube_of_negative_two_x_l2466_246652


namespace NUMINAMATH_CALUDE_prime_iff_no_equal_products_l2466_246634

theorem prime_iff_no_equal_products (p : ℕ) (h : p > 1) :
  Nat.Prime p ↔ 
  ∀ (a b c d : ℕ), 
    a > 0 → b > 0 → c > 0 → d > 0 → 
    a + b + c + d = p → 
    (a * b ≠ c * d ∧ a * c ≠ b * d ∧ a * d ≠ b * c) :=
by sorry

end NUMINAMATH_CALUDE_prime_iff_no_equal_products_l2466_246634


namespace NUMINAMATH_CALUDE_factorization_difference_l2466_246685

theorem factorization_difference (y : ℝ) (a b : ℤ) : 
  (5 * y^2 + 17 * y + 6 = (5 * y + a) * (y + b)) → (a - b = -1) := by
  sorry

end NUMINAMATH_CALUDE_factorization_difference_l2466_246685


namespace NUMINAMATH_CALUDE_debt_average_payment_l2466_246627

theorem debt_average_payment 
  (total_installments : ℕ) 
  (first_payment_count : ℕ) 
  (first_payment_amount : ℚ) 
  (payment_increase : ℚ) :
  total_installments = 65 →
  first_payment_count = 20 →
  first_payment_amount = 410 →
  payment_increase = 65 →
  let remaining_payment_count := total_installments - first_payment_count
  let remaining_payment_amount := first_payment_amount + payment_increase
  let total_amount := first_payment_count * first_payment_amount + 
                      remaining_payment_count * remaining_payment_amount
  total_amount / total_installments = 455 := by
sorry

end NUMINAMATH_CALUDE_debt_average_payment_l2466_246627


namespace NUMINAMATH_CALUDE_inequality_proof_l2466_246662

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) (h_prod : x * y * z = 1) :
  x^2 + y^2 + z^2 + x*y + y*z + z*x ≥ 2 * (Real.sqrt x + Real.sqrt y + Real.sqrt z) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2466_246662


namespace NUMINAMATH_CALUDE_problem_solution_l2466_246640

-- Define proposition p
def p : Prop := ∀ a b : ℝ, (a > b ∧ b > 0) → (1/a < 1/b)

-- Define proposition q
def q : Prop := ∀ f : ℝ → ℝ, (∀ x : ℝ, f (x - 1) = f (-(x - 1))) → 
  (∀ x : ℝ, f x = f (2 - x))

-- Theorem to prove
theorem problem_solution : p ∨ ¬q := by sorry

end NUMINAMATH_CALUDE_problem_solution_l2466_246640


namespace NUMINAMATH_CALUDE_november_rainfall_l2466_246632

/-- The total rainfall in November for a northwestern town -/
def total_rainfall (first_half_daily_rainfall : ℝ) (days_in_november : ℕ) : ℝ :=
  let first_half := 15
  let second_half := days_in_november - first_half
  let first_half_total := first_half * first_half_daily_rainfall
  let second_half_total := second_half * (2 * first_half_daily_rainfall)
  first_half_total + second_half_total

/-- Theorem stating the total rainfall in November is 180 inches -/
theorem november_rainfall : total_rainfall 4 30 = 180 := by
  sorry

end NUMINAMATH_CALUDE_november_rainfall_l2466_246632


namespace NUMINAMATH_CALUDE_incorrect_height_proof_l2466_246600

/-- Given a class of boys with an incorrect average height and one boy's height
    recorded incorrectly, prove the value of the incorrectly recorded height. -/
theorem incorrect_height_proof (n : ℕ) (incorrect_avg real_avg actual_height : ℝ) 
    (hn : n = 35)
    (hi : incorrect_avg = 182)
    (hr : real_avg = 180)
    (ha : actual_height = 106) :
  ∃ (incorrect_height : ℝ),
    incorrect_height = 176 ∧
    n * real_avg = (n - 1) * incorrect_avg + actual_height - incorrect_height :=
by
  sorry

end NUMINAMATH_CALUDE_incorrect_height_proof_l2466_246600


namespace NUMINAMATH_CALUDE_magnified_tissue_diameter_l2466_246601

/-- Given a circular piece of tissue with an actual diameter and a magnification factor,
    calculate the diameter of the magnified image. -/
def magnified_diameter (actual_diameter : ℝ) (magnification_factor : ℝ) : ℝ :=
  actual_diameter * magnification_factor

/-- Theorem: The diameter of a circular piece of tissue with an actual diameter of 0.001 cm,
    when magnified 1,000 times, is 1 cm. -/
theorem magnified_tissue_diameter :
  magnified_diameter 0.001 1000 = 1 := by
  sorry

end NUMINAMATH_CALUDE_magnified_tissue_diameter_l2466_246601


namespace NUMINAMATH_CALUDE_angle_symmetry_l2466_246660

/-- Given that the terminal side of angle α is symmetric to the terminal side of angle -690° about the y-axis, prove that α = k * 360° + 150° for some integer k. -/
theorem angle_symmetry (α : Real) : 
  (∃ k : ℤ, α = k * 360 + 150) ↔ 
  (∃ n : ℤ, α + (-690) = n * 360 + 180) :=
by sorry

end NUMINAMATH_CALUDE_angle_symmetry_l2466_246660


namespace NUMINAMATH_CALUDE_largest_coefficient_in_expansion_l2466_246638

theorem largest_coefficient_in_expansion (a : ℝ) : 
  (a - 1)^5 = 32 → 
  ∃ (r : ℕ), r ≤ 5 ∧ 
    ∀ (s : ℕ), s ≤ 5 → 
      |(-1)^r * a^(5-r) * (Nat.choose 5 r)| ≥ |(-1)^s * a^(5-s) * (Nat.choose 5 s)| ∧
      (-1)^r * a^(5-r) * (Nat.choose 5 r) = 270 :=
by sorry

end NUMINAMATH_CALUDE_largest_coefficient_in_expansion_l2466_246638


namespace NUMINAMATH_CALUDE_circle_symmetric_about_center_circle_symmetric_about_diameter_circle_is_symmetrical_l2466_246623

/-- Definition of a circle in a plane -/
def Circle (center : ℝ × ℝ) (radius : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | (p.1 - center.1)^2 + (p.2 - center.2)^2 = radius^2}

/-- Definition of symmetry for a set about a point -/
def IsSymmetricAbout (S : Set (ℝ × ℝ)) (p : ℝ × ℝ) : Prop :=
  ∀ x ∈ S, ∃ y ∈ S, p.1 = (x.1 + y.1) / 2 ∧ p.2 = (x.2 + y.2) / 2

/-- Theorem: Any circle is symmetric about its center -/
theorem circle_symmetric_about_center (center : ℝ × ℝ) (radius : ℝ) :
  IsSymmetricAbout (Circle center radius) center := by
  sorry

/-- Theorem: Any circle is symmetric about any of its diameters -/
theorem circle_symmetric_about_diameter (center : ℝ × ℝ) (radius : ℝ) (a b : ℝ × ℝ) 
  (ha : a ∈ Circle center radius) (hb : b ∈ Circle center radius)
  (hdiameter : (a.1 - b.1)^2 + (a.2 - b.2)^2 = 4 * radius^2) :
  IsSymmetricAbout (Circle center radius) ((a.1 + b.1) / 2, (a.2 + b.2) / 2) := by
  sorry

/-- Main theorem: Any circle is a symmetrical figure -/
theorem circle_is_symmetrical (center : ℝ × ℝ) (radius : ℝ) :
  ∃ p, IsSymmetricAbout (Circle center radius) p := by
  sorry

end NUMINAMATH_CALUDE_circle_symmetric_about_center_circle_symmetric_about_diameter_circle_is_symmetrical_l2466_246623


namespace NUMINAMATH_CALUDE_silver_cube_gold_coating_value_l2466_246676

/-- Calculate the combined value of a silver cube with gold coating and markup -/
theorem silver_cube_gold_coating_value
  (cube_side : ℝ)
  (silver_density : ℝ)
  (silver_price : ℝ)
  (gold_coating_coverage : ℝ)
  (gold_coating_weight : ℝ)
  (gold_price : ℝ)
  (markup : ℝ)
  (h_cube_side : cube_side = 3)
  (h_silver_density : silver_density = 6)
  (h_silver_price : silver_price = 25)
  (h_gold_coating_coverage : gold_coating_coverage = 1/2)
  (h_gold_coating_weight : gold_coating_weight = 0.1)
  (h_gold_price : gold_price = 1800)
  (h_markup : markup = 1.1)
  : ∃ (total_value : ℝ), total_value = 18711 :=
by
  sorry

end NUMINAMATH_CALUDE_silver_cube_gold_coating_value_l2466_246676


namespace NUMINAMATH_CALUDE_exists_valid_heptagon_arrangement_l2466_246689

/-- Represents a heptagon with numbers placed in its vertices -/
def Heptagon := Fin 7 → Nat

/-- Checks if a given heptagon arrangement satisfies the sum condition -/
def is_valid_arrangement (h : Heptagon) : Prop :=
  (∀ i : Fin 7, h i ∈ Finset.range 15 \ {0}) ∧
  (∀ i : Fin 7, h i + h ((i + 1) % 7) + h ((i + 2) % 7) = 19)

/-- Theorem stating the existence of a valid heptagon arrangement -/
theorem exists_valid_heptagon_arrangement : ∃ h : Heptagon, is_valid_arrangement h :=
sorry

end NUMINAMATH_CALUDE_exists_valid_heptagon_arrangement_l2466_246689


namespace NUMINAMATH_CALUDE_coefficient_of_x_squared_l2466_246682

/-- The coefficient of x^2 in the expansion of (1+x+x^2)^6 -/
def a₂ : ℕ := (6 * (6 + 1)) / 2

/-- The expansion of (1+x+x^2)^6 -/
def expansion (x : ℝ) : ℝ := (1 + x + x^2)^6

theorem coefficient_of_x_squared :
  ∃ (f : ℝ → ℝ) (g : ℝ → ℝ),
    expansion = λ x => a₂ * x^2 + f x * x^3 + g x := by
  sorry

end NUMINAMATH_CALUDE_coefficient_of_x_squared_l2466_246682
