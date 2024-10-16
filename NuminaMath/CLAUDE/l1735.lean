import Mathlib

namespace NUMINAMATH_CALUDE_subset_condition_disjoint_condition_l1735_173520

-- Define sets A and B
def A : Set ℝ := {x | 1 < x ∧ x < 2}
def B (a : ℝ) : Set ℝ := {x | 2*a - 1 < x ∧ x < 2*a + 1}

-- Theorem for part (I)
theorem subset_condition (a : ℝ) : A ⊆ B a ↔ 1/2 ≤ a ∧ a ≤ 1 := by sorry

-- Theorem for part (II)
theorem disjoint_condition (a : ℝ) : A ∩ B a = ∅ ↔ a ≥ 3/2 ∨ a ≤ 0 := by sorry

end NUMINAMATH_CALUDE_subset_condition_disjoint_condition_l1735_173520


namespace NUMINAMATH_CALUDE_parallel_segment_length_l1735_173539

/-- Represents a trapezoid with given base lengths -/
structure Trapezoid where
  shorter_base : ℝ
  longer_base : ℝ
  h : shorter_base > 0
  k : longer_base > shorter_base

/-- Represents a line segment parallel to the bases of a trapezoid -/
structure ParallelSegment (T : Trapezoid) where
  length : ℝ
  passes_through_diagonal_intersection : Bool

/-- The theorem statement -/
theorem parallel_segment_length 
  (T : Trapezoid) 
  (S : ParallelSegment T) 
  (h : T.shorter_base = 4) 
  (k : T.longer_base = 12) 
  (m : S.passes_through_diagonal_intersection = true) : 
  S.length = 6 := by
  sorry

end NUMINAMATH_CALUDE_parallel_segment_length_l1735_173539


namespace NUMINAMATH_CALUDE_hypotenuse_length_l1735_173546

-- Define a right triangle
structure RightTriangle where
  a : ℝ  -- First leg
  b : ℝ  -- Second leg
  c : ℝ  -- Hypotenuse
  right_angle : a^2 + b^2 = c^2  -- Pythagorean theorem

-- Theorem statement
theorem hypotenuse_length (t : RightTriangle) 
  (perimeter : t.a + t.b + t.c = 40)  -- Perimeter condition
  (area : (1/2) * t.a * t.b = 24)     -- Area condition
  : t.c = 18.8 := by
  sorry  -- Proof omitted

end NUMINAMATH_CALUDE_hypotenuse_length_l1735_173546


namespace NUMINAMATH_CALUDE_triangle_perimeter_l1735_173552

/-- Given a triangle ABC with side lengths a, b, c opposite to angles A, B, C respectively,
    prove that the perimeter is 6√2 under the given conditions. -/
theorem triangle_perimeter (a b c : ℝ) (A : ℝ) : 
  A = π / 3 →
  b + c = 2 * a →
  (1 / 2) * b * c * Real.sin A = 2 * Real.sqrt 3 →
  a + b + c = 6 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_perimeter_l1735_173552


namespace NUMINAMATH_CALUDE_perimeter_of_hundred_rectangles_l1735_173558

/-- The perimeter of a shape formed by arranging rectangles edge-to-edge -/
def perimeter_of_arranged_rectangles (n : ℕ) (length width : ℝ) : ℝ :=
  let single_rectangle_perimeter := 2 * (length + width)
  let total_perimeter_without_overlap := n * single_rectangle_perimeter
  let number_of_joins := n - 1
  let overlap_per_join := 2 * width
  total_perimeter_without_overlap - (number_of_joins * overlap_per_join)

/-- Theorem stating that the perimeter of a shape formed by 100 rectangles 
    (each 3 cm by 1 cm) arranged edge-to-edge is 602 cm -/
theorem perimeter_of_hundred_rectangles : 
  perimeter_of_arranged_rectangles 100 3 1 = 602 := by
  sorry

end NUMINAMATH_CALUDE_perimeter_of_hundred_rectangles_l1735_173558


namespace NUMINAMATH_CALUDE_inequality_proof_l1735_173534

theorem inequality_proof (a b c : ℝ) (h1 : a > b) (h2 : b > c) (h3 : a + b + c = 0) :
  Real.sqrt (b^2 - a*c) < Real.sqrt 3 * a := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l1735_173534


namespace NUMINAMATH_CALUDE_irrational_identification_l1735_173580

theorem irrational_identification :
  (¬ ∃ (a b : ℤ), b ≠ 0 ∧ (5 : ℚ)^(1/3) = a / b) ∧ 
  (∃ (a b : ℤ), b ≠ 0 ∧ (9 : ℚ)^(1/2) = a / b) ∧
  (∃ (a b : ℤ), b ≠ 0 ∧ (-8/3 : ℚ) = a / b) ∧
  (∃ (a b : ℤ), b ≠ 0 ∧ (60.25 : ℚ) = a / b) :=
by sorry

end NUMINAMATH_CALUDE_irrational_identification_l1735_173580


namespace NUMINAMATH_CALUDE_klinker_double_age_l1735_173585

/-- The age difference between Mr. Klinker and his daughter -/
def age_difference : ℕ := 35 - 10

/-- The current age of Mr. Klinker -/
def klinker_age : ℕ := 35

/-- The current age of Mr. Klinker's daughter -/
def daughter_age : ℕ := 10

/-- The number of years until Mr. Klinker is twice as old as his daughter -/
def years_until_double : ℕ := 15

theorem klinker_double_age :
  klinker_age + years_until_double = 2 * (daughter_age + years_until_double) :=
sorry

end NUMINAMATH_CALUDE_klinker_double_age_l1735_173585


namespace NUMINAMATH_CALUDE_soccer_goals_average_l1735_173521

theorem soccer_goals_average : 
  let players_with_3_goals : ℕ := 2
  let players_with_4_goals : ℕ := 3
  let players_with_5_goals : ℕ := 1
  let players_with_6_goals : ℕ := 1
  let total_goals : ℕ := 3 * players_with_3_goals + 4 * players_with_4_goals + 
                          5 * players_with_5_goals + 6 * players_with_6_goals
  let total_players : ℕ := players_with_3_goals + players_with_4_goals + 
                           players_with_5_goals + players_with_6_goals
  (total_goals : ℚ) / total_players = 29 / 7 := by
  sorry

end NUMINAMATH_CALUDE_soccer_goals_average_l1735_173521


namespace NUMINAMATH_CALUDE_sin_alpha_plus_beta_l1735_173509

theorem sin_alpha_plus_beta (α β t : ℝ) : 
  (Real.exp (α + π/6) - Real.exp (-α - π/6) + Real.cos (5*π/3 + α) = t) →
  (Real.exp (β - π/4) - Real.exp (π/4 - β) + Real.cos (5*π/4 + β) = -t) →
  Real.sin (α + β) = (Real.sqrt 6 - Real.sqrt 2) / 4 := by
sorry

end NUMINAMATH_CALUDE_sin_alpha_plus_beta_l1735_173509


namespace NUMINAMATH_CALUDE_collinearity_proof_collinear_vectors_l1735_173525

variable {V : Type*} [AddCommGroup V] [Module ℝ V]

noncomputable section

def are_collinear (a b c : V) : Prop := ∃ (t : ℝ), c - a = t • (b - a)

theorem collinearity_proof 
  (e₁ e₂ A B C D : V) 
  (h₁ : e₁ ≠ 0) 
  (h₂ : e₂ ≠ 0) 
  (h₃ : ¬ ∃ (t : ℝ), e₁ = t • e₂) 
  (h₄ : B - A = e₁ + e₂) 
  (h₅ : C - B = 2 • e₁ + 8 • e₂) 
  (h₆ : D - C = 3 • (e₁ - e₂)) : 
  are_collinear A B D :=
sorry

theorem collinear_vectors 
  (e₁ e₂ : V) 
  (h₁ : e₁ ≠ 0) 
  (h₂ : e₂ ≠ 0) 
  (h₃ : ¬ ∃ (t : ℝ), e₁ = t • e₂) :
  ∀ (k : ℝ), (∃ (t : ℝ), k • e₁ + e₂ = t • (e₁ + k • e₂)) ↔ (k = 1 ∨ k = -1) :=
sorry

end

end NUMINAMATH_CALUDE_collinearity_proof_collinear_vectors_l1735_173525


namespace NUMINAMATH_CALUDE_walking_distance_l1735_173573

theorem walking_distance (original_speed original_distance increased_speed additional_distance : ℝ) 
  (h1 : original_speed = 4)
  (h2 : increased_speed = 5)
  (h3 : additional_distance = 6)
  (h4 : original_distance / original_speed = (original_distance + additional_distance) / increased_speed) :
  original_distance = 24 := by
sorry

end NUMINAMATH_CALUDE_walking_distance_l1735_173573


namespace NUMINAMATH_CALUDE_chess_probability_l1735_173501

theorem chess_probability (prob_A_win prob_draw : ℝ) 
  (h1 : prob_A_win = 0.3)
  (h2 : prob_draw = 0.5) :
  prob_A_win + prob_draw = 0.8 := by
sorry

end NUMINAMATH_CALUDE_chess_probability_l1735_173501


namespace NUMINAMATH_CALUDE_smallest_w_l1735_173595

theorem smallest_w (x y w : ℕ+) (h1 : x + y ≤ 10) (h2 : x - y ≥ 2) 
  (h3 : (2^x.val : ℕ) ∣ (3125 * w.val))
  (h4 : (3^y.val : ℕ) ∣ (3125 * w.val))
  (h5 : (5^(x.val + y.val) : ℕ) ∣ (3125 * w.val))
  (h6 : (7^(x.val - y.val) : ℕ) ∣ (3125 * w.val))
  (h7 : (13^4 : ℕ) ∣ (3125 * w.val)) :
  w.val ≥ 33592336 ∧ ∃ (w' : ℕ+), w'.val = 33592336 ∧ 
    (2^x.val : ℕ) ∣ (3125 * w'.val) ∧
    (3^y.val : ℕ) ∣ (3125 * w'.val) ∧
    (5^(x.val + y.val) : ℕ) ∣ (3125 * w'.val) ∧
    (7^(x.val - y.val) : ℕ) ∣ (3125 * w'.val) ∧
    (13^4 : ℕ) ∣ (3125 * w'.val) :=
by sorry

end NUMINAMATH_CALUDE_smallest_w_l1735_173595


namespace NUMINAMATH_CALUDE_chess_tournament_players_l1735_173579

theorem chess_tournament_players (total_games : ℕ) (h : total_games = 306) :
  ∃ n : ℕ, n > 0 ∧ n * (n - 1) * 2 = total_games ∧ n = 18 := by
  sorry

end NUMINAMATH_CALUDE_chess_tournament_players_l1735_173579


namespace NUMINAMATH_CALUDE_choir_members_l1735_173559

theorem choir_members (n : ℕ) (h1 : 50 ≤ n) (h2 : n ≤ 200) 
  (h3 : n % 7 = 4) (h4 : n % 6 = 8) : 
  n = 60 ∨ n = 102 ∨ n = 144 ∨ n = 186 := by
sorry

end NUMINAMATH_CALUDE_choir_members_l1735_173559


namespace NUMINAMATH_CALUDE_parabola_line_intersection_l1735_173507

/-- Parabola C: y² = 4x -/
def parabola_C (x y : ℝ) : Prop := y^2 = 4*x

/-- Line l with slope k passing through (-1, 0) -/
def line_l (k x y : ℝ) : Prop := y = k*(x + 1)

/-- Point on parabola C -/
def on_parabola_C (x y : ℝ) : Prop := parabola_C x y

/-- Point on line l -/
def on_line_l (k x y : ℝ) : Prop := line_l k x y

/-- Intersection ratio condition -/
def intersection_ratio (y₁ y₂ : ℝ) : Prop := y₁/y₂ + y₂/y₁ = 18

theorem parabola_line_intersection (k : ℝ) 
  (hk : k > 0)
  (hA : ∃ x₁ y₁, on_parabola_C x₁ y₁ ∧ on_line_l k x₁ y₁)
  (hB : ∃ x₂ y₂, on_parabola_C x₂ y₂ ∧ on_line_l k x₂ y₂)
  (hM : ∃ xₘ yₘ, on_parabola_C xₘ yₘ)
  (hN : ∃ xₙ yₙ, on_parabola_C xₙ yₙ)
  (h_ratio : ∀ y₁ y₂, intersection_ratio y₁ y₂) :
  k = Real.sqrt 5 / 5 := by
  sorry

end NUMINAMATH_CALUDE_parabola_line_intersection_l1735_173507


namespace NUMINAMATH_CALUDE_special_sequence_2016th_term_l1735_173583

/-- A sequence with specific properties -/
def special_sequence (a : ℕ → ℝ) : Prop :=
  a 4 = 1 ∧ 
  a 11 = 9 ∧ 
  ∀ n : ℕ, a n + a (n + 1) + a (n + 2) = 15

/-- The 2016th term of the special sequence is 5 -/
theorem special_sequence_2016th_term (a : ℕ → ℝ) 
  (h : special_sequence a) : a 2016 = 5 := by
  sorry

end NUMINAMATH_CALUDE_special_sequence_2016th_term_l1735_173583


namespace NUMINAMATH_CALUDE_dice_sum_impossibility_l1735_173542

theorem dice_sum_impossibility (a b c d : ℕ) : 
  1 ≤ a ∧ a ≤ 6 →
  1 ≤ b ∧ b ≤ 6 →
  1 ≤ c ∧ c ≤ 6 →
  1 ≤ d ∧ d ≤ 6 →
  a * b * c * d = 216 →
  a + b + c + d ≠ 20 := by
sorry

end NUMINAMATH_CALUDE_dice_sum_impossibility_l1735_173542


namespace NUMINAMATH_CALUDE_negation_of_all_squares_positive_l1735_173570

theorem negation_of_all_squares_positive :
  (¬ ∀ x : ℝ, x^2 > 0) ↔ (∃ x : ℝ, ¬(x^2 > 0)) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_all_squares_positive_l1735_173570


namespace NUMINAMATH_CALUDE_base7_difference_to_decimal_l1735_173524

/-- Converts a base 7 number to decimal --/
def base7ToDecimal (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (7 ^ i)) 0

/-- The difference between two base 7 numbers --/
def base7Difference (a b : List Nat) : List Nat :=
  sorry -- Implementation of base 7 subtraction

theorem base7_difference_to_decimal : 
  let a := [4, 1, 2, 3] -- 3214 in base 7 (least significant digit first)
  let b := [4, 3, 2, 1] -- 1234 in base 7 (least significant digit first)
  base7ToDecimal (base7Difference a b) = 721 := by
  sorry

end NUMINAMATH_CALUDE_base7_difference_to_decimal_l1735_173524


namespace NUMINAMATH_CALUDE_unique_solution_for_pure_imaginary_l1735_173577

/-- A complex number z is pure imaginary if its real part is zero and its imaginary part is non-zero -/
def is_pure_imaginary (z : ℂ) : Prop :=
  z.re = 0 ∧ z.im ≠ 0

/-- The complex number constructed from m -/
def complex_number (m : ℝ) : ℂ :=
  ⟨m^2 - 5*m + 6, m^2 - 3*m⟩

theorem unique_solution_for_pure_imaginary :
  ∃! m : ℝ, is_pure_imaginary (complex_number m) ∧ m = 2 :=
sorry

end NUMINAMATH_CALUDE_unique_solution_for_pure_imaginary_l1735_173577


namespace NUMINAMATH_CALUDE_quadratic_inequalities_l1735_173516

theorem quadratic_inequalities (x : ℝ) :
  (((1/2 : ℝ) * x^2 - 4*x + 6 < 0) ↔ (2 < x ∧ x < 6)) ∧
  ((4*x^2 - 4*x + 1 ≥ 0) ↔ True) ∧
  ((2*x^2 - x - 1 ≤ 0) ↔ (-1/2 ≤ x ∧ x ≤ 1)) ∧
  ((3*(x-2)*(x+2) - 4*(x+1)^2 + 1 < 0) ↔ (x < -5 ∨ x > -3)) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequalities_l1735_173516


namespace NUMINAMATH_CALUDE_soccer_league_games_l1735_173588

theorem soccer_league_games (n : ℕ) (h : n = 25) : n * (n - 1) / 2 = 300 := by
  sorry

end NUMINAMATH_CALUDE_soccer_league_games_l1735_173588


namespace NUMINAMATH_CALUDE_prob_through_C_eq_25_63_l1735_173586

/-- Represents a point in the city grid -/
structure Point where
  x : ℕ
  y : ℕ

/-- The probability of choosing either direction at an intersection -/
def choice_prob : ℚ := 1/2

/-- The starting point A -/
def A : Point := ⟨0, 0⟩

/-- The intermediate point C -/
def C : Point := ⟨3, 2⟩

/-- The ending point B -/
def B : Point := ⟨5, 5⟩

/-- Calculates the number of paths between two points -/
def num_paths (start finish : Point) : ℕ :=
  Nat.choose (finish.x - start.x + finish.y - start.y) (finish.x - start.x)

/-- The probability of walking from A to B through C -/
def prob_through_C : ℚ :=
  (num_paths A C * num_paths C B : ℚ) / num_paths A B

theorem prob_through_C_eq_25_63 : prob_through_C = 25/63 := by
  sorry

end NUMINAMATH_CALUDE_prob_through_C_eq_25_63_l1735_173586


namespace NUMINAMATH_CALUDE_amusement_park_optimization_l1735_173540

/-- Represents the ticket cost and ride time for an attraction -/
structure Attraction where
  ticketCost : Nat
  rideTime : Float

/-- Represents a ticket purchase option -/
structure TicketOption where
  quantity : Nat
  price : Float

theorem amusement_park_optimization (budget : Float) 
  (ferrisWheel rollerCoaster bumperCars carousel hauntedHouse : Attraction)
  (entranceFee : Float) (initialTickets : Nat)
  (individualTicketPrice : Float) (tenTicketBundle twentyTicketBundle : TicketOption)
  (lunchMinCost lunchMaxCost : Float) (souvenirMinCost souvenirMaxCost : Float)
  (timeBeforeActivity activityDuration : Float) :
  budget = 50 ∧ 
  entranceFee = 10 ∧ initialTickets = 5 ∧
  ferrisWheel = { ticketCost := 5, rideTime := 0.3 } ∧
  rollerCoaster = { ticketCost := 4, rideTime := 0.3 } ∧
  bumperCars = { ticketCost := 4, rideTime := 0.3 } ∧
  carousel = { ticketCost := 3, rideTime := 0.3 } ∧
  hauntedHouse = { ticketCost := 6, rideTime := 0.3 } ∧
  individualTicketPrice = 1.5 ∧
  tenTicketBundle = { quantity := 10, price := 12 } ∧
  twentyTicketBundle = { quantity := 20, price := 22 } ∧
  lunchMinCost = 8 ∧ lunchMaxCost = 15 ∧
  souvenirMinCost = 5 ∧ souvenirMaxCost = 12 ∧
  timeBeforeActivity = 3 ∧ activityDuration = 1 →
  (∃ (optimalPurchase : TicketOption),
    optimalPurchase = twentyTicketBundle ∧
    ferrisWheel.rideTime + rollerCoaster.rideTime + bumperCars.rideTime + 
    carousel.rideTime + hauntedHouse.rideTime = 1.5) := by
  sorry

end NUMINAMATH_CALUDE_amusement_park_optimization_l1735_173540


namespace NUMINAMATH_CALUDE_chocolate_gain_percent_l1735_173569

theorem chocolate_gain_percent (C S : ℝ) (h : 165 * C = 150 * S) : 
  (S - C) / C * 100 = 10 := by sorry

end NUMINAMATH_CALUDE_chocolate_gain_percent_l1735_173569


namespace NUMINAMATH_CALUDE_red_balls_count_l1735_173593

theorem red_balls_count (black_balls white_balls : ℕ) (red_prob : ℝ) : 
  black_balls = 8 → white_balls = 4 → red_prob = 0.4 → 
  (black_balls + white_balls : ℝ) / (1 - red_prob) = black_balls + white_balls + 8 := by
  sorry

#check red_balls_count

end NUMINAMATH_CALUDE_red_balls_count_l1735_173593


namespace NUMINAMATH_CALUDE_bisected_line_segment_l1735_173523

/-- Given a line segment with endpoints (5,1) and (m,1) bisected by x-2y=0, m = -1 -/
theorem bisected_line_segment (m : ℝ) : 
  let endpoint1 : ℝ × ℝ := (5, 1)
  let endpoint2 : ℝ × ℝ := (m, 1)
  let bisector : ℝ → ℝ := fun x => x / 2
  (bisector (endpoint1.1 + endpoint2.1) - 2 * 1 = 0) → m = -1 := by
sorry

end NUMINAMATH_CALUDE_bisected_line_segment_l1735_173523


namespace NUMINAMATH_CALUDE_cookies_left_in_scenario_l1735_173537

/-- Represents the cookie-making scenario -/
structure CookieScenario where
  cookies_per_batch : ℕ
  flour_per_batch : ℕ
  flour_bags : ℕ
  flour_per_bag : ℕ
  cookies_eaten : ℕ

/-- Calculates the number of cookies left after baking and eating -/
def cookies_left (scenario : CookieScenario) : ℕ :=
  let total_flour := scenario.flour_bags * scenario.flour_per_bag
  let total_cookies := (total_flour / scenario.flour_per_batch) * scenario.cookies_per_batch
  total_cookies - scenario.cookies_eaten

/-- Theorem stating the number of cookies left in the given scenario -/
theorem cookies_left_in_scenario : 
  let scenario : CookieScenario := {
    cookies_per_batch := 12,
    flour_per_batch := 2,
    flour_bags := 4,
    flour_per_bag := 5,
    cookies_eaten := 15
  }
  cookies_left scenario = 105 := by
  sorry


end NUMINAMATH_CALUDE_cookies_left_in_scenario_l1735_173537


namespace NUMINAMATH_CALUDE_shenzhen_revenue_precision_l1735_173518

/-- Represents a large monetary amount in yuan -/
structure LargeAmount where
  value : ℝ
  unit : String

/-- Defines the precision of a number -/
inductive Precision
  | HundredBillion
  | TenBillion
  | Billion
  | HundredMillion
  | TenMillion
  | Million

/-- Returns the precision of a given LargeAmount -/
def getPrecision (amount : LargeAmount) : Precision :=
  sorry

theorem shenzhen_revenue_precision :
  let revenue : LargeAmount := { value := 21.658, unit := "billion yuan" }
  getPrecision revenue = Precision.HundredMillion := by sorry

end NUMINAMATH_CALUDE_shenzhen_revenue_precision_l1735_173518


namespace NUMINAMATH_CALUDE_car_rental_savings_l1735_173576

def trip_distance : ℝ := 150
def first_option_cost : ℝ := 50
def second_option_cost : ℝ := 90
def gasoline_efficiency : ℝ := 15
def gasoline_cost_per_liter : ℝ := 0.9

theorem car_rental_savings : 
  let total_distance := 2 * trip_distance
  let gasoline_needed := total_distance / gasoline_efficiency
  let gasoline_cost := gasoline_needed * gasoline_cost_per_liter
  let first_option_total := first_option_cost + gasoline_cost
  second_option_cost - first_option_total = 22 := by sorry

end NUMINAMATH_CALUDE_car_rental_savings_l1735_173576


namespace NUMINAMATH_CALUDE_equation_solution_l1735_173529

theorem equation_solution : 
  ∃ x : ℚ, (x + 4) / (x - 3) = (x - 2) / (x + 2) ↔ x = -2 / 11 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1735_173529


namespace NUMINAMATH_CALUDE_y1_less_than_y2_l1735_173553

/-- Given a linear function y = 3x - b and two points P₁(3, y₁) and P₂(4, y₂) on its graph,
    prove that y₁ < y₂. -/
theorem y1_less_than_y2 (b : ℝ) (y₁ y₂ : ℝ) 
    (h₁ : y₁ = 3 * 3 - b) 
    (h₂ : y₂ = 3 * 4 - b) : 
  y₁ < y₂ := by
  sorry

end NUMINAMATH_CALUDE_y1_less_than_y2_l1735_173553


namespace NUMINAMATH_CALUDE_sum_of_roots_quadratic_l1735_173596

theorem sum_of_roots_quadratic (x : ℝ) : (x + 3) * (x - 4) = 20 → ∃ x₁ x₂ : ℝ, x₁ + x₂ = 1 ∧ (x₁ + 3) * (x₁ - 4) = 20 ∧ (x₂ + 3) * (x₂ - 4) = 20 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_roots_quadratic_l1735_173596


namespace NUMINAMATH_CALUDE_lunch_to_reading_time_ratio_l1735_173565

theorem lunch_to_reading_time_ratio
  (book_pages : ℕ)
  (pages_per_hour : ℕ)
  (lunch_time : ℕ)
  (h1 : book_pages = 4000)
  (h2 : pages_per_hour = 250)
  (h3 : lunch_time = 4) :
  (lunch_time : ℚ) / ((book_pages : ℚ) / (pages_per_hour : ℚ)) = 1 / 4 := by
  sorry

end NUMINAMATH_CALUDE_lunch_to_reading_time_ratio_l1735_173565


namespace NUMINAMATH_CALUDE_meetings_percentage_theorem_l1735_173532

/-- Calculates the percentage of a workday spent in meetings -/
def percentage_in_meetings (workday_hours : ℕ) (first_meeting_minutes : ℕ) (second_meeting_multiplier : ℕ) : ℚ :=
  let workday_minutes : ℕ := workday_hours * 60
  let second_meeting_minutes : ℕ := first_meeting_minutes * second_meeting_multiplier
  let total_meeting_minutes : ℕ := first_meeting_minutes + second_meeting_minutes
  (total_meeting_minutes : ℚ) / (workday_minutes : ℚ) * 100

theorem meetings_percentage_theorem (workday_hours : ℕ) (first_meeting_minutes : ℕ) (second_meeting_multiplier : ℕ)
  (h1 : workday_hours = 10)
  (h2 : first_meeting_minutes = 60)
  (h3 : second_meeting_multiplier = 3) :
  percentage_in_meetings workday_hours first_meeting_minutes second_meeting_multiplier = 40 := by
  sorry

end NUMINAMATH_CALUDE_meetings_percentage_theorem_l1735_173532


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l1735_173549

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, Real.exp x > x^2) ↔ (∃ x : ℝ, Real.exp x ≤ x^2) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l1735_173549


namespace NUMINAMATH_CALUDE_rectangular_field_area_l1735_173531

/-- Represents a rectangular field with a specific ratio of width to length and a given perimeter. -/
structure RectangularField where
  width : ℝ
  length : ℝ
  width_length_ratio : width = length / 3
  perimeter : width * 2 + length * 2 = 72

/-- Calculates the area of a rectangular field. -/
def area (field : RectangularField) : ℝ :=
  field.width * field.length

/-- Theorem stating that a rectangular field with the given properties has an area of 243 square meters. -/
theorem rectangular_field_area (field : RectangularField) : area field = 243 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_field_area_l1735_173531


namespace NUMINAMATH_CALUDE_pushup_problem_l1735_173515

theorem pushup_problem (x : ℕ) (h : x = 51) : 
  let zachary := x
  let melanie := 2 * zachary - 7
  let david := zachary + 22
  let karen := (zachary + melanie + david) / 3 - 5
  let john := david - 4
  john + melanie + karen = 232 := by
sorry

end NUMINAMATH_CALUDE_pushup_problem_l1735_173515


namespace NUMINAMATH_CALUDE_mans_rowing_speed_l1735_173500

/-- The rowing speed of a man in still water, given his speeds with and against a stream -/
theorem mans_rowing_speed (speed_with_stream speed_against_stream : ℝ) 
  (h1 : speed_with_stream = 16)
  (h2 : speed_against_stream = 4) : 
  (speed_with_stream + speed_against_stream) / 2 = 10 := by
  sorry

#check mans_rowing_speed

end NUMINAMATH_CALUDE_mans_rowing_speed_l1735_173500


namespace NUMINAMATH_CALUDE_opposite_of_negative_one_l1735_173530

theorem opposite_of_negative_one :
  (∀ x : ℤ, x + (-x) = 0) →
  -(-1) = 1 := by sorry

end NUMINAMATH_CALUDE_opposite_of_negative_one_l1735_173530


namespace NUMINAMATH_CALUDE_total_books_read_l1735_173568

def summer_reading (june july august : ℕ) : Prop :=
  june = 8 ∧ july = 2 * june ∧ august = july - 3

theorem total_books_read (june july august : ℕ) 
  (h : summer_reading june july august) : june + july + august = 37 := by
  sorry

end NUMINAMATH_CALUDE_total_books_read_l1735_173568


namespace NUMINAMATH_CALUDE_coin_denominations_exist_l1735_173562

theorem coin_denominations_exist : ∃ (coins : Finset ℕ), 
  (Finset.card coins = 12) ∧ 
  (∀ n : ℕ, 1 ≤ n ∧ n ≤ 6543 → 
    ∃ (representation : Multiset ℕ), 
      (Multiset.toFinset representation ⊆ coins) ∧
      (Multiset.card representation ≤ 8) ∧
      (Multiset.sum representation = n)) :=
by sorry

end NUMINAMATH_CALUDE_coin_denominations_exist_l1735_173562


namespace NUMINAMATH_CALUDE_least_sum_of_equal_multiples_l1735_173555

theorem least_sum_of_equal_multiples (x y z : ℕ+) (h : (4 : ℕ) * x.val = (5 : ℕ) * y.val ∧ (5 : ℕ) * y.val = (6 : ℕ) * z.val) :
  ∃ (a b c : ℕ+), (4 : ℕ) * a.val = (5 : ℕ) * b.val ∧ (5 : ℕ) * b.val = (6 : ℕ) * c.val ∧
    a.val + b.val + c.val ≤ x.val + y.val + z.val ∧ a.val + b.val + c.val = 37 :=
by
  sorry

end NUMINAMATH_CALUDE_least_sum_of_equal_multiples_l1735_173555


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_difference_l1735_173513

theorem right_triangle_hypotenuse_difference (longer_side shorter_side hypotenuse : ℝ) : 
  hypotenuse = 17 →
  shorter_side = longer_side - 7 →
  longer_side^2 + shorter_side^2 = hypotenuse^2 →
  hypotenuse - longer_side = 2 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_difference_l1735_173513


namespace NUMINAMATH_CALUDE_set_equality_implies_m_value_l1735_173563

theorem set_equality_implies_m_value (m : ℝ) :
  let A : Set ℝ := {1, 3, m^2}
  let B : Set ℝ := {1, m}
  A ∪ B = A →
  m = 0 ∨ m = 3 := by
sorry

end NUMINAMATH_CALUDE_set_equality_implies_m_value_l1735_173563


namespace NUMINAMATH_CALUDE_initial_toothbrushes_l1735_173502

/-- The number of toothbrushes given away in January -/
def january : ℕ := 53

/-- The number of toothbrushes given away in February -/
def february : ℕ := 67

/-- The number of toothbrushes given away in March -/
def march : ℕ := 46

/-- The difference between the busiest and slowest month -/
def difference : ℕ := 36

/-- The number of toothbrushes given away in April (equal to May) -/
def april_may : ℕ := february - difference

/-- The total number of toothbrushes Dr. Banks had initially -/
def total_toothbrushes : ℕ := january + february + march + 2 * april_may

theorem initial_toothbrushes : total_toothbrushes = 228 := by
  sorry

end NUMINAMATH_CALUDE_initial_toothbrushes_l1735_173502


namespace NUMINAMATH_CALUDE_magic_square_y_value_l1735_173538

/-- Represents a 3x3 modified magic square -/
structure ModifiedMagicSquare where
  entries : Matrix (Fin 3) (Fin 3) ℕ
  is_magic : ∀ (i j : Fin 3), 
    (entries i 0 + entries i 1 + entries i 2 = 
     entries 0 j + entries 1 j + entries 2 j) ∧
    (entries 0 0 + entries 1 1 + entries 2 2 = 
     entries 0 2 + entries 1 1 + entries 2 0)

/-- The theorem stating that y must be 245 in the given modified magic square -/
theorem magic_square_y_value (square : ModifiedMagicSquare) 
  (h1 : square.entries 0 1 = 25)
  (h2 : square.entries 0 2 = 120)
  (h3 : square.entries 1 0 = 5) :
  square.entries 0 0 = 245 := by
  sorry

end NUMINAMATH_CALUDE_magic_square_y_value_l1735_173538


namespace NUMINAMATH_CALUDE_division_remainder_l1735_173575

theorem division_remainder : 
  let dividend : ℕ := 23
  let divisor : ℕ := 5
  let quotient : ℕ := 4
  dividend % divisor = 3 := by
  sorry

end NUMINAMATH_CALUDE_division_remainder_l1735_173575


namespace NUMINAMATH_CALUDE_fathers_sons_age_product_l1735_173541

theorem fathers_sons_age_product (father_age son_age : ℕ) : 
  father_age > 0 ∧ son_age > 0 ∧
  father_age = 7 * (son_age / 3) ∧
  (father_age + 6) = 2 * (son_age + 6) →
  father_age * son_age = 756 := by
sorry

end NUMINAMATH_CALUDE_fathers_sons_age_product_l1735_173541


namespace NUMINAMATH_CALUDE_principal_is_900_l1735_173526

/-- Proves that given the conditions of the problem, the principal must be $900 -/
theorem principal_is_900 (P R : ℝ) : 
  (P * (R + 3) * 3) / 100 = (P * R * 3) / 100 + 81 → P = 900 := by
  sorry

end NUMINAMATH_CALUDE_principal_is_900_l1735_173526


namespace NUMINAMATH_CALUDE_ordering_of_a_b_c_l1735_173561

theorem ordering_of_a_b_c :
  let a := Real.tan (1/2)
  let b := Real.tan (2/π)
  let c := Real.sqrt 3 / π
  a < c ∧ c < b := by sorry

end NUMINAMATH_CALUDE_ordering_of_a_b_c_l1735_173561


namespace NUMINAMATH_CALUDE_quadratic_root_implies_m_l1735_173574

theorem quadratic_root_implies_m (m : ℝ) : 
  (∀ x : ℝ, (m - 1) * x^2 + 3 * x - 5 * m + 4 = 0 → x = 2 ∨ x ≠ 2) →
  ((m - 1) * 2^2 + 3 * 2 - 5 * m + 4 = 0) →
  m = 6 := by
sorry

end NUMINAMATH_CALUDE_quadratic_root_implies_m_l1735_173574


namespace NUMINAMATH_CALUDE_martaFamily_childless_count_marta_childless_count_l1735_173551

/-- Represents a woman in Marta's family tree -/
structure Woman where
  daughters : Nat

/-- Marta's family tree -/
structure MartaFamily where
  marta : Woman
  daughters : Finset Woman

theorem martaFamily_childless_count (f : MartaFamily) : Nat :=
  let total_women := f.daughters.card + (f.daughters.sum fun d => d.daughters)
  let daughters_with_children := f.daughters.filter fun d => d.daughters > 0
  let childless_count := f.daughters.card + (f.daughters.sum fun d => d.daughters) - daughters_with_children.card
  childless_count

/-- The number of Marta's daughters and granddaughters without daughters is 37 -/
theorem marta_childless_count : ∃ (f : MartaFamily),
  f.marta.daughters = 0 ∧
  f.daughters.card = 7 ∧
  (f.daughters.card + (f.daughters.sum fun d => d.daughters) = 42) ∧
  (∀ d ∈ f.daughters, d.daughters = 0 ∨ d.daughters = 6) ∧
  (∀ d ∈ f.daughters, ∀ g ∈ f.daughters, g.daughters = 0) →
  martaFamily_childless_count f = 37 := by
  sorry

end NUMINAMATH_CALUDE_martaFamily_childless_count_marta_childless_count_l1735_173551


namespace NUMINAMATH_CALUDE_readers_of_both_genres_l1735_173517

theorem readers_of_both_genres (total : ℕ) (sci_fi : ℕ) (literary : ℕ) 
  (h_total : total = 150)
  (h_sci_fi : sci_fi = 120)
  (h_literary : literary = 90) :
  sci_fi + literary - total = 60 := by
  sorry

end NUMINAMATH_CALUDE_readers_of_both_genres_l1735_173517


namespace NUMINAMATH_CALUDE_arithmetic_sequence_properties_l1735_173556

/-- An arithmetic sequence with the given properties -/
structure ArithmeticSequence where
  a : ℕ+ → ℚ
  a_1_eq_1 : a 1 = 1
  is_arithmetic : ∃ d ≠ 0, ∀ n : ℕ+, a (n + 1) = a n + d
  is_geometric : (a 2)^2 = a 1 * a 5

/-- The b_n sequence derived from the arithmetic sequence -/
def b (seq : ArithmeticSequence) (n : ℕ+) : ℚ :=
  1 / (seq.a n * seq.a (n + 1))

/-- The sum of the first n terms of the b sequence -/
def T (seq : ArithmeticSequence) (n : ℕ+) : ℚ :=
  (Finset.range n).sum (λ i => b seq ⟨i + 1, Nat.succ_pos i⟩)

theorem arithmetic_sequence_properties (seq : ArithmeticSequence) :
  (∀ n : ℕ+, seq.a n = 2 * n - 1) ∧
  (∀ n : ℕ+, T seq n = n / (2 * n + 1)) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_properties_l1735_173556


namespace NUMINAMATH_CALUDE_complex_modulus_problem_l1735_173535

theorem complex_modulus_problem (z : ℂ) (h : z * (2 + Complex.I) = 5 * Complex.I) :
  Complex.abs (z - 1) = 2 := by sorry

end NUMINAMATH_CALUDE_complex_modulus_problem_l1735_173535


namespace NUMINAMATH_CALUDE_num_biology_books_is_15_l1735_173582

def num_chemistry_books : ℕ := 8
def total_ways_to_pick : ℕ := 2940

-- Function to calculate combinations
def choose (n k : ℕ) : ℕ := 
  if k > n then 0
  else Nat.factorial n / (Nat.factorial k * Nat.factorial (n - k))

-- Theorem to prove
theorem num_biology_books_is_15 : 
  ∃ (B : ℕ), choose B 2 * choose num_chemistry_books 2 = total_ways_to_pick ∧ B = 15 :=
by sorry

end NUMINAMATH_CALUDE_num_biology_books_is_15_l1735_173582


namespace NUMINAMATH_CALUDE_allan_initial_balloons_l1735_173564

def balloons_problem (initial_balloons : ℕ) : Prop :=
  let total_balloons := initial_balloons + 3
  6 = total_balloons + 1

theorem allan_initial_balloons : 
  ∃ (initial_balloons : ℕ), balloons_problem initial_balloons ∧ initial_balloons = 2 := by
  sorry

end NUMINAMATH_CALUDE_allan_initial_balloons_l1735_173564


namespace NUMINAMATH_CALUDE_remainder_11_power_603_mod_500_l1735_173548

theorem remainder_11_power_603_mod_500 : 11^603 % 500 = 331 := by
  sorry

end NUMINAMATH_CALUDE_remainder_11_power_603_mod_500_l1735_173548


namespace NUMINAMATH_CALUDE_al_original_amount_l1735_173578

/-- Represents the investment scenario with Al, Betty, and Clare --/
structure Investment where
  al : ℝ
  betty : ℝ
  clare : ℝ

/-- The conditions of the investment problem --/
def validInvestment (inv : Investment) : Prop :=
  inv.al + inv.betty + inv.clare = 1200 ∧
  (inv.al - 200) + (3 * inv.betty) + (4 * inv.clare) = 1800

/-- The theorem stating Al's original investment amount --/
theorem al_original_amount :
  ∀ inv : Investment, validInvestment inv → inv.al = 860 := by
  sorry

end NUMINAMATH_CALUDE_al_original_amount_l1735_173578


namespace NUMINAMATH_CALUDE_triangle_square_perimeter_l1735_173508

theorem triangle_square_perimeter (d : ℕ) : 
  let triangle_side := s + d
  let square_side := s
  (∃ s : ℚ, s > 0 ∧ 3 * triangle_side - 4 * square_side = 1989) →
  d > 663 :=
by
  sorry

end NUMINAMATH_CALUDE_triangle_square_perimeter_l1735_173508


namespace NUMINAMATH_CALUDE_grasshopper_jump_distance_l1735_173567

theorem grasshopper_jump_distance 
  (frog_distance : ℕ → ℕ → ℕ) 
  (mouse_distance : ℕ → ℕ → ℕ) 
  (grasshopper_distance : ℕ → ℕ) :
  (∀ g f, frog_distance g f = g + 32) →
  (∀ m f, mouse_distance m f = f - 26) →
  mouse_distance 31 (frog_distance (grasshopper_distance 31) 31) = 31 →
  grasshopper_distance 31 = 25 := by
sorry

end NUMINAMATH_CALUDE_grasshopper_jump_distance_l1735_173567


namespace NUMINAMATH_CALUDE_april_days_l1735_173566

/-- Proves the number of days in April based on Hannah's strawberry harvesting scenario -/
theorem april_days (daily_harvest : ℕ) (given_away : ℕ) (stolen : ℕ) (final_count : ℕ) :
  daily_harvest = 5 →
  given_away = 20 →
  stolen = 30 →
  final_count = 100 →
  (final_count + given_away + stolen) / daily_harvest = 30 := by
  sorry

#check april_days

end NUMINAMATH_CALUDE_april_days_l1735_173566


namespace NUMINAMATH_CALUDE_smallest_number_divisible_l1735_173599

theorem smallest_number_divisible (n : ℕ) : n = 32127 ↔ 
  (∀ m : ℕ, m < n → ¬(((m + 3) % 510 = 0) ∧ ((m + 3) % 4590 = 0) ∧ ((m + 3) % 105 = 0))) ∧
  ((n + 3) % 510 = 0) ∧ ((n + 3) % 4590 = 0) ∧ ((n + 3) % 105 = 0) :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_divisible_l1735_173599


namespace NUMINAMATH_CALUDE_smallest_sum_of_digits_of_sum_l1735_173503

/-- A function that returns the sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- A function that checks if all digits in a natural number are different -/
def all_digits_different (n : ℕ) : Prop := sorry

/-- A function that checks if two natural numbers have all different digits between them -/
def all_digits_different_between (a b : ℕ) : Prop := sorry

theorem smallest_sum_of_digits_of_sum :
  ∀ a b : ℕ,
    100 ≤ a ∧ a < 1000 ∧
    100 ≤ b ∧ b < 1000 ∧
    a ≠ b ∧
    all_digits_different a ∧
    all_digits_different b ∧
    all_digits_different_between a b ∧
    1000 ≤ a + b ∧ a + b < 10000 →
    ∃ (s : ℕ), s = a + b ∧ sum_of_digits s = 1 ∧
    ∀ (t : ℕ), t = a + b → sum_of_digits s ≤ sum_of_digits t :=
by sorry

end NUMINAMATH_CALUDE_smallest_sum_of_digits_of_sum_l1735_173503


namespace NUMINAMATH_CALUDE_largest_number_l1735_173512

theorem largest_number (a b c d : ℝ) (ha : a = -1.5) (hb : b = -3) (hc : c = -1) (hd : d = -5) :
  max a (max b (max c d)) = c := by
  sorry

end NUMINAMATH_CALUDE_largest_number_l1735_173512


namespace NUMINAMATH_CALUDE_ellipse_focal_length_l1735_173511

/-- The focal length of an ellipse with equation x²/4 + y² = 1 is 2√3 -/
theorem ellipse_focal_length : 
  ∃ (f : ℝ), f = 2 * Real.sqrt 3 ∧ 
  ∀ (x y : ℝ), x^2 / 4 + y^2 = 1 → 
  f = 2 * Real.sqrt ((2^2 - 1^2) : ℝ) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_focal_length_l1735_173511


namespace NUMINAMATH_CALUDE_boat_speed_ratio_l1735_173587

theorem boat_speed_ratio (boat_speed : ℝ) (current_speed : ℝ) (distance : ℝ) 
  (h1 : boat_speed = 20)
  (h2 : current_speed = 4)
  (h3 : distance = 2) : 
  (2 * distance) / ((distance / (boat_speed + current_speed)) + (distance / (boat_speed - current_speed))) / boat_speed = 24 / 25 := by
  sorry

end NUMINAMATH_CALUDE_boat_speed_ratio_l1735_173587


namespace NUMINAMATH_CALUDE_total_harvest_kg_l1735_173504

def apple_sections : ℕ := 8
def apple_yield_per_section : ℕ := 450

def orange_sections : ℕ := 10
def orange_crates_per_section : ℕ := 60
def orange_kg_per_crate : ℕ := 8

def peach_sections : ℕ := 3
def peach_sacks_per_section : ℕ := 55
def peach_kg_per_sack : ℕ := 12

def cherry_fields : ℕ := 5
def cherry_baskets_per_field : ℕ := 50
def cherry_kg_per_basket : ℚ := 3.5

theorem total_harvest_kg : 
  apple_sections * apple_yield_per_section + 
  orange_sections * orange_crates_per_section * orange_kg_per_crate + 
  peach_sections * peach_sacks_per_section * peach_kg_per_sack + 
  cherry_fields * cherry_baskets_per_field * cherry_kg_per_basket = 11255 := by
  sorry

end NUMINAMATH_CALUDE_total_harvest_kg_l1735_173504


namespace NUMINAMATH_CALUDE_unique_solution_set_l1735_173571

def A (m : ℝ) : Set ℝ := {x : ℝ | m * x^2 + 2 * x + 3 = 0}

def M : Set ℝ := {m : ℝ | ∃! x : ℝ, m * x^2 + 2 * x + 3 = 0}

theorem unique_solution_set : M = {0, 1/3} := by sorry

end NUMINAMATH_CALUDE_unique_solution_set_l1735_173571


namespace NUMINAMATH_CALUDE_representable_multiple_of_three_l1735_173522

/-- A number is representable if it can be written as x^2 + 2y^2 for some integers x and y -/
def Representable (n : ℤ) : Prop :=
  ∃ x y : ℤ, n = x^2 + 2*y^2

/-- If 3a is representable, then a is representable -/
theorem representable_multiple_of_three (a : ℤ) :
  Representable (3*a) → Representable a := by
  sorry

end NUMINAMATH_CALUDE_representable_multiple_of_three_l1735_173522


namespace NUMINAMATH_CALUDE_colombian_coffee_amount_l1735_173554

/-- Proves the amount of Colombian coffee in a specific coffee mix -/
theorem colombian_coffee_amount
  (total_mix : ℝ)
  (colombian_price : ℝ)
  (brazilian_price : ℝ)
  (mix_price : ℝ)
  (h1 : total_mix = 100)
  (h2 : colombian_price = 8.75)
  (h3 : brazilian_price = 3.75)
  (h4 : mix_price = 6.35) :
  ∃ (colombian_amount : ℝ),
    colombian_amount = 52 ∧
    colombian_amount ≥ 0 ∧
    colombian_amount ≤ total_mix ∧
    ∃ (brazilian_amount : ℝ),
      brazilian_amount = total_mix - colombian_amount ∧
      colombian_price * colombian_amount + brazilian_price * brazilian_amount = mix_price * total_mix :=
by sorry

end NUMINAMATH_CALUDE_colombian_coffee_amount_l1735_173554


namespace NUMINAMATH_CALUDE_mikes_ride_length_l1735_173598

/-- Proves that Mike's ride was 36 miles given the taxi fare conditions -/
theorem mikes_ride_length (mike_start : ℝ) (mike_per_mile : ℝ) (annie_start : ℝ) 
  (annie_toll : ℝ) (annie_per_mile : ℝ) (annie_miles : ℝ) :
  mike_start = 2.5 →
  mike_per_mile = 0.25 →
  annie_start = 2.5 →
  annie_toll = 5 →
  annie_per_mile = 0.25 →
  annie_miles = 26 →
  mike_start + mike_per_mile * 36 = annie_start + annie_toll + annie_per_mile * annie_miles :=
by sorry

end NUMINAMATH_CALUDE_mikes_ride_length_l1735_173598


namespace NUMINAMATH_CALUDE_vertically_opposite_angles_equal_l1735_173591

/-- Two angles are vertically opposite if they are formed by two intersecting lines
    and are not adjacent to each other. -/
def vertically_opposite (α β : Real) : Prop := sorry

theorem vertically_opposite_angles_equal {α β : Real} (h : vertically_opposite α β) : α = β := by
  sorry

end NUMINAMATH_CALUDE_vertically_opposite_angles_equal_l1735_173591


namespace NUMINAMATH_CALUDE_circle_condition_l1735_173528

theorem circle_condition (a : ℝ) : 
  (∃ (x y : ℝ), x^2 + y^2 + 2*y + 2*a - 1 = 0 ∧ ∀ (x' y' : ℝ), x'^2 + y'^2 + 2*y' + 2*a - 1 = 0 → (x - x')^2 + (y - y')^2 = (x' - x)^2 + (y' - y)^2) 
  → a < 1 := by
  sorry

end NUMINAMATH_CALUDE_circle_condition_l1735_173528


namespace NUMINAMATH_CALUDE_project_time_allocation_l1735_173572

theorem project_time_allocation (total_time research_time proposal_time : ℕ) 
  (h1 : total_time = 20)
  (h2 : research_time = 10)
  (h3 : proposal_time = 2) :
  total_time - (research_time + proposal_time) = 8 := by
  sorry

end NUMINAMATH_CALUDE_project_time_allocation_l1735_173572


namespace NUMINAMATH_CALUDE_alcohol_percentage_in_second_vessel_l1735_173536

theorem alcohol_percentage_in_second_vessel
  (vessel1_capacity : ℝ)
  (vessel1_alcohol_percentage : ℝ)
  (vessel2_capacity : ℝ)
  (total_liquid : ℝ)
  (final_vessel_capacity : ℝ)
  (final_mixture_percentage : ℝ)
  (h1 : vessel1_capacity = 3)
  (h2 : vessel1_alcohol_percentage = 25)
  (h3 : vessel2_capacity = 5)
  (h4 : total_liquid = 8)
  (h5 : final_vessel_capacity = 10)
  (h6 : final_mixture_percentage = 27.5)
  : ∃ (vessel2_alcohol_percentage : ℝ),
    vessel2_alcohol_percentage = 40 ∧
    vessel1_capacity * (vessel1_alcohol_percentage / 100) +
    vessel2_capacity * (vessel2_alcohol_percentage / 100) =
    final_vessel_capacity * (final_mixture_percentage / 100) :=
by sorry

end NUMINAMATH_CALUDE_alcohol_percentage_in_second_vessel_l1735_173536


namespace NUMINAMATH_CALUDE_problem_solution_l1735_173544

theorem problem_solution :
  (∀ x : ℝ, x + 1/x = 5 → x^2 + 1/x^2 = 23) ∧
  ((5/3)^2004 * (3/5)^2003 = 5/3) := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l1735_173544


namespace NUMINAMATH_CALUDE_stating_whack_a_mole_tickets_correct_l1735_173560

/-- Represents the number of tickets Tom won from 'whack a mole' -/
def whack_a_mole_tickets : ℕ := 32

/-- Represents the number of tickets Tom won from 'skee ball' -/
def skee_ball_tickets : ℕ := 25

/-- Represents the number of tickets Tom spent on a hat -/
def spent_tickets : ℕ := 7

/-- Represents the number of tickets Tom has left -/
def remaining_tickets : ℕ := 50

/-- 
Theorem stating that the number of tickets Tom won from 'whack a mole' 
is correct given the other known information
-/
theorem whack_a_mole_tickets_correct : 
  whack_a_mole_tickets + skee_ball_tickets = remaining_tickets + spent_tickets := by
  sorry

#check whack_a_mole_tickets_correct

end NUMINAMATH_CALUDE_stating_whack_a_mole_tickets_correct_l1735_173560


namespace NUMINAMATH_CALUDE_remainder_theorem_l1735_173545

theorem remainder_theorem (x y z : ℤ) 
  (hx : x % 102 = 56)
  (hy : y % 154 = 79)
  (hz : z % 297 = 183) :
  (x % 19 = 18) ∧ (y % 22 = 13) ∧ (z % 33 = 18) := by
  sorry

end NUMINAMATH_CALUDE_remainder_theorem_l1735_173545


namespace NUMINAMATH_CALUDE_first_nonzero_digit_of_one_over_157_l1735_173594

theorem first_nonzero_digit_of_one_over_157 : ∃ (n : ℕ), 
  (1000 : ℚ) / 157 > 6 ∧ (1000 : ℚ) / 157 < 7 ∧ 
  (1000 * (1 : ℚ) / 157 - 6) * 10 ≥ 3 ∧ (1000 * (1 : ℚ) / 157 - 6) * 10 < 4 := by
  sorry

end NUMINAMATH_CALUDE_first_nonzero_digit_of_one_over_157_l1735_173594


namespace NUMINAMATH_CALUDE_correct_operation_is_multiplication_by_three_l1735_173514

theorem correct_operation_is_multiplication_by_three (x : ℝ) : 
  (((3 * x - x / 5) / (3 * x)) * 100 = 93.33333333333333) → 
  (∃ (y : ℝ), y = 3 ∧ x * y = 3 * x) :=
by
  sorry

end NUMINAMATH_CALUDE_correct_operation_is_multiplication_by_three_l1735_173514


namespace NUMINAMATH_CALUDE_hyperbola_semi_focal_distance_l1735_173550

/-- Given a hyperbola with equation x²/20 - y²/5 = 1, its semi-focal distance is 5 -/
theorem hyperbola_semi_focal_distance :
  ∀ (x y : ℝ), x^2 / 20 - y^2 / 5 = 1 → ∃ (c : ℝ), c = 5 ∧ c^2 = 20 + 5 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_semi_focal_distance_l1735_173550


namespace NUMINAMATH_CALUDE_bicycle_sale_profit_l1735_173547

theorem bicycle_sale_profit (final_price : ℝ) (initial_cost : ℝ) (intermediate_profit_rate : ℝ) :
  final_price = 225 →
  initial_cost = 150 →
  intermediate_profit_rate = 0.25 →
  ((final_price / (1 + intermediate_profit_rate) - initial_cost) / initial_cost) * 100 = 20 := by
sorry

end NUMINAMATH_CALUDE_bicycle_sale_profit_l1735_173547


namespace NUMINAMATH_CALUDE_sufficient_but_not_necessary_l1735_173597

theorem sufficient_but_not_necessary (x : ℝ) :
  (∀ x, x > 2 → x > 0) ∧ (∃ x, x > 0 ∧ ¬(x > 2)) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_but_not_necessary_l1735_173597


namespace NUMINAMATH_CALUDE_number_of_red_balls_l1735_173510

/-- Given a bag with white and red balls, prove the number of red balls. -/
theorem number_of_red_balls
  (total_balls : ℕ) 
  (white_balls : ℕ) 
  (red_frequency : ℚ) 
  (h1 : white_balls = 60)
  (h2 : red_frequency = 1/4)
  (h3 : total_balls = white_balls / (1 - red_frequency)) :
  total_balls - white_balls = 20 :=
by sorry

end NUMINAMATH_CALUDE_number_of_red_balls_l1735_173510


namespace NUMINAMATH_CALUDE_shortest_distance_line_to_circle_l1735_173505

/-- The shortest distance from a point on the line y = x + 1 to a point on the circle x^2 + y^2 + 2x + 4y + 4 = 0 is √2 - 1 -/
theorem shortest_distance_line_to_circle :
  let line := {p : ℝ × ℝ | p.2 = p.1 + 1}
  let circle := {p : ℝ × ℝ | p.1^2 + p.2^2 + 2*p.1 + 4*p.2 + 4 = 0}
  ∃ (p : ℝ × ℝ) (q : ℝ × ℝ), p ∈ line ∧ q ∈ circle ∧
    ∀ (p' : ℝ × ℝ) (q' : ℝ × ℝ), p' ∈ line → q' ∈ circle →
      Real.sqrt 2 - 1 ≤ Real.sqrt ((p'.1 - q'.1)^2 + (p'.2 - q'.2)^2) :=
by sorry

end NUMINAMATH_CALUDE_shortest_distance_line_to_circle_l1735_173505


namespace NUMINAMATH_CALUDE_rational_absolute_value_equation_l1735_173519

theorem rational_absolute_value_equation (a : ℚ) : 
  |a - 1| = 4 → (a = 5 ∨ a = -3) := by
  sorry

end NUMINAMATH_CALUDE_rational_absolute_value_equation_l1735_173519


namespace NUMINAMATH_CALUDE_clock_shows_four_fifty_l1735_173590

/-- Represents a clock hand --/
inductive ClockHand
| A
| B
| C

/-- Represents the position of a clock hand --/
structure HandPosition where
  hand : ClockHand
  exactHourMarker : Bool

/-- Represents a clock with three hands --/
structure Clock where
  hands : List HandPosition
  handsEqualLength : Bool

/-- Theorem stating that given the specific clock configuration, the time shown is 4:50 --/
theorem clock_shows_four_fifty (c : Clock) 
  (h1 : c.handsEqualLength = true)
  (h2 : c.hands.length = 3)
  (h3 : ∃ h ∈ c.hands, h.hand = ClockHand.A ∧ h.exactHourMarker = true)
  (h4 : ∃ h ∈ c.hands, h.hand = ClockHand.B ∧ h.exactHourMarker = true)
  (h5 : ∃ h ∈ c.hands, h.hand = ClockHand.C ∧ h.exactHourMarker = false) :
  ∃ (hour : Nat) (minute : Nat), hour = 4 ∧ minute = 50 := by
  sorry


end NUMINAMATH_CALUDE_clock_shows_four_fifty_l1735_173590


namespace NUMINAMATH_CALUDE_tangent_line_derivative_l1735_173533

variable (f : ℝ → ℝ)

theorem tangent_line_derivative (h : ∀ y, y = (1/2) * 1 + 3 → y = f 1) :
  deriv f 1 = 1/2 := by sorry

end NUMINAMATH_CALUDE_tangent_line_derivative_l1735_173533


namespace NUMINAMATH_CALUDE_triangle_area_is_four_thirds_l1735_173589

-- Define the line m: 3x - y + 2 = 0
def line_m (x y : ℝ) : Prop := 3 * x - y + 2 = 0

-- Define the symmetric line l with respect to the x-axis
def line_l (x y : ℝ) : Prop := 3 * x + y + 2 = 0

-- Define the y-axis
def y_axis (x : ℝ) : Prop := x = 0

-- Theorem statement
theorem triangle_area_is_four_thirds :
  ∃ (x₁ y₁ x₂ y₂ x₃ y₃ : ℝ),
    line_m x₁ y₁ ∧ y_axis x₁ ∧
    line_m x₂ y₂ ∧ x₂ = -2/3 ∧ y₂ = 0 ∧
    line_l x₃ y₃ ∧ y_axis x₃ ∧
    (1/2 * abs (x₂ * (y₁ - y₃))) = 4/3 :=
sorry

end NUMINAMATH_CALUDE_triangle_area_is_four_thirds_l1735_173589


namespace NUMINAMATH_CALUDE_range_of_a_when_proposition_false_l1735_173543

theorem range_of_a_when_proposition_false (a : ℝ) :
  (∀ t : ℝ, t^2 - 2*t - a ≥ 0) → a ≤ -1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_when_proposition_false_l1735_173543


namespace NUMINAMATH_CALUDE_quadratic_roots_range_l1735_173557

theorem quadratic_roots_range (k : ℝ) : 
  (∃ x y : ℝ, x ≠ y ∧ x^2 - 4*x + 1 = -2*k ∧ y^2 - 4*y + 1 = -2*k) → k < 3/2 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_range_l1735_173557


namespace NUMINAMATH_CALUDE_min_value_of_D_l1735_173527

noncomputable def D (x a : ℝ) : ℝ := Real.sqrt ((x - a)^2 + (Real.exp x - 2 * Real.sqrt a)) + a + 2

theorem min_value_of_D :
  ∃ (min_D : ℝ), min_D = Real.sqrt 2 + 1 ∧
  ∀ (x a : ℝ), D x a ≥ min_D :=
sorry

end NUMINAMATH_CALUDE_min_value_of_D_l1735_173527


namespace NUMINAMATH_CALUDE_chess_tournament_players_l1735_173584

/-- Represents a chess tournament with the given conditions -/
structure ChessTournament where
  n : ℕ  -- number of players not in the lowest 8
  /-- Total number of players is n + 8 -/
  total_players : ℕ := n + 8
  /-- Each player played exactly one game against each other player -/
  total_games : ℕ := (total_players * (total_players - 1)) / 2
  /-- Points earned by n players against each other -/
  n_vs_n_points : ℕ := (n * (n - 1)) / 2
  /-- Points earned by n players against 8 lowest players -/
  n_vs_8_points : ℕ := n_vs_n_points
  /-- Points earned by 8 lowest players among themselves -/
  lowest_8_points : ℕ := 28
  /-- Total points in the tournament -/
  total_points : ℕ := 2 * n_vs_n_points + 2 * lowest_8_points

/-- The theorem stating that the total number of players is 22 -/
theorem chess_tournament_players : ∀ t : ChessTournament, t.total_players = 22 := by
  sorry

end NUMINAMATH_CALUDE_chess_tournament_players_l1735_173584


namespace NUMINAMATH_CALUDE_linear_combination_existence_l1735_173581

theorem linear_combination_existence (n : ℕ+) (a b c : ℕ+) 
  (ha : a ≤ 3*n^2 + 4*n) (hb : b ≤ 3*n^2 + 4*n) (hc : c ≤ 3*n^2 + 4*n) :
  ∃ (x y z : ℤ), 
    (abs x ≤ 2*n ∧ abs y ≤ 2*n ∧ abs z ≤ 2*n) ∧ 
    (x ≠ 0 ∨ y ≠ 0 ∨ z ≠ 0) ∧
    (a*x + b*y + c*z = 0) :=
by sorry

end NUMINAMATH_CALUDE_linear_combination_existence_l1735_173581


namespace NUMINAMATH_CALUDE_average_of_first_three_l1735_173592

theorem average_of_first_three (A B C D : ℝ) : 
  (B + C + D) / 3 = 5 → 
  A + D = 11 → 
  D = 4 → 
  (A + B + C) / 3 = 6 := by
sorry

end NUMINAMATH_CALUDE_average_of_first_three_l1735_173592


namespace NUMINAMATH_CALUDE_expression_equals_36_l1735_173506

theorem expression_equals_36 (x : ℝ) : (x + 2)^2 + 2*(x + 2)*(4 - x) + (4 - x)^2 = 36 := by
  sorry

end NUMINAMATH_CALUDE_expression_equals_36_l1735_173506
