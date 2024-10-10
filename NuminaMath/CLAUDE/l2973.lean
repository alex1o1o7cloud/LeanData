import Mathlib

namespace committee_arrangement_l2973_297338

theorem committee_arrangement (n m : ℕ) (hn : n = 6) (hm : m = 4) : 
  Nat.choose (n + m) m = 210 := by
  sorry

end committee_arrangement_l2973_297338


namespace least_seven_ternary_correct_l2973_297341

/-- Converts a base 10 number to its ternary (base 3) representation --/
def to_ternary (n : ℕ) : List ℕ :=
  sorry

/-- Checks if a number has exactly 7 digits in its ternary representation --/
def has_seven_ternary_digits (n : ℕ) : Prop :=
  (to_ternary n).length = 7

/-- The least positive base ten number with seven ternary digits --/
def least_seven_ternary : ℕ := 729

theorem least_seven_ternary_correct :
  (has_seven_ternary_digits least_seven_ternary) ∧
  (∀ m : ℕ, m > 0 ∧ m < least_seven_ternary → ¬(has_seven_ternary_digits m)) :=
sorry

end least_seven_ternary_correct_l2973_297341


namespace min_value_when_a_is_neg_three_range_of_a_when_inequality_holds_l2973_297331

-- Define the function f(x) = |x-1| + |x-a|
def f (a : ℝ) (x : ℝ) : ℝ := |x - 1| + |x - a|

-- Part 1: Minimum value when a = -3
theorem min_value_when_a_is_neg_three :
  ∃ (min_val : ℝ), min_val = 4 ∧ ∀ x, f (-3) x ≥ min_val :=
sorry

-- Part 2: Range of a when f(x) ≤ 2a + 2|x-1| for all x ∈ ℝ
theorem range_of_a_when_inequality_holds :
  (∀ x, f a x ≤ 2*a + 2*|x - 1|) → a ≥ 1/3 :=
sorry

end min_value_when_a_is_neg_three_range_of_a_when_inequality_holds_l2973_297331


namespace unique_solution_exponential_equation_l2973_297369

theorem unique_solution_exponential_equation :
  ∃! x : ℝ, (10 : ℝ)^x * (1000 : ℝ)^x = (10000 : ℝ)^4 :=
by
  sorry

end unique_solution_exponential_equation_l2973_297369


namespace unique_balance_point_iff_m_eq_two_or_neg_one_or_one_l2973_297319

/-- A function f : ℝ → ℝ has a balance point at t if f(t) = t -/
def HasBalancePoint (f : ℝ → ℝ) (t : ℝ) : Prop :=
  f t = t

/-- A function f : ℝ → ℝ has a unique balance point if there exists exactly one t such that f(t) = t -/
def HasUniqueBalancePoint (f : ℝ → ℝ) : Prop :=
  ∃! t, HasBalancePoint f t

/-- The function we're considering -/
def f (m : ℝ) (x : ℝ) : ℝ :=
  (m - 1) * x^2 - 3 * x + 2 * m

theorem unique_balance_point_iff_m_eq_two_or_neg_one_or_one :
  ∀ m : ℝ, HasUniqueBalancePoint (f m) ↔ m = 2 ∨ m = -1 ∨ m = 1 :=
sorry

end unique_balance_point_iff_m_eq_two_or_neg_one_or_one_l2973_297319


namespace unique_positive_integer_pair_l2973_297346

theorem unique_positive_integer_pair : 
  ∃! (a b : ℕ+), 
    (b ^ 2 + b + 1 : ℤ) ≡ 0 [ZMOD a] ∧ 
    (a ^ 2 + a + 1 : ℤ) ≡ 0 [ZMOD b] ∧
    a = 1 ∧ b = 1 := by
  sorry

end unique_positive_integer_pair_l2973_297346


namespace wheat_rate_proof_l2973_297326

/-- Represents the rate of the second batch of wheat in rupees per kg -/
def second_batch_rate : ℝ := 14.25

/-- Proves that the rate of the second batch of wheat is 14.25 rupees per kg -/
theorem wheat_rate_proof (first_batch_weight : ℝ) (second_batch_weight : ℝ) 
  (first_batch_rate : ℝ) (mixture_selling_rate : ℝ) (profit_percentage : ℝ) :
  first_batch_weight = 30 →
  second_batch_weight = 20 →
  first_batch_rate = 11.50 →
  mixture_selling_rate = 15.12 →
  profit_percentage = 0.20 →
  second_batch_rate = 14.25 := by
  sorry

#check wheat_rate_proof

end wheat_rate_proof_l2973_297326


namespace arithmetic_sequence_problem_l2973_297365

/-- An arithmetic sequence with its sum function -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  S : ℕ → ℝ  -- Sum function
  is_arithmetic : ∀ n : ℕ, a (n + 2) - a (n + 1) = a (n + 1) - a n
  sum_formula : ∀ n : ℕ, S n = (n : ℝ) * (a 1 + a n) / 2

/-- Main theorem -/
theorem arithmetic_sequence_problem (seq : ArithmeticSequence)
    (h1 : seq.a 4 + seq.S 5 = 2)
    (h2 : seq.S 7 = 14) :
  seq.a 10 = 14 := by
  sorry

end arithmetic_sequence_problem_l2973_297365


namespace completing_square_equivalence_l2973_297374

theorem completing_square_equivalence :
  ∀ x : ℝ, 2 * x^2 - 4 * x - 7 = 0 ↔ (x - 1)^2 = 9/2 := by
  sorry

end completing_square_equivalence_l2973_297374


namespace z_extrema_l2973_297382

-- Define the triangle G
def G : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | p.1 ≥ 0 ∧ p.2 ≥ 0 ∧ p.1 + p.2 ≤ 4}

-- Define the function z
def z (p : ℝ × ℝ) : ℝ :=
  p.1^2 + p.2^2 - 2*p.1*p.2 - p.1 - 2*p.2

theorem z_extrema :
  (∃ p ∈ G, ∀ q ∈ G, z q ≤ z p) ∧
  (∃ p ∈ G, ∀ q ∈ G, z q ≥ z p) ∧
  (∃ p ∈ G, z p = 12) ∧
  (∃ p ∈ G, z p = -1/4) :=
sorry

end z_extrema_l2973_297382


namespace remainder_3_304_mod_11_l2973_297355

theorem remainder_3_304_mod_11 : 3^304 % 11 = 4 := by
  sorry

end remainder_3_304_mod_11_l2973_297355


namespace base3_of_256_l2973_297381

/-- Converts a base-10 number to its base-3 representation -/
def toBase3 (n : ℕ) : List ℕ :=
  sorry

theorem base3_of_256 :
  toBase3 256 = [1, 0, 1, 1, 0, 1] :=
sorry

end base3_of_256_l2973_297381


namespace divisibility_of_n_squared_n_squared_minus_one_l2973_297300

theorem divisibility_of_n_squared_n_squared_minus_one (n : ℤ) : 
  12 ∣ n^2 * (n^2 - 1) := by sorry

end divisibility_of_n_squared_n_squared_minus_one_l2973_297300


namespace gcd_47_power_plus_one_l2973_297390

theorem gcd_47_power_plus_one : Nat.gcd (47^11 + 1) (47^11 + 47^3 + 1) = 1 := by
  sorry

end gcd_47_power_plus_one_l2973_297390


namespace dividend_proof_l2973_297361

theorem dividend_proof (dividend quotient remainder : ℕ) : 
  dividend / 9 = quotient → 
  dividend % 9 = remainder →
  quotient = 9 →
  remainder = 2 →
  dividend = 83 := by
sorry

end dividend_proof_l2973_297361


namespace cousins_ages_sum_l2973_297342

theorem cousins_ages_sum (ages : Fin 5 → ℕ) 
  (mean_condition : (ages 0 + ages 1 + ages 2 + ages 3 + ages 4) / 5 = 10)
  (median_condition : ages 2 = 12)
  (sorted : ∀ i j, i ≤ j → ages i ≤ ages j) :
  ages 0 + ages 1 = 14 := by
  sorry

end cousins_ages_sum_l2973_297342


namespace intersection_set_exists_l2973_297391

/-- A structure representing a collection of subsets with specific intersection properties -/
structure IntersectionSet (k : ℕ) where
  A : Set (Set ℕ)
  infinite : Set.Infinite A
  k_intersection : ∀ (S : Finset (Set ℕ)), S.card = k → S.toSet ⊆ A → ∃! x, ∀ s ∈ S, x ∈ s
  k_plus_one_empty : ∀ (S : Finset (Set ℕ)), S.card = k + 1 → S.toSet ⊆ A → ∀ x, ∃ s ∈ S, x ∉ s

/-- Theorem stating the existence of an IntersectionSet for any k > 1 -/
theorem intersection_set_exists (k : ℕ) (h : k > 1) : ∃ I : IntersectionSet k, True := by
  sorry

end intersection_set_exists_l2973_297391


namespace area_of_overlap_l2973_297376

/-- Represents a 30-60-90 triangle -/
structure Triangle30_60_90 where
  hypotenuse : ℝ
  shortLeg : ℝ
  longLeg : ℝ
  hypotenuse_eq : hypotenuse = 10
  shortLeg_eq : shortLeg = 5
  longLeg_eq : longLeg = 5 * Real.sqrt 3

/-- Represents the configuration of two overlapping 30-60-90 triangles -/
structure OverlappingTriangles where
  triangle1 : Triangle30_60_90
  triangle2 : Triangle30_60_90
  overlap_angle : ℝ
  overlap_angle_eq : overlap_angle = 60

/-- The theorem to be proved -/
theorem area_of_overlap (ot : OverlappingTriangles) :
  let base := 2 * ot.triangle1.shortLeg
  let height := ot.triangle1.longLeg
  base * height = 50 * Real.sqrt 3 :=
sorry

end area_of_overlap_l2973_297376


namespace arithmetic_expression_equality_l2973_297367

theorem arithmetic_expression_equality : 4 * (8 - 3) + 2^3 = 28 := by
  sorry

end arithmetic_expression_equality_l2973_297367


namespace max_tickets_purchasable_l2973_297350

theorem max_tickets_purchasable (ticket_price budget : ℚ) : 
  ticket_price = 18 → budget = 150 → 
  ∃ (n : ℕ), n * ticket_price ≤ budget ∧ 
  ∀ (m : ℕ), m * ticket_price ≤ budget → m ≤ n :=
by
  sorry

end max_tickets_purchasable_l2973_297350


namespace car_speed_problem_l2973_297372

/-- Proves that the speed of Car A is 50 km/hr given the problem conditions -/
theorem car_speed_problem (speed_B time_B time_A ratio : ℝ) 
  (h1 : speed_B = 25)
  (h2 : time_B = 4)
  (h3 : time_A = 8)
  (h4 : ratio = 4)
  (h5 : ratio = (speed_A * time_A) / (speed_B * time_B)) :
  speed_A = 50 :=
by
  sorry

#check car_speed_problem

end car_speed_problem_l2973_297372


namespace third_flip_expected_value_l2973_297386

/-- The expected value of a biased coin flip -/
def expected_value (p_heads : ℚ) (win_amount : ℚ) (loss_amount : ℚ) : ℚ :=
  p_heads * win_amount + (1 - p_heads) * (-loss_amount)

theorem third_flip_expected_value :
  let p_heads : ℚ := 2/5
  let win_amount : ℚ := 4
  let loss_amount : ℚ := 6  -- doubled loss amount due to previous two tails
  expected_value p_heads win_amount loss_amount = -2 := by
sorry

end third_flip_expected_value_l2973_297386


namespace quadratic_inequality_solution_l2973_297379

theorem quadratic_inequality_solution (a : ℝ) (h : 1 < a ∧ a < 2) :
  ∃ (x : ℝ), x^2 - (a^2 + 3*a + 2)*x + 3*a*(a^2 + 2) < 0 ↔ a^2 + 2 < x ∧ x < 3*a :=
by sorry

end quadratic_inequality_solution_l2973_297379


namespace max_value_f_l2973_297334

theorem max_value_f (x : ℝ) (h : x < 3) : 
  (x^2 - 3*x + 4) / (x - 3) ≤ -1 := by sorry

end max_value_f_l2973_297334


namespace max_parts_with_parallel_lines_l2973_297397

/-- The maximum number of parts a plane can be divided into by n lines -/
def max_parts (n : ℕ) : ℕ := sorry

/-- The number of additional parts created by adding a line that intersects all existing lines -/
def additional_parts (n : ℕ) : ℕ := sorry

theorem max_parts_with_parallel_lines 
  (total_lines : ℕ) 
  (parallel_lines : ℕ) 
  (h1 : total_lines = 10) 
  (h2 : parallel_lines = 4) 
  (h3 : parallel_lines ≤ total_lines) :
  max_parts total_lines = max_parts (total_lines - parallel_lines) + 
    parallel_lines * (additional_parts (total_lines - parallel_lines)) ∧
  max_parts total_lines = 50 := by sorry

end max_parts_with_parallel_lines_l2973_297397


namespace incorrect_statement_l2973_297311

theorem incorrect_statement (p q : Prop) 
  (hp : p ↔ (2 + 2 = 5)) 
  (hq : q ↔ (3 > 2)) : 
  ¬((¬p ∧ ¬q) ∧ ¬p) := by
sorry

end incorrect_statement_l2973_297311


namespace quadratic_polynomial_proof_l2973_297324

theorem quadratic_polynomial_proof : ∃ (q : ℝ → ℝ),
  (∀ x, q x = (19 * x^2 - 2 * x + 13) / 15) ∧
  q (-2) = 9 ∧
  q 1 = 2 ∧
  q 3 = 10 := by
  sorry

end quadratic_polynomial_proof_l2973_297324


namespace tangential_quadrilateral_additive_l2973_297377

/-- A function satisfying the given condition for tangential quadrilaterals is additive. -/
theorem tangential_quadrilateral_additive 
  (f : ℝ → ℝ) 
  (h : ∀ (a b c d : ℝ), a > 0 → b > 0 → c > 0 → d > 0 → 
    (∃ (r : ℝ), r > 0 ∧ a + c = b + d ∧ a * b = r * (a + b) ∧ b * c = r * (b + c) ∧ 
      c * d = r * (c + d) ∧ d * a = r * (d + a)) → 
    f (a + b + c + d) = f a + f b + f c + f d) :
  ∀ (x y : ℝ), x > 0 → y > 0 → f (x + y) = f x + f y :=
by sorry

end tangential_quadrilateral_additive_l2973_297377


namespace cubic_factorization_l2973_297371

theorem cubic_factorization (x : ℝ) : x^3 - 4*x = x*(x+2)*(x-2) := by sorry

end cubic_factorization_l2973_297371


namespace green_tractor_price_l2973_297373

-- Define the variables and conditions
def red_tractor_price : ℕ := 20000
def red_tractor_commission : ℚ := 1/10
def green_tractor_commission : ℚ := 1/5
def red_tractors_sold : ℕ := 2
def green_tractors_sold : ℕ := 3
def total_salary : ℕ := 7000

-- Define the theorem
theorem green_tractor_price :
  ∃ (green_tractor_price : ℕ),
    green_tractor_price * green_tractor_commission * green_tractors_sold +
    red_tractor_price * red_tractor_commission * red_tractors_sold =
    total_salary ∧
    green_tractor_price = 5000 :=
by
  sorry

end green_tractor_price_l2973_297373


namespace solution_set_l2973_297348

-- Define the function f
variable (f : ℝ → ℝ)

-- Define the conditions
variable (h1 : Differentiable ℝ f)
variable (h2 : ∀ x > 0, x * (deriv (deriv f) x) < f x)
variable (h3 : f 1 = 0)

-- Define the theorem
theorem solution_set (x : ℝ) :
  {x : ℝ | x > 0 ∧ f x / x < 0} = {x : ℝ | x > 1} :=
by sorry

end solution_set_l2973_297348


namespace sophie_donuts_to_sister_l2973_297395

/-- The number of donuts Sophie gave to her sister --/
def donuts_to_sister (total_boxes : ℕ) (donuts_per_box : ℕ) (boxes_to_mom : ℕ) (donuts_for_self : ℕ) : ℕ :=
  total_boxes * donuts_per_box - boxes_to_mom * donuts_per_box - donuts_for_self

theorem sophie_donuts_to_sister :
  donuts_to_sister 4 12 1 30 = 6 := by
  sorry

end sophie_donuts_to_sister_l2973_297395


namespace sum_of_interior_angles_formula_l2973_297385

-- Define a non-crossed polygon
def NonCrossedPolygon (n : ℕ) : Type := sorry

-- Define the sum of interior angles of a polygon
def SumOfInteriorAngles (p : NonCrossedPolygon n) : ℝ := sorry

-- Theorem statement
theorem sum_of_interior_angles_formula {n : ℕ} (h : n ≥ 3) (p : NonCrossedPolygon n) :
  SumOfInteriorAngles p = (n - 2) * 180 := by sorry

end sum_of_interior_angles_formula_l2973_297385


namespace initial_gummy_worms_l2973_297357

def gummy_worms (n : ℕ) : ℕ → ℕ
  | 0 => n  -- Initial number of gummy worms
  | d + 1 => (gummy_worms n d) / 2  -- Number of gummy worms after d + 1 days

theorem initial_gummy_worms :
  ∀ n : ℕ, gummy_worms n 4 = 4 → n = 64 := by
  sorry

end initial_gummy_worms_l2973_297357


namespace division_of_decimals_l2973_297358

theorem division_of_decimals : (0.05 : ℚ) / (0.002 : ℚ) = 25 := by sorry

end division_of_decimals_l2973_297358


namespace perpendicular_lines_a_equals_zero_l2973_297323

/-- Two lines are perpendicular if the product of their slopes is -1 -/
def perpendicular (m₁ m₂ : ℝ) : Prop := m₁ * m₂ = -1

/-- Definition of line l₁ -/
def line_l₁ (a : ℝ) : ℝ → ℝ → Prop :=
  λ x y => a * x + 2 * y - 1 = 0

/-- Definition of line l₂ -/
def line_l₂ (a : ℝ) : ℝ → ℝ → Prop :=
  λ x y => 3 * x - a * y + 1 = 0

/-- The main theorem -/
theorem perpendicular_lines_a_equals_zero (a : ℝ) :
  perpendicular (a / 2) (-3 / a) → a = 0 :=
by
  sorry

end perpendicular_lines_a_equals_zero_l2973_297323


namespace intersection_implies_a_range_l2973_297339

/-- The function f(x) defined in the problem -/
def f (a : ℝ) (x : ℝ) : ℝ := x^3 - 3 * a^2 * x + 1

/-- The condition that f(x) intersects y = 3 at only one point -/
def intersects_at_one_point (a : ℝ) : Prop :=
  ∃! x : ℝ, f a x = 3

/-- The theorem statement -/
theorem intersection_implies_a_range :
  ∀ a : ℝ, intersects_at_one_point a → -1 < a ∧ a < 1 := by
  sorry

end intersection_implies_a_range_l2973_297339


namespace rectangle_area_l2973_297301

/-- Given a rectangle made from a wire of length 28 cm with a width of 6 cm, prove that its area is 48 cm². -/
theorem rectangle_area (wire_length : ℝ) (width : ℝ) (area : ℝ) :
  wire_length = 28 →
  width = 6 →
  area = (wire_length / 2 - width) * width →
  area = 48 :=
by
  sorry

end rectangle_area_l2973_297301


namespace function_extrema_sum_l2973_297396

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^x

theorem function_extrema_sum (a : ℝ) : 
  a > 0 → a ≠ 1 → 
  (∃ (max min : ℝ), 
    (∀ x ∈ Set.Icc 1 2, f a x ≤ max) ∧ 
    (∃ x ∈ Set.Icc 1 2, f a x = max) ∧
    (∀ x ∈ Set.Icc 1 2, min ≤ f a x) ∧ 
    (∃ x ∈ Set.Icc 1 2, f a x = min) ∧
    max + min = 12) →
  a = 3 := by sorry

end function_extrema_sum_l2973_297396


namespace rafael_weekly_earnings_l2973_297347

/-- Rafael's weekly earnings calculation --/
theorem rafael_weekly_earnings :
  let monday_hours : ℕ := 10
  let tuesday_hours : ℕ := 8
  let remaining_hours : ℕ := 20
  let hourly_rate : ℕ := 20

  let total_hours : ℕ := monday_hours + tuesday_hours + remaining_hours
  let weekly_earnings : ℕ := total_hours * hourly_rate

  weekly_earnings = 760 := by sorry

end rafael_weekly_earnings_l2973_297347


namespace distance_to_x_axis_l2973_297398

def point_A : ℝ × ℝ := (-3, 5)

theorem distance_to_x_axis : 
  let (x, y) := point_A
  |y| = 5 := by sorry

end distance_to_x_axis_l2973_297398


namespace mutually_exclusive_events_l2973_297325

/-- Represents the color of a ball -/
inductive BallColor
| Red
| White

/-- Represents the outcome of drawing two balls -/
structure DrawOutcome :=
  (first second : BallColor)

/-- The bag of balls -/
def bag : Multiset BallColor := 
  2 • {BallColor.Red} + 2 • {BallColor.White}

/-- Event: At least one white ball is drawn -/
def atLeastOneWhite (outcome : DrawOutcome) : Prop :=
  outcome.first = BallColor.White ∨ outcome.second = BallColor.White

/-- Event: Both balls are red -/
def bothRed (outcome : DrawOutcome) : Prop :=
  outcome.first = BallColor.Red ∧ outcome.second = BallColor.Red

/-- The probability of an event occurring -/
noncomputable def probability (event : DrawOutcome → Prop) : ℝ :=
  sorry

theorem mutually_exclusive_events :
  probability (fun outcome => atLeastOneWhite outcome ∧ bothRed outcome) = 0 ∧
  probability atLeastOneWhite + probability bothRed = 1 :=
sorry

end mutually_exclusive_events_l2973_297325


namespace expression_evaluation_l2973_297384

theorem expression_evaluation : (8^5) / (4 * 2^5 + 16) = (2^11) / 9 := by
  sorry

end expression_evaluation_l2973_297384


namespace endpoint_coordinate_sum_l2973_297317

/-- Given a line segment with one endpoint (-2, 5) and midpoint (1, 0),
    the sum of the coordinates of the other endpoint is -1. -/
theorem endpoint_coordinate_sum : 
  ∀ (x y : ℝ), 
  ((-2 + x) / 2 = 1 ∧ (5 + y) / 2 = 0) → 
  x + y = -1 :=
by sorry

end endpoint_coordinate_sum_l2973_297317


namespace point_rotation_on_circle_l2973_297387

def circle_equation (x y : ℝ) : Prop := x^2 + y^2 = 25

def rotation_45_ccw (x y x' y' : ℝ) : Prop :=
  x' = x * (Real.sqrt 2 / 2) - y * (Real.sqrt 2 / 2) ∧
  y' = x * (Real.sqrt 2 / 2) + y * (Real.sqrt 2 / 2)

theorem point_rotation_on_circle :
  ∀ (x' y' : ℝ),
    circle_equation 3 4 →
    rotation_45_ccw 3 4 x' y' →
    x' = -(Real.sqrt 2 / 2) ∧ y' = 7 * (Real.sqrt 2 / 2) :=
by sorry

end point_rotation_on_circle_l2973_297387


namespace sum_of_roots_quartic_l2973_297330

theorem sum_of_roots_quartic (x : ℝ) : 
  let f : ℝ → ℝ := λ x => 6*x^4 + 7*x^3 - 10*x^2 - x
  ∃ (r₁ r₂ r₃ r₄ : ℝ), (∀ x, f x = 0 ↔ x = r₁ ∨ x = r₂ ∨ x = r₃ ∨ x = r₄) ∧ 
    r₁ + r₂ + r₃ + r₄ = -7/6 :=
by sorry

end sum_of_roots_quartic_l2973_297330


namespace tangent_line_property_l2973_297392

/-- Given a line x + y = b tangent to the curve y = ax + 2/x at the point P(1, m), 
    prove that a + b - m = 2 -/
theorem tangent_line_property (a b m : ℝ) : 
  (∀ x, x + (a * x + 2 / x) = b) →  -- Line is tangent to the curve
  (1 + m = b) →                     -- Point P(1, m) is on the line
  (m = a + 2) →                     -- Point P(1, m) is on the curve
  (a + b - m = 2) := by
  sorry

end tangent_line_property_l2973_297392


namespace gcd_m_n_l2973_297303

def m : ℕ := 333333333
def n : ℕ := 9999999999

theorem gcd_m_n : Nat.gcd m n = 9 := by
  sorry

end gcd_m_n_l2973_297303


namespace simplify_complex_fraction_l2973_297314

theorem simplify_complex_fraction (x : ℝ) (h1 : x ≠ 2) (h2 : x ≠ -3) :
  (x + 2 - 5 / (x - 2)) / ((x + 3) / (x - 2)) = x - 3 := by
  sorry

end simplify_complex_fraction_l2973_297314


namespace regular_polygon_45_symmetry_l2973_297329

/-- A regular polygon that coincides with its original shape for the first time after rotating 45° around its center -/
structure RegularPolygon45 where
  /-- The number of sides in the polygon -/
  sides : ℕ
  /-- The polygon is regular -/
  regular : True
  /-- The polygon coincides with its original shape for the first time after rotating 45° -/
  rotation : sides * 45 = 360

/-- Axial symmetry property -/
def axially_symmetric (p : RegularPolygon45) : Prop := sorry

/-- Central symmetry property -/
def centrally_symmetric (p : RegularPolygon45) : Prop := sorry

/-- Theorem stating that a RegularPolygon45 is both axially and centrally symmetric -/
theorem regular_polygon_45_symmetry (p : RegularPolygon45) : 
  axially_symmetric p ∧ centrally_symmetric p := by sorry

end regular_polygon_45_symmetry_l2973_297329


namespace triangle_area_with_given_conditions_l2973_297394

/-- Given a triangle PQR with inradius r, circumradius R, and angles P, Q, R,
    prove that if r = 8, R = 25, and 2 * cos Q = cos P + cos R, then the area of the triangle is 96. -/
theorem triangle_area_with_given_conditions (P Q R : Real) (r R : ℝ) : 
  r = 8 → R = 25 → 2 * Real.cos Q = Real.cos P + Real.cos R → 
  ∃ (area : ℝ), area = 96 ∧ area = r * (R * Real.sin Q) := by
  sorry

end triangle_area_with_given_conditions_l2973_297394


namespace tenth_term_is_123_a_plus_b_power_10_is_123_l2973_297343

-- Define the sequence
def seq : ℕ → ℕ
| 0 => 1  -- a + b
| 1 => 3  -- a² + b²
| 2 => 4  -- a³ + b³
| n + 3 => seq (n + 1) + seq (n + 2)

-- State the theorem
theorem tenth_term_is_123 : seq 9 = 123 := by
  sorry

-- Define a and b
noncomputable def a : ℝ := sorry
noncomputable def b : ℝ := sorry

-- State the given conditions
axiom sum_1 : a + b = 1
axiom sum_2 : a^2 + b^2 = 3
axiom sum_3 : a^3 + b^3 = 4
axiom sum_4 : a^4 + b^4 = 7
axiom sum_5 : a^5 + b^5 = 11

-- State the main theorem
theorem a_plus_b_power_10_is_123 : a^10 + b^10 = 123 := by
  sorry

end tenth_term_is_123_a_plus_b_power_10_is_123_l2973_297343


namespace tangent_line_equation_l2973_297310

-- Define the curve
def f (x : ℝ) : ℝ := x^3 - 3*x^2

-- Define the derivative of the curve
def f' (x : ℝ) : ℝ := 3*x^2 - 6*x

-- Define the slope of the line parallel to 3x + y = 0
def k : ℝ := -3

-- Define the point of tangency
def x₀ : ℝ := 1
def y₀ : ℝ := f x₀

-- State the theorem
theorem tangent_line_equation :
  ∃ (x y : ℝ), 3*x + y - 1 = 0 ∧
  y - y₀ = k * (x - x₀) ∧
  f' x₀ = k ∧
  y₀ = f x₀ := by sorry

end tangent_line_equation_l2973_297310


namespace green_marbles_count_l2973_297308

/-- The number of marbles in a jar with blue, red, yellow, and green marbles -/
def total_marbles : ℕ := 164

/-- The number of yellow marbles in the jar -/
def yellow_marbles : ℕ := 14

/-- The number of blue marbles in the jar -/
def blue_marbles : ℕ := total_marbles / 2

/-- The number of red marbles in the jar -/
def red_marbles : ℕ := total_marbles / 4

/-- The number of green marbles in the jar -/
def green_marbles : ℕ := total_marbles - (blue_marbles + red_marbles + yellow_marbles)

/-- Theorem stating that the number of green marbles is 27 -/
theorem green_marbles_count : green_marbles = 27 := by
  sorry

end green_marbles_count_l2973_297308


namespace last_boat_passengers_l2973_297337

/-- The number of people on a boat trip -/
def boat_trip (m : ℕ) : Prop :=
  ∃ (total : ℕ),
    -- Condition 1: m boats with 10 seats each leaves 8 people without seats
    total = 10 * m + 8 ∧
    -- Condition 2 & 3: Using boats with 16 seats each, 1 fewer boat is rented, and last boat is not full
    ∃ (last_boat : ℕ), last_boat > 0 ∧ last_boat < 16 ∧
      total = 16 * (m - 1) + last_boat

/-- The number of people on the last boat with 16 seats -/
theorem last_boat_passengers (m : ℕ) (h : boat_trip m) :
  ∃ (last_boat : ℕ), last_boat = 40 - 6 * m :=
by sorry

end last_boat_passengers_l2973_297337


namespace cube_sum_geq_triple_product_l2973_297340

theorem cube_sum_geq_triple_product (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) :
  a^3 + b^3 + c^3 ≥ 3 * a * b * c := by
  sorry

end cube_sum_geq_triple_product_l2973_297340


namespace runners_in_quarter_segment_time_l2973_297359

/-- Represents a runner on a circular track -/
structure Runner where
  lapTime : ℕ
  direction : Bool  -- True for clockwise, False for counterclockwise

/-- Calculates the time both runners spend simultaneously in a quarter segment of the track -/
def timeInQuarterSegment (runner1 runner2 : Runner) : ℕ :=
  sorry

theorem runners_in_quarter_segment_time :
  let runner1 : Runner := { lapTime := 72, direction := true }
  let runner2 : Runner := { lapTime := 80, direction := false }
  timeInQuarterSegment runner1 runner2 = 46 := by sorry

end runners_in_quarter_segment_time_l2973_297359


namespace arcsin_sqrt2_over_2_l2973_297349

theorem arcsin_sqrt2_over_2 : 
  Real.arcsin (Real.sqrt 2 / 2) = π / 4 := by sorry

end arcsin_sqrt2_over_2_l2973_297349


namespace remainder_problem_l2973_297345

theorem remainder_problem (x : ℕ+) (h : 7 * x.val ≡ 1 [MOD 31]) : (20 + x.val) % 31 = 29 := by
  sorry

end remainder_problem_l2973_297345


namespace infinite_special_integers_l2973_297383

theorem infinite_special_integers (m : ℕ) :
  let n : ℕ := (m^2 + m + 2)^2 + (m^2 + m + 2) + 3
  ∀ p : ℕ, Prime p → p ∣ (n^2 + 3) →
    ∃ k : ℕ, k^2 < n ∧ p ∣ (k^2 + 3) :=
by
  sorry

#check infinite_special_integers

end infinite_special_integers_l2973_297383


namespace doug_has_25_marbles_l2973_297375

/-- Calculates the number of marbles Doug has given the conditions of the problem. -/
def dougs_marbles (eds_initial_advantage : ℕ) (eds_lost_marbles : ℕ) (eds_current_marbles : ℕ) : ℕ :=
  eds_current_marbles + eds_lost_marbles - eds_initial_advantage

/-- Proves that Doug has 25 marbles given the conditions of the problem. -/
theorem doug_has_25_marbles :
  dougs_marbles 12 20 17 = 25 := by
  sorry

#eval dougs_marbles 12 20 17

end doug_has_25_marbles_l2973_297375


namespace shells_found_fourth_day_l2973_297363

/-- The number of shells Shara found on the fourth day of her vacation. -/
def shells_fourth_day (initial_shells : ℕ) (shells_per_day : ℕ) (vacation_days : ℕ) (total_shells : ℕ) : ℕ :=
  total_shells - (initial_shells + shells_per_day * vacation_days)

/-- Theorem stating that Shara found 6 shells on the fourth day of her vacation. -/
theorem shells_found_fourth_day :
  shells_fourth_day 20 5 3 41 = 6 := by
  sorry

end shells_found_fourth_day_l2973_297363


namespace range_of_a_given_p_necessary_not_sufficient_for_q_l2973_297305

theorem range_of_a_given_p_necessary_not_sufficient_for_q :
  ∀ a : ℝ,
  (∀ x : ℝ, x^2 ≤ 5*x - 4 → x^2 - (a+2)*x + 2*a ≤ 0) ∧
  (∃ x : ℝ, x^2 - (a+2)*x + 2*a ≤ 0 ∧ x^2 > 5*x - 4) →
  1 ≤ a ∧ a ≤ 4 :=
by sorry

end range_of_a_given_p_necessary_not_sufficient_for_q_l2973_297305


namespace root_implies_h_value_l2973_297362

theorem root_implies_h_value (h : ℝ) : 
  ((-1 : ℝ)^3 + h * (-1) - 20 = 0) → h = -21 := by
  sorry

end root_implies_h_value_l2973_297362


namespace smallest_prime_sum_of_three_composites_l2973_297368

def is_composite (n : ℕ) : Prop := ∃ a b, 1 < a ∧ 1 < b ∧ n = a * b

theorem smallest_prime_sum_of_three_composites : 
  ∀ p : ℕ, Prime p → 
    (∃ a b c : ℕ, is_composite a ∧ is_composite b ∧ is_composite c ∧ 
      a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ p = a + b + c) → 
    p ≥ 19 :=
sorry

end smallest_prime_sum_of_three_composites_l2973_297368


namespace mobius_decomposition_l2973_297364

theorem mobius_decomposition 
  (a b c d : ℂ) 
  (h : a * d - b * c ≠ 0) : 
  ∃ (p q R : ℂ), ∀ (z : ℂ), 
    (a * z + b) / (c * z + d) = p + R / (z + q) := by
  sorry

end mobius_decomposition_l2973_297364


namespace ababa_binary_bits_l2973_297316

/-- The decimal representation of ABABA₁₆ -/
def ababa_decimal : ℕ := 701162

/-- The number of bits in the binary representation of ABABA₁₆ -/
def num_bits : ℕ := 20

theorem ababa_binary_bits :
  (2 ^ (num_bits - 1) : ℕ) ≤ ababa_decimal ∧ ababa_decimal < 2 ^ num_bits :=
by sorry

end ababa_binary_bits_l2973_297316


namespace arrangement_counts_l2973_297304

/-- Represents the number of people in the row -/
def n : ℕ := 5

/-- Calculates the factorial of a natural number -/
def factorial (k : ℕ) : ℕ :=
  match k with
  | 0 => 1
  | k + 1 => (k + 1) * factorial k

/-- The number of arrangements with Person A at the head -/
def arrangements_A_at_head : ℕ := factorial (n - 1)

/-- The number of arrangements with Person A and Person B adjacent -/
def arrangements_A_B_adjacent : ℕ := factorial (n - 1) * 2

/-- The number of arrangements with Person A not at the head and Person B not at the end -/
def arrangements_A_not_head_B_not_end : ℕ := (n - 1) * (n - 2) * factorial (n - 2)

/-- The number of arrangements with Person A to the left of and taller than Person B, and not adjacent -/
def arrangements_A_left_taller_not_adjacent : ℕ := 3 * factorial (n - 2)

theorem arrangement_counts :
  arrangements_A_at_head = 24 ∧
  arrangements_A_B_adjacent = 48 ∧
  arrangements_A_not_head_B_not_end = 72 ∧
  arrangements_A_left_taller_not_adjacent = 18 := by
  sorry

end arrangement_counts_l2973_297304


namespace unique_prime_in_form_l2973_297306

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

def form_number (B A : ℕ) : ℕ := 210000 + B * 100 + A

theorem unique_prime_in_form :
  ∃! B : ℕ, B < 10 ∧ ∃ A : ℕ, A < 10 ∧ is_prime (form_number B A) :=
sorry

end unique_prime_in_form_l2973_297306


namespace binary_decimal_octal_conversion_l2973_297336

/-- Converts a binary number represented as a list of bits to a decimal number -/
def binary_to_decimal (bits : List Bool) : ℕ :=
  bits.enum.foldl (fun acc (i, b) => acc + if b then 2^i else 0) 0

/-- Converts a decimal number to an octal number represented as a list of digits -/
def decimal_to_octal (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
    let rec go (m : ℕ) (acc : List ℕ) :=
      if m = 0 then acc
      else go (m / 8) ((m % 8) :: acc)
    go n []

/-- The binary representation of 11011100₂ -/
def binary_num : List Bool := [false, false, true, true, true, false, true, true]

theorem binary_decimal_octal_conversion :
  (binary_to_decimal binary_num = 110) ∧
  (decimal_to_octal 110 = [1, 5, 6]) := by
  sorry


end binary_decimal_octal_conversion_l2973_297336


namespace complement_event_A_equiv_l2973_297360

/-- The number of products in the sample -/
def sample_size : ℕ := 10

/-- Event A: there are at least 2 defective products -/
def event_A (defective : ℕ) : Prop := defective ≥ 2

/-- The complement of event A -/
def complement_A (defective : ℕ) : Prop := ¬(event_A defective)

/-- Theorem: The complement of "at least 2 defective products" is "at most 1 defective product" -/
theorem complement_event_A_equiv :
  ∀ defective : ℕ, defective ≤ sample_size →
    complement_A defective ↔ defective ≤ 1 := by sorry

end complement_event_A_equiv_l2973_297360


namespace one_unpainted_cube_l2973_297351

/-- A cube painted on all surfaces and cut into 27 equal smaller cubes -/
structure PaintedCube where
  /-- The total number of smaller cubes -/
  total_cubes : ℕ
  /-- The number of smaller cubes with no painted surfaces -/
  unpainted_cubes : ℕ
  /-- Assertion that the total number of smaller cubes is 27 -/
  total_is_27 : total_cubes = 27

/-- Theorem stating that exactly one smaller cube has no painted surfaces -/
theorem one_unpainted_cube (c : PaintedCube) : c.unpainted_cubes = 1 := by
  sorry

end one_unpainted_cube_l2973_297351


namespace inequality_theorem_l2973_297318

theorem inequality_theorem (a b : ℝ) (h1 : b ≠ -1) (h2 : b ≠ 0) :
  (1 + a)^2 / (1 + b) ≤ 1 + a^2 / b ↔
    ((a ≠ b ∧ (b < -1 ∨ b > 0)) ∨ (a = b ∧ a ≠ -1 ∧ a ≠ 0)) :=
by sorry

end inequality_theorem_l2973_297318


namespace total_supervisors_is_25_l2973_297335

/-- The total number of supervisors on 5 buses -/
def total_supervisors : ℕ := 4 + 5 + 3 + 6 + 7

/-- Theorem stating that the total number of supervisors is 25 -/
theorem total_supervisors_is_25 : total_supervisors = 25 := by
  sorry

end total_supervisors_is_25_l2973_297335


namespace range_of_a_l2973_297393

theorem range_of_a (a : ℝ) : (∀ x : ℝ, x > 0 → a < x + 1/x) → a < 2 := by
  sorry

end range_of_a_l2973_297393


namespace andrews_apples_l2973_297321

theorem andrews_apples (n : ℕ) : 
  (6 * n = 5 * (n + 2)) → (6 * n = 60) := by
  sorry

end andrews_apples_l2973_297321


namespace dinos_third_gig_rate_l2973_297354

/-- Dino's monthly income calculation -/
def monthly_income (hours1 hours2 hours3 : ℕ) (rate1 rate2 rate3 : ℚ) : ℚ :=
  hours1 * rate1 + hours2 * rate2 + hours3 * rate3

/-- Theorem: Dino's hourly rate for the third gig is $40/hour -/
theorem dinos_third_gig_rate :
  ∀ (rate3 : ℚ),
  monthly_income 20 30 5 10 20 rate3 = 1000 →
  rate3 = 40 := by
sorry

end dinos_third_gig_rate_l2973_297354


namespace min_sum_of_product_l2973_297344

theorem min_sum_of_product (a b : ℤ) (h : a * b = 144) : a + b ≥ -24 := by
  sorry

end min_sum_of_product_l2973_297344


namespace optimal_price_maximizes_profit_l2973_297315

/-- Represents the price of type A Kiwi in yuan -/
def a : ℝ := 35

/-- Represents the price of type B Kiwi in yuan -/
def b : ℝ := 50

/-- The cost of 2 type A and 1 type B Kiwi is 120 yuan -/
axiom cost_equation_1 : 2 * a + b = 120

/-- The cost of 3 type A and 2 type B Kiwi is 205 yuan -/
axiom cost_equation_2 : 3 * a + 2 * b = 205

/-- The cost price of each type B Kiwi is 40 yuan -/
def cost_B : ℝ := 40

/-- Daily sales of type B Kiwi at price b -/
def initial_sales : ℝ := 100

/-- Decrease in sales for each yuan increase in price -/
def sales_decrease : ℝ := 5

/-- Daily profit function for type B Kiwi -/
def profit (x : ℝ) : ℝ := (x - cost_B) * (initial_sales - sales_decrease * (x - b))

/-- The optimal selling price for type B Kiwi -/
def optimal_price : ℝ := 55

/-- The maximum daily profit for type B Kiwi -/
def max_profit : ℝ := 1125

/-- Theorem stating that the optimal price maximizes the profit -/
theorem optimal_price_maximizes_profit :
  ∀ x : ℝ, profit x ≤ profit optimal_price ∧ profit optimal_price = max_profit :=
sorry

end optimal_price_maximizes_profit_l2973_297315


namespace externally_tangent_case_intersecting_case_l2973_297353

-- Define the circles
def circle_O₁ (x y : ℝ) : Prop := x^2 + (y + 1)^2 = 4
def center_O₂ : ℝ × ℝ := (2, 1)

-- Define the equations for O₂
def equation_O₂_tangent (x y : ℝ) : Prop := (x - 2)^2 + (y - 1)^2 = 12 - 8 * Real.sqrt 2
def equation_O₂_intersect_1 (x y : ℝ) : Prop := (x - 2)^2 + (y - 1)^2 = 4
def equation_O₂_intersect_2 (x y : ℝ) : Prop := (x - 2)^2 + (y - 1)^2 = 20

-- Theorem for externally tangent case
theorem externally_tangent_case :
  (∀ x y, circle_O₁ x y → ¬equation_O₂_tangent x y) ∧
  (∃ x y, circle_O₁ x y ∧ equation_O₂_tangent x y) →
  ∀ x y, equation_O₂_tangent x y :=
sorry

-- Theorem for intersecting case
theorem intersecting_case (A B : ℝ × ℝ) :
  (A ≠ B) ∧
  (∀ x y, circle_O₁ x y ↔ ((x - A.1)^2 + (y - A.2)^2 = 0 ∨ (x - B.1)^2 + (y - B.2)^2 = 0)) ∧
  ((A.1 - B.1)^2 + (A.2 - B.2)^2 = 8) →
  (∀ x y, equation_O₂_intersect_1 x y ∨ equation_O₂_intersect_2 x y) :=
sorry

end externally_tangent_case_intersecting_case_l2973_297353


namespace angle_X_measure_l2973_297388

-- Define a triangle XYZ
structure Triangle :=
  (X Y Z : ℝ)
  (sum_angles : X + Y + Z = 180)
  (all_positive : 0 < X ∧ 0 < Y ∧ 0 < Z)

-- State the theorem
theorem angle_X_measure (t : Triangle) 
  (h1 : t.Z = 3 * t.Y) 
  (h2 : t.Y = 15) : 
  t.X = 120 := by
  sorry

end angle_X_measure_l2973_297388


namespace carmela_money_distribution_l2973_297328

/-- Proves that Carmela giving $1 to each cousin results in equal money distribution -/
theorem carmela_money_distribution (carmela_initial : ℕ) (cousin_initial : ℕ) 
  (num_cousins : ℕ) (amount_given : ℕ) : 
  carmela_initial = 7 →
  cousin_initial = 2 →
  num_cousins = 4 →
  amount_given = 1 →
  (carmela_initial - num_cousins * amount_given) = 
  (cousin_initial + amount_given) :=
by
  sorry

end carmela_money_distribution_l2973_297328


namespace jupiter_properties_l2973_297380

/-- Given orbital parameters of a moon, calculate properties of Jupiter -/
theorem jupiter_properties 
  (T : ℝ) -- Orbital period of the moon
  (R : ℝ) -- Orbital distance of the moon
  (f : ℝ) -- Gravitational constant
  (ρ : ℝ) -- Radius of Jupiter
  (V : ℝ) -- Volume of Jupiter
  (T_rot : ℝ) -- Rotational period of Jupiter
  (h₁ : T > 0)
  (h₂ : R > 0)
  (h₃ : f > 0)
  (h₄ : ρ > 0)
  (h₅ : V > 0)
  (h₆ : T_rot > 0) :
  ∃ (M σ g₁ Cf : ℝ),
    M = 4 * Real.pi^2 * R^3 / (f * T^2) ∧
    σ = M / V ∧
    g₁ = f * M / ρ^2 ∧
    Cf = 4 * Real.pi^2 * ρ / T_rot^2 :=
by
  sorry


end jupiter_properties_l2973_297380


namespace simplify_expression_l2973_297312

theorem simplify_expression (x y : ℝ) : x^5 * x^3 * y^2 * y^4 = x^8 * y^6 := by
  sorry

end simplify_expression_l2973_297312


namespace follower_point_coords_follower_on_axis_follower_distance_l2973_297307

-- Define a-level follower point
def a_level_follower (a : ℝ) (P : ℝ × ℝ) : ℝ × ℝ :=
  let (x, y) := P
  (x + a * y, a * x + y)

-- Statement 1
theorem follower_point_coords : a_level_follower 3 (-3, 5) = (12, -4) := by sorry

-- Statement 2
theorem follower_on_axis (c : ℝ) : 
  (∃ x y, a_level_follower (-3) (c, 2*c + 2) = (x, y) ∧ (x = 0 ∨ y = 0)) →
  a_level_follower (-3) (c, 2*c + 2) = (-16, 0) ∨ 
  a_level_follower (-3) (c, 2*c + 2) = (0, 16/5) := by sorry

-- Statement 3
theorem follower_distance (x : ℝ) (a : ℝ) :
  x > 0 →
  let P : ℝ × ℝ := (x, 0)
  let P3 := a_level_follower a P
  let PP3_length := Real.sqrt ((P3.1 - P.1)^2 + (P3.2 - P.2)^2)
  let OP_length := Real.sqrt (P.1^2 + P.2^2)
  PP3_length = 2 * OP_length →
  a = 2 ∨ a = -2 := by sorry

end follower_point_coords_follower_on_axis_follower_distance_l2973_297307


namespace quadratic_equation_solution_l2973_297366

theorem quadratic_equation_solution :
  let f : ℂ → ℂ := λ x => x^2 + 6*x + 8 + (x + 2)*(x + 6)
  (f (-3 + I) = 0) ∧ (f (-3 - I) = 0) :=
by sorry

end quadratic_equation_solution_l2973_297366


namespace simplify_expression_l2973_297327

theorem simplify_expression (x : ℚ) : 
  (3 * x + 6 - 5 * x) / 3 = -2/3 * x + 2 := by
  sorry

end simplify_expression_l2973_297327


namespace serenity_shoes_pairs_serenity_bought_three_pairs_l2973_297320

theorem serenity_shoes_pairs : ℕ → ℕ → ℕ → Prop :=
  fun total_shoes shoes_per_pair pairs_bought =>
    total_shoes = 6 ∧ shoes_per_pair = 2 →
    pairs_bought = total_shoes / shoes_per_pair ∧
    pairs_bought = 3

-- Proof
theorem serenity_bought_three_pairs : serenity_shoes_pairs 6 2 3 := by
  sorry

end serenity_shoes_pairs_serenity_bought_three_pairs_l2973_297320


namespace min_value_quadratic_form_l2973_297313

theorem min_value_quadratic_form (x y z : ℝ) (h : 3 * x + 2 * y + z = 1) :
  ∃ (m : ℝ), m = 3 / 34 ∧ x^2 + 2 * y^2 + 3 * z^2 ≥ m ∧
  ∃ (x₀ y₀ z₀ : ℝ), 3 * x₀ + 2 * y₀ + z₀ = 1 ∧ x₀^2 + 2 * y₀^2 + 3 * z₀^2 = m :=
by sorry

end min_value_quadratic_form_l2973_297313


namespace andrea_pony_cost_l2973_297389

/-- The total annual cost for Andrea's pony -/
def annual_pony_cost (monthly_pasture_rent : ℕ) (daily_food_cost : ℕ) (lesson_cost : ℕ) 
  (lessons_per_week : ℕ) (months_per_year : ℕ) (days_per_year : ℕ) (weeks_per_year : ℕ) : ℕ :=
  monthly_pasture_rent * months_per_year +
  daily_food_cost * days_per_year +
  lesson_cost * lessons_per_week * weeks_per_year

theorem andrea_pony_cost :
  annual_pony_cost 500 10 60 2 12 365 52 = 15890 := by
  sorry

end andrea_pony_cost_l2973_297389


namespace total_students_count_l2973_297309

/-- The number of students wishing to go on a scavenger hunting trip -/
def scavenger_hunting : ℕ := 4000

/-- The number of students wishing to go on a skiing trip -/
def skiing : ℕ := 2 * scavenger_hunting

/-- The number of students wishing to go on a camping trip -/
def camping : ℕ := skiing + (skiing * 15 / 100)

/-- The total number of students wishing to go on any trip -/
def total_students : ℕ := scavenger_hunting + skiing + camping

theorem total_students_count : total_students = 21200 := by
  sorry

end total_students_count_l2973_297309


namespace lineup_theorem_l2973_297352

def total_people : ℕ := 7
def selected_people : ℕ := 5

def ways_including_A : ℕ := 1800
def ways_not_all_ABC : ℕ := 1800
def ways_ABC_adjacent : ℕ := 144

theorem lineup_theorem :
  (ways_including_A = 1800) ∧
  (ways_not_all_ABC = 1800) ∧
  (ways_ABC_adjacent = 144) :=
by sorry

end lineup_theorem_l2973_297352


namespace inequality_proof_l2973_297378

theorem inequality_proof (a b c d : ℝ) (h1 : a > b) (h2 : c > d) :
  a - d > b - c := by
  sorry

end inequality_proof_l2973_297378


namespace wizard_potion_combinations_l2973_297370

/-- Represents the number of valid potion combinations given the constraints. -/
def validPotionCombinations (plants : ℕ) (gemstones : ℕ) 
  (incompatible_2gem_1plant : ℕ) (incompatible_1gem_2plant : ℕ) : ℕ :=
  plants * gemstones - (incompatible_2gem_1plant + 2 * incompatible_1gem_2plant)

/-- Theorem stating that given the specific constraints, there are 20 valid potion combinations. -/
theorem wizard_potion_combinations : 
  validPotionCombinations 4 6 2 1 = 20 := by
  sorry

end wizard_potion_combinations_l2973_297370


namespace point_in_fourth_quadrant_l2973_297322

/-- A linear function y = ax + b where y increases as x increases and ab < 0 -/
structure LinearFunction where
  a : ℝ
  b : ℝ
  increasing : a > 0
  product_negative : a * b < 0

/-- The point P(a,b) -/
def point (f : LinearFunction) : ℝ × ℝ := (f.a, f.b)

/-- A point (x,y) lies in the fourth quadrant if x > 0 and y < 0 -/
def in_fourth_quadrant (p : ℝ × ℝ) : Prop :=
  p.1 > 0 ∧ p.2 < 0

theorem point_in_fourth_quadrant (f : LinearFunction) :
  in_fourth_quadrant (point f) := by
  sorry

end point_in_fourth_quadrant_l2973_297322


namespace percentage_failed_english_l2973_297333

theorem percentage_failed_english (failed_hindi : ℝ) (failed_both : ℝ) (passed_both : ℝ)
  (h1 : failed_hindi = 34)
  (h2 : failed_both = 22)
  (h3 : passed_both = 44) :
  ∃ failed_english : ℝ,
    failed_english = 44 ∧
    failed_hindi + failed_english - failed_both = 100 - passed_both :=
by sorry

end percentage_failed_english_l2973_297333


namespace asymptotic_stability_l2973_297332

noncomputable section

/-- The system of differential equations -/
def system (x y : ℝ) : ℝ × ℝ :=
  (y - x/2 - x*y^3/2, -y - 2*x + x^2*y^2)

/-- The Lyapunov function candidate -/
def V (x y : ℝ) : ℝ :=
  2*x^2 + y^2

/-- The time derivative of V along the system trajectories -/
def dVdt (x y : ℝ) : ℝ :=
  let (dx, dy) := system x y
  4*x*dx + 2*y*dy

theorem asymptotic_stability :
  ∃ δ > 0, ∀ x y : ℝ, x^2 + y^2 < δ^2 →
    (∀ t : ℝ, t ≥ 0 → 
      let (xt, yt) := system x y
      V xt yt ≤ V x y ∧ (x ≠ 0 ∨ y ≠ 0 → V xt yt < V x y)) ∧
    (∀ ε > 0, ∃ T : ℝ, T > 0 → 
      let (xT, yT) := system x y
      xT^2 + yT^2 < ε^2) :=
sorry

end

end asymptotic_stability_l2973_297332


namespace circle_equation_correct_l2973_297399

/-- The line on which the circle's center lies -/
def center_line (x y : ℝ) : Prop := y = -4 * x

/-- The line tangent to the circle -/
def tangent_line (x y : ℝ) : Prop := x + y - 1 = 0

/-- The point of tangency -/
def tangent_point : ℝ × ℝ := (3, -2)

/-- The equation of the circle -/
def circle_equation (x y : ℝ) : Prop := (x - 1)^2 + (y + 4)^2 = 8

theorem circle_equation_correct :
  ∃ (c : ℝ × ℝ), 
    center_line c.1 c.2 ∧
    (∃ (r : ℝ), r > 0 ∧
      ∀ (p : ℝ × ℝ), 
        circle_equation p.1 p.2 ↔ (p.1 - c.1)^2 + (p.2 - c.2)^2 = r^2) ∧
    tangent_line tangent_point.1 tangent_point.2 ∧
    circle_equation tangent_point.1 tangent_point.2 :=
sorry

end circle_equation_correct_l2973_297399


namespace peanut_plantation_revenue_l2973_297302

-- Define the plantation and region sizes
def plantation_size : ℕ × ℕ := (500, 500)
def region_a_size : ℕ × ℕ := (200, 300)
def region_b_size : ℕ × ℕ := (200, 200)
def region_c_size : ℕ × ℕ := (100, 500)

-- Define production rates (grams per square foot)
def region_a_rate : ℕ := 60
def region_b_rate : ℕ := 45
def region_c_rate : ℕ := 30

-- Define peanut butter production rate
def peanut_to_butter_ratio : ℚ := 5 / 20

-- Define monthly selling prices (dollars per kg)
def monthly_prices : List ℚ := [12, 10, 14, 8, 11]

-- Function to calculate area
def area (size : ℕ × ℕ) : ℕ := size.1 * size.2

-- Function to calculate peanut production for a region
def region_production (size : ℕ × ℕ) (rate : ℕ) : ℕ := area size * rate

-- Calculate total peanut production
def total_peanut_production : ℕ :=
  region_production region_a_size region_a_rate +
  region_production region_b_size region_b_rate +
  region_production region_c_size region_c_rate

-- Calculate peanut butter production in kg
def peanut_butter_production : ℚ :=
  (total_peanut_production : ℚ) * peanut_to_butter_ratio / 1000

-- Calculate total revenue
def total_revenue : ℚ :=
  monthly_prices.foldl (fun acc price => acc + price * peanut_butter_production) 0

-- Theorem statement
theorem peanut_plantation_revenue :
  total_revenue = 94875 := by sorry

end peanut_plantation_revenue_l2973_297302


namespace triangle_game_probability_l2973_297356

/-- A game board constructed from an equilateral triangle -/
structure GameBoard :=
  (total_sections : ℕ)
  (shaded_sections : ℕ)
  (h_positive : 0 < total_sections)
  (h_shaded_le_total : shaded_sections ≤ total_sections)

/-- The probability of the spinner landing in a shaded region -/
def landing_probability (board : GameBoard) : ℚ :=
  board.shaded_sections / board.total_sections

/-- Theorem stating that for a game board with 6 total sections and 2 shaded sections,
    the probability of landing in a shaded region is 1/3 -/
theorem triangle_game_probability :
  ∀ (board : GameBoard),
    board.total_sections = 6 →
    board.shaded_sections = 2 →
    landing_probability board = 1/3 :=
by sorry

end triangle_game_probability_l2973_297356
