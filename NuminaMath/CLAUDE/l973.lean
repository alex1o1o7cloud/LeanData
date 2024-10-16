import Mathlib

namespace NUMINAMATH_CALUDE_sand_bag_cost_l973_97317

/-- The cost of a bag of sand given the dimensions of a square sandbox,
    the area covered by one bag, and the total cost to fill the sandbox. -/
theorem sand_bag_cost
  (sandbox_side : ℝ)
  (bag_area : ℝ)
  (total_cost : ℝ)
  (h_square : sandbox_side = 3)
  (h_bag : bag_area = 3)
  (h_cost : total_cost = 12) :
  total_cost / (sandbox_side ^ 2 / bag_area) = 4 := by
sorry

end NUMINAMATH_CALUDE_sand_bag_cost_l973_97317


namespace NUMINAMATH_CALUDE_tan_alpha_equals_two_l973_97322

theorem tan_alpha_equals_two (α : ℝ) 
  (h : Real.sin (2 * α + Real.pi / 4) - 7 * Real.sin (2 * α + 3 * Real.pi / 4) = 5 * Real.sqrt 2) : 
  Real.tan α = 2 := by
  sorry

end NUMINAMATH_CALUDE_tan_alpha_equals_two_l973_97322


namespace NUMINAMATH_CALUDE_undefined_rock_ratio_l973_97303

-- Define the number of rocks Ted and Bill toss
def ted_rocks : ℕ := 10
def bill_rocks : ℕ := 0

-- Define a function to calculate the ratio
def rock_ratio (a b : ℕ) : Option ℚ :=
  if b = 0 then none else some (a / b)

-- Theorem statement
theorem undefined_rock_ratio :
  rock_ratio ted_rocks bill_rocks = none := by
sorry

end NUMINAMATH_CALUDE_undefined_rock_ratio_l973_97303


namespace NUMINAMATH_CALUDE_vector_magnitude_range_function_f_range_l973_97300

noncomputable section

def x : ℝ := sorry

-- Define vector a
def a : ℝ × ℝ := (Real.sin x + Real.cos x, Real.sqrt 2 * Real.cos x)

-- Define vector b
def b : ℝ × ℝ := (Real.cos x - Real.sin x, Real.sqrt 2 * Real.sin x)

-- Define the dot product of two 2D vectors
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

-- Define the magnitude of a 2D vector
def magnitude (v : ℝ × ℝ) : ℝ := Real.sqrt (v.1^2 + v.2^2)

-- Define the function f(x)
def f : ℝ → ℝ := λ x => dot_product a b - magnitude a

theorem vector_magnitude_range :
  x ∈ Set.Icc (-Real.pi / 8) 0 →
  magnitude a ∈ Set.Icc (Real.sqrt 2) (Real.sqrt 3) :=
sorry

theorem function_f_range :
  x ∈ Set.Icc (-Real.pi / 8) 0 →
  f x ∈ Set.Icc (-Real.sqrt 2) (1 - Real.sqrt 3) :=
sorry

end NUMINAMATH_CALUDE_vector_magnitude_range_function_f_range_l973_97300


namespace NUMINAMATH_CALUDE_g_equals_2x_minus_1_l973_97399

-- Define the functions f and g
def f (x : ℝ) : ℝ := 2 * x + 3

-- Define the property of g in relation to f
def g_property (g : ℝ → ℝ) : Prop :=
  ∀ x, g (x + 2) = f x

-- Theorem statement
theorem g_equals_2x_minus_1 (g : ℝ → ℝ) (h : g_property g) :
  ∀ x, g x = 2 * x - 1 := by
  sorry

end NUMINAMATH_CALUDE_g_equals_2x_minus_1_l973_97399


namespace NUMINAMATH_CALUDE_max_value_of_f_l973_97350

def f (x : ℝ) := 12 * x - 4 * x^2

theorem max_value_of_f :
  ∃ (c : ℝ), ∀ (x : ℝ), f x ≤ c ∧ ∃ (x₀ : ℝ), f x₀ = c ∧ c = 9 :=
sorry

end NUMINAMATH_CALUDE_max_value_of_f_l973_97350


namespace NUMINAMATH_CALUDE_sweater_price_calculation_l973_97321

/-- Given the price of shirts and the price difference between shirts and sweaters,
    calculate the total price of sweaters. -/
theorem sweater_price_calculation (shirt_total : ℕ) (shirt_count : ℕ) (sweater_count : ℕ)
    (price_difference : ℕ) (h1 : shirt_total = 360) (h2 : shirt_count = 20)
    (h3 : sweater_count = 45) (h4 : price_difference = 2) :
    let shirt_avg : ℚ := shirt_total / shirt_count
    let sweater_avg : ℚ := shirt_avg + price_difference
    sweater_avg * sweater_count = 900 := by
  sorry

end NUMINAMATH_CALUDE_sweater_price_calculation_l973_97321


namespace NUMINAMATH_CALUDE_pool_filling_time_l973_97332

def tap1_time : ℝ := 3
def tap2_time : ℝ := 6
def tap3_time : ℝ := 12

theorem pool_filling_time :
  let combined_rate := 1 / tap1_time + 1 / tap2_time + 1 / tap3_time
  (1 / combined_rate) = 12 / 7 :=
by sorry

end NUMINAMATH_CALUDE_pool_filling_time_l973_97332


namespace NUMINAMATH_CALUDE_f_mono_increasing_condition_l973_97396

/-- A quadratic function f(x) = ax^2 + x + 1 -/
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + x + 1

/-- The property of being monotonically increasing on (0, +∞) -/
def MonoIncreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, 0 < x ∧ x < y → f x < f y

/-- The condition a ≥ 0 is sufficient but not necessary for f to be monotonically increasing on (0, +∞) -/
theorem f_mono_increasing_condition (a : ℝ) :
  (a ≥ 0 → MonoIncreasing (f a)) ∧
  ¬(MonoIncreasing (f a) → a ≥ 0) :=
sorry

end NUMINAMATH_CALUDE_f_mono_increasing_condition_l973_97396


namespace NUMINAMATH_CALUDE_completing_square_quadratic_l973_97302

theorem completing_square_quadratic (x : ℝ) : 
  x^2 - 4*x + 3 = 0 ↔ (x - 2)^2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_completing_square_quadratic_l973_97302


namespace NUMINAMATH_CALUDE_point_between_parallel_lines_l973_97339

theorem point_between_parallel_lines :
  ∃ (b : ℤ),
    (31 - 8 * b) * (20 - 4 * b) < 0 ∧
    b = 4 :=
by sorry

end NUMINAMATH_CALUDE_point_between_parallel_lines_l973_97339


namespace NUMINAMATH_CALUDE_new_figure_length_l973_97394

/-- A polygon with adjacent perpendicular sides -/
structure PerpendicularPolygon where
  sides : List ℝ
  adjacent_perpendicular : Bool

/-- The new figure formed by removing four sides from the original polygon -/
def new_figure (p : PerpendicularPolygon) : List ℝ :=
  sorry

/-- Theorem: The total length of segments in the new figure is 22 units -/
theorem new_figure_length (p : PerpendicularPolygon) 
  (h1 : p.adjacent_perpendicular = true)
  (h2 : p.sides = [9, 3, 7, 1, 1]) :
  (new_figure p).sum = 22 := by
  sorry

end NUMINAMATH_CALUDE_new_figure_length_l973_97394


namespace NUMINAMATH_CALUDE_amy_and_noah_total_books_l973_97372

/-- The number of books owned by different people -/
structure BookCounts where
  maddie : ℕ
  luisa : ℕ
  amy : ℕ
  noah : ℕ

/-- The conditions of the book counting problem -/
def BookProblemConditions (bc : BookCounts) : Prop :=
  bc.maddie = 15 ∧
  bc.luisa = 18 ∧
  bc.amy + bc.luisa = bc.maddie + 9 ∧
  bc.noah = bc.amy / 3

/-- The theorem stating that under the given conditions, Amy and Noah have 8 books in total -/
theorem amy_and_noah_total_books (bc : BookCounts) 
  (h : BookProblemConditions bc) : bc.amy + bc.noah = 8 := by
  sorry

end NUMINAMATH_CALUDE_amy_and_noah_total_books_l973_97372


namespace NUMINAMATH_CALUDE_polynomial_factorization_l973_97374

theorem polynomial_factorization (x : ℝ) :
  x^12 + x^6 + 1 = (x^2 + x + 1) * (x^10 + x^8 + x^7 + x^5 + x^4 + x^2 + 1) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l973_97374


namespace NUMINAMATH_CALUDE_roots_sum_of_powers_l973_97326

theorem roots_sum_of_powers (α β : ℝ) : 
  α^2 - 2*α - 1 = 0 → β^2 - 2*β - 1 = 0 → 5*α^4 + 12*β^3 = 169 := by
  sorry

end NUMINAMATH_CALUDE_roots_sum_of_powers_l973_97326


namespace NUMINAMATH_CALUDE_equation_solution_inequality_system_solution_l973_97301

-- Define the equation
def equation (x : ℝ) : Prop :=
  (3 - x) / (x - 4) + 1 / (4 - x) = 1

-- Define the inequality system
def inequality_system (x : ℝ) : Prop :=
  2 * (x + 1) > x ∧ 1 - 2 * x ≥ (x + 7) / 2

theorem equation_solution :
  ∃ x : ℝ, equation x ∧ x = 3 :=
sorry

theorem inequality_system_solution :
  ∃ x : ℝ, inequality_system x ↔ -2 < x ∧ x ≤ -1 :=
sorry

end NUMINAMATH_CALUDE_equation_solution_inequality_system_solution_l973_97301


namespace NUMINAMATH_CALUDE_clock_synchronization_l973_97389

/-- The number of minutes in 12 hours -/
def minutes_in_12_hours : ℕ := 12 * 60

/-- The number of minutes Arthur's clock gains per day -/
def arthur_gain : ℕ := 15

/-- The number of minutes Oleg's clock gains per day -/
def oleg_gain : ℕ := 12

/-- The number of days it takes for Arthur's clock to gain 12 hours -/
def arthur_days : ℕ := minutes_in_12_hours / arthur_gain

/-- The number of days it takes for Oleg's clock to gain 12 hours -/
def oleg_days : ℕ := minutes_in_12_hours / oleg_gain

theorem clock_synchronization :
  Nat.lcm arthur_days oleg_days = 240 := by sorry

end NUMINAMATH_CALUDE_clock_synchronization_l973_97389


namespace NUMINAMATH_CALUDE_locus_of_points_l973_97341

/-- Two lines in a plane --/
structure TwoLines where
  l₁ : Set (ℝ × ℝ)
  l₃ : Set (ℝ × ℝ)

/-- Distance from a point to a line --/
def distanceToLine (p : ℝ × ℝ) (l : Set (ℝ × ℝ)) : ℝ := sorry

/-- Translate a line by a distance --/
def translateLine (l : Set (ℝ × ℝ)) (d : ℝ) : Set (ℝ × ℝ) := sorry

/-- Angle bisector of two lines --/
def angleBisector (l1 l2 : Set (ℝ × ℝ)) : Set (ℝ × ℝ) := sorry

/-- The theorem statement --/
theorem locus_of_points (lines : TwoLines) (a : ℝ) :
  ∀ (M : ℝ × ℝ), 
    (distanceToLine M lines.l₁ + distanceToLine M lines.l₃ = a) →
    ∃ (d : ℝ), M ∈ angleBisector lines.l₁ (translateLine lines.l₃ d) := by
  sorry

end NUMINAMATH_CALUDE_locus_of_points_l973_97341


namespace NUMINAMATH_CALUDE_regular_polygon_angle_characterization_l973_97311

def is_regular_polygon_angle (angle : ℕ) : Prop :=
  ∃ n : ℕ, n ≥ 3 ∧ angle = 180 - 360 / n

def regular_polygon_angles : Set ℕ :=
  {60, 90, 108, 120, 135, 140, 144, 150, 156, 160, 162, 165, 168, 170, 171, 172, 174, 175, 176, 177, 178, 179}

theorem regular_polygon_angle_characterization :
  ∀ angle : ℕ, is_regular_polygon_angle angle ↔ angle ∈ regular_polygon_angles :=
by sorry

end NUMINAMATH_CALUDE_regular_polygon_angle_characterization_l973_97311


namespace NUMINAMATH_CALUDE_base9_sequence_is_triangular_l973_97323

/-- Definition of triangular numbers -/
def triangular (n : ℕ) : ℕ := n * (n + 1) / 2

/-- Definition of the sequence in base-9 -/
def base9_sequence : ℕ → ℕ
  | 0 => 1
  | n + 1 => 9 * base9_sequence n + 1

/-- Theorem stating that each term in the base-9 sequence is a triangular number -/
theorem base9_sequence_is_triangular (n : ℕ) : 
  ∃ m : ℕ, base9_sequence n = triangular m := by sorry

end NUMINAMATH_CALUDE_base9_sequence_is_triangular_l973_97323


namespace NUMINAMATH_CALUDE_m_range_theorem_l973_97360

-- Define the conditions
def p (m : ℝ) : Prop := ∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ x₁ < 0 ∧ x₂ < 0 ∧ x₁^2 + m*x₁ + 4 = 0 ∧ x₂^2 + m*x₂ + 4 = 0

def q (m : ℝ) : Prop := ∀ x : ℝ, 4*x^2 + 4*(m-2)*x + 1 ≠ 0

-- Define the theorem
theorem m_range_theorem (m : ℝ) : 
  (p m ∨ q m) ∧ ¬(p m ∧ q m) → m ∈ (Set.Ioo 1 3) ∪ (Set.Ioi 4) :=
sorry

end NUMINAMATH_CALUDE_m_range_theorem_l973_97360


namespace NUMINAMATH_CALUDE_min_max_inequality_l973_97318

theorem min_max_inequality (a b c d : ℕ) : 
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d →
  ∃ (p q x m n y : ℕ),
    p = min a b ∧
    q = min c d ∧
    x = max p q ∧
    m = max a b ∧
    n = max c d ∧
    y = min m n ∧
    ((x > y) ∨ (x < y)) :=
by sorry

end NUMINAMATH_CALUDE_min_max_inequality_l973_97318


namespace NUMINAMATH_CALUDE_opponent_total_score_l973_97366

/-- Represents the score of a basketball game -/
structure GameScore where
  team : ℕ
  opponent : ℕ

/-- Calculates the total opponent score given a list of game scores -/
def totalOpponentScore (games : List GameScore) : ℕ :=
  games.foldr (fun game acc => game.opponent + acc) 0

theorem opponent_total_score : 
  ∃ (games : List GameScore),
    games.length = 12 ∧ 
    (∀ g ∈ games, 1 ≤ g.team ∧ g.team ≤ 12) ∧
    (games.filter (fun g => g.opponent = g.team + 2)).length = 6 ∧
    (∀ g ∈ games.filter (fun g => g.opponent ≠ g.team + 2), g.team = 3 * g.opponent) ∧
    totalOpponentScore games = 50 := by
  sorry


end NUMINAMATH_CALUDE_opponent_total_score_l973_97366


namespace NUMINAMATH_CALUDE_sugar_calculation_l973_97325

/-- Proves that the total amount of sugar the owner started with is 14100 grams --/
theorem sugar_calculation (total_packs : ℕ) (pack_weight : ℝ) (remaining_sugar : ℝ) :
  total_packs = 35 →
  pack_weight = 400 →
  remaining_sugar = 100 →
  (total_packs : ℝ) * pack_weight + remaining_sugar = 14100 := by
  sorry

end NUMINAMATH_CALUDE_sugar_calculation_l973_97325


namespace NUMINAMATH_CALUDE_taxi_fare_problem_l973_97340

/-- The fare structure for a taxi ride -/
structure TaxiFare where
  fixedCharge : ℝ
  ratePerMile : ℝ

/-- Calculate the total fare for a given distance -/
def totalFare (fare : TaxiFare) (distance : ℝ) : ℝ :=
  fare.fixedCharge + fare.ratePerMile * distance

/-- The problem statement -/
theorem taxi_fare_problem (fare : TaxiFare) 
  (h1 : totalFare fare 80 = 200)
  (h2 : fare.fixedCharge = 20) :
  totalFare fare 100 = 245 := by
  sorry


end NUMINAMATH_CALUDE_taxi_fare_problem_l973_97340


namespace NUMINAMATH_CALUDE_set_M_characterization_l973_97335

-- Define the set M
def M : Set ℝ := {y | ∃ x : ℝ, y = x^2 ∨ y = 1}

-- Define the set of valid x values
def valid_x : Set ℝ := {x | x ≠ 1 ∧ x ≠ -1}

-- Theorem statement
theorem set_M_characterization : 
  ∀ x : ℝ, (x^2 ∈ M ∧ 1 ∈ M ∧ x^2 ≠ 1) ↔ x ∈ valid_x :=
sorry

end NUMINAMATH_CALUDE_set_M_characterization_l973_97335


namespace NUMINAMATH_CALUDE_complement_of_N_in_M_l973_97387

def M : Set ℕ := {1, 2, 3, 4, 5}
def N : Set ℕ := {2, 5}

theorem complement_of_N_in_M :
  M \ N = {1, 3, 4} := by sorry

end NUMINAMATH_CALUDE_complement_of_N_in_M_l973_97387


namespace NUMINAMATH_CALUDE_prob_same_color_left_right_is_31_138_l973_97398

def total_pairs : ℕ := 12
def blue_pairs : ℕ := 7
def red_pairs : ℕ := 3
def green_pairs : ℕ := 2

def total_shoes : ℕ := total_pairs * 2

def prob_same_color_left_right : ℚ :=
  (blue_pairs * total_pairs + red_pairs * total_pairs + green_pairs * total_pairs) / 
  (total_shoes * (total_shoes - 1))

theorem prob_same_color_left_right_is_31_138 : 
  prob_same_color_left_right = 31 / 138 := by
  sorry

end NUMINAMATH_CALUDE_prob_same_color_left_right_is_31_138_l973_97398


namespace NUMINAMATH_CALUDE_hayden_ironing_time_l973_97379

/-- The time Hayden spends ironing his clothes over a given number of weeks -/
def ironingTime (shirtTime minutesPerDay : ℕ) (pantsTime minutesPerDay : ℕ) (daysPerWeek : ℕ) (numWeeks : ℕ) : ℕ :=
  (shirtTime + pantsTime) * daysPerWeek * numWeeks

/-- Theorem stating that Hayden spends 160 minutes ironing over 4 weeks -/
theorem hayden_ironing_time :
  ironingTime 5 3 5 4 = 160 := by
  sorry

end NUMINAMATH_CALUDE_hayden_ironing_time_l973_97379


namespace NUMINAMATH_CALUDE_vikki_take_home_pay_l973_97313

def vikki_problem (hours_worked : ℕ) (hourly_rate : ℚ) (tax_rate : ℚ) (insurance_rate : ℚ) (union_dues : ℚ) : Prop :=
  let gross_earnings := hours_worked * hourly_rate
  let tax_deduction := tax_rate * gross_earnings
  let insurance_deduction := insurance_rate * gross_earnings
  let total_deductions := tax_deduction + insurance_deduction + union_dues
  let take_home_pay := gross_earnings - total_deductions
  take_home_pay = 310

theorem vikki_take_home_pay :
  vikki_problem 42 10 (20/100) (5/100) 5 :=
sorry

end NUMINAMATH_CALUDE_vikki_take_home_pay_l973_97313


namespace NUMINAMATH_CALUDE_quadratic_inequality_integer_solution_l973_97353

theorem quadratic_inequality_integer_solution (a : ℤ) : 
  (∀ x : ℝ, x^2 + 2*↑a*x + 1 > 0) → a = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_integer_solution_l973_97353


namespace NUMINAMATH_CALUDE_loan_amount_calculation_l973_97327

/-- Calculates the total loan amount given the loan term, down payment, and monthly payment. -/
def total_loan_amount (loan_term_years : ℕ) (down_payment : ℕ) (monthly_payment : ℕ) : ℕ :=
  down_payment + loan_term_years * 12 * monthly_payment

/-- Theorem stating that given the specific loan conditions, the total loan amount is $46,000. -/
theorem loan_amount_calculation :
  total_loan_amount 5 10000 600 = 46000 := by
  sorry

end NUMINAMATH_CALUDE_loan_amount_calculation_l973_97327


namespace NUMINAMATH_CALUDE_towels_per_pack_l973_97362

/-- Given that Tiffany bought 9 packs of towels and 27 towels in total,
    prove that there were 3 towels in each pack. -/
theorem towels_per_pack (total_packs : ℕ) (total_towels : ℕ) 
  (h1 : total_packs = 9) 
  (h2 : total_towels = 27) : 
  total_towels / total_packs = 3 := by
  sorry

end NUMINAMATH_CALUDE_towels_per_pack_l973_97362


namespace NUMINAMATH_CALUDE_sector_arc_length_ratio_l973_97359

theorem sector_arc_length_ratio (r : ℝ) (h : r > 0) :
  let circle_area := π * r^2
  let sector_radius := 2 * r / 3
  let sector_area := 5 * circle_area / 27
  let circle_circumference := 2 * π * r
  ∃ α : ℝ, 
    sector_area = α * sector_radius^2 / 2 ∧ 
    (α * sector_radius) / circle_circumference = 5 / 18 :=
by
  sorry

end NUMINAMATH_CALUDE_sector_arc_length_ratio_l973_97359


namespace NUMINAMATH_CALUDE_bank_deposit_problem_l973_97397

/-- Calculates the total amount after maturity for a fixed-term deposit -/
def totalAmount (principal : ℝ) (rate : ℝ) (time : ℝ) : ℝ :=
  principal + principal * rate * time

theorem bank_deposit_problem :
  let principal : ℝ := 100000
  let rate : ℝ := 0.0315
  let time : ℝ := 2
  totalAmount principal rate time = 106300 := by
  sorry

end NUMINAMATH_CALUDE_bank_deposit_problem_l973_97397


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l973_97385

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x^2 + x + 1 ≠ 0) ↔ (∃ x : ℝ, x^2 + x + 1 = 0) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l973_97385


namespace NUMINAMATH_CALUDE_two_numbers_sum_l973_97310

theorem two_numbers_sum : ∃ (x y : ℝ), x * 15 = x + 196 ∧ y * 50 = y + 842 ∧ x + y = 31.2 := by
  sorry

end NUMINAMATH_CALUDE_two_numbers_sum_l973_97310


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l973_97324

theorem quadratic_equation_solution : 
  ∀ x : ℝ, x^2 = x ↔ x = 0 ∨ x = 1 := by sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l973_97324


namespace NUMINAMATH_CALUDE_students_not_enrolled_l973_97346

theorem students_not_enrolled (total : ℕ) (french : ℕ) (german : ℕ) (both : ℕ) 
  (h1 : total = 60)
  (h2 : french = 41)
  (h3 : german = 22)
  (h4 : both = 9) :
  total - (french + german - both) = 6 := by
  sorry

end NUMINAMATH_CALUDE_students_not_enrolled_l973_97346


namespace NUMINAMATH_CALUDE_complement_intersection_cardinality_l973_97357

def U : Finset ℕ := {3,4,5,7,8,9}
def A : Finset ℕ := {4,5,7,8}
def B : Finset ℕ := {3,4,7,8}

theorem complement_intersection_cardinality :
  Finset.card (U \ (A ∩ B)) = 3 := by sorry

end NUMINAMATH_CALUDE_complement_intersection_cardinality_l973_97357


namespace NUMINAMATH_CALUDE_tangent_line_parallel_implies_a_zero_l973_97395

/-- Given a function f(x) = x^2 + a/x where a is a real number,
    if the tangent line at x = 1 is parallel to 2x - y + 1 = 0,
    then a = 0. -/
theorem tangent_line_parallel_implies_a_zero 
  (f : ℝ → ℝ) 
  (a : ℝ) 
  (h1 : ∀ x, f x = x^2 + a/x) 
  (h2 : (deriv f 1) = 2) : 
  a = 0 := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_parallel_implies_a_zero_l973_97395


namespace NUMINAMATH_CALUDE_sum_x₁_x₂_equals_three_l973_97347

/-- A discrete random variable with two possible values -/
structure DiscreteRV where
  x₁ : ℝ
  x₂ : ℝ
  p₁ : ℝ
  p₂ : ℝ
  h_prob_sum : p₁ + p₂ = 1
  h_prob_pos : 0 < p₁ ∧ 0 < p₂

/-- The expected value of a discrete random variable -/
def expected_value (X : DiscreteRV) : ℝ := X.x₁ * X.p₁ + X.x₂ * X.p₂

/-- The variance of a discrete random variable -/
def variance (X : DiscreteRV) : ℝ :=
  X.p₁ * (X.x₁ - expected_value X)^2 + X.p₂ * (X.x₂ - expected_value X)^2

/-- Theorem stating the sum of x₁ and x₂ for the given conditions -/
theorem sum_x₁_x₂_equals_three (X : DiscreteRV)
  (h_p₁ : X.p₁ = 2/3)
  (h_p₂ : X.p₂ = 1/3)
  (h_order : X.x₁ < X.x₂)
  (h_exp : expected_value X = 4/3)
  (h_var : variance X = 2/9) :
  X.x₁ + X.x₂ = 3 := by
  sorry

end NUMINAMATH_CALUDE_sum_x₁_x₂_equals_three_l973_97347


namespace NUMINAMATH_CALUDE_line_segment_parameterization_l973_97378

/-- Given a line segment connecting points (1,-3) and (6,12) parameterized by
    x = at + b and y = ct + d where 0 ≤ t ≤ 1 and t = 0 corresponds to (1,-3),
    prove that a + c^2 + b^2 + d^2 = 240 -/
theorem line_segment_parameterization (a b c d : ℝ) : 
  (∀ t : ℝ, 0 ≤ t ∧ t ≤ 1 → ∃ x y : ℝ, x = a * t + b ∧ y = c * t + d) →
  (b = 1 ∧ d = -3) →
  (a + b = 6 ∧ c + d = 12) →
  a + c^2 + b^2 + d^2 = 240 := by
  sorry

end NUMINAMATH_CALUDE_line_segment_parameterization_l973_97378


namespace NUMINAMATH_CALUDE_greatest_integer_less_than_negative_25_over_6_l973_97328

theorem greatest_integer_less_than_negative_25_over_6 :
  Int.floor (-25 / 6 : ℚ) = -5 := by
  sorry

end NUMINAMATH_CALUDE_greatest_integer_less_than_negative_25_over_6_l973_97328


namespace NUMINAMATH_CALUDE_compute_expression_simplify_expression_l973_97373

-- Part 1
theorem compute_expression : (1/2)⁻¹ - Real.sqrt 3 * Real.cos (30 * π / 180) + (2014 - Real.pi)^0 = 3/2 := by
  sorry

-- Part 2
theorem simplify_expression (a : ℝ) : a * (a + 1) - (a + 1) * (a - 1) = a + 1 := by
  sorry

end NUMINAMATH_CALUDE_compute_expression_simplify_expression_l973_97373


namespace NUMINAMATH_CALUDE_research_development_percentage_l973_97319

/-- Represents the budget allocation of a company -/
structure BudgetAllocation where
  transportation : ℝ
  utilities : ℝ
  equipment : ℝ
  supplies : ℝ
  salaries : ℝ
  research_development : ℝ

/-- The theorem stating that the research and development budget is 9% -/
theorem research_development_percentage
  (budget : BudgetAllocation)
  (h1 : budget.transportation = 15)
  (h2 : budget.utilities = 5)
  (h3 : budget.equipment = 4)
  (h4 : budget.supplies = 2)
  (h5 : budget.salaries = 234 / 360 * 100)
  (h6 : budget.transportation + budget.utilities + budget.equipment + budget.supplies + budget.salaries + budget.research_development = 100) :
  budget.research_development = 9 := by
sorry


end NUMINAMATH_CALUDE_research_development_percentage_l973_97319


namespace NUMINAMATH_CALUDE_percentage_seats_sold_l973_97375

def stadium_capacity : ℕ := 60000
def fans_stayed_home : ℕ := 5000
def fans_attended : ℕ := 40000

theorem percentage_seats_sold :
  (fans_attended + fans_stayed_home) / stadium_capacity * 100 = 75 := by
  sorry

end NUMINAMATH_CALUDE_percentage_seats_sold_l973_97375


namespace NUMINAMATH_CALUDE_pawn_placement_count_l973_97355

/-- The number of ways to place distinct pawns on a square chess board -/
def placePawns (n : ℕ) : ℕ :=
  (n.factorial) ^ 2

/-- The size of the chess board -/
def boardSize : ℕ := 5

/-- The number of pawns to be placed -/
def numPawns : ℕ := 5

theorem pawn_placement_count :
  placePawns boardSize = 14400 :=
sorry

end NUMINAMATH_CALUDE_pawn_placement_count_l973_97355


namespace NUMINAMATH_CALUDE_largest_non_sum_of_composites_l973_97369

def isComposite (n : ℕ) : Prop :=
  ∃ k : ℕ, 1 < k ∧ k < n ∧ n % k = 0

def isSumOfTwoComposites (n : ℕ) : Prop :=
  ∃ a b : ℕ, isComposite a ∧ isComposite b ∧ a + b = n

theorem largest_non_sum_of_composites :
  (∀ n : ℕ, n > 11 → isSumOfTwoComposites n) ∧
  ¬isSumOfTwoComposites 11 :=
sorry

end NUMINAMATH_CALUDE_largest_non_sum_of_composites_l973_97369


namespace NUMINAMATH_CALUDE_min_operations_for_two_pints_l973_97365

/-- Represents the state of the two vessels -/
structure VesselState :=
  (v7 : ℕ)
  (v11 : ℕ)

/-- Represents an operation on the vessels -/
inductive Operation
  | Fill7
  | Fill11
  | Empty7
  | Empty11
  | Pour7To11
  | Pour11To7

/-- Applies an operation to a vessel state -/
def applyOperation (state : VesselState) (op : Operation) : VesselState :=
  match op with
  | Operation.Fill7 => ⟨7, state.v11⟩
  | Operation.Fill11 => ⟨state.v7, 11⟩
  | Operation.Empty7 => ⟨0, state.v11⟩
  | Operation.Empty11 => ⟨state.v7, 0⟩
  | Operation.Pour7To11 => 
      let amount := min state.v7 (11 - state.v11)
      ⟨state.v7 - amount, state.v11 + amount⟩
  | Operation.Pour11To7 => 
      let amount := min state.v11 (7 - state.v7)
      ⟨state.v7 + amount, state.v11 - amount⟩

/-- Checks if a sequence of operations results in 2 pints in either vessel -/
def isValidSolution (ops : List Operation) : Prop :=
  let finalState := ops.foldl applyOperation ⟨0, 0⟩
  finalState.v7 = 2 ∨ finalState.v11 = 2

/-- The main theorem stating that 14 is the minimum number of operations -/
theorem min_operations_for_two_pints :
  (∃ (ops : List Operation), ops.length = 14 ∧ isValidSolution ops) ∧
  (∀ (ops : List Operation), ops.length < 14 → ¬isValidSolution ops) :=
sorry

end NUMINAMATH_CALUDE_min_operations_for_two_pints_l973_97365


namespace NUMINAMATH_CALUDE_middle_number_proof_l973_97382

theorem middle_number_proof (numbers : List ℝ) 
  (h_count : numbers.length = 11)
  (h_avg_all : numbers.sum / numbers.length = 9.9)
  (h_avg_first6 : (numbers.take 6).sum / 6 = 10.5)
  (h_avg_last6 : (numbers.drop 5).sum / 6 = 11.4) :
  numbers[5] = 22.5 := by
  sorry

end NUMINAMATH_CALUDE_middle_number_proof_l973_97382


namespace NUMINAMATH_CALUDE_sin_geq_cos_range_l973_97337

theorem sin_geq_cos_range (x : ℝ) : 
  x ∈ Set.Ioo (0 : ℝ) (2 * Real.pi) →
  (Real.sin x ≥ Real.cos x ↔ x ∈ Set.Icc (Real.pi / 4) (5 * Real.pi / 4)) := by
sorry

end NUMINAMATH_CALUDE_sin_geq_cos_range_l973_97337


namespace NUMINAMATH_CALUDE_park_visitors_l973_97336

theorem park_visitors (visitors_day1 visitors_day2 : ℕ) : 
  visitors_day2 = visitors_day1 + 40 →
  visitors_day1 + visitors_day2 = 440 →
  visitors_day1 = 200 := by
sorry

end NUMINAMATH_CALUDE_park_visitors_l973_97336


namespace NUMINAMATH_CALUDE_expand_polynomial_l973_97342

theorem expand_polynomial (x : ℝ) : 
  (x + 3) * (4 * x^2 - 2 * x - 5) = 4 * x^3 + 10 * x^2 - 11 * x - 15 := by
  sorry

end NUMINAMATH_CALUDE_expand_polynomial_l973_97342


namespace NUMINAMATH_CALUDE_water_ratio_corn_to_pig_l973_97368

def water_pumping_rate : ℚ := 3
def pumping_time : ℕ := 25
def corn_rows : ℕ := 4
def corn_plants_per_row : ℕ := 15
def num_pigs : ℕ := 10
def water_per_pig : ℚ := 4
def num_ducks : ℕ := 20
def water_per_duck : ℚ := 1/4

theorem water_ratio_corn_to_pig :
  let total_water := water_pumping_rate * pumping_time
  let total_corn_plants := corn_rows * corn_plants_per_row
  let water_for_pigs := num_pigs * water_per_pig
  let water_for_ducks := num_ducks * water_per_duck
  let water_for_corn := total_water - water_for_pigs - water_for_ducks
  let water_per_corn := water_for_corn / total_corn_plants
  water_per_corn / water_per_pig = 1/8 := by sorry

end NUMINAMATH_CALUDE_water_ratio_corn_to_pig_l973_97368


namespace NUMINAMATH_CALUDE_sum_x_y_equals_seven_a_l973_97363

theorem sum_x_y_equals_seven_a (a x y : ℝ) (h1 : a / x = 1 / 3) (h2 : a / y = 1 / 4) :
  x + y = 7 * a := by
  sorry

end NUMINAMATH_CALUDE_sum_x_y_equals_seven_a_l973_97363


namespace NUMINAMATH_CALUDE_cloth_cost_price_l973_97356

/-- Calculates the cost price per meter of cloth given the total selling price,
    number of meters sold, and profit per meter. -/
def cost_price_per_meter (total_selling_price : ℕ) (meters_sold : ℕ) (profit_per_meter : ℕ) : ℕ :=
  (total_selling_price - profit_per_meter * meters_sold) / meters_sold

/-- Proves that the cost price of one meter of cloth is Rs. 100, given the conditions. -/
theorem cloth_cost_price :
  cost_price_per_meter 8925 85 5 = 100 := by
  sorry

end NUMINAMATH_CALUDE_cloth_cost_price_l973_97356


namespace NUMINAMATH_CALUDE_quadratic_roots_from_intersections_l973_97349

/-- Given a quadratic function f(x) = ax² + bx + c, if its graph intersects
    the x-axis at (1,0) and (4,0), then the solutions to ax² + bx + c = 0
    are x₁ = 1 and x₂ = 4. -/
theorem quadratic_roots_from_intersections
  (a b c : ℝ) (f : ℝ → ℝ) (h_f : ∀ x, f x = a * x^2 + b * x + c) :
  f 1 = 0 → f 4 = 0 →
  ∃ x₁ x₂ : ℝ, x₁ = 1 ∧ x₂ = 4 ∧ ∀ x, a * x^2 + b * x + c = 0 ↔ x = x₁ ∨ x = x₂ :=
sorry

end NUMINAMATH_CALUDE_quadratic_roots_from_intersections_l973_97349


namespace NUMINAMATH_CALUDE_chenny_spoons_count_l973_97358

/-- Proves that Chenny bought 4 spoons given the conditions of the problem -/
theorem chenny_spoons_count : 
  ∀ (num_plates : ℕ) (plate_cost spoon_cost total_cost : ℚ),
    num_plates = 9 →
    plate_cost = 2 →
    spoon_cost = 3/2 →
    total_cost = 24 →
    (total_cost - (↑num_plates * plate_cost)) / spoon_cost = 4 :=
by sorry

end NUMINAMATH_CALUDE_chenny_spoons_count_l973_97358


namespace NUMINAMATH_CALUDE_min_reciprocal_sum_l973_97334

theorem min_reciprocal_sum (a b : ℝ) (ha : 0 < a) (hb : 0 < b) (hab : a + b = 12) :
  (1 / a + 1 / b) ≥ 1 / 3 ∧ ∃ (x y : ℝ), 0 < x ∧ 0 < y ∧ x + y = 12 ∧ 1 / x + 1 / y = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_min_reciprocal_sum_l973_97334


namespace NUMINAMATH_CALUDE_original_number_of_people_l973_97388

theorem original_number_of_people (x : ℕ) : 
  (x / 3 : ℚ) = 18 → x = 54 := by sorry

end NUMINAMATH_CALUDE_original_number_of_people_l973_97388


namespace NUMINAMATH_CALUDE_satisfactory_grade_fraction_l973_97304

/-- Represents the grades in a science class -/
inductive Grade
  | A
  | B
  | C
  | D
  | F

/-- Returns true if the grade is satisfactory (A, B, or C) -/
def isSatisfactory (g : Grade) : Bool :=
  match g with
  | Grade.A => true
  | Grade.B => true
  | Grade.C => true
  | _ => false

/-- Represents the distribution of grades in the class -/
def gradeDistribution : List (Grade × Nat) :=
  [(Grade.A, 8), (Grade.B, 6), (Grade.C, 4), (Grade.D, 2), (Grade.F, 6)]

/-- Theorem: The fraction of satisfactory grades is 9/13 -/
theorem satisfactory_grade_fraction :
  let totalGrades := (gradeDistribution.map (·.2)).sum
  let satisfactoryGrades := (gradeDistribution.filter (isSatisfactory ·.1)).map (·.2) |>.sum
  (satisfactoryGrades : ℚ) / totalGrades = 9 / 13 := by
  sorry


end NUMINAMATH_CALUDE_satisfactory_grade_fraction_l973_97304


namespace NUMINAMATH_CALUDE_douglas_vote_percentage_l973_97306

theorem douglas_vote_percentage (total_percentage : ℝ) (county_y_percentage : ℝ) :
  total_percentage = 64 →
  county_y_percentage = 40.00000000000002 →
  let county_x_votes : ℝ := 2
  let county_y_votes : ℝ := 1
  let total_votes : ℝ := county_x_votes + county_y_votes
  let county_x_percentage : ℝ := 
    (total_percentage * total_votes - county_y_percentage * county_y_votes) / county_x_votes
  county_x_percentage = 76 := by
sorry

end NUMINAMATH_CALUDE_douglas_vote_percentage_l973_97306


namespace NUMINAMATH_CALUDE_total_letters_received_l973_97331

theorem total_letters_received (brother_letters : ℕ) 
  (h1 : brother_letters = 40) 
  (h2 : ∃ greta_letters : ℕ, greta_letters = brother_letters + 10) 
  (h3 : ∃ mother_letters : ℕ, mother_letters = 2 * (brother_letters + (brother_letters + 10))) :
  ∃ total_letters : ℕ, total_letters = brother_letters + (brother_letters + 10) + 2 * (brother_letters + (brother_letters + 10)) ∧ total_letters = 270 := by
sorry


end NUMINAMATH_CALUDE_total_letters_received_l973_97331


namespace NUMINAMATH_CALUDE_pure_imaginary_product_l973_97338

theorem pure_imaginary_product (z : ℂ) (a : ℝ) : 
  (∃ b : ℝ, z = b * I) → (3 - I) * z = a + I → a = 1/3 := by sorry

end NUMINAMATH_CALUDE_pure_imaginary_product_l973_97338


namespace NUMINAMATH_CALUDE_base_eight_subtraction_l973_97376

/-- Represents a number in base 8 --/
def BaseEight : Type := Nat

/-- Converts a base 8 number to its decimal representation --/
def to_decimal (n : BaseEight) : Nat := sorry

/-- Converts a decimal number to its base 8 representation --/
def from_decimal (n : Nat) : BaseEight := sorry

/-- Subtracts two base 8 numbers --/
def base_eight_sub (a b : BaseEight) : BaseEight := sorry

theorem base_eight_subtraction :
  base_eight_sub (from_decimal 4765) (from_decimal 2314) = from_decimal 2447 := by sorry

end NUMINAMATH_CALUDE_base_eight_subtraction_l973_97376


namespace NUMINAMATH_CALUDE_binomial_coefficient_22_5_l973_97386

theorem binomial_coefficient_22_5 (h1 : Nat.choose 20 3 = 1140)
                                  (h2 : Nat.choose 20 4 = 4845)
                                  (h3 : Nat.choose 20 5 = 15504) :
  Nat.choose 22 5 = 26334 := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_22_5_l973_97386


namespace NUMINAMATH_CALUDE_inequality_solution_range_l973_97381

theorem inequality_solution_range (a : ℝ) : 
  (∃! (x y : ℤ), x ≠ y ∧ 
    (∀ (z : ℤ), z^2 - (a+1)*z + a < 0 ↔ (z = x ∨ z = y))) ↔ 
  (a ∈ Set.Icc (-2) (-1) ∪ Set.Ioc 3 4) :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_range_l973_97381


namespace NUMINAMATH_CALUDE_a_greater_than_b_l973_97316

theorem a_greater_than_b : ∀ x : ℝ, (x - 3)^2 > (x - 2) * (x - 4) := by
  sorry

end NUMINAMATH_CALUDE_a_greater_than_b_l973_97316


namespace NUMINAMATH_CALUDE_king_game_winner_l973_97312

/-- Represents the result of the game -/
inductive GameResult
  | PlayerAWins
  | PlayerBWins

/-- Represents a chessboard of size m × n -/
structure Chessboard where
  m : Nat
  n : Nat

/-- Determines the winner of the game based on the chessboard size -/
def determineWinner (board : Chessboard) : GameResult :=
  if board.m * board.n % 2 == 0 then
    GameResult.PlayerAWins
  else
    GameResult.PlayerBWins

/-- Theorem stating the winning condition for the game -/
theorem king_game_winner (board : Chessboard) :
  determineWinner board = GameResult.PlayerAWins ↔ board.m * board.n % 2 == 0 := by
  sorry

end NUMINAMATH_CALUDE_king_game_winner_l973_97312


namespace NUMINAMATH_CALUDE_expression_equality_l973_97380

theorem expression_equality (x : ℝ) : 
  x * (x * (x * (3 - x) - 5) + 12) + 2 = -x^4 + 3*x^3 - 5*x^2 + 12*x + 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l973_97380


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l973_97367

/-- Given a = 2 and b = -1/2, prove that a - 2(a - b^2) + 3(-a + b^2) = -27/4 -/
theorem simplify_and_evaluate (a b : ℚ) (ha : a = 2) (hb : b = -1/2) :
  a - 2 * (a - b^2) + 3 * (-a + b^2) = -27/4 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l973_97367


namespace NUMINAMATH_CALUDE_pool_capacity_correct_l973_97351

/-- The amount of water Grace's pool can contain -/
def pool_capacity : ℕ := 390

/-- The rate at which the first hose sprays water -/
def first_hose_rate : ℕ := 50

/-- The rate at which the second hose sprays water -/
def second_hose_rate : ℕ := 70

/-- The time the first hose runs alone -/
def first_hose_time : ℕ := 3

/-- The time both hoses run together -/
def both_hoses_time : ℕ := 2

/-- Theorem stating that the pool capacity is correct given the conditions -/
theorem pool_capacity_correct :
  pool_capacity = first_hose_rate * first_hose_time + 
    (first_hose_rate + second_hose_rate) * both_hoses_time :=
by sorry

end NUMINAMATH_CALUDE_pool_capacity_correct_l973_97351


namespace NUMINAMATH_CALUDE_pauls_crayons_given_to_friends_l973_97333

/-- Given information about Paul's crayons --/
structure CrayonInfo where
  initial : ℕ  -- Initial number of crayons
  lost_difference : ℕ  -- Difference between lost and given crayons
  total_gone : ℕ  -- Total number of crayons no longer in possession

/-- Calculate the number of crayons given to friends --/
def crayons_given_to_friends (info : CrayonInfo) : ℕ :=
  (info.total_gone - info.lost_difference) / 2

/-- Theorem stating the number of crayons Paul gave to his friends --/
theorem pauls_crayons_given_to_friends :
  let info : CrayonInfo := {
    initial := 110,
    lost_difference := 322,
    total_gone := 412
  }
  crayons_given_to_friends info = 45 := by
  sorry

end NUMINAMATH_CALUDE_pauls_crayons_given_to_friends_l973_97333


namespace NUMINAMATH_CALUDE_base_2_representation_of_236_l973_97344

theorem base_2_representation_of_236 :
  ∃ (a : List Bool),
    a.length = 9 ∧
    a = [true, true, true, false, true, false, true, false, false] ∧
    (a.foldr (λ (b : Bool) (acc : Nat) => 2 * acc + if b then 1 else 0) 0) = 236 :=
by sorry

end NUMINAMATH_CALUDE_base_2_representation_of_236_l973_97344


namespace NUMINAMATH_CALUDE_alternate_interior_angles_parallel_l973_97371

-- Define a structure for lines in a plane
structure Line :=
  (slope : ℝ)
  (intercept : ℝ)

-- Define a structure for angles
structure Angle :=
  (measure : ℝ)

-- Define a function to check if two lines are parallel
def are_parallel (l1 l2 : Line) : Prop :=
  l1.slope = l2.slope

-- Define a function to represent alternate interior angles
def alternate_interior_angles (l1 l2 : Line) (t : Line) : (Angle × Angle) :=
  sorry

-- Theorem statement
theorem alternate_interior_angles_parallel (l1 l2 t : Line) :
  let (angle1, angle2) := alternate_interior_angles l1 l2 t
  (angle1.measure = angle2.measure) → are_parallel l1 l2 :=
sorry

end NUMINAMATH_CALUDE_alternate_interior_angles_parallel_l973_97371


namespace NUMINAMATH_CALUDE_M₁_on_curve_M₂_not_on_curve_M₃_a_value_l973_97348

-- Define the curve C
def curve_C (t : ℝ) : ℝ × ℝ := (3 * t, 2 * t^2 + 1)

-- Define the points
def M₁ : ℝ × ℝ := (0, 1)
def M₂ : ℝ × ℝ := (5, 4)
def M₃ (a : ℝ) : ℝ × ℝ := (6, a)

-- Theorem statements
theorem M₁_on_curve : ∃ t : ℝ, curve_C t = M₁ := by sorry

theorem M₂_not_on_curve : ¬ ∃ t : ℝ, curve_C t = M₂ := by sorry

theorem M₃_a_value : ∃ a : ℝ, (∃ t : ℝ, curve_C t = M₃ a) → a = 9 := by sorry

end NUMINAMATH_CALUDE_M₁_on_curve_M₂_not_on_curve_M₃_a_value_l973_97348


namespace NUMINAMATH_CALUDE_certain_value_multiplication_l973_97308

theorem certain_value_multiplication (x : ℝ) : x * (1/7)^2 = 7^3 → x = 16807 := by
  sorry

end NUMINAMATH_CALUDE_certain_value_multiplication_l973_97308


namespace NUMINAMATH_CALUDE_chairs_to_remove_chair_adjustment_problem_l973_97384

theorem chairs_to_remove (initial_chairs : ℕ) (chairs_per_row : ℕ) (expected_students : ℕ) : ℕ :=
  let min_chairs_needed := ((expected_students + chairs_per_row - 1) / chairs_per_row) * chairs_per_row
  initial_chairs - min_chairs_needed

theorem chair_adjustment_problem :
  chairs_to_remove 169 13 100 = 65 := by
  sorry

end NUMINAMATH_CALUDE_chairs_to_remove_chair_adjustment_problem_l973_97384


namespace NUMINAMATH_CALUDE_ad_eq_bc_necessary_not_sufficient_l973_97377

/-- A sequence of four non-zero real numbers forms a geometric sequence -/
def IsGeometricSequence (a b c d : ℝ) : Prop :=
  a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0 ∧ ∃ r : ℝ, r ≠ 0 ∧ b = a * r ∧ c = b * r ∧ d = c * r

/-- The condition ad=bc for four non-zero real numbers -/
def AdEqualsBc (a b c d : ℝ) : Prop :=
  a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ d ≠ 0 ∧ a * d = b * c

theorem ad_eq_bc_necessary_not_sufficient :
  (∀ a b c d : ℝ, IsGeometricSequence a b c d → AdEqualsBc a b c d) ∧
  (∃ a b c d : ℝ, AdEqualsBc a b c d ∧ ¬IsGeometricSequence a b c d) :=
sorry

end NUMINAMATH_CALUDE_ad_eq_bc_necessary_not_sufficient_l973_97377


namespace NUMINAMATH_CALUDE_ellipse_problem_l973_97345

-- Define the ellipse C
def ellipse (x y a b : ℝ) : Prop := x^2 / a^2 + y^2 / b^2 = 1

-- Define the line l
def line (x y k : ℝ) : Prop := y = k * (x - 1)

-- Define the theorem
theorem ellipse_problem (a b c k : ℝ) (h1 : a > b) (h2 : b > 0) (h3 : k > 0) :
  -- Condition 2: C passes through Q(√2, 1)
  ellipse (Real.sqrt 2) 1 a b →
  -- Condition 3: Right focus at F(√2, 0)
  c = Real.sqrt 2 →
  a^2 - b^2 = c^2 →
  -- Condition 6: CN = MD (implicitly used in the solution)
  (∃ x₁ y₁ x₂ y₂ : ℝ, 
    ellipse x₁ y₁ a b ∧ 
    ellipse x₂ y₂ a b ∧
    line x₁ y₁ k ∧ 
    line x₂ y₂ k ∧
    x₂ - 1 = -x₁ ∧ 
    y₂ = -k - y₁) →
  -- Conclusion I: Equation of ellipse C
  (a = 2 ∧ b = Real.sqrt 2) ∧
  -- Conclusion II: Value of k and length of MN
  (k = Real.sqrt 2 / 2 ∧ 
   ∃ x₁ x₂ : ℝ, 
     ellipse x₁ (k * (x₁ - 1)) 2 (Real.sqrt 2) ∧
     ellipse x₂ (k * (x₂ - 1)) 2 (Real.sqrt 2) ∧
     Real.sqrt ((x₂ - x₁)^2 + (k * (x₂ - x₁))^2) = Real.sqrt 42 / 2) :=
by sorry

end NUMINAMATH_CALUDE_ellipse_problem_l973_97345


namespace NUMINAMATH_CALUDE_chessboard_numbering_exists_l973_97383

theorem chessboard_numbering_exists : 
  ∃ f : ℕ → ℕ → ℕ, 
    (∀ i j, i ∈ Finset.range 8 ∧ j ∈ Finset.range 8 → f i j ∈ Finset.range 64) ∧ 
    (∀ i j, i ∈ Finset.range 7 ∧ j ∈ Finset.range 7 → 
      (f i j + f (i+1) j + f i (j+1) + f (i+1) (j+1)) % 4 = 0) ∧
    (∀ n, n ∈ Finset.range 64 → ∃ i j, i ∈ Finset.range 8 ∧ j ∈ Finset.range 8 ∧ f i j = n + 1) :=
by
  sorry

end NUMINAMATH_CALUDE_chessboard_numbering_exists_l973_97383


namespace NUMINAMATH_CALUDE_distance_is_90km_l973_97307

/-- Calculates the distance traveled downstream by a boat -/
def distance_downstream (boat_speed : ℝ) (stream_speed : ℝ) (time : ℝ) : ℝ :=
  (boat_speed + stream_speed) * time

/-- Theorem: The distance traveled downstream is 90 km -/
theorem distance_is_90km (boat_speed : ℝ) (stream_speed : ℝ) (time : ℝ)
  (h1 : boat_speed = 25)
  (h2 : stream_speed = 5)
  (h3 : time = 3) :
  distance_downstream boat_speed stream_speed time = 90 := by
  sorry

#eval distance_downstream 25 5 3

end NUMINAMATH_CALUDE_distance_is_90km_l973_97307


namespace NUMINAMATH_CALUDE_canteen_distance_l973_97305

/-- Given a right triangle with legs 450 and 600 rods, prove that a point on the hypotenuse 
    that is equidistant from both ends of the hypotenuse is 468.75 rods from each end. -/
theorem canteen_distance (a b c x : ℝ) (h1 : a = 450) (h2 : b = 600) 
  (h3 : c^2 = a^2 + b^2) (h4 : x^2 = a^2 + (b - x)^2) : x = 468.75 := by
  sorry

end NUMINAMATH_CALUDE_canteen_distance_l973_97305


namespace NUMINAMATH_CALUDE_ratio_problem_l973_97370

/-- Given ratios A : B : C where A = 4x, B = 6x, C = 9x, and A = 50, prove the values of B and C and their average -/
theorem ratio_problem (x : ℚ) (A B C : ℚ) (h1 : A = 4 * x) (h2 : B = 6 * x) (h3 : C = 9 * x) (h4 : A = 50) :
  B = 75 ∧ C = 112.5 ∧ (B + C) / 2 = 93.75 := by
  sorry


end NUMINAMATH_CALUDE_ratio_problem_l973_97370


namespace NUMINAMATH_CALUDE_distance_between_harper_and_jack_l973_97361

/-- Represents the distance between two runners at the end of a race. -/
def distance_between (race_length : ℕ) (jack_distance : ℕ) : ℕ :=
  race_length - jack_distance

/-- Proves that the distance between Harper and Jack at the end of the race is 848 meters. -/
theorem distance_between_harper_and_jack :
  let race_length_km : ℕ := 1
  let race_length_m : ℕ := race_length_km * 1000
  let jack_distance : ℕ := 152
  distance_between race_length_m jack_distance = 848 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_harper_and_jack_l973_97361


namespace NUMINAMATH_CALUDE_fractional_equation_positive_root_l973_97392

theorem fractional_equation_positive_root (m : ℝ) : 
  (∃ x : ℝ, x > 0 ∧ (x + 5) / (x - 3) = 2 - m / (3 - x)) → m = 8 := by
  sorry

end NUMINAMATH_CALUDE_fractional_equation_positive_root_l973_97392


namespace NUMINAMATH_CALUDE_distance_in_15_minutes_l973_97343

/-- Given a constant speed calculated from driving 80 miles in 2 hours, 
    prove that the distance traveled in 15 minutes is 10 miles. -/
theorem distance_in_15_minutes (total_distance : ℝ) (total_time : ℝ) 
  (travel_time : ℝ) (h1 : total_distance = 80) (h2 : total_time = 2) 
  (h3 : travel_time = 15 / 60) : 
  (total_distance / total_time) * travel_time = 10 := by
  sorry

end NUMINAMATH_CALUDE_distance_in_15_minutes_l973_97343


namespace NUMINAMATH_CALUDE_circle_properties_l973_97393

-- Define the circle C
def C (x y : ℝ) : Prop := x^2 + y^2 - 4*x + 2 = 0

-- Define the line L
def L (x y : ℝ) : Prop := x - y - 1 = 0

-- Define the symmetric line
def SymLine (x y : ℝ) : Prop := x - y = 0

-- Define the distance line
def DistLine (x y m : ℝ) : Prop := x + y + m = 0

theorem circle_properties :
  -- 1. Chord length
  (∃ l : ℝ, l = Real.sqrt 6 ∧
    ∀ x y : ℝ, C x y → L x y →
      ∃ x' y' : ℝ, C x' y' ∧ L x' y' ∧
        (x - x')^2 + (y - y')^2 = l^2) ∧
  -- 2. Symmetric circle
  (∀ x y : ℝ, (∃ x' y' : ℝ, C x' y' ∧ SymLine x' y' ∧
    x = y' ∧ y = x') → x^2 + (y-2)^2 = 2) ∧
  -- 3. Distance condition
  (∀ m : ℝ, (abs (m + 2) / Real.sqrt 2 = Real.sqrt 2 / 2) →
    m = -1 ∨ m = -3) :=
sorry

end NUMINAMATH_CALUDE_circle_properties_l973_97393


namespace NUMINAMATH_CALUDE_equation_solution_existence_l973_97309

theorem equation_solution_existence (z : ℝ) :
  (∀ y : ℝ, ∃ x : ℝ, x^2 + y^2 + 4*z^2 + 2*x*y*z - 9 = 0) ↔ 1 ≤ |z| ∧ |z| ≤ 3/2 :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_existence_l973_97309


namespace NUMINAMATH_CALUDE_total_accidents_l973_97391

/-- Represents the accident rate for a highway -/
structure AccidentRate where
  accidents : ℕ
  vehicles : ℕ

/-- Calculates the number of accidents for a given traffic volume -/
def calculateAccidents (rate : AccidentRate) (traffic : ℕ) : ℕ :=
  (rate.accidents * traffic + rate.vehicles - 1) / rate.vehicles

theorem total_accidents (highwayA_rate : AccidentRate) (highwayB_rate : AccidentRate) (highwayC_rate : AccidentRate)
  (highwayA_traffic : ℕ) (highwayB_traffic : ℕ) (highwayC_traffic : ℕ) :
  highwayA_rate = ⟨200, 100000000⟩ →
  highwayB_rate = ⟨150, 50000000⟩ →
  highwayC_rate = ⟨100, 150000000⟩ →
  highwayA_traffic = 2000000000 →
  highwayB_traffic = 1500000000 →
  highwayC_traffic = 2500000000 →
  calculateAccidents highwayA_rate highwayA_traffic +
  calculateAccidents highwayB_rate highwayB_traffic +
  calculateAccidents highwayC_rate highwayC_traffic = 10168 := by
  sorry

end NUMINAMATH_CALUDE_total_accidents_l973_97391


namespace NUMINAMATH_CALUDE_scientific_notation_of_219400_l973_97320

theorem scientific_notation_of_219400 :
  ∃ (a : ℝ) (n : ℤ), 
    219400 = a * (10 : ℝ) ^ n ∧ 
    1 ≤ |a| ∧ 
    |a| < 10 ∧
    a = 2.194 ∧
    n = 5 := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_of_219400_l973_97320


namespace NUMINAMATH_CALUDE_intersection_implies_m_value_subset_complement_iff_m_range_l973_97314

-- Define sets A and B
def A : Set ℝ := {x : ℝ | -2 ≤ x ∧ x ≤ 4}
def B (m : ℝ) : Set ℝ := {x : ℝ | m - 3 ≤ x ∧ x ≤ m}

-- Theorem 1
theorem intersection_implies_m_value :
  ∀ m : ℝ, (A ∩ B m = {x : ℝ | 2 ≤ x ∧ x ≤ 4}) → m = 5 := by
  sorry

-- Theorem 2
theorem subset_complement_iff_m_range :
  ∀ m : ℝ, A ⊆ (Set.univ \ B m) ↔ m < -2 ∨ m > 7 := by
  sorry

end NUMINAMATH_CALUDE_intersection_implies_m_value_subset_complement_iff_m_range_l973_97314


namespace NUMINAMATH_CALUDE_zero_exponent_l973_97329

theorem zero_exponent (x : ℝ) (hx : x ≠ 0) : x^0 = 1 := by
  sorry

end NUMINAMATH_CALUDE_zero_exponent_l973_97329


namespace NUMINAMATH_CALUDE_lily_lottery_tickets_l973_97315

/-- Represents the number of lottery tickets sold -/
def n : ℕ := 5

/-- The price of the i-th ticket -/
def ticket_price (i : ℕ) : ℕ := i

/-- The total amount collected from selling n tickets -/
def total_collected (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The profit Lily keeps -/
def profit : ℕ := 4

/-- The prize money for the lottery winner -/
def prize : ℕ := 11

theorem lily_lottery_tickets :
  (total_collected n = prize + profit) ∧
  (∀ m : ℕ, m ≠ n → total_collected m ≠ prize + profit) :=
by sorry

end NUMINAMATH_CALUDE_lily_lottery_tickets_l973_97315


namespace NUMINAMATH_CALUDE_exists_efficient_coin_ordering_strategy_l973_97390

/-- A strategy for ordering coins by weight using a balance scale. -/
structure CoinOrderingStrategy where
  /-- The number of coins to be ordered -/
  num_coins : Nat
  /-- The expected number of weighings required by the strategy -/
  expected_weighings : ℝ

/-- A weighing action compares two coins and determines which is heavier -/
def weighing_action (coin1 coin2 : Nat) : Bool := sorry

/-- Theorem stating that there exists a strategy for ordering 4 coins with expected weighings < 4.8 -/
theorem exists_efficient_coin_ordering_strategy :
  ∃ (strategy : CoinOrderingStrategy),
    strategy.num_coins = 4 ∧
    strategy.expected_weighings < 4.8 := by sorry

end NUMINAMATH_CALUDE_exists_efficient_coin_ordering_strategy_l973_97390


namespace NUMINAMATH_CALUDE_quaternary_30012_to_decimal_l973_97354

/-- Converts a list of digits in base 4 to its decimal representation -/
def quaternary_to_decimal (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (4 ^ i)) 0

/-- The quaternary number 30012 -/
def quaternary_30012 : List Nat := [2, 1, 0, 0, 3]

theorem quaternary_30012_to_decimal :
  quaternary_to_decimal quaternary_30012 = 774 := by
  sorry

end NUMINAMATH_CALUDE_quaternary_30012_to_decimal_l973_97354


namespace NUMINAMATH_CALUDE_margos_walking_distance_l973_97364

/-- Margo's Walking Problem -/
theorem margos_walking_distance
  (outbound_time : Real) (return_time : Real)
  (outbound_speed : Real) (return_speed : Real)
  (average_speed : Real)
  (h1 : outbound_time = 15 / 60)
  (h2 : return_time = 30 / 60)
  (h3 : outbound_speed = 5)
  (h4 : return_speed = 3)
  (h5 : average_speed = 3.6)
  (h6 : average_speed = (outbound_time + return_time) / 
        ((outbound_time / outbound_speed) + (return_time / return_speed))) :
  outbound_speed * outbound_time + return_speed * return_time = 2.75 := by
  sorry

#check margos_walking_distance

end NUMINAMATH_CALUDE_margos_walking_distance_l973_97364


namespace NUMINAMATH_CALUDE_probability_sum_nine_l973_97330

/-- A standard die with six faces -/
def Die : Type := Fin 6

/-- The sample space of rolling a die twice -/
def SampleSpace : Type := Die × Die

/-- The event of getting a sum of 9 -/
def SumNine (outcome : SampleSpace) : Prop :=
  (outcome.1.val + 1) + (outcome.2.val + 1) = 9

/-- The number of favorable outcomes (sum of 9) -/
def FavorableOutcomes : ℕ := 4

/-- The total number of possible outcomes -/
def TotalOutcomes : ℕ := 36

/-- The probability of getting a sum of 9 -/
def ProbabilitySumNine : ℚ := FavorableOutcomes / TotalOutcomes

theorem probability_sum_nine :
  ProbabilitySumNine = 1 / 9 := by
  sorry

end NUMINAMATH_CALUDE_probability_sum_nine_l973_97330


namespace NUMINAMATH_CALUDE_inequality_problem_l973_97352

theorem inequality_problem (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h_sum : 1/a + 1/b + 1/c = 1) :
  (∃ (max_val : ℝ), a = 2 → (∀ x y : ℝ, x > 0 ∧ y > 0 ∧ 1/2 + 1/x + 1/y = 1 →
    1/(x + y) ≤ max_val) ∧ max_val = 1/8) ∧
  1/(a + b) + 1/(b + c) + 1/(a + c) ≤ 1/2 :=
by sorry

end NUMINAMATH_CALUDE_inequality_problem_l973_97352
