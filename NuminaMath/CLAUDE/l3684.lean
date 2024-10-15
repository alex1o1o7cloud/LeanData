import Mathlib

namespace NUMINAMATH_CALUDE_pentagon_angle_problem_l3684_368475

def pentagon_largest_angle (P Q R S T : ℝ) : Prop :=
  -- Sum of angles in a pentagon is 540°
  P + Q + R + S + T = 540 ∧
  -- Given conditions
  P = 55 ∧
  Q = 120 ∧
  R = S ∧
  T = 2 * R + 20 ∧
  -- Largest angle is 192.5°
  max P (max Q (max R (max S T))) = 192.5

theorem pentagon_angle_problem :
  ∃ (P Q R S T : ℝ), pentagon_largest_angle P Q R S T := by
  sorry

end NUMINAMATH_CALUDE_pentagon_angle_problem_l3684_368475


namespace NUMINAMATH_CALUDE_solution_is_3_minus_i_l3684_368442

/-- Definition of the determinant operation -/
def det (a b c d : ℂ) : ℂ := a * d - b * c

/-- Theorem stating that the complex number z satisfying the given equation is 3 - i -/
theorem solution_is_3_minus_i :
  ∃ z : ℂ, det 1 (-1) z (z * Complex.I) = 4 + 2 * Complex.I ∧ z = 3 - Complex.I :=
sorry

end NUMINAMATH_CALUDE_solution_is_3_minus_i_l3684_368442


namespace NUMINAMATH_CALUDE_complex_subtraction_l3684_368466

theorem complex_subtraction (c d : ℂ) (hc : c = 5 - 3*I) (hd : d = 2 + 4*I) :
  c - 3*d = -1 - 15*I := by
  sorry

end NUMINAMATH_CALUDE_complex_subtraction_l3684_368466


namespace NUMINAMATH_CALUDE_interior_alternate_angles_parallel_l3684_368412

-- Define a structure for a line
structure Line :=
  (slope : ℝ)
  (intercept : ℝ)

-- Define a structure for an angle
structure Angle :=
  (measure : ℝ)

-- Define a function to check if two lines are parallel
def are_parallel (l1 l2 : Line) : Prop :=
  l1.slope = l2.slope

-- Define a function to represent interior alternate angles
def interior_alternate_angles (l1 l2 : Line) (t : Line) : (Angle × Angle) :=
  sorry

-- Theorem statement
theorem interior_alternate_angles_parallel (l1 l2 t : Line) :
  let (angle1, angle2) := interior_alternate_angles l1 l2 t
  angle1.measure = angle2.measure → are_parallel l1 l2 :=
sorry

end NUMINAMATH_CALUDE_interior_alternate_angles_parallel_l3684_368412


namespace NUMINAMATH_CALUDE_hyperbola_foci_l3684_368498

/-- The hyperbola equation -/
def hyperbola_equation (x y : ℝ) : Prop :=
  y^2 / 3 - x^2 = 1

/-- The coordinates of a focus -/
def is_focus (x y : ℝ) : Prop :=
  x = 0 ∧ (y = 2 ∨ y = -2)

/-- Theorem: The foci of the given hyperbola are at (0, ±2) -/
theorem hyperbola_foci :
  ∀ x y : ℝ, hyperbola_equation x y →
  (∃ a b : ℝ, is_focus a b ∧ 
    (x - a)^2 + (y - b)^2 = (x + a)^2 + (y + b)^2) :=
sorry

end NUMINAMATH_CALUDE_hyperbola_foci_l3684_368498


namespace NUMINAMATH_CALUDE_razorback_shop_revenue_l3684_368413

/-- Calculates the total revenue from selling t-shirts and hats at the Razorback t-shirt shop during a game. -/
theorem razorback_shop_revenue
  (t_shirt_price : ℕ)
  (hat_price : ℕ)
  (t_shirt_discount : ℕ)
  (hat_discount : ℕ)
  (t_shirts_sold : ℕ)
  (hats_sold : ℕ)
  (h1 : t_shirt_price = 51)
  (h2 : hat_price = 28)
  (h3 : t_shirt_discount = 8)
  (h4 : hat_discount = 5)
  (h5 : t_shirts_sold = 130)
  (h6 : hats_sold = 85) :
  (t_shirts_sold * (t_shirt_price - t_shirt_discount) + hats_sold * (hat_price - hat_discount)) = 7545 :=
by
  sorry

end NUMINAMATH_CALUDE_razorback_shop_revenue_l3684_368413


namespace NUMINAMATH_CALUDE_complete_graph_4_vertices_6_edges_l3684_368471

/-- The number of edges in a complete graph with n vertices -/
def complete_graph_edges (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem: A complete graph with 4 vertices has 6 edges -/
theorem complete_graph_4_vertices_6_edges : 
  complete_graph_edges 4 = 6 := by
  sorry

end NUMINAMATH_CALUDE_complete_graph_4_vertices_6_edges_l3684_368471


namespace NUMINAMATH_CALUDE_probability_in_standard_deck_l3684_368410

/-- Represents a standard deck of cards -/
structure Deck :=
  (total_cards : Nat)
  (number_cards : Nat)
  (face_cards : Nat)

/-- The probability of drawing a number card first and a face card second -/
def probability_number_then_face (d : Deck) : Rat :=
  (d.number_cards : Rat) / d.total_cards * d.face_cards / (d.total_cards - 1)

/-- Theorem stating the probability of drawing a number card first and a face card second from a standard deck -/
theorem probability_in_standard_deck :
  let standard_deck : Deck := ⟨52, 36, 12⟩
  probability_number_then_face standard_deck = 36 / 221 := by
  sorry

end NUMINAMATH_CALUDE_probability_in_standard_deck_l3684_368410


namespace NUMINAMATH_CALUDE_tan_45_degrees_equals_one_l3684_368467

theorem tan_45_degrees_equals_one :
  Real.tan (π / 4) = 1 := by
  sorry

end NUMINAMATH_CALUDE_tan_45_degrees_equals_one_l3684_368467


namespace NUMINAMATH_CALUDE_hockey_league_games_l3684_368485

theorem hockey_league_games (n : ℕ) (m : ℕ) (total_games : ℕ) 
  (h1 : n = 25)  -- number of teams
  (h2 : m = 15)  -- number of times each pair of teams face each other
  : total_games = 4500 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_hockey_league_games_l3684_368485


namespace NUMINAMATH_CALUDE_max_gcd_of_product_600_l3684_368424

theorem max_gcd_of_product_600 :
  ∃ (a b : ℕ), a * b = 600 ∧ 
  (∀ (c d : ℕ), c * d = 600 → Nat.gcd a b ≥ Nat.gcd c d) ∧
  Nat.gcd a b = 10 := by
sorry

end NUMINAMATH_CALUDE_max_gcd_of_product_600_l3684_368424


namespace NUMINAMATH_CALUDE_gcd_84_120_l3684_368453

theorem gcd_84_120 : Nat.gcd 84 120 = 12 := by
  sorry

end NUMINAMATH_CALUDE_gcd_84_120_l3684_368453


namespace NUMINAMATH_CALUDE_square_circle_union_area_l3684_368478

/-- The area of the union of a square and a circle with specific dimensions -/
theorem square_circle_union_area (square_side : ℝ) (circle_radius : ℝ) : 
  square_side = 8 →
  circle_radius = 12 →
  (square_side ^ 2) + (Real.pi * circle_radius ^ 2) - (Real.pi * circle_radius ^ 2 / 4) = 64 + 108 * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_square_circle_union_area_l3684_368478


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l3684_368430

theorem quadratic_inequality_range (a : ℝ) : 
  (∃ x : ℝ, x^2 + 2*a*x + 1 < 0) ↔ a ∈ Set.Ioi 1 ∪ Set.Iio (-1) :=
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l3684_368430


namespace NUMINAMATH_CALUDE_inequality_proof_l3684_368419

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h : a * b + a * c + b * c = a + b + c) :
  a + b + c + 1 ≥ 4 * a * b * c := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3684_368419


namespace NUMINAMATH_CALUDE_h_zero_at_seven_fifths_l3684_368459

/-- The function h(x) = 5x - 7 -/
def h (x : ℝ) : ℝ := 5 * x - 7

/-- Theorem: The value of b that satisfies h(b) = 0 is 7/5 -/
theorem h_zero_at_seven_fifths : ∃ b : ℝ, h b = 0 ∧ b = 7/5 := by
  sorry

end NUMINAMATH_CALUDE_h_zero_at_seven_fifths_l3684_368459


namespace NUMINAMATH_CALUDE_tony_age_at_period_end_l3684_368415

/-- Represents Tony's work and payment details -/
structure TonyWork where
  hoursPerDay : ℕ
  payPerHourPerYear : ℚ
  daysWorked : ℕ
  totalEarned : ℚ

/-- Proves Tony's age at the end of the six-month period -/
theorem tony_age_at_period_end (tw : TonyWork) 
  (h1 : tw.hoursPerDay = 2)
  (h2 : tw.payPerHourPerYear = 1/2)
  (h3 : tw.daysWorked = 50)
  (h4 : tw.totalEarned = 630) :
  ∃ (age : ℕ), age = 13 ∧ 
  (∃ (x : ℕ), x ≤ tw.daysWorked ∧
    (age - 1) * tw.hoursPerDay * tw.payPerHourPerYear * x + 
    age * tw.hoursPerDay * tw.payPerHourPerYear * (tw.daysWorked - x) = tw.totalEarned) :=
sorry

end NUMINAMATH_CALUDE_tony_age_at_period_end_l3684_368415


namespace NUMINAMATH_CALUDE_geometric_sequence_sixth_term_l3684_368461

/-- Given a geometric sequence with first term 512 and 8th term 2, the 6th term is 16 -/
theorem geometric_sequence_sixth_term : 
  ∀ (a : ℕ → ℝ), 
  (∀ n, a (n + 1) = a n * (a 8 / a 7)) →  -- Geometric sequence property
  a 1 = 512 →                            -- First term is 512
  a 8 = 2 →                              -- 8th term is 2
  a 6 = 16 := by
sorry

end NUMINAMATH_CALUDE_geometric_sequence_sixth_term_l3684_368461


namespace NUMINAMATH_CALUDE_kitchen_chairs_count_l3684_368427

/-- Represents the number of chairs bought for different rooms and in total. -/
structure ChairPurchase where
  total : Nat
  livingRoom : Nat

/-- Calculates the number of chairs bought for the kitchen. -/
def kitchenChairs (purchase : ChairPurchase) : Nat :=
  purchase.total - purchase.livingRoom

/-- Theorem stating that for the given purchase, the number of kitchen chairs is 6. -/
theorem kitchen_chairs_count (purchase : ChairPurchase) 
  (h1 : purchase.total = 9) 
  (h2 : purchase.livingRoom = 3) : 
  kitchenChairs purchase = 6 := by
  sorry

end NUMINAMATH_CALUDE_kitchen_chairs_count_l3684_368427


namespace NUMINAMATH_CALUDE_common_chord_length_l3684_368429

-- Define the circles in polar coordinates
def circle_O1 (ρ θ : ℝ) : Prop := ρ = 2
def circle_O2 (ρ θ : ℝ) : Prop := ρ^2 - 2 * Real.sqrt 2 * ρ * Real.cos (θ - Real.pi / 4) = 2

-- Define the circles in rectangular coordinates
def rect_O1 (x y : ℝ) : Prop := x^2 + y^2 = 4
def rect_O2 (x y : ℝ) : Prop := x^2 + y^2 - 2*x - 2*y - 2 = 0

-- Define the line AB
def line_AB (x y : ℝ) : Prop := x + y - 1 = 0

-- Theorem statement
theorem common_chord_length :
  ∃ (A B : ℝ × ℝ),
    rect_O1 A.1 A.2 ∧ rect_O1 B.1 B.2 ∧
    rect_O2 A.1 A.2 ∧ rect_O2 B.1 B.2 ∧
    line_AB A.1 A.2 ∧ line_AB B.1 B.2 ∧
    Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = Real.sqrt (2 * (2 + Real.sqrt 14)) :=
by sorry

end NUMINAMATH_CALUDE_common_chord_length_l3684_368429


namespace NUMINAMATH_CALUDE_z_greater_than_w_by_50_percent_l3684_368489

theorem z_greater_than_w_by_50_percent 
  (w e y z : ℝ) 
  (hw : w = 0.6 * e) 
  (he : e = 0.6 * y) 
  (hz : z = 0.54 * y) : 
  z = 1.5 * w := by
sorry

end NUMINAMATH_CALUDE_z_greater_than_w_by_50_percent_l3684_368489


namespace NUMINAMATH_CALUDE_circle_center_coordinate_sum_l3684_368474

theorem circle_center_coordinate_sum :
  ∀ (x y : ℝ), x^2 + y^2 = 6*x - 10*y + 24 →
  ∃ (center_x center_y : ℝ),
    (∀ (p_x p_y : ℝ), (p_x - center_x)^2 + (p_y - center_y)^2 = (x - center_x)^2 + (y - center_y)^2) ∧
    center_x + center_y = -2 :=
by sorry

end NUMINAMATH_CALUDE_circle_center_coordinate_sum_l3684_368474


namespace NUMINAMATH_CALUDE_polynomial_factorization_l3684_368404

theorem polynomial_factorization (x a : ℝ) : 
  x^3 - 3*x^2 + (a+2)*x - 2*a = (x^2 - x + a)*(x - 2) := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l3684_368404


namespace NUMINAMATH_CALUDE_vector_magnitude_problem_l3684_368452

variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V]

theorem vector_magnitude_problem (a b : V) 
  (h1 : ‖a - b‖ = Real.sqrt 3)
  (h2 : ‖a + b‖ = ‖2 • a - b‖) :
  ‖b‖ = Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_vector_magnitude_problem_l3684_368452


namespace NUMINAMATH_CALUDE_ellipse_eccentricity_l3684_368488

/-- The eccentricity of an ellipse with equation x²/16 + y²/25 = 1 is 3/5 -/
theorem ellipse_eccentricity : 
  let a : ℝ := 5
  let b : ℝ := 4
  let c : ℝ := Real.sqrt (a^2 - b^2)
  c / a = 3 / 5 := by sorry

end NUMINAMATH_CALUDE_ellipse_eccentricity_l3684_368488


namespace NUMINAMATH_CALUDE_smallest_y_for_prime_abs_quadratic_l3684_368428

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

def abs_quadratic (y : ℤ) : ℕ := Int.natAbs (5 * y^2 - 56 * y + 12)

theorem smallest_y_for_prime_abs_quadratic :
  (∀ y : ℤ, y < 11 → ¬(is_prime (abs_quadratic y))) ∧
  (is_prime (abs_quadratic 11)) :=
sorry

end NUMINAMATH_CALUDE_smallest_y_for_prime_abs_quadratic_l3684_368428


namespace NUMINAMATH_CALUDE_years_calculation_l3684_368446

/-- The number of years for which a sum was put at simple interest -/
def years_at_interest (principal : ℚ) (additional_interest : ℚ) : ℚ :=
  additional_interest * 100 / principal

theorem years_calculation (principal : ℚ) (additional_interest : ℚ) 
  (h1 : principal = 2600)
  (h2 : additional_interest = 78) :
  years_at_interest principal additional_interest = 3 := by
  sorry

#eval years_at_interest 2600 78

end NUMINAMATH_CALUDE_years_calculation_l3684_368446


namespace NUMINAMATH_CALUDE_purely_imaginary_z_l3684_368438

theorem purely_imaginary_z (a : ℝ) :
  let z : ℂ := a^2 - a + a * Complex.I
  (∃ b : ℝ, z = b * Complex.I ∧ b ≠ 0) → a = 1 := by
  sorry

end NUMINAMATH_CALUDE_purely_imaginary_z_l3684_368438


namespace NUMINAMATH_CALUDE_revenue_difference_l3684_368420

/-- Represents the sales data for a season -/
structure SeasonData where
  packsPerHour : ℕ
  pricePerPack : ℕ
  salesHours : ℕ

/-- Calculates the revenue for a given season -/
def calculateRevenue (data : SeasonData) : ℕ :=
  data.packsPerHour * data.pricePerPack * data.salesHours

/-- The peak season data -/
def peakSeason : SeasonData := {
  packsPerHour := 8,
  pricePerPack := 70,
  salesHours := 17
}

/-- The low season data -/
def lowSeason : SeasonData := {
  packsPerHour := 5,
  pricePerPack := 50,
  salesHours := 14
}

/-- The theorem stating the difference in revenue between peak and low seasons -/
theorem revenue_difference : 
  calculateRevenue peakSeason - calculateRevenue lowSeason = 6020 := by
  sorry


end NUMINAMATH_CALUDE_revenue_difference_l3684_368420


namespace NUMINAMATH_CALUDE_paper_strip_dimensions_l3684_368401

theorem paper_strip_dimensions (a b c : ℕ) 
  (h1 : a > 0) 
  (h2 : b > 0) 
  (h3 : c > 0) 
  (h4 : a * b + a * c + a * (b - a) + a^2 + a * (c - a) = 43) : 
  (a = 1 ∧ b + c = 22) ∨ (a = 22 ∧ b + c = 1) :=
sorry

end NUMINAMATH_CALUDE_paper_strip_dimensions_l3684_368401


namespace NUMINAMATH_CALUDE_loan_repayment_amount_l3684_368440

/-- The amount to be paid back after applying compound interest -/
def compound_interest (principal : ℝ) (rate : ℝ) (time : ℕ) : ℝ :=
  principal * (1 + rate) ^ time

/-- Theorem stating that the amount to be paid back is $168.54 -/
theorem loan_repayment_amount : 
  let principal : ℝ := 150
  let rate : ℝ := 0.06
  let time : ℕ := 2
  abs (compound_interest principal rate time - 168.54) < 0.01 := by
  sorry

end NUMINAMATH_CALUDE_loan_repayment_amount_l3684_368440


namespace NUMINAMATH_CALUDE_alloy_composition_proof_l3684_368476

-- Define the initial alloy properties
def initial_weight : ℝ := 25
def initial_gold_percentage : ℝ := 0.5
def target_gold_percentage : ℝ := 0.9

-- Define the amount of pure gold to be added
def added_gold : ℝ := 100

-- Theorem statement
theorem alloy_composition_proof :
  let initial_gold := initial_weight * initial_gold_percentage
  let final_weight := initial_weight + added_gold
  let final_gold := initial_gold + added_gold
  (final_gold / final_weight) = target_gold_percentage := by
  sorry

end NUMINAMATH_CALUDE_alloy_composition_proof_l3684_368476


namespace NUMINAMATH_CALUDE_tangent_circle_radius_l3684_368449

-- Define the curve C1
def C1 (t : ℝ) : ℝ × ℝ := (8 * t^2, 8 * t)

-- Define the circle C2
def C2 (r : ℝ) : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1^2 + p.2^2 = r^2}

-- Define the focus of C1
def focus_C1 : ℝ × ℝ := (2, 0)

-- Define the line with slope 1 passing through the focus of C1
def tangent_line (x y : ℝ) : Prop := y - x = -2

-- State the theorem
theorem tangent_circle_radius 
  (r : ℝ) 
  (hr : r > 0) 
  (h_tangent : ∃ p : ℝ × ℝ, p ∈ C2 r ∧ tangent_line p.1 p.2) : 
  r = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_tangent_circle_radius_l3684_368449


namespace NUMINAMATH_CALUDE_number_equation_solution_l3684_368421

theorem number_equation_solution : ∃ x : ℝ, (3/5) * x + 7 = (1/4) * x^2 - (1/2) * (1/3) * x := by
  sorry

end NUMINAMATH_CALUDE_number_equation_solution_l3684_368421


namespace NUMINAMATH_CALUDE_shopping_expenditure_theorem_l3684_368473

theorem shopping_expenditure_theorem (x : ℝ) : 
  x ≥ 0 ∧ x ≤ 100 ∧
  x / 100 + 0.3 + 0.3 = 1 ∧
  0.04 * (x / 100) + 0.08 * 0.3 = 0.04 →
  x = 40 := by sorry

end NUMINAMATH_CALUDE_shopping_expenditure_theorem_l3684_368473


namespace NUMINAMATH_CALUDE_quadratic_sum_l3684_368468

-- Define the quadratic function
def f (a b c x : ℝ) : ℝ := a * x^2 + b * x + c

-- State the theorem
theorem quadratic_sum (a b c : ℝ) :
  f a b c 0 = 8 → f a b c 1 = 9 → a + b = 1 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_sum_l3684_368468


namespace NUMINAMATH_CALUDE_f_increasing_on_interval_l3684_368493

-- Define the function f(x) = x^2 + 1
def f (x : ℝ) : ℝ := x^2 + 1

-- State the theorem
theorem f_increasing_on_interval :
  ∀ x y : ℝ, 0 < x ∧ x < y ∧ y < 2 → f x < f y :=
sorry

end NUMINAMATH_CALUDE_f_increasing_on_interval_l3684_368493


namespace NUMINAMATH_CALUDE_arithmetic_expression_equality_l3684_368464

theorem arithmetic_expression_equality : 
  1 / 2 + ((2 / 3 * 3 / 8) + 4) - 8 / 16 = 17 / 4 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_expression_equality_l3684_368464


namespace NUMINAMATH_CALUDE_simplify_trig_expression_l3684_368454

theorem simplify_trig_expression (θ : ℝ) (h : θ ∈ Set.Icc (5 * Real.pi / 4) (3 * Real.pi / 2)) :
  Real.sqrt (1 - Real.sin (2 * θ)) - Real.sqrt (1 + Real.sin (2 * θ)) = 2 * Real.sin θ := by
  sorry

end NUMINAMATH_CALUDE_simplify_trig_expression_l3684_368454


namespace NUMINAMATH_CALUDE_average_price_decrease_l3684_368482

theorem average_price_decrease (original_price final_price : ℝ) 
  (h1 : original_price = 10000)
  (h2 : final_price = 6400)
  (h3 : final_price = original_price * (1 - x)^2)
  (h4 : x > 0 ∧ x < 1) : 
  x = 0.2 :=
sorry

end NUMINAMATH_CALUDE_average_price_decrease_l3684_368482


namespace NUMINAMATH_CALUDE_paris_study_time_l3684_368441

/-- Calculates the total study time for a student during a semester. -/
def totalStudyTime (semesterWeeks : ℕ) (weekdayStudyHours : ℕ) (saturdayStudyHours : ℕ) (sundayStudyHours : ℕ) : ℕ :=
  let weekdaysTotalHours := 5 * weekdayStudyHours
  let weekendTotalHours := saturdayStudyHours + sundayStudyHours
  let weeklyTotalHours := weekdaysTotalHours + weekendTotalHours
  semesterWeeks * weeklyTotalHours

theorem paris_study_time :
  totalStudyTime 15 3 4 5 = 360 := by
  sorry

end NUMINAMATH_CALUDE_paris_study_time_l3684_368441


namespace NUMINAMATH_CALUDE_total_shirts_washed_l3684_368444

theorem total_shirts_washed (short_sleeve : ℕ) (long_sleeve : ℕ) 
  (h1 : short_sleeve = 4) (h2 : long_sleeve = 5) : 
  short_sleeve + long_sleeve = 9 := by
  sorry

end NUMINAMATH_CALUDE_total_shirts_washed_l3684_368444


namespace NUMINAMATH_CALUDE_total_area_of_triangles_l3684_368481

/-- Given two right triangles ABC and ABD with shared side AB, prove their total area -/
theorem total_area_of_triangles (AB AC BD : ℝ) : 
  AB = 15 →
  AC = 10 →
  BD = 8 →
  (1/2 * AB * AC) + (1/2 * AB * BD) = 135 := by
  sorry

end NUMINAMATH_CALUDE_total_area_of_triangles_l3684_368481


namespace NUMINAMATH_CALUDE_ratio_of_numbers_l3684_368458

theorem ratio_of_numbers (x y : ℝ) (hx : x > 0) (hy : y > 0) (hxy : x > y) 
  (sum_diff : x + y = 7 * (x - y)) : x / y = 4 / 3 := by
  sorry

end NUMINAMATH_CALUDE_ratio_of_numbers_l3684_368458


namespace NUMINAMATH_CALUDE_infinite_geometric_series_first_term_l3684_368463

theorem infinite_geometric_series_first_term
  (r : ℝ) (S : ℝ) (a : ℝ)
  (h_r : r = 1 / 4)
  (h_S : S = 40)
  (h_sum : S = a / (1 - r)) :
  a = 30 := by
sorry

end NUMINAMATH_CALUDE_infinite_geometric_series_first_term_l3684_368463


namespace NUMINAMATH_CALUDE_next_interval_is_two_point_five_l3684_368483

/-- Represents a decreasing geometric sequence of comet appearance intervals. -/
structure CometIntervals where
  /-- The common ratio of the geometric sequence. -/
  ratio : ℝ
  /-- Constraint that the ratio is positive and less than 1. -/
  ratio_pos : 0 < ratio
  ratio_lt_one : ratio < 1
  /-- The first term of the sequence (latest interval). -/
  first_term : ℝ
  /-- Constraint that the first term is positive. -/
  first_term_pos : 0 < first_term

/-- The three latest intervals satisfy the cubic equation. -/
def satisfies_cubic_equation (intervals : CometIntervals) : Prop :=
  ∃ c : ℝ, 
    let t₁ := intervals.first_term
    let t₂ := t₁ * intervals.ratio
    let t₃ := t₂ * intervals.ratio
    t₁^3 - c * t₁^2 + 350 * t₁ - 1000 = 0 ∧
    t₂^3 - c * t₂^2 + 350 * t₂ - 1000 = 0 ∧
    t₃^3 - c * t₃^2 + 350 * t₃ - 1000 = 0

/-- The theorem stating that the next interval will be 2.5 years. -/
theorem next_interval_is_two_point_five (intervals : CometIntervals) 
  (h : satisfies_cubic_equation intervals) : 
  intervals.first_term * intervals.ratio^3 = 2.5 := by
  sorry

end NUMINAMATH_CALUDE_next_interval_is_two_point_five_l3684_368483


namespace NUMINAMATH_CALUDE_factors_of_243_l3684_368425

theorem factors_of_243 : Finset.card (Nat.divisors 243) = 6 := by
  sorry

end NUMINAMATH_CALUDE_factors_of_243_l3684_368425


namespace NUMINAMATH_CALUDE_milk_for_nine_cookies_l3684_368416

/-- The number of pints in a quart -/
def pints_per_quart : ℚ := 2

/-- The number of cookies that can be baked with 3 quarts of milk -/
def cookies_per_three_quarts : ℚ := 18

/-- The number of cookies we want to bake -/
def target_cookies : ℚ := 9

/-- The function that calculates the number of pints of milk needed for a given number of cookies -/
def milk_needed (cookies : ℚ) : ℚ :=
  (3 * pints_per_quart * cookies) / cookies_per_three_quarts

theorem milk_for_nine_cookies :
  milk_needed target_cookies = 3 := by
  sorry

end NUMINAMATH_CALUDE_milk_for_nine_cookies_l3684_368416


namespace NUMINAMATH_CALUDE_part1_solution_part2_solution_l3684_368431

-- Define the function f
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + (1-a)*x + (1-a)

-- Part 1
theorem part1_solution :
  {x : ℝ | f 4 x ≥ 7} = {x : ℝ | x ≥ 5 ∨ x ≤ -2} := by sorry

-- Part 2
theorem part2_solution :
  (∀ x, x > -1 → f a x > 0) → a ≤ 1 := by sorry

end NUMINAMATH_CALUDE_part1_solution_part2_solution_l3684_368431


namespace NUMINAMATH_CALUDE_solution_set_when_a_is_one_f_always_greater_equal_g_iff_l3684_368491

-- Define the functions f and g
def f (a : ℝ) (x : ℝ) : ℝ := |2*x - a| + |2*x + 1|
def g (x : ℝ) : ℝ := x + 2

-- Statement for part (1)
theorem solution_set_when_a_is_one :
  {x : ℝ | f 1 x ≤ g x} = {x : ℝ | 0 ≤ x ∧ x ≤ 2/3} := by sorry

-- Statement for part (2)
theorem f_always_greater_equal_g_iff (a : ℝ) :
  (∀ x : ℝ, f a x ≥ g x) ↔ a ≥ 2 := by sorry

-- Condition that a > 0
axiom a_positive : ∀ a : ℝ, a > 0

end NUMINAMATH_CALUDE_solution_set_when_a_is_one_f_always_greater_equal_g_iff_l3684_368491


namespace NUMINAMATH_CALUDE_school_pairing_fraction_l3684_368402

theorem school_pairing_fraction :
  ∀ (s n : ℕ), 
    s > 0 → n > 0 →
    (n : ℚ) / 4 = (s : ℚ) / 3 →
    ((s : ℚ) / 3 + (n : ℚ) / 4) / ((s : ℚ) + (n : ℚ)) = 2 / 7 := by
  sorry

end NUMINAMATH_CALUDE_school_pairing_fraction_l3684_368402


namespace NUMINAMATH_CALUDE_x_one_minus_f_equals_one_l3684_368400

theorem x_one_minus_f_equals_one :
  let x : ℝ := (3 + Real.sqrt 8) ^ 100
  let n : ℤ := ⌊x⌋
  let f : ℝ := x - n
  x * (1 - f) = 1 := by
sorry

end NUMINAMATH_CALUDE_x_one_minus_f_equals_one_l3684_368400


namespace NUMINAMATH_CALUDE_point_M_coordinates_l3684_368407

/-- A point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- The distance of a point from the x-axis -/
def distanceFromXAxis (p : Point) : ℝ := |p.y|

/-- The distance of a point from the y-axis -/
def distanceFromYAxis (p : Point) : ℝ := |p.x|

theorem point_M_coordinates (a : ℝ) :
  let M : Point := ⟨2 - a, 3 * a + 6⟩
  distanceFromXAxis M = distanceFromYAxis M →
  M = ⟨3, 3⟩ := by
  sorry

end NUMINAMATH_CALUDE_point_M_coordinates_l3684_368407


namespace NUMINAMATH_CALUDE_mary_earnings_l3684_368462

def cleaning_rate : ℕ := 46
def babysitting_rate : ℕ := 35
def pet_care_rate : ℕ := 60

def homes_cleaned : ℕ := 4
def days_babysat : ℕ := 5
def days_pet_care : ℕ := 3

def total_earnings : ℕ := 
  cleaning_rate * homes_cleaned + 
  babysitting_rate * days_babysat + 
  pet_care_rate * days_pet_care

theorem mary_earnings : total_earnings = 539 := by
  sorry

end NUMINAMATH_CALUDE_mary_earnings_l3684_368462


namespace NUMINAMATH_CALUDE_not_sufficient_not_necessary_squared_inequality_l3684_368472

theorem not_sufficient_not_necessary_squared_inequality (a b : ℝ) :
  (∃ x y : ℝ, x > y ∧ x^2 ≤ y^2) ∧ (∃ u v : ℝ, u^2 > v^2 ∧ u ≤ v) := by
  sorry

end NUMINAMATH_CALUDE_not_sufficient_not_necessary_squared_inequality_l3684_368472


namespace NUMINAMATH_CALUDE_profit_percent_from_cost_price_ratio_l3684_368486

/-- Calculates the profit percent given the cost price as a percentage of the selling price -/
theorem profit_percent_from_cost_price_ratio (cost_price_ratio : ℝ) :
  cost_price_ratio = 0.25 → (1 / cost_price_ratio - 1) * 100 = 300 := by
  sorry

end NUMINAMATH_CALUDE_profit_percent_from_cost_price_ratio_l3684_368486


namespace NUMINAMATH_CALUDE_safe_journey_possible_l3684_368432

-- Define the duration of various segments of the journey
def road_duration : ℕ := 4
def trail_duration : ℕ := 4

-- Define the eruption patterns of the craters
def crater1_eruption : ℕ := 1
def crater1_silence : ℕ := 17
def crater2_eruption : ℕ := 1
def crater2_silence : ℕ := 9

-- Define the total journey time
def total_journey_time : ℕ := 2 * (road_duration + trail_duration)

-- Define a function to check if a given time is safe for travel
def is_safe_time (t : ℕ) : Prop :=
  let crater1_cycle := crater1_eruption + crater1_silence
  let crater2_cycle := crater2_eruption + crater2_silence
  ∀ i : ℕ, i < total_journey_time →
    (((t + i) % crater1_cycle ≥ crater1_eruption) ∨ (i < road_duration ∨ i ≥ road_duration + 2 * trail_duration)) ∧
    (((t + i) % crater2_cycle ≥ crater2_eruption) ∨ (i < road_duration ∨ i ≥ road_duration + trail_duration))

-- Theorem stating that a safe journey is possible
theorem safe_journey_possible : ∃ t : ℕ, is_safe_time t :=
sorry

end NUMINAMATH_CALUDE_safe_journey_possible_l3684_368432


namespace NUMINAMATH_CALUDE_sin_thirteen_pi_fourths_l3684_368434

theorem sin_thirteen_pi_fourths : Real.sin (13 * π / 4) = -Real.sqrt 2 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_thirteen_pi_fourths_l3684_368434


namespace NUMINAMATH_CALUDE_sequence_properties_l3684_368499

theorem sequence_properties (a b c : ℝ) : 
  (a ≠ b ∧ b ≠ c ∧ a ≠ c) →
  (∃ (d : ℝ), b = a + d ∧ c = b + d) →
  (b^2 = a*c ∧ a*c > 0) →
  (∃ (x y z : ℝ), x ≠ y ∧ y ≠ z ∧ x ≠ z ∧
    (∃ (d : ℝ), y = x + d ∧ z = y + d) ∧
    y^2 = x*z ∧ x*z > 0) ∧
  (∃ (p q r : ℝ), ¬ (∃ (m n : ℚ), p = m/n) ∧
                  ¬ (∃ (m n : ℚ), q = m/n) ∧
                  ¬ (∃ (m n : ℚ), r = m/n) ∧
                  (∃ (d : ℚ), q = p + d ∧ r = q + d)) :=
by sorry

end NUMINAMATH_CALUDE_sequence_properties_l3684_368499


namespace NUMINAMATH_CALUDE_simplify_expression_l3684_368469

theorem simplify_expression (p : ℝ) : 
  ((5 * p + 1) - 2 * p * 4) * 3 + (4 - 1 / 3) * (6 * p - 9) = 13 * p - 30 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l3684_368469


namespace NUMINAMATH_CALUDE_unique_k_for_rational_solutions_l3684_368457

def has_rational_solutions (a b c : ℤ) : Prop :=
  ∃ (x : ℚ), a * x^2 + b * x + c = 0

theorem unique_k_for_rational_solutions :
  ∀ k : ℕ+, (has_rational_solutions k 16 k ↔ k = 7) :=
sorry

end NUMINAMATH_CALUDE_unique_k_for_rational_solutions_l3684_368457


namespace NUMINAMATH_CALUDE_total_ways_eq_2501_l3684_368460

/-- The number of different types of cookies --/
def num_cookie_types : ℕ := 6

/-- The number of different types of milk --/
def num_milk_types : ℕ := 4

/-- The total number of item types --/
def total_item_types : ℕ := num_cookie_types + num_milk_types

/-- The number of items they purchase collectively --/
def total_items : ℕ := 4

/-- Represents a purchase combination for Charlie and Delta --/
structure Purchase where
  charlie_items : ℕ
  delta_items : ℕ
  charlie_items_le_total_types : charlie_items ≤ total_item_types
  delta_items_le_cookies : delta_items ≤ num_cookie_types
  sum_eq_total : charlie_items + delta_items = total_items

/-- The number of ways to choose items for a given purchase combination --/
def ways_to_choose (p : Purchase) : ℕ := sorry

/-- The total number of ways Charlie and Delta can purchase items --/
def total_ways : ℕ := sorry

/-- The main theorem: proving the total number of ways is 2501 --/
theorem total_ways_eq_2501 : total_ways = 2501 := sorry

end NUMINAMATH_CALUDE_total_ways_eq_2501_l3684_368460


namespace NUMINAMATH_CALUDE_solve_for_m_l3684_368479

theorem solve_for_m (m : ℝ) (h1 : m ≠ 0) :
  (∀ x : ℝ, (x^2 - m) * (x + m) = x^3 + m * (x^2 - x - 12)) →
  m = 12 := by
sorry

end NUMINAMATH_CALUDE_solve_for_m_l3684_368479


namespace NUMINAMATH_CALUDE_total_cost_is_2075_l3684_368408

def grapes_quantity : ℕ := 8
def grapes_price : ℕ := 70
def mangoes_quantity : ℕ := 9
def mangoes_price : ℕ := 55
def apples_quantity : ℕ := 4
def apples_price : ℕ := 40
def oranges_quantity : ℕ := 6
def oranges_price : ℕ := 30
def pineapples_quantity : ℕ := 2
def pineapples_price : ℕ := 90
def cherries_quantity : ℕ := 5
def cherries_price : ℕ := 100

def total_cost : ℕ := 
  grapes_quantity * grapes_price + 
  mangoes_quantity * mangoes_price + 
  apples_quantity * apples_price + 
  oranges_quantity * oranges_price + 
  pineapples_quantity * pineapples_price + 
  cherries_quantity * cherries_price

theorem total_cost_is_2075 : total_cost = 2075 := by
  sorry

end NUMINAMATH_CALUDE_total_cost_is_2075_l3684_368408


namespace NUMINAMATH_CALUDE_unemployment_forecast_l3684_368439

theorem unemployment_forecast (x : ℝ) (h1 : x > 0) : 
  let current_unemployed := 0.056 * x
  let next_year_active := 1.04 * x
  let next_year_unemployed := 0.91 * current_unemployed
  (next_year_unemployed / next_year_active) * 100 = 4.9 := by
sorry


end NUMINAMATH_CALUDE_unemployment_forecast_l3684_368439


namespace NUMINAMATH_CALUDE_number_relation_l3684_368484

theorem number_relation (first second multiple : ℕ) : 
  first = 15 →
  second = 55 →
  first + second = 70 →
  second = multiple * first + 10 →
  multiple = 3 := by
sorry

end NUMINAMATH_CALUDE_number_relation_l3684_368484


namespace NUMINAMATH_CALUDE_c_used_car_for_11_hours_l3684_368409

/-- Calculates the number of hours c used the car given the total cost,
    a's usage, b's usage and payment. -/
def calculate_c_hours (total_cost : ℕ) (a_hours : ℕ) (b_hours : ℕ) (b_payment : ℕ) : ℕ :=
  let hourly_rate := b_payment / b_hours
  let a_payment := a_hours * hourly_rate
  let c_payment := total_cost - a_payment - b_payment
  c_payment / hourly_rate

/-- Proves that given the problem conditions, c used the car for 11 hours -/
theorem c_used_car_for_11_hours :
  calculate_c_hours 520 7 8 160 = 11 := by
  sorry

end NUMINAMATH_CALUDE_c_used_car_for_11_hours_l3684_368409


namespace NUMINAMATH_CALUDE_journey_with_four_encounters_takes_five_days_l3684_368494

/-- A train journey with daily departures -/
structure TrainJourney where
  /-- The number of trains encountered during the journey -/
  trains_encountered : ℕ

/-- The duration of a train journey in days -/
def journey_duration (j : TrainJourney) : ℕ :=
  j.trains_encountered + 1

/-- Theorem: A train journey where 4 trains are encountered takes 5 days -/
theorem journey_with_four_encounters_takes_five_days (j : TrainJourney) 
    (h : j.trains_encountered = 4) : journey_duration j = 5 := by
  sorry

end NUMINAMATH_CALUDE_journey_with_four_encounters_takes_five_days_l3684_368494


namespace NUMINAMATH_CALUDE_negation_divisible_by_two_even_l3684_368497

theorem negation_divisible_by_two_even :
  (¬ ∀ n : ℤ, 2 ∣ n → Even n) ↔ (∃ n : ℤ, 2 ∣ n ∧ ¬ Even n) :=
by sorry

end NUMINAMATH_CALUDE_negation_divisible_by_two_even_l3684_368497


namespace NUMINAMATH_CALUDE_arithmetic_sequence_middle_term_l3684_368496

/-- Given an arithmetic sequence with 5 terms, first term 13, and last term 49,
    prove that the middle term (3rd term) is 31. -/
theorem arithmetic_sequence_middle_term :
  ∀ (a : ℕ → ℝ),
    (∀ i j, a (i + 1) - a i = a (j + 1) - a j) →  -- arithmetic sequence
    a 0 = 13 →                                   -- first term is 13
    a 4 = 49 →                                   -- last term is 49
    a 2 = 31 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_middle_term_l3684_368496


namespace NUMINAMATH_CALUDE_set_operations_l3684_368480

def A : Set ℝ := {x | 3 ≤ x ∧ x < 6}
def B : Set ℝ := {x | x^2 + 18 < 11*x}

theorem set_operations :
  (Set.compl (A ∩ B) = {x | x < 3 ∨ x ≥ 6}) ∧
  ((Set.compl B ∪ A) = {x | x ≤ 2 ∨ (3 ≤ x ∧ x < 6) ∨ x ≥ 9}) := by
  sorry

end NUMINAMATH_CALUDE_set_operations_l3684_368480


namespace NUMINAMATH_CALUDE_hyperbola_focus_coincides_with_parabola_focus_l3684_368445

/-- The focus of a parabola y^2 = 4x -/
def parabola_focus : ℝ × ℝ := (1, 0)

/-- The equation of the hyperbola -/
def hyperbola_equation (x y : ℝ) : Prop :=
  (5 * x^2) / 3 - (5 * y^2) / 2 = 1

/-- The right focus of the hyperbola -/
def hyperbola_right_focus : ℝ × ℝ := (1, 0)

/-- Theorem stating that the right focus of the hyperbola coincides with the focus of the parabola -/
theorem hyperbola_focus_coincides_with_parabola_focus :
  hyperbola_right_focus = parabola_focus :=
sorry

end NUMINAMATH_CALUDE_hyperbola_focus_coincides_with_parabola_focus_l3684_368445


namespace NUMINAMATH_CALUDE_min_value_theorem_l3684_368447

theorem min_value_theorem (x y z : ℝ) (h_pos_x : x > 0) (h_pos_y : y > 0) (h_pos_z : z > 0) 
  (h_sum_squares : x^2 + y^2 + z^2 = 1) : 
  ∀ ε > 0, ∃ x₀ y₀ z₀ : ℝ, 
    x₀ > 0 ∧ y₀ > 0 ∧ z₀ > 0 ∧ 
    x₀^2 + y₀^2 + z₀^2 = 1 ∧
    (z₀ + 1)^2 / (2 * x₀ * y₀ * z₀) < 3 + 2 * Real.sqrt 2 + ε ∧
    (z + 1)^2 / (2 * x * y * z) ≥ 3 + 2 * Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_min_value_theorem_l3684_368447


namespace NUMINAMATH_CALUDE_minimum_discount_rate_correct_l3684_368470

/-- The minimum discount rate (as a percentage) that ensures a gross profit of at least 12.5% 
    for a product with a cost price of 400 yuan and a selling price of 500 yuan. -/
def minimum_discount_rate : ℝ := 9

/-- The cost price of the product in yuan. -/
def cost_price : ℝ := 400

/-- The selling price of the product in yuan. -/
def selling_price : ℝ := 500

/-- The minimum required gross profit as a percentage of the cost price. -/
def min_gross_profit_percentage : ℝ := 12.5

theorem minimum_discount_rate_correct :
  ∀ x : ℝ, x ≥ minimum_discount_rate → 
  (selling_price - selling_price * (x / 100) - cost_price ≥ cost_price * (min_gross_profit_percentage / 100)) ∧
  ∀ y : ℝ, y < minimum_discount_rate → 
  (selling_price - selling_price * (y / 100) - cost_price > cost_price * (min_gross_profit_percentage / 100)) :=
by sorry

end NUMINAMATH_CALUDE_minimum_discount_rate_correct_l3684_368470


namespace NUMINAMATH_CALUDE_substitution_result_l3684_368451

theorem substitution_result (x y : ℝ) : 
  (y = 2 * x - 3) → 
  (x - 2 * y = 6) → 
  (x - 4 * x + 6 = 6) := by
sorry

end NUMINAMATH_CALUDE_substitution_result_l3684_368451


namespace NUMINAMATH_CALUDE_A_power_50_l3684_368450

def A : Matrix (Fin 2) (Fin 2) ℤ := !![5, 1; -16, -3]

theorem A_power_50 : A^50 = !![201, 50; -800, -199] := by sorry

end NUMINAMATH_CALUDE_A_power_50_l3684_368450


namespace NUMINAMATH_CALUDE_divisibility_by_24_l3684_368490

theorem divisibility_by_24 (p : ℕ) (h_prime : Nat.Prime p) (h_ge_5 : p ≥ 5) : 
  (p^2 - 1) % 24 = 0 := by
sorry

end NUMINAMATH_CALUDE_divisibility_by_24_l3684_368490


namespace NUMINAMATH_CALUDE_arithmetic_calculations_l3684_368435

theorem arithmetic_calculations :
  (12 - (-18) + (-11) - 15 = 4) ∧
  (-(2^2) / (4/9) * ((-2/3)^2) = -4) ∧
  ((-3) * (-7/5) + (-1 - 3/7) = 97/35) ∧
  (2 + 1/3 - (2 + 3/5) + (5 + 2/3) - (4 + 2/5) = 1) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculations_l3684_368435


namespace NUMINAMATH_CALUDE_worker_earnings_l3684_368487

/-- Calculate the total earnings of a worker for a week --/
theorem worker_earnings (ordinary_rate : ℚ) (overtime_rate : ℚ) 
  (total_hours : ℕ) (overtime_hours : ℕ) : 
  ordinary_rate = 60/100 →
  overtime_rate = 90/100 →
  total_hours = 50 →
  overtime_hours = 8 →
  (total_hours - overtime_hours) * ordinary_rate + 
    overtime_hours * overtime_rate = 3240/100 := by
  sorry

end NUMINAMATH_CALUDE_worker_earnings_l3684_368487


namespace NUMINAMATH_CALUDE_letter_e_to_square_l3684_368418

/-- Represents a piece cut from the letter E -/
structure Piece where
  area : ℝ
  shape : Set (ℝ × ℝ)

/-- Represents the letter E with its dimensions -/
structure LetterE where
  height : ℝ
  width : ℝ
  horizontal_bar_length : ℝ

/-- Checks if a set of pieces can form a square -/
def can_form_square (pieces : List Piece) : Prop :=
  ∃ (side_length : ℝ), 
    side_length > 0 ∧
    (pieces.map (λ p => p.area)).sum = side_length ^ 2

/-- Checks if a list of pieces is a valid cutting of the letter E -/
def is_valid_cutting (e : LetterE) (pieces : List Piece) : Prop :=
  pieces.length = 5 ∧
  (pieces.map (λ p => p.area)).sum = e.height * e.width + 3 * e.horizontal_bar_length * e.width

/-- Main theorem: It's possible to cut the letter E into five pieces to form a square -/
theorem letter_e_to_square (e : LetterE) : 
  ∃ (pieces : List Piece), is_valid_cutting e pieces ∧ can_form_square pieces := by
  sorry

end NUMINAMATH_CALUDE_letter_e_to_square_l3684_368418


namespace NUMINAMATH_CALUDE_total_arrangements_l3684_368456

def num_students : ℕ := 8
def max_participants : ℕ := 5
def num_activities : ℕ := 2

def valid_distribution (dist : List ℕ) : Prop :=
  dist.length = num_activities ∧
  dist.sum = num_students ∧
  ∀ x ∈ dist, x ≤ max_participants

def num_arrangements : ℕ := sorry

theorem total_arrangements :
  num_arrangements = 182 :=
sorry

end NUMINAMATH_CALUDE_total_arrangements_l3684_368456


namespace NUMINAMATH_CALUDE_f_4_equals_40_l3684_368477

-- Define the function f
def f (x : ℝ) (a b c : ℝ) : ℝ := 2 * a * x + b * x + c

-- State the theorem
theorem f_4_equals_40 
  (h1 : f 1 a b c = 10) 
  (h2 : f 2 a b c = 20) : 
  f 4 a b c = 40 := by
  sorry

end NUMINAMATH_CALUDE_f_4_equals_40_l3684_368477


namespace NUMINAMATH_CALUDE_will_money_distribution_l3684_368495

theorem will_money_distribution (x : ℚ) :
  let amount1 := 5 * x
  let amount2 := 3 * x
  let amount3 := 2 * x
  amount2 = 42 →
  amount1 + amount2 + amount3 = 140 :=
by sorry

end NUMINAMATH_CALUDE_will_money_distribution_l3684_368495


namespace NUMINAMATH_CALUDE_max_satiated_pikes_l3684_368426

/-- Represents the number of pikes in the pond -/
def total_pikes : ℕ := 30

/-- Represents the minimum number of pikes a satiated pike must eat -/
def min_eaten : ℕ := 3

/-- Predicate to check if a number is a valid count of satiated pikes -/
def is_valid_satiated_count (n : ℕ) : Prop :=
  n * min_eaten < total_pikes ∧ n ≤ total_pikes

/-- Theorem stating that the maximum number of satiated pikes is 9 -/
theorem max_satiated_pikes :
  ∃ (max : ℕ), is_valid_satiated_count max ∧
  ∀ (n : ℕ), is_valid_satiated_count n → n ≤ max :=
by sorry

end NUMINAMATH_CALUDE_max_satiated_pikes_l3684_368426


namespace NUMINAMATH_CALUDE_candy_count_l3684_368436

theorem candy_count (packages : ℕ) (pieces_per_package : ℕ) (h1 : packages = 45) (h2 : pieces_per_package = 9) :
  packages * pieces_per_package = 405 := by
  sorry

end NUMINAMATH_CALUDE_candy_count_l3684_368436


namespace NUMINAMATH_CALUDE_perpendicular_lines_m_values_l3684_368405

theorem perpendicular_lines_m_values (m : ℝ) : 
  (∀ x y : ℝ, (m + 2) * x + m * y + 1 = 0 ∧ 
               (m - 1) * x + (m - 4) * y + 2 = 0 → 
               ((m + 2) * (m - 1) + m * (m - 4) = 0)) → 
  m = 2 ∨ m = -1/2 := by
sorry

end NUMINAMATH_CALUDE_perpendicular_lines_m_values_l3684_368405


namespace NUMINAMATH_CALUDE_proportional_segments_l3684_368403

theorem proportional_segments (a : ℝ) :
  (a > 0) → ((a / 2) = (6 / (a + 1))) → a = 3 := by
  sorry

end NUMINAMATH_CALUDE_proportional_segments_l3684_368403


namespace NUMINAMATH_CALUDE_line_perpendicular_condition_l3684_368411

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the perpendicular and parallel relations
variable (perp : Line → Line → Prop)
variable (perpLP : Line → Plane → Prop)
variable (perpPP : Plane → Plane → Prop)
variable (parallel : Plane → Plane → Prop)
variable (subset : Line → Plane → Prop)

-- State the theorem
theorem line_perpendicular_condition 
  (a b : Line) (α β : Plane) :
  subset a α → perpLP b β → ¬ parallel α β → perp a b :=
sorry

end NUMINAMATH_CALUDE_line_perpendicular_condition_l3684_368411


namespace NUMINAMATH_CALUDE_quadratic_roots_ratio_l3684_368437

theorem quadratic_roots_ratio (k : ℝ) : 
  (∃ x y : ℝ, x ≠ 0 ∧ y ≠ 0 ∧ x / y = 3 ∧ 
   x^2 + 8*x + k = 0 ∧ y^2 + 8*y + k = 0) → k = 12 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_ratio_l3684_368437


namespace NUMINAMATH_CALUDE_cubic_poly_roots_theorem_l3684_368443

/-- A cubic polynomial p(x) = x^3 + cx + d -/
def cubic_poly (c d x : ℝ) : ℝ := x^3 + c*x + d

theorem cubic_poly_roots_theorem (c d : ℝ) :
  ∃ (u v : ℝ),
    (∀ x, cubic_poly c d x = 0 ↔ x = u ∨ x = v ∨ x = -u-v) ∧
    (∀ x, cubic_poly c (d + 360) x = 0 ↔ x = u+3 ∨ x = v-5 ∨ x = -u-v+2) →
    d = -2601 ∨ d = -693 :=
by sorry

end NUMINAMATH_CALUDE_cubic_poly_roots_theorem_l3684_368443


namespace NUMINAMATH_CALUDE_no_fixed_points_iff_negative_discriminant_l3684_368455

/-- A function f: ℝ → ℝ has a fixed point if there exists an x₀ ∈ ℝ such that f(x₀) = x₀ -/
def has_fixed_point (f : ℝ → ℝ) : Prop :=
  ∃ x₀ : ℝ, f x₀ = x₀

/-- The quadratic function f(x) = x² + ax + 1 -/
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + a*x + 1

/-- The discriminant of the equation x² + (a-1)x + 1 = 0 -/
def discriminant (a : ℝ) : ℝ := (a-1)^2 - 4

theorem no_fixed_points_iff_negative_discriminant (a : ℝ) :
  ¬(has_fixed_point (f a)) ↔ discriminant a < 0 :=
sorry

end NUMINAMATH_CALUDE_no_fixed_points_iff_negative_discriminant_l3684_368455


namespace NUMINAMATH_CALUDE_regular_polygon_vertices_l3684_368465

/-- A regular polygon with an angle of 135° between a vertex and the vertex two positions away has 12 vertices. -/
theorem regular_polygon_vertices (n : ℕ) (h_regular : n ≥ 3) : 
  (2 * (360 : ℝ) / n = 135) → n = 12 := by
  sorry

end NUMINAMATH_CALUDE_regular_polygon_vertices_l3684_368465


namespace NUMINAMATH_CALUDE_traffic_light_probabilities_l3684_368433

/-- Represents the duration of each traffic light state in seconds -/
structure TrafficLightDuration where
  red : ℕ
  yellow : ℕ
  green : ℕ

/-- Calculates the total duration of a traffic light cycle -/
def cycle_duration (d : TrafficLightDuration) : ℕ :=
  d.red + d.yellow + d.green

/-- Calculates the probability of encountering a specific light state -/
def light_probability (duration : ℕ) (total : ℕ) : ℚ :=
  ↑duration / ↑total

/-- Theorem stating the probabilities of encountering each traffic light state -/
theorem traffic_light_probabilities (d : TrafficLightDuration)
  (h_red : d.red = 40)
  (h_yellow : d.yellow = 5)
  (h_green : d.green = 50) :
  let total := cycle_duration d
  (light_probability d.red total = 8 / 19) ∧
  (light_probability d.yellow total = 1 / 19) ∧
  (light_probability (d.yellow + d.green) total = 11 / 19) := by
  sorry


end NUMINAMATH_CALUDE_traffic_light_probabilities_l3684_368433


namespace NUMINAMATH_CALUDE_product_of_roots_l3684_368448

theorem product_of_roots (x : ℝ) : (x + 3) * (x - 4) = 22 → ∃ y : ℝ, (x + 3) * (x - 4) = 22 ∧ x * y = -34 := by
  sorry

end NUMINAMATH_CALUDE_product_of_roots_l3684_368448


namespace NUMINAMATH_CALUDE_negation_equivalence_l3684_368423

-- Define the proposition for "at least one of a, b, c is positive"
def atLeastOnePositive (a b c : ℝ) : Prop := a > 0 ∨ b > 0 ∨ c > 0

-- Define the proposition for "a, b, c are all non-positive"
def allNonPositive (a b c : ℝ) : Prop := a ≤ 0 ∧ b ≤ 0 ∧ c ≤ 0

-- Theorem stating the negation equivalence
theorem negation_equivalence (a b c : ℝ) : 
  ¬(atLeastOnePositive a b c) ↔ allNonPositive a b c :=
by sorry

end NUMINAMATH_CALUDE_negation_equivalence_l3684_368423


namespace NUMINAMATH_CALUDE_complex_power_four_l3684_368492

theorem complex_power_four (i : ℂ) : i^2 = -1 → (1 - i)^4 = -4 := by sorry

end NUMINAMATH_CALUDE_complex_power_four_l3684_368492


namespace NUMINAMATH_CALUDE_solution_x_equals_two_l3684_368422

theorem solution_x_equals_two : 
  ∃ x : ℝ, x = 2 ∧ 7 * x - 14 = 0 := by
sorry

end NUMINAMATH_CALUDE_solution_x_equals_two_l3684_368422


namespace NUMINAMATH_CALUDE_sugar_recipe_reduction_l3684_368414

theorem sugar_recipe_reduction : 
  let original_recipe : ℚ := 5 + 3/4
  let reduced_recipe : ℚ := (1/3) * original_recipe
  reduced_recipe = 1 + 11/12 := by
sorry

end NUMINAMATH_CALUDE_sugar_recipe_reduction_l3684_368414


namespace NUMINAMATH_CALUDE_solution_set_theorem_l3684_368417

/-- A differentiable function satisfying the given condition -/
def SatisfyingFunction (f : ℝ → ℝ) : Prop :=
  Differentiable ℝ f ∧ 
  (∀ x > 0, x * (deriv f x) + 2 * f x > 0)

/-- The solution set of the inequality -/
def SolutionSet (f : ℝ → ℝ) : Set ℝ :=
  {x | (x + 2016) * f (x + 2016) / 5 < 5 * f 5 / (x + 2016)}

/-- The theorem stating the solution set of the inequality -/
theorem solution_set_theorem (f : ℝ → ℝ) (hf : SatisfyingFunction f) :
  SolutionSet f = {x | -2016 < x ∧ x < -2011} :=
sorry

end NUMINAMATH_CALUDE_solution_set_theorem_l3684_368417


namespace NUMINAMATH_CALUDE_smallest_m_for_integral_roots_existence_of_integral_roots_smallest_m_is_195_l3684_368406

theorem smallest_m_for_integral_roots (m : ℤ) : m > 0 → 
  (∃ x y : ℤ, 15 * x^2 - m * x + 630 = 0 ∧ 
              15 * y^2 - m * y + 630 = 0 ∧ 
              abs (x - y) ≤ 10) →
  m ≥ 195 :=
by sorry

theorem existence_of_integral_roots : 
  ∃ x y : ℤ, 15 * x^2 - 195 * x + 630 = 0 ∧ 
            15 * y^2 - 195 * y + 630 = 0 ∧ 
            abs (x - y) ≤ 10 :=
by sorry

theorem smallest_m_is_195 : 
  ∀ m : ℤ, m > 0 → 
  (∃ x y : ℤ, 15 * x^2 - m * x + 630 = 0 ∧ 
            15 * y^2 - m * y + 630 = 0 ∧ 
            abs (x - y) ≤ 10) → 
  m ≥ 195 ∧ 
  (∃ x y : ℤ, 15 * x^2 - 195 * x + 630 = 0 ∧ 
            15 * y^2 - 195 * y + 630 = 0 ∧ 
            abs (x - y) ≤ 10) :=
by sorry

end NUMINAMATH_CALUDE_smallest_m_for_integral_roots_existence_of_integral_roots_smallest_m_is_195_l3684_368406
