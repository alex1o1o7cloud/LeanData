import Mathlib

namespace NUMINAMATH_CALUDE_units_digit_of_six_to_seven_l2336_233679

/-- The units digit of a natural number -/
def unitsDigit (n : ℕ) : ℕ := n % 10

/-- Theorem: The units digit of 6^7 is 6 -/
theorem units_digit_of_six_to_seven :
  unitsDigit (6^7) = 6 := by sorry

end NUMINAMATH_CALUDE_units_digit_of_six_to_seven_l2336_233679


namespace NUMINAMATH_CALUDE_point_movement_l2336_233685

/-- Represents a point on a number line -/
structure Point where
  value : ℝ

/-- Moves a point on the number line by a given distance -/
def move (p : Point) (distance : ℝ) : Point :=
  ⟨p.value + distance⟩

theorem point_movement (A : Point) (h : A.value = -3) :
  (move A 7).value = 4 := by
  sorry

end NUMINAMATH_CALUDE_point_movement_l2336_233685


namespace NUMINAMATH_CALUDE_fifth_root_of_eight_to_fifteen_l2336_233623

theorem fifth_root_of_eight_to_fifteen (x : ℝ) : x = (8 ^ (1 / 5 : ℝ)) → x^15 = 512 := by
  sorry

end NUMINAMATH_CALUDE_fifth_root_of_eight_to_fifteen_l2336_233623


namespace NUMINAMATH_CALUDE_right_triangle_leg_square_l2336_233699

/-- In a right triangle with hypotenuse c and legs a and b, where c = a + 2, 
    the square of b is equal to 4a + 4. -/
theorem right_triangle_leg_square (a b c : ℝ) 
  (h_right : a^2 + b^2 = c^2)  -- Right triangle condition
  (h_diff : c = a + 2)         -- Hypotenuse and leg difference condition
  : b^2 = 4*a + 4 := by
sorry

end NUMINAMATH_CALUDE_right_triangle_leg_square_l2336_233699


namespace NUMINAMATH_CALUDE_centroid_quadrilateral_area_ratio_l2336_233674

/-- A point in a 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- A quadrilateral defined by four points -/
structure Quadrilateral where
  A : Point
  B : Point
  C : Point
  D : Point

/-- Checks if a quadrilateral is convex -/
def isConvex (q : Quadrilateral) : Prop := sorry

/-- Checks if a point is inside a quadrilateral -/
def isInterior (p : Point) (q : Quadrilateral) : Prop := sorry

/-- Calculates the centroid of a triangle -/
def centroid (a b c : Point) : Point := sorry

/-- Calculates the area of a quadrilateral -/
def area (q : Quadrilateral) : ℝ := sorry

/-- Main theorem -/
theorem centroid_quadrilateral_area_ratio 
  (ABCD : Quadrilateral) 
  (P : Point) 
  (h1 : isConvex ABCD) 
  (h2 : isInterior P ABCD) : 
  let G1 := centroid ABCD.A ABCD.B P
  let G2 := centroid ABCD.B ABCD.C P
  let G3 := centroid ABCD.C ABCD.D P
  let G4 := centroid ABCD.D ABCD.A P
  let centroidQuad : Quadrilateral := ⟨G1, G2, G3, G4⟩
  area centroidQuad / area ABCD = 1 / 9 := by sorry

end NUMINAMATH_CALUDE_centroid_quadrilateral_area_ratio_l2336_233674


namespace NUMINAMATH_CALUDE_jury_seating_arrangements_l2336_233603

/-- Represents the number of jury members -/
def n : ℕ := 12

/-- Represents the number of jury members excluding Nikolai Nikolaevich and the person whose seat he took -/
def m : ℕ := n - 2

/-- A function that calculates the number of distinct seating arrangements -/
def seating_arrangements (n : ℕ) : ℕ := 2^(n - 2)

/-- Theorem stating that the number of distinct seating arrangements for 12 jury members is 2^10 -/
theorem jury_seating_arrangements :
  seating_arrangements n = 2^m :=
by sorry

end NUMINAMATH_CALUDE_jury_seating_arrangements_l2336_233603


namespace NUMINAMATH_CALUDE_geometric_sequence_fifth_term_l2336_233631

/-- A geometric sequence is a sequence where the ratio of successive terms is constant. -/
def IsGeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

/-- Given a geometric sequence a_n where a_3 = -1 and a_7 = -9, then a_5 = -3. -/
theorem geometric_sequence_fifth_term
  (a : ℕ → ℝ)
  (h_geo : IsGeometricSequence a)
  (h_3 : a 3 = -1)
  (h_7 : a 7 = -9) :
  a 5 = -3 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_fifth_term_l2336_233631


namespace NUMINAMATH_CALUDE_existence_of_solution_l2336_233650

theorem existence_of_solution :
  ∃ t : ℝ, Real.exp (1 - 2*t) = 3 * Real.sin (2*t - 2) + Real.cos (2*t) := by
  sorry

end NUMINAMATH_CALUDE_existence_of_solution_l2336_233650


namespace NUMINAMATH_CALUDE_christmas_book_sales_l2336_233659

/-- Given a ratio of books to bookmarks and the number of bookmarks sold,
    calculate the number of books sold. -/
def books_sold (book_ratio : ℕ) (bookmark_ratio : ℕ) (bookmarks_sold : ℕ) : ℕ :=
  (book_ratio * bookmarks_sold) / bookmark_ratio

/-- Theorem stating that given the specific ratio and number of bookmarks sold,
    the number of books sold is 72. -/
theorem christmas_book_sales : books_sold 9 2 16 = 72 := by
  sorry

end NUMINAMATH_CALUDE_christmas_book_sales_l2336_233659


namespace NUMINAMATH_CALUDE_negative_five_times_three_l2336_233669

theorem negative_five_times_three : -5 * 3 = -15 := by
  sorry

end NUMINAMATH_CALUDE_negative_five_times_three_l2336_233669


namespace NUMINAMATH_CALUDE_expression_equality_l2336_233672

theorem expression_equality : (50 + 20 / 90) * 90 = 4520 := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l2336_233672


namespace NUMINAMATH_CALUDE_not_divides_power_minus_one_l2336_233682

theorem not_divides_power_minus_one (n : ℕ) (h : n > 1) :
  ¬(n ∣ (2^n - 1)) := by
sorry

end NUMINAMATH_CALUDE_not_divides_power_minus_one_l2336_233682


namespace NUMINAMATH_CALUDE_trash_cans_on_streets_l2336_233664

theorem trash_cans_on_streets (street_cans back_cans : ℕ) : 
  back_cans = 2 * street_cans → 
  street_cans + back_cans = 42 → 
  street_cans = 14 := by
sorry

end NUMINAMATH_CALUDE_trash_cans_on_streets_l2336_233664


namespace NUMINAMATH_CALUDE_not_perfect_square_2005_l2336_233676

/-- A polynomial with integer coefficients -/
def IntPolynomial := ℕ → ℤ

/-- Evaluates a polynomial at a given point -/
def eval (P : IntPolynomial) (x : ℤ) : ℤ :=
  sorry

/-- Checks if a number is a perfect square -/
def is_perfect_square (n : ℤ) : Prop :=
  sorry

theorem not_perfect_square_2005 (P : IntPolynomial) :
  eval P 5 = 2005 → ¬(is_perfect_square (eval P 2005)) :=
sorry

end NUMINAMATH_CALUDE_not_perfect_square_2005_l2336_233676


namespace NUMINAMATH_CALUDE_largest_divisor_of_five_consecutive_even_integers_l2336_233666

theorem largest_divisor_of_five_consecutive_even_integers (n : ℕ) (h : Even n) (h' : n > 0) :
  ∃ (d : ℕ), d = 96 ∧
  (∀ (k : ℕ), k > 96 → ¬(k ∣ n * (n + 2) * (n + 4) * (n + 6) * (n + 8))) ∧
  (96 ∣ n * (n + 2) * (n + 4) * (n + 6) * (n + 8)) :=
sorry

end NUMINAMATH_CALUDE_largest_divisor_of_five_consecutive_even_integers_l2336_233666


namespace NUMINAMATH_CALUDE_purely_imaginary_condition_fourth_quadrant_condition_l2336_233648

def z (a : ℝ) : ℂ := Complex.mk (a^2 - 7*a + 6) (a^2 - 5*a - 6)

theorem purely_imaginary_condition (a : ℝ) : 
  (z a).re = 0 ∧ (z a).im ≠ 0 → a = 1 := by sorry

theorem fourth_quadrant_condition (a : ℝ) :
  (z a).re > 0 ∧ (z a).im < 0 → -1 < a ∧ a < 1 := by sorry

end NUMINAMATH_CALUDE_purely_imaginary_condition_fourth_quadrant_condition_l2336_233648


namespace NUMINAMATH_CALUDE_max_runs_in_match_l2336_233625

/-- Represents the maximum number of runs that can be scored in a single delivery -/
def max_runs_per_delivery : ℕ := 6

/-- Represents the number of deliveries in an over -/
def deliveries_per_over : ℕ := 6

/-- Represents the total number of overs in the match -/
def total_overs : ℕ := 35

/-- Represents the maximum number of consecutive boundaries allowed in an over -/
def max_consecutive_boundaries : ℕ := 3

/-- Calculates the maximum runs that can be scored in a single over -/
def max_runs_per_over : ℕ :=
  max_consecutive_boundaries * max_runs_per_delivery + 
  (deliveries_per_over - max_consecutive_boundaries)

/-- Theorem: The maximum number of runs a batsman can score in the given match is 735 -/
theorem max_runs_in_match : 
  total_overs * max_runs_per_over = 735 := by
  sorry

end NUMINAMATH_CALUDE_max_runs_in_match_l2336_233625


namespace NUMINAMATH_CALUDE_f_min_at_three_l2336_233698

/-- The quadratic function f(x) = 3x^2 - 18x + 7 -/
def f (x : ℝ) : ℝ := 3 * x^2 - 18 * x + 7

/-- Theorem stating that f(x) attains its minimum value when x = 3 -/
theorem f_min_at_three : ∀ x : ℝ, f x ≥ f 3 := by sorry

end NUMINAMATH_CALUDE_f_min_at_three_l2336_233698


namespace NUMINAMATH_CALUDE_unique_line_through_sqrt3_and_rationals_l2336_233609

-- Define a point in R²
structure Point where
  x : ℝ
  y : ℝ

-- Define a line passing through (√3, 0)
structure Line where
  slope : ℝ

def isRational (x : ℝ) : Prop := ∃ (q : ℚ), x = q

def linePassesThroughRationalPoints (l : Line) : Prop :=
  ∃ (p1 p2 : Point), p1 ≠ p2 ∧ isRational p1.x ∧ isRational p1.y ∧ 
                     isRational p2.x ∧ isRational p2.y ∧
                     p1.y = l.slope * (p1.x - Real.sqrt 3) ∧
                     p2.y = l.slope * (p2.x - Real.sqrt 3)

theorem unique_line_through_sqrt3_and_rationals :
  ∃! (l : Line), linePassesThroughRationalPoints l :=
sorry

end NUMINAMATH_CALUDE_unique_line_through_sqrt3_and_rationals_l2336_233609


namespace NUMINAMATH_CALUDE_monica_has_27_peaches_l2336_233667

/-- The number of peaches each person has -/
structure Peaches where
  steven : ℕ
  jake : ℕ
  jill : ℕ
  monica : ℕ

/-- The conditions given in the problem -/
def peach_conditions (p : Peaches) : Prop :=
  p.steven = 16 ∧
  p.jake = p.steven - 7 ∧
  p.jake = p.jill + 9 ∧
  p.monica = 3 * p.jake

/-- Theorem: Given the conditions, Monica has 27 peaches -/
theorem monica_has_27_peaches (p : Peaches) (h : peach_conditions p) : p.monica = 27 := by
  sorry

end NUMINAMATH_CALUDE_monica_has_27_peaches_l2336_233667


namespace NUMINAMATH_CALUDE_mass_BaSO4_produced_l2336_233637

-- Define the molar masses of elements (in g/mol)
def molar_mass_Ba : ℝ := 137.327
def molar_mass_S : ℝ := 32.065
def molar_mass_O : ℝ := 15.999

-- Define the molar mass of Barium sulfate
def molar_mass_BaSO4 : ℝ := molar_mass_Ba + molar_mass_S + 4 * molar_mass_O

-- Define the number of moles of Barium bromide
def moles_BaBr2 : ℝ := 4

-- Theorem statement
theorem mass_BaSO4_produced (excess_Na2SO4 : Prop) (double_displacement : Prop) :
  moles_BaBr2 * molar_mass_BaSO4 = 933.552 := by
  sorry


end NUMINAMATH_CALUDE_mass_BaSO4_produced_l2336_233637


namespace NUMINAMATH_CALUDE_exists_primitive_root_mod_2p_alpha_l2336_233647

/-- Given an odd prime p and a natural number α, there exists a primitive root modulo 2p^α -/
theorem exists_primitive_root_mod_2p_alpha (p : Nat) (α : Nat) 
  (h_prime : Nat.Prime p) (h_odd : Odd p) : 
  ∃ x : Nat, IsPrimitiveRoot x (2 * p^α) := by
  sorry

end NUMINAMATH_CALUDE_exists_primitive_root_mod_2p_alpha_l2336_233647


namespace NUMINAMATH_CALUDE_pauls_initial_amount_l2336_233602

/-- The amount of money Paul initially had for shopping --/
def initial_amount : ℕ := 15

/-- The cost of bread --/
def bread_cost : ℕ := 2

/-- The cost of butter --/
def butter_cost : ℕ := 3

/-- The cost of juice (twice the price of bread) --/
def juice_cost : ℕ := 2 * bread_cost

/-- The amount Paul had left after shopping --/
def amount_left : ℕ := 6

/-- Theorem stating that Paul's initial amount equals the sum of his purchases and remaining money --/
theorem pauls_initial_amount :
  initial_amount = bread_cost + butter_cost + juice_cost + amount_left := by
  sorry

end NUMINAMATH_CALUDE_pauls_initial_amount_l2336_233602


namespace NUMINAMATH_CALUDE_fair_attendance_l2336_233632

/-- Proves the number of adults attending a fair given admission fees, total attendance, and total amount collected. -/
theorem fair_attendance 
  (child_fee : ℚ) 
  (adult_fee : ℚ) 
  (total_people : ℕ) 
  (total_amount : ℚ) 
  (h1 : child_fee = 3/2) 
  (h2 : adult_fee = 4) 
  (h3 : total_people = 2200) 
  (h4 : total_amount = 5050) : 
  ∃ (adults : ℕ), adults = 700 ∧ 
    ∃ (children : ℕ), 
      children + adults = total_people ∧ 
      child_fee * children + adult_fee * adults = total_amount := by
  sorry

end NUMINAMATH_CALUDE_fair_attendance_l2336_233632


namespace NUMINAMATH_CALUDE_special_number_fraction_l2336_233628

theorem special_number_fraction (numbers : List ℝ) (n : ℝ) :
  numbers.length = 21 ∧
  n ∈ numbers ∧
  n = 4 * ((numbers.sum - n) / 20) →
  n = (1 / 6) * numbers.sum :=
by sorry

end NUMINAMATH_CALUDE_special_number_fraction_l2336_233628


namespace NUMINAMATH_CALUDE_game_cost_l2336_233662

theorem game_cost (initial_money : ℕ) (num_toys : ℕ) (toy_cost : ℕ) (game_cost : ℕ) : 
  initial_money = 57 →
  num_toys = 5 →
  toy_cost = 6 →
  initial_money = game_cost + (num_toys * toy_cost) →
  game_cost = 27 := by
sorry

end NUMINAMATH_CALUDE_game_cost_l2336_233662


namespace NUMINAMATH_CALUDE_solar_panel_distribution_l2336_233691

theorem solar_panel_distribution (total_homes : ℕ) (installed_homes : ℕ) (panel_shortage : ℕ) :
  total_homes = 20 →
  installed_homes = 15 →
  panel_shortage = 50 →
  ∃ (panels_per_home : ℕ),
    panels_per_home = 10 ∧
    panels_per_home * total_homes = panels_per_home * installed_homes + panel_shortage :=
by sorry

end NUMINAMATH_CALUDE_solar_panel_distribution_l2336_233691


namespace NUMINAMATH_CALUDE_line_segment_proportions_l2336_233614

/-- Given line segments a and b, prove the fourth proportional and mean proportional -/
theorem line_segment_proportions (a b : ℝ) (ha : a = 5) (hb : b = 3) :
  let fourth_prop := b * (a - b) / a
  let mean_prop := Real.sqrt ((a + b) * (a - b))
  fourth_prop = 1.2 ∧ mean_prop = 4 := by
  sorry

end NUMINAMATH_CALUDE_line_segment_proportions_l2336_233614


namespace NUMINAMATH_CALUDE_simple_interest_time_period_l2336_233645

/-- Theorem: Simple Interest Time Period Calculation -/
theorem simple_interest_time_period 
  (P : ℝ) -- Principal amount
  (r : ℝ) -- Rate of interest per annum
  (t : ℝ) -- Time period in years
  (h1 : r = 12) -- Given rate is 12% per annum
  (h2 : (P * r * t) / 100 = (6/5) * P) -- Simple interest equation
  : t = 10 := by
sorry

end NUMINAMATH_CALUDE_simple_interest_time_period_l2336_233645


namespace NUMINAMATH_CALUDE_jose_profit_share_l2336_233616

/-- Calculates the share of profit for an investor in a partnership --/
def calculate_profit_share (investment1 : ℕ) (months1 : ℕ) (investment2 : ℕ) (months2 : ℕ) (total_profit : ℕ) : ℕ :=
  let total_investment := investment1 * months1 + investment2 * months2
  let share_ratio := investment2 * months2 * total_profit / total_investment
  share_ratio

/-- Proves that Jose's share of the profit is 3500 --/
theorem jose_profit_share :
  calculate_profit_share 3000 12 4500 10 6300 = 3500 := by
  sorry

end NUMINAMATH_CALUDE_jose_profit_share_l2336_233616


namespace NUMINAMATH_CALUDE_min_value_z_plus_inv_z_squared_l2336_233677

/-- Given a complex number z with positive real part, and a parallelogram formed by the points 0, z, 1/z, and z + 1/z with an area of 12/13, the minimum value of |z + 1/z|² is 16/13. -/
theorem min_value_z_plus_inv_z_squared (z : ℂ) (h_real_pos : 0 < z.re) 
  (h_area : abs (z.im * (1/z).re - z.re * (1/z).im) = 12/13) :
  ∃ d : ℝ, d^2 = 16/13 ∧ ∀ w : ℂ, w.re > 0 → 
    abs (w.im * (1/w).re - w.re * (1/w).im) = 12/13 → 
    d^2 ≤ Complex.normSq (w + 1/w) := by
  sorry

end NUMINAMATH_CALUDE_min_value_z_plus_inv_z_squared_l2336_233677


namespace NUMINAMATH_CALUDE_pedro_plums_problem_l2336_233657

theorem pedro_plums_problem (total_fruits : ℕ) (total_cost : ℕ) 
  (plum_cost peach_cost : ℕ) (h1 : total_fruits = 32) 
  (h2 : total_cost = 52) (h3 : plum_cost = 2) (h4 : peach_cost = 1) :
  ∃ (plums peaches : ℕ), 
    plums + peaches = total_fruits ∧
    plum_cost * plums + peach_cost * peaches = total_cost ∧
    plums = 20 := by
  sorry

end NUMINAMATH_CALUDE_pedro_plums_problem_l2336_233657


namespace NUMINAMATH_CALUDE_martha_started_with_three_cards_l2336_233665

/-- The number of cards Martha started with -/
def initial_cards : ℕ := sorry

/-- The number of cards Martha received from Emily -/
def cards_from_emily : ℕ := 76

/-- The total number of cards Martha ended up with -/
def total_cards : ℕ := 79

/-- Theorem stating that Martha started with 3 cards -/
theorem martha_started_with_three_cards : 
  initial_cards = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_martha_started_with_three_cards_l2336_233665


namespace NUMINAMATH_CALUDE_gmat_test_problem_l2336_233675

theorem gmat_test_problem (p_first : ℝ) (p_second : ℝ) (p_neither : ℝ)
  (h1 : p_first = 0.85)
  (h2 : p_second = 0.65)
  (h3 : p_neither = 0.05) :
  p_first + p_second - (1 - p_neither) = 0.55 := by
  sorry

end NUMINAMATH_CALUDE_gmat_test_problem_l2336_233675


namespace NUMINAMATH_CALUDE_point_symmetry_l2336_233620

/-- The line with respect to which we're finding symmetry -/
def symmetry_line (x y : ℝ) : Prop := x - y - 1 = 0

/-- Definition of symmetry with respect to a line -/
def symmetric_points (x₁ y₁ x₂ y₂ : ℝ) : Prop :=
  let midpoint_x := (x₁ + x₂) / 2
  let midpoint_y := (y₁ + y₂) / 2
  symmetry_line midpoint_x midpoint_y ∧
  (y₂ - y₁) / (x₂ - x₁) = -1

theorem point_symmetry :
  symmetric_points (-1) 1 2 (-2) := by sorry

end NUMINAMATH_CALUDE_point_symmetry_l2336_233620


namespace NUMINAMATH_CALUDE_farmer_budget_distribution_l2336_233607

theorem farmer_budget_distribution (g sh : ℕ) : 
  g > 0 ∧ sh > 0 ∧ 24 * g + 27 * sh = 1200 → g = 5 ∧ sh = 40 :=
by
  sorry

end NUMINAMATH_CALUDE_farmer_budget_distribution_l2336_233607


namespace NUMINAMATH_CALUDE_scientific_notation_equivalence_l2336_233604

theorem scientific_notation_equivalence : 
  ∃ (x : ℝ) (n : ℤ), 11580000 = x * (10 : ℝ) ^ n ∧ 1 ≤ x ∧ x < 10 := by
  sorry

end NUMINAMATH_CALUDE_scientific_notation_equivalence_l2336_233604


namespace NUMINAMATH_CALUDE_basketball_score_l2336_233613

theorem basketball_score (total_shots : ℕ) (three_point_shots : ℕ) : 
  total_shots = 11 → three_point_shots = 4 → 
  3 * three_point_shots + 2 * (total_shots - three_point_shots) = 26 := by
  sorry

end NUMINAMATH_CALUDE_basketball_score_l2336_233613


namespace NUMINAMATH_CALUDE_urn_problem_l2336_233619

theorem urn_problem (N : ℕ) : 
  let urn1_red : ℕ := 5
  let urn1_yellow : ℕ := 8
  let urn2_red : ℕ := 18
  let urn2_yellow : ℕ := N
  let total1 : ℕ := urn1_red + urn1_yellow
  let total2 : ℕ := urn2_red + urn2_yellow
  let prob_same_color : ℚ := (urn1_red / total1) * (urn2_red / total2) + 
                             (urn1_yellow / total1) * (urn2_yellow / total2)
  prob_same_color = 62/100 → N = 59 := by
sorry


end NUMINAMATH_CALUDE_urn_problem_l2336_233619


namespace NUMINAMATH_CALUDE_no_natural_numbers_satisfying_condition_l2336_233615

theorem no_natural_numbers_satisfying_condition : 
  ¬ ∃ (a b : ℕ), a < b ∧ ∃ (k : ℕ), b^2 + 4*a = k^2 := by
  sorry

end NUMINAMATH_CALUDE_no_natural_numbers_satisfying_condition_l2336_233615


namespace NUMINAMATH_CALUDE_plane_q_satisfies_conditions_l2336_233606

/-- Plane type representing ax + by + cz + d = 0 --/
structure Plane where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

/-- Line type representing the intersection of two planes --/
structure Line where
  p1 : Plane
  p2 : Plane

/-- Point type in 3D space --/
structure Point where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Function to check if a plane contains a line --/
def containsLine (p : Plane) (l : Line) : Prop := sorry

/-- Function to calculate the distance between a plane and a point --/
def distancePlanePoint (p : Plane) (pt : Point) : ℝ := sorry

/-- Given planes --/
def plane1 : Plane := ⟨1, 3, 2, -4⟩
def plane2 : Plane := ⟨2, -1, 3, -6⟩

/-- Line M --/
def lineM : Line := ⟨plane1, plane2⟩

/-- Given point --/
def givenPoint : Point := ⟨4, 2, -2⟩

/-- Plane Q --/
def planeQ : Plane := ⟨1, -9, 5, -2⟩

theorem plane_q_satisfies_conditions :
  containsLine planeQ lineM ∧
  planeQ ≠ plane1 ∧
  planeQ ≠ plane2 ∧
  distancePlanePoint planeQ givenPoint = 3 / Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_plane_q_satisfies_conditions_l2336_233606


namespace NUMINAMATH_CALUDE_a_4_equals_8_l2336_233695

def geometric_sequence (a : ℝ) (q : ℝ) (n : ℕ) : ℝ := a * q^(n - 1)

theorem a_4_equals_8 
  (a : ℝ) (q : ℝ) 
  (h1 : geometric_sequence a q 6 + geometric_sequence a q 2 = 34)
  (h2 : geometric_sequence a q 6 - geometric_sequence a q 2 = 30) :
  geometric_sequence a q 4 = 8 :=
sorry

end NUMINAMATH_CALUDE_a_4_equals_8_l2336_233695


namespace NUMINAMATH_CALUDE_exterior_angle_theorem_l2336_233681

theorem exterior_angle_theorem (α β γ : ℝ) : 
  α + β + γ = 180 →  -- Sum of angles in a triangle is 180°
  α + β = 150 →      -- Exterior angle is 150°
  γ = 70 →           -- One remote interior angle is 70°
  β = 80 :=          -- The other remote interior angle is 80°
by sorry

end NUMINAMATH_CALUDE_exterior_angle_theorem_l2336_233681


namespace NUMINAMATH_CALUDE_average_speed_calculation_l2336_233612

theorem average_speed_calculation (distance1 distance2 time1 time2 : ℝ) 
  (h1 : distance1 = 90)
  (h2 : distance2 = 80)
  (h3 : time1 = 1)
  (h4 : time2 = 1) :
  (distance1 + distance2) / (time1 + time2) = 85 := by sorry

end NUMINAMATH_CALUDE_average_speed_calculation_l2336_233612


namespace NUMINAMATH_CALUDE_right_triangle_max_ratio_l2336_233634

theorem right_triangle_max_ratio (a b c : ℝ) (h1 : a > 0) (h2 : b > 0) (h3 : c > 0) 
  (h4 : a^2 + b^2 = c^2) : 
  (∀ x y z : ℝ, x > 0 → y > 0 → z > 0 → x^2 + y^2 = z^2 → 
    (x^2 + y^2 + x*y) / z^2 ≤ (a^2 + b^2 + a*b) / c^2) → 
  (a^2 + b^2 + a*b) / c^2 = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_max_ratio_l2336_233634


namespace NUMINAMATH_CALUDE_erasers_left_in_box_l2336_233668

/-- The number of erasers left in the box after Doris, Mark, and Ellie take some out. -/
def erasers_left (initial : ℕ) (doris_takes : ℕ) (mark_takes : ℕ) (ellie_takes : ℕ) : ℕ :=
  initial - doris_takes - mark_takes - ellie_takes

/-- Theorem stating that 105 erasers are left in the box -/
theorem erasers_left_in_box :
  erasers_left 250 75 40 30 = 105 := by
  sorry

end NUMINAMATH_CALUDE_erasers_left_in_box_l2336_233668


namespace NUMINAMATH_CALUDE_investment_return_correct_l2336_233621

def investment_return (n : ℕ+) : ℚ :=
  2^(n.val - 2)

theorem investment_return_correct :
  ∀ (n : ℕ+),
  (n = 1 → investment_return n = (1/2)) ∧
  (∀ (k : ℕ+), investment_return (k + 1) = 2 * investment_return k) :=
by sorry

end NUMINAMATH_CALUDE_investment_return_correct_l2336_233621


namespace NUMINAMATH_CALUDE_basketball_game_score_theorem_l2336_233642

/-- Represents a team's scores for each quarter -/
structure TeamScores :=
  (q1 : ℕ) (q2 : ℕ) (q3 : ℕ) (q4 : ℕ)

/-- Checks if a sequence of four numbers is an arithmetic sequence -/
def isArithmeticSequence (s : TeamScores) : Prop :=
  s.q2 - s.q1 = s.q3 - s.q2 ∧ s.q3 - s.q2 = s.q4 - s.q3 ∧ s.q2 > s.q1

/-- Checks if a sequence of four numbers is a geometric sequence -/
def isGeometricSequence (s : TeamScores) : Prop :=
  s.q2 / s.q1 = s.q3 / s.q2 ∧ s.q3 / s.q2 = s.q4 / s.q3 ∧ s.q2 > s.q1

/-- Calculates the total score for a team -/
def totalScore (s : TeamScores) : ℕ := s.q1 + s.q2 + s.q3 + s.q4

theorem basketball_game_score_theorem 
  (eagles lions : TeamScores) : 
  isArithmeticSequence eagles →
  isGeometricSequence lions →
  eagles.q1 = lions.q1 + 2 →
  eagles.q1 + eagles.q2 + eagles.q3 = lions.q1 + lions.q2 + lions.q3 →
  totalScore eagles ≤ 100 →
  totalScore lions ≤ 100 →
  totalScore eagles + totalScore lions = 144 := by
  sorry

#check basketball_game_score_theorem

end NUMINAMATH_CALUDE_basketball_game_score_theorem_l2336_233642


namespace NUMINAMATH_CALUDE_largest_awesome_prime_l2336_233683

def is_awesome_prime (p : ℕ) : Prop :=
  Nat.Prime p ∧ ∀ q : ℕ, 0 < q → q < p → Nat.Prime (p + 2 * q)

theorem largest_awesome_prime : 
  (∃ p : ℕ, is_awesome_prime p) ∧ 
  (∀ p : ℕ, is_awesome_prime p → p ≤ 3) :=
sorry

end NUMINAMATH_CALUDE_largest_awesome_prime_l2336_233683


namespace NUMINAMATH_CALUDE_larger_number_proof_l2336_233687

theorem larger_number_proof (a b : ℕ+) (x y : ℕ+) 
  (hcf_eq : Nat.gcd a b = 30)
  (x_eq : x = 10)
  (y_eq : y = 15)
  (lcm_eq : Nat.lcm a b = 30 * x * y) :
  max a b = 450 := by
sorry

end NUMINAMATH_CALUDE_larger_number_proof_l2336_233687


namespace NUMINAMATH_CALUDE_octahedron_tetrahedron_volume_relation_l2336_233694

/-- Regular octahedron with side length 2√2 -/
def octahedron : Real → Set (Fin 3 → ℝ) := sorry

/-- Tetrahedron with vertices at the centers of octahedron faces -/
def tetrahedron (O : Set (Fin 3 → ℝ)) : Set (Fin 3 → ℝ) := sorry

/-- Volume of a set in ℝ³ -/
def volume (S : Set (Fin 3 → ℝ)) : ℝ := sorry

theorem octahedron_tetrahedron_volume_relation :
  let O := octahedron (2 * Real.sqrt 2)
  let T := tetrahedron O
  volume O = 4 * volume T →
  volume T = (4 * Real.sqrt 2) / 3 := by sorry

end NUMINAMATH_CALUDE_octahedron_tetrahedron_volume_relation_l2336_233694


namespace NUMINAMATH_CALUDE_sumata_family_vacation_miles_l2336_233630

/-- Proves that given a 5-day vacation with a total of 1250 miles driven, the average miles driven per day is 250 miles. -/
theorem sumata_family_vacation_miles (total_miles : ℕ) (num_days : ℕ) (miles_per_day : ℕ) :
  total_miles = 1250 ∧ num_days = 5 ∧ miles_per_day = total_miles / num_days →
  miles_per_day = 250 :=
by sorry

end NUMINAMATH_CALUDE_sumata_family_vacation_miles_l2336_233630


namespace NUMINAMATH_CALUDE_max_value_linear_program_l2336_233624

theorem max_value_linear_program (x y : ℝ) 
  (h1 : x - y ≥ 0) 
  (h2 : x + 2*y ≤ 4) 
  (h3 : x - 2*y ≤ 2) : 
  ∃ (z : ℝ), z = x + 3*y ∧ z ≤ 16/3 ∧ 
  (∀ (x' y' : ℝ), x' - y' ≥ 0 → x' + 2*y' ≤ 4 → x' - 2*y' ≤ 2 → x' + 3*y' ≤ z) :=
by sorry

end NUMINAMATH_CALUDE_max_value_linear_program_l2336_233624


namespace NUMINAMATH_CALUDE_solution_set_implies_a_value_l2336_233636

theorem solution_set_implies_a_value (a : ℝ) : 
  (∀ x : ℝ, ((a * x - 1) * (x + 2) > 0) ↔ (-3 < x ∧ x < -2)) →
  a = -1/3 := by
sorry

end NUMINAMATH_CALUDE_solution_set_implies_a_value_l2336_233636


namespace NUMINAMATH_CALUDE_distance_home_to_school_l2336_233641

/-- The distance between home and school given the travel conditions --/
theorem distance_home_to_school :
  ∀ (D T : ℝ),
  (3 * (T + 7/60) = D) →
  (6 * (T - 8/60) = D) →
  D = 1.5 := by
  sorry

end NUMINAMATH_CALUDE_distance_home_to_school_l2336_233641


namespace NUMINAMATH_CALUDE_serenas_mother_age_l2336_233690

/-- Serena's current age -/
def serena_age : ℕ := 9

/-- Years into the future when the age comparison is made -/
def years_future : ℕ := 6

/-- Serena's mother's age now -/
def mother_age : ℕ := 39

/-- Theorem stating that Serena's mother's current age is 39 -/
theorem serenas_mother_age : 
  (mother_age + years_future) = 3 * (serena_age + years_future) → 
  mother_age = 39 := by
  sorry

end NUMINAMATH_CALUDE_serenas_mother_age_l2336_233690


namespace NUMINAMATH_CALUDE_complex_fraction_inequality_l2336_233673

theorem complex_fraction_inequality (a b c : ℂ) 
  (h1 : a * b + a * c - b * c ≠ 0) 
  (h2 : b * a + b * c - a * c ≠ 0) 
  (h3 : c * a + c * b - a * b ≠ 0) : 
  Complex.abs (a^2 / (a * b + a * c - b * c)) + 
  Complex.abs (b^2 / (b * a + b * c - a * c)) + 
  Complex.abs (c^2 / (c * a + c * b - a * b)) ≥ 3/2 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_inequality_l2336_233673


namespace NUMINAMATH_CALUDE_mathematician_meeting_theorem_l2336_233600

theorem mathematician_meeting_theorem (n p q r : ℕ) (h1 : n = p - q * Real.sqrt r) 
  (h2 : 0 < p ∧ 0 < q ∧ 0 < r) (h3 : ∀ (prime : ℕ), Prime prime → ¬(prime^2 ∣ r)) 
  (h4 : ((120 - n : ℝ) / 120)^2 = 1/2) : p + q + r = 182 := by
sorry

end NUMINAMATH_CALUDE_mathematician_meeting_theorem_l2336_233600


namespace NUMINAMATH_CALUDE_function_properties_l2336_233688

/-- The function f(x) with parameter a -/
def f (a : ℝ) (x : ℝ) : ℝ := 2 * x^3 - 3 * x^2 + a

theorem function_properties (a : ℝ) :
  (∃ x, ∀ y, f a y ≤ f a x) ∧ (∃ x, f a x = 6) →
  a = 6 ∧
  ∀ k, (∀ x t, x ∈ [-2, 2] → t ∈ [-1, 1] → f a x ≥ k * t - 25) ↔ k ∈ [-3, 3] :=
sorry

end NUMINAMATH_CALUDE_function_properties_l2336_233688


namespace NUMINAMATH_CALUDE_two_number_difference_l2336_233692

theorem two_number_difference (x y : ℝ) 
  (sum_eq : x + y = 40)
  (triple_minus_quad : 3 * y - 4 * x = 20) :
  |y - x| = 100 / 7 := by
sorry

end NUMINAMATH_CALUDE_two_number_difference_l2336_233692


namespace NUMINAMATH_CALUDE_complement_M_intersect_N_l2336_233611

-- Define the set M
def M : Set ℝ := {x | 2/x < 1}

-- Define the set N
def N : Set ℝ := {y | ∃ x, y = Real.sqrt (x - 1)}

-- The theorem to prove
theorem complement_M_intersect_N : (Set.univ \ M) ∩ N = Set.Icc 0 2 := by sorry

end NUMINAMATH_CALUDE_complement_M_intersect_N_l2336_233611


namespace NUMINAMATH_CALUDE_arcade_tickets_l2336_233649

theorem arcade_tickets (initial_tickets yoyo_cost : ℝ) 
  (h1 : initial_tickets = 48.5)
  (h2 : yoyo_cost = 11.7) : 
  initial_tickets - (initial_tickets - yoyo_cost) = yoyo_cost := by
sorry

end NUMINAMATH_CALUDE_arcade_tickets_l2336_233649


namespace NUMINAMATH_CALUDE_water_depth_calculation_l2336_233652

/-- The depth of water given Dean's height and a multiplier -/
def water_depth (dean_height : ℝ) (depth_multiplier : ℝ) : ℝ :=
  dean_height * depth_multiplier

/-- Theorem: The water depth is 60 feet when Dean's height is 6 feet
    and the depth is 10 times his height -/
theorem water_depth_calculation :
  water_depth 6 10 = 60 := by
  sorry

end NUMINAMATH_CALUDE_water_depth_calculation_l2336_233652


namespace NUMINAMATH_CALUDE_f_properties_l2336_233640

/-- Definition of an odd function -/
def OddFunction (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

/-- Definition of the function f -/
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x ≥ 0 then a^x - 1 else 1 - a^(-x)

theorem f_properties (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  OddFunction (f a) ∧ 
  (f a 2 + f a (-2) = 0) ∧
  (∀ x, f a x = if x ≥ 0 then a^x - 1 else 1 - a^(-x)) := by
  sorry

end NUMINAMATH_CALUDE_f_properties_l2336_233640


namespace NUMINAMATH_CALUDE_equation_solutions_l2336_233638

theorem equation_solutions :
  (∀ x : ℝ, (x - 1)^2 - 25 = 0 ↔ x = 6 ∨ x = -4) ∧
  (∀ x : ℝ, 3*x*(x - 2) = x - 2 ↔ x = 2 ∨ x = 1/3) := by
  sorry

end NUMINAMATH_CALUDE_equation_solutions_l2336_233638


namespace NUMINAMATH_CALUDE_max_sum_xy_l2336_233661

theorem max_sum_xy (x y a b : ℝ) (hx : x > 0) (hy : y > 0)
  (ha : 0 ≤ a ∧ a ≤ x) (hb : 0 ≤ b ∧ b ≤ y)
  (h1 : a^2 + y^2 = 2) (h2 : b^2 + x^2 = 1) (h3 : a*x + b*y = 1) :
  x + y ≤ 2 ∧ ∃ (x₀ y₀ : ℝ), x₀ + y₀ = 2 ∧
    ∃ (a₀ b₀ : ℝ), 0 ≤ a₀ ∧ a₀ ≤ x₀ ∧ 0 ≤ b₀ ∧ b₀ ≤ y₀ ∧
      a₀^2 + y₀^2 = 2 ∧ b₀^2 + x₀^2 = 1 ∧ a₀*x₀ + b₀*y₀ = 1 :=
by sorry

end NUMINAMATH_CALUDE_max_sum_xy_l2336_233661


namespace NUMINAMATH_CALUDE_cosine_identity_l2336_233678

theorem cosine_identity (z : ℂ) (α : ℝ) (h : z + 1/z = 2 * Real.cos α) :
  ∀ n : ℕ, z^n + 1/z^n = 2 * Real.cos (n * α) := by
  sorry

end NUMINAMATH_CALUDE_cosine_identity_l2336_233678


namespace NUMINAMATH_CALUDE_find_transmitter_probability_l2336_233639

/-- The number of possible government vehicle license plates starting with 79 -/
def total_vehicles : ℕ := 900

/-- The number of vehicles police can inspect per hour -/
def inspection_rate : ℕ := 6

/-- The search time in hours -/
def search_time : ℕ := 3

/-- The probability of finding the transmitter within the given search time -/
theorem find_transmitter_probability :
  (inspection_rate * search_time : ℚ) / total_vehicles = 1 / 50 := by
  sorry

end NUMINAMATH_CALUDE_find_transmitter_probability_l2336_233639


namespace NUMINAMATH_CALUDE_polar_to_cartesian_equivalence_l2336_233643

theorem polar_to_cartesian_equivalence :
  ∀ (x y ρ θ : ℝ),
    ρ = 2 * Real.sin θ + 4 * Real.cos θ →
    x = ρ * Real.cos θ →
    y = ρ * Real.sin θ →
    (x - 8)^2 + (y - 2)^2 = 68 :=
by sorry

end NUMINAMATH_CALUDE_polar_to_cartesian_equivalence_l2336_233643


namespace NUMINAMATH_CALUDE_popcorn_package_solution_l2336_233617

/-- Represents a package of popcorn buckets -/
structure Package where
  buckets : ℕ
  cost : ℚ

/-- Proves that buying 48 packages of Package B satisfies all conditions -/
theorem popcorn_package_solution :
  let package_b : Package := ⟨9, 8⟩
  let num_packages : ℕ := 48
  let total_buckets : ℕ := num_packages * package_b.buckets
  let total_cost : ℚ := num_packages * package_b.cost
  (total_buckets ≥ 426) ∧ 
  (total_cost ≤ 400) ∧ 
  (num_packages ≤ 60) :=
by
  sorry


end NUMINAMATH_CALUDE_popcorn_package_solution_l2336_233617


namespace NUMINAMATH_CALUDE_completing_square_transformation_l2336_233655

theorem completing_square_transformation :
  ∃ (m n : ℝ), (∀ x : ℝ, x^2 - 4*x - 4 = 0 ↔ (x + m)^2 = n) ∧ m = -2 ∧ n = 8 := by
  sorry

end NUMINAMATH_CALUDE_completing_square_transformation_l2336_233655


namespace NUMINAMATH_CALUDE_solve_equation_l2336_233626

theorem solve_equation (x : ℝ) :
  (1 / 7 : ℝ) + 7 / x = 15 / x + (1 / 15 : ℝ) → x = 105 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l2336_233626


namespace NUMINAMATH_CALUDE_quadratic_solution_property_l2336_233658

theorem quadratic_solution_property (p q : ℝ) : 
  (3 * p^2 + 4 * p - 8 = 0) → 
  (3 * q^2 + 4 * q - 8 = 0) → 
  (p - 2) * (q - 2) = 4 := by
sorry

end NUMINAMATH_CALUDE_quadratic_solution_property_l2336_233658


namespace NUMINAMATH_CALUDE_constant_d_value_l2336_233622

theorem constant_d_value (a d : ℝ) (h : ∀ x : ℝ, (x - 3) * (x + a) = x^2 + d*x - 18) : d = 3 := by
  sorry

end NUMINAMATH_CALUDE_constant_d_value_l2336_233622


namespace NUMINAMATH_CALUDE_sum_of_squares_of_solutions_l2336_233644

theorem sum_of_squares_of_solutions : ∃ (a b c d : ℝ),
  (|a^2 - 2*a + 1/1004| = 1/502) ∧
  (|b^2 - 2*b + 1/1004| = 1/502) ∧
  (|c^2 - 2*c + 1/1004| = 1/502) ∧
  (|d^2 - 2*d + 1/1004| = 1/502) ∧
  (a^2 + b^2 + c^2 + d^2 = 8050/1008) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_squares_of_solutions_l2336_233644


namespace NUMINAMATH_CALUDE_east_northwest_angle_l2336_233671

/-- A circle with ten equally spaced rays -/
structure TenRayCircle where
  rays : Fin 10 → ℝ
  north_ray : rays 0 = 0
  equally_spaced : ∀ i : Fin 10, rays i = (i : ℝ) * 36

/-- The angle between two rays in a TenRayCircle -/
def angle_between (c : TenRayCircle) (i j : Fin 10) : ℝ :=
  ((j - i : ℤ) % 10 : ℤ) * 36

theorem east_northwest_angle (c : TenRayCircle) :
  min (angle_between c 3 8) (angle_between c 8 3) = 144 :=
sorry

end NUMINAMATH_CALUDE_east_northwest_angle_l2336_233671


namespace NUMINAMATH_CALUDE_promotional_activity_choices_l2336_233660

/-- The number of ways to choose k elements from n elements -/
def choose (n k : ℕ) : ℕ := sorry

/-- The number of ways to choose volunteers for the promotional activity -/
def chooseVolunteers (totalVolunteers boyCount girlCount chosenCount : ℕ) : ℕ :=
  choose boyCount 3 * choose girlCount 1 + choose boyCount 2 * choose girlCount 2

theorem promotional_activity_choices :
  chooseVolunteers 6 4 2 4 = 14 := by sorry

end NUMINAMATH_CALUDE_promotional_activity_choices_l2336_233660


namespace NUMINAMATH_CALUDE_golden_ratio_cubic_l2336_233635

theorem golden_ratio_cubic (p q : ℚ) : 
  let x : ℝ := (Real.sqrt 5 - 1) / 2
  (x^3 + p * x + q = 0) → (p + q = -1) := by
  sorry

end NUMINAMATH_CALUDE_golden_ratio_cubic_l2336_233635


namespace NUMINAMATH_CALUDE_roxanne_change_l2336_233633

def lemonade_price : ℚ := 2
def sandwich_price : ℚ := 2.5
def watermelon_price : ℚ := 1.25
def chips_price : ℚ := 1.75
def cookie_price : ℚ := 0.75

def num_lemonade : ℕ := 2
def num_sandwich : ℕ := 2
def num_watermelon : ℕ := 1
def num_chips : ℕ := 1
def num_cookies : ℕ := 3

def payment : ℚ := 50

theorem roxanne_change :
  payment - (num_lemonade * lemonade_price +
             num_sandwich * sandwich_price +
             num_watermelon * watermelon_price +
             num_chips * chips_price +
             num_cookies * cookie_price) = 35.75 := by
  sorry

end NUMINAMATH_CALUDE_roxanne_change_l2336_233633


namespace NUMINAMATH_CALUDE_factor_theorem_l2336_233618

theorem factor_theorem (h k : ℝ) : 
  (∃ c : ℝ, 3 * x^3 - h * x + k = c * (x + 3) * (x - 2)) →
  |3 * h - 2 * k| = 27 := by
sorry

end NUMINAMATH_CALUDE_factor_theorem_l2336_233618


namespace NUMINAMATH_CALUDE_point_not_on_graph_l2336_233654

def inverse_proportion (x y : ℝ) : Prop := x * y = 6

theorem point_not_on_graph :
  ¬(inverse_proportion 1 5) ∧ 
  (inverse_proportion (-2) (-3)) ∧ 
  (inverse_proportion (-3) (-2)) ∧ 
  (inverse_proportion 4 1.5) :=
by sorry

end NUMINAMATH_CALUDE_point_not_on_graph_l2336_233654


namespace NUMINAMATH_CALUDE_gumball_probability_l2336_233696

/-- Given a jar with pink and blue gumballs, if the probability of drawing two blue
    gumballs in a row with replacement is 16/36, then the probability of drawing
    a pink gumball is 1/3. -/
theorem gumball_probability (p_blue p_pink : ℝ) : 
  p_blue + p_pink = 1 →
  p_blue ^ 2 = 16 / 36 →
  p_pink = 1 / 3 :=
by sorry

end NUMINAMATH_CALUDE_gumball_probability_l2336_233696


namespace NUMINAMATH_CALUDE_complement_A_union_B_when_m_4_range_of_m_for_B_subset_A_l2336_233693

-- Define the sets A and B
def A : Set ℝ := {x | -2 ≤ x ∧ x ≤ 5}
def B (m : ℝ) : Set ℝ := {x | m + 1 ≤ x ∧ x ≤ 2*m - 1}

-- Theorem for part I
theorem complement_A_union_B_when_m_4 :
  (Set.univ : Set ℝ) \ (A ∪ B 4) = {x | x < -2 ∨ x > 7} := by sorry

-- Theorem for part II
theorem range_of_m_for_B_subset_A :
  {m : ℝ | (B m).Nonempty ∧ B m ⊆ A} = {m | 2 ≤ m ∧ m ≤ 3} := by sorry

end NUMINAMATH_CALUDE_complement_A_union_B_when_m_4_range_of_m_for_B_subset_A_l2336_233693


namespace NUMINAMATH_CALUDE_jacket_cost_ratio_l2336_233663

theorem jacket_cost_ratio (marked_price : ℝ) (h1 : marked_price > 0) : 
  let discount_rate : ℝ := 1/4
  let selling_price : ℝ := marked_price * (1 - discount_rate)
  let cost_rate : ℝ := 5/8
  let cost : ℝ := selling_price * cost_rate
  cost / marked_price = 15/32 := by
sorry

end NUMINAMATH_CALUDE_jacket_cost_ratio_l2336_233663


namespace NUMINAMATH_CALUDE_moon_speed_km_per_hour_l2336_233629

/-- The number of seconds in an hour -/
def seconds_per_hour : ℕ := 3600

/-- The moon's speed in kilometers per second -/
def moon_speed_km_per_sec : ℚ := 103/100

/-- Converts speed from km/s to km/h -/
def convert_km_per_sec_to_km_per_hour (speed_km_per_sec : ℚ) : ℚ :=
  speed_km_per_sec * seconds_per_hour

theorem moon_speed_km_per_hour :
  convert_km_per_sec_to_km_per_hour moon_speed_km_per_sec = 3708 := by
  sorry

end NUMINAMATH_CALUDE_moon_speed_km_per_hour_l2336_233629


namespace NUMINAMATH_CALUDE_congruence_theorem_l2336_233697

theorem congruence_theorem (x : ℤ) 
  (h1 : (8 + x) % 8 = 27 % 8)
  (h2 : (10 + x) % 27 = 16 % 27)
  (h3 : (13 + x) % 125 = 36 % 125) :
  x % 120 = 11 := by
sorry

end NUMINAMATH_CALUDE_congruence_theorem_l2336_233697


namespace NUMINAMATH_CALUDE_polar_to_cartesian_equivalence_l2336_233689

/-- Prove that the polar equation ρ²cos(2θ) = 16 is equivalent to the Cartesian equation x² - y² = 16 -/
theorem polar_to_cartesian_equivalence (ρ θ x y : ℝ) 
  (h1 : x = ρ * Real.cos θ) 
  (h2 : y = ρ * Real.sin θ) : 
  ρ^2 * Real.cos (2 * θ) = 16 ↔ x^2 - y^2 = 16 := by
  sorry

end NUMINAMATH_CALUDE_polar_to_cartesian_equivalence_l2336_233689


namespace NUMINAMATH_CALUDE_clara_age_multiple_of_anna_l2336_233656

theorem clara_age_multiple_of_anna (anna_current_age clara_current_age : ℕ) 
  (h1 : anna_current_age = 54)
  (h2 : clara_current_age = 80) :
  ∃ (years_ago : ℕ), 
    clara_current_age - years_ago = 3 * (anna_current_age - years_ago) ∧ 
    years_ago = 41 := by
  sorry

end NUMINAMATH_CALUDE_clara_age_multiple_of_anna_l2336_233656


namespace NUMINAMATH_CALUDE_square_perimeter_problem_l2336_233653

theorem square_perimeter_problem :
  ∀ (a b c : ℝ),
  (4 * a = 16) →  -- Perimeter of square A is 16
  (4 * b = 32) →  -- Perimeter of square B is 32
  (c = 4 * (b - a)) →  -- Side length of C is 4 times the difference of A and B's side lengths
  (4 * c = 64) :=  -- Perimeter of square C is 64
by
  sorry

end NUMINAMATH_CALUDE_square_perimeter_problem_l2336_233653


namespace NUMINAMATH_CALUDE_fraction_problem_l2336_233651

theorem fraction_problem (N : ℝ) (f : ℝ) 
  (h1 : (1 / 3) * f * N = 15) 
  (h2 : (3 / 10) * N = 54) : 
  f = 1 / 4 := by
sorry

end NUMINAMATH_CALUDE_fraction_problem_l2336_233651


namespace NUMINAMATH_CALUDE_sum_right_angles_rectangle_square_l2336_233610

-- Define a rectangle
def Rectangle := Nat

-- Define a square
def Square := Nat

-- Define the number of right angles in a rectangle
def right_angles_rectangle (r : Rectangle) : Nat := 4

-- Define the number of right angles in a square
def right_angles_square (s : Square) : Nat := 4

-- Theorem: The sum of right angles in a rectangle and a square is 8
theorem sum_right_angles_rectangle_square (r : Rectangle) (s : Square) :
  right_angles_rectangle r + right_angles_square s = 8 := by
  sorry

end NUMINAMATH_CALUDE_sum_right_angles_rectangle_square_l2336_233610


namespace NUMINAMATH_CALUDE_cos_150_degrees_l2336_233605

theorem cos_150_degrees : Real.cos (150 * Real.pi / 180) = -Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_cos_150_degrees_l2336_233605


namespace NUMINAMATH_CALUDE_hash_2_5_3_equals_1_l2336_233684

-- Define the # operation
def hash (a b c : ℝ) : ℝ := b^2 - 4*a*c

-- Theorem statement
theorem hash_2_5_3_equals_1 : hash 2 5 3 = 1 := by sorry

end NUMINAMATH_CALUDE_hash_2_5_3_equals_1_l2336_233684


namespace NUMINAMATH_CALUDE_necessary_but_not_sufficient_l2336_233670

theorem necessary_but_not_sufficient (p q : Prop) :
  ((p ∧ q) → (p ∨ q)) ∧ ¬((p ∨ q) → (p ∧ q)) :=
sorry

end NUMINAMATH_CALUDE_necessary_but_not_sufficient_l2336_233670


namespace NUMINAMATH_CALUDE_abs_neg_reciprocal_2023_l2336_233627

theorem abs_neg_reciprocal_2023 : |-1 / 2023| = 1 / 2023 := by
  sorry

end NUMINAMATH_CALUDE_abs_neg_reciprocal_2023_l2336_233627


namespace NUMINAMATH_CALUDE_sixth_group_frequency_l2336_233680

/-- Given a sample of 40 data points divided into 6 groups, with the frequencies
    of the first four groups and the fifth group as specified, 
    the frequency of the sixth group is 0.2. -/
theorem sixth_group_frequency 
  (total_points : ℕ) 
  (group_count : ℕ)
  (freq_1 freq_2 freq_3 freq_4 freq_5 : ℚ) :
  total_points = 40 →
  group_count = 6 →
  freq_1 = 10 / 40 →
  freq_2 = 5 / 40 →
  freq_3 = 7 / 40 →
  freq_4 = 6 / 40 →
  freq_5 = 1 / 10 →
  ∃ freq_6 : ℚ, freq_6 = 1 - (freq_1 + freq_2 + freq_3 + freq_4 + freq_5) ∧ freq_6 = 1 / 5 :=
by sorry

end NUMINAMATH_CALUDE_sixth_group_frequency_l2336_233680


namespace NUMINAMATH_CALUDE_bottle_weight_difference_l2336_233646

/-- The weight difference between a glass bottle and a plastic bottle -/
def weight_difference : ℝ := by sorry

theorem bottle_weight_difference :
  let glass_bottle_weight : ℝ := 600 / 3
  let plastic_bottle_weight : ℝ := (1050 - 4 * glass_bottle_weight) / 5
  weight_difference = glass_bottle_weight - plastic_bottle_weight :=
by sorry

end NUMINAMATH_CALUDE_bottle_weight_difference_l2336_233646


namespace NUMINAMATH_CALUDE_fridays_in_non_leap_year_starting_saturday_l2336_233601

/-- Represents a day of the week -/
inductive DayOfWeek
| Sunday
| Monday
| Tuesday
| Wednesday
| Thursday
| Friday
| Saturday

/-- Represents a year -/
structure Year where
  isLeapYear : Bool
  firstDayOfYear : DayOfWeek

/-- Counts the number of occurrences of a specific day in a year -/
def countDaysInYear (y : Year) (d : DayOfWeek) : Nat :=
  sorry

/-- Theorem: In a non-leap year where January 1st is a Saturday, there are 52 Fridays -/
theorem fridays_in_non_leap_year_starting_saturday (y : Year) 
  (h1 : y.isLeapYear = false) 
  (h2 : y.firstDayOfYear = DayOfWeek.Saturday) : 
  countDaysInYear y DayOfWeek.Friday = 52 :=
by sorry

end NUMINAMATH_CALUDE_fridays_in_non_leap_year_starting_saturday_l2336_233601


namespace NUMINAMATH_CALUDE_range_of_a_l2336_233686

-- Define the system of inequalities
def inequality_system (x a : ℝ) : Prop :=
  3 * x - a > x + 1 ∧ (3 * x - 2) / 2 < 1 + x

-- Define the condition of having exactly 3 integer solutions
def has_three_integer_solutions (a : ℝ) : Prop :=
  ∃! (s : Finset ℤ), s.card = 3 ∧ ∀ x ∈ s, inequality_system x a

-- The main theorem
theorem range_of_a (a : ℝ) :
  has_three_integer_solutions a → -1 ≤ a ∧ a < 1 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l2336_233686


namespace NUMINAMATH_CALUDE_college_student_count_l2336_233608

/-- Given a college with a ratio of boys to girls of 8:5 and 160 girls, 
    the total number of students is 416. -/
theorem college_student_count 
  (ratio_boys : ℕ) 
  (ratio_girls : ℕ) 
  (num_girls : ℕ) 
  (h_ratio : ratio_boys = 8 ∧ ratio_girls = 5)
  (h_girls : num_girls = 160) : 
  (ratio_boys * num_girls / ratio_girls + num_girls : ℕ) = 416 := by
  sorry

#check college_student_count

end NUMINAMATH_CALUDE_college_student_count_l2336_233608
