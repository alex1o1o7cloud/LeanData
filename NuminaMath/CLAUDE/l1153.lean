import Mathlib

namespace NUMINAMATH_CALUDE_mama_bird_worms_l1153_115322

/-- The number of additional worms Mama bird needs to catch -/
def additional_worms_needed (num_babies : ℕ) (worms_per_baby_per_day : ℕ) (days : ℕ) 
  (papa_worms : ℕ) (mama_worms : ℕ) (stolen_worms : ℕ) : ℕ :=
  num_babies * worms_per_baby_per_day * days - (papa_worms + mama_worms - stolen_worms)

/-- Theorem stating that Mama bird needs to catch 34 more worms -/
theorem mama_bird_worms : 
  additional_worms_needed 6 3 3 9 13 2 = 34 := by sorry

end NUMINAMATH_CALUDE_mama_bird_worms_l1153_115322


namespace NUMINAMATH_CALUDE_solve_head_circumference_problem_l1153_115387

def head_circumference_problem (jack_circumference charlie_circumference bill_circumference : ℝ) : Prop :=
  jack_circumference = 12 ∧
  bill_circumference = 10 ∧
  bill_circumference = (2/3) * charlie_circumference ∧
  ∃ x, charlie_circumference = (1/2) * jack_circumference + x ∧
  x = 9

theorem solve_head_circumference_problem :
  ∀ jack_circumference charlie_circumference bill_circumference,
  head_circumference_problem jack_circumference charlie_circumference bill_circumference :=
by
  sorry

end NUMINAMATH_CALUDE_solve_head_circumference_problem_l1153_115387


namespace NUMINAMATH_CALUDE_line_slope_l1153_115334

theorem line_slope (A B : ℝ × ℝ) : 
  A.1 = 2 * Real.sqrt 3 ∧ A.2 = -1 ∧ B.1 = Real.sqrt 3 ∧ B.2 = 2 →
  (B.2 - A.2) / (B.1 - A.1) = -Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_line_slope_l1153_115334


namespace NUMINAMATH_CALUDE_right_triangle_squares_area_l1153_115335

theorem right_triangle_squares_area (a b c : ℝ) : 
  a > 0 → b > 0 → c > 0 →
  a^2 + b^2 = c^2 →
  c^2 = 1009 →
  4*a^2 + 4*b^2 + 4*c^2 = 8072 := by sorry

end NUMINAMATH_CALUDE_right_triangle_squares_area_l1153_115335


namespace NUMINAMATH_CALUDE_triangle_inequality_cube_l1153_115371

theorem triangle_inequality_cube (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (htriangle : a + b > c ∧ b + c > a ∧ c + a > b) : a^3 + b^3 + 3*a*b*c > c^3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_inequality_cube_l1153_115371


namespace NUMINAMATH_CALUDE_S_inter_T_finite_l1153_115374

/-- Set S defined as {y | y = 3^x, x ∈ ℝ} -/
def S : Set ℝ := {y | ∃ x, y = Real.exp (Real.log 3 * x)}

/-- Set T defined as {y | y = x^2 - 1, x ∈ ℝ} -/
def T : Set ℝ := {y | ∃ x, y = x^2 - 1}

/-- The intersection of S and T is a finite set -/
theorem S_inter_T_finite : Set.Finite (S ∩ T) := by sorry

end NUMINAMATH_CALUDE_S_inter_T_finite_l1153_115374


namespace NUMINAMATH_CALUDE_acquaintance_pigeonhole_l1153_115304

theorem acquaintance_pigeonhole (n : ℕ) (h : n ≥ 2) :
  ∃ (i j : Fin n), i ≠ j ∧ 
  ∃ (f : Fin n → Fin n), (∀ k, f k < n) ∧ f i = f j :=
by
  sorry

end NUMINAMATH_CALUDE_acquaintance_pigeonhole_l1153_115304


namespace NUMINAMATH_CALUDE_integer_in_range_l1153_115379

theorem integer_in_range : ∃ x : ℤ, -Real.sqrt 2 < x ∧ x < Real.sqrt 5 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_integer_in_range_l1153_115379


namespace NUMINAMATH_CALUDE_square_perimeter_sum_l1153_115314

theorem square_perimeter_sum (x y : ℝ) (h1 : x^2 + y^2 = 85) (h2 : x^2 - y^2 = 45) :
  4 * (x + y) = 4 * (Real.sqrt 65 + 2 * Real.sqrt 5) := by
  sorry

end NUMINAMATH_CALUDE_square_perimeter_sum_l1153_115314


namespace NUMINAMATH_CALUDE_buratino_bet_exists_pierrot_bet_impossible_papa_carlo_minimum_bet_karabas_barabas_impossible_l1153_115331

/-- Represents a bet on a horse race --/
structure HorseBet where
  first : ℕ
  second : ℕ
  third : ℕ

/-- Represents the odds for each horse --/
def odds : Fin 3 → ℚ
  | 0 => 4
  | 1 => 3
  | 2 => 1

/-- Calculates the total bet amount --/
def totalBet (bet : HorseBet) : ℕ :=
  bet.first + bet.second + bet.third

/-- Calculates the return for a given horse winning --/
def returnForHorse (bet : HorseBet) (horse : Fin 3) : ℚ :=
  match horse with
  | 0 => (bet.first : ℚ) * (odds 0 + 1)
  | 1 => (bet.second : ℚ) * (odds 1 + 1)
  | 2 => (bet.third : ℚ) * (odds 2 + 1)

/-- Checks if a bet guarantees a minimum return --/
def guaranteesReturn (bet : HorseBet) (minReturn : ℚ) : Prop :=
  ∀ horse : Fin 3, returnForHorse bet horse ≥ minReturn

theorem buratino_bet_exists :
  ∃ bet : HorseBet, totalBet bet = 50 ∧ guaranteesReturn bet 52 :=
sorry

theorem pierrot_bet_impossible :
  ¬∃ bet : HorseBet, totalBet bet = 25 ∧ guaranteesReturn bet 26 :=
sorry

theorem papa_carlo_minimum_bet :
  (∃ bet : HorseBet, guaranteesReturn bet ((totalBet bet : ℚ) + 5)) ∧
  (∀ s : ℕ, s < 95 → ¬∃ bet : HorseBet, totalBet bet = s ∧ guaranteesReturn bet ((s : ℚ) + 5)) :=
sorry

theorem karabas_barabas_impossible :
  ¬∃ bet : HorseBet, guaranteesReturn bet ((totalBet bet : ℚ) * (106 / 100)) :=
sorry

end NUMINAMATH_CALUDE_buratino_bet_exists_pierrot_bet_impossible_papa_carlo_minimum_bet_karabas_barabas_impossible_l1153_115331


namespace NUMINAMATH_CALUDE_square_sum_17_5_l1153_115363

theorem square_sum_17_5 : 17^2 + 2*(17*5) + 5^2 = 484 := by
  sorry

end NUMINAMATH_CALUDE_square_sum_17_5_l1153_115363


namespace NUMINAMATH_CALUDE_number_division_problem_l1153_115301

theorem number_division_problem :
  ∃ x : ℝ, (x / 5 = 60 + x / 6) ∧ (x = 1800) := by
  sorry

end NUMINAMATH_CALUDE_number_division_problem_l1153_115301


namespace NUMINAMATH_CALUDE_hexagon_diagonals_l1153_115352

/-- The number of internal diagonals in a polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- A hexagon is a polygon with 6 sides -/
def hexagon_sides : ℕ := 6

/-- Theorem: A hexagon has 9 internal diagonals -/
theorem hexagon_diagonals : num_diagonals hexagon_sides = 9 := by
  sorry

end NUMINAMATH_CALUDE_hexagon_diagonals_l1153_115352


namespace NUMINAMATH_CALUDE_lake_crossing_cost_l1153_115300

/-- The cost of crossing a lake back and forth -/
theorem lake_crossing_cost (crossing_time : ℕ) (assistant_cost : ℕ) : 
  crossing_time = 4 → assistant_cost = 10 → crossing_time * 2 * assistant_cost = 80 := by
  sorry

#check lake_crossing_cost

end NUMINAMATH_CALUDE_lake_crossing_cost_l1153_115300


namespace NUMINAMATH_CALUDE_books_found_equals_26_l1153_115340

/-- The number of books Joan initially gathered -/
def initial_books : ℕ := 33

/-- The total number of books Joan has now -/
def total_books : ℕ := 59

/-- The number of additional books Joan found -/
def additional_books : ℕ := total_books - initial_books

theorem books_found_equals_26 : additional_books = 26 := by
  sorry

end NUMINAMATH_CALUDE_books_found_equals_26_l1153_115340


namespace NUMINAMATH_CALUDE_max_rational_products_l1153_115339

/-- Represents a table with rational and irrational numbers as labels -/
structure LabeledTable where
  size : ℕ
  rowLabels : Fin size → ℝ
  colLabels : Fin size → ℝ
  distinctLabels : ∀ i j, (rowLabels i = colLabels j) → i = j
  rationalCount : ℕ
  irrationalCount : ℕ
  labelCounts : rationalCount + irrationalCount = size + size

/-- Counts the number of rational products in the table -/
def countRationalProducts (t : LabeledTable) : ℕ :=
  sorry

/-- Theorem stating the maximum number of rational products -/
theorem max_rational_products (t : LabeledTable) : 
  t.size = 50 ∧ t.rationalCount = 50 ∧ t.irrationalCount = 50 → 
  countRationalProducts t ≤ 1275 :=
sorry

end NUMINAMATH_CALUDE_max_rational_products_l1153_115339


namespace NUMINAMATH_CALUDE_not_divisible_by_2310_l1153_115330

theorem not_divisible_by_2310 (n : ℕ) (h : n < 2310) : ¬(2310 ∣ n * (2310 - n)) := by
  sorry

end NUMINAMATH_CALUDE_not_divisible_by_2310_l1153_115330


namespace NUMINAMATH_CALUDE_sum_of_two_numbers_l1153_115328

theorem sum_of_two_numbers (x y : ℤ) : x + y = 32 ∧ y = -36 → x = 68 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_two_numbers_l1153_115328


namespace NUMINAMATH_CALUDE_simplify_expression_l1153_115307

theorem simplify_expression : (1 : ℝ) / (1 + Real.sqrt 3) * (1 / (1 + Real.sqrt 3)) = 1 - Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1153_115307


namespace NUMINAMATH_CALUDE_product_from_lcm_gcd_l1153_115381

theorem product_from_lcm_gcd : 
  ∀ a b : ℤ, (Nat.lcm a.natAbs b.natAbs = 72) → (Int.gcd a b = 8) → a * b = 576 := by
  sorry

end NUMINAMATH_CALUDE_product_from_lcm_gcd_l1153_115381


namespace NUMINAMATH_CALUDE_quadratic_distinct_roots_l1153_115353

theorem quadratic_distinct_roots (p q : ℚ) : 
  (∃ x y : ℚ, x ≠ y ∧ 
    x^2 + p*x + q = 0 ∧ 
    y^2 + p*y + q = 0 ∧ 
    x = 2*p ∧ 
    y = p + q) → 
  (p = 2/3 ∧ q = -8/3) :=
sorry

end NUMINAMATH_CALUDE_quadratic_distinct_roots_l1153_115353


namespace NUMINAMATH_CALUDE_star_equation_solution_l1153_115359

/-- The star operation -/
def star (a b : ℝ) : ℝ := a * b + 3 * b - a - 2

/-- Theorem: If 3 ★ y = 25, then y = 5 -/
theorem star_equation_solution (y : ℝ) (h : star 3 y = 25) : y = 5 := by
  sorry

end NUMINAMATH_CALUDE_star_equation_solution_l1153_115359


namespace NUMINAMATH_CALUDE_son_age_proof_l1153_115372

-- Define the variables
def your_age : ℕ := 45
def son_age : ℕ := 15

-- Define the conditions
theorem son_age_proof :
  (your_age = 3 * son_age) ∧
  (your_age + 5 = (5/2) * (son_age + 5)) →
  son_age = 15 := by
sorry


end NUMINAMATH_CALUDE_son_age_proof_l1153_115372


namespace NUMINAMATH_CALUDE_correct_average_l1153_115347

theorem correct_average (n : ℕ) (initial_avg : ℚ) 
  (correct_numbers incorrect_numbers : List ℚ) :
  n = 15 ∧ 
  initial_avg = 25 ∧ 
  correct_numbers = [86, 92, 48] ∧ 
  incorrect_numbers = [26, 62, 24] →
  (n * initial_avg + (correct_numbers.sum - incorrect_numbers.sum)) / n = 32.6 := by
  sorry

end NUMINAMATH_CALUDE_correct_average_l1153_115347


namespace NUMINAMATH_CALUDE_tom_distance_before_karen_wins_l1153_115361

/-- Represents the race between Karen and Tom -/
structure Race where
  karen_initial_speed : ℝ
  tom_initial_speed : ℝ
  karen_final_speed : ℝ
  tom_final_speed : ℝ
  karen_delay : ℝ
  winning_margin : ℝ

/-- Calculates the distance Tom drives before Karen wins the bet -/
def distance_tom_drives (race : Race) : ℝ :=
  sorry

/-- Theorem stating that Tom drives 21 miles before Karen wins the bet -/
theorem tom_distance_before_karen_wins (race : Race) 
  (h1 : race.karen_initial_speed = 60)
  (h2 : race.tom_initial_speed = 45)
  (h3 : race.karen_final_speed = 70)
  (h4 : race.tom_final_speed = 40)
  (h5 : race.karen_delay = 4/60)  -- 4 minutes converted to hours
  (h6 : race.winning_margin = 4) :
  distance_tom_drives race = 21 :=
sorry

end NUMINAMATH_CALUDE_tom_distance_before_karen_wins_l1153_115361


namespace NUMINAMATH_CALUDE_log_sum_greater_than_exp_l1153_115305

theorem log_sum_greater_than_exp (x : ℝ) (h : x < 0) :
  Real.log 2 + Real.log 5 > Real.exp x := by sorry

end NUMINAMATH_CALUDE_log_sum_greater_than_exp_l1153_115305


namespace NUMINAMATH_CALUDE_negative_half_less_than_negative_third_l1153_115396

theorem negative_half_less_than_negative_third : -1/2 < -1/3 := by
  sorry

end NUMINAMATH_CALUDE_negative_half_less_than_negative_third_l1153_115396


namespace NUMINAMATH_CALUDE_always_judge_available_l1153_115360

/-- Represents a tennis tournament in a sports club -/
structure TennisTournament where
  n : ℕ  -- number of matches played so far
  eliminated : ℕ  -- number of eliminated players
  judges : ℕ  -- number of players who have judged a match

/-- The state of the tournament after n matches -/
def tournamentState (n : ℕ) : TennisTournament :=
  { n := n
  , eliminated := n  -- each match eliminates one player
  , judges := if n = 0 then 0 else n - 1 }  -- judges needed for all but the first match

/-- There is always someone available to judge the next match -/
theorem always_judge_available (n : ℕ) :
  let t := tournamentState n
  t.eliminated > t.judges :=
by sorry

end NUMINAMATH_CALUDE_always_judge_available_l1153_115360


namespace NUMINAMATH_CALUDE_probability_red_from_B_probability_red_from_B_is_correct_l1153_115342

/-- Represents the number of red balls in Box A -/
def red_balls_A : ℕ := 5

/-- Represents the number of white balls in Box A -/
def white_balls_A : ℕ := 2

/-- Represents the number of red balls in Box B -/
def red_balls_B : ℕ := 4

/-- Represents the number of white balls in Box B -/
def white_balls_B : ℕ := 3

/-- Represents the total number of balls in Box A -/
def total_balls_A : ℕ := red_balls_A + white_balls_A

/-- Represents the total number of balls in Box B -/
def total_balls_B : ℕ := red_balls_B + white_balls_B

/-- The probability of drawing a red ball from Box B after the process -/
theorem probability_red_from_B : ℚ :=
  33 / 56

theorem probability_red_from_B_is_correct :
  probability_red_from_B = 33 / 56 := by sorry

end NUMINAMATH_CALUDE_probability_red_from_B_probability_red_from_B_is_correct_l1153_115342


namespace NUMINAMATH_CALUDE_square_perimeter_problem_l1153_115343

/-- Given squares I, II, and III, prove that the perimeter of III is 16√2 + 32 -/
theorem square_perimeter_problem (I II III : ℝ → ℝ → Prop) : 
  (∀ x, I x x → 4 * x = 16) →  -- Square I has perimeter 16
  (∀ y, II y y → 4 * y = 32) →  -- Square II has perimeter 32
  (∀ x y z, I x x → II y y → III z z → z = x * Real.sqrt 2 + y) →  -- Side of III is diagonal of I plus side of II
  (∃ z, III z z ∧ 4 * z = 16 * Real.sqrt 2 + 32) :=  -- Perimeter of III is 16√2 + 32
by sorry

end NUMINAMATH_CALUDE_square_perimeter_problem_l1153_115343


namespace NUMINAMATH_CALUDE_average_mark_calculation_l1153_115309

theorem average_mark_calculation (students_class1 students_class2 : ℕ) 
  (avg_class2 avg_total : ℚ) : 
  students_class1 = 20 →
  students_class2 = 50 →
  avg_class2 = 60 →
  avg_total = 54.285714285714285 →
  (students_class1 * (avg_total * (students_class1 + students_class2) - students_class2 * avg_class2)) / 
   (students_class1 * (students_class1 + students_class2)) = 40 := by
  sorry

end NUMINAMATH_CALUDE_average_mark_calculation_l1153_115309


namespace NUMINAMATH_CALUDE_equilateral_hyperbola_through_point_l1153_115308

/-- An equilateral hyperbola is a hyperbola with perpendicular asymptotes -/
def is_equilateral_hyperbola (f : ℝ → ℝ → ℝ) : Prop :=
  ∃ a b c : ℝ, a ≠ 0 ∧ f = λ x y => a * (x^2 - y^2) + b * x + c * y + 1

/-- A point (x, y) lies on a curve defined by function f if f x y = 0 -/
def point_on_curve (f : ℝ → ℝ → ℝ) (x y : ℝ) : Prop :=
  f x y = 0

/-- A curve is symmetric about the x-axis if for every point (x, y) on the curve,
    the point (x, -y) is also on the curve -/
def symmetric_about_x_axis (f : ℝ → ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f x y = 0 → f x (-y) = 0

/-- A curve is symmetric about the y-axis if for every point (x, y) on the curve,
    the point (-x, y) is also on the curve -/
def symmetric_about_y_axis (f : ℝ → ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f x y = 0 → f (-x) y = 0

theorem equilateral_hyperbola_through_point :
  ∃ f : ℝ → ℝ → ℝ,
    is_equilateral_hyperbola f ∧
    point_on_curve f 3 (-1) ∧
    symmetric_about_x_axis f ∧
    symmetric_about_y_axis f ∧
    f = λ x y => x^2 - y^2 - 8 := by sorry

end NUMINAMATH_CALUDE_equilateral_hyperbola_through_point_l1153_115308


namespace NUMINAMATH_CALUDE_inscribed_quadrilateral_symmetry_l1153_115377

/-- A circle in which the quadrilateral is inscribed -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- A point in the 2D plane -/
def Point := ℝ × ℝ

/-- A line in the 2D plane -/
structure Line where
  point1 : Point
  point2 : Point

/-- A quadrilateral inscribed in a circle -/
structure InscribedQuadrilateral where
  circle : Circle
  A : Point
  B : Point
  C : Point
  D : Point

/-- Intersection point of two lines -/
def intersectionPoint (l1 l2 : Line) : Point := sorry

/-- Check if a point is on a line -/
def isPointOnLine (p : Point) (l : Line) : Prop := sorry

/-- Check if two points are symmetrical with respect to a third point -/
def areSymmetrical (p1 p2 center : Point) : Prop := sorry

/-- The main theorem -/
theorem inscribed_quadrilateral_symmetry 
  (quad : InscribedQuadrilateral)
  (E : Point)
  (t : Line) :
  let AB := Line.mk quad.A quad.B
  let CD := Line.mk quad.C quad.D
  let AC := Line.mk quad.A quad.C
  let BD := Line.mk quad.B quad.D
  let BC := Line.mk quad.B quad.C
  let AD := Line.mk quad.A quad.D
  let O := quad.circle.center
  E = intersectionPoint AB CD →
  isPointOnLine E t →
  (∀ p : Point, isPointOnLine p (Line.mk O E) → isPointOnLine p t → p = E) →
  ∃ (P Q R S : Point),
    P = intersectionPoint AC t ∧
    Q = intersectionPoint BD t ∧
    R = intersectionPoint BC t ∧
    S = intersectionPoint AD t ∧
    areSymmetrical P Q E ∧
    areSymmetrical R S E :=
sorry

end NUMINAMATH_CALUDE_inscribed_quadrilateral_symmetry_l1153_115377


namespace NUMINAMATH_CALUDE_age_ratio_is_eleven_eighths_l1153_115348

/-- Represents the ages and relationships of Rehana, Phoebe, Jacob, and Xander -/
structure AgeGroup where
  rehana_age : ℕ
  phoebe_age : ℕ
  jacob_age : ℕ
  xander_age : ℕ

/-- Conditions for the age group -/
def valid_age_group (ag : AgeGroup) : Prop :=
  ag.rehana_age = 25 ∧
  ag.rehana_age + 5 = 3 * (ag.phoebe_age + 5) ∧
  ag.jacob_age = (3 * ag.phoebe_age) / 5 ∧
  ag.xander_age = ag.rehana_age + ag.jacob_age - 4

/-- The ratio of combined ages to Xander's age -/
def age_ratio (ag : AgeGroup) : ℚ :=
  (ag.rehana_age + ag.phoebe_age + ag.jacob_age : ℚ) / ag.xander_age

/-- Theorem stating the age ratio is 11/8 for a valid age group -/
theorem age_ratio_is_eleven_eighths (ag : AgeGroup) (h : valid_age_group ag) :
  age_ratio ag = 11/8 := by
  sorry

end NUMINAMATH_CALUDE_age_ratio_is_eleven_eighths_l1153_115348


namespace NUMINAMATH_CALUDE_closest_to_140_l1153_115320

def options : List ℝ := [120, 140, 160, 180, 200]

def expression : ℝ := 3.52 * 7.861 * (6.28 - 1.283)

theorem closest_to_140 : 
  ∀ x ∈ options, |expression - 140| ≤ |expression - x| := by
  sorry

end NUMINAMATH_CALUDE_closest_to_140_l1153_115320


namespace NUMINAMATH_CALUDE_gcd_324_135_l1153_115306

theorem gcd_324_135 : Nat.gcd 324 135 = 27 := by
  sorry

end NUMINAMATH_CALUDE_gcd_324_135_l1153_115306


namespace NUMINAMATH_CALUDE_prop_false_implies_a_lt_neg_13_div_2_l1153_115337

theorem prop_false_implies_a_lt_neg_13_div_2 (a : ℝ) :
  (¬ ∀ x ∈ Set.Icc 1 2, x^2 + a*x + 9 ≥ 0) → a < -13/2 := by
  sorry

end NUMINAMATH_CALUDE_prop_false_implies_a_lt_neg_13_div_2_l1153_115337


namespace NUMINAMATH_CALUDE_sum_and_reciprocal_sum_l1153_115370

theorem sum_and_reciprocal_sum (x : ℝ) (h : x > 0) (h_sum_squares : x^2 + (1/x)^2 = 23) : 
  x + (1/x) = 5 := by sorry

end NUMINAMATH_CALUDE_sum_and_reciprocal_sum_l1153_115370


namespace NUMINAMATH_CALUDE_mikes_ride_distance_l1153_115364

theorem mikes_ride_distance (mike_start_fee annie_start_fee : ℚ)
  (annie_bridge_toll : ℚ) (cost_per_mile : ℚ) (annie_distance : ℚ)
  (h1 : mike_start_fee = 2.5)
  (h2 : annie_start_fee = 2.5)
  (h3 : annie_bridge_toll = 5)
  (h4 : cost_per_mile = 0.25)
  (h5 : annie_distance = 22)
  (h6 : ∃ (mike_distance : ℚ),
    mike_start_fee + cost_per_mile * mike_distance =
    annie_start_fee + annie_bridge_toll + cost_per_mile * annie_distance) :
  ∃ (mike_distance : ℚ), mike_distance = 32 :=
by sorry

end NUMINAMATH_CALUDE_mikes_ride_distance_l1153_115364


namespace NUMINAMATH_CALUDE_necessary_but_not_sufficient_l1153_115316

theorem necessary_but_not_sufficient (a b : ℝ) :
  (((1 / b) < (1 / a) ∧ (1 / a) < 0) → a < b) ∧
  (∃ a b, a < b ∧ ¬((1 / b) < (1 / a) ∧ (1 / a) < 0)) :=
by sorry

end NUMINAMATH_CALUDE_necessary_but_not_sufficient_l1153_115316


namespace NUMINAMATH_CALUDE_domain_of_composite_function_l1153_115399

-- Define the function f with domain (1,3)
def f : Set ℝ := Set.Ioo 1 3

-- Define the composite function g(x) = f(2x-1)
def g (x : ℝ) : ℝ := 2 * x - 1

-- Theorem statement
theorem domain_of_composite_function :
  {x : ℝ | g x ∈ f} = Set.Ioo 1 2 := by sorry

end NUMINAMATH_CALUDE_domain_of_composite_function_l1153_115399


namespace NUMINAMATH_CALUDE_meat_spending_fraction_l1153_115311

/-- Represents John's spending at the supermarket -/
structure SupermarketSpending where
  total : ℝ
  fruitVeg : ℝ
  bakery : ℝ
  candy : ℝ
  meat : ℝ

/-- Theorem stating the fraction spent on meat products -/
theorem meat_spending_fraction (s : SupermarketSpending) 
  (h1 : s.total = 30)
  (h2 : s.fruitVeg = s.total / 5)
  (h3 : s.bakery = s.total / 10)
  (h4 : s.candy = 11)
  (h5 : s.total = s.fruitVeg + s.bakery + s.meat + s.candy) :
  s.meat / s.total = 8 / 15 := by
  sorry

end NUMINAMATH_CALUDE_meat_spending_fraction_l1153_115311


namespace NUMINAMATH_CALUDE_park_conditions_l1153_115369

-- Define the conditions
def temperature_at_least_75 : Prop := sorry
def sunny : Prop := sorry
def park_clean : Prop := sorry
def park_crowded : Prop := sorry

-- Define the main theorem
theorem park_conditions :
  (temperature_at_least_75 ∧ sunny ∧ park_clean → park_crowded) →
  (¬park_crowded → ¬temperature_at_least_75 ∨ ¬sunny ∨ ¬park_clean) :=
by sorry

end NUMINAMATH_CALUDE_park_conditions_l1153_115369


namespace NUMINAMATH_CALUDE_mike_total_spent_l1153_115346

def marbles_cost : ℝ := 9.05
def football_cost : ℝ := 4.95
def baseball_cost : ℝ := 6.52
def toy_car_cost : ℝ := 3.75
def puzzle_cost : ℝ := 8.99
def stickers_cost : ℝ := 1.25
def puzzle_discount : ℝ := 0.15
def toy_car_discount : ℝ := 0.10
def coupon_value : ℝ := 5.00

def discounted_puzzle_cost : ℝ := puzzle_cost * (1 - puzzle_discount)
def discounted_toy_car_cost : ℝ := toy_car_cost * (1 - toy_car_discount)

def total_cost : ℝ := marbles_cost + football_cost + baseball_cost + 
                       discounted_toy_car_cost + discounted_puzzle_cost + 
                       stickers_cost - coupon_value

theorem mike_total_spent :
  total_cost = 27.7865 :=
by sorry

end NUMINAMATH_CALUDE_mike_total_spent_l1153_115346


namespace NUMINAMATH_CALUDE_num_lines_in_4x4_grid_l1153_115382

/-- Represents a 4-by-4 grid of lattice points -/
structure Grid :=
  (size : Nat)
  (h_size : size = 4)

/-- Represents a line in the grid -/
structure Line :=
  (points : Finset (Nat × Nat))
  (h_distinct : points.card ≥ 2)
  (h_in_grid : ∀ p ∈ points, p.1 < 4 ∧ p.2 < 4)

/-- The set of all lines in the grid -/
def allLines (g : Grid) : Finset Line := sorry

/-- The number of distinct lines passing through at least two points in a 4-by-4 grid -/
def numLines (g : Grid) : Nat :=
  (allLines g).card

/-- Theorem stating that the number of distinct lines in a 4-by-4 grid is 70 -/
theorem num_lines_in_4x4_grid (g : Grid) : numLines g = 70 := by
  sorry

end NUMINAMATH_CALUDE_num_lines_in_4x4_grid_l1153_115382


namespace NUMINAMATH_CALUDE_apple_bags_theorem_l1153_115338

/-- Represents the possible number of apples in a bag -/
inductive BagSize
| small : BagSize  -- 6 apples
| large : BagSize  -- 12 apples

/-- Returns true if the given number is a valid total number of apples -/
def is_valid_total (n : ℕ) : Prop :=
  70 ≤ n ∧ n ≤ 80 ∧ ∃ (small large : ℕ), n = 6 * small + 12 * large

theorem apple_bags_theorem :
  ∀ n : ℕ, is_valid_total n ↔ (n = 72 ∨ n = 78) :=
sorry

end NUMINAMATH_CALUDE_apple_bags_theorem_l1153_115338


namespace NUMINAMATH_CALUDE_shaded_region_value_l1153_115336

/-- Rectangle PQRS with PS = 2 and PQ = 4 -/
structure Rectangle where
  ps : ℝ
  pq : ℝ
  h_ps : ps = 2
  h_pq : pq = 4

/-- Points T, U, V, W positioned so that RT = RU = PW = PV = a -/
def points_position (rect : Rectangle) (a : ℝ) : Prop :=
  ∃ (t u v w : ℝ × ℝ), 
    (rect.pq - a = t.1) ∧ (rect.pq - a = u.1) ∧ (a = v.1) ∧ (a = w.1) ∧
    (rect.ps = t.2) ∧ (0 = u.2) ∧ (rect.ps = v.2) ∧ (0 = w.2)

/-- VU and WT pass through the center of the rectangle -/
def lines_through_center (rect : Rectangle) (a : ℝ) : Prop :=
  ∃ (center : ℝ × ℝ), center = (rect.pq / 2, rect.ps / 2)

/-- The shaded region is 1/8 the area of PQRS -/
def shaded_region_ratio (rect : Rectangle) (a : ℝ) : Prop :=
  3 * a = 1/8 * (rect.ps * rect.pq)

/-- Main theorem -/
theorem shaded_region_value (rect : Rectangle) :
  points_position rect (1/3) ∧ 
  lines_through_center rect (1/3) ∧ 
  shaded_region_ratio rect (1/3) := by
  sorry

end NUMINAMATH_CALUDE_shaded_region_value_l1153_115336


namespace NUMINAMATH_CALUDE_line_equation_l1153_115303

/-- A line passing through the point (2, 3) with opposite intercepts on the coordinate axes -/
structure LineWithOppositeIntercepts where
  -- The slope-intercept form of the line: y = mx + b
  m : ℝ
  b : ℝ
  -- The line passes through (2, 3)
  point_condition : 3 = m * 2 + b
  -- The line has opposite intercepts on the axes
  opposite_intercepts : ∃ (k : ℝ), k ≠ 0 ∧ (b = k ∨ b = -k) ∧ (b / m = -k ∨ b / m = k)

/-- The equation of the line is either x - y + 1 = 0 or 3x - 2y = 0 -/
theorem line_equation (l : LineWithOppositeIntercepts) :
  (l.m = 1 ∧ l.b = -1) ∨ (l.m = 3/2 ∧ l.b = 0) :=
sorry

end NUMINAMATH_CALUDE_line_equation_l1153_115303


namespace NUMINAMATH_CALUDE_problem_1_l1153_115345

theorem problem_1 : 
  Real.sqrt ((-2)^2) + Real.sqrt 2 * (1 - Real.sqrt (1/2)) + |(-Real.sqrt 8)| = 1 + 3 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_problem_1_l1153_115345


namespace NUMINAMATH_CALUDE_equation_solution_l1153_115344

theorem equation_solution : ∃ (x₁ x₂ : ℚ), 
  (x₁ = 1 ∧ x₂ = 2/3) ∧ 
  (∀ x : ℚ, 3*x*(x-1) = 2*x-2 ↔ (x = x₁ ∨ x = x₂)) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l1153_115344


namespace NUMINAMATH_CALUDE_circle_equation_proof_l1153_115397

-- Define the given circle
def given_circle (x y : ℝ) : Prop := x^2 + y^2 - x + 2*y - 3 = 0

-- Define the given line
def given_line (x y : ℝ) : Prop := 5*x + 2*y + 1 = 0

-- Define the point P
def point_P : ℝ × ℝ := (0, 2)

-- Define the sought circle
def sought_circle (x y : ℝ) : Prop := x^2 + y^2 - 6*x - 4 = 0

-- Theorem statement
theorem circle_equation_proof :
  ∃ (D E F : ℝ),
    (∀ x y : ℝ, x^2 + y^2 + D*x + E*y + F = 0 ↔ sought_circle x y) ∧
    sought_circle point_P.1 point_P.2 ∧
    (∀ x y : ℝ, given_circle x y ∧ sought_circle x y → given_line x y) :=
sorry

end NUMINAMATH_CALUDE_circle_equation_proof_l1153_115397


namespace NUMINAMATH_CALUDE_absolute_value_inequality_solution_set_l1153_115355

theorem absolute_value_inequality_solution_set :
  {x : ℝ | |x - 1| ≤ 2} = Set.Icc (-1) 3 := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_inequality_solution_set_l1153_115355


namespace NUMINAMATH_CALUDE_orthogonal_vectors_magnitude_l1153_115385

def vector_a : ℝ × ℝ := (1, -3)
def vector_b (m : ℝ) : ℝ × ℝ := (6, m)

theorem orthogonal_vectors_magnitude (m : ℝ) 
  (h : vector_a.1 * (vector_b m).1 + vector_a.2 * (vector_b m).2 = 0) : 
  ‖(2 • vector_a - vector_b m)‖ = 4 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_orthogonal_vectors_magnitude_l1153_115385


namespace NUMINAMATH_CALUDE_initial_books_eq_sold_plus_left_l1153_115313

/-- The number of books Paul had initially -/
def initial_books : ℕ := 136

/-- The number of books Paul sold -/
def books_sold : ℕ := 109

/-- The number of books Paul was left with after the sale -/
def books_left : ℕ := 27

/-- Theorem stating that the initial number of books is equal to the sum of books sold and books left -/
theorem initial_books_eq_sold_plus_left : initial_books = books_sold + books_left := by
  sorry

end NUMINAMATH_CALUDE_initial_books_eq_sold_plus_left_l1153_115313


namespace NUMINAMATH_CALUDE_product_of_first_three_odd_numbers_l1153_115373

theorem product_of_first_three_odd_numbers : 
  (∀ a b c : ℕ, a * b * c = 38 → a = 3 ∧ b = 5 ∧ c = 7) →
  (∀ x y z : ℕ, x * y * z = 268 → x = 13 ∧ y = 15 ∧ z = 17) →
  1 * 3 * 5 = 15 :=
by sorry

end NUMINAMATH_CALUDE_product_of_first_three_odd_numbers_l1153_115373


namespace NUMINAMATH_CALUDE_complement_of_A_in_U_l1153_115323

-- Define the universal set U as the real numbers
def U := ℝ

-- Define set A as the non-negative real numbers
def A := {x : ℝ | x ≥ 0}

-- State the theorem
theorem complement_of_A_in_U : Set.compl A = {x : ℝ | x < 0} := by
  sorry

end NUMINAMATH_CALUDE_complement_of_A_in_U_l1153_115323


namespace NUMINAMATH_CALUDE_watch_sale_gain_percentage_l1153_115389

/-- Prove the gain percentage for a watch sale --/
theorem watch_sale_gain_percentage 
  (cost_price : ℝ) 
  (initial_loss_percentage : ℝ) 
  (price_increase : ℝ) : 
  cost_price = 875 → 
  initial_loss_percentage = 12 → 
  price_increase = 140 → 
  let initial_selling_price := cost_price * (1 - initial_loss_percentage / 100)
  let new_selling_price := initial_selling_price + price_increase
  let gain := new_selling_price - cost_price
  let gain_percentage := (gain / cost_price) * 100
  gain_percentage = 4 := by
sorry


end NUMINAMATH_CALUDE_watch_sale_gain_percentage_l1153_115389


namespace NUMINAMATH_CALUDE_class_fund_problem_l1153_115321

theorem class_fund_problem (total_amount : ℕ) (twenty_bill_count : ℕ) (other_bill_count : ℕ) 
  (h1 : total_amount = 120)
  (h2 : other_bill_count = 2 * twenty_bill_count)
  (h3 : twenty_bill_count = 3) :
  total_amount - (twenty_bill_count * 20) = 60 := by
  sorry

end NUMINAMATH_CALUDE_class_fund_problem_l1153_115321


namespace NUMINAMATH_CALUDE_two_tails_one_head_probability_l1153_115367

def coin_toss_probability : ℚ := 3/8

theorem two_tails_one_head_probability :
  let n_coins := 3
  let n_tails := 2
  let n_heads := 1
  let total_outcomes := 2^n_coins
  let favorable_outcomes := n_coins.choose n_tails
  coin_toss_probability = (favorable_outcomes : ℚ) / total_outcomes :=
by sorry

end NUMINAMATH_CALUDE_two_tails_one_head_probability_l1153_115367


namespace NUMINAMATH_CALUDE_dragon_boat_festival_probability_l1153_115362

theorem dragon_boat_festival_probability (pA pB pC : ℝ) 
  (hA : pA = 2/3) (hB : pB = 1/4) (hC : pC = 3/5) : 
  1 - (1 - pA) * (1 - pB) * (1 - pC) = 9/10 := by
  sorry

end NUMINAMATH_CALUDE_dragon_boat_festival_probability_l1153_115362


namespace NUMINAMATH_CALUDE_seven_balls_two_boxes_l1153_115391

/-- The number of ways to distribute indistinguishable balls into distinguishable boxes -/
def distribute_balls (n : ℕ) (k : ℕ) : ℕ :=
  n + 1

theorem seven_balls_two_boxes :
  distribute_balls 7 2 = 8 := by
  sorry

end NUMINAMATH_CALUDE_seven_balls_two_boxes_l1153_115391


namespace NUMINAMATH_CALUDE_winning_scenarios_is_60_l1153_115310

/-- The number of different winning scenarios for a lottery ticket distribution -/
def winning_scenarios : ℕ :=
  let total_tickets : ℕ := 8
  let num_people : ℕ := 4
  let tickets_per_person : ℕ := 2
  let first_prize : ℕ := 1
  let second_prize : ℕ := 1
  let third_prize : ℕ := 1
  let non_winning_tickets : ℕ := 5

  -- The actual computation of winning scenarios
  60

/-- Theorem stating that the number of winning scenarios is 60 -/
theorem winning_scenarios_is_60 : winning_scenarios = 60 := by
  sorry

end NUMINAMATH_CALUDE_winning_scenarios_is_60_l1153_115310


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l1153_115368

theorem imaginary_part_of_complex_fraction (i : ℂ) : i * i = -1 → Complex.im ((1 + i) / (1 - i)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_fraction_l1153_115368


namespace NUMINAMATH_CALUDE_dolls_made_l1153_115315

def accessories_per_doll : ℕ := 2 + 3 + 1 + 5

def time_per_doll_and_accessories : ℕ := 45 + accessories_per_doll * 10

def total_operation_time : ℕ := 1860000

theorem dolls_made : 
  total_operation_time / time_per_doll_and_accessories = 12000 := by sorry

end NUMINAMATH_CALUDE_dolls_made_l1153_115315


namespace NUMINAMATH_CALUDE_arithmetic_calculation_l1153_115386

theorem arithmetic_calculation : -8 + (-10) - 3 - (-6) = -15 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_calculation_l1153_115386


namespace NUMINAMATH_CALUDE_sum_of_roots_cubic_equation_l1153_115326

theorem sum_of_roots_cubic_equation : 
  let f (x : ℝ) := (x^3 - 3*x^2 - 12*x) / (x + 3)
  ∃ (a b : ℝ), (∀ x ≠ -3, f x = 3 ↔ x = a ∨ x = b) ∧ a + b = 4 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_roots_cubic_equation_l1153_115326


namespace NUMINAMATH_CALUDE_adrianna_gum_count_l1153_115398

/-- Calculates the remaining gum count for Adrianna --/
def remaining_gum (initial_gum : ℕ) (additional_gum : ℕ) (friends_given_gum : ℕ) : ℕ :=
  initial_gum + additional_gum - friends_given_gum

/-- Theorem stating that Adrianna has 2 pieces of gum left --/
theorem adrianna_gum_count :
  remaining_gum 10 3 11 = 2 := by
  sorry

end NUMINAMATH_CALUDE_adrianna_gum_count_l1153_115398


namespace NUMINAMATH_CALUDE_sequence_property_l1153_115390

/-- The function generating the sequence -/
def f (n : ℕ) : ℕ := 2 * (n + 1)^2 * (n + 2)^2

/-- Predicate to check if a number is the sum of two square integers -/
def isSumOfTwoSquares (m : ℕ) : Prop := ∃ a b : ℕ, m = a^2 + b^2

theorem sequence_property :
  (∀ n : ℕ, f n < f (n + 1)) ∧
  (∀ n : ℕ, isSumOfTwoSquares (f n)) ∧
  f 1 = 72 ∧ f 2 = 288 ∧ f 3 = 800 :=
sorry

end NUMINAMATH_CALUDE_sequence_property_l1153_115390


namespace NUMINAMATH_CALUDE_m_range_l1153_115333

-- Define the propositions p and q
def p (m : ℝ) : Prop := ∀ x : ℝ, 2 * x - x^2 < m

def q (m : ℝ) : Prop := |m - 1| ≥ 2

-- State the theorem
theorem m_range :
  (∀ m : ℝ, ¬(¬(p m))) ∧ (∀ m : ℝ, ¬(p m ∧ q m)) →
  ∀ m : ℝ, (m ∈ Set.Ioo 1 3) ↔ (p m ∧ ¬(q m)) :=
sorry

end NUMINAMATH_CALUDE_m_range_l1153_115333


namespace NUMINAMATH_CALUDE_elevator_problem_l1153_115384

theorem elevator_problem (x y z w v : ℕ) (h : x = 15 ∧ y = 9 ∧ z = 12 ∧ w = 6 ∧ v = 10) :
  x - y + z - w + v = 28 :=
by sorry

end NUMINAMATH_CALUDE_elevator_problem_l1153_115384


namespace NUMINAMATH_CALUDE_max_a_proof_l1153_115350

/-- The coefficient of x^4 in the expansion of (1 - 2x + ax^2)^8 -/
def coeff_x4 (a : ℝ) : ℝ := 28 * a^2 + 336 * a + 1120

/-- The maximum value of a that satisfies the condition -/
def max_a : ℝ := -5

theorem max_a_proof :
  (∀ a : ℝ, coeff_x4 a = -1540 → a ≤ max_a) ∧
  coeff_x4 max_a = -1540 := by sorry

end NUMINAMATH_CALUDE_max_a_proof_l1153_115350


namespace NUMINAMATH_CALUDE_unique_solution_l1153_115302

theorem unique_solution : ∃! x : ℝ, ((x / 8) + 8 - 30) * 6 = 12 := by
  sorry

end NUMINAMATH_CALUDE_unique_solution_l1153_115302


namespace NUMINAMATH_CALUDE_sum_y_coordinates_on_y_axis_l1153_115319

-- Define the circle
def circle_center : ℝ × ℝ := (-4, 3)
def circle_radius : ℝ := 5

-- Define a function to check if a point is on the circle
def on_circle (point : ℝ × ℝ) : Prop :=
  (point.1 - circle_center.1)^2 + (point.2 - circle_center.2)^2 = circle_radius^2

-- Define a function to check if a point is on the y-axis
def on_y_axis (point : ℝ × ℝ) : Prop :=
  point.1 = 0

-- Theorem statement
theorem sum_y_coordinates_on_y_axis :
  ∃ (p1 p2 : ℝ × ℝ),
    on_circle p1 ∧ on_circle p2 ∧
    on_y_axis p1 ∧ on_y_axis p2 ∧
    p1 ≠ p2 ∧
    p1.2 + p2.2 = 6 :=
  sorry

end NUMINAMATH_CALUDE_sum_y_coordinates_on_y_axis_l1153_115319


namespace NUMINAMATH_CALUDE_sine_shift_equivalence_l1153_115358

theorem sine_shift_equivalence :
  ∀ x : ℝ, Real.sin (2 * x + π / 6) = Real.sin (2 * (x + π / 4) - π / 3) :=
by sorry

end NUMINAMATH_CALUDE_sine_shift_equivalence_l1153_115358


namespace NUMINAMATH_CALUDE_distance_between_trees_l1153_115318

-- Define the yard length and number of trees
def yard_length : ℝ := 520
def num_trees : ℕ := 40

-- Theorem statement
theorem distance_between_trees :
  let num_spaces : ℕ := num_trees - 1
  let distance : ℝ := yard_length / num_spaces
  distance = 520 / 39 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_trees_l1153_115318


namespace NUMINAMATH_CALUDE_slide_wait_time_l1153_115329

theorem slide_wait_time (kids_swings : ℕ) (kids_slide : ℕ) (swing_wait_min : ℕ) (time_diff_sec : ℕ) :
  kids_swings = 3 →
  kids_slide = 2 * kids_swings →
  swing_wait_min = 2 →
  (kids_slide * swing_wait_min * 60 + time_diff_sec) - (kids_swings * swing_wait_min * 60) = 270 →
  kids_slide * swing_wait_min * 60 + time_diff_sec = 630 :=
by
  sorry

#check slide_wait_time

end NUMINAMATH_CALUDE_slide_wait_time_l1153_115329


namespace NUMINAMATH_CALUDE_edward_money_theorem_l1153_115349

def edward_money_problem (initial_amount spent1 spent2 remaining : ℕ) : Prop :=
  initial_amount = spent1 + spent2 + remaining

theorem edward_money_theorem :
  ∃ initial_amount : ℕ,
    edward_money_problem initial_amount 9 8 17 ∧ initial_amount = 34 := by
  sorry

end NUMINAMATH_CALUDE_edward_money_theorem_l1153_115349


namespace NUMINAMATH_CALUDE_unique_positive_number_l1153_115357

theorem unique_positive_number : ∃! x : ℝ, x > 0 ∧ x + 17 = 60 / x := by
  sorry

end NUMINAMATH_CALUDE_unique_positive_number_l1153_115357


namespace NUMINAMATH_CALUDE_pond_volume_calculation_l1153_115332

/-- The volume of a rectangular pond -/
def pond_volume (length width depth : ℝ) : ℝ :=
  length * width * depth

/-- Theorem: The volume of a rectangular pond with dimensions 28 m × 10 m × 5 m is 1400 cubic meters -/
theorem pond_volume_calculation : pond_volume 28 10 5 = 1400 := by
  sorry

end NUMINAMATH_CALUDE_pond_volume_calculation_l1153_115332


namespace NUMINAMATH_CALUDE_base_two_rep_of_125_l1153_115366

theorem base_two_rep_of_125 : 
  (125 : ℕ).digits 2 = [1, 0, 1, 1, 1, 1, 1] :=
by sorry

end NUMINAMATH_CALUDE_base_two_rep_of_125_l1153_115366


namespace NUMINAMATH_CALUDE_inequality_pattern_l1153_115324

theorem inequality_pattern (x : ℝ) (n : ℕ) (h : x > 0) : 
  x + (n^n : ℝ) / x^n ≥ (n : ℝ) + 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_pattern_l1153_115324


namespace NUMINAMATH_CALUDE_perfect_square_power_of_two_l1153_115317

theorem perfect_square_power_of_two (n : ℕ+) : 
  (∃ m : ℕ, 2^8 + 2^11 + 2^(n : ℕ) = m^2) ↔ n = 12 := by
  sorry

end NUMINAMATH_CALUDE_perfect_square_power_of_two_l1153_115317


namespace NUMINAMATH_CALUDE_joan_clothing_expenses_l1153_115378

theorem joan_clothing_expenses : 
  15 + 14.82 + 12.51 = 42.33 := by sorry

end NUMINAMATH_CALUDE_joan_clothing_expenses_l1153_115378


namespace NUMINAMATH_CALUDE_max_area_convex_quadrilateral_l1153_115327

/-- A convex quadrilateral with diagonals d₁ and d₂ has an area S. -/
structure ConvexQuadrilateral where
  d₁ : ℝ
  d₂ : ℝ
  S : ℝ
  d₁_pos : d₁ > 0
  d₂_pos : d₂ > 0
  S_pos : S > 0
  area_formula : ∃ α : ℝ, 0 ≤ α ∧ α ≤ π ∧ S = (1/2) * d₁ * d₂ * Real.sin α

/-- The maximum area of a convex quadrilateral is half the product of its diagonals. -/
theorem max_area_convex_quadrilateral (q : ConvexQuadrilateral) : 
  q.S ≤ (1/2) * q.d₁ * q.d₂ ∧ ∃ q' : ConvexQuadrilateral, q'.S = (1/2) * q'.d₁ * q'.d₂ := by
  sorry


end NUMINAMATH_CALUDE_max_area_convex_quadrilateral_l1153_115327


namespace NUMINAMATH_CALUDE_trees_in_garden_l1153_115356

/-- The number of trees in a yard with given length and spacing -/
def number_of_trees (yard_length : ℕ) (tree_spacing : ℕ) : ℕ :=
  (yard_length / tree_spacing) + 1

/-- Theorem: There are 26 trees in a 300-meter yard with 12-meter spacing -/
theorem trees_in_garden : number_of_trees 300 12 = 26 := by
  sorry

end NUMINAMATH_CALUDE_trees_in_garden_l1153_115356


namespace NUMINAMATH_CALUDE_smallest_a_inequality_two_ninths_satisfies_inequality_l1153_115341

theorem smallest_a_inequality (x y z : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) (hz : z ≥ 0) (hsum : x + y + z = 1) :
  ∀ a : ℝ, (∀ x y z : ℝ, x ≥ 0 → y ≥ 0 → z ≥ 0 → x + y + z = 1 → a * (x^2 + y^2 + z^2) + x*y*z ≥ 1/3) →
  a ≥ 2/9 :=
by sorry

theorem two_ninths_satisfies_inequality (x y z : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) (hz : z ≥ 0) (hsum : x + y + z = 1) :
  (2/9 : ℝ) * (x^2 + y^2 + z^2) + x*y*z ≥ 1/3 :=
by sorry

end NUMINAMATH_CALUDE_smallest_a_inequality_two_ninths_satisfies_inequality_l1153_115341


namespace NUMINAMATH_CALUDE_gcd_5280_12155_l1153_115383

theorem gcd_5280_12155 : Int.gcd 5280 12155 = 5 := by
  sorry

end NUMINAMATH_CALUDE_gcd_5280_12155_l1153_115383


namespace NUMINAMATH_CALUDE_tabitha_current_age_l1153_115380

/- Define the problem parameters -/
def start_age : ℕ := 15
def start_colors : ℕ := 2
def future_colors : ℕ := 8
def years_to_future : ℕ := 3

/- Define Tabitha's age as a function of the number of colors -/
def tabitha_age (colors : ℕ) : ℕ := start_age + (colors - start_colors)

/- Define the number of colors Tabitha has now -/
def current_colors : ℕ := future_colors - years_to_future

/- The theorem to prove -/
theorem tabitha_current_age :
  tabitha_age current_colors = 18 := by
  sorry


end NUMINAMATH_CALUDE_tabitha_current_age_l1153_115380


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l1153_115365

theorem sufficient_not_necessary : 
  (∀ x : ℝ, x > 0 → |x| > 0) ∧ (∃ x : ℝ, |x| > 0 ∧ x ≤ 0) := by sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l1153_115365


namespace NUMINAMATH_CALUDE_inverse_variation_problem_l1153_115394

/-- Given that x² varies inversely with ⁴√w, prove that if x = 3 when w = 16, then x = √6 when w = 81 -/
theorem inverse_variation_problem (x w : ℝ) (h : ∃ k : ℝ, ∀ x w, x^2 * w^(1/4) = k) :
  (x = 3 ∧ w = 16) → (w = 81 → x = Real.sqrt 6) := by
  sorry

end NUMINAMATH_CALUDE_inverse_variation_problem_l1153_115394


namespace NUMINAMATH_CALUDE_tiger_count_l1153_115351

/-- Given a zoo where the ratio of lions to tigers is 3:4 and there are 21 lions, 
    prove that the number of tigers is 28. -/
theorem tiger_count (lion_count : ℕ) (tiger_count : ℕ) : 
  (lion_count : ℚ) / tiger_count = 3 / 4 → 
  lion_count = 21 → 
  tiger_count = 28 := by
  sorry

end NUMINAMATH_CALUDE_tiger_count_l1153_115351


namespace NUMINAMATH_CALUDE_stating_bryans_books_l1153_115395

/-- 
Given the number of books per bookshelf and the number of bookshelves,
calculates the total number of books.
-/
def total_books (books_per_shelf : ℕ) (num_shelves : ℕ) : ℕ :=
  books_per_shelf * num_shelves

/-- 
Theorem stating that with 2 books per shelf and 21 shelves,
the total number of books is 42.
-/
theorem bryans_books : 
  total_books 2 21 = 42 := by
  sorry

end NUMINAMATH_CALUDE_stating_bryans_books_l1153_115395


namespace NUMINAMATH_CALUDE_smallest_valid_number_l1153_115388

def is_valid_number (n : ℕ) : Prop :=
  ∃ (chosen : Finset ℕ) (unchosen : Finset ℕ),
    chosen.card = 5 ∧
    unchosen.card = 4 ∧
    chosen ∪ unchosen = Finset.range 9 ∧
    chosen ∩ unchosen = ∅ ∧
    (∀ d ∈ chosen, n % d = 0) ∧
    (∀ d ∈ unchosen, n % d ≠ 0) ∧
    n ≥ 10000 ∧ n < 100000

theorem smallest_valid_number :
  ∃ (n : ℕ), is_valid_number n ∧ 
  (∀ m, is_valid_number m → n ≤ m) ∧
  n = 14728 := by
  sorry

end NUMINAMATH_CALUDE_smallest_valid_number_l1153_115388


namespace NUMINAMATH_CALUDE_perpendicular_bisector_value_l1153_115376

/-- The perpendicular bisector of a line segment passes through its midpoint and is perpendicular to the segment. -/
structure PerpendicularBisector (p₁ p₂ : ℝ × ℝ) (l : ℝ × ℝ → Prop) : Prop where
  passes_through_midpoint : l ((p₁.1 + p₂.1) / 2, (p₁.2 + p₂.2) / 2)
  is_perpendicular : True  -- We don't need to express this condition for this problem

/-- The line equation x + y = b -/
def line_equation (b : ℝ) : ℝ × ℝ → Prop :=
  fun p => p.1 + p.2 = b

/-- The main theorem: if x + y = b is the perpendicular bisector of the line segment
    from (2,5) to (8,11), then b = 13 -/
theorem perpendicular_bisector_value :
  PerpendicularBisector (2, 5) (8, 11) (line_equation b) → b = 13 :=
by
  sorry


end NUMINAMATH_CALUDE_perpendicular_bisector_value_l1153_115376


namespace NUMINAMATH_CALUDE_simplified_fraction_ratio_l1153_115354

theorem simplified_fraction_ratio (k c d : ℤ) : 
  (5 * k + 15) / 5 = c * k + d → c / d = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_simplified_fraction_ratio_l1153_115354


namespace NUMINAMATH_CALUDE_simplify_expression_l1153_115375

theorem simplify_expression (x : ℝ) : 2*x - 3*(2-x) + 4*(2+x) - 5*(1-3*x) = 24*x - 3 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l1153_115375


namespace NUMINAMATH_CALUDE_triangle_ratio_l1153_115393

theorem triangle_ratio (A B C : ℝ) (a b c : ℝ) :
  0 < A ∧ 0 < B ∧ 0 < C ∧
  A + B + C = π ∧
  0 < a ∧ 0 < b ∧ 0 < c ∧
  a * Real.cos B + b * Real.cos A = 3 * a →
  c / a = 3 := by sorry

end NUMINAMATH_CALUDE_triangle_ratio_l1153_115393


namespace NUMINAMATH_CALUDE_greatest_three_digit_multiple_of_17_l1153_115325

theorem greatest_three_digit_multiple_of_17 : 
  ∀ n : ℕ, n ≤ 999 → n ≥ 100 → n % 17 = 0 → n ≤ 986 :=
by
  sorry

end NUMINAMATH_CALUDE_greatest_three_digit_multiple_of_17_l1153_115325


namespace NUMINAMATH_CALUDE_sesame_seed_weight_scientific_notation_l1153_115392

theorem sesame_seed_weight_scientific_notation : 
  ∃ (a : ℝ) (n : ℤ), 0.00000201 = a * (10 : ℝ) ^ n ∧ 1 ≤ a ∧ a < 10 ∧ a = 2.01 ∧ n = -6 := by
  sorry

end NUMINAMATH_CALUDE_sesame_seed_weight_scientific_notation_l1153_115392


namespace NUMINAMATH_CALUDE_sqrt_three_minus_two_times_sqrt_three_plus_two_l1153_115312

theorem sqrt_three_minus_two_times_sqrt_three_plus_two : (Real.sqrt 3 - 2) * (Real.sqrt 3 + 2) = -1 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_three_minus_two_times_sqrt_three_plus_two_l1153_115312
