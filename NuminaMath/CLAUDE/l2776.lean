import Mathlib

namespace two_solutions_iff_a_gt_neg_one_l2776_277655

/-- The equation has exactly two solutions if and only if a > -1 -/
theorem two_solutions_iff_a_gt_neg_one (a : ℝ) :
  (∃! x y : ℝ, x ≠ y ∧ x^2 + 2*x + 2*|x+1| = a ∧ y^2 + 2*y + 2*|y+1| = a) ↔ a > -1 :=
sorry

end two_solutions_iff_a_gt_neg_one_l2776_277655


namespace square_area_from_diagonal_l2776_277602

/-- The area of a square with diagonal length 12√2 cm is 144 cm² -/
theorem square_area_from_diagonal : ∀ s : ℝ,
  s > 0 →
  s * s * 2 = (12 * Real.sqrt 2) ^ 2 →
  s * s = 144 :=
by sorry

end square_area_from_diagonal_l2776_277602


namespace reflect_across_y_axis_l2776_277683

/-- A point in a 2D plane. -/
structure Point2D where
  x : ℝ
  y : ℝ

/-- The reflection of a point across the y-axis. -/
def reflectAcrossYAxis (p : Point2D) : Point2D :=
  { x := -p.x, y := p.y }

/-- Theorem: The coordinates of a point P(x,y) with respect to the y-axis are (-x,y). -/
theorem reflect_across_y_axis (p : Point2D) :
  reflectAcrossYAxis p = { x := -p.x, y := p.y } := by
  sorry

#check reflect_across_y_axis

end reflect_across_y_axis_l2776_277683


namespace real_part_of_one_plus_i_squared_is_zero_l2776_277657

theorem real_part_of_one_plus_i_squared_is_zero :
  Complex.re ((1 : ℂ) + Complex.I) ^ 2 = 0 := by sorry

end real_part_of_one_plus_i_squared_is_zero_l2776_277657


namespace horner_v₁_value_l2776_277690

def f (x : ℝ) : ℝ := 3 * x^6 + 4 * x^5 + 5 * x^4 + 6 * x^3 + 7 * x^2 + 8 * x + 1

def horner_v₁ (a₆ a₅ a₄ a₃ a₂ a₁ a₀ x : ℝ) : ℝ :=
  ((((a₆ * x + a₅) * x + a₄) * x + a₃) * x + a₂) * x + a₁

theorem horner_v₁_value :
  horner_v₁ 3 4 5 6 7 8 1 0.4 = 5.2 :=
sorry

end horner_v₁_value_l2776_277690


namespace quadratic_roots_sum_of_sixth_powers_l2776_277698

theorem quadratic_roots_sum_of_sixth_powers (p q : ℂ) : 
  p^2 - 2*p*Real.sqrt 3 + 2 = 0 →
  q^2 - 2*q*Real.sqrt 3 + 2 = 0 →
  p^6 + q^6 = 3120 := by
  sorry

end quadratic_roots_sum_of_sixth_powers_l2776_277698


namespace midpoint_quadrilateral_perpendicular_diagonals_l2776_277631

/-- Represents a point on a circle --/
structure CirclePoint where
  angle : Real

/-- Represents a quadrilateral formed by midpoints of arcs on a circle --/
structure MidpointQuadrilateral where
  p1 : CirclePoint
  p2 : CirclePoint
  p3 : CirclePoint
  p4 : CirclePoint

/-- Calculates the angle between two diagonals of a quadrilateral --/
def diagonalAngle (q : MidpointQuadrilateral) : Real :=
  -- Implementation details omitted
  sorry

/-- States that the diagonals of a quadrilateral formed by midpoints of four arcs on a circle are perpendicular --/
theorem midpoint_quadrilateral_perpendicular_diagonals 
  (c : CirclePoint → CirclePoint → CirclePoint → CirclePoint → MidpointQuadrilateral) :
  ∀ (p1 p2 p3 p4 : CirclePoint), 
    diagonalAngle (c p1 p2 p3 p4) = Real.pi / 2 := by
  sorry

#check midpoint_quadrilateral_perpendicular_diagonals

end midpoint_quadrilateral_perpendicular_diagonals_l2776_277631


namespace abs_is_even_and_increasing_l2776_277663

def f (x : ℝ) := |x|

theorem abs_is_even_and_increasing :
  (∀ x, f (-x) = f x) ∧ 
  (∀ x y, 0 < x → x < y → f x < f y) := by
  sorry

end abs_is_even_and_increasing_l2776_277663


namespace min_real_roots_l2776_277603

/-- A polynomial of degree 12 with real coefficients -/
def RealPolynomial12 : Type := { p : Polynomial ℝ // p.degree = 12 }

/-- The roots of a polynomial -/
def roots (p : RealPolynomial12) : Finset ℂ := sorry

/-- The number of distinct absolute values among the roots -/
def distinctAbsValues (p : RealPolynomial12) : ℕ := sorry

/-- The number of real roots of a polynomial -/
def realRootCount (p : RealPolynomial12) : ℕ := sorry

/-- The theorem stating the minimum number of real roots -/
theorem min_real_roots (p : RealPolynomial12) 
  (h : distinctAbsValues p = 6) : 
  ∃ q : RealPolynomial12, realRootCount q = 1 ∧ 
    ∀ r : RealPolynomial12, realRootCount r ≥ 1 := by sorry

end min_real_roots_l2776_277603


namespace expression_evaluation_l2776_277641

theorem expression_evaluation (x y z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hz : z ≠ 0) (h : x = z / y) :
  (x - z / x) * (y + 1 / (z * y)) = (x^4 - z^3 + x^2 * (z^2 - z)) / (z * x^2) := by
  sorry

end expression_evaluation_l2776_277641


namespace mailbox_distance_l2776_277692

/-- Represents the problem of finding the distance to a mailbox --/
def MailboxProblem (initial_speed : ℝ) (return_speed : ℝ) (time_away : ℝ) : Prop :=
  let initial_speed_mpm := initial_speed * 1000 / 60
  let return_speed_mpm := return_speed * 1000 / 60
  let distance_mother_in_law := initial_speed_mpm * time_away
  let total_distance := return_speed_mpm * time_away
  let distance_to_mailbox := (total_distance + distance_mother_in_law) / 2
  distance_to_mailbox = 200

/-- The theorem stating the solution to the mailbox problem --/
theorem mailbox_distance :
  MailboxProblem 3 5 3 := by
  sorry


end mailbox_distance_l2776_277692


namespace optimal_move_is_six_l2776_277686

/-- Represents the state of a number in the game -/
inductive NumberState
| Unmarked
| Marked
| Blocked

/-- Represents the game state -/
structure GameState where
  numbers : Fin 17 → NumberState

/-- Checks if a move is valid according to the game rules -/
def isValidMove (state : GameState) (n : Fin 17) : Prop :=
  state.numbers n = NumberState.Unmarked ∧
  ∀ m : Fin 17, state.numbers m = NumberState.Marked →
    n.val ≠ 2 * m.val ∧ n.val ≠ m.val / 2

/-- Applies a move to the game state -/
def applyMove (state : GameState) (n : Fin 17) : GameState :=
  { numbers := λ m =>
      if m = n then NumberState.Marked
      else if m.val = 2 * n.val ∨ 2 * m.val = n.val then NumberState.Blocked
      else state.numbers m }

/-- Checks if the game is over (no more valid moves) -/
def isGameOver (state : GameState) : Prop :=
  ∀ n : Fin 17, ¬isValidMove state n

/-- Defines the initial game state after A marks 8 -/
def initialState : GameState :=
  { numbers := λ n => if n.val = 8 then NumberState.Marked else NumberState.Unmarked }

/-- Theorem: B's optimal move is to mark 6 -/
theorem optimal_move_is_six :
  ∃ (strategy : GameState → Fin 17),
    (∀ state : GameState, isValidMove state (strategy state)) ∧
    (∀ (state : GameState) (n : Fin 17),
      isValidMove state n →
      isGameOver (applyMove (applyMove state (strategy state)) n)) ∧
    strategy initialState = ⟨6, by norm_num⟩ := by
  sorry


end optimal_move_is_six_l2776_277686


namespace certain_person_age_l2776_277679

def sandy_age : ℕ := 34
def person_age : ℕ := 10

theorem certain_person_age :
  (sandy_age * 10 = 340) →
  ((sandy_age + 2) = 3 * (person_age + 2)) →
  person_age = 10 :=
by
  sorry

end certain_person_age_l2776_277679


namespace complex_fraction_simplification_l2776_277699

/-- Proves that (7 + 14i) / (3 - 4i) = 77/25 + 70/25 * i -/
theorem complex_fraction_simplification :
  (7 + 14 * Complex.I) / (3 - 4 * Complex.I) = 77/25 + 70/25 * Complex.I :=
by sorry


end complex_fraction_simplification_l2776_277699


namespace pool_filling_rounds_l2776_277684

/-- The number of buckets George can carry per round -/
def george_buckets : ℕ := 2

/-- The number of buckets Harry can carry per round -/
def harry_buckets : ℕ := 3

/-- The total number of buckets needed to fill the pool -/
def total_buckets : ℕ := 110

/-- The number of rounds needed to fill the pool -/
def rounds_to_fill : ℕ := total_buckets / (george_buckets + harry_buckets)

theorem pool_filling_rounds :
  rounds_to_fill = 22 := by sorry

end pool_filling_rounds_l2776_277684


namespace jose_age_l2776_277653

/-- Given the ages of Jose, Zack, and Inez, prove that Jose is 21 years old -/
theorem jose_age (jose zack inez : ℕ) 
  (h1 : jose = zack + 5) 
  (h2 : zack = inez + 4) 
  (h3 : inez = 12) : 
  jose = 21 := by
  sorry

end jose_age_l2776_277653


namespace rad_polynomial_characterization_l2776_277600

/-- rad(n) is the product of all distinct prime factors of n -/
def rad (n : ℕ+) : ℕ+ := sorry

/-- A number is square-free if it's not divisible by any perfect square other than 1 -/
def IsSquareFree (n : ℕ+) : Prop := sorry

/-- Polynomial with rational coefficients -/
def RationalPolynomial := Polynomial ℚ

theorem rad_polynomial_characterization (P : RationalPolynomial) :
  (∃ (s : Set ℕ+), Set.Infinite s ∧ ∀ n ∈ s, (P.eval n : ℚ) = (rad n : ℚ)) ↔
  (∃ b : ℕ+, P = Polynomial.monomial 1 (1 / (b : ℚ))) ∨
  (∃ k : ℕ+, IsSquareFree k ∧ P = Polynomial.C (k : ℚ)) := by sorry

end rad_polynomial_characterization_l2776_277600


namespace road_area_in_square_park_l2776_277668

/-- 
Given a square park with a road inside, this theorem proves that
if the road is 3 meters wide and the perimeter along its outer edge is 600 meters,
then the area occupied by the road is 1836 square meters.
-/
theorem road_area_in_square_park (park_side : ℝ) (road_width : ℝ) (outer_perimeter : ℝ) 
  (h1 : road_width = 3)
  (h2 : outer_perimeter = 600)
  (h3 : 4 * (park_side - 2 * road_width) = outer_perimeter) :
  park_side^2 - (park_side - 2 * road_width)^2 = 1836 := by
  sorry

end road_area_in_square_park_l2776_277668


namespace smallest_prime_divisor_of_sum_l2776_277615

theorem smallest_prime_divisor_of_sum (n : ℕ) : 
  ∃ (p : ℕ), Nat.Prime p ∧ p ∣ (3^19 + 11^13) ∧ ∀ (q : ℕ), Nat.Prime q → q ∣ (3^19 + 11^13) → p ≤ q :=
by sorry

end smallest_prime_divisor_of_sum_l2776_277615


namespace pants_price_proof_l2776_277606

/-- Given the total cost of a pair of pants and a belt, and the price difference between them,
    prove that the price of the pants is as stated. -/
theorem pants_price_proof (total_cost belt_price pants_price : ℝ) : 
  total_cost = 70.93 →
  pants_price = belt_price - 2.93 →
  total_cost = belt_price + pants_price →
  pants_price = 34.00 := by
sorry

end pants_price_proof_l2776_277606


namespace solution_range_l2776_277632

theorem solution_range (x y z : ℝ) 
  (sum_eq : x + y + z = 5) 
  (prod_eq : x*y + y*z + z*x = 3) : 
  -1 ≤ x ∧ x ≤ 13/3 ∧ 
  -1 ≤ y ∧ y ≤ 13/3 ∧ 
  -1 ≤ z ∧ z ≤ 13/3 := by
sorry

end solution_range_l2776_277632


namespace product_digit_sum_l2776_277697

def a : ℕ := 70707070707070707070707070707070707070707070707070707070707070707070707070707070707070707070707070707
def b : ℕ := 60606060606060606060606060606060606060606060606060606060606060606060606060606060606060606060606060606

theorem product_digit_sum :
  (a * b % 10) + ((a * b / 10000) % 10) = 6 := by
  sorry

end product_digit_sum_l2776_277697


namespace melanie_trout_l2776_277626

def melanie_catch : ℕ → ℕ → Prop
| m, t => t = 2 * m

theorem melanie_trout (tom_catch : ℕ) (h : melanie_catch 8 tom_catch) (h2 : tom_catch = 16) : 
  8 = 8 := by sorry

end melanie_trout_l2776_277626


namespace equation_solution_l2776_277637

theorem equation_solution : ∃ x : ℝ, 
  (216 + Real.sqrt 41472 - 18 * x - Real.sqrt (648 * x^2) = 0) ∧ 
  (x = (140 * Real.sqrt 2 - 140) / 9) := by
  sorry

end equation_solution_l2776_277637


namespace work_completion_theorem_l2776_277661

/-- The number of men originally employed to finish the work in 11 days -/
def original_men : ℕ := 27

/-- The number of additional men who joined -/
def additional_men : ℕ := 10

/-- The original number of days to finish the work -/
def original_days : ℕ := 11

/-- The number of days saved after additional men joined -/
def days_saved : ℕ := 3

theorem work_completion_theorem :
  original_men + additional_men = 37 ∧
  original_men * original_days = (original_men + additional_men) * (original_days - days_saved) :=
sorry

end work_completion_theorem_l2776_277661


namespace cubic_real_root_l2776_277665

theorem cubic_real_root (a b : ℝ) :
  (∃ x : ℂ, a * x^3 + 4 * x^2 + b * x - 35 = 0 ∧ x = -1 - 2*I) →
  (∃ x : ℝ, a * x^3 + 4 * x^2 + b * x - 35 = 0 ∧ x = 21/5) :=
by sorry

end cubic_real_root_l2776_277665


namespace tangent_line_at_one_l2776_277613

open Real

noncomputable def f (x : ℝ) := log x - 3 * x

theorem tangent_line_at_one :
  ∃ (m b : ℝ), (∀ x y, y = m * x + b ↔ 2 * x + y + 1 = 0) ∧
               m = deriv f 1 ∧
               f 1 = m * 1 + b := by
  sorry

end tangent_line_at_one_l2776_277613


namespace truck_rental_charge_per_mile_l2776_277688

/-- Given a truck rental scenario, calculate the charge per mile. -/
theorem truck_rental_charge_per_mile
  (rental_fee : ℚ)
  (total_paid : ℚ)
  (miles_driven : ℕ)
  (h1 : rental_fee = 2099 / 100)
  (h2 : total_paid = 9574 / 100)
  (h3 : miles_driven = 299)
  : (total_paid - rental_fee) / miles_driven = 1 / 4 := by
  sorry

#eval (9574 / 100 : ℚ) - (2099 / 100 : ℚ)
#eval ((9574 / 100 : ℚ) - (2099 / 100 : ℚ)) / 299

end truck_rental_charge_per_mile_l2776_277688


namespace greatest_integer_for_domain_all_reals_l2776_277620

theorem greatest_integer_for_domain_all_reals : 
  ∃ (b : ℤ), b = 11 ∧ 
  (∀ (c : ℤ), c > b → 
    ∃ (x : ℝ), 2 * x^2 + c * x + 18 = 0) ∧
  (∀ (x : ℝ), 2 * x^2 + b * x + 18 ≠ 0) :=
sorry

end greatest_integer_for_domain_all_reals_l2776_277620


namespace staircase_classroom_seats_l2776_277651

/-- Represents the number of seats in a row of the staircase classroom. -/
def seats (n : ℕ) (a : ℕ) : ℕ := 12 + (n - 1) * a

theorem staircase_classroom_seats :
  ∃ a : ℕ,
  (seats 15 a = 2 * seats 5 a) ∧ 
  (seats 21 a = 52) := by
  sorry

end staircase_classroom_seats_l2776_277651


namespace range_of_a_l2776_277639

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, (x^2 + (a+2)*x + 1) * ((3-2*a)*x^2 + 5*x + (3-2*a)) ≥ 0) →
  a ∈ Set.Icc (-4 : ℝ) 0 := by
sorry

end range_of_a_l2776_277639


namespace quadratic_inequality_solution_l2776_277659

/-- Given that the solution set of ax^2 + bx + 1 > 0 is (-1/2, 1/3), prove that a - b = -5 -/
theorem quadratic_inequality_solution (a b : ℝ) : 
  (∀ x, ax^2 + b*x + 1 > 0 ↔ -1/2 < x ∧ x < 1/3) → a - b = -5 := by
  sorry

end quadratic_inequality_solution_l2776_277659


namespace disease_test_probability_l2776_277609

theorem disease_test_probability (p_disease : ℝ) (p_positive_given_disease : ℝ) (p_positive_given_no_disease : ℝ) :
  p_disease = 1 / 300 →
  p_positive_given_disease = 1 →
  p_positive_given_no_disease = 0.03 →
  (p_disease * p_positive_given_disease) / 
  (p_disease * p_positive_given_disease + (1 - p_disease) * p_positive_given_no_disease) = 100 / 997 := by
  sorry

end disease_test_probability_l2776_277609


namespace fraction_sum_equals_point_three_l2776_277672

theorem fraction_sum_equals_point_three :
  (2 : ℚ) / 20 + (4 : ℚ) / 40 + (9 : ℚ) / 90 = (3 : ℚ) / 10 := by
  sorry

end fraction_sum_equals_point_three_l2776_277672


namespace problem_statement_l2776_277680

theorem problem_statement (a b c x y z : ℝ) 
  (h1 : x^2 - y^2 - z^2 = 2*a*y*z)
  (h2 : -x^2 + y^2 - z^2 = 2*b*z*x)
  (h3 : -x^2 - y^2 + z^2 = 2*c*x*y)
  (h4 : x*y*z ≠ 0) :
  a^2 + b^2 + c^2 - 2*a*b*c = 1 := by
sorry

end problem_statement_l2776_277680


namespace johns_allowance_l2776_277667

theorem johns_allowance (A : ℝ) : A = 2.40 ↔ 
  ∃ (arcade_spent toy_store_spent candy_store_spent : ℝ),
    arcade_spent = (3/5) * A ∧
    toy_store_spent = (1/3) * (A - arcade_spent) ∧
    candy_store_spent = A - arcade_spent - toy_store_spent ∧
    candy_store_spent = 0.64 := by
  sorry

end johns_allowance_l2776_277667


namespace decimal_multiplication_l2776_277652

theorem decimal_multiplication : (0.7 : ℝ) * 0.8 = 0.56 := by
  sorry

end decimal_multiplication_l2776_277652


namespace sin_square_sum_range_l2776_277623

theorem sin_square_sum_range (α β : ℝ) (h : 3 * (Real.sin α)^2 - 2 * Real.sin α + 2 * (Real.sin β)^2 = 0) :
  ∃ (x : ℝ), x = (Real.sin α)^2 + (Real.sin β)^2 ∧ 0 ≤ x ∧ x ≤ 4/9 :=
sorry

end sin_square_sum_range_l2776_277623


namespace cross_country_winning_scores_l2776_277621

/-- The number of teams in the cross-country meet -/
def num_teams : ℕ := 2

/-- The number of runners per team -/
def runners_per_team : ℕ := 6

/-- The total number of runners -/
def total_runners : ℕ := num_teams * runners_per_team

/-- The sum of positions from 1 to n -/
def sum_positions (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The total sum of all positions -/
def total_sum : ℕ := sum_positions total_runners

/-- A winning score is less than half of the total sum -/
def is_winning_score (score : ℕ) : Prop := score < total_sum / 2

/-- The minimum possible score for a team -/
def min_score : ℕ := sum_positions runners_per_team

/-- The maximum possible winning score -/
def max_winning_score : ℕ := total_sum / 2 - 1

/-- The number of different possible winning scores -/
def num_winning_scores : ℕ := max_winning_score - min_score + 1

theorem cross_country_winning_scores :
  num_winning_scores = 18 := by sorry

end cross_country_winning_scores_l2776_277621


namespace gcd_612_468_is_36_l2776_277654

theorem gcd_612_468_is_36 : Nat.gcd 612 468 = 36 := by
  sorry

end gcd_612_468_is_36_l2776_277654


namespace area_triangle_pqr_l2776_277604

/-- Given two lines intersecting at point P(2,8), with slopes 1 and 3 respectively,
    and Q and R being the intersections of these lines with the x-axis,
    the area of triangle PQR is 64/3. -/
theorem area_triangle_pqr :
  let P : ℝ × ℝ := (2, 8)
  let slope1 : ℝ := 1
  let slope2 : ℝ := 3
  let Q : ℝ × ℝ := (P.1 - P.2 / slope1, 0)
  let R : ℝ × ℝ := (P.1 - P.2 / slope2, 0)
  let area : ℝ := (1 / 2) * |Q.1 - R.1| * P.2
  area = 64 / 3 := by
sorry

end area_triangle_pqr_l2776_277604


namespace ellipse_line_intersection_l2776_277617

/-- An ellipse with equation x²/4 + y²/2 = 1 -/
def Ellipse (x y : ℝ) : Prop :=
  x^2 / 4 + y^2 / 2 = 1

/-- A line with equation y = k(x-1) -/
def Line (k x y : ℝ) : Prop :=
  y = k * (x - 1)

/-- The area of a triangle given three points -/
noncomputable def TriangleArea (x1 y1 x2 y2 x3 y3 : ℝ) : ℝ :=
  (1/2) * abs ((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))

/-- The main theorem -/
theorem ellipse_line_intersection (k : ℝ) :
  (∃ x1 y1 x2 y2,
    Ellipse x1 y1 ∧ Ellipse x2 y2 ∧
    Line k x1 y1 ∧ Line k x2 y2 ∧
    (x1 ≠ x2 ∨ y1 ≠ y2) ∧
    TriangleArea 2 0 x1 y1 x2 y2 = Real.sqrt 10 / 3) ↔
  k = 1 ∨ k = -1 := by
  sorry

end ellipse_line_intersection_l2776_277617


namespace smallest_b_for_factorization_l2776_277648

/-- 
Given a quadratic polynomial x^2 + bx + 3024, this theorem states that
111 is the smallest positive integer b for which the polynomial factors
into a product of two binomials with integer coefficients.
-/
theorem smallest_b_for_factorization : 
  ∃ (b : ℕ), b > 0 ∧ 
  (∀ (x : ℤ), ∃ (r s : ℤ), x^2 + b*x + 3024 = (x + r) * (x + s)) ∧
  (∀ (b' : ℕ), 0 < b' ∧ b' < b → 
    ¬(∀ (x : ℤ), ∃ (r s : ℤ), x^2 + b'*x + 3024 = (x + r) * (x + s))) ∧
  b = 111 :=
by sorry

end smallest_b_for_factorization_l2776_277648


namespace combined_age_when_mike_is_24_l2776_277629

/-- Calculates the combined age of Mike, Barbara, Tom, and Peter when Mike is 24 years old -/
def combinedAgeWhenMikeIs24 (mikesInitialAge barbarasInitialAge tomsInitialAge petersInitialAge : ℕ) : ℕ :=
  let ageIncrease := 24 - mikesInitialAge
  24 + (barbarasInitialAge + ageIncrease) + (tomsInitialAge + ageIncrease) + (petersInitialAge + ageIncrease)

/-- Theorem stating the combined age of Mike, Barbara, Tom, and Peter when Mike is 24 years old -/
theorem combined_age_when_mike_is_24 :
  ∀ (mikesInitialAge : ℕ),
    mikesInitialAge = 16 →
    ∀ (barbarasInitialAge : ℕ),
      barbarasInitialAge = mikesInitialAge / 2 →
      ∀ (tomsInitialAge : ℕ),
        tomsInitialAge = barbarasInitialAge + 4 →
        ∀ (petersInitialAge : ℕ),
          petersInitialAge = 2 * tomsInitialAge →
          combinedAgeWhenMikeIs24 mikesInitialAge barbarasInitialAge tomsInitialAge petersInitialAge = 92 :=
by
  sorry


end combined_age_when_mike_is_24_l2776_277629


namespace rebus_puzzle_solution_l2776_277694

theorem rebus_puzzle_solution :
  ∃! (A B C : ℕ),
    A ≠ 0 ∧ B ≠ 0 ∧ C ≠ 0 ∧
    A ≠ B ∧ B ≠ C ∧ A ≠ C ∧
    A < 10 ∧ B < 10 ∧ C < 10 ∧
    100 * A + 10 * B + A + 100 * A + 10 * B + C = 100 * A + 10 * C + C ∧
    100 * A + 10 * C + C = 1416 ∧
    A = 4 ∧ B = 7 ∧ C = 6 := by sorry

end rebus_puzzle_solution_l2776_277694


namespace rice_yields_variance_l2776_277627

def rice_yields : List ℝ := [9.8, 9.9, 10.1, 10, 10.2]

theorem rice_yields_variance : 
  let n : ℕ := rice_yields.length
  let mean : ℝ := rice_yields.sum / n
  let variance : ℝ := (rice_yields.map (fun x => (x - mean)^2)).sum / n
  variance = 0.02 := by sorry

end rice_yields_variance_l2776_277627


namespace managers_in_sample_l2776_277670

structure StaffUnit where
  total : ℕ
  managers : ℕ
  sample_size : ℕ

def stratified_sample_size (unit : StaffUnit) (stratum_size : ℕ) : ℕ :=
  (stratum_size * unit.sample_size) / unit.total

theorem managers_in_sample (unit : StaffUnit) 
    (h1 : unit.total = 160)
    (h2 : unit.managers = 32)
    (h3 : unit.sample_size = 20) :
  stratified_sample_size unit unit.managers = 4 := by
  sorry

end managers_in_sample_l2776_277670


namespace quadratic_inequality_solution_range_l2776_277675

theorem quadratic_inequality_solution_range (m : ℝ) : 
  (∃ x : ℝ, m * x^2 + 2 * m * x - 8 ≥ 0) ↔ m ≠ 0 :=
by sorry

end quadratic_inequality_solution_range_l2776_277675


namespace bakery_doughnuts_l2776_277628

/-- The number of doughnuts in each box -/
def doughnuts_per_box : ℕ := 10

/-- The number of boxes sold -/
def boxes_sold : ℕ := 27

/-- The number of doughnuts given away -/
def doughnuts_given_away : ℕ := 30

/-- The total number of doughnuts made by the bakery -/
def total_doughnuts : ℕ := doughnuts_per_box * boxes_sold + doughnuts_given_away

theorem bakery_doughnuts : total_doughnuts = 300 := by
  sorry

end bakery_doughnuts_l2776_277628


namespace redistribution_contribution_l2776_277664

theorem redistribution_contribution (earnings : Fin 5 → ℕ) 
  (h1 : earnings 0 = 18)
  (h2 : earnings 1 = 23)
  (h3 : earnings 2 = 30)
  (h4 : earnings 3 = 35)
  (h5 : earnings 4 = 50)
  (min_amount : ℕ := 30)
  : (earnings 4 - min_amount : ℕ) = 20 := by
  sorry

end redistribution_contribution_l2776_277664


namespace hyperbola_focus_asymptote_distance_l2776_277619

/-- A hyperbola with equation x^2 - y^2/b^2 = 1 where b > 0 -/
structure Hyperbola where
  b : ℝ
  h_pos : b > 0

/-- The asymptote of a hyperbola -/
structure Asymptote where
  slope : ℝ
  y_intercept : ℝ

/-- The focus of a hyperbola -/
structure Focus where
  x : ℝ
  y : ℝ

/-- Distance between a point and a line -/
def distance_point_to_line (p : ℝ × ℝ) (l : Asymptote) : ℝ :=
  sorry

/-- Theorem: If one asymptote of the hyperbola x^2 - y^2/b^2 = 1 (b > 0) is y = 2x, 
    then the distance from the focus to this asymptote is 2 -/
theorem hyperbola_focus_asymptote_distance 
  (h : Hyperbola) 
  (a : Asymptote) 
  (f : Focus) 
  (h_asymptote : a.slope = 2 ∧ a.y_intercept = 0) : 
  distance_point_to_line (f.x, f.y) a = 2 :=
sorry

end hyperbola_focus_asymptote_distance_l2776_277619


namespace beach_count_theorem_l2776_277618

/-- The total count of oysters and crabs over two days -/
def total_count (initial_oysters initial_crabs : ℕ) : ℕ :=
  initial_oysters + (initial_oysters / 2) +
  initial_crabs + (initial_crabs * 2 / 3)

/-- Theorem stating the total count for the given initial numbers -/
theorem beach_count_theorem :
  total_count 50 72 = 195 := by
  sorry

end beach_count_theorem_l2776_277618


namespace focus_to_line_distance_l2776_277687

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 8*x

-- Define the line
def line (x y : ℝ) : Prop := Real.sqrt 3 * x - y = 0

-- Define the focus of the parabola
def focus : ℝ × ℝ := (2, 0)

-- State the theorem
theorem focus_to_line_distance :
  let (fx, fy) := focus
  ∃ d : ℝ, d = Real.sqrt 3 ∧
    d = |Real.sqrt 3 * fx - fy| / Real.sqrt (3 + 1) :=
sorry

end focus_to_line_distance_l2776_277687


namespace shooting_competition_probability_l2776_277695

theorem shooting_competition_probability (p : ℝ) (n : ℕ) (k : ℕ) : 
  p = 0.4 → n = 3 → k = 2 →
  (Finset.sum (Finset.range (n + 1 - k)) (λ i => Nat.choose n (n - i) * p^(n - i) * (1 - p)^i)) = 0.352 := by
sorry

end shooting_competition_probability_l2776_277695


namespace quadratic_equation_solution_l2776_277685

theorem quadratic_equation_solution : ∃ x₁ x₂ : ℝ, 
  (x₁^2 - x₁ - 2 = 0) ∧ (x₂^2 - x₂ - 2 = 0) ∧ (x₁ = 2) ∧ (x₂ = -1) :=
by sorry

end quadratic_equation_solution_l2776_277685


namespace kitten_food_consumption_l2776_277640

/-- Proves that given the conditions, each kitten eats 0.75 cans of food per day -/
theorem kitten_food_consumption
  (num_kittens : ℕ)
  (num_adult_cats : ℕ)
  (initial_food : ℕ)
  (additional_food : ℕ)
  (days : ℕ)
  (adult_cat_consumption : ℚ)
  (h1 : num_kittens = 4)
  (h2 : num_adult_cats = 3)
  (h3 : initial_food = 7)
  (h4 : additional_food = 35)
  (h5 : days = 7)
  (h6 : adult_cat_consumption = 1)
  : (initial_food + additional_food - num_adult_cats * adult_cat_consumption * days) / (num_kittens * days) = 0.75 := by
  sorry


end kitten_food_consumption_l2776_277640


namespace quadratic_is_perfect_square_l2776_277649

theorem quadratic_is_perfect_square (c : ℝ) : 
  (∃ a b : ℝ, ∀ x : ℝ, 9*x^2 - 21*x + c = (a*x + b)^2) → c = 12.25 := by
  sorry

end quadratic_is_perfect_square_l2776_277649


namespace correct_proposition_l2776_277689

-- Define propositions p and q
variable (p q : Prop)

-- Define the conditions
axiom p_true : p
axiom q_false : ¬q

-- Theorem to prove
theorem correct_proposition : (¬p) ∨ (¬q) :=
sorry

end correct_proposition_l2776_277689


namespace xyz_sum_l2776_277693

theorem xyz_sum (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (hxy : x * y = 32) (hxz : x * z = 64) (hyz : y * z = 96) :
  x + y + z = 44 * Real.sqrt 3 / 3 := by
sorry

end xyz_sum_l2776_277693


namespace union_A_complementB_equals_result_l2776_277676

-- Define the sets A and B
def A : Set ℝ := {x | -1 ≤ x ∧ x ≤ 1}
def B : Set ℝ := {x | x^2 - 2*x < 0}

-- Define the complement of B
def complementB : Set ℝ := {x | ¬(x ∈ B)}

-- Define the result set
def result : Set ℝ := {x | x ≤ 1 ∨ 2 ≤ x}

-- Theorem statement
theorem union_A_complementB_equals_result : A ∪ complementB = result := by
  sorry

end union_A_complementB_equals_result_l2776_277676


namespace male_average_score_l2776_277601

theorem male_average_score (total_students : ℕ) (male_students : ℕ) (female_students : ℕ)
  (overall_average : ℚ) (female_average : ℚ) :
  total_students = male_students + female_students →
  total_students = 28 →
  male_students = 8 →
  female_students = 20 →
  overall_average = 90 →
  female_average = 92 →
  (total_students : ℚ) * overall_average = 
    (male_students : ℚ) * ((total_students : ℚ) * overall_average - (female_students : ℚ) * female_average) / (male_students : ℚ) + 
    (female_students : ℚ) * female_average →
  ((total_students : ℚ) * overall_average - (female_students : ℚ) * female_average) / (male_students : ℚ) = 85 :=
by sorry

end male_average_score_l2776_277601


namespace relationship_abc_l2776_277625

theorem relationship_abc (a b c : ℝ) 
  (ha : a = Real.rpow 0.6 0.4)
  (hb : b = Real.rpow 0.4 0.6)
  (hc : c = Real.rpow 0.4 0.4) :
  a > c ∧ c > b := by
  sorry

end relationship_abc_l2776_277625


namespace solve_equation_l2776_277656

theorem solve_equation (m : ℝ) (h : m + (m + 2) + (m + 4) = 24) : m = 6 := by
  sorry

end solve_equation_l2776_277656


namespace red_face_probability_l2776_277696

/-- The volume of the original cube in cubic centimeters -/
def original_volume : ℝ := 27

/-- The number of small cubes the original cube is sawn into -/
def num_small_cubes : ℕ := 27

/-- The volume of each small cube in cubic centimeters -/
def small_cube_volume : ℝ := 1

/-- The number of small cubes with at least one red face -/
def num_red_cubes : ℕ := 26

/-- The probability of selecting a cube with at least one red face -/
def prob_red_face : ℚ := 26 / 27

theorem red_face_probability :
  original_volume = num_small_cubes * small_cube_volume →
  (num_red_cubes : ℚ) / num_small_cubes = prob_red_face := by
  sorry

end red_face_probability_l2776_277696


namespace b_77_mod_40_l2776_277635

def b (n : ℕ) : ℕ := 5^n + 9^n

theorem b_77_mod_40 : b 77 ≡ 14 [MOD 40] := by
  sorry

end b_77_mod_40_l2776_277635


namespace ted_speed_l2776_277669

theorem ted_speed (frank_speed : ℝ) (h1 : frank_speed > 0) : 
  let ted_speed := (2 / 3) * frank_speed
  2 * frank_speed = 2 * ted_speed + 8 →
  ted_speed = 8 := by
sorry

end ted_speed_l2776_277669


namespace magic_8_ball_probability_l2776_277612

theorem magic_8_ball_probability :
  let n : ℕ := 6  -- total number of questions
  let k : ℕ := 3  -- number of positive answers we're looking for
  let p : ℚ := 1/3  -- probability of a positive answer
  Nat.choose n k * p^k * (1-p)^(n-k) = 160/729 := by
  sorry

end magic_8_ball_probability_l2776_277612


namespace adams_final_balance_l2776_277643

/-- Calculates the final balance after a series of transactions --/
def final_balance (initial : ℚ) (spent : List ℚ) (received : List ℚ) : ℚ :=
  initial - spent.sum + received.sum

/-- Theorem: Adam's final balance is $10.75 --/
theorem adams_final_balance :
  let initial : ℚ := 5
  let spent : List ℚ := [2, 1.5, 0.75]
  let received : List ℚ := [3, 2, 5]
  final_balance initial spent received = 10.75 := by
  sorry

end adams_final_balance_l2776_277643


namespace max_remainder_division_by_nine_l2776_277671

theorem max_remainder_division_by_nine (n : ℕ) : 
  n / 9 = 6 → n % 9 ≤ 8 ∧ ∃ m : ℕ, m / 9 = 6 ∧ m % 9 = 8 :=
by sorry

end max_remainder_division_by_nine_l2776_277671


namespace solve_equations_l2776_277638

theorem solve_equations :
  (∃ x₁ x₂ : ℝ, (x₁ - 3)^2 + 2*x₁*(x₁ - 3) = 0 ∧ (x₂ - 3)^2 + 2*x₂*(x₂ - 3) = 0 ∧ x₁ = 3 ∧ x₂ = 1) ∧
  (∃ y₁ y₂ : ℝ, y₁^2 - 4*y₁ + 1 = 0 ∧ y₂^2 - 4*y₂ + 1 = 0 ∧ y₁ = 2 + Real.sqrt 3 ∧ y₂ = 2 - Real.sqrt 3) :=
by sorry

end solve_equations_l2776_277638


namespace percentage_relation_l2776_277677

theorem percentage_relation (third_number : ℝ) (first_number : ℝ) (second_number : ℝ)
  (h1 : first_number = 0.08 * third_number)
  (h2 : second_number = 0.16 * third_number)
  (h3 : first_number = 0.5 * second_number) :
  first_number = 0.08 * third_number := by
sorry

end percentage_relation_l2776_277677


namespace tens_digit_of_13_pow_2017_l2776_277658

theorem tens_digit_of_13_pow_2017 : ∃ n : ℕ, 13^2017 ≡ 30 + n [ZMOD 100] :=
by sorry

end tens_digit_of_13_pow_2017_l2776_277658


namespace square_cube_root_product_l2776_277660

theorem square_cube_root_product (a b : ℝ) 
  (ha : a^2 = 16/25) (hb : b^3 = 125/8) : 
  Real.sqrt (a * b) = Real.sqrt 2 := by
  sorry

end square_cube_root_product_l2776_277660


namespace average_steps_needed_l2776_277681

def goal : ℕ := 10000
def days : ℕ := 9
def remaining_days : ℕ := 3

def steps_walked : List ℕ := [10200, 10400, 9400, 9100, 8300, 9200, 8900, 9500]

def total_goal : ℕ := goal * days

def steps_walked_so_far : ℕ := steps_walked.sum

def remaining_steps : ℕ := total_goal - steps_walked_so_far

theorem average_steps_needed (h : steps_walked.length = days - remaining_days) :
  remaining_steps / remaining_days = 5000 := by
  sorry

end average_steps_needed_l2776_277681


namespace inequality_preservation_l2776_277611

theorem inequality_preservation (a b c : ℝ) (h : a > b) : 
  a / (c^2 + 1) > b / (c^2 + 1) := by
  sorry

end inequality_preservation_l2776_277611


namespace yellow_balls_count_l2776_277610

theorem yellow_balls_count (total : ℕ) (white green red purple : ℕ) (prob : ℚ) :
  total = 60 ∧ 
  white = 22 ∧ 
  green = 18 ∧ 
  red = 5 ∧ 
  purple = 7 ∧ 
  prob = 4/5 ∧ 
  (white + green + (total - white - green - red - purple) : ℚ) / total = prob →
  total - white - green - red - purple = 8 := by
sorry

end yellow_balls_count_l2776_277610


namespace floor_equation_solutions_l2776_277662

theorem floor_equation_solutions : 
  (∃ (S : Finset ℕ), S.card = 9 ∧ 
    (∀ x : ℕ, x ∈ S ↔ ⌊(x : ℚ) / 5⌋ = ⌊(x : ℚ) / 7⌋)) :=
by sorry

end floor_equation_solutions_l2776_277662


namespace pipe_filling_time_l2776_277650

theorem pipe_filling_time (p q r t : ℝ) (hp : p = 6) (hr : r = 24) (ht : t = 3.4285714285714284)
  (h_total : 1/p + 1/q + 1/r = 1/t) : q = 8 := by
  sorry

end pipe_filling_time_l2776_277650


namespace all_three_live_to_75_exactly_one_lives_to_75_at_least_one_lives_to_75_l2776_277642

def probability_live_to_75 : ℝ := 0.60

-- Probability that all three policyholders live to 75
theorem all_three_live_to_75 : 
  probability_live_to_75 ^ 3 = 0.216 := by sorry

-- Probability that exactly one out of three policyholders lives to 75
theorem exactly_one_lives_to_75 : 
  3 * probability_live_to_75 * (1 - probability_live_to_75) ^ 2 = 0.288 := by sorry

-- Probability that at least one out of three policyholders lives to 75
theorem at_least_one_lives_to_75 : 
  1 - (1 - probability_live_to_75) ^ 3 = 0.936 := by sorry

end all_three_live_to_75_exactly_one_lives_to_75_at_least_one_lives_to_75_l2776_277642


namespace inequality_proof_l2776_277608

theorem inequality_proof (x : ℝ) : (x - 5) / ((x - 3)^2 + 1) < 0 ↔ x < 5 := by
  sorry

end inequality_proof_l2776_277608


namespace three_digit_repeating_decimal_cube_l2776_277636

theorem three_digit_repeating_decimal_cube (n : ℕ) : 
  (n < 1000 ∧ n > 0) →
  (∃ (a b : ℕ), b > a ∧ a > 0 ∧ b > 0 ∧ (n : ℚ) / 999 = (a : ℚ) / b ^ 3) →
  (n = 037 ∨ n = 296) :=
sorry

end three_digit_repeating_decimal_cube_l2776_277636


namespace candy_box_price_increase_l2776_277644

/-- Proves that the percentage increase in the price of a candy box is 25% --/
theorem candy_box_price_increase 
  (new_candy_price : ℝ) 
  (new_soda_price : ℝ) 
  (original_total : ℝ) 
  (h1 : new_candy_price = 15)
  (h2 : new_soda_price = 6)
  (h3 : new_soda_price = (3/2) * (original_total - new_candy_price + new_soda_price))
  (h4 : original_total = 16) :
  (new_candy_price - (original_total - (2/3) * new_soda_price)) / (original_total - (2/3) * new_soda_price) = 1/4 := by
  sorry

#check candy_box_price_increase

end candy_box_price_increase_l2776_277644


namespace ellipse_standard_equation_l2776_277674

/-- An ellipse with given foci and passing through specific points has the standard equation x²/8 + y²/4 = 1 -/
theorem ellipse_standard_equation (f1 f2 p1 p2 p3 : ℝ × ℝ) : 
  f1 = (0, -2) →
  f2 = (0, 2) →
  p1 = (-3/2, 5/2) →
  p2 = (2, -Real.sqrt 2) →
  p3 = (-1, Real.sqrt 14 / 2) →
  ∃ (ellipse : ℝ × ℝ → Prop),
    (∀ (x y : ℝ), ellipse (x, y) ↔ x^2/8 + y^2/4 = 1) ∧
    (ellipse f1 ∧ ellipse f2 ∧ ellipse p1 ∧ ellipse p2 ∧ ellipse p3) :=
by
  sorry

end ellipse_standard_equation_l2776_277674


namespace largest_rational_less_than_quarter_rank_3_l2776_277691

-- Define the rank of a rational number
def rank (q : ℚ) : ℕ :=
  -- The definition of rank is given in the problem statement
  sorry

-- Define the property of being the largest rational less than 1/4 with rank 3
def is_largest_less_than_quarter_rank_3 (q : ℚ) : Prop :=
  q < 1/4 ∧ rank q = 3 ∧ ∀ r, r < 1/4 ∧ rank r = 3 → r ≤ q

-- State the theorem
theorem largest_rational_less_than_quarter_rank_3 :
  ∃ q : ℚ, is_largest_less_than_quarter_rank_3 q ∧ q = 1/5 + 1/21 + 1/421 :=
sorry

end largest_rational_less_than_quarter_rank_3_l2776_277691


namespace average_marks_all_candidates_l2776_277633

/-- Proves that the average marks of all candidates is 35 given the specified conditions -/
theorem average_marks_all_candidates
  (total_candidates : ℕ)
  (passed_candidates : ℕ)
  (failed_candidates : ℕ)
  (avg_marks_passed : ℚ)
  (avg_marks_failed : ℚ)
  (h1 : total_candidates = 120)
  (h2 : passed_candidates = 100)
  (h3 : failed_candidates = total_candidates - passed_candidates)
  (h4 : avg_marks_passed = 39)
  (h5 : avg_marks_failed = 15) :
  (passed_candidates * avg_marks_passed + failed_candidates * avg_marks_failed) / total_candidates = 35 :=
by
  sorry

#check average_marks_all_candidates

end average_marks_all_candidates_l2776_277633


namespace fraction_irreducible_fraction_simplification_l2776_277616

-- Part (a)
theorem fraction_irreducible (a : ℤ) : 
  Int.gcd (a^3 + 2*a) (a^4 + 3*a^2 + 1) = 1 := by sorry

-- Part (b)
theorem fraction_simplification (n : ℤ) : 
  Int.gcd (5*n + 6) (8*n + 7) = 1 ∨ Int.gcd (5*n + 6) (8*n + 7) = 13 := by sorry

end fraction_irreducible_fraction_simplification_l2776_277616


namespace value_of_S_6_l2776_277678

theorem value_of_S_6 (x : ℝ) (h : x + 1/x = 5) : x^6 + 1/x^6 = 12196 := by
  sorry

end value_of_S_6_l2776_277678


namespace greatest_root_of_g_l2776_277666

def g (x : ℝ) : ℝ := 16 * x^4 - 20 * x^2 + 5

theorem greatest_root_of_g :
  ∃ (r : ℝ), r = Real.sqrt 5 / 2 ∧
  g r = 0 ∧
  ∀ (x : ℝ), g x = 0 → x ≤ r :=
sorry

end greatest_root_of_g_l2776_277666


namespace baseball_league_games_l2776_277630

/-- The number of games played in a baseball league -/
def total_games (n : ℕ) (games_per_matchup : ℕ) : ℕ :=
  n * (n - 1) * games_per_matchup / 2

/-- Theorem: In a 12-team league where each team plays 4 games with every other team, 
    the total number of games played is 264 -/
theorem baseball_league_games : 
  total_games 12 4 = 264 := by
  sorry

end baseball_league_games_l2776_277630


namespace intersection_determines_B_l2776_277634

def A : Set ℝ := {0, 1, 2, 3}

def B (m : ℝ) : Set ℝ := {x | x^2 - 5*x + m = 0}

theorem intersection_determines_B :
  ∃ m : ℝ, (A ∩ B m = {1}) → (B m = {1, 4}) := by sorry

end intersection_determines_B_l2776_277634


namespace expression_factorization_l2776_277673

theorem expression_factorization (b : ℝ) : 
  (10 * b^4 - 27 * b^3 + 18 * b^2) - (-6 * b^4 + 4 * b^3 - 3 * b^2) = 
  b^2 * (16 * b - 7) * (b - 3) := by
sorry

end expression_factorization_l2776_277673


namespace sector_area_l2776_277647

theorem sector_area (angle : Real) (radius : Real) : 
  angle = 150 * π / 180 → 
  radius = 2 → 
  (angle * radius^2) / 2 = (5/3) * π := by
  sorry

end sector_area_l2776_277647


namespace donna_truck_weight_l2776_277645

/-- The weight of Donna's fully loaded truck -/
def truck_weight : ℕ :=
  let empty_truck_weight : ℕ := 12000
  let soda_crate_weight : ℕ := 50
  let soda_crate_count : ℕ := 20
  let dryer_weight : ℕ := 3000
  let dryer_count : ℕ := 3
  let soda_weight : ℕ := soda_crate_weight * soda_crate_count
  let produce_weight : ℕ := 2 * soda_weight
  let dryers_weight : ℕ := dryer_weight * dryer_count
  empty_truck_weight + soda_weight + produce_weight + dryers_weight

/-- Theorem stating that Donna's fully loaded truck weighs 24,000 pounds -/
theorem donna_truck_weight : truck_weight = 24000 := by
  sorry

end donna_truck_weight_l2776_277645


namespace max_value_theorem_max_value_achievable_l2776_277614

theorem max_value_theorem (x y : ℝ) :
  (3 * x + 4 * y + 5) / Real.sqrt (3 * x^2 + 4 * y^2 + 6) ≤ Real.sqrt 50 :=
by sorry

theorem max_value_achievable :
  ∃ x y : ℝ, (3 * x + 4 * y + 5) / Real.sqrt (3 * x^2 + 4 * y^2 + 6) = Real.sqrt 50 :=
by sorry

end max_value_theorem_max_value_achievable_l2776_277614


namespace cubic_equation_solution_l2776_277624

theorem cubic_equation_solution :
  ∀ x : ℝ, x^3 - 4*x = 0 ↔ x = 0 ∨ x = -2 ∨ x = 2 := by
sorry

end cubic_equation_solution_l2776_277624


namespace task_completion_time_l2776_277682

/-- 
Given:
- Person A can complete a task in time a
- Person A and Person B together can complete the task in time c
- The rate of work is the reciprocal of the time taken

Prove:
- Person B can complete the task alone in time b, where 1/a + 1/b = 1/c
-/
theorem task_completion_time (a c : ℝ) (ha : a > 0) (hc : c > 0) (hac : c < a) :
  ∃ b : ℝ, b > 0 ∧ 1/a + 1/b = 1/c := by sorry

end task_completion_time_l2776_277682


namespace ten_teams_in_tournament_l2776_277646

/-- The number of games played in a round-robin tournament with n teams -/
def games_played (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem: In a round-robin tournament with 45 games, there were 10 teams -/
theorem ten_teams_in_tournament (h : games_played 10 = 45) : 
  ∃ (n : ℕ), n = 10 ∧ games_played n = 45 :=
by sorry

end ten_teams_in_tournament_l2776_277646


namespace complex_equation_proof_l2776_277605

theorem complex_equation_proof (a b : ℝ) : (-2 * I + 1 : ℂ) = a + b * I → a - b = 3 := by
  sorry

end complex_equation_proof_l2776_277605


namespace path_length_is_894_l2776_277622

/-- The length of the path with fencing and a bridge. -/
def path_length (pole_spacing : ℕ) (bridge_length : ℕ) (total_poles : ℕ) : ℕ :=
  let poles_one_side := total_poles / 2
  let intervals := poles_one_side - 1
  intervals * pole_spacing + bridge_length

/-- Theorem stating the length of the path given the conditions. -/
theorem path_length_is_894 :
  path_length 6 42 286 = 894 := by
  sorry

end path_length_is_894_l2776_277622


namespace josh_marbles_l2776_277607

theorem josh_marbles (initial : ℕ) (lost : ℕ) (difference : ℕ) (found : ℕ) : 
  initial = 15 →
  lost = 23 →
  difference = 14 →
  lost = found + difference →
  found = 9 := by
sorry

end josh_marbles_l2776_277607
