import Mathlib

namespace f_extrema_l1464_146447

-- Define the function f
def f (x y : ℝ) : ℝ := 2 * x^2 - 2 * y^2

-- Define the disk
def disk (x y : ℝ) : Prop := x^2 + y^2 ≤ 9

-- Theorem statement
theorem f_extrema :
  (∃ x y : ℝ, disk x y ∧ f x y = 18) ∧
  (∃ x y : ℝ, disk x y ∧ f x y = -18) ∧
  (∀ x y : ℝ, disk x y → f x y ≤ 18) ∧
  (∀ x y : ℝ, disk x y → f x y ≥ -18) := by
  sorry

end f_extrema_l1464_146447


namespace smallest_top_number_l1464_146444

/-- Represents the pyramid structure -/
structure Pyramid :=
  (layer1 : Fin 15 → ℕ)
  (layer2 : Fin 10 → ℕ)
  (layer3 : Fin 6 → ℕ)
  (layer4 : Fin 3 → ℕ)
  (layer5 : ℕ)

/-- The numbering rule for layers 2-5 -/
def validNumbering (p : Pyramid) : Prop :=
  (∀ i : Fin 10, p.layer2 i = p.layer1 (3*i) + p.layer1 (3*i+1) + p.layer1 (3*i+2)) ∧
  (∀ i : Fin 6, p.layer3 i = p.layer2 (3*i) + p.layer2 (3*i+1) + p.layer2 (3*i+2)) ∧
  (∀ i : Fin 3, p.layer4 i = p.layer3 (2*i) + p.layer3 (2*i+1) + p.layer3 (2*i+2)) ∧
  (p.layer5 = p.layer4 0 + p.layer4 1 + p.layer4 2)

/-- The bottom layer contains numbers 1 to 15 -/
def validBottomLayer (p : Pyramid) : Prop :=
  (∀ i : Fin 15, p.layer1 i ∈ Finset.range 16 \ {0}) ∧
  (∀ i j : Fin 15, i ≠ j → p.layer1 i ≠ p.layer1 j)

/-- The theorem stating the smallest possible number for the top block -/
theorem smallest_top_number (p : Pyramid) 
  (h1 : validNumbering p) (h2 : validBottomLayer p) : 
  p.layer5 ≥ 155 :=
sorry

end smallest_top_number_l1464_146444


namespace problem_statement_l1464_146420

theorem problem_statement : ∃ y : ℝ, (8000 * 6000 : ℝ) = 480 * (10 ^ y) → y = 5 := by
  sorry

end problem_statement_l1464_146420


namespace coloring_books_removed_l1464_146457

theorem coloring_books_removed (initial_stock : ℕ) (shelves : ℕ) (books_per_shelf : ℕ) : 
  initial_stock = 86 →
  shelves = 7 →
  books_per_shelf = 7 →
  initial_stock - (shelves * books_per_shelf) = 37 := by
sorry

end coloring_books_removed_l1464_146457


namespace john_mean_score_l1464_146425

def john_scores : List ℝ := [86, 90, 88, 82, 91]

theorem john_mean_score : (john_scores.sum / john_scores.length : ℝ) = 87.4 := by
  sorry

end john_mean_score_l1464_146425


namespace m_range_l1464_146434

-- Define the propositions p and q
def p (m : ℝ) : Prop := ∀ x : ℝ, m * x^2 + m * x + 1 > 0

def q (m : ℝ) : Prop := ∃ a b : ℝ, a > 0 ∧ b > 0 ∧ 
  ∀ x y : ℝ, x^2 / (m - 1) + y^2 / (m - 2) = 1 ↔ 
  x^2 / a^2 - y^2 / b^2 = 1

-- Define the set of m values
def M : Set ℝ := {m : ℝ | (0 ≤ m ∧ m ≤ 1) ∨ (2 ≤ m ∧ m < 4)}

-- State the theorem
theorem m_range : 
  (∀ m : ℝ, (p m ∨ q m) ∧ ¬(p m ∧ q m)) ↔ ∀ m : ℝ, m ∈ M :=
sorry

end m_range_l1464_146434


namespace range_of_m_inequality_for_nonzero_x_l1464_146410

-- Define the function f
def f (m : ℝ) (x : ℝ) : ℝ := |x + m|

-- Theorem 1: Range of m
theorem range_of_m (m : ℝ) : 
  (f m 1 + f m (-2) ≥ 5) ↔ (m ≤ -2 ∨ m ≥ 3) := by sorry

-- Theorem 2: Inequality for non-zero x
theorem inequality_for_nonzero_x (m : ℝ) (x : ℝ) (h : x ≠ 0) : 
  f m (1/x) + f m (-x) ≥ 2 := by sorry

end range_of_m_inequality_for_nonzero_x_l1464_146410


namespace elsas_final_marbles_l1464_146414

/-- Calculates the number of marbles Elsa has at the end of the day -/
def elsas_marbles : ℕ :=
  let initial := 150
  let after_breakfast := initial - (initial * 5 / 100)
  let after_lunch := after_breakfast - (after_breakfast * 2 / 5)
  let after_mom_gift := after_lunch + 25
  let after_susie_return := after_mom_gift + (after_breakfast * 2 / 5 * 150 / 100)
  let peter_exchange := 15
  let elsa_gives := peter_exchange * 3 / 5
  let elsa_receives := peter_exchange * 2 / 5
  let after_peter := after_susie_return - elsa_gives + elsa_receives
  let final := after_peter - (after_peter / 4)
  final

theorem elsas_final_marbles :
  elsas_marbles = 145 := by
  sorry

end elsas_final_marbles_l1464_146414


namespace state_fair_earnings_l1464_146477

theorem state_fair_earnings :
  let ticket_price : ℚ := 5
  let food_price : ℚ := 8
  let ride_price : ℚ := 4
  let souvenir_price : ℚ := 15
  let total_ticket_sales : ℚ := 2520
  let num_attendees : ℚ := total_ticket_sales / ticket_price
  let food_buyers_ratio : ℚ := 2/3
  let ride_goers_ratio : ℚ := 1/4
  let souvenir_buyers_ratio : ℚ := 1/8
  let food_earnings : ℚ := num_attendees * food_buyers_ratio * food_price
  let ride_earnings : ℚ := num_attendees * ride_goers_ratio * ride_price
  let souvenir_earnings : ℚ := num_attendees * souvenir_buyers_ratio * souvenir_price
  let total_earnings : ℚ := total_ticket_sales + food_earnings + ride_earnings + souvenir_earnings
  total_earnings = 6657 := by sorry

end state_fair_earnings_l1464_146477


namespace frannie_jump_count_l1464_146408

/-- The number of times Meg jumped -/
def meg_jumps : ℕ := 71

/-- The difference between Meg's and Frannie's jumps -/
def jump_difference : ℕ := 18

/-- The number of times Frannie jumped -/
def frannie_jumps : ℕ := meg_jumps - jump_difference

theorem frannie_jump_count : frannie_jumps = 53 := by sorry

end frannie_jump_count_l1464_146408


namespace max_correct_answers_l1464_146482

theorem max_correct_answers (total_questions : ℕ) (correct_points : ℤ) (incorrect_points : ℤ) (total_score : ℤ) : 
  total_questions = 25 →
  correct_points = 6 →
  incorrect_points = -3 →
  total_score = 60 →
  (∃ (correct incorrect unanswered : ℕ),
    correct + incorrect + unanswered = total_questions ∧
    correct * correct_points + incorrect * incorrect_points = total_score) →
  (∀ (correct : ℕ),
    (∃ (incorrect unanswered : ℕ),
      correct + incorrect + unanswered = total_questions ∧
      correct * correct_points + incorrect * incorrect_points = total_score) →
    correct ≤ 15) :=
by sorry

end max_correct_answers_l1464_146482


namespace wetland_area_scientific_notation_l1464_146489

/-- Represents the scientific notation of a number -/
structure ScientificNotation where
  coefficient : ℝ
  exponent : ℤ

/-- Converts a number to its scientific notation representation -/
def toScientificNotation (n : ℝ) : ScientificNotation :=
  sorry

theorem wetland_area_scientific_notation :
  toScientificNotation (29.47 * 1000) = ScientificNotation.mk 2.947 4 :=
sorry

end wetland_area_scientific_notation_l1464_146489


namespace janes_age_l1464_146461

theorem janes_age (joe_age jane_age : ℕ) 
  (sum_of_ages : joe_age + jane_age = 54)
  (age_difference : joe_age - jane_age = 22) : 
  jane_age = 16 := by
sorry

end janes_age_l1464_146461


namespace derivative_sin_cos_product_l1464_146430

open Real

theorem derivative_sin_cos_product (x : ℝ) : 
  deriv (λ x => sin x * cos x) x = cos (2 * x) := by
  sorry

end derivative_sin_cos_product_l1464_146430


namespace p_is_cubic_l1464_146475

/-- The polynomial under consideration -/
def p (x : ℝ) : ℝ := 2^3 + 2^2*x - 2*x^2 - x^3

/-- The degree of a polynomial -/
def degree (p : ℝ → ℝ) : ℕ := sorry

theorem p_is_cubic : degree p = 3 := by sorry

end p_is_cubic_l1464_146475


namespace fraction_equality_l1464_146403

theorem fraction_equality {a b c : ℝ} (h1 : c ≠ 0) (h2 : a / c = b / c) : a = b := by
  sorry

end fraction_equality_l1464_146403


namespace second_train_speed_l1464_146463

/-- Given two trains traveling towards each other, prove that the speed of the second train is 16 km/hr -/
theorem second_train_speed
  (speed_train1 : ℝ)
  (total_distance : ℝ)
  (distance_difference : ℝ)
  (h1 : speed_train1 = 20)
  (h2 : total_distance = 630)
  (h3 : distance_difference = 70)
  : ∃ (speed_train2 : ℝ), speed_train2 = 16 :=
by
  sorry

end second_train_speed_l1464_146463


namespace min_abs_sum_l1464_146462

theorem min_abs_sum (x : ℝ) : 
  |x + 2| + |x + 4| + |x + 5| ≥ 3 ∧ ∃ y : ℝ, |y + 2| + |y + 4| + |y + 5| = 3 := by
  sorry

end min_abs_sum_l1464_146462


namespace investment_problem_l1464_146424

theorem investment_problem (total_interest desired_interest fixed_investment fixed_rate variable_rate : ℝ) :
  desired_interest = 980 →
  fixed_investment = 6000 →
  fixed_rate = 0.09 →
  variable_rate = 0.11 →
  total_interest = fixed_rate * fixed_investment + variable_rate * (total_interest - fixed_investment) →
  total_interest = 10000 := by
  sorry

end investment_problem_l1464_146424


namespace perfect_square_condition_l1464_146466

theorem perfect_square_condition (n : ℕ+) : 
  (∃ m : ℕ, n.val^2 + 5*n.val + 13 = m^2) → n.val = 4 := by
  sorry

end perfect_square_condition_l1464_146466


namespace nested_sqrt_value_l1464_146454

noncomputable def nested_sqrt_sequence : ℕ → ℝ
  | 0 => Real.sqrt 86
  | n + 1 => Real.sqrt (86 + 41 * nested_sqrt_sequence n)

theorem nested_sqrt_value :
  ∃ (limit : ℝ), limit = Real.sqrt (86 + 41 * limit) ∧ limit = 43 := by
  sorry

end nested_sqrt_value_l1464_146454


namespace cubic_polynomial_determinant_l1464_146473

/-- Given a cubic polynomial x^3 + sx^2 + px + q with roots a, b, and c,
    the determinant of the matrix [[s + a, 1, 1], [1, s + b, 1], [1, 1, s + c]]
    is equal to s^3 + sp - q - 2s - 2(p - s) -/
theorem cubic_polynomial_determinant (s p q a b c : ℝ) : 
  a^3 + s*a^2 + p*a + q = 0 →
  b^3 + s*b^2 + p*b + q = 0 →
  c^3 + s*c^2 + p*c + q = 0 →
  Matrix.det ![![s + a, 1, 1], ![1, s + b, 1], ![1, 1, s + c]] = s^3 + s*p - q - 2*s - 2*(p - s) := by
  sorry

end cubic_polynomial_determinant_l1464_146473


namespace sum_of_squares_ratio_l1464_146458

theorem sum_of_squares_ratio (x y z a b c : ℝ) 
  (h1 : x/a + y/b + z/c = 3) 
  (h2 : a/x + b/y + c/z = -3) : 
  x^2/a^2 + y^2/b^2 + z^2/c^2 = 15 := by sorry

end sum_of_squares_ratio_l1464_146458


namespace original_number_proof_l1464_146487

theorem original_number_proof (x : ℝ) : x * 1.4 = 1680 ↔ x = 1200 := by
  sorry

end original_number_proof_l1464_146487


namespace system_of_equations_proof_l1464_146437

theorem system_of_equations_proof (a b c d : ℂ) 
  (eq1 : a - b - c + d = 12)
  (eq2 : a + b - c - d = 6)
  (eq3 : 2*a + c - d = 15) :
  (b - d)^2 = 9 := by sorry

end system_of_equations_proof_l1464_146437


namespace ellipse_chord_ratio_range_l1464_146401

/-- Define an ellipse with given properties -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_ab : a > b ∧ b > 0

/-- Define a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Define a line with slope k passing through a point -/
structure Line where
  k : ℝ
  p : Point
  h_k : k ≠ 0

/-- Theorem statement -/
theorem ellipse_chord_ratio_range (C : Ellipse) (F : Point) (l : Line) :
  F.x = 1 ∧ F.y = 0 ∧ l.p = F →
  ∃ (B₁ B₂ M N D P : Point),
    -- B₁ and B₂ are endpoints of minor axis
    B₁.x = 0 ∧ B₂.x = 0 ∧ B₁.y = -C.b ∧ B₂.y = C.b ∧
    -- Condition on FB₁ · FB₂
    (F.x - B₁.x) * (F.x - B₂.x) + (F.y - B₁.y) * (F.y - B₂.y) = -C.a ∧
    -- M and N are intersections of l and C
    (M.y - F.y = l.k * (M.x - F.x) ∧ M.x^2 / C.a^2 + M.y^2 / C.b^2 = 1) ∧
    (N.y - F.y = l.k * (N.x - F.x) ∧ N.x^2 / C.a^2 + N.y^2 / C.b^2 = 1) ∧
    -- P is midpoint of MN
    P.x = (M.x + N.x) / 2 ∧ P.y = (M.y + N.y) / 2 ∧
    -- D is on x-axis and PD is perpendicular to MN
    D.y = 0 ∧ (P.y - D.y) * (N.x - M.x) = -(P.x - D.x) * (N.y - M.y) →
    -- Conclusion: range of DP/MN
    ∀ r : ℝ, (r = (Real.sqrt ((P.x - D.x)^2 + (P.y - D.y)^2)) /
               (Real.sqrt ((M.x - N.x)^2 + (M.y - N.y)^2))) →
      0 < r ∧ r < 1/4 := by
  sorry

end ellipse_chord_ratio_range_l1464_146401


namespace incorrect_deduction_l1464_146492

/-- Definition of an exponential function -/
def IsExponentialFunction (f : ℝ → ℝ) : Prop :=
  ∃ a : ℝ, a > 1 ∧ ∀ x, f x = a^x

/-- Definition of a power function -/
def IsPowerFunction (f : ℝ → ℝ) : Prop :=
  ∃ α : ℝ, α > 1 ∧ ∀ x, f x = x^α

/-- Definition of an increasing function -/
def IsIncreasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x < f y

/-- The main theorem -/
theorem incorrect_deduction :
  (∀ f : ℝ → ℝ, IsExponentialFunction f → IsIncreasing f) →
  ¬(∀ f : ℝ → ℝ, IsPowerFunction f → IsIncreasing f) :=
by sorry

end incorrect_deduction_l1464_146492


namespace quadrilateral_area_l1464_146474

/-- A quadrilateral with vertices at (3,-1), (-1,4), (2,3), and (9,9) -/
def Quadrilateral : List (ℝ × ℝ) := [(3, -1), (-1, 4), (2, 3), (9, 9)]

/-- One side of the quadrilateral is horizontal -/
axiom horizontal_side : ∃ (a b : ℝ) (y : ℝ), ((a, y) ∈ Quadrilateral ∧ (b, y) ∈ Quadrilateral) ∧ a ≠ b

/-- The area of the quadrilateral -/
def area : ℝ := 22.5

/-- Theorem: The area of the quadrilateral is 22.5 -/
theorem quadrilateral_area : 
  let vertices := Quadrilateral
  area = (1/2) * abs (
    (vertices[0].1 * vertices[1].2 + vertices[1].1 * vertices[2].2 + 
     vertices[2].1 * vertices[3].2 + vertices[3].1 * vertices[0].2) - 
    (vertices[1].1 * vertices[0].2 + vertices[2].1 * vertices[1].2 + 
     vertices[3].1 * vertices[2].2 + vertices[0].1 * vertices[3].2)
  ) := by sorry

end quadrilateral_area_l1464_146474


namespace square_sum_equation_l1464_146464

theorem square_sum_equation (x y : ℝ) : 
  (x^2 + y^2)^2 = x^2 + y^2 + 12 → x^2 + y^2 = 4 := by
  sorry

end square_sum_equation_l1464_146464


namespace triangle_is_right_angle_l1464_146412

/-- 
Given a triangle ABC where a, b, and c are the lengths of the sides opposite to angles A, B, and C respectively,
if 1 + cos A = (b + c) / c, then the triangle is a right triangle.
-/
theorem triangle_is_right_angle 
  (a b c : ℝ) 
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0)
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b)
  (h_cos : 1 + Real.cos A = (b + c) / c)
  : a^2 + b^2 = c^2 := by
  sorry

end triangle_is_right_angle_l1464_146412


namespace quadratic_equation_solution_l1464_146433

theorem quadratic_equation_solution :
  let f : ℝ → ℝ := λ x => 3 * x^2 + 12 * x
  ∀ x : ℝ, f x = 0 ↔ x = 0 ∨ x = -4 := by
  sorry

end quadratic_equation_solution_l1464_146433


namespace chen_recorded_steps_l1464_146450

/-- The standard number of steps for the walking activity -/
def standard : ℕ := 5000

/-- The function to calculate the recorded steps -/
def recorded_steps (actual_steps : ℕ) : ℤ :=
  (actual_steps : ℤ) - standard

/-- Theorem stating that 4800 actual steps should be recorded as -200 -/
theorem chen_recorded_steps :
  recorded_steps 4800 = -200 := by sorry

end chen_recorded_steps_l1464_146450


namespace babies_age_sum_l1464_146453

def lioness_age : ℕ := 12

theorem babies_age_sum (hyena_age : ℕ) (lioness_baby_age : ℕ) (hyena_baby_age : ℕ)
  (h1 : lioness_age = 2 * hyena_age)
  (h2 : lioness_baby_age = lioness_age / 2)
  (h3 : hyena_baby_age = hyena_age / 2) :
  lioness_baby_age + 5 + hyena_baby_age + 5 = 19 := by
  sorry

end babies_age_sum_l1464_146453


namespace probability_of_blank_in_specific_lottery_l1464_146499

/-- The probability of getting a blank in a lottery with prizes and blanks. -/
def probability_of_blank (prizes : ℕ) (blanks : ℕ) : ℚ :=
  blanks / (prizes + blanks)

/-- Theorem stating that the probability of getting a blank in a lottery 
    with 10 prizes and 25 blanks is 5/7. -/
theorem probability_of_blank_in_specific_lottery : 
  probability_of_blank 10 25 = 5 / 7 := by
  sorry

end probability_of_blank_in_specific_lottery_l1464_146499


namespace equal_roots_quadratic_l1464_146421

theorem equal_roots_quadratic (m : ℝ) : 
  (∃ x : ℝ, x^2 - 4*x + 1 + 2*m = 0 ∧ 
   ∀ y : ℝ, y^2 - 4*y + 1 + 2*m = 0 → y = x) → 
  m = 3/2 := by
sorry

end equal_roots_quadratic_l1464_146421


namespace chessboard_division_exists_l1464_146432

-- Define a chessboard piece
structure ChessboardPiece where
  total_squares : ℕ
  black_squares : ℕ

-- Define a chessboard division
structure ChessboardDivision where
  piece1 : ChessboardPiece
  piece2 : ChessboardPiece

-- Define the property of being a valid chessboard division
def is_valid_division (d : ChessboardDivision) : Prop :=
  d.piece1.total_squares + d.piece2.total_squares = 64 ∧
  d.piece1.total_squares = d.piece2.total_squares + 4 ∧
  d.piece2.black_squares = d.piece1.black_squares + 4 ∧
  d.piece1.black_squares + d.piece2.black_squares = 32

-- Theorem statement
theorem chessboard_division_exists : ∃ d : ChessboardDivision, is_valid_division d :=
sorry

end chessboard_division_exists_l1464_146432


namespace statement_C_is_incorrect_l1464_146449

theorem statement_C_is_incorrect : ∃ (p q : Prop), ¬(p ∧ q) ∧ (p ∨ q) := by
  sorry

end statement_C_is_incorrect_l1464_146449


namespace train_passing_jogger_l1464_146409

/-- Time for a train to pass a jogger given their speeds and initial positions -/
theorem train_passing_jogger 
  (jogger_speed : ℝ) 
  (train_speed : ℝ) 
  (initial_distance : ℝ) 
  (train_length : ℝ) 
  (h1 : jogger_speed = 9 * (1000 / 3600))  -- 9 kmph in m/s
  (h2 : train_speed = 45 * (1000 / 3600))  -- 45 kmph in m/s
  (h3 : initial_distance = 240)
  (h4 : train_length = 120) : 
  (initial_distance + train_length) / (train_speed - jogger_speed) = 36 := by
  sorry


end train_passing_jogger_l1464_146409


namespace stations_between_cities_l1464_146470

theorem stations_between_cities (n : ℕ) : 
  (((n + 2) * (n + 1)) / 2 = 132) → n = 10 := by
  sorry

end stations_between_cities_l1464_146470


namespace cow_chicken_problem_l1464_146451

theorem cow_chicken_problem (num_cows num_chickens : ℕ) : 
  (4 * num_cows + 2 * num_chickens = 2 * (num_cows + num_chickens) + 14) → 
  num_cows = 7 := by
  sorry

end cow_chicken_problem_l1464_146451


namespace min_sum_with_constraint_l1464_146416

/-- For any real number p > 1, the minimum value of x + y, where x and y satisfy the equation
    (x + √(1 + x²))(y + √(1 + y²)) = p, is (p - 1) / √p. -/
theorem min_sum_with_constraint (p : ℝ) (hp : p > 1) :
  (∃ (x y : ℝ), (x + Real.sqrt (1 + x^2)) * (y + Real.sqrt (1 + y^2)) = p) →
  (∀ (x y : ℝ), (x + Real.sqrt (1 + x^2)) * (y + Real.sqrt (1 + y^2)) = p → 
    x + y ≥ (p - 1) / Real.sqrt p) ∧
  (∃ (x y : ℝ), (x + Real.sqrt (1 + x^2)) * (y + Real.sqrt (1 + y^2)) = p ∧
    x + y = (p - 1) / Real.sqrt p) :=
by sorry

end min_sum_with_constraint_l1464_146416


namespace circle_area_isosceles_triangle_l1464_146488

/-- The area of a circle passing through the vertices of an isosceles triangle -/
theorem circle_area_isosceles_triangle (a b c : ℝ) (h1 : a = 4) (h2 : b = 4) (h3 : c = 3) :
  ∃ (r : ℝ), r > 0 ∧ π * r^2 = (64 / 13.75) * π :=
sorry

end circle_area_isosceles_triangle_l1464_146488


namespace average_departure_time_l1464_146428

def minutes_after_noon (hour : ℕ) (minute : ℕ) : ℕ :=
  (hour - 12) * 60 + minute

def passing_time : ℕ := minutes_after_noon 15 11
def alice_arrival : ℕ := minutes_after_noon 15 19
def bob_arrival : ℕ := minutes_after_noon 15 29

theorem average_departure_time :
  let alice_departure := alice_arrival - (alice_arrival - passing_time)
  let bob_departure := bob_arrival - (bob_arrival - passing_time)
  (alice_departure + bob_departure) / 2 = 179 := by
sorry

end average_departure_time_l1464_146428


namespace modular_arithmetic_problem_l1464_146491

theorem modular_arithmetic_problem :
  ∃ (a b : ℤ), (3 * a + 9 * b) % 63 = 45 ∧ (7 * a) % 63 = 1 ∧ (13 * b) % 63 = 1 :=
by sorry

end modular_arithmetic_problem_l1464_146491


namespace least_positive_angle_solution_l1464_146493

theorem least_positive_angle_solution (θ : Real) : 
  (θ > 0 ∧ θ < 360 ∧ Real.cos (10 * Real.pi / 180) = Real.sin (30 * Real.pi / 180) + Real.sin (θ * Real.pi / 180)) →
  θ = 80 := by
sorry

end least_positive_angle_solution_l1464_146493


namespace equal_water_levels_l1464_146495

/-- Represents a pool with initial height and drain time -/
structure Pool where
  initial_height : ℝ
  drain_time : ℝ

/-- The time when water levels in two pools become equal -/
def equal_level_time (pool_a pool_b : Pool) : ℝ :=
  1 -- The actual value we want to prove

theorem equal_water_levels (pool_a pool_b : Pool) :
  pool_b.initial_height = 1.5 * pool_a.initial_height →
  pool_a.drain_time = 2 →
  pool_b.drain_time = 1.5 →
  equal_level_time pool_a pool_b = 1 := by
  sorry

#check equal_water_levels

end equal_water_levels_l1464_146495


namespace subcommittee_formation_count_l1464_146406

/-- The number of ways to choose k items from n items -/
def binomial (n k : ℕ) : ℕ := (Nat.factorial n) / (Nat.factorial k * Nat.factorial (n - k))

/-- The total number of Republicans in the committee -/
def total_republicans : ℕ := 10

/-- The total number of Democrats in the committee -/
def total_democrats : ℕ := 8

/-- The number of Republicans in the subcommittee -/
def subcommittee_republicans : ℕ := 4

/-- The number of Democrats in the subcommittee -/
def subcommittee_democrats : ℕ := 3

/-- The number of ways to form the subcommittee -/
def subcommittee_combinations : ℕ := 
  (binomial total_republicans subcommittee_republicans) * 
  (binomial total_democrats subcommittee_democrats)

theorem subcommittee_formation_count : subcommittee_combinations = 11760 := by
  sorry

end subcommittee_formation_count_l1464_146406


namespace minimize_F_l1464_146472

/-- The optimization problem -/
def OptimizationProblem (x₁ x₂ x₃ x₄ x₅ : ℝ) : Prop :=
  x₁ ≥ 0 ∧ x₂ ≥ 0 ∧
  -2 * x₁ + x₂ + x₃ = 2 ∧
  x₁ - 2 * x₂ + x₄ = 2 ∧
  x₁ + x₂ + x₅ = 5

/-- The objective function -/
def F (x₁ x₂ : ℝ) : ℝ := x₂ - x₁

/-- The theorem stating the minimum value of F and the point where it's achieved -/
theorem minimize_F :
  ∃ (x₁ x₂ x₃ x₄ x₅ : ℝ),
    OptimizationProblem x₁ x₂ x₃ x₄ x₅ ∧
    F x₁ x₂ = -3 ∧
    x₁ = 4 ∧ x₂ = 1 ∧ x₃ = 9 ∧ x₄ = 0 ∧ x₅ = 0 ∧
    ∀ (y₁ y₂ y₃ y₄ y₅ : ℝ), OptimizationProblem y₁ y₂ y₃ y₄ y₅ → F y₁ y₂ ≥ -3 := by
  sorry

end minimize_F_l1464_146472


namespace alien_tree_age_l1464_146448

/-- Converts a base-8 number to base-10 --/
def base8ToBase10 (hundreds : Nat) (tens : Nat) (ones : Nat) : Nat :=
  hundreds * 8^2 + tens * 8^1 + ones * 8^0

/-- The age of the tree in the alien's base-8 system --/
def alienAge : Nat := base8ToBase10 3 6 7

theorem alien_tree_age : alienAge = 247 := by
  sorry

end alien_tree_age_l1464_146448


namespace ping_pong_balls_sold_l1464_146411

/-- Calculates the number of ping pong balls sold in a shop -/
theorem ping_pong_balls_sold
  (initial_baseballs : ℕ)
  (initial_ping_pong : ℕ)
  (baseballs_sold : ℕ)
  (total_left : ℕ)
  (h1 : initial_baseballs = 2754)
  (h2 : initial_ping_pong = 1938)
  (h3 : baseballs_sold = 1095)
  (h4 : total_left = 3021)
  (h5 : total_left = initial_baseballs + initial_ping_pong - baseballs_sold - (initial_ping_pong - ping_pong_sold))
  : ping_pong_sold = 576 :=
by
  sorry

#check ping_pong_balls_sold

end ping_pong_balls_sold_l1464_146411


namespace zelda_success_probability_l1464_146417

theorem zelda_success_probability 
  (p_xavier : ℝ) 
  (p_yvonne : ℝ) 
  (p_xy_not_z : ℝ) 
  (h1 : p_xavier = 1/4)
  (h2 : p_yvonne = 2/3)
  (h3 : p_xy_not_z = 0.0625)
  (h4 : p_xy_not_z = p_xavier * p_yvonne * (1 - p_zelda)) :
  p_zelda = 5/8 := by
sorry

end zelda_success_probability_l1464_146417


namespace question_probabilities_l1464_146476

def total_questions : ℕ := 5
def algebra_questions : ℕ := 2
def geometry_questions : ℕ := 3

theorem question_probabilities :
  let prob_algebra_then_geometry := (algebra_questions : ℚ) / total_questions * 
                                    (geometry_questions : ℚ) / (total_questions - 1)
  let prob_geometry_given_algebra := (geometry_questions : ℚ) / (total_questions - 1)
  prob_algebra_then_geometry = 3 / 10 ∧ prob_geometry_given_algebra = 3 / 4 := by
  sorry

end question_probabilities_l1464_146476


namespace b_25_mod_35_l1464_146455

/-- b_n is the integer obtained by writing all integers from 1 to n from left to right, each repeated twice -/
def b (n : ℕ) : ℕ :=
  -- Definition of b_n goes here
  sorry

/-- The remainder when b_25 is divided by 35 is 6 -/
theorem b_25_mod_35 : b 25 % 35 = 6 := by
  sorry

end b_25_mod_35_l1464_146455


namespace vince_earnings_per_head_l1464_146497

/-- Represents Vince's hair salon business model -/
structure HairSalon where
  earningsPerHead : ℝ
  customersPerMonth : ℕ
  monthlyRentAndElectricity : ℝ
  recreationPercentage : ℝ
  monthlySavings : ℝ

/-- Theorem stating that Vince's earnings per head is $72 -/
theorem vince_earnings_per_head (salon : HairSalon)
    (h1 : salon.customersPerMonth = 80)
    (h2 : salon.monthlyRentAndElectricity = 280)
    (h3 : salon.recreationPercentage = 0.2)
    (h4 : salon.monthlySavings = 872)
    (h5 : salon.earningsPerHead * ↑salon.customersPerMonth * (1 - salon.recreationPercentage) =
          salon.earningsPerHead * ↑salon.customersPerMonth - salon.monthlyRentAndElectricity - salon.monthlySavings) :
    salon.earningsPerHead = 72 := by
  sorry

#check vince_earnings_per_head

end vince_earnings_per_head_l1464_146497


namespace orthogonal_vectors_k_value_l1464_146407

/-- Given vector a = (3, -4), a + 2b = (k+1, k-4), and a is orthogonal to b, prove that k = -6 -/
theorem orthogonal_vectors_k_value (k : ℝ) (b : ℝ × ℝ) : 
  let a : ℝ × ℝ := (3, -4)
  (a.1 + 2 * b.1 = k + 1 ∧ a.2 + 2 * b.2 = k - 4) → 
  (a.1 * b.1 + a.2 * b.2 = 0) → 
  k = -6 := by
  sorry

end orthogonal_vectors_k_value_l1464_146407


namespace solution_set_implies_k_value_l1464_146460

theorem solution_set_implies_k_value (k : ℝ) : 
  (∀ x : ℝ, |k * x - 4| ≤ 2 ↔ 1 ≤ x ∧ x ≤ 3) → k = 2 := by
  sorry

end solution_set_implies_k_value_l1464_146460


namespace price_of_zinc_l1464_146483

/-- Given the price of copper, the total weight of brass, the selling price of brass,
    and the amount of copper used, calculate the price of zinc per pound. -/
theorem price_of_zinc 
  (price_copper : ℚ)
  (total_weight : ℚ)
  (selling_price : ℚ)
  (copper_used : ℚ)
  (h1 : price_copper = 65/100)
  (h2 : total_weight = 70)
  (h3 : selling_price = 45/100)
  (h4 : copper_used = 30)
  : ∃ (price_zinc : ℚ), price_zinc = 30/100 := by
  sorry

#check price_of_zinc

end price_of_zinc_l1464_146483


namespace number_problem_l1464_146404

theorem number_problem : ∃ x : ℝ, x = 40 ∧ 0.8 * x > (4/5) * 25 + 12 := by
  sorry

end number_problem_l1464_146404


namespace fraction_equality_l1464_146427

theorem fraction_equality (a b : ℝ) (h : a / b = 2 / 3) : a / (a - b) = -2 := by
  sorry

end fraction_equality_l1464_146427


namespace student_count_l1464_146422

theorem student_count (n : ℕ) 
  (yellow : ℕ) (red : ℕ) (blue : ℕ) 
  (yellow_blue : ℕ) (yellow_red : ℕ) (blue_red : ℕ)
  (all_colors : ℕ) :
  yellow = 46 →
  red = 69 →
  blue = 104 →
  yellow_blue = 14 →
  yellow_red = 13 →
  blue_red = 19 →
  all_colors = 16 →
  n = yellow + red + blue - yellow_blue - yellow_red - blue_red + all_colors →
  n = 141 :=
by sorry

end student_count_l1464_146422


namespace condition_p_neither_sufficient_nor_necessary_for_q_l1464_146484

theorem condition_p_neither_sufficient_nor_necessary_for_q :
  ¬(∀ x : ℝ, (1 / x ≤ 1) → (x^2 - 2*x ≥ 0)) ∧
  ¬(∀ x : ℝ, (x^2 - 2*x ≥ 0) → (1 / x ≤ 1)) :=
by sorry

end condition_p_neither_sufficient_nor_necessary_for_q_l1464_146484


namespace angle_with_complement_40percent_of_supplement_l1464_146445

theorem angle_with_complement_40percent_of_supplement (x : ℝ) : 
  (90 - x = (2/5) * (180 - x)) → x = 30 := by
  sorry

end angle_with_complement_40percent_of_supplement_l1464_146445


namespace container_volume_ratio_l1464_146479

theorem container_volume_ratio (A B : ℝ) (h1 : A > 0) (h2 : B > 0) : 
  (2/3 * A = 1/2 * B) → (A / B = 3/4) := by
  sorry

end container_volume_ratio_l1464_146479


namespace gre_exam_month_l1464_146490

-- Define the months as an enumeration
inductive Month
| January | February | March | April | May | June
| July | August | September | October | November | December

def next_month : Month → Month
| Month.January => Month.February
| Month.February => Month.March
| Month.March => Month.April
| Month.April => Month.May
| Month.May => Month.June
| Month.June => Month.July
| Month.July => Month.August
| Month.August => Month.September
| Month.September => Month.October
| Month.October => Month.November
| Month.November => Month.December
| Month.December => Month.January

def months_later (start : Month) (n : Nat) : Month :=
  match n with
  | 0 => start
  | n + 1 => next_month (months_later start n)

theorem gre_exam_month (start_month : Month) (preparation_months : Nat) :
  start_month = Month.June ∧ preparation_months = 5 →
  months_later start_month preparation_months = Month.November :=
by sorry

end gre_exam_month_l1464_146490


namespace sachins_age_l1464_146413

/-- Proves that Sachin's age is 38.5 years given the conditions -/
theorem sachins_age (sachin rahul : ℝ) 
  (h1 : sachin = rahul + 7)
  (h2 : sachin / rahul = 11 / 9) : 
  sachin = 38.5 := by
  sorry

end sachins_age_l1464_146413


namespace pairwise_sums_distinct_digits_impossible_l1464_146418

theorem pairwise_sums_distinct_digits_impossible :
  ¬ ∃ (a b c d e : ℕ),
    let sums := [a + b, a + c, a + d, a + e, b + c, b + d, b + e, c + d, c + e, d + e]
    ∀ (i j : Fin 10), i ≠ j → sums[i] % 10 ≠ sums[j] % 10 := by
  sorry

#check pairwise_sums_distinct_digits_impossible

end pairwise_sums_distinct_digits_impossible_l1464_146418


namespace apple_weight_is_quarter_pound_l1464_146485

/-- The weight of a small apple in pounds -/
def apple_weight : ℝ := 0.25

/-- The cost of apples per pound in dollars -/
def cost_per_pound : ℝ := 2

/-- The total amount spent on apples in dollars -/
def total_spent : ℝ := 7

/-- The number of days the apples should last -/
def days : ℕ := 14

/-- Theorem stating that the weight of a small apple is 0.25 pounds -/
theorem apple_weight_is_quarter_pound :
  apple_weight = total_spent / (cost_per_pound * days) := by sorry

end apple_weight_is_quarter_pound_l1464_146485


namespace sugar_weighing_l1464_146436

theorem sugar_weighing (p q : ℝ) (hp : p > 0) (hq : q > 0) (hpq : p ≠ q) :
  p / q + q / p > 2 := by
  sorry

end sugar_weighing_l1464_146436


namespace qualified_products_l1464_146438

theorem qualified_products (defect_rate : ℝ) (total_items : ℕ) : 
  defect_rate = 0.005 →
  total_items = 18000 →
  ⌊(1 - defect_rate) * total_items⌋ = 17910 := by
sorry

end qualified_products_l1464_146438


namespace cube_sum_problem_l1464_146419

theorem cube_sum_problem (x y z : ℝ) 
  (sum_eq : x + y + z = 8)
  (sum_prod_eq : x*y + x*z + y*z = 17)
  (prod_eq : x*y*z = -14) :
  x^3 + y^3 + z^3 = 62 := by
  sorry

end cube_sum_problem_l1464_146419


namespace unique_number_with_property_l1464_146439

/-- A four-digit natural number -/
def FourDigitNumber (x y z w : ℕ) : ℕ := 1000 * x + 100 * y + 10 * z + w

/-- The property that the sum of the number and its digits equals 2003 -/
def HasProperty (x y z w : ℕ) : Prop :=
  FourDigitNumber x y z w + x + y + z + w = 2003

/-- The theorem stating that 1978 is the only four-digit number satisfying the property -/
theorem unique_number_with_property :
  (∃! n : ℕ, ∃ x y z w : ℕ, 
    x ≠ 0 ∧ 
    n = FourDigitNumber x y z w ∧ 
    HasProperty x y z w) ∧
  (∃ x y z w : ℕ, 
    x ≠ 0 ∧ 
    1978 = FourDigitNumber x y z w ∧ 
    HasProperty x y z w) :=
sorry

end unique_number_with_property_l1464_146439


namespace license_plate_increase_l1464_146435

theorem license_plate_increase : 
  let old_plates := 26^2 * 10^3
  let new_plates := 26^3 * 10^3 * 5
  new_plates / old_plates = 130 := by sorry

end license_plate_increase_l1464_146435


namespace total_flooring_cost_l1464_146440

/-- Represents the dimensions of a room -/
structure RoomDimensions where
  length : ℝ
  width : ℝ

/-- Calculates the area of a room given its dimensions -/
def roomArea (d : RoomDimensions) : ℝ := d.length * d.width

/-- Calculates the cost of flooring for a room given its area and slab rate -/
def roomCost (area : ℝ) (slabRate : ℝ) : ℝ := area * slabRate

/-- Theorem: The total cost of flooring for the house is Rs. 81,390 -/
theorem total_flooring_cost : 
  let room1 : RoomDimensions := ⟨5.5, 3.75⟩
  let room2 : RoomDimensions := ⟨6, 4.2⟩
  let room3 : RoomDimensions := ⟨4.8, 3.25⟩
  let slabRate1 : ℝ := 1200
  let slabRate2 : ℝ := 1350
  let slabRate3 : ℝ := 1450
  let totalCost : ℝ := 
    roomCost (roomArea room1) slabRate1 + 
    roomCost (roomArea room2) slabRate2 + 
    roomCost (roomArea room3) slabRate3
  totalCost = 81390 := by
  sorry

end total_flooring_cost_l1464_146440


namespace garden_multiplier_l1464_146496

theorem garden_multiplier (width length perimeter : ℝ) 
  (h1 : perimeter = 2 * length + 2 * width)
  (h2 : perimeter = 100)
  (h3 : length = 38)
  (h4 : ∃ m : ℝ, length = m * width + 2) :
  ∃ m : ℝ, length = m * width + 2 ∧ m = 3 := by
  sorry

end garden_multiplier_l1464_146496


namespace line_inclination_sine_l1464_146465

/-- Given a straight line 3x - 4y + 5 = 0 with angle of inclination α, prove that sin(α) = 3/5 -/
theorem line_inclination_sine (x y : ℝ) (α : ℝ) 
  (h : 3 * x - 4 * y + 5 = 0) 
  (h_incl : α = Real.arctan (3 / 4)) : 
  Real.sin α = 3 / 5 := by
  sorry

end line_inclination_sine_l1464_146465


namespace people_joined_line_l1464_146486

theorem people_joined_line (initial : ℕ) (left : ℕ) (final : ℕ) : 
  initial = 30 → left = 10 → final = 25 → final - (initial - left) = 5 := by
  sorry

end people_joined_line_l1464_146486


namespace min_sum_of_primes_l1464_146405

-- Define a function to check if a number is prime
def isPrime (n : ℕ) : Prop :=
  n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

-- Define the theorem
theorem min_sum_of_primes (m n p : ℕ) :
  isPrime m → isPrime n → isPrime p →
  m ≠ n → n ≠ p → m ≠ p →
  m + n = p →
  (∀ m' n' p' : ℕ,
    isPrime m' → isPrime n' → isPrime p' →
    m' ≠ n' → n' ≠ p' → m' ≠ p' →
    m' + n' = p' →
    m' * n' * p' ≥ m * n * p) →
  m * n * p = 30 :=
by sorry

end min_sum_of_primes_l1464_146405


namespace inequality_proof_l1464_146429

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  let H := 3 / (1/a + 1/b + 1/c)
  let G := (a * b * c) ^ (1/3)
  let A := (a + b + c) / 3
  let Q := Real.sqrt ((a^2 + b^2 + c^2) / 3)
  (A * G) / (Q * H) ≥ (27/32) ^ (1/6) := by
  sorry

end inequality_proof_l1464_146429


namespace square_vertices_not_on_arithmetic_circles_l1464_146402

theorem square_vertices_not_on_arithmetic_circles : ¬∃ (a d : ℝ), a > 0 ∧ d > 0 ∧
  ((a ^ 2 + (a + d) ^ 2 = (a + 2*d) ^ 2 + (a + 3*d) ^ 2) ∨
   (a ^ 2 + (a + 2*d) ^ 2 = (a + d) ^ 2 + (a + 3*d) ^ 2) ∨
   ((a + d) ^ 2 + (a + 2*d) ^ 2 = a ^ 2 + (a + 3*d) ^ 2)) :=
by sorry

end square_vertices_not_on_arithmetic_circles_l1464_146402


namespace arithmetic_sequence_product_l1464_146431

/-- An arithmetic sequence of integers -/
def arithmeticSequence (b : ℕ → ℤ) (d : ℤ) : Prop :=
  ∀ n : ℕ, b (n + 1) = b n + d

theorem arithmetic_sequence_product (b : ℕ → ℤ) (d : ℤ) :
  arithmeticSequence b d →
  (∀ n : ℕ, b (n + 1) > b n) →
  b 5 * b 6 = 21 →
  b 4 * b 7 = -11 := by
  sorry

end arithmetic_sequence_product_l1464_146431


namespace function_max_min_l1464_146452

def f (a : ℝ) (x : ℝ) : ℝ := x^2 + 2*a*x + a^2 - 1

theorem function_max_min (a : ℝ) :
  (∀ x, 0 ≤ x ∧ x ≤ 3 → f a x ≤ 24) ∧
  (∃ x, 0 ≤ x ∧ x ≤ 3 ∧ f a x = 24) ∧
  (∀ x, 0 ≤ x ∧ x ≤ 3 → f a x ≥ 3) ∧
  (∃ x, 0 ≤ x ∧ x ≤ 3 ∧ f a x = 3) →
  a = 2 ∨ a = -5 := by sorry

end function_max_min_l1464_146452


namespace candy_count_difference_l1464_146478

/-- The number of candies Bryan has -/
def bryan_candies : ℕ := 50

/-- The number of candies Ben has -/
def ben_candies : ℕ := 20

/-- The difference in candy count between Bryan and Ben -/
def candy_difference : ℕ := bryan_candies - ben_candies

theorem candy_count_difference :
  candy_difference = 30 :=
by sorry

end candy_count_difference_l1464_146478


namespace equilateral_triangles_count_l1464_146471

/-- Counts the number of equilateral triangles in an equilateral triangular grid -/
def count_equilateral_triangles (n : ℕ) : ℕ :=
  n * (n + 1) * (n + 2) * (n + 3) / 24

/-- The side length of the equilateral triangular grid -/
def grid_side_length : ℕ := 4

/-- Theorem: The number of equilateral triangles in a grid of side length 4 is 35 -/
theorem equilateral_triangles_count :
  count_equilateral_triangles grid_side_length = 35 := by
  sorry

#eval count_equilateral_triangles grid_side_length

end equilateral_triangles_count_l1464_146471


namespace sum_of_coordinates_after_reflection_l1464_146446

def reflect_over_y_axis (p : ℝ × ℝ) : ℝ × ℝ := (-p.1, p.2)

theorem sum_of_coordinates_after_reflection (x : ℝ) :
  let A : ℝ × ℝ := (x, 6)
  let B : ℝ × ℝ := reflect_over_y_axis A
  A.1 + A.2 + B.1 + B.2 = 12 := by
sorry

end sum_of_coordinates_after_reflection_l1464_146446


namespace eccentricity_of_conic_l1464_146468

/-- The conic section defined by the equation 6x^2 + 4xy + 9y^2 = 20 -/
def conic_section (x y : ℝ) : Prop :=
  6 * x^2 + 4 * x * y + 9 * y^2 = 20

/-- The eccentricity of a conic section -/
def eccentricity (c : (ℝ → ℝ → Prop)) : ℝ := sorry

/-- Theorem: The eccentricity of the given conic section is √2/2 -/
theorem eccentricity_of_conic : eccentricity conic_section = Real.sqrt 2 / 2 := by
  sorry

end eccentricity_of_conic_l1464_146468


namespace card_value_decrease_l1464_146469

/-- Proves that if a value decreases by x% in the first year and 10% in the second year, 
    and the total decrease over two years is 37%, then x = 30. -/
theorem card_value_decrease (x : ℝ) : 
  (1 - x / 100) * (1 - 0.1) = 1 - 0.37 → x = 30 := by
  sorry

end card_value_decrease_l1464_146469


namespace circle_radius_ratio_l1464_146400

theorem circle_radius_ratio (r₁ r₂ : ℝ) (h : r₁ > 0 ∧ r₂ > 0) 
  (area_ratio : π * r₂^2 = 4 * π * r₁^2) : 
  r₁ / r₂ = 1 / Real.sqrt 5 := by
sorry

end circle_radius_ratio_l1464_146400


namespace min_cost_for_range_l1464_146459

/-- The cost of a "yes" answer in rubles -/
def yes_cost : ℕ := 2

/-- The cost of a "no" answer in rubles -/
def no_cost : ℕ := 1

/-- The range of possible hidden numbers -/
def number_range : Set ℕ := Finset.range 144

/-- The minimum cost function for guessing a number in a given set -/
noncomputable def min_cost (S : Set ℕ) : ℕ :=
  sorry

/-- The theorem stating that the minimum cost to guess any number in [1, 144] is 11 rubles -/
theorem min_cost_for_range : min_cost number_range = 11 :=
  sorry

end min_cost_for_range_l1464_146459


namespace tangency_distance_value_l1464_146426

/-- Configuration of four circles where three small circles of radius 2 are externally
    tangent to each other and internally tangent to a larger circle -/
structure CircleConfiguration where
  -- Radius of each small circle
  small_radius : ℝ
  -- Center of the large circle
  large_center : ℝ × ℝ
  -- Centers of the three small circles
  small_centers : Fin 3 → ℝ × ℝ
  -- The three small circles are externally tangent to each other
  small_circles_tangent : ∀ (i j : Fin 3), i ≠ j →
    ‖small_centers i - small_centers j‖ = 2 * small_radius
  -- The three small circles are internally tangent to the large circle
  large_circle_tangent : ∀ (i : Fin 3),
    ‖large_center - small_centers i‖ = ‖large_center - small_centers 0‖

/-- The distance from the center of the large circle to the point of tangency
    on one of the small circles in the given configuration -/
def tangency_distance (config : CircleConfiguration) : ℝ :=
  ‖config.large_center - config.small_centers 0‖ - config.small_radius

/-- Theorem stating that the tangency distance is equal to 2√3 - 2 -/
theorem tangency_distance_value (config : CircleConfiguration) 
    (h : config.small_radius = 2) : 
    tangency_distance config = 2 * Real.sqrt 3 - 2 := by
  sorry

end tangency_distance_value_l1464_146426


namespace trigonometric_identity_l1464_146494

theorem trigonometric_identity (α : ℝ) :
  Real.cos (3 / 2 * Real.pi + 4 * α) + Real.sin (3 * Real.pi - 8 * α) - Real.sin (4 * Real.pi - 12 * α) =
  4 * Real.cos (2 * α) * Real.cos (4 * α) * Real.sin (6 * α) := by
  sorry

end trigonometric_identity_l1464_146494


namespace machine_work_time_l1464_146442

theorem machine_work_time (x : ℝ) (h1 : x > 0) 
  (h2 : 1/x + 1/2 + 1/6 = 11/12) : x = 4 := by
  sorry

#check machine_work_time

end machine_work_time_l1464_146442


namespace food_lasts_14_days_l1464_146481

/-- Represents the amount of food each dog consumes per meal in grams -/
def dog_food_per_meal : List ℕ := [250, 350, 450, 550, 300, 400]

/-- Number of meals per day -/
def meals_per_day : ℕ := 3

/-- Weight of each sack in kilograms -/
def sack_weight_kg : ℕ := 50

/-- Number of sacks -/
def num_sacks : ℕ := 2

/-- Conversion factor from kilograms to grams -/
def kg_to_g : ℕ := 1000

theorem food_lasts_14_days :
  let total_food_per_meal := dog_food_per_meal.sum
  let daily_consumption := total_food_per_meal * meals_per_day
  let total_food := num_sacks * sack_weight_kg * kg_to_g
  (total_food / daily_consumption : ℕ) = 14 := by sorry

end food_lasts_14_days_l1464_146481


namespace fraction_product_equivalence_l1464_146441

theorem fraction_product_equivalence (f g : ℝ → ℝ) :
  ∀ x : ℝ, g x ≠ 0 → (f x / g x > 0 ↔ f x * g x > 0) := by
  sorry

end fraction_product_equivalence_l1464_146441


namespace problem_solution_l1464_146456

theorem problem_solution (a b : ℝ) (h1 : |a| = 5) (h2 : b = -2) (h3 : a * b > 0) :
  a + b = -7 := by
sorry

end problem_solution_l1464_146456


namespace intersection_A_B_l1464_146480

-- Define set A
def A : Set ℝ := {x | |x - 1| < 2}

-- Define set B
def B : Set ℝ := {y | ∃ x ∈ Set.Icc 0 2, y = 2^x}

-- Theorem statement
theorem intersection_A_B : A ∩ B = Set.Ici 1 ∩ Set.Iio 3 := by sorry

end intersection_A_B_l1464_146480


namespace problem_solution_l1464_146498

theorem problem_solution : (2023^2 - 2023 - 1) / 2023 = 2022 - 1 / 2023 := by
  sorry

end problem_solution_l1464_146498


namespace museum_visitors_l1464_146423

theorem museum_visitors (yesterday : ℕ) (today_increase : ℕ) : 
  yesterday = 247 → today_increase = 131 → 
  yesterday + (yesterday + today_increase) = 625 := by
sorry

end museum_visitors_l1464_146423


namespace exists_quadratic_polynomial_with_constant_negative_two_l1464_146415

/-- A quadratic polynomial in x and y with constant term -2 -/
def quadratic_polynomial (x y : ℝ) : ℝ := 15 * x^2 - y - 2

/-- Theorem stating the existence of a quadratic polynomial in x and y with constant term -2 -/
theorem exists_quadratic_polynomial_with_constant_negative_two :
  ∃ (f : ℝ → ℝ → ℝ), (∃ (a b c d e : ℝ), ∀ (x y : ℝ), 
    f x y = a * x^2 + b * x * y + c * y^2 + d * x + e * y - 2) :=
sorry

end exists_quadratic_polynomial_with_constant_negative_two_l1464_146415


namespace smallest_solution_quadratic_equation_l1464_146467

theorem smallest_solution_quadratic_equation :
  let f : ℝ → ℝ := λ x => 12 * x^2 - 58 * x + 70
  ∃ x : ℝ, f x = 0 ∧ (∀ y : ℝ, f y = 0 → x ≤ y) ∧ x = 7/3 := by
  sorry

end smallest_solution_quadratic_equation_l1464_146467


namespace largest_digit_change_l1464_146443

def original_sum : ℕ := 2570
def correct_sum : ℕ := 2580
def num1 : ℕ := 725
def num2 : ℕ := 864
def num3 : ℕ := 991

theorem largest_digit_change :
  ∃ (d : ℕ), d ≤ 9 ∧ 
  (num1 + num2 + (num3 - 10) = correct_sum) ∧
  (∀ (d' : ℕ), d' > d → 
    (num1 + num2 + num3 - d' * 10 ≠ correct_sum ∧ 
     num1 + (num2 - d' * 10) + num3 ≠ correct_sum ∧
     (num1 - d' * 10) + num2 + num3 ≠ correct_sum)) :=
by sorry

end largest_digit_change_l1464_146443
