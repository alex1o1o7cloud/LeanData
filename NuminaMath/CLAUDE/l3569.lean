import Mathlib

namespace NUMINAMATH_CALUDE_masha_wins_l3569_356983

/-- Represents a pile of candies -/
structure Pile :=
  (size : ℕ)

/-- Represents the game state -/
structure GameState :=
  (piles : List Pile)

/-- Calculates the number of moves required to split a pile into single candies -/
def movesForPile (p : Pile) : ℕ :=
  p.size - 1

/-- Calculates the total number of moves for all piles -/
def totalMoves (gs : GameState) : ℕ :=
  gs.piles.map movesForPile |>.sum

/-- Determines if the first player wins given a game state -/
def firstPlayerWins (gs : GameState) : Prop :=
  Odd (totalMoves gs)

/-- Theorem: Masha (first player) wins the candy splitting game -/
theorem masha_wins :
  let initialState : GameState := ⟨[⟨10⟩, ⟨20⟩, ⟨30⟩]⟩
  firstPlayerWins initialState := by
  sorry


end NUMINAMATH_CALUDE_masha_wins_l3569_356983


namespace NUMINAMATH_CALUDE_total_is_450_l3569_356995

/-- The number of classes Kelvin has -/
def kelvin_classes : ℕ := 90

/-- The number of vacations Grant has -/
def grant_vacations : ℕ := 4 * kelvin_classes

/-- The total number of vacations and classes for Grant and Kelvin -/
def total_vacations_and_classes : ℕ := grant_vacations + kelvin_classes

theorem total_is_450 : total_vacations_and_classes = 450 := by
  sorry

end NUMINAMATH_CALUDE_total_is_450_l3569_356995


namespace NUMINAMATH_CALUDE_existence_of_common_source_l3569_356913

/-- The type of positive integers. -/
def PositiveInt := { n : ℕ // n > 0 }

/-- Predicate to check if a number contains the digit 5. -/
def containsFive (n : PositiveInt) : Prop :=
  ∃ d, d ∈ n.val.digits 10 ∧ d = 5

/-- The process of replacing two consecutive digits with the last digit of their product. -/
def replaceDigits (n : PositiveInt) : PositiveInt :=
  sorry

/-- A number m is obtainable from n if there exists a finite sequence of replaceDigits operations. -/
def isObtainable (m n : PositiveInt) : Prop :=
  sorry

/-- Main theorem: For any finite set of positive integers without digit 5, 
    there exists a positive integer from which all elements are obtainable. -/
theorem existence_of_common_source (S : Finset PositiveInt) 
  (h : ∀ s ∈ S, ¬containsFive s) : 
  ∃ N : PositiveInt, ∀ s ∈ S, isObtainable s N :=
sorry

end NUMINAMATH_CALUDE_existence_of_common_source_l3569_356913


namespace NUMINAMATH_CALUDE_factorial_equation_solutions_l3569_356912

theorem factorial_equation_solutions :
  ∀ x y z : ℕ, 2^x + 5^y + 63 = z.factorial → 
    ((x = 5 ∧ y = 2 ∧ z = 5) ∨ (x = 4 ∧ y = 4 ∧ z = 6)) := by
  sorry

end NUMINAMATH_CALUDE_factorial_equation_solutions_l3569_356912


namespace NUMINAMATH_CALUDE_vector_operation_l3569_356927

theorem vector_operation (a b : ℝ × ℝ) (h1 : a = (3, 2)) (h2 : b = (0, -1)) :
  2 • b - a = (-3, -4) := by sorry

end NUMINAMATH_CALUDE_vector_operation_l3569_356927


namespace NUMINAMATH_CALUDE_f_properties_l3569_356953

def f (x : ℝ) : ℝ := x^3 + x^2 - 8*x + 6

theorem f_properties :
  (∀ x, deriv f x = 3*x^2 + 2*x - 8) ∧
  deriv f (-2) = 0 ∧
  deriv f 1 = -3 ∧
  f 1 = 0 ∧
  (∀ x, x < -2 ∨ x > 4/3 → deriv f x > 0) ∧
  (∀ x, -2 < x ∧ x < 4/3 → deriv f x < 0) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l3569_356953


namespace NUMINAMATH_CALUDE_subset_implies_m_leq_two_l3569_356941

theorem subset_implies_m_leq_two (m : ℝ) : 
  (∀ x, x < m → x < 2) → m ≤ 2 := by sorry

end NUMINAMATH_CALUDE_subset_implies_m_leq_two_l3569_356941


namespace NUMINAMATH_CALUDE_cottage_rental_cost_l3569_356978

theorem cottage_rental_cost (hourly_rate : ℝ) (hours : ℝ) (num_people : ℝ) 
  (h1 : hourly_rate = 5)
  (h2 : hours = 8)
  (h3 : num_people = 2) :
  (hourly_rate * hours) / num_people = 20 := by
  sorry

end NUMINAMATH_CALUDE_cottage_rental_cost_l3569_356978


namespace NUMINAMATH_CALUDE_yellow_red_difference_l3569_356910

/-- The number of houses Isabella has -/
structure Houses where
  green : ℕ
  yellow : ℕ
  red : ℕ

/-- Conditions for Isabella's houses -/
def isabellaHouses (h : Houses) : Prop :=
  h.green = 3 * h.yellow ∧
  h.green = 90 ∧
  h.green + h.red = 160

/-- Theorem: Isabella has 40 fewer yellow houses than red houses -/
theorem yellow_red_difference (h : Houses) (hcond : isabellaHouses h) :
  h.red - h.yellow = 40 := by
  sorry

end NUMINAMATH_CALUDE_yellow_red_difference_l3569_356910


namespace NUMINAMATH_CALUDE_tangent_circle_equation_l3569_356972

/-- The circle C -/
def circle_C (x y : ℝ) : Prop := (x - 1)^2 + (y - 2)^2 = 1

/-- The point P -/
def point_P : ℝ × ℝ := (-1, 0)

/-- The center of circle C -/
def center_C : ℝ × ℝ := (1, 2)

/-- The equation of the circle passing through the tangency points and the center of C -/
def target_circle (x y : ℝ) : Prop := x^2 + (y - 1)^2 = 2

/-- The theorem stating that the target circle passes through the tangency points and the center of C -/
theorem tangent_circle_equation : 
  ∃ (A B : ℝ × ℝ), 
    (∀ x y, circle_C x y → ((x - point_P.1)^2 + (y - point_P.2)^2 ≤ (A.1 - point_P.1)^2 + (A.2 - point_P.2)^2)) ∧
    (∀ x y, circle_C x y → ((x - point_P.1)^2 + (y - point_P.2)^2 ≤ (B.1 - point_P.1)^2 + (B.2 - point_P.2)^2)) ∧
    circle_C A.1 A.2 ∧ 
    circle_C B.1 B.2 ∧
    target_circle A.1 A.2 ∧
    target_circle B.1 B.2 ∧
    target_circle center_C.1 center_C.2 :=
sorry

end NUMINAMATH_CALUDE_tangent_circle_equation_l3569_356972


namespace NUMINAMATH_CALUDE_complex_expression_equality_logarithmic_expression_equality_l3569_356986

-- Define the logarithm base 2
noncomputable def lg (x : ℝ) := Real.log x / Real.log 2

theorem complex_expression_equality : 
  (1) * (2^(1/3) * 3^(1/2))^6 + (2 * 2^(1/2))^(4/3) - 4 * (16/49)^(-1/2) - 2^(1/4) * 8^0.25 - (-2009)^0 = 100 :=
sorry

theorem logarithmic_expression_equality :
  2 * (lg (2^(1/2)))^2 + lg (2^(1/2)) + lg 5 + ((lg (2^(1/2)))^2 - lg 2 + 1)^(1/2) = 1 :=
sorry

end NUMINAMATH_CALUDE_complex_expression_equality_logarithmic_expression_equality_l3569_356986


namespace NUMINAMATH_CALUDE_weight_increase_percentage_l3569_356908

/-- Calculates the percentage increase in weight on the lowering portion of an exercise machine. -/
theorem weight_increase_percentage
  (num_plates : ℕ)
  (plate_weight : ℝ)
  (lowered_weight : ℝ)
  (h1 : num_plates = 10)
  (h2 : plate_weight = 30)
  (h3 : lowered_weight = 360) :
  ((lowered_weight - num_plates * plate_weight) / (num_plates * plate_weight)) * 100 = 20 := by
sorry

end NUMINAMATH_CALUDE_weight_increase_percentage_l3569_356908


namespace NUMINAMATH_CALUDE_stones_per_bracelet_l3569_356932

def total_stones : ℝ := 48.0
def num_bracelets : ℕ := 6

theorem stones_per_bracelet :
  total_stones / num_bracelets = 8 := by sorry

end NUMINAMATH_CALUDE_stones_per_bracelet_l3569_356932


namespace NUMINAMATH_CALUDE_multiplication_mistake_l3569_356998

theorem multiplication_mistake (x : ℝ) : 973 * x - 739 * x = 110305 → x = 471.4 := by
  sorry

end NUMINAMATH_CALUDE_multiplication_mistake_l3569_356998


namespace NUMINAMATH_CALUDE_inequality_range_l3569_356959

theorem inequality_range (m : ℝ) :
  (∀ x : ℝ, |x - 1| + |x + m| > 3) ↔ m ∈ Set.Iio (-4) ∪ Set.Ioi 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_range_l3569_356959


namespace NUMINAMATH_CALUDE_problem_statement_l3569_356985

theorem problem_statement (a b : ℝ) : 
  (a^2 + 4*a + 6) * (2*b^2 - 4*b + 7) ≤ 10 → a + 2*b = 0 :=
by sorry

end NUMINAMATH_CALUDE_problem_statement_l3569_356985


namespace NUMINAMATH_CALUDE_area_between_concentric_circles_l3569_356944

/-- Given two concentric circles where a chord of length 80 units is tangent to the smaller circle,
    the area between the two circles is equal to 1600π square units. -/
theorem area_between_concentric_circles (O : ℝ × ℝ) (r₁ r₂ : ℝ) (A B : ℝ × ℝ) :
  let circle₁ := {P : ℝ × ℝ | (P.1 - O.1)^2 + (P.2 - O.2)^2 = r₁^2}
  let circle₂ := {P : ℝ × ℝ | (P.1 - O.1)^2 + (P.2 - O.2)^2 = r₂^2}
  r₁ > r₂ →
  (A.1 - B.1)^2 + (A.2 - B.2)^2 = 80^2 →
  ∃ P ∈ circle₂, (P.1 - A.1) * (B.1 - A.1) + (P.2 - A.2) * (B.2 - A.2) = 0 →
  π * (r₁^2 - r₂^2) = 1600 * π :=
by sorry

end NUMINAMATH_CALUDE_area_between_concentric_circles_l3569_356944


namespace NUMINAMATH_CALUDE_correct_systematic_sample_l3569_356979

def total_products : ℕ := 60
def sample_size : ℕ := 5

def systematic_sample (start : ℕ) : List ℕ :=
  List.range sample_size |>.map (λ i => start + i * (total_products / sample_size))

theorem correct_systematic_sample :
  systematic_sample 5 = [5, 17, 29, 41, 53] := by sorry

end NUMINAMATH_CALUDE_correct_systematic_sample_l3569_356979


namespace NUMINAMATH_CALUDE_no_universal_triangle_relation_l3569_356925

/-- A triangle with perimeter, circumradius, and inradius -/
structure Triangle where
  perimeter : ℝ
  circumradius : ℝ
  inradius : ℝ

/-- There is no universal relationship among perimeter, circumradius, and inradius for all triangles -/
theorem no_universal_triangle_relation :
  ¬(∀ t : Triangle,
    (t.perimeter > t.circumradius + t.inradius) ∨
    (t.perimeter ≤ t.circumradius + t.inradius) ∨
    (1/6 < t.circumradius + t.inradius ∧ t.circumradius + t.inradius < 6*t.perimeter)) :=
by sorry

end NUMINAMATH_CALUDE_no_universal_triangle_relation_l3569_356925


namespace NUMINAMATH_CALUDE_first_day_over_200_acorns_l3569_356969

/-- Represents the number of acorns Mark has on a given day -/
def acorns (k : ℕ) : ℕ := 5 * 5^k - 2 * k

/-- Represents the day of the week -/
inductive Day
| Monday
| Tuesday
| Wednesday
| Thursday
| Friday
| Saturday
| Sunday

/-- Converts a natural number to a day of the week -/
def toDay (n : ℕ) : Day :=
  match n % 7 with
  | 0 => Day.Monday
  | 1 => Day.Tuesday
  | 2 => Day.Wednesday
  | 3 => Day.Thursday
  | 4 => Day.Friday
  | 5 => Day.Saturday
  | _ => Day.Sunday

theorem first_day_over_200_acorns :
  ∀ k : ℕ, k < 3 → acorns k ≤ 200 ∧
  acorns 3 > 200 ∧
  toDay 3 = Day.Thursday :=
sorry

end NUMINAMATH_CALUDE_first_day_over_200_acorns_l3569_356969


namespace NUMINAMATH_CALUDE_product_mod_seven_l3569_356949

theorem product_mod_seven : (2021 * 2022 * 2023 * 2024) % 7 = 0 := by
  sorry

end NUMINAMATH_CALUDE_product_mod_seven_l3569_356949


namespace NUMINAMATH_CALUDE_expression_equality_l3569_356960

theorem expression_equality (x : ℝ) :
  3 * (x + 2)^2 + 2 * (x + 2) * (5 - x) + (5 - x)^2 = 
  ((Real.sqrt 3 - 1) * x + 5 + 2 * Real.sqrt 3)^2 := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l3569_356960


namespace NUMINAMATH_CALUDE_chess_problem_l3569_356970

/-- Represents a chess piece (rook or king) -/
inductive Piece
| Rook
| King

/-- Represents a position on the chess board -/
structure Position :=
  (x : Nat) (y : Nat)

/-- Represents the state of the chess board -/
structure ChessBoard :=
  (size : Nat)
  (whiteRooks : List Position)
  (blackKing : Position)

/-- Checks if a position is in check -/
def isInCheck (board : ChessBoard) : Bool :=
  sorry

/-- Checks if the king can get into check after some finite number of moves -/
def canGetIntoCheck (board : ChessBoard) : Bool :=
  sorry

/-- Checks if the king can always be in check after its move (excluding initial moves) -/
def canAlwaysBeInCheckAfterMove (board : ChessBoard) : Bool :=
  sorry

/-- Checks if the king can always be in check (even after white's move, excluding initial moves) -/
def canAlwaysBeInCheck (board : ChessBoard) : Bool :=
  sorry

theorem chess_problem (board : ChessBoard) 
  (h1 : board.size = 1000) 
  (h2 : board.whiteRooks.length = 499) :
  (canGetIntoCheck board = true) ∧ 
  (canAlwaysBeInCheckAfterMove board = false) ∧
  (canAlwaysBeInCheck board = false) :=
  sorry

end NUMINAMATH_CALUDE_chess_problem_l3569_356970


namespace NUMINAMATH_CALUDE_count_points_is_ten_l3569_356946

def M : Finset Int := {1, -2, 3}
def N : Finset Int := {-4, 5, 6, -7}

def is_in_third_or_fourth_quadrant (p : Int × Int) : Bool :=
  p.2 < 0

def count_points : Nat :=
  (M.card * (N.filter (· < 0)).card) + (N.card * (M.filter (· < 0)).card)

theorem count_points_is_ten :
  count_points = 10 := by sorry

end NUMINAMATH_CALUDE_count_points_is_ten_l3569_356946


namespace NUMINAMATH_CALUDE_expression_equality_l3569_356958

theorem expression_equality : -1^4 + |1 - Real.sqrt 2| - (Real.pi - 3.14)^0 = Real.sqrt 2 - 3 := by
  sorry

end NUMINAMATH_CALUDE_expression_equality_l3569_356958


namespace NUMINAMATH_CALUDE_percentage_students_owning_only_cats_l3569_356911

/-- Proves that the percentage of students owning only cats is 10% -/
theorem percentage_students_owning_only_cats
  (total_students : ℕ)
  (students_with_dogs : ℕ)
  (students_with_cats : ℕ)
  (students_with_both : ℕ)
  (h1 : total_students = 500)
  (h2 : students_with_dogs = 200)
  (h3 : students_with_cats = 100)
  (h4 : students_with_both = 50) :
  (students_with_cats - students_with_both) / total_students = 1 / 10 := by
  sorry


end NUMINAMATH_CALUDE_percentage_students_owning_only_cats_l3569_356911


namespace NUMINAMATH_CALUDE_cookie_difference_l3569_356924

/-- Given that Alyssa has 129 cookies and Aiyanna has 140 cookies, 
    prove that Aiyanna has 11 more cookies than Alyssa. -/
theorem cookie_difference (alyssa_cookies : ℕ) (aiyanna_cookies : ℕ) 
    (h1 : alyssa_cookies = 129) (h2 : aiyanna_cookies = 140) : 
    aiyanna_cookies - alyssa_cookies = 11 := by
  sorry

end NUMINAMATH_CALUDE_cookie_difference_l3569_356924


namespace NUMINAMATH_CALUDE_missing_digit_is_four_l3569_356963

def set_of_numbers : List Nat := [8, 88, 888, 8888, 88888, 888888, 8888888, 88888888, 888888888]

def arithmetic_mean (numbers : List Nat) : Rat :=
  (numbers.sum : Rat) / numbers.length

theorem missing_digit_is_four :
  let mean := arithmetic_mean set_of_numbers
  ∃ (n : Nat), 
    (n = Int.floor mean) ∧ 
    (n ≥ 100000000 ∧ n < 1000000000) ∧ 
    (∀ d₁ d₂, d₁ ∈ n.digits 10 → d₂ ∈ n.digits 10 → d₁ ≠ d₂ → d₁ ≠ d₂) ∧
    (4 ∉ n.digits 10) :=
by sorry

end NUMINAMATH_CALUDE_missing_digit_is_four_l3569_356963


namespace NUMINAMATH_CALUDE_infinite_circles_inside_l3569_356935

-- Define a circle
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define a point
def Point := ℝ × ℝ

-- Define what it means for a point to be inside a circle
def isInside (p : Point) (c : Circle) : Prop :=
  (p.1 - c.center.1)^2 + (p.2 - c.center.2)^2 < c.radius^2

-- Define what it means for a circle to be entirely inside another circle
def isEntirelyInside (c1 c2 : Circle) : Prop :=
  ∀ p : Point, isInside p c1 → isInside p c2

-- The main theorem
theorem infinite_circles_inside (C : Circle) (A B : Point) 
  (hA : isInside A C) (hB : isInside B C) :
  ∃ f : ℕ → Circle, (∀ n : ℕ, isEntirelyInside (f n) C ∧ 
                               isInside A (f n) ∧ 
                               isInside B (f n)) ∧
                     (∀ m n : ℕ, m ≠ n → f m ≠ f n) := by
  sorry

end NUMINAMATH_CALUDE_infinite_circles_inside_l3569_356935


namespace NUMINAMATH_CALUDE_parabola_tangent_properties_l3569_356909

/-- Given a parabola and a point, proves properties of its tangent lines -/
theorem parabola_tangent_properties (S : ℝ × ℝ) (parabola : ℝ → ℝ → Prop) :
  S = (-3, 7) →
  (∀ x y, parabola x y ↔ y^2 = 5*x) →
  ∃ (t₁ t₂ : ℝ → ℝ) (P₁ P₂ : ℝ × ℝ) (α : ℝ),
    -- Tangent line equations
    (∀ x, t₁ x = x/6 + 15/2) ∧
    (∀ x, t₂ x = -5/2*x - 1/2) ∧
    -- Points of tangency
    P₁ = (45, 15) ∧
    P₂ = (1/5, -1) ∧
    -- Angle between tangents
    α = Real.arctan (32/7) ∧
    -- Tangent lines pass through S
    t₁ (S.1) = S.2 ∧
    t₂ (S.1) = S.2 ∧
    -- Points of tangency lie on the parabola
    parabola P₁.1 P₁.2 ∧
    parabola P₂.1 P₂.2 ∧
    -- Tangent lines touch the parabola at points of tangency
    t₁ P₁.1 = P₁.2 ∧
    t₂ P₂.1 = P₂.2 :=
by
  sorry


end NUMINAMATH_CALUDE_parabola_tangent_properties_l3569_356909


namespace NUMINAMATH_CALUDE_original_number_problem_l3569_356929

theorem original_number_problem (x : ℚ) : 2 * x + 5 = x / 2 + 20 → x = 10 := by
  sorry

end NUMINAMATH_CALUDE_original_number_problem_l3569_356929


namespace NUMINAMATH_CALUDE_special_circle_standard_equation_l3569_356965

/-- A circle passing through two points with its center on a given line -/
structure SpecialCircle where
  -- Center coordinates
  h : ℝ
  k : ℝ
  -- Radius
  r : ℝ
  -- The circle passes through (0,4)
  passes_through_A : h^2 + (k - 4)^2 = r^2
  -- The circle passes through (4,6)
  passes_through_B : (h - 4)^2 + (k - 6)^2 = r^2
  -- The center lies on the line x-2y-2=0
  center_on_line : h - 2*k - 2 = 0

/-- The standard equation of the special circle -/
def special_circle_equation (c : SpecialCircle) : Prop :=
  ∀ (x y : ℝ), (x - 4)^2 + (y - 1)^2 = 25 ↔ (x - c.h)^2 + (y - c.k)^2 = c.r^2

/-- The main theorem: proving the standard equation of the special circle -/
theorem special_circle_standard_equation :
  ∃ (c : SpecialCircle), special_circle_equation c :=
sorry

end NUMINAMATH_CALUDE_special_circle_standard_equation_l3569_356965


namespace NUMINAMATH_CALUDE_shaded_area_theorem_l3569_356976

/-- The fraction of the total area shaded in each iteration -/
def shaded_fraction : ℚ := 4 / 6

/-- The fraction of the remaining area subdivided in each iteration -/
def subdivision_fraction : ℚ := 1 / 6

/-- The sum of the shaded areas in an infinitely divided rectangle -/
def shaded_area_sum : ℚ := shaded_fraction / (1 - subdivision_fraction)

/-- 
Theorem: The sum of the shaded area in an infinitely divided rectangle, 
where 4/6 of each central subdivision is shaded in each iteration, 
is equal to 4/5 of the total area.
-/
theorem shaded_area_theorem : shaded_area_sum = 4 / 5 := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_theorem_l3569_356976


namespace NUMINAMATH_CALUDE_equation_satisfied_l3569_356950

theorem equation_satisfied (x y : ℝ) (hx : x = 5) (hy : y = 3) : 2 * x - 3 * y = 1 := by
  sorry

end NUMINAMATH_CALUDE_equation_satisfied_l3569_356950


namespace NUMINAMATH_CALUDE_mixing_solutions_theorem_l3569_356905

/-- Proves that mixing 300 mL of 10% alcohol solution with 900 mL of 30% alcohol solution 
    results in a 25% alcohol solution -/
theorem mixing_solutions_theorem (x_volume y_volume : ℝ) 
  (x_concentration y_concentration final_concentration : ℝ) :
  x_volume = 300 →
  y_volume = 900 →
  x_concentration = 0.1 →
  y_concentration = 0.3 →
  final_concentration = 0.25 →
  (x_volume * x_concentration + y_volume * y_concentration) / (x_volume + y_volume) = final_concentration :=
by
  sorry

#check mixing_solutions_theorem

end NUMINAMATH_CALUDE_mixing_solutions_theorem_l3569_356905


namespace NUMINAMATH_CALUDE_paper_cutting_impossibility_l3569_356990

theorem paper_cutting_impossibility : ¬ ∃ m : ℕ, 1 + 3 * m = 50 := by
  sorry

end NUMINAMATH_CALUDE_paper_cutting_impossibility_l3569_356990


namespace NUMINAMATH_CALUDE_quadratic_transformation_l3569_356966

theorem quadratic_transformation (x : ℝ) : 
  (2 * x^2 - 3 * x + 1 = 0) ↔ ((x - 3/4)^2 = 1/16) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_transformation_l3569_356966


namespace NUMINAMATH_CALUDE_chris_current_age_l3569_356943

-- Define Praveen's current age
def praveen_age : ℝ := sorry

-- Define Chris's current age
def chris_age : ℝ := sorry

-- Condition 1: Praveen's age after 10 years is 3 times his age 3 years back
axiom praveen_age_condition : praveen_age + 10 = 3 * (praveen_age - 3)

-- Condition 2: Chris is 2 years younger than Praveen was 4 years ago
axiom chris_age_condition : chris_age = (praveen_age - 4) - 2

-- Theorem to prove
theorem chris_current_age : chris_age = 3.5 := by sorry

end NUMINAMATH_CALUDE_chris_current_age_l3569_356943


namespace NUMINAMATH_CALUDE_run_6000_ends_at_S_S_associated_with_D_or_A_l3569_356903

/-- Represents the quarters of the circular track -/
inductive Quarter
| A
| B
| C
| D

/-- Represents a point on the circular track -/
structure Point where
  quarter : Quarter
  distance : ℝ
  h_distance_bound : 0 ≤ distance ∧ distance < 15

/-- The circular track -/
structure Track where
  circumference : ℝ
  h_circumference : circumference = 60

/-- Runner's position after running a given distance -/
def run_position (track : Track) (start : Point) (distance : ℝ) : Point :=
  sorry

/-- Theorem stating that running 6000 feet from point S ends at point S -/
theorem run_6000_ends_at_S (track : Track) (S : Point) 
    (h_S : S.quarter = Quarter.D ∧ S.distance = 0) : 
    run_position track S 6000 = S :=
  sorry

/-- Theorem stating that point S is associated with quarter D or A -/
theorem S_associated_with_D_or_A (S : Point) 
    (h_S : S.quarter = Quarter.D ∧ S.distance = 0) : 
    S.quarter = Quarter.D ∨ S.quarter = Quarter.A :=
  sorry

end NUMINAMATH_CALUDE_run_6000_ends_at_S_S_associated_with_D_or_A_l3569_356903


namespace NUMINAMATH_CALUDE_sum_A_B_equals_twice_cube_l3569_356920

/-- The sum of numbers in the nth group of positive integers -/
def A (n : ℕ) : ℕ :=
  (2 * n - 1) * (n^2 - n + 1)

/-- The difference between the latter and former number in the nth group of cubes -/
def B (n : ℕ) : ℕ :=
  3 * n^2 - 3 * n + 1

/-- Theorem stating that A_n + B_n = 2n^3 for all positive integers n -/
theorem sum_A_B_equals_twice_cube (n : ℕ) :
  A n + B n = 2 * n^3 := by
  sorry

end NUMINAMATH_CALUDE_sum_A_B_equals_twice_cube_l3569_356920


namespace NUMINAMATH_CALUDE_cards_distribution_l3569_356957

theorem cards_distribution (total_cards : Nat) (num_people : Nat) 
  (h1 : total_cards = 60) (h2 : num_people = 9) : 
  (num_people - (total_cards % num_people)) = 3 := by
  sorry

end NUMINAMATH_CALUDE_cards_distribution_l3569_356957


namespace NUMINAMATH_CALUDE_min_translation_for_even_function_l3569_356928

theorem min_translation_for_even_function (f : ℝ → ℝ) (m : ℝ) :
  (∀ x, f x = Real.sin (3 * x + π / 4)) →
  m > 0 →
  (∀ x, Real.sin (3 * (x + m) + π / 4) = Real.sin (3 * (-x + m) + π / 4)) →
  m ≥ π / 12 :=
by sorry

end NUMINAMATH_CALUDE_min_translation_for_even_function_l3569_356928


namespace NUMINAMATH_CALUDE_min_height_rectangular_container_l3569_356919

theorem min_height_rectangular_container (h : ℝ) (y : ℝ) :
  h = 2 * y →                -- height is twice the side length
  y > 0 →                    -- side length is positive
  10 * y^2 ≥ 150 →           -- surface area is at least 150
  h ≥ 2 * Real.sqrt 15 :=    -- minimum height is 2√15
sorry

end NUMINAMATH_CALUDE_min_height_rectangular_container_l3569_356919


namespace NUMINAMATH_CALUDE_foreign_stamps_count_l3569_356902

/-- Represents a collection of stamps -/
structure StampCollection where
  total : ℕ
  old : ℕ
  foreign_and_old : ℕ
  neither : ℕ

/-- Calculates the number of foreign stamps in a collection -/
def foreign_stamps (c : StampCollection) : ℕ :=
  c.total - c.old - c.neither + c.foreign_and_old

/-- Theorem stating the number of foreign stamps in the given collection -/
theorem foreign_stamps_count (c : StampCollection) 
  (h1 : c.total = 200)
  (h2 : c.old = 50)
  (h3 : c.foreign_and_old = 20)
  (h4 : c.neither = 80) :
  foreign_stamps c = 90 := by
  sorry

#eval foreign_stamps { total := 200, old := 50, foreign_and_old := 20, neither := 80 }

end NUMINAMATH_CALUDE_foreign_stamps_count_l3569_356902


namespace NUMINAMATH_CALUDE_geometric_subsequence_contains_342_l3569_356962

/-- An arithmetic sequence with a non-zero common difference -/
structure ArithmeticSequence :=
  (a : ℕ → ℝ)
  (d : ℝ)
  (h_arith : ∀ n : ℕ, a (n + 1) = a n + d)
  (h_d_nonzero : d ≠ 0)

/-- A geometric sequence extracted from an arithmetic sequence -/
structure GeometricSubsequence (as : ArithmeticSequence) :=
  (seq : ℕ → ℝ)
  (h_geom : ∃ q : ℝ, q ≠ 0 ∧ ∀ n : ℕ, seq (n + 1) = q * seq n)
  (h_sub : ∃ f : ℕ → ℕ, ∀ n : ℕ, seq n = as.a (f n))
  (h_2_6_22 : ∃ k₁ k₂ k₃ : ℕ, seq k₁ = as.a 2 ∧ seq k₂ = as.a 6 ∧ seq k₃ = as.a 22)

/-- The main theorem -/
theorem geometric_subsequence_contains_342 (as : ArithmeticSequence) 
  (gs : GeometricSubsequence as) : 
  ∃ k : ℕ, gs.seq k = as.a 342 := by
  sorry

end NUMINAMATH_CALUDE_geometric_subsequence_contains_342_l3569_356962


namespace NUMINAMATH_CALUDE_circle_theorem_part1_circle_theorem_part2_l3569_356956

-- Define the points A and B
def A : ℝ × ℝ := (2, -3)
def B : ℝ × ℝ := (-2, -5)

-- Define the line equation
def line_eq (x y : ℝ) : Prop := x - 2 * y - 3 = 0

-- Define the circle equation for part 1
def circle_eq1 (x y : ℝ) : Prop := (x + 1)^2 + (y + 2)^2 = 10

-- Define the circle equation for part 2
def circle_eq2 (x y : ℝ) : Prop := x^2 + (y + 4)^2 = 5

-- Part 1: Circle passing through A and B with center on the line
theorem circle_theorem_part1 :
  ∃ (center : ℝ × ℝ), 
    (line_eq center.1 center.2) ∧
    (∀ (x y : ℝ), circle_eq1 x y ↔ 
      ((x - center.1)^2 + (y - center.2)^2 = (A.1 - center.1)^2 + (A.2 - center.2)^2) ∧
      ((x - center.1)^2 + (y - center.2)^2 = (B.1 - center.1)^2 + (B.2 - center.2)^2)) :=
sorry

-- Part 2: Circle passing through A and B with minimum area
theorem circle_theorem_part2 :
  ∃ (center : ℝ × ℝ),
    (∀ (other_center : ℝ × ℝ),
      (A.1 - center.1)^2 + (A.2 - center.2)^2 ≤ (A.1 - other_center.1)^2 + (A.2 - other_center.2)^2) ∧
    (∀ (x y : ℝ), circle_eq2 x y ↔ 
      ((x - center.1)^2 + (y - center.2)^2 = (A.1 - center.1)^2 + (A.2 - center.2)^2) ∧
      ((x - center.1)^2 + (y - center.2)^2 = (B.1 - center.1)^2 + (B.2 - center.2)^2)) :=
sorry

end NUMINAMATH_CALUDE_circle_theorem_part1_circle_theorem_part2_l3569_356956


namespace NUMINAMATH_CALUDE_cosine_identities_l3569_356997

theorem cosine_identities :
  (Real.cos (36 * π / 180) - Real.cos (72 * π / 180) = 1/2) ∧
  (Real.cos (π/7) - Real.cos (2*π/7) + Real.cos (3*π/7) = 1/2) := by
  sorry

end NUMINAMATH_CALUDE_cosine_identities_l3569_356997


namespace NUMINAMATH_CALUDE_purely_imaginary_complex_number_l3569_356947

theorem purely_imaginary_complex_number (a : ℝ) : 
  (∃ z : ℂ, z = Complex.mk (a^2 - a - 2) (a + 1) ∧ z.re = 0 ∧ z.im ≠ 0) → a = 2 := by
  sorry

end NUMINAMATH_CALUDE_purely_imaginary_complex_number_l3569_356947


namespace NUMINAMATH_CALUDE_hormone_related_phenomena_l3569_356984

-- Define the set of all phenomena
def Phenomena : Set String :=
  {"Fruit ripening", "Leaves turning yellow", "Fruit shedding", "CO2 fixation",
   "Topping cotton plants", "Absorption of mineral elements"}

-- Define the set of phenomena related to plant hormones
def HormoneRelatedPhenomena : Set String :=
  {"Fruit ripening", "Fruit shedding", "Topping cotton plants"}

-- Define a predicate for phenomena related to plant hormones
def isHormoneRelated (p : String) : Prop :=
  p ∈ HormoneRelatedPhenomena

-- Theorem statement
theorem hormone_related_phenomena :
  ∀ p ∈ Phenomena, isHormoneRelated p ↔
    (p = "Fruit ripening" ∨ p = "Fruit shedding" ∨ p = "Topping cotton plants") :=
by sorry

end NUMINAMATH_CALUDE_hormone_related_phenomena_l3569_356984


namespace NUMINAMATH_CALUDE_carlas_apples_l3569_356999

/-- The number of apples Carla put in her backpack in the morning. -/
def initial_apples : ℕ := sorry

/-- The number of apples stolen by Buffy. -/
def stolen_apples : ℕ := 45

/-- The number of apples that fell out of the backpack. -/
def fallen_apples : ℕ := 26

/-- The number of apples remaining at lunchtime. -/
def remaining_apples : ℕ := 8

theorem carlas_apples : initial_apples = stolen_apples + fallen_apples + remaining_apples := by
  sorry

end NUMINAMATH_CALUDE_carlas_apples_l3569_356999


namespace NUMINAMATH_CALUDE_square_root_sum_equality_l3569_356933

theorem square_root_sum_equality (x : ℝ) :
  Real.sqrt (5 + x) + Real.sqrt (20 - x) = 7 →
  (5 + x) * (20 - x) = 144 := by
  sorry

end NUMINAMATH_CALUDE_square_root_sum_equality_l3569_356933


namespace NUMINAMATH_CALUDE_negative_result_l3569_356974

theorem negative_result : 1 - 9 < 0 := by
  sorry

#check negative_result

end NUMINAMATH_CALUDE_negative_result_l3569_356974


namespace NUMINAMATH_CALUDE_perfect_square_condition_l3569_356981

theorem perfect_square_condition (n : ℕ) : 
  (∃ k : ℕ, n^2 - 19*n + 95 = k^2) ↔ (n = 5 ∨ n = 14) := by
sorry

end NUMINAMATH_CALUDE_perfect_square_condition_l3569_356981


namespace NUMINAMATH_CALUDE_antonio_hamburger_usage_l3569_356942

/-- Calculates the total amount of hamburger used for meatballs given the number of family members,
    meatballs per person, and amount of hamburger per meatball. -/
def hamburger_used (family_members : ℕ) (meatballs_per_person : ℕ) (hamburger_per_meatball : ℚ) : ℚ :=
  (family_members * meatballs_per_person : ℚ) * hamburger_per_meatball

/-- Proves that given the conditions in the problem, Antonio used 4 pounds of hamburger. -/
theorem antonio_hamburger_usage :
  let family_members : ℕ := 8
  let meatballs_per_person : ℕ := 4
  let hamburger_per_meatball : ℚ := 1/8
  hamburger_used family_members meatballs_per_person hamburger_per_meatball = 4 := by
  sorry


end NUMINAMATH_CALUDE_antonio_hamburger_usage_l3569_356942


namespace NUMINAMATH_CALUDE_equation_has_two_real_roots_l3569_356987

theorem equation_has_two_real_roots :
  ∃ (x₁ x₂ : ℝ), x₁ ≠ x₂ ∧
  (x₁ - Real.sqrt (2 * x₁ + 6) = 2) ∧
  (x₂ - Real.sqrt (2 * x₂ + 6) = 2) ∧
  (∀ x : ℝ, x - Real.sqrt (2 * x + 6) = 2 → x = x₁ ∨ x = x₂) :=
by sorry

end NUMINAMATH_CALUDE_equation_has_two_real_roots_l3569_356987


namespace NUMINAMATH_CALUDE_circle_diameter_ratio_l3569_356964

theorem circle_diameter_ratio (D C : ℝ) : 
  D = 20 → -- Diameter of circle D is 20 cm
  C > 0 → -- Diameter of circle C is positive
  C < D → -- Circle C is inside circle D
  (π * D^2 / 4 - π * C^2 / 4) / (π * C^2 / 4) = 4 → -- Ratio of shaded area to area of C is 4:1
  C = 8 * Real.sqrt 5 := by sorry

end NUMINAMATH_CALUDE_circle_diameter_ratio_l3569_356964


namespace NUMINAMATH_CALUDE_factorization_proof_l3569_356939

theorem factorization_proof (x y : ℝ) : -2*x*y^2 + 4*x*y - 2*x = -2*x*(y - 1)^2 := by
  sorry

end NUMINAMATH_CALUDE_factorization_proof_l3569_356939


namespace NUMINAMATH_CALUDE_work_completion_time_l3569_356989

theorem work_completion_time (x_days y_days combined_days : ℝ) 
  (hx : x_days = 15)
  (hc : combined_days = 11.25)
  (h_combined : 1 / x_days + 1 / y_days = 1 / combined_days) :
  y_days = 45 := by sorry

end NUMINAMATH_CALUDE_work_completion_time_l3569_356989


namespace NUMINAMATH_CALUDE_possible_m_values_l3569_356917

def A (m : ℝ) : Set ℝ := {x | m * x - 1 = 0}
def B : Set ℝ := {2, 3}

theorem possible_m_values :
  ∀ m : ℝ, (A m) ⊆ B → (m = 0 ∨ m = 1/2 ∨ m = 1/3) :=
by sorry

end NUMINAMATH_CALUDE_possible_m_values_l3569_356917


namespace NUMINAMATH_CALUDE_parabola_fixed_point_l3569_356971

/-- Parabola E: y² = 4x -/
def parabola (x y : ℝ) : Prop := y^2 = 4*x

/-- Focus of the parabola -/
def focus : ℝ × ℝ := (1, 0)

/-- Point P -/
def P : ℝ × ℝ := (7, 3)

/-- Line with slope k passing through point P -/
def line_through_P (k : ℝ) (x y : ℝ) : Prop :=
  y - P.2 = k * (x - P.1)

/-- Line with slope 2/3 passing through point A -/
def line_AC (A : ℝ × ℝ) (x y : ℝ) : Prop :=
  y - A.2 = (2/3) * (x - A.1)

theorem parabola_fixed_point :
  ∀ (k : ℝ) (A B C : ℝ × ℝ),
  parabola A.1 A.2 ∧ parabola B.1 B.2 ∧ parabola C.1 C.2 →
  line_through_P k A.1 A.2 →
  line_through_P k B.1 B.2 →
  line_AC A C.1 C.2 →
  ∃ (m : ℝ), y - C.2 = m * (x - C.1) ∧ y - B.2 = m * (x - B.1) →
  y - 3 = m * (x + 5/2) :=
sorry

end NUMINAMATH_CALUDE_parabola_fixed_point_l3569_356971


namespace NUMINAMATH_CALUDE_middle_school_students_in_ganzhou_form_set_l3569_356914

-- Define the universe of discourse
def Universe : Type := Unit

-- Define the property of being a middle school student in Ganzhou
def IsMiddleSchoolStudentInGanzhou : Universe → Prop := sorry

-- Define what it means for a collection to have definite elements
def HasDefiniteElements (S : Set Universe) : Prop := sorry

-- Theorem: The set of all middle school students in Ganzhou has definite elements
theorem middle_school_students_in_ganzhou_form_set :
  HasDefiniteElements {x : Universe | IsMiddleSchoolStudentInGanzhou x} := by
  sorry

end NUMINAMATH_CALUDE_middle_school_students_in_ganzhou_form_set_l3569_356914


namespace NUMINAMATH_CALUDE_hyperbola_intersection_trajectory_l3569_356926

theorem hyperbola_intersection_trajectory
  (x1 y1 : ℝ)
  (h_on_hyperbola : x1^2 / 2 - y1^2 = 1)
  (h_distinct : x1 ≠ -Real.sqrt 2 ∧ x1 ≠ Real.sqrt 2)
  (x y : ℝ)
  (h_intersection : ∃ (t s : ℝ),
    x = -Real.sqrt 2 + t * (x1 + Real.sqrt 2) ∧
    y = t * y1 ∧
    x = Real.sqrt 2 + s * (x1 - Real.sqrt 2) ∧
    y = -s * y1) :
  x^2 / 2 + y^2 = 1 ∧ x ≠ 0 ∧ x ≠ -Real.sqrt 2 ∧ x ≠ Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_intersection_trajectory_l3569_356926


namespace NUMINAMATH_CALUDE_rectangular_field_perimeter_l3569_356952

/-- The perimeter of a rectangular field with length 7/5 of its width and width 70 meters is 336 meters. -/
theorem rectangular_field_perimeter : 
  ∀ (width length perimeter : ℝ),
  width = 70 →
  length = (7 / 5) * width →
  perimeter = 2 * (length + width) →
  perimeter = 336 := by
sorry

end NUMINAMATH_CALUDE_rectangular_field_perimeter_l3569_356952


namespace NUMINAMATH_CALUDE_solve_for_x_l3569_356968

theorem solve_for_x (x y : ℚ) (h1 : x / y = 8 / 3) (h2 : y = 21) : x = 56 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_x_l3569_356968


namespace NUMINAMATH_CALUDE_bobby_blocks_l3569_356907

theorem bobby_blocks (initial_blocks final_blocks given_blocks : ℕ) 
  (h1 : final_blocks = initial_blocks + given_blocks)
  (h2 : final_blocks = 8)
  (h3 : given_blocks = 6) : 
  initial_blocks = 2 := by sorry

end NUMINAMATH_CALUDE_bobby_blocks_l3569_356907


namespace NUMINAMATH_CALUDE_base_conversion_problem_l3569_356900

theorem base_conversion_problem :
  ∀ (a b : ℕ),
    a < 10 →
    b < 10 →
    235 = 1 * 7^2 + a * 7^1 + b * 7^0 →
    (a + b : ℚ) / 7 = 6 / 7 := by
  sorry

end NUMINAMATH_CALUDE_base_conversion_problem_l3569_356900


namespace NUMINAMATH_CALUDE_parallelogram_height_l3569_356967

/-- Given a parallelogram with sides 20 feet and 60 feet, and height 55 feet
    perpendicular to the 20-foot side, prove that the height perpendicular
    to the 60-foot side is 1100/60 feet. -/
theorem parallelogram_height (a b h : ℝ) (ha : a = 20) (hb : b = 60) (hh : h = 55) :
  a * h / b = 1100 / 60 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_height_l3569_356967


namespace NUMINAMATH_CALUDE_min_value_abs_sum_l3569_356994

theorem min_value_abs_sum (x : ℝ) : |x - 2| + |x + 1| ≥ 3 ∧ ∃ y : ℝ, |y - 2| + |y + 1| = 3 := by
  sorry

end NUMINAMATH_CALUDE_min_value_abs_sum_l3569_356994


namespace NUMINAMATH_CALUDE_cookie_count_l3569_356934

theorem cookie_count (paul_cookies : ℕ) (paula_difference : ℕ) : 
  paul_cookies = 45 →
  paula_difference = 3 →
  paul_cookies + (paul_cookies - paula_difference) = 87 :=
by
  sorry

end NUMINAMATH_CALUDE_cookie_count_l3569_356934


namespace NUMINAMATH_CALUDE_product_of_numbers_l3569_356973

theorem product_of_numbers (x y : ℝ) : x + y = 60 → x - y = 10 → x * y = 875 := by
  sorry

end NUMINAMATH_CALUDE_product_of_numbers_l3569_356973


namespace NUMINAMATH_CALUDE_cheryls_expenses_l3569_356993

/-- Cheryl's golf tournament expenses problem -/
theorem cheryls_expenses (electricity_bill : ℝ) : 
  -- Golf tournament cost is 20% more than monthly cell phone expenses
  -- Monthly cell phone expenses are $400 more than electricity bill
  -- Total payment for golf tournament is $1440
  (1.2 * (electricity_bill + 400) = 1440) →
  -- Cheryl's electricity bill cost is $800
  electricity_bill = 800 := by
  sorry

end NUMINAMATH_CALUDE_cheryls_expenses_l3569_356993


namespace NUMINAMATH_CALUDE_angle_b_in_special_triangle_l3569_356921

/-- In a triangle ABC, if angle A is 80° and angle B equals angle C, then angle B is 50°. -/
theorem angle_b_in_special_triangle (A B C : Real) (h1 : A = 80)
  (h2 : B = C) (h3 : A + B + C = 180) : B = 50 := by
  sorry

end NUMINAMATH_CALUDE_angle_b_in_special_triangle_l3569_356921


namespace NUMINAMATH_CALUDE_translation_theorem_l3569_356945

/-- A point in a 2D Cartesian coordinate system -/
structure Point where
  x : ℝ
  y : ℝ

/-- Translate a point to the left by a given distance -/
def translateLeft (p : Point) (d : ℝ) : Point :=
  { x := p.x - d, y := p.y }

/-- The theorem stating that translating point M(3, -4) 5 units to the left results in M'(-2, -4) -/
theorem translation_theorem :
  let M : Point := { x := 3, y := -4 }
  let M' : Point := translateLeft M 5
  M'.x = -2 ∧ M'.y = -4 := by sorry

end NUMINAMATH_CALUDE_translation_theorem_l3569_356945


namespace NUMINAMATH_CALUDE_fifth_root_monotone_l3569_356991

theorem fifth_root_monotone (x y : ℝ) (h : x < y) : (x^(1/5) : ℝ) < (y^(1/5) : ℝ) := by
  sorry

end NUMINAMATH_CALUDE_fifth_root_monotone_l3569_356991


namespace NUMINAMATH_CALUDE_cross_in_square_l3569_356988

/-- Given a square with side length s containing a cross made up of two squares
    with side length s/2 and two squares with side length s/4, if the total area
    of the cross is 810 cm², then s = 36 cm. -/
theorem cross_in_square (s : ℝ) :
  (2 * (s/2)^2 + 2 * (s/4)^2 = 810) → s = 36 := by sorry

end NUMINAMATH_CALUDE_cross_in_square_l3569_356988


namespace NUMINAMATH_CALUDE_pumpkin_patch_pie_filling_l3569_356938

/-- Represents the number of cans of pie filling produced from small and large pumpkins -/
def cans_of_pie_filling (small_pumpkins : ℕ) (large_pumpkins : ℕ) : ℕ :=
  (small_pumpkins / 2) + large_pumpkins

theorem pumpkin_patch_pie_filling :
  let small_pumpkins : ℕ := 50
  let large_pumpkins : ℕ := 33
  let total_sales : ℕ := 120
  let small_price : ℕ := 3
  let large_price : ℕ := 5
  cans_of_pie_filling small_pumpkins large_pumpkins = 58 := by
  sorry

#eval cans_of_pie_filling 50 33

end NUMINAMATH_CALUDE_pumpkin_patch_pie_filling_l3569_356938


namespace NUMINAMATH_CALUDE_book_cost_price_l3569_356955

theorem book_cost_price (cost : ℝ) : 
  (1.15 * cost - 1.10 * cost = 100) → cost = 2000 := by
sorry

end NUMINAMATH_CALUDE_book_cost_price_l3569_356955


namespace NUMINAMATH_CALUDE_perpendicular_tangents_locus_l3569_356918

/-- The locus of points where mutually perpendicular tangents to x² + y² = 32 intersect -/
theorem perpendicular_tangents_locus (x₀ y₀ : ℝ) : 
  (∃ t₁ t₂ : ℝ → ℝ, 
    (∀ x y, x^2 + y^2 = 32 → (t₁ x = y ∨ t₂ x = y) → (x - x₀) * (y - y₀) = 0) ∧ 
    (∀ x, (t₁ x - y₀) * (t₂ x - y₀) = -1)) →
  x₀^2 + y₀^2 = 64 :=
sorry

end NUMINAMATH_CALUDE_perpendicular_tangents_locus_l3569_356918


namespace NUMINAMATH_CALUDE_abs_sin_integral_over_2pi_l3569_356901

theorem abs_sin_integral_over_2pi (f : ℝ → ℝ) : 
  (∫ x in (0)..(2 * Real.pi), |Real.sin x|) = 4 := by
  sorry

end NUMINAMATH_CALUDE_abs_sin_integral_over_2pi_l3569_356901


namespace NUMINAMATH_CALUDE_students_left_on_bus_l3569_356961

theorem students_left_on_bus (initial_students : ℕ) (students_off : ℕ) : 
  initial_students = 10 → students_off = 3 → initial_students - students_off = 7 := by
  sorry

end NUMINAMATH_CALUDE_students_left_on_bus_l3569_356961


namespace NUMINAMATH_CALUDE_gene_separation_in_Aa_genotype_l3569_356996

-- Define the stages of spermatogenesis
inductive SpermatogenesisStage
  | formation_primary_spermatocytes
  | formation_secondary_spermatocytes
  | formation_spermatids
  | formation_sperm

-- Define alleles
inductive Allele
  | A
  | a

-- Define the separation event
structure SeparationEvent where
  allele1 : Allele
  allele2 : Allele
  stage : SpermatogenesisStage

-- Define the genotype
def GenotypeAa : List Allele := [Allele.A, Allele.a]

-- Define the correct separation sequence
def CorrectSeparationSequence : List SpermatogenesisStage :=
  [SpermatogenesisStage.formation_spermatids,
   SpermatogenesisStage.formation_spermatids,
   SpermatogenesisStage.formation_secondary_spermatocytes]

-- Theorem statement
theorem gene_separation_in_Aa_genotype :
  ∀ (separation_events : List SeparationEvent),
    (∀ e ∈ separation_events, e.allele1 ∈ GenotypeAa ∧ e.allele2 ∈ GenotypeAa) →
    (∃ (e1 e2 e3 : SeparationEvent),
      e1 ∈ separation_events ∧
      e2 ∈ separation_events ∧
      e3 ∈ separation_events ∧
      e1.allele1 = Allele.A ∧ e1.allele2 = Allele.A ∧
      e2.allele1 = Allele.a ∧ e2.allele2 = Allele.a ∧
      e3.allele1 = Allele.A ∧ e3.allele2 = Allele.a) →
    (separation_events.map (λ e => e.stage)) = CorrectSeparationSequence :=
by sorry

end NUMINAMATH_CALUDE_gene_separation_in_Aa_genotype_l3569_356996


namespace NUMINAMATH_CALUDE_max_districts_in_park_l3569_356948

theorem max_districts_in_park (park_side : ℝ) (district_length : ℝ) (district_width : ℝ)
  (h_park_side : park_side = 14)
  (h_district_length : district_length = 8)
  (h_district_width : district_width = 2) :
  ⌊(park_side^2) / (district_length * district_width)⌋ = 12 := by
sorry

end NUMINAMATH_CALUDE_max_districts_in_park_l3569_356948


namespace NUMINAMATH_CALUDE_sum_of_numbers_l3569_356906

theorem sum_of_numbers : ∀ (a b : ℤ), 
  a = 9 → 
  b = -a + 2 → 
  a + b = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_sum_of_numbers_l3569_356906


namespace NUMINAMATH_CALUDE_system_solution_l3569_356931

theorem system_solution :
  let x₁ := Real.sqrt 2 / Real.sqrt 5
  let x₂ := -Real.sqrt 2 / Real.sqrt 5
  let y₁ := 2 * Real.sqrt 2 / Real.sqrt 5
  let y₂ := -2 * Real.sqrt 2 / Real.sqrt 5
  let condition₁ (x y : ℝ) := x^2 + y^2 ≤ 2
  let condition₂ (x y : ℝ) := x^4 - 8*x^2*y^2 + 16*y^4 - 20*x^2 - 80*y^2 + 100 = 0
  (condition₁ x₁ y₁ ∧ condition₂ x₁ y₁) ∧
  (condition₁ x₁ y₂ ∧ condition₂ x₁ y₂) ∧
  (condition₁ x₂ y₁ ∧ condition₂ x₂ y₁) ∧
  (condition₁ x₂ y₂ ∧ condition₂ x₂ y₂) :=
by sorry


end NUMINAMATH_CALUDE_system_solution_l3569_356931


namespace NUMINAMATH_CALUDE_coefficient_of_x_cubed_in_expansion_l3569_356930

-- Define n such that 2^n = 64
def n : ℕ := 6

-- Define the binomial coefficient
def binomial (n k : ℕ) : ℕ := sorry

-- Theorem statement
theorem coefficient_of_x_cubed_in_expansion :
  ∃ (coeff : ℤ), 
    (2^n = 64) ∧
    (coeff = (-1)^3 * binomial n 3 * 2^(n-3)) ∧
    (coeff = -160) := by
  sorry

end NUMINAMATH_CALUDE_coefficient_of_x_cubed_in_expansion_l3569_356930


namespace NUMINAMATH_CALUDE_spinner_probability_l3569_356980

theorem spinner_probability (p_A p_B p_C p_D : ℚ) : 
  p_A = 1/4 → p_B = 1/3 → p_C = 5/12 → p_A + p_B + p_C + p_D = 1 → p_D = 0 := by
  sorry

end NUMINAMATH_CALUDE_spinner_probability_l3569_356980


namespace NUMINAMATH_CALUDE_initial_fee_equals_65_l3569_356992

/-- The initial fee of the first car rental plan -/
def initial_fee : ℝ := 65

/-- The cost per mile for the first plan -/
def cost_per_mile_plan1 : ℝ := 0.40

/-- The cost per mile for the second plan -/
def cost_per_mile_plan2 : ℝ := 0.60

/-- The number of miles driven -/
def miles_driven : ℝ := 325

/-- Theorem stating that the initial fee makes both plans cost the same for the given miles -/
theorem initial_fee_equals_65 :
  initial_fee + cost_per_mile_plan1 * miles_driven = cost_per_mile_plan2 * miles_driven :=
by sorry

end NUMINAMATH_CALUDE_initial_fee_equals_65_l3569_356992


namespace NUMINAMATH_CALUDE_largest_square_area_l3569_356915

theorem largest_square_area (a b c : ℝ) (h_right_angle : a^2 + b^2 = c^2) 
  (h_total_area : a^2 + b^2 + c^2 = 450) : c^2 = 225 := by
  sorry

end NUMINAMATH_CALUDE_largest_square_area_l3569_356915


namespace NUMINAMATH_CALUDE_four_numbers_solution_l3569_356922

/-- A sequence of four real numbers satisfying the given conditions -/
structure FourNumbers where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ
  arithmetic_seq : b - a = c - b
  geometric_seq : c * c = b * d
  sum_first_last : a + d = 16
  sum_middle : b + c = 12

/-- The theorem stating that there are only two possible sets of four numbers satisfying the conditions -/
theorem four_numbers_solution (x : FourNumbers) :
  (x.a = 0 ∧ x.b = 4 ∧ x.c = 8 ∧ x.d = 16) ∨
  (x.a = 15 ∧ x.b = 9 ∧ x.c = 3 ∧ x.d = 1) :=
by sorry

end NUMINAMATH_CALUDE_four_numbers_solution_l3569_356922


namespace NUMINAMATH_CALUDE_lateral_edge_length_l3569_356954

-- Define the regular triangular pyramid
structure RegularTriangularPyramid where
  baseEdge : ℝ
  lateralEdge : ℝ

-- Define the property of medians not intersecting and lying on cube edges
def mediansPropertyHolds (pyramid : RegularTriangularPyramid) : Prop :=
  -- This is a placeholder for the complex geometric condition
  -- In a real implementation, this would involve more detailed geometric definitions
  sorry

-- Theorem statement
theorem lateral_edge_length
  (pyramid : RegularTriangularPyramid)
  (h1 : pyramid.baseEdge = 1)
  (h2 : mediansPropertyHolds pyramid) :
  pyramid.lateralEdge = Real.sqrt 6 / 2 :=
sorry

end NUMINAMATH_CALUDE_lateral_edge_length_l3569_356954


namespace NUMINAMATH_CALUDE_summer_entry_proof_l3569_356977

-- Define a type for temperature data
structure TemperatureData where
  temps : List Int
  all_positive : ∀ t ∈ temps, t > 0

-- Define the criterion for entering summer
def entered_summer (data : TemperatureData) : Prop :=
  data.temps.length = 5 ∧ ∀ t ∈ data.temps, t ≥ 22

-- Define statistical measures
def median (data : TemperatureData) : Int := sorry
def mode (data : TemperatureData) : Int := sorry
def mean (data : TemperatureData) : ℚ := sorry
def variance (data : TemperatureData) : ℚ := sorry

-- Define the conditions for each location
def location_A_conditions (data : TemperatureData) : Prop :=
  median data = 24 ∧ mode data = 22

def location_B_conditions (data : TemperatureData) : Prop :=
  median data = 25 ∧ mean data = 24

def location_C_conditions (data : TemperatureData) : Prop :=
  mean data = 22 ∧ mode data = 22

def location_D_conditions (data : TemperatureData) : Prop :=
  28 ∈ data.temps ∧ mean data = 24 ∧ variance data = 4.8

-- Theorem statement
theorem summer_entry_proof :
  (∀ data : TemperatureData, location_A_conditions data → entered_summer data) ∧
  (∃ data : TemperatureData, location_B_conditions data ∧ ¬entered_summer data) ∧
  (∃ data : TemperatureData, location_C_conditions data ∧ ¬entered_summer data) ∧
  (∀ data : TemperatureData, location_D_conditions data → entered_summer data) :=
by sorry

end NUMINAMATH_CALUDE_summer_entry_proof_l3569_356977


namespace NUMINAMATH_CALUDE_product_from_lcm_hcf_l3569_356916

theorem product_from_lcm_hcf (a b c : ℕ+) 
  (h_lcm : Nat.lcm a (Nat.lcm b c) = 1800)
  (h_hcf : Nat.gcd a (Nat.gcd b c) = 12) :
  a * b * c = 21600 := by
  sorry

end NUMINAMATH_CALUDE_product_from_lcm_hcf_l3569_356916


namespace NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l3569_356982

/-- An arithmetic sequence with given conditions -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_common_difference
  (a : ℕ → ℝ)
  (h_arith : arithmetic_sequence a)
  (h_a5 : a 5 = 10)
  (h_a12 : a 12 = 31) :
  ∃ d : ℝ, d = 3 ∧ ∀ n : ℕ, a (n + 1) = a n + d :=
by
  sorry


end NUMINAMATH_CALUDE_arithmetic_sequence_common_difference_l3569_356982


namespace NUMINAMATH_CALUDE_profit_percentage_l3569_356940

theorem profit_percentage (cost_price selling_price : ℝ) 
  (h : cost_price = 0.96 * selling_price) :
  (selling_price - cost_price) / cost_price * 100 = (100/96 - 1) * 100 := by
sorry

end NUMINAMATH_CALUDE_profit_percentage_l3569_356940


namespace NUMINAMATH_CALUDE_intersection_range_l3569_356951

theorem intersection_range (a b : ℝ) (h1 : a^2 + b^2 = 1) (h2 : b ≠ 0) :
  (∃ x y : ℝ, a * x + b * y = 2 ∧ x^2 / 6 + y^2 / 2 = 1) →
  a / b ∈ Set.Iic (-1) ∪ Set.Ici 1 := by
sorry

end NUMINAMATH_CALUDE_intersection_range_l3569_356951


namespace NUMINAMATH_CALUDE_pattern_calculation_main_calculation_l3569_356975

theorem pattern_calculation : ℕ → Prop :=
  fun n => n * (n + 1) + (n + 1) * (n + 2) = 2 * (n + 1) * (n + 1)

theorem main_calculation : 
  75 * 222 + 76 * 225 - 25 * 14 * 15 - 25 * 15 * 16 = 302 := by
  sorry

end NUMINAMATH_CALUDE_pattern_calculation_main_calculation_l3569_356975


namespace NUMINAMATH_CALUDE_dot_product_range_l3569_356937

/-- Given vectors a and b in a plane such that their magnitudes and the magnitude of their difference
    are between 2 and 6 (inclusive), prove that their dot product is between -14 and 34 (inclusive). -/
theorem dot_product_range (a b : ℝ × ℝ) 
  (ha : 2 ≤ ‖a‖ ∧ ‖a‖ ≤ 6)
  (hb : 2 ≤ ‖b‖ ∧ ‖b‖ ≤ 6)
  (hab : 2 ≤ ‖a - b‖ ∧ ‖a - b‖ ≤ 6) :
  -14 ≤ a • b ∧ a • b ≤ 34 :=
by sorry

end NUMINAMATH_CALUDE_dot_product_range_l3569_356937


namespace NUMINAMATH_CALUDE_simplest_quadratic_radical_l3569_356904

/-- A quadratic radical is considered simpler if it cannot be further simplified 
    by extracting perfect square factors or rationalizing the denominator. -/
def IsSimplestQuadraticRadical (x : ℝ) : Prop := sorry

theorem simplest_quadratic_radical : 
  IsSimplestQuadraticRadical (-Real.sqrt 2) ∧
  ¬IsSimplestQuadraticRadical (Real.sqrt 12) ∧
  ¬IsSimplestQuadraticRadical (Real.sqrt (3/2)) ∧
  ¬IsSimplestQuadraticRadical (1 / Real.sqrt 5) :=
sorry

end NUMINAMATH_CALUDE_simplest_quadratic_radical_l3569_356904


namespace NUMINAMATH_CALUDE_power_sum_reciprocal_integer_l3569_356923

theorem power_sum_reciprocal_integer (x : ℝ) (h1 : x ≠ 0) (h2 : ∃ k : ℤ, x + 1/x = k) :
  ∀ n : ℕ, ∃ m : ℤ, x^n + 1/x^n = m :=
sorry

end NUMINAMATH_CALUDE_power_sum_reciprocal_integer_l3569_356923


namespace NUMINAMATH_CALUDE_smallest_four_digit_pascal_l3569_356936

/-- Pascal's triangle function -/
def pascal (n : ℕ) (k : ℕ) : ℕ :=
  Nat.choose n k

/-- Predicate for four-digit numbers -/
def is_four_digit (n : ℕ) : Prop :=
  1000 ≤ n ∧ n ≤ 9999

/-- Theorem: 1001 is the smallest four-digit number in Pascal's triangle -/
theorem smallest_four_digit_pascal :
  (∃ n k, pascal n k = 1001) ∧
  (∀ n k, pascal n k < 1001 → ¬is_four_digit (pascal n k)) :=
sorry

end NUMINAMATH_CALUDE_smallest_four_digit_pascal_l3569_356936
