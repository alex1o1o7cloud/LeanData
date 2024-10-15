import Mathlib

namespace NUMINAMATH_CALUDE_pyramid_height_equals_cube_volume_l3249_324959

theorem pyramid_height_equals_cube_volume (cube_edge : ℝ) (pyramid_base : ℝ) (pyramid_height : ℝ) :
  cube_edge = 5 →
  pyramid_base = 10 →
  cube_edge ^ 3 = (1 / 3) * pyramid_base ^ 2 * pyramid_height →
  pyramid_height = 3.75 := by
sorry

end NUMINAMATH_CALUDE_pyramid_height_equals_cube_volume_l3249_324959


namespace NUMINAMATH_CALUDE_ratio_satisfies_condition_l3249_324935

/-- Represents the number of people in each profession -/
structure ProfessionCount where
  doctors : ℕ
  lawyers : ℕ
  engineers : ℕ

/-- The average age of the entire group -/
def groupAverage : ℝ := 45

/-- The average age of doctors -/
def doctorAverage : ℝ := 40

/-- The average age of lawyers -/
def lawyerAverage : ℝ := 55

/-- The average age of engineers -/
def engineerAverage : ℝ := 35

/-- Checks if the given profession count satisfies the average age conditions -/
def satisfiesAverageCondition (count : ProfessionCount) : Prop :=
  let totalPeople := count.doctors + count.lawyers + count.engineers
  let totalAge := doctorAverage * count.doctors + lawyerAverage * count.lawyers + engineerAverage * count.engineers
  totalAge / totalPeople = groupAverage

/-- The theorem stating that the ratio 2:2:1 satisfies the average age conditions -/
theorem ratio_satisfies_condition :
  ∃ (k : ℕ), k > 0 ∧ satisfiesAverageCondition { doctors := 2 * k, lawyers := 2 * k, engineers := k } :=
sorry

end NUMINAMATH_CALUDE_ratio_satisfies_condition_l3249_324935


namespace NUMINAMATH_CALUDE_percentage_relation_l3249_324971

theorem percentage_relation (x y z w : ℝ) 
  (h1 : x = 1.2 * y)
  (h2 : y = 0.4 * z)
  (h3 : z = 0.7 * w) :
  x = 0.336 * w := by
sorry

end NUMINAMATH_CALUDE_percentage_relation_l3249_324971


namespace NUMINAMATH_CALUDE_distance_from_negative_two_l3249_324919

theorem distance_from_negative_two : 
  {x : ℝ | |x - (-2)| = 1} = {-3, -1} := by sorry

end NUMINAMATH_CALUDE_distance_from_negative_two_l3249_324919


namespace NUMINAMATH_CALUDE_cuboid_height_calculation_l3249_324906

/-- The surface area of a cuboid given its length, width, and height. -/
def surfaceArea (l w h : ℝ) : ℝ := 2 * (l * w + l * h + w * h)

/-- Theorem: The height of a cuboid with surface area 2400 cm², length 15 cm, and width 10 cm is 42 cm. -/
theorem cuboid_height_calculation (sa l w h : ℝ) 
  (h_sa : sa = 2400)
  (h_l : l = 15)
  (h_w : w = 10)
  (h_surface_area : surfaceArea l w h = sa) : h = 42 := by
  sorry

#check cuboid_height_calculation

end NUMINAMATH_CALUDE_cuboid_height_calculation_l3249_324906


namespace NUMINAMATH_CALUDE_circle_parameter_range_l3249_324962

/-- Represents the equation of a potential circle -/
def circle_equation (x y a : ℝ) : Prop :=
  x^2 + y^2 - 2*a*x - 4*y + 5*a = 0

/-- Determines if the equation represents a valid circle -/
def is_valid_circle (a : ℝ) : Prop :=
  ∃ (h k r : ℝ), r > 0 ∧ ∀ (x y : ℝ), circle_equation x y a ↔ (x - h)^2 + (y - k)^2 = r^2

/-- The main theorem stating the range of 'a' for which the equation represents a circle -/
theorem circle_parameter_range :
  ∀ a : ℝ, is_valid_circle a ↔ (a > 4 ∨ a < 1) :=
sorry

end NUMINAMATH_CALUDE_circle_parameter_range_l3249_324962


namespace NUMINAMATH_CALUDE_sum_interior_eighth_row_l3249_324973

/-- Sum of interior numbers in a row of Pascal's Triangle -/
def sum_interior (n : ℕ) : ℕ := 2^(n-1) - 2

/-- The row number where interior numbers begin in Pascal's Triangle -/
def interior_start : ℕ := 3

theorem sum_interior_eighth_row :
  sum_interior 6 = 30 →
  sum_interior 8 = 126 :=
by sorry

end NUMINAMATH_CALUDE_sum_interior_eighth_row_l3249_324973


namespace NUMINAMATH_CALUDE_credit_card_balance_proof_l3249_324924

/-- Calculates the new credit card balance after purchases and returns -/
def new_credit_card_balance (initial_balance groceries_cost towels_return : ℚ) : ℚ :=
  initial_balance + groceries_cost + (groceries_cost / 2) - towels_return

/-- Proves that the new credit card balance is correct given the initial conditions -/
theorem credit_card_balance_proof :
  new_credit_card_balance 126 60 45 = 171 := by
  sorry

#eval new_credit_card_balance 126 60 45

end NUMINAMATH_CALUDE_credit_card_balance_proof_l3249_324924


namespace NUMINAMATH_CALUDE_imaginary_part_of_i_squared_is_zero_l3249_324910

theorem imaginary_part_of_i_squared_is_zero :
  Complex.im (Complex.I ^ 2) = 0 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_i_squared_is_zero_l3249_324910


namespace NUMINAMATH_CALUDE_mixture_composition_l3249_324982

theorem mixture_composition (alcohol_water_ratio : ℚ) (alcohol_fraction : ℚ) :
  alcohol_water_ratio = 1/2 →
  alcohol_fraction = 1/7 →
  1 - alcohol_fraction = 2/7 :=
by
  sorry

end NUMINAMATH_CALUDE_mixture_composition_l3249_324982


namespace NUMINAMATH_CALUDE_even_function_property_l3249_324938

/-- A function f: ℝ → ℝ is even if f(x) = f(-x) for all x -/
def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

/-- The main theorem -/
theorem even_function_property (f : ℝ → ℝ) 
  (h_even : IsEven f) 
  (h_neg : ∀ x < 0, f x = 3 * x - 1) : 
  ∀ x > 0, f x = -3 * x - 1 := by
sorry

end NUMINAMATH_CALUDE_even_function_property_l3249_324938


namespace NUMINAMATH_CALUDE_units_digit_of_fraction_l3249_324949

def numerator : ℕ := 15 * 16 * 17 * 18 * 19 * 20
def denominator : ℕ := 500

theorem units_digit_of_fraction : 
  (numerator / denominator) % 10 = 2 := by sorry

end NUMINAMATH_CALUDE_units_digit_of_fraction_l3249_324949


namespace NUMINAMATH_CALUDE_root_sum_squares_l3249_324916

-- Define the polynomial
def p (x : ℝ) : ℝ := x^3 - 12*x^2 + 44*x - 85

-- Define the roots of the polynomial
def roots_condition (a b c : ℝ) : Prop := p a = 0 ∧ p b = 0 ∧ p c = 0

-- Theorem statement
theorem root_sum_squares (a b c : ℝ) (h : roots_condition a b c) :
  (a + b)^2 + (b + c)^2 + (c + a)^2 = 200 := by
  sorry

end NUMINAMATH_CALUDE_root_sum_squares_l3249_324916


namespace NUMINAMATH_CALUDE_inscribed_square_side_length_l3249_324911

/-- A right triangle with sides 12, 16, and 20 -/
structure RightTriangle where
  de : ℝ
  ef : ℝ
  df : ℝ
  is_right : de^2 + ef^2 = df^2
  de_eq : de = 12
  ef_eq : ef = 16
  df_eq : df = 20

/-- A square inscribed in the right triangle -/
structure InscribedSquare (t : RightTriangle) where
  side_length : ℝ
  on_hypotenuse : side_length ≤ t.df
  on_other_sides : side_length ≤ t.de ∧ side_length ≤ t.ef

/-- The side length of the inscribed square is 80/9 -/
theorem inscribed_square_side_length (t : RightTriangle) (s : InscribedSquare t) :
  s.side_length = 80 / 9 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_square_side_length_l3249_324911


namespace NUMINAMATH_CALUDE_not_enough_money_l3249_324964

theorem not_enough_money (book1_price book2_price available_money : ℝ) 
  (h1 : book1_price = 21.8)
  (h2 : book2_price = 19.5)
  (h3 : available_money = 40) :
  book1_price + book2_price > available_money := by
  sorry

end NUMINAMATH_CALUDE_not_enough_money_l3249_324964


namespace NUMINAMATH_CALUDE_total_sales_l3249_324930

def candy_bar_sales (max_sales seth_sales emma_sales : ℕ) : Prop :=
  (max_sales = 24) ∧
  (seth_sales = 3 * max_sales + 6) ∧
  (emma_sales = seth_sales / 2 + 5)

theorem total_sales (max_sales seth_sales emma_sales : ℕ) :
  candy_bar_sales max_sales seth_sales emma_sales →
  seth_sales + emma_sales = 122 := by
  sorry

end NUMINAMATH_CALUDE_total_sales_l3249_324930


namespace NUMINAMATH_CALUDE_subset_existence_l3249_324942

theorem subset_existence (X : Finset ℕ) (hX : X.card = 20) :
  ∀ (f : Finset ℕ → ℕ),
  (∀ S : Finset ℕ, S ⊆ X → S.card = 9 → f S ∈ X) →
  ∃ Y : Finset ℕ, Y ⊆ X ∧ Y.card = 10 ∧
  ∀ k ∈ Y, f (Y \ {k}) ≠ k :=
by sorry

end NUMINAMATH_CALUDE_subset_existence_l3249_324942


namespace NUMINAMATH_CALUDE_rectangle_width_l3249_324926

theorem rectangle_width (length width : ℝ) : 
  width = length + 3 →
  2 * length + 2 * width = 54 →
  width = 15 := by
sorry

end NUMINAMATH_CALUDE_rectangle_width_l3249_324926


namespace NUMINAMATH_CALUDE_repeating_decimal_property_l3249_324927

def is_repeating_decimal_period_2 (n : ℕ) : Prop :=
  ∃ (a b : ℕ), a < 10 ∧ b < 10 ∧ 1 / n = (10 * a + b) / 99

def is_repeating_decimal_period_3 (n : ℕ) : Prop :=
  ∃ (u v w : ℕ), u < 10 ∧ v < 10 ∧ w < 10 ∧ 1 / n = (100 * u + 10 * v + w) / 999

theorem repeating_decimal_property (n : ℕ) :
  n > 0 ∧ n < 3000 ∧
  is_repeating_decimal_period_2 n ∧
  is_repeating_decimal_period_3 (n + 8) →
  601 ≤ n ∧ n ≤ 1200 := by
  sorry

end NUMINAMATH_CALUDE_repeating_decimal_property_l3249_324927


namespace NUMINAMATH_CALUDE_board_game_spaces_l3249_324997

/-- A board game with a certain number of spaces -/
structure BoardGame where
  total_spaces : ℕ

/-- A player's progress in the board game -/
structure PlayerProgress where
  spaces_moved : ℕ
  spaces_to_win : ℕ

/-- Susan's moves in the game -/
def susan_moves : ℕ := 8 + (2 - 5) + 6

/-- The number of spaces Susan needs to move to win -/
def spaces_to_win : ℕ := 37

/-- Theorem stating that the total number of spaces in the game
    is equal to the spaces Susan has moved plus the remaining spaces to win -/
theorem board_game_spaces (game : BoardGame) (susan : PlayerProgress) 
    (h1 : susan.spaces_moved = susan_moves)
    (h2 : susan.spaces_to_win = spaces_to_win - susan_moves) :
  game.total_spaces = susan.spaces_moved + susan.spaces_to_win := by
  sorry

end NUMINAMATH_CALUDE_board_game_spaces_l3249_324997


namespace NUMINAMATH_CALUDE_williams_points_l3249_324991

/-- The number of classes in the contest -/
def num_classes : ℕ := 4

/-- Points scored by Mr. Adams' class -/
def adams_points : ℕ := 57

/-- Points scored by Mrs. Brown's class -/
def brown_points : ℕ := 49

/-- Points scored by Mrs. Daniel's class -/
def daniel_points : ℕ := 57

/-- The mean of the number of points scored -/
def mean_points : ℚ := 53.3

/-- Theorem stating that Mrs. William's class scored 50 points -/
theorem williams_points : ℕ := by
  sorry

end NUMINAMATH_CALUDE_williams_points_l3249_324991


namespace NUMINAMATH_CALUDE_mixture_cost_ratio_l3249_324931

/-- Given the conditions of the mixture problem, prove that the ratio of nut cost to raisin cost is 3:1 -/
theorem mixture_cost_ratio (R N : ℝ) (h1 : R > 0) (h2 : N > 0) : 
  3 * R = 0.25 * (3 * R + 3 * N) → N / R = 3 := by
  sorry

end NUMINAMATH_CALUDE_mixture_cost_ratio_l3249_324931


namespace NUMINAMATH_CALUDE_number_of_bad_oranges_l3249_324974

/-- Given a basket with good and bad oranges, where the number of good oranges
    is known and the ratio of good to bad oranges is given, this theorem proves
    the number of bad oranges. -/
theorem number_of_bad_oranges
  (good_oranges : ℕ)
  (ratio_good : ℕ)
  (ratio_bad : ℕ)
  (h1 : good_oranges = 24)
  (h2 : ratio_good = 3)
  (h3 : ratio_bad = 1)
  : ∃ bad_oranges : ℕ, bad_oranges = 8 ∧ good_oranges * ratio_bad = bad_oranges * ratio_good :=
by
  sorry


end NUMINAMATH_CALUDE_number_of_bad_oranges_l3249_324974


namespace NUMINAMATH_CALUDE_apples_left_l3249_324990

def apples_bought : ℕ := 15
def apples_given : ℕ := 7

theorem apples_left : apples_bought - apples_given = 8 := by
  sorry

end NUMINAMATH_CALUDE_apples_left_l3249_324990


namespace NUMINAMATH_CALUDE_square_of_complex_number_l3249_324978

theorem square_of_complex_number (z : ℂ) (i : ℂ) :
  z = 5 + 2 * i →
  i^2 = -1 →
  z^2 = 21 + 20 * i :=
by sorry

end NUMINAMATH_CALUDE_square_of_complex_number_l3249_324978


namespace NUMINAMATH_CALUDE_geometric_sequence_property_l3249_324961

/-- A geometric sequence with positive terms -/
def GeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, r > 0 ∧ ∀ n : ℕ, a (n + 1) = a n * r

theorem geometric_sequence_property (a : ℕ → ℝ) (h : GeometricSequence a) :
  (∀ n : ℕ, a n > 0) →
  a 1 * a 3 + 2 * a 2 * a 4 + a 3 * a 5 = 16 →
  a 2 + a 4 = 4 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_property_l3249_324961


namespace NUMINAMATH_CALUDE_compound_mass_percentage_sum_l3249_324954

/-- Given a compound with two parts, where one part's mass percentage is known,
    prove that the sum of both parts' mass percentages is 100%. -/
theorem compound_mass_percentage_sum (part1_percentage : ℝ) :
  part1_percentage = 80.12 →
  100 - part1_percentage = 19.88 := by
  sorry

end NUMINAMATH_CALUDE_compound_mass_percentage_sum_l3249_324954


namespace NUMINAMATH_CALUDE_line_parallel_perpendicular_l3249_324951

-- Define the space
variable {V : Type*} [NormedAddCommGroup V] [InnerProductSpace ℝ V] [Finite V]

-- Define lines and planes
def Line (V : Type*) [NormedAddCommGroup V] := Set V
def Plane (V : Type*) [NormedAddCommGroup V] := Set V

-- Define parallel and perpendicular relations
def parallel (l₁ l₂ : Line V) : Prop := sorry
def perpendicular (l : Line V) (p : Plane V) : Prop := sorry

-- Theorem statement
theorem line_parallel_perpendicular 
  (a b : Line V) (α : Plane V) 
  (h₁ : a ≠ b) 
  (h₂ : parallel a b) 
  (h₃ : perpendicular a α) : 
  perpendicular b α := 
sorry

end NUMINAMATH_CALUDE_line_parallel_perpendicular_l3249_324951


namespace NUMINAMATH_CALUDE_mac_loss_is_three_dollars_l3249_324968

-- Define the values of coins in cents
def dime_value : ℕ := 10
def nickel_value : ℕ := 5
def quarter_value : ℕ := 25

-- Define the number of coins in each trade
def dimes_per_trade : ℕ := 3
def nickels_per_trade : ℕ := 7

-- Define the number of trades
def dime_trades : ℕ := 20
def nickel_trades : ℕ := 20

-- Calculate the loss per trade in cents
def dime_trade_loss : ℕ := dimes_per_trade * dime_value - quarter_value
def nickel_trade_loss : ℕ := nickels_per_trade * nickel_value - quarter_value

-- Calculate the total loss in cents
def total_loss_cents : ℕ := dime_trade_loss * dime_trades + nickel_trade_loss * nickel_trades

-- Convert cents to dollars
def cents_to_dollars (cents : ℕ) : ℚ := (cents : ℚ) / 100

-- Theorem: Mac's total loss is $3.00
theorem mac_loss_is_three_dollars :
  cents_to_dollars total_loss_cents = 3 := by
  sorry

end NUMINAMATH_CALUDE_mac_loss_is_three_dollars_l3249_324968


namespace NUMINAMATH_CALUDE_orange_face_probability_l3249_324996

def die_sides : ℕ := 12
def orange_faces : ℕ := 4

theorem orange_face_probability :
  (orange_faces : ℚ) / die_sides = 1 / 3 := by sorry

end NUMINAMATH_CALUDE_orange_face_probability_l3249_324996


namespace NUMINAMATH_CALUDE_train_station_wheels_l3249_324941

theorem train_station_wheels :
  let num_trains : ℕ := 6
  let carriages_per_train : ℕ := 5
  let wheel_rows_per_carriage : ℕ := 4
  let wheels_per_row : ℕ := 6
  
  num_trains * carriages_per_train * wheel_rows_per_carriage * wheels_per_row = 720 :=
by
  sorry

end NUMINAMATH_CALUDE_train_station_wheels_l3249_324941


namespace NUMINAMATH_CALUDE_junior_score_l3249_324913

theorem junior_score (n : ℝ) (h_pos : n > 0) : 
  let junior_count := 0.2 * n
  let senior_count := 0.8 * n
  let total_score := 86 * n
  let senior_score := 85 * senior_count
  junior_count * (total_score - senior_score) / junior_count = 90 :=
by sorry

end NUMINAMATH_CALUDE_junior_score_l3249_324913


namespace NUMINAMATH_CALUDE_condition_sufficient_not_necessary_l3249_324937

theorem condition_sufficient_not_necessary (a b : ℝ) :
  (∀ a b : ℝ, a > 0 ∧ b > 0 → a^2 + b^2 ≥ 2*a*b) ∧
  ¬(∀ a b : ℝ, a^2 + b^2 ≥ 2*a*b → a > 0 ∧ b > 0) :=
sorry

end NUMINAMATH_CALUDE_condition_sufficient_not_necessary_l3249_324937


namespace NUMINAMATH_CALUDE_tangent_line_cubic_curve_l3249_324903

/-- Given a cubic function f(x) = x^3 + ax + b represented by curve C,
    if the line y = kx - 2 is tangent to C at point (1, 0),
    then k = 2 and f(x) = x^3 - x -/
theorem tangent_line_cubic_curve (a b k : ℝ) :
  let f : ℝ → ℝ := fun x ↦ x^3 + a*x + b
  let tangent_line : ℝ → ℝ := fun x ↦ k*x - 2
  (f 1 = 0) →
  (tangent_line 1 = 0) →
  (∀ x, tangent_line x ≤ f x) →
  (∃ x₀, x₀ ≠ 1 ∧ tangent_line x₀ = f x₀) →
  (k = 2 ∧ ∀ x, f x = x^3 - x) := by
  sorry

end NUMINAMATH_CALUDE_tangent_line_cubic_curve_l3249_324903


namespace NUMINAMATH_CALUDE_bananas_permutations_count_l3249_324963

/-- The number of unique permutations of the letters in "BANANAS" -/
def bananas_permutations : ℕ := 420

/-- The total number of letters in "BANANAS" -/
def total_letters : ℕ := 7

/-- The number of occurrences of 'A' in "BANANAS" -/
def a_count : ℕ := 3

/-- The number of occurrences of 'N' in "BANANAS" -/
def n_count : ℕ := 2

/-- Theorem stating that the number of unique permutations of the letters in "BANANAS"
    is equal to 420, given the total number of letters and the counts of repeated letters. -/
theorem bananas_permutations_count : 
  bananas_permutations = (Nat.factorial total_letters) / 
    ((Nat.factorial a_count) * (Nat.factorial n_count)) := by
  sorry

end NUMINAMATH_CALUDE_bananas_permutations_count_l3249_324963


namespace NUMINAMATH_CALUDE_quadratic_equation_positive_solutions_l3249_324952

theorem quadratic_equation_positive_solutions :
  ∃! (x : ℝ), x > 0 ∧ x^2 = -6*x + 9 := by sorry

end NUMINAMATH_CALUDE_quadratic_equation_positive_solutions_l3249_324952


namespace NUMINAMATH_CALUDE_division_problem_l3249_324976

theorem division_problem (L S q : ℕ) : 
  L - S = 1365 → 
  L = 1634 → 
  L = S * q + 20 → 
  q = 6 := by
sorry

end NUMINAMATH_CALUDE_division_problem_l3249_324976


namespace NUMINAMATH_CALUDE_symmetry_of_shifted_even_function_l3249_324912

-- Define a function f
variable (f : ℝ → ℝ)

-- Define what it means for a function to be even
def is_even (g : ℝ → ℝ) : Prop := ∀ x, g x = g (-x)

-- Define the axis of symmetry for a function
def axis_of_symmetry (g : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ x, g (a + x) = g (a - x)

-- State the theorem
theorem symmetry_of_shifted_even_function :
  is_even (fun x ↦ f (x - 2)) → axis_of_symmetry f (-2) :=
by sorry

end NUMINAMATH_CALUDE_symmetry_of_shifted_even_function_l3249_324912


namespace NUMINAMATH_CALUDE_science_fair_ratio_l3249_324943

/-- Represents the number of adults and children at the science fair -/
structure Attendance where
  adults : ℕ
  children : ℕ

/-- Calculates the total fee collected given an attendance -/
def totalFee (a : Attendance) : ℕ := 30 * a.adults + 15 * a.children

/-- Calculates the ratio of adults to children -/
def ratio (a : Attendance) : ℚ := a.adults / a.children

theorem science_fair_ratio : 
  ∃ (a : Attendance), 
    a.adults ≥ 1 ∧ 
    a.children ≥ 1 ∧ 
    totalFee a = 2250 ∧ 
    ∀ (b : Attendance), 
      b.adults ≥ 1 → 
      b.children ≥ 1 → 
      totalFee b = 2250 → 
      |ratio a - 2| ≤ |ratio b - 2| := by
  sorry

end NUMINAMATH_CALUDE_science_fair_ratio_l3249_324943


namespace NUMINAMATH_CALUDE_sin_sum_to_product_l3249_324965

theorem sin_sum_to_product (x : ℝ) : 
  Real.sin (5 * x) + Real.sin (7 * x) = 2 * Real.sin (6 * x) * Real.cos x := by
  sorry

end NUMINAMATH_CALUDE_sin_sum_to_product_l3249_324965


namespace NUMINAMATH_CALUDE_cake_cutting_l3249_324917

/-- Represents a rectangular grid --/
structure RectangularGrid :=
  (rows : ℕ)
  (cols : ℕ)

/-- The maximum number of pieces created by a single straight line cut in a rectangular grid --/
def max_pieces (grid : RectangularGrid) : ℕ :=
  grid.rows * grid.cols + (grid.rows + grid.cols - 1)

/-- The minimum number of straight cuts required to intersect all cells in a rectangular grid --/
def min_cuts (grid : RectangularGrid) : ℕ :=
  min grid.rows grid.cols

theorem cake_cutting (grid : RectangularGrid) 
  (h1 : grid.rows = 3) 
  (h2 : grid.cols = 5) : 
  max_pieces grid = 22 ∧ min_cuts grid = 3 := by
  sorry

#eval max_pieces ⟨3, 5⟩
#eval min_cuts ⟨3, 5⟩

end NUMINAMATH_CALUDE_cake_cutting_l3249_324917


namespace NUMINAMATH_CALUDE_green_lab_coat_pairs_l3249_324940

theorem green_lab_coat_pairs 
  (total_students : ℕ) 
  (white_coat_students : ℕ) 
  (green_coat_students : ℕ) 
  (total_pairs : ℕ) 
  (white_white_pairs : ℕ) 
  (h1 : total_students = 142)
  (h2 : white_coat_students = 68)
  (h3 : green_coat_students = 74)
  (h4 : total_pairs = 71)
  (h5 : white_white_pairs = 29)
  (h6 : total_students = white_coat_students + green_coat_students)
  (h7 : total_students = 2 * total_pairs) :
  ∃ (green_green_pairs : ℕ), green_green_pairs = 32 ∧ 
    green_green_pairs + white_white_pairs + (white_coat_students - 2 * white_white_pairs) = total_pairs :=
by
  sorry

end NUMINAMATH_CALUDE_green_lab_coat_pairs_l3249_324940


namespace NUMINAMATH_CALUDE_max_y_over_x_l3249_324977

/-- Given that x and y satisfy (x-2)^2 + y^2 = 1, the maximum value of y/x is √3/3 -/
theorem max_y_over_x (x y : ℝ) (h : (x - 2)^2 + y^2 = 1) :
  ∃ (max : ℝ), max = Real.sqrt 3 / 3 ∧ ∀ (x' y' : ℝ), (x' - 2)^2 + y'^2 = 1 → |y' / x'| ≤ max := by
  sorry

end NUMINAMATH_CALUDE_max_y_over_x_l3249_324977


namespace NUMINAMATH_CALUDE_sequence_properties_l3249_324983

/-- Sequence type representing our 0-1 sequence --/
def Sequence := ℕ → Bool

/-- Generate the nth term of the sequence --/
def generateTerm (n : ℕ) : Sequence := sorry

/-- Check if a sequence is periodic --/
def isPeriodic (s : Sequence) : Prop := sorry

/-- Get the nth digit of the sequence --/
def nthDigit (s : Sequence) (n : ℕ) : Bool := sorry

/-- Get the position of the nth occurrence of a digit --/
def nthOccurrence (s : Sequence) (digit : Bool) (n : ℕ) : ℕ := sorry

theorem sequence_properties (s : Sequence) :
  (s = generateTerm 0) →
  (¬ isPeriodic s) ∧
  (nthDigit s 1000 = true) ∧
  (nthOccurrence s true 10000 = 21328) ∧
  (∀ n : ℕ, nthOccurrence s true n = ⌊(2 + Real.sqrt 2) * n⌋) ∧
  (∀ n : ℕ, nthOccurrence s false n = ⌊Real.sqrt 2 * n⌋) := by
  sorry

end NUMINAMATH_CALUDE_sequence_properties_l3249_324983


namespace NUMINAMATH_CALUDE_math_chemistry_intersection_l3249_324905

/-- Represents the number of students in various groups and their intersections -/
structure StudentGroups where
  total : ℕ
  math : ℕ
  physics : ℕ
  chemistry : ℕ
  math_physics : ℕ
  physics_chemistry : ℕ
  math_chemistry : ℕ

/-- The given conditions for the student groups -/
def given_groups : StudentGroups :=
  { total := 36
  , math := 26
  , physics := 15
  , chemistry := 13
  , math_physics := 6
  , physics_chemistry := 4
  , math_chemistry := 8 }

/-- Theorem stating that the number of students in both math and chemistry is 8 -/
theorem math_chemistry_intersection (g : StudentGroups) (h : g = given_groups) :
  g.math_chemistry = 8 := by
  sorry

end NUMINAMATH_CALUDE_math_chemistry_intersection_l3249_324905


namespace NUMINAMATH_CALUDE_total_coins_l3249_324929

def coin_distribution (x : ℕ) : Prop :=
  let paul_coins := x
  let pete_coins := x * (x + 1) / 2
  pete_coins = 5 * paul_coins

theorem total_coins : ∃ x : ℕ, 
  coin_distribution x ∧ 
  x + 5 * x = 54 := by
  sorry

end NUMINAMATH_CALUDE_total_coins_l3249_324929


namespace NUMINAMATH_CALUDE_decagon_triangles_l3249_324923

theorem decagon_triangles : 
  let n : ℕ := 10  -- number of vertices in a regular decagon
  let k : ℕ := 3   -- number of vertices needed to form a triangle
  Nat.choose n k = 120 := by
sorry

end NUMINAMATH_CALUDE_decagon_triangles_l3249_324923


namespace NUMINAMATH_CALUDE_eleven_bonnets_per_orphanage_l3249_324999

/-- The number of bonnets Mrs. Young makes in a week and distributes to orphanages -/
def bonnet_distribution (monday : ℕ) : ℕ → ℕ :=
  fun orphanages =>
    let tuesday_wednesday := 2 * monday
    let thursday := monday + 5
    let friday := thursday - 5
    let total := monday + tuesday_wednesday + thursday + friday
    total / orphanages

/-- Theorem stating that given the conditions in the problem, each orphanage receives 11 bonnets -/
theorem eleven_bonnets_per_orphanage :
  bonnet_distribution 10 5 = 11 := by
  sorry

end NUMINAMATH_CALUDE_eleven_bonnets_per_orphanage_l3249_324999


namespace NUMINAMATH_CALUDE_min_value_of_f_l3249_324975

-- Define the function f(x) = x^2 + 6x + 13
def f (x : ℝ) : ℝ := x^2 + 6*x + 13

-- Theorem: The minimum value of f(x) is 4 for all real x
theorem min_value_of_f : ∀ x : ℝ, f x ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_min_value_of_f_l3249_324975


namespace NUMINAMATH_CALUDE_common_solution_conditions_l3249_324998

theorem common_solution_conditions (x y : ℝ) : 
  (∃ x : ℝ, x^2 + y^2 - 16 = 0 ∧ x^2 - 3*y - 12 = 0) ↔ (y = -4 ∨ y = 1) :=
by sorry

end NUMINAMATH_CALUDE_common_solution_conditions_l3249_324998


namespace NUMINAMATH_CALUDE_cube_volume_ratio_l3249_324993

theorem cube_volume_ratio (e : ℝ) (h : e > 0) :
  let small_cube_volume := e^3
  let large_cube_volume := (3*e)^3
  large_cube_volume = 27 * small_cube_volume := by
sorry

end NUMINAMATH_CALUDE_cube_volume_ratio_l3249_324993


namespace NUMINAMATH_CALUDE_parking_arrangements_l3249_324901

def parking_spaces : ℕ := 7
def car_models : ℕ := 3
def consecutive_empty : ℕ := 3

theorem parking_arrangements :
  (car_models.factorial) *
  (parking_spaces - car_models).choose 2 *
  ((parking_spaces - car_models - consecutive_empty + 1).factorial) = 72 := by
  sorry

end NUMINAMATH_CALUDE_parking_arrangements_l3249_324901


namespace NUMINAMATH_CALUDE_mechanic_parts_cost_l3249_324947

theorem mechanic_parts_cost (hourly_rate : ℕ) (daily_hours : ℕ) (work_days : ℕ) (total_cost : ℕ) : 
  hourly_rate = 60 →
  daily_hours = 8 →
  work_days = 14 →
  total_cost = 9220 →
  total_cost - (hourly_rate * daily_hours * work_days) = 2500 := by
sorry

end NUMINAMATH_CALUDE_mechanic_parts_cost_l3249_324947


namespace NUMINAMATH_CALUDE_t_100_gt_t_99_l3249_324969

/-- The number of ways to place n objects with weights 1 to n on a balance with equal weight in each pan. -/
def T (n : ℕ) : ℕ := sorry

/-- Theorem: T(100) is greater than T(99). -/
theorem t_100_gt_t_99 : T 100 > T 99 := by sorry

end NUMINAMATH_CALUDE_t_100_gt_t_99_l3249_324969


namespace NUMINAMATH_CALUDE_power_sum_cosine_l3249_324922

theorem power_sum_cosine (θ : Real) (x : ℂ) (n : ℕ) 
  (h1 : 0 < θ ∧ θ < π) 
  (h2 : x + 1/x = 2 * Real.cos θ) :
  x^n + 1/x^n = 2 * Real.cos (n * θ) := by
  sorry

end NUMINAMATH_CALUDE_power_sum_cosine_l3249_324922


namespace NUMINAMATH_CALUDE_smallest_n_for_digit_rearrangement_l3249_324944

/-- Represents a natural number as a list of its digits -/
def Digits : Type := List Nat

/-- Returns true if two lists of digits represent numbers that differ by a sequence of n ones -/
def differsBy111 (a b : Digits) (n : Nat) : Prop := sorry

/-- Returns true if two lists of digits are permutations of each other -/
def isPermutation (a b : Digits) : Prop := sorry

/-- Theorem: The smallest n for which there exist two numbers A and B,
    where B is a permutation of A's digits and A - B is n ones, is 9 -/
theorem smallest_n_for_digit_rearrangement :
  ∃ (a b : Digits),
    isPermutation a b ∧
    differsBy111 a b 9 ∧
    ∀ (n : Nat), n < 9 →
      ¬∃ (x y : Digits), isPermutation x y ∧ differsBy111 x y n :=
by sorry

end NUMINAMATH_CALUDE_smallest_n_for_digit_rearrangement_l3249_324944


namespace NUMINAMATH_CALUDE_hypotenuse_area_change_l3249_324958

theorem hypotenuse_area_change
  (a b c : ℝ)
  (h_right_triangle : a^2 + b^2 = c^2)
  (h_area_increase : (a - 5) * (b + 5) / 2 = a * b / 2 + 5)
  : c^2 - ((a - 5)^2 + (b + 5)^2) = 20 :=
by sorry

end NUMINAMATH_CALUDE_hypotenuse_area_change_l3249_324958


namespace NUMINAMATH_CALUDE_koala_fiber_consumption_l3249_324981

/-- Given a koala that absorbs 30% of the fiber it eats and absorbed 15 ounces of fiber in one day,
    prove that the total amount of fiber eaten is 50 ounces. -/
theorem koala_fiber_consumption (absorption_rate : ℝ) (absorbed_fiber : ℝ) (total_fiber : ℝ) : 
  absorption_rate = 0.30 →
  absorbed_fiber = 15 →
  total_fiber * absorption_rate = absorbed_fiber →
  total_fiber = 50 := by
  sorry


end NUMINAMATH_CALUDE_koala_fiber_consumption_l3249_324981


namespace NUMINAMATH_CALUDE_abs_x_minus_two_integral_l3249_324945

theorem abs_x_minus_two_integral : ∫ x in (0)..(4), |x - 2| = 4 := by sorry

end NUMINAMATH_CALUDE_abs_x_minus_two_integral_l3249_324945


namespace NUMINAMATH_CALUDE_no_baby_cries_iff_even_l3249_324915

/-- Represents the direction a baby is facing -/
inductive Direction
  | Right
  | Down
  | Left
  | Up

/-- Represents a position on the grid -/
structure Position where
  x : Nat
  y : Nat

/-- Represents the state of a baby on the grid -/
structure Baby where
  pos : Position
  dir : Direction

/-- The grid of babies -/
def Grid := List Baby

/-- Function to check if a position is within the grid -/
def isWithinGrid (n m : Nat) (pos : Position) : Prop :=
  0 ≤ pos.x ∧ pos.x < n ∧ 0 ≤ pos.y ∧ pos.y < m

/-- Function to move a baby according to the rules -/
def moveBaby (n m : Nat) (baby : Baby) : Baby :=
  sorry

/-- Function to check if any baby cries after a move -/
def anyCry (n m : Nat) (grid : Grid) : Prop :=
  sorry

/-- Main theorem: No baby cries if and only if n and m are even -/
theorem no_baby_cries_iff_even (n m : Nat) :
  (∀ (grid : Grid), ¬(anyCry n m grid)) ↔ (∃ (k l : Nat), n = 2 * k ∧ m = 2 * l) :=
  sorry

end NUMINAMATH_CALUDE_no_baby_cries_iff_even_l3249_324915


namespace NUMINAMATH_CALUDE_radical_simplification_l3249_324925

theorem radical_simplification (p : ℝ) :
  Real.sqrt (42 * p^2) * Real.sqrt (7 * p^2) * Real.sqrt (14 * p^2) = 14 * p^3 * Real.sqrt 21 := by
  sorry

end NUMINAMATH_CALUDE_radical_simplification_l3249_324925


namespace NUMINAMATH_CALUDE_total_birds_l3249_324932

def geese : ℕ := 58
def ducks : ℕ := 37

theorem total_birds : geese + ducks = 95 := by
  sorry

end NUMINAMATH_CALUDE_total_birds_l3249_324932


namespace NUMINAMATH_CALUDE_x_needs_seven_days_l3249_324933

/-- The number of days X needs to finish the remaining work after Y leaves -/
def days_for_x_to_finish (x_days y_days y_worked_days : ℕ) : ℚ :=
  let x_rate : ℚ := 1 / x_days
  let y_rate : ℚ := 1 / y_days
  let work_done_by_y : ℚ := y_rate * y_worked_days
  let remaining_work : ℚ := 1 - work_done_by_y
  remaining_work / x_rate

/-- Theorem stating that X needs 7 days to finish the remaining work -/
theorem x_needs_seven_days :
  days_for_x_to_finish 21 15 10 = 7 := by
  sorry

end NUMINAMATH_CALUDE_x_needs_seven_days_l3249_324933


namespace NUMINAMATH_CALUDE_bons_win_probability_main_theorem_l3249_324984

/-- The probability of rolling a six. -/
def prob_six : ℚ := 1/6

/-- The probability of not rolling a six. -/
def prob_not_six : ℚ := 1 - prob_six

/-- The probability that Mr. B. Bons wins the game. -/
def prob_bons_win : ℚ := 5/11

theorem bons_win_probability :
  prob_bons_win = prob_not_six * prob_six + prob_not_six * prob_not_six * prob_bons_win :=
by sorry

/-- The main theorem stating that the probability of Mr. B. Bons winning is 5/11. -/
theorem main_theorem : prob_bons_win = 5/11 :=
by sorry

end NUMINAMATH_CALUDE_bons_win_probability_main_theorem_l3249_324984


namespace NUMINAMATH_CALUDE_product_of_largest_primes_l3249_324995

/-- The largest two-digit prime number -/
def largest_two_digit_prime : ℕ := 97

/-- The largest four-digit prime number -/
def largest_four_digit_prime : ℕ := 9973

/-- Theorem stating that the product of the largest two-digit prime and the largest four-digit prime is 967781 -/
theorem product_of_largest_primes : 
  largest_two_digit_prime * largest_four_digit_prime = 967781 := by
  sorry

end NUMINAMATH_CALUDE_product_of_largest_primes_l3249_324995


namespace NUMINAMATH_CALUDE_normal_distribution_two_std_dev_below_mean_l3249_324987

theorem normal_distribution_two_std_dev_below_mean 
  (μ σ : ℝ) 
  (h_mean : μ = 14.5) 
  (h_std_dev : σ = 1.5) : 
  μ - 2 * σ = 11.5 := by
  sorry

end NUMINAMATH_CALUDE_normal_distribution_two_std_dev_below_mean_l3249_324987


namespace NUMINAMATH_CALUDE_negation_of_proposition_l3249_324988

theorem negation_of_proposition (x₀ : ℝ) : 
  ¬(x₀^2 + 2*x₀ + 2 ≤ 0) ↔ (x₀^2 + 2*x₀ + 2 > 0) := by sorry

end NUMINAMATH_CALUDE_negation_of_proposition_l3249_324988


namespace NUMINAMATH_CALUDE_divisibility_of_factorial_products_l3249_324920

theorem divisibility_of_factorial_products (a b : ℕ) : 
  Nat.Prime (a + b + 1) → 
  (∃ k : ℤ, (k = a.factorial * b.factorial + 1 ∨ k = a.factorial * b.factorial - 1) ∧ 
   (a + b + 1 : ℤ) ∣ k) := by
sorry

end NUMINAMATH_CALUDE_divisibility_of_factorial_products_l3249_324920


namespace NUMINAMATH_CALUDE_star_specific_value_l3249_324939

/-- Custom binary operation star -/
def star (a b : ℝ) : ℝ := 2 * a^2 + 3 * a * b + 2 * b^2

/-- Theorem: Given the custom operation star and specific values for a and b,
    prove that the result equals 113 -/
theorem star_specific_value : star 3 5 = 113 := by
  sorry

end NUMINAMATH_CALUDE_star_specific_value_l3249_324939


namespace NUMINAMATH_CALUDE_inequality_solution_set_l3249_324986

theorem inequality_solution_set (x : ℝ) :
  (4 * x^4 + x^2 + 4*x - 5 * x^2 * |x + 2| + 4 ≥ 0) ↔ 
  (x ≤ -1 ∨ ((1 - Real.sqrt 33) / 8 ≤ x ∧ x ≤ (1 + Real.sqrt 33) / 8) ∨ x ≥ 2) := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l3249_324986


namespace NUMINAMATH_CALUDE_polynomial_equality_l3249_324948

theorem polynomial_equality (g : ℝ → ℝ) : 
  (∀ x, 5 * x^5 + 3 * x^3 - 4 * x + 2 + g x = 7 * x^3 - 9 * x^2 + x + 5) →
  (∀ x, g x = -5 * x^5 + 4 * x^3 - 9 * x^2 + 5 * x + 3) :=
by sorry

end NUMINAMATH_CALUDE_polynomial_equality_l3249_324948


namespace NUMINAMATH_CALUDE_exam_average_problem_l3249_324900

theorem exam_average_problem (total_students : ℕ) (high_score_students : ℕ) (high_score : ℝ) (total_average : ℝ) :
  total_students = 25 →
  high_score_students = 10 →
  high_score = 90 →
  total_average = 84 →
  ∃ (low_score_students : ℕ) (low_score : ℝ),
    low_score_students + high_score_students = total_students ∧
    low_score = 80 ∧
    (low_score_students * low_score + high_score_students * high_score) / total_students = total_average ∧
    low_score_students = 15 :=
by sorry

end NUMINAMATH_CALUDE_exam_average_problem_l3249_324900


namespace NUMINAMATH_CALUDE_max_value_fraction_l3249_324994

theorem max_value_fraction (x : ℝ) :
  x^4 / (x^8 + 2*x^6 - 4*x^4 + 8*x^2 + 16) ≤ 1/12 ∧
  ∃ y : ℝ, y^4 / (y^8 + 2*y^6 - 4*y^4 + 8*y^2 + 16) = 1/12 :=
by sorry

end NUMINAMATH_CALUDE_max_value_fraction_l3249_324994


namespace NUMINAMATH_CALUDE_perpendicular_line_x_intercept_l3249_324953

/-- Given a line L1 defined by 4x + 5y = 15, prove that the x-intercept of the line L2
    that is perpendicular to L1 and has a y-intercept of -3 is 12/5. -/
theorem perpendicular_line_x_intercept :
  let L1 : ℝ → ℝ → Prop := λ x y ↦ 4 * x + 5 * y = 15
  let m1 : ℝ := -4 / 5  -- slope of L1
  let m2 : ℝ := 5 / 4   -- slope of L2 (perpendicular to L1)
  let b2 : ℝ := -3      -- y-intercept of L2
  let L2 : ℝ → ℝ → Prop := λ x y ↦ y = m2 * x + b2
  let x_intercept : ℝ := 12 / 5
  (∀ x y, L2 x y → y = 0 → x = x_intercept) :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_line_x_intercept_l3249_324953


namespace NUMINAMATH_CALUDE_common_ratio_of_geometric_sequence_l3249_324955

def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = q * a n

theorem common_ratio_of_geometric_sequence
  (a : ℕ → ℝ)
  (q : ℝ)
  (h_geometric : geometric_sequence a q)
  (h_odd_product : a 1 * a 3 * a 5 * a 7 * a 9 = 2)
  (h_even_product : a 2 * a 4 * a 6 * a 8 * a 10 = 64) :
  q = 2 :=
sorry

end NUMINAMATH_CALUDE_common_ratio_of_geometric_sequence_l3249_324955


namespace NUMINAMATH_CALUDE_y_sum_theorem_l3249_324928

theorem y_sum_theorem (y₁ y₂ y₃ y₄ y₅ : ℝ) 
  (eq1 : y₁ + 3*y₂ + 6*y₃ + 10*y₄ + 15*y₅ = 3)
  (eq2 : 3*y₁ + 6*y₂ + 10*y₃ + 15*y₄ + 21*y₅ = 20)
  (eq3 : 6*y₁ + 10*y₂ + 15*y₃ + 21*y₄ + 28*y₅ = 86)
  (eq4 : 10*y₁ + 15*y₂ + 21*y₃ + 28*y₄ + 36*y₅ = 225) :
  15*y₁ + 21*y₂ + 28*y₃ + 36*y₄ + 45*y₅ = 395 := by
  sorry

end NUMINAMATH_CALUDE_y_sum_theorem_l3249_324928


namespace NUMINAMATH_CALUDE_hyperbola_foci_distance_l3249_324956

theorem hyperbola_foci_distance (x y : ℝ) :
  x^2 / 25 - y^2 / 4 = 1 →
  ∃ (f₁ f₂ : ℝ × ℝ), (f₁.1 - f₂.1)^2 + (f₁.2 - f₂.2)^2 = 4 * 29 :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_foci_distance_l3249_324956


namespace NUMINAMATH_CALUDE_hexagon_tessellation_l3249_324992

-- Define a hexagon
structure Hexagon where
  vertices : Fin 6 → ℝ × ℝ

-- Define properties of the hexagon
def is_convex (h : Hexagon) : Prop :=
  sorry

def has_parallel_opposite_sides (h : Hexagon) : Prop :=
  sorry

def parallel_sides_length_one (h : Hexagon) : Prop :=
  sorry

-- Define tessellation
def can_tessellate_plane (h : Hexagon) : Prop :=
  sorry

-- Theorem statement
theorem hexagon_tessellation :
  ∃ (h : Hexagon), 
    is_convex h ∧ 
    has_parallel_opposite_sides h ∧ 
    parallel_sides_length_one h ∧ 
    can_tessellate_plane h :=
sorry

end NUMINAMATH_CALUDE_hexagon_tessellation_l3249_324992


namespace NUMINAMATH_CALUDE_miriam_homework_time_l3249_324966

theorem miriam_homework_time (laundry_time bathroom_time room_time total_time : ℕ) 
  (h1 : laundry_time = 30)
  (h2 : bathroom_time = 15)
  (h3 : room_time = 35)
  (h4 : total_time = 120) :
  total_time - (laundry_time + bathroom_time + room_time) = 40 := by
  sorry

end NUMINAMATH_CALUDE_miriam_homework_time_l3249_324966


namespace NUMINAMATH_CALUDE_macaron_distribution_theorem_l3249_324980

/-- The number of kids who receive macarons given the conditions of macaron production and distribution -/
def kids_receiving_macarons (mitch_total : ℕ) (mitch_burnt : ℕ) (joshua_extra : ℕ) 
  (joshua_undercooked : ℕ) (renz_burnt : ℕ) (leah_total : ℕ) (leah_undercooked : ℕ) 
  (first_kids : ℕ) (first_kids_macarons : ℕ) (remaining_kids_macarons : ℕ) : ℕ :=
  let miles_total := 2 * (mitch_total + joshua_extra)
  let renz_total := (3 * miles_total) / 4 - 1
  let total_good_macarons := (mitch_total - mitch_burnt) + 
    (mitch_total + joshua_extra - joshua_undercooked) + 
    miles_total + (renz_total - renz_burnt) + 
    (leah_total - leah_undercooked)
  let remaining_macarons := total_good_macarons - (first_kids * first_kids_macarons)
  first_kids + (remaining_macarons / remaining_kids_macarons)

theorem macaron_distribution_theorem : 
  kids_receiving_macarons 20 2 6 3 4 35 5 10 3 2 = 73 := by
  sorry

end NUMINAMATH_CALUDE_macaron_distribution_theorem_l3249_324980


namespace NUMINAMATH_CALUDE_system_solution_l3249_324909

theorem system_solution (x y b : ℝ) : 
  (4 * x + 2 * y = b) → 
  (3 * x + 7 * y = 3 * b) → 
  (x = 3) → 
  (b = 66) := by
sorry

end NUMINAMATH_CALUDE_system_solution_l3249_324909


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_mod_17_l3249_324979

def arithmetic_sequence_sum (a₁ : ℕ) (d : ℕ) (aₙ : ℕ) : ℕ :=
  let n := (aₙ - a₁) / d + 1
  n * (a₁ + aₙ) / 2

theorem arithmetic_sequence_sum_mod_17 :
  arithmetic_sequence_sum 4 6 100 % 17 = 0 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_mod_17_l3249_324979


namespace NUMINAMATH_CALUDE_correct_algebraic_simplification_l3249_324908

theorem correct_algebraic_simplification (x y : ℝ) : 3 * x^2 * y - 8 * y * x^2 = -5 * x^2 * y := by
  sorry

end NUMINAMATH_CALUDE_correct_algebraic_simplification_l3249_324908


namespace NUMINAMATH_CALUDE_factor_quadratic_l3249_324921

theorem factor_quadratic (m : ℤ) : 
  let s : ℤ := 5
  m^2 - s*m - 24 = (m - 8) * (m + 3) :=
by sorry

end NUMINAMATH_CALUDE_factor_quadratic_l3249_324921


namespace NUMINAMATH_CALUDE_floor_length_approximately_18_78_l3249_324957

/-- Represents the dimensions and painting cost of a rectangular floor. -/
structure Floor :=
  (breadth : ℝ)
  (paintCost : ℝ)
  (paintRate : ℝ)

/-- Calculates the length of the floor given its specifications. -/
def calculateFloorLength (floor : Floor) : ℝ :=
  let length := 3 * floor.breadth
  let area := floor.paintCost / floor.paintRate
  length

/-- Theorem stating that the calculated floor length is approximately 18.78 meters. -/
theorem floor_length_approximately_18_78 (floor : Floor) 
  (h1 : floor.paintCost = 529)
  (h2 : floor.paintRate = 3) :
  ∃ ε > 0, |calculateFloorLength floor - 18.78| < ε :=
sorry

end NUMINAMATH_CALUDE_floor_length_approximately_18_78_l3249_324957


namespace NUMINAMATH_CALUDE_not_always_true_converse_l3249_324967

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (perpendicular : Line → Plane → Prop)
variable (contained_in : Line → Plane → Prop)
variable (plane_perpendicular : Plane → Plane → Prop)

-- Define the lines and planes
variable (a b c : Line) (α β : Plane)

-- State the theorem
theorem not_always_true_converse
  (h1 : contained_in b α)
  (h2 : ¬ contained_in c α) :
  ¬ (∀ (α β : Plane), plane_perpendicular α β → perpendicular b β) :=
sorry

end NUMINAMATH_CALUDE_not_always_true_converse_l3249_324967


namespace NUMINAMATH_CALUDE_greatest_x_value_l3249_324936

def is_prime_power (n : ℕ) : Prop :=
  ∃ p k, p.Prime ∧ k > 0 ∧ n = p ^ k

theorem greatest_x_value (x : ℕ) 
  (h1 : Nat.lcm x (Nat.lcm 15 21) = 105)
  (h2 : is_prime_power x) :
  x ≤ 7 ∧ (∀ y, y > x → is_prime_power y → Nat.lcm y (Nat.lcm 15 21) ≠ 105) :=
sorry

end NUMINAMATH_CALUDE_greatest_x_value_l3249_324936


namespace NUMINAMATH_CALUDE_prime_extension_l3249_324989

theorem prime_extension (n : ℕ) (h1 : n ≥ 2) 
  (h2 : ∀ k : ℕ, 0 ≤ k ∧ k ≤ Real.sqrt (n / 3) → Nat.Prime (k^2 + k + n)) :
  ∀ k : ℕ, 0 ≤ k ∧ k ≤ n - 2 → Nat.Prime (k^2 + k + n) := by
  sorry

end NUMINAMATH_CALUDE_prime_extension_l3249_324989


namespace NUMINAMATH_CALUDE_sequence_well_defined_and_nonzero_l3249_324950

def x : ℕ → ℚ
  | 0 => 2
  | n + 1 => 2 * x n / ((x n)^2 - 1)

def y : ℕ → ℚ
  | 0 => 4
  | n + 1 => 2 * y n / ((y n)^2 - 1)

def z : ℕ → ℚ
  | 0 => 6/7
  | n + 1 => 2 * z n / ((z n)^2 - 1)

theorem sequence_well_defined_and_nonzero (n : ℕ) :
  (x n ≠ 1 ∧ x n ≠ -1) ∧
  (y n ≠ 1 ∧ y n ≠ -1) ∧
  (z n ≠ 1 ∧ z n ≠ -1) ∧
  x n + y n + z n ≠ 0 :=
sorry

end NUMINAMATH_CALUDE_sequence_well_defined_and_nonzero_l3249_324950


namespace NUMINAMATH_CALUDE_f_inequality_solution_f_minimum_value_l3249_324960

def f (x : ℝ) := |x - 2| + |x + 1|

theorem f_inequality_solution (x : ℝ) :
  f x > 4 ↔ x < -1.5 ∨ x > 2.5 := by sorry

theorem f_minimum_value :
  ∃ (a : ℝ), (∀ x, f x ≥ a) ∧ (∀ b, (∀ x, f x ≥ b) → b ≤ a) ∧ a = 3 := by sorry

end NUMINAMATH_CALUDE_f_inequality_solution_f_minimum_value_l3249_324960


namespace NUMINAMATH_CALUDE_stamp_problem_solution_l3249_324904

/-- Represents the number of stamps of each type -/
structure StampCounts where
  twopenny : ℕ
  penny : ℕ
  twohalfpenny : ℕ

/-- Calculates the total value of stamps in pence -/
def total_value (s : StampCounts) : ℕ :=
  2 * s.twopenny + s.penny + (5 * s.twohalfpenny) / 2

/-- Checks if the number of penny stamps is six times the number of twopenny stamps -/
def penny_constraint (s : StampCounts) : Prop :=
  s.penny = 6 * s.twopenny

/-- The main theorem stating the unique solution to the stamp problem -/
theorem stamp_problem_solution :
  ∃! s : StampCounts,
    total_value s = 60 ∧
    penny_constraint s ∧
    s.twopenny = 5 ∧
    s.penny = 30 ∧
    s.twohalfpenny = 8 := by
  sorry

end NUMINAMATH_CALUDE_stamp_problem_solution_l3249_324904


namespace NUMINAMATH_CALUDE_finite_decimal_fraction_l3249_324972

theorem finite_decimal_fraction (n : ℕ) : 
  (∃ (k m : ℕ), n * (2 * n - 1) = 2^k * 5^m) ↔ n = 1 :=
sorry

end NUMINAMATH_CALUDE_finite_decimal_fraction_l3249_324972


namespace NUMINAMATH_CALUDE_fred_basketball_games_l3249_324934

/-- The number of basketball games Fred went to last year -/
def last_year_games : ℕ := 36

/-- The number of games less Fred went to this year compared to last year -/
def games_difference : ℕ := 11

/-- The number of basketball games Fred went to this year -/
def this_year_games : ℕ := last_year_games - games_difference

theorem fred_basketball_games : this_year_games = 25 := by
  sorry

end NUMINAMATH_CALUDE_fred_basketball_games_l3249_324934


namespace NUMINAMATH_CALUDE_division_equality_l3249_324985

theorem division_equality : (180 : ℚ) / (12 + 13 * 2) = 90 / 19 := by sorry

end NUMINAMATH_CALUDE_division_equality_l3249_324985


namespace NUMINAMATH_CALUDE_change_percentage_l3249_324946

-- Define the prices of the items
def price1 : ℚ := 15.50
def price2 : ℚ := 3.25
def price3 : ℚ := 6.75

-- Define the amount paid
def amountPaid : ℚ := 50.00

-- Define the total price of items
def totalPrice : ℚ := price1 + price2 + price3

-- Define the change received
def change : ℚ := amountPaid - totalPrice

-- Define the percentage of change
def percentageChange : ℚ := (change / amountPaid) * 100

-- Theorem statement
theorem change_percentage : percentageChange = 49 := by
  sorry

end NUMINAMATH_CALUDE_change_percentage_l3249_324946


namespace NUMINAMATH_CALUDE_field_width_proof_l3249_324970

/-- Proves that a rectangular field with given conditions has a width of 20 feet -/
theorem field_width_proof (total_tape : ℝ) (field_length : ℝ) (leftover_tape : ℝ) 
  (h1 : total_tape = 250)
  (h2 : field_length = 60)
  (h3 : leftover_tape = 90) :
  let used_tape := total_tape - leftover_tape
  let perimeter := used_tape
  let width := (perimeter - 2 * field_length) / 2
  width = 20 := by sorry

end NUMINAMATH_CALUDE_field_width_proof_l3249_324970


namespace NUMINAMATH_CALUDE_arrangement_count_l3249_324918

def num_boxes : ℕ := 6
def num_digits : ℕ := 5

theorem arrangement_count :
  (num_boxes.factorial : ℕ) = 720 :=
sorry

end NUMINAMATH_CALUDE_arrangement_count_l3249_324918


namespace NUMINAMATH_CALUDE_equation_solution_l3249_324907

theorem equation_solution : ∃! x : ℝ, (10 : ℝ)^(2*x) * (100 : ℝ)^x = (1000 : ℝ)^4 :=
  by
    use 3
    constructor
    · -- Proof that x = 3 satisfies the equation
      sorry
    · -- Proof of uniqueness
      sorry

#check equation_solution

end NUMINAMATH_CALUDE_equation_solution_l3249_324907


namespace NUMINAMATH_CALUDE_solve_equation_l3249_324902

theorem solve_equation (x : ℝ) : (x^3)^(1/2) = 18 * 18^(1/9) → x = 18^(20/27) := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l3249_324902


namespace NUMINAMATH_CALUDE_pinwheel_area_is_four_l3249_324914

/-- Represents a point on a 2D grid -/
structure GridPoint where
  x : Int
  y : Int

/-- Represents a triangle on the grid -/
structure Triangle where
  v1 : GridPoint
  v2 : GridPoint
  v3 : GridPoint

/-- Represents the pinwheel design -/
structure Pinwheel where
  center : GridPoint
  arms : List Triangle

/-- Calculates the area of a triangle using Pick's theorem -/
def triangleArea (t : Triangle) : Int :=
  sorry

/-- Calculates the total area of the pinwheel -/
def pinwheelArea (p : Pinwheel) : Int :=
  sorry

/-- The main theorem to prove -/
theorem pinwheel_area_is_four :
  let center := GridPoint.mk 3 3
  let arm1 := Triangle.mk center (GridPoint.mk 6 3) (GridPoint.mk 3 6)
  let arm2 := Triangle.mk center (GridPoint.mk 3 6) (GridPoint.mk 0 3)
  let arm3 := Triangle.mk center (GridPoint.mk 0 3) (GridPoint.mk 3 0)
  let arm4 := Triangle.mk center (GridPoint.mk 3 0) (GridPoint.mk 6 3)
  let pinwheel := Pinwheel.mk center [arm1, arm2, arm3, arm4]
  pinwheelArea pinwheel = 4 :=
sorry

end NUMINAMATH_CALUDE_pinwheel_area_is_four_l3249_324914
