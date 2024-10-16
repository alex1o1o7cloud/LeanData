import Mathlib

namespace NUMINAMATH_CALUDE_almond_butter_ratio_is_one_third_l2563_256346

/-- The cost of a jar of peanut butter in dollars -/
def peanut_butter_cost : ℚ := 3

/-- The cost of a jar of almond butter in dollars -/
def almond_butter_cost : ℚ := 3 * peanut_butter_cost

/-- The additional cost per batch for almond butter cookies compared to peanut butter cookies -/
def additional_cost_per_batch : ℚ := 3

/-- The ratio of almond butter needed for a batch to the amount in a jar -/
def almond_butter_ratio : ℚ := additional_cost_per_batch / almond_butter_cost

theorem almond_butter_ratio_is_one_third :
  almond_butter_ratio = 1 / 3 := by sorry

end NUMINAMATH_CALUDE_almond_butter_ratio_is_one_third_l2563_256346


namespace NUMINAMATH_CALUDE_expression_simplification_l2563_256397

theorem expression_simplification :
  ((1 + 2 + 3 + 4 + 5 + 6) / 3) + ((3 * 5 + 12) / 4) = 13.75 := by
sorry

end NUMINAMATH_CALUDE_expression_simplification_l2563_256397


namespace NUMINAMATH_CALUDE_right_triangle_hypotenuse_l2563_256370

theorem right_triangle_hypotenuse : ∀ (a b c : ℝ),
  a = 60 →
  b = 100 →
  c^2 = a^2 + b^2 →
  c = 20 * Real.sqrt 34 :=
by sorry

end NUMINAMATH_CALUDE_right_triangle_hypotenuse_l2563_256370


namespace NUMINAMATH_CALUDE_min_sum_squares_exists_min_sum_squares_l2563_256354

def S : Finset ℤ := {3, -5, 0, 9, -2}

theorem min_sum_squares (a b c : ℤ) (ha : a ∈ S) (hb : b ∈ S) (hc : c ∈ S) 
  (hab : a ≠ b) (hbc : b ≠ c) (hac : a ≠ c) :
  13 ≤ a^2 + b^2 + c^2 :=
by sorry

theorem exists_min_sum_squares :
  ∃ (a b c : ℤ), a ∈ S ∧ b ∈ S ∧ c ∈ S ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c ∧ a^2 + b^2 + c^2 = 13 :=
by sorry

end NUMINAMATH_CALUDE_min_sum_squares_exists_min_sum_squares_l2563_256354


namespace NUMINAMATH_CALUDE_complex_power_32_l2563_256377

open Complex

theorem complex_power_32 : (((1 : ℂ) - I) / (Real.sqrt 2 : ℂ)) ^ 32 = 1 := by
  sorry

end NUMINAMATH_CALUDE_complex_power_32_l2563_256377


namespace NUMINAMATH_CALUDE_h_at_two_equals_negative_three_l2563_256349

/-- The function h(x) = -5x + 7 -/
def h (x : ℝ) : ℝ := -5 * x + 7

/-- Theorem stating that h(2) = -3 -/
theorem h_at_two_equals_negative_three : h 2 = -3 := by
  sorry

end NUMINAMATH_CALUDE_h_at_two_equals_negative_three_l2563_256349


namespace NUMINAMATH_CALUDE_range_m_when_not_p_false_range_m_when_p_or_q_true_and_p_and_q_false_l2563_256302

-- Define propositions p and q
def p (m : ℝ) : Prop := ∀ x ∈ Set.Icc 0 1, x^2 - m ≤ 0

def q (m : ℝ) : Prop := ∃ a b : ℝ, a > b ∧ b > 0 ∧
  ∀ x y : ℝ, x^2 / m^2 + y^2 / 4 = 1 ↔ (x/a)^2 + (y/b)^2 = 1

-- Theorem 1
theorem range_m_when_not_p_false (m : ℝ) :
  ¬(¬(p m)) → m ≥ 1 := by sorry

-- Theorem 2
theorem range_m_when_p_or_q_true_and_p_and_q_false (m : ℝ) :
  (p m ∨ q m) ∧ ¬(p m ∧ q m) →
  m ∈ Set.Ioi (-2) ∪ Set.Icc 1 2 := by sorry

end NUMINAMATH_CALUDE_range_m_when_not_p_false_range_m_when_p_or_q_true_and_p_and_q_false_l2563_256302


namespace NUMINAMATH_CALUDE_johns_calculation_l2563_256337

theorem johns_calculation (y : ℝ) : (y - 15) / 7 = 25 → (y - 7) / 5 = 36 := by
  sorry

end NUMINAMATH_CALUDE_johns_calculation_l2563_256337


namespace NUMINAMATH_CALUDE_regular_octagon_exterior_angle_regular_octagon_exterior_angle_is_45_l2563_256311

/-- The measure of an exterior angle in a regular octagon is 45 degrees. -/
theorem regular_octagon_exterior_angle : ℝ :=
  let n : ℕ := 8  -- number of sides in an octagon
  let interior_angle_sum : ℝ := (n - 2) * 180
  let interior_angle : ℝ := interior_angle_sum / n
  let exterior_angle : ℝ := 180 - interior_angle
  exterior_angle

/-- The measure of an exterior angle in a regular octagon is 45 degrees. -/
theorem regular_octagon_exterior_angle_is_45 :
  regular_octagon_exterior_angle = 45 := by
  sorry

end NUMINAMATH_CALUDE_regular_octagon_exterior_angle_regular_octagon_exterior_angle_is_45_l2563_256311


namespace NUMINAMATH_CALUDE_radical_axes_property_l2563_256328

-- Define a circle in 2D space
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define a line in 2D space (ax + by + c = 0)
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

-- Define the radical axis of two circles
def radical_axis (c1 c2 : Circle) : Line :=
  sorry

-- Define the property of lines being coincident
def coincident (l1 l2 l3 : Line) : Prop :=
  sorry

-- Define the property of lines being concurrent
def concurrent (l1 l2 l3 : Line) : Prop :=
  sorry

-- Define the property of lines being parallel
def parallel (l1 l2 l3 : Line) : Prop :=
  sorry

-- Theorem statement
theorem radical_axes_property (Γ₁ Γ₂ Γ₃ : Circle) :
  let Δ₁ := radical_axis Γ₁ Γ₂
  let Δ₂ := radical_axis Γ₂ Γ₃
  let Δ₃ := radical_axis Γ₃ Γ₁
  coincident Δ₁ Δ₂ Δ₃ ∨ concurrent Δ₁ Δ₂ Δ₃ ∨ parallel Δ₁ Δ₂ Δ₃ :=
by
  sorry

end NUMINAMATH_CALUDE_radical_axes_property_l2563_256328


namespace NUMINAMATH_CALUDE_f_properties_l2563_256314

noncomputable def f (x : ℝ) : ℝ := Real.log (x * (Real.exp x - Real.exp (-x)) / 2)

theorem f_properties :
  (∀ x, f (-x) = f x) ∧
  (∀ x y, 0 < x → x < y → f x < f y) :=
sorry

end NUMINAMATH_CALUDE_f_properties_l2563_256314


namespace NUMINAMATH_CALUDE_equation_solution_l2563_256317

theorem equation_solution : ∃ x : ℚ, 
  (((5 - 4*x) / (5 + 4*x) + 3) / (3 + (5 + 4*x) / (5 - 4*x))) - 
  (((5 - 4*x) / (5 + 4*x) + 2) / (2 + (5 + 4*x) / (5 - 4*x))) = 
  ((5 - 4*x) / (5 + 4*x) + 1) / (1 + (5 + 4*x) / (5 - 4*x)) ∧ 
  x = -5/14 := by
sorry

end NUMINAMATH_CALUDE_equation_solution_l2563_256317


namespace NUMINAMATH_CALUDE_simplify_expression_l2563_256305

theorem simplify_expression (x y : ℝ) : (3 * x^2 * y)^3 + (4 * x * y) * (y^4) = 27 * x^6 * y^3 + 4 * x * y^5 := by
  sorry

end NUMINAMATH_CALUDE_simplify_expression_l2563_256305


namespace NUMINAMATH_CALUDE_magnified_cell_size_l2563_256334

/-- The diameter of a certain type of cell in meters -/
def cell_diameter : ℝ := 1.56e-6

/-- The magnification factor -/
def magnification : ℝ := 1e6

/-- The magnified size of the cell -/
def magnified_size : ℝ := cell_diameter * magnification

theorem magnified_cell_size :
  magnified_size = 1.56 := by sorry

end NUMINAMATH_CALUDE_magnified_cell_size_l2563_256334


namespace NUMINAMATH_CALUDE_max_l_shapes_in_grid_l2563_256391

/-- Represents a 6x6 grid --/
def Grid := Fin 6 → Fin 6 → Bool

/-- An L-shape tetromino --/
structure LShape :=
  (position : Fin 6 × Fin 6)
  (orientation : Fin 4)

/-- Checks if an L-shape is within the grid bounds --/
def isWithinBounds (l : LShape) : Bool :=
  sorry

/-- Checks if two L-shapes overlap --/
def doOverlap (l1 l2 : LShape) : Bool :=
  sorry

/-- Checks if a set of L-shapes is valid (within bounds and non-overlapping) --/
def isValidPlacement (shapes : List LShape) : Bool :=
  sorry

/-- The main theorem stating the maximum number of L-shapes in a 6x6 grid --/
theorem max_l_shapes_in_grid :
  ∃ (shapes : List LShape),
    shapes.length = 4 ∧
    isValidPlacement shapes ∧
    ∀ (other_shapes : List LShape),
      isValidPlacement other_shapes →
      other_shapes.length ≤ 4 :=
sorry

end NUMINAMATH_CALUDE_max_l_shapes_in_grid_l2563_256391


namespace NUMINAMATH_CALUDE_polynomial_identity_l2563_256347

theorem polynomial_identity (x : ℝ) : 
  (x - 2)^5 + 5*(x - 2)^4 + 10*(x - 2)^3 + 10*(x - 2)^2 + 5*(x - 2) + 1 = (x - 1)^5 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_identity_l2563_256347


namespace NUMINAMATH_CALUDE_cafe_combinations_l2563_256385

/-- The number of drinks on the menu -/
def menu_size : ℕ := 8

/-- Whether Yann orders coffee -/
def yann_orders_coffee : Bool := sorry

/-- The number of options available to Camille -/
def camille_options : ℕ :=
  if yann_orders_coffee then menu_size - 1 else menu_size

/-- The number of combinations when Yann orders coffee -/
def coffee_combinations : ℕ := 1 * (menu_size - 1)

/-- The number of combinations when Yann doesn't order coffee -/
def non_coffee_combinations : ℕ := (menu_size - 1) * menu_size

/-- The total number of different combinations of drinks Yann and Camille can order -/
def total_combinations : ℕ := coffee_combinations + non_coffee_combinations

theorem cafe_combinations : total_combinations = 63 := by sorry

end NUMINAMATH_CALUDE_cafe_combinations_l2563_256385


namespace NUMINAMATH_CALUDE_root_quadratic_equation_l2563_256381

theorem root_quadratic_equation (m : ℝ) : 
  m^2 - 2*m - 3 = 0 → m^2 - 2*m + 2020 = 2023 := by
  sorry

end NUMINAMATH_CALUDE_root_quadratic_equation_l2563_256381


namespace NUMINAMATH_CALUDE_circle_area_ratio_l2563_256368

theorem circle_area_ratio (r : ℝ) (hr : r > 0) :
  let small_radius := (2 : ℝ) / 3 * r
  (π * small_radius^2) / (π * r^2) = 4 / 9 := by
sorry

end NUMINAMATH_CALUDE_circle_area_ratio_l2563_256368


namespace NUMINAMATH_CALUDE_num_factors_48_mult_6_eq_4_l2563_256325

/-- The number of positive factors of 48 that are also multiples of 6 -/
def num_factors_48_mult_6 : ℕ :=
  (Finset.filter (λ x => x ∣ 48 ∧ 6 ∣ x) (Finset.range 49)).card

/-- Theorem stating that the number of positive factors of 48 that are also multiples of 6 is 4 -/
theorem num_factors_48_mult_6_eq_4 : num_factors_48_mult_6 = 4 := by
  sorry

end NUMINAMATH_CALUDE_num_factors_48_mult_6_eq_4_l2563_256325


namespace NUMINAMATH_CALUDE_diophantine_equation_solutions_l2563_256396

theorem diophantine_equation_solutions :
  let S : Set (ℤ × ℤ × ℤ) := {(x, y, z) | 5 * x^2 + y^2 + 3 * z^2 - 2 * y * z = 30}
  S = {(1, 5, 0), (1, -5, 0), (-1, 5, 0), (-1, -5, 0)} := by
  sorry

end NUMINAMATH_CALUDE_diophantine_equation_solutions_l2563_256396


namespace NUMINAMATH_CALUDE_impossible_to_cover_modified_chessboard_l2563_256341

/-- Represents a chessboard square --/
structure Square where
  row : Fin 8
  col : Fin 8

/-- Defines the color of a square on the chessboard --/
def squareColor (s : Square) : Bool :=
  (s.row + s.col) % 2 = 0

/-- Represents the modified chessboard after removing two squares --/
def ModifiedChessboard : Set Square :=
  { s : Square | s ≠ ⟨0, 0⟩ ∧ s ≠ ⟨7, 7⟩ }

/-- A domino covers two adjacent squares --/
def validDomino (s1 s2 : Square) : Prop :=
  (s1.row = s2.row ∧ s1.col.val + 1 = s2.col.val) ∨
  (s1.col = s2.col ∧ s1.row.val + 1 = s2.row.val)

/-- A valid domino placement on the modified chessboard --/
def validPlacement (placement : Set (Square × Square)) : Prop :=
  ∀ (s1 s2 : Square), (s1, s2) ∈ placement →
    s1 ∈ ModifiedChessboard ∧ s2 ∈ ModifiedChessboard ∧ validDomino s1 s2

/-- The main theorem stating that it's impossible to cover the modified chessboard with dominos --/
theorem impossible_to_cover_modified_chessboard :
  ¬∃ (placement : Set (Square × Square)),
    validPlacement placement ∧
    (∀ s ∈ ModifiedChessboard, ∃ s1 s2, (s1, s2) ∈ placement ∧ (s = s1 ∨ s = s2)) :=
sorry

end NUMINAMATH_CALUDE_impossible_to_cover_modified_chessboard_l2563_256341


namespace NUMINAMATH_CALUDE_jamie_marbles_l2563_256356

theorem jamie_marbles (n : ℕ) : 
  n > 0 ∧ 
  (2 * n) % 5 = 0 ∧ 
  n % 3 = 0 ∧ 
  n ≥ 15 → 
  ∃ (blue red green yellow : ℕ), 
    blue = 2 * n / 5 ∧
    red = n / 3 ∧
    green = 4 ∧
    yellow = n - (blue + red + green) ∧
    yellow ≥ 0 ∧
    ∀ (m : ℕ), m < n → 
      (2 * m) % 5 = 0 → 
      m % 3 = 0 → 
      m - (2 * m / 5 + m / 3 + 4) < 0 :=
by sorry

end NUMINAMATH_CALUDE_jamie_marbles_l2563_256356


namespace NUMINAMATH_CALUDE_total_team_score_l2563_256330

def team_score (team_size : ℕ) (faye_score : ℕ) (other_player_score : ℕ) : ℕ :=
  faye_score + (team_size - 1) * other_player_score

theorem total_team_score :
  team_score 5 28 8 = 60 := by
  sorry

end NUMINAMATH_CALUDE_total_team_score_l2563_256330


namespace NUMINAMATH_CALUDE_yw_equals_two_l2563_256383

/-- A right triangle with specific side lengths and a median -/
structure RightTriangleWithMedian where
  /-- The length of side XY -/
  xy : ℝ
  /-- The length of side YZ -/
  yz : ℝ
  /-- The point where the median from X meets YZ -/
  w : ℝ
  /-- XY equals 3 -/
  xy_eq : xy = 3
  /-- YZ equals 4 -/
  yz_eq : yz = 4

/-- The length YW in a right triangle with specific side lengths and median -/
def yw (t : RightTriangleWithMedian) : ℝ := t.w

/-- Theorem: In a right triangle XYZ with XY = 3 and YZ = 4, 
    if W is where the median from X meets YZ, then YW = 2 -/
theorem yw_equals_two (t : RightTriangleWithMedian) : yw t = 2 := by
  sorry

end NUMINAMATH_CALUDE_yw_equals_two_l2563_256383


namespace NUMINAMATH_CALUDE_base_prime_repr_360_l2563_256306

/-- Base prime representation of a natural number --/
def base_prime_repr (n : ℕ) : List ℕ :=
  sorry

/-- The base prime representation of 360 --/
theorem base_prime_repr_360 : base_prime_repr 360 = [3, 2, 1] := by
  sorry

end NUMINAMATH_CALUDE_base_prime_repr_360_l2563_256306


namespace NUMINAMATH_CALUDE_bank_a_investment_l2563_256393

/-- Represents the investment scenario described in the problem -/
structure InvestmentScenario where
  total_investment : ℝ
  bank_a_rate : ℝ
  bank_b_rate : ℝ
  bank_b_fee : ℝ
  years : ℕ
  final_amount : ℝ

/-- Calculates the final amount after compound interest -/
def compound_interest (principal : ℝ) (rate : ℝ) (years : ℕ) : ℝ :=
  principal * (1 + rate) ^ years

/-- Theorem stating the correct amount invested in Bank A -/
theorem bank_a_investment (scenario : InvestmentScenario) 
  (h1 : scenario.total_investment = 2000)
  (h2 : scenario.bank_a_rate = 0.04)
  (h3 : scenario.bank_b_rate = 0.06)
  (h4 : scenario.bank_b_fee = 50)
  (h5 : scenario.years = 3)
  (h6 : scenario.final_amount = 2430) :
  ∃ (bank_a_amount : ℝ),
    bank_a_amount = 1625 ∧
    compound_interest bank_a_amount scenario.bank_a_rate scenario.years +
    compound_interest (scenario.total_investment - scenario.bank_b_fee - bank_a_amount) scenario.bank_b_rate scenario.years =
    scenario.final_amount :=
  sorry

end NUMINAMATH_CALUDE_bank_a_investment_l2563_256393


namespace NUMINAMATH_CALUDE_smallest_x_quadratic_l2563_256315

theorem smallest_x_quadratic : 
  let f : ℝ → ℝ := λ x => 5 * x^2 + 8 * x + 3
  ∃ x : ℝ, f x = 9 ∧ ∀ y : ℝ, f y = 9 → x ≤ y ∧ x = (-8 - 2 * Real.sqrt 46) / 10 :=
by sorry

end NUMINAMATH_CALUDE_smallest_x_quadratic_l2563_256315


namespace NUMINAMATH_CALUDE_max_sales_price_l2563_256392

/-- Represents the sales function for a product -/
def sales_function (x : ℝ) : ℝ := 400 - 20 * (x - 30)

/-- Represents the profit function for a product -/
def profit_function (x : ℝ) : ℝ := (x - 20) * (sales_function x)

/-- The unit purchase price of the product -/
def purchase_price : ℝ := 20

/-- The initial selling price of the product -/
def initial_price : ℝ := 30

/-- The initial sales volume in half a month -/
def initial_volume : ℝ := 400

/-- The price-volume relationship: change in volume per unit price increase -/
def price_volume_ratio : ℝ := -20

theorem max_sales_price : 
  ∃ (x : ℝ), x = 35 ∧ 
  ∀ (y : ℝ), profit_function y ≤ profit_function x :=
by sorry

end NUMINAMATH_CALUDE_max_sales_price_l2563_256392


namespace NUMINAMATH_CALUDE_min_value_of_expression_l2563_256312

theorem min_value_of_expression (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (h : x * y * z * (x + y + z) = 1) : 
  (x + y) * (y + z) ≥ 2 ∧ ∃ (x' y' z' : ℝ), x' > 0 ∧ y' > 0 ∧ z' > 0 ∧ 
    x' * y' * z' * (x' + y' + z') = 1 ∧ (x' + y') * (y' + z') = 2 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_expression_l2563_256312


namespace NUMINAMATH_CALUDE_wendy_bought_four_tables_l2563_256364

/-- The number of chairs Wendy bought -/
def num_chairs : ℕ := 4

/-- The time spent on each piece of furniture (in minutes) -/
def time_per_piece : ℕ := 6

/-- The total assembly time (in minutes) -/
def total_time : ℕ := 48

/-- The number of tables Wendy bought -/
def num_tables : ℕ := (total_time - num_chairs * time_per_piece) / time_per_piece

theorem wendy_bought_four_tables : num_tables = 4 := by
  sorry

end NUMINAMATH_CALUDE_wendy_bought_four_tables_l2563_256364


namespace NUMINAMATH_CALUDE_part_not_scrap_l2563_256351

/-- The probability of producing scrap in the first process -/
def p1 : ℝ := 0.01

/-- The probability of producing scrap in the second process -/
def p2 : ℝ := 0.02

/-- The probability that a part is not scrap after two independent processes -/
def prob_not_scrap : ℝ := (1 - p1) * (1 - p2)

theorem part_not_scrap : prob_not_scrap = 0.9702 := by sorry

end NUMINAMATH_CALUDE_part_not_scrap_l2563_256351


namespace NUMINAMATH_CALUDE_stocking_stuffers_l2563_256384

/-- Calculates the total cost of stocking stuffers for all kids and the number of unique combinations of books and toys for each kid's stocking. -/
theorem stocking_stuffers (num_kids : ℕ) (num_candy_canes : ℕ) (candy_cane_price : ℚ)
  (num_beanie_babies : ℕ) (beanie_baby_price : ℚ) (num_books : ℕ) (book_price : ℚ)
  (num_toys_per_stocking : ℕ) (num_toy_options : ℕ) (toy_price : ℚ) (gift_card_value : ℚ) :
  num_kids = 4 →
  num_candy_canes = 4 →
  candy_cane_price = 1/2 →
  num_beanie_babies = 2 →
  beanie_baby_price = 3 →
  num_books = 5 →
  book_price = 5 →
  num_toys_per_stocking = 3 →
  num_toy_options = 10 →
  toy_price = 1 →
  gift_card_value = 10 →
  (num_kids * (num_candy_canes * candy_cane_price +
               num_beanie_babies * beanie_baby_price +
               book_price +
               num_toys_per_stocking * toy_price +
               gift_card_value) = 104) ∧
  (num_books * (num_toy_options.choose num_toys_per_stocking) = 600) :=
by sorry

end NUMINAMATH_CALUDE_stocking_stuffers_l2563_256384


namespace NUMINAMATH_CALUDE_sand_art_problem_l2563_256319

/-- The amount of sand needed to fill one square inch -/
def sand_per_square_inch (rectangle_length rectangle_width square_side total_sand : ℕ) : ℚ :=
  total_sand / (rectangle_length * rectangle_width + square_side * square_side)

theorem sand_art_problem (rectangle_length rectangle_width square_side total_sand : ℕ) 
  (h1 : rectangle_length = 6)
  (h2 : rectangle_width = 7)
  (h3 : square_side = 5)
  (h4 : total_sand = 201) :
  sand_per_square_inch rectangle_length rectangle_width square_side total_sand = 3 := by
  sorry

end NUMINAMATH_CALUDE_sand_art_problem_l2563_256319


namespace NUMINAMATH_CALUDE_dot_path_length_rolling_cube_l2563_256357

/-- The path length of a dot on a rolling cube -/
theorem dot_path_length_rolling_cube (edge_length : ℝ) (h_edge : edge_length = 2) :
  let diagonal := edge_length * Real.sqrt 2
  let radius := diagonal / 2
  let quarter_turn := π * radius / 2
  let full_rotation := 4 * quarter_turn
  full_rotation = 2 * Real.sqrt 2 * π :=
by sorry

end NUMINAMATH_CALUDE_dot_path_length_rolling_cube_l2563_256357


namespace NUMINAMATH_CALUDE_no_equal_roots_for_quadratic_l2563_256327

/-- The quadratic equation x^2 - (p+1)x + (p-1) = 0 has no real values of p for which its roots are equal. -/
theorem no_equal_roots_for_quadratic :
  ¬ ∃ p : ℝ, ∃ x : ℝ, x^2 - (p + 1) * x + (p - 1) = 0 ∧
    ∀ y : ℝ, y^2 - (p + 1) * y + (p - 1) = 0 → y = x :=
by sorry

end NUMINAMATH_CALUDE_no_equal_roots_for_quadratic_l2563_256327


namespace NUMINAMATH_CALUDE_plains_total_area_l2563_256301

def plain_problem (region_B region_A total : ℕ) : Prop :=
  (region_B = 200) ∧
  (region_A = region_B - 50) ∧
  (total = region_A + region_B)

theorem plains_total_area : 
  ∃ (region_B region_A total : ℕ), 
    plain_problem region_B region_A total ∧ total = 350 :=
by
  sorry

end NUMINAMATH_CALUDE_plains_total_area_l2563_256301


namespace NUMINAMATH_CALUDE_equal_apple_distribution_l2563_256309

theorem equal_apple_distribution (total_apples : Nat) (num_students : Nat) 
  (h1 : total_apples = 360) (h2 : num_students = 60) :
  total_apples / num_students = 6 := by
  sorry

end NUMINAMATH_CALUDE_equal_apple_distribution_l2563_256309


namespace NUMINAMATH_CALUDE_point_in_first_quadrant_l2563_256386

theorem point_in_first_quadrant (x y : ℝ) (h1 : x + y = 2) (h2 : x - y = 1) : x > 0 ∧ y > 0 := by
  sorry

end NUMINAMATH_CALUDE_point_in_first_quadrant_l2563_256386


namespace NUMINAMATH_CALUDE_white_balls_count_l2563_256304

theorem white_balls_count (total : ℕ) (green yellow red purple : ℕ) (prob_not_red_purple : ℚ) :
  total = 100 →
  green = 30 →
  yellow = 10 →
  red = 37 →
  purple = 3 →
  prob_not_red_purple = 3/5 →
  ∃ white : ℕ, white = 20 ∧ total = white + green + yellow + red + purple :=
by sorry

end NUMINAMATH_CALUDE_white_balls_count_l2563_256304


namespace NUMINAMATH_CALUDE_find_other_number_l2563_256348

theorem find_other_number (A B : ℕ+) (h1 : A = 24) (h2 : Nat.gcd A B = 15) (h3 : Nat.lcm A B = 312) :
  B = 195 := by
  sorry

end NUMINAMATH_CALUDE_find_other_number_l2563_256348


namespace NUMINAMATH_CALUDE_john_jenny_meeting_point_l2563_256388

/-- Represents the running scenario of John and Jenny -/
structure RunningScenario where
  total_distance : ℝ
  uphill_distance : ℝ
  downhill_distance : ℝ
  john_start_time_diff : ℝ
  john_uphill_speed : ℝ
  john_downhill_speed : ℝ
  jenny_uphill_speed : ℝ
  jenny_downhill_speed : ℝ

/-- Calculates the meeting point of John and Jenny -/
def meeting_point (scenario : RunningScenario) : ℝ :=
  sorry

/-- Theorem stating that John and Jenny meet 45/32 km from the top of the hill -/
theorem john_jenny_meeting_point :
  let scenario : RunningScenario := {
    total_distance := 12,
    uphill_distance := 6,
    downhill_distance := 6,
    john_start_time_diff := 1/4,
    john_uphill_speed := 12,
    john_downhill_speed := 18,
    jenny_uphill_speed := 14,
    jenny_downhill_speed := 21
  }
  meeting_point scenario = 45/32 := by sorry

end NUMINAMATH_CALUDE_john_jenny_meeting_point_l2563_256388


namespace NUMINAMATH_CALUDE_fraction_product_is_one_l2563_256380

theorem fraction_product_is_one :
  (4 / 2) * (3 / 6) * (10 / 5) * (15 / 30) * (20 / 10) * (45 / 90) * (50 / 25) * (60 / 120) = 1 := by
  sorry

end NUMINAMATH_CALUDE_fraction_product_is_one_l2563_256380


namespace NUMINAMATH_CALUDE_negation_of_existence_negation_of_greater_than_prop_l2563_256365

theorem negation_of_existence (p : ℕ → Prop) : 
  (¬ ∃ n : ℕ, p n) ↔ (∀ n : ℕ, ¬ p n) := by sorry

theorem negation_of_greater_than_prop :
  (¬ ∃ n : ℕ, n^2 > 2^n) ↔ (∀ n : ℕ, n^2 ≤ 2^n) := by sorry

end NUMINAMATH_CALUDE_negation_of_existence_negation_of_greater_than_prop_l2563_256365


namespace NUMINAMATH_CALUDE_union_of_A_and_B_l2563_256369

def A : Set ℝ := {x | 3 < x ∧ x ≤ 7}
def B : Set ℝ := {x | 4 < x ∧ x ≤ 10}

theorem union_of_A_and_B : A ∪ B = {x | 3 < x ∧ x ≤ 10} := by
  sorry

end NUMINAMATH_CALUDE_union_of_A_and_B_l2563_256369


namespace NUMINAMATH_CALUDE_triangular_prism_area_bound_l2563_256300

/-- Given a triangular prism P-ABC with specified side lengths, 
    the sum of squared areas of triangles ABC and PBC is bounded. -/
theorem triangular_prism_area_bound 
  (AB : ℝ) (AC : ℝ) (PB : ℝ) (PC : ℝ)
  (h_AB : AB = Real.sqrt 3)
  (h_AC : AC = 1)
  (h_PB : PB = Real.sqrt 2)
  (h_PC : PC = Real.sqrt 2) :
  ∃ (S_ABC S_PBC : ℝ),
    (1/4 : ℝ) < S_ABC^2 + S_PBC^2 ∧ 
    S_ABC^2 + S_PBC^2 ≤ (7/4 : ℝ) :=
by sorry

end NUMINAMATH_CALUDE_triangular_prism_area_bound_l2563_256300


namespace NUMINAMATH_CALUDE_sequence_formula_l2563_256353

/-- Given a sequence {a_n} defined by a₁ = 2 and a_{n+1} = a_n + ln(1 + 1/n) for n ≥ 1,
    prove that a_n = 2 + ln(n) for all n ≥ 1 -/
theorem sequence_formula (a : ℕ → ℝ) (h1 : a 1 = 2) 
    (h2 : ∀ n : ℕ, n ≥ 1 → a (n + 1) = a n + Real.log (1 + 1 / n)) :
    ∀ n : ℕ, n ≥ 1 → a n = 2 + Real.log n := by
  sorry

end NUMINAMATH_CALUDE_sequence_formula_l2563_256353


namespace NUMINAMATH_CALUDE_tan_double_angle_l2563_256333

theorem tan_double_angle (α : Real) 
  (h : (Real.sin α + Real.cos α) / (Real.sin α - Real.cos α) = 1/2) : 
  Real.tan (2 * α) = 3/4 := by sorry

end NUMINAMATH_CALUDE_tan_double_angle_l2563_256333


namespace NUMINAMATH_CALUDE_intersection_when_m_is_one_union_equals_B_iff_l2563_256371

def A (m : ℝ) : Set ℝ := {x | 0 < x - m ∧ x - m < 3}
def B : Set ℝ := {x | x ≤ 0 ∨ x ≥ 3}

theorem intersection_when_m_is_one :
  A 1 ∩ B = {x | 3 ≤ x ∧ x < 4} := by sorry

theorem union_equals_B_iff (m : ℝ) :
  A m ∪ B = B ↔ m ≥ 3 ∨ m ≤ -3 := by sorry

end NUMINAMATH_CALUDE_intersection_when_m_is_one_union_equals_B_iff_l2563_256371


namespace NUMINAMATH_CALUDE_correct_mark_l2563_256324

theorem correct_mark (wrong_mark : ℕ) (class_size : ℕ) (average_increase : ℚ) 
  (h1 : wrong_mark = 79)
  (h2 : class_size = 68)
  (h3 : average_increase = 1/2) : 
  ∃ (correct_mark : ℕ), 
    (wrong_mark : ℚ) - correct_mark = average_increase * class_size ∧ 
    correct_mark = 45 := by
  sorry

end NUMINAMATH_CALUDE_correct_mark_l2563_256324


namespace NUMINAMATH_CALUDE_angle_sum_equal_pi_over_two_l2563_256398

theorem angle_sum_equal_pi_over_two (α β : Real) : 
  0 < α ∧ α < π/2 →
  0 < β ∧ β < π/2 →
  Real.sin α ^ 2 + Real.sin β ^ 2 = Real.sin (α + β) →
  α + β = π/2 := by
sorry

end NUMINAMATH_CALUDE_angle_sum_equal_pi_over_two_l2563_256398


namespace NUMINAMATH_CALUDE_calculation_proof_l2563_256340

theorem calculation_proof : -2^2 - Real.sqrt 9 + (-5)^2 * (2/5) = 3 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l2563_256340


namespace NUMINAMATH_CALUDE_not_mapping_A_to_B_l2563_256336

def A : Set ℝ := {x | 1 ≤ x ∧ x ≤ 2}
def B : Set ℝ := {y | 1 ≤ y ∧ y ≤ 4}

def f (x : ℝ) : ℝ := 4 - x^2

theorem not_mapping_A_to_B :
  ¬(∀ x ∈ A, f x ∈ B) :=
by sorry

end NUMINAMATH_CALUDE_not_mapping_A_to_B_l2563_256336


namespace NUMINAMATH_CALUDE_odd_factorial_product_equals_sum_factorial_l2563_256373

def oddFactorialProduct (m : ℕ) : ℕ := (List.range m).foldl (λ acc i => acc * Nat.factorial (2 * i + 1)) 1

def sumFirstNaturals (m : ℕ) : ℕ := m * (m + 1) / 2

theorem odd_factorial_product_equals_sum_factorial (m : ℕ) :
  oddFactorialProduct m = Nat.factorial (sumFirstNaturals m) ↔ m = 1 ∨ m = 2 ∨ m = 3 := by
  sorry

end NUMINAMATH_CALUDE_odd_factorial_product_equals_sum_factorial_l2563_256373


namespace NUMINAMATH_CALUDE_solution_product_l2563_256308

theorem solution_product (p q : ℝ) : 
  (p - 7) * (2 * p + 11) = p^2 - 19 * p + 60 →
  (q - 7) * (2 * q + 11) = q^2 - 19 * q + 60 →
  p ≠ q →
  (p - 2) * (q - 2) = -55 := by
sorry

end NUMINAMATH_CALUDE_solution_product_l2563_256308


namespace NUMINAMATH_CALUDE_hyperbola_asymptote_slope_l2563_256359

theorem hyperbola_asymptote_slope (x y m : ℝ) : 
  (x^2 / 144 - y^2 / 81 = 1) →  -- hyperbola equation
  (∃ (k : ℝ), y = k * m * x ∧ y = -k * m * x) →  -- asymptotes
  (m > 0) →  -- m is positive
  (m = 3/4) :=  -- conclusion
by sorry

end NUMINAMATH_CALUDE_hyperbola_asymptote_slope_l2563_256359


namespace NUMINAMATH_CALUDE_job_fair_problem_l2563_256355

/-- The probability of individual A being hired -/
def prob_A : ℚ := 4/9

/-- The probability of individuals B and C being hired -/
def prob_BC (t : ℚ) : ℚ := t/3

/-- The condition that t is between 0 and 3 -/
def t_condition (t : ℚ) : Prop := 0 < t ∧ t < 3

/-- The probability of all three individuals being hired -/
def prob_all (t : ℚ) : ℚ := prob_A * prob_BC t * prob_BC t

/-- The number of people hired from A and B -/
def ξ : Fin 3 → ℚ
| 0 => 0
| 1 => 1
| 2 => 2

/-- The probability distribution of ξ -/
def prob_ξ (t : ℚ) : Fin 3 → ℚ
| 0 => (1 - prob_A) * (1 - prob_BC t)
| 1 => prob_A * (1 - prob_BC t) + (1 - prob_A) * prob_BC t
| 2 => prob_A * prob_BC t

/-- The mathematical expectation of ξ -/
def expectation_ξ (t : ℚ) : ℚ :=
  (ξ 0) * (prob_ξ t 0) + (ξ 1) * (prob_ξ t 1) + (ξ 2) * (prob_ξ t 2)

theorem job_fair_problem (t : ℚ) (h : t_condition t) (h_prob : prob_all t = 16/81) :
  t = 2 ∧ expectation_ξ t = 10/9 := by
  sorry

end NUMINAMATH_CALUDE_job_fair_problem_l2563_256355


namespace NUMINAMATH_CALUDE_fly_probabilities_l2563_256320

def fly_path (n m : ℕ) : ℕ := Nat.choose (n + m) n

theorem fly_probabilities :
  let p1 := (fly_path 8 10 : ℚ) / 2^18
  let p2 := ((fly_path 5 6 : ℚ) * (fly_path 2 4) : ℚ) / 2^18
  let p3 := (2 * (fly_path 2 7 : ℚ) * (fly_path 6 3) + 
             2 * (fly_path 3 6 : ℚ) * (fly_path 5 4) + 
             (fly_path 4 5 : ℚ) * (fly_path 4 5)) / 2^18
  (p1 = (fly_path 8 10 : ℚ) / 2^18) ∧
  (p2 = ((fly_path 5 6 : ℚ) * (fly_path 2 4) : ℚ) / 2^18) ∧
  (p3 = (2 * (fly_path 2 7 : ℚ) * (fly_path 6 3) + 
         2 * (fly_path 3 6 : ℚ) * (fly_path 5 4) + 
         (fly_path 4 5 : ℚ) * (fly_path 4 5)) / 2^18) :=
by sorry

end NUMINAMATH_CALUDE_fly_probabilities_l2563_256320


namespace NUMINAMATH_CALUDE_reflection_about_y_eq_neg_x_l2563_256387

/-- Reflects a point (x, y) about the line y = -x -/
def reflect_about_y_eq_neg_x (p : ℝ × ℝ) : ℝ × ℝ :=
  let (x, y) := p
  (-y, -x)

/-- The original point -/
def original_point : ℝ × ℝ := (3, -7)

/-- The reflected point -/
def reflected_point : ℝ × ℝ := (7, -3)

theorem reflection_about_y_eq_neg_x :
  reflect_about_y_eq_neg_x original_point = reflected_point := by
  sorry

end NUMINAMATH_CALUDE_reflection_about_y_eq_neg_x_l2563_256387


namespace NUMINAMATH_CALUDE_louise_wallet_amount_l2563_256389

/-- The amount of money in Louise's wallet --/
def wallet_amount : ℕ → ℕ → ℕ → ℕ → ℕ
  | num_toys, toy_price, num_bears, bear_price =>
    num_toys * toy_price + num_bears * bear_price

/-- Theorem stating the amount in Louise's wallet --/
theorem louise_wallet_amount :
  wallet_amount 28 10 20 15 = 580 := by
  sorry

end NUMINAMATH_CALUDE_louise_wallet_amount_l2563_256389


namespace NUMINAMATH_CALUDE_pen_transaction_profit_l2563_256338

/-- Calculates the profit percentage for a given transaction -/
def profit_percent (items_bought : ℕ) (price_paid : ℕ) (discount_percent : ℚ) : ℚ :=
  let cost_per_item : ℚ := price_paid / items_bought
  let selling_price_per_item : ℚ := 1 - (discount_percent / 100)
  let total_revenue : ℚ := items_bought * selling_price_per_item
  let profit : ℚ := total_revenue - price_paid
  (profit / price_paid) * 100

/-- The profit percent for the given transaction is approximately 20.52% -/
theorem pen_transaction_profit :
  ∃ ε > 0, |profit_percent 56 46 1 - 20.52| < ε :=
sorry

end NUMINAMATH_CALUDE_pen_transaction_profit_l2563_256338


namespace NUMINAMATH_CALUDE_jessies_cars_l2563_256313

theorem jessies_cars (tommy : ℕ) (total : ℕ) (brother_extra : ℕ) :
  tommy = 3 →
  brother_extra = 5 →
  total = 17 →
  ∃ (jessie : ℕ), jessie = 3 ∧ tommy + jessie + (tommy + jessie + brother_extra) = total :=
by sorry

end NUMINAMATH_CALUDE_jessies_cars_l2563_256313


namespace NUMINAMATH_CALUDE_intersecting_rectangles_area_l2563_256375

/-- The total shaded area of two intersecting rectangles -/
theorem intersecting_rectangles_area (rect1_width rect1_height rect2_width rect2_height overlap_width overlap_height : ℕ) 
  (h1 : rect1_width = 4 ∧ rect1_height = 12)
  (h2 : rect2_width = 5 ∧ rect2_height = 7)
  (h3 : overlap_width = 4 ∧ overlap_height = 5) :
  rect1_width * rect1_height + rect2_width * rect2_height - overlap_width * overlap_height = 63 :=
by sorry

end NUMINAMATH_CALUDE_intersecting_rectangles_area_l2563_256375


namespace NUMINAMATH_CALUDE_only_one_true_l2563_256343

-- Define the four propositions
def prop1 : Prop := sorry
def prop2 : Prop := ∀ x : ℝ, x^2 + x + 1 ≥ 0
def prop3 : Prop := sorry
def prop4 : Prop := sorry

-- Theorem stating that only one proposition is true
theorem only_one_true : (prop1 ∧ ¬prop2 ∧ ¬prop3 ∧ ¬prop4) ∨
                        (¬prop1 ∧ prop2 ∧ ¬prop3 ∧ ¬prop4) ∨
                        (¬prop1 ∧ ¬prop2 ∧ prop3 ∧ ¬prop4) ∨
                        (¬prop1 ∧ ¬prop2 ∧ ¬prop3 ∧ prop4) :=
  sorry

end NUMINAMATH_CALUDE_only_one_true_l2563_256343


namespace NUMINAMATH_CALUDE_total_digits_100000_l2563_256316

def total_digits (n : ℕ) : ℕ :=
  let d1 := 9
  let d2 := 90 * 2
  let d3 := 900 * 3
  let d4 := 9000 * 4
  let d5 := (n - 10000 + 1) * 5
  let d6 := if n = 100000 then 6 else 0
  d1 + d2 + d3 + d4 + d5 + d6

theorem total_digits_100000 :
  total_digits 100000 = 488895 := by
  sorry

end NUMINAMATH_CALUDE_total_digits_100000_l2563_256316


namespace NUMINAMATH_CALUDE_grocery_bag_capacity_l2563_256367

theorem grocery_bag_capacity 
  (green_beans : ℝ) 
  (milk : ℝ) 
  (carrots : ℝ) 
  (additional_capacity : ℝ) :
  green_beans = 4 →
  milk = 6 →
  carrots = 2 * green_beans →
  additional_capacity = 2 →
  green_beans + milk + carrots + additional_capacity = 20 :=
by sorry

end NUMINAMATH_CALUDE_grocery_bag_capacity_l2563_256367


namespace NUMINAMATH_CALUDE_intersection_M_N_l2563_256307

def M : Set ℤ := {-1, 0, 1}

def N : Set ℤ := {x | ∃ a ∈ M, x = 2 * a}

theorem intersection_M_N : M ∩ N = {0} := by
  sorry

end NUMINAMATH_CALUDE_intersection_M_N_l2563_256307


namespace NUMINAMATH_CALUDE_book_purchase_total_price_l2563_256352

theorem book_purchase_total_price : 
  let total_books : ℕ := 90
  let math_books : ℕ := 53
  let math_book_price : ℕ := 4
  let history_book_price : ℕ := 5
  let history_books : ℕ := total_books - math_books
  let total_price : ℕ := math_books * math_book_price + history_books * history_book_price
  total_price = 397 := by
sorry

end NUMINAMATH_CALUDE_book_purchase_total_price_l2563_256352


namespace NUMINAMATH_CALUDE_brownies_made_next_morning_l2563_256382

def initial_brownies : ℕ := 2 * 12
def father_ate : ℕ := 8
def mooney_ate : ℕ := 4
def total_next_morning : ℕ := 36

theorem brownies_made_next_morning :
  total_next_morning - (initial_brownies - father_ate - mooney_ate) = 24 := by
  sorry

end NUMINAMATH_CALUDE_brownies_made_next_morning_l2563_256382


namespace NUMINAMATH_CALUDE_f_properties_l2563_256332

open Real

noncomputable def f (x : ℝ) := exp x - (1/2) * x^2

theorem f_properties :
  (∃ (m b : ℝ), m = 1 ∧ b = -1 ∧ ∀ x y, y = f x → m * x + b * y + 1 = 0) ∧
  (3/2 < f (log 2) ∧ f (log 2) < 2) ∧
  (∃! x, f x = 0) := by
  sorry

end NUMINAMATH_CALUDE_f_properties_l2563_256332


namespace NUMINAMATH_CALUDE_ten_thousandths_digit_of_seven_thirty_seconds_l2563_256321

theorem ten_thousandths_digit_of_seven_thirty_seconds (x : ℚ) : 
  x = 7 / 32 → (x * 10000).floor % 10 = 7 := by
  sorry

end NUMINAMATH_CALUDE_ten_thousandths_digit_of_seven_thirty_seconds_l2563_256321


namespace NUMINAMATH_CALUDE_scout_troop_profit_l2563_256331

-- Define the problem parameters
def candy_bars : ℕ := 1500
def buy_price : ℚ := 1 / 3
def transport_cost : ℕ := 50
def sell_price : ℚ := 3 / 5

-- Define the net profit calculation
def net_profit : ℚ :=
  candy_bars * sell_price - (candy_bars * buy_price + transport_cost)

-- Theorem statement
theorem scout_troop_profit :
  net_profit = 350 := by
  sorry

end NUMINAMATH_CALUDE_scout_troop_profit_l2563_256331


namespace NUMINAMATH_CALUDE_system_equation_solution_l2563_256344

theorem system_equation_solution (m : ℝ) : 
  (∃ x y : ℝ, x - y = m + 2 ∧ x + 3*y = m ∧ x + y = -2) → m = -3 := by
  sorry

end NUMINAMATH_CALUDE_system_equation_solution_l2563_256344


namespace NUMINAMATH_CALUDE_max_tiles_on_floor_l2563_256390

theorem max_tiles_on_floor (floor_length floor_width tile_length tile_width : ℕ) 
  (h1 : floor_length = 120) 
  (h2 : floor_width = 150) 
  (h3 : tile_length = 50) 
  (h4 : tile_width = 40) : 
  (max 
    ((floor_length / tile_length) * (floor_width / tile_width))
    ((floor_length / tile_width) * (floor_width / tile_length))) = 9 := by
  sorry

end NUMINAMATH_CALUDE_max_tiles_on_floor_l2563_256390


namespace NUMINAMATH_CALUDE_even_function_iff_a_eq_one_l2563_256322

def f (a : ℝ) (x : ℝ) : ℝ := (x + 1) * (x - a)

theorem even_function_iff_a_eq_one (a : ℝ) :
  (∀ x, f a x = f a (-x)) ↔ a = 1 := by sorry

end NUMINAMATH_CALUDE_even_function_iff_a_eq_one_l2563_256322


namespace NUMINAMATH_CALUDE_sarah_toads_count_l2563_256342

/-- Proves that Sarah has 100 toads given the conditions of the problem -/
theorem sarah_toads_count : ∀ (tim_toads jim_toads sarah_toads : ℕ),
  tim_toads = 30 →
  jim_toads = tim_toads + 20 →
  sarah_toads = 2 * jim_toads →
  sarah_toads = 100 := by
sorry

end NUMINAMATH_CALUDE_sarah_toads_count_l2563_256342


namespace NUMINAMATH_CALUDE_radio_cost_price_l2563_256329

/-- Calculates the cost price of an item given its selling price and loss percentage. -/
def cost_price (selling_price : ℚ) (loss_percentage : ℚ) : ℚ :=
  selling_price / (1 - loss_percentage / 100)

/-- Proves that the cost price of a radio sold for 1305 with a 13% loss is 1500. -/
theorem radio_cost_price : cost_price 1305 13 = 1500 := by
  sorry

end NUMINAMATH_CALUDE_radio_cost_price_l2563_256329


namespace NUMINAMATH_CALUDE_banana_cream_pie_angle_l2563_256345

def total_students : ℕ := 48
def chocolate_preference : ℕ := 15
def apple_preference : ℕ := 9
def blueberry_preference : ℕ := 11

def remaining_students : ℕ := total_students - (chocolate_preference + apple_preference + blueberry_preference)

def banana_cream_preference : ℕ := remaining_students / 2

theorem banana_cream_pie_angle :
  (banana_cream_preference : ℝ) / total_students * 360 = 45 := by
  sorry

end NUMINAMATH_CALUDE_banana_cream_pie_angle_l2563_256345


namespace NUMINAMATH_CALUDE_find_a_value_l2563_256378

theorem find_a_value (a b : ℝ) (h1 : 0 < a) (h2 : 0 < b) 
    (h3 : a^b = b^a) (h4 : b = 4*a) : a = (4 : ℝ)^(1/3) := by
  sorry

end NUMINAMATH_CALUDE_find_a_value_l2563_256378


namespace NUMINAMATH_CALUDE_product_of_sum_and_cube_sum_l2563_256361

theorem product_of_sum_and_cube_sum (c d : ℝ) 
  (h1 : c + d = 10) 
  (h2 : c^3 + d^3 = 370) : 
  c * d = 21 := by
sorry

end NUMINAMATH_CALUDE_product_of_sum_and_cube_sum_l2563_256361


namespace NUMINAMATH_CALUDE_sin_negative_600_degrees_l2563_256394

theorem sin_negative_600_degrees : Real.sin ((-600 : ℝ) * π / 180) = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_negative_600_degrees_l2563_256394


namespace NUMINAMATH_CALUDE_vegetables_for_movie_day_l2563_256350

theorem vegetables_for_movie_day 
  (points_needed : ℕ) 
  (points_per_vegetable : ℕ) 
  (num_students : ℕ) 
  (num_days : ℕ) 
  (h1 : points_needed = 200) 
  (h2 : points_per_vegetable = 2) 
  (h3 : num_students = 25) 
  (h4 : num_days = 10) : 
  (points_needed / (points_per_vegetable * num_students * (num_days / 2))) = 2 := by
  sorry

end NUMINAMATH_CALUDE_vegetables_for_movie_day_l2563_256350


namespace NUMINAMATH_CALUDE_quadratic_roots_distinct_and_sum_constant_l2563_256376

/-- Represents the quadratic equation -3(x-1)^2 + m = 0 --/
def quadratic_equation (m : ℝ) (x : ℝ) : Prop :=
  -3 * (x - 1)^2 + m = 0

/-- The discriminant of the quadratic equation --/
def discriminant (m : ℝ) : ℝ :=
  12 * m

theorem quadratic_roots_distinct_and_sum_constant (m : ℝ) (h : m > 0) :
  ∃ (x₁ x₂ : ℝ),
    x₁ ≠ x₂ ∧
    quadratic_equation m x₁ ∧
    quadratic_equation m x₂ ∧
    x₁ + x₂ = 2 :=
sorry

end NUMINAMATH_CALUDE_quadratic_roots_distinct_and_sum_constant_l2563_256376


namespace NUMINAMATH_CALUDE_apollo_chariot_cost_l2563_256374

/-- The number of months in a year -/
def months_in_year : ℕ := 12

/-- The number of months before the price increase -/
def months_before_increase : ℕ := 6

/-- The initial price in golden apples -/
def initial_price : ℕ := 3

/-- The price increase factor -/
def price_increase_factor : ℕ := 2

/-- The total cost of chariot wheels for Apollo in golden apples for a year -/
def total_cost : ℕ := 
  (months_before_increase * initial_price) + 
  ((months_in_year - months_before_increase) * (initial_price * price_increase_factor))

/-- Theorem stating that the total cost for Apollo is 54 golden apples -/
theorem apollo_chariot_cost : total_cost = 54 := by
  sorry

end NUMINAMATH_CALUDE_apollo_chariot_cost_l2563_256374


namespace NUMINAMATH_CALUDE_opposite_of_negative_one_sixth_l2563_256366

theorem opposite_of_negative_one_sixth (x : ℚ) : x = -1/6 → -x = 1/6 := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_negative_one_sixth_l2563_256366


namespace NUMINAMATH_CALUDE_binomial_7_2_l2563_256360

theorem binomial_7_2 : Nat.choose 7 2 = 21 := by
  sorry

end NUMINAMATH_CALUDE_binomial_7_2_l2563_256360


namespace NUMINAMATH_CALUDE_smaller_rectangle_dimensions_l2563_256339

theorem smaller_rectangle_dimensions (square_side : ℝ) (small_width : ℝ) :
  square_side = 10 →
  small_width > 0 →
  small_width < square_side →
  small_width * square_side = (1 / 3) * square_side * square_side →
  (small_width, square_side) = (10 / 3, 10) :=
by sorry

end NUMINAMATH_CALUDE_smaller_rectangle_dimensions_l2563_256339


namespace NUMINAMATH_CALUDE_six_digit_divisibility_l2563_256335

theorem six_digit_divisibility (a b c : ℕ) 
  (h1 : a < 10) (h2 : b < 10) (h3 : c < 10) : 
  ∃ k : ℤ, (a * 100000 + b * 10000 + c * 1000 + a * 100 + b * 10 + c : ℤ) = 1001 * k :=
sorry

end NUMINAMATH_CALUDE_six_digit_divisibility_l2563_256335


namespace NUMINAMATH_CALUDE_planar_figures_l2563_256326

-- Define the types of figures
inductive Figure
  | TwoSegmentPolyline
  | ThreeSegmentPolyline
  | TriangleClosed
  | QuadrilateralEqualOppositeSides
  | Trapezoid

-- Define what it means for a figure to be planar
def isPlanar (f : Figure) : Prop :=
  match f with
  | Figure.TwoSegmentPolyline => true
  | Figure.ThreeSegmentPolyline => false
  | Figure.TriangleClosed => true
  | Figure.QuadrilateralEqualOppositeSides => false
  | Figure.Trapezoid => true

-- Theorem statement
theorem planar_figures :
  (∀ f : Figure, isPlanar f ↔ (f = Figure.TwoSegmentPolyline ∨ f = Figure.TriangleClosed ∨ f = Figure.Trapezoid)) :=
by sorry

end NUMINAMATH_CALUDE_planar_figures_l2563_256326


namespace NUMINAMATH_CALUDE_inequality_proof_l2563_256362

theorem inequality_proof (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  (x + 8*y + 2*z) * (x + 2*y + z) * (x + 4*y + 4*z) ≥ 256*x*y*z := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l2563_256362


namespace NUMINAMATH_CALUDE_marble_jar_problem_l2563_256318

/-- The number of marbles in the jar -/
def M : ℕ := 364

/-- The initial number of people -/
def initial_people : ℕ := 26

/-- The number of people who join later -/
def joining_people : ℕ := 2

/-- The number of marbles each person gets in the initial distribution -/
def initial_distribution : ℕ := M / initial_people

/-- The number of marbles each person would get after more people join -/
def later_distribution : ℕ := M / (initial_people + joining_people)

theorem marble_jar_problem :
  (M = initial_people * initial_distribution) ∧
  (M = (initial_people + joining_people) * (initial_distribution - 1)) :=
sorry

end NUMINAMATH_CALUDE_marble_jar_problem_l2563_256318


namespace NUMINAMATH_CALUDE_ascending_order_real_numbers_l2563_256372

theorem ascending_order_real_numbers : -6 < (0 : ℝ) ∧ 0 < Real.sqrt 5 ∧ Real.sqrt 5 < Real.pi := by
  sorry

end NUMINAMATH_CALUDE_ascending_order_real_numbers_l2563_256372


namespace NUMINAMATH_CALUDE_prime_multiple_all_ones_l2563_256323

theorem prime_multiple_all_ones (p : ℕ) (hp : Prime p) (hp_not_two : p ≠ 2) (hp_not_five : p ≠ 5) :
  ∃ k : ℕ, ∃ n : ℕ, p * k = 10^n - 1 :=
sorry

end NUMINAMATH_CALUDE_prime_multiple_all_ones_l2563_256323


namespace NUMINAMATH_CALUDE_f_form_when_a_equals_b_f_max_value_with_three_zeros_l2563_256303

-- Define the function f(x) with parameters a and b
def f (a b x : ℝ) : ℝ := (x - a) * (x^2 - (b - 1) * x - b)

-- Theorem 1: When a = b = 1, f(x) = (x-1)^2(x+1)
theorem f_form_when_a_equals_b (x : ℝ) :
  f 1 1 x = (x - 1)^2 * (x + 1) := by sorry

-- Theorem 2: When f(x) = x(x-1)(x+1), the maximum value is 2√3/9
theorem f_max_value_with_three_zeros :
  let g (x : ℝ) := x * (x - 1) * (x + 1)
  ∃ (x_max : ℝ), g x_max = 2 * Real.sqrt 3 / 9 ∧ ∀ (x : ℝ), g x ≤ g x_max := by sorry

end NUMINAMATH_CALUDE_f_form_when_a_equals_b_f_max_value_with_three_zeros_l2563_256303


namespace NUMINAMATH_CALUDE_andrey_stamps_problem_l2563_256363

theorem andrey_stamps_problem :
  ∃! x : ℕ, x % 3 = 1 ∧ x % 5 = 3 ∧ x % 7 = 5 ∧ 150 < x ∧ x ≤ 300 ∧ x = 208 := by
  sorry

end NUMINAMATH_CALUDE_andrey_stamps_problem_l2563_256363


namespace NUMINAMATH_CALUDE_right_triangle_rotation_volume_l2563_256379

/-- Given a right-angled triangle with legs b and c, and hypotenuse a, 
    where b + c = 25 and angle α = 61°55'40", 
    the volume of the solid formed by rotating the triangle around its hypotenuse 
    is approximately 887. -/
theorem right_triangle_rotation_volume 
  (b c a : ℝ) (α : Real) 
  (h_right_angle : b^2 + c^2 = a^2)
  (h_sum : b + c = 25)
  (h_angle : α = Real.pi * (61 + 55/60 + 40/3600) / 180) :
  ∃ (V : ℝ), abs (V - 887) < 1 ∧ V = (1/3) * Real.pi * c * (a * b / Real.sqrt (a^2 + b^2))^2 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_rotation_volume_l2563_256379


namespace NUMINAMATH_CALUDE_martin_oranges_l2563_256310

/-- Represents the number of fruits Martin has initially -/
def initial_fruits : ℕ := 150

/-- Represents the number of oranges Martin has after eating half of his fruits -/
def oranges : ℕ := 50

/-- Represents the number of limes Martin has after eating half of his fruits -/
def limes : ℕ := 25

/-- Proves that Martin has 50 oranges after eating half of his fruits -/
theorem martin_oranges :
  (oranges + limes = initial_fruits / 2) ∧
  (oranges = 2 * limes) ∧
  (oranges = 50) :=
sorry

end NUMINAMATH_CALUDE_martin_oranges_l2563_256310


namespace NUMINAMATH_CALUDE_perpendicular_lines_theorem_l2563_256395

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the perpendicular relation between planes
variable (perp_planes : Plane → Plane → Prop)

-- Define the perpendicular relation between a line and a plane
variable (perp_line_plane : Line → Plane → Prop)

-- Define the parallel relation between a line and a plane
variable (parallel_line_plane : Line → Plane → Prop)

-- Define the perpendicular relation between lines
variable (perp_lines : Line → Line → Prop)

-- Define the intersection operation between planes
variable (intersection : Plane → Plane → Line)

-- State the theorem
theorem perpendicular_lines_theorem 
  (α β : Plane) (a b l : Line) 
  (h1 : perp_planes α β)
  (h2 : intersection α β = l)
  (h3 : parallel_line_plane a α)
  (h4 : perp_line_plane b β) :
  perp_lines b l :=
sorry

end NUMINAMATH_CALUDE_perpendicular_lines_theorem_l2563_256395


namespace NUMINAMATH_CALUDE_journey_speed_l2563_256358

/-- Proves that given a journey of 200 km completed in 10 hours with constant speed throughout, the speed of travel is 20 km/hr. -/
theorem journey_speed (total_distance : ℝ) (total_time : ℝ) (speed : ℝ) 
  (h1 : total_distance = 200) 
  (h2 : total_time = 10) 
  (h3 : speed * total_time = total_distance) : 
  speed = 20 := by
  sorry

end NUMINAMATH_CALUDE_journey_speed_l2563_256358


namespace NUMINAMATH_CALUDE_smallest_prime_8_less_than_square_l2563_256399

theorem smallest_prime_8_less_than_square : 
  ∃ (n : ℕ), 17 = n^2 - 8 ∧ 
  Prime 17 ∧ 
  ∀ (m : ℕ) (p : ℕ), m < n → p = m^2 - 8 → p ≤ 0 ∨ ¬ Prime p :=
by sorry

end NUMINAMATH_CALUDE_smallest_prime_8_less_than_square_l2563_256399
