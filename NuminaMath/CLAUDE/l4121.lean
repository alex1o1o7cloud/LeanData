import Mathlib

namespace NUMINAMATH_CALUDE_minimum_value_implies_a_l4121_412167

def f (a : ℝ) (x : ℝ) : ℝ := x^3 - 3*x + a

theorem minimum_value_implies_a (a : ℝ) :
  (∀ x ∈ Set.Icc (-2 : ℝ) 0, f a x ≥ 1) ∧
  (∃ x ∈ Set.Icc (-2 : ℝ) 0, f a x = 1) →
  a = 3 := by
  sorry

end NUMINAMATH_CALUDE_minimum_value_implies_a_l4121_412167


namespace NUMINAMATH_CALUDE_inverse_proportion_l4121_412144

theorem inverse_proportion (x y : ℝ) (k : ℝ) (h1 : x * y = k) (h2 : 10 * 3 = k) :
  -15 * -2 = k := by
  sorry

end NUMINAMATH_CALUDE_inverse_proportion_l4121_412144


namespace NUMINAMATH_CALUDE_number_of_valid_outfits_l4121_412117

/-- Represents the total number of shirts -/
def num_shirts : ℕ := 8

/-- Represents the total number of pants -/
def num_pants : ℕ := 4

/-- Represents the total number of hats -/
def num_hats : ℕ := 6

/-- Represents the number of colors for pants -/
def num_pants_colors : ℕ := 4

/-- Represents the number of colors for shirts and hats -/
def num_shirt_hat_colors : ℕ := 6

/-- Represents the number of colors shared between shirts and hats -/
def num_shared_colors : ℕ := 4

/-- Calculates the total number of outfits without restrictions -/
def total_outfits : ℕ := num_shirts * num_pants * num_hats

/-- Calculates the number of outfits where shirt and hat have the same color -/
def same_color_outfits : ℕ := (num_shirts / num_shirt_hat_colors) * (num_hats / num_shirt_hat_colors) * num_shared_colors * num_pants

/-- Theorem stating the number of valid outfits -/
theorem number_of_valid_outfits : total_outfits - same_color_outfits = 160 := by
  sorry

end NUMINAMATH_CALUDE_number_of_valid_outfits_l4121_412117


namespace NUMINAMATH_CALUDE_expression_evaluation_l4121_412116

theorem expression_evaluation :
  let x : ℚ := -1
  let expr := (x - 3) / (2 * x - 4) / ((5 / (x - 2)) - x - 2)
  expr = -1/4 := by
sorry

end NUMINAMATH_CALUDE_expression_evaluation_l4121_412116


namespace NUMINAMATH_CALUDE_slope_angle_of_parametric_line_l4121_412164

/-- The slope angle of a line given by parametric equations -/
theorem slope_angle_of_parametric_line :
  let x : ℝ → ℝ := λ t ↦ 5 - 3 * t
  let y : ℝ → ℝ := λ t ↦ 3 + Real.sqrt 3 * t
  (∃ α : ℝ, α = 150 * π / 180 ∧
    ∀ t : ℝ, (y t - y 0) / (x t - x 0) = Real.tan α) :=
by sorry

end NUMINAMATH_CALUDE_slope_angle_of_parametric_line_l4121_412164


namespace NUMINAMATH_CALUDE_no_infected_computers_after_attack_l4121_412154

/-- Represents a computer in the ring. -/
structure Computer where
  id : Nat
  infected : Bool

/-- Represents a virus in the attack. -/
structure Virus where
  id : Nat
  active : Bool

/-- Represents the state of the computer network. -/
structure NetworkState where
  computers : List Computer
  viruses : List Virus

/-- The rules of virus propagation and computer infection. -/
def propagateVirus (state : NetworkState) : NetworkState :=
  sorry

/-- The final state of the network after all viruses have been processed. -/
def finalState (n : Nat) : NetworkState :=
  sorry

/-- Counts the number of infected computers in the network. -/
def countInfected (state : NetworkState) : Nat :=
  sorry

/-- Theorem stating that after the attack, no computers remain infected. -/
theorem no_infected_computers_after_attack (n : Nat) :
  countInfected (finalState n) = 0 := by
  sorry

end NUMINAMATH_CALUDE_no_infected_computers_after_attack_l4121_412154


namespace NUMINAMATH_CALUDE_solve_candy_problem_l4121_412179

def candy_problem (candy_from_neighbors : ℝ) : Prop :=
  let candy_from_sister : ℝ := 5.0
  let candy_per_day : ℝ := 8.0
  let days_lasted : ℝ := 2.0
  let total_candy_eaten : ℝ := candy_per_day * days_lasted
  candy_from_neighbors = total_candy_eaten - candy_from_sister

theorem solve_candy_problem :
  ∃ (candy_from_neighbors : ℝ), candy_problem candy_from_neighbors ∧ candy_from_neighbors = 11.0 := by
  sorry

end NUMINAMATH_CALUDE_solve_candy_problem_l4121_412179


namespace NUMINAMATH_CALUDE_total_birds_in_marsh_l4121_412104

def geese : ℕ := 58
def ducks : ℕ := 37
def herons : ℕ := 23
def kingfishers : ℕ := 46
def swans : ℕ := 15

theorem total_birds_in_marsh : geese + ducks + herons + kingfishers + swans = 179 := by
  sorry

end NUMINAMATH_CALUDE_total_birds_in_marsh_l4121_412104


namespace NUMINAMATH_CALUDE_geometric_series_sum_proof_l4121_412195

/-- The sum of the infinite geometric series 1/4 + 1/12 + 1/36 + 1/108 + ... -/
def geometric_series_sum : ℚ := 3/8

/-- The first term of the geometric series -/
def a : ℚ := 1/4

/-- The common ratio of the geometric series -/
def r : ℚ := 1/3

/-- Theorem stating that the sum of the infinite geometric series
    1/4 + 1/12 + 1/36 + 1/108 + ... is equal to 3/8 -/
theorem geometric_series_sum_proof :
  geometric_series_sum = a / (1 - r) :=
sorry

end NUMINAMATH_CALUDE_geometric_series_sum_proof_l4121_412195


namespace NUMINAMATH_CALUDE_tank_capacity_is_640_l4121_412131

/-- The capacity of the tank in liters -/
def tank_capacity : ℝ := 640

/-- The time in hours it takes to empty the tank with only the outlet pipe open -/
def outlet_time : ℝ := 10

/-- The rate at which the inlet pipe adds water in liters per minute -/
def inlet_rate : ℝ := 4

/-- The time in hours it takes to empty the tank with both pipes open -/
def both_pipes_time : ℝ := 16

/-- Theorem stating that the tank capacity is 640 liters given the conditions -/
theorem tank_capacity_is_640 :
  tank_capacity = outlet_time * (inlet_rate * 60) * both_pipes_time / (both_pipes_time - outlet_time) :=
by sorry

end NUMINAMATH_CALUDE_tank_capacity_is_640_l4121_412131


namespace NUMINAMATH_CALUDE_tourism_max_value_l4121_412181

noncomputable def f (x : ℝ) : ℝ := (51/50) * x - 0.01 * x^2 - Real.log x + Real.log 10

theorem tourism_max_value (x : ℝ) (h1 : 6 < x) (h2 : x ≤ 12) :
  ∃ (y : ℝ), y = f 12 ∧ ∀ z ∈ Set.Ioo 6 12, f z ≤ y := by
  sorry

end NUMINAMATH_CALUDE_tourism_max_value_l4121_412181


namespace NUMINAMATH_CALUDE_range_of_a_l4121_412107

-- Define the quadratic function
def f (a : ℝ) (x : ℝ) : ℝ := x^2 + a*x + 4

-- Define the property that the solution set is not empty
def has_solution (a : ℝ) : Prop := ∃ x : ℝ, f a x < 0

-- Theorem statement
theorem range_of_a (a : ℝ) : has_solution a ↔ a < -4 ∨ a > 4 := by sorry

end NUMINAMATH_CALUDE_range_of_a_l4121_412107


namespace NUMINAMATH_CALUDE_power_of_three_mod_eight_l4121_412127

theorem power_of_three_mod_eight : 3^2010 % 8 = 1 := by
  sorry

end NUMINAMATH_CALUDE_power_of_three_mod_eight_l4121_412127


namespace NUMINAMATH_CALUDE_end_on_multiple_of_four_probability_l4121_412149

def num_cards : ℕ := 12
def move_right_prob : ℚ := 1/2
def move_left_prob : ℚ := 1/4
def move_two_right_prob : ℚ := 1/4

def is_multiple_of_four (n : ℕ) : Prop := ∃ k, n = 4 * k

theorem end_on_multiple_of_four_probability :
  let total_outcomes := num_cards * 4 * 4  -- 12 cards * 4 spinner outcomes * 4 spinner outcomes
  let favorable_outcomes := 21  -- This is derived from the problem constraints
  (favorable_outcomes : ℚ) / total_outcomes = 21 / 192 := by sorry

end NUMINAMATH_CALUDE_end_on_multiple_of_four_probability_l4121_412149


namespace NUMINAMATH_CALUDE_min_at_five_l4121_412112

/-- The function to be minimized -/
def f (c : ℝ) : ℝ := (c - 3)^2 + (c - 4)^2 + (c - 8)^2

/-- The theorem stating that 5 minimizes the function f -/
theorem min_at_five : 
  ∀ x : ℝ, f 5 ≤ f x :=
sorry

end NUMINAMATH_CALUDE_min_at_five_l4121_412112


namespace NUMINAMATH_CALUDE_quadratic_roots_difference_l4121_412140

-- Define the quadratic equation
def quadratic (x p : ℝ) : ℝ := 2 * x^2 + p * x + 4

-- Define the condition that the roots differ by 2
def roots_differ_by_two (p : ℤ) : Prop :=
  ∃ (x y : ℝ), x ≠ y ∧ quadratic x p = 0 ∧ quadratic y p = 0 ∧ |x - y| = 2

-- The theorem to prove
theorem quadratic_roots_difference (p : ℤ) :
  roots_differ_by_two p → p = 7 ∨ p = -7 := by
  sorry


end NUMINAMATH_CALUDE_quadratic_roots_difference_l4121_412140


namespace NUMINAMATH_CALUDE_opposite_abs_equal_l4121_412148

theorem opposite_abs_equal (x : ℝ) : |x| = |-x| := by sorry

end NUMINAMATH_CALUDE_opposite_abs_equal_l4121_412148


namespace NUMINAMATH_CALUDE_angle_adjustment_l4121_412155

def are_complementary (a b : ℝ) : Prop := a + b = 90

theorem angle_adjustment (x y : ℝ) 
  (h1 : are_complementary x y)
  (h2 : x / y = 1 / 2)
  (h3 : x < y) :
  let new_x := x * 1.2
  let new_y := 90 - new_x
  (y - new_y) / y = 0.1 := by sorry

end NUMINAMATH_CALUDE_angle_adjustment_l4121_412155


namespace NUMINAMATH_CALUDE_stream_rate_calculation_l4121_412191

/-- Proves that given a boat with speed 16 km/hr in still water, traveling 126 km downstream in 6 hours, the rate of the stream is 5 km/hr. -/
theorem stream_rate_calculation (boat_speed : ℝ) (distance : ℝ) (time : ℝ) (stream_rate : ℝ) :
  boat_speed = 16 →
  distance = 126 →
  time = 6 →
  distance = (boat_speed + stream_rate) * time →
  stream_rate = 5 := by
sorry

end NUMINAMATH_CALUDE_stream_rate_calculation_l4121_412191


namespace NUMINAMATH_CALUDE_calculate_expression_l4121_412187

theorem calculate_expression : (8 * 2.25 - 5 * 0.85 / 2.5) = 16.3 := by
  sorry

end NUMINAMATH_CALUDE_calculate_expression_l4121_412187


namespace NUMINAMATH_CALUDE_colored_rectangle_iff_same_parity_l4121_412186

/-- Represents the four colors used to color the squares -/
inductive Color
  | Red
  | Yellow
  | Blue
  | Green

/-- Represents a unit square with colored sides -/
structure ColoredSquare where
  top : Color
  right : Color
  bottom : Color
  left : Color
  different_colors : top ≠ right ∧ top ≠ bottom ∧ top ≠ left ∧ 
                     right ≠ bottom ∧ right ≠ left ∧ 
                     bottom ≠ left

/-- Represents a rectangle formed by colored squares -/
structure ColoredRectangle where
  width : ℕ
  height : ℕ
  top_color : Color
  right_color : Color
  bottom_color : Color
  left_color : Color
  different_colors : top_color ≠ right_color ∧ top_color ≠ bottom_color ∧ top_color ≠ left_color ∧ 
                     right_color ≠ bottom_color ∧ right_color ≠ left_color ∧ 
                     bottom_color ≠ left_color

/-- Theorem stating that a colored rectangle can be formed if and only if its side lengths have the same parity -/
theorem colored_rectangle_iff_same_parity (r : ColoredRectangle) :
  (∃ (squares : List (List ColoredSquare)), 
    squares.length = r.height ∧ 
    (∀ row ∈ squares, row.length = r.width) ∧ 
    -- Additional conditions for correct arrangement of squares
    sorry
  ) ↔ 
  (r.width % 2 = r.height % 2) :=
sorry

end NUMINAMATH_CALUDE_colored_rectangle_iff_same_parity_l4121_412186


namespace NUMINAMATH_CALUDE_triangle_side_length_l4121_412126

theorem triangle_side_length (A B C : ℝ) (a b c : ℝ) :
  0 < A ∧ A < π / 2 →
  C = 4 * A →
  a = 21 →
  c = 54 →
  ∃ (x : ℝ), 0 < x ∧ x < 1 ∧ 8 * x^2 - 12 * x - 4.5714 = 0 ∧
    b = 21 * (16 * x^2 - 20 * x + 5) :=
by sorry

end NUMINAMATH_CALUDE_triangle_side_length_l4121_412126


namespace NUMINAMATH_CALUDE_flower_calculation_l4121_412109

/- Define the initial quantities -/
def initial_roses : ℕ := 36
def initial_chocolates : ℕ := 5
def initial_cupcakes : ℕ := 10
def initial_sunflowers : ℕ := 24

/- Define the trading events -/
def trade_day5 : ℕ × ℕ := (12, 6)  -- (roses, sunflowers)
def trade_day6 : ℕ × ℕ := (12, 20)  -- (roses, cupcakes)
def trade_day7 : ℕ := 15  -- daffodils

/- Define the wilting rates -/
def wilt_rate_day5 : ℚ := 1/10
def wilt_rate_day6_roses : ℚ := 1/5
def wilt_rate_day6_sunflowers : ℚ := 3/10
def wilt_rate_day7_roses : ℚ := 1/4
def wilt_rate_day7_sunflowers : ℚ := 3/20
def wilt_rate_day7_daffodils : ℚ := 1/5

/- Define the function to calculate the number of unwilted flowers -/
def calculate_unwilted_flowers (initial_roses initial_sunflowers : ℕ) 
  (trade_day5 trade_day6 : ℕ × ℕ) (trade_day7 : ℕ)
  (wilt_rate_day5 wilt_rate_day6_roses wilt_rate_day6_sunflowers 
   wilt_rate_day7_roses wilt_rate_day7_sunflowers wilt_rate_day7_daffodils : ℚ) :
  ℕ × ℕ × ℕ := sorry

/- Theorem statement -/
theorem flower_calculation :
  calculate_unwilted_flowers initial_roses initial_sunflowers
    trade_day5 trade_day6 trade_day7
    wilt_rate_day5 wilt_rate_day6_roses wilt_rate_day6_sunflowers
    wilt_rate_day7_roses wilt_rate_day7_sunflowers wilt_rate_day7_daffodils
  = (34, 18, 12) := by sorry

end NUMINAMATH_CALUDE_flower_calculation_l4121_412109


namespace NUMINAMATH_CALUDE_successive_discounts_equivalence_l4121_412193

theorem successive_discounts_equivalence (original_price : ℝ) 
  (first_discount second_discount : ℝ) (equivalent_discount : ℝ) : 
  original_price = 50 ∧ 
  first_discount = 0.15 ∧ 
  second_discount = 0.10 ∧ 
  equivalent_discount = 0.235 →
  original_price * (1 - first_discount) * (1 - second_discount) = 
  original_price * (1 - equivalent_discount) := by
  sorry

end NUMINAMATH_CALUDE_successive_discounts_equivalence_l4121_412193


namespace NUMINAMATH_CALUDE_circle_intersection_theorem_l4121_412114

-- Define the circle C
def circle_C (x y : ℝ) : Prop := (x - 3)^2 + (y - 1)^2 = 9

-- Define the line l
def line_l (x y m : ℝ) : Prop := x - y + m = 0

-- Define the perpendicularity condition
def perpendicular (xa ya xb yb xc yc : ℝ) : Prop :=
  (xa - xc) * (xb - xc) + (ya - yc) * (yb - yc) = 0

-- State the theorem
theorem circle_intersection_theorem (m : ℝ) :
  (∃ (xa ya xb yb : ℝ),
    circle_C xa ya ∧ circle_C xb yb ∧
    line_l xa ya m ∧ line_l xb yb m ∧
    perpendicular xa ya xb yb 3 1) →
  m = 1 ∨ m = -5 :=
by sorry

end NUMINAMATH_CALUDE_circle_intersection_theorem_l4121_412114


namespace NUMINAMATH_CALUDE_equality_of_fractions_l4121_412162

theorem equality_of_fractions (x y z k : ℝ) 
  (h : (5 : ℝ) / (x + y) = k / (x - z) ∧ k / (x - z) = (9 : ℝ) / (z + y)) : 
  k = 14 := by
  sorry

end NUMINAMATH_CALUDE_equality_of_fractions_l4121_412162


namespace NUMINAMATH_CALUDE_ellipse_line_slope_product_l4121_412184

/-- An ellipse with given properties -/
structure Ellipse where
  a : ℝ
  b : ℝ
  h_ab : a > b
  h_b_pos : b > 0
  h_ecc : (a^2 - b^2) / a^2 = 1/2
  h_point : 4/a^2 + 2/b^2 = 1

/-- A line not passing through origin and not parallel to axes -/
structure Line where
  k : ℝ
  b : ℝ
  h_k_nonzero : k ≠ 0
  h_b_nonzero : b ≠ 0

/-- The theorem statement -/
theorem ellipse_line_slope_product (C : Ellipse) (l : Line) : 
  ∃ (A B M : ℝ × ℝ), 
    (A.1^2 / C.a^2 + A.2^2 / C.b^2 = 1) ∧ 
    (B.1^2 / C.a^2 + B.2^2 / C.b^2 = 1) ∧
    (A.2 = l.k * A.1 + l.b) ∧ 
    (B.2 = l.k * B.1 + l.b) ∧
    (M = ((A.1 + B.1)/2, (A.2 + B.2)/2)) →
    (M.2 / M.1) * l.k = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_ellipse_line_slope_product_l4121_412184


namespace NUMINAMATH_CALUDE_tangent_line_at_one_two_l4121_412106

-- Define the function f
def f (x : ℝ) : ℝ := x^3 - x^2 + x + 1

-- State the theorem
theorem tangent_line_at_one_two :
  let f' : ℝ → ℝ := λ x ↦ 3*x^2 - 2*x + 1
  (∀ x, HasDerivAt f (f' x) x) →
  f 1 = 2 →
  (λ x ↦ 2*x) = λ x ↦ f 1 + f' 1 * (x - 1) :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_at_one_two_l4121_412106


namespace NUMINAMATH_CALUDE_angle_D_measure_l4121_412156

-- Define the hexagon and its angles
structure ConvexHexagon where
  A : ℝ
  B : ℝ
  C : ℝ
  D : ℝ
  E : ℝ
  F : ℝ

-- Define the properties of the hexagon
def is_valid_hexagon (h : ConvexHexagon) : Prop :=
  h.A > 0 ∧ h.B > 0 ∧ h.C > 0 ∧ h.D > 0 ∧ h.E > 0 ∧ h.F > 0 ∧
  h.A + h.B + h.C + h.D + h.E + h.F = 720

-- Define the conditions of the problem
def satisfies_conditions (h : ConvexHexagon) : Prop :=
  h.A = h.B ∧ h.B = h.C ∧
  h.D = h.E ∧ h.E = h.F ∧
  h.A + 30 = h.D

-- Theorem statement
theorem angle_D_measure (h : ConvexHexagon) 
  (h_valid : is_valid_hexagon h) 
  (h_cond : satisfies_conditions h) : 
  h.D = 135 :=
sorry

end NUMINAMATH_CALUDE_angle_D_measure_l4121_412156


namespace NUMINAMATH_CALUDE_rectangle_area_ratio_l4121_412163

/-- Given two rectangles A and B with sides (a, b) and (c, d) respectively,
    if a / c = b / d = 3 / 5, then the ratio of their areas is 9:25 -/
theorem rectangle_area_ratio (a b c d : ℝ) (h1 : a / c = 3 / 5) (h2 : b / d = 3 / 5) :
  (a * b) / (c * d) = 9 / 25 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_ratio_l4121_412163


namespace NUMINAMATH_CALUDE_representatives_count_l4121_412108

/-- The number of ways to select representatives from male and female students -/
def selectRepresentatives (numMale numFemale numReps : ℕ) (minMale minFemale : ℕ) : ℕ :=
  (numMale.choose (numReps - minFemale) * numFemale.choose minFemale) +
  (numMale.choose minMale * numFemale.choose (numReps - minMale))

/-- Theorem stating the number of ways to select representatives -/
theorem representatives_count :
  selectRepresentatives 5 4 4 2 1 = 100 := by
  sorry

#eval selectRepresentatives 5 4 4 2 1

end NUMINAMATH_CALUDE_representatives_count_l4121_412108


namespace NUMINAMATH_CALUDE_delores_initial_money_l4121_412118

def computer_cost : ℕ := 400
def printer_cost : ℕ := 40
def money_left : ℕ := 10

theorem delores_initial_money : 
  computer_cost + printer_cost + money_left = 450 := by sorry

end NUMINAMATH_CALUDE_delores_initial_money_l4121_412118


namespace NUMINAMATH_CALUDE_polynomial_equality_l4121_412177

theorem polynomial_equality (x : ℝ) : 
  let g : ℝ → ℝ := λ x => -2*x^5 + 4*x^4 - 12*x^3 + 2*x^2 + 4*x + 4
  2*x^5 + 3*x^3 - 4*x + 1 + g x = 4*x^4 - 9*x^3 + 2*x^2 + 5 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_equality_l4121_412177


namespace NUMINAMATH_CALUDE_line_through_points_l4121_412123

/-- Proves that for a line passing through (-3,1) and (1,3), m + b = 3 --/
theorem line_through_points (m b : ℚ) : 
  (1 = m * (-3) + b) ∧ (3 = m * 1 + b) → m + b = 3 := by
  sorry

end NUMINAMATH_CALUDE_line_through_points_l4121_412123


namespace NUMINAMATH_CALUDE_x_power_n_plus_reciprocal_l4121_412136

theorem x_power_n_plus_reciprocal (θ : ℝ) (x : ℂ) (n : ℕ) (h1 : 0 < θ) (h2 : θ < π) (h3 : x + 1/x = 2 * Real.cos θ) :
  x^n + 1/x^n = 2 * Real.cos (n * θ) := by
  sorry

end NUMINAMATH_CALUDE_x_power_n_plus_reciprocal_l4121_412136


namespace NUMINAMATH_CALUDE_bobby_blocks_l4121_412170

theorem bobby_blocks (initial_blocks : ℕ) (given_blocks : ℕ) : 
  initial_blocks = 2 → given_blocks = 6 → initial_blocks + given_blocks = 8 :=
by sorry

end NUMINAMATH_CALUDE_bobby_blocks_l4121_412170


namespace NUMINAMATH_CALUDE_sara_movie_day_total_expense_l4121_412183

def movie_day_expenses (ticket_price : ℚ) (num_tickets : ℕ) (rented_movie : ℚ) (snacks : ℚ) (parking : ℚ) (movie_poster : ℚ) (bought_movie : ℚ) : ℚ :=
  ticket_price * num_tickets + rented_movie + snacks + parking + movie_poster + bought_movie

theorem sara_movie_day_total_expense :
  movie_day_expenses 10.62 2 1.59 8.75 5.50 12.50 13.95 = 63.53 := by
  sorry

end NUMINAMATH_CALUDE_sara_movie_day_total_expense_l4121_412183


namespace NUMINAMATH_CALUDE_constant_zero_function_l4121_412178

theorem constant_zero_function (f : ℝ → ℝ) 
  (h : ∀ x y : ℝ, (f (x + y))^2 = (f x)^2 + (f y)^2) : 
  ∀ x : ℝ, f x = 0 := by
sorry

end NUMINAMATH_CALUDE_constant_zero_function_l4121_412178


namespace NUMINAMATH_CALUDE_base_notes_on_hour_l4121_412122

/-- Represents the number of notes rung at each quarter-hour mark --/
def quarter_hour_notes : Fin 3 → ℕ
| 0 => 2  -- quarter past
| 1 => 4  -- half past
| 2 => 6  -- three-quarters past

/-- The total number of notes rung from 1:00 p.m. to 5:00 p.m. --/
def total_notes : ℕ := 103

/-- The number of hours from 1:00 p.m. to 5:00 p.m. --/
def hours : ℕ := 5

/-- Calculates the total notes rung at quarter-hour marks between two consecutive hours --/
def notes_between_hours : ℕ := (Finset.sum Finset.univ quarter_hour_notes)

/-- Theorem stating that the number of base notes rung on the hour is 8 --/
theorem base_notes_on_hour : 
  ∃ (B : ℕ), 
    hours * B + (Finset.sum (Finset.range (hours + 1)) id) + 
    (hours - 1) * notes_between_hours = total_notes ∧ B = 8 := by
  sorry

end NUMINAMATH_CALUDE_base_notes_on_hour_l4121_412122


namespace NUMINAMATH_CALUDE_melissa_points_per_game_l4121_412111

theorem melissa_points_per_game 
  (total_points : ℕ) 
  (num_games : ℕ) 
  (points_per_game : ℕ) 
  (h1 : total_points = 21) 
  (h2 : num_games = 3) 
  (h3 : total_points = num_games * points_per_game) : 
  points_per_game = 7 := by
  sorry

end NUMINAMATH_CALUDE_melissa_points_per_game_l4121_412111


namespace NUMINAMATH_CALUDE_sixtieth_pair_is_five_six_l4121_412192

/-- Represents a pair of integers in the sequence -/
structure Pair :=
  (first : ℕ)
  (second : ℕ)

/-- Returns the sum of elements in a pair -/
def pairSum (p : Pair) : ℕ := p.first + p.second

/-- Returns the number of pairs in the first n levels -/
def pairsInFirstNLevels (n : ℕ) : ℕ := n * (n + 1) / 2

/-- Returns the nth pair in the sequence -/
def nthPair (n : ℕ) : Pair :=
  sorry -- Implementation details omitted

/-- The main theorem to prove -/
theorem sixtieth_pair_is_five_six :
  nthPair 60 = Pair.mk 5 6 := by
  sorry

end NUMINAMATH_CALUDE_sixtieth_pair_is_five_six_l4121_412192


namespace NUMINAMATH_CALUDE_linear_equation_solution_l4121_412159

theorem linear_equation_solution (a : ℝ) : 
  (a * 1 + (-2) = 1) → a = 3 := by
  sorry

end NUMINAMATH_CALUDE_linear_equation_solution_l4121_412159


namespace NUMINAMATH_CALUDE_highest_probability_l4121_412101

-- Define the sample space
variable (Ω : Type)

-- Define the probability measure
variable (P : Set Ω → ℝ)

-- Define events A, B, and C
variable (A B C : Set Ω)

-- State the theorem
theorem highest_probability :
  C ⊆ B → B ⊆ A → P A ≥ P B ∧ P A ≥ P C := by
  sorry

end NUMINAMATH_CALUDE_highest_probability_l4121_412101


namespace NUMINAMATH_CALUDE_weight_order_l4121_412151

theorem weight_order (P Q R S T : ℝ) 
  (h1 : P < 1000) (h2 : Q < 1000) (h3 : R < 1000) (h4 : S < 1000) (h5 : T < 1000)
  (h6 : Q + S = 1200) (h7 : R + T = 2100) (h8 : Q + T = 800) (h9 : Q + R = 900) (h10 : P + T = 700) :
  S > R ∧ R > T ∧ T > Q ∧ Q > P :=
by sorry

end NUMINAMATH_CALUDE_weight_order_l4121_412151


namespace NUMINAMATH_CALUDE_monochromatic_right_triangle_exists_l4121_412185

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a triangle defined by three points -/
structure Triangle where
  a : Point
  b : Point
  c : Point

/-- Represents a color (either 0 or 1) -/
inductive Color where
  | zero : Color
  | one : Color

/-- Checks if a triangle is equilateral -/
def isEquilateral (t : Triangle) : Prop := sorry

/-- Checks if a triangle is right-angled -/
def isRightAngled (t : Triangle) : Prop := sorry

/-- Checks if a point is on the side of a triangle -/
def isOnSide (p : Point) (t : Triangle) : Prop := sorry

/-- Represents a coloring of points on the sides of a triangle -/
def Coloring (t : Triangle) := Point → Color

/-- The main theorem to be proved -/
theorem monochromatic_right_triangle_exists 
  (t : Triangle) 
  (h_equilateral : isEquilateral t) 
  (coloring : Coloring t) : 
  ∃ (p q r : Point), 
    isOnSide p t ∧ isOnSide q t ∧ isOnSide r t ∧
    isRightAngled ⟨p, q, r⟩ ∧
    coloring p = coloring q ∧ coloring q = coloring r :=
sorry

end NUMINAMATH_CALUDE_monochromatic_right_triangle_exists_l4121_412185


namespace NUMINAMATH_CALUDE_fraction_subtraction_simplification_l4121_412147

theorem fraction_subtraction_simplification :
  8 / 21 - 10 / 63 = 2 / 9 := by
  sorry

end NUMINAMATH_CALUDE_fraction_subtraction_simplification_l4121_412147


namespace NUMINAMATH_CALUDE_middle_number_of_consecutive_integers_l4121_412160

theorem middle_number_of_consecutive_integers (n : ℤ) : 
  (n - 2) + (n - 1) + n + (n + 1) + (n + 2) = 10^2018 → n = 2 * 10^2017 := by
  sorry

end NUMINAMATH_CALUDE_middle_number_of_consecutive_integers_l4121_412160


namespace NUMINAMATH_CALUDE_rectangle_hyperbola_eccentricity_l4121_412125

/-- Rectangle with sides of length 4 and 3 -/
structure Rectangle :=
  (length : ℝ)
  (width : ℝ)
  (length_pos : length > 0)
  (width_pos : width > 0)
  (length_gt_width : length > width)

/-- Hyperbola passing through the vertices of the rectangle -/
structure Hyperbola (rect : Rectangle) :=
  (passes_through_vertices : Bool)

/-- Eccentricity of a hyperbola -/
def eccentricity (h : Hyperbola rect) : ℝ := sorry

theorem rectangle_hyperbola_eccentricity (rect : Rectangle) 
  (h : Hyperbola rect) (h_passes : h.passes_through_vertices = true) :
  rect.length = 4 → rect.width = 3 → eccentricity h = 2 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_hyperbola_eccentricity_l4121_412125


namespace NUMINAMATH_CALUDE_initial_deposit_proof_l4121_412176

def bank_account (initial_deposit : ℝ) : ℝ := 
  ((initial_deposit * 1.1 + 10) * 1.1 + 10)

theorem initial_deposit_proof (initial_deposit : ℝ) : 
  bank_account initial_deposit = 142 → initial_deposit = 100 := by
sorry

end NUMINAMATH_CALUDE_initial_deposit_proof_l4121_412176


namespace NUMINAMATH_CALUDE_bamboo_volume_proof_l4121_412110

theorem bamboo_volume_proof (a : ℕ → ℚ) :
  (∀ i : ℕ, i < 8 → a (i + 1) - a i = a (i + 2) - a (i + 1)) →  -- arithmetic progression
  a 1 + a 2 + a 3 + a 4 = 3 →                                   -- sum of first 4 terms
  a 7 + a 8 + a 9 = 4 →                                         -- sum of last 3 terms
  a 5 + a 6 = 31/9 := by
sorry

end NUMINAMATH_CALUDE_bamboo_volume_proof_l4121_412110


namespace NUMINAMATH_CALUDE_trailing_zeros_factorial_product_l4121_412175

/-- The number of trailing zeros in n! -/
def trailingZeros (n : ℕ) : ℕ := sorry

/-- The sum of trailing zeros in factorials from 1! to n! -/
def sumTrailingZeros (n : ℕ) : ℕ := sorry

/-- The theorem stating that the number of trailing zeros in the product of factorials 
    from 1! to 50!, when divided by 100, yields a remainder of 14 -/
theorem trailing_zeros_factorial_product : 
  (sumTrailingZeros 50) % 100 = 14 := by sorry

end NUMINAMATH_CALUDE_trailing_zeros_factorial_product_l4121_412175


namespace NUMINAMATH_CALUDE_polyhedron_edge_intersection_l4121_412174

/-- A polyhedron with a given number of edges -/
structure Polyhedron where
  edges : ℕ

/-- A plane that can intersect edges of a polyhedron -/
structure IntersectingPlane where
  intersectedEdges : ℕ

/-- Represents a convex polyhedron -/
def ConvexPolyhedron (p : Polyhedron) : Prop := sorry

/-- Represents a non-convex polyhedron -/
def NonConvexPolyhedron (p : Polyhedron) : Prop := sorry

/-- The maximum number of edges that can be intersected by a plane in a convex polyhedron -/
def maxIntersectedEdgesConvex (p : Polyhedron) (plane : IntersectingPlane) : Prop :=
  ConvexPolyhedron p ∧ plane.intersectedEdges ≤ 68

/-- The number of edges that can be intersected by a plane in a non-convex polyhedron -/
def intersectedEdgesNonConvex (p : Polyhedron) (plane : IntersectingPlane) : Prop :=
  NonConvexPolyhedron p ∧ plane.intersectedEdges = 96

/-- The impossibility of intersecting all edges in any polyhedron -/
def cannotIntersectAllEdges (p : Polyhedron) (plane : IntersectingPlane) : Prop :=
  plane.intersectedEdges < p.edges

theorem polyhedron_edge_intersection (p : Polyhedron) (plane : IntersectingPlane) 
    (h : p.edges = 100) : 
    (maxIntersectedEdgesConvex p plane) ∧ 
    (∃ (p' : Polyhedron) (plane' : IntersectingPlane), intersectedEdgesNonConvex p' plane') ∧ 
    (cannotIntersectAllEdges p plane) := by
  sorry

end NUMINAMATH_CALUDE_polyhedron_edge_intersection_l4121_412174


namespace NUMINAMATH_CALUDE_distance_not_proportional_to_time_l4121_412119

/-- Uniform motion equation -/
def uniform_motion (a v t : ℝ) : ℝ := a + v * t

/-- Proportionality definition -/
def proportional (f : ℝ → ℝ) : Prop := ∀ (k t : ℝ), f (k * t) = k * f t

/-- Theorem: In uniform motion, distance is not generally proportional to time -/
theorem distance_not_proportional_to_time (a v : ℝ) (h : a ≠ 0) :
  ¬ proportional (uniform_motion a v) := by
  sorry

end NUMINAMATH_CALUDE_distance_not_proportional_to_time_l4121_412119


namespace NUMINAMATH_CALUDE_probability_two_heads_in_four_flips_l4121_412103

def coin_flip_probability (n k : ℕ) : ℚ :=
  (Nat.choose n k : ℚ) / (2 ^ n : ℚ)

theorem probability_two_heads_in_four_flips :
  coin_flip_probability 4 2 = 3/8 := by
  sorry

end NUMINAMATH_CALUDE_probability_two_heads_in_four_flips_l4121_412103


namespace NUMINAMATH_CALUDE_fifteenth_term_of_sequence_l4121_412158

def arithmetic_sequence (a₁ : ℝ) (d : ℝ) (n : ℕ) : ℝ := a₁ + (n - 1 : ℝ) * d

theorem fifteenth_term_of_sequence : arithmetic_sequence 3 4 15 = 59 := by
  sorry

end NUMINAMATH_CALUDE_fifteenth_term_of_sequence_l4121_412158


namespace NUMINAMATH_CALUDE_deepak_age_l4121_412165

theorem deepak_age (arun_age deepak_age : ℕ) : 
  (arun_age : ℚ) / deepak_age = 5 / 7 →
  arun_age + 6 = 36 →
  deepak_age = 42 := by
sorry

end NUMINAMATH_CALUDE_deepak_age_l4121_412165


namespace NUMINAMATH_CALUDE_cost_formula_l4121_412132

def cost (P : ℕ) : ℕ :=
  15 + 4 * (P - 1) - 10 * (if P > 5 then 1 else 0)

theorem cost_formula (P : ℕ) :
  cost P = 15 + 4 * (P - 1) - 10 * (if P > 5 then 1 else 0) :=
by sorry

end NUMINAMATH_CALUDE_cost_formula_l4121_412132


namespace NUMINAMATH_CALUDE_function_must_be_constant_l4121_412142

-- Define the function type
def FunctionType := ℤ × ℤ → ℝ

-- Define the property of the function
def SatisfiesEquation (f : FunctionType) : Prop :=
  ∀ x y : ℤ, f (x, y) = (f (x - 1, y) + f (x, y - 1)) / 2

-- Define the range constraint
def InRange (f : FunctionType) : Prop :=
  ∀ x y : ℤ, 0 ≤ f (x, y) ∧ f (x, y) ≤ 1

-- Main theorem statement
theorem function_must_be_constant (f : FunctionType) 
  (h_eq : SatisfiesEquation f) (h_range : InRange f) : 
  ∃ c : ℝ, ∀ x y : ℤ, f (x, y) = c ∧ 0 ≤ c ∧ c ≤ 1 :=
sorry

end NUMINAMATH_CALUDE_function_must_be_constant_l4121_412142


namespace NUMINAMATH_CALUDE_ln_inequality_l4121_412153

theorem ln_inequality (x : ℝ) (h : x > 0) : Real.log x ≥ x - 1 := by
  sorry

end NUMINAMATH_CALUDE_ln_inequality_l4121_412153


namespace NUMINAMATH_CALUDE_grasshopper_jump_distance_l4121_412105

/-- The jumping distances of animals in a contest -/
structure JumpContest where
  frog : ℕ
  grasshopper : ℕ
  grasshopper_frog_diff : grasshopper = frog + 4

/-- Theorem: In a jump contest where the frog jumped 15 inches and the grasshopper
    jumped 4 inches farther than the frog, the grasshopper's jump distance is 19 inches. -/
theorem grasshopper_jump_distance (contest : JumpContest) 
  (h : contest.frog = 15) : contest.grasshopper = 19 := by
  sorry

end NUMINAMATH_CALUDE_grasshopper_jump_distance_l4121_412105


namespace NUMINAMATH_CALUDE_rectangle_sides_and_solvability_l4121_412113

/-- Given a rectangle with perimeter k and area t, this theorem proves the lengths of its sides
    and the condition for solvability. -/
theorem rectangle_sides_and_solvability (k t : ℝ) (k_pos : k > 0) (t_pos : t > 0) :
  let a := (k + Real.sqrt (k^2 - 16*t)) / 4
  let b := (k - Real.sqrt (k^2 - 16*t)) / 4
  (k^2 ≥ 16*t) →
  (a + b = k/2 ∧ a * b = t ∧ a > 0 ∧ b > 0) :=
by sorry

end NUMINAMATH_CALUDE_rectangle_sides_and_solvability_l4121_412113


namespace NUMINAMATH_CALUDE_b_share_of_earnings_l4121_412100

theorem b_share_of_earnings 
  (a_days b_days c_days : ℕ) 
  (total_earnings : ℚ) 
  (ha : a_days = 6)
  (hb : b_days = 8)
  (hc : c_days = 12)
  (htotal : total_earnings = 2340) :
  (1 / b_days) / ((1 / a_days) + (1 / b_days) + (1 / c_days)) * total_earnings = 780 := by
  sorry

end NUMINAMATH_CALUDE_b_share_of_earnings_l4121_412100


namespace NUMINAMATH_CALUDE_arithmetic_sequence_angles_l4121_412146

/-- Given five angles in an arithmetic sequence with the smallest angle 25° and the largest 105°,
    the common difference is 20°. -/
theorem arithmetic_sequence_angles (a₁ a₂ a₃ a₄ a₅ : ℝ) : 
  a₁ < a₂ ∧ a₂ < a₃ ∧ a₃ < a₄ ∧ a₄ < a₅ →  -- ensuring the sequence is increasing
  a₁ = 25 →  -- smallest angle is 25°
  a₅ = 105 →  -- largest angle is 105°
  ∃ d : ℝ, d = 20 ∧  -- common difference exists and equals 20°
    a₂ = a₁ + d ∧ 
    a₃ = a₂ + d ∧ 
    a₄ = a₃ + d ∧ 
    a₅ = a₄ + d :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_angles_l4121_412146


namespace NUMINAMATH_CALUDE_opposite_of_negative_2023_l4121_412190

theorem opposite_of_negative_2023 : 
  ∀ x : ℤ, x + (-2023) = 0 → x = 2023 := by
  sorry

end NUMINAMATH_CALUDE_opposite_of_negative_2023_l4121_412190


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l4121_412173

def is_arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  is_arithmetic_sequence a → a 4 = 4 → a 2 + a 6 = 8 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l4121_412173


namespace NUMINAMATH_CALUDE_treasure_chest_contains_all_coins_l4121_412152

/-- Represents the scuba diving scenario with gold coins --/
structure ScubaDiving where
  hours : ℕ
  coins_per_hour : ℕ
  smaller_bags : ℕ

/-- Calculates the number of gold coins in the treasure chest --/
def treasure_chest_coins (dive : ScubaDiving) : ℕ :=
  dive.hours * dive.coins_per_hour

/-- Theorem stating that the treasure chest contains all the coins found --/
theorem treasure_chest_contains_all_coins (dive : ScubaDiving) 
  (h1 : dive.hours = 8)
  (h2 : dive.coins_per_hour = 25)
  (h3 : dive.smaller_bags = 2) :
  treasure_chest_coins dive = 200 :=
sorry

end NUMINAMATH_CALUDE_treasure_chest_contains_all_coins_l4121_412152


namespace NUMINAMATH_CALUDE_max_a_proof_l4121_412121

/-- The coefficient of x^4 in the expansion of (1 - 2x + ax^2)^8 -/
def coeff_x4 (a : ℝ) : ℝ := 28 * a^2 + 336 * a + 1120

/-- The maximum value of a that satisfies the condition -/
def max_a : ℝ := -5

theorem max_a_proof :
  (∀ a : ℝ, coeff_x4 a = -1540 → a ≤ max_a) ∧
  coeff_x4 max_a = -1540 := by sorry

end NUMINAMATH_CALUDE_max_a_proof_l4121_412121


namespace NUMINAMATH_CALUDE_max_value_sqrt_x_over_x_plus_one_l4121_412182

theorem max_value_sqrt_x_over_x_plus_one :
  (∃ x : ℝ, x ≥ 0 ∧ Real.sqrt x / (x + 1) = 1/2) ∧
  (∀ x : ℝ, x ≥ 0 → Real.sqrt x / (x + 1) ≤ 1/2) := by
  sorry

end NUMINAMATH_CALUDE_max_value_sqrt_x_over_x_plus_one_l4121_412182


namespace NUMINAMATH_CALUDE_range_of_a_l4121_412189

theorem range_of_a (a : ℝ) : 
  Real.sqrt (a^3 + 2*a^2) = -a * Real.sqrt (a + 2) → 
  -2 ≤ a ∧ a ≤ 0 := by sorry

end NUMINAMATH_CALUDE_range_of_a_l4121_412189


namespace NUMINAMATH_CALUDE_carrot_sticks_leftover_l4121_412134

theorem carrot_sticks_leftover (total_carrots : ℕ) (num_people : ℕ) (h1 : total_carrots = 74) (h2 : num_people = 12) :
  total_carrots % num_people = 2 := by
  sorry

end NUMINAMATH_CALUDE_carrot_sticks_leftover_l4121_412134


namespace NUMINAMATH_CALUDE_inequality_solution_range_l4121_412197

theorem inequality_solution_range (a : ℝ) : 
  (∃ x : ℝ, 1 < x ∧ x < 4 ∧ 2 * x^2 - 8 * x - 4 - a > 0) → a < -4 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_range_l4121_412197


namespace NUMINAMATH_CALUDE_set_operations_l4121_412166

def A : Set ℕ := {x | x > 0 ∧ x < 9}
def B : Set ℕ := {1, 2, 3}
def C : Set ℕ := {3, 4, 5, 6}

theorem set_operations :
  (A ∩ B = {1, 2, 3}) ∧
  (A ∩ C = {3, 4, 5, 6}) ∧
  (A ∩ (B ∪ C) = {1, 2, 3, 4, 5, 6}) ∧
  (A ∪ (B ∩ C) = {1, 2, 3, 4, 5, 6, 7, 8}) := by
sorry

end NUMINAMATH_CALUDE_set_operations_l4121_412166


namespace NUMINAMATH_CALUDE_ground_school_cost_proof_l4121_412129

/-- Represents the cost of a private pilot course -/
def total_cost : ℕ := 1275

/-- Represents the additional cost of the flight portion compared to the ground school portion -/
def flight_additional_cost : ℕ := 625

/-- Represents the cost of the flight portion -/
def flight_cost : ℕ := 950

/-- Represents the cost of the ground school portion -/
def ground_school_cost : ℕ := total_cost - flight_cost

theorem ground_school_cost_proof : ground_school_cost = 325 := by
  sorry

end NUMINAMATH_CALUDE_ground_school_cost_proof_l4121_412129


namespace NUMINAMATH_CALUDE_sum_of_squares_divisibility_l4121_412102

theorem sum_of_squares_divisibility (n : ℕ) : 
  (∃ k : ℕ, n = 6 * k - 1 ∨ n = 6 * k + 1) ↔ 
  (∃ m : ℕ, n * (n + 1) * (2 * n + 1) = 6 * m) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squares_divisibility_l4121_412102


namespace NUMINAMATH_CALUDE_function_value_at_cos_15_degrees_l4121_412128

theorem function_value_at_cos_15_degrees 
  (f : ℝ → ℝ) 
  (h : ∀ x, f (Real.sin x) = Real.cos (2 * x) - 1) :
  f (Real.cos (15 * π / 180)) = -Real.sqrt 3 / 2 - 1 := by
  sorry

end NUMINAMATH_CALUDE_function_value_at_cos_15_degrees_l4121_412128


namespace NUMINAMATH_CALUDE_problem_solution_l4121_412135

-- Define the function f
def f (x : ℝ) : ℝ := |x - 2|

-- Define the theorem
theorem problem_solution (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  -- Part 1: Solution set of f(x) + f(2x+1) ≥ 6
  {x : ℝ | f x + f (2*x + 1) ≥ 6} = Set.Iic (-1) ∪ Set.Ici 3 ∧
  -- Part 2: Range of m given the condition
  ∀ m : ℝ, (∀ x : ℝ, f (x - m) - f (-x) ≤ 4/a + 1/b) → -13 ≤ m ∧ m ≤ 5 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l4121_412135


namespace NUMINAMATH_CALUDE_min_value_y_l4121_412145

theorem min_value_y (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : y * Real.log y = Real.exp (2 * x) - y * Real.log (2 * x)) : 
  (∀ z, z > 0 → z * Real.log z = Real.exp (2 * x) - z * Real.log (2 * x) → y ≤ z) ∧ y = Real.exp 1 :=
sorry

end NUMINAMATH_CALUDE_min_value_y_l4121_412145


namespace NUMINAMATH_CALUDE_max_sum_abcd_l4121_412157

theorem max_sum_abcd (a c d : ℤ) (b : ℕ+) 
  (h1 : a + b = c) 
  (h2 : b + c = d) 
  (h3 : c + d = a) : 
  (∃ (a' c' d' : ℤ) (b' : ℕ+), 
    a' + b' = c' ∧ 
    b' + c' = d' ∧ 
    c' + d' = a' ∧ 
    a' + b' + c' + d' = -5) ∧ 
  (∀ (a' c' d' : ℤ) (b' : ℕ+), 
    a' + b' = c' → 
    b' + c' = d' → 
    c' + d' = a' → 
    a' + b' + c' + d' ≤ -5) :=
sorry

end NUMINAMATH_CALUDE_max_sum_abcd_l4121_412157


namespace NUMINAMATH_CALUDE_complex_number_imaginary_part_l4121_412194

theorem complex_number_imaginary_part (a : ℝ) :
  let z : ℂ := (1 - a * Complex.I) / (1 - Complex.I)
  Complex.im z = 4 → a = -7 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_imaginary_part_l4121_412194


namespace NUMINAMATH_CALUDE_count_good_numbers_formula_l4121_412138

/-- A number is considered "good" if it contains an even number (including zero) of the digit 8 -/
def is_good (x : ℕ) : Prop := sorry

/-- The count of "good numbers" with length not exceeding n -/
def count_good_numbers (n : ℕ) : ℕ := sorry

/-- The main theorem: The count of "good numbers" with length not exceeding n 
    is equal to (8^n + 10^n) / 2 - 1 -/
theorem count_good_numbers_formula (n : ℕ) (h : n > 0) : 
  count_good_numbers n = (8^n + 10^n) / 2 - 1 := by sorry

end NUMINAMATH_CALUDE_count_good_numbers_formula_l4121_412138


namespace NUMINAMATH_CALUDE_chess_team_arrangements_l4121_412161

/-- Represents the number of boys in the chess team -/
def num_boys : ℕ := 3

/-- Represents the number of girls in the chess team -/
def num_girls : ℕ := 3

/-- Represents the total number of students in the chess team -/
def total_students : ℕ := num_boys + num_girls

/-- Represents the number of ways to arrange the ends of the row -/
def end_arrangements : ℕ := 2 * num_boys * num_girls

/-- Represents the number of ways to arrange the middle of the row -/
def middle_arrangements : ℕ := 2 * 2

/-- The total number of possible arrangements -/
def total_arrangements : ℕ := end_arrangements * middle_arrangements

theorem chess_team_arrangements :
  total_arrangements = 72 :=
sorry

end NUMINAMATH_CALUDE_chess_team_arrangements_l4121_412161


namespace NUMINAMATH_CALUDE_sum_of_squared_residuals_l4121_412133

theorem sum_of_squared_residuals 
  (total_sum_squared_deviations : ℝ) 
  (correlation_coefficient : ℝ) 
  (h1 : total_sum_squared_deviations = 100) 
  (h2 : correlation_coefficient = 0.818) : 
  total_sum_squared_deviations * (1 - correlation_coefficient ^ 2) = 33.0876 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_squared_residuals_l4121_412133


namespace NUMINAMATH_CALUDE_train_speed_calculation_l4121_412124

/-- Proves that given the specified conditions, the speed of the first train is approximately 120.016 km/hr -/
theorem train_speed_calculation (length_train1 : ℝ) (length_train2 : ℝ) (speed_train2 : ℝ) (crossing_time : ℝ) :
  length_train1 = 250 →
  length_train2 = 250.04 →
  speed_train2 = 80 →
  crossing_time = 9 →
  ∃ (speed_train1 : ℝ), abs (speed_train1 - 120.016) < 0.001 :=
by
  sorry


end NUMINAMATH_CALUDE_train_speed_calculation_l4121_412124


namespace NUMINAMATH_CALUDE_total_hats_bought_l4121_412143

theorem total_hats_bought (blue_cost green_cost total_price green_count : ℕ) 
  (h1 : blue_cost = 6)
  (h2 : green_cost = 7)
  (h3 : total_price = 530)
  (h4 : green_count = 20) :
  ∃ (blue_count : ℕ), blue_count * blue_cost + green_count * green_cost = total_price ∧
                      blue_count + green_count = 85 := by
  sorry

end NUMINAMATH_CALUDE_total_hats_bought_l4121_412143


namespace NUMINAMATH_CALUDE_compare_sqrt_sums_l4121_412150

theorem compare_sqrt_sums (a : ℝ) (h : a > 0) :
  Real.sqrt a + Real.sqrt (a + 3) < Real.sqrt (a + 1) + Real.sqrt (a + 2) := by
sorry

end NUMINAMATH_CALUDE_compare_sqrt_sums_l4121_412150


namespace NUMINAMATH_CALUDE_race_time_calculation_l4121_412171

/-- Given that Prejean's speed is three-quarters of Rickey's speed and Rickey took 40 minutes to finish a race, 
    prove that the total time taken by both runners is 40 + 40 * (4/3) minutes. -/
theorem race_time_calculation (rickey_speed rickey_time prejean_speed : ℝ) 
    (h1 : rickey_time = 40)
    (h2 : prejean_speed = 3/4 * rickey_speed) : 
  rickey_time + (rickey_time / (prejean_speed / rickey_speed)) = 40 + 40 * (4/3) := by
  sorry

end NUMINAMATH_CALUDE_race_time_calculation_l4121_412171


namespace NUMINAMATH_CALUDE_goods_train_passing_time_l4121_412115

/-- The time taken for a goods train to pass a man in an opposing train -/
theorem goods_train_passing_time (man_speed goods_speed : ℝ) (goods_length : ℝ) : 
  man_speed = 70 →
  goods_speed = 42 →
  goods_length = 280 →
  ∃ t : ℝ, t > 0 ∧ t < 10 ∧ t * (man_speed + goods_speed) * (1000 / 3600) = goods_length :=
by sorry

end NUMINAMATH_CALUDE_goods_train_passing_time_l4121_412115


namespace NUMINAMATH_CALUDE_boat_speed_in_still_water_l4121_412198

/-- Represents the speed of a boat in still water and a stream -/
structure BoatAndStreamSpeeds where
  boat : ℝ
  stream : ℝ

/-- Represents the time taken to travel upstream and downstream -/
structure TravelTimes where
  downstream : ℝ
  upstream : ℝ

/-- The problem statement -/
theorem boat_speed_in_still_water 
  (speeds : BoatAndStreamSpeeds)
  (times : TravelTimes)
  (h1 : speeds.stream = 13)
  (h2 : times.upstream = 2 * times.downstream)
  (h3 : (speeds.boat + speeds.stream) * times.downstream = 
        (speeds.boat - speeds.stream) * times.upstream) :
  speeds.boat = 39 := by
sorry

end NUMINAMATH_CALUDE_boat_speed_in_still_water_l4121_412198


namespace NUMINAMATH_CALUDE_square_value_when_product_zero_l4121_412137

theorem square_value_when_product_zero (a : ℝ) :
  (a^2 - 3) * (a^2 + 1) = 0 → a^2 = 3 :=
by
  sorry

end NUMINAMATH_CALUDE_square_value_when_product_zero_l4121_412137


namespace NUMINAMATH_CALUDE_point_P_location_l4121_412120

-- Define the points on a line
structure Point :=
  (x : ℝ)

-- Define the distances
def OA (a : ℝ) : ℝ := a
def OB (b : ℝ) : ℝ := b
def OC (c : ℝ) : ℝ := c
def OE (e : ℝ) : ℝ := e

-- Define the condition for P being between B and C
def between (B C P : Point) : Prop :=
  B.x ≤ P.x ∧ P.x ≤ C.x

-- Define the ratio condition
def ratio_condition (A B C E P : Point) : Prop :=
  (A.x - P.x) * (P.x - C.x) = (B.x - P.x) * (P.x - E.x)

-- Theorem statement
theorem point_P_location 
  (O A B C E P : Point) 
  (a b c e : ℝ) 
  (h1 : O.x = 0) 
  (h2 : A.x = a) 
  (h3 : B.x = b) 
  (h4 : C.x = c) 
  (h5 : E.x = e) 
  (h6 : between B C P) 
  (h7 : ratio_condition A B C E P) : 
  P.x = (b * e - a * c) / (a - b + e - c) :=
sorry

end NUMINAMATH_CALUDE_point_P_location_l4121_412120


namespace NUMINAMATH_CALUDE_meat_spending_fraction_l4121_412168

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

end NUMINAMATH_CALUDE_meat_spending_fraction_l4121_412168


namespace NUMINAMATH_CALUDE_derivative_extrema_l4121_412139

-- Define the function
def f (x : ℝ) := x^4 - 6*x^2 + 1

-- Define the derivative of the function
def f' (x : ℝ) := 4*x^3 - 12*x

-- Define the interval
def interval : Set ℝ := {x | -1 ≤ x ∧ x ≤ 3}

-- Theorem statement
theorem derivative_extrema :
  (∃ x ∈ interval, ∀ y ∈ interval, f' y ≤ f' x) ∧
  (∃ x ∈ interval, ∀ y ∈ interval, f' y ≥ f' x) ∧
  (∃ x ∈ interval, f' x = 72) ∧
  (∃ x ∈ interval, f' x = -8) :=
sorry

end NUMINAMATH_CALUDE_derivative_extrema_l4121_412139


namespace NUMINAMATH_CALUDE_total_cards_l4121_412199

theorem total_cards (brenda janet mara : ℕ) : 
  janet = brenda + 9 →
  mara = 2 * janet →
  mara = 150 - 40 →
  brenda + janet + mara = 211 := by
  sorry

end NUMINAMATH_CALUDE_total_cards_l4121_412199


namespace NUMINAMATH_CALUDE_print_shop_charge_difference_l4121_412141

/-- The charge difference between two print shops for a given number of copies -/
def charge_difference (price_x price_y : ℚ) (num_copies : ℕ) : ℚ :=
  num_copies * (price_y - price_x)

/-- The price per copy at print shop X -/
def price_x : ℚ := 1.25

/-- The price per copy at print shop Y -/
def price_y : ℚ := 2.75

/-- The number of copies to be printed -/
def num_copies : ℕ := 60

theorem print_shop_charge_difference :
  charge_difference price_x price_y num_copies = 90 := by
  sorry

end NUMINAMATH_CALUDE_print_shop_charge_difference_l4121_412141


namespace NUMINAMATH_CALUDE_binomial_square_example_l4121_412196

theorem binomial_square_example : 15^2 + 2*(15*3) + 3^2 = 324 := by
  sorry

end NUMINAMATH_CALUDE_binomial_square_example_l4121_412196


namespace NUMINAMATH_CALUDE_p_plus_q_equals_21_over_2_l4121_412180

theorem p_plus_q_equals_21_over_2 (p q : ℝ) 
  (hp : p^3 - 18*p^2 + 27*p - 81 = 0)
  (hq : 9*q^3 - 81*q^2 - 243*q + 3645 = 0) : 
  p + q = 21/2 := by
  sorry

end NUMINAMATH_CALUDE_p_plus_q_equals_21_over_2_l4121_412180


namespace NUMINAMATH_CALUDE_even_quadratic_function_range_l4121_412130

/-- A quadratic function that is even -/
def EvenQuadraticFunction (f : ℝ → ℝ) : Prop :=
  ∃ a c : ℝ, ∀ x : ℝ, f x = a * x^2 + c

theorem even_quadratic_function_range
  (f : ℝ → ℝ)
  (hf : EvenQuadraticFunction f)
  (h1 : 1 ≤ f 1 ∧ f 1 ≤ 2)
  (h2 : 3 ≤ f 2 ∧ f 2 ≤ 4) :
  14/3 ≤ f 3 ∧ f 3 ≤ 9 := by
sorry

end NUMINAMATH_CALUDE_even_quadratic_function_range_l4121_412130


namespace NUMINAMATH_CALUDE_truck_toll_calculation_l4121_412172

/-- Calculates the total toll for a truck crossing a bridge --/
def calculate_total_toll (B A1 A2 X1 X2 w : ℚ) (is_peak_hour : Bool) : ℚ :=
  let T := B + A1 * (X1 - 2) + A2 * X2
  let F := if w > 10000 then 0.1 * (w - 10000) else 0
  let total_without_surcharge := T + F
  let S := if is_peak_hour then 0.02 * total_without_surcharge else 0
  total_without_surcharge + S

theorem truck_toll_calculation :
  let B : ℚ := 0.50
  let A1 : ℚ := 0.75
  let A2 : ℚ := 0.50
  let X1 : ℚ := 1  -- One axle with 2 wheels
  let X2 : ℚ := 4  -- Four axles with 4 wheels each
  let w : ℚ := 12000
  let is_peak_hour : Bool := true  -- 9 AM is during peak hours
  calculate_total_toll B A1 A2 X1 X2 w is_peak_hour = 205.79 := by
  sorry


end NUMINAMATH_CALUDE_truck_toll_calculation_l4121_412172


namespace NUMINAMATH_CALUDE_range_of_a_when_B_subset_A_l4121_412188

/-- The set A -/
def A : Set ℝ := {x | x^2 + 4*x = 0}

/-- The set B parameterized by a -/
def B (a : ℝ) : Set ℝ := {x | x^2 + 2*(a+1)*x + a^2 - 1 = 0}

/-- The range of a -/
def range_a : Set ℝ := {a | a = 1 ∨ a ≤ -1}

/-- Theorem stating the range of a when B is a subset of A -/
theorem range_of_a_when_B_subset_A :
  ∀ a : ℝ, B a ⊆ A → a ∈ range_a :=
sorry

end NUMINAMATH_CALUDE_range_of_a_when_B_subset_A_l4121_412188


namespace NUMINAMATH_CALUDE_product_of_roots_l4121_412169

theorem product_of_roots (x y : ℝ) (hx : x > 0) (hy : y > 0) 
  (h : Real.sqrt (12 * x) * Real.sqrt (20 * x) * Real.sqrt (4 * y) * Real.sqrt (25 * y) = 50) : 
  x * y = Real.sqrt (25 / 24) := by
sorry

end NUMINAMATH_CALUDE_product_of_roots_l4121_412169
