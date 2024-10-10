import Mathlib

namespace lisa_speed_equals_eugene_l2671_267113

def eugene_speed : ℚ := 5
def carlos_speed_ratio : ℚ := 3/4
def lisa_speed_ratio : ℚ := 4/3

theorem lisa_speed_equals_eugene (eugene_speed : ℚ) (carlos_speed_ratio : ℚ) (lisa_speed_ratio : ℚ) :
  eugene_speed * carlos_speed_ratio * lisa_speed_ratio = eugene_speed :=
by sorry

end lisa_speed_equals_eugene_l2671_267113


namespace ping_pong_table_distribution_l2671_267170

theorem ping_pong_table_distribution (total_tables : Nat) (total_players : Nat)
  (h_tables : total_tables = 15)
  (h_players : total_players = 38) :
  ∃ (singles_tables doubles_tables : Nat),
    singles_tables + doubles_tables = total_tables ∧
    2 * singles_tables + 4 * doubles_tables = total_players ∧
    singles_tables = 11 ∧
    doubles_tables = 4 := by
  sorry

end ping_pong_table_distribution_l2671_267170


namespace identity_proof_l2671_267184

theorem identity_proof (a b c : ℝ) (hab : a ≠ b) (hac : a ≠ c) (hbc : b ≠ c) :
  (b - c) / ((a - b) * (a - c)) + (c - a) / ((b - c) * (b - a)) + (a - b) / ((c - a) * (c - b)) =
  2 / (a - b) + 2 / (b - c) + 2 / (c - a) := by
  sorry

end identity_proof_l2671_267184


namespace inequality_proof_l2671_267196

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  (a * b) / (a + b) + (b * c) / (b + c) + (c * a) / (c + a) ≤ 3 * (a * b + b * c + c * a) / (2 * (a + b + c)) := by
  sorry

end inequality_proof_l2671_267196


namespace floor_sqrt_18_squared_l2671_267180

theorem floor_sqrt_18_squared : ⌊Real.sqrt 18⌋^2 = 16 := by
  sorry

end floor_sqrt_18_squared_l2671_267180


namespace sneakers_final_price_l2671_267125

/-- Calculates the final price of sneakers after applying discounts and sales tax -/
def finalPrice (originalPrice couponDiscount promoDiscountRate membershipDiscountRate salesTaxRate : ℚ) : ℚ :=
  let priceAfterCoupon := originalPrice - couponDiscount
  let priceAfterPromo := priceAfterCoupon * (1 - promoDiscountRate)
  let priceAfterMembership := priceAfterPromo * (1 - membershipDiscountRate)
  let finalPriceBeforeTax := priceAfterMembership
  finalPriceBeforeTax * (1 + salesTaxRate)

/-- Theorem stating that the final price of the sneakers is $100.63 -/
theorem sneakers_final_price :
  finalPrice 120 10 (5/100) (10/100) (7/100) = 10063/100 := by
  sorry

end sneakers_final_price_l2671_267125


namespace min_value_theorem_l2671_267148

theorem min_value_theorem (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : 2 * x + y - 3 = 0) :
  2 / x + 1 / y ≥ 3 ∧ ∃ (x₀ y₀ : ℝ), x₀ > 0 ∧ y₀ > 0 ∧ 2 * x₀ + y₀ - 3 = 0 ∧ 2 / x₀ + 1 / y₀ = 3 :=
by sorry

end min_value_theorem_l2671_267148


namespace initial_men_correct_l2671_267171

/-- The number of men initially working on the digging project. -/
def initial_men : ℕ := 55

/-- The number of hours worked per day in the initial condition. -/
def initial_hours : ℕ := 8

/-- The depth dug in meters in the initial condition. -/
def initial_depth : ℕ := 30

/-- The number of hours worked per day in the new condition. -/
def new_hours : ℕ := 6

/-- The depth to be dug in meters in the new condition. -/
def new_depth : ℕ := 50

/-- The additional number of men needed for the new condition. -/
def extra_men : ℕ := 11

/-- Theorem stating that the initial number of men is correct given the conditions. -/
theorem initial_men_correct :
  initial_men * initial_hours * initial_depth = (initial_men + extra_men) * new_hours * new_depth :=
by sorry

end initial_men_correct_l2671_267171


namespace impossible_to_reach_in_six_moves_l2671_267126

/-- Represents a position on the coordinate plane -/
structure Position :=
  (x : Int) (y : Int)

/-- Represents a single move of the ant -/
inductive Move
  | Up
  | Down
  | Left
  | Right

/-- Applies a move to a position -/
def apply_move (p : Position) (m : Move) : Position :=
  match m with
  | Move.Up    => ⟨p.x, p.y + 1⟩
  | Move.Down  => ⟨p.x, p.y - 1⟩
  | Move.Left  => ⟨p.x - 1, p.y⟩
  | Move.Right => ⟨p.x + 1, p.y⟩

/-- Applies a list of moves to a starting position -/
def apply_moves (start : Position) (moves : List Move) : Position :=
  moves.foldl apply_move start

/-- The sum of coordinates of a position -/
def coord_sum (p : Position) : Int := p.x + p.y

/-- Theorem: It's impossible to reach (2,1) or (1,2) from (0,0) in exactly 6 moves -/
theorem impossible_to_reach_in_six_moves :
  ∀ (moves : List Move),
    moves.length = 6 →
    (apply_moves ⟨0, 0⟩ moves ≠ ⟨2, 1⟩) ∧
    (apply_moves ⟨0, 0⟩ moves ≠ ⟨1, 2⟩) := by
  sorry

end impossible_to_reach_in_six_moves_l2671_267126


namespace f_of_f_of_2_l2671_267167

def f (x : ℝ) : ℝ := 4 * x^2 - 7

theorem f_of_f_of_2 : f (f 2) = 317 := by
  sorry

end f_of_f_of_2_l2671_267167


namespace construct_angles_l2671_267130

-- Define the given angle
def given_angle : ℝ := 40

-- Define the target angles
def target_angle_a : ℝ := 80
def target_angle_b : ℝ := 160
def target_angle_c : ℝ := 20

-- Theorem to prove the construction of target angles
theorem construct_angles :
  (given_angle + given_angle = target_angle_a) ∧
  (given_angle + given_angle + given_angle + given_angle = target_angle_b) ∧
  (180 - (given_angle + given_angle + given_angle + given_angle) = target_angle_c) :=
by sorry

end construct_angles_l2671_267130


namespace first_terrific_tuesday_after_school_starts_l2671_267138

/-- Represents a date with a month and day. -/
structure Date where
  month : Nat
  day : Nat

/-- Represents a day of the week. -/
inductive DayOfWeek
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

/-- Returns the number of days in a given month. -/
def daysInMonth (month : Nat) : Nat :=
  match month with
  | 10 => 31  -- October
  | 11 => 30  -- November
  | 12 => 31  -- December
  | _ => 30   -- Default for other months (not used in this problem)

/-- Returns the day of the week for a given date. -/
def dayOfWeek (date : Date) : DayOfWeek :=
  sorry  -- Implementation not needed for the statement

/-- Returns true if the given date is a Terrific Tuesday. -/
def isTerrificTuesday (date : Date) : Prop :=
  dayOfWeek date = DayOfWeek.Tuesday ∧
  (∀ d : Nat, d < date.day → dayOfWeek { month := date.month, day := d } = DayOfWeek.Tuesday →
    (∃ d' : Nat, d' < d ∧ dayOfWeek { month := date.month, day := d' } = DayOfWeek.Tuesday))

/-- The main theorem stating that December 31 is the first Terrific Tuesday after October 3. -/
theorem first_terrific_tuesday_after_school_starts :
  let schoolStart : Date := { month := 10, day := 3 }
  let firstTerrificTuesday : Date := { month := 12, day := 31 }
  dayOfWeek schoolStart = DayOfWeek.Tuesday →
  isTerrificTuesday firstTerrificTuesday ∧
  (∀ date : Date, schoolStart.month ≤ date.month ∧ date.month ≤ firstTerrificTuesday.month →
    (if date.month = schoolStart.month then schoolStart.day ≤ date.day else True) →
    (if date.month = firstTerrificTuesday.month then date.day ≤ firstTerrificTuesday.day else True) →
    date.day ≤ daysInMonth date.month →
    isTerrificTuesday date → date = firstTerrificTuesday) :=
by sorry

end first_terrific_tuesday_after_school_starts_l2671_267138


namespace arccos_equation_solution_l2671_267155

theorem arccos_equation_solution (x : ℝ) : 
  Real.arccos (2 * x - 1) = π / 4 → x = (Real.sqrt 2 + 2) / 4 := by
  sorry

end arccos_equation_solution_l2671_267155


namespace remaining_black_cards_l2671_267129

/-- The number of black cards in a full deck of cards -/
def full_black_cards : ℕ := 26

/-- The number of black cards taken out from the deck -/
def removed_black_cards : ℕ := 5

/-- Theorem: The number of remaining black cards is 21 -/
theorem remaining_black_cards :
  full_black_cards - removed_black_cards = 21 := by
  sorry

end remaining_black_cards_l2671_267129


namespace total_percentage_solid_colored_sum_solid_color_not_yellow_percentage_l2671_267142

/-- The percentage of marbles that are solid-colored -/
def solid_colored_percentage : ℝ := 0.90

/-- The percentage of marbles that have patterns -/
def patterned_percentage : ℝ := 0.10

/-- The percentage of red marbles among solid-colored marbles -/
def red_percentage : ℝ := 0.40

/-- The percentage of blue marbles among solid-colored marbles -/
def blue_percentage : ℝ := 0.30

/-- The percentage of green marbles among solid-colored marbles -/
def green_percentage : ℝ := 0.20

/-- The percentage of yellow marbles among solid-colored marbles -/
def yellow_percentage : ℝ := 0.10

/-- All marbles are either solid-colored or patterned -/
theorem total_percentage : solid_colored_percentage + patterned_percentage = 1 := by sorry

/-- The sum of percentages for all solid-colored marbles is 100% -/
theorem solid_colored_sum :
  red_percentage + blue_percentage + green_percentage + yellow_percentage = 1 := by sorry

/-- The percentage of marbles that are a solid color other than yellow is 81% -/
theorem solid_color_not_yellow_percentage :
  solid_colored_percentage * (red_percentage + blue_percentage + green_percentage) = 0.81 := by sorry

end total_percentage_solid_colored_sum_solid_color_not_yellow_percentage_l2671_267142


namespace emily_necklaces_l2671_267159

def beads_per_necklace : ℕ := 5
def total_beads_used : ℕ := 20

theorem emily_necklaces :
  total_beads_used / beads_per_necklace = 4 :=
by sorry

end emily_necklaces_l2671_267159


namespace two_digit_divisible_by_six_sum_fifteen_l2671_267143

theorem two_digit_divisible_by_six_sum_fifteen (n : ℕ) : 
  10 ≤ n ∧ n < 100 ∧                 -- n is a two-digit number
  n % 6 = 0 ∧                        -- n is divisible by 6
  (n / 10 + n % 10 = 15) →           -- sum of digits is 15
  (n / 10) * (n % 10) = 56 ∨ (n / 10) * (n % 10) = 54 := by
sorry

end two_digit_divisible_by_six_sum_fifteen_l2671_267143


namespace congruent_triangles_x_value_l2671_267119

/-- Given two congruent triangles ABC and DEF, where ABC has sides 3, 4, and 5,
    and DEF has sides 3, 3x-2, and 2x+1, prove that x = 2. -/
theorem congruent_triangles_x_value (x : ℝ) : 
  let a₁ : ℝ := 3
  let b₁ : ℝ := 4
  let c₁ : ℝ := 5
  let a₂ : ℝ := 3
  let b₂ : ℝ := 3 * x - 2
  let c₂ : ℝ := 2 * x + 1
  (a₁ + b₁ + c₁ = a₂ + b₂ + c₂) → x = 2 := by
  sorry

end congruent_triangles_x_value_l2671_267119


namespace abc_ordering_l2671_267198

noncomputable def a : ℝ := (1/2)^(1/4 : ℝ)
noncomputable def b : ℝ := (1/3)^(1/2 : ℝ)
noncomputable def c : ℝ := (1/4)^(1/3 : ℝ)

theorem abc_ordering : b < c ∧ c < a := by sorry

end abc_ordering_l2671_267198


namespace butterfly_cocoon_time_l2671_267134

theorem butterfly_cocoon_time :
  ∀ (L C : ℕ),
    L + C = 120 →
    L = 3 * C →
    C = 30 :=
by
  sorry

end butterfly_cocoon_time_l2671_267134


namespace real_part_of_z_l2671_267149

theorem real_part_of_z (z : ℂ) (h : Complex.I * (z + 1) = -3 + 2 * Complex.I) : 
  z.re = 1 := by
  sorry

end real_part_of_z_l2671_267149


namespace coin_order_l2671_267173

/-- Represents a coin in the arrangement -/
inductive Coin
| A | B | C | D | E | F

/-- Represents the relative position of two coins -/
inductive Position
| Above | Below

/-- Defines the relationship between two coins -/
def relation (c1 c2 : Coin) : Position := sorry

/-- Theorem stating the correct order of coins from top to bottom -/
theorem coin_order :
  (relation Coin.F Coin.E = Position.Above) ∧
  (relation Coin.F Coin.C = Position.Above) ∧
  (relation Coin.F Coin.D = Position.Above) ∧
  (relation Coin.F Coin.A = Position.Above) ∧
  (relation Coin.F Coin.B = Position.Above) ∧
  (relation Coin.E Coin.C = Position.Above) ∧
  (relation Coin.E Coin.D = Position.Above) ∧
  (relation Coin.E Coin.A = Position.Above) ∧
  (relation Coin.E Coin.B = Position.Above) ∧
  (relation Coin.C Coin.A = Position.Above) ∧
  (relation Coin.C Coin.B = Position.Above) ∧
  (relation Coin.D Coin.A = Position.Above) ∧
  (relation Coin.D Coin.B = Position.Above) ∧
  (relation Coin.A Coin.B = Position.Above) →
  ∀ (c : Coin), c ≠ Coin.F →
    (relation Coin.F c = Position.Above) ∧
    (∀ (d : Coin), d ≠ Coin.B →
      (relation c Coin.B = Position.Above)) :=
by sorry

end coin_order_l2671_267173


namespace championship_and_expectation_l2671_267150

-- Define the probabilities of class A winning each event
def p_basketball : ℝ := 0.4
def p_soccer : ℝ := 0.8
def p_badminton : ℝ := 0.6

-- Define the points awarded for winning and losing
def win_points : ℕ := 8
def lose_points : ℕ := 0

-- Define the probability of class A winning the championship
def p_championship : ℝ := 
  p_basketball * p_soccer * p_badminton +
  (1 - p_basketball) * p_soccer * p_badminton +
  p_basketball * (1 - p_soccer) * p_badminton +
  p_basketball * p_soccer * (1 - p_badminton)

-- Define the distribution of class B's total score
def p_score (x : ℕ) : ℝ :=
  if x = 0 then (1 - p_basketball) * (1 - p_soccer) * (1 - p_badminton)
  else if x = win_points then 
    p_basketball * (1 - p_soccer) * (1 - p_badminton) +
    (1 - p_basketball) * p_soccer * (1 - p_badminton) +
    (1 - p_basketball) * (1 - p_soccer) * p_badminton
  else if x = 2 * win_points then
    p_basketball * p_soccer * (1 - p_badminton) +
    p_basketball * (1 - p_soccer) * p_badminton +
    (1 - p_basketball) * p_soccer * p_badminton
  else if x = 3 * win_points then
    p_basketball * p_soccer * p_badminton
  else 0

-- Define the expectation of class B's total score
def expectation_B : ℝ :=
  0 * p_score 0 +
  win_points * p_score win_points +
  (2 * win_points) * p_score (2 * win_points) +
  (3 * win_points) * p_score (3 * win_points)

theorem championship_and_expectation :
  p_championship = 0.656 ∧ expectation_B = 12 := by
  sorry

end championship_and_expectation_l2671_267150


namespace investment_return_calculation_l2671_267181

/-- Calculates the monthly return given the current value, duration, and growth factor of an investment. -/
def calculateMonthlyReturn (currentValue : ℚ) (months : ℕ) (growthFactor : ℚ) : ℚ :=
  (currentValue * (growthFactor - 1)) / months

/-- Theorem stating that an investment tripling over 5 months with a current value of $90 has a monthly return of $12. -/
theorem investment_return_calculation :
  let currentValue : ℚ := 90
  let months : ℕ := 5
  let growthFactor : ℚ := 3
  calculateMonthlyReturn currentValue months growthFactor = 12 := by
  sorry

end investment_return_calculation_l2671_267181


namespace ellipse_m_range_l2671_267110

/-- Represents an ellipse with the given equation and foci on the y-axis -/
structure Ellipse where
  m : ℝ
  eq : ∀ (x y : ℝ), x^2 / (4 - m) + y^2 / (m - 3) = 1
  foci_on_y_axis : True

/-- The range of m for a valid ellipse with foci on the y-axis -/
theorem ellipse_m_range (e : Ellipse) : 7/2 < e.m ∧ e.m < 4 := by
  sorry

end ellipse_m_range_l2671_267110


namespace problem_solution_l2671_267146

theorem problem_solution : ∃ n : ℕ, 
  n = (2123 + 1787) * (6 * (2123 - 1787)) + 384 ∧ n = 7884144 := by
  sorry

end problem_solution_l2671_267146


namespace front_wheel_cost_l2671_267144

def initial_amount : ℕ := 60
def frame_cost : ℕ := 15
def remaining_amount : ℕ := 20

theorem front_wheel_cost : 
  initial_amount - frame_cost - remaining_amount = 25 := by sorry

end front_wheel_cost_l2671_267144


namespace inequality_solution_set_l2671_267131

def isSolutionSet (f : ℝ → ℝ) (S : Set ℝ) : Prop :=
  ∀ x, (x - 1) * f x < 0 ↔ x ∈ S

theorem inequality_solution_set 
  (f : ℝ → ℝ) 
  (h_odd : ∀ x, f (-x) = -f x)
  (h_increasing : ∀ x y, 0 < x → x < y → f x < f y)
  (h_f2 : f 2 = 0) :
  isSolutionSet f (Set.Ioo (-2) 0 ∪ Set.Ioo 1 2) :=
sorry

end inequality_solution_set_l2671_267131


namespace partial_fraction_A_value_l2671_267137

-- Define the polynomial
def p (x : ℝ) : ℝ := x^4 - 20*x^3 + 147*x^2 - 490*x + 588

-- Define the partial fraction decomposition
def partial_fraction (A B C D x : ℝ) : Prop :=
  1 / p x = A / (x + 3) + B / (x - 4) + C / (x - 4)^2 + D / (x - 7)

-- Theorem statement
theorem partial_fraction_A_value :
  ∀ A B C D : ℝ, (∀ x : ℝ, partial_fraction A B C D x) → A = -1/490 :=
by
  sorry

end partial_fraction_A_value_l2671_267137


namespace cookies_in_box_graemes_cookies_l2671_267157

/-- Given a box that can hold a certain weight of cookies and cookies of a specific weight,
    calculate the number of cookies that can fit in the box. -/
theorem cookies_in_box (box_capacity : ℕ) (cookie_weight : ℕ) (ounces_per_pound : ℕ) : ℕ :=
  let cookies_per_pound := ounces_per_pound / cookie_weight
  box_capacity * cookies_per_pound

/-- Prove that given a box that can hold 40 pounds of cookies, and each cookie weighing 2 ounces,
    the number of cookies that can fit in the box is equal to 320. -/
theorem graemes_cookies :
  cookies_in_box 40 2 16 = 320 := by
  sorry

end cookies_in_box_graemes_cookies_l2671_267157


namespace cone_angle_cosine_l2671_267168

/-- Given a cone whose side surface unfolds into a sector with central angle 4π/3 and radius 18 cm,
    prove that the cosine of the angle between the slant height and the base is 2/3 -/
theorem cone_angle_cosine (θ : Real) (l r : Real) : 
  θ = 4 / 3 * π → 
  l = 18 → 
  θ = 2 * π * r / l → 
  r / l = 2 / 3 := by sorry

end cone_angle_cosine_l2671_267168


namespace tangent_circle_rectangle_area_l2671_267147

/-- A rectangle with a circle tangent to three sides and passing through the diagonal midpoint -/
structure TangentCircleRectangle where
  /-- The length of the rectangle -/
  w : ℝ
  /-- The height of the rectangle -/
  h : ℝ
  /-- The radius of the circle -/
  r : ℝ
  /-- The circle is tangent to three sides -/
  tangent_to_sides : h = r
  /-- The circle passes through the midpoint of the diagonal -/
  passes_through_midpoint : r^2 = (w/2)^2 + (h/2)^2

/-- The area of the rectangle is √3 * r^2 -/
theorem tangent_circle_rectangle_area (rect : TangentCircleRectangle) : 
  rect.w * rect.h = Real.sqrt 3 * rect.r^2 := by
  sorry

end tangent_circle_rectangle_area_l2671_267147


namespace completing_square_transformation_l2671_267133

theorem completing_square_transformation (x : ℝ) :
  (x^2 - 2*x - 5 = 0) ↔ ((x - 1)^2 = 6) :=
sorry

end completing_square_transformation_l2671_267133


namespace sum_of_last_two_digits_l2671_267179

theorem sum_of_last_two_digits (n : ℕ) : (7^25 + 13^25) % 100 = 0 := by
  sorry

end sum_of_last_two_digits_l2671_267179


namespace route_time_difference_l2671_267116

-- Define the times for each stage of the first route
def first_route_uphill : ℕ := 6
def first_route_path : ℕ := 2 * first_route_uphill
def first_route_final (t : ℕ) : ℕ := t / 3

-- Define the times for each stage of the second route
def second_route_flat : ℕ := 14
def second_route_final : ℕ := 2 * second_route_flat

-- Calculate the total time for the first route
def first_route_total : ℕ := 
  first_route_uphill + first_route_path + first_route_final (first_route_uphill + first_route_path)

-- Calculate the total time for the second route
def second_route_total : ℕ := second_route_flat + second_route_final

-- Theorem stating the difference between the two routes
theorem route_time_difference : second_route_total - first_route_total = 18 := by
  sorry

end route_time_difference_l2671_267116


namespace probability_three_white_balls_l2671_267114

def total_balls : ℕ := 15
def white_balls : ℕ := 7
def black_balls : ℕ := 8
def drawn_balls : ℕ := 3

theorem probability_three_white_balls :
  (Nat.choose white_balls drawn_balls : ℚ) / (Nat.choose total_balls drawn_balls : ℚ) = 7 / 91 := by
  sorry

end probability_three_white_balls_l2671_267114


namespace number_of_sides_interior_angle_measure_l2671_267100

/-- A regular polygon where the sum of interior angles is 180° more than three times the sum of exterior angles. -/
structure RegularPolygon where
  n : ℕ  -- number of sides
  sum_interior_angles : ℝ
  sum_exterior_angles : ℝ
  h1 : sum_interior_angles = (n - 2) * 180
  h2 : sum_exterior_angles = 360
  h3 : sum_interior_angles = 3 * sum_exterior_angles + 180

/-- The number of sides of the regular polygon is 9. -/
theorem number_of_sides (p : RegularPolygon) : p.n = 9 := by sorry

/-- The measure of each interior angle of the regular polygon is 140°. -/
theorem interior_angle_measure (p : RegularPolygon) : 
  (p.sum_interior_angles / p.n : ℝ) = 140 := by sorry

end number_of_sides_interior_angle_measure_l2671_267100


namespace fraction_simplification_l2671_267183

/-- 
For any integer n, the fraction (5n+3)/(7n+8) can be simplified by 5 
if and only if n is divisible by 5 or n is of the form 19k + 7 for some integer k.
-/
theorem fraction_simplification (n : ℤ) : 
  (∃ (m : ℤ), 5 * (7*n + 8) = 7 * (5*n + 3) * m) ↔ 
  (∃ (k : ℤ), n = 5*k ∨ n = 19*k + 7) := by
sorry

end fraction_simplification_l2671_267183


namespace probability_two_red_one_blue_l2671_267107

/-- Represents a cube composed of smaller unit cubes -/
structure Cube where
  edge_length : ℕ

/-- Represents the painting state of a smaller cube -/
inductive PaintState
  | Unpainted
  | Red
  | Blue
  | RedAndBlue

/-- Represents a painted cube -/
structure PaintedCube where
  cube : Cube
  paint : Cube → PaintState

/-- Calculates the number of cubes with exactly two red faces and one blue face -/
def cubes_with_two_red_one_blue (c : PaintedCube) : ℕ := sorry

/-- Calculates the total number of unit cubes in a larger cube -/
def total_unit_cubes (c : Cube) : ℕ := c.edge_length ^ 3

/-- Theorem stating the probability of selecting a cube with two red faces and one blue face -/
theorem probability_two_red_one_blue (c : PaintedCube) 
  (h1 : c.cube.edge_length = 8)
  (h2 : ∀ (x : Cube), x.edge_length = 1 → c.paint x ≠ PaintState.Unpainted) 
  (h3 : ∃ (layer : ℕ), layer < c.cube.edge_length ∧ 
    ∀ (x : Cube), x.edge_length = 1 → 
      (∃ (i j : ℕ), i < c.cube.edge_length ∧ j < c.cube.edge_length ∧
        (i = layer ∨ i = c.cube.edge_length - 1 - layer ∨
         j = layer ∨ j = c.cube.edge_length - 1 - layer)) →
      c.paint x = PaintState.Blue) :
  (cubes_with_two_red_one_blue c : ℚ) / (total_unit_cubes c.cube : ℚ) = 3 / 32 := by sorry

end probability_two_red_one_blue_l2671_267107


namespace sine_function_period_l2671_267192

theorem sine_function_period (a b c d : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0) :
  (∀ x : ℝ, ∃ y : ℝ, y = a * Real.sin (b * x + c) + d) →
  (∃ x : ℝ, x = 4 * Real.pi ∧ (x / (2 * Real.pi / b) = 5)) →
  b = 5 / 2 := by
sorry

end sine_function_period_l2671_267192


namespace baseball_season_length_l2671_267186

/-- The number of baseball games in a month -/
def games_per_month : ℕ := 7

/-- The total number of baseball games in a season -/
def games_in_season : ℕ := 14

/-- The number of months in a baseball season -/
def season_length : ℕ := games_in_season / games_per_month

theorem baseball_season_length :
  season_length = 2 := by sorry

end baseball_season_length_l2671_267186


namespace bin_draw_probability_l2671_267190

def bin_probability (black white drawn : ℕ) : ℚ :=
  let total := black + white
  let ways_3b1w := (black.choose 3) * (white.choose 1)
  let ways_1b3w := (black.choose 1) * (white.choose 3)
  let favorable := ways_3b1w + ways_1b3w
  let total_ways := total.choose drawn
  (favorable : ℚ) / total_ways

theorem bin_draw_probability : 
  bin_probability 10 8 4 = 19 / 38 := by
  sorry

end bin_draw_probability_l2671_267190


namespace april_roses_unsold_l2671_267118

/-- Calculates the number of roses left unsold given the initial number of roses,
    the price per rose, and the total amount earned from sales. -/
def roses_left_unsold (initial_roses : ℕ) (price_per_rose : ℕ) (total_earned : ℕ) : ℕ :=
  initial_roses - (total_earned / price_per_rose)

/-- Proves that the number of roses left unsold is 4 given the problem conditions. -/
theorem april_roses_unsold :
  roses_left_unsold 9 7 35 = 4 := by
  sorry

end april_roses_unsold_l2671_267118


namespace smallest_k_cosine_squared_l2671_267191

theorem smallest_k_cosine_squared (k : ℕ) : k = 53 ↔ 
  (k > 0 ∧ 
   ∀ m : ℕ, m > 0 → m < k → (Real.cos ((m^2 + 7^2 : ℝ) * Real.pi / 180))^2 ≠ 1) ∧
  (Real.cos ((k^2 + 7^2 : ℝ) * Real.pi / 180))^2 = 1 :=
sorry

end smallest_k_cosine_squared_l2671_267191


namespace ratio_of_numbers_l2671_267158

theorem ratio_of_numbers (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x > y) (h4 : x + y = 7 * (x - y)) : x / y = 2 := by
  sorry

end ratio_of_numbers_l2671_267158


namespace min_value_quadratic_l2671_267151

theorem min_value_quadratic (x y : ℝ) : 5 * x^2 - 4 * x * y + y^2 + 6 * x + 25 ≥ 16 := by
  sorry

end min_value_quadratic_l2671_267151


namespace function_property_l2671_267123

-- Define the functions
def f₁ (x : ℝ) : ℝ := |2 * x|
def f₂ (x : ℝ) : ℝ := x
noncomputable def f₃ (x : ℝ) : ℝ := Real.sqrt x
def f₄ (x : ℝ) : ℝ := x - |x|

-- State the theorem
theorem function_property :
  (∀ x, f₁ (2 * x) = 2 * f₁ x) ∧
  (∀ x, f₂ (2 * x) = 2 * f₂ x) ∧
  (∃ x, f₃ (2 * x) ≠ 2 * f₃ x) ∧
  (∀ x, f₄ (2 * x) = 2 * f₄ x) :=
by
  sorry

end function_property_l2671_267123


namespace min_framing_for_picture_l2671_267115

/-- Calculates the minimum number of linear feet of framing needed for an enlarged and bordered picture. -/
def min_framing_feet (original_width original_height enlarge_factor border_width : ℕ) : ℕ :=
  let enlarged_width := original_width * enlarge_factor
  let enlarged_height := original_height * enlarge_factor
  let total_width := enlarged_width + 2 * border_width
  let total_height := enlarged_height + 2 * border_width
  let perimeter_inches := 2 * (total_width + total_height)
  (perimeter_inches + 11) / 12  -- Round up to the nearest foot

/-- Theorem stating the minimum framing needed for the given picture specifications. -/
theorem min_framing_for_picture :
  min_framing_feet 5 7 4 3 = 10 := by
  sorry

end min_framing_for_picture_l2671_267115


namespace ratio_problem_l2671_267177

theorem ratio_problem (a b c d : ℚ) 
  (hab : a / b = 5 / 4)
  (hcd : c / d = 4 / 3)
  (hdb : d / b = 1 / 5) :
  a / c = 75 / 16 := by
  sorry

end ratio_problem_l2671_267177


namespace second_storm_duration_l2671_267127

/-- Represents the duration and rainfall rate of a rainstorm -/
structure Rainstorm where
  duration : ℝ
  rate : ℝ

/-- Proves that the second rainstorm lasted 25 hours given the conditions -/
theorem second_storm_duration
  (storm1 : Rainstorm)
  (storm2 : Rainstorm)
  (h1 : storm1.rate = 30)
  (h2 : storm2.rate = 15)
  (h3 : storm1.duration + storm2.duration = 45)
  (h4 : storm1.rate * storm1.duration + storm2.rate * storm2.duration = 975) :
  storm2.duration = 25 := by
  sorry

#check second_storm_duration

end second_storm_duration_l2671_267127


namespace right_triangle_consecutive_legs_l2671_267112

theorem right_triangle_consecutive_legs (a b c : ℕ) : 
  a + 1 = b →                 -- legs are consecutive whole numbers
  a^2 + b^2 = 41^2 →          -- Pythagorean theorem with hypotenuse 41
  a + b = 57 := by            -- sum of legs is 57
sorry

end right_triangle_consecutive_legs_l2671_267112


namespace division_value_problem_l2671_267178

theorem division_value_problem (x : ℝ) : (3 / x) * 12 = 9 → x = 4 := by
  sorry

end division_value_problem_l2671_267178


namespace drug_price_reduction_l2671_267122

theorem drug_price_reduction (initial_price final_price : ℝ) (x : ℝ) 
  (h_initial : initial_price = 56)
  (h_final : final_price = 31.5)
  (h_positive : 0 < x ∧ x < 1) :
  initial_price * (1 - x)^2 = final_price := by
  sorry

end drug_price_reduction_l2671_267122


namespace strip_covers_cube_l2671_267117

/-- A rectangular strip can cover a cube in two layers -/
theorem strip_covers_cube (strip_length : ℝ) (strip_width : ℝ) (cube_edge : ℝ) :
  strip_length = 12 →
  strip_width = 1 →
  cube_edge = 1 →
  strip_length * strip_width = 2 * 6 * cube_edge ^ 2 := by
  sorry

#check strip_covers_cube

end strip_covers_cube_l2671_267117


namespace decimal_sum_to_fraction_l2671_267199

theorem decimal_sum_to_fraction :
  (0.4 + 0.05 + 0.006 + 0.0007 + 0.00008 : ℚ) = 22839 / 50000 := by
  sorry

end decimal_sum_to_fraction_l2671_267199


namespace arithmetic_sequence_forms_straight_line_l2671_267182

def isArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) - a n = d

theorem arithmetic_sequence_forms_straight_line
  (a : ℕ → ℝ) (h : isArithmeticSequence a) :
  ∃ m b : ℝ, ∀ n : ℕ, a n = m * n + b :=
sorry

end arithmetic_sequence_forms_straight_line_l2671_267182


namespace min_time_for_all_flashes_l2671_267135

/-- Represents the three possible colors of the lights -/
inductive Color
  | Red
  | Yellow
  | Green

/-- A flash is a sequence of three different colors -/
def Flash := { seq : Fin 3 → Color // ∀ i j, i ≠ j → seq i ≠ seq j }

/-- The number of different possible flashes -/
def numFlashes : Nat := 6

/-- Duration of one flash in seconds -/
def flashDuration : Nat := 3

/-- Interval between consecutive flashes in seconds -/
def intervalDuration : Nat := 3

/-- Theorem: The minimum time required to achieve all different flashes is 33 seconds -/
theorem min_time_for_all_flashes : 
  numFlashes * flashDuration + (numFlashes - 1) * intervalDuration = 33 := by
  sorry

end min_time_for_all_flashes_l2671_267135


namespace intersection_condition_l2671_267136

/-- The set A parameterized by m -/
def A (m : ℝ) : Set (ℝ × ℝ) :=
  {p | p.1^2 + m * p.1 - p.2 + 2 = 0}

/-- The set B -/
def B : Set (ℝ × ℝ) :=
  {p | p.1 - p.2 + 1 = 0}

/-- Theorem stating the condition for non-empty intersection of A and B -/
theorem intersection_condition (m : ℝ) :
  (A m ∩ B).Nonempty ↔ m ≤ -1 ∨ m ≥ 3 := by
  sorry

end intersection_condition_l2671_267136


namespace perimeter_of_z_shape_l2671_267175

-- Define the complex number z
variable (z : ℂ)

-- Define the condition that z satisfies
def satisfies_condition (z : ℂ) : Prop :=
  Complex.arg z = Complex.arg (Complex.I * z + Complex.I)

-- Define the shape corresponding to z
def shape_of_z (z : ℂ) : Set ℂ :=
  {w : ℂ | ∃ t : ℝ, w = t * z ∧ 0 ≤ t ∧ t ≤ 1}

-- Define the perimeter of the shape
def perimeter_of_shape (s : Set ℂ) : ℝ := sorry

-- State the theorem
theorem perimeter_of_z_shape (h : satisfies_condition z) :
  perimeter_of_shape (shape_of_z z) = π / 2 :=
sorry

end perimeter_of_z_shape_l2671_267175


namespace complement_P_intersect_Q_l2671_267140

-- Define the sets P and Q
def P : Set ℝ := {x | x ≥ 2}
def Q : Set ℝ := {x | 1 < x ∧ x ≤ 2}

-- State the theorem
theorem complement_P_intersect_Q : (P.compl) ∩ Q = Set.Ioo 1 2 := by sorry

end complement_P_intersect_Q_l2671_267140


namespace unique_number_with_three_prime_factors_l2671_267165

theorem unique_number_with_three_prime_factors (x n : ℕ) : 
  x = 6^n + 1 →
  Odd n →
  (∃ p q : ℕ, Prime p ∧ Prime q ∧ p ≠ q ∧ p ≠ 11 ∧ q ≠ 11 ∧ x = 11 * p * q) →
  (∀ r : ℕ, Prime r ∧ r ∣ x → r = 11 ∨ r = p ∨ r = q) →
  x = 7777 := by
sorry

end unique_number_with_three_prime_factors_l2671_267165


namespace no_integer_sqrt_difference_150_l2671_267176

theorem no_integer_sqrt_difference_150 :
  (∃ (x : ℤ), x - Real.sqrt x = 20) ∧
  (∃ (x : ℤ), x - Real.sqrt x = 30) ∧
  (∃ (x : ℤ), x - Real.sqrt x = 110) ∧
  (∀ (x : ℤ), x - Real.sqrt x ≠ 150) ∧
  (∃ (x : ℤ), x - Real.sqrt x = 600) := by
  sorry

#check no_integer_sqrt_difference_150

end no_integer_sqrt_difference_150_l2671_267176


namespace sets_equality_l2671_267101

-- Define the sets M and N
def M : Set ℝ := {x | x^2 - 2*x + 1 = 0}
def N : Set ℝ := {1}

-- Theorem statement
theorem sets_equality : M = N := by
  sorry

end sets_equality_l2671_267101


namespace marathon_distance_l2671_267109

theorem marathon_distance (marathons : ℕ) (miles_per_marathon : ℕ) (yards_per_marathon : ℕ) 
  (yards_per_mile : ℕ) (h1 : marathons = 15) (h2 : miles_per_marathon = 26) 
  (h3 : yards_per_marathon = 395) (h4 : yards_per_mile = 1760) : 
  ∃ (m : ℕ) (y : ℕ), 
    (marathons * miles_per_marathon * yards_per_mile + marathons * yards_per_marathon = 
      m * yards_per_mile + y) ∧ 
    y < yards_per_mile ∧ 
    y = 645 := by
  sorry

end marathon_distance_l2671_267109


namespace finite_intersection_l2671_267103

def sequence_a : ℕ → ℕ → ℕ
  | a₁, 0 => a₁
  | a₁, n + 1 => n * sequence_a a₁ n + 1

def sequence_b : ℕ → ℕ → ℕ
  | b₁, 0 => b₁
  | b₁, n + 1 => n * sequence_b b₁ n - 1

theorem finite_intersection (a₁ b₁ : ℕ) :
  ∃ (N : ℕ), ∀ (n : ℕ), n ≥ N → sequence_a a₁ n ≠ sequence_b b₁ n :=
sorry

end finite_intersection_l2671_267103


namespace range_of_f_l2671_267160

def P : Set ℕ := {1, 2, 3}

def f (x : ℕ) : ℕ := 2^x

theorem range_of_f :
  {y | ∃ x ∈ P, f x = y} = {2, 4, 8} := by sorry

end range_of_f_l2671_267160


namespace complementary_events_l2671_267169

-- Define the sample space for a die throw
def DieThrow : Type := Fin 6

-- Define event A: upward face shows an odd number
def eventA : Set DieThrow := {x | x.val % 2 = 1}

-- Define event B: upward face shows an even number
def eventB : Set DieThrow := {x | x.val % 2 = 0}

-- Theorem stating that A and B are complementary events
theorem complementary_events : 
  eventA ∪ eventB = Set.univ ∧ eventA ∩ eventB = ∅ := by
  sorry

end complementary_events_l2671_267169


namespace game_lives_theorem_l2671_267106

/-- Calculates the total number of lives after two levels in a game. -/
def totalLives (initial : ℕ) (firstLevelGain : ℕ) (secondLevelGain : ℕ) : ℕ :=
  initial + firstLevelGain + secondLevelGain

/-- Theorem stating that with 2 initial lives, gaining 6 in the first level
    and 11 in the second level, the total number of lives is 19. -/
theorem game_lives_theorem :
  totalLives 2 6 11 = 19 := by
  sorry

end game_lives_theorem_l2671_267106


namespace inverse_sum_mod_13_l2671_267197

theorem inverse_sum_mod_13 :
  (((2⁻¹ : ZMod 13) + (5⁻¹ : ZMod 13) + (9⁻¹ : ZMod 13))⁻¹ : ZMod 13) = 8 := by
  sorry

end inverse_sum_mod_13_l2671_267197


namespace league_games_l2671_267162

theorem league_games (n : ℕ) (h : n = 11) : 
  (n * (n - 1)) / 2 = 55 := by
  sorry

end league_games_l2671_267162


namespace camp_hair_colors_l2671_267185

theorem camp_hair_colors (total : ℕ) (brown green black : ℕ) : 
  brown = total / 2 →
  brown = 25 →
  green = 10 →
  black = 5 →
  total - (brown + green + black) = 10 := by
sorry

end camp_hair_colors_l2671_267185


namespace average_tomatoes_proof_l2671_267102

/-- The number of tomatoes reaped on day 1 -/
def day1_tomatoes : ℕ := 120

/-- The number of tomatoes reaped on day 2 -/
def day2_tomatoes : ℕ := day1_tomatoes + 50

/-- The number of tomatoes reaped on day 3 -/
def day3_tomatoes : ℕ := 2 * day2_tomatoes

/-- The number of tomatoes reaped on day 4 -/
def day4_tomatoes : ℕ := day1_tomatoes / 2

/-- The total number of tomatoes reaped over 4 days -/
def total_tomatoes : ℕ := day1_tomatoes + day2_tomatoes + day3_tomatoes + day4_tomatoes

/-- The number of days -/
def num_days : ℕ := 4

/-- The average number of tomatoes reaped per day -/
def average_tomatoes : ℚ := total_tomatoes / num_days

theorem average_tomatoes_proof : average_tomatoes = 172.5 := by
  sorry

end average_tomatoes_proof_l2671_267102


namespace sin_thirty_degrees_l2671_267132

theorem sin_thirty_degrees : Real.sin (30 * π / 180) = 1 / 2 := by
  sorry

end sin_thirty_degrees_l2671_267132


namespace projectile_max_height_l2671_267193

/-- The height function of the projectile -/
def h (t : ℝ) : ℝ := -20 * t^2 + 100 * t + 36

/-- The maximum height reached by the projectile -/
theorem projectile_max_height : 
  ∃ (t_max : ℝ), ∀ (t : ℝ), h t ≤ h t_max ∧ h t_max = 161 :=
sorry

end projectile_max_height_l2671_267193


namespace savings_theorem_l2671_267161

def initial_amount : ℚ := 2000

def wife_share (amount : ℚ) : ℚ := (2 / 5) * amount

def first_son_share (amount : ℚ) : ℚ := (2 / 5) * amount

def second_son_share (amount : ℚ) : ℚ := (2 / 5) * amount

def savings_amount (initial : ℚ) : ℚ :=
  let after_wife := initial - wife_share initial
  let after_first_son := after_wife - first_son_share after_wife
  after_first_son - second_son_share after_first_son

theorem savings_theorem : savings_amount initial_amount = 432 := by
  sorry

end savings_theorem_l2671_267161


namespace first_term_of_specific_series_l2671_267104

/-- Given an infinite geometric series with sum S and sum of squares T,
    this function returns the first term of the series. -/
def first_term_of_geometric_series (S : ℝ) (T : ℝ) : ℝ := 
  sorry

/-- Theorem stating that for an infinite geometric series with sum 27 and 
    sum of squares 108, the first term is 216/31. -/
theorem first_term_of_specific_series : 
  first_term_of_geometric_series 27 108 = 216 / 31 := by
  sorry

end first_term_of_specific_series_l2671_267104


namespace valid_pythagorean_grid_exists_l2671_267111

/-- A 3x3 grid of integers -/
def Grid := Fin 3 → Fin 3 → ℕ

/-- Check if three numbers form a Pythagorean triple -/
def isPythagoreanTriple (a b c : ℕ) : Prop :=
  a * a + b * b = c * c

/-- Check if all numbers in the grid are distinct -/
def allDistinct (g : Grid) : Prop :=
  ∀ i j k l, (i ≠ k ∨ j ≠ l) → g i j ≠ g k l

/-- Check if all numbers in the grid are less than 100 -/
def allLessThan100 (g : Grid) : Prop :=
  ∀ i j, g i j < 100

/-- Check if all rows in the grid form Pythagorean triples -/
def allRowsPythagorean (g : Grid) : Prop :=
  ∀ i, isPythagoreanTriple (g i 0) (g i 1) (g i 2)

/-- Check if all columns in the grid form Pythagorean triples -/
def allColumnsPythagorean (g : Grid) : Prop :=
  ∀ j, isPythagoreanTriple (g 0 j) (g 1 j) (g 2 j)

/-- The main theorem: there exists a valid grid satisfying all conditions -/
theorem valid_pythagorean_grid_exists : ∃ (g : Grid),
  allDistinct g ∧
  allLessThan100 g ∧
  allRowsPythagorean g ∧
  allColumnsPythagorean g :=
sorry

end valid_pythagorean_grid_exists_l2671_267111


namespace population_reaches_capacity_l2671_267156

-- Define the constants
def land_area : ℕ := 40000
def acres_per_person : ℕ := 1
def base_population : ℕ := 500
def years_to_quadruple : ℕ := 20

-- Define the population growth function
def population (years : ℕ) : ℕ :=
  base_population * (4 ^ (years / years_to_quadruple))

-- Define the maximum capacity
def max_capacity : ℕ := land_area / acres_per_person

-- Theorem to prove
theorem population_reaches_capacity : 
  population 60 ≥ max_capacity ∧ population 40 < max_capacity :=
sorry

end population_reaches_capacity_l2671_267156


namespace equation_solution_l2671_267108

theorem equation_solution : ∃ y : ℚ, (3 * (y + 1) / 4 - (1 - y) / 8 = 1) ∧ (y = 3 / 7) := by
  sorry

end equation_solution_l2671_267108


namespace unique_three_digit_number_l2671_267163

theorem unique_three_digit_number : ∃! n : ℕ, 
  100 ≤ n ∧ n < 1000 ∧ 
  (n / 11 : ℚ) = (n / 100 : ℕ) + ((n / 10) % 10 : ℕ) + (n % 10 : ℕ) ∧
  n = 198 := by
sorry

end unique_three_digit_number_l2671_267163


namespace calculate_expression_l2671_267105

theorem calculate_expression : (3752 / (39 * 2) + 5030 / (39 * 10) : ℚ) = 61 := by
  sorry

end calculate_expression_l2671_267105


namespace carries_cake_profit_l2671_267166

/-- Calculates the profit for a cake decorator given their work hours, pay rate, and supply cost. -/
def cake_decorator_profit (hours_per_day : ℕ) (days_worked : ℕ) (hourly_rate : ℕ) (supply_cost : ℕ) : ℕ :=
  hours_per_day * days_worked * hourly_rate - supply_cost

/-- Proves that Carrie's profit from decorating a wedding cake is $122. -/
theorem carries_cake_profit :
  cake_decorator_profit 2 4 22 54 = 122 := by
  sorry

end carries_cake_profit_l2671_267166


namespace max_b_plus_c_l2671_267188

theorem max_b_plus_c (a b c : ℕ) (h1 : a > b) (h2 : a + b = 18) (h3 : c = b + 2) :
  b + c ≤ 18 ∧ ∃ (b' c' : ℕ), b' + c' = 18 ∧ a > b' ∧ a + b' = 18 ∧ c' = b' + 2 := by
  sorry

end max_b_plus_c_l2671_267188


namespace profit_function_correct_max_profit_production_break_even_range_l2671_267124

/-- Represents the production and profit model of a company -/
structure CompanyModel where
  fixedCost : ℝ
  variableCost : ℝ
  annualDemand : ℝ
  revenueFunction : ℝ → ℝ

/-- The company's specific model -/
def company : CompanyModel :=
  { fixedCost := 0.5,  -- In ten thousand yuan
    variableCost := 0.025,  -- In ten thousand yuan per hundred units
    annualDemand := 5,  -- In hundreds of units
    revenueFunction := λ x => 5 * x - x^2 }  -- In ten thousand yuan

/-- The profit function for the company -/
def profitFunction (x : ℝ) : ℝ :=
  company.revenueFunction x - (company.variableCost * x + company.fixedCost)

/-- Theorem stating the correctness of the profit function -/
theorem profit_function_correct (x : ℝ) (h : 0 ≤ x ∧ x ≤ 5) :
  profitFunction x = 5 * x - x^2 - (0.025 * x + 0.5) := by sorry

/-- Theorem stating the maximum profit production -/
theorem max_profit_production :
  ∃ x, x = 4.75 ∧ ∀ y, 0 ≤ y ∧ y ≤ 5 → profitFunction x ≥ profitFunction y := by sorry

/-- Theorem stating the break-even production range -/
theorem break_even_range :
  ∃ a b, a = 0.1 ∧ b = 48 ∧ 
  ∀ x, (a ≤ x ∧ x ≤ 5) ∨ (5 < x ∧ x < b) → profitFunction x ≥ 0 := by sorry

end profit_function_correct_max_profit_production_break_even_range_l2671_267124


namespace triangle_ap_tangent_relation_l2671_267189

/-- A triangle with sides in arithmetic progression satisfies 3 * tan(α/2) * tan(γ/2) = 1, 
    where α is the smallest angle and γ is the largest angle. -/
theorem triangle_ap_tangent_relation (a b c : ℝ) (α β γ : ℝ) : 
  a > 0 → b > 0 → c > 0 → 
  α > 0 → β > 0 → γ > 0 →
  α + β + γ = π →
  a + c = 2 * b →  -- Arithmetic progression condition
  α ≤ β → β ≤ γ →  -- α is smallest, γ is largest
  3 * Real.tan (α / 2) * Real.tan (γ / 2) = 1 := by
  sorry

end triangle_ap_tangent_relation_l2671_267189


namespace power_equation_solution_l2671_267145

theorem power_equation_solution : ∃ y : ℕ, (12 ^ 3 * 6 ^ y) / 432 = 5184 :=
by
  use 4
  sorry

end power_equation_solution_l2671_267145


namespace homework_points_calculation_l2671_267128

theorem homework_points_calculation (total_points : ℕ) 
  (h1 : total_points = 265)
  (h2 : ∀ (test_points quiz_points : ℕ), test_points = 4 * quiz_points)
  (h3 : ∀ (quiz_points homework_points : ℕ), quiz_points = homework_points + 5) :
  ∃ (homework_points : ℕ), 
    homework_points = 40 ∧ 
    homework_points + (homework_points + 5) + 4 * (homework_points + 5) = total_points :=
by sorry

end homework_points_calculation_l2671_267128


namespace cos_sin_sum_zero_l2671_267120

theorem cos_sin_sum_zero (θ a : Real) (h : Real.cos (π / 6 - θ) = a) :
  Real.cos (5 * π / 6 + θ) + Real.sin (2 * π / 3 - θ) = 0 := by
  sorry

end cos_sin_sum_zero_l2671_267120


namespace heart_king_probability_l2671_267154

/-- Represents a standard deck of cards -/
def StandardDeck : ℕ := 52

/-- Number of hearts in a standard deck -/
def NumHearts : ℕ := 13

/-- Number of kings in a standard deck -/
def NumKings : ℕ := 4

/-- Probability of drawing a heart as the first card and a king as the second card -/
def prob_heart_then_king (deck : ℕ) (hearts : ℕ) (kings : ℕ) : ℚ :=
  (hearts / deck) * (kings / (deck - 1))

theorem heart_king_probability :
  prob_heart_then_king StandardDeck NumHearts NumKings = 1 / StandardDeck := by
  sorry

end heart_king_probability_l2671_267154


namespace complex_equation_solution_existence_l2671_267195

theorem complex_equation_solution_existence :
  ∃ (z : ℂ), z * (z + 2*I) * (z + 4*I) = 4012*I ∧
  ∃ (a b : ℝ), z = a + b*I :=
sorry

end complex_equation_solution_existence_l2671_267195


namespace male_average_height_l2671_267139

/-- Proves that the average height of males in a school is 185 cm given the following conditions:
  - The average height of all students is 180 cm
  - The average height of females is 170 cm
  - The ratio of men to women is 2:1
-/
theorem male_average_height (total_avg : ℝ) (female_avg : ℝ) (male_female_ratio : ℚ) :
  total_avg = 180 →
  female_avg = 170 →
  male_female_ratio = 2 →
  ∃ (male_avg : ℝ), male_avg = 185 := by
  sorry

end male_average_height_l2671_267139


namespace four_numbers_property_l2671_267174

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(d ∣ n)

theorem four_numbers_property (a b c d : ℕ) : 
  a = 1 → b = 2 → c = 3 → d = 5 →
  is_prime (a * b + c * d) ∧ 
  is_prime (a * c + b * d) ∧ 
  is_prime (a * d + b * c) := by
sorry

end four_numbers_property_l2671_267174


namespace rectangle_perimeter_area_ratio_bound_l2671_267164

/-- A function that checks if a number is prime --/
def isPrime (n : ℕ) : Prop := n > 1 ∧ ∀ m : ℕ, m > 1 → m < n → ¬(n % m = 0)

/-- The theorem statement --/
theorem rectangle_perimeter_area_ratio_bound :
  ∀ l w : ℕ,
    l < 100 →
    w < 100 →
    l ≠ w →
    isPrime l →
    isPrime w →
    (2 * l + 2 * w)^2 / (l * w : ℚ) ≥ 82944 / 5183 := by
  sorry

end rectangle_perimeter_area_ratio_bound_l2671_267164


namespace bank_teller_bills_l2671_267152

theorem bank_teller_bills (total_bills : ℕ) (total_value : ℕ) : 
  total_bills = 54 → total_value = 780 → 
  ∃ (five_dollar_bills twenty_dollar_bills : ℕ), 
    five_dollar_bills + twenty_dollar_bills = total_bills ∧
    5 * five_dollar_bills + 20 * twenty_dollar_bills = total_value ∧
    five_dollar_bills = 20 := by
  sorry

end bank_teller_bills_l2671_267152


namespace bryans_books_l2671_267187

/-- Given that Bryan has 9 bookshelves and each bookshelf contains 56 books,
    prove that the total number of books Bryan has is 504. -/
theorem bryans_books (num_shelves : ℕ) (books_per_shelf : ℕ) 
    (h1 : num_shelves = 9) (h2 : books_per_shelf = 56) : 
    num_shelves * books_per_shelf = 504 := by
  sorry

end bryans_books_l2671_267187


namespace weight_difference_l2671_267121

theorem weight_difference (anne_weight douglas_weight : ℕ) 
  (h1 : anne_weight = 67) 
  (h2 : douglas_weight = 52) : 
  anne_weight - douglas_weight = 15 := by
  sorry

end weight_difference_l2671_267121


namespace M_factors_l2671_267153

/-- The number of positive integer factors of M, where
    M = 57^6 + 6 * 57^5 + 15 * 57^4 + 20 * 57^3 + 15 * 57^2 + 6 * 57 + 1 -/
def M : ℕ := 57^6 + 6 * 57^5 + 15 * 57^4 + 20 * 57^3 + 15 * 57^2 + 6 * 57 + 1

/-- The number of positive integer factors of a natural number n -/
def numFactors (n : ℕ) : ℕ := (Finset.filter (· ∣ n) (Finset.range (n + 1))).card

theorem M_factors : numFactors M = 49 := by
  sorry

end M_factors_l2671_267153


namespace problem_1_l2671_267141

theorem problem_1 (x : ℝ) : 4 * x^2 * (x - 1/4) = 4 * x^3 - x^2 := by
  sorry

end problem_1_l2671_267141


namespace calculation_proof_l2671_267172

theorem calculation_proof : (-3/4 - 5/9 + 7/12) / (-1/36) = 26 := by
  sorry

end calculation_proof_l2671_267172


namespace gear_system_rotation_l2671_267194

/-- Represents a circular arrangement of gears -/
structure GearSystem where
  n : ℕ  -- number of gears
  circular : Bool  -- true if the arrangement is circular

/-- Defines when a gear system can rotate -/
def can_rotate (gs : GearSystem) : Prop :=
  gs.circular ∧ Even gs.n

/-- Theorem: A circular gear system can rotate if and only if the number of gears is even -/
theorem gear_system_rotation (gs : GearSystem) (h : gs.circular = true) : 
  can_rotate gs ↔ Even gs.n :=
sorry

end gear_system_rotation_l2671_267194
