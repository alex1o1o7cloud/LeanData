import Mathlib

namespace equation_solution_l910_91098

theorem equation_solution (x y k z : ℝ) (hx : x ≠ 0) (hy : y ≠ 0) (hk : k ≠ 0)
  (h : 1/x + 1/y = k/z) : z = x*y / (k*(y+x)) :=
sorry

end equation_solution_l910_91098


namespace endocrine_cells_synthesize_both_l910_91078

structure Cell :=
  (canSynthesizeEnzymes : Bool)
  (canSynthesizeHormones : Bool)

structure Hormone :=
  (producedByEndocrine : Bool)
  (directlyParticipateInCells : Bool)

structure Enzyme :=
  (producedByLivingCells : Bool)

def EndocrineCell := {c : Cell // c.canSynthesizeHormones = true}

theorem endocrine_cells_synthesize_both :
  ∀ (h : Hormone) (e : Enzyme) (ec : EndocrineCell),
    h.directlyParticipateInCells = false →
    e.producedByLivingCells = true →
    h.producedByEndocrine = true →
    ec.val.canSynthesizeEnzymes = true ∧ ec.val.canSynthesizeHormones = true :=
by sorry

end endocrine_cells_synthesize_both_l910_91078


namespace cat_groupings_count_l910_91096

/-- The number of ways to divide 12 cats into groups of 4, 6, and 2,
    with Whiskers in the 4-cat group and Paws in the 6-cat group. -/
def cat_groupings : ℕ :=
  Nat.choose 10 3 * Nat.choose 7 5

theorem cat_groupings_count : cat_groupings = 2520 := by
  sorry

end cat_groupings_count_l910_91096


namespace value_calculation_l910_91038

theorem value_calculation (number : ℕ) (value : ℕ) : 
  number = 48 → value = (number / 4 + 15) → value = 27 := by sorry

end value_calculation_l910_91038


namespace polynomial_evaluation_l910_91064

theorem polynomial_evaluation (w x y z : ℝ) 
  (eq1 : w + x + y + z = 5)
  (eq2 : 2*w + 4*x + 8*y + 16*z = 7)
  (eq3 : 3*w + 9*x + 27*y + 81*z = 11)
  (eq4 : 4*w + 16*x + 64*y + 256*z = 1) :
  5*w + 25*x + 125*y + 625*z = -60 := by
  sorry

end polynomial_evaluation_l910_91064


namespace average_after_12th_innings_l910_91099

/-- Represents a batsman's performance -/
structure Batsman where
  innings : ℕ
  lastScore : ℕ
  averageIncrease : ℕ
  neverNotOut : Bool

/-- Calculates the average score after the latest innings -/
def calculateAverage (b : Batsman) : ℚ :=
  if b.innings = 0 then 0
  else (b.innings * (b.averageIncrease : ℚ) + b.lastScore) / b.innings

/-- Theorem stating the average after 12th innings -/
theorem average_after_12th_innings (b : Batsman) 
  (h1 : b.innings = 12)
  (h2 : b.lastScore = 55)
  (h3 : b.averageIncrease = 1)
  (h4 : b.neverNotOut = true) :
  calculateAverage b = 44 := by
  sorry


end average_after_12th_innings_l910_91099


namespace cloth_sold_meters_l910_91071

/-- Proves that the number of meters of cloth sold is 85 given the total selling price,
    profit per meter, and cost price per meter. -/
theorem cloth_sold_meters (total_selling_price : ℕ) (profit_per_meter : ℕ) (cost_price_per_meter : ℕ)
    (h1 : total_selling_price = 8925)
    (h2 : profit_per_meter = 25)
    (h3 : cost_price_per_meter = 80) :
    (total_selling_price / (cost_price_per_meter + profit_per_meter) : ℕ) = 85 := by
  sorry

#eval (8925 / (80 + 25) : ℕ)  -- Should output 85

end cloth_sold_meters_l910_91071


namespace max_distance_is_three_l910_91080

/-- A figure constructed from an equilateral triangle with semicircles on each side -/
structure TriangleWithSemicircles where
  /-- Side length of the equilateral triangle -/
  triangleSide : ℝ
  /-- Radius of the semicircles -/
  semicircleRadius : ℝ

/-- The maximum distance between any two points on the boundary of the figure -/
def maxBoundaryDistance (figure : TriangleWithSemicircles) : ℝ :=
  figure.triangleSide + 2 * figure.semicircleRadius

/-- Theorem stating the maximum distance for the specific figure described in the problem -/
theorem max_distance_is_three :
  let figure : TriangleWithSemicircles := ⟨2, 1⟩
  maxBoundaryDistance figure = 3 := by
  sorry


end max_distance_is_three_l910_91080


namespace mock_exam_girls_count_l910_91014

theorem mock_exam_girls_count :
  ∀ (total_students : ℕ) (boys girls : ℕ) (boys_cleared girls_cleared : ℕ),
    total_students = 400 →
    boys + girls = total_students →
    boys_cleared = (60 * boys) / 100 →
    girls_cleared = (80 * girls) / 100 →
    boys_cleared + girls_cleared = (65 * total_students) / 100 →
    girls = 100 := by
  sorry

end mock_exam_girls_count_l910_91014


namespace two_numbers_difference_l910_91018

theorem two_numbers_difference (x y : ℝ) : 
  x + y = 30 → 2 * y - 3 * x = 5 → |y - x| = 8 := by
  sorry

end two_numbers_difference_l910_91018


namespace pirate_coin_sharing_l910_91011

/-- The number of coins Pete gives himself in the final round -/
def x : ℕ := 9

/-- The total number of coins Pete has at the end -/
def petes_coins (x : ℕ) : ℕ := x * (x + 1) / 2

/-- The total number of coins Paul has at the end -/
def pauls_coins (x : ℕ) : ℕ := x

/-- The condition that Pete has 5 times as many coins as Paul -/
def pete_five_times_paul (x : ℕ) : Prop :=
  petes_coins x = 5 * pauls_coins x

/-- The total number of coins shared -/
def total_coins (x : ℕ) : ℕ := petes_coins x + pauls_coins x

theorem pirate_coin_sharing :
  pete_five_times_paul x ∧ total_coins x = 54 := by
  sorry

end pirate_coin_sharing_l910_91011


namespace twenty_player_tournament_games_l910_91043

/-- Calculates the number of games in a chess tournament --/
def chess_tournament_games (n : ℕ) : ℕ :=
  n * (n - 1)

/-- Theorem: In a chess tournament with 20 players, where each player plays twice with every other player, 
    the total number of games played is 760. --/
theorem twenty_player_tournament_games : 
  chess_tournament_games 20 * 2 = 760 := by
  sorry

end twenty_player_tournament_games_l910_91043


namespace shortest_leg_of_smallest_triangle_l910_91079

/-- Represents a 30-60-90 triangle -/
structure Triangle30_60_90 where
  short_leg : ℝ
  long_leg : ℝ
  hypotenuse : ℝ
  short_leg_prop : short_leg = hypotenuse / 2
  long_leg_prop : long_leg = short_leg * Real.sqrt 3

/-- Represents a series of three 30-60-90 triangles -/
structure TriangleSeries where
  large : Triangle30_60_90
  medium : Triangle30_60_90
  small : Triangle30_60_90
  large_medium_relation : large.short_leg = medium.hypotenuse
  medium_small_relation : medium.short_leg = small.hypotenuse
  largest_hypotenuse : large.hypotenuse = 12

theorem shortest_leg_of_smallest_triangle (series : TriangleSeries) :
  series.small.short_leg = 1.5 := by sorry

end shortest_leg_of_smallest_triangle_l910_91079


namespace sequence_convergence_l910_91020

theorem sequence_convergence (a : ℕ → ℚ) :
  a 1 = 3 / 5 →
  (∀ n : ℕ, a (n + 1) = 2 - 1 / (a n)) →
  a 2018 = 4031 / 4029 := by
sorry

end sequence_convergence_l910_91020


namespace complex_multiplication_l910_91019

theorem complex_multiplication (i : ℂ) (z₁ z₂ : ℂ) :
  i * i = -1 →
  z₁ = 1 + 2 * i →
  z₂ = -3 * i →
  z₁ * z₂ = 6 - 3 * i :=
by sorry

end complex_multiplication_l910_91019


namespace marys_cake_recipe_l910_91067

/-- Mary's cake recipe problem -/
theorem marys_cake_recipe 
  (total_flour : ℕ) 
  (sugar : ℕ) 
  (flour_to_add : ℕ) 
  (h1 : total_flour = 9)
  (h2 : flour_to_add = sugar + 1)
  (h3 : sugar = 6) :
  total_flour - flour_to_add = 2 := by
  sorry

end marys_cake_recipe_l910_91067


namespace cannot_reach_all_same_l910_91047

/-- Represents the state of the circle of numbers -/
structure CircleState where
  ones : Nat
  zeros : Nat
  deriving Repr

/-- The operation performed on the circle each second -/
def next_state (s : CircleState) : CircleState :=
  sorry

/-- Predicate to check if all numbers in the circle are the same -/
def all_same (s : CircleState) : Prop :=
  s.ones = 0 ∨ s.zeros = 0

/-- The initial state of the circle -/
def initial_state : CircleState :=
  { ones := 4, zeros := 5 }

/-- Theorem stating that it's impossible to reach a state where all numbers are the same -/
theorem cannot_reach_all_same :
  ¬ ∃ (n : Nat), all_same (n.iterate next_state initial_state) :=
sorry

end cannot_reach_all_same_l910_91047


namespace inequality_holds_iff_b_greater_than_one_l910_91021

theorem inequality_holds_iff_b_greater_than_one (b : ℝ) (h : b > 0) :
  (∃ x : ℝ, |x - 2| + |x - 1| < b) ↔ b > 1 := by
  sorry

end inequality_holds_iff_b_greater_than_one_l910_91021


namespace sqrt_sum_fraction_simplification_l910_91010

theorem sqrt_sum_fraction_simplification :
  Real.sqrt ((36 : ℝ) / 49 + 16 / 9 + 1 / 16) = 45 / 28 := by
  sorry

end sqrt_sum_fraction_simplification_l910_91010


namespace circle_area_from_circumference_l910_91095

theorem circle_area_from_circumference (C : ℝ) (h : C = 36) :
  let r := C / (2 * Real.pi)
  (Real.pi * r^2) = 324 / Real.pi := by sorry

end circle_area_from_circumference_l910_91095


namespace pecan_amount_correct_l910_91028

/-- Represents the composition of a nut mixture -/
structure NutMixture where
  pecan_pounds : ℝ
  cashew_pounds : ℝ
  pecan_price : ℝ
  mixture_price : ℝ

/-- Verifies if a given nut mixture satisfies the problem conditions -/
def is_valid_mixture (m : NutMixture) : Prop :=
  m.cashew_pounds = 2 ∧
  m.pecan_price = 5.60 ∧
  m.mixture_price = 4.34

/-- Calculates the total value of the mixture -/
def mixture_value (m : NutMixture) : ℝ :=
  (m.pecan_pounds + m.cashew_pounds) * m.mixture_price

/-- Calculates the value of pecans in the mixture -/
def pecan_value (m : NutMixture) : ℝ :=
  m.pecan_pounds * m.pecan_price

/-- The main theorem stating that the mixture with 1.33333333333 pounds of pecans
    satisfies the problem conditions -/
theorem pecan_amount_correct (m : NutMixture) 
  (h_valid : is_valid_mixture m)
  (h_pecan : m.pecan_pounds = 1.33333333333) :
  mixture_value m = pecan_value m + m.cashew_pounds * (mixture_value m / (m.pecan_pounds + m.cashew_pounds)) :=
by
  sorry


end pecan_amount_correct_l910_91028


namespace jeans_wednesday_calls_l910_91039

/-- Represents the number of calls Jean answered each day of the week --/
structure WeekCalls where
  monday : ℕ
  tuesday : ℕ
  wednesday : ℕ
  thursday : ℕ
  friday : ℕ

/-- The average number of calls per day --/
def average_calls : ℕ := 40

/-- The number of working days --/
def working_days : ℕ := 5

/-- Calculates the total number of calls in a week --/
def total_calls (w : WeekCalls) : ℕ :=
  w.monday + w.tuesday + w.wednesday + w.thursday + w.friday

/-- Jean's calls for the week --/
def jeans_calls : WeekCalls := {
  monday := 35,
  tuesday := 46,
  wednesday := 27,  -- This is what we want to prove
  thursday := 61,
  friday := 31
}

/-- Theorem stating that Jean answered 27 calls on Wednesday --/
theorem jeans_wednesday_calls :
  jeans_calls.wednesday = 27 ∧
  total_calls jeans_calls = average_calls * working_days :=
sorry

end jeans_wednesday_calls_l910_91039


namespace sum_sqrt_inequality_l910_91068

theorem sum_sqrt_inequality (a b c : ℝ) 
  (h1 : a + b + c = 1) 
  (h2 : 0 ≤ a) (h3 : 0 ≤ b) (h4 : 0 ≤ c) : 
  Real.sqrt a + Real.sqrt b + Real.sqrt c ≤ Real.sqrt 3 := by
  sorry

end sum_sqrt_inequality_l910_91068


namespace unique_years_arithmetic_sequence_l910_91062

/-- A year in the 19th century -/
structure Year19thCentury where
  x : Nat
  y : Nat
  x_range : x ≤ 9
  y_range : y ≤ 9

/-- Check if the differences between adjacent digits form an arithmetic sequence -/
def isArithmeticSequence (year : Year19thCentury) : Prop :=
  ∃ d : Int, (year.x - 8 : Int) - 7 = d ∧ (year.y - year.x : Int) - (year.x - 8) = d

/-- The theorem stating that 1881 and 1894 are the only years satisfying the condition -/
theorem unique_years_arithmetic_sequence :
  ∀ year : Year19thCentury, isArithmeticSequence year ↔ (year.x = 8 ∧ year.y = 1) ∨ (year.x = 9 ∧ year.y = 4) :=
by sorry

end unique_years_arithmetic_sequence_l910_91062


namespace f_max_value_l910_91083

/-- The function f(x) = 6x - 2x^2 -/
def f (x : ℝ) := 6 * x - 2 * x^2

/-- The maximum value of f(x) is 9/2 -/
theorem f_max_value : ∃ (M : ℝ), M = 9/2 ∧ ∀ (x : ℝ), f x ≤ M :=
sorry

end f_max_value_l910_91083


namespace power_of_two_triples_l910_91087

def is_power_of_two (n : ℕ) : Prop := ∃ k : ℕ, n = 2^k

def satisfies_condition (a b c : ℕ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧
  is_power_of_two (a * b - c) ∧
  is_power_of_two (b * c - a) ∧
  is_power_of_two (c * a - b)

theorem power_of_two_triples :
  ∀ a b c : ℕ, satisfies_condition a b c ↔
    ((a, b, c) = (2, 2, 2) ∨
     (a, b, c) = (2, 2, 3) ∨
     (a, b, c) = (3, 5, 7) ∨
     (a, b, c) = (2, 6, 11)) :=
by sorry

end power_of_two_triples_l910_91087


namespace negative_cube_squared_l910_91053

theorem negative_cube_squared (a : ℝ) : (-a^3)^2 = a^6 := by
  sorry

end negative_cube_squared_l910_91053


namespace two_digit_number_theorem_l910_91091

/-- Represents a two-digit number with specific properties -/
structure TwoDigitNumber where
  x : ℕ  -- tens digit
  -- Ensure x is a single digit
  h1 : x ≥ 1 ∧ x ≤ 9
  -- Ensure the units digit is non-negative
  h2 : 2 * x ≥ 3

/-- The value of the two-digit number -/
def TwoDigitNumber.value (n : TwoDigitNumber) : ℕ :=
  10 * n.x + (2 * n.x - 3)

theorem two_digit_number_theorem (n : TwoDigitNumber) :
  n.value = 12 * n.x - 3 := by
  sorry

end two_digit_number_theorem_l910_91091


namespace house_occupancy_l910_91058

/-- The number of people in the house given specific room occupancies. -/
def people_in_house (bedroom living_room kitchen garage patio : ℕ) : ℕ :=
  bedroom + living_room + kitchen + garage + patio

/-- The problem statement as a theorem. -/
theorem house_occupancy : ∃ (bedroom living_room kitchen garage patio : ℕ),
  bedroom = 7 ∧
  living_room = 8 ∧
  kitchen = living_room + 3 ∧
  garage * 2 = kitchen ∧
  patio = garage * 2 ∧
  people_in_house bedroom living_room kitchen garage patio = 41 := by
  sorry

end house_occupancy_l910_91058


namespace brochure_calculation_l910_91057

/-- Calculates the number of brochures created by a printing press given specific conditions -/
theorem brochure_calculation (single_page_spreads : ℕ) 
  (h1 : single_page_spreads = 20)
  (h2 : ∀ n : ℕ, n = single_page_spreads → 2 * n = number_of_double_page_spreads)
  (h3 : ∀ n : ℕ, n = total_spread_pages → n / 4 = number_of_ad_blocks)
  (h4 : ∀ n : ℕ, n = number_of_ad_blocks → 4 * n = total_ads)
  (h5 : ∀ n : ℕ, n = total_ads → n / 4 = ad_pages)
  (h6 : ∀ n : ℕ, n = total_pages → n / 5 = number_of_brochures)
  : number_of_brochures = 25 := by
  sorry

#check brochure_calculation

end brochure_calculation_l910_91057


namespace ship_storm_problem_l910_91094

/-- A problem about a ship's journey and a storm -/
theorem ship_storm_problem (initial_speed : ℝ) (initial_time : ℝ) 
  (h1 : initial_speed = 30)
  (h2 : initial_time = 20)
  (h3 : initial_speed * initial_time = (1/2) * (total_distance : ℝ))
  (h4 : distance_after_storm = (1/3) * total_distance) : 
  initial_speed * initial_time - distance_after_storm = 200 := by
  sorry

#check ship_storm_problem

end ship_storm_problem_l910_91094


namespace quadratic_factoring_l910_91031

theorem quadratic_factoring (x : ℝ) : x^2 - 2*x - 2 = 0 ↔ (x - 1)^2 = 3 := by
  sorry

end quadratic_factoring_l910_91031


namespace specific_ellipse_major_axis_l910_91009

/-- An ellipse with specific properties -/
structure Ellipse where
  -- The ellipse is tangent to both x-axis and y-axis
  tangent_to_axes : Bool
  -- The x-coordinate of both foci
  focus_x : ℝ
  -- The y-coordinates of the foci
  focus_y1 : ℝ
  focus_y2 : ℝ

/-- The length of the major axis of the ellipse -/
def major_axis_length (e : Ellipse) : ℝ :=
  sorry

/-- Theorem stating the length of the major axis for a specific ellipse -/
theorem specific_ellipse_major_axis :
  ∃ (e : Ellipse), 
    e.tangent_to_axes = true ∧
    e.focus_x = 3 ∧
    e.focus_y1 = -4 + 2 * Real.sqrt 2 ∧
    e.focus_y2 = -4 - 2 * Real.sqrt 2 ∧
    major_axis_length e = 8 :=
  sorry

end specific_ellipse_major_axis_l910_91009


namespace fourth_power_difference_l910_91030

theorem fourth_power_difference (a b : ℝ) : 
  (a - b)^4 = a^4 - 4*a^3*b + 6*a^2*b^2 - 4*a*b^3 + b^4 :=
by
  sorry

end fourth_power_difference_l910_91030


namespace only_b_q_rotationally_symmetric_l910_91082

/-- Represents an English letter -/
inductive Letter
| B
| D
| P
| Q

/-- Defines rotational symmetry between two letters -/
def rotationallySymmetric (l1 l2 : Letter) : Prop :=
  match l1, l2 with
  | Letter.B, Letter.Q => True
  | Letter.Q, Letter.B => True
  | _, _ => False

/-- Theorem stating that only B and Q are rotationally symmetric -/
theorem only_b_q_rotationally_symmetric :
  ∀ (l1 l2 : Letter),
    rotationallySymmetric l1 l2 ↔ (l1 = Letter.B ∧ l2 = Letter.Q) ∨ (l1 = Letter.Q ∧ l2 = Letter.B) :=
by sorry

#check only_b_q_rotationally_symmetric

end only_b_q_rotationally_symmetric_l910_91082


namespace intersection_P_Q_l910_91054

def P : Set ℤ := {x | (x - 3) * (x - 6) ≤ 0}
def Q : Set ℤ := {5, 7}

theorem intersection_P_Q : P ∩ Q = {5} := by sorry

end intersection_P_Q_l910_91054


namespace initial_crayons_l910_91048

theorem initial_crayons (taken_out : ℕ) (left : ℕ) : 
  taken_out = 3 → left = 4 → taken_out + left = 7 :=
by sorry

end initial_crayons_l910_91048


namespace initial_mixture_volume_l910_91066

/-- Given a mixture of milk and water with an initial ratio of 4:1,
    adding 3 litres of water results in a new ratio of 3:1.
    This theorem proves that the initial volume of the mixture was 45 litres. -/
theorem initial_mixture_volume
  (initial_milk : ℝ)
  (initial_water : ℝ)
  (h1 : initial_milk / initial_water = 4)
  (h2 : initial_milk / (initial_water + 3) = 3) :
  initial_milk + initial_water = 45 := by
sorry

end initial_mixture_volume_l910_91066


namespace star_three_four_l910_91044

def star (x y : ℝ) : ℝ := 4 * x + 6 * y

theorem star_three_four : star 3 4 = 36 := by
  sorry

end star_three_four_l910_91044


namespace least_addition_to_perfect_square_l910_91076

theorem least_addition_to_perfect_square : ∃ (x : ℝ), 
  (x ≥ 0) ∧ 
  (∃ (n : ℕ), (0.0320 + x) = n^2) ∧
  (∀ (y : ℝ), y ≥ 0 → (∃ (m : ℕ), (0.0320 + y) = m^2) → y ≥ x) ∧
  (x = 0.9680) := by
sorry

end least_addition_to_perfect_square_l910_91076


namespace four_pencils_per_child_l910_91088

/-- Given a group of children and pencils, calculate the number of pencils per child. -/
def pencils_per_child (num_children : ℕ) (total_pencils : ℕ) : ℕ :=
  total_pencils / num_children

/-- Theorem stating that with 8 children and 32 pencils, each child has 4 pencils. -/
theorem four_pencils_per_child :
  pencils_per_child 8 32 = 4 := by
  sorry

#eval pencils_per_child 8 32

end four_pencils_per_child_l910_91088


namespace sum_of_fractions_geq_one_l910_91007

theorem sum_of_fractions_geq_one (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  a / (a + 2*b) + b / (b + 2*c) + c / (c + 2*a) ≥ 1 := by
  sorry

end sum_of_fractions_geq_one_l910_91007


namespace largest_multiple_of_18_with_8_and_0_l910_91050

def is_valid_number (n : ℕ) : Prop :=
  ∀ d, d ∈ n.digits 10 → d = 8 ∨ d = 0

theorem largest_multiple_of_18_with_8_and_0 :
  ∃ m : ℕ,
    m > 0 ∧
    m % 18 = 0 ∧
    is_valid_number m ∧
    (∀ k : ℕ, k > m → k % 18 = 0 → ¬is_valid_number k) ∧
    m / 18 = 493826048 :=
sorry

end largest_multiple_of_18_with_8_and_0_l910_91050


namespace min_value_quadratic_l910_91032

theorem min_value_quadratic (x : ℝ) (y : ℝ) :
  x ∈ Set.Icc (-1 : ℝ) (1 : ℝ) →
  y = 2 * x^2 - 6 * x + 3 →
  ∃ (m : ℝ), m = -1 ∧ ∀ z ∈ Set.Icc (-1 : ℝ) (1 : ℝ), 2 * z^2 - 6 * z + 3 ≥ m :=
by sorry

end min_value_quadratic_l910_91032


namespace larger_box_capacity_l910_91002

/-- Represents the dimensions of a box -/
structure BoxDimensions where
  height : ℝ
  width : ℝ
  length : ℝ

/-- Calculates the volume of a box given its dimensions -/
def boxVolume (d : BoxDimensions) : ℝ :=
  d.height * d.width * d.length

/-- The capacity of clay a box can carry -/
def boxCapacity (volume : ℝ) (clayPerUnit : ℝ) : ℝ :=
  volume * clayPerUnit

theorem larger_box_capacity 
  (small_box : BoxDimensions)
  (small_box_clay : ℝ)
  (h_small_height : small_box.height = 1)
  (h_small_width : small_box.width = 2)
  (h_small_length : small_box.length = 4)
  (h_small_capacity : small_box_clay = 30) :
  let large_box : BoxDimensions := {
    height := 3 * small_box.height,
    width := 2 * small_box.width,
    length := 2 * small_box.length
  }
  let small_volume := boxVolume small_box
  let large_volume := boxVolume large_box
  let clay_per_unit := small_box_clay / small_volume
  boxCapacity large_volume clay_per_unit = 360 := by
sorry

end larger_box_capacity_l910_91002


namespace fraction_equality_implies_numerator_equality_l910_91022

theorem fraction_equality_implies_numerator_equality
  (a b c : ℝ) (h1 : c ≠ 0) (h2 : a / c = b / c) : a = b := by
  sorry

end fraction_equality_implies_numerator_equality_l910_91022


namespace candies_eaten_l910_91052

theorem candies_eaten (initial : ℕ) (remaining : ℕ) (eaten : ℕ) : 
  initial = 23 → remaining = 7 → eaten = initial - remaining → eaten = 16 := by sorry

end candies_eaten_l910_91052


namespace digit_58_is_8_l910_91037

/-- The decimal representation of 1/7 -/
def decimal_rep_1_7 : ℕ → ℕ
| 0 => 1
| 1 => 4
| 2 => 2
| 3 => 8
| 4 => 5
| 5 => 7
| n + 6 => decimal_rep_1_7 n

/-- The period of the decimal representation of 1/7 -/
def period : ℕ := 6

/-- The 58th digit after the decimal point in the decimal representation of 1/7 -/
def digit_58 : ℕ := decimal_rep_1_7 ((58 - 1) % period)

theorem digit_58_is_8 : digit_58 = 8 := by sorry

end digit_58_is_8_l910_91037


namespace bajazet_winning_strategy_l910_91065

-- Define a polynomial of degree 4
def polynomial (a b c : ℝ) (x : ℝ) : ℝ := x^4 + a*x^3 + b*x^2 + c*x + 1

-- State the theorem
theorem bajazet_winning_strategy :
  ∀ (a b c : ℝ), ∃ (x : ℝ), polynomial a b c x = 0 :=
by
  sorry


end bajazet_winning_strategy_l910_91065


namespace largest_valid_number_l910_91069

def is_valid_number (n : ℕ) : Prop :=
  ∃ (a b c d e f : ℕ),
    n = a * 100000 + b * 10000 + c * 1000 + d * 100 + e * 10 + f ∧
    a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ a ≠ e ∧ a ≠ f ∧
    b ≠ c ∧ b ≠ d ∧ b ≠ e ∧ b ≠ f ∧
    c ≠ d ∧ c ≠ e ∧ c ≠ f ∧
    d ≠ e ∧ d ≠ f ∧
    e ≠ f ∧
    (c = a + b ∨ c = a - b ∨ c = b - a) ∧
    (d = b + c ∨ d = b - c ∨ d = c - b) ∧
    (e = c + d ∨ e = c - d ∨ e = d - c) ∧
    0 ≤ a ∧ a ≤ 9 ∧
    0 ≤ b ∧ b ≤ 9 ∧
    0 ≤ c ∧ c ≤ 9 ∧
    0 ≤ d ∧ d ≤ 9 ∧
    0 ≤ e ∧ e ≤ 9 ∧
    0 ≤ f ∧ f ≤ 9

theorem largest_valid_number :
  ∀ n : ℕ, is_valid_number n → n ≤ 972538 :=
by sorry

end largest_valid_number_l910_91069


namespace tom_brick_cost_l910_91063

/-- The total cost for Tom's bricks -/
def total_cost (total_bricks : ℕ) (original_price : ℚ) (discount_percent : ℚ) : ℚ :=
  let discounted_bricks := total_bricks / 2
  let full_price_bricks := total_bricks - discounted_bricks
  let discounted_price := original_price * (1 - discount_percent)
  (discounted_bricks : ℚ) * discounted_price + (full_price_bricks : ℚ) * original_price

/-- Theorem stating that the total cost for Tom's bricks is $375 -/
theorem tom_brick_cost :
  total_cost 1000 (1/2) (1/2) = 375 := by
  sorry

end tom_brick_cost_l910_91063


namespace hyperbola_triangle_l910_91005

-- Define the hyperbola
def hyperbola (x y : ℝ) : Prop := x * y = 1

-- Define the branches of the hyperbola
def C₁ (x y : ℝ) : Prop := hyperbola x y ∧ x > 0
def C₂ (x y : ℝ) : Prop := hyperbola x y ∧ x < 0

-- Define a regular triangle
def regular_triangle (P Q R : ℝ × ℝ) : Prop :=
  let (px, py) := P
  let (qx, qy) := Q
  let (rx, ry) := R
  (px - qx)^2 + (py - qy)^2 = (qx - rx)^2 + (qy - ry)^2 ∧
  (qx - rx)^2 + (qy - ry)^2 = (rx - px)^2 + (ry - py)^2

-- Theorem statement
theorem hyperbola_triangle :
  ∀ (Q R : ℝ × ℝ),
  let P := (-1, -1)
  regular_triangle P Q R ∧
  C₂ P.1 P.2 ∧
  C₁ Q.1 Q.2 ∧
  C₁ R.1 R.2 →
  (¬(C₁ P.1 P.2 ∧ C₁ Q.1 Q.2 ∧ C₁ R.1 R.2) ∧
   ¬(C₂ P.1 P.2 ∧ C₂ Q.1 Q.2 ∧ C₂ R.1 R.2)) ∧
  Q = (2 - Real.sqrt 3, 2 + Real.sqrt 3) ∧
  R = (2 + Real.sqrt 3, 2 - Real.sqrt 3) :=
by sorry

end hyperbola_triangle_l910_91005


namespace inequality_proof_l910_91004

theorem inequality_proof (x y z : ℝ) :
  -3/2 * (x^2 + y^2 + 2*z^2) ≤ 3*x*y + y*z + z*x ∧
  3*x*y + y*z + z*x ≤ (3 + Real.sqrt 13)/4 * (x^2 + y^2 + 2*z^2) := by
  sorry

end inequality_proof_l910_91004


namespace isosceles_triangle_condition_l910_91017

/-- Given a triangle ABC with sides a, b, c opposite to angles A, B, C respectively,
    if a⋅cos B = b⋅cos A, then the triangle is isosceles with A = B -/
theorem isosceles_triangle_condition (A B C : ℝ) (a b c : ℝ) :
  A + B + C = π →
  a > 0 → b > 0 → c > 0 →
  a * Real.cos B = b * Real.cos A →
  A = B :=
sorry

end isosceles_triangle_condition_l910_91017


namespace count_even_multiples_of_three_squares_l910_91012

theorem count_even_multiples_of_three_squares (n : Nat) : 
  (∃ k, k ∈ Finset.range n ∧ 36 * k * k < 3000) ↔ n = 10 :=
sorry

end count_even_multiples_of_three_squares_l910_91012


namespace P_intersect_Q_equals_target_l910_91026

-- Define the sets P and Q
def P : Set ℝ := {x | ∃ y, y = Real.log (2 - x)}
def Q : Set ℝ := {x | x^2 - 5*x + 4 ≤ 0}

-- State the theorem
theorem P_intersect_Q_equals_target : P ∩ Q = {x | 1 ≤ x ∧ x < 2} := by sorry

end P_intersect_Q_equals_target_l910_91026


namespace shirt_ratio_l910_91006

/-- Given that Hazel received 6 shirts and the total number of shirts is 18,
    prove that the ratio of Razel's shirts to Hazel's shirts is 2:1. -/
theorem shirt_ratio (hazel_shirts : ℕ) (total_shirts : ℕ) (razel_shirts : ℕ) : 
  hazel_shirts = 6 → total_shirts = 18 → razel_shirts = total_shirts - hazel_shirts →
  (razel_shirts : ℚ) / hazel_shirts = 2 / 1 := by
  sorry

end shirt_ratio_l910_91006


namespace average_age_when_youngest_born_l910_91084

/-- Proves that given a group of 7 people with an average age of 50 years,
    where the youngest is 5 years old, the average age of the remaining 6 people
    5 years ago was 57.5 years. -/
theorem average_age_when_youngest_born
  (total_people : ℕ)
  (average_age : ℝ)
  (youngest_age : ℝ)
  (total_age : ℝ)
  (h1 : total_people = 7)
  (h2 : average_age = 50)
  (h3 : youngest_age = 5)
  (h4 : total_age = average_age * total_people)
  : (total_age - youngest_age) / (total_people - 1) = 57.5 := by
  sorry

#check average_age_when_youngest_born

end average_age_when_youngest_born_l910_91084


namespace min_value_of_u_l910_91061

theorem min_value_of_u (x y : ℝ) (hx : x ≥ 0) (hy : y ≥ 0) (h : 2 * x + y = 6) :
  ∃ (min_u : ℝ), min_u = 27 / 2 ∧ ∀ (u : ℝ), u = 4 * x^2 + 3 * x * y + y^2 - 6 * x - 3 * y → u ≥ min_u :=
sorry

end min_value_of_u_l910_91061


namespace complex_inequality_l910_91033

open Complex

theorem complex_inequality (x y : ℂ) (z : ℂ) 
  (h1 : abs x = 1) (h2 : abs y = 1)
  (h3 : π / 3 ≤ arg x - arg y) (h4 : arg x - arg y ≤ 5 * π / 3) :
  abs z + abs (z - x) + abs (z - y) ≥ abs (z * x - y) := by
  sorry

end complex_inequality_l910_91033


namespace quadratic_real_roots_k_range_l910_91040

theorem quadratic_real_roots_k_range (k : ℝ) : 
  (∃ x : ℝ, 2 * x^2 + 4 * x + k - 1 = 0) → k ≤ 3 :=
by sorry

end quadratic_real_roots_k_range_l910_91040


namespace units_digit_periodicity_l910_91077

theorem units_digit_periodicity (k : ℕ) : 
  (k * (k + 1) * (k + 2)) % 10 = ((k + 10) * (k + 11) * (k + 12)) % 10 := by
  sorry

end units_digit_periodicity_l910_91077


namespace smallest_solution_of_equation_l910_91049

theorem smallest_solution_of_equation (x : ℝ) :
  (3 * x^2 + 33 * x - 90 = x * (x + 15)) →
  x ≥ -15 :=
by
  sorry

end smallest_solution_of_equation_l910_91049


namespace rotation_90_clockwise_l910_91029

-- Define the possible positions in the circle
inductive Position
  | Top
  | Left
  | Right

-- Define the shapes
inductive Shape
  | Pentagon
  | SmallerCircle
  | Rectangle

-- Define a function to represent the initial configuration
def initial_config : Position → Shape
  | Position.Top => Shape.Pentagon
  | Position.Left => Shape.SmallerCircle
  | Position.Right => Shape.Rectangle

-- Define a function to represent the configuration after 90° clockwise rotation
def rotated_config : Position → Shape
  | Position.Top => Shape.SmallerCircle
  | Position.Right => Shape.Pentagon
  | Position.Left => Shape.Rectangle

-- Theorem stating that the rotated configuration is correct
theorem rotation_90_clockwise :
  ∀ p : Position, rotated_config p = initial_config (match p with
    | Position.Top => Position.Right
    | Position.Right => Position.Left
    | Position.Left => Position.Top
  ) :=
by sorry

end rotation_90_clockwise_l910_91029


namespace problem_solution_l910_91024

theorem problem_solution (a b c d : ℝ) 
  (h1 : a < b ∧ b < d)
  (h2 : ∀ x, (x - a) * (x - b) * (x - d) / (x - c) ≥ 0 ↔ x ≤ -7 ∨ (30 ≤ x ∧ x ≤ 32)) :
  a + 2*b + 3*c + 4*d = 160 := by
sorry

end problem_solution_l910_91024


namespace cara_seating_arrangements_l910_91001

/-- The number of people at the circular table, including Cara -/
def total_people : ℕ := 7

/-- The number of Cara's friends -/
def num_friends : ℕ := 6

/-- Alex is one of Cara's friends -/
def alex_is_friend : Prop := true

/-- The number of different pairs Cara could be sitting between, where one must be Alex -/
def num_seating_arrangements : ℕ := 5

theorem cara_seating_arrangements :
  total_people = num_friends + 1 →
  alex_is_friend →
  num_seating_arrangements = num_friends - 1 :=
by
  sorry

end cara_seating_arrangements_l910_91001


namespace negation_equivalence_l910_91003

theorem negation_equivalence :
  (¬ ∃ x : ℝ, x^2 - 4*x + 2 > 0) ↔ (∀ x : ℝ, x^2 - 4*x + 2 ≤ 0) :=
by sorry

end negation_equivalence_l910_91003


namespace plane_relationships_l910_91070

-- Define the types for planes and lines
variable (Plane Line : Type)

-- Define the relationships between planes and lines
variable (in_plane : Line → Plane → Prop)
variable (perpendicular : Line → Line → Prop)
variable (plane_perpendicular : Plane → Plane → Prop)
variable (line_perpendicular_to_plane : Line → Plane → Prop)
variable (line_parallel_to_plane : Line → Plane → Prop)
variable (plane_parallel : Plane → Plane → Prop)

-- State the theorem
theorem plane_relationships 
  (α β : Plane) 
  (h_different : α ≠ β) :
  (∀ l : Line, in_plane l α → 
    (∀ m : Line, in_plane m β → perpendicular l m) → 
    plane_perpendicular α β) ∧
  ((∀ l : Line, in_plane l α → line_parallel_to_plane l β) → 
    plane_parallel α β) ∧
  (plane_parallel α β → 
    ∀ l : Line, in_plane l α → line_parallel_to_plane l β) :=
sorry

end plane_relationships_l910_91070


namespace cost_price_percentage_l910_91074

theorem cost_price_percentage (selling_price cost_price : ℝ) :
  selling_price > 0 →
  cost_price > 0 →
  selling_price - cost_price = (1 / 3) * cost_price →
  (cost_price / selling_price) * 100 = 75 := by
  sorry

end cost_price_percentage_l910_91074


namespace quadratic_real_solutions_l910_91013

theorem quadratic_real_solutions (a : ℝ) :
  (∃ x : ℝ, a * x^2 - 4*x - 1 = 0) ↔ a ≥ -4 := by
sorry

end quadratic_real_solutions_l910_91013


namespace perpendicular_lines_slope_l910_91008

theorem perpendicular_lines_slope (a : ℝ) : 
  (∀ x y : ℝ, ax + 2*y + 2 = 0 → x - y - 2 = 0 → 
   ((-a/2) * 1 = -1)) → a = 2 := by
  sorry

end perpendicular_lines_slope_l910_91008


namespace problem_1_problem_2_problem_3_problem_4_l910_91025

-- Problem 1
theorem problem_1 : 0.108 / 1.2 + 0.7 = 0.79 := by sorry

-- Problem 2
theorem problem_2 : (9.8 - 3.75) / 25 / 0.4 = 0.605 := by sorry

-- Problem 3
theorem problem_3 : 6.3 * 15 + 1/3 * 75/100 = 94.75 := by sorry

-- Problem 4
theorem problem_4 : 8 * 0.56 + 5.4 * 0.8 - 80/100 = 8 := by sorry

end problem_1_problem_2_problem_3_problem_4_l910_91025


namespace union_equals_A_l910_91056

-- Define set A
def A : Set ℝ := {x | x^2 - 2*x - 3 ≤ 0}

-- Define set B
def B : Set ℝ := {-1, 0, 1, 2, 3}

-- Theorem statement
theorem union_equals_A : A ∪ B = A := by
  sorry

end union_equals_A_l910_91056


namespace corn_ears_per_stalk_l910_91046

/-- The number of corn stalks -/
def num_stalks : ℕ := 108

/-- The number of kernels in half of the ears -/
def kernels_half1 : ℕ := 500

/-- The number of kernels in the other half of the ears -/
def kernels_half2 : ℕ := 600

/-- The total number of kernels -/
def total_kernels : ℕ := 237600

/-- The number of ears per stalk -/
def ears_per_stalk : ℕ := 4

theorem corn_ears_per_stalk :
  num_stalks * (ears_per_stalk / 2 * kernels_half1 + ears_per_stalk / 2 * kernels_half2) = total_kernels :=
by sorry

end corn_ears_per_stalk_l910_91046


namespace function_composition_l910_91092

theorem function_composition (f : ℝ → ℝ) (h : ∀ x > 0, f (3 * x) = 3 / (3 + x)) :
  ∀ x > 0, 2 * f x = 18 / (9 + x) := by
  sorry

end function_composition_l910_91092


namespace garage_sale_items_count_l910_91036

theorem garage_sale_items_count 
  (prices : Finset ℕ) 
  (radio_price : ℕ) 
  (h1 : radio_price ∈ prices) 
  (h2 : (prices.filter (λ x => x > radio_price)).card = 14) 
  (h3 : (prices.filter (λ x => x < radio_price)).card = 24) :
  prices.card = 39 :=
sorry

end garage_sale_items_count_l910_91036


namespace sqrt3_cos_minus_sin_eq_sqrt2_l910_91059

theorem sqrt3_cos_minus_sin_eq_sqrt2 :
  Real.sqrt 3 * Real.cos (π / 12) - Real.sin (π / 12) = Real.sqrt 2 := by
  sorry

end sqrt3_cos_minus_sin_eq_sqrt2_l910_91059


namespace mechanic_parts_cost_l910_91097

/-- A problem about calculating the cost of parts in a mechanic's bill -/
theorem mechanic_parts_cost
  (hourly_rate : ℝ)
  (job_duration : ℝ)
  (total_bill : ℝ)
  (h1 : hourly_rate = 45)
  (h2 : job_duration = 5)
  (h3 : total_bill = 450) :
  total_bill - hourly_rate * job_duration = 225 := by
  sorry

end mechanic_parts_cost_l910_91097


namespace polynomial_irreducibility_l910_91060

theorem polynomial_irreducibility (n : ℕ) (hn : n > 1) :
  let f : Polynomial ℤ := X^n + 5 * X^(n-1) + 3
  Irreducible f := by sorry

end polynomial_irreducibility_l910_91060


namespace brads_running_speed_l910_91093

/-- Proves that Brad's running speed is 6 km/h given the conditions of the problem -/
theorem brads_running_speed (maxwell_speed : ℝ) (total_distance : ℝ) (maxwell_time : ℝ) 
  (h1 : maxwell_speed = 4)
  (h2 : total_distance = 94)
  (h3 : maxwell_time = 10) : 
  let brad_time := maxwell_time - 1
  let maxwell_distance := maxwell_speed * maxwell_time
  let brad_distance := total_distance - maxwell_distance
  brad_distance / brad_time = 6 := by
  sorry

end brads_running_speed_l910_91093


namespace impossible_to_fill_board_l910_91015

/-- Represents a piece on the board -/
inductive Piece
  | Regular
  | Special

/-- Represents the color of a square -/
inductive Color
  | White
  | Grey

/-- Represents the board configuration -/
structure Board :=
  (rows : Nat)
  (cols : Nat)
  (total_squares : Nat)
  (white_squares : Nat)
  (grey_squares : Nat)

/-- Represents the coverage of a piece -/
structure PieceCoverage :=
  (white : Nat)
  (grey : Nat)

/-- The board configuration -/
def puzzle_board : Board :=
  { rows := 5
  , cols := 8
  , total_squares := 40
  , white_squares := 20
  , grey_squares := 20 }

/-- The coverage of a regular piece -/
def regular_coverage : PieceCoverage :=
  { white := 2, grey := 2 }

/-- The coverage of the special piece -/
def special_coverage : PieceCoverage :=
  { white := 3, grey := 1 }

/-- The theorem to be proved -/
theorem impossible_to_fill_board : 
  ∀ (special_piece_count : Nat) (regular_piece_count : Nat),
    special_piece_count = 1 →
    regular_piece_count = 9 →
    ¬ (special_piece_count * special_coverage.white + regular_piece_count * regular_coverage.white = puzzle_board.white_squares ∧
       special_piece_count * special_coverage.grey + regular_piece_count * regular_coverage.grey = puzzle_board.grey_squares) :=
by sorry

end impossible_to_fill_board_l910_91015


namespace garage_spokes_count_l910_91086

/-- The number of bicycles in the garage -/
def num_bicycles : ℕ := 4

/-- The number of wheels per bicycle -/
def wheels_per_bicycle : ℕ := 2

/-- The number of spokes per wheel -/
def spokes_per_wheel : ℕ := 10

/-- The total number of spokes in the garage -/
def total_spokes : ℕ := num_bicycles * wheels_per_bicycle * spokes_per_wheel

theorem garage_spokes_count : total_spokes = 80 := by
  sorry

end garage_spokes_count_l910_91086


namespace defined_implies_continuous_but_not_conversely_l910_91081

-- Define a function f from real numbers to real numbers
variable (f : ℝ → ℝ)
-- Define a point x₀
variable (x₀ : ℝ)

-- Statement: If f is defined at x₀, then f is continuous at x₀,
-- but the converse is not always true
theorem defined_implies_continuous_but_not_conversely :
  (∃ y, f x₀ = y) → ContinuousAt f x₀ ∧ 
  ¬(∀ g : ℝ → ℝ, ContinuousAt g x₀ → ∃ y, g x₀ = y) :=
sorry

end defined_implies_continuous_but_not_conversely_l910_91081


namespace five_students_three_villages_l910_91042

/-- The number of ways to assign n students to m villages with at least one student per village -/
def assignmentCount (n m : ℕ) : ℕ := sorry

/-- The number of ways to assign 5 students to 3 villages with at least one student per village -/
theorem five_students_three_villages : assignmentCount 5 3 = 150 := by sorry

end five_students_three_villages_l910_91042


namespace fixed_point_theorem_l910_91045

open Set

theorem fixed_point_theorem (f g : (Set.Icc 0 1) → (Set.Icc 0 1))
  (hf_cont : Continuous f)
  (hg_cont : Continuous g)
  (h_comm : ∀ x ∈ Set.Icc 0 1, f (g x) = g (f x))
  (hf_incr : StrictMono f) :
  ∃ a ∈ Set.Icc 0 1, f a = a ∧ g a = a := by
  sorry

end fixed_point_theorem_l910_91045


namespace abs_opposite_neg_six_l910_91051

theorem abs_opposite_neg_six : |-(- 6)| = 6 := by
  sorry

end abs_opposite_neg_six_l910_91051


namespace sum_of_squares_16_to_30_l910_91073

def sum_of_squares (n : ℕ) : ℕ := n * (n + 1) * (2 * n + 1) / 6

theorem sum_of_squares_16_to_30 :
  sum_of_squares 30 - sum_of_squares 15 = 8195 :=
by
  sorry

end sum_of_squares_16_to_30_l910_91073


namespace whitney_max_sets_l910_91023

/-- Represents the number of items Whitney has -/
structure Inventory where
  tshirts : ℕ
  buttons : ℕ
  stickers : ℕ

/-- Represents the composition of each set -/
structure SetComposition where
  tshirts : ℕ
  buttons : ℕ
  stickers : ℕ

def max_sets (inv : Inventory) (comp : SetComposition) : ℕ :=
  min (inv.tshirts / comp.tshirts)
      (min (inv.buttons / comp.buttons) (inv.stickers / comp.stickers))

/-- Theorem stating that the maximum number of sets Whitney can make is 5 -/
theorem whitney_max_sets :
  let inv : Inventory := { tshirts := 5, buttons := 24, stickers := 12 }
  let comp : SetComposition := { tshirts := 1, buttons := 2, stickers := 1 }
  max_sets inv comp = 5 := by
  sorry

end whitney_max_sets_l910_91023


namespace mask_production_optimization_l910_91000

/-- Represents the production plan for masks -/
structure MaskProduction where
  typeA : ℕ  -- Number of type A masks produced
  typeB : ℕ  -- Number of type B masks produced
  days : ℕ   -- Number of days used for production

/-- Checks if a production plan is valid according to the given conditions -/
def isValidProduction (p : MaskProduction) : Prop :=
  p.typeA + p.typeB = 50000 ∧
  p.typeA ≥ 18000 ∧
  p.days ≤ 8 ∧
  p.typeA ≤ 6000 * p.days ∧
  p.typeB ≤ 8000 * (p.days - (p.typeA / 6000))

/-- Calculates the profit for a given production plan -/
def profit (p : MaskProduction) : ℕ :=
  (p.typeA * 5 + p.typeB * 3) / 10

/-- Theorem stating the maximum profit and minimum production time -/
theorem mask_production_optimization :
  (∃ p : MaskProduction, isValidProduction p ∧
    (∀ q : MaskProduction, isValidProduction q → profit q ≤ profit p) ∧
    profit p = 23400) ∧
  (∃ p : MaskProduction, isValidProduction p ∧
    (∀ q : MaskProduction, isValidProduction q → p.days ≤ q.days) ∧
    p.days = 7) := by
  sorry


end mask_production_optimization_l910_91000


namespace jack_barbecue_sauce_l910_91075

/-- The amount of vinegar used in Jack's barbecue sauce recipe -/
def vinegar_amount : ℚ → Prop :=
  fun v =>
    let ketchup : ℚ := 3
    let honey : ℚ := 1
    let burger_sauce : ℚ := 1/4
    let sandwich_sauce : ℚ := 1/6
    let num_burgers : ℚ := 8
    let num_sandwiches : ℚ := 18
    let total_sauce : ℚ := num_burgers * burger_sauce + num_sandwiches * sandwich_sauce
    ketchup + v + honey = total_sauce

theorem jack_barbecue_sauce :
  vinegar_amount 1 := by sorry

end jack_barbecue_sauce_l910_91075


namespace arithmetic_sequence_with_geometric_mean_l910_91041

/-- An arithmetic sequence with common difference 2 -/
def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∀ n, a (n + 1) = a n + 2

/-- The 4th term is the geometric mean of the 2nd and 5th terms -/
def geometric_mean_condition (a : ℕ → ℝ) : Prop :=
  a 4 ^ 2 = a 2 * a 5

/-- Main theorem: If a is an arithmetic sequence with common difference 2
    and the 4th term is the geometric mean of the 2nd and 5th terms,
    then the 2nd term is -8 -/
theorem arithmetic_sequence_with_geometric_mean
  (a : ℕ → ℝ)
  (h1 : arithmetic_sequence a)
  (h2 : geometric_mean_condition a) :
  a 2 = -8 := by sorry

end arithmetic_sequence_with_geometric_mean_l910_91041


namespace coin_identification_possible_l910_91090

/-- Represents the expert's response, which is always an overestimate -/
structure ExpertResponse :=
  (reported : ℕ)
  (actual : ℕ)
  (overestimate : ℕ)
  (h : reported = actual + overestimate)

/-- Represents the coin identification process -/
def can_identify_counterfeit (total_coins : ℕ) (max_presentation : ℕ) : Prop :=
  ∀ (counterfeit : Finset ℕ) (overestimate : ℕ),
    counterfeit.card ≤ total_coins →
    (∀ subset : Finset ℕ, subset.card ≤ max_presentation →
      ∃ response : ExpertResponse,
        response.actual = (subset ∩ counterfeit).card ∧
        response.overestimate = overestimate) →
    ∃ process : ℕ → Bool,
      ∀ coin, coin < total_coins → (process coin ↔ coin ∈ counterfeit)

theorem coin_identification_possible :
  can_identify_counterfeit 100 20 :=
sorry

end coin_identification_possible_l910_91090


namespace cube_root_sum_theorem_l910_91072

theorem cube_root_sum_theorem :
  ∃ (x : ℝ), (x^(1/3) + (27 - x)^(1/3) = 3) ∧
  (∀ (y : ℝ), (y^(1/3) + (27 - y)^(1/3) = 3) → x ≤ y) →
  ∃ (r s : ℤ), (x = r - Real.sqrt s) ∧ (r + s = 0) := by
  sorry

end cube_root_sum_theorem_l910_91072


namespace percentage_female_officers_on_duty_l910_91085

/-- Given a police force with female officers, calculate the percentage on duty. -/
theorem percentage_female_officers_on_duty
  (total_on_duty : ℕ)
  (half_on_duty_female : ℕ)
  (total_female_officers : ℕ)
  (h1 : total_on_duty = 300)
  (h2 : half_on_duty_female = total_on_duty / 2)
  (h3 : total_female_officers = 1000) :
  (half_on_duty_female : ℚ) / total_female_officers * 100 = 15 := by
sorry

end percentage_female_officers_on_duty_l910_91085


namespace f_value_l910_91034

-- Define the ceiling function
def ceiling (x : ℚ) : ℤ := Int.ceil x

-- Define the function f
def f (x y : ℚ) : ℚ := x - y * ceiling (x / y)

-- State the theorem
theorem f_value : f (1/3) (-3/7) = -2/21 := by
  sorry

end f_value_l910_91034


namespace derek_initial_money_l910_91055

theorem derek_initial_money (initial_money : ℚ) : 
  (initial_money / 2 - (initial_money / 2) / 4 = 360) → initial_money = 960 := by
  sorry

end derek_initial_money_l910_91055


namespace no_ab_term_when_m_is_neg_six_l910_91089

-- Define the polynomial as a function of a, b, and m
def polynomial (a b m : ℝ) : ℝ := 3 * (a^2 - 2*a*b - b^2) - (a^2 + m*a*b + 2*b^2)

-- Theorem stating that the polynomial has no ab term when m = -6
theorem no_ab_term_when_m_is_neg_six :
  ∀ a b : ℝ, (∀ m : ℝ, polynomial a b m = 2*a^2 - (6+m)*a*b - 5*b^2) →
  (∃! m : ℝ, ∀ a b : ℝ, polynomial a b m = 2*a^2 - 5*b^2) →
  (∃ m : ℝ, m = -6 ∧ ∀ a b : ℝ, polynomial a b m = 2*a^2 - 5*b^2) :=
by sorry

end no_ab_term_when_m_is_neg_six_l910_91089


namespace john_investment_proof_l910_91027

/-- The amount John invested in total -/
def total_investment : ℝ := 1200

/-- The annual interest rate for Bank A -/
def rate_A : ℝ := 0.04

/-- The annual interest rate for Bank B -/
def rate_B : ℝ := 0.06

/-- The number of years the money is invested -/
def years : ℕ := 2

/-- The total amount after two years -/
def final_amount : ℝ := 1300.50

/-- The amount John invested in Bank A -/
def investment_A : ℝ := 1138.57

theorem john_investment_proof :
  ∃ (x : ℝ), 
    x = investment_A ∧ 
    x ≥ 0 ∧ 
    x ≤ total_investment ∧
    x * (1 + rate_A) ^ years + (total_investment - x) * (1 + rate_B) ^ years = final_amount :=
by sorry

end john_investment_proof_l910_91027


namespace diagonal_division_ratio_equality_l910_91035

/-- A point in 2D space -/
structure Point :=
  (x : ℝ)
  (y : ℝ)

/-- A quadrilateral defined by four points -/
structure Quadrilateral :=
  (A B C D : Point)

/-- The orthocenter of a triangle -/
def orthocenter (A B C : Point) : Point :=
  sorry

/-- The ratio in which a line segment is divided by a point -/
def divisionRatio (A B P : Point) : ℝ :=
  sorry

/-- The intersection point of two line segments -/
def intersectionPoint (A B C D : Point) : Point :=
  sorry

/-- Theorem: In convex quadrilaterals ABCD and A'B'C'D', where A', B', C', D' are orthocenters
    of triangles BCD, CDA, DAB, ABC respectively, the corresponding diagonals are divided by 
    the points of intersection in the same ratio -/
theorem diagonal_division_ratio_equality 
  (ABCD : Quadrilateral) 
  (A' B' C' D' : Point) 
  (h_convex : sorry) -- Assume ABCD is convex
  (h_A' : A' = orthocenter ABCD.B ABCD.C ABCD.D)
  (h_B' : B' = orthocenter ABCD.C ABCD.D ABCD.A)
  (h_C' : C' = orthocenter ABCD.D ABCD.A ABCD.B)
  (h_D' : D' = orthocenter ABCD.A ABCD.B ABCD.C) :
  let P := intersectionPoint ABCD.A ABCD.C ABCD.B ABCD.D
  let P' := intersectionPoint A' C' B' D'
  divisionRatio ABCD.A ABCD.C P = divisionRatio A' C' P' ∧
  divisionRatio ABCD.B ABCD.D P = divisionRatio B' D' P' :=
sorry

end diagonal_division_ratio_equality_l910_91035


namespace length_ae_is_21_l910_91016

/-- Given 5 consecutive points on a straight line, prove that under certain conditions, the length of ae is 21 -/
theorem length_ae_is_21
  (a b c d e : ℝ) -- Representing points as real numbers on a line
  (h_consecutive : a < b ∧ b < c ∧ c < d ∧ d < e) -- Consecutive points
  (h_bc_cd : c - b = 3 * (d - c)) -- bc = 3 cd
  (h_de : e - d = 8) -- de = 8
  (h_ab : b - a = 5) -- ab = 5
  (h_ac : c - a = 11) -- ac = 11
  : e - a = 21 := by
  sorry

end length_ae_is_21_l910_91016
