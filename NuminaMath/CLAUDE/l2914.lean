import Mathlib

namespace cubic_equation_root_b_value_l2914_291482

theorem cubic_equation_root_b_value (a b : ℚ) : 
  (∃ x : ℝ, x = 3 + Real.sqrt 5 ∧ x^3 + a*x^2 + b*x + 12 = 0) → b = -14 := by
  sorry

end cubic_equation_root_b_value_l2914_291482


namespace rectangle_area_increase_l2914_291416

theorem rectangle_area_increase (L W : ℝ) (h1 : L * W = 500) : 
  (1.2 * L) * (1.2 * W) = 720 := by
sorry

end rectangle_area_increase_l2914_291416


namespace recurring_decimal_sum_l2914_291448

theorem recurring_decimal_sum : 
  (1 : ℚ) / 3 + 4 / 99 + 5 / 999 = 42 / 111 := by sorry

end recurring_decimal_sum_l2914_291448


namespace teacher_wang_pen_purchase_l2914_291424

/-- Given that Teacher Wang has enough money to buy 72 pens at 5 yuan each,
    prove that he can buy 60 pens when the price increases to 6 yuan each. -/
theorem teacher_wang_pen_purchase (initial_pens : ℕ) (initial_price : ℕ) (new_price : ℕ) :
  initial_pens = 72 → initial_price = 5 → new_price = 6 →
  (initial_pens * initial_price) / new_price = 60 := by
  sorry

end teacher_wang_pen_purchase_l2914_291424


namespace max_valid_domains_l2914_291475

def f (x : ℝ) : ℝ := x^2 - 1

def is_valid_domain (D : Set ℝ) : Prop :=
  (∀ x ∈ D, f x = 0 ∨ f x = 1) ∧
  (∃ x ∈ D, f x = 0) ∧
  (∃ x ∈ D, f x = 1)

theorem max_valid_domains :
  ∃ (domains : Finset (Set ℝ)),
    (∀ D ∈ domains, is_valid_domain D) ∧
    (∀ D, is_valid_domain D → D ∈ domains) ∧
    domains.card = 9 :=
sorry

end max_valid_domains_l2914_291475


namespace sum_of_roots_l2914_291438

theorem sum_of_roots (a b c d : ℝ) : 
  (∀ x : ℝ, x^2 - 2*c*x - 5*d = 0 ↔ x = a ∨ x = b) →
  (∀ x : ℝ, x^2 - 2*a*x - 5*b = 0 ↔ x = c ∨ x = d) →
  a ≠ b ∧ a ≠ c ∧ a ≠ d ∧ b ≠ c ∧ b ≠ d ∧ c ≠ d →
  a + b + c + d = 30 := by
sorry

end sum_of_roots_l2914_291438


namespace quadratic_sum_of_constants_l2914_291459

theorem quadratic_sum_of_constants (x : ℝ) : 
  ∃ (b c : ℝ), x^2 - 20*x + 49 = (x + b)^2 + c ∧ b + c = -61 := by
  sorry

end quadratic_sum_of_constants_l2914_291459


namespace percentage_problem_l2914_291486

theorem percentage_problem (number : ℝ) (P : ℝ) : number = 15 → P = 20 / 100 * number + 47 := by
  sorry

end percentage_problem_l2914_291486


namespace min_value_expression_l2914_291405

theorem min_value_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  ∃ m : ℝ, m = -500000 ∧ ∀ x y : ℝ, x > 0 → y > 0 →
    (x + 1/y) * (x + 1/y - 1000) + (y + 1/x) * (y + 1/x - 1000) ≥ m :=
by sorry

end min_value_expression_l2914_291405


namespace complex_sum_square_l2914_291409

variable (a b c : ℂ)

theorem complex_sum_square (h1 : a^2 + a*b + b^2 = 1 + I)
                           (h2 : b^2 + b*c + c^2 = -2)
                           (h3 : c^2 + c*a + a^2 = 1) :
  (a*b + b*c + c*a)^2 = (-11 - 4*I) / 3 := by
  sorry

end complex_sum_square_l2914_291409


namespace squares_in_figure_100_l2914_291432

/-- The number of nonoverlapping unit squares in figure n -/
def f (n : ℕ) : ℕ := 2 * n^2 + 2 * n + 1

/-- The sequence of nonoverlapping unit squares follows the pattern -/
axiom sequence_pattern :
  f 0 = 1 ∧ f 1 = 5 ∧ f 2 = 13 ∧ f 3 = 25

/-- The number of nonoverlapping unit squares in figure 100 is 20201 -/
theorem squares_in_figure_100 : f 100 = 20201 := by sorry

end squares_in_figure_100_l2914_291432


namespace function_difference_l2914_291418

/-- Given a function f(x) = 3x^2 + 5x - 4, prove that f(x+h) - f(x) = h(3h + 6x + 5) for all real x and h. -/
theorem function_difference (x h : ℝ) : 
  let f : ℝ → ℝ := λ t => 3 * t^2 + 5 * t - 4
  f (x + h) - f x = h * (3 * h + 6 * x + 5) := by
  sorry

end function_difference_l2914_291418


namespace f_minimum_value_l2914_291408

-- Define the function f
def f (x a : ℝ) : ℝ := |x + 2| + |x - a|

-- State the theorem
theorem f_minimum_value (a : ℝ) :
  (∀ x : ℝ, f x a ≥ 3) ↔ (a ≤ -5 ∨ a ≥ 1) :=
sorry

end f_minimum_value_l2914_291408


namespace sequence_is_geometric_l2914_291440

theorem sequence_is_geometric (a : ℝ) (h : a ≠ 0) :
  (∃ S : ℕ → ℝ, ∀ n : ℕ, S n = a^n - 1) →
  (∃ r : ℝ, ∀ n : ℕ, ∃ u : ℕ → ℝ, u (n+1) = r * u n) :=
by sorry

end sequence_is_geometric_l2914_291440


namespace gcd_47_power_plus_one_l2914_291401

theorem gcd_47_power_plus_one : Nat.gcd (47^5 + 1) (47^5 + 47^3 + 1) = 1 := by
  sorry

end gcd_47_power_plus_one_l2914_291401


namespace hyperbola_eccentricity_l2914_291436

/-- The eccentricity of a hyperbola with equation x²/a² - y²/b² = 1 is √(a² + b²) / a -/
theorem hyperbola_eccentricity (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let e := Real.sqrt (a^2 + b^2) / a
  (∀ x y, x^2 / a^2 - y^2 / b^2 = 1) → e = Real.sqrt 6 / 2 :=
by sorry

end hyperbola_eccentricity_l2914_291436


namespace bull_work_hours_equality_l2914_291413

/-- Represents the work rate of bulls ploughing fields -/
structure BullWork where
  bulls : ℕ
  fields : ℕ
  days : ℕ
  hours_per_day : ℝ

/-- Calculates the total bull-hours for a given BullWork -/
def total_bull_hours (work : BullWork) : ℝ :=
  work.bulls * work.fields * work.days * work.hours_per_day

theorem bull_work_hours_equality (work1 work2 : BullWork) 
  (h1 : work1.bulls = 10)
  (h2 : work1.fields = 20)
  (h3 : work1.days = 3)
  (h4 : work2.bulls = 30)
  (h5 : work2.fields = 32)
  (h6 : work2.days = 2)
  (h7 : work2.hours_per_day = 8)
  (h8 : total_bull_hours work1 = total_bull_hours work2) :
  work1.hours_per_day = 12.8 := by
  sorry

end bull_work_hours_equality_l2914_291413


namespace dance_troupe_members_l2914_291490

theorem dance_troupe_members : ∃! n : ℕ, 
  150 < n ∧ n < 300 ∧ 
  n % 6 = 2 ∧ 
  n % 8 = 3 ∧ 
  n % 9 = 4 ∧ 
  n = 176 := by
  sorry

end dance_troupe_members_l2914_291490


namespace remaining_cube_volume_l2914_291447

/-- The remaining volume of a cube after removing a cylindrical section -/
theorem remaining_cube_volume (cube_side : ℝ) (cylinder_radius : ℝ) (cylinder_height : ℝ) :
  cube_side = 6 →
  cylinder_radius = 3 →
  cylinder_height = 6 →
  cube_side ^ 3 - π * cylinder_radius ^ 2 * cylinder_height = 216 - 54 * π :=
by
  sorry

end remaining_cube_volume_l2914_291447


namespace intersection_range_of_b_l2914_291457

-- Define the sets M and N
def M : Set (ℝ × ℝ) := {p | p.1^2 + 2*p.2^2 = 3}
def N (m b : ℝ) : Set (ℝ × ℝ) := {p | p.2 = m*p.1 + b}

-- State the theorem
theorem intersection_range_of_b :
  (∀ m : ℝ, (M ∩ N m b).Nonempty) ↔ b ∈ Set.Icc (-Real.sqrt 6 / 2) (Real.sqrt 6 / 2) :=
sorry

end intersection_range_of_b_l2914_291457


namespace percentage_and_subtraction_l2914_291402

theorem percentage_and_subtraction (y : ℝ) : 
  (20 : ℝ) / y = (80 : ℝ) / 100 → y = 25 ∧ y - 15 = 10 := by
  sorry

end percentage_and_subtraction_l2914_291402


namespace initial_weight_proof_l2914_291491

theorem initial_weight_proof (W : ℝ) : 
  (W > 0) →
  (0.8 * (0.9 * W) = 36000) →
  (W = 50000) := by
sorry

end initial_weight_proof_l2914_291491


namespace cube_surface_area_l2914_291431

/-- Given a cube with volume 1728 cubic centimeters, its surface area is 864 square centimeters. -/
theorem cube_surface_area (volume : ℝ) (side : ℝ) (surface_area : ℝ) : 
  volume = 1728 → 
  volume = side^3 → 
  surface_area = 6 * side^2 → 
  surface_area = 864 := by
sorry

end cube_surface_area_l2914_291431


namespace recycle_243_cans_l2914_291470

/-- The number of new cans that can be made from recycling a given number of aluminum cans. -/
def recycle_cans (initial_cans : ℕ) : ℕ :=
  if initial_cans < 3 then 0
  else (initial_cans / 3) + recycle_cans (initial_cans / 3)

/-- Theorem stating that recycling 243 aluminum cans results in 121 new cans. -/
theorem recycle_243_cans :
  recycle_cans 243 = 121 := by sorry

end recycle_243_cans_l2914_291470


namespace linear_equation_condition_l2914_291458

theorem linear_equation_condition (m : ℝ) : 
  (|m - 1| = 1 ∧ m - 2 ≠ 0) ↔ m = 0 := by
sorry

end linear_equation_condition_l2914_291458


namespace complement_of_union_in_U_l2914_291437

def U : Set ℕ := {x | x > 0 ∧ x < 6}
def A : Set ℕ := {1, 3}
def B : Set ℕ := {3, 5}

theorem complement_of_union_in_U : (U \ (A ∪ B)) = {2, 4} := by sorry

end complement_of_union_in_U_l2914_291437


namespace zebra_population_last_year_l2914_291481

/-- The number of zebras in a national park over two consecutive years. -/
structure ZebraPopulation where
  current : ℕ
  born : ℕ
  died : ℕ
  last_year : ℕ

/-- Theorem stating the relationship between the zebra population this year and last year. -/
theorem zebra_population_last_year (zp : ZebraPopulation)
    (h1 : zp.current = 725)
    (h2 : zp.born = 419)
    (h3 : zp.died = 263)
    : zp.last_year = 569 := by
  sorry

end zebra_population_last_year_l2914_291481


namespace arithmetic_sequence_sum_l2914_291461

/-- An arithmetic sequence {a_n} with its partial sums S_n -/
structure ArithmeticSequence where
  a : ℕ → ℝ
  S : ℕ → ℝ
  is_arithmetic : ∀ n, a (n + 1) - a n = a (n + 2) - a (n + 1)
  sum_def : ∀ n, S n = (n : ℝ) * (2 * a 1 + (n - 1) * (a 2 - a 1)) / 2

/-- Theorem: For an arithmetic sequence, if S_4 = 25 and S_8 = 100, then S_12 = 225 -/
theorem arithmetic_sequence_sum (seq : ArithmeticSequence) 
  (h1 : seq.S 4 = 25) (h2 : seq.S 8 = 100) : seq.S 12 = 225 := by
  sorry

end arithmetic_sequence_sum_l2914_291461


namespace fraction_equality_l2914_291443

theorem fraction_equality (a : ℝ) (h1 : a ≠ 2) (h2 : a ≠ -2) :
  (2 * a) / (a^2 - 4) - 1 / (a - 2) = 1 / (a + 2) := by
  sorry

end fraction_equality_l2914_291443


namespace expression_perfect_square_iff_l2914_291454

def is_perfect_square (x : ℕ) : Prop := ∃ y : ℕ, y * y = x

def expression (n : ℕ) : ℕ := (n^2 + 11*n - 4) * n.factorial + 33 * 13^n + 4

theorem expression_perfect_square_iff (n : ℕ) (hn : n > 0) :
  is_perfect_square (expression n) ↔ n = 1 ∨ n = 2 :=
sorry

end expression_perfect_square_iff_l2914_291454


namespace eat_porridge_together_l2914_291488

/-- Masha's time to eat one bowl of porridge in minutes -/
def mashaTime : ℝ := 12

/-- The Bear's eating speed relative to Masha's -/
def bearRelativeSpeed : ℝ := 2

/-- Number of bowls to eat together -/
def totalBowls : ℝ := 6

/-- Time for Masha and the Bear to eat all bowls together -/
def totalTime : ℝ := 24

theorem eat_porridge_together :
  (totalBowls * mashaTime) / (1 + bearRelativeSpeed) = totalTime := by
  sorry

end eat_porridge_together_l2914_291488


namespace widget_count_l2914_291415

theorem widget_count : ∃ (a b c d e f : ℕ),
  3 * a + 11 * b + 5 * c + 7 * d + 13 * e + 17 * f = 3255 ∧
  3^a * 11^b * 5^c * 7^d * 13^e * 17^f = 351125648000 ∧
  c = 3 := by
sorry

end widget_count_l2914_291415


namespace park_visitors_difference_l2914_291493

theorem park_visitors_difference (total : ℕ) (hikers : ℕ) (bikers : ℕ) : 
  total = 676 → hikers = 427 → total = hikers + bikers → hikers - bikers = 178 := by
  sorry

end park_visitors_difference_l2914_291493


namespace division_in_base5_l2914_291456

/-- Converts a number from base 5 to base 10 -/
def base5ToBase10 (n : List Nat) : Nat :=
  n.enum.foldl (fun acc (i, d) => acc + d * (5 ^ i)) 0

/-- Converts a number from base 10 to base 5 -/
def base10ToBase5 (n : Nat) : List Nat :=
  if n = 0 then [0] else
    let rec aux (m : Nat) (acc : List Nat) : List Nat :=
      if m = 0 then acc else aux (m / 5) ((m % 5) :: acc)
    aux n []

theorem division_in_base5 :
  let dividend := base5ToBase10 [2, 3, 2, 3]  -- 3232 in base 5
  let divisor := base5ToBase10 [1, 2]         -- 21 in base 5
  let quotient := base5ToBase10 [0, 3, 1]     -- 130 in base 5
  let remainder := 2
  dividend = divisor * quotient + remainder ∧
  remainder < divisor ∧
  base10ToBase5 (dividend / divisor) = [0, 3, 1] ∧
  base10ToBase5 (dividend % divisor) = [2] :=
by sorry


end division_in_base5_l2914_291456


namespace lotto_winnings_theorem_l2914_291421

/-- The amount of money won by each boy in the "Russian Lotto" draw. -/
structure LottoWinnings where
  kolya : ℕ
  misha : ℕ
  vitya : ℕ

/-- The conditions of the "Russian Lotto" draw. -/
def lotto_conditions (w : LottoWinnings) : Prop :=
  w.misha = w.kolya + 943 ∧
  w.vitya = w.misha + 127 ∧
  w.misha + w.kolya = w.vitya + 479

/-- The theorem stating the correct winnings for each boy. -/
theorem lotto_winnings_theorem :
  ∃ (w : LottoWinnings), lotto_conditions w ∧ w.kolya = 606 ∧ w.misha = 1549 ∧ w.vitya = 1676 :=
by
  sorry

end lotto_winnings_theorem_l2914_291421


namespace tan_difference_absolute_value_l2914_291414

theorem tan_difference_absolute_value (α β : Real) : 
  (∃ x y : Real, x^2 - 2*x - 4 = 0 ∧ y^2 - 2*y - 4 = 0 ∧ x = Real.tan α ∧ y = Real.tan β) →
  |Real.tan (α - β)| = 2 * Real.sqrt 5 / 3 := by
  sorry

end tan_difference_absolute_value_l2914_291414


namespace candy_bar_consumption_l2914_291492

/-- Given that a candy bar contains 31 calories and a person consumed 341 calories,
    prove that the number of candy bars eaten is 11. -/
theorem candy_bar_consumption (calories_per_bar : ℕ) (total_calories : ℕ) : 
  calories_per_bar = 31 → total_calories = 341 → total_calories / calories_per_bar = 11 := by
  sorry

end candy_bar_consumption_l2914_291492


namespace even_function_period_2_equivalence_l2914_291483

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

def has_period_2 (f : ℝ → ℝ) : Prop := ∀ x, f (x + 2) = f x

theorem even_function_period_2_equivalence (f : ℝ → ℝ) (h : is_even f) :
  (∀ x, f (1 - x) = f (1 + x)) ↔ has_period_2 f :=
sorry

end even_function_period_2_equivalence_l2914_291483


namespace lexie_paintings_l2914_291429

theorem lexie_paintings (num_rooms : ℕ) (paintings_per_room : ℕ) 
  (h1 : num_rooms = 4) 
  (h2 : paintings_per_room = 8) : 
  num_rooms * paintings_per_room = 32 := by
sorry

end lexie_paintings_l2914_291429


namespace initial_group_size_l2914_291410

theorem initial_group_size (average_increase : ℝ) (old_weight new_weight : ℝ) :
  average_increase = 3.5 ∧ old_weight = 47 ∧ new_weight = 68 →
  (new_weight - old_weight) / average_increase = 6 :=
by sorry

end initial_group_size_l2914_291410


namespace car_distance_l2914_291489

/-- Given a car that travels 180 miles in 4 hours, prove that it will travel 135 miles in the next 3 hours at the same rate. -/
theorem car_distance (initial_distance : ℝ) (initial_time : ℝ) (next_time : ℝ) :
  initial_distance = 180 ∧ initial_time = 4 ∧ next_time = 3 →
  (initial_distance / initial_time) * next_time = 135 := by
  sorry

end car_distance_l2914_291489


namespace conjunction_implies_disjunction_l2914_291451

theorem conjunction_implies_disjunction (p q : Prop) : (p ∧ q) → (p ∨ q) := by
  sorry

end conjunction_implies_disjunction_l2914_291451


namespace new_average_income_l2914_291474

/-- Given a family's initial average income, number of earning members, and the income of a deceased member,
    calculate the new average income after the member's death. -/
theorem new_average_income
  (initial_average : ℚ)
  (initial_members : ℕ)
  (deceased_income : ℚ)
  (new_members : ℕ)
  (h1 : initial_average = 735)
  (h2 : initial_members = 4)
  (h3 : deceased_income = 1170)
  (h4 : new_members = initial_members - 1) :
  let initial_total := initial_average * initial_members
  let new_total := initial_total - deceased_income
  new_total / new_members = 590 := by
sorry


end new_average_income_l2914_291474


namespace stack_probability_l2914_291444

/-- Represents the dimensions of a crate -/
structure CrateDimensions where
  length : ℕ
  width : ℕ
  height : ℕ

/-- Calculates the probability of a stack of crates being a certain height -/
def probabilityOfStackHeight (crate : CrateDimensions) (numCrates : ℕ) (targetHeight : ℕ) : ℚ :=
  sorry

/-- The main theorem stating the probability of a stack of 10 crates being 41 ft tall -/
theorem stack_probability :
  let crate : CrateDimensions := { length := 3, width := 4, height := 6 }
  probabilityOfStackHeight crate 10 41 = 190 / 2187 := by
  sorry

end stack_probability_l2914_291444


namespace sum_geq_sqrt_three_l2914_291425

theorem sum_geq_sqrt_three (a b c : ℝ) (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c) 
  (h : a * b + b * c + c * a = 1) : a + b + c ≥ Real.sqrt 3 := by
  sorry

end sum_geq_sqrt_three_l2914_291425


namespace twelve_percent_of_700_is_84_l2914_291498

theorem twelve_percent_of_700_is_84 : ∃ x : ℝ, (12 / 100) * x = 84 ∧ x = 700 := by
  sorry

end twelve_percent_of_700_is_84_l2914_291498


namespace min_value_of_function_l2914_291423

open Real

theorem min_value_of_function (x : ℝ) (h : x > 0) :
  let f := fun (x : ℝ) => x - 1 - (log x) / x
  (∀ y > 0, f y ≥ 0) ∧ (∃ z > 0, f z = 0) :=
by sorry

end min_value_of_function_l2914_291423


namespace distance_between_points_l2914_291400

/-- The distance between two points A(4,-3) and B(4,5) is 8. -/
theorem distance_between_points : Real.sqrt ((4 - 4)^2 + (5 - (-3))^2) = 8 := by
  sorry

end distance_between_points_l2914_291400


namespace car_distance_theorem_l2914_291403

/-- Calculates the total distance travelled by a car with increasing speed over a given number of hours -/
def totalDistance (initialDistance : ℕ) (speedIncrease : ℕ) (hours : ℕ) : ℕ :=
  let distanceList := List.range hours |>.map (fun h => initialDistance + h * speedIncrease)
  distanceList.sum

/-- Theorem stating that a car with given initial speed and speed increase travels 546 km in 12 hours -/
theorem car_distance_theorem :
  totalDistance 35 2 12 = 546 := by
  sorry

end car_distance_theorem_l2914_291403


namespace game_ends_after_17_rounds_l2914_291417

/-- Represents a player in the token game -/
structure Player where
  name : String
  tokens : ℕ

/-- Represents the state of the game -/
structure GameState where
  players : List Player
  rounds : ℕ

/-- Determines if the game has ended -/
def gameEnded (state : GameState) : Bool :=
  state.players.any (fun p => p.tokens = 0)

/-- Updates the game state for one round -/
def updateGameState (state : GameState) : GameState :=
  sorry -- Implementation details omitted

/-- Runs the game until it ends -/
def runGame (initialState : GameState) : ℕ :=
  sorry -- Implementation details omitted

/-- Theorem stating that the game ends after 17 rounds -/
theorem game_ends_after_17_rounds :
  let initialState := GameState.mk
    [Player.mk "A" 20, Player.mk "B" 18, Player.mk "C" 16]
    0
  runGame initialState = 17 := by
  sorry

end game_ends_after_17_rounds_l2914_291417


namespace middle_number_in_ratio_l2914_291468

theorem middle_number_in_ratio (a b c : ℝ) : 
  a / b = 2 / 3 → 
  b / c = 3 / 4 → 
  a^2 + c^2 = 180 → 
  b = 9 := by
sorry

end middle_number_in_ratio_l2914_291468


namespace max_profit_l2914_291450

/-- Represents the shopping mall's helmet purchasing problem --/
structure HelmetProblem where
  costA : ℕ → ℕ  -- Cost function for type A helmets
  costB : ℕ → ℕ  -- Cost function for type B helmets
  sellA : ℕ      -- Selling price of type A helmet
  sellB : ℕ      -- Selling price of type B helmet
  totalHelmets : ℕ  -- Total number of helmets to purchase
  maxCost : ℕ    -- Maximum total cost
  minProfit : ℕ  -- Minimum required profit

/-- Calculates the profit for a given number of type A helmets --/
def profit (p : HelmetProblem) (numA : ℕ) : ℤ :=
  let numB := p.totalHelmets - numA
  (p.sellA - p.costA 1) * numA + (p.sellB - p.costB 1) * numB

/-- Theorem stating the maximum profit configuration --/
theorem max_profit (p : HelmetProblem) : 
  p.costA 8 + p.costB 6 = 630 →
  p.costA 6 + p.costB 8 = 700 →
  p.sellA = 58 →
  p.sellB = 98 →
  p.totalHelmets = 200 →
  p.maxCost = 10200 →
  p.minProfit = 6180 →
  p.costA 1 = 30 →
  p.costB 1 = 65 →
  (∀ n : ℕ, n ≤ p.totalHelmets → 
    p.costA n + p.costB (p.totalHelmets - n) ≤ p.maxCost →
    profit p n ≥ p.minProfit →
    profit p n ≤ profit p 80) ∧
  profit p 80 = 6200 := by
  sorry

end max_profit_l2914_291450


namespace population_change_approx_19_58_percent_l2914_291434

/-- Represents the population change over three years given specific growth and decrease rates -/
def population_change (natural_growth : ℝ) (migration_year1 : ℝ) (migration_year2 : ℝ) (migration_year3 : ℝ) (disaster_decrease : ℝ) : ℝ :=
  let year1 := (1 + natural_growth) * (1 + migration_year1)
  let year2 := (1 + natural_growth) * (1 + migration_year2)
  let year3 := (1 + natural_growth) * (1 + migration_year3)
  let three_year_change := year1 * year2 * year3
  three_year_change * (1 - disaster_decrease)

/-- Theorem stating that the population change over three years is approximately 19.58% -/
theorem population_change_approx_19_58_percent :
  let natural_growth := 0.09
  let migration_year1 := -0.01
  let migration_year2 := -0.015
  let migration_year3 := -0.02
  let disaster_decrease := 0.03
  abs (population_change natural_growth migration_year1 migration_year2 migration_year3 disaster_decrease - 1.1958) < 0.0001 :=
sorry

end population_change_approx_19_58_percent_l2914_291434


namespace number_divisibility_problem_l2914_291462

theorem number_divisibility_problem :
  ∃ (N : ℕ), N > 0 ∧ N % 44 = 0 ∧ N % 35 = 3 ∧ N / 44 = 12 :=
by sorry

end number_divisibility_problem_l2914_291462


namespace siblings_water_consumption_l2914_291453

def cups_per_week (daily_cups : ℕ) : ℕ := daily_cups * 7

theorem siblings_water_consumption :
  let theo_daily := 8
  let mason_daily := 7
  let roxy_daily := 9
  let zara_daily := 10
  let lily_daily := 6
  cups_per_week theo_daily +
  cups_per_week mason_daily +
  cups_per_week roxy_daily +
  cups_per_week zara_daily +
  cups_per_week lily_daily = 280 :=
by
  sorry

end siblings_water_consumption_l2914_291453


namespace probability_square_or_circle_l2914_291487

theorem probability_square_or_circle (total : ℕ) (squares : ℕ) (circles : ℕ) 
  (h1 : total = 10) 
  (h2 : squares = 4) 
  (h3 : circles = 3) :
  (squares + circles : ℚ) / total = 7 / 10 := by
  sorry

end probability_square_or_circle_l2914_291487


namespace greatest_four_digit_multiple_of_23_l2914_291452

theorem greatest_four_digit_multiple_of_23 : ∀ n : ℕ, 1000 ≤ n ∧ n ≤ 9999 ∧ 23 ∣ n → n ≤ 9978 := by
  sorry

end greatest_four_digit_multiple_of_23_l2914_291452


namespace section_formula_vector_form_l2914_291445

/-- Given a line segment CD and a point Q on CD such that CQ:QD = 3:5,
    prove that Q⃗ = (5/8)C⃗ + (3/8)D⃗ --/
theorem section_formula_vector_form (C D Q : EuclideanSpace ℝ (Fin 3)) :
  (∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ Q = (1 - t) • C + t • D) →  -- Q is on line segment CD
  (∃ s : ℝ, s > 0 ∧ dist C Q = (3 / (3 + 5)) * s ∧ dist Q D = (5 / (3 + 5)) * s) →  -- CQ:QD = 3:5
  Q = (5 / 8) • C + (3 / 8) • D :=
by sorry

end section_formula_vector_form_l2914_291445


namespace expression_evaluation_l2914_291463

theorem expression_evaluation :
  ((2^2009)^2 - (2^2007)^2) / ((2^2008)^2 - (2^2006)^2) = 4 := by
sorry

end expression_evaluation_l2914_291463


namespace complex_sum_parts_zero_l2914_291433

theorem complex_sum_parts_zero (a b : ℝ) (i : ℂ) (h : i * i = -1) :
  let z : ℂ := 1 / (i * (1 - i))
  a + b = 0 ∧ z = Complex.mk a b :=
by sorry

end complex_sum_parts_zero_l2914_291433


namespace value_of_x_l2914_291494

theorem value_of_x (w y z x : ℕ) 
  (hw : w = 90)
  (hz : z = w + 25)
  (hy : y = z + 15)
  (hx : x = y + 8) : 
  x = 138 := by
  sorry

end value_of_x_l2914_291494


namespace cucumber_salad_problem_l2914_291441

theorem cucumber_salad_problem (total : ℕ) (cucumber : ℕ) (tomato : ℕ) : 
  total = 280 →
  tomato = 3 * cucumber →
  total = cucumber + tomato →
  cucumber = 70 := by
sorry

end cucumber_salad_problem_l2914_291441


namespace train_speed_l2914_291427

/-- Proves that a train with given length crossing a bridge with given length in a given time has a specific speed -/
theorem train_speed (train_length bridge_length : ℝ) (crossing_time : ℝ) :
  train_length = 160 →
  bridge_length = 215 →
  crossing_time = 30 →
  (train_length + bridge_length) / crossing_time * 3.6 = 45 := by
  sorry

end train_speed_l2914_291427


namespace tangent_length_specific_circle_l2914_291464

/-- A circle passing through three points -/
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

/-- The circle passing through three given points -/
def circleThrough (A B C : ℝ × ℝ) : Circle :=
  sorry

/-- The length of a tangent segment from a point to a circle -/
def tangentLength (P : ℝ × ℝ) (c : Circle) : ℝ :=
  sorry

/-- The theorem stating the length of the tangent segment -/
theorem tangent_length_specific_circle :
  let A : ℝ × ℝ := (4, 5)
  let B : ℝ × ℝ := (7, 9)
  let C : ℝ × ℝ := (6, 14)
  let P : ℝ × ℝ := (1, 1)
  let c := circleThrough A B C
  tangentLength P c = 5 * Real.sqrt 2 := by
  sorry

end tangent_length_specific_circle_l2914_291464


namespace paper_width_calculation_l2914_291439

theorem paper_width_calculation (length : Real) (comparison_width : Real) (area_difference : Real) :
  length = 11 →
  comparison_width = 4.5 →
  area_difference = 100 →
  ∃ width : Real,
    2 * length * width = 2 * comparison_width * length + area_difference ∧
    width = 199 / 22 := by
  sorry

end paper_width_calculation_l2914_291439


namespace probability_white_or_black_l2914_291485

-- Define the total number of balls and the number of balls to be drawn
def total_balls : ℕ := 5
def drawn_balls : ℕ := 3

-- Define the number of favorable outcomes (combinations including white or black)
def favorable_outcomes : ℕ := 9

-- Define the total number of possible outcomes
def total_outcomes : ℕ := Nat.choose total_balls drawn_balls

-- State the theorem
theorem probability_white_or_black :
  (favorable_outcomes : ℚ) / total_outcomes = 9 / 10 := by sorry

end probability_white_or_black_l2914_291485


namespace equation_solution_l2914_291496

-- Define the function f
def f (x : ℝ) : ℝ := x + 4

-- State the theorem
theorem equation_solution :
  ∃ (x : ℝ), (3 * f (x - 2)) / f 0 + 4 = f (2 * x + 1) ∧ x = 2 / 5 :=
by
  sorry

end equation_solution_l2914_291496


namespace joans_kittens_l2914_291473

theorem joans_kittens (initial_kittens given_away_kittens : ℕ) 
  (h1 : initial_kittens = 15)
  (h2 : given_away_kittens = 7) :
  initial_kittens - given_away_kittens = 8 := by
  sorry

end joans_kittens_l2914_291473


namespace tripled_base_and_exponent_l2914_291495

/-- Given c and d are real numbers with d ≠ 0, and s and y are defined such that
    s = (3c)^(3d) and s = c^d * y^(3d), prove that y = 3c. -/
theorem tripled_base_and_exponent (c d : ℝ) (s y : ℝ) (h1 : d ≠ 0) 
    (h2 : s = (3 * c) ^ (3 * d)) (h3 : s = c^d * y^(3*d)) : y = 3 * c := by
  sorry

end tripled_base_and_exponent_l2914_291495


namespace retail_price_calculation_l2914_291404

/-- Proves that the retail price of a machine is $120 given specific conditions --/
theorem retail_price_calculation (wholesale_price : ℝ) (discount_rate : ℝ) (profit_rate : ℝ) :
  wholesale_price = 90 →
  discount_rate = 0.1 →
  profit_rate = 0.2 →
  ∃ (retail_price : ℝ),
    retail_price * (1 - discount_rate) = wholesale_price * (1 + profit_rate) ∧
    retail_price = 120 := by
  sorry

end retail_price_calculation_l2914_291404


namespace ball_placement_methods_l2914_291442

/-- The number of different balls -/
def n : ℕ := 4

/-- The number of different boxes -/
def m : ℕ := 4

/-- The number of ways to place n different balls into m different boxes without any empty boxes -/
def noEmptyBoxes (n m : ℕ) : ℕ := Nat.factorial n

/-- The number of ways to place n different balls into m different boxes allowing empty boxes -/
def allowEmptyBoxes (n m : ℕ) : ℕ := m ^ n

/-- The number of ways to place n different balls into m different boxes with exactly one box left empty -/
def oneEmptyBox (n m : ℕ) : ℕ := 
  Nat.choose m 1 * Nat.choose n (m - 1) * Nat.factorial (m - 1)

theorem ball_placement_methods :
  noEmptyBoxes n m = 24 ∧
  allowEmptyBoxes n m = 256 ∧
  oneEmptyBox n m = 144 := by sorry

end ball_placement_methods_l2914_291442


namespace trigonometric_identity_l2914_291479

theorem trigonometric_identity : 
  Real.cos (43 * π / 180) * Real.cos (77 * π / 180) + 
  Real.sin (43 * π / 180) * Real.cos (167 * π / 180) = -1/2 := by
  sorry

end trigonometric_identity_l2914_291479


namespace integral_sin_over_one_minus_cos_squared_l2914_291499

theorem integral_sin_over_one_minus_cos_squared (f : ℝ → ℝ) :
  (∫ x in Set.Icc (π / 2) π, (2 * Real.sin x) / ((1 - Real.cos x)^2)) = 1 := by
  sorry

end integral_sin_over_one_minus_cos_squared_l2914_291499


namespace f_sum_zero_l2914_291467

-- Define the function f
def f : ℝ → ℝ := sorry

-- Define the properties of f
axiom f_odd (x : ℝ) : f (-x) = -f x
axiom f_property (x : ℝ) : f (2 - x) + f x = 0

-- State the theorem
theorem f_sum_zero : f 2022 + f 2023 = 0 := by sorry

end f_sum_zero_l2914_291467


namespace log_cube_difference_l2914_291412

-- Define the logarithm function
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log a

-- State the theorem
theorem log_cube_difference 
  (a : ℝ) (x₁ x₂ : ℝ) 
  (h_a_pos : a > 0) 
  (h_a_neq_one : a ≠ 1) 
  (h_diff : f a x₁ - f a x₂ = 2) : 
  f a (x₁^3) - f a (x₂^3) = 6 := by
sorry

end log_cube_difference_l2914_291412


namespace largest_square_from_string_l2914_291497

theorem largest_square_from_string (string_length : ℝ) (side_length : ℝ) : 
  string_length = 32 →
  side_length * 4 = string_length →
  side_length = 8 := by
  sorry

end largest_square_from_string_l2914_291497


namespace sally_quarters_l2914_291407

/-- Given that Sally had 760 quarters initially and spent 418 quarters,
    prove that she now has 342 quarters. -/
theorem sally_quarters : 
  ∀ (initial spent remaining : ℕ), 
  initial = 760 → 
  spent = 418 → 
  remaining = initial - spent → 
  remaining = 342 := by sorry

end sally_quarters_l2914_291407


namespace polynomial_factor_l2914_291406

variables {F : Type*} [Field F]
variables (P Q R S : F → F)

theorem polynomial_factor 
  (h : ∀ x, P (x^3) + x * Q (x^3) + x^2 * R (x^5) = (x^4 + x^3 + x^2 + x + 1) * S x) : 
  P 1 = 0 := by
  sorry

end polynomial_factor_l2914_291406


namespace half_of_large_number_l2914_291422

theorem half_of_large_number : (1.2 * 10^30) / 2 = 6.0 * 10^29 := by
  sorry

end half_of_large_number_l2914_291422


namespace exercise_minimum_sets_l2914_291426

/-- Represents the exercise routine over 100 days -/
structure ExerciseRoutine where
  pushups_per_set : ℕ
  pullups_per_set : ℕ
  initial_reps : ℕ
  days : ℕ

/-- Calculates the total number of repetitions over the given days -/
def total_reps (routine : ExerciseRoutine) : ℕ :=
  routine.days * (2 * routine.initial_reps + routine.days - 1) / 2

/-- Represents the solution to the exercise problem -/
structure ExerciseSolution where
  pushup_sets : ℕ
  pullup_sets : ℕ

/-- Theorem stating the minimum number of sets for push-ups and pull-ups -/
theorem exercise_minimum_sets (routine : ExerciseRoutine) 
  (h1 : routine.pushups_per_set = 8)
  (h2 : routine.pullups_per_set = 5)
  (h3 : routine.initial_reps = 41)
  (h4 : routine.days = 100) :
  ∃ (solution : ExerciseSolution), 
    solution.pushup_sets ≥ 100 ∧ 
    solution.pullup_sets ≥ 106 ∧
    solution.pushup_sets * routine.pushups_per_set + 
    solution.pullup_sets * routine.pullups_per_set = 
    total_reps routine :=
  sorry

end exercise_minimum_sets_l2914_291426


namespace octal_is_smallest_l2914_291449

-- Define the base conversion function
def toDecimal (digits : List Nat) (base : Nat) : Nat :=
  digits.foldr (fun d acc => d + base * acc) 0

-- Define the given numbers
def binary : Nat := toDecimal [1, 0, 1, 0, 1, 0] 2
def quinary : Nat := toDecimal [1, 1, 1] 5
def octal : Nat := toDecimal [3, 2] 8
def senary : Nat := toDecimal [5, 4] 6

-- Theorem statement
theorem octal_is_smallest : 
  octal ≤ binary ∧ octal ≤ quinary ∧ octal ≤ senary :=
sorry

end octal_is_smallest_l2914_291449


namespace total_basketballs_l2914_291478

/-- Calculates the total number of basketballs for three basketball teams -/
theorem total_basketballs (spurs_players spurs_balls dynamos_players dynamos_balls lions_players lions_balls : ℕ) :
  spurs_players = 22 →
  spurs_balls = 11 →
  dynamos_players = 18 →
  dynamos_balls = 9 →
  lions_players = 26 →
  lions_balls = 7 →
  spurs_players * spurs_balls + dynamos_players * dynamos_balls + lions_players * lions_balls = 586 :=
by
  sorry

#check total_basketballs

end total_basketballs_l2914_291478


namespace prob_two_red_balls_l2914_291420

/-- The probability of picking two red balls from a bag containing red, blue, and green balls -/
theorem prob_two_red_balls (red blue green : ℕ) (h : red = 5 ∧ blue = 6 ∧ green = 4) :
  let total := red + blue + green
  (red : ℚ) / total * ((red - 1) : ℚ) / (total - 1) = 2 / 21 := by
  sorry

end prob_two_red_balls_l2914_291420


namespace value_added_to_doubled_number_l2914_291419

theorem value_added_to_doubled_number (initial_number : ℕ) (added_value : ℕ) : 
  initial_number = 10 → 
  3 * (2 * initial_number + added_value) = 84 → 
  added_value = 8 := by
sorry

end value_added_to_doubled_number_l2914_291419


namespace stratified_sampling_theorem_l2914_291480

/-- Represents the number of students in each grade and the total sample size -/
structure SchoolPopulation where
  total : Nat
  firstYear : Nat
  secondYear : Nat
  thirdYear : Nat
  sampleSize : Nat

/-- Represents the number of students sampled from each grade -/
structure StratifiedSample where
  firstYear : Nat
  secondYear : Nat
  thirdYear : Nat

/-- Function to calculate the stratified sample given a school population -/
def calculateStratifiedSample (pop : SchoolPopulation) : StratifiedSample :=
  { firstYear := pop.firstYear * pop.sampleSize / pop.total,
    secondYear := pop.secondYear * pop.sampleSize / pop.total,
    thirdYear := pop.thirdYear * pop.sampleSize / pop.total }

theorem stratified_sampling_theorem (pop : SchoolPopulation)
    (h1 : pop.total = 1000)
    (h2 : pop.firstYear = 500)
    (h3 : pop.secondYear = 300)
    (h4 : pop.thirdYear = 200)
    (h5 : pop.sampleSize = 100)
    (h6 : pop.total = pop.firstYear + pop.secondYear + pop.thirdYear) :
    let sample := calculateStratifiedSample pop
    sample.firstYear = 50 ∧ sample.secondYear = 30 ∧ sample.thirdYear = 20 :=
  sorry

#check stratified_sampling_theorem

end stratified_sampling_theorem_l2914_291480


namespace pool_and_deck_area_l2914_291484

/-- Given a rectangular pool with dimensions 10 feet by 12 feet and a surrounding deck
    with a uniform width of 4 feet, the total area of the pool and deck is 360 square feet. -/
theorem pool_and_deck_area :
  let pool_length : ℕ := 12
  let pool_width : ℕ := 10
  let deck_width : ℕ := 4
  let total_length : ℕ := pool_length + 2 * deck_width
  let total_width : ℕ := pool_width + 2 * deck_width
  total_length * total_width = 360 := by
  sorry

end pool_and_deck_area_l2914_291484


namespace probability_AR55_l2914_291446

/-- Represents the set of possible symbols for each position in the license plate -/
def LicensePlateSymbols : Fin 4 → Type
  | 0 => Fin 5  -- Vowels (A, E, I, O, U)
  | 1 => Fin 21 -- Non-vowels (consonants)
  | 2 => Fin 10 -- Digits (0-9)
  | 3 => Fin 10 -- Digits (0-9)

/-- The total number of possible license plates -/
def totalLicensePlates : ℕ := 5 * 21 * 10 * 10

/-- Represents a specific license plate -/
def SpecificPlate : Fin 4 → ℕ
  | 0 => 0  -- 'A' (first vowel)
  | 1 => 17 -- 'R' (18th consonant, 0-indexed)
  | 2 => 5  -- '5'
  | 3 => 5  -- '5'

/-- The probability of randomly selecting the license plate "AR55" -/
theorem probability_AR55 : 
  (1 : ℚ) / totalLicensePlates = 1 / 10500 :=
sorry

end probability_AR55_l2914_291446


namespace sticker_distribution_l2914_291469

/-- The number of ways to distribute n identical objects into k distinct containers,
    with each container receiving at least one object -/
def distribute (n k : ℕ) : ℕ := sorry

/-- The number of ways to distribute 10 identical stickers onto 5 sheets of paper,
    with each sheet receiving at least one sticker -/
theorem sticker_distribution : distribute 10 5 = 7 := by sorry

end sticker_distribution_l2914_291469


namespace piece_exits_at_A2_l2914_291411

/-- Represents the directions a piece can move --/
inductive Direction
  | Up
  | Down
  | Left
  | Right

/-- Represents a cell on the 4x4 board --/
structure Cell where
  row : Fin 4
  col : Fin 4

/-- Represents the state of the board --/
structure BoardState where
  piece_position : Cell
  arrows : Fin 4 → Fin 4 → Direction

/-- Defines the initial state of the board --/
def initial_state : BoardState := sorry

/-- Defines a single move on the board --/
def move (state : BoardState) : BoardState := sorry

/-- Checks if a cell is on the edge of the board --/
def is_edge_cell (cell : Cell) : Bool := sorry

/-- Simulates the movement of the piece until it exits the board --/
def simulate_until_exit (state : BoardState) : Cell := sorry

/-- The main theorem to prove --/
theorem piece_exits_at_A2 :
  let final_cell := simulate_until_exit initial_state
  final_cell.row = 0 ∧ final_cell.col = 1 := by sorry

end piece_exits_at_A2_l2914_291411


namespace descending_order_proof_l2914_291477

theorem descending_order_proof (a b c d : ℝ) : 
  a = Real.sin (33 * π / 180) →
  b = Real.cos (55 * π / 180) →
  c = Real.tan (35 * π / 180) →
  d = Real.log 5 →
  d > c ∧ c > b ∧ b > a :=
by sorry

end descending_order_proof_l2914_291477


namespace range_of_f_l2914_291435

def f (x : ℕ) : ℕ := 2 * x + 1

def domain : Set ℕ := {1, 2, 3}

theorem range_of_f :
  {y | ∃ x ∈ domain, f x = y} = {3, 5, 7} := by sorry

end range_of_f_l2914_291435


namespace possible_in_99_attempts_possible_in_75_attempts_impossible_in_74_attempts_l2914_291476

/-- A type representing a door or a key --/
def DoorKey := Fin 100

/-- A function representing the mapping of keys to doors --/
def KeyToDoor := DoorKey → DoorKey

/-- Predicate to check if a key-to-door mapping is valid --/
def IsValidMapping (f : KeyToDoor) : Prop :=
  ∀ k : DoorKey, (f k).val = k.val ∨ (f k).val = k.val + 1 ∨ (f k).val = k.val - 1

/-- Theorem stating that it's possible to determine the key-door mapping in 99 attempts --/
theorem possible_in_99_attempts (f : KeyToDoor) (h : IsValidMapping f) :
  ∃ (algorithm : ℕ → DoorKey × DoorKey),
    (∀ n : ℕ, n < 99 → (algorithm n).1 ≠ (algorithm n).2 → f ((algorithm n).1) ≠ (algorithm n).2) →
    ∀ k : DoorKey, ∃ n : ℕ, n < 99 ∧ (algorithm n).1 = k ∧ (algorithm n).2 = f k :=
  sorry

/-- Theorem stating that it's possible to determine the key-door mapping in 75 attempts --/
theorem possible_in_75_attempts (f : KeyToDoor) (h : IsValidMapping f) :
  ∃ (algorithm : ℕ → DoorKey × DoorKey),
    (∀ n : ℕ, n < 75 → (algorithm n).1 ≠ (algorithm n).2 → f ((algorithm n).1) ≠ (algorithm n).2) →
    ∀ k : DoorKey, ∃ n : ℕ, n < 75 ∧ (algorithm n).1 = k ∧ (algorithm n).2 = f k :=
  sorry

/-- Theorem stating that it's impossible to determine the key-door mapping in 74 attempts --/
theorem impossible_in_74_attempts :
  ∃ f : KeyToDoor, IsValidMapping f ∧
    ∀ (algorithm : ℕ → DoorKey × DoorKey),
      (∀ n : ℕ, n < 74 → (algorithm n).1 ≠ (algorithm n).2 → f ((algorithm n).1) ≠ (algorithm n).2) →
      ∃ k : DoorKey, ∀ n : ℕ, n < 74 → (algorithm n).1 ≠ k ∨ (algorithm n).2 ≠ f k :=
  sorry

end possible_in_99_attempts_possible_in_75_attempts_impossible_in_74_attempts_l2914_291476


namespace train_speed_l2914_291472

/-- The speed of a train given its length and time to cross a pole -/
theorem train_speed (length : Real) (time : Real) :
  length = 125.01 →
  time = 5 →
  let speed := (length / 1000) / (time / 3600)
  ∃ ε > 0, abs (speed - 90.0072) < ε := by
  sorry

end train_speed_l2914_291472


namespace hyperbola_equation_l2914_291455

-- Define the hyperbola and its properties
def hyperbola_foci (F₁ F₂ : ℝ × ℝ) : Prop :=
  F₁ = (-Real.sqrt 10, 0) ∧ F₂ = (Real.sqrt 10, 0)

def point_on_hyperbola (M : ℝ × ℝ) (F₁ F₂ : ℝ × ℝ) : Prop :=
  let MF₁ := (M.1 - F₁.1, M.2 - F₁.2)
  let MF₂ := (M.2 - F₂.1, M.2 - F₂.2)
  MF₁.1 * MF₂.1 + MF₁.2 * MF₂.2 = 0 ∧
  Real.sqrt (MF₁.1^2 + MF₁.2^2) * Real.sqrt (MF₂.1^2 + MF₂.2^2) = 2

-- Theorem statement
theorem hyperbola_equation (F₁ F₂ M : ℝ × ℝ) :
  hyperbola_foci F₁ F₂ →
  point_on_hyperbola M F₁ F₂ →
  ∃ (x y : ℝ), M = (x, y) ∧ x^2 / 9 - y^2 = 1 :=
sorry

end hyperbola_equation_l2914_291455


namespace gianna_savings_period_l2914_291465

/-- Proves that Gianna saved money for 365 days given the conditions -/
theorem gianna_savings_period (daily_savings : ℕ) (total_savings : ℕ) 
  (h1 : daily_savings = 39)
  (h2 : total_savings = 14235) :
  total_savings / daily_savings = 365 := by
  sorry

end gianna_savings_period_l2914_291465


namespace sphere_cylinder_equal_area_l2914_291471

theorem sphere_cylinder_equal_area (h : ℝ) (d : ℝ) (r : ℝ) :
  h = 16 →
  d = 16 →
  4 * Real.pi * r^2 = 2 * Real.pi * (d / 2) * h →
  r = 8 :=
by sorry

end sphere_cylinder_equal_area_l2914_291471


namespace roots_sum_cube_plus_linear_l2914_291460

theorem roots_sum_cube_plus_linear (α β : ℝ) : 
  (α^2 + 2*α - 1 = 0) → 
  (β^2 + 2*β - 1 = 0) → 
  α^3 + 5*β + 10 = -2 := by
sorry

end roots_sum_cube_plus_linear_l2914_291460


namespace smallest_multiple_l2914_291466

theorem smallest_multiple (x : ℕ) : x = 40 ↔ 
  (x > 0 ∧ 
   800 ∣ (360 * x) ∧ 
   ∀ y : ℕ, y > 0 → 800 ∣ (360 * y) → x ≤ y) :=
by sorry

end smallest_multiple_l2914_291466


namespace set_intersection_theorem_l2914_291430

-- Define the sets M and N
def M : Set ℝ := {x | -2 ≤ x ∧ x ≤ 2}
def N : Set ℝ := {x | x^2 - 3*x ≤ 0}

-- State the theorem
theorem set_intersection_theorem :
  M ∩ (Set.univ \ N) = {x : ℝ | -2 ≤ x ∧ x < 0} := by sorry

end set_intersection_theorem_l2914_291430


namespace linear_function_decreasing_l2914_291428

theorem linear_function_decreasing (x₁ x₂ y₁ y₂ : ℝ) :
  y₁ = -3 * x₁ - 7 →
  y₂ = -3 * x₂ - 7 →
  x₁ > x₂ →
  y₁ < y₂ := by
sorry

end linear_function_decreasing_l2914_291428
