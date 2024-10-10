import Mathlib

namespace water_pouring_theorem_l259_25922

/-- Represents a state of water distribution among three containers -/
structure WaterState where
  a : ℕ
  b : ℕ
  c : ℕ

/-- Represents a pouring action from one container to another -/
inductive PourAction
  | AtoB
  | AtoC
  | BtoA
  | BtoC
  | CtoA
  | CtoB

/-- Applies a pouring action to a water state -/
def applyPour (state : WaterState) (action : PourAction) : WaterState :=
  match action with
  | PourAction.AtoB => { a := state.a - state.b, b := state.b * 2, c := state.c }
  | PourAction.AtoC => { a := state.a - state.c, b := state.b, c := state.c * 2 }
  | PourAction.BtoA => { a := state.a * 2, b := state.b - state.a, c := state.c }
  | PourAction.BtoC => { a := state.a, b := state.b - state.c, c := state.c * 2 }
  | PourAction.CtoA => { a := state.a * 2, b := state.b, c := state.c - state.a }
  | PourAction.CtoB => { a := state.a, b := state.b * 2, c := state.c - state.b }

/-- Predicate to check if a container is empty -/
def isEmptyContainer (state : WaterState) : Prop :=
  state.a = 0 ∨ state.b = 0 ∨ state.c = 0

/-- The main theorem to be proved -/
theorem water_pouring_theorem (initialState : WaterState) :
  ∃ (actions : List PourAction), isEmptyContainer (actions.foldl applyPour initialState) :=
sorry


end water_pouring_theorem_l259_25922


namespace x_greater_abs_y_sufficient_not_necessary_l259_25923

theorem x_greater_abs_y_sufficient_not_necessary (x y : ℝ) :
  (∀ x y : ℝ, x > |y| → x > y) ∧
  (∃ x y : ℝ, x > y ∧ ¬(x > |y|)) :=
sorry

end x_greater_abs_y_sufficient_not_necessary_l259_25923


namespace thursday_five_times_in_july_l259_25956

/-- Represents the days of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Represents a specific date in a month -/
structure Date :=
  (day : Nat)
  (dayOfWeek : DayOfWeek)

/-- A month with its number of days and list of dates -/
structure Month :=
  (numDays : Nat)
  (dates : List Date)

def june : Month := sorry
def july : Month := sorry

/-- Counts the number of occurrences of a specific day in a month -/
def countDayInMonth (m : Month) (d : DayOfWeek) : Nat := sorry

theorem thursday_five_times_in_july 
  (h1 : june.numDays = 30)
  (h2 : july.numDays = 31)
  (h3 : countDayInMonth june DayOfWeek.Tuesday = 5) :
  countDayInMonth july DayOfWeek.Thursday = 5 := by
  sorry

end thursday_five_times_in_july_l259_25956


namespace solve_quadratic_equation_l259_25986

theorem solve_quadratic_equation : 
  ∃ x : ℚ, (10 - 2*x)^2 = 4*x^2 + 20*x ∧ x = 5/3 := by
  sorry

end solve_quadratic_equation_l259_25986


namespace perpendicular_construction_l259_25952

-- Define the basic geometric elements
structure Point := (x y : ℝ)
structure Line := (a b c : ℝ)

-- Define the concept of a point being on a line
def Point.on_line (p : Point) (l : Line) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

-- Define parallel lines
def parallel (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l1.b * l2.a ∧ l1.a * l2.c ≠ l1.c * l2.a

-- Define perpendicular lines
def perpendicular (l1 l2 : Line) : Prop :=
  l1.a * l2.a + l1.b * l2.b = 0

-- Define the construction process
def construct_perpendicular (l : Line) (A : Point) (m l1 l2 lm : Line) (M1 M2 B : Point) : Prop :=
  A.on_line l ∧
  A.on_line m ∧
  parallel l1 m ∧
  parallel l2 m ∧
  M1.on_line l ∧
  M1.on_line l1 ∧
  M2.on_line l ∧
  M2.on_line l2 ∧
  parallel lm (Line.mk (M1.x - A.x) (M1.y - A.y) 0) ∧
  B.on_line l2 ∧
  B.on_line lm

-- State the theorem
theorem perpendicular_construction (l : Line) (A : Point) :
  ∃ (m l1 l2 lm : Line) (M1 M2 B : Point),
    construct_perpendicular l A m l1 l2 lm M1 M2 B →
    perpendicular l (Line.mk (B.x - A.x) (B.y - A.y) 0) :=
sorry

end perpendicular_construction_l259_25952


namespace max_leftover_stickers_l259_25965

theorem max_leftover_stickers (y : ℕ+) : 
  ∃ (q r : ℕ), y = 12 * q + r ∧ r ≤ 11 ∧ 
  ∀ (q' r' : ℕ), y = 12 * q' + r' → r' ≤ r :=
sorry

end max_leftover_stickers_l259_25965


namespace unique_permutations_count_l259_25930

/-- The number of elements in our multiset -/
def n : ℕ := 5

/-- The number of occurrences of the digit 3 -/
def k₁ : ℕ := 3

/-- The number of occurrences of the digit 7 -/
def k₂ : ℕ := 2

/-- The theorem stating that the number of unique permutations of our multiset is 10 -/
theorem unique_permutations_count : (n.factorial) / (k₁.factorial * k₂.factorial) = 10 := by
  sorry

end unique_permutations_count_l259_25930


namespace intersection_and_union_of_sets_l259_25978

theorem intersection_and_union_of_sets (a : ℝ) :
  let A : Set ℝ := {-3, a + 1}
  let B : Set ℝ := {2 * a - 1, a^2 + 1}
  (A ∩ B = {3}) →
  (a = 2 ∧ A ∪ B = {-3, 3, 5}) := by
sorry

end intersection_and_union_of_sets_l259_25978


namespace original_aotd_votes_l259_25911

/-- Represents the vote counts for three books --/
structure VoteCounts where
  got : ℕ  -- Game of Thrones
  twi : ℕ  -- Twilight
  aotd : ℕ  -- The Art of the Deal

/-- Represents the vote alteration process --/
def alter_votes (v : VoteCounts) : ℚ × ℚ × ℚ :=
  (v.got, v.twi / 2, v.aotd / 5)

/-- The theorem to be proved --/
theorem original_aotd_votes (v : VoteCounts) : 
  v.got = 10 ∧ v.twi = 12 ∧ 
  (let (got, twi, aotd) := alter_votes v
   got = (got + twi + aotd) / 2) →
  v.aotd = 20 :=
by sorry

end original_aotd_votes_l259_25911


namespace trapezoid_perimeter_l259_25971

/-- Represents a trapezoid ABCD with given properties -/
structure Trapezoid where
  BC : ℝ
  AP : ℝ
  DQ : ℝ
  AB : ℝ
  CD : ℝ
  bc_length : BC = 32
  ap_length : AP = 24
  dq_length : DQ = 18
  ab_length : AB = 29
  cd_length : CD = 35

/-- Calculates the perimeter of the trapezoid -/
def perimeter (t : Trapezoid) : ℝ :=
  t.AB + t.BC + t.CD + (t.AP + t.BC + t.DQ)

/-- Theorem: The perimeter of the trapezoid is 170 units -/
theorem trapezoid_perimeter (t : Trapezoid) : perimeter t = 170 := by
  sorry

#check trapezoid_perimeter

end trapezoid_perimeter_l259_25971


namespace lcm_gcd_problem_l259_25940

theorem lcm_gcd_problem (a b : ℕ+) 
  (h1 : Nat.lcm a b = 2520)
  (h2 : Nat.gcd a b = 30)
  (h3 : a = 150) :
  b = 504 := by
  sorry

end lcm_gcd_problem_l259_25940


namespace divisibility_condition_l259_25967

theorem divisibility_condition (a b : ℕ+) (h : b ≥ 2) :
  (2^a.val + 1) % (2^b.val - 1) = 0 ↔ b = 2 ∧ a.val % 2 = 1 := by
sorry

end divisibility_condition_l259_25967


namespace tourists_distribution_theorem_l259_25980

/-- The number of ways to distribute tourists among guides -/
def distribute_tourists (num_tourists : ℕ) (num_guides : ℕ) : ℕ :=
  num_guides ^ num_tourists

/-- The number of ways to distribute tourists among guides, excluding cases where some guides have no tourists -/
def distribute_tourists_with_restriction (num_tourists : ℕ) (num_guides : ℕ) : ℕ :=
  distribute_tourists num_tourists num_guides - 
  (num_guides.choose 1 * (num_guides - 1) ^ num_tourists) +
  (num_guides.choose 2 * (num_guides - 2) ^ num_tourists)

/-- The theorem stating that distributing 8 tourists among 3 guides, with each guide having at least one tourist, results in 5796 possible groupings -/
theorem tourists_distribution_theorem : 
  distribute_tourists_with_restriction 8 3 = 5796 := by
  sorry

end tourists_distribution_theorem_l259_25980


namespace johns_bagels_l259_25915

theorem johns_bagels (b m : ℕ) : 
  b + m = 7 →
  (90 * b + 60 * m) % 100 = 0 →
  b = 6 :=
by sorry

end johns_bagels_l259_25915


namespace no_solution_with_vasyas_correction_l259_25997

theorem no_solution_with_vasyas_correction (r : ℝ) : ¬ ∃ (a h : ℝ),
  (0 < r) ∧                           -- radius is positive
  (0 < a) ∧ (0 < h) ∧                 -- base and height are positive
  (a ≤ 2*r) ∧                         -- base is at most diameter
  (h < 2*r) ∧                         -- height is less than diameter
  (a + h = 2*Real.pi*r) :=            -- sum equals circumference (Vasya's condition)
by
  sorry

end no_solution_with_vasyas_correction_l259_25997


namespace floor_sum_for_specific_x_l259_25920

theorem floor_sum_for_specific_x : 
  let x : ℝ := 9.42
  ⌊x⌋ + ⌊2 * x⌋ + ⌊3 * x⌋ = 55 := by sorry

end floor_sum_for_specific_x_l259_25920


namespace employed_males_percentage_l259_25955

theorem employed_males_percentage
  (total_population : ℝ)
  (employed_percentage : ℝ)
  (employed_females_percentage : ℝ)
  (h1 : employed_percentage = 70)
  (h2 : employed_females_percentage = 70)
  (h3 : total_population > 0) :
  (employed_percentage / 100 * (1 - employed_females_percentage / 100) * 100) = 21 := by
  sorry

end employed_males_percentage_l259_25955


namespace winter_sales_proof_l259_25951

/-- Represents the sales of hamburgers in millions for each season and the total year -/
structure HamburgerSales where
  spring_summer : ℝ
  fall : ℝ
  winter : ℝ
  total : ℝ

/-- Given the conditions of the hamburger sales, prove that winter sales are 4 million -/
theorem winter_sales_proof (sales : HamburgerSales) 
  (h1 : sales.total = 20)
  (h2 : sales.spring_summer = 0.6 * sales.total)
  (h3 : sales.fall = 0.2 * sales.total)
  (h4 : sales.total = sales.spring_summer + sales.fall + sales.winter) :
  sales.winter = 4 := by
  sorry

#check winter_sales_proof

end winter_sales_proof_l259_25951


namespace fraction_product_equals_27_l259_25909

theorem fraction_product_equals_27 : 
  (1 : ℚ) / 3 * 9 / 1 * 1 / 27 * 81 / 1 * 1 / 243 * 729 / 1 = 27 := by
  sorry

end fraction_product_equals_27_l259_25909


namespace isosceles_trapezoid_area_l259_25910

/-- The area of an isosceles trapezoid with given dimensions -/
theorem isosceles_trapezoid_area (side : ℝ) (base1 base2 : ℝ) :
  side > 0 ∧ base1 > 0 ∧ base2 > 0 ∧ base1 < base2 ∧ side^2 > ((base2 - base1)/2)^2 →
  let height := Real.sqrt (side^2 - ((base2 - base1)/2)^2)
  (1/2 : ℝ) * (base1 + base2) * height = 48 ∧ side = 5 ∧ base1 = 9 ∧ base2 = 15 := by
  sorry

end isosceles_trapezoid_area_l259_25910


namespace prob_all_players_five_coins_l259_25985

/-- Represents a player in the coin game -/
inductive Player : Type
| Abby : Player
| Bernardo : Player
| Carl : Player
| Debra : Player

/-- Represents a ball color in the game -/
inductive BallColor : Type
| Green : BallColor
| Red : BallColor
| Blue : BallColor
| White : BallColor

/-- Represents the state of the game after each round -/
structure GameState :=
(coins : Player → ℕ)
(round : ℕ)

/-- Represents a single round of the game -/
def play_round (state : GameState) : GameState :=
sorry

/-- The probability of drawing both green and blue balls by the same player in a single round -/
def prob_green_blue_same_player : ℚ :=
1 / 20

/-- The number of rounds in the game -/
def num_rounds : ℕ := 5

/-- The initial number of coins for each player -/
def initial_coins : ℕ := 5

/-- The probability that each player has exactly 5 coins after 5 rounds -/
theorem prob_all_players_five_coins :
  (prob_green_blue_same_player ^ num_rounds : ℚ) = 1 / 3200000 :=
sorry

end prob_all_players_five_coins_l259_25985


namespace ninety_eight_squared_l259_25998

theorem ninety_eight_squared : (100 - 2)^2 = 9604 := by
  sorry

end ninety_eight_squared_l259_25998


namespace nested_root_equality_l259_25962

theorem nested_root_equality (x : ℝ) (h : x ≥ 0) :
  Real.sqrt (x * Real.sqrt (x * Real.sqrt x)) = (x ^ 7) ^ (1 / 8) := by
  sorry

end nested_root_equality_l259_25962


namespace john_total_calories_l259_25925

/-- The number of potato chips John eats -/
def num_chips : ℕ := 10

/-- The total calories of the potato chips -/
def total_chip_calories : ℕ := 60

/-- The number of cheezits John eats -/
def num_cheezits : ℕ := 6

/-- The calories of one potato chip -/
def calories_per_chip : ℚ := total_chip_calories / num_chips

/-- The calories of one cheezit -/
def calories_per_cheezit : ℚ := calories_per_chip * (1 + 1/3)

/-- The total calories John ate -/
def total_calories : ℚ := num_chips * calories_per_chip + num_cheezits * calories_per_cheezit

theorem john_total_calories : total_calories = 108 := by
  sorry

end john_total_calories_l259_25925


namespace movie_marathon_end_time_correct_l259_25969

/-- Represents time in hours and minutes -/
structure Time where
  hours : Nat
  minutes : Nat
  deriving Repr

/-- Represents a duration in hours and minutes -/
structure Duration where
  hours : Nat
  minutes : Nat
  deriving Repr

def addDurationToTime (t : Time) (d : Duration) : Time :=
  let totalMinutes := t.minutes + d.minutes + d.hours * 60
  { hours := t.hours + totalMinutes / 60,
    minutes := totalMinutes % 60 }

def movie_marathon_end_time (start : Time) 
  (movie1 : Duration) (break1 : Duration) 
  (movie2 : Duration) (break2 : Duration) 
  (movie3 : Duration) : Time :=
  let t1 := addDurationToTime start movie1
  let t2 := addDurationToTime t1 break1
  let t3 := addDurationToTime t2 movie2
  let t4 := addDurationToTime t3 break2
  addDurationToTime t4 movie3

theorem movie_marathon_end_time_correct :
  let start := Time.mk 13 0  -- 1:00 p.m.
  let movie1 := Duration.mk 2 20
  let break1 := Duration.mk 0 20
  let movie2 := Duration.mk 1 45
  let break2 := Duration.mk 0 20
  let movie3 := Duration.mk 2 10
  movie_marathon_end_time start movie1 break1 movie2 break2 movie3 = Time.mk 19 55  -- 7:55 p.m.
  := by sorry

end movie_marathon_end_time_correct_l259_25969


namespace twins_ratios_l259_25995

/-- Represents the family composition before and after the birth of twins -/
structure Family where
  initial_boys : ℕ
  initial_girls : ℕ
  k : ℚ
  t : ℚ

/-- The ratio of brothers to sisters for boys after the birth of twins -/
def boys_ratio (f : Family) : ℚ :=
  f.initial_boys / (f.initial_girls + 1)

/-- The ratio of brothers to sisters for girls after the birth of twins -/
def girls_ratio (f : Family) : ℚ :=
  (f.initial_boys + 1) / f.initial_girls

/-- Theorem stating the ratios after the birth of twins -/
theorem twins_ratios (f : Family) 
  (h1 : (f.initial_boys + 2) / f.initial_girls = f.k)
  (h2 : f.initial_boys / (f.initial_girls + 2) = f.t) :
  boys_ratio f = f.t ∧ girls_ratio f = f.k := by
  sorry

#check twins_ratios

end twins_ratios_l259_25995


namespace inequality_theorem_l259_25993

/-- A function satisfying the given conditions -/
def SatisfiesConditions (f : ℝ → ℝ) : Prop :=
  (∀ x, DifferentiableAt ℝ f x) ∧
  (∀ x, DifferentiableAt ℝ (deriv f) x) ∧
  f 0 = 1 ∧
  deriv f 0 = 0 ∧
  ∀ x ≥ 0, deriv (deriv f) x - 5 * deriv f x + 6 * f x ≥ 0

/-- The main theorem -/
theorem inequality_theorem (f : ℝ → ℝ) (h : SatisfiesConditions f) :
  ∀ x ≥ 0, f x ≥ 3 * Real.exp (2 * x) - 2 * Real.exp (3 * x) := by
  sorry

end inequality_theorem_l259_25993


namespace summer_jolly_degree_difference_l259_25935

theorem summer_jolly_degree_difference :
  ∀ (summer_degrees jolly_degrees : ℕ),
    summer_degrees = 150 →
    summer_degrees + jolly_degrees = 295 →
    summer_degrees - jolly_degrees = 5 :=
by
  sorry

end summer_jolly_degree_difference_l259_25935


namespace solve_factorial_equation_l259_25927

def factorial (n : ℕ) : ℕ := Nat.factorial n

theorem solve_factorial_equation : ∃ n : ℕ, n * factorial n + factorial n = 720 ∧ n = 5 := by
  sorry

end solve_factorial_equation_l259_25927


namespace polynomial_equality_l259_25958

-- Define the polynomials P and Q
variable (P Q : ℝ → ℝ)

-- Define the property of being a nonconstant polynomial
def IsNonconstantPolynomial (f : ℝ → ℝ) : Prop := sorry

-- Define the theorem
theorem polynomial_equality
  (hP : IsNonconstantPolynomial P)
  (hQ : IsNonconstantPolynomial Q)
  (h : ∀ y : ℝ, ⌊P y⌋ = ⌊Q y⌋) :
  ∀ x : ℝ, P x = Q x := by sorry

end polynomial_equality_l259_25958


namespace arithmetic_sequence_count_l259_25906

theorem arithmetic_sequence_count :
  let a : ℤ := -5  -- First term
  let l : ℤ := 85  -- Last term
  let d : ℤ := 5   -- Common difference
  (l - a) / d + 1 = 19
  :=
by
  sorry

end arithmetic_sequence_count_l259_25906


namespace dinner_bill_friends_l259_25919

theorem dinner_bill_friends (total_bill : ℝ) (silas_payment : ℝ) (one_friend_payment : ℝ) : 
  total_bill = 150 →
  silas_payment = total_bill / 2 →
  one_friend_payment = 18 →
  ∃ (num_friends : ℕ),
    num_friends = 6 ∧
    (num_friends - 1) * one_friend_payment = (total_bill - silas_payment) * 1.1 :=
by sorry

end dinner_bill_friends_l259_25919


namespace jerry_collection_cost_l259_25929

/-- The amount of money Jerry needs to finish his action figure collection. -/
def money_needed (current : ℕ) (total : ℕ) (cost : ℕ) : ℕ :=
  (total - current) * cost

/-- Theorem stating the amount Jerry needs to finish his collection. -/
theorem jerry_collection_cost :
  money_needed 7 25 12 = 216 := by
  sorry

end jerry_collection_cost_l259_25929


namespace n_equals_ten_l259_25934

/-- The number of sides in a regular polygon satisfying the given condition -/
def n : ℕ := sorry

/-- The measure of the internal angle in a regular polygon with k sides -/
def internal_angle (k : ℕ) : ℚ := (k - 2) * 180 / k

/-- The condition that the internal angle of an n-sided polygon is 12° less
    than that of a polygon with n/4 fewer sides -/
axiom angle_condition : internal_angle n = internal_angle (3 * n / 4) - 12

/-- Theorem stating that n = 10 -/
theorem n_equals_ten : n = 10 := by sorry

end n_equals_ten_l259_25934


namespace joyce_apples_l259_25943

theorem joyce_apples (initial : Real) (received : Real) : 
  initial = 75.0 → received = 52.0 → initial + received = 127.0 := by
  sorry

end joyce_apples_l259_25943


namespace cleanup_drive_total_l259_25990

/-- The total amount of garbage collected by two groups, given the amount collected by one group
    and the difference between the two groups' collections. -/
def totalGarbageCollected (group1Amount : ℕ) (difference : ℕ) : ℕ :=
  group1Amount + (group1Amount - difference)

/-- Theorem stating that given Lizzie's group collected 387 pounds of garbage and another group
    collected 39 pounds less, the total amount of garbage collected by both groups is 735 pounds. -/
theorem cleanup_drive_total :
  totalGarbageCollected 387 39 = 735 := by
  sorry

end cleanup_drive_total_l259_25990


namespace polynomial_sum_l259_25950

theorem polynomial_sum (a b c d : ℤ) : 
  (∀ x, (x^2 + a*x + b) * (x^2 + c*x + d) = x^4 + 2*x^3 - 5*x^2 + 8*x - 12) → 
  a + b + c + d = 6 := by
sorry

end polynomial_sum_l259_25950


namespace sum_of_squares_not_perfect_square_l259_25903

def sum_of_squares (n : ℕ) (a : ℕ) : ℕ := 
  (2*n + 1) * a^2 + (2*n*(n+1)*(2*n+1)) / 3

def is_perfect_square (m : ℕ) : Prop :=
  ∃ k : ℕ, m = k^2

theorem sum_of_squares_not_perfect_square (n : ℕ) (h : n = 1 ∨ n = 2 ∨ n = 3 ∨ n = 4) :
  ∀ a : ℕ, ¬(is_perfect_square (sum_of_squares n a)) := by
  sorry

end sum_of_squares_not_perfect_square_l259_25903


namespace necessary_not_sufficient_condition_l259_25989

theorem necessary_not_sufficient_condition (x y : ℝ) (hx : x > 0) (hy : y > 0) :
  (∀ x y, x > 0 → y > 0 → x + y < 4 → x * y < 4) ∧
  (∃ x y, x > 0 ∧ y > 0 ∧ x * y < 4 ∧ x + y ≥ 4) := by
  sorry

end necessary_not_sufficient_condition_l259_25989


namespace parallelepiped_inequality_l259_25926

theorem parallelepiped_inequality (a b c d : ℝ) 
  (h_pos : a > 0 ∧ b > 0 ∧ c > 0) 
  (h_diagonal : d^2 = a^2 + b^2 + c^2) : 
  a^2 + b^2 + c^2 ≥ d^2 / 3 := by
sorry

end parallelepiped_inequality_l259_25926


namespace intersection_point_x_coordinate_l259_25901

theorem intersection_point_x_coordinate (x y : ℝ) : 
  y = 3 * x + 4 ∧ 5 * x - y = 41 → x = 22.5 := by sorry

end intersection_point_x_coordinate_l259_25901


namespace pie_baking_difference_l259_25959

def alice_bake_time : ℕ := 5
def bob_bake_time : ℕ := 6
def charlie_bake_time : ℕ := 7
def total_time : ℕ := 90

def pies_baked (bake_time : ℕ) : ℕ := total_time / bake_time

theorem pie_baking_difference :
  (pies_baked alice_bake_time) - (pies_baked bob_bake_time) + 
  (pies_baked alice_bake_time) - (pies_baked charlie_bake_time) = 9 := by
  sorry

end pie_baking_difference_l259_25959


namespace smallest_prime_with_composite_odd_digit_sum_l259_25900

/-- Sum of digits of a natural number -/
def sum_of_digits (n : ℕ) : ℕ := sorry

/-- Predicate for prime numbers -/
def is_prime (n : ℕ) : Prop := sorry

/-- Predicate for composite numbers -/
def is_composite (n : ℕ) : Prop := sorry

theorem smallest_prime_with_composite_odd_digit_sum :
  (is_prime 997) ∧ 
  (is_composite (sum_of_digits 997)) ∧ 
  (sum_of_digits 997 % 2 = 1) ∧
  (∀ p < 997, is_prime p → ¬(is_composite (sum_of_digits p) ∧ sum_of_digits p % 2 = 1)) :=
sorry

end smallest_prime_with_composite_odd_digit_sum_l259_25900


namespace windows_preference_l259_25981

theorem windows_preference (total : ℕ) (mac : ℕ) (no_pref : ℕ) : 
  total = 210 → 
  mac = 60 → 
  no_pref = 90 → 
  ∃ (windows : ℕ), windows = total - (mac + no_pref + mac / 3) ∧ windows = 40 := by
  sorry

#check windows_preference

end windows_preference_l259_25981


namespace range_of_a_l259_25937

-- Define the conditions
def sufficient_condition (x : ℝ) : Prop := -2 < x ∧ x < 4

def necessary_condition (x a : ℝ) : Prop := (x + 2) * (x - a) < 0

-- Define the theorem
theorem range_of_a : 
  (∀ x a : ℝ, sufficient_condition x → necessary_condition x a) ∧ 
  (∃ x a : ℝ, ¬sufficient_condition x ∧ necessary_condition x a) → 
  ∀ a : ℝ, (a ∈ Set.Ioi 4) ↔ (∃ x : ℝ, necessary_condition x a) :=
sorry

end range_of_a_l259_25937


namespace large_triangle_toothpicks_l259_25982

/-- The number of small triangles in the base row of the large triangle -/
def base_triangles : ℕ := 1001

/-- The total number of small triangles in the large triangle -/
def total_triangles : ℕ := base_triangles * (base_triangles + 1) / 2

/-- The number of toothpicks needed to construct the large triangle -/
def toothpicks : ℕ := (3 * total_triangles) / 2

theorem large_triangle_toothpicks :
  toothpicks = 752252 :=
sorry

end large_triangle_toothpicks_l259_25982


namespace laylas_score_l259_25992

theorem laylas_score (total : ℕ) (difference : ℕ) (laylas_score : ℕ) : 
  total = 112 → difference = 28 → laylas_score = 70 →
  ∃ (nahimas_score : ℕ), 
    nahimas_score + laylas_score = total ∧ 
    laylas_score = nahimas_score + difference :=
by
  sorry

end laylas_score_l259_25992


namespace hawks_victory_margin_l259_25924

/-- Calculates the total score for a team given their scoring details --/
def team_score (touchdowns extra_points two_point_conversions field_goals safeties : ℕ) : ℕ :=
  touchdowns * 6 + extra_points + two_point_conversions * 2 + field_goals * 3 + safeties * 2

/-- Represents the scoring details of the Hawks --/
def hawks_score : ℕ :=
  team_score 4 2 1 2 1

/-- Represents the scoring details of the Eagles --/
def eagles_score : ℕ :=
  team_score 3 3 1 3 1

/-- Theorem stating that the Hawks won by a margin of 2 points --/
theorem hawks_victory_margin :
  hawks_score - eagles_score = 2 :=
sorry

end hawks_victory_margin_l259_25924


namespace cafe_bill_difference_l259_25991

theorem cafe_bill_difference (amy_tip beth_tip : ℝ) 
  (amy_percentage beth_percentage : ℝ) : 
  amy_tip = 4 →
  beth_tip = 5 →
  amy_percentage = 0.08 →
  beth_percentage = 0.10 →
  amy_tip = amy_percentage * (amy_tip / amy_percentage) →
  beth_tip = beth_percentage * (beth_tip / beth_percentage) →
  (amy_tip / amy_percentage) - (beth_tip / beth_percentage) = 0 := by
sorry

end cafe_bill_difference_l259_25991


namespace four_equal_area_volume_prisms_l259_25946

/-- A square prism with integer edge lengths where the surface area equals the volume. -/
structure EqualAreaVolumePrism where
  a : ℕ  -- length of the base
  b : ℕ  -- height of the prism
  h : 2 * a^2 + 4 * a * b = a^2 * b

/-- The set of all square prisms with integer edge lengths where the surface area equals the volume. -/
def allEqualAreaVolumePrisms : Set EqualAreaVolumePrism :=
  {p : EqualAreaVolumePrism | True}

/-- The theorem stating that there are only four square prisms with integer edge lengths
    where the surface area equals the volume. -/
theorem four_equal_area_volume_prisms :
  allEqualAreaVolumePrisms = {
    ⟨12, 3, by sorry⟩,
    ⟨8, 4, by sorry⟩,
    ⟨6, 6, by sorry⟩,
    ⟨5, 10, by sorry⟩
  } := by sorry

end four_equal_area_volume_prisms_l259_25946


namespace only_four_not_divide_98_l259_25902

theorem only_four_not_divide_98 :
  (∀ n ∈ ({2, 7, 14, 49} : Set Nat), 98 % n = 0) ∧ 98 % 4 ≠ 0 := by
  sorry

end only_four_not_divide_98_l259_25902


namespace odd_quadruple_composition_l259_25947

/-- A function g is odd if g(-x) = -g(x) for all x -/
def IsOdd (g : ℝ → ℝ) : Prop :=
  ∀ x, g (-x) = -g x

theorem odd_quadruple_composition (g : ℝ → ℝ) (h : IsOdd g) :
  IsOdd (fun x ↦ g (g (g (g x)))) := by
  sorry

end odd_quadruple_composition_l259_25947


namespace spinner_direction_l259_25939

-- Define the directions
inductive Direction
| North
| East
| South
| West

-- Define the rotation
def rotate (initial : Direction) (revolutions : ℚ) : Direction :=
  sorry

-- Theorem statement
theorem spinner_direction (initial : Direction) 
  (clockwise : ℚ) (counterclockwise : ℚ) :
  initial = Direction.North ∧ 
  clockwise = 7/2 ∧ 
  counterclockwise = 17/4 →
  rotate (rotate initial clockwise) (-counterclockwise) = Direction.East :=
sorry

end spinner_direction_l259_25939


namespace symmetry_line_l259_25907

-- Define the circles
def circle1 (x y : ℝ) : Prop := x^2 + y^2 = 4
def circle2 (x y : ℝ) : Prop := x^2 + y^2 - 4*x + 4*y + 4 = 0

-- Define the line
def line_l (x y : ℝ) : Prop := y = x - 2

-- Theorem statement
theorem symmetry_line :
  ∀ (x1 y1 x2 y2 : ℝ),
  circle1 x1 y1 → circle2 x2 y2 →
  ∃ (x y : ℝ), line_l x y ∧
  (x = (x1 + x2) / 2) ∧ (y = (y1 + y2) / 2) ∧
  ((x - x1)^2 + (y - y1)^2 = (x - x2)^2 + (y - y2)^2) :=
sorry

end symmetry_line_l259_25907


namespace tangent_circles_bc_length_l259_25977

/-- Two externally tangent circles with centers A and B -/
structure TangentCircles where
  A : ℝ × ℝ  -- Center of first circle
  B : ℝ × ℝ  -- Center of second circle
  radius_A : ℝ  -- Radius of first circle
  radius_B : ℝ  -- Radius of second circle
  externally_tangent : ‖A - B‖ = radius_A + radius_B

/-- A line tangent to both circles intersecting ray AB at point C -/
def tangent_line (tc : TangentCircles) (C : ℝ × ℝ) : Prop :=
  ∃ D E : ℝ × ℝ,
    ‖D - tc.A‖ = tc.radius_A ∧
    ‖E - tc.B‖ = tc.radius_B ∧
    (D - C) • (tc.A - C) = 0 ∧
    (E - C) • (tc.B - C) = 0 ∧
    (C - tc.A) • (tc.B - tc.A) ≥ 0

/-- The main theorem -/
theorem tangent_circles_bc_length 
  (tc : TangentCircles) 
  (hA : tc.radius_A = 7)
  (hB : tc.radius_B = 4)
  (C : ℝ × ℝ)
  (h_tangent : tangent_line tc C) :
  ‖C - tc.B‖ = 44 / 3 :=
sorry

end tangent_circles_bc_length_l259_25977


namespace continuity_at_three_l259_25905

def f (x : ℝ) : ℝ := 2 * x^2 - 4

theorem continuity_at_three :
  ∀ ε > 0, ∃ δ > 0, ∀ x, |x - 3| < δ → |f x - f 3| < ε :=
by sorry

end continuity_at_three_l259_25905


namespace inequality_proof_l259_25904

theorem inequality_proof (x y z : ℝ) 
  (h_pos_x : x > 0) (h_pos_y : y > 0) (h_pos_z : z > 0)
  (h_condition : x * y + y * z + z * x + 2 * x * y * z = 1) : 
  4 * x + y + z ≥ 2 := by
  sorry

end inequality_proof_l259_25904


namespace total_seashells_l259_25970

-- Define the number of seashells found by Sam
def sam_shells : ℕ := 18

-- Define the number of seashells found by Mary
def mary_shells : ℕ := 47

-- Theorem stating the total number of seashells found
theorem total_seashells : sam_shells + mary_shells = 65 := by
  sorry

end total_seashells_l259_25970


namespace simple_interest_problem_l259_25954

theorem simple_interest_problem (P : ℝ) : 
  P * 0.08 * 3 = 0.5 * 4000 * ((1 + 0.10)^2 - 1) ↔ P = 1750 :=
by sorry

end simple_interest_problem_l259_25954


namespace function_inequality_l259_25917

open Real

theorem function_inequality (f : ℝ → ℝ) (h : Differentiable ℝ f) 
  (h1 : ∀ x, deriv f x < f x) : 
  f 1 < ℯ * f 0 ∧ f 2014 < ℯ^2014 * f 0 := by
  sorry

end function_inequality_l259_25917


namespace square_root_fraction_equals_one_l259_25960

theorem square_root_fraction_equals_one : 
  Real.sqrt (3^2 + 4^2) / Real.sqrt (20 + 5) = 1 := by sorry

end square_root_fraction_equals_one_l259_25960


namespace a_2017_is_one_sixty_fifth_l259_25921

/-- Represents a proper fraction -/
structure ProperFraction where
  numerator : Nat
  denominator : Nat
  is_proper : numerator < denominator

/-- The sequence of proper fractions -/
def fraction_sequence : Nat → ProperFraction := sorry

/-- The 2017th term of the sequence -/
def a_2017 : ProperFraction := fraction_sequence 2017

/-- Theorem stating that the 2017th term is 1/65 -/
theorem a_2017_is_one_sixty_fifth : 
  a_2017.numerator = 1 ∧ a_2017.denominator = 65 := by sorry

end a_2017_is_one_sixty_fifth_l259_25921


namespace nicky_received_card_value_l259_25908

/-- The value of a card Nicky received in a trade, given the value of cards he traded and his profit -/
def card_value (traded_card_value : ℕ) (num_traded_cards : ℕ) (profit : ℕ) : ℕ :=
  traded_card_value * num_traded_cards + profit

/-- Theorem stating the value of the card Nicky received from Jill -/
theorem nicky_received_card_value :
  card_value 8 2 5 = 21 := by
  sorry

end nicky_received_card_value_l259_25908


namespace jake_peaches_l259_25941

/-- Given the number of peaches Steven, Jill, and Jake have, prove that Jake has 8 peaches. -/
theorem jake_peaches (steven jill jake : ℕ) 
  (h1 : steven = 15)
  (h2 : steven = jill + 14)
  (h3 : jake + 7 = steven) : 
  jake = 8 := by
sorry

end jake_peaches_l259_25941


namespace mans_rowing_speed_l259_25953

/-- The rowing speed of a man in still water, given his speeds with and against the stream -/
theorem mans_rowing_speed (speed_with_stream speed_against_stream : ℝ) 
  (h1 : speed_with_stream = 18)
  (h2 : speed_against_stream = 8) :
  (speed_with_stream + speed_against_stream) / 2 = 13 :=
by sorry

end mans_rowing_speed_l259_25953


namespace smallest_cube_root_with_small_remainder_l259_25944

theorem smallest_cube_root_with_small_remainder : 
  ∃ (n : ℕ), n > 0 ∧ 
  (∀ (m : ℕ) (r : ℝ), 0 < r ∧ r < 1/1000 ∧ (m : ℝ)^(1/3) = n + r →
    ∀ (k : ℕ) (s : ℝ), 0 < s ∧ s < 1/1000 ∧ (k : ℝ)^(1/3) = (n-1) + s → k > m) ∧
  n = 19 :=
sorry

end smallest_cube_root_with_small_remainder_l259_25944


namespace percentage_difference_l259_25957

theorem percentage_difference (p t j : ℝ) 
  (h1 : j = 0.75 * p) 
  (h2 : j = 0.8 * t) : 
  t = (15/16) * p := by
  sorry

end percentage_difference_l259_25957


namespace solve_equation_l259_25936

theorem solve_equation : ∃ x : ℚ, (3 * x + 5) / 7 = 13 ∧ x = 86 / 3 := by
  sorry

end solve_equation_l259_25936


namespace equation_solution_l259_25987

theorem equation_solution (x : ℝ) : 2 * x^2 + 9 = (4 - x)^2 ↔ x = 4 + Real.sqrt 23 ∨ x = 4 - Real.sqrt 23 := by
  sorry

end equation_solution_l259_25987


namespace point_O_is_circumcenter_l259_25932

-- Define the necessary structures
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

structure Plane where
  a : ℝ
  b : ℝ
  c : ℝ
  d : ℝ

-- Define the triangle ABC
def Triangle (A B C : Point3D) : Prop := True

-- Define the plane α containing the triangle
def PlaneContainsTriangle (α : Plane) (A B C : Point3D) : Prop := True

-- Define a point being outside a plane
def PointOutsidePlane (P : Point3D) (α : Plane) : Prop := True

-- Define perpendicularity between a line and a plane
def PerpendicularToPlane (P O : Point3D) (α : Plane) : Prop := True

-- Define the foot of a perpendicular
def FootOfPerpendicular (O : Point3D) (P : Point3D) (α : Plane) : Prop := True

-- Define equality of distances
def EqualDistances (P A B C : Point3D) : Prop := True

-- Define circumcenter
def Circumcenter (O : Point3D) (A B C : Point3D) : Prop := True

theorem point_O_is_circumcenter 
  (α : Plane) (A B C P O : Point3D)
  (h1 : Triangle A B C)
  (h2 : PlaneContainsTriangle α A B C)
  (h3 : PointOutsidePlane P α)
  (h4 : PerpendicularToPlane P O α)
  (h5 : FootOfPerpendicular O P α)
  (h6 : EqualDistances P A B C) :
  Circumcenter O A B C := by
  sorry


end point_O_is_circumcenter_l259_25932


namespace matthew_crackers_l259_25974

theorem matthew_crackers (initial : ℕ) (friends : ℕ) (given_each : ℕ) (left : ℕ) : 
  friends = 3 → given_each = 7 → left = 17 → 
  initial = friends * given_each + left → initial = 38 :=
by sorry

end matthew_crackers_l259_25974


namespace consecutive_even_numbers_product_l259_25918

theorem consecutive_even_numbers_product (x : ℤ) : 
  (x % 2 = 0) →  -- x is even
  ((x + 2) % 2 = 0) →  -- x + 2 is even (consecutive even number)
  (x * (x + 2) = 224) →  -- their product is 224
  x * (x + 2) = 224 :=
by
  sorry

end consecutive_even_numbers_product_l259_25918


namespace optimal_price_reduction_l259_25988

/-- Represents the price reduction and sales model of a shopping mall. -/
structure MallSalesModel where
  initial_cost : ℝ
  initial_price : ℝ
  initial_sales : ℝ
  sales_increase_rate : ℝ

/-- Calculates the daily profit given a price reduction. -/
def daily_profit (model : MallSalesModel) (price_reduction : ℝ) : ℝ :=
  let new_sales := model.initial_sales + model.sales_increase_rate * price_reduction
  let new_profit_per_item := (model.initial_price - model.initial_cost) - price_reduction
  new_sales * new_profit_per_item

/-- Theorem stating that a price reduction of 30 yuan results in a daily profit of 3600 yuan. -/
theorem optimal_price_reduction (model : MallSalesModel) 
  (h1 : model.initial_cost = 220)
  (h2 : model.initial_price = 280)
  (h3 : model.initial_sales = 30)
  (h4 : model.sales_increase_rate = 3) :
  daily_profit model 30 = 3600 := by
  sorry

#check optimal_price_reduction

end optimal_price_reduction_l259_25988


namespace expression_simplification_l259_25914

theorem expression_simplification (x : ℝ) (h : x = Real.sqrt 3 - 2) :
  (1 + 1 / (x - 2)) * ((x^2 - 4) / (x - 1)) = Real.sqrt 3 := by
  sorry

end expression_simplification_l259_25914


namespace remainder_2013_div_85_l259_25948

theorem remainder_2013_div_85 : 2013 % 85 = 58 := by
  sorry

end remainder_2013_div_85_l259_25948


namespace equation_solution_l259_25966

theorem equation_solution :
  let f : ℝ → ℝ := λ x => (x^2 + 1)^2 - 4*(x^2 + 1) - 12
  ∀ x : ℝ, f x = 0 ↔ x = Real.sqrt 5 ∨ x = -Real.sqrt 5 := by
  sorry

end equation_solution_l259_25966


namespace ellipse_eccentricity_is_one_seventh_l259_25942

noncomputable def ellipse_eccentricity (a : ℝ) : ℝ :=
  let b := Real.sqrt 3
  let c := (1 : ℝ) / 4
  c / a

theorem ellipse_eccentricity_is_one_seventh :
  ∃ a : ℝ, (a > 0) ∧ 
  ((1 : ℝ) / 4)^2 / a^2 + (0 : ℝ)^2 / 3 = 1 ∧
  ellipse_eccentricity a = (1 : ℝ) / 7 := by
sorry

end ellipse_eccentricity_is_one_seventh_l259_25942


namespace factorial_ratio_l259_25983

theorem factorial_ratio : Nat.factorial 50 / Nat.factorial 48 = 2450 := by
  sorry

end factorial_ratio_l259_25983


namespace percentage_problem_l259_25931

theorem percentage_problem (P : ℝ) : 
  0 ≤ P ∧ P ≤ 100 → P * 700 = (60 / 100) * 150 + 120 := by
  sorry

end percentage_problem_l259_25931


namespace arrangement_count_is_2880_l259_25912

/-- The number of ways to choose k items from n items without replacement and where order matters. -/
def permutations (n k : ℕ) : ℕ := sorry

/-- The number of ways to choose k items from n items without replacement and where order doesn't matter. -/
def combinations (n k : ℕ) : ℕ := sorry

/-- The number of different arrangements of 4 boys and 3 girls in a row, where exactly 2 of the 3 girls are adjacent. -/
def arrangement_count : ℕ := sorry

theorem arrangement_count_is_2880 : arrangement_count = 2880 := by sorry

end arrangement_count_is_2880_l259_25912


namespace smallest_prime_factor_in_C_l259_25964

def C : Set Nat := {33, 35, 36, 39, 41}

theorem smallest_prime_factor_in_C :
  ∃ (n : Nat), n ∈ C ∧ (∀ (m : Nat), m ∈ C → ∀ (p q : Nat), Prime p → Prime q → p ∣ n → q ∣ m → p ≤ q) ∧ n = 36 := by
  sorry

end smallest_prime_factor_in_C_l259_25964


namespace select_cards_probability_l259_25928

-- Define the total number of cards
def total_cards : ℕ := 12

-- Define the number of cards for Alex's name
def alex_cards : ℕ := 4

-- Define the number of cards for Jamie's name
def jamie_cards : ℕ := 8

-- Define the number of cards to be selected
def selected_cards : ℕ := 3

-- Define the probability of selecting 2 cards from Alex's name and 1 from Jamie's name
def probability : ℚ := 12 / 55

-- Theorem statement
theorem select_cards_probability :
  (Nat.choose alex_cards 2 * Nat.choose jamie_cards 1) / Nat.choose total_cards selected_cards = probability :=
sorry

end select_cards_probability_l259_25928


namespace apple_distribution_l259_25996

theorem apple_distribution (total_apples : ℕ) (min_apples : ℕ) (num_people : ℕ) : 
  total_apples = 30 → min_apples = 3 → num_people = 3 →
  (Nat.choose (total_apples - num_people * min_apples + num_people - 1) (num_people - 1)) = 253 := by
sorry

end apple_distribution_l259_25996


namespace contradiction_method_correctness_l259_25913

theorem contradiction_method_correctness :
  (∀ p q : ℝ, (p^3 + q^3 = 2) → (¬(p + q ≤ 2) ↔ p + q > 2)) ∧
  (∀ a b : ℝ, (|a| + |b| < 1) →
    (∃ x₁ : ℝ, x₁^2 + a*x₁ + b = 0 ∧ |x₁| ≥ 1) →
    False) :=
sorry

end contradiction_method_correctness_l259_25913


namespace strawberry_jam_earnings_l259_25984

/-- Represents the number of strawberries picked by each person and the jam-making process. -/
structure StrawberryPicking where
  betty : ℕ
  matthew : ℕ
  natalie : ℕ
  strawberries_per_jar : ℕ
  price_per_jar : ℕ

/-- Calculates the total money earned from selling jam made from picked strawberries. -/
def total_money_earned (sp : StrawberryPicking) : ℕ :=
  let total_strawberries := sp.betty + sp.matthew + sp.natalie
  let jars_of_jam := total_strawberries / sp.strawberries_per_jar
  jars_of_jam * sp.price_per_jar

/-- Theorem stating that under the given conditions, the total money earned is $40. -/
theorem strawberry_jam_earnings : ∀ (sp : StrawberryPicking),
  sp.betty = 16 ∧
  sp.matthew = sp.betty + 20 ∧
  sp.matthew = 2 * sp.natalie ∧
  sp.strawberries_per_jar = 7 ∧
  sp.price_per_jar = 4 →
  total_money_earned sp = 40 :=
by
  sorry


end strawberry_jam_earnings_l259_25984


namespace josh_marbles_l259_25979

theorem josh_marbles (lost : ℕ) (left : ℕ) (initial : ℕ) : 
  lost = 7 → left = 9 → initial = lost + left → initial = 16 := by
  sorry

end josh_marbles_l259_25979


namespace chess_club_mixed_groups_l259_25963

theorem chess_club_mixed_groups 
  (total_children : Nat) 
  (total_groups : Nat) 
  (group_size : Nat)
  (boy_games : Nat)
  (girl_games : Nat) :
  total_children = 90 →
  total_groups = 30 →
  group_size = 3 →
  boy_games = 30 →
  girl_games = 14 →
  (∃ (mixed_groups : Nat), 
    mixed_groups = 23 ∧ 
    mixed_groups * 2 = total_children - boy_games - girl_games) := by
  sorry

end chess_club_mixed_groups_l259_25963


namespace final_price_is_12_l259_25994

/-- The price of a set consisting of one cup of coffee and one piece of cheesecake,
    with a 25% discount when bought together. -/
def discounted_set_price (coffee_price cheesecake_price : ℚ) (discount_rate : ℚ) : ℚ :=
  (coffee_price + cheesecake_price) * (1 - discount_rate)

/-- Theorem stating that the final price of the set is $12 -/
theorem final_price_is_12 :
  discounted_set_price 6 10 (1/4) = 12 := by
  sorry

end final_price_is_12_l259_25994


namespace johns_remaining_money_l259_25976

/-- The amount of money John has left after buying pizzas and sodas -/
theorem johns_remaining_money (q : ℝ) : 
  (3 : ℝ) * (2 : ℝ) * q + -- cost of small pizzas
  (2 : ℝ) * (3 : ℝ) * q + -- cost of medium pizzas
  (4 : ℝ) * q             -- cost of sodas
  ≤ (50 : ℝ) →
  (50 : ℝ) - ((3 : ℝ) * (2 : ℝ) * q + (2 : ℝ) * (3 : ℝ) * q + (4 : ℝ) * q) = 50 - 16 * q :=
by sorry

end johns_remaining_money_l259_25976


namespace collinearity_condition_l259_25973

-- Define a triangle ABC
structure Triangle :=
  (A B C : ℝ × ℝ)

-- Define points A₁, B₁, C₁
variables (A₁ B₁ C₁ : ℝ × ℝ)

-- Define the R value as in Problem 191
def R (ABC : Triangle) (A₁ B₁ C₁ : ℝ × ℝ) : ℝ := sorry

-- Define a function to check if a point is on a side of the triangle
def onSide (ABC : Triangle) (P : ℝ × ℝ) : Bool := sorry

-- Define a function to count how many points are on the sides of the triangle
def countOnSides (ABC : Triangle) (A₁ B₁ C₁ : ℝ × ℝ) : Nat := sorry

-- Define collinearity
def collinear (A₁ B₁ C₁ : ℝ × ℝ) : Prop := sorry

-- The main theorem
theorem collinearity_condition (ABC : Triangle) (A₁ B₁ C₁ : ℝ × ℝ) :
  collinear A₁ B₁ C₁ ↔ R ABC A₁ B₁ C₁ = 1 ∧ Even (countOnSides ABC A₁ B₁ C₁) :=
sorry

end collinearity_condition_l259_25973


namespace our_parabola_properties_l259_25949

/-- A parabola with specific properties -/
structure Parabola where
  -- The equation of the parabola in the form ax^2 + bxy + cy^2 + dx + ey + f = 0
  a : ℤ
  b : ℤ
  c : ℤ
  d : ℤ
  e : ℤ
  f : ℤ
  c_pos : c > 0
  gcd_one : Nat.gcd (Int.natAbs a) (Nat.gcd (Int.natAbs b) (Nat.gcd (Int.natAbs c) (Nat.gcd (Int.natAbs d) (Nat.gcd (Int.natAbs e) (Int.natAbs f))))) = 1

/-- The specific parabola we are interested in -/
def our_parabola : Parabola where
  a := 0
  b := 0
  c := 1
  d := -8
  e := -8
  f := 16
  c_pos := by sorry
  gcd_one := by sorry

/-- Theorem stating that our_parabola satisfies all the required properties -/
theorem our_parabola_properties :
  -- Passes through (2,8)
  (2 : ℝ)^2 * our_parabola.a + 2 * 8 * our_parabola.b + 8^2 * our_parabola.c + 2 * our_parabola.d + 8 * our_parabola.e + our_parabola.f = 0 ∧
  -- Vertex lies on y-axis (x-coordinate of vertex is 0)
  our_parabola.b^2 - 4 * our_parabola.a * our_parabola.c = 0 ∧
  -- y-coordinate of focus is 4
  (our_parabola.e^2 - 4 * our_parabola.c * our_parabola.f) / (4 * our_parabola.c^2) = 4 ∧
  -- Axis of symmetry is parallel to x-axis
  our_parabola.b = 0 ∧ our_parabola.a = 0 := by
  sorry

end our_parabola_properties_l259_25949


namespace system_I_solution_system_II_solution_l259_25945

-- System I
theorem system_I_solution :
  ∃ (x y : ℝ), (y = x + 3 ∧ x - 2*y + 12 = 0) → (x = 6 ∧ y = 9) := by sorry

-- System II
theorem system_II_solution :
  ∃ (x y : ℝ), (4*(x - y - 1) = 3*(1 - y) - 2 ∧ x/2 + y/3 = 2) → (x = 2 ∧ y = 3) := by sorry

end system_I_solution_system_II_solution_l259_25945


namespace rope_length_ratio_l259_25961

/-- Given three ropes with lengths A, B, and C, where A is the longest, B is the middle, and C is the shortest,
    if A + C = B + 100 and C = 80, then the ratio of their lengths is (B + 20):B:80. -/
theorem rope_length_ratio (A B C : ℕ) (h1 : A ≥ B) (h2 : B ≥ C) (h3 : A + C = B + 100) (h4 : C = 80) :
  ∃ (k : ℕ), k > 0 ∧ A = k * (B + 20) ∧ B = k * B ∧ C = k * 80 :=
sorry

end rope_length_ratio_l259_25961


namespace pentagon_area_l259_25938

/-- The area of a pentagon formed by an equilateral triangle sharing a side with a square -/
theorem pentagon_area (s : ℝ) (h_perimeter : 5 * s = 20) : 
  s^2 + (s^2 * Real.sqrt 3) / 4 = 16 + 4 * Real.sqrt 3 := by
  sorry

#check pentagon_area

end pentagon_area_l259_25938


namespace range_of_a_l259_25916

-- Define the piecewise function f
noncomputable def f (a : ℝ) (x : ℝ) : ℝ :=
  if x < 1 then (3*a - 1)*x + 4*a else Real.log x / Real.log a

-- State the theorem
theorem range_of_a (a : ℝ) :
  (0 < a ∧ a < 1) ∧ 
  (∀ x y : ℝ, x < y → f a x > f a y) →
  1/7 ≤ a ∧ a < 1/3 :=
by sorry

end range_of_a_l259_25916


namespace time_to_destination_l259_25968

/-- The time it takes to reach a destination given relative speeds and distances -/
theorem time_to_destination
  (your_speed : ℝ)
  (harris_speed : ℝ)
  (harris_time : ℝ)
  (distance_ratio : ℝ)
  (h1 : your_speed = 3 * harris_speed)
  (h2 : harris_time = 3)
  (h3 : distance_ratio = 5) :
  your_speed * (distance_ratio * harris_time / your_speed) = 5 :=
by
  sorry

end time_to_destination_l259_25968


namespace mean_proportion_of_3_and_4_l259_25975

theorem mean_proportion_of_3_and_4 :
  ∃ x : ℝ, (3 : ℝ) / x = x / 4 ∧ (x = 2 * Real.sqrt 3 ∨ x = -2 * Real.sqrt 3) :=
by sorry

end mean_proportion_of_3_and_4_l259_25975


namespace cubic_equation_solutions_mean_l259_25933

theorem cubic_equation_solutions_mean (x : ℝ) : 
  x^3 + 2*x^2 - 8*x - 4 = 0 → 
  ∃ (s : Finset ℝ), (∀ y ∈ s, y^3 + 2*y^2 - 8*y - 4 = 0) ∧ 
                    (s.card = 3) ∧ 
                    ((s.sum id) / s.card = -2/3) :=
sorry

end cubic_equation_solutions_mean_l259_25933


namespace hiking_rate_up_l259_25999

/-- Hiking problem statement -/
theorem hiking_rate_up (total_time : ℝ) (time_up : ℝ) (rate_down : ℝ) :
  total_time = 3 →
  time_up = 1.2 →
  rate_down = 6 →
  ∃ (rate_up : ℝ), rate_up = 9 ∧ rate_up * time_up = rate_down * (total_time - time_up) :=
by
  sorry

end hiking_rate_up_l259_25999


namespace ratio_of_divisor_sums_l259_25972

def M : ℕ := 36 * 36 * 65 * 275

def sum_odd_divisors (n : ℕ) : ℕ := sorry
def sum_even_divisors (n : ℕ) : ℕ := sorry

theorem ratio_of_divisor_sums :
  (sum_odd_divisors M : ℚ) / (sum_even_divisors M : ℚ) = 1 / 30 := by sorry

end ratio_of_divisor_sums_l259_25972
