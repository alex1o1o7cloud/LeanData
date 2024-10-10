import Mathlib

namespace average_of_multiples_10_to_100_l3494_349474

def multiples_of_10 : List ℕ := [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

def average (list : List ℕ) : ℚ :=
  (list.sum : ℚ) / list.length

theorem average_of_multiples_10_to_100 : 
  average multiples_of_10 = 55 := by
  sorry

end average_of_multiples_10_to_100_l3494_349474


namespace D_300_l3494_349487

/-- D(n) represents the number of ways to write a positive integer n as a product of integers greater than 1, where the order matters. -/
def D (n : ℕ+) : ℕ := sorry

/-- The prime factorization of 300 -/
def primeFactor300 : List ℕ+ := [2, 2, 3, 5, 5]

/-- Theorem: The number of ways to write 300 as a product of integers greater than 1, where the order matters, is 35. -/
theorem D_300 : D 300 = 35 := by sorry

end D_300_l3494_349487


namespace martha_cakes_l3494_349407

/-- The number of cakes Martha needs to buy -/
def total_cakes (num_children : Float) (cakes_per_child : Float) : Float :=
  num_children * cakes_per_child

/-- Theorem: Martha needs to buy 54 cakes -/
theorem martha_cakes : total_cakes 3.0 18.0 = 54.0 := by
  sorry

end martha_cakes_l3494_349407


namespace jim_grove_other_row_l3494_349494

/-- The number of lemons produced by a normal lemon tree per year -/
def normal_lemon_production : ℕ := 60

/-- The percentage increase in lemon production for Jim's engineered trees -/
def engineered_production_increase : ℚ := 50 / 100

/-- The number of trees in one row of Jim's grove -/
def trees_in_one_row : ℕ := 50

/-- The total number of lemons produced by Jim's grove in 5 years -/
def total_lemons_produced : ℕ := 675000

/-- The number of years of lemon production -/
def years_of_production : ℕ := 5

/-- The number of trees in the other row of Jim's grove -/
def trees_in_other_row : ℕ := 1450

theorem jim_grove_other_row :
  trees_in_other_row = 
    (total_lemons_produced / (normal_lemon_production * (1 + engineered_production_increase) * years_of_production)).floor - trees_in_one_row :=
by sorry

end jim_grove_other_row_l3494_349494


namespace line_equation_through_points_l3494_349416

/-- The equation 3y - 5x = 1 represents the line passing through points (-2, -3) and (4, 7) -/
theorem line_equation_through_points :
  let point1 : ℝ × ℝ := (-2, -3)
  let point2 : ℝ × ℝ := (4, 7)
  let line_eq (x y : ℝ) := 3 * y - 5 * x = 1
  (line_eq point1.1 point1.2 ∧ line_eq point2.1 point2.2) := by sorry

end line_equation_through_points_l3494_349416


namespace ten_player_tournament_matches_l3494_349408

/-- The number of matches in a round-robin tournament. -/
def num_matches (n : ℕ) : ℕ := n * (n - 1) / 2

/-- Theorem: A 10-player round-robin tournament has 45 matches. -/
theorem ten_player_tournament_matches : num_matches 10 = 45 := by
  sorry

end ten_player_tournament_matches_l3494_349408


namespace jane_dolls_l3494_349414

theorem jane_dolls (total : ℕ) (difference : ℕ) : 
  total = 32 → difference = 6 → ∃ (jane : ℕ), jane + (jane + difference) = total ∧ jane = 13 := by
sorry

end jane_dolls_l3494_349414


namespace husk_consumption_rate_l3494_349495

/-- Given that 20 cows eat 20 bags of husk in 20 days, prove that 1 cow will eat 1 bag of husk in 20 days -/
theorem husk_consumption_rate (cows bags days : ℕ) (h1 : cows = 20) (h2 : bags = 20) (h3 : days = 20) :
  (1 : ℚ) / cows * bags * days = 20 := by
  sorry

end husk_consumption_rate_l3494_349495


namespace sum_of_cubes_l3494_349450

theorem sum_of_cubes : (3 + 9)^3 + (3^3 + 9^3) = 2484 := by
  sorry

end sum_of_cubes_l3494_349450


namespace gcd_product_l3494_349447

theorem gcd_product (a b a' b' : ℕ+) (d d' : ℕ+) 
  (h1 : d = Nat.gcd a b) (h2 : d' = Nat.gcd a' b') : 
  Nat.gcd (a * a') (Nat.gcd (a * b') (Nat.gcd (b * a') (b * b'))) = d * d' := by
  sorry

end gcd_product_l3494_349447


namespace sandy_siding_cost_l3494_349454

-- Define the dimensions and costs
def wall_width : ℝ := 10
def wall_height : ℝ := 8
def roof_base : ℝ := 10
def roof_height : ℝ := 6
def siding_section_size : ℝ := 100  -- 10 ft x 10 ft = 100 sq ft
def siding_section_cost : ℝ := 30

-- Theorem to prove
theorem sandy_siding_cost :
  let wall_area := wall_width * wall_height
  let roof_area := roof_base * roof_height
  let total_area := wall_area + roof_area
  let sections_needed := ⌈total_area / siding_section_size⌉
  let total_cost := sections_needed * siding_section_cost
  total_cost = 60 :=
by sorry

end sandy_siding_cost_l3494_349454


namespace distribute_five_books_four_bags_l3494_349443

/-- The number of ways to distribute n distinct objects into k identical containers --/
def distribute (n k : ℕ) : ℕ := sorry

/-- The main theorem stating that there are 41 ways to distribute 5 books into 4 bags --/
theorem distribute_five_books_four_bags : distribute 5 4 = 41 := by sorry

end distribute_five_books_four_bags_l3494_349443


namespace certain_number_calculation_l3494_349471

theorem certain_number_calculation (x y : ℝ) :
  x = 77.7 ∧ x = y + 0.11 * y → y = 77.7 / 1.11 := by
  sorry

end certain_number_calculation_l3494_349471


namespace staff_age_calculation_l3494_349455

theorem staff_age_calculation (num_students : ℕ) (student_avg_age : ℕ) (num_staff : ℕ) (age_increase : ℕ) :
  num_students = 50 →
  student_avg_age = 25 →
  num_staff = 5 →
  age_increase = 2 →
  (num_students * student_avg_age + num_staff * ((student_avg_age + age_increase) * (num_students + num_staff) - num_students * student_avg_age)) / num_staff = 235 := by
  sorry

end staff_age_calculation_l3494_349455


namespace absolute_value_equation_solution_l3494_349498

theorem absolute_value_equation_solution :
  ∃! x : ℝ, |x| = 2 * x + 1 :=
by
  -- The unique solution is x = -1/3
  use -1/3
  sorry

end absolute_value_equation_solution_l3494_349498


namespace arithmetic_sequence_sum_8_l3494_349458

def arithmetic_sequence (a : ℕ → ℤ) : Prop :=
  ∃ d : ℤ, ∀ n : ℕ, a (n + 1) = a n + d

def sum_of_arithmetic_sequence (a : ℕ → ℤ) (n : ℕ) : ℤ :=
  (n : ℤ) * a 1 + (n * (n - 1) : ℤ) / 2 * (a 2 - a 1)

theorem arithmetic_sequence_sum_8 (a : ℕ → ℤ) :
  arithmetic_sequence a →
  a 1 = -40 →
  a 6 + a 10 = -10 →
  sum_of_arithmetic_sequence a 8 = -180 := by
  sorry

end arithmetic_sequence_sum_8_l3494_349458


namespace arrange_40521_eq_96_l3494_349472

/-- The number of ways to arrange the digits of 40,521 to form a 5-digit number -/
def arrange_40521 : ℕ :=
  let digits : List ℕ := [4, 0, 5, 2, 1]
  let n : ℕ := digits.length
  let non_zero_digits : ℕ := (digits.filter (· ≠ 0)).length
  (n - 1) * Nat.factorial (n - 1)

theorem arrange_40521_eq_96 : arrange_40521 = 96 := by
  sorry

end arrange_40521_eq_96_l3494_349472


namespace f_odd_and_increasing_l3494_349440

def f (x : ℝ) : ℝ := x * abs x

theorem f_odd_and_increasing :
  (∀ x : ℝ, f (-x) = -f x) ∧ 
  (∀ x y : ℝ, x < y → f x < f y) := by
sorry

end f_odd_and_increasing_l3494_349440


namespace sum_not_prime_l3494_349499

theorem sum_not_prime (a b c d : ℕ) (h : a * b = c * d) : ¬ Nat.Prime (a + b + c + d) := by
  sorry

end sum_not_prime_l3494_349499


namespace ratio_and_closest_whole_number_l3494_349403

theorem ratio_and_closest_whole_number : 
  let ratio := (10^2010 + 10^2013) / (10^2011 + 10^2014)
  ratio = 1/10 ∧ 
  ∀ n : ℤ, |ratio - (n : ℚ)| ≥ |ratio - 0| :=
by sorry

end ratio_and_closest_whole_number_l3494_349403


namespace ski_camp_directions_l3494_349429

-- Define the four cardinal directions
inductive Direction
| North
| South
| East
| West

-- Define the four friends
inductive Friend
| Karel
| Mojmir
| Pepa
| Zdenda

-- Define a function that assigns a direction to each friend
def came_from : Friend → Direction := sorry

-- Define the statements made by each friend
def karel_statement : Prop :=
  came_from Friend.Karel ≠ Direction.North ∧ came_from Friend.Karel ≠ Direction.South

def mojmir_statement : Prop :=
  came_from Friend.Mojmir = Direction.South

def pepa_statement : Prop :=
  came_from Friend.Pepa = Direction.North

def zdenda_statement : Prop :=
  came_from Friend.Zdenda ≠ Direction.South

-- Define a function that checks if a statement is true
def is_true_statement : Friend → Prop
| Friend.Karel => karel_statement
| Friend.Mojmir => mojmir_statement
| Friend.Pepa => pepa_statement
| Friend.Zdenda => zdenda_statement

-- Theorem to prove
theorem ski_camp_directions :
  (∃! f : Friend, ¬is_true_statement f) ∧
  (came_from Friend.Zdenda = Direction.North) ∧
  (came_from Friend.Mojmir = Direction.South) ∧
  (¬is_true_statement Friend.Pepa) :=
by sorry

end ski_camp_directions_l3494_349429


namespace same_grade_percentage_l3494_349427

/-- Given a class of students who took two tests, this theorem proves
    the percentage of students who received the same grade on both tests. -/
theorem same_grade_percentage
  (total_students : ℕ)
  (same_grade_students : ℕ)
  (h1 : total_students = 30)
  (h2 : same_grade_students = 12) :
  (same_grade_students : ℚ) / total_students * 100 = 40 := by
  sorry

end same_grade_percentage_l3494_349427


namespace sqrt_t6_plus_t4_l3494_349497

theorem sqrt_t6_plus_t4 (t : ℝ) : Real.sqrt (t^6 + t^4) = t^2 * Real.sqrt (t^2 + 1) := by
  sorry

end sqrt_t6_plus_t4_l3494_349497


namespace total_consumption_30_days_l3494_349465

/-- Represents the daily food consumption for each dog -/
structure DogConsumption where
  a : Float
  b : Float
  c : Float
  d : Float
  e : Float

/-- Calculates the total daily consumption for all dogs -/
def totalDailyConsumption (dc : DogConsumption) : Float :=
  dc.a + dc.b + dc.c + dc.d + dc.e

/-- Represents the food consumption for each dog on Sundays -/
structure SundayConsumption where
  a : Float
  b : Float
  c : Float
  d : Float
  e : Float

/-- Calculates the total consumption for all dogs on a Sunday -/
def totalSundayConsumption (sc : SundayConsumption) : Float :=
  sc.a + sc.b + sc.c + sc.d + sc.e

/-- Theorem: Total dog food consumption over 30 days is 60 scoops -/
theorem total_consumption_30_days 
  (dc : DogConsumption)
  (sc : SundayConsumption)
  (h1 : dc.a = 0.125)
  (h2 : dc.b = 0.25)
  (h3 : dc.c = 0.375)
  (h4 : dc.d = 0.5)
  (h5 : dc.e = 0.75)
  (h6 : sc.a = dc.a)
  (h7 : sc.b = dc.b)
  (h8 : sc.c = dc.c + 0.1)
  (h9 : sc.d = dc.d)
  (h10 : sc.e = dc.e - 0.1)
  (h11 : totalDailyConsumption dc = totalSundayConsumption sc) :
  30 * totalDailyConsumption dc = 60 := by
  sorry


end total_consumption_30_days_l3494_349465


namespace courtyard_length_l3494_349493

/-- Proves that a courtyard with given dimensions and number of bricks has a specific length -/
theorem courtyard_length 
  (width : ℝ) 
  (brick_length : ℝ) 
  (brick_width : ℝ) 
  (num_bricks : ℕ) : 
  width = 18 → 
  brick_length = 0.2 → 
  brick_width = 0.1 → 
  num_bricks = 22500 → 
  (width * (num_bricks * brick_length * brick_width / width)) = 25 := by
  sorry

end courtyard_length_l3494_349493


namespace scientific_notation_equality_l3494_349449

theorem scientific_notation_equality : 284000000 = 2.84 * (10 ^ 8) := by
  sorry

end scientific_notation_equality_l3494_349449


namespace F_propagation_l3494_349483

-- Define F as a proposition on natural numbers
variable (F : ℕ → Prop)

-- State the theorem
theorem F_propagation (h1 : ∀ k : ℕ, k > 0 → (F k → F (k + 1)))
                      (h2 : ¬ F 7) :
  ¬ F 6 ∧ ¬ F 5 := by
  sorry

end F_propagation_l3494_349483


namespace prime_q_value_l3494_349461

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d ∣ n → d = 1 ∨ d = n

theorem prime_q_value (p : ℕ) (hp : is_prime p) :
  let q := 13 * p + 2
  is_prime q ∧ (q - 1) % 3 = 0 → q = 67 := by
  sorry

end prime_q_value_l3494_349461


namespace tiffany_pies_eaten_l3494_349432

theorem tiffany_pies_eaten (pies_per_day : ℕ) (days : ℕ) (cans_per_pie : ℕ) (remaining_cans : ℕ) : 
  pies_per_day = 3 → days = 11 → cans_per_pie = 2 → remaining_cans = 58 →
  (pies_per_day * days * cans_per_pie - remaining_cans) / cans_per_pie = 4 := by
sorry

end tiffany_pies_eaten_l3494_349432


namespace a_4_equals_8_l3494_349475

def a (n : ℕ) : ℤ := (-1)^n * (2 * n)

theorem a_4_equals_8 : a 4 = 8 := by sorry

end a_4_equals_8_l3494_349475


namespace unique_five_numbers_l3494_349457

theorem unique_five_numbers : 
  ∃! (a b c d e : ℕ), 
    a < b ∧ b < c ∧ c < d ∧ d < e ∧
    a * b > 25 ∧ d * e < 75 ∧
    a = 5 ∧ b = 6 ∧ c = 7 ∧ d = 8 ∧ e = 9 :=
by sorry

end unique_five_numbers_l3494_349457


namespace archer_weekly_expenditure_is_1056_l3494_349463

/-- The archer's weekly expenditure on arrows -/
def archer_weekly_expenditure (shots_per_day : ℕ) (days_per_week : ℕ) 
  (recovery_rate : ℚ) (arrow_cost : ℚ) (team_contribution_rate : ℚ) : ℚ :=
  let total_shots := shots_per_day * days_per_week
  let recovered_arrows := (total_shots : ℚ) * recovery_rate
  let arrows_used := (total_shots : ℚ) - recovered_arrows
  let total_cost := arrows_used * arrow_cost
  let team_contribution := total_cost * team_contribution_rate
  total_cost - team_contribution

/-- Theorem stating the archer's weekly expenditure on arrows -/
theorem archer_weekly_expenditure_is_1056 :
  archer_weekly_expenditure 200 4 (1/5) (11/2) (7/10) = 1056 := by
  sorry

end archer_weekly_expenditure_is_1056_l3494_349463


namespace equilateral_triangle_tiling_l3494_349490

theorem equilateral_triangle_tiling (m : ℕ) : 
  (∃ (t₁ t₂ : ℕ), 
    m = t₁ + t₂ ∧ 
    t₁ - t₂ = 5 ∧ 
    t₁ ≥ 5 ∧ 
    3 * t₁ + t₂ + 2 * (25 - t₁ - t₂) = 55) ↔ 
  (m % 2 = 1 ∧ m ≥ 5 ∧ m ≤ 25) :=
by sorry

end equilateral_triangle_tiling_l3494_349490


namespace max_value_of_f_l3494_349492

noncomputable def f (x : ℝ) : ℝ := Real.sin x ^ 2 + Real.sqrt 3 * Real.sin x * Real.cos x

theorem max_value_of_f :
  ∃ (M : ℝ), M = 3/2 ∧ ∀ (x : ℝ), f x ≤ M :=
sorry

end max_value_of_f_l3494_349492


namespace min_cuts_for_hendecagons_l3494_349436

/-- Represents a polygon on the table --/
structure Polygon :=
  (sides : ℕ)

/-- Represents the state of the table after some cuts --/
structure TableState :=
  (polygons : List Polygon)

/-- Performs a single straight cut on the table --/
def makeCut (state : TableState) : TableState :=
  sorry

/-- Checks if the table state contains at least 252 hendecagons --/
def hasEnoughHendecagons (state : TableState) : Prop :=
  (state.polygons.filter (λ p => p.sides = 11)).length ≥ 252

/-- The minimum number of cuts needed to create at least 252 hendecagons --/
def minCuts : ℕ := 2015

theorem min_cuts_for_hendecagons :
  ∀ (initialState : TableState),
    initialState.polygons = [Polygon.mk 4] →
    ∀ (n : ℕ),
      (∃ (finalState : TableState),
        (Nat.iterate makeCut n initialState = finalState) ∧
        hasEnoughHendecagons finalState) →
      n ≥ minCuts :=
sorry

end min_cuts_for_hendecagons_l3494_349436


namespace opposite_roots_n_value_l3494_349437

/-- Given a rational function equal to (n-2)/(n+2) with roots of opposite signs, prove n = 2b + 2 -/
theorem opposite_roots_n_value (b d p q n : ℝ) (x : ℝ → ℝ) :
  (∀ x, (x^2 - b*x + d) / (p*x - q) = (n - 2) / (n + 2)) →
  (∃ r : ℝ, x r = r ∧ x (-r) = -r) →
  p = b + 1 →
  n = 2*b + 2 := by
sorry

end opposite_roots_n_value_l3494_349437


namespace partition_modular_sum_l3494_349426

theorem partition_modular_sum (p : ℕ) (h_prime : Nat.Prime p) (h_p_ge_5 : p ≥ 5) :
  ∀ (A B C : Set ℕ), 
    (A ∪ B ∪ C = Finset.range (p - 1)) →
    (A ∩ B = ∅) → (B ∩ C = ∅) → (A ∩ C = ∅) →
    ∃ (x y z : ℕ), x ∈ A ∧ y ∈ B ∧ z ∈ C ∧ (x + y) % p = z % p :=
by sorry

end partition_modular_sum_l3494_349426


namespace rectangle_area_change_l3494_349446

/-- The new area of a rectangle after changing its dimensions -/
def new_area (original_area : ℝ) (length_increase : ℝ) (width_decrease : ℝ) : ℝ :=
  original_area * (1 + length_increase) * (1 - width_decrease)

theorem rectangle_area_change :
  new_area 432 0.2 0.1 = 466.56 := by
  sorry

end rectangle_area_change_l3494_349446


namespace car_departure_time_l3494_349468

/-- 
Given two cars A and B that start simultaneously from locations A and B respectively:
- They will meet at some point
- If Car A departs earlier, they will meet 30 minutes earlier
- Car A travels at 60 kilometers per hour
- Car B travels at 40 kilometers per hour

Prove that Car A needs to depart 50 minutes earlier for them to meet 30 minutes earlier.
-/
theorem car_departure_time (speed_A speed_B : ℝ) (meeting_time_diff : ℝ) :
  speed_A = 60 →
  speed_B = 40 →
  meeting_time_diff = 30 →
  ∃ (departure_time : ℝ), 
    departure_time = 50 ∧
    speed_A * (departure_time / 60) = speed_A * (meeting_time_diff / 60) + speed_B * (meeting_time_diff / 60) :=
by sorry

end car_departure_time_l3494_349468


namespace coin_problem_l3494_349434

theorem coin_problem (x : ℚ) (h : x > 0) : 
  let lost := (2 : ℚ) / 3 * x
  let recovered := (3 : ℚ) / 4 * lost
  x - (x - lost + recovered) = x / 6 := by sorry

end coin_problem_l3494_349434


namespace quadratic_function_range_l3494_349420

/-- Given a quadratic function f(x) = x^2 - 2x + 3 defined on [0,m], 
    if its maximum value on this interval is 3 and its minimum value is 2, 
    then m is in the closed interval [1,2] -/
theorem quadratic_function_range (m : ℝ) : 
  let f : ℝ → ℝ := λ x ↦ x^2 - 2*x + 3
  (∀ x ∈ Set.Icc 0 m, f x ≤ 3) ∧ 
  (∃ x ∈ Set.Icc 0 m, f x = 3) ∧
  (∀ x ∈ Set.Icc 0 m, f x ≥ 2) ∧ 
  (∃ x ∈ Set.Icc 0 m, f x = 2) →
  m ∈ Set.Icc 1 2 :=
by sorry


end quadratic_function_range_l3494_349420


namespace pure_imaginary_m_value_l3494_349469

/-- A complex number z is defined as z = (m^2+m-2) + (m^2+4m-5)i -/
def z (m : ℝ) : ℂ := Complex.mk (m^2 + m - 2) (m^2 + 4*m - 5)

/-- A complex number is pure imaginary if its real part is zero and imaginary part is non-zero -/
def is_pure_imaginary (z : ℂ) : Prop := z.re = 0 ∧ z.im ≠ 0

theorem pure_imaginary_m_value :
  ∃! m : ℝ, is_pure_imaginary (z m) ∧ m = -2 := by sorry

end pure_imaginary_m_value_l3494_349469


namespace shooting_scores_l3494_349422

def scores_A : List ℝ := [4, 5, 5, 6, 6, 7, 7, 8, 8, 9]
def scores_B : List ℝ := [2, 5, 6, 6, 7, 7, 7, 8, 9, 10]

def variance_A : ℝ := 2.25
def variance_B : ℝ := 4.41

theorem shooting_scores :
  let avg_A := (scores_A.sum) / scores_A.length
  let avg_B := (scores_B.sum) / scores_B.length
  let avg_all := ((scores_A ++ scores_B).sum) / (scores_A.length + scores_B.length)
  avg_A < avg_B ∧ avg_all = 6.6 := by
  sorry

end shooting_scores_l3494_349422


namespace jenny_sweets_division_l3494_349453

theorem jenny_sweets_division :
  ∃ n : ℕ, n ≠ 5 ∧ n ≠ 12 ∧ 30 % n = 0 :=
by
  sorry

end jenny_sweets_division_l3494_349453


namespace square_root_of_sixteen_l3494_349486

theorem square_root_of_sixteen (x : ℝ) : (x + 3) ^ 2 = 16 → x = 1 ∨ x = -7 := by
  sorry

end square_root_of_sixteen_l3494_349486


namespace bells_toll_together_l3494_349496

theorem bells_toll_together (bell_intervals : List ℕ := [13, 17, 21, 26, 34, 39]) : 
  Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm (Nat.lcm 13 17) 21) 26) 34) 39 = 9272 := by
  sorry

end bells_toll_together_l3494_349496


namespace knife_percentage_after_trade_l3494_349430

/-- Represents Carolyn's silverware set -/
structure SilverwareSet where
  knives : ℕ
  forks : ℕ
  spoons : ℕ

/-- Calculates the total number of pieces in a silverware set -/
def total (s : SilverwareSet) : ℕ := s.knives + s.forks + s.spoons

/-- Represents the trade operation -/
def trade (s : SilverwareSet) (knivesGained spoonsTrade : ℕ) : SilverwareSet :=
  { knives := s.knives + knivesGained,
    forks := s.forks,
    spoons := s.spoons - spoonsTrade }

/-- Calculates the percentage of knives in a silverware set -/
def knifePercentage (s : SilverwareSet) : ℚ :=
  (s.knives : ℚ) / (total s : ℚ) * 100

theorem knife_percentage_after_trade :
  let initialSet : SilverwareSet := { knives := 6, forks := 12, spoons := 6 * 3 }
  let finalSet := trade initialSet 10 6
  knifePercentage finalSet = 40 := by sorry

end knife_percentage_after_trade_l3494_349430


namespace employee_preference_city_y_l3494_349419

/-- Proves that given the conditions of the employee relocation problem,
    the percentage of employees preferring city Y is 40%. -/
theorem employee_preference_city_y (
  total_employees : ℕ)
  (relocated_to_x_percent : ℚ)
  (relocated_to_y_percent : ℚ)
  (max_preferred_relocation : ℕ)
  (h1 : total_employees = 200)
  (h2 : relocated_to_x_percent = 30 / 100)
  (h3 : relocated_to_y_percent = 70 / 100)
  (h4 : relocated_to_x_percent + relocated_to_y_percent = 1)
  (h5 : max_preferred_relocation = 140) :
  ∃ (prefer_y_percent : ℚ),
    prefer_y_percent = 40 / 100 ∧
    prefer_y_percent * total_employees = max_preferred_relocation - relocated_to_x_percent * total_employees :=
by sorry

end employee_preference_city_y_l3494_349419


namespace gcd_lcm_3869_6497_l3494_349425

theorem gcd_lcm_3869_6497 :
  (Nat.gcd 3869 6497 = 73) ∧
  (Nat.lcm 3869 6497 = 344341) := by
  sorry

end gcd_lcm_3869_6497_l3494_349425


namespace unique_number_with_specific_factors_l3494_349456

theorem unique_number_with_specific_factors :
  ∀ (x n : ℕ),
  x = 7^n + 1 →
  Odd n →
  (∃ (p q : ℕ), Prime p ∧ Prime q ∧ p ≠ q ∧ p ≠ 11 ∧ q ≠ 11 ∧ x = 2 * 11 * p * q) →
  x = 16808 := by
sorry

end unique_number_with_specific_factors_l3494_349456


namespace sock_problem_solution_l3494_349412

/-- Represents the number of pairs of socks at each price point --/
structure SockInventory where
  two_dollar : ℕ
  four_dollar : ℕ
  five_dollar : ℕ

/-- Calculates the total number of sock pairs --/
def total_pairs (s : SockInventory) : ℕ :=
  s.two_dollar + s.four_dollar + s.five_dollar

/-- Calculates the total cost of all socks --/
def total_cost (s : SockInventory) : ℕ :=
  2 * s.two_dollar + 4 * s.four_dollar + 5 * s.five_dollar

theorem sock_problem_solution :
  ∃ (s : SockInventory),
    total_pairs s = 15 ∧
    total_cost s = 41 ∧
    s.two_dollar ≥ 1 ∧
    s.four_dollar ≥ 1 ∧
    s.five_dollar ≥ 1 ∧
    s.two_dollar = 11 :=
by sorry

#check sock_problem_solution

end sock_problem_solution_l3494_349412


namespace final_short_oak_count_l3494_349482

/-- The number of short oak trees in the park after planting -/
def short_oak_trees_after_planting (current : ℕ) (to_plant : ℕ) : ℕ :=
  current + to_plant

/-- Theorem stating the number of short oak trees after planting -/
theorem final_short_oak_count :
  short_oak_trees_after_planting 3 9 = 12 := by
  sorry

end final_short_oak_count_l3494_349482


namespace a_8_equals_15_l3494_349404

-- Define the sequence S_n
def S (n : ℕ) : ℕ := n^2

-- Define the sequence a_n
def a (n : ℕ) : ℤ :=
  if n = 0 then 0
  else S n - S (n-1)

-- Theorem statement
theorem a_8_equals_15 : a 8 = 15 := by
  sorry

end a_8_equals_15_l3494_349404


namespace monotonic_quadratic_function_l3494_349489

/-- A function f(x) = ax² + x + 1 is monotonically increasing in the interval [-2, +∞) if and only if 0 ≤ a ≤ 1/4 -/
theorem monotonic_quadratic_function (a : ℝ) :
  (∀ x : ℝ, x ≥ -2 → Monotone (fun x => a * x^2 + x + 1)) ↔ 0 ≤ a ∧ a ≤ 1/4 := by
  sorry

end monotonic_quadratic_function_l3494_349489


namespace opposite_sign_power_l3494_349478

theorem opposite_sign_power (x y : ℝ) : 
  (|x + 3| + (y - 2)^2 = 0) → x^y = 9 := by
  sorry

end opposite_sign_power_l3494_349478


namespace rectangle_length_fraction_of_circle_radius_l3494_349445

theorem rectangle_length_fraction_of_circle_radius : 
  ∀ (square_area : ℝ) (rectangle_area : ℝ) (rectangle_breadth : ℝ),
  square_area = 900 →
  rectangle_area = 120 →
  rectangle_breadth = 10 →
  (rectangle_area / rectangle_breadth) / Real.sqrt square_area = 2 / 5 := by
  sorry

end rectangle_length_fraction_of_circle_radius_l3494_349445


namespace product_from_lcm_hcf_l3494_349405

/-- Given two positive integers with LCM 750 and HCF 25, prove their product is 18750 -/
theorem product_from_lcm_hcf (a b : ℕ+) : 
  Nat.lcm a b = 750 → Nat.gcd a b = 25 → a * b = 18750 := by sorry

end product_from_lcm_hcf_l3494_349405


namespace log_xyz_equals_one_l3494_349467

-- Define the logarithm function
noncomputable def log : ℝ → ℝ := Real.log

-- State the theorem
theorem log_xyz_equals_one 
  (x y z : ℝ) 
  (h1 : log (x^2 * y^2 * z) = 2) 
  (h2 : log (x * y * z^3) = 2) : 
  log (x * y * z) = 1 := by
  sorry

end log_xyz_equals_one_l3494_349467


namespace bank_profit_l3494_349402

/-- Bank's profit calculation -/
theorem bank_profit 
  (K : ℝ) (p p₁ : ℝ) (n : ℕ) 
  (h₁ : p₁ > p) 
  (h₂ : p > 0) 
  (h₃ : p₁ > 0) :
  K * ((1 + p₁ / 100) ^ n - (1 + p / 100) ^ n) = 
  K * ((1 + p₁ / 100) ^ n - (1 + p / 100) ^ n) :=
by sorry

end bank_profit_l3494_349402


namespace second_brand_growth_rate_l3494_349473

/-- Proves that the growth rate of the second brand of computers is approximately 0.7 million households per year -/
theorem second_brand_growth_rate (initial_first : ℝ) (growth_first : ℝ) (initial_second : ℝ) (time_to_equal : ℝ)
  (h1 : initial_first = 4.9)
  (h2 : growth_first = 0.275)
  (h3 : initial_second = 2.5)
  (h4 : time_to_equal = 5.647)
  (h5 : initial_first + growth_first * time_to_equal = initial_second + growth_second * time_to_equal) :
  ∃ growth_second : ℝ, abs (growth_second - 0.7) < 0.001 := by
  sorry

end second_brand_growth_rate_l3494_349473


namespace interesting_pairs_characterization_l3494_349448

/-- An ordered pair (a, b) of positive integers is interesting if for any positive integer n,
    there exists a positive integer k such that a^k + b is divisible by 2^n. -/
def IsInteresting (a b : ℕ+) : Prop :=
  ∀ n : ℕ+, ∃ k : ℕ+, (a.val ^ k.val + b.val) % (2^n.val) = 0

/-- Characterization of interesting pairs -/
theorem interesting_pairs_characterization (a b : ℕ+) :
  IsInteresting a b ↔ 
  (∃ (k l q : ℕ+), k ≥ 2 ∧ l.val % 2 = 1 ∧ q.val % 2 = 1 ∧
    ((a = 2^k.val * l + 1 ∧ b = 2^k.val * q - 1) ∨
     (a = 2^k.val * l - 1 ∧ b = 2^k.val * q + 1))) :=
sorry

end interesting_pairs_characterization_l3494_349448


namespace assignment_validity_l3494_349400

/-- Represents a variable in a programming language -/
structure Variable where
  name : String

/-- Represents an expression in a programming language -/
inductive Expression
  | Var : Variable → Expression
  | Product : Expression → Expression → Expression
  | Literal : Int → Expression

/-- Represents an assignment statement -/
structure Assignment where
  lhs : Expression
  rhs : Expression

/-- Predicate to check if an expression is a single variable -/
def isSingleVariable : Expression → Prop
  | Expression.Var _ => True
  | _ => False

/-- Theorem: An assignment statement is valid if and only if its left-hand side is a single variable -/
theorem assignment_validity (a : Assignment) :
  isSingleVariable a.lhs ↔ True :=
sorry

#check assignment_validity

end assignment_validity_l3494_349400


namespace arithmetic_sequence_length_l3494_349442

theorem arithmetic_sequence_length (a₁ aₙ d : ℤ) (n : ℕ) : 
  a₁ = 128 → aₙ = 14 → d = -3 → aₙ = a₁ + (n - 1) * d → n = 39 := by
  sorry

end arithmetic_sequence_length_l3494_349442


namespace ace_spade_probability_l3494_349491

/-- Represents a standard deck of cards -/
structure Deck :=
  (cards : Finset (Nat × Nat))
  (size_eq : cards.card = 52)

/-- Represents the event of drawing an Ace first and a spade second -/
def ace_spade_event (deck : Deck) : Finset (Nat × Nat × Nat × Nat) :=
  sorry

/-- The probability of the ace_spade_event -/
def ace_spade_prob (deck : Deck) : ℚ :=
  (ace_spade_event deck).card / deck.cards.card / (deck.cards.card - 1)

theorem ace_spade_probability (deck : Deck) : 
  ace_spade_prob deck = 1 / 52 := by
  sorry

end ace_spade_probability_l3494_349491


namespace andrew_runs_two_miles_l3494_349415

/-- Andrew's daily run in miles -/
def andrew_daily_run : ℝ := 2

/-- Peter's daily run in miles -/
def peter_daily_run : ℝ := andrew_daily_run + 3

/-- Total number of days they run -/
def days : ℕ := 5

/-- Total miles run by both Peter and Andrew -/
def total_miles : ℝ := 35

theorem andrew_runs_two_miles :
  andrew_daily_run = 2 ∧
  peter_daily_run = andrew_daily_run + 3 ∧
  days * (andrew_daily_run + peter_daily_run) = total_miles :=
by sorry

end andrew_runs_two_miles_l3494_349415


namespace game_ends_in_36_rounds_l3494_349476

/-- Represents the state of a player in the game -/
structure PlayerState :=
  (tokens : ℕ)

/-- Represents the state of the game -/
structure GameState :=
  (a : PlayerState)
  (b : PlayerState)
  (c : PlayerState)
  (round : ℕ)

/-- Updates the game state for a single round -/
def updateRound (state : GameState) : GameState :=
  sorry

/-- Updates the game state for the extra discard every 5 rounds -/
def extraDiscard (state : GameState) : GameState :=
  sorry

/-- Checks if the game has ended (any player has 0 tokens) -/
def gameEnded (state : GameState) : Bool :=
  sorry

/-- The main theorem stating that the game ends after exactly 36 rounds -/
theorem game_ends_in_36_rounds :
  let initialState : GameState := {
    a := { tokens := 17 },
    b := { tokens := 16 },
    c := { tokens := 15 },
    round := 0
  }
  ∃ (finalState : GameState),
    (finalState.round = 36) ∧
    (gameEnded finalState = true) ∧
    (∀ (intermediateState : GameState),
      intermediateState.round < 36 →
      gameEnded intermediateState = false) :=
sorry

end game_ends_in_36_rounds_l3494_349476


namespace sum_of_first_five_primes_l3494_349462

def first_five_primes : List Nat := [2, 3, 5, 7, 11]

theorem sum_of_first_five_primes :
  first_five_primes.sum = 28 := by
  sorry

end sum_of_first_five_primes_l3494_349462


namespace smallest_valid_N_sum_of_digits_l3494_349439

def P (N : ℕ) : ℚ := (N + 1 - Int.ceil (N / 3 : ℚ)) / (N + 1 : ℚ)

def is_valid (N : ℕ) : Prop :=
  N > 0 ∧ N % 5 = 0 ∧ N % 6 = 0 ∧ P N < 2/3

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sum_of_digits (n / 10)

theorem smallest_valid_N_sum_of_digits :
  ∃ N, is_valid N ∧
    (∀ M, is_valid M → N ≤ M) ∧
    sum_of_digits N = 9 :=
sorry

end smallest_valid_N_sum_of_digits_l3494_349439


namespace number_puzzle_l3494_349423

theorem number_puzzle (N A : ℝ) : N = 295 ∧ N / 5 + A = 65 → A = 6 := by
  sorry

end number_puzzle_l3494_349423


namespace wendys_recycling_points_l3494_349401

/-- Wendy's recycling points calculation -/
theorem wendys_recycling_points :
  ∀ (points_per_bag : ℕ) (total_bags : ℕ) (unrecycled_bags : ℕ),
    points_per_bag = 5 →
    total_bags = 11 →
    unrecycled_bags = 2 →
    points_per_bag * (total_bags - unrecycled_bags) = 45 := by
  sorry

end wendys_recycling_points_l3494_349401


namespace hyperbola_focus_distance_ratio_l3494_349406

/-- Given a hyperbola x²/a - y²/a = 1 with a > 0, prove that |FP|/|MN| = √2/2 where:
    F is the right focus
    M and N are intersection points of any line through F with the right branch
    P is the intersection of the perpendicular bisector of MN with the x-axis -/
theorem hyperbola_focus_distance_ratio (a : ℝ) (h : a > 0) :
  ∃ (F M N P : ℝ × ℝ),
    (∀ (x y : ℝ), x^2/a - y^2/a = 1 → 
      (∃ (t : ℝ), (x, y) = M ∨ (x, y) = N) → 
      (F.1 > 0 ∧ F.2 = 0) ∧
      (∃ (m : ℝ), (M.2 - F.2) = m * (M.1 - F.1) ∧ (N.2 - F.2) = m * (N.1 - F.1)) ∧
      (P.2 = 0 ∧ P.1 = (M.1 + N.1)/2 + (M.2 + N.2)^2 / (2 * (M.1 + N.1)))) →
    ‖F - P‖ / ‖M - N‖ = Real.sqrt 2 / 2 := by
  sorry

end hyperbola_focus_distance_ratio_l3494_349406


namespace specific_trapezoid_area_l3494_349424

/-- An isosceles trapezoid with given dimensions -/
structure IsoscelesTrapezoid where
  leg_length : ℝ
  diagonal_length : ℝ
  longer_base : ℝ

/-- Calculate the area of an isosceles trapezoid -/
def trapezoid_area (t : IsoscelesTrapezoid) : ℝ :=
  sorry

/-- Theorem stating the area of the specific isosceles trapezoid -/
theorem specific_trapezoid_area :
  let t : IsoscelesTrapezoid := {
    leg_length := 40,
    diagonal_length := 50,
    longer_base := 60
  }
  ∃ ε > 0, |trapezoid_area t - 1242.425| < ε :=
sorry

end specific_trapezoid_area_l3494_349424


namespace adjacent_same_face_exists_l3494_349410

/-- Represents a coin showing either heads or tails -/
inductive CoinFace
| Heads
| Tails

/-- Checks if two coin faces are the same -/
def same_face (a b : CoinFace) : Prop :=
  (a = CoinFace.Heads ∧ b = CoinFace.Heads) ∨ (a = CoinFace.Tails ∧ b = CoinFace.Tails)

/-- A circular arrangement of 11 coins -/
def CoinArrangement := Fin 11 → CoinFace

theorem adjacent_same_face_exists (arrangement : CoinArrangement) :
  ∃ i : Fin 11, same_face (arrangement i) (arrangement ((i + 1) % 11)) :=
sorry


end adjacent_same_face_exists_l3494_349410


namespace geometric_sequence_problem_l3494_349421

/-- A geometric sequence is a sequence where the ratio between any two consecutive terms is constant. -/
def IsGeometricSequence (a : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, a (n + 1) = r * a n

/-- Given a geometric sequence a with a_3 = 4 and a_7 = 12, prove that a_11 = 36 -/
theorem geometric_sequence_problem (a : ℕ → ℝ) 
    (h_geo : IsGeometricSequence a) 
    (h_3 : a 3 = 4) 
    (h_7 : a 7 = 12) : 
  a 11 = 36 := by
  sorry

end geometric_sequence_problem_l3494_349421


namespace smallest_n_for_subset_sequence_l3494_349466

theorem smallest_n_for_subset_sequence (X : Finset ℕ) (h : X.card = 100) :
  let n := 2 * Nat.choose 100 50 + 2 * Nat.choose 100 49 + 1
  ∀ (A : Fin n → Finset ℕ), (∀ i, A i ⊆ X) →
    (∃ i j k, i < j ∧ j < k ∧ (A i ⊆ A j ∧ A j ⊆ A k ∨ A i ⊇ A j ∧ A j ⊇ A k)) ∧
  ∀ m < n, ∃ (B : Fin m → Finset ℕ), (∀ i, B i ⊆ X) ∧
    ¬(∃ i j k, i < j ∧ j < k ∧ (B i ⊆ B j ∧ B j ⊆ B k ∨ B i ⊇ B j ∧ B j ⊇ B k)) :=
by sorry

end smallest_n_for_subset_sequence_l3494_349466


namespace complex_fraction_simplification_l3494_349433

theorem complex_fraction_simplification :
  (5 + 7*I) / (2 + 3*I) = 31/13 - (1/13)*I :=
by sorry

end complex_fraction_simplification_l3494_349433


namespace square_of_binomial_l3494_349428

theorem square_of_binomial (a : ℚ) : 
  (∃ (r s : ℚ), ∀ (x : ℚ), a * x^2 + 15 * x + 16 = (r * x + s)^2) → 
  a = 225 / 64 := by
sorry

end square_of_binomial_l3494_349428


namespace unit_circle_from_sin_cos_l3494_349488

-- Define the set of points (x,y) = (sin t, cos t) for all real t
def unitCirclePoints : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | ∃ t : ℝ, p.1 = Real.sin t ∧ p.2 = Real.cos t}

-- Theorem: The set of points forms a circle with radius 1 centered at the origin
theorem unit_circle_from_sin_cos :
  unitCirclePoints = {p : ℝ × ℝ | p.1^2 + p.2^2 = 1} := by
  sorry


end unit_circle_from_sin_cos_l3494_349488


namespace sequence_term_value_l3494_349431

/-- Given a finite sequence {a_n} with m terms, where S(n) represents the sum of all terms
    starting from the n-th term, prove that a_n = -2n - 1 when 1 ≤ n < m. -/
theorem sequence_term_value (m : ℕ) (a : ℕ → ℤ) (S : ℕ → ℤ) (n : ℕ) 
    (h1 : 1 ≤ n) (h2 : n < m) (h3 : ∀ k, 1 ≤ k → k ≤ m → S k = k^2) :
  a n = -2 * n - 1 := by
  sorry

end sequence_term_value_l3494_349431


namespace max_profit_theorem_l3494_349470

/-- Represents the production plan and profit calculation for a company producing two types of crafts. -/
structure ProductionPlan where
  /-- Cost of material A in yuan -/
  cost_A : ℕ
  /-- Cost of material B in yuan -/
  cost_B : ℕ
  /-- Number of units of craft X produced -/
  units_X : ℕ
  /-- Number of units of craft Y produced -/
  units_Y : ℕ
  /-- Condition: Cost of B is 40 yuan more than A -/
  cost_diff : cost_B = cost_A + 40
  /-- Condition: 2 units of A and 3 units of B cost 420 yuan -/
  total_cost : 2 * cost_A + 3 * cost_B = 420
  /-- Condition: Total number of crafts is 560 -/
  total_units : units_X + units_Y = 560
  /-- Condition: X should not exceed 180 units -/
  max_X : units_X ≤ 180

/-- Calculates the profit for a given production plan -/
def profit (plan : ProductionPlan) : ℕ :=
  360 * plan.units_X + 450 * plan.units_Y -
  (plan.cost_A * (plan.units_X + 3 * plan.units_Y) +
   plan.cost_B * (2 * plan.units_X + 2 * plan.units_Y))

/-- Theorem stating the maximum profit and optimal production plan -/
theorem max_profit_theorem (plan : ProductionPlan) : 
  plan.cost_A = 60 ∧ plan.cost_B = 100 ∧ plan.units_X = 180 ∧ plan.units_Y = 380 →
  profit plan = 44600 ∧ ∀ other_plan : ProductionPlan, profit other_plan ≤ profit plan := by
  sorry


end max_profit_theorem_l3494_349470


namespace largest_angle_measure_l3494_349441

/-- Represents a pentagon with angles in the ratio 3:3:3:4:5 -/
structure RatioPentagon where
  /-- The common factor for the angle measures -/
  x : ℝ
  /-- The sum of interior angles of a pentagon is 540° -/
  angle_sum : 3*x + 3*x + 3*x + 4*x + 5*x = 540

/-- Theorem: The largest angle in a RatioPentagon is 150° -/
theorem largest_angle_measure (p : RatioPentagon) : 5 * p.x = 150 := by
  sorry

#check largest_angle_measure

end largest_angle_measure_l3494_349441


namespace john_money_left_l3494_349418

/-- The amount of money John has left after buying pizzas and drinks -/
def money_left (q : ℝ) : ℝ :=
  let drink_cost := q
  let small_pizza_cost := q
  let large_pizza_cost := 4 * q
  let total_spent := 4 * drink_cost + 2 * small_pizza_cost + large_pizza_cost
  50 - total_spent

/-- Theorem: John has 50 - 10q dollars left after his purchases -/
theorem john_money_left (q : ℝ) : money_left q = 50 - 10 * q := by
  sorry

end john_money_left_l3494_349418


namespace investment_proportion_l3494_349485

/-- Given two investors X and Y, where X invested 5000 and their profit is divided in the ratio 2:6,
    prove that Y's investment is 15000. -/
theorem investment_proportion (x_investment y_investment : ℕ) (profit_ratio_x profit_ratio_y : ℕ) :
  x_investment = 5000 →
  profit_ratio_x = 2 →
  profit_ratio_y = 6 →
  y_investment = 15000 :=
by
  sorry

end investment_proportion_l3494_349485


namespace probability_sum_25_is_7_200_l3494_349484

-- Define the structure of a die
structure Die :=
  (faces : Finset ℕ)
  (blank_face : Bool)
  (fair : Bool)

-- Define the two dice
def die1 : Die :=
  { faces := Finset.range 20 \ {20},
    blank_face := true,
    fair := true }

def die2 : Die :=
  { faces := (Finset.range 21 \ {0, 8}),
    blank_face := true,
    fair := true }

-- Define the function to calculate the probability
def probability_sum_25 (d1 d2 : Die) : ℚ :=
  let total_outcomes := 20 * 20
  let valid_combinations := 14
  valid_combinations / total_outcomes

-- State the theorem
theorem probability_sum_25_is_7_200 :
  probability_sum_25 die1 die2 = 7 / 200 := by
  sorry

end probability_sum_25_is_7_200_l3494_349484


namespace solution_count_3x_4y_815_l3494_349452

theorem solution_count_3x_4y_815 : 
  (Finset.filter (fun p : ℕ × ℕ => 3 * p.1 + 4 * p.2 = 815 ∧ p.1 > 0 ∧ p.2 > 0) (Finset.product (Finset.range 816) (Finset.range 816))).card = 68 :=
by sorry

end solution_count_3x_4y_815_l3494_349452


namespace rent_increase_percentage_l3494_349409

theorem rent_increase_percentage (last_year_earnings : ℝ) : 
  let last_year_rent := 0.20 * last_year_earnings
  let this_year_earnings := 1.20 * last_year_earnings
  let this_year_rent := 0.30 * this_year_earnings
  (this_year_rent / last_year_rent) * 100 = 180 := by
sorry

end rent_increase_percentage_l3494_349409


namespace smallest_with_eight_prime_power_divisors_l3494_349464

def is_prime_power (n : ℕ) : Prop := ∃ p k, Prime p ∧ n = p ^ k

def divisors (n : ℕ) : Finset ℕ :=
  Finset.filter (· ∣ n) (Finset.range (n + 1))

theorem smallest_with_eight_prime_power_divisors :
  (∀ m : ℕ, m < 24 →
    (divisors m).card ≠ 8 ∨
    ¬(∀ d ∈ divisors m, is_prime_power d)) ∧
  (divisors 24).card = 8 ∧
  (∀ d ∈ divisors 24, is_prime_power d) :=
sorry

end smallest_with_eight_prime_power_divisors_l3494_349464


namespace sum_of_arguments_fifth_roots_l3494_349479

/-- The sum of arguments of the fifth roots of 81(1+i) is 765 degrees -/
theorem sum_of_arguments_fifth_roots (z₁ z₂ z₃ z₄ z₅ : ℂ) 
  (h₁ : z₁^5 = 81 * (1 + Complex.I))
  (h₂ : z₂^5 = 81 * (1 + Complex.I))
  (h₃ : z₃^5 = 81 * (1 + Complex.I))
  (h₄ : z₄^5 = 81 * (1 + Complex.I))
  (h₅ : z₅^5 = 81 * (1 + Complex.I))
  (hr₁ : Complex.abs z₁ > 0)
  (hr₂ : Complex.abs z₂ > 0)
  (hr₃ : Complex.abs z₃ > 0)
  (hr₄ : Complex.abs z₄ > 0)
  (hr₅ : Complex.abs z₅ > 0)
  (hθ₁ : 0 ≤ Complex.arg z₁ ∧ Complex.arg z₁ < 2 * Real.pi)
  (hθ₂ : 0 ≤ Complex.arg z₂ ∧ Complex.arg z₂ < 2 * Real.pi)
  (hθ₃ : 0 ≤ Complex.arg z₃ ∧ Complex.arg z₃ < 2 * Real.pi)
  (hθ₄ : 0 ≤ Complex.arg z₄ ∧ Complex.arg z₄ < 2 * Real.pi)
  (hθ₅ : 0 ≤ Complex.arg z₅ ∧ Complex.arg z₅ < 2 * Real.pi) :
  Complex.arg z₁ + Complex.arg z₂ + Complex.arg z₃ + Complex.arg z₄ + Complex.arg z₅ = 
    (765 * Real.pi) / 180 := by
  sorry

end sum_of_arguments_fifth_roots_l3494_349479


namespace negation_of_existence_square_greater_than_self_negation_l3494_349451

theorem negation_of_existence (P : ℕ → Prop) : 
  (¬ ∃ x : ℕ, P x) ↔ (∀ x : ℕ, ¬ P x) := by sorry

theorem square_greater_than_self_negation :
  (¬ ∃ x : ℕ, x^2 ≤ x) ↔ (∀ x : ℕ, x^2 > x) := by sorry

end negation_of_existence_square_greater_than_self_negation_l3494_349451


namespace student_count_theorem_l3494_349459

def valid_student_count (n : ℕ) : Prop :=
  n < 50 ∧ n % 6 = 5 ∧ n % 3 = 2

theorem student_count_theorem : 
  {n : ℕ | valid_student_count n} = {5, 11, 17, 23, 29, 35, 41, 47} :=
sorry

end student_count_theorem_l3494_349459


namespace henry_skittles_l3494_349480

/-- The number of Skittles Bridget has initially -/
def bridget_initial : ℕ := 4

/-- The total number of Skittles Bridget has after receiving Henry's Skittles -/
def bridget_final : ℕ := 8

/-- The number of Skittles Henry has -/
def henry : ℕ := bridget_final - bridget_initial

theorem henry_skittles : henry = 4 := by
  sorry

end henry_skittles_l3494_349480


namespace horner_v2_value_l3494_349411

-- Define the polynomial
def f (x : ℝ) : ℝ := 2*x^7 + x^6 + x^4 + x^2 + 1

-- Define Horner's method for the first two steps
def horner_v2 (a : ℝ) : ℝ := 
  let v0 := 2
  let v1 := v0 * a + 1
  v1 * a

-- Theorem statement
theorem horner_v2_value :
  horner_v2 2 = 10 :=
sorry

end horner_v2_value_l3494_349411


namespace expression_evaluation_l3494_349417

theorem expression_evaluation :
  let f (x : ℝ) := 3 * x^2 - 4 * x + 2
  f 2 = 6 := by sorry

end expression_evaluation_l3494_349417


namespace tangent_line_at_A_l3494_349481

/-- The curve C defined by y = x^3 - x + 2 -/
def C (x : ℝ) : ℝ := x^3 - x + 2

/-- The point A on the curve C -/
def A : ℝ × ℝ := (1, 2)

/-- The tangent line equation at point A -/
def tangent_line (x y : ℝ) : Prop := 2*x - y = 0

theorem tangent_line_at_A :
  tangent_line A.1 A.2 ∧
  ∀ x : ℝ, (tangent_line x (C x) ↔ x = A.1) :=
sorry

end tangent_line_at_A_l3494_349481


namespace f_decreasing_interval_f_min_value_f_max_value_l3494_349438

-- Define the function f(x)
def f (x : ℝ) : ℝ := 3 * x^3 - 9 * x + 5

-- Theorem for monotonically decreasing interval
theorem f_decreasing_interval :
  ∀ x ∈ Set.Ioo (-1 : ℝ) 1, ∀ y ∈ Set.Ioo (-1 : ℝ) 1, x < y → f x > f y :=
sorry

-- Theorem for minimum value on [-3, 3]
theorem f_min_value :
  ∃ x ∈ Set.Icc (-3 : ℝ) 3, ∀ y ∈ Set.Icc (-3 : ℝ) 3, f x ≤ f y ∧ f x = -49 :=
sorry

-- Theorem for maximum value on [-3, 3]
theorem f_max_value :
  ∃ x ∈ Set.Icc (-3 : ℝ) 3, ∀ y ∈ Set.Icc (-3 : ℝ) 3, f x ≥ f y ∧ f x = 59 :=
sorry

end f_decreasing_interval_f_min_value_f_max_value_l3494_349438


namespace sqrt_18_div_sqrt_2_equals_3_l3494_349444

theorem sqrt_18_div_sqrt_2_equals_3 : Real.sqrt 18 / Real.sqrt 2 = 3 := by
  sorry

end sqrt_18_div_sqrt_2_equals_3_l3494_349444


namespace mans_speed_with_current_l3494_349435

theorem mans_speed_with_current 
  (current_speed : ℝ) 
  (speed_against_current : ℝ) 
  (h1 : current_speed = 2.5)
  (h2 : speed_against_current = 10) : 
  ∃ (speed_with_current : ℝ), speed_with_current = 15 :=
by
  sorry

end mans_speed_with_current_l3494_349435


namespace pythagorean_triple_check_l3494_349477

def is_pythagorean_triple (a b c : ℕ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ a * a + b * b = c * c

theorem pythagorean_triple_check :
  ¬ is_pythagorean_triple 12 15 18 ∧
  is_pythagorean_triple 3 4 5 ∧
  ¬ is_pythagorean_triple 6 9 15 :=
by sorry

end pythagorean_triple_check_l3494_349477


namespace quadratic_root_form_l3494_349413

theorem quadratic_root_form (d : ℝ) : 
  (∀ x : ℝ, x^2 + 7*x + d = 0 ↔ x = (-7 + Real.sqrt d) / 2 ∨ x = (-7 - Real.sqrt d) / 2) → 
  d = 9.8 := by
  sorry

end quadratic_root_form_l3494_349413


namespace unit_digit_of_product_l3494_349460

def numbers : List Nat := [6245, 7083, 9137, 4631, 5278, 3974]

theorem unit_digit_of_product (nums : List Nat := numbers) :
  (nums.foldl (· * ·) 1) % 10 = 0 := by
  sorry

end unit_digit_of_product_l3494_349460
