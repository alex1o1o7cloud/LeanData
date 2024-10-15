import Mathlib

namespace NUMINAMATH_CALUDE_congruence_solution_l183_18321

theorem congruence_solution : ∃ n : ℕ, n ≤ 4 ∧ n ≡ -2323 [ZMOD 5] ∧ n = 2 := by
  sorry

end NUMINAMATH_CALUDE_congruence_solution_l183_18321


namespace NUMINAMATH_CALUDE_flower_pollination_l183_18387

/-- Represents the types of flowers -/
inductive FlowerType
| Rose
| Sunflower
| Tulip
| Daisy
| Orchid

/-- Represents a bee -/
structure Bee where
  roses_per_hour : ℕ
  sunflowers_per_hour : ℕ
  tulips_per_hour : ℕ
  daisies_per_hour : ℕ
  orchids_per_hour : ℕ

/-- The problem setup -/
def flower_problem : Prop :=
  let total_flowers : ℕ := 60
  let roses : ℕ := 12
  let sunflowers : ℕ := 15
  let tulips : ℕ := 9
  let daisies : ℕ := 18
  let orchids : ℕ := 6
  let hours : ℕ := 3
  let bee_A : Bee := ⟨2, 3, 1, 0, 0⟩
  let bee_B : Bee := ⟨0, 0, 0, 4, 1⟩
  let bee_C : Bee := ⟨1, 2, 2, 3, 1⟩
  let bees : List Bee := [bee_A, bee_B, bee_C]

  total_flowers = roses + sunflowers + tulips + daisies + orchids ∧
  (bees.map (λ b => b.roses_per_hour + b.sunflowers_per_hour + b.tulips_per_hour + 
                    b.daisies_per_hour + b.orchids_per_hour)).sum * hours = 60 ∧
  ∀ ft : FlowerType, 
    (bees.map (λ b => match ft with
      | FlowerType.Rose => b.roses_per_hour
      | FlowerType.Sunflower => b.sunflowers_per_hour
      | FlowerType.Tulip => b.tulips_per_hour
      | FlowerType.Daisy => b.daisies_per_hour
      | FlowerType.Orchid => b.orchids_per_hour
    )).sum * hours ≤ match ft with
      | FlowerType.Rose => roses
      | FlowerType.Sunflower => sunflowers
      | FlowerType.Tulip => tulips
      | FlowerType.Daisy => daisies
      | FlowerType.Orchid => orchids

theorem flower_pollination : flower_problem := by sorry

end NUMINAMATH_CALUDE_flower_pollination_l183_18387


namespace NUMINAMATH_CALUDE_abs_neg_2023_l183_18335

theorem abs_neg_2023 : |(-2023 : ℤ)| = 2023 := by
  sorry

end NUMINAMATH_CALUDE_abs_neg_2023_l183_18335


namespace NUMINAMATH_CALUDE_triangle_side_length_l183_18375

theorem triangle_side_length (a b c : ℝ) (A B C : ℝ) (S : ℝ) :
  a = 3 →
  C = 2 * π / 3 →
  S = (15 * Real.sqrt 3) / 4 →
  S = (1 / 2) * a * b * Real.sin C →
  c^2 = a^2 + b^2 - 2 * a * b * Real.cos C →
  c = 7 := by
  sorry

end NUMINAMATH_CALUDE_triangle_side_length_l183_18375


namespace NUMINAMATH_CALUDE_tangent_line_equation_l183_18351

-- Define the curve
def f (x : ℝ) : ℝ := x^3 - 1

-- Define the point of tangency
def P : ℝ × ℝ := (1, 0)

-- Theorem statement
theorem tangent_line_equation :
  let m := (deriv f) P.1  -- Slope of the tangent line
  let b := P.2 - m * P.1  -- y-intercept of the tangent line
  (∀ x y, y = m * x + b ↔ 3 * x - y - 3 = 0) ∧ 
  (f P.1 = P.2) ∧  -- The point P lies on the curve
  (∀ x, (deriv f) x = 3 * x^2) -- The derivative of f is correct
  :=
by sorry

end NUMINAMATH_CALUDE_tangent_line_equation_l183_18351


namespace NUMINAMATH_CALUDE_great_wall_soldiers_l183_18332

/-- Calculates the total number of soldiers in beacon towers along a wall -/
def total_soldiers (wall_length : ℕ) (tower_interval : ℕ) (soldiers_per_tower : ℕ) : ℕ :=
  (wall_length / tower_interval) * soldiers_per_tower

/-- Theorem stating that for a wall of 7300 km with towers every 5 km and 2 soldiers per tower, 
    the total number of soldiers is 2920 -/
theorem great_wall_soldiers : 
  total_soldiers 7300 5 2 = 2920 := by
  sorry

end NUMINAMATH_CALUDE_great_wall_soldiers_l183_18332


namespace NUMINAMATH_CALUDE_dodecagon_diagonals_l183_18308

/-- The number of diagonals in a convex n-gon --/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- A dodecagon has 12 sides --/
def dodecagon_sides : ℕ := 12

theorem dodecagon_diagonals :
  num_diagonals dodecagon_sides = 54 := by sorry

end NUMINAMATH_CALUDE_dodecagon_diagonals_l183_18308


namespace NUMINAMATH_CALUDE_intersection_line_equation_l183_18333

/-- Given two lines l₁ and l₂ that intersect to form a line segment with midpoint P(0, 0),
    prove that the line l passing through their intersection points has equation y = 7/6 * x. -/
theorem intersection_line_equation 
  (l₁ : Set (ℝ × ℝ)) 
  (l₂ : Set (ℝ × ℝ)) 
  (h₁ : l₁ = {(x, y) | 4 * x + y + 6 = 0})
  (h₂ : l₂ = {(x, y) | 3 * x - 5 * y - 6 = 0})
  (h_midpoint : ∃ (a b : ℝ × ℝ), a ∈ l₁ ∧ a ∈ l₂ ∧ b ∈ l₁ ∧ b ∈ l₂ ∧ (0, 0) = ((a.1 + b.1) / 2, (a.2 + b.2) / 2)) :
  ∃ (l : Set (ℝ × ℝ)), l = {(x, y) | y = 7/6 * x} ∧ 
    ∀ (p : ℝ × ℝ), (p ∈ l₁ ∧ p ∈ l₂) → p ∈ l :=
sorry

end NUMINAMATH_CALUDE_intersection_line_equation_l183_18333


namespace NUMINAMATH_CALUDE_condition_relationship_l183_18353

theorem condition_relationship (a : ℝ) : 
  (a = 1 → a^2 - 3*a + 2 = 0) ∧ 
  (∃ b : ℝ, b ≠ 1 ∧ b^2 - 3*b + 2 = 0) :=
by sorry

end NUMINAMATH_CALUDE_condition_relationship_l183_18353


namespace NUMINAMATH_CALUDE_distance_to_focus_l183_18305

/-- The distance from a point on a parabola to its focus -/
theorem distance_to_focus (x : ℝ) : 
  x^2 = 16 → -- Point A(x, 4) is on the parabola x^2 = 4y
  ∃ (f : ℝ × ℝ), -- There exists a focus f
    (∀ (p : ℝ × ℝ), p.2 = p.1^2 / 4 → dist p f = p.2 + 1) ∧ -- Definition of parabola
    dist (x, 4) f = 5 -- The distance from A to the focus is 5
:= by sorry

end NUMINAMATH_CALUDE_distance_to_focus_l183_18305


namespace NUMINAMATH_CALUDE_min_moves_to_equalize_l183_18309

/-- Represents the state of coin stacks -/
structure CoinStacks :=
  (stack1 : ℕ)
  (stack2 : ℕ)
  (stack3 : ℕ)
  (stack4 : ℕ)

/-- Represents a move in the coin stacking game -/
def move (s : CoinStacks) : CoinStacks := sorry

/-- Checks if all stacks have equal coins -/
def is_equal (s : CoinStacks) : Prop := 
  s.stack1 = s.stack2 ∧ s.stack2 = s.stack3 ∧ s.stack3 = s.stack4

/-- The initial state of coin stacks -/
def initial_state : CoinStacks := ⟨9, 7, 5, 10⟩

/-- Applies n moves to a given state -/
def apply_moves (s : CoinStacks) (n : ℕ) : CoinStacks := sorry

/-- The main theorem stating the minimum number of moves required -/
theorem min_moves_to_equalize : 
  ∃ (n : ℕ), n = 11 ∧ is_equal (apply_moves initial_state n) ∧ 
  ∀ (m : ℕ), m < n → ¬is_equal (apply_moves initial_state m) :=
sorry

end NUMINAMATH_CALUDE_min_moves_to_equalize_l183_18309


namespace NUMINAMATH_CALUDE_used_car_seller_problem_l183_18388

theorem used_car_seller_problem (num_clients : ℕ) (cars_per_client : ℕ) (selections_per_car : ℕ) 
  (h1 : num_clients = 15)
  (h2 : cars_per_client = 2)
  (h3 : selections_per_car = 3) :
  (num_clients * cars_per_client) / selections_per_car = 10 := by
  sorry

#check used_car_seller_problem

end NUMINAMATH_CALUDE_used_car_seller_problem_l183_18388


namespace NUMINAMATH_CALUDE_age_fraction_proof_l183_18356

theorem age_fraction_proof (age : ℕ) (h : age = 64) :
  (8 * (age + 8) - 8 * (age - 8)) / age = 2 := by
  sorry

end NUMINAMATH_CALUDE_age_fraction_proof_l183_18356


namespace NUMINAMATH_CALUDE_completing_square_equivalence_l183_18359

theorem completing_square_equivalence :
  ∀ x : ℝ, x^2 - 4*x - 22 = 0 ↔ (x - 2)^2 = 26 := by
  sorry

end NUMINAMATH_CALUDE_completing_square_equivalence_l183_18359


namespace NUMINAMATH_CALUDE_tim_has_156_golf_balls_l183_18358

/-- The number of units in a dozen -/
def dozen : ℕ := 12

/-- The number of dozens of golf balls Tim has -/
def tims_dozens : ℕ := 13

/-- The total number of golf balls Tim has -/
def tims_golf_balls : ℕ := tims_dozens * dozen

theorem tim_has_156_golf_balls : tims_golf_balls = 156 := by
  sorry

end NUMINAMATH_CALUDE_tim_has_156_golf_balls_l183_18358


namespace NUMINAMATH_CALUDE_equation_solution_l183_18345

theorem equation_solution :
  let x : ℚ := 1/2
  2 * x - 1 = 0 :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l183_18345


namespace NUMINAMATH_CALUDE_smallest_n_for_rain_probability_l183_18382

def rain_probability (n : ℕ) : ℝ :=
  let rec prob (k : ℕ) (p : ℝ) : ℝ :=
    match k with
    | 0 => p
    | k + 1 => prob k (0.5 * p + 0.25)
  prob n 0

theorem smallest_n_for_rain_probability :
  ∀ k : ℕ, k < 9 → rain_probability k ≤ 0.499 ∧
  rain_probability 9 > 0.499 := by
  sorry

end NUMINAMATH_CALUDE_smallest_n_for_rain_probability_l183_18382


namespace NUMINAMATH_CALUDE_four_digit_greater_than_product_l183_18306

theorem four_digit_greater_than_product (a b c d : ℕ) : 
  a ≤ 9 → b ≤ 9 → c ≤ 9 → d ≤ 9 → 
  (1000 * a + 100 * b + 10 * c + d > (10 * a + b) * (10 * c + d)) ∨ 
  (a = 0 ∧ b = 0 ∧ c = 0 ∧ d = 0) := by
  sorry

end NUMINAMATH_CALUDE_four_digit_greater_than_product_l183_18306


namespace NUMINAMATH_CALUDE_field_trip_participation_l183_18316

/-- Given a class of students where:
    - 4/5 of students left on the first vehicle
    - Of those who stayed, 1/3 didn't want to go
    - When another vehicle was found, 1/2 of the remaining students who wanted to go were able to join
    Prove that the fraction of students who went on the field trip is 13/15 -/
theorem field_trip_participation (total_students : ℕ) (total_students_pos : total_students > 0) :
  let first_vehicle := (4 : ℚ) / 5 * total_students
  let stayed_behind := total_students - first_vehicle
  let not_wanting_to_go := (1 : ℚ) / 3 * stayed_behind
  let wanting_to_go := stayed_behind - not_wanting_to_go
  let additional_joiners := (1 : ℚ) / 2 * wanting_to_go
  first_vehicle + additional_joiners = (13 : ℚ) / 15 * total_students :=
by sorry

end NUMINAMATH_CALUDE_field_trip_participation_l183_18316


namespace NUMINAMATH_CALUDE_inequality_proof_l183_18365

theorem inequality_proof (a b c : ℝ) 
  (h_pos_a : 0 < a) (h_pos_b : 0 < b) (h_pos_c : 0 < c)
  (h_sum : a + b + c = 1) :
  a / (2 * a + 1) + b / (3 * b + 1) + c / (6 * c + 1) ≤ 1 / 2 ∧
  (a / (2 * a + 1) + b / (3 * b + 1) + c / (6 * c + 1) = 1 / 2 ↔ 
   a = 1 / 2 ∧ b = 1 / 3 ∧ c = 1 / 6) :=
by sorry

end NUMINAMATH_CALUDE_inequality_proof_l183_18365


namespace NUMINAMATH_CALUDE_functional_equation_solutions_l183_18343

-- Define the set of positive integers
def PositiveInt := {n : ℤ | n > 0}

-- Define the functional equation
def SatisfiesEquation (f : ℚ → ℤ) : Prop :=
  ∀ (x : ℚ) (a : ℤ) (b : PositiveInt), f ((f x + a) / b) = f ((x + a) / b)

-- Define the possible solution functions
def ConstantFunction (C : ℤ) : ℚ → ℤ := λ _ => C
def FloorFunction : ℚ → ℤ := λ x => ⌊x⌋
def CeilingFunction : ℚ → ℤ := λ x => ⌈x⌉

-- State the theorem
theorem functional_equation_solutions (f : ℚ → ℤ) (h : SatisfiesEquation f) :
  (∃ C : ℤ, f = ConstantFunction C) ∨ f = FloorFunction ∨ f = CeilingFunction :=
sorry

end NUMINAMATH_CALUDE_functional_equation_solutions_l183_18343


namespace NUMINAMATH_CALUDE_dodo_is_sane_l183_18372

-- Define the characters
inductive Character : Type
| Dodo : Character
| Lori : Character
| Eagle : Character

-- Define the "thinks" relation
def thinks (x y : Character) (p : Prop) : Prop := sorry

-- Define sanity
def is_sane (x : Character) : Prop := sorry

-- State the theorem
theorem dodo_is_sane :
  (thinks Dodo Lori (¬ is_sane Eagle)) →
  (thinks Lori Dodo (¬ is_sane Dodo)) →
  (thinks Eagle Dodo (is_sane Dodo)) →
  is_sane Dodo := by sorry

end NUMINAMATH_CALUDE_dodo_is_sane_l183_18372


namespace NUMINAMATH_CALUDE_jogger_speed_l183_18377

/-- Jogger's speed calculation -/
theorem jogger_speed (train_length : ℝ) (initial_distance : ℝ) (train_speed : ℝ) (passing_time : ℝ) :
  let relative_speed : ℝ := (train_length + initial_distance) / passing_time
  let train_speed_mps : ℝ := train_speed * (5/18)
  let jogger_speed_mps : ℝ := train_speed_mps - relative_speed
  let jogger_speed_kmh : ℝ := jogger_speed_mps * (18/5)
  train_length = 120 →
  initial_distance = 280 →
  train_speed = 45 →
  passing_time = 40 →
  jogger_speed_kmh = 9 := by
  sorry

end NUMINAMATH_CALUDE_jogger_speed_l183_18377


namespace NUMINAMATH_CALUDE_golden_ratio_between_zero_and_one_l183_18331

theorem golden_ratio_between_zero_and_one :
  let φ := (Real.sqrt 5 - 1) / 2
  0 < φ ∧ φ < 1 := by sorry

end NUMINAMATH_CALUDE_golden_ratio_between_zero_and_one_l183_18331


namespace NUMINAMATH_CALUDE_sin_210_degrees_l183_18381

theorem sin_210_degrees : Real.sin (210 * π / 180) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_sin_210_degrees_l183_18381


namespace NUMINAMATH_CALUDE_iron_ball_surface_area_l183_18313

/-- The surface area of a spherical iron ball that displaces a specific volume of water -/
theorem iron_ball_surface_area (r : ℝ) (h : ℝ) (R : ℝ) : 
  r = 10 → h = 5/3 → (4/3) * Real.pi * R^3 = Real.pi * r^2 * h → 4 * Real.pi * R^2 = 100 * Real.pi :=
by sorry

end NUMINAMATH_CALUDE_iron_ball_surface_area_l183_18313


namespace NUMINAMATH_CALUDE_average_of_numbers_between_40_and_80_divisible_by_3_l183_18393

def numbers_between_40_and_80_divisible_by_3 : List ℕ :=
  (List.range 41).filter (λ n => 40 < n ∧ n ≤ 80 ∧ n % 3 = 0)

theorem average_of_numbers_between_40_and_80_divisible_by_3 :
  (List.sum numbers_between_40_and_80_divisible_by_3) / 
  (List.length numbers_between_40_and_80_divisible_by_3) = 63 := by
  sorry

end NUMINAMATH_CALUDE_average_of_numbers_between_40_and_80_divisible_by_3_l183_18393


namespace NUMINAMATH_CALUDE_complement_A_intersect_B_l183_18329

-- Define the universal set U as ℝ
def U : Set ℝ := Set.univ

-- Define set A
def A : Set ℝ := {x | x^2 - 2*x > 0}

-- Define set B
def B : Set ℝ := {x | ∃ y, y = Real.log (x - 1)}

-- Statement to prove
theorem complement_A_intersect_B :
  (Set.compl A) ∩ B = {x | 1 < x ∧ x ≤ 2} := by sorry

end NUMINAMATH_CALUDE_complement_A_intersect_B_l183_18329


namespace NUMINAMATH_CALUDE_cube_volume_from_surface_area_l183_18369

theorem cube_volume_from_surface_area :
  ∀ (s : ℝ), s > 0 → 6 * s^2 = 600 → s^3 = 1000 :=
by
  sorry

end NUMINAMATH_CALUDE_cube_volume_from_surface_area_l183_18369


namespace NUMINAMATH_CALUDE_correct_taobao_shopping_order_l183_18334

-- Define the type for shopping steps
inductive ShoppingStep
| select_products
| buy_and_pay
| transfer_payment
| receive_and_confirm
| ship_goods

-- Define the shopping process
def shopping_process : List ShoppingStep :=
  [ShoppingStep.select_products, ShoppingStep.buy_and_pay, ShoppingStep.ship_goods, 
   ShoppingStep.receive_and_confirm, ShoppingStep.transfer_payment]

-- Define a function to check if the order is correct
def is_correct_order (order : List ShoppingStep) : Prop :=
  order = shopping_process

-- Theorem stating the correct order
theorem correct_taobao_shopping_order :
  is_correct_order [ShoppingStep.select_products, ShoppingStep.buy_and_pay, 
                    ShoppingStep.ship_goods, ShoppingStep.receive_and_confirm, 
                    ShoppingStep.transfer_payment] :=
by
  sorry

#check correct_taobao_shopping_order

end NUMINAMATH_CALUDE_correct_taobao_shopping_order_l183_18334


namespace NUMINAMATH_CALUDE_ivar_water_planning_l183_18367

def water_planning (initial_horses : ℕ) (added_horses : ℕ) (drinking_water : ℕ) (bathing_water : ℕ) (total_water : ℕ) : ℕ :=
  let total_horses := initial_horses + added_horses
  let daily_consumption := total_horses * (drinking_water + bathing_water)
  total_water / daily_consumption

theorem ivar_water_planning :
  water_planning 3 5 5 2 1568 = 28 := by
  sorry

end NUMINAMATH_CALUDE_ivar_water_planning_l183_18367


namespace NUMINAMATH_CALUDE_fahrenheit_to_celsius_l183_18357

theorem fahrenheit_to_celsius (F C : ℚ) : F = (9 / 5) * C + 32 → F = 10 → C = -110 / 9 := by
  sorry

end NUMINAMATH_CALUDE_fahrenheit_to_celsius_l183_18357


namespace NUMINAMATH_CALUDE_num_ways_to_sum_equals_two_pow_n_minus_one_l183_18396

/-- The number of ways to express a positive integer as a sum of one or more positive integers. -/
def num_ways_to_sum (n : ℕ+) : ℕ :=
  2^(n.val - 1)

/-- Theorem: For any positive integer n, the number of ways to express n as a sum of one or more
    positive integers is equal to 2^(n-1). -/
theorem num_ways_to_sum_equals_two_pow_n_minus_one (n : ℕ+) :
  (num_ways_to_sum n) = 2^(n.val - 1) := by
  sorry

end NUMINAMATH_CALUDE_num_ways_to_sum_equals_two_pow_n_minus_one_l183_18396


namespace NUMINAMATH_CALUDE_min_throws_correct_l183_18323

/-- The number of faces on each die -/
def numFaces : ℕ := 6

/-- The number of dice thrown -/
def numDice : ℕ := 4

/-- The minimum possible sum when rolling the dice -/
def minSum : ℕ := numDice

/-- The maximum possible sum when rolling the dice -/
def maxSum : ℕ := numDice * numFaces

/-- The number of possible unique sums -/
def numUniqueSums : ℕ := maxSum - minSum + 1

/-- The minimum number of throws required to ensure the same sum is rolled twice -/
def minThrows : ℕ := numUniqueSums + 1

/-- Theorem stating that minThrows is the minimum number of throws required -/
theorem min_throws_correct :
  minThrows = 22 ∧
  ∀ n : ℕ, n < minThrows → ∃ outcome : Fin n → Fin (maxSum - minSum + 1),
    Function.Injective outcome :=
by sorry

end NUMINAMATH_CALUDE_min_throws_correct_l183_18323


namespace NUMINAMATH_CALUDE_dog_food_per_meal_l183_18320

/-- Calculates the amount of dog food each dog eats per meal given the total amount bought,
    amount left after a week, number of dogs, and number of meals per day. -/
theorem dog_food_per_meal
  (total_food : ℝ)
  (food_left : ℝ)
  (num_dogs : ℕ)
  (meals_per_day : ℕ)
  (days_per_week : ℕ)
  (h1 : total_food = 30)
  (h2 : food_left = 9)
  (h3 : num_dogs = 3)
  (h4 : meals_per_day = 2)
  (h5 : days_per_week = 7)
  : (total_food - food_left) / (num_dogs * meals_per_day * days_per_week) = 0.5 := by
  sorry

#check dog_food_per_meal

end NUMINAMATH_CALUDE_dog_food_per_meal_l183_18320


namespace NUMINAMATH_CALUDE_limit_x_minus_sin_x_ln_x_at_zero_l183_18346

theorem limit_x_minus_sin_x_ln_x_at_zero :
  ∀ ε > 0, ∃ δ > 0, ∀ x : ℝ, 0 < |x| ∧ |x| < δ → |(x - Real.sin x) * Real.log x| < ε :=
sorry

end NUMINAMATH_CALUDE_limit_x_minus_sin_x_ln_x_at_zero_l183_18346


namespace NUMINAMATH_CALUDE_first_group_size_l183_18304

/-- The number of days it takes the first group to complete the work -/
def first_group_days : ℕ := 30

/-- The number of days it takes 20 men to complete the work -/
def second_group_days : ℕ := 24

/-- The number of men in the second group -/
def second_group_men : ℕ := 20

/-- The number of men in the first group -/
def first_group_men : ℕ := (second_group_men * second_group_days) / first_group_days

theorem first_group_size :
  first_group_men = 16 :=
by sorry

end NUMINAMATH_CALUDE_first_group_size_l183_18304


namespace NUMINAMATH_CALUDE_interpretation_correct_l183_18379

-- Define propositions
variable (p : Prop)  -- Student A's math score is not less than 100 points
variable (q : Prop)  -- Student B's math score is less than 100 points

-- Define the interpretation of p∨(¬q)
def interpretation : Prop := p ∨ (¬q)

-- Theorem statement
theorem interpretation_correct : 
  interpretation p q ↔ (p ∨ ¬q) :=
sorry

end NUMINAMATH_CALUDE_interpretation_correct_l183_18379


namespace NUMINAMATH_CALUDE_correct_object_clause_introducer_l183_18389

-- Define a type for words that can introduce clauses
inductive ClauseIntroducer
  | That
  | What
  | Where
  | Which

-- Define a function to check if a word is the correct introducer for an object clause
def isCorrectObjectClauseIntroducer (word : ClauseIntroducer) : Prop :=
  word = ClauseIntroducer.What

-- Theorem stating that "what" is the correct word to introduce the object clause
theorem correct_object_clause_introducer :
  isCorrectObjectClauseIntroducer ClauseIntroducer.What :=
by sorry

end NUMINAMATH_CALUDE_correct_object_clause_introducer_l183_18389


namespace NUMINAMATH_CALUDE_remaining_sales_l183_18362

-- Define the weekly goal
def weekly_goal : ℕ := 90

-- Define Monday's sales
def monday_sales : ℕ := 45

-- Define Tuesday's sales
def tuesday_sales : ℕ := monday_sales - 16

-- Define the total sales so far
def total_sales : ℕ := monday_sales + tuesday_sales

-- Theorem to prove
theorem remaining_sales : weekly_goal - total_sales = 16 := by
  sorry

end NUMINAMATH_CALUDE_remaining_sales_l183_18362


namespace NUMINAMATH_CALUDE_original_sum_is_600_l183_18374

/-- Simple interest calculation function -/
def simpleInterest (principal rate time : ℝ) : ℝ :=
  principal * (1 + rate * time)

theorem original_sum_is_600 (P R : ℝ) 
  (h1 : simpleInterest P R 2 = 720)
  (h2 : simpleInterest P R 7 = 1020) : 
  P = 600 := by
  sorry

end NUMINAMATH_CALUDE_original_sum_is_600_l183_18374


namespace NUMINAMATH_CALUDE_books_per_shelf_l183_18337

theorem books_per_shelf (total_books : ℕ) (mystery_shelves : ℕ) (picture_shelves : ℕ) 
  (h1 : total_books = 72) 
  (h2 : mystery_shelves = 5) 
  (h3 : picture_shelves = 4) :
  ∃ (books_per_shelf : ℕ), 
    books_per_shelf * (mystery_shelves + picture_shelves) = total_books ∧ 
    books_per_shelf = 8 := by
  sorry

end NUMINAMATH_CALUDE_books_per_shelf_l183_18337


namespace NUMINAMATH_CALUDE_max_value_sine_cosine_l183_18348

/-- Given a function f(x) = a*sin(x) + 3*cos(x) where its maximum value is 5, 
    prove that a = ±4 -/
theorem max_value_sine_cosine (a : ℝ) :
  (∀ x, a * Real.sin x + 3 * Real.cos x ≤ 5) ∧ 
  (∃ x, a * Real.sin x + 3 * Real.cos x = 5) →
  a = 4 ∨ a = -4 := by
sorry

end NUMINAMATH_CALUDE_max_value_sine_cosine_l183_18348


namespace NUMINAMATH_CALUDE_non_degenerate_ellipse_condition_l183_18325

/-- The equation of an ellipse in general form -/
def ellipse_equation (x y k : ℝ) : Prop :=
  2 * x^2 + 9 * y^2 - 12 * x - 27 * y = k

/-- Condition for the equation to represent a non-degenerate ellipse -/
def is_non_degenerate_ellipse (k : ℝ) : Prop :=
  k > -135/4

/-- Theorem stating the condition for a non-degenerate ellipse -/
theorem non_degenerate_ellipse_condition :
  ∀ k, (∃ x y, ellipse_equation x y k) ∧ is_non_degenerate_ellipse k ↔
    (∀ x y, ellipse_equation x y k → is_non_degenerate_ellipse k) :=
by sorry

end NUMINAMATH_CALUDE_non_degenerate_ellipse_condition_l183_18325


namespace NUMINAMATH_CALUDE_rectangle_circle_ratio_l183_18373

theorem rectangle_circle_ratio (r : ℝ) (h : r > 0) : 
  ∃ (x y : ℝ), 
    x > 0 ∧ y > 0 ∧ 
    (x + 2*y)^2 = 16 * π * r^2 ∧ 
    y = r * Real.sqrt π ∧
    x / y = 2 :=
by sorry

end NUMINAMATH_CALUDE_rectangle_circle_ratio_l183_18373


namespace NUMINAMATH_CALUDE_at_least_one_is_diff_of_squares_l183_18354

theorem at_least_one_is_diff_of_squares (a b : ℕ) : 
  ∃ (x y z w : ℤ), (a = x^2 - y^2) ∨ (b = z^2 - w^2) ∨ (a + b = x^2 - y^2) := by
  sorry

end NUMINAMATH_CALUDE_at_least_one_is_diff_of_squares_l183_18354


namespace NUMINAMATH_CALUDE_class_test_probabilities_l183_18366

theorem class_test_probabilities (P_A P_B P_neither : ℝ)
  (h_A : P_A = 0.8)
  (h_B : P_B = 0.55)
  (h_neither : P_neither = 0.55) :
  P_A + P_B - (1 - P_neither) = 0.9 :=
by sorry

end NUMINAMATH_CALUDE_class_test_probabilities_l183_18366


namespace NUMINAMATH_CALUDE_floor_painted_by_all_colors_l183_18302

/-- Represents the percentage of floor painted by each painter -/
structure PainterCoverage where
  red : Real
  green : Real
  blue : Real

/-- Theorem: Given the paint coverage, at least 10% of the floor is painted by all three colors -/
theorem floor_painted_by_all_colors (coverage : PainterCoverage) 
  (h_red : coverage.red = 75)
  (h_green : coverage.green = -70)
  (h_blue : coverage.blue = -65) :
  ∃ (all_colors_coverage : Real),
    all_colors_coverage ≥ 10 ∧ 
    all_colors_coverage ≤ 100 ∧
    all_colors_coverage ≤ coverage.red ∧
    all_colors_coverage ≤ -coverage.green ∧
    all_colors_coverage ≤ -coverage.blue :=
sorry

end NUMINAMATH_CALUDE_floor_painted_by_all_colors_l183_18302


namespace NUMINAMATH_CALUDE_equation_solutions_l183_18347

-- Define the equation
def equation (x : ℂ) : Prop :=
  (x - 2)^4 + (x - 6)^4 = 32

-- State the theorem
theorem equation_solutions :
  ∀ x : ℂ, equation x ↔ (x = 4 ∨ x = 4 + 2*Complex.I*Real.sqrt 6 ∨ x = 4 - 2*Complex.I*Real.sqrt 6) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l183_18347


namespace NUMINAMATH_CALUDE_both_teasers_count_l183_18336

/-- The number of brainiacs who like both rebus teasers and math teasers -/
def both_teasers (total : ℕ) (rebus : ℕ) (math : ℕ) (neither : ℕ) (math_only : ℕ) : ℕ :=
  total - rebus - math + (rebus + math - (total - neither))

theorem both_teasers_count :
  both_teasers 100 (2 * 50) 50 4 20 = 18 := by
  sorry

end NUMINAMATH_CALUDE_both_teasers_count_l183_18336


namespace NUMINAMATH_CALUDE_movie_shelf_distribution_l183_18392

/-- The number of shelves in a movie store given the following conditions:
  * There are 9 movies in total
  * The owner wants to distribute the movies evenly among the shelves
  * The owner needs 1 more movie to achieve an even distribution
-/
def numShelves : ℕ := 4

theorem movie_shelf_distribution (total_movies : ℕ) (movies_needed : ℕ) : 
  total_movies = 9 → movies_needed = 1 → numShelves = 4 := by
  sorry

#check movie_shelf_distribution

end NUMINAMATH_CALUDE_movie_shelf_distribution_l183_18392


namespace NUMINAMATH_CALUDE_f_minimum_at_neg_nine_halves_l183_18376

-- Define the function
def f (x : ℝ) : ℝ := x^2 + 9*x + 7

-- State the theorem
theorem f_minimum_at_neg_nine_halves :
  ∃ (x_min : ℝ), ∀ (x : ℝ), f x_min ≤ f x ∧ x_min = -9/2 :=
sorry

end NUMINAMATH_CALUDE_f_minimum_at_neg_nine_halves_l183_18376


namespace NUMINAMATH_CALUDE_abs_equality_implies_geq_one_l183_18340

theorem abs_equality_implies_geq_one (m : ℝ) : |m - 1| = m - 1 → m ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_abs_equality_implies_geq_one_l183_18340


namespace NUMINAMATH_CALUDE_first_tribe_term_longer_l183_18317

/-- Represents the calendar system of the first tribe -/
structure Tribe1Calendar where
  months_per_year : Nat := 12
  days_per_month : Nat := 30

/-- Represents the calendar system of the second tribe -/
structure Tribe2Calendar where
  moons_per_year : Nat := 13
  weeks_per_moon : Nat := 4
  days_per_week : Nat := 7

/-- Calculates the number of days for the first tribe's term -/
def tribe1_term_days (cal : Tribe1Calendar) : Nat :=
  7 * cal.months_per_year * cal.days_per_month +
  1 * cal.days_per_month +
  18

/-- Calculates the number of days for the second tribe's term -/
def tribe2_term_days (cal : Tribe2Calendar) : Nat :=
  6 * cal.moons_per_year * cal.weeks_per_moon * cal.days_per_week +
  12 * cal.weeks_per_moon * cal.days_per_week +
  1 * cal.days_per_week +
  3

/-- Theorem stating that the first tribe's term is longer -/
theorem first_tribe_term_longer (cal1 : Tribe1Calendar) (cal2 : Tribe2Calendar) :
  tribe1_term_days cal1 > tribe2_term_days cal2 := by
  sorry

end NUMINAMATH_CALUDE_first_tribe_term_longer_l183_18317


namespace NUMINAMATH_CALUDE_floor_x_eq_1994_minus_n_l183_18385

def x : ℕ → ℚ
  | 0 => 1994
  | n + 1 => (x n)^2 / (x n + 1)

theorem floor_x_eq_1994_minus_n (n : ℕ) (h : n ≤ 998) :
  ⌊x n⌋ = 1994 - n :=
by sorry

end NUMINAMATH_CALUDE_floor_x_eq_1994_minus_n_l183_18385


namespace NUMINAMATH_CALUDE_problem_solution_l183_18364

theorem problem_solution (x y z : ℝ) (hx : x = 550) (hy : y = 104) (hz : z = Real.sqrt 20.8) :
  x - (y / z^2)^3 = 425 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l183_18364


namespace NUMINAMATH_CALUDE_dave_initial_boxes_l183_18395

def boxes_given : ℕ := 5
def pieces_per_box : ℕ := 3
def pieces_left : ℕ := 21

theorem dave_initial_boxes : 
  ∃ (initial_boxes : ℕ), 
    initial_boxes * pieces_per_box = 
      boxes_given * pieces_per_box + pieces_left ∧
    initial_boxes = 12 :=
by sorry

end NUMINAMATH_CALUDE_dave_initial_boxes_l183_18395


namespace NUMINAMATH_CALUDE_rectangle_width_is_fifteen_l183_18338

/-- Represents a rectangle with length and width in centimeters. -/
structure Rectangle where
  length : ℝ
  width : ℝ

/-- Calculates the perimeter of a rectangle. -/
def perimeter (r : Rectangle) : ℝ := 2 * (r.length + r.width)

/-- Theorem: For a rectangle where the width is 3 cm longer than the length
    and the perimeter is 54 cm, the width is 15 cm. -/
theorem rectangle_width_is_fifteen (r : Rectangle)
    (h1 : r.width = r.length + 3)
    (h2 : perimeter r = 54) :
    r.width = 15 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_width_is_fifteen_l183_18338


namespace NUMINAMATH_CALUDE_tournament_games_theorem_l183_18311

/-- Represents a single-elimination tournament. -/
structure Tournament :=
  (num_teams : ℕ)
  (no_ties : Bool)

/-- Calculates the number of games needed to determine a winner in a single-elimination tournament. -/
def games_to_win (t : Tournament) : ℕ :=
  t.num_teams - 1

/-- Theorem stating that a single-elimination tournament with 23 teams and no ties requires 22 games to determine a winner. -/
theorem tournament_games_theorem (t : Tournament) (h1 : t.num_teams = 23) (h2 : t.no_ties = true) : 
  games_to_win t = 22 := by
  sorry

end NUMINAMATH_CALUDE_tournament_games_theorem_l183_18311


namespace NUMINAMATH_CALUDE_sufficient_condition_implies_m_range_l183_18398

theorem sufficient_condition_implies_m_range (m : ℝ) : 
  (∀ x : ℝ, (x - 1) / x ≤ 0 → 4^x + 2^x - m ≤ 0) → m ≥ 6 := by
  sorry

end NUMINAMATH_CALUDE_sufficient_condition_implies_m_range_l183_18398


namespace NUMINAMATH_CALUDE_range_of_a_l183_18350

/-- Given propositions p and q, where p: x^2 + 2x - 3 > 0 and q: x > a,
    and a sufficient but not necessary condition for ¬q is ¬p,
    prove that the range of values for a is a ≥ 1 -/
theorem range_of_a (x a : ℝ) : 
  (∀ x, (x^2 + 2*x - 3 > 0 → x > a) ∧ 
       (x ≤ a → x^2 + 2*x - 3 ≤ 0)) → 
  a ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l183_18350


namespace NUMINAMATH_CALUDE_billy_crayons_l183_18324

/-- The number of crayons left after a monkey and hippopotamus eat some crayons -/
def crayons_left (total : ℕ) (monkey_ate : ℕ) : ℕ :=
  total - (monkey_ate + 2 * monkey_ate)

/-- Theorem stating that given 200 total crayons, if a monkey eats 64 crayons,
    then 8 crayons are left -/
theorem billy_crayons : crayons_left 200 64 = 8 := by
  sorry

end NUMINAMATH_CALUDE_billy_crayons_l183_18324


namespace NUMINAMATH_CALUDE_odd_function_product_negative_l183_18303

-- Define an odd function
def IsOdd (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

-- Theorem statement
theorem odd_function_product_negative
  (f : ℝ → ℝ)
  (h_odd : IsOdd f)
  (h_nonzero : ∀ x, f x ≠ 0) :
  ∀ x, f x * f (-x) < 0 :=
by sorry

end NUMINAMATH_CALUDE_odd_function_product_negative_l183_18303


namespace NUMINAMATH_CALUDE_least_k_for_inequality_l183_18370

theorem least_k_for_inequality : ∃ k : ℤ, k = 5 ∧ 
  (∀ n : ℤ, 0.0010101 * (10 : ℝ)^n > 10 → n ≥ k) ∧
  (0.0010101 * (10 : ℝ)^k > 10) := by
  sorry

end NUMINAMATH_CALUDE_least_k_for_inequality_l183_18370


namespace NUMINAMATH_CALUDE_fraction_irreducibility_l183_18315

/-- The fraction (3n^2 + 2n + 4) / (n + 1) is irreducible if and only if n is not congruent to 4 modulo 5 -/
theorem fraction_irreducibility (n : ℤ) : 
  (Int.gcd (3*n^2 + 2*n + 4) (n + 1) = 1) ↔ (n % 5 ≠ 4) := by
  sorry

end NUMINAMATH_CALUDE_fraction_irreducibility_l183_18315


namespace NUMINAMATH_CALUDE_max_x_minus_y_l183_18342

theorem max_x_minus_y (x y : ℝ) (h : x^2 + y^2 - 4*x - 2*y - 4 = 0) :
  ∃ (z : ℝ), z = x - y ∧ z ≤ 1 + 3 * Real.sqrt 2 ∧
  ∀ (w : ℝ), (∃ (a b : ℝ), a^2 + b^2 - 4*a - 2*b - 4 = 0 ∧ w = a - b) → w ≤ 1 + 3 * Real.sqrt 2 :=
sorry

end NUMINAMATH_CALUDE_max_x_minus_y_l183_18342


namespace NUMINAMATH_CALUDE_no_common_root_for_specific_quadratics_l183_18386

theorem no_common_root_for_specific_quadratics
  (a b c d : ℝ)
  (h_order : 0 < a ∧ a < b ∧ b < c ∧ c < d) :
  ¬∃ (x : ℝ), (x^2 + b*x + c = 0 ∧ x^2 + a*x + d = 0) :=
by sorry

end NUMINAMATH_CALUDE_no_common_root_for_specific_quadratics_l183_18386


namespace NUMINAMATH_CALUDE_lcm_gcf_product_24_150_l183_18307

theorem lcm_gcf_product_24_150 : Nat.lcm 24 150 * Nat.gcd 24 150 = 3600 := by
  sorry

end NUMINAMATH_CALUDE_lcm_gcf_product_24_150_l183_18307


namespace NUMINAMATH_CALUDE_smallest_b_for_factorization_l183_18383

theorem smallest_b_for_factorization : 
  ∃ (b : ℕ), b > 0 ∧ 
  (∃ (r s : ℤ), x^2 + b*x + 1512 = (x + r) * (x + s)) ∧
  (∀ (b' : ℕ), 0 < b' ∧ b' < b → 
    ¬∃ (r s : ℤ), x^2 + b'*x + 1512 = (x + r) * (x + s)) ∧
  b = 78 :=
sorry

end NUMINAMATH_CALUDE_smallest_b_for_factorization_l183_18383


namespace NUMINAMATH_CALUDE_five_variable_inequality_l183_18380

theorem five_variable_inequality (x₁ x₂ x₃ x₄ x₅ : ℝ) 
  (h₁ : x₁ > 0) (h₂ : x₂ > 0) (h₃ : x₃ > 0) (h₄ : x₄ > 0) (h₅ : x₅ > 0) : 
  (x₁ + x₂ + x₃ + x₄ + x₅)^2 > 4 * (x₁*x₂ + x₂*x₃ + x₃*x₄ + x₄*x₅ + x₅*x₁) := by
  sorry

end NUMINAMATH_CALUDE_five_variable_inequality_l183_18380


namespace NUMINAMATH_CALUDE_boys_percentage_in_class_l183_18339

theorem boys_percentage_in_class (total_students : ℕ) (boys_ratio girls_ratio : ℕ) 
  (h1 : total_students = 42)
  (h2 : boys_ratio = 3)
  (h3 : girls_ratio = 4) :
  (boys_ratio * total_students : ℚ) / ((boys_ratio + girls_ratio) * total_students) * 100 = 42857 / 1000 := by
  sorry

end NUMINAMATH_CALUDE_boys_percentage_in_class_l183_18339


namespace NUMINAMATH_CALUDE_biased_coin_prob_l183_18394

/-- The probability of getting heads for a biased coin -/
def h : ℚ := 2/5

/-- The number of flips -/
def n : ℕ := 4

/-- The probability of getting exactly k heads in n flips -/
def prob_k_heads (k : ℕ) : ℚ := 
  (n.choose k) * h^k * (1-h)^(n-k)

theorem biased_coin_prob : 
  prob_k_heads 1 = prob_k_heads 2 → 
  prob_k_heads 2 = 216/625 :=
by sorry

end NUMINAMATH_CALUDE_biased_coin_prob_l183_18394


namespace NUMINAMATH_CALUDE_ticket_price_difference_l183_18319

def prebought_count : ℕ := 20
def prebought_price : ℕ := 155
def gate_count : ℕ := 30
def gate_price : ℕ := 200

theorem ticket_price_difference : 
  gate_count * gate_price - prebought_count * prebought_price = 2900 := by
  sorry

end NUMINAMATH_CALUDE_ticket_price_difference_l183_18319


namespace NUMINAMATH_CALUDE_books_added_to_bin_l183_18322

theorem books_added_to_bin (initial_books : ℕ) (books_sold : ℕ) (final_books : ℕ)
  (h1 : initial_books = 4)
  (h2 : books_sold = 3)
  (h3 : final_books = 11)
  (h4 : initial_books ≥ books_sold) :
  final_books - (initial_books - books_sold) = 10 := by
  sorry

end NUMINAMATH_CALUDE_books_added_to_bin_l183_18322


namespace NUMINAMATH_CALUDE_vote_ways_l183_18361

/-- The number of ways an open vote can occur in a society of n members -/
def openVoteWays (n : ℕ) : ℕ := n^n

/-- The number of ways a secret vote can occur in a society of n members -/
def secretVoteWays (n : ℕ) : ℕ := Nat.choose (2*n - 1) (n - 1)

/-- Theorem stating the number of ways for open and secret votes in a society of n members -/
theorem vote_ways (n : ℕ) :
  (openVoteWays n = n^n) ∧ (secretVoteWays n = Nat.choose (2*n - 1) (n - 1)) := by
  sorry

end NUMINAMATH_CALUDE_vote_ways_l183_18361


namespace NUMINAMATH_CALUDE_apple_slice_packing_l183_18312

/-- The number of apple slices per group that satisfies the packing conditions -/
def apple_slices_per_group : ℕ := sorry

/-- The number of grapes per group -/
def grapes_per_group : ℕ := 9

/-- The smallest total number of grapes -/
def smallest_total_grapes : ℕ := 18

theorem apple_slice_packing :
  (apple_slices_per_group > 0) ∧
  (apple_slices_per_group * (smallest_total_grapes / grapes_per_group) = smallest_total_grapes) ∧
  (apple_slices_per_group ∣ smallest_total_grapes) ∧
  (grapes_per_group ∣ apple_slices_per_group * grapes_per_group) →
  apple_slices_per_group = 9 := by sorry

end NUMINAMATH_CALUDE_apple_slice_packing_l183_18312


namespace NUMINAMATH_CALUDE_money_division_l183_18360

/-- 
Given an amount of money divided between three people in the ratio 3:7:12,
where the difference between the first two shares is 4000,
prove that the difference between the second and third shares is 5000.
-/
theorem money_division (total : ℝ) : 
  let p := (3 / 22) * total
  let q := (7 / 22) * total
  let r := (12 / 22) * total
  q - p = 4000 → r - q = 5000 := by
sorry

end NUMINAMATH_CALUDE_money_division_l183_18360


namespace NUMINAMATH_CALUDE_divisibility_problem_l183_18314

theorem divisibility_problem (n : ℕ) : 
  n > 2 → 
  (((1 + n + n * (n - 1) / 2 + n * (n - 1) * (n - 2) / 6) ∣ 2^2000) ↔ (n = 3 ∨ n = 7 ∨ n = 23)) :=
by sorry

end NUMINAMATH_CALUDE_divisibility_problem_l183_18314


namespace NUMINAMATH_CALUDE_bear_discount_calculation_l183_18399

/-- The discount per bear after the first bear, given the price of the first bear,
    the total number of bears, and the total amount paid. -/
def discount_per_bear (first_bear_price : ℚ) (total_bears : ℕ) (total_paid : ℚ) : ℚ :=
  let full_price := first_bear_price * total_bears
  let discount := full_price - total_paid
  discount / (total_bears - 1)

/-- Theorem stating that under the given conditions, the discount per bear after the first bear is $0.50 -/
theorem bear_discount_calculation :
  let first_bear_price : ℚ := 4
  let total_bears : ℕ := 101
  let total_paid : ℚ := 354
  discount_per_bear first_bear_price total_bears total_paid = 1/2 := by
sorry


end NUMINAMATH_CALUDE_bear_discount_calculation_l183_18399


namespace NUMINAMATH_CALUDE_least_three_digit_multiple_of_11_l183_18310

theorem least_three_digit_multiple_of_11 : 
  ∀ n : ℕ, 100 ≤ n ∧ n < 1000 ∧ 11 ∣ n → 110 ≤ n :=
by sorry

end NUMINAMATH_CALUDE_least_three_digit_multiple_of_11_l183_18310


namespace NUMINAMATH_CALUDE_prob_rain_weekend_l183_18397

-- Define the probabilities
def prob_rain_sat : ℝ := 0.30
def prob_rain_sun : ℝ := 0.60
def prob_rain_sun_given_rain_sat : ℝ := 0.40

-- Define the theorem
theorem prob_rain_weekend : 
  let prob_no_rain_sat := 1 - prob_rain_sat
  let prob_no_rain_sun := 1 - prob_rain_sun
  let prob_no_rain_sun_given_rain_sat := 1 - prob_rain_sun_given_rain_sat
  let prob_no_rain_both := prob_no_rain_sat * prob_no_rain_sun
  let prob_rain_sat_no_rain_sun := prob_rain_sat * prob_no_rain_sun_given_rain_sat
  let prob_no_rain_all_scenarios := prob_no_rain_both + prob_rain_sat_no_rain_sun
  1 - prob_no_rain_all_scenarios = 0.54 :=
by
  sorry

#check prob_rain_weekend

end NUMINAMATH_CALUDE_prob_rain_weekend_l183_18397


namespace NUMINAMATH_CALUDE_managers_salary_l183_18355

theorem managers_salary (num_employees : ℕ) (avg_salary : ℝ) (salary_increase : ℝ) :
  num_employees = 20 ∧ 
  avg_salary = 1200 ∧ 
  salary_increase = 100 → 
  (num_employees * avg_salary + (avg_salary + salary_increase) * (num_employees + 1) - num_employees * avg_salary) = 3300 := by
  sorry

end NUMINAMATH_CALUDE_managers_salary_l183_18355


namespace NUMINAMATH_CALUDE_same_city_probability_l183_18378

/-- The probability that two specific students are assigned to the same city
    given the total number of students and the number of spots in each city. -/
theorem same_city_probability
  (total_students : ℕ)
  (spots_moscow : ℕ)
  (spots_tula : ℕ)
  (spots_voronezh : ℕ)
  (h1 : total_students = 30)
  (h2 : spots_moscow = 15)
  (h3 : spots_tula = 8)
  (h4 : spots_voronezh = 7)
  (h5 : total_students = spots_moscow + spots_tula + spots_voronezh) :
  (spots_moscow.choose 2 + spots_tula.choose 2 + spots_voronezh.choose 2) / total_students.choose 2 = 154 / 435 :=
by sorry

end NUMINAMATH_CALUDE_same_city_probability_l183_18378


namespace NUMINAMATH_CALUDE_complex_square_root_l183_18384

theorem complex_square_root (z : ℂ) : z^2 = -4 ∧ z.im > 0 → z = 2*I :=
sorry

end NUMINAMATH_CALUDE_complex_square_root_l183_18384


namespace NUMINAMATH_CALUDE_sqrt_2023_divided_by_sum_of_digits_l183_18368

theorem sqrt_2023_divided_by_sum_of_digits : Real.sqrt (2023 / (2 + 0 + 2 + 3)) = 17 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_2023_divided_by_sum_of_digits_l183_18368


namespace NUMINAMATH_CALUDE_staff_distribution_theorem_l183_18371

def distribute_staff (n : ℕ) (k : ℕ) : ℕ :=
  let arrangements := (n.choose 1 * (n-1).choose 1) / 2 +
                      (n.choose 2 * (n-2).choose 2) / 2 +
                      (n.choose 3 * (n-3).choose 3) / 2
  arrangements * (k.factorial)

theorem staff_distribution_theorem :
  distribute_staff 7 3 = 1176 := by
  sorry

end NUMINAMATH_CALUDE_staff_distribution_theorem_l183_18371


namespace NUMINAMATH_CALUDE_max_value_of_ab_l183_18344

theorem max_value_of_ab (a b : ℝ) : 
  (Real.sqrt 3 = Real.sqrt (3^a * 3^b)) → (∀ x y : ℝ, (Real.sqrt 3 = Real.sqrt (3^x * 3^y)) → a * b ≥ x * y) → 
  a * b = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_max_value_of_ab_l183_18344


namespace NUMINAMATH_CALUDE_negation_of_exp_gt_ln_proposition_l183_18330

open Real

theorem negation_of_exp_gt_ln_proposition :
  (¬ ∀ x : ℝ, x > 0 → exp x > log x) ↔ (∃ x : ℝ, x > 0 ∧ exp x ≤ log x) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_exp_gt_ln_proposition_l183_18330


namespace NUMINAMATH_CALUDE_cubic_sum_reciprocal_l183_18300

theorem cubic_sum_reciprocal (x : ℝ) (h : x^3 + 1/x^3 = 110) : x + 1/x = 5 := by
  sorry

end NUMINAMATH_CALUDE_cubic_sum_reciprocal_l183_18300


namespace NUMINAMATH_CALUDE_science_homework_duration_l183_18301

/-- Calculates the time remaining for science homework given the total time and time spent on other subjects. -/
def science_homework_time (total_time math_time english_time history_time project_time : ℕ) : ℕ :=
  total_time - (math_time + english_time + history_time + project_time)

/-- Proves that given the specified times for total work and other subjects, the remaining time for science homework is 50 minutes. -/
theorem science_homework_duration :
  science_homework_time 180 45 30 25 30 = 50 := by
  sorry

end NUMINAMATH_CALUDE_science_homework_duration_l183_18301


namespace NUMINAMATH_CALUDE_exists_valid_five_by_five_division_l183_18318

/-- Represents a square grid -/
structure SquareGrid :=
  (side : ℕ)

/-- Represents a division of a square grid -/
structure GridDivision :=
  (grid : SquareGrid)
  (num_parts : ℕ)
  (segment_length : ℕ)

/-- Checks if a division of a square grid is valid -/
def is_valid_division (d : GridDivision) : Prop :=
  d.grid.side * d.grid.side % d.num_parts = 0 ∧
  d.segment_length ≤ 16

/-- Theorem: There exists a valid division of a 5x5 square grid into 5 equal parts
    with total segment length not exceeding 16 units -/
theorem exists_valid_five_by_five_division :
  ∃ (d : GridDivision), d.grid.side = 5 ∧ d.num_parts = 5 ∧ is_valid_division d :=
sorry

end NUMINAMATH_CALUDE_exists_valid_five_by_five_division_l183_18318


namespace NUMINAMATH_CALUDE_first_piece_cost_l183_18390

/-- Given the total spent on clothing, the number of pieces, and the prices of some pieces,
    prove the cost of the first piece. -/
theorem first_piece_cost (total : ℕ) (num_pieces : ℕ) (price_one : ℕ) (price_others : ℕ) :
  total = 610 →
  num_pieces = 7 →
  price_one = 81 →
  price_others = 96 →
  ∃ (first_piece : ℕ), first_piece + price_one + (num_pieces - 2) * price_others = total ∧ first_piece = 49 := by
  sorry

end NUMINAMATH_CALUDE_first_piece_cost_l183_18390


namespace NUMINAMATH_CALUDE_ball_rolling_cycloid_l183_18327

/-- Represents the path of a ball rolling down a smooth cycloidal trough -/
noncomputable def path (a g t : ℝ) : ℝ :=
  4 * a * (1 - Real.cos (t * Real.sqrt (g / (4 * a))))

/-- Time for the ball to roll from the start to the lowest point along the cycloid -/
noncomputable def time_cycloid (a g : ℝ) : ℝ :=
  Real.pi * Real.sqrt (a / g)

/-- Time for the ball to roll from the start to the lowest point along a straight line -/
noncomputable def time_straight (a g : ℝ) : ℝ :=
  Real.sqrt (a * (4 + Real.pi^2) / g)

theorem ball_rolling_cycloid (a g : ℝ) (ha : a > 0) (hg : g > 0) :
  (∀ t, path a g t = 4 * a * (1 - Real.cos (t * Real.sqrt (g / (4 * a))))) ∧
  time_cycloid a g = Real.pi * Real.sqrt (a / g) ∧
  time_straight a g = Real.sqrt (a * (4 + Real.pi^2) / g) ∧
  time_cycloid a g < time_straight a g :=
sorry

end NUMINAMATH_CALUDE_ball_rolling_cycloid_l183_18327


namespace NUMINAMATH_CALUDE_baseball_cards_per_pack_l183_18349

theorem baseball_cards_per_pack : 
  ∀ (total_cards : ℕ) (num_people : ℕ) (cards_per_person : ℕ) (total_packs : ℕ),
    num_people = 4 →
    cards_per_person = 540 →
    total_packs = 108 →
    total_cards = num_people * cards_per_person →
    total_cards / total_packs = 20 :=
by
  sorry

end NUMINAMATH_CALUDE_baseball_cards_per_pack_l183_18349


namespace NUMINAMATH_CALUDE_solve_equation_l183_18363

theorem solve_equation (x : ℝ) (h : 5 - 5/x = 4 + 4/x) : x = 9 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l183_18363


namespace NUMINAMATH_CALUDE_halfway_fraction_l183_18326

theorem halfway_fraction : (3 / 4 + 5 / 6) / 2 = 19 / 24 := by
  sorry

end NUMINAMATH_CALUDE_halfway_fraction_l183_18326


namespace NUMINAMATH_CALUDE_company_employee_count_l183_18391

theorem company_employee_count (december_count : ℕ) (percent_increase : ℚ) : 
  december_count = 480 → percent_increase = 15 / 100 → 
  ∃ (january_count : ℕ), 
    (↑december_count : ℚ) = (1 + percent_increase) * ↑january_count ∧ 
    january_count = 417 := by
  sorry

end NUMINAMATH_CALUDE_company_employee_count_l183_18391


namespace NUMINAMATH_CALUDE_cars_with_no_features_l183_18328

theorem cars_with_no_features (total : ℕ) (airbags : ℕ) (power_windows : ℕ) (sunroofs : ℕ)
  (airbags_power : ℕ) (airbags_sunroofs : ℕ) (power_sunroofs : ℕ) (all_features : ℕ) :
  total = 80 →
  airbags = 45 →
  power_windows = 40 →
  sunroofs = 25 →
  airbags_power = 20 →
  airbags_sunroofs = 15 →
  power_sunroofs = 10 →
  all_features = 8 →
  total - (airbags + power_windows + sunroofs - airbags_power - airbags_sunroofs - power_sunroofs + all_features) = 7 :=
by sorry

end NUMINAMATH_CALUDE_cars_with_no_features_l183_18328


namespace NUMINAMATH_CALUDE_min_vertical_distance_l183_18341

-- Define the two functions
def f (x : ℝ) : ℝ := |x|
def g (x : ℝ) : ℝ := -x^2 + 2*x - 1

-- Define the vertical distance between the two functions
def vertical_distance (x : ℝ) : ℝ := |f x - g x|

-- Theorem statement
theorem min_vertical_distance :
  ∃ (x₀ : ℝ), vertical_distance x₀ = 3/4 ∧
  ∀ (x : ℝ), vertical_distance x ≥ 3/4 := by
sorry

end NUMINAMATH_CALUDE_min_vertical_distance_l183_18341


namespace NUMINAMATH_CALUDE_expression_evaluation_l183_18352

theorem expression_evaluation : 
  (0 : ℝ) - 2 - 2 * Real.sin (45 * π / 180) + (π - 3.14) * 0 + (-1)^3 = -3 - Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l183_18352
