import Mathlib

namespace NUMINAMATH_CALUDE_system_solution_l3679_367989

theorem system_solution (x y b : ℝ) : 
  (4 * x + y = b) → 
  (3 * x + 4 * y = 3 * b) → 
  (x = 3) → 
  (b = 39) := by
sorry

end NUMINAMATH_CALUDE_system_solution_l3679_367989


namespace NUMINAMATH_CALUDE_bracelet_profit_l3679_367925

/-- Given the following conditions:
    - Total bracelets made
    - Number of bracelets given away
    - Cost of materials
    - Selling price per bracelet
    Prove that the profit equals $8.00 -/
theorem bracelet_profit 
  (total_bracelets : ℕ) 
  (given_away : ℕ) 
  (material_cost : ℚ) 
  (price_per_bracelet : ℚ) 
  (h1 : total_bracelets = 52)
  (h2 : given_away = 8)
  (h3 : material_cost = 3)
  (h4 : price_per_bracelet = 1/4) : 
  (total_bracelets - given_away : ℚ) * price_per_bracelet - material_cost = 8 := by
  sorry

end NUMINAMATH_CALUDE_bracelet_profit_l3679_367925


namespace NUMINAMATH_CALUDE_cubic_root_sum_l3679_367971

theorem cubic_root_sum (c d : ℝ) : 
  (Complex.I * Real.sqrt 2 + 2 : ℂ) ^ 3 + c * (Complex.I * Real.sqrt 2 + 2) + d = 0 →
  c + d = 14 := by
sorry

end NUMINAMATH_CALUDE_cubic_root_sum_l3679_367971


namespace NUMINAMATH_CALUDE_perpendicular_vectors_vector_sum_magnitude_l3679_367950

def a : ℝ × ℝ := (2, 4)
def b (m : ℝ) : ℝ × ℝ := (m, -1)

theorem perpendicular_vectors (m : ℝ) :
  (a.1 * (b m).1 + a.2 * (b m).2 = 0) → m = 2 := by sorry

theorem vector_sum_magnitude (m : ℝ) :
  ((a.1 + (b m).1)^2 + (a.2 + (b m).2)^2 = 25) → (m = 2 ∨ m = -6) := by sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_vector_sum_magnitude_l3679_367950


namespace NUMINAMATH_CALUDE_pigeons_flew_in_l3679_367995

theorem pigeons_flew_in (initial_count final_count : ℕ) 
  (h_initial : initial_count = 15)
  (h_final : final_count = 21) :
  final_count - initial_count = 6 := by
  sorry

end NUMINAMATH_CALUDE_pigeons_flew_in_l3679_367995


namespace NUMINAMATH_CALUDE_sequence_difference_l3679_367952

theorem sequence_difference (a : ℕ → ℤ) (h : ∀ n, a (n + 1) - a n - n = 0) : 
  a 2017 - a 2016 = 2016 := by
sorry

end NUMINAMATH_CALUDE_sequence_difference_l3679_367952


namespace NUMINAMATH_CALUDE_sqrt_difference_equality_l3679_367968

theorem sqrt_difference_equality : Real.sqrt (49 + 81) - Real.sqrt (36 - 9) = Real.sqrt 130 - Real.sqrt 27 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_difference_equality_l3679_367968


namespace NUMINAMATH_CALUDE_sequence_general_term_l3679_367973

def S (n : ℕ+) (a : ℕ+ → ℚ) : ℚ := 2 * n.val - a n

theorem sequence_general_term (a : ℕ+ → ℚ)
  (h : ∀ n : ℕ+, S n a = (n.val : ℚ)) :
  ∀ n : ℕ+, a n = (2^n.val - 1) / 2^(n.val - 1) :=
sorry

end NUMINAMATH_CALUDE_sequence_general_term_l3679_367973


namespace NUMINAMATH_CALUDE_balance_after_school_days_l3679_367965

/-- Represents the balance after spending money for a certain number of days. -/
def balance (initial_balance : ℝ) (daily_spending : ℝ) (days : ℝ) : ℝ :=
  initial_balance - daily_spending * days

/-- Theorem stating the relationship between balance and days spent at school. -/
theorem balance_after_school_days 
  (initial_balance : ℝ) 
  (daily_spending : ℝ) 
  (days : ℝ) 
  (h1 : initial_balance = 200)
  (h2 : daily_spending = 36)
  (h3 : 0 ≤ days)
  (h4 : days ≤ 5) :
  balance initial_balance daily_spending days = 200 - 36 * days :=
by sorry

end NUMINAMATH_CALUDE_balance_after_school_days_l3679_367965


namespace NUMINAMATH_CALUDE_floor_painting_problem_l3679_367956

def is_valid_pair (a b : ℕ) : Prop :=
  b > a ∧ 
  (a - 4) * (b - 4) = 2 * a * b / 3 ∧
  a > 4 ∧ b > 4

theorem floor_painting_problem :
  ∃! (pairs : List (ℕ × ℕ)), 
    pairs.length = 3 ∧ 
    ∀ (p : ℕ × ℕ), p ∈ pairs ↔ is_valid_pair p.1 p.2 :=
by sorry

end NUMINAMATH_CALUDE_floor_painting_problem_l3679_367956


namespace NUMINAMATH_CALUDE_girls_in_school_l3679_367923

/-- Proves the number of girls in a school given stratified sampling conditions -/
theorem girls_in_school (total_students : ℕ) (sample_size : ℕ) (girls_boys_diff : ℕ) :
  total_students = 2400 →
  sample_size = 200 →
  girls_boys_diff = 10 →
  ∃ (girls_in_sample : ℕ) (girls_in_school : ℕ),
    girls_in_sample + (girls_in_sample + girls_boys_diff) = sample_size ∧
    (girls_in_sample : ℚ) / sample_size = (girls_in_school : ℚ) / total_students ∧
    girls_in_school = 1140 :=
by sorry

end NUMINAMATH_CALUDE_girls_in_school_l3679_367923


namespace NUMINAMATH_CALUDE_alison_has_4000_l3679_367982

-- Define the amounts of money for each person
def kent_money : ℕ := 1000
def brooke_money : ℕ := 2 * kent_money
def brittany_money : ℕ := 4 * brooke_money
def alison_money : ℕ := brittany_money / 2

-- Theorem statement
theorem alison_has_4000 : alison_money = 4000 := by
  sorry

end NUMINAMATH_CALUDE_alison_has_4000_l3679_367982


namespace NUMINAMATH_CALUDE_percentage_of_males_l3679_367961

theorem percentage_of_males (total_employees : ℕ) (males_below_50 : ℕ) 
  (h1 : total_employees = 2200)
  (h2 : males_below_50 = 616)
  (h3 : (70 : ℚ) / 100 * (males_below_50 / ((70 : ℚ) / 100)) = males_below_50) :
  (males_below_50 / ((70 : ℚ) / 100)) / total_employees = (40 : ℚ) / 100 := by
  sorry

end NUMINAMATH_CALUDE_percentage_of_males_l3679_367961


namespace NUMINAMATH_CALUDE_min_value_abc_l3679_367918

theorem min_value_abc (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) 
  (h : 2*a + 4*b + 7*c ≤ 2*a*b*c) : 
  a + b + c ≥ 15/2 := by
  sorry


end NUMINAMATH_CALUDE_min_value_abc_l3679_367918


namespace NUMINAMATH_CALUDE_apartment_buildings_count_l3679_367941

/-- The number of floors in each apartment building -/
def floors_per_building : ℕ := 12

/-- The number of apartments on each floor -/
def apartments_per_floor : ℕ := 6

/-- The number of doors needed for each apartment -/
def doors_per_apartment : ℕ := 7

/-- The total number of doors needed to be bought -/
def total_doors : ℕ := 1008

/-- The number of apartment buildings being constructed -/
def num_buildings : ℕ := total_doors / (floors_per_building * apartments_per_floor * doors_per_apartment)

theorem apartment_buildings_count : num_buildings = 2 := by
  sorry

end NUMINAMATH_CALUDE_apartment_buildings_count_l3679_367941


namespace NUMINAMATH_CALUDE_rectangle_dimension_change_l3679_367946

theorem rectangle_dimension_change (L B : ℝ) (x : ℝ) (h_pos_L : L > 0) (h_pos_B : B > 0) :
  let new_length := L * (1 + x / 100)
  let new_breadth := B * 0.9
  let new_area := new_length * new_breadth
  let original_area := L * B
  new_area = original_area * 1.035 → x = 15 := by
sorry

end NUMINAMATH_CALUDE_rectangle_dimension_change_l3679_367946


namespace NUMINAMATH_CALUDE_percentage_relation_l3679_367914

theorem percentage_relation (x : ℝ) (h : 0.4 * x = 160) : 0.6 * x = 240 := by
  sorry

end NUMINAMATH_CALUDE_percentage_relation_l3679_367914


namespace NUMINAMATH_CALUDE_value_of_a_l3679_367962

def A (a : ℝ) : Set ℝ := {0, 2, a}
def B (a : ℝ) : Set ℝ := {1, a^2}

theorem value_of_a : ∀ a : ℝ, A a ∪ B a = {0, 1, 2, 4, 16} → a = 4 := by
  sorry

end NUMINAMATH_CALUDE_value_of_a_l3679_367962


namespace NUMINAMATH_CALUDE_no_real_solutions_for_inequality_l3679_367948

theorem no_real_solutions_for_inequality :
  ¬ ∃ x : ℝ, -x^2 + 2*x - 3 > 0 := by
sorry

end NUMINAMATH_CALUDE_no_real_solutions_for_inequality_l3679_367948


namespace NUMINAMATH_CALUDE_perpendicular_line_parallel_planes_l3679_367940

-- Define the types for lines and planes
variable (Line Plane : Type)

-- Define the relations
variable (perpendicular : Line → Line → Prop)
variable (perpendicularToPlane : Line → Plane → Prop)
variable (parallel : Plane → Plane → Prop)
variable (containedIn : Line → Plane → Prop)

-- State the theorem
theorem perpendicular_line_parallel_planes 
  (l m n : Line) (α β : Plane) 
  (h1 : l ≠ m) (h2 : l ≠ n) (h3 : m ≠ n) (h4 : α ≠ β)
  (h5 : perpendicularToPlane l α) 
  (h6 : parallel α β) 
  (h7 : containedIn m β) : 
  perpendicular l m :=
sorry

end NUMINAMATH_CALUDE_perpendicular_line_parallel_planes_l3679_367940


namespace NUMINAMATH_CALUDE_brianna_marbles_l3679_367958

/-- The number of marbles Brianna has remaining after a series of events -/
def remaining_marbles (initial : ℕ) (lost : ℕ) : ℕ :=
  initial - lost - (2 * lost) - (lost / 2)

/-- Theorem stating that Brianna has 10 marbles remaining -/
theorem brianna_marbles : remaining_marbles 24 4 = 10 := by
  sorry

end NUMINAMATH_CALUDE_brianna_marbles_l3679_367958


namespace NUMINAMATH_CALUDE_lattice_points_count_is_35_l3679_367906

/-- The number of lattice points in the region bounded by the x-axis, 
    the line x=4, and the parabola y=x^2 -/
def lattice_points_count : ℕ :=
  (Finset.range 5).sum (λ x => x^2 + 1)

/-- The theorem stating that the number of lattice points in the specified region is 35 -/
theorem lattice_points_count_is_35 : lattice_points_count = 35 := by
  sorry

end NUMINAMATH_CALUDE_lattice_points_count_is_35_l3679_367906


namespace NUMINAMATH_CALUDE_inequality_proof_l3679_367974

theorem inequality_proof (x y : ℝ) (h : x ≠ 0 ∨ y ≠ 0) :
  (x + y) / (x^2 - x*y + y^2) ≤ (2 * Real.sqrt 2) / Real.sqrt (x^2 + y^2) := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3679_367974


namespace NUMINAMATH_CALUDE_f_3_equals_6_l3679_367907

def f : ℕ → ℕ
  | 0 => 1
  | n + 1 => (n + 1) * f n

theorem f_3_equals_6 : f 3 = 6 := by
  sorry

end NUMINAMATH_CALUDE_f_3_equals_6_l3679_367907


namespace NUMINAMATH_CALUDE_machine_tool_supervision_probability_l3679_367922

theorem machine_tool_supervision_probability :
  let p_no_supervision : ℝ := 0.8000
  let n_tools : ℕ := 4
  let p_at_most_two_require_supervision : ℝ := 1 - (Nat.choose n_tools 3 * (1 - p_no_supervision)^3 * p_no_supervision + Nat.choose n_tools 4 * (1 - p_no_supervision)^4)
  p_at_most_two_require_supervision = 0.9728 := by
sorry

end NUMINAMATH_CALUDE_machine_tool_supervision_probability_l3679_367922


namespace NUMINAMATH_CALUDE_farm_water_consumption_l3679_367977

/-- Calculates the total weekly water consumption for Mr. Reyansh's farm animals -/
theorem farm_water_consumption : 
  let num_cows : ℕ := 40
  let num_goats : ℕ := 25
  let num_pigs : ℕ := 30
  let cow_water : ℕ := 80
  let goat_water : ℕ := cow_water / 2
  let pig_water : ℕ := cow_water / 3
  let num_sheep : ℕ := num_cows * 10
  let sheep_water : ℕ := cow_water / 4
  let daily_consumption : ℕ := 
    num_cows * cow_water + 
    num_goats * goat_water + 
    num_pigs * pig_water + 
    num_sheep * sheep_water
  let weekly_consumption : ℕ := daily_consumption * 7
  weekly_consumption = 91000 := by
  sorry

end NUMINAMATH_CALUDE_farm_water_consumption_l3679_367977


namespace NUMINAMATH_CALUDE_tangent_line_to_ln_l3679_367967

-- Define the natural logarithm function
noncomputable def ln : ℝ → ℝ := Real.log

-- Define the tangent line
def tangent_line (a : ℝ) (x : ℝ) : ℝ := x + a

-- State the theorem
theorem tangent_line_to_ln (a : ℝ) :
  (∃ x : ℝ, x > 0 ∧ 
    tangent_line a x = ln x ∧ 
    (∀ y : ℝ, y > 0 → tangent_line a y ≥ ln y)) →
  a = -1 :=
sorry

end NUMINAMATH_CALUDE_tangent_line_to_ln_l3679_367967


namespace NUMINAMATH_CALUDE_trigonometric_identity_l3679_367955

theorem trigonometric_identity (x y : ℝ) :
  Real.sin (x - y + π/6) * Real.cos (y + π/6) + Real.cos (x - y + π/6) * Real.sin (y + π/6) = Real.sin (x + π/3) := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_identity_l3679_367955


namespace NUMINAMATH_CALUDE_square_one_on_top_l3679_367931

/-- Represents the possible positions of a square after folding and rotation. -/
inductive Position
  | TopLeft | TopMiddle | TopRight
  | MiddleLeft | Center | MiddleRight
  | BottomLeft | BottomMiddle | BottomRight

/-- Represents the state of the grid after each operation. -/
structure GridState :=
  (positions : Fin 9 → Position)

/-- Folds the right half over the left half. -/
def foldRightOverLeft (state : GridState) : GridState := sorry

/-- Folds the top half over the bottom half. -/
def foldTopOverBottom (state : GridState) : GridState := sorry

/-- Folds the left half over the right half. -/
def foldLeftOverRight (state : GridState) : GridState := sorry

/-- Rotates the entire grid 90 degrees clockwise. -/
def rotateClockwise (state : GridState) : GridState := sorry

/-- The initial state of the grid. -/
def initialState : GridState :=
  { positions := λ i => match i with
    | 0 => Position.TopLeft
    | 1 => Position.TopMiddle
    | 2 => Position.TopRight
    | 3 => Position.MiddleLeft
    | 4 => Position.Center
    | 5 => Position.MiddleRight
    | 6 => Position.BottomLeft
    | 7 => Position.BottomMiddle
    | 8 => Position.BottomRight }

theorem square_one_on_top :
  (rotateClockwise (foldLeftOverRight (foldTopOverBottom (foldRightOverLeft initialState)))).positions 0 = Position.TopLeft := by
  sorry

end NUMINAMATH_CALUDE_square_one_on_top_l3679_367931


namespace NUMINAMATH_CALUDE_inclination_angle_range_l3679_367984

-- Define the slope range
def slope_range : Set ℝ := {k : ℝ | -Real.sqrt 3 ≤ k ∧ k ≤ Real.sqrt 3 / 3}

-- Define the inclination angle range
def angle_range : Set ℝ := {α : ℝ | (0 ≤ α ∧ α ≤ Real.pi / 6) ∨ (2 * Real.pi / 3 ≤ α ∧ α < Real.pi)}

-- Theorem statement
theorem inclination_angle_range (k : ℝ) (α : ℝ) :
  k ∈ slope_range → α = Real.arctan k → α ∈ angle_range := by sorry

end NUMINAMATH_CALUDE_inclination_angle_range_l3679_367984


namespace NUMINAMATH_CALUDE_eggs_in_box_l3679_367953

/-- Given an initial count of eggs and a number of whole eggs added, 
    calculate the total number of whole eggs, ignoring fractional parts. -/
def total_whole_eggs (initial : ℕ) (added : ℕ) : ℕ :=
  initial + added

/-- Theorem stating that with 7 initial eggs and 3 added whole eggs, 
    the total number of whole eggs is 10. -/
theorem eggs_in_box : total_whole_eggs 7 3 = 10 := by
  sorry

end NUMINAMATH_CALUDE_eggs_in_box_l3679_367953


namespace NUMINAMATH_CALUDE_grid_midpoint_theorem_l3679_367938

theorem grid_midpoint_theorem (points : Finset (ℤ × ℤ)) :
  points.card = 5 →
  ∃ p1 p2 : ℤ × ℤ, p1 ∈ points ∧ p2 ∈ points ∧ p1 ≠ p2 ∧
  (∃ x y : ℤ, ((p1.1 + p2.1) / 2 : ℚ) = x ∧ ((p1.2 + p2.2) / 2 : ℚ) = y) :=
by sorry

end NUMINAMATH_CALUDE_grid_midpoint_theorem_l3679_367938


namespace NUMINAMATH_CALUDE_angle_CAD_measure_l3679_367939

-- Define the points
variable (A B C D E : ℝ × ℝ)

-- Define the properties of the triangle and square
def is_right_triangle (A B C : ℝ × ℝ) : Prop := sorry
def is_isosceles (A B C : ℝ × ℝ) : Prop := sorry
def is_square (B C D E : ℝ × ℝ) : Prop := sorry

-- Define angle measurement function
def angle_measure (A B C : ℝ × ℝ) : ℝ := sorry

-- Theorem statement
theorem angle_CAD_measure 
  (h_right : is_right_triangle A B C)
  (h_isosceles : is_isosceles A B C)
  (h_square : is_square B C D E) :
  angle_measure C A D = 22.5 := by sorry

end NUMINAMATH_CALUDE_angle_CAD_measure_l3679_367939


namespace NUMINAMATH_CALUDE_train_delivery_wood_cars_l3679_367902

/-- Represents the train's cargo and delivery parameters -/
structure TrainDelivery where
  coal_cars : ℕ
  iron_cars : ℕ
  station_distance : ℕ
  travel_time : ℕ
  max_coal_deposit : ℕ
  max_iron_deposit : ℕ
  max_wood_deposit : ℕ
  total_delivery_time : ℕ

/-- Calculates the initial number of wood cars -/
def initial_wood_cars (td : TrainDelivery) : ℕ :=
  (td.total_delivery_time / td.travel_time) * td.max_wood_deposit

/-- Theorem stating that given the problem conditions, the initial number of wood cars is 4 -/
theorem train_delivery_wood_cars :
  let td : TrainDelivery := {
    coal_cars := 6,
    iron_cars := 12,
    station_distance := 6,
    travel_time := 25,
    max_coal_deposit := 2,
    max_iron_deposit := 3,
    max_wood_deposit := 1,
    total_delivery_time := 100
  }
  initial_wood_cars td = 4 := by
  sorry


end NUMINAMATH_CALUDE_train_delivery_wood_cars_l3679_367902


namespace NUMINAMATH_CALUDE_three_number_problem_l3679_367920

theorem three_number_problem (x y z : ℚ) : 
  x + (1/3) * z = y ∧ 
  y + (1/3) * x = z ∧ 
  z - x = 10 → 
  x = 10 ∧ y = 50/3 ∧ z = 20 := by
sorry

end NUMINAMATH_CALUDE_three_number_problem_l3679_367920


namespace NUMINAMATH_CALUDE_diamonds_G6_l3679_367969

/-- The k-th triangular number -/
def T (k : ℕ) : ℕ := k * (k + 1) / 2

/-- The number of diamonds in the n-th figure -/
def diamonds (n : ℕ) : ℕ :=
  1 + 4 * (Finset.sum (Finset.range (n - 1)) (λ i => T (i + 1)))

/-- The theorem stating that the number of diamonds in G_6 is 141 -/
theorem diamonds_G6 : diamonds 6 = 141 := by
  sorry

end NUMINAMATH_CALUDE_diamonds_G6_l3679_367969


namespace NUMINAMATH_CALUDE_tangent_slope_exponential_l3679_367994

theorem tangent_slope_exponential (x : ℝ) :
  let f : ℝ → ℝ := fun x => Real.exp x
  (deriv f) 1 = Real.exp 1 := by sorry

end NUMINAMATH_CALUDE_tangent_slope_exponential_l3679_367994


namespace NUMINAMATH_CALUDE_friends_money_distribution_l3679_367954

structure Friend :=
  (name : String)
  (initialMoney : ℚ)

def giveMoneyTo (giver receiver : Friend) (fraction : ℚ) : ℚ :=
  giver.initialMoney * fraction

theorem friends_money_distribution (loki moe nick ott pam : Friend) 
  (h1 : ott.initialMoney = 0)
  (h2 : pam.initialMoney = 0)
  (h3 : giveMoneyTo moe ott (1/6) = giveMoneyTo loki ott (1/5))
  (h4 : giveMoneyTo moe ott (1/6) = giveMoneyTo nick ott (1/4))
  (h5 : giveMoneyTo moe pam (1/6) = giveMoneyTo loki pam (1/5))
  (h6 : giveMoneyTo moe pam (1/6) = giveMoneyTo nick pam (1/4)) :
  let totalInitialMoney := loki.initialMoney + moe.initialMoney + nick.initialMoney
  let moneyReceivedByOttAndPam := 2 * (giveMoneyTo moe ott (1/6) + giveMoneyTo loki ott (1/5) + giveMoneyTo nick ott (1/4))
  moneyReceivedByOttAndPam / totalInitialMoney = 2/5 := by
    sorry

#check friends_money_distribution

end NUMINAMATH_CALUDE_friends_money_distribution_l3679_367954


namespace NUMINAMATH_CALUDE_rachel_apple_picking_l3679_367928

theorem rachel_apple_picking (num_trees : ℕ) (apples_per_tree : ℕ) (remaining_apples : ℕ) : 
  num_trees = 4 → 
  apples_per_tree = 7 → 
  remaining_apples = 29 → 
  num_trees * apples_per_tree = 28 :=
by sorry

end NUMINAMATH_CALUDE_rachel_apple_picking_l3679_367928


namespace NUMINAMATH_CALUDE_claires_remaining_balance_l3679_367905

/-- Calculates the remaining balance on Claire's gift card after a week of purchases --/
def remaining_balance (gift_card latte_price croissant_price bagel_price holiday_drink_price cookie_price : ℚ)
  (days bagel_occasions cookies : ℕ) : ℚ :=
  let daily_total := latte_price + croissant_price
  let weekly_total := daily_total * days
  let bagel_total := bagel_price * bagel_occasions
  let friday_treats := holiday_drink_price + cookie_price * cookies
  let friday_adjustment := friday_treats - latte_price
  let total_expenses := weekly_total + bagel_total + friday_adjustment
  gift_card - total_expenses

/-- Theorem stating that Claire's remaining balance is $35.50 --/
theorem claires_remaining_balance :
  remaining_balance 100 3.75 3.50 2.25 4.50 1.25 7 3 5 = 35.50 := by
  sorry

end NUMINAMATH_CALUDE_claires_remaining_balance_l3679_367905


namespace NUMINAMATH_CALUDE_convex_ngon_regions_l3679_367936

/-- The number of regions into which the diagonals of a convex n-gon divide it -/
def f (n : ℕ) : ℕ := (n - 1) * (n - 2) * (n^2 - 3*n + 12) / 24

/-- A convex n-gon is divided into f(n) regions by its diagonals, 
    given that no three diagonals intersect at a single point -/
theorem convex_ngon_regions (n : ℕ) (h : n ≥ 3) : 
  f n = (n - 1) * (n - 2) * (n^2 - 3*n + 12) / 24 := by
  sorry

end NUMINAMATH_CALUDE_convex_ngon_regions_l3679_367936


namespace NUMINAMATH_CALUDE_min_value_of_f_l3679_367921

open Real

noncomputable def f (x : ℝ) : ℝ := x^2 - log x

theorem min_value_of_f :
  ∃ (x : ℝ), x > 0 ∧ ∀ (y : ℝ), y > 0 → f y ≥ f x ∧ f x = (1 + log 2) / 2 :=
sorry

end NUMINAMATH_CALUDE_min_value_of_f_l3679_367921


namespace NUMINAMATH_CALUDE_right_triangle_ratio_l3679_367951

theorem right_triangle_ratio (a d : ℝ) : 
  (a - d) ^ 2 + a ^ 2 = (a + d) ^ 2 → 
  a = d * (2 + Real.sqrt 3) := by
sorry

end NUMINAMATH_CALUDE_right_triangle_ratio_l3679_367951


namespace NUMINAMATH_CALUDE_walker_speed_l3679_367926

-- Define the track properties
def track_A_width : ℝ := 6
def track_B_width : ℝ := 8
def track_A_time_diff : ℝ := 36
def track_B_time_diff : ℝ := 48

-- Define the theorem
theorem walker_speed (speed : ℝ) : 
  (2 * Real.pi * track_A_width = speed * track_A_time_diff) →
  (2 * Real.pi * track_B_width = speed * track_B_time_diff) →
  speed = Real.pi / 3 := by
  sorry

end NUMINAMATH_CALUDE_walker_speed_l3679_367926


namespace NUMINAMATH_CALUDE_tan_pi_minus_alpha_l3679_367935

open Real

theorem tan_pi_minus_alpha (α : ℝ) :
  tan (π - α) = 3/4 →
  π/2 < α ∧ α < π →
  1 / (sin ((π + α)/2) * sin ((π - α)/2)) = 10 := by
  sorry

end NUMINAMATH_CALUDE_tan_pi_minus_alpha_l3679_367935


namespace NUMINAMATH_CALUDE_arithmetic_sequence_13th_term_l3679_367964

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_13th_term
  (a : ℕ → ℝ)
  (h_arith : arithmetic_sequence a)
  (h_5th : a 5 = 3)
  (h_9th : a 9 = 6) :
  a 13 = 9 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_13th_term_l3679_367964


namespace NUMINAMATH_CALUDE_saddle_value_l3679_367999

theorem saddle_value (total_value : ℝ) (horse_saddle_ratio : ℝ) :
  total_value = 100 →
  horse_saddle_ratio = 7 →
  ∃ (saddle_value : ℝ),
    saddle_value + horse_saddle_ratio * saddle_value = total_value ∧
    saddle_value = 12.5 := by
  sorry

end NUMINAMATH_CALUDE_saddle_value_l3679_367999


namespace NUMINAMATH_CALUDE_bus_journey_l3679_367975

theorem bus_journey (total_distance : ℝ) (speed1 speed2 : ℝ) (total_time : ℝ)
  (h1 : total_distance = 250)
  (h2 : speed1 = 40)
  (h3 : speed2 = 60)
  (h4 : total_time = 6) :
  ∃ (distance1 : ℝ), 
    distance1 / speed1 + (total_distance - distance1) / speed2 = total_time ∧
    distance1 = 220 := by
  sorry

end NUMINAMATH_CALUDE_bus_journey_l3679_367975


namespace NUMINAMATH_CALUDE_second_class_end_time_l3679_367957

/-- Represents time in hours and minutes -/
structure Time where
  hours : Nat
  minutes : Nat
  deriving Repr

/-- Adds minutes to a given time -/
def addMinutes (t : Time) (m : Nat) : Time :=
  let totalMinutes := t.hours * 60 + t.minutes + m
  { hours := totalMinutes / 60
  , minutes := totalMinutes % 60 }

theorem second_class_end_time :
  let start_time : Time := { hours := 9, minutes := 25 }
  let class_duration : Nat := 35
  let end_time := addMinutes start_time class_duration
  end_time = { hours := 10, minutes := 0 } := by
  sorry

end NUMINAMATH_CALUDE_second_class_end_time_l3679_367957


namespace NUMINAMATH_CALUDE_band_to_orchestra_ratio_l3679_367959

/-- The number of male musicians in the orchestra -/
def orchestra_males : ℕ := 11

/-- The number of female musicians in the orchestra -/
def orchestra_females : ℕ := 12

/-- The number of male musicians in the choir -/
def choir_males : ℕ := 12

/-- The number of female musicians in the choir -/
def choir_females : ℕ := 17

/-- The total number of musicians in all groups -/
def total_musicians : ℕ := 98

/-- The number of musicians in the orchestra -/
def orchestra_total : ℕ := orchestra_males + orchestra_females

/-- The number of musicians in the choir -/
def choir_total : ℕ := choir_males + choir_females

theorem band_to_orchestra_ratio :
  ∃ (band_musicians : ℕ),
    band_musicians = 2 * orchestra_total ∧
    orchestra_total + band_musicians + choir_total = total_musicians :=
by sorry

end NUMINAMATH_CALUDE_band_to_orchestra_ratio_l3679_367959


namespace NUMINAMATH_CALUDE_even_function_implies_f_2_equals_3_l3679_367942

def f (a : ℝ) (x : ℝ) : ℝ := (x + 1) * (x - a)

theorem even_function_implies_f_2_equals_3 (a : ℝ) 
  (h : ∀ x, f a x = f a (-x)) : 
  f a 2 = 3 := by
sorry

end NUMINAMATH_CALUDE_even_function_implies_f_2_equals_3_l3679_367942


namespace NUMINAMATH_CALUDE_ceramic_cup_price_l3679_367937

theorem ceramic_cup_price 
  (total_cups : ℕ) 
  (total_revenue : ℚ) 
  (plastic_cup_price : ℚ) 
  (ceramic_cups_sold : ℕ) 
  (plastic_cups_sold : ℕ) :
  total_cups = 400 →
  total_revenue = 1458 →
  plastic_cup_price = (7/2) →
  ceramic_cups_sold = 284 →
  plastic_cups_sold = 116 →
  (total_revenue - (plastic_cup_price * plastic_cups_sold)) / ceramic_cups_sold = (37/10) := by
  sorry

end NUMINAMATH_CALUDE_ceramic_cup_price_l3679_367937


namespace NUMINAMATH_CALUDE_exists_farther_point_l3679_367947

/-- A rectangular parallelepiped (box) -/
structure Box where
  length : ℝ
  width : ℝ
  height : ℝ
  length_pos : 0 < length
  width_pos : 0 < width
  height_pos : 0 < height

/-- A point on the surface of the box -/
inductive SurfacePoint (b : Box)
  | front (x y : ℝ) : 0 ≤ x ∧ x ≤ b.width → 0 ≤ y ∧ y ≤ b.height → SurfacePoint b
  | back (x y : ℝ) : 0 ≤ x ∧ x ≤ b.width → 0 ≤ y ∧ y ≤ b.height → SurfacePoint b
  | left (y z : ℝ) : 0 ≤ y ∧ y ≤ b.height → 0 ≤ z ∧ z ≤ b.length → SurfacePoint b
  | right (y z : ℝ) : 0 ≤ y ∧ y ≤ b.height → 0 ≤ z ∧ z ≤ b.length → SurfacePoint b
  | top (x z : ℝ) : 0 ≤ x ∧ x ≤ b.width → 0 ≤ z ∧ z ≤ b.length → SurfacePoint b
  | bottom (x z : ℝ) : 0 ≤ x ∧ x ≤ b.width → 0 ≤ z ∧ z ≤ b.length → SurfacePoint b

/-- The distance between two points on the surface of the box -/
def surfaceDistance (b : Box) (p q : SurfacePoint b) : ℝ := sorry

/-- The opposite corner of a given corner -/
def oppositeCorner (b : Box) (p : SurfacePoint b) : SurfacePoint b := sorry

/-- Theorem: There exists a point on the surface farther from a corner than the opposite corner -/
theorem exists_farther_point (b : Box) :
  ∃ (corner : SurfacePoint b) (p : SurfacePoint b),
    surfaceDistance b corner p > surfaceDistance b corner (oppositeCorner b corner) := by sorry

end NUMINAMATH_CALUDE_exists_farther_point_l3679_367947


namespace NUMINAMATH_CALUDE_complex_number_quadrant_l3679_367970

theorem complex_number_quadrant (z : ℂ) (h : (1 + 2*Complex.I)*z = 3 + Complex.I*z) :
  z.re > 0 ∧ z.im < 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_quadrant_l3679_367970


namespace NUMINAMATH_CALUDE_probability_same_color_is_31_364_l3679_367997

def red_plates : ℕ := 6
def blue_plates : ℕ := 5
def green_plates : ℕ := 3
def total_plates : ℕ := red_plates + blue_plates + green_plates

def probability_same_color : ℚ :=
  (Nat.choose red_plates 3 + Nat.choose blue_plates 3 + Nat.choose green_plates 3) /
  Nat.choose total_plates 3

theorem probability_same_color_is_31_364 :
  probability_same_color = 31 / 364 := by
  sorry

end NUMINAMATH_CALUDE_probability_same_color_is_31_364_l3679_367997


namespace NUMINAMATH_CALUDE_jinho_ribbon_length_l3679_367900

/-- The number of students in Minsu's class -/
def minsu_students : ℕ := 8

/-- The number of students in Jinho's class -/
def jinho_students : ℕ := minsu_students + 1

/-- The total length of ribbon in meters -/
def total_ribbon_m : ℝ := 3.944

/-- The length of ribbon given to each student in Minsu's class in centimeters -/
def ribbon_per_minsu_student_cm : ℝ := 29.05

/-- Conversion factor from meters to centimeters -/
def m_to_cm : ℝ := 100

theorem jinho_ribbon_length :
  let total_ribbon_cm := total_ribbon_m * m_to_cm
  let minsu_total_ribbon_cm := ribbon_per_minsu_student_cm * minsu_students
  let remaining_ribbon_cm := total_ribbon_cm - minsu_total_ribbon_cm
  remaining_ribbon_cm / jinho_students = 18 := by sorry

end NUMINAMATH_CALUDE_jinho_ribbon_length_l3679_367900


namespace NUMINAMATH_CALUDE_coins_problem_l3679_367949

theorem coins_problem (a b c d : ℕ) : 
  a = 21 →                  -- A has 21 coins
  a = b + 9 →               -- A has 9 more coins than B
  c = b + 17 →              -- C has 17 more coins than B
  a + b = c + d - 5 →       -- Sum of A and B is 5 less than sum of C and D
  d = 9 :=                  -- D has 9 coins
by sorry

end NUMINAMATH_CALUDE_coins_problem_l3679_367949


namespace NUMINAMATH_CALUDE_max_volume_inscribed_cone_l3679_367919

/-- Given a sphere with volume 36π, the maximum volume of an inscribed cone is 32π/3 -/
theorem max_volume_inscribed_cone (sphere_volume : ℝ) (h_volume : sphere_volume = 36 * Real.pi) :
  ∃ (max_cone_volume : ℝ),
    (∀ (cone_volume : ℝ), cone_volume ≤ max_cone_volume) ∧
    (max_cone_volume = (32 * Real.pi) / 3) :=
sorry

end NUMINAMATH_CALUDE_max_volume_inscribed_cone_l3679_367919


namespace NUMINAMATH_CALUDE_shirt_cost_problem_l3679_367992

theorem shirt_cost_problem (x : ℝ) : 
  (3 * x + 2 * 20 = 85) → x = 15 := by
  sorry

end NUMINAMATH_CALUDE_shirt_cost_problem_l3679_367992


namespace NUMINAMATH_CALUDE_unique_solution_quadratic_inequality_l3679_367987

theorem unique_solution_quadratic_inequality (b : ℝ) : 
  (∃! x : ℝ, |x^2 - 6*b*x + 5*b| ≤ 3) ↔ 
  (b = (5 + Real.sqrt 73) / 8 ∨ b = (5 - Real.sqrt 73) / 8) :=
sorry

end NUMINAMATH_CALUDE_unique_solution_quadratic_inequality_l3679_367987


namespace NUMINAMATH_CALUDE_equation_solution_l3679_367944

theorem equation_solution : ∃ x : ℝ, 
  6 * ((1/2) * x - 4) + 2 * x = 7 - ((1/3) * x - 1) ∧ x = 6 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3679_367944


namespace NUMINAMATH_CALUDE_problem_statement_l3679_367927

theorem problem_statement (a b : ℝ) : 
  let M := {b/a, 1}
  let N := {a, 0}
  (∃ f : ℝ → ℝ, f = id ∧ f '' M ⊆ N) →
  a + b = 1 := by
sorry

end NUMINAMATH_CALUDE_problem_statement_l3679_367927


namespace NUMINAMATH_CALUDE_retirement_fund_decrease_l3679_367930

/-- Proves that the decrease in Kate's retirement fund is $12 --/
theorem retirement_fund_decrease (previous_value current_value : ℕ) 
  (h1 : previous_value = 1472)
  (h2 : current_value = 1460) : 
  previous_value - current_value = 12 := by
  sorry

end NUMINAMATH_CALUDE_retirement_fund_decrease_l3679_367930


namespace NUMINAMATH_CALUDE_school_teacher_count_l3679_367901

/-- Represents the number of students and teachers in a grade --/
structure GradeData where
  students : ℕ
  teachers : ℕ

/-- Proves that given the conditions, the number of teachers in grade A is 8 and in grade B is 26 --/
theorem school_teacher_count 
  (gradeA gradeB : GradeData)
  (ratioA : gradeA.students = 30 * gradeA.teachers)
  (ratioB : gradeB.students = 40 * gradeB.teachers)
  (newRatioA : gradeA.students + 60 = 25 * (gradeA.teachers + 4))
  (newRatioB : gradeB.students + 80 = 35 * (gradeB.teachers + 6))
  : gradeA.teachers = 8 ∧ gradeB.teachers = 26 := by
  sorry

#check school_teacher_count

end NUMINAMATH_CALUDE_school_teacher_count_l3679_367901


namespace NUMINAMATH_CALUDE_num_connected_subsets_2x1_l3679_367908

/-- A rectangle in the Cartesian plane -/
structure Rectangle :=
  (bottomLeft : ℝ × ℝ)
  (topRight : ℝ × ℝ)

/-- An edge of a rectangle -/
inductive Edge
  | BottomLeft
  | BottomRight
  | TopLeft
  | TopRight
  | Left
  | Middle
  | Right

/-- A subset of edges -/
def EdgeSubset := Set Edge

/-- Predicate to determine if a subset of edges is connected -/
def is_connected (s : EdgeSubset) : Prop := sorry

/-- The number of connected subsets of edges in a 2x1 rectangle divided into two unit squares -/
def num_connected_subsets (r : Rectangle) : ℕ := sorry

/-- Theorem stating that the number of connected subsets is 81 -/
theorem num_connected_subsets_2x1 :
  ∀ r : Rectangle,
  r.bottomLeft = (0, 0) ∧ r.topRight = (2, 1) →
  num_connected_subsets r = 81 :=
sorry

end NUMINAMATH_CALUDE_num_connected_subsets_2x1_l3679_367908


namespace NUMINAMATH_CALUDE_units_digit_factorial_25_l3679_367983

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

theorem units_digit_factorial_25 : factorial 25 % 10 = 0 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_factorial_25_l3679_367983


namespace NUMINAMATH_CALUDE_circle_center_radius_sum_l3679_367985

theorem circle_center_radius_sum :
  ∀ (a b r : ℝ),
  (∀ (x y : ℝ), x^2 - 16*x + y^2 + 6*y = 20 ↔ (x - a)^2 + (y - b)^2 = r^2) →
  a + b + r = 5 + Real.sqrt 93 :=
by sorry

end NUMINAMATH_CALUDE_circle_center_radius_sum_l3679_367985


namespace NUMINAMATH_CALUDE_value_of_a_l3679_367943

def A (a : ℝ) : Set ℝ := {a + 2, 2 * a^2 + a}

theorem value_of_a : ∀ a : ℝ, 3 ∈ A a → a = -3/2 := by
  sorry

end NUMINAMATH_CALUDE_value_of_a_l3679_367943


namespace NUMINAMATH_CALUDE_monotonic_unique_zero_l3679_367979

/-- A function f is monotonic on (a, b) -/
def Monotonic (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a < x ∧ x < y ∧ y < b → (f x < f y ∨ f y < f x)

/-- f has exactly one zero in [a, b] -/
def HasUniqueZero (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∃! x, a ≤ x ∧ x ≤ b ∧ f x = 0

theorem monotonic_unique_zero (f : ℝ → ℝ) (a b : ℝ) 
  (h1 : Monotonic f a b) (h2 : f a * f b < 0) :
  HasUniqueZero f a b :=
sorry

end NUMINAMATH_CALUDE_monotonic_unique_zero_l3679_367979


namespace NUMINAMATH_CALUDE_min_value_expression_l3679_367915

theorem min_value_expression (x : ℝ) (hx : x > 0) : 3 * x + 2 / x^5 + 3 / x ≥ 8 ∧
  (3 * x + 2 / x^5 + 3 / x = 8 ↔ x = 1) := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l3679_367915


namespace NUMINAMATH_CALUDE_product_325_4_base_7_l3679_367912

/-- Converts a number from base 7 to base 10 -/
def to_base_10 (digits : List Nat) : Nat :=
  digits.enum.foldl (fun acc (i, d) => acc + d * (7 ^ i)) 0

/-- Converts a number from base 10 to base 7 -/
def to_base_7 (n : Nat) : List Nat :=
  if n = 0 then [0] else
    let rec aux (m : Nat) (acc : List Nat) : List Nat :=
      if m = 0 then acc
      else aux (m / 7) ((m % 7) :: acc)
    aux n []

/-- Multiplies two numbers in base 7 -/
def mult_base_7 (a b : List Nat) : List Nat :=
  to_base_7 (to_base_10 a * to_base_10 b)

theorem product_325_4_base_7 :
  mult_base_7 [5, 2, 3] [4] = [6, 3, 6, 1] := by sorry

end NUMINAMATH_CALUDE_product_325_4_base_7_l3679_367912


namespace NUMINAMATH_CALUDE_investment_problem_l3679_367932

/-- The investment problem with three partners A, B, and C. -/
theorem investment_problem (investment_B investment_C : ℕ) 
  (profit_B : ℕ) (profit_diff_A_C : ℕ) (investment_A : ℕ) : 
  investment_B = 8000 →
  investment_C = 10000 →
  profit_B = 1000 →
  profit_diff_A_C = 500 →
  (investment_A : ℚ) / investment_B = ((profit_B : ℚ) + profit_diff_A_C) / profit_B →
  (investment_A : ℚ) / investment_C = ((profit_B : ℚ) + profit_diff_A_C) / profit_B →
  investment_A = 12000 := by
  sorry

end NUMINAMATH_CALUDE_investment_problem_l3679_367932


namespace NUMINAMATH_CALUDE_train_passenger_count_l3679_367988

/-- Calculates the total number of passengers transported by a train between two stations -/
def total_passengers (num_round_trips : ℕ) (passengers_first_trip : ℕ) (passengers_return_trip : ℕ) : ℕ :=
  num_round_trips * (passengers_first_trip + passengers_return_trip)

/-- Proves that the total number of passengers transported is 640 given the specified conditions -/
theorem train_passenger_count :
  let num_round_trips : ℕ := 4
  let passengers_first_trip : ℕ := 100
  let passengers_return_trip : ℕ := 60
  total_passengers num_round_trips passengers_first_trip passengers_return_trip = 640 :=
by
  sorry


end NUMINAMATH_CALUDE_train_passenger_count_l3679_367988


namespace NUMINAMATH_CALUDE_sequence_a_bounds_l3679_367990

def sequence_a : ℕ → ℚ
  | 0     => 1/2
  | (n+1) => sequence_a n + (1 / (n+1)^2) * (sequence_a n)^2

theorem sequence_a_bounds (n : ℕ) : 
  1 - 1 / (2^(n+1)) ≤ sequence_a n ∧ sequence_a n < 7/5 := by
  sorry

end NUMINAMATH_CALUDE_sequence_a_bounds_l3679_367990


namespace NUMINAMATH_CALUDE_inequality_proof_l3679_367996

theorem inequality_proof (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  a / Real.sqrt b + b / Real.sqrt a ≥ Real.sqrt a + Real.sqrt b := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3679_367996


namespace NUMINAMATH_CALUDE_hyperbola_circle_eccentricity_l3679_367917

/-- Given a hyperbola with equation x^2 - ny^2 = 1 and eccentricity 2, 
    prove that the eccentricity of the circle x^2 + ny^2 = 1 is √6/3 -/
theorem hyperbola_circle_eccentricity (m n : ℝ) (hm : m > 0) (hn : n > 0) 
  (h_hyperbola_ecc : (m⁻¹ + n⁻¹) / m⁻¹ = 4) :
  Real.sqrt ((n⁻¹ - m⁻¹) / n⁻¹) = Real.sqrt 6 / 3 := by
  sorry

end NUMINAMATH_CALUDE_hyperbola_circle_eccentricity_l3679_367917


namespace NUMINAMATH_CALUDE_min_omega_value_l3679_367960

theorem min_omega_value (ω : ℝ) (n : ℤ) : 
  ω > 0 ∧ (4 * π / 3 = n * (2 * π / ω)) → ω ≥ 3 / 2 :=
by sorry

end NUMINAMATH_CALUDE_min_omega_value_l3679_367960


namespace NUMINAMATH_CALUDE_mass_B13N3O12H12_value_l3679_367978

/-- The mass in grams of 12 moles of Trinitride dodecahydroxy tridecaborate (B13N3O12H12) -/
def mass_B13N3O12H12 : ℝ :=
  let atomic_mass_B : ℝ := 10.81
  let atomic_mass_N : ℝ := 14.01
  let atomic_mass_O : ℝ := 16.00
  let atomic_mass_H : ℝ := 1.01
  let molar_mass : ℝ := 13 * atomic_mass_B + 3 * atomic_mass_N + 12 * atomic_mass_O + 12 * atomic_mass_H
  12 * molar_mass

/-- Theorem stating that the mass of 12 moles of B13N3O12H12 is 4640.16 grams -/
theorem mass_B13N3O12H12_value : mass_B13N3O12H12 = 4640.16 := by
  sorry

end NUMINAMATH_CALUDE_mass_B13N3O12H12_value_l3679_367978


namespace NUMINAMATH_CALUDE_correct_oranges_to_remove_l3679_367980

/-- Represents the fruit selection problem -/
structure FruitSelection where
  applePrice : ℚ  -- Price of each apple in cents
  orangePrice : ℚ  -- Price of each orange in cents
  totalFruits : ℕ  -- Total number of fruits initially selected
  initialAvgPrice : ℚ  -- Initial average price of all fruits
  desiredAvgPrice : ℚ  -- Desired average price after removing oranges

/-- Calculates the number of oranges to remove -/
def orangesToRemove (fs : FruitSelection) : ℕ :=
  sorry

/-- Theorem stating the correct number of oranges to remove -/
theorem correct_oranges_to_remove (fs : FruitSelection) 
  (h1 : fs.applePrice = 40/100)
  (h2 : fs.orangePrice = 60/100)
  (h3 : fs.totalFruits = 20)
  (h4 : fs.initialAvgPrice = 56/100)
  (h5 : fs.desiredAvgPrice = 52/100) :
  orangesToRemove fs = 10 := by sorry

end NUMINAMATH_CALUDE_correct_oranges_to_remove_l3679_367980


namespace NUMINAMATH_CALUDE_marys_bag_check_time_l3679_367924

/-- Represents the time in minutes for Mary's trip to the airport -/
structure AirportTrip where
  uberToHouse : ℕ
  uberToAirport : ℕ
  bagCheck : ℕ
  security : ℕ
  waitForBoarding : ℕ
  waitForTakeoff : ℕ

/-- The total trip time in minutes -/
def totalTripTime (trip : AirportTrip) : ℕ :=
  trip.uberToHouse + trip.uberToAirport + trip.bagCheck + trip.security + trip.waitForBoarding + trip.waitForTakeoff

/-- Mary's airport trip satisfies the given conditions -/
def marysTrip (trip : AirportTrip) : Prop :=
  trip.uberToHouse = 10 ∧
  trip.uberToAirport = 5 * trip.uberToHouse ∧
  trip.security = 3 * trip.bagCheck ∧
  trip.waitForBoarding = 20 ∧
  trip.waitForTakeoff = 2 * trip.waitForBoarding ∧
  totalTripTime trip = 180  -- 3 hours in minutes

theorem marys_bag_check_time (trip : AirportTrip) (h : marysTrip trip) : trip.bagCheck = 15 := by
  sorry

end NUMINAMATH_CALUDE_marys_bag_check_time_l3679_367924


namespace NUMINAMATH_CALUDE_circle_radius_from_chords_and_midpoint_distance_l3679_367945

theorem circle_radius_from_chords_and_midpoint_distance 
  (chord1 : ℝ) (chord2 : ℝ) (midpoint_distance : ℝ) (radius : ℝ) : 
  chord1 = 10 → 
  chord2 = 12 → 
  midpoint_distance = 4 → 
  (8 * (2 * radius - 8) = 6 * 6) → 
  radius = 6.25 := by sorry

end NUMINAMATH_CALUDE_circle_radius_from_chords_and_midpoint_distance_l3679_367945


namespace NUMINAMATH_CALUDE_condition_sufficient_not_necessary_l3679_367986

theorem condition_sufficient_not_necessary (a b : ℝ) :
  (∀ a b : ℝ, (abs a < 1 ∧ abs b < 1) → abs (1 - a * b) > abs (a - b)) ∧
  (∃ a b : ℝ, abs (1 - a * b) > abs (a - b) ∧ ¬(abs a < 1 ∧ abs b < 1)) :=
by sorry

end NUMINAMATH_CALUDE_condition_sufficient_not_necessary_l3679_367986


namespace NUMINAMATH_CALUDE_newlyGrownUneatenCorrect_l3679_367910

/-- Represents the number of potatoes in Mary's garden -/
structure PotatoGarden where
  initial : ℕ
  current : ℕ

/-- Calculates the number of newly grown potatoes left uneaten -/
def newlyGrownUneaten (garden : PotatoGarden) : ℕ :=
  garden.current - garden.initial

theorem newlyGrownUneatenCorrect (garden : PotatoGarden) 
  (h1 : garden.initial = 8) 
  (h2 : garden.current = 11) : 
  newlyGrownUneaten garden = 3 := by
  sorry

end NUMINAMATH_CALUDE_newlyGrownUneatenCorrect_l3679_367910


namespace NUMINAMATH_CALUDE_valid_fractions_l3679_367998

def is_valid_fraction (num den : ℕ) : Prop :=
  10 ≤ num ∧ num < 100 ∧ 10 ≤ den ∧ den < 100 ∧
  (num / 10 : ℕ) = den % 10 ∧
  (num % 10 : ℚ) / (den / 10 : ℚ) = (num : ℚ) / (den : ℚ)

theorem valid_fractions :
  {f : ℚ | ∃ (num den : ℕ), is_valid_fraction num den ∧ f = (num : ℚ) / (den : ℚ)} =
  {64/16, 98/49, 95/19, 65/26} :=
by sorry

end NUMINAMATH_CALUDE_valid_fractions_l3679_367998


namespace NUMINAMATH_CALUDE_range_of_a_l3679_367913

theorem range_of_a (a : ℝ) :
  (∀ x : ℝ, x^2 - 2*x > a^2 - a - 3) → a ∈ Set.Ioo (-1 : ℝ) 2 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l3679_367913


namespace NUMINAMATH_CALUDE_arithmetic_sequence_properties_l3679_367976

/-- An arithmetic sequence with specific properties -/
structure ArithmeticSequence where
  a : ℕ → ℝ  -- The sequence
  s : ℕ → ℝ  -- The sum of the first n terms
  second_term : a 2 = 4
  sum_formula : ∀ n : ℕ, s n = n^2 + c * n
  c : ℝ       -- The constant in the sum formula

/-- Theorem stating the properties of the arithmetic sequence -/
theorem arithmetic_sequence_properties (seq : ArithmeticSequence) :
  seq.c = 1 ∧ ∀ n : ℕ, seq.a n = 2 * n := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_properties_l3679_367976


namespace NUMINAMATH_CALUDE_remainder_of_y_l3679_367981

theorem remainder_of_y (y : ℤ) 
  (h1 : (4 + y) % 8 = 3^2 % 8)
  (h2 : (6 + y) % 27 = 2^3 % 27)
  (h3 : (8 + y) % 125 = 3^3 % 125) :
  y % 30 = 4 := by
sorry

end NUMINAMATH_CALUDE_remainder_of_y_l3679_367981


namespace NUMINAMATH_CALUDE_bisection_interval_valid_l3679_367963

-- Define the function f(x) = x^3 + 5
def f (x : ℝ) : ℝ := x^3 + 5

-- Theorem statement
theorem bisection_interval_valid :
  f (-2) * f 1 < 0 := by sorry

end NUMINAMATH_CALUDE_bisection_interval_valid_l3679_367963


namespace NUMINAMATH_CALUDE_abc_inequality_l3679_367929

theorem abc_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  Real.sqrt (a * b * c) * (Real.sqrt a + Real.sqrt b + Real.sqrt c) + (a + b + c)^2 ≥ 
  4 * Real.sqrt (3 * a * b * c * (a + b + c)) :=
by sorry

end NUMINAMATH_CALUDE_abc_inequality_l3679_367929


namespace NUMINAMATH_CALUDE_valid_solution_l3679_367904

def set_difference (A B : Set ℕ) : Set ℕ := {x | x ∈ A ∧ x ∉ B}

theorem valid_solution :
  let A : Set ℕ := {1, 2, 3, 4, 5}
  let B : Set ℕ := {4, 5}
  set_difference A B = {1, 2, 3} := by sorry

end NUMINAMATH_CALUDE_valid_solution_l3679_367904


namespace NUMINAMATH_CALUDE_total_workers_l3679_367911

theorem total_workers (monkeys termites : ℕ) 
  (h1 : monkeys = 239) 
  (h2 : termites = 622) : 
  monkeys + termites = 861 := by
  sorry

end NUMINAMATH_CALUDE_total_workers_l3679_367911


namespace NUMINAMATH_CALUDE_problem_solution_l3679_367933

-- Define proposition p
def p (k : ℝ) : Prop := k^2 - 8*k - 20 ≤ 0

-- Define proposition q
def q (k : ℝ) : Prop := ∃ (a b : ℝ), a > 0 ∧ b < 0 ∧ a = 4 - k ∧ b = 1 - k

-- Define the range of k
def k_range (k : ℝ) : Prop := (-2 ≤ k ∧ k ≤ 1) ∨ (4 ≤ k ∧ k ≤ 10)

-- Theorem statement
theorem problem_solution (k : ℝ) : (p k ∨ q k) ∧ ¬(p k ∧ q k) → k_range k := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l3679_367933


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l3679_367966

theorem sqrt_equation_solution (x : ℝ) : 
  Real.sqrt (5 * x + 9) = 11 → x = 22.4 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l3679_367966


namespace NUMINAMATH_CALUDE_f_satisfies_all_points_l3679_367934

/-- The relation between x and y --/
def f (x : ℝ) : ℝ := -50 * x + 200

/-- The set of points from the given table --/
def points : List (ℝ × ℝ) := [(0, 200), (1, 150), (2, 100), (3, 50), (4, 0)]

/-- Theorem stating that the function f satisfies all points in the given table --/
theorem f_satisfies_all_points : ∀ (p : ℝ × ℝ), p ∈ points → f p.1 = p.2 := by
  sorry

end NUMINAMATH_CALUDE_f_satisfies_all_points_l3679_367934


namespace NUMINAMATH_CALUDE_weight_of_nine_moles_972_l3679_367909

/-- The weight of a compound given its number of moles and molecular weight -/
def weight_of_compound (moles : ℝ) (molecular_weight : ℝ) : ℝ :=
  moles * molecular_weight

/-- Theorem: The weight of 9 moles of a compound with molecular weight 972 g/mol is 8748 grams -/
theorem weight_of_nine_moles_972 : 
  weight_of_compound 9 972 = 8748 := by
  sorry

end NUMINAMATH_CALUDE_weight_of_nine_moles_972_l3679_367909


namespace NUMINAMATH_CALUDE_cookie_sales_l3679_367903

theorem cookie_sales (n : ℕ) (a : ℕ) (h1 : n = 10) (h2 : 1 ≤ a) (h3 : a < n) (h4 : 1 + a < n) : a ≤ 8 := by
  sorry

end NUMINAMATH_CALUDE_cookie_sales_l3679_367903


namespace NUMINAMATH_CALUDE_smith_payment_l3679_367972

-- Define the original balance
def original_balance : ℝ := 150

-- Define the finance charge rate
def finance_charge_rate : ℝ := 0.02

-- Define the finance charge calculation
def finance_charge : ℝ := original_balance * finance_charge_rate

-- Define the total payment calculation
def total_payment : ℝ := original_balance + finance_charge

-- Theorem to prove
theorem smith_payment : total_payment = 153 := by
  sorry

end NUMINAMATH_CALUDE_smith_payment_l3679_367972


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l3679_367993

theorem quadratic_equation_solution :
  ∃ (x₁ x₂ : ℝ), (∀ x : ℝ, x * (x + 1) = 3 * (x + 1) ↔ x = x₁ ∨ x = x₂) ∧ x₁ = -1 ∧ x₂ = 3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l3679_367993


namespace NUMINAMATH_CALUDE_max_x0_value_l3679_367991

theorem max_x0_value (x : Fin 1996 → ℝ) 
  (h1 : x 0 = x 1995)
  (h2 : ∀ i : Fin 1995, x i.val + 2 / x i.val = 2 * x (i.val + 1) + 1 / x (i.val + 1))
  (h3 : ∀ i : Fin 1996, x i > 0) :
  x 0 ≤ 2^997 ∧ ∃ y : Fin 1996 → ℝ, 
    y 0 = 2^997 ∧ 
    y 1995 = y 0 ∧ 
    (∀ i : Fin 1995, y i + 2 / y i = 2 * y (i.val + 1) + 1 / y (i.val + 1)) ∧
    (∀ i : Fin 1996, y i > 0) :=
by sorry

end NUMINAMATH_CALUDE_max_x0_value_l3679_367991


namespace NUMINAMATH_CALUDE_y_squared_value_l3679_367916

theorem y_squared_value (x y : ℤ) 
  (eq1 : 4 * x + y = 34) 
  (eq2 : 2 * x - y = 20) : 
  y ^ 2 = 4 := by
  sorry

end NUMINAMATH_CALUDE_y_squared_value_l3679_367916
