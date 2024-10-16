import Mathlib

namespace NUMINAMATH_CALUDE_alpha_squared_greater_than_beta_squared_l2474_247448

theorem alpha_squared_greater_than_beta_squared
  (α β : ℝ)
  (h1 : α ∈ Set.Icc (-π/2) (π/2))
  (h2 : β ∈ Set.Icc (-π/2) (π/2))
  (h3 : α * Real.sin α - β * Real.sin β > 0) :
  α^2 > β^2 := by
  sorry

end NUMINAMATH_CALUDE_alpha_squared_greater_than_beta_squared_l2474_247448


namespace NUMINAMATH_CALUDE_problem_statement_l2474_247416

theorem problem_statement (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 1) :
  (∃ m : ℝ, (∀ a b : ℝ, a > 0 → b > 0 → a + b = 1 → a * b ≤ m) ∧ m = 1/4) ∧
  (∀ x : ℝ, 4/a + 1/b ≥ |2*x - 1| - |x + 2| ↔ -6 ≤ x ∧ x ≤ 12) :=
by sorry

end NUMINAMATH_CALUDE_problem_statement_l2474_247416


namespace NUMINAMATH_CALUDE_art_club_participation_l2474_247427

theorem art_club_participation (total : ℕ) (painting : ℕ) (sculpting : ℕ) (both : ℕ)
  (h1 : total = 150)
  (h2 : painting = 80)
  (h3 : sculpting = 60)
  (h4 : both = 20) :
  total - (painting + sculpting - both) = 30 := by
  sorry

end NUMINAMATH_CALUDE_art_club_participation_l2474_247427


namespace NUMINAMATH_CALUDE_complex_sum_problem_l2474_247405

theorem complex_sum_problem (a b c d e f : ℝ) :
  d = 2 →
  e = -a - c →
  (Complex.mk a b) + (Complex.mk c d) + (Complex.mk e f) = Complex.I * 2 →
  b + f = 0 := by
  sorry

end NUMINAMATH_CALUDE_complex_sum_problem_l2474_247405


namespace NUMINAMATH_CALUDE_quadratic_form_ratio_l2474_247491

theorem quadratic_form_ratio (x : ℝ) : ∃ b c : ℝ, 
  x^2 + 500*x + 1000 = (x + b)^2 + c ∧ c / b = -246 := by
sorry

end NUMINAMATH_CALUDE_quadratic_form_ratio_l2474_247491


namespace NUMINAMATH_CALUDE_trees_on_road_l2474_247449

/-- Calculates the number of trees that can be planted along a road -/
def numTrees (roadLength : ℕ) (interval : ℕ) : ℕ :=
  roadLength / interval + 1

/-- Theorem stating the number of trees that can be planted on a 100-meter road with 5-meter intervals -/
theorem trees_on_road :
  numTrees 100 5 = 21 := by
  sorry

end NUMINAMATH_CALUDE_trees_on_road_l2474_247449


namespace NUMINAMATH_CALUDE_fish_cost_l2474_247438

/-- Given that 530 pesos can buy 4 kg of fish and 2 kg of pork,
    and 875 pesos can buy 7 kg of fish and 3 kg of pork,
    prove that the cost of 1 kg of fish is 80 pesos. -/
theorem fish_cost (fish_price pork_price : ℝ) 
  (h1 : 4 * fish_price + 2 * pork_price = 530)
  (h2 : 7 * fish_price + 3 * pork_price = 875) : 
  fish_price = 80 := by
  sorry

end NUMINAMATH_CALUDE_fish_cost_l2474_247438


namespace NUMINAMATH_CALUDE_books_second_shop_l2474_247408

def books_first_shop : ℕ := 65
def cost_first_shop : ℕ := 1080
def cost_second_shop : ℕ := 840
def average_price : ℕ := 16

theorem books_second_shop :
  (cost_first_shop + cost_second_shop) / average_price - books_first_shop = 55 := by
  sorry

end NUMINAMATH_CALUDE_books_second_shop_l2474_247408


namespace NUMINAMATH_CALUDE_two_propositions_are_true_l2474_247442

-- Define the types for lines and planes
def Line : Type := Unit
def Plane : Type := Unit

-- Define the relations
def parallel : Line → Line → Prop := sorry
def perpendicular : Line → Line → Prop := sorry
def planeParallel : Plane → Plane → Prop := sorry
def planePerpendicular : Plane → Plane → Prop := sorry
def lineParallelToPlane : Line → Plane → Prop := sorry
def linePerpendicularToPlane : Line → Plane → Prop := sorry

-- Define the propositions
def prop1 (α β : Plane) (c : Line) : Prop :=
  planeParallel α β ∧ linePerpendicularToPlane c α → linePerpendicularToPlane c β

def prop2 (α : Plane) (b : Line) (γ : Plane) : Prop :=
  lineParallelToPlane b α ∧ planePerpendicular α γ → linePerpendicularToPlane b γ

def prop3 (a : Line) (β γ : Plane) : Prop :=
  lineParallelToPlane a β ∧ linePerpendicularToPlane a γ → planePerpendicular β γ

-- The main theorem
theorem two_propositions_are_true :
  ∃ (α β γ : Plane) (a b c : Line),
    (prop1 α β c ∧ prop3 a β γ) ∧ ¬prop2 α b γ :=
sorry

end NUMINAMATH_CALUDE_two_propositions_are_true_l2474_247442


namespace NUMINAMATH_CALUDE_burger_cost_l2474_247430

theorem burger_cost : ∃ (burger_cost : ℝ),
  burger_cost = 9 ∧
  ∃ (pizza_cost : ℝ),
  pizza_cost = 2 * burger_cost ∧
  pizza_cost + 3 * burger_cost = 45 := by
sorry

end NUMINAMATH_CALUDE_burger_cost_l2474_247430


namespace NUMINAMATH_CALUDE_fib_identity_fib_1094_1096_minus_1095_squared_l2474_247437

/-- The Fibonacci sequence -/
def fib : ℕ → ℤ
  | 0 => 0
  | 1 => 1
  | n + 2 => fib (n + 1) + fib n

/-- The identity for Fibonacci numbers -/
theorem fib_identity (n : ℕ) :
  fib (n + 2) * fib n - fib (n + 1)^2 = (-1)^(n + 1) := by sorry

/-- The main theorem to prove -/
theorem fib_1094_1096_minus_1095_squared :
  fib 1094 * fib 1096 - fib 1095^2 = -1 := by sorry

end NUMINAMATH_CALUDE_fib_identity_fib_1094_1096_minus_1095_squared_l2474_247437


namespace NUMINAMATH_CALUDE_line_parameterization_l2474_247497

/-- The line y = 5x - 7 is parameterized by (x, y) = (r, 2) + t(3, k). 
    This theorem proves that r = 9/5 and k = 15. -/
theorem line_parameterization (x y r k t : ℝ) : 
  y = 5 * x - 7 ∧ 
  x = r + 3 * t ∧ 
  y = 2 + k * t → 
  r = 9 / 5 ∧ k = 15 := by
sorry

end NUMINAMATH_CALUDE_line_parameterization_l2474_247497


namespace NUMINAMATH_CALUDE_tom_buys_four_papayas_l2474_247402

/-- Represents the fruit purchase scenario --/
structure FruitPurchase where
  lemon_price : ℕ
  papaya_price : ℕ
  mango_price : ℕ
  discount_threshold : ℕ
  discount_amount : ℕ
  lemons_bought : ℕ
  mangos_bought : ℕ
  total_paid : ℕ

/-- Calculates the number of papayas bought --/
def papayas_bought (fp : FruitPurchase) (p : ℕ) : Prop :=
  let total_fruits := fp.lemons_bought + fp.mangos_bought + p
  let total_cost := fp.lemon_price * fp.lemons_bought + 
                    fp.papaya_price * p + 
                    fp.mango_price * fp.mangos_bought
  let discount := (total_fruits / fp.discount_threshold) * fp.discount_amount
  total_cost - discount = fp.total_paid

/-- Theorem stating that Tom buys 4 papayas --/
theorem tom_buys_four_papayas : 
  ∃ (fp : FruitPurchase), 
    fp.lemon_price = 2 ∧ 
    fp.papaya_price = 1 ∧ 
    fp.mango_price = 4 ∧ 
    fp.discount_threshold = 4 ∧ 
    fp.discount_amount = 1 ∧ 
    fp.lemons_bought = 6 ∧ 
    fp.mangos_bought = 2 ∧ 
    fp.total_paid = 21 ∧ 
    papayas_bought fp 4 :=
sorry

end NUMINAMATH_CALUDE_tom_buys_four_papayas_l2474_247402


namespace NUMINAMATH_CALUDE_fidos_yard_l2474_247489

theorem fidos_yard (r : ℝ) (h : r > 0) : 
  let circle_area := π * r^2
  let hexagon_area := 3 * r^2 * Real.sqrt 3 / 2
  let ratio := circle_area / hexagon_area
  ratio = Real.sqrt 3 * π / 6 ∧ 3 * 6 = 18 := by sorry

end NUMINAMATH_CALUDE_fidos_yard_l2474_247489


namespace NUMINAMATH_CALUDE_colin_average_mile_time_l2474_247478

def average_mile_time (first_mile : ℕ) (second_mile : ℕ) (third_mile : ℕ) (fourth_mile : ℕ) : ℚ :=
  (first_mile + second_mile + third_mile + fourth_mile) / 4

theorem colin_average_mile_time :
  average_mile_time 6 5 5 4 = 5 := by
  sorry

end NUMINAMATH_CALUDE_colin_average_mile_time_l2474_247478


namespace NUMINAMATH_CALUDE_additional_distance_for_average_speed_l2474_247403

theorem additional_distance_for_average_speed
  (initial_distance : ℝ)
  (initial_speed : ℝ)
  (second_speed : ℝ)
  (target_average_speed : ℝ)
  (h : initial_distance = 20)
  (h1 : initial_speed = 40)
  (h2 : second_speed = 60)
  (h3 : target_average_speed = 55)
  : ∃ (additional_distance : ℝ),
    (initial_distance + additional_distance) / ((initial_distance / initial_speed) + (additional_distance / second_speed)) = target_average_speed ∧
    additional_distance = 90 := by
  sorry

end NUMINAMATH_CALUDE_additional_distance_for_average_speed_l2474_247403


namespace NUMINAMATH_CALUDE_school_council_composition_l2474_247401

theorem school_council_composition :
  -- Total number of classes
  ∀ (total_classes : ℕ),
  -- Number of students per council
  ∀ (students_per_council : ℕ),
  -- Number of classes with more girls than boys
  ∀ (classes_more_girls : ℕ),
  -- Number of boys and girls in Petya's class
  ∀ (petyas_class_boys petyas_class_girls : ℕ),
  -- Total number of boys and girls across all councils
  ∀ (total_boys total_girls : ℕ),

  total_classes = 20 →
  students_per_council = 5 →
  classes_more_girls = 15 →
  petyas_class_boys = 1 →
  petyas_class_girls = 4 →
  total_boys = total_girls →
  total_boys + total_girls = total_classes * students_per_council →

  -- Conclusion: In the remaining 4 classes, there are 19 boys and 1 girl
  ∃ (remaining_boys remaining_girls : ℕ),
    remaining_boys = 19 ∧
    remaining_girls = 1 ∧
    remaining_boys + remaining_girls = (total_classes - classes_more_girls - 1) * students_per_council :=
by sorry

end NUMINAMATH_CALUDE_school_council_composition_l2474_247401


namespace NUMINAMATH_CALUDE_floor_equation_solutions_l2474_247441

theorem floor_equation_solutions (n : ℤ) : 
  (⌊n^2 / 9⌋ : ℤ) - (⌊n / 3⌋ : ℤ)^2 = 3 ↔ n = 8 ∨ n = 10 :=
sorry

end NUMINAMATH_CALUDE_floor_equation_solutions_l2474_247441


namespace NUMINAMATH_CALUDE_max_distinct_angles_for_ten_points_l2474_247444

/-- The number of points on the circle -/
def n : ℕ := 10

/-- The maximum number of distinct inscribed angle values -/
def max_distinct_angles : ℕ := 80

/-- A function that calculates the number of distinct inscribed angle values
    given the number of points on a circle -/
noncomputable def distinct_angles (points : ℕ) : ℕ := sorry

/-- Theorem stating that the maximum number of distinct inscribed angle values
    for 10 points on a circle is 80 -/
theorem max_distinct_angles_for_ten_points :
  distinct_angles n = max_distinct_angles :=
sorry

end NUMINAMATH_CALUDE_max_distinct_angles_for_ten_points_l2474_247444


namespace NUMINAMATH_CALUDE_constant_function_l2474_247414

theorem constant_function (t : ℝ) (f : ℝ → ℝ) 
  (h1 : f 0 = (1 : ℝ) / 2)
  (h2 : ∀ x y : ℝ, f (x + y) = f x * f (t - y) + f y * f (t - x)) :
  ∀ x : ℝ, f x = (1 : ℝ) / 2 :=
by sorry

end NUMINAMATH_CALUDE_constant_function_l2474_247414


namespace NUMINAMATH_CALUDE_vincent_songs_before_camp_l2474_247443

/-- The number of songs Vincent knows now -/
def total_songs : ℕ := 74

/-- The number of songs Vincent learned at summer camp -/
def learned_at_camp : ℕ := 18

/-- The number of songs Vincent knew before summer camp -/
def songs_before_camp : ℕ := total_songs - learned_at_camp

theorem vincent_songs_before_camp :
  songs_before_camp = 56 :=
sorry

end NUMINAMATH_CALUDE_vincent_songs_before_camp_l2474_247443


namespace NUMINAMATH_CALUDE_perfect_square_polynomial_l2474_247447

/-- If the polynomial x^4 + 3x^3 + x^2 + ax + b is the square of some other polynomial, then b = 25/64 -/
theorem perfect_square_polynomial (a b : ℝ) : 
  (∃ (p q : ℝ), ∀ x, x^4 + 3*x^3 + x^2 + a*x + b = (x^2 + p*x + q)^2) →
  b = 25/64 := by
sorry

end NUMINAMATH_CALUDE_perfect_square_polynomial_l2474_247447


namespace NUMINAMATH_CALUDE_solution_to_system_of_equations_l2474_247481

theorem solution_to_system_of_equations :
  ∃ (x y : ℝ), 
    (1 / x + 1 / y = 1) ∧
    (2 / x + 3 / y = 4) ∧
    (x = -1) ∧
    (y = 1 / 2) := by
  sorry

end NUMINAMATH_CALUDE_solution_to_system_of_equations_l2474_247481


namespace NUMINAMATH_CALUDE_salesman_profit_l2474_247471

/-- Calculates the profit of a salesman selling backpacks --/
theorem salesman_profit (total_backpacks : ℕ) (case_cost : ℕ)
  (swap_meet_qty : ℕ) (swap_meet_price : ℕ)
  (dept_store_qty : ℕ) (dept_store_price : ℕ)
  (online_qty : ℕ) (online_price : ℕ)
  (online_shipping : ℕ) (local_market_price : ℕ) :
  total_backpacks = 72 →
  case_cost = 1080 →
  swap_meet_qty = 25 →
  swap_meet_price = 20 →
  dept_store_qty = 18 →
  dept_store_price = 30 →
  online_qty = 12 →
  online_price = 28 →
  online_shipping = 40 →
  local_market_price = 24 →
  (swap_meet_qty * swap_meet_price +
   dept_store_qty * dept_store_price +
   online_qty * online_price - online_shipping +
   (total_backpacks - swap_meet_qty - dept_store_qty - online_qty) * local_market_price) -
  case_cost = 664 :=
by sorry

end NUMINAMATH_CALUDE_salesman_profit_l2474_247471


namespace NUMINAMATH_CALUDE_profit_starts_third_year_max_average_profit_at_six_option_i_more_cost_effective_l2474_247473

-- Define f(n) in ten thousand yuan
def f (n : ℕ) : ℤ := -2*n^2 + 40*n - 72

-- Question 1: Prove that the factory starts to make a profit from the third year
theorem profit_starts_third_year : 
  ∀ n : ℕ, n > 0 → (f n > 0 ↔ n ≥ 3) :=
sorry

-- Question 2: Prove that the annual average net profit reaches its maximum when n = 6
theorem max_average_profit_at_six :
  ∀ n : ℕ, n > 0 → f n / n ≤ f 6 / 6 :=
sorry

-- Question 3: Prove that option (i) is more cost-effective
theorem option_i_more_cost_effective :
  f 6 + 48 > f 10 + 10 :=
sorry

end NUMINAMATH_CALUDE_profit_starts_third_year_max_average_profit_at_six_option_i_more_cost_effective_l2474_247473


namespace NUMINAMATH_CALUDE_managers_wage_l2474_247485

/-- Represents the hourly wages of employees at Joe's Steakhouse -/
structure Wages where
  manager : ℝ
  chef : ℝ
  dishwasher : ℝ

/-- The wages at Joe's Steakhouse satisfy the given conditions -/
def valid_wages (w : Wages) : Prop :=
  w.chef = w.dishwasher * 1.25 ∧
  w.dishwasher = w.manager / 2 ∧
  w.chef = w.manager - 3.1875

theorem managers_wage (w : Wages) (h : valid_wages w) : w.manager = 8.5 := by
  sorry

end NUMINAMATH_CALUDE_managers_wage_l2474_247485


namespace NUMINAMATH_CALUDE_iggys_pace_l2474_247457

/-- Iggy's running schedule for the week -/
def daily_miles : List Nat := [3, 4, 6, 8, 3]

/-- Total time Iggy spent running in hours -/
def total_hours : Nat := 4

/-- Calculate Iggy's pace in minutes per mile -/
def calculate_pace (miles : List Nat) (hours : Nat) : Nat :=
  let total_miles := miles.sum
  let total_minutes := hours * 60
  total_minutes / total_miles

/-- Theorem: Iggy's pace is 10 minutes per mile -/
theorem iggys_pace :
  calculate_pace daily_miles total_hours = 10 := by
  sorry

end NUMINAMATH_CALUDE_iggys_pace_l2474_247457


namespace NUMINAMATH_CALUDE_vector_arrangement_exists_l2474_247424

theorem vector_arrangement_exists : ∃ (a b c : ℝ × ℝ),
  (‖a + b‖ = 1) ∧
  (‖b + c‖ = 1) ∧
  (‖c + a‖ = 1) ∧
  (a + b + c = (0, 0)) := by
  sorry

end NUMINAMATH_CALUDE_vector_arrangement_exists_l2474_247424


namespace NUMINAMATH_CALUDE_length_of_BC_l2474_247458

-- Define the triangles and their properties
def triangle_ABC (AB AC BC : ℝ) : Prop :=
  AB^2 + AC^2 = BC^2 ∧ AB > 0 ∧ AC > 0 ∧ BC > 0

def triangle_ABD (AB AD BD : ℝ) : Prop :=
  AB^2 + AD^2 = BD^2 ∧ AB > 0 ∧ AD > 0 ∧ BD > 0

-- State the theorem
theorem length_of_BC :
  ∀ AB AC BC AD BD,
  triangle_ABC AB AC BC →
  triangle_ABD AB AD BD →
  AB = 12 →
  AC = 16 →
  AD = 30 →
  BC = 20 :=
sorry

end NUMINAMATH_CALUDE_length_of_BC_l2474_247458


namespace NUMINAMATH_CALUDE_albert_large_pizzas_l2474_247406

/-- The number of large pizzas Albert bought -/
def large_pizzas : ℕ := 2

/-- The number of small pizzas Albert bought -/
def small_pizzas : ℕ := 2

/-- The number of slices in a large pizza -/
def large_pizza_slices : ℕ := 16

/-- The number of slices in a small pizza -/
def small_pizza_slices : ℕ := 8

/-- The total number of slices Albert ate -/
def total_slices : ℕ := 48

/-- Theorem stating that Albert bought 2 large pizzas -/
theorem albert_large_pizzas : 
  large_pizzas * large_pizza_slices + small_pizzas * small_pizza_slices = total_slices :=
by sorry

end NUMINAMATH_CALUDE_albert_large_pizzas_l2474_247406


namespace NUMINAMATH_CALUDE_integer_solution_of_inequalities_l2474_247466

theorem integer_solution_of_inequalities :
  ∃! (x : ℤ), (3 * x - 4 ≤ 6 * x - 2) ∧ ((2 * x + 1) / 3 - 1 < (x - 1) / 2) ∧ x = 0 := by
  sorry

end NUMINAMATH_CALUDE_integer_solution_of_inequalities_l2474_247466


namespace NUMINAMATH_CALUDE_count_standing_orders_l2474_247440

/-- The number of different standing orders for 9 students -/
def standing_orders : ℕ := 20

/-- The number of students -/
def num_students : ℕ := 9

/-- The position of the tallest student (middle position) -/
def tallest_position : ℕ := 5

/-- The rank of the student who must stand next to the tallest -/
def adjacent_rank : ℕ := 4

/-- Theorem stating the number of different standing orders -/
theorem count_standing_orders :
  standing_orders = 20 ∧
  num_students = 9 ∧
  tallest_position = 5 ∧
  adjacent_rank = 4 := by
  sorry


end NUMINAMATH_CALUDE_count_standing_orders_l2474_247440


namespace NUMINAMATH_CALUDE_rectangle_area_l2474_247436

/-- The area of a rectangle with perimeter equal to a triangle with sides 7, 9, and 10,
    and length twice its width, is 338/9 square centimeters. -/
theorem rectangle_area (w : ℝ) (h : 2 * (2 * w + w) = 7 + 9 + 10) :
  2 * w * w = 338 / 9 := by
  sorry

end NUMINAMATH_CALUDE_rectangle_area_l2474_247436


namespace NUMINAMATH_CALUDE_total_insects_l2474_247459

def insect_collection (R S C P B E : ℕ) : Prop :=
  R = 15 ∧
  S = 2 * R - 8 ∧
  C = R / 2 + 3 ∧
  P = 3 * S + 7 ∧
  B = 4 * C - 2 ∧
  E = 3 * (R + S + C + P + B)

theorem total_insects (R S C P B E : ℕ) :
  insect_collection R S C P B E →
  R + S + C + P + B + E = 652 :=
by sorry

end NUMINAMATH_CALUDE_total_insects_l2474_247459


namespace NUMINAMATH_CALUDE_S_is_infinite_l2474_247418

/-- Number of distinct odd prime divisors of a natural number -/
def num_odd_prime_divisors (m : ℕ) : ℕ := sorry

/-- The set of natural numbers n for which the number of distinct odd prime divisors of n(n+3) is divisible by 3 -/
def S : Set ℕ := {n : ℕ | 3 ∣ num_odd_prime_divisors (n * (n + 3))}

/-- The set S is infinite -/
theorem S_is_infinite : Set.Infinite S := by sorry

end NUMINAMATH_CALUDE_S_is_infinite_l2474_247418


namespace NUMINAMATH_CALUDE_sams_effective_speed_l2474_247415

/-- Represents the problem of calculating Sam's effective average speed -/
theorem sams_effective_speed (total_distance total_time first_speed second_speed stop_time : ℝ) :
  total_distance = 120 →
  total_time = 120 →
  first_speed = 50 →
  second_speed = 55 →
  stop_time = 5 →
  let first_segment_time : ℝ := 40
  let second_segment_time : ℝ := 40
  let last_segment_time : ℝ := total_time - first_segment_time - second_segment_time
  let first_segment_distance : ℝ := first_speed * (first_segment_time / 60)
  let second_segment_distance : ℝ := second_speed * (second_segment_time / 60)
  let last_segment_distance : ℝ := total_distance - first_segment_distance - second_segment_distance
  let effective_driving_time : ℝ := last_segment_time - stop_time
  let effective_average_speed : ℝ := (last_segment_distance / effective_driving_time) * 60
  effective_average_speed = 85 := by
  sorry

end NUMINAMATH_CALUDE_sams_effective_speed_l2474_247415


namespace NUMINAMATH_CALUDE_jakes_weight_l2474_247404

/-- Given the weights of Mildred, Carol, and Jake, prove Jake's weight -/
theorem jakes_weight (mildred_weight : ℕ) (carol_weight : ℕ) (jake_weight : ℕ) 
  (h1 : mildred_weight = 59)
  (h2 : carol_weight = mildred_weight + 9)
  (h3 : jake_weight = 2 * carol_weight) : 
  jake_weight = 136 := by
  sorry

end NUMINAMATH_CALUDE_jakes_weight_l2474_247404


namespace NUMINAMATH_CALUDE_pizza_combinations_six_toppings_l2474_247463

/-- The number of different one- and two-topping pizzas that can be ordered from a pizza parlor with a given number of toppings. -/
def pizza_combinations (n : ℕ) : ℕ :=
  n + n * (n - 1) / 2

/-- Theorem: The number of different one- and two-topping pizzas that can be ordered from a pizza parlor with 6 toppings is equal to 21. -/
theorem pizza_combinations_six_toppings :
  pizza_combinations 6 = 21 := by
  sorry

end NUMINAMATH_CALUDE_pizza_combinations_six_toppings_l2474_247463


namespace NUMINAMATH_CALUDE_banana_bread_ratio_l2474_247468

def bananas_per_loaf : ℕ := 4
def monday_loaves : ℕ := 3
def total_bananas : ℕ := 36

def tuesday_loaves : ℕ := (total_bananas - monday_loaves * bananas_per_loaf) / bananas_per_loaf

theorem banana_bread_ratio :
  tuesday_loaves / monday_loaves = 2 :=
sorry

end NUMINAMATH_CALUDE_banana_bread_ratio_l2474_247468


namespace NUMINAMATH_CALUDE_prob_at_least_one_from_subset_l2474_247410

/-- The probability of selecting at least one song from a specified subset when randomly choosing 2 out of 4 songs -/
theorem prob_at_least_one_from_subset :
  let total_songs : ℕ := 4
  let songs_to_play : ℕ := 2
  let subset_size : ℕ := 2
  Nat.choose total_songs songs_to_play = 6 →
  (1 : ℚ) - (Nat.choose (total_songs - subset_size) songs_to_play : ℚ) / (Nat.choose total_songs songs_to_play : ℚ) = 5/6 :=
by sorry

end NUMINAMATH_CALUDE_prob_at_least_one_from_subset_l2474_247410


namespace NUMINAMATH_CALUDE_binomial_coefficient_20_10_l2474_247433

theorem binomial_coefficient_20_10 (h1 : Nat.choose 17 7 = 19448)
                                   (h2 : Nat.choose 17 8 = 24310)
                                   (h3 : Nat.choose 17 9 = 24310) :
  Nat.choose 20 10 = 111826 := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_20_10_l2474_247433


namespace NUMINAMATH_CALUDE_total_revenue_over_three_days_l2474_247413

-- Define pie types
inductive PieType
  | Apple
  | Blueberry
  | Cherry

-- Define a structure for daily sales data
structure DailySales where
  apple_price : ℕ
  blueberry_price : ℕ
  cherry_price : ℕ
  apple_sold : ℕ
  blueberry_sold : ℕ
  cherry_sold : ℕ

def slices_per_pie : ℕ := 6

def day1_sales : DailySales := {
  apple_price := 5,
  blueberry_price := 6,
  cherry_price := 7,
  apple_sold := 12,
  blueberry_sold := 8,
  cherry_sold := 10
}

def day2_sales : DailySales := {
  apple_price := 6,
  blueberry_price := 7,
  cherry_price := 8,
  apple_sold := 15,
  blueberry_sold := 10,
  cherry_sold := 14
}

def day3_sales : DailySales := {
  apple_price := 4,
  blueberry_price := 7,
  cherry_price := 9,
  apple_sold := 18,
  blueberry_sold := 7,
  cherry_sold := 13
}

def calculate_daily_revenue (sales : DailySales) : ℕ :=
  sales.apple_price * slices_per_pie * sales.apple_sold +
  sales.blueberry_price * slices_per_pie * sales.blueberry_sold +
  sales.cherry_price * slices_per_pie * sales.cherry_sold

theorem total_revenue_over_three_days :
  calculate_daily_revenue day1_sales +
  calculate_daily_revenue day2_sales +
  calculate_daily_revenue day3_sales = 4128 := by
  sorry


end NUMINAMATH_CALUDE_total_revenue_over_three_days_l2474_247413


namespace NUMINAMATH_CALUDE_euler_negative_two_i_in_third_quadrant_l2474_247465

-- Define the complex exponential function
noncomputable def cexp (z : ℂ) : ℂ := Real.exp z.re * (Real.cos z.im + Complex.I * Real.sin z.im)

-- Define the third quadrant
def third_quadrant (z : ℂ) : Prop := z.re < 0 ∧ z.im < 0

-- Theorem statement
theorem euler_negative_two_i_in_third_quadrant :
  third_quadrant (cexp (-2 * Complex.I)) := by
  sorry

end NUMINAMATH_CALUDE_euler_negative_two_i_in_third_quadrant_l2474_247465


namespace NUMINAMATH_CALUDE_exists_four_axes_symmetry_l2474_247488

/-- A type representing a figure on a grid paper -/
structure GridFigure where
  cells : Set (ℤ × ℤ)

/-- A type representing an axis of symmetry -/
structure AxisOfSymmetry where
  -- Define properties of an axis of symmetry

/-- Function to count the number of axes of symmetry in a figure -/
def countAxesOfSymmetry (f : GridFigure) : ℕ := sorry

/-- Function to shade one more cell in a figure -/
def shadeOneMoreCell (f : GridFigure) : GridFigure := sorry

/-- Theorem stating that it's possible to create a figure with four axes of symmetry 
    by shading one more cell in a figure with no axes of symmetry -/
theorem exists_four_axes_symmetry :
  ∃ (f : GridFigure), 
    countAxesOfSymmetry f = 0 ∧ 
    ∃ (g : GridFigure), g = shadeOneMoreCell f ∧ countAxesOfSymmetry g = 4 := by
  sorry

end NUMINAMATH_CALUDE_exists_four_axes_symmetry_l2474_247488


namespace NUMINAMATH_CALUDE_shaded_area_fraction_l2474_247453

theorem shaded_area_fraction (total_squares : ℕ) (half_shaded : ℕ) (full_shaded : ℕ) :
  total_squares = 18 →
  half_shaded = 10 →
  full_shaded = 3 →
  (half_shaded / 2 + full_shaded : ℚ) / total_squares = 4 / 9 := by
  sorry

end NUMINAMATH_CALUDE_shaded_area_fraction_l2474_247453


namespace NUMINAMATH_CALUDE_point_on_line_l2474_247487

theorem point_on_line : ∃ (x y : ℚ), x = 3 ∧ y = 16/7 ∧ 4*x + 7*y = 28 := by
  sorry

end NUMINAMATH_CALUDE_point_on_line_l2474_247487


namespace NUMINAMATH_CALUDE_not_p_sufficient_not_necessary_q_l2474_247407

-- Define the propositions p and q
def p (m : ℝ) : Prop := m ≥ (1/4 : ℝ)
def q (m : ℝ) : Prop := ∃ x : ℝ, x^2 + x + m = 0

-- Define the sufficient but not necessary condition
def sufficient_not_necessary (P Q : Prop) : Prop :=
  (P → Q) ∧ ¬(Q → P)

-- Theorem statement
theorem not_p_sufficient_not_necessary_q :
  sufficient_not_necessary (¬∀ m, p m) (∀ m, q m) :=
sorry

end NUMINAMATH_CALUDE_not_p_sufficient_not_necessary_q_l2474_247407


namespace NUMINAMATH_CALUDE_specific_conference_handshakes_l2474_247431

/-- The number of handshakes in a conference with gremlins and imps -/
def conference_handshakes (num_gremlins num_imps : ℕ) : ℕ :=
  let gremlin_gremlin := num_gremlins.choose 2
  let gremlin_imp := num_gremlins * num_imps
  gremlin_gremlin + gremlin_imp

/-- Theorem stating the number of handshakes in the specific conference -/
theorem specific_conference_handshakes :
  conference_handshakes 25 10 = 550 := by
  sorry

end NUMINAMATH_CALUDE_specific_conference_handshakes_l2474_247431


namespace NUMINAMATH_CALUDE_hulk_seventh_jump_exceeds_1500_l2474_247425

def hulk_jump (n : ℕ) : ℝ :=
  3 * (3 ^ (n - 1))

theorem hulk_seventh_jump_exceeds_1500 :
  (∀ k < 7, hulk_jump k ≤ 1500) ∧ hulk_jump 7 > 1500 :=
sorry

end NUMINAMATH_CALUDE_hulk_seventh_jump_exceeds_1500_l2474_247425


namespace NUMINAMATH_CALUDE_unique_final_number_l2474_247411

def sum_of_digits (n : ℕ) : ℕ :=
  if n < 10 then n else (n % 10) + sum_of_digits (n / 10)

def is_valid_final_number (n : ℕ) : Prop :=
  15 ≤ n ∧ n ≤ 25 ∧ sum_of_digits n % 9 = 1

theorem unique_final_number :
  ∃! n : ℕ, is_valid_final_number n ∧ n = 19 :=
sorry

end NUMINAMATH_CALUDE_unique_final_number_l2474_247411


namespace NUMINAMATH_CALUDE_compound_interest_calculation_l2474_247484

/-- Calculates the final amount after compound interest for two years with different rates -/
def final_amount (initial : ℝ) (rate1 : ℝ) (rate2 : ℝ) : ℝ :=
  let amount1 := initial * (1 + rate1)
  amount1 * (1 + rate2)

/-- Theorem stating that given the initial amount and interest rates, 
    the final amount after 2 years is as calculated -/
theorem compound_interest_calculation :
  final_amount 4368 0.04 0.05 = 4769.856 := by
  sorry

#eval final_amount 4368 0.04 0.05

end NUMINAMATH_CALUDE_compound_interest_calculation_l2474_247484


namespace NUMINAMATH_CALUDE_arithmetic_sequence_10th_term_l2474_247454

/-- An arithmetic sequence with given conditions -/
def arithmetic_sequence (a : ℕ → ℚ) : Prop :=
  (∃ d : ℚ, ∀ n : ℕ, a (n + 1) = a n + d) ∧ 
  a 1 = 4 ∧
  a 2 + a 4 = 4

/-- The 10th term of the arithmetic sequence is -5 -/
theorem arithmetic_sequence_10th_term 
  (a : ℕ → ℚ) 
  (h : arithmetic_sequence a) : 
  a 10 = -5 := by
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_10th_term_l2474_247454


namespace NUMINAMATH_CALUDE_dogsled_speed_difference_l2474_247492

/-- Calculates the difference in average speed between two dogsled teams -/
theorem dogsled_speed_difference 
  (course_distance : ℝ) 
  (team_e_speed : ℝ) 
  (time_difference : ℝ) : 
  course_distance = 300 →
  team_e_speed = 20 →
  time_difference = 3 →
  (course_distance / (course_distance / team_e_speed - time_difference)) - team_e_speed = 5 := by
  sorry


end NUMINAMATH_CALUDE_dogsled_speed_difference_l2474_247492


namespace NUMINAMATH_CALUDE_rug_design_inner_length_l2474_247446

theorem rug_design_inner_length : 
  ∀ (y : ℝ), 
  let inner_area := 2 * y
  let middle_area := 6 * y + 24
  let outer_area := 10 * y + 80
  (middle_area - inner_area = outer_area - middle_area) →
  y = 4 := by
sorry

end NUMINAMATH_CALUDE_rug_design_inner_length_l2474_247446


namespace NUMINAMATH_CALUDE_fraction_equivalence_l2474_247496

theorem fraction_equivalence : 
  ∀ x : ℝ, x ≠ 0 → (x / (740/999)) * (5/9) = x / 1.4814814814814814 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equivalence_l2474_247496


namespace NUMINAMATH_CALUDE_polygon_sides_count_l2474_247477

/-- The number of diagonals in a polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ := n * (n - 3) / 2

/-- Theorem: If 3n minus the number of diagonals equals 6, then n equals 6 -/
theorem polygon_sides_count (n : ℕ) (h : 3 * n - num_diagonals n = 6) : n = 6 := by
  sorry

end NUMINAMATH_CALUDE_polygon_sides_count_l2474_247477


namespace NUMINAMATH_CALUDE_triangle_two_solutions_l2474_247498

-- Define the triangle
def Triangle (a b c : ℝ) (A B C : ℝ) : Prop :=
  a > 0 ∧ b > 0 ∧ c > 0 ∧ A > 0 ∧ B > 0 ∧ C > 0

-- State the theorem
theorem triangle_two_solutions 
  (a b c : ℝ) (A B C : ℝ) 
  (h_triangle : Triangle a b c A B C)
  (h_a : a = 2)
  (h_A : A = Real.pi / 3) -- 60° in radians
  (h_two_solutions : b * Real.sin A < a ∧ a < b) :
  2 < b ∧ b < 4 * Real.sqrt 3 / 3 :=
sorry

end NUMINAMATH_CALUDE_triangle_two_solutions_l2474_247498


namespace NUMINAMATH_CALUDE_negation_of_universal_non_negative_square_l2474_247450

theorem negation_of_universal_non_negative_square (p : ℝ → Prop) : 
  (¬ ∀ x : ℝ, x^2 ≥ 0) ↔ (∃ x₀ : ℝ, x₀^2 < 0) :=
by sorry

end NUMINAMATH_CALUDE_negation_of_universal_non_negative_square_l2474_247450


namespace NUMINAMATH_CALUDE_max_third_side_triangle_l2474_247483

theorem max_third_side_triangle (D E F : Real) (a b : Real) :
  -- Triangle DEF exists
  0 < D ∧ 0 < E ∧ 0 < F ∧ D + E + F = π →
  -- Two sides are 12 and 15
  a = 12 ∧ b = 15 →
  -- Angle condition
  Real.cos (2 * D) + Real.cos (2 * E) + Real.cos (2 * F) = 1 / 2 →
  -- Maximum length of third side
  ∃ c : Real, c ≤ Real.sqrt 549 ∧
    ∀ c' : Real, (c' = Real.sqrt (a^2 + b^2 - 2*a*b*(Real.cos E)) → c' ≤ c) :=
by sorry

end NUMINAMATH_CALUDE_max_third_side_triangle_l2474_247483


namespace NUMINAMATH_CALUDE_angle_value_l2474_247486

def A (θ : ℝ) : Set ℝ := {1, Real.cos θ}
def B : Set ℝ := {0, 1/2, 1}

theorem angle_value (θ : ℝ) (h1 : A θ ⊆ B) (h2 : 0 < θ ∧ θ < π / 2) : θ = π / 3 := by
  sorry

end NUMINAMATH_CALUDE_angle_value_l2474_247486


namespace NUMINAMATH_CALUDE_difference_of_squares_and_sum_l2474_247461

theorem difference_of_squares_and_sum (m n : ℤ) 
  (h1 : m^2 - n^2 = 6) 
  (h2 : m + n = 3) : 
  n - m = -2 := by sorry

end NUMINAMATH_CALUDE_difference_of_squares_and_sum_l2474_247461


namespace NUMINAMATH_CALUDE_sum_of_common_x_coords_l2474_247493

/-- Given two congruences modulo 16, find the sum of x-coordinates of common points -/
theorem sum_of_common_x_coords : ∃ (S : Finset ℕ),
  (∀ x ∈ S, ∃ y : ℕ, (y ≡ 5 * x + 2 [ZMOD 16] ∧ y ≡ 11 * x + 12 [ZMOD 16])) ∧
  (∀ x : ℕ, (∃ y : ℕ, y ≡ 5 * x + 2 [ZMOD 16] ∧ y ≡ 11 * x + 12 [ZMOD 16]) → x ∈ S) ∧
  (Finset.sum S id = 10) :=
by sorry

end NUMINAMATH_CALUDE_sum_of_common_x_coords_l2474_247493


namespace NUMINAMATH_CALUDE_complex_modulus_problem_l2474_247456

theorem complex_modulus_problem (b : ℝ) (z : ℂ) : 
  z = (b * Complex.I) / (4 + 3 * Complex.I) → 
  Complex.abs z = 5 → 
  b = 25 ∨ b = -25 := by
sorry

end NUMINAMATH_CALUDE_complex_modulus_problem_l2474_247456


namespace NUMINAMATH_CALUDE_alex_exam_result_l2474_247435

/-- Represents the scoring system and result of a multiple-choice exam -/
structure ExamResult where
  total_questions : ℕ
  correct_points : ℕ
  blank_points : ℕ
  incorrect_points : ℤ
  total_score : ℤ

/-- Calculates the maximum number of correctly answered questions -/
def max_correct_answers (exam : ExamResult) : ℕ :=
  sorry

/-- Theorem stating that for the given exam conditions, the maximum number of correct answers is 38 -/
theorem alex_exam_result :
  let exam : ExamResult :=
    { total_questions := 60
      correct_points := 5
      blank_points := 0
      incorrect_points := -2
      total_score := 150 }
  max_correct_answers exam = 38 := by
  sorry

end NUMINAMATH_CALUDE_alex_exam_result_l2474_247435


namespace NUMINAMATH_CALUDE_sum_of_reciprocals_squared_l2474_247464

theorem sum_of_reciprocals_squared (a b c d : ℝ) : 
  a = Real.sqrt 2 + Real.sqrt 3 + Real.sqrt 5 →
  b = -Real.sqrt 2 + Real.sqrt 3 + Real.sqrt 5 →
  c = Real.sqrt 2 - Real.sqrt 3 + Real.sqrt 5 →
  d = -Real.sqrt 2 - Real.sqrt 3 + Real.sqrt 5 →
  (1/a + 1/b + 1/c + 1/d)^2 = 1600 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_reciprocals_squared_l2474_247464


namespace NUMINAMATH_CALUDE_circular_arrangement_students_l2474_247419

theorem circular_arrangement_students (n : ℕ) 
  (h1 : n > 0) 
  (h2 : 10 ≤ n ∧ 40 ≤ n) 
  (h3 : (40 - 10) * 2 = n) : n = 60 := by
  sorry

end NUMINAMATH_CALUDE_circular_arrangement_students_l2474_247419


namespace NUMINAMATH_CALUDE_division_remainder_l2474_247467

theorem division_remainder (dividend : ℕ) (divisor : ℕ) (quotient : ℕ) (remainder : ℕ) : 
  dividend = 222 → divisor = 13 → quotient = 17 → 
  dividend = divisor * quotient + remainder → remainder = 1 := by
sorry

end NUMINAMATH_CALUDE_division_remainder_l2474_247467


namespace NUMINAMATH_CALUDE_students_in_two_classes_l2474_247479

theorem students_in_two_classes
  (total_students : ℕ)
  (history : ℕ)
  (math : ℕ)
  (english : ℕ)
  (science : ℕ)
  (geography : ℕ)
  (all_five : ℕ)
  (history_and_math : ℕ)
  (english_and_science : ℕ)
  (math_and_geography : ℕ)
  (h_total : total_students = 500)
  (h_history : history = 120)
  (h_math : math = 105)
  (h_english : english = 145)
  (h_science : science = 133)
  (h_geography : geography = 107)
  (h_all_five : all_five = 15)
  (h_history_and_math : history_and_math = 40)
  (h_english_and_science : english_and_science = 35)
  (h_math_and_geography : math_and_geography = 25)
  (h_at_least_one : total_students ≤ history + math + english + science + geography) :
  (history_and_math - all_five) + (english_and_science - all_five) + (math_and_geography - all_five) = 55 := by
  sorry

end NUMINAMATH_CALUDE_students_in_two_classes_l2474_247479


namespace NUMINAMATH_CALUDE_volunteers_assignment_l2474_247451

/-- The number of ways to assign volunteers to service points -/
def assign_volunteers (n_volunteers : ℕ) (n_points : ℕ) : ℕ :=
  sorry

/-- Theorem stating that assigning 4 volunteers to 3 service points results in 36 ways -/
theorem volunteers_assignment :
  assign_volunteers 4 3 = 36 :=
sorry

end NUMINAMATH_CALUDE_volunteers_assignment_l2474_247451


namespace NUMINAMATH_CALUDE_factorization_equality_l2474_247422

theorem factorization_equality (a b : ℝ) : a * b^2 - 4 * a * b + 4 * a = a * (b - 2)^2 := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l2474_247422


namespace NUMINAMATH_CALUDE_exists_greater_than_product_l2474_247495

/-- A doubly infinite array of positive integers -/
def InfiniteArray := ℕ+ → ℕ+ → ℕ+

/-- The property that each positive integer appears exactly eight times in the array -/
def EightOccurrences (a : InfiniteArray) : Prop :=
  ∀ k : ℕ+, (∃ (S : Finset (ℕ+ × ℕ+)), S.card = 8 ∧ (∀ (p : ℕ+ × ℕ+), p ∈ S ↔ a p.1 p.2 = k))

theorem exists_greater_than_product (a : InfiniteArray) (h : EightOccurrences a) :
  ∃ (m n : ℕ+), a m n > m * n := by
  sorry

end NUMINAMATH_CALUDE_exists_greater_than_product_l2474_247495


namespace NUMINAMATH_CALUDE_intersection_P_Q_l2474_247494

def P : Set ℝ := {x | 1/2 ≤ x ∧ x ≤ 3}
def Q : Set ℝ := {x | x > 4 ∨ x ≤ 2}

theorem intersection_P_Q : P ∩ Q = {x : ℝ | 1/2 ≤ x ∧ x ≤ 2} := by sorry

end NUMINAMATH_CALUDE_intersection_P_Q_l2474_247494


namespace NUMINAMATH_CALUDE_parabola_symmetry_l2474_247423

/-- Represents a quadratic function of the form ax² + bx + c -/
structure QuadraticFunction where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Shifts a quadratic function vertically by a given amount -/
def shift_vertical (f : QuadraticFunction) (shift : ℝ) : QuadraticFunction :=
  { a := f.a, b := f.b, c := f.c + shift }

/-- Checks if two quadratic functions are symmetric about the y-axis -/
def symmetric_about_y_axis (f g : QuadraticFunction) : Prop :=
  f.a = g.a ∧ f.b = -g.b ∧ f.c = g.c

theorem parabola_symmetry (a b : ℝ) (h_a : a ≠ 0) :
  let f : QuadraticFunction := { a := a, b := b, c := -2 }
  let g : QuadraticFunction := { a := 1/2, b := 1, c := -4 }
  symmetric_about_y_axis (shift_vertical f (-2)) g →
  a = 1/2 ∧ b = -1 := by
  sorry

end NUMINAMATH_CALUDE_parabola_symmetry_l2474_247423


namespace NUMINAMATH_CALUDE_vector_b_solution_l2474_247470

def a : ℝ × ℝ := (1, -2)

theorem vector_b_solution (b : ℝ × ℝ) :
  (b.1^2 + b.2^2 = 20) →  -- |b| = 2√5
  (a.1 * b.2 = a.2 * b.1) →  -- a ∥ b
  (b = (2, -4) ∨ b = (-2, 4)) :=
by sorry

end NUMINAMATH_CALUDE_vector_b_solution_l2474_247470


namespace NUMINAMATH_CALUDE_x_value_l2474_247420

/-- The value of x is equal to (47% of 1442 - 36% of 1412) + 65 -/
theorem x_value : 
  (0.47 * 1442 - 0.36 * 1412) + 65 = 234.42 := by
  sorry

end NUMINAMATH_CALUDE_x_value_l2474_247420


namespace NUMINAMATH_CALUDE_odd_even_sum_difference_problem_solution_l2474_247462

theorem odd_even_sum_difference : ℕ → Prop :=
  fun n =>
    let odd_sum := (n^2 + n) / 2
    let even_sum := n * (n + 1)
    odd_sum - even_sum = n + 1

theorem problem_solution :
  let n : ℕ := 1009
  let odd_sum := ((2*n + 1)^2 + (2*n + 1)) / 2
  let even_sum := n * (n + 1)
  odd_sum - even_sum = 1010 := by
  sorry

end NUMINAMATH_CALUDE_odd_even_sum_difference_problem_solution_l2474_247462


namespace NUMINAMATH_CALUDE_jill_peach_count_jill_peach_count_proof_l2474_247412

/-- Given the peach distribution among Jake, Steven, Jill, and Sam, prove that Jill has 6 peaches. -/
theorem jill_peach_count : ℕ → ℕ → ℕ → ℕ → Prop :=
  fun jake steven jill sam =>
    (jake = steven - 18) →
    (steven = jill + 13) →
    (steven = 19) →
    (sam = 2 * jill) →
    (jill = 6)

/-- Proof of the theorem -/
theorem jill_peach_count_proof : ∃ jake steven jill sam, jill_peach_count jake steven jill sam :=
  sorry

end NUMINAMATH_CALUDE_jill_peach_count_jill_peach_count_proof_l2474_247412


namespace NUMINAMATH_CALUDE_least_integer_with_12_factors_l2474_247429

/-- The number of positive factors of a positive integer -/
def num_factors (n : ℕ+) : ℕ := sorry

/-- Theorem: 72 is the least positive integer with exactly 12 positive factors -/
theorem least_integer_with_12_factors :
  (∀ m : ℕ+, m < 72 → num_factors m ≠ 12) ∧ num_factors 72 = 12 := by sorry

end NUMINAMATH_CALUDE_least_integer_with_12_factors_l2474_247429


namespace NUMINAMATH_CALUDE_quadratic_equation_root_quadratic_equation_rational_coefficients_quadratic_equation_leading_coefficient_l2474_247421

theorem quadratic_equation_root (x : ℝ) : x^2 + 6*x + 4 = 0 ↔ x = Real.sqrt 5 - 3 ∨ x = -Real.sqrt 5 - 3 := by sorry

theorem quadratic_equation_rational_coefficients : ∃ a b c : ℚ, a = 1 ∧ ∀ x : ℝ, x^2 + 6*x + 4 = a*x^2 + b*x + c := by sorry

theorem quadratic_equation_leading_coefficient : ∃ a b c : ℝ, a = 1 ∧ ∀ x : ℝ, x^2 + 6*x + 4 = a*x^2 + b*x + c := by sorry

end NUMINAMATH_CALUDE_quadratic_equation_root_quadratic_equation_rational_coefficients_quadratic_equation_leading_coefficient_l2474_247421


namespace NUMINAMATH_CALUDE_ratio_simplification_l2474_247409

theorem ratio_simplification : 
  (10^2001 + 10^2003) / (10^2002 + 10^2002) = 101 / 20 := by sorry

end NUMINAMATH_CALUDE_ratio_simplification_l2474_247409


namespace NUMINAMATH_CALUDE_remainder_sum_of_powers_l2474_247472

theorem remainder_sum_of_powers (n : ℕ) : (9^6 + 8^8 + 7^9) % 7 = 2 := by
  sorry

end NUMINAMATH_CALUDE_remainder_sum_of_powers_l2474_247472


namespace NUMINAMATH_CALUDE_trigonometric_problem_l2474_247400

theorem trigonometric_problem (x y : ℝ) (h_nonzero : x ≠ 0 ∧ y ≠ 0) 
  (h_eq : (x * Real.sin (π/5) + y * Real.cos (π/5)) / (x * Real.cos (π/5) - y * Real.sin (π/5)) = Real.tan (9*π/20)) :
  (y / x = 1) ∧
  (∀ A B : ℝ, 0 < A ∧ 0 < B ∧ A + B = 3*π/4 → 
    Real.sin (2*A) + 2 * Real.cos B ≤ 3/2) ∧
  (∃ A B : ℝ, 0 < A ∧ 0 < B ∧ A + B = 3*π/4 ∧ 
    Real.sin (2*A) + 2 * Real.cos B = 3/2) :=
by sorry

end NUMINAMATH_CALUDE_trigonometric_problem_l2474_247400


namespace NUMINAMATH_CALUDE_rectangular_field_area_l2474_247455

theorem rectangular_field_area (a b c : ℝ) (h1 : a = 13) (h2 : c = 17) (h3 : a^2 + b^2 = c^2) :
  a * b = 26 * Real.sqrt 30 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_field_area_l2474_247455


namespace NUMINAMATH_CALUDE_sum_of_four_sqrt_inequality_l2474_247499

theorem sum_of_four_sqrt_inequality (a b c d : ℝ) 
  (pos_a : 0 < a) (pos_b : 0 < b) (pos_c : 0 < c) (pos_d : 0 < d)
  (sum_eq_one : a + b + c + d = 1) :
  Real.sqrt (4 * a + 1) + Real.sqrt (4 * b + 1) + Real.sqrt (4 * c + 1) + Real.sqrt (4 * d + 1) < 6 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_four_sqrt_inequality_l2474_247499


namespace NUMINAMATH_CALUDE_cubic_roots_sum_l2474_247474

theorem cubic_roots_sum (a b c : ℝ) : 
  (a^3 - a - 2 = 0) → 
  (b^3 - b - 2 = 0) → 
  (c^3 - c - 2 = 0) → 
  2*a*(b - c)^2 + 2*b*(c - a)^2 + 2*c*(a - b)^2 = -36 := by
  sorry

end NUMINAMATH_CALUDE_cubic_roots_sum_l2474_247474


namespace NUMINAMATH_CALUDE_polynomial_coefficient_sum_difference_squares_l2474_247476

theorem polynomial_coefficient_sum_difference_squares (a₀ a₁ a₂ a₃ a₄ : ℝ) :
  (∀ x : ℝ, (2 * x + Real.sqrt 3) ^ 4 = a₀ + a₁ * x + a₂ * x^2 + a₃ * x^3 + a₄ * x^4) →
  (a₀ + a₂ + a₄)^2 - (a₁ + a₃)^2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_coefficient_sum_difference_squares_l2474_247476


namespace NUMINAMATH_CALUDE_total_fruit_cost_l2474_247426

def grapes_cost : ℚ := 12.08
def cherries_cost : ℚ := 9.85

theorem total_fruit_cost : grapes_cost + cherries_cost = 21.93 := by
  sorry

end NUMINAMATH_CALUDE_total_fruit_cost_l2474_247426


namespace NUMINAMATH_CALUDE_specific_student_not_front_l2474_247490

/-- The number of ways to arrange n students in a line. -/
def arrangements (n : ℕ) : ℕ := Nat.factorial n

/-- The number of ways to arrange n students in a line with a specific student at the front. -/
def arrangementsWithSpecificFront (n : ℕ) : ℕ := Nat.factorial (n - 1)

/-- The number of students. -/
def numStudents : ℕ := 5

theorem specific_student_not_front :
  arrangements numStudents - arrangementsWithSpecificFront numStudents = 96 :=
sorry

end NUMINAMATH_CALUDE_specific_student_not_front_l2474_247490


namespace NUMINAMATH_CALUDE_locus_of_C1_l2474_247445

-- Define the parabola
def parabola (x : ℝ) : ℝ := x^2

-- Define a chord parallel to x-axis
structure Chord :=
  (a : ℝ)
  (property : parabola a = parabola (-a))

-- Define a point on the parabola
structure ParabolaPoint :=
  (x : ℝ)
  (y : ℝ)
  (on_parabola : y = parabola x)

-- Define the circumcircle of a triangle
def circumcircle (A B C : ParabolaPoint) : Set (ℝ × ℝ) := sorry

-- Define a point on the circumcircle with the same x-coordinate as C
def C1 (C : ParabolaPoint) (circle : Set (ℝ × ℝ)) : ℝ × ℝ := sorry

-- The main theorem
theorem locus_of_C1 (AB : Chord) (C : ParabolaPoint) 
  (hC : C.x ≠ AB.a ∧ C.x ≠ -AB.a) :
  let A := ⟨AB.a, parabola AB.a, rfl⟩
  let B := ⟨-AB.a, parabola (-AB.a), rfl⟩
  let circle := circumcircle A B C
  let c1 := C1 C circle
  c1.2 = 1 + AB.a^2 := by sorry

end NUMINAMATH_CALUDE_locus_of_C1_l2474_247445


namespace NUMINAMATH_CALUDE_max_value_A_l2474_247434

theorem max_value_A (x y z : ℝ) (hx : 0 < x ∧ x ≤ 1) (hy : 0 < y ∧ y ≤ 1) (hz : 0 < z ∧ z ≤ 1) :
  (Real.sqrt (8 * x^4 + y) + Real.sqrt (8 * y^4 + z) + Real.sqrt (8 * z^4 + x) - 3) / (x + y + z) ≤ 2 :=
by sorry

end NUMINAMATH_CALUDE_max_value_A_l2474_247434


namespace NUMINAMATH_CALUDE_johns_age_is_15_l2474_247469

-- Define John's age and his father's age
def johns_age : ℕ := sorry
def fathers_age : ℕ := sorry

-- State the conditions
axiom sum_of_ages : johns_age + fathers_age = 77
axiom father_age_relation : fathers_age = 2 * johns_age + 32

-- Theorem to prove
theorem johns_age_is_15 : johns_age = 15 := by sorry

end NUMINAMATH_CALUDE_johns_age_is_15_l2474_247469


namespace NUMINAMATH_CALUDE_inequality_equivalence_l2474_247460

theorem inequality_equivalence (x : ℝ) : (x + 1) * (2 - x) > 0 ↔ x ∈ Set.Ioo (-1) 2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_equivalence_l2474_247460


namespace NUMINAMATH_CALUDE_y₁_less_than_y₂_l2474_247480

/-- A linear function f(x) = -4x + 3 -/
def f (x : ℝ) : ℝ := -4 * x + 3

/-- Point P₁ is on the graph of f -/
def P₁_on_graph (y₁ : ℝ) : Prop := f 1 = y₁

/-- Point P₂ is on the graph of f -/
def P₂_on_graph (y₂ : ℝ) : Prop := f (-3) = y₂

/-- Theorem stating the relationship between y₁ and y₂ -/
theorem y₁_less_than_y₂ (y₁ y₂ : ℝ) (h₁ : P₁_on_graph y₁) (h₂ : P₂_on_graph y₂) : y₁ < y₂ := by
  sorry

end NUMINAMATH_CALUDE_y₁_less_than_y₂_l2474_247480


namespace NUMINAMATH_CALUDE_homework_ratio_l2474_247439

theorem homework_ratio (total : ℕ) (finished : ℕ) 
  (h1 : total = 65) (h2 : finished = 45) : 
  (finished : ℚ) / (total - finished) = 9 / 4 := by
  sorry

end NUMINAMATH_CALUDE_homework_ratio_l2474_247439


namespace NUMINAMATH_CALUDE_f_greater_than_one_range_l2474_247432

noncomputable def f (x : ℝ) : ℝ :=
  if x ≤ 0 then 2^(-x) - 1 else Real.sqrt x

theorem f_greater_than_one_range :
  {x : ℝ | f x > 1} = {x : ℝ | x > 1 ∨ x < -1} := by sorry

end NUMINAMATH_CALUDE_f_greater_than_one_range_l2474_247432


namespace NUMINAMATH_CALUDE_acme_cheaper_at_min_shirts_l2474_247475

/-- Acme's setup fee -/
def acme_setup : ℕ := 75

/-- Acme's per-shirt cost -/
def acme_per_shirt : ℕ := 8

/-- Gamma's per-shirt cost -/
def gamma_per_shirt : ℕ := 16

/-- The minimum number of shirts for which Acme becomes cheaper than Gamma -/
def min_shirts : ℕ := 10

theorem acme_cheaper_at_min_shirts :
  acme_setup + acme_per_shirt * min_shirts < gamma_per_shirt * min_shirts ∧
  ∀ n : ℕ, n < min_shirts →
    acme_setup + acme_per_shirt * n ≥ gamma_per_shirt * n :=
by sorry

end NUMINAMATH_CALUDE_acme_cheaper_at_min_shirts_l2474_247475


namespace NUMINAMATH_CALUDE_goldfish_disappeared_l2474_247482

theorem goldfish_disappeared (original : ℕ) (left : ℕ) (disappeared : ℕ) : 
  original = 15 → left = 4 → disappeared = original - left → disappeared = 11 := by
  sorry

end NUMINAMATH_CALUDE_goldfish_disappeared_l2474_247482


namespace NUMINAMATH_CALUDE_determinant_evaluation_l2474_247428

theorem determinant_evaluation (x y z : ℝ) : 
  Matrix.det !![x + 1, y, z; y, x + 1, z; z, y, x + 1] = 
    x^3 + 3*x^2 + 3*x + 1 - x*y*z - x*y^2 - y*z^2 - z*x^2 - z*x + y*z^2 + z*y^2 := by
  sorry

end NUMINAMATH_CALUDE_determinant_evaluation_l2474_247428


namespace NUMINAMATH_CALUDE_school_buses_l2474_247417

def bus_seats (columns : ℕ) (rows : ℕ) : ℕ := columns * rows

def total_capacity (num_buses : ℕ) (seats_per_bus : ℕ) : ℕ := num_buses * seats_per_bus

theorem school_buses (columns : ℕ) (rows : ℕ) (total_students : ℕ) (num_buses : ℕ) :
  columns = 4 →
  rows = 10 →
  total_students = 240 →
  total_capacity num_buses (bus_seats columns rows) = total_students →
  num_buses = 6 := by
  sorry

#check school_buses

end NUMINAMATH_CALUDE_school_buses_l2474_247417


namespace NUMINAMATH_CALUDE_identical_answers_possible_l2474_247452

/-- A person who either always tells the truth or always lies -/
inductive TruthTeller
  | Always
  | Never

/-- The response to a question, either Yes or No -/
inductive Response
  | Yes
  | No

/-- Given a question, determine the response of a TruthTeller -/
def respond (person : TruthTeller) (questionTruth : Bool) : Response :=
  match person, questionTruth with
  | TruthTeller.Always, true => Response.Yes
  | TruthTeller.Always, false => Response.No
  | TruthTeller.Never, true => Response.No
  | TruthTeller.Never, false => Response.Yes

theorem identical_answers_possible :
  ∃ (question : Bool),
    respond TruthTeller.Always question = respond TruthTeller.Never question :=
by sorry

end NUMINAMATH_CALUDE_identical_answers_possible_l2474_247452
