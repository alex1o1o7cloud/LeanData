import Mathlib

namespace NUMINAMATH_CALUDE_floor_product_equation_l3093_309307

theorem floor_product_equation (x : ℝ) : 
  (⌊x * ⌊x⌋⌋ = 48) ↔ (x = -48/7) := by sorry

end NUMINAMATH_CALUDE_floor_product_equation_l3093_309307


namespace NUMINAMATH_CALUDE_geometric_mean_sqrt2_plus_minus_one_l3093_309363

theorem geometric_mean_sqrt2_plus_minus_one : 
  ∃ x : ℝ, x^2 = (Real.sqrt 2 - 1) * (Real.sqrt 2 + 1) ∧ (x = 1 ∨ x = -1) :=
by sorry

end NUMINAMATH_CALUDE_geometric_mean_sqrt2_plus_minus_one_l3093_309363


namespace NUMINAMATH_CALUDE_min_a_for_p_geq_half_l3093_309370

def p (a : ℕ) : ℚ :=
  (Nat.choose (36 - a) 2 + Nat.choose (a - 1) 2) / Nat.choose 50 2

theorem min_a_for_p_geq_half :
  ∀ a : ℕ, 1 ≤ a ∧ a ≤ 37 → p a < 1/2 ∧ p 38 ≥ 1/2 := by sorry

end NUMINAMATH_CALUDE_min_a_for_p_geq_half_l3093_309370


namespace NUMINAMATH_CALUDE_major_axis_coincide_condition_l3093_309335

/-- Represents the coefficients of a general ellipse equation -/
structure EllipseCoefficients where
  A : ℝ
  B : ℝ
  C : ℝ
  D : ℝ
  E : ℝ
  F : ℝ

/-- Predicate to check if the major axis coincides with a conjugate diameter -/
def majorAxisCoincideWithConjugateDiameter (c : EllipseCoefficients) : Prop :=
  c.A * c.E - c.B * c.D = 0 ∧ 2 * c.B^2 + (c.A - c.C) * c.A = 0

/-- Theorem stating the conditions for the major axis to coincide with a conjugate diameter -/
theorem major_axis_coincide_condition (c : EllipseCoefficients) :
  majorAxisCoincideWithConjugateDiameter c ↔
  (c.A * c.E - c.B * c.D = 0 ∧ 2 * c.B^2 + (c.A - c.C) * c.A = 0) :=
by sorry

end NUMINAMATH_CALUDE_major_axis_coincide_condition_l3093_309335


namespace NUMINAMATH_CALUDE_B_power_101_l3093_309332

def B : Matrix (Fin 3) (Fin 3) ℤ :=
  ![![0, 1, 0],
    ![0, 0, 1],
    ![1, 0, 0]]

theorem B_power_101 : B^101 = B^2 := by sorry

end NUMINAMATH_CALUDE_B_power_101_l3093_309332


namespace NUMINAMATH_CALUDE_no_real_roots_quadratic_l3093_309326

theorem no_real_roots_quadratic (k : ℝ) :
  (∀ x : ℝ, x^2 - 2*x - k ≠ 0) → k < -1 := by
  sorry

end NUMINAMATH_CALUDE_no_real_roots_quadratic_l3093_309326


namespace NUMINAMATH_CALUDE_horseshoe_profit_is_22000_l3093_309306

/-- Represents the profit calculation for Redo's Horseshoe Company --/
def horseshoe_profit : ℝ :=
  let type_a_initial_outlay : ℝ := 10000
  let type_a_cost_per_set : ℝ := 20
  let type_a_price_high : ℝ := 60
  let type_a_price_low : ℝ := 50
  let type_a_sets_high : ℝ := 300
  let type_a_sets_low : ℝ := 200
  let type_b_initial_outlay : ℝ := 6000
  let type_b_cost_per_set : ℝ := 15
  let type_b_price : ℝ := 40
  let type_b_sets : ℝ := 800

  let type_a_revenue := type_a_price_high * type_a_sets_high + type_a_price_low * type_a_sets_low
  let type_a_cost := type_a_initial_outlay + type_a_cost_per_set * (type_a_sets_high + type_a_sets_low)
  let type_a_profit := type_a_revenue - type_a_cost

  let type_b_revenue := type_b_price * type_b_sets
  let type_b_cost := type_b_initial_outlay + type_b_cost_per_set * type_b_sets
  let type_b_profit := type_b_revenue - type_b_cost

  type_a_profit + type_b_profit

/-- The total profit for Redo's Horseshoe Company is $22,000 --/
theorem horseshoe_profit_is_22000 : horseshoe_profit = 22000 := by
  sorry

end NUMINAMATH_CALUDE_horseshoe_profit_is_22000_l3093_309306


namespace NUMINAMATH_CALUDE_john_computer_cost_l3093_309381

/-- Calculates the total cost of a computer after replacing a video card -/
def totalComputerCost (initialCost oldCardSale newCardCost : ℕ) : ℕ :=
  initialCost - oldCardSale + newCardCost

/-- Proves that the total cost of John's computer is $1400 -/
theorem john_computer_cost :
  totalComputerCost 1200 300 500 = 1400 := by
  sorry

end NUMINAMATH_CALUDE_john_computer_cost_l3093_309381


namespace NUMINAMATH_CALUDE_merry_go_round_ride_times_l3093_309399

theorem merry_go_round_ride_times 
  (dave_time : ℝ) 
  (erica_time : ℝ) 
  (erica_longer_percent : ℝ) :
  dave_time = 10 →
  erica_time = 65 →
  erica_longer_percent = 0.30 →
  ∃ (chuck_time : ℝ),
    erica_time = chuck_time * (1 + erica_longer_percent) ∧
    chuck_time / dave_time = 5 :=
by sorry

end NUMINAMATH_CALUDE_merry_go_round_ride_times_l3093_309399


namespace NUMINAMATH_CALUDE_simplify_sqrt_expression_l3093_309353

theorem simplify_sqrt_expression : 
  Real.sqrt 10 - Real.sqrt 40 + Real.sqrt 90 = 2 * Real.sqrt 10 := by
  sorry

end NUMINAMATH_CALUDE_simplify_sqrt_expression_l3093_309353


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l3093_309347

def M : Set ℕ := {1, 2, 3}
def N : Set ℕ := {2, 3, 4}

theorem intersection_of_M_and_N : M ∩ N = {2, 3} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l3093_309347


namespace NUMINAMATH_CALUDE_method2_more_profitable_above_15000_l3093_309393

/-- Profit calculation for Method 1 (end of month) -/
def profit_method1 (x : ℝ) : ℝ := 0.3 * x - 900

/-- Profit calculation for Method 2 (beginning of month with reinvestment) -/
def profit_method2 (x : ℝ) : ℝ := 0.26 * x

/-- Theorem stating that Method 2 is more profitable when x > 15000 -/
theorem method2_more_profitable_above_15000 (x : ℝ) (h : x > 15000) :
  profit_method2 x > profit_method1 x :=
sorry

end NUMINAMATH_CALUDE_method2_more_profitable_above_15000_l3093_309393


namespace NUMINAMATH_CALUDE_star_associativity_l3093_309301

-- Define the universal set
variable {U : Type}

-- Define the * operation
def star (X Y : Set U) : Set U := (X ∩ Y)ᶜ

-- State the theorem
theorem star_associativity (X Y Z : Set U) : 
  star (star X Y) Z = (Xᶜ ∩ Yᶜ) ∪ Z := by sorry

end NUMINAMATH_CALUDE_star_associativity_l3093_309301


namespace NUMINAMATH_CALUDE_philip_paintings_per_day_l3093_309374

/-- The number of paintings Philip makes per day -/
def paintings_per_day (initial_paintings : ℕ) (total_paintings : ℕ) (days : ℕ) : ℚ :=
  (total_paintings - initial_paintings : ℚ) / days

/-- Theorem: Philip makes 2 paintings per day -/
theorem philip_paintings_per_day :
  paintings_per_day 20 80 30 = 2 := by
  sorry

end NUMINAMATH_CALUDE_philip_paintings_per_day_l3093_309374


namespace NUMINAMATH_CALUDE_townspeople_win_probability_l3093_309313

/-- The probability that the townspeople win in a game with 2 townspeople and 1 goon -/
theorem townspeople_win_probability :
  let total_participants : ℕ := 2 + 1
  let num_goons : ℕ := 1
  let townspeople_win_condition := (num_goons / total_participants : ℚ)
  townspeople_win_condition = 1 / 3 := by
  sorry

end NUMINAMATH_CALUDE_townspeople_win_probability_l3093_309313


namespace NUMINAMATH_CALUDE_largest_integer_satisfying_inequality_l3093_309321

theorem largest_integer_satisfying_inequality : 
  ∀ x : ℤ, x ≤ 3 ↔ (x : ℚ) / 4 + 7 / 6 < 8 / 4 :=
by sorry

end NUMINAMATH_CALUDE_largest_integer_satisfying_inequality_l3093_309321


namespace NUMINAMATH_CALUDE_min_teachers_is_ten_l3093_309356

/-- Represents the number of teachers for each subject -/
structure TeacherCounts where
  math : Nat
  physics : Nat
  chemistry : Nat
  biology : Nat
  computerScience : Nat

/-- Represents the school schedule constraints -/
structure SchoolConstraints where
  teacherCounts : TeacherCounts
  maxSubjectsPerTeacher : Nat
  periodsPerDay : Nat

/-- Calculates the total number of teaching slots required per day -/
def totalSlotsPerDay (c : SchoolConstraints) : Nat :=
  (c.teacherCounts.math + c.teacherCounts.physics + c.teacherCounts.chemistry +
   c.teacherCounts.biology + c.teacherCounts.computerScience) * c.periodsPerDay

/-- Calculates the number of slots a single teacher can fill per day -/
def slotsPerTeacher (c : SchoolConstraints) : Nat :=
  c.maxSubjectsPerTeacher * c.periodsPerDay

/-- Calculates the minimum number of teachers required -/
def minTeachersRequired (c : SchoolConstraints) : Nat :=
  (totalSlotsPerDay c + slotsPerTeacher c - 1) / slotsPerTeacher c

/-- The main theorem stating the minimum number of teachers required -/
theorem min_teachers_is_ten (c : SchoolConstraints) :
  c.teacherCounts = { math := 5, physics := 4, chemistry := 4, biology := 4, computerScience := 3 } →
  c.maxSubjectsPerTeacher = 2 →
  c.periodsPerDay = 6 →
  minTeachersRequired c = 10 := by
  sorry


end NUMINAMATH_CALUDE_min_teachers_is_ten_l3093_309356


namespace NUMINAMATH_CALUDE_riyas_speed_l3093_309362

/-- Proves that Riya's speed is 21 kmph given the problem conditions -/
theorem riyas_speed (riya_speed priya_speed : ℝ) (time : ℝ) (distance : ℝ) : 
  priya_speed = 22 →
  time = 1 →
  distance = 43 →
  distance = (riya_speed + priya_speed) * time →
  riya_speed = 21 := by
  sorry

#check riyas_speed

end NUMINAMATH_CALUDE_riyas_speed_l3093_309362


namespace NUMINAMATH_CALUDE_santinos_fruits_l3093_309375

/-- The number of papaya trees Santino has -/
def papaya_trees : ℕ := 2

/-- The number of mango trees Santino has -/
def mango_trees : ℕ := 3

/-- The number of papayas each papaya tree produces -/
def papayas_per_tree : ℕ := 10

/-- The number of mangos each mango tree produces -/
def mangos_per_tree : ℕ := 20

/-- The total number of fruits Santino has -/
def total_fruits : ℕ := papaya_trees * papayas_per_tree + mango_trees * mangos_per_tree

theorem santinos_fruits : total_fruits = 80 := by
  sorry

end NUMINAMATH_CALUDE_santinos_fruits_l3093_309375


namespace NUMINAMATH_CALUDE_science_club_neither_math_nor_physics_l3093_309355

theorem science_club_neither_math_nor_physics 
  (total : ℕ) 
  (math : ℕ) 
  (physics : ℕ) 
  (both : ℕ) 
  (h1 : total = 100) 
  (h2 : math = 65) 
  (h3 : physics = 43) 
  (h4 : both = 10) : 
  total - (math + physics - both) = 2 :=
by sorry

end NUMINAMATH_CALUDE_science_club_neither_math_nor_physics_l3093_309355


namespace NUMINAMATH_CALUDE_total_time_circling_island_l3093_309357

def time_per_round : ℕ := 30
def saturday_rounds : ℕ := 11
def sunday_rounds : ℕ := 15

theorem total_time_circling_island : 
  time_per_round * (saturday_rounds + sunday_rounds) = 780 := by
  sorry

end NUMINAMATH_CALUDE_total_time_circling_island_l3093_309357


namespace NUMINAMATH_CALUDE_complex_magnitude_problem_l3093_309367

theorem complex_magnitude_problem (z : ℂ) : z = 1 + 2 * I + I ^ 3 → Complex.abs z = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_problem_l3093_309367


namespace NUMINAMATH_CALUDE_dividend_calculation_l3093_309318

theorem dividend_calculation (remainder quotient divisor : ℕ) 
  (h_remainder : remainder = 19)
  (h_quotient : quotient = 61)
  (h_divisor : divisor = 8) :
  divisor * quotient + remainder = 507 :=
by sorry

end NUMINAMATH_CALUDE_dividend_calculation_l3093_309318


namespace NUMINAMATH_CALUDE_quadratic_function_properties_l3093_309331

/-- A quadratic function passing through specific points -/
structure QuadraticFunction where
  f : ℝ → ℝ
  passesA : f 1 = 0
  passesB : f (-3) = 0
  passesC : f 0 = -3

/-- The theorem stating properties of the quadratic function -/
theorem quadratic_function_properties (qf : QuadraticFunction) :
  (∃ a b c : ℝ, ∀ x, qf.f x = a * x^2 + b * x + c) →
  (∀ x, qf.f x = x^2 + 2*x - 3) ∧
  (qf.f (-1) = -4 ∧ ∀ x, qf.f x ≥ qf.f (-1)) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_function_properties_l3093_309331


namespace NUMINAMATH_CALUDE_smallest_positive_multiple_of_45_l3093_309345

theorem smallest_positive_multiple_of_45 :
  ∀ n : ℕ, n > 0 ∧ 45 ∣ n → n ≥ 45 := by
sorry

end NUMINAMATH_CALUDE_smallest_positive_multiple_of_45_l3093_309345


namespace NUMINAMATH_CALUDE_team_size_l3093_309377

theorem team_size (best_score : ℕ) (hypothetical_score : ℕ) (hypothetical_average : ℕ) (total_score : ℕ) :
  best_score = 85 →
  hypothetical_score = 92 →
  hypothetical_average = 84 →
  total_score = 497 →
  ∃ n : ℕ, n = 6 ∧ n * hypothetical_average - (hypothetical_score - best_score) = total_score :=
by sorry

end NUMINAMATH_CALUDE_team_size_l3093_309377


namespace NUMINAMATH_CALUDE_typhoon_tree_difference_l3093_309336

theorem typhoon_tree_difference (initial_trees : ℕ) (dead_trees : ℕ) : 
  initial_trees = 3 → dead_trees = 13 → dead_trees - initial_trees = 10 :=
by sorry

end NUMINAMATH_CALUDE_typhoon_tree_difference_l3093_309336


namespace NUMINAMATH_CALUDE_five_kg_to_g_eight_thousand_g_to_kg_l3093_309396

-- Define the conversion factor
def kg_to_g : ℝ := 1000

-- Theorem for converting 5 kg to grams
theorem five_kg_to_g : 5 * kg_to_g = 5000 := by sorry

-- Theorem for converting 8000 g to kg
theorem eight_thousand_g_to_kg : 8000 / kg_to_g = 8 := by sorry

end NUMINAMATH_CALUDE_five_kg_to_g_eight_thousand_g_to_kg_l3093_309396


namespace NUMINAMATH_CALUDE_math_textbooks_same_box_probability_l3093_309391

/-- The probability of all mathematics textbooks ending up in the same box -/
theorem math_textbooks_same_box_probability :
  let total_books : ℕ := 15
  let math_books : ℕ := 4
  let box1_capacity : ℕ := 4
  let box2_capacity : ℕ := 5
  let box3_capacity : ℕ := 6
  
  -- Total number of ways to distribute books
  let total_distributions : ℕ := (Nat.choose total_books box1_capacity) * 
                                 (Nat.choose (total_books - box1_capacity) box2_capacity) *
                                 (Nat.choose (total_books - box1_capacity - box2_capacity) box3_capacity)
  
  -- Number of ways where all math books are in the same box
  let favorable_outcomes : ℕ := (Nat.choose (total_books - math_books) 0) +
                                (Nat.choose (total_books - math_books) 1) +
                                (Nat.choose (total_books - math_books) 2)
  
  (favorable_outcomes : ℚ) / total_distributions = 67 / 630630 :=
by sorry

end NUMINAMATH_CALUDE_math_textbooks_same_box_probability_l3093_309391


namespace NUMINAMATH_CALUDE_square_adjacent_to_multiple_of_five_l3093_309322

theorem square_adjacent_to_multiple_of_five (n : ℤ) (h : ¬ 5 ∣ n) :
  ∃ k : ℤ, n^2 = 5*k + 1 ∨ n^2 = 5*k - 1 := by
  sorry

end NUMINAMATH_CALUDE_square_adjacent_to_multiple_of_five_l3093_309322


namespace NUMINAMATH_CALUDE_absolute_value_theorem_l3093_309324

theorem absolute_value_theorem (x y : ℝ) (hx : x > 0) :
  |x + 1 - Real.sqrt ((x + y)^2)| = 
    if x + y ≥ 0 then |1 - y| else |2*x + y + 1| := by sorry

end NUMINAMATH_CALUDE_absolute_value_theorem_l3093_309324


namespace NUMINAMATH_CALUDE_paige_dresser_capacity_l3093_309373

/-- Represents the capacity of a dresser in pieces of clothing. -/
def dresser_capacity (pieces_per_drawer : ℕ) (num_drawers : ℕ) : ℕ :=
  pieces_per_drawer * num_drawers

/-- Theorem stating that a dresser with 8 drawers, each holding 5 pieces, has a total capacity of 40 pieces. -/
theorem paige_dresser_capacity :
  dresser_capacity 5 8 = 40 := by
  sorry

end NUMINAMATH_CALUDE_paige_dresser_capacity_l3093_309373


namespace NUMINAMATH_CALUDE_largest_square_area_l3093_309380

theorem largest_square_area (x y z : ℝ) (h1 : x^2 + y^2 = z^2) (h2 : x^2 + y^2 + z^2 = 450) :
  z^2 = 225 := by
  sorry

end NUMINAMATH_CALUDE_largest_square_area_l3093_309380


namespace NUMINAMATH_CALUDE_sum_of_integers_l3093_309360

theorem sum_of_integers (x y : ℕ+) 
  (h1 : x^2 + y^2 = 130)
  (h2 : x * y = 45) : 
  ∃ (ε : ℝ), abs ((x : ℝ) + y - 15) < ε ∧ ε > 0 := by
sorry

end NUMINAMATH_CALUDE_sum_of_integers_l3093_309360


namespace NUMINAMATH_CALUDE_lemon_head_problem_l3093_309304

/-- Given a package size and a total number of Lemon Heads eaten, 
    calculate the number of whole boxes eaten and Lemon Heads left over. -/
def lemonHeadBoxes (packageSize : ℕ) (totalEaten : ℕ) : ℕ × ℕ :=
  (totalEaten / packageSize, totalEaten % packageSize)

/-- Theorem: Given a package size of 6 Lemon Heads and 54 Lemon Heads eaten,
    prove that 9 whole boxes were eaten with 0 Lemon Heads left over. -/
theorem lemon_head_problem : lemonHeadBoxes 6 54 = (9, 0) := by
  sorry

end NUMINAMATH_CALUDE_lemon_head_problem_l3093_309304


namespace NUMINAMATH_CALUDE_total_payment_proof_l3093_309387

def apple_quantity : ℕ := 15
def apple_price : ℕ := 85
def mango_quantity : ℕ := 12
def mango_price : ℕ := 60
def grape_quantity : ℕ := 10
def grape_price : ℕ := 75
def strawberry_quantity : ℕ := 6
def strawberry_price : ℕ := 150

def total_cost : ℕ := 
  apple_quantity * apple_price + 
  mango_quantity * mango_price + 
  grape_quantity * grape_price + 
  strawberry_quantity * strawberry_price

theorem total_payment_proof : total_cost = 3645 := by
  sorry

end NUMINAMATH_CALUDE_total_payment_proof_l3093_309387


namespace NUMINAMATH_CALUDE_unique_product_list_l3093_309372

def letter_value (c : Char) : Nat :=
  c.toNat - 'A'.toNat + 1

def is_valid_list (s : List Char) : Prop :=
  s.length = 4 ∧ s.Nodup ∧ s ≠ ['B', 'E', 'H', 'K']

def product_of_list (s : List Char) : Nat :=
  s.map letter_value |>.prod

theorem unique_product_list : 
  ∀ s : List Char, is_valid_list s → 
    product_of_list s = product_of_list ['B', 'E', 'H', 'K'] → 
    False :=
sorry

end NUMINAMATH_CALUDE_unique_product_list_l3093_309372


namespace NUMINAMATH_CALUDE_sum_of_degrees_l3093_309352

/-- Represents the degrees of four people in a specific ratio -/
structure DegreeRatio :=
  (a b c d : ℕ)
  (ratio : a = 5 ∧ b = 4 ∧ c = 6 ∧ d = 3)

/-- The theorem stating the sum of degrees given the ratio and highest degree -/
theorem sum_of_degrees (r : DegreeRatio) (highest_degree : ℕ) 
  (h : highest_degree = 150) : 
  (r.a + r.b + r.c + r.d) * (highest_degree / r.c) = 450 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_degrees_l3093_309352


namespace NUMINAMATH_CALUDE_parallel_vectors_x_value_l3093_309369

/-- Two vectors are parallel if their cross product is zero -/
def parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

theorem parallel_vectors_x_value :
  ∀ x : ℝ,
  let a : ℝ × ℝ := (2, 1)
  let b : ℝ × ℝ := (x, -2)
  parallel a b → x = -4 := by
sorry

end NUMINAMATH_CALUDE_parallel_vectors_x_value_l3093_309369


namespace NUMINAMATH_CALUDE_equation_solution_l3093_309348

theorem equation_solution :
  let x : ℝ := (173 * 240) / 120
  ∃ ε > 0, ε < 0.005 ∧ |x - 345.33| < ε :=
by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l3093_309348


namespace NUMINAMATH_CALUDE_gcf_of_210_and_294_l3093_309339

theorem gcf_of_210_and_294 : Nat.gcd 210 294 = 42 := by
  sorry

end NUMINAMATH_CALUDE_gcf_of_210_and_294_l3093_309339


namespace NUMINAMATH_CALUDE_percentage_increase_l3093_309349

theorem percentage_increase (N : ℝ) (P : ℝ) : 
  N = 80 →
  N + (P / 100) * N - (N - (25 / 100) * N) = 30 →
  P = 12.5 := by
  sorry

end NUMINAMATH_CALUDE_percentage_increase_l3093_309349


namespace NUMINAMATH_CALUDE_opponents_team_points_l3093_309383

-- Define the points for each player
def max_points : ℕ := 5
def dulce_points : ℕ := 3

-- Define Val's points as twice the combined points of Max and Dulce
def val_points : ℕ := 2 * (max_points + dulce_points)

-- Define the total points of their team
def team_points : ℕ := max_points + dulce_points + val_points

-- Define the point difference between the teams
def point_difference : ℕ := 16

-- Theorem to prove
theorem opponents_team_points : 
  team_points + point_difference = 40 := by sorry

end NUMINAMATH_CALUDE_opponents_team_points_l3093_309383


namespace NUMINAMATH_CALUDE_inequality_proof_l3093_309311

theorem inequality_proof (a b c : ℝ) (h1 : a < b) (h2 : b < 0) : 
  a * (c^2 + 1) < b * (c^2 + 1) := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l3093_309311


namespace NUMINAMATH_CALUDE_skittles_and_erasers_grouping_l3093_309300

theorem skittles_and_erasers_grouping :
  let skittles : ℕ := 4502
  let erasers : ℕ := 4276
  let total_items : ℕ := skittles + erasers
  let num_groups : ℕ := 154
  total_items / num_groups = 57 := by
  sorry

end NUMINAMATH_CALUDE_skittles_and_erasers_grouping_l3093_309300


namespace NUMINAMATH_CALUDE_semiperimeter_equals_diagonal_l3093_309351

/-- A rectangle inscribed in a square --/
structure InscribedRectangle where
  /-- Side length of the square --/
  a : ℝ
  /-- Width of the rectangle --/
  b : ℝ
  /-- Height of the rectangle --/
  c : ℝ
  /-- The rectangle is not a square --/
  not_square : b ≠ c
  /-- The rectangle is inscribed in the square --/
  inscribed : b + c = a * Real.sqrt 2

/-- The semiperimeter of the inscribed rectangle equals the diagonal of the square --/
theorem semiperimeter_equals_diagonal (rect : InscribedRectangle) :
  (rect.b + rect.c) / 2 = rect.a * Real.sqrt 2 / 2 := by
  sorry

#check semiperimeter_equals_diagonal

end NUMINAMATH_CALUDE_semiperimeter_equals_diagonal_l3093_309351


namespace NUMINAMATH_CALUDE_quadratic_roots_counterexample_l3093_309312

theorem quadratic_roots_counterexample : 
  ∃ (a b c : ℝ), b - c > a ∧ a ≠ 0 ∧ 
  ¬(∃ (x y : ℝ), x ≠ y ∧ a * x^2 + b * x + c = 0 ∧ a * y^2 + b * y + c = 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_roots_counterexample_l3093_309312


namespace NUMINAMATH_CALUDE_mean_calculation_l3093_309346

def set1 : List ℝ := [28, 42, 78, 104]
def set2 : List ℝ := [128, 255, 511, 1023]

theorem mean_calculation (x : ℝ) :
  (List.sum set1 + x) / 5 = 90 →
  (List.sum set2 + x) / 5 = 423 := by
  sorry

end NUMINAMATH_CALUDE_mean_calculation_l3093_309346


namespace NUMINAMATH_CALUDE_triangle_properties_l3093_309358

-- Define a triangle ABC
structure Triangle where
  a : ℝ
  b : ℝ
  c : ℝ
  A : ℝ
  B : ℝ
  C : ℝ

-- State the theorem
theorem triangle_properties (t : Triangle) 
  (h : t.a * t.c * Real.sin t.B = t.b^2 - (t.a - t.c)^2) :
  Real.sin t.B = 4/5 ∧ 
  (∀ (x y : ℝ), x > 0 → y > 0 → x^2 / (x^2 + y^2) ≥ 2/5) ∧
  (∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ x^2 / (x^2 + y^2) = 2/5) := by
  sorry

end NUMINAMATH_CALUDE_triangle_properties_l3093_309358


namespace NUMINAMATH_CALUDE_race_probability_l3093_309314

theorem race_probability (total_cars : ℕ) (prob_x prob_y prob_z : ℚ) 
  (h_total : total_cars = 10)
  (h_x : prob_x = 1/7)
  (h_y : prob_y = 1/3)
  (h_z : prob_z = 1/5)
  (h_no_tie : ∀ a b : ℕ, a ≠ b → a ≤ total_cars → b ≤ total_cars → 
    (prob_x + prob_y + prob_z ≤ 1)) :
  prob_x + prob_y + prob_z = 71/105 := by
sorry

end NUMINAMATH_CALUDE_race_probability_l3093_309314


namespace NUMINAMATH_CALUDE_continuous_stripe_probability_l3093_309338

/-- Represents the type of stripe on a cube face -/
inductive StripeType
| Solid
| Dashed

/-- Represents the orientation of a stripe on a cube face -/
inductive StripeOrientation
| Horizontal
| Vertical

/-- Represents a single face configuration -/
structure FaceConfig where
  stripeType : StripeType
  orientation : StripeOrientation

/-- Represents a complete cube configuration -/
structure CubeConfig where
  faces : Fin 6 → FaceConfig

/-- Determines if a given cube configuration has a continuous stripe -/
def hasContinuousStripe (config : CubeConfig) : Bool := sorry

/-- The total number of possible cube configurations -/
def totalConfigurations : Nat := 4^6

/-- The number of configurations with a continuous stripe -/
def continuousStripeConfigurations : Nat := 3 * 16

/-- The probability of a continuous stripe encircling the cube -/
theorem continuous_stripe_probability :
  (continuousStripeConfigurations : ℚ) / totalConfigurations = 3 / 256 := by sorry

end NUMINAMATH_CALUDE_continuous_stripe_probability_l3093_309338


namespace NUMINAMATH_CALUDE_exists_in_set_A_l3093_309315

/-- Sum of digits of a positive integer -/
def digit_sum (n : ℕ+) : ℕ := sorry

/-- Predicate for a number having all non-zero digits -/
def all_digits_nonzero (n : ℕ+) : Prop := sorry

/-- Number of digits in a positive integer -/
def num_digits (n : ℕ+) : ℕ := sorry

/-- The main theorem -/
theorem exists_in_set_A (k : ℕ+) : 
  ∃ x : ℕ+, num_digits x = k ∧ all_digits_nonzero x ∧ (digit_sum x ∣ x) := by
  sorry

end NUMINAMATH_CALUDE_exists_in_set_A_l3093_309315


namespace NUMINAMATH_CALUDE_greatest_integer_b_for_quadratic_range_l3093_309389

theorem greatest_integer_b_for_quadratic_range : 
  (∃ b : ℤ, (∀ x : ℝ, x^2 + b*x + 15 ≠ -9) ∧ 
            (∀ b' : ℤ, b' > b → ∃ x : ℝ, x^2 + b'*x + 15 = -9)) ∧ 
  (∀ b : ℤ, (∀ x : ℝ, x^2 + b*x + 15 ≠ -9) → b ≤ 9) := by
  sorry

end NUMINAMATH_CALUDE_greatest_integer_b_for_quadratic_range_l3093_309389


namespace NUMINAMATH_CALUDE_rhombus_side_length_l3093_309361

-- Define a rhombus with area K and diagonals d and 3d
structure Rhombus where
  K : ℝ  -- Area of the rhombus
  d : ℝ  -- Length of the shorter diagonal
  h : K = (3/2) * d^2  -- Area formula for rhombus

-- Theorem: The side length of the rhombus is sqrt(5K/3)
theorem rhombus_side_length (r : Rhombus) : 
  ∃ s : ℝ, s^2 = (5/3) * r.K ∧ s > 0 := by
  sorry

end NUMINAMATH_CALUDE_rhombus_side_length_l3093_309361


namespace NUMINAMATH_CALUDE_picnic_theorem_l3093_309390

def picnic_problem (people : ℕ) (sandwich_price : ℚ) (fruit_salad_price : ℚ) (soda_price : ℚ) (sodas_per_person : ℕ) (snack_bags : ℕ) (total_spent : ℚ) : Prop :=
  let sandwich_cost := people * sandwich_price
  let fruit_salad_cost := people * fruit_salad_price
  let soda_cost := people * sodas_per_person * soda_price
  let food_cost := sandwich_cost + fruit_salad_cost + soda_cost
  let snack_cost := total_spent - food_cost
  snack_cost / snack_bags = 4

theorem picnic_theorem : 
  picnic_problem 4 5 3 2 2 3 60 := by
  sorry

end NUMINAMATH_CALUDE_picnic_theorem_l3093_309390


namespace NUMINAMATH_CALUDE_complement_of_M_in_U_l3093_309329

def U : Set ℕ := {1, 2, 3, 4}

def M : Set ℕ := {x | (x - 1) * (x - 4) = 0}

theorem complement_of_M_in_U : U \ M = {2, 3} := by sorry

end NUMINAMATH_CALUDE_complement_of_M_in_U_l3093_309329


namespace NUMINAMATH_CALUDE_banana_difference_l3093_309398

theorem banana_difference (total : ℕ) (lydia_bananas : ℕ) (donna_bananas : ℕ)
  (h1 : total = 200)
  (h2 : lydia_bananas = 60)
  (h3 : donna_bananas = 40) :
  total - donna_bananas - lydia_bananas - lydia_bananas = 40 := by
  sorry

end NUMINAMATH_CALUDE_banana_difference_l3093_309398


namespace NUMINAMATH_CALUDE_inverse_64_mod_97_l3093_309340

theorem inverse_64_mod_97 (h : (8⁻¹ : ZMod 97) = 85) : (64⁻¹ : ZMod 97) = 47 := by
  sorry

end NUMINAMATH_CALUDE_inverse_64_mod_97_l3093_309340


namespace NUMINAMATH_CALUDE_camp_wonka_ratio_l3093_309344

theorem camp_wonka_ratio : 
  ∀ (total_campers : ℕ) (boys : ℕ),
    total_campers = 96 →
    boys = (2 * total_campers) / 3 →
    (total_campers - boys) * 3 = total_campers :=
by
  sorry

end NUMINAMATH_CALUDE_camp_wonka_ratio_l3093_309344


namespace NUMINAMATH_CALUDE_sum_of_coefficients_l3093_309327

def f (m n : ℕ) : ℕ := Nat.choose 6 m * Nat.choose 4 n

theorem sum_of_coefficients : f 3 0 + f 2 1 + f 1 2 + f 0 3 = 120 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_coefficients_l3093_309327


namespace NUMINAMATH_CALUDE_divisibility_implication_l3093_309310

theorem divisibility_implication (a b : ℕ+) :
  (∀ n : ℕ, a^n ∣ b^(n+1)) → a ∣ b := by sorry

end NUMINAMATH_CALUDE_divisibility_implication_l3093_309310


namespace NUMINAMATH_CALUDE_money_distribution_l3093_309382

/-- Represents the distribution of money among x, y, and z -/
structure Distribution where
  x : ℚ  -- Amount x gets in rupees
  y : ℚ  -- Amount y gets in rupees
  z : ℚ  -- Amount z gets in rupees

/-- The problem statement and conditions -/
theorem money_distribution (d : Distribution) : 
  -- For each rupee x gets, z gets 30 paisa
  d.z = 0.3 * d.x →
  -- The share of y is Rs. 27
  d.y = 27 →
  -- The total amount is Rs. 105
  d.x + d.y + d.z = 105 →
  -- Prove that y gets 45 paisa for each rupee x gets
  d.y / d.x = 0.45 := by
sorry

end NUMINAMATH_CALUDE_money_distribution_l3093_309382


namespace NUMINAMATH_CALUDE_jerry_hawk_feathers_l3093_309394

/-- The number of hawk feathers Jerry found -/
def hawk_feathers : ℕ := 6

/-- The number of eagle feathers Jerry found -/
def eagle_feathers : ℕ := 17 * hawk_feathers

/-- The total number of feathers Jerry initially had -/
def total_feathers : ℕ := hawk_feathers + eagle_feathers

/-- The number of feathers Jerry had after giving 10 to his sister -/
def feathers_after_giving : ℕ := total_feathers - 10

/-- The number of feathers Jerry had after selling half of the remaining feathers -/
def feathers_after_selling : ℕ := feathers_after_giving / 2

theorem jerry_hawk_feathers :
  hawk_feathers = 6 ∧
  eagle_feathers = 17 * hawk_feathers ∧
  total_feathers = hawk_feathers + eagle_feathers ∧
  feathers_after_giving = total_feathers - 10 ∧
  feathers_after_selling = feathers_after_giving / 2 ∧
  feathers_after_selling = 49 :=
by sorry

end NUMINAMATH_CALUDE_jerry_hawk_feathers_l3093_309394


namespace NUMINAMATH_CALUDE_sum_of_digits_problem_l3093_309328

def S (n : ℕ) : ℕ := sorry  -- Sum of digits function

theorem sum_of_digits_problem (N : ℕ) 
  (h1 : S N + S (N + 1) = 200)
  (h2 : S (N + 2) + S (N + 3) = 105) :
  S (N + 1) + S (N + 2) = 103 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_digits_problem_l3093_309328


namespace NUMINAMATH_CALUDE_survey_respondents_l3093_309359

/-- The number of people who preferred brand X -/
def X : ℕ := 60

/-- The number of people who preferred brand Y -/
def Y : ℕ := X / 3

/-- The number of people who preferred brand Z -/
def Z : ℕ := X * 3 / 2

/-- The total number of respondents to the survey -/
def total_respondents : ℕ := X + Y + Z

/-- Theorem stating that the total number of respondents is 170 -/
theorem survey_respondents : total_respondents = 170 := by
  sorry

end NUMINAMATH_CALUDE_survey_respondents_l3093_309359


namespace NUMINAMATH_CALUDE_catchup_distance_proof_l3093_309379

/-- The speed of walker A in km/h -/
def speed_A : ℝ := 10

/-- The speed of cyclist B in km/h -/
def speed_B : ℝ := 20

/-- The time difference between A's start and B's start in hours -/
def time_difference : ℝ := 7

/-- The distance at which B catches up with A in km -/
def catchup_distance : ℝ := 140

theorem catchup_distance_proof :
  speed_A * time_difference +
  speed_A * (catchup_distance / speed_B) =
  catchup_distance :=
sorry

end NUMINAMATH_CALUDE_catchup_distance_proof_l3093_309379


namespace NUMINAMATH_CALUDE_water_remaining_after_four_replacements_l3093_309397

/-- Represents the fraction of original water remaining after a number of replacements -/
def water_remaining (initial_water : ℚ) (tank_capacity : ℚ) (replacement_volume : ℚ) (n : ℕ) : ℚ :=
  (1 - replacement_volume / tank_capacity) ^ n * initial_water / tank_capacity

/-- Theorem stating the fraction of original water remaining after 4 replacements -/
theorem water_remaining_after_four_replacements : 
  water_remaining 10 20 5 4 = 81 / 256 := by
  sorry

end NUMINAMATH_CALUDE_water_remaining_after_four_replacements_l3093_309397


namespace NUMINAMATH_CALUDE_tank_capacity_l3093_309364

/-- Represents a cylindrical water tank -/
structure WaterTank where
  capacity : ℝ
  currentVolume : ℝ
  fillPercentage : ℝ

/-- The tank contains 120 liters when it is 24% full -/
def partiallyFilledTank : WaterTank :=
  { capacity := 500,
    currentVolume := 120,
    fillPercentage := 0.24 }

/-- Theorem stating that the tank's capacity is 500 liters -/
theorem tank_capacity :
  partiallyFilledTank.capacity = 500 ∧
  partiallyFilledTank.currentVolume = 120 ∧
  partiallyFilledTank.fillPercentage = 0.24 ∧
  partiallyFilledTank.currentVolume = partiallyFilledTank.capacity * partiallyFilledTank.fillPercentage :=
by
  sorry

#check tank_capacity

end NUMINAMATH_CALUDE_tank_capacity_l3093_309364


namespace NUMINAMATH_CALUDE_parabola_equation_dot_product_focus_fixed_point_l3093_309341

-- Define the parabola
def Parabola := {p : ℝ × ℝ | p.2^2 = 4 * p.1}

-- Define the focus of the parabola
def focus : ℝ × ℝ := (1, 0)

-- Define the dot product of two points
def dot_product (p q : ℝ × ℝ) : ℝ := p.1 * q.1 + p.2 * q.2

-- Theorem 1: Standard equation of the parabola
theorem parabola_equation (p : ℝ × ℝ) : p ∈ Parabola ↔ p.2^2 = 4 * p.1 := by sorry

-- Theorem 2: Dot product of OA and OB when line passes through focus
theorem dot_product_focus (A B : ℝ × ℝ) (hA : A ∈ Parabola) (hB : B ∈ Parabola) 
  (h_line : ∃ (m : ℝ), A.2 = m * (A.1 - 1) ∧ B.2 = m * (B.1 - 1) ∧ focus.2 = m * (focus.1 - 1)) :
  dot_product A B = -3 := by sorry

-- Theorem 3: Fixed point when dot product is -4
theorem fixed_point (A B : ℝ × ℝ) (hA : A ∈ Parabola) (hB : B ∈ Parabola) 
  (h_dot : dot_product A B = -4) :
  ∃ (m : ℝ), A.2 = m * (A.1 - 2) ∧ B.2 = m * (B.1 - 2) := by sorry

end NUMINAMATH_CALUDE_parabola_equation_dot_product_focus_fixed_point_l3093_309341


namespace NUMINAMATH_CALUDE_athlete_A_most_stable_l3093_309337

/-- Represents an athlete with their performance variance -/
structure Athlete where
  name : String
  variance : Float

/-- Determines if an athlete has the most stable performance among a list of athletes -/
def hasMostStablePerformance (a : Athlete) (athletes : List Athlete) : Prop :=
  ∀ b ∈ athletes, a.variance ≤ b.variance

/-- The list of athletes with their variances -/
def athleteList : List Athlete :=
  [⟨"A", 0.019⟩, ⟨"B", 0.021⟩, ⟨"C", 0.020⟩, ⟨"D", 0.022⟩]

theorem athlete_A_most_stable :
  ∃ a ∈ athleteList, a.name = "A" ∧ hasMostStablePerformance a athleteList := by
  sorry


end NUMINAMATH_CALUDE_athlete_A_most_stable_l3093_309337


namespace NUMINAMATH_CALUDE_arithmetic_sequence_problem_l3093_309333

/-- 
Given an arithmetic sequence where the third term is 3 and the eleventh term is 15,
prove that the first term is 0 and the common difference is 3/2.
-/
theorem arithmetic_sequence_problem (a : ℕ → ℚ) 
  (h1 : a 3 = 3) 
  (h2 : a 11 = 15) 
  (h3 : ∀ n : ℕ, a (n + 1) - a n = a 2 - a 1) : 
  a 1 = 0 ∧ a 2 - a 1 = 3/2 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_problem_l3093_309333


namespace NUMINAMATH_CALUDE_system_solution_negative_implies_m_range_l3093_309386

theorem system_solution_negative_implies_m_range (m x y : ℝ) : 
  x - y = 2 * m + 7 →
  x + y = 4 * m - 3 →
  x < 0 →
  y < 0 →
  m < -2/3 := by
sorry

end NUMINAMATH_CALUDE_system_solution_negative_implies_m_range_l3093_309386


namespace NUMINAMATH_CALUDE_quadratic_factor_implies_n_l3093_309371

theorem quadratic_factor_implies_n (n : ℤ) : 
  (∃ k : ℤ, ∀ x : ℤ, x^2 + 7*x + n = (x + 5) * (x + k)) → n = 10 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_factor_implies_n_l3093_309371


namespace NUMINAMATH_CALUDE_class_size_calculation_l3093_309303

theorem class_size_calculation (female_students : ℕ) (male_students : ℕ) : 
  female_students = 13 → 
  male_students = 3 * female_students → 
  female_students + male_students = 52 := by
sorry

end NUMINAMATH_CALUDE_class_size_calculation_l3093_309303


namespace NUMINAMATH_CALUDE_cube_root_equation_solution_l3093_309385

theorem cube_root_equation_solution :
  ∃! x : ℝ, (5 - x / 2) ^ (1/3 : ℝ) = 2 :=
by
  sorry

end NUMINAMATH_CALUDE_cube_root_equation_solution_l3093_309385


namespace NUMINAMATH_CALUDE_equal_temperament_sequence_l3093_309320

theorem equal_temperament_sequence (a : ℕ → ℝ) :
  (∀ n, 1 ≤ n → n ≤ 13 → a n > 0) →
  (∀ n, 1 ≤ n → n < 13 → a (n + 1) / a n = a 2 / a 1) →
  a 1 = 1 →
  a 13 = 2 →
  a 3 = 2^(1/6) :=
by sorry

end NUMINAMATH_CALUDE_equal_temperament_sequence_l3093_309320


namespace NUMINAMATH_CALUDE_triplet_sum_to_two_l3093_309334

theorem triplet_sum_to_two :
  -- Triplet A
  (1/4 : ℚ) + (1/4 : ℚ) + (3/2 : ℚ) = 2 ∧
  -- Triplet B
  (3 : ℤ) + (-1 : ℤ) + (0 : ℤ) = 2 ∧
  -- Triplet C
  (0.2 : ℝ) + (0.7 : ℝ) + (1.1 : ℝ) = 2 ∧
  -- Triplet D
  (2.2 : ℝ) + (-0.5 : ℝ) + (0.5 : ℝ) ≠ 2 ∧
  -- Triplet E
  (3/5 : ℚ) + (4/5 : ℚ) + (1/5 : ℚ) ≠ 2 := by
  sorry

end NUMINAMATH_CALUDE_triplet_sum_to_two_l3093_309334


namespace NUMINAMATH_CALUDE_gcd_circle_impossibility_l3093_309388

theorem gcd_circle_impossibility (numbers : Fin 49 → ℕ) :
  (∀ i j, i ≠ j → numbers i ≠ numbers j) →  -- All numbers are distinct
  (∀ i, numbers i < 100) →                  -- All numbers are less than 100
  ¬(∀ i j, i ≠ j → 
    Nat.gcd (numbers i) (numbers ((i + 1) % 49)) ≠ 
    Nat.gcd (numbers j) (numbers ((j + 1) % 49))) := 
by sorry

end NUMINAMATH_CALUDE_gcd_circle_impossibility_l3093_309388


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l3093_309325

theorem sqrt_equation_solution (x : ℝ) :
  (Real.sqrt (5 * x - 4) + 15 / Real.sqrt (5 * x - 4) = 8) ↔ (x = 29 / 5 ∨ x = 13 / 5) := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l3093_309325


namespace NUMINAMATH_CALUDE_root_condition_implies_m_range_l3093_309354

theorem root_condition_implies_m_range (m : ℝ) :
  (∀ x : ℝ, (m / (2 * x - 4) = (1 - x) / (2 - x) - 2) → x > 0) →
  m < 6 ∧ m ≠ 2 :=
by sorry

end NUMINAMATH_CALUDE_root_condition_implies_m_range_l3093_309354


namespace NUMINAMATH_CALUDE_min_value_of_2a_plus_3b_l3093_309309

theorem min_value_of_2a_plus_3b (a b : ℝ) (ha : a > 0) (hb : b > 0)
  (h : (8 / ((b + 1)^3)) + (6 / (b + 1)) ≤ a^3 + 3*a) :
  ∀ x y, x > 0 → y > 0 → (8 / ((y + 1)^3)) + (6 / (y + 1)) ≤ x^3 + 3*x →
  2*a + 3*b ≤ 2*x + 3*y →
  2*a + 3*b = 4 * Real.sqrt 3 - 3 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_2a_plus_3b_l3093_309309


namespace NUMINAMATH_CALUDE_sale_price_calculation_l3093_309342

theorem sale_price_calculation (ticket_price : ℝ) (discount_percentage : ℝ) 
  (h1 : ticket_price = 25)
  (h2 : discount_percentage = 25) :
  ticket_price * (1 - discount_percentage / 100) = 18.75 := by
  sorry

end NUMINAMATH_CALUDE_sale_price_calculation_l3093_309342


namespace NUMINAMATH_CALUDE_collinear_points_x_value_l3093_309368

/-- Given three points A, B, and C in 2D space, 
    returns true if they are collinear -/
def collinear (A B C : ℝ × ℝ) : Prop :=
  (B.2 - A.2) * (C.1 - A.1) = (C.2 - A.2) * (B.1 - A.1)

/-- The main theorem: if A(-1,-2), B(4,8), and C(5,x) are collinear, 
    then x = 10 -/
theorem collinear_points_x_value :
  collinear (-1, -2) (4, 8) (5, x) → x = 10 := by
  sorry

end NUMINAMATH_CALUDE_collinear_points_x_value_l3093_309368


namespace NUMINAMATH_CALUDE_plums_added_l3093_309317

theorem plums_added (initial : ℕ) (final : ℕ) (h1 : initial = 17) (h2 : final = 21) :
  final - initial = 4 := by
  sorry

end NUMINAMATH_CALUDE_plums_added_l3093_309317


namespace NUMINAMATH_CALUDE_christine_and_siri_money_l3093_309365

theorem christine_and_siri_money (christine_money siri_money : ℚ) : 
  christine_money = 20.5 → 
  christine_money = siri_money + 20 → 
  christine_money + siri_money = 21 := by
sorry

end NUMINAMATH_CALUDE_christine_and_siri_money_l3093_309365


namespace NUMINAMATH_CALUDE_complex_equation_solution_l3093_309323

theorem complex_equation_solution (z : ℂ) : (1 - Complex.I) * z = 2 * Complex.I → z = -1 + Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l3093_309323


namespace NUMINAMATH_CALUDE_expected_cereal_difference_l3093_309302

/-- Represents the outcome of rolling a fair six-sided die -/
inductive DieRoll
  | one
  | two
  | three
  | four
  | five
  | six

/-- Represents the type of cereal Bob eats based on his die roll -/
inductive CerealType
  | sweetened
  | unsweetened
  | healthy

/-- Maps a die roll to the corresponding cereal type -/
def rollToCereal (roll : DieRoll) : CerealType :=
  match roll with
  | DieRoll.one => CerealType.healthy
  | DieRoll.two => CerealType.unsweetened
  | DieRoll.three => CerealType.unsweetened
  | DieRoll.four => CerealType.sweetened
  | DieRoll.five => CerealType.unsweetened
  | DieRoll.six => CerealType.sweetened

/-- The number of days in a non-leap year -/
def daysInYear : ℕ := 365

/-- The probability of rolling any specific number on a fair six-sided die -/
def probSingle : ℚ := 1 / 6

/-- Theorem stating the expected difference between unsweetened and sweetened cereal days -/
theorem expected_cereal_difference :
  ∃ (diff : ℚ), abs (diff - 60.83) < 0.01 ∧
  diff = daysInYear * (3 * probSingle - 2 * probSingle) := by
  sorry

end NUMINAMATH_CALUDE_expected_cereal_difference_l3093_309302


namespace NUMINAMATH_CALUDE_find_y_l3093_309366

theorem find_y (x z : ℤ) (y : ℚ) 
  (h1 : x = -2) 
  (h2 : z = 4) 
  (h3 : x^2 * y * z - x * y * z^2 = 48) : 
  y = 1 := by
  sorry

end NUMINAMATH_CALUDE_find_y_l3093_309366


namespace NUMINAMATH_CALUDE_largest_t_value_for_60_degrees_l3093_309376

-- Define the temperature function
def temperature (t : ℝ) : ℝ := -t^2 + 12*t + 50

-- Define the theorem
theorem largest_t_value_for_60_degrees :
  let t := 6 + Real.sqrt 46
  (∀ s ≥ 0, temperature s = 60 → s ≤ t) ∧ temperature t = 60 := by
  sorry

end NUMINAMATH_CALUDE_largest_t_value_for_60_degrees_l3093_309376


namespace NUMINAMATH_CALUDE_min_xy_l3093_309330

theorem min_xy (x y : ℝ) (hx : x > 0) (hy : y > 0) (h : x * y = 2 * x + 8 * y) :
  (∀ x' y' : ℝ, x' > 0 → y' > 0 → x' * y' = 2 * x' + 8 * y' → x * y ≤ x' * y') →
  x = 16 ∧ y = 4 := by
sorry

end NUMINAMATH_CALUDE_min_xy_l3093_309330


namespace NUMINAMATH_CALUDE_range_of_m_l3093_309378

-- Define the polynomials P and Q
def P (x : ℝ) : Prop := x^2 - 8*x - 20 ≤ 0
def Q (x m : ℝ) : Prop := x^2 - 2*x + 1 - m^2 ≤ 0

-- Define the negations of P and Q
def not_P (x : ℝ) : Prop := ¬(P x)
def not_Q (x m : ℝ) : Prop := ¬(Q x m)

-- State the theorem
theorem range_of_m :
  ∀ m : ℝ, 
    (m > 0) →
    (∀ x : ℝ, not_P x → not_Q x m) →
    (∃ x : ℝ, not_Q x m ∧ ¬(not_P x)) →
    (0 < m ∧ m ≤ 3) :=
by sorry

end NUMINAMATH_CALUDE_range_of_m_l3093_309378


namespace NUMINAMATH_CALUDE_no_valid_x_l3093_309305

theorem no_valid_x : ¬ ∃ (x : ℕ+), 
  (x : ℝ) - 7 > 0 ∧ 
  (x + 5) * (x - 7) * (x^2 + x + 30) < 800 :=
sorry

end NUMINAMATH_CALUDE_no_valid_x_l3093_309305


namespace NUMINAMATH_CALUDE_vector_addition_and_scalar_multiplication_l3093_309343

theorem vector_addition_and_scalar_multiplication :
  let v1 : Fin 2 → ℝ := ![3, -8]
  let v2 : Fin 2 → ℝ := ![2, -6]
  let scalar : ℝ := 5
  v1 + scalar • v2 = ![13, -38] := by sorry

end NUMINAMATH_CALUDE_vector_addition_and_scalar_multiplication_l3093_309343


namespace NUMINAMATH_CALUDE_linear_function_proof_l3093_309384

theorem linear_function_proof (f : ℝ → ℝ) 
  (h1 : ∀ x y : ℝ, f (x + y) = f x + f y) -- linearity
  (h2 : ∀ x y : ℝ, x < y → f x < f y) -- monotonically increasing
  (h3 : ∀ x : ℝ, f (f x) = 16 * x + 9) : -- given condition
  ∀ x : ℝ, f x = 4 * x + 9/5 := by
sorry

end NUMINAMATH_CALUDE_linear_function_proof_l3093_309384


namespace NUMINAMATH_CALUDE_binary_110101_to_base7_l3093_309319

def binary_to_decimal (b : List Bool) : ℕ :=
  b.enum.foldl (λ acc (i, bit) => acc + if bit then 2^i else 0) 0

def decimal_to_base7 (n : ℕ) : List ℕ :=
  if n = 0 then [0] else
    let rec aux (m : ℕ) (acc : List ℕ) :=
      if m = 0 then acc else aux (m / 7) ((m % 7) :: acc)
    aux n []

theorem binary_110101_to_base7 :
  decimal_to_base7 (binary_to_decimal [true, false, true, false, true, true]) = [1, 0, 4] :=
sorry

end NUMINAMATH_CALUDE_binary_110101_to_base7_l3093_309319


namespace NUMINAMATH_CALUDE_banana_orange_equivalence_l3093_309395

/-- Given that 3/5 of 15 bananas are worth as much as 12 oranges,
    prove that 2/3 of 9 bananas are worth as much as 8 oranges. -/
theorem banana_orange_equivalence :
  ∀ (banana_value orange_value : ℚ),
  (3 / 5 : ℚ) * 15 * banana_value = 12 * orange_value →
  (2 / 3 : ℚ) * 9 * banana_value = 8 * orange_value :=
by sorry

end NUMINAMATH_CALUDE_banana_orange_equivalence_l3093_309395


namespace NUMINAMATH_CALUDE_other_factors_of_twenty_l3093_309316

theorem other_factors_of_twenty (y : ℕ) : 
  y = 20 ∧ y % 5 = 0 ∧ y % 8 ≠ 0 → 
  (∀ x : ℕ, x ≠ 1 ∧ x ≠ 5 ∧ y % x = 0 → x = 2 ∨ x = 4 ∨ x = 10) :=
by sorry

end NUMINAMATH_CALUDE_other_factors_of_twenty_l3093_309316


namespace NUMINAMATH_CALUDE_student_meeting_distance_l3093_309350

theorem student_meeting_distance (initial_distance : ℝ) (time : ℝ) (speed1 : ℝ) (speed2 : ℝ) :
  initial_distance = 350 →
  time = 100 →
  speed1 = 1.6 →
  speed2 = 1.9 →
  speed2 * time = 190 :=
by sorry

end NUMINAMATH_CALUDE_student_meeting_distance_l3093_309350


namespace NUMINAMATH_CALUDE_square_minus_product_equals_one_l3093_309392

theorem square_minus_product_equals_one : 2015^2 - 2016 * 2014 = 1 := by
  sorry

end NUMINAMATH_CALUDE_square_minus_product_equals_one_l3093_309392


namespace NUMINAMATH_CALUDE_union_of_M_and_N_l3093_309308

def M : Set ℝ := {x | x^2 - 4*x + 3 < 0}
def N : Set ℝ := {x | 2*x + 1 < 5}

theorem union_of_M_and_N : M ∪ N = {x : ℝ | x < 3} := by
  sorry

end NUMINAMATH_CALUDE_union_of_M_and_N_l3093_309308
