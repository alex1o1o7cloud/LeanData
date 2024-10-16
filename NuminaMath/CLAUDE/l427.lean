import Mathlib

namespace NUMINAMATH_CALUDE_problem_solution_l427_42791

theorem problem_solution (x y z : ℝ) 
  (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h1 : x^2 / y = 2)
  (h2 : y^2 / z = 3)
  (h3 : z^2 / x = 4) :
  x = 576^(1/7) := by
sorry

end NUMINAMATH_CALUDE_problem_solution_l427_42791


namespace NUMINAMATH_CALUDE_orange_count_l427_42711

theorem orange_count (b t o : ℕ) : 
  (b + t) / 2 = 89 →
  (b + t + o) / 3 = 91 →
  o = 95 := by
sorry

end NUMINAMATH_CALUDE_orange_count_l427_42711


namespace NUMINAMATH_CALUDE_range_of_a_l427_42782

theorem range_of_a (a : ℝ) : 
  (∀ x : ℝ, a^2 - 3*a ≤ |x + 3| + |x - 1|) → 
  -1 < a ∧ a < 4 :=
by sorry

end NUMINAMATH_CALUDE_range_of_a_l427_42782


namespace NUMINAMATH_CALUDE_parabola_transformation_l427_42764

def original_parabola (x : ℝ) : ℝ := 3 * x^2

def transformed_parabola (x : ℝ) : ℝ := 3 * (x - 3)^2 - 1

theorem parabola_transformation :
  ∀ x : ℝ, transformed_parabola x = original_parabola (x - 3) - 1 :=
by sorry

end NUMINAMATH_CALUDE_parabola_transformation_l427_42764


namespace NUMINAMATH_CALUDE_circles_intersect_l427_42775

-- Define the circles
def circle1 (x y : ℝ) : Prop := (x + 2)^2 + y^2 = 4
def circle2 (x y : ℝ) : Prop := (x - 2)^2 + (y - 1)^2 = 9

-- Define the centers and radii
def center1 : ℝ × ℝ := (-2, 0)
def center2 : ℝ × ℝ := (2, 1)
def radius1 : ℝ := 2
def radius2 : ℝ := 3

-- Theorem stating that the circles are intersecting
theorem circles_intersect :
  let d := Real.sqrt ((center1.1 - center2.1)^2 + (center1.2 - center2.2)^2)
  radius2 + radius1 > d ∧ d > radius2 - radius1 := by sorry

end NUMINAMATH_CALUDE_circles_intersect_l427_42775


namespace NUMINAMATH_CALUDE_father_son_age_difference_l427_42762

/-- Represents the ages of a father and son pair -/
structure FatherSonAges where
  father : ℕ
  son : ℕ

/-- The conditions of the problem -/
def satisfiesConditions (ages : FatherSonAges) : Prop :=
  ages.father = 44 ∧
  (ages.father + 4 = 2 * (ages.son + 4) + 20)

/-- The theorem to be proved -/
theorem father_son_age_difference (ages : FatherSonAges) 
  (h : satisfiesConditions ages) : 
  ages.father - 4 * ages.son = 4 := by
  sorry

end NUMINAMATH_CALUDE_father_son_age_difference_l427_42762


namespace NUMINAMATH_CALUDE_smartphone_price_l427_42710

theorem smartphone_price :
  ∀ (S : ℝ),
  (∃ (PC Tablet : ℝ),
    PC = S + 500 ∧
    Tablet = S + (S + 500) ∧
    S + PC + Tablet = 2200) →
  S = 300 := by
sorry

end NUMINAMATH_CALUDE_smartphone_price_l427_42710


namespace NUMINAMATH_CALUDE_machine_purchase_price_l427_42717

/-- Proves that given the specified conditions, the original purchase price of the machine was Rs 9000 -/
theorem machine_purchase_price 
  (repair_cost : ℕ) 
  (transport_cost : ℕ) 
  (profit_percentage : ℚ) 
  (selling_price : ℕ) 
  (h1 : repair_cost = 5000)
  (h2 : transport_cost = 1000)
  (h3 : profit_percentage = 50 / 100)
  (h4 : selling_price = 22500) :
  ∃ (purchase_price : ℕ), 
    selling_price = (1 + profit_percentage) * (purchase_price + repair_cost + transport_cost) ∧
    purchase_price = 9000 := by
  sorry


end NUMINAMATH_CALUDE_machine_purchase_price_l427_42717


namespace NUMINAMATH_CALUDE_rational_roots_condition_l427_42700

theorem rational_roots_condition (p : ℤ) : 
  (∃ x : ℚ, 4 * x^4 + 4 * p * x^3 = (p - 4) * x^2 - 4 * p * x + p) ↔ (p = 0 ∨ p = -1) :=
by sorry

end NUMINAMATH_CALUDE_rational_roots_condition_l427_42700


namespace NUMINAMATH_CALUDE_max_candies_l427_42780

theorem max_candies (vitya maria sasha : ℕ) : 
  vitya = 35 →
  maria < vitya →
  sasha = vitya + maria →
  Even sasha →
  vitya + maria + sasha ≤ 136 :=
by sorry

end NUMINAMATH_CALUDE_max_candies_l427_42780


namespace NUMINAMATH_CALUDE_optimal_rectangle_dimensions_l427_42755

theorem optimal_rectangle_dimensions :
  ∀ w l : ℝ,
  w > 0 →
  l = 2 * w →
  w * l ≥ 800 →
  ∀ w' l' : ℝ,
  w' > 0 →
  l' = 2 * w' →
  w' * l' ≥ 800 →
  2 * w + 2 * l ≤ 2 * w' + 2 * l' →
  w = 20 ∧ l = 40 := by
sorry

end NUMINAMATH_CALUDE_optimal_rectangle_dimensions_l427_42755


namespace NUMINAMATH_CALUDE_f_2_eq_1_l427_42778

/-- The function f(x) = x^5 - 5x^4 + 10x^3 - 10x^2 + 5x - 1 -/
def f (x : ℝ) : ℝ := x^5 - 5*x^4 + 10*x^3 - 10*x^2 + 5*x - 1

/-- Theorem: f(2) = 1 -/
theorem f_2_eq_1 : f 2 = 1 := by
  sorry

end NUMINAMATH_CALUDE_f_2_eq_1_l427_42778


namespace NUMINAMATH_CALUDE_dawn_monthly_payments_l427_42792

/-- Dawn's annual salary in dollars -/
def annual_salary : ℕ := 48000

/-- Dawn's monthly savings rate as a fraction -/
def savings_rate : ℚ := 1/10

/-- Dawn's monthly savings in dollars -/
def monthly_savings : ℕ := 400

/-- The number of months in a year -/
def months_in_year : ℕ := 12

theorem dawn_monthly_payments :
  (annual_salary / months_in_year : ℚ) * savings_rate = monthly_savings :=
sorry

end NUMINAMATH_CALUDE_dawn_monthly_payments_l427_42792


namespace NUMINAMATH_CALUDE_prob_three_odd_dice_l427_42772

/-- The number of sides on each die -/
def numSides : ℕ := 8

/-- The number of dice rolled -/
def numDice : ℕ := 5

/-- The number of odd numbers on each die -/
def numOddSides : ℕ := 4

/-- The number of dice that should show an odd number -/
def targetOddDice : ℕ := 3

/-- The probability of rolling exactly three odd numbers when five 8-sided dice are rolled -/
theorem prob_three_odd_dice : 
  (numOddSides / numSides) ^ targetOddDice * 
  ((numSides - numOddSides) / numSides) ^ (numDice - targetOddDice) * 
  (Nat.choose numDice targetOddDice) = 5 / 16 := by
  sorry

end NUMINAMATH_CALUDE_prob_three_odd_dice_l427_42772


namespace NUMINAMATH_CALUDE_function_properties_l427_42763

def is_even (f : ℝ → ℝ) : Prop := ∀ x, f (-x) = f x

def is_increasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x < f y

def is_symmetric_about (f : ℝ → ℝ) (c : ℝ) : Prop :=
  ∀ x, f (c + x) = f (c - x)

theorem function_properties (f : ℝ → ℝ) 
  (h1 : is_even f)
  (h2 : ∀ x, f (x + 1) = -f x)
  (h3 : is_increasing_on f (-1) 0) :
  (∀ x, f (x + 2) = f x) ∧ 
  (is_symmetric_about f 1) ∧
  (f 2 = f 0) := by
  sorry

end NUMINAMATH_CALUDE_function_properties_l427_42763


namespace NUMINAMATH_CALUDE_initial_position_of_moving_point_l427_42770

theorem initial_position_of_moving_point (M : ℝ) : 
  (M - 7) + 4 = 0 → M = 3 := by
  sorry

end NUMINAMATH_CALUDE_initial_position_of_moving_point_l427_42770


namespace NUMINAMATH_CALUDE_john_drinks_42_quarts_per_week_l427_42761

/-- The number of quarts John drinks in a week -/
def quarts_per_week (gallons_per_day : ℚ) (days_per_week : ℕ) (quarts_per_gallon : ℕ) : ℚ :=
  gallons_per_day * days_per_week * quarts_per_gallon

/-- Proof that John drinks 42 quarts of water in a week -/
theorem john_drinks_42_quarts_per_week :
  quarts_per_week (3/2) 7 4 = 42 := by
  sorry

end NUMINAMATH_CALUDE_john_drinks_42_quarts_per_week_l427_42761


namespace NUMINAMATH_CALUDE_tangent_sum_and_double_sum_l427_42744

theorem tangent_sum_and_double_sum (α β : Real) 
  (h1 : Real.tan α = 1/7) (h2 : Real.tan β = 1/3) : 
  Real.tan (α + β) = 1/2 ∧ Real.tan (α + 2*β) = 1 := by sorry

end NUMINAMATH_CALUDE_tangent_sum_and_double_sum_l427_42744


namespace NUMINAMATH_CALUDE_sheets_left_after_sharing_l427_42769

def initial_stickers : ℕ := 150
def shared_stickers : ℕ := 100
def stickers_per_sheet : ℕ := 10

theorem sheets_left_after_sharing :
  (initial_stickers - shared_stickers) / stickers_per_sheet = 5 := by
  sorry

end NUMINAMATH_CALUDE_sheets_left_after_sharing_l427_42769


namespace NUMINAMATH_CALUDE_clubsuit_calculation_l427_42745

-- Define the new operation
def clubsuit (x y : ℤ) : ℤ := x^2 - y^2

-- Theorem statement
theorem clubsuit_calculation : clubsuit 5 (clubsuit 6 7) = -144 := by
  sorry

end NUMINAMATH_CALUDE_clubsuit_calculation_l427_42745


namespace NUMINAMATH_CALUDE_pool_filling_time_l427_42774

def spring1_rate : ℚ := 1
def spring2_rate : ℚ := 1/2
def spring3_rate : ℚ := 1/3
def spring4_rate : ℚ := 1/4

def combined_rate : ℚ := spring1_rate + spring2_rate + spring3_rate + spring4_rate

theorem pool_filling_time : (1 : ℚ) / combined_rate = 12/25 := by
  sorry

end NUMINAMATH_CALUDE_pool_filling_time_l427_42774


namespace NUMINAMATH_CALUDE_largest_x_absolute_value_equation_l427_42751

theorem largest_x_absolute_value_equation :
  ∃ (x : ℝ), x = 17 ∧ |2*x - 4| = 30 ∧ ∀ (y : ℝ), |2*y - 4| = 30 → y ≤ x :=
by sorry

end NUMINAMATH_CALUDE_largest_x_absolute_value_equation_l427_42751


namespace NUMINAMATH_CALUDE_equation_one_solutions_l427_42708

theorem equation_one_solutions (x : ℝ) : 
  x^2 - 6*x - 1 = 0 ↔ x = 3 + Real.sqrt 10 ∨ x = 3 - Real.sqrt 10 := by
sorry

end NUMINAMATH_CALUDE_equation_one_solutions_l427_42708


namespace NUMINAMATH_CALUDE_solve_equation_l427_42794

theorem solve_equation (x : ℚ) : 
  3 - 1 / (3 - 2 * x) = 2 / 3 * (1 / (3 - 2 * x)) → x = 2 / 3 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l427_42794


namespace NUMINAMATH_CALUDE_speed_conversion_l427_42728

/-- Conversion factor from meters per second to kilometers per hour -/
def mps_to_kmph : ℝ := 3.6

/-- The initial speed in meters per second -/
def initial_speed : ℝ := 5

/-- Theorem: 5 mps is equal to 18 kmph -/
theorem speed_conversion : initial_speed * mps_to_kmph = 18 := by
  sorry

end NUMINAMATH_CALUDE_speed_conversion_l427_42728


namespace NUMINAMATH_CALUDE_candy_ratio_l427_42773

theorem candy_ratio : ∀ (red yellow blue : ℕ),
  red = 40 →
  yellow = 3 * red - 20 →
  red + blue = 90 →
  blue * 2 = yellow :=
by sorry

end NUMINAMATH_CALUDE_candy_ratio_l427_42773


namespace NUMINAMATH_CALUDE_sum_of_perfect_square_integers_l427_42716

theorem sum_of_perfect_square_integers : ∃ (S : Finset ℕ), 
  (∀ n ∈ S, ∃ k : ℕ, n^2 - 19*n + 99 = k^2) ∧ 
  (∀ n : ℕ, n ∉ S → ¬∃ k : ℕ, n^2 - 19*n + 99 = k^2) ∧
  (S.sum id = 38) := by
  sorry

end NUMINAMATH_CALUDE_sum_of_perfect_square_integers_l427_42716


namespace NUMINAMATH_CALUDE_antonieta_initial_tickets_l427_42768

/-- The number of tickets required for the Ferris wheel -/
def ferris_wheel_cost : ℕ := 6

/-- The number of tickets required for the roller coaster -/
def roller_coaster_cost : ℕ := 5

/-- The number of tickets required for the log ride -/
def log_ride_cost : ℕ := 7

/-- The number of additional tickets Antonieta needs to buy -/
def additional_tickets : ℕ := 16

/-- The initial number of tickets Antonieta has -/
def initial_tickets : ℕ := 2

theorem antonieta_initial_tickets :
  initial_tickets + additional_tickets = ferris_wheel_cost + roller_coaster_cost + log_ride_cost := by
  sorry

end NUMINAMATH_CALUDE_antonieta_initial_tickets_l427_42768


namespace NUMINAMATH_CALUDE_cuboid_edge_length_l427_42723

/-- Theorem: Given a cuboid with edges x cm, 5 cm, and 6 cm, and a volume of 180 cm³,
    the length of the first edge (x) is 6 cm. -/
theorem cuboid_edge_length (x : ℝ) : x * 5 * 6 = 180 → x = 6 := by
  sorry

end NUMINAMATH_CALUDE_cuboid_edge_length_l427_42723


namespace NUMINAMATH_CALUDE_system_solution_l427_42748

theorem system_solution (x y z : ℝ) 
  (eq1 : x * y = 5 - 3 * x - 2 * y)
  (eq2 : y * z = 8 - 5 * y - 3 * z)
  (eq3 : x * z = 18 - 2 * x - 5 * z)
  (pos_x : x > 0) : x = 6 := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l427_42748


namespace NUMINAMATH_CALUDE_least_value_x_minus_y_plus_z_l427_42758

theorem least_value_x_minus_y_plus_z (x y z : ℕ+) 
  (h : (3 : ℕ) * x = (4 : ℕ) * y ∧ (4 : ℕ) * y = (7 : ℕ) * z) : 
  (∀ a b c : ℕ+, (3 : ℕ) * a = (4 : ℕ) * b ∧ (4 : ℕ) * b = (7 : ℕ) * c → 
    (x : ℤ) - (y : ℤ) + (z : ℤ) ≤ (a : ℤ) - (b : ℤ) + (c : ℤ)) ∧
  (x : ℤ) - (y : ℤ) + (z : ℤ) = 19 :=
sorry

end NUMINAMATH_CALUDE_least_value_x_minus_y_plus_z_l427_42758


namespace NUMINAMATH_CALUDE_book_arrangement_count_book_arrangement_proof_l427_42741

theorem book_arrangement_count : ℕ :=
  let total_books : ℕ := 4 + 3 + 2
  let geometry_books : ℕ := 4
  let number_theory_books : ℕ := 3
  let algebra_books : ℕ := 2
  Nat.choose total_books geometry_books * 
  Nat.choose (total_books - geometry_books) number_theory_books * 
  Nat.choose (total_books - geometry_books - number_theory_books) algebra_books

theorem book_arrangement_proof : book_arrangement_count = 1260 := by
  sorry

end NUMINAMATH_CALUDE_book_arrangement_count_book_arrangement_proof_l427_42741


namespace NUMINAMATH_CALUDE_quadratic_has_real_root_l427_42754

theorem quadratic_has_real_root (a b : ℝ) : ∃ x : ℝ, x^2 + a*x + b = 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_has_real_root_l427_42754


namespace NUMINAMATH_CALUDE_cubic_function_properties_monotonic_cubic_function_range_l427_42733

/-- A cubic function with specified properties -/
def f (a b c d : ℝ) (x : ℝ) : ℝ := a * x^3 + b * x^2 + c * x + d

/-- The derivative of f -/
def f' (a b c : ℝ) (x : ℝ) : ℝ := 3 * a * x^2 + 2 * b * x + c

/-- Theorem for part I -/
theorem cubic_function_properties (a b c d : ℝ) (ha : a ≠ 0) :
  (∀ x, f a b c d x = -f a b c d (-x)) →  -- Symmetry about origin
  (f a b c d (1/2) = -1) →                -- Minimum value at x = 1/2
  (f' a b c (1/2) = 0) →                  -- Critical point at x = 1/2
  (f a b c d = f 4 0 (-3) 0) :=
sorry

/-- Theorem for part II -/
theorem monotonic_cubic_function_range (c : ℝ) :
  (∀ x y, x < y → (f 1 1 c 1 x < f 1 1 c 1 y) ∨ (∀ x y, x < y → f 1 1 c 1 x > f 1 1 c 1 y)) →
  c ≥ 1/3 :=
sorry

end NUMINAMATH_CALUDE_cubic_function_properties_monotonic_cubic_function_range_l427_42733


namespace NUMINAMATH_CALUDE_kolya_parallelepiped_edge_length_l427_42739

/-- A rectangular parallelepiped constructed from unit cubes -/
structure Parallelepiped where
  length : ℕ
  width : ℕ
  height : ℕ
  volume : ℕ
  edge_min : ℕ

/-- The total length of all edges of a rectangular parallelepiped -/
def total_edge_length (p : Parallelepiped) : ℕ :=
  4 * (p.length + p.width + p.height)

/-- Theorem stating the total edge length of the specific parallelepiped -/
theorem kolya_parallelepiped_edge_length :
  ∃ (p : Parallelepiped),
    p.volume = 440 ∧
    p.edge_min = 5 ∧
    p.length ≥ p.edge_min ∧
    p.width ≥ p.edge_min ∧
    p.height ≥ p.edge_min ∧
    p.volume = p.length * p.width * p.height ∧
    total_edge_length p = 96 := by
  sorry

end NUMINAMATH_CALUDE_kolya_parallelepiped_edge_length_l427_42739


namespace NUMINAMATH_CALUDE_shaded_area_semicircle_pattern_l427_42766

/-- The area of the shaded region formed by semicircles in a pattern -/
theorem shaded_area_semicircle_pattern (diameter : ℝ) (pattern_length : ℝ) : 
  diameter = 3 →
  pattern_length = 18 →
  (pattern_length / diameter) * (π * (diameter / 2)^2 / 2) = 27 / 4 * π := by
  sorry


end NUMINAMATH_CALUDE_shaded_area_semicircle_pattern_l427_42766


namespace NUMINAMATH_CALUDE_person_age_puzzle_l427_42795

theorem person_age_puzzle : ∃ (x : ℝ), x > 0 ∧ x = 4 * (x + 4) - 4 * (x - 4) + (1/2) * (x - 6) ∧ x = 58 := by
  sorry

end NUMINAMATH_CALUDE_person_age_puzzle_l427_42795


namespace NUMINAMATH_CALUDE_cake_price_is_twelve_l427_42785

/-- Represents the daily sales and expenses of Marie's bakery --/
structure BakeryFinances where
  cashRegisterCost : ℕ
  breadPrice : ℕ
  breadQuantity : ℕ
  cakeQuantity : ℕ
  rentCost : ℕ
  electricityCost : ℕ
  profitDays : ℕ

/-- Calculates the price of each cake based on the given finances --/
def calculateCakePrice (finances : BakeryFinances) : ℕ :=
  let dailyBreadIncome := finances.breadPrice * finances.breadQuantity
  let dailyExpenses := finances.rentCost + finances.electricityCost
  let dailyProfitWithoutCakes := dailyBreadIncome - dailyExpenses
  let totalProfit := finances.cashRegisterCost
  let profitFromCakes := totalProfit - (finances.profitDays * dailyProfitWithoutCakes)
  profitFromCakes / (finances.cakeQuantity * finances.profitDays)

/-- Theorem stating that the cake price is $12 given the specific conditions --/
theorem cake_price_is_twelve (finances : BakeryFinances)
  (h1 : finances.cashRegisterCost = 1040)
  (h2 : finances.breadPrice = 2)
  (h3 : finances.breadQuantity = 40)
  (h4 : finances.cakeQuantity = 6)
  (h5 : finances.rentCost = 20)
  (h6 : finances.electricityCost = 2)
  (h7 : finances.profitDays = 8) :
  calculateCakePrice finances = 12 := by
  sorry

end NUMINAMATH_CALUDE_cake_price_is_twelve_l427_42785


namespace NUMINAMATH_CALUDE_total_exercise_hours_l427_42738

/-- Exercise duration in minutes for each person -/
def natasha_minutes : ℕ := 30 * 7
def esteban_minutes : ℕ := 10 * 9
def charlotte_minutes : ℕ := 20 + 45 + 70 + 100

/-- Total exercise duration in minutes -/
def total_minutes : ℕ := natasha_minutes + esteban_minutes + charlotte_minutes

/-- Conversion factor from minutes to hours -/
def minutes_per_hour : ℕ := 60

/-- Total exercise duration in hours -/
def total_hours : ℚ := total_minutes / minutes_per_hour

/-- Theorem: The total hours of exercise for all three individuals is 8.92 hours -/
theorem total_exercise_hours : total_hours = 892 / 100 := by
  sorry

end NUMINAMATH_CALUDE_total_exercise_hours_l427_42738


namespace NUMINAMATH_CALUDE_money_ratio_is_two_to_one_l427_42767

/-- The ratio of Peter's money to John's money -/
def money_ratio : ℚ :=
  let peter_money : ℕ := 320
  let quincy_money : ℕ := peter_money + 20
  let andrew_money : ℕ := quincy_money + (quincy_money * 15 / 100)
  let total_money : ℕ := 1200 + 11
  let john_money : ℕ := total_money - peter_money - quincy_money - andrew_money
  peter_money / john_money

theorem money_ratio_is_two_to_one : money_ratio = 2 := by
  sorry

end NUMINAMATH_CALUDE_money_ratio_is_two_to_one_l427_42767


namespace NUMINAMATH_CALUDE_percentage_50_59_range_l427_42749

/-- Represents the frequency distribution of scores in Mrs. Lopez's geometry class -/
structure ScoreDistribution :=
  (score_90_100 : Nat)
  (score_80_89 : Nat)
  (score_70_79 : Nat)
  (score_60_69 : Nat)
  (score_50_59 : Nat)
  (score_below_50 : Nat)

/-- Calculates the total number of students -/
def totalStudents (dist : ScoreDistribution) : Nat :=
  dist.score_90_100 + dist.score_80_89 + dist.score_70_79 + 
  dist.score_60_69 + dist.score_50_59 + dist.score_below_50

/-- The actual score distribution in Mrs. Lopez's class -/
def lopezClassDist : ScoreDistribution :=
  { score_90_100 := 3
  , score_80_89 := 6
  , score_70_79 := 8
  , score_60_69 := 4
  , score_50_59 := 3
  , score_below_50 := 4
  }

/-- Theorem stating that the percentage of students who scored in the 50%-59% range is 3/28 * 100% -/
theorem percentage_50_59_range (dist : ScoreDistribution) :
  dist = lopezClassDist →
  (dist.score_50_59 : Rat) / (totalStudents dist : Rat) = 3 / 28 := by
  sorry

end NUMINAMATH_CALUDE_percentage_50_59_range_l427_42749


namespace NUMINAMATH_CALUDE_two_valid_antonyms_exist_l427_42719

/-- A word is represented as a string of characters. -/
def Word := String

/-- The maximum allowed length for an antonym. -/
def MaxLength : Nat := 10

/-- Predicate to check if a word is an antonym of "seldom". -/
def IsAntonymOfSeldom (w : Word) : Prop := sorry

/-- Predicate to check if two words have distinct meanings. -/
def HasDistinctMeaning (w1 w2 : Word) : Prop := sorry

/-- Theorem stating the existence of two valid antonyms for "seldom". -/
theorem two_valid_antonyms_exist : 
  ∃ (w1 w2 : Word), 
    IsAntonymOfSeldom w1 ∧ 
    IsAntonymOfSeldom w2 ∧ 
    w1.length ≤ MaxLength ∧ 
    w2.length ≤ MaxLength ∧ 
    w1.front = 'o' ∧ 
    w2.front = 'u' ∧ 
    HasDistinctMeaning w1 w2 :=
  sorry

end NUMINAMATH_CALUDE_two_valid_antonyms_exist_l427_42719


namespace NUMINAMATH_CALUDE_distance_A_to_C_l427_42790

/-- Given four collinear points A, B, C, and D in that order, with specific distance relationships,
    prove that the distance from A to C is 15. -/
theorem distance_A_to_C (A B C D : ℝ) : 
  A < B ∧ B < C ∧ C < D →  -- Points are on a line in order
  D - A = 24 →             -- Distance from A to D is 24
  D - B = 3 * (B - A) →    -- Distance from B to D is 3 times the distance from A to B
  C - B = (D - B) / 2 →    -- C is halfway between B and D
  C - A = 15 := by         -- Distance from A to C is 15
sorry

end NUMINAMATH_CALUDE_distance_A_to_C_l427_42790


namespace NUMINAMATH_CALUDE_money_distribution_l427_42798

theorem money_distribution (A B C : ℕ) : 
  A + B + C = 500 → A + C = 200 → B + C = 330 → C = 30 := by
  sorry

end NUMINAMATH_CALUDE_money_distribution_l427_42798


namespace NUMINAMATH_CALUDE_min_value_expression_l427_42730

theorem min_value_expression (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  ∃ (min : ℝ), min = -500000 ∧
  ∀ (x y : ℝ), x > 0 → y > 0 →
    (x + 1/y) * (x + 1/y - 1000) + (y + 1/x) * (y + 1/x - 1000) ≥ min :=
by sorry

end NUMINAMATH_CALUDE_min_value_expression_l427_42730


namespace NUMINAMATH_CALUDE_smallest_number_proof_l427_42747

theorem smallest_number_proof (a b c : ℕ) : 
  a > 0 ∧ b > 0 ∧ c > 0 →
  (a + b + c) / 3 = 24 →
  b = 25 →
  max a (max b c) = b + 6 →
  min a (min b c) = 16 :=
by sorry

end NUMINAMATH_CALUDE_smallest_number_proof_l427_42747


namespace NUMINAMATH_CALUDE_inequality_solution_set_l427_42781

theorem inequality_solution_set (x : ℝ) :
  Set.Icc (-3 : ℝ) 1 \ {1} = {x | (5*x + 3)/(x - 1) ≤ 3 ∧ x ≠ 1} :=
sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l427_42781


namespace NUMINAMATH_CALUDE_remainder_problem_l427_42760

theorem remainder_problem (n : ℤ) (h : n % 9 = 4) : (4 * n - 3) % 9 = 4 := by
  sorry

end NUMINAMATH_CALUDE_remainder_problem_l427_42760


namespace NUMINAMATH_CALUDE_fifth_term_of_specific_sequence_l427_42771

/-- Represents an arithmetic sequence -/
structure ArithmeticSequence where
  first : ℝ
  second : ℝ

/-- Get the nth term of an arithmetic sequence -/
def nthTerm (seq : ArithmeticSequence) (n : ℕ) : ℝ :=
  seq.first + (n - 1) * (seq.second - seq.first)

/-- The theorem to prove -/
theorem fifth_term_of_specific_sequence :
  let seq := ArithmeticSequence.mk 3 8
  nthTerm seq 5 = 23 := by sorry

end NUMINAMATH_CALUDE_fifth_term_of_specific_sequence_l427_42771


namespace NUMINAMATH_CALUDE_point_on_angle_terminal_side_l427_42732

theorem point_on_angle_terminal_side (y : ℝ) :
  let P : ℝ × ℝ := (-1, y)
  let θ : ℝ := 2 * Real.pi / 3
  (P.1 = -1) →   -- x-coordinate is -1
  (Real.tan θ = y / P.1) →  -- point is on terminal side of angle θ
  y = Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_point_on_angle_terminal_side_l427_42732


namespace NUMINAMATH_CALUDE_max_dominoes_formula_l427_42784

/-- Represents a grid of size 2n × 2n -/
structure Grid (n : ℕ+) where
  size : ℕ := 2 * n

/-- Represents a domino placement on the grid -/
structure DominoPlacement (n : ℕ+) where
  grid : Grid n
  num_dominoes : ℕ
  valid : Prop  -- This represents the validity of the placement according to the rules

/-- The maximum number of dominoes that can be placed on a 2n × 2n grid -/
def max_dominoes (n : ℕ+) : ℕ := n * (n + 1) / 2

/-- Theorem stating that the maximum number of dominoes is n(n+1)/2 -/
theorem max_dominoes_formula (n : ℕ+) :
  ∀ (p : DominoPlacement n), p.valid → p.num_dominoes ≤ max_dominoes n :=
sorry

end NUMINAMATH_CALUDE_max_dominoes_formula_l427_42784


namespace NUMINAMATH_CALUDE_quadratic_coefficient_l427_42706

/-- A quadratic function with vertex (3, 2) passing through (-2, -43) has a = -1.8 -/
theorem quadratic_coefficient (a b c : ℝ) : 
  (∀ x, (a * x^2 + b * x + c) = a * (x - 3)^2 + 2) → 
  (a * (-2)^2 + b * (-2) + c = -43) → 
  a = -1.8 := by
sorry

end NUMINAMATH_CALUDE_quadratic_coefficient_l427_42706


namespace NUMINAMATH_CALUDE_positive_number_property_l427_42734

theorem positive_number_property (x : ℝ) (h1 : x > 0) (h2 : 0.01 * x^2 + 16 = 36) : x = 20 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_positive_number_property_l427_42734


namespace NUMINAMATH_CALUDE_solution_set_eq_four_points_l427_42753

/-- The set of solutions to the system of equations:
    a^4 - b^4 = c
    b^4 - c^4 = a
    c^4 - a^4 = b
    where a, b, c are real numbers. -/
def SolutionSet : Set (ℝ × ℝ × ℝ) :=
  {abc | let (a, b, c) := abc
         a^4 - b^4 = c ∧
         b^4 - c^4 = a ∧
         c^4 - a^4 = b}

/-- The theorem stating that the solution set is equal to the given set of four points. -/
theorem solution_set_eq_four_points :
  SolutionSet = {(0, 0, 0), (0, 1, -1), (-1, 0, 1), (1, -1, 0)} := by
  sorry

end NUMINAMATH_CALUDE_solution_set_eq_four_points_l427_42753


namespace NUMINAMATH_CALUDE_sandy_change_l427_42714

def football_price : ℚ := 9.14
def baseball_price : ℚ := 6.81
def payment : ℚ := 20

theorem sandy_change : payment - (football_price + baseball_price) = 4.05 := by
  sorry

end NUMINAMATH_CALUDE_sandy_change_l427_42714


namespace NUMINAMATH_CALUDE_vector_rotation_angle_l427_42776

theorem vector_rotation_angle (p : ℂ) (α : ℝ) (h_p : p ≠ 0) :
  p + p * Complex.exp (2 * α * Complex.I) = p * Complex.exp (α * Complex.I) →
  α = π / 3 + 2 * π * ↑k ∨ α = -π / 3 + 2 * π * ↑n :=
by sorry

end NUMINAMATH_CALUDE_vector_rotation_angle_l427_42776


namespace NUMINAMATH_CALUDE_difference_of_squares_special_case_l427_42720

theorem difference_of_squares_special_case : (3 + Real.sqrt 2) * (3 - Real.sqrt 2) = 7 := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_special_case_l427_42720


namespace NUMINAMATH_CALUDE_no_solution_equation_l427_42752

theorem no_solution_equation : 
  ¬∃ (x : ℝ), x - 9 / (x - 4) = 4 - 9 / (x - 4) := by
  sorry

end NUMINAMATH_CALUDE_no_solution_equation_l427_42752


namespace NUMINAMATH_CALUDE_indeterminate_relation_product_and_means_l427_42777

/-- Given two positive real numbers, their arithmetic mean, and their geometric mean,
    the relationship between the product of the numbers and the product of their means
    cannot be determined. -/
theorem indeterminate_relation_product_and_means (a b : ℝ) (A G : ℝ) 
    (ha : 0 < a) (hb : 0 < b)
    (hA : A = (a + b) / 2)
    (hG : G = Real.sqrt (a * b)) :
    ¬ ∀ (R : ℝ → ℝ → Prop), R (a * b) (A * G) ∨ R (A * G) (a * b) := by
  sorry

end NUMINAMATH_CALUDE_indeterminate_relation_product_and_means_l427_42777


namespace NUMINAMATH_CALUDE_quadratic_coefficient_l427_42783

/-- Given a quadratic function y = ax² + bx + c, if (2, y₁) and (-2, y₂) are points on this function
    and y₁ - y₂ = -16, then b = -4. -/
theorem quadratic_coefficient (a b c y₁ y₂ : ℝ) : 
  y₁ = a * 2^2 + b * 2 + c →
  y₂ = a * (-2)^2 + b * (-2) + c →
  y₁ - y₂ = -16 →
  b = -4 := by
sorry

end NUMINAMATH_CALUDE_quadratic_coefficient_l427_42783


namespace NUMINAMATH_CALUDE_solutions_for_20_l427_42701

/-- The number of integer solutions for |x| + |y| = n -/
def num_solutions (n : ℕ) : ℕ := 4 * n

/-- Given conditions -/
axiom solution_1 : num_solutions 1 = 4
axiom solution_2 : num_solutions 2 = 8
axiom solution_3 : num_solutions 3 = 12

/-- Theorem: The number of different integer solutions for |x| + |y| = 20 is 80 -/
theorem solutions_for_20 : num_solutions 20 = 80 := by sorry

end NUMINAMATH_CALUDE_solutions_for_20_l427_42701


namespace NUMINAMATH_CALUDE_sum_of_constants_l427_42713

def polynomial (x : ℝ) : ℝ := x^3 - 7*x^2 + 14*x - 8

def t (k : ℕ) : ℝ := sorry

theorem sum_of_constants (x y z : ℝ) : 
  (∀ k ≥ 2, t (k+1) = x * t k + y * t (k-1) + z * t (k-2)) →
  t 0 = 3 →
  t 1 = 7 →
  t 2 = 15 →
  x + y + z = 3 := by sorry

end NUMINAMATH_CALUDE_sum_of_constants_l427_42713


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l427_42737

theorem sufficient_not_necessary 
  (a b : ℝ) (ha : a > 0) (hb : b > 0) : 
  (∃ x y, x + y > a + b ∧ x * y > a * b ∧ ¬(x > a ∧ y > b)) ∧ 
  (∀ x y, x > a ∧ y > b → x + y > a + b ∧ x * y > a * b) := by
sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l427_42737


namespace NUMINAMATH_CALUDE_perpendicular_vectors_k_value_l427_42727

-- Define the vectors i and j
def i : ℝ × ℝ := (1, 0)
def j : ℝ × ℝ := (0, 1)

-- Define vectors a and b
def a : ℝ × ℝ := (2 * i.1 + 3 * j.1, 2 * i.2 + 3 * j.2)
def b (k : ℝ) : ℝ × ℝ := (k * i.1 - 4 * j.1, k * i.2 - 4 * j.2)

-- Define the dot product for 2D vectors
def dot_product (v w : ℝ × ℝ) : ℝ := v.1 * w.1 + v.2 * w.2

-- Define perpendicularity
def perpendicular (v w : ℝ × ℝ) : Prop := dot_product v w = 0

-- Theorem statement
theorem perpendicular_vectors_k_value :
  ∃ k : ℝ, perpendicular a (b k) ∧ k = 6 :=
sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_k_value_l427_42727


namespace NUMINAMATH_CALUDE_equilateral_triangle_side_length_l427_42709

theorem equilateral_triangle_side_length 
  (circumference : ℝ) 
  (h1 : circumference = 4 * 21) 
  (h2 : circumference > 0) : 
  ∃ (side_length : ℝ), side_length = 28 ∧ 3 * side_length = circumference :=
sorry

end NUMINAMATH_CALUDE_equilateral_triangle_side_length_l427_42709


namespace NUMINAMATH_CALUDE_power_inequality_l427_42740

theorem power_inequality (x : ℝ) (α : ℝ) (h1 : x > -1) :
  (0 < α ∧ α < 1 → (1 + x)^α ≤ 1 + α * x) ∧
  ((α < 0 ∨ α > 1) → (1 + x)^α ≥ 1 + α * x) := by
  sorry

end NUMINAMATH_CALUDE_power_inequality_l427_42740


namespace NUMINAMATH_CALUDE_cubic_identity_fraction_l427_42725

theorem cubic_identity_fraction (x y z : ℝ) :
  ((x - y)^3 + (y - z)^3 + (z - x)^3) / (15 * (x - y) * (y - z) * (z - x)) = 1/5 := by
  sorry

end NUMINAMATH_CALUDE_cubic_identity_fraction_l427_42725


namespace NUMINAMATH_CALUDE_cube_surface_area_l427_42789

theorem cube_surface_area (d : ℝ) (h : d = 8 * Real.sqrt 3) :
  6 * (d / Real.sqrt 3)^2 = 384 := by
  sorry

end NUMINAMATH_CALUDE_cube_surface_area_l427_42789


namespace NUMINAMATH_CALUDE_sarah_brother_books_l427_42787

/-- The number of books Sarah's brother bought -/
def brothers_books (sarah_paperbacks sarah_hardbacks : ℕ) : ℕ :=
  (sarah_paperbacks / 3) + (sarah_hardbacks * 2)

/-- Theorem: Sarah's brother bought 10 books in total -/
theorem sarah_brother_books :
  brothers_books 6 4 = 10 := by
  sorry

end NUMINAMATH_CALUDE_sarah_brother_books_l427_42787


namespace NUMINAMATH_CALUDE_mouse_testes_most_appropriate_l427_42793

-- Define the possible experimental materials
inductive ExperimentalMaterial
| AscarisEggs
| ChickenLiver
| MouseTestes
| OnionEpidermis

-- Define the cell division processes
inductive CellDivisionProcess
| Mitosis
| Meiosis
| NoDivision

-- Define the property of continuous cell formation
def hasContinuousCellFormation : ExperimentalMaterial → Prop
| ExperimentalMaterial.MouseTestes => True
| _ => False

-- Define the cell division process for each material
def cellDivisionProcess : ExperimentalMaterial → CellDivisionProcess
| ExperimentalMaterial.AscarisEggs => CellDivisionProcess.Mitosis
| ExperimentalMaterial.ChickenLiver => CellDivisionProcess.Mitosis
| ExperimentalMaterial.MouseTestes => CellDivisionProcess.Meiosis
| ExperimentalMaterial.OnionEpidermis => CellDivisionProcess.NoDivision

-- Define the property of being appropriate for observing meiosis
def isAppropriateForMeiosis (material : ExperimentalMaterial) : Prop :=
  cellDivisionProcess material = CellDivisionProcess.Meiosis ∧ hasContinuousCellFormation material

-- Theorem statement
theorem mouse_testes_most_appropriate :
  ∀ material : ExperimentalMaterial,
    isAppropriateForMeiosis material → material = ExperimentalMaterial.MouseTestes :=
by
  sorry

end NUMINAMATH_CALUDE_mouse_testes_most_appropriate_l427_42793


namespace NUMINAMATH_CALUDE_neg_p_sufficient_not_necessary_for_neg_q_l427_42712

-- Define the conditions
def p (x : ℝ) : Prop := abs x > 1
def q (x : ℝ) : Prop := x < -2

-- State the theorem
theorem neg_p_sufficient_not_necessary_for_neg_q :
  (∀ x : ℝ, ¬(p x) → ¬(q x)) ∧ (∃ x : ℝ, ¬(q x) ∧ p x) := by
  sorry

end NUMINAMATH_CALUDE_neg_p_sufficient_not_necessary_for_neg_q_l427_42712


namespace NUMINAMATH_CALUDE_problem_solution_l427_42721

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (Real.exp (-x)) / a + a / (Real.exp (-x))

theorem problem_solution (a : ℝ) (h_a : a > 0) 
  (h_even : ∀ x, f a x = f a (-x)) :
  (a = 1) ∧
  (∀ x y, x ≥ 0 → y ≥ 0 → x < y → f a x < f a y) ∧
  (∀ m, (∀ x, f 1 x - m^2 + m ≥ 0) ↔ -1 ≤ m ∧ m ≤ 2) :=
by sorry

end NUMINAMATH_CALUDE_problem_solution_l427_42721


namespace NUMINAMATH_CALUDE_faulty_odometer_conversion_l427_42799

/-- Represents an odometer that skips certain digits -/
structure FaultyOdometer where
  reading : Nat
  skipped_digits : List Nat

/-- Converts a faulty odometer reading to actual miles traveled -/
def actual_miles (odo : FaultyOdometer) : Nat :=
  sorry

/-- The theorem stating that a faulty odometer reading of 003006 
    (skipping 3 and 4) represents 1030 actual miles -/
theorem faulty_odometer_conversion :
  let odo : FaultyOdometer := { reading := 3006, skipped_digits := [3, 4] }
  actual_miles odo = 1030 := by
  sorry

end NUMINAMATH_CALUDE_faulty_odometer_conversion_l427_42799


namespace NUMINAMATH_CALUDE_red_lucky_stars_count_l427_42736

theorem red_lucky_stars_count (blue : ℕ) (yellow : ℕ) (red : ℕ) :
  blue = 20 →
  yellow = 15 →
  (red : ℚ) / (red + blue + yellow : ℚ) = 1/2 →
  red = 35 := by
sorry

end NUMINAMATH_CALUDE_red_lucky_stars_count_l427_42736


namespace NUMINAMATH_CALUDE_symmetric_points_count_l427_42735

-- Define the function f
noncomputable def f (x : ℝ) : ℝ :=
  if x < 0 then 2 * x^2 + 4 * x + 1 else 2 / Real.exp x

-- Define symmetry about the origin
def symmetric_about_origin (p q : ℝ × ℝ) : Prop :=
  p.1 = -q.1 ∧ p.2 = -q.2

-- State the theorem
theorem symmetric_points_count :
  ∃ (p₁ q₁ p₂ q₂ : ℝ × ℝ),
    p₁ ≠ q₁ ∧ p₂ ≠ q₂ ∧ p₁ ≠ p₂ ∧
    symmetric_about_origin p₁ q₁ ∧
    symmetric_about_origin p₂ q₂ ∧
    (∀ x, f x = p₁.2 ↔ x = p₁.1) ∧
    (∀ x, f x = q₁.2 ↔ x = q₁.1) ∧
    (∀ x, f x = p₂.2 ↔ x = p₂.1) ∧
    (∀ x, f x = q₂.2 ↔ x = q₂.1) ∧
    (∀ p q : ℝ × ℝ, 
      p ≠ p₁ ∧ p ≠ q₁ ∧ p ≠ p₂ ∧ p ≠ q₂ ∧
      q ≠ p₁ ∧ q ≠ q₁ ∧ q ≠ p₂ ∧ q ≠ q₂ ∧
      symmetric_about_origin p q ∧
      (∀ x, f x = p.2 ↔ x = p.1) ∧
      (∀ x, f x = q.2 ↔ x = q.1) →
      False) :=
sorry

end NUMINAMATH_CALUDE_symmetric_points_count_l427_42735


namespace NUMINAMATH_CALUDE_badge_exchange_l427_42779

theorem badge_exchange (t : ℕ) (v : ℕ) : 
  v = t + 5 →
  (v - (24 * v) / 100 + (20 * t) / 100) + 1 = (t - (20 * t) / 100 + (24 * v) / 100) →
  t = 45 ∧ v = 50 :=
by sorry

end NUMINAMATH_CALUDE_badge_exchange_l427_42779


namespace NUMINAMATH_CALUDE_double_elim_advantage_l427_42742

-- Define the probability of team A winning against other teams
variable (p : ℝ)

-- Define the conditions
def knockout_prob := p^2
def double_elim_prob := p^3 * (3 - 2*p)

-- State the theorem
theorem double_elim_advantage (h1 : 1/2 < p) (h2 : p < 1) :
  knockout_prob p < double_elim_prob p :=
sorry

end NUMINAMATH_CALUDE_double_elim_advantage_l427_42742


namespace NUMINAMATH_CALUDE_circumcircle_radius_obtuse_triangle_consecutive_sides_l427_42703

/-- The radius of the circumcircle of an obtuse triangle with consecutive integer sides --/
theorem circumcircle_radius_obtuse_triangle_consecutive_sides : 
  ∀ (a b c : ℕ) (R : ℝ),
    a + 1 = b → b + 1 = c →  -- Consecutive integer sides
    a < b ∧ b < c →          -- Ordered sides
    a^2 + b^2 < c^2 →        -- Obtuse triangle condition
    R = (8 * Real.sqrt 15) / 15 →  -- Radius of circumcircle
    2 * R * Real.sin (Real.arccos ((a^2 + b^2 - c^2) / (2 * a * b))) = c := by
  sorry


end NUMINAMATH_CALUDE_circumcircle_radius_obtuse_triangle_consecutive_sides_l427_42703


namespace NUMINAMATH_CALUDE_at_least_one_non_integer_distance_l427_42705

/-- Given four points A, B, C, D on a plane with specified distances,
    prove that at least one of BD or CD is not an integer. -/
theorem at_least_one_non_integer_distance
  (A B C D : EuclideanSpace ℝ (Fin 2))
  (h_AB : dist A B = 1)
  (h_BC : dist B C = 9)
  (h_CA : dist C A = 9)
  (h_AD : dist A D = 7) :
  ¬(∃ (bd cd : ℤ), dist B D = bd ∧ dist C D = cd) :=
sorry

end NUMINAMATH_CALUDE_at_least_one_non_integer_distance_l427_42705


namespace NUMINAMATH_CALUDE_f_expression_f_range_l427_42796

/-- A quadratic function f satisfying the given conditions -/
def f (x : ℝ) : ℝ := sorry

/-- The property that f(x+1) - f(x) = 2x -/
axiom f_diff (x : ℝ) : f (x + 1) - f x = 2 * x

/-- The property that f(0) = 1 -/
axiom f_zero : f 0 = 1

/-- Theorem: The analytical expression of f(x) -/
theorem f_expression (x : ℝ) : f x = x^2 - x + 1 := sorry

/-- Theorem: The range of f(x) when x ∈ [-1, 1] -/
theorem f_range : Set.Icc (3/4 : ℝ) 3 = { y | ∃ x ∈ Set.Icc (-1 : ℝ) 1, f x = y } := sorry

end NUMINAMATH_CALUDE_f_expression_f_range_l427_42796


namespace NUMINAMATH_CALUDE_decimal_division_proof_l427_42702

theorem decimal_division_proof : 
  (0.182 : ℚ) / (0.0021 : ℚ) = 86 + 14 / 21 := by sorry

end NUMINAMATH_CALUDE_decimal_division_proof_l427_42702


namespace NUMINAMATH_CALUDE_distance_between_points_l427_42757

theorem distance_between_points : 
  let p1 : ℝ × ℝ := (-3, -4)
  let p2 : ℝ × ℝ := (4, -5)
  Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2) = 5 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_points_l427_42757


namespace NUMINAMATH_CALUDE_another_max_occurrence_sequence_l427_42750

/-- Represents a circular strip of zeros and ones -/
def CircularStrip := List Bool

/-- Counts the number of occurrences of a sequence in a circular strip -/
def count_occurrences (strip : CircularStrip) (seq : List Bool) : Nat :=
  sorry

/-- The sequence with the maximum number of occurrences -/
def max_seq (n : Nat) : List Bool :=
  [true, true] ++ List.replicate (n - 2) false

/-- The sequence with the minimum number of occurrences -/
def min_seq (n : Nat) : List Bool :=
  List.replicate (n - 2) false ++ [true, true]

theorem another_max_occurrence_sequence 
  (n : Nat) 
  (h_n : n > 5) 
  (strip : CircularStrip) 
  (h_strip : strip.length > 0) 
  (h_max : ∀ seq : List Bool, seq.length = n → 
    count_occurrences strip seq ≤ count_occurrences strip (max_seq n)) 
  (h_min : count_occurrences strip (min_seq n) < count_occurrences strip (max_seq n)) :
  ∃ seq : List Bool, 
    seq.length = n ∧ 
    seq ≠ max_seq n ∧ 
    count_occurrences strip seq = count_occurrences strip (max_seq n) :=
  sorry

end NUMINAMATH_CALUDE_another_max_occurrence_sequence_l427_42750


namespace NUMINAMATH_CALUDE_sequence_existence_l427_42746

theorem sequence_existence : ∃ (a : ℕ → ℕ) (M : ℕ), 
  (∀ n, a n ≤ a (n + 1)) ∧ 
  (∀ k, ∃ n, a n > k) ∧
  (∀ n ≥ M, ¬(Nat.Prime (n + 1)) → 
    ∀ p, Nat.Prime p → p ∣ (Nat.factorial n + 1) → p > n + a n) := by
  sorry

end NUMINAMATH_CALUDE_sequence_existence_l427_42746


namespace NUMINAMATH_CALUDE_matrix_inverse_proof_l427_42731

def A : Matrix (Fin 2) (Fin 2) ℝ := !![5, -3; -2, 1]

theorem matrix_inverse_proof :
  A⁻¹ = !![(-1 : ℝ), -3; -2, -5] := by
  sorry

end NUMINAMATH_CALUDE_matrix_inverse_proof_l427_42731


namespace NUMINAMATH_CALUDE_expression_evaluation_l427_42765

theorem expression_evaluation (c a b d : ℚ) 
  (h1 : d = a + 1)
  (h2 : a = b - 3)
  (h3 : b = c + 5)
  (h4 : c = 6)
  (h5 : d + 3 ≠ 0)
  (h6 : a + 2 ≠ 0)
  (h7 : b - 5 ≠ 0)
  (h8 : c + 7 ≠ 0) :
  ((d + 5) / (d + 3)) * ((a + 3) / (a + 2)) * ((b - 3) / (b - 5)) * ((c + 10) / (c + 7)) = 1232 / 585 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l427_42765


namespace NUMINAMATH_CALUDE_name_calculation_result_l427_42726

/-- Represents the alphabetical position of a letter (A=1, B=2, ..., Z=26) -/
def alphabeticalPosition (c : Char) : Nat :=
  match c with
  | 'A' => 1 | 'B' => 2 | 'C' => 3 | 'D' => 4 | 'E' => 5
  | 'F' => 6 | 'G' => 7 | 'H' => 8 | 'I' => 9 | 'J' => 10
  | 'K' => 11 | 'L' => 12 | 'M' => 13 | 'N' => 14 | 'O' => 15
  | 'P' => 16 | 'Q' => 17 | 'R' => 18 | 'S' => 19 | 'T' => 20
  | 'U' => 21 | 'V' => 22 | 'W' => 23 | 'X' => 24 | 'Y' => 25
  | 'Z' => 26
  | _ => 0

theorem name_calculation_result :
  let elida := "ELIDA"
  let adrianna := "ADRIANNA"
  let belinda := "BELINDA"

  let elida_sum := (elida.data.map alphabeticalPosition).sum
  let adrianna_sum := (adrianna.data.map alphabeticalPosition).sum
  let belinda_sum := (belinda.data.map alphabeticalPosition).sum

  let total_sum := elida_sum + adrianna_sum + belinda_sum
  let average := total_sum / 3

  elida.length = 5 →
  adrianna.length = 2 * elida.length - 2 →
  (average * 3 : ℕ) - elida_sum = 109 := by
  sorry

#check name_calculation_result

end NUMINAMATH_CALUDE_name_calculation_result_l427_42726


namespace NUMINAMATH_CALUDE_new_solutions_introduced_l427_42797

variables {α : Type*} [LinearOrder α]
variable (x : α)
variable (F₁ F₂ f : α → ℝ)

theorem new_solutions_introduced (h : F₁ x > F₂ x) :
  (f x < 0 ∧ F₁ x < F₂ x) ↔ (f x * F₁ x < f x * F₂ x ∧ ¬(F₁ x > F₂ x)) :=
by sorry

end NUMINAMATH_CALUDE_new_solutions_introduced_l427_42797


namespace NUMINAMATH_CALUDE_translation_proof_l427_42756

/-- A translation in 2D space. -/
structure Translation (α : Type*) [Add α] where
  dx : α
  dy : α

/-- Apply a translation to a point. -/
def apply_translation {α : Type*} [Add α] (t : Translation α) (p : α × α) : α × α :=
  (p.1 + t.dx, p.2 + t.dy)

theorem translation_proof :
  let t : Translation ℝ := { dx := 2, dy := -2 }
  let A : ℝ × ℝ := (-1, 4)
  let B : ℝ × ℝ := (2, 1)
  let C : ℝ × ℝ := (1, 2)
  let D : ℝ × ℝ := (4, -1)
  apply_translation t A = C →
  apply_translation t B = D := by
  sorry

end NUMINAMATH_CALUDE_translation_proof_l427_42756


namespace NUMINAMATH_CALUDE_ball_probability_l427_42743

theorem ball_probability (total : ℕ) (white green yellow red purple : ℕ) 
  (h_total : total = 60)
  (h_white : white = 22)
  (h_green : green = 18)
  (h_yellow : yellow = 17)
  (h_red : red = 3)
  (h_purple : purple = 1)
  (h_sum : white + green + yellow + red + purple = total) :
  (white + green + yellow : ℚ) / total = 14 / 15 := by
sorry

end NUMINAMATH_CALUDE_ball_probability_l427_42743


namespace NUMINAMATH_CALUDE_smallest_sum_M_N_l427_42788

/-- Alice's transformation function -/
def aliceTransform (x : ℕ) : ℕ := 3 * x + 2

/-- Bob's transformation function -/
def bobTransform (x : ℕ) : ℕ := 2 * x + 27

/-- Alice's board after 4 moves -/
def aliceFourMoves (M : ℕ) : ℕ := aliceTransform (aliceTransform (aliceTransform (aliceTransform M)))

/-- Bob's board after 4 moves -/
def bobFourMoves (N : ℕ) : ℕ := bobTransform (bobTransform (bobTransform (bobTransform N)))

/-- The theorem stating the smallest sum of M and N -/
theorem smallest_sum_M_N : 
  ∃ (M N : ℕ), 
    M > 0 ∧ N > 0 ∧
    aliceFourMoves M = bobFourMoves N ∧
    (∀ (M' N' : ℕ), M' > 0 → N' > 0 → aliceFourMoves M' = bobFourMoves N' → M + N ≤ M' + N') ∧
    M + N = 10 := by
  sorry

end NUMINAMATH_CALUDE_smallest_sum_M_N_l427_42788


namespace NUMINAMATH_CALUDE_no_rational_solution_for_odd_coeff_quadratic_l427_42704

theorem no_rational_solution_for_odd_coeff_quadratic 
  (a b c : ℤ) 
  (ha : Odd a) 
  (hb : Odd b) 
  (hc : Odd c) : 
  ¬ ∃ (x : ℚ), a * x^2 + b * x + c = 0 := by
  sorry

end NUMINAMATH_CALUDE_no_rational_solution_for_odd_coeff_quadratic_l427_42704


namespace NUMINAMATH_CALUDE_birthday_square_l427_42707

theorem birthday_square (x y : ℕ+) (h1 : 40000 + 1000 * x + 100 * y + 29 < 100000) : 
  ∃ (T : ℕ), T = 2379 ∧ T^2 = 40000 + 1000 * x + 100 * y + 29 := by
  sorry

end NUMINAMATH_CALUDE_birthday_square_l427_42707


namespace NUMINAMATH_CALUDE_green_socks_count_l427_42722

theorem green_socks_count (total : ℕ) (white : ℕ) (blue : ℕ) (red : ℕ) (green : ℕ) :
  total = 900 ∧
  white = total / 3 ∧
  blue = total / 4 ∧
  red = total / 5 ∧
  green = total - (white + blue + red) →
  green = 195 := by
sorry

end NUMINAMATH_CALUDE_green_socks_count_l427_42722


namespace NUMINAMATH_CALUDE_terrell_hike_distance_l427_42724

/-- The total distance hiked by Terrell over two days -/
def total_distance (saturday_distance sunday_distance : ℝ) : ℝ :=
  saturday_distance + sunday_distance

/-- Theorem stating that Terrell's total hiking distance is 9.8 miles -/
theorem terrell_hike_distance :
  total_distance 8.2 1.6 = 9.8 := by
  sorry

end NUMINAMATH_CALUDE_terrell_hike_distance_l427_42724


namespace NUMINAMATH_CALUDE_rhombus_side_length_l427_42715

/-- A rhombus with an inscribed circle of radius 2, where the diagonal divides the rhombus into two equilateral triangles, has a side length of 8√3/3. -/
theorem rhombus_side_length (r : ℝ) (s : ℝ) :
  r = 2 →  -- The radius of the inscribed circle is 2
  s > 0 →  -- The side length is positive
  s^2 = (s/2)^2 + 16 →  -- From the diagonal relationship
  s * 4 = (s * s) / 2 →  -- Area equality
  s = 8 * Real.sqrt 3 / 3 :=
by sorry

end NUMINAMATH_CALUDE_rhombus_side_length_l427_42715


namespace NUMINAMATH_CALUDE_f_properties_l427_42759

noncomputable def f (x : ℝ) : ℝ := Real.sqrt 3 * Real.cos (2 * Real.pi - x) - Real.cos (Real.pi / 2 + x) + 1

theorem f_properties :
  (∀ x, -1 ≤ f x ∧ f x ≤ 3) ∧
  (∀ k : ℤ, ∀ x, -5 * Real.pi / 6 + 2 * k * Real.pi ≤ x ∧ x ≤ Real.pi / 6 + 2 * k * Real.pi → 
    ∀ y, x ≤ y → f x ≤ f y) ∧
  (∀ α, f α = 13 / 5 → Real.pi / 6 < α ∧ α < 2 * Real.pi / 3 → 
    Real.cos (2 * α) = (7 - 24 * Real.sqrt 3) / 50) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l427_42759


namespace NUMINAMATH_CALUDE_arthurs_fitness_routine_l427_42718

/-- The expected number of chocolate balls eaten during Arthur's fitness routine -/
def expected_chocolate_balls (n : ℕ) : ℝ :=
  if n < 2 then 0 else 1

/-- Arthur's fitness routine on Édes Street -/
theorem arthurs_fitness_routine (n : ℕ) (h : n ≥ 2) :
  expected_chocolate_balls n = 1 := by
  sorry

#check arthurs_fitness_routine

end NUMINAMATH_CALUDE_arthurs_fitness_routine_l427_42718


namespace NUMINAMATH_CALUDE_quadratic_roots_range_l427_42729

theorem quadratic_roots_range (a : ℝ) : 
  (∃ x₁ x₂ : ℝ, x₁ < 2 ∧ 2 < x₂ ∧ 
   x₁^2 + 2*a*x₁ - 9 = 0 ∧ 
   x₂^2 + 2*a*x₂ - 9 = 0) → 
  a < 5/4 := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_range_l427_42729


namespace NUMINAMATH_CALUDE_opposite_of_negative_nine_l427_42786

theorem opposite_of_negative_nine : 
  (-(- 9 : ℤ)) = (9 : ℤ) := by sorry

end NUMINAMATH_CALUDE_opposite_of_negative_nine_l427_42786
