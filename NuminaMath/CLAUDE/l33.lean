import Mathlib

namespace NUMINAMATH_CALUDE_nine_hundred_in_column_B_l33_3355

/-- The column type representing the six columns A, B, C, D, E, F -/
inductive Column
| A | B | C | D | E | F

/-- The function that determines the column for a given positive integer -/
def column_for_number (n : ℕ) : Column :=
  match (n - 3) % 12 with
  | 0 => Column.A
  | 1 => Column.B
  | 2 => Column.C
  | 3 => Column.D
  | 4 => Column.A
  | 5 => Column.F
  | 6 => Column.E
  | 7 => Column.F
  | 8 => Column.D
  | 9 => Column.C
  | 10 => Column.B
  | 11 => Column.A
  | _ => Column.A  -- This case should never occur

theorem nine_hundred_in_column_B :
  column_for_number 900 = Column.B :=
by sorry

end NUMINAMATH_CALUDE_nine_hundred_in_column_B_l33_3355


namespace NUMINAMATH_CALUDE_smallest_possible_S_l33_3388

/-- The number of faces on each die -/
def num_faces : ℕ := 8

/-- The target sum we're comparing to -/
def target_sum : ℕ := 3000

/-- The function to calculate the smallest possible value of S -/
def smallest_S (n : ℕ) : ℕ := 9 * n - target_sum

/-- The theorem stating the smallest possible value of S -/
theorem smallest_possible_S :
  ∃ (n : ℕ), 
    (n * num_faces ≥ target_sum) ∧ 
    (∀ m : ℕ, m < n → m * num_faces < target_sum) ∧
    (smallest_S n = 375) := by
  sorry

#check smallest_possible_S

end NUMINAMATH_CALUDE_smallest_possible_S_l33_3388


namespace NUMINAMATH_CALUDE_range_of_a_l33_3362

def statement_p (a : ℝ) : Prop :=
  ∀ x, x^2 + (a - 1) * x + a^2 > 0

def statement_q (a : ℝ) : Prop :=
  ∀ x y, x < y → (2 * a^2 - a)^x < (2 * a^2 - a)^y

theorem range_of_a :
  ∀ a : ℝ, (statement_p a ∨ statement_q a) ∧ ¬(statement_p a ∧ statement_q a) →
    (1/3 < a ∧ a ≤ 1) ∨ (-1 ≤ a ∧ a < -1/2) := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l33_3362


namespace NUMINAMATH_CALUDE_marco_strawberries_weight_l33_3316

/-- The weight of Marco's strawberries in pounds -/
def marco_weight : ℝ := 37 - 22

/-- Theorem stating that Marco's strawberries weighed 15 pounds -/
theorem marco_strawberries_weight :
  marco_weight = 15 := by sorry

end NUMINAMATH_CALUDE_marco_strawberries_weight_l33_3316


namespace NUMINAMATH_CALUDE_diesel_tank_capacity_l33_3315

/-- Given the cost of a certain volume of diesel fuel and the cost of a full tank,
    calculate the capacity of the tank in liters. -/
theorem diesel_tank_capacity 
  (fuel_volume : ℝ) 
  (fuel_cost : ℝ) 
  (full_tank_cost : ℝ) 
  (h1 : fuel_volume = 36) 
  (h2 : fuel_cost = 18) 
  (h3 : full_tank_cost = 32) : 
  (full_tank_cost / (fuel_cost / fuel_volume)) = 64 := by
  sorry

#check diesel_tank_capacity

end NUMINAMATH_CALUDE_diesel_tank_capacity_l33_3315


namespace NUMINAMATH_CALUDE_arithmetic_geometric_sequence_l33_3339

theorem arithmetic_geometric_sequence : 
  ∃ (a b c : ℝ), 
    (a > 0 ∧ b > 0 ∧ c > 0) ∧ 
    (b - a = c - b) ∧ 
    (a + b + c = 15) ∧ 
    ((a + 1) * (c + 9) = (b + 3)^2) ∧
    (a = 1 ∧ b = 5 ∧ c = 9) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_geometric_sequence_l33_3339


namespace NUMINAMATH_CALUDE_line_parabola_intersection_l33_3377

theorem line_parabola_intersection (k : ℝ) : 
  (∃! p : ℝ × ℝ, p.2 = k * p.1 + 2 ∧ p.2^2 = 8 * p.1) ↔ (k = 0 ∨ k = 1) :=
sorry

end NUMINAMATH_CALUDE_line_parabola_intersection_l33_3377


namespace NUMINAMATH_CALUDE_calculation_proof_l33_3303

theorem calculation_proof (h1 : 9 + 3/4 = 9.75) (h2 : 975/100 = 9.75) (h3 : 0.142857 = 1/7) :
  4/7 * (9 + 3/4) + 9.75 * 2/7 + 0.142857 * 975/100 = 9.75 := by
  sorry

end NUMINAMATH_CALUDE_calculation_proof_l33_3303


namespace NUMINAMATH_CALUDE_training_cost_calculation_l33_3395

/-- Represents the financial details of a job applicant -/
structure Applicant where
  salary : ℝ
  revenue : ℝ
  trainingMonths : ℕ
  hiringBonus : ℝ

/-- Calculates the net gain for the company from an applicant -/
def netGain (a : Applicant) (trainingCostPerMonth : ℝ) : ℝ :=
  a.revenue - (a.salary + a.hiringBonus + a.trainingMonths * trainingCostPerMonth)

theorem training_cost_calculation (applicant1 applicant2 : Applicant) 
  (h1 : applicant1.salary = 42000)
  (h2 : applicant1.revenue = 93000)
  (h3 : applicant1.trainingMonths = 3)
  (h4 : applicant1.hiringBonus = 0)
  (h5 : applicant2.salary = 45000)
  (h6 : applicant2.revenue = 92000)
  (h7 : applicant2.trainingMonths = 0)
  (h8 : applicant2.hiringBonus = 0.01 * applicant2.salary)
  (h9 : ∃ (trainingCostPerMonth : ℝ), 
    netGain applicant1 trainingCostPerMonth - netGain applicant2 0 = 850 ∨
    netGain applicant2 0 - netGain applicant1 trainingCostPerMonth = 850) :
  ∃ (trainingCostPerMonth : ℝ), trainingCostPerMonth = 17866.67 := by
  sorry

end NUMINAMATH_CALUDE_training_cost_calculation_l33_3395


namespace NUMINAMATH_CALUDE_largest_divisor_of_consecutive_odd_product_l33_3397

theorem largest_divisor_of_consecutive_odd_product (n : ℕ) (h : Even n) (h' : n > 0) :
  ∃ (k : ℕ), k > 15 → ¬(∀ (m : ℕ), Even m → m > 0 →
    k ∣ (m + 1) * (m + 3) * (m + 5) * (m + 7) * (m + 9)) :=
by sorry

end NUMINAMATH_CALUDE_largest_divisor_of_consecutive_odd_product_l33_3397


namespace NUMINAMATH_CALUDE_min_value_of_function_l33_3385

theorem min_value_of_function (x : ℝ) (h : x > 0) : 
  9 * x + 1 / (x^3) ≥ 10 ∧ ∃ y > 0, 9 * y + 1 / (y^3) = 10 :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_function_l33_3385


namespace NUMINAMATH_CALUDE_routes_8x5_grid_l33_3326

/-- The number of routes on a grid from (0,0) to (m,n) where only right and up movements are allowed -/
def numRoutes (m n : ℕ) : ℕ := Nat.choose (m + n) n

/-- The theorem stating that the number of routes on an 8x5 grid is 12870 -/
theorem routes_8x5_grid : numRoutes 8 5 = 12870 := by sorry

end NUMINAMATH_CALUDE_routes_8x5_grid_l33_3326


namespace NUMINAMATH_CALUDE_fixed_point_of_f_l33_3308

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^(x - 2) + 2

theorem fixed_point_of_f (a : ℝ) (h1 : a > 0) (h2 : a ≠ 1) :
  f a 2 = 2 ∧ f a 2 = 3 :=
by sorry

end NUMINAMATH_CALUDE_fixed_point_of_f_l33_3308


namespace NUMINAMATH_CALUDE_kelly_games_theorem_l33_3369

/-- The number of Nintendo games Kelly needs to give away to have 12 left -/
def games_to_give_away (initial_games : ℕ) (desired_games : ℕ) : ℕ :=
  initial_games - desired_games

theorem kelly_games_theorem :
  let initial_nintendo_games : ℕ := 20
  let desired_nintendo_games : ℕ := 12
  games_to_give_away initial_nintendo_games desired_nintendo_games = 8 := by
  sorry

end NUMINAMATH_CALUDE_kelly_games_theorem_l33_3369


namespace NUMINAMATH_CALUDE_division_problem_l33_3374

theorem division_problem (a b q : ℕ) (h1 : a - b = 1365) (h2 : a = 1637) (h3 : a = b * q + 5) : q = 6 := by
  sorry

end NUMINAMATH_CALUDE_division_problem_l33_3374


namespace NUMINAMATH_CALUDE_investment_proof_l33_3356

/-- Compound interest calculation -/
def compound_interest (principal : ℝ) (rate : ℝ) (time : ℕ) : ℝ :=
  principal * (1 + rate) ^ time

/-- Proof of the investment problem -/
theorem investment_proof :
  let principal : ℝ := 315.84
  let rate : ℝ := 0.12
  let time : ℕ := 6
  let final_value : ℝ := 635.48
  abs (compound_interest principal rate time - final_value) < 0.01 := by
sorry


end NUMINAMATH_CALUDE_investment_proof_l33_3356


namespace NUMINAMATH_CALUDE_polygon_missing_angle_l33_3352

theorem polygon_missing_angle (n : ℕ) (sum_n_minus_1 : ℝ) (h1 : n > 2) (h2 : sum_n_minus_1 = 2843) : 
  (n - 2) * 180 - sum_n_minus_1 = 37 := by
  sorry

end NUMINAMATH_CALUDE_polygon_missing_angle_l33_3352


namespace NUMINAMATH_CALUDE_f_is_increasing_l33_3378

-- Define the function f
def f (x : ℝ) : ℝ := x^3 - 3*x^2 + 6*x

-- Theorem statement
theorem f_is_increasing : ∀ x : ℝ, Monotone f := by sorry

end NUMINAMATH_CALUDE_f_is_increasing_l33_3378


namespace NUMINAMATH_CALUDE_consecutive_integers_around_sqrt_13_l33_3384

theorem consecutive_integers_around_sqrt_13 (a b : ℤ) :
  (b = a + 1) → (a < Real.sqrt 13) → (Real.sqrt 13 < b) → (a + b = 7) := by
  sorry

end NUMINAMATH_CALUDE_consecutive_integers_around_sqrt_13_l33_3384


namespace NUMINAMATH_CALUDE_mary_cut_roses_l33_3353

/-- The number of roses Mary cut from her garden -/
def roses_cut (initial final : ℕ) : ℕ := final - initial

theorem mary_cut_roses : roses_cut 6 16 = 10 := by sorry

end NUMINAMATH_CALUDE_mary_cut_roses_l33_3353


namespace NUMINAMATH_CALUDE_grape_juice_mixture_l33_3318

theorem grape_juice_mixture (initial_volume : ℝ) (added_volume : ℝ) (final_percentage : ℝ) :
  initial_volume = 40 →
  added_volume = 10 →
  final_percentage = 0.36 →
  let final_volume := initial_volume + added_volume
  let initial_percentage := (final_percentage * final_volume - added_volume) / initial_volume
  initial_percentage = 0.2 := by
sorry

end NUMINAMATH_CALUDE_grape_juice_mixture_l33_3318


namespace NUMINAMATH_CALUDE_zeros_before_first_nonzero_of_fraction_l33_3380

/-- The number of zeros after the decimal point and before the first non-zero digit
    in the decimal representation of 1/(2^3 * 5^6) -/
def zeros_before_first_nonzero : ℕ :=
  6

/-- The fraction we're considering -/
def fraction : ℚ :=
  1 / (2^3 * 5^6)

theorem zeros_before_first_nonzero_of_fraction :
  zeros_before_first_nonzero = 
    (fraction.den.factors.count 2 + fraction.den.factors.count 5).min
      (fraction.den.factors.count 5) :=
by
  sorry

end NUMINAMATH_CALUDE_zeros_before_first_nonzero_of_fraction_l33_3380


namespace NUMINAMATH_CALUDE_green_shirt_pairs_l33_3389

theorem green_shirt_pairs (red_students green_students total_students total_pairs red_red_pairs : ℕ)
  (h1 : red_students = 70)
  (h2 : green_students = 94)
  (h3 : total_students = red_students + green_students)
  (h4 : total_pairs = 82)
  (h5 : red_red_pairs = 28)
  : ∃ green_green_pairs : ℕ, green_green_pairs = 40 ∧
    green_green_pairs = total_pairs - red_red_pairs - (red_students - 2 * red_red_pairs) := by
  sorry

end NUMINAMATH_CALUDE_green_shirt_pairs_l33_3389


namespace NUMINAMATH_CALUDE_hyperbola_ellipse_dot_product_l33_3365

-- Define the hyperbola C'
def hyperbola_C' (x y : ℝ) : Prop :=
  x^2 / 16 - y^2 / 9 = 1

-- Define the ellipse M
def ellipse_M (x y : ℝ) : Prop :=
  x^2 / 25 + y^2 / 9 = 1

-- Define the dot product of AP and BP
def AP_dot_BP (x y : ℝ) : ℝ :=
  (x + 1) * (x - 1) + y * y

-- Define the range of AP⋅BP
def range_AP_dot_BP : Set ℝ :=
  {z : ℝ | 191/34 ≤ z ∧ z ≤ 24}

-- Theorem statement
theorem hyperbola_ellipse_dot_product :
  -- Conditions
  (∀ x y : ℝ, 3*x = 4*y ∨ 3*x = -4*y → ¬(hyperbola_C' x y)) →  -- Asymptotes
  (hyperbola_C' 5 (9/4)) →  -- Hyperbola passes through (5, 9/4)
  (∃ x₀ : ℝ, x₀ > 0 ∧ ellipse_M x₀ 0 ∧ hyperbola_C' x₀ 0) →  -- Shared focus/vertex
  (∀ x y : ℝ, ellipse_M x y → x ≤ 5 ∧ y ≤ 3) →  -- Bounds on ellipse
  -- Conclusions
  (∀ x y : ℝ, hyperbola_C' x y ↔ x^2 / 16 - y^2 / 9 = 1) ∧
  (∀ x y : ℝ, ellipse_M x y ↔ x^2 / 25 + y^2 / 9 = 1) ∧
  (∀ x y : ℝ, ellipse_M x y ∧ x ≥ 0 → AP_dot_BP x y ∈ range_AP_dot_BP) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_ellipse_dot_product_l33_3365


namespace NUMINAMATH_CALUDE_min_pizzas_for_johns_car_l33_3307

/-- Calculates the minimum number of pizzas needed to recover car cost -/
def min_pizzas_to_recover_cost (car_cost : ℕ) (earnings_per_pizza : ℕ) (expenses_per_pizza : ℕ) : ℕ :=
  ((car_cost + (earnings_per_pizza - expenses_per_pizza - 1)) / (earnings_per_pizza - expenses_per_pizza))

/-- Theorem: Given the specified conditions, the minimum number of pizzas to recover car cost is 1667 -/
theorem min_pizzas_for_johns_car : 
  min_pizzas_to_recover_cost 5000 10 7 = 1667 := by
  sorry

#eval min_pizzas_to_recover_cost 5000 10 7

end NUMINAMATH_CALUDE_min_pizzas_for_johns_car_l33_3307


namespace NUMINAMATH_CALUDE_composite_product_quotient_l33_3359

/-- The first ten positive composite integers -/
def first_ten_composites : List ℕ := [4, 6, 8, 9, 10, 12, 14, 15, 16, 18]

/-- The product of the first five positive composite integers -/
def product_first_five : ℕ := (first_ten_composites.take 5).prod

/-- The product of the next five positive composite integers -/
def product_next_five : ℕ := (first_ten_composites.drop 5).prod

/-- Theorem stating that the quotient of the product of the first five positive composite integers
    divided by the product of the next five composite integers equals 1/42 -/
theorem composite_product_quotient :
  (product_first_five : ℚ) / (product_next_five : ℚ) = 1 / 42 := by
  sorry

end NUMINAMATH_CALUDE_composite_product_quotient_l33_3359


namespace NUMINAMATH_CALUDE_average_fish_caught_l33_3309

def fish_caught (person : String) : ℕ :=
  match person with
  | "Aang" => 7
  | "Sokka" => 5
  | "Toph" => 12
  | _ => 0

def people : List String := ["Aang", "Sokka", "Toph"]

theorem average_fish_caught :
  (people.map fish_caught).sum / people.length = 8 := by
  sorry

end NUMINAMATH_CALUDE_average_fish_caught_l33_3309


namespace NUMINAMATH_CALUDE_quadratic_always_nonnegative_l33_3360

theorem quadratic_always_nonnegative (x y : ℝ) : x^2 + x*y + y^2 ≥ 0 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_always_nonnegative_l33_3360


namespace NUMINAMATH_CALUDE_laptop_price_proof_l33_3313

/-- The sticker price of the laptop -/
def stickerPrice : ℝ := 750

/-- The price at Store A after discount and rebate -/
def storePriceA (x : ℝ) : ℝ := 0.8 * x - 100

/-- The price at Store B after discount -/
def storePriceB (x : ℝ) : ℝ := 0.7 * x

/-- Theorem stating that the sticker price satisfies the given conditions -/
theorem laptop_price_proof :
  storePriceB stickerPrice - storePriceA stickerPrice = 25 :=
sorry

end NUMINAMATH_CALUDE_laptop_price_proof_l33_3313


namespace NUMINAMATH_CALUDE_misread_weight_l33_3367

theorem misread_weight (n : ℕ) (initial_avg correct_avg correct_weight : ℝ) :
  n = 20 →
  initial_avg = 58.4 →
  correct_avg = 58.9 →
  correct_weight = 66 →
  ∃ (misread_weight : ℝ),
    n * initial_avg + (correct_weight - misread_weight) = n * correct_avg ∧
    misread_weight = 56 :=
by sorry

end NUMINAMATH_CALUDE_misread_weight_l33_3367


namespace NUMINAMATH_CALUDE_prime_divisor_property_l33_3379

theorem prime_divisor_property (n k : ℕ) (h1 : n > 1) 
  (h2 : ∀ d : ℕ, d ∣ n → (d + k) ∣ n ∨ (d - k) ∣ n) : 
  Nat.Prime n :=
sorry

end NUMINAMATH_CALUDE_prime_divisor_property_l33_3379


namespace NUMINAMATH_CALUDE_complex_equation_solution_l33_3393

theorem complex_equation_solution : 
  ∃ (a b c d e : ℤ),
    (2 * (2 : ℝ)^(2/3) + (2 : ℝ)^(1/3) * a + 2 * b + (2 : ℝ)^(2/3) * c + (2 : ℝ)^(1/3) * d + e = 0) ∧
    (25 * (Complex.I * Real.sqrt 5) + 25 * a - 5 * (Complex.I * Real.sqrt 5) * b - 5 * c + (Complex.I * Real.sqrt 5) * d + e = 0) ∧
    (abs (a + b + c + d + e) = 7) :=
by sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l33_3393


namespace NUMINAMATH_CALUDE_cannot_distinguish_normal_l33_3324

/-- Represents the three types of people on the island -/
inductive PersonType
  | Knight
  | Liar
  | Normal

/-- Represents a statement that can be true or false -/
structure Statement where
  content : Prop

/-- A function that determines whether a given person type would make a given statement -/
def wouldMakeStatement (personType : PersonType) (statement : Statement) : Prop :=
  match personType with
  | PersonType.Knight => statement.content
  | PersonType.Liar => ¬statement.content
  | PersonType.Normal => True

/-- The main theorem stating that it's impossible to distinguish a normal person from a knight or liar with any finite number of statements -/
theorem cannot_distinguish_normal (n : ℕ) :
  ∃ (statements : Fin n → Statement),
    (∀ i, wouldMakeStatement PersonType.Normal (statements i)) ∧
    ((∀ i, wouldMakeStatement PersonType.Knight (statements i)) ∨
     (∀ i, wouldMakeStatement PersonType.Liar (statements i))) :=
sorry

end NUMINAMATH_CALUDE_cannot_distinguish_normal_l33_3324


namespace NUMINAMATH_CALUDE_eight_paths_A_to_C_l33_3373

/-- Represents a simple directed graph with four nodes -/
structure DirectedGraph :=
  (paths_A_to_B : ℕ)
  (paths_B_to_C : ℕ)
  (paths_B_to_D : ℕ)
  (paths_D_to_C : ℕ)

/-- Calculates the total number of paths from A to C -/
def total_paths_A_to_C (g : DirectedGraph) : ℕ :=
  g.paths_A_to_B * (g.paths_B_to_C + g.paths_B_to_D * g.paths_D_to_C)

/-- Theorem stating that for the given graph configuration, there are 8 paths from A to C -/
theorem eight_paths_A_to_C :
  ∃ (g : DirectedGraph),
    g.paths_A_to_B = 2 ∧
    g.paths_B_to_C = 3 ∧
    g.paths_B_to_D = 1 ∧
    g.paths_D_to_C = 1 ∧
    total_paths_A_to_C g = 8 :=
by sorry

end NUMINAMATH_CALUDE_eight_paths_A_to_C_l33_3373


namespace NUMINAMATH_CALUDE_square_difference_theorem_l33_3376

theorem square_difference_theorem (x y : ℝ) 
  (sum_eq : x + y = 8) 
  (diff_eq : x - y = 4) : 
  x^2 - y^2 = 32 := by
sorry

end NUMINAMATH_CALUDE_square_difference_theorem_l33_3376


namespace NUMINAMATH_CALUDE_parabola_properties_l33_3319

-- Define the parabola function
def parabola (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

-- Define the theorem
theorem parabola_properties (a b c : ℝ) :
  (parabola a b c (-2) = 0) →
  (parabola a b c (-1) = 4) →
  (parabola a b c 0 = 6) →
  (parabola a b c 1 = 6) →
  (a < 0) ∧
  (∀ x, parabola a b c x ≤ parabola a b c (1/2)) ∧
  (parabola a b c (1/2) = 25/4) := by
  sorry

end NUMINAMATH_CALUDE_parabola_properties_l33_3319


namespace NUMINAMATH_CALUDE_intersection_implies_a_value_l33_3394

theorem intersection_implies_a_value (a : ℝ) : 
  let M : Set ℝ := {1, 2, a^2 - 3*a - 1}
  let N : Set ℝ := {-1, a, 3}
  (M ∩ N = {3}) → a = 4 := by
sorry

end NUMINAMATH_CALUDE_intersection_implies_a_value_l33_3394


namespace NUMINAMATH_CALUDE_percentage_calculation_l33_3349

theorem percentage_calculation (P : ℝ) : 
  (0.16 * (P / 100) * 93.75 = 6) → P = 40 := by
  sorry

end NUMINAMATH_CALUDE_percentage_calculation_l33_3349


namespace NUMINAMATH_CALUDE_tomato_price_equation_l33_3306

/-- The original price per pound of tomatoes -/
def P : ℝ := sorry

/-- The selling price per pound of remaining tomatoes -/
def selling_price : ℝ := 0.968888888888889

/-- The proportion of tomatoes that were not ruined -/
def remaining_proportion : ℝ := 0.9

/-- The profit percentage as a decimal -/
def profit_percentage : ℝ := 0.09

theorem tomato_price_equation : 
  (1 + profit_percentage) * P = selling_price * remaining_proportion := by sorry

end NUMINAMATH_CALUDE_tomato_price_equation_l33_3306


namespace NUMINAMATH_CALUDE_road_signs_ratio_l33_3333

/-- Represents the number of road signs at each intersection -/
structure RoadSigns where
  s1 : ℕ  -- First intersection
  s2 : ℕ  -- Second intersection
  s3 : ℕ  -- Third intersection
  s4 : ℕ  -- Fourth intersection

/-- Theorem stating the ratio of road signs at the third to second intersection -/
theorem road_signs_ratio 
  (signs : RoadSigns) 
  (h1 : signs.s1 = 40)
  (h2 : signs.s2 = signs.s1 + signs.s1 / 4)
  (h3 : signs.s4 = signs.s3 - 20)
  (h4 : signs.s1 + signs.s2 + signs.s3 + signs.s4 = 270) :
  signs.s3 / signs.s2 = 2 := by
  sorry

#eval (100 : ℚ) / 50  -- Expected output: 2

end NUMINAMATH_CALUDE_road_signs_ratio_l33_3333


namespace NUMINAMATH_CALUDE_solution_set_inequality_holds_l33_3348

-- Define the function f
def f (x : ℝ) : ℝ := |x + 1| + |x - 1| - 1

-- Theorem 1: Solution set of f(x) ≤ x + 1
theorem solution_set (x : ℝ) : f x ≤ x + 1 ↔ 0 ≤ x ∧ x ≤ 2 := by
  sorry

-- Theorem 2: 3f(x) ≥ f(2x) for all x
theorem inequality_holds (x : ℝ) : 3 * f x ≥ f (2 * x) := by
  sorry

end NUMINAMATH_CALUDE_solution_set_inequality_holds_l33_3348


namespace NUMINAMATH_CALUDE_sin_180_degrees_l33_3336

theorem sin_180_degrees : Real.sin (π) = 0 := by
  sorry

end NUMINAMATH_CALUDE_sin_180_degrees_l33_3336


namespace NUMINAMATH_CALUDE_determinant_trigonometric_matrix_l33_3354

theorem determinant_trigonometric_matrix (α β : Real) :
  let M : Matrix (Fin 3) (Fin 3) Real := ![
    ![Real.sin α * Real.sin β, Real.sin α * Real.cos β, Real.cos α],
    ![Real.cos β, -Real.sin β, 0],
    ![Real.cos α * Real.sin β, Real.cos α * Real.cos β, Real.sin α]
  ]
  Matrix.det M = Real.cos (2 * α) := by
  sorry

end NUMINAMATH_CALUDE_determinant_trigonometric_matrix_l33_3354


namespace NUMINAMATH_CALUDE_seventh_root_of_unity_product_l33_3375

theorem seventh_root_of_unity_product (r : ℂ) (h1 : r^7 = 1) (h2 : r ≠ 1) :
  (r - 1) * (r^2 - 1) * (r^3 - 1) * (r^4 - 1) * (r^5 - 1) * (r^6 - 1) = 8 := by
  sorry

end NUMINAMATH_CALUDE_seventh_root_of_unity_product_l33_3375


namespace NUMINAMATH_CALUDE_intersection_complement_M_and_N_l33_3331

def M : Set ℝ := {x : ℝ | x^2 - 3*x - 4 ≥ 0}
def N : Set ℝ := {x : ℝ | 1 < x ∧ x < 5}

theorem intersection_complement_M_and_N :
  (Mᶜ ∩ N) = {x : ℝ | 1 < x ∧ x < 4} :=
sorry

end NUMINAMATH_CALUDE_intersection_complement_M_and_N_l33_3331


namespace NUMINAMATH_CALUDE_consecutive_circle_selections_l33_3335

/-- Represents the arrangement of circles in the figure -/
structure CircleArrangement :=
  (total_circles : Nat)
  (long_side_rows : Nat)
  (perpendicular_rows : Nat)

/-- Calculates the number of ways to choose three consecutive circles along the long side -/
def long_side_selections (arr : CircleArrangement) : Nat :=
  (arr.long_side_rows * (arr.long_side_rows + 1)) / 2

/-- Calculates the number of ways to choose three consecutive circles along one perpendicular direction -/
def perpendicular_selections (arr : CircleArrangement) : Nat :=
  (3 * arr.perpendicular_rows + (arr.perpendicular_rows * (arr.perpendicular_rows - 1)) / 2)

/-- The main theorem stating the total number of ways to choose three consecutive circles -/
theorem consecutive_circle_selections (arr : CircleArrangement) 
  (h1 : arr.total_circles = 33)
  (h2 : arr.long_side_rows = 6)
  (h3 : arr.perpendicular_rows = 6) :
  long_side_selections arr + 2 * perpendicular_selections arr = 57 := by
  sorry


end NUMINAMATH_CALUDE_consecutive_circle_selections_l33_3335


namespace NUMINAMATH_CALUDE_units_digit_problem_l33_3314

theorem units_digit_problem : (8 * 18 * 1988 - 8^3) % 10 = 0 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_problem_l33_3314


namespace NUMINAMATH_CALUDE_product_remainder_mod_five_l33_3381

theorem product_remainder_mod_five : (1483 * 1773 * 1827 * 2001) % 5 = 3 := by
  sorry

end NUMINAMATH_CALUDE_product_remainder_mod_five_l33_3381


namespace NUMINAMATH_CALUDE_max_area_triangle_l33_3342

noncomputable def m (x : ℝ) : ℝ × ℝ := (2 * Real.cos x, Real.sin x + Real.cos x)

noncomputable def n (x : ℝ) : ℝ × ℝ := (Real.sqrt 3 * Real.sin x, Real.sin x - Real.cos x)

noncomputable def f (x : ℝ) : ℝ := (m x).1 * (n x).1 + (m x).2 * (n x).2

theorem max_area_triangle (C : ℝ) (a b c : ℝ) (hf : f C = 2) (hc : c = Real.sqrt 3) :
  ∃ S : ℝ, S ≤ (3 * Real.sqrt 3) / 4 ∧ 
  (∀ S' : ℝ, S' = 1/2 * a * b * Real.sin C → S' ≤ S) :=
sorry

end NUMINAMATH_CALUDE_max_area_triangle_l33_3342


namespace NUMINAMATH_CALUDE_composite_function_evaluation_l33_3361

-- Define the functions f and g
def f (x : ℝ) : ℝ := 2 * x + 4
def g (x : ℝ) : ℝ := 5 * x + 2

-- State the theorem
theorem composite_function_evaluation :
  g (f (g 3)) = 192 := by
  sorry

end NUMINAMATH_CALUDE_composite_function_evaluation_l33_3361


namespace NUMINAMATH_CALUDE_probability_two_red_balls_l33_3370

def bag_total_balls : ℕ := 4
def red_balls : ℕ := 2
def white_balls : ℕ := 2
def drawn_balls : ℕ := 2

theorem probability_two_red_balls : 
  (Nat.choose red_balls drawn_balls : ℚ) / (Nat.choose bag_total_balls drawn_balls) = 1 / 6 :=
sorry

end NUMINAMATH_CALUDE_probability_two_red_balls_l33_3370


namespace NUMINAMATH_CALUDE_mixture_composition_l33_3321

/-- Represents a solution with percentages of materials A and B -/
structure Solution :=
  (percentA : ℝ)
  (percentB : ℝ)
  (sum_to_100 : percentA + percentB = 100)

/-- Represents a mixture of two solutions -/
structure Mixture :=
  (solutionX : Solution)
  (solutionY : Solution)
  (finalPercentA : ℝ)

theorem mixture_composition 
  (m : Mixture)
  (hX : m.solutionX.percentA = 20 ∧ m.solutionX.percentB = 80)
  (hY : m.solutionY.percentA = 30 ∧ m.solutionY.percentB = 70)
  (hFinal : m.finalPercentA = 22) :
  100 - m.finalPercentA = 78 := by
  sorry

end NUMINAMATH_CALUDE_mixture_composition_l33_3321


namespace NUMINAMATH_CALUDE_crazy_silly_school_series_l33_3392

theorem crazy_silly_school_series (total_books : ℕ) (books_read : ℕ) (movies_watched : ℕ) :
  total_books = 11 →
  books_read = 7 →
  movies_watched = 21 →
  movies_watched = books_read + 14 →
  ∃ (total_movies : ℕ), total_movies = 7 ∧ total_movies = movies_watched - 14 :=
by
  sorry

end NUMINAMATH_CALUDE_crazy_silly_school_series_l33_3392


namespace NUMINAMATH_CALUDE_complement_of_union_l33_3399

def U : Set ℝ := Set.univ
def A : Set ℝ := {x | x < 1}
def B : Set ℝ := {x | x ≥ 2}

theorem complement_of_union :
  (A ∪ B)ᶜ = {x : ℝ | 1 ≤ x ∧ x < 2} := by sorry

end NUMINAMATH_CALUDE_complement_of_union_l33_3399


namespace NUMINAMATH_CALUDE_cookies_problem_l33_3382

theorem cookies_problem (mona jasmine rachel : ℕ) 
  (h1 : jasmine = mona - 5)
  (h2 : rachel = jasmine + 10)
  (h3 : mona + jasmine + rachel = 60) :
  mona = 20 := by
sorry

end NUMINAMATH_CALUDE_cookies_problem_l33_3382


namespace NUMINAMATH_CALUDE_min_cost_tree_purchase_l33_3338

/-- Represents the cost and quantity of trees --/
structure TreePurchase where
  cypress_price : ℕ
  pine_price : ℕ
  cypress_count : ℕ
  pine_count : ℕ

/-- The conditions of the tree purchasing problem --/
def tree_problem (p : TreePurchase) : Prop :=
  2 * p.cypress_price + 3 * p.pine_price = 850 ∧
  3 * p.cypress_price + 2 * p.pine_price = 900 ∧
  p.cypress_count + p.pine_count = 80 ∧
  p.cypress_count ≥ 2 * p.pine_count

/-- The total cost of a tree purchase --/
def total_cost (p : TreePurchase) : ℕ :=
  p.cypress_price * p.cypress_count + p.pine_price * p.pine_count

/-- The theorem stating the minimum cost and optimal purchase --/
theorem min_cost_tree_purchase :
  ∃ (p : TreePurchase), tree_problem p ∧
    total_cost p = 14700 ∧
    p.cypress_count = 54 ∧
    p.pine_count = 26 ∧
    (∀ (q : TreePurchase), tree_problem q → total_cost q ≥ total_cost p) :=
by sorry

end NUMINAMATH_CALUDE_min_cost_tree_purchase_l33_3338


namespace NUMINAMATH_CALUDE_overall_percentage_calculation_l33_3344

theorem overall_percentage_calculation (grade1 grade2 grade3 : ℚ) :
  grade1 = 50 / 100 →
  grade2 = 60 / 100 →
  grade3 = 70 / 100 →
  (grade1 + grade2 + grade3) / 3 = 60 / 100 := by
  sorry

end NUMINAMATH_CALUDE_overall_percentage_calculation_l33_3344


namespace NUMINAMATH_CALUDE_yogurt_production_cost_l33_3347

/-- Represents the cost and quantity of an ingredient -/
structure Ingredient where
  cost_per_unit : ℚ
  quantity_per_batch : ℚ

/-- Calculates the total cost for producing a given number of batches -/
def total_cost (milk : Ingredient) (fruit : Ingredient) (num_batches : ℚ) : ℚ :=
  num_batches * (milk.cost_per_unit * milk.quantity_per_batch + 
                 fruit.cost_per_unit * fruit.quantity_per_batch)

theorem yogurt_production_cost :
  let milk : Ingredient := ⟨3/2, 10⟩
  let fruit : Ingredient := ⟨2, 3⟩
  let num_batches : ℚ := 3
  total_cost milk fruit num_batches = 63 := by
  sorry

end NUMINAMATH_CALUDE_yogurt_production_cost_l33_3347


namespace NUMINAMATH_CALUDE_unique_angle_satisfying_conditions_l33_3300

theorem unique_angle_satisfying_conditions :
  ∃! x : ℝ, 0 ≤ x ∧ x < 2 * π ∧ 
    Real.sin x = -(1/2) ∧ Real.cos x = Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_unique_angle_satisfying_conditions_l33_3300


namespace NUMINAMATH_CALUDE_solution_set_inequality_l33_3328

open Set
open Function
open Real

-- Define f as a function from ℝ to ℝ
variable (f : ℝ → ℝ)

-- Define the derivative of f
variable (f' : ℝ → ℝ)

-- Theorem statement
theorem solution_set_inequality 
  (h_domain : ∀ x, x > 0 → DifferentiableAt ℝ f x)
  (h_derivative : ∀ x, x > 0 → deriv f x = f' x)
  (h_inequality : ∀ x, x > 0 → f x > f' x) :
  {x : ℝ | Real.exp (x + 2) * f (x^2 - x) > Real.exp (x^2) * f 2} = 
  Ioo (-1) 0 ∪ Ioo 1 2 := by
sorry

end NUMINAMATH_CALUDE_solution_set_inequality_l33_3328


namespace NUMINAMATH_CALUDE_tent_max_profit_l33_3304

/-- Represents the purchase and sales information for tents --/
structure TentInfo where
  regular_purchase_price : ℝ
  sunshade_purchase_price : ℝ
  regular_selling_price : ℝ
  sunshade_selling_price : ℝ
  total_budget : ℝ

/-- Represents the constraints on tent purchases --/
structure TentConstraints where
  min_regular_tents : ℕ
  regular_not_exceeding_sunshade : Bool

/-- Calculates the maximum profit given tent information and constraints --/
def max_profit (info : TentInfo) (constraints : TentConstraints) : ℝ :=
  sorry

/-- Theorem stating the maximum profit for the given scenario --/
theorem tent_max_profit :
  let info : TentInfo := {
    regular_purchase_price := 150,
    sunshade_purchase_price := 300,
    regular_selling_price := 180,
    sunshade_selling_price := 380,
    total_budget := 9000
  }
  let constraints : TentConstraints := {
    min_regular_tents := 12,
    regular_not_exceeding_sunshade := true
  }
  max_profit info constraints = 2280 := by sorry

end NUMINAMATH_CALUDE_tent_max_profit_l33_3304


namespace NUMINAMATH_CALUDE_min_sum_squares_l33_3302

theorem min_sum_squares (x y z : ℝ) (h : 2*x + 3*y + 4*z = 11) :
  x^2 + y^2 + z^2 ≥ 121/29 :=
by sorry

end NUMINAMATH_CALUDE_min_sum_squares_l33_3302


namespace NUMINAMATH_CALUDE_fourth_term_of_geometric_sequence_l33_3305

/-- Given a geometric sequence with first term a and common ratio r,
    the nth term is given by a * r^(n-1) -/
def geometric_term (a : ℚ) (r : ℚ) (n : ℕ) : ℚ := a * r^(n-1)

/-- The fourth term of a geometric sequence with first term 3 and second term 1/3 is 1/243 -/
theorem fourth_term_of_geometric_sequence :
  let a := 3
  let a₂ := 1/3
  let r := a₂ / a
  geometric_term a r 4 = 1/243 := by sorry

end NUMINAMATH_CALUDE_fourth_term_of_geometric_sequence_l33_3305


namespace NUMINAMATH_CALUDE_andy_wrappers_l33_3371

theorem andy_wrappers (total : ℕ) (max_wrappers : ℕ) (andy_wrappers : ℕ) :
  total = 49 →
  max_wrappers = 15 →
  total = andy_wrappers + max_wrappers →
  andy_wrappers = 34 := by
sorry

end NUMINAMATH_CALUDE_andy_wrappers_l33_3371


namespace NUMINAMATH_CALUDE_ellipse_k_value_l33_3368

/-- An ellipse with equation 4x² + ky² = 4 and a focus at (0, 1) has k = 2 -/
theorem ellipse_k_value (k : ℝ) : 
  (∀ x y : ℝ, 4 * x^2 + k * y^2 = 4) →  -- Ellipse equation
  (0, 1) ∈ {p : ℝ × ℝ | p.1^2 / 1^2 + p.2^2 / (4/k) = 1} →  -- Focus condition
  k = 2 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_k_value_l33_3368


namespace NUMINAMATH_CALUDE_range_of_m_l33_3334

theorem range_of_m : ∃ (a b : ℝ), a = 1 ∧ b = 3 ∧
  ∀ m : ℝ, (∀ x : ℝ, |m - x| < 2 → -1 < x ∧ x < 5) →
  a ≤ m ∧ m ≤ b :=
by sorry

end NUMINAMATH_CALUDE_range_of_m_l33_3334


namespace NUMINAMATH_CALUDE_stewart_farm_ratio_l33_3327

/-- The ratio of sheep to horses at Stewart farm -/
theorem stewart_farm_ratio : 
  ∀ (num_sheep num_horses : ℕ) (food_per_horse total_horse_food : ℕ),
  num_sheep = 24 →
  food_per_horse = 230 →
  total_horse_food = 12880 →
  num_horses * food_per_horse = total_horse_food →
  (num_sheep : ℚ) / (num_horses : ℚ) = 3 / 7 := by
  sorry

end NUMINAMATH_CALUDE_stewart_farm_ratio_l33_3327


namespace NUMINAMATH_CALUDE_geometric_with_arithmetic_subsequence_l33_3372

/-- A geometric sequence with common ratio q -/
def geometric_sequence (a : ℕ → ℝ) (q : ℝ) : Prop :=
  ∀ n : ℕ, a (n + 1) = a n * q

/-- An arithmetic subsequence of a sequence -/
def arithmetic_subsequence (a : ℕ → ℝ) (sub : ℕ → ℕ) (d : ℝ) : Prop :=
  ∀ n : ℕ, a (sub (n + 1)) - a (sub n) = d

/-- The main theorem: if a geometric sequence has an infinite arithmetic subsequence,
    then its common ratio is -1 -/
theorem geometric_with_arithmetic_subsequence
  (a : ℕ → ℝ) (q : ℝ) (sub : ℕ → ℕ) (d : ℝ) (h_ne_one : q ≠ 1) :
  geometric_sequence a q →
  (∃ d, arithmetic_subsequence a sub d) →
  q = -1 := by
sorry

end NUMINAMATH_CALUDE_geometric_with_arithmetic_subsequence_l33_3372


namespace NUMINAMATH_CALUDE_sqrt_three_multiplication_l33_3322

theorem sqrt_three_multiplication : Real.sqrt 3 * (2 * Real.sqrt 3 - 2) = 6 - 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_three_multiplication_l33_3322


namespace NUMINAMATH_CALUDE_fourth_power_of_cube_of_third_smallest_prime_l33_3366

def third_smallest_prime : ℕ := 5

theorem fourth_power_of_cube_of_third_smallest_prime :
  (third_smallest_prime ^ 3) ^ 4 = 244140625 := by
  sorry

end NUMINAMATH_CALUDE_fourth_power_of_cube_of_third_smallest_prime_l33_3366


namespace NUMINAMATH_CALUDE_vector_equation_solution_l33_3337

/-- Given four distinct points P, A, B, C on a plane, prove that if 
    PA + PB + PC = 0 and AB + AC + m * AP = 0, then m = -3 -/
theorem vector_equation_solution (P A B C : EuclideanSpace ℝ (Fin 2)) 
    (h1 : P ≠ A ∧ P ≠ B ∧ P ≠ C ∧ A ≠ B ∧ A ≠ C ∧ B ≠ C)
    (h2 : (A - P) + (B - P) + (C - P) = 0)
    (h3 : ∃ m : ℝ, (B - A) + (C - A) + m • (P - A) = 0) : 
  ∃ m : ℝ, (B - A) + (C - A) + m • (P - A) = 0 ∧ m = -3 := by
  sorry

end NUMINAMATH_CALUDE_vector_equation_solution_l33_3337


namespace NUMINAMATH_CALUDE_cauliflower_area_l33_3310

theorem cauliflower_area (this_year_side : ℕ) (last_year_side : ℕ) 
  (h1 : this_year_side ^ 2 = 12544)
  (h2 : this_year_side ^ 2 = last_year_side ^ 2 + 223) :
  1 = 1 := by
  sorry

end NUMINAMATH_CALUDE_cauliflower_area_l33_3310


namespace NUMINAMATH_CALUDE_complex_fraction_equality_l33_3398

theorem complex_fraction_equality : Complex.I * Complex.I = -1 → (2 : ℂ) / (1 + Complex.I) = 1 - Complex.I := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_equality_l33_3398


namespace NUMINAMATH_CALUDE_min_attempts_for_two_unknown_digits_l33_3383

/-- Represents a phone number with known and unknown digits -/
structure PhoneNumber :=
  (known_digits : Nat)
  (unknown_digits : Nat)
  (total_digits : Nat)
  (h_total : total_digits = known_digits + unknown_digits)

/-- The number of possible combinations for the unknown digits -/
def possible_combinations (pn : PhoneNumber) : Nat :=
  10 ^ pn.unknown_digits

/-- The minimum number of attempts required to guarantee dialing the correct number -/
def min_attempts (pn : PhoneNumber) : Nat :=
  possible_combinations pn

theorem min_attempts_for_two_unknown_digits 
  (pn : PhoneNumber) 
  (h_seven_digits : pn.total_digits = 7) 
  (h_five_known : pn.known_digits = 5) 
  (h_two_unknown : pn.unknown_digits = 2) : 
  min_attempts pn = 100 := by
  sorry

#check min_attempts_for_two_unknown_digits

end NUMINAMATH_CALUDE_min_attempts_for_two_unknown_digits_l33_3383


namespace NUMINAMATH_CALUDE_quadratic_inequality_range_l33_3312

theorem quadratic_inequality_range (a : ℝ) : 
  (¬ ∀ x : ℝ, 4 * x^2 + (a - 2) * x + 1/4 > 0) ↔ 
  (a ≤ 0 ∨ a ≥ 4) := by
sorry

end NUMINAMATH_CALUDE_quadratic_inequality_range_l33_3312


namespace NUMINAMATH_CALUDE_polygon_D_has_largest_area_l33_3350

-- Define the basic shapes
def square_area : ℝ := 1
def isosceles_right_triangle_area : ℝ := 0.5
def parallelogram_area : ℝ := 1

-- Define the polygons
def polygon_A_area : ℝ := 3 * square_area + 2 * isosceles_right_triangle_area
def polygon_B_area : ℝ := 2 * square_area + 4 * isosceles_right_triangle_area
def polygon_C_area : ℝ := square_area + 2 * isosceles_right_triangle_area + parallelogram_area
def polygon_D_area : ℝ := 4 * square_area + parallelogram_area
def polygon_E_area : ℝ := 2 * square_area + 3 * isosceles_right_triangle_area + parallelogram_area

-- Theorem statement
theorem polygon_D_has_largest_area :
  polygon_D_area > polygon_A_area ∧
  polygon_D_area > polygon_B_area ∧
  polygon_D_area > polygon_C_area ∧
  polygon_D_area > polygon_E_area :=
by sorry

end NUMINAMATH_CALUDE_polygon_D_has_largest_area_l33_3350


namespace NUMINAMATH_CALUDE_polynomial_factorization_l33_3396

theorem polynomial_factorization (x : ℝ) : 
  (x + 1) * (x + 2) * (x + 3) * (x + 4) + 1 = (x^2 + 5*x + 5)^2 := by sorry

end NUMINAMATH_CALUDE_polynomial_factorization_l33_3396


namespace NUMINAMATH_CALUDE_wheels_equation_l33_3325

theorem wheels_equation (x y : ℕ) : 2 * x + 4 * y = 66 → y = (33 - x) / 2 :=
by sorry

end NUMINAMATH_CALUDE_wheels_equation_l33_3325


namespace NUMINAMATH_CALUDE_probability_two_black_balls_l33_3363

def total_balls : ℕ := 15
def black_balls : ℕ := 10
def white_balls : ℕ := 5

theorem probability_two_black_balls :
  let p_first_black : ℚ := black_balls / total_balls
  let p_second_black : ℚ := (black_balls - 1) / (total_balls - 1)
  p_first_black * p_second_black = 3 / 7 := by sorry

end NUMINAMATH_CALUDE_probability_two_black_balls_l33_3363


namespace NUMINAMATH_CALUDE_young_inequality_l33_3311

theorem young_inequality (A B p q : ℝ) (hA : A > 0) (hB : B > 0) (hp : p > 0) (hq : q > 0) (hpq : 1/p + 1/q = 1) :
  A^(1/p) * B^(1/q) ≤ A/p + B/q :=
by sorry

end NUMINAMATH_CALUDE_young_inequality_l33_3311


namespace NUMINAMATH_CALUDE_oliver_candy_boxes_l33_3387

theorem oliver_candy_boxes (initial_boxes final_boxes : ℕ) : 
  initial_boxes = 8 → final_boxes = 6 → initial_boxes + final_boxes = 14 :=
by sorry

end NUMINAMATH_CALUDE_oliver_candy_boxes_l33_3387


namespace NUMINAMATH_CALUDE_investment_solution_l33_3351

/-- Represents the investment scenario described in the problem -/
structure Investment where
  total : ℝ
  rate1 : ℝ
  rate2 : ℝ
  years : ℕ
  finalAmount : ℝ

/-- Calculates the final amount after compound interest -/
def compoundInterest (principal : ℝ) (rate : ℝ) (years : ℕ) : ℝ :=
  principal * (1 + rate) ^ years

/-- Theorem stating the solution to the investment problem -/
theorem investment_solution (inv : Investment) 
  (h1 : inv.total = 1500)
  (h2 : inv.rate1 = 0.04)
  (h3 : inv.rate2 = 0.06)
  (h4 : inv.years = 3)
  (h5 : inv.finalAmount = 1824.89) :
  ∃ (x : ℝ), x = 580 ∧ 
    compoundInterest x inv.rate1 inv.years + 
    compoundInterest (inv.total - x) inv.rate2 inv.years = 
    inv.finalAmount := by
  sorry


end NUMINAMATH_CALUDE_investment_solution_l33_3351


namespace NUMINAMATH_CALUDE_percentage_boys_playing_soccer_l33_3346

/-- Proves that the percentage of boys among students playing soccer is 86% -/
theorem percentage_boys_playing_soccer
  (total_students : ℕ)
  (num_boys : ℕ)
  (num_playing_soccer : ℕ)
  (num_girls_not_playing : ℕ)
  (h1 : total_students = 450)
  (h2 : num_boys = 320)
  (h3 : num_playing_soccer = 250)
  (h4 : num_girls_not_playing = 95)
  : (((num_playing_soccer - (total_students - num_boys - num_girls_not_playing)) / num_playing_soccer) : ℚ) = 86 / 100 := by
  sorry


end NUMINAMATH_CALUDE_percentage_boys_playing_soccer_l33_3346


namespace NUMINAMATH_CALUDE_complex_number_problem_l33_3345

def z (b : ℝ) : ℂ := 3 + b * Complex.I

theorem complex_number_problem (b : ℝ) 
  (h : ∃ (y : ℝ), (1 + 3 * Complex.I) * z b = y * Complex.I) :
  z b = 3 + Complex.I ∧ Complex.abs (z b / (2 + Complex.I)) = Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_number_problem_l33_3345


namespace NUMINAMATH_CALUDE_max_stores_visited_l33_3364

/-- Represents the shopping scenario in the town -/
structure ShoppingScenario where
  stores : Nat
  total_visits : Nat
  unique_visitors : Nat
  double_visitors : Nat

/-- Theorem stating the maximum number of stores visited by any single person -/
theorem max_stores_visited (s : ShoppingScenario) 
  (h1 : s.stores = 7)
  (h2 : s.total_visits = 21)
  (h3 : s.unique_visitors = 11)
  (h4 : s.double_visitors = 7)
  (h5 : s.double_visitors ≤ s.unique_visitors)
  (h6 : s.double_visitors * 2 ≤ s.total_visits) :
  ∃ (max_visits : Nat), max_visits = 4 ∧ 
  ∀ (individual_visits : Nat), individual_visits ≤ max_visits :=
by sorry

end NUMINAMATH_CALUDE_max_stores_visited_l33_3364


namespace NUMINAMATH_CALUDE_courtyard_stones_l33_3357

theorem courtyard_stones (stones : ℕ) (trees : ℕ) (birds : ℕ) : 
  trees = 3 * stones →
  birds = 2 * (trees + stones) →
  birds = 400 →
  stones = 40 := by
sorry

end NUMINAMATH_CALUDE_courtyard_stones_l33_3357


namespace NUMINAMATH_CALUDE_normal_distribution_value_l33_3341

/-- For a normal distribution with mean 16.5 and standard deviation 1.5,
    the value that is exactly 2 standard deviations less than the mean is 13.5. -/
theorem normal_distribution_value (μ σ : ℝ) (h1 : μ = 16.5) (h2 : σ = 1.5) :
  μ - 2 * σ = 13.5 := by
  sorry

end NUMINAMATH_CALUDE_normal_distribution_value_l33_3341


namespace NUMINAMATH_CALUDE_garden_ratio_l33_3332

/-- A rectangular garden with given perimeter and length has a specific length-to-width ratio -/
theorem garden_ratio (perimeter length width : ℝ) : 
  perimeter = 300 →
  length = 100 →
  perimeter = 2 * length + 2 * width →
  length / width = 2 := by
  sorry

end NUMINAMATH_CALUDE_garden_ratio_l33_3332


namespace NUMINAMATH_CALUDE_exactly_two_trains_on_time_l33_3323

-- Define the probabilities of each train arriving on time
def P_A : ℝ := 0.8
def P_B : ℝ := 0.7
def P_C : ℝ := 0.9

-- Define the probability of exactly two trains arriving on time
def P_exactly_two : ℝ := 
  P_A * P_B * (1 - P_C) + P_A * (1 - P_B) * P_C + (1 - P_A) * P_B * P_C

-- Theorem statement
theorem exactly_two_trains_on_time : P_exactly_two = 0.398 := by
  sorry

end NUMINAMATH_CALUDE_exactly_two_trains_on_time_l33_3323


namespace NUMINAMATH_CALUDE_square_of_binomial_theorem_l33_3390

-- Define the expressions
def expr_A (x y : ℝ) := (x + y) * (x - y)
def expr_B (x y : ℝ) := (-x + y) * (x - y)
def expr_C (x y : ℝ) := (-x + y) * (-x - y)
def expr_D (x y : ℝ) := (-x + y) * (x + y)

-- Define what it means for an expression to be a square of a binomial
def is_square_of_binomial (f : ℝ → ℝ → ℝ) : Prop :=
  ∃ (g : ℝ → ℝ → ℝ), ∀ x y, f x y = (g x y)^2

-- State the theorem
theorem square_of_binomial_theorem :
  is_square_of_binomial expr_A ∧
  ¬(is_square_of_binomial expr_B) ∧
  is_square_of_binomial expr_C ∧
  is_square_of_binomial expr_D :=
sorry

end NUMINAMATH_CALUDE_square_of_binomial_theorem_l33_3390


namespace NUMINAMATH_CALUDE_grains_per_teaspoon_l33_3329

/-- Represents the number of grains of rice in one cup -/
def grains_per_cup : ℕ := 480

/-- Represents the number of tablespoons in half a cup -/
def tablespoons_per_half_cup : ℕ := 8

/-- Represents the number of teaspoons in one tablespoon -/
def teaspoons_per_tablespoon : ℕ := 3

/-- Theorem stating that there are 10 grains of rice in a teaspoon -/
theorem grains_per_teaspoon :
  (grains_per_cup / (2 * tablespoons_per_half_cup * teaspoons_per_tablespoon)) = 10 := by
  sorry

end NUMINAMATH_CALUDE_grains_per_teaspoon_l33_3329


namespace NUMINAMATH_CALUDE_isabel_toy_cost_l33_3330

theorem isabel_toy_cost (total_money : ℕ) (num_toys : ℕ) (cost_per_toy : ℕ) 
  (h1 : total_money = 14) 
  (h2 : num_toys = 7) 
  (h3 : total_money = num_toys * cost_per_toy) : 
  cost_per_toy = 2 := by
  sorry

end NUMINAMATH_CALUDE_isabel_toy_cost_l33_3330


namespace NUMINAMATH_CALUDE_equation_solution_in_interval_l33_3391

theorem equation_solution_in_interval :
  ∃! x : ℝ, x ∈ (Set.Ioo 0 1) ∧ 3^x + x = 3 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_in_interval_l33_3391


namespace NUMINAMATH_CALUDE_max_sum_divisible_into_two_parts_l33_3301

theorem max_sum_divisible_into_two_parts (S : ℕ) : 
  (∃ (nums : List ℕ), 
    (∀ n ∈ nums, 0 < n ∧ n ≤ 10) ∧ 
    (nums.sum = S) ∧ 
    (∀ (partition : List ℕ × List ℕ), 
      partition.1 ∪ partition.2 = nums → 
      partition.1.sum ≤ 70 ∧ partition.2.sum ≤ 70)) →
  S ≤ 133 :=
by sorry

end NUMINAMATH_CALUDE_max_sum_divisible_into_two_parts_l33_3301


namespace NUMINAMATH_CALUDE_time_to_install_remaining_windows_l33_3358

/-- Calculates the time to install remaining windows -/
def timeToInstallRemaining (totalWindows installedWindows timePerWindow : ℕ) : ℕ :=
  (totalWindows - installedWindows) * timePerWindow

/-- Theorem: Time to install remaining windows is 18 hours -/
theorem time_to_install_remaining_windows :
  timeToInstallRemaining 9 6 6 = 18 := by
  sorry

end NUMINAMATH_CALUDE_time_to_install_remaining_windows_l33_3358


namespace NUMINAMATH_CALUDE_solution_value_l33_3343

theorem solution_value (t : ℝ) : 
  (let y := -(t - 1)
   2 * y - 4 = 3 * (y - 2)) → 
  t = -1 := by
  sorry

end NUMINAMATH_CALUDE_solution_value_l33_3343


namespace NUMINAMATH_CALUDE_functions_are_odd_l33_3386

-- Define the property for functions f and g
def has_property (f g : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (g x) = g (f x) ∧ f (g x) = -x

-- Define what it means for a function to be odd
def is_odd (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f (-x) = -f x

-- Theorem statement
theorem functions_are_odd (f g : ℝ → ℝ) (h : has_property f g) :
  is_odd f ∧ is_odd g :=
sorry

end NUMINAMATH_CALUDE_functions_are_odd_l33_3386


namespace NUMINAMATH_CALUDE_smaller_root_of_equation_l33_3320

theorem smaller_root_of_equation (x : ℝ) : 
  (x - 2/3) * (x - 2/3) + (x - 2/3) * (x - 1/3) = 0 →
  (x = 1/2 ∨ x = 2/3) ∧ 1/2 < 2/3 := by
sorry

end NUMINAMATH_CALUDE_smaller_root_of_equation_l33_3320


namespace NUMINAMATH_CALUDE_system_solution_ratio_l33_3317

theorem system_solution_ratio (x y c d : ℝ) :
  (4 * x - 2 * y = c) →
  (6 * y - 12 * x = d) →
  d ≠ 0 →
  (∃ x y, (4 * x - 2 * y = c) ∧ (6 * y - 12 * x = d)) →
  c / d = -1 / 3 := by
sorry

end NUMINAMATH_CALUDE_system_solution_ratio_l33_3317


namespace NUMINAMATH_CALUDE_sqrt_three_parts_l33_3340

theorem sqrt_three_parts (x y : ℝ) : 
  (x = ⌊Real.sqrt 3⌋) → 
  (y = Real.sqrt 3 - x) → 
  Real.sqrt 3 * x - y = 1 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_three_parts_l33_3340
