import Mathlib

namespace NUMINAMATH_CALUDE_shaded_cubes_count_l587_58798

/-- Represents a 3x3x3 cube with shaded faces -/
structure ShadedCube where
  /-- Total number of smaller cubes -/
  total_cubes : Nat
  /-- Number of cubes per edge -/
  edge_length : Nat
  /-- Number of shaded cubes per face -/
  shaded_per_face : Nat

/-- Calculates the number of cubes with at least one face shaded -/
def count_shaded_cubes (cube : ShadedCube) : Nat :=
  sorry

/-- Theorem stating that the number of shaded cubes is 14 -/
theorem shaded_cubes_count (cube : ShadedCube) 
  (h1 : cube.total_cubes = 27)
  (h2 : cube.edge_length = 3)
  (h3 : cube.shaded_per_face = 5) : 
  count_shaded_cubes cube = 14 := by
  sorry

end NUMINAMATH_CALUDE_shaded_cubes_count_l587_58798


namespace NUMINAMATH_CALUDE_oneRedOneWhite_bothWhite_mutually_exclusive_not_contradictory_l587_58752

/-- Represents the color of a ball -/
inductive BallColor
| White
| Red

/-- Represents the outcome of drawing two balls -/
structure DrawOutcome :=
  (first second : BallColor)

/-- The bag containing balls -/
def bag : Multiset BallColor := 2 • {BallColor.White} + 3 • {BallColor.Red}

/-- Event: One red ball and one white ball -/
def oneRedOneWhite (outcome : DrawOutcome) : Prop :=
  (outcome.first = BallColor.Red ∧ outcome.second = BallColor.White) ∨
  (outcome.first = BallColor.White ∧ outcome.second = BallColor.Red)

/-- Event: Both balls are white -/
def bothWhite (outcome : DrawOutcome) : Prop :=
  outcome.first = BallColor.White ∧ outcome.second = BallColor.White

/-- Two events are mutually exclusive -/
def mutuallyExclusive (e1 e2 : DrawOutcome → Prop) : Prop :=
  ∀ outcome, ¬(e1 outcome ∧ e2 outcome)

/-- Two events are contradictory -/
def contradictory (e1 e2 : DrawOutcome → Prop) : Prop :=
  mutuallyExclusive e1 e2 ∧ ∀ outcome, e1 outcome ∨ e2 outcome

theorem oneRedOneWhite_bothWhite_mutually_exclusive_not_contradictory :
  mutuallyExclusive oneRedOneWhite bothWhite ∧
  ¬contradictory oneRedOneWhite bothWhite :=
sorry

end NUMINAMATH_CALUDE_oneRedOneWhite_bothWhite_mutually_exclusive_not_contradictory_l587_58752


namespace NUMINAMATH_CALUDE_balls_picked_is_two_l587_58736

/-- The number of red balls in the bag -/
def red_balls : ℕ := 3

/-- The number of blue balls in the bag -/
def blue_balls : ℕ := 2

/-- The number of green balls in the bag -/
def green_balls : ℕ := 3

/-- The total number of balls in the bag -/
def total_balls : ℕ := red_balls + blue_balls + green_balls

/-- The probability of picking two red balls -/
def prob_two_red : ℚ := 3 / 28

/-- The number of balls picked at random -/
def balls_picked : ℕ := 2

/-- Theorem stating that the number of balls picked is 2 given the conditions -/
theorem balls_picked_is_two :
  (red_balls = 3 ∧ blue_balls = 2 ∧ green_balls = 3) →
  (prob_two_red = 3 / 28) →
  (balls_picked = 2) := by sorry

end NUMINAMATH_CALUDE_balls_picked_is_two_l587_58736


namespace NUMINAMATH_CALUDE_same_terminal_side_angles_l587_58777

/-- Given an angle α = -51°, this theorem states that all angles with the same terminal side as α
    can be represented as k · 360° - 51°, where k is an integer. -/
theorem same_terminal_side_angles (α : ℝ) (h : α = -51) :
  ∀ θ : ℝ, (∃ k : ℤ, θ = k * 360 - 51) ↔ (∃ n : ℤ, θ = α + n * 360) :=
by sorry

end NUMINAMATH_CALUDE_same_terminal_side_angles_l587_58777


namespace NUMINAMATH_CALUDE_draw_probability_l587_58750

/-- The probability of player A winning a chess game -/
def prob_A_wins : ℝ := 0.4

/-- The probability that player A does not lose a chess game -/
def prob_A_not_lose : ℝ := 0.9

/-- The probability of a draw between players A and B -/
def prob_draw : ℝ := prob_A_not_lose - prob_A_wins

theorem draw_probability :
  prob_draw = 0.5 :=
sorry

end NUMINAMATH_CALUDE_draw_probability_l587_58750


namespace NUMINAMATH_CALUDE_even_digits_finite_fissile_squares_odd_digits_infinite_fissile_squares_l587_58742

/-- A fissile square is a positive integer which is a perfect square,
    and whose digits form two perfect squares in a row. -/
def is_fissile_square (n : ℕ) : Prop :=
  ∃ (x y r : ℕ) (d : ℕ), 
    n = x^2 ∧ 
    n = 10^d * y^2 + r^2 ∧ 
    y^2 ≠ 0 ∧ r^2 ≠ 0

/-- The number of digits in a natural number -/
def num_digits (n : ℕ) : ℕ := sorry

/-- Theorem: Every square with an even number of digits is the right square of only finitely many fissile squares -/
theorem even_digits_finite_fissile_squares (r : ℕ) (h : Even (num_digits (r^2))) :
  {x : ℕ | is_fissile_square (x^2) ∧ ∃ (y : ℕ) (d : ℕ), x^2 = 10^d * y^2 + r^2 ∧ Even d}.Finite :=
sorry

/-- Theorem: Every square with an odd number of digits is the right square of infinitely many fissile squares -/
theorem odd_digits_infinite_fissile_squares (r : ℕ) (h : Odd (num_digits (r^2))) :
  {x : ℕ | is_fissile_square (x^2) ∧ ∃ (y : ℕ) (d : ℕ), x^2 = 10^d * y^2 + r^2 ∧ Odd d}.Infinite :=
sorry

end NUMINAMATH_CALUDE_even_digits_finite_fissile_squares_odd_digits_infinite_fissile_squares_l587_58742


namespace NUMINAMATH_CALUDE_square_difference_characterization_l587_58793

theorem square_difference_characterization (N : ℕ+) :
  (∃ k : ℕ, (2^N.val : ℕ) - 2 * N.val = k^2) ↔ N = 1 ∨ N = 2 := by
  sorry

end NUMINAMATH_CALUDE_square_difference_characterization_l587_58793


namespace NUMINAMATH_CALUDE_bisection_method_for_f_l587_58775

def f (x : ℝ) := 3 * x^2 - 1

theorem bisection_method_for_f :
  ∃ (x₀ : ℝ), x₀ ∈ Set.Ioo 0 1 ∧ f x₀ = 0 ∧ ∀ x ∈ Set.Ioo 0 1, f x = 0 → x = x₀ →
  let ε : ℝ := 0.05
  let n : ℕ := 5
  let approx : ℝ := 37/64
  (∀ m : ℕ, m < n → 1 / 2^m > ε) ∧
  1 / 2^n ≤ ε ∧
  |approx - x₀| < ε :=
sorry

end NUMINAMATH_CALUDE_bisection_method_for_f_l587_58775


namespace NUMINAMATH_CALUDE_urn_problem_l587_58741

theorem urn_problem (w : ℕ) : 
  (10 : ℝ) / (10 + w) * 9 / (9 + w) = 0.4285714285714286 → w = 5 := by
  sorry

end NUMINAMATH_CALUDE_urn_problem_l587_58741


namespace NUMINAMATH_CALUDE_percentage_problem_l587_58721

theorem percentage_problem (x : ℝ) (h1 : x > 0) (h2 : x * (x / 100) = 9) : x = 30 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l587_58721


namespace NUMINAMATH_CALUDE_existence_of_monochromatic_triangle_l587_58707

/-- A point in the six-pointed star --/
structure Point :=
  (index : Fin 13)

/-- The color of a point --/
inductive Color
| Red
| Green

/-- A coloring of the points in the star --/
def Coloring := Point → Color

/-- Predicate to check if three points form an equilateral triangle --/
def IsEquilateralTriangle (p q r : Point) : Prop := sorry

/-- The main theorem --/
theorem existence_of_monochromatic_triangle (coloring : Coloring) :
  ∃ (p q r : Point), coloring p = coloring q ∧ coloring q = coloring r ∧ IsEquilateralTriangle p q r :=
sorry

end NUMINAMATH_CALUDE_existence_of_monochromatic_triangle_l587_58707


namespace NUMINAMATH_CALUDE_min_students_in_math_club_l587_58771

/-- Represents a math club with boys and girls -/
structure MathClub where
  boys : ℕ
  girls : ℕ

/-- The condition that more than 60% of students are boys -/
def moreThan60PercentBoys (club : MathClub) : Prop :=
  (club.boys : ℚ) / (club.boys + club.girls : ℚ) > 60 / 100

/-- The theorem stating the minimum number of students in the club -/
theorem min_students_in_math_club :
  ∀ (club : MathClub),
  moreThan60PercentBoys club →
  club.girls = 5 →
  club.boys + club.girls ≥ 13 :=
by sorry

end NUMINAMATH_CALUDE_min_students_in_math_club_l587_58771


namespace NUMINAMATH_CALUDE_A_intersect_B_eq_B_l587_58769

-- Define set A
def A : Set ℝ := {y | ∃ x : ℝ, y = |x| - 1}

-- Define set B
def B : Set ℝ := {x | x ≥ 2}

-- Theorem statement
theorem A_intersect_B_eq_B : A ∩ B = B := by sorry

end NUMINAMATH_CALUDE_A_intersect_B_eq_B_l587_58769


namespace NUMINAMATH_CALUDE_stock_price_return_l587_58737

theorem stock_price_return (initial_price : ℝ) (h : initial_price > 0) : 
  let increased_price := initial_price * 1.3
  let decrease_percentage := (1 - 1 / 1.3) * 100
  increased_price * (1 - decrease_percentage / 100) = initial_price :=
by sorry

end NUMINAMATH_CALUDE_stock_price_return_l587_58737


namespace NUMINAMATH_CALUDE_zoo_animal_difference_l587_58734

def zoo_problem (parrots snakes monkeys elephants zebras : ℕ) : Prop :=
  (parrots = 8) ∧
  (snakes = 3 * parrots) ∧
  (monkeys = 2 * snakes) ∧
  (elephants = (parrots + snakes) / 2) ∧
  (zebras = elephants - 3)

theorem zoo_animal_difference :
  ∀ parrots snakes monkeys elephants zebras : ℕ,
  zoo_problem parrots snakes monkeys elephants zebras →
  monkeys - zebras = 35 :=
by
  sorry

end NUMINAMATH_CALUDE_zoo_animal_difference_l587_58734


namespace NUMINAMATH_CALUDE_sum_of_real_solutions_l587_58728

theorem sum_of_real_solutions (b : ℝ) (h : b > 0) :
  ∃ x : ℝ, x ≥ 0 ∧ Real.sqrt (b - Real.sqrt (b + 2*x)) = x ∧
  (∀ y : ℝ, y ≥ 0 ∧ Real.sqrt (b - Real.sqrt (b + 2*y)) = y → y = x) ∧
  x = Real.sqrt (b - 1) - 1 :=
sorry

end NUMINAMATH_CALUDE_sum_of_real_solutions_l587_58728


namespace NUMINAMATH_CALUDE_equation_solution_l587_58776

theorem equation_solution : 
  ∀ x : ℝ, x > 0 → (x^(Real.log x / Real.log 5) = x^4 / 250 ↔ x = 5 ∨ x = 125) :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l587_58776


namespace NUMINAMATH_CALUDE_cube_sum_inequality_cube_sum_equality_l587_58762

theorem cube_sum_inequality (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  x^3 / (y*z) + y^3 / (z*x) + z^3 / (x*y) ≥ x + y + z :=
sorry

theorem cube_sum_equality (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0) :
  x^3 / (y*z) + y^3 / (z*x) + z^3 / (x*y) = x + y + z ↔ x = y ∧ y = z :=
sorry

end NUMINAMATH_CALUDE_cube_sum_inequality_cube_sum_equality_l587_58762


namespace NUMINAMATH_CALUDE_valid_numbers_l587_58759

def is_valid_number (n : ℕ) : Prop :=
  (1000 ≤ n) ∧ (n < 10000) ∧
  (∃ (a d : ℕ), 
    0 ≤ a ∧ a ≤ 9 ∧ 0 ≤ d ∧ d ≤ 9 ∧
    n = 120 * (10 * a + d))

theorem valid_numbers :
  {n : ℕ | is_valid_number n} = {1200, 2400, 3600, 4800} := by sorry

end NUMINAMATH_CALUDE_valid_numbers_l587_58759


namespace NUMINAMATH_CALUDE_rented_cars_at_3600_max_revenue_l587_58760

/-- Represents the rental company's car fleet and pricing model. -/
structure RentalCompany where
  total_cars : ℕ
  base_rent : ℕ
  rent_increment : ℕ
  rented_maintenance : ℕ
  unrented_maintenance : ℕ

/-- Calculates the number of rented cars given a certain rent. -/
def rented_cars (company : RentalCompany) (rent : ℕ) : ℕ :=
  company.total_cars - (rent - company.base_rent) / company.rent_increment

/-- Calculates the monthly revenue given a certain rent. -/
def monthly_revenue (company : RentalCompany) (rent : ℕ) : ℕ :=
  let rented := rented_cars company rent
  rent * rented - company.rented_maintenance * rented - 
    company.unrented_maintenance * (company.total_cars - rented)

/-- The rental company with the given parameters. -/
def our_company : RentalCompany := {
  total_cars := 100,
  base_rent := 3000,
  rent_increment := 50,
  rented_maintenance := 150,
  unrented_maintenance := 50
}

/-- Theorem stating the number of rented cars when rent is 3600 yuan. -/
theorem rented_cars_at_3600 : 
  rented_cars our_company 3600 = 88 := by sorry

/-- Theorem stating the rent that maximizes revenue and the maximum revenue. -/
theorem max_revenue : 
  ∃ (max_rent : ℕ), max_rent = 4050 ∧ 
  monthly_revenue our_company max_rent = 37050 ∧
  ∀ (rent : ℕ), monthly_revenue our_company rent ≤ monthly_revenue our_company max_rent := by sorry

end NUMINAMATH_CALUDE_rented_cars_at_3600_max_revenue_l587_58760


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l587_58719

theorem sqrt_equation_solution (y : ℝ) :
  Real.sqrt (2 + Real.sqrt (3 * y - 4)) = 4 → y = 200 / 3 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l587_58719


namespace NUMINAMATH_CALUDE_reflex_angle_at_H_l587_58766

-- Define the points
variable (C D F M H : Point)

-- Define the angles
def angle_CDH : ℝ := 150
def angle_HFM : ℝ := 95

-- Define the properties
def collinear (C D F M : Point) : Prop := sorry
def angle (A B C : Point) : ℝ := sorry
def reflex_angle (A : Point) : ℝ := sorry

-- State the theorem
theorem reflex_angle_at_H (h_collinear : collinear C D F M) 
  (h_CDH : angle C D H = angle_CDH)
  (h_HFM : angle H F M = angle_HFM) : 
  reflex_angle H = 180 := by sorry

end NUMINAMATH_CALUDE_reflex_angle_at_H_l587_58766


namespace NUMINAMATH_CALUDE_find_number_l587_58796

theorem find_number : ∃ x : ℝ, 3 * (2 * x + 9) = 69 :=
by sorry

end NUMINAMATH_CALUDE_find_number_l587_58796


namespace NUMINAMATH_CALUDE_cube_remainder_l587_58785

theorem cube_remainder (n : ℤ) : n % 6 = 3 → n^3 % 8 = 3 := by
  sorry

end NUMINAMATH_CALUDE_cube_remainder_l587_58785


namespace NUMINAMATH_CALUDE_polynomial_property_l587_58716

/-- Polynomial P(x) = 3x^3 + ax^2 + bx + c satisfying given conditions -/
def P (a b c : ℝ) (x : ℝ) : ℝ := 3 * x^3 + a * x^2 + b * x + c

theorem polynomial_property (a b c : ℝ) :
  (∀ x₁ x₂ x₃ : ℝ, x₁ + x₂ + x₃ = 0 → P a b c x₁ = 0 → P a b c x₂ = 0 → P a b c x₃ = 0 →
    ((x₁ + x₂ + x₃) / 3 = x₁ * x₂ * x₃)) →  -- mean of zeros equals product of zeros
  (∀ x₁ x₂ x₃ : ℝ, x₁ + x₂ + x₃ = 0 → P a b c x₁ = 0 → P a b c x₂ = 0 → P a b c x₃ = 0 →
    (x₁ * x₂ * x₃ = 3 + a + b + c)) →  -- product of zeros equals sum of coefficients
  P a b c 0 = 15 →  -- y-intercept is 15
  b = -38 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_property_l587_58716


namespace NUMINAMATH_CALUDE_hotel_breakfast_probability_l587_58794

def num_guests : ℕ := 3
def num_roll_types : ℕ := 4
def total_rolls : ℕ := 12
def rolls_per_guest : ℕ := 4

def probability_one_of_each : ℚ :=
  (9 : ℚ) / 55 * (8 : ℚ) / 35 * (1 : ℚ) / 1

theorem hotel_breakfast_probability :
  probability_one_of_each = (72 : ℚ) / 1925 :=
by sorry

end NUMINAMATH_CALUDE_hotel_breakfast_probability_l587_58794


namespace NUMINAMATH_CALUDE_help_desk_services_percentage_l587_58738

theorem help_desk_services_percentage (total_hours software_hours help_user_hours : ℝ) 
  (h1 : total_hours = 68.33333333333333)
  (h2 : software_hours = 24)
  (h3 : help_user_hours = 17) :
  (total_hours - software_hours - help_user_hours) / total_hours * 100 = 40 := by
  sorry

end NUMINAMATH_CALUDE_help_desk_services_percentage_l587_58738


namespace NUMINAMATH_CALUDE_savings_account_balance_l587_58715

theorem savings_account_balance (initial_amount : ℝ) 
  (increase_percentage : ℝ) (decrease_percentage : ℝ) : 
  initial_amount = 125 ∧ 
  increase_percentage = 0.25 ∧ 
  decrease_percentage = 0.20 →
  initial_amount = initial_amount * (1 + increase_percentage) * (1 - decrease_percentage) :=
by sorry

end NUMINAMATH_CALUDE_savings_account_balance_l587_58715


namespace NUMINAMATH_CALUDE_house_ratio_l587_58730

theorem house_ratio (houses_one_side : ℕ) (total_houses : ℕ) : 
  houses_one_side = 40 → 
  total_houses = 160 → 
  (total_houses - houses_one_side) / houses_one_side = 3 := by
sorry

end NUMINAMATH_CALUDE_house_ratio_l587_58730


namespace NUMINAMATH_CALUDE_class_mean_calculation_l587_58709

theorem class_mean_calculation (total_students : ℕ) 
  (group1_students : ℕ) (group1_mean : ℚ) 
  (group2_students : ℕ) (group2_mean : ℚ) : 
  total_students = 50 →
  group1_students = 45 →
  group2_students = 5 →
  group1_mean = 85 / 100 →
  group2_mean = 90 / 100 →
  let overall_mean := (group1_students * group1_mean + group2_students * group2_mean) / total_students
  overall_mean = 855 / 1000 := by
sorry

end NUMINAMATH_CALUDE_class_mean_calculation_l587_58709


namespace NUMINAMATH_CALUDE_intersection_implies_x_value_l587_58754

def A (x : ℝ) : Set ℝ := {1, 4, x}
def B (x : ℝ) : Set ℝ := {1, 2*x, x^2}

theorem intersection_implies_x_value :
  ∀ x : ℝ, A x ∩ B x = {1, 4} → x = -2 := by sorry

end NUMINAMATH_CALUDE_intersection_implies_x_value_l587_58754


namespace NUMINAMATH_CALUDE_max_value_of_a_l587_58780

theorem max_value_of_a (a b c : ℝ) (sum_eq : a + b + c = 3) (prod_sum_eq : a * b + a * c + b * c = 3) :
  a ≤ 1 + Real.sqrt 2 ∧ ∃ (a₀ b₀ c₀ : ℝ), a₀ + b₀ + c₀ = 3 ∧ a₀ * b₀ + a₀ * c₀ + b₀ * c₀ = 3 ∧ a₀ = 1 + Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_a_l587_58780


namespace NUMINAMATH_CALUDE_triangle_with_sine_sides_l587_58724

theorem triangle_with_sine_sides 
  (α β γ : Real) 
  (h_triangle : α + β + γ = π) 
  (h_positive : 0 < α ∧ 0 < β ∧ 0 < γ) 
  (h_less_than_pi : α < π ∧ β < π ∧ γ < π) : 
  ∃ (a b c : Real), 
    a = Real.sin α ∧ 
    b = Real.sin β ∧ 
    c = Real.sin γ ∧ 
    a + b > c ∧ 
    b + c > a ∧ 
    c + a > b := by
  sorry

end NUMINAMATH_CALUDE_triangle_with_sine_sides_l587_58724


namespace NUMINAMATH_CALUDE_total_rent_is_435_l587_58745

/-- Represents the rent calculation for a pasture shared by multiple parties -/
structure PastureRent where
  a_horses : ℕ
  a_months : ℕ
  b_horses : ℕ
  b_months : ℕ
  c_horses : ℕ
  c_months : ℕ
  b_payment : ℕ

/-- Calculates the total rent for the pasture -/
def calculate_total_rent (pr : PastureRent) : ℕ :=
  let total_horse_months := pr.a_horses * pr.a_months + pr.b_horses * pr.b_months + pr.c_horses * pr.c_months
  let b_horse_months := pr.b_horses * pr.b_months
  (pr.b_payment * total_horse_months) / b_horse_months

/-- Theorem stating that the total rent for the given conditions is 435 -/
theorem total_rent_is_435 (pr : PastureRent) 
  (h1 : pr.a_horses = 12) (h2 : pr.a_months = 8)
  (h3 : pr.b_horses = 16) (h4 : pr.b_months = 9)
  (h5 : pr.c_horses = 18) (h6 : pr.c_months = 6)
  (h7 : pr.b_payment = 180) : 
  calculate_total_rent pr = 435 := by
  sorry

end NUMINAMATH_CALUDE_total_rent_is_435_l587_58745


namespace NUMINAMATH_CALUDE_train_travel_time_l587_58789

/-- Proves that a train traveling at 150 km/h for 1200 km takes 8 hours -/
theorem train_travel_time :
  ∀ (speed distance time : ℝ),
    speed = 150 ∧ distance = 1200 ∧ time = distance / speed →
    time = 8 := by
  sorry

end NUMINAMATH_CALUDE_train_travel_time_l587_58789


namespace NUMINAMATH_CALUDE_lucas_sandwich_problem_l587_58706

/-- Luca's sandwich shop problem --/
theorem lucas_sandwich_problem (sandwich_price : ℝ) (discount_rate : ℝ) 
  (avocado_price : ℝ) (salad_price : ℝ) (total_bill : ℝ) 
  (h1 : sandwich_price = 8)
  (h2 : discount_rate = 1/4)
  (h3 : avocado_price = 1)
  (h4 : salad_price = 3)
  (h5 : total_bill = 12) :
  total_bill - (sandwich_price * (1 - discount_rate) + avocado_price + salad_price) = 2 := by
  sorry

#check lucas_sandwich_problem

end NUMINAMATH_CALUDE_lucas_sandwich_problem_l587_58706


namespace NUMINAMATH_CALUDE_jerry_weller_votes_l587_58755

theorem jerry_weller_votes 
  (total_votes : ℕ) 
  (vote_difference : ℕ) 
  (h1 : total_votes = 196554)
  (h2 : vote_difference = 20196) :
  ∃ (jerry_votes john_votes : ℕ),
    jerry_votes + john_votes = total_votes ∧
    jerry_votes = john_votes + vote_difference ∧
    jerry_votes = 108375 := by
sorry

end NUMINAMATH_CALUDE_jerry_weller_votes_l587_58755


namespace NUMINAMATH_CALUDE_square_root_equation_l587_58740

theorem square_root_equation (x : ℝ) : Real.sqrt (x + 4) = 12 → x = 140 := by
  sorry

end NUMINAMATH_CALUDE_square_root_equation_l587_58740


namespace NUMINAMATH_CALUDE_sheila_tuesday_thursday_hours_l587_58784

/-- Represents Sheila's work schedule and earnings --/
structure WorkSchedule where
  hours_mwf : ℕ  -- Hours worked on Monday, Wednesday, Friday
  hours_tt : ℕ   -- Hours worked on Tuesday, Thursday
  hourly_rate : ℕ -- Hourly rate in dollars
  weekly_earnings : ℕ -- Total weekly earnings in dollars

/-- Calculates the total hours worked in a week --/
def total_hours (s : WorkSchedule) : ℕ :=
  3 * s.hours_mwf + 2 * s.hours_tt

/-- Calculates the total earnings based on hours worked and hourly rate --/
def calculated_earnings (s : WorkSchedule) : ℕ :=
  s.hourly_rate * (total_hours s)

/-- Theorem stating that Sheila works 6 hours on Tuesday and Thursday --/
theorem sheila_tuesday_thursday_hours (s : WorkSchedule) 
  (h1 : s.hours_mwf = 8)
  (h2 : s.hourly_rate = 11)
  (h3 : s.weekly_earnings = 396)
  (h4 : calculated_earnings s = s.weekly_earnings) :
  s.hours_tt = 6 := by
  sorry

end NUMINAMATH_CALUDE_sheila_tuesday_thursday_hours_l587_58784


namespace NUMINAMATH_CALUDE_square_root_of_square_negative_two_l587_58768

theorem square_root_of_square_negative_two : Real.sqrt ((-2)^2) = 2 := by
  sorry

end NUMINAMATH_CALUDE_square_root_of_square_negative_two_l587_58768


namespace NUMINAMATH_CALUDE_initial_deposit_l587_58700

theorem initial_deposit (P R : ℝ) : 
  P + (P * R * 3) / 100 = 9200 →
  P + (P * (R + 0.5) * 3) / 100 = 9320 →
  P = 8000 := by
sorry

end NUMINAMATH_CALUDE_initial_deposit_l587_58700


namespace NUMINAMATH_CALUDE_intersection_at_midpoint_l587_58757

/-- A line with equation x - y = c intersects the line segment from (1, 4) to (3, 8) at its midpoint -/
theorem intersection_at_midpoint (c : ℝ) : 
  (∃ (x y : ℝ), x - y = c ∧ 
    x = (1 + 3) / 2 ∧ 
    y = (4 + 8) / 2 ∧ 
    (x, y) = ((1 + 3) / 2, (4 + 8) / 2)) → 
  c = -4 := by
sorry

end NUMINAMATH_CALUDE_intersection_at_midpoint_l587_58757


namespace NUMINAMATH_CALUDE_tan_alpha_plus_pi_sixth_l587_58747

theorem tan_alpha_plus_pi_sixth (α : ℝ) 
  (h : Real.cos (3 * Real.pi / 2 - α) = 2 * Real.sin (α + Real.pi / 3)) : 
  Real.tan (α + Real.pi / 6) = -Real.sqrt 3 / 9 := by
sorry

end NUMINAMATH_CALUDE_tan_alpha_plus_pi_sixth_l587_58747


namespace NUMINAMATH_CALUDE_fundraising_average_contribution_l587_58781

/-- Proves that the average contribution required from the remaining targeted people
    is $400 / 0.36, given the conditions of the fundraising problem. -/
theorem fundraising_average_contribution
  (total_amount : ℝ) 
  (total_people : ℝ) 
  (h1 : total_amount > 0)
  (h2 : total_people > 0)
  (h3 : 0.6 * total_amount = 0.4 * total_people * 400) :
  (0.4 * total_amount) / (0.6 * total_people) = 400 / 0.36 := by
sorry

end NUMINAMATH_CALUDE_fundraising_average_contribution_l587_58781


namespace NUMINAMATH_CALUDE_total_spent_is_575_l587_58782

def vacuum_original_cost : ℚ := 250
def vacuum_discount_rate : ℚ := 20 / 100
def dishwasher_cost : ℚ := 450
def combined_discount : ℚ := 75

def total_spent : ℚ :=
  (vacuum_original_cost * (1 - vacuum_discount_rate) + dishwasher_cost) - combined_discount

theorem total_spent_is_575 : total_spent = 575 := by
  sorry

end NUMINAMATH_CALUDE_total_spent_is_575_l587_58782


namespace NUMINAMATH_CALUDE_base_length_is_double_half_length_l587_58720

/-- An isosceles triangle with a line bisector from the vertex angle -/
structure IsoscelesTriangleWithBisector :=
  (base_half_length : ℝ)

/-- The theorem stating that the total base length is twice the length of each half -/
theorem base_length_is_double_half_length (triangle : IsoscelesTriangleWithBisector) 
  (h : triangle.base_half_length = 4) : 
  2 * triangle.base_half_length = 8 := by
  sorry

#check base_length_is_double_half_length

end NUMINAMATH_CALUDE_base_length_is_double_half_length_l587_58720


namespace NUMINAMATH_CALUDE_number_of_boys_l587_58708

theorem number_of_boys (total : ℕ) (girls : ℕ) (boys : ℕ) : 
  girls = (60 : ℕ) * total / 100 →
  girls = 450 →
  boys = total - girls →
  boys = 300 := by
sorry

end NUMINAMATH_CALUDE_number_of_boys_l587_58708


namespace NUMINAMATH_CALUDE_scientific_notation_of_56_99_million_l587_58726

theorem scientific_notation_of_56_99_million :
  (56.99 * 1000000 : ℝ) = 5.699 * (10 ^ 7) :=
by sorry

end NUMINAMATH_CALUDE_scientific_notation_of_56_99_million_l587_58726


namespace NUMINAMATH_CALUDE_number_difference_l587_58795

theorem number_difference (a b : ℕ) 
  (sum_eq : a + b = 25220)
  (div_12 : 12 ∣ a)
  (relation : b = a / 100) : 
  a - b = 24750 := by
sorry

end NUMINAMATH_CALUDE_number_difference_l587_58795


namespace NUMINAMATH_CALUDE_age_difference_l587_58731

/-- Given the ages of Frank, Ty, Carla, and Karen, prove that Ty's current age is 4 years more than twice Carla's age. -/
theorem age_difference (frank_future ty_now carla_now karen_now : ℕ) : 
  karen_now = 2 →
  carla_now = karen_now + 2 →
  frank_future = 36 →
  frank_future = ty_now * 3 + 5 →
  ty_now > 2 * carla_now →
  ty_now - 2 * carla_now = 4 := by
  sorry

end NUMINAMATH_CALUDE_age_difference_l587_58731


namespace NUMINAMATH_CALUDE_travel_ways_count_l587_58704

/-- The number of available train trips -/
def train_trips : ℕ := 4

/-- The number of available ferry trips -/
def ferry_trips : ℕ := 3

/-- The total number of ways to travel from A to B -/
def total_ways : ℕ := train_trips + ferry_trips

theorem travel_ways_count : total_ways = 7 := by
  sorry

end NUMINAMATH_CALUDE_travel_ways_count_l587_58704


namespace NUMINAMATH_CALUDE_abc_zero_l587_58739

theorem abc_zero (a b c : ℝ) 
  (h1 : (a + b) * (b + c) * (c + a) = a * b * c)
  (h2 : (a^9 + b^9) * (b^9 + c^9) * (c^9 + a^9) = (a * b * c)^9) :
  a = 0 ∨ b = 0 ∨ c = 0 := by
sorry

end NUMINAMATH_CALUDE_abc_zero_l587_58739


namespace NUMINAMATH_CALUDE_beth_crayons_l587_58723

theorem beth_crayons (initial_packs : ℚ) : 
  (initial_packs / 10 + 6 = 6.4) → initial_packs = 4 := by
  sorry

end NUMINAMATH_CALUDE_beth_crayons_l587_58723


namespace NUMINAMATH_CALUDE_complex_product_pure_imaginary_l587_58729

theorem complex_product_pure_imaginary (a : ℝ) : 
  let z₁ : ℂ := 1 - 2 * Complex.I
  let z₂ : ℂ := a + Complex.I
  (∃ (b : ℝ), z₁ * z₂ = b * Complex.I ∧ b ≠ 0) → a = -2 := by
  sorry

end NUMINAMATH_CALUDE_complex_product_pure_imaginary_l587_58729


namespace NUMINAMATH_CALUDE_theater_line_arrangements_l587_58718

def number_of_people : ℕ := 8
def number_of_fixed_group : ℕ := 3

theorem theater_line_arrangements :
  (number_of_people - number_of_fixed_group + 1).factorial = 720 := by
  sorry

end NUMINAMATH_CALUDE_theater_line_arrangements_l587_58718


namespace NUMINAMATH_CALUDE_calculate_initial_weight_l587_58713

/-- Calculates the initial weight of a person on a constant weight loss diet -/
theorem calculate_initial_weight 
  (current_weight : ℝ) 
  (future_weight : ℝ) 
  (months_to_future : ℝ) 
  (months_on_diet : ℝ) 
  (h1 : current_weight > future_weight) 
  (h2 : months_to_future > 0) 
  (h3 : months_on_diet > 0) :
  ∃ (initial_weight : ℝ),
    initial_weight = current_weight + (current_weight - future_weight) / months_to_future * months_on_diet :=
by
  sorry

#check calculate_initial_weight

end NUMINAMATH_CALUDE_calculate_initial_weight_l587_58713


namespace NUMINAMATH_CALUDE_solve_scarf_knitting_problem_l587_58787

/-- Represents the time (in hours) to knit various items --/
structure KnittingTime where
  hat : ℝ
  mitten : ℝ
  sock : ℝ
  sweater : ℝ

/-- The problem of finding the time to knit a scarf --/
def scarf_knitting_problem (kt : KnittingTime) (num_children : ℕ) (total_time : ℝ) : Prop :=
  let scarf_time := (total_time - num_children * (kt.hat + 2 * kt.mitten + 2 * kt.sock + kt.sweater)) / num_children
  scarf_time = 3

/-- The theorem stating the solution to the scarf knitting problem --/
theorem solve_scarf_knitting_problem :
  ∀ (kt : KnittingTime) (num_children : ℕ),
  kt.hat = 2 ∧ kt.mitten = 1 ∧ kt.sock = 1.5 ∧ kt.sweater = 6 ∧ num_children = 3 →
  scarf_knitting_problem kt num_children 48 :=
by
  sorry


end NUMINAMATH_CALUDE_solve_scarf_knitting_problem_l587_58787


namespace NUMINAMATH_CALUDE_triangle_area_triangle_area_is_eight_l587_58763

/-- Given two lines with slopes 1/4 and 5/4 intersecting at (1,1), and a vertical line x=5,
    the area of the triangle formed by these three lines is 8. -/
theorem triangle_area : ℝ → Prop :=
  fun area =>
    ∀ (line1 line2 : ℝ → ℝ) (x : ℝ),
      (∀ x, line1 x = 1/4 * x + 3/4) →  -- Equation of line with slope 1/4 passing through (1,1)
      (∀ x, line2 x = 5/4 * x - 1/4) →  -- Equation of line with slope 5/4 passing through (1,1)
      line1 1 = 1 →                     -- Both lines pass through (1,1)
      line2 1 = 1 →
      x = 5 →                           -- The vertical line is x=5
      area = 8                          -- The area of the formed triangle is 8

-- The proof of this theorem
theorem triangle_area_is_eight : triangle_area 8 := by
  sorry

end NUMINAMATH_CALUDE_triangle_area_triangle_area_is_eight_l587_58763


namespace NUMINAMATH_CALUDE_correct_mean_calculation_l587_58756

theorem correct_mean_calculation (n : ℕ) (incorrect_mean : ℚ) 
  (correct_values wrong_values : List ℚ) :
  n = 30 ∧ 
  incorrect_mean = 170 ∧
  correct_values = [190, 200, 175] ∧
  wrong_values = [150, 195, 160] →
  (n * incorrect_mean - wrong_values.sum + correct_values.sum) / n = 172 :=
by sorry

end NUMINAMATH_CALUDE_correct_mean_calculation_l587_58756


namespace NUMINAMATH_CALUDE_coat_price_calculation_l587_58783

/-- Calculates the final price of a coat after two discounts and tax --/
def finalPrice (originalPrice : ℝ) (discount1 : ℝ) (discount2 : ℝ) (taxRate : ℝ) : ℝ :=
  let priceAfterDiscount1 := originalPrice * (1 - discount1)
  let priceAfterDiscount2 := priceAfterDiscount1 * (1 - discount2)
  priceAfterDiscount2 * (1 + taxRate)

/-- Theorem stating that the final price of the coat is approximately 84.7 --/
theorem coat_price_calculation :
  let originalPrice : ℝ := 120
  let discount1 : ℝ := 0.30
  let discount2 : ℝ := 0.10
  let taxRate : ℝ := 0.12
  abs (finalPrice originalPrice discount1 discount2 taxRate - 84.7) < 0.1 := by
  sorry

#eval finalPrice 120 0.30 0.10 0.12

end NUMINAMATH_CALUDE_coat_price_calculation_l587_58783


namespace NUMINAMATH_CALUDE_common_tangent_implies_a_equals_one_l587_58790

/-- Given two curves y = (1/2e)x^2 and y = a ln x with a common tangent at their common point P(s, t), prove that a = 1 -/
theorem common_tangent_implies_a_equals_one (s t a : ℝ) : 
  t = (1 / (2 * Real.exp 1)) * s^2 →  -- Point P(s, t) lies on the first curve
  t = a * Real.log s →                -- Point P(s, t) lies on the second curve
  (s / Real.exp 1 = a / s) →          -- Common tangent condition
  a = 1 := by
sorry


end NUMINAMATH_CALUDE_common_tangent_implies_a_equals_one_l587_58790


namespace NUMINAMATH_CALUDE_system_solution_l587_58779

def satisfies_system (u v w : ℝ) : Prop :=
  u + v * w = 12 ∧ v + w * u = 12 ∧ w + u * v = 12

def solution_set : Set (ℝ × ℝ × ℝ) :=
  {(3, 3, 3), (-4, -4, -4), (1, 1, 11), (11, 1, 1), (1, 11, 1)}

theorem system_solution :
  {p : ℝ × ℝ × ℝ | satisfies_system p.1 p.2.1 p.2.2} = solution_set := by
  sorry

end NUMINAMATH_CALUDE_system_solution_l587_58779


namespace NUMINAMATH_CALUDE_bottle_production_l587_58772

/-- Given that 6 identical machines produce 270 bottles per minute at a constant rate,
    prove that 5 such machines will produce 900 bottles in 4 minutes. -/
theorem bottle_production 
  (machines : ℕ) 
  (bottles_per_minute : ℕ) 
  (h1 : machines = 6) 
  (h2 : bottles_per_minute = 270) : 
  (5 : ℕ) * (4 : ℕ) * (bottles_per_minute / machines) = 900 := by
  sorry


end NUMINAMATH_CALUDE_bottle_production_l587_58772


namespace NUMINAMATH_CALUDE_sum_of_children_ages_l587_58761

/-- Represents the ages of Cynthia's children -/
structure ChildrenAges where
  freddy : ℕ
  matthew : ℕ
  rebecca : ℕ

/-- Theorem stating the sum of Cynthia's children's ages -/
theorem sum_of_children_ages (ages : ChildrenAges) : 
  ages.freddy = 15 → 
  ages.matthew = ages.freddy - 4 → 
  ages.rebecca = ages.matthew - 2 → 
  ages.freddy + ages.matthew + ages.rebecca = 35 := by
  sorry


end NUMINAMATH_CALUDE_sum_of_children_ages_l587_58761


namespace NUMINAMATH_CALUDE_average_score_calculation_l587_58744

theorem average_score_calculation (total_students : ℕ) (male_students : ℕ) (female_students : ℕ)
  (male_avg_score : ℝ) (female_avg_score : ℝ) :
  male_students = (0.4 : ℝ) * total_students →
  female_students = total_students - male_students →
  male_avg_score = 75 →
  female_avg_score = 80 →
  (male_avg_score * male_students + female_avg_score * female_students) / total_students = 78 :=
by
  sorry

#check average_score_calculation

end NUMINAMATH_CALUDE_average_score_calculation_l587_58744


namespace NUMINAMATH_CALUDE_marble_count_l587_58703

/-- Given a bag of marbles with blue, red, and white marbles, 
    prove that the total number of marbles is 50 -/
theorem marble_count (blue red white : ℕ) (total : ℕ) 
    (h1 : blue = 5)
    (h2 : red = 9)
    (h3 : total = blue + red + white)
    (h4 : (red + white : ℚ) / total = 9/10) :
  total = 50 := by
  sorry

end NUMINAMATH_CALUDE_marble_count_l587_58703


namespace NUMINAMATH_CALUDE_cubic_yards_to_cubic_feet_l587_58711

/-- Given that 1 yard equals 3 feet, prove that 5 cubic yards is equal to 135 cubic feet. -/
theorem cubic_yards_to_cubic_feet :
  (1 : ℝ) * (1 : ℝ) * (1 : ℝ) = 27 * (1 / 3 : ℝ) * (1 / 3 : ℝ) * (1 / 3 : ℝ) →
  5 * (1 : ℝ) * (1 : ℝ) * (1 : ℝ) = 135 * (1 / 3 : ℝ) * (1 / 3 : ℝ) * (1 / 3 : ℝ) :=
by sorry

end NUMINAMATH_CALUDE_cubic_yards_to_cubic_feet_l587_58711


namespace NUMINAMATH_CALUDE_exchange_rate_scaling_l587_58714

theorem exchange_rate_scaling (x : ℝ) :
  2994 * 14.5 = 177 → 29.94 * 1.45 = 0.177 := by
  sorry

end NUMINAMATH_CALUDE_exchange_rate_scaling_l587_58714


namespace NUMINAMATH_CALUDE_correct_calculation_l587_58749

theorem correct_calculation (x : ℤ) : 66 + x = 93 → (66 - x) + 21 = 60 := by
  sorry

end NUMINAMATH_CALUDE_correct_calculation_l587_58749


namespace NUMINAMATH_CALUDE_mollys_gift_cost_l587_58751

/-- Represents the cost and family structure for Molly's gift-sending problem -/
structure GiftSendingProblem where
  cost_per_package : ℕ
  num_parents : ℕ
  num_brothers : ℕ
  num_sisters : ℕ
  children_per_brother : ℕ
  children_of_sister : ℕ
  num_grandparents : ℕ
  num_cousins : ℕ

/-- Calculates the total number of packages to be sent -/
def total_packages (p : GiftSendingProblem) : ℕ :=
  p.num_parents + p.num_brothers + p.num_sisters +
  (p.num_brothers * p.children_per_brother) +
  p.children_of_sister + p.num_grandparents + p.num_cousins

/-- Calculates the total cost of sending all packages -/
def total_cost (p : GiftSendingProblem) : ℕ :=
  p.cost_per_package * total_packages p

/-- Theorem stating that the total cost for Molly's specific situation is $182 -/
theorem mollys_gift_cost :
  let p : GiftSendingProblem := {
    cost_per_package := 7,
    num_parents := 2,
    num_brothers := 4,
    num_sisters := 1,
    children_per_brother := 3,
    children_of_sister := 2,
    num_grandparents := 2,
    num_cousins := 3
  }
  total_cost p = 182 := by sorry

end NUMINAMATH_CALUDE_mollys_gift_cost_l587_58751


namespace NUMINAMATH_CALUDE_sugar_solution_percentage_l587_58770

theorem sugar_solution_percentage (original_percentage : ℝ) (final_percentage : ℝ) : 
  original_percentage = 10 →
  final_percentage = 18 →
  ∃ (second_percentage : ℝ),
    second_percentage = 42 ∧
    (3/4 * original_percentage + 1/4 * second_percentage) / 100 = final_percentage / 100 :=
by sorry

end NUMINAMATH_CALUDE_sugar_solution_percentage_l587_58770


namespace NUMINAMATH_CALUDE_share_ratio_l587_58778

/-- Prove that the ratio of A's share to the combined share of B and C is 2:3 --/
theorem share_ratio (total a b c : ℚ) (x : ℚ) : 
  total = 200 →
  a = 80 →
  a = x * (b + c) →
  b = (6/9) * (a + c) →
  a + b + c = total →
  a / (b + c) = 2/3 :=
by sorry

end NUMINAMATH_CALUDE_share_ratio_l587_58778


namespace NUMINAMATH_CALUDE_journey_gas_cost_l587_58743

/-- Calculates the cost of gas for a journey given odometer readings, fuel efficiency, and gas price -/
def gas_cost (initial_reading : ℕ) (final_reading : ℕ) (fuel_efficiency : ℚ) (gas_price : ℚ) : ℚ :=
  ((final_reading - initial_reading : ℚ) / fuel_efficiency) * gas_price

/-- Proves that the cost of gas for the given journey is $3.46 -/
theorem journey_gas_cost :
  gas_cost 85340 85368 32 (395/100) = 346/100 := by
  sorry

end NUMINAMATH_CALUDE_journey_gas_cost_l587_58743


namespace NUMINAMATH_CALUDE_sum_of_square_areas_l587_58722

/-- Given three squares with side lengths satisfying certain conditions, 
    prove that the sum of their areas is 189. -/
theorem sum_of_square_areas (x a b : ℝ) 
  (h1 : a + b + x = 23)
  (h2 : 9 ≤ (min a b)^2)
  (h3 : (min a b)^2 ≤ 25)
  (h4 : max a b ≥ 5) :
  x^2 + a^2 + b^2 = 189 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_square_areas_l587_58722


namespace NUMINAMATH_CALUDE_unique_number_triple_and_square_l587_58717

theorem unique_number_triple_and_square (x : ℝ) : 
  (x > 0 ∧ 3 * x = (x / 2)^2 + 45) ↔ x = 18 := by
  sorry

end NUMINAMATH_CALUDE_unique_number_triple_and_square_l587_58717


namespace NUMINAMATH_CALUDE_monday_count_l587_58701

/-- The number of Mondays on which it rained -/
def num_mondays : ℕ := sorry

/-- The rainfall on each Monday in centimeters -/
def rainfall_per_monday : ℚ := 3/2

/-- The number of Tuesdays on which it rained -/
def num_tuesdays : ℕ := 9

/-- The rainfall on each Tuesday in centimeters -/
def rainfall_per_tuesday : ℚ := 5/2

/-- The difference in total rainfall between Tuesdays and Mondays in centimeters -/
def rainfall_difference : ℚ := 12

theorem monday_count : 
  num_mondays * rainfall_per_monday + rainfall_difference = 
  num_tuesdays * rainfall_per_tuesday ∧ num_mondays = 7 := by sorry

end NUMINAMATH_CALUDE_monday_count_l587_58701


namespace NUMINAMATH_CALUDE_range_of_c_l587_58792

def is_decreasing (f : ℝ → ℝ) := ∀ x y, x < y → f y < f x

def has_two_distinct_real_roots (a b c : ℝ) := 
  b^2 - 4*a*c > 0

def proposition_p (c : ℝ) := is_decreasing (fun x ↦ c^x)

def proposition_q (c : ℝ) := has_two_distinct_real_roots 1 (2 * Real.sqrt c) (1/2)

theorem range_of_c (c : ℝ) 
  (h1 : c > 0) 
  (h2 : c ≠ 1) 
  (h3 : proposition_p c ∨ proposition_q c) 
  (h4 : ¬(proposition_p c ∧ proposition_q c)) : 
  c ∈ Set.Ioc 0 (1/2) ∪ Set.Ioi 1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_c_l587_58792


namespace NUMINAMATH_CALUDE_max_disjoint_paths_iff_equal_outgoing_roads_l587_58788

/-- Represents a city in the network -/
structure City where
  id : Nat

/-- Represents the road network -/
structure RoadNetwork where
  n : Nat
  cities : Finset City
  roads : City → City → Prop

/-- The maximum number of disjoint paths between two cities -/
def maxDisjointPaths (net : RoadNetwork) (start finish : City) : Nat :=
  sorry

/-- The number of outgoing roads from a city -/
def outgoingRoads (net : RoadNetwork) (city : City) : Nat :=
  sorry

theorem max_disjoint_paths_iff_equal_outgoing_roads
  (net : RoadNetwork) (A V : City) :
  maxDisjointPaths net A V = maxDisjointPaths net V A ↔
  outgoingRoads net A = outgoingRoads net V :=
by sorry

end NUMINAMATH_CALUDE_max_disjoint_paths_iff_equal_outgoing_roads_l587_58788


namespace NUMINAMATH_CALUDE_dragon_legs_correct_l587_58758

/-- Represents the number of legs of a three-headed dragon -/
def dragon_legs : ℕ := 14

/-- Represents the number of centipedes -/
def num_centipedes : ℕ := 5

/-- Represents the number of three-headed dragons -/
def num_dragons : ℕ := 7

/-- The total number of heads in the herd -/
def total_heads : ℕ := 26

/-- The total number of legs in the herd -/
def total_legs : ℕ := 298

/-- Each centipede has one head -/
def centipede_heads : ℕ := 1

/-- Each centipede has 40 legs -/
def centipede_legs : ℕ := 40

/-- Each dragon has three heads -/
def dragon_heads : ℕ := 3

theorem dragon_legs_correct :
  (num_centipedes * centipede_heads + num_dragons * dragon_heads = total_heads) ∧
  (num_centipedes * centipede_legs + num_dragons * dragon_legs = total_legs) :=
by sorry

end NUMINAMATH_CALUDE_dragon_legs_correct_l587_58758


namespace NUMINAMATH_CALUDE_equal_distribution_of_eggs_l587_58764

-- Define the number of eggs
def total_eggs : ℕ := 2 * 12

-- Define the number of people
def num_people : ℕ := 4

-- Define the function to calculate eggs per person
def eggs_per_person (total : ℕ) (people : ℕ) : ℕ := total / people

-- Theorem to prove
theorem equal_distribution_of_eggs :
  eggs_per_person total_eggs num_people = 6 := by
  sorry

end NUMINAMATH_CALUDE_equal_distribution_of_eggs_l587_58764


namespace NUMINAMATH_CALUDE_pizza_toppings_combinations_l587_58786

theorem pizza_toppings_combinations (n m : ℕ) (h1 : n = 8) (h2 : m = 5) : 
  Nat.choose n m = 56 := by
  sorry

end NUMINAMATH_CALUDE_pizza_toppings_combinations_l587_58786


namespace NUMINAMATH_CALUDE_perpendicular_line_through_point_l587_58767

/-- The equation of a line perpendicular to 2x + y - 5 = 0 and passing through (3, 0) -/
theorem perpendicular_line_through_point (x y : ℝ) :
  (2 : ℝ) * x + y - 5 = 0 →  -- Given line
  (∃ c : ℝ, x - 2 * y + c = 0 ∧  -- General form of perpendicular line
            3 - 2 * 0 + c = 0 ∧  -- Passes through (3, 0)
            x - 2 * y - 3 = 0) :=  -- The specific equation we want to prove
by sorry

end NUMINAMATH_CALUDE_perpendicular_line_through_point_l587_58767


namespace NUMINAMATH_CALUDE_sum_of_cosines_l587_58774

theorem sum_of_cosines (z : ℂ) (α : ℝ) (h1 : z^7 = 1) (h2 : z ≠ 1) (h3 : z = Complex.exp (Complex.I * α)) :
  Real.cos α + Real.cos (2 * α) + Real.cos (4 * α) = -1/2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_cosines_l587_58774


namespace NUMINAMATH_CALUDE_person_height_from_shadow_ratio_l587_58733

/-- Proves that given a tree's height and shadow length, and a person's shadow length,
    we can determine the person's height assuming a constant ratio of height to shadow length. -/
theorem person_height_from_shadow_ratio (tree_height tree_shadow person_shadow : ℝ) 
  (h1 : tree_height = 50) 
  (h2 : tree_shadow = 25)
  (h3 : person_shadow = 20) :
  (tree_height / tree_shadow) * person_shadow = 40 := by
  sorry

end NUMINAMATH_CALUDE_person_height_from_shadow_ratio_l587_58733


namespace NUMINAMATH_CALUDE_sandwich_combinations_l587_58773

theorem sandwich_combinations (meat : ℕ) (cheese : ℕ) (condiment : ℕ) :
  meat = 12 →
  cheese = 8 →
  condiment = 5 →
  (meat) * (cheese.choose 2) * (condiment) = 1680 :=
by
  sorry

end NUMINAMATH_CALUDE_sandwich_combinations_l587_58773


namespace NUMINAMATH_CALUDE_smallest_visible_sum_l587_58712

/-- Represents a die in the cube -/
structure Die :=
  (faces : Fin 6 → ℕ)
  (opposite_sum : ∀ i : Fin 3, faces i + faces (i + 3) = 7)

/-- Represents the 4x4x4 cube made of dice -/
def Cube := Fin 4 → Fin 4 → Fin 4 → Die

/-- Calculates the sum of visible faces on the large cube -/
def visible_sum (c : Cube) : ℕ :=
  sorry

theorem smallest_visible_sum (c : Cube) : 
  visible_sum c ≥ 144 :=
sorry

end NUMINAMATH_CALUDE_smallest_visible_sum_l587_58712


namespace NUMINAMATH_CALUDE_function_with_two_integer_solutions_l587_58746

theorem function_with_two_integer_solutions (a : ℝ) : 
  (∃ x y : ℤ, x ≠ y ∧ 
   (∀ z : ℤ, (Real.log (↑z) - a * (↑z)^2 - (a - 2) * ↑z > 0) ↔ (z = x ∨ z = y))) →
  (1 < a ∧ a ≤ (4 + Real.log 2) / 6) :=
sorry

end NUMINAMATH_CALUDE_function_with_two_integer_solutions_l587_58746


namespace NUMINAMATH_CALUDE_min_ratio_folded_strings_l587_58725

theorem min_ratio_folded_strings (m n : ℕ) : 
  (∃ a : ℕ+, (2^m + 1 : ℕ) = a * (2^n + 1) ∧ a > 1) → 
  (∃ a : ℕ+, (2^m + 1 : ℕ) = a * (2^n + 1) ∧ a > 1 ∧ 
    ∀ b : ℕ+, (2^m + 1 : ℕ) = b * (2^n + 1) ∧ b > 1 → a ≤ b) → 
  (∃ m n : ℕ, (2^m + 1 : ℕ) = 3 * (2^n + 1)) :=
by sorry

end NUMINAMATH_CALUDE_min_ratio_folded_strings_l587_58725


namespace NUMINAMATH_CALUDE_sum_of_x_and_y_l587_58727

theorem sum_of_x_and_y (x y : ℝ) (hx : 3 + x = 5) (hy : -3 + y = 5) : x + y = 10 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_x_and_y_l587_58727


namespace NUMINAMATH_CALUDE_maintenance_interval_doubled_l587_58702

/-- 
Given an original maintenance check interval and a percentage increase,
this function calculates the new maintenance check interval.
-/
def new_maintenance_interval (original : ℕ) (percent_increase : ℕ) : ℕ :=
  original * (100 + percent_increase) / 100

/-- 
Theorem: If the original maintenance check interval is 30 days and 
the interval is increased by 100%, then the new interval is 60 days.
-/
theorem maintenance_interval_doubled :
  new_maintenance_interval 30 100 = 60 := by
  sorry

end NUMINAMATH_CALUDE_maintenance_interval_doubled_l587_58702


namespace NUMINAMATH_CALUDE_prime_consecutive_divisibility_l587_58797

theorem prime_consecutive_divisibility (p : ℤ) : 
  Prime p → p > 3 → Prime (p + 2) → (6 : ℤ) ∣ (p + 1) := by
  sorry

end NUMINAMATH_CALUDE_prime_consecutive_divisibility_l587_58797


namespace NUMINAMATH_CALUDE_line_equation_from_circle_and_symmetry_l587_58748

/-- The equation of a line given a circle and a point of symmetry -/
theorem line_equation_from_circle_and_symmetry (x y : ℝ) :
  let circle := {(x, y) | x^2 + (y - 4)^2 = 4}
  let center := (0, 4)
  let P := (2, 0)
  ∃ l : Set (ℝ × ℝ), 
    (∀ (p : ℝ × ℝ), p ∈ l ↔ x - 2*y + 3 = 0) ∧
    (∀ (q : ℝ × ℝ), q ∈ circle → ∃ (r : ℝ × ℝ), r ∈ l ∧ 
      center.1 + r.1 = q.1 + P.1 ∧ 
      center.2 + r.2 = q.2 + P.2) := by
  sorry

end NUMINAMATH_CALUDE_line_equation_from_circle_and_symmetry_l587_58748


namespace NUMINAMATH_CALUDE_maggie_plant_books_l587_58753

/-- The number of books about plants Maggie bought -/
def num_plant_books : ℕ := 9

/-- The number of books about fish Maggie bought -/
def num_fish_books : ℕ := 1

/-- The number of science magazines Maggie bought -/
def num_magazines : ℕ := 10

/-- The cost of each book in dollars -/
def book_cost : ℕ := 15

/-- The cost of each magazine in dollars -/
def magazine_cost : ℕ := 2

/-- The total amount Maggie spent in dollars -/
def total_spent : ℕ := 170

theorem maggie_plant_books :
  num_plant_books * book_cost + num_fish_books * book_cost + num_magazines * magazine_cost = total_spent :=
by sorry

end NUMINAMATH_CALUDE_maggie_plant_books_l587_58753


namespace NUMINAMATH_CALUDE_average_lawn_cuts_l587_58735

/-- The number of months in a year -/
def months_in_year : ℕ := 12

/-- The number of months Mr. Roper cuts his lawn 15 times -/
def high_frequency_months : ℕ := 6

/-- The number of months Mr. Roper cuts his lawn 3 times -/
def low_frequency_months : ℕ := 6

/-- The number of times Mr. Roper cuts his lawn in high frequency months -/
def high_frequency_cuts : ℕ := 15

/-- The number of times Mr. Roper cuts his lawn in low frequency months -/
def low_frequency_cuts : ℕ := 3

/-- Theorem stating the average number of times Mr. Roper cuts his yard per month -/
theorem average_lawn_cuts :
  (high_frequency_months * high_frequency_cuts + low_frequency_months * low_frequency_cuts) / months_in_year = 9 := by
  sorry

end NUMINAMATH_CALUDE_average_lawn_cuts_l587_58735


namespace NUMINAMATH_CALUDE_trophy_count_proof_l587_58799

/-- The number of trophies Michael has right now -/
def michael_trophies : ℕ := 30

/-- The number of trophies Jack will have in three years -/
def jack_future_trophies : ℕ := 10 * michael_trophies

/-- The number of trophies Michael will have in three years -/
def michael_future_trophies : ℕ := michael_trophies + 100

theorem trophy_count_proof :
  michael_trophies = 30 ∧
  jack_future_trophies = 10 * michael_trophies ∧
  michael_future_trophies = michael_trophies + 100 ∧
  jack_future_trophies + michael_future_trophies = 430 :=
by sorry

end NUMINAMATH_CALUDE_trophy_count_proof_l587_58799


namespace NUMINAMATH_CALUDE_stratified_sampling_juniors_l587_58705

theorem stratified_sampling_juniors 
  (total_students : ℕ) 
  (juniors : ℕ) 
  (sample_size : ℕ) 
  (h1 : total_students = 1200)
  (h2 : juniors = 500)
  (h3 : sample_size = 120) :
  (juniors : ℚ) / total_students * sample_size = 50 := by
sorry

end NUMINAMATH_CALUDE_stratified_sampling_juniors_l587_58705


namespace NUMINAMATH_CALUDE_decomposable_exponential_linear_cos_decomposable_l587_58732

-- Define a decomposable function
def Decomposable (f : ℝ → ℝ) : Prop :=
  ∃ x₀ : ℝ, f (x₀ + 1) = f x₀ + f 1

-- Theorem for part 2
theorem decomposable_exponential_linear (b : ℝ) :
  Decomposable (λ x => 2*x + b + 2^x) → b > -2 :=
by sorry

-- Theorem for part 3
theorem cos_decomposable :
  Decomposable cos :=
by sorry

end NUMINAMATH_CALUDE_decomposable_exponential_linear_cos_decomposable_l587_58732


namespace NUMINAMATH_CALUDE_first_discount_percentage_l587_58765

theorem first_discount_percentage (initial_price final_price : ℝ) 
  (second_discount : ℝ) (h1 : initial_price = 560) 
  (h2 : final_price = 313.6) (h3 : second_discount = 0.3) : 
  ∃ (first_discount : ℝ), 
    first_discount = 0.2 ∧ 
    final_price = initial_price * (1 - first_discount) * (1 - second_discount) :=
by sorry

end NUMINAMATH_CALUDE_first_discount_percentage_l587_58765


namespace NUMINAMATH_CALUDE_frog_jumps_theorem_l587_58791

/-- Represents a hexagon with vertices A, B, C, D, E, F -/
inductive Vertex
| A | B | C | D | E | F

/-- Represents the number of paths from A to C in n jumps -/
def paths_to_C (n : ℕ) : ℕ := (2^n - 1) / 3

/-- Represents the number of paths from A to C in n jumps avoiding D -/
def paths_to_C_avoiding_D (n : ℕ) : ℕ := 3^(n/2 - 1)

/-- Represents the probability of survival after n jumps with a mine at D -/
def survival_probability (n : ℕ) : ℚ := (3/4)^((n + 1)/2 - 1)

/-- The average lifespan of frogs -/
def average_lifespan : ℕ := 9

/-- Main theorem stating the properties of frog jumps on a hexagon -/
theorem frog_jumps_theorem :
  ∀ n : ℕ,
  (paths_to_C n = (2^n - 1) / 3) ∧
  (paths_to_C_avoiding_D n = 3^(n/2 - 1)) ∧
  (survival_probability n = (3/4)^((n + 1)/2 - 1)) ∧
  (average_lifespan = 9) :=
by sorry

end NUMINAMATH_CALUDE_frog_jumps_theorem_l587_58791


namespace NUMINAMATH_CALUDE_extra_apples_l587_58710

theorem extra_apples (red_apples green_apples students : ℕ) 
  (h1 : red_apples = 43)
  (h2 : green_apples = 32)
  (h3 : students = 2) :
  red_apples + green_apples - students = 73 := by
  sorry

end NUMINAMATH_CALUDE_extra_apples_l587_58710
