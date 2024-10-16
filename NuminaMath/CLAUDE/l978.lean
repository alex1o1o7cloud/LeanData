import Mathlib

namespace NUMINAMATH_CALUDE_black_men_tshirt_cost_l978_97822

/-- Represents the cost of t-shirts and number of employees --/
structure TShirtData where
  white_men_cost : ℝ
  black_men_cost : ℝ
  total_employees : ℕ
  total_spent : ℝ

/-- Theorem stating the cost of black men's t-shirts --/
theorem black_men_tshirt_cost (data : TShirtData) 
  (h1 : data.white_men_cost = 20)
  (h2 : data.total_employees = 40)
  (h3 : data.total_spent = 660)
  (h4 : ∃ (n : ℕ), n * 4 = data.total_employees) :
  data.black_men_cost = 18 := by
  sorry

end NUMINAMATH_CALUDE_black_men_tshirt_cost_l978_97822


namespace NUMINAMATH_CALUDE_inverse_false_implies_negation_false_l978_97876

theorem inverse_false_implies_negation_false (p : Prop) :
  (¬p → False) → (¬p = False) := by
  sorry

end NUMINAMATH_CALUDE_inverse_false_implies_negation_false_l978_97876


namespace NUMINAMATH_CALUDE_tower_height_ratio_l978_97843

theorem tower_height_ratio :
  ∀ (grace_height clyde_height : ℕ),
    grace_height = 40 →
    grace_height = clyde_height + 35 →
    (grace_height : ℚ) / (clyde_height : ℚ) = 8 := by
  sorry

end NUMINAMATH_CALUDE_tower_height_ratio_l978_97843


namespace NUMINAMATH_CALUDE_seven_ways_to_make_eight_cents_l978_97812

/-- Represents the number of ways to make a certain amount with given coins -/
def num_ways_to_make_amount (one_cent : ℕ) (two_cent : ℕ) (five_cent : ℕ) (target : ℕ) : ℕ :=
  sorry

/-- Theorem stating that there are 7 ways to make 8 cents with the given coins -/
theorem seven_ways_to_make_eight_cents :
  num_ways_to_make_amount 8 4 1 8 = 7 := by
  sorry

end NUMINAMATH_CALUDE_seven_ways_to_make_eight_cents_l978_97812


namespace NUMINAMATH_CALUDE_sufficient_but_not_necessary_l978_97846

theorem sufficient_but_not_necessary (a : ℝ) : 
  (a = 2 → (a - 1) * (a - 2) = 0) ∧ 
  ¬((a - 1) * (a - 2) = 0 → a = 2) :=
by sorry

end NUMINAMATH_CALUDE_sufficient_but_not_necessary_l978_97846


namespace NUMINAMATH_CALUDE_xiao_he_purchase_cost_l978_97814

/-- The total cost of Xiao He's purchase -/
def total_cost (notebook_price pen_price : ℝ) : ℝ :=
  4 * notebook_price + 10 * pen_price

/-- Theorem: The total cost of Xiao He's purchase is 4a + 10b -/
theorem xiao_he_purchase_cost (a b : ℝ) :
  total_cost a b = 4 * a + 10 * b := by
  sorry

end NUMINAMATH_CALUDE_xiao_he_purchase_cost_l978_97814


namespace NUMINAMATH_CALUDE_yellow_balls_count_l978_97861

def total_balls : ℕ := 60
def white_balls : ℕ := 22
def green_balls : ℕ := 10
def red_balls : ℕ := 15
def purple_balls : ℕ := 6
def prob_not_red_or_purple : ℚ := 65/100

theorem yellow_balls_count :
  ∃ (y : ℕ), y = total_balls - (white_balls + green_balls + red_balls + purple_balls) ∧
  (white_balls + green_balls + y : ℚ) / total_balls = prob_not_red_or_purple :=
by sorry

end NUMINAMATH_CALUDE_yellow_balls_count_l978_97861


namespace NUMINAMATH_CALUDE_area_between_concentric_circles_l978_97806

/-- The area of the region between two concentric circles, where the diameter of the larger circle
    is twice the diameter of the smaller circle, and the smaller circle has a diameter of 4 units,
    is equal to 12π square units. -/
theorem area_between_concentric_circles (π : ℝ) : 
  let d_small : ℝ := 4
  let r_small : ℝ := d_small / 2
  let r_large : ℝ := 2 * r_small
  let area_small : ℝ := π * r_small^2
  let area_large : ℝ := π * r_large^2
  area_large - area_small = 12 * π :=
by sorry

end NUMINAMATH_CALUDE_area_between_concentric_circles_l978_97806


namespace NUMINAMATH_CALUDE_exponent_multiplication_l978_97853

theorem exponent_multiplication (x : ℝ) : x^2 * x^3 = x^5 := by
  sorry

end NUMINAMATH_CALUDE_exponent_multiplication_l978_97853


namespace NUMINAMATH_CALUDE_value_of_expression_l978_97868

theorem value_of_expression (x : ℝ) (h : x = 5) : 2 * x^2 + 3 = 53 := by
  sorry

end NUMINAMATH_CALUDE_value_of_expression_l978_97868


namespace NUMINAMATH_CALUDE_indefinite_integral_proof_l978_97848

theorem indefinite_integral_proof (x : Real) :
  let f := fun x => -(1 / (x - Real.sin x))
  let g := fun x => (1 - Real.cos x) / (x - Real.sin x)^2
  deriv f x = g x :=
by sorry

end NUMINAMATH_CALUDE_indefinite_integral_proof_l978_97848


namespace NUMINAMATH_CALUDE_max_value_of_a_l978_97881

theorem max_value_of_a (f : ℝ → ℝ) (a : ℝ) :
  (∀ x, f x = |x - 5/2| + |x - a|) →
  (∀ x, f x ≥ a) →
  ∃ a_max : ℝ, a_max = 5/4 ∧ ∀ a' : ℝ, (∀ x, |x - 5/2| + |x - a'| ≥ a') → a' ≤ a_max :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_a_l978_97881


namespace NUMINAMATH_CALUDE_minimum_peanuts_min_peanuts_is_25_l978_97821

theorem minimum_peanuts : ℕ → Prop :=
  fun n => (n % 3 = 1) ∧ 
           ((n - 1) / 3 - 1) % 3 = 0 ∧ 
           (((n - 1) / 3 - 1 - 1) / 3 - 1) % 3 = 0

theorem min_peanuts_is_25 : minimum_peanuts 25 ∧ ∀ m < 25, ¬minimum_peanuts m := by
  sorry

end NUMINAMATH_CALUDE_minimum_peanuts_min_peanuts_is_25_l978_97821


namespace NUMINAMATH_CALUDE_total_working_days_l978_97809

/-- Commute options for a person over a period of working days. -/
structure CommuteData where
  /-- Number of days driving car in the morning and riding bicycle in the afternoon -/
  car_morning_bike_afternoon : ℕ
  /-- Number of days riding bicycle in the morning and driving car in the afternoon -/
  bike_morning_car_afternoon : ℕ
  /-- Number of days using only bicycle both morning and afternoon -/
  bike_only : ℕ

/-- Theorem stating the total number of working days based on given commute data. -/
theorem total_working_days (data : CommuteData) : 
  data.car_morning_bike_afternoon + data.bike_morning_car_afternoon + data.bike_only = 23 :=
  by
  have morning_car : data.car_morning_bike_afternoon + data.bike_only = 12 := by sorry
  have afternoon_bike : data.bike_morning_car_afternoon + data.bike_only = 20 := by sorry
  have total_car : data.car_morning_bike_afternoon + data.bike_morning_car_afternoon = 14 := by sorry
  sorry

#check total_working_days

end NUMINAMATH_CALUDE_total_working_days_l978_97809


namespace NUMINAMATH_CALUDE_binary_representation_253_l978_97864

def decimal_to_binary (n : ℕ) : List Bool :=
  sorry

def count_ones (binary : List Bool) : ℕ :=
  sorry

def count_zeros (binary : List Bool) : ℕ :=
  sorry

theorem binary_representation_253 :
  let binary := decimal_to_binary 253
  let y := count_ones binary
  let x := count_zeros binary
  y - x = 6 := by sorry

end NUMINAMATH_CALUDE_binary_representation_253_l978_97864


namespace NUMINAMATH_CALUDE_perpendicular_bisector_of_intersection_points_l978_97835

/-- First circle equation -/
def circle1 (x y : ℝ) : Prop := x^2 + y^2 - 4*x + 6*y = 0

/-- Second circle equation -/
def circle2 (x y : ℝ) : Prop := x^2 + y^2 - 6*x = 0

/-- Perpendicular bisector equation -/
def perp_bisector (x y : ℝ) : Prop := 3*x - y - 9 = 0

/-- Theorem stating that the perpendicular bisector of AB is 3x - y - 9 = 0 -/
theorem perpendicular_bisector_of_intersection_points :
  ∃ (A B : ℝ × ℝ), 
    (circle1 A.1 A.2 ∧ circle2 A.1 A.2) ∧ 
    (circle1 B.1 B.2 ∧ circle2 B.1 B.2) ∧ 
    (∀ (x y : ℝ), perp_bisector x y ↔ 
      (x - (A.1 + B.1)/2)^2 + (y - (A.2 + B.2)/2)^2 = 
      ((A.1 - B.1)^2 + (A.2 - B.2)^2) / 4) :=
sorry

end NUMINAMATH_CALUDE_perpendicular_bisector_of_intersection_points_l978_97835


namespace NUMINAMATH_CALUDE_convex_polygon_30_sides_diagonals_l978_97871

/-- The number of diagonals in a convex polygon with n sides -/
def num_diagonals (n : ℕ) : ℕ :=
  (n * (n - 3)) / 2

/-- Theorem: A convex polygon with 30 sides has 202 diagonals -/
theorem convex_polygon_30_sides_diagonals :
  num_diagonals 30 = 202 := by
  sorry

end NUMINAMATH_CALUDE_convex_polygon_30_sides_diagonals_l978_97871


namespace NUMINAMATH_CALUDE_water_from_river_calculation_l978_97863

/-- The amount of water Jacob collects from the river daily -/
def water_from_river : ℕ := 1700

/-- Jacob's water tank capacity in milliliters -/
def tank_capacity : ℕ := 50000

/-- Water collected from rain daily in milliliters -/
def water_from_rain : ℕ := 800

/-- Number of days to fill the tank -/
def days_to_fill : ℕ := 20

/-- Theorem stating that the amount of water Jacob collects from the river daily is 1700 milliliters -/
theorem water_from_river_calculation :
  water_from_river = (tank_capacity - water_from_rain * days_to_fill) / days_to_fill :=
by sorry

end NUMINAMATH_CALUDE_water_from_river_calculation_l978_97863


namespace NUMINAMATH_CALUDE_megan_cupcakes_l978_97804

theorem megan_cupcakes (todd_ate : ℕ) (packages : ℕ) (cupcakes_per_package : ℕ) 
  (h1 : todd_ate = 32)
  (h2 : packages = 6)
  (h3 : cupcakes_per_package = 6) :
  todd_ate + packages * cupcakes_per_package = 68 := by
  sorry

end NUMINAMATH_CALUDE_megan_cupcakes_l978_97804


namespace NUMINAMATH_CALUDE_average_monthly_balance_l978_97891

def monthly_balances : List ℝ := [120, 150, 180, 150, 210, 180]

theorem average_monthly_balance :
  (monthly_balances.sum / monthly_balances.length : ℝ) = 165 := by sorry

end NUMINAMATH_CALUDE_average_monthly_balance_l978_97891


namespace NUMINAMATH_CALUDE_arcsin_one_half_l978_97836

theorem arcsin_one_half : Real.arcsin (1/2) = π/6 := by
  sorry

end NUMINAMATH_CALUDE_arcsin_one_half_l978_97836


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_l978_97872

/-- An arithmetic sequence -/
def ArithmeticSequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_sum (a : ℕ → ℝ) :
  ArithmeticSequence a →
  (a 3 + a 4 + a 5 + a 6 + a 7 + a 8 + a 9 = 420) →
  (a 2 + a 10 = 120) := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_l978_97872


namespace NUMINAMATH_CALUDE_chromium_percentage_in_new_alloy_l978_97878

/-- Calculates the percentage of chromium in a new alloy formed by combining two alloys -/
theorem chromium_percentage_in_new_alloy 
  (weight1 : ℝ) (percentage1 : ℝ) 
  (weight2 : ℝ) (percentage2 : ℝ) :
  weight1 = 10 →
  weight2 = 30 →
  percentage1 = 12 →
  percentage2 = 8 →
  (weight1 * percentage1 / 100 + weight2 * percentage2 / 100) / (weight1 + weight2) * 100 = 9 :=
by sorry

end NUMINAMATH_CALUDE_chromium_percentage_in_new_alloy_l978_97878


namespace NUMINAMATH_CALUDE_root_sum_theorem_l978_97874

theorem root_sum_theorem (a b c : ℝ) : 
  (a^3 - 6*a^2 + 8*a - 3 = 0) → 
  (b^3 - 6*b^2 + 8*b - 3 = 0) → 
  (c^3 - 6*c^2 + 8*c - 3 = 0) → 
  (a/(b*c + 2) + b/(a*c + 2) + c/(a*b + 2) = 0) := by
sorry

end NUMINAMATH_CALUDE_root_sum_theorem_l978_97874


namespace NUMINAMATH_CALUDE_tan_eq_two_solution_set_l978_97832

theorem tan_eq_two_solution_set :
  {x : ℝ | ∃ k : ℤ, x = k * Real.pi + Real.arctan 2} = {x : ℝ | Real.tan x = 2} := by
sorry

end NUMINAMATH_CALUDE_tan_eq_two_solution_set_l978_97832


namespace NUMINAMATH_CALUDE_bob_second_week_hours_l978_97858

/-- Calculates the total pay for a given number of hours worked --/
def calculatePay (hours : ℕ) : ℕ :=
  if hours ≤ 40 then
    hours * 5
  else
    40 * 5 + (hours - 40) * 6

theorem bob_second_week_hours :
  ∃ (second_week_hours : ℕ),
    calculatePay 44 + calculatePay second_week_hours = 472 ∧
    second_week_hours = 48 := by
  sorry

end NUMINAMATH_CALUDE_bob_second_week_hours_l978_97858


namespace NUMINAMATH_CALUDE_radish_basket_difference_l978_97884

theorem radish_basket_difference (total : ℕ) (first_basket : ℕ) : total = 88 → first_basket = 37 → total - first_basket - first_basket = 14 := by
  sorry

end NUMINAMATH_CALUDE_radish_basket_difference_l978_97884


namespace NUMINAMATH_CALUDE_prime_square_mod_twelve_l978_97879

theorem prime_square_mod_twelve (p : ℕ) (hp : Nat.Prime p) (hp_gt_3 : p > 3) :
  p ^ 2 % 12 = 1 := by
sorry

end NUMINAMATH_CALUDE_prime_square_mod_twelve_l978_97879


namespace NUMINAMATH_CALUDE_z_range_difference_l978_97865

theorem z_range_difference (x y z : ℝ) 
  (sum_eq : x + y + z = 2) 
  (prod_eq : x * y + y * z + x * z = 0) : 
  ∃ (a b : ℝ), (∀ z', (∃ x' y', x' + y' + z' = 2 ∧ x' * y' + y' * z' + x' * z' = 0) → a ≤ z' ∧ z' ≤ b) ∧ b - a = 8/3 := by
  sorry

end NUMINAMATH_CALUDE_z_range_difference_l978_97865


namespace NUMINAMATH_CALUDE_smallest_b_value_l978_97857

theorem smallest_b_value (a b : ℕ+) 
  (h1 : a.val - b.val = 8)
  (h2 : Nat.gcd ((a.val^3 + b.val^3) / (a.val + b.val)) (a.val * b.val) = 16) :
  ∀ x : ℕ+, x.val < b.val → 
    (∃ y : ℕ+, y.val - x.val = 8 ∧ 
      Nat.gcd ((y.val^3 + x.val^3) / (y.val + x.val)) (y.val * x.val) ≠ 16) ∧
    b.val = 3 :=
sorry

end NUMINAMATH_CALUDE_smallest_b_value_l978_97857


namespace NUMINAMATH_CALUDE_sams_age_l978_97801

theorem sams_age (billy joe sam : ℕ) 
  (h1 : billy = 2 * joe) 
  (h2 : billy + joe = 60) 
  (h3 : sam = (billy + joe) / 2) : 
  sam = 30 := by
sorry

end NUMINAMATH_CALUDE_sams_age_l978_97801


namespace NUMINAMATH_CALUDE_f_monotone_and_no_min_l978_97850

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := (x - a - 1) * Real.exp (x - 1) - (1/2) * x^2 + a * x

theorem f_monotone_and_no_min (x : ℝ) (hx : x > 0) :
  (∀ x₁ x₂, x₁ > 0 → x₂ > 0 → x₁ < x₂ → f 1 x₁ < f 1 x₂) ∧
  (∃ a₁ a₂ : ℤ, (∀ x > 0, ∃ y > x, f a₁ y < f a₁ x) ∧
                (∀ x > 0, ∃ y > x, f a₂ y < f a₂ x) ∧
                a₁ + a₂ = 3) :=
by sorry

end NUMINAMATH_CALUDE_f_monotone_and_no_min_l978_97850


namespace NUMINAMATH_CALUDE_equation_solutions_l978_97825

theorem equation_solutions :
  (∃ x₁ x₂ : ℝ, x₁ = 2 + Real.sqrt 5 ∧ x₂ = 2 - Real.sqrt 5 ∧
    x₁^2 - 4*x₁ - 1 = 0 ∧ x₂^2 - 4*x₂ - 1 = 0) ∧
  (∃ x₁ x₂ : ℝ, x₁ = -4 ∧ x₂ = 1 ∧
    (x₁ + 4)^2 = 5*(x₁ + 4) ∧ (x₂ + 4)^2 = 5*(x₂ + 4)) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l978_97825


namespace NUMINAMATH_CALUDE_women_average_age_l978_97818

theorem women_average_age 
  (n : ℕ) 
  (A : ℝ) 
  (age1 age2 : ℕ) 
  (h1 : n = 8) 
  (h2 : age1 = 20) 
  (h3 : age2 = 22) 
  (h4 : (n * A - (age1 + age2 : ℝ) + (W1 + W2)) / n = A + 2) :
  (W1 + W2) / 2 = 29 :=
by sorry

end NUMINAMATH_CALUDE_women_average_age_l978_97818


namespace NUMINAMATH_CALUDE_gcd_product_equivalence_l978_97890

theorem gcd_product_equivalence (a m n : ℤ) : 
  Int.gcd a (m * n) = 1 ↔ Int.gcd a m = 1 ∧ Int.gcd a n = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_product_equivalence_l978_97890


namespace NUMINAMATH_CALUDE_smallest_sum_pell_equation_l978_97800

theorem smallest_sum_pell_equation :
  ∃ (x y : ℕ), x ≥ 1 ∧ y ≥ 1 ∧ x^2 - 29*y^2 = 1 ∧
  ∀ (x' y' : ℕ), x' ≥ 1 → y' ≥ 1 → x'^2 - 29*y'^2 = 1 → x + y ≤ x' + y' ∧
  x + y = 11621 :=
by sorry

end NUMINAMATH_CALUDE_smallest_sum_pell_equation_l978_97800


namespace NUMINAMATH_CALUDE_salary_savings_percentage_l978_97839

/-- Represents the percentage of salary saved -/
def P : ℝ := by sorry

theorem salary_savings_percentage :
  let S : ℝ := 20000  -- Monthly salary in Rs.
  let increase_factor : ℝ := 1.1  -- 10% increase in expenses
  let new_savings : ℝ := 200  -- New monthly savings in Rs.
  S - increase_factor * (S - P / 100 * S) = new_savings →
  P = 10 := by sorry

end NUMINAMATH_CALUDE_salary_savings_percentage_l978_97839


namespace NUMINAMATH_CALUDE_root_quadratic_equation_property_l978_97823

theorem root_quadratic_equation_property (m : ℝ) : 
  m^2 - 2*m - 3 = 0 → 2*m^2 - 4*m + 5 = 11 := by
  sorry

end NUMINAMATH_CALUDE_root_quadratic_equation_property_l978_97823


namespace NUMINAMATH_CALUDE_square_roots_problem_l978_97810

theorem square_roots_problem (m : ℝ) (a : ℝ) (h1 : a > 0) 
  (h2 : (2 * m - 6)^2 = a) (h3 : (m + 3)^2 = a) (h4 : 2 * m - 6 ≠ m + 3) : m = 1 := by
  sorry

end NUMINAMATH_CALUDE_square_roots_problem_l978_97810


namespace NUMINAMATH_CALUDE_max_m_value_l978_97841

noncomputable def f (m : ℝ) (x : ℝ) : ℝ := x * Real.log x + x^2 - m*x + Real.exp (2 - x)

theorem max_m_value :
  ∃ (m_max : ℝ), m_max = 3 ∧ 
  (∀ (m : ℝ), (∀ (x : ℝ), x > 0 → f m x ≥ 0) → m ≤ m_max) ∧
  (∀ (x : ℝ), x > 0 → f m_max x ≥ 0) :=
sorry

end NUMINAMATH_CALUDE_max_m_value_l978_97841


namespace NUMINAMATH_CALUDE_initial_persons_count_l978_97892

/-- The initial number of persons in a group where:
    1. The average weight increases by 4.5 kg when a new person joins.
    2. The person being replaced weighs 65 kg.
    3. The new person weighs 101 kg. -/
def initialPersons : ℕ := 8

theorem initial_persons_count :
  let avgWeightIncrease : ℚ := 4.5
  let replacedPersonWeight : ℕ := 65
  let newPersonWeight : ℕ := 101
  let totalWeightIncrease : ℚ := avgWeightIncrease * initialPersons
  totalWeightIncrease = (newPersonWeight - replacedPersonWeight) →
  initialPersons = 8 := by sorry

end NUMINAMATH_CALUDE_initial_persons_count_l978_97892


namespace NUMINAMATH_CALUDE_hugh_initial_candy_l978_97819

/-- The amount of candy Hugh had initially -/
def hugh_candy : ℕ := sorry

/-- The amount of candy Tommy had initially -/
def tommy_candy : ℕ := 6

/-- The amount of candy Melany had initially -/
def melany_candy : ℕ := 7

/-- The amount of candy each person had after sharing equally -/
def shared_candy : ℕ := 7

/-- The number of people sharing the candy -/
def num_people : ℕ := 3

theorem hugh_initial_candy :
  hugh_candy = 8 :=
by
  sorry

end NUMINAMATH_CALUDE_hugh_initial_candy_l978_97819


namespace NUMINAMATH_CALUDE_function_property_l978_97828

def IsMonotonic (f : ℝ → ℝ) : Prop :=
  ∀ x y, x ≤ y → f x ≤ f y

theorem function_property (f : ℝ → ℝ) (h_monotonic : IsMonotonic f) 
    (h_property : ∀ x > 0, f (f x + 2 / x) = 1) : 
  f 1 = 0 := by
  sorry

end NUMINAMATH_CALUDE_function_property_l978_97828


namespace NUMINAMATH_CALUDE_score_calculation_l978_97886

/-- Proves that given the average score and difference between subjects, we can determine the individual scores -/
theorem score_calculation (average : ℝ) (difference : ℝ) 
  (h_average : average = 96) 
  (h_difference : difference = 8) : 
  ∃ (chinese : ℝ) (math : ℝ), 
    chinese + math = 2 * average ∧ 
    math = chinese + difference ∧
    chinese = 92 ∧ 
    math = 100 := by
  sorry

end NUMINAMATH_CALUDE_score_calculation_l978_97886


namespace NUMINAMATH_CALUDE_equal_roots_quadratic_l978_97896

theorem equal_roots_quadratic (m : ℝ) : 
  (∃ x : ℝ, x^2 - x + m = 0 ∧ (∀ y : ℝ, y^2 - y + m = 0 → y = x)) → m = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_equal_roots_quadratic_l978_97896


namespace NUMINAMATH_CALUDE_triangle_side_length_l978_97820

theorem triangle_side_length (a b c : Real) (angle_A angle_B : Real) :
  angle_A = 30 * Real.pi / 180 →
  angle_B = 45 * Real.pi / 180 →
  c = 8 →
  b = 4 * Real.sqrt 2 :=
by sorry

end NUMINAMATH_CALUDE_triangle_side_length_l978_97820


namespace NUMINAMATH_CALUDE_linear_system_solution_l978_97867

theorem linear_system_solution (x y z : ℝ) 
  (eq1 : x + 2*y - z = 8) 
  (eq2 : 2*x - y + z = 18) : 
  8*x + y + z = 70 := by
sorry

end NUMINAMATH_CALUDE_linear_system_solution_l978_97867


namespace NUMINAMATH_CALUDE_tan_sum_half_angles_l978_97805

theorem tan_sum_half_angles (p q : ℝ) 
  (h1 : Real.cos p + Real.cos q = 1/3)
  (h2 : Real.sin p + Real.sin q = 5/13) : 
  Real.tan ((p + q)/2) = 15/13 := by
  sorry

end NUMINAMATH_CALUDE_tan_sum_half_angles_l978_97805


namespace NUMINAMATH_CALUDE_trigonometric_equality_l978_97851

theorem trigonometric_equality : 
  (Real.sin (20 * π / 180) * Real.cos (10 * π / 180) + 
   Real.cos (160 * π / 180) * Real.cos (100 * π / 180)) / 
  (Real.sin (21 * π / 180) * Real.cos (9 * π / 180) + 
   Real.cos (159 * π / 180) * Real.cos (99 * π / 180)) = 1 := by
  sorry

end NUMINAMATH_CALUDE_trigonometric_equality_l978_97851


namespace NUMINAMATH_CALUDE_floor_product_twenty_l978_97888

theorem floor_product_twenty (x : ℝ) : 
  ⌊x * ⌊x⌋⌋ = 20 ↔ 5 ≤ x ∧ x < 5.25 := by sorry

end NUMINAMATH_CALUDE_floor_product_twenty_l978_97888


namespace NUMINAMATH_CALUDE_square_land_side_length_l978_97838

theorem square_land_side_length (area : ℝ) (h : area = Real.sqrt 900) :
  ∃ (side : ℝ), side * side = area ∧ side = 30 := by
  sorry

end NUMINAMATH_CALUDE_square_land_side_length_l978_97838


namespace NUMINAMATH_CALUDE_largest_sample_size_l978_97837

def population : Nat := 36

theorem largest_sample_size (X : Nat) : 
  (X > 0 ∧ 
   population % X = 0 ∧ 
   population % (X + 1) ≠ 0 ∧ 
   ∀ Y : Nat, Y > X → (population % Y = 0 → population % (Y + 1) = 0)) → 
  X = 9 := by sorry

end NUMINAMATH_CALUDE_largest_sample_size_l978_97837


namespace NUMINAMATH_CALUDE_sin_sum_to_product_l978_97883

theorem sin_sum_to_product (x : ℝ) : 
  Real.sin (3 * x) + Real.sin (5 * x) = 2 * Real.sin (4 * x) * Real.cos x := by
  sorry

end NUMINAMATH_CALUDE_sin_sum_to_product_l978_97883


namespace NUMINAMATH_CALUDE_f_properties_l978_97893

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := |2*x + a| + |x - 1/a|

theorem f_properties :
  (∀ x : ℝ, (f 1 x < x + 3) ↔ (x > -3/4 ∧ x < 3/2)) ∧
  (∀ a : ℝ, a > 0 → ∀ x : ℝ, f a x ≥ Real.sqrt 2) := by
  sorry

end NUMINAMATH_CALUDE_f_properties_l978_97893


namespace NUMINAMATH_CALUDE_infinite_matrices_squared_zero_l978_97830

/-- The set of 2x2 real matrices B satisfying B^2 = 0 is infinite -/
theorem infinite_matrices_squared_zero :
  Set.Infinite {B : Matrix (Fin 2) (Fin 2) ℝ | B * B = 0} := by
  sorry

end NUMINAMATH_CALUDE_infinite_matrices_squared_zero_l978_97830


namespace NUMINAMATH_CALUDE_smallest_n_perfect_square_and_cube_l978_97866

/-- A number is a perfect square if it's equal to some integer multiplied by itself. -/
def IsPerfectSquare (n : ℕ) : Prop :=
  ∃ m : ℕ, n = m * m

/-- A number is a perfect cube if it's equal to some integer multiplied by itself twice. -/
def IsPerfectCube (n : ℕ) : Prop :=
  ∃ m : ℕ, n = m * m * m

/-- 45 is the smallest positive integer n such that 5n is a perfect square and 3n is a perfect cube. -/
theorem smallest_n_perfect_square_and_cube :
  (∀ n : ℕ, 0 < n ∧ n < 45 → ¬(IsPerfectSquare (5 * n) ∧ IsPerfectCube (3 * n))) ∧
  (IsPerfectSquare (5 * 45) ∧ IsPerfectCube (3 * 45)) :=
sorry

end NUMINAMATH_CALUDE_smallest_n_perfect_square_and_cube_l978_97866


namespace NUMINAMATH_CALUDE_selection_from_three_female_two_male_l978_97827

/-- The number of ways to select one person from a group of female and male students. -/
def selection_methods (num_female : ℕ) (num_male : ℕ) : ℕ :=
  num_female + num_male

/-- Theorem: The number of ways to select one person from 3 female students and 2 male students is 5. -/
theorem selection_from_three_female_two_male :
  selection_methods 3 2 = 5 := by
  sorry

end NUMINAMATH_CALUDE_selection_from_three_female_two_male_l978_97827


namespace NUMINAMATH_CALUDE_unique_three_digit_number_twelve_times_sum_of_digits_l978_97813

theorem unique_three_digit_number_twelve_times_sum_of_digits : 
  ∃! n : ℕ, 
    100 ≤ n ∧ n < 1000 ∧ 
    n = 12 * (n / 100 + (n / 10 % 10) + (n % 10)) := by
  sorry

end NUMINAMATH_CALUDE_unique_three_digit_number_twelve_times_sum_of_digits_l978_97813


namespace NUMINAMATH_CALUDE_bodyguard_hourly_rate_l978_97856

/-- Proves that the hourly rate for each bodyguard is $20 -/
theorem bodyguard_hourly_rate :
  let num_bodyguards : ℕ := 2
  let hours_per_day : ℕ := 8
  let days_per_week : ℕ := 7
  let total_weekly_payment : ℕ := 2240
  (num_bodyguards * hours_per_day * days_per_week * hourly_rate = total_weekly_payment) →
  hourly_rate = 20 := by
  sorry

end NUMINAMATH_CALUDE_bodyguard_hourly_rate_l978_97856


namespace NUMINAMATH_CALUDE_nested_fourth_root_equation_solution_l978_97860

noncomputable def nested_fourth_root (x : ℝ) : ℝ := Real.sqrt (x + Real.sqrt (x + Real.sqrt (x + Real.sqrt x)))

noncomputable def nested_fourth_root_product (x : ℝ) : ℝ := Real.sqrt (x * Real.sqrt (x * Real.sqrt (x * Real.sqrt x)))

def cubic_equation (y : ℝ) : Prop := y^3 - y^2 - 1 = 0

theorem nested_fourth_root_equation_solution :
  ∃ (x : ℝ), x > 0 ∧ nested_fourth_root x = nested_fourth_root_product x ∧
  ∃ (y : ℝ), cubic_equation y ∧ x = y^3 := by
  sorry

end NUMINAMATH_CALUDE_nested_fourth_root_equation_solution_l978_97860


namespace NUMINAMATH_CALUDE_boat_speed_l978_97847

theorem boat_speed (t : ℝ) (h : t > 0) : 
  let v_s : ℝ := 21
  let upstream_time : ℝ := 2 * t
  let downstream_time : ℝ := t
  let v_b : ℝ := (v_s * (upstream_time + downstream_time)) / (upstream_time - downstream_time)
  v_b = 63 := by sorry

end NUMINAMATH_CALUDE_boat_speed_l978_97847


namespace NUMINAMATH_CALUDE_min_value_a_l978_97889

theorem min_value_a (a : ℝ) (h : a > 0) : 
  (∀ x y : ℝ, x > 0 → y > 0 → (x + y) * (1/x + a/y) ≥ 9) → a ≥ 4 :=
by sorry

end NUMINAMATH_CALUDE_min_value_a_l978_97889


namespace NUMINAMATH_CALUDE_monotonic_increasing_range_l978_97808

theorem monotonic_increasing_range (a : Real) (h1 : 0 < a) (h2 : a < 1) :
  (∀ x > 0, Monotone (fun x => a^x + (1+a)^x)) →
  (a ≥ (Real.sqrt 5 - 1) / 2 ∧ a < 1) := by
  sorry

end NUMINAMATH_CALUDE_monotonic_increasing_range_l978_97808


namespace NUMINAMATH_CALUDE_problem_solution_l978_97834

theorem problem_solution (a : ℝ) (h : (a + 1/a)^2 = 5) :
  a^2 + 1/a^2 + a^3 + 1/a^3 = 3 + 2 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l978_97834


namespace NUMINAMATH_CALUDE_percentage_difference_l978_97815

theorem percentage_difference (z y x : ℝ) (total : ℝ) : 
  y = 1.2 * z →
  z = 300 →
  total = 1110 →
  x = total - y - z →
  (x - y) / y * 100 = 25 :=
by sorry

end NUMINAMATH_CALUDE_percentage_difference_l978_97815


namespace NUMINAMATH_CALUDE_inequality_proof_l978_97807

theorem inequality_proof (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) (h_abc : a * b * c = 1) :
  1 / (a^2 * (b + c)) + 1 / (b^2 * (c + a)) + 1 / (c^2 * (a + b)) ≥ 3/2 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l978_97807


namespace NUMINAMATH_CALUDE_carmela_sharing_amount_l978_97877

def carmela_money : ℕ := 7
def cousin_count : ℕ := 4
def cousin_money : ℕ := 2

def total_money : ℕ := carmela_money + cousin_count * cousin_money
def equal_share : ℕ := total_money / (cousin_count + 1)

theorem carmela_sharing_amount : 
  equal_share - cousin_money = 1 := by sorry

end NUMINAMATH_CALUDE_carmela_sharing_amount_l978_97877


namespace NUMINAMATH_CALUDE_successive_discounts_l978_97802

theorem successive_discounts (P d1 d2 : ℝ) (h1 : 0 ≤ d1 ∧ d1 < 1) (h2 : 0 ≤ d2 ∧ d2 < 1) :
  let final_price := P * (1 - d1) * (1 - d2)
  let percentage := (final_price / P) * 100
  percentage = (1 - d1) * (1 - d2) * 100 :=
by sorry

end NUMINAMATH_CALUDE_successive_discounts_l978_97802


namespace NUMINAMATH_CALUDE_distribution_count_l978_97899

theorem distribution_count (num_items : ℕ) (num_recipients : ℕ) : 
  num_items = 6 → num_recipients = 8 → num_recipients ^ num_items = 262144 := by
  sorry

end NUMINAMATH_CALUDE_distribution_count_l978_97899


namespace NUMINAMATH_CALUDE_tom_final_coin_value_l978_97826

/-- Represents the types of coins --/
inductive Coin
  | Penny
  | Nickel
  | Dime
  | Quarter
  | HalfDollar

/-- Returns the value of a coin in cents --/
def coinValue (c : Coin) : ℕ :=
  match c with
  | Coin.Penny => 1
  | Coin.Nickel => 5
  | Coin.Dime => 10
  | Coin.Quarter => 25
  | Coin.HalfDollar => 50

/-- Calculates the total value of a collection of coins --/
def totalValue (coins : List (Coin × ℕ)) : ℕ :=
  coins.foldl (fun acc (c, n) => acc + n * coinValue c) 0

/-- Tom's initial coins --/
def initialCoins : List (Coin × ℕ) :=
  [(Coin.Penny, 27), (Coin.Dime, 15), (Coin.Quarter, 9), (Coin.HalfDollar, 2)]

/-- Coins given by dad --/
def coinsFromDad : List (Coin × ℕ) :=
  [(Coin.Dime, 33), (Coin.Nickel, 49), (Coin.Quarter, 7), (Coin.HalfDollar, 4)]

/-- Coins spent by Tom --/
def spentCoins : List (Coin × ℕ) :=
  [(Coin.Dime, 11), (Coin.Quarter, 5)]

/-- Number of half dollars exchanged for quarters --/
def exchangedHalfDollars : ℕ := 5

/-- Theorem stating the final value of Tom's coins --/
theorem tom_final_coin_value :
  totalValue initialCoins +
  totalValue coinsFromDad -
  totalValue spentCoins +
  exchangedHalfDollars * 2 * coinValue Coin.Quarter =
  1702 := by sorry

end NUMINAMATH_CALUDE_tom_final_coin_value_l978_97826


namespace NUMINAMATH_CALUDE_max_y_value_l978_97880

theorem max_y_value (x y : ℤ) (h : x * y + 3 * x + 2 * y = -6) : y ≤ 3 := by
  sorry

end NUMINAMATH_CALUDE_max_y_value_l978_97880


namespace NUMINAMATH_CALUDE_helen_made_56_pies_l978_97852

/-- The number of pies Helen made -/
def helen_pies (pinky_pies total_pies : ℕ) : ℕ := total_pies - pinky_pies

/-- Proof that Helen made 56 pies -/
theorem helen_made_56_pies : helen_pies 147 203 = 56 := by
  sorry

end NUMINAMATH_CALUDE_helen_made_56_pies_l978_97852


namespace NUMINAMATH_CALUDE_eugene_pencils_l978_97885

/-- The number of pencils Eugene has after receiving more from Joyce -/
def total_pencils (initial : ℕ) (received : ℕ) : ℕ :=
  initial + received

/-- Theorem stating that Eugene has 57 pencils in total -/
theorem eugene_pencils : total_pencils 51 6 = 57 := by
  sorry

end NUMINAMATH_CALUDE_eugene_pencils_l978_97885


namespace NUMINAMATH_CALUDE_largest_value_l978_97894

theorem largest_value (a b : ℝ) (ha : 0 < a) (ha1 : a < 1) (hb : 0 < b) (hb1 : b < 1) (hab : a ≠ b) :
  (a + b) = max (a + b) (max (2 * Real.sqrt (a * b)) (max (a^2 + b^2) (2 * a * b))) :=
by sorry

end NUMINAMATH_CALUDE_largest_value_l978_97894


namespace NUMINAMATH_CALUDE_abs_3_minus_4i_l978_97811

theorem abs_3_minus_4i : Complex.abs (3 - 4 * Complex.I) = 5 := by
  sorry

end NUMINAMATH_CALUDE_abs_3_minus_4i_l978_97811


namespace NUMINAMATH_CALUDE_rectangular_field_area_l978_97859

/-- A rectangular field with width one-third of length and perimeter 80 meters has an area of 300 square meters. -/
theorem rectangular_field_area : 
  ∀ (w l : ℝ), 
  w > 0 → l > 0 →  -- Positive dimensions
  w = l / 3 →      -- Width is one-third of length
  2 * (w + l) = 80 →  -- Perimeter is 80 meters
  w * l = 300 := by  -- Area is 300 square meters
sorry


end NUMINAMATH_CALUDE_rectangular_field_area_l978_97859


namespace NUMINAMATH_CALUDE_no_solution_for_digit_difference_l978_97824

theorem no_solution_for_digit_difference : 
  ¬ ∃ (x : ℕ), x < 10 ∧ 
    (max (max (max x 3) 1) 4 * 1000 + 
     max (max (min x 3) 1) 4 * 100 + 
     min (min (max x 3) 1) 4 * 10 + 
     min (min (min x 3) 1) 4) - 
    (min (min (min x 3) 1) 4 * 1000 + 
     min (min (max x 3) 1) 4 * 100 + 
     max (max (min x 3) 1) 4 * 10 + 
     max (max (max x 3) 1) 4) = 4086 := by
  sorry

end NUMINAMATH_CALUDE_no_solution_for_digit_difference_l978_97824


namespace NUMINAMATH_CALUDE_no_valid_triples_l978_97870

theorem no_valid_triples : ¬∃ (a b c : ℤ), 
  (|a + b| + c = 23) ∧ 
  (a * b + |c| = 85) ∧ 
  (∃ k : ℤ, b = 3 * k) := by
sorry

end NUMINAMATH_CALUDE_no_valid_triples_l978_97870


namespace NUMINAMATH_CALUDE_garden_breadth_calculation_l978_97869

/-- Represents a rectangular garden --/
structure RectangularGarden where
  length : ℝ
  breadth : ℝ

/-- Calculates the perimeter of a rectangular garden --/
def perimeter (g : RectangularGarden) : ℝ :=
  2 * (g.length + g.breadth)

theorem garden_breadth_calculation :
  ∀ g : RectangularGarden,
    g.length = 205 →
    perimeter g = 600 →
    g.breadth = 95 := by
  sorry

end NUMINAMATH_CALUDE_garden_breadth_calculation_l978_97869


namespace NUMINAMATH_CALUDE_cake_and_icing_sum_l978_97831

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents the cake piece -/
structure CakePiece where
  top : List Point3D
  height : ℝ

/-- Calculates the volume of the cake piece -/
def cakeVolume (piece : CakePiece) : ℝ :=
  sorry

/-- Calculates the area of icing on the cake piece -/
def icingArea (piece : CakePiece) : ℝ :=
  sorry

/-- The main theorem -/
theorem cake_and_icing_sum (R P N : Point3D) (piece : CakePiece) :
  R.x = 0 ∧ R.y = 0 ∧ R.z = 3 ∧
  P.x = 3 ∧ P.y = 0 ∧ P.z = 3 ∧
  N.x = 2 ∧ N.y = 0 ∧ N.z = 3 ∧
  piece.top = [R, N, P] ∧
  piece.height = 3 →
  cakeVolume piece + icingArea piece = 13 := by
  sorry

end NUMINAMATH_CALUDE_cake_and_icing_sum_l978_97831


namespace NUMINAMATH_CALUDE_complement_of_P_l978_97845

def U : Set ℝ := Set.univ

def P : Set ℝ := {x : ℝ | x^2 - 5*x - 6 ≥ 0}

theorem complement_of_P (x : ℝ) : x ∈ Set.compl P ↔ x ∈ Set.Ioo (-1) 6 := by
  sorry

end NUMINAMATH_CALUDE_complement_of_P_l978_97845


namespace NUMINAMATH_CALUDE_equal_area_parallelograms_locus_l978_97854

/-- Given a triangle ABC and an interior point P, this theorem states that if the areas of
    parallelograms GPDC and FPEB (formed by lines parallel to the sides through P) are equal,
    then P lies on a specific line. -/
theorem equal_area_parallelograms_locus (a b c k l : ℝ) :
  let A : ℝ × ℝ := (0, a)
  let B : ℝ × ℝ := (-b, 0)
  let C : ℝ × ℝ := (c, 0)
  let P : ℝ × ℝ := (k, l)
  let E : ℝ × ℝ := (k - b*l/a, 0)
  let D : ℝ × ℝ := (k + l*c/a, 0)
  let F : ℝ × ℝ := (b*l/a - b, l)
  let G : ℝ × ℝ := (c - l*c/a, l)
  a > 0 ∧ b > 0 ∧ c > 0 ∧ k > -b ∧ k < c ∧ l > 0 ∧ l < a →
  abs (l/2 * (-c + 2*l*c/a)) = abs (l/2 * (-b + 2*l*b/a)) →
  2*a*k + (c - b)*l + a*(b - c) = 0 :=
sorry

end NUMINAMATH_CALUDE_equal_area_parallelograms_locus_l978_97854


namespace NUMINAMATH_CALUDE_shaded_area_square_with_quarter_circles_l978_97833

/-- The area of the shaded region inside a square with quarter circles at its corners -/
theorem shaded_area_square_with_quarter_circles 
  (square_side : ℝ) 
  (circle_radius : ℝ) 
  (h1 : square_side = 15) 
  (h2 : circle_radius = square_side / 3) :
  square_side ^ 2 - π * circle_radius ^ 2 = 225 - 25 * π := by
sorry

end NUMINAMATH_CALUDE_shaded_area_square_with_quarter_circles_l978_97833


namespace NUMINAMATH_CALUDE_loan_principal_calculation_l978_97895

/-- Calculates the total interest paid on a loan with varying interest rates over different periods. -/
def total_interest (principal : ℝ) : ℝ :=
  principal * (0.08 * 4 + 0.10 * 6 + 0.12 * 5)

/-- Proves that for the given interest rates and periods, a principal of 8000 results in a total interest of 12160. -/
theorem loan_principal_calculation :
  ∃ (principal : ℝ), principal > 0 ∧ total_interest principal = 12160 ∧ principal = 8000 :=
by
  sorry

#eval total_interest 8000

end NUMINAMATH_CALUDE_loan_principal_calculation_l978_97895


namespace NUMINAMATH_CALUDE_range_of_a_l978_97898

theorem range_of_a (a : ℝ) : 
  (∃ x : ℝ, |x + 1| + |x - a| < 4) → a ∈ Set.Ioo (-5) 3 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l978_97898


namespace NUMINAMATH_CALUDE_sum_of_repeating_decimals_l978_97840

/-- Represents a repeating decimal with a single repeating digit -/
def single_repeating_decimal (n : ℕ) : ℚ :=
  n / 9

/-- Represents a repeating decimal with two repeating digits -/
def double_repeating_decimal (n : ℕ) : ℚ :=
  n / 99

/-- The sum of 0.3̄ and 0.0̄2̄ equals 35/99 -/
theorem sum_of_repeating_decimals :
  single_repeating_decimal 3 + double_repeating_decimal 2 = 35 / 99 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_repeating_decimals_l978_97840


namespace NUMINAMATH_CALUDE_radius_of_circle_M_l978_97882

/-- Definition of Circle M -/
def CircleM (x y : ℝ) : Prop :=
  x^2 + y^2 - 8*x + 6*y = 0

/-- Theorem: The radius of Circle M is 5 -/
theorem radius_of_circle_M : ∃ (h k r : ℝ), r = 5 ∧ 
  ∀ (x y : ℝ), CircleM x y ↔ (x - h)^2 + (y - k)^2 = r^2 :=
sorry

end NUMINAMATH_CALUDE_radius_of_circle_M_l978_97882


namespace NUMINAMATH_CALUDE_ruble_combinations_l978_97842

theorem ruble_combinations : 
  ∃! n : ℕ, n = (Finset.filter 
    (fun p : ℕ × ℕ => 5 * p.1 + 3 * p.2 = 78) 
    (Finset.product (Finset.range 79) (Finset.range 79))).card ∧ n = 5 := by
  sorry

end NUMINAMATH_CALUDE_ruble_combinations_l978_97842


namespace NUMINAMATH_CALUDE_probability_of_desired_event_l978_97873

/-- Represents the outcome of a coin flip -/
inductive CoinFlip
| Heads
| Tails

/-- Represents the set of coins being flipped -/
structure CoinSet :=
(penny : CoinFlip)
(nickel : CoinFlip)
(dime : CoinFlip)
(quarter : CoinFlip)
(fifty_cent : CoinFlip)

/-- The total number of possible outcomes when flipping 5 coins -/
def total_outcomes : ℕ := 32

/-- Predicate for the desired event: at least penny, dime, and 50-cent coin are heads -/
def desired_event (cs : CoinSet) : Prop :=
  cs.penny = CoinFlip.Heads ∧ cs.dime = CoinFlip.Heads ∧ cs.fifty_cent = CoinFlip.Heads

/-- The number of outcomes satisfying the desired event -/
def successful_outcomes : ℕ := 4

/-- Theorem stating the probability of the desired event -/
theorem probability_of_desired_event :
  (successful_outcomes : ℚ) / total_outcomes = 1 / 8 := by sorry

end NUMINAMATH_CALUDE_probability_of_desired_event_l978_97873


namespace NUMINAMATH_CALUDE_balloon_distribution_l978_97803

theorem balloon_distribution (total_balloons : ℕ) (num_friends : ℕ) (balloons_returned : ℕ) : 
  total_balloons = 250 → 
  num_friends = 5 → 
  balloons_returned = 11 → 
  (total_balloons / num_friends) - balloons_returned = 39 := by
sorry

end NUMINAMATH_CALUDE_balloon_distribution_l978_97803


namespace NUMINAMATH_CALUDE_triangle_angle_inequality_l978_97817

theorem triangle_angle_inequality (a b c α β γ : Real) 
  (h_positive : a > 0 ∧ b > 0 ∧ c > 0 ∧ α > 0 ∧ β > 0 ∧ γ > 0)
  (h_triangle : α + β + γ = Real.pi)
  (h_sides : (a - b) * (α - β) ≥ 0 ∧ (b - c) * (β - γ) ≥ 0 ∧ (a - c) * (α - γ) ≥ 0) :
  Real.pi / 3 ≤ (a * α + b * β + c * γ) / (a + b + c) ∧ 
  (a * α + b * β + c * γ) / (a + b + c) < Real.pi / 2 := by
  sorry

end NUMINAMATH_CALUDE_triangle_angle_inequality_l978_97817


namespace NUMINAMATH_CALUDE_factorization_equality_l978_97875

theorem factorization_equality (a b m : ℝ) : 
  a^2 * (m - 1) + b^2 * (1 - m) = (m - 1) * (a + b) * (a - b) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l978_97875


namespace NUMINAMATH_CALUDE_fence_cost_square_plot_l978_97887

/-- The cost of building a fence around a square plot -/
theorem fence_cost_square_plot (area : ℝ) (price_per_foot : ℝ) (cost : ℝ) : 
  area = 289 → price_per_foot = 54 → cost = 4 * Real.sqrt area * price_per_foot → cost = 3672 := by
  sorry

end NUMINAMATH_CALUDE_fence_cost_square_plot_l978_97887


namespace NUMINAMATH_CALUDE_surjective_iff_coprime_l978_97855

/-- Euler's totient function -/
def phi : ℕ → ℕ := sorry

/-- The function f(x) = x^x mod n -/
def f (n : ℕ) (x : ℕ+) : ZMod n :=
  (x : ZMod n) ^ (x : ℕ)

/-- Surjectivity of f -/
def is_surjective (n : ℕ) : Prop :=
  Function.Surjective (f n)

theorem surjective_iff_coprime (n : ℕ) (h : n > 0) :
  is_surjective n ↔ Nat.Coprime n (phi n) := by sorry

end NUMINAMATH_CALUDE_surjective_iff_coprime_l978_97855


namespace NUMINAMATH_CALUDE_product_equality_l978_97829

theorem product_equality : 500 * 2468 * 0.2468 * 100 = 30485120 := by
  sorry

end NUMINAMATH_CALUDE_product_equality_l978_97829


namespace NUMINAMATH_CALUDE_inscribed_circle_radius_l978_97849

/-- A quadrilateral circumscribed around a circle -/
structure CircumscribedQuadrilateral where
  /-- The sum of two opposite sides -/
  opposite_sides_sum : ℝ
  /-- The area of the quadrilateral -/
  area : ℝ
  /-- The radius of the inscribed circle -/
  inradius : ℝ

/-- Theorem: If the sum of opposite sides is 10 and the area is 12, 
    then the radius of the inscribed circle is 6/5 -/
theorem inscribed_circle_radius 
  (q : CircumscribedQuadrilateral) 
  (h1 : q.opposite_sides_sum = 10) 
  (h2 : q.area = 12) : 
  q.inradius = 6/5 := by
  sorry

end NUMINAMATH_CALUDE_inscribed_circle_radius_l978_97849


namespace NUMINAMATH_CALUDE_y_value_l978_97816

theorem y_value (y : ℚ) (h : (1 : ℚ) / 3 - (1 : ℚ) / 4 = 4 / y) : y = 48 := by
  sorry

end NUMINAMATH_CALUDE_y_value_l978_97816


namespace NUMINAMATH_CALUDE_negation_of_proposition_l978_97844

theorem negation_of_proposition (p : Prop) :
  (¬(∀ x : ℝ, x > 0 → x > Real.log x)) ↔ (∃ x₀ : ℝ, x₀ > 0 ∧ x₀ ≤ Real.log x₀) := by
  sorry

end NUMINAMATH_CALUDE_negation_of_proposition_l978_97844


namespace NUMINAMATH_CALUDE_golden_state_team_points_l978_97862

/-- The number of points earned by Draymond in the Golden State Team -/
def draymondPoints : ℕ := 12

/-- The total points earned by the Golden State Team -/
def totalTeamPoints : ℕ := 69

/-- The number of points earned by Kelly -/
def kellyPoints : ℕ := 9

theorem golden_state_team_points :
  ∃ (D : ℕ), 
    D = draymondPoints ∧
    D + 2*D + kellyPoints + 2*kellyPoints + D/2 = totalTeamPoints :=
by sorry

end NUMINAMATH_CALUDE_golden_state_team_points_l978_97862


namespace NUMINAMATH_CALUDE_existence_of_m_l978_97897

theorem existence_of_m : ∃ m : ℝ, m ≤ 3 ∧ ∀ x : ℝ, |x - 1| ≤ m → -2 ≤ x ∧ x ≤ 10 := by
  sorry

end NUMINAMATH_CALUDE_existence_of_m_l978_97897
