import Mathlib

namespace NUMINAMATH_CALUDE_smallest_k_no_real_roots_l2448_244848

theorem smallest_k_no_real_roots :
  ∃ (k : ℤ),
    (∀ (j : ℤ), j < k → ∃ (x : ℝ), 3 * x * (j * x - 5) - 2 * x^2 + 9 = 0) ∧
    (∀ (x : ℝ), 3 * x * (k * x - 5) - 2 * x^2 + 9 ≠ 0) ∧
    k = 3 := by
  sorry

end NUMINAMATH_CALUDE_smallest_k_no_real_roots_l2448_244848


namespace NUMINAMATH_CALUDE_unique_prime_with_prime_successors_l2448_244883

theorem unique_prime_with_prime_successors :
  ∀ p : ℕ, Prime p ∧ Prime (p + 4) ∧ Prime (p + 8) → p = 3 := by
  sorry

end NUMINAMATH_CALUDE_unique_prime_with_prime_successors_l2448_244883


namespace NUMINAMATH_CALUDE_regular_polygon_properties_l2448_244813

theorem regular_polygon_properties (exterior_angle : ℝ) 
  (h1 : exterior_angle = 15)
  (h2 : exterior_angle > 0) :
  ∃ (n : ℕ) (sum_interior : ℝ),
    n = 24 ∧ 
    sum_interior = 3960 ∧
    n * exterior_angle = 360 ∧
    sum_interior = 180 * (n - 2) :=
by sorry

end NUMINAMATH_CALUDE_regular_polygon_properties_l2448_244813


namespace NUMINAMATH_CALUDE_cuboid_length_is_40_l2448_244893

/-- The surface area of a cuboid given its length, breadth, and height -/
def cuboidSurfaceArea (l b h : ℝ) : ℝ := 2 * (l * b + b * h + h * l)

/-- Theorem: The length of a cuboid with surface area 2400, breadth 10, and height 16 is 40 -/
theorem cuboid_length_is_40 :
  ∃ l : ℝ, cuboidSurfaceArea l 10 16 = 2400 ∧ l = 40 :=
by sorry

end NUMINAMATH_CALUDE_cuboid_length_is_40_l2448_244893


namespace NUMINAMATH_CALUDE_power_of_point_l2448_244821

-- Define a circle in a plane
structure Circle where
  center : ℝ × ℝ
  radius : ℝ

-- Define a point in a plane
def Point := ℝ × ℝ

-- Define a line passing through two points
structure Line where
  point1 : Point
  point2 : Point

-- Define the distance between two points
def distance (p1 p2 : Point) : ℝ := sorry

-- Define the intersection of a line and a circle
def intersect (l : Line) (c : Circle) : Option (Point × Point) := sorry

-- Theorem statement
theorem power_of_point (S : Circle) (P A B A1 B1 : Point) 
  (l1 l2 : Line) : 
  l1.point1 = P → l2.point1 = P → 
  intersect l1 S = some (A, B) → 
  intersect l2 S = some (A1, B1) → 
  distance P A * distance P B = distance P A1 * distance P B1 := by
  sorry

end NUMINAMATH_CALUDE_power_of_point_l2448_244821


namespace NUMINAMATH_CALUDE_danny_steve_time_ratio_l2448_244879

/-- The time it takes Danny to reach Steve's house -/
def danny_time : ℝ := 29

/-- The time it takes Steve to reach Danny's house -/
def steve_time : ℝ := 58

/-- The difference in time it takes Steve and Danny to reach the halfway point -/
def halfway_time_difference : ℝ := 14.5

theorem danny_steve_time_ratio :
  danny_time / steve_time = 1 / 2 ∧
  steve_time / 2 = danny_time / 2 + halfway_time_difference :=
by sorry

end NUMINAMATH_CALUDE_danny_steve_time_ratio_l2448_244879


namespace NUMINAMATH_CALUDE_valid_triples_eq_solution_set_l2448_244896

def is_valid_triple (a m n : ℕ) : Prop :=
  a ≥ 2 ∧ m ≥ 2 ∧ (a^n + 203) % (a^(m*n) + 1) = 0

def solution_set : Set (ℕ × ℕ × ℕ) :=
  {(a, m, n) | 
    (∃ k, (a = 2 ∧ m = 2 ∧ n = 4*k + 1) ∨
          (a = 2 ∧ m = 3 ∧ n = 6*k + 2) ∨
          (a = 2 ∧ m = 4 ∧ n = 8*k + 8) ∨
          (a = 2 ∧ m = 6 ∧ n = 12*k + 9) ∨
          (a = 3 ∧ m = 2 ∧ n = 4*k + 3) ∨
          (a = 4 ∧ m = 2 ∧ n = 4*k + 4) ∨
          (a = 5 ∧ m = 2 ∧ n = 4*k + 1) ∨
          (a = 8 ∧ m = 2 ∧ n = 4*k + 3) ∨
          (a = 10 ∧ m = 2 ∧ n = 4*k + 2)) ∨
    (a = 203 ∧ m ≥ 2 ∧ ∃ k, n = (2*k + 1)*m + 1)}

theorem valid_triples_eq_solution_set :
  {(a, m, n) : ℕ × ℕ × ℕ | is_valid_triple a m n} = solution_set :=
sorry

end NUMINAMATH_CALUDE_valid_triples_eq_solution_set_l2448_244896


namespace NUMINAMATH_CALUDE_shelter_puppies_count_l2448_244881

theorem shelter_puppies_count :
  ∀ (puppies kittens : ℕ),
    kittens = 2 * puppies + 14 →
    puppies > 0 →
    kittens = 78 →
    puppies = 32 := by
  sorry

end NUMINAMATH_CALUDE_shelter_puppies_count_l2448_244881


namespace NUMINAMATH_CALUDE_tip_percentage_calculation_l2448_244809

theorem tip_percentage_calculation (meal_cost drink_cost payment change : ℚ) : 
  meal_cost = 10 →
  drink_cost = 5/2 →
  payment = 20 →
  change = 5 →
  ((payment - change) - (meal_cost + drink_cost)) / (meal_cost + drink_cost) * 100 = 20 := by
  sorry

end NUMINAMATH_CALUDE_tip_percentage_calculation_l2448_244809


namespace NUMINAMATH_CALUDE_fencing_cost_is_225_rupees_l2448_244802

/-- Represents a rectangular park -/
structure RectangularPark where
  length : ℝ
  width : ℝ
  area : ℝ
  fencing_cost_per_meter : ℝ

/-- Calculate the total fencing cost for a rectangular park -/
def calculate_fencing_cost (park : RectangularPark) : ℝ :=
  2 * (park.length + park.width) * park.fencing_cost_per_meter

/-- Theorem: The fencing cost for the given rectangular park is 225 rupees -/
theorem fencing_cost_is_225_rupees :
  ∀ (park : RectangularPark),
    park.length / park.width = 3 / 2 →
    park.area = 3750 →
    park.fencing_cost_per_meter = 0.9 →
    calculate_fencing_cost park = 225 := by
  sorry


end NUMINAMATH_CALUDE_fencing_cost_is_225_rupees_l2448_244802


namespace NUMINAMATH_CALUDE_ratio_of_first_to_third_term_l2448_244880

/-- An arithmetic sequence with first four terms a, y, b, 3y -/
def ArithmeticSequence (a y b : ℝ) : Prop :=
  ∃ d : ℝ, y - a = d ∧ b - y = d ∧ 3*y - b = d

theorem ratio_of_first_to_third_term (a y b : ℝ) 
  (h : ArithmeticSequence a y b) : a / b = 0 := by
  sorry

end NUMINAMATH_CALUDE_ratio_of_first_to_third_term_l2448_244880


namespace NUMINAMATH_CALUDE_rent_expense_calculation_l2448_244868

def monthly_salary : ℕ := 23500
def savings_percentage : ℚ := 1/10
def savings : ℕ := 2350
def milk_expense : ℕ := 1500
def groceries_expense : ℕ := 4500
def education_expense : ℕ := 2500
def petrol_expense : ℕ := 2000
def miscellaneous_expense : ℕ := 5650

theorem rent_expense_calculation :
  let total_expenses := milk_expense + groceries_expense + education_expense + petrol_expense + miscellaneous_expense
  let rent := monthly_salary - savings - total_expenses
  rent = 4850 := by sorry

end NUMINAMATH_CALUDE_rent_expense_calculation_l2448_244868


namespace NUMINAMATH_CALUDE_bags_sold_is_30_l2448_244856

-- Define the variables
def cost_price : ℕ := 4
def selling_price : ℕ := 8
def total_profit : ℕ := 120

-- Define the profit per bag
def profit_per_bag : ℕ := selling_price - cost_price

-- Theorem to prove
theorem bags_sold_is_30 : total_profit / profit_per_bag = 30 := by
  sorry

end NUMINAMATH_CALUDE_bags_sold_is_30_l2448_244856


namespace NUMINAMATH_CALUDE_paper_folding_thickness_l2448_244816

/-- The thickness of a folded paper after a given number of folds. -/
def thickness (initial_thickness : ℝ) (num_folds : ℕ) : ℝ :=
  initial_thickness * (2 ^ num_folds)

/-- Theorem stating that folding a 0.1 mm thick paper 5 times results in 3.2 mm thickness. -/
theorem paper_folding_thickness :
  thickness 0.1 5 = 3.2 := by
  sorry

end NUMINAMATH_CALUDE_paper_folding_thickness_l2448_244816


namespace NUMINAMATH_CALUDE_nine_students_left_l2448_244841

/-- The number of students left after some were checked out early -/
def students_left (initial : ℕ) (checked_out : ℕ) : ℕ :=
  initial - checked_out

/-- Theorem: Given 16 initial students and 7 checked out early, 9 students are left -/
theorem nine_students_left :
  students_left 16 7 = 9 := by
  sorry

end NUMINAMATH_CALUDE_nine_students_left_l2448_244841


namespace NUMINAMATH_CALUDE_toy_store_problem_l2448_244864

/-- Toy store problem -/
theorem toy_store_problem 
  (purchase_price : ℝ) 
  (base_price : ℝ) 
  (base_sales : ℝ) 
  (price_increment : ℝ) 
  (sales_decrement : ℝ) 
  (max_price : ℝ) 
  (max_cost : ℝ) 
  (profit : ℝ) :
  purchase_price = 49 →
  base_price = 50 →
  base_sales = 50 →
  price_increment = 0.5 →
  sales_decrement = 3 →
  max_price = 60 →
  max_cost = 686 →
  profit = 147 →
  ∃ (x : ℝ) (a : ℝ),
    -- Part 1: Price range
    56 ≤ x ∧ x ≤ 60 ∧
    x ≤ max_price ∧
    purchase_price * (base_sales - sales_decrement * ((x - base_price) / price_increment)) ≤ max_cost ∧
    -- Part 2: Value of a
    a = 25 ∧
    (x * (1 + a / 100) - purchase_price) * (base_sales - sales_decrement * ((x - base_price) / price_increment)) * (1 - 2 * a / 100) = profit :=
by sorry

end NUMINAMATH_CALUDE_toy_store_problem_l2448_244864


namespace NUMINAMATH_CALUDE_least_prime_factor_of_5_5_minus_5_3_l2448_244832

theorem least_prime_factor_of_5_5_minus_5_3 :
  Nat.minFac (5^5 - 5^3) = 2 := by
  sorry

end NUMINAMATH_CALUDE_least_prime_factor_of_5_5_minus_5_3_l2448_244832


namespace NUMINAMATH_CALUDE_popcorn_tablespoons_needed_l2448_244851

/-- The number of cups of popcorn produced by 2 tablespoons of kernels -/
def cups_per_two_tablespoons : ℕ := 4

/-- The number of cups of popcorn Joanie wants -/
def joanie_cups : ℕ := 3

/-- The number of cups of popcorn Mitchell wants -/
def mitchell_cups : ℕ := 4

/-- The number of cups of popcorn Miles and Davis will split -/
def miles_davis_cups : ℕ := 6

/-- The number of cups of popcorn Cliff wants -/
def cliff_cups : ℕ := 3

/-- The total number of cups of popcorn wanted -/
def total_cups : ℕ := joanie_cups + mitchell_cups + miles_davis_cups + cliff_cups

/-- Theorem stating the number of tablespoons of popcorn kernels needed -/
theorem popcorn_tablespoons_needed : 
  (total_cups / cups_per_two_tablespoons) * 2 = 8 := by
  sorry

end NUMINAMATH_CALUDE_popcorn_tablespoons_needed_l2448_244851


namespace NUMINAMATH_CALUDE_mabel_transactions_l2448_244829

theorem mabel_transactions :
  ∀ (mabel anthony cal jade : ℕ),
    anthony = mabel + mabel / 10 →
    cal = (2 * anthony) / 3 →
    jade = cal + 17 →
    jade = 83 →
    mabel = 90 :=
by
  sorry

end NUMINAMATH_CALUDE_mabel_transactions_l2448_244829


namespace NUMINAMATH_CALUDE_largest_difference_of_three_digit_numbers_l2448_244857

/-- A function that represents a 3-digit number given its digits -/
def threeDigitNumber (a b c : Nat) : Nat := 100 * a + 10 * b + c

/-- The set of valid digits -/
def validDigits : Finset Nat := Finset.range 9

theorem largest_difference_of_three_digit_numbers :
  ∃ (U V W X Y Z : Nat),
    U ∈ validDigits ∧ V ∈ validDigits ∧ W ∈ validDigits ∧
    X ∈ validDigits ∧ Y ∈ validDigits ∧ Z ∈ validDigits ∧
    U ≠ V ∧ U ≠ W ∧ U ≠ X ∧ U ≠ Y ∧ U ≠ Z ∧
    V ≠ W ∧ V ≠ X ∧ V ≠ Y ∧ V ≠ Z ∧
    W ≠ X ∧ W ≠ Y ∧ W ≠ Z ∧
    X ≠ Y ∧ X ≠ Z ∧
    Y ≠ Z ∧
    threeDigitNumber U V W - threeDigitNumber X Y Z = 864 ∧
    ∀ (A B C D E F : Nat),
      A ∈ validDigits → B ∈ validDigits → C ∈ validDigits →
      D ∈ validDigits → E ∈ validDigits → F ∈ validDigits →
      A ≠ B ∧ A ≠ C ∧ A ≠ D ∧ A ≠ E ∧ A ≠ F ∧
      B ≠ C ∧ B ≠ D ∧ B ≠ E ∧ B ≠ F ∧
      C ≠ D ∧ C ≠ E ∧ C ≠ F ∧
      D ≠ E ∧ D ≠ F ∧
      E ≠ F →
      threeDigitNumber A B C - threeDigitNumber D E F ≤ 864 :=
by
  sorry


end NUMINAMATH_CALUDE_largest_difference_of_three_digit_numbers_l2448_244857


namespace NUMINAMATH_CALUDE_remainder_2023_div_73_l2448_244817

theorem remainder_2023_div_73 : 2023 % 73 = 52 := by
  sorry

end NUMINAMATH_CALUDE_remainder_2023_div_73_l2448_244817


namespace NUMINAMATH_CALUDE_cafe_chair_distribution_l2448_244819

/-- Given a cafe with indoor and outdoor tables, prove the number of chairs at each indoor table. -/
theorem cafe_chair_distribution (indoor_tables outdoor_tables : ℕ) 
  (chairs_per_outdoor_table : ℕ) (total_chairs : ℕ) :
  indoor_tables = 9 →
  outdoor_tables = 11 →
  chairs_per_outdoor_table = 3 →
  total_chairs = 123 →
  ∃ (chairs_per_indoor_table : ℕ),
    chairs_per_indoor_table = 10 ∧
    total_chairs = indoor_tables * chairs_per_indoor_table + outdoor_tables * chairs_per_outdoor_table :=
by
  sorry

end NUMINAMATH_CALUDE_cafe_chair_distribution_l2448_244819


namespace NUMINAMATH_CALUDE_largest_x_value_largest_x_exists_l2448_244806

theorem largest_x_value (x y : ℝ) : 
  (|x - 3| = 15 ∧ x + y = 10) → x ≤ 18 := by
  sorry

theorem largest_x_exists : 
  ∃ x y : ℝ, |x - 3| = 15 ∧ x + y = 10 ∧ x = 18 := by
  sorry

end NUMINAMATH_CALUDE_largest_x_value_largest_x_exists_l2448_244806


namespace NUMINAMATH_CALUDE_smallest_n_for_Q_less_than_threshold_l2448_244827

def Q (n : ℕ) : ℚ := 2 / (n * (n + 1) * (n + 2))

theorem smallest_n_for_Q_less_than_threshold : 
  ∀ k : ℕ, k > 0 → k < 19 → Q (5 * k) ≥ 1 / 2500 ∧ Q (5 * 19) < 1 / 2500 :=
sorry

end NUMINAMATH_CALUDE_smallest_n_for_Q_less_than_threshold_l2448_244827


namespace NUMINAMATH_CALUDE_equal_hot_dogs_and_buns_l2448_244853

/-- The number of hot dogs in each package -/
def hot_dogs_per_package : ℕ := 7

/-- The number of buns in each package -/
def buns_per_package : ℕ := 9

/-- The smallest number of hot dog packages needed to have an equal number of hot dogs and buns -/
def smallest_number_of_packages : ℕ := 9

theorem equal_hot_dogs_and_buns :
  smallest_number_of_packages * hot_dogs_per_package =
  (smallest_number_of_packages * hot_dogs_per_package / buns_per_package) * buns_per_package ∧
  ∀ n : ℕ, n < smallest_number_of_packages →
    n * hot_dogs_per_package ≠
    (n * hot_dogs_per_package / buns_per_package) * buns_per_package :=
by sorry

end NUMINAMATH_CALUDE_equal_hot_dogs_and_buns_l2448_244853


namespace NUMINAMATH_CALUDE_largest_digit_divisible_by_six_l2448_244831

theorem largest_digit_divisible_by_six :
  ∃ (M : ℕ), M < 10 ∧ 
  (∀ (n : ℕ), n < 10 → 6 ∣ (3190 * 10 + n) → n ≤ M) ∧
  (6 ∣ (3190 * 10 + M)) :=
by sorry

end NUMINAMATH_CALUDE_largest_digit_divisible_by_six_l2448_244831


namespace NUMINAMATH_CALUDE_cubic_monotonically_increasing_iff_l2448_244861

/-- A cubic function f(x) = ax³ + bx² + cx + d -/
def cubic_function (a b c d : ℝ) : ℝ → ℝ := λ x => a * x^3 + b * x^2 + c * x + d

/-- A function is monotonically increasing -/
def monotonically_increasing (f : ℝ → ℝ) : Prop :=
  ∀ x y, x < y → f x < f y

/-- Theorem: For a cubic function f(x) = ax³ + bx² + cx + d with a > 0,
    f(x) is monotonically increasing on ℝ if and only if b² - 3ac ≤ 0 -/
theorem cubic_monotonically_increasing_iff {a b c d : ℝ} (ha : a > 0) :
  monotonically_increasing (cubic_function a b c d) ↔ b^2 - 3*a*c ≤ 0 :=
sorry

end NUMINAMATH_CALUDE_cubic_monotonically_increasing_iff_l2448_244861


namespace NUMINAMATH_CALUDE_three_in_all_curriculums_l2448_244872

/-- Represents the number of people in each curriculum or combination of curriculums -/
structure CurriculumParticipants where
  yoga : ℕ
  cooking : ℕ
  weaving : ℕ
  cookingOnly : ℕ
  cookingAndYoga : ℕ
  cookingAndWeaving : ℕ
  allCurriculums : ℕ

/-- Theorem stating that given the conditions, 3 people participate in all curriculums -/
theorem three_in_all_curriculums (p : CurriculumParticipants) 
  (h1 : p.yoga = 25)
  (h2 : p.cooking = 15)
  (h3 : p.weaving = 8)
  (h4 : p.cookingOnly = 2)
  (h5 : p.cookingAndYoga = 7)
  (h6 : p.cookingAndWeaving = 3)
  (h7 : p.cooking = p.cookingOnly + p.cookingAndYoga + p.cookingAndWeaving + p.allCurriculums) :
  p.allCurriculums = 3 := by
  sorry


end NUMINAMATH_CALUDE_three_in_all_curriculums_l2448_244872


namespace NUMINAMATH_CALUDE_max_sum_with_constraint_l2448_244894

theorem max_sum_with_constraint (x y : ℝ) (h1 : 0 < x) (h2 : 0 < y) (h3 : x^2 + y = 20) :
  x + y ≤ 81/4 :=
sorry

end NUMINAMATH_CALUDE_max_sum_with_constraint_l2448_244894


namespace NUMINAMATH_CALUDE_tan_half_product_l2448_244837

theorem tan_half_product (a b : ℝ) :
  7 * (Real.cos a + Real.cos b) + 2 * (Real.cos a * Real.cos b + 1) + 3 * Real.sin a * Real.sin b = 0 →
  Real.tan (a / 2) * Real.tan (b / 2) = -2 ∨ Real.tan (a / 2) * Real.tan (b / 2) = 2 := by
  sorry

end NUMINAMATH_CALUDE_tan_half_product_l2448_244837


namespace NUMINAMATH_CALUDE_cos_equality_problem_l2448_244849

theorem cos_equality_problem (n : ℤ) : 
  0 ≤ n ∧ n ≤ 180 ∧ Real.cos (n * π / 180) = Real.cos (317 * π / 180) → n = 43 := by
  sorry

end NUMINAMATH_CALUDE_cos_equality_problem_l2448_244849


namespace NUMINAMATH_CALUDE_race_inequality_l2448_244808

theorem race_inequality (x : ℝ) : 
  (∀ (race_length : ℝ) (initial_speed : ℝ) (ming_speed : ℝ) (li_speed : ℝ) (distance_ahead : ℝ),
    race_length = 10000 ∧ 
    initial_speed = 200 ∧ 
    ming_speed = 250 ∧ 
    li_speed = 300 ∧ 
    distance_ahead = 200 ∧ 
    x > 0 ∧ 
    x < 50 ∧  -- This ensures Xiao Ming doesn't finish before encountering Xiao Li
    (race_length - initial_speed * x - distance_ahead) / ming_speed < 
      (race_length - initial_speed * x) / li_speed) →
  (10000 - 200 * x - 200) / 250 > (10000 - 200 * x) / 300 :=
by sorry

end NUMINAMATH_CALUDE_race_inequality_l2448_244808


namespace NUMINAMATH_CALUDE_total_tulips_l2448_244844

theorem total_tulips (arwen_tulips : ℕ) (elrond_tulips : ℕ) : 
  arwen_tulips = 20 → 
  elrond_tulips = 2 * arwen_tulips → 
  arwen_tulips + elrond_tulips = 60 := by
  sorry

end NUMINAMATH_CALUDE_total_tulips_l2448_244844


namespace NUMINAMATH_CALUDE_xiao_li_commute_l2448_244899

/-- Xiao Li's commute problem -/
theorem xiao_li_commute 
  (distance : ℝ) 
  (walk_late : ℝ) 
  (bike_early : ℝ) 
  (bike_speed_factor : ℝ) 
  (breakdown_distance : ℝ) 
  (early_arrival : ℝ)
  (h1 : distance = 4.5)
  (h2 : walk_late = 5 / 60)
  (h3 : bike_early = 10 / 60)
  (h4 : bike_speed_factor = 1.5)
  (h5 : breakdown_distance = 1.5)
  (h6 : early_arrival = 5 / 60) :
  ∃ (walk_speed bike_speed min_run_speed : ℝ),
    walk_speed = 6 ∧ 
    bike_speed = 9 ∧ 
    min_run_speed = 7.2 ∧
    distance / walk_speed - walk_late = distance / bike_speed + bike_early ∧
    bike_speed = bike_speed_factor * walk_speed ∧
    breakdown_distance + (distance / bike_speed + bike_early - breakdown_distance / bike_speed - early_arrival) * min_run_speed ≥ distance :=
by
  sorry

end NUMINAMATH_CALUDE_xiao_li_commute_l2448_244899


namespace NUMINAMATH_CALUDE_additive_implies_zero_and_odd_l2448_244820

/-- A function satisfying f(x+y) = f(x) + f(y) for all real x and y is zero at 0 and odd -/
theorem additive_implies_zero_and_odd (f : ℝ → ℝ) 
  (h : ∀ x y : ℝ, f (x + y) = f x + f y) : 
  (f 0 = 0) ∧ (∀ x : ℝ, f (-x) = -f x) := by
  sorry

end NUMINAMATH_CALUDE_additive_implies_zero_and_odd_l2448_244820


namespace NUMINAMATH_CALUDE_flight_750_male_first_class_fraction_l2448_244859

theorem flight_750_male_first_class_fraction 
  (total_passengers : ℕ) 
  (female_percentage : ℚ) 
  (first_class_percentage : ℚ) 
  (female_coach : ℕ) :
  total_passengers = 120 →
  female_percentage = 45/100 →
  first_class_percentage = 10/100 →
  female_coach = 46 →
  (total_passengers * first_class_percentage * (1 - female_percentage / (1 - first_class_percentage)) / 
   (total_passengers * first_class_percentage) : ℚ) = 1/3 := by
sorry

end NUMINAMATH_CALUDE_flight_750_male_first_class_fraction_l2448_244859


namespace NUMINAMATH_CALUDE_complete_square_factorization_quadratic_factorization_l2448_244845

/-- A quadratic expression ax^2 + bx + c can be factored using the complete square formula
    if and only if b = ±2√(ac) -/
theorem complete_square_factorization (a b c : ℝ) :
  (∃ (k : ℝ), b = 2 * k * Real.sqrt (a * c)) ∨ (∃ (k : ℝ), b = -2 * k * Real.sqrt (a * c)) ↔
  ∃ (p q : ℝ), a * x^2 + b * x + c = a * (x - p)^2 + q := sorry

/-- For the quadratic expression 4x^2 - (m+1)x + 9 to be factored using the complete square formula,
    m must equal 11 or -13 -/
theorem quadratic_factorization (m : ℝ) :
  (∃ (p q : ℝ), 4 * x^2 - (m + 1) * x + 9 = 4 * (x - p)^2 + q) ↔ (m = 11 ∨ m = -13) := by
  sorry

end NUMINAMATH_CALUDE_complete_square_factorization_quadratic_factorization_l2448_244845


namespace NUMINAMATH_CALUDE_arithmetic_sequence_squared_l2448_244887

theorem arithmetic_sequence_squared (x y z : ℝ) (h : y - x = z - y) :
  (x^2 + x*z + z^2) - (x^2 + x*y + y^2) = (y^2 + y*z + z^2) - (x^2 + x*z + z^2) :=
by sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_squared_l2448_244887


namespace NUMINAMATH_CALUDE_goldfish_price_theorem_l2448_244870

/-- Represents the selling price of a goldfish -/
def selling_price : ℝ := sorry

/-- Represents the cost price of a goldfish -/
def cost_price : ℝ := 0.25

/-- Represents the price of the new tank -/
def tank_price : ℝ := 100

/-- Represents the number of goldfish sold -/
def goldfish_sold : ℕ := 110

/-- Represents the percentage short of the tank price -/
def percentage_short : ℝ := 0.45

theorem goldfish_price_theorem :
  selling_price = 0.75 :=
by
  sorry

end NUMINAMATH_CALUDE_goldfish_price_theorem_l2448_244870


namespace NUMINAMATH_CALUDE_last_two_digits_product_l2448_244888

theorem last_two_digits_product (n : ℤ) : ∃ k : ℤ, 122 * 123 * 125 * 127 * n ≡ 50 [ZMOD 100] := by
  sorry

end NUMINAMATH_CALUDE_last_two_digits_product_l2448_244888


namespace NUMINAMATH_CALUDE_equation_solution_l2448_244803

theorem equation_solution : ∀ x : ℝ, (9 / x^2 = x / 25) → x = 5 := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l2448_244803


namespace NUMINAMATH_CALUDE_billys_book_pages_l2448_244882

/-- Proves that given Billy's reading habits and time allocation, each book he reads contains 80 pages. -/
theorem billys_book_pages : 
  -- Billy's free time per day
  (free_time_per_day : ℕ) →
  -- Number of weekend days
  (weekend_days : ℕ) →
  -- Percentage of time spent on video games
  (video_game_percentage : ℚ) →
  -- Pages Billy can read per hour
  (pages_per_hour : ℕ) →
  -- Number of books Billy reads
  (number_of_books : ℕ) →
  -- Conditions
  (free_time_per_day = 8) →
  (weekend_days = 2) →
  (video_game_percentage = 3/4) →
  (pages_per_hour = 60) →
  (number_of_books = 3) →
  -- Conclusion: each book contains 80 pages
  (∃ (pages_per_book : ℕ), pages_per_book = 80 ∧ 
    pages_per_book * number_of_books = 
      (1 - video_game_percentage) * (free_time_per_day * weekend_days : ℚ) * pages_per_hour) :=
by
  sorry


end NUMINAMATH_CALUDE_billys_book_pages_l2448_244882


namespace NUMINAMATH_CALUDE_gcd_459_357_l2448_244866

theorem gcd_459_357 : Nat.gcd 459 357 = 51 := by
  sorry

end NUMINAMATH_CALUDE_gcd_459_357_l2448_244866


namespace NUMINAMATH_CALUDE_altitude_B_correct_median_A_correct_circumcircle_correct_l2448_244892

-- Define the triangle ABC
def A : ℝ × ℝ := (0, 1)
def B : ℝ × ℝ := (0, -1)
def C : ℝ × ℝ := (-2, 1)

-- Define the altitude from B to BC
def altitude_B (x y : ℝ) : Prop := x - y + 1 = 0

-- Define the median from A to AC
def median_A (x : ℝ) : Prop := x = -1

-- Define the circumcircle of triangle ABC
def circumcircle (x y : ℝ) : Prop := x^2 + y^2 + 2*x - 1 = 0

-- Theorem statements
theorem altitude_B_correct :
  ∀ x y : ℝ, altitude_B x y ↔ (x - y + 1 = 0) :=
sorry

theorem median_A_correct :
  ∀ x : ℝ, median_A x ↔ (x = -1) :=
sorry

theorem circumcircle_correct :
  ∀ x y : ℝ, circumcircle x y ↔ (x^2 + y^2 + 2*x - 1 = 0) :=
sorry

end NUMINAMATH_CALUDE_altitude_B_correct_median_A_correct_circumcircle_correct_l2448_244892


namespace NUMINAMATH_CALUDE_rectangle_area_l2448_244823

/-- Given a rectangle with width 6 cm and perimeter 28 cm, prove its area is 48 square cm. -/
theorem rectangle_area (width : ℝ) (perimeter : ℝ) (length : ℝ) (area : ℝ) : 
  width = 6 →
  perimeter = 28 →
  perimeter = 2 * (length + width) →
  area = length * width →
  area = 48 := by
sorry

end NUMINAMATH_CALUDE_rectangle_area_l2448_244823


namespace NUMINAMATH_CALUDE_sum_of_first_n_naturals_l2448_244891

theorem sum_of_first_n_naturals (n : ℕ) : 
  (List.range (n + 1)).sum = n * (n + 1) / 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_first_n_naturals_l2448_244891


namespace NUMINAMATH_CALUDE_lemonade_water_amount_l2448_244865

/- Define the ratios and amounts -/
def water_sugar_ratio : ℚ := 3
def sugar_lemon_ratio : ℚ := 3
def lemon_juice_amount : ℚ := 4

/- Define the function to calculate water amount -/
def water_amount (water_sugar : ℚ) (sugar_lemon : ℚ) (lemon : ℚ) : ℚ :=
  water_sugar * sugar_lemon * lemon

/- Theorem statement -/
theorem lemonade_water_amount :
  water_amount water_sugar_ratio sugar_lemon_ratio lemon_juice_amount = 36 := by
  sorry


end NUMINAMATH_CALUDE_lemonade_water_amount_l2448_244865


namespace NUMINAMATH_CALUDE_range_of_m_l2448_244871

theorem range_of_m (m : ℝ) : 
  (∀ x : ℝ, x > 1 → 2*x + m + 2/(x-1) > 0) → m > -6 := by
  sorry

end NUMINAMATH_CALUDE_range_of_m_l2448_244871


namespace NUMINAMATH_CALUDE_reflection_across_x_axis_l2448_244830

/-- Reflects a point across the x-axis in a 2D Cartesian coordinate system -/
def reflect_x (p : ℝ × ℝ) : ℝ × ℝ := (p.1, -p.2)

/-- The original point -/
def original_point : ℝ × ℝ := (2, -1)

/-- The reflected point -/
def reflected_point : ℝ × ℝ := (2, 1)

theorem reflection_across_x_axis :
  reflect_x original_point = reflected_point := by sorry

end NUMINAMATH_CALUDE_reflection_across_x_axis_l2448_244830


namespace NUMINAMATH_CALUDE_f_properties_l2448_244843

open Real

noncomputable def f (x : ℝ) : ℝ := 2 / x + log x

theorem f_properties :
  (∃ δ > 0, ∀ x, 0 < |x - 2| ∧ |x - 2| < δ → f x ≥ f 2) ∧
  (∀ x₁ x₂, 0 < x₁ ∧ 0 < x₂ ∧ x₁ < x₂ ∧ f x₁ = f x₂ → x₁ + x₂ > 4) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l2448_244843


namespace NUMINAMATH_CALUDE_range_of_a_l2448_244839

theorem range_of_a (a : ℝ) : (∀ x : ℝ, |2*x + 2| - |2*x - 2| ≤ a) → a ≥ 4 := by
  sorry

end NUMINAMATH_CALUDE_range_of_a_l2448_244839


namespace NUMINAMATH_CALUDE_triangle_side_values_l2448_244854

theorem triangle_side_values (n : ℕ) : 
  (3 * n - 3 > 0) ∧ 
  (2 * n + 12 > 0) ∧ 
  (2 * n + 7 > 0) ∧ 
  (3 * n - 3 + 2 * n + 7 > 2 * n + 12) ∧
  (3 * n - 3 + 2 * n + 12 > 2 * n + 7) ∧
  (2 * n + 7 + 2 * n + 12 > 3 * n - 3) ∧
  (2 * n + 12 > 2 * n + 7) ∧
  (2 * n + 7 > 3 * n - 3) →
  (∃ (count : ℕ), count = 7 ∧ 
    (∀ (m : ℕ), (m ≥ 1 ∧ m ≤ count) ↔ 
      (∃ (k : ℕ), k ≥ 3 ∧ k ≤ 9 ∧
        (3 * k - 3 > 0) ∧ 
        (2 * k + 12 > 0) ∧ 
        (2 * k + 7 > 0) ∧ 
        (3 * k - 3 + 2 * k + 7 > 2 * k + 12) ∧
        (3 * k - 3 + 2 * k + 12 > 2 * k + 7) ∧
        (2 * k + 7 + 2 * k + 12 > 3 * k - 3) ∧
        (2 * k + 12 > 2 * k + 7) ∧
        (2 * k + 7 > 3 * k - 3)))) :=
by sorry

end NUMINAMATH_CALUDE_triangle_side_values_l2448_244854


namespace NUMINAMATH_CALUDE_wendy_sweaters_l2448_244886

/-- Represents the number of pieces of clothing a washing machine can wash in one load. -/
def machine_capacity : ℕ := 8

/-- Represents the number of shirts Wendy has to wash. -/
def num_shirts : ℕ := 39

/-- Represents the total number of loads Wendy has to do. -/
def total_loads : ℕ := 9

/-- Calculates the number of sweaters Wendy has to wash. -/
def num_sweaters : ℕ := (machine_capacity * total_loads) - num_shirts

theorem wendy_sweaters : num_sweaters = 33 := by
  sorry

end NUMINAMATH_CALUDE_wendy_sweaters_l2448_244886


namespace NUMINAMATH_CALUDE_perpendicular_implies_m_eq_half_parallel_implies_m_eq_neg_one_l2448_244860

-- Define the lines l₁ and l₂
def l₁ (m : ℝ) (x y : ℝ) : Prop := x + m * y + 6 = 0
def l₂ (m : ℝ) (x y : ℝ) : Prop := (m - 2) * x + 3 * y + 2 * m = 0

-- Define perpendicularity condition for two lines
def perpendicular (m : ℝ) : Prop := 1 * (m - 2) + m * 3 = 0

-- Define parallelism condition for two lines
def parallel (m : ℝ) : Prop := 1 / (m - 2) = m / 3 ∧ m ≠ 3

-- Theorem 1: If l₁ is perpendicular to l₂, then m = 1/2
theorem perpendicular_implies_m_eq_half :
  ∀ m : ℝ, perpendicular m → m = 1/2 :=
by sorry

-- Theorem 2: If l₁ is parallel to l₂, then m = -1
theorem parallel_implies_m_eq_neg_one :
  ∀ m : ℝ, parallel m → m = -1 :=
by sorry

end NUMINAMATH_CALUDE_perpendicular_implies_m_eq_half_parallel_implies_m_eq_neg_one_l2448_244860


namespace NUMINAMATH_CALUDE_elvis_studio_time_l2448_244824

/-- Calculates the total time spent in the studio for Elvis's album production -/
def total_studio_time (num_songs : ℕ) (record_time : ℕ) (edit_time : ℕ) (write_time : ℕ) : ℚ :=
  let total_minutes := num_songs * (record_time + write_time) + edit_time
  total_minutes / 60

/-- Proves that Elvis spent 5 hours in the studio given the specified conditions -/
theorem elvis_studio_time :
  total_studio_time 10 12 30 15 = 5 := by
sorry

end NUMINAMATH_CALUDE_elvis_studio_time_l2448_244824


namespace NUMINAMATH_CALUDE_cos_equality_problem_l2448_244874

theorem cos_equality_problem (n : ℤ) :
  0 ≤ n ∧ n ≤ 360 →
  (Real.cos (n * π / 180) = Real.cos (145 * π / 180) ↔ n = 145 ∨ n = 215) :=
by sorry

end NUMINAMATH_CALUDE_cos_equality_problem_l2448_244874


namespace NUMINAMATH_CALUDE_adam_chocolate_boxes_l2448_244815

/-- The number of boxes of chocolate candy Adam bought -/
def chocolate_boxes : ℕ := sorry

/-- The number of boxes of caramel candy Adam bought -/
def caramel_boxes : ℕ := 5

/-- The number of pieces of candy in each box -/
def pieces_per_box : ℕ := 4

/-- The total number of candies Adam had -/
def total_candies : ℕ := 28

theorem adam_chocolate_boxes :
  chocolate_boxes = 2 :=
by sorry

end NUMINAMATH_CALUDE_adam_chocolate_boxes_l2448_244815


namespace NUMINAMATH_CALUDE_chord_of_ellipse_l2448_244884

-- Define the real numbers m, n, s, t
variable (m n s t : ℝ)

-- Define the conditions
def conditions (m n s t : ℝ) : Prop :=
  m > 0 ∧ n > 0 ∧ s > 0 ∧ t > 0 ∧
  m + n = 3 ∧
  m / s + n / t = 1 ∧
  m < n ∧
  ∀ (s' t' : ℝ), s' > 0 → t' > 0 → m / s' + n / t' = 1 → s + t ≤ s' + t'

-- Define the ellipse
def ellipse (x y : ℝ) : Prop :=
  x^2 / 4 + y^2 / 16 = 1

-- Define the chord equation
def chord_equation (x y : ℝ) : Prop :=
  2 * x + y - 4 = 0

-- Theorem statement
theorem chord_of_ellipse (m n s t : ℝ) :
  conditions m n s t →
  (∃ (x₁ y₁ x₂ y₂ : ℝ),
    ellipse x₁ y₁ ∧ ellipse x₂ y₂ ∧
    (x₁ + x₂) / 2 = m ∧ (y₁ + y₂) / 2 = n ∧
    ∀ (x y : ℝ), x = (x₁ + x₂) / 2 ∧ y = (y₁ + y₂) / 2 → chord_equation x y) :=
by sorry

end NUMINAMATH_CALUDE_chord_of_ellipse_l2448_244884


namespace NUMINAMATH_CALUDE_imaginary_part_of_complex_number_l2448_244835

theorem imaginary_part_of_complex_number : Complex.im (Complex.I^2 * (1 + Complex.I)) = -1 := by
  sorry

end NUMINAMATH_CALUDE_imaginary_part_of_complex_number_l2448_244835


namespace NUMINAMATH_CALUDE_divisibility_by_square_of_n_minus_one_l2448_244847

theorem divisibility_by_square_of_n_minus_one (n : ℕ) (h : n > 2) :
  ∃ k : ℤ, (n : ℤ)^(n - 1) - 1 = k * (n - 1)^2 := by
  sorry

end NUMINAMATH_CALUDE_divisibility_by_square_of_n_minus_one_l2448_244847


namespace NUMINAMATH_CALUDE_quadratic_polynomial_negative_root_l2448_244855

theorem quadratic_polynomial_negative_root
  (f : ℝ → ℝ)
  (h1 : ∃ (a b : ℝ), a ≠ b ∧ f a = 0 ∧ f b = 0)
  (h2 : ∀ (a b : ℝ), f (a^2 + b^2) ≥ f (2*a*b)) :
  ∃ (x : ℝ), x < 0 ∧ f x = 0 :=
sorry

end NUMINAMATH_CALUDE_quadratic_polynomial_negative_root_l2448_244855


namespace NUMINAMATH_CALUDE_raisin_count_l2448_244810

theorem raisin_count (total : ℕ) (box1 : ℕ) (box345 : ℕ) (h1 : total = 437) 
  (h2 : box1 = 72) (h3 : box345 = 97) : 
  total - box1 - 3 * box345 = 74 := by
  sorry

end NUMINAMATH_CALUDE_raisin_count_l2448_244810


namespace NUMINAMATH_CALUDE_landmark_distance_set_l2448_244852

def distance_to_landmark (d : ℝ) : Prop :=
  d > 0 ∧ (d < 7 ∨ d > 7) ∧ (d ≤ 8 ∨ d > 8) ∧ (d ≤ 10 ∨ d > 10)

theorem landmark_distance_set :
  ∀ d : ℝ, distance_to_landmark d ↔ d > 10 :=
sorry

end NUMINAMATH_CALUDE_landmark_distance_set_l2448_244852


namespace NUMINAMATH_CALUDE_percentage_problem_l2448_244840

theorem percentage_problem (N : ℝ) (P : ℝ) 
  (h1 : 0.3 * N = 120) 
  (h2 : (P / 100) * N = 160) : 
  P = 40 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l2448_244840


namespace NUMINAMATH_CALUDE_intersection_of_M_and_N_l2448_244885

def M : Set ℝ := {2, 4, 6, 8, 10}
def N : Set ℝ := {x | -1 < x ∧ x < 6}

theorem intersection_of_M_and_N : M ∩ N = {2, 4} := by sorry

end NUMINAMATH_CALUDE_intersection_of_M_and_N_l2448_244885


namespace NUMINAMATH_CALUDE_equal_area_rectangles_width_l2448_244876

/-- Given two rectangles with equal area, where one rectangle has dimensions 5 inches by W inches,
    and the other has dimensions 8 inches by 15 inches, prove that W equals 24 inches. -/
theorem equal_area_rectangles_width (W : ℝ) : 
  (5 * W = 8 * 15) → W = 24 := by sorry

end NUMINAMATH_CALUDE_equal_area_rectangles_width_l2448_244876


namespace NUMINAMATH_CALUDE_min_reciprocal_sum_l2448_244875

theorem min_reciprocal_sum (m n : ℝ) (h1 : m > 0) (h2 : n > 0) (h3 : 2*m + 2*n = 1) :
  1/m + 1/n ≥ 8 := by
  sorry

end NUMINAMATH_CALUDE_min_reciprocal_sum_l2448_244875


namespace NUMINAMATH_CALUDE_given_equation_is_quadratic_l2448_244867

/-- A quadratic equation is of the form ax² + bx + c = 0, where a ≠ 0 -/
def is_quadratic_equation (f : ℝ → ℝ) : Prop :=
  ∃ (a b c : ℝ), a ≠ 0 ∧ ∀ x, f x = a * x^2 + b * x + c

/-- The given equation x² = 2x - 3x² -/
def given_equation (x : ℝ) : ℝ := x^2 - 2*x + 3*x^2

theorem given_equation_is_quadratic :
  is_quadratic_equation given_equation :=
sorry

end NUMINAMATH_CALUDE_given_equation_is_quadratic_l2448_244867


namespace NUMINAMATH_CALUDE_joe_tax_fraction_l2448_244898

/-- The fraction of income that goes to taxes -/
def tax_fraction (tax_payment : ℚ) (income : ℚ) : ℚ :=
  tax_payment / income

theorem joe_tax_fraction :
  let monthly_tax : ℚ := 848
  let monthly_income : ℚ := 2120
  tax_fraction monthly_tax monthly_income = 106 / 265 := by
sorry

end NUMINAMATH_CALUDE_joe_tax_fraction_l2448_244898


namespace NUMINAMATH_CALUDE_inequality_solution_set_l2448_244842

theorem inequality_solution_set (x : ℝ) :
  (4 * x^3 + 9 * x^2 - 6 * x < 2) ↔ ((-2 < x ∧ x < -1) ∨ (-1 < x ∧ x < 1/4)) := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l2448_244842


namespace NUMINAMATH_CALUDE_unique_prime_seven_digit_number_l2448_244801

def is_prime (n : ℕ) : Prop := n > 1 ∧ ∀ d : ℕ, d > 1 → d < n → ¬(n % d = 0)

def seven_digit_number (B : ℕ) : ℕ := 2024050 + B

theorem unique_prime_seven_digit_number :
  ∃! B : ℕ, B < 10 ∧ is_prime (seven_digit_number B) ∧ seven_digit_number B = 2024051 :=
sorry

end NUMINAMATH_CALUDE_unique_prime_seven_digit_number_l2448_244801


namespace NUMINAMATH_CALUDE_maggie_spent_170_l2448_244836

/-- The total amount Maggie spent on books and magazines -/
def total_spent (num_books num_magazines book_price magazine_price : ℕ) : ℕ :=
  num_books * book_price + num_magazines * magazine_price

/-- Theorem stating that Maggie spent $170 in total -/
theorem maggie_spent_170 :
  total_spent 10 10 15 2 = 170 := by
  sorry

end NUMINAMATH_CALUDE_maggie_spent_170_l2448_244836


namespace NUMINAMATH_CALUDE_distinct_tower_heights_l2448_244877

/-- Represents the number of bricks in the tower. -/
def num_bricks : ℕ := 50

/-- Represents the minimum possible height of the tower in inches. -/
def min_height : ℕ := 250

/-- Represents the maximum possible height of the tower in inches. -/
def max_height : ℕ := 900

/-- The theorem stating the number of distinct tower heights achievable. -/
theorem distinct_tower_heights :
  ∃ (heights : Finset ℕ),
    (∀ h ∈ heights, min_height ≤ h ∧ h ≤ max_height) ∧
    (∀ h, min_height ≤ h → h ≤ max_height →
      (∃ (a b c : ℕ), a + b + c = num_bricks ∧ 5*a + 12*b + 18*c = h) ↔ h ∈ heights) ∧
    heights.card = 651 := by
  sorry

end NUMINAMATH_CALUDE_distinct_tower_heights_l2448_244877


namespace NUMINAMATH_CALUDE_smartphone_loss_percentage_l2448_244833

theorem smartphone_loss_percentage (initial_cost selling_price : ℝ) :
  initial_cost = 300 →
  selling_price = 255 →
  (initial_cost - selling_price) / initial_cost * 100 = 15 := by
sorry

end NUMINAMATH_CALUDE_smartphone_loss_percentage_l2448_244833


namespace NUMINAMATH_CALUDE_robin_candy_count_l2448_244834

/-- The number of candy packages Robin has -/
def candy_packages : ℕ := 45

/-- The number of pieces in each package -/
def pieces_per_package : ℕ := 9

/-- The total number of candy pieces Robin has -/
def total_candy_pieces : ℕ := candy_packages * pieces_per_package

theorem robin_candy_count : total_candy_pieces = 405 := by
  sorry

end NUMINAMATH_CALUDE_robin_candy_count_l2448_244834


namespace NUMINAMATH_CALUDE_factorial_product_less_than_factorial_sum_l2448_244846

theorem factorial_product_less_than_factorial_sum {n : ℕ} (k : ℕ) (a : Fin n → ℕ) 
  (h_pos : ∀ i, a i > 0) (h_sum : (Finset.univ.sum a) < k) : 
  (Finset.univ.prod (λ i => Nat.factorial (a i))) < Nat.factorial k := by
sorry

end NUMINAMATH_CALUDE_factorial_product_less_than_factorial_sum_l2448_244846


namespace NUMINAMATH_CALUDE_inequality_solution_set_l2448_244804

theorem inequality_solution_set (m n : ℝ) : 
  (∀ x, mx - n > 0 ↔ x < 1/3) →
  (∀ x, (m + n) * x < n - m ↔ x > -1/2) :=
by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l2448_244804


namespace NUMINAMATH_CALUDE_oliver_shelf_capacity_l2448_244889

/-- Given the total number of books, the number of books taken by the librarian,
    and the number of shelves needed, calculate the number of books that can fit on each shelf. -/
def books_per_shelf (total_books : ℕ) (books_taken : ℕ) (shelves_needed : ℕ) : ℕ :=
  (total_books - books_taken) / shelves_needed

/-- Prove that Oliver can fit 4 books on each shelf given the problem conditions. -/
theorem oliver_shelf_capacity :
  books_per_shelf 46 10 9 = 4 := by
  sorry

end NUMINAMATH_CALUDE_oliver_shelf_capacity_l2448_244889


namespace NUMINAMATH_CALUDE_square_difference_l2448_244895

theorem square_difference (x y : ℚ) 
  (h1 : x + y = 7/13) 
  (h2 : x - y = 1/91) : 
  x^2 - y^2 = 1/169 := by
sorry

end NUMINAMATH_CALUDE_square_difference_l2448_244895


namespace NUMINAMATH_CALUDE_dragon_tower_theorem_l2448_244818

/-- Represents the configuration of a dragon tethered to a cylindrical tower. -/
structure DragonTower where
  towerRadius : ℝ
  ropeLength : ℝ
  dragonHeight : ℝ
  ropeTowerDistance : ℝ

/-- Represents the parameters of the rope touching the tower. -/
structure RopeParameters where
  p : ℕ
  q : ℕ
  r : ℕ

/-- Theorem stating the relationship between the dragon-tower configuration
    and the rope parameters. -/
theorem dragon_tower_theorem (dt : DragonTower) (rp : RopeParameters) :
  dt.towerRadius = 10 ∧
  dt.ropeLength = 30 ∧
  dt.dragonHeight = 6 ∧
  dt.ropeTowerDistance = 6 ∧
  Nat.Prime rp.r ∧
  (rp.p - Real.sqrt rp.q) / rp.r = Real.sqrt ((dt.ropeLength - dt.ropeTowerDistance)^2 - dt.towerRadius^2) -
    (dt.ropeLength * Real.sqrt (dt.towerRadius^2 + dt.dragonHeight^2)) / dt.towerRadius +
    dt.dragonHeight * Real.sqrt (dt.towerRadius^2 + dt.dragonHeight^2) / dt.towerRadius →
  rp.p + rp.q + rp.r = 993 :=
by sorry

end NUMINAMATH_CALUDE_dragon_tower_theorem_l2448_244818


namespace NUMINAMATH_CALUDE_gcf_three_digit_palindromes_l2448_244825

def is_three_digit_palindrome (n : ℕ) : Prop :=
  100 ≤ n ∧ n ≤ 999 ∧ ∃ (a b : ℕ), a ≠ 0 ∧ a ≤ 9 ∧ b ≤ 9 ∧ n = 101 * a + 10 * b

theorem gcf_three_digit_palindromes :
  ∃ (g : ℕ), g > 0 ∧
    (∀ (n : ℕ), is_three_digit_palindrome n → g ∣ n) ∧
    (∀ (d : ℕ), d > 0 → (∀ (n : ℕ), is_three_digit_palindrome n → d ∣ n) → d ≤ g) ∧
    g = 101 := by sorry

end NUMINAMATH_CALUDE_gcf_three_digit_palindromes_l2448_244825


namespace NUMINAMATH_CALUDE_centroid_locus_l2448_244822

/-- The locus of the centroid of a triangle formed by specific points on a parabola and a line -/
theorem centroid_locus (k : ℝ) (x y : ℝ) : 
  -- Line l: y = k(x - 2)
  -- Parabola: y = x^2 + 2
  (k ≠ 0) →
  (k < 4 - 2 * Real.sqrt 6 ∨ k > 4 + 2 * Real.sqrt 6) →
  -- Point A(2,0)
  -- Origin O(0,0)
  -- B and C are intersections of line l and the parabola
  -- P on BC such that BP = (|BB1|/|CC1|) * PC
  -- G is the centroid of triangle POA
  (x = (4 / (k - 4)) + 4/3) →
  (y = (4 * k) / (k - 4)) →
  (12 * x - 3 * y - 4 = 0 ∧ 
   4 - (4/3) * Real.sqrt 6 < y ∧ 
   y < 4 + (4/3) * Real.sqrt 6 ∧ 
   y ≠ 4) :=
by sorry

end NUMINAMATH_CALUDE_centroid_locus_l2448_244822


namespace NUMINAMATH_CALUDE_article_word_count_l2448_244812

theorem article_word_count 
  (total_pages : ℕ) 
  (small_type_pages : ℕ) 
  (words_per_large_page : ℕ) 
  (words_per_small_page : ℕ) 
  (h1 : total_pages = 21) 
  (h2 : small_type_pages = 17) 
  (h3 : words_per_large_page = 1800) 
  (h4 : words_per_small_page = 2400) : 
  (total_pages - small_type_pages) * words_per_large_page + 
  small_type_pages * words_per_small_page = 48000 := by
sorry

end NUMINAMATH_CALUDE_article_word_count_l2448_244812


namespace NUMINAMATH_CALUDE_quadratic_equation_solution_l2448_244858

theorem quadratic_equation_solution (a : ℝ) : 
  (∀ x : ℝ, x = 1 → a * x^2 - 6 * x + 3 = 0) → a = 3 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_equation_solution_l2448_244858


namespace NUMINAMATH_CALUDE_pure_imaginary_sum_l2448_244897

theorem pure_imaginary_sum (a b c d : ℝ) : 
  let z₁ : ℂ := a + b * Complex.I
  let z₂ : ℂ := c + d * Complex.I
  (z₁ + z₂).re = 0 ∧ (z₁ + z₂).im ≠ 0 → a + c = 0 ∧ b + d ≠ 0 :=
by sorry

end NUMINAMATH_CALUDE_pure_imaginary_sum_l2448_244897


namespace NUMINAMATH_CALUDE_crayons_per_unit_is_six_l2448_244862

/-- Given the total number of units, cost per crayon, and total cost,
    calculate the number of crayons in each unit. -/
def crayons_per_unit (total_units : ℕ) (cost_per_crayon : ℚ) (total_cost : ℚ) : ℚ :=
  (total_cost / cost_per_crayon) / total_units

/-- Theorem stating that under the given conditions, there are 6 crayons in each unit. -/
theorem crayons_per_unit_is_six :
  crayons_per_unit 4 2 48 = 6 := by
  sorry

#eval crayons_per_unit 4 2 48

end NUMINAMATH_CALUDE_crayons_per_unit_is_six_l2448_244862


namespace NUMINAMATH_CALUDE_sphere_cube_volume_constant_l2448_244814

/-- The value of K when a sphere has the same surface area as a cube with side length 3
    and its volume is expressed as (K * sqrt(6)) / sqrt(π) -/
theorem sphere_cube_volume_constant (cube_side : ℝ) (sphere_volume : ℝ → ℝ) : 
  cube_side = 3 →
  (4 * π * (sphere_volume K / ((4 / 3) * π))^(2/3) = 6 * cube_side^2) →
  sphere_volume K = K * Real.sqrt 6 / Real.sqrt π →
  K = 27 * Real.sqrt 6 / Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_sphere_cube_volume_constant_l2448_244814


namespace NUMINAMATH_CALUDE_bryans_bookshelves_l2448_244807

theorem bryans_bookshelves (total_books : ℕ) (books_per_shelf : ℕ) (h1 : total_books = 38) (h2 : books_per_shelf = 2) :
  total_books / books_per_shelf = 19 := by
  sorry

end NUMINAMATH_CALUDE_bryans_bookshelves_l2448_244807


namespace NUMINAMATH_CALUDE_equal_sandwiched_segments_imply_parallel_or_intersecting_planes_l2448_244869

-- Define a plane
structure Plane where
  -- Add necessary fields for a plane

-- Define a line segment
structure LineSegment where
  -- Add necessary fields for a line segment

-- Define the property of being sandwiched between two planes
def sandwichedBetween (l : LineSegment) (p1 p2 : Plane) : Prop :=
  sorry

-- Define the property of line segments being parallel
def areParallel (l1 l2 l3 : LineSegment) : Prop :=
  sorry

-- Define the property of line segments being equal
def areEqual (l1 l2 l3 : LineSegment) : Prop :=
  sorry

-- Define the property of planes being parallel
def arePlanesParallel (p1 p2 : Plane) : Prop :=
  sorry

-- Define the property of planes intersecting
def arePlanesIntersecting (p1 p2 : Plane) : Prop :=
  sorry

-- The main theorem
theorem equal_sandwiched_segments_imply_parallel_or_intersecting_planes 
  (p1 p2 : Plane) (l1 l2 l3 : LineSegment) :
  sandwichedBetween l1 p1 p2 →
  sandwichedBetween l2 p1 p2 →
  sandwichedBetween l3 p1 p2 →
  areParallel l1 l2 l3 →
  areEqual l1 l2 l3 →
  arePlanesParallel p1 p2 ∨ arePlanesIntersecting p1 p2 :=
sorry

end NUMINAMATH_CALUDE_equal_sandwiched_segments_imply_parallel_or_intersecting_planes_l2448_244869


namespace NUMINAMATH_CALUDE_range_of_m_l2448_244873

-- Define the propositions r(x) and s(x)
def r (x m : ℝ) : Prop := Real.sin x + Real.cos x > m
def s (x m : ℝ) : Prop := x^2 + m*x + 1 > 0

-- Define the theorem
theorem range_of_m :
  (∀ x : ℝ, (r x m ∧ ¬(s x m)) ∨ (¬(r x m) ∧ s x m)) →
  (m ≤ -2 ∨ (-Real.sqrt 2 ≤ m ∧ m < 2)) :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l2448_244873


namespace NUMINAMATH_CALUDE_same_color_probability_l2448_244828

theorem same_color_probability (N : ℕ) : 
  (4 : ℚ) / 10 * 16 / (16 + N) + (6 : ℚ) / 10 * N / (16 + N) = 29 / 50 → N = 144 := by
  sorry

end NUMINAMATH_CALUDE_same_color_probability_l2448_244828


namespace NUMINAMATH_CALUDE_table_runner_coverage_l2448_244890

theorem table_runner_coverage (total_runner_area : ℝ) (table_area : ℝ) 
  (two_layer_area : ℝ) (three_layer_area : ℝ) 
  (h1 : total_runner_area = 204)
  (h2 : table_area = 175)
  (h3 : two_layer_area = 24)
  (h4 : three_layer_area = 20) :
  (((total_runner_area - 2 * two_layer_area - 3 * three_layer_area) + 
    two_layer_area + three_layer_area) / table_area) * 100 = 80 := by
  sorry

end NUMINAMATH_CALUDE_table_runner_coverage_l2448_244890


namespace NUMINAMATH_CALUDE_notebook_pen_cost_l2448_244800

theorem notebook_pen_cost :
  ∀ (n p : ℕ),
  15 * n + 4 * p = 160 →
  n > p →
  n + p = 18 :=
by
  sorry

end NUMINAMATH_CALUDE_notebook_pen_cost_l2448_244800


namespace NUMINAMATH_CALUDE_notebook_cost_l2448_244805

theorem notebook_cost (total_students : Nat) (total_cost : Nat) : ∃ (buying_students : Nat) (notebooks_per_student : Nat) (cost_per_notebook : Nat),
  total_students = 50 ∧
  total_cost = 2739 ∧
  buying_students > total_students / 2 ∧
  notebooks_per_student % 2 = 1 ∧
  notebooks_per_student > 1 ∧
  cost_per_notebook > notebooks_per_student ∧
  buying_students * notebooks_per_student * cost_per_notebook = total_cost ∧
  cost_per_notebook = 7 := by
  sorry

end NUMINAMATH_CALUDE_notebook_cost_l2448_244805


namespace NUMINAMATH_CALUDE_pure_imaginary_product_l2448_244863

theorem pure_imaginary_product (x : ℝ) : 
  (∃ k : ℝ, (x + 2*Complex.I) * ((x + 2) + 2*Complex.I) * ((x + 4) + 2*Complex.I) = k * Complex.I) ↔ 
  (x = -4 ∨ x = -1 + 2 * Real.sqrt 5 ∨ x = -1 - 2 * Real.sqrt 5) :=
sorry

end NUMINAMATH_CALUDE_pure_imaginary_product_l2448_244863


namespace NUMINAMATH_CALUDE_train_speed_problem_l2448_244878

theorem train_speed_problem (x : ℝ) (v : ℝ) :
  x > 0 →
  (x / v + 2 * x / 20 = 4 * x / 32) →
  v = 8.8 := by
sorry

end NUMINAMATH_CALUDE_train_speed_problem_l2448_244878


namespace NUMINAMATH_CALUDE_geometric_series_sum_l2448_244826

/-- Sum of a geometric series -/
def geometric_sum (a : ℚ) (r : ℚ) (n : ℕ) : ℚ :=
  a * (1 - r^n) / (1 - r)

/-- The first term of the geometric series -/
def a : ℚ := 2

/-- The common ratio of the geometric series -/
def r : ℚ := -2

/-- The number of terms in the geometric series -/
def n : ℕ := 10

theorem geometric_series_sum :
  geometric_sum a r n = 2050 / 3 := by sorry

end NUMINAMATH_CALUDE_geometric_series_sum_l2448_244826


namespace NUMINAMATH_CALUDE_first_part_to_total_ratio_l2448_244838

theorem first_part_to_total_ratio (total : ℚ) (first_part : ℚ) : 
  total = 782 →
  first_part = 204 →
  ∃ (x : ℚ), (x + 2/3 + 3/4) * first_part = total →
  first_part / total = 102 / 391 := by
  sorry

end NUMINAMATH_CALUDE_first_part_to_total_ratio_l2448_244838


namespace NUMINAMATH_CALUDE_units_digit_sum_of_powers_l2448_244811

theorem units_digit_sum_of_powers : ∃ n : ℕ, n < 10 ∧ (45^125 + 7^87) % 10 = n ∧ n = 8 := by
  sorry

end NUMINAMATH_CALUDE_units_digit_sum_of_powers_l2448_244811


namespace NUMINAMATH_CALUDE_min_value_expression_l2448_244850

theorem min_value_expression (x y : ℝ) : (x*y + 1)^2 + (x^2 + y^2)^2 ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_min_value_expression_l2448_244850
