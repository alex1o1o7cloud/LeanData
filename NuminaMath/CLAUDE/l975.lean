import Mathlib

namespace NUMINAMATH_CALUDE_trajectory_area_is_8_l975_97549

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a cuboid -/
structure Cuboid where
  a : Point3D
  b : Point3D
  c : Point3D
  d : Point3D
  a₁ : Point3D
  b₁ : Point3D
  c₁ : Point3D
  d₁ : Point3D

/-- The length of AB in the cuboid -/
def ab_length : ℝ := 6

/-- The length of BC in the cuboid -/
def bc_length : ℝ := 3

/-- A moving point P on line segment BD -/
def P (t : ℝ) : Point3D :=
  sorry

/-- A moving point Q on line segment A₁C₁ -/
def Q (t : ℝ) : Point3D :=
  sorry

/-- Point M on PQ such that PM = 2MQ -/
def M (t₁ t₂ : ℝ) : Point3D :=
  sorry

/-- The area of the trajectory of point M -/
def trajectory_area (c : Cuboid) : ℝ :=
  sorry

/-- Theorem stating that the area of the trajectory of point M is 8 -/
theorem trajectory_area_is_8 (c : Cuboid) :
  trajectory_area c = 8 :=
  sorry

end NUMINAMATH_CALUDE_trajectory_area_is_8_l975_97549


namespace NUMINAMATH_CALUDE_crayons_left_l975_97589

theorem crayons_left (initial_crayons lost_crayons : ℕ) 
  (h1 : initial_crayons = 253)
  (h2 : lost_crayons = 70) :
  initial_crayons - lost_crayons = 183 := by
sorry

end NUMINAMATH_CALUDE_crayons_left_l975_97589


namespace NUMINAMATH_CALUDE_unique_solution_l975_97550

/-- A triplet of natural numbers (a, b, c) where b and c are two-digit numbers. -/
structure Triplet where
  a : ℕ
  b : ℕ
  c : ℕ
  b_twodigit : 10 ≤ b ∧ b ≤ 99
  c_twodigit : 10 ≤ c ∧ c ≤ 99

/-- The property that a triplet (a, b, c) satisfies the equation 10^4*a + 100*b + c = (a + b + c)^3. -/
def satisfies_equation (t : Triplet) : Prop :=
  10^4 * t.a + 100 * t.b + t.c = (t.a + t.b + t.c)^3

/-- Theorem stating that (9, 11, 25) is the only triplet satisfying the equation. -/
theorem unique_solution :
  ∃! t : Triplet, satisfies_equation t ∧ t.a = 9 ∧ t.b = 11 ∧ t.c = 25 :=
sorry

end NUMINAMATH_CALUDE_unique_solution_l975_97550


namespace NUMINAMATH_CALUDE_highest_power_of_three_for_concatenated_range_l975_97510

def concatenate_range (a b : ℕ) : ℕ := sorry

def highest_power_of_three (n : ℕ) : ℕ := sorry

theorem highest_power_of_three_for_concatenated_range :
  let N := concatenate_range 31 73
  highest_power_of_three N = 1 := by sorry

end NUMINAMATH_CALUDE_highest_power_of_three_for_concatenated_range_l975_97510


namespace NUMINAMATH_CALUDE_remainder_theorem_l975_97587

-- Define the polynomial p(x)
variable (p : ℝ → ℝ)

-- Define the conditions
axiom remainder_x_minus_3 : ∃ q : ℝ → ℝ, ∀ x, p x = (x - 3) * q x + 7
axiom remainder_x_plus_2 : ∃ q : ℝ → ℝ, ∀ x, p x = (x + 2) * q x - 3

-- Theorem statement
theorem remainder_theorem :
  ∃ q : ℝ → ℝ, ∀ x, p x = (x - 3) * (x + 2) * q x + (2 * x + 1) :=
sorry

end NUMINAMATH_CALUDE_remainder_theorem_l975_97587


namespace NUMINAMATH_CALUDE_geometric_sequence_product_threshold_l975_97548

theorem geometric_sequence_product_threshold (n : ℕ) : 
  (n > 0 ∧ 3^((n * (n + 1)) / 12) > 1000) ↔ n ≥ 6 := by
  sorry

end NUMINAMATH_CALUDE_geometric_sequence_product_threshold_l975_97548


namespace NUMINAMATH_CALUDE_power_digits_sum_l975_97547

theorem power_digits_sum : ∃ (m n : ℕ), 
  (100 ≤ 2^m ∧ 2^m < 10000) ∧ 
  (100 ≤ 5^n ∧ 5^n < 10000) ∧ 
  (2^m / 100 % 10 = 5^n / 100 % 10) ∧
  (2^m / 100 % 10 + 5^n / 100 % 10 = 4) :=
sorry

end NUMINAMATH_CALUDE_power_digits_sum_l975_97547


namespace NUMINAMATH_CALUDE_average_sum_is_six_l975_97584

theorem average_sum_is_six (a b c d e : ℕ) (h : a + b + c + d + e > 0) :
  let teacher_avg := (5*a + 4*b + 3*c + 2*d + e) / (a + b + c + d + e)
  let kati_avg := (5*e + 4*d + 3*c + 2*b + a) / (a + b + c + d + e)
  teacher_avg + kati_avg = 6 := by
  sorry

end NUMINAMATH_CALUDE_average_sum_is_six_l975_97584


namespace NUMINAMATH_CALUDE_complex_expression_equality_l975_97543

theorem complex_expression_equality : 
  ∀ (z₁ z₂ : ℂ), 
    z₁ = 2 - I → 
    z₂ = -I → 
    z₁ / z₂ + Complex.abs z₂ = 2 + 2*I := by
sorry

end NUMINAMATH_CALUDE_complex_expression_equality_l975_97543


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_sets_l975_97565

theorem quadratic_inequality_solution_sets (a b c : ℝ) :
  (∀ x, ax^2 - b*x + c ≥ 0 ↔ 1 ≤ x ∧ x ≤ 2) →
  (∀ x, c*x^2 + b*x + a ≤ 0 ↔ x ≤ -1 ∨ x ≥ -1/2) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_sets_l975_97565


namespace NUMINAMATH_CALUDE_chocolates_left_first_method_candies_left_second_method_chocolates_left_second_method_total_items_is_35_l975_97531

/-- Represents the number of bags packed using the first method -/
def x : ℕ := 1

/-- Represents the number of bags packed using the second method -/
def y : ℕ := 0

/-- The total number of chocolates initially -/
def total_chocolates : ℕ := 3 * x + 5 * y + 25

/-- The total number of fruit candies initially -/
def total_candies : ℕ := 7 * x + 5 * y

/-- Condition: When fruit candies are used up in the first method, 25 chocolates are left -/
theorem chocolates_left_first_method : total_chocolates - (3 * x + 5 * y) = 25 := by sorry

/-- Condition: In the second method, 4 fruit candies are left in the end -/
theorem candies_left_second_method : total_candies - (7 * x + 5 * y) = 4 := by sorry

/-- Condition: In the second method, 1 chocolate is left in the end -/
theorem chocolates_left_second_method : total_chocolates - (3 * x + 5 * y) - 4 = 1 := by sorry

/-- The main theorem: The total number of chocolates and fruit candies is 35 -/
theorem total_items_is_35 : total_chocolates + total_candies = 35 := by sorry

end NUMINAMATH_CALUDE_chocolates_left_first_method_candies_left_second_method_chocolates_left_second_method_total_items_is_35_l975_97531


namespace NUMINAMATH_CALUDE_sachins_age_l975_97512

theorem sachins_age (sachin_age rahul_age : ℝ) 
  (age_difference : rahul_age = sachin_age + 7)
  (age_ratio : sachin_age / rahul_age = 7 / 9) :
  sachin_age = 24.5 := by
  sorry

end NUMINAMATH_CALUDE_sachins_age_l975_97512


namespace NUMINAMATH_CALUDE_balcony_orchestra_difference_l975_97530

/-- Represents the number of tickets sold for a theater performance -/
structure TheaterTickets where
  orchestra : ℕ
  balcony : ℕ

/-- Calculates the total number of tickets sold -/
def totalTickets (t : TheaterTickets) : ℕ := t.orchestra + t.balcony

/-- Calculates the total revenue from ticket sales -/
def totalRevenue (t : TheaterTickets) : ℕ := 12 * t.orchestra + 8 * t.balcony

theorem balcony_orchestra_difference (t : TheaterTickets) :
  totalTickets t = 355 → totalRevenue t = 3320 → t.balcony - t.orchestra = 115 := by
  sorry

#check balcony_orchestra_difference

end NUMINAMATH_CALUDE_balcony_orchestra_difference_l975_97530


namespace NUMINAMATH_CALUDE_intersection_M_N_l975_97527

-- Define the sets M and N
def M : Set ℝ := {x | x < 1}
def N : Set ℝ := {y | y > -1}

-- State the theorem
theorem intersection_M_N : M ∩ N = Set.Ioo (-1) 1 := by sorry

end NUMINAMATH_CALUDE_intersection_M_N_l975_97527


namespace NUMINAMATH_CALUDE_percentage_problem_l975_97517

theorem percentage_problem (P : ℝ) : 
  (P / 100) * 40 = (5 / 100) * 60 + 23 → P = 65 := by
  sorry

end NUMINAMATH_CALUDE_percentage_problem_l975_97517


namespace NUMINAMATH_CALUDE_gcf_seven_eight_factorial_l975_97590

theorem gcf_seven_eight_factorial :
  Nat.gcd (Nat.factorial 7) (Nat.factorial 8) = Nat.factorial 7 := by
  sorry

end NUMINAMATH_CALUDE_gcf_seven_eight_factorial_l975_97590


namespace NUMINAMATH_CALUDE_parallelogram_network_l975_97598

theorem parallelogram_network (first_set : ℕ) (total_parallelograms : ℕ) (second_set : ℕ) : 
  first_set = 7 → 
  total_parallelograms = 588 → 
  total_parallelograms = (first_set - 1) * (second_set - 1) → 
  second_set = 99 :=
by sorry

end NUMINAMATH_CALUDE_parallelogram_network_l975_97598


namespace NUMINAMATH_CALUDE_min_sheets_for_boats_l975_97599

theorem min_sheets_for_boats (boats_per_sheet : ℕ) (planes_per_sheet : ℕ) (total_toys : ℕ) :
  boats_per_sheet = 8 →
  planes_per_sheet = 6 →
  total_toys = 80 →
  ∃ (sheets : ℕ), 
    sheets * boats_per_sheet = total_toys ∧
    sheets = 10 ∧
    (∀ (s : ℕ), s * boats_per_sheet = total_toys → s ≥ sheets) :=
by sorry

end NUMINAMATH_CALUDE_min_sheets_for_boats_l975_97599


namespace NUMINAMATH_CALUDE_men_work_hours_l975_97564

theorem men_work_hours (men : ℕ) (women : ℕ) (men_days : ℕ) (women_days : ℕ) (women_hours : ℕ) (H : ℚ) :
  men = 15 →
  women = 21 →
  men_days = 21 →
  women_days = 60 →
  women_hours = 3 →
  (3 : ℚ) * men * men_days * H = 2 * women * women_days * women_hours →
  H = 8 := by
sorry

end NUMINAMATH_CALUDE_men_work_hours_l975_97564


namespace NUMINAMATH_CALUDE_greatest_integer_inequality_l975_97532

theorem greatest_integer_inequality (y : ℤ) : (8 : ℚ) / 11 > (y : ℚ) / 15 ↔ y ≤ 10 := by sorry

end NUMINAMATH_CALUDE_greatest_integer_inequality_l975_97532


namespace NUMINAMATH_CALUDE_unique_special_number_l975_97558

/-- Represents a four-digit number --/
structure FourDigitNumber where
  first : Nat
  second : Nat
  third : Nat
  fourth : Nat
  first_is_digit : first < 10
  second_is_digit : second < 10
  third_is_digit : third < 10
  fourth_is_digit : fourth < 10

/-- Conditions for the special four-digit number --/
def SpecialNumber (n : FourDigitNumber) : Prop :=
  n.first + n.second + n.third + n.fourth = 8 ∧
  n.first = 3 * n.second ∧
  n.fourth = 4 * n.third

theorem unique_special_number :
  ∃! n : FourDigitNumber, SpecialNumber n ∧ 
    n.first = 6 ∧ n.second = 2 ∧ n.third = 0 ∧ n.fourth = 0 :=
by sorry

#check unique_special_number

end NUMINAMATH_CALUDE_unique_special_number_l975_97558


namespace NUMINAMATH_CALUDE_inequality_proof_l975_97576

theorem inequality_proof (a b c : ℝ) (ha : 0 < a) (hb : 0 < b) (hc : 0 < c) (habc : a * b * c = 1) :
  (a - 1 + 1/b) * (b - 1 + 1/c) * (c - 1 + 1/a) ≤ 1 := by
sorry

end NUMINAMATH_CALUDE_inequality_proof_l975_97576


namespace NUMINAMATH_CALUDE_sum_of_ages_in_three_years_l975_97513

-- Define the current ages
def jeremy_current_age : ℕ := 40
def sebastian_current_age : ℕ := jeremy_current_age + 4
def sophia_future_age : ℕ := 60

-- Define the ages in three years
def jeremy_future_age : ℕ := jeremy_current_age + 3
def sebastian_future_age : ℕ := sebastian_current_age + 3
def sophia_current_age : ℕ := sophia_future_age - 3

-- Theorem to prove
theorem sum_of_ages_in_three_years :
  jeremy_future_age + sebastian_future_age + sophia_future_age = 150 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_ages_in_three_years_l975_97513


namespace NUMINAMATH_CALUDE_reinforcement_size_problem_l975_97577

/-- Given a garrison with initial men, initial provision days, days before reinforcement,
    and remaining days after reinforcement, calculate the size of the reinforcement. -/
def reinforcement_size (initial_men : ℕ) (initial_days : ℕ) (days_before_reinforcement : ℕ) 
                       (remaining_days : ℕ) : ℕ :=
  let total_provisions := initial_men * initial_days
  let remaining_provisions := initial_men * (initial_days - days_before_reinforcement)
  let total_men_after := remaining_provisions / remaining_days
  total_men_after - initial_men

/-- The size of the reinforcement for the given problem is 1300. -/
theorem reinforcement_size_problem : 
  reinforcement_size 2000 54 21 20 = 1300 := by
  sorry

end NUMINAMATH_CALUDE_reinforcement_size_problem_l975_97577


namespace NUMINAMATH_CALUDE_vector_combination_l975_97540

-- Define the vector space
variable {V : Type*} [AddCommGroup V] [Module ℝ V]

-- Define the basis vectors
variable (e₁ e₂ : V)

-- Define vectors a and b
def a (e₁ e₂ : V) : V := e₁ + 2 • e₂
def b (e₁ e₂ : V) : V := 3 • e₁ - e₂

-- State the theorem
theorem vector_combination (e₁ e₂ : V) :
  3 • (a e₁ e₂) - 2 • (b e₁ e₂) = -3 • e₁ + 8 • e₂ := by
  sorry

end NUMINAMATH_CALUDE_vector_combination_l975_97540


namespace NUMINAMATH_CALUDE_solution_system_l975_97581

theorem solution_system (x y : ℝ) 
  (h1 : x * y = 10)
  (h2 : x^2 * y + x * y^2 + 2*x + 2*y = 88) : 
  x^2 + y^2 = 304/9 := by
sorry

end NUMINAMATH_CALUDE_solution_system_l975_97581


namespace NUMINAMATH_CALUDE_negation_equivalence_l975_97574

theorem negation_equivalence :
  (¬ ∀ x : ℝ, x > Real.sin x) ↔ (∃ x : ℝ, x ≤ Real.sin x) := by
  sorry

end NUMINAMATH_CALUDE_negation_equivalence_l975_97574


namespace NUMINAMATH_CALUDE_carrie_tshirt_purchase_l975_97502

/-- The number of t-shirts Carrie bought -/
def num_tshirts : ℕ := 24

/-- The cost of each t-shirt in dollars -/
def cost_per_tshirt : ℚ := 9.95

/-- The total amount Carrie spent in dollars -/
def total_spent : ℚ := 248

/-- Theorem stating that the number of t-shirts Carrie bought is correct -/
theorem carrie_tshirt_purchase : 
  (↑num_tshirts : ℚ) * cost_per_tshirt ≤ total_spent ∧ 
  (↑(num_tshirts + 1) : ℚ) * cost_per_tshirt > total_spent :=
by sorry

end NUMINAMATH_CALUDE_carrie_tshirt_purchase_l975_97502


namespace NUMINAMATH_CALUDE_wire_ratio_theorem_l975_97538

theorem wire_ratio_theorem (B C : ℝ) (h1 : B > 0) (h2 : C > 0) (h3 : B + C = 80) : 
  ∃ (r : ℝ → ℝ → ℝ → Prop), r 16 B C ∧ 
  (∀ (x y z : ℝ), r x y z ↔ ∃ (k : ℝ), k > 0 ∧ x = 16 * k ∧ y = B * k ∧ z = C * k) :=
sorry

end NUMINAMATH_CALUDE_wire_ratio_theorem_l975_97538


namespace NUMINAMATH_CALUDE_polynomial_factor_sum_l975_97503

theorem polynomial_factor_sum (m n : ℚ) : 
  (∀ y, my^2 + n*y + 2 = (y + 1)*(y + 2)) → m + n = 4 := by
  sorry

end NUMINAMATH_CALUDE_polynomial_factor_sum_l975_97503


namespace NUMINAMATH_CALUDE_rogers_retirement_experience_l975_97562

/-- Represents the years of experience for each coworker -/
structure Experience where
  roger : ℕ
  peter : ℕ
  tom : ℕ
  robert : ℕ
  mike : ℕ

/-- The conditions given in the problem -/
def satisfies_conditions (e : Experience) : Prop :=
  e.roger = e.peter + e.tom + e.robert + e.mike ∧
  e.peter = 12 ∧
  e.tom = 2 * e.robert ∧
  e.robert = e.peter - 4 ∧
  e.robert = e.mike + 2

/-- The theorem to be proved -/
theorem rogers_retirement_experience (e : Experience) :
  satisfies_conditions e → e.roger + 8 = 50 :=
by
  sorry

end NUMINAMATH_CALUDE_rogers_retirement_experience_l975_97562


namespace NUMINAMATH_CALUDE_max_truck_load_is_2000_l975_97519

/-- Represents the maximum load a truck can carry given the following conditions:
    - There are three trucks for delivery
    - Boxes come in two weights: 10 pounds and 40 pounds
    - Customer ordered equal quantities of both lighter and heavier products
    - Total number of boxes shipped is 240
-/
def max_truck_load : ℕ :=
  let total_boxes : ℕ := 240
  let num_trucks : ℕ := 3
  let light_box_weight : ℕ := 10
  let heavy_box_weight : ℕ := 40
  let boxes_per_type : ℕ := total_boxes / 2
  let total_weight : ℕ := boxes_per_type * light_box_weight + boxes_per_type * heavy_box_weight
  total_weight / num_trucks

theorem max_truck_load_is_2000 : max_truck_load = 2000 := by
  sorry

end NUMINAMATH_CALUDE_max_truck_load_is_2000_l975_97519


namespace NUMINAMATH_CALUDE_expression_evaluation_l975_97573

theorem expression_evaluation : (-2 : ℤ) ^ (4^2) + 1^(3^3) = 65537 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l975_97573


namespace NUMINAMATH_CALUDE_circle_inequality_l975_97583

theorem circle_inequality (a b c d : ℝ) (x₁ x₂ x₃ x₄ y₁ y₂ y₃ y₄ : ℝ) 
    (ha : a > 0) (hb : b > 0) (hc : c > 0) (hd : d > 0)
    (hab : a * b + c * d = 1)
    (h1 : x₁^2 + y₁^2 = 1) (h2 : x₂^2 + y₂^2 = 1) 
    (h3 : x₃^2 + y₃^2 = 1) (h4 : x₄^2 + y₄^2 = 1) : 
  (a * y₁ + b * y₂ + c * y₃ + d * y₄)^2 + (a * x₁ + b * x₃ + c * x₂ + d * x₁)^2 
    ≤ 2 * ((a^2 + b^2) / (a * b) + (c^2 + d^2) / (c * d)) := by
  sorry

end NUMINAMATH_CALUDE_circle_inequality_l975_97583


namespace NUMINAMATH_CALUDE_three_by_three_grid_paths_l975_97516

/-- The number of paths from (0,0) to (n,m) on a grid, moving only right or down -/
def grid_paths (n m : ℕ) : ℕ := Nat.choose (n + m) n

/-- Theorem: There are 20 distinct paths from the top-left to the bottom-right corner of a 3x3 grid -/
theorem three_by_three_grid_paths : grid_paths 3 3 = 20 := by sorry

end NUMINAMATH_CALUDE_three_by_three_grid_paths_l975_97516


namespace NUMINAMATH_CALUDE_sin_25pi_div_6_l975_97536

theorem sin_25pi_div_6 : Real.sin (25 * π / 6) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_sin_25pi_div_6_l975_97536


namespace NUMINAMATH_CALUDE_origin_and_slope_condition_vertical_tangent_condition_l975_97570

-- Define the function f
def f (a b x : ℝ) : ℝ := x^3 + (1 - a)*x^2 - a*(a + 2)*x + b

-- Define the derivative of f
def f' (a x : ℝ) : ℝ := 3*x^2 + 2*(1 - a)*x - a*(a + 2)

-- Theorem 1: If f(0) = 0 and f'(0) = -3, then (a = -3 or a = 1) and b = 0
theorem origin_and_slope_condition (a b : ℝ) :
  f a b 0 = 0 ∧ f' a 0 = -3 → (a = -3 ∨ a = 1) ∧ b = 0 := by sorry

-- Theorem 2: The curve y = f(x) has two vertical tangent lines iff a ∈ (-∞, -1/2) ∪ (-1/2, +∞)
theorem vertical_tangent_condition (a : ℝ) :
  (∃ x₁ x₂ : ℝ, x₁ ≠ x₂ ∧ f' a x₁ = 0 ∧ f' a x₂ = 0) ↔ 
  a < -1/2 ∨ a > -1/2 := by sorry

end NUMINAMATH_CALUDE_origin_and_slope_condition_vertical_tangent_condition_l975_97570


namespace NUMINAMATH_CALUDE_brandon_rabbit_catching_l975_97568

/-- The number of squirrels Brandon can catch in an hour -/
def squirrels_per_hour : ℕ := 6

/-- The number of calories in each squirrel -/
def calories_per_squirrel : ℕ := 300

/-- The number of calories in each rabbit -/
def calories_per_rabbit : ℕ := 800

/-- The additional calories Brandon gets from catching squirrels instead of rabbits -/
def additional_calories : ℕ := 200

/-- The number of rabbits Brandon can catch in an hour -/
def rabbits_per_hour : ℕ := 2

theorem brandon_rabbit_catching :
  squirrels_per_hour * calories_per_squirrel =
  rabbits_per_hour * calories_per_rabbit + additional_calories :=
by sorry

end NUMINAMATH_CALUDE_brandon_rabbit_catching_l975_97568


namespace NUMINAMATH_CALUDE_probability_sum_five_l975_97500

def dice_outcomes : ℕ := 6 * 6

def favorable_outcomes : ℕ := 4

theorem probability_sum_five (dice_outcomes : ℕ) (favorable_outcomes : ℕ) :
  dice_outcomes = 36 →
  favorable_outcomes = 4 →
  (favorable_outcomes : ℚ) / dice_outcomes = 1/9 := by
  sorry

end NUMINAMATH_CALUDE_probability_sum_five_l975_97500


namespace NUMINAMATH_CALUDE_correct_num_spiders_l975_97591

/-- The number of spiders introduced to control pests in a garden --/
def num_spiders : ℕ := 12

/-- The initial number of bugs in the garden --/
def initial_bugs : ℕ := 400

/-- The number of bugs each spider eats --/
def bugs_per_spider : ℕ := 7

/-- The fraction of bugs remaining after spraying --/
def spray_factor : ℚ := 4/5

/-- The number of bugs remaining after pest control measures --/
def remaining_bugs : ℕ := 236

/-- Theorem stating that the number of spiders introduced is correct --/
theorem correct_num_spiders :
  (initial_bugs : ℚ) * spray_factor - (num_spiders : ℚ) * bugs_per_spider = remaining_bugs := by
  sorry

end NUMINAMATH_CALUDE_correct_num_spiders_l975_97591


namespace NUMINAMATH_CALUDE_weight_2019_is_9_5_l975_97567

/-- The weight of a stick in kilograms -/
def stick_weight : ℝ := 0.5

/-- The number of sticks in each digit -/
def sticks_in_digit : Fin 10 → ℕ
  | 0 => 6
  | 1 => 2
  | 2 => 5
  | 9 => 6
  | _ => 0

/-- The weight of the number 2019 in kilograms -/
def weight_2019 : ℝ :=
  (sticks_in_digit 2 + sticks_in_digit 0 + sticks_in_digit 1 + sticks_in_digit 9) * stick_weight

/-- Theorem: The weight of the number 2019 is 9.5 kg -/
theorem weight_2019_is_9_5 : weight_2019 = 9.5 := by
  sorry

end NUMINAMATH_CALUDE_weight_2019_is_9_5_l975_97567


namespace NUMINAMATH_CALUDE_two_numbers_difference_l975_97559

theorem two_numbers_difference (x y : ℝ) 
  (sum_eq : x + y = 10) 
  (square_diff_eq : x^2 - y^2 = 40) : 
  |x - y| = 4 := by
sorry

end NUMINAMATH_CALUDE_two_numbers_difference_l975_97559


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l975_97529

-- Define the sets A and B
def A : Set ℝ := {x : ℝ | -1 ≤ 2*x + 1 ∧ 2*x + 1 ≤ 3}
def B : Set ℝ := {x : ℝ | x ≠ 0 ∧ (x - 3) / (2*x) ≤ 0}

-- State the theorem
theorem intersection_of_A_and_B : A ∩ B = {x : ℝ | 0 < x ∧ x ≤ 1} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l975_97529


namespace NUMINAMATH_CALUDE_line_passes_through_fixed_point_l975_97594

/-- The line equation passing through a fixed point -/
def line_equation (m x y : ℝ) : Prop :=
  (m - 2) * x - y + 3 * m + 2 = 0

/-- Theorem stating that the line always passes through the point (-3, 8) -/
theorem line_passes_through_fixed_point :
  ∀ m : ℝ, line_equation m (-3) 8 := by
  sorry

end NUMINAMATH_CALUDE_line_passes_through_fixed_point_l975_97594


namespace NUMINAMATH_CALUDE_seven_by_seven_dissection_l975_97526

theorem seven_by_seven_dissection :
  ∀ (a b : ℕ),
  (3 * a + 4 * b = 7 * 7) →
  (b = 1) := by
sorry

end NUMINAMATH_CALUDE_seven_by_seven_dissection_l975_97526


namespace NUMINAMATH_CALUDE_max_common_segment_length_theorem_l975_97551

/-- The maximum length of the common initial segment of two sequences with coprime periods -/
def max_common_segment_length (m n : ℕ) : ℕ :=
  m + n - 2

/-- Theorem stating that for two sequences with coprime periods m and n,
    the maximum length of their common initial segment is m + n - 2 -/
theorem max_common_segment_length_theorem (m n : ℕ) (h : Nat.Coprime m n) :
  max_common_segment_length m n = m + n - 2 := by
  sorry


end NUMINAMATH_CALUDE_max_common_segment_length_theorem_l975_97551


namespace NUMINAMATH_CALUDE_debby_vacation_pictures_l975_97580

/-- Calculates the number of remaining pictures after deletion -/
def remaining_pictures (zoo_pictures museum_pictures deleted_pictures : ℕ) : ℕ :=
  (zoo_pictures + museum_pictures) - deleted_pictures

/-- Theorem: The number of remaining pictures is correct for Debby's vacation -/
theorem debby_vacation_pictures : remaining_pictures 24 12 14 = 22 := by
  sorry

end NUMINAMATH_CALUDE_debby_vacation_pictures_l975_97580


namespace NUMINAMATH_CALUDE_intersection_distance_squared_is_396_8_l975_97535

/-- Two circles in a 2D plane -/
structure TwoCircles where
  /-- Center of the first circle -/
  center1 : ℝ × ℝ
  /-- Radius of the first circle -/
  radius1 : ℝ
  /-- Center of the second circle -/
  center2 : ℝ × ℝ
  /-- Radius of the second circle -/
  radius2 : ℝ

/-- The square of the distance between intersection points of two circles -/
def intersectionPointsDistanceSquared (circles : TwoCircles) : ℝ :=
  sorry

/-- Theorem stating that the square of the distance between intersection points
    of the given circles is 396.8 -/
theorem intersection_distance_squared_is_396_8 :
  let circles : TwoCircles := {
    center1 := (0, 0),
    radius1 := 5,
    center2 := (4, -2),
    radius2 := 3
  }
  intersectionPointsDistanceSquared circles = 396.8 := by
  sorry

end NUMINAMATH_CALUDE_intersection_distance_squared_is_396_8_l975_97535


namespace NUMINAMATH_CALUDE_f_properties_l975_97572

noncomputable def f (a : ℝ) (x : ℝ) : ℝ := a^x + (x - 2) / (x + 1)

theorem f_properties (a : ℝ) (h : a > 1) :
  (∀ x₁ x₂ : ℝ, -1 < x₁ → x₁ < x₂ → f a x₁ < f a x₂) ∧
  (¬ ∃ x : ℝ, x < 0 ∧ f a x = 0) := by
  sorry

end NUMINAMATH_CALUDE_f_properties_l975_97572


namespace NUMINAMATH_CALUDE_ellipse_max_product_l975_97585

theorem ellipse_max_product (x y : ℝ) (P : ℝ × ℝ) (F₁ F₂ : ℝ × ℝ) :
  (x^2 / 25 + y^2 / 9 = 1) →
  (P = (x, y)) →
  (F₁ ≠ F₂) →
  (∀ (x' y' : ℝ), x'^2 / 25 + y'^2 / 9 = 1 → 
    dist P F₁ + dist P F₂ = dist (x', y') F₁ + dist (x', y') F₂) →
  (∃ (M : ℝ), ∀ (x' y' : ℝ), x'^2 / 25 + y'^2 / 9 = 1 → 
    dist (x', y') F₁ * dist (x', y') F₂ ≤ M ∧ 
    ∃ (x'' y'' : ℝ), x''^2 / 25 + y''^2 / 9 = 1 ∧ 
      dist (x'', y'') F₁ * dist (x'', y'') F₂ = M) →
  M = 25 := by
sorry

end NUMINAMATH_CALUDE_ellipse_max_product_l975_97585


namespace NUMINAMATH_CALUDE_smallest_integer_square_triple_plus_100_l975_97579

theorem smallest_integer_square_triple_plus_100 : 
  ∃ (x : ℤ), x^2 = 3*x + 100 ∧ ∀ (y : ℤ), y^2 = 3*y + 100 → x ≤ y :=
by sorry

end NUMINAMATH_CALUDE_smallest_integer_square_triple_plus_100_l975_97579


namespace NUMINAMATH_CALUDE_expression_equivalence_l975_97578

theorem expression_equivalence (x y : ℝ) (h : x * y ≠ 0) :
  ((x^2 + 1) / x) * ((y^2 + 1) / y) + ((x^2 - 1) / y) * ((y^2 - 1) / x) = 2 * x * y + 2 / (x * y) :=
by sorry

end NUMINAMATH_CALUDE_expression_equivalence_l975_97578


namespace NUMINAMATH_CALUDE_intersection_equality_l975_97555

theorem intersection_equality (a : ℝ) : 
  let A : Set ℝ := {x | x^2 - 8*x + 15 = 0}
  let B : Set ℝ := {x | a*x - 1 = 0}
  (A ∩ B = B) → (a = 0 ∨ a = 1/3 ∨ a = 1/5) :=
by
  sorry

end NUMINAMATH_CALUDE_intersection_equality_l975_97555


namespace NUMINAMATH_CALUDE_max_value_problem_l975_97509

theorem max_value_problem (a b c d : Real) 
  (ha : 0 ≤ a ∧ a ≤ 1) (hb : 0 ≤ b ∧ b ≤ 1) (hc : 0 ≤ c ∧ c ≤ 1) (hd : 0 ≤ d ∧ d ≤ 1) :
  ∀ x, x = (a * b * c * d) ^ (1/4) + ((1 - a) * (1 - b) * (1 - c) * (1 - d)) ^ (1/2) → x ≤ 1 :=
by sorry

end NUMINAMATH_CALUDE_max_value_problem_l975_97509


namespace NUMINAMATH_CALUDE_road_repair_group_size_l975_97561

/-- The number of persons in the first group that can repair a road -/
def first_group_size : ℕ := 39

/-- The number of days the first group works -/
def first_group_days : ℕ := 12

/-- The number of hours per day the first group works -/
def first_group_hours : ℕ := 5

/-- The number of persons in the second group -/
def second_group_size : ℕ := 15

/-- The number of days the second group works -/
def second_group_days : ℕ := 26

/-- The number of hours per day the second group works -/
def second_group_hours : ℕ := 6

/-- The theorem stating that the first group size is 39 -/
theorem road_repair_group_size :
  first_group_size * first_group_days * first_group_hours =
  second_group_size * second_group_days * second_group_hours :=
by
  sorry

#check road_repair_group_size

end NUMINAMATH_CALUDE_road_repair_group_size_l975_97561


namespace NUMINAMATH_CALUDE_german_french_fraction_l975_97537

/-- Conference language distribution -/
structure ConferenceLanguages where
  total : ℝ
  english : ℝ
  french : ℝ
  german : ℝ
  english_french : ℝ
  english_german : ℝ
  french_german : ℝ
  all_three : ℝ

/-- Language distribution satisfies the given conditions -/
def ValidDistribution (c : ConferenceLanguages) : Prop :=
  c.english_french = (1/5) * c.english ∧
  c.english_german = (1/3) * c.english ∧
  c.english_french = (1/8) * c.french ∧
  c.french_german = (1/2) * c.french ∧
  c.english_german = (1/6) * c.german

/-- The fraction of German speakers who also speak French is 2/5 -/
theorem german_french_fraction (c : ConferenceLanguages) 
  (h : ValidDistribution c) : 
  c.french_german / c.german = 2/5 := by
  sorry

end NUMINAMATH_CALUDE_german_french_fraction_l975_97537


namespace NUMINAMATH_CALUDE_sqrt_equation_solution_l975_97553

theorem sqrt_equation_solution :
  ∃! z : ℚ, Real.sqrt (5 - 4 * z) = 7 :=
by
  -- The unique solution is z = -11
  use -11
  sorry

end NUMINAMATH_CALUDE_sqrt_equation_solution_l975_97553


namespace NUMINAMATH_CALUDE_x_values_l975_97511

theorem x_values (x : ℝ) : 
  ({1, 2} ∪ {x + 1, x^2 - 4*x + 6} : Set ℝ) = {1, 2, 3} → x = 2 ∨ x = 1 := by
  sorry

end NUMINAMATH_CALUDE_x_values_l975_97511


namespace NUMINAMATH_CALUDE_product_maximum_l975_97563

theorem product_maximum (s : ℝ) (hs : s > 0) :
  ∃ (x y : ℝ), x > 0 ∧ y > 0 ∧ x + y = s ∧
  ∀ (a b : ℝ), a > 0 → b > 0 → a + b = s → x * y ≥ a * b ∧
  x * y = s^2 / 4 :=
sorry

end NUMINAMATH_CALUDE_product_maximum_l975_97563


namespace NUMINAMATH_CALUDE_xyz_product_l975_97541

theorem xyz_product (x y z : ℝ) (hx : x > 0) (hy : y > 0) (hz : z > 0)
  (h1 : x + 1/y = 5) (h2 : y + 1/z = 2) (h3 : z + 1/x = 3) :
  x * y * z = 10 + 3 * Real.sqrt 11 := by
  sorry

end NUMINAMATH_CALUDE_xyz_product_l975_97541


namespace NUMINAMATH_CALUDE_rogers_money_l975_97506

theorem rogers_money (x : ℤ) : 
  x - 20 + 46 = 71 → x = 45 := by
sorry

end NUMINAMATH_CALUDE_rogers_money_l975_97506


namespace NUMINAMATH_CALUDE_parking_ticket_multiple_l975_97588

theorem parking_ticket_multiple (total_tickets : ℕ) (alan_tickets : ℕ) (marcy_tickets : ℕ) (m : ℕ) :
  total_tickets = 150 →
  alan_tickets = 26 →
  marcy_tickets = m * alan_tickets - 6 →
  total_tickets = alan_tickets + marcy_tickets →
  m = 5 := by
sorry

end NUMINAMATH_CALUDE_parking_ticket_multiple_l975_97588


namespace NUMINAMATH_CALUDE_binary_sum_equals_expected_l975_97596

def binary_to_nat (b : List Bool) : Nat :=
  b.foldl (fun acc x => 2 * acc + if x then 1 else 0) 0

def nat_to_binary (n : Nat) : List Bool :=
  if n = 0 then [false] else
  let rec to_binary_aux (m : Nat) : List Bool :=
    if m = 0 then [] else (m % 2 = 1) :: to_binary_aux (m / 2)
  to_binary_aux n

def b1 : List Bool := [false, true, true, false, true]  -- 10110₂
def b2 : List Bool := [false, true, true]               -- 110₂
def b3 : List Bool := [true]                            -- 1₂
def b4 : List Bool := [true, false, true]               -- 101₂

def expected_sum : List Bool := [false, false, false, false, true, true]  -- 110000₂

theorem binary_sum_equals_expected :
  nat_to_binary (binary_to_nat b1 + binary_to_nat b2 + binary_to_nat b3 + binary_to_nat b4) = expected_sum :=
by sorry

end NUMINAMATH_CALUDE_binary_sum_equals_expected_l975_97596


namespace NUMINAMATH_CALUDE_factorization_equality_l975_97556

theorem factorization_equality (a b : ℝ) : 2 * a^2 - 8 * b^2 = 2 * (a + 2*b) * (a - 2*b) := by
  sorry

end NUMINAMATH_CALUDE_factorization_equality_l975_97556


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_expression_l975_97534

theorem simplify_and_evaluate_expression (a : ℚ) : 
  a = -1/2 → a * (a^4 - a + 1) * (a - 2) = 59/32 := by sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_expression_l975_97534


namespace NUMINAMATH_CALUDE_equal_variance_sequence_properties_l975_97593

/-- Definition of an equal variance sequence -/
def is_equal_variance_sequence (a : ℕ+ → ℝ) (p : ℝ) :=
  ∀ n : ℕ+, a n ^ 2 - a (n + 1) ^ 2 = p

theorem equal_variance_sequence_properties
  (a : ℕ+ → ℝ) (p : ℝ) (h : is_equal_variance_sequence a p) :
  (∀ n : ℕ+, ∃ d : ℝ, a (n + 1) ^ 2 - a n ^ 2 = d) ∧
  is_equal_variance_sequence (fun n ↦ (-1) ^ (n : ℕ)) 0 ∧
  (∀ k : ℕ+, is_equal_variance_sequence (fun n ↦ a (k * n)) (k * p)) :=
by sorry

end NUMINAMATH_CALUDE_equal_variance_sequence_properties_l975_97593


namespace NUMINAMATH_CALUDE_triangle_sine_sum_inequality_l975_97504

theorem triangle_sine_sum_inequality (A B C : Real) : 
  A + B + C = Real.pi → 0 < A → 0 < B → 0 < C →
  Real.sin A + Real.sin B + Real.sin C ≤ (3 / 2) * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_sine_sum_inequality_l975_97504


namespace NUMINAMATH_CALUDE_max_value_quadratic_function_l975_97528

/-- Given a > 1, the maximum value of f(x) = -x^2 - 2ax + 1 on the interval [-1,1] is 2a -/
theorem max_value_quadratic_function (a : ℝ) (h : a > 1) :
  ∃ (max : ℝ), max = 2 * a ∧ ∀ x ∈ Set.Icc (-1) 1, -x^2 - 2*a*x + 1 ≤ max :=
sorry

end NUMINAMATH_CALUDE_max_value_quadratic_function_l975_97528


namespace NUMINAMATH_CALUDE_cos_x_plus_3y_eq_one_l975_97571

/-- Given x and y in [-π/6, π/6] and a ∈ ℝ satisfying the system of equations,
    prove that cos(x + 3y) = 1 -/
theorem cos_x_plus_3y_eq_one 
  (x y : ℝ) 
  (hx : x ∈ Set.Icc (-π/6) (π/6))
  (hy : y ∈ Set.Icc (-π/6) (π/6))
  (a : ℝ)
  (eq1 : x^3 + Real.sin x - 3*a = 0)
  (eq2 : 9*y^3 + (1/3) * Real.sin (3*y) + a = 0) :
  Real.cos (x + 3*y) = 1 := by sorry

end NUMINAMATH_CALUDE_cos_x_plus_3y_eq_one_l975_97571


namespace NUMINAMATH_CALUDE_sqrt_49_times_sqrt_25_l975_97545

theorem sqrt_49_times_sqrt_25 : Real.sqrt (49 * Real.sqrt 25) = 7 * Real.sqrt 5 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_49_times_sqrt_25_l975_97545


namespace NUMINAMATH_CALUDE_altitude_not_integer_l975_97582

/-- Represents a right triangle with integer sides -/
structure RightTriangle where
  a : ℕ  -- First leg
  b : ℕ  -- Second leg
  c : ℕ  -- Hypotenuse
  is_right : a^2 + b^2 = c^2  -- Pythagorean theorem

/-- The altitude to the hypotenuse in a right triangle -/
def altitude (t : RightTriangle) : ℚ :=
  (t.a * t.b : ℚ) / t.c

/-- Theorem: In a right triangle with pairwise coprime integer sides, 
    the altitude to the hypotenuse is not an integer -/
theorem altitude_not_integer (t : RightTriangle) 
  (h_coprime : Nat.gcd t.a t.b = 1 ∧ Nat.gcd t.b t.c = 1 ∧ Nat.gcd t.c t.a = 1) : 
  ¬ ∃ (n : ℕ), altitude t = n :=
sorry

end NUMINAMATH_CALUDE_altitude_not_integer_l975_97582


namespace NUMINAMATH_CALUDE_least_odd_prime_factor_of_2100_8_plus_1_l975_97539

theorem least_odd_prime_factor_of_2100_8_plus_1 :
  (Nat.minFac (2100^8 + 1)) = 193 := by
  sorry

end NUMINAMATH_CALUDE_least_odd_prime_factor_of_2100_8_plus_1_l975_97539


namespace NUMINAMATH_CALUDE_triangle_side_length_l975_97560

theorem triangle_side_length (a b c : ℝ) (A B C : ℝ) :
  -- Conditions
  (2 * b = a + c) →  -- arithmetic sequence
  (B = π / 6) →  -- 30° in radians
  (1 / 2 * a * c * Real.sin B = 3 / 2) →  -- area of triangle
  -- Conclusion
  b = Real.sqrt 3 + 1 := by sorry

end NUMINAMATH_CALUDE_triangle_side_length_l975_97560


namespace NUMINAMATH_CALUDE_closest_integer_to_two_plus_sqrt_fifteen_l975_97518

theorem closest_integer_to_two_plus_sqrt_fifteen :
  ∀ n : ℤ, n ≠ 6 → |6 - (2 + Real.sqrt 15)| < |n - (2 + Real.sqrt 15)| := by
  sorry

end NUMINAMATH_CALUDE_closest_integer_to_two_plus_sqrt_fifteen_l975_97518


namespace NUMINAMATH_CALUDE_distance_against_current_equals_swimming_speed_l975_97521

/-- The distance swam against the current given swimming speed in still water and current speed -/
def distanceAgainstCurrent (swimmingSpeed currentSpeed : ℝ) : ℝ :=
  swimmingSpeed

theorem distance_against_current_equals_swimming_speed
  (swimmingSpeed currentSpeed : ℝ)
  (h1 : swimmingSpeed = 12)
  (h2 : currentSpeed = 10)
  (h3 : swimmingSpeed > currentSpeed) :
  distanceAgainstCurrent swimmingSpeed currentSpeed = 12 := by
  sorry

#eval distanceAgainstCurrent 12 10

end NUMINAMATH_CALUDE_distance_against_current_equals_swimming_speed_l975_97521


namespace NUMINAMATH_CALUDE_missing_number_proof_l975_97569

theorem missing_number_proof (x : ℝ) : 11 + Real.sqrt (-4 + x * 4 / 3) = 13 ↔ x = 6 := by
  sorry

end NUMINAMATH_CALUDE_missing_number_proof_l975_97569


namespace NUMINAMATH_CALUDE_fraction_simplification_fraction_value_at_one_l975_97546

theorem fraction_simplification (x : ℤ) (h1 : -2 < x) (h2 : x < 2) (h3 : x ≠ 0) :
  (((x^2 - 1) / (x^2 + 2*x + 1)) / ((1 / (x + 1)) - 1)) = -(x - 1) / x :=
sorry

theorem fraction_value_at_one :
  (((1^2 - 1) / (1^2 + 2*1 + 1)) / ((1 / (1 + 1)) - 1)) = 0 :=
sorry

end NUMINAMATH_CALUDE_fraction_simplification_fraction_value_at_one_l975_97546


namespace NUMINAMATH_CALUDE_min_value_sum_of_squares_l975_97592

theorem min_value_sum_of_squares (x y z : ℝ) 
  (hx : x > 0) (hy : y > 0) (hz : z > 0) 
  (hsum : x + y + z = 9) : 
  (x^2 + y^2)/(x + y) + (x^2 + z^2)/(x + z) + (y^2 + z^2)/(y + z) ≥ 9 ∧ 
  ∃ (a b c : ℝ), a > 0 ∧ b > 0 ∧ c > 0 ∧ a + b + c = 9 ∧
  (a^2 + b^2)/(a + b) + (a^2 + c^2)/(a + c) + (b^2 + c^2)/(b + c) = 9 :=
sorry

end NUMINAMATH_CALUDE_min_value_sum_of_squares_l975_97592


namespace NUMINAMATH_CALUDE_peanut_butter_jars_l975_97514

/-- Given the total amount of peanut butter and jar sizes, calculate the number of jars. -/
def number_of_jars (total_ounces : ℕ) (jar_sizes : List ℕ) : ℕ :=
  if jar_sizes.length = 0 then 0
  else
    let jars_per_size := total_ounces / (jar_sizes.sum)
    jars_per_size * jar_sizes.length

/-- Theorem stating that given 252 ounces of peanut butter in equal numbers of 16, 28, and 40 ounce jars, the total number of jars is 9. -/
theorem peanut_butter_jars :
  number_of_jars 252 [16, 28, 40] = 9 := by
  sorry

end NUMINAMATH_CALUDE_peanut_butter_jars_l975_97514


namespace NUMINAMATH_CALUDE_parallel_vectors_m_value_l975_97508

/-- Two vectors are parallel if their components are proportional -/
def are_parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 = a.2 * b.1

theorem parallel_vectors_m_value :
  let a : ℝ × ℝ := (1, -2)
  let b : ℝ × ℝ := (-1, m)
  are_parallel a b → m = 2 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_m_value_l975_97508


namespace NUMINAMATH_CALUDE_division_inequality_l975_97542

theorem division_inequality (a b : ℝ) (h : a > b) : a / 3 > b / 3 := by
  sorry

end NUMINAMATH_CALUDE_division_inequality_l975_97542


namespace NUMINAMATH_CALUDE_point_C_range_l975_97501

def parabola (x y : ℝ) : Prop := y^2 = x + 4

def perpendicular (x1 y1 x2 y2 x3 y3 : ℝ) : Prop :=
  (y2 - y1) * (y3 - y2) = -(x3 - x2) * (x2 - x1)

theorem point_C_range :
  ∀ y : ℝ,
  (∃ y1 : ℝ,
    parabola (y1^2 - 4) y1 ∧
    parabola (y^2 - 4) y ∧
    perpendicular 0 2 (y1^2 - 4) y1 (y^2 - 4) y) →
  y ≤ 0 ∨ y ≥ 4 := by
sorry

end NUMINAMATH_CALUDE_point_C_range_l975_97501


namespace NUMINAMATH_CALUDE_equation_solution_l975_97554

theorem equation_solution : ∃ b : ℝ, ∀ a : ℝ, (-6) * a^2 = 3 * (4*a + b) ∧ (a = 1 → b = -6) := by
  sorry

end NUMINAMATH_CALUDE_equation_solution_l975_97554


namespace NUMINAMATH_CALUDE_train_platform_crossing_time_l975_97523

/-- Given a train of length 1200 meters that takes 120 seconds to pass a point,
    the time required for this train to completely pass a platform of length 700 meters is 190 seconds. -/
theorem train_platform_crossing_time
  (train_length : ℝ)
  (point_crossing_time : ℝ)
  (platform_length : ℝ)
  (h1 : train_length = 1200)
  (h2 : point_crossing_time = 120)
  (h3 : platform_length = 700) :
  (train_length + platform_length) / (train_length / point_crossing_time) = 190 :=
sorry

end NUMINAMATH_CALUDE_train_platform_crossing_time_l975_97523


namespace NUMINAMATH_CALUDE_inequality_system_solutions_l975_97597

theorem inequality_system_solutions :
  let S := {x : ℤ | x ≥ 0 ∧ x - 3 * (x - 1) ≥ 1 ∧ (1 + 3 * x) / 2 > x - 1}
  S = {0, 1} := by sorry

end NUMINAMATH_CALUDE_inequality_system_solutions_l975_97597


namespace NUMINAMATH_CALUDE_fourth_power_complex_equality_l975_97595

theorem fourth_power_complex_equality (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  (Complex.mk a b)^4 = (Complex.mk a (-b))^4 → b / a = 1 := by
  sorry

end NUMINAMATH_CALUDE_fourth_power_complex_equality_l975_97595


namespace NUMINAMATH_CALUDE_equation_implies_difference_l975_97566

theorem equation_implies_difference (m n : ℝ) :
  (∀ x : ℝ, (x - m) * (x + 2) = x^2 + n*x - 8) →
  m - n = 6 := by
  sorry

end NUMINAMATH_CALUDE_equation_implies_difference_l975_97566


namespace NUMINAMATH_CALUDE_obtuse_triangle_count_and_largest_perimeter_l975_97525

/-- Represents a triangle with side lengths in arithmetic progression --/
structure ArithmeticTriangle where
  a : ℕ  -- middle length
  d : ℕ  -- common difference

/-- Checks if the triangle is obtuse --/
def ArithmeticTriangle.isObtuse (t : ArithmeticTriangle) : Prop :=
  (t.a - t.d)^2 + t.a^2 < (t.a + t.d)^2

/-- Checks if the triangle satisfies the given conditions --/
def ArithmeticTriangle.isValid (t : ArithmeticTriangle) : Prop :=
  t.d > 0 ∧ t.a > t.d ∧ t.a + t.d ≤ 50

/-- Counts the number of valid obtuse triangles --/
def countValidObtuseTriangles : ℕ := sorry

/-- Finds the triangle with the largest perimeter --/
def largestPerimeterTriangle : ArithmeticTriangle := sorry

theorem obtuse_triangle_count_and_largest_perimeter :
  countValidObtuseTriangles = 157 ∧
  let t := largestPerimeterTriangle
  t.a - t.d = 29 ∧ t.a = 39 ∧ t.a + t.d = 50 := by sorry

end NUMINAMATH_CALUDE_obtuse_triangle_count_and_largest_perimeter_l975_97525


namespace NUMINAMATH_CALUDE_estimate_value_l975_97544

theorem estimate_value : 
  3 < (Real.sqrt 3 + 3 * Real.sqrt 2) * Real.sqrt (1/3) ∧ 
  (Real.sqrt 3 + 3 * Real.sqrt 2) * Real.sqrt (1/3) < 4 := by
  sorry

end NUMINAMATH_CALUDE_estimate_value_l975_97544


namespace NUMINAMATH_CALUDE_quadratic_ratio_l975_97586

theorem quadratic_ratio (x : ℝ) : 
  ∃ (d e : ℝ), x^2 + 2600*x + 2600 = (x + d)^2 + e ∧ e / d = -1298 := by
sorry

end NUMINAMATH_CALUDE_quadratic_ratio_l975_97586


namespace NUMINAMATH_CALUDE_min_value_reciprocal_sum_l975_97522

theorem min_value_reciprocal_sum (a b : ℝ) (ha : a > 0) (hb : b > 0) (hab : a + b = 2) :
  1 / a + 4 / b ≥ 9 / 2 := by sorry

end NUMINAMATH_CALUDE_min_value_reciprocal_sum_l975_97522


namespace NUMINAMATH_CALUDE_combined_transformation_correct_l975_97557

/-- A dilation centered at the origin with scale factor k -/
def dilation (k : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  Matrix.diagonal (λ _ => k)

/-- A reflection across the x-axis -/
def reflectionX : Matrix (Fin 2) (Fin 2) ℝ :=
  Matrix.diagonal (λ i => if i = 0 then 1 else -1)

/-- The combined transformation matrix -/
def combinedTransformation : Matrix (Fin 2) (Fin 2) ℝ :=
  Matrix.diagonal (λ i => if i = 0 then 5 else -5)

theorem combined_transformation_correct :
  combinedTransformation = reflectionX * dilation 5 := by
  sorry


end NUMINAMATH_CALUDE_combined_transformation_correct_l975_97557


namespace NUMINAMATH_CALUDE_f_negative_three_value_l975_97552

/-- Given a function f(x) = a*sin(x) + b*tan(x) + x^3 + 1, 
    if f(3) = 7, then f(-3) = -5 -/
theorem f_negative_three_value 
  (f : ℝ → ℝ) 
  (a b : ℝ) 
  (h1 : ∀ x, f x = a * Real.sin x + b * Real.tan x + x^3 + 1) 
  (h2 : f 3 = 7) : 
  f (-3) = -5 := by sorry

end NUMINAMATH_CALUDE_f_negative_three_value_l975_97552


namespace NUMINAMATH_CALUDE_street_length_calculation_l975_97524

/-- Proves that the length of a street is 1800 meters, given that a person crosses it in 12 minutes at a speed of 9 km per hour. -/
theorem street_length_calculation (crossing_time : ℝ) (speed_kmh : ℝ) :
  crossing_time = 12 →
  speed_kmh = 9 →
  (speed_kmh * 1000 / 60) * crossing_time = 1800 := by
sorry

end NUMINAMATH_CALUDE_street_length_calculation_l975_97524


namespace NUMINAMATH_CALUDE_max_value_reciprocal_sum_l975_97575

theorem max_value_reciprocal_sum (x y a b : ℝ) 
  (ha : a > 1) (hb : b > 1) 
  (hax : a^x = 3) (hby : b^y = 3) 
  (hab : a + b = 2 * Real.sqrt 3) : 
  ∃ (max : ℝ), max = 1 ∧ ∀ (x' y' : ℝ), 
    (∃ (a' b' : ℝ), a' > 1 ∧ b' > 1 ∧ a'^x' = 3 ∧ b'^y' = 3 ∧ a' + b' = 2 * Real.sqrt 3) →
    1/x' + 1/y' ≤ max :=
by
  sorry

end NUMINAMATH_CALUDE_max_value_reciprocal_sum_l975_97575


namespace NUMINAMATH_CALUDE_principal_calculation_l975_97515

/-- Given a principal P and an interest rate R (as a percentage),
    if the amount after 2 years is 780 and after 7 years is 1020,
    then the principal P is 684. -/
theorem principal_calculation (P R : ℚ) 
  (h1 : P + (P * R * 2) / 100 = 780)
  (h2 : P + (P * R * 7) / 100 = 1020) : 
  P = 684 := by
sorry

end NUMINAMATH_CALUDE_principal_calculation_l975_97515


namespace NUMINAMATH_CALUDE_tadd_250th_number_l975_97505

/-- Represents the block size for a player in the n-th round -/
def blockSize (n : ℕ) : ℕ := 6 * n - 5

/-- Sum of numbers spoken up to the k-th block -/
def sumUpToBlock (k : ℕ) : ℕ := 3 * k * (k - 1)

/-- The counting game as described in the problem -/
def countingGame : Prop :=
  ∃ (k : ℕ),
    sumUpToBlock (k - 1) < 250 ∧
    250 ≤ sumUpToBlock k ∧
    250 = sumUpToBlock (k - 1) + (250 - sumUpToBlock (k - 1))

theorem tadd_250th_number :
  countingGame → (∃ (k : ℕ), 250 = sumUpToBlock (k - 1) + (250 - sumUpToBlock (k - 1))) :=
by sorry

end NUMINAMATH_CALUDE_tadd_250th_number_l975_97505


namespace NUMINAMATH_CALUDE_gcd_consecutive_b_terms_bound_l975_97533

def b (n : ℕ) : ℕ := (2 * n).factorial + n^2

theorem gcd_consecutive_b_terms_bound (n : ℕ) (h : n ≥ 1) :
  Nat.gcd (b n) (b (n + 1)) ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_consecutive_b_terms_bound_l975_97533


namespace NUMINAMATH_CALUDE_sequence_sum_bound_l975_97507

/-- Given a sequence of positive integers satisfying certain conditions, 
    prove that the sum of its first n terms is at most n². -/
theorem sequence_sum_bound (n : ℕ) (a : ℕ → ℕ) : n > 0 →
  (∀ i, a (i + n) = a i) →
  (∀ i ∈ Finset.range n, a i > 0) →
  (∀ i ∈ Finset.range (n - 1), a i ≤ a (i + 1)) →
  a n ≤ a 1 + n →
  (∀ i ∈ Finset.range n, a (a i) ≤ n + i) →
  (Finset.range n).sum a ≤ n^2 := by
  sorry


end NUMINAMATH_CALUDE_sequence_sum_bound_l975_97507


namespace NUMINAMATH_CALUDE_eight_factorial_equals_product_l975_97520

-- Define factorial function
def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

-- State the theorem
theorem eight_factorial_equals_product : 4 * 6 * 3 * 560 = factorial 8 := by
  sorry


end NUMINAMATH_CALUDE_eight_factorial_equals_product_l975_97520
