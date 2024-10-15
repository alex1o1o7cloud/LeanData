import Mathlib

namespace NUMINAMATH_CALUDE_max_product_sum_2020_l344_34419

theorem max_product_sum_2020 :
  (∃ (x y : ℤ), x + y = 2020 ∧ x * y = 1020100) ∧
  (∀ (a b : ℤ), a + b = 2020 → a * b ≤ 1020100) := by
  sorry

end NUMINAMATH_CALUDE_max_product_sum_2020_l344_34419


namespace NUMINAMATH_CALUDE_order_amount_for_88200_l344_34493

/-- Calculates the discount rate based on the order quantity -/
def discount_rate (x : ℕ) : ℚ :=
  if x < 250 then 0
  else if x < 500 then 1/20
  else if x < 1000 then 1/10
  else 3/20

/-- Calculates the payable amount given the order quantity and unit price -/
def payable_amount (x : ℕ) (A : ℚ) : ℚ :=
  A * x * (1 - discount_rate x)

/-- The unit price determined from the given condition -/
def unit_price : ℚ := 100

theorem order_amount_for_88200 :
  payable_amount 980 unit_price = 88200 :=
sorry

end NUMINAMATH_CALUDE_order_amount_for_88200_l344_34493


namespace NUMINAMATH_CALUDE_weight_of_water_moles_l344_34415

/-- The atomic weight of hydrogen in g/mol -/
def atomic_weight_H : ℝ := 1.008

/-- The atomic weight of oxygen in g/mol -/
def atomic_weight_O : ℝ := 16.00

/-- The number of hydrogen atoms in a water molecule -/
def H_count : ℕ := 2

/-- The number of oxygen atoms in a water molecule -/
def O_count : ℕ := 1

/-- The number of moles of water -/
def moles_of_water : ℝ := 4

/-- The molecular weight of water (H2O) in g/mol -/
def molecular_weight_H2O : ℝ := H_count * atomic_weight_H + O_count * atomic_weight_O

theorem weight_of_water_moles : 
  moles_of_water * molecular_weight_H2O = 72.064 := by sorry

end NUMINAMATH_CALUDE_weight_of_water_moles_l344_34415


namespace NUMINAMATH_CALUDE_prove_present_age_of_B_l344_34436

/-- The present age of person B given the conditions:
    1. In 10 years, A will be twice as old as B was 10 years ago
    2. A is now 9 years older than B -/
def present_age_of_B (a b : ℕ) : Prop :=
  (a + 10 = 2 * (b - 10)) ∧ (a = b + 9) → b = 39

theorem prove_present_age_of_B :
  ∀ (a b : ℕ), present_age_of_B a b :=
by
  sorry

end NUMINAMATH_CALUDE_prove_present_age_of_B_l344_34436


namespace NUMINAMATH_CALUDE_inscribed_sphere_radius_l344_34423

/-- Represents a conical flask with an inscribed sphere -/
structure ConicalFlask where
  base_radius : ℝ
  height : ℝ
  liquid_height : ℝ
  sphere_radius : ℝ

/-- Checks if the sphere is properly inscribed in the flask -/
def is_properly_inscribed (flask : ConicalFlask) : Prop :=
  flask.sphere_radius > 0 ∧
  flask.sphere_radius ≤ flask.base_radius ∧
  flask.sphere_radius + flask.liquid_height ≤ flask.height

/-- The main theorem about the inscribed sphere's radius -/
theorem inscribed_sphere_radius 
  (flask : ConicalFlask)
  (h_base : flask.base_radius = 15)
  (h_height : flask.height = 30)
  (h_liquid : flask.liquid_height = 10)
  (h_inscribed : is_properly_inscribed flask) :
  flask.sphere_radius = 10 :=
sorry

end NUMINAMATH_CALUDE_inscribed_sphere_radius_l344_34423


namespace NUMINAMATH_CALUDE_vampire_conversion_theorem_l344_34487

/-- The number of people each vampire turns into vampires per night. -/
def vampire_conversion_rate : ℕ → Prop := λ x =>
  let initial_population : ℕ := 300
  let initial_vampires : ℕ := 2
  let nights : ℕ := 2
  let final_vampires : ℕ := 72
  
  -- After first night: initial_vampires + (initial_vampires * x)
  -- After second night: (initial_vampires + (initial_vampires * x)) + 
  --                     (initial_vampires + (initial_vampires * x)) * x
  
  (initial_vampires + (initial_vampires * x)) + 
  (initial_vampires + (initial_vampires * x)) * x = final_vampires

theorem vampire_conversion_theorem : vampire_conversion_rate 5 := by
  sorry

end NUMINAMATH_CALUDE_vampire_conversion_theorem_l344_34487


namespace NUMINAMATH_CALUDE_at_least_one_non_negative_l344_34424

theorem at_least_one_non_negative 
  (a b c d e f g h : ℝ) : 
  (max (ac + bd) (max (ae + bf) (max (ag + bh) (max (ce + df) (max (cg + dh) (eg + fh)))))) ≥ 0 :=
by sorry

end NUMINAMATH_CALUDE_at_least_one_non_negative_l344_34424


namespace NUMINAMATH_CALUDE_square_sum_representation_l344_34414

theorem square_sum_representation : ∃ (a b c : ℕ), 
  15129 = a^2 + b^2 + c^2 ∧ 
  a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
  (a ≠ 27 ∨ b ≠ 72 ∨ c ≠ 96) ∧
  ∃ (d e f g h i : ℕ), 
    378225 = d^2 + e^2 + f^2 + g^2 + h^2 + i^2 ∧
    d ≠ e ∧ d ≠ f ∧ d ≠ g ∧ d ≠ h ∧ d ≠ i ∧
    e ≠ f ∧ e ≠ g ∧ e ≠ h ∧ e ≠ i ∧
    f ≠ g ∧ f ≠ h ∧ f ≠ i ∧
    g ≠ h ∧ g ≠ i ∧
    h ≠ i := by
  sorry

end NUMINAMATH_CALUDE_square_sum_representation_l344_34414


namespace NUMINAMATH_CALUDE_diamond_olivine_difference_l344_34491

theorem diamond_olivine_difference (agate olivine diamond : ℕ) : 
  agate = 30 →
  olivine = agate + 5 →
  diamond > olivine →
  agate + olivine + diamond = 111 →
  diamond - olivine = 11 :=
by sorry

end NUMINAMATH_CALUDE_diamond_olivine_difference_l344_34491


namespace NUMINAMATH_CALUDE_town_population_l344_34433

theorem town_population (present_population : ℝ) 
  (growth_rate : ℝ) (future_population : ℝ) : 
  growth_rate = 0.1 → 
  future_population = present_population * (1 + growth_rate) → 
  future_population = 220 → 
  present_population = 200 := by
sorry

end NUMINAMATH_CALUDE_town_population_l344_34433


namespace NUMINAMATH_CALUDE_x_value_l344_34494

theorem x_value : ∃ x : ℚ, (3 * x) / 7 = 15 ∧ x = 35 := by
  sorry

end NUMINAMATH_CALUDE_x_value_l344_34494


namespace NUMINAMATH_CALUDE_x_equals_y_squared_plus_two_y_minus_one_l344_34480

theorem x_equals_y_squared_plus_two_y_minus_one (x y : ℝ) :
  x / (x - 1) = (y^2 + 2*y - 1) / (y^2 + 2*y - 2) → x = y^2 + 2*y - 1 := by
  sorry

end NUMINAMATH_CALUDE_x_equals_y_squared_plus_two_y_minus_one_l344_34480


namespace NUMINAMATH_CALUDE_jungkook_boxes_l344_34420

/-- The number of boxes needed to hold a given number of balls -/
def boxes_needed (total_balls : ℕ) (balls_per_box : ℕ) : ℕ :=
  (total_balls + balls_per_box - 1) / balls_per_box

theorem jungkook_boxes (total_balls : ℕ) (balls_per_box : ℕ) 
  (h1 : total_balls = 10) (h2 : balls_per_box = 5) : 
  boxes_needed total_balls balls_per_box = 2 := by
sorry

end NUMINAMATH_CALUDE_jungkook_boxes_l344_34420


namespace NUMINAMATH_CALUDE_factorial_division_l344_34429

-- Define factorial function
def factorial (n : ℕ) : ℕ :=
  match n with
  | 0 => 1
  | n + 1 => (n + 1) * factorial n

-- Theorem statement
theorem factorial_division : 
  factorial 8 / factorial (8 - 2) = 56 := by
  sorry

end NUMINAMATH_CALUDE_factorial_division_l344_34429


namespace NUMINAMATH_CALUDE_cube_root_of_product_l344_34402

theorem cube_root_of_product (a b c : ℕ) :
  (2^9 * 5^3 * 7^6 : ℝ)^(1/3) = 1960 := by
  sorry

end NUMINAMATH_CALUDE_cube_root_of_product_l344_34402


namespace NUMINAMATH_CALUDE_john_nada_money_multiple_l344_34484

/-- Given the money distribution among Ali, Nada, and John, prove that John has 4 times Nada's amount. -/
theorem john_nada_money_multiple (total : ℕ) (john_money : ℕ) (nada_money : ℕ) :
  total = 67 →
  john_money = 48 →
  nada_money + (nada_money - 5) + john_money = total →
  john_money = 4 * nada_money :=
by sorry

end NUMINAMATH_CALUDE_john_nada_money_multiple_l344_34484


namespace NUMINAMATH_CALUDE_profit_maximized_at_optimal_reduction_optimal_reduction_is_five_profit_function_correct_l344_34448

/-- Profit function for a product with given initial conditions -/
def profit_function (x : ℝ) : ℝ := -x^2 + 10*x + 600

/-- The price reduction that maximizes profit -/
def optimal_reduction : ℝ := 5

theorem profit_maximized_at_optimal_reduction :
  ∀ x : ℝ, profit_function x ≤ profit_function optimal_reduction :=
sorry

theorem optimal_reduction_is_five :
  optimal_reduction = 5 :=
sorry

theorem profit_function_correct (x : ℝ) :
  profit_function x = (100 - 70 - x) * (20 + x) :=
sorry

end NUMINAMATH_CALUDE_profit_maximized_at_optimal_reduction_optimal_reduction_is_five_profit_function_correct_l344_34448


namespace NUMINAMATH_CALUDE_intersection_point_l₁_l₂_l344_34479

/-- The intersection point of two lines in 2D space -/
structure IntersectionPoint (l₁ l₂ : ℝ → ℝ → Prop) where
  x : ℝ
  y : ℝ
  on_l₁ : l₁ x y
  on_l₂ : l₂ x y
  unique : ∀ x' y', l₁ x' y' → l₂ x' y' → x' = x ∧ y' = y

/-- Line l₁: 2x - y - 10 = 0 -/
def l₁ (x y : ℝ) : Prop := 2 * x - y - 10 = 0

/-- Line l₂: 3x + 4y - 4 = 0 -/
def l₂ (x y : ℝ) : Prop := 3 * x + 4 * y - 4 = 0

/-- The intersection point of l₁ and l₂ is (4, -2) -/
theorem intersection_point_l₁_l₂ : IntersectionPoint l₁ l₂ where
  x := 4
  y := -2
  on_l₁ := by sorry
  on_l₂ := by sorry
  unique := by sorry

end NUMINAMATH_CALUDE_intersection_point_l₁_l₂_l344_34479


namespace NUMINAMATH_CALUDE_x_plus_y_equals_9_l344_34456

theorem x_plus_y_equals_9 (x y m : ℝ) (h1 : x + m = 4) (h2 : y - 5 = m) : x + y = 9 := by
  sorry

end NUMINAMATH_CALUDE_x_plus_y_equals_9_l344_34456


namespace NUMINAMATH_CALUDE_clothing_combinations_l344_34422

theorem clothing_combinations (hoodies sweatshirts jeans slacks : ℕ) 
  (h_hoodies : hoodies = 5)
  (h_sweatshirts : sweatshirts = 4)
  (h_jeans : jeans = 3)
  (h_slacks : slacks = 5) :
  (hoodies + sweatshirts) * (jeans + slacks) = 72 := by
  sorry

end NUMINAMATH_CALUDE_clothing_combinations_l344_34422


namespace NUMINAMATH_CALUDE_unknown_number_exists_l344_34478

theorem unknown_number_exists : ∃ x : ℝ, 
  (0.15 : ℝ)^3 - (0.06 : ℝ)^3 / (0.15 : ℝ)^2 + x + (0.06 : ℝ)^2 = 0.08999999999999998 ∧ 
  abs (x - 0.092625) < 0.000001 := by
  sorry

end NUMINAMATH_CALUDE_unknown_number_exists_l344_34478


namespace NUMINAMATH_CALUDE_range_of_m_l344_34440

def f (a x : ℝ) : ℝ := x^2 - 2*a*x + 1

def set_A : Set ℝ := {a | -1 < a ∧ a < 1}

def set_B (m : ℝ) : Set ℝ := {a | m < a ∧ a < m + 3}

theorem range_of_m (m : ℝ) :
  (∀ x, x ∈ set_A → x ∈ set_B m) ∧
  (∃ x, x ∈ set_B m ∧ x ∉ set_A) →
  -2 ≤ m ∧ m ≤ -1 :=
sorry

end NUMINAMATH_CALUDE_range_of_m_l344_34440


namespace NUMINAMATH_CALUDE_six_eight_x_ten_y_l344_34421

theorem six_eight_x_ten_y (x y Q : ℝ) (h : 3 * (4 * x + 5 * y) = Q) : 
  6 * (8 * x + 10 * y) = 4 * Q := by
  sorry

end NUMINAMATH_CALUDE_six_eight_x_ten_y_l344_34421


namespace NUMINAMATH_CALUDE_equation_solution_l344_34437

theorem equation_solution : 
  ∃! x : ℚ, (x^2 + 3*x + 4) / (x + 5) = x + 6 :=
by
  use -13/4
  sorry

end NUMINAMATH_CALUDE_equation_solution_l344_34437


namespace NUMINAMATH_CALUDE_triangle_equilateral_condition_l344_34499

open Real

theorem triangle_equilateral_condition (a b c A B C : ℝ) :
  0 < a ∧ 0 < b ∧ 0 < c →
  0 < A ∧ 0 < B ∧ 0 < C →
  A + B + C = π →
  a / sin A = b / sin B →
  a / sin A = c / sin C →
  a / cos A = b / cos B →
  a / cos A = c / cos C →
  A = B ∧ B = C :=
by sorry

end NUMINAMATH_CALUDE_triangle_equilateral_condition_l344_34499


namespace NUMINAMATH_CALUDE_base_five_of_232_l344_34461

def base_five_repr (n : ℕ) : List ℕ :=
  if n = 0 then [0]
  else
    let rec aux (m : ℕ) (acc : List ℕ) : List ℕ :=
      if m = 0 then acc
      else aux (m / 5) ((m % 5) :: acc)
    aux n []

theorem base_five_of_232 :
  base_five_repr 232 = [1, 4, 1, 2] := by
sorry

end NUMINAMATH_CALUDE_base_five_of_232_l344_34461


namespace NUMINAMATH_CALUDE_dealer_purchase_problem_l344_34470

theorem dealer_purchase_problem (total_cost : ℚ) (selling_price : ℚ) (num_sold : ℕ) (profit_percentage : ℚ) :
  total_cost = 25 →
  selling_price = 32 →
  num_sold = 12 →
  profit_percentage = 60 →
  (∃ (num_purchased : ℕ), 
    num_purchased * (selling_price / num_sold) = total_cost * (1 + profit_percentage / 100) ∧
    num_purchased = 15) :=
by sorry

end NUMINAMATH_CALUDE_dealer_purchase_problem_l344_34470


namespace NUMINAMATH_CALUDE_min_value_of_f_l344_34431

/-- The quadratic function f(x) = 2x^2 + 8x + 7 -/
def f (x : ℝ) : ℝ := 2 * x^2 + 8 * x + 7

/-- Theorem: The minimum value of f(x) = 2x^2 + 8x + 7 is -1 -/
theorem min_value_of_f :
  ∃ (min : ℝ), min = -1 ∧ ∀ (x : ℝ), f x ≥ min :=
by sorry

end NUMINAMATH_CALUDE_min_value_of_f_l344_34431


namespace NUMINAMATH_CALUDE_shelleys_weight_l344_34462

/-- Given the weights of three people on a scale in pairs, find one person's weight -/
theorem shelleys_weight (p s r : ℕ) : 
  p + s = 151 → s + r = 132 → p + r = 115 → s = 84 := by
  sorry

end NUMINAMATH_CALUDE_shelleys_weight_l344_34462


namespace NUMINAMATH_CALUDE_paving_cost_l344_34446

/-- The cost of paving a rectangular floor -/
theorem paving_cost (length width rate : ℝ) (h1 : length = 5.5) (h2 : width = 4) (h3 : rate = 750) :
  length * width * rate = 16500 := by
  sorry

end NUMINAMATH_CALUDE_paving_cost_l344_34446


namespace NUMINAMATH_CALUDE_sum_three_consecutive_integers_divisible_by_three_l344_34472

theorem sum_three_consecutive_integers_divisible_by_three (a : ℕ) (h : a > 1) :
  ∃ k : ℤ, (a - 1 : ℤ) + a + (a + 1) = 3 * k :=
by sorry

end NUMINAMATH_CALUDE_sum_three_consecutive_integers_divisible_by_three_l344_34472


namespace NUMINAMATH_CALUDE_relay_race_arrangements_l344_34473

/-- The number of students in the class --/
def total_students : ℕ := 6

/-- The number of students needed for the relay race --/
def relay_team_size : ℕ := 4

/-- The possible positions for student A --/
inductive PositionA
| first
| second

/-- The possible positions for student B --/
inductive PositionB
| second
| fourth

/-- A function to calculate the number of arrangements --/
def count_arrangements (total : ℕ) (team_size : ℕ) : ℕ :=
  sorry

/-- The theorem stating that the number of arrangements is 36 --/
theorem relay_race_arrangements :
  count_arrangements total_students relay_team_size = 36 :=
sorry

end NUMINAMATH_CALUDE_relay_race_arrangements_l344_34473


namespace NUMINAMATH_CALUDE_absolute_value_equals_sqrt_of_square_l344_34404

theorem absolute_value_equals_sqrt_of_square (x : ℝ) : |x| = Real.sqrt (x^2) := by
  sorry

end NUMINAMATH_CALUDE_absolute_value_equals_sqrt_of_square_l344_34404


namespace NUMINAMATH_CALUDE_yellow_to_red_ratio_l344_34488

/-- Represents the number of chairs of each color in Rodrigo's classroom --/
structure ChairCounts where
  red : ℕ
  yellow : ℕ
  blue : ℕ

/-- Represents the state of chairs in Rodrigo's classroom --/
def classroom_state : ChairCounts → Prop
  | ⟨red, yellow, blue⟩ => 
    red = 4 ∧ 
    blue = yellow - 2 ∧ 
    red + yellow + blue = 18 ∧ 
    red + yellow + blue - 3 = 15

/-- The theorem stating the ratio of yellow to red chairs --/
theorem yellow_to_red_ratio (chairs : ChairCounts) :
  classroom_state chairs → chairs.yellow / chairs.red = 2 := by
  sorry

#check yellow_to_red_ratio

end NUMINAMATH_CALUDE_yellow_to_red_ratio_l344_34488


namespace NUMINAMATH_CALUDE_workshop_workers_count_l344_34469

theorem workshop_workers_count :
  let total_average : ℝ := 8000
  let technician_count : ℕ := 7
  let technician_average : ℝ := 14000
  let non_technician_average : ℝ := 6000
  ∃ (total_workers : ℕ) (non_technician_workers : ℕ),
    total_workers = technician_count + non_technician_workers ∧
    total_average * (technician_count + non_technician_workers : ℝ) =
      technician_average * technician_count + non_technician_average * non_technician_workers ∧
    total_workers = 28 :=
by
  sorry

#check workshop_workers_count

end NUMINAMATH_CALUDE_workshop_workers_count_l344_34469


namespace NUMINAMATH_CALUDE_a_33_mod_77_l344_34413

/-- Defines a_n as the large integer formed by concatenating integers from 1 to n -/
def a (n : ℕ) : ℕ :=
  -- Definition of a_n goes here
  sorry

/-- The remainder when a_33 is divided by 77 is 22 -/
theorem a_33_mod_77 : a 33 % 77 = 22 := by
  sorry

end NUMINAMATH_CALUDE_a_33_mod_77_l344_34413


namespace NUMINAMATH_CALUDE_contrapositive_equivalence_l344_34453

theorem contrapositive_equivalence :
  (∀ x : ℝ, x^2 ≤ 1 → -1 ≤ x ∧ x ≤ 1) ↔
  (∀ x : ℝ, (x < -1 ∨ x > 1) → x^2 > 1) :=
by sorry

end NUMINAMATH_CALUDE_contrapositive_equivalence_l344_34453


namespace NUMINAMATH_CALUDE_largest_six_digit_divisible_by_41_l344_34430

theorem largest_six_digit_divisible_by_41 : 
  ∀ n : ℕ, n ≤ 999999 ∧ n % 41 = 0 → n ≤ 999990 :=
by sorry

end NUMINAMATH_CALUDE_largest_six_digit_divisible_by_41_l344_34430


namespace NUMINAMATH_CALUDE_fibonacci_5k_divisible_by_5_l344_34442

def fibonacci : ℕ → ℕ
  | 0 => 1
  | 1 => 1
  | (n + 2) => fibonacci n + fibonacci (n + 1)

theorem fibonacci_5k_divisible_by_5 (k : ℕ) : ∃ m : ℕ, fibonacci (5 * k) = 5 * m := by
  sorry

end NUMINAMATH_CALUDE_fibonacci_5k_divisible_by_5_l344_34442


namespace NUMINAMATH_CALUDE_consecutive_integers_average_l344_34400

theorem consecutive_integers_average (a b : ℤ) : 
  (a > 0) → 
  (b = (7 * a + 21) / 7) → 
  ((7 * b + 21) / 7 = a + 6) := by
sorry

end NUMINAMATH_CALUDE_consecutive_integers_average_l344_34400


namespace NUMINAMATH_CALUDE_quadratic_roots_sum_of_squares_l344_34452

theorem quadratic_roots_sum_of_squares (p q : ℝ) (r s : ℝ) : 
  (2 * r^2 - p * r + q = 0) → 
  (2 * s^2 - p * s + q = 0) → 
  (r^2 + s^2 = p^2 / 4 - q) := by
sorry

end NUMINAMATH_CALUDE_quadratic_roots_sum_of_squares_l344_34452


namespace NUMINAMATH_CALUDE_sum_of_x_and_y_equals_two_l344_34441

theorem sum_of_x_and_y_equals_two (x y : ℝ) (h : x^2 + y^2 = 8*x - 4*y - 20) : x + y = 2 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_x_and_y_equals_two_l344_34441


namespace NUMINAMATH_CALUDE_fourth_sample_is_nineteen_l344_34426

/-- Represents a systematic sampling scenario in a class. -/
structure SystematicSample where
  total_students : ℕ
  sample_size : ℕ
  known_samples : List ℕ

/-- Calculates the interval for systematic sampling. -/
def sampling_interval (s : SystematicSample) : ℕ :=
  s.total_students / s.sample_size

/-- Checks if a number is part of the systematic sample. -/
def is_in_sample (s : SystematicSample) (n : ℕ) : Prop :=
  ∃ k, n = k * sampling_interval s + (s.known_samples.head? ).getD 0

/-- The main theorem stating that 19 must be the fourth sample in the given scenario. -/
theorem fourth_sample_is_nineteen (s : SystematicSample)
    (h1 : s.total_students = 56)
    (h2 : s.sample_size = 4)
    (h3 : s.known_samples = [5, 33, 47])
    (h4 : ∀ n, is_in_sample s n → n ∈ [5, 19, 33, 47]) :
    is_in_sample s 19 :=
  sorry

#check fourth_sample_is_nineteen

end NUMINAMATH_CALUDE_fourth_sample_is_nineteen_l344_34426


namespace NUMINAMATH_CALUDE_f_minimum_value_l344_34474

noncomputable def f (x : ℝ) : ℝ := (1 + 4 * x) / Real.sqrt x

theorem f_minimum_value (x : ℝ) (hx : x > 0) : 
  f x ≥ 4 ∧ ∃ x₀ > 0, f x₀ = 4 :=
sorry

end NUMINAMATH_CALUDE_f_minimum_value_l344_34474


namespace NUMINAMATH_CALUDE_parabola_tangent_and_intersecting_line_l344_34464

-- Define the parabola
def parabola (x y : ℝ) : Prop := y^2 = 4*x

-- Define the tangent line passing through (-1, 0)
def tangent_line (x y : ℝ) : Prop := ∃ t : ℝ, x = t*y - 1

-- Define the point P in the first quadrant
def point_P (x y : ℝ) : Prop := x > 0 ∧ y > 0 ∧ parabola x y ∧ tangent_line x y

-- Define the line l passing through (2, 0)
def line_l (x y : ℝ) : Prop := ∃ m : ℝ, x = m*y + 2

-- Define the circle M with AB as diameter passing through P
def circle_M (xa ya xb yb : ℝ) : Prop :=
  ∃ xc yc : ℝ, (xc - 1)^2 + (yc - 2)^2 = ((xa - xb)^2 + (ya - yb)^2) / 4

theorem parabola_tangent_and_intersecting_line :
  -- Part 1: Point of tangency P
  (∃! x y : ℝ, point_P x y ∧ x = 1 ∧ y = 2) ∧
  -- Part 2: Equation of line l
  (∀ xa ya xb yb : ℝ,
    parabola xa ya ∧ parabola xb yb ∧
    line_l xa ya ∧ line_l xb yb ∧
    circle_M xa ya xb yb →
    ∃ m : ℝ, m = -2/3 ∧ ∀ x y : ℝ, line_l x y ↔ y = m*x + 4/3) :=
by sorry

end NUMINAMATH_CALUDE_parabola_tangent_and_intersecting_line_l344_34464


namespace NUMINAMATH_CALUDE_sum_of_digits_94_eights_times_94_sevens_l344_34407

/-- Represents a number with 94 repeated digits --/
def RepeatedDigit (d : ℕ) : ℕ := 
  d * (10^94 - 1) / 9

/-- Sum of digits of a natural number --/
def sumOfDigits (n : ℕ) : ℕ :=
  if n < 10 then n else n % 10 + sumOfDigits (n / 10)

/-- The main theorem --/
theorem sum_of_digits_94_eights_times_94_sevens : 
  sumOfDigits (RepeatedDigit 8 * RepeatedDigit 7) = 1034 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_digits_94_eights_times_94_sevens_l344_34407


namespace NUMINAMATH_CALUDE_remainder_x14_minus_1_div_x_plus_1_l344_34425

theorem remainder_x14_minus_1_div_x_plus_1 (x : ℝ) : 
  (x^14 - 1) % (x + 1) = 0 := by sorry

end NUMINAMATH_CALUDE_remainder_x14_minus_1_div_x_plus_1_l344_34425


namespace NUMINAMATH_CALUDE_diary_pieces_not_complete_l344_34498

theorem diary_pieces_not_complete : ¬∃ (n : ℕ), 4^n = 50 := by sorry

end NUMINAMATH_CALUDE_diary_pieces_not_complete_l344_34498


namespace NUMINAMATH_CALUDE_necessary_but_not_sufficient_l344_34444

theorem necessary_but_not_sufficient (x : ℝ) :
  (∀ x, (1/2)^x > 1 → 1/x < 1) ∧
  ¬(∀ x, 1/x < 1 → (1/2)^x > 1) :=
by sorry

end NUMINAMATH_CALUDE_necessary_but_not_sufficient_l344_34444


namespace NUMINAMATH_CALUDE_monkey_liar_puzzle_l344_34406

-- Define the possible characteristics
inductive Character
| Monkey
| NonMonkey

inductive Truthfulness
| TruthTeller
| Liar

-- Define a structure for an individual
structure Individual where
  species : Character
  honesty : Truthfulness

-- Define the statements made by A and B
def statement_A (a b : Individual) : Prop :=
  a.species = Character.Monkey ∧ b.species = Character.Monkey

def statement_B (a b : Individual) : Prop :=
  a.honesty = Truthfulness.Liar ∧ b.honesty = Truthfulness.Liar

-- Theorem stating the solution
theorem monkey_liar_puzzle :
  ∃ (a b : Individual),
    (statement_A a b ↔ a.honesty = Truthfulness.TruthTeller) ∧
    (statement_B a b ↔ b.honesty = Truthfulness.Liar) ∧
    a.species = Character.Monkey ∧
    b.species = Character.Monkey ∧
    a.honesty = Truthfulness.TruthTeller ∧
    b.honesty = Truthfulness.Liar :=
  sorry


end NUMINAMATH_CALUDE_monkey_liar_puzzle_l344_34406


namespace NUMINAMATH_CALUDE_carnival_game_days_l344_34445

def carnival_game (first_period_earnings : ℕ) (remaining_earnings : ℕ) (daily_earnings : ℕ) : Prop :=
  let first_period_days : ℕ := 20
  let remaining_days : ℕ := remaining_earnings / daily_earnings
  let total_days : ℕ := first_period_days + remaining_days
  (first_period_earnings = first_period_days * daily_earnings) ∧
  (remaining_earnings = remaining_days * daily_earnings) ∧
  (total_days = 31)

theorem carnival_game_days :
  carnival_game 120 66 6 := by
  sorry

end NUMINAMATH_CALUDE_carnival_game_days_l344_34445


namespace NUMINAMATH_CALUDE_shift_increasing_interval_l344_34483

-- Define a real-valued function f
variable (f : ℝ → ℝ)

-- Define what it means for a function to be increasing on an interval
def IncreasingOn (f : ℝ → ℝ) (a b : ℝ) :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f x < f y

-- State the theorem
theorem shift_increasing_interval :
  IncreasingOn f (-2) 3 → IncreasingOn (fun x ↦ f (x + 5)) (-7) (-2) :=
by
  sorry

end NUMINAMATH_CALUDE_shift_increasing_interval_l344_34483


namespace NUMINAMATH_CALUDE_family_size_l344_34409

theorem family_size (boys girls : ℕ) 
  (sister_condition : boys = girls - 1)
  (brother_condition : girls = 2 * (boys - 1)) : 
  boys + girls = 7 := by
sorry

end NUMINAMATH_CALUDE_family_size_l344_34409


namespace NUMINAMATH_CALUDE_complement_A_in_U_l344_34428

-- Define the universal set U
def U : Set ℝ := {x | x^2 ≥ 1}

-- Define set A
def A : Set ℝ := {x | Real.log (x - 1) ≤ 0}

-- State the theorem
theorem complement_A_in_U : 
  (U \ A) = {x | x ≤ -1 ∨ x = 1 ∨ x > 2} := by sorry

end NUMINAMATH_CALUDE_complement_A_in_U_l344_34428


namespace NUMINAMATH_CALUDE_bird_weight_equations_l344_34410

/-- Represents the weight of birds in jin -/
structure BirdWeight where
  sparrow : ℝ
  swallow : ℝ

/-- The total weight of 5 sparrows and 6 swallows is 1 jin -/
def total_weight (w : BirdWeight) : Prop :=
  5 * w.sparrow + 6 * w.swallow = 1

/-- Sparrows are heavier than swallows -/
def sparrow_heavier (w : BirdWeight) : Prop :=
  w.sparrow > w.swallow

/-- Exchanging one sparrow with one swallow doesn't change the total weight -/
def exchange_weight (w : BirdWeight) : Prop :=
  4 * w.sparrow + 7 * w.swallow = 5 * w.swallow + w.sparrow

/-- The system of equations correctly represents the bird weight problem -/
theorem bird_weight_equations (w : BirdWeight) 
  (h1 : total_weight w) 
  (h2 : sparrow_heavier w) 
  (h3 : exchange_weight w) : 
  5 * w.sparrow + 6 * w.swallow = 1 ∧ 3 * w.sparrow = -2 * w.swallow := by
  sorry

end NUMINAMATH_CALUDE_bird_weight_equations_l344_34410


namespace NUMINAMATH_CALUDE_v_domain_characterization_l344_34454

/-- The function v(x) = 1 / sqrt(x^2 - 4) -/
noncomputable def v (x : ℝ) : ℝ := 1 / Real.sqrt (x^2 - 4)

/-- The domain of v(x) -/
def domain_v : Set ℝ := {x | x < -2 ∨ x > 2}

theorem v_domain_characterization :
  ∀ x : ℝ, v x ∈ Set.univ ↔ x ∈ domain_v :=
by sorry

end NUMINAMATH_CALUDE_v_domain_characterization_l344_34454


namespace NUMINAMATH_CALUDE_locus_and_fixed_points_l344_34438

-- Define the points and vectors
variable (P Q R M A S B D E F : ℝ × ℝ)
variable (a b : ℝ)

-- Define the conditions
axiom P_on_x_axis : P.2 = 0
axiom Q_on_y_axis : Q.1 = 0
axiom R_coord : R = (0, -3)
axiom S_coord : S = (0, 2)
axiom PR_dot_PM : (R.1 - P.1) * (M.1 - P.1) + (R.2 - P.2) * (M.2 - P.2) = 0
axiom PQ_half_QM : (Q.1 - P.1, Q.2 - P.2) = (1/2 : ℝ) • (M.1 - Q.1, M.2 - Q.2)
axiom A_coord : A = (a, b)
axiom A_outside_C : a ≠ 0 ∧ b ≠ 2
axiom AB_AD_tangent : True  -- This condition is implied but not directly stated
axiom E_on_line : E.2 = -2
axiom F_on_line : F.2 = -2

-- Define the locus C
def C (x y : ℝ) : Prop := x^2 = 4*y

-- State the theorem
theorem locus_and_fixed_points :
  (∀ x y, C x y ↔ x^2 = 4*y) ∧
  (∃ r : ℝ, r > 0 ∧ 
    ((E.1 + F.1)/2 - 0)^2 + ((E.2 + F.2)/2 - (-2 + 2*Real.sqrt 2))^2 = r^2 ∧
    ((E.1 + F.1)/2 - 0)^2 + ((E.2 + F.2)/2 - (-2 - 2*Real.sqrt 2))^2 = r^2) :=
sorry

end NUMINAMATH_CALUDE_locus_and_fixed_points_l344_34438


namespace NUMINAMATH_CALUDE_race_result_l344_34457

/-- Represents the state of the race between Alex and Max -/
structure RaceState where
  alex_lead : Int
  distance_covered : Int

/-- Calculates the remaining distance for Max to catch up to Alex -/
def remaining_distance (total_length : Int) (final_state : RaceState) : Int :=
  total_length - final_state.distance_covered - final_state.alex_lead

/-- Updates the race state after a change in lead -/
def update_state (state : RaceState) (lead_change : Int) : RaceState :=
  { alex_lead := state.alex_lead + lead_change,
    distance_covered := state.distance_covered }

theorem race_result (total_length : Int) (initial_even : Int) (alex_lead1 : Int) 
                     (max_lead : Int) (alex_lead2 : Int) : 
  total_length = 5000 →
  initial_even = 200 →
  alex_lead1 = 300 →
  max_lead = 170 →
  alex_lead2 = 440 →
  let initial_state : RaceState := { alex_lead := 0, distance_covered := initial_even }
  let state1 := update_state initial_state alex_lead1
  let state2 := update_state state1 (-max_lead)
  let final_state := update_state state2 alex_lead2
  remaining_distance total_length final_state = 4430 := by
  sorry

#check race_result

end NUMINAMATH_CALUDE_race_result_l344_34457


namespace NUMINAMATH_CALUDE_expression_evaluation_l344_34412

theorem expression_evaluation (a b : ℝ) (h1 : a = 3) (h2 : b = 2) :
  (a^2 + b)^2 - (a^2 - b)^2 + 2*a*b = 78 := by
  sorry

end NUMINAMATH_CALUDE_expression_evaluation_l344_34412


namespace NUMINAMATH_CALUDE_spring_length_theorem_l344_34416

/-- Represents the relationship between spring length and attached mass -/
def spring_length (x : ℝ) : ℝ :=
  0.3 * x + 6

/-- Theorem stating the relationship between spring length and attached mass -/
theorem spring_length_theorem (x : ℝ) :
  let initial_length : ℝ := 6
  let extension_rate : ℝ := 0.3
  spring_length x = initial_length + extension_rate * x :=
by
  sorry

#check spring_length_theorem

end NUMINAMATH_CALUDE_spring_length_theorem_l344_34416


namespace NUMINAMATH_CALUDE_restaurant_bill_calculation_total_bill_is_140_l344_34408

/-- Calculates the total bill for a restaurant order with given conditions -/
theorem restaurant_bill_calculation 
  (tax_rate : ℝ) 
  (striploin_cost : ℝ) 
  (wine_cost : ℝ) 
  (gratuities : ℝ) : ℝ :=
  let total_before_tax := striploin_cost + wine_cost
  let tax_amount := tax_rate * total_before_tax
  let total_after_tax := total_before_tax + tax_amount
  let total_bill := total_after_tax + gratuities
  total_bill

/-- Proves that the total bill is $140 given the specified conditions -/
theorem total_bill_is_140 : 
  restaurant_bill_calculation 0.1 80 10 41 = 140 := by
  sorry

end NUMINAMATH_CALUDE_restaurant_bill_calculation_total_bill_is_140_l344_34408


namespace NUMINAMATH_CALUDE_investor_share_price_l344_34463

theorem investor_share_price 
  (dividend_rate : ℝ) 
  (face_value : ℝ) 
  (roi : ℝ) 
  (h1 : dividend_rate = 0.125)
  (h2 : face_value = 40)
  (h3 : roi = 0.25) :
  let dividend_per_share := dividend_rate * face_value
  let price := dividend_per_share / roi
  price = 20 := by sorry

end NUMINAMATH_CALUDE_investor_share_price_l344_34463


namespace NUMINAMATH_CALUDE_number_count_proof_l344_34481

/-- Given a set of numbers with specific average properties, prove that the total count is 8 -/
theorem number_count_proof (n : ℕ) (S : ℝ) (S₅ : ℝ) (S₃ : ℝ) : 
  S / n = 20 →
  S₅ / 5 = 12 →
  S₃ / 3 = 33.333333333333336 →
  S = S₅ + S₃ →
  n = 8 := by
sorry

end NUMINAMATH_CALUDE_number_count_proof_l344_34481


namespace NUMINAMATH_CALUDE_percentage_of_burpees_l344_34475

/-- Calculate the percentage of burpees in Emmett's workout routine -/
theorem percentage_of_burpees (jumping_jacks pushups situps burpees lunges : ℕ) :
  jumping_jacks = 25 →
  pushups = 15 →
  situps = 30 →
  burpees = 10 →
  lunges = 20 →
  (burpees : ℚ) / (jumping_jacks + pushups + situps + burpees + lunges) * 100 = 10 := by
  sorry

end NUMINAMATH_CALUDE_percentage_of_burpees_l344_34475


namespace NUMINAMATH_CALUDE_range_of_c_l344_34439

open Real

theorem range_of_c (c : ℝ) : 
  (∀ x y : ℝ, x < y → c^x > c^y) →  -- y = c^x is decreasing
  (∃ x : ℝ, x^2 - Real.sqrt 2 * x + c ≤ 0) →  -- negation of q
  (0 < c ∧ c < 1) →  -- derived from decreasing function condition
  0 < c ∧ c ≤ (1/2) := by
sorry

end NUMINAMATH_CALUDE_range_of_c_l344_34439


namespace NUMINAMATH_CALUDE_toy_cost_price_l344_34477

/-- The cost price of one toy -/
def cost_price : ℕ := sorry

/-- The selling price of 18 toys -/
def selling_price : ℕ := 18900

/-- The number of toys sold -/
def toys_sold : ℕ := 18

/-- The number of toys whose cost price equals the gain -/
def gain_toys : ℕ := 3

theorem toy_cost_price : 
  (toys_sold + gain_toys) * cost_price = selling_price → 
  cost_price = 900 := by sorry

end NUMINAMATH_CALUDE_toy_cost_price_l344_34477


namespace NUMINAMATH_CALUDE_point_in_second_quadrant_l344_34405

def second_quadrant (x y : ℝ) : Prop := x < 0 ∧ y > 0

theorem point_in_second_quadrant :
  let x : ℝ := -5
  let y : ℝ := 4
  second_quadrant x y :=
by
  sorry

end NUMINAMATH_CALUDE_point_in_second_quadrant_l344_34405


namespace NUMINAMATH_CALUDE_soccer_substitutions_remainder_l344_34468

/-- Represents the number of ways to make exactly n substitutions -/
def b (n : Nat) : Nat :=
  match n with
  | 0 => 1
  | n + 1 => 12 * (12 - n) * b n

/-- The total number of possible substitution ways -/
def total_ways : Nat :=
  b 0 + b 1 + b 2 + b 3 + b 4 + b 5

theorem soccer_substitutions_remainder :
  total_ways % 100 = 93 := by
  sorry

end NUMINAMATH_CALUDE_soccer_substitutions_remainder_l344_34468


namespace NUMINAMATH_CALUDE_odd_function_zero_condition_l344_34489

-- Define a real-valued function
def RealFunction := ℝ → ℝ

-- Define what it means for a function to be odd
def IsOdd (f : RealFunction) : Prop := ∀ x : ℝ, f (-x) = -f x

-- State the theorem
theorem odd_function_zero_condition :
  (∀ f : RealFunction, IsOdd f → f 0 = 0) ∧
  (∃ f : RealFunction, f 0 = 0 ∧ ¬IsOdd f) :=
sorry

end NUMINAMATH_CALUDE_odd_function_zero_condition_l344_34489


namespace NUMINAMATH_CALUDE_min_time_two_students_l344_34466

-- Define the processes
inductive Process : Type
| A | B | C | D | E | F | G

-- Define the time required for each process
def processTime (p : Process) : Nat :=
  match p with
  | Process.A => 9
  | Process.B => 9
  | Process.C => 7
  | Process.D => 9
  | Process.E => 7
  | Process.F => 10
  | Process.G => 2

-- Define the dependency relation between processes
def dependsOn : Process → Process → Prop
| Process.C, Process.A => True
| Process.D, Process.A => True
| Process.E, Process.B => True
| Process.E, Process.D => True
| Process.F, Process.C => True
| Process.F, Process.D => True
| _, _ => False

-- Define a valid schedule as a list of processes
def ValidSchedule (schedule : List Process) : Prop := sorry

-- Define the time taken by a schedule
def scheduleTime (schedule : List Process) : Nat := sorry

-- Theorem: The minimum time for two students to complete the project is 28 minutes
theorem min_time_two_students :
  ∃ (schedule1 schedule2 : List Process),
    ValidSchedule schedule1 ∧
    ValidSchedule schedule2 ∧
    (∀ p, p ∈ schedule1 ∨ p ∈ schedule2) ∧
    max (scheduleTime schedule1) (scheduleTime schedule2) = 28 ∧
    (∀ s1 s2, ValidSchedule s1 → ValidSchedule s2 →
      (∀ p, p ∈ s1 ∨ p ∈ s2) →
      max (scheduleTime s1) (scheduleTime s2) ≥ 28) := by sorry

end NUMINAMATH_CALUDE_min_time_two_students_l344_34466


namespace NUMINAMATH_CALUDE_sum_of_squares_quadratic_roots_sum_of_squares_specific_quadratic_l344_34434

theorem sum_of_squares_quadratic_roots (a b c : ℝ) (h : a ≠ 0) :
  let r₁ := (-b + Real.sqrt (b^2 - 4*a*c)) / (2*a)
  let r₂ := (-b - Real.sqrt (b^2 - 4*a*c)) / (2*a)
  a * x^2 + b * x + c = 0 → r₁^2 + r₂^2 = (b^2 - 2*a*c) / a^2 :=
by sorry

theorem sum_of_squares_specific_quadratic :
  let r₁ := (16 + Real.sqrt 256) / 2
  let r₂ := (16 - Real.sqrt 256) / 2
  x^2 - 16*x + 4 = 0 → r₁^2 + r₂^2 = 248 :=
by sorry

end NUMINAMATH_CALUDE_sum_of_squares_quadratic_roots_sum_of_squares_specific_quadratic_l344_34434


namespace NUMINAMATH_CALUDE_simplify_fraction_l344_34465

theorem simplify_fraction : 5 * (18 / 7) * (21 / -45) = -6 / 5 := by
  sorry

end NUMINAMATH_CALUDE_simplify_fraction_l344_34465


namespace NUMINAMATH_CALUDE_smallest_b_for_integer_solutions_l344_34411

theorem smallest_b_for_integer_solutions : 
  ∃ (b : ℕ), b > 0 ∧ 
  (∀ (x : ℤ), x^2 + b*x = -21 → ∃ (y : ℤ), x = y) ∧
  (∀ (b' : ℕ), 0 < b' ∧ b' < b → 
    ∃ (x : ℝ), x^2 + b'*x = -21 ∧ ¬∃ (y : ℤ), x = y) ∧
  b = 10 := by
sorry

end NUMINAMATH_CALUDE_smallest_b_for_integer_solutions_l344_34411


namespace NUMINAMATH_CALUDE_correct_large_slices_per_pepper_l344_34458

/-- The number of bell peppers Tamia uses. -/
def num_peppers : ℕ := 5

/-- The total number of slices and pieces Tamia wants to add to her meal. -/
def total_slices : ℕ := 200

/-- Calculates the total number of slices and pieces based on the number of large slices per pepper. -/
def total_slices_func (x : ℕ) : ℕ :=
  num_peppers * x + num_peppers * (x / 2) * 3

/-- The number of large slices Tamia cuts each bell pepper into. -/
def large_slices_per_pepper : ℕ := 16

/-- Theorem stating that the number of large slices per pepper is correct. -/
theorem correct_large_slices_per_pepper : 
  total_slices_func large_slices_per_pepper = total_slices :=
by sorry

end NUMINAMATH_CALUDE_correct_large_slices_per_pepper_l344_34458


namespace NUMINAMATH_CALUDE_A_empty_iff_a_in_range_l344_34497

/-- The set A for a given real number a -/
def A (a : ℝ) : Set ℝ := {x : ℝ | a * x^2 - a * x + 1 ≤ 0}

/-- Theorem stating the equivalence between A being empty and the range of a -/
theorem A_empty_iff_a_in_range : 
  ∀ a : ℝ, A a = ∅ ↔ 0 ≤ a ∧ a < 4 := by sorry

end NUMINAMATH_CALUDE_A_empty_iff_a_in_range_l344_34497


namespace NUMINAMATH_CALUDE_student_rank_theorem_l344_34495

/-- Given a line of students, this function calculates a student's rank from the right
    based on their rank from the left and the total number of students. -/
def rankFromRight (totalStudents : ℕ) (rankFromLeft : ℕ) : ℕ :=
  totalStudents - rankFromLeft + 1

/-- Theorem stating that for a line of 10 students, 
    a student ranked 5th from the left is ranked 6th from the right. -/
theorem student_rank_theorem :
  rankFromRight 10 5 = 6 := by
  sorry

end NUMINAMATH_CALUDE_student_rank_theorem_l344_34495


namespace NUMINAMATH_CALUDE_cross_section_distance_theorem_l344_34471

/-- Represents a right hexagonal pyramid -/
structure RightHexagonalPyramid where
  -- Add any necessary fields here
  mk ::

/-- Represents a cross-section of the pyramid -/
structure CrossSection where
  area : ℝ
  distance_from_apex : ℝ

/-- Theorem about the distance of cross-sections in a right hexagonal pyramid -/
theorem cross_section_distance_theorem 
  (pyramid : RightHexagonalPyramid)
  (cs1 cs2 : CrossSection)
  (h : cs1.distance_from_apex < cs2.distance_from_apex)
  (area_h : cs1.area < cs2.area)
  (d : ℝ)
  (h_d : d = cs2.distance_from_apex - cs1.distance_from_apex) :
  cs2.distance_from_apex = d / (1 - Real.sqrt (cs1.area / cs2.area)) :=
by sorry

end NUMINAMATH_CALUDE_cross_section_distance_theorem_l344_34471


namespace NUMINAMATH_CALUDE_product_is_2008th_power_l344_34476

theorem product_is_2008th_power : ∃ (a b c : ℕ), 
  (a ≠ b ∧ b ≠ c ∧ a ≠ c) ∧ 
  ((a = (b + c) / 2) ∨ (b = (a + c) / 2) ∨ (c = (a + b) / 2)) ∧
  (∃ (n : ℕ), a * b * c = n ^ 2008) :=
by sorry

end NUMINAMATH_CALUDE_product_is_2008th_power_l344_34476


namespace NUMINAMATH_CALUDE_exam_pass_count_l344_34486

theorem exam_pass_count (total_boys : ℕ) (overall_avg : ℚ) (pass_avg : ℚ) (fail_avg : ℚ) :
  total_boys = 120 →
  overall_avg = 38 →
  pass_avg = 39 →
  fail_avg = 15 →
  ∃ (pass_count : ℕ), pass_count = 115 ∧ 
    pass_count * pass_avg + (total_boys - pass_count) * fail_avg = total_boys * overall_avg :=
by sorry

end NUMINAMATH_CALUDE_exam_pass_count_l344_34486


namespace NUMINAMATH_CALUDE_sum_of_roots_for_f_l344_34432

def f (x : ℝ) : ℝ := (3*x)^2 + 2*(3*x) + 1

theorem sum_of_roots_for_f (z : ℝ) : 
  (∃ z₁ z₂, f z₁ = 13 ∧ f z₂ = 13 ∧ z₁ ≠ z₂ ∧ z₁ + z₂ = -2/9) :=
sorry

end NUMINAMATH_CALUDE_sum_of_roots_for_f_l344_34432


namespace NUMINAMATH_CALUDE_average_of_three_numbers_l344_34417

theorem average_of_three_numbers (y : ℝ) : (14 + 23 + y) / 3 = 21 → y = 26 := by
  sorry

end NUMINAMATH_CALUDE_average_of_three_numbers_l344_34417


namespace NUMINAMATH_CALUDE_complex_magnitude_squared_l344_34449

theorem complex_magnitude_squared (z : ℂ) (h : z + Complex.abs z = 6 + 10 * Complex.I) : 
  Complex.abs z ^ 2 = 1156 / 9 := by
  sorry

end NUMINAMATH_CALUDE_complex_magnitude_squared_l344_34449


namespace NUMINAMATH_CALUDE_sum_and_product_calculation_l344_34459

theorem sum_and_product_calculation :
  (199 + 298 + 397 + 496 + 595 + 20 = 2005) ∧
  (39 * 25 = 975) := by
sorry

end NUMINAMATH_CALUDE_sum_and_product_calculation_l344_34459


namespace NUMINAMATH_CALUDE_parabola_directrix_l344_34403

/-- Given a parabola y = 3x^2 - 6x + 2, its directrix is y = -13/12 -/
theorem parabola_directrix (x y : ℝ) :
  y = 3 * x^2 - 6 * x + 2 →
  ∃ (k : ℝ), k = -13/12 ∧ k = y - 3 * (x - 1)^2 + 1 := by
  sorry

end NUMINAMATH_CALUDE_parabola_directrix_l344_34403


namespace NUMINAMATH_CALUDE_solution_set_inequalities_l344_34490

theorem solution_set_inequalities :
  {x : ℝ | x - 2 > 1 ∧ x < 4} = {x : ℝ | 3 < x ∧ x < 4} := by
  sorry

end NUMINAMATH_CALUDE_solution_set_inequalities_l344_34490


namespace NUMINAMATH_CALUDE_produce_worth_l344_34435

theorem produce_worth (asparagus_bundles : ℕ) (asparagus_price : ℚ)
                      (grape_boxes : ℕ) (grape_price : ℚ)
                      (apple_count : ℕ) (apple_price : ℚ) :
  asparagus_bundles = 60 ∧ asparagus_price = 3 ∧
  grape_boxes = 40 ∧ grape_price = 5/2 ∧
  apple_count = 700 ∧ apple_price = 1/2 →
  asparagus_bundles * asparagus_price +
  grape_boxes * grape_price +
  apple_count * apple_price = 630 :=
by sorry

end NUMINAMATH_CALUDE_produce_worth_l344_34435


namespace NUMINAMATH_CALUDE_right_triangle_third_side_product_l344_34450

theorem right_triangle_third_side_product (a b c d : ℝ) : 
  a = 6 → b = 8 → 
  ((a^2 + b^2 = c^2) ∨ (a^2 + d^2 = b^2)) → 
  c * d = 20 * Real.sqrt 7 := by
  sorry

end NUMINAMATH_CALUDE_right_triangle_third_side_product_l344_34450


namespace NUMINAMATH_CALUDE_A_suff_not_nec_D_l344_34492

-- Define propositions
variable (A B C D : Prop)

-- Define the relationships between the propositions
axiom A_suff_not_nec_B : (A → B) ∧ ¬(B → A)
axiom B_iff_C : B ↔ C
axiom D_nec_not_suff_C : (C → D) ∧ ¬(D → C)

-- Theorem to prove
theorem A_suff_not_nec_D : (A → D) ∧ ¬(D → A) :=
by sorry

end NUMINAMATH_CALUDE_A_suff_not_nec_D_l344_34492


namespace NUMINAMATH_CALUDE_floor_ceil_sum_l344_34401

theorem floor_ceil_sum : ⌊(-3.67 : ℝ)⌋ + ⌈(34.2 : ℝ)⌉ = 31 := by sorry

end NUMINAMATH_CALUDE_floor_ceil_sum_l344_34401


namespace NUMINAMATH_CALUDE_one_third_minus_decimal_l344_34485

theorem one_third_minus_decimal : (1 : ℚ) / 3 - 333 / 1000 = 1 / 3000 := by
  sorry

end NUMINAMATH_CALUDE_one_third_minus_decimal_l344_34485


namespace NUMINAMATH_CALUDE_function_and_tangent_line_l344_34427

-- Define the function f(x)
def f (a b c : ℝ) (x : ℝ) : ℝ := a * x^3 + b * x^2 + c * x + 2

-- State the theorem
theorem function_and_tangent_line 
  (a b c : ℝ) 
  (h1 : ∃ (x : ℝ), x = 2 ∧ (3 * a * x^2 + 2 * b * x + c = 0)) -- extremum at x = 2
  (h2 : f a b c 2 = -6) -- f(2) = -6
  (h3 : c = -4) -- f'(0) = -4
  : 
  (∀ x, f a b c x = x^3 - 2*x^2 - 4*x + 2) ∧ -- f(x) = x³ - 2x² - 4x + 2
  (∃ (m n : ℝ), m = 3 ∧ n = 6 ∧ ∀ x y, y = (f a b c (-1)) + (3 * a * (-1)^2 + 2 * b * (-1) + c) * (x - (-1)) ↔ m * x - y + n = 0) -- Tangent line equation at x = -1
  := by sorry

end NUMINAMATH_CALUDE_function_and_tangent_line_l344_34427


namespace NUMINAMATH_CALUDE_tank_water_level_l344_34455

theorem tank_water_level (tank_capacity : ℚ) (initial_fraction : ℚ) (added_water : ℚ) :
  tank_capacity = 72 →
  initial_fraction = 3 / 4 →
  added_water = 9 →
  (initial_fraction * tank_capacity + added_water) / tank_capacity = 7 / 8 := by
  sorry

end NUMINAMATH_CALUDE_tank_water_level_l344_34455


namespace NUMINAMATH_CALUDE_initial_water_amount_l344_34443

theorem initial_water_amount (initial_amount : ℝ) : 
  initial_amount + 6.8 = 9.8 → initial_amount = 3 := by
  sorry

end NUMINAMATH_CALUDE_initial_water_amount_l344_34443


namespace NUMINAMATH_CALUDE_arc_length_for_36_degree_angle_l344_34467

theorem arc_length_for_36_degree_angle (d : ℝ) (θ : ℝ) (L : ℝ) : 
  d = 4 → θ = 36 * π / 180 → L = d * π * θ / 360 → L = 2 * π / 5 := by
  sorry

end NUMINAMATH_CALUDE_arc_length_for_36_degree_angle_l344_34467


namespace NUMINAMATH_CALUDE_function_property_l344_34482

theorem function_property (k : ℝ) (h_k : k > 0) :
  ∀ (f : ℝ → ℝ), 
  (∀ (x : ℝ), x > 0 → (f (x^2 + 1))^(Real.sqrt x) = k) →
  ∀ (y : ℝ), y > 0 → (f ((9 + y^2) / y^2))^(Real.sqrt (12 / y)) = k^2 := by
  sorry

end NUMINAMATH_CALUDE_function_property_l344_34482


namespace NUMINAMATH_CALUDE_point_distance_on_number_line_l344_34496

theorem point_distance_on_number_line :
  ∀ x : ℝ, |x - (-3)| = 4 ↔ x = -7 ∨ x = 1 := by sorry

end NUMINAMATH_CALUDE_point_distance_on_number_line_l344_34496


namespace NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l344_34451

def is_geometric_sequence (a : ℕ → ℝ) : Prop :=
  ∃ q : ℝ, ∀ n : ℕ, a (n + 1) = q * a n

theorem geometric_sequence_common_ratio 
  (a : ℕ → ℝ) 
  (h_geometric : is_geometric_sequence a) 
  (h_condition : a 1 * a 7 = 3 * a 3 * a 4) :
  ∃ q : ℝ, (∀ n : ℕ, a (n + 1) = q * a n) ∧ q = 3 :=
sorry

end NUMINAMATH_CALUDE_geometric_sequence_common_ratio_l344_34451


namespace NUMINAMATH_CALUDE_coefficient_x2y4_is_30_l344_34418

/-- The coefficient of x^2y^4 in the expansion of (1+x+y^2)^5 -/
def coefficient_x2y4 : ℕ :=
  (Nat.choose 5 2) * (Nat.choose 3 2)

/-- Theorem stating that the coefficient of x^2y^4 in (1+x+y^2)^5 is 30 -/
theorem coefficient_x2y4_is_30 : coefficient_x2y4 = 30 := by
  sorry

end NUMINAMATH_CALUDE_coefficient_x2y4_is_30_l344_34418


namespace NUMINAMATH_CALUDE_set_size_from_average_change_l344_34447

theorem set_size_from_average_change (S : Finset ℝ) (initial_avg final_avg : ℝ) :
  initial_avg = (S.sum id) / S.card →
  final_avg = ((S.sum id) + 6) / S.card →
  initial_avg = 6.2 →
  final_avg = 6.8 →
  S.card = 10 := by
  sorry

end NUMINAMATH_CALUDE_set_size_from_average_change_l344_34447


namespace NUMINAMATH_CALUDE_fuel_purchase_l344_34460

/-- Fuel purchase problem -/
theorem fuel_purchase (total_spent : ℝ) (initial_cost final_cost : ℝ) :
  total_spent = 90 ∧
  initial_cost = 3 ∧
  final_cost = 4 ∧
  ∃ (mid_cost : ℝ), initial_cost < mid_cost ∧ mid_cost < final_cost →
  ∃ (quantity : ℝ),
    quantity > 0 ∧
    total_spent = initial_cost * quantity + ((initial_cost + final_cost) / 2) * quantity + final_cost * quantity ∧
    initial_cost * quantity + final_cost * quantity = 60 :=
by sorry

end NUMINAMATH_CALUDE_fuel_purchase_l344_34460
