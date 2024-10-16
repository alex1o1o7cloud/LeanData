import Mathlib

namespace NUMINAMATH_CALUDE_cubic_root_sum_l3936_393606

theorem cubic_root_sum (p q r : ℝ) : 
  p^3 - 8*p^2 + 11*p - 3 = 0 →
  q^3 - 8*q^2 + 11*q - 3 = 0 →
  r^3 - 8*r^2 + 11*r - 3 = 0 →
  p/(q*r + 1) + q/(p*r + 1) + r/(p*q + 1) = 32/15 := by
sorry

end NUMINAMATH_CALUDE_cubic_root_sum_l3936_393606


namespace NUMINAMATH_CALUDE_tv_production_average_l3936_393611

theorem tv_production_average (total_days : ℕ) (first_period : ℕ) (first_avg : ℝ) (total_avg : ℝ) :
  total_days = 30 →
  first_period = 25 →
  first_avg = 50 →
  total_avg = 45 →
  (total_days * total_avg - first_period * first_avg) / (total_days - first_period) = 20 := by
sorry

end NUMINAMATH_CALUDE_tv_production_average_l3936_393611


namespace NUMINAMATH_CALUDE_cloth_sale_quantity_l3936_393663

/-- Proves that the number of metres of cloth sold is 300 given the specified conditions --/
theorem cloth_sale_quantity (total_selling_price : ℕ) (loss_per_metre : ℕ) (cost_price_per_metre : ℕ) :
  total_selling_price = 18000 →
  loss_per_metre = 5 →
  cost_price_per_metre = 65 →
  (total_selling_price / (cost_price_per_metre - loss_per_metre) : ℕ) = 300 := by
  sorry

end NUMINAMATH_CALUDE_cloth_sale_quantity_l3936_393663


namespace NUMINAMATH_CALUDE_f_properties_l3936_393669

-- Define the function f(x)
noncomputable def f (a : ℝ) (x : ℝ) : ℝ := -x^3 + 3*x^2 + 9*x + a

-- Define the derivative of f(x)
noncomputable def f' (x : ℝ) : ℝ := -3*x^2 + 6*x + 9

theorem f_properties (a : ℝ) :
  -- Part 1: Monotonically decreasing intervals
  (∀ x < -1, (f' x) < 0) ∧
  (∀ x > 3, (f' x) < 0) ∧
  -- Part 2: Maximum and minimum values
  (∃ x ∈ Set.Icc (-2) 2, f a x = 20) →
  (∃ y ∈ Set.Icc (-2) 2, f a y = -7 ∧ ∀ z ∈ Set.Icc (-2) 2, f a z ≥ f a y) :=
by sorry

end NUMINAMATH_CALUDE_f_properties_l3936_393669


namespace NUMINAMATH_CALUDE_max_value_not_one_l3936_393613

theorem max_value_not_one :
  let f : ℝ → ℝ := λ x ↦ Real.sin (x + π/4)
  let g : ℝ → ℝ := λ x ↦ Real.cos (x - π/4)
  let y : ℝ → ℝ := λ x ↦ f x * g x
  ∃ M : ℝ, (∀ x, y x ≤ M) ∧ M < 1 :=
by sorry

end NUMINAMATH_CALUDE_max_value_not_one_l3936_393613


namespace NUMINAMATH_CALUDE_savings_double_l3936_393630

/-- Represents the financial situation of a man over two years -/
structure FinancialSituation where
  first_year_income : ℝ
  first_year_savings_rate : ℝ
  income_increase_rate : ℝ
  expenditure_ratio : ℝ

/-- Calculates the percentage increase in savings -/
def savings_increase_percentage (fs : FinancialSituation) : ℝ :=
  -- The actual calculation will be implemented in the proof
  sorry

/-- Theorem stating that the savings increase by 100% -/
theorem savings_double (fs : FinancialSituation) 
  (h1 : fs.first_year_savings_rate = 0.35)
  (h2 : fs.income_increase_rate = 0.35)
  (h3 : fs.expenditure_ratio = 2)
  : savings_increase_percentage fs = 100 := by
  sorry

end NUMINAMATH_CALUDE_savings_double_l3936_393630


namespace NUMINAMATH_CALUDE_charlies_laps_l3936_393618

/-- Given Charlie's steps per lap and total steps in a session, calculate the number of complete laps --/
theorem charlies_laps (steps_per_lap : ℕ) (total_steps : ℕ) : 
  steps_per_lap = 5350 → total_steps = 13375 → (total_steps / steps_per_lap : ℕ) = 2 :=
by
  sorry

#eval (13375 / 5350 : ℕ)

end NUMINAMATH_CALUDE_charlies_laps_l3936_393618


namespace NUMINAMATH_CALUDE_rope_initial_length_l3936_393673

/-- Given a rope cut into pieces, calculate its initial length -/
theorem rope_initial_length
  (num_pieces : ℕ)
  (tied_pieces : ℕ)
  (knot_reduction : ℕ)
  (final_length : ℕ)
  (h1 : num_pieces = 12)
  (h2 : tied_pieces = 3)
  (h3 : knot_reduction = 1)
  (h4 : final_length = 15) :
  (final_length + knot_reduction) * num_pieces = 192 :=
by sorry

end NUMINAMATH_CALUDE_rope_initial_length_l3936_393673


namespace NUMINAMATH_CALUDE_last_day_sales_l3936_393683

/-- The number of packs sold by Lucy and Robyn on their last day -/
def total_packs_sold (lucy_packs robyn_packs : ℕ) : ℕ :=
  lucy_packs + robyn_packs

/-- Theorem stating that the total number of packs sold by Lucy and Robyn is 35 -/
theorem last_day_sales : total_packs_sold 19 16 = 35 := by
  sorry

end NUMINAMATH_CALUDE_last_day_sales_l3936_393683


namespace NUMINAMATH_CALUDE_point_M_in_first_quadrant_l3936_393635

/-- If point P(0,m) lies on the negative half-axis of the y-axis, 
    then point M(-m,-m+1) lies in the first quadrant. -/
theorem point_M_in_first_quadrant (m : ℝ) : 
  m < 0 → -m > 0 ∧ -m + 1 > 0 := by sorry

end NUMINAMATH_CALUDE_point_M_in_first_quadrant_l3936_393635


namespace NUMINAMATH_CALUDE_probability_different_suits_l3936_393610

def deck_size : ℕ := 60
def num_suits : ℕ := 5
def cards_per_suit : ℕ := 12

theorem probability_different_suits :
  let remaining_cards : ℕ := deck_size - 1
  let cards_not_same_suit : ℕ := remaining_cards - (cards_per_suit - 1)
  (cards_not_same_suit : ℚ) / remaining_cards = 48 / 59 :=
by sorry

end NUMINAMATH_CALUDE_probability_different_suits_l3936_393610


namespace NUMINAMATH_CALUDE_factorial_ratio_squared_l3936_393662

theorem factorial_ratio_squared : (Nat.factorial 45 / Nat.factorial 43)^2 = 3920400 := by
  sorry

end NUMINAMATH_CALUDE_factorial_ratio_squared_l3936_393662


namespace NUMINAMATH_CALUDE_square_of_sum_m_plus_two_n_l3936_393696

theorem square_of_sum_m_plus_two_n (m n : ℝ) : (m + 2*n)^2 = m^2 + 4*n^2 + 4*m*n := by
  sorry

end NUMINAMATH_CALUDE_square_of_sum_m_plus_two_n_l3936_393696


namespace NUMINAMATH_CALUDE_log_inequality_range_l3936_393624

open Real

theorem log_inequality_range (f : ℝ → ℝ) (t : ℝ) : 
  (∀ x > 0, f x = log x) →
  (∀ x > 0, f x + f t ≤ f (x^2 + t)) →
  0 < t ∧ t ≤ 4 :=
by sorry

end NUMINAMATH_CALUDE_log_inequality_range_l3936_393624


namespace NUMINAMATH_CALUDE_ryan_solution_unique_l3936_393604

/-- Represents the solution to Ryan's grocery purchase --/
structure GrocerySolution where
  corn : ℝ
  beans : ℝ
  rice : ℝ

/-- Checks if a given solution satisfies all the problem conditions --/
def is_valid_solution (s : GrocerySolution) : Prop :=
  s.corn + s.beans + s.rice = 30 ∧
  1.20 * s.corn + 0.60 * s.beans + 0.80 * s.rice = 24 ∧
  s.beans = s.rice

/-- The unique solution to the problem --/
def ryan_solution : GrocerySolution :=
  { corn := 6, beans := 12, rice := 12 }

/-- Theorem stating that ryan_solution is the only valid solution --/
theorem ryan_solution_unique :
  is_valid_solution ryan_solution ∧
  ∀ s : GrocerySolution, is_valid_solution s → s = ryan_solution :=
sorry

end NUMINAMATH_CALUDE_ryan_solution_unique_l3936_393604


namespace NUMINAMATH_CALUDE_fairCoinDifference_l3936_393614

def fairCoinProbability : ℚ := 1 / 2

def binomialProbability (n k : ℕ) (p : ℚ) : ℚ :=
  (Nat.choose n k : ℚ) * p^k * (1 - p)^(n - k)

def probabilityThreeHeads : ℚ :=
  binomialProbability 4 3 fairCoinProbability

def probabilityFourHeads : ℚ :=
  fairCoinProbability^4

theorem fairCoinDifference :
  probabilityThreeHeads - probabilityFourHeads = 3 / 16 := by
  sorry

end NUMINAMATH_CALUDE_fairCoinDifference_l3936_393614


namespace NUMINAMATH_CALUDE_people_sitting_on_benches_l3936_393636

theorem people_sitting_on_benches (num_benches : ℕ) (bench_capacity : ℕ) (available_spaces : ℕ) : 
  num_benches = 50 → bench_capacity = 4 → available_spaces = 120 → 
  num_benches * bench_capacity - available_spaces = 80 :=
by
  sorry

end NUMINAMATH_CALUDE_people_sitting_on_benches_l3936_393636


namespace NUMINAMATH_CALUDE_students_in_row_l3936_393631

theorem students_in_row (S R : ℕ) : 
  S = 5 * R + 6 →
  S = 6 * (R - 3) →
  6 = S / R - 18 := by
sorry

end NUMINAMATH_CALUDE_students_in_row_l3936_393631


namespace NUMINAMATH_CALUDE_candy_store_spending_l3936_393671

/-- Proves that given a weekly allowance of $2.25, after spending 3/5 of it at the arcade
    and 1/3 of the remainder at the toy store, the amount left for the candy store is $0.60. -/
theorem candy_store_spending (weekly_allowance : ℚ) (h1 : weekly_allowance = 2.25) :
  let arcade_spending := (3 / 5) * weekly_allowance
  let remaining_after_arcade := weekly_allowance - arcade_spending
  let toy_store_spending := (1 / 3) * remaining_after_arcade
  let candy_store_spending := remaining_after_arcade - toy_store_spending
  candy_store_spending = 0.60 := by
  sorry


end NUMINAMATH_CALUDE_candy_store_spending_l3936_393671


namespace NUMINAMATH_CALUDE_power_two_plus_two_gt_square_l3936_393645

theorem power_two_plus_two_gt_square (n : ℕ) (h : n > 0) : 2^n + 2 > n^2 := by
  sorry

end NUMINAMATH_CALUDE_power_two_plus_two_gt_square_l3936_393645


namespace NUMINAMATH_CALUDE_hiking_resupply_percentage_l3936_393698

/-- A hiking problem with resupply calculation -/
theorem hiking_resupply_percentage
  (supplies_per_mile : Real)
  (hiking_speed : Real)
  (hours_per_day : Real)
  (days : Real)
  (first_pack_weight : Real)
  (h1 : supplies_per_mile = 0.5)
  (h2 : hiking_speed = 2.5)
  (h3 : hours_per_day = 8)
  (h4 : days = 5)
  (h5 : first_pack_weight = 40) :
  let total_distance := hiking_speed * hours_per_day * days
  let total_supplies := total_distance * supplies_per_mile
  let resupply_weight := total_supplies - first_pack_weight
  resupply_weight / first_pack_weight * 100 = 25 := by
  sorry

#check hiking_resupply_percentage

end NUMINAMATH_CALUDE_hiking_resupply_percentage_l3936_393698


namespace NUMINAMATH_CALUDE_equation_solutions_l3936_393661

theorem equation_solutions (a : ℝ) : 
  ((∃! x : ℝ, 5 + |x - 2| = a) ∧ (∃ x y : ℝ, x ≠ y ∧ 7 - |2*x + 6| = a ∧ 7 - |2*y + 6| = a) ∨
   (∃ x y : ℝ, x ≠ y ∧ 5 + |x - 2| = a ∧ 5 + |y - 2| = a) ∧ (∃! x : ℝ, 7 - |2*x + 6| = a)) ↔
  (a = 5 ∨ a = 7) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l3936_393661


namespace NUMINAMATH_CALUDE_production_days_l3936_393640

theorem production_days (n : ℕ) : 
  (n * 50 + 100) / (n + 1) = 55 → n = 9 := by
sorry

end NUMINAMATH_CALUDE_production_days_l3936_393640


namespace NUMINAMATH_CALUDE_percentage_to_full_amount_l3936_393682

theorem percentage_to_full_amount (amount : ℝ) : 
  (25 / 100) * amount = 200 → amount = 800 := by
  sorry

end NUMINAMATH_CALUDE_percentage_to_full_amount_l3936_393682


namespace NUMINAMATH_CALUDE_parabola_point_distance_to_x_axis_l3936_393632

/-- Prove that for a point on a specific parabola with a given distance to the focus,
    its distance to the x-axis is 15/16 -/
theorem parabola_point_distance_to_x_axis 
  (x₀ y₀ : ℝ) -- Coordinates of point M
  (h_parabola : x₀^2 = (1/4) * y₀) -- M is on the parabola
  (h_focus_dist : (x₀^2 + (y₀ - 1/16)^2) = 1) -- Distance from M to focus is 1
  : |y₀| = 15/16 := by
  sorry

end NUMINAMATH_CALUDE_parabola_point_distance_to_x_axis_l3936_393632


namespace NUMINAMATH_CALUDE_no_valid_m_l3936_393626

/-- The trajectory of point M -/
def trajectory (x y m : ℝ) : Prop :=
  x^2 / 4 - y^2 / (m^2 - 4) = 1 ∧ x ≥ 2

/-- Line L -/
def line_L (x y : ℝ) : Prop :=
  y = (1/2) * x - 3

/-- Intersection points of trajectory and line L -/
def intersection_points (m : ℝ) : Prop :=
  ∃ x₁ y₁ x₂ y₂ : ℝ,
    trajectory x₁ y₁ m ∧ trajectory x₂ y₂ m ∧
    line_L x₁ y₁ ∧ line_L x₂ y₂

/-- Vector dot product condition -/
def dot_product_condition (m : ℝ) : Prop :=
  ∃ x₁ y₁ x₂ y₂ : ℝ,
    trajectory x₁ y₁ m ∧ trajectory x₂ y₂ m ∧
    line_L x₁ y₁ ∧ line_L x₂ y₂ ∧
    (x₁ * x₂ + (y₁ - 1) * (y₂ - 1) = 9/2)

theorem no_valid_m :
  ¬∃ m : ℝ, m > 2 ∧ intersection_points m ∧ dot_product_condition m :=
sorry

end NUMINAMATH_CALUDE_no_valid_m_l3936_393626


namespace NUMINAMATH_CALUDE_time_ratio_third_to_first_l3936_393612

-- Define the distances and speed ratios
def distance_first : ℝ := 60
def distance_second : ℝ := 240
def distance_third : ℝ := 180
def speed_ratio_second : ℝ := 4
def speed_ratio_third : ℝ := 2

-- Define the theorem
theorem time_ratio_third_to_first :
  let time_first := distance_first / (distance_first / time_first)
  let time_third := distance_third / (speed_ratio_third * (distance_first / time_first))
  time_third / time_first = 1.5 := by
  sorry

end NUMINAMATH_CALUDE_time_ratio_third_to_first_l3936_393612


namespace NUMINAMATH_CALUDE_sqrt_64_equals_8_l3936_393625

theorem sqrt_64_equals_8 : Real.sqrt 64 = 8 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_64_equals_8_l3936_393625


namespace NUMINAMATH_CALUDE_max_temp_difference_example_l3936_393672

/-- The maximum temperature difference given the highest and lowest temperatures -/
def max_temp_difference (highest lowest : ℤ) : ℤ :=
  highest - lowest

/-- Theorem: The maximum temperature difference is 20℃ given the highest temperature of 18℃ and lowest temperature of -2℃ -/
theorem max_temp_difference_example : max_temp_difference 18 (-2) = 20 := by
  sorry

end NUMINAMATH_CALUDE_max_temp_difference_example_l3936_393672


namespace NUMINAMATH_CALUDE_simplify_and_evaluate_l3936_393680

theorem simplify_and_evaluate (a : ℝ) (h : a = 3) : 
  (2 * a / (a + 1) - 1) / ((a - 1)^2 / (a + 1)) = 1 / 2 := by
  sorry

end NUMINAMATH_CALUDE_simplify_and_evaluate_l3936_393680


namespace NUMINAMATH_CALUDE_translation_theorem_l3936_393674

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a translation in 2D space -/
structure Translation where
  dx : ℝ
  dy : ℝ

/-- Apply a translation to a point -/
def applyTranslation (t : Translation) (p : Point) : Point :=
  { x := p.x + t.dx, y := p.y + t.dy }

theorem translation_theorem (A B C D : Point) (t : Translation) :
  A.x = -1 ∧ A.y = 4 ∧
  C.x = 4 ∧ C.y = 7 ∧
  B.x = -4 ∧ B.y = -1 ∧
  C = applyTranslation t A ∧
  D = applyTranslation t B →
  D.x = 1 ∧ D.y = 2 := by
  sorry

end NUMINAMATH_CALUDE_translation_theorem_l3936_393674


namespace NUMINAMATH_CALUDE_melanie_dimes_count_l3936_393602

/-- Calculates the final number of dimes Melanie has -/
def final_dimes (initial : ℕ) (given_away : ℕ) (received : ℕ) : ℕ :=
  initial - given_away + received

/-- Theorem: The final number of dimes is correct given the problem conditions -/
theorem melanie_dimes_count : final_dimes 8 7 4 = 5 := by
  sorry

end NUMINAMATH_CALUDE_melanie_dimes_count_l3936_393602


namespace NUMINAMATH_CALUDE_fruit_box_ratio_l3936_393652

/-- Proves that the ratio of peaches to oranges is 1:2 given the conditions of the fruit box problem -/
theorem fruit_box_ratio : 
  ∀ (total_fruits oranges apples peaches : ℕ),
  total_fruits = 56 →
  oranges = total_fruits / 4 →
  apples = 35 →
  apples = 5 * peaches →
  (peaches : ℚ) / oranges = 1 / 2 := by
sorry

end NUMINAMATH_CALUDE_fruit_box_ratio_l3936_393652


namespace NUMINAMATH_CALUDE_warehouse_temp_restoration_time_l3936_393646

def initial_temp : ℝ := 43
def increase_rate : ℝ := 8
def outage_duration : ℝ := 3
def decrease_rate : ℝ := 4

theorem warehouse_temp_restoration_time :
  let total_increase : ℝ := increase_rate * outage_duration
  let restoration_time : ℝ := total_increase / decrease_rate
  restoration_time = 6 := by sorry

end NUMINAMATH_CALUDE_warehouse_temp_restoration_time_l3936_393646


namespace NUMINAMATH_CALUDE_opposite_silver_is_orange_l3936_393647

-- Define the colors
inductive Color
| Orange | Blue | Yellow | Black | Silver | Pink

-- Define the cube faces
inductive Face
| Top | Bottom | Front | Back | Left | Right

-- Define the cube structure
structure Cube where
  color : Face → Color

-- Define the three views of the cube
def view1 (c : Cube) : Prop :=
  c.color Face.Top = Color.Black ∧
  c.color Face.Front = Color.Blue ∧
  c.color Face.Right = Color.Yellow

def view2 (c : Cube) : Prop :=
  c.color Face.Top = Color.Black ∧
  c.color Face.Front = Color.Pink ∧
  c.color Face.Right = Color.Yellow

def view3 (c : Cube) : Prop :=
  c.color Face.Top = Color.Black ∧
  c.color Face.Front = Color.Silver ∧
  c.color Face.Right = Color.Yellow

-- Define the theorem
theorem opposite_silver_is_orange (c : Cube) :
  view1 c → view2 c → view3 c →
  c.color Face.Back = Color.Orange :=
sorry

end NUMINAMATH_CALUDE_opposite_silver_is_orange_l3936_393647


namespace NUMINAMATH_CALUDE_unique_positive_number_sum_with_square_l3936_393615

theorem unique_positive_number_sum_with_square : ∃! x : ℝ, x > 0 ∧ x^2 + x = 156 := by
  sorry

end NUMINAMATH_CALUDE_unique_positive_number_sum_with_square_l3936_393615


namespace NUMINAMATH_CALUDE_inequality_proof_l3936_393668

theorem inequality_proof (a b c d : ℝ) :
  (a / (1 + 3*a)) + (b^2 / (1 + 3*b^2)) + (c^3 / (1 + 3*c^3)) + (d^4 / (1 + 3*d^4)) ≤ 1 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_l3936_393668


namespace NUMINAMATH_CALUDE_sqrt_121_equals_plus_minus_11_l3936_393678

theorem sqrt_121_equals_plus_minus_11 : ∀ (x : ℝ), x^2 = 121 ↔ x = 11 ∨ x = -11 := by
  sorry

end NUMINAMATH_CALUDE_sqrt_121_equals_plus_minus_11_l3936_393678


namespace NUMINAMATH_CALUDE_no_four_digit_number_divisible_by_94_sum_l3936_393653

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

def first_two_digits (n : ℕ) : ℕ := n / 100

def last_two_digits (n : ℕ) : ℕ := n % 100

theorem no_four_digit_number_divisible_by_94_sum :
  ¬ ∃ (n : ℕ), is_four_digit n ∧ 
    n % (first_two_digits n + last_two_digits n) = 0 ∧
    first_two_digits n + last_two_digits n = 94 := by
  sorry

end NUMINAMATH_CALUDE_no_four_digit_number_divisible_by_94_sum_l3936_393653


namespace NUMINAMATH_CALUDE_intersection_A_complement_B_l3936_393660

def U : Set Nat := {1, 2, 3, 4, 5}
def A : Set Nat := {1, 2}
def B : Set Nat := {2, 3}

theorem intersection_A_complement_B : A ∩ (U \ B) = {1} := by sorry

end NUMINAMATH_CALUDE_intersection_A_complement_B_l3936_393660


namespace NUMINAMATH_CALUDE_max_value_x_plus_2y_l3936_393687

theorem max_value_x_plus_2y (x y : ℝ) (h : x^2 + 4*y^2 - 2*x*y = 4) :
  ∃ (M : ℝ), M = 4 ∧ ∀ (z : ℝ), z = x + 2*y → z ≤ M :=
by sorry

end NUMINAMATH_CALUDE_max_value_x_plus_2y_l3936_393687


namespace NUMINAMATH_CALUDE_evaluate_expression_l3936_393649

theorem evaluate_expression : (24 ^ 40) / (72 ^ 20) = 2 ^ 60 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l3936_393649


namespace NUMINAMATH_CALUDE_vector_parallel_condition_l3936_393679

/-- Two-dimensional vector type -/
def Vector2D := ℝ × ℝ

/-- Check if two vectors are parallel -/
def are_parallel (v w : Vector2D) : Prop :=
  ∃ (k : ℝ), v.1 * w.2 = k * v.2 * w.1

theorem vector_parallel_condition (x : ℝ) : 
  let a : Vector2D := (1, 2)
  let b : Vector2D := (-2, x)
  are_parallel (a.1 + b.1, a.2 + b.2) (a.1 - b.1, a.2 - b.2) → x = -4 := by
  sorry

end NUMINAMATH_CALUDE_vector_parallel_condition_l3936_393679


namespace NUMINAMATH_CALUDE_cubic_polynomial_c_value_l3936_393642

/-- A cubic polynomial function with integer coefficients -/
def f (a b c : ℤ) (x : ℝ) : ℝ := x^3 + a*x^2 + b*x + c

/-- Theorem stating that under given conditions, c must equal 16 -/
theorem cubic_polynomial_c_value (a b c : ℤ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) :
  f a b c a = a^3 → f a b c b = b^3 → c = 16 := by
  sorry


end NUMINAMATH_CALUDE_cubic_polynomial_c_value_l3936_393642


namespace NUMINAMATH_CALUDE_jenna_concert_spending_percentage_l3936_393684

/-- Proves that Jenna spends 10% of her monthly salary on a concert outing -/
theorem jenna_concert_spending_percentage :
  let concert_ticket_cost : ℚ := 181
  let drink_ticket_cost : ℚ := 7
  let num_drink_tickets : ℕ := 5
  let hourly_wage : ℚ := 18
  let weekly_hours : ℕ := 30
  let weeks_per_month : ℕ := 4

  let total_outing_cost : ℚ := concert_ticket_cost + drink_ticket_cost * num_drink_tickets
  let weekly_salary : ℚ := hourly_wage * weekly_hours
  let monthly_salary : ℚ := weekly_salary * weeks_per_month
  let spending_percentage : ℚ := total_outing_cost / monthly_salary * 100

  spending_percentage = 10 :=
by
  sorry


end NUMINAMATH_CALUDE_jenna_concert_spending_percentage_l3936_393684


namespace NUMINAMATH_CALUDE_optimal_production_time_l3936_393658

/-- The number of workers -/
def total_workers : ℕ := 50

/-- The number of products to be produced -/
def total_products : ℕ := 150

/-- The number of type A components per product -/
def type_a_per_product : ℕ := 3

/-- The number of type B components per product -/
def type_b_per_product : ℕ := 1

/-- The number of type A components a worker can process per hour -/
def type_a_rate : ℕ := 5

/-- The number of type B components a worker can process per hour -/
def type_b_rate : ℕ := 3

/-- The time required to process type A components -/
def f (x : ℕ) : ℚ := 90 / x

/-- The time required to process type B components -/
def g (x : ℕ) : ℚ := 50 / (50 - x)

/-- The total production time -/
def h (x : ℕ) : ℚ := max (f x) (g x)

theorem optimal_production_time :
  ∃ (x : ℕ), x > 0 ∧ x < total_workers ∧
  (∀ (y : ℕ), y > 0 ∧ y < total_workers → h x ≤ h y) ∧
  h x = 45 / 16 :=
sorry

end NUMINAMATH_CALUDE_optimal_production_time_l3936_393658


namespace NUMINAMATH_CALUDE_sum_of_max_min_g_l3936_393637

-- Define the function g(x)
def g (x : ℝ) : ℝ := |x - 1| + |x - 5| - |3*x - 9|

-- Define the domain
def domain (x : ℝ) : Prop := 1 ≤ x ∧ x ≤ 10

-- Theorem statement
theorem sum_of_max_min_g :
  ∃ (max_g min_g : ℝ),
    (∀ x, domain x → g x ≤ max_g) ∧
    (∃ x, domain x ∧ g x = max_g) ∧
    (∀ x, domain x → min_g ≤ g x) ∧
    (∃ x, domain x ∧ g x = min_g) ∧
    max_g + min_g = 1 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_max_min_g_l3936_393637


namespace NUMINAMATH_CALUDE_marks_money_l3936_393639

/-- The total value in cents of a collection of dimes and nickels -/
def total_value (num_dimes num_nickels : ℕ) : ℕ :=
  10 * num_dimes + 5 * num_nickels

/-- Theorem: Mark's total money is 90 cents -/
theorem marks_money :
  let num_dimes := 5
  let num_nickels := num_dimes + 3
  total_value num_dimes num_nickels = 90 := by
  sorry

end NUMINAMATH_CALUDE_marks_money_l3936_393639


namespace NUMINAMATH_CALUDE_min_initial_coins_l3936_393667

/-- Represents the game state at each round -/
structure GameState where
  huanhuan : ℕ
  lele : ℕ

/-- Represents the game with initial state and two rounds -/
structure Game where
  initial : GameState
  first_round : ℕ
  second_round : ℕ

/-- Checks if the game satisfies all the given conditions -/
def valid_game (g : Game) : Prop :=
  g.initial.huanhuan = 7 * g.initial.lele ∧
  g.initial.huanhuan + g.first_round = 6 * (g.initial.lele + g.first_round) ∧
  g.initial.huanhuan + g.first_round + g.second_round = 
    5 * (g.initial.lele + g.first_round + g.second_round)

/-- Theorem stating the minimum number of gold coins Huanhuan had at the beginning -/
theorem min_initial_coins (g : Game) (h : valid_game g) : g.initial.huanhuan ≥ 70 := by
  sorry

#check min_initial_coins

end NUMINAMATH_CALUDE_min_initial_coins_l3936_393667


namespace NUMINAMATH_CALUDE_fresh_grapes_water_content_l3936_393644

/-- Percentage of water in dried grapes -/
def dried_water_percentage : ℝ := 20

/-- Weight of fresh grapes in kg -/
def fresh_weight : ℝ := 40

/-- Weight of dried grapes in kg -/
def dried_weight : ℝ := 10

/-- Percentage of water in fresh grapes -/
def fresh_water_percentage : ℝ := 80

theorem fresh_grapes_water_content :
  (1 - fresh_water_percentage / 100) * fresh_weight = (1 - dried_water_percentage / 100) * dried_weight :=
sorry

end NUMINAMATH_CALUDE_fresh_grapes_water_content_l3936_393644


namespace NUMINAMATH_CALUDE_set_M_equals_one_two_three_l3936_393692

def M : Set ℤ := {a | 0 < 2*a - 1 ∧ 2*a - 1 ≤ 5}

theorem set_M_equals_one_two_three : M = {1, 2, 3} := by
  sorry

end NUMINAMATH_CALUDE_set_M_equals_one_two_three_l3936_393692


namespace NUMINAMATH_CALUDE_fraction_sum_l3936_393605

theorem fraction_sum (m n : ℚ) (h : n / m = 3 / 7) : (m + n) / m = 10 / 7 := by
  sorry

end NUMINAMATH_CALUDE_fraction_sum_l3936_393605


namespace NUMINAMATH_CALUDE_inconsistent_average_and_sum_l3936_393655

theorem inconsistent_average_and_sum :
  let numbers : List ℕ := [54, 55, 57, 58, 62, 62, 63, 65, 65]
  let average : ℕ := 60
  let total_sum : ℕ := average * numbers.length
  let sum_of_numbers : ℕ := numbers.sum
  sum_of_numbers > total_sum := by sorry

end NUMINAMATH_CALUDE_inconsistent_average_and_sum_l3936_393655


namespace NUMINAMATH_CALUDE_sum_equals_three_l3936_393638

/-- The largest proper fraction with denominator 9 -/
def largest_proper_fraction : ℚ := 8/9

/-- The smallest improper fraction with denominator 9 -/
def smallest_improper_fraction : ℚ := 9/9

/-- The smallest mixed number with fractional part having denominator 9 -/
def smallest_mixed_number : ℚ := 1 + 1/9

/-- The sum of the largest proper fraction, smallest improper fraction, and smallest mixed number -/
def sum_of_fractions : ℚ := largest_proper_fraction + smallest_improper_fraction + smallest_mixed_number

theorem sum_equals_three : sum_of_fractions = 3 := by
  sorry

end NUMINAMATH_CALUDE_sum_equals_three_l3936_393638


namespace NUMINAMATH_CALUDE_remainder_theorem_l3936_393617

theorem remainder_theorem (r : ℤ) : (r^17 + 1) % (r - 1) = 2 := by
  sorry

end NUMINAMATH_CALUDE_remainder_theorem_l3936_393617


namespace NUMINAMATH_CALUDE_intersection_A_B_l3936_393695

-- Define set A
def A : Set ℝ := {x | x ≥ 1}

-- Define set B
def B : Set ℝ := {y | y ≤ 2}

-- Theorem statement
theorem intersection_A_B : A ∩ B = {x | 1 ≤ x ∧ x ≤ 2} := by
  sorry

end NUMINAMATH_CALUDE_intersection_A_B_l3936_393695


namespace NUMINAMATH_CALUDE_problem_1_l3936_393656

theorem problem_1 (x : ℝ) : (x - 1)^2 + x*(3 - x) = x + 1 := by
  sorry

end NUMINAMATH_CALUDE_problem_1_l3936_393656


namespace NUMINAMATH_CALUDE_two_variables_scatter_plot_l3936_393623

-- Define a type for statistical variables
def StatisticalVariable : Type := ℝ

-- Define a type for a dataset of two variables
def Dataset : Type := List (StatisticalVariable × StatisticalVariable)

-- Statement: Any two statistical variables can be represented with a scatter plot
theorem two_variables_scatter_plot (data : Dataset) :
  ∃ (scatter_plot : Dataset → Bool), scatter_plot data = true :=
sorry

end NUMINAMATH_CALUDE_two_variables_scatter_plot_l3936_393623


namespace NUMINAMATH_CALUDE_average_speed_round_trip_l3936_393685

theorem average_speed_round_trip (outbound_speed inbound_speed : ℝ) 
  (h1 : outbound_speed = 130)
  (h2 : inbound_speed = 88)
  (h3 : outbound_speed > 0)
  (h4 : inbound_speed > 0) :
  (2 * outbound_speed * inbound_speed) / (outbound_speed + inbound_speed) = 105 := by
sorry

end NUMINAMATH_CALUDE_average_speed_round_trip_l3936_393685


namespace NUMINAMATH_CALUDE_don_tiles_per_minute_l3936_393691

/-- The number of tiles Don can paint per minute -/
def D : ℕ := sorry

/-- The number of tiles Ken can paint per minute -/
def ken_tiles : ℕ := D + 2

/-- The number of tiles Laura can paint per minute -/
def laura_tiles : ℕ := 2 * (D + 2)

/-- The number of tiles Kim can paint per minute -/
def kim_tiles : ℕ := 2 * (D + 2) - 3

/-- The total number of tiles painted by all four people in 15 minutes -/
def total_tiles : ℕ := 375

theorem don_tiles_per_minute :
  D + ken_tiles + laura_tiles + kim_tiles = total_tiles / 15 ∧ D = 3 := by sorry

end NUMINAMATH_CALUDE_don_tiles_per_minute_l3936_393691


namespace NUMINAMATH_CALUDE_fraction_evaluation_l3936_393648

theorem fraction_evaluation : (450 : ℚ) / (6 * 5 - 10 / 2) = 18 := by
  sorry

end NUMINAMATH_CALUDE_fraction_evaluation_l3936_393648


namespace NUMINAMATH_CALUDE_unique_determinable_score_l3936_393697

/-- The AHSME scoring system and constraints -/
structure AHSME where
  total_questions : ℕ
  score : ℕ
  correct : ℕ
  wrong : ℕ
  score_formula : score = 30 + 4 * correct - wrong
  total_answered : correct + wrong ≤ total_questions

/-- The uniqueness of the score for determining correct answers -/
def is_unique_determinable_score (s : ℕ) : Prop :=
  s > 80 ∧
  ∃! (exam : AHSME),
    exam.total_questions = 30 ∧
    exam.score = s ∧
    ∀ (s' : ℕ), 80 < s' ∧ s' < s →
      ¬∃! (exam' : AHSME),
        exam'.total_questions = 30 ∧
        exam'.score = s'

/-- The theorem stating that 119 is the unique score that satisfies the conditions -/
theorem unique_determinable_score :
  is_unique_determinable_score 119 :=
sorry

end NUMINAMATH_CALUDE_unique_determinable_score_l3936_393697


namespace NUMINAMATH_CALUDE_mark_payment_l3936_393681

def bread_cost : ℚ := 21/5
def cheese_cost : ℚ := 41/20
def nickel_value : ℚ := 1/20
def dime_value : ℚ := 1/10
def quarter_value : ℚ := 1/4
def num_nickels : ℕ := 8

theorem mark_payment (total_cost change payment : ℚ) :
  total_cost = bread_cost + cheese_cost →
  change = num_nickels * nickel_value + dime_value + quarter_value →
  payment = total_cost + change →
  payment = 7 := by sorry

end NUMINAMATH_CALUDE_mark_payment_l3936_393681


namespace NUMINAMATH_CALUDE_arithmetic_sequence_line_passes_through_point_l3936_393666

/-- Given that A, B, and C form an arithmetic sequence,
    prove that the line Ax + By + C = 0 passes through the point (1, -2) -/
theorem arithmetic_sequence_line_passes_through_point
  (A B C : ℝ) (h : 2 * B = A + C) :
  A * 1 + B * (-2) + C = 0 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_line_passes_through_point_l3936_393666


namespace NUMINAMATH_CALUDE_solution_p_proportion_l3936_393620

/-- Represents a solution mixture with lemonade and carbonated water -/
structure Solution where
  lemonade : ℝ
  carbonated_water : ℝ
  sum_to_one : lemonade + carbonated_water = 1

/-- The final mixture of solutions P and Q -/
structure Mixture where
  p : ℝ
  q : ℝ
  sum_to_one : p + q = 1

/-- Given two solutions and their mixture, prove that the proportion of Solution P is 0.4 -/
theorem solution_p_proportion
  (P : Solution)
  (Q : Solution)
  (M : Mixture)
  (h_P : P.carbonated_water = 0.8)
  (h_Q : Q.carbonated_water = 0.55)
  (h_M : P.carbonated_water * M.p + Q.carbonated_water * M.q = 0.65) :
  M.p = 0.4 := by
sorry

end NUMINAMATH_CALUDE_solution_p_proportion_l3936_393620


namespace NUMINAMATH_CALUDE_largest_invertible_interval_containing_two_l3936_393654

-- Define the function g
def g (x : ℝ) : ℝ := 3 * x^2 - 6 * x - 8

-- State the theorem
theorem largest_invertible_interval_containing_two :
  ∃ (a : ℝ), a ≤ 2 ∧ 
  (∀ (x y : ℝ), a ≤ x ∧ x < y → g x < g y) ∧
  (∀ (b : ℝ), b < a → ¬(∀ (x y : ℝ), b ≤ x ∧ x < y → g x < g y)) :=
by sorry

end NUMINAMATH_CALUDE_largest_invertible_interval_containing_two_l3936_393654


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l3936_393676

theorem quadratic_inequality_solution_set (k : ℝ) : 
  (∀ x : ℝ, k * x^2 + 2 * k * x - 1 < 0) ↔ (-1 < k ∧ k ≤ 0) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l3936_393676


namespace NUMINAMATH_CALUDE_polygon_sides_count_l3936_393688

theorem polygon_sides_count (n : ℕ) : n > 2 →
  (n - 2) * 180 = 4 * 360 → n = 10 := by
  sorry

end NUMINAMATH_CALUDE_polygon_sides_count_l3936_393688


namespace NUMINAMATH_CALUDE_nonagon_diagonal_count_l3936_393621

/-- The number of sides in a nonagon -/
def nonagon_sides : ℕ := 9

/-- The number of distinct diagonals in a convex nonagon -/
def nonagon_diagonals : ℕ := 27

/-- Theorem: The number of distinct diagonals in a convex nonagon is 27 -/
theorem nonagon_diagonal_count : nonagon_diagonals = 27 := by
  sorry

end NUMINAMATH_CALUDE_nonagon_diagonal_count_l3936_393621


namespace NUMINAMATH_CALUDE_intersection_in_square_l3936_393675

-- Define the trajectory function
def trajectory (x : ℝ) : ℝ :=
  (((x^5 - 2013)^5 - 2013)^5 - 2013)^5

-- Define the radar line function
def radar_line (x : ℝ) : ℝ :=
  x + 2013

-- Define the function for the difference between trajectory and radar line
def intersection_function (x : ℝ) : ℝ :=
  trajectory x - radar_line x

-- Theorem statement
theorem intersection_in_square :
  ∃ (x y : ℝ), 
    intersection_function x = 0 ∧ 
    4 ≤ x ∧ x < 5 ∧
    2017 ≤ y ∧ y < 2018 ∧
    y = radar_line x :=
sorry

end NUMINAMATH_CALUDE_intersection_in_square_l3936_393675


namespace NUMINAMATH_CALUDE_cans_for_reduced_people_l3936_393608

/-- Given 600 cans feed 40 people, proves the number of cans needed for 30% fewer people is 420 -/
theorem cans_for_reduced_people (total_cans : ℕ) (original_people : ℕ) (reduction_percent : ℚ) : 
  total_cans = 600 → 
  original_people = 40 → 
  reduction_percent = 30 / 100 →
  (total_cans / original_people : ℚ) * (original_people * (1 - reduction_percent) : ℚ) = 420 := by
  sorry

end NUMINAMATH_CALUDE_cans_for_reduced_people_l3936_393608


namespace NUMINAMATH_CALUDE_discount_percentage_l3936_393627

def original_price : ℝ := 6
def num_bags : ℕ := 2
def total_spent : ℝ := 3

theorem discount_percentage : 
  (1 - total_spent / (original_price * num_bags)) * 100 = 75 := by
  sorry

end NUMINAMATH_CALUDE_discount_percentage_l3936_393627


namespace NUMINAMATH_CALUDE_certain_number_proof_l3936_393609

theorem certain_number_proof (x : ℝ) : 3 * (x + 8) = 36 → x = 4 := by
  sorry

end NUMINAMATH_CALUDE_certain_number_proof_l3936_393609


namespace NUMINAMATH_CALUDE_four_different_suits_count_l3936_393699

/-- Represents a standard deck of cards -/
structure Deck :=
  (cards : Fin 52)

/-- Represents a suit in a deck of cards -/
inductive Suit
  | Hearts | Diamonds | Clubs | Spades

/-- Number of suits in a standard deck -/
def num_suits : Nat := 4

/-- Number of cards in each suit -/
def cards_per_suit : Nat := 13

/-- 
Theorem: The number of ways to choose 4 cards from a standard deck of 52 cards, 
where all four cards must be of different suits and the order doesn't matter, 
is equal to 28561.
-/
theorem four_different_suits_count (d : Deck) : 
  (cards_per_suit ^ num_suits) = 28561 := by
  sorry

end NUMINAMATH_CALUDE_four_different_suits_count_l3936_393699


namespace NUMINAMATH_CALUDE_triangle_third_side_bounds_l3936_393664

theorem triangle_third_side_bounds (a b : ℝ) (ha : a = 7) (hb : b = 11) :
  let c_min := Int.ceil (max (b - a) (a - b))
  let c_max := Int.floor (a + b - 1)
  (c_min = 5 ∧ c_max = 17) := by sorry

end NUMINAMATH_CALUDE_triangle_third_side_bounds_l3936_393664


namespace NUMINAMATH_CALUDE_trip_time_is_ten_weeks_l3936_393622

/-- Calculates the total time spent on a trip visiting three countries -/
def totalTripTime (firstStay : ℕ) (otherStaysMultiplier : ℕ) : ℕ :=
  firstStay + 2 * otherStaysMultiplier * firstStay

/-- Proves that the total trip time is 10 weeks given the specified conditions -/
theorem trip_time_is_ten_weeks :
  totalTripTime 2 2 = 10 := by
  sorry

#eval totalTripTime 2 2

end NUMINAMATH_CALUDE_trip_time_is_ten_weeks_l3936_393622


namespace NUMINAMATH_CALUDE_ordering_abc_l3936_393643

theorem ordering_abc (a b c : ℝ) : a = Real.exp 0.1 - 1 → b = 0.1 → c = Real.log 1.1 → c < b ∧ b < a := by
  sorry

end NUMINAMATH_CALUDE_ordering_abc_l3936_393643


namespace NUMINAMATH_CALUDE_p_necessary_not_sufficient_for_q_l3936_393651

-- Define geometric bodies
structure GeometricBody where
  height : ℝ
  crossSectionalArea : ℝ → ℝ
  volume : ℝ

-- Define the Gougu Principle
def gougu_principle (A B : GeometricBody) : Prop :=
  A.height = B.height →
  (∀ h, 0 ≤ h ∧ h ≤ A.height → A.crossSectionalArea h = B.crossSectionalArea h) →
  A.volume = B.volume

-- Define the relationship between p and q
theorem p_necessary_not_sufficient_for_q (A B : GeometricBody) :
  (∀ h, 0 ≤ h ∧ h ≤ A.height → A.crossSectionalArea h = B.crossSectionalArea h) →
  A.volume = B.volume ∧
  ¬(A.volume = B.volume →
    ∀ h, 0 ≤ h ∧ h ≤ A.height → A.crossSectionalArea h = B.crossSectionalArea h) :=
by
  sorry

end NUMINAMATH_CALUDE_p_necessary_not_sufficient_for_q_l3936_393651


namespace NUMINAMATH_CALUDE_complex_fraction_evaluation_l3936_393601

theorem complex_fraction_evaluation : 
  (2 + 2)^2 / 2^2 * (3 + 3 + 3 + 3)^3 / (3 + 3 + 3)^3 * (6 + 6 + 6 + 6 + 6 + 6)^6 / (6 + 6 + 6 + 6)^6 = 108 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_evaluation_l3936_393601


namespace NUMINAMATH_CALUDE_max_advancing_players_16_10_l3936_393689

/-- Represents a chess tournament -/
structure ChessTournament where
  players : ℕ
  points_to_advance : ℕ

/-- Calculates the total number of games in a round-robin tournament -/
def total_games (t : ChessTournament) : ℕ :=
  t.players * (t.players - 1) / 2

/-- Calculates the total points awarded in the tournament -/
def total_points (t : ChessTournament) : ℕ :=
  total_games t

/-- Defines the maximum number of players that can advance -/
def max_advancing_players (t : ChessTournament) : ℕ :=
  11

/-- Theorem: In a 16-player tournament where players need at least 10 points to advance,
    the maximum number of players who can advance is 11 -/
theorem max_advancing_players_16_10 :
  ∀ t : ChessTournament,
    t.players = 16 →
    t.points_to_advance = 10 →
    max_advancing_players t = 11 :=
by sorry


end NUMINAMATH_CALUDE_max_advancing_players_16_10_l3936_393689


namespace NUMINAMATH_CALUDE_pascal_interior_sum_l3936_393677

/-- Sum of interior numbers in a row of Pascal's Triangle -/
def interior_sum (n : ℕ) : ℕ := 2^(n-1) - 2

/-- The problem statement -/
theorem pascal_interior_sum :
  interior_sum 5 = 14 →
  interior_sum 6 = 30 →
  interior_sum 8 = 126 := by
sorry

end NUMINAMATH_CALUDE_pascal_interior_sum_l3936_393677


namespace NUMINAMATH_CALUDE_divisible_sequence_eventually_periodic_l3936_393641

/-- A sequence of positive integers satisfying the given divisibility property -/
def DivisibleSequence (a : ℕ → ℕ) : Prop :=
  ∀ n m : ℕ, (a (n + 2*m)) ∣ (a n + a (n + m))

/-- The property of eventual periodicity for a sequence -/
def EventuallyPeriodic (a : ℕ → ℕ) : Prop :=
  ∃ N d : ℕ, d > 0 ∧ ∀ n : ℕ, n > N → a n = a (n + d)

/-- The main theorem: A divisible sequence is eventually periodic -/
theorem divisible_sequence_eventually_periodic (a : ℕ → ℕ) 
  (h : DivisibleSequence a) : EventuallyPeriodic a := by
  sorry

end NUMINAMATH_CALUDE_divisible_sequence_eventually_periodic_l3936_393641


namespace NUMINAMATH_CALUDE_parabola_point_relationship_l3936_393616

/-- Theorem: For a parabola y = ax² - 4ax + c with a > 0, and points A(2, y₁), B(3, y₂), and C(-1, y₃) on this parabola, the relationship y₁ < y₂ < y₃ holds. -/
theorem parabola_point_relationship (a c y₁ y₂ y₃ : ℝ) (ha : a > 0) 
  (hA : y₁ = a * 2^2 - 4 * a * 2 + c)
  (hB : y₂ = a * 3^2 - 4 * a * 3 + c)
  (hC : y₃ = a * (-1)^2 - 4 * a * (-1) + c) :
  y₁ < y₂ ∧ y₂ < y₃ := by
  sorry

#check parabola_point_relationship

end NUMINAMATH_CALUDE_parabola_point_relationship_l3936_393616


namespace NUMINAMATH_CALUDE_order_of_numbers_l3936_393657

def Ψ : ℤ := (1 + 2 - 3 - 4 + 5 + 6 - 7 - 8 + (-2012)) / 2

def Ω : ℤ := 1 - 2 + 3 - 4 + 2014

def Θ : ℤ := 1 - 3 + 5 - 7 + 2015

theorem order_of_numbers : Θ < Ω ∧ Ω < Ψ :=
  sorry

end NUMINAMATH_CALUDE_order_of_numbers_l3936_393657


namespace NUMINAMATH_CALUDE_greatest_prime_factor_of_factorial_sum_l3936_393600

def factorial (n : ℕ) : ℕ := (List.range n).foldl (· * ·) 1

def is_prime (p : ℕ) : Prop := p > 1 ∧ ∀ m : ℕ, m > 0 → m < p → p % m ≠ 0

theorem greatest_prime_factor_of_factorial_sum :
  ∃ (p : ℕ), is_prime p ∧ 
    p ∣ (factorial 15 + factorial 18) ∧ 
    ∀ (q : ℕ), is_prime q → q ∣ (factorial 15 + factorial 18) → q ≤ p :=
  sorry

end NUMINAMATH_CALUDE_greatest_prime_factor_of_factorial_sum_l3936_393600


namespace NUMINAMATH_CALUDE_sqrt_8000_minus_50_cube_l3936_393659

theorem sqrt_8000_minus_50_cube (a b : ℕ+) :
  (Real.sqrt 8000 - 50 : ℝ) = (Real.sqrt a.val - b.val)^3 →
  a.val + b.val = 16 := by
sorry

end NUMINAMATH_CALUDE_sqrt_8000_minus_50_cube_l3936_393659


namespace NUMINAMATH_CALUDE_H_upper_bound_l3936_393633

open Real

noncomputable def f (x : ℝ) : ℝ := x + log x

noncomputable def H (x m : ℝ) : ℝ := f x - log (exp x - 1)

theorem H_upper_bound {m : ℝ} (hm : m > 0) :
  ∀ x, 0 < x → x < m → H x m < m / 2 := by sorry

end NUMINAMATH_CALUDE_H_upper_bound_l3936_393633


namespace NUMINAMATH_CALUDE_binomial_coefficient_sum_l3936_393628

theorem binomial_coefficient_sum (x a : ℝ) (x_nonzero : x ≠ 0) (a_nonzero : a ≠ 0) :
  (Finset.range 7).sum (λ k => Nat.choose 6 k) = 64 := by
  sorry

end NUMINAMATH_CALUDE_binomial_coefficient_sum_l3936_393628


namespace NUMINAMATH_CALUDE_consecutive_even_ages_l3936_393665

def is_even (n : ℕ) : Prop := ∃ k, n = 2 * k

theorem consecutive_even_ages (a b c : ℕ) 
  (h1 : is_even a)
  (h2 : is_even b)
  (h3 : is_even c)
  (h4 : b = a + 2)
  (h5 : c = b + 2)
  (h6 : a + b + c = 48) :
  a = 14 ∧ c = 18 := by
sorry

end NUMINAMATH_CALUDE_consecutive_even_ages_l3936_393665


namespace NUMINAMATH_CALUDE_inequality_condition_l3936_393670

theorem inequality_condition (a : ℝ) : 
  (∀ x, -2 < x ∧ x < -1 → (x + a) * (x + 1) < 0) ∧ 
  (∃ x, (x + a) * (x + 1) < 0 ∧ (x ≤ -2 ∨ x ≥ -1)) ↔ 
  a > 2 := by sorry

end NUMINAMATH_CALUDE_inequality_condition_l3936_393670


namespace NUMINAMATH_CALUDE_camilla_original_strawberry_l3936_393693

/-- Represents the number of strawberry jelly beans Camilla originally had. -/
def original_strawberry : ℕ := sorry

/-- Represents the number of grape jelly beans Camilla originally had. -/
def original_grape : ℕ := sorry

/-- States that Camilla originally had three times as many strawberry jelly beans as grape jelly beans. -/
axiom initial_ratio : original_strawberry = 3 * original_grape

/-- States that after eating 12 strawberry jelly beans and 8 grape jelly beans, 
    Camilla now has four times as many strawberry jelly beans as grape jelly beans. -/
axiom final_ratio : original_strawberry - 12 = 4 * (original_grape - 8)

/-- Theorem stating that Camilla originally had 60 strawberry jelly beans. -/
theorem camilla_original_strawberry : original_strawberry = 60 := by sorry

end NUMINAMATH_CALUDE_camilla_original_strawberry_l3936_393693


namespace NUMINAMATH_CALUDE_inequality_solution_l3936_393690

theorem inequality_solution (x : ℝ) : 
  x^3 - 3*x^2 - 4*x - 12 ≤ 0 ∧ 2*x + 6 > 0 → x ∈ Set.Icc (-2) 3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_solution_l3936_393690


namespace NUMINAMATH_CALUDE_nadine_pebbles_l3936_393603

def white_pebbles : ℕ := 20

def red_pebbles : ℕ := white_pebbles / 2

def total_pebbles : ℕ := white_pebbles + red_pebbles

theorem nadine_pebbles : total_pebbles = 30 := by
  sorry

end NUMINAMATH_CALUDE_nadine_pebbles_l3936_393603


namespace NUMINAMATH_CALUDE_abundant_product_l3936_393634

/-- Sum of divisors function -/
def sigma (n : ℕ) : ℕ := sorry

/-- A number is abundant if the sum of its divisors is greater than twice the number -/
def abundant (n : ℕ) : Prop := sigma n > 2 * n

/-- If a is abundant, then ab is abundant for any positive integer b -/
theorem abundant_product {a b : ℕ} (ha : a > 0) (hb : b > 0) (hab : abundant a) : abundant (a * b) := by
  sorry

end NUMINAMATH_CALUDE_abundant_product_l3936_393634


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_l3936_393694

theorem quadratic_inequality_solution (x : ℝ) :
  x^2 - 4*x > 12 ↔ x ∈ Set.Iio (-2) ∪ Set.Ioi 6 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_l3936_393694


namespace NUMINAMATH_CALUDE_cosine_odd_function_phi_l3936_393650

def is_odd_function (f : ℝ → ℝ) : Prop :=
  ∀ x, f (-x) = -f x

theorem cosine_odd_function_phi (φ : ℝ) :
  is_odd_function (λ x => Real.cos (x + φ + π/3)) → φ = π/6 ∨ ∃ k : ℤ, φ = k * π + π/6 :=
by sorry

end NUMINAMATH_CALUDE_cosine_odd_function_phi_l3936_393650


namespace NUMINAMATH_CALUDE_arithmetic_sequence_ninth_term_l3936_393607

/-- An arithmetic sequence is defined by its first term and common difference -/
structure ArithmeticSequence where
  firstTerm : ℤ
  commonDiff : ℤ

/-- The nth term of an arithmetic sequence -/
def ArithmeticSequence.nthTerm (seq : ArithmeticSequence) (n : ℕ) : ℤ :=
  seq.firstTerm + (n - 1) * seq.commonDiff

theorem arithmetic_sequence_ninth_term
  (seq : ArithmeticSequence)
  (h3 : seq.nthTerm 3 = 5)
  (h6 : seq.nthTerm 6 = 17) :
  seq.nthTerm 9 = 29 := by
  sorry

#check arithmetic_sequence_ninth_term

end NUMINAMATH_CALUDE_arithmetic_sequence_ninth_term_l3936_393607


namespace NUMINAMATH_CALUDE_tangent_line_2sinx_at_pi_l3936_393686

/-- The equation of the tangent line to y = 2sin(x) at (π, 0) is y = -2x + 2π -/
theorem tangent_line_2sinx_at_pi (x y : ℝ) : 
  (y = 2 * Real.sin x) → -- curve equation
  (y = -2 * (x - Real.pi) + 0) → -- point-slope form of tangent line
  (y = -2 * x + 2 * Real.pi) -- final equation of tangent line
  := by sorry

end NUMINAMATH_CALUDE_tangent_line_2sinx_at_pi_l3936_393686


namespace NUMINAMATH_CALUDE_car_speed_proof_l3936_393629

/-- 
Proves that a car traveling at a constant speed v km/h takes 2 seconds longer 
to travel 1 kilometer than it would at 450 km/h if and only if v = 360 km/h.
-/
theorem car_speed_proof (v : ℝ) : v > 0 → (
  (1 / v) * 3600 = (1 / 450) * 3600 + 2 ↔ v = 360
) := by sorry

end NUMINAMATH_CALUDE_car_speed_proof_l3936_393629


namespace NUMINAMATH_CALUDE_triangle_altitude_and_median_l3936_393619

/-- Triangle with vertices A(0,1), B(-2,0), and C(2,0) -/
structure Triangle where
  A : ℝ × ℝ := (0, 1)
  B : ℝ × ℝ := (-2, 0)
  C : ℝ × ℝ := (2, 0)

/-- The equation of a line in the form ax + by + c = 0 -/
structure LineEquation where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Altitude from A to AC -/
def altitude (t : Triangle) : LineEquation :=
  { a := 2, b := -1, c := 1 }

/-- Median from A to BC -/
def median (t : Triangle) : LineEquation :=
  { a := 1, b := 0, c := 0 }

theorem triangle_altitude_and_median (t : Triangle) :
  (altitude t = { a := 2, b := -1, c := 1 }) ∧
  (median t = { a := 1, b := 0, c := 0 }) := by
  sorry

end NUMINAMATH_CALUDE_triangle_altitude_and_median_l3936_393619
