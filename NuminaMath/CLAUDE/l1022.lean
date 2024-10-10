import Mathlib

namespace projectile_height_time_l1022_102248

theorem projectile_height_time (t : ℝ) : 
  (∃ t₁ t₂ : ℝ, t₁ < t₂ ∧ -4.9 * t₁^2 + 30 * t₁ = 35 ∧ -4.9 * t₂^2 + 30 * t₂ = 35) → 
  (∀ t' : ℝ, -4.9 * t'^2 + 30 * t' = 35 → t' ≥ 10/7) ∧
  -4.9 * (10/7)^2 + 30 * (10/7) = 35 :=
sorry

end projectile_height_time_l1022_102248


namespace card_position_retained_l1022_102256

theorem card_position_retained (n : ℕ) : 
  (∃ (total_cards : ℕ), 
    total_cards = 2 * n ∧ 
    201 ≤ n ∧ 
    (∀ (card : ℕ), card ≤ total_cards → 
      (card ≤ n → (card + n).mod 2 = 1) ∧ 
      (n < card → card.mod 2 = 0))) →
  201 = n :=
by sorry

end card_position_retained_l1022_102256


namespace bruce_fruit_purchase_total_l1022_102293

/-- Calculates the discounted price for a fruit purchase -/
def discountedPrice (quantity : ℕ) (pricePerKg : ℚ) (discountPercentage : ℚ) : ℚ :=
  let originalPrice := quantity * pricePerKg
  originalPrice - (originalPrice * discountPercentage / 100)

/-- Represents Bruce's fruit purchases -/
structure FruitPurchase where
  grapes : ℕ × ℚ × ℚ
  mangoes : ℕ × ℚ × ℚ
  oranges : ℕ × ℚ × ℚ
  apples : ℕ × ℚ × ℚ

/-- Calculates the total amount paid for all fruit purchases -/
def totalAmountPaid (purchase : FruitPurchase) : ℚ :=
  discountedPrice purchase.grapes.1 purchase.grapes.2.1 purchase.grapes.2.2 +
  discountedPrice purchase.mangoes.1 purchase.mangoes.2.1 purchase.mangoes.2.2 +
  discountedPrice purchase.oranges.1 purchase.oranges.2.1 purchase.oranges.2.2 +
  discountedPrice purchase.apples.1 purchase.apples.2.1 purchase.apples.2.2

theorem bruce_fruit_purchase_total :
  let purchase : FruitPurchase := {
    grapes := (9, 70, 10),
    mangoes := (7, 55, 5),
    oranges := (5, 45, 15),
    apples := (3, 80, 20)
  }
  totalAmountPaid purchase = 1316.25 := by
  sorry

end bruce_fruit_purchase_total_l1022_102293


namespace cubic_identity_l1022_102219

theorem cubic_identity (x : ℝ) : (2*x - 1)^3 = 5*x^3 + (3*x + 1)*(x^2 - x - 1) - 10*x^2 + 10*x := by
  sorry

end cubic_identity_l1022_102219


namespace billy_initial_dandelions_l1022_102205

/-- The number of dandelions Billy picked initially -/
def billy_initial : ℕ := sorry

/-- The number of dandelions George picked initially -/
def george_initial : ℕ := sorry

/-- The number of additional dandelions each person picked -/
def additional_picks : ℕ := 10

/-- The average number of dandelions picked -/
def average_picks : ℕ := 34

theorem billy_initial_dandelions :
  billy_initial = 36 ∧
  george_initial = billy_initial / 3 ∧
  (billy_initial + george_initial + 2 * additional_picks) / 2 = average_picks :=
sorry

end billy_initial_dandelions_l1022_102205


namespace largest_four_digit_divisible_by_6_l1022_102266

def is_four_digit (n : ℕ) : Prop := 1000 ≤ n ∧ n ≤ 9999

def divisible_by_6 (n : ℕ) : Prop := n % 6 = 0

theorem largest_four_digit_divisible_by_6 :
  ∀ n : ℕ, is_four_digit n → divisible_by_6 n → n ≤ 9996 :=
by sorry

end largest_four_digit_divisible_by_6_l1022_102266


namespace no_root_intersection_l1022_102265

theorem no_root_intersection : ∀ x : ℝ,
  (∃ y : ℝ, y = Real.sqrt x ∧ y = Real.sqrt (x - 6) + 1) →
  x^2 - 5*x + 6 ≠ 0 := by
  sorry

end no_root_intersection_l1022_102265


namespace fraction_equality_l1022_102218

theorem fraction_equality (p q : ℚ) (h : p / q = 4 / 5) : 
  18 / 7 + (2 * q - p) / ((14/5) * q) = 3 := by
  sorry

end fraction_equality_l1022_102218


namespace parabola_point_x_coord_l1022_102253

/-- The x-coordinate of a point on a parabola given its distance from the focus -/
theorem parabola_point_x_coord (x y : ℝ) : 
  y^2 = 4*x →  -- Point P(x,y) lies on the parabola y^2 = 4x
  (x - 1)^2 + y^2 = 3^2 →  -- Distance from P to focus (1,0) is 3
  x = 2 := by sorry

end parabola_point_x_coord_l1022_102253


namespace sqrt_difference_equals_negative_four_sqrt_five_l1022_102210

theorem sqrt_difference_equals_negative_four_sqrt_five :
  Real.sqrt (16 - 8 * Real.sqrt 5) - Real.sqrt (16 + 8 * Real.sqrt 5) = -4 * Real.sqrt 5 := by
  sorry

end sqrt_difference_equals_negative_four_sqrt_five_l1022_102210


namespace last_released_position_l1022_102225

/-- Represents the state of the ransom process -/
structure RansomState where
  remaining_captives : ℕ
  purses_on_table : ℕ
  last_released_position : ℕ

/-- Simulates the ransom process for Robin Hood's captives -/
def ransom_process (initial_captives : ℕ) : ℕ → RansomState := sorry

/-- Theorem stating the position of the last released captive based on the final number of purses -/
theorem last_released_position 
  (initial_captives : ℕ) 
  (final_purses : ℕ) :
  initial_captives = 7 →
  (final_purses = 28 → (ransom_process initial_captives final_purses).last_released_position = 7) ∧
  (final_purses = 27 → 
    ((ransom_process initial_captives final_purses).last_released_position = 6 ∨
     (ransom_process initial_captives final_purses).last_released_position = 7)) :=
by sorry

end last_released_position_l1022_102225


namespace linear_function_increasing_l1022_102275

/-- A linear function f(x) = mx + b where m > 0 is increasing -/
theorem linear_function_increasing (m b : ℝ) (h : m > 0) :
  Monotone (fun x => m * x + b) := by sorry

end linear_function_increasing_l1022_102275


namespace parabola_coefficient_sum_l1022_102280

/-- A parabola with equation y = ax^2 + bx + c -/
structure Parabola where
  a : ℝ
  b : ℝ
  c : ℝ

/-- The y-coordinate of a point on the parabola given its x-coordinate -/
def Parabola.y_coord (p : Parabola) (x : ℝ) : ℝ :=
  p.a * x^2 + p.b * x + p.c

/-- Predicate for a parabola having a vertical axis of symmetry -/
def has_vertical_axis_of_symmetry (p : Parabola) : Prop :=
  ∃ h : ℝ, ∀ x y : ℝ, p.y_coord (h + x) = p.y_coord (h - x)

theorem parabola_coefficient_sum (p : Parabola) :
  p.y_coord (-3) = 4 →  -- vertex condition
  has_vertical_axis_of_symmetry p →  -- vertical axis of symmetry
  p.y_coord (-1) = 16 →  -- point condition
  p.a + p.b + p.c = 52 := by
  sorry

end parabola_coefficient_sum_l1022_102280


namespace conference_tables_theorem_l1022_102214

/-- Represents the available table sizes -/
inductive TableSize
  | Four
  | Six
  | Eight

/-- Calculates the minimum number of tables needed -/
def minTablesNeeded (totalInvited : ℕ) (noShows : ℕ) (tableSizes : List TableSize) : ℕ :=
  sorry

/-- Theorem stating the minimum number of tables needed for the given problem -/
theorem conference_tables_theorem (totalInvited noShows : ℕ) (tableSizes : List TableSize) :
  totalInvited = 75 →
  noShows = 33 →
  tableSizes = [TableSize.Four, TableSize.Six, TableSize.Eight] →
  minTablesNeeded totalInvited noShows tableSizes = 6 :=
sorry

end conference_tables_theorem_l1022_102214


namespace gcd_182_98_l1022_102257

theorem gcd_182_98 : Nat.gcd 182 98 = 14 := by
  sorry

end gcd_182_98_l1022_102257


namespace geometric_sequence_ratio_l1022_102224

/-- An arithmetic sequence -/
def is_arithmetic (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

/-- A geometric sequence -/
def is_geometric (b : ℕ → ℝ) : Prop :=
  ∃ r : ℝ, ∀ n : ℕ, b (n + 1) = b n * r

/-- The sequence a_n + 2^n * b_n forms an arithmetic sequence for n = 1, 3, 5 -/
def special_sequence_arithmetic (a b : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, (a 3 + 4 * b 3) - (a 1 + 2 * b 1) = d ∧
            (a 5 + 8 * b 5) - (a 3 + 4 * b 3) = d

theorem geometric_sequence_ratio (a b : ℕ → ℝ) :
  is_arithmetic a →
  is_geometric b →
  special_sequence_arithmetic a b →
  b 3 * b 7 / (b 4 ^ 2) = 1 / 4 := by
  sorry

end geometric_sequence_ratio_l1022_102224


namespace family_children_count_l1022_102215

theorem family_children_count :
  ∀ (num_children : ℕ),
    (5 * (num_children + 3) + 2 * num_children + 4 * 3 = 55) →
    num_children = 4 :=
by
  sorry

end family_children_count_l1022_102215


namespace cookie_radius_l1022_102220

theorem cookie_radius (x y : ℝ) :
  (∃ r, r > 0 ∧ ∀ x y, x^2 + y^2 - 6*x + 10*y + 12 = 0 ↔ (x - 3)^2 + (y + 5)^2 = r^2) →
  (∃ r, r > 0 ∧ ∀ x y, x^2 + y^2 - 6*x + 10*y + 12 = 0 ↔ (x - 3)^2 + (y + 5)^2 = r^2 ∧ r = Real.sqrt 22) :=
by sorry

end cookie_radius_l1022_102220


namespace difference_in_balls_l1022_102287

/-- The number of packs of red bouncy balls Jill bought -/
def red_packs : ℕ := 5

/-- The number of packs of yellow bouncy balls Jill bought -/
def yellow_packs : ℕ := 4

/-- The number of bouncy balls in each package -/
def balls_per_pack : ℕ := 18

/-- The total number of red bouncy balls Jill bought -/
def total_red_balls : ℕ := red_packs * balls_per_pack

/-- The total number of yellow bouncy balls Jill bought -/
def total_yellow_balls : ℕ := yellow_packs * balls_per_pack

theorem difference_in_balls : total_red_balls - total_yellow_balls = 18 := by
  sorry

end difference_in_balls_l1022_102287


namespace sufficient_but_not_necessary_l1022_102284

theorem sufficient_but_not_necessary : 
  (∀ x₁ x₂ : ℝ, x₁ > 3 ∧ x₂ > 3 → x₁ * x₂ > 9 ∧ x₁ + x₂ > 6) ∧
  (∃ x₁ x₂ : ℝ, x₁ * x₂ > 9 ∧ x₁ + x₂ > 6 ∧ ¬(x₁ > 3 ∧ x₂ > 3)) :=
by sorry

end sufficient_but_not_necessary_l1022_102284


namespace expression_evaluation_l1022_102222

theorem expression_evaluation :
  let x : ℝ := 2
  let y : ℝ := 3
  let z : ℝ := 1
  x^2 + y^2 - z^2 + 2*x*y + 3*y*z = 33 := by sorry

end expression_evaluation_l1022_102222


namespace max_stores_visited_l1022_102286

theorem max_stores_visited (total_stores : ℕ) (total_visits : ℕ) (total_shoppers : ℕ) 
  (two_store_visitors : ℕ) (h1 : total_stores = 8) (h2 : total_visits = 23) 
  (h3 : total_shoppers = 12) (h4 : two_store_visitors = 8) 
  (h5 : two_store_visitors ≤ total_shoppers) 
  (h6 : 2 * two_store_visitors ≤ total_visits) :
  ∃ (max_visits : ℕ), max_visits ≤ 4 ∧ 
  ∀ (individual_visits : ℕ), individual_visits ≤ max_visits ∧
  (total_visits = 2 * two_store_visitors + 
    (total_shoppers - two_store_visitors) + 
    (individual_visits - 1)) :=
by sorry

end max_stores_visited_l1022_102286


namespace platform_length_l1022_102259

/-- The length of a platform given a goods train's speed, length, and time to cross the platform. -/
theorem platform_length (train_speed : ℝ) (train_length : ℝ) (crossing_time : ℝ) : 
  train_speed = 72 → 
  train_length = 280.0416 → 
  crossing_time = 26 → 
  (train_speed * 1000 / 3600 * crossing_time) - train_length = 239.9584 := by
  sorry

end platform_length_l1022_102259


namespace meal_cost_calculation_l1022_102216

theorem meal_cost_calculation (adults children : ℕ) (total_bill : ℚ) :
  adults = 2 →
  children = 5 →
  total_bill = 21 →
  total_bill / (adults + children : ℚ) = 3 := by
  sorry

end meal_cost_calculation_l1022_102216


namespace equation_has_real_root_l1022_102206

theorem equation_has_real_root (K : ℝ) (h : K ≠ 0) :
  ∃ x : ℝ, x = K^2 * (x - 1) * (x - 2) * (x - 3) :=
by sorry

end equation_has_real_root_l1022_102206


namespace inequalities_not_always_true_l1022_102254

theorem inequalities_not_always_true (x y a b : ℝ) 
  (hx : x ≠ 0) (hy : y ≠ 0) (ha : a ≠ 0) (hb : b ≠ 0)
  (hxa : abs x < abs a) (hyb : abs y > abs b) :
  ∃ (x' y' a' b' : ℝ), 
    x' ≠ 0 ∧ y' ≠ 0 ∧ a' ≠ 0 ∧ b' ≠ 0 ∧
    abs x' < abs a' ∧ abs y' > abs b' ∧
    ¬(abs (x' + y') < abs (a' + b')) ∧
    ¬(abs (x' - y') < abs (a' - b')) ∧
    ¬(abs (x' * y') < abs (a' * b')) ∧
    ¬(abs (x' / y') < abs (a' / b')) :=
by sorry

end inequalities_not_always_true_l1022_102254


namespace greatest_three_digit_number_mod_11_and_7_l1022_102243

theorem greatest_three_digit_number_mod_11_and_7 :
  ∃ n : ℕ, 
    100 ≤ n ∧ n ≤ 999 ∧ 
    n % 11 = 10 ∧ 
    n % 7 = 4 ∧
    (∀ m : ℕ, 100 ≤ m ∧ m ≤ 999 ∧ m % 11 = 10 ∧ m % 7 = 4 → m ≤ n) ∧
    n = 956 :=
by sorry

end greatest_three_digit_number_mod_11_and_7_l1022_102243


namespace inequality_system_solution_set_l1022_102233

theorem inequality_system_solution_set :
  let S := { x : ℝ | x - 1 < 7 ∧ 3 * x + 1 ≥ -2 }
  S = { x : ℝ | -1 ≤ x ∧ x < 8 } :=
by sorry

end inequality_system_solution_set_l1022_102233


namespace smallest_integer_satisfying_inequality_l1022_102283

theorem smallest_integer_satisfying_inequality :
  ∀ x : ℤ, x + 3 < 3*x - 4 → x ≥ 4 ∧ 4 + 3 < 3*4 - 4 :=
by
  sorry

end smallest_integer_satisfying_inequality_l1022_102283


namespace distance_ratio_forms_circle_l1022_102217

/-- Given points A(0,0) and B(1,0) on a plane, the set of all points M(x,y) such that 
    the distance from M to A is three times the distance from M to B forms a circle 
    with center (-1/8, 0) and radius 3/8. -/
theorem distance_ratio_forms_circle :
  ∀ (x y : ℝ),
    (Real.sqrt (x^2 + y^2) = 3 * Real.sqrt ((x-1)^2 + y^2)) →
    ((x + 1/8)^2 + y^2 = (3/8)^2) := by
  sorry

end distance_ratio_forms_circle_l1022_102217


namespace car_speed_problem_l1022_102296

-- Define the parameters of the problem
def initial_distance : ℝ := 10
def final_distance : ℝ := 8
def time : ℝ := 2.25
def speed_A : ℝ := 58

-- Define the speed of Car B as a variable
def speed_B : ℝ := 50

-- Theorem statement
theorem car_speed_problem :
  initial_distance + 
  speed_A * time = 
  speed_B * time + 
  initial_distance + 
  final_distance := by sorry

end car_speed_problem_l1022_102296


namespace fixed_point_of_exponential_function_l1022_102255

/-- The function f(x) = 1 + 2a^(x-1) has a fixed point at (1, 3), where a > 0 and a ≠ 1 -/
theorem fixed_point_of_exponential_function (a : ℝ) (ha_pos : a > 0) (ha_neq_one : a ≠ 1) :
  let f : ℝ → ℝ := λ x ↦ 1 + 2 * a^(x - 1)
  f 1 = 3 := by sorry

end fixed_point_of_exponential_function_l1022_102255


namespace fraction_difference_l1022_102211

theorem fraction_difference (p q : ℝ) (hp : 3 ≤ p ∧ p ≤ 10) (hq : 12 ≤ q ∧ q ≤ 21) :
  (10 / 12 : ℝ) - (3 / 21 : ℝ) = 29 / 42 := by sorry

end fraction_difference_l1022_102211


namespace trigonometric_identity_l1022_102213

theorem trigonometric_identity (α : Real) 
  (h1 : α > 0) 
  (h2 : α < Real.pi / 2) 
  (h3 : 3 * Real.cos (2 * α) + Real.sin α = 1) : 
  Real.sin (Real.pi - α) = 2/3 := by
sorry

end trigonometric_identity_l1022_102213


namespace insertPluses_l1022_102212

/-- The number of ones in the original number -/
def n : ℕ := 15

/-- The number of plus signs to be inserted -/
def k : ℕ := 9

/-- The number of spaces between the ones where plus signs can be inserted -/
def spaces : ℕ := n - 1

-- Statement of the theorem
theorem insertPluses : 
  (Nat.choose spaces k : ℕ) = (2002 : ℕ) :=
sorry

end insertPluses_l1022_102212


namespace range_of_f_l1022_102242

def f (x : ℝ) : ℝ := x^2 - 2*x + 9

theorem range_of_f :
  ∀ y ∈ Set.range f,
  (∃ x ∈ Set.Icc (-1 : ℝ) 2, f x = y) →
  y ∈ Set.Icc 8 12 :=
by sorry

end range_of_f_l1022_102242


namespace chocolate_milk_students_l1022_102282

theorem chocolate_milk_students (strawberry_milk : ℕ) (regular_milk : ℕ) (total_milk : ℕ) :
  strawberry_milk = 15 →
  regular_milk = 3 →
  total_milk = 20 →
  total_milk - (strawberry_milk + regular_milk) = 2 := by
sorry

end chocolate_milk_students_l1022_102282


namespace count_five_digit_integers_l1022_102289

/-- The number of different positive five-digit integers that can be formed using the digits 1, 1, 1, 2, and 2 -/
def num_five_digit_integers : ℕ := 10

/-- The multiset of digits used to form the integers -/
def digit_multiset : Multiset ℕ := {1, 1, 1, 2, 2}

/-- The theorem stating that the number of different positive five-digit integers
    that can be formed using the digits 1, 1, 1, 2, and 2 is equal to 10 -/
theorem count_five_digit_integers :
  (Multiset.card digit_multiset = 5) →
  (Multiset.card (Multiset.erase digit_multiset 1) = 2) →
  (Multiset.card (Multiset.erase digit_multiset 2) = 3) →
  num_five_digit_integers = 10 := by
  sorry


end count_five_digit_integers_l1022_102289


namespace quadratic_root_sqrt5_minus_1_l1022_102285

theorem quadratic_root_sqrt5_minus_1 :
  ∃ (a b c : ℚ), (a ≠ 0) ∧ 
  (∀ x : ℝ, a * x^2 + b * x + c = 0 ↔ x^2 + 2*x - 6 = 0) ∧
  (Real.sqrt 5 - 1)^2 + 2*(Real.sqrt 5 - 1) - 6 = 0 := by
  sorry

end quadratic_root_sqrt5_minus_1_l1022_102285


namespace rowing_speed_problem_l1022_102276

/-- Represents the rowing speed problem -/
theorem rowing_speed_problem (v c : ℝ) : 
  c = 1.4 → 
  (v + c) = 2 * (v - c) → 
  v = 4.2 := by
  sorry

end rowing_speed_problem_l1022_102276


namespace cube_tetrahedron_surface_area_ratio_l1022_102241

theorem cube_tetrahedron_surface_area_ratio :
  let cube_side_length : ℝ := 2
  let tetrahedron_vertices : List (ℝ × ℝ × ℝ) := [(0, 0, 0), (2, 2, 0), (2, 0, 2), (0, 2, 2)]
  let cube_surface_area : ℝ := 6 * cube_side_length ^ 2
  let tetrahedron_side_length : ℝ := Real.sqrt ((2 - 0)^2 + (2 - 0)^2 + (0 - 0)^2)
  let tetrahedron_surface_area : ℝ := Real.sqrt 3 * tetrahedron_side_length ^ 2
  cube_surface_area / tetrahedron_surface_area = Real.sqrt 3 := by
  sorry

end cube_tetrahedron_surface_area_ratio_l1022_102241


namespace equality_check_l1022_102290

theorem equality_check : 
  (3^2 ≠ 2^3) ∧ 
  ((-2)^3 = -2^3) ∧ 
  (-3^2 ≠ (-3)^2) ∧ 
  (-(-2) ≠ -|-2|) :=
by sorry

end equality_check_l1022_102290


namespace scenario_one_count_scenario_two_count_l1022_102294

/-- Represents the number of products --/
def total_products : ℕ := 10

/-- Represents the number of defective products --/
def defective_products : ℕ := 4

/-- Calculates the number of testing methods for scenario 1 --/
def scenario_one_methods : ℕ := sorry

/-- Calculates the number of testing methods for scenario 2 --/
def scenario_two_methods : ℕ := sorry

/-- Theorem for scenario 1 --/
theorem scenario_one_count :
  scenario_one_methods = 103680 :=
sorry

/-- Theorem for scenario 2 --/
theorem scenario_two_count :
  scenario_two_methods = 576 :=
sorry

end scenario_one_count_scenario_two_count_l1022_102294


namespace taxi_charge_theorem_l1022_102234

/-- Calculates the total charge for a taxi trip given the initial fee, rate per increment, increment distance, and total distance. -/
def totalCharge (initialFee : ℚ) (ratePerIncrement : ℚ) (incrementDistance : ℚ) (totalDistance : ℚ) : ℚ :=
  initialFee + (totalDistance / incrementDistance).floor * ratePerIncrement

/-- Theorem stating that the total charge for a 3.6-mile trip with given fee structure is $3.60 -/
theorem taxi_charge_theorem :
  let initialFee : ℚ := 225/100
  let ratePerIncrement : ℚ := 15/100
  let incrementDistance : ℚ := 2/5
  let totalDistance : ℚ := 36/10
  totalCharge initialFee ratePerIncrement incrementDistance totalDistance = 360/100 := by
  sorry


end taxi_charge_theorem_l1022_102234


namespace tetrahedron_symmetry_l1022_102271

/-- Represents a point in 3D space -/
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

/-- Represents a tetrahedron in 3D space -/
structure Tetrahedron where
  A : Point3D
  B : Point3D
  C : Point3D
  D : Point3D

/-- Represents a line in 3D space -/
structure Line3D where
  point : Point3D
  direction : Point3D

/-- The center of mass of a tetrahedron -/
def centerOfMass (t : Tetrahedron) : Point3D := sorry

/-- The center of the circumscribed sphere of a tetrahedron -/
def circumCenter (t : Tetrahedron) : Point3D := sorry

/-- Check if a line intersects an edge of a tetrahedron -/
def intersectsEdge (l : Line3D) (p1 p2 : Point3D) : Prop := sorry

/-- Calculate the distance between two points -/
def distance (p1 p2 : Point3D) : ℝ := sorry

/-- Theorem statement -/
theorem tetrahedron_symmetry (t : Tetrahedron) 
  (l : Line3D) 
  (h1 : l.point = centerOfMass t) 
  (h2 : l.point = circumCenter t) 
  (h3 : intersectsEdge l t.A t.B) 
  (h4 : intersectsEdge l t.C t.D) : 
  distance t.A t.C = distance t.B t.D ∧ 
  distance t.A t.D = distance t.B t.C := by
  sorry

end tetrahedron_symmetry_l1022_102271


namespace circular_garden_area_l1022_102270

theorem circular_garden_area (AB DC : ℝ) (h1 : AB = 20) (h2 : DC = 12) : 
  let AD := AB / 2
  let R := (AD ^ 2 + DC ^ 2).sqrt
  π * R ^ 2 = 244 * π := by sorry

end circular_garden_area_l1022_102270


namespace specific_ellipse_foci_distance_l1022_102291

/-- An ellipse with axes parallel to the coordinate axes -/
structure ParallelAxisEllipse where
  /-- The point where the ellipse is tangent to the x-axis -/
  x_tangent : ℝ × ℝ
  /-- The point where the ellipse is tangent to the y-axis -/
  y_tangent : ℝ × ℝ

/-- The distance between the foci of an ellipse -/
def foci_distance (e : ParallelAxisEllipse) : ℝ := sorry

/-- Theorem: The distance between the foci of the specific ellipse is 6√3 -/
theorem specific_ellipse_foci_distance :
  let e : ParallelAxisEllipse := ⟨(6, 0), (0, 3)⟩
  foci_distance e = 6 * Real.sqrt 3 := by sorry

end specific_ellipse_foci_distance_l1022_102291


namespace c_k_value_l1022_102200

/-- Arithmetic sequence with first term 1 and common difference d -/
def arithmetic_seq (d : ℝ) (n : ℕ) : ℝ :=
  1 + (n - 1 : ℝ) * d

/-- Geometric sequence with first term 1 and common ratio r -/
def geometric_seq (r : ℝ) (n : ℕ) : ℝ :=
  r ^ (n - 1)

/-- Sum of nth terms of arithmetic and geometric sequences -/
def c_seq (d r : ℝ) (n : ℕ) : ℝ :=
  arithmetic_seq d n + geometric_seq r n

theorem c_k_value (d r : ℝ) (k : ℕ) :
  (∃ k : ℕ, c_seq d r (k - 1) = 150 ∧ c_seq d r (k + 1) = 900) →
  c_seq d r k = 314 := by
  sorry

end c_k_value_l1022_102200


namespace cone_base_circumference_l1022_102297

/-- The circumference of the base of a right circular cone formed from a sector of a circle --/
theorem cone_base_circumference (r : ℝ) (θ : ℝ) (h_r : r = 6) (h_θ : θ = 240) :
  let original_circumference := 2 * π * r
  let sector_proportion := θ / 360
  let base_circumference := sector_proportion * original_circumference
  base_circumference = 8 * π := by
  sorry

end cone_base_circumference_l1022_102297


namespace smallest_integer_solution_l1022_102295

-- Define the custom operation
def custom_op (x y : ℚ) : ℚ := (x * y / 3) - 2 * y

-- Theorem statement
theorem smallest_integer_solution :
  ∀ a : ℤ, (custom_op 2 (↑a) ≤ 2) → (a ≥ -1) ∧ 
  ∀ b : ℤ, (b < -1) → (custom_op 2 (↑b) > 2) :=
by sorry

end smallest_integer_solution_l1022_102295


namespace mateo_absent_days_l1022_102269

/-- Calculates the number of days not worked given weekly salary, work days per week, and deducted salary -/
def daysNotWorked (weeklySalary workDaysPerWeek deductedSalary : ℚ) : ℕ :=
  let dailySalary := weeklySalary / workDaysPerWeek
  let exactDaysNotWorked := deductedSalary / dailySalary
  (exactDaysNotWorked + 1/2).floor.toNat

/-- Proves that given the specific conditions, the number of days not worked is 2 -/
theorem mateo_absent_days :
  daysNotWorked 791 5 339 = 2 := by
  sorry

#eval daysNotWorked 791 5 339

end mateo_absent_days_l1022_102269


namespace day_relationship_l1022_102245

/-- Represents days of the week -/
inductive DayOfWeek
  | Sunday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday

/-- Represents a specific day in a year -/
structure YearDay where
  year : Int
  day : Nat

/-- Function to determine the day of the week for a given YearDay -/
def dayOfWeek : YearDay → DayOfWeek := sorry

/-- Theorem stating the relationship between the days in different years -/
theorem day_relationship (N : Int) :
  dayOfWeek { year := N, day := 275 } = DayOfWeek.Thursday →
  dayOfWeek { year := N + 1, day := 215 } = DayOfWeek.Thursday →
  dayOfWeek { year := N - 1, day := 150 } = DayOfWeek.Saturday :=
by sorry

end day_relationship_l1022_102245


namespace line_translation_l1022_102268

/-- A line in the 2D plane represented by its slope and y-intercept. -/
structure Line where
  slope : ℝ
  intercept : ℝ

/-- Vertical translation of a line. -/
def verticalTranslate (l : Line) (d : ℝ) : Line :=
  { slope := l.slope, intercept := l.intercept - d }

theorem line_translation (x : ℝ) :
  let original := Line.mk 2 0
  let transformed := Line.mk 2 (-3)
  transformed = verticalTranslate original 3 := by sorry

end line_translation_l1022_102268


namespace exists_double_application_negation_l1022_102277

theorem exists_double_application_negation :
  ∃ f : ℝ → ℝ, ∀ x : ℝ, f (f x) = -x := by
  sorry

end exists_double_application_negation_l1022_102277


namespace min_value_a_plus_2b_l1022_102273

theorem min_value_a_plus_2b (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : 2 * a + b = a * b - 1) :
  ∀ x y : ℝ, x > 0 → y > 0 → 2 * x + y = x * y - 1 → a + 2 * b ≤ x + 2 * y ∧
  ∃ a₀ b₀ : ℝ, a₀ > 0 ∧ b₀ > 0 ∧ 2 * a₀ + b₀ = a₀ * b₀ - 1 ∧ a₀ + 2 * b₀ = 5 + 2 * Real.sqrt 6 :=
by sorry

end min_value_a_plus_2b_l1022_102273


namespace gizmos_produced_75_workers_2_hours_l1022_102249

/-- Represents the production rates and worker information for a manufacturing plant. -/
structure ProductionData where
  gadget_rate : ℝ  -- Gadgets produced per worker per hour
  gizmo_rate : ℝ   -- Gizmos produced per worker per hour
  workers : ℕ      -- Number of workers
  hours : ℝ        -- Number of hours worked

/-- Calculates the number of gizmos produced given production data. -/
def gizmos_produced (data : ProductionData) : ℝ :=
  data.gizmo_rate * data.workers * data.hours

/-- States that the number of gizmos produced by 75 workers in 2 hours is 450. -/
theorem gizmos_produced_75_workers_2_hours :
  let data : ProductionData := {
    gadget_rate := 2,
    gizmo_rate := 3,
    workers := 75,
    hours := 2
  }
  gizmos_produced data = 450 := by sorry

end gizmos_produced_75_workers_2_hours_l1022_102249


namespace eleventh_number_with_digit_sum_13_l1022_102251

/-- A function that returns the sum of digits of a positive integer -/
def sumOfDigits (n : ℕ+) : ℕ := sorry

/-- A function that returns the nth positive integer whose digits sum to 13 -/
def nthNumberWithDigitSum13 (n : ℕ+) : ℕ+ := sorry

/-- The theorem stating that the 11th number with digit sum 13 is 166 -/
theorem eleventh_number_with_digit_sum_13 : 
  nthNumberWithDigitSum13 11 = 166 := by sorry

end eleventh_number_with_digit_sum_13_l1022_102251


namespace lines_are_parallel_l1022_102247

/-- Represents a line in the form ax + by + c = 0 --/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Two lines are parallel if they have the same slope --/
def parallel (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l2.a * l1.b

theorem lines_are_parallel : 
  let line1 : Line := { a := 1, b := -1, c := 2 }
  let line2 : Line := { a := 1, b := -1, c := 1 }
  parallel line1 line2 := by sorry

end lines_are_parallel_l1022_102247


namespace shoes_lost_l1022_102208

theorem shoes_lost (initial_pairs : ℕ) (max_pairs_left : ℕ) (h1 : initial_pairs = 25) (h2 : max_pairs_left = 20) :
  initial_pairs * 2 - max_pairs_left * 2 = 10 := by
  sorry

end shoes_lost_l1022_102208


namespace hiker_speed_day3_l1022_102203

/-- A hiker's three-day journey --/
structure HikerJourney where
  day1_distance : ℝ
  day1_speed : ℝ
  day2_hours_reduction : ℝ
  day2_speed_increase : ℝ
  day3_hours : ℝ
  total_distance : ℝ

/-- Theorem about the hiker's speed on the third day --/
theorem hiker_speed_day3 (journey : HikerJourney)
  (h1 : journey.day1_distance = 18)
  (h2 : journey.day1_speed = 3)
  (h3 : journey.day2_hours_reduction = 1)
  (h4 : journey.day2_speed_increase = 1)
  (h5 : journey.day3_hours = 3)
  (h6 : journey.total_distance = 53) :
  (journey.total_distance
    - journey.day1_distance
    - (journey.day1_distance / journey.day1_speed - journey.day2_hours_reduction)
      * (journey.day1_speed + journey.day2_speed_increase))
  / journey.day3_hours = 5 := by
  sorry


end hiker_speed_day3_l1022_102203


namespace sphere_radius_is_one_l1022_102202

/-- Represents a cone with a given base radius -/
structure Cone where
  baseRadius : ℝ
  height : ℝ

/-- Represents a sphere with a given radius -/
structure Sphere where
  radius : ℝ

/-- Represents the configuration of three cones and a sphere -/
structure ConeSphereProblem where
  cone1 : Cone
  cone2 : Cone
  cone3 : Cone
  sphere : Sphere
  sameHeight : cone1.height = cone2.height ∧ cone2.height = cone3.height
  baseRadii : cone1.baseRadius = 1 ∧ cone2.baseRadius = 2 ∧ cone3.baseRadius = 3
  touching : True  -- Cones are touching each other
  sphereTouchingCones : True  -- Sphere touches all cones
  sphereTouchingTable : True  -- Sphere touches the table
  centerEquidistant : True  -- Center of sphere is equidistant from all points of contact with cones

/-- The theorem stating that the radius of the sphere is 1 -/
theorem sphere_radius_is_one (problem : ConeSphereProblem) : problem.sphere.radius = 1 := by
  sorry

end sphere_radius_is_one_l1022_102202


namespace total_dogs_count_l1022_102209

/-- The number of boxes of stuffed toy dogs -/
def num_boxes : ℕ := 15

/-- The number of dogs in each box -/
def dogs_per_box : ℕ := 8

/-- The total number of dogs -/
def total_dogs : ℕ := num_boxes * dogs_per_box

theorem total_dogs_count : total_dogs = 120 := by
  sorry

end total_dogs_count_l1022_102209


namespace typhoon_tree_difference_l1022_102201

theorem typhoon_tree_difference (initial_trees : ℕ) (dead_trees : ℕ) : 
  initial_trees = 3 → dead_trees = 13 → dead_trees - initial_trees = 10 :=
by sorry

end typhoon_tree_difference_l1022_102201


namespace inverse_proportion_problem_l1022_102238

/-- Given that x and y are inversely proportional, and x + y = 30 and x - y = 10, 
    prove that y = 200/7 when x = 7. -/
theorem inverse_proportion_problem (x y : ℝ) (k : ℝ) 
  (h1 : x * y = k)  -- x and y are inversely proportional
  (h2 : x + y = 30) -- sum condition
  (h3 : x - y = 10) -- difference condition
  : (7 : ℝ) * (200 / 7) = k := by
  sorry

end inverse_proportion_problem_l1022_102238


namespace train_speed_l1022_102223

/-- Given a train of length 800 meters that crosses an electric pole in 20 seconds,
    prove that its speed is 144 km/h. -/
theorem train_speed (train_length : ℝ) (crossing_time : ℝ) (speed_ms : ℝ) (speed_kmh : ℝ)
    (h1 : train_length = 800)
    (h2 : crossing_time = 20)
    (h3 : speed_ms = train_length / crossing_time)
    (h4 : speed_kmh = speed_ms * 3.6) :
    speed_kmh = 144 := by
  sorry

end train_speed_l1022_102223


namespace inequality_solution_set_l1022_102288

theorem inequality_solution_set (x : ℝ) : 
  (x - 1) / (x + 3) > (2 * x + 5) / (3 * x + 8) ↔ 
  (x > -3 ∧ x < -8/3) ∨ (x > (3 - Real.sqrt 69) / 2 ∧ x < (3 + Real.sqrt 69) / 2) := by
sorry

end inequality_solution_set_l1022_102288


namespace income_of_M_l1022_102240

theorem income_of_M (M N O : ℝ) 
  (avg_MN : (M + N) / 2 = 5050)
  (avg_NO : (N + O) / 2 = 6250)
  (avg_MO : (M + O) / 2 = 5200) :
  M = 2666.67 := by
  sorry

end income_of_M_l1022_102240


namespace correct_oranges_to_put_back_l1022_102261

/-- Represents the fruit selection problem -/
structure FruitSelection where
  apple_price : ℚ
  orange_price : ℚ
  total_fruits : ℕ
  initial_avg_price : ℚ
  desired_avg_price : ℚ

/-- Calculates the number of oranges to put back -/
def oranges_to_put_back (fs : FruitSelection) : ℕ :=
  sorry

/-- Theorem stating the correct number of oranges to put back -/
theorem correct_oranges_to_put_back (fs : FruitSelection) 
  (h1 : fs.apple_price = 40/100)
  (h2 : fs.orange_price = 60/100)
  (h3 : fs.total_fruits = 10)
  (h4 : fs.initial_avg_price = 56/100)
  (h5 : fs.desired_avg_price = 50/100) :
  oranges_to_put_back fs = 6 := by
  sorry

end correct_oranges_to_put_back_l1022_102261


namespace max_sum_of_squares_max_sum_of_squares_achievable_l1022_102260

theorem max_sum_of_squares (a b c d : ℝ) : 
  a + b = 20 →
  a * b + c + d = 90 →
  a * d + b * c = 210 →
  c * d = 125 →
  a^2 + b^2 + c^2 + d^2 ≤ 1450 := by
  sorry

theorem max_sum_of_squares_achievable : 
  ∃ (a b c d : ℝ),
    a + b = 20 ∧
    a * b + c + d = 90 ∧
    a * d + b * c = 210 ∧
    c * d = 125 ∧
    a^2 + b^2 + c^2 + d^2 = 1450 := by
  sorry

end max_sum_of_squares_max_sum_of_squares_achievable_l1022_102260


namespace experiment_success_probability_l1022_102229

-- Define the experiment setup
structure ExperimentSetup where
  box1_total : ℕ := 10
  box1_a : ℕ := 7
  box1_b : ℕ := 3
  box2_total : ℕ := 10
  box2_red : ℕ := 5
  box3_total : ℕ := 10
  box3_red : ℕ := 8

-- Define the probability of success
def probability_of_success (setup : ExperimentSetup) : ℚ :=
  let p1 := (setup.box1_a : ℚ) / setup.box1_total * setup.box2_red / setup.box2_total
  let p2 := (setup.box1_b : ℚ) / setup.box1_total * setup.box3_red / setup.box3_total
  p1 + p2

-- Theorem statement
theorem experiment_success_probability (setup : ExperimentSetup) :
  probability_of_success setup = 59 / 100 := by
  sorry

end experiment_success_probability_l1022_102229


namespace fred_tim_marbles_comparison_l1022_102279

theorem fred_tim_marbles_comparison :
  let fred_marbles : ℕ := 110
  let tim_marbles : ℕ := 5
  (fred_marbles / tim_marbles : ℚ) = 22 :=
by sorry

end fred_tim_marbles_comparison_l1022_102279


namespace expression_evaluation_l1022_102227

theorem expression_evaluation :
  1 - 1 / (1 + Real.sqrt (2 + Real.sqrt 3)) + 1 / (1 - Real.sqrt (2 - Real.sqrt 3)) =
  1 + (Real.sqrt (2 - Real.sqrt 3) + Real.sqrt (2 + Real.sqrt 3)) / (-1 - Real.sqrt 3) := by
sorry

end expression_evaluation_l1022_102227


namespace max_y_value_max_y_value_achievable_l1022_102236

theorem max_y_value (x y : ℤ) (h : x * y + 3 * x + 2 * y = -4) : y ≤ -1 := by
  sorry

theorem max_y_value_achievable : ∃ x y : ℤ, x * y + 3 * x + 2 * y = -4 ∧ y = -1 := by
  sorry

end max_y_value_max_y_value_achievable_l1022_102236


namespace final_number_can_be_zero_l1022_102221

/-- Represents the operation of replacing two numbers with their absolute difference -/
def difference_operation (S : Finset ℕ) : Finset ℕ :=
  sorry

/-- The initial set of integers from 1 to 2013 -/
def initial_set : Finset ℕ :=
  Finset.range 2013

/-- Applies the difference operation n times to the given set -/
def apply_n_times (S : Finset ℕ) (n : ℕ) : Finset ℕ :=
  sorry

theorem final_number_can_be_zero :
  ∃ (result : Finset ℕ), apply_n_times initial_set 2012 = result ∧ 0 ∈ result :=
sorry

end final_number_can_be_zero_l1022_102221


namespace centroid_distance_theorem_l1022_102267

/-- Represents the possible distances from the centroid of a triangle to a plane -/
inductive CentroidDistance : Type
  | six : CentroidDistance
  | two : CentroidDistance
  | eight_thirds : CentroidDistance
  | four_thirds : CentroidDistance

/-- Given a triangle with vertices at distances 5, 6, and 7 from a plane,
    the distance from the centroid to the same plane is one of the defined values -/
theorem centroid_distance_theorem (d1 d2 d3 : ℝ) (h1 : d1 = 5) (h2 : d2 = 6) (h3 : d3 = 7) :
  ∃ (cd : CentroidDistance), true :=
sorry

end centroid_distance_theorem_l1022_102267


namespace rectangle_area_l1022_102237

theorem rectangle_area (square_side : ℝ) (circle_radius : ℝ) (rectangle_length : ℝ) (rectangle_breadth : ℝ) :
  square_side ^ 2 = 1296 →
  circle_radius = square_side →
  rectangle_length = circle_radius / 6 →
  rectangle_breadth = 10 →
  rectangle_length * rectangle_breadth = 60 := by
sorry

end rectangle_area_l1022_102237


namespace susie_investment_l1022_102262

/-- Proves that Susie's investment at Safe Savings Bank is 0 --/
theorem susie_investment (total_investment : ℝ) (safe_rate : ℝ) (risky_rate : ℝ) (total_after_year : ℝ) 
  (h1 : total_investment = 2000)
  (h2 : safe_rate = 0.04)
  (h3 : risky_rate = 0.06)
  (h4 : total_after_year = 2120)
  (h5 : ∀ x : ℝ, x * (1 + safe_rate) + (total_investment - x) * (1 + risky_rate) = total_after_year) :
  ∃ x : ℝ, x = 0 ∧ x * (1 + safe_rate) + (total_investment - x) * (1 + risky_rate) = total_after_year :=
sorry

end susie_investment_l1022_102262


namespace tagged_fish_in_second_catch_l1022_102264

theorem tagged_fish_in_second_catch 
  (initial_tagged : ℕ) 
  (second_catch : ℕ) 
  (total_fish : ℕ) 
  (h1 : initial_tagged = 80) 
  (h2 : second_catch = 80) 
  (h3 : total_fish = 3200) :
  ∃ (tagged_in_second : ℕ), 
    tagged_in_second = 2 ∧ 
    (tagged_in_second : ℚ) / second_catch = initial_tagged / total_fish :=
by
  sorry

end tagged_fish_in_second_catch_l1022_102264


namespace smallest_number_l1022_102246

theorem smallest_number (s : Set ℚ) (hs : s = {-2, 0, 3, 5}) : 
  ∃ m ∈ s, ∀ x ∈ s, m ≤ x ∧ m = -2 :=
sorry

end smallest_number_l1022_102246


namespace oil_after_eight_hours_l1022_102252

/-- Represents the remaining oil in a car's fuel tank as a function of time -/
def remaining_oil (initial_oil : ℝ) (consumption_rate : ℝ) (time : ℝ) : ℝ :=
  initial_oil - consumption_rate * time

theorem oil_after_eight_hours 
  (initial_oil : ℝ) 
  (consumption_rate : ℝ) 
  (h1 : initial_oil = 50) 
  (h2 : consumption_rate = 5) :
  remaining_oil initial_oil consumption_rate 8 = 10 := by
  sorry

#check oil_after_eight_hours

end oil_after_eight_hours_l1022_102252


namespace cyclic_iff_concurrent_l1022_102226

/-- A point in the plane -/
structure Point :=
  (x : ℝ) (y : ℝ)

/-- A line in the plane -/
structure Line :=
  (a : ℝ) (b : ℝ) (c : ℝ)

/-- Check if four points are cyclic -/
def are_cyclic (A B C D : Point) : Prop :=
  sorry

/-- Check if three lines are concurrent -/
def are_concurrent (l1 l2 l3 : Line) : Prop :=
  sorry

/-- Get the line passing through two points -/
def line_through_points (A B : Point) : Line :=
  sorry

theorem cyclic_iff_concurrent (A B C D E F : Point) :
  are_cyclic A B C D → are_cyclic C D E F →
  (are_cyclic A B E F ↔ 
    are_concurrent 
      (line_through_points A B) 
      (line_through_points C D) 
      (line_through_points E F)) :=
by sorry

end cyclic_iff_concurrent_l1022_102226


namespace circle_equation_l1022_102274

/-- The ellipse with equation x²/16 + y²/4 = 1 -/
def Ellipse : Set (ℝ × ℝ) :=
  {p | (p.1^2 / 16) + (p.2^2 / 4) = 1}

/-- The vertices of the ellipse -/
def EllipseVertices : Set (ℝ × ℝ) :=
  {p | p ∈ Ellipse ∧ (p.1 = 0 ∨ p.2 = 0)}

/-- The circle C passing through (6,0) and the vertices of the ellipse -/
def CircleC : Set (ℝ × ℝ) :=
  {p | ∃ (c : ℝ), (c, 0) ∈ Ellipse ∧ 
    ((p.1 - c)^2 + p.2^2 = (6 - c)^2) ∧
    (∀ v ∈ EllipseVertices, (p.1 - c)^2 + p.2^2 = (v.1 - c)^2 + v.2^2)}

theorem circle_equation : 
  CircleC = {p | (p.1 - 8/3)^2 + p.2^2 = 100/9} := by
  sorry


end circle_equation_l1022_102274


namespace savings_calculation_l1022_102298

theorem savings_calculation (income expenditure savings : ℕ) : 
  (income : ℚ) / expenditure = 8 / 7 →
  income = 40000 →
  savings = income - expenditure →
  savings = 5000 := by
  sorry

end savings_calculation_l1022_102298


namespace petya_vasya_journey_l1022_102244

/-- Represents the problem of Petya and Vasya's journey to the football match. -/
theorem petya_vasya_journey
  (distance : ℝ)
  (walking_speed : ℝ)
  (bicycle_speed_multiplier : ℝ)
  (late_time : ℝ)
  (h1 : distance = 4)
  (h2 : walking_speed = 4)
  (h3 : bicycle_speed_multiplier = 3)
  (h4 : late_time = 10)
  (h5 : distance / walking_speed * 60 - late_time = 50) :
  let bicycle_speed := walking_speed * bicycle_speed_multiplier
  let half_distance := distance / 2
  let walking_time := half_distance / walking_speed * 60
  let cycling_time := half_distance / bicycle_speed * 60
  let total_time := walking_time + cycling_time
  50 - total_time = 10 := by
  sorry

end petya_vasya_journey_l1022_102244


namespace largest_integer_satisfying_inequality_l1022_102232

theorem largest_integer_satisfying_inequality : 
  ∀ x : ℤ, x ≤ 3 ↔ (x : ℚ) / 4 + 7 / 6 < 8 / 4 :=
by sorry

end largest_integer_satisfying_inequality_l1022_102232


namespace continued_fraction_value_l1022_102235

theorem continued_fraction_value : ∃ y : ℝ, y > 0 ∧ y = 3 + 9 / (2 + 9 / y) ∧ y = 6 := by sorry

end continued_fraction_value_l1022_102235


namespace voice_area_greater_than_ground_area_l1022_102239

/-- The side length of the square ground in meters -/
def ground_side : ℝ := 25

/-- The maximum distance the trainer's voice can be heard in meters -/
def voice_range : ℝ := 140

/-- The area of the ground where the trainer's voice can be heard is greater than the area of the square ground -/
theorem voice_area_greater_than_ground_area : π * voice_range^2 > ground_side^2 := by
  sorry

end voice_area_greater_than_ground_area_l1022_102239


namespace odd_power_sum_divisible_l1022_102204

theorem odd_power_sum_divisible (x y : ℤ) :
  ∀ k : ℕ, k > 0 →
    (∃ m : ℤ, x^(2*k-1) + y^(2*k-1) = m * (x + y)) →
    (∃ n : ℤ, x^(2*k+1) + y^(2*k+1) = n * (x + y)) :=
by sorry

end odd_power_sum_divisible_l1022_102204


namespace calculate_expression_l1022_102278

theorem calculate_expression : 
  3 / Real.sqrt 3 - (Real.pi + Real.sqrt 3) ^ 0 - Real.sqrt 27 + |Real.sqrt 3 - 2| = -3 * Real.sqrt 3 + 1 := by
  sorry

end calculate_expression_l1022_102278


namespace solve_parking_problem_l1022_102250

def parking_problem (initial_balance : ℚ) (first_three_cost : ℚ) (fourth_cost_ratio : ℚ) (fifth_cost_ratio : ℚ) (roommate_payment_ratio : ℚ) : Prop :=
  let total_first_three := 3 * first_three_cost
  let fourth_ticket_cost := fourth_cost_ratio * first_three_cost
  let fifth_ticket_cost := fifth_cost_ratio * first_three_cost
  let total_cost := total_first_three + fourth_ticket_cost + fifth_ticket_cost
  let roommate_payment := roommate_payment_ratio * total_cost
  let james_payment := total_cost - roommate_payment
  let remaining_balance := initial_balance - james_payment
  remaining_balance = 871.88

theorem solve_parking_problem :
  parking_problem 1200 250 (1/4) (1/2) 0.65 :=
sorry

end solve_parking_problem_l1022_102250


namespace tangent_lines_to_circle_l1022_102281

/-- The circle equation -/
def circle_equation (x y : ℝ) : Prop :=
  x^2 + y^2 - 6*x - 2*y + 6 = 0

/-- Point P -/
def P : ℝ × ℝ := (1, -2)

/-- First tangent line equation -/
def tangent1 (x y : ℝ) : Prop :=
  5*x - 12*y - 29 = 0

/-- Second tangent line equation -/
def tangent2 (x : ℝ) : Prop :=
  x = 1

/-- Theorem stating that the tangent lines from P to the circle have the given equations -/
theorem tangent_lines_to_circle :
  ∃ (x y : ℝ), circle_equation x y ∧
  ((tangent1 x y ∧ (x, y) ≠ P) ∨ (tangent2 x ∧ y ≠ -2)) :=
sorry

end tangent_lines_to_circle_l1022_102281


namespace expression_factorization_l1022_102207

theorem expression_factorization (x : ℝ) :
  (10 * x^3 + 45 * x^2 - 5 * x) - (-5 * x^3 + 10 * x^2 - 5 * x) = 5 * x^2 * (3 * x + 7) := by
  sorry

end expression_factorization_l1022_102207


namespace isosceles_triangle_with_50_degree_angle_l1022_102258

-- Define an isosceles triangle
structure IsoscelesTriangle where
  -- We only need to define two angles, as the third can be derived
  angle1 : ℝ
  angle2 : ℝ
  -- Condition: The triangle is isosceles (two angles are equal)
  isIsosceles : angle1 = angle2 ∨ angle1 = 180 - angle1 - angle2 ∨ angle2 = 180 - angle1 - angle2
  -- Condition: The sum of angles in a triangle is 180°
  sumIs180 : angle1 + angle2 + (180 - angle1 - angle2) = 180

-- Define our theorem
theorem isosceles_triangle_with_50_degree_angle 
  (triangle : IsoscelesTriangle) 
  (has50DegreeAngle : triangle.angle1 = 50 ∨ triangle.angle2 = 50 ∨ (180 - triangle.angle1 - triangle.angle2) = 50) :
  triangle.angle1 = 50 ∨ triangle.angle1 = 65 ∨ triangle.angle2 = 50 ∨ triangle.angle2 = 65 :=
sorry

end isosceles_triangle_with_50_degree_angle_l1022_102258


namespace x_value_l1022_102228

/-- The equation that defines x -/
def x_equation (x : ℝ) : Prop := x = Real.sqrt (2 + x)

/-- Theorem stating that the solution to the equation is 2 -/
theorem x_value : ∃ x : ℝ, x_equation x ∧ x = 2 := by sorry

end x_value_l1022_102228


namespace area_triangle_DEF_area_triangle_DEF_is_six_l1022_102231

/-- Triangle DEF with vertices D, E, and F, where F lies on the line x + y = 6 -/
structure TriangleDEF where
  D : ℝ × ℝ
  E : ℝ × ℝ
  F : ℝ × ℝ
  h_D : D = (2, 1)
  h_E : E = (1, 4)
  h_F : F.1 + F.2 = 6

/-- The area of triangle DEF is 6 -/
theorem area_triangle_DEF (t : TriangleDEF) : ℝ :=
  6

/-- The area of triangle DEF is indeed 6 -/
theorem area_triangle_DEF_is_six (t : TriangleDEF) :
  area_triangle_DEF t = 6 := by
  sorry

end area_triangle_DEF_area_triangle_DEF_is_six_l1022_102231


namespace smallest_divisible_by_10_and_24_l1022_102230

theorem smallest_divisible_by_10_and_24 :
  ∃ n : ℕ, n > 0 ∧ n % 10 = 0 ∧ n % 24 = 0 ∧ ∀ m : ℕ, m > 0 → m % 10 = 0 → m % 24 = 0 → n ≤ m :=
by
  sorry

end smallest_divisible_by_10_and_24_l1022_102230


namespace value_of_a_l1022_102263

theorem value_of_a (a c : ℝ) (h1 : c / a = 4) (h2 : a + c = 30) : a = 6 := by
  sorry

end value_of_a_l1022_102263


namespace binary_decimal_octal_conversion_l1022_102292

def binary_to_decimal (b : List Bool) : Nat :=
  b.enum.foldl (fun acc (i, bit) => acc + if bit then 2^i else 0) 0

def decimal_to_octal (n : Nat) : List Nat :=
  let rec aux (m : Nat) (acc : List Nat) : List Nat :=
    if m = 0 then acc
    else aux (m / 8) ((m % 8) :: acc)
  aux n []

def binary_101101 : List Bool := [true, false, true, true, false, true]

theorem binary_decimal_octal_conversion :
  binary_to_decimal binary_101101 = 45 ∧
  decimal_to_octal 45 = [5, 5] := by
  sorry

end binary_decimal_octal_conversion_l1022_102292


namespace a_perpendicular_to_a_minus_b_l1022_102299

def a : ℝ × ℝ := (-2, 1)
def b : ℝ × ℝ := (-1, 3)

theorem a_perpendicular_to_a_minus_b : a.1 * (a.1 - b.1) + a.2 * (a.2 - b.2) = 0 := by
  sorry

end a_perpendicular_to_a_minus_b_l1022_102299


namespace initial_retail_price_l1022_102272

/-- Calculates the initial retail price of a machine given wholesale price, shipping, tax, discount, and profit margin. -/
theorem initial_retail_price
  (wholesale_with_shipping : ℝ)
  (shipping : ℝ)
  (tax_rate : ℝ)
  (discount_rate : ℝ)
  (profit_rate : ℝ)
  (h1 : wholesale_with_shipping = 90)
  (h2 : shipping = 10)
  (h3 : tax_rate = 0.05)
  (h4 : discount_rate = 0.10)
  (h5 : profit_rate = 0.20) :
  let wholesale := (wholesale_with_shipping - shipping) / (1 + tax_rate)
  let cost := wholesale_with_shipping
  let initial_price := cost / (1 - profit_rate - discount_rate + discount_rate * profit_rate)
  initial_price = 125 := by sorry

end initial_retail_price_l1022_102272
