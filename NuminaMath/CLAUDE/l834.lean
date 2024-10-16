import Mathlib

namespace NUMINAMATH_CALUDE_smaller_cuboid_height_l834_83472

/-- Proves that the height of smaller cuboids is 2 meters when a large cuboid
    is divided into smaller ones with given dimensions. -/
theorem smaller_cuboid_height
  (large_length : ℝ) (large_width : ℝ) (large_height : ℝ)
  (small_length : ℝ) (small_width : ℝ)
  (num_small_cuboids : ℕ) :
  large_length = 12 →
  large_width = 14 →
  large_height = 10 →
  small_length = 5 →
  small_width = 3 →
  num_small_cuboids = 56 →
  ∃ (small_height : ℝ),
    large_length * large_width * large_height =
    ↑num_small_cuboids * small_length * small_width * small_height ∧
    small_height = 2 := by
  sorry

end NUMINAMATH_CALUDE_smaller_cuboid_height_l834_83472


namespace NUMINAMATH_CALUDE_initial_mixture_volume_l834_83482

/-- Proves that the initial volume of a milk-water mixture is 45 litres given specific conditions -/
theorem initial_mixture_volume (initial_milk : ℝ) (initial_water : ℝ) : 
  initial_milk / initial_water = 4 →
  initial_milk / (initial_water + 11) = 1.8 →
  initial_milk + initial_water = 45 :=
by
  sorry

#check initial_mixture_volume

end NUMINAMATH_CALUDE_initial_mixture_volume_l834_83482


namespace NUMINAMATH_CALUDE_intersection_A_B_l834_83499

def A : Set ℕ := {0, 1, 2, 3}
def B : Set ℕ := {x : ℕ | Real.log x < 1}

theorem intersection_A_B : A ∩ B = {1, 2} := by sorry

end NUMINAMATH_CALUDE_intersection_A_B_l834_83499


namespace NUMINAMATH_CALUDE_count_valid_functions_l834_83487

def polynomial_function (a b c d : ℝ) (x : ℝ) : ℝ := a * x^3 + b * x^2 + c * x + d

def satisfies_condition (f : ℝ → ℝ) : Prop :=
  ∀ x : ℝ, f x * f (-x) = f (x^3)

theorem count_valid_functions :
  ∃! (valid_functions : Finset (ℝ → ℝ)),
    (∀ f ∈ valid_functions, ∃ a b c d : ℝ, 
      (∀ x : ℝ, f x = polynomial_function a b c d x) ∧
      satisfies_condition f) ∧
    (Finset.card valid_functions = 12) :=
sorry

end NUMINAMATH_CALUDE_count_valid_functions_l834_83487


namespace NUMINAMATH_CALUDE_brandon_job_applications_l834_83401

theorem brandon_job_applications (total_businesses : ℕ) 
  (h1 : total_businesses = 72) 
  (fired : ℕ) (h2 : fired = total_businesses / 2)
  (quit : ℕ) (h3 : quit = total_businesses / 3) : 
  total_businesses - (fired + quit) = 12 :=
by sorry

end NUMINAMATH_CALUDE_brandon_job_applications_l834_83401


namespace NUMINAMATH_CALUDE_complex_power_magnitude_l834_83416

theorem complex_power_magnitude : Complex.abs ((1 - Complex.I) ^ 8) = 16 := by sorry

end NUMINAMATH_CALUDE_complex_power_magnitude_l834_83416


namespace NUMINAMATH_CALUDE_geometric_propositions_l834_83447

-- Define the four propositions
def vertical_angles_equal : Prop := sorry
def alternate_interior_angles_equal : Prop := sorry
def parallel_transitivity : Prop := sorry
def parallel_sides_equal_angles : Prop := sorry

-- Theorem stating which propositions are true
theorem geometric_propositions :
  vertical_angles_equal ∧ 
  parallel_transitivity ∧ 
  ¬alternate_interior_angles_equal ∧ 
  ¬parallel_sides_equal_angles := by
  sorry

end NUMINAMATH_CALUDE_geometric_propositions_l834_83447


namespace NUMINAMATH_CALUDE_coffee_cost_per_pound_l834_83446

/-- Calculates the cost per pound of coffee given the initial gift card amount,
    the amount left after purchase, and the number of pounds bought. -/
def cost_per_pound (initial_amount : ℚ) (amount_left : ℚ) (pounds_bought : ℚ) : ℚ :=
  (initial_amount - amount_left) / pounds_bought

/-- Proves that the cost per pound of coffee is $8.58 given the problem conditions. -/
theorem coffee_cost_per_pound :
  cost_per_pound 70 35.68 4 = 8.58 := by
  sorry

end NUMINAMATH_CALUDE_coffee_cost_per_pound_l834_83446


namespace NUMINAMATH_CALUDE_rostov_true_supporters_l834_83464

structure Island where
  total_population : ℕ
  knights : ℕ
  liars : ℕ
  rostov_yes : ℕ
  zenit_yes : ℕ
  lokomotiv_yes : ℕ
  cska_yes : ℕ

def percentage (n : ℕ) (total : ℕ) : ℚ :=
  (n : ℚ) / (total : ℚ) * 100

theorem rostov_true_supporters (i : Island) :
  i.knights + i.liars = i.total_population →
  percentage i.rostov_yes i.total_population = 40 →
  percentage i.zenit_yes i.total_population = 30 →
  percentage i.lokomotiv_yes i.total_population = 50 →
  percentage i.cska_yes i.total_population = 0 →
  percentage i.liars i.total_population = 10 →
  percentage (i.rostov_yes - i.liars) i.total_population = 30 := by
  sorry

#check rostov_true_supporters

end NUMINAMATH_CALUDE_rostov_true_supporters_l834_83464


namespace NUMINAMATH_CALUDE_triangle_ratio_l834_83492

theorem triangle_ratio (A B C : ℝ) (a b c : ℝ) :
  0 < A ∧ 0 < B ∧ 0 < C ∧
  A + B + C = π ∧
  0 < a ∧ 0 < b ∧ 0 < c ∧
  a / (Real.sin A) = b / (Real.sin B) ∧
  a / (Real.sin A) = c / (Real.sin C) ∧
  A = π / 3 ∧
  a = Real.sqrt 3 →
  (a + b) / (Real.sin A + Real.sin B) = 2 :=
by sorry

end NUMINAMATH_CALUDE_triangle_ratio_l834_83492


namespace NUMINAMATH_CALUDE_harkamal_fruit_purchase_cost_l834_83486

/-- Calculates the discounted price of a fruit purchase -/
def discountedPrice (quantity : ℕ) (pricePerKg : ℚ) (discountPercent : ℚ) : ℚ :=
  quantity * pricePerKg * (1 - discountPercent / 100)

/-- Represents Harkamal's fruit purchases -/
def fruitPurchases : List (ℕ × ℚ × ℚ) := [
  (10, 70, 10),  -- grapes
  (9, 55, 0),    -- mangoes
  (12, 80, 5),   -- apples
  (7, 45, 15),   -- papayas
  (15, 30, 0),   -- oranges
  (5, 25, 0)     -- bananas
]

/-- Calculates the total cost of Harkamal's fruit purchases -/
def totalCost : ℚ :=
  fruitPurchases.foldr (fun (purchase : ℕ × ℚ × ℚ) (acc : ℚ) =>
    acc + discountedPrice purchase.1 purchase.2.1 purchase.2.2
  ) 0

/-- Theorem stating that the total cost of Harkamal's fruit purchases is $2879.75 -/
theorem harkamal_fruit_purchase_cost :
  totalCost = 2879.75 := by sorry

end NUMINAMATH_CALUDE_harkamal_fruit_purchase_cost_l834_83486


namespace NUMINAMATH_CALUDE_jack_lifetime_l834_83469

theorem jack_lifetime :
  ∀ (L : ℝ),
  (L = (1/6)*L + (1/12)*L + (1/7)*L + 5 + (1/2)*L + 4) →
  L = 84 := by
sorry

end NUMINAMATH_CALUDE_jack_lifetime_l834_83469


namespace NUMINAMATH_CALUDE_total_phone_cost_l834_83419

def phone_cost : ℝ := 1000
def monthly_contract : ℝ := 200
def case_cost_percentage : ℝ := 0.20
def headphones_cost_ratio : ℝ := 0.5
def months_in_year : ℕ := 12

def total_cost : ℝ :=
  phone_cost +
  (monthly_contract * months_in_year) +
  (phone_cost * case_cost_percentage) +
  (phone_cost * case_cost_percentage * headphones_cost_ratio)

theorem total_phone_cost : total_cost = 3700 := by
  sorry

end NUMINAMATH_CALUDE_total_phone_cost_l834_83419


namespace NUMINAMATH_CALUDE_arithmetic_sequence_eighth_term_l834_83458

def arithmetic_sequence (a : ℕ → ℝ) : Prop :=
  ∃ d : ℝ, ∀ n : ℕ, a (n + 1) = a n + d

theorem arithmetic_sequence_eighth_term
  (a : ℕ → ℝ)
  (h_arith : arithmetic_sequence a)
  (h_sum : a 5 + a 6 = 22)
  (h_third : a 3 = 7) :
  a 8 = 15 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_eighth_term_l834_83458


namespace NUMINAMATH_CALUDE_union_covers_reals_l834_83498

-- Define the sets A and B
def A : Set ℝ := {x | x < -4 ∨ x > 2}
def B (a : ℝ) : Set ℝ := {x | x < a}

-- State the theorem
theorem union_covers_reals (a : ℝ) :
  A ∪ B a = Set.univ → a > 2 := by
  sorry

end NUMINAMATH_CALUDE_union_covers_reals_l834_83498


namespace NUMINAMATH_CALUDE_inequality_solution_set_l834_83476

theorem inequality_solution_set (x : ℝ) : 3 * x - 2 > x ↔ x > 1 := by sorry

end NUMINAMATH_CALUDE_inequality_solution_set_l834_83476


namespace NUMINAMATH_CALUDE_min_sum_absolute_differences_l834_83444

theorem min_sum_absolute_differences (a : ℚ) : 
  ∃ (min : ℚ), min = 4 ∧ ∀ (x : ℚ), |x-1| + |x-2| + |x-3| + |x-4| ≥ min := by
  sorry

end NUMINAMATH_CALUDE_min_sum_absolute_differences_l834_83444


namespace NUMINAMATH_CALUDE_triangle_angle_relation_l834_83466

theorem triangle_angle_relation (P Q R : Real) (h1 : 5 * Real.sin P + 2 * Real.cos Q = 5) 
  (h2 : 2 * Real.sin Q + 5 * Real.cos P = 3) (h3 : P + Q + R = Real.pi) : 
  Real.sin R = 1/20 :=
sorry

end NUMINAMATH_CALUDE_triangle_angle_relation_l834_83466


namespace NUMINAMATH_CALUDE_max_value_of_s_l834_83407

theorem max_value_of_s (x y : ℝ) (h : 4 * x^2 - 5 * x * y + 4 * y^2 = 5) :
  ∃ (s_max : ℝ), s_max = (10 : ℝ) / 3 ∧ ∀ (s : ℝ), s = x^2 + y^2 → s ≤ s_max :=
sorry

end NUMINAMATH_CALUDE_max_value_of_s_l834_83407


namespace NUMINAMATH_CALUDE_shaded_area_problem_l834_83422

/-- Given a square FGHI with area 80 and points J, K, L, M on its sides
    such that FK = GL = HM = IJ and FK = 3KG, 
    the area of the quadrilateral JKLM is 50. -/
theorem shaded_area_problem (F G H I J K L M : ℝ × ℝ) : 
  (∃ s : ℝ, s > 0 ∧ (G.1 - F.1)^2 + (G.2 - F.2)^2 = s^2 ∧ s^2 = 80) →
  (K.1 - F.1)^2 + (K.2 - F.2)^2 = (L.1 - G.1)^2 + (L.2 - G.2)^2 ∧
   (L.1 - G.1)^2 + (L.2 - G.2)^2 = (M.1 - H.1)^2 + (M.2 - H.2)^2 ∧
   (M.1 - H.1)^2 + (M.2 - H.2)^2 = (J.1 - I.1)^2 + (J.2 - I.2)^2 →
  (K.1 - F.1)^2 + (K.2 - F.2)^2 = 9 * ((G.1 - K.1)^2 + (G.2 - K.2)^2) →
  (K.1 - J.1)^2 + (K.2 - J.2)^2 = 50 :=
by sorry


end NUMINAMATH_CALUDE_shaded_area_problem_l834_83422


namespace NUMINAMATH_CALUDE_kaleb_books_l834_83449

theorem kaleb_books (initial_books sold_books new_books : ℕ) : 
  initial_books = 34 → sold_books = 17 → new_books = 7 →
  initial_books - sold_books + new_books = 24 :=
by sorry

end NUMINAMATH_CALUDE_kaleb_books_l834_83449


namespace NUMINAMATH_CALUDE_fraction_equality_l834_83465

theorem fraction_equality (a b : ℝ) (h1 : 2 * a = 3 * b) (h2 : b ≠ 0) : a / b = 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l834_83465


namespace NUMINAMATH_CALUDE_shopkeeper_loss_percentage_l834_83452

theorem shopkeeper_loss_percentage
  (initial_value : ℝ)
  (profit_percentage : ℝ)
  (stolen_percentage : ℝ)
  (sales_tax_percentage : ℝ)
  (h_profit : profit_percentage = 20)
  (h_stolen : stolen_percentage = 85)
  (h_tax : sales_tax_percentage = 5)
  (h_positive : initial_value > 0) :
  let selling_price := initial_value * (1 + profit_percentage / 100)
  let remaining_value := initial_value * (1 - stolen_percentage / 100)
  let after_tax_value := remaining_value * (1 - sales_tax_percentage / 100)
  let loss := selling_price - after_tax_value
  loss / selling_price * 100 = 88.125 := by
sorry

end NUMINAMATH_CALUDE_shopkeeper_loss_percentage_l834_83452


namespace NUMINAMATH_CALUDE_doughnuts_per_box_l834_83474

theorem doughnuts_per_box (total : ℕ) (boxes : ℕ) (h1 : total = 48) (h2 : boxes = 4) :
  total / boxes = 12 := by
  sorry

end NUMINAMATH_CALUDE_doughnuts_per_box_l834_83474


namespace NUMINAMATH_CALUDE_regular_tetrahedron_is_connected_l834_83424

-- Define a point in 3D space
structure Point3D where
  x : ℝ
  y : ℝ
  z : ℝ

-- Define a regular tetrahedron
def RegularTetrahedron : Set Point3D := sorry

-- Define a line segment between two points
def LineSegment (p q : Point3D) : Set Point3D := sorry

-- Define the property of being a connected set
def IsConnectedSet (S : Set Point3D) : Prop :=
  ∀ p q : Point3D, p ∈ S → q ∈ S → LineSegment p q ⊆ S

-- Theorem statement
theorem regular_tetrahedron_is_connected : IsConnectedSet RegularTetrahedron := by
  sorry

end NUMINAMATH_CALUDE_regular_tetrahedron_is_connected_l834_83424


namespace NUMINAMATH_CALUDE_acme_cheaper_min_shirts_l834_83481

def acme_cost (x : ℕ) : ℚ := 50 + 9 * x
def beta_cost (x : ℕ) : ℚ := 25 + 15 * x

theorem acme_cheaper_min_shirts : 
  ∀ n : ℕ, (∀ k : ℕ, k < n → acme_cost k ≥ beta_cost k) ∧ 
           (acme_cost n < beta_cost n) → n = 5 := by
  sorry

end NUMINAMATH_CALUDE_acme_cheaper_min_shirts_l834_83481


namespace NUMINAMATH_CALUDE_absolute_value_equation_solution_l834_83423

theorem absolute_value_equation_solution :
  ∃! x : ℝ, |x - 10| + |x - 14| = |2*x - 24| :=
by
  sorry

end NUMINAMATH_CALUDE_absolute_value_equation_solution_l834_83423


namespace NUMINAMATH_CALUDE_evaluate_expression_l834_83421

theorem evaluate_expression : (2^(2+1) - 4*(2-1)^2)^2 = 16 := by
  sorry

end NUMINAMATH_CALUDE_evaluate_expression_l834_83421


namespace NUMINAMATH_CALUDE_max_value_of_expression_l834_83477

theorem max_value_of_expression (a b c : ℝ) (h : a^2 + b^2 + c^2 = 9) :
  (∃ x y z : ℝ, x^2 + y^2 + z^2 = 9 ∧ (x - y)^2 + (y - z)^2 + (z - x)^2 ≥ (a - b)^2 + (b - c)^2 + (c - a)^2) ∧
  (a - b)^2 + (b - c)^2 + (c - a)^2 ≤ 27 :=
by sorry

end NUMINAMATH_CALUDE_max_value_of_expression_l834_83477


namespace NUMINAMATH_CALUDE_solve_for_y_l834_83439

theorem solve_for_y (x y : ℝ) (h1 : x - y = 10) (h2 : x + y = 18) : y = 4 := by
  sorry

end NUMINAMATH_CALUDE_solve_for_y_l834_83439


namespace NUMINAMATH_CALUDE_max_value_of_trigonometric_function_l834_83434

theorem max_value_of_trigonometric_function :
  let y : ℝ → ℝ := λ x => Real.tan (x + 2 * Real.pi / 3) - Real.tan (x + Real.pi / 6) + Real.cos (x + Real.pi / 6)
  ∃ (max_y : ℝ), max_y = 11 / 6 * Real.sqrt 3 ∧
    ∀ x ∈ Set.Icc (-5 * Real.pi / 12) (-Real.pi / 3), y x ≤ max_y :=
by
  sorry

end NUMINAMATH_CALUDE_max_value_of_trigonometric_function_l834_83434


namespace NUMINAMATH_CALUDE_parallelogram_area_l834_83495

/-- The area of a parallelogram with one angle of 150 degrees and two consecutive sides of lengths 10 inches and 20 inches is 100√3 square inches. -/
theorem parallelogram_area (a b : ℝ) (θ : ℝ) : 
  a = 10 → b = 20 → θ = 150 * π / 180 → 
  a * b * Real.sin (π - θ) = 100 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_parallelogram_area_l834_83495


namespace NUMINAMATH_CALUDE_jelly_bean_distribution_l834_83417

theorem jelly_bean_distribution (n : ℕ) (h1 : 10 ≤ n) (h2 : n ≤ 20) : 
  (∃ (total : ℕ), total = n^2 ∧ total % 5 = 0) → 
  (∃ (per_bag : ℕ), per_bag = 45 ∧ 5 * per_bag = n^2) := by
sorry

end NUMINAMATH_CALUDE_jelly_bean_distribution_l834_83417


namespace NUMINAMATH_CALUDE_original_number_proof_l834_83451

theorem original_number_proof (x : ℝ) : 
  (1.1 * x = 660) → (x = 600) := by
  sorry

end NUMINAMATH_CALUDE_original_number_proof_l834_83451


namespace NUMINAMATH_CALUDE_water_level_rise_l834_83420

/-- The rise in water level when a cube is immersed in a rectangular vessel --/
theorem water_level_rise (cube_edge : ℝ) (vessel_length vessel_width : ℝ) 
  (h_cube_edge : cube_edge = 10)
  (h_vessel_length : vessel_length = 20)
  (h_vessel_width : vessel_width = 15) :
  (cube_edge ^ 3) / (vessel_length * vessel_width) = 10 ^ 3 / (20 * 15) :=
by sorry

end NUMINAMATH_CALUDE_water_level_rise_l834_83420


namespace NUMINAMATH_CALUDE_quadratic_one_solution_sum_l834_83415

theorem quadratic_one_solution_sum (b : ℝ) : 
  (∃! x : ℝ, 6 * x^2 + b * x + 12 * x + 18 = 0) →
  (∃ b₁ b₂ : ℝ, b = b₁ ∨ b = b₂) ∧ (b₁ + b₂ = -24) :=
by sorry

end NUMINAMATH_CALUDE_quadratic_one_solution_sum_l834_83415


namespace NUMINAMATH_CALUDE_ramanujan_hardy_complex_game_l834_83491

theorem ramanujan_hardy_complex_game (product h r : ℂ) : 
  product = 24 - 10*I ∧ h = 3 + 4*I ∧ product = h * r →
  r = 112/25 - 126/25*I := by sorry

end NUMINAMATH_CALUDE_ramanujan_hardy_complex_game_l834_83491


namespace NUMINAMATH_CALUDE_baking_scoops_calculation_l834_83443

/-- Calculates the total number of scoops needed for baking a cake --/
def total_scoops (flour_cups : ℚ) (sugar_cups : ℚ) (scoop_size : ℚ) : ℕ :=
  (flour_cups / scoop_size + sugar_cups / scoop_size).ceil.toNat

/-- Proves that given 3 cups of flour, 2 cups of sugar, and a 1/3 cup scoop, 
    the total number of scoops needed is 15 --/
theorem baking_scoops_calculation : 
  total_scoops 3 2 (1/3) = 15 := by sorry

end NUMINAMATH_CALUDE_baking_scoops_calculation_l834_83443


namespace NUMINAMATH_CALUDE_two_people_available_l834_83426

-- Define the types for people and days
inductive Person : Type
| Anna : Person
| Bill : Person
| Carl : Person
| Dana : Person

inductive Day : Type
| Monday : Day
| Tuesday : Day
| Wednesday : Day
| Thursday : Day
| Friday : Day
| Saturday : Day

-- Define a function to represent availability
def isAvailable : Person → Day → Bool
| Person.Anna, Day.Monday => false
| Person.Anna, Day.Tuesday => true
| Person.Anna, Day.Wednesday => false
| Person.Anna, Day.Thursday => true
| Person.Anna, Day.Friday => true
| Person.Anna, Day.Saturday => false
| Person.Bill, Day.Monday => true
| Person.Bill, Day.Tuesday => false
| Person.Bill, Day.Wednesday => true
| Person.Bill, Day.Thursday => false
| Person.Bill, Day.Friday => false
| Person.Bill, Day.Saturday => true
| Person.Carl, Day.Monday => false
| Person.Carl, Day.Tuesday => false
| Person.Carl, Day.Wednesday => true
| Person.Carl, Day.Thursday => false
| Person.Carl, Day.Friday => false
| Person.Carl, Day.Saturday => true
| Person.Dana, Day.Monday => true
| Person.Dana, Day.Tuesday => true
| Person.Dana, Day.Wednesday => false
| Person.Dana, Day.Thursday => true
| Person.Dana, Day.Friday => true
| Person.Dana, Day.Saturday => false

-- Define a function to count available people for a given day
def countAvailable (d : Day) : Nat :=
  List.foldl (λ count p => if isAvailable p d then count + 1 else count) 0 [Person.Anna, Person.Bill, Person.Carl, Person.Dana]

-- Theorem: For each day, exactly 2 people can attend the meeting
theorem two_people_available (d : Day) : countAvailable d = 2 := by
  sorry

#eval [Day.Monday, Day.Tuesday, Day.Wednesday, Day.Thursday, Day.Friday, Day.Saturday].map countAvailable

end NUMINAMATH_CALUDE_two_people_available_l834_83426


namespace NUMINAMATH_CALUDE_train_length_problem_l834_83435

theorem train_length_problem (faster_speed slower_speed : ℝ) 
  (passing_time : ℝ) (h1 : faster_speed = 46) (h2 : slower_speed = 36) 
  (h3 : passing_time = 18) : ∃ (train_length : ℝ), 
  train_length = 50 ∧ 
  train_length * 1000 = (faster_speed - slower_speed) * passing_time / 3600 := by
  sorry

end NUMINAMATH_CALUDE_train_length_problem_l834_83435


namespace NUMINAMATH_CALUDE_reaping_capacity_theorem_l834_83433

/-- Represents the reaping capacity of a group of men -/
structure ReapingCapacity where
  men : ℕ
  hectares : ℝ
  days : ℕ

/-- Given the reaping capacity of one group, calculate the reaping capacity of another group -/
def calculate_reaping_capacity (base : ReapingCapacity) (target : ReapingCapacity) : Prop :=
  (target.men : ℝ) / base.men * (base.hectares / base.days) * target.days = target.hectares

/-- Theorem stating the relationship between the reaping capacities of two groups -/
theorem reaping_capacity_theorem (base target : ReapingCapacity) :
  base.men = 10 ∧ base.hectares = 80 ∧ base.days = 24 ∧
  target.men = 36 ∧ target.hectares = 360 ∧ target.days = 30 →
  calculate_reaping_capacity base target := by
  sorry

end NUMINAMATH_CALUDE_reaping_capacity_theorem_l834_83433


namespace NUMINAMATH_CALUDE_max_sum_of_factors_l834_83425

theorem max_sum_of_factors (A B C : ℕ) : 
  A ≠ B ∧ B ≠ C ∧ A ≠ C →
  A > 0 ∧ B > 0 ∧ C > 0 →
  A * B * C = 2550 →
  A + B + C ≤ 98 := by
sorry

end NUMINAMATH_CALUDE_max_sum_of_factors_l834_83425


namespace NUMINAMATH_CALUDE_sequence_integer_value_l834_83440

def u (M : ℤ) : ℕ → ℚ
  | 0 => M + 1/2
  | n + 1 => u M n * ⌊u M n⌋

theorem sequence_integer_value (M : ℤ) (h : M ≥ 1) :
  (∃ n : ℕ, ∃ k : ℤ, u M n = k) ↔ M > 1 :=
sorry

end NUMINAMATH_CALUDE_sequence_integer_value_l834_83440


namespace NUMINAMATH_CALUDE_duplicate_page_sum_l834_83413

theorem duplicate_page_sum (n : ℕ) (p : ℕ) : 
  p ≤ n →
  n * (n + 1) / 2 + p = 3005 →
  p = 2 :=
sorry

end NUMINAMATH_CALUDE_duplicate_page_sum_l834_83413


namespace NUMINAMATH_CALUDE_power_minus_product_equals_one_l834_83414

theorem power_minus_product_equals_one : 3^2 - (4 * 2) = 1 := by
  sorry

end NUMINAMATH_CALUDE_power_minus_product_equals_one_l834_83414


namespace NUMINAMATH_CALUDE_stating_pond_population_species_c_l834_83485

/-- Represents the number of fish initially tagged for each species -/
def initial_tagged : ℕ := 40

/-- Represents the total number of fish caught in the second catch -/
def second_catch : ℕ := 180

/-- Represents the number of tagged fish of Species C found in the second catch -/
def tagged_species_c : ℕ := 2

/-- Represents the total number of fish of Species C in the pond -/
def total_species_c : ℕ := 3600

/-- 
Theorem stating that given the conditions from the problem, 
the total number of fish for Species C in the pond is 3600 
-/
theorem pond_population_species_c : 
  initial_tagged * second_catch / tagged_species_c = total_species_c := by
  sorry

end NUMINAMATH_CALUDE_stating_pond_population_species_c_l834_83485


namespace NUMINAMATH_CALUDE_original_number_proof_l834_83411

theorem original_number_proof (x : ℝ) : 
  (((x + 5) - (x - 5)) / (x + 5)) * 100 = 76.92 → x = 8 := by
  sorry

end NUMINAMATH_CALUDE_original_number_proof_l834_83411


namespace NUMINAMATH_CALUDE_quadratic_minimum_l834_83479

theorem quadratic_minimum (x : ℝ) : 
  (∀ x, 2 * x^2 - 8 * x + 15 ≥ 7) ∧ (∃ x, 2 * x^2 - 8 * x + 15 = 7) := by
  sorry

end NUMINAMATH_CALUDE_quadratic_minimum_l834_83479


namespace NUMINAMATH_CALUDE_function_equality_l834_83463

theorem function_equality (f g h k : ℝ → ℝ) (a b : ℝ) 
  (h1 : ∀ x, f x = (x - 1) * g x + 3)
  (h2 : ∀ x, f x = (x + 1) * h x + 1)
  (h3 : ∀ x, f x = (x^2 - 1) * k x + a * x + b) :
  a = 1 ∧ b = 2 := by
  sorry

end NUMINAMATH_CALUDE_function_equality_l834_83463


namespace NUMINAMATH_CALUDE_product_of_roots_l834_83459

theorem product_of_roots (x : ℝ) : (x + 3) * (x - 5) = 20 → ∃ y : ℝ, (x + 3) * (x - 5) = 20 ∧ (y + 3) * (y - 5) = 20 ∧ x * y = -35 := by
  sorry

end NUMINAMATH_CALUDE_product_of_roots_l834_83459


namespace NUMINAMATH_CALUDE_travel_time_calculation_l834_83497

theorem travel_time_calculation (total_distance : ℝ) (average_speed : ℝ) (return_time : ℝ) :
  total_distance = 2000 ∧ 
  average_speed = 142.85714285714286 ∧ 
  return_time = 4 →
  total_distance / average_speed - return_time = 10 := by
  sorry

end NUMINAMATH_CALUDE_travel_time_calculation_l834_83497


namespace NUMINAMATH_CALUDE_line_intersection_with_axes_l834_83405

/-- A line passing through two given points intersects the x-axis and y-axis at specific points -/
theorem line_intersection_with_axes (x₁ y₁ x₂ y₂ : ℝ) :
  let m : ℝ := (y₂ - y₁) / (x₂ - x₁)
  let b : ℝ := y₁ - m * x₁
  let line : ℝ → ℝ := λ x => m * x + b
  x₁ = 8 ∧ y₁ = 2 ∧ x₂ = 4 ∧ y₂ = 6 →
  (∃ x : ℝ, line x = 0 ∧ x = 10) ∧
  (∃ y : ℝ, line 0 = y ∧ y = 10) :=
by sorry

#check line_intersection_with_axes

end NUMINAMATH_CALUDE_line_intersection_with_axes_l834_83405


namespace NUMINAMATH_CALUDE_correct_num_persons_first_group_l834_83430

/-- The number of persons in the first group that can repair a road -/
def num_persons_first_group : ℕ := 39

/-- The number of days the first group works -/
def days_first_group : ℕ := 12

/-- The number of hours per day the first group works -/
def hours_per_day_first_group : ℕ := 5

/-- The number of persons in the second group -/
def num_persons_second_group : ℕ := 30

/-- The number of days the second group works -/
def days_second_group : ℕ := 13

/-- The number of hours per day the second group works -/
def hours_per_day_second_group : ℕ := 6

/-- Theorem stating that the number of persons in the first group is correct -/
theorem correct_num_persons_first_group :
  num_persons_first_group * days_first_group * hours_per_day_first_group =
  num_persons_second_group * days_second_group * hours_per_day_second_group :=
by sorry

end NUMINAMATH_CALUDE_correct_num_persons_first_group_l834_83430


namespace NUMINAMATH_CALUDE_oldest_child_age_l834_83478

theorem oldest_child_age (age1 age2 age3 : ℕ) : 
  age1 = 6 → age2 = 8 → (age1 + age2 + age3) / 3 = 10 → age3 = 16 := by
sorry

end NUMINAMATH_CALUDE_oldest_child_age_l834_83478


namespace NUMINAMATH_CALUDE_modified_short_bingo_arrangements_l834_83409

theorem modified_short_bingo_arrangements : Nat.factorial 15 / Nat.factorial 8 = 1816214400 := by
  sorry

end NUMINAMATH_CALUDE_modified_short_bingo_arrangements_l834_83409


namespace NUMINAMATH_CALUDE_max_type_a_accessories_l834_83429

/-- Represents the cost and quantity of drone accessories. -/
structure DroneAccessories where
  costA : ℕ  -- Cost of type A accessory
  costB : ℕ  -- Cost of type B accessory
  totalQuantity : ℕ  -- Total number of accessories
  maxCost : ℕ  -- Maximum total cost

/-- Calculates the maximum number of type A accessories that can be purchased. -/
def maxTypeA (d : DroneAccessories) : ℕ :=
  let m := (d.maxCost - d.costB * d.totalQuantity) / (d.costA - d.costB)
  min m d.totalQuantity

/-- Theorem stating the maximum number of type A accessories that can be purchased. -/
theorem max_type_a_accessories (d : DroneAccessories) : 
  d.costA = 230 ∧ d.costB = 100 ∧ d.totalQuantity = 30 ∧ d.maxCost = 4180 ∧
  d.costA + 3 * d.costB = 530 ∧ 3 * d.costA + 2 * d.costB = 890 →
  maxTypeA d = 9 := by
  sorry

#eval maxTypeA { costA := 230, costB := 100, totalQuantity := 30, maxCost := 4180 }

end NUMINAMATH_CALUDE_max_type_a_accessories_l834_83429


namespace NUMINAMATH_CALUDE_fraction_simplification_l834_83488

theorem fraction_simplification (x : ℝ) (h1 : x ≠ -1) (h2 : x ≠ 2) :
  (3*x^3 - 2*x^2 - 5*x + 1) / ((x+1)*(x-2)) - (2*x^2 - 7*x + 3) / ((x+1)*(x-2)) =
  (x-1)*(3*x^2 - x + 2) / ((x+1)*(x-2)) := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l834_83488


namespace NUMINAMATH_CALUDE_transform_equation_5x2_eq_6x_minus_8_l834_83400

/-- Represents a quadratic equation in general form ax² + bx + c = 0 --/
structure QuadraticEquation where
  a : ℝ
  b : ℝ
  c : ℝ
  h : a ≠ 0

/-- Transforms an equation of the form px² = qx + r into general quadratic form --/
def transform_to_general_form (p q r : ℝ) (hp : p ≠ 0) : QuadraticEquation :=
  { a := p
  , b := -q
  , c := r
  , h := hp }

theorem transform_equation_5x2_eq_6x_minus_8 :
  let eq := transform_to_general_form 5 6 (-8) (by norm_num)
  eq.a = 5 ∧ eq.b = -6 ∧ eq.c = 8 := by sorry

end NUMINAMATH_CALUDE_transform_equation_5x2_eq_6x_minus_8_l834_83400


namespace NUMINAMATH_CALUDE_sin_3alpha_inequality_l834_83468

theorem sin_3alpha_inequality (α : Real) (h1 : 0 < α) (h2 : α < Real.pi / 6) :
  2 * Real.sin α < Real.sin (3 * α) ∧ Real.sin (3 * α) < 3 * Real.sin α := by
  sorry

end NUMINAMATH_CALUDE_sin_3alpha_inequality_l834_83468


namespace NUMINAMATH_CALUDE_initial_cookies_l834_83404

/-- Given that 2 cookies were eaten and 5 cookies remain, prove that the initial number of cookies was 7. -/
theorem initial_cookies (eaten : ℕ) (remaining : ℕ) (h1 : eaten = 2) (h2 : remaining = 5) :
  eaten + remaining = 7 := by
  sorry

end NUMINAMATH_CALUDE_initial_cookies_l834_83404


namespace NUMINAMATH_CALUDE_initial_barking_dogs_l834_83467

theorem initial_barking_dogs (initial : ℕ) (additional : ℕ) (total : ℕ) : 
  additional = 10 → total = 40 → initial + additional = total → initial = 30 := by
sorry

end NUMINAMATH_CALUDE_initial_barking_dogs_l834_83467


namespace NUMINAMATH_CALUDE_mp3_player_problem_l834_83412

def initial_songs : Nat := 8
def deleted_songs : Nat := 5
def added_songs : Nat := 30
def added_song_durations : List Nat := [3, 4, 2, 6, 5, 3, 4, 5, 6, 2, 3, 4, 5, 6, 7, 2, 3, 4, 5, 6, 7, 8, 4, 3, 5, 6, 7, 8, 9, 10]

theorem mp3_player_problem :
  (initial_songs - deleted_songs + added_songs = 33) ∧
  (added_song_durations.sum = 145) := by
  sorry

end NUMINAMATH_CALUDE_mp3_player_problem_l834_83412


namespace NUMINAMATH_CALUDE_greatest_power_of_three_l834_83454

def w : ℕ := (List.range 30).foldl (· * ·) 1

theorem greatest_power_of_three (p : ℕ) : 
  (3^p ∣ w) ∧ ∀ q, q > p → ¬(3^q ∣ w) ↔ p = 15 :=
sorry

end NUMINAMATH_CALUDE_greatest_power_of_three_l834_83454


namespace NUMINAMATH_CALUDE_intersection_P_complement_Q_l834_83471

universe u

def U : Set (Fin 7) := {1, 2, 3, 4, 5, 6, 7}
def P : Set (Fin 7) := {1, 2, 3, 4, 5}
def Q : Set (Fin 7) := {3, 4, 5, 6, 7}

theorem intersection_P_complement_Q : P ∩ (U \ Q) = {1, 2} := by
  sorry

end NUMINAMATH_CALUDE_intersection_P_complement_Q_l834_83471


namespace NUMINAMATH_CALUDE_fraction_simplification_l834_83484

theorem fraction_simplification :
  ((2^1010)^2 - (2^1008)^2) / ((2^1009)^2 - (2^1007)^2) = 4 := by
  sorry

end NUMINAMATH_CALUDE_fraction_simplification_l834_83484


namespace NUMINAMATH_CALUDE_debby_dvd_count_l834_83437

/-- The number of DVDs Debby sold -/
def sold_dvds : ℕ := 6

/-- The number of DVDs Debby had left after selling -/
def remaining_dvds : ℕ := 7

/-- The initial number of DVDs Debby owned -/
def initial_dvds : ℕ := sold_dvds + remaining_dvds

theorem debby_dvd_count : initial_dvds = 13 := by sorry

end NUMINAMATH_CALUDE_debby_dvd_count_l834_83437


namespace NUMINAMATH_CALUDE_original_selling_price_l834_83489

theorem original_selling_price (P : ℝ) : 
  (P + 0.1 * P) - ((0.9 * P) + 0.3 * (0.9 * P)) = 70 → 
  P + 0.1 * P = 1100 := by
sorry

end NUMINAMATH_CALUDE_original_selling_price_l834_83489


namespace NUMINAMATH_CALUDE_mitch_family_milk_consumption_l834_83493

/-- The total milk consumption in cartons for Mitch's family in one week -/
def total_milk_consumption (regular_milk soy_milk : ℝ) : ℝ :=
  regular_milk + soy_milk

/-- Proof that Mitch's family's total milk consumption is 0.6 cartons in one week -/
theorem mitch_family_milk_consumption :
  let regular_milk : ℝ := 0.5
  let soy_milk : ℝ := 0.1
  total_milk_consumption regular_milk soy_milk = 0.6 := by
  sorry

end NUMINAMATH_CALUDE_mitch_family_milk_consumption_l834_83493


namespace NUMINAMATH_CALUDE_mat_weavers_problem_l834_83408

/-- The number of mat-weavers in the first group -/
def first_group_weavers : ℕ := 4

/-- The number of mats woven by the first group -/
def first_group_mats : ℕ := 4

/-- The number of days taken by the first group -/
def first_group_days : ℕ := 4

/-- The number of mat-weavers in the second group -/
def second_group_weavers : ℕ := 16

/-- The number of mats woven by the second group -/
def second_group_mats : ℕ := 64

/-- The number of days taken by the second group -/
def second_group_days : ℕ := 16

theorem mat_weavers_problem :
  first_group_weavers * second_group_mats * first_group_days =
  second_group_weavers * first_group_mats * second_group_days :=
by sorry

end NUMINAMATH_CALUDE_mat_weavers_problem_l834_83408


namespace NUMINAMATH_CALUDE_product_of_fractions_and_root_l834_83442

theorem product_of_fractions_and_root : 
  (2 : ℝ) / 3 * (3 : ℝ) / 5 * ((4 : ℝ) / 7) ^ (1 / 2) = 4 * Real.sqrt 7 / 35 := by
  sorry

end NUMINAMATH_CALUDE_product_of_fractions_and_root_l834_83442


namespace NUMINAMATH_CALUDE_cubic_polynomial_roots_l834_83436

/-- A cubic polynomial with rational coefficients -/
structure CubicPolynomial where
  a : ℚ
  b : ℚ
  c : ℚ

/-- Predicate to check if a, b, c are roots of the polynomial -/
def has_roots (p : CubicPolynomial) : Prop :=
  (p.a^3 + p.a * p.a^2 + p.b * p.a + p.c = 0) ∧
  (p.b^3 + p.a * p.b^2 + p.b * p.b + p.c = 0) ∧
  (p.c^3 + p.a * p.c^2 + p.b * p.c + p.c = 0)

/-- The set of valid polynomials -/
def valid_polynomials : Set CubicPolynomial :=
  {⟨0, 0, 0⟩, ⟨1, -2, 0⟩}

theorem cubic_polynomial_roots (p : CubicPolynomial) :
  has_roots p ↔ p ∈ valid_polynomials := by sorry

end NUMINAMATH_CALUDE_cubic_polynomial_roots_l834_83436


namespace NUMINAMATH_CALUDE_min_k_for_f_geq_3_solution_set_f_lt_3x_l834_83428

-- Define the function f(x, k)
def f (x k : ℝ) : ℝ := |x - 3| + |x - 2| + k

-- Theorem for part I
theorem min_k_for_f_geq_3 :
  (∀ x : ℝ, f x 2 ≥ 3) ∧ (∀ k < 2, ∃ x : ℝ, f x k < 3) :=
sorry

-- Theorem for part II
theorem solution_set_f_lt_3x :
  {x : ℝ | f x 1 < 3 * x} = {x : ℝ | x > 6/5} :=
sorry

end NUMINAMATH_CALUDE_min_k_for_f_geq_3_solution_set_f_lt_3x_l834_83428


namespace NUMINAMATH_CALUDE_total_profit_calculation_l834_83403

-- Define the profit for 3 shirts
def profit_3_shirts : ℚ := 21

-- Define the profit for 2 pairs of sandals
def profit_2_sandals : ℚ := 4 * profit_3_shirts

-- Define the number of shirts and sandals sold
def shirts_sold : ℕ := 7
def sandals_sold : ℕ := 3

-- Theorem statement
theorem total_profit_calculation :
  (shirts_sold * (profit_3_shirts / 3) + sandals_sold * (profit_2_sandals / 2)) = 175 := by
  sorry


end NUMINAMATH_CALUDE_total_profit_calculation_l834_83403


namespace NUMINAMATH_CALUDE_triangle_altitude_l834_83490

/-- Given a triangle with area 800 square feet and base 40 feet, its altitude is 40 feet. -/
theorem triangle_altitude (area : ℝ) (base : ℝ) (altitude : ℝ) : 
  area = 800 → base = 40 → area = (1/2) * base * altitude → altitude = 40 := by
  sorry

end NUMINAMATH_CALUDE_triangle_altitude_l834_83490


namespace NUMINAMATH_CALUDE_mr_green_potato_yield_l834_83418

/-- Represents the dimensions of a rectangular garden in steps -/
structure GardenDimensions where
  length : ℕ
  width : ℕ

/-- Calculates the expected potato yield from a garden -/
def expected_potato_yield (garden : GardenDimensions) (step_length : ℕ) (yield_per_sqft : ℚ) : ℚ :=
  (garden.length * step_length * (garden.width * step_length) : ℚ) * yield_per_sqft

/-- Theorem stating the expected potato yield for Mr. Green's garden -/
theorem mr_green_potato_yield :
  let garden := GardenDimensions.mk 15 20
  let step_length := 2
  let yield_per_sqft := 1/2
  expected_potato_yield garden step_length yield_per_sqft = 600 := by
  sorry

end NUMINAMATH_CALUDE_mr_green_potato_yield_l834_83418


namespace NUMINAMATH_CALUDE_max_product_constraint_l834_83473

theorem max_product_constraint (a b : ℝ) (ha : a > 0) (hb : b > 0) (h : a/4 + b/5 = 1) :
  a * b ≤ 5 := by
sorry

end NUMINAMATH_CALUDE_max_product_constraint_l834_83473


namespace NUMINAMATH_CALUDE_a_equals_3_necessary_not_sufficient_l834_83475

/-- Two lines in the plane -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Condition for two lines to be parallel -/
def parallel (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l1.b * l2.a ∧ l1.a * l2.c ≠ l1.c * l2.a

/-- The first line ax-2y-1=0 -/
def line1 (a : ℝ) : Line :=
  { a := a, b := -2, c := -1 }

/-- The second line 6x-4y+c=0 -/
def line2 (c : ℝ) : Line :=
  { a := 6, b := -4, c := c }

theorem a_equals_3_necessary_not_sufficient :
  (∀ c, parallel (line1 3) (line2 c)) ∧
  (∃ a c, a ≠ 3 ∧ parallel (line1 a) (line2 c)) :=
sorry

end NUMINAMATH_CALUDE_a_equals_3_necessary_not_sufficient_l834_83475


namespace NUMINAMATH_CALUDE_snowflake_weight_scientific_notation_l834_83438

theorem snowflake_weight_scientific_notation :
  ∃ (a : ℝ) (n : ℤ), 0.00003 = a * 10^n ∧ 1 ≤ a ∧ a < 10 :=
by
  -- The proof goes here
  sorry

end NUMINAMATH_CALUDE_snowflake_weight_scientific_notation_l834_83438


namespace NUMINAMATH_CALUDE_lines_cannot_form_triangle_l834_83462

/-- A line in the 2D plane represented by ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

/-- Check if two lines are parallel -/
def parallel (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l1.b * l2.a

/-- Check if three lines intersect at a single point -/
def intersect_at_point (l1 l2 l3 : Line) : Prop :=
  ∃ x y : ℝ, l1.a * x + l1.b * y + l1.c = 0 ∧
             l2.a * x + l2.b * y + l2.c = 0 ∧
             l3.a * x + l3.b * y + l3.c = 0

/-- The set of m values for which the three lines cannot form a triangle -/
def m_values : Set ℝ := {-3, 2, -1}

theorem lines_cannot_form_triangle (m : ℝ) :
  let l1 : Line := ⟨3, -1, 2⟩
  let l2 : Line := ⟨2, 1, 3⟩
  let l3 : Line := ⟨m, 1, 0⟩
  (parallel l1 l3 ∨ parallel l2 l3 ∨ intersect_at_point l1 l2 l3) ↔ m ∈ m_values := by
  sorry

end NUMINAMATH_CALUDE_lines_cannot_form_triangle_l834_83462


namespace NUMINAMATH_CALUDE_full_price_revenue_l834_83470

/-- Represents the ticket sales data for a charity event -/
structure TicketSales where
  fullPrice : ℕ  -- Price of a full-price ticket in dollars
  fullCount : ℕ  -- Number of full-price tickets sold
  halfCount : ℕ  -- Number of half-price tickets sold
  premiumCount : ℕ := 12  -- Number of premium tickets sold (fixed at 12)

/-- Calculates the total number of tickets sold -/
def TicketSales.totalTickets (ts : TicketSales) : ℕ :=
  ts.fullCount + ts.halfCount + ts.premiumCount

/-- Calculates the total revenue from all ticket sales -/
def TicketSales.totalRevenue (ts : TicketSales) : ℕ :=
  ts.fullPrice * ts.fullCount + 
  (ts.fullPrice / 2) * ts.halfCount + 
  (2 * ts.fullPrice) * ts.premiumCount

/-- Theorem stating the revenue from full-price tickets -/
theorem full_price_revenue (ts : TicketSales) : 
  ts.totalTickets = 160 ∧ 
  ts.totalRevenue = 2514 ∧ 
  ts.fullPrice > 0 →
  ts.fullPrice * ts.fullCount = 770 := by
  sorry

#check full_price_revenue

end NUMINAMATH_CALUDE_full_price_revenue_l834_83470


namespace NUMINAMATH_CALUDE_higher_rate_fewer_attendees_possible_l834_83456

/-- Represents a workshop with attendees and total capacity -/
structure Workshop where
  attendees : ℕ
  capacity : ℕ
  attendance_rate : ℚ
  attendance_rate_def : attendance_rate = attendees / capacity

/-- Theorem stating that it's possible for a workshop to have a higher attendance rate
    but fewer attendees than another workshop -/
theorem higher_rate_fewer_attendees_possible :
  ∃ (A B : Workshop), A.attendance_rate > B.attendance_rate ∧ A.attendees < B.attendees := by
  sorry


end NUMINAMATH_CALUDE_higher_rate_fewer_attendees_possible_l834_83456


namespace NUMINAMATH_CALUDE_inequality_system_solutions_l834_83427

theorem inequality_system_solutions (m : ℝ) : 
  (∃ (s : Finset ℤ), s.card = 3 ∧ 
    (∀ x : ℤ, x ∈ s ↔ (x + 5 > 0 ∧ x - m ≤ 1))) ↔ 
  (-3 ≤ m ∧ m < -2) :=
sorry

end NUMINAMATH_CALUDE_inequality_system_solutions_l834_83427


namespace NUMINAMATH_CALUDE_triangle_abc_properties_l834_83432

theorem triangle_abc_properties (a b c : ℝ) (A B C : ℝ) (S : ℝ) :
  a * Real.sin B = Real.sqrt 3 * b * Real.cos A →
  a = Real.sqrt 3 →
  S = Real.sqrt 3 / 2 →
  0 < a ∧ 0 < b ∧ 0 < c →
  0 < A ∧ A < π ∧ 0 < B ∧ B < π ∧ 0 < C ∧ C < π →
  A + B + C = π →
  S = 1/2 * a * b * Real.sin C →
  A = π/3 ∧ b + c = 3 := by
  sorry

end NUMINAMATH_CALUDE_triangle_abc_properties_l834_83432


namespace NUMINAMATH_CALUDE_infinite_sum_evaluation_l834_83480

theorem infinite_sum_evaluation :
  (∑' n : ℕ, (3 : ℝ)^n / (1 + 3^n + 3^(n + 1) + 3^(2*n + 1))) = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_infinite_sum_evaluation_l834_83480


namespace NUMINAMATH_CALUDE_banana_bunch_count_l834_83448

theorem banana_bunch_count (x : ℕ) : 
  (6 * x + 5 * 7 = 83) → x = 8 := by
sorry

end NUMINAMATH_CALUDE_banana_bunch_count_l834_83448


namespace NUMINAMATH_CALUDE_smallest_odd_divisible_by_three_l834_83441

theorem smallest_odd_divisible_by_three :
  ∀ n : ℕ, n % 2 = 1 → n % 3 = 0 → n ≥ 3 :=
sorry

end NUMINAMATH_CALUDE_smallest_odd_divisible_by_three_l834_83441


namespace NUMINAMATH_CALUDE_relationship_between_x_squared_ax_bx_l834_83496

theorem relationship_between_x_squared_ax_bx
  (x a b : ℝ)
  (h1 : x < a)
  (h2 : a < 0)
  (h3 : b > 0) :
  x^2 > a*x ∧ a*x > b*x :=
by sorry

end NUMINAMATH_CALUDE_relationship_between_x_squared_ax_bx_l834_83496


namespace NUMINAMATH_CALUDE_measurable_eq_set_l834_83455

open MeasureTheory

variable {Ω : Type*} [MeasurableSpace Ω]
variable (F : MeasurableSpace Ω)
variable (ξ η : Ω → ℝ)

theorem measurable_eq_set (hξ : Measurable ξ) (hη : Measurable η) :
  MeasurableSet {ω | ξ ω = η ω} :=
by
  sorry

end NUMINAMATH_CALUDE_measurable_eq_set_l834_83455


namespace NUMINAMATH_CALUDE_question_1_question_2_l834_83461

-- Define the given conditions
def p (x : ℝ) : Prop := -x^2 + 2*x + 8 ≥ 0
def q (x m : ℝ) : Prop := x^2 - 2*x + 1 - m^2 ≤ 0
def s (x : ℝ) : Prop := -x^2 + 8*x + 20 ≥ 0

-- Define the sufficient but not necessary conditions
def sufficient_not_necessary (P Q : ℝ → Prop) : Prop :=
  (∀ x, P x → Q x) ∧ ∃ x, Q x ∧ ¬(P x)

-- Theorem for the first question
theorem question_1 (m : ℝ) :
  m > 0 → (sufficient_not_necessary (p) (q m) → m ≥ 3) :=
sorry

-- Theorem for the second question
theorem question_2 (m : ℝ) :
  m > 0 → (sufficient_not_necessary (fun x => ¬(s x)) (fun x => ¬(q m x)) → False) :=
sorry

end NUMINAMATH_CALUDE_question_1_question_2_l834_83461


namespace NUMINAMATH_CALUDE_find_b_plus_c_l834_83494

theorem find_b_plus_c (a b c d : ℚ)
  (eq1 : a * b + a * c + b * d + c * d = 40)
  (eq2 : a + d = 6)
  (eq3 : a * b + b * c + c * d + d * a = 28) :
  b + c = 17 / 3 := by
sorry

end NUMINAMATH_CALUDE_find_b_plus_c_l834_83494


namespace NUMINAMATH_CALUDE_ofelia_savings_l834_83450

/-- Represents the amount saved in a given month -/
def savings (month : ℕ) (initial : ℚ) : ℚ :=
  initial * 2^month

/-- Proves that if Ofelia saves twice the amount each month starting from January,
    and saves $160 in May, then she must have saved $10 in January -/
theorem ofelia_savings (initial : ℚ) :
  savings 4 initial = 160 → initial = 10 := by
  sorry

end NUMINAMATH_CALUDE_ofelia_savings_l834_83450


namespace NUMINAMATH_CALUDE_simple_interest_principal_calculation_l834_83460

/-- Proves that given a simple interest of 4025.25, an interest rate of 9% per annum, 
and a time period of 5 years, the principal sum is 8950. -/
theorem simple_interest_principal_calculation :
  let simple_interest : ℝ := 4025.25
  let rate : ℝ := 9
  let time : ℝ := 5
  let principal : ℝ := simple_interest / (rate * time / 100)
  principal = 8950 := by sorry

end NUMINAMATH_CALUDE_simple_interest_principal_calculation_l834_83460


namespace NUMINAMATH_CALUDE_equation_solutions_l834_83445

theorem equation_solutions :
  (∃ x1 x2 : ℝ, x1 = 1 + (Real.sqrt 10) / 2 ∧ 
                x2 = 1 - (Real.sqrt 10) / 2 ∧ 
                2 * x1^2 - 4 * x1 - 3 = 0 ∧ 
                2 * x2^2 - 4 * x2 - 3 = 0) ∧
  (∃ x1 x2 : ℝ, x1 = -3 ∧ 
                x2 = 2 ∧ 
                (x1^2 + x1)^2 - x1^2 - x1 = 30 ∧ 
                (x2^2 + x2)^2 - x2^2 - x2 = 30) :=
by sorry

end NUMINAMATH_CALUDE_equation_solutions_l834_83445


namespace NUMINAMATH_CALUDE_total_pints_picked_l834_83483

def annie_pints : ℕ := 8

def kathryn_pints (annie : ℕ) : ℕ := annie + 2

def ben_pints (kathryn : ℕ) : ℕ := kathryn - 3

theorem total_pints_picked :
  annie_pints + kathryn_pints annie_pints + ben_pints (kathryn_pints annie_pints) = 25 := by
  sorry

end NUMINAMATH_CALUDE_total_pints_picked_l834_83483


namespace NUMINAMATH_CALUDE_john_age_is_thirteen_l834_83402

/-- Represents John's work and earnings over a six-month period --/
structure JohnWork where
  hoursPerDay : ℕ
  hourlyRatePerAge : ℚ
  weeklyBonusThreshold : ℕ
  weeklyBonus : ℚ
  totalDaysWorked : ℕ
  totalEarned : ℚ

/-- Calculates John's age based on his work and earnings --/
def calculateAge (work : JohnWork) : ℕ :=
  sorry

/-- Theorem stating that John's calculated age is 13 --/
theorem john_age_is_thirteen (work : JohnWork) 
  (h1 : work.hoursPerDay = 3)
  (h2 : work.hourlyRatePerAge = 1/2)
  (h3 : work.weeklyBonusThreshold = 3)
  (h4 : work.weeklyBonus = 5)
  (h5 : work.totalDaysWorked = 75)
  (h6 : work.totalEarned = 900) :
  calculateAge work = 13 :=
sorry

end NUMINAMATH_CALUDE_john_age_is_thirteen_l834_83402


namespace NUMINAMATH_CALUDE_smallest_candy_count_l834_83431

theorem smallest_candy_count (x : ℕ) : 
  x > 0 ∧ 
  x % 6 = 5 ∧ 
  x % 8 = 3 ∧ 
  x % 9 = 7 ∧
  (∀ y : ℕ, y > 0 → y % 6 = 5 → y % 8 = 3 → y % 9 = 7 → x ≤ y) → 
  x = 203 := by
sorry

end NUMINAMATH_CALUDE_smallest_candy_count_l834_83431


namespace NUMINAMATH_CALUDE_total_area_calculation_l834_83406

/-- Calculates the total area of rooms given initial dimensions and modifications --/
theorem total_area_calculation (initial_length initial_width increase : ℕ) : 
  let new_length : ℕ := initial_length + increase
  let new_width : ℕ := initial_width + increase
  let single_room_area : ℕ := new_length * new_width
  let total_area : ℕ := 4 * single_room_area + 2 * single_room_area
  (initial_length = 13 ∧ initial_width = 18 ∧ increase = 2) → total_area = 1800 := by
  sorry

#check total_area_calculation

end NUMINAMATH_CALUDE_total_area_calculation_l834_83406


namespace NUMINAMATH_CALUDE_first_restaurant_meals_first_restaurant_meals_proof_l834_83457

theorem first_restaurant_meals (total_restaurants : Nat) 
  (second_restaurant_meals : Nat) (third_restaurant_meals : Nat) 
  (total_weekly_meals : Nat) (days_per_week : Nat) : Nat :=
  let first_restaurant_daily_meals := 
    (total_weekly_meals - (second_restaurant_meals + third_restaurant_meals) * days_per_week) / days_per_week
  first_restaurant_daily_meals

#check @first_restaurant_meals

theorem first_restaurant_meals_proof 
  (h1 : total_restaurants = 3)
  (h2 : second_restaurant_meals = 40)
  (h3 : third_restaurant_meals = 50)
  (h4 : total_weekly_meals = 770)
  (h5 : days_per_week = 7) :
  first_restaurant_meals total_restaurants second_restaurant_meals third_restaurant_meals total_weekly_meals days_per_week = 20 := by
  sorry

end NUMINAMATH_CALUDE_first_restaurant_meals_first_restaurant_meals_proof_l834_83457


namespace NUMINAMATH_CALUDE_composite_increasing_pos_l834_83453

/-- An odd function that is positive and increasing for negative x -/
def OddPositiveIncreasingNeg (f : ℝ → ℝ) : Prop :=
  (∀ x, f (-x) = -f x) ∧ 
  (∀ x < 0, 0 < f x) ∧
  (∀ x y, x < y ∧ y < 0 → f x < f y)

/-- The composite function f[f(x)] is increasing for positive x -/
theorem composite_increasing_pos 
  (f : ℝ → ℝ) 
  (h : OddPositiveIncreasingNeg f) : 
  ∀ x y, 0 < x ∧ x < y → f (f x) < f (f y) := by
sorry

end NUMINAMATH_CALUDE_composite_increasing_pos_l834_83453


namespace NUMINAMATH_CALUDE_fraction_calculation_l834_83410

theorem fraction_calculation (x y : ℚ) (hx : x = 2/3) (hy : y = 5/2) :
  (1/3) * x^7 * y^6 = 125/261 := by
  sorry

end NUMINAMATH_CALUDE_fraction_calculation_l834_83410
