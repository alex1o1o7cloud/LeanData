import Mathlib

namespace NUMINAMATH_CALUDE_gcd_n_cube_plus_nine_and_n_plus_two_l1334_133410

theorem gcd_n_cube_plus_nine_and_n_plus_two (n : ℕ) (h : n > 2^3) :
  Nat.gcd (n^3 + 3^2) (n + 2) = 1 := by
  sorry

end NUMINAMATH_CALUDE_gcd_n_cube_plus_nine_and_n_plus_two_l1334_133410


namespace NUMINAMATH_CALUDE_stating_acceleration_implies_speed_increase_l1334_133456

/-- Represents a train's acceleration scenario -/
structure TrainAcceleration where
  s : ℝ  -- distance traveled before acceleration (km)
  v : ℝ  -- acceleration rate (km/h)
  x : ℝ  -- initial speed (km/h)

/-- The equation holds for the given train acceleration scenario -/
def equation_holds (t : TrainAcceleration) : Prop :=
  t.s / t.x + t.v = (t.s + 50) / t.x

/-- The train's speed increases by v km/h after acceleration -/
def speed_increase (t : TrainAcceleration) : Prop :=
  ∃ (final_speed : ℝ), final_speed = t.x + t.v

/-- 
Theorem stating that if the equation holds, 
then the train's speed increases by v km/h after acceleration 
-/
theorem acceleration_implies_speed_increase 
  (t : TrainAcceleration) (h : equation_holds t) : speed_increase t :=
sorry

end NUMINAMATH_CALUDE_stating_acceleration_implies_speed_increase_l1334_133456


namespace NUMINAMATH_CALUDE_max_volume_inscribed_cone_l1334_133470

/-- Given a sphere with volume 36π, the maximum volume of an inscribed cone is 32π/3 -/
theorem max_volume_inscribed_cone (sphere_volume : ℝ) (h_volume : sphere_volume = 36 * Real.pi) :
  ∃ (max_cone_volume : ℝ),
    (∀ (cone_volume : ℝ), cone_volume ≤ max_cone_volume) ∧
    (max_cone_volume = (32 * Real.pi) / 3) :=
sorry

end NUMINAMATH_CALUDE_max_volume_inscribed_cone_l1334_133470


namespace NUMINAMATH_CALUDE_unique_positive_root_interval_l1334_133457

theorem unique_positive_root_interval :
  ∃! r : ℝ, r > 0 ∧ r^3 - r - 1 = 0 →
  ∃ r : ℝ, r ∈ Set.Ioo 1 2 ∧ r^3 - r - 1 = 0 := by
  sorry

end NUMINAMATH_CALUDE_unique_positive_root_interval_l1334_133457


namespace NUMINAMATH_CALUDE_find_a_value_l1334_133453

-- Define the function f
def f (x : ℝ) : ℝ := sorry

-- State the theorem
theorem find_a_value :
  (∀ x, f (x + 3) = 3 * f x) →
  (∀ x ∈ Set.Ioo 0 3, f x = Real.log x - a * x) →
  (a > 1/3) →
  (Set.Ioo (-6) (-3)).image f ⊆ Set.Iic (-1/9) →
  (∃ x ∈ Set.Ioo (-6) (-3), f x = -1/9) →
  a = 1 := by sorry

end NUMINAMATH_CALUDE_find_a_value_l1334_133453


namespace NUMINAMATH_CALUDE_min_value_f_max_value_g_l1334_133487

-- Define the functions
def f (m : ℝ) : ℝ := m^2 + 2*m + 3
def g (m : ℝ) : ℝ := -m^2 + 2*m + 3

-- Theorem for the minimum value of f
theorem min_value_f : ∀ m : ℝ, f m ≥ 2 ∧ ∃ m₀ : ℝ, f m₀ = 2 :=
sorry

-- Theorem for the maximum value of g
theorem max_value_g : ∀ m : ℝ, g m ≤ 4 ∧ ∃ m₀ : ℝ, g m₀ = 4 :=
sorry

end NUMINAMATH_CALUDE_min_value_f_max_value_g_l1334_133487


namespace NUMINAMATH_CALUDE_weight_of_nine_moles_972_l1334_133433

/-- The weight of a compound given its number of moles and molecular weight -/
def weight_of_compound (moles : ℝ) (molecular_weight : ℝ) : ℝ :=
  moles * molecular_weight

/-- Theorem: The weight of 9 moles of a compound with molecular weight 972 g/mol is 8748 grams -/
theorem weight_of_nine_moles_972 : 
  weight_of_compound 9 972 = 8748 := by
  sorry

end NUMINAMATH_CALUDE_weight_of_nine_moles_972_l1334_133433


namespace NUMINAMATH_CALUDE_pens_bought_l1334_133431

/-- Represents the cost of a single notebook in dollars -/
def notebook_cost : ℝ := sorry

/-- Represents the number of pens Maria bought -/
def num_pens : ℝ := sorry

/-- Theorem stating the relationship between the number of pens, total cost, and notebook cost -/
theorem pens_bought (notebook_cost num_pens : ℝ) : 
  (10 * notebook_cost + 2 * num_pens = 30) → 
  (num_pens = (30 - 10 * notebook_cost) / 2) := by
  sorry

end NUMINAMATH_CALUDE_pens_bought_l1334_133431


namespace NUMINAMATH_CALUDE_cube_volume_relation_l1334_133477

theorem cube_volume_relation (V : ℝ) : 
  (∃ (s : ℝ), V = s^3 ∧ 512 = (2*s)^3) → V = 64 := by
  sorry

end NUMINAMATH_CALUDE_cube_volume_relation_l1334_133477


namespace NUMINAMATH_CALUDE_domain_of_g_l1334_133473

-- Define the function f with domain [0,4]
def f : Set ℝ := Set.Icc 0 4

-- Define the function g
def g (f : Set ℝ) : Set ℝ := {x | x ∈ f ∧ x^2 ∈ f}

-- Theorem statement
theorem domain_of_g (f : Set ℝ) (hf : f = Set.Icc 0 4) : 
  g f = Set.Icc 0 2 := by sorry

end NUMINAMATH_CALUDE_domain_of_g_l1334_133473


namespace NUMINAMATH_CALUDE_fraction_to_decimal_l1334_133493

theorem fraction_to_decimal : (2 : ℚ) / 25 = 0.08 := by sorry

end NUMINAMATH_CALUDE_fraction_to_decimal_l1334_133493


namespace NUMINAMATH_CALUDE_class_test_problem_l1334_133465

theorem class_test_problem (p_first : ℝ) (p_second : ℝ) (p_neither : ℝ) 
  (h1 : p_first = 0.75)
  (h2 : p_second = 0.7)
  (h3 : p_neither = 0.2) :
  p_first + p_second - (1 - p_neither) = 0.65 := by
  sorry

end NUMINAMATH_CALUDE_class_test_problem_l1334_133465


namespace NUMINAMATH_CALUDE_right_triangle_ratio_l1334_133481

theorem right_triangle_ratio (a d : ℝ) : 
  (a - d) ^ 2 + a ^ 2 = (a + d) ^ 2 → 
  a = d * (2 + Real.sqrt 3) := by
sorry

end NUMINAMATH_CALUDE_right_triangle_ratio_l1334_133481


namespace NUMINAMATH_CALUDE_train_length_l1334_133458

/-- The length of a train given its speed and time to cross a point -/
theorem train_length (speed : Real) (time : Real) : 
  speed = 72 → time = 4.499640028797696 → 
  ∃ (length : Real), abs (length - 89.99280057595392) < 0.000001 := by
  sorry

end NUMINAMATH_CALUDE_train_length_l1334_133458


namespace NUMINAMATH_CALUDE_waiter_customers_l1334_133488

theorem waiter_customers (num_tables : ℕ) (women_per_table : ℕ) (men_per_table : ℕ) 
  (h1 : num_tables = 8)
  (h2 : women_per_table = 7)
  (h3 : men_per_table = 4) :
  num_tables * (women_per_table + men_per_table) = 88 := by
sorry

end NUMINAMATH_CALUDE_waiter_customers_l1334_133488


namespace NUMINAMATH_CALUDE_lcm_9_12_15_l1334_133435

theorem lcm_9_12_15 : Nat.lcm 9 (Nat.lcm 12 15) = 180 := by
  sorry

end NUMINAMATH_CALUDE_lcm_9_12_15_l1334_133435


namespace NUMINAMATH_CALUDE_cistern_length_l1334_133474

/-- Given a cistern with specified dimensions, prove its length is 12 meters. -/
theorem cistern_length (width : ℝ) (depth : ℝ) (total_area : ℝ) :
  width = 14 →
  depth = 1.25 →
  total_area = 233 →
  width * depth * 2 + width * (total_area / width / depth - width) + depth * (total_area / width / depth - width) * 2 = total_area →
  total_area / width / depth - width = 12 :=
by sorry

end NUMINAMATH_CALUDE_cistern_length_l1334_133474


namespace NUMINAMATH_CALUDE_order_total_price_l1334_133478

/-- Calculate the total price of an order given the number of ice-cream bars, number of sundaes,
    price per ice-cream bar, and price per sundae. -/
def total_price (ice_cream_bars : ℕ) (sundaes : ℕ) (price_ice_cream : ℚ) (price_sundae : ℚ) : ℚ :=
  ice_cream_bars * price_ice_cream + sundaes * price_sundae

/-- Theorem stating that the total price of the order is $200 given the specific quantities and prices. -/
theorem order_total_price :
  total_price 225 125 (60/100) (52/100) = 200 := by
  sorry

end NUMINAMATH_CALUDE_order_total_price_l1334_133478


namespace NUMINAMATH_CALUDE_exprC_is_factorization_left_to_right_l1334_133443

/-- Represents a polynomial expression -/
structure PolynomialExpression where
  left : ℝ → ℝ → ℝ
  right : ℝ → ℝ → ℝ

/-- Checks if an expression is in product form -/
def isProductForm (expr : ℝ → ℝ → ℝ) : Prop :=
  ∃ (f g : ℝ → ℝ → ℝ), ∀ x y, expr x y = f x y * g x y

/-- Defines factorization from left to right -/
def isFactorizationLeftToRight (expr : PolynomialExpression) : Prop :=
  ¬(isProductForm expr.left) ∧ (isProductForm expr.right)

/-- The specific expression we're examining -/
def exprC : PolynomialExpression :=
  { left := λ a b => a^2 - 4*a*b + 4*b^2,
    right := λ a b => (a - 2*b)^2 }

/-- Theorem stating that exprC represents factorization from left to right -/
theorem exprC_is_factorization_left_to_right :
  isFactorizationLeftToRight exprC :=
sorry

end NUMINAMATH_CALUDE_exprC_is_factorization_left_to_right_l1334_133443


namespace NUMINAMATH_CALUDE_expression_equals_two_l1334_133463

theorem expression_equals_two (a b c : ℝ) 
  (h_pos_a : a > 0) (h_pos_b : b > 0) (h_pos_c : c > 0) 
  (h_abc : a * b * c = 1) : 
  (1 + a) / (1 + a + a * b) + 
  (1 + b) / (1 + b + b * c) + 
  (1 + c) / (1 + c + c * a) = 2 := by
sorry

end NUMINAMATH_CALUDE_expression_equals_two_l1334_133463


namespace NUMINAMATH_CALUDE_circle_number_placement_l1334_133461

theorem circle_number_placement :
  ∃ (a₁ b₁ c₁ d₁ e₁ a₂ b₂ c₂ d₂ e₂ : ℕ),
    (1 ≤ a₁ ∧ a₁ ≤ 9) ∧ (1 ≤ b₁ ∧ b₁ ≤ 9) ∧ (1 ≤ c₁ ∧ c₁ ≤ 9) ∧ (1 ≤ d₁ ∧ d₁ ≤ 9) ∧ (1 ≤ e₁ ∧ e₁ ≤ 9) ∧
    (1 ≤ a₂ ∧ a₂ ≤ 9) ∧ (1 ≤ b₂ ∧ b₂ ≤ 9) ∧ (1 ≤ c₂ ∧ c₂ ≤ 9) ∧ (1 ≤ d₂ ∧ d₂ ≤ 9) ∧ (1 ≤ e₂ ∧ e₂ ≤ 9) ∧
    b₁ - d₁ = 2 ∧ d₁ - a₁ = 3 ∧ a₁ - c₁ = 1 ∧
    b₂ - d₂ = 2 ∧ d₂ - a₂ = 3 ∧ a₂ - c₂ = 1 ∧
    a₁ ≠ b₁ ∧ a₁ ≠ c₁ ∧ a₁ ≠ d₁ ∧ a₁ ≠ e₁ ∧
    b₁ ≠ c₁ ∧ b₁ ≠ d₁ ∧ b₁ ≠ e₁ ∧
    c₁ ≠ d₁ ∧ c₁ ≠ e₁ ∧
    d₁ ≠ e₁ ∧
    a₂ ≠ b₂ ∧ a₂ ≠ c₂ ∧ a₂ ≠ d₂ ∧ a₂ ≠ e₂ ∧
    b₂ ≠ c₂ ∧ b₂ ≠ d₂ ∧ b₂ ≠ e₂ ∧
    c₂ ≠ d₂ ∧ c₂ ≠ e₂ ∧
    d₂ ≠ e₂ ∧
    (a₁ ≠ a₂ ∨ b₁ ≠ b₂ ∨ c₁ ≠ c₂ ∨ d₁ ≠ d₂ ∨ e₁ ≠ e₂) :=
by sorry

end NUMINAMATH_CALUDE_circle_number_placement_l1334_133461


namespace NUMINAMATH_CALUDE_aqua_park_earnings_l1334_133447

def admission_cost : ℚ := 12
def tour_cost : ℚ := 6
def meal_cost : ℚ := 10
def souvenir_cost : ℚ := 8

def group1_size : ℕ := 10
def group2_size : ℕ := 15
def group3_size : ℕ := 8

def group1_discount_rate : ℚ := 0.10
def group2_meal_discount_rate : ℚ := 0.05

def group1_total (admission_cost tour_cost meal_cost souvenir_cost : ℚ) (group_size : ℕ) (discount_rate : ℚ) : ℚ :=
  (1 - discount_rate) * (admission_cost + tour_cost + meal_cost + souvenir_cost) * group_size

def group2_total (admission_cost meal_cost : ℚ) (group_size : ℕ) (meal_discount_rate : ℚ) : ℚ :=
  admission_cost * group_size + (1 - meal_discount_rate) * meal_cost * group_size

def group3_total (admission_cost tour_cost souvenir_cost : ℚ) (group_size : ℕ) : ℚ :=
  (admission_cost + tour_cost + souvenir_cost) * group_size

theorem aqua_park_earnings : 
  group1_total admission_cost tour_cost meal_cost souvenir_cost group1_size group1_discount_rate +
  group2_total admission_cost meal_cost group2_size group2_meal_discount_rate +
  group3_total admission_cost tour_cost souvenir_cost group3_size = 854.5 := by
  sorry

end NUMINAMATH_CALUDE_aqua_park_earnings_l1334_133447


namespace NUMINAMATH_CALUDE_union_of_sets_l1334_133475

theorem union_of_sets : 
  let A : Set ℤ := {0, 1}
  let B : Set ℤ := {0, -1}
  A ∪ B = {-1, 0, 1} := by
sorry

end NUMINAMATH_CALUDE_union_of_sets_l1334_133475


namespace NUMINAMATH_CALUDE_drawer_pull_cost_l1334_133460

/-- Given the conditions of Amanda's kitchen upgrade, prove the cost of each drawer pull. -/
theorem drawer_pull_cost (num_knobs : ℕ) (cost_per_knob : ℚ) (num_pulls : ℕ) (total_cost : ℚ) :
  num_knobs = 18 →
  cost_per_knob = 5/2 →
  num_pulls = 8 →
  total_cost = 77 →
  (total_cost - num_knobs * cost_per_knob) / num_pulls = 4 := by
  sorry

end NUMINAMATH_CALUDE_drawer_pull_cost_l1334_133460


namespace NUMINAMATH_CALUDE_book_lending_solution_l1334_133439

/-- Represents the book lending problem with three people. -/
structure BookLending where
  xiaoqiang : ℕ  -- Initial number of books Xiaoqiang has
  feifei : ℕ     -- Initial number of books Feifei has
  xiaojing : ℕ   -- Initial number of books Xiaojing has

/-- The book lending problem satisfies the given conditions. -/
def satisfiesConditions (b : BookLending) : Prop :=
  b.xiaoqiang - 20 + 10 = 35 ∧
  b.feifei + 20 - 15 = 35 ∧
  b.xiaojing + 15 - 10 = 35

/-- The theorem stating the solution to the book lending problem. -/
theorem book_lending_solution :
  ∃ (b : BookLending), satisfiesConditions b ∧ b.xiaoqiang = 45 ∧ b.feifei = 30 ∧ b.xiaojing = 30 := by
  sorry

end NUMINAMATH_CALUDE_book_lending_solution_l1334_133439


namespace NUMINAMATH_CALUDE_unique_solution_system_l1334_133459

theorem unique_solution_system (x y z : ℝ) :
  x > 0 ∧ y > 0 ∧ z > 0 →
  x^2 + y^2 + x*y = 7 →
  x^2 + z^2 + x*z = 13 →
  y^2 + z^2 + y*z = 19 →
  x = 1 ∧ y = 2 ∧ z = 3 := by sorry

end NUMINAMATH_CALUDE_unique_solution_system_l1334_133459


namespace NUMINAMATH_CALUDE_one_student_owns_all_pets_l1334_133445

/-- Represents the pet ownership distribution in Sara's class -/
structure PetOwnership where
  total : ℕ
  dog_owners : ℕ
  cat_owners : ℕ
  bird_owners : ℕ
  no_pets : ℕ
  just_dogs : ℕ
  just_cats : ℕ
  just_birds : ℕ
  dogs_and_cats : ℕ
  dogs_and_birds : ℕ
  cats_and_birds : ℕ
  all_three : ℕ

/-- The theorem stating that exactly one student owns all three types of pets -/
theorem one_student_owns_all_pets (p : PetOwnership) : 
  p.total = 48 ∧ 
  p.dog_owners = p.total / 2 ∧ 
  p.cat_owners = p.total * 5 / 16 ∧ 
  p.bird_owners = 8 ∧ 
  p.no_pets = 7 ∧
  p.just_dogs = 12 ∧
  p.just_cats = 2 ∧
  p.just_birds = 4 ∧
  p.dog_owners = p.just_dogs + p.dogs_and_cats + p.dogs_and_birds + p.all_three ∧
  p.cat_owners = p.just_cats + p.dogs_and_cats + p.cats_and_birds + p.all_three ∧
  p.bird_owners = p.just_birds + p.dogs_and_birds + p.cats_and_birds + p.all_three ∧
  p.total = p.just_dogs + p.just_cats + p.just_birds + p.dogs_and_cats + p.dogs_and_birds + p.cats_and_birds + p.all_three + p.no_pets
  →
  p.all_three = 1 := by
  sorry

end NUMINAMATH_CALUDE_one_student_owns_all_pets_l1334_133445


namespace NUMINAMATH_CALUDE_intersection_of_A_and_B_l1334_133472

def A : Set ℕ := {1, 2, 3}
def B : Set ℕ := {2, 4}

theorem intersection_of_A_and_B : A ∩ B = {2} := by
  sorry

end NUMINAMATH_CALUDE_intersection_of_A_and_B_l1334_133472


namespace NUMINAMATH_CALUDE_average_score_calculation_l1334_133449

theorem average_score_calculation (total : ℝ) (male_ratio : ℝ) (male_avg : ℝ) (female_avg : ℝ)
  (h1 : male_ratio = 0.4)
  (h2 : male_avg = 75)
  (h3 : female_avg = 80) :
  (male_ratio * male_avg + (1 - male_ratio) * female_avg) = 78 := by
  sorry

end NUMINAMATH_CALUDE_average_score_calculation_l1334_133449


namespace NUMINAMATH_CALUDE_other_bases_with_square_property_existence_of_other_bases_l1334_133407

theorem other_bases_with_square_property (B : ℕ) (V : ℕ) : Prop :=
  2 < B ∧ 1 < V ∧ V < B ∧ V * V % B = V % B

theorem existence_of_other_bases :
  ∃ B V, B ≠ 50 ∧ other_bases_with_square_property B V := by
  sorry

end NUMINAMATH_CALUDE_other_bases_with_square_property_existence_of_other_bases_l1334_133407


namespace NUMINAMATH_CALUDE_newlyGrownUneatenCorrect_l1334_133482

/-- Represents the number of potatoes in Mary's garden -/
structure PotatoGarden where
  initial : ℕ
  current : ℕ

/-- Calculates the number of newly grown potatoes left uneaten -/
def newlyGrownUneaten (garden : PotatoGarden) : ℕ :=
  garden.current - garden.initial

theorem newlyGrownUneatenCorrect (garden : PotatoGarden) 
  (h1 : garden.initial = 8) 
  (h2 : garden.current = 11) : 
  newlyGrownUneaten garden = 3 := by
  sorry

end NUMINAMATH_CALUDE_newlyGrownUneatenCorrect_l1334_133482


namespace NUMINAMATH_CALUDE_cucumber_weight_after_evaporation_l1334_133437

/-- Given 100 pounds of cucumbers initially composed of 99% water by weight,
    when the water composition changes to 98% by weight due to evaporation,
    the new total weight of the cucumbers is 50 pounds. -/
theorem cucumber_weight_after_evaporation
  (initial_weight : ℝ)
  (initial_water_percentage : ℝ)
  (final_water_percentage : ℝ)
  (h1 : initial_weight = 100)
  (h2 : initial_water_percentage = 0.99)
  (h3 : final_water_percentage = 0.98)
  : ∃ (final_weight : ℝ), final_weight = 50 ∧
    (1 - initial_water_percentage) * initial_weight =
    (1 - final_water_percentage) * final_weight :=
by sorry

end NUMINAMATH_CALUDE_cucumber_weight_after_evaporation_l1334_133437


namespace NUMINAMATH_CALUDE_min_value_of_f_l1334_133430

noncomputable def f (x : ℝ) : ℝ := x * Real.exp x

theorem min_value_of_f :
  ∃ (min : ℝ), min = -1 / Real.exp 1 ∧ ∀ (x : ℝ), f x ≥ min :=
sorry

end NUMINAMATH_CALUDE_min_value_of_f_l1334_133430


namespace NUMINAMATH_CALUDE_sum_of_squares_of_roots_l1334_133462

theorem sum_of_squares_of_roots (a b c : ℚ) (h : a = 6 ∧ b = 9 ∧ c = -21) :
  let x₁ := (-b + Real.sqrt (b^2 - 4*a*c)) / (2*a)
  let x₂ := (-b - Real.sqrt (b^2 - 4*a*c)) / (2*a)
  x₁^2 + x₂^2 = 37/4 := by sorry

end NUMINAMATH_CALUDE_sum_of_squares_of_roots_l1334_133462


namespace NUMINAMATH_CALUDE_p_greater_than_q_greater_than_r_l1334_133420

def P : ℚ := -1 / (201603 * 201604)
def Q : ℚ := -1 / (201602 * 201604)
def R : ℚ := -1 / (201602 * 201603)

theorem p_greater_than_q_greater_than_r : P > Q ∧ Q > R := by sorry

end NUMINAMATH_CALUDE_p_greater_than_q_greater_than_r_l1334_133420


namespace NUMINAMATH_CALUDE_lcm_problem_l1334_133406

theorem lcm_problem (m : ℕ+) 
  (h1 : Nat.lcm 40 m = 120) 
  (h2 : Nat.lcm m 45 = 180) : 
  m = 60 := by
  sorry

end NUMINAMATH_CALUDE_lcm_problem_l1334_133406


namespace NUMINAMATH_CALUDE_abc_inequality_l1334_133428

theorem abc_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (h_sum : a + b + c = a^(1/7) + b^(1/7) + c^(1/7)) :
  a^a * b^b * c^c ≥ 1 := by
  sorry

end NUMINAMATH_CALUDE_abc_inequality_l1334_133428


namespace NUMINAMATH_CALUDE_solve_proportion_l1334_133499

theorem solve_proportion (x y : ℚ) (h1 : x / y = 8 / 3) (h2 : y = 27) : x = 72 := by
  sorry

end NUMINAMATH_CALUDE_solve_proportion_l1334_133499


namespace NUMINAMATH_CALUDE_zoo_trip_buses_l1334_133408

/-- Given a school trip to the zoo with the following conditions:
  * There are 396 total students
  * 4 students traveled in cars
  * Each bus can hold 56 students
  * All buses were filled
  Prove that the number of buses required is 7. -/
theorem zoo_trip_buses (total_students : ℕ) (car_students : ℕ) (students_per_bus : ℕ) :
  total_students = 396 →
  car_students = 4 →
  students_per_bus = 56 →
  (total_students - car_students) % students_per_bus = 0 →
  (total_students - car_students) / students_per_bus = 7 :=
by sorry

end NUMINAMATH_CALUDE_zoo_trip_buses_l1334_133408


namespace NUMINAMATH_CALUDE_vector_param_validity_l1334_133401

/-- A vector parameterization of a line -/
structure VectorParam where
  x0 : ℝ
  y0 : ℝ
  dx : ℝ
  dy : ℝ

/-- The line equation y = -3/2x - 5 -/
def line_equation (x y : ℝ) : Prop := y = -3/2 * x - 5

/-- Predicate for a valid vector parameterization -/
def is_valid_param (p : VectorParam) : Prop :=
  line_equation p.x0 p.y0 ∧ p.dy = -3/2 * p.dx

theorem vector_param_validity (p : VectorParam) :
  is_valid_param p ↔ ∀ t : ℝ, line_equation (p.x0 + t * p.dx) (p.y0 + t * p.dy) :=
sorry

end NUMINAMATH_CALUDE_vector_param_validity_l1334_133401


namespace NUMINAMATH_CALUDE_product_profit_properties_l1334_133466

/-- A product with given cost and sales characteristics -/
structure Product where
  cost_price : ℝ
  initial_price : ℝ
  initial_sales : ℝ
  sales_increase : ℝ

/-- Daily profit as a function of price decrease -/
def daily_profit (p : Product) (x : ℝ) : ℝ :=
  (p.initial_price - x - p.cost_price) * (p.initial_sales + p.sales_increase * x)

/-- Theorem stating the properties of the product and its profit function -/
theorem product_profit_properties (p : Product) 
  (h_cost : p.cost_price = 3.5)
  (h_initial_price : p.initial_price = 14.5)
  (h_initial_sales : p.initial_sales = 500)
  (h_sales_increase : p.sales_increase = 100) :
  (∀ x, 0 ≤ x ∧ x ≤ 11 → daily_profit p x = -100 * (x - 3)^2 + 6400) ∧
  (∃ max_profit, max_profit = 6400 ∧ 
    ∀ x, 0 ≤ x ∧ x ≤ 11 → daily_profit p x ≤ max_profit) ∧
  (∃ optimal_price, optimal_price = 11.5 ∧
    ∀ x, 0 ≤ x ∧ x ≤ 11 → 
      daily_profit p ((p.initial_price - optimal_price) : ℝ) ≥ daily_profit p x) :=
sorry

end NUMINAMATH_CALUDE_product_profit_properties_l1334_133466


namespace NUMINAMATH_CALUDE_bob_has_77_pennies_l1334_133464

/-- The number of pennies Alex currently has -/
def alex_pennies : ℕ := sorry

/-- The number of pennies Bob currently has -/
def bob_pennies : ℕ := sorry

/-- If Alex gives Bob three pennies, Bob will have four times as many pennies as Alex has -/
axiom condition1 : bob_pennies + 3 = 4 * (alex_pennies - 3)

/-- If Bob gives Alex two pennies, Bob will have three times as many pennies as Alex has -/
axiom condition2 : bob_pennies - 2 = 3 * (alex_pennies + 2)

/-- Bob currently has 77 pennies -/
theorem bob_has_77_pennies : bob_pennies = 77 := by sorry

end NUMINAMATH_CALUDE_bob_has_77_pennies_l1334_133464


namespace NUMINAMATH_CALUDE_correct_day_is_thursday_l1334_133427

-- Define the days of the week
inductive DayOfWeek
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

-- Define the statements made by each person
def statement_A (today : DayOfWeek) : Prop := today = DayOfWeek.Friday
def statement_B (today : DayOfWeek) : Prop := today = DayOfWeek.Wednesday
def statement_C (today : DayOfWeek) : Prop := ¬(statement_A today ∨ statement_B today)
def statement_D (today : DayOfWeek) : Prop := today ≠ DayOfWeek.Thursday

-- Define the condition that only one statement is correct
def only_one_correct (today : DayOfWeek) : Prop :=
  (statement_A today ∧ ¬statement_B today ∧ ¬statement_C today ∧ ¬statement_D today) ∨
  (¬statement_A today ∧ statement_B today ∧ ¬statement_C today ∧ ¬statement_D today) ∨
  (¬statement_A today ∧ ¬statement_B today ∧ statement_C today ∧ ¬statement_D today) ∨
  (¬statement_A today ∧ ¬statement_B today ∧ ¬statement_C today ∧ statement_D today)

-- Theorem stating that Thursday is the only day satisfying all conditions
theorem correct_day_is_thursday :
  ∃! today : DayOfWeek, only_one_correct today ∧ today = DayOfWeek.Thursday :=
sorry

end NUMINAMATH_CALUDE_correct_day_is_thursday_l1334_133427


namespace NUMINAMATH_CALUDE_exists_left_absorbing_l1334_133484

variable {S : Type}
variable (star : S → S → S)

axiom commutative : ∀ a b : S, star a b = star b a
axiom associative : ∀ a b c : S, star (star a b) c = star a (star b c)
axiom exists_idempotent : ∃ a : S, star a a = a

theorem exists_left_absorbing : ∃ a : S, ∀ b : S, star a b = a := by
  sorry

end NUMINAMATH_CALUDE_exists_left_absorbing_l1334_133484


namespace NUMINAMATH_CALUDE_not_perfect_square_exists_l1334_133491

theorem not_perfect_square_exists (a b : ℕ) (ha : a > 1) (hb : b > 1) (hab : a ≠ b) :
  ∃ n : ℕ+, ¬ ∃ k : ℕ, (a^n.val - 1) * (b^n.val - 1) = k^2 := by
  sorry

end NUMINAMATH_CALUDE_not_perfect_square_exists_l1334_133491


namespace NUMINAMATH_CALUDE_correct_calculation_l1334_133438

theorem correct_calculation (x y : ℝ) : 3 * x^2 * y - 2 * x^2 * y = x^2 * y := by
  sorry

end NUMINAMATH_CALUDE_correct_calculation_l1334_133438


namespace NUMINAMATH_CALUDE_inequality_system_solution_l1334_133446

theorem inequality_system_solution : 
  {x : ℝ | x + 1 > 0 ∧ -2 * x ≤ 6} = {x : ℝ | x > -1} := by
  sorry

end NUMINAMATH_CALUDE_inequality_system_solution_l1334_133446


namespace NUMINAMATH_CALUDE_total_workers_l1334_133483

theorem total_workers (monkeys termites : ℕ) 
  (h1 : monkeys = 239) 
  (h2 : termites = 622) : 
  monkeys + termites = 861 := by
  sorry

end NUMINAMATH_CALUDE_total_workers_l1334_133483


namespace NUMINAMATH_CALUDE_infinite_geometric_series_l1334_133409

/-- Given an infinite geometric series with first term a and sum S,
    prove the common ratio r and the second term -/
theorem infinite_geometric_series
  (a : ℝ) (S : ℝ) (h_a : a = 540) (h_S : S = 4500) :
  ∃ (r : ℝ),
    r = 0.88 ∧
    S = a / (1 - r) ∧
    abs r < 1 ∧
    a * r = 475.2 := by
  sorry

end NUMINAMATH_CALUDE_infinite_geometric_series_l1334_133409


namespace NUMINAMATH_CALUDE_books_count_l1334_133469

def total_books (beatrix alannah queen kingston : ℕ) : ℕ :=
  beatrix + alannah + queen + kingston

theorem books_count :
  ∀ (beatrix alannah queen kingston : ℕ),
    beatrix = 30 →
    alannah = beatrix + 20 →
    queen = alannah + alannah / 5 →
    kingston = 2 * (beatrix + queen) →
    total_books beatrix alannah queen kingston = 320 := by
  sorry

end NUMINAMATH_CALUDE_books_count_l1334_133469


namespace NUMINAMATH_CALUDE_fence_poles_needed_l1334_133489

theorem fence_poles_needed (length width pole_distance : ℕ) : 
  length = 90 → width = 40 → pole_distance = 5 →
  (2 * (length + width)) / pole_distance = 52 := by
sorry

end NUMINAMATH_CALUDE_fence_poles_needed_l1334_133489


namespace NUMINAMATH_CALUDE_sector_central_angle_l1334_133476

/-- Proves that a circular sector with arc length 4 and area 2 has a central angle of 4 radians -/
theorem sector_central_angle (l : ℝ) (A : ℝ) (θ : ℝ) (r : ℝ) :
  l = 4 →
  A = 2 →
  l = r * θ →
  A = 1/2 * r^2 * θ →
  θ = 4 := by
sorry


end NUMINAMATH_CALUDE_sector_central_angle_l1334_133476


namespace NUMINAMATH_CALUDE_cubic_function_parallel_tangents_l1334_133412

/-- Given a cubic function f(x) = x³ + ax + b where a ≠ b, and the tangent lines
    to the graph of f at x=a and x=b are parallel, prove that f(1) = 1. -/
theorem cubic_function_parallel_tangents (a b : ℝ) (h : a ≠ b) : 
  let f : ℝ → ℝ := λ x ↦ x^3 + a*x + b
  (∃ k : ℝ, (3*a^2 + a = k) ∧ (3*b^2 + a = k)) → f 1 = 1 := by
  sorry


end NUMINAMATH_CALUDE_cubic_function_parallel_tangents_l1334_133412


namespace NUMINAMATH_CALUDE_proposition_false_iff_a_in_range_l1334_133496

theorem proposition_false_iff_a_in_range (a : ℝ) :
  (¬ ∃ x : ℝ, |x - a| + |x + 1| ≤ 2) ↔ a ∈ Set.Iio (-3) ∪ Set.Ioi 1 := by
  sorry

end NUMINAMATH_CALUDE_proposition_false_iff_a_in_range_l1334_133496


namespace NUMINAMATH_CALUDE_birthday_75_days_later_l1334_133455

theorem birthday_75_days_later (birthday : ℕ) : 
  (birthday % 7 = 0) → ((birthday + 75) % 7 = 5) := by
  sorry

#check birthday_75_days_later

end NUMINAMATH_CALUDE_birthday_75_days_later_l1334_133455


namespace NUMINAMATH_CALUDE_samoa_price_is_4_l1334_133402

/-- The price of a box of samoas -/
def samoa_price : ℝ := sorry

/-- The number of boxes of samoas sold -/
def samoa_boxes : ℕ := 3

/-- The price of a box of thin mints -/
def thin_mint_price : ℝ := 3.5

/-- The number of boxes of thin mints sold -/
def thin_mint_boxes : ℕ := 2

/-- The price of a box of fudge delights -/
def fudge_delight_price : ℝ := 5

/-- The number of boxes of fudge delights sold -/
def fudge_delight_boxes : ℕ := 1

/-- The price of a box of sugar cookies -/
def sugar_cookie_price : ℝ := 2

/-- The number of boxes of sugar cookies sold -/
def sugar_cookie_boxes : ℕ := 9

/-- The total sales amount -/
def total_sales : ℝ := 42

theorem samoa_price_is_4 : 
  samoa_price = 4 :=
by sorry

end NUMINAMATH_CALUDE_samoa_price_is_4_l1334_133402


namespace NUMINAMATH_CALUDE_div_negative_powers_l1334_133404

theorem div_negative_powers (a : ℝ) (h : a ≠ 0) : -28 * a^3 / (7 * a) = -4 * a^2 := by
  sorry

end NUMINAMATH_CALUDE_div_negative_powers_l1334_133404


namespace NUMINAMATH_CALUDE_fixed_cost_calculation_publishing_company_fixed_cost_l1334_133417

theorem fixed_cost_calculation (marketing_cost : ℕ) (selling_price : ℕ) (break_even_quantity : ℕ) : ℕ :=
  let net_revenue_per_book := selling_price - marketing_cost
  let fixed_cost := net_revenue_per_book * break_even_quantity
  fixed_cost

theorem publishing_company_fixed_cost :
  fixed_cost_calculation 4 9 10000 = 50000 := by
  sorry

end NUMINAMATH_CALUDE_fixed_cost_calculation_publishing_company_fixed_cost_l1334_133417


namespace NUMINAMATH_CALUDE_common_tangent_implies_m_eq_four_l1334_133467

/-- The function f(x) -/
noncomputable def f (m : ℝ) (x : ℝ) : ℝ := Real.log x + (m * x) / (x + 1)

/-- The function g(x) -/
def g (x : ℝ) : ℝ := x^2 + 1

/-- The derivative of f(x) -/
noncomputable def f_deriv (m : ℝ) (x : ℝ) : ℝ := 1/x + m / ((x + 1)^2)

/-- The derivative of g(x) -/
def g_deriv (x : ℝ) : ℝ := 2 * x

theorem common_tangent_implies_m_eq_four :
  ∀ m : ℝ, ∃ a x₁ x₂ : ℝ,
    a > 0 ∧
    f_deriv m x₁ = a ∧
    g_deriv x₂ = a ∧
    f m x₁ = a * x₁ ∧
    g x₂ = a * x₂ →
    m = 4 := by
  sorry

end NUMINAMATH_CALUDE_common_tangent_implies_m_eq_four_l1334_133467


namespace NUMINAMATH_CALUDE_coins_problem_l1334_133413

theorem coins_problem (a b c d : ℕ) : 
  a = 21 →                  -- A has 21 coins
  a = b + 9 →               -- A has 9 more coins than B
  c = b + 17 →              -- C has 17 more coins than B
  a + b = c + d - 5 →       -- Sum of A and B is 5 less than sum of C and D
  d = 9 :=                  -- D has 9 coins
by sorry

end NUMINAMATH_CALUDE_coins_problem_l1334_133413


namespace NUMINAMATH_CALUDE_function_bound_l1334_133479

open Real

theorem function_bound (f : ℝ → ℝ) (m : ℝ) : 
  (∀ x ∈ Set.Ioo 0 (π/4), f x = sin (2*x) - Real.sqrt 3 * cos (2*x)) →
  (∀ x ∈ Set.Ioo 0 (π/4), |f x| < m) →
  m ≥ Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_function_bound_l1334_133479


namespace NUMINAMATH_CALUDE_odd_factorials_equal_sum_factorial_l1334_133400

def product_of_odd_factorials (m : ℕ) : ℕ :=
  (List.range m).foldl (λ acc i => acc * Nat.factorial (2 * i + 1)) 1

def sum_of_first_n (m : ℕ) : ℕ :=
  m * (m + 1) / 2

theorem odd_factorials_equal_sum_factorial (m : ℕ) :
  (product_of_odd_factorials m = Nat.factorial (sum_of_first_n m)) ↔ (m = 1 ∨ m = 2 ∨ m = 3 ∨ m = 4) := by
  sorry

end NUMINAMATH_CALUDE_odd_factorials_equal_sum_factorial_l1334_133400


namespace NUMINAMATH_CALUDE_geometric_progression_first_term_l1334_133498

/-- A geometric progression with second term 5 and third term 1 has first term 25. -/
theorem geometric_progression_first_term (a : ℝ) (q : ℝ) : 
  a * q = 5 ∧ a * q^2 = 1 → a = 25 := by
  sorry

end NUMINAMATH_CALUDE_geometric_progression_first_term_l1334_133498


namespace NUMINAMATH_CALUDE_election_winner_votes_l1334_133423

theorem election_winner_votes 
  (total_votes : ℕ)
  (winner_percentage : ℚ)
  (vote_difference : ℕ)
  (h1 : winner_percentage = 62 / 100)
  (h2 : vote_difference = 312)
  (h3 : ↑total_votes * winner_percentage - ↑total_votes * (1 - winner_percentage) = vote_difference) :
  ↑total_votes * winner_percentage = 806 :=
by sorry

end NUMINAMATH_CALUDE_election_winner_votes_l1334_133423


namespace NUMINAMATH_CALUDE_profit_rate_equal_with_without_discount_l1334_133497

-- Define the discount rate
def discount_rate : ℝ := 0.05

-- Define the profit rate with discount
def profit_rate_with_discount : ℝ := 0.235

-- Theorem statement
theorem profit_rate_equal_with_without_discount :
  profit_rate_with_discount = (1 + profit_rate_with_discount) / (1 - discount_rate) - 1 :=
by sorry

end NUMINAMATH_CALUDE_profit_rate_equal_with_without_discount_l1334_133497


namespace NUMINAMATH_CALUDE_james_semesters_l1334_133471

/-- Proves the number of semesters James is paying for, given the conditions -/
theorem james_semesters (units_per_semester : ℕ) (cost_per_unit : ℕ) (total_cost : ℕ) :
  units_per_semester = 20 →
  cost_per_unit = 50 →
  total_cost = 2000 →
  total_cost / (units_per_semester * cost_per_unit) = 2 := by
  sorry

#check james_semesters

end NUMINAMATH_CALUDE_james_semesters_l1334_133471


namespace NUMINAMATH_CALUDE_square_and_sqrt_properties_l1334_133415

theorem square_and_sqrt_properties : 
  let a : ℕ := 10001
  let b : ℕ := 100010001
  let c : ℕ := 1000200030004000300020001
  (a^2 = 100020001) ∧ 
  (b^2 = 10002000300020001) ∧ 
  (c.sqrt = 1000100010001) := by
  sorry

end NUMINAMATH_CALUDE_square_and_sqrt_properties_l1334_133415


namespace NUMINAMATH_CALUDE_abc_inequality_l1334_133448

theorem abc_inequality (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  Real.sqrt (a * b * c) * (Real.sqrt a + Real.sqrt b + Real.sqrt c) + (a + b + c)^2 ≥ 
  4 * Real.sqrt (3 * a * b * c * (a + b + c)) :=
by sorry

end NUMINAMATH_CALUDE_abc_inequality_l1334_133448


namespace NUMINAMATH_CALUDE_exponential_inequality_l1334_133432

theorem exponential_inequality (m n : ℝ) (h1 : m > n) (h2 : n > 0) : (0.3 : ℝ) ^ m < (0.3 : ℝ) ^ n := by
  sorry

end NUMINAMATH_CALUDE_exponential_inequality_l1334_133432


namespace NUMINAMATH_CALUDE_total_feed_amount_l1334_133485

/-- Represents the total amount of dog feed mixed -/
def total_feed (cheap_feed expensive_feed : ℝ) : ℝ := cheap_feed + expensive_feed

/-- Represents the total cost of the mixed feed -/
def total_cost (cheap_feed expensive_feed : ℝ) : ℝ :=
  0.18 * cheap_feed + 0.53 * expensive_feed

/-- The theorem stating the total amount of feed mixed -/
theorem total_feed_amount :
  ∃ (expensive_feed : ℝ),
    total_feed 17 expensive_feed = 35 ∧
    total_cost 17 expensive_feed = 0.36 * total_feed 17 expensive_feed :=
sorry

end NUMINAMATH_CALUDE_total_feed_amount_l1334_133485


namespace NUMINAMATH_CALUDE_papaya_height_after_five_years_l1334_133441

/-- The height of a papaya tree after n years -/
def papayaHeight (n : ℕ) : ℝ :=
  match n with
  | 0 => 0
  | 1 => 2
  | 2 => papayaHeight 1 + 1.5 * papayaHeight 1
  | 3 => papayaHeight 2 + 1.5 * papayaHeight 2
  | 4 => papayaHeight 3 + 2 * papayaHeight 3
  | 5 => papayaHeight 4 + 0.5 * papayaHeight 4
  | _ => 0  -- undefined for years beyond 5

theorem papaya_height_after_five_years :
  papayaHeight 5 = 23 := by
  sorry

end NUMINAMATH_CALUDE_papaya_height_after_five_years_l1334_133441


namespace NUMINAMATH_CALUDE_inequality_proof_root_mean_square_arithmetic_mean_l1334_133405

theorem inequality_proof (x y : ℝ) (hx : x > 0) (hy : y > 0) (hsum : x + y > 2) :
  (1 + x) / y < 2 ∨ (1 + y) / x < 2 := by
  sorry

theorem root_mean_square_arithmetic_mean (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0) :
  Real.sqrt ((a^2 + b^2 + c^2) / 3) ≥ (a + b + c) / 3 := by
  sorry

end NUMINAMATH_CALUDE_inequality_proof_root_mean_square_arithmetic_mean_l1334_133405


namespace NUMINAMATH_CALUDE_circle_and_line_intersection_l1334_133450

-- Define the circle C
def circle_C : Set (ℝ × ℝ) :=
  {p | (p.1 - 1)^2 + p.2^2 = 25}

-- Define the line that contains the center of C
def center_line : Set (ℝ × ℝ) :=
  {p | 2 * p.1 - p.2 - 2 = 0}

-- Define the line l
def line_l (k : ℝ) : Set (ℝ × ℝ) :=
  {p | p.2 - 5 = k * (p.1 + 2)}

-- State the theorem
theorem circle_and_line_intersection
  (h1 : ((-3, 3) : ℝ × ℝ) ∈ circle_C)
  (h2 : ((1, -5) : ℝ × ℝ) ∈ circle_C)
  (h3 : ∃ c, c ∈ circle_C ∧ c ∈ center_line)
  (h4 : ∀ k > 0, ∃ A B, A ≠ B ∧ A ∈ circle_C ∧ B ∈ circle_C ∧ A ∈ line_l k ∧ B ∈ line_l k)
  (h5 : (-2, 5) ∈ line_l k) :
  (∀ p ∈ circle_C, (p.1 - 1)^2 + p.2^2 = 25) ∧
  (∀ k > 15/8, ∃ A B, A ≠ B ∧ A ∈ circle_C ∧ B ∈ circle_C ∧ A ∈ line_l k ∧ B ∈ line_l k) ∧
  (∀ k ≤ 15/8, ¬∃ A B, A ≠ B ∧ A ∈ circle_C ∧ B ∈ circle_C ∧ A ∈ line_l k ∧ B ∈ line_l k) :=
by sorry

end NUMINAMATH_CALUDE_circle_and_line_intersection_l1334_133450


namespace NUMINAMATH_CALUDE_expression_equals_one_l1334_133426

theorem expression_equals_one (a b c k : ℝ) (ha : a ≠ 0) (hb : b ≠ 0) (hc : c ≠ 0) (hk : k ≠ 0)
  (sum_zero : a + b + c = 0) (a_squared : a^2 = k * b^2) :
  (a^2 * b^2) / ((a^2 - b*c) * (b^2 - a*c)) +
  (a^2 * c^2) / ((a^2 - b*c) * (c^2 - a*b)) +
  (b^2 * c^2) / ((b^2 - a*c) * (c^2 - a*b)) = 1 :=
by sorry

end NUMINAMATH_CALUDE_expression_equals_one_l1334_133426


namespace NUMINAMATH_CALUDE_exists_sequence_with_finite_primes_l1334_133422

theorem exists_sequence_with_finite_primes :
  ∃ (a : ℕ → ℕ), 
    (∀ n m : ℕ, n < m → a n < a m) ∧ 
    (∀ k : ℕ, k ≥ 2 → ∃ N : ℕ, ∀ n ≥ N, ¬ Prime (k + a n)) :=
by sorry

end NUMINAMATH_CALUDE_exists_sequence_with_finite_primes_l1334_133422


namespace NUMINAMATH_CALUDE_quadratic_equation_real_roots_quadratic_equation_real_roots_for_m_1_l1334_133451

theorem quadratic_equation_real_roots (m : ℝ) : 
  (∃ x : ℝ, (x + 2)^2 = m + 2) ↔ m ≥ -2 :=
by sorry

-- Example for m = 1
theorem quadratic_equation_real_roots_for_m_1 : 
  ∃ x : ℝ, (x + 2)^2 = 1 + 2 :=
by sorry

end NUMINAMATH_CALUDE_quadratic_equation_real_roots_quadratic_equation_real_roots_for_m_1_l1334_133451


namespace NUMINAMATH_CALUDE_weight_loss_challenge_l1334_133419

theorem weight_loss_challenge (initial_weight : ℝ) (clothes_weight_percentage : ℝ) 
  (h1 : clothes_weight_percentage > 0)
  (h2 : initial_weight > 0) : 
  (0.90 * initial_weight + clothes_weight_percentage * 0.90 * initial_weight) / initial_weight = 0.918 → 
  clothes_weight_percentage = 0.02 := by
sorry

end NUMINAMATH_CALUDE_weight_loss_challenge_l1334_133419


namespace NUMINAMATH_CALUDE_school_teacher_count_l1334_133414

/-- Represents the number of students and teachers in a grade --/
structure GradeData where
  students : ℕ
  teachers : ℕ

/-- Proves that given the conditions, the number of teachers in grade A is 8 and in grade B is 26 --/
theorem school_teacher_count 
  (gradeA gradeB : GradeData)
  (ratioA : gradeA.students = 30 * gradeA.teachers)
  (ratioB : gradeB.students = 40 * gradeB.teachers)
  (newRatioA : gradeA.students + 60 = 25 * (gradeA.teachers + 4))
  (newRatioB : gradeB.students + 80 = 35 * (gradeB.teachers + 6))
  : gradeA.teachers = 8 ∧ gradeB.teachers = 26 := by
  sorry

#check school_teacher_count

end NUMINAMATH_CALUDE_school_teacher_count_l1334_133414


namespace NUMINAMATH_CALUDE_quadratic_polynomial_negative_root_l1334_133452

-- Define a quadratic polynomial type
def QuadraticPolynomial (α : Type*) [Ring α] := α → α

-- Define the property of having two distinct real roots
def HasTwoDistinctRealRoots (P : QuadraticPolynomial ℝ) : Prop :=
  ∃ (r₁ r₂ : ℝ), r₁ ≠ r₂ ∧ P r₁ = 0 ∧ P r₂ = 0

-- Define the inequality condition
def SatisfiesInequality (P : QuadraticPolynomial ℝ) : Prop :=
  ∀ (a b : ℝ), (abs a ≥ 2017 ∧ abs b ≥ 2017) → P (a^2 + b^2) ≥ P (2*a*b)

-- Define the property of having at least one negative root
def HasNegativeRoot (P : QuadraticPolynomial ℝ) : Prop :=
  ∃ (r : ℝ), r < 0 ∧ P r = 0

-- The main theorem
theorem quadratic_polynomial_negative_root 
  (P : QuadraticPolynomial ℝ) 
  (h1 : HasTwoDistinctRealRoots P) 
  (h2 : SatisfiesInequality P) : 
  HasNegativeRoot P :=
sorry

end NUMINAMATH_CALUDE_quadratic_polynomial_negative_root_l1334_133452


namespace NUMINAMATH_CALUDE_double_sum_equals_seven_fourths_l1334_133454

/-- The double sum of 1/(i^2j + 2ij + ij^2) over positive integers i and j from 1 to infinity equals 7/4 -/
theorem double_sum_equals_seven_fourths :
  (∑' i : ℕ+, ∑' j : ℕ+, (1 : ℝ) / ((i.val^2 * j.val) + (2 * i.val * j.val) + (i.val * j.val^2))) = 7/4 :=
by sorry

end NUMINAMATH_CALUDE_double_sum_equals_seven_fourths_l1334_133454


namespace NUMINAMATH_CALUDE_unique_special_number_l1334_133490

/-- A two-digit number satisfying specific divisibility properties -/
def SpecialNumber (n : ℕ) : Prop :=
  10 ≤ n ∧ n ≤ 99 ∧ 
  2 ∣ n ∧
  3 ∣ (n + 1) ∧
  4 ∣ (n + 2) ∧
  5 ∣ (n + 3)

/-- Theorem stating that 62 is the unique two-digit number satisfying the given conditions -/
theorem unique_special_number : ∃! n, SpecialNumber n :=
  sorry

end NUMINAMATH_CALUDE_unique_special_number_l1334_133490


namespace NUMINAMATH_CALUDE_conic_equation_not_parabola_l1334_133429

/-- Represents a conic section equation of the form mx² + ny² = 1 -/
structure ConicEquation where
  m : ℝ
  n : ℝ

/-- Defines the possible types of conic sections -/
inductive ConicType
  | Circle
  | Ellipse
  | Hyperbola
  | Parabola

/-- States that a conic equation cannot represent a parabola -/
theorem conic_equation_not_parabola (eq : ConicEquation) : 
  ∃ (t : ConicType), t ≠ ConicType.Parabola ∧ 
  (∀ (x y : ℝ), eq.m * x^2 + eq.n * y^2 = 1 → 
    ∃ (a b c d e f : ℝ), a * x^2 + b * y^2 + c * x * y + d * x + e * y + f = 0) :=
sorry

end NUMINAMATH_CALUDE_conic_equation_not_parabola_l1334_133429


namespace NUMINAMATH_CALUDE_largest_n_divisibility_l1334_133492

theorem largest_n_divisibility : ∃ (n : ℕ), 
  (∀ m : ℕ, m > n → ¬((m + 20) ∣ (m^3 + 200))) ∧ 
  ((n + 20) ∣ (n^3 + 200)) ∧ 
  n = 7780 := by
  sorry

end NUMINAMATH_CALUDE_largest_n_divisibility_l1334_133492


namespace NUMINAMATH_CALUDE_nikolai_faster_l1334_133444

/-- Represents a mountain goat with a specific jump distance -/
structure Goat where
  name : String
  jump_distance : ℕ

/-- Calculates the number of jumps needed to cover a given distance -/
def jumps_needed (g : Goat) (distance : ℕ) : ℕ :=
  (distance + g.jump_distance - 1) / g.jump_distance

theorem nikolai_faster (nikolai gennady : Goat)
  (h1 : nikolai.jump_distance = 4)
  (h2 : gennady.jump_distance = 6)
  (h3 : jumps_needed nikolai 2000 * nikolai.jump_distance = 2000)
  (h4 : jumps_needed gennady 2000 * gennady.jump_distance = 2004) :
  jumps_needed nikolai 2000 < jumps_needed gennady 2000 := by
  sorry

#eval jumps_needed (Goat.mk "Nikolai" 4) 2000
#eval jumps_needed (Goat.mk "Gennady" 6) 2000

end NUMINAMATH_CALUDE_nikolai_faster_l1334_133444


namespace NUMINAMATH_CALUDE_smallest_integer_with_remainders_l1334_133418

theorem smallest_integer_with_remainders : ∃ n : ℕ, 
  (∀ m : ℕ, m > 0 ∧ m < n → 
    (m % 6 ≠ 3 ∨ m % 8 ≠ 5 ∨ m % 9 ≠ 2)) ∧
  n % 6 = 3 ∧ 
  n % 8 = 5 ∧ 
  n % 9 = 2 ∧
  n = 237 :=
by sorry

end NUMINAMATH_CALUDE_smallest_integer_with_remainders_l1334_133418


namespace NUMINAMATH_CALUDE_solve_equation_l1334_133495

theorem solve_equation (p q : ℝ) (h1 : 1 < p) (h2 : p < q) 
  (h3 : 1 / p + 1 / q = 1) (h4 : p * q = 8) : q = 4 + 2 * Real.sqrt 2 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l1334_133495


namespace NUMINAMATH_CALUDE_two_digit_number_ratio_l1334_133486

theorem two_digit_number_ratio (a b : ℕ) : 
  a ≤ 9 ∧ b ≤ 9 ∧ a ≠ 0 → -- Ensure a and b are single digits and a is not 0
  (10 * a + b) * 6 = (10 * b + a) * 5 → -- Ratio condition
  10 * a + b = 45 := by
sorry

end NUMINAMATH_CALUDE_two_digit_number_ratio_l1334_133486


namespace NUMINAMATH_CALUDE_complex_equation_solution_l1334_133434

theorem complex_equation_solution (z₁ z₂ : ℂ) : 
  z₁ = 1 - I ∧ z₁ * z₂ = 1 + I → z₂ = I :=
by sorry

end NUMINAMATH_CALUDE_complex_equation_solution_l1334_133434


namespace NUMINAMATH_CALUDE_quadratic_rewrite_l1334_133421

theorem quadratic_rewrite (x : ℝ) : 
  ∃ (a b c : ℤ), 16 * x^2 - 40 * x + 18 = (a * x + b)^2 + c ∧ a * b = -20 := by
  sorry

end NUMINAMATH_CALUDE_quadratic_rewrite_l1334_133421


namespace NUMINAMATH_CALUDE_log_216_equals_3log2_plus_3log3_l1334_133440

theorem log_216_equals_3log2_plus_3log3 : Real.log 216 = 3 * Real.log 2 + 3 * Real.log 3 := by
  sorry

end NUMINAMATH_CALUDE_log_216_equals_3log2_plus_3log3_l1334_133440


namespace NUMINAMATH_CALUDE_f_is_decreasing_l1334_133411

-- Define the function f
def f (a b : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x - 2

-- Define the property of being an even function
def is_even_function (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)

-- Define the property of being a decreasing function on an interval
def is_decreasing_on (f : ℝ → ℝ) (a b : ℝ) : Prop :=
  ∀ x y, a ≤ x ∧ x < y ∧ y ≤ b → f y < f x

-- State the theorem
theorem f_is_decreasing (a b : ℝ) :
  is_even_function (f a b) ∧ (Set.Icc (1 + a) 2).Nonempty →
  is_decreasing_on (f a b) 1 2 := by
  sorry

end NUMINAMATH_CALUDE_f_is_decreasing_l1334_133411


namespace NUMINAMATH_CALUDE_smallest_consecutive_even_integer_l1334_133403

/-- Represents three consecutive even integers -/
structure ConsecutiveEvenIntegers where
  middle : ℕ
  is_even : Even middle

/-- Checks if a number is two-digit -/
def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99

/-- The property that the sum of the integers is one-fifth of their product -/
def sum_is_one_fifth_of_product (integers : ConsecutiveEvenIntegers) : Prop :=
  (integers.middle - 2) + integers.middle + (integers.middle + 2) = 
    ((integers.middle - 2) * integers.middle * (integers.middle + 2)) / 5

theorem smallest_consecutive_even_integer :
  ∃ (integers : ConsecutiveEvenIntegers),
    (is_two_digit (integers.middle - 2)) ∧
    (is_two_digit integers.middle) ∧
    (is_two_digit (integers.middle + 2)) ∧
    (sum_is_one_fifth_of_product integers) ∧
    (integers.middle - 2 = 86) := by
  sorry

end NUMINAMATH_CALUDE_smallest_consecutive_even_integer_l1334_133403


namespace NUMINAMATH_CALUDE_sufficient_not_necessary_l1334_133442

theorem sufficient_not_necessary (a : ℝ) : 
  (∀ a, a > 1 → 1/a < 1) ∧ (∃ a, 1/a < 1 ∧ ¬(a > 1)) := by
  sorry

end NUMINAMATH_CALUDE_sufficient_not_necessary_l1334_133442


namespace NUMINAMATH_CALUDE_min_modulus_on_circle_l1334_133416

theorem min_modulus_on_circle (z : ℂ) (h : Complex.abs (z - (1 + Complex.I)) = 1) :
  ∃ (w : ℂ), Complex.abs w = Real.sqrt 2 - 1 ∧ 
  ∀ (v : ℂ), Complex.abs (v - (1 + Complex.I)) = 1 → Complex.abs v ≥ Complex.abs w :=
by sorry

end NUMINAMATH_CALUDE_min_modulus_on_circle_l1334_133416


namespace NUMINAMATH_CALUDE_division_problem_l1334_133425

theorem division_problem (number quotient remainder divisor : ℕ) : 
  number = quotient * divisor + remainder →
  divisor = 163 →
  quotient = 76 →
  remainder = 13 →
  number = 12401 := by
sorry

end NUMINAMATH_CALUDE_division_problem_l1334_133425


namespace NUMINAMATH_CALUDE_perpendicular_vectors_vector_sum_magnitude_l1334_133480

def a : ℝ × ℝ := (2, 4)
def b (m : ℝ) : ℝ × ℝ := (m, -1)

theorem perpendicular_vectors (m : ℝ) :
  (a.1 * (b m).1 + a.2 * (b m).2 = 0) → m = 2 := by sorry

theorem vector_sum_magnitude (m : ℝ) :
  ((a.1 + (b m).1)^2 + (a.2 + (b m).2)^2 = 25) → (m = 2 ∨ m = -6) := by sorry

end NUMINAMATH_CALUDE_perpendicular_vectors_vector_sum_magnitude_l1334_133480


namespace NUMINAMATH_CALUDE_investment_percentage_proof_l1334_133436

theorem investment_percentage_proof (total_sum P1 P2 x : ℝ) : 
  total_sum = 1600 →
  P1 + P2 = total_sum →
  P2 = 1100 →
  (P1 * x / 100) + (P2 * 5 / 100) = 85 →
  x = 6 := by
sorry

end NUMINAMATH_CALUDE_investment_percentage_proof_l1334_133436


namespace NUMINAMATH_CALUDE_chessboard_star_property_l1334_133424

/-- Represents a chessboard with stars -/
structure Chessboard (n : ℕ) where
  has_star : Fin n → Fin n → Prop

/-- Represents a set of rows or columns -/
def Subset (n : ℕ) := Fin n → Prop

/-- Checks if a subset is not the entire set -/
def is_proper_subset {n : ℕ} (s : Subset n) : Prop :=
  ∃ i, ¬s i

/-- Checks if a column has exactly one uncrossed star after crossing out rows -/
def column_has_one_star {n : ℕ} (b : Chessboard n) (crossed_rows : Subset n) (j : Fin n) : Prop :=
  ∃! i, ¬crossed_rows i ∧ b.has_star i j

/-- Checks if a row has exactly one uncrossed star after crossing out columns -/
def row_has_one_star {n : ℕ} (b : Chessboard n) (crossed_cols : Subset n) (i : Fin n) : Prop :=
  ∃! j, ¬crossed_cols j ∧ b.has_star i j

/-- The main theorem -/
theorem chessboard_star_property {n : ℕ} (b : Chessboard n) :
  (∀ crossed_rows : Subset n, is_proper_subset crossed_rows →
    ∃ j, column_has_one_star b crossed_rows j) →
  (∀ crossed_cols : Subset n, is_proper_subset crossed_cols →
    ∃ i, row_has_one_star b crossed_cols i) :=
by sorry

end NUMINAMATH_CALUDE_chessboard_star_property_l1334_133424


namespace NUMINAMATH_CALUDE_horizontal_grid_lines_length_6_10_l1334_133494

/-- Represents a right-angled triangle on a grid -/
structure GridTriangle where
  base : ℕ
  height : ℕ

/-- Calculates the total length of horizontal grid lines inside a right-angled triangle -/
def horizontalGridLinesLength (t : GridTriangle) : ℕ :=
  (t.base * (t.height - 1)) / 2

/-- The theorem stating the total length of horizontal grid lines for the specific triangle -/
theorem horizontal_grid_lines_length_6_10 :
  horizontalGridLinesLength { base := 10, height := 6 } = 27 := by
  sorry

#eval horizontalGridLinesLength { base := 10, height := 6 }

end NUMINAMATH_CALUDE_horizontal_grid_lines_length_6_10_l1334_133494


namespace NUMINAMATH_CALUDE_mikes_books_l1334_133468

theorem mikes_books (tim_books : ℕ) (total_books : ℕ) (h1 : tim_books = 22) (h2 : total_books = 42) :
  total_books - tim_books = 20 := by
  sorry

end NUMINAMATH_CALUDE_mikes_books_l1334_133468
