import Mathlib

namespace divisible_by_4_or_5_count_l2725_272575

def count_divisible (n : ℕ) : ℕ :=
  (n / 4) + (n / 5) - (n / 20)

theorem divisible_by_4_or_5_count :
  count_divisible 60 = 24 := by
  sorry

end divisible_by_4_or_5_count_l2725_272575


namespace central_angle_unchanged_l2725_272543

/-- Theorem: When a circle's radius is doubled and the arc length is doubled, the central angle of the sector remains unchanged. -/
theorem central_angle_unchanged 
  (r : ℝ) 
  (l : ℝ) 
  (h_positive_r : r > 0) 
  (h_positive_l : l > 0) : 
  (l / r) = ((2 * l) / (2 * r)) := by 
sorry

end central_angle_unchanged_l2725_272543


namespace complex_operations_l2725_272588

theorem complex_operations (z₁ z₂ : ℂ) 
  (h₁ : z₁ = 2 - 3 * Complex.I) 
  (h₂ : z₂ = (15 - 5 * Complex.I) / (2 + Complex.I)^2) : 
  z₁ * z₂ = -7 - 9 * Complex.I ∧ 
  z₁ / z₂ = 11/10 + 3/10 * Complex.I := by
  sorry

end complex_operations_l2725_272588


namespace exists_x_squared_sum_l2725_272567

theorem exists_x_squared_sum : ∃ x : ℕ, 106 * 106 + x * x = 19872 := by
  sorry

end exists_x_squared_sum_l2725_272567


namespace inequality_and_equality_conditions_l2725_272583

theorem inequality_and_equality_conditions (x y z : ℝ) 
  (hx : x ≥ 0) (hy : y ≥ 0) (hz : z ≥ 0) (hnz : x > 0 ∨ y > 0 ∨ z > 0) : 
  (2*x^2 - x + y + z)/(x + y^2 + z^2) + 
  (2*y^2 + x - y + z)/(x^2 + y + z^2) + 
  (2*z^2 + x + y - z)/(x^2 + y^2 + z) ≥ 3 ∧ 
  ((2*x^2 - x + y + z)/(x + y^2 + z^2) + 
   (2*y^2 + x - y + z)/(x^2 + y + z^2) + 
   (2*z^2 + x + y - z)/(x^2 + y^2 + z) = 3 ↔ 
   (∃ t : ℝ, t > 0 ∧ x = t ∧ y = t ∧ z = t) ∨ 
   (∃ t : ℝ, 0 ≤ t ∧ t ≤ 1 ∧ x = t ∧ y = t ∧ z = 1 - t)) :=
by sorry

end inequality_and_equality_conditions_l2725_272583


namespace trapezoid_longer_base_l2725_272572

/-- Represents a trapezoid with given properties -/
structure Trapezoid where
  shorter_base : ℝ
  longer_base : ℝ
  midline_segment : ℝ
  height : ℝ

/-- Theorem: The longer base of a trapezoid with specific properties is 90 -/
theorem trapezoid_longer_base 
  (t : Trapezoid) 
  (h1 : t.shorter_base = 80) 
  (h2 : t.midline_segment = 5) 
  (h3 : t.height = 3 * t.midline_segment) 
  (h4 : t.midline_segment = (t.longer_base - t.shorter_base) / 2) : 
  t.longer_base = 90 := by
  sorry

end trapezoid_longer_base_l2725_272572


namespace angle_with_double_supplement_is_60_degrees_l2725_272587

theorem angle_with_double_supplement_is_60_degrees (α : Real) :
  (180 - α = 2 * α) → α = 60 := by
  sorry

end angle_with_double_supplement_is_60_degrees_l2725_272587


namespace arithmetic_sequence_common_difference_l2725_272513

-- Define an arithmetic sequence
def is_arithmetic_sequence (a : ℕ → ℚ) : Prop :=
  ∃ d : ℚ, ∀ n : ℕ, a (n + 1) = a n + d

-- State the theorem
theorem arithmetic_sequence_common_difference 
  (a : ℕ → ℚ) 
  (h_arithmetic : is_arithmetic_sequence a) 
  (h_a6 : a 6 = 5) 
  (h_a10 : a 10 = 6) : 
  ∃ d : ℚ, d = 1/4 ∧ ∀ n : ℕ, a (n + 1) = a n + d := by
sorry

end arithmetic_sequence_common_difference_l2725_272513


namespace bryan_total_books_l2725_272574

/-- The number of bookshelves Bryan has -/
def num_bookshelves : ℕ := 9

/-- The number of books in each bookshelf -/
def books_per_shelf : ℕ := 56

/-- The total number of books Bryan has -/
def total_books : ℕ := num_bookshelves * books_per_shelf

/-- Theorem stating that the total number of books Bryan has is 504 -/
theorem bryan_total_books : total_books = 504 := by sorry

end bryan_total_books_l2725_272574


namespace phi_equality_l2725_272507

-- Define the set M_φ
def M_phi (φ : ℕ → ℕ) : Set (ℕ → ℤ) :=
  {f | ∀ x, f x > f (φ x)}

-- State the theorem
theorem phi_equality (φ₁ φ₂ : ℕ → ℕ) :
  M_phi φ₁ = M_phi φ₂ → M_phi φ₁ ≠ ∅ → φ₁ = φ₂ := by
  sorry

end phi_equality_l2725_272507


namespace double_earnings_cars_needed_l2725_272509

/-- Represents the earnings and sales of a car salesman -/
structure CarSalesman where
  baseSalary : ℕ
  commissionPerCar : ℕ
  marchEarnings : ℕ

/-- Calculates the number of cars needed to be sold to reach a target earning -/
def carsNeededForTarget (s : CarSalesman) (targetEarnings : ℕ) : ℕ :=
  ((targetEarnings - s.baseSalary) / s.commissionPerCar : ℕ)

/-- Theorem: A car salesman needs to sell 15 cars in April to double his March earnings -/
theorem double_earnings_cars_needed (s : CarSalesman) 
    (h1 : s.baseSalary = 1000)
    (h2 : s.commissionPerCar = 200)
    (h3 : s.marchEarnings = 2000) : 
  carsNeededForTarget s (2 * s.marchEarnings) = 15 := by
  sorry

#eval carsNeededForTarget ⟨1000, 200, 2000⟩ 4000

end double_earnings_cars_needed_l2725_272509


namespace complex_equation_solution_l2725_272520

theorem complex_equation_solution (a b : ℝ) (i : ℂ) 
  (h1 : i * i = -1) 
  (h2 : (2 : ℂ) + a * i = (b * i - 1) * i) : 
  (a : ℂ) + b * i = -1 - 2 * i :=
by sorry

end complex_equation_solution_l2725_272520


namespace smallest_integer_solution_l2725_272518

theorem smallest_integer_solution : 
  (∀ x : ℤ, x < 1 → (x : ℚ) / 4 + 3 / 7 ≤ 2 / 3) ∧ 
  (1 : ℚ) / 4 + 3 / 7 > 2 / 3 :=
by sorry

end smallest_integer_solution_l2725_272518


namespace opposite_of_four_l2725_272594

-- Define the concept of opposite
def opposite (a : ℝ) : ℝ := -a

-- Theorem statement
theorem opposite_of_four : opposite 4 = -4 := by
  -- The proof goes here
  sorry

end opposite_of_four_l2725_272594


namespace rectangle_count_l2725_272542

/-- The number of different rectangles in a 5x5 grid -/
def num_rectangles : ℕ := 100

/-- The number of rows in the grid -/
def num_rows : ℕ := 5

/-- The number of columns in the grid -/
def num_columns : ℕ := 5

/-- The number of ways to choose 2 items from n items -/
def choose_two (n : ℕ) : ℕ := n * (n - 1) / 2

theorem rectangle_count :
  num_rectangles = choose_two num_rows * choose_two num_columns :=
sorry

end rectangle_count_l2725_272542


namespace exactly_three_non_congruent_triangles_l2725_272548

/-- A triangle with integer side lengths -/
structure IntTriangle where
  a : ℕ
  b : ℕ
  c : ℕ
  triangle_inequality : a + b > c ∧ b + c > a ∧ c + a > b

/-- Check if two triangles are congruent -/
def congruent (t1 t2 : IntTriangle) : Prop :=
  (t1.a = t2.a ∧ t1.b = t2.b ∧ t1.c = t2.c) ∨
  (t1.a = t2.b ∧ t1.b = t2.c ∧ t1.c = t2.a) ∨
  (t1.a = t2.c ∧ t1.b = t2.a ∧ t1.c = t2.b)

/-- The set of all triangles with perimeter 11 -/
def triangles_with_perimeter_11 : Set IntTriangle :=
  {t : IntTriangle | t.a + t.b + t.c = 11}

/-- The theorem to be proved -/
theorem exactly_three_non_congruent_triangles :
  ∃ (t1 t2 t3 : IntTriangle),
    t1 ∈ triangles_with_perimeter_11 ∧
    t2 ∈ triangles_with_perimeter_11 ∧
    t3 ∈ triangles_with_perimeter_11 ∧
    ¬congruent t1 t2 ∧ ¬congruent t1 t3 ∧ ¬congruent t2 t3 ∧
    ∀ (t : IntTriangle),
      t ∈ triangles_with_perimeter_11 →
      (congruent t t1 ∨ congruent t t2 ∨ congruent t t3) :=
by sorry

end exactly_three_non_congruent_triangles_l2725_272548


namespace value_of_y_l2725_272501

theorem value_of_y (x y : ℝ) (h1 : 1.5 * x = 0.3 * y) (h2 : x = 24) : y = 120 := by
  sorry

end value_of_y_l2725_272501


namespace root_product_theorem_l2725_272526

theorem root_product_theorem (a b m p : ℝ) : 
  (a^2 - m*a + 3 = 0) → 
  (b^2 - m*b + 3 = 0) → 
  ∃ r, ((a + 1/b)^2 - p*(a + 1/b) + r = 0) ∧ 
       ((b + 1/a)^2 - p*(b + 1/a) + r = 0) → 
  r = 16/3 := by
sorry

end root_product_theorem_l2725_272526


namespace souvenir_problem_l2725_272595

/-- Represents the cost and quantity of souvenirs --/
structure SouvenirPlan where
  costA : ℕ
  costB : ℕ
  quantityA : ℕ
  quantityB : ℕ

/-- Checks if a souvenir plan satisfies all conditions --/
def isValidPlan (plan : SouvenirPlan) : Prop :=
  plan.costA + 20 = plan.costB ∧
  9 * plan.costA = 7 * plan.costB ∧
  plan.quantityB = 2 * plan.quantityA + 5 ∧
  plan.quantityA ≥ 18 ∧
  plan.costA * plan.quantityA + plan.costB * plan.quantityB ≤ 5450

/-- The correct costs and possible purchasing plans --/
def correctSolution : Prop :=
  ∃ (plan : SouvenirPlan),
    isValidPlan plan ∧
    plan.costA = 70 ∧
    plan.costB = 90 ∧
    (plan.quantityA = 18 ∧ plan.quantityB = 41) ∨
    (plan.quantityA = 19 ∧ plan.quantityB = 43) ∨
    (plan.quantityA = 20 ∧ plan.quantityB = 45)

theorem souvenir_problem : correctSolution := by
  sorry

end souvenir_problem_l2725_272595


namespace sqrt_sum_problem_l2725_272571

theorem sqrt_sum_problem (y : ℝ) (h : Real.sqrt (64 - y^2) - Real.sqrt (36 - y^2) = 4) :
  Real.sqrt (64 - y^2) + Real.sqrt (36 - y^2) = 7 := by
  sorry

end sqrt_sum_problem_l2725_272571


namespace higher_variance_greater_fluctuations_l2725_272586

-- Define the properties of the two data sets
def mean_A : ℝ := 5
def mean_B : ℝ := 5
def variance_A : ℝ := 0.1
def variance_B : ℝ := 0.2

-- Define a function to represent fluctuations based on variance
def fluctuations (variance : ℝ) : ℝ := variance

-- Theorem stating that higher variance implies greater fluctuations
theorem higher_variance_greater_fluctuations :
  variance_A < variance_B →
  fluctuations variance_A < fluctuations variance_B :=
by sorry

end higher_variance_greater_fluctuations_l2725_272586


namespace not_p_necessary_not_sufficient_for_not_q_l2725_272546

theorem not_p_necessary_not_sufficient_for_not_q :
  ∃ (x : ℝ), (¬(x^2 < 1) → ¬(x < 1)) ∧
  ∃ (y : ℝ), ¬(y < 1) ∧ ¬¬(y^2 < 1) :=
by sorry

end not_p_necessary_not_sufficient_for_not_q_l2725_272546


namespace park_tree_count_l2725_272530

/-- The number of dogwood trees in the park after 5 days of planting and one uprooting event -/
def final_tree_count (initial : ℕ) (day1 : ℕ) (day5 : ℕ) : ℕ :=
  let day2 := day1 / 2
  let day3 := day2 * 4
  let day4 := 5  -- Trees replaced due to uprooting
  initial + day1 + day2 + day3 + day4 + day5

/-- Theorem stating the final number of trees in the park -/
theorem park_tree_count : final_tree_count 39 24 15 = 143 := by
  sorry

end park_tree_count_l2725_272530


namespace cube_opposite_face_l2725_272564

structure Cube where
  faces : Finset Char
  adjacent : Char → Finset Char

def is_opposite (c : Cube) (face1 face2 : Char) : Prop :=
  face1 ∈ c.faces ∧ face2 ∈ c.faces ∧ face1 ≠ face2 ∧
  c.adjacent face1 ∩ c.adjacent face2 = ∅

theorem cube_opposite_face (c : Cube) :
  c.faces = {'A', 'B', 'C', 'D', 'E', 'F'} →
  c.adjacent 'E' = {'A', 'B', 'C', 'D'} →
  is_opposite c 'E' 'F' :=
by sorry

end cube_opposite_face_l2725_272564


namespace fraction_property_l2725_272566

theorem fraction_property (a : ℕ) (h : a > 1) :
  let b := 2 * a - 1
  0 < a ∧ a < b ∧ (a - 1) / (b - 1) = 1 / 2 := by
  sorry

end fraction_property_l2725_272566


namespace square_arrangement_sum_l2725_272539

/-- The sum of integers from -12 to 18 inclusive -/
def total_sum : ℤ := 93

/-- The size of the square matrix -/
def matrix_size : ℕ := 6

/-- The common sum for each row, column, and main diagonal -/
def common_sum : ℚ := 15.5

theorem square_arrangement_sum :
  total_sum = matrix_size * (common_sum : ℚ).num / (common_sum : ℚ).den :=
sorry

end square_arrangement_sum_l2725_272539


namespace salary_solution_l2725_272521

def salary_problem (J F M A May : ℕ) : Prop :=
  (J + F + M + A) / 4 = 8000 ∧
  (F + M + A + May) / 4 = 8700 ∧
  J = 3700 ∧
  May = 6500

theorem salary_solution :
  ∀ J F M A May : ℕ,
    salary_problem J F M A May →
    May = 6500 := by
  sorry

end salary_solution_l2725_272521


namespace andys_walk_distance_l2725_272517

/-- Proves the distance between Andy's house and the market given his walking routes --/
theorem andys_walk_distance (house_to_school : ℝ) (school_to_park : ℝ) (total_distance : ℝ)
  (h1 : house_to_school = 50)
  (h2 : school_to_park = 25)
  (h3 : total_distance = 345) :
  total_distance - (2 * house_to_school + school_to_park + school_to_park / 2) = 195 := by
  sorry


end andys_walk_distance_l2725_272517


namespace hcf_problem_l2725_272568

def is_hcf (h : ℕ) (a b : ℕ) : Prop :=
  h ∣ a ∧ h ∣ b ∧ ∀ d : ℕ, d ∣ a → d ∣ b → d ≤ h

def is_lcm (l : ℕ) (a b : ℕ) : Prop :=
  a ∣ l ∧ b ∣ l ∧ ∀ m : ℕ, a ∣ m → b ∣ m → l ≤ m

theorem hcf_problem (a b : ℕ) (h : ℕ) :
  a = 345 →
  (∃ l : ℕ, is_lcm l a b ∧ l = h * 13 * 15) →
  is_hcf h a b →
  h = 15 := by sorry

end hcf_problem_l2725_272568


namespace stock_price_after_two_years_l2725_272511

/-- The stock price after two years of changes -/
def final_stock_price (initial_price : ℝ) (first_year_increase : ℝ) (second_year_decrease : ℝ) : ℝ :=
  initial_price * (1 + first_year_increase) * (1 - second_year_decrease)

/-- Theorem stating the final stock price after two years -/
theorem stock_price_after_two_years :
  final_stock_price 50 1.5 0.3 = 87.5 := by
  sorry

end stock_price_after_two_years_l2725_272511


namespace additive_fun_properties_l2725_272580

/-- A function satisfying f(x+y) = f(x) + f(y) for all x, y ∈ ℝ -/
def AdditiveFun (f : ℝ → ℝ) : Prop :=
  ∀ x y : ℝ, f (x + y) = f x + f y

theorem additive_fun_properties
  (f : ℝ → ℝ)
  (h_additive : AdditiveFun f)
  (h_increasing : Monotone f)
  (h_f1 : f 1 = 1)
  (h_f2a : ∀ a : ℝ, f (2 * a) > f (a - 1) + 2) :
  (f 0 = 0) ∧
  (∀ x : ℝ, f (-x) = -f x) ∧
  (∀ a : ℝ, a > 1) :=
by sorry

end additive_fun_properties_l2725_272580


namespace count_blocks_with_three_differences_l2725_272557

-- Define the properties of a block
structure BlockProperty where
  material : Fin 2
  size : Fin 2
  color : Fin 4
  shape : Fin 4
  pattern : Fin 2

-- Define the set of all possible blocks
def AllBlocks : Finset BlockProperty := sorry

-- Define a function to count the differences between two blocks
def countDifferences (b1 b2 : BlockProperty) : Nat := sorry

-- Define the reference block (plastic large red circle striped)
def referenceBlock : BlockProperty := sorry

-- Theorem statement
theorem count_blocks_with_three_differences :
  (AllBlocks.filter (fun b => countDifferences b referenceBlock = 3)).card = 21 := by sorry

end count_blocks_with_three_differences_l2725_272557


namespace absolute_value_equation_solution_l2725_272534

theorem absolute_value_equation_solution :
  ∀ x : ℝ, |x + 2| = 3*x - 6 ↔ x = 4 := by
  sorry

end absolute_value_equation_solution_l2725_272534


namespace system_solution_l2725_272556

theorem system_solution :
  ∀ x y : ℝ, x^2 + y^2 = 13 ∧ x * y = 6 → x = 3 ∧ y = 2 := by
sorry

end system_solution_l2725_272556


namespace sum_first_150_remainder_l2725_272516

theorem sum_first_150_remainder (n : Nat) (h : n = 150) : 
  (n * (n + 1) / 2) % 11325 = 0 := by
  sorry

end sum_first_150_remainder_l2725_272516


namespace smallest_quotient_two_digit_numbers_l2725_272531

theorem smallest_quotient_two_digit_numbers :
  ∀ a b : ℕ,
    1 ≤ a ∧ a ≤ 9 ∧ 
    0 ≤ b ∧ b ≤ 9 ∧
    a ≠ b →
    (10 * a + b : ℚ) / (a + b) ≥ 1.9 :=
by sorry

end smallest_quotient_two_digit_numbers_l2725_272531


namespace exposed_sides_count_l2725_272547

/-- Represents a regular polygon with a given number of sides. -/
structure RegularPolygon where
  sides : Nat
  sides_positive : sides > 0

/-- The sequence of regular polygons in the construction. -/
def polygon_sequence : List RegularPolygon :=
  [⟨3, by norm_num⟩, ⟨4, by norm_num⟩, ⟨5, by norm_num⟩, 
   ⟨6, by norm_num⟩, ⟨7, by norm_num⟩, ⟨8, by norm_num⟩, ⟨9, by norm_num⟩]

/-- Calculates the number of exposed sides for a given polygon in the sequence. -/
def exposed_sides (p : RegularPolygon) (index : Nat) : Nat :=
  if index = 0 ∨ index = 6 then p.sides - 1 else p.sides - 2

/-- The total number of exposed sides in the polygon sequence. -/
def total_exposed_sides : Nat :=
  (List.zipWith exposed_sides polygon_sequence (List.range 7)).sum

theorem exposed_sides_count :
  total_exposed_sides = 30 := by
  sorry

end exposed_sides_count_l2725_272547


namespace rectangle_area_reduction_l2725_272541

theorem rectangle_area_reduction (l w : ℝ) : 
  l > 0 ∧ w > 0 ∧ (l - 1) * w = 24 → l * (w - 1) = 18 := by
  sorry

end rectangle_area_reduction_l2725_272541


namespace faster_train_speed_l2725_272527

/-- Given two trains moving in the same direction, prove that the speed of the faster train is 90 kmph -/
theorem faster_train_speed 
  (speed_difference : ℝ) 
  (faster_train_length : ℝ) 
  (crossing_time : ℝ) 
  (h1 : speed_difference = 36) 
  (h2 : faster_train_length = 435) 
  (h3 : crossing_time = 29) : 
  ∃ (faster_speed slower_speed : ℝ), 
    faster_speed - slower_speed = speed_difference ∧ 
    faster_train_length / crossing_time * 3.6 = speed_difference ∧
    faster_speed = 90 := by
  sorry

end faster_train_speed_l2725_272527


namespace water_needed_for_noah_l2725_272559

/-- Represents the recipe ratios and quantities for Noah's orange juice --/
structure OrangeJuiceRecipe where
  orange : ℝ  -- Amount of orange concentrate
  sugar : ℝ   -- Amount of sugar
  water : ℝ   -- Amount of water
  sugar_to_orange_ratio : sugar = 3 * orange
  water_to_sugar_ratio : water = 3 * sugar

/-- Theorem: Given Noah's recipe ratios and 4 cups of orange concentrate, 36 cups of water are needed --/
theorem water_needed_for_noah's_recipe : 
  ∀ (recipe : OrangeJuiceRecipe), 
  recipe.orange = 4 → 
  recipe.water = 36 := by
sorry


end water_needed_for_noah_l2725_272559


namespace sector_arc_length_l2725_272523

/-- Given a sector with central angle 120° and area 300π cm², its arc length is 20π cm. -/
theorem sector_arc_length (θ : ℝ) (S : ℝ) (l : ℝ) : 
  θ = 120 * π / 180 → 
  S = 300 * π → 
  l = 20 * π := by
  sorry

end sector_arc_length_l2725_272523


namespace winter_break_probability_l2725_272514

/-- The probability of getting exactly k successes in n independent trials,
    where each trial has probability p of success. -/
def binomial_probability (n k : ℕ) (p : ℝ) : ℝ :=
  (n.choose k : ℝ) * p^k * (1 - p)^(n - k)

/-- The number of days in the winter break -/
def num_days : ℕ := 5

/-- The probability of clear weather on each day -/
def prob_clear : ℝ := 0.4

/-- The desired number of clear days -/
def desired_clear_days : ℕ := 2

theorem winter_break_probability :
  binomial_probability num_days desired_clear_days prob_clear = 216 / 625 := by
  sorry

end winter_break_probability_l2725_272514


namespace geometric_sequence_common_ratio_l2725_272561

/-- A geometric sequence with sum of first n terms S_n = 3 * 2^n + m has common ratio 2 -/
theorem geometric_sequence_common_ratio 
  (a : ℕ → ℝ) 
  (S : ℕ → ℝ) 
  (h : ∀ n, S n = 3 * 2^n + (S 0 - 3)) : 
  ∃ r : ℝ, r = 2 ∧ ∀ n : ℕ, a (n + 1) = r * a n :=
sorry

end geometric_sequence_common_ratio_l2725_272561


namespace third_side_length_l2725_272504

-- Define a right-angled triangle with side lengths a, b, and x
structure RightTriangle where
  a : ℝ
  b : ℝ
  x : ℝ
  is_right : a^2 + b^2 = x^2 ∨ a^2 + x^2 = b^2 ∨ x^2 + b^2 = a^2

-- Define the theorem
theorem third_side_length (t : RightTriangle) :
  (t.a - 3)^2 + |t.b - 4| = 0 → t.x = 5 ∨ t.x = Real.sqrt 7 := by
  sorry

end third_side_length_l2725_272504


namespace f_min_value_a_range_l2725_272560

-- Define the function f
def f (x : ℝ) : ℝ := 2 * abs (x - 2) - x + 5

-- Theorem for the minimum value of f
theorem f_min_value :
  ∃ (m : ℝ), (∀ x, f x ≥ m) ∧ (∃ x₀, f x₀ = m) ∧ m = 3 :=
sorry

-- Theorem for the range of a
theorem a_range (a : ℝ) :
  (∀ x, |x - a| + |x + 2| ≥ 3) → (a ≤ -5 ∨ a ≥ 1) :=
sorry

end f_min_value_a_range_l2725_272560


namespace program_output_is_one_l2725_272578

/-- Represents the state of the program -/
structure ProgramState :=
  (S : ℕ)
  (n : ℕ)

/-- The update function for the program state -/
def updateState (state : ProgramState) : ProgramState :=
  if state.n > 1 then
    { S := state.S + state.n, n := state.n - 1 }
  else
    state

/-- The termination condition for the program -/
def isTerminated (state : ProgramState) : Prop :=
  state.S ≥ 17 ∧ state.n ≤ 1

/-- The initial state of the program -/
def initialState : ProgramState :=
  { S := 0, n := 5 }

/-- The theorem stating that the program terminates with n = 1 -/
theorem program_output_is_one :
  ∃ (finalState : ProgramState), 
    (∃ (k : ℕ), finalState = (updateState^[k] initialState)) ∧
    isTerminated finalState ∧
    finalState.n = 1 := by
  sorry

end program_output_is_one_l2725_272578


namespace alex_jamie_pairing_probability_l2725_272536

/-- Represents the probability of Alex being paired with Jamie in a class pairing scenario -/
theorem alex_jamie_pairing_probability 
  (total_students : ℕ) 
  (paired_students : ℕ) 
  (h1 : total_students = 50) 
  (h2 : paired_students = 20) 
  (h3 : paired_students < total_students) :
  (1 : ℚ) / (total_students - paired_students - 1 : ℚ) = 1/29 := by
sorry

end alex_jamie_pairing_probability_l2725_272536


namespace equation_solution_l2725_272554

theorem equation_solution (m n : ℝ) : 
  (∀ x : ℝ, (2*x - 5)*(x + m) = 2*x^2 - 3*x + n) → 
  (m = 1 ∧ n = -5) :=
by sorry

end equation_solution_l2725_272554


namespace perpendicular_vectors_l2725_272597

/-- Given two vectors a and b in R², where a is perpendicular to (a - b),
    prove that the y-coordinate of b must be 3. -/
theorem perpendicular_vectors (a b : ℝ × ℝ) (h : a.1 = 1 ∧ a.2 = 2 ∧ b.1 = -1) :
  (a.1 * (a.1 - b.1) + a.2 * (a.2 - b.2) = 0) → b.2 = 3 := by
  sorry

end perpendicular_vectors_l2725_272597


namespace stock_transaction_l2725_272579

/-- Represents the number of shares for each stock --/
structure StockHoldings where
  v : ℕ
  w : ℕ
  x : ℕ
  y : ℕ
  z : ℕ

/-- Calculates the range of a set of numbers --/
def range (s : StockHoldings) : ℕ :=
  max s.v (max s.w (max s.x (max s.y s.z))) - min s.v (min s.w (min s.x (min s.y s.z)))

/-- Theorem representing the stock transaction problem --/
theorem stock_transaction (initial : StockHoldings) 
  (h1 : initial.v = 68)
  (h2 : initial.w = 112)
  (h3 : initial.x = 56)
  (h4 : initial.y = 94)
  (h5 : initial.z = 45)
  (bought_y : ℕ)
  (h6 : bought_y = 23)
  (range_increase : ℕ)
  (h7 : range_increase = 14)
  : ∃ (sold_x : ℕ), 
    let final := StockHoldings.mk 
      initial.v 
      initial.w 
      (initial.x - sold_x)
      (initial.y + bought_y)
      initial.z
    range final = range initial + range_increase ∧ sold_x = 20 := by
  sorry


end stock_transaction_l2725_272579


namespace next_joint_tutoring_day_l2725_272512

def jaclyn_schedule : ℕ := 3
def marcelle_schedule : ℕ := 4
def susanna_schedule : ℕ := 6
def wanda_schedule : ℕ := 7

theorem next_joint_tutoring_day :
  Nat.lcm jaclyn_schedule (Nat.lcm marcelle_schedule (Nat.lcm susanna_schedule wanda_schedule)) = 84 := by
  sorry

end next_joint_tutoring_day_l2725_272512


namespace wire_cutting_l2725_272596

theorem wire_cutting (total_length : ℝ) (ratio : ℝ) (shorter_length : ℝ) : 
  total_length = 21 →
  ratio = 2 / 5 →
  shorter_length + shorter_length / ratio = total_length →
  shorter_length = 6 := by
sorry

end wire_cutting_l2725_272596


namespace artists_contemporary_probability_l2725_272537

/-- Represents the birth year of an artist, measured in years ago --/
def BirthYear := Fin 301

/-- Represents the lifetime of an artist --/
structure Lifetime where
  birth : BirthYear
  death : BirthYear
  age_constraint : death.val = birth.val + 80

/-- Two artists are contemporaries if their lifetimes overlap --/
def are_contemporaries (a b : Lifetime) : Prop :=
  (a.birth.val ≤ b.death.val ∧ b.birth.val ≤ a.death.val) ∨
  (b.birth.val ≤ a.death.val ∧ a.birth.val ≤ b.death.val)

/-- The probability of two artists being contemporaries --/
def probability_contemporaries : ℚ :=
  209 / 225

theorem artists_contemporary_probability :
  probability_contemporaries = 209 / 225 := by sorry


end artists_contemporary_probability_l2725_272537


namespace a4_to_a5_booklet_l2725_272550

theorem a4_to_a5_booklet (n : ℕ) (h : 2 * n + 2 = 74) : n / 4 = 9 := by
  sorry

end a4_to_a5_booklet_l2725_272550


namespace exists_m_divisible_by_2005_l2725_272599

def f (x : ℤ) : ℤ := 3 * x + 2

theorem exists_m_divisible_by_2005 : 
  ∃ m : ℕ+, (3^100 * (m.val + 1) - 1) % 2005 = 0 :=
sorry

end exists_m_divisible_by_2005_l2725_272599


namespace min_tablets_for_given_box_l2725_272573

/-- Given a box with tablets of two types of medicine, this function calculates
    the minimum number of tablets that must be extracted to guarantee at least
    two tablets of each type. -/
def min_tablets_to_extract (tablets_a tablets_b : ℕ) : ℕ :=
  max ((tablets_b + 1) + 2) ((tablets_a + 1) + 2)

/-- Theorem stating that for a box with 10 tablets of medicine A and 13 tablets
    of medicine B, the minimum number of tablets to extract to guarantee at
    least two of each kind is 15. -/
theorem min_tablets_for_given_box :
  min_tablets_to_extract 10 13 = 15 := by sorry

end min_tablets_for_given_box_l2725_272573


namespace min_value_abs_diff_l2725_272569

-- Define the logarithm function
noncomputable def log (base : ℝ) (x : ℝ) : ℝ := Real.log x / Real.log base

-- Define the theorem
theorem min_value_abs_diff (x y : ℝ) :
  log 4 (x + 2*y) + log 1 (x - 2*y) = 1 ∧ 
  x + 2*y > 0 ∧ 
  x - 2*y > 0 →
  ∃ (min_val : ℝ), min_val = Real.sqrt 3 ∧ ∀ (a b : ℝ), 
    (log 4 (a + 2*b) + log 1 (a - 2*b) = 1 ∧ a + 2*b > 0 ∧ a - 2*b > 0) →
    |a| - |b| ≥ min_val :=
by sorry

end min_value_abs_diff_l2725_272569


namespace harrys_age_l2725_272545

theorem harrys_age :
  ∀ (H : ℕ),
  (H + 24 : ℕ) - H / 25 = H + 22 →
  H = 50 :=
by
  sorry

end harrys_age_l2725_272545


namespace front_view_of_given_stack_map_l2725_272565

/-- Represents a column of stacked cubes -/
def Column := List Nat

/-- Represents a stack map as a list of columns -/
def StackMap := List Column

/-- Calculates the maximum height of a column -/
def maxHeight (column : Column) : Nat :=
  column.foldl max 0

/-- Calculates the front view heights of a stack map -/
def frontView (stackMap : StackMap) : List Nat :=
  stackMap.map maxHeight

/-- The given stack map from the problem -/
def givenStackMap : StackMap :=
  [[1, 3], [2, 4, 2], [3, 5], [2]]

theorem front_view_of_given_stack_map :
  frontView givenStackMap = [3, 4, 5, 2] := by
  sorry

end front_view_of_given_stack_map_l2725_272565


namespace smallest_integer_greater_than_root_sum_power_6_l2725_272562

theorem smallest_integer_greater_than_root_sum_power_6 :
  ∃ n : ℕ, n = 970 ∧ (∀ m : ℤ, m > (Real.sqrt 3 + Real.sqrt 2)^6 → m ≥ n) :=
sorry

end smallest_integer_greater_than_root_sum_power_6_l2725_272562


namespace school_pens_problem_l2725_272555

theorem school_pens_problem (pencils : ℕ) (pen_cost pencil_cost total_cost : ℚ) :
  pencils = 38 →
  pencil_cost = 5/2 →
  pen_cost = 7/2 →
  total_cost = 291 →
  ∃ (pens : ℕ), pens * pen_cost + pencils * pencil_cost = total_cost ∧ pens = 56 := by
  sorry

end school_pens_problem_l2725_272555


namespace alan_told_seven_jokes_l2725_272525

/-- The number of jokes Jessy told on Saturday -/
def jessy_jokes : ℕ := 11

/-- The number of jokes Alan told on Saturday -/
def alan_jokes : ℕ := sorry

/-- The total number of jokes both told over two Saturdays -/
def total_jokes : ℕ := 54

/-- Theorem stating that Alan told 7 jokes on Saturday -/
theorem alan_told_seven_jokes :
  alan_jokes = 7 ∧
  jessy_jokes + alan_jokes + 2 * jessy_jokes + 2 * alan_jokes = total_jokes :=
sorry

end alan_told_seven_jokes_l2725_272525


namespace sqrt_x_minus_3_defined_l2725_272532

theorem sqrt_x_minus_3_defined (x : ℝ) : 
  (∃ y : ℝ, y ^ 2 = x - 3) ↔ x ≥ 3 := by
  sorry

end sqrt_x_minus_3_defined_l2725_272532


namespace polynomial_divisibility_l2725_272505

theorem polynomial_divisibility (m : ℤ) : 
  ∃ k : ℤ, (4*m + 5)^2 - 9 = 8 * k := by
  sorry

end polynomial_divisibility_l2725_272505


namespace distance_ratio_is_one_to_one_l2725_272589

def walking_speed : ℝ := 4
def running_speed : ℝ := 8
def total_time : ℝ := 1.5
def total_distance : ℝ := 8

theorem distance_ratio_is_one_to_one :
  ∃ (d_w d_r : ℝ),
    d_w / walking_speed + d_r / running_speed = total_time ∧
    d_w + d_r = total_distance ∧
    d_w / d_r = 1 := by
  sorry

end distance_ratio_is_one_to_one_l2725_272589


namespace ice_cubes_per_tray_l2725_272591

theorem ice_cubes_per_tray (total_ice_cubes : ℕ) (number_of_trays : ℕ) 
  (h1 : total_ice_cubes = 72) 
  (h2 : number_of_trays = 8) 
  (h3 : total_ice_cubes % number_of_trays = 0) : 
  total_ice_cubes / number_of_trays = 9 := by
  sorry

end ice_cubes_per_tray_l2725_272591


namespace mrs_martin_bagels_l2725_272576

/-- The cost of one bagel in dollars -/
def bagel_cost : ℚ := 3/2

/-- Mrs. Martin's purchase -/
def mrs_martin_purchase (coffee_cost bagels : ℚ) : Prop :=
  3 * coffee_cost + bagels * bagel_cost = 51/4

/-- Mr. Martin's purchase -/
def mr_martin_purchase (coffee_cost : ℚ) : Prop :=
  2 * coffee_cost + 5 * bagel_cost = 14

theorem mrs_martin_bagels :
  ∃ (coffee_cost : ℚ), mr_martin_purchase coffee_cost →
    mrs_martin_purchase coffee_cost 2 := by sorry

end mrs_martin_bagels_l2725_272576


namespace paul_needs_21_cans_l2725_272551

/-- Represents the amount of frosting needed for different baked goods -/
structure FrostingNeeds where
  layerCake : ℕ  -- number of layer cakes
  cupcakesDozens : ℕ  -- number of dozens of cupcakes
  singleCakes : ℕ  -- number of single cakes
  browniePans : ℕ  -- number of brownie pans

/-- Calculates the total number of cans of frosting needed -/
def totalFrostingCans (needs : FrostingNeeds) : ℕ :=
  needs.layerCake + (needs.cupcakesDozens + needs.singleCakes + needs.browniePans) / 2

/-- Paul's specific frosting needs for Saturday -/
def paulsFrostingNeeds : FrostingNeeds :=
  { layerCake := 3
  , cupcakesDozens := 6
  , singleCakes := 12
  , browniePans := 18 }

/-- Theorem stating that Paul needs 21 cans of frosting -/
theorem paul_needs_21_cans : totalFrostingCans paulsFrostingNeeds = 21 := by
  sorry

end paul_needs_21_cans_l2725_272551


namespace count_self_inverse_pairs_l2725_272582

/-- A 2x2 matrix of the form [[a, 4], [-9, d]] -/
def special_matrix (a d : ℝ) : Matrix (Fin 2) (Fin 2) ℝ :=
  !![a, 4; -9, d]

/-- The identity matrix of size 2x2 -/
def identity_matrix : Matrix (Fin 2) (Fin 2) ℝ :=
  !![1, 0; 0, 1]

/-- Predicate to check if a matrix is its own inverse -/
def is_self_inverse (a d : ℝ) : Prop :=
  special_matrix a d * special_matrix a d = identity_matrix

/-- The set of all pairs (a, d) where the special matrix is its own inverse -/
def self_inverse_pairs : Set (ℝ × ℝ) :=
  {p | is_self_inverse p.1 p.2}

theorem count_self_inverse_pairs :
  ∃ (s : Finset (ℝ × ℝ)), s.card = 2 ∧ ↑s = self_inverse_pairs :=
sorry

end count_self_inverse_pairs_l2725_272582


namespace simplify_fraction_l2725_272508

theorem simplify_fraction (m n : ℝ) (h : n ≠ 0) : m * n / (n ^ 2) = m / n := by
  sorry

end simplify_fraction_l2725_272508


namespace function_value_range_l2725_272558

theorem function_value_range :
  ∀ x : ℝ, -2 ≤ Real.sqrt 3 * Real.sin (2 * x) + 2 * (Real.cos x) ^ 2 - 1 ∧
           Real.sqrt 3 * Real.sin (2 * x) + 2 * (Real.cos x) ^ 2 - 1 ≤ 2 := by
  sorry

end function_value_range_l2725_272558


namespace apollonius_circle_symmetric_x_axis_l2725_272540

/-- Apollonius Circle -/
def ApolloniusCircle (a : ℝ) : Set (ℝ × ℝ) :=
  {p : ℝ × ℝ | let (x, y) := p
               (x + 1)^2 + y^2 = a^2 * ((x - 1)^2 + y^2)}

/-- Symmetry about x-axis -/
def SymmetricAboutXAxis (S : Set (ℝ × ℝ)) : Prop :=
  ∀ x y, (x, y) ∈ S ↔ (x, -y) ∈ S

theorem apollonius_circle_symmetric_x_axis (a : ℝ) (ha : a > 1) :
  SymmetricAboutXAxis (ApolloniusCircle a) := by
  sorry

end apollonius_circle_symmetric_x_axis_l2725_272540


namespace four_valid_start_days_l2725_272515

/-- Represents the days of the week -/
inductive Weekday
  | Monday
  | Tuesday
  | Wednesday
  | Thursday
  | Friday
  | Saturday
  | Sunday

/-- Counts the number of occurrences of a specific weekday in a 30-day month starting from a given day -/
def countWeekday (start : Weekday) (target : Weekday) : Nat :=
  sorry

/-- Checks if Tuesdays and Fridays are equal in number for a given starting day -/
def hasSameTuesdaysAndFridays (start : Weekday) : Bool :=
  countWeekday start Weekday.Tuesday = countWeekday start Weekday.Friday

/-- The set of all weekdays -/
def allWeekdays : List Weekday :=
  [Weekday.Monday, Weekday.Tuesday, Weekday.Wednesday, Weekday.Thursday, 
   Weekday.Friday, Weekday.Saturday, Weekday.Sunday]

/-- The main theorem stating that exactly 4 weekdays satisfy the condition -/
theorem four_valid_start_days :
  (allWeekdays.filter hasSameTuesdaysAndFridays).length = 4 :=
  sorry

end four_valid_start_days_l2725_272515


namespace sum_of_powers_of_two_l2725_272506

theorem sum_of_powers_of_two : 1 + 1/2 + 1/4 + 1/8 = 15/8 := by sorry

end sum_of_powers_of_two_l2725_272506


namespace rhino_horn_segment_area_l2725_272592

/-- Represents a point in 2D space -/
structure Point where
  x : ℝ
  y : ℝ

/-- Represents a circle -/
structure Circle where
  center : Point
  radius : ℝ

/-- Represents the "rhino's horn segment" region -/
structure RhinoHornSegment where
  largeCircle : Circle
  smallCircle : Circle
  basePoint : Point
  endPoint : Point

/-- Calculates the area of the "rhino's horn segment" -/
def rhinoHornSegmentArea (r : RhinoHornSegment) : ℝ :=
  sorry

/-- The main theorem stating that the area of the "rhino's horn segment" is 2π -/
theorem rhino_horn_segment_area :
  let r := RhinoHornSegment.mk
    (Circle.mk (Point.mk 0 0) 4)
    (Circle.mk (Point.mk 0 2) 2)
    (Point.mk 0 0)
    (Point.mk 4 0)
  rhinoHornSegmentArea r = 2 * Real.pi := by sorry

end rhino_horn_segment_area_l2725_272592


namespace percentage_not_participating_l2725_272538

theorem percentage_not_participating (total_students : ℕ) (music_and_sports : ℕ) (music_only : ℕ) (sports_only : ℕ) :
  total_students = 50 →
  music_and_sports = 5 →
  music_only = 15 →
  sports_only = 20 →
  (total_students - (music_and_sports + music_only + sports_only)) / total_students * 100 = 20 := by
  sorry

end percentage_not_participating_l2725_272538


namespace sum_of_a_and_b_is_eleven_l2725_272529

-- Define the equation that x satisfies
def satisfies_equation (x : ℝ) : Prop :=
  x^2 + 4*x + 4/x + 1/x^2 = 34

-- Define the condition that x can be written as a + √b
def can_be_written_as_a_plus_sqrt_b (x : ℝ) : Prop :=
  ∃ (a b : ℕ), x = a + Real.sqrt b ∧ a > 0 ∧ b > 0

-- State the theorem
theorem sum_of_a_and_b_is_eleven :
  ∀ x : ℝ, satisfies_equation x → can_be_written_as_a_plus_sqrt_b x →
  ∃ (a b : ℕ), x = a + Real.sqrt b ∧ a > 0 ∧ b > 0 ∧ a + b = 11 :=
sorry

end sum_of_a_and_b_is_eleven_l2725_272529


namespace peppers_total_weight_l2725_272503

theorem peppers_total_weight (green : Real) (red : Real) (yellow : Real) (jalapeno : Real) (habanero : Real)
  (h1 : green = 1.45)
  (h2 : red = 0.68)
  (h3 : yellow = 1.6)
  (h4 : jalapeno = 2.25)
  (h5 : habanero = 3.2) :
  green + red + yellow + jalapeno + habanero = 9.18 := by
  sorry

end peppers_total_weight_l2725_272503


namespace total_cost_of_books_l2725_272519

-- Define the number of books for each category
def animal_books : ℕ := 10
def space_books : ℕ := 1
def train_books : ℕ := 3

-- Define the cost per book
def cost_per_book : ℕ := 16

-- Define the total number of books
def total_books : ℕ := animal_books + space_books + train_books

-- Theorem to prove
theorem total_cost_of_books : total_books * cost_per_book = 224 := by
  sorry

end total_cost_of_books_l2725_272519


namespace condition_for_a_greater_than_b_l2725_272593

-- Define the property of being sufficient but not necessary
def sufficient_but_not_necessary (P Q : Prop) : Prop :=
  (P → Q) ∧ ¬(Q → P)

-- Theorem statement
theorem condition_for_a_greater_than_b (a b : ℝ) :
  sufficient_but_not_necessary (a > b + 1) (a > b) := by
  sorry

end condition_for_a_greater_than_b_l2725_272593


namespace caravan_keepers_caravan_keepers_proof_l2725_272553

theorem caravan_keepers : ℕ → Prop :=
  fun k =>
    let hens : ℕ := 50
    let goats : ℕ := 45
    let camels : ℕ := 8
    let total_heads : ℕ := hens + goats + camels + k
    let total_feet : ℕ := hens * 2 + goats * 4 + camels * 4 + k * 2
    total_feet = total_heads + 224 → k = 15

-- The proof goes here
theorem caravan_keepers_proof : ∃ k : ℕ, caravan_keepers k :=
  sorry

end caravan_keepers_caravan_keepers_proof_l2725_272553


namespace second_number_value_l2725_272584

theorem second_number_value (A B C : ℚ) 
  (sum_eq : A + B + C = 98)
  (ratio_AB : A = (2/3) * B)
  (ratio_BC : C = (8/5) * B) : 
  B = 30 := by
sorry

end second_number_value_l2725_272584


namespace danny_bottle_caps_l2725_272563

def initial_bottle_caps (current : ℕ) (lost : ℕ) : ℕ := current + lost

theorem danny_bottle_caps : initial_bottle_caps 25 66 = 91 := by
  sorry

end danny_bottle_caps_l2725_272563


namespace least_number_for_divisibility_l2725_272510

theorem least_number_for_divisibility (n m : ℕ) (hn : n = 1056) (hm : m = 23) :
  ∃ k : ℕ, k > 0 ∧ k ≤ m ∧ (n + k) % m = 0 ∧ ∀ j : ℕ, 0 < j ∧ j < k → (n + j) % m ≠ 0 :=
sorry

end least_number_for_divisibility_l2725_272510


namespace quadratic_inequality_solution_set_l2725_272552

theorem quadratic_inequality_solution_set :
  ∀ x : ℝ, x^2 + x - 12 ≥ 0 ↔ x ≤ -4 ∨ x ≥ 3 := by sorry

end quadratic_inequality_solution_set_l2725_272552


namespace symmetric_circle_l2725_272585

/-- Given a circle C with equation x^2 + y^2 = 25 and a point of symmetry (-3, 4),
    the symmetric circle has the equation (x + 6)^2 + (y - 8)^2 = 25 -/
theorem symmetric_circle (x y : ℝ) :
  (∀ x y, x^2 + y^2 = 25 → (x + 6)^2 + (y - 8)^2 = 25) ∧
  (∃ x₀ y₀, x₀^2 + y₀^2 = 25 ∧ 
    2 * (-3) = x₀ + (x₀ - 6) ∧
    2 * 4 = y₀ + (y₀ - (-8))) :=
by sorry

end symmetric_circle_l2725_272585


namespace company_fund_problem_l2725_272533

theorem company_fund_problem (n : ℕ) (initial_fund : ℕ) : 
  (initial_fund = 60 * n - 10) →  -- The fund was $10 short of giving $60 to each employee
  (initial_fund = 50 * n + 150) → -- After giving $50 to each employee, $150 remained
  initial_fund = 950 := by
  sorry

end company_fund_problem_l2725_272533


namespace acid_dilution_l2725_272500

/-- Given a p% solution of acid with volume p ounces (where p > 45),
    adding y ounces of water to create a (2p/3)% solution results in y = p/2 -/
theorem acid_dilution (p : ℝ) (y : ℝ) (h₁ : p > 45) :
  (p^2 / 100 = (2 * p / 300) * (p + y)) → y = p / 2 := by
  sorry

end acid_dilution_l2725_272500


namespace riverview_village_l2725_272577

theorem riverview_village (p h s c d : ℕ) : 
  p = 4 * h → 
  s = 5 * c → 
  d = 4 * p → 
  ¬∃ (h c : ℕ), 52 = 21 * h + 6 * c :=
by sorry

end riverview_village_l2725_272577


namespace ratio_of_x_to_y_l2725_272598

theorem ratio_of_x_to_y (x y : ℝ) (h1 : x > 0) (h2 : y > 0) (h3 : x * y = 9) (h4 : y = 0.5) :
  x / y = 36 := by
  sorry

end ratio_of_x_to_y_l2725_272598


namespace toby_photo_shoot_l2725_272524

/-- The number of photos Toby took in the photo shoot -/
def photos_in_shoot (initial : ℕ) (deleted_bad : ℕ) (cat_pics : ℕ) (deleted_after : ℕ) (final : ℕ) : ℕ :=
  final - (initial - deleted_bad + cat_pics - deleted_after)

theorem toby_photo_shoot :
  photos_in_shoot 63 7 15 3 84 = 16 := by
  sorry

end toby_photo_shoot_l2725_272524


namespace imaginary_part_of_z_plus_inverse_l2725_272590

def z : ℂ := 3 + Complex.I

theorem imaginary_part_of_z_plus_inverse (z : ℂ) (h : z = 3 + Complex.I) :
  Complex.im (z + z⁻¹) = 9 / 10 := by sorry

end imaginary_part_of_z_plus_inverse_l2725_272590


namespace ryan_has_twenty_more_l2725_272522

/-- The number of stickers each person has -/
structure StickerCount where
  karl : ℕ
  ryan : ℕ
  ben : ℕ

/-- The conditions of the sticker problem -/
def StickerProblem (s : StickerCount) : Prop :=
  s.karl = 25 ∧
  s.ryan > s.karl ∧
  s.ben = s.ryan - 10 ∧
  s.karl + s.ryan + s.ben = 105

/-- The theorem stating Ryan has 20 more stickers than Karl -/
theorem ryan_has_twenty_more (s : StickerCount) 
  (h : StickerProblem s) : s.ryan - s.karl = 20 := by
  sorry

#check ryan_has_twenty_more

end ryan_has_twenty_more_l2725_272522


namespace negation_of_proposition_l2725_272535

theorem negation_of_proposition :
  (¬ ∀ x : ℝ, x ≥ 1 → x^2 - 1 < 0) ↔ (∃ x : ℝ, x ≥ 1 ∧ x^2 - 1 ≥ 0) :=
by sorry

end negation_of_proposition_l2725_272535


namespace base8_calculation_l2725_272528

-- Define a function to convert base 8 to natural numbers
def base8ToNat (x : ℕ) : ℕ := sorry

-- Define a function to convert natural numbers to base 8
def natToBase8 (x : ℕ) : ℕ := sorry

-- Theorem statement
theorem base8_calculation : 
  natToBase8 ((base8ToNat 452 - base8ToNat 126) + base8ToNat 237) = 603 := by
  sorry

end base8_calculation_l2725_272528


namespace tangent_line_implies_sum_l2725_272544

/-- A cubic function with parameters a and b -/
def f (a b x : ℝ) : ℝ := x^3 - x^2 - a*x + b

/-- The derivative of f with respect to x -/
def f' (a x : ℝ) : ℝ := 3*x^2 - 2*x - a

theorem tangent_line_implies_sum (a b : ℝ) :
  f a b 0 = 1 ∧ f' a 0 = 2 → a + b = -1 := by sorry

end tangent_line_implies_sum_l2725_272544


namespace halloween_candies_l2725_272581

/-- The total number of candies collected by a group of friends on Halloween. -/
def total_candies (bob : ℕ) (mary : ℕ) (john : ℕ) (sue : ℕ) (sam : ℕ) : ℕ :=
  bob + mary + john + sue + sam

/-- Theorem stating that the total number of candies collected by the friends is 50. -/
theorem halloween_candies : total_candies 10 5 5 20 10 = 50 := by
  sorry

end halloween_candies_l2725_272581


namespace alternating_sequence_sum_l2725_272570

def sequence_sum (first last : ℤ) (step : ℤ) : ℤ :=
  let n := (last - first) / step + 1
  let sum := (first + last) * n / 2
  if n % 2 = 0 then -sum else sum

theorem alternating_sequence_sum : 
  sequence_sum 2 74 4 = 38 := by sorry

end alternating_sequence_sum_l2725_272570


namespace odd_expression_proof_l2725_272549

theorem odd_expression_proof (p q : ℕ) (hp : Odd p) (hq : Odd q) (hp_pos : p > 0) (hq_pos : q > 0) :
  Odd (4 * p^2 + 2 * q^2 + 1) := by
  sorry

end odd_expression_proof_l2725_272549


namespace first_interest_rate_is_ten_percent_l2725_272502

/-- Calculates the interest rate for the first part of an investment given the total amount,
    the amount in the first part, the interest rate for the second part, and the total profit. -/
def calculate_first_interest_rate (total_amount : ℕ) (first_part : ℕ) (second_interest_rate : ℕ) (total_profit : ℕ) : ℚ :=
  let second_part := total_amount - first_part
  let second_part_profit := (second_part * second_interest_rate) / 100
  let first_part_profit := total_profit - second_part_profit
  (first_part_profit * 100) / first_part

theorem first_interest_rate_is_ten_percent :
  calculate_first_interest_rate 80000 70000 20 9000 = 10 := by
  sorry

end first_interest_rate_is_ten_percent_l2725_272502
