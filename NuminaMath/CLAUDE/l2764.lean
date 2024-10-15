import Mathlib

namespace NUMINAMATH_CALUDE_complex_modulus_l2764_276478

theorem complex_modulus (z : ℂ) (i : ℂ) (h : i * i = -1) (eq : z / (1 + i) = 2 * i) : 
  Complex.abs z = 2 * Real.sqrt 2 := by
sorry

end NUMINAMATH_CALUDE_complex_modulus_l2764_276478


namespace NUMINAMATH_CALUDE_election_win_probability_l2764_276466

/-- Represents the state of an election --/
structure ElectionState :=
  (total_voters : ℕ)
  (votes_a : ℕ)
  (votes_b : ℕ)

/-- Calculates the probability of candidate A winning given the current state --/
noncomputable def win_probability (state : ElectionState) : ℚ :=
  sorry

/-- The main theorem stating the probability of the initially leading candidate winning --/
theorem election_win_probability :
  let initial_state : ElectionState := ⟨2019, 2, 1⟩
  win_probability initial_state = 1513 / 2017 :=
sorry

end NUMINAMATH_CALUDE_election_win_probability_l2764_276466


namespace NUMINAMATH_CALUDE_stable_yield_promotion_l2764_276483

/-- Represents a type of red rice -/
structure RedRice where
  typeName : String
  averageYield : ℝ
  variance : ℝ

/-- Determines if a type of red rice is suitable for promotion based on yield stability -/
def isSuitableForPromotion (rice1 rice2 : RedRice) : Prop :=
  rice1.averageYield = rice2.averageYield ∧ 
  rice1.variance < rice2.variance

theorem stable_yield_promotion (A B : RedRice) 
  (h_yield : A.averageYield = B.averageYield)
  (h_variance : A.variance < B.variance) : 
  isSuitableForPromotion A B := by
  sorry

#check stable_yield_promotion

end NUMINAMATH_CALUDE_stable_yield_promotion_l2764_276483


namespace NUMINAMATH_CALUDE_grazing_area_fence_posts_l2764_276413

/-- Calculates the number of fence posts needed for a rectangular grazing area -/
def fencePostsRequired (length width postSpacing : ℕ) : ℕ :=
  let longSide := max length width
  let shortSide := min length width
  let longSidePosts := longSide / postSpacing + 1
  let shortSidePosts := (shortSide / postSpacing + 1) * 2 - 2
  longSidePosts + shortSidePosts

/-- The problem statement -/
theorem grazing_area_fence_posts :
  fencePostsRequired 70 50 10 = 18 := by
  sorry


end NUMINAMATH_CALUDE_grazing_area_fence_posts_l2764_276413


namespace NUMINAMATH_CALUDE_smallest_whole_number_larger_than_sum_l2764_276460

def mixed_to_fraction (whole : ℤ) (num : ℕ) (denom : ℕ) : ℚ :=
  whole + (num : ℚ) / (denom : ℚ)

def sum_of_mixed_numbers : ℚ :=
  mixed_to_fraction 1 2 3 +
  mixed_to_fraction 2 1 4 +
  mixed_to_fraction 3 3 8 +
  mixed_to_fraction 4 1 6

theorem smallest_whole_number_larger_than_sum :
  (⌈sum_of_mixed_numbers⌉ : ℤ) = 12 := by sorry

end NUMINAMATH_CALUDE_smallest_whole_number_larger_than_sum_l2764_276460


namespace NUMINAMATH_CALUDE_probability_of_four_boys_l2764_276435

open BigOperators Finset

theorem probability_of_four_boys (total_students : ℕ) (total_boys : ℕ) (selected_students : ℕ) :
  total_students = 15 →
  total_boys = 7 →
  selected_students = 10 →
  (Nat.choose total_boys 4 * Nat.choose (total_students - total_boys) (selected_students - 4)) /
  Nat.choose total_students selected_students =
  Nat.choose 7 4 * Nat.choose 8 6 / Nat.choose 15 10 :=
by sorry

end NUMINAMATH_CALUDE_probability_of_four_boys_l2764_276435


namespace NUMINAMATH_CALUDE_donut_selections_l2764_276487

theorem donut_selections (n : ℕ) (k : ℕ) (h1 : n = 6) (h2 : k = 4) :
  Nat.choose (n + k - 1) (k - 1) = 84 := by
  sorry

end NUMINAMATH_CALUDE_donut_selections_l2764_276487


namespace NUMINAMATH_CALUDE_prime_divisibility_l2764_276441

theorem prime_divisibility (p a b : ℤ) : 
  Prime p → 
  ∃ k : ℤ, p = 4 * k + 3 → 
  p ∣ (a^2 + b^2) → 
  p ∣ a ∧ p ∣ b := by
  sorry

end NUMINAMATH_CALUDE_prime_divisibility_l2764_276441


namespace NUMINAMATH_CALUDE_system_solution_condition_l2764_276434

theorem system_solution_condition (n p : ℕ) :
  (∃ x y : ℕ+, x + p * y = n ∧ x + y = p^2) ↔
  (p > 1 ∧ (p - 1) ∣ (n - 1) ∧ ∀ k : ℕ+, n ≠ p^(k : ℕ)) :=
sorry

end NUMINAMATH_CALUDE_system_solution_condition_l2764_276434


namespace NUMINAMATH_CALUDE_no_natural_squares_diff_2014_l2764_276448

theorem no_natural_squares_diff_2014 : ¬∃ (m n : ℕ), m^2 = n^2 + 2014 := by
  sorry

end NUMINAMATH_CALUDE_no_natural_squares_diff_2014_l2764_276448


namespace NUMINAMATH_CALUDE_furniture_markup_proof_l2764_276427

/-- Calculates the percentage markup given the selling price and cost price -/
def percentage_markup (selling_price cost_price : ℚ) : ℚ :=
  ((selling_price - cost_price) / cost_price) * 100

/-- Proves that the percentage markup is 25% for the given selling and cost prices -/
theorem furniture_markup_proof (selling_price cost_price : ℚ) 
  (h1 : selling_price = 4800)
  (h2 : cost_price = 3840) : 
  percentage_markup selling_price cost_price = 25 := by
  sorry

end NUMINAMATH_CALUDE_furniture_markup_proof_l2764_276427


namespace NUMINAMATH_CALUDE_gold_silver_weight_problem_l2764_276429

theorem gold_silver_weight_problem (x y : ℝ) : 
  (9 * x = 11 * y) ∧ ((10 * y + x) - (8 * x + y) = 13) ↔ 
  (9 * x = 11 * y ∧ 
   ∃ (gold_bag silver_bag : ℝ),
     gold_bag = 9 * x ∧
     silver_bag = 11 * y ∧
     gold_bag = silver_bag ∧
     (silver_bag + x - y) - (gold_bag - x + y) = 13) :=
by sorry

end NUMINAMATH_CALUDE_gold_silver_weight_problem_l2764_276429


namespace NUMINAMATH_CALUDE_moving_circle_trajectory_l2764_276473

-- Define the fixed point M
def M : ℝ × ℝ := (-4, 0)

-- Define the equation of the known circle N
def circle_N (x y : ℝ) : Prop := (x - 4)^2 + y^2 = 16

-- Define the trajectory equation
def trajectory (x y : ℝ) : Prop := x^2/4 - y^2/12 = 1

-- Theorem statement
theorem moving_circle_trajectory :
  ∀ (x y : ℝ),
    (∃ (r : ℝ), 
      -- The moving circle passes through M
      (x + 4)^2 + y^2 = r^2 ∧
      -- The moving circle is tangent to N
      ∃ (x_n y_n : ℝ), circle_N x_n y_n ∧ (x - x_n)^2 + (y - y_n)^2 = r^2) →
    trajectory x y :=
sorry

end NUMINAMATH_CALUDE_moving_circle_trajectory_l2764_276473


namespace NUMINAMATH_CALUDE_degree_of_composed_product_l2764_276479

/-- Given polynomials f and g with degrees 3 and 6 respectively,
    the degree of f(x^2) · g(x^3) is 24. -/
theorem degree_of_composed_product (f g : Polynomial ℝ) 
  (hf : Polynomial.degree f = 3)
  (hg : Polynomial.degree g = 6) :
  Polynomial.degree (f.comp (Polynomial.X ^ 2) * g.comp (Polynomial.X ^ 3)) = 24 := by
  sorry

end NUMINAMATH_CALUDE_degree_of_composed_product_l2764_276479


namespace NUMINAMATH_CALUDE_pencil_sharpening_mean_l2764_276456

def pencil_sharpening_data : List ℕ := [13, 8, 13, 21, 7, 23, 15, 19, 12, 9, 28, 6, 17, 29, 31, 10, 4, 20, 16, 12, 2, 18, 27, 22, 5, 14, 31, 29, 8, 25]

theorem pencil_sharpening_mean :
  (pencil_sharpening_data.sum : ℚ) / pencil_sharpening_data.length = 543 / 30 := by
  sorry

end NUMINAMATH_CALUDE_pencil_sharpening_mean_l2764_276456


namespace NUMINAMATH_CALUDE_number_with_specific_remainders_l2764_276454

theorem number_with_specific_remainders : ∃ n : ℕ, 
  (∀ k : ℕ, 2 ≤ k → k ≤ 10 → n % k = k - 1) ∧ n = 2519 := by
  sorry

end NUMINAMATH_CALUDE_number_with_specific_remainders_l2764_276454


namespace NUMINAMATH_CALUDE_quadratic_points_relation_l2764_276422

theorem quadratic_points_relation (c : ℝ) (y₁ y₂ y₃ : ℝ) : 
  let f : ℝ → ℝ := λ x ↦ x^2 + 2*x + c
  (f (-3) = y₁) → (f (1/2) = y₂) → (f 2 = y₃) → 
  (y₂ < y₁) ∧ (y₁ < y₃) := by
sorry

end NUMINAMATH_CALUDE_quadratic_points_relation_l2764_276422


namespace NUMINAMATH_CALUDE_tv_sets_in_shop_a_l2764_276494

/-- The number of electronic shops in the Naza market -/
def num_shops : ℕ := 5

/-- The average number of TV sets in each shop -/
def average_tv_sets : ℕ := 48

/-- The number of TV sets in shop b -/
def tv_sets_b : ℕ := 30

/-- The number of TV sets in shop c -/
def tv_sets_c : ℕ := 60

/-- The number of TV sets in shop d -/
def tv_sets_d : ℕ := 80

/-- The number of TV sets in shop e -/
def tv_sets_e : ℕ := 50

/-- Theorem: Given the conditions, shop a must have 20 TV sets -/
theorem tv_sets_in_shop_a : 
  (num_shops * average_tv_sets) - (tv_sets_b + tv_sets_c + tv_sets_d + tv_sets_e) = 20 := by
  sorry

end NUMINAMATH_CALUDE_tv_sets_in_shop_a_l2764_276494


namespace NUMINAMATH_CALUDE_lateral_surface_area_regular_triangular_prism_l2764_276420

/-- Given a regular triangular prism with height h, where a line passing through 
    the center of the upper base and the midpoint of the side of the lower base 
    is inclined at an angle 60° to the plane of the base, 
    the lateral surface area of the prism is 6h². -/
theorem lateral_surface_area_regular_triangular_prism 
  (h : ℝ) 
  (h_pos : h > 0) 
  (incline_angle : ℝ) 
  (incline_angle_eq : incline_angle = 60 * π / 180) : 
  ∃ (S : ℝ), S = 6 * h^2 ∧ S > 0 := by
  sorry

end NUMINAMATH_CALUDE_lateral_surface_area_regular_triangular_prism_l2764_276420


namespace NUMINAMATH_CALUDE_line_through_point_l2764_276407

/-- 
Given a line with equation -1/3 - 3kx = 4y that passes through the point (1/3, -8),
prove that k = 95/3.
-/
theorem line_through_point (k : ℚ) : 
  (-1/3 : ℚ) - 3 * k * (1/3 : ℚ) = 4 * (-8 : ℚ) → k = 95/3 := by
  sorry

end NUMINAMATH_CALUDE_line_through_point_l2764_276407


namespace NUMINAMATH_CALUDE_second_person_speed_l2764_276497

/-- Given two persons starting at the same point, walking in opposite directions
    for 3.5 hours, with one person walking at 6 km/hr, and ending up 45.5 km apart,
    the speed of the second person is 7 km/hr. -/
theorem second_person_speed (person1_speed : ℝ) (person2_speed : ℝ) (time : ℝ) (distance : ℝ) :
  person1_speed = 6 →
  time = 3.5 →
  distance = 45.5 →
  distance = (person1_speed + person2_speed) * time →
  person2_speed = 7 := by
  sorry

end NUMINAMATH_CALUDE_second_person_speed_l2764_276497


namespace NUMINAMATH_CALUDE_toothpicks_15th_stage_l2764_276426

/-- The number of toothpicks in the nth stage of the pattern -/
def toothpicks (n : ℕ) : ℕ :=
  3 + 2 * (n - 1)

/-- Theorem stating that the 15th stage of the pattern has 31 toothpicks -/
theorem toothpicks_15th_stage :
  toothpicks 15 = 31 := by
  sorry

end NUMINAMATH_CALUDE_toothpicks_15th_stage_l2764_276426


namespace NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l2764_276465

theorem quadratic_inequality_solution_set :
  {x : ℝ | x^2 < 2*x} = Set.Ioo 0 2 := by sorry

end NUMINAMATH_CALUDE_quadratic_inequality_solution_set_l2764_276465


namespace NUMINAMATH_CALUDE_relatively_prime_2n_plus_1_and_4n_squared_plus_1_l2764_276408

theorem relatively_prime_2n_plus_1_and_4n_squared_plus_1 (n : ℕ+) :
  Nat.gcd (2 * n.val + 1) (4 * n.val^2 + 1) = 1 := by
  sorry

end NUMINAMATH_CALUDE_relatively_prime_2n_plus_1_and_4n_squared_plus_1_l2764_276408


namespace NUMINAMATH_CALUDE_mary_change_theorem_l2764_276450

-- Define the ticket prices and discounts
def adult_price : ℚ := 2
def child_price : ℚ := 1
def first_child_discount : ℚ := 0.5
def second_child_discount : ℚ := 0.75
def third_child_discount : ℚ := 1
def sales_tax_rate : ℚ := 0.08
def amount_paid : ℚ := 20

-- Calculate the total cost before tax
def total_cost_before_tax : ℚ :=
  adult_price +
  child_price * first_child_discount +
  child_price * second_child_discount +
  child_price * third_child_discount

-- Calculate the sales tax
def sales_tax : ℚ := total_cost_before_tax * sales_tax_rate

-- Calculate the total cost including tax
def total_cost_with_tax : ℚ := total_cost_before_tax + sales_tax

-- Calculate the change
def change : ℚ := amount_paid - total_cost_with_tax

-- Theorem to prove
theorem mary_change_theorem : change = 15.41 := by sorry

end NUMINAMATH_CALUDE_mary_change_theorem_l2764_276450


namespace NUMINAMATH_CALUDE_symmetry_axes_symmetry_origin_l2764_276444

-- Define the curve C
def C (x y : ℝ) : Prop :=
  ((x + 1)^2 + y^2) * ((x - 1)^2 + y^2) = 2

-- Theorem for symmetry with respect to axes
theorem symmetry_axes :
  (∀ x y, C x y ↔ C (-x) y) ∧ (∀ x y, C x y ↔ C x (-y)) :=
sorry

-- Theorem for symmetry with respect to origin
theorem symmetry_origin :
  ∀ x y, C x y ↔ C (-x) (-y) :=
sorry

end NUMINAMATH_CALUDE_symmetry_axes_symmetry_origin_l2764_276444


namespace NUMINAMATH_CALUDE_total_glasses_at_restaurant_l2764_276495

/-- Represents the number of glasses in a small box -/
def small_box : ℕ := 12

/-- Represents the number of glasses in a large box -/
def large_box : ℕ := 16

/-- Represents the difference in the number of large boxes compared to small boxes -/
def box_difference : ℕ := 16

/-- Represents the average number of glasses per box -/
def average_glasses : ℕ := 15

theorem total_glasses_at_restaurant :
  ∃ (small_boxes large_boxes : ℕ),
    large_boxes = small_boxes + box_difference ∧
    (small_box * small_boxes + large_box * large_boxes) / (small_boxes + large_boxes) = average_glasses ∧
    small_box * small_boxes + large_box * large_boxes = 480 :=
sorry

end NUMINAMATH_CALUDE_total_glasses_at_restaurant_l2764_276495


namespace NUMINAMATH_CALUDE_no_valid_grid_l2764_276447

/-- Represents a 3x3 grid with elements from 1 to 4 -/
def Grid := Fin 3 → Fin 3 → Fin 4

/-- Checks if all elements in a list are distinct -/
def allDistinct (l : List (Fin 4)) : Prop :=
  l.Nodup

/-- Checks if a row in the grid contains distinct elements -/
def rowDistinct (g : Grid) (i : Fin 3) : Prop :=
  allDistinct [g i 0, g i 1, g i 2]

/-- Checks if a column in the grid contains distinct elements -/
def colDistinct (g : Grid) (j : Fin 3) : Prop :=
  allDistinct [g 0 j, g 1 j, g 2 j]

/-- Checks if the main diagonal contains distinct elements -/
def mainDiagDistinct (g : Grid) : Prop :=
  allDistinct [g 0 0, g 1 1, g 2 2]

/-- Checks if the anti-diagonal contains distinct elements -/
def antiDiagDistinct (g : Grid) : Prop :=
  allDistinct [g 0 2, g 1 1, g 2 0]

/-- A grid is valid if all rows, columns, and diagonals contain distinct elements -/
def validGrid (g : Grid) : Prop :=
  (∀ i, rowDistinct g i) ∧
  (∀ j, colDistinct g j) ∧
  mainDiagDistinct g ∧
  antiDiagDistinct g

theorem no_valid_grid : ¬∃ g : Grid, validGrid g := by
  sorry

end NUMINAMATH_CALUDE_no_valid_grid_l2764_276447


namespace NUMINAMATH_CALUDE_mei_fruit_baskets_l2764_276425

theorem mei_fruit_baskets : Nat.gcd 15 (Nat.gcd 9 18) = 3 := by
  sorry

end NUMINAMATH_CALUDE_mei_fruit_baskets_l2764_276425


namespace NUMINAMATH_CALUDE_no_integer_solution_l2764_276432

theorem no_integer_solution : ∀ (x y z : ℤ), x ≠ 0 → 2*x^4 + 2*x^2*y^2 + y^4 ≠ z^2 := by
  sorry

end NUMINAMATH_CALUDE_no_integer_solution_l2764_276432


namespace NUMINAMATH_CALUDE_not_always_prime_l2764_276467

def P (n : ℤ) : ℤ := n^2 + n + 41

theorem not_always_prime : ∃ n : ℤ, ¬(Nat.Prime (Int.natAbs (P n))) := by
  sorry

end NUMINAMATH_CALUDE_not_always_prime_l2764_276467


namespace NUMINAMATH_CALUDE_instantaneous_velocity_at_3_seconds_l2764_276414

-- Define the displacement function
def s (t : ℝ) : ℝ := 4 - 2*t + t^2

-- Define the velocity function (derivative of s)
def v (t : ℝ) : ℝ := 2*t - 2

-- Theorem statement
theorem instantaneous_velocity_at_3_seconds :
  v 3 = 4 := by sorry

end NUMINAMATH_CALUDE_instantaneous_velocity_at_3_seconds_l2764_276414


namespace NUMINAMATH_CALUDE_sally_peaches_l2764_276451

/-- The number of peaches Sally picked from the orchard -/
def peaches_picked : ℕ := 55

/-- The total number of peaches after picking -/
def total_peaches : ℕ := 68

/-- The initial number of peaches Sally had -/
def initial_peaches : ℕ := total_peaches - peaches_picked

theorem sally_peaches : initial_peaches + peaches_picked = total_peaches := by
  sorry

end NUMINAMATH_CALUDE_sally_peaches_l2764_276451


namespace NUMINAMATH_CALUDE_units_digit_power_four_l2764_276471

theorem units_digit_power_four (a : ℤ) (n : ℕ) : 
  10 ∣ (a^(n+4) - a^n) := by
  sorry

end NUMINAMATH_CALUDE_units_digit_power_four_l2764_276471


namespace NUMINAMATH_CALUDE_cube_sum_equality_l2764_276417

theorem cube_sum_equality (a b c : ℝ) : 
  a ≠ b ∧ b ≠ c ∧ a ≠ c →
  (a^3 + 10) / a^2 = (b^3 + 10) / b^2 ∧
  (b^3 + 10) / b^2 = (c^3 + 10) / c^2 →
  a^3 + b^3 + c^3 = 1301 := by
  sorry

end NUMINAMATH_CALUDE_cube_sum_equality_l2764_276417


namespace NUMINAMATH_CALUDE_garden_area_calculation_l2764_276457

/-- The total area of Mancino's and Marquita's gardens -/
def total_garden_area (mancino_garden_length mancino_garden_width mancino_garden_count
                       marquita_garden_length marquita_garden_width marquita_garden_count : ℕ) : ℕ :=
  (mancino_garden_length * mancino_garden_width * mancino_garden_count) +
  (marquita_garden_length * marquita_garden_width * marquita_garden_count)

/-- Theorem stating that the total area of Mancino's and Marquita's gardens is 304 square feet -/
theorem garden_area_calculation :
  total_garden_area 16 5 3 8 4 2 = 304 := by
  sorry

end NUMINAMATH_CALUDE_garden_area_calculation_l2764_276457


namespace NUMINAMATH_CALUDE_range_of_x_l2764_276493

-- Define the determinant operation
def det (a b c d : ℝ) : ℝ := a * d - b * c

-- Define the theorem
theorem range_of_x (x : ℝ) : 
  det x 3 (-x) x < det 2 0 1 2 → -4 < x ∧ x < 1 := by
  sorry

end NUMINAMATH_CALUDE_range_of_x_l2764_276493


namespace NUMINAMATH_CALUDE_symmetry_about_x_equals_one_l2764_276458

-- Define a function f over the reals
variable (f : ℝ → ℝ)

-- State the theorem
theorem symmetry_about_x_equals_one (x : ℝ) : f (x - 1) = f (-(x - 2) + 1) := by
  sorry

end NUMINAMATH_CALUDE_symmetry_about_x_equals_one_l2764_276458


namespace NUMINAMATH_CALUDE_movie_length_after_cut_l2764_276470

theorem movie_length_after_cut (final_length cut_length : ℕ) (h1 : final_length = 57) (h2 : cut_length = 3) :
  final_length + cut_length = 60 := by
  sorry

end NUMINAMATH_CALUDE_movie_length_after_cut_l2764_276470


namespace NUMINAMATH_CALUDE_average_children_in_families_with_children_l2764_276428

theorem average_children_in_families_with_children 
  (total_families : ℕ) 
  (average_children : ℚ) 
  (childless_families : ℕ) 
  (h1 : total_families = 15)
  (h2 : average_children = 3)
  (h3 : childless_families = 3) :
  (total_families * average_children) / (total_families - childless_families) = 3.75 := by
sorry

end NUMINAMATH_CALUDE_average_children_in_families_with_children_l2764_276428


namespace NUMINAMATH_CALUDE_g_243_equals_118_l2764_276485

/-- A function g with the property that g(a) + g(b) = m^3 when a + b = 3^m -/
def g_property (g : ℕ → ℝ) : Prop :=
  ∀ (a b m : ℕ), a > 0 → b > 0 → m > 0 → a + b = 3^m → g a + g b = (m : ℝ)^3

/-- The main theorem stating that g(243) = 118 -/
theorem g_243_equals_118 (g : ℕ → ℝ) (h : g_property g) : g 243 = 118 := by
  sorry


end NUMINAMATH_CALUDE_g_243_equals_118_l2764_276485


namespace NUMINAMATH_CALUDE_smallest_n_with_five_pairs_l2764_276496

/-- The function f(n) returns the number of distinct ordered pairs of positive integers (a, b) such that a² + b² = n -/
def f (n : ℕ) : ℕ := (Finset.filter (fun p : ℕ × ℕ => p.1 * p.1 + p.2 * p.2 = n ∧ p.1 > 0 ∧ p.2 > 0) (Finset.product (Finset.range n) (Finset.range n))).card

/-- 125 is the smallest positive integer n for which f(n) = 5 -/
theorem smallest_n_with_five_pairs : (∀ m : ℕ, m > 0 ∧ m < 125 → f m ≠ 5) ∧ f 125 = 5 := by
  sorry

end NUMINAMATH_CALUDE_smallest_n_with_five_pairs_l2764_276496


namespace NUMINAMATH_CALUDE_intersection_point_is_unique_l2764_276421

/-- The intersection point of two lines -/
def intersection_point : ℚ × ℚ := (-3/8, 17/8)

/-- The first line equation -/
def line1 (x y : ℚ) : Prop := y = -3 * x + 1

/-- The second line equation -/
def line2 (x y : ℚ) : Prop := y = 5 * x + 4

theorem intersection_point_is_unique :
  let (x, y) := intersection_point
  (line1 x y ∧ line2 x y) ∧
  ∀ x' y', (line1 x' y' ∧ line2 x' y') → (x' = x ∧ y' = y) :=
by sorry

end NUMINAMATH_CALUDE_intersection_point_is_unique_l2764_276421


namespace NUMINAMATH_CALUDE_store_purchase_combinations_l2764_276416

/-- The number of oreo flavors available -/
def num_oreo_flavors : ℕ := 6

/-- The number of milk flavors available -/
def num_milk_flavors : ℕ := 4

/-- The total number of items Alpha can choose from -/
def total_items : ℕ := num_oreo_flavors + num_milk_flavors

/-- The number of items they collectively buy -/
def total_purchased : ℕ := 3

/-- Represents the ways Alpha can choose items without repeats -/
def alpha_choices (k : ℕ) : ℕ := Nat.choose total_items k

/-- Represents the ways Beta can choose k oreos with possible repeats -/
def beta_choices (k : ℕ) : ℕ :=
  Nat.choose num_oreo_flavors k +  -- All different
  (if k ≥ 2 then num_oreo_flavors * (num_oreo_flavors - 1) else 0) +  -- Two same, one different (if k ≥ 2)
  (if k = 3 then num_oreo_flavors else 0)  -- All same (if k = 3)

/-- The total number of ways for Alpha and Beta to collectively buy 3 items -/
def total_ways : ℕ :=
  alpha_choices 3 +  -- Alpha buys 3, Beta 0
  alpha_choices 2 * num_oreo_flavors +  -- Alpha buys 2, Beta 1
  alpha_choices 1 * beta_choices 2 +  -- Alpha buys 1, Beta 2
  beta_choices 3  -- Alpha buys 0, Beta 3

theorem store_purchase_combinations :
  total_ways = 656 := by sorry

end NUMINAMATH_CALUDE_store_purchase_combinations_l2764_276416


namespace NUMINAMATH_CALUDE_ellen_dinner_calories_l2764_276409

/-- Calculates the remaining calories for dinner given a daily limit and calories consumed for breakfast, lunch, and snack. -/
def remaining_calories_for_dinner (daily_limit : ℕ) (breakfast : ℕ) (lunch : ℕ) (snack : ℕ) : ℕ :=
  daily_limit - (breakfast + lunch + snack)

/-- Proves that given the specific calorie values in the problem, the remaining calories for dinner is 832. -/
theorem ellen_dinner_calories : 
  remaining_calories_for_dinner 2200 353 885 130 = 832 := by
sorry

end NUMINAMATH_CALUDE_ellen_dinner_calories_l2764_276409


namespace NUMINAMATH_CALUDE_inverse_343_mod_103_l2764_276438

theorem inverse_343_mod_103 (h : (7⁻¹ : ZMod 103) = 44) : (343⁻¹ : ZMod 103) = 3 := by
  sorry

end NUMINAMATH_CALUDE_inverse_343_mod_103_l2764_276438


namespace NUMINAMATH_CALUDE_geometric_progression_sum_change_l2764_276481

/-- Given a geometric progression with 3000 terms, all positive, prove that
    if increasing every third term by 50 times increases the sum by 10 times,
    then doubling every even term increases the sum by 11/8 times. -/
theorem geometric_progression_sum_change (b₁ : ℝ) (q : ℝ) (S : ℝ) : 
  b₁ > 0 ∧ q > 0 ∧ S > 0 →
  S = b₁ * (1 - q^3000) / (1 - q) →
  S + 49 * b₁ * q^2 * (1 - q^3000) / ((1 - q) * (1 + q + q^2)) = 10 * S →
  S + 2 * b₁ * q * (1 - q^3000) / (1 - q^2) = 11 * S / 8 := by
sorry

end NUMINAMATH_CALUDE_geometric_progression_sum_change_l2764_276481


namespace NUMINAMATH_CALUDE_power_sum_equality_l2764_276403

theorem power_sum_equality : (-1)^49 + 2^(4^3 + 3^2 - 7^2) = 16777215 := by
  sorry

end NUMINAMATH_CALUDE_power_sum_equality_l2764_276403


namespace NUMINAMATH_CALUDE_quadratic_function_uniqueness_l2764_276491

theorem quadratic_function_uniqueness (a b c : ℝ) :
  (∀ x : ℝ, a * (x - 4)^2 + b * (x - 4) + c = a * (2 - x)^2 + b * (2 - x) + c) →
  (∀ x : ℝ, a * x^2 + b * x + c ≥ x) →
  (∀ x : ℝ, x > 0 ∧ x < 2 → a * x^2 + b * x + c ≤ ((x + 1) / 2)^2) →
  (∃ x : ℝ, ∀ y : ℝ, a * x^2 + b * x + c ≤ a * y^2 + b * y + c) →
  (∃ x : ℝ, a * x^2 + b * x + c = 0) →
  (∀ x : ℝ, a * x^2 + b * x + c = (1/4) * (x + 1)^2) :=
sorry

end NUMINAMATH_CALUDE_quadratic_function_uniqueness_l2764_276491


namespace NUMINAMATH_CALUDE_fraction_problem_l2764_276433

theorem fraction_problem (N : ℝ) (F : ℝ) : 
  N = 8 → 0.5 * N = F * N + 2 → F = 1/4 := by
  sorry

end NUMINAMATH_CALUDE_fraction_problem_l2764_276433


namespace NUMINAMATH_CALUDE_sum_of_digits_of_seven_to_fifteen_l2764_276477

/-- The sum of the tens digit and the ones digit of (3 + 4)^15 is 7 -/
theorem sum_of_digits_of_seven_to_fifteen (n : ℕ) : n = (3 + 4)^15 → 
  (n / 10 % 10 + n % 10 = 7) := by
sorry

end NUMINAMATH_CALUDE_sum_of_digits_of_seven_to_fifteen_l2764_276477


namespace NUMINAMATH_CALUDE_repeating_decimal_equiv_fraction_l2764_276423

/-- The repeating decimal 0.3̄6 is equal to the fraction 11/30 -/
theorem repeating_decimal_equiv_fraction : 
  (∃ (x : ℚ), x = 0.3 + (6 / 9) / 10 ∧ x = 11 / 30) := by sorry

end NUMINAMATH_CALUDE_repeating_decimal_equiv_fraction_l2764_276423


namespace NUMINAMATH_CALUDE_cubic_equation_solution_l2764_276492

theorem cubic_equation_solution (x : ℝ) : (x + 2)^3 = 64 → x = 2 := by
  sorry

end NUMINAMATH_CALUDE_cubic_equation_solution_l2764_276492


namespace NUMINAMATH_CALUDE_speedster_convertibles_count_l2764_276437

theorem speedster_convertibles_count (total : ℕ) (speedsters : ℕ) (convertibles : ℕ) :
  speedsters = total / 3 →
  30 = total - speedsters →
  convertibles = (4 * speedsters) / 5 →
  convertibles = 12 := by
sorry

end NUMINAMATH_CALUDE_speedster_convertibles_count_l2764_276437


namespace NUMINAMATH_CALUDE_complex_power_2013_l2764_276430

theorem complex_power_2013 : (((1 + Complex.I) / (1 - Complex.I)) ^ 2013 : ℂ) = Complex.I := by sorry

end NUMINAMATH_CALUDE_complex_power_2013_l2764_276430


namespace NUMINAMATH_CALUDE_age_difference_proof_l2764_276401

theorem age_difference_proof : ∃ (a b : ℕ), 
  (a ≥ 10 ∧ a < 100) ∧ 
  (b ≥ 10 ∧ b < 100) ∧ 
  (a / 10 = b % 10) ∧ 
  (a % 10 = b / 10) ∧ 
  (a + 7 = 3 * (b + 7)) ∧ 
  (a - b = 36) := by
sorry

end NUMINAMATH_CALUDE_age_difference_proof_l2764_276401


namespace NUMINAMATH_CALUDE_perpendicular_lines_from_quadratic_l2764_276415

/-- Two lines with slopes that are the roots of x^2 - 3x - 1 = 0 are perpendicular -/
theorem perpendicular_lines_from_quadratic (k₁ k₂ : ℝ) : 
  k₁^2 - 3*k₁ - 1 = 0 → k₂^2 - 3*k₂ - 1 = 0 → k₁ ≠ k₂ → k₁ * k₂ = -1 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_lines_from_quadratic_l2764_276415


namespace NUMINAMATH_CALUDE_eliana_steps_proof_l2764_276404

/-- The number of steps Eliana walked on the first day before adding 300 steps -/
def first_day_steps : ℕ := 200

/-- The total number of steps for all three days -/
def total_steps : ℕ := 1600

theorem eliana_steps_proof :
  first_day_steps + 300 + 2 * (first_day_steps + 300) + 100 = total_steps :=
by sorry

end NUMINAMATH_CALUDE_eliana_steps_proof_l2764_276404


namespace NUMINAMATH_CALUDE_y_value_l2764_276431

theorem y_value (x y : ℝ) (h1 : x^2 = y - 5) (h2 : x = 7) : y = 54 := by
  sorry

end NUMINAMATH_CALUDE_y_value_l2764_276431


namespace NUMINAMATH_CALUDE_largest_mediocre_number_l2764_276484

def is_mediocre (n : ℕ) : Prop :=
  n ≥ 100 ∧ n < 1000 ∧
  let a := n / 100
  let b := (n / 10) % 10
  let c := n % 10
  a ≠ 0 ∧ b ≠ 0 ∧ c ≠ 0 ∧ a ≠ b ∧ b ≠ c ∧ a ≠ c ∧
  n = (100 * a + 10 * b + c +
       100 * a + 10 * c + b +
       100 * b + 10 * a + c +
       100 * b + 10 * c + a +
       100 * c + 10 * a + b +
       100 * c + 10 * b + a) / 6

theorem largest_mediocre_number :
  is_mediocre 629 ∧ ∀ n : ℕ, is_mediocre n → n ≤ 629 :=
sorry

end NUMINAMATH_CALUDE_largest_mediocre_number_l2764_276484


namespace NUMINAMATH_CALUDE_transform_f_to_g_l2764_276482

/-- The original function -/
def f (x : ℝ) : ℝ := (x - 1)^2 + 2

/-- The resulting function after transformation -/
def g (x : ℝ) : ℝ := (x - 5)^2 + 5

/-- Vertical shift transformation -/
def vertical_shift (h : ℝ → ℝ) (k : ℝ) (x : ℝ) : ℝ := h x + k

/-- Horizontal shift transformation -/
def horizontal_shift (h : ℝ → ℝ) (k : ℝ) (x : ℝ) : ℝ := h (x - k)

/-- Theorem stating that the transformation of f results in g -/
theorem transform_f_to_g : 
  ∀ x, horizontal_shift (vertical_shift f 3) 4 x = g x :=
sorry

end NUMINAMATH_CALUDE_transform_f_to_g_l2764_276482


namespace NUMINAMATH_CALUDE_det_A_squared_l2764_276459

def A : Matrix (Fin 2) (Fin 2) ℝ := !![5, 3; 7, 2]

theorem det_A_squared : (Matrix.det A)^2 = 121 := by
  sorry

end NUMINAMATH_CALUDE_det_A_squared_l2764_276459


namespace NUMINAMATH_CALUDE_trigonometric_equation_solution_l2764_276412

theorem trigonometric_equation_solution (x : ℝ) : 
  (Real.sin (3 * x) + Real.sin x - Real.sin (2 * x) = 2 * Real.cos x * (Real.cos x - 1)) ↔ 
  (∃ k : ℤ, x = π / 2 * (2 * k + 1)) ∨ 
  (∃ n : ℤ, x = 2 * π * n) ∨ 
  (∃ l : ℤ, x = π / 4 * (4 * l - 1)) := by
sorry

end NUMINAMATH_CALUDE_trigonometric_equation_solution_l2764_276412


namespace NUMINAMATH_CALUDE_problem_solution_l2764_276489

theorem problem_solution (a b m : ℝ) (h1 : 2^a = m) (h2 : 5^b = m) (h3 : 1/a + 1/b = 1/2) : m = 100 := by
  sorry

end NUMINAMATH_CALUDE_problem_solution_l2764_276489


namespace NUMINAMATH_CALUDE_sum_of_roots_quadratic_sum_of_roots_specific_equation_l2764_276410

theorem sum_of_roots_quadratic (a b c : ℝ) (h : a ≠ 0) :
  let f : ℝ → ℝ := λ x => a * x^2 + b * x + c
  (∃ x y : ℝ, x ≠ y ∧ f x = 0 ∧ f y = 0) →
  (x + y : ℝ) = -b / a :=
sorry

theorem sum_of_roots_specific_equation :
  let f : ℝ → ℝ := λ x => x^2 - 7*x + 2 - 11
  (∃ x y : ℝ, x ≠ y ∧ f x = 0 ∧ f y = 0) →
  (x + y : ℝ) = 7 :=
sorry

end NUMINAMATH_CALUDE_sum_of_roots_quadratic_sum_of_roots_specific_equation_l2764_276410


namespace NUMINAMATH_CALUDE_factorization_cubic_minus_linear_l2764_276462

theorem factorization_cubic_minus_linear (x : ℝ) : x^3 - 4*x = x*(x+2)*(x-2) := by
  sorry

end NUMINAMATH_CALUDE_factorization_cubic_minus_linear_l2764_276462


namespace NUMINAMATH_CALUDE_overlapping_tape_length_l2764_276469

/-- 
Given three tapes of equal length attached with equal overlapping parts,
this theorem proves the length of one overlapping portion.
-/
theorem overlapping_tape_length 
  (tape_length : ℝ) 
  (attached_length : ℝ) 
  (h1 : tape_length = 217) 
  (h2 : attached_length = 627) : 
  (3 * tape_length - attached_length) / 2 = 12 := by
  sorry

#check overlapping_tape_length

end NUMINAMATH_CALUDE_overlapping_tape_length_l2764_276469


namespace NUMINAMATH_CALUDE_wall_bricks_count_l2764_276405

/-- The number of bricks in the wall after adjustments -/
def total_bricks : ℕ :=
  let initial_courses := 5
  let additional_courses := 7
  let bricks_per_course := 450
  let initial_bricks := initial_courses * bricks_per_course
  let added_bricks := additional_courses * bricks_per_course
  let removed_bricks := [
    bricks_per_course / 3,
    bricks_per_course / 4,
    bricks_per_course / 5,
    bricks_per_course / 6,
    bricks_per_course / 7,
    bricks_per_course / 9,
    10
  ]
  initial_bricks + added_bricks - removed_bricks.sum

/-- Theorem stating that the total number of bricks in the wall is 4848 -/
theorem wall_bricks_count : total_bricks = 4848 := by
  sorry

end NUMINAMATH_CALUDE_wall_bricks_count_l2764_276405


namespace NUMINAMATH_CALUDE_function_properties_function_range_l2764_276486

def f (a x : ℝ) : ℝ := x^2 - 2*a*x + 5

theorem function_properties (a : ℝ) :
  (∀ x ∈ Set.Icc 1 a, f a x ∈ Set.Icc 1 a) ∧
  (Set.range (f a) = Set.Icc 1 a) →
  a = 2 :=
sorry

theorem function_range (a : ℝ) :
  (∀ x ≤ 2, ∀ y ≤ 2, x < y → f a x > f a y) ∧
  (∀ x ∈ Set.Icc 1 (a + 1), ∀ y ∈ Set.Icc 1 (a + 1), |f a x - f a y| ≤ 4) →
  a ∈ Set.Icc 2 3 :=
sorry

end NUMINAMATH_CALUDE_function_properties_function_range_l2764_276486


namespace NUMINAMATH_CALUDE_lisa_candy_consumption_l2764_276468

/-- The number of candies Lisa has initially -/
def initial_candies : ℕ := 36

/-- The number of candies Lisa eats on Mondays and Wednesdays -/
def candies_on_mon_wed : ℕ := 2

/-- The number of candies Lisa eats on other days -/
def candies_on_other_days : ℕ := 1

/-- The number of days Lisa eats 2 candies per week -/
def days_with_two_candies : ℕ := 2

/-- The number of days Lisa eats 1 candy per week -/
def days_with_one_candy : ℕ := 5

/-- The total number of candies Lisa eats in a week -/
def candies_per_week : ℕ := 
  days_with_two_candies * candies_on_mon_wed + 
  days_with_one_candy * candies_on_other_days

/-- The number of weeks it takes for Lisa to eat all the candies -/
def weeks_to_eat_all_candies : ℕ := initial_candies / candies_per_week

theorem lisa_candy_consumption : weeks_to_eat_all_candies = 4 := by
  sorry

end NUMINAMATH_CALUDE_lisa_candy_consumption_l2764_276468


namespace NUMINAMATH_CALUDE_cube_difference_l2764_276443

theorem cube_difference (x y : ℝ) (h1 : x - y = 3) (h2 : x^2 + y^2 = 27) : 
  x^3 - y^3 = 108 := by
sorry

end NUMINAMATH_CALUDE_cube_difference_l2764_276443


namespace NUMINAMATH_CALUDE_wendy_count_problem_l2764_276476

theorem wendy_count_problem (total_heads : ℕ) (total_legs : ℕ) 
  (h1 : total_heads = 28) 
  (h2 : total_legs = 92) : 
  ∃ (people animals : ℕ), 
    people + animals = total_heads ∧ 
    2 * people + 4 * animals = total_legs ∧ 
    people = 10 := by
  sorry

end NUMINAMATH_CALUDE_wendy_count_problem_l2764_276476


namespace NUMINAMATH_CALUDE_arcade_spending_fraction_l2764_276488

theorem arcade_spending_fraction (allowance : ℚ) (remaining : ℚ) 
  (h1 : allowance = 480 / 100)
  (h2 : remaining = 128 / 100)
  (h3 : remaining = (2/3) * (1 - (arcade_fraction : ℚ)) * allowance) :
  arcade_fraction = 3/5 := by
sorry

end NUMINAMATH_CALUDE_arcade_spending_fraction_l2764_276488


namespace NUMINAMATH_CALUDE_find_multiplier_l2764_276446

theorem find_multiplier (x : ℕ) : 72514 * x = 724777430 → x = 10001 := by
  sorry

end NUMINAMATH_CALUDE_find_multiplier_l2764_276446


namespace NUMINAMATH_CALUDE_A_minus_3B_formula_A_minus_3B_value_x_value_when_independent_of_y_l2764_276455

-- Define A and B as functions of x and y
def A (x y : ℝ) : ℝ := 3 * x^2 - x + 2 * y - 4 * x * y
def B (x y : ℝ) : ℝ := x^2 - 2 * x - y + x * y - 5

-- Theorem 1: A - 3B = 5x + 5y - 7xy + 15
theorem A_minus_3B_formula (x y : ℝ) :
  A x y - 3 * B x y = 5 * x + 5 * y - 7 * x * y + 15 := by sorry

-- Theorem 2: A - 3B = 26 when (x + y - 4/5)^2 + |xy + 1| = 0
theorem A_minus_3B_value (x y : ℝ) 
  (h : (x + y - 4/5)^2 + |x * y + 1| = 0) :
  A x y - 3 * B x y = 26 := by sorry

-- Theorem 3: x = 5/7 when the coefficient of y in A - 3B is zero
theorem x_value_when_independent_of_y (x : ℝ) 
  (h : ∀ y : ℝ, 5 - 7 * x = 0) :
  x = 5/7 := by sorry

end NUMINAMATH_CALUDE_A_minus_3B_formula_A_minus_3B_value_x_value_when_independent_of_y_l2764_276455


namespace NUMINAMATH_CALUDE_logistics_center_equidistant_l2764_276400

def rectilinear_distance (x₁ y₁ x₂ y₂ : ℝ) : ℝ :=
  |x₁ - x₂| + |y₁ - y₂|

def town_A : ℝ × ℝ := (2, 3)
def town_B : ℝ × ℝ := (-6, 9)
def town_C : ℝ × ℝ := (-3, -8)
def logistics_center : ℝ × ℝ := (-5, 0)

theorem logistics_center_equidistant :
  let (x, y) := logistics_center
  rectilinear_distance x y town_A.1 town_A.2 =
  rectilinear_distance x y town_B.1 town_B.2 ∧
  rectilinear_distance x y town_B.1 town_B.2 =
  rectilinear_distance x y town_C.1 town_C.2 :=
by sorry

end NUMINAMATH_CALUDE_logistics_center_equidistant_l2764_276400


namespace NUMINAMATH_CALUDE_exists_x_tan_eq_two_l2764_276464

theorem exists_x_tan_eq_two : ∃ x : ℝ, Real.tan x = 2 := by
  sorry

end NUMINAMATH_CALUDE_exists_x_tan_eq_two_l2764_276464


namespace NUMINAMATH_CALUDE_S_equation_holds_iff_specific_pairs_l2764_276499

/-- Given real numbers x, y, z with x + y + z = 0, S_r is defined as x^r + y^r + z^r -/
def S (r : ℕ+) (x y z : ℝ) : ℝ := x^(r:ℕ) + y^(r:ℕ) + z^(r:ℕ)

/-- The theorem states that for positive integers m and n, 
    the equation S_{m+n}/(m+n) = (S_m/m) * (S_n/n) holds if and only if 
    (m, n) is one of the pairs (2, 3), (3, 2), (2, 5), or (5, 2) -/
theorem S_equation_holds_iff_specific_pairs (x y z : ℝ) (h : x + y + z = 0) :
  ∀ m n : ℕ+, 
    (S (m + n) x y z) / (m + n : ℝ) = (S m x y z) / (m : ℝ) * (S n x y z) / (n : ℝ) ↔ 
    ((m = 2 ∧ n = 3) ∨ (m = 3 ∧ n = 2) ∨ (m = 2 ∧ n = 5) ∨ (m = 5 ∧ n = 2)) :=
by sorry

end NUMINAMATH_CALUDE_S_equation_holds_iff_specific_pairs_l2764_276499


namespace NUMINAMATH_CALUDE_digit_sum_property_l2764_276439

/-- A function that returns the sum of digits of a natural number -/
def sumOfDigits (n : ℕ) : ℕ := sorry

/-- A proposition stating that a number is a 1962-digit number -/
def is1962DigitNumber (n : ℕ) : Prop := sorry

theorem digit_sum_property (n : ℕ) 
  (h1 : is1962DigitNumber n) 
  (h2 : n % 9 = 0) : 
  sumOfDigits (sumOfDigits (sumOfDigits n)) = 9 := by sorry

end NUMINAMATH_CALUDE_digit_sum_property_l2764_276439


namespace NUMINAMATH_CALUDE_complex_fraction_calculation_l2764_276402

theorem complex_fraction_calculation : 
  (13/6 : ℚ) + ((((432/100 - 168/100 - 33/25) * 5/11 - 2/7) / (44/35)) : ℚ) = 521/210 := by
  sorry

end NUMINAMATH_CALUDE_complex_fraction_calculation_l2764_276402


namespace NUMINAMATH_CALUDE_solution_set_inequality_l2764_276445

theorem solution_set_inequality (x : ℝ) : 
  (Set.Ioo 1 2 : Set ℝ) = {x | (x - 1) * (2 - x) > 0} :=
sorry

end NUMINAMATH_CALUDE_solution_set_inequality_l2764_276445


namespace NUMINAMATH_CALUDE_smallest_k_for_mutual_criticism_l2764_276411

/-- Represents a group of deputies and their criticisms. -/
structure DeputyGroup where
  n : ℕ  -- Number of deputies
  k : ℕ  -- Number of deputies each deputy criticizes

/-- Defines when a DeputyGroup has mutual criticism. -/
def has_mutual_criticism (g : DeputyGroup) : Prop :=
  g.n * g.k > (g.n.choose 2)

/-- The smallest k that guarantees mutual criticism in a group of 15 deputies. -/
theorem smallest_k_for_mutual_criticism :
  ∃ k : ℕ, k = 8 ∧
  (∀ g : DeputyGroup, g.n = 15 → g.k ≥ k → has_mutual_criticism g) ∧
  (∀ k' : ℕ, k' < k → ∃ g : DeputyGroup, g.n = 15 ∧ g.k = k' ∧ ¬has_mutual_criticism g) :=
sorry

end NUMINAMATH_CALUDE_smallest_k_for_mutual_criticism_l2764_276411


namespace NUMINAMATH_CALUDE_largest_angle_in_pentagon_l2764_276461

/-- In a pentagon FGHIJ, given the following conditions:
  - Angle F measures 50°
  - Angle G measures 75°
  - Angles H and I are equal
  - Angle J is 10° more than twice angle H
  Prove that the largest angle measures 212.5° -/
theorem largest_angle_in_pentagon (F G H I J : ℝ) : 
  F = 50 ∧ 
  G = 75 ∧ 
  H = I ∧ 
  J = 2 * H + 10 ∧ 
  F + G + H + I + J = 540 → 
  max F (max G (max H (max I J))) = 212.5 := by
sorry

end NUMINAMATH_CALUDE_largest_angle_in_pentagon_l2764_276461


namespace NUMINAMATH_CALUDE_andy_ball_count_l2764_276498

theorem andy_ball_count : ∃ (a r m : ℕ), 
  (a = 2 * r) ∧ 
  (a = m + 5) ∧ 
  (a + r + m = 35) → 
  a = 16 := by
  sorry

end NUMINAMATH_CALUDE_andy_ball_count_l2764_276498


namespace NUMINAMATH_CALUDE_bell_rings_count_l2764_276475

def number_of_classes : Nat := 5

def current_class : Nat := 5

def bell_rings_per_class : Nat := 2

theorem bell_rings_count (n : Nat) (c : Nat) (r : Nat) 
  (h1 : n = number_of_classes) 
  (h2 : c = current_class) 
  (h3 : r = bell_rings_per_class) 
  (h4 : c ≤ n) : 
  (c - 1) * r + 1 = 9 := by
  sorry

#check bell_rings_count

end NUMINAMATH_CALUDE_bell_rings_count_l2764_276475


namespace NUMINAMATH_CALUDE_perpendicular_condition_l2764_276474

/-- Two lines are perpendicular if and only if the product of their slopes is -1 -/
def perpendicular (m₁ m₂ : ℝ) : Prop := m₁ * m₂ = -1

/-- The slope of the first line y = ax + 1 -/
def slope₁ (a : ℝ) : ℝ := a

/-- The slope of the second line y = (a-2)x - 1 -/
def slope₂ (a : ℝ) : ℝ := a - 2

/-- Theorem: a = 1 is a necessary and sufficient condition for the lines to be perpendicular -/
theorem perpendicular_condition (a : ℝ) : 
  perpendicular (slope₁ a) (slope₂ a) ↔ a = 1 := by
  sorry

end NUMINAMATH_CALUDE_perpendicular_condition_l2764_276474


namespace NUMINAMATH_CALUDE_krishans_money_l2764_276463

theorem krishans_money (ram gopal krishan : ℕ) : 
  (ram : ℚ) / gopal = 7 / 17 →
  (gopal : ℚ) / krishan = 7 / 17 →
  ram = 490 →
  krishan = 2890 := by
sorry

end NUMINAMATH_CALUDE_krishans_money_l2764_276463


namespace NUMINAMATH_CALUDE_max_value_implies_a_l2764_276453

/-- Given a function y = x(1-ax) where 0 < x < 1/a, if the maximum value of y is 1/12, then a = 3 -/
theorem max_value_implies_a (a : ℝ) : 
  (∃ (y : ℝ → ℝ), (∀ x : ℝ, 0 < x → x < 1/a → y x = x*(1-a*x)) ∧ 
   (∃ M : ℝ, M = 1/12 ∧ ∀ x : ℝ, 0 < x → x < 1/a → y x ≤ M)) →
  a = 3 := by
sorry

end NUMINAMATH_CALUDE_max_value_implies_a_l2764_276453


namespace NUMINAMATH_CALUDE_max_value_of_sum_products_l2764_276406

theorem max_value_of_sum_products (x y z : ℝ) (h : x + 2 * y + z = 6) :
  ∃ (max : ℝ), max = 6 ∧ ∀ (a b c : ℝ), a + 2 * b + c = 6 → a * b + a * c + b * c ≤ max :=
sorry

end NUMINAMATH_CALUDE_max_value_of_sum_products_l2764_276406


namespace NUMINAMATH_CALUDE_words_lost_in_oz_l2764_276480

/-- The number of letters in the alphabet -/
def alphabet_size : ℕ := 69

/-- The number of letters prohibited -/
def prohibited_letters : ℕ := 1

/-- The maximum word length -/
def max_word_length : ℕ := 2

/-- Calculate the number of words lost due to letter prohibition -/
def words_lost (alphabet_size : ℕ) (prohibited_letters : ℕ) (max_word_length : ℕ) : ℕ :=
  prohibited_letters + 
  (alphabet_size * prohibited_letters + alphabet_size * prohibited_letters - prohibited_letters * prohibited_letters)

theorem words_lost_in_oz : 
  words_lost alphabet_size prohibited_letters max_word_length = 138 := by
  sorry

end NUMINAMATH_CALUDE_words_lost_in_oz_l2764_276480


namespace NUMINAMATH_CALUDE_biking_problem_solution_l2764_276449

/-- Represents the problem of Andrea and Lauren biking in a park -/
def BikingProblem (park_length : ℝ) (distance_decrease_rate : ℝ) (andrea_initial_time : ℝ) (andrea_wait_time : ℝ) : Prop :=
  ∃ (lauren_speed : ℝ),
    lauren_speed > 0 ∧
    2 * lauren_speed + lauren_speed = distance_decrease_rate ∧
    let initial_distance := distance_decrease_rate * andrea_initial_time
    let remaining_distance := park_length - initial_distance
    let lauren_time := remaining_distance / lauren_speed
    andrea_initial_time + andrea_wait_time + lauren_time = 79

/-- The theorem stating the solution to the biking problem -/
theorem biking_problem_solution :
  BikingProblem 24 0.8 7 3 := by
  sorry

end NUMINAMATH_CALUDE_biking_problem_solution_l2764_276449


namespace NUMINAMATH_CALUDE_max_profundity_eq_fib_l2764_276424

/-- The dog dictionary consists of words made from letters A and U -/
inductive DogLetter
| A
| U

/-- A word in the dog dictionary is a list of DogLetters -/
def DogWord := List DogLetter

/-- The Fibonacci sequence -/
def fib : ℕ → ℕ
| 0 => 0
| 1 => 1
| (n + 2) => fib n + fib (n + 1)

/-- The profundity of a word is the number of its subwords -/
def profundity (w : DogWord) : ℕ := sorry

/-- The maximum profundity for words of length n -/
def max_profundity (n : ℕ) : ℕ := sorry

/-- The main theorem: maximum profundity equals F_{n+3} - 3 -/
theorem max_profundity_eq_fib (n : ℕ) :
  max_profundity n = fib (n + 3) - 3 := by sorry

end NUMINAMATH_CALUDE_max_profundity_eq_fib_l2764_276424


namespace NUMINAMATH_CALUDE_modulus_of_complex_number_l2764_276490

theorem modulus_of_complex_number (z : ℂ) :
  z = Complex.mk (Real.sqrt 3 / 2) (-3 / 2) →
  Complex.abs z = Real.sqrt 3 := by
sorry

end NUMINAMATH_CALUDE_modulus_of_complex_number_l2764_276490


namespace NUMINAMATH_CALUDE_five_rooks_on_five_by_five_l2764_276442

/-- The number of ways to place n distinct rooks on an nxn chess board 
    such that each column and row contains no more than one rook -/
def rook_placements (n : ℕ) : ℕ := Nat.factorial n

/-- Theorem: There are 120 ways to place 5 distinct rooks on a 5x5 chess board 
    such that each column and row contains no more than one rook -/
theorem five_rooks_on_five_by_five : rook_placements 5 = 120 := by
  sorry

end NUMINAMATH_CALUDE_five_rooks_on_five_by_five_l2764_276442


namespace NUMINAMATH_CALUDE_last_divisor_problem_l2764_276440

theorem last_divisor_problem (initial : ℚ) (div1 div2 mult last_div : ℚ) (result : ℚ) : 
  initial = 377 →
  div1 = 13 →
  div2 = 29 →
  mult = 1/4 →
  result = 0.125 →
  (((initial / div1) / div2) * mult) / last_div = result →
  last_div = 2 :=
by sorry

end NUMINAMATH_CALUDE_last_divisor_problem_l2764_276440


namespace NUMINAMATH_CALUDE_complex_expression_equality_l2764_276472

theorem complex_expression_equality : 
  (Real.pi - 3.14) ^ 0 + |-Real.sqrt 3| - (1/2)⁻¹ - Real.sin (π/3) = -1 + Real.sqrt 3 / 2 := by
  sorry

end NUMINAMATH_CALUDE_complex_expression_equality_l2764_276472


namespace NUMINAMATH_CALUDE_chord_length_l2764_276418

-- Define the line L: 3x + 4y - 5 = 0
def L : Set (ℝ × ℝ) := {p : ℝ × ℝ | 3 * p.1 + 4 * p.2 - 5 = 0}

-- Define the circle C: x^2 + y^2 = 4
def C : Set (ℝ × ℝ) := {p : ℝ × ℝ | p.1^2 + p.2^2 = 4}

-- Define the intersection points A and B
def A : ℝ × ℝ := sorry
def B : ℝ × ℝ := sorry

-- State the theorem
theorem chord_length : 
  A ∈ L ∧ A ∈ C ∧ B ∈ L ∧ B ∈ C → 
  Real.sqrt ((A.1 - B.1)^2 + (A.2 - B.2)^2) = 2 * Real.sqrt 3 := by
  sorry

end NUMINAMATH_CALUDE_chord_length_l2764_276418


namespace NUMINAMATH_CALUDE_existence_of_cube_triplet_l2764_276452

theorem existence_of_cube_triplet :
  ∃ n₀ : ℕ, ∀ m : ℕ, m ≥ n₀ →
    ∃ a b c : ℕ+,
      (m ^ 3 : ℝ) < (a : ℝ) ∧
      (a : ℝ) < (b : ℝ) ∧
      (b : ℝ) < (c : ℝ) ∧
      (c : ℝ) < ((m + 1) ^ 3 : ℝ) ∧
      ∃ k : ℕ, (a * b * c : ℕ) = k ^ 3 :=
by
  sorry

end NUMINAMATH_CALUDE_existence_of_cube_triplet_l2764_276452


namespace NUMINAMATH_CALUDE_triangle_sum_equals_nine_l2764_276436

def triangle_operation (a b c : ℤ) : ℤ := a * b - c

theorem triangle_sum_equals_nine : 
  triangle_operation 3 4 5 + triangle_operation 1 2 4 + triangle_operation 2 5 6 = 9 := by
  sorry

end NUMINAMATH_CALUDE_triangle_sum_equals_nine_l2764_276436


namespace NUMINAMATH_CALUDE_golden_ratio_properties_l2764_276419

theorem golden_ratio_properties :
  let a : ℝ := (Real.sqrt 5 + 1) / 2
  let b : ℝ := (Real.sqrt 5 - 1) / 2
  (b / a + a / b = 3) ∧ (a^2 + b^2 + a*b = 4) := by
  sorry

end NUMINAMATH_CALUDE_golden_ratio_properties_l2764_276419
