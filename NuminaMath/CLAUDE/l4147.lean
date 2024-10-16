import Mathlib

namespace NUMINAMATH_CALUDE_paper_torn_fraction_l4147_414707

theorem paper_torn_fraction (perimeter : ℝ) (remaining_area : ℝ) : 
  perimeter = 32 → remaining_area = 48 → 
  (perimeter / 4)^2 - remaining_area = (1 / 4) * (perimeter / 4)^2 := by
  sorry

end NUMINAMATH_CALUDE_paper_torn_fraction_l4147_414707


namespace NUMINAMATH_CALUDE_marble_remainder_l4147_414702

theorem marble_remainder (l j : ℕ) 
  (hl : l % 8 = 5) 
  (hj : j % 8 = 6) : 
  (l + j) % 8 = 3 := by
sorry

end NUMINAMATH_CALUDE_marble_remainder_l4147_414702


namespace NUMINAMATH_CALUDE_second_point_x_coordinate_l4147_414783

/-- Given two points on a line, prove that the x-coordinate of the second point is m + 5 -/
theorem second_point_x_coordinate 
  (m n : ℝ) 
  (h1 : m = 2 * n + 5) 
  (h2 : m + 5 = 2 * (n + 2.5) + 5) : 
  m + 5 = m + 5 := by
  sorry

end NUMINAMATH_CALUDE_second_point_x_coordinate_l4147_414783


namespace NUMINAMATH_CALUDE_arithmetic_sequence_sum_equals_five_l4147_414769

/-- Arithmetic sequence with first term a₁ and common difference d -/
def arithmetic_sequence (a₁ : ℤ) (d : ℤ) : ℕ → ℤ
  | 0 => a₁
  | n + 1 => arithmetic_sequence a₁ d n + d

/-- Sum of first n terms of an arithmetic sequence -/
def sum_arithmetic_sequence (a₁ : ℤ) (d : ℤ) (n : ℕ) : ℤ :=
  n * (2 * a₁ + (n - 1) * d) / 2

theorem arithmetic_sequence_sum_equals_five (k : ℕ) :
  k > 0 ∧ sum_arithmetic_sequence (-3) 2 k = 5 → k = 5 := by
  sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_sum_equals_five_l4147_414769


namespace NUMINAMATH_CALUDE_passengers_gained_at_halfway_l4147_414763

theorem passengers_gained_at_halfway (num_cars : ℕ) (initial_people_per_car : ℕ) (total_people_at_end : ℕ) : 
  num_cars = 20 →
  initial_people_per_car = 3 →
  total_people_at_end = 80 →
  (total_people_at_end - num_cars * initial_people_per_car) / num_cars = 1 :=
by sorry

end NUMINAMATH_CALUDE_passengers_gained_at_halfway_l4147_414763


namespace NUMINAMATH_CALUDE_zoo_animals_count_l4147_414760

/-- The number of tiger enclosures in the zoo -/
def tiger_enclosures : ℕ := 4

/-- The number of zebra enclosures behind each tiger enclosure -/
def zebra_enclosures_per_tiger : ℕ := 2

/-- The ratio of giraffe enclosures to zebra enclosures -/
def giraffe_to_zebra_ratio : ℕ := 3

/-- The number of tigers in each tiger enclosure -/
def tigers_per_enclosure : ℕ := 4

/-- The number of zebras in each zebra enclosure -/
def zebras_per_enclosure : ℕ := 10

/-- The number of giraffes in each giraffe enclosure -/
def giraffes_per_enclosure : ℕ := 2

/-- The total number of animals in the zoo -/
def total_animals : ℕ := 144

theorem zoo_animals_count :
  tiger_enclosures * tigers_per_enclosure +
  (tiger_enclosures * zebra_enclosures_per_tiger) * zebras_per_enclosure +
  (tiger_enclosures * zebra_enclosures_per_tiger * giraffe_to_zebra_ratio) * giraffes_per_enclosure =
  total_animals := by sorry

end NUMINAMATH_CALUDE_zoo_animals_count_l4147_414760


namespace NUMINAMATH_CALUDE_stratified_sampling_theorem_l4147_414776

/-- Represents the number of students in each grade level -/
structure GradePopulation where
  freshmen : ℕ
  sophomores : ℕ
  juniors : ℕ

/-- Represents the sample size for each grade level -/
structure GradeSample where
  freshmen : ℕ
  sophomores : ℕ
  juniors : ℕ

/-- The total population of the school -/
def totalPopulation (gp : GradePopulation) : ℕ :=
  gp.freshmen + gp.sophomores + gp.juniors

/-- The total sample size -/
def totalSample (gs : GradeSample) : ℕ :=
  gs.freshmen + gs.sophomores + gs.juniors

/-- Checks if the sample is proportional to the population for each grade -/
def isProportionalSample (gp : GradePopulation) (gs : GradeSample) : Prop :=
  gs.freshmen * totalPopulation gp = gp.freshmen * totalSample gs ∧
  gs.sophomores * totalPopulation gp = gp.sophomores * totalSample gs ∧
  gs.juniors * totalPopulation gp = gp.juniors * totalSample gs

theorem stratified_sampling_theorem (gp : GradePopulation) (gs : GradeSample) :
  gp.freshmen = 300 →
  gp.sophomores = 200 →
  gp.juniors = 400 →
  totalPopulation gp = 900 →
  totalSample gs = 45 →
  isProportionalSample gp gs →
  gs.freshmen = 15 ∧ gs.sophomores = 10 ∧ gs.juniors = 20 := by
  sorry


end NUMINAMATH_CALUDE_stratified_sampling_theorem_l4147_414776


namespace NUMINAMATH_CALUDE_geometric_series_relation_l4147_414722

/-- Given real numbers c and d satisfying an infinite geometric series condition,
    prove that another related infinite geometric series equals 5/7. -/
theorem geometric_series_relation (c d : ℝ) 
    (h : (c/d) / (1 - 1/d) = 5) : 
    (c/(c+2*d)) / (1 - 1/(c+2*d)) = 5/7 := by
  sorry

end NUMINAMATH_CALUDE_geometric_series_relation_l4147_414722


namespace NUMINAMATH_CALUDE_original_proposition_converse_is_false_inverse_is_false_contrapositive_is_true_l4147_414717

-- Original proposition
theorem original_proposition (a b : ℝ) : a = b → a^2 = b^2 := by sorry

-- Converse is false
theorem converse_is_false : ¬ (∀ a b : ℝ, a^2 = b^2 → a = b) := by sorry

-- Inverse is false
theorem inverse_is_false : ¬ (∀ a b : ℝ, a ≠ b → a^2 ≠ b^2) := by sorry

-- Contrapositive is true
theorem contrapositive_is_true : ∀ a b : ℝ, a^2 ≠ b^2 → a ≠ b := by sorry

end NUMINAMATH_CALUDE_original_proposition_converse_is_false_inverse_is_false_contrapositive_is_true_l4147_414717


namespace NUMINAMATH_CALUDE_problem_statement_l4147_414728

theorem problem_statement :
  (∀ a b : ℝ, a > 0 → b > 0 → 1 / (a + b) ≤ (1 / 4) * (1 / a + 1 / b)) ∧
  (∀ x₁ x₂ x₃ : ℝ, x₁ > 0 → x₂ > 0 → x₃ > 0 → 
    1 / x₁ + 1 / x₂ + 1 / x₃ = 1 →
    (x₁ + x₂ + x₃) / (x₁ * x₃ + x₃ * x₂) + 
    (x₁ + x₂ + x₃) / (x₁ * x₂ + x₃ * x₁) + 
    (x₁ + x₂ + x₃) / (x₂ * x₁ + x₃ * x₂) ≤ 3 / 2) := by
  sorry

end NUMINAMATH_CALUDE_problem_statement_l4147_414728


namespace NUMINAMATH_CALUDE_bird_count_l4147_414754

theorem bird_count (total_wings : ℕ) (wings_per_bird : ℕ) (h1 : total_wings = 20) (h2 : wings_per_bird = 2) :
  total_wings / wings_per_bird = 10 := by
sorry

end NUMINAMATH_CALUDE_bird_count_l4147_414754


namespace NUMINAMATH_CALUDE_parallel_vectors_x_value_l4147_414732

/-- Two 2D vectors are parallel if their cross product is zero -/
def are_parallel (a b : ℝ × ℝ) : Prop :=
  a.1 * b.2 - a.2 * b.1 = 0

/-- The theorem stating that if vectors (1, 2) and (x, -3) are parallel, then x = -3/2 -/
theorem parallel_vectors_x_value :
  ∀ x : ℝ, are_parallel (1, 2) (x, -3) → x = -3/2 := by
  sorry

end NUMINAMATH_CALUDE_parallel_vectors_x_value_l4147_414732


namespace NUMINAMATH_CALUDE_kellys_snacks_l4147_414778

theorem kellys_snacks (peanuts raisins : ℝ) (h1 : peanuts = 0.1) (h2 : raisins = 0.4) :
  peanuts + raisins = 0.5 := by
  sorry

end NUMINAMATH_CALUDE_kellys_snacks_l4147_414778


namespace NUMINAMATH_CALUDE_geometric_progression_values_l4147_414759

theorem geometric_progression_values (p : ℝ) : 
  (4*p + 5 ≠ 0 ∧ 2*p ≠ 0 ∧ |p - 3| ≠ 0) ∧
  (2*p)^2 = (4*p + 5) * |p - 3| ↔ 
  p = -1 ∨ p = 15/8 := by sorry

end NUMINAMATH_CALUDE_geometric_progression_values_l4147_414759


namespace NUMINAMATH_CALUDE_book_price_increase_l4147_414799

theorem book_price_increase (new_price : ℝ) (increase_percentage : ℝ) (original_price : ℝ) : 
  new_price = 390 ∧ increase_percentage = 30 →
  original_price * (1 + increase_percentage / 100) = new_price →
  original_price = 300 := by
sorry

end NUMINAMATH_CALUDE_book_price_increase_l4147_414799


namespace NUMINAMATH_CALUDE_line_equations_l4147_414751

/-- Point in 2D plane -/
structure Point where
  x : ℝ
  y : ℝ

/-- Line in 2D plane represented by ax + by + c = 0 -/
structure Line where
  a : ℝ
  b : ℝ
  c : ℝ

def parallel (l1 l2 : Line) : Prop :=
  l1.a * l2.b = l1.b * l2.a

def perpendicular (l1 l2 : Line) : Prop :=
  l1.a * l2.a + l1.b * l2.b = 0

def passes_through (l : Line) (p : Point) : Prop :=
  l.a * p.x + l.b * p.y + l.c = 0

theorem line_equations 
  (A : Point)
  (l m n : Line)
  (h_A : A = ⟨-2, 1⟩)
  (h_l : l = ⟨2, -1, -3⟩)
  (h_m_parallel : parallel m l)
  (h_m_passes : passes_through m A)
  (h_n_perpendicular : perpendicular n l)
  (h_n_passes : passes_through n A) :
  (m = ⟨2, -1, 5⟩) ∧ (n = ⟨1, 2, 0⟩) := by
  sorry

end NUMINAMATH_CALUDE_line_equations_l4147_414751


namespace NUMINAMATH_CALUDE_angle_sum_ninety_degrees_l4147_414716

theorem angle_sum_ninety_degrees (A B : Real) (h : (Real.cos A / Real.sin B) + (Real.cos B / Real.sin A) = 2) :
  A + B = π / 2 := by
  sorry

end NUMINAMATH_CALUDE_angle_sum_ninety_degrees_l4147_414716


namespace NUMINAMATH_CALUDE_ellipse_and_outer_point_properties_l4147_414721

/-- Definition of an ellipse C with given properties -/
structure Ellipse :=
  (a b : ℝ)
  (h1 : a > b)
  (h2 : b > 0)
  (h3 : a^2 - b^2 = 5)
  (h4 : a = 3)

/-- Definition of a point P outside the ellipse -/
structure OuterPoint (C : Ellipse) :=
  (x₀ y₀ : ℝ)
  (h5 : x₀^2 / C.a^2 + y₀^2 / C.b^2 > 1)

/-- Theorem stating the properties of the ellipse and outer point -/
theorem ellipse_and_outer_point_properties (C : Ellipse) (P : OuterPoint C) :
  (∀ x y, x^2 / 9 + y^2 / 4 = 1 ↔ x^2 / C.a^2 + y^2 / C.b^2 = 1) ∧
  (P.x₀^2 + P.y₀^2 = 13) :=
sorry

end NUMINAMATH_CALUDE_ellipse_and_outer_point_properties_l4147_414721


namespace NUMINAMATH_CALUDE_buy_three_items_count_l4147_414740

/-- Represents the inventory of a store selling computer peripherals -/
structure StoreInventory where
  headphones : Nat
  mice : Nat
  keyboards : Nat
  keyboard_mouse_sets : Nat
  headphone_mouse_sets : Nat

/-- Calculates the number of ways to buy a headphone, a keyboard, and a mouse -/
def ways_to_buy_three (inventory : StoreInventory) : Nat :=
  inventory.keyboard_mouse_sets * inventory.headphones +
  inventory.headphone_mouse_sets * inventory.keyboards +
  inventory.headphones * inventory.mice * inventory.keyboards

/-- The theorem stating that there are 646 ways to buy three items -/
theorem buy_three_items_count (inventory : StoreInventory) 
  (h1 : inventory.headphones = 9)
  (h2 : inventory.mice = 13)
  (h3 : inventory.keyboards = 5)
  (h4 : inventory.keyboard_mouse_sets = 4)
  (h5 : inventory.headphone_mouse_sets = 5) :
  ways_to_buy_three inventory = 646 := by
  sorry

#eval ways_to_buy_three { headphones := 9, mice := 13, keyboards := 5, keyboard_mouse_sets := 4, headphone_mouse_sets := 5 }

end NUMINAMATH_CALUDE_buy_three_items_count_l4147_414740


namespace NUMINAMATH_CALUDE_matrix_power_2023_l4147_414753

def A : Matrix (Fin 2) (Fin 2) ℤ := !![1, 0; 2, 1]

theorem matrix_power_2023 :
  A ^ 2023 = !![1, 0; 4046, 1] := by sorry

end NUMINAMATH_CALUDE_matrix_power_2023_l4147_414753


namespace NUMINAMATH_CALUDE_divisors_of_2121_with_units_digit_1_l4147_414730

/-- The number of positive integer divisors of 2121 with a units digit of 1 is 4. -/
theorem divisors_of_2121_with_units_digit_1 : 
  (Finset.filter (fun d => d % 10 = 1) (Nat.divisors 2121)).card = 4 := by
  sorry

end NUMINAMATH_CALUDE_divisors_of_2121_with_units_digit_1_l4147_414730


namespace NUMINAMATH_CALUDE_mariam_neighborhood_houses_l4147_414796

/-- The number of houses on one side of the main road -/
def houses_on_first_side : ℕ := 40

/-- The function representing the number of houses on the other side of the road -/
def f (x : ℕ) : ℕ := x^2 + 3*x

/-- The total number of houses in Mariam's neighborhood -/
def total_houses : ℕ := houses_on_first_side + f houses_on_first_side

theorem mariam_neighborhood_houses :
  total_houses = 1760 := by sorry

end NUMINAMATH_CALUDE_mariam_neighborhood_houses_l4147_414796


namespace NUMINAMATH_CALUDE_f_sum_negative_l4147_414785

def f (x : ℝ) : ℝ := 2 * x^3 + 4 * x

theorem f_sum_negative (a b c : ℝ) 
  (hab : a + b < 0) (hbc : b + c < 0) (hca : c + a < 0) : 
  f a + f b + f c < 0 := by
  sorry

end NUMINAMATH_CALUDE_f_sum_negative_l4147_414785


namespace NUMINAMATH_CALUDE_time_to_draw_picture_l4147_414709

/-- Proves that the time to draw each picture is 2 hours -/
theorem time_to_draw_picture (num_pictures : ℕ) (coloring_ratio : ℚ) (total_time : ℚ) :
  num_pictures = 10 →
  coloring_ratio = 7/10 →
  total_time = 34 →
  ∃ (draw_time : ℚ), draw_time = 2 ∧ num_pictures * draw_time * (1 + coloring_ratio) = total_time :=
by sorry

end NUMINAMATH_CALUDE_time_to_draw_picture_l4147_414709


namespace NUMINAMATH_CALUDE_rice_mixture_cost_problem_l4147_414710

/-- The cost of the second variety of rice per kg -/
def second_variety_cost : ℝ := 12.50

/-- The cost of the first variety of rice per kg -/
def first_variety_cost : ℝ := 5

/-- The cost of the mixture per kg -/
def mixture_cost : ℝ := 7.50

/-- The ratio of the two varieties of rice -/
def rice_ratio : ℝ := 0.5

theorem rice_mixture_cost_problem :
  first_variety_cost * 1 + second_variety_cost * rice_ratio = mixture_cost * (1 + rice_ratio) :=
by sorry

end NUMINAMATH_CALUDE_rice_mixture_cost_problem_l4147_414710


namespace NUMINAMATH_CALUDE_haley_halloween_candy_l4147_414795

/-- Represents the number of candy pieces Haley scored on Halloween -/
def initial_candy : ℕ := sorry

/-- Represents the number of candy pieces Haley ate -/
def eaten_candy : ℕ := 17

/-- Represents the number of candy pieces Haley received from her sister -/
def received_candy : ℕ := 19

/-- Represents the number of candy pieces Haley has now -/
def current_candy : ℕ := 35

/-- Proves that Haley scored 33 pieces of candy on Halloween -/
theorem haley_halloween_candy : initial_candy = 33 :=
  by
    have h : initial_candy - eaten_candy + received_candy = current_candy := sorry
    sorry

end NUMINAMATH_CALUDE_haley_halloween_candy_l4147_414795


namespace NUMINAMATH_CALUDE_max_backpacks_filled_fifteen_backpacks_possible_max_backpacks_is_fifteen_l4147_414780

def pencils : ℕ := 150
def notebooks : ℕ := 255
def pens : ℕ := 315

theorem max_backpacks_filled (n : ℕ) : 
  (pencils % n = 0 ∧ notebooks % n = 0 ∧ pens % n = 0) →
  n ≤ 15 :=
by
  sorry

theorem fifteen_backpacks_possible : 
  pencils % 15 = 0 ∧ notebooks % 15 = 0 ∧ pens % 15 = 0 :=
by
  sorry

theorem max_backpacks_is_fifteen : 
  ∀ n : ℕ, (pencils % n = 0 ∧ notebooks % n = 0 ∧ pens % n = 0) → n ≤ 15 :=
by
  sorry

end NUMINAMATH_CALUDE_max_backpacks_filled_fifteen_backpacks_possible_max_backpacks_is_fifteen_l4147_414780


namespace NUMINAMATH_CALUDE_three_integer_pairs_satisfy_equation_l4147_414703

theorem three_integer_pairs_satisfy_equation : 
  ∃! (s : Finset (ℤ × ℤ)), 
    (∀ (x y : ℤ), (x, y) ∈ s ↔ x^3 + y^2 = 2*y + 1) ∧ 
    s.card = 3 := by
  sorry

end NUMINAMATH_CALUDE_three_integer_pairs_satisfy_equation_l4147_414703


namespace NUMINAMATH_CALUDE_investment_principal_l4147_414770

/-- Given an investment with simple interest, prove that the principal is $200 -/
theorem investment_principal (P r : ℝ) 
  (h1 : P + 2 * P * r = 260)  -- Condition after 2 years
  (h2 : P + 5 * P * r = 350)  -- Condition after 5 years
  : P = 200 := by
  sorry

end NUMINAMATH_CALUDE_investment_principal_l4147_414770


namespace NUMINAMATH_CALUDE_decimal_sum_and_subtraction_l4147_414771

theorem decimal_sum_and_subtraction : 
  (0.804 + 0.007 + 0.0008) - 0.00009 = 0.81171 := by sorry

end NUMINAMATH_CALUDE_decimal_sum_and_subtraction_l4147_414771


namespace NUMINAMATH_CALUDE_cone_volume_l4147_414712

/-- The volume of a cone with lateral surface area 2√3π and central angle √3π is π. -/
theorem cone_volume (r l : ℝ) (h_angle : 2 * π * r / l = Real.sqrt 3 * π)
  (h_area : π * r * l = 2 * Real.sqrt 3 * π) : 
  (1/3) * π * r^2 * Real.sqrt (l^2 - r^2) = π :=
sorry

end NUMINAMATH_CALUDE_cone_volume_l4147_414712


namespace NUMINAMATH_CALUDE_distance_between_points_l4147_414704

/-- The distance between points (1, 3) and (-5, 7) is 2√13. -/
theorem distance_between_points : 
  let p1 : ℝ × ℝ := (1, 3)
  let p2 : ℝ × ℝ := (-5, 7)
  Real.sqrt ((p2.1 - p1.1)^2 + (p2.2 - p1.2)^2) = 2 * Real.sqrt 13 := by
  sorry

end NUMINAMATH_CALUDE_distance_between_points_l4147_414704


namespace NUMINAMATH_CALUDE_smaller_pyramid_volume_theorem_l4147_414762

/-- A right square pyramid with given dimensions -/
structure RightSquarePyramid where
  base_edge : ℝ
  slant_edge : ℝ

/-- A plane cutting the pyramid parallel to its base -/
structure CuttingPlane where
  height : ℝ

/-- The volume of the smaller pyramid cut off by the plane -/
def smaller_pyramid_volume (p : RightSquarePyramid) (c : CuttingPlane) : ℝ :=
  sorry

/-- Theorem stating the volume of the smaller pyramid -/
theorem smaller_pyramid_volume_theorem (p : RightSquarePyramid) (c : CuttingPlane) :
  p.base_edge = 12 * Real.sqrt 2 →
  p.slant_edge = 15 →
  c.height = 5 →
  smaller_pyramid_volume p c = 24576 / 507 :=
sorry

end NUMINAMATH_CALUDE_smaller_pyramid_volume_theorem_l4147_414762


namespace NUMINAMATH_CALUDE_inequality_properties_l4147_414747

theorem inequality_properties (a b c d : ℝ) 
  (h1 : a > b) (h2 : c > d) (h3 : d > 0) : 
  (a - d > b - c) ∧ 
  (a * c^2 > b * c^2) ∧
  (∃ a b c d : ℝ, a > b ∧ c > d ∧ d > 0 ∧ a * c ≤ b * d) ∧
  (∃ a b c d : ℝ, a > b ∧ c > d ∧ d > 0 ∧ a / d ≤ b / c) :=
by sorry

end NUMINAMATH_CALUDE_inequality_properties_l4147_414747


namespace NUMINAMATH_CALUDE_ing_catches_bo_l4147_414701

/-- The distance Bo jumps after n jumps -/
def bo_distance (n : ℕ) : ℕ := 6 * n

/-- The distance Ing jumps after n jumps -/
def ing_distance (n : ℕ) : ℕ := n * (n + 1) / 2

/-- The number of jumps needed for Ing to catch Bo -/
def catch_up_jumps : ℕ := 11

theorem ing_catches_bo : 
  bo_distance catch_up_jumps = ing_distance catch_up_jumps :=
sorry

end NUMINAMATH_CALUDE_ing_catches_bo_l4147_414701


namespace NUMINAMATH_CALUDE_probability_is_half_l4147_414714

/-- A game where a square is divided into triangular sections and some are shaded -/
structure SquareGame where
  total_sections : ℕ
  shaded_sections : ℕ
  h_total : total_sections = 8
  h_shaded : shaded_sections = 4

/-- The probability of landing on a shaded section -/
def probability_shaded (game : SquareGame) : ℚ :=
  game.shaded_sections / game.total_sections

/-- Theorem: The probability of landing on a shaded section is 1/2 -/
theorem probability_is_half (game : SquareGame) : probability_shaded game = 1/2 := by
  sorry

#eval probability_shaded { total_sections := 8, shaded_sections := 4, h_total := rfl, h_shaded := rfl }

end NUMINAMATH_CALUDE_probability_is_half_l4147_414714


namespace NUMINAMATH_CALUDE_y1_greater_than_y2_l4147_414750

/-- Given two points M(-3, y₁) and N(2, y₂) on the line y = -3x + 1, prove that y₁ > y₂ -/
theorem y1_greater_than_y2 (y₁ y₂ : ℝ) : 
  (y₁ = -3 * (-3) + 1) → (y₂ = -3 * 2 + 1) → y₁ > y₂ := by
  sorry

end NUMINAMATH_CALUDE_y1_greater_than_y2_l4147_414750


namespace NUMINAMATH_CALUDE_function_decreasing_condition_l4147_414723

-- Define the function f(x)
def f (a : ℝ) (x : ℝ) : ℝ := a * x^2 + 4 * (a + 1) * x - 3

-- State the theorem
theorem function_decreasing_condition (a : ℝ) :
  (∀ x ∈ Set.Ici 2, ∀ y ∈ Set.Ici 2, x < y → f a x > f a y) ↔ a ≤ -1/2 := by
  sorry

end NUMINAMATH_CALUDE_function_decreasing_condition_l4147_414723


namespace NUMINAMATH_CALUDE_max_x2_plus_y2_l4147_414739

theorem max_x2_plus_y2 (x y a : ℝ) (h1 : x + y = a + 1) (h2 : x * y = a^2 - 7*a + 16) :
  ∃ (max : ℝ), max = 32 ∧ ∀ (x' y' a' : ℝ), x' + y' = a' + 1 → x' * y' = a'^2 - 7*a' + 16 → x'^2 + y'^2 ≤ max :=
sorry

end NUMINAMATH_CALUDE_max_x2_plus_y2_l4147_414739


namespace NUMINAMATH_CALUDE_solution_set_equality_l4147_414720

def S : Set ℝ := {x : ℝ | |x - 1| + |x + 2| ≤ 4}

theorem solution_set_equality : S = Set.Icc (-5/2) (3/2) := by
  sorry

end NUMINAMATH_CALUDE_solution_set_equality_l4147_414720


namespace NUMINAMATH_CALUDE_negation_of_universal_proposition_l4147_414764

theorem negation_of_universal_proposition :
  (¬ ∀ x : ℝ, x > 1 → x^2 ≥ 3) ↔ (∃ x : ℝ, x > 1 ∧ x^2 < 3) := by sorry

end NUMINAMATH_CALUDE_negation_of_universal_proposition_l4147_414764


namespace NUMINAMATH_CALUDE_present_age_of_b_l4147_414782

theorem present_age_of_b (a b : ℕ) 
  (h1 : a + 10 = 2 * (b - 10))  -- In 10 years, A will be twice as old as B was 10 years ago
  (h2 : a = b + 9)              -- A is now 9 years older than B
  : b = 39 := by               -- The present age of B is 39 years
  sorry

end NUMINAMATH_CALUDE_present_age_of_b_l4147_414782


namespace NUMINAMATH_CALUDE_power_difference_equals_one_l4147_414744

theorem power_difference_equals_one (x y : ℕ) : 
  (2^x ∣ 180) ∧ 
  (3^y ∣ 180) ∧ 
  (∀ z : ℕ, z > x → ¬(2^z ∣ 180)) ∧ 
  (∀ w : ℕ, w > y → ¬(3^w ∣ 180)) → 
  (1/3 : ℚ)^(y - x) = 1 := by
sorry

end NUMINAMATH_CALUDE_power_difference_equals_one_l4147_414744


namespace NUMINAMATH_CALUDE_least_addition_for_divisibility_l4147_414757

theorem least_addition_for_divisibility (n m k : ℕ) (h : n + k = m * 29) : 
  ∀ j : ℕ, j < k → ¬(∃ l : ℕ, n + j = l * 29) :=
by
  sorry

#check least_addition_for_divisibility 1056 37 17

end NUMINAMATH_CALUDE_least_addition_for_divisibility_l4147_414757


namespace NUMINAMATH_CALUDE_arithmetic_sequence_terms_l4147_414772

/-- An arithmetic sequence with given parameters has 13 terms -/
theorem arithmetic_sequence_terms (a d l : ℤ) (h1 : a = -5) (h2 : d = 5) (h3 : l = 55) :
  ∃ n : ℕ, n = 13 ∧ l = a + (n - 1) * d :=
sorry

end NUMINAMATH_CALUDE_arithmetic_sequence_terms_l4147_414772


namespace NUMINAMATH_CALUDE_investment_rate_correct_l4147_414765

/-- Represents the annual interest rate as a real number between 0 and 1 -/
def annual_interest_rate : ℝ := sorry

/-- The initial investment amount in yuan -/
def initial_investment : ℝ := 20000

/-- The amount withdrawn after the first year in yuan -/
def withdrawal : ℝ := 10000

/-- The final amount received after two years in yuan -/
def final_amount : ℝ := 13200

/-- Theorem stating that the annual interest rate satisfies the investment conditions -/
theorem investment_rate_correct : 
  (initial_investment * (1 + annual_interest_rate) - withdrawal) * (1 + annual_interest_rate) = final_amount ∧ 
  annual_interest_rate = 0.1 := by sorry

end NUMINAMATH_CALUDE_investment_rate_correct_l4147_414765


namespace NUMINAMATH_CALUDE_leak_empty_time_proof_l4147_414777

/-- The time (in hours) it takes to fill the tank without a leak -/
def fill_time_without_leak : ℝ := 3

/-- The time (in hours) it takes to fill the tank with a leak -/
def fill_time_with_leak : ℝ := 4

/-- The capacity of the tank -/
def tank_capacity : ℝ := 1

/-- The time (in hours) it takes for the leak to empty the tank -/
def leak_empty_time : ℝ := 12

theorem leak_empty_time_proof :
  let fill_rate := tank_capacity / fill_time_without_leak
  let combined_rate := tank_capacity / fill_time_with_leak
  let leak_rate := fill_rate - combined_rate
  leak_empty_time = tank_capacity / leak_rate :=
by sorry

end NUMINAMATH_CALUDE_leak_empty_time_proof_l4147_414777


namespace NUMINAMATH_CALUDE_monthly_income_p_l4147_414749

/-- Given the average monthly incomes of pairs of individuals, prove that the monthly income of p is 4000. -/
theorem monthly_income_p (p q r : ℕ) : 
  (p + q) / 2 = 5050 →
  (q + r) / 2 = 6250 →
  (p + r) / 2 = 5200 →
  p = 4000 := by
sorry

end NUMINAMATH_CALUDE_monthly_income_p_l4147_414749


namespace NUMINAMATH_CALUDE_hamburgers_left_over_l4147_414798

theorem hamburgers_left_over (total : ℕ) (served : ℕ) (left_over : ℕ) : 
  total = 9 → served = 3 → left_over = total - served → left_over = 6 := by
sorry

end NUMINAMATH_CALUDE_hamburgers_left_over_l4147_414798


namespace NUMINAMATH_CALUDE_largest_base5_is_124_l4147_414791

/-- Represents a three-digit base-5 number -/
structure Base5Number where
  hundreds : Fin 5
  tens : Fin 5
  ones : Fin 5

/-- Converts a Base5Number to its decimal (base 10) representation -/
def toDecimal (n : Base5Number) : ℕ :=
  n.hundreds * 25 + n.tens * 5 + n.ones

/-- The largest three-digit base-5 number -/
def largestBase5 : Base5Number :=
  { hundreds := 4, tens := 4, ones := 4 }

theorem largest_base5_is_124 : toDecimal largestBase5 = 124 := by
  sorry

end NUMINAMATH_CALUDE_largest_base5_is_124_l4147_414791


namespace NUMINAMATH_CALUDE_sum_of_abc_l4147_414756

theorem sum_of_abc (a b c : ℝ) (ha : a > 0) (hb : b > 0) (hc : c > 0)
  (hab : a * b = 36) (hac : a * c = 72) (hbc : b * c = 108) :
  a + b + c = 13 * Real.sqrt 6 := by
  sorry

end NUMINAMATH_CALUDE_sum_of_abc_l4147_414756


namespace NUMINAMATH_CALUDE_polynomial_division_remainder_l4147_414700

theorem polynomial_division_remainder : ∃ (q r : Polynomial ℝ), 
  (X^3 + 2*X^2 - 3 : Polynomial ℝ) = (X^2 + 2) * q + r ∧ 
  r.degree < (X^2 + 2).degree ∧
  r = -2*X - 7 :=
sorry

end NUMINAMATH_CALUDE_polynomial_division_remainder_l4147_414700


namespace NUMINAMATH_CALUDE_max_p_value_l4147_414775

/-- The maximum value of p for two rectangular boxes with given conditions -/
theorem max_p_value (m n p : ℕ+) (h1 : m ≤ n) (h2 : n ≤ p)
  (h3 : 2 * (m * n * p) = (m + 2) * (n + 2) * (p + 2)) : 
  p ≤ 130 := by
sorry

end NUMINAMATH_CALUDE_max_p_value_l4147_414775


namespace NUMINAMATH_CALUDE_equation_root_in_interval_l4147_414737

theorem equation_root_in_interval : ∃! x : ℝ, x ∈ (Set.Ioo 0 1) ∧ 2^x = 2 - x := by sorry

end NUMINAMATH_CALUDE_equation_root_in_interval_l4147_414737


namespace NUMINAMATH_CALUDE_line_intersection_canonical_equations_l4147_414767

/-- The canonical equations of the line of intersection of two planes -/
theorem line_intersection_canonical_equations
  (p₁ : Real → Real → Real → Real)
  (p₂ : Real → Real → Real → Real)
  (h₁ : ∀ x y z, p₁ x y z = 3*x + y - z - 6)
  (h₂ : ∀ x y z, p₂ x y z = 3*x - y + 2*z)
  : ∃ (t : Real), ∀ x y z,
    (p₁ x y z = 0 ∧ p₂ x y z = 0) ↔
    (x = 1 + t ∧ y = 3 - 9*t ∧ z = -6*t) :=
sorry

end NUMINAMATH_CALUDE_line_intersection_canonical_equations_l4147_414767


namespace NUMINAMATH_CALUDE_exists_min_n_all_rows_shaded_l4147_414761

/-- Calculates the square number of the nth shaded square -/
def shadedSquareNumber (n : ℕ) : ℕ :=
  n * (n + 1) / 2

/-- Calculates the row number for a given square number -/
def squareToRow (square : ℕ) : ℕ :=
  (square - 1) / 5 + 1

/-- Checks if all rows are shaded up to the nth shaded square -/
def allRowsShaded (n : ℕ) : Prop :=
  ∀ row : ℕ, row ≤ 10 → ∃ k : ℕ, k ≤ n ∧ squareToRow (shadedSquareNumber k) = row

/-- The main theorem stating the existence of a minimum n that shades all rows -/
theorem exists_min_n_all_rows_shaded :
  ∃ n : ℕ, allRowsShaded n ∧ ∀ m : ℕ, m < n → ¬allRowsShaded m :=
sorry

end NUMINAMATH_CALUDE_exists_min_n_all_rows_shaded_l4147_414761


namespace NUMINAMATH_CALUDE_solution_eq_200_div_253_l4147_414788

/-- A binary operation on nonzero real numbers satisfying certain properties -/
def diamond (a b : ℝ) : ℝ := sorry

/-- The binary operation satisfies a ◇ (b ◇ c) = (a ◇ b) · c -/
axiom diamond_assoc (a b c : ℝ) : a ≠ 0 → b ≠ 0 → c ≠ 0 → diamond a (diamond b c) = (diamond a b) * c

/-- The binary operation satisfies a ◇ a = 1 -/
axiom diamond_self (a : ℝ) : a ≠ 0 → diamond a a = 1

/-- The solution to the equation 2024 ◇ (8 ◇ x) = 200 is 200/253 -/
theorem solution_eq_200_div_253 : ∃ (x : ℝ), x ≠ 0 ∧ diamond 2024 (diamond 8 x) = 200 ∧ x = 200/253 := by sorry

end NUMINAMATH_CALUDE_solution_eq_200_div_253_l4147_414788


namespace NUMINAMATH_CALUDE_parabola_vertex_l4147_414726

/-- The parabola equation -/
def parabola (x : ℝ) : ℝ := 2 * (x - 1)^2 + 2

/-- The vertex of the parabola -/
def vertex : ℝ × ℝ := (1, 2)

/-- Theorem: The vertex of the parabola y = 2(x-1)^2 + 2 is at the point (1, 2) -/
theorem parabola_vertex : 
  (∀ x : ℝ, parabola x ≥ parabola (vertex.1)) ∧ 
  parabola (vertex.1) = vertex.2 := by
  sorry

end NUMINAMATH_CALUDE_parabola_vertex_l4147_414726


namespace NUMINAMATH_CALUDE_complex_square_value_l4147_414705

theorem complex_square_value (m n : ℝ) (h : m * (1 + Complex.I) = 1 + n * Complex.I) :
  ((m + n * Complex.I) / (m - n * Complex.I)) ^ 2 = -1 := by
  sorry

end NUMINAMATH_CALUDE_complex_square_value_l4147_414705


namespace NUMINAMATH_CALUDE_hockey_players_count_l4147_414708

/-- The number of hockey players in a games hour -/
def hockey_players (total players : ℕ) (cricket football softball : ℕ) : ℕ :=
  total - (cricket + football + softball)

/-- Theorem stating the number of hockey players -/
theorem hockey_players_count :
  hockey_players 51 10 16 13 = 12 := by
  sorry

end NUMINAMATH_CALUDE_hockey_players_count_l4147_414708


namespace NUMINAMATH_CALUDE_parabola_reflection_translation_sum_l4147_414773

/-- Original parabola function -/
def original_parabola (a b c : ℝ) (x : ℝ) : ℝ := a * x^2 + b * x + c

/-- Reflected parabola function -/
def reflected_parabola (a b c : ℝ) (x : ℝ) : ℝ := -(a * x^2 + b * x + c)

/-- Translated original parabola (3 units right) -/
def f (a b c : ℝ) (x : ℝ) : ℝ := original_parabola a b c (x - 3)

/-- Translated reflected parabola (4 units left) -/
def g (a b c : ℝ) (x : ℝ) : ℝ := reflected_parabola a b c (x + 4)

/-- Sum of translated original and reflected parabolas -/
def f_plus_g (a b c : ℝ) (x : ℝ) : ℝ := f a b c x + g a b c x

theorem parabola_reflection_translation_sum (a b c : ℝ) :
  ∀ x, f_plus_g a b c x = -14 * a * x - 19 * a - 7 * b :=
by sorry

end NUMINAMATH_CALUDE_parabola_reflection_translation_sum_l4147_414773


namespace NUMINAMATH_CALUDE_symmetric_points_sum_l4147_414755

/-- Two points are symmetric with respect to the origin if their coordinates sum to zero -/
def symmetric_wrt_origin (p q : ℝ × ℝ) : Prop :=
  p.1 + q.1 = 0 ∧ p.2 + q.2 = 0

theorem symmetric_points_sum (m n : ℝ) :
  symmetric_wrt_origin (m, 5) (3, n) → m + n = -8 := by
  sorry

end NUMINAMATH_CALUDE_symmetric_points_sum_l4147_414755


namespace NUMINAMATH_CALUDE_triangle_inequalities_and_side_relationships_l4147_414743

theorem triangle_inequalities_and_side_relationships (a b c : ℝ) 
  (h_triangle : a + b > c ∧ b + c > a ∧ c + a > b) :
  (∃ (x y z : ℝ), x^2 = a ∧ y^2 = b ∧ z^2 = c ∧ x + y > z ∧ y + z > x ∧ z + x > y) ∧
  (Real.sqrt (a * b) + Real.sqrt (b * c) + Real.sqrt (c * a) ≤ a + b + c) ∧
  (a + b + c ≤ 2 * Real.sqrt (a * b) + 2 * Real.sqrt (b * c) + 2 * Real.sqrt (c * a)) :=
by sorry

end NUMINAMATH_CALUDE_triangle_inequalities_and_side_relationships_l4147_414743


namespace NUMINAMATH_CALUDE_triangle_third_side_l4147_414794

theorem triangle_third_side (a b c : ℕ) : 
  a = 3 → b = 6 → c % 2 = 1 → 
  (a + b > c ∧ b + c > a ∧ c + a > b) →
  (c > b - a ∧ c < b + a) →
  c = 5 ∨ c = 7 := by
sorry

end NUMINAMATH_CALUDE_triangle_third_side_l4147_414794


namespace NUMINAMATH_CALUDE_ceiling_floor_sum_l4147_414736

theorem ceiling_floor_sum : ⌈(7:ℚ)/3⌉ + ⌊-(7:ℚ)/3⌋ = 0 := by sorry

end NUMINAMATH_CALUDE_ceiling_floor_sum_l4147_414736


namespace NUMINAMATH_CALUDE_square_difference_l4147_414766

theorem square_difference (x y z : ℝ) 
  (sum_xy : x + y = 10)
  (diff_xy : x - y = 8)
  (sum_yz : y + z = 15) :
  x^2 - z^2 = -115 := by
sorry

end NUMINAMATH_CALUDE_square_difference_l4147_414766


namespace NUMINAMATH_CALUDE_chapters_per_book_l4147_414711

theorem chapters_per_book (total_books : ℕ) (total_chapters : ℕ) (h1 : total_books = 4) (h2 : total_chapters = 68) :
  total_chapters / total_books = 17 := by
  sorry

end NUMINAMATH_CALUDE_chapters_per_book_l4147_414711


namespace NUMINAMATH_CALUDE_factorization_difference_of_squares_l4147_414797

theorem factorization_difference_of_squares (m x y : ℝ) : m * x^2 - m * y^2 = m * (x + y) * (x - y) := by
  sorry

end NUMINAMATH_CALUDE_factorization_difference_of_squares_l4147_414797


namespace NUMINAMATH_CALUDE_distance_difference_after_six_hours_l4147_414789

/-- Represents a cyclist with a given travel rate in miles per hour. -/
structure Cyclist where
  name : String
  rate : ℝ

/-- Calculates the distance traveled by a cyclist in a given time. -/
def distance_traveled (c : Cyclist) (time : ℝ) : ℝ :=
  c.rate * time

/-- The time period in hours for which we calculate the travel distance. -/
def travel_time : ℝ := 6

/-- Carmen, a cyclist with a travel rate of 15 miles per hour. -/
def carmen : Cyclist :=
  { name := "Carmen", rate := 15 }

/-- Daniel, a cyclist with a travel rate of 12.5 miles per hour. -/
def daniel : Cyclist :=
  { name := "Daniel", rate := 12.5 }

/-- Theorem stating the difference in distance traveled between Carmen and Daniel after 6 hours. -/
theorem distance_difference_after_six_hours :
    distance_traveled carmen travel_time - distance_traveled daniel travel_time = 15 := by
  sorry

end NUMINAMATH_CALUDE_distance_difference_after_six_hours_l4147_414789


namespace NUMINAMATH_CALUDE_expected_small_supermarkets_l4147_414779

/-- Represents the types of supermarkets --/
inductive SupermarketType
| Small
| Medium
| Large

/-- Represents the count of each type of supermarket --/
structure SupermarketCounts where
  small : ℕ
  medium : ℕ
  large : ℕ

/-- Represents the sample size for each type of supermarket --/
structure SampleSizes where
  small : ℕ
  medium : ℕ
  large : ℕ

/-- Calculates the expected number of small supermarkets in a subsample --/
def expectedSmallInSubsample (counts : SupermarketCounts) (sample : SampleSizes) (subsampleSize : ℕ) : ℚ :=
  (sample.small : ℚ) / ((sample.small + sample.medium + sample.large) : ℚ) * subsampleSize

/-- Theorem stating the expected number of small supermarkets in the subsample --/
theorem expected_small_supermarkets 
  (counts : SupermarketCounts)
  (sample : SampleSizes)
  (h1 : counts.small = 72 ∧ counts.medium = 24 ∧ counts.large = 12)
  (h2 : sample.small + sample.medium + sample.large = 9)
  (h3 : sample.small = 6 ∧ sample.medium = 2 ∧ sample.large = 1)
  : expectedSmallInSubsample counts sample 3 = 2 := by
  sorry

end NUMINAMATH_CALUDE_expected_small_supermarkets_l4147_414779


namespace NUMINAMATH_CALUDE_jake_weight_loss_l4147_414752

/-- Given Jake and his sister's combined weight and Jake's current weight,
    calculate how many pounds Jake needs to lose to weigh twice as much as his sister. -/
theorem jake_weight_loss (total_weight sister_weight jake_weight : ℕ) 
    (h1 : total_weight = 278)
    (h2 : jake_weight = 188)
    (h3 : total_weight = jake_weight + sister_weight) :
  jake_weight - 2 * sister_weight = 8 := by
  sorry

end NUMINAMATH_CALUDE_jake_weight_loss_l4147_414752


namespace NUMINAMATH_CALUDE_power_sum_equals_product_l4147_414786

theorem power_sum_equals_product (m n : ℕ+) (a b : ℝ) 
  (h1 : 3^(m.val) = a) (h2 : 3^(n.val) = b) : 
  3^(m.val + n.val) = a * b := by sorry

end NUMINAMATH_CALUDE_power_sum_equals_product_l4147_414786


namespace NUMINAMATH_CALUDE_johns_bill_total_l4147_414706

/-- Calculates the total amount due on a bill after applying late charges and annual interest. -/
def totalAmountDue (originalBill : ℝ) (lateChargeRate : ℝ) (numLateCharges : ℕ) (annualInterestRate : ℝ) : ℝ :=
  let afterLateCharges := originalBill * (1 + lateChargeRate) ^ numLateCharges
  afterLateCharges * (1 + annualInterestRate)

/-- Proves that the total amount due on John's bill is $557.13 after one year. -/
theorem johns_bill_total : 
  let originalBill : ℝ := 500
  let lateChargeRate : ℝ := 0.02
  let numLateCharges : ℕ := 3
  let annualInterestRate : ℝ := 0.05
  totalAmountDue originalBill lateChargeRate numLateCharges annualInterestRate = 557.13 := by
  sorry


end NUMINAMATH_CALUDE_johns_bill_total_l4147_414706


namespace NUMINAMATH_CALUDE_even_plus_abs_odd_is_even_l4147_414719

def IsEven (f : ℝ → ℝ) : Prop := ∀ x, f x = f (-x)
def IsOdd (f : ℝ → ℝ) : Prop := ∀ x, f x = -f (-x)

theorem even_plus_abs_odd_is_even
  (f g : ℝ → ℝ) (hf : IsEven f) (hg : IsOdd g) :
  IsEven (fun x ↦ f x + |g x|) := by
  sorry

end NUMINAMATH_CALUDE_even_plus_abs_odd_is_even_l4147_414719


namespace NUMINAMATH_CALUDE_quadratic_function_and_area_bisection_l4147_414738

/-- A quadratic function with specific properties -/
structure QuadraticFunction where
  f : ℝ → ℝ
  is_quadratic : ∃ a b c : ℝ, ∀ x, f x = a * x^2 + b * x + c
  equal_roots : ∃ r : ℝ, (∀ x, f x = 0 ↔ x = r)
  derivative : ∀ x, HasDerivAt f (2 * x + 2) x

/-- The main theorem about the quadratic function and area bisection -/
theorem quadratic_function_and_area_bisection (qf : QuadraticFunction) :
  (∀ x, qf.f x = x^2 + 2*x + 1) ∧
  ∃ t : ℝ, 0 < t ∧ t < 1 ∧
    (∫ x in (-1)..(-t), qf.f x) = (∫ x in (-t)..0, qf.f x) ∧
    t = 1 - 1 / Real.sqrt 2 := by sorry

end NUMINAMATH_CALUDE_quadratic_function_and_area_bisection_l4147_414738


namespace NUMINAMATH_CALUDE_rectangular_field_fencing_l4147_414715

theorem rectangular_field_fencing (area : ℝ) (uncovered_side : ℝ) : 
  area = 210 → uncovered_side = 20 → 
  2 * (area / uncovered_side) + uncovered_side = 41 := by
  sorry

end NUMINAMATH_CALUDE_rectangular_field_fencing_l4147_414715


namespace NUMINAMATH_CALUDE_train_length_l4147_414787

/-- The length of a train given its speed and time to cross a fixed point -/
theorem train_length (speed_kmh : ℝ) (time_s : ℝ) (length_m : ℝ) : 
  speed_kmh = 144 →
  time_s = 8.7493 →
  length_m = speed_kmh * (1000 / 3600) * time_s →
  length_m = 350 := by
sorry

end NUMINAMATH_CALUDE_train_length_l4147_414787


namespace NUMINAMATH_CALUDE_cone_height_from_semicircle_l4147_414790

/-- The distance from the highest point of a tipped-over cone to the table,
    where the cone is formed by rolling a semicircular paper. -/
theorem cone_height_from_semicircle (R : ℝ) (h : R = 4) : 
  let r := R / 2
  let h := Real.sqrt (R^2 - r^2)
  2 * (h * r / R) = 2 * Real.sqrt 3 :=
by sorry

end NUMINAMATH_CALUDE_cone_height_from_semicircle_l4147_414790


namespace NUMINAMATH_CALUDE_fraction_addition_l4147_414735

theorem fraction_addition : (1 : ℚ) / 210 + 17 / 35 = 103 / 210 := by
  sorry

end NUMINAMATH_CALUDE_fraction_addition_l4147_414735


namespace NUMINAMATH_CALUDE_smallest_integer_in_special_average_l4147_414734

def is_two_digit (n : ℕ) : Prop := 10 ≤ n ∧ n ≤ 99
def is_three_digit (n : ℕ) : Prop := 100 ≤ n ∧ n ≤ 999

theorem smallest_integer_in_special_average (m n : ℕ) 
  (h1 : is_two_digit m) 
  (h2 : is_three_digit n) 
  (h3 : (m + n) / 2 = m + n / 1000) : 
  min m n = 1 :=
sorry

end NUMINAMATH_CALUDE_smallest_integer_in_special_average_l4147_414734


namespace NUMINAMATH_CALUDE_fraction_equality_l4147_414758

theorem fraction_equality (P Q : ℤ) (x : ℝ) 
  (h : x ≠ -3 ∧ x ≠ 0 ∧ x ≠ 5) :
  (P / (x + 3 : ℝ)) + (Q / ((x^2 : ℝ) - 5*x)) = 
    ((x^2 : ℝ) - 3*x + 12) / (x^3 + x^2 - 15*x) →
  (Q : ℚ) / P = 20 / 9 := by
  sorry

end NUMINAMATH_CALUDE_fraction_equality_l4147_414758


namespace NUMINAMATH_CALUDE_sqrt_inequality_equivalence_l4147_414718

theorem sqrt_inequality_equivalence : 
  (Real.sqrt 2 - Real.sqrt 3 < Real.sqrt 6 - Real.sqrt 7) ↔ 
  ((Real.sqrt 2 + Real.sqrt 7)^2 < (Real.sqrt 3 + Real.sqrt 6)^2) := by
sorry

end NUMINAMATH_CALUDE_sqrt_inequality_equivalence_l4147_414718


namespace NUMINAMATH_CALUDE_tony_total_payment_l4147_414725

def lego_price : ℕ := 250
def sword_price : ℕ := 120
def dough_price : ℕ := 35

def lego_quantity : ℕ := 3
def sword_quantity : ℕ := 7
def dough_quantity : ℕ := 10

def total_cost : ℕ := lego_price * lego_quantity + sword_price * sword_quantity + dough_price * dough_quantity

theorem tony_total_payment : total_cost = 1940 := by
  sorry

end NUMINAMATH_CALUDE_tony_total_payment_l4147_414725


namespace NUMINAMATH_CALUDE_jamie_yellow_balls_l4147_414741

/-- Proves that Jamie bought 32 yellow balls given the initial conditions and final total -/
theorem jamie_yellow_balls :
  let initial_red : ℕ := 16
  let initial_blue : ℕ := 2 * initial_red
  let lost_red : ℕ := 6
  let final_total : ℕ := 74
  let remaining_red : ℕ := initial_red - lost_red
  let yellow_balls : ℕ := final_total - (remaining_red + initial_blue)
  yellow_balls = 32 := by
  sorry

end NUMINAMATH_CALUDE_jamie_yellow_balls_l4147_414741


namespace NUMINAMATH_CALUDE_archie_marbles_l4147_414781

theorem archie_marbles (initial : ℕ) : 
  (initial : ℝ) * (1 - 0.6) * 0.5 = 20 → initial = 100 := by
  sorry

end NUMINAMATH_CALUDE_archie_marbles_l4147_414781


namespace NUMINAMATH_CALUDE_intersection_line_l4147_414724

/-- Definition of the first circle -/
def circle1 (x y : ℝ) : Prop :=
  (x + 5)^2 + (y - 3)^2 = 100

/-- Definition of the second circle -/
def circle2 (x y : ℝ) : Prop :=
  (x - 4)^2 + (y + 6)^2 = 121

/-- Theorem stating that the line passing through the intersection points of the two circles
    has the equation x - y = -17/9 -/
theorem intersection_line : ∃ (x y : ℝ), 
  circle1 x y ∧ circle2 x y ∧ (x - y = -17/9) := by
  sorry

end NUMINAMATH_CALUDE_intersection_line_l4147_414724


namespace NUMINAMATH_CALUDE_imaginary_unit_power_2013_l4147_414793

theorem imaginary_unit_power_2013 (i : ℂ) (h : i^2 = -1) : i^2013 = i := by
  sorry

end NUMINAMATH_CALUDE_imaginary_unit_power_2013_l4147_414793


namespace NUMINAMATH_CALUDE_cubic_root_nature_l4147_414742

theorem cubic_root_nature :
  ∃ (p n1 n2 : ℝ), p > 0 ∧ n1 < 0 ∧ n2 < 0 ∧
  p^3 + 3*p^2 - 4*p - 12 = 0 ∧
  n1^3 + 3*n1^2 - 4*n1 - 12 = 0 ∧
  n2^3 + 3*n2^2 - 4*n2 - 12 = 0 ∧
  ∀ x : ℝ, x^3 + 3*x^2 - 4*x - 12 = 0 → x = p ∨ x = n1 ∨ x = n2 :=
by sorry

end NUMINAMATH_CALUDE_cubic_root_nature_l4147_414742


namespace NUMINAMATH_CALUDE_eldest_age_difference_l4147_414727

/-- Represents the ages of three grandchildren -/
structure GrandchildrenAges where
  youngest : ℕ
  middle : ℕ
  eldest : ℕ

/-- Checks if the given ages satisfy the problem conditions -/
def satisfiesConditions (ages : GrandchildrenAges) : Prop :=
  ages.middle = ages.youngest + 3 ∧
  ages.eldest = 3 * ages.youngest ∧
  ages.eldest = 15

theorem eldest_age_difference (ages : GrandchildrenAges) :
  satisfiesConditions ages →
  ages.eldest = ages.youngest + ages.middle + 2 := by
  sorry

end NUMINAMATH_CALUDE_eldest_age_difference_l4147_414727


namespace NUMINAMATH_CALUDE_ascending_order_abc_l4147_414768

theorem ascending_order_abc : 
  let a := Real.sin (5 * Real.pi / 7)
  let b := Real.cos (2 * Real.pi / 7)
  let c := Real.tan (2 * Real.pi / 7)
  b < a ∧ a < c := by sorry

end NUMINAMATH_CALUDE_ascending_order_abc_l4147_414768


namespace NUMINAMATH_CALUDE_difference_of_squares_factorization_l4147_414746

theorem difference_of_squares_factorization (y : ℝ) : 
  100 - 16 * y^2 = 4 * (5 - 2*y) * (5 + 2*y) := by
  sorry

end NUMINAMATH_CALUDE_difference_of_squares_factorization_l4147_414746


namespace NUMINAMATH_CALUDE_minimum_days_to_plant_trees_l4147_414784

def tree_sequence (n : ℕ) : ℕ := 2^(n + 1) - 2

theorem minimum_days_to_plant_trees :
  ∃ (n : ℕ), n > 0 ∧ tree_sequence n ≥ 100 ∧ ∀ m : ℕ, m > 0 → m < n → tree_sequence m < 100 :=
by sorry

end NUMINAMATH_CALUDE_minimum_days_to_plant_trees_l4147_414784


namespace NUMINAMATH_CALUDE_tangent_line_equation_l4147_414748

/-- The curve function -/
def f (x : ℝ) : ℝ := x^3 - 2*x^2 - 4*x + 2

/-- The derivative of the curve function -/
def f' (x : ℝ) : ℝ := 3*x^2 - 4*x - 4

theorem tangent_line_equation :
  let x₀ : ℝ := 1
  let y₀ : ℝ := -3
  let k : ℝ := f' x₀
  (f x₀ = y₀) ∧ 
  (∀ x y : ℝ, y = k * (x - x₀) + y₀ ↔ 5*x + y - 2 = 0) :=
sorry

end NUMINAMATH_CALUDE_tangent_line_equation_l4147_414748


namespace NUMINAMATH_CALUDE_ellipse_standard_equation_l4147_414729

/-- The standard equation of an ellipse passing through two specific points -/
theorem ellipse_standard_equation :
  ∃ (m n : ℝ), m > 0 ∧ n > 0 ∧ m ≠ n ∧
  m * (1/3)^2 + n * (1/3)^2 = 1 ∧
  n * (-1/2)^2 = 1 →
  ∀ (x y : ℝ), x^2 / (1/5) + y^2 / (1/4) = 1 ↔ m * x^2 + n * y^2 = 1 :=
by sorry

end NUMINAMATH_CALUDE_ellipse_standard_equation_l4147_414729


namespace NUMINAMATH_CALUDE_algebraic_identities_l4147_414731

theorem algebraic_identities :
  -- Part 1
  (3 * Real.sqrt 2 - Real.sqrt 12) * (Real.sqrt 18 + 2 * Real.sqrt 3) = 6 ∧
  -- Part 2
  (Real.sqrt 5 - Real.sqrt 6)^2 - (Real.sqrt 5 + Real.sqrt 6)^2 = -4 * Real.sqrt 30 ∧
  -- Part 3
  (2 * Real.sqrt (3/2) - Real.sqrt (1/2)) * (1/2 * Real.sqrt 8 + Real.sqrt (2/3)) = 5/3 * Real.sqrt 3 + 1 :=
by sorry

end NUMINAMATH_CALUDE_algebraic_identities_l4147_414731


namespace NUMINAMATH_CALUDE_proportion_solution_l4147_414745

theorem proportion_solution :
  ∀ x : ℝ, (0.75 / x = 5 / 6) → x = 0.9 := by
  sorry

end NUMINAMATH_CALUDE_proportion_solution_l4147_414745


namespace NUMINAMATH_CALUDE_solve_equation_l4147_414733

theorem solve_equation (y x : ℝ) (h1 : 9^y = x^12) (h2 : y = 6) : x = 3 := by
  sorry

end NUMINAMATH_CALUDE_solve_equation_l4147_414733


namespace NUMINAMATH_CALUDE_percent_of_number_l4147_414792

theorem percent_of_number (percent : ℝ) (number : ℝ) (result : ℝ) :
  percent = 37.5 ∧ number = 725 ∧ result = 271.875 →
  (percent / 100) * number = result :=
by
  sorry

end NUMINAMATH_CALUDE_percent_of_number_l4147_414792


namespace NUMINAMATH_CALUDE_equation_solution_l4147_414713

theorem equation_solution :
  let f (x : ℝ) := (x^2 - 11*x + 24)/(x-3) + (4*x^2 + 20*x - 32)/(2*x - 4)
  ∃ x₁ x₂ : ℝ, 
    x₁ = (-15 - Real.sqrt 417) / 4 ∧
    x₂ = (-15 + Real.sqrt 417) / 4 ∧
    f x₁ = 0 ∧ f x₂ = 0 ∧
    ∀ x : ℝ, f x = 0 → x = x₁ ∨ x = x₂ :=
by sorry

end NUMINAMATH_CALUDE_equation_solution_l4147_414713


namespace NUMINAMATH_CALUDE_hyperbola_asymptotes_l4147_414774

/-- Given a hyperbola with equation x²/a² - y²/b² = 1 where a > 0 and b > 0,
    and eccentricity √3, its asymptotes have the equation y = ±√2x -/
theorem hyperbola_asymptotes (a b : ℝ) (ha : a > 0) (hb : b > 0) :
  let e := Real.sqrt 3
  let c := e * a
  (c^2 = a^2 + b^2) →
  (∀ x y : ℝ, x^2/a^2 - y^2/b^2 = 1 → y = Real.sqrt 2 * x ∨ y = -Real.sqrt 2 * x) :=
by sorry

end NUMINAMATH_CALUDE_hyperbola_asymptotes_l4147_414774
